from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from utils.ground_alignment import (
    GroundAlignmentConfig,
    estimate_ground_alignment,
    transform_camera_rt,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer tennis court white lines and project them onto the estimated ground plane.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/meiji-court-large/3dgs-2fps-sequential"))
    parser.add_argument("--images", default="images")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/meiji-court-large-line-projection"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/kamimura/projects/tennis-lab/outputs/court_detection/line/logs/versino_0/"
            "checkpoints/court-detection-epoch=19.ckpt"
        ),
    )
    parser.add_argument(
        "--tennis-lab-root",
        type=Path,
        default=Path("submodules/tennis-lab"),
    )
    parser.add_argument(
        "--dino-swin-checkpoint",
        type=Path,
        default=Path("/home/kamimura/projects/tennis-lab/checkpoints/DINO/swin_backbone_state_checkpoint0027_5scale.pth"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--top-k-per-image", type=int, default=20000)
    parser.add_argument("--min-ray-z", type=float, default=1e-6)
    parser.add_argument("--max-output-size", type=int, default=2400)
    parser.add_argument("--pixels-per-unit", type=float, default=120.0)
    parser.add_argument("--bounds-percentile", type=float, default=1.0)
    parser.add_argument("--save-points", action="store_true")
    parser.add_argument("--ground-transform", type=Path, default=None)
    parser.add_argument("--ground-voxel-size", type=float, default=0.05)
    parser.add_argument("--ground-ransac-iters", type=int, default=1000)
    parser.add_argument("--ground-ransac-threshold", type=float, default=0.03)
    parser.add_argument("--ground-low-percentile", type=float, default=35.0)
    parser.add_argument("--ground-min-inlier-ratio", type=float, default=0.01)
    parser.add_argument("--ground-up-axis", default="z")
    return parser.parse_args()


def load_colmap_model(dataset: Path):
    sparse = dataset / "sparse" / "0"
    try:
        cameras = read_intrinsics_binary(sparse / "cameras.bin")
        images = read_extrinsics_binary(sparse / "images.bin")
        points, _, _ = read_points3D_binary(sparse / "points3D.bin")
    except Exception:
        cameras = read_intrinsics_text(sparse / "cameras.txt")
        images = read_extrinsics_text(sparse / "images.txt")
        points, _, _ = read_points3D_text(sparse / "points3D.txt")
    return cameras, images, points


def load_or_estimate_ground_transform(args: argparse.Namespace, points: np.ndarray) -> tuple[np.ndarray, dict]:
    if args.ground_transform and args.ground_transform.exists():
        with open(args.ground_transform, "r") as file:
            metadata = json.load(file)
        return np.array(metadata["world_to_ground"], dtype=np.float64), metadata

    output_transform = args.output / "ground_transform.json"
    if output_transform.exists():
        with open(output_transform, "r") as file:
            metadata = json.load(file)
        return np.array(metadata["world_to_ground"], dtype=np.float64), metadata

    config = GroundAlignmentConfig(
        voxel_size=args.ground_voxel_size,
        sor_k=0,
        ransac_iters=args.ground_ransac_iters,
        ransac_threshold=args.ground_ransac_threshold,
        low_percentile=args.ground_low_percentile,
        min_inlier_ratio=args.ground_min_inlier_ratio,
        up_axis=args.ground_up_axis,
    )
    result = estimate_ground_alignment(points, config)
    metadata = result.to_json_dict()
    metadata["config"] = asdict(config)
    return result.transform, metadata


def add_tennis_lab_to_path(tennis_lab_root: Path) -> None:
    root = tennis_lab_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_line_model(args: argparse.Namespace):
    add_tennis_lab_to_path(args.tennis_lab_root)
    from src.tasks.court_detection.training.lightning_module import CourtDetectionLightningModule

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]["config"]
    dino_checkpoint = resolve_dino_checkpoint(args)
    if dino_checkpoint is not None:
        config.model.encoder.checkpoint_path = str(dino_checkpoint)

    module = CourtDetectionLightningModule.load_from_checkpoint(
        str(args.checkpoint),
        map_location="cpu",
        weights_only=False,
        config=config,
    )
    model = module.model.to(args.device)
    model.eval()

    data_cfg = dict(module.config.get("data", {}))
    aug_cfg = data_cfg.get("augmentation", {})
    short_side = int(aug_cfg.get("val_short_side", 288))
    return model, short_side, dino_checkpoint


def resolve_dino_checkpoint(args: argparse.Namespace) -> Path | None:
    candidates = [
        args.dino_swin_checkpoint,
        args.tennis_lab_root / "checkpoints/DINO/swin_backbone_state_checkpoint0027_5scale.pth",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def resize_for_model(image: Image.Image, short_side: int) -> tuple[Image.Image, float, float]:
    orig_w, orig_h = image.size
    if orig_h <= orig_w:
        new_h = short_side
        new_w = int(round(orig_w * new_h / orig_h))
    else:
        new_w = short_side
        new_h = int(round(orig_h * new_w / orig_w))
    new_h = max((new_h // 8) * 8, 8)
    new_w = max((new_w // 8) * 8, 8)
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    return resized, orig_w / new_w, orig_h / new_h


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def camera_intrinsic_matrix(camera) -> np.ndarray:
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = float(camera.params[0])
        cx = float(camera.params[1])
        cy = float(camera.params[2])
    elif camera.model == "PINHOLE":
        fx = float(camera.params[0])
        fy = float(camera.params[1])
        cx = float(camera.params[2])
        cy = float(camera.params[3])
    else:
        raise ValueError(f"Unsupported camera model for projection: {camera.model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_mask_to_ground(
    probs: np.ndarray,
    scale_x: float,
    scale_y: float,
    camera,
    image_meta,
    world_to_ground: np.ndarray,
    threshold: float,
    top_k: int,
    min_ray_z: float,
) -> np.ndarray:
    ys, xs = np.nonzero(probs >= threshold)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    values = probs[ys, xs]
    if top_k > 0 and len(xs) > top_k:
        keep = np.argpartition(values, -top_k)[-top_k:]
        xs = xs[keep]
        ys = ys[keep]

    pixels = np.column_stack(
        [
            (xs.astype(np.float64) + 0.5) * scale_x,
            (ys.astype(np.float64) + 0.5) * scale_y,
            np.ones(len(xs), dtype=np.float64),
        ]
    )

    K_inv = np.linalg.inv(camera_intrinsic_matrix(camera))
    Rcw = qvec2rotmat(image_meta.qvec)
    camera_R = Rcw.T
    camera_T = np.array(image_meta.tvec, dtype=np.float64)
    ground_R, ground_T = transform_camera_rt(camera_R, camera_T, world_to_ground)
    Rcw_ground = ground_R.T
    camera_center = -Rcw_ground.T @ ground_T

    rays_camera = pixels @ K_inv.T
    rays_ground = rays_camera @ Rcw_ground
    valid = np.abs(rays_ground[:, 2]) > min_ray_z
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32)

    rays_ground = rays_ground[valid]
    t = -camera_center[2] / rays_ground[:, 2]
    valid_t = t > 0
    if not np.any(valid_t):
        return np.empty((0, 2), dtype=np.float32)

    points = camera_center[None, :] + rays_ground[valid_t] * t[valid_t, None]
    return points[:, :2].astype(np.float32)


def run_inference_and_projection(args: argparse.Namespace) -> tuple[list[np.ndarray], list[dict], dict]:
    cameras, images, sparse_points = load_colmap_model(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    world_to_ground, ground_metadata = load_or_estimate_ground_transform(args, sparse_points)
    with open(args.output / "ground_transform.json", "w") as file:
        json.dump(ground_metadata, file, indent=2)

    model, short_side, dino_checkpoint = load_line_model(args)
    image_dir = args.dataset / args.images
    ordered_images = sorted(images.values(), key=lambda image: image.name)
    if args.limit > 0:
        ordered_images = ordered_images[:args.limit]

    projected_points: list[np.ndarray] = []
    rows: list[dict] = []

    for start in tqdm(range(0, len(ordered_images), args.batch_size), desc="Projecting line masks"):
        batch_images = ordered_images[start:start + args.batch_size]
        tensors = []
        batch_meta = []
        for image_meta in batch_images:
            image_path = image_dir / image_meta.name
            if not image_path.exists():
                rows.append({"image": image_meta.name, "status": "missing", "points": 0})
                continue
            pil_image = Image.open(image_path).convert("RGB")
            resized, scale_x, scale_y = resize_for_model(pil_image, short_side)
            tensors.append(image_to_tensor(resized))
            batch_meta.append((image_meta, scale_x, scale_y, resized.size))

        if not tensors:
            continue

        input_tensor = torch.stack(tensors).to(args.device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

        for prob, (image_meta, scale_x, scale_y, resized_size) in zip(probs, batch_meta):
            camera = cameras[image_meta.camera_id]
            points_xy = project_mask_to_ground(
                prob,
                scale_x,
                scale_y,
                camera,
                image_meta,
                world_to_ground,
                args.threshold,
                args.top_k_per_image,
                args.min_ray_z,
            )
            if len(points_xy):
                projected_points.append(points_xy)
            rows.append(
                {
                    "image": image_meta.name,
                    "status": "ok",
                    "points": int(len(points_xy)),
                    "resized_width": int(resized_size[0]),
                    "resized_height": int(resized_size[1]),
                }
            )

    metadata = {
        "dataset": str(args.dataset),
        "images": args.images,
        "checkpoint": str(args.checkpoint),
        "dino_swin_checkpoint": str(dino_checkpoint) if dino_checkpoint else None,
        "short_side": short_side,
        "threshold": args.threshold,
        "top_k_per_image": args.top_k_per_image,
        "processed_images": sum(1 for row in rows if row["status"] == "ok"),
        "missing_images": sum(1 for row in rows if row["status"] == "missing"),
        "requested_limit": int(args.limit),
        "projected_points": int(sum(int(row["points"]) for row in rows)),
        "world_to_ground": world_to_ground.tolist(),
    }
    return projected_points, rows, metadata


def rasterize_points(points_xy: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    lower_pct = args.bounds_percentile
    upper_pct = 100.0 - args.bounds_percentile
    min_xy = np.percentile(points_xy, lower_pct, axis=0)
    max_xy = np.percentile(points_xy, upper_pct, axis=0)
    extent = np.maximum(max_xy - min_xy, 1e-6)

    pixels_per_unit = args.pixels_per_unit
    width = int(np.ceil(extent[0] * pixels_per_unit)) + 1
    height = int(np.ceil(extent[1] * pixels_per_unit)) + 1
    scale = 1.0
    if max(width, height) > args.max_output_size:
        scale = args.max_output_size / max(width, height)
        pixels_per_unit *= scale
        width = int(np.ceil(extent[0] * pixels_per_unit)) + 1
        height = int(np.ceil(extent[1] * pixels_per_unit)) + 1

    cols = np.floor((points_xy[:, 0] - min_xy[0]) * pixels_per_unit).astype(np.int64)
    rows = np.floor((max_xy[1] - points_xy[:, 1]) * pixels_per_unit).astype(np.int64)
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)

    accumulator = np.zeros((height, width), dtype=np.float32)
    np.add.at(accumulator, (rows[valid], cols[valid]), 1.0)
    bounds = {
        "min_xy": min_xy.tolist(),
        "max_xy": max_xy.tolist(),
        "pixels_per_unit": float(pixels_per_unit),
        "width": int(width),
        "height": int(height),
        "kept_points": int(valid.sum()),
        "bounds_percentile": float(args.bounds_percentile),
    }
    return accumulator, bounds


def save_outputs(
    projected_points: list[np.ndarray],
    rows: list[dict],
    metadata: dict,
    args: argparse.Namespace,
) -> None:
    if not projected_points:
        raise RuntimeError("No line pixels were projected onto the ground plane.")

    points_xy = np.concatenate(projected_points, axis=0)
    accumulator, bounds = rasterize_points(points_xy, args)
    metadata["raster"] = bounds

    positive = accumulator[accumulator > 0]
    vmax = float(np.percentile(positive, 99.5)) if len(positive) else 1.0
    norm = np.clip(accumulator / max(vmax, 1.0), 0.0, 1.0)
    grayscale = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
    binary = (accumulator > 0).astype(np.uint8) * 255

    cv2.imwrite(str(args.output / "projected_lines_accum.png"), grayscale)
    cv2.imwrite(str(args.output / "projected_lines_heatmap.png"), heatmap)
    cv2.imwrite(str(args.output / "projected_lines_binary.png"), binary)

    with open(args.output / "projection_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    with open(args.output / "per_image_projection_counts.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "status", "points", "resized_width", "resized_height"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.save_points:
        np.save(args.output / "projected_line_points_xy.npy", points_xy)


def main() -> None:
    args = parse_args()
    projected_points, rows, metadata = run_inference_and_projection(args)
    save_outputs(projected_points, rows, metadata, args)
    print(f"Saved projected line images to {args.output}")


if __name__ == "__main__":
    main()
