from __future__ import annotations

import argparse
import importlib
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def import_required(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing optional dependency '{module_name}'.\n{install_hint}"
        ) from exc


def list_images(source_images: Path) -> list[Path]:
    image_paths = sorted(p for p in source_images.iterdir() if p.suffix in IMAGE_SUFFIXES)
    if not image_paths:
        raise RuntimeError(f"No images found in {source_images}")
    return image_paths


def copy_images(image_paths: Iterable[Path], dataset_dir: Path) -> list[str]:
    image_names: list[str] = []
    for dirname in ("input", "images"):
        (dataset_dir / dirname).mkdir(parents=True, exist_ok=True)

    for src in image_paths:
        for dirname in ("input", "images"):
            shutil.copy2(src, dataset_dir / dirname / src.name)
        image_names.append(src.name)
    return image_names


def ensure_clean_dataset_dir(dataset_dir: Path, overwrite: bool) -> None:
    if dataset_dir.exists():
        if not overwrite:
            raise RuntimeError(f"{dataset_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(dataset_dir)
    (dataset_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = r.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def first_existing_scene_value(scene, names: tuple[str, ...]):
    for name in names:
        if hasattr(scene, name):
            value = getattr(scene, name)
            return value() if callable(value) else value
    raise AttributeError(f"MASt3R scene does not expose any of: {', '.join(names)}")


def get_intrinsics(scene) -> list[np.ndarray]:
    intrinsics = first_existing_scene_value(scene, ("get_intrinsics", "get_focals"))
    intrinsics = to_numpy(intrinsics)

    if intrinsics.ndim == 1:
        raise RuntimeError(
            "MASt3R returned focal lengths only. This script needs 3x3 intrinsics; "
            "try a newer MASt3R checkout or export COLMAP from MASt3R directly."
        )
    return [np.asarray(k, dtype=np.float64) for k in intrinsics]


def get_camera_to_worlds(scene) -> list[np.ndarray]:
    poses = first_existing_scene_value(scene, ("get_im_poses", "get_c2ws", "get_camera_poses"))
    poses = to_numpy(poses)
    if poses.shape[-2:] == (4, 4):
        return [np.asarray(p, dtype=np.float64) for p in poses]
    if poses.shape[-2:] == (3, 4):
        c2ws = []
        for pose in poses:
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :4] = pose
            c2ws.append(c2w)
        return c2ws
    raise RuntimeError(f"Unsupported MASt3R camera pose shape: {poses.shape}")


def get_points_and_masks(scene) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    pts3d = first_existing_scene_value(scene, ("get_pts3d", "get_pts3d_raw"))
    pts3d = [to_numpy(p).astype(np.float32) for p in pts3d]

    masks = None
    for name in ("get_masks", "get_valid_masks"):
        if hasattr(scene, name):
            masks = [to_numpy(m).astype(bool) for m in getattr(scene, name)()]
            break
    if masks is None:
        for name in ("get_conf", "get_confidences"):
            if hasattr(scene, name):
                confs = [to_numpy(c) for c in getattr(scene, name)()]
                masks = [np.isfinite(c) & (c > 0) for c in confs]
                break
    return pts3d, masks


def sample_point_cloud(
    pts3d: list[np.ndarray],
    masks: list[np.ndarray] | None,
    image_paths: list[Path],
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []

    for index, points in enumerate(pts3d):
        if points.ndim != 3 or points.shape[-1] != 3:
            continue
        valid = np.isfinite(points).all(axis=-1)
        if masks is not None:
            valid &= masks[index]
        xyz = points[valid]
        if xyz.size == 0:
            continue

        image = np.asarray(Image.open(image_paths[index]).convert("RGB"))
        if image.shape[:2] != points.shape[:2]:
            # MASt3R may resize internally. Nearest-neighbour sampling is enough for initial colors.
            pil = Image.fromarray(image)
            pil = pil.resize((points.shape[1], points.shape[0]), resample=Image.Resampling.BILINEAR)
            image = np.asarray(pil)
        rgb = image[valid]
        xyz_chunks.append(xyz)
        rgb_chunks.append(rgb)

    if not xyz_chunks:
        raise RuntimeError("MASt3R produced no valid 3D points")

    xyz_all = np.concatenate(xyz_chunks, axis=0)
    rgb_all = np.concatenate(rgb_chunks, axis=0)

    if max_points > 0 and len(xyz_all) > max_points:
        rng = np.random.default_rng(0)
        selected = rng.choice(len(xyz_all), size=max_points, replace=False)
        xyz_all = xyz_all[selected]
        rgb_all = rgb_all[selected]

    return xyz_all.astype(np.float32), rgb_all.astype(np.uint8)


def write_cameras_txt(path: Path, image_paths: list[Path], intrinsics: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(image_paths)}\n")
        for camera_id, (image_path, k) in enumerate(zip(image_paths, intrinsics), start=1):
            with Image.open(image_path) as image:
                width, height = image.size
            fx = float(k[0, 0])
            fy = float(k[1, 1])
            cx = float(k[0, 2])
            cy = float(k[1, 2])
            f.write(f"{camera_id} PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n")


def write_images_txt(path: Path, image_names: list[str], c2ws: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_names)}\n")
        for image_id, (image_name, c2w) in enumerate(zip(image_names, c2ws), start=1):
            w2c = np.linalg.inv(c2w)
            qvec = rotmat_to_qvec(w2c[:3, :3])
            tvec = w2c[:3, 3]
            f.write(
                f"{image_id} "
                f"{qvec[0]:.12g} {qvec[1]:.12g} {qvec[2]:.12g} {qvec[3]:.12g} "
                f"{tvec[0]:.12g} {tvec[1]:.12g} {tvec[2]:.12g} "
                f"{image_id} {image_name}\n"
            )
            # 3DGS only needs registered camera poses and the initial sparse cloud.
            # Empty tracks keep the COLMAP text format valid while avoiding huge files.
            f.write("\n")


def write_points3d_txt(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(xyz)}\n")
        for point_id, (point, color) in enumerate(zip(xyz, rgb), start=1):
            r, g, b = [int(c) for c in color]
            f.write(
                f"{point_id} "
                f"{float(point[0]):.12g} {float(point[1]):.12g} {float(point[2]):.12g} "
                f"{r} {g} {b} 0\n"
            )


def write_points3d_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(len(xyz), dtype=dtype)
    attributes = np.concatenate([xyz.astype(np.float32), normals, rgb.astype(np.uint8)], axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def export_colmap_text(dataset_dir: Path, image_paths: list[Path], image_names: list[str], scene, max_points: int) -> None:
    sparse_dir = dataset_dir / "sparse" / "0"
    intrinsics = get_intrinsics(scene)
    c2ws = get_camera_to_worlds(scene)
    if len(intrinsics) != len(image_paths) or len(c2ws) != len(image_paths):
        raise RuntimeError(
            f"MASt3R returned {len(intrinsics)} intrinsics and {len(c2ws)} poses "
            f"for {len(image_paths)} images"
        )

    pts3d, masks = get_points_and_masks(scene)
    xyz, rgb = sample_point_cloud(pts3d, masks, image_paths, max_points=max_points)

    write_cameras_txt(sparse_dir / "cameras.txt", image_paths, intrinsics)
    write_images_txt(sparse_dir / "images.txt", image_names, c2ws)
    write_points3d_txt(sparse_dir / "points3D.txt", xyz, rgb)
    write_points3d_ply(sparse_dir / "points3D.ply", xyz, rgb)

    print(f"Wrote MASt3R COLMAP text model to {sparse_dir}")
    print(f"Registered images: {len(image_paths)}")
    print(f"Initial points: {len(xyz)}")


def run_mast3r(args, image_paths: list[Path]):
    torch = import_required("torch", "Install PyTorch first, then install MASt3R and DUSt3R.")
    mast3r_model = import_required(
        "mast3r.model",
        "Install MASt3R, for example: uv pip install --python .venv/bin/python -e /path/to/mast3r",
    )
    mast3r_sparse_ga = import_required("mast3r.cloud_opt.sparse_ga", "Install MASt3R with its DUSt3R dependency.")
    dust3r_image = import_required("dust3r.utils.image", "Install DUSt3R dependency used by MASt3R.")
    dust3r_pairs = import_required("dust3r.image_pairs", "Install DUSt3R dependency used by MASt3R.")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")

    print(f"Loading MASt3R model: {args.model_name}")
    model = mast3r_model.AsymmetricMASt3R.from_pretrained(args.model_name).to(device)
    model.eval()

    print(f"Loading {len(image_paths)} images for MASt3R")
    images = dust3r_image.load_images([str(p) for p in image_paths], size=args.image_size, verbose=not args.quiet)
    pairs = dust3r_pairs.make_pairs(
        images,
        scene_graph=args.scene_graph,
        prefilter=None,
        symmetrize=True,
    )

    cache_dir = args.dataset_dir / "mast3r_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Running MASt3R sparse global alignment")
    return mast3r_sparse_ga.sparse_global_alignment(
        [str(p) for p in image_paths],
        pairs,
        str(cache_dir),
        model,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        device=device,
        opt_depth=args.opt_depth,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess images with MASt3R and export a 3DGS-compatible COLMAP text dataset."
    )
    parser.add_argument("--source-images", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--scene-graph", default="complete")
    parser.add_argument("--lr1", default=0.07, type=float)
    parser.add_argument("--niter1", default=500, type=int)
    parser.add_argument("--lr2", default=0.014, type=float)
    parser.add_argument("--niter2", default=200, type=int)
    parser.add_argument("--matching-conf-thr", default=5.0, type=float)
    parser.add_argument("--max-points", default=300000, type=int)
    parser.add_argument("--shared-intrinsics", action="store_true")
    parser.add_argument("--opt-depth", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    image_paths = list_images(args.source_images)
    ensure_clean_dataset_dir(args.dataset_dir, overwrite=args.overwrite)
    image_names = copy_images(image_paths, args.dataset_dir)
    scene = run_mast3r(args, image_paths)
    export_colmap_text(args.dataset_dir, image_paths, image_names, scene, max_points=args.max_points)
    print("Done")


if __name__ == "__main__":
    main()
