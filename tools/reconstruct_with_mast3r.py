#!/usr/bin/env python3
"""Reconstruct a 3DGS-ready COLMAP-style dataset with MASt3R SparseGA."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import rotmat2qvec  # noqa: E402
from scene.dataset_readers import storePly  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MASt3R Sparse Global Alignment and export COLMAP text files for 3DGS."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/meiji-court-large/video-frames"),
        help="Input image folder. jpg/jpeg/png are supported.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/meiji-court-large/3dgs-2fps-mast3r"),
        help="Output 3DGS dataset folder.",
    )
    parser.add_argument(
        "--mast3r-root",
        type=Path,
        default=Path("submodules/mast3r-slam/thirdparty/mast3r"),
        help="Path to the MASt3R source tree.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("downloads/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"),
        help="MASt3R model checkpoint.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--scenegraph", default="swin", choices=["swin", "logwin", "oneref", "complete"])
    parser.add_argument("--winsize", type=int, default=5)
    parser.add_argument("--refid", type=int, default=0)
    parser.add_argument("--cyclic", action="store_true", help="Use cyclic sliding windows.")
    parser.add_argument("--shared-intrinsics", action="store_true", default=True)
    parser.add_argument("--no-shared-intrinsics", dest="shared_intrinsics", action="store_false")
    parser.add_argument("--subsample", type=int, default=8, help="SparseGA correspondence subsampling.")
    parser.add_argument("--lr1", type=float, default=0.2)
    parser.add_argument("--niter1", type=int, default=300)
    parser.add_argument("--lr2", type=float, default=0.02)
    parser.add_argument("--niter2", type=int, default=200)
    parser.add_argument("--max-points", type=int, default=800_000)
    parser.add_argument("--limit", type=int, default=0, help="Debug: only process first N images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_mast3r_paths(mast3r_root: Path) -> None:
    mast3r_root = mast3r_root.resolve()
    dust3r_root = mast3r_root / "dust3r"
    croco_root = dust3r_root / "croco"
    for path in (mast3r_root, dust3r_root, croco_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def allow_trusted_torch_checkpoint_loads() -> None:
    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_default


def list_images(image_dir: Path, limit: int = 0) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in suffixes)
    if limit > 0:
        images = images[:limit]
    if len(images) < 2:
        raise ValueError(f"MASt3R reconstruction needs at least two images, got {len(images)}")
    return images


def scene_graph_name(args: argparse.Namespace) -> str:
    if args.scenegraph in {"swin", "logwin"}:
        parts = [args.scenegraph, str(args.winsize)]
        if not args.cyclic:
            parts.append("noncyclic")
        return "-".join(parts)
    if args.scenegraph == "oneref":
        return f"oneref-{args.refid}"
    return "complete"


def save_resized_images(scene, image_paths: list[Path], output_images: Path) -> list[str]:
    output_images.mkdir(parents=True, exist_ok=True)
    names = []
    for idx, src in enumerate(image_paths):
        # scene.imgs are the exact cropped/resized RGB images used by SparseGA, in [0, 1].
        arr = (np.asarray(scene.imgs[idx]).clip(0.0, 1.0) * 255.0).astype(np.uint8)
        name = src.stem + ".png"
        Image.fromarray(arr).save(output_images / name)
        names.append(name)
    return names


def collect_sparse_points(scene, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    pts_parts = []
    rgb_parts = []
    pts3d = scene.get_sparse_pts3d()
    colors = scene.get_pts3d_colors()
    depthmaps = scene.get_depthmaps()
    for pts, rgb, depth in zip(pts3d, colors, depthmaps):
        pts_np = pts.detach().cpu().numpy() if torch.is_tensor(pts) else np.asarray(pts)
        rgb_np = np.asarray(rgb)
        valid = np.isfinite(pts_np).all(axis=1)
        if depth is not None:
            depth_np = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
            depth_flat = np.asarray(depth_np).reshape(-1)
            if depth_flat.shape[0] == valid.shape[0]:
                valid &= np.isfinite(depth_flat)
        pts_parts.append(pts_np[valid])
        rgb_parts.append(rgb_np[valid])

    if not pts_parts:
        raise RuntimeError("MASt3R did not produce any sparse 3D points.")

    xyz = np.concatenate(pts_parts, axis=0).astype(np.float32)
    rgb = np.concatenate(rgb_parts, axis=0)
    if rgb.dtype != np.uint8:
        rgb = (rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8)

    finite = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[finite], rgb[finite]
    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(xyz), size=max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    return xyz, rgb


def write_colmap_text(
    sparse_dir: Path,
    image_names: list[str],
    intrinsics: np.ndarray,
    cam2world: np.ndarray,
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    with (sparse_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(image_names)}\n")
        for idx, (name, K) in enumerate(zip(image_names, intrinsics), start=1):
            with Image.open(sparse_dir.parents[1] / "images" / name) as im:
                width, height = im.size
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            f.write(f"{idx} PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n")

    with (sparse_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_names)}, mean observations per image: 0\n")
        for idx, (name, c2w) in enumerate(zip(image_names, cam2world), start=1):
            c2w = np.asarray(c2w, dtype=np.float64)
            R_w2c = c2w[:3, :3].T
            t_w2c = -R_w2c @ c2w[:3, 3]
            qvec = rotmat2qvec(R_w2c)
            pose = " ".join(f"{v:.12g}" for v in (*qvec, *t_w2c))
            f.write(f"{idx} {pose} {idx} {name}\n\n")

    with (sparse_dir / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")


def append_points3d_text(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with path.open("a", encoding="utf-8") as f:
        for idx, (p, c) in enumerate(zip(xyz, rgb), start=1):
            f.write(
                f"{idx} {p[0]:.12g} {p[1]:.12g} {p[2]:.12g} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} 0\n"
            )


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} already exists. Use --overwrite to replace it.")
    if not args.weights.exists():
        raise FileNotFoundError(f"MASt3R checkpoint not found: {args.weights}")

    add_mast3r_paths(args.mast3r_root)
    allow_trusted_torch_checkpoint_loads()
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.model import AsymmetricMASt3R

    image_paths = list_images(args.images, args.limit)
    output_images = args.output / "images"
    sparse_dir = args.output / "sparse" / "0"
    cache_dir = args.output / "mast3r_cache"
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MASt3R] loading {len(image_paths)} images")
    imgs = load_images([str(p) for p in image_paths], size=args.image_size, verbose=True)
    graph = scene_graph_name(args)
    pairs = make_pairs(imgs, scene_graph=graph, prefilter=None, symmetrize=True)
    print(f"[MASt3R] scene graph={graph}, pairs={len(pairs)}")

    model = AsymmetricMASt3R.from_pretrained(str(args.weights)).to(args.device)
    model.eval()
    scene = sparse_global_alignment(
        [str(p) for p in image_paths],
        pairs,
        str(cache_dir),
        model,
        subsample=args.subsample,
        device=args.device,
        shared_intrinsics=args.shared_intrinsics,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        verbose=True,
    )

    image_names = save_resized_images(scene, image_paths, output_images)
    intrinsics = scene.intrinsics.detach().cpu().numpy()
    cam2world = scene.get_im_poses().detach().cpu().numpy()
    xyz, rgb = collect_sparse_points(scene, args.max_points)

    write_colmap_text(sparse_dir, image_names, intrinsics, cam2world)
    append_points3d_text(sparse_dir / "points3D.txt", xyz, rgb)
    storePly(str(sparse_dir / "points3D.ply"), xyz, rgb)

    metadata = {
        "source_images": str(args.images),
        "num_images": len(image_names),
        "num_pairs": len(pairs),
        "scene_graph": graph,
        "image_size": args.image_size,
        "shared_intrinsics": args.shared_intrinsics,
        "weights": str(args.weights),
        "num_points": int(len(xyz)),
        "format": "COLMAP text + resized MASt3R input images for gaussian-splatting",
    }
    with (args.output / "mast3r_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[MASt3R] wrote 3DGS dataset to {args.output}")
    print(f"[MASt3R] images={len(image_names)} points={len(xyz)}")


if __name__ == "__main__":
    main()
