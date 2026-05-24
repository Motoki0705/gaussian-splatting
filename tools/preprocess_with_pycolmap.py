from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pycolmap


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def copy_input_images(source_images: Path, input_dir: Path) -> list[str]:
    input_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(p for p in source_images.iterdir() if p.suffix in IMAGE_SUFFIXES)
    if not image_paths:
        raise RuntimeError(f"No images found in {source_images}")

    image_names = []
    for src in image_paths:
        dst = input_dir / src.name
        shutil.copy2(src, dst)
        image_names.append(src.name)
    return image_names


def normalize_sparse_dir(dataset_dir: Path) -> None:
    sparse_dir = dataset_dir / "sparse"
    sparse_zero = sparse_dir / "0"
    sparse_zero.mkdir(parents=True, exist_ok=True)

    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        src = sparse_dir / name
        if src.exists():
            shutil.move(str(src), sparse_zero / name)


def best_reconstruction(reconstructions: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction:
    if not reconstructions:
        raise RuntimeError("COLMAP mapping did not produce a reconstruction")
    return max(
        reconstructions.values(),
        key=lambda reconstruction: (reconstruction.num_reg_images(), reconstruction.num_points3D()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-images", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--max-image-size", default=1600, type=int)
    parser.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive")
    parser.add_argument("--sequential-overlap", default=10, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"{args.dataset_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.dataset_dir)

    input_dir = args.dataset_dir / "input"
    distorted_dir = args.dataset_dir / "distorted"
    distorted_sparse_dir = distorted_dir / "sparse"
    database_path = distorted_dir / "database.db"

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    distorted_sparse_dir.mkdir(parents=True, exist_ok=True)

    image_names = copy_input_images(args.source_images, input_dir)
    print(f"Copied {len(image_names)} input images")

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = args.camera_model

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.max_image_size = args.max_image_size
    extraction_options.use_gpu = False

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = False

    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.ba_global_function_tolerance = 1e-6
    mapper_options.random_seed = 0
    mapper_options.mapper.random_seed = 0

    print("Extracting features")
    pycolmap.extract_features(
        database_path,
        input_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )

    print(f"Matching features with {args.matcher} matcher")
    if args.matcher == "sequential":
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = args.sequential_overlap
        pycolmap.match_sequential(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
        )
    else:
        pycolmap.match_exhaustive(database_path, matching_options=matching_options)

    print("Running incremental mapping")
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        input_dir,
        distorted_sparse_dir,
        options=mapper_options,
    )
    reconstruction = best_reconstruction(reconstructions)
    model_dir = distorted_sparse_dir / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write_binary(model_dir)
    print(
        f"Selected reconstruction with {reconstruction.num_reg_images()} images "
        f"and {reconstruction.num_points3D()} points"
    )

    undistort_options = pycolmap.UndistortCameraOptions()
    undistort_options.max_image_size = args.max_image_size

    print("Undistorting images")
    pycolmap.undistort_images(
        args.dataset_dir,
        model_dir,
        input_dir,
        output_type="COLMAP",
        undistort_options=undistort_options,
    )
    normalize_sparse_dir(args.dataset_dir)
    print("Done")


if __name__ == "__main__":
    main()
