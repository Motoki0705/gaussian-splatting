#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class GroundAlignmentConfig:
    voxel_size: float = 0.05
    sor_k: int = 16
    sor_std_ratio: float = 2.0
    ransac_iters: int = 1000
    ransac_threshold: float = 0.03
    low_percentile: float = 35.0
    min_inlier_ratio: float = 0.05
    up_axis: str = "z"
    seed: int = 0
    max_ransac_points: int = 50000


@dataclass
class GroundAlignmentResult:
    transform: np.ndarray
    normal: np.ndarray
    centroid: np.ndarray
    inlier_count: int
    sample_count: int
    rms_error: float
    config: GroundAlignmentConfig

    def to_json_dict(self):
        return {
            "world_to_ground": self.transform.tolist(),
            "normal_world": self.normal.tolist(),
            "centroid_world": self.centroid.tolist(),
            "inlier_count": int(self.inlier_count),
            "sample_count": int(self.sample_count),
            "rms_error": float(self.rms_error),
            "config": asdict(self.config),
        }


def axis_vector(axis: str) -> np.ndarray:
    key = axis.lower()
    signs = {"x": 1.0, "y": 1.0, "z": 1.0, "-x": -1.0, "-y": -1.0, "-z": -1.0}
    if key not in signs:
        raise ValueError(f"Unsupported ground up axis '{axis}'. Expected x, y, z, -x, -y, or -z.")
    idx = key[-1]
    vec = np.zeros(3, dtype=np.float64)
    vec[{"x": 0, "y": 1, "z": 2}[idx]] = signs[key]
    return vec


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(points) == 0:
        return points.astype(np.float64, copy=True)

    coords = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(coords, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def statistical_outlier_removal(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    if k <= 0 or std_ratio <= 0 or len(points) <= k + 1:
        return points

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        query_k = min(k + 1, len(points))
        try:
            distances, _ = tree.query(points, k=query_k, workers=-1)
        except TypeError:
            distances, _ = tree.query(points, k=query_k)
        mean_distances = distances[:, 1:].mean(axis=1)
    except Exception:
        if len(points) > 8000:
            print("[GroundAlignment] scipy is unavailable; skipping SOR for more than 8000 points.")
            return points
        mean_distances = _mean_knn_distances(points, k)

    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    return points[mean_distances <= threshold]


def _mean_knn_distances(points: np.ndarray, k: int, chunk_size: int = 512) -> np.ndarray:
    means = np.empty(len(points), dtype=np.float64)
    query_k = min(k + 1, len(points))
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        diff = points[start:stop, None, :] - points[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        kth = np.partition(dist2, query_k - 1, axis=1)[:, :query_k]
        means[start:stop] = np.sqrt(kth[:, 1:]).mean(axis=1)
    return means


def filter_low_points(points: np.ndarray, up_hint: np.ndarray, percentile: float) -> np.ndarray:
    percentile = np.clip(percentile, 0.0, 100.0)
    heights = points @ up_hint
    cutoff = np.percentile(heights, percentile)
    low_points = points[heights <= cutoff]
    return low_points if len(low_points) >= 3 else points


def estimate_ground_alignment(points: np.ndarray, config: GroundAlignmentConfig) -> GroundAlignmentResult:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError("At least 3 finite points are required for ground alignment.")

    up_hint = axis_vector(config.up_axis)
    preprocessed = voxel_downsample(points, config.voxel_size)
    preprocessed = statistical_outlier_removal(preprocessed, config.sor_k, config.sor_std_ratio)
    candidate_points = filter_low_points(preprocessed, up_hint, config.low_percentile)
    candidate_points = _subsample(candidate_points, config.max_ransac_points, config.seed)

    if len(candidate_points) < 3:
        raise ValueError("Ground alignment has fewer than 3 candidate points after preprocessing.")

    normal, d, inliers = ransac_plane(
        candidate_points,
        threshold=config.ransac_threshold,
        iterations=config.ransac_iters,
        seed=config.seed,
    )
    inlier_count = int(inliers.sum())
    min_inliers = max(3, int(np.ceil(len(candidate_points) * config.min_inlier_ratio)))
    if inlier_count < min_inliers:
        raise ValueError(
            f"Ground plane RANSAC found only {inlier_count} inliers; "
            f"expected at least {min_inliers}."
        )

    refined_normal, centroid = refine_plane_pca(candidate_points[inliers], up_hint)
    distances = candidate_points[inliers] @ refined_normal - centroid @ refined_normal
    rms_error = float(np.sqrt(np.mean(distances * distances)))
    transform = build_world_to_ground_transform(refined_normal, centroid)
    return GroundAlignmentResult(
        transform=transform,
        normal=refined_normal,
        centroid=centroid,
        inlier_count=inlier_count,
        sample_count=len(candidate_points),
        rms_error=rms_error,
        config=config,
    )


def _subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def ransac_plane(points: np.ndarray, threshold: float, iterations: int, seed: int):
    if threshold <= 0:
        raise ValueError("RANSAC threshold must be positive.")

    rng = np.random.default_rng(seed)
    best_normal: Optional[np.ndarray] = None
    best_d = 0.0
    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    best_error = np.inf

    for _ in range(max(1, iterations)):
        ids = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue

        normal = normal / norm
        d = -float(normal @ p0)
        distances = np.abs(points @ normal + d)
        inliers = distances <= threshold
        count = int(inliers.sum())
        if count < 3:
            continue

        error = float(distances[inliers].mean())
        if count > best_count or (count == best_count and error < best_error):
            best_normal = normal
            best_d = d
            best_inliers = inliers
            best_count = count
            best_error = error

    if best_normal is None or best_inliers is None:
        raise ValueError("RANSAC failed to find a valid plane.")
    return best_normal, best_d, best_inliers


def refine_plane_pca(points: np.ndarray, up_hint: np.ndarray):
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / len(points)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    if normal @ up_hint < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


def build_world_to_ground_transform(normal: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rotation = rotation_between_vectors(normal, ez)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return transform


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    dot = float(np.clip(source @ target, -1.0, 1.0))

    if dot > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-12:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(source @ basis) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(source, basis)
        axis = axis / np.linalg.norm(axis)
        return _axis_angle_rotation(axis, np.pi)

    axis = np.cross(source, target)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    return _axis_angle_rotation(axis, angle)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_camera_rt(camera_R: np.ndarray, camera_T: np.ndarray, world_to_ground: np.ndarray):
    rotation = world_to_ground[:3, :3]
    translation = world_to_ground[:3, 3]
    world_to_camera_rotation = camera_R.T
    new_world_to_camera_rotation = world_to_camera_rotation @ rotation.T
    new_translation = camera_T - new_world_to_camera_rotation @ translation
    return new_world_to_camera_rotation.T, new_translation
