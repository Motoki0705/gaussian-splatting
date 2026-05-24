from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class GanScheduleConfig:
    enabled: bool = False
    gan_lambda: float = 0.0
    start_iter: int = 15_000
    warmup_iters: int = 2_000
    warmup_curve: str = "linear"
    view_every: int = 1


@dataclass
class ViewNoiseConfig:
    translation_std_start: float = 0.0
    translation_std_end: float = 0.02
    rotation_std_deg_start: float = 0.0
    rotation_std_deg_end: float = 1.0


class GanWeightScheduler:
    def __init__(self, config: GanScheduleConfig):
        self.config = config

    def progress(self, iteration: int) -> float:
        if not self.config.enabled or iteration < self.config.start_iter:
            return 0.0
        if self.config.warmup_iters <= 0:
            return 1.0
        raw = (iteration - self.config.start_iter) / self.config.warmup_iters
        raw = min(1.0, max(0.0, raw))
        if self.config.warmup_curve == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * raw)
        if self.config.warmup_curve != "linear":
            raise ValueError(f"Unsupported GAN warmup curve: {self.config.warmup_curve}")
        return raw

    def weight(self, iteration: int) -> float:
        return self.config.gan_lambda * self.progress(iteration)

    def should_apply(self, iteration: int) -> bool:
        if self.weight(iteration) <= 0.0:
            return False
        return self.config.view_every <= 1 or iteration % self.config.view_every == 0


class PerturbedCamera:
    """Camera-like object with a perturbed pose and inherited intrinsics."""

    def __init__(self, base_camera, world_view_transform: torch.Tensor):
        self.uid = getattr(base_camera, "uid", None)
        self.colmap_id = getattr(base_camera, "colmap_id", None)
        self.R = getattr(base_camera, "R", None)
        self.T = getattr(base_camera, "T", None)
        self.FoVx = base_camera.FoVx
        self.FoVy = base_camera.FoVy
        self.image_name = f"{base_camera.image_name}_gan_perturbed"
        self.image_width = base_camera.image_width
        self.image_height = base_camera.image_height
        self.znear = base_camera.znear
        self.zfar = base_camera.zfar
        self.alpha_mask = None
        self.original_image = getattr(base_camera, "original_image", None)
        self.depth_reliable = False
        self.invdepthmap = None
        self.depth_mask = None
        self.world_view_transform = world_view_transform
        self.projection_matrix = base_camera.projection_matrix
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    if angle.item() < 1e-8:
        return eye
    axis = axis_angle / angle
    x, y, z = axis
    zeros = torch.zeros((), dtype=axis.dtype, device=axis.device)
    k = torch.stack(
        [
            torch.stack([zeros, -z, y]),
            torch.stack([z, zeros, -x]),
            torch.stack([-y, x, zeros]),
        ]
    )
    return eye + torch.sin(angle) * k + (1.0 - torch.cos(angle)) * (k @ k)


class CameraNoiseScheduler:
    def __init__(self, schedule: GanWeightScheduler, config: ViewNoiseConfig):
        self.schedule = schedule
        self.config = config

    def current_stds(self, iteration: int) -> tuple[float, float]:
        p = self.schedule.progress(iteration)
        t = self.config.translation_std_start + p * (
            self.config.translation_std_end - self.config.translation_std_start
        )
        r = self.config.rotation_std_deg_start + p * (
            self.config.rotation_std_deg_end - self.config.rotation_std_deg_start
        )
        return t, r

    def perturb(self, camera, iteration: int, generator: torch.Generator | None = None) -> PerturbedCamera:
        trans_std, rot_std_deg = self.current_stds(iteration)
        world_view = camera.world_view_transform
        c2w = torch.inverse(world_view).clone()

        if trans_std > 0:
            noise = torch.randn(
                3,
                dtype=c2w.dtype,
                device=c2w.device,
                generator=generator,
            ) * trans_std
            c2w[3, :3] = c2w[3, :3] + noise

        if rot_std_deg > 0:
            rot_std_rad = math.radians(rot_std_deg)
            axis_angle = torch.randn(
                3,
                dtype=c2w.dtype,
                device=c2w.device,
                generator=generator,
            ) * rot_std_rad
            delta = _axis_angle_to_matrix(axis_angle)
            c2w[:3, :3] = c2w[:3, :3] @ delta

        return PerturbedCamera(camera, torch.inverse(c2w))
