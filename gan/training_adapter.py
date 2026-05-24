from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import deque
import random

import torch

from .qwen_discriminator import QwenDiscriminatorConfig, QwenImageDiscriminator, QwenPreprocessConfig
from .scheduler import CameraNoiseScheduler, GanScheduleConfig, GanWeightScheduler, ViewNoiseConfig


def add_gan_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("GAN / LLM Discriminator Parameters")
    group.add_argument("--gan_enabled", action="store_true", default=False)
    group.add_argument("--gan_lambda", type=float, default=0.0)
    group.add_argument("--gan_start_iter", type=int, default=15_000)
    group.add_argument("--gan_warmup_iters", type=int, default=2_000)
    group.add_argument("--gan_warmup_curve", type=str, default="linear", choices=["linear", "cosine"])
    group.add_argument("--gan_view_every", type=int, default=1)
    group.add_argument("--gan_d_steps", type=int, default=1)
    group.add_argument("--gan_d_buffer_size", type=int, default=16)
    group.add_argument("--gan_d_batch_pairs", type=int, default=4)

    group.add_argument("--gan_translation_std_start", type=float, default=0.0)
    group.add_argument("--gan_translation_std_end", type=float, default=0.02)
    group.add_argument("--gan_rotation_std_deg_start", type=float, default=0.0)
    group.add_argument("--gan_rotation_std_deg_end", type=float, default=1.0)

    group.add_argument("--gan_llm_model", type=str, default="Qwen/Qwen3.5-0.8B")
    group.add_argument("--gan_llm_dtype", type=str, default="bfloat16")
    group.add_argument("--gan_llm_device", type=str, default="cuda")
    group.add_argument("--gan_scene_prompt", type=str, default="A 3D Gaussian Splatting reconstruction.")

    group.add_argument("--gan_lora_r", type=int, default=8)
    group.add_argument("--gan_lora_alpha", type=int, default=16)
    group.add_argument("--gan_lora_dropout", type=float, default=0.05)
    group.add_argument("--gan_lora_lr", type=float, default=2e-4)
    group.add_argument("--gan_lora_weight_decay", type=float, default=0.0)
    group.add_argument("--gan_lora_grad_clip", type=float, default=1.0)
    group.add_argument(
        "--gan_lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    group.add_argument("--gan_preprocess_min_pixels", type=int, default=256 * 256)
    group.add_argument("--gan_preprocess_max_pixels", type=int, default=4096 * 4096)
    group.add_argument("--gan_preprocess_patch_size", type=int, default=16)
    group.add_argument("--gan_preprocess_temporal_patch_size", type=int, default=2)
    group.add_argument("--gan_preprocess_merge_size", type=int, default=2)
    group.add_argument("--gan_preprocess_interpolation", type=str, default="bicubic", choices=["bicubic", "bilinear"])
    group.add_argument("--gan_preprocess_float_input_range", type=str, default="zero_one", choices=["zero_one", "zero_255"])


class GanTrainingAdapter:
    """Coordinates simultaneous Qwen-LoRA discriminator and 3DGS generator updates."""

    def __init__(self, args: Namespace):
        self.args = args
        self.schedule = GanWeightScheduler(
            GanScheduleConfig(
                enabled=args.gan_enabled,
                gan_lambda=args.gan_lambda,
                start_iter=args.gan_start_iter,
                warmup_iters=args.gan_warmup_iters,
                warmup_curve=args.gan_warmup_curve,
                view_every=args.gan_view_every,
            )
        )
        self.camera_noise = CameraNoiseScheduler(
            self.schedule,
            ViewNoiseConfig(
                translation_std_start=args.gan_translation_std_start,
                translation_std_end=args.gan_translation_std_end,
                rotation_std_deg_start=args.gan_rotation_std_deg_start,
                rotation_std_deg_end=args.gan_rotation_std_deg_end,
            ),
        )
        self.discriminator = QwenImageDiscriminator(self._discriminator_config())
        self.supervised_buffer = deque(maxlen=max(1, args.gan_d_buffer_size))

    def _discriminator_config(self) -> QwenDiscriminatorConfig:
        preprocess = QwenPreprocessConfig(
            use_differentiable_processor=True,
            min_pixels=self.args.gan_preprocess_min_pixels,
            max_pixels=self.args.gan_preprocess_max_pixels,
            patch_size=self.args.gan_preprocess_patch_size,
            temporal_patch_size=self.args.gan_preprocess_temporal_patch_size,
            merge_size=self.args.gan_preprocess_merge_size,
            interpolation=self.args.gan_preprocess_interpolation,
            float_input_range=self.args.gan_preprocess_float_input_range,
        )
        return QwenDiscriminatorConfig(
            model_name=self.args.gan_llm_model,
            device=self.args.gan_llm_device,
            dtype=self.args.gan_llm_dtype,
            scene_prompt=self.args.gan_scene_prompt,
            preprocess=preprocess,
            enable_thinking_for_logits=False,
            lora_r=self.args.gan_lora_r,
            lora_alpha=self.args.gan_lora_alpha,
            lora_dropout=self.args.gan_lora_dropout,
            lora_target_modules=self.args.gan_lora_target_modules,
            lora_lr=self.args.gan_lora_lr,
            lora_weight_decay=self.args.gan_lora_weight_decay,
            lora_grad_clip=self.args.gan_lora_grad_clip,
        )

    @classmethod
    def from_args(cls, args: Namespace) -> "GanTrainingAdapter | None":
        if not getattr(args, "gan_enabled", False):
            return None
        return cls(args)

    @staticmethod
    def inactive_metrics() -> dict:
        return {
            "gan_weight": 0.0,
            "gan_loss": 0.0,
            "gan_weighted_loss": 0.0,
            "gan_prob_gt": 0.0,
            "gan_prob_pred": 0.0,
            "gan_pred_label": -1,
            "gan_top_token_id": -1,
            "gan_d_loss": 0.0,
            "gan_d_real_loss": 0.0,
            "gan_d_fake_loss": 0.0,
            "gan_d_real_prob_gt": 0.0,
            "gan_d_fake_prob_gt": 0.0,
            "gan_d_batch_pairs": 0,
            "gan_d_buffer_size": 0,
        }

    def observe_supervised(self, gt_image: torch.Tensor, pred_image: torch.Tensor) -> None:
        self.supervised_buffer.append(
            {
                "gt": gt_image.detach(),
                "pred": pred_image.detach(),
            }
        )

    def discriminator_batch(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch_pairs = min(
            max(1, self.args.gan_d_batch_pairs),
            len(self.supervised_buffer),
        )
        if batch_pairs == len(self.supervised_buffer):
            samples = list(self.supervised_buffer)
        else:
            samples = random.sample(list(self.supervised_buffer), batch_pairs)
        return [sample["gt"] for sample in samples], [sample["pred"] for sample in samples]

    def maybe_perturb_camera(self, camera, iteration: int):
        if not self.schedule.should_apply(iteration):
            return None, 0.0
        return self.camera_noise.perturb(camera, iteration), self.schedule.weight(iteration)

    def step(
        self,
        iteration: int,
        viewpoint_cam,
        gaussians,
        pipe,
        bg,
        render_fn,
        train_test_exp: bool,
        separate_sh: bool,
    ) -> tuple[torch.Tensor | None, dict]:
        perturbed_cam, weight = self.maybe_perturb_camera(viewpoint_cam, iteration)
        if perturbed_cam is None:
            return None, self.inactive_metrics()

        d_metrics = {}
        d_loss = None
        for _ in range(max(1, self.args.gan_d_steps)):
            gt_batch, pred_batch = self.discriminator_batch()
            d_loss, d_metrics = self.discriminator.train_discriminator_batch(
                gt_batch,
                pred_batch,
                scene_prompt=self.args.gan_scene_prompt,
            )

        render_pkg = render_fn(
            perturbed_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh,
        )
        raw_gan_loss, g_metrics = self.discriminator.generator_lsgan_loss(
            render_pkg["render"],
            scene_prompt=self.args.gan_scene_prompt,
        )
        weighted_gan_loss = raw_gan_loss * weight
        metrics = {
            **self.inactive_metrics(),
            "gan_weight": weight,
            "gan_loss": float(raw_gan_loss.detach().cpu()),
            "gan_weighted_loss": float(weighted_gan_loss.detach().cpu()),
            **g_metrics,
            **d_metrics,
        }
        if d_loss is not None:
            metrics["gan_d_loss"] = float(d_loss.detach().cpu())
        metrics["gan_d_buffer_size"] = len(self.supervised_buffer)
        return weighted_gan_loss, metrics
