from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvF


DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-0.8B"


DEFAULT_DISCRIMINATOR_PROMPT = (
    "You are a binary discriminator for 3D Gaussian Splatting renders.\n"
    "The image is paired with this scene prompt: {scene_prompt}\n"
    "Decide whether the image is a generated prediction or a ground-truth photo.\n"
    "Use 0 for pred and 1 for GT.\n"
    "Think inside <think>...</think> in at most 40 words, then put the final answer on the last line.\n"
    "The final line must be exactly one character: 0 or 1."
)


@dataclass
class QwenPreprocessConfig:
    use_differentiable_processor: bool = True
    min_pixels: int = 256 * 256
    max_pixels: int = 4096 * 4096
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    rescale_factor: float = 1.0 / 255.0
    interpolation: str = "bicubic"
    antialias: bool = True
    float_input_range: str = "zero_one"


@dataclass
class QwenDiscriminatorConfig:
    model_name: str = DEFAULT_MODEL_NAME
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 1024
    scene_prompt: str = "A real outdoor scene reconstructed by 3D Gaussian Splatting."
    prompt_template: str = DEFAULT_DISCRIMINATOR_PROMPT
    enable_thinking_for_logits: bool = False
    enable_thinking_for_generation: bool = True
    preprocess: QwenPreprocessConfig = field(default_factory=QwenPreprocessConfig)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_lr: float = 2e-4
    lora_weight_decay: float = 0.0
    lora_grad_clip: float = 1.0


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16", "half"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def label_token_ids(tokenizer) -> dict[int, int]:
    """Return vocabulary indices for discriminator labels.

    For Qwen/Qwen3.5-0.8B these are expected to be {0: 15, 1: 16}; this
    function keeps the mapping explicit and model-tokenizer-derived.
    """

    ids: dict[int, int] = {}
    for label in (0, 1):
        encoded = tokenizer.encode(str(label), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Label {label!r} is not a single token: {encoded}")
        ids[label] = encoded[0]
    return ids


def load_image(image: str | Path | Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def qwen_smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Resize rule used by Qwen VL processors."""

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class QwenTorchImageProcessor:
    """Differentiable Qwen image processor.

    The Hugging Face Qwen image processor does:
    RGB conversion, smart resize to multiples of patch_size * merge_size,
    rescale/normalize, then Qwen VL patch flattening. This implementation keeps
    those operations in torch for render tensors so autograd can flow back to
    the 3DGS renderer.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, config: QwenPreprocessConfig):
        self.config = config
        self.size = {
            "shortest_edge": config.min_pixels,
            "longest_edge": config.max_pixels,
        }
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.merge_size = config.merge_size
        self.image_mean = list(config.image_mean)
        self.image_std = list(config.image_std)
        self.rescale_factor = config.rescale_factor
        self.do_convert_rgb = True
        self.do_resize = True
        self.do_rescale = True
        self.do_normalize = True

    def __call__(self, images=None, **kwargs):
        return self.preprocess(images, **kwargs)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs.get("min_pixels", self.size["shortest_edge"])
        max_pixels = images_kwargs.get("max_pixels", self.size["longest_edge"])
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        resized_height, resized_width = qwen_smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)

    def preprocess(self, images, return_tensors: str | None = None, **kwargs):
        from transformers.image_processing_base import BatchFeature

        image_list = self._as_image_list(images)
        processed = [self._preprocess_one(image, **kwargs) for image in image_list]
        pixel_values = torch.cat([item[0] for item in processed], dim=0)
        image_grid_thw = torch.tensor(
            [item[1] for item in processed],
            dtype=torch.long,
            device=pixel_values.device,
        )
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def _as_image_list(self, images) -> list[Any]:
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            return [image for image in images]
        if isinstance(images, (list, tuple)):
            flattened = []
            for image in images:
                flattened.extend(self._as_image_list(image))
            return flattened
        return [images]

    def _to_chw_tensor(self, image) -> torch.Tensor:
        if isinstance(image, dict) and "image" in image:
            image = image["image"]
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            return tvF.pil_to_tensor(image)
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)

        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[0] not in (1, 3, 4) and image.shape[-1] in (1, 3, 4):
            image = image.permute(2, 0, 1).contiguous()
        elif image.ndim != 3:
            raise ValueError(f"Expected CHW/HWC image tensor, got shape {tuple(image.shape)}")

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        elif image.shape[0] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {image.shape[0]}")
        return image

    def _preprocess_one(self, image, **kwargs) -> tuple[torch.Tensor, list[int]]:
        image = self._to_chw_tensor(image)
        height, width = image.shape[-2:]
        patch_size = kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = kwargs.get("temporal_patch_size", self.temporal_patch_size)
        merge_size = kwargs.get("merge_size", self.merge_size)
        min_pixels = kwargs.get("min_pixels", self.size["shortest_edge"])
        max_pixels = kwargs.get("max_pixels", self.size["longest_edge"])

        resized_height, resized_width = qwen_smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if (height, width) != (resized_height, resized_width):
            image = tvF.resize(
                image,
                [resized_height, resized_width],
                interpolation=self._interpolation_mode(),
                antialias=self.config.antialias,
            )

        image = self._normalize(image)
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        patches = image.unsqueeze(0).reshape(
            1,
            image.shape[0],
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        flatten_patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(1, grid_h * grid_w, image.shape[0] * temporal_patch_size * patch_size * patch_size)
        )
        return flatten_patches.reshape(grid_h * grid_w, -1), [1, grid_h, grid_w]

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        if image.is_floating_point():
            image = image.to(dtype=torch.float32)
            if self.config.float_input_range == "zero_one":
                return tvF.normalize(image, self.image_mean, self.image_std)
            if self.config.float_input_range == "zero_255":
                pass
            else:
                raise ValueError(f"Unsupported float_input_range: {self.config.float_input_range}")

        image = image.to(dtype=torch.float32)
        mean = [value / self.rescale_factor for value in self.image_mean]
        std = [value / self.rescale_factor for value in self.image_std]
        return tvF.normalize(image, mean, std)

    def _interpolation_mode(self) -> InterpolationMode:
        if self.config.interpolation == "bicubic":
            return InterpolationMode.BICUBIC
        if self.config.interpolation == "bilinear":
            return InterpolationMode.BILINEAR
        raise ValueError(f"Unsupported interpolation: {self.config.interpolation}")


class QwenProcessor:
    """AutoProcessor-compatible wrapper with differentiable image preprocessing."""

    def __init__(self, processor, preprocess_config: QwenPreprocessConfig):
        self.processor = processor
        self.preprocess_config = preprocess_config
        self.tokenizer = processor.tokenizer
        if preprocess_config.use_differentiable_processor:
            self.image_processor = QwenTorchImageProcessor(preprocess_config)
            self.processor.image_processor = self.image_processor
        else:
            self.image_processor = processor.image_processor

    @classmethod
    def from_pretrained(cls, model_name: str, preprocess_config: QwenPreprocessConfig | None = None):
        from transformers import AutoProcessor

        return cls(
            AutoProcessor.from_pretrained(model_name),
            preprocess_config or QwenPreprocessConfig(),
        )

    def __getattr__(self, name: str):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.processor.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


class QwenImageDiscriminator:
    """Small wrapper around Qwen3.5 image-text logits for 0/1 decisions."""

    def __init__(self, config: QwenDiscriminatorConfig):
        from transformers import AutoModelForImageTextToText
        from peft import LoraConfig, get_peft_model

        self.config = config
        self.processor = QwenProcessor.from_pretrained(config.model_name, config.preprocess)
        dtype = resolve_torch_dtype(config.dtype)
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map=None,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        target_modules = [item.strip() for item in config.lora_target_modules.split(",") if item.strip()]
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, peft_config).to(config.device)
        self.model.eval()

        self.token_ids = label_token_ids(self.processor.tokenizer)
        self.lora_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if "lora_" in name
        ]
        if not self.lora_parameters:
            raise RuntimeError("No LoRA parameters were created for the Qwen discriminator.")
        self.optimizer = torch.optim.AdamW(
            self.lora_parameters,
            lr=config.lora_lr,
            weight_decay=config.lora_weight_decay,
        )
        self.set_lora_trainable(False)

    @property
    def label_ids_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.token_ids[0], self.token_ids[1]],
            dtype=torch.long,
            device=self.model.device,
        )

    def set_lora_trainable(self, trainable: bool) -> None:
        for parameter in self.lora_parameters:
            parameter.requires_grad_(trainable)

    def build_messages(self, image: Image.Image | torch.Tensor, scene_prompt: str | None = None):
        prompt = self.config.prompt_template.format(
            scene_prompt=scene_prompt or self.config.scene_prompt
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def build_messages_batch(
        self,
        images: list[Image.Image | torch.Tensor],
        scene_prompt: str | None = None,
    ):
        return [self.build_messages(image, scene_prompt) for image in images]

    def prepare_inputs(
        self,
        image: str | Path | Image.Image | torch.Tensor,
        scene_prompt: str | None = None,
        enable_thinking: bool | None = None,
    ):
        messages = self.build_messages(load_image(image), scene_prompt)
        if enable_thinking is None:
            enable_thinking = self.config.enable_thinking_for_logits
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
        return {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def prepare_inputs_batch(
        self,
        images: list[str | Path | Image.Image | torch.Tensor] | torch.Tensor,
        scene_prompt: str | None = None,
        enable_thinking: bool | None = None,
    ):
        if isinstance(images, torch.Tensor):
            image_list = [image for image in images] if images.ndim == 4 else [images]
        else:
            image_list = [load_image(image) for image in images]
        messages = self.build_messages_batch(image_list, scene_prompt)
        if enable_thinking is None:
            enable_thinking = self.config.enable_thinking_for_logits
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
        return {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    @torch.no_grad()
    def generate(self, image: str | Path | Image.Image | torch.Tensor, scene_prompt: str | None = None) -> str:
        inputs = self.prepare_inputs(
            image,
            scene_prompt,
            enable_thinking=self.config.enable_thinking_for_generation,
        )
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.processor.decode(new_ids, skip_special_tokens=True)

    @torch.no_grad()
    def next_label_logits(self, image: str | Path | Image.Image, scene_prompt: str | None = None):
        """Return next-token logits/probabilities restricted to labels [0, 1]."""

        logits_01, probs_01, next_logits = self.label_logits_and_probs(
            image,
            scene_prompt,
            enable_thinking=self.config.enable_thinking_for_logits,
        )
        return {
            "logits_01": logits_01[0].detach().float().cpu(),
            "probs_01": probs_01[0].detach().cpu(),
            "top_token_id": int(next_logits.argmax(dim=-1).item()),
            "label_token_ids": dict(self.token_ids),
        }

    def label_logits_and_probs(
        self,
        image: str | Path | Image.Image | torch.Tensor,
        scene_prompt: str | None = None,
        enable_thinking: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return differentiable next-token logits/probs for labels [0, 1].

        Qwen parameters are normally frozen, but this method intentionally does
        not use torch.no_grad(): gradients must flow through pixel_values back
        to the rendered image.
        """

        inputs = self.prepare_inputs(image, scene_prompt, enable_thinking=enable_thinking)
        outputs = self.model(**inputs, use_cache=False)
        next_logits = outputs.logits[:, -1, :]
        logits_01 = next_logits.index_select(dim=-1, index=self.label_ids_tensor)
        probs_01 = F.softmax(logits_01.float(), dim=-1)
        return logits_01, probs_01, next_logits

    def label_logits_and_probs_batch(
        self,
        images: list[str | Path | Image.Image | torch.Tensor] | torch.Tensor,
        scene_prompt: str | None = None,
        enable_thinking: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.prepare_inputs_batch(images, scene_prompt, enable_thinking=enable_thinking)
        outputs = self.model(**inputs, use_cache=False)
        next_logits = outputs.logits[:, -1, :]
        logits_01 = next_logits.index_select(dim=-1, index=self.label_ids_tensor)
        probs_01 = F.softmax(logits_01.float(), dim=-1)
        return logits_01, probs_01, next_logits

    def generator_lsgan_loss(
        self,
        image: torch.Tensor,
        scene_prompt: str | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Generator loss: make a rendered pred image classified as GT."""

        self.model.eval()
        self.set_lora_trainable(False)
        logits_01, probs_01, next_logits = self.label_logits_and_probs(
            image,
            scene_prompt,
            enable_thinking=self.config.enable_thinking_for_logits,
        )
        loss = lsgan_generator_loss_from_probs(probs_01)
        pred_label = int(torch.argmax(probs_01.detach(), dim=-1)[0].item())
        return loss, {
            "gan_logits_0": float(logits_01.detach().float()[0, 0].cpu()),
            "gan_logits_1": float(logits_01.detach().float()[0, 1].cpu()),
            "gan_prob_pred": float(probs_01.detach()[0, 0].cpu()),
            "gan_prob_gt": float(probs_01.detach()[0, 1].cpu()),
            "gan_pred_label": pred_label,
            "gan_top_token_id": int(next_logits.detach().argmax(dim=-1).item()),
        }

    def train_discriminator_batch(
        self,
        gt_images: list[torch.Tensor],
        pred_images: list[torch.Tensor],
        scene_prompt: str | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Update LoRA discriminator on a batch of GT/render pairs.

        Renders are detached so the discriminator step never updates 3DGS.
        Only LoRA parameters are trainable here.
        """

        self.model.train()
        self.set_lora_trainable(True)
        self.optimizer.zero_grad(set_to_none=True)

        num_real = len(gt_images)
        num_fake = len(pred_images)
        if num_real == 0 or num_fake == 0:
            raise ValueError("Discriminator batch needs at least one GT and one pred image.")

        images = [image.detach() for image in gt_images] + [image.detach() for image in pred_images]
        logits_01, probs_01, _ = self.label_logits_and_probs_batch(
            images,
            scene_prompt,
            enable_thinking=self.config.enable_thinking_for_logits,
        )
        labels = torch.cat(
            [
                torch.ones(num_real, dtype=torch.float32, device=probs_01.device),
                torch.zeros(num_fake, dtype=torch.float32, device=probs_01.device),
            ]
        )
        loss = lsgan_discriminator_loss_from_probs(probs_01, labels)
        real_loss = lsgan_discriminator_loss_from_probs(probs_01[:num_real], labels[:num_real])
        fake_loss = lsgan_discriminator_loss_from_probs(probs_01[num_real:], labels[num_real:])
        loss.backward()
        if self.config.lora_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.lora_parameters, self.config.lora_grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.set_lora_trainable(False)
        self.model.eval()

        return loss.detach(), {
            "gan_d_loss": float(loss.detach().cpu()),
            "gan_d_real_loss": float(real_loss.detach().cpu()),
            "gan_d_fake_loss": float(fake_loss.detach().cpu()),
            "gan_d_real_prob_gt": float(probs_01.detach()[:num_real, 1].mean().cpu()),
            "gan_d_fake_prob_gt": float(probs_01.detach()[num_real:, 1].mean().cpu()),
            "gan_d_real_logits_0": float(logits_01.detach().float()[:num_real, 0].mean().cpu()),
            "gan_d_real_logits_1": float(logits_01.detach().float()[:num_real, 1].mean().cpu()),
            "gan_d_fake_logits_0": float(logits_01.detach().float()[num_real:, 0].mean().cpu()),
            "gan_d_fake_logits_1": float(logits_01.detach().float()[num_real:, 1].mean().cpu()),
            "gan_d_batch_pairs": min(num_real, num_fake),
        }

    def train_discriminator(
        self,
        gt_image: torch.Tensor,
        pred_image: torch.Tensor,
        scene_prompt: str | None = None,
    ) -> tuple[torch.Tensor, dict]:
        return self.train_discriminator_batch([gt_image], [pred_image], scene_prompt)


def lsgan_generator_loss_from_probs(probs_01: torch.Tensor) -> torch.Tensor:
    """Generator wants rendered pred images to be classified as GT (label 1)."""

    gt_prob = probs_01[..., 1]
    return 0.5 * torch.mean((gt_prob - 1.0) ** 2)


def lsgan_discriminator_loss_from_probs(probs_01: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    gt_prob = probs_01[..., 1]
    targets = labels.float()
    return 0.5 * torch.mean((gt_prob - targets) ** 2)
