"""Loading the upstream MOSS-Image components as the upstream pipeline uses them, and the few lines
of `MOSSImagePipeline.__call__` the references need outside a pipeline call (ADR 0138 decision 11).

Every class comes from the pinned, hash-checked `pipeline_moss_image.py`
({@link moss.sources.load_pipeline_module}) or from the pinned diffusers / transformers. The loaders
follow the HF model card's quick start (`DiffusionPipeline.from_pretrained(...,
torch_dtype={"default": bf16, "vae": f32})`, then the scheduler re-made from its config) with one
deliberate change: the DiT is read in f32 for the references (bf16 → f32 is exact, so the weights
are the same values; only the arithmetic is f32).

The initial noise of the references is torch's CPU `randn` from a seeded generator, injected as
`latents=` (the Wan precedent — `wan.few_step_ref`); karume's own seeded noise (splitmix64 +
Box–Muller) is a different sequence, and its tests read these latents from the records.

MUST: diffusers and transformers are imported inside the functions (the `moss` group is not in the
default sync — `tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from moss import sources

#: The text-encoder width (`transformer/config.json` `text_dim`).
TEXT_DIM = 4096

#: The DiT's latent channels (`latent_channels`: 32 VAE channels × a 2×2 patch).
LATENT_CHANNELS = 128

#: Pixels per DiT latent cell (`MOSSImagePipeline.__call__`: `height // 16`).
PIXELS_PER_CELL = 16

#: The upstream defaults (HF README "Sampling settings"; `MOSSImagePipeline.__call__`).
GUIDANCE_SCALE = 3.0
UNCOND_SKIP_LAYERS: tuple[int, ...] = (22,)
NUM_INFERENCE_STEPS = 50

#: The longest text upstream accepts (`max_sequence_length` of `encode_prompt` and `__call__`).
MAX_SEQUENCE_LENGTH = 4096


def upstream(*, text: bool = False) -> tuple[Path, ModuleType]:
    """The pinned snapshot (product parts, or with `text` the text parts too) and the verified
    pipeline module."""
    snapshot = sources.local_snapshot()
    if text:
        text_snapshot = sources.text_snapshot()
        if text_snapshot != snapshot:
            raise sources.MossSourceError(f"{text_snapshot} and {snapshot} are different snapshots")
    return snapshot, sources.load_pipeline_module(snapshot)


def model_index(snapshot: Path) -> dict[str, Any]:
    """`model_index.json` of the snapshot (the classes `DiffusionPipeline.from_pretrained` uses)."""
    return json.loads((snapshot / "model_index.json").read_text(encoding="utf-8"))


def load_transformer(snapshot: Path, module: ModuleType, dtype: torch.dtype = torch.float32) -> Any:
    """`MOSSImageTransformer` from `transformer/`, in eval mode, every tensor in `dtype`.

    `rope.bands` is a derived f32 constant that `_RoPE` rebuilds on load whatever the dtype; it is
    checked to be f32 (upstream's "fp32 RoPE frequencies").
    """
    transformer = module.MOSSImageTransformer.from_pretrained(
        snapshot / "transformer", torch_dtype=dtype
    ).eval()
    wrong = [n for n, p in transformer.named_parameters() if p.dtype != dtype]
    if wrong:
        raise RuntimeError(f"transformer parameters not in {dtype}: {wrong[:5]}")
    if transformer.rope.bands.dtype != torch.float32:
        raise RuntimeError(f"rope.bands is {transformer.rope.bands.dtype}, not float32")
    return transformer


def load_scheduler(snapshot: Path, module: ModuleType) -> Any:
    """`MOSSImageScheduler` re-made from its config (the quick start's
    `type(pipe.scheduler).from_config(pipe.scheduler.config)`)."""
    scheduler_class = module.MOSSImageScheduler
    return scheduler_class.from_config(scheduler_class.load_config(snapshot / "scheduler"))


def load_vae(snapshot: Path) -> Any:
    """`AutoencoderKLFlux2` from `vae/` in f32 (upstream decodes in f32), in eval mode."""
    from diffusers import AutoencoderKLFlux2

    index = model_index(snapshot)
    if index["vae"] != ["diffusers", "AutoencoderKLFlux2"]:
        raise RuntimeError(f"model_index.json names the VAE {index['vae']}")
    vae = AutoencoderKLFlux2.from_pretrained(snapshot / "vae", torch_dtype=torch.float32).eval()
    if any(p.dtype != torch.float32 for p in vae.parameters()):
        raise RuntimeError("the VAE is not float32")
    return vae


def build_pipeline(module: ModuleType, *, transformer: Any, scheduler: Any, vae: Any = None) -> Any:
    """`MOSSImagePipeline` without the text encoder (the references pass embeddings)."""
    return module.MOSSImagePipeline(
        transformer=transformer, scheduler=scheduler, text_encoder=None, tokenizer=None, vae=vae
    )


def seeded_latents(seed: int, height: int, width: int) -> torch.Tensor:
    """The initial noise `[1, 128, height/16, width/16]` in f32 (torch CPU `randn`, seeded)."""
    if height <= 0 or width <= 0 or height % PIXELS_PER_CELL or width % PIXELS_PER_CELL:
        raise ValueError(f"{height}×{width}: height and width must be positive multiples of 16")
    generator = torch.Generator().manual_seed(seed)
    shape = (1, LATENT_CHANNELS, height // PIXELS_PER_CELL, width // PIXELS_PER_CELL)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


# Third-party code notice. `denormalize` and `unpatchify` below are adapted from
# `MOSSImagePipeline.__call__` in `pipeline_moss_image.py` of OpenMOSS-Team/MOSS-Image-SFT (revision
# pinned in `moss.sources`), licensed under the Apache License, Version 2.0. The statements are
# upstream's, in upstream's order, split into two functions so each step can be recorded; nothing
# else was changed. `moss.reference` checks them bit for bit against the decode input of the
# upstream pipeline in the same process.


def denormalize(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Undo the latent BatchNorm with its running statistics (`z·sqrt(var+eps) + mean`)."""
    z = latents.to(dtype=vae.dtype)
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    return z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(
        1, -1, 1, 1
    )


def unpatchify(z: torch.Tensor) -> torch.Tensor:
    """`[B, 128, h, w]` → `[B, 32, 2h, 2w]` (the 2×2 patch back to space)."""
    b, c, h, w = z.shape
    return (
        z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    )


class VaeRecorder:
    """Forward hooks on the VAE that keep the decode input and the tensors around the mid-block
    attention (where the decoder is split into three graphs — ADR 0138 decision 8).

    Kept: `vae.z` (the input of `post_quant_conv`, i.e. what `decode` received), `vae.pre_attention`
    (the output of the mid block's first resnet = the attention's input), `vae.post_attention` (the
    attention's output) and `vae.decoded` (the decoder's output before the clamp).
    """

    def __init__(self, vae: Any) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        mid = vae.decoder.mid_block
        self._handles = [
            vae.post_quant_conv.register_forward_pre_hook(self._keep_input("vae.z")),
            mid.resnets[0].register_forward_hook(self._keep_output("vae.pre_attention")),
            mid.attentions[0].register_forward_hook(self._keep_output("vae.post_attention")),
            vae.decoder.register_forward_hook(self._keep_output("vae.decoded")),
        ]

    def _keep(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.tensors:
            raise RuntimeError(f"{name} was recorded twice — one decode per recorder")
        self.tensors[name] = tensor.detach().clone()

    def _keep_input(self, name: str) -> Any:
        return lambda _module, args: self._keep(name, args[0])

    def _keep_output(self, name: str) -> Any:
        return lambda _module, _args, output: self._keep(name, output)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __enter__(self) -> VaeRecorder:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
