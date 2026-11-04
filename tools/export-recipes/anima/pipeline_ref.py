"""Build the reference fixture of Anima's **host-side** pipeline (M1-P4 iteration 3 / step 2 of
ADR 0016).

Only the 4 graphs that `anima/export.py` writes go into the IR; everything around them —
tokenization / the scheduler (sigmas) / the timestep embedding table / CFG / the Euler update /
latent denormalization / padding to 512 — is host code. **This is the numeric source of truth for
that host side**: the end-to-end chain E2E on the Deno side (`packages/e2e/tests/e2e_anima_test.ts`)
compares the glue written in TS with the corresponding values of this fixture before using it.

The source of truth is diffusers 0.39's `modular_pipelines/anima/` (encoders.py /
before_denoise.py / denoise.py / decoders.py). This script writes those 4 blocks out
line by line and takes the reference on **the plain diffusers path that does not go through the
patch layer** (the equivalence of `anima.patch` is measured separately by
`anima/export.py --verify` — this keeps the checks independent. Going through the patch layer
here too would put a patch bug into both the reference and the code under test in the same form,
and it would pass with a difference of 0).

Output (default `<repo>/outputs/series/anima-pipeline/`; with `--dtype f16`, its `-f16` variant):

    pipeline.safetensors   tensors of each step (`_TENSOR_ROLES` below indexes their purpose)
    pipeline.json          prompt, step count, shift, CFG scale, LoRA (`lora` / `lora_scale`),
                           list of shapes

MUST: do not write the output directly under `models/<repo>/` (the distribution layout). That
tree holds only the files the manifest declares and is **uploaded to HF as is**; a file outside
the declaration makes `verify_dist` stop (the same reason the audio demo got its own
`outputs/misc/sbv2-demo/`).

MUST: a fixture of a compressed series is taken after fake-quantizing **all 4 components**
(ADR 0006). Leaving even one with the original weights makes that step's reference the numbers of
a different model, and the difference along the chain becomes a mix of "quantization error +
implementation error". The rounding is applied right after each model is loaded.

MUST: `--dtype i8` is **i8 for the DiT only, f16 for the other 3** (`COMPONENT_DTYPES`). The asset
series is laid out as `anima-i8/transformer` +
`anima-f16/{text_encoder,text_conditioner,vae_decoder}` (ADR 0019), so the fixture's rounding
maps 1:1 to that layout. Making everything i8 would make only the text path's reference the
numbers of a model different from the assets that run.

    uv run python -m anima.pipeline_ref
    uv run python -m anima.pipeline_ref --steps 32 --ref-steps 2
    uv run python -m anima.pipeline_ref --dtype f16
    uv run python -m anima.pipeline_ref --dtype i8
    uv run python -m anima.pipeline_ref --dtype f16 --steps 10 --ref-steps 10 \
        --guidance-scale 1.0 --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors \
        --out ../../outputs/series/anima-pipeline-turbo-f16

`--act-quant` mirrors **w8a8** (`SessionOptions.linearCompute: "a8"`): it fake-quantizes the
inputs of the DiT's eligible `nn.Linear`s to per-token i8 before taking the reference (the source
of truth for the numeric spec is `karume.act_quant`). It pairs with the i8 weights, so it must be
combined with `--dtype i8`:

    uv run python -m anima.pipeline_ref --dtype i8 --act-quant --steps 10 \
        --ref-steps 10 --guidance-scale 1.0 \
        --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors \
        --out ../../outputs/series/anima-pipeline-turbo-i8a8

`--resolution` is **WxH** (a square can use the shorthand). A non-square reference is what the
DiT of `--dit-graph dyn` and the VAE of `--vae-tiling` are compared against (#23), and the
spelling is the same as the demo's `--resolution`:

    uv run python -m anima.pipeline_ref --dtype f16 --steps 10 --ref-steps 2 \
        --guidance-scale 1.0 --resolution 1344x768 \
        --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors \
        --out ../../outputs/series/anima-pipeline-turbo-f16-1344x768

The DiT forward runs 4 times by default (2 steps × cond/uncond); the CFG=1 turbo series drops
uncond, so it runs as many times as `--ref-steps` (10 in the example above). It is CPU f32, so it
takes a few minutes. Peak RAM is about 12GiB: the DiT's resident 7.29GiB + activations
(recon §7).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _shared.paths import SERIES_ROOT
from karume.act_quant import attach_act_quant, detach_act_quant
from karume.convert import normalize_boundary_tensor
from karume.quantize import fake_quant_int8, round_weights_to_bf16, round_weights_to_f16

from .resolution import parse_resolution, resolution_meta

DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
#: The series this script can take references for. MUST **not be taken from `WEIGHT_DTYPES` of
#: `export.py`** — that is the set of "storage dtypes that can be written as assets", which
#: includes series that **have no reference path** like i4 (no entry in the 3 tables below).
#: Coupling them would widen only what argparse accepts, and nothing would fail until the bare
#: KeyError of `component_dtype` / `DEFAULT_OUTS` — dying with no diagnostic after minutes spent
#: loading text_encoder, which does not fail loudly (failing with the reason). When adding an
#: entry, add it to the 3 tables below at the same time.
REF_DTYPES: tuple[str, ...] = ("f32", "f16", "bf16", "i8")
#: The default location of the outputs (per storage dtype). **On the series side, separate from
#: the distribution layout `models/`** (the MUST above).
DEFAULT_OUTS = {
    "f32": SERIES_ROOT / "anima-pipeline",
    "f16": SERIES_ROOT / "anima-pipeline-f16",
    "bf16": SERIES_ROOT / "anima-pipeline-bf16",
    "i8": SERIES_ROOT / "anima-pipeline-i8",
}

#: Series → effective storage dtype per component (1:1 with the asset series — the MUST above).
#: The keys are the labels passed to `_fake_quant`; only the i8 series splits the DiT from the
#: other 3.
COMPONENT_DTYPES = {
    "f32": {},
    "f16": {},
    "bf16": {},
    "i8": {"transformer": "i8"},
}
#: The default for components not listed in `COMPONENT_DTYPES` (the f32 series does not round).
_FALLBACK_DTYPE = {"f32": "f32", "f16": "f16", "bf16": "bf16", "i8": "f16"}


def component_dtype(dtype: str, label: str) -> str:
    """The effective storage dtype of one component (the series layout is decided here, in one
    place)."""
    return COMPONENT_DTYPES[dtype].get(label, _FALLBACK_DTYPE[dtype])


def _fake_quant(dtype: str, model: torch.nn.Module, label: str) -> None:
    """Round the weights to values the series' storage can represent (ADR 0006 — MUST happen before
    the reference is taken)."""
    effective = component_dtype(dtype, label)
    if effective == "f32":
        return
    if effective == "i8":
        report = fake_quant_int8(model)
    elif effective == "bf16":
        report = round_weights_to_bf16(model)
    else:
        report = round_weights_to_f16(model)
    print(f"[fake-quant] {label} ({effective}): {report.describe()}", flush=True)


def _apply_lora(path: Path | None, scale: float, model: torch.nn.Module, prefix: str) -> None:
    """Bake a distillation LoRA such as turbo in before export (the same discipline as
    `anima/export.py`).

    MUST: call it before `_fake_quant` (the same ordering constraint as ADR 0016 — baking after
    rounding puts ΔW off the rounding grid, and the re-rounding at storage time disagrees with the
    reference). On the text_conditioner side lora_B was measured to be all zeros (a no-op), so
    this pipeline applies it to the transformer only.
    """
    if path is None:
        return
    from .lora import fuse_lora, load_lora_state_dict

    state = load_lora_state_dict(path)
    report = fuse_lora(model, state, prefix, scale)
    print(f"[lora] {prefix}: {report.describe()}", flush=True)


#: Anima specializes in anime images (danbooru-style tags). One fixed prompt, in English.
PROMPT = (
    "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, "
    "outdoors, smile, upper body, masterpiece, best quality"
)
#: MUST: **must not be the empty string** — the T5 ids of an empty prompt have length 1, which
#: falls outside the conditioner's accepted set (`Dim("Ttgt", min=2)`) and cannot be fed to the
#: exported graph. The pipeline default is `""`, but the model card also recommends CFG 4–5 + a
#: negative prompt.
NEGATIVE_PROMPT = "low quality, worst quality, blurry, bad anatomy, jpeg artifacts"

NUM_TRAIN_TIMESTEPS = 1000  # scheduler_config.json
SHIFT = 3.0  # ditto (use_dynamic_shifting=false, so a static shift)
GUIDANCE_SCALE = 4.0  # the ClassifierFreeGuidance default (encoders.py / denoise.py)
MIN_SEQUENCE_LENGTH = 512  # AnimaTextConditioner.config.min_sequence_length
SPATIAL_COMPRESSION = 8
LATENT_CHANNELS = 16

#: Seed of `latents_init`. **Fixed** (a regeneration is byte-identical — the same rule as the
#: goldens).
SEED = 20260802

#: Index of the purpose of the fixture's tensors (written as is into the JSON metadata, for
#: readers on the Deno side).
_TENSOR_ROLES: dict[str, str] = {
    "sigmas": "FlowMatchEuler sigma sequence (terminal 0 at the end). Parity target of the TS code",
    "timesteps_proj": "timestep embedding table for all steps [steps, 2048]. 2nd input of the DiT",
    "latents_mean": "VAE per-channel latent mean (denormalization constant)",
    "latents_std": "VAE per-channel latent standard deviation (denormalization constant)",
    "latents_init": "initial noise [1,16,h,w] (fixed seed). Start of the chain",
    "qwen_input_ids": "Qwen2 BPE ids. Input of text_encoder",
    "t5_input_ids": "T5 Unigram ids. 2nd input of text_conditioner",
    "qwen_hidden_states": "text_encoder output (after multiplying by the all-ones mask = identity)",
    "encoder_hidden_states": "conditioner output zero-padded to 512. 3rd input of the DiT",
    "noise_cond_stepNNNN": "DiT output of step NNNN (cond side). Parity target of the CFG glue",
    "noise_uncond_stepNNNN": "DiT output of step NNNN (uncond side). Same as above. "
    "A guidance_scale=1.0 fixture has no such key (the uncond branch is not computed)",
    "latents_stepNNNN": "latent after the Euler update of step NNNN",
    "latents_denorm": "final latent after denormalization. Input of the VAE decoder",
    "image": "VAE decode output [1,3,H,W] (with `_decode`'s clamp(-1,1) = same place as the IR)",
}


def sigma_schedule(steps: int, shift: float) -> np.ndarray:
    """Reproduces `AnimaSetTimestepsStep` + `FlowMatchEulerDiscreteScheduler.set_timesteps`.

    This path passes `sigmas=linspace(1, 1/N, N)`, so `_sigma_to_t` is not used and only the
    static shift `shift·s / (1 + (shift−1)·s)` applies. A terminal 0 is appended because `step()`
    reads `sigmas[i+1]`.

    MUST: convert to f32 at the same point as the scheduler (moving `astype` later computes in
    f64, changes the last digit, and is no longer bit-identical to the TS side's sequential
    `Math.fround` code).
    """
    sigmas = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])


@dataclass(frozen=True)
class TextStack:
    """The loaded text front end (2 tokenizers + Qwen3 + conditioner).

    They are held together for callers that "load once and run several prompts" (the calibration
    inputs of `anima.calib`) — calling `from_pretrained` per prompt would load Qwen3 that many
    times.
    """

    tokenizer: Any
    t5_tokenizer: Any
    encoder: torch.nn.Module
    conditioner: torch.nn.Module


def load_text_stack(repo: str, dtype: str) -> TextStack:
    """Load the text front end and round it to the series' storage dtype (the precondition of
    {@link encode_prompt}).

    MUST: round here (before the reference is taken) — moving it later computes only the reference
    with the original weights.
    """
    from diffusers import AnimaTextConditioner
    from transformers import AutoTokenizer, Qwen3Model

    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    t5_tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="t5_tokenizer")
    encoder = Qwen3Model.from_pretrained(
        repo, subfolder="text_encoder", dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    conditioner = AnimaTextConditioner.from_pretrained(repo, subfolder="text_conditioner")
    conditioner.to(torch.float32).eval()
    _fake_quant(dtype, encoder, "text_encoder")
    _fake_quant(dtype, conditioner, "text_conditioner")
    return TextStack(
        tokenizer=tokenizer, t5_tokenizer=t5_tokenizer, encoder=encoder, conditioner=conditioner
    )


def encode_prompt(stack: TextStack, max_len: int, text: str) -> dict[str, torch.Tensor]:
    """Run one prompt through `AnimaTextEncoderStep` + `AnimaTextConditioningStep`.

    The tokenizers are the repository's bundled `tokenizer.json` (Qwen2 BPE / T5 Unigram). With
    `padding="longest"` and a single prompt the mask is **all ones**, so both
    `prompt_embeds * mask` and the conditioner's mask multiplication are the identity — this is
    why the wrapper (`anima.patch`) has no mask.

    The keys have no prefix (the caller adds role spellings such as `neg_`).
    """
    qwen = stack.tokenizer(
        [text], padding="longest", max_length=max_len, truncation=True, return_tensors="pt"
    )
    t5 = stack.t5_tokenizer(
        [text], padding="longest", max_length=max_len, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        hidden = stack.encoder(
            input_ids=qwen.input_ids,
            attention_mask=qwen.attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
        hidden = hidden * qwen.attention_mask.to(hidden).unsqueeze(-1)
        embeds = stack.conditioner(
            source_hidden_states=hidden,
            target_input_ids=t5.input_ids,
            target_attention_mask=t5.attention_mask,
            source_attention_mask=qwen.attention_mask,
        )
    if embeds.shape[1] != MIN_SEQUENCE_LENGTH:
        raise ValueError(
            f"conditioner output length {embeds.shape[1]} is not {MIN_SEQUENCE_LENGTH}"
            f" (the T5 ids {t5.input_ids.shape[1]} are too long) — does not match the DiT's"
            " input shape"
        )
    # MUST: write the ids in the IR's semantic dtype (i32) — one conversion point (ADR 0009).
    return {
        "qwen_input_ids": normalize_boundary_tensor(qwen.input_ids, "qwen ids"),
        "t5_input_ids": normalize_boundary_tensor(t5.input_ids, "t5 ids"),
        "qwen_hidden_states": hidden.contiguous(),
        "encoder_hidden_states": embeds.contiguous(),
    }


def encode_text(repo: str, max_len: int, dtype: str) -> dict[str, torch.Tensor]:
    """Map the 2 fixed prompts (positive / negative) into the fixture's key space."""
    stack = load_text_stack(repo, dtype)
    out: dict[str, torch.Tensor] = {}
    for tag, text in (("", PROMPT), ("neg_", NEGATIVE_PROMPT)):
        for key, value in encode_prompt(stack, max_len, text).items():
            out[f"{tag}{key}"] = value
    return out


def timesteps_proj_table(
    repo: str, sigmas: np.ndarray, dtype: str, lora: Path | None, lora_scale: float
) -> tuple[torch.Tensor, torch.nn.Module]:
    """Return the timestep embedding table for all steps `[steps, 2048]` and the loaded DiT.

    The table is exactly the 2nd input of `AnimaDit` (where `anima.patch` promoted the timestep
    embedding to a graph input). The DiT is returned too so that the reference denoise right
    after reuses the same instance and the 7.29GiB load happens once.

    It follows `AnimaLoopBeforeDenoiser`'s `timestep = timesteps[i] / num_train_timesteps` as is
    (`timesteps = sigmas · num_train_timesteps`, so the value returns to sigma).

    NOTE (measured; the earlier statement "the f32 round trip changes the last digit" **was wrong
    and is corrected**): for all 32 values of steps=32 / shift=3, `f32(f32(s·1000)/1000) == s`, so
    the round trip changes not a single bit. The order is copied so that it can be followed 1:1
    against the upstream formula, not as a constraint based on measured rounding.
    """
    from diffusers import CosmosTransformer3DModel

    model = CosmosTransformer3DModel.from_pretrained(repo, subfolder="transformer")
    model.to(torch.float32).eval()
    # MUST: bake the LoRA before rounding (the same ordering constraint as anima/export.py —
    # ADR 0016).
    _apply_lora(lora, lora_scale, model, "transformer")
    # MUST: round **before** building the table. time_proj is a plain sinusoid with no weights,
    # but unless the order is uniformly "before the reference", it would drift silently if
    # time_embed gains weights in the future.
    _fake_quant(dtype, model, "transformer")
    rows = []
    for sigma in sigmas[:-1]:
        t = torch.tensor([sigma * NUM_TRAIN_TIMESTEPS], dtype=torch.float32)
        with torch.no_grad():
            rows.append(model.time_embed.time_proj(t / NUM_TRAIN_TIMESTEPS).to(torch.float32))
    return torch.cat(rows, dim=0), model


def reference_steps(
    model: torch.nn.Module,
    latents: torch.Tensor,
    embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    sigmas: np.ndarray,
    size: tuple[int, int],
    ref_steps: int,
    guidance: float,
) -> dict[str, torch.Tensor]:
    """`AnimaDenoiseStep`, `ref_steps` times. CFG is `uncond + scale·(cond − uncond)`.

    On the plain path the latent is `(B,C,1,h,w)`. `padding_mask` is zeros at pixel resolution
    (the pipeline default); on the Karume side `anima.patch.AnimaDit` folds it into a constant-zero
    channel.

    The raw cond / uncond DiT outputs are kept too — without them the Deno side **cannot
    parity-check the CFG and Euler host glue on its own**, and sees it only mixed with the DiT's
    error.

    MUST: when `guidance == 1.0`, the uncond branch is **not computed** (Turbo LoRA is designed to
    run at CFG=1 — recon L3). `uncond + 1.0·(cond − uncond)` reduces to `cond` mathematically, but
    naively computing both (1) adds one unneeded DiT call and (2) is not bit-identical to `cond`
    because of the floating-point rounding order. To mirror the "single call" of real use, the
    `noise_uncond_stepNNNN` key itself is not written in this case — the fixture's key set shows
    directly that "the uncond branch is gone".
    """
    # MUST: `padding_mask` is at **pixel resolution** (not latent), with axes `[1,1,H,W]`. For a
    # non-square shape this is the only place where a W / H mix-up does not slip past a shape
    # check, so `size` is taken as `(width, height)` and swapped here (the caller uses the WxH
    # spelling order).
    width, height = size
    padding_mask = latents.new_zeros(1, 1, height, width)
    skip_uncond = guidance == 1.0
    out: dict[str, torch.Tensor] = {}
    x = latents
    for index in range(ref_steps):
        started = time.perf_counter()
        timestep = torch.tensor(
            [sigmas[index] * NUM_TRAIN_TIMESTEPS / NUM_TRAIN_TIMESTEPS], dtype=torch.float32
        )
        branches = (embeds,) if skip_uncond else (embeds, negative_embeds)
        preds = []
        for encoder_hidden_states in branches:
            with torch.no_grad():
                preds.append(
                    model(
                        hidden_states=x,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                )
        tag = f"step{index + 1:04d}"
        if skip_uncond:
            (cond,) = preds
            noise_pred = cond
        else:
            cond, uncond = preds
            noise_pred = uncond + guidance * (cond - uncond)
            out[f"noise_uncond_{tag}"] = uncond.squeeze(2).contiguous()
        x = x + float(sigmas[index + 1] - sigmas[index]) * noise_pred
        out[f"noise_cond_{tag}"] = cond.squeeze(2).contiguous()
        out[f"latents_{tag}"] = x.squeeze(2).contiguous()
        print(
            f"[denoise] step {index + 1}/{ref_steps} sigma={sigmas[index]:.4f}"
            f" ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    return out


def denormalize_latents(vae: torch.nn.Module, latents4: torch.Tensor) -> torch.Tensor:
    """The denormalization of `AnimaVaeDecoderStep` (rank4 → rank5).

    MUST: decoders.py **builds the reciprocal of std and divides by it**. Rewriting it as
    `latents * std` changes the last digit (the TS side is written in the same order).
    """
    z = latents4.unsqueeze(2)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    return z / inv_std + mean


def decode_latents(repo: str, latents4: torch.Tensor, dtype: str) -> dict[str, torch.Tensor]:
    """Reproduces `AnimaVaeDecoderStep` (denormalize → decode). Taken on the plain **unpatched**
    path.

    `image` is exactly the return value of `vae.decode`. `AutoencoderKLQwenImage._decode` applies
    `torch.clamp(-1, 1)` at the end (the clamp that `AnimaVaeDecoder` bakes in is this one, not
    one from postprocess), so **both the fixture side and the IR side are clamped at the same
    place** — the comparison needs no host-side mirror of the clamp.
    """
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(repo, subfolder="vae")
    vae.to(torch.float32).eval()
    _fake_quant(dtype, vae, "vae")
    z = denormalize_latents(vae, latents4)
    with torch.no_grad():
        image = vae.decode(z, return_dict=False)[0][:, :, 0]
    return {
        "latents_mean": torch.tensor(vae.config.latents_mean),
        "latents_std": torch.tensor(vae.config.latents_std),
        "latents_denorm": z.squeeze(2).contiguous(),
        "image": image.contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default per --dtype: outputs/series/anima-pipeline{,-f16,-i8}/)",
    )
    parser.add_argument(
        "--dtype",
        choices=REF_DTYPES,
        default="f32",
        help="a compressed series fake-quantizes all 4 components before taking the reference"
        " (i8 makes only the DiT i8)",
    )
    parser.add_argument("--steps", type=int, default=32, help="total steps of the sigma schedule")
    parser.add_argument(
        "--ref-steps",
        type=int,
        default=2,
        help="steps to run in torch, keeping the reference latents",
    )
    parser.add_argument(
        "--resolution",
        default="512",
        help="WxH (e.g. 1344x768). A square can use the shorthand (512 = 512x512)"
        " — the same spelling as the demo's --resolution",
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=GUIDANCE_SCALE,
        help="CFG guidance scale (1.0 skips the uncond branch — the value Turbo LoRA runs at)",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        default=None,
        help="LoRA to bake into the DiT before export (not applied to text_conditioner, a known"
        " no-op)",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale")
    parser.add_argument(
        "--act-quant",
        action="store_true",
        help="fake-quantize the inputs of the DiT's eligible linears to per-token i8"
        " (mirrors the runtime's linearCompute:'a8' — combine with --dtype i8 for the weights)",
    )
    args = parser.parse_args()
    try:
        width, height = parse_resolution(args.resolution)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.act_quant and args.dtype != "i8":
        raise SystemExit(
            f"--act-quant must be combined with --dtype i8 (got {args.dtype}) — "
            "i8 activations apply only to linears with resident i8 weights (design §4.1)"
        )
    if args.out is None:
        args.out = DEFAULT_OUTS[args.dtype]

    if args.ref_steps > args.steps:
        raise SystemExit(f"--ref-steps ({args.ref_steps}) exceeds --steps ({args.steps})")

    sigmas = sigma_schedule(args.steps, SHIFT)
    tensors: dict[str, torch.Tensor] = {"sigmas": torch.from_numpy(sigmas)}
    print(f"[sigmas] {sigmas[0]:.4f} … {sigmas[-2]:.4f} → 0 ({args.steps} steps)", flush=True)

    tensors.update(encode_text(args.repo, args.max_sequence_length, args.dtype))
    print(
        f"[text] qwen={list(tensors['qwen_input_ids'].shape)}"
        f" t5={list(tensors['t5_input_ids'].shape)}"
        f" neg qwen={list(tensors['neg_qwen_input_ids'].shape)}"
        f" neg t5={list(tensors['neg_t5_input_ids'].shape)}",
        flush=True,
    )

    # Drawn with the same shape and dtype as `AnimaPrepareLatentsStep.prepare_latents`.
    # MUST: the axis order is `[..., H, W]` (the reverse of the WxH spelling). Swapping them here
    # for a non-square shape keeps the element count but transposes the latent, and only the
    # reference becomes a different image.
    latents = torch.randn(
        (1, LATENT_CHANNELS, 1, height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float32,
    )
    tensors["latents_init"] = latents.squeeze(2).contiguous()

    proj, model = timesteps_proj_table(args.repo, sigmas, args.dtype, args.lora, args.lora_scale)
    tensors["timesteps_proj"] = proj
    # Mirrors w8a8 (`SessionOptions.linearCompute: "a8"`). Applied to **the DiT only** — the
    # runtime also sets the knob only on the transformer's Session, so applying it to text / VAE
    # too would make only the reference the numbers of a different model (the same discipline as
    # COMPONENT_DTYPES of `_fake_quant`).
    act_handles: list[object] = []
    if args.act_quant:
        act_handles, attached = attach_act_quant(model)
        # MUST: print the count. Taking the reference with 0 would hide that "w8 numbers were
        # taken while meaning w8a8" (the same practice as the always-on diagnostics of ADR 0006).
        print(f"[act-quant] transformer: per-token i8 applied to {attached} eligible linears")
        if attached == 0:
            raise SystemExit(
                "--act-quant was given but 0 linears are eligible (the eligibility check is broken)"
            )
    try:
        tensors.update(
            reference_steps(
                model,
                latents,
                tensors["encoder_hidden_states"],
                tensors["neg_encoder_hidden_states"],
                sigmas,
                (width, height),
                args.ref_steps,
                args.guidance_scale,
            )
        )
    finally:
        detach_act_quant(act_handles)
    del model

    tensors.update(
        decode_latents(args.repo, tensors[f"latents_step{args.ref_steps:04d}"], args.dtype)
    )
    image = tensors["image"]
    print(f"[decode] image={list(image.shape)} range=[{image.min():.4f}, {image.max():.4f}]")

    from safetensors.torch import save_file

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "pipeline.safetensors"))
    meta: dict[str, Any] = {
        "repo": args.repo,
        "dtype": args.dtype,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": args.steps,
        "ref_steps": args.ref_steps,
        **resolution_meta(width, height),
        "seed": SEED,
        "shift": SHIFT,
        "guidance_scale": args.guidance_scale,
        "act_quant": args.act_quant,
        "lora": str(args.lora) if args.lora is not None else None,
        "lora_scale": args.lora_scale if args.lora is not None else None,
        "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
        "min_sequence_length": MIN_SEQUENCE_LENGTH,
        "image_range": [float(image.min()), float(image.max())],
        "roles": _TENSOR_ROLES,
        "tensors": {name: list(value.shape) for name, value in tensors.items()},
    }
    (args.out / "pipeline.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"fixture OK: {len(tensors)} tensors → {args.out}", flush=True)


if __name__ == "__main__":
    main()
