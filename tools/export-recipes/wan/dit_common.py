"""The parts of the Wan DiT export shared by the export script and the probes: the golden case form,
the eager-equivalence check and the weight rounding (ADR 0118 decisions 3 / 8, ADR 0121 decisions
4 / 7).

The users are `wan.ti2v_export_dit` (the i8 series of the TI2V-5B DiT and its goldens),
`wan.dit_probe` (the RAM probe and the isolation runs), `wan.ti2v_few_step_ref` (the i8 fake-quant
DiT of the few-step reference) and `wan.dit_host_fixture` (the upstream DiT of the TS host
fixture).

## Golden files

One case is two files ({@link case_file_names}): `io.<case>.safetensors` holds the graph inputs
(`input.*`) and the patched eager output (`output.0`); `reference.<case>.safetensors` holds the
upstream values ({@link reference_tensors}) — `latents`, `timestep` (i32), `output`, `block.NN`
(each block's output, for cases that keep them) and `output.f64` (the upstream output with the
activations in f64 too, rounded to f32 — the key the TS side reads as `REFERENCE_F64_KEY`). The
graph's first five inputs are {@link INPUT_NAMES}.

## The eager-equivalence check

{@link eager_report} compares the patched S-form wrapper with the upstream plain forward, and
{@link eager_failures} names every claim that breaks: the trunk (every rewrite except the patch
embedding) is bit-identical to upstream, the Linear form of the patch embedding stays within the
reduction-order bound of conv3d ({@link patch_embedding_error_bound}), and the patched output and
the numeric fields are finite. A writer that gets a failure stops before publishing.

MUST: take the reference with **upstream's plain forward** (`dit_patch.reference_dit` /
`dit_patch.reference_dit_layers`). Replacing the attn1 processor does not change the plain path
(module docstring of `dit_patch`), so no check on the order in which the reference is taken is
needed.

## Weight rounding

MUST: the weight rounding ({@link fake_quant_i8} — per-channel i8 RTN; {@link round_to_f16} —
f16-representable values) comes **before** the reference (ADR 0006 / 0019). The rounding is applied
to the S-shaped wrapper and reaches the shared upstream components with the same values (the patch
embedding is a view of the conv weights — `dit_patch.patch_embedding_linear`). The RoPE table (the
buffers of `model.rope`) is not rounded — the product bakes the f32 base tables, and the reference
runs on those values too.
MUST: the eager-equivalence check looks at the weights **after** rounding. The reference and the
patched eager share the same rounded weights. The patch's equivalence (bit-exact trunk, the bound on
the patch embedding) is a claim independent of the weight values (the RoPE rewrite reads no
weights), so checking after rounding does not change what the check means. The same check also pins
that the reference was taken with the rounded weights — if the rounding had not reached the
reference side, the trunk would not be bit-identical.

MUST: diffusers is imported inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from karume.container import Provenance
from karume.convert import normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.quantize import (
    QUANT_MODULE_TYPES,
    Int8Report,
    channel_scale,
    fake_quant_int8,
    iter_quant_targets,
    quantize_to_int8,
    round_weights_to_f16,
)
from wan import dit_patch
from wan.pipeline_ref import TEXT_DIM, assert_no_mps, pad_text_embeds
from wan.sources import DEFAULT_MODEL, SOURCES, local_snapshot

#: The storage dtypes that can be written (= `weight_dtype` of `export_to_file`).
DTYPES = ("f16", "i8")


#: The asset's name and role (the same spelling as Anima's S-shaped DiT — ADR 0109 decision 4).
ROPE_BASE_ASSET = "rope_base"


ROPE_BASE_ROLE = "rope-base"


IO_PREFIX = "io."


REFERENCE_PREFIX = "reference."


#: The key of the f64 reference in `reference.<case>` of full-size cases (`REFERENCE_F64_KEY` on the
#: TS side).
REFERENCE_F64_KEY = "output.f64"


CASE_SUFFIX = ".safetensors"


INPUT_PREFIX = "input."


OUTPUT_PREFIX = "output."


#: The graph input names (the argument names of the wrapper's forward become the IR input names as
#: is).
INPUT_NAMES = ("tokens", "timesteps_proj", "encoder_hidden_states", "rope_cos", "rope_sin")


#: The lower bound likewise (the line that avoids 0 / 1 specialization — the same as Anima's
#: `DIT_SYM_MIN`).
DIT_SYM_MIN = 2


@dataclass(frozen=True)
class CaseSpec:
    """The conditions of one golden case (latent shape, timestep, valid text length, random
    seed)."""

    role: str
    latent_shape: tuple[int, int, int]
    timestep: int
    text_length: int
    seed: int
    #: A suffix that tells apart cases with the same role, S and timestep (the second full-size
    #: t = 999 case).
    variant: str = ""
    #: Whether the golden holds each block's output (`block.NN`) (the counterpart of the per-layer
    #: record — only 2 at full size).
    blocks: bool = True

    def name(self, patch_size: tuple[int, int, int]) -> str:
        frames, height, width = self.latent_shape
        tokens = (frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])
        suffix = f"-{self.variant}" if self.variant else ""
        return f"{self.role}-s{tokens:05d}-t{self.timestep:04d}{suffix}"

    @property
    def full_size(self) -> bool:
        """A full-size case (33 frames of ADR 0118 step 3, 81 frames of step 8) — takes an f64
        reference in addition to f32."""
        return self.role.startswith("full-")


@dataclass(frozen=True)
class Case:
    """One assembled case (graph inputs, upstream reference, patched eager output)."""

    name: str
    spec: CaseSpec
    inputs: tuple[torch.Tensor, ...]
    latents: torch.Tensor
    timestep: torch.Tensor
    reference: torch.Tensor
    #: Each block's output (empty for cases without `spec.blocks`).
    reference_blocks: list[torch.Tensor]
    #: The time taken by upstream's plain forward (the reference) (seconds, CPU f32 — recorded for
    #: the full-size cases).
    reference_seconds: float


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def load_transformer(model_name: str = DEFAULT_MODEL) -> nn.Module:
    """Loads the DiT of the pinned revision on CPU f32 (stops on a machine where MPS is visible —
    decision 3)."""
    assert_no_mps()
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(
        local_snapshot(model_name), subfolder="transformer", torch_dtype=torch.float32
    )
    return model.eval()


def unrounded_weights(dtype: str, model: nn.Module) -> list[str]:
    """The names of upstream weights that are not at the fixed point of the series storage dtype's
    rounding (empty means already rounded).

    - `f16`: every parameter is an f16-representable value (bit-unchanged by an f16 → f32 round
      trip)
    - `i8`: upstream's weight slots (linear and patch embedding) are per-channel i8-representable
      values (bit-unchanged by rounding once more — the note on `karume.quantize.INT8_MAX`)

    The rounding writers ({@link round_to_f16} / {@link fake_quant_i8}) and the side that receives
    the rounded model ({@link float64_references}) go through the same check.
    """
    with torch.no_grad():
        if dtype == "f16":
            return [
                name
                for name, parameter in model.named_parameters()
                if not torch.equal(parameter, parameter.to(torch.float16).to(torch.float32))
            ]
        if dtype == "i8":
            missed = []
            for name, weight, axis in iter_quant_targets(model, op_types=QUANT_MODULE_TYPES):
                scale = channel_scale(weight, axis)
                rounded = quantize_to_int8(weight, scale).to(torch.float32) * scale
                if not torch.equal(rounded, weight):
                    missed.append(name)
            return missed
    raise ValueError(f"unknown rounding for storage dtype {dtype!r} ({' / '.join(DTYPES)})")


def round_to_f16(model: nn.Module, wrapper: nn.Module) -> str:
    """Rounds the weights to f16-representable values through the S-shaped wrapper and checks that
    the rounding reached every upstream parameter.

    The wrapper holds only references to upstream's components and the Linear that is a view of the
    conv weights (it does not hold `rope`). So the rounding reaches every upstream parameter and
    does not reach the buffers of `rope` (the RoPE table).
    MUST: check both — if either fails, it becomes "only the reference has other weights" or "only
    the reference's table is f16", quantization error mixes into the difference and the band loses
    its meaning.
    """
    rope_before = {name: buffer.clone() for name, buffer in model.rope.named_buffers()}
    report = round_weights_to_f16(wrapper)
    missed = unrounded_weights("f16", model)
    if missed:
        raise AssertionError(f"the f16 rounding did not reach upstream parameters: {missed[:5]}")
    moved = [
        name
        for name, buffer in model.rope.named_buffers()
        if not torch.equal(buffer, rope_before[name])
    ]
    if moved:
        raise AssertionError(f"the RoPE table (the rope buffers) was rounded: {moved}")
    return report.describe()


def fake_quant_i8(model: nn.Module, wrapper: nn.Module) -> Int8Report:
    """Rounds the linear weights to per-channel symmetric i8-representable values through the
    S-shaped wrapper (RTN — ADR 0019) and checks that the rounding reached every upstream weight
    slot (the i8 series of ADR 0120 decision 2).

    The wrapper's linears are only upstream's Linears themselves and the patch embedding that is a
    view of the conv weights. So the rounding reaches every upstream Linear and the Conv3d of the
    patch embedding. Biases, norm weights, `scale_shift_table` and the buffers of `rope` are not
    touched by `fake_quant_int8` (they stay upstream's f32 values — decision 2).
    MUST: check that the rounding reached the upstream side — if even one weight is missed, it
    becomes "only the reference has other weights", quantization error mixes into the difference
    and the band loses its meaning (the same reason as {@link round_to_f16}). The check is
    idempotence: an i8-representable value is a fixed point of per-channel quantization (the note on
    `karume.quantize.INT8_MAX`), so bit-unchanged after rounding once more means it was reached. The
    count must also equal the number of upstream weight slots.
    """
    report = fake_quant_int8(wrapper)
    upstream = list(iter_quant_targets(model, op_types=QUANT_MODULE_TYPES))
    missed = unrounded_weights("i8", model)
    if missed:
        raise AssertionError(f"the i8 rounding did not reach upstream weights: {missed[:5]}")
    if len(upstream) != report.modules:
        raise AssertionError(
            f"{report.modules} weights rounded through the wrapper differ from the"
            f" {len(upstream)} upstream weight slots"
        )
    return report


def fake_quant(
    dtype: str, model: nn.Module, wrapper: nn.Module
) -> tuple[str, Mapping[str, torch.Tensor] | None]:
    """Rounds the weights to the representable values of the series storage dtype (MUST come before
    taking the reference — ADR 0006).

    Returns a one-line summary and the i8 scale ledger (passed to emit as is — None for f16).
    """
    if dtype == "i8":
        report = fake_quant_i8(model, wrapper)
        return f"rounded to i8 per-channel — {report.describe()}", report.scales
    if dtype == "f16":
        return f"rounded to f16-representable values — {round_to_f16(model, wrapper)}", None
    raise ValueError(f"cannot write a series of storage dtype {dtype!r} ({' / '.join(DTYPES)})")


def case_inputs(
    model: nn.Module, spec: CaseSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The upstream inputs of one case (latent `[1,16,F,H,W]`, timestep, text context
    `[1,512,4096]`).

    The random numbers are drawn in this one place so that the f32 and f64 references receive the
    same values.
    """
    generator = _generator(spec.seed)
    latents = torch.randn(1, model.config.in_channels, *spec.latent_shape, generator=generator)
    text = torch.randn(spec.text_length, TEXT_DIM, generator=generator)
    # Upstream's pipeline passes the scheduler's int64 timestep after `expand(batch)`.
    timestep = torch.tensor([spec.timestep], dtype=torch.int64)
    return latents, timestep, pad_text_embeds(text)


def build_case(model: nn.Module, spec: CaseSpec) -> Case:
    """Builds the inputs of one case and takes the reference with upstream's plain forward (the
    final output and, with `spec.blocks`, each block's output)."""
    patch_size = tuple(int(size) for size in model.config.patch_size)
    latents, timestep, encoder_hidden_states = case_inputs(model, spec)
    with torch.no_grad():
        started = time.perf_counter()
        if spec.blocks:
            reference, blocks = dit_patch.reference_dit_layers(
                model, latents, timestep, encoder_hidden_states
            )
        else:
            reference = dit_patch.reference_dit(model, latents, timestep, encoder_hidden_states)
            blocks = []
        reference_seconds = time.perf_counter() - started
        print(
            f"[case] {spec.name(patch_size)}: upstream reference {reference_seconds:.1f} s",
            flush=True,
        )
        rope_cos, rope_sin = dit_patch.dit_rope_tables(model.rope, spec.latent_shape)
        inputs = (
            dit_patch.dit_patchify(latents, patch_size),
            dit_patch.dit_timesteps_proj(model, timestep),
            encoder_hidden_states,
            rope_cos,
            rope_sin,
        )
    return Case(
        name=spec.name(patch_size),
        spec=spec,
        inputs=inputs,
        latents=latents,
        timestep=timestep,
        reference=reference,
        reference_blocks=blocks,
        reference_seconds=reference_seconds,
    )


def provenance(model_name: str) -> Provenance:
    """The provenance baked into the container (upstream's license identifier and the pinned
    revision — `wan.sources` is the source of truth)."""
    source = SOURCES[model_name]
    return Provenance(
        license=source.license, notice=NOTICE_FILENAME, upstream_revision=source.revision
    )


class EagerEquivalenceError(AssertionError):
    """The patched eager departs from the claims of ADR 0118 (bit-exact trunk, the bound on the
    difference of the Linear form, finite)."""


def patch_embedding_error_bound(tokens: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    """The upper bound (per element) on the difference between the Linear form of the patch
    embedding and upstream's conv3d that the different reduction order alone can produce.

    `2·γ_{K+1}·(|x|·|W|ᵀ + |b|)` (`γ_n = n·u / (1 − n·u)`, `u = 2⁻²⁴`, K = the number of window
    elements, and the bias addition adds one term). Both reduction orders stay within
    `γ_{K+1}·Σ|terms|` of the true value, so the difference is within twice that. The Linear form
    is not expected to be bit-exact (ADR 0118 — the difference is recorded), but an implementation
    error such as a swapped layout shows up in the size of the values themselves and exceeds this
    bound.
    """
    weight = linear.weight
    magnitude = tokens.abs() @ weight.abs().T
    terms = tokens.shape[-1]
    if linear.bias is not None:
        magnitude = magnitude + linear.bias.abs()
        terms += 1
    unit = 2.0**-24
    return 2 * (terms * unit / (1 - terms * unit)) * magnitude


#: The numeric summary fields that must be finite (`reference_f32_vs_f64_ratio` exists only for the
#: full-size cases — {@link eager_failures} below).
_FINITE_FIELDS = (
    "reference_max_abs",
    "trunk_max_abs_diff",
    "patch_embedding_max_abs_diff",
    "full_max_abs_diff",
)


def eager_failures(report: Mapping[str, Any]) -> list[str]:
    """The points where one eager summary departs from the claims of ADR 0118 (empty means pass).

    MUST: if there is even one, emit does not publish ({@link emit}) and `--verify` exits non-zero.
    It checks that the trunk is bit-exact (the patch's eager equivalence MUST), that the
    patch-embedding difference is within the reduction-order bound
    ({@link patch_embedding_error_bound}), and that the patched output and the numeric fields are
    finite.
    """
    failures = []
    if report.get("trunk_bit_exact") is not True:
        failures.append(
            "the trunk (the rewrites other than the patch embedding) is not bit-identical to"
            " upstream"
        )
    if report.get("patch_embedding_within_bound") is not True:
        failures.append(
            "the difference between the patch embedding's Linear form and conv3d exceeds the"
            " reduction-order bound"
        )
    if report.get("output_finite") is not True:
        failures.append("the patched graph output has non-finite values")
    fields = [*_FINITE_FIELDS]
    if "reference_f32_vs_f64_ratio" in report:
        fields.append("reference_f32_vs_f64_ratio")
    for field in fields:
        value = report.get(field)
        if not isinstance(value, float) or not math.isfinite(value):
            failures.append(f"{field} is not a finite number ({value!r})")
    return failures


def eager_report(
    wrapper: dit_patch.WanDitTokens, model: nn.Module, case: Case
) -> tuple[dict[str, Any], torch.Tensor]:
    """The difference between the patched eager (up to the host unpatchify) and upstream's plain
    output (one case), and the patched graph output `[1,S,pt·ph·pw·C]` (the golden's `output.0` —
    the writer uses it without running again).

    `trunk` is the path with the patch embedding replaced by upstream's conv3d output (a comparison
    of the body alone, including the RoPE rewrite — the claim is bit-exactness); `full` is the
    product path including the Linear form of the patch embedding (the difference is recorded — the
    verdict is whether the patch-embedding difference is within
    {@link patch_embedding_error_bound}). How to read the verdict fields: {@link eager_failures}.

    If the patch embedding's output is bit-identical to upstream's conv3d, trunk does the same
    computation on the same input as full (`forward` is `forward_hidden(patch_embedding(tokens),
    …)`), so it is not run again (`trunk_from_full`). At full size (S = 14,040) one forward takes
    minutes on CPU, so the forwards per case are cut down to this one.

    Attention runs under {@link dit_patch.flash_attention_only} as for the reference (falling back
    to MATH at S = 32,760 OOMs).
    """
    patch_size = wrapper.patch_size
    with torch.no_grad(), dit_patch.flash_attention_only():
        hidden = model.patch_embedding(case.latents).flatten(2).transpose(1, 2)
        embedded = wrapper.patch_embedding(case.inputs[0])
        bound = patch_embedding_error_bound(case.inputs[0], wrapper.patch_embedding)
        started = time.perf_counter()
        tokens_out = wrapper(*case.inputs)
        patched_seconds = time.perf_counter() - started
        full = dit_patch.dit_unpatchify(tokens_out, case.spec.latent_shape, patch_size)
        trunk_from_full = torch.equal(embedded, hidden)
        trunk = (
            full
            if trunk_from_full
            else dit_patch.dit_unpatchify(
                wrapper.forward_hidden(hidden, *case.inputs[1:]),
                case.spec.latent_shape,
                patch_size,
            )
        )
    report = {
        "case": case.name,
        "reference_max_abs": float(case.reference.abs().max()),
        "trunk_bit_exact": torch.equal(trunk, case.reference),
        "trunk_max_abs_diff": float((trunk - case.reference).abs().max()),
        "trunk_from_full": trunk_from_full,
        "patch_embedding_max_abs_diff": float((embedded - hidden).abs().max()),
        "patch_embedding_max_abs": float(hidden.abs().max()),
        "patch_embedding_within_bound": bool(((embedded - hidden).abs() <= bound).all()),
        "output_finite": bool(tokens_out.isfinite().all()),
        "full_bit_exact": torch.equal(full, case.reference),
        "full_max_abs_diff": float((full - case.reference).abs().max()),
        "reference_seconds": round(case.reference_seconds, 1),
        "patched_seconds": round(patched_seconds, 1),
    }
    return report, tokens_out


def _ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """The band metric (max absolute difference ÷ max absolute value of the reference — the same
    shape as `ratioOf` on the TS side), in f64."""
    expected = expected.double()
    return float((actual.double() - expected).abs().max() / expected.abs().max())


def reference_tensors(
    name: str,
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    output: torch.Tensor | None,
    blocks: Sequence[torch.Tensor],
    output_f64: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """The contents of `reference.<case>` (`latents` / `timestep` (i32) / `output` / `block.NN` /
    `output.f64`).

    The writer ({@link _write_case_files}) and the comparison against the layer-by-layer reference
    (`wan.dit_reference compare` — `--f64-only`, which does not take `output`, passes
    `output=None`) go through the same assembly — spelling it in two places would let the comparison
    call a shape that differs from the writer's "identical".
    """
    reference = {
        "latents": latents.contiguous(),
        "timestep": normalize_boundary_tensor(timestep, f"timestep of {name}"),
    }
    if output is not None:
        reference["output"] = output.contiguous()
    reference.update(
        {f"block.{index:02d}": block.contiguous() for index, block in enumerate(blocks)}
    )
    if output_f64 is not None:
        # Stored as values rounded to f32 (the TS safetensors reader does not read F64). The
        # rounding difference is at most 2⁻²⁴·|x| per element, at most 6e-8 as a ratio. How much it
        # moves the normalization's denominator (the CPU f32 reference's ratio against f64) is set
        # by the smallest denominator: for the band cases (`full-band`) alone, under 2% with the
        # S = 14,040 minimum of 3.72e-6 and about 2.9% with the S = 32,760 minimum of 2.07e-6;
        # including acceptance, about 3.3% with the minimum 1.83e-6 (`full-accept-s14040-t0600`).
        # It does not affect the band verdict (worst × 5).
        reference[REFERENCE_F64_KEY] = output_f64.to(torch.float32).contiguous()
    return reference


def case_file_names(name: str) -> tuple[str, str]:
    """The golden file names of one case (`io.<case>` and `reference.<case>`).

    The writer ({@link _write_case_files}) and the comparison's exclusion ({@link check_absent}) go
    through the same spelling.
    """
    return f"{IO_PREFIX}{name}{CASE_SUFFIX}", f"{REFERENCE_PREFIX}{name}{CASE_SUFFIX}"
