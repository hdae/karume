"""Measure the **quantization quality** of w8 / w8a8 / attention a8 on torch CPU (the Q0 quality
gate).

Zero lines of GPU code. Under ADR 0006's fake-quant methodology the E2E tests **measure only
implementation error**, so the quality of the quantization itself has to be measured on a separate
axis (quality gate ① of ADR 0019). This script covers that axis for the Anima DiT.

It runs 10 configurations **in the same process with the same seed** and decomposes them
orthogonally ({@link CONFIGS}):

    (a) f32               the baseline (the plain weights with the LoRA merged)
    (b) w8                weight-only per-channel symmetric i8 fake-quant (`fake_quant_int8`)
    (c) w8a8              (b) + per-token symmetric i8 fake-quant of **the input activations of the
                          eligible linears**
    (d) w8a8-qk           (c) + i8 on **only q / k** of attention
    (e) w8a8-pv           (c) + i8 on **only P̃ / V** of attention
    (f) w8a8-qkpv         (c) + i8 on q/k/P̃/V of attention (the main candidate of attention a8)
    (g) w8-qkpv           (b) + i8 on attention only (its contribution alone with the linear
                          activations in f32)
    (i) w8a8-qkpv-s16     (f) + **the attention score S stored in f16** (the production form of
                          option γ, phase 1)
    (j) w8a8-s16          (c) + only the f16 storage of S (isolates the contribution of the
                          S rounding alone)
    (h) w8a8-qkpv-statsq  (f) with the softmax denominator replaced by **the post-quantization sum**
                          `Σ round(127·P̃)/127`

The labels are **kept as they are so that external references stay stable** (ADR 0030 and
ACTIVE_DESIGN refer to "configuration (h)" by name). The run order is the order above, with (h)
staying last — (h) was **definitively rejected** in phase Q0, so `--until w8a8-qkpv-s16` is the
minimal forward prefix that the option γ gate needs (the comment on {@link CONFIGS}).

The activation quantization of (c) mirrors the numeric spec of
`docs/research/2026-08-03-dp4a-w8a8-design.md` §4.2:

    s = clamp(rowmax(|x|) / 127, f32 tiny)      … row = last axis (per-token)
    q = clamp(round(x / s), -127, +127)         … **closed at ±127** (−128 unused)
    x̂ = q · s

`torch.round` rounds half to even. An all-zero row is exact: `s = tiny` → `q = 0` → `x̂ = 0`.
Eligibility is the same as §4.1 of that doc: "an `nn.Linear` inside the DiT (with or without bias)
with `k % 4 == 0`".

The granularity of the attention quantization of (d)–(h) is the granularity MUST table of
`docs/research/2026-08-04-attention-a8-design.md` §5.1 (= exactly the granularity the runtime can
implement). Details in {@link attention_sim}.
The f16 storage of S in (i)(j) matches the implementation of
`docs/research/2026-08-04-intermediate-f16-design.md` §3.4② / §7 phase 1 exactly in granularity
({@link round_scores_f16}).

MUST: (a)→(b)→… use **the same model instance** in order (fake_quant_int8 overwrites the f32
weights in place, so the order cannot be reversed). A side effect is that the 7.29GiB load happens
only once.

Outputs (`--out`; by convention given explicitly on the scratch side, not next to this file):

    report.md              tables of numbers (relRMS per step / PSNR / relRMS per layer / P̃
                           distribution / gate verdicts)
    attn.json              the attention quantization summary (machine-readable — the gate
                           verdicts are here too)
    image_<config>.png     final images (falls back to image_*.npy without PIL)
    dit_out.safetensors    the latents of every configuration per step (raw data — for further
                           analysis)
    layers.csv             the raw per-layer table (all 454 layers — report.md carries only a
                           summary)

    uv run python -m anima.measure_quant --out /path/to/q0

`--w4-screen` is **a separate mode** (none of the 10 configurations runs — {@link run_w4_screen}).
It lines up the w4 rounding methods (rtn-i4-g32 / nf4 / mxfp4 / kmeans:shared) and **calibrated
rounding** (gptq-rtn / gptq-nf4 / gptq-kmeans — phase J-2) side by side, on a **path separate from
the stacking semantics** of the 10 configurations above: those stack in place on the same instance,
whereas the method comparison restores the pristine weights (the plain f32 weights) before each
configuration (MUST: no stacking). Labels, baselines and output file names are not shared either.
The calibrated methods target **only the linears of the DiT block list**, a different set from the
four methods ({@link CALIB_TARGET}).

NOTE: everything outside the DiT (text_encoder / text_conditioner / VAE) **stays f32** in every
configuration, to narrow the comparison to the DiT's quantization alone (in the asset series
`outputs/series/anima-i8/` the other three are f16, but that difference lands identically on every
configuration and cancels in a relative comparison). The attention replacement is also active **only
during the DiT's denoise loop** (text encode / VAE decode are passed through — the restore is
checked mechanically against `PRISTINE_SDPA`).

NOTE: report generation (assembling `report.md` / `attn.json` / `layers.csv`) lives in
`anima/measure_report.py` — it changes at a different rate from the numeric spec.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from _shared.paths import INPUTS_ROOT
from karume.act_quant import (
    attach_act_quant,
    detach_act_quant,
    is_eligible,
    quantize_rows,
    quantize_rows_parts,
)
from karume.quant_calib import (
    CalibMethod,
    CalibReport,
    GridSpec,
    StageBatch,
    StageSpec,
    calibrate_stages,
)
from karume.quant_methods import (
    DEFAULT_CODEBOOK_LEVELS,
    fake_quant_kmeans,
    fake_quant_mxfp4,
    fake_quant_nf4,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    INT8_MAX,
    QUANT_MODULE_TYPES,
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
)

from .measure_report import build_report, build_summary, write_layer_csv
from .pipeline_ref import (
    LATENT_CHANNELS,
    PROMPT,
    SEED,
    SHIFT,
    SPATIAL_COMPRESSION,
    denormalize_latents,
    encode_text,
    reference_steps,
    sigma_schedule,
    timesteps_proj_table,
)

DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
DEFAULT_LORA = INPUTS_ROOT / "anima" / "anima-turbo-lora-v0.2.safetensors"

#: The turbo operating value (CFG=1 does not compute the uncond branch — the same form as the MUST
#: in anima/pipeline_ref.py).
GUIDANCE = 1.0

#: The SDPA implementation before replacement (the source of truth for the restore check — captured
#: once at import).
PRISTINE_SDPA = F.scaled_dot_product_attention

TINY = torch.finfo(torch.float32).tiny


@dataclass(frozen=True)
class Config:
    """The recipe of one configuration.

    `weight_i8` is the flag "drop the weights to i8 **before** entering this configuration"
    (`fake_quant_int8` is in place, so the table order is the application order as is). `attn` is
    the range of the attention quantization.
    """

    name: str
    weight_i8: bool
    linear_act: bool
    #: `None` / `"qk"` / `"pv"` / `"qkpv"`.
    attn: str | None = None
    #: Use `Σ round(127·P̃)/127` (the post-quantization sum) as the softmax denominator (design doc
    #: §4.3).
    stats_quantized: bool = False
    #: Store the attention score S **in f16** (option γ, phase 1 — {@link round_scores_f16}).
    score_f16: bool = False
    #: Collect the P̃ row peak/rms distribution (true only in a configuration that sees the plain,
    #: unquantized P̃).
    measure_pt: bool = False


#: Run order (MUST: weight rounding is in place, so **the order carries meaning**).
#:
#: The two s16 entries go right after `w8-qkpv` and before `w8a8-qkpv-statsq`. The weights are
#: dropped to i8 in place exactly once, at (b), so anywhere after (b) **the weight state is the
#: same** (the position only decides what falls into the forward prefix of `--until`). The gated
#: `w8a8-qkpv-s16` comes first, so `--until w8a8-qkpv-s16` is the minimal prefix the gate needs: it
#: includes (a)(b)(c), which the summary needs, and (f), the direct control for "the degradation S
#: f16 adds on top of attn-a8", and **leaves out only (h) statsq, which was definitively rejected in
#: phase Q0**. To also see the contribution of the S rounding alone, use `--until w8a8-s16`.
CONFIGS: tuple[Config, ...] = (
    Config("f32", weight_i8=False, linear_act=False),
    Config("w8", weight_i8=True, linear_act=False),
    Config("w8a8", weight_i8=False, linear_act=True, measure_pt=True),
    Config("w8a8-qk", weight_i8=False, linear_act=True, attn="qk"),
    Config("w8a8-pv", weight_i8=False, linear_act=True, attn="pv"),
    Config("w8a8-qkpv", weight_i8=False, linear_act=True, attn="qkpv"),
    Config("w8-qkpv", weight_i8=False, linear_act=False, attn="qkpv"),
    Config("w8a8-qkpv-s16", weight_i8=False, linear_act=True, attn="qkpv", score_f16=True),
    Config("w8a8-s16", weight_i8=False, linear_act=True, score_f16=True),
    Config("w8a8-qkpv-statsq", weight_i8=False, linear_act=True, attn="qkpv", stats_quantized=True),
)


def attn_config_names(configs: tuple[Config, ...]) -> tuple[str, ...]:
    """The configurations that **touch attention** (the subjects of the gate verdicts).

    a8 quantization (`attn`) and the f16 storage of S (`score_f16`) are both changes on the
    attention side, so the gate, the PSNR pairs of ② and the per-layer table of ⑤ look at the same
    set.
    """
    return tuple(config.name for config in configs if config.attn or config.score_f16)


CONFIG_NAMES = tuple(config.name for config in CONFIGS)
#: The baseline of the relative degradation (the PSNR against f32 is already saturated at w8a8 —
#: design doc §5.3).
BASELINE = "w8a8"
ATTN_CONFIGS = attn_config_names(CONFIGS)


# ---- Per-token fake-quant of activations --------------------------------------
#
# The source of truth for the numeric spec (`quantize_rows`) and eligibility (`is_eligible`) is
# `karume.act_quant` (shared with the mirror fixture of `anima/pipeline_ref.py` — copying it into
# two places lets one of them drift from the spec). This is a hook that adds **per-layer error
# accounting** on top of it.


@dataclass
class LayerStat:
    """Error accounting of one layer (held as sums of squares over all steps and all calls)."""

    in_features: int
    out_features: int
    calls: int = 0
    #: The quantization error of the input activations (Σ(x̂−x)² and Σx²).
    in_err2: float = 0.0
    in_ref2: float = 0.0
    #: The output difference (w8a8 − w8). Measured only at the `--measure-step` step (the linear
    #: runs once more).
    out_err2: float = 0.0
    out_ref2: float = 0.0
    out_calls: int = 0
    #: amax/rms per row (an outlier indicator — the larger it is, the more of the i8 grid is
    #: wasted).
    peak_sum: float = 0.0
    peak_rows: int = 0
    peak_max: float = 0.0

    def in_rel_rms(self) -> float:
        return math.sqrt(self.in_err2 / self.in_ref2) if self.in_ref2 > 0 else 0.0

    def out_rel_rms(self) -> float:
        return math.sqrt(self.out_err2 / self.out_ref2) if self.out_ref2 > 0 else 0.0

    def peak_mean(self) -> float:
        return self.peak_sum / self.peak_rows if self.peak_rows else 0.0


@dataclass
class ActQuantHooks:
    """The hooks that drop the inputs of the eligible linears to per-token i8 and account the error
    per layer.

    Only at the `measure_step` step they additionally compute "the output without quantization" to
    get the per-layer output relRMS (the w8 weights are already applied, so this difference is the
    contribution of **activation quantization alone**).
    """

    model: nn.Module
    measure_step: int
    stats: dict[str, LayerStat] = field(default_factory=dict)
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    step: int = -1
    _pending: dict[str, torch.Tensor] = field(default_factory=dict)

    def attach(self) -> None:
        self.handles.append(self.model.register_forward_pre_hook(self._on_model))
        for name, module in self.model.named_modules():
            if not is_eligible(module):
                continue
            self.stats[name] = LayerStat(module.in_features, module.out_features)
            self.handles.append(
                module.register_forward_pre_hook(self._make_pre(name), with_kwargs=False)
            )
            self.handles.append(module.register_forward_hook(self._make_post(name)))

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _on_model(self, _module: nn.Module, _args: tuple[object, ...]) -> None:
        # CFG=1, so one DiT call = one step.
        self.step += 1

    def _make_pre(self, name: str):
        def pre(_module: nn.Module, args: tuple[torch.Tensor, ...]):
            if not args:
                # A linear not called with positional arguments cannot be a target of activation
                # quantization (a drop in their number shows in the `[eligible]` count and in
                # `stat.calls == 0`).
                return None
            x = args[0]
            xq = quantize_rows(x)
            stat = self.stats[name]
            stat.calls += 1
            diff = xq - x
            stat.in_err2 += float(diff.square().sum())
            stat.in_ref2 += float(x.square().sum())
            flat = x.reshape(-1, x.shape[-1])
            peak = flat.abs().amax(dim=-1) / flat.square().mean(dim=-1).sqrt().clamp_min(
                torch.finfo(torch.float32).tiny
            )
            stat.peak_sum += float(peak.sum())
            stat.peak_rows += peak.numel()
            stat.peak_max = max(stat.peak_max, float(peak.amax()))
            if self.step == self.measure_step:
                self._pending[name] = x
            return (xq, *args[1:])

        return pre

    def _make_post(self, name: str):
        def post(module: nn.Module, _args: tuple[object, ...], output: torch.Tensor):
            x = self._pending.pop(name, None)
            if x is None:
                return None
            reference = F.linear(x, module.weight, module.bias)
            stat = self.stats[name]
            stat.out_err2 += float((output - reference).square().sum())
            stat.out_ref2 += float(reference.square().sum())
            stat.out_calls += 1
            return None

        return post


# ---- The a8 shim of attention ---------------------------------------------------
#
# Design = `docs/research/2026-08-04-attention-a8-design.md`. The granularity is the MUST table of
# its §5.1, kept **exactly equal to the granularity the runtime can implement** (loosening it gives
# "an optimistic upper bound that cannot be implemented"):
#
#   q     amax per row of [B·H·M, D] (the reduction axis D is innermost and contiguous, so it is
#         `quantize_rows` itself)
#   k     amax per row of [B·H·N, D] (same as above)
#   P̃     **fixed scale 1/127** (`q_P = round(127·exp(S−m))`). The in-row max is structurally 1.0,
#         so a per-token amax would only give 1/127 = zero room for adaptation (§2.2).
#         **The amax MUST NOT be recomputed** (not even written in the formula, so that the
#         implementation is not misread as "it can").
#   V     amax over **the whole N** per `(b,h,d)` (a scale independent of the reduction axis n = a
#         form that can be factored out). Per-token (per row n) is an optimistic upper bound the
#         runtime cannot implement, so it is **forbidden** (§2.3).
#   S     **stored in f16** (the `score_f16` configurations — option γ design doc §3.4② / §7 phase
#         1). The "half-scaled raw scores" that ①QK writes are dropped to f16 and back, and **that
#         is the only rounding**. ② the row max `m` and the denominator `l = Σexp(S−m)`, and ③'s
#         `exp(S−m)` and `qP = round(127·exp(S−m))` are **all computed in f32 from the same rounded
#         S** (exactly the same granularity as the runtime's ②③ reading the same f16 buffer and
#         widening it to `f32()` — `read()` in attention.ts and the A-tile fill of
#         attention-i8a8.ts).
#
# The half scale (√scale_factor) is applied twice **on the dequant side** (applying it before
# quantization shifts the rounding by one stage and also stops `quantize_rows` from being usable
# unmodified — the ruling of §2.1).
#
# The integer reductions are exact in float64 (|acc| ≤ D·127² and N·127² are orders of magnitude
# below 2^53, so they are exact regardless of the addition order). The purpose is a quality
# measurement, so rounding differences at the f32 ULP level are not reproduced.

#: Histogram of the P̃ row peak/rms (quantiles with bounded memory). peak/rms ≤ √N, so a log₂ upper
#: bound of 8 suffices up to N ≤ 65,536. The resolution is 0.01 log₂ ≈ 0.7%.
PT_HIST_BINS = 800
PT_HIST_LOG_MAX = 8.0


@dataclass
class AttnStat:
    """Accounting of one attention node (over all steps)."""

    calls: int = 0
    #: Histogram and raw maximum of the P̃ row peak/rms (= 1/rms — peak is structurally 1.0).
    pt_hist: torch.Tensor = field(
        default_factory=lambda: torch.zeros(PT_HIST_BINS, dtype=torch.int64)
    )
    pt_rows: int = 0
    pt_max: float = 0.0
    pt_sum: float = 0.0
    #: How far S / O moved from "attention without quantization" (measured at `measure_step` only).
    s_err2: float = 0.0
    s_ref2: float = 0.0
    o_err2: float = 0.0
    o_ref2: float = 0.0
    measured: int = 0

    def s_rel_rms(self) -> float:
        return math.sqrt(self.s_err2 / self.s_ref2) if self.s_ref2 > 0 else 0.0

    def o_rel_rms(self) -> float:
        return math.sqrt(self.o_err2 / self.o_ref2) if self.o_ref2 > 0 else 0.0

    def pt_mean(self) -> float:
        return self.pt_sum / self.pt_rows if self.pt_rows else 0.0

    def pt_quantile(self, q: float) -> float:
        return hist_quantile(self.pt_hist, q)


def hist_add(counts: torch.Tensor, values: torch.Tensor) -> None:
    """Add into log₂-spaced bins (values are ≥ 1 — peak/rms is never below 1 by definition)."""
    scaled = torch.log2(values.reshape(-1).clamp_min(1.0)) * (PT_HIST_BINS / PT_HIST_LOG_MAX)
    index = scaled.to(torch.int64).clamp_(0, PT_HIST_BINS - 1)
    counts += torch.bincount(index, minlength=PT_HIST_BINS)


def hist_quantile(counts: torch.Tensor, q: float) -> float:
    """A quantile of the histogram (linear interpolation on log₂ within a bin)."""
    total = int(counts.sum())
    if total == 0:
        return 0.0
    cumulative = counts.cumsum(0)
    target = q * total
    index = int(torch.searchsorted(cumulative, torch.tensor(target)).clamp_(0, PT_HIST_BINS - 1))
    below = float(cumulative[index - 1]) if index else 0.0
    width = PT_HIST_LOG_MAX / PT_HIST_BINS
    fraction = (target - below) / float(counts[index]) if counts[index] > 0 else 0.0
    return float(2.0 ** ((index + min(max(fraction, 0.0), 1.0)) * width))


def peak_over_rms(pt: torch.Tensor) -> torch.Tensor:
    """`peak/rms` per row of P̃. peak is structurally 1.0 from `exp(0) = 1`, so this is `1/rms`."""
    mean_square = pt.square().sum(dim=-1, dtype=torch.float64) / pt.shape[-1]
    return mean_square.clamp_min(TINY).rsqrt()


@dataclass
class ScoreF16Probe:
    """A mechanical check that the f16 rounding of S **actually fired** (guarding against a vacuous
    pass).

    "Meant to apply the rounding, but it was the identity" shows neither in the numbers nor in the
    elapsed time (the same kind of silence as the ADR 0024 supplement / 0030 decision 5). Every call
    checks that **no call leaves every element unchanged** across the rounding, and the tally goes
    to the gate ({@link build_summary}).
    """

    calls: int = 0
    fired: int = 0
    elements: int = 0
    #: |max| of the raw S (the headroom against f16's limit of 65,504 — evidence that the storage
    #: form of phase 1 does not saturate).
    abs_max: float = 0.0

    def observe(self, before: torch.Tensor, after: torch.Tensor) -> None:
        self.calls += 1
        self.elements += before.numel()
        # Take |max| without a huge `abs()` temporary (S has 270 million elements per node at
        # 1024px).
        self.abs_max = max(self.abs_max, float(before.max()), -float(before.min()))
        # `torch.equal` short-circuits, so it creates no temporary bool tensor.
        if not torch.equal(after, before):
            self.fired += 1


def round_scores_f16(scores: torch.Tensor, probe: ScoreF16Probe | None = None) -> torch.Tensor:
    """Drop S to f16 and back to f32 (the mirror of "the storage form of S" in option γ, phase 1).

    MUST: per attention call this is called **in this one place only**. Both ② the row statistics
    and ③ PV are computed from the rounded S it returns (the runtime reads the same f16 buffer
    twice, so a form that rounds twice would be an upper bound that cannot be implemented).
    """
    rounded = scores.half().float()
    if probe is not None:
        probe.observe(scores, rounded)
    return rounded


def flatten_heads(x: torch.Tensor) -> torch.Tensor:
    """`[B,H,S,D]` → `[B·H,S,D]` (the row/column binding of the scale is defined on this flat
    form)."""
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])


def attention_scores(
    query: torch.Tensor, key: torch.Tensor, scale: float, *, quant_qk: bool
) -> torch.Tensor:
    """Return `S = (q·√sf)·(k·√sf)ᵀ` as `[B·H,M,N]`."""
    q, k = flatten_heads(query), flatten_heads(key)
    if not quant_qk:
        return torch.matmul(q, k.transpose(-1, -2)) * scale
    # MUST: the half scale is on the dequant side (`qs·√sf` and `ks·√sf`) — never applied before
    # quantization.
    half = math.sqrt(scale)
    qq, qs = quantize_rows_parts(q)
    kq, ks = quantize_rows_parts(k)
    accumulator = qq.double() @ kq.double().transpose(-1, -2)
    row_scale = (qs.double() * half) * (ks.double() * half).transpose(-1, -2)
    return (accumulator * row_scale).to(torch.float32)


def attention_sim(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    quant_qk: bool,
    quant_pv: bool,
    stats_quantized: bool,
    score_f16: bool,
    scale: float,
    probe: ScoreF16Probe | None = None,
) -> torch.Tensor:
    """Evaluate SDPA in the `[B,H,M,D]` layout with only the given ranges replaced at i8
    granularity."""
    v = flatten_heads(value)
    scores = attention_scores(query, key, scale, quant_qk=quant_qk)
    if score_f16:
        # The storage rounding happens only here (m / Σ / exp / qP all come from the rounded S
        # afterwards).
        scores = round_scores_f16(scores, probe)
    exponent = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    if not quant_pv:
        probabilities = exponent / exponent.sum(dim=-1, keepdim=True)
        return torch.matmul(probabilities, v).reshape(query.shape)

    # The scale of P̃ is fixed at 1/127 (no clamp needed — the 127 upper bound follows structurally
    # from `exp(0)=1`).
    quantized_pt = torch.round(exponent * INT8_MAX)
    if stats_quantized:
        total = quantized_pt.sum(dim=-1, keepdim=True, dtype=torch.float32) / INT8_MAX
    else:
        total = exponent.sum(dim=-1, keepdim=True)
    # V uses the amax over the whole N per (b,h,d) (independent of the reduction axis n = can be
    # factored out).
    value_scale = torch.clamp(v.abs().amax(dim=-2, keepdim=True) / INT8_MAX, min=TINY)
    quantized_v = torch.round(v / value_scale).clamp_(-INT8_MAX, INT8_MAX)
    accumulator = quantized_pt.double() @ quantized_v.double()
    row_inv = (1.0 / total).double() * (1.0 / INT8_MAX)
    return (accumulator * row_inv * value_scale.double()).to(torch.float32).reshape(query.shape)


@dataclass
class AttnSim:
    """Replace `F.scaled_dot_product_attention` **at op granularity** (module hooks are forbidden).

    Why the replacement has to be at op granularity is exactly the lesson of ADR 0029: diffusers'
    `CosmosAttnProcessor2_*` calls `F.scaled_dot_product_attention` directly **with keyword
    arguments** from inside `attn.processor`, through `attention_dispatch._native_attention` (it
    does not exist as an `nn.Module`). A module hook never fires and stays silent in the direction
    of "meant to quantize, but the degradation looks small".

    Only for attribution, push/pop hooks go on every module under the DiT to keep a name stack (the
    same device as `measure_quant_sbv2.ActQuant`). The top of the stack is the attention module's
    name.

    The scope is **the DiT only**: `attach` / `detach` are called just around the denoise loop, so
    text encode (the start of `run_dit`) and VAE decode (`decode_all`) are passed through.
    """

    model: nn.Module
    mode: str | None
    stats_quantized: bool
    #: Store S in f16 (option γ, phase 1 — an axis independent of a8 quantization).
    score_f16: bool
    measure_pt: bool
    measure_step: int
    #: Fault injection (`--inject drop-attn-quant`): build only the attribution scaffolding, and
    #: apply neither the quantization nor the S rounding.
    disable: bool = False
    #: The firing check of the S f16 rounding (guarding against a vacuous pass — tallied across the
    #: configuration).
    s16: ScoreF16Probe = field(default_factory=ScoreF16Probe)
    stats: dict[str, AttnStat] = field(default_factory=dict)
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)
    step: int = -1
    calls: int = 0
    skipped: int = 0
    patched: bool = False

    def attach(self) -> None:
        self.handles.append(self.model.register_forward_pre_hook(self._on_model))
        for name, module in self.model.named_modules():
            self.handles.append(module.register_forward_pre_hook(self._make_push(name)))
            self.handles.append(module.register_forward_hook(self._pop))
        F.scaled_dot_product_attention = self._wrapper
        self.patched = True

    def detach(self) -> None:
        if self.patched:
            F.scaled_dot_product_attention = PRISTINE_SDPA
            self.patched = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stack.clear()

    def _on_model(self, _module: nn.Module, _args: tuple[object, ...]) -> None:
        # CFG=1, so one DiT call = one step.
        self.step += 1

    def _make_push(self, name: str):
        def push(_module: nn.Module, _args: tuple[object, ...]) -> None:
            self.stack.append(name)

        return push

    def _pop(self, _module: nn.Module, _args: tuple[object, ...], _output: object) -> None:
        self.stack.pop()

    def _wrapper(self, *args: object, **kwargs: object) -> torch.Tensor:
        query = kwargs.get("query", args[0] if len(args) > 0 else None)
        key = kwargs.get("key", args[1] if len(args) > 1 else None)
        value = kwargs.get("value", args[2] if len(args) > 2 else None)
        if not isinstance(query, torch.Tensor) or query.dim() != 4:
            # Count an unexpected call shape instead of silently passing it through (a non-zero
            # count shows up in the report).
            self.skipped += 1
            return PRISTINE_SDPA(*args, **kwargs)
        for name in ("attn_mask", "is_causal", "dropout_p", "enable_gqa"):
            given = kwargs.get(name)
            if given not in (None, False, 0.0, 0):
                raise NotImplementedError(
                    f"SDPA argument the attention shim does not expect: {name}={given}"
                )
        scale = kwargs.get("scale")
        scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else float(scale)

        node = self.stack[-1] if self.stack else "<root>"
        stat = self.stats.setdefault(node, AttnStat())
        stat.calls += 1
        self.calls += 1

        if self.measure_pt:
            # Collected during the run of (c) = **the plain P̃** that the quantizer sees (attention
            # still f32).
            scores = attention_scores(query, key, scale, quant_qk=False)
            ratio = peak_over_rms(torch.exp(scores - scores.amax(dim=-1, keepdim=True)))
            hist_add(stat.pt_hist, ratio)
            stat.pt_rows += ratio.numel()
            stat.pt_sum += float(ratio.sum())
            stat.pt_max = max(stat.pt_max, float(ratio.amax()))
            del scores, ratio

        if self.disable or (self.mode is None and not self.score_f16):
            return PRISTINE_SDPA(*args, **kwargs)

        quant_qk = self.mode is not None and "qk" in self.mode
        quant_pv = self.mode is not None and "pv" in self.mode
        output = attention_sim(
            query,
            key,
            value,
            quant_qk=quant_qk,
            quant_pv=quant_pv,
            stats_quantized=self.stats_quantized,
            score_f16=self.score_f16,
            scale=scale,
            probe=self.s16,
        )
        if self.step == self.measure_step:
            # The difference from "attention without quantization" = the contribution of attention
            # quantization alone (the linear activation quantization is already inside q/k/v).
            reference = attention_sim(
                query,
                key,
                value,
                quant_qk=False,
                quant_pv=False,
                stats_quantized=False,
                score_f16=False,
                scale=scale,
            )
            scores_ref = attention_scores(query, key, scale, quant_qk=False)
            scores_value = (
                attention_scores(query, key, scale, quant_qk=True) if quant_qk else scores_ref
            )
            if self.score_f16:
                # The S-side error also includes the storage rounding (no probe — already counted in
                # the main run).
                scores_value = round_scores_f16(scores_value)
            stat.s_err2 += float((scores_value - scores_ref).square().sum())
            stat.s_ref2 += float(scores_ref.square().sum())
            stat.o_err2 += float((output - reference).square().sum())
            stat.o_ref2 += float(reference.square().sum())
            stat.measured += 1
        return output


def count_attention_nodes(model: nn.Module) -> list[str]:
    """The attention module names inside the DiT (the source of the expected call count)."""
    return [
        name
        for name, module in model.named_modules()
        if hasattr(module, "to_q") and hasattr(module, "to_out")
    ]


# ---- Metrics -----------------------------------------------------------------


def rel_rms(value: torch.Tensor, reference: torch.Tensor) -> float:
    """‖v − r‖₂ / ‖r‖₂."""
    denom = float(reference.square().sum())
    if denom == 0.0:
        return 0.0
    return math.sqrt(float((value - reference).square().sum()) / denom)


def psnr(value: torch.Tensor, reference: torch.Tensor, data_range: float) -> float:
    mse = float((value - reference).square().mean())
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(data_range * data_range / mse)


def to_uint8(image: torch.Tensor) -> torch.Tensor:
    """[1,3,H,W] in [-1,1] to [H,W,3] uint8 (the common path of the PNG and the visual PSNR)."""
    scaled = (image[0].permute(1, 2, 0).clamp(-1.0, 1.0) + 1.0) * 127.5
    return scaled.round().clamp(0.0, 255.0).to(torch.uint8)


# ---- Stages ------------------------------------------------------------------


@dataclass
class DitRun:
    """The result of `run_dit` (raw data and diagnostics across configurations)."""

    latents: dict[str, torch.Tensor]
    #: Per-layer statistics of the eligible linears of (c) w8a8 (the source of table ③).
    linear_stats: dict[str, LayerStat]
    #: Configuration name → statistics per attention node.
    attn_stats: dict[str, dict[str, AttnStat]]
    #: Configuration name → diagnostics such as call counts.
    diagnostics: dict[str, dict[str, object]]
    attention_nodes: list[str]


def run_dit(args: argparse.Namespace) -> DitRun:
    """Run the DiT of every configuration in order on the same instance, and return the latents per
    step and the diagnostics."""
    sigmas = sigma_schedule(args.steps, SHIFT)
    print(f"[sigmas] {sigmas[0]:.4f} … {sigmas[-2]:.4f} → 0 ({args.steps} steps)", flush=True)

    text = encode_text(args.repo, args.max_sequence_length, "f32")
    embeds = text["encoder_hidden_states"]
    print(f"[text] encoder_hidden_states={list(embeds.shape)}", flush=True)

    latent = args.resolution // SPATIAL_COMPRESSION
    latents = torch.randn(
        (1, LATENT_CHANNELS, 1, latent, latent),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float32,
    )

    _proj, model = timesteps_proj_table(args.repo, sigmas, "f32", args.lora, args.lora_scale)
    eligible = sum(1 for _, m in model.named_modules() if is_eligible(m))
    linears = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    print(f"[eligible] of {linears} linears, {eligible} have k%4==0", flush=True)

    attention_nodes = count_attention_nodes(model)
    print(f"[attention] {len(attention_nodes)} nodes / step", flush=True)

    out: dict[str, torch.Tensor] = {}
    attn_stats: dict[str, dict[str, AttnStat]] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    linear_stats: dict[str, LayerStat] = {}

    weights_i8 = False
    for config in CONFIGS:
        if config.weight_i8:
            report = fake_quant_int8(model)
            print(f"[fake-quant] transformer (i8): {report.describe()}", flush=True)
            weights_i8 = True

        # Per-layer statistics are collected only for the baseline configuration (c) (the others
        # have no reason to pay the measurement cost).
        hooks: ActQuantHooks | None = None
        handles: list[object] = []
        attached = 0
        if config.linear_act and config.name == BASELINE:
            hooks = ActQuantHooks(model, measure_step=args.measure_step)
            hooks.attach()
            attached = len(hooks.stats)
        elif config.linear_act:
            handles, attached = attach_act_quant(model)

        sim: AttnSim | None = None
        if config.attn or config.score_f16 or config.measure_pt:
            sim = AttnSim(
                model=model,
                mode=config.attn,
                stats_quantized=config.stats_quantized,
                score_f16=config.score_f16,
                measure_pt=config.measure_pt,
                measure_step=args.measure_step,
                disable=(args.inject == "drop-attn-quant"),
            )
            sim.attach()

        started = time.perf_counter()
        result = reference_steps(
            model,
            latents,
            embeds,
            embeds,
            sigmas,
            # `reference_steps` takes (width, height) since the move to the Anima family (c84dd07)
            # — an int fails the unpacking with a TypeError (resolved in known-issues 2026-08-19;
            # assumes a square).
            (args.resolution, args.resolution),
            args.steps,
            GUIDANCE,
        )
        for key, value in result.items():
            if key.startswith("latents_"):
                out[f"{config.name}/{key}"] = value
        elapsed = time.perf_counter() - started

        if hooks is not None:
            hooks.detach()
            linear_stats = hooks.stats
        detach_act_quant(handles)
        calls = skipped = 0
        s16 = ScoreF16Probe()
        if sim is not None:
            sim.detach()
            attn_stats[config.name] = sim.stats
            calls, skipped = sim.calls, sim.skipped
            s16 = sim.s16
        # The replacement is the only global mutable state that crosses configurations. Check with
        # `is` **after every configuration** that the plain implementation is back (a leftover
        # would silently quantize the next configuration and contaminate the text/VAE stages too).
        if F.scaled_dot_product_attention is not PRISTINE_SDPA:
            raise AssertionError(f"{config.name}: the SDPA replacement is still in place")

        diagnostics[config.name] = {
            "weight_i8": weights_i8,
            "linear_act_i8": config.linear_act,
            "linear_act_sites": attached,
            "attn": config.attn,
            "stats_quantized": config.stats_quantized,
            "score_f16": config.score_f16,
            "s16_rounded_calls": s16.calls,
            "s16_fired_calls": s16.fired,
            "s16_elements": s16.elements,
            "s16_abs_max": s16.abs_max,
            "attn_calls": calls,
            "attn_calls_expected": len(attention_nodes) * args.steps if sim is not None else 0,
            "attn_nodes_seen": len(attn_stats.get(config.name, {})),
            "attn_skipped_calls": skipped,
            "elapsed": round(elapsed, 1),
        }
        s16_note = f" s16={s16.fired}/{s16.calls} fired" if config.score_f16 else ""
        print(
            f"[{config.name}] {elapsed:.1f}s linear-act={attached} attn={calls} calls{s16_note}",
            flush=True,
        )

    del model
    gc.collect()
    return DitRun(out, linear_stats, attn_stats, diagnostics, attention_nodes)


def decode_all(
    repo: str, latents: dict[str, torch.Tensor], last: str, tags: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Decode the final latents of every configuration with **the same f32 VAE** (the VAE is loaded
    once).

    The configuration names are taken as an argument because `--w4-screen` has its own table
    (reading the module globals of the 10 configurations from here would not allow passing a
    different set per mode).
    """
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(repo, subfolder="vae")
    vae.to(torch.float32).eval()
    images: dict[str, torch.Tensor] = {}
    for tag in tags:
        started = time.perf_counter()
        z = denormalize_latents(vae, latents[f"{tag}/{last}"])
        with torch.no_grad():
            images[tag] = vae.decode(z, return_dict=False)[0][:, :, 0].contiguous()
        print(f"[decode] {tag} ({time.perf_counter() - started:.1f}s)", flush=True)
    del vae
    gc.collect()
    return images


def save_images(images: dict[str, torch.Tensor], out: Path) -> str:
    """Save as PNG (falls back to npy without PIL, and returns a note saying so)."""
    arrays = {tag: to_uint8(image) for tag, image in images.items()}
    try:
        from PIL import Image
    except ImportError:
        import numpy as np

        for tag, array in arrays.items():
            np.save(out / f"image_{tag}.npy", array.numpy())
        return "saved as image_*.npy (uint8 [H,W,3]) because PIL is unavailable"
    for tag, array in arrays.items():
        Image.fromarray(array.numpy()).save(out / f"image_{tag}.png")
    return "image_*.png"


# ---- w4 method screening (`--w4-screen` — a separate mode) ----------------------
#
# MUST: a **separate path** from the 10 configurations above. Those stack (a)→(b)→… in place on the
# same instance (activation i8 on top of weight i8), whereas this is **a comparison between
# methods**, so each configuration first restores the pristine weights (the plain f32 weights).
# Applying without restoring would measure "MXFP4 on top of NF4", and the method comparison itself
# would not hold (the same discipline as the EG / MiniCPM5 scripts). Labels, baselines and output
# file names are not shared with the table above either.
#
# The four methods brought in are the winners that survived the screening of two cheap families
# (ADR 0069 addendum 5):
#
#     rtn-i4-g32     the storage of the distributed `i4` itself (the baseline of the comparison)
#     nf4            a grid of normal-distribution quantiles (fp4 was dropped: dominated by nf4
#                    at the same cost)
#     mxfp4          FP4 table × a power-of-two scale per group (the 4.25 bpw side)
#     kmeans:shared  one 16-centroid table shared across layers (per_tensor collapses,
#                    per_channel is dominated)
#
# Every rounding implementation is shared from core (`karume.quantize` / `karume.quant_methods`),
# and so is the target selection, `iter_quant_targets` — a copied implementation would silently
# split "the targets and formula measured" from "the targets and formula shipped". The scope is the
# same as the 10 configurations, **the DiT only** (text_encoder / text_conditioner / VAE stay f32 in
# every configuration).

#: The group length of the method comparison = **fixed at 32** (the default settled in ADR 0069
#: addendum 1 = core's default). The g axis is not swept at the same time because, with the method
#: difference and the g difference mixed, one could not say "which one had the effect"
#: (ADR 0069 addendum 5 ③ / user ruling 2026-08-19).
W4_GROUP_SIZE = DEFAULT_GROUP_SIZE

#: The module types to round. **The only quantizable type present in the DiT is `nn.Linear`**
#: (patchify is `CosmosPatchEmbed.proj` = `nn.Linear` too, and there is no embedding table and no
#: conv), so the targets match the i4 execution path (linear only, ADR 0069 decision 5) = every
#: configuration is **a distributable form**. If a non-linear appears, {@link scan_w4_targets} fails
#: loudly.
W4_OP_TYPES: tuple[type[nn.Module], ...] = (nn.Linear,)

#: The spelling of the baseline configuration (no rounding — the denominator of every relRMS /
#: PSNR).
W4_BASE_CONFIG = "f32"

#: The outputs of `--w4-screen` (named differently from `report.md` / `attn.json` of the 10
#: configurations — so that they are not mixed up when written to the same `--out`).
W4_REPORT_FILE = "w4_report.md"
W4_JSON_FILE = "w4_screen.json"
W4_LATENTS_FILE = "w4_latents.safetensors"

#: The sample budget for **fitting the table** of `kmeans:shared`. shared concatenates the
#: normalised values of every target into one vector before running Lloyd's algorithm, so the f32
#: concatenation + the f64 workspace + the int64 assignment indices need ~28B per sample element.
#: The DiT has 1.96 billion elements; a full fit needs 7.3GiB for the concatenation alone and around
#: 50GiB with the workspace = it does not fit in physical memory (31GB). 20 million elements fit in
#: ~0.6GiB, which is more than enough by orders of magnitude for a 1-D k-means with 16 centroids
#: (application is always to everything — core's `fit_stride`).
W4_KMEANS_FIT_BUDGET = 20_000_000

# Bit widths of the size estimate (verbatim from the storage rules). The sources are `int4-sym-g` in
# the codec ledger of `docs/container-v1.md` §6.3 (the scale is **F32**, one per group) and OCP
# Microscaling Formats v1.0 (the shared scale of MX is E8M0 = a one-byte exponent). k-means is a
# measurement-only method without a storage form, so it is counted **including the table cost**.

#: The payload of the 4-bit grid (common to every method — what is compared is "how the grid is
#: laid out", not the bit count).
W4_PAYLOAD_BITS = 4.0
#: The bit width of the group scale (MUST: `i4` storage uses an F32 group scale —
#: `int4-sym-g` of `docs/container-v1.md` §6.3).
W4_F32_SCALE_BITS = 32.0
#: The shared scale of MXFP4 is E8M0 (a one-byte exponent).
W4_MX_SCALE_BITS = 8.0
#: The bit width of one codebook entry (centroids are held in F32).
W4_CODEBOOK_ENTRY_BITS = 32.0

W4_F32_BITS = 32.0
BITS_PER_BYTE = 8
MIB = 1024 * 1024


@dataclass(frozen=True)
class W4Targets:
    """The profile of the w4 target set (the input of the size estimate, and the ledger of
    exclusions).

    `groups` is the total number of g32 groups, counted from the flat form of
    {@link karume.quantize.channel_rows} (= the axis along which the rounding splits groups) —
    counting on another axis would make only the estimate a number for a different shape.
    """

    modules: int
    elements: int
    groups: int
    #: Weights **excluded from the targets** because the quantization axis is not divisible by
    #: `W4_GROUP_SIZE` (FQN → quantization axis length).
    excluded: Mapping[str, int]

    def include(self, name: str) -> bool:
        """The `include` predicate of `fake_quant_*` (called with **the module FQN**).

        The FQN is spelled the same way as in {@link karume.quantize.iter_quant_targets} (spelling
        only one side differently would silently keep a module meant to be excluded among the
        targets).
        """
        return (f"{name}.weight" if name else "weight") not in self.excluded


def scan_w4_targets(model: nn.Module) -> W4Targets:
    """Take the profile of the target set and the list of exclusions whose quantization axis is not
    divisible by g32, in one pass.

    MUST: fail loudly if a non-linear quantizable type (conv types / embedding) appears. This script
    builds its table on the premise "targets = linear only = matches the i4 execution path =
    distributable form", and if upstream diffusers switched patchify to a conv the premise would
    break silently (the table would still come out, with numbers for a form that cannot ship). When
    that happens, first decide **whether to add a column that includes the non-linears**.

    MUST: exclude a non-divisible target through `include` and list the exclusions in the output
    (ADR 0069 decision 2 — a partial group cannot be stored, so neither its bpw nor its quality is
    reported). Do not silently give up on the whole set.
    """
    wide = {fqn for fqn, _weight, _axis in iter_quant_targets(model, QUANT_MODULE_TYPES)}
    targets = list(iter_quant_targets(model, W4_OP_TYPES))
    extra = sorted(wide - {fqn for fqn, _weight, _axis in targets})
    if extra:
        raise AssertionError(
            f"DiT に非 linear の量子化可能な重みがある: {extra}"
            "（この台本は linear 限定 = i4 の実行経路と一致する形を測る前提 — "
            "非 linear 込みの列を足すかどうかを先に決めること）"
        )
    modules = elements = groups = 0
    excluded: dict[str, int] = {}
    for fqn, weight, axis in targets:
        rows = channel_rows(weight, axis)
        span = int(rows.shape[1])
        if span % W4_GROUP_SIZE:
            excluded[fqn] = span
            continue
        modules += 1
        elements += int(weight.numel())
        groups += int(rows.shape[0]) * (span // W4_GROUP_SIZE)
    if not modules:
        raise AssertionError(
            f"g{W4_GROUP_SIZE} で量子化できる重みが 1 本も無い（除外 {sorted(excluded)}）"
        )
    return W4Targets(modules=modules, elements=elements, groups=groups, excluded=excluded)


def w4_fit_stride(elements: int) -> int:
    """Decide the even stride of `fake_quant_kmeans(fit_stride=…)` from the number of target
    elements.

    If it fits within the budget ({@link W4_KMEANS_FIT_BUDGET}), 1 = a full fit. It **rounds up to
    an odd number** because the group length is a power of two — an even stride only visits the even
    lanes of a group (`gcd > 1`), whereas an odd one is always coprime and the samples visit every
    lane of a group equally.
    """
    stride = -(-elements // W4_KMEANS_FIT_BUDGET)
    return stride + 1 if stride % 2 == 0 else stride


class W4RoundReport(Protocol):
    """Only **the surface this script reads** of the return value of `fake_quant_*` (the types
    differ between i4 and the measurement-only methods)."""

    @property
    def modules(self) -> int: ...

    def describe(self) -> str: ...


@dataclass(frozen=True)
class W4Method:
    """The recipe of one method (binds how the rounding is applied and the size-estimate formula in
    one row).

    Spreading the two over separate tables would silently split "the methods whose quality was
    measured" from "the methods whose size was estimated" (adding a method would update only one of
    them). `apply` assumes it is applied **to the plain f32 weights**; the prohibition on stacking
    is guaranteed by the caller's {@link restore_w4}.
    """

    name: str
    #: `(model, include predicate, kmeans fit_stride) -> rounding counts`.
    apply: Callable[[nn.Module, Callable[[str], bool], int], W4RoundReport]
    #: The projected bit count of the target set.
    bits: Callable[[W4Targets], float]
    #: The formula of `bits` (**carried into the output as is** — so that the premise of the
    #: projection can be traced from the table).
    formula: str


def _w4_group_scaled_bits(scale_bits: float) -> Callable[[W4Targets], float]:
    """The total bits of a method with a group scale (`4·N + scale_bits·G`)."""

    def bits(targets: W4Targets) -> float:
        return W4_PAYLOAD_BITS * targets.elements + scale_bits * targets.groups

    return bits


W4_METHODS: tuple[W4Method, ...] = (
    W4Method(
        "rtn-i4-g32",
        lambda model, include, _stride: fake_quant_int4(
            model, W4_GROUP_SIZE, include=include, op_types=W4_OP_TYPES
        ),
        _w4_group_scaled_bits(W4_F32_SCALE_BITS),
        "4·N + 32·G (G = one F32 scale per group of N/32 — the storage of the distributed `i4`"
        " itself)",
    ),
    W4Method(
        "nf4",
        lambda model, include, _stride: fake_quant_nf4(
            model, W4_GROUP_SIZE, include=include, op_types=W4_OP_TYPES
        ),
        _w4_group_scaled_bits(W4_F32_SCALE_BITS),
        "4·N + 32·G (the same group absmax scale as i4; the grid is normal-distribution quantiles)",
    ),
    W4Method(
        "mxfp4",
        lambda model, include, _stride: fake_quant_mxfp4(
            model, W4_GROUP_SIZE, include=include, op_types=W4_OP_TYPES
        ),
        _w4_group_scaled_bits(W4_MX_SCALE_BITS),
        "4·N + 8·G (the shared scale is a power of two in E8M0 — OCP MX v1.0)",
    ),
    W4Method(
        "kmeans:shared",
        lambda model, include, stride: fake_quant_kmeans(
            model,
            "shared",
            W4_GROUP_SIZE,
            include=include,
            op_types=W4_OP_TYPES,
            fit_stride=stride,
        ),
        lambda targets: (
            W4_PAYLOAD_BITS * targets.elements
            + W4_F32_SCALE_BITS * targets.groups
            + W4_CODEBOOK_ENTRY_BITS * DEFAULT_CODEBOOK_LEVELS
        ),
        "4·N + 32·G + 32·16 (one table for the whole model, with a group absmax scale)",
    ),
)

W4_METHOD_NAMES = tuple(method.name for method in W4_METHODS)


def select_w4_methods(only: Sequence[str]) -> tuple[W4Method, ...]:
    """Return the methods chosen with `--w4-only` in declaration order (the baseline `f32` always
    runs — the denominator of the comparison)."""
    if not only:
        return W4_METHODS
    chosen = set(only)
    return tuple(method for method in W4_METHODS if method.name in chosen)


def w4_size_projection(method: W4Method, targets: W4Targets) -> dict[str, Any]:
    """The effective bpw and projected MiB of method × target set (the total of the target tensors
    only — not the whole model)."""
    bits = method.bits(targets)
    baseline_bits = W4_F32_BITS * targets.elements
    return {
        "modules": targets.modules,
        "elements": targets.elements,
        "groups": targets.groups,
        "bitsPerWeight": bits / targets.elements,
        "projectedMiB": bits / BITS_PER_BYTE / MIB,
        "f32MiB": baseline_bits / BITS_PER_BYTE / MIB,
        "ratio": bits / baseline_bits,
        "formula": method.formula,
    }


def restore_w4(weights: Mapping[str, torch.Tensor], pristine: Mapping[str, torch.Tensor]) -> None:
    """Restore every target to the pristine f32 weights (the implementation of the MUST **do not
    stack methods**)."""
    with torch.no_grad():
        for fqn, weight in weights.items():
            weight.copy_(pristine[fqn])


# ---- Calibrated rounding (the calibrated columns of `--w4-screen`, phase J-2) --------------
#
# The four methods above round by looking at the weights only (the RTN family). **Calibrated
# rounding** (GPTQ — core's `karume.quant_calib`) re-chooses the rounding targets **within the same
# grid** from "the activations that actually flow into that layer", and adds three storage grids to
# the same table. The prohibition on stacking and the pristine restore are the same as for the four
# methods.
#
# The targets are **only the `nn.Linear` of the DiT block list** ({@link CALIB_TARGET}), **a
# different set** from the targets of the four methods (the linears of the whole DiT) — the
# calibration driver (`calibrate_stages`) is closed over the `nn.Linear` inside a stage, so what is
# outside the blocks (`patch_embed.proj` / `norm_out` / `proj_out` / `time_embed`) is not included.
# The size estimate and the quality are read on that set too (the table and the JSON state the
# targets).
#
# The calibration inputs are taken **across the reference step sequence** (`--steps` of them, fixed
# PROMPT / fixed SEED). The activations of a diffusion model shift in distribution with sigma, so
# looking at one step only would leave the distributions of the later steps out of the calibration.

#: The target name of the calibrated configurations (the "targets" column of the table and the
#: JSON).
CALIB_TARGET = "dit:blocks-linear"

#: **The in-model FQN prefix** of the DiT block list (passed to the stages to align the keys of the
#: scale ledger with the FQN space of `Int4Report` — the contract of core's `StageSpec`).
CALIB_BLOCK_PREFIX = "transformer_blocks"


@dataclass(frozen=True)
class CalibConfig:
    """The specification of one calibrated rounding (method × storage grid).

    `bits` / `formula` are the same container as {@link W4Method} ({@link w4_size_projection}
    accepts both) — calibration **does not change the grid by a single byte**, so the formula of the
    storage form can be shared with the method side.
    """

    name: str
    method: CalibMethod
    grid: GridSpec
    bits: Callable[[W4Targets], float]
    formula: str


def _calib_codebook_bits(targets: W4Targets) -> float:
    """The total bits of `kmeans_shared` — **one table per layer** (the NOTE of core's
    `GridSpec`)."""
    return (
        W4_PAYLOAD_BITS * targets.elements
        + W4_F32_SCALE_BITS * targets.groups
        + W4_CODEBOOK_ENTRY_BITS * DEFAULT_CODEBOOK_LEVELS * targets.modules
    )


#: The three calibrated configurations (listed in the table in this order). AWQ is not included
#: because its equivalent scale `s` cannot be stored on its own (it needs a fold or a companion),
#: which would make a column without an exit (the MUST of core's `quant_calib`).
CALIB_CONFIGS: tuple[CalibConfig, ...] = (
    CalibConfig(
        "gptq-rtn",
        "gptq",
        GridSpec(kind="rtn", group_size=W4_GROUP_SIZE),
        _w4_group_scaled_bits(W4_F32_SCALE_BITS),
        "4·N + 32·G (the storage is the distributed `i4` itself — calibration changes only the"
        " rounding targets)",
    ),
    CalibConfig(
        "gptq-nf4",
        "gptq",
        GridSpec(kind="nf4", group_size=W4_GROUP_SIZE),
        _w4_group_scaled_bits(W4_F32_SCALE_BITS),
        "4·N + 32·G (the grid is NF4's fixed table)",
    ),
    CalibConfig(
        "gptq-kmeans",
        "gptq",
        GridSpec(kind="kmeans_shared", group_size=W4_GROUP_SIZE),
        _calib_codebook_bits,
        "4·N + 32·G + 32·16·(layers) (the table's scope is **per layer** — a different formula from"
        " kmeans:shared with one table overall)",
    ),
)

CALIB_NAMES: tuple[str, ...] = tuple(config.name for config in CALIB_CONFIGS)


def select_calib(only: Sequence[str]) -> tuple[CalibConfig, ...]:
    """Return the calibrated configurations chosen with `--w4-only` in declaration order (the same
    approach as {@link select_w4_methods})."""
    if not only:
        return CALIB_CONFIGS
    chosen = set(only)
    return tuple(config for config in CALIB_CONFIGS if config.name in chosen)


def calib_stages(model: nn.Module) -> tuple[StageSpec, ...]:
    """Return the DiT blocks in execution order as `(in-model FQN prefix, stage)`.

    A block can be a stage **as is** because `CosmosTransformerBlock.forward` has the shape "takes
    hidden positionally and the rest as keywords, returns hidden" (re-wrapping it could make the
    copy drift from upstream, so not wrapping is best). The prefix includes the block index, so the
    local FQN inside a stage (`attn1.to_q.weight`) maps back to the in-model FQN as is.
    """
    return tuple(
        (f"{CALIB_BLOCK_PREFIX}.{index}", block)
        for index, block in enumerate(model.transformer_blocks)
    )


def scan_calib_targets(stages: Sequence[StageSpec]) -> tuple[dict[str, torch.Tensor], W4Targets]:
    """Return the calibration targets inside the stages as weights keyed by fqn plus their profile
    (**the scan** = the reference of the exact-match check).

    MUST: fail loudly if a non-linear quantizable type appears (the same as {@link
    scan_w4_targets}).

    MUST: a non-g32-aligned target **fails loudly instead of being excluded** — calibration drives a
    whole stage, so leaving out one layer midway would make the check "number scanned = number
    rounded" impossible (the four methods are independent per layer, so listing exclusions is fine
    there).
    """
    weights: dict[str, torch.Tensor] = {}
    modules = elements = groups = 0
    for prefix, stage in stages:
        wide = {fqn for fqn, _weight, _axis in iter_quant_targets(stage, QUANT_MODULE_TYPES)}
        targets = list(iter_quant_targets(stage, W4_OP_TYPES))
        extra = sorted(wide - {fqn for fqn, _weight, _axis in targets})
        if extra:
            raise AssertionError(
                f"{prefix} に非 linear の量子化可能な重みがある: {extra}"
                "（校正は nn.Linear 限定 — 非 linear 込みの列を足すかどうかを先に決めること）"
            )
        for fqn, weight, axis in targets:
            rows = channel_rows(weight, axis)
            span = int(rows.shape[1])
            if span % W4_GROUP_SIZE:
                raise AssertionError(
                    f"{prefix}.{fqn}: 量子化軸 {span} が g{W4_GROUP_SIZE} で割り切れない"
                    "（校正は stage 単位で駆動するので 1 本だけ外す逃げ道が無い）"
                )
            weights[f"{prefix}.{fqn}"] = weight
            modules += 1
            elements += int(weight.numel())
            groups += int(rows.shape[0]) * (span // W4_GROUP_SIZE)
    if not modules:
        raise AssertionError(f"校正対象 '{CALIB_TARGET}' に量子化できる重みが 1 本も無い")
    return weights, W4Targets(modules=modules, elements=elements, groups=groups, excluded={})


def capture_stage_batches(
    model: nn.Module, run_reference: Callable[[], dict[str, torch.Tensor]], limit: int
) -> tuple[dict[str, torch.Tensor], tuple[StageBatch, ...]]:
    """Run one reference denoise pass while capturing `(args, kwargs)` to the first block per step.

    CFG=1 ({@link GUIDANCE}), so 1 step = 1 DiT forward = 1 batch. The accompanying arguments are
    built once by `CosmosTransformer3DModel.forward` **before** the block loop and the same ones are
    passed to every block, so they can stay invariant across stages (no `advance_kwargs` needed).

    The names of the positional arguments are taken **from the signature** of `block.forward` — so
    that the spelling does not silently shift if upstream reorders the arguments (no copied list of
    names is kept).

    MUST: the caller uses this only **before any rounding is applied** (the pass of the baseline
    `f32`).
    """
    blocks = list(model.transformer_blocks)
    names = tuple(inspect.signature(blocks[0].forward).parameters)
    batches: list[StageBatch] = []

    def pre(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if len(batches) >= limit:
            return
        batches.append(
            (
                (args[0].detach(),),
                {**dict(zip(names[1:], args[1:], strict=False)), **kwargs},
            )
        )

    handle = blocks[0].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        result = run_reference()
    finally:
        handle.remove()
    if not batches:
        raise AssertionError(
            "校正入力を 1 step も捕まえられなかった"
            "（DiT の block ループが台本の想定と食い違っている）"
        )
    return result, tuple(batches)


@dataclass(frozen=True)
class CalibRig:
    """The scaffolding the calibrated configurations share (the stage list, the scan, the inputs to
    the first stage).

    It is not rebuilt per configuration — the capture happens once, in the pass of the baseline
    `f32` (**before any rounding is applied**).
    """

    stages: tuple[StageSpec, ...]
    scan: Mapping[str, torch.Tensor]
    targets: W4Targets
    batches: tuple[StageBatch, ...]


def apply_calib(config: CalibConfig, rig: CalibRig) -> CalibReport:
    """Apply one calibrated rounding to the DiT in place (the per-stage driver is on the core
    side)."""
    report = calibrate_stages(rig.stages, rig.batches, method=config.method, spec=config.grid)
    assert_calib_covers_scan(report, rig.scan, config.name)
    return report


def assert_calib_covers_scan(
    report: CalibReport, scan: Mapping[str, torch.Tensor], name: str
) -> None:
    """Check that the layers the calibration rounded match the stage scan **exactly**.

    MUST: fail loudly. Even if a change in the stage spelling or the target type left part of a
    block out of the calibration, the table would keep its rows, and the PSNR would come out on
    **the good side** by the amount left unrounded (a passthrough cannot be read from the numbers).
    """
    rounded = {layer.fqn for layer in report.layers}
    missing = sorted(set(scan) - rounded)
    extra = sorted(rounded - set(scan))
    if missing or extra or report.modules != len(scan):
        raise AssertionError(
            f"[{name}] 校正が丸めた {report.modules} 本が走査の {len(scan)} 本と一致しない"
            f"（丸め漏れ {missing[:3]} / 走査に無い {extra[:3]}）"
        )


def run_w4_screen(args: argparse.Namespace) -> dict[str, Any]:
    """Re-round from pristine per method, run the DiT, and collect the quality and the size
    estimate.

    The model is loaded only once (7.29GiB). Only the target weights are saved as pristine (the
    excluded ones are never touched, so they are not saved), and the peak together with the rounding
    workspace decides the physical memory.

    The calibrated configurations ({@link CALIB_CONFIGS}) run **after** the four methods (the target
    sets differ, so they do not mix in the table order). The scaffolding is built only once, in the
    pass of the baseline `f32` — the calibration inputs have to be taken **before any rounding is
    applied**.
    """
    methods = select_w4_methods(args.w4_only)
    calib_configs = select_calib(args.w4_only)
    sigmas = sigma_schedule(args.steps, SHIFT)
    print(f"[sigmas] {sigmas[0]:.4f} … {sigmas[-2]:.4f} → 0 ({args.steps} steps)", flush=True)

    text = encode_text(args.repo, args.max_sequence_length, "f32")
    embeds = text["encoder_hidden_states"]
    print(f"[text] encoder_hidden_states={list(embeds.shape)}", flush=True)

    latent = args.resolution // SPATIAL_COMPRESSION
    latents_init = torch.randn(
        (1, LATENT_CHANNELS, 1, latent, latent),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float32,
    )

    _proj, model = timesteps_proj_table(args.repo, sigmas, "f32", args.lora, args.lora_scale)
    targets = scan_w4_targets(model)
    stride = w4_fit_stride(targets.elements)
    print(
        f"[w4] {targets.modules} targets / {targets.elements:,} elements"
        f" / {targets.groups:,} groups (g{W4_GROUP_SIZE})",
        flush=True,
    )
    for fqn, span in sorted(targets.excluded.items()):
        print(
            f"[w4] excluded {fqn}: quantization axis {span} is not divisible by {W4_GROUP_SIZE}",
            flush=True,
        )
    if stride > 1:
        print(
            f"[w4] the kmeans:shared table is fitted on an even 1/{stride} subsample"
            f" (≈ {targets.elements // stride:,} elements; applied to everything)",
            flush=True,
        )

    weights = {
        fqn: weight
        for fqn, weight, _axis in iter_quant_targets(model, W4_OP_TYPES)
        if fqn not in targets.excluded
    }
    pristine = {fqn: weight.detach().clone() for fqn, weight in weights.items()}

    rig: CalibRig | None = None
    if calib_configs:
        stages = calib_stages(model)
        calib_scan, calib_targets = scan_calib_targets(stages)
        print(
            f"[calib] {len(stages)} stages / {calib_targets.modules} target linears"
            f" / {calib_targets.elements:,} elements ({CALIB_TARGET} — nothing outside the blocks)",
            flush=True,
        )

    # `model` is **bound through a default argument** because the `del` below removes the local of
    # the same name (a plain closure would leave a form that raises NameError if called after it is
    # gone).
    def run_reference(model: nn.Module = model) -> dict[str, torch.Tensor]:
        return reference_steps(
            model,
            latents_init,
            embeds,
            embeds,
            sigmas,
            (args.resolution, args.resolution),
            args.steps,
            GUIDANCE,
        )

    latents: dict[str, torch.Tensor] = {}
    configs: dict[str, dict[str, Any]] = {}
    sizes: dict[str, dict[str, Any]] = {}
    for method in (None, *methods):
        name = W4_BASE_CONFIG if method is None else method.name
        started = time.perf_counter()
        if method is None:
            report = "no rounding (baseline)"
        else:
            # MUST: always restore before applying (do not stack methods — the start of the
            # section).
            restore_w4(weights, pristine)
            rounded = method.apply(model, targets.include, stride)
            if rounded.modules != targets.modules:
                raise AssertionError(
                    f"{name}: rounded {rounded.modules} modules, not the {targets.modules} targets"
                    " (the include predicate or op_types has no effect)"
                )
            report = rounded.describe()
            sizes[name] = w4_size_projection(method, targets)
        if method is None and calib_configs:
            # MUST: the capture happens once only, **before any rounding is applied** (the pass of
            # the baseline f32).
            result, batches = capture_stage_batches(model, run_reference, args.steps)
            rig = CalibRig(stages=stages, scan=calib_scan, targets=calib_targets, batches=batches)
            print(f"[calib] captured {len(batches)} calibration batches (across steps)", flush=True)
        else:
            result = run_reference()
        for key, value in result.items():
            if key.startswith("latents_"):
                latents[f"{name}/{key}"] = value
        elapsed = time.perf_counter() - started
        configs[name] = {"method": name, "quantReport": report, "elapsed": round(elapsed, 1)}
        print(f"[{name}] {report} ({elapsed:.1f}s)", flush=True)

    for config in calib_configs:
        if rig is None:
            raise AssertionError(
                "the calibration scaffolding was not built"
                " (the pass of the baseline f32 did not run)"
            )
        started = time.perf_counter()
        # MUST: calibrated configurations do not stack either — restore pristine before applying
        # (the same as the four methods).
        restore_w4(weights, pristine)
        calib_report = apply_calib(config, rig)
        result = run_reference()
        for key, value in result.items():
            if key.startswith("latents_"):
                latents[f"{config.name}/{key}"] = value
        elapsed = time.perf_counter() - started
        configs[config.name] = {
            "method": f"{calib_report.method}/{calib_report.grid}",
            "target": CALIB_TARGET,
            "quantReport": calib_report.describe(),
            "calibModules": calib_report.modules,
            "calibBatches": len(rig.batches),
            "elapsed": round(elapsed, 1),
        }
        sizes[config.name] = w4_size_projection(config, rig.targets)
        print(f"[{config.name}] {calib_report.describe()} ({elapsed:.1f}s)", flush=True)

    calib_meta = (
        None
        if rig is None
        else {
            "name": CALIB_TARGET,
            "stages": len(rig.stages),
            "modules": rig.targets.modules,
            "elements": rig.targets.elements,
            "groups": rig.targets.groups,
            "batches": len(rig.batches),
        }
    )
    # Release the rounded weights and pristine (the same size as the targets) before the VAE decode.
    # `rig` still holds the blocks through the stages and the calibration inputs through batches, so
    # it goes too.
    del model, weights, pristine, rig
    if calib_configs:
        del stages, calib_scan, calib_targets
    gc.collect()

    last = f"latents_step{args.steps:04d}"
    tags = tuple(configs)
    images = decode_all(args.repo, latents, last, tags)
    keys = [f"latents_step{index:04d}" for index in range(1, args.steps + 1)]
    failures: list[str] = []
    base_image = images[W4_BASE_CONFIG]
    u8_base = to_uint8(base_image).to(torch.float32)
    # The baseline row itself is not measured (comparing with itself gives relRMS 0 / PSNR ∞, which
    # carries no information, and ∞ is not a valid JSON value either). The table and the gate look
    # only at the rounded configurations.
    for name, entry in ((name, configs[name]) for name in configs if name != W4_BASE_CONFIG):
        u8 = to_uint8(images[name]).to(torch.float32)
        entry["latentRelRms"] = [
            rel_rms(latents[f"{name}/{key}"], latents[f"{W4_BASE_CONFIG}/{key}"]) for key in keys
        ]
        entry["psnrF32"] = psnr(images[name], base_image, 2.0)
        entry["psnrUint8"] = psnr(u8, u8_base, 255.0)
        entry["imageRelRms"] = rel_rms(images[name], base_image)
        entry["uint8MaxDiff"] = float((u8 - u8_base).abs().max())
        entry["moved"] = not torch.equal(
            latents[f"{name}/{last}"], latents[f"{W4_BASE_CONFIG}/{last}"]
        )
        entry["size"] = sizes[name]
        # Guarding against a vacuous pass: a passthrough is always a lie on the "good quality" side
        # (only when no rounding was applied at all does it match the baseline bit for bit, and then
        # the numbers look perfect).
        if not entry["moved"]:
            failures.append(
                f"{name}: the final latent matches `{W4_BASE_CONFIG}` bit for bit (passthrough)"
            )

    image_note = save_images(images, args.out)
    if not args.no_latents:
        from safetensors.torch import save_file

        save_file(latents, str(args.out / W4_LATENTS_FILE))
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/anima/measure_quant.py --w4-screen",
        "torch": torch.__version__,
        "repo": args.repo,
        "prompt": PROMPT,
        "steps": args.steps,
        "resolution": args.resolution,
        "guidanceScale": GUIDANCE,
        "seed": SEED,
        "lora": str(args.lora),
        "groupSize": W4_GROUP_SIZE,
        "codebookLevels": DEFAULT_CODEBOOK_LEVELS,
        "kmeansFitStride": stride,
        "kmeansFitSamplesApprox": targets.elements // stride,
        "targets": {
            "opTypes": [cls.__name__ for cls in W4_OP_TYPES],
            "modules": targets.modules,
            "elements": targets.elements,
            "groups": targets.groups,
        },
        # The targets of the calibrated columns are **a different set** from the four methods above
        # (nothing outside the blocks) — their profile is listed alongside so that the bpw and the
        # quality of the table are not mixed up when read.
        "calibTargets": calib_meta,
        "excluded": [
            {"weight": fqn, "quantAxis": span} for fqn, span in sorted(targets.excluded.items())
        ],
        "imageNote": image_note,
        "configs": configs,
        "gates": {
            "moved": {
                name: entry["moved"] for name, entry in configs.items() if name != W4_BASE_CONFIG
            },
            "failures": failures,
        },
    }


def _w4_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def w4_report_markdown(payload: Mapping[str, Any]) -> str:
    """The body of `w4_report.md` (built **from the payload only** — so that the table and the JSON
    do not diverge)."""
    configs: dict[str, dict[str, Any]] = payload["configs"]
    names = list(configs)
    quantized = [name for name in names if name != W4_BASE_CONFIG]
    lines: list[str] = []
    lines.append(
        f"# w4 method screening — {len(quantized)} rounding methods for the Anima DiT"
        " (measured with torch on CPU)"
    )
    lines.append("")
    lines.append(
        f"- Date: {payload['generated']} / measured by: `{payload['script']}`"
        f" (torch {payload['torch']}, CPU f32)"
    )
    lines.append(
        f"- Conditions: Anima turbo {payload['steps']} steps, {payload['resolution']}px"
        f", CFG={payload['guidanceScale']:g}, LoRA merged, seed {payload['seed']}"
    )
    lines.append(f"- Prompt: `{payload['prompt']}`")
    lines.append(
        f"- Targets: `{'` / `'.join(payload['targets']['opTypes'])}` of the DiT,"
        f" {payload['targets']['modules']} modules / {payload['targets']['elements']:,} elements"
        f" (g={payload['groupSize']} fixed, {payload['targets']['groups']:,} groups)."
        " **Everything outside the DiT (text_encoder / text_conditioner / VAE) stays f32 in every"
        " configuration**."
    )
    lines.append(
        "- MUST: each configuration first restores the pristine weights (the plain f32 weights)"
        " — **methods are not stacked**. A **separate path** from the stacking semantics of the 10"
        " configurations of the same script (w8 / w8a8 / attention a8)."
    )
    lines.append(
        "- Every rounding is core's shared implementation (`karume.quantize.fake_quant_int4` / "
        "nf4 / mxfp4 / kmeans of `karume.quant_methods`), and so is the target selection, "
        "`iter_quant_targets` (so that the targets measured and the targets shipped do not split)."
    )
    excluded = payload["excluded"]
    if excluded:
        lines.append(
            "- 除外（量子化軸が g"
            f"{payload['groupSize']} で割り切れないので対象から外した — ADR 0069 決定 2）: "
            + " / ".join(f"`{item['weight']}`（軸 {item['quantAxis']}）" for item in excluded)
        )
    calib = payload.get("calibTargets")
    if calib:
        lines.append(
            f"- **校正付き構成**（`{'` / `'.join(CALIB_NAMES)}`）の対象は上とは別集合 —"
            f" `{calib['name']}` = DiT block 列の `nn.Linear` {calib['modules']} 本 /"
            f" {calib['elements']:,} 要素（block の外の `patch_embed` / `norm_out` /"
            " `proj_out` / `time_embed` は入らない）。校正入力は参照 step 列を横断した"
            f" {calib['batches']} バッチ（stage {calib['stages']} 段）。**bpw も品質も"
            "この集合で読む**（4 方式の行と直接は比較できない）。"
        )
    if payload["kmeansFitStride"] > 1:
        lines.append(
            f"- `kmeans:shared` の**表の fit だけ** 1/{payload['kmeansFitStride']} の等間隔"
            f"部分標本（≈ {payload['kmeansFitSamplesApprox']:,} 要素）で採った — **適用は全量**。"
            "全量 fit は作業領域が実メモリに載らないため（部分標本の表と全量の表は別物に"
            "なりうるので、数値を読む側が区別できるようここに明記する）。"
        )
    lines.append("")

    lines.append(f"## ① Latent relRMS per step (baseline = `{W4_BASE_CONFIG}`)")
    lines.append("")
    lines.append("`relRMS(v, r) = ‖v − r‖₂ / ‖r‖₂`.")
    lines.append("")
    rows = [
        [str(index + 1), *(f"{configs[name]['latentRelRms'][index]:.4e}" for name in quantized)]
        for index in range(payload["steps"])
    ]
    lines.append(_w4_table(["step", *(f"`{name}`" for name in quantized)], rows))
    lines.append("")

    lines.append(f"## ② Final image after VAE decode (baseline = `{W4_BASE_CONFIG}`)")
    lines.append("")
    lines.append(
        "PSNR is given two ways: on the raw `[-1,1]` tensor (data range 2.0) and after the same"
        " uint8 quantization as the PNG (data range 255). relRMS is on the raw tensor."
    )
    lines.append("")
    rows = [
        [
            f"`{name}`",
            f"{configs[name]['psnrF32']:.2f}",
            f"{configs[name]['psnrUint8']:.2f}",
            f"{configs[name]['imageRelRms']:.4e}",
            f"{configs[name]['uint8MaxDiff']:.0f}",
            f"{configs[name]['elapsed']:.0f}s",
        ]
        for name in quantized
    ]
    lines.append(
        _w4_table(
            [
                "configuration",
                "PSNR f32 (dB)",
                "PSNR uint8 (dB)",
                "relRMS",
                "uint8 max diff (/255)",
                "elapsed",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        f"Images: {payload['imageNote']} ({' / '.join(f'`image_{name}`' for name in names)})"
    )
    lines.append("")

    lines.append("## ③ Size estimate (target tensor set only, **projected by formula**)")
    lines.append("")
    lines.append(
        f"N = {payload['targets']['elements']:,} target elements / G = N/{payload['groupSize']} = "
        f"{payload['targets']['groups']:,}. These come from the storage-rule formulas, not from"
        " measurement, and a method without a storage form (anything but rtn) cannot even be"
        " written — they are laid out in the same table as the quality so that nothing is chosen on"
        " quality alone while the table cost stays invisible."
    )
    lines.append("")
    rows = [
        [
            f"`{name}`",
            f"{configs[name]['size']['bitsPerWeight']:.3f}",
            f"{configs[name]['size']['projectedMiB']:.1f}",
            f"{configs[name]['size']['f32MiB']:.1f}",
            f"{configs[name]['size']['ratio'] * 100:.1f}%",
            configs[name]["size"]["formula"],
        ]
        for name in quantized
    ]
    lines.append(
        _w4_table(["configuration", "bpw", "projected MiB", "f32 MiB", "ratio", "formula"], rows)
    )
    lines.append("")

    lines.append("## ④ Gate")
    lines.append("")
    lines.append(
        "The rounding was not passed through (each configuration's final latent **does not match**"
        " the baseline bit for bit) — a match happens only when no rounding was applied at all, and"
        ' then the quality comes out on the "perfect" side.'
    )
    lines.append("")
    rows = [
        [
            f"`{name}`",
            "OK" if configs[name]["moved"] else "**NG（素通り）**",
            configs[name]["quantReport"],
        ]
        for name in quantized
    ]
    lines.append(_w4_table(["configuration", "differs from baseline", "rounding counts"], rows))
    lines.append("")
    return "\n".join(lines) + "\n"


def main_w4(args: argparse.Namespace) -> None:
    """The entry point of `--w4-screen` (none of the 10 configurations runs)."""
    args.out.mkdir(parents=True, exist_ok=True)
    payload = run_w4_screen(args)
    (args.out / W4_REPORT_FILE).write_text(w4_report_markdown(payload), encoding="utf-8")
    (args.out / W4_JSON_FILE).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print(f"report OK → {args.out / W4_REPORT_FILE}", flush=True)
    failures = payload["gates"]["failures"]
    if failures:
        raise AssertionError("verification gate is red: " + " / ".join(failures))
    print("gates: all green", flush=True)


def main() -> None:
    # Because --until replaces the derived tables too with the forward part (the declaration has to
    # come before any use, as the grammar requires).
    global CONFIGS, CONFIG_NAMES, ATTN_CONFIGS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--out", type=Path, required=True, help="output directory for the report and images"
    )
    parser.add_argument("--steps", type=int, default=10, help="number of turbo steps")
    parser.add_argument("--resolution", type=int, default=512, help="side of the square (px)")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--lora", type=Path, default=DEFAULT_LORA, help="LoRA to merge into the DiT"
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--measure-step",
        type=int,
        default=1,
        help="step at which the per-layer output relRMS is collected (0-based)",
    )
    parser.add_argument(
        "--inject",
        choices=("drop-attn-quant",),
        default=None,
        help="fault injection (demonstrates the detectors' power) — installs the attention SDPA"
        " replacement but applies neither quantization nor the S f16 rounding (measures that"
        " (d)–(j) collapse toward the baseline configuration and that the rounding call count of"
        " `*-s16` drops to 0)",
    )
    parser.add_argument(
        "--no-latents",
        action="store_true",
        help="do not write dit_out.safetensors (the raw latents of configuration × step)",
    )
    parser.add_argument(
        "--until",
        choices=CONFIG_NAMES,
        default=None,
        help="run only the **forward part** up to this configuration (`w8a8` is enough to check"
        " the P̃ distribution). The i8 rounding of the weights is in place and order-dependent, so"
        " an arbitrary subset is not allowed — forward prefixes only."
        f" A cutoff before the baseline `{BASELINE}` is rejected because the summary would not"
        " hold",
    )
    parser.add_argument(
        "--w4-screen",
        action="store_true",
        help="run the w4 rounding method screening **instead of the 10 configurations**"
        f" ({' / '.join(W4_METHOD_NAMES)} + calibrated {' / '.join(CALIB_NAMES)}"
        f" + baseline f32, g{W4_GROUP_SIZE} fixed)."
        " A separate path that restores the plain f32 weights before each configuration, with"
        " differently named outputs",
    )
    parser.add_argument(
        "--w4-only",
        action="append",
        default=[],
        choices=(*W4_METHOD_NAMES, *CALIB_NAMES),
        help="with `--w4-screen`, run only this method / calibrated configuration (repeatable; for"
        " partial re-runs). The baseline f32 always runs",
    )
    args = parser.parse_args()
    if args.w4_only and not args.w4_screen:
        parser.error("--w4-only requires --w4-screen")
    if args.w4_screen:
        # The knobs of the stacking path (the forward prefix and the attention fault injection)
        # mean nothing to the method screening — reject them instead of silently ignoring them.
        if args.until is not None or args.inject is not None:
            parser.error("--w4-screen cannot be combined with --until / --inject (separate path)")
        main_w4(args)
        return
    if args.until is not None:
        cutoff = CONFIG_NAMES.index(args.until)
        if cutoff < CONFIG_NAMES.index(BASELINE):
            parser.error(
                f"--until accepts only configurations from {BASELINE} on (no summary without the"
                " baseline)"
            )
        # Replace the derived tables too with the forward part (every later stage reads the same
        # source of truth).
        CONFIGS = CONFIGS[: cutoff + 1]
        CONFIG_NAMES = tuple(config.name for config in CONFIGS)
        ATTN_CONFIGS = attn_config_names(CONFIGS)
    args.out.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    run = run_dit(args)
    latents, stats = run.latents, run.linear_stats
    if not args.no_latents:
        save_file(latents, str(args.out / "dit_out.safetensors"))
    write_layer_csv(stats, args.out / "layers.csv")

    images = decode_all(args.repo, latents, f"latents_step{args.steps:04d}", CONFIG_NAMES)
    image_note = save_images(images, args.out)
    # The derived tables --until rewrote and the engine-side metrics are **all passed as
    # arguments** (importing this module from the report side would be circular, so the reverse
    # edge is kept to type annotations only).
    summary = build_summary(
        args,
        run,
        latents,
        images,
        config_names=CONFIG_NAMES,
        attn_configs=ATTN_CONFIGS,
        baseline=BASELINE,
        pt_hist_bins=PT_HIST_BINS,
        rel_rms=rel_rms,
        psnr=psnr,
        hist_quantile=hist_quantile,
    )
    report = build_report(
        args,
        run,
        latents,
        images,
        image_note,
        summary,
        config_names=CONFIG_NAMES,
        attn_configs=ATTN_CONFIGS,
        baseline=BASELINE,
        guidance=GUIDANCE,
        pt_hist_bins=PT_HIST_BINS,
        rel_rms=rel_rms,
        psnr=psnr,
        to_uint8=to_uint8,
        hist_quantile=hist_quantile,
    )
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "attn.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "steps": args.steps,
                "resolution": args.resolution,
                "guidance_scale": GUIDANCE,
                "seed": SEED,
                "lora": str(args.lora),
                "torch": torch.__version__,
                "eligible_linears": len(stats),
                "attention_nodes": len(run.attention_nodes),
                "configs": list(CONFIG_NAMES),
                "inject": args.inject,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    for key, value in summary["gates"].items():
        print(f"[gate] {key}: {value}", flush=True)
    print(f"report OK → {args.out / 'report.md'}", flush=True)
    # This script fills quality gate ① of ADR 0019, so red also shows in the exit code (the same
    # strength as the `--w4-screen` path). `--inject` demonstrates detection power, and **turning it
    # red is the very purpose**, so it does not raise.
    if args.inject is None and summary["failed"]:
        raise AssertionError("品質ゲートが赤: " + " / ".join(summary["failed"]))


if __name__ == "__main__":
    main()
