r"""Numeric source of truth for the **host-side algorithm** of Irodori-TTS v4 (full-loop latent
golden).

Where `irodori/export.py` emits the graphs and `irodori/tokenizer_ref.py` the text-preprocessing
assets, this emits the **final values** of "running the 7 graphs until a latent is produced". It
is the reference for the integration check that W3's host implementation (the TS pipeline)
compares against, and the values are taken **on the exported graphs** (= the Graph wrappers of
`irodori/export.py` = the eager-equivalent implementation).

    cd tools/export-recipes
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype f16
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype i8
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype i4

`--dtype f16` / `--dtype i8` / `--dtype i4` write to **separate series**
(`outputs/series/irodori-v4-small-{f16,i8,i4}/pipeline/`). Rounding applies
`irodori.export.fake_quant` right after load, so both the golden and the upstream reference are
computed **with the same rounded weights** (ADR 0018 / 0019 / 0027 / 0050 — the golden is rebaked
per series).

Only `--dtype i4` (the distribution's quant preset `i8+dit4`) rounds in two stages: stage 1 is
**the same fake-quant as the i8 preset** for every role, and stage 2 **reads back the two DiT
containers (`dit` / `dit-context`) exported with `--dtype i4`** and overwrites the parameters
the wrappers own ({@link restore_dit_from_i4_series}). This series therefore assumes **the
export has already been run**.

Output (default `outputs/series/irodori-v4-small{,-f16,-i8,-i4}/pipeline/`, under `.gitignore`):

    meta.json                input text / caption / S / seed / CFG scale / forward count /
                             t schedule / sources, and the measured values of the standing checks
    case.<name>.safetensors  reference latent, initial noise, token ids, final z
    t-embed.safetensors      the t actually used over 40 steps and its host-formula `t_embed`
                             table `[40,512]`

## The stages the host owns (the only thing this file copies)

1. preprocessing → tokenize → prepend BOS (the body is cut at `max_*_len - 1`). Preprocessing
   differs by kind: text is `normalize_text` → strip, caption is **strip only** (the spelling of
   upstream `_synthesize`)
2. `backbone` → `text-proj` / `caption-proj` (called **with the sequence kept packed** = the
   static scheme)
3. patch the reference latent by `speaker_patch_size` → `speaker` → prepend the mean token
4. build the 5 inputs of `duration` (`speaker_vec` = the mean token / `caption_vec` = the masked
   mean of **the second output of `caption-proj`**)
5. decide S: `expm1(log_frames)` → banker's rounding → clamp to `[min_frames, max_frames]`
6. right-pad the condition states to Tmax and build the segment mask (ADR 0047) → run
   `dit-context` **only once** to build the condition-side K/V (ADR 0114 — every later forward
   shares these values)
7. Euler 40 steps + independent CFG (only at steps with `t ∈ [0.5, 1.0]`; uncond **only sets the
   mask of the affected segment to all False** — ADR 0047 decision 1)

`t_embed` is generated on the host (`cos` is not in the IR vocabulary — ADR 0043). **The formula
is not copied; upstream's `get_timestep_embedding` is called.** For the same reason the mean
token prefix, the latent patching and the caption's masked mean also call upstream functions /
methods. Only 5 and 7 are copied (upstream buries them inside `_synthesize` /
`sample_euler_rf_cfg`, so they cannot be called as functions), and those two are guarded by
**comparison against upstream's canonical path** (the checks below).

## What the checks are

Everything is measured before emit, and if any one fails **nothing is written** (ADR 0005's fail
loudly):

- **S is decided as upstream decides it** (S is computed independently from upstream
  `predict_duration_log_frames`)
- **the final z matches upstream's `sample_euler_rf_cfg`** (a two-stage judgement).
  The graph path differs from upstream's implementation in three points — "the condition KV is
  built once from the cond state (upstream builds a cache per variant in a batch), the backbone
  is called on the packed sequence, uncond is reduced to masking" — so it is **not bit-identical**.
  The first stage is the fixed threshold
  {@link EULER_REFERENCE_ATOL}. How the error accumulates and is amplified moves by orders of
  magnitude with model × input × rounding (37,107× measured for v4.1 f16 — the addendum of
  2026-09-01 in the constant's comment), so when it is exceeded the amplification factor is
  measured and the result is judged normalized by {@link euler_reference_within_sensitivity}. If
  it still fails, the CFG combination formula, the t schedule or the segment split of the mask
  differs
- CFG **actually has an effect** (the difference from the z run with cond only — blocks the check
  from becoming vacuous)
- the initial noise is **bit-identical** to upstream's (same seed, same generation order)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from _shared.calib_provenance import calib_complaint
from _shared.container_read import StoredTensor, read_stored, weight_tensors
from karume.container import container_parts
from karume.convert import normalize_boundary_tensor
from karume.emit import unpack_int4
from karume.quantize import dequantize_int4

from . import export as ex
from . import patch
from .distribution import (
    CALIB_PROVENANCE_FILE,
    CALIB_SHIPPABLE_METHOD,
    IRODORI_FLOW_MEANFLOW,
    IRODORI_FLOW_RF,
    irodori_calib_floor,
)

META_FILE = "meta.json"
CASE_PREFIX = "case."
CASE_SUFFIX = ".safetensors"
T_EMBED_FILE = "t-embed.safetensors"

#: Number of Euler steps (the default of upstream `SamplingRequest.num_steps`).
NUM_STEPS = 40

#: Number of steps of a **MeanFlow** checkpoint (ADR 0150 — upstream's default for one:
#: `inference_runtime` takes `4 if is_meanflow else 40` when the request names no step count, and
#: `meanflow.sample_euler_meanflow` defaults to 4 as well).
MEANFLOW_NUM_STEPS = 4


def reference_steps(flow: str) -> int:
    """The step count of the reference loop for a flow parameterization (the RF Euler loop or
    the MeanFlow loop)."""
    if flow == IRODORI_FLOW_MEANFLOW:
        return MEANFLOW_NUM_STEPS
    if flow == IRODORI_FLOW_RF:
        return NUM_STEPS
    raise SystemExit(f"Unknown flow parameterization {flow!r}")


#: Initial factor of the t schedule (`init_scale` of `rf.sample_euler_rf_cfg`).
INIT_SCALE = 0.999

#: The t range where CFG applies (the defaults of upstream `SamplingRequest.cfg_min_t` /
#: `cfg_max_t`).
CFG_MIN_T, CFG_MAX_T = 0.5, 1.0

#: CFG strength (the defaults of `cfg_scale_text` / `cfg_scale_speaker` / `cfg_scale_caption`
#: there). The order is that of ADR 0047's uncond variants (`DIT_UNCOND_VARIANTS`).
CFG_SCALES: Mapping[str, float] = {"text": 3.0, "speaker": 5.0, "caption": 3.0}

#: Seconds that set the clamp range of S (the defaults of upstream
#: `SamplingRequest.min_seconds` / `max_seconds`). Conversion to frames uses `CODEC_FRAME_RATE`
#: (= sample_rate / hop_length).
MIN_SECONDS, MAX_SECONDS = 0.5, 30.0

#: The largest absolute difference at which the final z counts as matching upstream
#: `sample_euler_rf_cfg`.
#:
#: Measured (the standing check below prints it on every run; 40 steps): **full = 1.03e-4 /
#: no-ref = 1.80e-5**, with z's range topping out at \|z\| 5.10 / 4.33. The threshold 1e-3 is
#: about 9.7× the measured worst and 1/5000 of the range. All three sources of the difference are
#: **structural**: ① the graph path builds the condition KV once from the cond state (upstream
#: builds a cache per variant in the cond / uncond batch) ② the backbone is called on the packed
#: sequence (upstream pads to 256 —
#: `_static_scheme_evidence` measured it in the 1e-5 range) ③ uncond is expressed by masking
#: (upstream uses state 0 + mask 0). Each is mathematically an identity; what remains is only the
#: difference in f32 reduction order. 40 Euler steps amplify that to this order (1.4e-5 / 4.5e-6
#: measured for 1 step), while **a mistaken formula** (the CFG sign or scale, the t schedule, the
#: mask's segment split) shows up at O(1), the same as the range — indeed, CFG alone moves z by
#: 5.56 / 3.23 (`cfgEffectMaxAbs`).
#:
#: **The compressed series (`--dtype f16` / `i8` / `i4`) use the same threshold**: rounding hits
#: **both** the graph path and the upstream path exactly once and identically (upstream
#: `sample_euler_rf_cfg` uses the `dit` that `fake_quant` rounded as is; the i4 read-back
#: overwrites the Parameters of that same `dit`, so likewise), so the difference measured here is
#: "the implementation difference only", as in the f32 series, and the quantization error cancels
#: on both sides.
#: The order of magnitude can change only when rounding lands in an ill-conditioned region, and
#: that is **settled by measurement** (split the threshold per series only after a measurement
#: actually fails — loosening it in advance would hide the failure).
#:
#: Addendum (2026-09-01): v4.1-small f16 actually failed (full = 1.33e-2). Settled by measurement:
#: "the seed of the implementation difference is at the usual level, and this input made the
#: 40-step iteration accumulate and amplify it 37,107×" (v4-small f16 is 353× —
#: docs/research/2026-09-01-irodori-v41-euler-sensitivity.md).
#: The accumulated value moves by orders of magnitude with model × input × rounding, so this
#: constant was demoted to **the first stage (fast path)**; when it is exceeded, the second stage
#: measures the amplification factor and normalizes by it to decide pass or fail
#: ({@link euler_reference_within_sensitivity} below).
EULER_REFERENCE_ATOL = 1e-3

#: Second stage: the ceiling of the implementation difference normalized by the amplification
#: factor. Measured (2026-09-01, 10 cells of v4 / v4.1 × f32 / f16 / i8), worst / amp falls in
#: 1.6e-8 to 2.2e-6 — this is about twice that upper end.
EULER_NOISE_PER_AMP = 5e-6

#: Second stage: an absolute ceiling applied regardless of the amplification factor. A mistaken
#: formula (the CFG sign or scale, the t schedule, the mask segments) shows up at O(1)
#: (`cfgEffectMaxAbs` measures 3 to 5), so this stops far below that.
EULER_REFERENCE_ABS_CEILING = 5e-2

#: Perturbation size for measuring the amplification factor. Large enough relative to the seed
#: of the implementation difference (f32 reduction-order differences ~1e-7) for the linear
#: response to be readable.
SENSITIVITY_EPS = 1e-6


def euler_reference_within_sensitivity(worst: float, amp: float) -> bool:
    """Second-stage pass / fail — whether the implementation difference normalized by the
    amplification factor `amp` is at the seed level, and within the absolute ceiling.

    `worst` is the largest absolute difference of the final z from upstream, `amp` is the factor
    by which a tiny perturbation of the initial noise reaches the final z (the
    `sensitivity_probe` of {@link run_case}). Called only when the first stage
    ({@link EULER_REFERENCE_ATOL}) is exceeded.
    """
    return worst <= amp * EULER_NOISE_PER_AMP and worst <= EULER_REFERENCE_ABS_CEILING


#: The lower bound for the interval of a MeanFlow loop actually having an effect (the largest
#: absolute difference from the z run with every interval set to 0).
#: MUST: not vacuous — the TS side builds `delta_embed` on the host, and if the trained branch
#: barely moved the output, a host that dropped or misplaced it would still pass the comparison
#: with this golden (the counterpart of {@link CFG_EFFECT_MIN} for a loop without CFG).
DELTA_EFFECT_MIN = 1e-2

#: The lower bound for CFG actually having an effect (the largest absolute difference from the
#: z run with cond only).
#: MUST: not vacuous — setting the scale to 0, or getting the uncond mask wrong so that it equals
#: cond, would still pass the comparison with upstream **as long as both make the same mistake**
#: (upstream is run on the same inputs).
CFG_EFFECT_MIN = 1e-2


class ReferenceSpec(NamedTuple):
    """Source of the reference latent (**kept synthetic** — wave 4 made real-speech latents
    available [`irodori/dacvae/host.py`], but replacing this golden would move every compared
    value, so the existing one is kept. The real-speech range is covered by the real-latent cases
    of the standalone speaker check [`irodori.export.SPEAKER_REAL_CASES`]).

    `frames` is the frame count before patching and must be divisible by `speaker_patch_size`
    (upstream `patch_sequence_with_mask` drops a remainder, so the golden's input and the sequence
    actually used would diverge).
    """

    frames: int
    seed: int


class PipelineCase(NamedTuple):
    """One full-loop case. `reference` of `None` means the no-reference (`no_ref`) path."""

    name: str
    why: str
    text: str
    caption: str
    reference: ReferenceSpec | None
    seed: int


#: The two full-loop cases.
#:
#: - `full` — real text + reference latent + caption. All three CFGs are active (4 forwards per
#:   step)
#: - `no-ref` — no reference (the host puts zeros for speaker, speaker CFG off) + empty caption
#:   (caption CFG off, mask all 0). 2 forwards per step
PIPELINE_CASES: tuple[PipelineCase, ...] = (
    PipelineCase(
        name="full",
        why="with reference + caption (3 CFGs, 4 forwards per step)",
        text="今日は近くの店まで歩いて行きました。とても良い天気でしたね。",
        caption="若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
        reference=ReferenceSpec(frames=124, seed=201),
        seed=1234,
    ),
    PipelineCase(
        name="no-ref",
        why="no reference (speaker zero short-circuit, CFG off) + empty caption (mask 0, CFG off)",
        text="本日はお越しいただき、誠にありがとうございます。",
        caption="",
        reference=None,
        seed=1235,
    ),
)


def t_schedule(num_steps: int) -> torch.Tensor:
    """The t sequence `[num_steps+1]` (exactly the linear schedule of upstream
    `sample_euler_rf_cfg`).

    MUST: built with the same formula as upstream (`(1 - linspace(0,1,n+1)) * 0.999`). The TS
    side is expected to use the closed form `0.999 * (1 - i/n)`, so
    {@link closed_form_matches} measures on every run **whether the two are bit-identical** (if
    not, the TS formula is adjusted).
    """
    return (1.0 - torch.linspace(0.0, 1.0, num_steps + 1)) * INIT_SCALE


def meanflow_schedule(num_steps: int) -> torch.Tensor:
    """The t sequence `[num_steps+1]` of a MeanFlow loop — **exactly** upstream's
    `sample_euler_meanflow` (`torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)`).

    Each step uses `delta = t − t_next` and the update `x + v·(t_next − t)`, both in f32 tensor
    arithmetic. The TS side reproduces these f32 values for any step count (ADR 0150 — torch's f32
    `linspace` is not the closed form `1 − i/n`), so the golden records the sequence it used.
    """
    return torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)


def closed_form_matches(schedule: torch.Tensor, num_steps: int) -> float:
    """Largest absolute difference between the closed form `0.999 * (1 - i/n)` and the upstream
    schedule (0 means bit-identical).

    Measured: **5.96e-8** at `num_steps=40` (not bit-identical — `linspace` accumulates
    `start + i*step` in f32, so at some points the last ulp splits from the closed form's
    `1 - i/n`). t enters only `t_embed` and the step size `t_next - t`, and the effect of this
    difference on the final z is three orders below {@link EULER_REFERENCE_ATOL}. **The value is
    recorded in meta** — input for the host implementation's choice of formula.
    """
    closed = torch.tensor(
        [INIT_SCALE * (1.0 - index / num_steps) for index in range(num_steps + 1)],
        dtype=schedule.dtype,
    )
    return float((closed - schedule).abs().max())


def banker_round_frames(frames: float) -> int:
    """`int(round(x))` — Python's **banker's rounding** (`inference_runtime._synthesize`).

    MUST: do not read it as "round half up". Exactly 0.5 rounds to the even side, so using
    `Math.round` (always up) on the TS side is off by one frame. The upstream spelling is
    `int(round(x))`; `round` already returns an int, so `int(...)` is copied only to keep the
    shape.
    """
    return round(frames)


def sequence_length(log_frames: torch.Tensor, frame_rate: int) -> tuple[int, dict[str, Any]]:
    """Decide the latent length S from duration's output (log frames).

    The same as the 5 lines of upstream `inference_runtime._synthesize` (`expm1` → mean → times
    `duration_scale` → banker's rounding → clamp to `[min_frames, max_frames]`).
    `duration_scale` defaults to 1.0. `latent_patch_size` is 1, so the patched length = the
    latent length (the value is checked against the config).
    """
    frames = float(torch.expm1(log_frames).float().mean())
    min_frames = max(1, math.ceil(MIN_SECONDS * frame_rate))
    max_frames = max(1, math.floor(MAX_SECONDS * frame_rate))
    steps = max(min_frames, min(max_frames, banker_round_frames(frames)))
    return steps, {
        "predictedFrames": round(frames, 4),
        "minFrames": min_frames,
        "maxFrames": max_frames,
    }


def _pack_body(tokenizer: Any, body: str, bos_id: int, max_length: int) -> torch.Tensor:
    """Preprocessed body → a **packed** `[1,T]` token sequence (BOS + the body cut at
    `max_length-1`).

    Upstream `PretrainedTextTokenizer.batch_encode` returns the same sequence right-padded. In
    the static scheme the host removes the padding before the call (the module docstring of
    `irodori/export.py`), so this builds the unpadded sequence.
    """
    ids = list(tokenizer.encode(body, add_special_tokens=False).ids)[: max_length - 1]
    return torch.tensor([[bos_id, *ids]], dtype=torch.int64)


def _packed_ids(
    tokenizer: Any, text: str, bos_id: int, max_length: int, normalize_text: Any
) -> torch.Tensor:
    """The **text-side** token sequence (`normalize_text` → `strip` → packed sequence)."""
    normalized = normalize_text(text).strip()
    if not normalized:
        raise SystemExit("the body is empty after normalization")
    return _pack_body(tokenizer, normalized, bos_id, max_length)


def _packed_caption_ids(tokenizer: Any, caption: str, bos_id: int, max_length: int) -> torch.Tensor:
    """The **caption-side** token sequence (`strip` only → packed sequence).

    MUST: `normalize_text` is not applied. Upstream `inference_runtime._synthesize` applies only
    `str(...).strip()` to the caption (`normalize_text` is text-only); adding normalization would
    silently change the conditioning by the stripping of outer brackets, NFKC and symbol
    removal.
    """
    stripped = caption.strip()
    if not stripped:
        raise SystemExit("the caption is empty after strip")
    return _pack_body(tokenizer, stripped, bos_id, max_length)


class HostGraphs(NamedTuple):
    """The 7 graphs the host runs (exactly the eager-equivalent wrappers of
    `irodori/export.py`)."""

    backbone: nn.Module
    text_proj: nn.Module
    caption_proj: nn.Module
    speaker: nn.Module
    duration: nn.Module
    #: Condition-side K/V projection (once per generation — ADR 0114). Its outputs are exactly
    #: `dit`'s inputs from the 4th on.
    dit_context: nn.Module
    dit: nn.Module


def _segment_masks(
    length: int, used: Mapping[str, int], caps: Mapping[str, int], uncond: str | None
) -> torch.Tensor:
    """The concatenated mask `[1,1,1,S+1519]` (in the order self / text / speaker / caption).

    Only the segment named by `uncond` is set to all False = the uncond of ADR 0047 decision 1.
    """
    segments = [torch.ones((1, length), dtype=torch.bool)]
    for name in ex.DIT_UNCOND_VARIANTS:
        segments.append(ex._segment_mask(caps[name], 0 if uncond == name else used[name]))
    return torch.cat(segments, dim=1)[:, None, None, :]


def _euler(
    graphs: HostGraphs,
    source: ex.IrodoriSource,
    noise: torch.Tensor,
    context: Sequence[torch.Tensor],
    used: Mapping[str, int],
    caps: Mapping[str, int],
    enabled: Sequence[str],
    embed_dim: int,
    schedule: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Euler 40 steps + independent CFG. Returns `(final z, forward count)`.

    Only 3 points of upstream `rf.sample_euler_rf_cfg` are copied (everything else calls upstream
    functions): `v = v_cond + Σ scale_k (v_cond - v_k)` / `x += v (t_next - t)` / the CFG range
    `cfg_min_t ≤ t ≤ cfg_max_t`. The comparison uses {@link EULER_REFERENCE_ATOL}.

    `context` is the output of `dit-context` (the condition-side K/V, built once by
    {@link run_case}), and cond and uncond use the same values (uncond is expressed by the mask
    alone — ADR 0047 decision 1). The forward count is the number of `dit` runs; the single
    `dit-context` run is not counted (the same as the TS side's count).
    """
    x_t = noise
    masks = {
        name: _segment_masks(int(noise.shape[1]), used, caps, name) for name in (None, *enabled)
    }
    forwards = 0
    for index in range(len(schedule) - 1):
        t, t_next = schedule[index], schedule[index + 1]
        with torch.no_grad():
            t_embed = source.timestep_embedding(t.reshape(1), embed_dim).to(x_t.dtype)
            v_cond = graphs.dit(x_t, t_embed, masks[None], *context)
            forwards += 1
            v = v_cond
            if enabled and CFG_MIN_T <= float(t) <= CFG_MAX_T:
                for name in enabled:
                    v_uncond = graphs.dit(x_t, t_embed, masks[name], *context)
                    forwards += 1
                    v = v + CFG_SCALES[name] * (v_cond - v_uncond)
            x_t = x_t + v * (t_next - t)
    return x_t, forwards


def _meanflow(
    graphs: HostGraphs,
    source: ex.IrodoriSource,
    noise: torch.Tensor,
    context: Sequence[torch.Tensor],
    used: Mapping[str, int],
    caps: Mapping[str, int],
    embed_dim: int,
    schedule: torch.Tensor,
    *,
    zero_interval: bool = False,
) -> tuple[torch.Tensor, int]:
    """The MeanFlow loop (ADR 0150). Returns `(final z, forward count)`.

    The same 3 points of upstream `meanflow.sample_euler_meanflow` are copied as in the RF loop
    ({@link _euler}) — the interval `delta = t − t_next`, one **conditional** forward per step (no
    CFG: guidance was fused into the teacher during distillation) and `x += v (t_next − t)` — and
    both embeddings call upstream's `get_timestep_embedding`. The comparison with upstream's own
    loop is {@link EULER_REFERENCE_ATOL} (the same two-stage judgement as the RF loop).

    `zero_interval` runs the same loop with every interval embedded as 0 — only for the
    non-vacuity measurement ({@link DELTA_EFFECT_MIN}).
    """
    x_t = noise
    mask = _segment_masks(int(noise.shape[1]), used, caps, None)
    forwards = 0
    for index in range(len(schedule) - 1):
        t, t_next = schedule[index], schedule[index + 1]
        delta = t - t_next
        if zero_interval:
            delta = torch.zeros_like(delta)
        with torch.no_grad():
            t_embed = source.timestep_embedding(t.reshape(1), embed_dim).to(x_t.dtype)
            delta_embed = source.timestep_embedding(delta.reshape(1), embed_dim).to(x_t.dtype)
            velocity = graphs.dit(x_t, t_embed, delta_embed, mask, *context)
            forwards += 1
            x_t = x_t + velocity * (t_next - t)
    return x_t, forwards


def run_case(
    case: PipelineCase,
    graphs: HostGraphs,
    source: ex.IrodoriSource,
    duration_predictor: nn.Module,
    tokenizer: Any,
    text_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    config: Any,
    caps: Mapping[str, int],
    *,
    frames_override: int | None = None,
    sensitivity_probe: bool = False,
    flow: str = IRODORI_FLOW_RF,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run the host path of one case and return `(tensors to save, meta)`.

    `frames_override` is the injection point that **fixes S from outside** (default `None` = as
    duration predicts). The golden emit always calls it with `None` (S is part of the golden).
    It is used by `irodori/measure_quant.py`, because SNR / LSD are not defined unless the
    waveform length is the same across quantization configurations (the same reason
    `measure_quant_sbv2.py` fixes the time grid with the dump side's `w_ceil`). **The prediction
    itself** stays in `meta["predictedS"]` — S drift is not a quantity this script measures but a
    diagnostic the measuring side reads.

    `sensitivity_probe=True` runs the Euler loop once more (noise + {@link SENSITIVITY_EPS}) and
    puts the amplification factor in `meta["sensitivityAmp"]`. Only for emit's second stage
    ({@link euler_reference_within_sensitivity}) — not measured by default (it doubles the cost).

    `flow` is the checkpoint's flow parameterization: a MeanFlow checkpoint runs {@link _meanflow}
    instead of the Euler + CFG loop (ADR 0150), and its non-vacuity check is the interval's
    effect ({@link DELTA_EFFECT_MIN}) instead of CFG's.
    """
    bos_id = int(text_config["bos_token_id"])
    embed_dim = int(config.timestep_embed_dim)
    latent_dim = int(config.patched_latent_dim)
    if int(config.latent_patch_size) != 1:
        raise SystemExit(
            f"latent_patch_size is {config.latent_patch_size} — breaks the assumption"
            " S = latent length"
        )

    with torch.no_grad():
        # ---- ①–② text side (backbone → projector with the sequence kept packed) ----
        text_ids = _packed_ids(tokenizer, case.text, bos_id, caps["text"], source.normalize_text)
        text_state = graphs.text_proj(graphs.backbone(text_ids))
        caption_ids = None
        caption_length = 0
        caption_state = torch.zeros((1, 1, int(model_config["caption_dim"])))
        caption_pooled = None
        if case.caption:
            caption_ids = _packed_caption_ids(tokenizer, case.caption, bos_id, caps["caption"])
            caption_state, caption_normed = graphs.caption_proj(graphs.backbone(caption_ids))
            caption_length = int(caption_state.shape[1])
            caption_pooled = duration_predictor._caption_vec(
                batch_size=1,
                device=caption_normed.device,
                dtype=caption_normed.dtype,
                caption_state=caption_normed,
                caption_mask=torch.ones(caption_normed.shape[:2], dtype=torch.bool),
                has_caption=torch.ones((1,), dtype=torch.bool),
            )

        # ---- ③ speaker (with no reference the host puts zeros — `_no_reference_evidence` is the
        # evidence) ----
        reference_latent = torch.zeros((1, int(config.speaker_patch_size), int(config.latent_dim)))
        if case.reference is not None:
            generator = torch.Generator().manual_seed(case.reference.seed)
            reference_latent = torch.randn(
                1, case.reference.frames, int(config.latent_dim), generator=generator
            )
        reference_mask = torch.full(
            reference_latent.shape[:2], case.reference is not None, dtype=torch.bool
        )
        from irodori_tts.model import patch_sequence_with_mask

        patched, patched_mask = patch_sequence_with_mask(
            reference_latent, reference_mask, int(config.speaker_patch_size)
        )
        if case.reference is None:
            speaker_state = torch.zeros((1, int(patched.shape[1]) + 1, int(config.speaker_dim)))
        else:
            speaker_state, _mask = source.prepend_masked_mean_token(
                graphs.speaker(patched), patched_mask
            )

        # ---- ④ the 5 inputs of duration ----
        has_speaker = case.reference is not None
        speaker_vec = (
            speaker_state[:, 0] if has_speaker else torch.zeros((1, int(config.speaker_dim)))
        )
        caption_vec = (
            caption_pooled
            if caption_pooled is not None
            else torch.zeros((1, int(config.caption_dim_resolved)))
        )
        log_frames = graphs.duration(
            text_state,
            speaker_vec,
            torch.tensor([[has_speaker]], dtype=torch.bool),
            caption_vec,
            torch.tensor([[caption_pooled is not None]], dtype=torch.bool),
        )

    # ---- ⑤ decide S ----
    predicted, duration_meta = sequence_length(log_frames, ex.CODEC_FRAME_RATE)
    steps = predicted if frames_override is None else frames_override

    # ---- ⑥ right-pad the conditions and segment lengths ----
    # With an empty caption the state stays zero and the mask is all 0 (equivalent to upstream
    # `_synthesize`'s `caption_mask.zero_()` — the segment contributes exp(−inf)=0, exactly 0).
    used = {
        "text": int(text_state.shape[1]),
        "speaker": int(speaker_state.shape[1]) if has_speaker else 0,
        "caption": caption_length,
    }
    states = (
        ex._right_pad(text_state, caps["text"], f"the text condition of {case.name}"),
        ex._right_pad(speaker_state, caps["speaker"], f"the speaker condition of {case.name}"),
        ex._right_pad(caption_state, caps["caption"], f"the caption condition of {case.name}"),
    )
    # The condition-side K/V is built once per generation and shared by all the Euler runs below
    # (with / without CFG / the sensitivity perturbation run) — built at the same place and the
    # same number of times as the values the TS side passes as resident tensors (ADR 0114).
    with torch.no_grad():
        context = graphs.dit_context(*states)

    noise = torch.randn(
        (1, steps, latent_dim),
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(case.seed),
    )
    sensitivity_amp = None
    if flow == IRODORI_FLOW_MEANFLOW:
        # ---- ⑦ the MeanFlow loop (one conditional forward per step, no CFG — ADR 0150) ----
        enabled: list[str] = []
        schedule = meanflow_schedule(MEANFLOW_NUM_STEPS)
        latent, forwards = _meanflow(
            graphs, source, noise, context, used, caps, embed_dim, schedule
        )
        zero_interval, _forwards = _meanflow(
            graphs, source, noise, context, used, caps, embed_dim, schedule, zero_interval=True
        )
        effect = {"deltaEffectMaxAbs": float((latent - zero_interval).abs().max())}
        if effect["deltaEffectMaxAbs"] < DELTA_EFFECT_MIN:
            raise SystemExit(
                f"{case.name}: the difference with and without the interval is"
                f" {effect['deltaEffectMaxAbs']} — the interval embedding seems to have no effect"
            )
        if sensitivity_probe:
            perturbed, _fw = _meanflow(
                graphs, source, noise + SENSITIVITY_EPS, context, used, caps, embed_dim, schedule
            )
            sensitivity_amp = float((perturbed - latent).abs().max()) / SENSITIVITY_EPS
    else:
        # ---- ⑦ Euler + CFG ----
        # The active CFGs follow the conditions of upstream `resolve_cfg_scales` /
        # `sample_euler_rf_cfg`: text always / speaker only with a reference / caption only when
        # non-empty.
        enabled = ["text"]
        if has_speaker:
            enabled.append("speaker")
        if used["caption"] > 0:
            enabled.append("caption")
        schedule = t_schedule(NUM_STEPS)
        latent, forwards = _euler(
            graphs, source, noise, context, used, caps, enabled, embed_dim, schedule
        )
        cond_only, _forwards = _euler(
            graphs, source, noise, context, used, caps, (), embed_dim, schedule
        )
        effect = {"cfgEffectMaxAbs": float((latent - cond_only).abs().max())}
        if effect["cfgEffectMaxAbs"] < CFG_EFFECT_MIN:
            raise SystemExit(
                f"{case.name}: the difference with and without CFG is"
                f" {effect['cfgEffectMaxAbs']} — CFG seems to have no effect (scale 0 / the uncond"
                " mask equals cond)"
            )
        if sensitivity_probe:
            perturbed, _fw = _euler(
                graphs,
                source,
                noise + SENSITIVITY_EPS,
                context,
                used,
                caps,
                enabled,
                embed_dim,
                schedule,
            )
            sensitivity_amp = float((perturbed - latent).abs().max()) / SENSITIVITY_EPS

    # MUST: the id sequences are lowered to the IR's actual representation (i64 → i32) when saved
    # — the TS side reads them under the same boundary convention (ADR 0009). Here they stay int64,
    # and the upstream comparison is given these sequences too.
    tensors = {
        "noise": noise,
        "z": latent.detach().contiguous(),
        "reference_latent": reference_latent,
        "text_ids": text_ids,
    }
    if caption_ids is not None:
        tensors["caption_ids"] = caption_ids
    meta = {
        "why": case.why,
        "text": case.text,
        "caption": case.caption,
        "seed": case.seed,
        "S": steps,
        "duration": {**duration_meta, "logFrames": round(float(log_frames), 6)},
        "tokens": {"text": int(text_ids.shape[1]), "caption": used["caption"]},
        "segments": dict(used),
        # MUST: the RF meta keeps its keys and their order (the TS-side latent check reads them);
        # a MeanFlow loop has no CFG, so it carries no `cfg`.
        **(
            {}
            if flow == IRODORI_FLOW_MEANFLOW
            else {"cfg": {name: CFG_SCALES[name] for name in enabled}}
        ),
        "forwards": forwards,
        **{key: float(f"{value:.4e}") for key, value in effect.items()},
        **(
            {"sensitivityAmp": float(f"{sensitivity_amp:.4e}")}
            if sensitivity_amp is not None
            else {}
        ),
        "reference": (
            None
            if case.reference is None
            else {
                "kind": "synthetic-standard-normal",
                "why": "kept synthetic (for golden stability — the real-speech range is covered"
                " by SPEAKER_REAL_CASES of the standalone speaker check; see the ReferenceSpec"
                " docstring)",
                "frames": case.reference.frames,
                "seed": case.reference.seed,
                "patchedTokens": int(patched.shape[1]),
            }
        ),
        "zAbsMax": float(f"{float(latent.abs().max()):.6f}"),
    }
    if frames_override is not None:
        # MUST: added only when fixed (does not move the golden meta.json across series).
        meta["predictedS"] = predicted
    return tensors, meta


def _right_pad_ids(
    ids: torch.Tensor | None, length: int, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token sequence and mask right-padded to `[1,length]` (the final form of upstream
    `batch_encode`).

    `ids` of `None` gives all padding + an all-0 mask (the same form as upstream
    `caption_mask.zero_()`). The caption is built from upstream's entry point by
    {@link upstream_caption_condition}, so the only current caller is text.
    """
    padded = torch.full((1, length), pad_id, dtype=torch.int64)
    mask = torch.zeros((1, length), dtype=torch.bool)
    if ids is not None:
        used = int(ids.shape[1])
        padded[0, :used] = ids[0]
        mask[0, :used] = True
    return padded, mask


def upstream_caption_tokenizer(model_dir: Path, add_bos: bool) -> Any:
    """The **same** tokenizer upstream `_synthesize` uses for the caption
    (`caption_tokenizer`).

    `add_bos` comes from upstream `ModelConfig.caption_add_bos_resolved` (not hard-coded).
    """
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from transformers import AutoTokenizer

    source = str((model_dir / ex.TOKENIZER_FILE).parent)
    return PretrainedTextTokenizer(AutoTokenizer.from_pretrained(source), add_bos=add_bos)


def upstream_caption_condition(
    caption_tokenizer: Any, caption: str, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the caption's condition input `[1,max_caption_len]` (token sequence + mask) **from
    upstream's entry point**.

    MUST: do not pass the host-built `caption_ids` back into the upstream loop — doing so puts a
    preprocessing mistake equally on both sides, the comparison becomes vacuous, and z matches
    even when the caption spellings diverge. This calls upstream `inference_runtime._synthesize`'s
    caption path (`str(...).strip()` → `caption_tokenizer.batch_encode` → `caption_mask.zero_()`
    if empty) as is.
    """
    caption_text = str(caption).strip()
    ids, mask = caption_tokenizer.batch_encode([caption_text], max_length=max_length)
    if caption_text == "":
        mask.zero_()
    return ids, mask


def upstream_latent(
    model: nn.Module,
    case: PipelineCase,
    text_ids: torch.Tensor,
    caption_padded: torch.Tensor,
    caption_mask: torch.Tensor,
    reference_latent: torch.Tensor,
    steps: int,
    text_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    config: Any,
    *,
    flow: str = IRODORI_FLOW_RF,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the same conditions through upstream's canonical path (`rf.sample_euler_rf_cfg`, or
    `meanflow.sample_euler_meanflow` for a MeanFlow checkpoint — the sampler upstream's runtime
    picks from the same `flow_parameterization`). Returns `(z, initial noise)`.

    MUST: the inputs follow upstream's calling convention — the text token sequence is
    right-padded to `max_text_len` (restoring the padding the host removed in the static scheme),
    the caption takes the sequence {@link upstream_caption_condition} built at upstream's entry
    point as is, and no reference is the same zero latent + all-0 mask as `no_ref`.
    """
    from irodori_tts.rf import sample_euler_rf_cfg

    pad_id = int(text_config["pad_token_id"])
    text_padded, text_mask = _right_pad_ids(text_ids, int(model_config["max_text_len"]), pad_id)
    reference_mask = torch.full(
        reference_latent.shape[:2], case.reference is not None, dtype=torch.bool
    )
    noise = torch.randn(
        (1, steps, int(config.patched_latent_dim)),
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(case.seed),
    )
    if flow == IRODORI_FLOW_MEANFLOW:
        from irodori_tts.meanflow import sample_euler_meanflow

        # Upstream ignores every RF-only control for a MeanFlow checkpoint (CFG scales, mode and
        # bounds — docs/meanflow.md), so none is passed: one conditional forward per step.
        latent = sample_euler_meanflow(
            model=model,
            text_input_ids=text_padded,
            text_mask=text_mask,
            ref_latent=reference_latent,
            ref_mask=reference_mask,
            sequence_length=steps,
            caption_input_ids=caption_padded,
            caption_mask=caption_mask,
            num_steps=MEANFLOW_NUM_STEPS,
            seed=case.seed,
        )
        return latent, noise
    latent = sample_euler_rf_cfg(
        model=model,
        text_input_ids=text_padded,
        text_mask=text_mask,
        ref_latent=reference_latent,
        ref_mask=reference_mask,
        sequence_length=steps,
        caption_input_ids=caption_padded,
        caption_mask=caption_mask,
        num_steps=NUM_STEPS,
        cfg_scale_text=CFG_SCALES["text"],
        cfg_scale_caption=CFG_SCALES["caption"],
        cfg_scale_speaker=CFG_SCALES["speaker"] if case.reference is not None else 0.0,
        cfg_guidance_mode="independent",
        cfg_min_t=CFG_MIN_T,
        cfg_max_t=CFG_MAX_T,
        seed=case.seed,
    )
    return latent, noise


def upstream_sequence_length(
    model: nn.Module,
    text_ids: torch.Tensor,
    caption_padded: torch.Tensor,
    caption_mask: torch.Tensor,
    reference_latent: torch.Tensor,
    has_speaker: bool,
    text_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    config: Any,
) -> int:
    """Compute S independently from upstream's `encode_conditions` +
    `predict_duration_log_frames`."""
    reference_mask = torch.full(reference_latent.shape[:2], has_speaker, dtype=torch.bool)
    pad_id = int(text_config["pad_token_id"])
    text_padded, text_mask = _right_pad_ids(text_ids, int(model_config["max_text_len"]), pad_id)
    with torch.no_grad():
        (
            text_state,
            text_state_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_state_mask,
        ) = model.encode_conditions(
            text_input_ids=text_padded,
            text_mask=text_mask,
            ref_latent=reference_latent,
            ref_mask=reference_mask,
            caption_input_ids=caption_padded,
            caption_mask=caption_mask,
        )
        log_frames = model.predict_duration_log_frames(
            text_state=text_state,
            text_mask=text_state_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_state_mask,
            duration_features=torch.zeros((1, int(config.duration_aux_dim))),
            has_speaker=torch.tensor([has_speaker], dtype=torch.bool),
            has_caption=torch.tensor([bool(caption_mask.any())], dtype=torch.bool),
        )
    steps, _meta = sequence_length(log_frames, ex.CODEC_FRAME_RATE)
    return steps


def t_embed_table(source: ex.IrodoriSource, schedule: torch.Tensor, dim: int) -> torch.Tensor:
    """The `[40, dim]` `t_embed` table for the t **actually used** over 40 steps (for W3's host
    unit test).

    MUST: the formula is not copied; upstream `get_timestep_embedding` is called (if the θ split
    changed, a copied formula would silently stay at the old one).
    """
    return embedding_table(source, schedule[:-1], dim)


def embedding_table(source: ex.IrodoriSource, values: torch.Tensor, dim: int) -> torch.Tensor:
    """The `[len(values), dim]` table of upstream's `get_timestep_embedding`, one row per value
    (the `t` of each step, or a MeanFlow loop's intervals)."""
    with torch.no_grad():
        rows = [source.timestep_embedding(value.reshape(1), dim) for value in values]
    return torch.cat(rows, dim=0).contiguous()


#: Storage layouts accepted when reading back the i4 series → the holder dtype for the raw bytes.
#:
#: i4 is packed 4-bit (2 elements per byte — ADR 0069 decision 2), so the holder is uint8 and
#: `karume.emit.unpack_int4` restores the logical shape. i8 appears in the `dit` container for
#: **the 5 weights outside the blocks** (`in_proj` / `out_proj` / `cond_module.{0,2,4}`) and **the
#: 144 adaLN weights** (`attention_adaln` / `mlp_adaln`), 149 in total (both taken out of i4 by the
#: listening ruling of 2026-08-23; `irodori.export._fake_quant_i4`); the `dit-context` container
#: holds only i4 (the 72 condition-side K/V projections) and f32 (norms).
#: A storage not listed here (f16 / bf16) appearing in a DiT container means the mix of the
#: i8+dit4 preset is shipping in an unexpected form ({@link restore_dit_from_i4_series} fails).
_RESTORE_STORAGE: Mapping[str, torch.dtype] = {
    "f32": torch.float32,
    "i8": torch.int8,
    "i4": torch.uint8,
}

#: Storages whose inverse transform needs a scale (no scale in the declaration = cannot be read
#: back = immediate error).
_SCALED_STORAGE = frozenset({"i8", "i4"})


class RestoredDit(NamedTuple):
    """The record of reading back one container of the i4 series (one per target in meta.json's
    `i4Source`)."""

    #: The container read (= the very bytes that go into the distribution).
    container: Path
    #: The series' calibration conditions (`calib_provenance.json` — method, grid, case count,
    #: step count).
    calib: Mapping[str, Any]
    #: Number of parameters stored as i4 (the container is the source of truth — no expected
    #: value is baked into code).
    int4: int
    #: Number of parameters stored as i8 (`in_proj` / `out_proj` / `cond_module` outside the
    #: blocks of `dit` and adaLN inside the blocks — 149 with the real weights; 0 for
    #: `dit-context`).
    int8: int
    #: Number of parameters read back still stored as f32 (bias / norm).
    plain: int
    #: Number of i4 parameters whose **value changed** on read-back (the measured value of the
    #: check that the preset takes effect).
    changed: int


def _shippable_calib(series_dir: Path, flow: str = IRODORI_FLOW_RF) -> Mapping[str, Any]:
    """Read the calibration conditions of an i4 series, check that they are **conditions fit
    for distribution** (GPTQ × a budget at or above the floor) and return them.

    MUST: no golden is baked from the output of `--no-calib` (plain RTN) or `--calib-steps 1` (a
    smoke budget). Neither the calibration method nor the budget changes a single byte of the
    storage form (the grid stays RTN i4 g32), so nothing in the container tells them apart and
    the only symptom is degraded audio — the **same predicate** that the distribution build
    applies in `irodori.distribution.assert_irodori_calib_provenance` (`_shared.calib_provenance`
    is the source of truth) is applied to rebaking the golden too (if only the golden were taken
    with a smoke rounding, the comparison with the distributed assets would be green while "the
    two sides are different weights").
    """
    path = series_dir / CALIB_PROVENANCE_FILE
    if not path.is_file():
        raise SystemExit(
            f"no record of the i4 series' calibration conditions: {path}"
            " (written by `python -m irodori.export --dtype i4`)"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as cause:
        raise SystemExit(
            f"cannot parse the calibration conditions record: {path} — {cause}"
        ) from cause
    complaint = calib_complaint(
        record, method=CALIB_SHIPPABLE_METHOD, at_least=irodori_calib_floor(flow)
    )
    if complaint is not None:
        raise SystemExit(
            f"the i4 series' calibration conditions do not meet the distribution's: {path}:"
            f" {complaint} — no golden is baked from the output of `--no-calib` / a smoke budget"
        )
    return record


def _restored_tensor(container: Path, key: str, entry: StoredTensor) -> torch.Tensor:
    """A container's stored payload → a torch tensor of the logical shape (i4 includes nibble
    unpacking).

    The unpacking itself is **not written here**; it calls the core (`karume.emit.unpack_int4`).
    A payload whose length disagrees with the declaration (shape × codec) is rejected earlier by
    the container-side verification (`karume.verify.verify_container`), so all this looks at is
    "whether the storage can be read back".
    """
    holder = _RESTORE_STORAGE.get(entry.layout)
    if holder is None:
        raise SystemExit(
            f"{container}: '{key}' has the storage {entry.layout!r}, which cannot be read back"
            f" (the dit of the i8+dit4 preset holds only {' / '.join(sorted(_RESTORE_STORAGE))})"
        )
    flat = torch.frombuffer(bytearray(entry.payload), dtype=holder)
    shape = list(entry.shape)
    return unpack_int4(flat, shape) if entry.layout == "i4" else flat.reshape(shape)


def _restored_scale(container: Path, key: str, entry: StoredTensor) -> torch.Tensor:
    """The companion scale → the shape the inverse transform needs (`[rows, groups]`).

    Per-channel (i8) has 1 group, so it becomes the keepdim form `[rows, 1]` and broadcasting
    lands on the right axis as is (the NOTE of {@link _dequantize_stored}).
    """
    if entry.scale is None or entry.group_size is None:
        raise SystemExit(f"{container}: the {entry.layout}-stored '{key}' has no scale declaration")
    rows = entry.shape[0] if entry.shape else 1
    row_length = math.prod(entry.shape[1:]) if len(entry.shape) > 1 else 1
    return torch.frombuffer(bytearray(entry.scale), dtype=torch.float32).reshape(
        rows, row_length // entry.group_size
    )


def _dequantize_stored(dtype: str, raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Shipped bytes (i4 packed / i8) → the f32 logical form via the scale.

    i4 calls the core's `karume.quantize.dequantize_int4` as is (**the same path** by which
    `karume.emit` measures "restored from the bytes it writes, bit-identical" before storing —
    ADR 0069 decision 4 ③).

    NOTE: the core has no counterpart function for i8 (`karume.quantize` has only
    `quantize_to_int8`, and `karume.emit._convert_for_storage` writes the inverse directly as
    `quantized * scale` inside its bit-identity check). This is the same single expression, and the
    per-channel scale has the keepdim form, so broadcasting lands on the right axis as is.
    """
    if dtype == "i4":
        return dequantize_int4(raw, scale)
    return raw.to(torch.float32) * scale


def restore_dit_from_i4_series(
    wrappers: Mapping[str, nn.Module], series_root: Path, flow: str = IRODORI_FLOW_RF
) -> dict[str, RestoredDit]:
    """Overwrite every parameter the DiT distribution wrappers own with the **shipped bytes** of
    the i4 series (rounding stage 2).

    Stage 1 ({@link emit}) rounds every role with **the same fake-quant as the i8 preset** — the
    quant preset `i8+dit4` is a mixed preset in which the other 7 roles share the i8 preset's i8
    bytes (`irodori.distribution.IRODORI_QUANT_SEATS`), so the condition encoders' rounding is
    correctly identical to the i8 golden. This stage layers only the DiT on top, and **does not
    run calibration again**: GPTQ's rounding targets depend on the captured activations, so
    nothing guarantees that "two calibrations produce the same rounding". The only form that can
    state mechanically that "the weights the golden saw are the distributed bytes themselves" is
    "read back the exported series".

    `wrappers` maps target name → wrapper (`irodori.export.DitWrappers.by_target` — `dit` and
    `dit-context`), and the container is `series_root/<target>/model.krm`. The parameters a
    wrapper owns are the Parameters of `TextToLatentRFDiT` themselves (held under the same
    attribute names without rebinding), so overwriting through the wrapper reaches the DiT itself.
    The two wrappers' ownership overlaps only in `k_norm` (ADR 0114).

    The overwrite covers not only the i4 storage but **every parameter the wrapper owns** (the
    149 i8-stored weights of `dit` and the f32-stored bias / norm are written from the container
    too). Writing from the container even the side that should match the original values leaves
    no exception to "the weights the golden saw = the distributed bytes".

    Checks (every one fails loudly; if any fails, not one byte of the golden is written) — per
    container:

    - **provenance**: the series was rounded with GPTQ calibration ({@link _shippable_calib})
    - **shape**: the FQNs to overwrite match the parameters the wrapper owns exactly (in the
      container but not in the module, or the reverse, is an immediate error). Letting a mismatch
      through would put only the weights that were not overwritten into the golden with their i8
      values
    - **count**: at least one tensor is stored as i4 (the container is the source of truth for
      the count — no expected value is baked in). **i8 is not counted** — it is the same rounding
      as stage 1, so it is no evidence that "the i4 series is being read"
    - **preset takes effect**: fail when no i4 parameter changed value on overwrite (the accident
      of calling an i8 golden an i8+dit4 golden while the read-back has no effect matches in both
      numbers and shapes, so no other check catches it)

    Between the two:

    - **the calibration conditions records are identical** — the two are the same rounding written
      out by one export. If they differ, series from separate runs sit side by side
    - **the storage of tensors present in both containers (the shared `k_norm`) is identical** —
      the side read later must not silently overwrite the earlier values

    NOTE: `karume/4` had a comparison "the I4 set of the header = the i4 set of the IR
    declaration", but in the container the storage has a single source of truth, the binding
    table (the declaration and the payload cannot move separately — container-v1 §5), so that
    comparison became vacuous. The fit between declaration and payload is covered by
    `verify_container`.
    """
    if not wrappers:
        raise SystemExit("no wrapper to read the i4 series back into")
    seen: dict[str, tuple[str, str]] = {}
    restored = {
        target: _restore_container(target, wrapper, series_root / target, seen, flow)
        for target, wrapper in wrappers.items()
    }
    calibs = {target: dict(record.calib) for target, record in restored.items()}
    if len({json.dumps(calib, sort_keys=True) for calib in calibs.values()}) > 1:
        raise SystemExit(
            f"the i4 series' calibration conditions records differ per target ({calibs}) — the"
            " two DiT graphs must have been written by one export (rewrite both with"
            " `python -m irodori.export --dtype i4`)"
        )
    return restored


def _stored_digest(entry: StoredTensor) -> str:
    """The identity key of one stored tensor (sha256 of layout, shape, payload and scale)."""
    digest = hashlib.sha256()
    digest.update(f"{entry.layout}:{list(entry.shape)}:{entry.group_size}".encode())
    digest.update(entry.payload)
    digest.update(entry.scale or b"")
    return digest.hexdigest()


def _restore_container(
    target: str,
    wrapper: nn.Module,
    series_dir: Path,
    seen: dict[str, tuple[str, str]],
    flow: str = IRODORI_FLOW_RF,
) -> RestoredDit:
    """Read back one container (the checks are described in
    {@link restore_dit_from_i4_series}).

    `seen` maps the tensor keys of the containers read so far → `(target, storage key)`. When the
    same key appears in the second container, the storage is checked for identity before it is
    overwritten.
    """
    container = series_dir / ex.MODEL_FILE
    if not all(part.is_file() for part in container_parts(container)):
        raise SystemExit(
            f"no container of the i4 series: {container}"
            " (run `python -m irodori.export --dtype i4` first)"
        )
    calib = _shippable_calib(series_dir, flow)
    stored = weight_tensors(read_stored(container))
    owned = dict(wrapper.named_parameters())
    absent = sorted(set(owned) - set(stored))
    extra = sorted(set(stored) - set(owned))
    if absent or extra:
        raise SystemExit(
            f"{container}: the overwrite targets do not match the parameters the wrapper owns —"
            f" missing from the container {absent[:3]} / missing from the module {extra[:3]}"
            " (either the DiT wrappers' structure or the exported series has moved)"
        )
    int4_keys = frozenset(key for key, entry in stored.items() if entry.layout == "i4")
    if not int4_keys:
        raise SystemExit(f"{container}: no tensor is stored as i4 (not an i4 series)")
    for key, entry in stored.items():
        digest = _stored_digest(entry)
        previous = seen.get(key)
        if previous is not None and previous[1] != digest:
            raise SystemExit(
                f"{container}: the storage of '{key}' differs from the {previous[0]} container —"
                " the two containers come from separate export runs (shared tensors should be"
                " the same bytes)"
            )
        seen[key] = (target, digest)

    # One tensor at a time: "restore → compare → write → discard". Holding everything as f32
    # before writing would keep alive a second copy as large as `dit`'s f32 set (1.4 GB class) at
    # the same time (the counterpart of `karume.emit` serializing the storage side for the same
    # reason). A failed check raises at that point, so a partly overwritten module never writes a
    # golden.
    changed = 0
    with torch.no_grad():
        for key in sorted(stored):
            entry = stored[key]
            raw = _restored_tensor(container, key, entry)
            value = (
                _dequantize_stored(entry.layout, raw, _restored_scale(container, key, entry))
                if entry.layout in _SCALED_STORAGE
                else raw
            )
            parameter = owned[key]
            if tuple(value.shape) != tuple(parameter.shape):
                raise SystemExit(
                    f"{container}: the shape of '{key}' disagrees: container"
                    f" {list(value.shape)} / module {list(parameter.shape)}"
                )
            # MUST: only **the i4 entries** count toward the takes-effect check. The 149 i8
            # weights share stage 1's i8 scale and grid, so their values not moving is normal;
            # counting them would let the i8 matches cover up "not a single i4 took effect".
            if entry.layout == "i4" and not torch.equal(parameter.detach(), value):
                changed += 1
            parameter.copy_(value)
            del raw, value
    if changed == 0:
        raise SystemExit(
            f"{container}: all {len(int4_keys)} read-back i4 tensors equal the stage-1 (i8"
            " rounding) values — the i4 read-back has no effect (the accident of calling an i8"
            " golden an i8+dit4 golden)"
        )
    int8 = sum(1 for entry in stored.values() if entry.layout == "i8")
    plain = len(stored) - len(int4_keys) - int8
    print(
        f"[fake-quant] {target}: overwritten with the i4 series' shipped bytes —"
        f" i4 {len(int4_keys)} ({changed} of them differ from stage 1) /"
        f" i8 {int8} / f32 {plain}, calibration {calib}",
        flush=True,
    )
    return RestoredDit(
        container=container,
        calib=calib,
        int4=len(int4_keys),
        int8=int8,
        plain=plain,
        changed=changed,
    )


def emit(model_dir: Path, source_dir: Path, out_dir: Path, dtype: str = "f32") -> dict[str, Any]:
    """Write the full-loop golden and return a summary (writes not one byte if verification
    fails)."""
    from tokenizers import Tokenizer

    source = ex.IrodoriSource(source_dir)
    text_config, model_config = ex.read_configs(model_dir)
    state = load_file(str(model_dir / ex.CHECKPOINT_FILE))
    backbone = ex.load_backbone(source, state, text_config)
    hidden_size = int(backbone.hidden_size)
    text_projector = ex.load_projector(
        source, state, model_config, ex.TEXT_PROJ_PREFIX, hidden_size, int(model_config["text_dim"])
    )
    caption_projector = ex.load_projector(
        source,
        state,
        model_config,
        ex.CAPTION_PROJ_PREFIX,
        hidden_size,
        int(model_config["caption_dim"]),
    )
    config = source.model_config(model_config)
    speaker_encoder = ex.load_speaker_encoder(source, state, config)
    speaker_norm = ex.load_rms_norm(
        source, state, ex.SPEAKER_NORM_PREFIX, int(config.speaker_dim), float(config.norm_eps)
    )
    text_norm = ex.load_rms_norm(
        source, state, ex.TEXT_NORM_PREFIX, int(config.text_dim), float(config.norm_eps)
    )
    caption_norm = ex.load_rms_norm(
        source,
        state,
        ex.CAPTION_NORM_PREFIX,
        int(config.caption_dim_resolved),
        float(config.norm_eps),
    )
    duration_predictor = ex.load_duration_predictor(source, state, config)
    dit = ex.load_dit(source, state, config, text_config)
    # The checkpoint's own flow parameterization picks the loop and the upstream sampler compared
    # with (ADR 0150).
    flow = ex.dit_flow(dit)

    # MUST: rounding happens right after load and before the golden is taken (the ordering MUST of
    # `irodori.export.fake_quant`). `dit` is the **whole** `TextToLatentRFDiT` (it holds copies of
    # backbone / projector / speaker / duration inside), so the upstream comparisons
    # (`sample_euler_rf_cfg` / `predict_duration_log_frames`) also run on the weights rounded here
    # = both the graph path and the upstream path receive the same rounding.
    #
    # MUST (i4 = the distribution's quant preset `i8+dit4`): rounding is split into two stages.
    # Stage 1 is **the same fake-quant as the i8 preset** here (the i8+dit4 preset has the other 7
    # roles share the i8 preset's i8 bytes, so the condition encoders' rounding is correctly
    # identical to the i8 golden); stage 2 has {@link restore_dit_from_i4_series} overwrite only the
    # parameters the wrappers of the two DiT graphs (`dit` / `dit-context`) own with the shipped
    # bytes. Why this does not call the i4 path of `irodori.export.fake_quant` and run calibration
    # once more here: see stage 2's docstring.
    quantized = ex.fake_quant(
        "i8" if dtype == "i4" else dtype,
        {
            ex.TARGET_BACKBONE: backbone,
            ex.TARGET_TEXT_PROJ: text_projector,
            ex.TARGET_CAPTION_PROJ: caption_projector,
            ex.TARGET_SPEAKER: speaker_encoder,
            ex.TARGET_DURATION: duration_predictor,
            ex.TARGET_DIT: dit,
            "speaker_norm": speaker_norm,
            "text_norm": text_norm,
            "caption_norm": caption_norm,
        },
    )
    restored = (
        restore_dit_from_i4_series(
            # The parameters the wrappers own are `dit`'s Parameters themselves (the DiT wrappers
            # hold them under the same attribute names without rebinding), so overwriting through
            # the wrappers reaches `dit` too.
            ex.dit_wrappers(dit, ex.dit_sym_max(config)).by_target(),
            ex.default_out_root(model_dir, dtype),
            flow,
        )
        if dtype == "i4"
        else None
    )

    # MUST: the graph wrappers work correctly only **after patching** (they pass the real-valued
    # RoPE table). This file does not take the "unpatched reference" (irodori/export.py does), so
    # applying the patches here is fine.
    patch.apply_patches()
    speaker_max = ex.speaker_sym_max(model_config)
    dit_graphs = ex.dit_wrappers(dit, ex.dit_sym_max(config))
    graphs = HostGraphs(
        backbone=ex.BackboneGraph(backbone),
        text_proj=ex.ProjectorGraph(text_projector),
        caption_proj=ex.CaptionProjectorGraph(caption_projector, caption_norm),
        speaker=ex.SpeakerGraph(speaker_encoder, speaker_norm, speaker_max),
        duration=ex.DurationGraph(duration_predictor, text_norm),
        dit_context=dit_graphs.context,
        dit=dit_graphs.dit,
    )
    caps = {
        "text": int(model_config["max_text_len"]),
        "speaker": speaker_max + 1,
        "caption": int(model_config["max_caption_len"]),
    }
    tokenizer = Tokenizer.from_file(str(model_dir / ex.TOKENIZER_FILE))
    caption_tokenizer = upstream_caption_tokenizer(model_dir, bool(config.caption_add_bos_resolved))

    schedule = (
        meanflow_schedule(MEANFLOW_NUM_STEPS)
        if flow == IRODORI_FLOW_MEANFLOW
        else t_schedule(NUM_STEPS)
    )
    cases: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, torch.Tensor]] = {}
    for case in PIPELINE_CASES:
        tensors, meta = run_case(
            case,
            graphs,
            source,
            duration_predictor,
            tokenizer,
            text_config,
            model_config,
            config,
            caps,
            flow=flow,
        )
        # ---- standing check: compare S and the final z with upstream's canonical path ----
        text_ids = tensors["text_ids"]
        # MUST: the caption's condition input is built independently from upstream's entry point
        # (the host's sequence is not passed back — {@link upstream_caption_condition}). That the
        # two are the same sequence is measured right below.
        caption_padded, caption_mask = upstream_caption_condition(
            caption_tokenizer, case.caption, caps["caption"]
        )
        host_caption_ids = tensors.get("caption_ids")
        upstream_used = int(caption_mask.sum())
        host_used = 0 if host_caption_ids is None else int(host_caption_ids.shape[1])
        if host_used != upstream_used or (
            host_caption_ids is not None
            and not torch.equal(host_caption_ids[0], caption_padded[0, :upstream_used])
        ):
            raise SystemExit(
                f"{case.name}: the caption token sequences disagree between the host ({host_used}"
                f" tokens) and upstream ({upstream_used} tokens) — the preprocessing spellings"
                " have diverged"
            )
        upstream_steps = upstream_sequence_length(
            dit,
            text_ids,
            caption_padded,
            caption_mask,
            tensors["reference_latent"],
            case.reference is not None,
            text_config,
            model_config,
            config,
        )
        if upstream_steps != meta["S"]:
            raise SystemExit(
                f"{case.name}: S disagrees: host {meta['S']} / upstream {upstream_steps}"
                " (something in expm1 → banker's rounding → clamp differs)"
            )
        reference, upstream_noise = upstream_latent(
            dit,
            case,
            text_ids,
            caption_padded,
            caption_mask,
            tensors["reference_latent"],
            meta["S"],
            text_config,
            model_config,
            config,
            flow=flow,
        )
        if not torch.equal(upstream_noise, tensors["noise"]):
            raise SystemExit(
                f"{case.name}: the initial noise does not match upstream (seed is used differently)"
            )
        worst = float((reference - tensors["z"]).abs().max())
        if worst > EULER_REFERENCE_ATOL:
            # ---- second stage: measure the amplification factor and judge normalized (the
            # addendum of 2026-09-01 in the constant's comment) ----
            # run_case is deterministic, so a rerun + a perturbation run add only the amplification
            # factor. The identity of z is a precondition of the judgement (otherwise "the
            # sensitivity of the same trajectory" is not what is measured), so it is checked
            # explicitly.
            probed_tensors, probed_meta = run_case(
                case,
                graphs,
                source,
                duration_predictor,
                tokenizer,
                text_config,
                model_config,
                config,
                caps,
                sensitivity_probe=True,
                flow=flow,
            )
            if not torch.equal(probed_tensors["z"], tensors["z"]):
                raise SystemExit(
                    f"{case.name}: the rerun's z does not match — the host path is"
                    " non-deterministic, so the amplification measurement does not hold"
                )
            amp = float(probed_meta["sensitivityAmp"])
            meta["sensitivityAmp"] = probed_meta["sensitivityAmp"]
            if not euler_reference_within_sensitivity(worst, amp):
                raise SystemExit(
                    f"{case.name}: the final z differs from upstream by {worst} (exceeds the"
                    f" first-stage tolerance {EULER_REFERENCE_ATOL}). Normalized by the"
                    f" amplification factor {amp:.4g} it is still {worst / amp:.4g} >"
                    f" {EULER_NOISE_PER_AMP}, or it exceeds the absolute ceiling"
                    f" {EULER_REFERENCE_ABS_CEILING} — a difference error accumulation cannot"
                    " explain; the CFG combination formula / t schedule / interval / mask segments"
                    " seem broken"
                )
        meta["upstreamMaxAbs"] = float(f"{worst:.4e}")
        cases[case.name] = meta
        payloads[case.name] = tensors

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for name, tensors in payloads.items():
        path = out_dir / f"{CASE_PREFIX}{name}{CASE_SUFFIX}"
        save_file(
            {
                key: normalize_boundary_tensor(value.contiguous(), f"{name} '{key}'")
                for key, value in tensors.items()
            },
            str(path),
        )
        written[path.name] = path.stat().st_size
    embed_dim = int(config.timestep_embed_dim)
    table = t_embed_table(source, schedule, embed_dim)
    embeddings = {"t": schedule[:-1].contiguous(), "t_embed": table}
    if flow == IRODORI_FLOW_MEANFLOW:
        # The intervals the loop used and their embeddings (the same upstream function — the
        # TS-side `delta_embed` is checked against this table).
        intervals = (schedule[:-1] - schedule[1:]).contiguous()
        embeddings |= {
            "delta": intervals,
            "delta_embed": embedding_table(source, intervals, embed_dim),
        }
    save_file(embeddings, str(out_dir / T_EMBED_FILE))
    written[T_EMBED_FILE] = (out_dir / T_EMBED_FILE).stat().st_size

    # MUST: the RF meta keeps its keys and their order (the TS-side latent check reads them). A
    # MeanFlow loop declares its parameterization and has no initial scale and no CFG.
    sampler: dict[str, Any] = (
        {
            "flowParameterization": IRODORI_FLOW_MEANFLOW,
            "steps": MEANFLOW_NUM_STEPS,
        }
        if flow == IRODORI_FLOW_MEANFLOW
        else {
            "steps": NUM_STEPS,
            "initScale": INIT_SCALE,
            "cfgRange": [CFG_MIN_T, CFG_MAX_T],
            "cfgScales": dict(CFG_SCALES),
        }
    )
    schedule_meta: dict[str, Any] = (
        {"deltaSchedule": [float(value) for value in (schedule[:-1] - schedule[1:]).tolist()]}
        if flow == IRODORI_FLOW_MEANFLOW
        else {"tScheduleClosedFormMaxAbs": closed_form_matches(schedule, NUM_STEPS)}
    )
    meta_payload: dict[str, Any] = {
        "dtype": dtype,
        "fakeQuant": quantized.reports,
        **sampler,
        "frameRate": ex.CODEC_FRAME_RATE,
        "tSchedule": [float(value) for value in schedule.tolist()],
        **schedule_meta,
        "tEmbedDim": embed_dim,
        "caps": caps,
        "cases": cases,
    }
    if restored is not None:
        # MUST: no existing key moves (the deno-side latent check reads them). This is the one
        # key added only for i4, covering the provenance that `fakeQuant`, which only speaks of
        # stage 1 (i8), leaves out. Its content is a record per target (the two DiT graphs), read
        # by humans only (the machine comparison is the read-back checks).
        meta_payload["i4Source"] = {
            target: {
                "container": str(record.container),
                "calib": record.calib,
                "int4Tensors": record.int4,
                "int8Tensors": record.int8,
                "f32Tensors": record.plain,
                "changedByRestore": record.changed,
            }
            for target, record in restored.items()
        }
    (out_dir / META_FILE).write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    written[META_FILE] = (out_dir / META_FILE).stat().st_size
    return {"dir": str(out_dir), "bytes": written, **meta_payload}


def default_out_dir(model_dir: Path, dtype: str = "f32") -> Path:
    """The default location (`outputs/series/irodori-<the real-weight directory
    name>{,-f16,-i8,-i4}/pipeline/`).

    The spelling of the series root is identical to `irodori.export.default_out_root` —
    `pipeline/` is one level below it (alongside the graph targets).
    """
    return ex.default_out_root(model_dir, dtype) / "pipeline"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=ex.DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=ex.DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--dtype",
        choices=ex.WEIGHT_DTYPES,
        default="f32",
        help="storage dtype of the weights (f16 / i8 rebake the golden with the weights after"
        " fake-quant — ADR 0018 / 0019 / 0027 / 0050. i4 starts from the same rounding as the i8"
        " preset and overwrites the two DiT graphs (dit / dit-context) with the shipped bytes"
        " of the exported i4 series (derived from --model-dir) — ADR 0069 / 0114)",
    )
    args = parser.parse_args(argv)
    out_dir = default_out_dir(args.model_dir, args.dtype) if args.out is None else args.out
    summary = emit(args.model_dir, args.source_dir, out_dir, args.dtype)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
