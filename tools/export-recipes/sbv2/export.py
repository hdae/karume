"""Script that writes the SBV2 acoustic chain to IR v2 containers (`krm`) + golden io (the emit
targets of ADR 0013).

There are 5 targets (all of ADR 0013):

- **`dp`** (M1-P3 wave 1) — DurationPredictor alone. It needs neither the patch layer nor
  vocabulary growth; a scaffold for getting "real weights → export → real-GPU E2E" through end to
  end first.
- **`front`** (wave 6) — enc_p + dp + sdp(reverse) fused into one graph. Dynamic `P` (phoneme
  count); it can be exported only with the patch layer of `sbv2.patch` applied (branch-free spline
  / gather-based relative-position attention / folding of the FFN pad).
- **`flow`** (wave 7) — the reverse of TransformerCouplingBlock. Dynamic `T` (frame count); the
  `(T,T)` tables of the relative-position attention are **graph inputs** (baking them in would be
  134MB — ADR 0013).
- **`dec`** (wave 7) — HiFi-GAN Generator. No patches; the only preprocessing is
  `remove_weight_norm`.
- **`voice`** (wave 7) — flow + dec fused into one graph. **Once this E2E is green, the whole SBV2
  chain holds**.

    uv sync --all-groups                                    # once, in tools/export-recipes/
    uv run --group sbv2 python -m sbv2.export               # emit every target
    uv run --group sbv2 python -m sbv2.export --target front
    uv run --group sbv2 python -m sbv2.export --dtype f16   # → outputs/series/sbv2-F1-f16/
    uv run --group sbv2 python -m sbv2.export --dtype i8    # → outputs/series/sbv2-F1-i8/
    uv run --group sbv2 python -m sbv2.export --dtype i4 --target front --target voice
    uv run --group sbv2 python -m sbv2.export --verify flow  # eager equivalence vs the reference

NOTE: the sync uses `--all-groups` because the workspace has one venv (`tools/pyproject.toml`);
syncing only `--group sbv2` **drops** the groups of the other families (measured).

MUST: `--verify` and emit **cannot be combined in one process** (the CLI rejects it
mechanically). The patches replace class attributes process-wide, so if the emit side patched
first, "the reference before patching" could no longer be taken, and the equivalence check would
become vacuous and give a false PASS (ADR 0013). The verification itself also keeps the order
"settle the references of every case → change → compare" and fails right before taking a
reference if the order is broken — front / flow / voice look at `sbv2.patch.patches_applied()`,
while **dec looks the other way**, at "parameters derived from weight_norm are still there" (the
source of contamination is remove).

MUST: `--verify` **takes exactly one target** (`--verify front` / `--verify dec` …). Allowing
several verifications in a row would make the CLI hold a pairwise exclusion table ("the MHA patches
exclude each other / dec's remove excludes only voice / rounding excludes everything"), and any
hole in the table would turn straight into a false PASS. With **one verification per process**,
contaminating combinations do not exist structurally.

The weights are not part of what is shipped (`inputs/` is in `.gitignore`). See the README for how
to get them.

Output layout (enumerated by the Deno side, `packages/runtime/tests/e2e_sbv2_test.ts`). The series
name is derived from the directory name of `--model-dir` (`sbv2-F1` for the default
`inputs/sbv2/F1/`):

    outputs/series/sbv2-F1/<target>/model.krm             weights, constants + 2 descriptors
    outputs/series/sbv2-F1/<target>/io.<case>.safetensors inputs + torch CPU expected outputs

The tensor key convention of io is the same as for the tiny goldens / DeBERTa
(`input.<graph input name>` / `output.<position>`).

## Series per storage dtype (ADR 0018 / 0019 / 0069)

`--dtype f16` / `--dtype i8` / `--dtype i4` each write to a **separate series** (`sbv2-F1-f16/` /
`sbv2-F1-i8/` / `sbv2-F1-i4/`) — sharing a directory with the f32 series would silently put the
existing E2E net (the f32 tolerance) on different assets. Rounding (fake-quant) applies the shared
`quantize.round_weights_to_f16` / `quantize.fake_quant_int8` / `quantize.fake_quant_int4`
**after remove_weight_norm / patching and before taking the references and goldens** (the order
MUST of `_fake_quant`).

Only `--dtype i4` is **mixed** (eligible `nn.Linear` / `nn.Conv1d` = i4 group32, the rest i8
per-channel), and in the distribution it goes into the `front` / `voice` seats of the `i4` quant
(`sbv2/distribution.py`). The i4 execution path is limited to the weight slots of linear /
embedding / conv1d (ADR 0069 decision 5 and its addendum), and `nn.ConvTranspose1d`, depthwise
convs (`groups > 1`) and weights whose row length is not divisible by group32 drop to the i8 side,
so the series can only be mixed.

MUST: `--dtype` is **emit-only** (treated like `--sym-max`). `--verify` is an eager comparison that
ignores the storage format, and for dec / voice rounding can only be applied **after** remove while
the reference is taken **before** remove, so combining them would compare "rounded vs unrounded"
and break the `bit_exact` claim. The CLI rejects it mechanically.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import normalize_boundary_tensor
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QUANT_MODULE_TYPES,
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
    round_weights_to_f16,
)

# The provenance record's file name and **the component names of the distribution** are owned by
# the reading side (the distribution assembly) — with the spelling in 2 places, the day only one
# moves silently produces "a record written but never read" or "a graph name the runtime cannot
# find" (the same direction as anima's `CALIB_PROVENANCE_FILE`). The allow-list of voice families
# lives on the same side — the assembly uses the same table for the repository boundary (the
# Pipeline's field).
from sbv2.distribution import (
    EXPORT_PROVENANCE_FILE,
    SBV2_FAMILY_DIRS,
    SBV2_FRONT_ROLE,
    SBV2_VOICE_ROLE,
)

from . import patch
from .card import SBV2_CARD_PROFILES

#: Where the real weights live. Outside version control (`inputs/` in `.gitignore`), placed by hand.
DEFAULT_MODEL_DIR = INPUTS_ROOT / "sbv2" / "F1"

#: The storage dtypes that can be written. `i8` was added by **the maintainer ruling of
#: 2026-08-04** (overriding "no i8" of ADR 0027 decision 1 — recorded in the ADR of the SBV2 w8
#: series). It is still **not meant for w8a8** (in all 5 targets conv1d takes 86–90% and linear is
#: practically 0 GFLOP — ADR 0025 decision ⑤). The aim is asset size and load time; the execution
#: path is exactly the w8a32 of ADR 0019 (i8 storage, f32 compute).
#: `i4` is the series name of a **mix**: in fact "eligible `nn.Linear` / `nn.Conv1d` = i4 group32,
#: everything else (ineligible linear / conv1d and `ConvTranspose1d` / embedding) = i8 per-channel
#: as before" ({@link BASE_WEIGHT_DTYPES} / {@link I4_MODULE_TYPES} / {@link _i4_module_names}).
#: Ineligible weights always remain (`ups` of `dec` has a transposed layout, `convs_sep` of `dp` is
#: depthwise), so the series can only ever be mixed. net_g has only 6 linears (front 2 / voice 4),
#: but **conv1d is the bulk of all 5 targets** (ADR 0025 decision ⑤), so only the conv1d addendum
#: (wave J-5b) moves the distribution size of net_g itself.
#:
#: MUST: passing `i4` for a target with no i4-eligible weight makes `fake_quant_int4` fail loudly
#: with "0 targets" (the check that keeps a silent i8 series from being produced).
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "i8", "i4")

#: Series name → the **default** storage dtype passed to `export_to_file`. Only the i4 series
#: defaults to i8, and the eligible weights are switched to i4 one by one through
#: `weight_dtype_overrides` (the same form as deberta's i4 mixed series).
BASE_WEIGHT_DTYPES: Mapping[str, str] = {"f32": "f32", "f16": "f16", "i8": "i8", "i4": "i8"}

#: The module types rounded to i4 group32 (the module-side spelling of emit's `I4_WEIGHT_OPS` =
#: linear / embedding / conv1d). **embedding is not included** — i4 for net_g's vocabulary table
#: (`enc_p.emb`) is a separate decision, outside the scope of wave J-5b (`nn.ConvTranspose1d` does
#: not derive from `nn.Conv1d`, so `ups` of `dec` drops out naturally by type). The actual
#: eligibility is this type combined with `groups == 1` and row-length divisibility
#: ({@link _i4_module_names}).
I4_MODULE_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv1d)


class Sbv2FamilyError(ValueError):
    """The spelling of `--model-dir` does not determine a voice family (= source and license).

    The repository's convention is `Error` subclasses (`DistError` / `RopeSpecError` /
    `ContainerReadError`). With a plain `ValueError`, a caller's `except` could not tell this
    **legal-fact check** apart from "a bad argument" in general — on the day a swallowing path
    appears, it would silently go back to a distribution that declares the default license (what
    the MUST of {@link SBV2_FAMILY_DIRS} prevents).
    """


def sbv2_family(model_dir: Path) -> str:
    """The directory name of `--model-dir` → voice family (`fn` / `jvnv`).

    The decision uses only the allow-list of {@link SBV2_FAMILY_DIRS}; a name that matches none
    **fails loudly** (the MUST of that constant — {@link Sbv2FamilyError}).
    """
    name = model_dir.name
    for family, prefixes in SBV2_FAMILY_DIRS.items():
        if name.startswith(prefixes):
            return family
    listed = " / ".join(
        f"{family}: {', '.join(prefixes)}" for family, prefixes in sorted(SBV2_FAMILY_DIRS.items())
    )
    raise Sbv2FamilyError(
        f"'{name}' matches the spelling of no voice family (known prefixes — {listed})"
        " — its provenance and license cannot be determined, so no distribution is baked"
        " (follow the '--model-dir' convention in the README)"
    )


def sbv2_provenance(model_dir: Path) -> Provenance:
    """The provenance baked into the series' containers (container-v1 §2.3).

    SBV2 has **a different legal fact** per voice family (`sbv2.card.SBV2_CARD_PROFILES` — FN is
    `other` + Booth's distribution terms, jvnv is `cc-by-sa-4.0` from the JVNV corpus). The source
    of truth for the identifier is the card side, so this only decides **which family it is**
    ({@link sbv2_family}).
    """
    profile = SBV2_CARD_PROFILES[sbv2_family(model_dir)]
    return Provenance(
        license=profile.metadata.license,
        # The revision (commit SHA) of the upstream checkpoint. The config's `version`
        # (`source_version`) is the SBV2 generation and does not identify a checkpoint, so it does
        # not go into this field (container-v1 §2.3).
        upstream_revision=profile.source_revision,
    )


def default_out_root(model_dir: Path, dtype: str) -> Path:
    """The default location of the output (`outputs/series/sbv2-<directory name of the real
    weights>{,-f16,-i8,-i4}/`).

    The caller adds the subdirectory of the target name (`dp` / `front`).

    The speaker name (the directory name of `--model-dir`) is baked into the series spelling so
    that series do not collide once there are several speakers — sharing the spelling `sbv2/`
    would silently overwrite the earlier assets the moment another speaker is written.

    MUST: a separate directory per dtype (ADR 0018 / 0019) — sharing one would silently put the net
    of the f32 series (per-target tolerances derived from measurements) on compressed assets.
    """
    suffix = "" if dtype == "f32" else f"-{dtype}"
    return SERIES_ROOT / f"sbv2-{model_dir.name}{suffix}"


#: Target names (also the subdirectory names of the series). `front` / `voice` are also **the
#: component names of the distribution**, so their spelling's source of truth comes from the side
#: that needs no torch (`sbv2/distribution.py`) — the container's graph name MUST match the key in
#: the manifest's weights (container-v1 §2.1). `dp` / `flow` / `dec` are for golden verification
#: only and are not in the distribution, so this is their only spelling.
TARGET_DP = "dp"
TARGET_FRONT = SBV2_FRONT_ROLE
TARGET_FLOW = "flow"
TARGET_DEC = "dec"
TARGET_VOICE = SBV2_VOICE_ROLE
TARGETS = (TARGET_DP, TARGET_FRONT, TARGET_FLOW, TARGET_DEC, TARGET_VOICE)
CONFIG_FILE = "config.json"
STYLE_FILE = "style_vectors.npy"
MODEL_FILE = "model.krm"

IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: Limit of the symbolic dimension P (phoneme count) — **the default for the front group (dp /
#: front)**.
SYM_MAX = 512

#: Limit of the symbolic dimension T (frame count) — **the default for the flow group (flow / dec /
#: voice)**.
FLOW_SYM_MAX = 4096

#: Target → symbolic-dimension limit. Exactly ADR 0013's "sym_max defaults per target"; a mix-up is
#: silent (emitting front at 4096 bakes 134MB of relative-position tables, emitting flow at 512
#: makes utterances longer than 512 frames ≈ 6 seconds fail with Tmax exceeded). **The CLI's
#: `--sym-max` defaults to None** and must be given explicitly only to deviate (and only for a
#: single target — one value handed to several targets is always wrong for one of them).
TARGET_SYM_MAX: dict[str, int] = {
    TARGET_DP: SYM_MAX,
    TARGET_FRONT: SYM_MAX,
    TARGET_FLOW: FLOW_SYM_MAX,
    TARGET_DEC: FLOW_SYM_MAX,
    TARGET_VOICE: FLOW_SYM_MAX,
}

#: Window size of the relative-position attention. The gather patch builds its index tables
#: assuming embeddings of this width (2w+1 = 9 rows), and the exporter's constant folding bakes them
#: in at Pmax.
EXPECTED_WINDOW_SIZE = 4

#: The speaker ID whose embedding is taken. The real weights are a single-speaker model
#: (`n_speakers` of the config).
SPEAKER_ID = 0

#: The row of the style vector used (`style_vectors.npy` is [number of styles, 256]). 0 is the
#: average style.
STYLE_ID = 0

#: Random numbers derive from this (independent of the global seed — regeneration is
#: byte-identical).
SEED = 20260802

#: Noise scale of the sdp reverse. **Not baked into the graph** (a runtime knob — ADR 0013): the
#: golden `z_noise` is "the already-multiplied sequence", and the runtime reproduces the same
#: numbers without knowing the scale. Uses 0.8, the same as the default of the reference
#: implementation `StochasticDurationPredictor.forward(reverse=True)`.
NOISE_SCALE = 0.8

#: Golden cases `(name, length, trailing padding)`. **Shared by every target** (the Deno-side
#: `EXPECTED_CASES` checks them for equality across targets with one table). What the length
#: means depends on the target — P (phoneme count) for the front group, T (frame count) for the
#: flow group.
#:
#: The lengths are spread from the lower bound (2 — the smallest value that avoids the 0/1
#: specialization of torch.export) up to front's declared limit SYM_MAX. `padded` is **the only case
#: with 0s at the end of the mask**, exercising that the mask multiplication takes effect (if it
#: does not, the outputs in the padding columns are no longer 0).
#:
#: NOTE: there is **no** golden at the flow group's declared limit 4096. The relative-position
#: tables and attention scores are O(T²), so at T=4096 the tables alone make one io case 134MB, and
#: dec's output is 512·4096 = 2.1 million points — not worth it as a golden asset or as a real-GPU
#: test. Behaviour near the limit is covered instead by exercising "no implementation depends on the
#: declared limit (such as building a plan at Tmax)" through the spread of lengths (5 cases where
#: P/T varies 256× from 2 → 512).
#: **This table cannot exercise grid-stride degradation** — that path triggers only when the
#: required workgroup count exceeds the 1-D dispatch limit (65535), which T=512 does not reach (pad
#: exceeds it at T ≳ 2900). It is covered by the Deno-side degradation harness
#: `packages/runtime/tests/gpu_gridstride_test.ts` (which deliberately limits the dispatch count and
#: applies it to every kernel family).
GOLDEN_CASES: tuple[tuple[str, int, int], ...] = (
    ("p2", 2, 0),
    ("p37", 37, 0),
    ("p203", 203, 0),
    ("p512", SYM_MAX, 0),
    ("padded", 16, 5),
)

#: The argument order of `Sbv2Front.forward` (= the declaration order of the IR inputs).
#: dynamic_shapes correspond by position, so the order of the example inputs is fixed in this one
#: place (reordering would attach the dynamic axes to other inputs).
FRONT_INPUT_ORDER = ("x", "x_mask", "tone", "language", "bert", "style_vec", "g", "z_noise")

#: front's output names (corresponding to `output.<position>` of the IR; for the equivalence
#: report).
FRONT_OUTPUT_NAMES = ("logw_sdp", "logw_dp", "m_p", "logs_p")

#: Cases of the patch equivalence check `(P, trailing padding)`. Taken more widely than the
#: goldens — equivalence is a property that should hold over **the continuous range of P**, not
#: just at the 5 golden points, and in particular it must exercise `P ≤ window_size` (the range
#: where the slice width of the relative-position embedding changes with P) and both sides of that
#: boundary.
VERIFY_CASES: tuple[tuple[int, int], ...] = (
    (2, 0),
    (4, 0),
    (5, 0),
    (9, 0),
    (16, 5),
    (37, 0),
    (64, 0),
    (203, 0),
    (SYM_MAX, 0),
)

#: The input order of `FlowReverse` / `Sbv2Voice` / `dec` (= the declaration order of the IR
#: inputs). As for front, dynamic_shapes correspond by position, so the order is fixed in this one
#: place.
FLOW_INPUT_ORDER = ("z_p", "y_mask", "g", "idx_k", "valid")
DEC_INPUT_ORDER = ("x", "g")

#: Equivalence-check cases of the flow group `(T, trailing padding)`. Taken more widely than the
#: goldens (equivalence should hold over the continuous range of T, not at the 5 golden points). It
#: exercises **both sides of the window size 4** (T ≤ w / T = w+1 / T > w) — the relative-position
#: table is `clamp(rel+w, 0, 2w)`, so for small T the clamp applies everywhere and takes a different
#: branch from large T.
FLOW_VERIFY_CASES: tuple[tuple[int, int], ...] = (
    (2, 0),
    (4, 0),
    (5, 0),
    (9, 0),
    (16, 5),
    (24, 0),
    (50, 0),
    (137, 0),
    (203, 0),
    (SYM_MAX, 0),
)


class DurationPredictorGraph(nn.Module):
    """Export wrapper that fixes dp to the form "g required, argument names = IR input names".

    The plain `DurationPredictor.forward` has `g` as `Optional` with a `g is not None` branch. The
    wrapper makes it required to remove the branch and aligns the input names with the names used
    in the recon (h / x_mask / g) — the IR input names are the forward's argument names as is, so
    this is the source of truth for the golden's `input.<name>` keys.
    """

    def __init__(self, dp: nn.Module) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, h: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.dp(h, x_mask, g=g)


def load_net_g(model_dir: Path) -> tuple[nn.Module, Any]:
    """Builds net_g from the real weights placed in `inputs/sbv2/<name>/` and returns it in eval
    mode.

    Requires exactly one `*.safetensors` ckpt (with several, "which one was read" would change
    silently).
    """
    from style_bert_vits2.models.hyper_parameters import HyperParameters
    from style_bert_vits2.models.infer import get_net_g

    # Non-recursive glob. The series live under `outputs/series/`, so only input material is here.
    ckpts = sorted(model_dir.glob("*.safetensors"))
    if len(ckpts) != 1:
        raise FileNotFoundError(
            f"the ckpt in {model_dir} is not unique ({len(ckpts)}: {[p.name for p in ckpts]})"
        )
    config = model_dir / CONFIG_FILE
    if not config.is_file():
        raise FileNotFoundError(f"{config} is missing (put it in the same directory as the ckpt)")

    hps = HyperParameters.load_from_json(str(config))
    net_g = get_net_g(str(ckpts[0]), hps.version, "cpu", hps)
    net_g.eval()
    _assert_window_size(net_g)
    # The 4 modules measured to have 0 weight_norm instances (recon §3 — the remaining 95 are all
    # in dec). Looking only at dp would leave a hole that silently writes out a model in which
    # derived parameters came back to enc_p / sdp / flow. **dec is not included** — it ships with
    # an active weight_norm, and `ensure_dec_plain` removes it right before export.
    _assert_no_weight_norm(net_g.enc_p, net_g.sdp, net_g.dp, net_g.flow)
    return net_g, hps


def _assert_window_size(net_g: nn.Module) -> None:
    """Pins at load time that the window size of the relative-position attention is 4, as assumed
    (ADR 0013).

    The index tables of the gather patch bake in `clamp(rel + 4, 0, 8)`, so a different window size
    on the model side would **gather embeddings of a different width**. The element counts match,
    so there is no shape error and it silently reads the wrong embeddings (the silent-wrong-value
    class). The goldens are generated with the same error too, so the numeric comparison misses it
    as well. Failing the moment the ckpt is swapped is the smallest fix, so it sits in the loader.

    MUST: the scan covers **all of net_g** (not just the first layer of each). The patch replaces
    class attributes of MultiHeadAttention, and `FlowReverse` hands the same tables to **every
    coupling × every layer**, so checking only the first layer of a model whose window size differs
    per layer would let the other layers silently read another width. The granularity of the check
    matches how the tables are handed out.
    MUST: fail if no module has `window_size` at all (the vacuity check). If the attribute name
    changes upstream, the scan silently finds nothing, and from then on any window size passes.
    """
    found = [
        (name or type(module).__name__, module.window_size)
        for name, module in net_g.named_modules()
        # Layers that do not use relative-position attention have the attribute set to None
        # (measured: only the attn layers have a window size).
        if getattr(module, "window_size", None) is not None
    ]
    if not found:
        raise ValueError(
            "no module has window_size — the window size check is vacuous"
            " (the attribute name of relative-position attention may have changed upstream)"
        )
    mismatched = [(where, size) for where, size in found if size != EXPECTED_WINDOW_SIZE]
    if mismatched:
        raise ValueError(
            "the relative-position attention window size differs from the assumed one"
            f" ({EXPECTED_WINDOW_SIZE}) in {len(mismatched)}/{len(found)} modules: {mismatched}"
            " — when it disagrees with the width the table generator bakes in, wrong embeddings"
            " are read silently"
        )


def _weight_norm_parameters(*modules: nn.Module) -> list[str]:
    """The list of derived parameters from weight_norm (`weight_g` / `weight_v` /
    parametrizations).

    The checks for both "none left" and "still there" share this one test (writing the test in 2
    places always makes them diverge, leaving one check looser).
    """
    return [
        f"{type(module).__name__}.{name}"
        for module in modules
        for name, _ in module.named_parameters()
        if name.endswith(("weight_g", "weight_v")) or ".parametrizations." in name
    ]


def _assert_no_weight_norm(*modules: nn.Module) -> None:
    """Pins that there are no derived parameters from weight_norm.

    While weight_norm remains, `weight` is "not the effective weight", so writing it into the IR as
    is would give a different model. enc_p / sdp / dp measure 0 instances, but they are the side
    that does nothing on the assumption that there are none, so it is pinned with fail loudly (the
    ordering constraint of running remove first matters in the wave that handles dec).
    """
    derived = _weight_norm_parameters(*modules)
    if derived:
        raise ValueError(f"parameters derived from weight_norm remain: {derived}")


def _assert_patches_not_applied(where: str) -> None:
    """Pins that references are taken **before** patching (the check of ADR 0013's ordering
    constraint).

    The patches replace class attributes process-wide, so a "reference" taken after patching is the
    patched value itself, and the equivalence check becomes vacuous and gives a false PASS. It
    breaks in the direction where **the difference is always 0**, so a green verification proves
    nothing by itself — only a check can close it.

    """
    if patch.patches_applied():
        raise RuntimeError(
            f"{where}: tried to take a reference after patching — the equivalence check becomes"
            " vacuous (the order is 'fix the references of every case → patch → compare')"
        )


def _assert_weight_norm_present(module: nn.Module, where: str) -> None:
    """Pins that dec's references are taken **before** `remove_weight_norm` (**the check pointing
    the other way**).

    The reference of dec's equivalence check is "the original path with weight_norm active", and
    the claim is **bit identity before and after remove**. Taking the reference on an already
    removed net_g compares removed against removed, a difference of 0 holds trivially, and it gives
    a false PASS. While the other 3 targets require "no patches applied", this one alone requires
    "derived parameters **still present**" — because the source of contamination points the other
    way.
    """
    if not _weight_norm_parameters(module):
        raise RuntimeError(
            f"{where}: tried to take a reference on a net_g after remove_weight_norm"
            " — the equivalence check before and after remove becomes vacuous"
            " (the order is 'fix the references of every case → remove → compare')"
        )


def speaker_embedding(net_g: nn.Module, speaker_id: int = SPEAKER_ID) -> torch.Tensor:
    """Takes the speaker embedding g `[1, 512, 1]` from the real weights (the same shape as the
    reference implementation's infer).

    The real weights' embedding is used rather than synthetic noise — g goes into the conditioning
    of enc_p (the cond_layer_idx layer) and the cond of dp / sdp, so its real value range decides
    how the downstream LayerNorm and spline behave. A choice that ties the basis of the goldens'
    numeric tolerances to the value range of real use.
    """
    num_speakers = net_g.emb_g.num_embeddings
    if not 0 <= speaker_id < num_speakers:
        raise ValueError(f"speaker ID {speaker_id} is outside the model's {num_speakers} speakers")
    with torch.no_grad():
        return net_g.emb_g(torch.tensor([speaker_id], dtype=torch.int64)).unsqueeze(-1)


def style_vector(model_dir: Path, style_id: int = STYLE_ID) -> torch.Tensor:
    """Takes the style vector `[1, 256]` from the real asset (`style_vectors.npy`).

    No synthetic noise, for the same reason as the speaker embedding. style_proj adds the same value
    to every phoneme (a broadcast add `[1,1,256] → [1,1,192]`), so a shifted value range would put
    the mean of h somewhere other than in real use.
    """
    path = model_dir / STYLE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing (put it in the same directory as the ckpt)")
    table = np.load(path)
    if table.ndim != 2 or table.shape[1] != 256:
        raise ValueError(f"{path} has shape {table.shape}, not [number of styles, 256]")
    if not 0 <= style_id < table.shape[0]:
        raise ValueError(f"style ID {style_id} is outside the {table.shape[0]} styles")
    return torch.from_numpy(np.ascontiguousarray(table[style_id])).to(torch.float32).unsqueeze(0)


def _generator(length: int, pad: int, salt: int) -> torch.Generator:
    """Draws an independent random sequence per case (length, padding) and purpose.

    It does not depend on the global seed, so the result is byte-identical even when the
    generation order changes or a single case is rebuilt alone (the premise of regenerable
    goldens).
    """
    return torch.Generator().manual_seed(SEED + salt * 1_000_003 + length * 31 + pad)


def make_hidden(length: int) -> torch.Tensor:
    """h `[1, 192, P]` for dp (the counterpart of the enc_p output; a separate fixed seed per case).

    The real h is already masked by x_mask on the enc_p side, but here **the padding columns get
    values too** — if dp's own masking (the 3 `x * x_mask` and the last stage) takes effect, the end
    of the output is exactly 0, and if it does not, values leak. A detector for the mask path can
    only be built this way.
    """
    generator = torch.Generator().manual_seed(SEED + length)
    return torch.randn(1, 192, length, generator=generator)


def make_mask(length: int, pad: int) -> torch.Tensor:
    """x_mask `[1, 1, P]` — f32 with the first `P - pad` set to 1 and the last `pad` set to 0."""
    if not 0 <= pad < length:
        raise ValueError(f"pad={pad} is invalid for P={length}")
    mask = torch.ones(1, 1, length)
    if pad:
        mask[..., length - pad :] = 0.0
    return mask


def make_noise(length: int) -> torch.Tensor:
    """The external noise `z_noise` `[1, 2, P]` of the sdp reverse (already multiplied by
    `noise_scale`).

    MUST: it is **the same sequence** as the `torch.randn(1, 2, P) * noise_scale` the reference
    implementation draws internally (the equivalence check depends on this). torch's default CPU
    generator returns the same sequence for the same seed whether through a `Generator` or the
    global seed, so the reference side calls `torch.manual_seed(_noise_seed(P))` before running the
    original forward.
    """
    generator = torch.Generator().manual_seed(_noise_seed(length))
    return torch.randn(1, 2, length, generator=generator) * NOISE_SCALE


def _noise_seed(length: int) -> int:
    """The seed of `make_noise` (the same value the reference side uses as the global seed)."""
    return SEED + 7 * length


def make_phones(length: int) -> torch.Tensor:
    """Phoneme IDs `x` `[1, P]` (i64). 0 is `_` (the silence symbol), so they are drawn from 1
    up."""
    from style_bert_vits2.nlp.symbols import SYMBOLS

    return torch.randint(1, len(SYMBOLS), (1, length), generator=_generator(length, 0, 1))


def make_tones(length: int) -> torch.Tensor:
    """Accent IDs `tone` `[1, P]` (i64)."""
    from style_bert_vits2.nlp.symbols import NUM_TONES

    return torch.randint(0, NUM_TONES, (1, length), generator=_generator(length, 0, 2))


def make_language(length: int) -> torch.Tensor:
    """Language IDs `language` `[1, P]` (i64). JP-Extra is Japanese only, so all 0."""
    return torch.zeros((1, length), dtype=torch.int64)


def make_bert(length: int) -> torch.Tensor:
    """BERT features `bert` `[1, 1024, P]`.

    NOTE: synthetic randn values rather than real DeBERTa hidden states (building a real hidden
    state needs the 1.3GB of `outputs/series/deberta` and a phoneme ↔ character alignment, far from
    the purpose of front's numeric verification). bert_proj is a single 1x1 conv followed by
    LayerNorms, so a different value range only scales h uniformly and does not change how the error
    grows.
    """
    return torch.randn(1, 1024, length, generator=_generator(length, 0, 3))


def build_cases(
    g: torch.Tensor,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
) -> tuple[tuple[str, dict[str, torch.Tensor]], ...]:
    """Golden cases of dp `(name, {input name: tensor})`. The keys match the graph input names."""
    return tuple(
        (name, {"h": make_hidden(length), "x_mask": make_mask(length, pad), "g": g})
        for name, length, pad in cases
    )


def front_inputs(length: int, pad: int, g: torch.Tensor, style: torch.Tensor) -> dict[str, Any]:
    """The full input set of front (the graph input names + `x_lengths`, which the reference
    implementation needs).

    x / tone / bert **get values in the padding columns too** (the same reason as dp's h — if the
    mask path takes effect, the end of the output is exactly 0).
    """
    return {
        "x": make_phones(length),
        "x_mask": make_mask(length, pad),
        "tone": make_tones(length),
        "language": make_language(length),
        "bert": make_bert(length),
        "style_vec": style,
        "g": g,
        "z_noise": make_noise(length),
        "x_lengths": torch.tensor([length - pad], dtype=torch.int64),
    }


def make_latent(length: int, salt: int) -> torch.Tensor:
    """The latent sequence `[1, 192, T]` fed to flow / dec (a separate fixed seed per case and
    purpose).

    In the real chain `z_p` is `m_p + randn·exp(logs_p)·noise_scale`, whose value range is a normal
    distribution of O(1) itself. That is why synthetic randn is used (unlike front's `bert`, it does
    not stray far from the real distribution).
    """
    return torch.randn(1, 192, length, generator=_generator(length, 0, salt))


def flow_inputs(length: int, pad: int, g: torch.Tensor) -> dict[str, torch.Tensor]:
    """The full input set of flow / voice (keyed by graph input name).

    `z_p` **gets values in the padding columns too** (the same reason as dp / front — if the
    `y_mask` multiplication takes effect, the end of the output is exactly 0, and if not, values
    leak). The tables `idx_k` / `valid` come from `sbv2.patch.build_relattn_tables` (the
    Python-side source of truth mirrored by the host TS).
    """
    idx_k, valid = patch.build_relattn_tables(length, EXPECTED_WINDOW_SIZE)
    return {
        "z_p": make_latent(length, salt=4),
        "y_mask": make_mask(length, pad),
        "g": g,
        "idx_k": idx_k,
        "valid": valid,
    }


def dec_inputs(length: int, pad: int, g: torch.Tensor) -> dict[str, torch.Tensor]:
    """The full input set of dec.

    dec has **no mask input** (`Generator.forward(x, g)`), so the `padded` case zeroes the end **on
    the input side**, as in the real chain (the same form as the fused voice calling
    `dec(z * y_mask, g)`). dec alone has no mask path, so all this case exercises is "an input
    ending in 0 still gives the same numbers as upstream"; it is not a leak detector like front's /
    flow's padded — voice holds the detector.
    """
    return {"x": make_latent(length, salt=5) * make_mask(length, pad), "g": g}


def build_flow_cases(
    g: torch.Tensor,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
) -> tuple[tuple[str, dict[str, torch.Tensor]], ...]:
    """Golden cases of flow / voice (shared inputs — they differ only in whether dec is fused)."""
    return tuple((name, flow_inputs(length, pad, g)) for name, length, pad in cases)


def build_dec_cases(
    g: torch.Tensor,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
) -> tuple[tuple[str, dict[str, torch.Tensor]], ...]:
    """Golden cases of dec."""
    return tuple((name, dec_inputs(length, pad, g)) for name, length, pad in cases)


def build_front_cases(
    g: torch.Tensor,
    style: torch.Tensor,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
) -> tuple[tuple[str, dict[str, torch.Tensor]], ...]:
    """Golden cases of front. `x_lengths` is only for the reference, so it is not in the golden."""
    return tuple(
        (
            name,
            {
                key: value
                for key, value in front_inputs(length, pad, g, style).items()
                if key != "x_lengths"
            },
        )
        for name, length, pad in cases
    )


def _write_io(
    module: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, dict[str, torch.Tensor]]],
    out_dir: Path,
) -> list[str]:
    """Writes each case's inputs and torch CPU expected outputs to `io.<case>.safetensors`.

    forward is called in **the declaration order of the graph inputs** (IR input names = forward
    argument names, so a single mismatched key of the case dict fails with KeyError — closing off
    the form where names drift silently and values flow into other arguments).
    """
    written: list[str] = []
    for name, args in cases:
        with torch.no_grad():
            output = module(*(args[declared.name] for declared in graph.inputs))
        outputs = (output,) if isinstance(output, torch.Tensor) else tuple(output)
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{name}: {len(outputs)} eager outputs differ from {len(graph.outputs)} IR outputs"
            )
        # MUST: io is also reduced to the concrete representation of the IR's semantic dtype (the
        # boundary normalization of ADR 0009).
        tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name}: input '{declared.name}'"
            )
            for declared in graph.inputs
        }
        for index, value in enumerate(outputs):
            tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
                value.detach().contiguous(), f"{name}: output {index}"
            )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def _module_name(weight_fqn: str) -> str:
    """The weight FQN returned by `iter_quant_targets` (`<module>.weight` / `weight` for the root)
    → the module FQN.

    The `include` predicate (`fake_quant_int8` / `fake_quant_int4`) looks at **module** FQNs, while
    the ledger keys are **weight** FQNs — mixing up the 2 spaces makes the predicate match nothing.
    A single `removesuffix` does not do it because only the root module (`""`) has no `.`.
    """
    return weight_fqn[: -len(".weight")] if weight_fqn.endswith(".weight") else ""


def _i4_module_names(module: nn.Module) -> frozenset[str]:
    """The set of FQNs of modules rounded to i4 group32 (the only source of the **exclusive split**
    of the mix).

    Eligibility is the conjunction of 3 conditions: ① the type is one of {@link I4_MODULE_TYPES}
    (emit's i4 eligibility = limited to the weight slots of `emit.I4_WEIGHT_OPS` — ADR 0069 decision
    5 and its conv1d addendum; explicitly assigning i4 to a tensor outside it makes emit fail
    loudly) ② `groups == 1` (conv1d in i4 can be expanded only by the igemm variant; depthwise
    (`convs_sep` of `DDSConv`) uses the direct kernel = no i4 support) ③ the row length after
    flattening is divisible by the group length (i4 MUST NOT make partial groups — decision 2 of the
    same ADR; a weight that fails this is **dropped from the targets rather than failing the whole
    configuration** (the same treatment as `census_w4_targets` of `measure_quant`), so ineligible
    linear / conv1d drop to the i8 side).

    The targets are enumerated through the core (`iter_quant_targets`) so that the set the rounding
    looks at and the set counted here stay one implementation. The i8-side predicate is written as
    "not in this set" for the same reason: building the 2 predicates from separate spellings would
    leave weights in neither **silently in f32** (the hole on the opposite side of the no-double-
    rounding rule, where the values stay correct and only the size goes back).

    NOTE: `nn.ConvTranspose1d` does not derive from `nn.Conv1d`, so it drops out naturally by type
    (`ups` of `dec` — a weight whose row axis is not the first does not match the pack order and
    cannot be i4).
    """
    modules = dict(module.named_modules())
    names: set[str] = set()
    for fqn, weight, axis in iter_quant_targets(module, I4_MODULE_TYPES):
        name = _module_name(fqn)
        if getattr(modules[name], "groups", 1) != 1:
            continue
        if channel_rows(weight, axis).shape[-1] % DEFAULT_GROUP_SIZE == 0:
            names.add(name)
    return frozenset(names)


def _fake_quant(
    dtype: str, module: nn.Module, target: str
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """Rounds **the effective weights** to values representable in the storage dtype, and returns
    the scale ledger and the per-weight storage assignments.

    The fake-quant of ADR 0006 / 0018 (f16) / 0019 (i8) / 0069 (i4). The ledger keys are in-model
    FQNs, which the emit side matches as safetensors tensor keys (matching by `id()` is forbidden —
    ADR 0006).

    The i4 series is mixed (eligible linear / conv1d = i4 group32, the rest = i8 per-channel), and
    the 2 predicates are split **exclusively** from {@link _i4_module_names} (the mixed MUST of
    `quantize.py`). The returned override is the mapping "assign every key of the i4 scale ledger to
    i4", and the emit side fails loudly if it cannot honour an explicit assignment
    (`emit._plan_i4` — there is no path that drops silently to i8).

    MUST: it is applied to **the `nn.Module` being exported itself** (not to all of `net_g`).
    Moving weights that do not appear in that target's graph would muddy the premise when another
    target is taken from the same net_g.

    MUST (order): ① call it **after** `remove_weight_norm` / patching. Rounding before remove would
    round `weight_g` / `weight_v`, and the effective weights built from them would not sit on the
    f16 / i8 grid (exactly the ordering constraint of `ensure_dec_plain`). i8 is **even stricter**:
    if discarded elements affect amax, the per-channel scale itself shifts. ② call it **before**
    taking references and goldens (the MUST of the `quantize` module — applied afterwards, only the
    goldens would be computed with the original weights, quantization error would mix into the E2E
    difference, and the tolerance would lose its meaning).

    Each of the 5 targets builds a new net_g with `load_net_g`, so rounding never leaks across
    targets. voice bundles flow and dec in one wrapper, so this single call covers both (rounding is
    idempotent for f16 and i8 alike — for i8, quantization closed under ±127 is a fixed point in f32
    (ADR 0019). What it depends on, though, is not idempotence but the order above, "call it after
    the effective weights are settled"; idempotence on the real weights is pinned by
    `sbv2/tests/test_export.py`).

    NOTE: `g` (the speaker embedding of `emb_g`) and `style_vec` are **graph inputs** not contained
    in the exported module, so they are not rounded — inputs are where the reference and the GPU
    read the same bytes, unaffected by the storage dtype. front's baked relative-position tables are
    not module weights either, so they are not touched (graph constants are ineligible = f32 storage
    — the eligibility rule of emit.py).
    """
    if dtype == "f32":
        return {}, {}
    if dtype == "i8":
        int8 = fake_quant_int8(module)
        print(
            f"[fake-quant] {target}: rounded to i8 per-channel — {int8.describe()}",
            flush=True,
        )
        return int8.scales, {}
    if dtype == "f16":
        report = round_weights_to_f16(module)
        print(
            f"[fake-quant] {target}: rounded to f16-representable values — {report.describe()}",
            flush=True,
        )
        return {}, {}
    i4_names = _i4_module_names(module)
    i8_names = {
        _module_name(fqn) for fqn, _, _ in iter_quant_targets(module, QUANT_MODULE_TYPES)
    } - i4_names
    int4 = fake_quant_int4(
        module, DEFAULT_GROUP_SIZE, include=lambda name: name in i4_names, op_types=I4_MODULE_TYPES
    )
    # MUST: round only **after enumerating both sides** of the exclusive split. Writing the i8 side
    # just as the negation "not in i4" would, for a series where every quantization target is
    # i4-eligible (`flow` consists only of conv1d, all with groups == 1 and divisible), trigger the
    # "0 targets" of `fake_quant_int8` and fail the export — since the 0 is **the result of a
    # count**, it is not silent, so the rounding is skipped altogether.
    i8_scales: Mapping[str, torch.Tensor] = {}
    if i8_names:
        rest = fake_quant_int8(module, include=lambda name: name in i8_names)
        i8_scales = rest.scales
        remainder = f"the rest in i8 per-channel — {rest.describe()}"
    else:
        remainder = "no i8 targets (every quantization target is i4-eligible)"
    print(
        f"[fake-quant] {target}: rounded eligible linear / conv1d to i4 groups —"
        f" {int4.describe()} / {remainder}",
        flush=True,
    )
    return {**i8_scales, **int4.scales}, dict.fromkeys(int4.scales, "i4")


@contextmanager
def _staged_target(target: str, out_dir: Path, sym_max: int) -> Iterator[Path]:
    """Hands over the staging directory of one target and publishes it to `out_dir` once every
    check has passed.

    MUST: the output is written to the staging directory and published only after passing **every
    check** (output count, boundary normalization). Placing it in the final location before the
    checks would let a failed run leave "assets that can pass the acceptance check" — the io goldens
    come from the same broken module, so they agree with each other and the TS-side comparison
    **goes green** (the spelling of "when publishing is allowed" is in
    {@link _shared.decode_series._publish}; the discipline of swapping in and cleaning up is the
    core primitive {@link karume.artifacts.staged_publication}).

    The provenance record ({@link _write_export_provenance}) is written into the same directory
    right before leaving it — the swap happens once, so a new container can never pair with an old
    record.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        # The writer creates the directory (the primitive creates nothing — it only hands over a
        # path).
        staged.mkdir()
        yield staged
        _write_export_provenance(target, sym_max, staged)


def _write_export_provenance(target: str, sym_max: int, out_dir: Path) -> None:
    """Records in the series **with which symbolic-dimension limit** this target was baked.

    `--sym-max` accepts any value, for a single target only (for research — {@link TARGET_SYM_MAX}).
    That value becomes the max of `Dim` and, as is, the static dimension of the baked constants, so
    a series written away from the default has a "limit the graph accepts" different from the
    default. But the distribution side's `pipelineConfig.maxTokens` / `maxFrames` are baked as
    constants (`sbv2/distribution.py`), so a mismatch leaves **export green and distribution
    green**, and the consumer is the first to hit Tmax exceeded in `sym_prefix_slice`. So the
    writing side records the fact, and {@link sbv2.distribution.assert_sym_provenance} compares it
    at assembly time (the same discipline as `--lora` / `calib_provenance.json`: "the same fact held
    by separate scripts is always compared at assembly time").

    MUST: it is written **inside the staging directory** (inside the `with` of
    `staged_publication`) — the swap happens once, so a new container can never pair with an old
    record.
    """
    record = {"target": target, "sym_max": sym_max}
    (out_dir / EXPORT_PROVENANCE_FILE).write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _summary(
    target: str,
    out_dir: Path,
    hps: Any,
    sym_max: int,
    graph: IrGraph,
    written: Sequence[str],
    cases: Sequence[tuple[str, int, int]],
    elapsed: float,
    dtype: str,
) -> dict[str, Any]:
    # Eligible = weights kept resident on the GPU compressed / ineligible = f32 storage (the
    # standing diagnostic of ADR 0006).
    breakdown = storage_breakdown(graph)
    return {
        "target": target,
        "dir": str(out_dir),
        "dtype": dtype,
        "version": hps.version,
        "sym_max": sym_max,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        # The companion scale of i8 (ADR 0019). 0 for f32 / f16.
        "scale_bytes": breakdown.scale_bytes,
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": list(written),
        "case_lengths": {name: length for name, length, _ in cases},
        "seconds": round(elapsed, 1),
    }


def export_dp(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """Writes the IR container and golden io of dp and returns a summary.

    MUST: the output is written to the staging directory and published only after passing every
    check (output count, boundary normalization) ({@link _staged_target} spells out why).
    """
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    module = DurationPredictorGraph(net_g.dp)
    # There is no preprocessing (neither patches nor remove_weight_norm), so this is "after the
    # effective weights are settled".
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_DP)
    g = speaker_embedding(net_g)
    built = build_cases(g, cases)

    # The example input is the padded case (an x_mask containing 0). min=2 avoids the 0/1
    # specialization, max is the declaration itself (ADR 0010 — the evaluation point of constant
    # folding is taken from range_constraints).
    example = dict(built[-1][1])
    phonemes = Dim("P", min=2, max=sym_max)
    with _staged_target(TARGET_DP, out_dir, sym_max) as staged:
        graph = export_to_file(
            module,
            (example["h"], example["x_mask"], example["g"]),
            staged / MODEL_FILE,
            provenance=sbv2_provenance(model_dir),
            # The graph name is **the component name** (= the key of weights in karume.json). It is
            # not derived from the directory name (container-v1 §2.1) — the staging directory is
            # `<target>.staging/`.
            graph_name=TARGET_DP,
            dynamic_shapes=({2: phonemes}, {2: phonemes}, {}),
            symbol_names=("P",),
            weight_dtype=BASE_WEIGHT_DTYPES[dtype],
            weight_scales=scales,
            weight_dtype_overrides=dtype_overrides,
        )
        written = _write_io(module, graph, built, staged)
    return _summary(
        TARGET_DP,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def export_front(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """Writes the IR container and golden io of front (the fused graph of enc_p + dp + sdp
    reverse).

    The patch layer is applied here — the golden is the eager output **after patching** (= the
    ground truth of what the IR must compute), and the difference from the reference implementation
    before patching is measured by `--verify` in a separate process: a two-layer structure (ADR
    0013).

    MUST: the output is written to the staging directory and published only after passing every
    check ({@link _staged_target}).
    """
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    patch.apply_all_patches()
    module = patch.Sbv2Front(net_g)
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_FRONT)
    built = build_front_cases(speaker_embedding(net_g), style_vector(model_dir), cases)

    # The example input is the padded case (an x_mask containing 0).
    example = dict(built[-1][1])
    phonemes = Dim("P", min=2, max=sym_max)
    # Dynamic axes: x / tone / language are [1,P], x_mask / bert / z_noise are [...,P].
    # style_vec [1,256] and g [1,512,1] do not depend on P.
    dynamic_shapes = (
        {1: phonemes},
        {2: phonemes},
        {1: phonemes},
        {1: phonemes},
        {2: phonemes},
        {},
        {},
        {2: phonemes},
    )
    with _staged_target(TARGET_FRONT, out_dir, sym_max) as staged:
        graph = export_to_file(
            module,
            tuple(example[declared] for declared in FRONT_INPUT_ORDER),
            staged / MODEL_FILE,
            provenance=sbv2_provenance(model_dir),
            # The graph name is **the component name** (= the key of weights in karume.json). It is
            # not derived from the directory name (container-v1 §2.1) — the staging directory is
            # `<target>.staging/`.
            graph_name=TARGET_FRONT,
            dynamic_shapes=dynamic_shapes,
            symbol_names=("P",),
            weight_dtype=BASE_WEIGHT_DTYPES[dtype],
            weight_scales=scales,
            weight_dtype_overrides=dtype_overrides,
        )
        written = _write_io(module, graph, built, staged)
    return _summary(
        TARGET_FRONT,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def ensure_dec_plain(net_g: nn.Module) -> None:
    """Removes dec's weight_norm and folds it into the effective weights (idempotent).

    While weight_norm remains, `Conv1d.weight` is **not the effective weight** (a forward pre-hook
    builds it from `weight_g` / `weight_v` every time). Writing it into the IR as is would give a
    different model, so this always runs before export. It is idempotent because the fused voice
    shares the same net_g with dec (applying it twice makes `remove_weight_norm` fail with
    RuntimeError).

    MUST (ordering constraint — ADR 0013 / 0018): weight rounding (`_fake_quant`) is applied to
    **the effective weights after remove**. Rounding before remove would round
    `weight_g`/`weight_v`, and the effective weights would not sit on the rounding grid (the
    download shrinks, but the numbers become something else). The exports of dec / voice are both
    arranged to call `_fake_quant` **after** this function.

    Numeric equivalence (before and after remove) is measured by `--verify dec` on the real weights
    for every case.
    """
    if any(hasattr(module, "weight_g") for module in net_g.dec.modules()):
        net_g.dec.remove_weight_norm()
    _assert_no_weight_norm(net_g.dec)


def export_flow(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = FLOW_SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """Writes the IR container and golden io of flow (TransformerCouplingBlock reverse).

    The `(T,T)` tables of the relative-position attention are **graph inputs** (`idx_k` /
    `valid`). It differs from front's baking because sym_max is orders of magnitude larger (O(T²)
    at 4096 = 134MB — ADR 0013).

    MUST: the output is written to the staging directory and published only after passing every
    check ({@link _staged_target}).
    """
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    patch.apply_all_patches()
    module = patch.FlowReverse(net_g)
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_FLOW)
    built = build_flow_cases(speaker_embedding(net_g), cases)

    example = dict(built[-1][1])  # the padded case (a y_mask containing 0)
    frames = Dim("T", min=2, max=sym_max)
    # z_p [1,192,T] / y_mask [1,1,T] / g [1,512,1] / idx_k [T,T] / valid [T,T].
    dynamic_shapes = (
        {2: frames},
        {2: frames},
        {},
        {0: frames, 1: frames},
        {0: frames, 1: frames},
    )
    with _staged_target(TARGET_FLOW, out_dir, sym_max) as staged:
        graph = export_to_file(
            module,
            tuple(example[declared] for declared in FLOW_INPUT_ORDER),
            staged / MODEL_FILE,
            provenance=sbv2_provenance(model_dir),
            # The graph name is **the component name** (= the key of weights in karume.json). It is
            # not derived from the directory name (container-v1 §2.1) — the staging directory is
            # `<target>.staging/`.
            graph_name=TARGET_FLOW,
            dynamic_shapes=dynamic_shapes,
            symbol_names=("T",),
            weight_dtype=BASE_WEIGHT_DTYPES[dtype],
            weight_scales=scales,
            weight_dtype_overrides=dtype_overrides,
        )
        written = _write_io(module, graph, built, staged)
    return _summary(
        TARGET_FLOW,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def export_dec(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = FLOW_SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """Writes the IR container and golden io of dec (HiFi-GAN Generator).

    No patch layer is needed (it goes through neither attention nor spline). The only preprocessing
    is `remove_weight_norm`, and there is no wrapper either — the argument names of
    `Generator.forward(x, g)` become the IR input names as is.

    MUST: the output is written to the staging directory and published only after passing every
    check ({@link _staged_target}).
    """
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    ensure_dec_plain(net_g)
    module = net_g.dec
    # MUST: **after** remove_weight_norm (the ordering constraint announced by `ensure_dec_plain`).
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_DEC)
    built = build_dec_cases(speaker_embedding(net_g), cases)

    example = dict(built[-1][1])
    frames = Dim("T", min=2, max=sym_max)
    with _staged_target(TARGET_DEC, out_dir, sym_max) as staged:
        graph = export_to_file(
            module,
            tuple(example[declared] for declared in DEC_INPUT_ORDER),
            staged / MODEL_FILE,
            provenance=sbv2_provenance(model_dir),
            # The graph name is **the component name** (= the key of weights in karume.json). It is
            # not derived from the directory name (container-v1 §2.1) — the staging directory is
            # `<target>.staging/`.
            graph_name=TARGET_DEC,
            dynamic_shapes=({2: frames}, {}),
            symbol_names=("T",),
            weight_dtype=BASE_WEIGHT_DTYPES[dtype],
            weight_scales=scales,
            weight_dtype_overrides=dtype_overrides,
        )
        written = _write_io(module, graph, built, staged)
    return _summary(
        TARGET_DEC,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def export_voice(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = FLOW_SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """Writes the IR container and golden io of voice (flow reverse + dec fused into one graph).

    **Once this target's E2E is green, the whole SBV2 chain holds** (front produces the durations,
    the host builds z_p, and this turns it into a waveform). The gain of fusing is removing the
    readback round trip of the intermediate z; in exchange, comparing z for debugging is possible
    only on the standalone flow.

    MUST: the output is written to the staging directory and published only after passing every
    check ({@link _staged_target}).
    """
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    patch.apply_all_patches()
    ensure_dec_plain(net_g)
    module = patch.Sbv2Voice(net_g)
    # MUST: **after** remove_weight_norm (it contains dec, so the same ordering constraint as dec
    # alone).
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_VOICE)
    built = build_flow_cases(speaker_embedding(net_g), cases)

    example = dict(built[-1][1])
    frames = Dim("T", min=2, max=sym_max)
    dynamic_shapes = (
        {2: frames},
        {2: frames},
        {},
        {0: frames, 1: frames},
        {0: frames, 1: frames},
    )
    with _staged_target(TARGET_VOICE, out_dir, sym_max) as staged:
        graph = export_to_file(
            module,
            tuple(example[declared] for declared in FLOW_INPUT_ORDER),
            staged / MODEL_FILE,
            provenance=sbv2_provenance(model_dir),
            # The graph name is **the component name** (= the key of weights in karume.json). It is
            # not derived from the directory name (container-v1 §2.1) — the staging directory is
            # `<target>.staging/`.
            graph_name=TARGET_VOICE,
            dynamic_shapes=dynamic_shapes,
            symbol_names=("T",),
            weight_dtype=BASE_WEIGHT_DTYPES[dtype],
            weight_scales=scales,
            weight_dtype_overrides=dtype_overrides,
        )
        written = _write_io(module, graph, built, staged)
    return _summary(
        TARGET_VOICE,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def reference_front_outputs(net_g: nn.Module, inputs: dict[str, Any]) -> list[torch.Tensor]:
    """Gets the 4 outputs of front from the reference implementation **before patching**.

    MUST: callable only before patching. Called after patching, the reference itself would be the
    patched value, and the equivalence check would become vacuous and give a false PASS (ADR 0013).
    """
    _assert_patches_not_applied("taking the front reference")
    with torch.no_grad():
        h, m_p, logs_p, x_mask = net_g.enc_p(
            inputs["x"],
            inputs["x_lengths"],
            inputs["tone"],
            inputs["language"],
            inputs["bert"],
            inputs["style_vec"],
            g=inputs["g"],
        )
        # The reference implementation builds x_mask from x_lengths. If it disagreed with the
        # x_mask passed from outside, it would no longer be "a comparison on the same input", so it
        # is pinned here.
        if not torch.equal(x_mask, inputs["x_mask"]):
            raise AssertionError(
                "the x_mask built by the reference implementation differs from the golden x_mask"
            )
        # The reference's reverse path draws randn exactly once internally. Putting the same seed
        # into the global state gives the same sequence as make_noise (through a Generator) (the
        # default CPU generator is the same).
        torch.manual_seed(_noise_seed(int(inputs["x"].shape[1])))
        logw_sdp = net_g.sdp(h, x_mask, g=inputs["g"], reverse=True, noise_scale=NOISE_SCALE)
        logw_dp = net_g.dp(h, x_mask, g=inputs["g"])
    return [logw_sdp, logw_dp, m_p, logs_p]


def _diff_entry(
    length: int,
    pad: int,
    names: Sequence[str],
    got: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """The equivalence report for one case (output name → maxdiff, and bit identity of all
    outputs).

    `bit_exact` is a stronger claim than "difference 0" (`0.0 == -0.0` has difference 0 but
    different bits), and it matters for verifications where **bit identity is the claim**, such as
    remove_weight_norm.
    """
    return {
        "length": length,
        "pad": pad,
        "maxdiff": {
            name: float((a - b).abs().max())
            for name, a, b in zip(names, got, expected, strict=True)
        },
        "bit_exact": all(torch.equal(a, b) for a, b in zip(got, expected, strict=True)),
    }


def verify_front(
    model_dir: Path, *, cases: Sequence[tuple[int, int]] = VERIFY_CASES
) -> list[dict[str, Any]]:
    """Measures the eager equivalence of front before and after patching, on the real weights.

    MUST: the order is "**settle the references of every case** → patch → compare". The patches
    replace class attributes process-wide, so running "reference → compare" one case at a time would
    make the references from the second case on the patched values (a vacuous check whose
    difference is always 0).
    """
    net_g, _hps = load_net_g(model_dir)
    g = speaker_embedding(net_g)
    style = style_vector(model_dir)

    references = {
        (length, pad): reference_front_outputs(net_g, front_inputs(length, pad, g, style))
        for length, pad in cases
    }

    patch.apply_all_patches()
    front = patch.Sbv2Front(net_g)

    report: list[dict[str, Any]] = []
    for length, pad in cases:
        inputs = front_inputs(length, pad, g, style)
        with torch.no_grad():
            outputs = front(*(inputs[name] for name in FRONT_INPUT_ORDER))
        report.append(
            _diff_entry(length, pad, FRONT_OUTPUT_NAMES, outputs, references[(length, pad)])
        )
    return report


def verify_flow(
    model_dir: Path, *, cases: Sequence[tuple[int, int]] = FLOW_VERIFY_CASES
) -> list[dict[str, Any]]:
    """flow: equivalence of the table-input `FlowReverse` with the reference
    `TransformerCouplingBlock(reverse=True)`.

    MUST: the same ordering constraint as front (take every reference, then patch). The MHA patch
    replaces class attributes, so it also takes effect on the attention layers of `net_g.flow` at
    the same time.
    """
    net_g, _hps = load_net_g(model_dir)
    g = speaker_embedding(net_g)

    _assert_patches_not_applied("taking the flow reference")
    references = {}
    for length, pad in cases:
        inputs = flow_inputs(length, pad, g)
        with torch.no_grad():
            references[(length, pad)] = net_g.flow(
                inputs["z_p"], inputs["y_mask"], g=inputs["g"], reverse=True
            )

    patch.apply_all_patches()
    flow_rev = patch.FlowReverse(net_g)

    report: list[dict[str, Any]] = []
    for length, pad in cases:
        inputs = flow_inputs(length, pad, g)
        with torch.no_grad():
            got = flow_rev(*(inputs[name] for name in FLOW_INPUT_ORDER))
        report.append(_diff_entry(length, pad, ("z",), (got,), (references[(length, pad)],)))
    return report


def verify_dec(
    model_dir: Path, *, cases: Sequence[tuple[int, int]] = FLOW_VERIFY_CASES
) -> list[dict[str, Any]]:
    """dec: eager equivalence **before and after** `remove_weight_norm` (reference = the original
    path with weight_norm active).

    The recon measured bit identity on only one case (z=(1,192,50)), and nothing in the spec
    guarantees that weight_norm's effective weight `g · v / ‖v‖` is reproduced exactly in f32. This
    compares **every case** to close that unverified point (`bit_exact` is the claim).

    MUST: take the references of every case before remove. `remove_weight_norm` folds the
    parameters destructively, so running one case at a time would make the references from the
    second case on the values after remove.
    """
    net_g, _hps = load_net_g(model_dir)
    g = speaker_embedding(net_g)

    _assert_weight_norm_present(net_g.dec, "taking the dec reference")
    references = {}
    for length, pad in cases:
        inputs = dec_inputs(length, pad, g)
        with torch.no_grad():
            references[(length, pad)] = net_g.dec(inputs["x"], g=inputs["g"])

    ensure_dec_plain(net_g)

    report: list[dict[str, Any]] = []
    for length, pad in cases:
        inputs = dec_inputs(length, pad, g)
        with torch.no_grad():
            got = net_g.dec(inputs["x"], g=inputs["g"])
        report.append(_diff_entry(length, pad, ("audio",), (got,), (references[(length, pad)],)))
    return report


def verify_voice(
    model_dir: Path, *, cases: Sequence[tuple[int, int]] = FLOW_VERIFY_CASES
) -> list[dict[str, Any]]:
    """voice: equivalence of the fused wrapper with the reference chain (unpatched flow + dec with
    weight_norm active).

    It spans the sources of contamination of **both** flow and dec, so no other verification runs
    in this process (the CLI enforces one verification per process).
    """
    net_g, _hps = load_net_g(model_dir)
    g = speaker_embedding(net_g)

    # There are 2 sources of contamination (patches and remove), so there are 2 checks too. With
    # only one, an ordering violation of the other would pass as "a false PASS with difference 0".
    _assert_patches_not_applied("taking the voice reference")
    _assert_weight_norm_present(net_g.dec, "taking the voice reference")
    references = {}
    for length, pad in cases:
        inputs = flow_inputs(length, pad, g)
        with torch.no_grad():
            z = net_g.flow(inputs["z_p"], inputs["y_mask"], g=inputs["g"], reverse=True)
            references[(length, pad)] = net_g.dec(z * inputs["y_mask"], g=inputs["g"])

    patch.apply_all_patches()
    ensure_dec_plain(net_g)
    voice = patch.Sbv2Voice(net_g)

    report: list[dict[str, Any]] = []
    for length, pad in cases:
        inputs = flow_inputs(length, pad, g)
        with torch.no_grad():
            got = voice(*(inputs[name] for name in FLOW_INPUT_ORDER))
        report.append(_diff_entry(length, pad, ("audio",), (got,), (references[(length, pad)],)))
    return report


EXPORTERS = {
    TARGET_DP: export_dp,
    TARGET_FRONT: export_front,
    TARGET_FLOW: export_flow,
    TARGET_DEC: export_dec,
    TARGET_VOICE: export_voice,
}

#: `--verify <target>` → verification function. **There is none for dp** — it goes through neither
#: patches nor preprocessing, so "reference" and "subject" would be the same module and the
#: comparison would be vacuous.
VERIFIERS = {
    TARGET_FRONT: verify_front,
    TARGET_FLOW: verify_flow,
    TARGET_DEC: verify_dec,
    TARGET_VOICE: verify_voice,
}

#: The symbol names shown in the verification report (P = phoneme count for the front group, T =
#: frame count for the flow group).
VERIFY_SYMBOL = {
    TARGET_FRONT: "P",
    TARGET_FLOW: "T",
    TARGET_DEC: "T",
    TARGET_VOICE: "T",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: the series of each --dtype —"
        " outputs/series/sbv2-<directory name of --model-dir>{,-f16,-i8,-i4}/)",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help="storage dtype of the weights (f16 / i8 fake-quantize, then store only the eligible"
        " slots compressed — ADR 0018 / 0019; i4 is mixed: only eligible linear / conv1d in"
        " group32 i4, the rest i8 — ADR 0069. **Emit-only**,"
        " cannot be combined with --verify)",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        default=None,
        help=f"target to emit (repeatable; default: all of {', '.join(TARGETS)})",
    )
    parser.add_argument(
        "--sym-max",
        type=int,
        default=None,
        help=(
            "limit of the symbolic dimension (default per target: "
            + " / ".join(f"{name}={value}" for name, value in TARGET_SYM_MAX.items())
            + "). A deviation is allowed only with a single --target"
        ),
    )
    parser.add_argument(
        "--verify",
        choices=sorted(VERIFIERS),
        default=None,
        help="verify eager equivalence with the reference implementation (no emit; one target per"
        " process)",
    )
    args = parser.parse_args(argv)

    # MUST: reject combining them in one process, mechanically. emit applies the patches and folds
    # weight_norm, so a "before" reference taken afterwards would already be contaminated, and the
    # equivalence check would become vacuous and give a false PASS (ADR 0013). The order is not
    # left as a discipline to observe inside main — a discipline can be broken, failing here
    # cannot.
    if args.verify is not None and args.target is not None:
        parser.error(
            "--verify and --target cannot be combined in one process"
            " (the process-wide patch replacement contaminates the reference and makes the"
            " equivalence check vacuous)"
        )

    if args.verify is not None:
        if args.sym_max is not None:
            parser.error(
                "--sym-max is emit-only (verification runs eagerly and has no symbolic dimensions)"
            )
        # MUST: --dtype is emit-only too. Verification is an eager comparison that never looks at
        # the storage format, and adding rounding does not change the claim (equivalence before
        # and after patching / bit identity before and after remove). Worse, for dec / voice it
        # **cannot be applied** — rounding can only go after remove while the reference is taken
        # before remove, so it would compare rounded against unrounded and bit_exact would always
        # drop to False (making it impossible to tell a broken check from broken assets).
        if args.dtype != "f32":
            parser.error(
                f"--dtype {args.dtype} is emit-only (cannot be combined with --verify) —"
                " verification is an eager comparison that ignores the storage format, and for"
                " dec / voice rounding can only be applied after remove_weight_norm while the"
                " reference is taken before remove, "
                "so it would compare rounded against unrounded and break the bit_exact claim"
            )
        report = VERIFIERS[args.verify](args.model_dir)
        symbol = VERIFY_SYMBOL[args.verify]
        for entry in report:
            diffs = " ".join(f"{k}={v:.3e}" for k, v in entry["maxdiff"].items())
            print(
                f"{symbol}={entry['length']:4d} pad={entry['pad']}:"
                f" {diffs} bit_exact={entry['bit_exact']}"
            )
        worst = max(max(entry["maxdiff"].values()) for entry in report)
        print(f"worst maxdiff = {worst:.3e}")
        print(f"bit_exact all = {all(entry['bit_exact'] for entry in report)}")
        return

    targets = list(dict.fromkeys(args.target if args.target is not None else TARGETS))
    # MUST: handing one --sym-max to several targets always gives a wrong value for one of the
    # front and flow groups, whose defaults differ by orders of magnitude (silently, too — ADR
    # 0013).
    if args.sym_max is not None and len(targets) != 1:
        parser.error("--sym-max can be used only when exactly one --target is given")
    if args.out is None:
        args.out = default_out_root(args.model_dir, args.dtype)

    summaries = [
        EXPORTERS[target](
            args.model_dir,
            args.out / target,
            sym_max=args.sym_max if args.sym_max is not None else TARGET_SYM_MAX[target],
            dtype=args.dtype,
        )
        for target in targets
    ]
    print(json.dumps(summaries, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
