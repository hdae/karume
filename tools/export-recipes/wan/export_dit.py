"""The script that writes the Wan2.1 DiT out as an S-shaped IR (`krm`) + goldens (ADR 0118
decisions 3 / 8, step 2).

The starting point is the diffusers version at the pinned revision (`wan.sources.local_snapshot`).
The graph is {@link wan.dit_patch.WanDitTokens} (entry after patchify, exit before unpatchify, one
dimension symbol: the token length `S`). Storage is the f16 preset (the weights are rounded to
f16-representable values before the reference and the goldens are taken, and only eligible weight
slots are stored in f16 — ADR 0006 / 0018). Besides the weights, the container holds **the
`rope_base` asset** (per-axis base tables from which the host builds the RoPE table — role
`rope-base`, ADR 0109 decision 4).

    uv run --group wan --inexact python -m wan.export_dit            # product graph + goldens
    uv run --group wan --inexact python -m wan.export_dit --layers   # measurement (per-layer exits)
    uv run --group wan --inexact python -m wan.export_dit --verify   # eager equivalence only
    uv run --group wan --inexact python -m wan.export_dit --dtype i8            # i8 series (below)
    uv run --group wan --inexact python -m wan.export_dit --dtype i8 --no-full  # same, no full size
    uv run --group wan --inexact python -m wan.export_dit --dtype i8 --check    # rewrite + compare

Output (default `outputs/series/wan2.1-t2v-1.3b-f16-dyn/transformer/`; with `--dtype i8`
`outputs/series/wan2.1-t2v-1.3b-i8-dyn/transformer/`):

    model-NNNNN-of-NNNNN.krm      weights, constants + the descriptions of the 2 documents + the
                                  `rope_base` asset
    io.<case>.safetensors         the graph inputs (`input.*`) and the patched torch CPU output
                                  (`output.0`)
    reference.<case>.safetensors  from plain upstream diffusers (CPU f32, the same f16-rounded
                                  weights): `latents` / `timestep` (i32) / `output`
                                  (`[1,16,F,H,W]`) / `block.NN` (each block's output
                                  `[1,S,1536]` — only for cases with `CaseSpec.blocks`).
                                  Full-size cases add `output.f64` (upstream's output with
                                  activations also in f64, rounded to f32 — the layer-by-layer
                                  `wan.dit_reference.LayerwiseDit` that makes one block at a
                                  time f64, ADR 0121 decision 7)

`--layers` writes the graph for measurement with per-layer exits added
({@link wan.dit_patch.WanDitTokensLayers}) to `…-f16-dyn-probe/transformer/`. It writes no goldens
(the inputs are the product series' `io.*` and the expected values the `block.NN` of the same
`reference.*` — the inputs and weights are the same).

## Cases (decision 8 and the addendum of 2026-10-02, "separate the band decision from acceptance")

| Role | Latent `[1,16,F,H,W]` | S | timestep | Valid length | seed |
| --- | --- | --: | --: | --: | --: |
| `band` (sets the band) | `[1,16,3,16,16]` | 192 | 999 | 24 | `SEED` + 0 |
| `band` | `[1,16,2,16,24]` | 192 | 750 | 50 | `SEED` + 3 |
| `band` | `[1,16,3,16,16]` | 192 | 500 | 12 | `SEED` + 4 |
| `band` | `[1,16,1,16,48]` | 192 | 250 | 80 | `SEED` + 5 |
| `band` | `[1,16,2,16,24]` | 192 | 600 | 30 | `SEED` + 8 |
| `band` | `[1,16,3,16,16]` | 192 | 113 | 60 | `SEED` + 6 |
| `accept` (acceptance) | `[1,16,2,16,24]` | 192 | 600 | 45 | 777001 |
| `accept` | `[1,16,4,8,16]` | 128 | 30 | 9 | 777002 |
| `accept` | `[1,16,2,24,16]` | 192 | 400 | 70 | 777003 |
| `growth` (growth with S) | `[1,16,3,32,32]` | 768 | 999 | 24 | `SEED` + 2 |
| `growth` | `[1,16,3,32,32]` | 768 | 500 | 37 | `SEED` + 7 |
| `full-band` (sets the full-size band) | `[1,16,9,60,104]` | 14,040 | 999 | 40 | `SEED` + 10 |
| `full-band` (name suffix `-2`) | `[1,16,9,60,104]` | 14,040 | 999 | 112 | `SEED` + 11 |
| `full-band` | `[1,16,9,60,104]` | 14,040 | 750 | 23 | `SEED` + 12 |
| `full-band` | `[1,16,9,60,104]` | 14,040 | 500 | 64 | `SEED` + 9 |
| `full-band` | `[1,16,9,60,104]` | 14,040 | 250 | 91 | `SEED` + 13 |
| `full-band` | `[1,16,9,60,104]` | 14,040 | 113 | 7 | `SEED` + 14 |
| `full-accept` (full-size acceptance) | `[1,16,9,60,104]` | 14,040 | 999 | 28 | 777006 |
| `full-accept` | `[1,16,9,60,104]` | 14,040 | 600 | 77 | 777007 |
| `full-band` (sets the 81-frame band) | `[1,16,21,60,104]` | 32,760 | 999 | 40 | `SEED` + 20 |
| `full-band` (name suffix `-2`) | `[1,16,21,60,104]` | 32,760 | 999 | 112 | `SEED` + 21 |
| `full-band` | `[1,16,21,60,104]` | 32,760 | 750 | 23 | `SEED` + 22 |
| `full-band` | `[1,16,21,60,104]` | 32,760 | 500 | 64 | `SEED` + 23 |
| `full-band` | `[1,16,21,60,104]` | 32,760 | 250 | 91 | `SEED` + 24 |
| `full-band` | `[1,16,21,60,104]` | 32,760 | 113 | 7 | `SEED` + 25 |
| `full-accept` (81-frame acceptance) | `[1,16,21,60,104]` | 32,760 | 999 | 28 | 777008 |
| `full-accept` | `[1,16,21,60,104]` | 32,760 | 600 | 77 | 777009 |

The band metric is "max absolute difference ÷ max absolute value of the reference" (the shape of
the guideline in decision 8), and the band = the worst ratio of the 6 `band` cases × 5
(`DIT_RATIO_BAND` in the TS `e2e_wan_dit_test.ts`). The 3 `accept` cases differ from `band` in seed
and, except for one (600), in timestep too (the one at 600 reproduces as is the unseen case of the
check that exceeded the old band `atol` 3.2e-4 — seed 777001, valid length 45).

MUST: do not add or remove `band` cases after looking at the `accept` results (that breaks the
independence of the band decision from acceptance). If acceptance falls outside the band,
investigate the cause instead of widening the band.

The `full-band` / `full-accept` cases at S = 14,040 derive the band at full size (832×480,
33 frames — ADR 0118 step 3) **independently** of the S = 192 band (the extrapolation discipline of
decision 8). The band = the worst normalized ratio of the 6 `full-band` cases × 5
(`DIT_FULL_NORMALIZED_BAND` on the TS side), accepted on the 2 `full-accept` cases. The band cases
spread their timesteps over 999 (the first generation step — 2 seeds) / 750 / 500 / 250 / 113.
Acceptance is at 999 and 600, and its seeds (777006 / 777007) differ from every step-2 case and
from the band cases. The acceptance seeds were renewed when the metric changed to the normalized
ratio (the old 777004 / 777005 had been seen before the metric was decided, which broke the
independence of acceptance).
MUST: do not change the `full-band` cases or the metric after looking at the `full-accept` results.

The 8 cases at S = 32,760 (832×480, 81 frames = exactly the upper bound of `Dim("S")` — ADR 0118
step 8) derive the band **independently** of the S = 14,040 band too (decision 8 — a band is derived
per S, and the S = 14,040 band is not carried over). Timesteps and valid lengths follow the same
order as the 8 cases at S = 14,040; only the seeds are new (band cases `SEED` + 20–25, acceptance
777008 / 777009 — different from every other case). The shape of the band (the worst normalized
ratio of 6 band cases × 5, 2 acceptance cases) and the MUST above are the same as for S = 14,040.

There are 2 full-size references: **the f64 reference** (`output.f64` — activations also in f64,
weights with the same f16 rounding) is the truth, and the CPU f32 reference (`output`) is used as
the denominator of the normalization. The metric is "the GPU's ratio against f64 ÷ the CPU f32
reference's ratio against f64" (how many times the CPU f32 error the GPU error is). The ratio itself
moves by a factor of 240 with the input (on inputs that amplify rounding the CPU f32 reference also
moves just as far from f64), so dividing by the CPU f32 error cancels the amplification by the
input. f64 takes about 2.7 times as long as f32 (printed in the `[case]` lines — for one S = 14,040
case about 370 s in f64 and 136 s in f32). The attention of the references (f32 / f64) and of the
patched eager is pinned to the CPU flash path (`dit_patch.flash_attention_only` — falling back to
MATH makes one score matrix 51.5 GB at S = 32,760 and OOMs).

Each block's output (`block.NN` — 2.6 GB for one S = 14,040 case) is held by every S = 192 / 768
case and, at full size, only by the 2 cases band `full-band-s14040-t0999` and acceptance
`full-accept-s14040-t0999` (`CaseSpec.blocks` — these 2 suffice for the per-layer record, and the
rest can be judged on the final output alone). No S = 32,760 case holds them: the per-layer exits
they would be compared against (the `--layers` probe) do not fit on the B570 with the readback
staging of 30 blocks (about 6 GB).

The band is not decided on a single case because the error moves by nearly an order of magnitude
with the input (measured: a band decided on one t = 999 case was exceeded 1.6-fold by an unseen
t = 500 case. Mid-range timesteps amplify a 1e-5-level input jitter about 4-fold even in CPU eager —
a sensitivity of the input, not a porting error). The band cases spread their timesteps across the
whole reference schedule (999 → 60) and vary the seed and the grid too. A ratio is used instead of
the max absolute difference because the absolute value, set by a single element in the tail of
12,288 elements, moves with the reference's amplitude.

The grids mix square (F'·H'·W' = 3·8·8) and non-square (2·8·12 / 1·8·24 / 2·12·8 / 4·4·8). A host
implementation that swaps h and w still matches on a square grid, so it first breaks on a
non-square one. The text context is a fixed-seed random `[valid length, 4096]` put into upstream's
shape (zero-padded at the end to 512 rows — `wan.pipeline_ref.pad_text_embeds`).

MUST: take the reference with **upstream's plain forward** (`dit_patch.reference_dit_layers`).
Replacing the attn1 processor does not change the plain path (module docstring of `dit_patch`), so
no check on the order in which the reference is taken is needed.
MUST: the weight rounding (f16 rounding of the f16 series, fake-quant of the i8 series) comes
**before** the reference (ADR 0006 / 0019). The rounding is applied to the S-shaped wrapper and
reaches the shared upstream components with the same values (the patch embedding is a view of the
conv weights — `dit_patch.patch_embedding_linear`). The RoPE table (the buffers of `model.rope`)
is not rounded — the product bakes the f32 base tables, and the reference runs on those values too.
MUST: the writer's eager-equivalence check ({@link eager_failures}) looks at the weights **after**
rounding. The reference and the patched eager share the same rounded weights. The patch's
equivalence (bit-exact trunk, the bound on the patch embedding) is a claim independent of the weight
values (the RoPE rewrite reads no weights), so checking after rounding does not change what the
check means. The same check also pins that the reference was taken with the rounded weights — if
the rounding had not reached the reference side, the trunk would not be bit-identical. The
equivalence before rounding (on the plain f32 weights) is covered by `--verify --no-f16` and by the
pytest real-weight test (`TestRealWeights` in `wan/tests/test_dit_patch.py`).

## i8 series (ADR 0120 decisions 2 / 3)

`--dtype i8` writes the weight series of the reference preset `f16+dit8` and the practical preset
`f16+dit8-a8-attn8-s16`. Only the transformer is covered; the VAE stays in the f16 series
(decision 2).

- Weights: the `weight` of the 307 linears (including the Linear of the patch embedding) is rounded
  with per-channel symmetric i8 RTN (round to nearest) (`karume.quantize.fake_quant_int8` on the
  S-shaped wrapper — {@link fake_quant_i8}). Stored as i8 + an f32 scale per output channel. There
  is no calibration (weights use RTN and activations per-token dynamic quantization at run time).
- Biases, norm weights and `scale_shift_table` stay **upstream's f32 values**. The
  {@link round_to_f16} of the f16 series is not applied (decision 2 — no f16 rounding mixed into
  the i8 preset. The computation and the storage shapes are the same and only the values differ;
  mixing them would add modification notices to the distribution).
- Cases: the S = 192 set of the f16 series (`band` / `accept` — including the one at S = 128) and
  the S = 14,040 set (`full-band` / `full-accept`) are re-taken with the i8 weights at the same
  seeds and timesteps ({@link I8_CASE_ROLES} / {@link I8_FULL_LATENTS}). `growth` (S = 768) and
  S = 32,760 are not taken (decision 3 — the growth at S = 32,760 is covered by the f16 series'
  check).
- References: the check on the normalized ratio r also exists at S = 192, so **every case has an
  f64 reference** (`output.f64`) (in the f16 series only the full-size cases do —
  {@link wants_float64}).
- `--no-full` writes everything except the 8 S = 14,040 cases (for splitting the work into steps —
  S = 14,040 takes about 1.5 hours on CPU). The series is replaced as a whole through the staging
  directory, so a complete series is rewritten by a real run without `--no-full` (the cases'
  random numbers derive from their seeds, so the S = 192 goldens are rewritten with the same
  values).

## Comparison with the existing series (`--check` — ADR 0121 decision 10)

`--check` rewrites the component directory with the same writer into **a temporary directory inside
the series** (`<series>/.check-*/` — placed the same way as umT5's `write --check`), and compares
it with the existing component directory by **the sha256 of every file** (every container part and
`io.*` / `reference.*`) ({@link series_mismatches}). The series is not replaced. On a difference it
names the differing files and exits non-zero (the temporary directory is removed).

What it protects: when the writers shared with Wan2.1 (`dit_patch` / `dit_reference` / this script)
change for Wan2.2, the 1.3B containers and goldens have not moved by a single byte. The writer's
check (eager equivalence) only sees the internal consistency of the rewritten side, so a comparison
with the existing series is needed separately.

- `--dtype i8 --no-full --check` does not write the 8 full-size (S = 14,040) cases and leaves those
  existing files out of the comparison ({@link check_absent} — a left-out case file missing from
  the existing series is named too). Every container part is checked.
- `--layers --check` compares the measurement series (container only).
- Reproducing the values of the full-size goldens is checked separately by
  `wan.dit_reference compare` (re-taken with the layer-by-layer reference and compared tensor by
  tensor).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import AssetInput, Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.emit import storage_breakdown
from karume.pipeline import export_to_file
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
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS, local_snapshot

#: The series (ADR 0118 decision 7 — the `-dyn` suffix is the convention of ADR 0077) and the series
#: of per-layer exits for measurement.
SERIES = SERIES_ROOT / "wan2.1-t2v-1.3b-f16-dyn"
PROBE_SERIES = SERIES_ROOT / "wan2.1-t2v-1.3b-f16-dyn-probe"
#: The i8 series (ADR 0120 decision 2 — transformer only; the spelling follows anima's
#: `<model>-i8-dyn`).
I8_SERIES = SERIES_ROOT / "wan2.1-t2v-1.3b-i8-dyn"

#: The storage dtypes that can be written (= `weight_dtype` of `export_to_file`).
DTYPES = ("f16", "i8")

#: The component name (= the container's graph name = the key under weights of the distribution
#: manifest — container-v1 §2.1).
TARGET = "transformer"
MODEL_FILE = "model.krm"

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
#: The prefix of the `--check` temporary directory (inside the series — the same spelling as umT5's
#: `write --check`).
CHECK_PREFIX = ".check-"

#: The graph input names (the argument names of the wrapper's forward become the IR input names as
#: is).
INPUT_NAMES = ("tokens", "timesteps_proj", "encoder_hidden_states", "rope_cos", "rope_sin")

#: The upper bound of `Dim("S")` = 81 frames (latent `[16,21,60,104]` → 21·30·52). The graph has no
#: constraint on the token length (the RoPE tables were moved to the inputs, so nothing S-dependent
#: is baked in), so this declares the maximum of the accepted set.
DIT_SYM_MAX = 32_760
#: The lower bound likewise (the line that avoids 0 / 1 specialization — the same as Anima's
#: `DIT_SYM_MIN`).
DIT_SYM_MIN = 2

#: The random numbers of the band cases and of the growth record derive from this (independent of
#: the global seed — regeneration is byte-identical). The acceptance cases carry seeds of a separate
#: line (777001 and up) directly.
SEED = 20261002


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


#: The case table (the same as the table of the module docstring). The first is the export's example
#: input (the smaller one — `torch.export` runs one forward while tracing).
#: MUST: `band` (sets the band) and `accept` (acceptance) are fixed — do not change `band` after
#: looking at the `accept` results (module docstring). The same holds for the full-size
#: `full-band` / `full-accept`.
CASES: tuple[CaseSpec, ...] = (
    CaseSpec("band", (3, 16, 16), 999, 24, SEED + 0),
    CaseSpec("band", (2, 16, 24), 750, 50, SEED + 3),
    CaseSpec("band", (3, 16, 16), 500, 12, SEED + 4),
    CaseSpec("band", (1, 16, 48), 250, 80, SEED + 5),
    CaseSpec("band", (2, 16, 24), 600, 30, SEED + 8),
    CaseSpec("band", (3, 16, 16), 113, 60, SEED + 6),
    CaseSpec("accept", (2, 16, 24), 600, 45, 777001),
    CaseSpec("accept", (4, 8, 16), 30, 9, 777002),
    CaseSpec("accept", (2, 24, 16), 400, 70, 777003),
    CaseSpec("growth", (3, 32, 32), 999, 24, SEED + 2),
    CaseSpec("growth", (3, 32, 32), 500, 37, SEED + 7),
    CaseSpec("full-band", (9, 60, 104), 999, 40, SEED + 10),
    CaseSpec("full-band", (9, 60, 104), 999, 112, SEED + 11, variant="2", blocks=False),
    CaseSpec("full-band", (9, 60, 104), 750, 23, SEED + 12, blocks=False),
    CaseSpec("full-band", (9, 60, 104), 500, 64, SEED + 9, blocks=False),
    CaseSpec("full-band", (9, 60, 104), 250, 91, SEED + 13, blocks=False),
    CaseSpec("full-band", (9, 60, 104), 113, 7, SEED + 14, blocks=False),
    CaseSpec("full-accept", (9, 60, 104), 999, 28, 777006),
    CaseSpec("full-accept", (9, 60, 104), 600, 77, 777007, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 999, 40, SEED + 20, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 999, 112, SEED + 21, variant="2", blocks=False),
    CaseSpec("full-band", (21, 60, 104), 750, 23, SEED + 22, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 500, 64, SEED + 23, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 250, 91, SEED + 24, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 113, 7, SEED + 25, blocks=False),
    CaseSpec("full-accept", (21, 60, 104), 999, 28, 777008, blocks=False),
    CaseSpec("full-accept", (21, 60, 104), 600, 77, 777009, blocks=False),
)

#: The cases the i8 series takes (ADR 0120 decision 3 — the r check of the reference preset
#: `f16+dit8` is at S = 192 and S = 14,040): among the non-full-size roles the S = 192 set
#: (`band` / `accept`), and among the full-size ones the 33-frame grid (S = 14,040). The rows of
#: {@link CASES} are used as they are (seed, timestep and valid length match the f16 series).
I8_CASE_ROLES = ("band", "accept")
I8_FULL_LATENTS: tuple[tuple[int, int, int], ...] = ((9, 60, 104),)


def series_dir(dtype: str) -> Path:
    """Storage dtype → the series directory."""
    return {"f16": SERIES, "i8": I8_SERIES}[dtype]


def series_cases(dtype: str, *, full: bool = True) -> tuple[CaseSpec, ...]:
    """The cases the series of a storage dtype takes (the first is the same example input as in
    {@link CASES}). `full=False` drops the full-size cases (`--no-full`)."""
    if dtype == "f16":
        cases = CASES
    else:
        cases = tuple(
            spec
            for spec in CASES
            if (
                spec.latent_shape in I8_FULL_LATENTS
                if spec.full_size
                else spec.role in I8_CASE_ROLES
            )
        )
    return cases if full else tuple(spec for spec in cases if not spec.full_size)


def wants_float64(dtype: str, spec: CaseSpec) -> bool:
    """Whether the case has an f64 reference (`output.f64`).

    In the f16 series only the full-size cases (the S = 192 band is a ratio of max absolute
    differences — ADR 0118 decision 8); in the i8 series every case (the r check also exists at
    S = 192 — ADR 0120 decision 3).
    """
    return dtype == "i8" or spec.full_size


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


@dataclass(frozen=True)
class Float64Reference:
    """The f64 reference of one full-size case (the final output `[1,16,F,H,W]` stays f64 — it is
    rounded when written)."""

    output: torch.Tensor
    #: Time taken (seconds, CPU f64).
    seconds: float


def float64_references(
    model: nn.Module, specs: Sequence[CaseSpec], dtype: str
) -> dict[str, Float64Reference]:
    """The f64 references of the cases that have one ({@link wants_float64}) (upstream's plain
    forward with activations also in f64 — the layer-by-layer `wan.dit_reference.LayerwiseDit`,
    ADR 0121 decision 7).

    `model` is the f32 model after the series storage dtype's rounding ({@link fake_quant}). The
    weights widen those f32 values to f64 one block at a time, and the inputs widen the same values
    from {@link case_inputs}. The rounding is applied while still in f32 (rounding in f64 would make
    the scale and the rounded values disagree with the f32 reference). The whole model is not made
    f64 — the f64 held at once is the part outside the blocks and one block (for 5B the full f64
    37.25 GiB does not fit a 31 GiB machine. 1.3B also goes through this single path — no
    whole-model f64 path is kept).
    MUST: check here that `model` is rounded ({@link unrounded_weights}) — if the call order breaks
    and an unrounded model is passed, f64 goldens of unrounded weights would be written silently.
    """
    from wan import dit_reference

    chosen = [spec for spec in specs if wants_float64(dtype, spec)]
    if not chosen:
        return {}
    unrounded = unrounded_weights(dtype, model)
    if unrounded:
        raise AssertionError(
            f"the f64 reference was given unrounded {dtype} weights (call it after fake_quant):"
            f" {unrounded[:5]}"
        )
    writer = dit_reference.LayerwiseDit(model.config)
    source = dit_reference.ModuleWeights(model)
    references: dict[str, Float64Reference] = {}
    for spec in chosen:
        latents, timestep, encoder_hidden_states = case_inputs(model, spec)
        result = writer.forward(source, torch.float64, latents, timestep, encoder_hidden_states)
        name = spec.name(writer.patch_size)
        print(
            f"[case] {name}: upstream f64 reference (layer by layer) {result.seconds:.1f} s",
            flush=True,
        )
        references[name] = Float64Reference(output=result.output, seconds=result.seconds)
    return references


def dynamic_shapes() -> tuple[Any, ...]:
    """The symbolic dimensions per input. The RoPE tables are declared with the same `S` too (with a
    separate symbol, a shape where "the table and the body have different lengths" would be
    accepted and give silently wrong values — the same as Anima)."""
    tokens = Dim("S", min=DIT_SYM_MIN, max=DIT_SYM_MAX)
    return ({1: tokens}, None, None, {1: tokens}, {1: tokens})


def provenance(model_name: str) -> Provenance:
    """The provenance baked into the container (upstream's license identifier and the pinned
    revision — `wan.sources` is the source of truth)."""
    source = SOURCES[model_name]
    return Provenance(
        license=source.license, notice=NOTICE_FILENAME, upstream_revision=source.revision
    )


def rope_base_asset(model: nn.Module) -> dict[str, AssetInput]:
    """The `rope_base` asset (the bytes of the per-axis base tables' safetensors themselves — the
    same slot as Anima)."""
    payload = save(dit_patch.dit_rope_base_tables(model.rope))
    return {ROPE_BASE_ASSET: AssetInput(ROPE_BASE_ROLE, len(payload), payload)}


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
        # It does not affect the band verdict (worst × 5) (the measured table is at
        # `REFERENCE_F64_KEY` and `DIT_FULL_NORMALIZED_BAND` in the TS `e2e_wan_dit_test.ts`).
        reference[REFERENCE_F64_KEY] = output_f64.to(torch.float32).contiguous()
    return reference


def case_file_names(name: str) -> tuple[str, str]:
    """The golden file names of one case (`io.<case>` and `reference.<case>`).

    The writer ({@link _write_case_files}) and the comparison's exclusion ({@link check_absent}) go
    through the same spelling.
    """
    return f"{IO_PREFIX}{name}{CASE_SUFFIX}", f"{REFERENCE_PREFIX}{name}{CASE_SUFFIX}"


def _write_case_files(
    case: Case, output: torch.Tensor, reference_f64: Float64Reference | None, out_dir: Path
) -> list[str]:
    """Writes `io.<case>` (graph inputs + the patched eager output `output`) and `reference.<case>`
    (upstream's values). For full-size cases `reference.<case>` also holds the f64 reference
    `output.f64`."""
    io = {
        f"{INPUT_PREFIX}{key}": normalize_boundary_tensor(value, f"{case.name}: input '{key}'")
        for key, value in zip(INPUT_NAMES, case.inputs, strict=True)
    }
    io[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
        output.detach().contiguous(), f"output of {case.name}"
    )
    reference = reference_tensors(
        case.name,
        latents=case.latents,
        timestep=case.timestep,
        output=case.reference,
        blocks=case.reference_blocks,
        output_f64=None if reference_f64 is None else reference_f64.output,
    )
    io_name, reference_name = case_file_names(case.name)
    save_file(io, str(out_dir / io_name))
    save_file(reference, str(out_dir / reference_name))
    return [io_name, reference_name]


class SeriesMismatchError(AssertionError):
    """The component directory rewritten by `--check` differs bytewise from the existing series (or
    there is nothing to compare against)."""


def file_digests(directory: Path) -> dict[str, str]:
    """File name → sha256 directly under the component directory (anything that is not a file fails
    loudly — nothing the comparison does not look at is left behind)."""
    digests: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            raise SeriesMismatchError(
                f"the component directory has an item that is not a file: {path}"
            )
        with path.open("rb") as stream:
            digests[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return digests


def series_mismatches(
    written: Path, existing: Path, *, absent: Collection[str] = ()
) -> dict[str, str]:
    """The discrepancies between the rewritten and the existing component directory (file name →
    reason; empty means every file is byte-identical).

    `absent` are the existing files not written this time (the full-size cases `--no-full` drops —
    {@link check_absent}). If they were written, or are missing from the existing directory, that
    is named as a discrepancy too (the exclusion never silently misses).
    """
    actual = file_digests(written)
    expected = file_digests(existing)
    mismatches: dict[str, str] = {}
    for name in sorted(actual.keys() | expected.keys() | set(absent)):
        if name in absent:
            if name in actual:
                mismatches[name] = "wrote the file of an excluded case"
            elif name not in expected:
                mismatches[name] = (
                    "the file of an excluded case is missing from the existing directory"
                )
        elif name not in expected:
            mismatches[name] = "missing from the existing directory"
        elif name not in actual:
            mismatches[name] = "missing from the written directory"
        elif actual[name] != expected[name]:
            mismatches[name] = "sha256 differs"
    return mismatches


def check_absent(
    dtype: str, patch_size: tuple[int, int, int], *, full: bool, layers: bool
) -> frozenset[str]:
    """The existing files `--check` leaves out of the comparison (the goldens of the full-size cases
    `--no-full` does not write).

    The measurement series (`--layers`) has no goldens, so nothing is left out.
    """
    if full or layers:
        return frozenset()
    dropped = [spec for spec in series_cases(dtype) if spec.full_size]
    return frozenset(name for spec in dropped for name in case_file_names(spec.name(patch_size)))


@contextmanager
def _write_root(series: Path, *, check: bool) -> Iterator[Path]:
    """The parent of the component directory (with `check`, a temporary directory inside the series
    — removed on exit).

    The staging directory (`staged_publication`) opens below it, so the writes of a `check` run
    never reach the series' component directory by a single byte.
    """
    if not check:
        yield series
        return
    with tempfile.TemporaryDirectory(dir=series, prefix=CHECK_PREFIX) as scratch:
        yield Path(scratch)


def emit(args: argparse.Namespace) -> dict[str, Any]:
    """Writes the product graph (or the measurement graph of `--layers`) and returns a summary.

    Cases are built, written and dropped one at a time (at full size the ones holding each block's
    output are 2.6 GB each, and holding all cases at once does not fit in memory). The graph export
    comes after the eager of every case (the example input is the first case).

    MUST: if even one eager summary trips {@link eager_failures}, stop inside the staging directory
    — the series' existing final does not change by a single byte. Publishing without stopping would
    let the goldens of a broken wrapper (`io.*` is taken from the same wrapper) reach the TS-side
    comparison only after publication.

    MUST: a run with `args.check` only writes the component directory into a temporary directory
    inside the series and compares it with the existing component directory by the sha256 of every
    file ({@link series_mismatches} — the series is not replaced). If there is nothing to compare
    against, it stops before reading upstream.
    """
    started = time.perf_counter()
    series = PROBE_SERIES if args.layers else series_dir(args.dtype)
    existing = series / TARGET
    if args.check and not existing.is_dir():
        raise SeriesMismatchError(f"no component directory to compare against ({existing})")
    # The measurement graph writes no goldens, so only the example input (the first case) is built.
    specs = CASES[:1] if args.layers else series_cases(args.dtype, full=not args.no_full)
    model = load_transformer(args.model)
    wrapper_class = dit_patch.WanDitTokensLayers if args.layers else dit_patch.WanDitTokens
    wrapper = wrapper_class(model)
    rounded, scales = fake_quant(args.dtype, model, wrapper)
    print(f"[fake-quant] {TARGET}: {rounded}", flush=True)
    references_f64 = float64_references(model, specs, args.dtype)
    first = build_case(model, specs[0])
    eager: list[dict[str, Any]] = []
    written: list[str] = []

    with _write_root(series, check=args.check) as root:
        out_dir = root / TARGET
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with staged_publication(out_dir) as staged:
            staged.mkdir()
            if not args.layers:
                for spec in specs:
                    case = first if spec is specs[0] else build_case(model, spec)
                    report, output = eager_report(wrapper, model, case)
                    reference_f64 = references_f64.get(case.name)
                    if reference_f64 is not None:
                        report["reference_f64_seconds"] = round(reference_f64.seconds, 1)
                        report["reference_f32_vs_f64_ratio"] = _ratio(
                            case.reference, reference_f64.output
                        )
                    print(f"[eager] {json.dumps(report, ensure_ascii=False)}", flush=True)
                    failures = eager_failures(report)
                    if failures:
                        raise EagerEquivalenceError(f"{case.name}: {failures}")
                    eager.append(report)
                    written += _write_case_files(case, output, reference_f64, staged)
                    del case, output
            graph = export_to_file(
                wrapper,
                first.inputs,
                staged / MODEL_FILE,
                provenance=provenance(args.model),
                graph_name=TARGET,
                assets=rope_base_asset(model),
                dynamic_shapes=dynamic_shapes(),
                symbol_names=("S",),
                weight_dtype=args.dtype,
                weight_scales=scales,
                preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            )
            declared = [entry.name for entry in graph.inputs]
            if not args.layers and declared != list(INPUT_NAMES):
                raise AssertionError(
                    f"graph input names do not match the declaration: {declared} vs"
                    f" {list(INPUT_NAMES)}"
                )
        model_bytes = sum(part.stat().st_size for part in container_parts(out_dir / MODEL_FILE))
        check: dict[str, Any] | None = None
        if args.check:
            patch_size = tuple(int(size) for size in model.config.patch_size)
            absent = check_absent(args.dtype, patch_size, full=not args.no_full, layers=args.layers)
            mismatches = series_mismatches(out_dir, existing, absent=absent)
            if mismatches:
                raise SeriesMismatchError(
                    f"the rewritten {out_dir.name} differs from the existing series ({existing}):"
                    f" {json.dumps(mismatches, ensure_ascii=False)}"
                )
            check = {
                "existing": str(existing),
                "identical_files": sorted(file_digests(out_dir)),
                "skipped_existing": sorted(absent),
            }
    breakdown = storage_breakdown(graph)
    return {
        "target": TARGET,
        "dtype": args.dtype,
        "dir": str(existing),
        "layers": args.layers,
        "check": check,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "model_bytes": model_bytes,
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "inputs": [[entry.name, list(entry.shape)] for entry in graph.inputs],
        "output_shapes": [list(graph.values[name].shape) for name in graph.outputs],
        "io": written,
        "eager": eager,
        "seconds": round(time.perf_counter() - started, 1),
    }


def verify(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Measures only the pre/post-patch eager equivalence on the real weights (writes no container
    — {@link main} applies the verdict with {@link eager_failures})."""
    model = load_transformer(args.model)
    wrapper = dit_patch.WanDitTokens(model)
    if not args.no_f16:
        fake_quant(args.dtype, model, wrapper)
    return [
        eager_report(wrapper, model, build_case(model, spec))[0]
        for spec in series_cases(args.dtype, full=not args.no_full)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
    parser.add_argument(
        "--dtype",
        default="f16",
        choices=DTYPES,
        help="the series storage (i8 is the transformer-only …-i8-dyn/ — ADR 0120 decision 2)",
    )
    parser.add_argument(
        "--layers",
        action="store_true",
        help="write the measurement graph with per-layer exits to …-probe/ (writes no goldens)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="measure only the pre/post-patch eager equivalence (writes no container)",
    )
    parser.add_argument(
        "--no-f16",
        action="store_true",
        help="do not apply the f16 rounding with --verify (measure on the plain f32 weights)",
    )
    parser.add_argument(
        "--no-full",
        action="store_true",
        help=(
            "do not take the full-size (S = 14,040) cases of the i8 series (for splitting the work"
            " into steps — rewrite later without it)"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rewrite into a temporary directory inside the series and only compare it with the"
        " existing component directory by the sha256 of every file (the series is not replaced)",
    )
    args = parser.parse_args(argv)
    if args.no_f16 and (not args.verify or args.dtype != "f16"):
        parser.error(
            "--no-f16 goes only with the f16 --verify (the way to measure only the equivalence on"
            " the plain f32 weights)"
        )
    if args.no_full and args.dtype != "i8":
        # The f16 series has a complete set of 27 golden cases in place (ADR 0118 steps 3 / 8).
        # No way is provided to replace it with incomplete goldens.
        parser.error("--no-full goes only with --dtype i8")
    if args.layers and args.dtype != "f16":
        parser.error("--layers is for the f16 series only (per-layer exits …-f16-dyn-probe/)")
    if args.verify and args.layers:
        parser.error("--verify and --layers cannot be combined")
    if args.verify and args.check:
        parser.error("--verify and --check cannot be combined (--verify writes no container)")
    if args.verify:
        reports = verify(args)
        print(json.dumps(reports, indent=1, ensure_ascii=False))
        failures = {
            report["case"]: found for report in reports if (found := eager_failures(report))
        }
        if failures:
            print(
                f"eager equivalence broke: {json.dumps(failures, ensure_ascii=False)}",
                file=sys.stderr,
            )
            return 1
        return 0
    print(json.dumps(emit(args), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
