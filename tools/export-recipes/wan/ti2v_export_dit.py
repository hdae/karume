"""The script that writes the Wan2.2 TI2V-5B DiT to an I2V-capable S-form IR (`krm`, i8 series) and
small-S goldens (ADR 0121 step 1 — decisions 2 / 3 / 4 / 5 / 7).

    uv run --group wan --inexact python -m wan.ti2v_export_dit inspect        # trace + structure
    uv run --group wan --inexact python -m wan.ti2v_export_dit write          # container + golden
    uv run --group wan --inexact python -m wan.ti2v_export_dit write-full     # real-size goldens
    uv run --group wan --inexact python -m wan.ti2v_export_dit observe-quant  # observe quantization
    uv run --group wan --inexact python -m wan.ti2v_export_dit eager-full     # whole eager equiv.

Every subcommand writes a JSON summary to standard output and to `<command>.json` under `--out`
(default `outputs/bench/karume-wan2.2/<date>_dit-export/` — separate from the series), and exits
non-zero when its `passed` field is false.

## The graph (decision 3)

`wan.dit_patch.WanDitTokensTi2v`: 7 inputs — 2.1's 5 inputs (tokens, `timesteps_proj`, context,
the two RoPE tables) followed by `timesteps_proj_condition [1,256]` and `condition_mask [1,S,1]`
(bool). The time MLP runs twice with M = 1, and each modulation component is selected by a `where`
right before it is consumed. The upper bound of `Dim("S")` is 27,280 (1280×704×121 — the TS side
narrows the accepted set).

## The export form (decision 7 — the form of ADR 0119 step 10b, no change to the exporter core)

The trace runs on the upstream on meta without weights ({@link trace}), and the 307 linears (the
patch embedding's Linear included) are turned into i8 per block of rows from the checkpoint and
passed as `fixed_weights`. bias, norm and `scale_shift_table` stay the checkpoint's f32 (the
material preparation of `wan.dit_probe.prepare` runs through {@link trace} — measured peak at step 0
5.84 GiB). The container holds the asset `rope_base` (the same slot as in 2.1).

## The IR check ({@link inspect_graph} — the acceptance of step 1)

- the linear nodes are "the quantization targets + the 3 of the time MLP" (310 for the 5B — the
  dispatch count of decision 2)
- the i8 weights are exactly the quantization targets (307 for the 5B), and the remaining weights
  (bias, norm, `scale_shift_table`) are f32
- there are 6 `where`s per block + 2 in the head, and for each, the consumer of the output
  `[1,S,dim]` is **only the next node** (no `where` output lives across attention —
  {@link where_placement})
- the graph inputs are the 7 declared ones, and `condition_mask` is bool `[1,S,1]`

## The goldens (step 1 — S = 192 in the T2V form and the I2V form)

The cases are {@link CASES} (6 decision cases = 3 in the T2V form + 3 in the I2V form, 4 acceptance
cases = 2 T2V + 2 I2V — the counts of step 2's r check). The context is the fixed 4 prompts of
Wan2.1's precomputed embedding asset (a synthetic random context amplifies the difference between
CPU f32 and f64 on the 5B — the doc of {@link Ti2vCase}), and the latents are random from the seed.
The weights are read layer by layer from **the container just written**
(`dit_reference.ContainerDitWeights` — i8 as packed × scale in f32). f32 runs with the same address
offsets as the upstream's `from_pretrained` (the last bits of the f32 GEMM depend on the addresses —
the module doc of `wan.dit_reference`).

- `reference.<case>`: `latents` / `timestep` (i32, generation side) / `condition_timestep` (i32,
  condition side — 0 for I2V, the same as the generation side for T2V) / `output` (CPU f32) /
  `output.f64` (activations in f64 too, rounded to f32). How the reference represents time is
  decision 4: T2V uses the upstream forward with the one-dimensional timestep, I2V the reference
  wrapper (the time MLP with M = 1 per value — `condition` of `dit_patch.reference_dit`)
- `io.<case>`: the 7 graph inputs (`input.*` — the condition mask as u32 0 / 1) and the patched
  eager output `output.0` (the S-form wrapper run with the same layer-by-layer weights)

The eager-equivalence check (the same judgment as `export_dit.eager_failures`): the T2V form is
bit-identical in the trunk to the upstream path with the one-dimensional timestep, and the I2V form
to the reference wrapper. If even one fails, it stops inside the working directory (the series does
not change by a single byte). Both sides read the same layer-by-layer weights, so an error common to
the loading is not visible to this check (`eager-full` below checks with independent weights). The
difference from diffusers' two-dimensional timestep path (linear with M = S) is listed in the
summaries of the decision cases as an observation (not a gate — decision 4).

## Real-size goldens (step 2 — `write-full`)

The cases are {@link FULL_CASES} (the two shapes of the numeric check — latent `[48,21,30,52]`
(832×480, 81 frames, S = 8,190) and `[48,9,44,80]` (1280×704, 33 frames, S = 7,920). The shapes
were taken from the acceptance upper bounds of the ADR 0121 addendum "settling ruling 1"; 832×480
fell outside the accepted set with the addendum (2026-10-05) "accepted sizes down to the two
official sizes". The DiT graph does not depend on the size, so it is used as a numeric check shape
unrelated to the accepted sizes). 6 decision cases (3 per shape, T2V 3 + I2V 3) and 8 acceptance
cases (T2V 2 + I2V 2 per shape). The contents of one case are the same as the S = 192 golden
({@link golden_case} — f32 / f64 references, the patched eager, the eager-equivalence check), and
the weights are read layer by layer from **the installed container** (the container is not
rewritten). No observation of the M = S path is taken ({@link OBSERVED_ROLES} is only the S = 192
decision cases — at real size it would add one forward per case).

- **How they are added**: one case at a time, written into a working directory right under the
  series (`.write-full-*` — outside the component directory), then the two files are moved into the
  component directory with `os.replace` (a rename within the same file system — no half-written
  file appears in the component directory). A case that fails a check moves nothing.
- **Resuming**: a case with both files present is skipped. Before skipping, the installed
  `input.*` (7) and `latents` / `timestep` / `condition_timestep` are compared bit by bit with the
  values rebuilt from the current case table, and it stops on a difference
  ({@link stored_case_state} — an old golden from before a table change is not counted as "done").
  A case with only one file (it failed partway through the move) is rewritten. A working directory
  left by a previous run is deleted at the start.
- **Progress**: per case a `[write-full]` line (done, remaining, the remaining estimate — the
  estimate is the average of the finished cases, or {@link FULL_CASE_SECONDS} from step 0's
  measurement when there is none), and the `[block]` lines per block of the f32 / f64 references
  (for S at or above `dit_reference.PROGRESS_TOKENS`). The RAM peak is about 5 GiB as measured at
  step 0 ({@link FULL_PEAK_BYTES} — MemAvailable is checked before writing).
- MUST: `write` replaces the component directory together with the working directory, so the
  real-size goldens are gone after `write`. After re-running `write`, re-run `write-full` too.

## Eager equivalence with independent weights (`eager-full` — the form of decision 4)

The `write` check runs both the patched graph and the reference through **the same layer-by-layer
mechanism that reads from the same container** (to fit within 8 GiB of RAM). So an error common to
the loading (the container reader, how the layer-by-layer sequence is assembled) cancels out on both
sides and can slip through the check. `eager-full` ({@link eager_full}) closes that hole once, in
the form decision 4 describes:

- the upstream is loaded whole as one f32 model with `from_pretrained`, and i8 fake-quant is applied
  through the patched wrapper (the same as 2.1's `export_dit` — neither the container nor the
  layer-by-layer path is involved)
- for each case, the eager of the patched graph ({@link dit_patch.WanDitTokensTi2v} — the T2V /
  I2V form) is compared with the upstream (T2V with the forward of the one-dimensional timestep,
  I2V with the reference wrapper) (the same judgment as `export_dit.eager_failures` — bit
  identity of the trunk)
- it is also compared with the installed goldens: the upstream reference = `output` of
  `reference.<case>`, the patched output = `output.0` of `io.<case>` (both bit-identical — the
  goldens are values taken from the container + the upstream's address offsets, relying on decision
  7's claim that the layer-by-layer f32 is bit-identical to the upstream f32 eager. NOTE: this is
  MKL's behavior, not a guarantee of a specification — the module doc of `wan.dit_reference`). It
  also checks that the 7 graph inputs are the same as the golden's (that the same case was built).

The RAM peak is about 21 GiB (the whole f32 18.63 GiB + base). Before loading the upstream it reads
MemAvailable and stops if it falls short of the estimate (f32 of `dit_probe.eager_memory` + base)
plus a margin (`dit_probe.require_available`). If the installed goldens are incomplete, it also
stops before loading.

## Observing quantization (decision 5 — `observe-quant`)

The difference (max abs ÷ max ref) of the CPU f32 reference with i8 weights only against the
unquantized CPU f32 reference is taken on the S = 192 decision cases for both the 5B and the 1.3B,
with the same assignment of prompts (the reader is the checkpoint's F32 and the values with i8
fake-quant applied one at a time — `dit_reference.CheckpointDitWeights`). When the 5B's worst
exceeds twice the 1.3B's worst (a rule of thumb), `escalate` becomes true (raised for a ruling
before step 2).

MUST: diffusers is imported inside functions (the `wan` group is not part of the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open
from safetensors.torch import save, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import BENCH_ROOT, SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import AssetInput, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_module, publish_model
from karume.quantize import QUANT_MODULE_TYPES, iter_quant_targets
from wan import dit_patch, dit_probe, dit_reference, export_dit
from wan.sources import local_snapshot
from wan.umt5_export import MemoryMonitor

#: The default model (Wan2.2 TI2V-5B — ADR 0121).
DEFAULT_MODEL = "ti2v-5b"

#: The i8 series (the series name of decision 7 — the suffix `-dyn` is the convention of ADR 0077).
SERIES_NAME = "wan2.2-ti2v-5b-i8-dyn"
SERIES = SERIES_ROOT / SERIES_NAME

#: The container's graph name = the distribution's component name = the key in weights
#: (container-v1 §2.1 — the same slot as 2.1's DiT; the match is checked by
#: `tests/test_graph_names.py`). The location inside the series uses the same spelling (a
#: convention, not a derivation).
TARGET = "transformer"
MODEL_FILE = "model.krm"

#: The graph input names (the arguments of the wrapper's forward become the IR input names as is —
#: the first 5 are the same as in 2.1).
INPUT_NAMES = (*export_dit.INPUT_NAMES, "timesteps_proj_condition", "condition_mask")

#: The upper bound of `Dim("S")` (1280×704×121 = 27,280 — decision 3) and the lower bound (the same
#: as in 2.1).
SYM_MAX = 27_280
SYM_MIN = export_dit.DIT_SYM_MIN

#: The number of linears of the time MLP (the 2 of `time_embedder` + `time_proj`) — it runs twice
#: with M = 1, so there are this many more linear nodes than quantization targets (decision 2).
TIME_MLP_LINEARS = 3

#: The number of modulation `where`s (6 components per block, 2 components in the head —
#: decision 3).
WHERE_PER_BLOCK = 6
WHERE_HEAD = 2

#: The 5B's expected values (decision 2 — the number of i8 weights / the number of linear dispatches
#: per pass).
EXPECTED_5B = {"i8_weights": 307, "linear_nodes": 310}

#: The base of the golden's random seeds (for the decision cases; the acceptance cases hold seeds
#: from a separate line directly — a value that overlaps neither the 1.3B's `export_dit.SEED` +
#: 0–25 nor the acceptance 777001 and up).
SEED = 20262000

#: The distribution name of the location (`outputs/bench/<distribution name>/` —
#: `docs/assets-layout.md`).
BENCH_NAME = "karume-wan2.2"


class Ti2vExportError(RuntimeError):
    """The graph, the weights or the goldens departed from the form of step 1's acceptance."""


Form = Literal["t2v", "i2v"]


@dataclass(frozen=True)
class Ti2vCase:
    """The conditions of one golden case (T2V / I2V, the latent shape, the generation-side
    timestep, the context prompt, the seed).

    The context is one prompt of Wan2.1's precomputed embedding asset (the same umT5 output — an
    asset ADR 0121 decision 9 carries over to 2.2 as is) (`dit_probe.probe_inputs` — zero-padded to
    512 rows after the valid length). The latents are random from the seed.

    Why the synthetic random context (how 2.1's goldens are made) is not used (step 1's isolation —
    results in `outputs/bench/karume-wan2.2/2026-10-04_dit-f32-f64-diagnosis/`): N(0,1) random
    values are outside the distribution, about 12 times the size of the real umT5 output (std about
    0.08), and on the 5B that alone amplifies the difference between CPU f32 and f64 layer after
    layer. The output ratio at S = 1,170, t = 999 is 1.38e-2 synthetic / 1.24e-6 with a real
    prompt, and even at S = 192 the synthetic one is 5.3e-4 (the same construction on the 1.3B gives
    2e-6 to 1.6e-4). When the denominator of the r check (the ratio of CPU f32 against f64)
    saturates from this amplification, subtle fault injections get lost in the band.
    """

    role: str
    form: Form
    latent_shape: tuple[int, int, int]
    timestep: int
    text: str
    seed: int

    def name(self, patch_size: tuple[int, int, int]) -> str:
        return f"{self.role}-{self.form}-s{self.tokens(patch_size):05d}-t{self.timestep:04d}"

    def tokens(self, patch_size: tuple[int, int, int]) -> int:
        frames, height, width = self.latent_shape
        return (frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])

    def spec(self) -> export_dit.CaseSpec:
        """The form that draws the random latents (`export_dit.case_inputs` — the latents are drawn
        first, so the synthetic text's valid length (1 here) does not affect the latent values.
        `probe_inputs` replaces the context with the prompt)."""
        return export_dit.CaseSpec(
            self.role, self.latent_shape, self.timestep, 1, self.seed, blocks=False
        )


#: The golden cases (S = 192, latent `[48,F,H,W]`). The decision cases spread the timestep over
#: 999–113 and mix square grids (3·8·8) with non-square ones (2·8·12 / 1·8·24 / 2·12·8 / 6·4·8 /
#: 4·4·12) (an h / w mix-up hides on a square). The I2V form has F' ≥ 2 (only the first latent frame
#: is the condition — with F' = 1 every token would be the condition). The fixed 4 prompts (3
#: positive + the negative — the context of CFG's second pass) are spread over the forms and roles.
#: MUST: do not add or remove `band` (decision) cases after seeing the results of `accept`
#: (acceptance) (deciding the band and accepting are independent — the same discipline as 2.1's
#: `export_dit.CASES`).
CASES: tuple[Ti2vCase, ...] = (
    Ti2vCase("band", "t2v", (3, 16, 16), 999, "boxing-cats", SEED + 0),
    Ti2vCase("band", "t2v", (2, 16, 24), 500, "ferret", SEED + 1),
    Ti2vCase("band", "t2v", (1, 16, 48), 250, "cat-dog-baking", SEED + 2),
    Ti2vCase("band", "i2v", (3, 16, 16), 999, "negative", SEED + 3),
    Ti2vCase("band", "i2v", (2, 24, 16), 750, "boxing-cats", SEED + 4),
    Ti2vCase("band", "i2v", (6, 8, 16), 113, "ferret", SEED + 5),
    Ti2vCase("accept", "t2v", (2, 16, 24), 600, "cat-dog-baking", 778001),
    Ti2vCase("accept", "t2v", (4, 8, 24), 30, "negative", 778002),
    Ti2vCase("accept", "i2v", (3, 16, 16), 400, "ferret", 778003),
    Ti2vCase("accept", "i2v", (4, 8, 24), 900, "cat-dog-baking", 778004),
)

#: The grid of the trace's example inputs (the first case — the values do not affect the trace;
#: S = 192 does not hit the 0 / 1 specialization).
TRACE_LATENT = CASES[0].latent_shape

#: The two real-size shapes (the shapes of the numeric check — taken from the acceptance upper
#: bounds of the ADR 0121 addendum "settling ruling 1"): 832×480 with 81 frames = latent
#: `[48,21,30,52]` (S = 21·15·26 = 8,190) and 1280×704 with 33 frames = `[48,9,44,80]`
#: (S = 9·22·40 = 7,920). 832×480 fell outside the accepted set with the addendum (2026-10-05), but
#: the DiT graph does not depend on the size, so it is used as a numeric check shape unrelated to
#: the accepted sizes.
FULL_LATENTS: tuple[tuple[int, int, int], ...] = ((21, 30, 52), (9, 44, 80))

#: The real-size golden cases (step 2's r check — the module doc's "Real-size goldens"). The
#: decision cases (`full-band`) are 3 per shape, split into 3 T2V (t 999 / 113 / 750) and 3 I2V
#: (t 500 / 999 / 250), with T2V and I2V mixed in both shapes. The acceptance cases (`full-accept`)
#: are T2V 2 + I2V 2 per shape, with seeds distinct from every other case (778011 and up). The fixed
#: 4 prompts are spread over the shapes and roles ({@link Ti2vCase} — no synthetic random context).
#: MUST: do not add or remove `full-band` cases after seeing the results of `full-accept` (the same
#: discipline as {@link CASES}).
FULL_CASES: tuple[Ti2vCase, ...] = (
    Ti2vCase("full-band", "t2v", (21, 30, 52), 999, "boxing-cats", SEED + 10),
    Ti2vCase("full-band", "i2v", (21, 30, 52), 500, "ferret", SEED + 11),
    Ti2vCase("full-band", "t2v", (21, 30, 52), 113, "negative", SEED + 12),
    Ti2vCase("full-band", "i2v", (9, 44, 80), 999, "cat-dog-baking", SEED + 13),
    Ti2vCase("full-band", "t2v", (9, 44, 80), 750, "ferret", SEED + 14),
    Ti2vCase("full-band", "i2v", (9, 44, 80), 250, "boxing-cats", SEED + 15),
    Ti2vCase("full-accept", "t2v", (21, 30, 52), 999, "cat-dog-baking", 778011),
    Ti2vCase("full-accept", "t2v", (21, 30, 52), 600, "boxing-cats", 778012),
    Ti2vCase("full-accept", "i2v", (21, 30, 52), 900, "negative", 778013),
    Ti2vCase("full-accept", "i2v", (21, 30, 52), 400, "ferret", 778014),
    Ti2vCase("full-accept", "t2v", (9, 44, 80), 999, "negative", 778015),
    Ti2vCase("full-accept", "t2v", (9, 44, 80), 300, "ferret", 778016),
    Ti2vCase("full-accept", "i2v", (9, 44, 80), 800, "boxing-cats", 778017),
    Ti2vCase("full-accept", "i2v", (9, 44, 80), 50, "cat-dog-baking", 778018),
)

#: The estimated run time of one real-size case (seconds — the initial value for progress).
#: Step 0's measurement (S = 8,190, layer by layer) of f32 233 s + f64 510 s, plus the patched
#: eager (f32, the same layer-by-layer path) taken as the same 233 s as f32.
FULL_CASE_SECONDS = 233 + 510 + 233

#: The estimated RAM peak of one real-size case (measured 5.04 GiB at step 0 — the layer-by-layer
#: reference, S = 8,190). Before writing, it stops if MemAvailable falls short of this + a margin
#: (`dit_probe.MEMORY_HEADROOM_BYTES`).
FULL_PEAK_BYTES = int(5.1 * 2**30)

#: The prefix of the `write-full` working directory (right under the series — outside the component
#: directory).
FULL_STAGING_PREFIX = ".write-full-"


def default_out() -> Path:
    """The default location of the summaries (separate from the series —
    `outputs/bench/karume-wan2.2/<date>_dit-export/`)."""
    return BENCH_ROOT / BENCH_NAME / f"{date.today().isoformat()}_dit-export"


def transformer_dir(model: str = DEFAULT_MODEL) -> Path:
    """The location of the pinned revision's DiT (`local_snapshot` fails loudly unless it has been
    fetched)."""
    return local_snapshot(model) / dit_reference.TRANSFORMER_SUBFOLDER


# ---------------------------------------------------------------------------
# trace (an upstream without weights)
# ---------------------------------------------------------------------------


def meta_inputs(config: Mapping[str, Any], latent_shape: tuple[int, int, int]) -> tuple[Any, ...]:
    """The 7 example inputs of the trace (meta — the weights are on meta, so the inputs are too)."""
    patch = [int(size) for size in config["patch_size"]]
    frames, height, width = latent_shape
    tokens = (frames // patch[0]) * (height // patch[1]) * (width // patch[2])
    width_in = int(config["in_channels"]) * math.prod(patch)
    head_dim = int(config["attention_head_dim"])
    freq = int(config["freq_dim"])
    with torch.device("meta"):
        return (
            torch.empty(1, tokens, width_in),
            torch.empty(1, freq),
            torch.empty(1, 512, int(config["text_dim"])),
            torch.empty(1, tokens, 1, head_dim),
            torch.empty(1, tokens, 1, head_dim),
            torch.empty(1, freq),
            torch.empty(1, tokens, 1, dtype=torch.bool),
        )


def dynamic_shapes(sym_max: int = SYM_MAX) -> tuple[Any, ...]:
    """The symbolic dimensions per input. The RoPE tables and the condition mask share the same
    `S` (a separate symbol would accept shapes of mismatched length and give silently wrong
    values — the same judgment as in 2.1)."""
    tokens = Dim("S", min=SYM_MIN, max=sym_max)
    return ({1: tokens}, None, None, {1: tokens}, {1: tokens}, None, {1: tokens})


def trace(
    config: Mapping[str, Any], sym_max: int = SYM_MAX
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, int]]:
    """Run the I2V-capable S-form export on the upstream on meta (the tracer of
    `dit_probe.prepare`).

    Returns the graph, the stored tensors (weights on meta) and the quantization targets (tensor key
    → channel axis).
    """
    wrapper = dit_patch.WanDitTokensTi2v(dit_reference.meta_model(config))
    graph, tensors = export_module(
        wrapper,
        meta_inputs(config, TRACE_LATENT),
        dynamic_shapes=dynamic_shapes(sym_max),
        symbol_names=("S",),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    declared = [entry.name for entry in graph.inputs]
    if declared != list(INPUT_NAMES):
        raise Ti2vExportError(f"graph input names do not match the declaration: {declared}")
    targets = {
        fqn: axis for fqn, _, axis in iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES)
    }
    return graph, tensors, targets


# ---------------------------------------------------------------------------
# The IR check
# ---------------------------------------------------------------------------


def where_placement(graph: IrGraph) -> dict[str, Any]:
    """Where the `where`s sit (decision 3): whether the only consumer of the output is the next
    node.

    When the only consumer is the next node, the `where` output `[1,S,dim]` lives only for that
    node and does not cross attention. Written in the upstream order (all 6 components made together
    at the start of the block), it is named here.
    """
    consumers: dict[str, list[int]] = {}
    for position, node in enumerate(graph.nodes):
        for name in node.ins:
            consumers.setdefault(name, []).append(position)
    count = 0
    violations: list[str] = []
    for position, node in enumerate(graph.nodes):
        if node.op != "where":
            continue
        count += 1
        (output,) = node.outs
        users = consumers.get(output, [])
        if users != [position + 1]:
            crossed = sorted(
                {
                    graph.nodes[between].op
                    for between in range(position + 1, max(users, default=position) + 1)
                    if graph.nodes[between].op == "attention"
                }
            )
            violations.append(
                f"where #{position} ({output}) is consumed by nodes {users} (not only the next one"
                f"{' — crosses attention' if crossed else ''})"
            )
        shape = graph.values[output].shape
        if len(shape) != 3 or shape[1] != "S":
            violations.append(f"where #{position} ({output}) outputs {shape} (not [1,S,dim])")
    return {"count": count, "violations": violations}


def inspect_structure(graph: IrGraph, quant_targets: Sequence[str], layers: int) -> dict[str, Any]:
    """The checks that do not depend on storage (the linear count, the `where` count and placement,
    the input declarations) — passes when the returned `failures` is empty."""
    ops = Counter(node.op for node in graph.nodes)
    placement = where_placement(graph)
    inputs = [[entry.name, list(entry.shape), entry.dtype] for entry in graph.inputs]
    failures: list[str] = []
    if ops["linear"] != len(quant_targets) + TIME_MLP_LINEARS:
        failures.append(
            f"{ops['linear']} linear nodes (expected the {len(quant_targets)} quantization"
            f" targets + the {TIME_MLP_LINEARS} of the time MLP)"
        )
    if placement["count"] != WHERE_PER_BLOCK * layers + WHERE_HEAD:
        failures.append(
            f"{placement['count']} where nodes"
            f" (expected {WHERE_PER_BLOCK} × {layers} + {WHERE_HEAD})"
        )
    failures.extend(placement["violations"])
    if [name for name, _, _ in inputs] != list(INPUT_NAMES):
        failures.append(f"the graph inputs differ from the declaration: {inputs}")
    elif inputs[-1][1:] != [[1, "S", 1], "bool"]:
        failures.append(f"the condition mask is not bool [1,S,1]: {inputs[-1]}")
    return {
        "ops": dict(sorted(ops.items())),
        "linear_nodes": ops["linear"],
        "where": placement["count"],
        "symbols": list(graph.symbols),
        "inputs": inputs,
        "outputs": [list(graph.values[name].shape) for name in graph.outputs],
        "failures": failures,
    }


def inspect_storage(
    graph: IrGraph, quant_targets: Sequence[str], plain: Sequence[str]
) -> dict[str, Any]:
    """The storage check of the graph whose storage declarations were committed (i8 = exactly the
    quantization targets, the remaining weights f32)."""
    storage = {
        initializer.tensor: initializer.storage.dtype
        for initializer in graph.initializers.values()
        if initializer.tensor is not None
    }
    i8 = sorted(key for key, dtype in storage.items() if dtype == "i8")
    not_f32 = sorted(key for key in plain if storage.get(key) != "f32")
    failures: list[str] = []
    if i8 != sorted(quant_targets):
        failures.append(
            f"the i8 weights differ from the quantization targets (i8 {len(i8)} / targets"
            f" {len(quant_targets)})"
        )
    if not_f32:
        failures.append(f"weights that should be f32 are not f32: {not_f32[:5]}")
    return {"i8_weights": len(i8), "f32_weights": len(plain) - len(not_f32), "failures": failures}


def inspect_graph(
    graph: IrGraph, quant_targets: Sequence[str], plain: Sequence[str], layers: int
) -> dict[str, Any]:
    """The check of the graph whose storage declarations were committed (the module doc's "The IR
    check" — structure + storage)."""
    structure = inspect_structure(graph, quant_targets, layers)
    storage = inspect_storage(graph, quant_targets, plain)
    return {**structure, **storage, "failures": structure["failures"] + storage["failures"]}


def expected_counts(model: str, inspection: Mapping[str, Any]) -> list[str]:
    """Mismatches against the 5B's expected counts (decision 2 — {@link EXPECTED_5B}) (models other
    than the 5B, and fields not in the inspection, are not checked)."""
    if model != DEFAULT_MODEL:
        return []
    return [
        f"{key} is {inspection[key]} (decision 2 expects {value})"
        for key, value in EXPECTED_5B.items()
        if key in inspection and inspection[key] != value
    ]


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------


def rope_base_asset(writer: dit_reference.LayerwiseDit) -> dict[str, AssetInput]:
    """The asset `rope_base` (per-axis base tables cut from the upstream RoPE tables — the same
    slot and format as in 2.1)."""
    payload = save(dit_patch.dit_rope_base_tables(writer.rope))
    return {
        export_dit.ROPE_BASE_ASSET: AssetInput(export_dit.ROPE_BASE_ROLE, len(payload), payload)
    }


def write_container(
    export: dit_probe.DitExport,
    path: Path,
    assets: Mapping[str, AssetInput],
    model: str = DEFAULT_MODEL,
) -> IrGraph:
    """Write the material (what `dit_probe.prepare` returns) into a container (`path` is the
    representative path — the part sequence of the split form).

    The storage is the material's (the quantization targets as `fixed`'s packed + scale, the rest
    the checkpoint's f32). Returns the graph whose storage declarations were committed (the input of
    {@link inspect_graph}).
    """
    return publish_model(
        path,
        export.graph,
        dict(export.tensors),
        provenance=export_dit.provenance(model),
        graph_name=TARGET,
        fixed_weights=export.fixed,
        assets=assets,
    )


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseInputs:
    """One case's upstream inputs, the reference's time, and the 7 graph inputs."""

    latents: torch.Tensor
    timestep: torch.Tensor
    condition_timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    #: How the reference represents time (T2V: None = the upstream one-dimensional timestep, I2V:
    #: the reference wrapper — decision 4).
    condition: dit_patch.TimestepCondition | None
    graph_inputs: tuple[torch.Tensor, ...]


def case_inputs(writer: dit_reference.LayerwiseDit, case: Ti2vCase) -> CaseInputs:
    """One case's inputs (the random values and the context are made in one place,
    `dit_probe.probe_inputs` — the same values for f32 / f64 / eager)."""
    patch = writer.patch_size
    latents, timestep, encoder_hidden_states = dit_probe.probe_inputs(
        writer, case.spec(), case.text
    )
    conditioned = case.form == "i2v"
    condition_timestep = torch.zeros_like(timestep) if conditioned else timestep.clone()
    mask = dit_patch.dit_condition_mask(case.latent_shape, patch, conditioned=conditioned)
    rope_cos, rope_sin = dit_patch.dit_rope_tables(writer.rope, case.latent_shape)
    graph_inputs = (
        dit_patch.dit_patchify(latents, patch),
        writer.timesteps_proj(timestep),
        encoder_hidden_states,
        rope_cos,
        rope_sin,
        writer.timesteps_proj(condition_timestep),
        mask,
    )
    return CaseInputs(
        latents=latents,
        timestep=timestep,
        condition_timestep=condition_timestep,
        encoder_hidden_states=encoder_hidden_states,
        condition=dit_patch.TimestepCondition(condition_timestep, mask[..., 0])
        if conditioned
        else None,
        graph_inputs=graph_inputs,
    )


def patched_eager(
    writer: dit_reference.LayerwiseDit,
    source: dit_reference.WeightSource,
    inputs: CaseInputs,
    latent_shape: tuple[int, int, int],
    reference: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Run the eager of the patched graph ({@link dit_patch.WanDitTokensTi2v}) with the same
    layer-by-layer weights and return a summary of its difference from the reference (the same
    fields as `export_dit.eager_report` — judged by `export_dit.eager_failures`) and the graph
    output `[1,S,pt·ph·pw·C]`.

    The trunk is the path with the patch embedding replaced by the upstream conv3d output (claimed
    bit-identical). When the output of the Linear-ized patch embedding is bit-identical to conv3d,
    trunk = full, so the sequence runs only once.
    """
    patch = writer.patch_size

    def body(model: nn.Module) -> dict[str, Any]:
        wrapper = dit_patch.WanDitTokensTi2v(model, install_processors=False)
        tokens = inputs.graph_inputs[0]
        with dit_patch.flash_attention_only():
            hidden = model.patch_embedding(inputs.latents).flatten(2).transpose(1, 2)
            embedded = wrapper.patch_embedding(tokens)
            bound = export_dit.patch_embedding_error_bound(tokens, wrapper.patch_embedding)
            started = time.perf_counter()
            tokens_out = wrapper(*inputs.graph_inputs)
            seconds = time.perf_counter() - started
            full = dit_patch.dit_unpatchify(tokens_out, latent_shape, patch)
            trunk_from_full = torch.equal(embedded, hidden)
            trunk = (
                full
                if trunk_from_full
                else dit_patch.dit_unpatchify(
                    wrapper.forward_hidden(hidden, *inputs.graph_inputs[1:]), latent_shape, patch
                )
            )
        return {
            "tokens_out": tokens_out,
            "report": {
                "reference_max_abs": float(reference.abs().max()),
                "trunk_bit_exact": torch.equal(trunk, reference),
                "trunk_max_abs_diff": float((trunk - reference).abs().max()),
                "trunk_from_full": trunk_from_full,
                "patch_embedding_max_abs_diff": float((embedded - hidden).abs().max()),
                "patch_embedding_max_abs": float(hidden.abs().max()),
                "patch_embedding_within_bound": bool(((embedded - hidden).abs() <= bound).all()),
                "output_finite": bool(tokens_out.isfinite().all()),
                "full_bit_exact": torch.equal(full, reference),
                "full_max_abs_diff": float((full - reference).abs().max()),
                "patched_seconds": round(seconds, 1),
            },
        }

    result, _, _ = writer.run(
        source,
        torch.float32,
        body,
        prepare_block=lambda block: dit_patch.install_real_pair_processors([block]),
    )
    return result["report"], result["tokens_out"]


def per_token_timestep(inputs: CaseInputs) -> torch.Tensor:
    """diffusers' two-dimensional timestep `[1,S]` (the observation's counterpart —
    `WanPipeline(expand_timesteps=True)` lines t up over every token even for T2V, and
    `WanImageToVideoPipeline` sets the tokens of the condition frames to 0)."""
    mask = inputs.graph_inputs[-1][..., 0]
    return torch.where(mask, inputs.condition_timestep, inputs.timestep).to(torch.int64)


def golden_case(
    writer: dit_reference.LayerwiseDit,
    source: dit_reference.WeightSource,
    case: Ti2vCase,
    out_dir: Path,
) -> dict[str, Any]:
    """Take one case's references (f32 / f64), the patched eager and the observation, run them
    through the checks, then write the golden.

    For cases with S at or above `dit_reference.PROGRESS_TOKENS` (real size — one forward takes
    minutes), it prints the reference's per-block `[block]` lines and a line when the patched eager
    starts (the values do not change).
    """
    patch = writer.patch_size
    name = case.name(patch)
    inputs = case_inputs(writer, case)
    common = (inputs.latents, inputs.timestep, inputs.encoder_hidden_states)
    long_case = case.tokens(patch) >= dit_reference.PROGRESS_TOKENS

    def progress(label: str) -> Any:
        return dit_reference._block_progress(name, label, writer.layers) if long_case else None

    reference32 = writer.forward(
        source, torch.float32, *common, condition=inputs.condition, on_block=progress("f32")
    )
    reference64 = writer.forward(
        source, torch.float64, *common, condition=inputs.condition, on_block=progress("f64")
    )
    if long_case:
        print(f"[golden] {name}: patched eager (f32, the same layer-by-layer weights)", flush=True)
    report, tokens_out = patched_eager(
        writer, source, inputs, case.latent_shape, reference32.output
    )
    report.update(
        {
            "case": name,
            "form": case.form,
            "reference_f32_seconds": round(reference32.seconds, 1),
            "reference_f64_seconds": round(reference64.seconds, 1),
            "reference_f32_vs_f64_ratio": export_dit._ratio(reference32.output, reference64.output),
        }
    )
    if case.role in OBSERVED_ROLES:
        observed = writer.forward(
            source, torch.float32, inputs.latents, per_token_timestep(inputs), common[2]
        )
        report["per_token_timestep_vs_reference_ratio"] = export_dit._ratio(
            observed.output, reference32.output
        )
        report["per_token_timestep_bit_exact"] = torch.equal(observed.output, reference32.output)
    failures = export_dit.eager_failures(report)
    print(f"[eager] {json.dumps(report, ensure_ascii=False)}", flush=True)
    if failures:
        raise export_dit.EagerEquivalenceError(f"{name}: {failures}")
    io = {
        f"{export_dit.INPUT_PREFIX}{key}": normalize_boundary_tensor(value, f"{name} input '{key}'")
        for key, value in zip(INPUT_NAMES, inputs.graph_inputs, strict=True)
    }
    io[f"{export_dit.OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
        tokens_out.contiguous(), f"{name} output"
    )
    reference = export_dit.reference_tensors(
        name,
        latents=inputs.latents,
        timestep=inputs.timestep,
        output=reference32.output,
        blocks=[],
        output_f64=reference64.output,
    )
    reference[CONDITION_TIMESTEP_KEY] = normalize_boundary_tensor(
        inputs.condition_timestep, f"{name} condition-side timestep"
    )
    save_file(io, str(out_dir / f"{export_dit.IO_PREFIX}{name}{export_dit.CASE_SUFFIX}"))
    save_file(
        reference, str(out_dir / f"{export_dit.REFERENCE_PREFIX}{name}{export_dit.CASE_SUFFIX}")
    )
    return report


#: The roles for which the difference from diffusers' two-dimensional timestep path is taken (an
#: observation — not a gate) (the 6 decision cases — both the T2V and I2V forms. Taking it for the
#: acceptance cases too would add one forward per case to the writing).
OBSERVED_ROLES = frozenset({"band"})

#: The key of the condition-side timestep in `reference.<case>` (0 for I2V, the same as the
#: generation side for T2V — the TS host test uses it to check `timesteps_proj_condition`).
CONDITION_TIMESTEP_KEY = "condition_timestep"


def write_series(model: str = DEFAULT_MODEL, *, names: Sequence[str] = ()) -> dict[str, Any]:
    """Write the container and goldens into a working directory, and install them into the series
    only when every check (the IR check, eager equivalence) passes.

    `names` narrows the cases by name (for development — a narrowed series is not installed: it
    stops inside the working directory).
    """
    directory = transformer_dir(model)
    writer = dit_reference.LayerwiseDit(
        dit_reference.load_config(directory),
        alignment=dit_reference.upstream_alignment(directory),
    )
    known = {case.name(writer.patch_size) for case in CASES}
    if set(names) - known:
        raise Ti2vExportError(f"unknown case names: {sorted(set(names) - known)}")
    chosen = [case for case in CASES if not names or case.name(writer.patch_size) in names]
    target = SERIES / TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    eager: list[dict[str, Any]] = []
    with MemoryMonitor() as monitor:
        export, prepared = dit_probe.prepare(
            directory, stage=monitor.stage, sym_max=SYM_MAX, tracer=trace
        )
        quant_targets = sorted(export.fixed)
        plain = export.plain
        with staged_publication(target) as staged:
            staged.mkdir()
            with monitor.stage("write") as record:
                graph = write_container(export, staged / MODEL_FILE, rope_base_asset(writer), model)
                record.details["initializers"] = len(graph.initializers)
            # The goldens are read from the container, so the material (i8 packed, about 4.65 GiB)
            # is released first.
            del export
            gc.collect()
            inspection = inspect_graph(graph, quant_targets, plain, writer.layers)
            failures = inspection["failures"] + expected_counts(model, inspection)
            if failures:
                raise Ti2vExportError(f"failed the IR check: {failures}")
            source = dit_reference.ContainerDitWeights(staged / MODEL_FILE)
            for case in chosen:
                with monitor.stage(f"golden {case.name(writer.patch_size)}"):
                    eager.append(golden_case(writer, source, case, staged))
            if names:
                raise Ti2vExportError(
                    f"a run with narrowed cases ({len(chosen)} / {len(CASES)}) does not install the"
                    " series (only a series with every case is installed — each case's result is"
                    " in the [eager] lines)"
                )
    breakdown = storage_breakdown(graph)
    return {
        "series": str(target),
        "prepare": prepared,
        "inspection": inspection,
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "text_embeds": {
            "path": str(dit_probe.TEXT_EMBEDS_ASSET),
            "sha256": dit_reference.file_sha256(dit_probe.TEXT_EMBEDS_ASSET),
        },
        "eager": eager,
        "worst_reference_f32_vs_f64_ratio": max(r["reference_f32_vs_f64_ratio"] for r in eager),
        "stages": [record.to_dict() for record in monitor.records],
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Real-size goldens (step 2 — add goldens to the installed container)
# ---------------------------------------------------------------------------

#: The input keys of the installed `reference.<case>` that can be rebuilt from the case table (the
#: resume check).
_STORED_REFERENCE_INPUTS = ("latents", "timestep", CONDITION_TIMESTEP_KEY)

#: The output keys an installed case's files must hold (`io` / `reference`).
_STORED_IO_OUTPUT = f"{export_dit.OUTPUT_PREFIX}0"
_STORED_REFERENCE_OUTPUTS = ("output", export_dit.REFERENCE_F64_KEY)

#: What {@link stored_case_state} returns.
StoredState = Literal["absent", "partial", "present"]


def stored_case_state(
    writer: dit_reference.LayerwiseDit, case: Ti2vCase, target: Path
) -> StoredState:
    """The state of one case's golden in the component directory (the resume of the module doc's
    "Real-size goldens").

    When both files are present, the installed inputs (the 7 `input.*` of `io` and
    {@link _STORED_REFERENCE_INPUTS} of `reference`) are compared bit by bit with the values rebuilt
    from the current case table, and the output keys are checked to be present. MUST: fail loudly
    on a mismatch — counting an old golden from before a table change as "done" would silently
    give the band's decision cases and the acceptance cases different inputs (deleting it is a
    human decision).
    """
    name = case.name(writer.patch_size)
    io_name, reference_name = export_dit.case_file_names(name)
    present = [(target / file).is_file() for file in (io_name, reference_name)]
    if not any(present):
        return "absent"
    if not all(present):
        return "partial"
    inputs = case_inputs(writer, case)
    expected = {
        "latents": inputs.latents,
        "timestep": normalize_boundary_tensor(inputs.timestep, f"{name} timestep"),
        CONDITION_TIMESTEP_KEY: normalize_boundary_tensor(
            inputs.condition_timestep, f"{name} condition-side timestep"
        ),
    }
    differing: list[str] = []
    with safe_open(str(target / io_name), "pt") as handle:
        keys = set(handle.keys())
        for key, value in zip(INPUT_NAMES, inputs.graph_inputs, strict=True):
            stored = f"{export_dit.INPUT_PREFIX}{key}"
            built = normalize_boundary_tensor(value, f"{name} input '{key}'")
            if stored not in keys or not torch.equal(handle.get_tensor(stored), built):
                differing.append(stored)
        if _STORED_IO_OUTPUT not in keys:
            differing.append(f"{_STORED_IO_OUTPUT} (missing)")
    with safe_open(str(target / reference_name), "pt") as handle:
        keys = set(handle.keys())
        for key in _STORED_REFERENCE_INPUTS:
            if key not in keys or not torch.equal(handle.get_tensor(key), expected[key]):
                differing.append(f"reference.{key}")
        differing += [
            f"reference.{key} (missing)" for key in _STORED_REFERENCE_OUTPUTS if key not in keys
        ]
    if differing:
        raise Ti2vExportError(
            f"{target}: the golden of {name} disagrees with the case table ({differing}) — if"
            f" the table changed, delete the old {io_name} / {reference_name} and run again"
        )
    return "present"


def _stored_ratio(target: Path, name: str) -> float:
    """The ratio of an installed case's CPU f32 reference against f64 (for the summary).

    It is taken from the installed `output.f64` (rounded to f32), so it equals the denominator of
    the TS r check's normalization (the ratio in `golden_case`'s summary is against f64 before
    rounding, and differs by a few % on cases with a small denominator).
    """
    _, reference_name = export_dit.case_file_names(name)
    with safe_open(str(target / reference_name), "pt") as handle:
        return export_dit._ratio(
            handle.get_tensor("output"), handle.get_tensor(export_dit.REFERENCE_F64_KEY)
        )


def _append_log(log: Path, row: Mapping[str, Any]) -> None:
    """Write one case's result row, newline included, in one write and flush it to disk (a record —
    the resume marker is the files in the component directory)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_full(
    model: str = DEFAULT_MODEL, *, names: Sequence[str] = (), log: Path | None = None
) -> dict[str, Any]:
    """Add real-size goldens to the installed series (the module doc's "Real-size goldens" — the
    container is not rewritten).

    `names` narrows the cases by name (even narrowed, the written cases are installed — cases are
    added one at a time, so a long run can be split). With `log`, one JSON line is appended per
    written case.
    """
    directory = transformer_dir(model)
    target = SERIES / TARGET
    container = target / MODEL_FILE
    if not all(part.is_file() for part in container_parts(container)):
        raise Ti2vExportError(
            f"no installed container ({container}); install the series with write first"
        )
    writer = dit_reference.LayerwiseDit(
        dit_reference.load_config(directory),
        alignment=dit_reference.upstream_alignment(directory),
    )
    patch = writer.patch_size
    known = {case.name(patch) for case in FULL_CASES}
    if set(names) - known:
        raise Ti2vExportError(f"unknown case names: {sorted(set(names) - known)}")
    chosen = [case for case in FULL_CASES if not names or case.name(patch) in names]
    for stale in sorted(SERIES.glob(f"{FULL_STAGING_PREFIX}*")):
        print(f"[write-full] deleting the working directory {stale} of a previous run", flush=True)
        shutil.rmtree(stale)
    states = {case.name(patch): stored_case_state(writer, case, target) for case in chosen}
    pending = [case for case in chosen if states[case.name(patch)] != "present"]
    print(
        f"[write-full] {target}: of {len(chosen)} cases, {len(chosen) - len(pending)} done,"
        f" {len(pending)} to go (estimate per case {FULL_CASE_SECONDS} s,"
        f" environment {json.dumps(dit_reference.numeric_environment(), ensure_ascii=False)})",
        flush=True,
    )
    if pending:
        dit_probe.require_available("write-full", FULL_PEAK_BYTES)
    source = dit_reference.ContainerDitWeights(container)
    written: dict[str, dict[str, Any]] = {}
    elapsed: list[float] = []
    with MemoryMonitor() as monitor:
        for position, case in enumerate(pending, start=1):
            name = case.name(patch)
            each = statistics.mean(elapsed) if elapsed else FULL_CASE_SECONDS
            print(
                f"[write-full] {position}/{len(pending)} {name} ({states[name]}): starting —"
                f" remaining estimate {each * (len(pending) - position + 1) / 60:.0f} min",
                flush=True,
            )
            started = time.perf_counter()
            with (
                monitor.stage(f"golden {name}") as record,
                tempfile.TemporaryDirectory(dir=SERIES, prefix=FULL_STAGING_PREFIX) as scratch,
            ):
                report = golden_case(writer, source, case, Path(scratch))
                # Only a case that passed the checks is moved into the component directory (a
                # rename within the same file system).
                for file in export_dit.case_file_names(name):
                    os.replace(Path(scratch) / file, target / file)
            elapsed.append(time.perf_counter() - started)
            row = {
                **report,
                "role": case.role,
                "seconds": round(elapsed[-1], 1),
                "stage": record.to_dict(),
            }
            written[name] = row
            if log is not None:
                _append_log(log, row)
            print(
                f"[write-full] {position}/{len(pending)} {name}: installed {elapsed[-1]:.0f} s",
                flush=True,
            )
    cases = []
    for case in chosen:
        name = case.name(patch)
        row = written.get(name)
        cases.append(
            {
                "case": name,
                "role": case.role,
                "form": case.form,
                "status": "written" if row is not None else "present",
                "reference_f32_vs_f64_ratio": _stored_ratio(target, name),
                **({} if row is None else {"seconds": row["seconds"]}),
            }
        )
    return {
        "series": str(target),
        "cases": cases,
        "eager": list(written.values()),
        "text_embeds": {
            "path": str(dit_probe.TEXT_EMBEDS_ASSET),
            "sha256": dit_reference.file_sha256(dit_probe.TEXT_EMBEDS_ASSET),
        },
        "min_reference_f32_vs_f64_ratio": min(c["reference_f32_vs_f64_ratio"] for c in cases),
        "stages": [record.to_dict() for record in monitor.records],
        "environment": dit_reference.numeric_environment(),
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Eager equivalence with independent weights (the form of decision 4)
# ---------------------------------------------------------------------------


def golden_agreement(
    name: str,
    inputs: CaseInputs,
    reference: torch.Tensor,
    tokens_out: torch.Tensor,
    golden_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Compare the in-memory upstream reference and patched output with the installed golden (the
    summary fields and the mismatches).

    The 7 graph inputs are also compared bit by bit with the golden's `input.*` (if the same case
    was not built, comparing the outputs means nothing).
    """
    io_name, reference_name = export_dit.case_file_names(name)
    with safe_open(str(golden_dir / io_name), "pt") as handle:
        stored_inputs = {
            key: handle.get_tensor(f"{export_dit.INPUT_PREFIX}{key}") for key in INPUT_NAMES
        }
        stored_output = handle.get_tensor(f"{export_dit.OUTPUT_PREFIX}0")
    with safe_open(str(golden_dir / reference_name), "pt") as handle:
        stored_reference = handle.get_tensor("output")
    differing = [
        key
        for key, value in zip(INPUT_NAMES, inputs.graph_inputs, strict=True)
        if not torch.equal(
            normalize_boundary_tensor(value, f"{name} input '{key}'"), stored_inputs[key]
        )
    ]
    record = {
        "golden_inputs_bit_exact": not differing,
        "golden_reference_bit_exact": torch.equal(reference, stored_reference),
        "golden_reference_ratio": export_dit._ratio(reference, stored_reference),
        "golden_output_bit_exact": torch.equal(tokens_out, stored_output),
        "golden_output_ratio": export_dit._ratio(tokens_out, stored_output),
    }
    failures = []
    if differing:
        failures.append(
            f"graph inputs {differing} differ from the golden's io (the same case was not built)"
        )
    if not record["golden_reference_bit_exact"]:
        failures.append(
            "the upstream f32 reference is not bit-identical to output of the golden's reference"
            f" (ratio {record['golden_reference_ratio']:.3e})"
        )
    if not record["golden_output_bit_exact"]:
        failures.append(
            "the patched graph's output is not bit-identical to output.0 of the golden's io"
            f" (ratio {record['golden_output_ratio']:.3e})"
        )
    return record, failures


def full_eager_case(
    upstream: nn.Module,
    wrapper: dit_patch.WanDitTokensTi2v,
    writer: dit_reference.LayerwiseDit,
    case: Ti2vCase,
    golden_dir: Path,
) -> dict[str, Any]:
    """One case of eager equivalence with independent weights (the module doc's "Eager
    equivalence with independent weights").

    `writer` only builds the inputs (the RoPE tables and the time sinusoid — no weights are read).
    The reference and the patched eager run on the whole f32 `upstream` (fake-quant applied) and the
    `wrapper` around it. Passes when the returned field `failures` is empty.
    """
    name = case.name(writer.patch_size)
    inputs = case_inputs(writer, case)
    with torch.no_grad():
        started = time.perf_counter()
        reference = dit_patch.reference_dit(
            upstream,
            inputs.latents,
            inputs.timestep,
            inputs.encoder_hidden_states,
            inputs.condition,
        )
        seconds = time.perf_counter() - started
    built = export_dit.Case(
        name=name,
        spec=case.spec(),
        inputs=inputs.graph_inputs,
        latents=inputs.latents,
        timestep=inputs.timestep,
        reference=reference,
        reference_blocks=[],
        reference_seconds=seconds,
    )
    report, tokens_out = export_dit.eager_report(wrapper, upstream, built)
    golden, mismatches = golden_agreement(name, inputs, reference, tokens_out, golden_dir)
    report.update({"form": case.form, **golden})
    report["failures"] = export_dit.eager_failures(report) + mismatches
    print(f"[eager-full] {json.dumps(report, ensure_ascii=False)}", flush=True)
    return report


def eager_full(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """With one whole f32 fake-quant model, compare the patched graph with the upstream and also
    with the installed goldens (the module doc's "Eager equivalence with independent weights" —
    every case).

    MUST: before loading the upstream (about 21 GiB), check that the installed goldens are complete
    and check MemAvailable.
    """
    directory = transformer_dir(model)
    config = dit_reference.load_config(directory)
    writer = dit_reference.LayerwiseDit(config)
    target = SERIES / TARGET
    missing = [
        file
        for case in CASES
        for file in export_dit.case_file_names(case.name(writer.patch_size))
        if not (target / file).is_file()
    ]
    if missing:
        raise Ti2vExportError(
            f"the installed goldens are incomplete ({target} — {len(missing)} missing:"
            f" {missing[:3]}); install the series with write first"
        )
    estimate = dit_probe.eager_memory(config)["f32"] + dit_probe.EAGER_BASE_BYTES
    dit_probe.require_available("load", estimate)
    reports: list[dict[str, Any]] = []
    with MemoryMonitor() as monitor:
        with monitor.stage("load") as record:
            upstream = export_dit.load_transformer(model)
            record.details["parameters"] = sum(p.numel() for p in upstream.parameters())
        with monitor.stage("fake-quant") as record:
            wrapper = dit_patch.WanDitTokensTi2v(upstream)
            record.details["rounded"] = export_dit.fake_quant_i8(upstream, wrapper).describe()
        for case in CASES:
            with monitor.stage(f"eager {case.name(writer.patch_size)}"):
                reports.append(full_eager_case(upstream, wrapper, writer, case, target))
    failures = {report["case"]: report["failures"] for report in reports if report["failures"]}
    return {
        "series": str(target),
        "memory_estimate_gib": round(estimate / 2**30, 2),
        "cases": reports,
        "failures": failures,
        "stages": [record.to_dict() for record in monitor.records],
        "passed": not failures,
    }


def inspect_only(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Only the trace and the structure check (no weights are read — only the config).

    The storage check (the i8 count, the f32 remainder) is done on the graph of the container
    `write` wrote. What this checks is the linear count, the `where` placement and the input
    declarations (none depends on storage), and the number of quantization targets.
    """
    config = dit_reference.load_config(transformer_dir(model))
    graph, _, targets = trace(config)
    inspection = inspect_structure(graph, sorted(targets), int(config["num_layers"]))
    failures = inspection["failures"] + expected_counts(
        model, {**inspection, "i8_weights": len(targets)}
    )
    return {
        **inspection,
        "quant_targets": len(targets),
        "failures": failures,
        "passed": not failures,
    }


# ---------------------------------------------------------------------------
# Observing quantization (decision 5)
# ---------------------------------------------------------------------------

#: The observed models → each model's decision cases (S = 192). For the 1.3B, the 6 decision cases
#: of the i8 series (T2V only).
OBSERVE_MODELS = ("t2v-1.3b", DEFAULT_MODEL)

#: How many times the 1.3B's observed value the 5B's must exceed to be raised for a ruling (the
#: rule of thumb of decision 5).
ESCALATE_FACTOR = 2.0


def _observe_cases(
    model: str, patch: tuple[int, int, int]
) -> list[tuple[str, export_dit.CaseSpec, Form, str]]:
    """The observation cases (name, latent shape, T2V / I2V form, context prompt).

    For the 5B, the 6 decision cases of {@link CASES}. For the 1.3B, the 6 decision cases of the i8
    series (T2V only), with the prompts assigned in the same order as the 5B's decision cases (the
    context is built the same way on both models — a synthetic random context amplifies only the
    5B, {@link Ti2vCase}).
    """
    band = [case for case in CASES if case.role == "band"]
    if model == DEFAULT_MODEL:
        return [(case.name(patch), case.spec(), case.form, case.text) for case in band]
    specs = [spec for spec in export_dit.series_cases("i8", full=False) if spec.role == "band"]
    return [
        (spec.name(patch), spec, "t2v", case.text) for spec, case in zip(specs, band, strict=True)
    ]


def observe_quant(models: Sequence[str] = OBSERVE_MODELS) -> dict[str, Any]:
    """The difference of the f32 reference with i8 weights only against the unquantized f32
    reference (the module doc's "Observing quantization").

    Taken on the decision cases of the 5B and the 1.3B, it lists how many times the 1.3B's worst the
    5B's worst is.
    """
    results: dict[str, Any] = {}
    for model in models:
        directory = transformer_dir(model)
        config = dit_reference.load_config(directory)
        writer = dit_reference.LayerwiseDit(
            config, alignment=dit_reference.upstream_alignment(directory)
        )
        unquantized = dit_reference.CheckpointDitWeights(directory, "none")
        quantized = dit_reference.CheckpointDitWeights(
            directory, "i8", dit_reference.quant_keys(config)
        )
        rows = []
        for name, spec, form, prompt in _observe_cases(model, writer.patch_size):
            latents, timestep, text = dit_probe.probe_inputs(writer, spec, prompt)
            condition = None
            if form == "i2v":
                mask = dit_patch.dit_condition_mask(
                    spec.latent_shape, writer.patch_size, conditioned=True
                )[..., 0]
                condition = dit_patch.TimestepCondition(torch.zeros_like(timestep), mask)
            plain = writer.forward(
                unquantized, torch.float32, latents, timestep, text, condition=condition
            )
            rounded = writer.forward(
                quantized, torch.float32, latents, timestep, text, condition=condition
            )
            ratio = export_dit._ratio(rounded.output, plain.output)
            rows.append({"case": name, "form": form, "text": prompt, "ratio": ratio})
            print(f"[observe] {model} {name}: {ratio:.3e}", flush=True)
        ratios = [row["ratio"] for row in rows]
        results[model] = {
            "cases": rows,
            "worst": max(ratios),
            "median": statistics.median(ratios),
        }
    summary: dict[str, Any] = {"models": results}
    if set(OBSERVE_MODELS) <= set(results):
        factor = results[DEFAULT_MODEL]["worst"] / results["t2v-1.3b"]["worst"]
        summary["worst_factor_5b_over_1_3b"] = factor
        summary["escalate"] = factor > ESCALATE_FACTOR
    summary["passed"] = True
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "command", choices=("inspect", "write", "write-full", "observe-quant", "eager-full")
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="narrow by case name (write: the series is not installed; write-full: the written"
        " cases are installed)",
    )
    parser.add_argument("--out", type=Path, default=None, help="where the summaries go")
    args = parser.parse_args(argv)
    if args.case and args.command not in ("write", "write-full"):
        parser.error("--case applies only to write / write-full")
    out = default_out() if args.out is None else args.out
    started = time.perf_counter()
    if args.command == "inspect":
        summary = inspect_only()
    elif args.command == "write":
        summary = write_series(names=args.case)
    elif args.command == "write-full":
        summary = write_full(names=args.case, log=out / "write-full.jsonl")
    elif args.command == "eager-full":
        summary = eager_full()
    else:
        summary = observe_quant()
    document = {
        "command": args.command,
        "result": summary,
        "seconds": round(time.perf_counter() - started, 1),
        "environment": dit_reference.numeric_environment(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.command}.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(document, ensure_ascii=False, indent=1))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
