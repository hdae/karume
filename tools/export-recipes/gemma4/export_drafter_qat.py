"""Writes the q4_0 QAT MTP drafter of Gemma 4 E2B / E4B (`--model`) as the borrower graph of the
QAT product series, every drafter weight imported from the checkpoint's own 4-bit grid, with its
goldens.

    cd tools/export-recipes
    uv run --with 'transformers==5.14.1' python -m gemma4.export_drafter_qat              # E2B
    uv run --with 'transformers==5.14.1' python -m gemma4.export_drafter_qat --model e4b  # E4B

## The regular drafter's graph, from other weights

The graph is the regular drafter's (`gemma4.export_drafter`): the same wrapper (`DrafterWrapper` —
k = 3 unrolled steps, the full-vocabulary head + argmax with `use_ordered_embeddings=False`), the
same trace, the same surgery to the external-states form, the same shared declaration of the main
embedding table and the same form check (`assert_ir_form_drafter`). The lender is the QAT product
series (`gemma4.export_product_qat`, `outputs/series/gemma4-<model>-qat-product/`). What differs:

1. **Pinned sources** (`gemma4.sources`): the drafter is Google's
   `google/gemma-4-E{2,4}B-it-qat-q4_0-unquantized-assistant` (`QAT_DRAFTER_PINS`), the target's
   weights and tokenizer the QAT line's two pins (`QAT_PINS`). All three are checked before the
   load, and the record's full digests once more before it is written. Google's model card: with
   a QAT target, the assistant must be a QAT checkpoint of the same precision.
2. **Every drafter weight is imported from its q4_0 grid instead of rounded to i8**
   ({@link import_drafter}): the 22 linears (pre / post projection; per layer q_proj, o_proj,
   gate / up / down) and the head (`lm_head`, tied to the drafter's own `embed_tokens`
   `[262144, 256]` — one tensor). Measured 2026-11-03 on both pinned snapshots: every group of 32
   along the last axis recovers (`gemma4.export_product_qat.import_grid` — 12B's `grid_import`:
   codes in [−8, 7], one F32 step per group; a group that does not recover fails the export with
   the tensor and the group). The wrapper's F32 values are replaced by the grid's
   `dequantize_int4(codes, steps)` — what the container's bytes decode to — and written as fixed
   packed i4 g32; no i8 is left. `masked_embedding.centroids` is not on the grid and is not read
   (the sparse head is off).
3. **The borrowed table is the QAT target's i4 g32 table**: the shared declaration's storage is
   i4 ({@link SHARED_STORAGE}), and the lender's container must hold that table as i4 with group
   size {@link GROUP_SIZE} and come from the pinned QAT weights ({@link read_lender}). The
   drafter's own copy of it (golden-only, no bytes) is imported from the target checkpoint's grid
   by the same function the product series uses, and the goldens check it bit for bit against the
   target reference's table.
4. **The goldens are taken against the QAT target's reference**: `gemma4.export_product_qat`'s
   loader and grid import, i.e. the values its container decodes to (exact grid values), so they
   do not depend on how a runtime later stores either graph. The cases, the cycle count and the
   call convention are the regular drafter's (`gemma4.export_drafter.draft_case`).
5. E4B's drafter is the same graph at E4B's widths (backbone 2560, two KV heads): the lender's KV
   owner layers (E4B 22 / 23) come from the target's layer layout, the slot shapes from the
   lender's declaration.

## Why i4 here and i8 for the regular drafter

The regular E2B assistant's linears are not on any grid; rounding them to i4 g32 cut E[a] by
15–33% (2026-09-08 — `gemma4.export_drafter`). The QAT assistant's weights are the grid, so the
i4 container holds them within BF16 rounding. Measured 2026-11-03 the 2026-09-08 way — E[a] at
k = 3 (the mean accepted prefix: `tokens_per_cycle − 1` of
`gemma4.export_drafter.acceptance_stats`), one teacher-forced target trajectory per case,
N = 200 cycles × the three golden cases, both arms against the QAT target reference and
borrowing its table — the drafter with the checkpoint's BF16 values against the grid-imported
one:

| case (prompt tokens)     | E2B BF16 | E2B i4 | E4B BF16 | E4B i4 |
| ------------------------ | -------- | ------ | -------- | ------ |
| `short-en` (25)          | 0.55     | 0.55   | 0.54     | 0.54   |
| `readme-recipes` (2441)  | 1.22     | 1.23   | 1.325    | 1.325  |
| `readme-exporter` (3690) | 1.21     | 1.22   | 1.33     | 1.33   |

The two arms emit the same draft id in 99.8 / 99.3 / 99.3% (E2B) and 99.2 / 100 / 100% (E4B) of
the 600 positions of each case; no imported tensor is bit-identical to its BF16 values, the
largest move is 0.31% (E2B) / 0.37% (E4B) of the tensor's max |w|.

## Output layout

    outputs/series/gemma4-<model>-qat-drafter/model.krm                          the container
    outputs/series/gemma4-<model>-qat-drafter/drafter-golden.<case>.safetensors  expected drafts
    outputs/series/gemma4-<model>-qat-drafter/reference.json                     provenance

NOTE: no distribution reads the series yet; `gemma4.distribution` still requires the drafter
component to be i8 and refuses i4 in it.

MUST: transformers is imported inside functions (the loader's rule).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn

from _shared.container_read import open_container, read_layouts
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_drafter as drafter
from gemma4 import export_product_qat as qat
from gemma4 import provenance, sources
from gemma4.card import GEMMA4_LICENSE
from gemma4.distribution import GEMMA4_DRAFTER_ROLE
from gemma4.export_drafter import DrafterWrapper, LenderSlots
from gemma4.sources import HubPin, ModelPins
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.dist import NOTICE_FILENAME, ir_graph
from karume.emit import FixedQuantizedWeight, bakeable_initializers
from karume.ir import IrGraph, IrStorage
from karume.pipeline import publish_model
from karume.states import to_external_states_form
from karume.verify import VerifiedContainer
from qwen3_5.export import MEMORY_FLOOR_GIB, MemoryWatch

#: The series suffix (`outputs/series/gemma4-<model>-qat-drafter/` — the regular drafter's
#: `gemma4-<model>-drafter` with the QAT product's `qat-` in front). NOTE: no distribution reads the
#: series yet; when one does, the word moves next to `GEMMA4_DRAFTER_SUFFIX`.
QAT_DRAFTER_SUFFIX = "qat-drafter"

#: The q4_0 grid's group (the QAT product series' — the borrowed table must share it).
GROUP_SIZE = qat.GROUP_SIZE

#: The shared declaration's storage: i4, without a group size (a shared declaration names only the
#: dtype — the lender's resident weight holds the step and the group, `karume.verify`).
SHARED_STORAGE = IrStorage(dtype="i4")

#: The drafter files the provenance record fingerprints — exactly the files its pin names.
DRAFTER_FILES = ("model.safetensors", "config.json")

#: The borrowed table's wrapper key (`target_embed` of `gemma4.export_drafter.DrafterWrapper`).
TARGET_EMBED_WEIGHT = f"{drafter.TARGET_EMBED_MODULE}.weight"

#: The target checkpoint's main embedding key (`gemma4.export_drafter.load_target_embed`'s source).
TARGET_EMBED_CHECKPOINT_KEY = f"{one_shot.CHECKPOINT_TEXT_PREFIX}embed_tokens.weight"

#: The device the goldens run on. CPU: this workspace's torch is the CPU build, and E4B's F32
#: target (about 18 GiB besides the PLE tables) does not fit a 12 GiB card.
DEFAULT_DEVICE = "cpu"


class QatDrafterError(RuntimeError):
    """The drafter, the lender series or the target reference is not what the QAT line expects."""


# ---------------------------------------------------------------------------
# The grid import
# ---------------------------------------------------------------------------


def int4_seats(wrapper: nn.Module) -> set[str]:
    """The weights stored as packed i4: every `nn.Linear` of the wrapper (the drafter's 22 linears
    and its head `drafter.lm_head`)."""
    return {
        f"{name}.weight"
        for name, module in wrapper.named_modules()
        if isinstance(module, nn.Linear)
    }


def import_drafter(wrapper: DrafterWrapper) -> dict[str, FixedQuantizedWeight]:
    """Import the grid of every 2-D weight of `wrapper` in place
    (`gemma4.export_product_qat.import_grid`) and return the packed i4 of the stored ones, keyed by
    wrapper key ({@link int4_seats}). The borrowed table (`target_embed`) is imported too — its
    values become the target's stored table — but has no bytes of its own here.

    MUST: the 2-D parameters, tied names included, are exactly the seats plus the borrowed table —
    a resurrected `drafter.model.embed_tokens` (the head's tied twin) or a new 2-D weight upstream
    fails here instead of being imported twice or stored as F32 next to the grid.
    """
    seats = int4_seats(wrapper)
    planes = {
        name
        for name, parameter in wrapper.named_parameters(remove_duplicate=False)
        if parameter.dim() == 2
    }
    expected = seats | {TARGET_EMBED_WEIGHT}
    if planes != expected:
        raise QatDrafterError(
            f"2-D weights outside the grid import: {sorted(planes - expected)[:4]}"
            f" / seats that are not parameters: {sorted(expected - planes)[:4]}"
        )
    fixed = {
        key: qat.import_grid(wrapper.get_parameter(key).data, f"'{key}'") for key in sorted(seats)
    }
    import_target_table(wrapper)
    return fixed


def import_target_table(wrapper: DrafterWrapper) -> None:
    """Replace the borrowed table's values by its grid (the QAT product series imports the same
    tensor of the same checkpoint with the same function); its packed form is dropped."""
    qat.import_grid(
        wrapper.get_parameter(TARGET_EMBED_WEIGHT).data,
        f"'{TARGET_EMBED_CHECKPOINT_KEY}' (the target's table)",
    )


def container_material(
    graph: IrGraph, tensors: Mapping[str, torch.Tensor], fixed: Mapping[str, FixedQuantizedWeight]
) -> dict[str, torch.Tensor]:
    """The writer's tensors: meta for the fixed weights, the trace's F32 for the rest (norms,
    layer scalars, trace-made constants), for exactly the keys the graph stores bytes for.

    MUST: every fixed weight is declared by the graph (a linear the trace dropped would leave its
    grid unwritten without a word), and no rank-2 tensor is left outside them (it would ship as
    F32 next to the grid).
    """
    declared = {graph.initializers[name].tensor for name in bakeable_initializers(graph)}
    missing = sorted(set(fixed) - declared)
    if missing:
        raise QatDrafterError(f"imported weights the graph does not declare: {missing[:4]}")
    planes = sorted(key for key in declared - set(fixed) if tensors[key].dim() == 2)
    if planes:
        raise QatDrafterError(f"rank-2 tensors outside the grid import: {planes[:4]}")
    return {
        key: (torch.empty(tensors[key].shape, device="meta") if key in fixed else tensors[key])
        for key in sorted(declared)
    }


# ---------------------------------------------------------------------------
# The lender (the QAT product series)
# ---------------------------------------------------------------------------


def assert_shared_table(key: str, layout: str | None, group_size: int | None) -> None:
    """The lender stores its main table `key` the way the QAT drafter borrows it: i4 with group
    {@link GROUP_SIZE} (the regular product's i8 table and any other grid are refused)."""
    if (layout, group_size) != (SHARED_STORAGE.dtype, GROUP_SIZE):
        raise QatDrafterError(
            f"the lender's main table '{key}' is stored as {layout} with group {group_size} —"
            f" the QAT drafter borrows an {SHARED_STORAGE.dtype} g{GROUP_SIZE} table"
        )


def stored_group_size(container: VerifiedContainer, key: str) -> int | None:
    """The group length of tensor `key`'s stored encoding in a container (the declaration only —
    no payload is read; `_shared.container_read.read_stored` reads the same field)."""
    for bound in container.graphs.values():
        supply = bound.supplies.get(key)
        if supply is not None:
            return supply.encoding.group_size
    raise QatDrafterError(f"the lender container stores no tensor '{key}'")


def assert_lender_checkpoint(product_dir: Path, pins: ModelPins) -> None:
    """The lender series' provenance record names the pinned QAT weights (the record's full
    digests — `gemma4.export_product_qat` writes them), so the drafter is never paired with a
    product series of other weights whose table happens to have the same form."""
    path = product_dir / provenance.REFERENCE_FILE
    if not path.is_file():
        raise QatDrafterError(
            f"{path} is missing — export the QAT product series first"
            " (python -m gemma4.export_product_qat)"
        )
    files = json.loads(path.read_text(encoding="utf-8"))["checkpoint"]["files"]
    wrong = {
        name: (files.get(name, {}).get("sha256"), expected)
        for name, expected in sorted(pins.weights.sha256.items())
        if files.get(name, {}).get("sha256") != expected
    }
    if wrong:
        raise QatDrafterError(
            f"the lender series {product_dir.name} is not from {pins.weights.label}"
            f" (recorded, pinned): {wrong}"
        )


def read_lender(product_dir: Path, config: Any, target_config: Any, pins: ModelPins) -> LenderSlots:
    """The lender's slots and shared tensor key (`gemma4.export_drafter.lender_slots` — read from
    the container's declaration, its table's layout checked against {@link SHARED_STORAGE}), plus
    the table's group size ({@link assert_shared_table}) and the series' weights
    ({@link assert_lender_checkpoint})."""
    assert_lender_checkpoint(product_dir, pins)
    path = drafter.lender_container(product_dir)
    container = open_container(path)
    layouts = read_layouts(container)
    lender = drafter.lender_slots(
        ir_graph(path, container), layouts, config, target_config, shared_storage=SHARED_STORAGE
    )
    key = lender.shared_tensor
    assert_shared_table(key, layouts.get(key), stored_group_size(container, key))
    return lender


# ---------------------------------------------------------------------------
# The target reference
# ---------------------------------------------------------------------------


def load_target_reference(model_dir: Path) -> Any:
    """The QAT target as the QAT product series stores it: its loader and its grid import of every
    2-D text weight (`gemma4.export_product_qat` — the values the container decodes to). The packed
    form is dropped; only the wrapper's values are kept."""
    wrapper = qat.load_wrapper(model_dir)
    qat.import_wrapper(wrapper)
    gc.collect()
    return wrapper


def assert_borrowed_table(wrapper: DrafterWrapper, target: Any) -> None:
    """The drafter's copy of the borrowed table equals the target reference's table bit for bit
    (both imported from the same checkpoint tensor) — the goldens run the drafter on the table the
    lender holds, not on a second rounding of it."""
    own = wrapper.get_parameter(TARGET_EMBED_WEIGHT)
    lent = target.get_submodule(one_shot.EMBED_TOKENS_MODULE).weight
    if own.shape != lent.shape or not torch.equal(own.detach(), lent.detach()):
        raise QatDrafterError(
            "the drafter's copy of the borrowed table is not the target reference's table"
        )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def assert_pinned_files(files: Mapping[str, Any], pin: HubPin) -> None:
    """The drafter files the record names (every byte hashed) are the pinned ones, and the pin
    names exactly those files (a file the record names but the pin does not would go unchecked)."""
    if sorted(pin.sha256) != sorted(files):
        raise QatDrafterError(
            f"the drafter pin ({pin.label}) names {sorted(pin.sha256)}, the record {sorted(files)}"
        )
    wrong = {
        name: (files[name]["sha256"], expected)
        for name, expected in sorted(pin.sha256.items())
        if files[name]["sha256"] != expected
    }
    if wrong:
        raise QatDrafterError(f"the drafter read is not {pin.label} (actual, pin): {wrong}")


def drafter_record(
    out_dir: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    pins: ModelPins,
    drafter_dir: Path,
    drafter_pin: HubPin,
    goldens: Mapping[str, Any],
) -> dict[str, Any]:
    """`reference.json`: the regular drafter's record (`gemma4.export_drafter.drafter_reference`)
    with the target named by its two pinned sources (`gemma4.provenance.checkpoint_fingerprint`)
    and the drafter by its pin; both are checked against their pins before it is returned."""
    checkpoint = provenance.checkpoint_fingerprint(
        model_dir, pins.label, tokenizer_dir=tokenizer_dir, pins=pins
    )
    provenance.assert_pinned_fingerprint(checkpoint, pins)
    files = {name: provenance.file_digest(drafter_dir / name) for name in DRAFTER_FILES}
    assert_pinned_files(files, drafter_pin)
    return {
        "schema": provenance.SCHEMA,
        "series": out_dir.name,
        "checkpoint": checkpoint,
        "drafter": {
            "repo": drafter_pin.repo,
            "revision": drafter_pin.revision,
            "dir": drafter_pin.label,
            "files": files,
        },
        "goldens": dict(goldens),
    }


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    tokenizer_dir: Path,
    pins: ModelPins,
    drafter_dir: Path,
    drafter_pin: HubPin,
    product_dir: Path,
    steps: int = drafter.DRAFT_STEPS,
    cycles: int = drafter.GOLDEN_CYCLES,
    goldens: bool = True,
    device: str = DEFAULT_DEVICE,
    floor: float = MEMORY_FLOOR_GIB,
) -> dict[str, Any]:
    """Write the borrower container, the goldens and the provenance record through one staged
    seat, and return the summary.

    `model_dir` / `tokenizer_dir` are the QAT target's two sources (`pins`), `drafter_dir` the QAT
    drafter's snapshot (`drafter_pin`), `product_dir` the lender (the QAT product series).

    MUST: every gate (the pins, the lender, the form, the borrowed table, the goldens, the record's
    digests) passes before the seat is published. MUST: the grid import precedes the trace and the
    goldens (ADR 0006) — the container and the goldens run on the same values.
    """
    sources.assert_model_pins(model_dir, tokenizer_dir, pins)
    sources.assert_snapshot_pin(drafter_dir, drafter_pin)
    with MemoryWatch(floor) as monitor:
        with monitor.stage("load"):
            print(
                "[load] QAT drafter (no sparse head) + the target's table",
                file=sys.stderr,
                flush=True,
            )
            wrapper = drafter.load_drafter(steps, model_dir, source=drafter_dir)
        config = wrapper.drafter.config
        text = config.get_text_config()
        target_config = one_shot.load_text_config(model_dir)
        with monitor.stage("grid") as record:
            print(
                "[grid] import every drafter weight and the borrowed table",
                file=sys.stderr,
                flush=True,
            )
            fixed = import_drafter(wrapper)
            record.details["fixed"] = len(fixed)
        lender = read_lender(product_dir, config, target_config, pins)
        example = drafter.example_inputs(wrapper, decode.rope_specs(text))

        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with staged_publication(out_dir) as staged:
            staged.mkdir()
            with monitor.stage("trace") as record:
                print(
                    "[export] torch.export → external states → shared", file=sys.stderr, flush=True
                )
                graph, tensors = drafter.export_module_with_attention(wrapper, example)
                surgical = to_external_states_form(
                    graph, drafter.external_states_plan(graph, text, lender)
                )
                shared_graph, remaining, shared_name = drafter.share_target_embedding(
                    surgical, tensors, lender, target_config, storage=SHARED_STORAGE
                )
                record.details["nodes"] = len(shared_graph.nodes)
            with monitor.stage("write"):
                print("[export] write the container", file=sys.stderr, flush=True)
                verified = publish_model(
                    staged / one_shot.MODEL_FILE,
                    shared_graph,
                    container_material(shared_graph, remaining, fixed),
                    provenance=Provenance(
                        license=GEMMA4_LICENSE,
                        notice=NOTICE_FILENAME,
                        upstream_revision=drafter_pin.revision,
                    ),
                    # The component name (the regular drafter's — container-v1 §2.1).
                    graph_name=GEMMA4_DRAFTER_ROLE,
                    weight_dtype="f32",
                    fixed_weights=fixed,
                )
            form = drafter.assert_ir_form_drafter(verified, config, lender, steps)
            # MUST: the stored weights are i4 and nothing is i8 — an i8 head leaves the graph's
            # shapes and ops unchanged. The shared declaration counts as one more i4.
            if form["storage"].get("i4") != len(fixed) + 1 or "i8" in form["storage"]:
                raise QatDrafterError(
                    f"storage {form['storage']}: expected {len(fixed)} fixed i4 + the shared i4"
                )

            golden_summary: dict[str, Any] = {}
            if goldens:
                cases = drafter.build_cases(tokenizer_dir)
                with monitor.stage("reference"):
                    print(
                        "[golden] the QAT target reference (grid import)",
                        file=sys.stderr,
                        flush=True,
                    )
                    target = load_target_reference(model_dir)
                    assert_borrowed_table(wrapper, target)
                with monitor.stage("goldens"):
                    golden_summary = drafter.goldens_against(
                        wrapper, target, cases, staged, cycles, device
                    )
                del target
                gc.collect()
            with monitor.stage("provenance"):
                print("[provenance] fingerprints → reference.json", file=sys.stderr, flush=True)
                provenance.write_record(
                    staged,
                    drafter_record(
                        out_dir,
                        model_dir,
                        tokenizer_dir,
                        pins,
                        drafter_dir,
                        drafter_pin,
                        golden_summary,
                    ),
                )

    parts = container_parts(out_dir / one_shot.MODEL_FILE)
    return {
        "dir": str(out_dir),
        "lender": str(product_dir),
        "steps": steps,
        "nodes": len(verified.nodes),
        "initializers": len(verified.initializers),
        "shared_initializer": shared_name,
        "model_bytes": sum(path.stat().st_size for path in parts),
        "parts": [path.stat().st_size for path in parts],
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "quantized": {"i4": len(fixed)},
        "form": form,
        "goldens": golden_summary,
        "stages": [stage.to_dict() for stage in monitor.records],
    }


def build_parser() -> argparse.ArgumentParser:
    """The CLI: the family's skeleton without `--sym-max` (the drafter has no chunk symbol), the
    models with a QAT drafter, and every source defaulting to its pin."""
    parser = one_shot.series_parser(
        __doc__.split("\n\n")[0],
        partial(one_shot.series_dir, suffix=QAT_DRAFTER_SUFFIX),
        with_sym_max=False,
        models=tuple(sorted(sources.QAT_DRAFTER_PINS)),
    )
    # The weights' default is the QAT line's pinned snapshot, not the regular line's hand-placed
    # checkpoint (`series_parser` registers that one).
    one_shot.add_model_default(parser, "model_dir", sources.qat_checkpoint_dir)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    one_shot.add_model_default(parser, "tokenizer_dir", sources.qat_tokenizer_dir)
    parser.add_argument("--drafter-dir", type=Path, default=None)
    one_shot.add_model_default(parser, "drafter_dir", sources.qat_drafter_dir)
    parser.add_argument("--product-dir", type=Path, default=None)
    one_shot.add_model_default(
        parser, "product_dir", partial(one_shot.series_dir, suffix=qat.QAT_PRODUCT_SUFFIX)
    )
    parser.add_argument("--steps", type=int, default=drafter.DRAFT_STEPS)
    parser.add_argument("--cycles", type=int, default=drafter.GOLDEN_CYCLES)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--no-goldens", dest="goldens", action="store_false")
    parser.add_argument("--memory-floor", dest="floor", type=float, default=MEMORY_FLOOR_GIB)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model = args.model
    options = one_shot.resolve_series_args(args)
    model_dir = options.pop("model_dir")
    summary = export_series(
        model_dir,
        options.pop("out"),
        pins=sources.qat_pins(model),
        drafter_pin=sources.qat_drafter_pin(model),
        **options,
    )
    print(json.dumps({"model_dir": str(model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
