"""Synthetic model and generator of the cross-language fixture.

It lives in `packages/runtime/tests/fixtures/container/` (the single form `synthetic.krm` and the
split form `synthetic-split-NNNNN-of-NNNNN.krm`).

The asset shows that the TS reader (`packages/runtime/src/format/container/*.ts`) can open a
container written **outside this machine**: `packages/runtime/tests/container_fixture_test.ts`
opens it with `openContainer` and checks the sha256 of every block and the canonical serialization
of the descriptors.

Regeneration is one command:

```
KARUME_FIXTURE=write uv run pytest tests/test_container.py -k fixture
```

The synthetic content is chosen only to take each branch of the reader once:

- a const region (two entries, f32 and i32) and weights (f16 / int8-sym / int4-sym-g)
- a payload that needs trailing padding (42 bytes of f16 → 44 bytes)
- a weight that exceeds the block limit (256 B in this fixture) and becomes a piece list
- a per-channel scale with `rowAxis: 1` (the `conv_transpose1d` shape)
- a `shared` declaration (an initializer outside the set the binding table is matched against)
- a layout that crosses the part length (512 B) (the parts split into 3)

The split form ({@link SPLIT_FIXTURE_PATH}) adds the three things the single form cannot take:

- **assets** (`ple_index` 64 B and `rope_base` 37 B — the latter is an odd length that needs 0x00
  padding) sit in a dedicated part after the weight parts
- **the const region is empty** and part 1 is a **0-byte file**
- the weight parts split into **2** (5 files including parts 0 / 1)

MUST: payloads are made from a **simple deterministic pattern** ({@link pattern_bytes}), not a
random generator — the same input gives the same fixture, and the TS test can spell its expected
values without carrying a copy of the generator.
"""

from __future__ import annotations

import os
from pathlib import Path

from karume.container import (
    AssetInput,
    Encoding,
    Provenance,
    container_paths,
    write_model_container,
)
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrShared, IrStorage, IrValue

#: Repository root (tools/exporter/tests → tools/exporter → tools → repo).
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Location of the cross-language fixture (read by the TS test).
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "runtime" / "tests" / "fixtures" / "container" / "synthetic.krm"
)

#: Representative path of the split-form fixture (the real files are numbered
#: `-NNNNN-of-NNNNN` — from part 0).
SPLIT_FIXTURE_PATH = FIXTURE_PATH.with_name("synthetic-split.krm")

#: Part count of the split-form fixture (part 0 + const [0 bytes] + 2 weight parts + 1 asset part).
SPLIT_FIXTURE_PARTS = 5

#: Graph name of the synthetic model.
GRAPH_NAME = "synthetic"

#: Graph name of the split-form fixture.
SPLIT_GRAPH_NAME = "synthetic-split"

#: Sizes lowered for the fixture (the real ones are 32 MiB / 256 MiB). Takes piece splitting and
#: part crossing in a few KiB.
FIXTURE_BLOCK_BYTES = 256
FIXTURE_PART_BYTES = 512


def pattern_bytes(length: int, seed: int) -> bytes:
    """Bytes rising by 1 from `seed` (`seed` differs per tensor so a mix-up shows).

    MUST: **no random generator** — keep the form such that the TS test can spell its expected
    values without copying the Python generator (with a copy, rebuilding the fixture would turn
    red as "the mirror drifted" rather than "the reader contract broke").
    """
    return bytes((seed + index) & 0xFF for index in range(length))


def synthetic_graph() -> IrGraph:
    """Synthetic IR v1 graph (the form before `ir_v2_document` renames to the real keys)."""
    return IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 8])],
        outputs=["h"],
        initializers={
            "p_enc_w": IrInitializer(
                tensor="enc.weight",
                storage=IrStorage(dtype="i4", scale="enc.weight_scale", group_size=32),
            ),
            "p_dec_w": IrInitializer(tensor="dec.weight", storage=IrStorage(dtype="f16")),
            "p_big_w": IrInitializer(
                tensor="big.weight", storage=IrStorage(dtype="i8", scale="big.weight_scale")
            ),
            "p_conv_w": IrInitializer(
                tensor="conv.weight", storage=IrStorage(dtype="i8", scale="conv.weight_scale")
            ),
            "const_a1b2c3d4e5f60718": IrInitializer(
                tensor="const.a1b2c3d4e5f60718", storage=IrStorage(dtype="f32")
            ),
            "const_b1b2c3d4e5f6071a": IrInitializer(
                tensor="const.b1b2c3d4e5f6071a", storage=IrStorage(dtype="i32")
            ),
            "p_lm_head": IrInitializer(
                shared=IrShared(tensor="lm_head.weight"), storage=IrStorage(dtype="f16")
            ),
        },
        values={
            "p_enc_w": IrValue(dtype="f32", shape=[8, 32]),
            "p_dec_w": IrValue(dtype="f32", shape=[3, 7]),
            "p_big_w": IrValue(dtype="f32", shape=[16, 32]),
            "p_conv_w": IrValue(dtype="f32", shape=[4, 3, 8]),
            "const_a1b2c3d4e5f60718": IrValue(dtype="f32", shape=[8]),
            "const_b1b2c3d4e5f6071a": IrValue(dtype="i32", shape=[4]),
            "p_lm_head": IrValue(dtype="f32", shape=[8, 8]),
            "h": IrValue(dtype="f32", shape=["T", 32]),
        },
        nodes=[IrNode(op="matmul", ins=["x", "p_enc_w"], outs=["h"], attrs={})],
    )


def synthetic_bindings() -> dict[str, Encoding]:
    """Tensor key → storage spec (covers every initializer that is not shared)."""
    return {
        "enc.weight": Encoding(
            "int4-sym-g", group_size=32, row_axis=0, scale_key="enc.weight_scale"
        ),
        "dec.weight": Encoding("f16"),
        "big.weight": Encoding("int8-sym", group_size=32, row_axis=0, scale_key="big.weight_scale"),
        # conv_transpose1d shape: the row axis is 1 (the per-channel scale has length shape[1]).
        "conv.weight": Encoding(
            "int8-sym", group_size=32, row_axis=1, scale_key="conv.weight_scale"
        ),
        "const.a1b2c3d4e5f60718": Encoding("f32"),
        "const.b1b2c3d4e5f6071a": Encoding("i32"),
    }


def synthetic_tensors() -> dict[str, bytes]:
    """Tensor key → raw bytes (exactly the length the declared shape and codec give)."""
    return {
        # i4: 8 x 32 = 256 elements → 128 bytes; the scale is f32 [8, 1].
        "enc.weight": pattern_bytes(128, 21),
        "enc.weight_scale": pattern_bytes(8 * 4, 22),
        # f16: 3 x 7 = 21 elements → 42 bytes (the writer writes 2 bytes of trailing padding).
        "dec.weight": pattern_bytes(42, 23),
        # i8: 16 x 32 = 512 bytes → 2 pieces at the 256 B block limit (8 rows each).
        "big.weight": pattern_bytes(512, 24),
        "big.weight_scale": pattern_bytes(16 * 4, 25),
        # i8 rowAxis 1: 4 x 3 x 8 = 96 bytes; the scale is f32 [3, 1].
        "conv.weight": pattern_bytes(96, 26),
        "conv.weight_scale": pattern_bytes(3 * 4, 27),
        "const.a1b2c3d4e5f60718": pattern_bytes(8 * 4, 31),
        "const.b1b2c3d4e5f6071a": pattern_bytes(4 * 4, 32),
    }


def write_synthetic_container(path: Path, *, single: bool = True) -> list[Path]:
    """Write the synthetic model to `path` (single form by default)."""
    return write_model_container(
        path,
        synthetic_graph(),
        synthetic_tensors(),
        synthetic_bindings(),
        graph_name=GRAPH_NAME,
        provenance=Provenance(license="apache-2.0", writer="karume"),
        part_bytes=FIXTURE_PART_BYTES,
        block_bytes=FIXTURE_BLOCK_BYTES,
        single=single,
    )


def build_fixture_bytes(tmp_path: Path) -> bytes:
    """The fixture bytes (`tmp_path` is the scratch space for writing)."""
    written = write_synthetic_container(tmp_path / FIXTURE_PATH.name)
    return written[0].read_bytes()


# ---------------------------------------------------------------------------
# Split-form fixture (assets + empty const + 2 weight parts)
# ---------------------------------------------------------------------------


def split_graph() -> IrGraph:
    """Synthetic graph with no const at all (the form where part 1 has length 0)."""
    return IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 4])],
        outputs=["h"],
        initializers={
            "p_a": IrInitializer(tensor="a.weight", storage=IrStorage(dtype="f32")),
            "p_b": IrInitializer(tensor="b.weight", storage=IrStorage(dtype="f32")),
            "p_c": IrInitializer(tensor="c.weight", storage=IrStorage(dtype="f32")),
        },
        values={
            "p_a": IrValue(dtype="f32", shape=[4, 16]),
            "p_b": IrValue(dtype="f32", shape=[4, 16]),
            "p_c": IrValue(dtype="f32", shape=[4, 16]),
            "h": IrValue(dtype="f32", shape=["T", 16]),
        },
        nodes=[IrNode(op="matmul", ins=["x", "p_a"], outs=["h"], attrs={})],
    )


def split_bindings() -> dict[str, Encoding]:
    return {name: Encoding("f32") for name in ("a.weight", "b.weight", "c.weight")}


def split_tensors() -> dict[str, bytes]:
    """Three f32 4 x 16 = 256-byte tensors (exactly the block limit; split into 2 parts at 512)."""
    return {
        "a.weight": pattern_bytes(4 * 16 * 4, 41),
        "b.weight": pattern_bytes(4 * 16 * 4, 42),
        "c.weight": pattern_bytes(4 * 16 * 4, 43),
    }


def split_asset_payloads() -> dict[str, bytes]:
    """Asset name → payload. `rope_base` is 37 bytes (an odd length needing 3 bytes of 0x00)."""
    return {"ple_index": pattern_bytes(64, 51), "rope_base": pattern_bytes(37, 52)}


def split_assets() -> dict[str, AssetInput]:
    """Asset name → input. The role names the models-side interpreter (the runtime ignores it).

    Both are read-whole assets, so no dedicated part each is needed (the two share one part).
    """
    payloads = split_asset_payloads()
    return {
        "ple_index": AssetInput("ple-index", len(payloads["ple_index"]), payloads["ple_index"]),
        "rope_base": AssetInput("rope-base", len(payloads["rope_base"]), payloads["rope_base"]),
    }


def write_split_container(directory: Path) -> list[Path]:
    """Write the split-form synthetic model to `directory` (returns the real files from part 0)."""
    return write_model_container(
        directory / SPLIT_FIXTURE_PATH.name,
        split_graph(),
        split_tensors(),
        split_bindings(),
        graph_name=SPLIT_GRAPH_NAME,
        provenance=Provenance(license="apache-2.0", writer="karume"),
        assets=split_assets(),
        part_bytes=FIXTURE_PART_BYTES,
        block_bytes=FIXTURE_BLOCK_BYTES,
        single=False,
    )


def split_fixture_paths() -> list[Path]:
    """Real files of the split-form fixture (in the resources location, from part 0)."""
    return container_paths(SPLIT_FIXTURE_PATH, SPLIT_FIXTURE_PARTS)


def fixture_write_requested() -> bool:
    """`KARUME_FIXTURE=write` — rewrite the resources only on an explicit rebuild request."""
    return os.environ.get("KARUME_FIXTURE") == "write"
