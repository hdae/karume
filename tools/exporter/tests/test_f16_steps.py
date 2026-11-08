"""The codec `int4-sym-g-f16` on the exporter side (ADR 0145 decision 3): the ledger mirror, the
scale byte count, the writer and the reader, the IR vocabulary and the f16 step table of emit.

The codec has the packed bytes of `int4-sym-g` and one f16 step per group (two per u32 word, step
`2i` in the low half), under its own layout `i4f16`. Only `expert_linear` reads it, and no op does
yet, so the runtime support check and the fixed-weight emit refuse it for every consumer.

Expected byte counts and f16 bit patterns are written out by hand (deriving them from the code
under test would make the checks tautological).
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import RUNTIME_FIXTURES
from test_container import edit_model_document

from karume.container import (
    CODEC_LEDGER,
    SCALE_DTYPES,
    ContainerFormatError,
    Encoding,
    Provenance,
    container_bindings,
    read_container,
    scale_bytes,
    write_model_container,
)
from karume.emit import (
    F16_MIN_NORMAL,
    EmitError,
    FixedQuantizedWeight,
    f16_step_table,
    pack_int4,
    storage_breakdown,
    stored_model,
)
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.verify import (
    ContainerError,
    IrError,
    assert_ir_accepted,
    assert_runtime_support,
    parse_ir_graph,
    verify_container,
)

CODECS_TS = RUNTIME_FIXTURES.parents[1] / "src" / "format" / "container" / "codecs.ts"

REFUSAL = re.escape("int4-sym-g-f16 (f16 steps) is read by expert_linear only — ADR 0145")

PROVENANCE = Provenance(license="apache-2.0", writer="karume")


def pattern_bytes(length: int, seed: int) -> bytes:
    """Bytes counting up from `seed` (distinct per tensor, so a mix-up shows)."""
    return bytes((seed + index) & 0xFF for index in range(length))


def f16_bytes(*bits: int) -> bytes:
    """f16 bit patterns, little endian (the bytes a u32 word holds, step `2i` in the low half)."""
    return struct.pack(f"<{len(bits)}H", *bits)


def i4f16_graph(shape: list[int], storage: str = "i4f16") -> IrGraph:
    """One graph `y = matmul(x, w)` with `w` of `shape` (the consumer does not matter here)."""
    return IrGraph(
        symbols=[],
        inputs=[IrInput(name="x", dtype="f32", shape=[2, shape[0]])],
        outputs=["y"],
        initializers={
            "p_w": IrInitializer(
                tensor="w.weight",
                storage=IrStorage(dtype=storage, scale="w.weight_scale", group_size=16),
            )
        },
        values={
            "p_w": IrValue(dtype="f32", shape=shape),
            "y": IrValue(dtype="f32", shape=[2, shape[1]]),
        },
        nodes=[IrNode(op="matmul", ins=["x", "p_w"], outs=["y"], attrs={})],
    )


def write_i4(
    path: Path,
    shape: list[int],
    scale: bytes,
    *,
    codec: str = "int4-sym-g-f16",
    block_bytes: int = 512,
    single: bool = True,
) -> list[Path]:
    """Write `w` (packed i4 bytes of `shape`, group 16) with the companion `scale` bytes."""
    storage = "i4f16" if codec == "int4-sym-g-f16" else "i4"
    return write_model_container(
        path,
        i4f16_graph(shape, storage),
        {"w.weight": pattern_bytes(shape[0] * shape[1] // 2, 11), "w.weight_scale": scale},
        {"w.weight": Encoding(codec, group_size=16, row_axis=0, scale_key="w.weight_scale")},
        graph_name="g",
        provenance=PROVENANCE,
        part_bytes=4096,
        block_bytes=block_bytes,
        single=single,
    )


class TestTheLedgerMirror:
    """The Python ledger holds the codec as the TS ledger does (`codecs.ts`)."""

    def test_it_is_int4_sym_g_with_its_own_layout_and_f16_steps(self) -> None:
        sibling = CODEC_LEDGER["int4-sym-g-f16"]
        base = CODEC_LEDGER["int4-sym-g"]

        assert replace(sibling, layout=base.layout, scale_dtype=base.scale_dtype) == base
        assert (sibling.layout, sibling.scale_dtype) == ("i4f16", "f16")
        assert base.scale_dtype == "f32"

    def test_the_scale_dtypes_match_the_runtime_ledger(self) -> None:
        """Each entry's `scaleDtype` on the TS side (read with regular expressions, as the existing
        ledger cross-check does — the TS side is immutable data in the repository)."""
        source = CODECS_TS.read_text(encoding="utf-8")
        raw = re.search(r"const RAW = \(.*?scaleDtype: (\w+)", source, re.S)
        assert raw is not None and raw.group(1) == "undefined", "cannot read the RAW helper"
        theirs: dict[str, str | None] = {
            name: None for name in re.findall(r'\["([\w-]+)", RAW\(', source)
        }
        for name, body in re.findall(r'\[\s*"([\w-]+)",\s*\{(.*?)\n\s*\},?\s*\]', source, re.S):
            found = re.search(r'scaleDtype: "(\w+)"', body)
            assert found is not None, f"{name}: cannot read scaleDtype"
            theirs[name] = found.group(1)

        assert theirs == {name: entry.scale_dtype for name, entry in CODEC_LEDGER.items()}

    def test_the_accepted_scale_dtypes_match_the_runtime_ones(self) -> None:
        source = CODECS_TS.read_text(encoding="utf-8")
        found = re.search(r"export const SCALE_DTYPES = \[(.*?)\] as const;", source)
        assert found is not None, "cannot read SCALE_DTYPES"

        assert tuple(re.findall(r'"(\w+)"', found.group(1))) == SCALE_DTYPES


class TestTheScaleByteCount:
    @pytest.mark.parametrize(
        ("codec", "counts", "expected"),
        [
            ("int4-sym-g", [0, 1, 2, 3, 4], [0, 4, 8, 12, 16]),
            # 2 bytes per f16 step, an odd count rounded up to a whole u32 word.
            ("int4-sym-g-f16", [0, 1, 2, 3, 4, 5], [0, 4, 4, 8, 8, 12]),
        ],
    )
    def test_it_counts_the_steps_of_the_codecs_scale_dtype(
        self, codec: str, counts: list[int], expected: list[int]
    ) -> None:
        assert [scale_bytes(codec, count) for count in counts] == expected

    def test_a_codec_without_a_scale_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="is not quantized"):
            scale_bytes("f32", 1)


class TestTheWriterAndTheReader:
    @pytest.mark.parametrize(
        ("rows", "padded"), [(3, 8), (4, 8), (5, 12)], ids=["odd", "even", "odd-5"]
    )
    def test_the_writer_declares_f16_steps_padded_to_a_whole_word(
        self, tmp_path: Path, rows: int, padded: int
    ) -> None:
        scale = pattern_bytes(padded, 21)
        paths = write_i4(tmp_path / "model.krm", [rows, 16], scale)

        verified = verify_container(paths, blocks=True)
        supply = verified.graphs["g"].supplies["w.weight"]
        assert supply.encoding.codec == "int4-sym-g-f16"
        assert supply.scale is not None and supply.scale.payload_bytes == padded
        assert verified.read.block(supply.scale.id)[:padded] == scale
        binding = json.loads(verified.read.model_descriptor_bytes)["binding"]["g"]["w.weight"]
        assert binding["encoding"]["scale"]["dtype"] == "f16"

    def test_the_writer_refuses_an_odd_step_table_without_its_word_padding(
        self, tmp_path: Path
    ) -> None:
        # 3 steps take 8 bytes (6 + 2 of padding), not 6.
        with pytest.raises(ContainerFormatError, match="differ from the 8 bytes"):
            write_i4(tmp_path / "model.krm", [3, 16], pattern_bytes(6, 21))

    def test_an_odd_step_table_splits_on_whole_rows_padding_only_the_last(
        self, tmp_path: Path
    ) -> None:
        """[9, 48] with group 16: 3 steps (6 bytes) per row, 27 steps = 56 bytes once padded. At a
        48-byte block limit the data (24 bytes per row) is cut 2 rows per piece and the scale into
        12-byte pieces on the same rows; the last piece is one row of 6 bytes. A scale row is 6
        bytes, not 56 / 9."""
        scale = pattern_bytes(54, 31) + b"\x00\x00"
        paths = write_i4(tmp_path / "model.krm", [9, 48], scale, block_bytes=48)

        verified = verify_container(paths, blocks=True)
        supply = verified.graphs["g"].supplies["w.weight"]
        assert supply.scale is not None and supply.scale.pieces is not None
        assert [(piece.rows, piece.payload_bytes) for piece in supply.scale.pieces] == [
            ((0, 2), 12),
            ((2, 4), 12),
            ((4, 6), 12),
            ((6, 8), 12),
            ((8, 9), 6),
        ]
        assert supply.scale.payload_bytes == 56
        assert verified.read.block(supply.scale.id) == scale

    @pytest.mark.parametrize(
        ("codec", "scale_length", "declared", "message"),
        [
            (
                "int4-sym-g-f16",
                8,
                "f32",
                "is not in the accepted set of codec 'int4-sym-g-f16' (f16 steps): 'f32'",
            ),
            (
                "int4-sym-g",
                16,
                "f16",
                "is not in the accepted set of codec 'int4-sym-g' (f32 steps): 'f16'",
            ),
        ],
        ids=["f16-codec-f32-scale", "f32-codec-f16-scale"],
    )
    def test_the_reader_refuses_a_scale_dtype_its_codec_does_not_imply(
        self, tmp_path: Path, codec: str, scale_length: int, declared: str, message: str
    ) -> None:
        paths = write_i4(
            tmp_path / "model.krm",
            [4, 16],
            pattern_bytes(scale_length, 21),
            codec=codec,
            single=False,
        )

        def respell(document: dict) -> None:
            document["binding"]["g"]["w.weight"]["encoding"]["scale"]["dtype"] = declared

        edit_model_document(paths, respell)

        with pytest.raises(ContainerFormatError, match=re.escape(message)):
            read_container(paths)


class TestTheIrVocabulary:
    """`i4f16` is in the IR vocabulary (a container that carries it reads back) and outside the
    runtime support table (no op reads its f16 steps yet)."""

    @staticmethod
    def _parse(storage: dict) -> IrGraph:
        return parse_ir_graph(
            json.dumps(
                {
                    "format": "karume-ir",
                    "version": 1,
                    "requires": {"ops": ["matmul"]},
                    "symbols": [],
                    "inputs": [{"name": "x", "dtype": "f32", "shape": [2, 4]}],
                    "outputs": ["y"],
                    "initializers": {"w": {"tensor": "enc.w", "storage": storage}},
                    "values": {
                        "w": {"dtype": "f32", "shape": [4, 32]},
                        "y": {"dtype": "f32", "shape": [2, 32]},
                    },
                    "nodes": [{"op": "matmul", "ins": ["x", "w"], "outs": ["y"], "attrs": {}}],
                }
            )
        )

    def test_a_group_quantized_declaration_is_accepted(self) -> None:
        graph = self._parse({"dtype": "i4f16", "scale": "enc.s", "group_size": 16})

        assert graph.initializers["w"].storage == IrStorage("i4f16", "enc.s", 16)

    def test_its_group_size_is_required(self) -> None:
        with pytest.raises(IrError, match="storage dtype 'i4f16' requires group_size"):
            self._parse({"dtype": "i4f16", "scale": "enc.s"})

    @pytest.mark.parametrize("group_size", [24, 8], ids=["not-a-power-of-two", "below-16"])
    def test_a_group_size_outside_the_accepted_set_is_refused(self, group_size: int) -> None:
        with pytest.raises(IrError, match="is not a power of 2 that is 16 or more"):
            self._parse({"dtype": "i4f16", "scale": "enc.s", "group_size": group_size})

    def test_the_runtime_support_check_refuses_it(self) -> None:
        graph = self._parse({"dtype": "i4f16", "scale": "enc.s", "group_size": 16})

        with pytest.raises(ContainerError, match=r"unsupported storage dtype 'i4f16' \(1\): w"):
            assert_runtime_support(graph)

    def test_a_container_that_carries_it_reads_back_and_fails_the_acceptance_rules(
        self, tmp_path: Path
    ) -> None:
        paths = write_i4(tmp_path / "model.krm", [4, 16], pattern_bytes(8, 21))

        with pytest.raises(ContainerError, match=r"unsupported storage dtype 'i4f16' \(1\)"):
            assert_ir_accepted(read_container(paths))

    def test_the_binding_derivation_maps_it_to_the_codec(self) -> None:
        assert container_bindings(i4f16_graph([3, 16])) == {
            "w.weight": Encoding(
                "int4-sym-g-f16", group_size=16, row_axis=0, scale_key="w.weight_scale"
            )
        }

    @pytest.mark.parametrize(("rows", "scale"), [(3, 8), (4, 8), (5, 12)])
    def test_the_storage_breakdown_counts_its_padded_f16_steps(self, rows: int, scale: int) -> None:
        breakdown = storage_breakdown(i4f16_graph([rows, 16]))

        assert (breakdown.compressed_bytes, breakdown.scale_bytes) == (rows * 8, scale)


# ---------------------------------------------------------------------------
# The f16 step table of emit
# ---------------------------------------------------------------------------


def packed_codes(q: torch.Tensor) -> torch.Tensor:
    """`q` (int8, `[rows, width]`) packed into the `[rows, width / 2]` uint8 codes."""
    return pack_int4(q).reshape(q.shape[0], q.shape[1] // 2)


def table_bytes(table: torch.Tensor) -> bytes:
    return memoryview(table.numpy()).cast("B").tobytes()


#: One code per row of `[3, 16]`, all different and none zero (1 group of 16 per row).
NONZERO = torch.tensor([[1] * 16, [-3] * 16, [7] * 16], dtype=torch.int8)


class TestTheF16StepTable:
    def test_each_step_is_rounded_to_f16_once_and_an_odd_count_is_padded(self) -> None:
        # f16(0.1) = 0x2e66, f16(1/3) = 0x3555 (round to nearest even), 1 = 0x3c00; one zero step
        # pads 3 steps to two u32 words.
        steps = torch.tensor([[0.1], [1 / 3], [1.0]], dtype=torch.float32)

        table = f16_step_table(packed_codes(NONZERO), steps, "w")

        assert table.dtype == torch.float16
        assert table_bytes(table) == f16_bytes(0x2E66, 0x3555, 0x3C00, 0x0000)

    def test_an_even_count_takes_no_padding(self) -> None:
        q = torch.ones((2, 32), dtype=torch.int8)
        steps = torch.tensor([[1.0, 2.0], [0.5, 0.25]], dtype=torch.float32)

        table = f16_step_table(packed_codes(q), steps, "w")

        assert table_bytes(table) == f16_bytes(0x3C00, 0x4000, 0x3800, 0x3400)

    def test_an_all_zero_group_takes_at_least_the_smallest_f16_normal(self) -> None:
        # Every code is 0: a zero step and the f32 tiny floor (which rounds to a zero f16) both
        # become 2^-14 (0x0400); a step above the floor is kept.
        q = torch.zeros((3, 16), dtype=torch.int8)
        steps = torch.tensor([[0.0], [torch.finfo(torch.float32).tiny], [1.0]], dtype=torch.float32)

        table = f16_step_table(packed_codes(q), steps, "w")

        assert F16_MIN_NORMAL == 2.0**-14
        assert table_bytes(table) == f16_bytes(0x0400, 0x0400, 0x3C00, 0x0000)

    def test_a_subnormal_step_of_a_nonzero_group_is_kept(self) -> None:
        # 2^-20 is the f16 subnormal 0x0010 — finite and non-zero, so it is stored as is.
        steps = torch.tensor([[2.0**-20], [1.0], [1.0]], dtype=torch.float32)

        table = f16_step_table(packed_codes(NONZERO), steps, "w")

        assert table_bytes(table)[:2] == f16_bytes(0x0010)

    @pytest.mark.parametrize(
        "step",
        [1e-9, 1e5, float("inf"), float("nan")],
        ids=["rounds-to-zero", "rounds-to-inf", "inf", "nan"],
    )
    def test_a_step_that_rounds_to_a_zero_or_non_finite_f16_is_refused(self, step: float) -> None:
        steps = torch.tensor([[1.0], [step], [1.0]], dtype=torch.float32)

        with pytest.raises(EmitError, match="round to a zero or non-finite f16 — the first is"):
            f16_step_table(packed_codes(NONZERO), steps, "w")

    def test_a_non_finite_step_of_an_all_zero_group_is_refused_too(self) -> None:
        # The floor does not hide it: 0 × inf is NaN.
        q = torch.zeros((3, 16), dtype=torch.int8)
        steps = torch.tensor([[1.0], [float("inf")], [1.0]], dtype=torch.float32)

        with pytest.raises(EmitError, match="round to a zero or non-finite f16"):
            f16_step_table(packed_codes(q), steps, "w")

    def test_its_words_read_as_the_runtime_reads_them_hold_the_rounded_steps(self) -> None:
        """The runtime's read of the table: u32 words, step `2i` in the low half and `2i+1` in the
        high half, widened — the steps rounded once to f16, row-major (`struct`'s `e` rounds a
        value to f16 to nearest even, independently of torch)."""
        q = torch.tensor(
            [[(i % 15) - 7 for i in range(32)], [7 - (i % 15) for i in range(32)]],
            dtype=torch.int8,
        )
        steps = torch.tensor([[0.1, 0.3], [1.7, 1e-3]], dtype=torch.float32)

        raw = table_bytes(f16_step_table(packed_codes(q), steps, "w"))

        words = struct.unpack("<2I", raw)
        halves = [half for word in words for half in (word & 0xFFFF, word >> 16)]
        widened = [struct.unpack("<e", struct.pack("<H", half))[0] for half in halves]
        assert widened == [
            struct.unpack("<e", struct.pack("<e", float(value)))[0] for value in steps.flatten()
        ]

    def test_codes_and_steps_that_do_not_pair_are_refused(self) -> None:
        with pytest.raises(EmitError, match="do not divide the row length"):
            f16_step_table(packed_codes(NONZERO), torch.ones((3, 3), dtype=torch.float32), "w")
        with pytest.raises(EmitError, match="F32 steps"):
            f16_step_table(packed_codes(NONZERO), torch.ones((3, 1), dtype=torch.float16), "w")


# ---------------------------------------------------------------------------
# No consumer admits it yet
# ---------------------------------------------------------------------------


def fixed_graph(op: str) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """`w [3, 16]` consumed by `linear` or `embedding` (the two ops of fixed weights)."""
    nodes = {
        "linear": IrNode(op="linear", ins=["x", "w", "b"], outs=["y"], attrs={}),
        "embedding": IrNode(
            op="embedding", ins=["w", "ids"], outs=["y"], attrs={"padding_idx": -1}
        ),
    }
    inputs = {
        "linear": IrInput(name="x", dtype="f32", shape=[2, 16]),
        "embedding": IrInput(name="ids", dtype="i32", shape=[2]),
    }
    initializers = {"w": IrInitializer(tensor="layer.weight", storage=IrStorage(dtype="f32"))}
    values = {"w": IrValue(dtype="f32", shape=[3, 16])}
    tensors = {"layer.weight": torch.empty(3, 16, device="meta")}
    if op == "linear":
        initializers["b"] = IrInitializer(tensor="layer.bias", storage=IrStorage(dtype="f32"))
        values["b"] = IrValue(dtype="f32", shape=[3])
        values["y"] = IrValue(dtype="f32", shape=[2, 3])
        tensors["layer.bias"] = torch.zeros(3)
    else:
        values["y"] = IrValue(dtype="f32", shape=[2, 16])
    graph = IrGraph(
        symbols=[],
        inputs=[inputs[op]],
        outputs=["y"],
        initializers=initializers,
        values=values,
        nodes=[nodes[op]],
    )
    return graph, tensors


class TestNoConsumerAdmitsItYet:
    @pytest.mark.parametrize("op", ["linear", "embedding"])
    def test_a_fixed_weight_with_f16_steps_is_refused(self, op: str) -> None:
        graph, tensors = fixed_graph(op)
        fixed = FixedQuantizedWeight(
            "i4f16", packed_codes(NONZERO), torch.ones((3, 1), dtype=torch.float32)
        )

        with pytest.raises(EmitError, match=f"{REFUSAL} \\(consumed by {op}\\)"):
            stored_model(graph, tensors, fixed_weights={"layer.weight": fixed})

    def test_a_rank_3_conv1d_weight_with_f16_steps_is_refused(self) -> None:
        graph = IrGraph(
            symbols=[],
            inputs=[IrInput(name="x", dtype="f32", shape=[1, 8, 6])],
            outputs=["y"],
            initializers={"w": IrInitializer(tensor="conv.weight", storage=IrStorage(dtype="f32"))},
            values={
                "w": IrValue(dtype="f32", shape=[4, 8, 2]),
                "y": IrValue(dtype="f32", shape=[1, 4, 5]),
            },
            nodes=[
                IrNode(
                    op="conv1d",
                    ins=["x", "w"],
                    outs=["y"],
                    attrs={"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
                )
            ],
        )
        fixed = FixedQuantizedWeight(
            "i4f16",
            torch.zeros((4, 8), dtype=torch.uint8),
            torch.ones((4, 1), dtype=torch.float32),
        )

        with pytest.raises(EmitError):
            stored_model(
                graph,
                {"conv.weight": torch.empty(4, 8, 2, device="meta")},
                fixed_weights={"conv.weight": fixed},
            )

    @pytest.mark.parametrize("option", ["weight_dtype", "weight_dtype_overrides"])
    def test_the_automatic_path_cannot_request_it(self, option: str) -> None:
        graph, _ = fixed_graph("linear")
        tensors = {"layer.weight": torch.zeros(3, 16), "layer.bias": torch.zeros(3)}
        kwargs = (
            {"weight_dtype": "i4f16"}
            if option == "weight_dtype"
            else {"weight_dtype_overrides": {"layer.weight": "i4f16"}}
        )

        with pytest.raises(EmitError, match="dtype 'i4f16' cannot be written"):
            stored_model(graph, tensors, **kwargs)
