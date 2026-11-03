"""The container merge layer (`karume.verify.bind_graphs` / `verify_container`) — container-v1
§5 / §6.

It mirrors TS `packages/runtime/src/format/container/bind.ts`, so this file pins only **the rules
that need declared shapes** (the structure of the two documents and the block layout belong to the
reader = `test_container.py`). Faults are injected through hand-built descriptors — rather than
breaking a correct container, this shows one to one "which field broke the rule".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from container_fixture import (
    FIXTURE_BLOCK_BYTES,
    FIXTURE_PATH,
    GRAPH_NAME,
    split_fixture_paths,
    synthetic_bindings,
    synthetic_graph,
    synthetic_tensors,
)
from ir_fixtures import ir_container, with_an_unknown_op

from karume import verify
from karume.container import (
    BlockEncoding,
    ContainerFormatError,
    DataBlockRecord,
    GraphDescriptor,
    ModelDescriptor,
    PartRecord,
    Provenance,
    WeightSupply,
    numbered_name,
    write_graph_container,
    write_model_container,
)
from karume.emit import FixedQuantizedWeight, stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.verify import InitializerSupply, bind_graphs, verify_container

#: sha256 field of the hand-built descriptors (the merge layer reads no real bytes, so the value
#: does not matter).
ZERO_SHA = "0" * 64

GRAPH = "g"
WEIGHT = "w"


def declaration(values: dict[str, tuple[str, list[int]]], shared: Sequence[str] = ()) -> dict:
    """IR v2 graph declaration (the merge layer reads only `initializers` and `values`)."""
    return {
        "format": "karume-ir",
        "version": 2,
        "requires": {"ops": []},
        "symbols": [],
        "inputs": [],
        "outputs": [],
        "initializers": {name: ({"shared": True} if name in shared else {}) for name in values},
        "values": {
            name: {"dtype": dtype, "shape": shape} for name, (dtype, shape) in values.items()
        },
        "nodes": [],
    }


def bind(
    shape: list[Any],
    supply: WeightSupply,
    blocks: Sequence[tuple[str, int, str]],
    *,
    dtype: str = "f32",
) -> InitializerSupply:
    """Build descriptors for one graph and one initializer and merge them (`blocks` are id /
    length / role)."""
    graph = GraphDescriptor(
        graphs={GRAPH: declaration({WEIGHT: (dtype, shape)})},
        const_length=0,
        const_blocks=(),
        constants=(),
    )
    model = ModelDescriptor(
        parts=(PartRecord(1, 0, ZERO_SHA), PartRecord(2, 1 << 20, ZERO_SHA)),
        blocks=tuple(
            DataBlockRecord(name, 2, 0, length, ZERO_SHA, role)  # type: ignore[arg-type]
            for name, length, role in blocks
        ),
        binding={GRAPH: {WEIGHT: supply}},
        assets={},
        provenance=Provenance(license="mit"),
    )
    return bind_graphs(graph, model)[GRAPH].supplies[WEIGHT]


def quantized(codec: str, group_size: int, row_axis: int = 0) -> BlockEncoding:
    return BlockEncoding(codec, row_axis, group_size, "s1")


class TestTheFixtureContainer:
    """The cross-language fixture (the real file Python wrote) gets through the merge."""

    def test_every_initializer_gets_a_supply(self) -> None:
        bound = verify_container([FIXTURE_PATH]).graphs[GRAPH_NAME]

        assert {name: supply.encoding.codec for name, supply in bound.supplies.items()} == {
            "big.weight": "int8-sym",
            "conv.weight": "int8-sym",
            "dec.weight": "f16",
            "enc.weight": "int4-sym-g",
            "const.a1b2c3d4e5f60718": "f32",
            "const.b1b2c3d4e5f6071a": "i32",
        }
        # A shared declaration has no supply plan (the lender's resident weights are the real data).
        assert "lm_head.weight" not in bound.supplies

    def test_a_const_initializer_is_supplied_from_the_graph(self) -> None:
        bound = verify_container([FIXTURE_PATH]).graphs[GRAPH_NAME]
        supply = bound.supplies["const.a1b2c3d4e5f60718"]

        assert supply.origin == "const"
        assert [block.part for block in supply.blocks] == [1]

    def test_a_piece_series_covers_the_rows_and_carries_its_payload_length(self) -> None:
        supply = verify_container([FIXTURE_PATH]).graphs[GRAPH_NAME].supplies["big.weight"]

        assert [block.rows for block in supply.blocks] == [(0, 8), (8, 16)]
        assert [block.payload_bytes for block in supply.blocks] == [256, 256]

    def test_a_tail_pad_is_outside_the_payload(self) -> None:
        """42 bytes of f16 go into a 44-byte block (the 2-byte difference is padding)."""
        supply = verify_container([FIXTURE_PATH]).graphs[GRAPH_NAME].supplies["dec.weight"]

        assert [(block.length, block.payload_bytes) for block in supply.blocks] == [(44, 42)]

    def test_a_row_axis_1_scale_is_one_value_per_channel(self) -> None:
        supply = verify_container([FIXTURE_PATH]).graphs[GRAPH_NAME].supplies["conv.weight"]
        assert supply.scale is not None

        # conv_transpose1d shape [4,3,8]: the row count is shape[1] = 3.
        assert supply.encoding.row_axis == 1
        assert supply.scale.rows == (0, 3)
        assert supply.scale.payload_bytes == 12

    def test_a_graph_container_supplies_only_its_const_region(self, tmp_path: Path) -> None:
        """A `krg` has no binding table — initializers that need weights stay without a supply."""
        bindings = synthetic_bindings()
        path = write_graph_container(
            tmp_path / "synthetic.krg",
            synthetic_graph(),
            synthetic_tensors(),
            {name: bindings[name] for name in bindings if name.startswith("const.")},
            graph_name=GRAPH_NAME,
            block_bytes=FIXTURE_BLOCK_BYTES,
        )
        bound = verify_container([path]).graphs[GRAPH_NAME]

        assert sorted(bound.supplies) == [
            "const.a1b2c3d4e5f60718",
            "const.b1b2c3d4e5f6071a",
        ]
        assert all(supply.origin == "const" for supply in bound.supplies.values())


class TestTheDeclarationRules:
    """Rules decided by declared shape × encoding × block index (the accepted set of bind.ts)."""

    def test_a_whole_block_carries_the_declared_payload(self) -> None:
        supply = bind(
            [4, 32], WeightSupply(BlockEncoding("f32"), block="w1"), [("w1", 512, "weight")]
        )

        assert supply.origin == "model"
        assert [(block.rows, block.payload_bytes) for block in supply.blocks] == [((0, 4), 512)]

    def test_a_codec_outside_the_semantic_dtype_is_refused(self) -> None:
        """An `i32` declaration is raw int32 only — no form that reads it as f16 bits."""
        with pytest.raises(ContainerFormatError, match="cannot pair with codec 'f16'"):
            bind(
                [4, 32],
                WeightSupply(BlockEncoding("f16"), block="w1"),
                [("w1", 256, "weight")],
                dtype="i32",
            )

    def test_a_block_longer_than_the_payload_plus_its_pad_is_refused(self) -> None:
        with pytest.raises(
            ContainerFormatError, match=r"is not payload \d+ bytes \+ padding \(less than 4\)"
        ):
            bind([4, 32], WeightSupply(BlockEncoding("f32"), block="w1"), [("w1", 516, "weight")])

    def test_a_symbolic_shape_cannot_hold_an_entity(self) -> None:
        with pytest.raises(ContainerFormatError, match="cannot use a symbolic dimension"):
            bind([4, "T"], WeightSupply(BlockEncoding("f32"), block="w1"), [("w1", 512, "weight")])

    def test_an_integral_float_dimension_is_read_as_the_integer(self) -> None:
        """Another writer's `4.0` reads as dimension 4, as in the runtime (a JSON number).

        The same accepted set as the graph's `_parse_shape` — if only the exporter's checks
        refused it, the exporter alone could not read a container the runtime can.
        """
        supply = bind(
            [4.0, 32], WeightSupply(BlockEncoding("f32"), block="w1"), [("w1", 512, "weight")]
        )

        assert [(block.rows, block.payload_bytes) for block in supply.blocks] == [((0, 4), 512)]

    def test_a_fractional_dimension_is_still_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match=r"initializer's shape.*\(4\.5\)"):
            bind([4.5, 32], WeightSupply(BlockEncoding("f32"), block="w1"), [("w1", 512, "weight")])

    def test_an_initializer_without_a_value_declaration_is_refused(self) -> None:
        graph = GraphDescriptor(
            graphs={GRAPH: declaration({WEIGHT: ("f32", [4])}) | {"values": {}}},
            const_length=0,
            const_blocks=(),
            constants=(),
        )
        model = ModelDescriptor(
            parts=(PartRecord(1, 0, ZERO_SHA),),
            blocks=(DataBlockRecord("w1", 2, 0, 16, ZERO_SHA, "weight"),),
            binding={GRAPH: {WEIGHT: WeightSupply(BlockEncoding("f32"), block="w1")}},
            assets={},
            provenance=Provenance(license="mit"),
        )

        with pytest.raises(ContainerFormatError, match="no dtype / shape declaration"):
            bind_graphs(graph, model)

    def test_an_undeclared_block_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="undeclared block 'w9'"):
            bind([4, 32], WeightSupply(BlockEncoding("f32"), block="w9"), [("w1", 512, "weight")])


class TestThePieceRules:
    def test_the_last_piece_must_reach_the_declared_rows(self) -> None:
        supply = WeightSupply(BlockEncoding("f32"), pieces=(("w1", (0, 2)), ("w2", (2, 3))))

        with pytest.raises(
            ContainerFormatError,
            match="differs from the declared shape's leading dimension of 4 rows",
        ):
            bind([4, 32], supply, [("w1", 256, "weight"), ("w2", 128, "weight")])

    def test_an_intermediate_piece_cannot_be_padded(self) -> None:
        """Padding an intermediate piece would overwrite the start of the next (it MUST be exactly
        the bytes of its row range)."""
        supply = WeightSupply(BlockEncoding("f16"), pieces=(("w1", (0, 1)), ("w2", (1, 3))))

        with pytest.raises(ContainerFormatError, match="no padding allowed"):
            # 1 row = 7 elements = 14 bytes. Inflate the intermediate piece to 16 bytes.
            bind([3, 7], supply, [("w1", 16, "weight"), ("w2", 28, "weight")])

    def test_a_row_axis_1_initializer_cannot_be_split(self) -> None:
        supply = WeightSupply(
            quantized("int8-sym", 24, row_axis=1), pieces=(("w1", (0, 2)), ("w2", (2, 4)))
        )

        with pytest.raises(ContainerFormatError, match="cannot be split into pieces"):
            bind(
                [4, 3, 8],
                supply,
                [("w1", 48, "weight"), ("w2", 48, "weight"), ("s1", 12, "scale")],
            )


class TestTheQuantizedRules:
    def test_a_per_channel_group_size_is_the_row_length(self) -> None:
        with pytest.raises(ContainerFormatError, match="groupSize MUST equal the row length 32"):
            bind(
                [4, 32],
                WeightSupply(quantized("int8-sym", 16), block="w1"),
                [("w1", 128, "weight"), ("s1", 16, "scale")],
            )

    def test_a_group_size_must_be_a_power_of_two_at_or_above_the_floor(self) -> None:
        with pytest.raises(ContainerFormatError, match="is not a power of 2 that is 16 or more"):
            bind(
                [4, 32],
                WeightSupply(quantized("int4-sym-g", 8), block="w1"),
                [("w1", 64, "weight"), ("s1", 64, "scale")],
            )

    def test_a_group_size_must_divide_the_row_length(self) -> None:
        with pytest.raises(ContainerFormatError, match="is not divisible by"):
            bind(
                [4, 24],
                WeightSupply(quantized("int4-sym-g", 32), block="w1"),
                [("w1", 48, "weight"), ("s1", 16, "scale")],
            )

    def test_the_scale_block_holds_one_value_per_group(self) -> None:
        supply = bind(
            [4, 32],
            WeightSupply(quantized("int4-sym-g", 16), block="w1"),
            [("w1", 64, "weight"), ("s1", 32, "scale")],
        )
        assert supply.scale is not None

        # 4 rows × (32 / 16) groups × 4 bytes.
        assert supply.scale.payload_bytes == 32

    def test_a_scale_block_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="scale: block 's1'"):
            bind(
                [4, 32],
                WeightSupply(quantized("int4-sym-g", 16), block="w1"),
                [("w1", 64, "weight"), ("s1", 16, "scale")],
            )

    def test_an_i2_declaration_must_be_rank_2_with_a_16_multiple_row(self) -> None:
        with pytest.raises(
            ContainerFormatError, match="positive rank 2 with a row length that is a multiple of 16"
        ):
            bind(
                [4, 3, 8],
                WeightSupply(quantized("int2-off", 24), block="w1"),
                [("w1", 24, "weight"), ("s1", 16, "scale")],
            )

    def test_a_quantized_encoding_without_its_three_fields_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="rowAxis / groupSize / scale are required"):
            bind(
                [4, 32],
                WeightSupply(BlockEncoding("int8-sym"), block="w1"),
                [("w1", 128, "weight")],
            )


class TestTheContainerCli:
    """`karume verify` — structure check of the two documents + merge + sha256 of every block
    (it takes containers only)."""

    def test_it_reports_the_split_form_from_the_part_0_file(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        part0 = split_fixture_paths()[0]
        verify.main([str(part0)])
        printed = capsys.readouterr().out

        assert "parts=5" in printed
        assert "graphs=synthetic-split" in printed
        assert "initializers=3" in printed
        assert "blocks=5" in printed
        assert "assets=ple_index,rope_base" in printed

    def test_it_reports_the_single_form(self, capsys: pytest.CaptureFixture[str]) -> None:
        verify.main([str(FIXTURE_PATH)])
        printed = capsys.readouterr().out

        assert "parts=1" in printed
        assert "graphs=synthetic" in printed
        assert "assets=(none)" in printed

    def test_a_flipped_block_byte_is_caught(self, tmp_path: Path) -> None:
        """Checks down to the block sha256 (passing declarations alone is not "verified")."""
        copied = tmp_path / FIXTURE_PATH.name
        raw = bytearray(FIXTURE_PATH.read_bytes())
        raw[-1] ^= 0xFF
        copied.write_bytes(bytes(raw))

        with pytest.raises(
            ContainerFormatError, match=r"sha256 of block .* differs from the declaration"
        ):
            verify.main([str(copied)])

    def test_the_retired_flag_is_gone(self) -> None:
        """`--container` is retired (only containers are taken, so there is nothing to choose).

        Left in place it would read as "there is an entry point that also takes the old format" —
        only the migration CLI MUST read the old format (container-v1 §12).
        """
        with pytest.raises(SystemExit) as raised:
            verify.main(["--container", str(FIXTURE_PATH)])

        assert raised.value.code == 2


class TestTheIrAcceptanceGate:
    """Rebuild an `IrGraph` from the container's two documents and apply the IR acceptance rules
    ({@link karume.verify.ir_graph_from_container} / {@link karume.verify.assert_ir_accepted}).

    The structure checks (the classes above) only ask "does it open as a container". The op
    vocabulary, runtime support and op contracts are covered here — assembly (`karume dist`) and
    `karume verify` go through **the same single path**.
    """

    def _written(self, tmp_path: Path, parts: Sequence[bytes]) -> list[Path]:
        paths = [
            tmp_path / numbered_name("model.krm", index, len(parts))
            for index in range(1, len(parts) + 1)
        ]
        for path, payload in zip(paths, parts, strict=True):
            path.write_bytes(payload)
        return paths

    def test_the_storage_declaration_comes_from_the_binding_table(self, tmp_path: Path) -> None:
        """The binding table's codec comes back as the IR v1 `storage` (one source of truth for the
        declaration)."""
        paths = self._written(tmp_path, ir_container(mark="gate", storage="i4"))
        read = verify_container(paths).read

        graph = verify.ir_graph_from_container(read, "gate")
        storages = {
            name: init.storage.dtype
            for name, init in graph.initializers.items()
            if init.storage.dtype != "f32"
        }

        # The i4 series is mixed (i4-eligible weights are i4, the others i8) — both come back from
        # the container's binding table.
        assert sorted(set(storages.values())) == ["i4", "i8"]
        for name, init in graph.initializers.items():
            if init.storage.dtype in ("i4", "i8"):
                assert init.storage.scale is not None, name

    def test_a_missing_graph_name_fails_loudly(self, tmp_path: Path) -> None:
        paths = self._written(tmp_path, ir_container(mark="gate"))
        read = verify_container(paths).read

        with pytest.raises(verify.ContainerError, match="has no graph 'absent'"):
            verify.ir_graph_from_container(read, "absent")

    def test_a_container_declaring_an_unknown_op_is_rejected(self, tmp_path: Path) -> None:
        paths = self._written(tmp_path, with_an_unknown_op(ir_container(mark="gate")))
        read = verify_container(paths).read

        with pytest.raises(verify.ContainerError, match="unsupported ops"):
            verify.assert_ir_accepted(read)

    def test_the_same_container_without_the_unknown_op_is_accepted(self, tmp_path: Path) -> None:
        """Counterpart — the other side showing the check does not "always fail"."""
        paths = self._written(tmp_path, ir_container(mark="gate"))

        verify.assert_ir_accepted(verify_container(paths).read)

    def test_the_cli_rejects_a_container_declaring_an_unknown_op(self, tmp_path: Path) -> None:
        """`karume verify` goes through **the same check** as assembly (the accepting half)."""
        paths = self._written(tmp_path, with_an_unknown_op(ir_container(mark="gate")))

        with pytest.raises(verify.ContainerError, match="unsupported ops"):
            verify.main([str(paths[0])])

    def test_a_graph_container_is_accepted(self, tmp_path: Path) -> None:
        """A `krg` has no binding table — a slot without a supply is rebuilt as raw storage of its
        semantic dtype.

        The storage claim belongs to slots with no data in this container (a fact of the lender /
        the `krm`), so this checks only the op vocabulary, runtime support and contracts.
        """
        bindings = synthetic_bindings()
        path = write_graph_container(
            tmp_path / "synthetic.krg",
            synthetic_graph(),
            synthetic_tensors(),
            {name: bindings[name] for name in bindings if name.startswith("const.")},
            graph_name=GRAPH_NAME,
            block_bytes=FIXTURE_BLOCK_BYTES,
        )

        verify.assert_ir_accepted(verify_container([path]).read)


def ternary_container(tmp_path: Path, packed: bytes) -> list[Path]:
    """Write a container declaring `ternary` with 2 rows × 16 elements (4 bytes per row).

    There is no ternary quantizer yet (step 6), so a fixed i2 weight goes through the writer and
    only the binding's codec is swapped to `ternary` (the packing and the scale are those of
    `int2-off` — container-v1 §6.3).
    """
    graph = IrGraph(
        symbols=[],
        inputs=[IrInput(name="ids", dtype="i32", shape=[2])],
        outputs=["y"],
        initializers={"w": IrInitializer(tensor="w", storage=IrStorage(dtype="f32"))},
        values={"w": IrValue(dtype="f32", shape=[2, 16]), "y": IrValue(dtype="f32", shape=[2, 16])},
        nodes=[IrNode(op="embedding", ins=["w", "ids"], outs=["y"], attrs={"padding_idx": -1})],
    )
    fixed = FixedQuantizedWeight(
        dtype="i2",
        packed=torch.frombuffer(bytearray(packed), dtype=torch.uint8).reshape(2, 4),
        scale=torch.tensor([[0.375], [1.25]]),
    )
    stored = stored_model(
        graph,
        {"w": torch.empty(2, 16, dtype=torch.float32, device="meta")},
        fixed_weights={"w": fixed},
    )
    bindings = {
        key: replace(encoding, codec="ternary") if encoding.codec == "int2-off" else encoding
        for key, encoding in stored.bindings.items()
    }
    return write_model_container(
        tmp_path / "ternary.krm",
        stored.graph,
        stored.tensors,
        bindings,
        graph_name="t",
        provenance=Provenance(license="mit"),
    )


#: Two rows in which every 2-bit code is in {1, 2, 3} (0x6D = 01 10 11 01 and so on).
TERNARY_PACKED = bytes([0x6D, 0xB6, 0xDB, 0x79] * 2)


class TestTheTernaryPayload:
    """verify rejects a `ternary` payload containing code 0 (q = −2) (container-v1 §6.3).

    TS applies the same check at load (`assertTernaryCodes`), so without this one "verify is
    green and it first fails at load".
    """

    def test_a_payload_whose_codes_are_all_in_1_to_3_is_accepted(self, tmp_path: Path) -> None:
        verified = verify_container(ternary_container(tmp_path, TERNARY_PACKED), blocks=True)

        assert verified.graphs["t"].supplies["w"].encoding.codec == "ternary"

    def test_a_payload_with_a_code_0_is_refused(self, tmp_path: Path) -> None:
        packed = bytearray(TERNARY_PACKED)
        packed[5] = 0b00_10_10_10
        paths = ternary_container(tmp_path, bytes(packed))

        with pytest.raises(
            ContainerFormatError,
            match=r"initializer 'w': the ternary payload has code 0.*\(byte 5\)",
        ):
            verify_container(paths, blocks=True)


class TestTheCodecRowAxes:
    """The `rowAxis` each codec allows (`rowAxes` of the TS ledger — container-v1 §6.3, ADR 0108
    addendum 1, item 4).

    i4 / i2 / ternary quantize only along rows (`rowAxis: 0`). Checking only the general "0 or 1"
    would let `rowAxis: 1` of a square shape pass the Python checks and fail only in the browser's
    `bindGraphs`.
    """

    @pytest.mark.parametrize(
        ("codec", "group_size", "payload"),
        [("int4-sym-g", 16, 128), ("int2-off", 16, 64), ("ternary", 16, 64)],
    )
    def test_row_axis_1_is_refused_for_a_codec_without_it(
        self, codec: str, group_size: int, payload: int
    ) -> None:
        supply = WeightSupply(quantized(codec, group_size, row_axis=1), block="w1")
        blocks = [("w1", payload, "weight"), ("s1", 64, "scale")]
        # Counterpart: rowAxis 0 with the same declaration gets through the merge (the failure below
        # is the axis rule's).
        bind([16, 16], replace(supply, encoding=quantized(codec, group_size)), blocks)

        with pytest.raises(ContainerFormatError, match=f"rowAxis of codec '{codec}' must be 0"):
            bind([16, 16], supply, blocks)


def split_scale(pieces: Sequence[tuple[str, tuple[int, int]]]) -> BlockEncoding:
    """An i4 group-16 encoding whose scale is split into `pieces` (ADR 0108 addendum 15)."""
    return BlockEncoding("int4-sym-g", 0, 16, scale_pieces=tuple(pieces))


#: `[8, 32]` i4 body in two pieces of 4 rows (64 B each); its scale is 8 rows x 2 groups x 4 B =
#: 64 B, so 32 B per scale piece over the same rows.
BODY_PIECES = (("w1", (0, 4)), ("w2", (4, 8)))
BODY_BLOCKS = [("w1", 64, "weight"), ("w2", 64, "weight")]


class TestASplitScale:
    """The merge layer's rules of a split scale (container-v1 §5 rule ③)."""

    def test_one_scale_piece_per_body_piece_is_one_contiguous_scale(self) -> None:
        supply = bind(
            [8, 32],
            WeightSupply(split_scale([("s1", (0, 4)), ("s2", (4, 8))]), pieces=BODY_PIECES),
            [*BODY_BLOCKS, ("s1", 32, "scale"), ("s2", 32, "scale")],
        )
        assert supply.scale is not None

        assert (supply.scale.id, supply.scale.length, supply.scale.payload_bytes) == (
            "s1+s2",
            64,
            64,
        )
        assert [(piece.rows, piece.payload_bytes) for piece in supply.scale.pieces] == [
            ((0, 4), 32),
            ((4, 8), 32),
        ]

    def test_a_scale_piece_count_that_differs_from_the_bodys_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="3 scale pieces for 2 body pieces"):
            bind(
                [8, 32],
                WeightSupply(
                    split_scale([("s1", (0, 2)), ("s2", (2, 4)), ("s3", (4, 8))]),
                    pieces=BODY_PIECES,
                ),
                [*BODY_BLOCKS, ("s1", 16, "scale"), ("s2", 16, "scale"), ("s3", 32, "scale")],
            )

    def test_scale_rows_that_are_not_the_body_pieces_rows_are_refused(self) -> None:
        with pytest.raises(
            ContainerFormatError,
            match=r"rows \[0, 2\) differ from the rows \[0, 4\) of body piece 0",
        ):
            bind(
                [8, 32],
                WeightSupply(split_scale([("s1", (0, 2)), ("s2", (2, 8))]), pieces=BODY_PIECES),
                [*BODY_BLOCKS, ("s1", 16, "scale"), ("s2", 48, "scale")],
            )

    def test_a_middle_scale_piece_with_padding_is_refused(self) -> None:
        with pytest.raises(
            ContainerFormatError, match="block 's1' length 36 differs from the 32 bytes of its rows"
        ):
            bind(
                [8, 32],
                WeightSupply(split_scale([("s1", (0, 4)), ("s2", (4, 8))]), pieces=BODY_PIECES),
                [*BODY_BLOCKS, ("s1", 36, "scale"), ("s2", 32, "scale")],
            )

    def test_a_last_scale_piece_longer_than_its_rows_and_padding_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="block 's2' length 36 is not payload 32"):
            bind(
                [8, 32],
                WeightSupply(split_scale([("s1", (0, 4)), ("s2", (4, 8))]), pieces=BODY_PIECES),
                [*BODY_BLOCKS, ("s1", 32, "scale"), ("s2", 36, "scale")],
            )

    def test_a_split_scale_beside_a_whole_body_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="a split scale needs a piece list"):
            bind(
                [8, 32],
                WeightSupply(split_scale([("s1", (0, 4)), ("s2", (4, 8))]), block="w"),
                [("w", 128, "weight"), ("s1", 32, "scale"), ("s2", 32, "scale")],
            )
