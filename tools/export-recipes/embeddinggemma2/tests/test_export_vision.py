"""Script-level contracts of `embeddinggemma2/export_vision.py` (the part that needs no real
weights).

The real-weight export is manual. What is pinned here is what turns into a **false PASS** when
broken:

- the host's order of the patch rows (the io contract): the 3 × 3 blocks in raster order, each
  block's 9 patches in raster order; the padded processor rows are stripped and reordered, and
  rows of another form are refused
- on a tiny random tower (transformers at the pinned release): the wrapper equals upstream's own
  forward on the processor's padded rows and on the graph's rows, its gathered RoPE equals
  upstream's rotary module bit for bit, and the rows in the processor's raster order or with the
  position axes swapped move the soft tokens far (the two breakages the order and the gather guard
  against)
- the traced form: three inputs on `9n`, one output `[1, n, 512]`, the attention without a mask
  and with upstream's scale 1, the six lookups, no `where` — and each broken form is refused
- the storage: bf16 in every weight slot but the RoPE tables, which stay f32
- the pins: the snapshot files the script reads are pinned, the constants are the config's

The cases that need transformers at the pinned release SKIP explicitly without it.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embeddinggemma2 import export, export_vision, sources
from karume.container import Provenance
from karume.ir import IrGraph, IrNode
from karume.pipeline import publish_model


def _require_transformers() -> None:
    pytest.importorskip("transformers")
    version = metadata.version("transformers")
    if version != export.REQUIRED_TRANSFORMERS:
        pytest.skip(f"transformers {version} is installed, not {export.REQUIRED_TRANSFORMERS}")


class TestBlockOrder:
    def test_a_six_by_three_grid_is_two_blocks_of_nine_in_raster_order(self) -> None:
        assert export_vision.block_order(6, 3) == [
            *(0, 1, 2, 6, 7, 8, 12, 13, 14),
            *(3, 4, 5, 9, 10, 11, 15, 16, 17),
        ]

    def test_blocks_run_row_of_blocks_by_row_of_blocks(self) -> None:
        order = export_vision.block_order(3, 6)

        assert order[:9] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert order[9:] == [9, 10, 11, 12, 13, 14, 15, 16, 17]

    def test_every_patch_appears_once(self) -> None:
        order = export_vision.block_order(12, 9)

        assert sorted(order) == list(range(12 * 9))

    @pytest.mark.parametrize(("width", "height"), [(4, 3), (3, 5), (0, 3)])
    def test_a_grid_not_cut_into_three_by_three_blocks_is_refused(
        self, width: int, height: int
    ) -> None:
        with pytest.raises(export_vision.VisionExportError, match="3 × 3 blocks"):
            export_vision.block_order(width, height)


def _padded(width: int, height: int, padding: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Processor-like rows of a `width × height` patch grid: raster patches in [0, 1], then
    `padding` zero rows at (−1, −1)."""
    generator = torch.Generator().manual_seed(seed)
    real = width * height
    pixels = torch.zeros(1, real + padding, 768)
    pixels[0, :real] = torch.rand(real, 768, generator=generator)
    grid = torch.stack(
        torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy"), dim=-1
    ).reshape(-1, 2)
    positions = torch.full((1, real + padding, 2), -1, dtype=torch.int64)
    positions[0, :real] = grid
    return pixels, positions


class TestHostRows:
    def test_the_real_rows_are_reordered_into_blocks_with_their_positions(self) -> None:
        pixels, positions = _padded(6, 3, padding=9)

        rows = export_vision.host_rows(pixels, positions)

        order = export_vision.block_order(6, 3)
        assert rows.soft_tokens == 2
        assert torch.equal(rows.pixels[0], pixels[0, order])
        assert rows.position_x[0].tolist() == [index % 6 for index in order]
        assert rows.position_y[0].tolist() == [index // 6 for index in order]
        # Each soft token's 9 patches are one 3 × 3 block.
        for token in range(2):
            xs = rows.position_x[0, 9 * token : 9 * token + 9]
            ys = rows.position_y[0, 9 * token : 9 * token + 9]
            assert len({(int(x) // 3, int(y) // 3) for x, y in zip(xs, ys, strict=True)}) == 1

    def test_padding_before_a_real_row_is_refused(self) -> None:
        pixels, positions = _padded(6, 3, padding=2)
        positions[0, [0, -1]] = positions[0, [-1, 0]]

        with pytest.raises(export_vision.VisionExportError, match="not the first rows"):
            export_vision.host_rows(pixels, positions)

    def test_a_padding_row_holding_values_is_refused(self) -> None:
        pixels, positions = _padded(6, 3, padding=2)
        pixels[0, -1, 0] = 0.5

        with pytest.raises(export_vision.VisionExportError, match="not zeros"):
            export_vision.host_rows(pixels, positions)

    def test_real_rows_out_of_raster_order_are_refused(self) -> None:
        pixels, positions = _padded(6, 3, padding=0)
        positions[0, [0, 1]] = positions[0, [1, 0]]

        with pytest.raises(export_vision.VisionExportError, match="not the raster"):
            export_vision.host_rows(pixels, positions)


class TestConstants:
    def test_the_rope_covers_the_longest_side_at_the_budget(self) -> None:
        """At 280 soft tokens the longest side is (280 · 9 // 9) · 48 px = 13,440 px = 840 patches
        (upstream's `max_side_length`)."""
        assert export_vision.ROPE_POSITIONS == (280 * 9 // 9) * 48 // 16 == 840

    def test_the_files_read_are_pinned(self) -> None:
        assert set(export_vision.READ_FILES) <= set(sources.PIN.sha256)

    def test_the_series_are_one_per_storage(self) -> None:
        assert export_vision.DEFAULT_OUT_DIRS["f32"].name == "embeddinggemma-2-vision"
        assert export_vision.DEFAULT_OUT_DIRS["bf16"].name == "embeddinggemma-2-vision-bf16"
        assert sorted(export_vision.DEFAULT_OUT_DIRS) == sorted(export_vision.WEIGHT_DTYPES)

    def test_the_soft_width_is_the_text_graphs(self) -> None:
        from embeddinggemma2 import distribution

        names = [name for name, *_ in distribution.EMBEDDINGGEMMA2_GRAPH_INPUTS]
        assert names[2] == export.SOFT_EMBEDS
        assert export_vision.SOFT_WIDTH == 512

    def test_another_storage_is_refused_before_anything_is_read(self, tmp_path: Path) -> None:
        with pytest.raises(export_vision.VisionExportError, match="not one of"):
            export_vision.export_series(tmp_path / "missing", tmp_path / "out", dtype="i8")


#: The tiny tower: 2 layers, hidden 16, 2 heads of 8, a 64-row position table.
TINY_POSITIONS = 64


@pytest.fixture
def tiny_tower() -> export_vision.Tower:
    """A randomly initialised tiny `Gemma4VisionModel` and projector on the hook (f32)."""
    _require_transformers()
    from transformers import AutoModel
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2MultimodalEmbedder,
    )
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    torch.manual_seed(0)
    export.register_attention()
    vision = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        position_embedding_size=TINY_POSITIONS,
    )
    vision._attn_implementation = export.ATTENTION_NAME
    tower = AutoModel.from_config(vision)
    embedder = EmbeddingGemma2MultimodalEmbedder(vision, SimpleNamespace(hidden_size=512))
    with torch.no_grad():
        for module in (tower, embedder):
            for parameter in module.parameters():
                parameter.normal_(0.0, 0.5)
    return export_vision.Tower(
        tower=tower.eval().requires_grad_(False),
        embedder=embedder.eval().requires_grad_(False),
        config=SimpleNamespace(vision_config=vision),
    )


def _wrapper(tower: export_vision.Tower, positions: int = TINY_POSITIONS):
    return export_vision.VisionTowerWrapper(tower, positions).eval()


class TestTinyTower:
    """The wrapper on a tiny random tower (transformers at the pinned release)."""

    def test_it_equals_upstreams_forward_on_the_padded_rows_and_on_its_own(
        self, tiny_tower
    ) -> None:
        pixels, positions = _padded(6, 9, padding=27)
        rows = export_vision.host_rows(pixels, positions)

        golden = export_vision.upstream_soft_tokens(tiny_tower, pixels, positions)
        same_rows = export_vision.upstream_soft_tokens(tiny_tower, rows.pixels, rows.position_ids())
        with torch.no_grad():
            ours = _wrapper(tiny_tower)(rows.pixels, rows.position_x, rows.position_y)

        assert list(ours.shape) == [1, 6, 512]
        torch.testing.assert_close(ours[0], golden, atol=1e-5, rtol=0)
        torch.testing.assert_close(ours[0], same_rows, atol=1e-5, rtol=0)

    def test_the_gathered_rope_is_upstreams_bit_for_bit(self, tiny_tower) -> None:
        rows = export_vision.host_rows(*_padded(9, 6, padding=0))

        export_vision.assert_rope_tables(_wrapper(tiny_tower), tiny_tower, rows)

    def test_rope_tables_of_shifted_positions_are_refused(self, tiny_tower) -> None:
        rows = export_vision.host_rows(*_padded(9, 6, padding=0))
        wrapper = _wrapper(tiny_tower)
        wrapper.rope_cos = torch.roll(wrapper.rope_cos, 1, dims=0)

        with pytest.raises(AssertionError, match="RoPE cos differs"):
            export_vision.assert_rope_tables(wrapper, tiny_tower, rows)

    def test_raster_order_or_swapped_axes_move_the_soft_tokens_far(self, tiny_tower) -> None:
        """What the block order and the per-axis gather guard against: both keep every shape."""
        pixels, positions = _padded(6, 9, padding=0)
        rows = export_vision.host_rows(pixels, positions)
        wrapper = _wrapper(tiny_tower)

        with torch.no_grad():
            ours = wrapper(rows.pixels, rows.position_x, rows.position_y)
            raster = wrapper(pixels, positions[..., 0], positions[..., 1])
            swapped = wrapper(rows.pixels, rows.position_y, rows.position_x)

        assert float((raster - ours).abs().max()) > 1e-2
        assert float((swapped - ours).abs().max()) > 1e-2

    def test_the_rope_halves_are_one_table(self, tiny_tower) -> None:
        cos, sin = export_vision.rope_tables(tiny_tower.tower.encoder.rotary_emb, 5)

        assert list(cos.shape) == list(sin.shape) == [5, 4]
        assert torch.equal(cos[0], torch.ones(4)) and torch.equal(sin[0], torch.zeros(4))


@pytest.fixture
def tiny_graph(tiny_tower) -> tuple[IrGraph, dict[str, torch.Tensor], export_vision.Tower]:
    graph, tensors = export_vision.trace(_wrapper(tiny_tower, export_vision.ROPE_POSITIONS))
    return graph, tensors, tiny_tower


def _node(graph: IrGraph, op: str) -> IrNode:
    return next(node for node in graph.nodes if node.op == op)


class TestIrForm:
    def test_the_traced_form_passes(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph

        form = export_vision.assert_ir_form(graph, tower.config)

        assert form["inputs"] == [
            ["pixels", "f32", [1, "9n", 768]],
            ["position_x", "i32", [1, "9n"]],
            ["position_y", "i32", [1, "9n"]],
        ]
        assert form["attention_nodes"] == 2
        assert form["ops"]["embedding"] == 6

    def test_a_mask_on_the_attention_is_refused(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph
        _node(graph, "attention").ins.append(export_vision.PIXELS)

        with pytest.raises(AssertionError, match="no mask"):
            export_vision.assert_ir_form(graph, tower.config)

    def test_another_attention_scale_is_refused(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph
        _node(graph, "attention").attrs["scale"] = 8**-0.5

        with pytest.raises(AssertionError, match="scale 1"):
            export_vision.assert_ir_form(graph, tower.config)

    def test_a_table_read_by_the_other_axis_is_refused(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph
        lookup = next(
            node
            for node in graph.nodes
            if node.op == "embedding" and graph.initializers[node.ins[0]].tensor == "pos_x"
        )
        lookup.ins[1] = export_vision.POSITION_Y

        with pytest.raises(AssertionError, match="embedding lookups"):
            export_vision.assert_ir_form(graph, tower.config)

    def test_a_where_is_refused(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph
        graph.nodes.append(
            IrNode(op="where", ins=["position_x", "pixels", "pixels"], outs=["x"], attrs={})
        )

        with pytest.raises(AssertionError, match="where"):
            export_vision.assert_ir_form(graph, tower.config)


def _round_weight_slots_to_bf16(wrapper: torch.nn.Module) -> None:
    """Round what the bf16 storage stores as bf16 — every linear weight and the two learned
    position tables — as the checkpoint's bf16 values are (the RoPE tables are left as they are)."""
    with torch.no_grad():
        targets = [
            module.weight for module in wrapper.modules() if isinstance(module, torch.nn.Linear)
        ]
        for tensor in (*targets, wrapper.pos_x, wrapper.pos_y):
            tensor.copy_(tensor.to(torch.bfloat16).to(torch.float32))


class TestStorage:
    def test_bf16_in_every_weight_slot_but_the_rope_tables(self, tiny_tower, tmp_path) -> None:
        wrapper = _wrapper(tiny_tower, export_vision.ROPE_POSITIONS)
        _round_weight_slots_to_bf16(wrapper)
        graph, tensors = export_vision.trace(wrapper)

        stored = publish_model(
            tmp_path / "model.krm",
            graph,
            tensors,
            provenance=Provenance(license=export.LICENSE),
            graph_name=export_vision.GRAPH_NAME,
            weight_dtype="bf16",
            weight_dtype_overrides={name: "f32" for name in export_vision.ROPE_TABLES},
        )
        counts = export_vision.assert_storage_form(stored, "bf16")

        # 2 layers × 7 linears, the input projection, the projector, the two position tables.
        assert counts["bf16"] == 2 * 7 + 2 + 2

    def test_the_rope_tables_in_bf16_fail_the_write(self, tiny_tower, tmp_path) -> None:
        """They are no bf16 values: the writer refuses them unless they are kept f32."""
        from karume.emit import EmitError

        wrapper = _wrapper(tiny_tower, export_vision.ROPE_POSITIONS)
        _round_weight_slots_to_bf16(wrapper)
        graph, tensors = export_vision.trace(wrapper)

        with pytest.raises(EmitError, match="bf16 cannot represent"):
            publish_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                provenance=Provenance(license=export.LICENSE),
                graph_name=export_vision.GRAPH_NAME,
                weight_dtype="bf16",
            )

    def test_the_f32_storage_holds_f32_alone(self, tiny_tower, tmp_path) -> None:
        graph, tensors = export_vision.trace(_wrapper(tiny_tower, export_vision.ROPE_POSITIONS))
        stored = publish_model(
            tmp_path / "model.krm",
            graph,
            tensors,
            provenance=Provenance(license=export.LICENSE),
            graph_name=export_vision.GRAPH_NAME,
        )

        assert set(export_vision.assert_storage_form(stored, "f32")) == {"f32"}
        with pytest.raises(AssertionError, match=r"pos_x: f32 \(expected bf16\)"):
            export_vision.assert_storage_form(stored, "bf16")
