"""Script-level contracts of `embeddinggemma2/export_audio.py` (the part that needs no real
weights).

The real-weight export is manual. What is pinned here is what turns into a **false PASS** when
broken:

- the host's frame layout (the io contract): the valid frames zero-padded to `4n`, the mask 1 on
  them; frames of another form are refused
- the window: the one-hot selector copies each token's 12 keys exactly, the bias removes only the
  keys before the first token
- on a tiny random tower (transformers at the pinned release): the wrapper equals upstream's own
  forward on the extractor-like frames and on the graph's frames for every residue of the frame
  count mod 4; without the mask it is bit-identical except at `L ≡ 2 (mod 4)`, where it moves far;
  upstream's modules are left as they are
- the traced form: two inputs on `4n`, one output `[1, n, 512]`, no fused attention and no
  `where`, the softmaxes on the window, one selector shared by every window copy, one bias prefix —
  and each broken form is refused
- the storage: bf16 in every weight slot, f32 elsewhere
- the pins: the snapshot files the script reads are pinned, the bound is the front end's 30 s

The cases that need transformers at the pinned release SKIP explicitly without it.
"""

from __future__ import annotations

import copy
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from embeddinggemma2 import export, export_audio, sources
from karume.container import Provenance
from karume.ir import IrGraph, IrNode
from karume.pipeline import publish_model


def _require_transformers() -> None:
    pytest.importorskip("transformers")
    version = metadata.version("transformers")
    if version != export.REQUIRED_TRANSFORMERS:
        pytest.skip(f"transformers {version} is installed, not {export.REQUIRED_TRANSFORMERS}")


def _extracted(valid: int, padding: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Extractor-like frames of one audio: `valid` log-mel frames, then `padding` zero frames with
    the mask off (as upstream's extractor pads to a multiple of 128 samples)."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.zeros(1, valid + padding, export_audio.MEL_BINS)
    features[0, :valid] = torch.randn(valid, export_audio.MEL_BINS, generator=generator) - 4
    mask = torch.zeros(1, valid + padding, dtype=torch.bool)
    mask[0, :valid] = True
    return features, mask


class TestHostFrames:
    @pytest.mark.parametrize(("valid", "tokens"), [(1, 1), (4, 1), (5, 2), (99, 25), (102, 26)])
    def test_the_valid_frames_are_padded_to_four_per_token(self, valid: int, tokens: int) -> None:
        features, mask = _extracted(valid, padding=1)

        frames = export_audio.host_frames(features, mask)

        assert frames.soft_tokens == tokens
        assert list(frames.features.shape) == [1, 4 * tokens, export_audio.MEL_BINS]
        assert torch.equal(frames.features[0, :valid], features[0, :valid])
        assert not bool(frames.features[0, valid:].any())
        assert frames.frame_mask.dtype == torch.float32
        assert frames.frame_mask[0].tolist() == [1.0] * valid + [0.0] * (4 * tokens - valid)
        assert frames.valid == valid

    def test_valid_frames_after_a_masked_one_are_refused(self) -> None:
        features, mask = _extracted(8, padding=2)
        mask[0, 3] = False

        with pytest.raises(export_audio.AudioExportError, match="not the first frames"):
            export_audio.host_frames(features, mask)

    def test_a_masked_frame_holding_values_is_refused(self) -> None:
        features, mask = _extracted(8, padding=2)
        features[0, -1, 0] = 0.5

        with pytest.raises(export_audio.AudioExportError, match="not zeros"):
            export_audio.host_frames(features, mask)

    def test_an_audio_without_a_valid_frame_is_refused(self) -> None:
        features, mask = _extracted(0, padding=2)

        with pytest.raises(export_audio.AudioExportError, match="no soft token"):
            export_audio.host_frames(features, mask)

    def test_more_tokens_than_the_bound_are_refused(self) -> None:
        features, mask = _extracted(4 * export_audio.SOFT_TOKENS_MAX + 1, padding=0)

        with pytest.raises(export_audio.AudioExportError, match="more than the graph's 750"):
            export_audio.host_frames(features, mask)


class TestWindow:
    def test_the_selector_copies_each_token_s_twelve_keys(self) -> None:
        channels, tokens, window = 3, 20, export_audio.WINDOW
        rows = torch.randn(1, channels, tokens)

        padded = functional.pad(rows, (window - 1, 0))
        copies = functional.conv1d(
            padded, export_audio.window_selector(channels), groups=channels
        ).view(channels, window, tokens)

        for shift in range(window):
            for token in range(tokens):
                key = token - (window - 1) + shift
                expected = rows[0, :, key] if key >= 0 else torch.zeros(channels)
                assert torch.equal(copies[:, shift, token], expected)

    def test_the_bias_removes_only_the_keys_before_the_first_token(self) -> None:
        bias = export_audio.window_bias(15, -1e9)

        assert list(bias.shape) == [15, export_audio.WINDOW]
        for token in range(15):
            for column in range(export_audio.WINDOW):
                before = token - (export_audio.WINDOW - 1) + column < 0
                assert float(bias[token, column]) == (-1e9 if before else 0.0)


class TestConstants:
    def test_the_bound_is_the_front_end_s_thirty_seconds(self) -> None:
        """480,000 samples (upstream's truncation) → 2,999 frames → 750 soft tokens."""
        assert export_audio.MAX_SAMPLES == 30 * 16_000
        assert export_audio.SOFT_TOKENS_MAX == 750

    def test_the_files_read_are_pinned(self) -> None:
        assert set(export_audio.READ_FILES) <= set(sources.PIN.sha256)

    def test_the_series_are_one_per_storage(self) -> None:
        assert export_audio.DEFAULT_OUT_DIRS["f32"].name == "embeddinggemma-2-audio"
        assert export_audio.DEFAULT_OUT_DIRS["bf16"].name == "embeddinggemma-2-audio-bf16"
        assert sorted(export_audio.DEFAULT_OUT_DIRS) == sorted(export_audio.WEIGHT_DTYPES)

    def test_the_soft_width_is_the_text_graphs(self) -> None:
        from embeddinggemma2 import distribution

        names = [name for name, *_ in distribution.EMBEDDINGGEMMA2_GRAPH_INPUTS]
        assert names[2] == export.SOFT_EMBEDS
        assert export_audio.SOFT_WIDTH == 512

    def test_another_storage_is_refused_before_anything_is_read(self, tmp_path: Path) -> None:
        with pytest.raises(export_audio.AudioExportError, match="not one of"):
            export_audio.export_series(tmp_path / "missing", tmp_path / "out", dtype="i8")


#: The tiny tower: 2 layers, hidden 16, 2 heads of 8 (the real window and subsampling).
TINY_LAYERS = 2


@pytest.fixture
def tiny_tower() -> export_audio.Tower:
    """A randomly initialised tiny `Gemma4AudioModel` and projector, with finite clip bounds
    (f32, eager — upstream's only attention)."""
    _require_transformers()
    from transformers import AutoModel
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2MultimodalEmbedder,
    )
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    torch.manual_seed(0)
    audio = Gemma4AudioConfig(
        hidden_size=16,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=2,
        subsampling_conv_channels=[export_audio.MEL_BINS, 4],
        output_proj_dims=24,
    )
    audio._attn_implementation = export_audio.UPSTREAM_ATTENTION
    tower = AutoModel.from_config(audio)
    embedder = EmbeddingGemma2MultimodalEmbedder(audio, SimpleNamespace(hidden_size=512))
    with torch.no_grad():
        for module in (tower, embedder):
            for parameter in module.parameters():
                parameter.normal_(0.0, 0.3)
        clipped = [m for m in tower.modules() if isinstance(m, Gemma4ClippableLinear)]
        for index, module in enumerate(clipped):
            bound = 1.5 + 0.25 * (index % 4)
            module.input_min.fill_(-bound)
            module.input_max.fill_(bound)
            module.output_min.fill_(-2 * bound)
            module.output_max.fill_(2 * bound)
    return export_audio.Tower(
        tower=tower.eval().requires_grad_(False),
        embedder=embedder.eval().requires_grad_(False),
        config=SimpleNamespace(audio_config=audio),
    )


def _wrapper(tower: export_audio.Tower) -> export_audio.AudioTowerWrapper:
    return export_audio.AudioTowerWrapper(tower).eval()


class TestTinyTower:
    """The wrapper on a tiny random tower (transformers at the pinned release)."""

    @pytest.mark.parametrize("valid", [36, 37, 38, 39])
    def test_it_equals_upstreams_forward_for_every_residue(self, tiny_tower, valid: int) -> None:
        features, mask = _extracted(valid, padding=1, seed=valid)
        frames = export_audio.host_frames(features, mask)

        golden = export_audio.upstream_soft_tokens(tiny_tower, features, mask)
        same_frames = export_audio.upstream_soft_tokens(
            tiny_tower, frames.features, frames.frame_mask
        )
        with torch.no_grad():
            ours = _wrapper(tiny_tower)(frames.features, frames.frame_mask)

        assert list(ours.shape) == [1, frames.soft_tokens, 512]
        torch.testing.assert_close(ours[0], golden, atol=1e-5, rtol=0)
        torch.testing.assert_close(ours[0], same_frames, atol=1e-5, rtol=0)

    @pytest.mark.parametrize("valid", [36, 37, 38, 39])
    def test_the_mask_matters_only_when_two_frames_are_left_over(
        self, tiny_tower, valid: int
    ) -> None:
        """At `L ≡ 2 (mod 4)` the last layer-0 frame reads the last real frame and upstream masks
        it; at the other residues the padded frames are zeros either way."""
        frames = export_audio.host_frames(*_extracted(valid, padding=0, seed=valid))
        wrapper = _wrapper(tiny_tower)

        with torch.no_grad():
            masked = wrapper(frames.features, frames.frame_mask)
            unmasked = wrapper(frames.features, torch.ones_like(frames.frame_mask))

        if valid % 4 == 2:
            assert float((unmasked - masked).abs()[0, -1].max()) > 1e-2
            assert torch.equal(unmasked[0, :-1], masked[0, :-1])
        else:
            assert torch.equal(unmasked, masked)

    def test_upstream_s_modules_are_left_as_they_are(self, tiny_tower) -> None:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4AudioAttention,
            Gemma4ClippableLinear,
        )

        before = copy.deepcopy(tiny_tower.tower.state_dict())
        wrapper = _wrapper(tiny_tower)

        for layer in tiny_tower.tower.layers:
            assert isinstance(layer.self_attn, Gemma4AudioAttention)
            assert isinstance(layer.feed_forward1.ffw_layer_1, Gemma4ClippableLinear)
        assert not any(isinstance(m, Gemma4ClippableLinear) for m in wrapper.modules())
        assert not any(isinstance(m, Gemma4AudioAttention) for m in wrapper.modules())
        after = tiny_tower.tower.state_dict()
        assert all(torch.equal(before[key], after[key]) for key in before)

    def test_a_clipped_linear_without_finite_bounds_is_refused(self, tiny_tower) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

        module = next(m for m in tiny_tower.tower.modules() if isinstance(m, Gemma4ClippableLinear))
        module.input_max.fill_(float("inf"))

        with pytest.raises(export_audio.AudioExportError, match="non-finite bounds"):
            _wrapper(tiny_tower)


@pytest.fixture
def tiny_graph(tiny_tower) -> tuple[IrGraph, dict[str, torch.Tensor], export_audio.Tower]:
    graph, tensors = export_audio.trace(_wrapper(tiny_tower))
    return graph, tensors, tiny_tower


def _node(graph: IrGraph, op: str) -> IrNode:
    return next(node for node in graph.nodes if node.op == op)


class TestIrForm:
    def test_the_traced_form_passes(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph

        form = export_audio.assert_ir_form(graph, tower.config)

        assert form["inputs"] == [
            ["features", "f32", [1, "4n", 128]],
            ["frame_mask", "f32", [1, "4n"]],
        ]
        assert form["softmax_nodes"] == TINY_LAYERS
        assert form["window_copies"] == 2 * TINY_LAYERS
        assert form["ops"]["conv2d"] == 2

    def test_a_fused_attention_or_a_where_is_refused(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph
        graph.nodes.append(
            IrNode(op="where", ins=["frame_mask", "features", "features"], outs=["x"], attrs={})
        )

        with pytest.raises(AssertionError, match="window form has neither"):
            export_audio.assert_ir_form(graph, tower.config)

    def test_a_layer_without_its_softmax_is_refused(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph
        graph.nodes.remove(_node(graph, "softmax"))

        with pytest.raises(AssertionError, match="softmaxes run on"):
            export_audio.assert_ir_form(graph, tower.config)

    def test_a_window_copy_of_another_form_is_refused(self, tiny_graph) -> None:
        """A window read through the light convolution's weight (another width) is refused."""
        graph, _tensors, tower = tiny_graph
        convolutions = [node for node in graph.nodes if node.op == "conv1d"]
        width = export_audio.WINDOW
        windows = [node for node in convolutions if graph.values[node.ins[1]].shape[-1] == width]
        light = next(node for node in convolutions if graph.values[node.ins[1]].shape[-1] != width)
        windows[0].ins[1] = light.ins[1]

        with pytest.raises(AssertionError, match="the conv1d nodes are"):
            export_audio.assert_ir_form(graph, tower.config)

    def test_a_second_selector_is_refused(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph
        windows = [
            node
            for node in graph.nodes
            if node.op == "conv1d" and graph.values[node.ins[1]].shape[-1] == export_audio.WINDOW
        ]
        graph.values["selector_copy"] = copy.deepcopy(graph.values[windows[0].ins[1]])
        windows[-1].ins[1] = "selector_copy"

        with pytest.raises(AssertionError, match="over 2 selectors"):
            export_audio.assert_ir_form(graph, tower.config)

    def test_a_second_bias_prefix_is_refused(self, tiny_graph) -> None:
        graph, _tensors, tower = tiny_graph
        graph.nodes.append(copy.deepcopy(_node(graph, "sym_prefix_slice")))

        with pytest.raises(AssertionError, match="not one window bias"):
            export_audio.assert_ir_form(graph, tower.config)


def _round_weight_slots_to_bf16(wrapper: torch.nn.Module) -> None:
    """Round what the bf16 storage stores as bf16 — every linear and convolution weight — as the
    checkpoint's bf16 values are (the selector's 0 and 1 are bf16 values already)."""
    with torch.no_grad():
        for module in wrapper.modules():
            if isinstance(module, torch.nn.Linear | torch.nn.Conv1d | torch.nn.Conv2d):
                module.weight.copy_(module.weight.to(torch.bfloat16).to(torch.float32))


class TestStorage:
    def test_bf16_in_every_weight_slot(self, tiny_tower, tmp_path) -> None:
        wrapper = _wrapper(tiny_tower)
        _round_weight_slots_to_bf16(wrapper)
        graph, tensors = export_audio.trace(wrapper)

        stored = publish_model(
            tmp_path / "model.krm",
            graph,
            tensors,
            provenance=Provenance(license=export.LICENSE),
            graph_name=export_audio.GRAPH_NAME,
            weight_dtype="bf16",
        )
        counts = export_audio.assert_storage_form(stored, "bf16")

        # Per layer 10 clipped linears, the depthwise convolution; the input projection, the
        # output projection, the projector, the two subsampling convolutions, one selector.
        assert counts["bf16"] == TINY_LAYERS * (10 + 1) + 3 + 2 + 1

    def test_the_f32_storage_holds_f32_alone(self, tiny_tower, tmp_path) -> None:
        graph, tensors = export_audio.trace(_wrapper(tiny_tower))
        stored = publish_model(
            tmp_path / "model.krm",
            graph,
            tensors,
            provenance=Provenance(license=export.LICENSE),
            graph_name=export_audio.GRAPH_NAME,
        )

        assert set(export_audio.assert_storage_form(stored, "f32")) == {"f32"}
        with pytest.raises(AssertionError, match=r"\(expected bf16\)"):
            export_audio.assert_storage_form(stored, "bf16")
