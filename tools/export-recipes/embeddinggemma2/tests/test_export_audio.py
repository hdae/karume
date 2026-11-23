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
- across upstream's 12-query chunks (1 to 40 soft tokens and a partial fifth chunk): every
  layer's attention gives nonzero weight to exactly the keys upstream's does (the query and the 11
  before it), and in f64 (upstream's f32 casts lifted) the wrapper's soft tokens equal upstream's to
  f64 noise
- the traced form: two inputs on `4n`, one output `[1, n, 512]`, no fused attention and no
  `where`, the softmaxes on the window, one selector shared by every window copy, one bias prefix —
  and each broken form is refused
- the storage: bf16 in every weight slot, f32 elsewhere; under i8 every weight slot i8 but the
  two subsampling convolutions and the selector, which stay bf16
- the i8 rounding: every linear and depthwise convolution, never the kept convolutions, and
  upstream's forward — the goldens — runs the same rounded weights (the wrapper holds upstream's
  own parameters, no copy)
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
        assert export_audio.DEFAULT_OUT_DIRS["i8"].name == "embeddinggemma-2-audio-i8"
        assert sorted(export_audio.DEFAULT_OUT_DIRS) == sorted(export_audio.WEIGHT_DTYPES)

    def test_the_i8_exceptions_are_the_two_subsampling_convolutions_and_the_selector(
        self,
    ) -> None:
        assert export_audio.I8_KEPT_BF16_WEIGHTS == ("layer0.conv.weight", "layer1.conv.weight")
        assert export_audio.i8_storage_overrides("layers.11.self_attn.window_selector") == {
            "layer0.conv.weight": "bf16",
            "layer1.conv.weight": "bf16",
            "layers.11.self_attn.window_selector": "bf16",
        }

    def test_the_soft_width_is_the_text_graphs(self) -> None:
        from embeddinggemma2 import distribution

        names = [name for name, *_ in distribution.EMBEDDINGGEMMA2_GRAPH_INPUTS]
        assert names[2] == export.SOFT_EMBEDS
        assert export_audio.SOFT_WIDTH == 512

    def test_another_storage_is_refused_before_anything_is_read(self, tmp_path: Path) -> None:
        with pytest.raises(export_audio.AudioExportError, match="not one of"):
            export_audio.export_series(tmp_path / "missing", tmp_path / "out", dtype="f16")


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

    def test_the_wrapper_holds_upstreams_own_weights(self, tiny_tower) -> None:
        """No copy: every parameter of the wrapper is one of upstream's (the copied layers share
        them), so a rounding of the wrapper's weights is a rounding of the modules the goldens
        run."""
        upstream = {
            parameter.untyped_storage().data_ptr()
            for module in (tiny_tower.tower, tiny_tower.embedder)
            for parameter in module.parameters()
        }

        wrapper = _wrapper(tiny_tower)

        for name, parameter in wrapper.named_parameters():
            assert parameter.untyped_storage().data_ptr() in upstream, name

    def test_a_clipped_linear_without_finite_bounds_is_refused(self, tiny_tower) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

        module = next(m for m in tiny_tower.tower.modules() if isinstance(m, Gemma4ClippableLinear))
        module.input_max.fill_(float("inf"))

        with pytest.raises(export_audio.AudioExportError, match="non-finite bounds"):
            _wrapper(tiny_tower)


#: Upstream's attention chunk: 12 queries per block over a 24-key context (the block's 12 and the
#: 12 before it).
CHUNK = 12

#: The soft-token counts across the chunks: 1 to 40 (every position of the first three chunks and
#: into the fourth, so the first, second and third boundaries) and 53 (a partial fifth chunk of 5).
CHUNK_TOKENS = [*range(1, 41), 53]

#: The bound of the wrapper against upstream in f64 (max abs of the tiny tower's soft tokens, which
#: reach about 4.7). Measured on 2026-11-23 over {@link CHUNK_TOKENS}: at most 3.6e-15 — the two
#: compute the same values in another order. Far below any wrong key: a window off by one key for
#: the queries past the first chunk moves the soft tokens by more than 1e-2.
F64_TOLERANCE = 1e-13


def _chunk_frames(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extractor-like frames of an audio of `tokens` soft tokens, the residue of `L` mod 4 cycling
    with `tokens`, then five padding frames."""
    return _extracted(4 * tokens - tokens % 4, padding=5, seed=tokens)


def _upstream_pattern(weights: torch.Tensor, tokens: int) -> torch.Tensor:
    """`[H, n, n]` from upstream's returned weights `[1, H, blocks, 12, 24]`: block `b`'s query `i`
    is token `12b + i`, its context column `c` key `12b − 12 + c`. MUST: a key outside `[0, n)` or
    a query past `n` has weight 0 (the tokens upstream pads and masks)."""
    _, heads, blocks, chunk, context = weights.shape
    assert (chunk, context) == (CHUNK, 2 * CHUNK)
    dense = torch.zeros(heads, tokens, tokens, dtype=weights.dtype)
    for block in range(blocks):
        for row in range(chunk):
            query = CHUNK * block + row
            for column in range(context):
                key = CHUNK * block - CHUNK + column
                value = weights[0, :, block, row, column]
                if query < tokens and 0 <= key < tokens:
                    dense[:, query, key] = value
                elif query < tokens:
                    assert not bool(value.any()), (query, key)
    return dense


def _wrapper_pattern(weights: torch.Tensor) -> torch.Tensor:
    """`[H, n, n]` from the wrapper's softmax `[H, n, 12]`: query `t`'s column `j` is key
    `t − 11 + j`. MUST: a column before the first token has weight 0."""
    heads, tokens, window = weights.shape
    dense = torch.zeros(heads, tokens, tokens, dtype=weights.dtype)
    for query in range(tokens):
        for column in range(window):
            key = query - (window - 1) + column
            if key >= 0:
                dense[:, query, key] = weights[:, query, column]
            else:
                assert not bool(weights[:, query, column].any()), (query, key)
    return dense


def _band(tokens: int) -> torch.Tensor:
    """`[n, n]`: True where key `k` is query `t` or one of the 11 before it."""
    query = torch.arange(tokens).unsqueeze(1)
    key = torch.arange(tokens).unsqueeze(0)
    return (key <= query) & (key > query - export_audio.WINDOW)


class TestChunkBoundaries:
    """The wrapper across upstream's 12-query chunks on the tiny tower (transformers at the pinned
    release)."""

    @pytest.mark.parametrize("tokens", CHUNK_TOKENS)
    def test_every_layer_weighs_exactly_upstream_s_keys(
        self, tiny_tower, tokens: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        features, mask = _chunk_frames(tokens)
        frames = export_audio.host_frames(features, mask)
        with torch.no_grad():
            upstream = tiny_tower.tower(features, mask, return_dict=True, output_attentions=True)
        softmax = functional.softmax
        captured: list[torch.Tensor] = []

        def recording(*args, **kwargs) -> torch.Tensor:
            result = softmax(*args, **kwargs)
            captured.append(result.detach().clone())
            return result

        # The wrapper's only softmaxes are its attentions' (the traced form holds one per layer).
        monkeypatch.setattr(functional, "softmax", recording)
        with torch.no_grad():
            _wrapper(tiny_tower)(frames.features, frames.frame_mask)
        monkeypatch.undo()

        assert frames.soft_tokens == tokens
        assert len(captured) == len(upstream.attentions) == TINY_LAYERS
        band = _band(tokens)
        for layer, (ours, theirs) in enumerate(zip(captured, upstream.attentions, strict=True)):
            ours_dense = _wrapper_pattern(ours)
            theirs_dense = _upstream_pattern(theirs, tokens)
            assert torch.equal(ours_dense != 0, theirs_dense != 0), f"layer {layer}"
            assert torch.equal(theirs_dense != 0, band.expand_as(theirs_dense)), f"layer {layer}"

    @pytest.mark.parametrize("tokens", CHUNK_TOKENS)
    def test_in_f64_it_equals_upstream_s_forward(
        self, tiny_tower, tokens: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        features, mask = _chunk_frames(tokens)
        frames = export_audio.host_frames(features, mask)
        tower = export_audio.Tower(
            tower=copy.deepcopy(tiny_tower.tower).double(),
            embedder=copy.deepcopy(tiny_tower.embedder).double(),
            config=tiny_tower.config,
        )
        # Lift upstream's f32 casts (`.float()` and the softmax's `dtype=torch.float32`) for f64
        # tensors, so both sides run in f64 end to end.
        to_float, softmax = torch.Tensor.float, functional.softmax
        monkeypatch.setattr(
            torch.Tensor,
            "float",
            lambda self, *args, **kwargs: (
                self if self.dtype == torch.float64 else to_float(self, *args, **kwargs)
            ),
        )
        monkeypatch.setattr(
            functional,
            "softmax",
            lambda tensor, dim=None, _stacklevel=3, dtype=None: softmax(
                tensor, dim=dim, dtype=None if tensor.dtype == torch.float64 else dtype
            ),
        )

        golden = export_audio.upstream_soft_tokens(tower, features.double(), mask)
        with torch.no_grad():
            ours = (
                export_audio.AudioTowerWrapper(tower)
                .double()
                .eval()(frames.features.double(), frames.frame_mask.double())
            )

        assert golden.dtype == ours.dtype == torch.float64
        assert list(ours.shape) == [1, tokens, 512]
        torch.testing.assert_close(ours[0], golden, atol=F64_TOLERANCE, rtol=0)


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


class TestInt8Rounding:
    """`round_weights_to_i8` on the tiny tower: every linear and depthwise convolution becomes
    per-row int8, the two subsampling convolutions stay as they are, and upstream's forward (the
    goldens) runs the same rounded weights."""

    def test_it_rounds_every_weight_but_the_two_subsampling_convolutions(self, tiny_tower) -> None:
        wrapper = _wrapper(tiny_tower)
        kept = {
            name: wrapper.get_parameter(name).clone() for name in export_audio.I8_KEPT_BF16_WEIGHTS
        }
        selector = wrapper.layers[0].self_attn.window_selector.clone()

        report = export_audio.round_weights_to_i8(wrapper)

        linears = [m for m in wrapper.modules() if isinstance(m, torch.nn.Linear)]
        depthwise = [m for m in wrapper.modules() if isinstance(m, torch.nn.Conv1d)]
        # Per layer 10 clipped linears; the input projection, the output projection, the
        # projector; per layer one depthwise convolution.
        assert len(linears) == TINY_LAYERS * 10 + 3 and len(depthwise) == TINY_LAYERS
        assert len(report.scales) == len(linears) + len(depthwise)
        assert not set(report.scales) & set(export_audio.I8_KEPT_BF16_WEIGHTS)
        for name, before in kept.items():
            assert torch.equal(wrapper.get_parameter(name), before), name
        assert torch.equal(wrapper.layers[0].self_attn.window_selector, selector)

    def test_upstreams_forward_runs_the_rounded_weights(self, tiny_tower) -> None:
        features, mask = _extracted(38, padding=1, seed=38)
        frames = export_audio.host_frames(features, mask)
        wrapper = _wrapper(tiny_tower)
        before = export_audio.upstream_soft_tokens(tiny_tower, features, mask)

        export_audio.round_weights_to_i8(wrapper)
        golden = export_audio.upstream_soft_tokens(tiny_tower, features, mask)
        with torch.no_grad():
            ours = wrapper(frames.features, frames.frame_mask)

        assert float((golden - before).abs().max()) > 1e-4
        torch.testing.assert_close(ours[0], golden, atol=1e-5, rtol=0)

    def test_a_kept_name_that_is_no_module_weight_is_refused(self, tiny_tower, monkeypatch) -> None:
        wrapper = _wrapper(tiny_tower)
        monkeypatch.setattr(export_audio, "I8_KEPT_BF16_WEIGHTS", ("layer0.conv.wieght",))

        with pytest.raises(export_audio.AudioExportError, match="not module weights"):
            export_audio.round_weights_to_i8(wrapper)


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

    def test_i8_but_the_two_subsampling_convolutions_and_the_selector(
        self, tiny_tower, tmp_path
    ) -> None:
        wrapper = _wrapper(tiny_tower)
        # The kept weights are the checkpoint's bf16 values (the real tower's are).
        with torch.no_grad():
            for name in export_audio.I8_KEPT_BF16_WEIGHTS:
                weight = wrapper.get_parameter(name)
                weight.copy_(weight.to(torch.bfloat16).to(torch.float32))
        report = export_audio.round_weights_to_i8(wrapper)
        graph, tensors = export_audio.trace(wrapper)
        selector = export_audio.assert_ir_form(graph, tiny_tower.config)["window_selector"]
        overrides = export_audio.i8_storage_overrides(selector)

        stored = publish_model(
            tmp_path / "model.krm",
            graph,
            tensors,
            provenance=Provenance(license=export.LICENSE),
            graph_name=export_audio.GRAPH_NAME,
            weight_dtype="i8",
            weight_scales=report.scales,
            weight_dtype_overrides=overrides,
        )
        counts = export_audio.assert_storage_form(stored, "i8", tuple(report.scales), overrides)

        assert selector == f"layers.{TINY_LAYERS - 1}.self_attn.window_selector"
        assert counts["i8"] == TINY_LAYERS * (10 + 1) + 3
        assert counts["bf16"] == 2 + 1
        with pytest.raises(AssertionError, match=r"layer0.conv.weight: bf16 \(expected i8\)"):
            export_audio.assert_storage_form(stored, "i8", tuple(report.scales), {})
