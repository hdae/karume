"""Script-level contracts of `embeddinggemma2/export.py` (the part that needs no real weights).

The real-weight export is manual. What is pinned here is what turns into a **false PASS** when
broken:

- the band mask is the inclusive, symmetric `|q − kv| <= window` band, commutes with a prefix (the
  folding's premise) and equals the reference the IR check compares against; the full layers get
  no mask
- the attention hook keeps GQA in its true form, reads a missing mask as "every position visible"
  only for a bidirectional module, and refuses what would change the numbers
- the SentenceTransformer layout check rejects every layout the wrapper does not fold
- the golden cases are pinned (names, lengths, one case cutting the window, one at Tmax) and the
  case builder refuses drift and the multimodal placeholder ids
- every file the script reads is pinned
- {@link export.assert_ir_form} rejects a decomposed attention, a widened k / v, a mask on a full
  layer (upstream under torch.export materialises an all-zero one), a missing band, and a band
  from a graph input or of the wrong extent or values

The cases that need transformers at the pinned release, tokenizers or the snapshot SKIP explicitly
without them (CI has none of them).
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embeddinggemma2 import export, sources
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_module
from karume.rope import assert_rope_lifted

#: The upstream SentenceTransformer declarations at the pin.
UPSTREAM_MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": export.SENTENCE_TRANSFORMER_MODULES[0][1]},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": export.SENTENCE_TRANSFORMER_MODULES[1][1]},
    {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": export.SENTENCE_TRANSFORMER_MODULES[2][1],
    },
]
UPSTREAM_POOLING = {"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": True}
UPSTREAM_SENTENCE_BERT = {
    "transformer_task": "feature-extraction",
    "modality_config": {
        "text": {"method": "forward", "method_output_name": "last_hidden_state"},
        "message": {"method": "forward", "method_output_name": "last_hidden_state"},
    },
    "module_output_name": "token_embeddings",
}


def _require_transformers() -> None:
    pytest.importorskip("transformers")
    version = metadata.version("transformers")
    if version != export.REQUIRED_TRANSFORMERS:
        pytest.skip(f"transformers {version} is installed, not {export.REQUIRED_TRANSFORMERS}")


def _snapshot() -> Path:
    if not (sources.SNAPSHOT_DIR / "config.json").is_file():
        pytest.skip(f"the pinned snapshot is not placed at {sources.SNAPSHOT_DIR}")
    return sources.SNAPSHOT_DIR


class TestBandMask:
    def test_it_is_the_inclusive_symmetric_band(self) -> None:
        band = export.bidirectional_band_mask(7, 2)

        assert band.shape == (1, 1, 7, 7)
        assert band.dtype == torch.float32
        for row in range(7):
            for col in range(7):
                expected = 0.0 if abs(row - col) <= 2 else float("-inf")
                assert band[0, 0, row, col].item() == expected, (row, col)

    def test_it_first_masks_at_two_past_the_window(self) -> None:
        """`|q − kv|` reaches `window + 1` only from `length = window + 2` on."""
        assert bool((export.bidirectional_band_mask(3, 2) == 0).all())
        assert not bool((export.bidirectional_band_mask(4, 2) == 0).all())

    def test_a_prefix_of_the_longest_band_is_the_shorter_band(self) -> None:
        """The folding premise: the band at Tmax, sliced by T, is the band at T."""
        longest = export.bidirectional_band_mask(16, 3)

        for length in (2, 4, 5, 11, 16):
            assert torch.equal(
                longest[..., :length, :length], export.bidirectional_band_mask(length, 3)
            )

    def test_it_equals_the_reference_the_ir_check_uses(self) -> None:
        assert torch.equal(export.bidirectional_band_mask(16, 3), export.reference_band(16, 3))

    def test_the_middle_row_sees_two_windows_and_itself(self) -> None:
        band = export.reference_band(2048, 512)

        assert int((band[0, 0, 1024] == 0).sum()) == 1025

    def test_the_full_layers_get_no_mask(self) -> None:
        masks = export.layer_masks(9, 3)

        assert masks[export.FULL_ATTENTION] is None
        assert torch.equal(masks[export.SLIDING_ATTENTION], export.bidirectional_band_mask(9, 3))
        assert export.all_visible_masks() == {
            export.FULL_ATTENTION: None,
            export.SLIDING_ATTENTION: None,
        }

    def test_it_agrees_with_upstreams_mask_function(self) -> None:
        _require_transformers()

        export.assert_band_matches_upstream(64, 5)

    def test_the_upstream_check_rejects_another_window(self, monkeypatch) -> None:
        _require_transformers()
        reference = export.reference_band
        monkeypatch.setattr(
            export, "reference_band", lambda sym_max, window: reference(sym_max, window - 1)
        )

        with pytest.raises(export.EmbeddingGemma2ExportError, match="upstream's mask function"):
            export.assert_band_matches_upstream(64, 5)


BIDIRECTIONAL = SimpleNamespace(is_causal=False)


def _qkv(kv_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return (
        torch.randn(1, 4, 6, 8),
        torch.randn(1, kv_heads, 6, 8),
        torch.randn(1, kv_heads, 6, 8),
    )


def _expanded_attention(query, key, value, mask, scale):
    """Plain attention with k / v repeated to the query heads (what GQA must equal)."""
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = query @ key.transpose(-1, -2) * scale
    if mask is not None:
        scores = scores + mask
    return (scores.softmax(-1) @ value).transpose(1, 2)


class TestAttentionHook:
    @pytest.mark.parametrize("kv_heads", [2, 1], ids=["gqa-2", "mqa-1"])
    def test_a_missing_mask_sees_every_position(self, kv_heads: int) -> None:
        query, key, value = _qkv(kv_heads)

        output, weights = export.bidirectional_gqa_attention(
            BIDIRECTIONAL, query, key, value, None, scaling=1.0
        )

        assert weights is None
        assert output.shape == (1, 6, 4, 8)
        torch.testing.assert_close(output, _expanded_attention(query, key, value, None, 1.0))

    def test_the_band_is_applied(self) -> None:
        query, key, value = _qkv(2)
        mask = export.bidirectional_band_mask(6, 1)

        output, _ = export.bidirectional_gqa_attention(
            BIDIRECTIONAL, query, key, value, mask, scaling=1.0
        )

        torch.testing.assert_close(output, _expanded_attention(query, key, value, mask, 1.0))
        unmasked, _ = export.bidirectional_gqa_attention(
            BIDIRECTIONAL, query, key, value, None, scaling=1.0
        )
        assert not torch.allclose(output, unmasked)

    def test_the_sliding_window_kwarg_is_ignored(self) -> None:
        query, key, value = _qkv(2)
        mask = export.bidirectional_band_mask(6, 1)

        with_kwarg, _ = export.bidirectional_gqa_attention(
            BIDIRECTIONAL, query, key, value, mask, scaling=1.0, sliding_window=2
        )
        without, _ = export.bidirectional_gqa_attention(
            BIDIRECTIONAL, query, key, value, mask, scaling=1.0
        )

        assert torch.equal(with_kwarg, without)

    @pytest.mark.parametrize(
        "module", [SimpleNamespace(is_causal=True), SimpleNamespace()], ids=["causal", "unstated"]
    )
    def test_a_module_that_is_not_bidirectional_is_refused(self, module) -> None:
        """For a causal module a missing mask would silently drop causality."""
        with pytest.raises(ValueError, match="is_causal"):
            export.bidirectional_gqa_attention(module, *_qkv(2), None)

    def test_dropout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            export.bidirectional_gqa_attention(BIDIRECTIONAL, *_qkv(2), None, dropout=0.1)

    @pytest.mark.parametrize("kwarg", ["is_causal", "position_bias", "softcap"])
    def test_numeric_kwargs_are_refused(self, kwarg: str) -> None:
        with pytest.raises(ValueError, match=kwarg):
            export.bidirectional_gqa_attention(BIDIRECTIONAL, *_qkv(2), None, **{kwarg: 1.0})


def _write_layout(
    directory: Path,
    *,
    modules=UPSTREAM_MODULES,
    pooling=UPSTREAM_POOLING,
    sentence_bert=UPSTREAM_SENTENCE_BERT,
) -> None:
    (directory / "1_Pooling").mkdir(parents=True, exist_ok=True)
    (directory / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    (directory / "1_Pooling" / "config.json").write_text(json.dumps(pooling), encoding="utf-8")
    (directory / "sentence_bert_config.json").write_text(
        json.dumps(sentence_bert), encoding="utf-8"
    )


DENSE_MODULE = {
    "idx": 2,
    "name": "2",
    "path": "2_Dense",
    "type": "sentence_transformers.base.modules.dense.Dense",
}


class TestSentenceTransformerLayout:
    def test_it_accepts_the_upstream_layout(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)

        export.assert_sentence_transformer_layout(tmp_path)

    def test_it_accepts_the_pinned_snapshot(self) -> None:
        export.assert_sentence_transformer_layout(_snapshot())

    @pytest.mark.parametrize(
        ("layout", "message"),
        [
            ({"modules": [*UPSTREAM_MODULES[:2], DENSE_MODULE, UPSTREAM_MODULES[2]]}, "modules"),
            ({"modules": UPSTREAM_MODULES[:2]}, "modules"),
            (
                {"modules": [UPSTREAM_MODULES[0], UPSTREAM_MODULES[2], UPSTREAM_MODULES[1]]},
                "modules",
            ),
            ({"pooling": {**UPSTREAM_POOLING, "pooling_mode": "cls"}}, "1_Pooling"),
            ({"pooling": {**UPSTREAM_POOLING, "include_prompt": False}}, "1_Pooling"),
            ({"pooling": {**UPSTREAM_POOLING, "pooling_mode_mean_tokens": True}}, "1_Pooling"),
            (
                {
                    "sentence_bert": {
                        **UPSTREAM_SENTENCE_BERT,
                        "modality_config": {
                            "text": {"method": "forward", "method_output_name": "pooler_output"}
                        },
                    }
                },
                "sentence_bert_config",
            ),
        ],
        ids=[
            "dense-added",
            "normalize-dropped",
            "reordered",
            "cls-pooling",
            "prompt-excluded",
            "extra-option",
            "other-output",
        ],
    )
    def test_it_rejects_a_layout_the_wrapper_does_not_fold(
        self, tmp_path: Path, layout, message: str
    ) -> None:
        _write_layout(tmp_path, **layout)

        with pytest.raises(export.EmbeddingGemma2ExportError, match=message):
            export.assert_sentence_transformer_layout(tmp_path)


class TestTransformersVersion:
    def test_the_pinned_release_passes(self) -> None:
        export.assert_transformers_version(export.REQUIRED_TRANSFORMERS)

    @pytest.mark.parametrize("version", ["5.14.1", "5.19.0", "5.24.0"])
    def test_any_other_release_is_refused(self, version: str) -> None:
        with pytest.raises(export.EmbeddingGemma2ExportError, match=version):
            export.assert_transformers_version(version)


class TestPin:
    def test_every_file_the_script_reads_is_pinned(self) -> None:
        assert set(export.READ_FILES) <= set(sources.PIN.sha256)

    def test_the_text_tower_prefix_is_the_composite_models(self) -> None:
        assert export.CHECKPOINT_TEXT_PREFIX == "language_model."


class TestGoldenCases:
    def test_the_case_names_are_unique_and_each_has_a_pinned_length(self) -> None:
        names = [name for name, _kind, _body in export.GOLDEN_CASES]

        assert len(names) == len(set(names))
        assert set(names) == set(export.CASE_LENGTHS)

    def test_the_sanity_pairs_are_cases(self) -> None:
        names = {name for name, _kind, _body in export.GOLDEN_CASES}

        assert set(export.NEAR_PAIR) <= names
        assert set(export.FAR_PAIR) <= names

    def test_the_lengths_cut_the_window_and_reach_the_bound(self) -> None:
        """Pinned on the 512-position window of the config and the 2,048 bound."""
        lengths = sorted(export.CASE_LENGTHS.values())

        assert export.SYM_MAX in lengths
        assert any(514 <= length < export.SYM_MAX for length in lengths)
        assert all(2 <= length <= export.SYM_MAX for length in lengths)

    def test_cases_with_and_without_a_prompt(self) -> None:
        kinds = {kind for _name, kind, _body in export.GOLDEN_CASES}

        assert {None, export.QUERY, export.DOCUMENT} <= kinds

    def test_the_long_documents_differ_only_by_their_tail(self) -> None:
        assert export.long_document(19).startswith(export.LONG_PASSAGE)
        assert export.long_document(72).startswith(export.long_document(19))
        assert export.long_document(3, filler=2).endswith(" and and")


def _case(ids: list[int]) -> tuple[str, torch.Tensor, torch.Tensor]:
    tensor = torch.tensor([ids])
    return ("case", tensor, torch.ones_like(tensor, dtype=torch.float32))


class TestCaseCoverage:
    def test_cases_that_cut_the_window_and_reach_the_bound_pass(self) -> None:
        export.assert_case_coverage([_case([1] * 5), _case([1] * 8)], sym_max=8, window=3)

    def test_no_case_cutting_the_window_fails(self) -> None:
        with pytest.raises(export.EmbeddingGemma2ExportError, match="window first masks"):
            export.assert_case_coverage([_case([1] * 4), _case([1] * 4)], sym_max=4, window=3)

    def test_no_case_at_the_bound_fails(self) -> None:
        with pytest.raises(export.EmbeddingGemma2ExportError, match="no case runs at T=8"):
            export.assert_case_coverage([_case([1] * 5), _case([1] * 7)], sym_max=8, window=3)


def _composite(directory: Path) -> SimpleNamespace:
    """The fields `build_cases` reads, from the pinned `config.json` (no transformers needed)."""
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    return SimpleNamespace(
        image_token_id=config["image_token_id"],
        video_token_id=config["video_token_id"],
        audio_token_id=config["audio_token_id"],
        text_config=SimpleNamespace(
            bos_token_id=text["bos_token_id"], eos_token_id=text["eos_token_id"]
        ),
    )


class TestBuildCases:
    def test_the_cases_have_their_pinned_lengths_and_run_bos_to_eos(self) -> None:
        pytest.importorskip("tokenizers")
        directory = _snapshot()

        cases = export.build_cases(directory, _composite(directory), export.SYM_MAX)

        assert [name for name, _ids, _mask in cases] == [n for n, _k, _b in export.GOLDEN_CASES]
        for name, ids, pool_mask in cases:
            assert ids.dtype == torch.int64
            assert ids.shape == (1, export.CASE_LENGTHS[name])
            assert ids[0, 0].item() == 2
            assert ids[0, -1].item() == 1
            assert torch.equal(pool_mask, torch.ones_like(ids, dtype=torch.float32))

    def test_a_drifted_length_fails(self, monkeypatch) -> None:
        pytest.importorskip("tokenizers")
        directory = _snapshot()
        monkeypatch.setattr(export, "CASE_LENGTHS", {**export.CASE_LENGTHS, "bare": 13})

        with pytest.raises(export.EmbeddingGemma2ExportError, match="bare: T=12"):
            export.build_cases(directory, _composite(directory), export.SYM_MAX)

    def test_a_multimodal_placeholder_fails(self, monkeypatch) -> None:
        """The composite model swaps these ids for the pad id; the text tower does not."""
        pytest.importorskip("tokenizers")
        directory = _snapshot()
        monkeypatch.setattr(export, "GOLDEN_CASES", (("image", None, "a <|image|> b"),))
        monkeypatch.setattr(export, "CASE_LENGTHS", {"image": 5})

        with pytest.raises(export.EmbeddingGemma2ExportError, match="placeholder ids"):
            export.build_cases(directory, _composite(directory), export.SYM_MAX)


#: The tiny IR's dimensions: 3 layers (sliding, full, sliding), Tmax 16, window 3. They are numbers
#: different from the real ones.
TINY_LAYOUTS = (
    export.AttentionLayout(export.SLIDING_ATTENTION, heads=4, kv_heads=2, head_dim=8),
    export.AttentionLayout(export.FULL_ATTENTION, heads=4, kv_heads=1, head_dim=16),
    export.AttentionLayout(export.SLIDING_ATTENTION, heads=4, kv_heads=2, head_dim=8),
)
TINY_SYM_MAX = 16
TINY_WINDOW = 3
TINY_EMBEDDING = 6
BAND = "const_band"


def _ir_graph(layouts=TINY_LAYOUTS) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """The smallest IR of the expected form (the band a Tmax constant + `sym_prefix_slice`, on the
    sliding layers only) and its tensors."""
    graph = IrGraph(symbols=["T"])
    graph.inputs.append(IrInput(name="input_ids", dtype="i32", shape=[1, "T"]))
    graph.inputs.append(IrInput(name="pool_mask", dtype="f32", shape=[1, "T"]))
    graph.initializers[BAND] = IrInitializer(tensor=f"const.{BAND}", storage=IrStorage("f32"))
    graph.values[BAND] = IrValue(dtype="f32", shape=[1, 1, TINY_SYM_MAX, TINY_SYM_MAX])
    graph.values[f"{BAND}.T"] = IrValue(dtype="f32", shape=[1, 1, "T", "T"])
    graph.nodes.append(
        IrNode(
            op=export.SYM_PREFIX_SLICE_OP,
            ins=[BAND],
            outs=[f"{BAND}.T"],
            attrs={"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )
    for layer, layout in enumerate(layouts):
        graph.values[f"q{layer}"] = IrValue("f32", [1, layout.heads, "T", layout.head_dim])
        for slot in ("k", "v"):
            graph.values[f"{slot}{layer}"] = IrValue(
                "f32", [1, layout.kv_heads, "T", layout.head_dim]
            )
        graph.values[f"attn{layer}"] = IrValue("f32", [1, layout.heads, "T", layout.head_dim])
        mask = [f"{BAND}.T"] if layout.layer_type == export.SLIDING_ATTENTION else []
        graph.nodes.append(
            IrNode(
                op=export.ATTENTION_OP,
                ins=[f"q{layer}", f"k{layer}", f"v{layer}", *mask],
                outs=[f"attn{layer}"],
                attrs={"scale": 1.0},
            )
        )
    graph.values["embedding"] = IrValue("f32", [1, TINY_EMBEDDING])
    graph.outputs.append("embedding")
    tensors = {f"const.{BAND}": export.reference_band(TINY_SYM_MAX, TINY_WINDOW)}
    return graph, tensors


def _check(graph: IrGraph, tensors, layouts=TINY_LAYOUTS) -> dict:
    return export.assert_ir_form(graph, tensors, layouts, TINY_SYM_MAX, TINY_WINDOW, TINY_EMBEDDING)


def _attention(graph: IrGraph, index: int) -> IrNode:
    return [node for node in graph.nodes if node.op == export.ATTENTION_OP][index]


class TestAssertIrForm:
    def test_the_expected_form_passes(self) -> None:
        form = _check(*_ir_graph())

        assert form["attention_nodes"] == 3
        assert form["sliding_layers"] == 2
        assert form["full_layers"] == [1]
        assert form["band_constant"] == BAND
        assert form["band_visible_per_middle_row"] == 2 * TINY_WINDOW + 1

    def test_a_decomposed_attention_is_rejected(self) -> None:
        graph, tensors = _ir_graph(TINY_LAYOUTS[:2])

        with pytest.raises(AssertionError, match="2 attention nodes for 3 layers"):
            _check(graph, tensors)

    def test_k_and_v_widened_to_the_query_heads_are_rejected(self) -> None:
        graph, tensors = _ir_graph()
        graph.values["k1"] = IrValue("f32", [1, 4, "T", 16])

        with pytest.raises(AssertionError, match="repeat_kv"):
            _check(graph, tensors)

    def test_another_head_dim_is_rejected(self) -> None:
        graph, tensors = _ir_graph()
        graph.values["q1"] = IrValue("f32", [1, 4, "T", 8])

        with pytest.raises(AssertionError, match=r"attention\[1\]"):
            _check(graph, tensors)

    def test_a_mask_on_a_full_layer_is_rejected(self) -> None:
        """Upstream under torch.export materialises an all-zero mask for the full layers."""
        graph, tensors = _ir_graph()
        _attention(graph, 1).ins.append(f"{BAND}.T")

        with pytest.raises(AssertionError, match="a full layer has no mask"):
            _check(graph, tensors)

    def test_a_sliding_layer_without_the_band_is_rejected(self) -> None:
        graph, tensors = _ir_graph()
        _attention(graph, 2).ins.pop()

        with pytest.raises(AssertionError, match="a sliding layer has the band"):
            _check(graph, tensors)

    def test_a_band_fed_from_a_graph_input_is_rejected(self) -> None:
        graph, tensors = _ir_graph()
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "T", "T"]))
        _attention(graph, 0).ins[3] = "attention_mask"

        with pytest.raises(AssertionError, match="graph inputs"):
            _check(graph, tensors)

    def test_a_band_not_sliced_by_t_is_rejected(self) -> None:
        graph, tensors = _ir_graph()
        _attention(graph, 0).ins[3] = BAND

        with pytest.raises(AssertionError, match="comes from no node"):
            _check(graph, tensors)

    def test_two_band_constants_are_rejected(self) -> None:
        graph, tensors = _ir_graph()
        graph.initializers["const_other"] = IrInitializer(
            tensor="const.const_other", storage=IrStorage("f32")
        )
        graph.values["const_other"] = IrValue("f32", [1, 1, TINY_SYM_MAX, TINY_SYM_MAX])
        graph.values["const_other.T"] = IrValue("f32", [1, 1, "T", "T"])
        graph.nodes.append(
            IrNode(
                op=export.SYM_PREFIX_SLICE_OP,
                ins=["const_other"],
                outs=["const_other.T"],
                attrs={"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
            )
        )
        _attention(graph, 2).ins[3] = "const_other.T"

        with pytest.raises(AssertionError, match="2 mask constants"):
            _check(graph, tensors)

    def test_a_band_of_another_window_is_rejected(self) -> None:
        graph, tensors = _ir_graph()
        tensors[f"const.{BAND}"] = export.reference_band(TINY_SYM_MAX, TINY_WINDOW + 1)

        with pytest.raises(AssertionError, match="not the inclusive"):
            _check(graph, tensors)

    def test_a_band_of_the_wrong_extent_is_rejected(self) -> None:
        graph, tensors = _ir_graph()
        tensors[f"const.{BAND}"] = export.reference_band(8, TINY_WINDOW)

        with pytest.raises(AssertionError, match=r"\[1, 1, 16, 16\]"):
            _check(graph, tensors)

    def test_another_output_shape_is_rejected(self) -> None:
        graph, tensors = _ir_graph()
        graph.values["embedding"] = IrValue("f32", [1, "T", TINY_EMBEDDING])

        with pytest.raises(AssertionError, match="the output is"):
            _check(graph, tensors)


def _tiny_config():
    """A three-layer EmbeddingGemma 2 text config (sliding, sliding, full) with small numbers."""
    from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
        EmbeddingGemma2TextConfig,
    )

    return EmbeddingGemma2TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=1,
        layer_types=[export.SLIDING_ATTENTION, export.SLIDING_ATTENTION, export.FULL_ATTENTION],
        hidden_size_per_layer_input=8,
        embedding_dim=12,
        sliding_window=TINY_WINDOW,
        max_position_embeddings=64,
    )


@pytest.fixture
def tiny_wrapper():
    """A randomly initialised tiny text tower on the hook, RoPE demoted, wrapped."""
    _require_transformers()
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2TextModel,
    )

    torch.manual_seed(0)
    export.register_attention()
    config = _tiny_config()
    config._attn_implementation = export.ATTENTION_NAME
    model = EmbeddingGemma2TextModel(config).eval()
    assert_rope_lifted(model, "tiny")
    return export.EmbeddingWrapper(model, TINY_WINDOW).eval()


class TestTinyModel:
    """The real export path on a tiny random text tower (transformers at the pinned release)."""

    def test_the_wrapper_equals_upstreams_own_forward(self, tiny_wrapper) -> None:
        torch.manual_seed(1)
        ids = torch.randint(3, 64, (1, 12))
        pool = torch.ones_like(ids, dtype=torch.float32)
        model = tiny_wrapper.model

        with torch.no_grad():
            ours, _ = export.embed(model, ids, pool, export.layer_masks(12, TINY_WINDOW))
            visible, _ = export.embed(model, ids, pool, export.all_visible_masks())
            with export.upstream_attention(model):
                upstream, _ = export.embed(model, ids, pool, torch.ones_like(ids))

        torch.testing.assert_close(ours, upstream, atol=1e-6, rtol=0)
        assert float((ours - visible).abs().max()) > 1e-4
        assert model.config._attn_implementation == export.ATTENTION_NAME

    def test_the_export_has_the_expected_form(self, tiny_wrapper) -> None:
        ids = torch.randint(3, 64, (1, TINY_SYM_MAX))
        pool = torch.ones_like(ids, dtype=torch.float32)
        seq = torch.export.Dim("T", min=2, max=TINY_SYM_MAX)

        graph, tensors = export_module(
            tiny_wrapper,
            (ids, pool),
            dynamic_shapes=({1: seq}, {1: seq}),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        config = tiny_wrapper.model.config
        form = export.assert_ir_form(
            graph,
            tensors,
            export.attention_layouts(config),
            TINY_SYM_MAX,
            TINY_WINDOW,
            int(config.embedding_dim),
        )

        assert form["attention_nodes"] == 3
        assert form["full_layers"] == [2]
