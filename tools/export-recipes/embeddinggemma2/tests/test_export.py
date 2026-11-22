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
- the tracked golden-case fixture (the TypeScript pipeline e2e's texts) holds exactly those cases
- every file the script reads is pinned
- {@link export.assert_ir_form} rejects a decomposed attention, a widened k / v, a mask on a full
  layer (upstream under torch.export materialises an all-zero one), a missing band, and a band
  from a graph input or of the wrong extent or values
- the soft rows: without one the wrapper is the `input_ids` path bit for bit, with them it is the
  composite model's merge (the pad id looked up, `masked_scatter` of unscaled rows), the io holds
  the four inputs (zeros and false), and {@link export.assert_soft_merge} rejects soft rows of
  another width, a full-width mask, scaled soft rows, a soft input read elsewhere, a per-layer
  projection of the unmerged lookup, a text operand not from the ids, and a second `where`
- {@link export.declare_band} declares the window on exactly the sliding layers and prunes the
  traced band constant alone, and {@link export.assert_band_form} rejects a mask left on an
  attention, a band missing, of another width or on a full layer, anything else pruned and a graph
  whose bound of T is no longer readable at Tmax; the result passes the writer's rules
- `--dtype` writes each storage to a series of its own, and {@link export.assert_storage_form}
  rejects a weight slot left f32 under bf16 or i8, a compressed storage outside a weight slot, any
  other storage, and under i8 weight slots that are not exactly the rounded weights; a weight bf16
  cannot hold fails the bf16 write
- {@link export.round_weights_to_i8} rounds exactly every linear and the token table to per-row
  int8, keyed by the exported tensor keys, and refuses a model with another table or another
  weight-slot module before it rounds anything (on a toy module shaped like the wrapper, without
  transformers, and on the tiny tower); the rounded model is what the i8 write stores (each weight
  with its scale), and an i8 write without the rounding fails

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
from embeddinggemma2.card import EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS
from karume.container import Provenance
from karume.emit import EmitError, eligible_compressed_initializers
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import publish_model
from karume.quantize import quantize_to_int8, round_weights_to_bf16
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
        """Pinned on the 512-position window of the config and the 8,192 bound."""
        lengths = sorted(export.CASE_LENGTHS.values())

        assert export.SYM_MAX in lengths
        assert any(514 <= length < export.SYM_MAX for length in lengths)
        assert all(2 <= length <= export.SYM_MAX for length in lengths)

    def test_the_bound_is_the_upstream_context(self) -> None:
        """The card states the upstream context; the graph's bound is that context."""
        assert export.SYM_MAX == EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS == 8192

    def test_the_long_cases_run_past_the_earlier_bound_up_to_the_context(self) -> None:
        """Past 2,048 (the bound before the band was declared), near 5,000–6,000, and at 8,192."""
        lengths = set(export.CASE_LENGTHS.values())

        assert any(2048 < length <= 4096 for length in lengths)
        assert any(5000 <= length <= 6000 for length in lengths)
        assert 8192 in lengths
        assert {609, 1508, 2048} <= lengths

    def test_cases_with_and_without_a_prompt(self) -> None:
        kinds = {kind for _name, kind, _body in export.GOLDEN_CASES}

        assert {None, export.QUERY, export.DOCUMENT} <= kinds

    def test_the_long_documents_differ_only_by_their_tail(self) -> None:
        assert export.long_document(19).startswith(export.LONG_PASSAGE)
        assert export.long_document(72).startswith(export.long_document(19))
        assert export.long_document(3, filler=2).endswith(" and and")


class TestGoldenCaseFixture:
    """The tracked fixture the TypeScript pipeline's e2e reads holds exactly the declared cases (a
    case edited without regenerating it would leave the e2e embedding the old text against the new
    golden)."""

    def test_the_tracked_fixture_holds_the_golden_cases(self) -> None:
        fixture = json.loads(export.GOLDEN_CASE_FIXTURE.read_text(encoding="utf-8"))

        assert fixture == export.golden_case_fixture()

    def test_each_case_carries_its_prompt_name_body_and_pinned_length(self) -> None:
        cases = export.golden_case_fixture()["cases"]

        assert [case["name"] for case in cases] == [name for name, _, _ in export.GOLDEN_CASES]
        for (name, kind, body), case in zip(export.GOLDEN_CASES, cases, strict=True):
            assert case.get("prompt") == kind, name
            assert case["text"] == body, name
            assert case["tokens"] == export.CASE_LENGTHS[name], name

    def test_a_case_without_a_prompt_has_no_prompt_key(self) -> None:
        bare = [case for case in export.golden_case_fixture()["cases"] if case["name"] == "bare"]

        assert bare == [{"name": "bare", "text": bare[0]["text"], "tokens": 12}]

    def test_the_fixture_names_the_series_whose_goldens_it_matches(self) -> None:
        assert export.golden_case_fixture()["series"] == export.TEXT_SERIES

    def test_the_writer_writes_what_the_tracked_fixture_holds(self, tmp_path: Path) -> None:
        written = export.write_golden_case_fixture(tmp_path / "nested" / "golden-cases.json")

        assert written.read_bytes() == export.GOLDEN_CASE_FIXTURE.read_bytes()

    def test_every_prompt_name_is_a_prompt_of_the_pinned_snapshot(self) -> None:
        directory = _snapshot()
        sources.assert_pinned(directory, ["config_sentence_transformers.json"])
        prompts = export.load_prompts(directory)

        for case in export.golden_case_fixture()["cases"]:
            if "prompt" in case:
                assert case["prompt"] in prompts, case["name"]


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
#: The tiny IR's hidden width (the soft rows' width).
TINY_HIDDEN = 4
BAND = "const_band"


def _ir_graph(layouts=TINY_LAYOUTS) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """The smallest IR of the expected form (the band a Tmax constant + `sym_prefix_slice`, on the
    sliding layers only) and its tensors."""
    graph = IrGraph(symbols=["T"])
    graph.inputs.append(IrInput(name="input_ids", dtype="i32", shape=[1, "T"]))
    graph.inputs.append(IrInput(name="pool_mask", dtype="f32", shape=[1, "T"]))
    graph.inputs.append(IrInput(name=export.SOFT_EMBEDS, dtype="f32", shape=[1, "T", TINY_HIDDEN]))
    graph.inputs.append(IrInput(name=export.SOFT_MASK, dtype="bool", shape=[1, "T", 1]))
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

    def test_a_graph_without_the_soft_inputs_is_rejected(self) -> None:
        graph, tensors = _ir_graph()
        graph.inputs = graph.inputs[:2]

        with pytest.raises(AssertionError, match="the soft inputs are missing"):
            _check(graph, tensors)


def _merge_graph() -> IrGraph:
    """The smallest graph of the merge's form: `input_ids` → `embedding` → `mul` (the scale) →
    `where(soft_mask, soft_embeds, ·)` → the per-layer projection and a layer, both reading the
    merged rows."""
    graph = IrGraph(symbols=["T"])
    graph.inputs += [
        IrInput(name="input_ids", dtype="i32", shape=[1, "T"]),
        IrInput(name="pool_mask", dtype="f32", shape=[1, "T"]),
        IrInput(name=export.SOFT_EMBEDS, dtype="f32", shape=[1, "T", TINY_HIDDEN]),
        IrInput(name=export.SOFT_MASK, dtype="bool", shape=[1, "T", 1]),
    ]
    for name, shape in (("table", [8, TINY_HIDDEN]), ("scale", []), ("ple", [6, TINY_HIDDEN])):
        graph.initializers[name] = IrInitializer(tensor=name, storage=IrStorage("f32"))
        graph.values[name] = IrValue("f32", shape)
    for name in ("lookup", "scaled", "merged", "layer"):
        graph.values[name] = IrValue("f32", [1, "T", TINY_HIDDEN])
    graph.values["projected"] = IrValue("f32", [1, "T", 6])
    graph.nodes += [
        IrNode(op="embedding", ins=["table", "input_ids"], outs=["lookup"], attrs={}),
        IrNode(op="mul", ins=["lookup", "scale"], outs=["scaled"], attrs={}),
        IrNode(
            op=export.WHERE_OP,
            ins=[export.SOFT_MASK, export.SOFT_EMBEDS, "scaled"],
            outs=["merged"],
            attrs={},
        ),
        IrNode(op="linear", ins=["merged", "ple"], outs=["projected"], attrs={}),
        IrNode(op="rms_norm", ins=["merged", "scale"], outs=["layer"], attrs={}),
    ]
    graph.outputs.append("layer")
    return graph


def _node(graph: IrGraph, op: str) -> IrNode:
    return next(node for node in graph.nodes if node.op == op)


class TestAssertSoftMerge:
    """{@link export.assert_soft_merge} — the form that keeps soft rows unscaled and reaching every
    reader of the rows; each broken form still passes every text golden."""

    def test_the_expected_form_passes(self) -> None:
        merge = export.assert_soft_merge(_merge_graph(), TINY_HIDDEN)

        assert merge == {
            "merge": "merged",
            "text_operand": "scaled",
            "text_chain": ["input_ids", "lookup", "scaled"],
        }

    def test_soft_rows_of_another_width_are_rejected(self) -> None:
        with pytest.raises(AssertionError, match="'soft_embeds' is"):
            export.assert_soft_merge(_merge_graph(), TINY_HIDDEN + 1)

    def test_a_mask_of_the_full_width_is_rejected(self) -> None:
        graph = _merge_graph()
        graph.inputs[3] = IrInput(name=export.SOFT_MASK, dtype="bool", shape=[1, "T", TINY_HIDDEN])

        with pytest.raises(AssertionError, match="'soft_mask' is"):
            export.assert_soft_merge(graph, TINY_HIDDEN)

    def test_soft_rows_scaled_before_the_merge_are_rejected(self) -> None:
        """Upstream writes the soft tokens over the scaled lookup as they are."""
        graph = _merge_graph()
        graph.values["soft_scaled"] = IrValue("f32", [1, "T", TINY_HIDDEN])
        graph.nodes.insert(
            0, IrNode(op="mul", ins=[export.SOFT_EMBEDS, "scale"], outs=["soft_scaled"], attrs={})
        )
        _node(graph, export.WHERE_OP).ins[1] = "soft_scaled"

        with pytest.raises(AssertionError, match="unscaled"):
            export.assert_soft_merge(graph, TINY_HIDDEN)

    def test_a_soft_input_read_besides_the_merge_is_rejected(self) -> None:
        graph = _merge_graph()
        _node(graph, "rms_norm").ins[0] = export.SOFT_EMBEDS

        with pytest.raises(AssertionError, match=r"\['soft_embeds'\] are read besides the merge"):
            export.assert_soft_merge(graph, TINY_HIDDEN)

    def test_a_per_layer_projection_of_the_unmerged_lookup_is_rejected(self) -> None:
        graph = _merge_graph()
        _node(graph, "linear").ins[0] = "scaled"

        with pytest.raises(AssertionError, match=r"read outside the merge: \['scaled → linear'\]"):
            export.assert_soft_merge(graph, TINY_HIDDEN)

    def test_a_text_operand_that_does_not_come_from_the_ids_is_rejected(self) -> None:
        graph = _merge_graph()
        _node(graph, "mul").ins[0] = "pool_mask"

        with pytest.raises(AssertionError, match="does not come from 'input_ids' alone"):
            export.assert_soft_merge(graph, TINY_HIDDEN)

    def test_a_second_where_is_rejected(self) -> None:
        graph = _merge_graph()
        graph.values["again"] = IrValue("f32", [1, "T", TINY_HIDDEN])
        graph.nodes.append(
            IrNode(
                op=export.WHERE_OP,
                ins=[export.SOFT_MASK, export.SOFT_EMBEDS, "layer"],
                outs=["again"],
                attrs={},
            )
        )

        with pytest.raises(AssertionError, match="not one"):
            export.assert_soft_merge(graph, TINY_HIDDEN)


class TestSoftInputs:
    """The host's soft inputs as the recipe builds them (zeros and false around each block)."""

    def test_no_soft_row_is_zeros_and_all_false(self) -> None:
        ids = torch.zeros(1, 5, dtype=torch.int64)

        soft_embeds, soft_mask = export.no_soft_inputs(ids, 3)

        assert soft_embeds.dtype == torch.float32 and list(soft_embeds.shape) == [1, 5, 3]
        assert soft_mask.dtype == torch.bool and list(soft_mask.shape) == [1, 5, 1]
        assert not soft_embeds.any() and not soft_mask.any()

    def test_blocks_are_placed_at_their_rows_only(self) -> None:
        ids = torch.zeros(1, 8, dtype=torch.int64)
        first, second = torch.ones(2, 3), torch.full((3, 3), 2.0)

        soft_embeds, soft_mask = export.soft_inputs(ids, 3, [(1, first), (5, second)])

        assert soft_mask[0, :, 0].tolist() == [False, True, True, False, False, True, True, True]
        assert torch.equal(soft_embeds[0, 1:3], first)
        assert torch.equal(soft_embeds[0, 5:8], second)
        assert not soft_embeds[0, [0, 3, 4]].any()

    @pytest.mark.parametrize(
        ("blocks", "match"),
        [
            ([(7, torch.ones(2, 3))], "does not fit"),
            ([(0, torch.ones(2, 4))], "does not fit"),
            ([(0, torch.ones(3, 3)), (2, torch.ones(1, 3))], "overlaps"),
        ],
    )
    def test_a_block_past_the_end_of_another_width_or_overlapping_fails(
        self, blocks, match: str
    ) -> None:
        with pytest.raises(export.EmbeddingGemma2ExportError, match=match):
            export.soft_inputs(torch.zeros(1, 8, dtype=torch.int64), 3, blocks)


ROPE = "const_rope"


def _traced_graph() -> IrGraph:
    """The tiny traced IR plus a RoPE-like table `[1, Tmax, 8]` sliced by T (what still bounds T
    once the band constant is pruned) feeding the output."""
    graph, _tensors = _ir_graph()
    graph.initializers[ROPE] = IrInitializer(tensor=f"const.{ROPE}", storage=IrStorage("f32"))
    graph.values[ROPE] = IrValue("f32", [1, TINY_SYM_MAX, 8])
    graph.values[f"{ROPE}.T"] = IrValue("f32", [1, "T", 8])
    graph.nodes.insert(
        0,
        IrNode(
            op=export.SYM_PREFIX_SLICE_OP,
            ins=[ROPE],
            outs=[f"{ROPE}.T"],
            attrs={"sym": "T", "slices": [{"dim": 1, "coeff": 1, "offset": 0}]},
        ),
    )
    # The inputs, every attention output and the table reach the output (the pruning keeps live
    # values only and refuses an input left unread).
    graph.nodes.append(
        IrNode(
            op="concat",
            ins=[
                "input_ids",
                "pool_mask",
                export.SOFT_EMBEDS,
                export.SOFT_MASK,
                *(f"attn{layer}" for layer in range(len(TINY_LAYOUTS))),
                f"{ROPE}.T",
            ],
            outs=["embedding"],
            attrs={},
        )
    )
    return graph


def _band_check(graph: IrGraph, pruned=(BAND,)) -> dict:
    return export.assert_band_form(graph, TINY_LAYOUTS, TINY_SYM_MAX, TINY_WINDOW, pruned, BAND)


class TestDeclareBand:
    def test_the_sliding_layers_declare_the_window_and_the_band_constant_is_pruned(self) -> None:
        traced = _traced_graph()

        graph, pruned = export.declare_band(traced, TINY_LAYOUTS, TINY_WINDOW)

        assert pruned == [BAND]
        assert [node.attrs.get("band") for node in _attentions(graph)] == [
            TINY_WINDOW,
            None,
            TINY_WINDOW,
        ]
        assert all(len(node.ins) == 3 for node in _attentions(graph))
        assert BAND not in graph.initializers
        assert f"{BAND}.T" not in graph.values
        assert ROPE in graph.initializers
        # The traced graph is only read.
        assert len(_attentions(traced)[0].ins) == 4

    def test_the_result_has_the_band_form(self) -> None:
        graph, pruned = export.declare_band(_traced_graph(), TINY_LAYOUTS, TINY_WINDOW)

        form = _band_check(graph, pruned)

        assert form == {"band_layers": [0, 2], "band": TINY_WINDOW, "pruned": [BAND], "bound": 16}

    def test_layouts_of_another_layer_count_are_refused(self) -> None:
        with pytest.raises(AssertionError, match="3 attention nodes for 2 layers"):
            export.declare_band(_traced_graph(), TINY_LAYOUTS[:2], TINY_WINDOW)


def _attentions(graph: IrGraph) -> list[IrNode]:
    return [node for node in graph.nodes if node.op == export.ATTENTION_OP]


def _declared() -> IrGraph:
    graph, _pruned = export.declare_band(_traced_graph(), TINY_LAYOUTS, TINY_WINDOW)
    return graph


class TestAssertBandForm:
    def test_an_attention_that_still_takes_a_mask_is_rejected(self) -> None:
        graph = _declared()
        _attentions(graph)[0].ins.append(f"{ROPE}.T")

        with pytest.raises(AssertionError, match="no attention takes a mask"):
            _band_check(graph)

    def test_a_sliding_layer_without_the_band_is_rejected(self) -> None:
        graph = _declared()
        del _attentions(graph)[2].attrs["band"]

        with pytest.raises(AssertionError, match=r"attention\[2\] \(sliding_attention\)"):
            _band_check(graph)

    def test_another_width_is_rejected(self) -> None:
        graph = _declared()
        _attentions(graph)[0].attrs["band"] = TINY_WINDOW + 1

        with pytest.raises(AssertionError, match=r"attrs\.band is 4, not 3"):
            _band_check(graph)

    def test_a_band_on_a_full_layer_is_rejected(self) -> None:
        graph = _declared()
        _attentions(graph)[1].attrs["band"] = TINY_WINDOW

        with pytest.raises(AssertionError, match=r"attention\[1\] \(full_attention\)"):
            _band_check(graph)

    def test_pruning_anything_but_the_band_constant_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="the surgery pruned"):
            _band_check(_declared(), pruned=(BAND, ROPE))

    def test_a_graph_whose_bound_is_no_longer_readable_is_rejected(self) -> None:
        graph = _declared()
        graph.nodes = [node for node in graph.nodes if node.op != export.SYM_PREFIX_SLICE_OP]

        with pytest.raises(AssertionError, match=r"bound T at \[\], not exactly 16"):
            _band_check(graph)

    @pytest.mark.parametrize("cut", [{"coeff": 2, "offset": 0}, {"coeff": 1, "offset": 1}])
    def test_a_slice_with_a_coefficient_or_an_offset_is_rejected(self, cut) -> None:
        graph = _declared()
        rope = next(node for node in graph.nodes if node.op == export.SYM_PREFIX_SLICE_OP)
        rope.attrs["slices"] = [{"dim": 1, **cut}]

        with pytest.raises(
            AssertionError, match="read only from a slice with coeff 1 and offset 0"
        ):
            _band_check(graph)

    def test_a_bound_other_than_tmax_is_rejected(self) -> None:
        graph = _declared()
        graph.values[ROPE] = IrValue("f32", [1, 8, 8])

        with pytest.raises(AssertionError, match=r"bound T at \[8\], not exactly 16"):
            _band_check(graph)


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

        graph, tensors = export.trace(tiny_wrapper, ids, pool, TINY_SYM_MAX)
        config = tiny_wrapper.model.config
        form = export.assert_ir_form(
            graph,
            tensors,
            export.attention_layouts(config),
            TINY_SYM_MAX,
            TINY_WINDOW,
            int(config.embedding_dim),
        )
        merge = export.assert_soft_merge(graph, int(config.hidden_size))

        assert form["attention_nodes"] == 3
        assert form["full_layers"] == [2]
        assert [spec.name for spec in graph.inputs] == list(export.GRAPH_INPUTS)
        # The text operand is the lookup and its scale, nothing more (ids → embedding → mul).
        assert len(merge["text_chain"]) == 3 and "input_ids" in merge["text_chain"]

    def test_without_soft_rows_the_wrapper_is_the_input_ids_path_bit_for_bit(
        self, tiny_wrapper
    ) -> None:
        torch.manual_seed(2)
        ids = torch.randint(3, 64, (1, 12))
        pool = torch.ones_like(ids, dtype=torch.float32)

        with torch.no_grad():
            ours = tiny_wrapper.text(ids, pool)
            text, _ = export.embed(
                tiny_wrapper.model, ids, pool, export.layer_masks(12, TINY_WINDOW)
            )

        assert torch.equal(ours, text)

    def test_soft_rows_are_upstreams_merge_of_the_composite_model(self, tiny_wrapper) -> None:
        """`EmbeddingGemma2Model.forward`'s merge: the placeholder ids looked up as the pad id,
        then `masked_scatter` of the unscaled soft tokens, then the text tower on the merged
        rows (its per-layer embedding included)."""
        torch.manual_seed(3)
        ids = torch.randint(3, 64, (1, 12))
        pool = torch.ones_like(ids, dtype=torch.float32)
        model = tiny_wrapper.model
        hidden = int(model.config.hidden_size)
        rows = torch.randn(4, hidden) * 3
        soft_embeds, soft_mask = export.soft_inputs(ids, hidden, [(5, rows)])
        placed = ids.clone()
        placed[0, 5:9] = 63  # a placeholder-like id the host may leave under a soft row

        with torch.no_grad():
            ours = tiny_wrapper(placed, pool, soft_embeds, soft_mask)
            scatter = soft_mask.expand(-1, -1, hidden)
            pad = torch.where(soft_mask[..., 0], int(model.config.pad_token_id), placed)
            merged = model.embed_tokens(pad).masked_scatter(scatter, rows)
            hidden_states = model(
                inputs_embeds=merged, attention_mask=export.layer_masks(12, TINY_WINDOW)
            ).last_hidden_state
            upstream = export._l2_normalize(export._masked_mean(hidden_states, pool))
            text = tiny_wrapper.text(ids, pool)

        assert torch.equal(ours, upstream)
        assert float((ours - text).abs().max()) > 1e-3

    def test_the_io_holds_the_four_inputs_without_soft_rows_and_the_text_output(
        self, tiny_wrapper, tmp_path: Path
    ) -> None:
        from safetensors.torch import load_file

        ids = torch.randint(3, 64, (1, TINY_SYM_MAX))
        pool = torch.ones_like(ids, dtype=torch.float32)
        graph, _tensors = export.trace(tiny_wrapper, ids, pool, TINY_SYM_MAX)
        short = ids[:, :5]

        written, outputs = export.write_io(
            tiny_wrapper, graph, [("short", short, pool[:, :5])], tmp_path
        )

        io = load_file(str(tmp_path / written[0]))
        hidden = int(tiny_wrapper.model.config.hidden_size)
        assert sorted(io) == sorted(
            [*(f"input.{name}" for name in export.GRAPH_INPUTS), "output.0"]
        )
        assert io["input.input_ids"].dtype == torch.int32
        assert list(io["input.soft_embeds"].shape) == [1, 5, hidden]
        assert not io["input.soft_embeds"].any()
        assert io["input.soft_mask"].dtype == torch.uint32
        assert list(io["input.soft_mask"].shape) == [1, 5, 1]
        assert not io["input.soft_mask"].to(torch.int64).any()
        with torch.no_grad():
            text, _ = export.embed(
                tiny_wrapper.model, short, pool[:, :5], export.layer_masks(5, TINY_WINDOW)
            )
        assert torch.equal(io["output.0"], text)
        assert torch.equal(outputs["short"], text)

    def test_the_band_form_passes_the_writer(self, tiny_wrapper, tmp_path: Path) -> None:
        """The surgery's result passes the writer's acceptance rules (runtime support and the op
        contracts) and is written without the band constant."""
        ids = torch.randint(3, 64, (1, TINY_SYM_MAX))
        pool = torch.ones_like(ids, dtype=torch.float32)
        graph, tensors = export.trace(tiny_wrapper, ids, pool, TINY_SYM_MAX)
        layouts = export.attention_layouts(tiny_wrapper.model.config)
        width = int(tiny_wrapper.model.config.embedding_dim)
        traced = export.assert_ir_form(graph, tensors, layouts, TINY_SYM_MAX, TINY_WINDOW, width)

        declared, pruned = export.declare_band(graph, layouts, TINY_WINDOW)
        form = export.assert_band_form(
            declared, layouts, TINY_SYM_MAX, TINY_WINDOW, pruned, traced["band_constant"]
        )
        kept = {initializer.tensor for initializer in declared.initializers.values()}
        stored = publish_model(
            tmp_path / "model.krm",
            declared,
            {key: value for key, value in tensors.items() if key in kept},
            provenance=Provenance(license=export.LICENSE),
            graph_name=export.GRAPH_NAME,
        )

        assert form["band_layers"] == [0, 1]
        assert traced["band_constant"] not in stored.initializers
        assert [node.attrs.get("band") for node in _attentions(stored)] == [
            TINY_WINDOW,
            TINY_WINDOW,
            None,
        ]


def _storage_graph(table: str = "bf16", projection: str = "bf16", band: str = "f32") -> IrGraph:
    """A graph with two weight-slot initializers (the token table and a projection) and one that
    is not (a band sliced by T), stored as given."""
    graph = IrGraph(symbols=["T"])
    graph.inputs.append(IrInput(name="input_ids", dtype="i32", shape=[1, "T"]))
    for name, shape, dtype in (
        ("table", [8, 4], table),
        ("projection", [6, 4], projection),
        (BAND, [1, 1, TINY_SYM_MAX, 1], band),
    ):
        graph.initializers[name] = IrInitializer(tensor=name, storage=IrStorage(dtype))
        graph.values[name] = IrValue("f32", shape)
    graph.values["tokens"] = IrValue("f32", [1, "T", 4])
    graph.values["projected"] = IrValue("f32", [1, "T", 6])
    graph.values["band.T"] = IrValue("f32", [1, 1, "T", 1])
    graph.nodes += [
        IrNode(op="embedding", ins=["table", "input_ids"], outs=["tokens"], attrs={}),
        IrNode(op="linear", ins=["tokens", "projection"], outs=["projected"], attrs={}),
        IrNode(
            op=export.SYM_PREFIX_SLICE_OP,
            ins=[BAND],
            outs=["band.T"],
            attrs={"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        ),
    ]
    graph.outputs.append("projected")
    return graph


class TestStorageForm:
    """{@link export.assert_storage_form} — what the writer committed is the storage asked for."""

    def test_bf16_in_every_weight_slot_and_f32_elsewhere_passes(self) -> None:
        assert export.assert_storage_form(_storage_graph(), "bf16") == {"bf16": 2, "f32": 1}

    def test_an_all_f32_graph_passes_as_f32(self) -> None:
        graph = _storage_graph(table="f32", projection="f32")

        assert export.assert_storage_form(graph, "f32") == {"f32": 3}

    def test_a_weight_slot_left_f32_fails_the_bf16_form(self) -> None:
        with pytest.raises(AssertionError, match=r"projection: f32 \(expected bf16\)"):
            export.assert_storage_form(_storage_graph(projection="f32"), "bf16")

    def test_a_weight_in_another_compressed_storage_fails_the_bf16_form(self) -> None:
        with pytest.raises(AssertionError, match=r"table: f16 \(expected bf16\)"):
            export.assert_storage_form(_storage_graph(table="f16"), "bf16")

    def test_bf16_outside_a_weight_slot_fails(self) -> None:
        with pytest.raises(AssertionError, match=rf"{BAND}: bf16 \(expected f32 / i32\)"):
            export.assert_storage_form(_storage_graph(band="bf16"), "bf16")

    def test_bf16_in_the_f32_form_fails(self) -> None:
        with pytest.raises(AssertionError, match="the f32 storage form does not hold"):
            export.assert_storage_form(_storage_graph(), "f32")

    def test_a_bf16_graph_without_a_weight_slot_fails(self) -> None:
        graph = _storage_graph()
        graph.nodes = [node for node in graph.nodes if node.op == export.SYM_PREFIX_SLICE_OP]
        for name in ("table", "projection"):
            graph.initializers[name] = IrInitializer(tensor=name, storage=IrStorage("f32"))

        with pytest.raises(AssertionError, match="no initializer is stored as bf16"):
            export.assert_storage_form(graph, "bf16")

    def test_a_storage_the_script_does_not_write_is_refused(self) -> None:
        with pytest.raises(export.EmbeddingGemma2ExportError, match="not one of"):
            export.assert_storage_form(_storage_graph(), "f16")

    def test_i8_in_every_rounded_weight_slot_and_f32_elsewhere_passes(self) -> None:
        graph = _storage_graph(table="i8", projection="i8")

        assert export.assert_storage_form(graph, "i8", ("table", "projection")) == {
            "f32": 1,
            "i8": 2,
        }

    def test_a_weight_slot_left_f32_fails_the_i8_form(self) -> None:
        graph = _storage_graph(table="i8", projection="f32")

        with pytest.raises(AssertionError, match=r"projection: f32 \(expected i8\)"):
            export.assert_storage_form(graph, "i8", ("table", "projection"))

    def test_bf16_in_a_weight_slot_fails_the_i8_form(self) -> None:
        graph = _storage_graph(table="bf16", projection="i8")

        with pytest.raises(AssertionError, match=r"table: bf16 \(expected i8\)"):
            export.assert_storage_form(graph, "i8", ("table", "projection"))

    def test_i8_outside_a_weight_slot_fails(self) -> None:
        graph = _storage_graph(table="i8", projection="i8", band="i8")

        with pytest.raises(AssertionError, match=rf"{BAND}: i8 \(expected f32 / i32\)"):
            export.assert_storage_form(graph, "i8", ("table", "projection"))

    def test_a_rounded_weight_outside_a_weight_slot_fails_the_i8_form(self) -> None:
        """The band is rounded but stays f32 — the preset would hold rounding it does not store."""
        graph = _storage_graph(table="i8", projection="i8")

        with pytest.raises(AssertionError, match=rf"rounded but outside a slot \['{BAND}'\]"):
            export.assert_storage_form(graph, "i8", ("table", "projection", BAND))

    def test_a_weight_slot_the_rounding_missed_fails_the_i8_form(self) -> None:
        graph = _storage_graph(table="i8", projection="i8")

        with pytest.raises(AssertionError, match=r"in a slot but not rounded \['projection'\]"):
            export.assert_storage_form(graph, "i8", ("table",))

    def test_the_i8_form_needs_the_rounded_weights(self) -> None:
        graph = _storage_graph(table="i8", projection="i8")

        with pytest.raises(export.EmbeddingGemma2ExportError, match="takes the rounded weights"):
            export.assert_storage_form(graph, "i8")

    def test_the_other_forms_take_no_rounded_weights(self) -> None:
        with pytest.raises(export.EmbeddingGemma2ExportError, match="takes no rounded weights"):
            export.assert_storage_form(_storage_graph(), "bf16", ("table",))


class TestStorageOption:
    """`--dtype` picks the storage and, without `--out`, the series of that storage."""

    @pytest.fixture
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str]]:
        recorded: list[tuple[Path, str]] = []

        def export_series(directory: Path, out_dir: Path, *, dtype: str) -> dict[str, str]:
            recorded.append((out_dir, dtype))
            return {"dir": str(out_dir)}

        monkeypatch.setattr(export, "export_series", export_series)
        return recorded

    def test_the_default_is_the_f32_series(self, calls) -> None:
        export.main([])

        assert calls == [(export.DEFAULT_OUT_DIRS["f32"], "f32")]
        assert export.DEFAULT_OUT_DIRS["f32"].name == export.TEXT_SERIES

    def test_bf16_writes_its_own_series(self, calls) -> None:
        export.main(["--dtype", "bf16"])

        assert calls == [(export.DEFAULT_OUT_DIRS["bf16"], "bf16")]
        assert export.DEFAULT_OUT_DIRS["bf16"].name == f"{export.TEXT_SERIES}-bf16"

    def test_i8_writes_its_own_series(self, calls) -> None:
        export.main(["--dtype", "i8"])

        assert calls == [(export.DEFAULT_OUT_DIRS["i8"], "i8")]
        assert export.DEFAULT_OUT_DIRS["i8"].name == f"{export.TEXT_SERIES}-i8"

    def test_out_overrides_the_series(self, calls, tmp_path: Path) -> None:
        export.main(["--dtype", "bf16", "--out", str(tmp_path)])

        assert calls == [(tmp_path, "bf16")]

    def test_every_storage_has_a_series_of_its_own(self) -> None:
        assert sorted(export.DEFAULT_OUT_DIRS) == sorted(export.WEIGHT_DTYPES)
        assert len(set(export.DEFAULT_OUT_DIRS.values())) == len(export.WEIGHT_DTYPES)

    def test_another_storage_is_refused_by_the_parser(self, calls) -> None:
        with pytest.raises(SystemExit):
            export.main(["--dtype", "f16"])
        assert calls == []

    def test_the_export_refuses_another_storage_before_reading_anything(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(export.EmbeddingGemma2ExportError, match="not one of"):
            export.export_series(tmp_path / "missing", tmp_path / "out", dtype="f16")


class TestTinyModelBf16:
    """The bf16 write on the real export path (a tiny random text tower — transformers at the
    pinned release)."""

    def _export(self, wrapper) -> tuple[IrGraph, dict[str, torch.Tensor]]:
        ids = torch.randint(3, 64, (1, TINY_SYM_MAX))
        pool = torch.ones_like(ids, dtype=torch.float32)
        return export.trace(wrapper, ids, pool, TINY_SYM_MAX)

    def test_weights_bf16_holds_are_stored_as_bf16_in_every_weight_slot(
        self, tiny_wrapper, tmp_path: Path
    ) -> None:
        round_weights_to_bf16(tiny_wrapper)
        graph, tensors = self._export(tiny_wrapper)

        stored = publish_model(
            tmp_path / "model.krm",
            graph,
            tensors,
            provenance=Provenance(license=export.LICENSE),
            graph_name=export.GRAPH_NAME,
            weight_dtype="bf16",
        )
        counts = export.assert_storage_form(stored, "bf16")

        # The 3 layers' q / k / v / o and the 3 MLP projections, the per-layer input projections,
        # the final projection and the token table — at least one per layer plus the table.
        assert counts["bf16"] > 3
        assert counts["bf16"] == len(eligible_compressed_initializers(stored))

    def test_a_weight_bf16_cannot_hold_fails_the_write(self, tiny_wrapper, tmp_path: Path) -> None:
        graph, tensors = self._export(tiny_wrapper)

        with pytest.raises(EmitError, match="bf16 cannot represent"):
            publish_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                provenance=Provenance(license=export.LICENSE),
                graph_name=export.GRAPH_NAME,
                weight_dtype="bf16",
            )
        assert not list(tmp_path.iterdir())


class _ToyNorm(torch.nn.Module):
    """A norm-style module: a `weight` the rounding must leave alone (not a weight-slot type)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(width))


class _ToyTower(torch.nn.Module):
    """The shape of the text tower as the rounding sees it: the token table, linears with and
    without a bias, and a norm."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(6, 4)
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 3)
        self.norm = _ToyNorm(4)


class _ToyWrapper(torch.nn.Module):
    """`EmbeddingWrapper`'s layout: the tower as `model`, so the keys are
    `model.<module>.weight`."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyTower()


@pytest.fixture
def toy_wrapper() -> _ToyWrapper:
    torch.manual_seed(0)
    return _ToyWrapper()


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in module.state_dict().items()}


class TestI8RoundingOnAToyTower:
    """{@link export.round_weights_to_i8} on a module shaped like the wrapper (no transformers)."""

    def test_it_rounds_exactly_the_linears_and_the_token_table(self, toy_wrapper) -> None:
        before = _state(toy_wrapper)

        report = export.round_weights_to_i8(toy_wrapper)

        assert set(report.scales) == {
            "model.embed_tokens.weight",
            "model.q_proj.weight",
            "model.down_proj.weight",
        }
        after = toy_wrapper.state_dict()
        changed = {key for key in before if not torch.equal(before[key], after[key])}
        assert changed == set(report.scales)
        assert torch.equal(after["model.down_proj.bias"], before["model.down_proj.bias"])
        assert torch.equal(after["model.norm.weight"], before["model.norm.weight"])

    def test_each_weight_has_one_f32_scale_per_row(self, toy_wrapper) -> None:
        report = export.round_weights_to_i8(toy_wrapper)

        after = toy_wrapper.state_dict()
        assert {key: list(scale.shape) for key, scale in report.scales.items()} == {
            "model.embed_tokens.weight": [6, 1],
            "model.q_proj.weight": [4, 1],
            "model.down_proj.weight": [3, 1],
        }
        for key, scale in report.scales.items():
            assert scale.dtype == torch.float32
            levels = quantize_to_int8(after[key], scale)
            assert torch.equal(levels.to(torch.float32) * scale, after[key])

    def test_a_second_table_is_refused_and_nothing_is_rounded(self, toy_wrapper) -> None:
        toy_wrapper.model.extra_table = torch.nn.Embedding(4, 2)
        before = _state(toy_wrapper)

        with pytest.raises(export.EmbeddingGemma2ExportError, match="not the token table alone"):
            export.round_weights_to_i8(toy_wrapper)

        after = toy_wrapper.state_dict()
        assert all(torch.equal(before[key], after[key]) for key in before)

    def test_another_quantizable_module_type_is_refused_and_nothing_is_rounded(
        self, toy_wrapper
    ) -> None:
        """A conv is a target of the core's rounding but none of the weights the bf16 series
        stores as bf16."""
        toy_wrapper.model.extra_conv = torch.nn.Conv1d(2, 2, 1)
        before = _state(toy_wrapper)

        with pytest.raises(
            export.EmbeddingGemma2ExportError, match=r"extra \['model\.extra_conv\.weight'\]"
        ):
            export.round_weights_to_i8(toy_wrapper)

        after = toy_wrapper.state_dict()
        assert all(torch.equal(before[key], after[key]) for key in before)


class TestTinyModelI8:
    """The i8 rounding and write on the real export path (a tiny random text tower — transformers at
    the pinned release)."""

    def _export(self, wrapper) -> tuple[IrGraph, dict[str, torch.Tensor]]:
        ids = torch.randint(3, 64, (1, TINY_SYM_MAX))
        pool = torch.ones_like(ids, dtype=torch.float32)
        return export.trace(wrapper, ids, pool, TINY_SYM_MAX)

    def test_it_rounds_every_linear_and_the_token_table_and_nothing_else(
        self, tiny_wrapper
    ) -> None:
        linears = {
            f"{name}.weight"
            for name, module in tiny_wrapper.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        before = {key: value.clone() for key, value in tiny_wrapper.state_dict().items()}

        report = export.round_weights_to_i8(tiny_wrapper)

        assert set(report.scales) == linears | {f"model.{export.TABLE_MODULE}.weight"}
        after = tiny_wrapper.state_dict()
        changed = {key for key in before if not torch.equal(before[key], after[key])}
        assert changed == set(report.scales)
        for key, scale in report.scales.items():
            # One F32 scale per output row (per vocabulary row for the table), and the rounded
            # values are an int8 times it, bit for bit (what the writer checks before storing).
            assert scale.dtype == torch.float32
            assert list(scale.shape) == [after[key].shape[0], 1]
            levels = quantize_to_int8(after[key], scale)
            assert torch.equal(levels.to(torch.float32) * scale, after[key])

    def test_another_table_is_refused(self, tiny_wrapper) -> None:
        tiny_wrapper.model.extra_table = torch.nn.Embedding(4, 2)

        with pytest.raises(export.EmbeddingGemma2ExportError, match="not the token table alone"):
            export.round_weights_to_i8(tiny_wrapper)

    def test_another_weight_slot_module_is_refused(self, tiny_wrapper) -> None:
        """A conv is a target of the core's rounding but none of the weights the bf16 series
        stores as bf16."""
        tiny_wrapper.model.extra_conv = torch.nn.Conv1d(2, 2, 1)

        with pytest.raises(export.EmbeddingGemma2ExportError, match=r"extra \['model\.extra_conv"):
            export.round_weights_to_i8(tiny_wrapper)

    def test_the_rounded_model_is_stored_as_i8_in_exactly_its_rounded_weight_slots(
        self, tiny_wrapper, tmp_path: Path
    ) -> None:
        report = export.round_weights_to_i8(tiny_wrapper)
        graph, tensors = self._export(tiny_wrapper)

        stored = publish_model(
            tmp_path / "model.krm",
            graph,
            tensors,
            provenance=Provenance(license=export.LICENSE),
            graph_name=export.GRAPH_NAME,
            weight_dtype="i8",
            weight_scales=report.scales,
        )
        counts = export.assert_storage_form(stored, "i8", tuple(report.scales))

        assert counts["i8"] == len(report.scales)
        assert counts["i8"] == len(eligible_compressed_initializers(stored))
        assert set(counts) == {"f32", "i8"}

    def test_an_i8_write_without_the_rounding_fails(self, tiny_wrapper, tmp_path: Path) -> None:
        graph, tensors = self._export(tiny_wrapper)

        with pytest.raises(EmitError, match="no per-channel scale"):
            publish_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                provenance=Provenance(license=export.LICENSE),
                graph_name=export.GRAPH_NAME,
                weight_dtype="i8",
            )
        assert not list(tmp_path.iterdir())
