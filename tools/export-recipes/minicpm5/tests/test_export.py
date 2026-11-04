"""Script-level guarantees of `minicpm5/export.py` (the part that needs no real weights).

Emitting with real weights is manual (the same rule as the `deberta` / `embeddinggemma` tests).
What is pinned here is only the rules whose breakage gives a **false PASS**:

- `gqa_sdpa_attention` returns **the same numbers** as the `repeat_kv` materialized version (the
  very justification of the detour)
- no mask / dropout / `is_causal` fail loudly (no path left where causality silently disappears)
- the additive causal mask **commutes with the prefix** (the condition for ADR 0010's folding)
- `assert_ir_form` **actually detects** the repeat_kv materialized form and a mask left as a graph
  input (the subject is a tiny real model exported with plain `sdpa` — the check against a vacuous
  pass)
- the sanity check fails on both an expected-continuation mismatch and constant output (either one
  alone would be vacuous)

Only the cases that need transformers SKIP via `importorskip` (the default-sync CI does not install
the model dependencies — the 2-job layout of ADR 0065).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.export import Dim
from upstream_fixture import OTHER_REVISION, write_snapshot

from _shared.upstream import UpstreamProvenanceError
from karume.container import Provenance
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_to_file
from karume.rope import assert_rope_lifted
from minicpm5 import export as mc

#: Shape of the tiny subject (head_dim is set to **a value other than** hidden/heads = 8 — to set
#: the same trap as the real MiniCPM5-1B's `head_dim: 128` ≠ 1536/16 = 96).
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
LAYERS = 2
HIDDEN = 32
VOCAB = 64
TINY_SYM_MAX = 32


def _tiny_config(attn_implementation: str):
    transformers = pytest.importorskip("transformers")
    return transformers.LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=48,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
        max_position_embeddings=256,
        rope_theta=5_000_000.0,
        tie_word_embeddings=False,
        attn_implementation=attn_implementation,
    )


def _tiny_checkpoint(tmp_path: Path) -> Path:
    """Write a tiny `LlamaForCausalLM` as a real checkpoint (for the real path of
    `load_wrapper`)."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(_tiny_config("sdpa")).to(torch.float32).eval()
    model.save_pretrained(tmp_path)
    return tmp_path


def _export_tiny(tmp_path: Path, wrapper) -> IrGraph:
    ids = torch.randint(0, VOCAB, (1, 7), dtype=torch.int64)
    return export_to_file(
        wrapper,
        (ids,),
        tmp_path / mc.MODEL_FILE,
        provenance=Provenance(license="fixture"),
        graph_name="tiny",
        dynamic_shapes=({1: Dim("T", min=2, max=TINY_SYM_MAX)},),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )


@pytest.fixture
def tiny_gqa(tmp_path):
    """Export the tiny wrapper built by the script's `load_wrapper` → `(graph, config)`."""
    wrapper = mc.load_wrapper(_tiny_checkpoint(tmp_path / "ckpt"))
    return _export_tiny(tmp_path, wrapper), wrapper.model.config


@pytest.fixture
def tiny_plain_sdpa(tmp_path):
    """Export the same model loaded with **plain `sdpa`** → `(graph, config)` (the repeat_kv
    materialized form)."""
    transformers = pytest.importorskip("transformers")
    model = transformers.LlamaForCausalLM.from_pretrained(
        _tiny_checkpoint(tmp_path / "ckpt"), dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    assert_rope_lifted(model, "tiny")
    wrapper = mc.CausalLmWrapper(model).eval()
    return _export_tiny(tmp_path, wrapper), model.config


class TestGqaSdpaAttentionMatchesTheMaterializedForm:
    """Justification of the detour: `enable_gqa=True` MUST give the same numbers as calling after
    `repeat_kv` (if this broke, the result would be "the GQA form was kept but the numbers
    differ")."""

    @staticmethod
    def _reference(query, key, value, mask, scaling):
        repeats = query.shape[1] // key.shape[1]
        expanded = [
            tensor.repeat_interleave(repeats, dim=1).contiguous() for tensor in (key, value)
        ]
        output = torch.nn.functional.scaled_dot_product_attention(
            query, *expanded, attn_mask=mask, dropout_p=0.0, scale=scaling, is_causal=False
        )
        return output.transpose(1, 2).contiguous()

    def test_the_outputs_agree(self):
        torch.manual_seed(0)
        query = torch.randn(1, 8, 5, HEAD_DIM)
        key = torch.randn(1, 2, 5, HEAD_DIM)
        value = torch.randn(1, 2, 5, HEAD_DIM)
        mask = mc.additive_causal_mask(5)
        scaling = HEAD_DIM**-0.5

        actual, weights = mc.gqa_sdpa_attention(
            torch.nn.Identity(), query, key, value, mask, scaling=scaling
        )

        assert weights is None
        expected = self._reference(query, key, value, mask, scaling)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_a_wrong_group_mapping_would_be_visible(self):
        """Fault injection: swapping the kv heads breaks the comparison above (evidence that it is
        not vacuous)."""
        torch.manual_seed(0)
        query = torch.randn(1, 8, 5, HEAD_DIM)
        key = torch.randn(1, 2, 5, HEAD_DIM)
        value = torch.randn(1, 2, 5, HEAD_DIM)
        mask = mc.additive_causal_mask(5)
        scaling = HEAD_DIM**-0.5

        actual, _ = mc.gqa_sdpa_attention(
            torch.nn.Identity(), query, key.flip(1), value, mask, scaling=scaling
        )

        expected = self._reference(query, key, value, mask, scaling)
        assert not torch.allclose(actual, expected, atol=1e-6)


class TestGqaSdpaAttentionGates:
    def test_a_missing_mask_fails_loudly(self):
        """MUST: passing a missing mask through silently emits non-causal attention."""
        query = torch.zeros(1, 4, 3, HEAD_DIM)
        key = torch.zeros(1, 2, 3, HEAD_DIM)

        with pytest.raises(ValueError, match="without attn_mask"):
            mc.gqa_sdpa_attention(torch.nn.Identity(), query, key, key, None)

    def test_dropout_fails_loudly(self):
        query = torch.zeros(1, 4, 3, HEAD_DIM)
        key = torch.zeros(1, 2, 3, HEAD_DIM)
        mask = mc.additive_causal_mask(3)

        with pytest.raises(ValueError, match="dropout"):
            mc.gqa_sdpa_attention(torch.nn.Identity(), query, key, key, mask, dropout=0.1)

    @pytest.mark.parametrize("name", ["is_causal", "position_bias"])
    def test_semantics_changing_kwargs_fail_loudly(self, name):
        """MUST: if ignored, nothing would show that the numerical path differs from eager."""
        query = torch.zeros(1, 4, 3, HEAD_DIM)
        key = torch.zeros(1, 2, 3, HEAD_DIM)
        mask = mc.additive_causal_mask(3)

        with pytest.raises(ValueError, match=name):
            mc.gqa_sdpa_attention(
                torch.nn.Identity(), query, key, key, mask, **{name: torch.ones(1)}
            )


class TestAdditiveCausalMask:
    def test_the_band_is_zero_below_the_diagonal_and_neg_inf_above(self):
        mask = mc.additive_causal_mask(4)

        assert tuple(mask.shape) == (1, 1, 4, 4)
        assert mask.dtype is torch.float32
        assert torch.equal(mask[0, 0].tril(), torch.zeros(4, 4))
        assert bool(torch.isneginf(mask[0, 0]).sum() == 6)

    def test_the_prefix_of_a_longer_mask_is_the_shorter_mask(self):
        """MUST: commutes with the prefix (ADR 0010) — if this broke, the Tmax folding would
        silently give other values."""
        long_mask = mc.additive_causal_mask(TINY_SYM_MAX)

        prefix = long_mask[:, :, :5, :5]

        assert torch.equal(prefix, mc.additive_causal_mask(5))


def _ir_graph(heads: int, kv_heads: int, layers: int = LAYERS) -> IrGraph:
    """Minimal IR with `layers` attention nodes (the mask is a Tmax constant + sym_prefix_slice)."""
    graph = IrGraph(symbols=["T"])
    graph.inputs.append(IrInput(name=mc.INPUT_IDS, dtype="i32", shape=[1, "T"]))
    graph.initializers["mask_const"] = IrInitializer(
        tensor="const.mask", storage=IrStorage(dtype="f32")
    )
    graph.values["mask_const"] = IrValue(dtype="f32", shape=[1, 1, TINY_SYM_MAX, TINY_SYM_MAX])
    graph.values["mask"] = IrValue(dtype="f32", shape=[1, 1, "T", "T"])
    graph.nodes.append(
        IrNode(
            op=mc.SYM_PREFIX_SLICE_OP,
            ins=["mask_const"],
            outs=["mask"],
            attrs={"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )
    for layer in range(layers):
        for slot, count in (("q", heads), ("k", kv_heads), ("v", kv_heads)):
            graph.values[f"{slot}{layer}"] = IrValue(dtype="f32", shape=[1, count, "T", HEAD_DIM])
        graph.values[f"attn{layer}"] = IrValue(dtype="f32", shape=[1, heads, "T", HEAD_DIM])
        graph.nodes.append(
            IrNode(
                op=mc.ATTENTION_OP,
                ins=[f"q{layer}", f"k{layer}", f"v{layer}", "mask"],
                outs=[f"attn{layer}"],
                attrs={"scale": 0.5},
            )
        )
    graph.outputs.append(f"attn{layers - 1}")
    return graph


TINY_IR_CONFIG = SimpleNamespace(
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    num_hidden_layers=LAYERS,
)


class TestAssertIrForm:
    def test_the_grouped_shape_passes(self):
        form = mc.assert_ir_form(_ir_graph(HEADS, KV_HEADS), TINY_IR_CONFIG, TINY_SYM_MAX)

        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]
        assert form["attention_nodes"] == LAYERS
        assert form["mask_constant"] == "mask_const"

    def test_a_materialized_kv_shape_is_rejected(self):
        """MUST: the form where Hkv turned into H is an asset where "the numbers agree but the
        acceptance has lost its meaning"."""
        with pytest.raises(AssertionError, match="repeat_kv"):
            mc.assert_ir_form(_ir_graph(HEADS, HEADS), TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _ir_graph(HEADS, KV_HEADS)
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "T", "T"]))

        with pytest.raises(AssertionError, match="graph inputs"):
            mc.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_mask_constant_of_the_wrong_extent_is_rejected(self):
        """A constant shorter than Tmax is the form that "silently falls short when T grows to the
        upper bound"."""
        graph = _ir_graph(HEADS, KV_HEADS)
        graph.values["mask_const"] = IrValue(dtype="f32", shape=[1, 1, 8, 8])

        with pytest.raises(AssertionError, match="Tmax shape"):
            mc.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_missing_layer_is_rejected(self):
        with pytest.raises(AssertionError, match=r"does not match the \d+ layers"):
            mc.assert_ir_form(_ir_graph(HEADS, KV_HEADS, layers=1), TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_derived_head_dim_is_rejected(self):
        """MUST: D is compared with `config.head_dim` (a value derived from hidden/heads does not
        pass)."""
        derived = SimpleNamespace(
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HIDDEN // HEADS,
            num_hidden_layers=LAYERS,
        )

        with pytest.raises(AssertionError, match=r"config\.head_dim"):
            mc.assert_ir_form(_ir_graph(HEADS, KV_HEADS), derived, TINY_SYM_MAX)


class TestExportedForm:
    """Actually export a tiny real model and look at the form (needs transformers)."""

    def test_the_graph_keeps_the_grouped_shape_and_folds_the_mask(self, tiny_gqa):
        graph, config = tiny_gqa

        form = mc.assert_ir_form(graph, config, TINY_SYM_MAX)

        assert [spec.name for spec in graph.inputs] == [mc.INPUT_IDS]
        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]
        assert form["head_dim"] == HEAD_DIM
        assert mc.SYM_PREFIX_SLICE_OP in graph.required_ops

    def test_plain_sdpa_materializes_repeat_kv_and_is_rejected(self, tiny_plain_sdpa):
        """The check against a vacuous pass: upstream's default path really collapses Hkv (= the
        green above is the result of the check working)."""
        graph, config = tiny_plain_sdpa

        with pytest.raises(AssertionError, match="repeat_kv"):
            mc.assert_ir_form(graph, config, TINY_SYM_MAX)


#: The subject of `_sanity` (case name → expected token id) and the display names.
SANITY_EXPECTED = {"a": 10, "b": 20}
SANITY_LABELS = {10: "Ġx", 20: "Ġy", 30: "Ġz"}


class TestSanity:
    def test_matching_greedy_tokens_pass(self):
        result = mc._sanity({"a": 10, "b": 20}, SANITY_EXPECTED, SANITY_LABELS)

        assert result == {"a": "10(Ġx)", "b": "20(Ġy)"}

    def test_a_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="differs from the expected continuation"):
            mc._sanity({"a": 30, "b": 20}, SANITY_EXPECTED, SANITY_LABELS)

    def test_a_constant_output_fails_loudly(self):
        """MUST: the detection line that remains even if the expectation table is loosened (both
        cases have the same top-1 = constant output)."""
        with pytest.raises(AssertionError, match="constant output"):
            mc._sanity({"a": 10, "b": 10}, {"a": 10, "b": 10}, SANITY_LABELS)

    def test_a_case_set_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="case sets differ"):
            mc._sanity({"a": 10}, SANITY_EXPECTED, SANITY_LABELS)


class TestGreedyTokens:
    def test_the_last_position_is_the_one_that_is_read(self):
        """MUST: the last position. Reading another position would compare a quantity that is not
        "the prediction of the continuation"."""
        logits = torch.zeros(1, 3, 5)
        logits[0, 0, 4] = 1.0
        logits[0, -1, 2] = 1.0

        assert mc.greedy_tokens({"case": logits}) == {"case": 2}


class _StubTokenizer:
    """A minimal tokenizer with only `encode(...).ids` (for the single-token check of the expected
    continuations)."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def encode(self, text: str, add_special_tokens: bool = True):
        """Matches only the calling convention of `tokenizers.Tokenizer.encode` (the arguments are
        unused)."""
        return SimpleNamespace(ids=list(self._ids))


class TestExpectedTokenIds:
    def test_a_single_token_continuation_is_accepted(self):
        assert set(mc.expected_token_ids(_StubTokenizer([7])).values()) == {7}

    def test_a_multi_token_continuation_fails_loudly(self):
        """MUST: with several tokens "top-1 agreement" has no defined meaning (do not silently look
        at the first one only)."""
        with pytest.raises(AssertionError, match="single token"):
            mc.expected_token_ids(_StubTokenizer([7, 8]))


class TestGoldenCases:
    def test_every_case_has_a_greedy_expectation(self):
        """MUST: no case without an expectation (the sanity check does not look at only some
        cases)."""
        assert {name for name, _ in mc.GOLDEN_CASES} == set(mc.GREEDY_EXPECTATIONS)

    def test_the_expectations_are_not_all_the_same_token(self):
        """The condition that keeps the constant-output detection line from being vacuous (with a
        single kind of expectation, the latter half of `_sanity` is dead)."""
        assert len(set(mc.GREEDY_EXPECTATIONS.values())) > 1


class TestUpstreamProvenance:
    """The container's revision is derived from the download record of `--model-dir` (not
    hard-coded)."""

    def test_the_revision_comes_from_the_download_record(self, tmp_path) -> None:
        write_snapshot(tmp_path, license="ignored", revision=OTHER_REVISION)

        assert mc.upstream_provenance(tmp_path) == Provenance(
            license=mc.LICENSE, upstream_revision=OTHER_REVISION
        )

    def test_a_checkpoint_without_the_record_fails_loudly(self, tmp_path) -> None:
        with pytest.raises(UpstreamProvenanceError, match="metadata is missing"):
            mc.upstream_provenance(tmp_path)
