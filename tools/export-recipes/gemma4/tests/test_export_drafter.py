"""MTP drafter 台本（`gemma4/export_drafter.py`）と、その配布の門の約束事。

実重みの emit は手動（chunk 系列 3 本のテストと同じ規律）。ここで固定するのは、壊れると
**偽 PASS** になる側の規律だけ:

- 量子化の席が **i8 の 1 周だけ**で、`nn.Linear` 全部 + 共有主表ちょうどに掛かること
  （i4 が混ざっても tied 実体が二重に丸まっても、壊れるのは値だけ）
- golden ケースの素材がリポ内の文書で、**段落境界**で切れること（token 途中で切ると継続が
  数語ループへ退化して golden が痩せる）
- 貸し手（製品コンテナ）から**構造で**引く 2 つ — external スロットの実形と共有テンソルキー
- tiny な実モデルを export → 手術 → 共有宣言 → コンテナ検証まで通し、
  {@link drafter.assert_ir_form_drafter} が借り手の形（入力 6 本 / 出力 k 本 / 全スロット
  external / append 0 本 / readonly の本数 / 共有 1 本 / 記号 1 本）を**実際に見る**こと
- 配布の門 {@link gemma4.distribution.assert_gemma4_drafter_graph} が、借り手と貸し手の
  噛み合わせ（スロット名と実形・共有の指し先・容量記号）の破れを実際に落とすこと

transformers を要するケースだけ `importorskip` で SKIP する（ADR 0065 の 2 job 構成）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gemma4 import distribution as gemma4_distribution
from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_drafter as drafter
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from karume.dist import DistError
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrState, IrStorage, IrValue
from karume.pipeline import publish_model
from karume.verify import parse_ir_graph

#: tiny な drafter の寸法（実物は hidden 256 / 層 4 / heads 4 / head_dim 256|512 /
#: backbone 1536 / 語彙 262144）。**実物と違う数**にする — 寸法を焼き込んでいれば落ちる。
TINY_VOCAB = 64
TINY_HIDDEN = 8
TINY_BACKBONE = 12
TINY_HEADS = 2
TINY_KV_HEADS = 1
TINY_SLIDING_DEPTH = 4
TINY_FULL_DEPTH = 8
TINY_WINDOW = 4
TINY_LAYER_TYPES = (SLIDING_ATTENTION, FULL_ATTENTION)
#: 段数は**実物と同じ k**（配布の門 `GEMMA4_DRAFT_STEPS` が本数ちょうどを見る）。
TINY_STEPS = drafter.DRAFT_STEPS

#: 貸し手のスロット（実物は l13 / l14）。容量は sliding が実数・full が記号。
TINY_SLIDING_CAPACITY = 6
TINY_SHARED_TENSOR = "lender.embed.weight"


def _tiny_lender() -> drafter.LenderSlots:
    """貸し手のスロット実形と共有テンソルキー（製品コンテナから読んだつもりの値）。"""
    return drafter.LenderSlots(
        shapes={
            "l0.k": [1, TINY_KV_HEADS, TINY_SLIDING_CAPACITY, TINY_SLIDING_DEPTH],
            "l0.v": [1, TINY_KV_HEADS, TINY_SLIDING_CAPACITY, TINY_SLIDING_DEPTH],
            "l1.k": [1, TINY_KV_HEADS, decode.CAPACITY_SYMBOL, TINY_FULL_DEPTH],
            "l1.v": [1, TINY_KV_HEADS, decode.CAPACITY_SYMBOL, TINY_FULL_DEPTH],
        },
        slots={SLIDING_ATTENTION: ("l0.k", "l0.v"), FULL_ATTENTION: ("l1.k", "l1.v")},
        shared_tensor=TINY_SHARED_TENSOR,
        shared_initializer="lender_embed",
    )


def _tiny_target_config() -> SimpleNamespace:
    """共有する主表の形を決める上流 text config（読むのは 2 欄だけ）。"""
    return SimpleNamespace(vocab_size=TINY_VOCAB, hidden_size=TINY_BACKBONE)


def _toy_wrapper() -> torch.nn.Module:
    """{@link drafter.quantize_wrapper} が見る**モジュール構成だけ**を模した器。

    実物のラッパを組むには transformers と HF チェックポイントが要るので、席の規律
    （`nn.Linear` 全部 + 共有主表を丸め、norm は触らない）はこの器で固定する。FQN の綴りは
    実物と同じ形（`target_embed` / `drafter.<何か>`）にする。
    """
    torch.manual_seed(0)
    inner = torch.nn.Module()
    inner.lm_head = torch.nn.Linear(4, 6, bias=False)
    inner.pre_projection = torch.nn.Linear(8, 4, bias=False)
    inner.norm = torch.nn.LayerNorm(4)
    wrapper = torch.nn.Module()
    wrapper.target_embed = torch.nn.Embedding(6, 8)
    wrapper.drafter = inner
    return wrapper


class TestTheQuantizationSeats:
    """丸めるのは **i8 の 1 周だけ**（drafter に i4 席は無い — 2026-09-08 の受理率実測）。"""

    def test_every_linear_and_the_shared_table_are_the_seats(self):
        assert drafter.int8_seats(_toy_wrapper()) == {
            f"{drafter.TARGET_EMBED_MODULE}.weight",
            "drafter.lm_head.weight",
            "drafter.pre_projection.weight",
        }

    def test_it_rounds_exactly_those_seats_and_leaves_the_norm_alone(self):
        """norm は `fake_quant_int8` の対象型に無いので 1 ビットも動かない（f32 席）。"""
        wrapper = _toy_wrapper()
        before = wrapper.drafter.norm.weight.detach().clone()

        report, scales = drafter.quantize_wrapper(wrapper)

        assert set(scales) == drafter.int8_seats(wrapper)
        assert report.modules == 3
        assert torch.equal(wrapper.drafter.norm.weight, before)

    def test_a_resurrected_tied_embedding_is_rejected(self):
        """`drafter.model.embed_tokens` が生き返った世代 — 同じ実体が 1 周で 2 度丸まる。

        `load_drafter` の `del` が落ちた日に、scale 台帳は後勝ちの 1 本しか残らないのに重みは
        2 度丸まる形になる（宣言した scale で戻らない重みが配布形に載る）。
        """
        wrapper = _toy_wrapper()
        tied = torch.nn.Embedding(6, 4)
        tied.weight = wrapper.drafter.lm_head.weight
        wrapper.drafter.embed_tokens = tied

        with pytest.raises(AssertionError, match="余剰"):
            drafter.quantize_wrapper(wrapper)


class TestTheCaseMaterial:
    """golden ケースの素材（リポ内の文書を段落境界で切る）。"""

    def test_text_under_the_limit_is_kept_whole(self):
        assert drafter.truncate_paragraph("a\n\nb\n", 100) == "a\n\nb"

    def test_it_cuts_at_the_last_paragraph_break_before_the_limit(self):
        text = "one\n\ntwo\n\nthree and more words"

        assert drafter.truncate_paragraph(text, len("one\n\ntwo\n\n") + 2) == "one\n\ntwo"

    def test_a_document_without_a_paragraph_break_is_rejected(self):
        """段落境界が無い素材は「途中で切れた prompt」になるので受けない。"""
        with pytest.raises(AssertionError, match="段落境界が無い"):
            drafter.truncate_paragraph("x" * 100, 10)

    def test_every_declared_document_exists_and_has_a_break(self):
        """宣言した素材がリポに在り、上限の手前に段落境界を持つこと。"""
        for name, relative, limit in drafter.CASE_DOCUMENTS:
            path = drafter.REPO_ROOT / relative
            assert path.is_file(), name
            assert drafter.truncate_paragraph(path.read_text(encoding="utf-8"), limit)


class TestTheSpellingsMirrorTheDistribution:
    """焼く側と配る側で綴りが割れないこと（片方だけ動くと束ねられない / 門が空振りする）。"""

    def test_the_graph_input_names_match(self):
        assert list(drafter.DRAFTER_INPUTS) == list(gemma4_distribution.GEMMA4_DRAFTER_GRAPH_INPUTS)

    def test_the_draft_step_count_matches(self):
        assert drafter.DRAFT_STEPS == gemma4_distribution.GEMMA4_DRAFT_STEPS

    def test_the_series_directory_name_matches(self):
        assert drafter.DEFAULT_OUT_DIR.name == gemma4_distribution.gemma4_series_name(
            gemma4_distribution.GEMMA4_DEFAULT_MODEL, gemma4_distribution.GEMMA4_DRAFTER_SUFFIX
        )

    def test_the_rope_inputs_come_from_the_shared_spelling(self):
        """RoPE 4 本は decode 系列と同じ 1 箇所から組む（別々に綴らない）。"""
        assert list(drafter.DRAFTER_INPUTS[2:]) == list(decode.ROPE_INPUTS)


def _lender_graph() -> IrGraph:
    """貸し手の製品コンテナに相当する最小グラフ（states 4 本 + 主表の embedding）。"""
    return IrGraph(
        symbols=[decode.CAPACITY_SYMBOL],
        inputs=[IrInput(name="tok", dtype="i32", shape=[1, 1])],
        outputs=["e"],
        initializers={
            "lender_embed": IrInitializer(
                tensor=TINY_SHARED_TENSOR, storage=IrStorage(dtype="i8", scale="s")
            )
        },
        values={
            "lender_embed": IrValue(dtype="f32", shape=[TINY_VOCAB, TINY_BACKBONE]),
            "e": IrValue(dtype="f32", shape=[1, 1, TINY_BACKBONE]),
        },
        states={
            name: IrState(dtype="f32", shape=shape) for name, shape in _tiny_lender().shapes.items()
        },
        nodes=[
            IrNode(
                op="embedding",
                ins=["lender_embed", "tok"],
                outs=["e"],
                attrs={"padding_idx": -1},
            )
        ],
    )


class TestReadingTheLender:
    """貸し手から**構造で**引く（綴りを写経しない）ことと、その検出線。"""

    def test_the_shared_tensor_key_is_read_from_the_lender_container(self):
        found = drafter.lender_slots(
            _lender_graph(),
            SimpleNamespace(),
            SimpleNamespace(
                vocab_size=TINY_VOCAB,
                hidden_size=TINY_BACKBONE,
                num_hidden_layers=2,
                layer_types=list(TINY_LAYER_TYPES),
                num_kv_shared_layers=0,
            ),
        )

        assert found.shared_tensor == TINY_SHARED_TENSOR
        assert found.slots == {
            SLIDING_ATTENTION: ("l0.k", "l0.v"),
            FULL_ATTENTION: ("l1.k", "l1.v"),
        }
        assert found.shapes["l1.k"] == [1, TINY_KV_HEADS, decode.CAPACITY_SYMBOL, TINY_FULL_DEPTH]

    def test_a_lender_without_the_owner_slot_is_rejected(self):
        """貸し手のスロット名が変わった世代は「スロットが無い」で落ちる（沈黙しない）。"""
        graph = _lender_graph()
        graph.states.pop("l0.k")

        with pytest.raises(AssertionError, match=re.escape("state スロット 'l0.k' が無い")):
            drafter.lender_slots(
                graph,
                SimpleNamespace(),
                SimpleNamespace(
                    vocab_size=TINY_VOCAB,
                    hidden_size=TINY_BACKBONE,
                    num_hidden_layers=2,
                    layer_types=list(TINY_LAYER_TYPES),
                    num_kv_shared_layers=0,
                ),
            )

    def test_a_lender_with_two_candidate_tables_is_rejected(self):
        """主表が 1 本に決まらなければ落とす（どちらを借りるかを勘で決めない）。"""
        graph = _lender_graph()
        graph.initializers["other"] = IrInitializer(
            tensor="lender.other", storage=IrStorage(dtype="i8", scale="s2")
        )
        graph.values["other"] = IrValue(dtype="f32", shape=[TINY_VOCAB, TINY_BACKBONE])
        graph.values["e2"] = IrValue(dtype="f32", shape=[1, 1, TINY_BACKBONE])
        graph.nodes.append(
            IrNode(op="embedding", ins=["other", "tok"], outs=["e2"], attrs={"padding_idx": -1})
        )

        with pytest.raises(AssertionError, match=r"主埋め込み表.*が 2 本"):
            drafter.lender_slots(
                graph,
                SimpleNamespace(),
                SimpleNamespace(
                    vocab_size=TINY_VOCAB,
                    hidden_size=TINY_BACKBONE,
                    num_hidden_layers=2,
                    layer_types=list(TINY_LAYER_TYPES),
                    num_kv_shared_layers=0,
                ),
            )


def _tiny_drafter_config():
    """tiny な assistant config（疎 softmax 無し・KV は全層共有）。"""
    transformers = pytest.importorskip("transformers")
    text = transformers.Gemma4TextConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        intermediate_size=16,
        num_hidden_layers=len(TINY_LAYER_TYPES),
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_KV_HEADS,
        head_dim=TINY_SLIDING_DEPTH,
        global_head_dim=TINY_FULL_DEPTH,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        layer_types=list(TINY_LAYER_TYPES),
        num_kv_shared_layers=len(TINY_LAYER_TYPES),
        sliding_window=TINY_WINDOW,
        final_logit_softcapping=None,
        tie_word_embeddings=True,
        max_position_embeddings=64,
        rope_parameters={
            SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 100.0},
            FULL_ATTENTION: {
                "rope_type": "proportional",
                "rope_theta": 1000.0,
                "partial_rotary_factor": 0.5,
            },
        },
    )
    return transformers.Gemma4AssistantConfig(
        backbone_hidden_size=TINY_BACKBONE,
        num_centroids=4,
        centroid_intermediate_top_k=2,
        use_ordered_embeddings=False,
        text_config=text.to_dict(),
    )


@pytest.fixture
def tiny_drafter():
    """tiny な実モデルで組んだ {@link drafter.DrafterWrapper}（実重みは読まない）。"""
    transformers = pytest.importorskip("transformers")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

    torch.manual_seed(0)
    gx.register_attention()
    config = _tiny_drafter_config()
    config._attn_implementation = gx.ATTENTION_NAME
    text = config.get_text_config()
    text._attn_implementation = gx.ATTENTION_NAME
    model = transformers.Gemma4AssistantForCausalLM(config).to(torch.float32).eval()
    assert model.masked_embedding is None, "use_ordered_embeddings=False で疎 softmax が生えた"
    del model.model.embed_tokens
    model.model.rotary_emb = decode.RopeInputs(decode.unique_layer_types(text))
    embed = Gemma4TextScaledWordEmbedding(
        TINY_VOCAB, TINY_BACKBONE, 0, embed_scale=TINY_BACKBONE**0.5
    )
    embed.weight.data = torch.randn(TINY_VOCAB, TINY_BACKBONE)
    return drafter.DrafterWrapper(model, embed, TINY_STEPS).eval()


def _point_the_shared_tensor_elsewhere(borrow: dict, lend: dict) -> None:
    """共有 initializer の指し先だけを別のキーへ差し替える（貸し手には無い綴り）。"""
    del lend
    shared = next(entry for entry in borrow["initializers"].values() if "shared" in entry)
    shared["shared"]["tensor"] = "someone.else"


class TestExportedDrafterForm:
    """tiny な実モデルを export → 手術 → 共有宣言 → コンテナ検証まで通す（transformers が要る）。"""

    @pytest.fixture
    def tiny_container(self, tiny_drafter, tmp_path):
        """`export_series` と同じ順序で 1 周させた `(検証済みグラフ, config, lender)`。"""
        config = tiny_drafter.drafter.config
        text = config.get_text_config()
        lender = _tiny_lender()
        specs = decode.rope_specs(text)
        graph, tensors = drafter.export_module_with_attention(
            tiny_drafter, drafter.example_inputs(tiny_drafter, specs)
        )
        surgical = drafter.to_external_states_form(
            graph, drafter.external_states_plan(graph, text, lender)
        )
        shared_graph, remaining, _ = drafter.share_target_embedding(
            surgical, tensors, lender, _tiny_target_config()
        )
        declared = {
            init.tensor for init in shared_graph.initializers.values() if not init.is_shared
        }
        verified = publish_model(
            tmp_path / "model.safetensors",
            shared_graph,
            {name: value for name, value in remaining.items() if name in declared},
        )
        return verified, config, lender

    def test_the_form_gate_accepts_the_exported_graph(self, tiny_container):
        verified, config, lender = tiny_container

        form = drafter.assert_ir_form_drafter(verified, config, lender, TINY_STEPS)

        assert form["inputs"] == list(drafter.DRAFTER_INPUTS)
        assert form["outputs"] == TINY_STEPS
        assert form["attention_nodes"] == len(TINY_LAYER_TYPES) * TINY_STEPS
        assert form["external_slots"] == sorted(lender.shapes)
        assert form["shared_tensor"] == TINY_SHARED_TENSOR

    def test_the_traced_kv_inputs_and_masks_are_gone(self, tiny_container):
        """K/V の placeholder も開いた mask の定数も、配布形には 1 バイトも残らない。"""
        verified, _, _ = tiny_container

        assert [spec.name for spec in verified.inputs] == list(drafter.DRAFTER_INPUTS)
        # drafter の重みは rank 2 以下（linear と embedding だけ）なので、rank-4 の initializer が
        # 在れば **開いた mask の定数が刈られていない**印になる（誰も読まない `[1,1,1,N]` が
        # 配布物へ居座る）。
        assert not [name for name in verified.initializers if len(verified.values[name].shape) > 2]

    def test_every_slot_is_external_and_no_append_survives(self, tiny_container):
        verified, _, _ = tiny_container

        assert all(slot.external for slot in verified.states.values())
        assert all(node.op != "state_append" for node in verified.nodes)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (
                lambda g: g.states.__setitem__("l0.k", IrState(dtype="f32", shape=[1, 1, 6, 4])),
                "external でない",
            ),
            (lambda g: g.outputs.pop(), "IR 出力が"),
            (lambda g: g.inputs.pop(), "グラフ入力が"),
        ],
        ids=["owned-slot", "one-output-short", "missing-input"],
    )
    def test_the_form_gate_is_effective(self, tiny_container, mutate, message):
        """故障注入 — 3 本とも通ると「形も型も合ったまま別の資産」が配れる。"""
        verified, config, lender = tiny_container
        mutate(verified)

        with pytest.raises(AssertionError, match=message):
            drafter.assert_ir_form_drafter(verified, config, lender, TINY_STEPS)

    def test_the_distribution_gate_accepts_the_pair(self, tiny_container):
        """配る側の門は借り手 + 貸し手の**組**で見る（片方だけ差し替えた世代を落とす）。"""
        verified, config, _ = tiny_container
        borrowed = verified.to_dict()
        lent = _lender_graph().to_dict()
        rope = gemma4_distribution.gemma4_rope(config.get_text_config(), "tiny")

        gemma4_distribution.assert_gemma4_drafter_graph(
            borrowed, Path("borrower"), lent, Path("lender"), rope, TINY_BACKBONE
        )

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda borrow, lend: lend["states"].pop("l0.k"), "貸し手"),
            (
                lambda borrow, lend: lend["states"]["l1.k"].__setitem__("shape", [1, 1, "C", 99]),
                "貸し手の",
            ),
            (_point_the_shared_tensor_elsewhere, "指し先"),
            (lambda borrow, lend: lend["symbols"].clear(), "容量記号"),
        ],
        ids=["missing-slot", "capacity-mismatch", "foreign-tensor", "symbol-not-lent"],
    )
    def test_the_distribution_gate_is_effective(self, tiny_container, mutate, message):
        verified, config, _ = tiny_container
        borrowed = verified.to_dict()
        lent = _lender_graph().to_dict()
        mutate(borrowed, lent)
        rope = gemma4_distribution.gemma4_rope(config.get_text_config(), "tiny")

        with pytest.raises(DistError, match=message):
            gemma4_distribution.assert_gemma4_drafter_graph(
                borrowed, Path("borrower"), lent, Path("lender"), rope, TINY_BACKBONE
            )

    def test_the_graph_round_trips_through_the_parser(self, tiny_container):
        """配布形のグラフ JSON は往復でバイトが動かない（3 宣言とも受理集合が同じ）。"""
        verified, _, _ = tiny_container
        text = verified.to_json()

        assert parse_ir_graph(text).to_json() == text
        assert json.loads(text)["states"]["l0.k"]["external"] is True
