"""The MTP drafter script (`gemma4/export_drafter.py`) and the contract of its distribution gate.

Emitting with real weights is manual (the same discipline as the tests of the 3 chunk series).
What is pinned here is only the discipline whose breakage would give a **false PASS**:

- the quantization is **a single i8 pass**, applied to exactly every `nn.Linear`, and the shared
  table stays unrounded (whether i4 creeps in or the tied data is rounded twice, only the values
  break)
- the golden cases' source texts are documents in the repository, cut at **paragraph boundaries**
  (cutting mid-token makes the continuation degenerate into a loop of a few words, thinning the
  golden)
- the two things taken **structurally** from the lender (the product container) — the real shapes
  of the external slots and the shared tensor key
- a tiny real model goes through export → surgery → shared declaration → container verification,
  and {@link drafter.assert_ir_form_drafter} **actually checks** the borrower's form (6 inputs / k
  outputs / every slot external / 0 appends / the readonly count / 1 shared / 1 symbol)
- the distribution gate {@link gemma4.distribution.assert_gemma4_drafter_graph} actually rejects a
  broken fit between borrower and lender (slot names and real shapes, what the shared declaration
  points at, the capacity symbol)

Only the cases that need transformers SKIP via `importorskip` (the 2-job setup of ADR 0065).
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from gemma4 import distribution as gemma4_distribution
from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_drafter as drafter
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from karume.container import canonical_json, container_parts, ir_v2_document
from karume.dist import DistError
from karume.ir import IrState
from karume.pipeline import publish_model
from karume.verify import verify_container

#: Dimensions of the tiny drafter (the real one is hidden 256 / layers 4 / heads 4 / head_dim
#: 256|512 / backbone 1536 / vocabulary 262144). **Numbers different from the real ones** — a
#: baked-in dimension then fails.
TINY_VOCAB = 64
TINY_HIDDEN = 8
TINY_BACKBONE = 12
TINY_HEADS = 2
TINY_KV_HEADS = 1
TINY_SLIDING_DEPTH = 4
TINY_FULL_DEPTH = 8
TINY_WINDOW = 4
TINY_LAYER_TYPES = (SLIDING_ATTENTION, FULL_ATTENTION)
#: The step count is **the same k as the real one** (the distribution gate `GEMMA4_DRAFT_STEPS`
#: checks the exact count).
TINY_STEPS = drafter.DRAFT_STEPS

#: The lender's slots (the real ones are l13 / l14). The sliding capacity is a number, the full one
#: a symbol.
TINY_SLIDING_CAPACITY = 6
TINY_SHARED_TENSOR = "lender.embed.weight"


def _tiny_lender() -> drafter.LenderSlots:
    """The lender's real slot shapes and shared tensor key (values as if read from the product
    container)."""
    return drafter.LenderSlots(
        shapes={
            "l0.k": [1, TINY_KV_HEADS, TINY_SLIDING_CAPACITY, TINY_SLIDING_DEPTH],
            "l0.v": [1, TINY_KV_HEADS, TINY_SLIDING_CAPACITY, TINY_SLIDING_DEPTH],
            "l1.k": [1, TINY_KV_HEADS, decode.CAPACITY_SYMBOL, TINY_FULL_DEPTH],
            "l1.v": [1, TINY_KV_HEADS, decode.CAPACITY_SYMBOL, TINY_FULL_DEPTH],
        },
        slots={SLIDING_ATTENTION: ("l0.k", "l0.v"), FULL_ATTENTION: ("l1.k", "l1.v")},
        shared_tensor=TINY_SHARED_TENSOR,
    )


def _tiny_target_config() -> SimpleNamespace:
    """The upstream text config that determines the shape of the shared main table (only 2 fields
    are read)."""
    return SimpleNamespace(vocab_size=TINY_VOCAB, hidden_size=TINY_BACKBONE)


def _toy_wrapper() -> torch.nn.Module:
    """A holder that mimics **only the module layout** {@link drafter.quantize_wrapper} looks at.

    Building the real wrapper needs transformers and the HF checkpoint, so the discipline of the
    weights (round every `nn.Linear`, leave the shared main table and the norms alone) is pinned
    with this holder. The FQNs are spelled in the same form as the real one (`target_embed` /
    `drafter.<something>`).
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
    """Only **a single i8 pass** rounds (the drafter has no i4 option — the acceptance-rate
    measurement of 2026-09-08)."""

    def test_every_linear_is_a_seat_and_the_shared_table_is_not(self):
        """The shared table is the lender's bf16 values, so it is no int8 seat."""
        assert drafter.int8_seats(_toy_wrapper()) == {
            "drafter.lm_head.weight",
            "drafter.pre_projection.weight",
        }

    def test_it_rounds_exactly_those_seats_and_leaves_the_norm_and_the_table_alone(self):
        """A norm is not among `fake_quant_int8`'s target types, so not a single bit moves (f32).
        The shared table has a weight slot of an eligible type, so it is left alone by name — its
        values are what the goldens see of the lender's bf16 table."""
        wrapper = _toy_wrapper()
        norm = wrapper.drafter.norm.weight.detach().clone()
        table = wrapper.target_embed.weight.detach().clone()

        report, scales = drafter.quantize_wrapper(wrapper)

        assert set(scales) == drafter.int8_seats(wrapper)
        assert report.modules == 2
        assert torch.equal(wrapper.drafter.norm.weight, norm)
        assert torch.equal(wrapper.target_embed.weight, table)

    def test_a_resurrected_tied_embedding_is_rejected(self):
        """A generation where `drafter.model.embed_tokens` came back — the same data is rounded
        twice in one pass.

        On the day the `del` in `load_drafter` goes missing, the scale ledger keeps only the last
        one while the weight is rounded twice (a weight that does not restore with the declared
        scale goes into the distribution).
        """
        wrapper = _toy_wrapper()
        tied = torch.nn.Embedding(6, 4)
        tied.weight = wrapper.drafter.lm_head.weight
        wrapper.drafter.embed_tokens = tied

        with pytest.raises(AssertionError, match="extra"):
            drafter.quantize_wrapper(wrapper)


class TestTheCaseMaterial:
    """The source texts of the golden cases (documents in the repository cut at paragraph
    boundaries)."""

    def test_text_under_the_limit_is_kept_whole(self):
        assert drafter.truncate_paragraph("a\n\nb\n", 100) == "a\n\nb"

    def test_it_cuts_at_the_last_paragraph_break_before_the_limit(self):
        text = "one\n\ntwo\n\nthree and more words"

        assert drafter.truncate_paragraph(text, len("one\n\ntwo\n\n") + 2) == "one\n\ntwo"

    def test_a_document_without_a_paragraph_break_is_rejected(self):
        """A source text without a paragraph boundary would become "a prompt cut midway", so it is
        refused."""
        with pytest.raises(AssertionError, match="no paragraph boundary"):
            drafter.truncate_paragraph("x" * 100, 10)

    def test_every_declared_document_exists_and_has_a_break(self):
        """The declared source texts exist in the repository and have a paragraph boundary before
        the limit."""
        for name, relative, limit in drafter.CASE_DOCUMENTS:
            path = drafter.REPO_ROOT / relative
            assert path.is_file(), name
            assert drafter.truncate_paragraph(path.read_text(encoding="utf-8"), limit)


class TestTheAcceptanceStats:
    """The acceptance outlook taken from the goldens (`draft[t][j]` is compared with
    `tokens[t + 1 + j]`)."""

    @staticmethod
    def _stats(tokens: list[int], draft: list[list[int]]) -> dict:
        return drafter.acceptance_stats(
            {
                drafter.TOKENS_KEY: torch.tensor(tokens, dtype=torch.int32),
                drafter.DRAFT_KEY: torch.tensor(draft, dtype=torch.int32),
            }
        )

    def test_a_draft_that_hits_every_step_confirms_one_plus_k_tokens(self):
        """If every step hits, one cycle confirms `1 + k` tokens (1 bonus + k accepted)."""
        stats = self._stats([10, 11, 12, 13, 14], [[11, 12, 13], [12, 13, 14]])

        assert stats["per_step"] == [1.0, 1.0, 1.0]
        assert stats["tokens_per_cycle"] == 4.0

    def test_a_miss_at_the_head_stops_the_run_even_if_later_steps_hit(self):
        """Sequential acceptance counts only **the run from the start** (the per-position match
        rate counts the hits)."""
        stats = self._stats([10, 11, 12, 13], [[99, 12, 13]])

        assert stats["per_step"] == [0.0, 1.0, 1.0]
        assert stats["tokens_per_cycle"] == 1.0

    def test_a_run_that_stops_midway_confirms_the_prefix_plus_the_bonus(self):
        stats = self._stats([10, 11, 12, 13], [[11, 99, 13]])

        assert stats["per_step"] == [1.0, 0.0, 1.0]
        assert stats["tokens_per_cycle"] == 2.0

    def test_a_token_sequence_that_is_too_short_is_rejected(self):
        """The last cycle's drafts also need something to compare with (`cycles + k` tokens)."""
        with pytest.raises(AssertionError, match="cycles"):
            self._stats([10, 11, 12], [[11, 12, 13]])


class TestTheSpellingsMirrorTheDistribution:
    """The baking side and the distributing side do not split in spelling (if only one side moves,
    binding fails / the gate checks nothing)."""

    def test_the_graph_input_names_match(self):
        assert list(drafter.DRAFTER_INPUTS) == list(gemma4_distribution.GEMMA4_DRAFTER_GRAPH_INPUTS)

    def test_the_draft_step_count_matches(self):
        assert drafter.DRAFT_STEPS == gemma4_distribution.GEMMA4_DRAFT_STEPS

    def test_the_series_directory_name_matches(self):
        assert drafter.DEFAULT_OUT_DIR.name == gemma4_distribution.gemma4_series_name(
            gemma4_distribution.GEMMA4_DEFAULT_MODEL, gemma4_distribution.GEMMA4_DRAFTER_SUFFIX
        )

    def test_the_rope_inputs_come_from_the_shared_spelling(self):
        """The 4 RoPE inputs are built from the same one place as the decode series (not spelled
        separately)."""
        assert list(drafter.DRAFTER_INPUTS[2:]) == list(decode.ROPE_INPUTS)


def _lender_document() -> dict[str, Any]:
    """Graph descriptor of the lender's product container (**the IR v2 document** — 4 states + the
    main table's embedding).

    In v2 the initializer name is the tensor key itself (docs/ir-v2.md), so what the shared
    declaration points at and the declaration's name are the same single word. Storage does not
    appear in the document (the source of truth is the binding table), so {@link _lender_layouts}
    passes it separately.
    """
    return {
        "format": "karume-ir",
        "version": 2,
        "requires": {"ops": ["embedding"]},
        "symbols": [decode.CAPACITY_SYMBOL],
        "inputs": [{"name": "tok", "dtype": "i32", "shape": [1, 1]}],
        "outputs": ["e"],
        "initializers": {TINY_SHARED_TENSOR: {}},
        "values": {
            TINY_SHARED_TENSOR: {"dtype": "f32", "shape": [TINY_VOCAB, TINY_BACKBONE]},
            "e": {"dtype": "f32", "shape": [1, 1, TINY_BACKBONE]},
        },
        "states": {
            name: {"dtype": "f32", "shape": shape} for name, shape in _tiny_lender().shapes.items()
        },
        "nodes": [
            {
                "op": "embedding",
                "ins": [TINY_SHARED_TENSOR, "tok"],
                "outs": ["e"],
                "attrs": {"padding_idx": -1},
            }
        ],
    }


def _lender_layouts() -> dict[str, str]:
    """The lender's binding table (tensor key → storage layout): the regular product holds its
    table as bf16."""
    return {TINY_SHARED_TENSOR: "bf16"}


class TestTheCommandLine:
    """The drafter's CLI accepts only the knobs that take effect (knobs without effect are not
    silently dropped — fail loudly)."""

    def test_the_chunk_symbol_bound_is_not_accepted(self, capsys):
        """The drafter has no chunk symbol, so `--sym-max` is not accepted and startup fails."""
        with pytest.raises(SystemExit):
            drafter.build_parser().parse_args(["--sym-max", "512"])

        assert "--sym-max" in capsys.readouterr().err

    def test_the_shared_skeleton_is_still_accepted(self, tmp_path):
        """The skeleton's `--model-dir` / `--out` are accepted with the same spelling as the 1-shot
        script."""
        args = drafter.build_parser().parse_args(
            ["--model-dir", str(tmp_path / "m"), "--out", str(tmp_path / "o")]
        )

        assert (args.model_dir, args.out) == (tmp_path / "m", tmp_path / "o")
        assert not hasattr(args, "sym_max")

    def test_it_refuses_a_model_it_has_no_drafter_for(self, capsys):
        """The drafter script is E2B only (both upstream and the lender's default are E2B) —
        `--model e4b` fails at startup.

        Accepting it would silently bake E2B's drafter against E4B's lender (E4B's drafter is not
        supported).
        """
        with pytest.raises(SystemExit):
            drafter.build_parser().parse_args(["--model", "e4b"])

        assert "--model" in capsys.readouterr().err

    def test_the_one_shot_skeleton_keeps_the_chunk_symbol_bound(self):
        """Only the drafter omits it — the default skeleton keeps `--sym-max`."""
        args = gx.series_parser("one-shot", Path("out")).parse_args(["--sym-max", "512"])

        assert args.sym_max == 512


class TestReadingTheLender:
    """Taking things **structurally** from the lender (not copying the spelling), and its detection
    lines."""

    def test_the_shared_tensor_key_is_read_from_the_lender_container(self):
        found = drafter.lender_slots(
            _lender_document(),
            _lender_layouts(),
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
        """A generation where the lender's slot names changed fails with "no slot" (not
        silently)."""
        graph = _lender_document()
        graph["states"].pop("l0.k")

        with pytest.raises(AssertionError, match=re.escape("no state slot 'l0.k'")):
            drafter.lender_slots(
                graph,
                _lender_layouts(),
                SimpleNamespace(),
                SimpleNamespace(
                    vocab_size=TINY_VOCAB,
                    hidden_size=TINY_BACKBONE,
                    num_hidden_layers=2,
                    layer_types=list(TINY_LAYER_TYPES),
                    num_kv_shared_layers=0,
                ),
            )

    def test_a_lender_whose_table_is_not_bf16_is_rejected(self):
        """The shared declaration follows the regular product's bf16 table: a lender holding the
        table as int8 (a rounded product series) is another lender than the one declared."""
        with pytest.raises(AssertionError, match="'i8'"):
            drafter.lender_slots(
                _lender_document(),
                {TINY_SHARED_TENSOR: "i8"},
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
        """If the main table is not exactly one, fail (which one to borrow is not guessed)."""
        graph = _lender_document()
        graph["initializers"]["lender.other"] = {}
        graph["values"]["lender.other"] = {"dtype": "f32", "shape": [TINY_VOCAB, TINY_BACKBONE]}
        graph["values"]["e2"] = {"dtype": "f32", "shape": [1, 1, TINY_BACKBONE]}
        graph["nodes"].append(
            {
                "op": "embedding",
                "ins": ["lender.other", "tok"],
                "outs": ["e2"],
                "attrs": {"padding_idx": -1},
            }
        )

        with pytest.raises(AssertionError, match=r"main embedding tables.*: 2 found"):
            drafter.lender_slots(
                graph,
                {**_lender_layouts(), "lender.other": "i8"},
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
    """A tiny assistant config (no sparse softmax; the KV is shared by every layer)."""
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
    """{@link drafter.DrafterWrapper} built from a tiny real model (no real weights are read)."""
    transformers = pytest.importorskip("transformers")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

    torch.manual_seed(0)
    gx.register_attention()
    config = _tiny_drafter_config()
    config._attn_implementation = gx.ATTENTION_NAME
    text = config.get_text_config()
    text._attn_implementation = gx.ATTENTION_NAME
    model = transformers.Gemma4AssistantForCausalLM(config).to(torch.float32).eval()
    assert model.masked_embedding is None, (
        "the sparse softmax appeared with use_ordered_embeddings=False"
    )
    del model.model.embed_tokens
    model.model.rotary_emb = decode.RopeInputs(decode.unique_layer_types(text))
    embed = Gemma4TextScaledWordEmbedding(
        TINY_VOCAB, TINY_BACKBONE, 0, embed_scale=TINY_BACKBONE**0.5
    )
    embed.weight.data = torch.randn(TINY_VOCAB, TINY_BACKBONE)
    return drafter.DrafterWrapper(model, embed, TINY_STEPS).eval()


def _point_the_shared_tensor_elsewhere(borrow: dict, lend: dict) -> None:
    """Replace only what the shared initializer points at with another key (a spelling the lender
    does not have).

    In IR v2 the target = the initializer's name, so the key is renamed.
    """
    del lend
    name = next(key for key, entry in borrow["initializers"].items() if entry.get("shared"))
    borrow["initializers"]["someone.else"] = borrow["initializers"].pop(name)


class TestExportedDrafterForm:
    """Take a tiny real model through export → surgery → shared declaration → container
    verification (needs transformers)."""

    @pytest.fixture
    def tiny_container(self, tiny_drafter, tmp_path):
        """`(verified graph, config, lender)` after one pass in the same order as
        `export_series`."""
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
            tmp_path / gx.MODEL_FILE,
            shared_graph,
            {name: value for name, value in remaining.items() if name in declared},
            provenance=gx.PROVENANCE,
            graph_name="tiny",
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
        """Neither the K/V placeholders nor the open mask's constant leaves a byte in the
        distribution."""
        verified, _, _ = tiny_container

        assert [spec.name for spec in verified.inputs] == list(drafter.DRAFTER_INPUTS)
        # The drafter's weights are rank 2 or lower (only linears and embeddings), so a rank-4
        # initializer would be a sign that **the open mask's constant was not pruned** (an unread
        # `[1,1,1,N]` sitting in the distribution).
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
                "is not external",
            ),
            (lambda g: g.outputs.pop(), "IR outputs"),
            (lambda g: g.inputs.pop(), "graph inputs are"),
        ],
        ids=["owned-slot", "one-output-short", "missing-input"],
    )
    def test_the_form_gate_is_effective(self, tiny_container, mutate, message):
        """Fault injection — if all 3 passed, "a different asset that still matches in shape and
        type" could be distributed."""
        verified, config, lender = tiny_container
        mutate(verified)

        with pytest.raises(AssertionError, match=message):
            drafter.assert_ir_form_drafter(verified, config, lender, TINY_STEPS)

    @pytest.mark.parametrize("part", decode.ROPE_PARTS)
    def test_the_form_gate_checks_the_width_of_every_rope_part(self, tiny_container, part):
        """Fault injection — whichever of the cos / sin widths breaks, the writer-side gate fails.

        The order is bound by the input names, so only a width mix-up on one side gets past the
        name check (the distribution-side gate would catch it, but the writer-side gate's coverage
        would have a hole).
        """
        verified, config, lender = tiny_container
        name = decode.rope_input_name(decode.ROPE_LAYER_TYPES[0], part)
        index = next(i for i, spec in enumerate(verified.inputs) if spec.name == name)
        spec = verified.inputs[index]
        verified.inputs[index] = replace(spec, shape=[*spec.shape[:-1], spec.shape[-1] + 1])

        with pytest.raises(AssertionError, match=f"graph input '{name}'"):
            drafter.assert_ir_form_drafter(verified, config, lender, TINY_STEPS)

    def test_the_distribution_gate_accepts_the_pair(self, tiny_container):
        """The distributing side's gate checks borrower + lender **as a pair** (rejects a
        generation where only one was replaced)."""
        verified, config, _ = tiny_container
        borrowed = ir_v2_document(verified)
        lent = _lender_document()
        rope = gemma4_distribution.gemma4_rope(config.get_text_config(), "tiny")

        gemma4_distribution.assert_gemma4_drafter_graph(
            borrowed, Path("borrower"), lent, Path("lender"), rope, TINY_BACKBONE
        )

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda borrow, lend: lend["states"].pop("l0.k"), "is not in the lender"),
            (
                lambda borrow, lend: lend["states"]["l1.k"].__setitem__("shape", [1, 1, "C", 99]),
                "not the lender's",
            ),
            (_point_the_shared_tensor_elsewhere, "of the shared initializer is not in the lender"),
            (lambda borrow, lend: lend["symbols"].clear(), "capacity symbol"),
        ],
        ids=["missing-slot", "capacity-mismatch", "foreign-tensor", "symbol-not-lent"],
    )
    def test_the_distribution_gate_is_effective(self, tiny_container, mutate, message):
        verified, config, _ = tiny_container
        borrowed = ir_v2_document(verified)
        lent = _lender_document()
        mutate(borrowed, lent)
        rope = gemma4_distribution.gemma4_rope(config.get_text_config(), "tiny")

        with pytest.raises(DistError, match=message):
            gemma4_distribution.assert_gemma4_drafter_graph(
                borrowed, Path("borrower"), lent, Path("lender"), rope, TINY_BACKBONE
            )

    def test_the_graph_round_trips_through_the_container(self, tiny_container, tmp_path):
        """The distribution's graph JSON (IR v2's canonical serialization) does not move a byte
        when written to the container and read back (the accepted set is the same for all 3
        declarations). The read-back goes through the container's reader (`verify_container`).
        """
        verified, _, _ = tiny_container
        text = canonical_json(ir_v2_document(verified))

        read = verify_container(container_parts(tmp_path / gx.MODEL_FILE)).read
        assert canonical_json(read.graph.graphs["tiny"]) == text
        assert json.loads(text)["states"]["l0.k"]["external"] is True
