"""26B A4B's q4_0 QAT drafter (`gemma4/export_drafter_qat.py` against `gemma4.export_moe`'s
target): upstream's `Gemma4AssistantForCausalLM` borrowing from a target with experts, and the
test-only drafter of the cut model (`--layers`).

The target is `gemma4.tests.moe_fixture`'s tiny 26B checkpoint (5 layers — sliding, sliding,
full, sliding, full — 8 experts of which the router picks 2, K = V on the full layers with 1 KV head
against 2 on the sliding ones, no per-layer embeddings, every linear, the table and every expert
weight on a q4_0 grid, **two shards plus their index**), its table scaled by 16 (a power of two that
keeps it on the grid, so the greedy steps are decided by margins far above F32 noise). The drafter
is a `Gemma4AssistantForCausalLM` at that target's widths, saved like a hub snapshot. What is
pinned here is what would pass silently:

- the borrowed table is read from the shard that holds it, and a single-file checkpoint reads the
  same table; the record fingerprints the index and both shards;
- the export wrapper drafts what upstream's own assistant forward drafts, inside the sliding
  window and past it;
- the trace's KV placeholders and the written graph's slots carry 2 heads on the sliding layer and
  1 on the full one, and every stored weight is i4 g32;
- 26B's reference is the product series' own (`gemma4.export_moe`'s trace, grid import and
  reference), its table the drafter's copy bit for bit, and the goldens' continuation through the
  lender's cache — a target without PLE, with experts — is the full re-forward greedy one;
- a cut target keeps its first layers, so the KV owners are the cut's; its reference is the cut
  series' own; a series run lent by the cut series binds to the cut's slots and records the depth,
  and the full target's drafter does not bind to the cut;
- only 26B takes `--layers`.

transformers is required (`importorskip`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from _shared.container_read import read_stored
from _shared.decode_series import greedy_continuation
from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_drafter as drafter
from gemma4 import export_drafter_qat as qd
from gemma4 import export_moe as moe
from gemma4 import export_unified as unified
from gemma4 import provenance, sources
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from gemma4.tests import moe_fixture as fx
from gemma4.tests.test_export_drafter_qat import _grid_tensor
from karume.pipeline import publish_model

transformers = pytest.importorskip("transformers")

#: The tiny drafter's own widths (the backbone, KV heads, head widths, window and RoPE are the tiny
#: target's — read from its config). Multiples of the grid's group (32) on every row.
TINY_HIDDEN = 32
TINY_INTERMEDIATE = 64
TINY_HEADS = 4
TINY_LAYER_TYPES = (SLIDING_ATTENTION, FULL_ATTENTION)
TINY_STEPS = drafter.DRAFT_STEPS
TABLE_KEY = f"{gx.CHECKPOINT_TEXT_PREFIX}embed_tokens.weight"
TABLE_SCALE = 16.0
#: The weights' fingerprinted files of a sharded checkpoint: the index, both shards and the config.
WEIGHTS_FILES = (unified.CHECKPOINT_INDEX, *fx.SHARDS, "config.json")


def _ids(length: int, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, fx.VOCAB, (1, length), generator=generator, dtype=torch.int64)


#: The stand-in golden cases of `gemma4.export_moe`'s trace (the tiny checkpoint's tokenizer is a
#: stand-in): lengths inside the traced chunk range.
TRACE_CASES = (("capital-en", _ids(fx.WINDOW + 3)), ("capital-ja", _ids(fx.WINDOW + 5, seed=4)))


def _scaled_table() -> dict[str, torch.Tensor]:
    return {TABLE_KEY: fx.checkpoint_state()[TABLE_KEY] * TABLE_SCALE}


@pytest.fixture(scope="module")
def target_dir(tmp_path_factory) -> Path:
    """The tiny sharded 26B checkpoint with its table scaled by {@link TABLE_SCALE}."""
    return fx.write_checkpoint(tmp_path_factory.mktemp("tiny-26b"), extra=_scaled_table())


@pytest.fixture(scope="module")
def target_text(target_dir) -> Any:
    return gx.load_text_config(target_dir)


@pytest.fixture(scope="module")
def drafter_dir(tmp_path_factory, target_text) -> Path:
    """A tiny `Gemma4AssistantForCausalLM` snapshot at the tiny target's widths, every 2-D weight
    on the grid."""
    text = transformers.Gemma4TextConfig(
        vocab_size=int(target_text.vocab_size),
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_INTERMEDIATE,
        num_hidden_layers=len(TINY_LAYER_TYPES),
        layer_types=list(TINY_LAYER_TYPES),
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=int(target_text.num_key_value_heads),
        num_global_key_value_heads=int(target_text.num_global_key_value_heads),
        attention_k_eq_v=True,
        head_dim=int(target_text.head_dim),
        global_head_dim=int(target_text.global_head_dim),
        num_kv_shared_layers=len(TINY_LAYER_TYPES),
        sliding_window=int(target_text.sliding_window),
        final_logit_softcapping=None,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        tie_word_embeddings=True,
        max_position_embeddings=256,
        rope_parameters=dict(target_text.rope_parameters),
    )
    config = transformers.Gemma4AssistantConfig(
        backbone_hidden_size=int(target_text.hidden_size),
        num_centroids=4,
        centroid_intermediate_top_k=2,
        use_ordered_embeddings=False,
        text_config=text.to_dict(),
    )
    torch.manual_seed(0)
    model = transformers.Gemma4AssistantForCausalLM(config).eval()
    generator = torch.Generator().manual_seed(20261127)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 2:
                parameter.copy_(_grid_tensor(*parameter.shape, generator))
    directory = tmp_path_factory.mktemp("tiny-26b-assistant")
    model.save_pretrained(directory)
    return directory


@pytest.fixture
def loaded(target_dir, drafter_dir) -> drafter.DrafterWrapper:
    return drafter.load_drafter(TINY_STEPS, target_dir, source=drafter_dir)


@pytest.fixture
def stub_cases(monkeypatch) -> None:
    """`gemma4.export_moe`'s trace encodes the golden cases with the tokenizer, a stand-in here."""
    monkeypatch.setattr(gx, "encode_cases", lambda model_dir, texts: TRACE_CASES)


def _product_reference(target_dir: Path, layers: int | None) -> unified.UnifiedProductWrapper:
    """The reference `gemma4.export_moe`'s series runs build: its trace, one grid import, its
    reference builder (`test_export_moe._reference`'s steps)."""
    config = moe.load_config(target_dir, layers)
    wrapper = unified.shape_only_model(config)
    graph, tensors = unified.trace(wrapper, _ids(fx.WINDOW + 7), fx.TINY_SYM_MAX)
    keys = unified.trace_keys(wrapper)
    checkpoint = moe.open_checkpoint(target_dir)
    material = unified.prepare_material(graph, tensors, keys, checkpoint)
    return moe.reference_wrapper(config, checkpoint, keys, material.fixed)


def _call(wrapper: unified.UnifiedProductWrapper, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    config = wrapper.model.config
    with torch.no_grad():
        return wrapper(
            ids,
            *unified.no_soft_inputs(ids, int(config.hidden_size)),
            *decode.rope_args(decode.rope_specs(config), decode.positions_for(ids)),
            decode.last_rows_for(ids, 3),
        )


def _lender(target_text: Any) -> drafter.LenderSlots:
    """The tiny target's lender declaration (owners: its last sliding and last full layer)."""
    owners = decode.kv_owner_layers(target_text)
    sliding = [
        1,
        decode.kv_heads(target_text, SLIDING_ATTENTION),
        decode.sliding_capacity(target_text),
        int(target_text.head_dim),
    ]
    full = [
        1,
        decode.kv_heads(target_text, FULL_ATTENTION),
        decode.CAPACITY_SYMBOL,
        int(target_text.global_head_dim),
    ]
    slots = {
        layer_type: (
            decode.slot_name(owners[layer_type], "k"),
            decode.slot_name(owners[layer_type], "v"),
        )
        for layer_type in TINY_LAYER_TYPES
    }
    shapes = {
        name: (sliding if layer_type == SLIDING_ATTENTION else full)
        for layer_type, pair in slots.items()
        for name in pair
    }
    return drafter.LenderSlots(shapes=shapes, slots=slots, shared_tensor=unified.EMBED_KEY)


class TestTheShardedTarget:
    """26B's checkpoint is two shards and their index."""

    def test_the_drafter_and_the_table_load_as_upstream_s_classes_from_the_shard(
        self, loaded, target_dir, target_text
    ) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

        index = json.loads((target_dir / unified.CHECKPOINT_INDEX).read_text(encoding="utf-8"))
        shard = target_dir / index["weight_map"][TABLE_KEY]
        with safe_open(str(shard), framework="pt") as handle:
            table = handle.get_tensor(TABLE_KEY).to(torch.float32)

        assert not (target_dir / gx.CHECKPOINT_FILE).exists()
        assert type(loaded.drafter) is transformers.Gemma4AssistantForCausalLM
        assert loaded.drafter.masked_embedding is None
        assert type(loaded.target_embed) is Gemma4TextScaledWordEmbedding
        hidden = int(target_text.hidden_size)
        assert float(loaded.target_embed.embed_scale) == pytest.approx(hidden**0.5)
        assert torch.equal(loaded.target_embed.weight.detach(), table)

    def test_a_single_file_checkpoint_reads_the_same_table(
        self, target_dir, target_text, tmp_path
    ) -> None:
        single = fx.write_checkpoint(tmp_path / "single", extra=_scaled_table(), sharded=False)

        sharded = drafter.load_target_embed(target_dir, target_text)
        one_file = drafter.load_target_embed(single, target_text)

        assert torch.equal(sharded.weight, one_file.weight)

    def test_a_checkpoint_without_the_table_fails_loudly(self, target_text, tmp_path) -> None:
        directory = tmp_path / "no-table"
        directory.mkdir()
        (directory / unified.CHECKPOINT_INDEX).write_text(
            json.dumps({"weight_map": {"model.language_model.norm.weight": fx.SHARDS[0]}}),
            encoding="utf-8",
        )
        from safetensors.torch import save_file

        save_file(
            {"model.language_model.norm.weight": torch.ones(4)}, str(directory / fx.SHARDS[0])
        )

        with pytest.raises(AssertionError, match=r"has no 'model\.language_model\.embed_tokens"):
            drafter.load_target_embed(directory, target_text)

    def test_the_record_fingerprints_the_index_and_both_shards(self, target_dir, tmp_path) -> None:
        single = fx.write_checkpoint(tmp_path / "single", sharded=False)

        assert qd.target_weights_files(target_dir) == WEIGHTS_FILES
        assert qd.target_weights_files(single) == provenance.WEIGHTS_FILES


class TestTheDraftsAreUpstreamS:
    """The export wrapper consumes the target's state as upstream's candidate generator feeds the
    assistant: the target's scaled embedding of the last token next to its last hidden row, the
    K / V of the target's last sliding and full layer, the drafter's own token and projected hidden
    carried to the next step (the K / V a target's cache returns after a one-token step — the
    goldens' lender: a sliding layer's last `W` rows, a full layer's `P`)."""

    @pytest.mark.parametrize("past", [3, 7], ids=["inside-the-window", "past-the-window"])
    def test_the_wrapper_drafts_what_upstream_s_assistant_drafts(
        self, loaded, drafter_dir, target_text, past: int
    ) -> None:
        upstream = transformers.Gemma4AssistantForCausalLM.from_pretrained(
            drafter_dir, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        text = loaded.drafter.config.get_text_config()
        window = int(text.sliding_window)
        generator = torch.Generator().manual_seed(past)
        sliding_heads = decode.kv_heads(text, SLIDING_ATTENTION)
        full_heads = decode.kv_heads(text, FULL_ATTENTION)
        hidden = torch.randn(1, int(target_text.hidden_size), generator=generator)
        columns = min(past, window)
        held = [
            torch.randn(1, sliding_heads, columns, int(text.head_dim), generator=generator)
            for _ in range(2)
        ]
        full = [
            torch.randn(1, full_heads, past, int(text.global_head_dim), generator=generator)
            for _ in range(2)
        ]
        token = torch.tensor([[5]])
        rope = decode.rope_args(decode.rope_specs(text), torch.tensor([[past]]))

        with torch.no_grad():
            ours = loaded(token, hidden, *rope, *held, *full)
            shared = {SLIDING_ATTENTION: tuple(held), FULL_ATTENTION: tuple(full)}
            state, last, theirs = hidden.unsqueeze(1), token, []
            for _ in range(TINY_STEPS):
                out = upstream(
                    inputs_embeds=torch.cat([loaded.target_embed(last), state], dim=-1),
                    attention_mask=None,
                    position_ids=torch.tensor([[past]]),
                    shared_kv_states=shared,
                    use_cache=False,
                )
                last = out.logits.argmax(dim=-1)
                state = out.last_hidden_state
                theirs.append(int(last))

        assert [int(value) for value in ours] == theirs


class TestTheExportedForm:
    """The tiny drafter through the script's steps: import → trace → surgery → shared → write."""

    @pytest.fixture
    def exported(self, loaded, target_text, tmp_path):
        fixed = qd.import_drafter(loaded)
        text = loaded.drafter.config.get_text_config()
        lender = _lender(target_text)
        graph, tensors = drafter.export_module_with_attention(
            loaded, drafter.example_inputs(loaded, decode.rope_specs(text))
        )
        surgical = drafter.to_external_states_form(
            graph, drafter.external_states_plan(graph, text, lender)
        )
        shared_graph, remaining, _ = drafter.share_target_embedding(
            surgical, tensors, lender, target_text, storage=qd.SHARED_STORAGE
        )
        path = tmp_path / gx.MODEL_FILE
        verified = publish_model(
            path,
            shared_graph,
            qd.container_material(shared_graph, remaining, fixed),
            provenance=gx.PROVENANCE,
            graph_name="tiny",
            weight_dtype="f32",
            fixed_weights=fixed,
        )
        return verified, fixed, lender, path

    def test_the_kv_placeholders_carry_each_layer_type_s_heads(self, loaded) -> None:
        text = loaded.drafter.config.get_text_config()

        example = drafter.example_inputs(loaded, decode.rope_specs(text))

        assert [int(tensor.shape[1]) for tensor in example[-4:]] == [2, 2, 1, 1]

    def test_the_form_gate_takes_the_last_layers_of_each_type_as_owners(
        self, exported, loaded
    ) -> None:
        verified, fixed, lender, _ = exported

        form = drafter.assert_ir_form_drafter(verified, loaded.drafter.config, lender, TINY_STEPS)

        assert lender.slots == {
            SLIDING_ATTENTION: ("l3.k", "l3.v"),
            FULL_ATTENTION: ("l4.k", "l4.v"),
        }
        assert verified.states["l4.k"].shape[1] == verified.states["l4.v"].shape[1] == 1
        assert verified.states["l3.k"].shape[1] == 2
        assert form["storage"]["i4"] == len(fixed) + 1
        assert "i8" not in form["storage"]

    def test_every_stored_plane_is_an_imported_i4_g32_weight(self, exported) -> None:
        _, fixed, _, path = exported

        stored = read_stored(path)

        layers = len(TINY_LAYER_TYPES)
        # q / o / gate / up / down per layer, the two projections and the head.
        assert len(fixed) == 5 * layers + 3
        planes = sorted(key for key, entry in stored.items() if len(entry.shape) == 2)
        assert planes == sorted(fixed)
        assert all(
            (stored[key].layout, stored[key].group_size) == ("i4", qd.GROUP_SIZE) for key in fixed
        )


class TestTheGoldensAgainstTheMoeTarget:
    """The goldens' target is the product series' reference: experts per pick, no PLE."""

    @pytest.fixture
    def reference(self, target_dir, stub_cases) -> unified.UnifiedProductWrapper:
        return qd.load_target_reference(target_dir, tokenizer_dir=target_dir)

    def test_26b_s_reference_is_the_product_series_reference(self, reference, target_dir) -> None:
        expected = _product_reference(target_dir, None)
        ids = _ids(fx.RING + 3, seed=5)

        assert type(reference) is unified.UnifiedProductWrapper
        assert int(reference.model.config.num_hidden_layers) == fx.LAYERS
        for ours, theirs in zip(_call(reference, ids), _call(expected, ids), strict=True):
            assert torch.equal(ours, theirs)

    def test_the_drafter_borrows_the_reference_s_table_bit_for_bit(self, loaded, reference) -> None:
        qd.import_drafter(loaded)

        qd.assert_borrowed_table(loaded, reference)

    def test_the_continuation_is_the_full_re_forward_greedy_one(
        self, loaded, reference, tmp_path
    ) -> None:
        qd.import_drafter(loaded)
        ids = _ids(6, seed=11)
        cycles = 5

        summary = drafter.goldens_against(
            loaded, reference, [("case", ids)], tmp_path, cycles, "cpu"
        )

        tensors = load_file(str(tmp_path / f"{drafter.GOLDEN_PREFIX}case{drafter.GOLDEN_SUFFIX}"))
        view = unified.GreedyView(reference, decode.rope_specs(reference.model.config), ())
        expected, margins = greedy_continuation(view, ids, cycles + TINY_STEPS)
        assert summary["case"]["cycles"] == cycles
        assert min(margins) > 1e-3  # every step is decided, far above F32 noise
        assert tensors[drafter.TOKENS_KEY].tolist() == expected
        assert list(tensors[drafter.DRAFT_KEY].shape) == [cycles, TINY_STEPS]

    def test_the_reference_needs_the_tokenizer_directory(self, target_dir) -> None:
        with pytest.raises(qd.QatDrafterError, match="tokenizer directory"):
            qd.load_target_reference(target_dir)


class TestTheCutTarget:
    """`--layers N`: the target is the model cut after N layers (test only)."""

    def test_the_cut_config_keeps_the_first_layers_and_their_kv_owners(self, target_dir) -> None:
        full = qd.target_text_config(target_dir)
        cut = qd.target_text_config(target_dir, fx.CUT)

        assert int(full.num_hidden_layers) == fx.LAYERS
        assert decode.kv_owner_layers(full) == {SLIDING_ATTENTION: 3, FULL_ATTENTION: 4}
        assert int(cut.num_hidden_layers) == fx.CUT
        assert decode.kv_owner_layers(cut) == {SLIDING_ATTENTION: 1, FULL_ATTENTION: 2}

    def test_a_cut_of_a_target_without_experts_fails_loudly(self, tmp_path) -> None:
        from gemma4.tests.test_export_unified import write_checkpoint

        twelve_b = write_checkpoint(tmp_path / "tiny-12b")

        with pytest.raises(qd.QatDrafterError, match="needs 26B's form"):
            qd.target_text_config(twelve_b, 3)
        with pytest.raises(qd.QatDrafterError, match="needs 26B's form"):
            qd.load_target_reference(twelve_b, tokenizer_dir=twelve_b, layers=3)

    def test_the_cut_reference_is_the_cut_series_reference(self, target_dir, stub_cases) -> None:
        reference = qd.load_target_reference(target_dir, tokenizer_dir=target_dir, layers=fx.CUT)
        expected = _product_reference(target_dir, fx.CUT)
        ids = _ids(fx.RING + 2, seed=6)

        assert int(reference.model.config.num_hidden_layers) == fx.CUT
        for ours, theirs in zip(_call(reference, ids), _call(expected, ids), strict=True):
            assert torch.equal(ours, theirs)

    @pytest.fixture
    def cut_run(self, target_dir, drafter_dir, stub_cases, monkeypatch, tmp_path):
        """The two pinned sources, the drafter's snapshot and pin, and the cut series the drafter
        is lent by (`gemma4.export_moe.export_cut` on the tiny checkpoint)."""
        weights = tmp_path / "sources" / "weights"
        tokenizer = tmp_path / "sources" / "tokenizer"
        for directory, names in (
            (weights, WEIGHTS_FILES),
            (tokenizer, ("tokenizer.json", "generation_config.json")),
        ):
            directory.mkdir(parents=True)
            for name in names:
                shutil.copy(target_dir / name, directory / name)

        def pin(repo: str, revision: str, directory: Path, names: tuple[str, ...]):
            return sources.HubPin(
                repo=repo,
                revision=revision,
                sha256={
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in names
                },
            )

        pins = sources.ModelPins(
            weights=pin("test/weights", "0" * 40, weights, WEIGHTS_FILES),
            tokenizer=pin("test/tokenizer", "1" * 40, tokenizer, provenance.TOKENIZER_FILES),
        )
        drafter_pin = pin("test/assistant", "2" * 40, drafter_dir, qd.DRAFTER_FILES)
        lender = tmp_path / "series" / "gemma4-26b-cut3"
        moe.export_cut(
            weights,
            lender,
            tokenizer_dir=tokenizer,
            pins=pins,
            layers=fx.CUT,
            full_series=None,
            sym_max=fx.TINY_SYM_MAX,
        )
        ids = _ids(7, seed=12)
        monkeypatch.setattr(drafter, "build_cases", lambda model_dir: (("case", ids),))
        return SimpleNamespace(
            weights=weights,
            tokenizer=tokenizer,
            pins=pins,
            drafter_pin=drafter_pin,
            lender=lender,
            ids=ids,
        )

    def test_a_series_lent_by_the_cut_binds_to_the_cut_s_slots_and_records_the_depth(
        self, cut_run, drafter_dir, tmp_path
    ) -> None:
        out = tmp_path / "series" / "gemma4-26b-cut3-qat-drafter"
        cycles = 4

        summary = qd.export_series(
            cut_run.weights,
            out,
            tokenizer_dir=cut_run.tokenizer,
            pins=cut_run.pins,
            drafter_dir=drafter_dir,
            drafter_pin=cut_run.drafter_pin,
            product_dir=cut_run.lender,
            cycles=cycles,
            layers=fx.CUT,
        )

        record = json.loads((out / provenance.REFERENCE_FILE).read_text(encoding="utf-8"))
        assert summary["target_layers"] == fx.CUT
        assert summary["form"]["external_slots"] == ["l1.k", "l1.v", "l2.k", "l2.v"]
        assert record["cut"] == {"layers": fx.CUT}
        assert sorted(record["checkpoint"]["files"]) == sorted(
            (*WEIGHTS_FILES, *provenance.TOKENIZER_FILES)
        )
        assert record["drafter"]["revision"] == "2" * 40
        assert record["suppressTokens"] == []
        golden = load_file(str(out / f"{drafter.GOLDEN_PREFIX}case{drafter.GOLDEN_SUFFIX}"))
        cut = _product_reference(cut_run.weights, fx.CUT)
        view = unified.GreedyView(cut, decode.rope_specs(cut.model.config), ())
        expected, _ = greedy_continuation(view, cut_run.ids, cycles + TINY_STEPS)
        assert golden[drafter.TOKENS_KEY].tolist() == expected

    def test_the_full_target_s_drafter_does_not_bind_to_the_cut(
        self, cut_run, drafter_dir, tmp_path
    ) -> None:
        """Without `layers` the KV owners are the full model's (3 / 4), which the cut lacks."""
        with pytest.raises(AssertionError, match=r"no state slot 'l3\.k'"):
            qd.export_series(
                cut_run.weights,
                tmp_path / "series" / "gemma4-26b-qat-drafter",
                tokenizer_dir=cut_run.tokenizer,
                pins=cut_run.pins,
                drafter_dir=drafter_dir,
                drafter_pin=cut_run.drafter_pin,
                product_dir=cut_run.lender,
                goldens=False,
            )


class TestTheCutCommandLine:
    def test_only_a_model_with_cut_models_takes_layers(self, capsys) -> None:
        with pytest.raises(SystemExit):
            qd.parse_args(["--model", "e2b", "--layers"])
        assert "has no cut model" in capsys.readouterr().err

    def test_the_cut_drafter_s_series_is_never_the_distribution_s(self) -> None:
        assert qd.cut_suffix(12) == "cut12-qat-drafter"
        assert moe.cut_suffix(12) == "cut12"
