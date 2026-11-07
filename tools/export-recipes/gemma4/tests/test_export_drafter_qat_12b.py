"""12B's q4_0 QAT drafter (`gemma4/export_drafter_qat.py --model 12b`): upstream's
`Gemma4UnifiedAssistantForCausalLM` through the regular drafter's shared pieces.

The tiny models have 12B's structure in small numbers: the target is `gemma4.export_unified`'s
tiny 12B checkpoint (`gemma4.tests.test_export_unified.write_checkpoint` — 7 layers, K = V on the
full layers with 1 KV head against 2 on the sliding ones, no per-layer embeddings, every linear and
the tied table on a q4_0 grid; its table scaled by 16, a power of two that keeps it on the grid, so
the greedy steps are decided by margins far above F32 noise), and the drafter a
`Gemma4UnifiedAssistantForCausalLM` at that target's widths, saved like a hub snapshot. What is
pinned here is what would pass silently:

- the drafter class and the target's embedding class are upstream's, chosen by the configs' model
  types (an unknown one fails loudly);
- the export wrapper drafts what upstream's own assistant forward drafts (the candidate
  generator's loop), inside the sliding window and past it;
- the trace's KV placeholders and the written graph's slots carry 2 heads on the sliding layer and
  1 on the full one, and every stored weight is i4 g32;
- the goldens' target continuation through the lender's cache — a target without PLE — is the
  full re-forward greedy continuation with the suppressed ids masked;
- 12B's reference is the product goldens' own reference;
- 12B's defaults: its one pinned line as the target, its product series as the lender;
- the real configs and the real lender series (SKIP without them).

transformers is required (`importorskip`).
"""

from __future__ import annotations

import json
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
from gemma4 import export_unified as unified
from gemma4 import sources
from gemma4.distribution import (
    GEMMA4_PRODUCT_SUFFIX,
    GEMMA4_QAT_DRAFTER_SUFFIX,
    GEMMA4_QAT_PRODUCT_SUFFIX,
    gemma4_series_name,
)
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from gemma4.tests.test_export_drafter_qat import _grid_tensor
from gemma4.tests.test_export_unified import write_checkpoint
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
#: The lender's shared tensor key in the tiny lender declaration.
TINY_SHARED_TENSOR = unified.EMBED_KEY


@pytest.fixture(scope="module")
def target_dir(tmp_path_factory) -> Path:
    """The tiny 12B checkpoint with its table scaled by {@link TABLE_SCALE}."""
    plain = write_checkpoint(tmp_path_factory.mktemp("tiny-12b-plain"))
    with safe_open(str(plain / gx.CHECKPOINT_FILE), framework="pt") as handle:
        table = handle.get_tensor(TABLE_KEY) * TABLE_SCALE
    return write_checkpoint(tmp_path_factory.mktemp("tiny-12b"), extra={TABLE_KEY: table})


@pytest.fixture(scope="module")
def target_text(target_dir) -> Any:
    return gx.load_text_config(target_dir)


@pytest.fixture(scope="module")
def drafter_dir(tmp_path_factory, target_text) -> Path:
    """A tiny `Gemma4UnifiedAssistantForCausalLM` snapshot at the tiny target's widths, every 2-D
    weight on the grid."""
    text = transformers.Gemma4UnifiedTextConfig(
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
        tie_word_embeddings=True,
        max_position_embeddings=256,
        rope_parameters=dict(target_text.rope_parameters),
    )
    config = transformers.Gemma4UnifiedAssistantConfig(
        backbone_hidden_size=int(target_text.hidden_size),
        num_centroids=4,
        centroid_intermediate_top_k=2,
        use_ordered_embeddings=False,
        text_config=text.to_dict(),
    )
    torch.manual_seed(0)
    model = transformers.Gemma4UnifiedAssistantForCausalLM(config).eval()
    generator = torch.Generator().manual_seed(20261103)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 2:
                parameter.copy_(_grid_tensor(*parameter.shape, generator))
    directory = tmp_path_factory.mktemp("tiny-12b-assistant")
    model.save_pretrained(directory)
    return directory


@pytest.fixture
def loaded(target_dir, drafter_dir) -> drafter.DrafterWrapper:
    return drafter.load_drafter(TINY_STEPS, target_dir, source=drafter_dir)


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
    return drafter.LenderSlots(shapes=shapes, slots=slots, shared_tensor=TINY_SHARED_TENSOR)


class TestTheUnifiedClass:
    """The classes are upstream's, chosen by the configs' model types."""

    def test_the_drafter_loads_as_upstream_s_unified_assistant(
        self, loaded, target_dir, target_text
    ) -> None:
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedTextScaledWordEmbedding,
        )

        assert type(loaded.drafter) is transformers.Gemma4UnifiedAssistantForCausalLM
        assert loaded.drafter.masked_embedding is None
        assert not hasattr(loaded.drafter.model, "embed_tokens")
        assert type(loaded.target_embed) is Gemma4UnifiedTextScaledWordEmbedding
        hidden = int(target_text.hidden_size)
        assert float(loaded.target_embed.embed_scale) == pytest.approx(hidden**0.5)
        with safe_open(str(target_dir / gx.CHECKPOINT_FILE), framework="pt") as handle:
            table = handle.get_tensor(TABLE_KEY).to(torch.float32)
        assert torch.equal(loaded.target_embed.weight.detach(), table)

    def test_the_e2b_and_e4b_classes_are_unchanged(self) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

        regular = drafter.assistant_class(SimpleNamespace(model_type="gemma4_assistant"))
        table = drafter.target_embedding_class(SimpleNamespace(model_type="gemma4_text"))

        assert regular is transformers.Gemma4AssistantForCausalLM
        assert table is Gemma4TextScaledWordEmbedding

    def test_an_unknown_model_type_fails_loudly(self) -> None:
        with pytest.raises(AssertionError, match="not an MTP assistant"):
            drafter.assistant_class(SimpleNamespace(model_type="gemma4_text"))
        with pytest.raises(AssertionError, match="no known embedding"):
            drafter.target_embedding_class(SimpleNamespace(model_type="gemma4_assistant"))


class TestTheDraftsAreUpstreamS:
    """The export wrapper consumes the target's state as upstream's candidate generator feeds the
    unified assistant: the target's scaled embedding of the last token next to its last hidden
    row, the K / V of the target's last sliding and full layer, the drafter's own token and
    projected hidden carried to the next step.

    Both are fed the K / V a target's cache returns after a one-token step — the goldens' lender
    (`gemma4.export_drafter.draft_case`): a sliding layer's last `W` rows
    (`DynamicSlidingWindowLayer` keeps `W − 1` and appends the step's), a full layer's `P`. NOTE:
    fed more sliding rows (after a multi-token verify forward), upstream's bidirectional window
    admits the last `W + 1` of them; the runtime's contract stays `[P − min(P, W), P)` — the
    regular E2B drafter's.
    """

    @pytest.mark.parametrize("past", [3, 7], ids=["inside-the-window", "past-the-window"])
    def test_the_wrapper_drafts_what_upstream_s_assistant_drafts(
        self, loaded, drafter_dir, target_text, past: int
    ) -> None:
        upstream = transformers.Gemma4UnifiedAssistantForCausalLM.from_pretrained(
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
    """The tiny unified drafter through the script's steps: import → trace → surgery → shared →
    write."""

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

    def test_the_form_gate_accepts_a_full_slot_of_one_head(self, exported, loaded) -> None:
        verified, fixed, lender, _ = exported

        form = drafter.assert_ir_form_drafter(verified, loaded.drafter.config, lender, TINY_STEPS)

        k_full, v_full = lender.slots[FULL_ATTENTION]
        k_sliding, _ = lender.slots[SLIDING_ATTENTION]
        assert verified.states[k_full].shape[1] == verified.states[v_full].shape[1] == 1
        assert verified.states[k_sliding].shape[1] == 2
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


class TestTheGoldensAgainstAPleFreeTarget:
    """The goldens' target is 12B's reference, a target without per-layer embeddings."""

    @pytest.fixture
    def reference(self, target_dir) -> unified.UnifiedProductWrapper:
        return qd.load_target_reference(target_dir)

    def test_12b_s_reference_is_the_product_goldens_reference(self, reference, target_dir) -> None:
        config = unified.load_config(target_dir)
        keys = unified.trace_keys(unified.shape_only_model(config))
        expected = unified.reference_wrapper(config, unified.Checkpoint(target_dir), keys.int4)
        ids = torch.tensor([[2, 7, 11, 3, 5, 9]])
        specs = decode.rope_specs(config)

        def call(wrapper: unified.UnifiedProductWrapper) -> tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                return wrapper(
                    ids,
                    *unified.no_soft_inputs(ids, int(config.hidden_size)),
                    *decode.rope_args(specs, decode.positions_for(ids)),
                    decode.last_row_for(ids),
                )

        assert type(reference) is unified.UnifiedProductWrapper
        for ours, theirs in zip(call(reference), call(expected), strict=True):
            assert torch.equal(ours, theirs)

    def test_the_drafter_borrows_the_reference_s_table_bit_for_bit(self, loaded, reference) -> None:
        qd.import_drafter(loaded)

        qd.assert_borrowed_table(loaded, reference)

    def test_the_continuation_is_the_full_re_forward_one_with_the_suppressed_ids_masked(
        self, loaded, reference, tmp_path
    ) -> None:
        qd.import_drafter(loaded)
        ids = torch.randint(0, 64, (1, 6), generator=torch.Generator().manual_seed(3))
        cycles = 5
        specs = decode.rope_specs(reference.model.config)

        def continuation(suppress: tuple[int, ...]) -> dict[str, torch.Tensor]:
            out = tmp_path / f"suppress-{len(suppress)}"
            out.mkdir()
            summary = drafter.goldens_against(
                loaded, reference, [("case", ids)], out, cycles, "cpu", suppress=suppress
            )
            assert summary["case"]["cycles"] == cycles
            return load_file(str(out / f"{drafter.GOLDEN_PREFIX}case{drafter.GOLDEN_SUFFIX}"))

        plain = continuation(())
        suppress = (int(plain[drafter.TOKENS_KEY][0]),)
        masked = continuation(suppress)

        for tensors, ids_masked in ((plain, ()), (masked, suppress)):
            view = unified.GreedyView(reference, specs, ids_masked)
            expected, margins = greedy_continuation(view, ids, cycles + TINY_STEPS)
            assert min(margins) > 1e-3  # every step is decided, far above F32 noise
            assert tensors[drafter.TOKENS_KEY].tolist() == expected
            assert list(tensors[drafter.DRAFT_KEY].shape) == [cycles, TINY_STEPS]
        assert not set(masked[drafter.TOKENS_KEY].tolist()) & set(suppress)

    def test_a_ple_free_lender_reads_no_tables_and_a_ple_lender_reads_its_own(
        self, target_text
    ) -> None:
        """The config decides (`hidden_size_per_layer_input`), not whether the wrapper holds
        tables: a PLE target without its tables fails instead of running without them."""
        free = SimpleNamespace(model=SimpleNamespace(config=target_text))
        tables = object()
        ple = SimpleNamespace(
            model=SimpleNamespace(
                config=transformers.Gemma4TextConfig(
                    hidden_size_per_layer_input=8, vocab_size_per_layer_input=64
                )
            ),
            per_layer=tables,
            per_layer_scale=0.5,
        )

        assert drafter._new_lender(free, {}).tables is None
        assert drafter._new_lender(ple, {}).tables is tables
        del ple.per_layer
        with pytest.raises(AttributeError):
            drafter._new_lender(ple, {})


class TestTheTwelveBDefaults:
    """`--model 12b`: 12B's one pinned line as the target, its product series as the lender."""

    def test_the_lender_is_the_qat_product_series_or_12b_s_product_series(self) -> None:
        assert qd.lender_suffix("e2b") == qd.lender_suffix("e4b") == GEMMA4_QAT_PRODUCT_SUFFIX
        assert qd.lender_suffix("12b") == GEMMA4_PRODUCT_SUFFIX
        with pytest.raises(ValueError, match="no q4_0 QAT target"):
            qd.lender_suffix("31b")

    def test_the_defaults_are_12b_s_pinned_sources_and_its_product_series(self) -> None:
        options = gx.resolve_series_args(qd.build_parser().parse_args(["--model", "12b"]))

        assert options["model_dir"] == sources.checkpoint_dir("12b")
        assert options["tokenizer_dir"] == sources.tokenizer_dir("12b")
        assert options["drafter_dir"] == sources.qat_drafter_dir("12b")
        assert options["out"].name == gemma4_series_name("12b", GEMMA4_QAT_DRAFTER_SUFFIX)
        assert options["out"].name == "gemma4-12b-qat-drafter"
        assert options["product_dir"].name == "gemma4-12b-product"

    def test_main_hands_the_export_12b_s_pins(self, monkeypatch, capsys) -> None:
        calls: list[dict[str, Any]] = []

        def export(model_dir: Path, out: Path, **options: Any) -> dict[str, Any]:
            calls.append(options)
            return {}

        monkeypatch.setattr(qd, "export_series", export)

        qd.main(["--model", "12b"])

        assert calls[0]["pins"] is sources.HUB_PINS["12b"]
        assert calls[0]["drafter_pin"] is sources.QAT_DRAFTER_PINS["12b"]
        assert json.loads(capsys.readouterr().out)["model_dir"] == str(
            sources.checkpoint_dir("12b")
        )


def _real_configs() -> tuple[Any, Any]:
    drafter_dir, target_dir = (
        sources.qat_drafter_dir("12b"),
        sources.qat_target_checkpoint_dir("12b"),
    )
    if not all((path / "config.json").is_file() for path in (drafter_dir, target_dir)):
        pytest.skip("12B's QAT drafter or target config is not in the hub cache")
    return transformers.AutoConfig.from_pretrained(drafter_dir), gx.load_text_config(target_dir)


class TestTheRealTwelveB:
    """The real configs and the real lender series (SKIP without them): the facts the shared
    pieces read from upstream instead of hard-coding."""

    def test_the_drafter_matches_its_target(self) -> None:
        config, target = _real_configs()
        text = config.get_text_config()

        assert drafter.assistant_class(config) is transformers.Gemma4UnifiedAssistantForCausalLM
        assert config.use_ordered_embeddings is False
        assert int(config.backbone_hidden_size) == int(target.hidden_size) == 3840
        for layer_type, heads in ((SLIDING_ATTENTION, 8), (FULL_ATTENTION, 1)):
            assert decode.kv_heads(text, layer_type) == decode.kv_heads(target, layer_type) == heads
        assert (int(text.head_dim), int(text.global_head_dim)) == (
            int(target.head_dim),
            int(target.global_head_dim),
        )
        assert int(text.sliding_window) == int(target.sliding_window) == 1024
        assert decode.rope_specs(text) == decode.rope_specs(target)
        assert decode.kv_owner_layers(target) == {SLIDING_ATTENTION: 46, FULL_ATTENTION: 47}

    def test_every_drafter_linear_splits_into_grid_groups(self) -> None:
        config, _ = _real_configs()
        config.use_ordered_embeddings = False

        with torch.device("meta"):
            model = drafter.assistant_class(config)(config)

        linears = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        assert len(linears) == 23
        assert all(module.in_features % qd.GROUP_SIZE == 0 for module in linears.values())
        assert model.pre_projection.in_features == 2 * 3840
        assert model.post_projection.out_features == 3840

    def test_the_product_series_lends_an_i4_g32_table_and_the_owner_slots(self) -> None:
        config, target = _real_configs()
        product_dir = gx.series_dir("12b", GEMMA4_PRODUCT_SUFFIX)
        if not (product_dir / "reference.json").is_file():
            pytest.skip(f"12B's product series is not exported ({product_dir})")

        lender = qd.read_lender(product_dir, config, target, sources.HUB_PINS["12b"])

        assert lender.slots == {
            SLIDING_ATTENTION: ("l46.k", "l46.v"),
            FULL_ATTENTION: ("l47.k", "l47.v"),
        }
        assert lender.shapes["l46.k"] == [1, 8, 1032, 256]
        assert lender.shapes["l47.k"] == [1, 1, decode.CAPACITY_SYMBOL, 512]
