"""The q4_0 QAT drafter script (`gemma4/export_drafter_qat.py`) and what it keeps of the regular
drafter (`gemma4/export_drafter.py`).

The real export is run by hand (the regular drafter's rule). What is pinned here is what breaks
silently:

- the pins of the three QAT assistants, and that every drafter weight of the pinned snapshots
  lies on the q4_0 grid (the real hub cache; SKIP without it); 12B's class is pinned in
  `test_export_drafter_qat_12b.py`
- the grid import covers every drafter weight class — the 22 linears and the head as packed i4
  g32, the borrowed table as values only — and refuses a weight outside it
- the borrowed table's declaration matches the QAT target's: i4 g32 in the lender's container,
  the pinned QAT weights in the lender's record, the same values as the target reference
- E4B's widths (backbone 2560, two KV heads) reach the trace and the form check
- the regular E2B drafter keeps its i8 path (its own tests are unchanged)

The cases that need transformers SKIP through `importorskip` (ADR 0065's two-job layout).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from _shared.container_read import open_container, read_layouts, read_stored
from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_drafter as drafter
from gemma4 import export_drafter_qat as qd
from gemma4 import export_product_qat as qat
from gemma4 import export_unified as unified
from gemma4 import provenance, sources
from gemma4.distribution import gemma4_series_name
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from karume.container import Provenance
from karume.emit import FixedQuantizedWeight, unpack_int4
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import publish_model
from karume.quantize import channel_scale, dequantize_int4, quantize_to_int8

#: A tiny QAT drafter's widths — multiples of the grid's group (32) on every row, and **not the
#: real numbers** (real: hidden 256 / backbone 1536 or 2560 / vocabulary 262144). Two KV heads, as
#: E4B's drafter, so the trace exercises a head count the regular tiny drafter (one) does not.
TINY_VOCAB = 48
TINY_HIDDEN = 32
TINY_BACKBONE = 64
TINY_HEADS = 2
TINY_KV_HEADS = 2
TINY_SLIDING_DEPTH = 16
TINY_FULL_DEPTH = 32
TINY_INTERMEDIATE = 64
TINY_WINDOW = 4
TINY_LAYER_TYPES = (SLIDING_ATTENTION, FULL_ATTENTION)
TINY_STEPS = drafter.DRAFT_STEPS
TINY_SLIDING_CAPACITY = 6
TINY_SHARED_TENSOR = "lender.embed.weight"

#: The real per-model facts the E4B generalisation rests on (backbone width = the target's
#: hidden size, the drafter's KV heads = the target's, the target's KV owner layers).
REAL_WIDTHS: dict[str, tuple[int, int, dict[str, int]]] = {
    "e2b": (1536, 1, {SLIDING_ATTENTION: 13, FULL_ATTENTION: 14}),
    "e4b": (2560, 2, {SLIDING_ATTENTION: 22, FULL_ATTENTION: 23}),
}


def _grid_tensor(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    """A rank-2 F32 tensor on a q4_0 grid: codes in [−8, 7], one power-of-two step per group of
    32 — the step is exact in F32, so the imported values must equal these bit for bit."""
    codes = torch.randint(-8, 8, (rows, cols), generator=generator).to(torch.float32)
    exponents = torch.randint(-9, -4, (rows, cols // qd.GROUP_SIZE), generator=generator)
    steps = torch.pow(2.0, exponents.to(torch.float32))
    return codes * steps.repeat_interleave(qd.GROUP_SIZE, dim=1)


def _tiny_config(kv_heads: int = TINY_KV_HEADS):
    transformers = pytest.importorskip("transformers")
    text = transformers.Gemma4TextConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_INTERMEDIATE,
        num_hidden_layers=len(TINY_LAYER_TYPES),
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=kv_heads,
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


def _tiny_wrapper(kv_heads: int = TINY_KV_HEADS) -> drafter.DrafterWrapper:
    """A tiny drafter built the way `gemma4.export_drafter.load_drafter` builds the real one, every
    2-D weight (and the borrowed table) on the grid."""
    transformers = pytest.importorskip("transformers")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

    torch.manual_seed(0)
    gx.register_attention()
    config = _tiny_config(kv_heads)
    config._attn_implementation = gx.ATTENTION_NAME
    text = config.get_text_config()
    text._attn_implementation = gx.ATTENTION_NAME
    model = transformers.Gemma4AssistantForCausalLM(config).to(torch.float32).eval()
    del model.model.embed_tokens
    model.model.rotary_emb = decode.RopeInputs(decode.unique_layer_types(text))
    embed = Gemma4TextScaledWordEmbedding(
        TINY_VOCAB, TINY_BACKBONE, 0, embed_scale=TINY_BACKBONE**0.5
    )
    wrapper = drafter.DrafterWrapper(model, embed, TINY_STEPS).eval()
    generator = torch.Generator().manual_seed(20261103)
    with torch.no_grad():
        for _, parameter in wrapper.named_parameters():
            if parameter.dim() == 2:
                parameter.copy_(_grid_tensor(*parameter.shape, generator))
    return wrapper


def _tiny_lender(kv_heads: int = TINY_KV_HEADS) -> drafter.LenderSlots:
    sliding = [1, kv_heads, TINY_SLIDING_CAPACITY, TINY_SLIDING_DEPTH]
    full = [1, kv_heads, decode.CAPACITY_SYMBOL, TINY_FULL_DEPTH]
    return drafter.LenderSlots(
        shapes={"l0.k": sliding, "l0.v": sliding, "l1.k": full, "l1.v": full},
        slots={SLIDING_ATTENTION: ("l0.k", "l0.v"), FULL_ATTENTION: ("l1.k", "l1.v")},
        shared_tensor=TINY_SHARED_TENSOR,
    )


def _target_config() -> SimpleNamespace:
    return SimpleNamespace(vocab_size=TINY_VOCAB, hidden_size=TINY_BACKBONE)


def _linear_weights(wrapper: torch.nn.Module) -> set[str]:
    return {
        f"{name}.weight"
        for name, module in wrapper.named_modules()
        if isinstance(module, torch.nn.Linear)
    }


class TestThePins:
    """The QAT assistants are pinned like the QAT targets (`gemma4.sources`)."""

    @pytest.mark.parametrize(
        ("model", "revision"),
        [
            ("e2b", "ebc7e1a211354561464cb82ed6d886792138dcb6"),
            ("e4b", "27f8d204f09f2be353d6ff0bf0d012792b13c79f"),
        ],
    )
    def test_each_model_pins_its_qat_assistant(self, model: str, revision: str) -> None:
        pin = sources.qat_drafter_pin(model)

        assert pin.repo == f"google/gemma-4-{model.upper()}-it-qat-q4_0-unquantized-assistant"
        assert pin.revision == revision
        assert sorted(pin.sha256) == sorted(qd.DRAFTER_FILES)

    def test_every_qat_drafter_has_a_qat_target_line(self) -> None:
        """E2B / E4B pair with their QAT line, 12B with its one line (whose weights are the QAT
        checkpoint) — each drafter with the QAT weights of its own model."""
        assert set(sources.QAT_DRAFTER_PINS) == set(sources.QAT_TARGET_PINS)
        for model, pins in sources.QAT_TARGET_PINS.items():
            assert pins.weights.repo == sources.QAT_DRAFTER_UPSTREAM[model].removesuffix(
                "-assistant"
            ), model

    def test_the_regular_assistant_is_not_a_qat_drafter(self) -> None:
        """Google's card: a QAT target needs a QAT drafter of the same precision."""
        assert drafter.ASSISTANT_REPO not in sources.QAT_DRAFTER_UPSTREAM.values()

    def test_a_model_without_a_qat_drafter_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="no q4_0 QAT drafter"):
            sources.qat_drafter_pin("31b")

    def test_the_directory_is_the_hub_cache_snapshot(self, tmp_path, monkeypatch) -> None:
        constants = pytest.importorskip("huggingface_hub.constants")
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
        pin = sources.qat_drafter_pin("e4b")

        assert sources.qat_drafter_dir("e4b") == (
            tmp_path
            / "models--google--gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
            / "snapshots"
            / pin.revision
        )

    @pytest.mark.parametrize("model", ["e2b", "e4b", "12b"])
    def test_the_cached_snapshot_matches_the_pin(self, model: str) -> None:
        """The real hub cache (needs the snapshot)."""
        directory = sources.qat_drafter_dir(model)
        if not (directory / "config.json").is_file():
            pytest.skip(f"{model}'s QAT drafter is not in the hub cache ({directory})")

        sources.assert_snapshot_pin(directory, sources.qat_drafter_pin(model))

    @pytest.mark.parametrize("model", ["e2b", "e4b", "12b"])
    def test_every_drafter_weight_of_the_pinned_snapshot_is_on_the_grid(self, model: str) -> None:
        """The fact the i4 storage rests on, on the real checkpoint: every rank-2 tensor but the
        unused sparse-head centroids recovers its q4_0 grid (needs the snapshot)."""
        from safetensors import safe_open

        path = sources.qat_drafter_dir(model) / "model.safetensors"
        if not path.is_file():
            pytest.skip(f"{model}'s QAT drafter is not in the hub cache ({path})")
        imported = []
        with safe_open(str(path), framework="pt") as handle:
            for key in sorted(handle.keys()):
                if (
                    key.startswith("masked_embedding.")
                    or len(handle.get_slice(key).get_shape()) != 2
                ):
                    continue
                qat.import_grid(handle.get_tensor(key).to(torch.float32), key)
                imported.append(key)

        # 22 linears + the drafter's embedding (the head's tied tensor).
        assert len(imported) == 23
        assert "model.embed_tokens.weight" in imported


class TestTheGridImport:
    """{@link qd.import_drafter}: every drafter weight class, and nothing outside it."""

    def test_every_linear_and_the_head_become_packed_i4_g32(self) -> None:
        wrapper = _tiny_wrapper()
        before = {
            key: wrapper.get_parameter(key).detach().clone() for key in qd.int4_seats(wrapper)
        }

        fixed = qd.import_drafter(wrapper)

        assert set(fixed) == _linear_weights(wrapper)
        assert "drafter.lm_head.weight" in fixed
        assert "drafter.pre_projection.weight" in fixed
        assert "drafter.post_projection.weight" in fixed
        for key, weight in fixed.items():
            values = wrapper.get_parameter(key).detach()
            rows, cols = values.shape
            assert weight.dtype == "i4", key
            assert list(weight.packed.shape) == [rows, cols // 2], key
            assert list(weight.scale.shape) == [rows, cols // qd.GROUP_SIZE], key
            # The stored bytes decode to the wrapper's values, which are the upstream values.
            decoded = dequantize_int4(unpack_int4(weight.packed, values.shape), weight.scale)
            assert torch.equal(decoded, values), key
            assert torch.equal(values, before[key]), key

    def test_the_borrowed_table_is_imported_without_bytes(self) -> None:
        """Its values become what the product series' import of the same tensor gives (here a
        tensor near the grid, so the import visibly moves it)."""
        wrapper = _tiny_wrapper()
        table = wrapper.get_parameter(qd.TARGET_EMBED_WEIGHT)
        table.data += 2.0**-20
        perturbed = table.detach().clone()
        expected = perturbed.clone()
        qat.import_grid(expected, "the same tensor")

        fixed = qd.import_drafter(wrapper)

        assert qd.TARGET_EMBED_WEIGHT not in fixed
        assert torch.equal(table.detach(), expected)
        assert not torch.equal(table.detach(), perturbed)

    def test_the_norms_are_not_touched(self) -> None:
        wrapper = _tiny_wrapper()
        norms = {
            name: parameter.detach().clone()
            for name, parameter in wrapper.named_parameters()
            if parameter.dim() < 2
        }

        qd.import_drafter(wrapper)

        assert norms
        for name, value in norms.items():
            assert torch.equal(wrapper.get_parameter(name), value), name

    def test_a_linear_off_the_grid_fails_naming_it(self) -> None:
        wrapper = _tiny_wrapper()
        key = "drafter.model.layers.1.self_attn.q_proj.weight"
        wrapper.get_parameter(key).data[0, : qd.GROUP_SIZE] = torch.randn(qd.GROUP_SIZE)

        with pytest.raises(unified.UnifiedExportError, match=key):
            qd.import_drafter(wrapper)

    def test_a_borrowed_table_off_the_grid_fails(self) -> None:
        wrapper = _tiny_wrapper()
        wrapper.get_parameter(qd.TARGET_EMBED_WEIGHT).data[3, : qd.GROUP_SIZE] = torch.randn(
            qd.GROUP_SIZE
        )

        with pytest.raises(unified.UnifiedExportError, match="the target's table"):
            qd.import_drafter(wrapper)

    def test_a_resurrected_tied_embedding_is_rejected(self) -> None:
        """`drafter.model.embed_tokens` back again — the head's tensor seen under two names."""
        wrapper = _tiny_wrapper()
        tied = torch.nn.Embedding(TINY_VOCAB, TINY_HIDDEN)
        tied.weight = wrapper.drafter.lm_head.weight
        wrapper.drafter.model.embed_tokens = tied

        with pytest.raises(qd.QatDrafterError, match="outside the grid import"):
            qd.import_drafter(wrapper)


def _exported(
    wrapper: drafter.DrafterWrapper, tmp_path: Path, *, storage: IrStorage | None
) -> tuple[IrGraph, dict[str, FixedQuantizedWeight], Path]:
    """The script's order on a tiny model: import → trace → surgery → shared → write."""
    fixed = qd.import_drafter(wrapper)
    text = wrapper.drafter.config.get_text_config()
    lender = _tiny_lender(int(text.num_key_value_heads))
    graph, tensors = drafter.export_module_with_attention(
        wrapper, drafter.example_inputs(wrapper, decode.rope_specs(text))
    )
    surgical = drafter.to_external_states_form(
        graph, drafter.external_states_plan(graph, text, lender)
    )
    shared_graph, remaining, _ = drafter.share_target_embedding(
        surgical, tensors, lender, _target_config(), storage=storage
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
    return verified, fixed, path


class TestTheExportedForm:
    """A tiny model through the script's steps (transformers is needed)."""

    @pytest.fixture
    def exported(self, tmp_path):
        wrapper = _tiny_wrapper()
        verified, fixed, path = _exported(wrapper, tmp_path, storage=qd.SHARED_STORAGE)
        return wrapper, verified, fixed, path

    def test_the_form_gate_accepts_it_with_two_kv_heads(self, exported) -> None:
        wrapper, verified, fixed, _ = exported

        form = drafter.assert_ir_form_drafter(
            verified, wrapper.drafter.config, _tiny_lender(), TINY_STEPS
        )

        assert form["inputs"] == list(drafter.DRAFTER_INPUTS)
        assert form["storage"]["i4"] == len(fixed) + 1
        assert "i8" not in form["storage"]

    def test_the_shared_declaration_is_i4_without_a_group(self, exported) -> None:
        _, verified, _, _ = exported

        shared = [init for init in verified.initializers.values() if init.is_shared]

        assert len(shared) == 1
        assert shared[0].shared.tensor == TINY_SHARED_TENSOR
        assert shared[0].storage == IrStorage(dtype="i4")

    def test_every_stored_weight_reads_back_as_i4_g32_with_the_wrapper_s_values(
        self, exported
    ) -> None:
        wrapper, _, fixed, path = exported

        stored = read_stored(path)

        for key in fixed:
            entry = stored[key]
            values = wrapper.get_parameter(key).detach()
            assert (entry.layout, entry.group_size) == ("i4", qd.GROUP_SIZE), key
            packed = torch.frombuffer(bytearray(entry.payload), dtype=torch.uint8)
            scale = torch.frombuffer(bytearray(entry.scale), dtype=torch.float32)
            decoded = dequantize_int4(
                unpack_int4(packed.reshape(values.shape[0], -1), values.shape),
                scale.reshape(values.shape[0], -1),
            )
            assert torch.equal(decoded, values), key
        assert qd.TARGET_EMBED_WEIGHT not in stored
        planes = [key for key, entry in stored.items() if len(entry.shape) == 2]
        assert sorted(planes) == sorted(fixed)

    def test_the_kv_placeholders_carry_the_drafter_s_kv_heads(self) -> None:
        for heads in (1, 2):
            wrapper = _tiny_wrapper(heads)
            text = wrapper.drafter.config.get_text_config()
            example = drafter.example_inputs(wrapper, decode.rope_specs(text))

            assert [list(tensor.shape[:2]) for tensor in example[-4:]] == [[1, heads]] * 4

    def test_a_rank_2_tensor_outside_the_import_is_refused(self, tmp_path) -> None:
        wrapper = _tiny_wrapper()
        fixed = qd.import_drafter(wrapper)
        dropped = dict(fixed)
        dropped.pop("drafter.post_projection.weight")
        text = wrapper.drafter.config.get_text_config()
        graph, tensors = drafter.export_module_with_attention(
            wrapper, drafter.example_inputs(wrapper, decode.rope_specs(text))
        )

        with pytest.raises(qd.QatDrafterError, match="post_projection"):
            qd.container_material(graph, tensors, dropped)


def _table_container(path: Path, storage: str) -> Path:
    """A one-node container holding a `[V, H]` embedding table as `storage` (`i4` g32 imported from
    a grid, or `i8` per row) — the lender's main table as a container stores it."""
    table = _grid_tensor(TINY_VOCAB, TINY_BACKBONE, torch.Generator().manual_seed(7))
    graph = IrGraph(
        symbols=[],
        inputs=[IrInput(name="tok", dtype="i32", shape=[1, 1])],
        outputs=["e"],
        initializers={"table": IrInitializer(tensor=TINY_SHARED_TENSOR, storage=IrStorage("f32"))},
        values={
            "table": IrValue(dtype="f32", shape=[TINY_VOCAB, TINY_BACKBONE]),
            "e": IrValue(dtype="f32", shape=[1, 1, TINY_BACKBONE]),
        },
        states={},
        nodes=[IrNode(op="embedding", ins=["table", "tok"], outs=["e"], attrs={"padding_idx": -1})],
    )
    if storage == "i4":
        fixed = {TINY_SHARED_TENSOR: qat.import_grid(table.clone(), "table")}
        options: dict[str, Any] = {"weight_dtype": "f32", "fixed_weights": fixed}
        tensors = {TINY_SHARED_TENSOR: torch.empty(table.shape, device="meta")}
    else:
        scale = channel_scale(table, 0)
        options = {"weight_dtype": "i8", "weight_scales": {TINY_SHARED_TENSOR: scale}}
        tensors = {TINY_SHARED_TENSOR: quantize_to_int8(table, scale).to(torch.float32) * scale}
    publish_model(
        path / gx.MODEL_FILE,
        graph,
        tensors,
        provenance=Provenance(license="apache-2.0"),
        graph_name="model",
        **options,
    )
    return path / gx.MODEL_FILE


def _lender_document() -> dict[str, Any]:
    """The lender's IR v2 graph document as `gemma4.export_drafter.lender_slots` reads it."""
    lender = _tiny_lender()
    return {
        "initializers": {TINY_SHARED_TENSOR: {}},
        "values": {TINY_SHARED_TENSOR: {"dtype": "f32", "shape": [TINY_VOCAB, TINY_BACKBONE]}},
        "states": {name: {"dtype": "f32", "shape": shape} for name, shape in lender.shapes.items()},
        "nodes": [{"op": "embedding", "ins": [TINY_SHARED_TENSOR, "tok"], "outs": ["e"]}],
    }


def _lender_target_config() -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_BACKBONE,
        num_hidden_layers=2,
        layer_types=list(TINY_LAYER_TYPES),
        num_kv_shared_layers=0,
    )


class TestTheBorrowedTable:
    """The borrowed table's declaration matches the QAT target's i4 g32 table."""

    @pytest.mark.parametrize("storage", ["i4", "i8"])
    def test_the_lender_s_layout_and_group_are_read_from_its_container(
        self, tmp_path, storage: str
    ) -> None:
        container = open_container(_table_container(tmp_path, storage))
        layout = read_layouts(container)[TINY_SHARED_TENSOR]
        group = qd.stored_group_size(container, TINY_SHARED_TENSOR)

        assert layout == storage
        if storage == "i4":
            assert group == qd.GROUP_SIZE
            qd.assert_shared_table(TINY_SHARED_TENSOR, layout, group)
        else:
            with pytest.raises(qd.QatDrafterError, match="i4 g32"):
                qd.assert_shared_table(TINY_SHARED_TENSOR, layout, group)

    @pytest.mark.parametrize(("layout", "group"), [("i4", 64), ("i4", None), (None, None)])
    def test_another_grid_or_no_table_is_refused(self, layout, group) -> None:
        with pytest.raises(qd.QatDrafterError, match=TINY_SHARED_TENSOR):
            qd.assert_shared_table(TINY_SHARED_TENSOR, layout, group)

    def test_the_qat_declaration_accepts_an_i4_lender_and_refuses_the_regular_i8_one(self) -> None:
        def read(layout: str, storage: IrStorage | None) -> drafter.LenderSlots:
            return drafter.lender_slots(
                _lender_document(),
                {TINY_SHARED_TENSOR: layout},
                SimpleNamespace(),
                _lender_target_config(),
                shared_storage=storage,
            )

        assert read("i4", qd.SHARED_STORAGE).shared_tensor == TINY_SHARED_TENSOR
        with pytest.raises(AssertionError, match="'i8'"):
            read("i8", qd.SHARED_STORAGE)

    def test_the_lender_record_must_name_the_pinned_qat_weights(self, tmp_path) -> None:
        pins = sources.qat_pins("e2b")
        files = {
            name: {"bytes": 1, "sha256": digest} for name, digest in pins.weights.sha256.items()
        }
        record = tmp_path / provenance.REFERENCE_FILE
        record.write_text(json.dumps({"checkpoint": {"files": files}}), encoding="utf-8")

        qd.assert_lender_checkpoint(tmp_path, pins)
        with pytest.raises(qd.QatDrafterError, match="is not from"):
            qd.assert_lender_checkpoint(tmp_path, sources.qat_pins("e4b"))
        record.unlink()
        with pytest.raises(qd.QatDrafterError, match="export the QAT product series first"):
            qd.assert_lender_checkpoint(tmp_path, pins)

    def test_the_drafter_s_copy_must_equal_the_target_reference_s_table(self) -> None:
        wrapper = _tiny_wrapper()
        qd.import_drafter(wrapper)
        target = torch.nn.Module()
        target.model = torch.nn.Module()
        target.model.model = torch.nn.Module()
        target.model.model.embed_tokens = torch.nn.Embedding(TINY_VOCAB, TINY_BACKBONE)
        target.model.model.embed_tokens.weight.data = (
            wrapper.get_parameter(qd.TARGET_EMBED_WEIGHT).detach().clone()
        )

        qd.assert_borrowed_table(wrapper, target)
        target.model.model.embed_tokens.weight.data[0, 0] += 1.0
        with pytest.raises(qd.QatDrafterError, match="borrowed table"):
            qd.assert_borrowed_table(wrapper, target)


def _snapshot(root: Path, files: dict[str, bytes]) -> tuple[Path, dict[str, str]]:
    root.mkdir(parents=True)
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    return root, {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()}


class TestTheRecord:
    """`reference.json`: the target by its two pinned sources, the drafter by its pin."""

    @pytest.fixture
    def stand_ins(self, tmp_path):
        weights_dir, weights = _snapshot(
            tmp_path / "weights", {"model.safetensors": b"w", "config.json": b"{}"}
        )
        tokenizer_dir, tokenizer = _snapshot(tmp_path / "tokenizer", {"tokenizer.json": b"t"})
        drafter_dir, drafted = _snapshot(
            tmp_path / "drafter", {"model.safetensors": b"d", "config.json": b"{ }"}
        )
        pins = sources.ModelPins(
            weights=sources.HubPin(repo="test/w", revision="a" * 40, sha256=weights),
            tokenizer=sources.HubPin(repo="test/t", revision="b" * 40, sha256=tokenizer),
        )
        pin = sources.HubPin(repo="test/d", revision="c" * 40, sha256=drafted)
        return weights_dir, tokenizer_dir, pins, drafter_dir, pin

    def test_it_names_both_sources_and_the_drafter_pin(self, stand_ins, tmp_path) -> None:
        weights_dir, tokenizer_dir, pins, drafter_dir, pin = stand_ins

        record = qd.drafter_record(
            tmp_path / "gemma4-e2b-qat-drafter",
            weights_dir,
            tokenizer_dir,
            pins,
            drafter_dir,
            pin,
            {"case": {"tokens_per_cycle": 2.0}},
        )

        assert record["series"] == "gemma4-e2b-qat-drafter"
        assert record["checkpoint"]["dir"] == pins.label
        assert sorted(record["checkpoint"]["sources"]) == ["tokenizer", "weights"]
        assert record["drafter"]["repo"] == "test/d"
        assert record["drafter"]["revision"] == "c" * 40
        assert sorted(record["drafter"]["files"]) == sorted(qd.DRAFTER_FILES)
        assert record["goldens"] == {"case": {"tokens_per_cycle": 2.0}}
        assert record["suppressTokens"] == []

    def test_it_names_the_ids_the_target_continuation_was_taken_with_masked(
        self, stand_ins, tmp_path
    ) -> None:
        """12B's `generation_config.json` suppresses two ids; the host must mask them too to
        reproduce the goldens' `tokens`."""
        weights_dir, tokenizer_dir, pins, drafter_dir, pin = stand_ins

        record = qd.drafter_record(
            tmp_path / "gemma4-12b-qat-drafter",
            weights_dir,
            tokenizer_dir,
            pins,
            drafter_dir,
            pin,
            {},
            suppress=(258882, 258883),
        )

        assert record["suppressTokens"] == [258882, 258883]

    def test_a_drafter_that_is_not_the_pinned_one_fails(self, stand_ins, tmp_path) -> None:
        weights_dir, tokenizer_dir, pins, drafter_dir, pin = stand_ins
        (drafter_dir / "model.safetensors").write_bytes(b"other")

        with pytest.raises(qd.QatDrafterError, match="is not test/d@"):
            qd.drafter_record(
                tmp_path / "s", weights_dir, tokenizer_dir, pins, drafter_dir, pin, {}
            )

    def test_a_pin_naming_other_files_fails(self) -> None:
        pin = sources.HubPin(repo="test/d", revision="c" * 40, sha256={"model.safetensors": "0"})

        with pytest.raises(qd.QatDrafterError, match="names"):
            qd.assert_pinned_files({name: {"sha256": "0"} for name in qd.DRAFTER_FILES}, pin)


class TestE4BShapes:
    """The real configs (the hub cache — SKIP without it): the facts the E4B generalisation reads
    from upstream instead of hard-coding."""

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_the_drafter_matches_its_target_s_width_and_kv_heads(self, model: str) -> None:
        transformers = pytest.importorskip("transformers")
        drafter_dir, target_dir = sources.qat_drafter_dir(model), sources.qat_checkpoint_dir(model)
        if not all((path / "config.json").is_file() for path in (drafter_dir, target_dir)):
            pytest.skip(f"{model}'s QAT drafter or target config is not in the hub cache")
        backbone, kv_heads, owners = REAL_WIDTHS[model]

        config = transformers.AutoConfig.from_pretrained(drafter_dir)
        text = config.get_text_config()
        target = gx.load_text_config(target_dir)

        assert int(config.backbone_hidden_size) == int(target.hidden_size) == backbone
        assert int(text.num_key_value_heads) == int(target.num_key_value_heads) == kv_heads
        assert (int(text.head_dim), int(text.global_head_dim)) == (
            int(target.head_dim),
            int(target.global_head_dim),
        )
        assert int(text.sliding_window) == int(target.sliding_window)
        assert decode.kv_owner_layers(target) == owners

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_every_drafter_linear_splits_into_grid_groups(self, model: str) -> None:
        transformers = pytest.importorskip("transformers")
        directory = sources.qat_drafter_dir(model)
        if not (directory / "config.json").is_file():
            pytest.skip(f"{model}'s QAT drafter is not in the hub cache ({directory})")
        backbone = REAL_WIDTHS[model][0]
        config = transformers.AutoConfig.from_pretrained(directory)
        config.use_ordered_embeddings = False

        with torch.device("meta"):
            model_ = transformers.Gemma4AssistantForCausalLM(config)

        linears = {
            name: module
            for name, module in model_.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        assert len(linears) == 23
        assert all(module.in_features % qd.GROUP_SIZE == 0 for module in linears.values())
        assert model_.pre_projection.in_features == 2 * backbone
        assert model_.post_projection.out_features == backbone


class TestTheCommandLine:
    def test_it_offers_the_models_with_a_qat_drafter(self, capsys) -> None:
        parser = qd.build_parser()

        assert parser.parse_args(["--model", "e4b"]).model == "e4b"
        assert parser.parse_args(["--model", "12b"]).model == "12b"
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "31b"])
        assert "--model" in capsys.readouterr().err

    def test_the_defaults_are_the_pinned_sources_and_the_qat_lender(self) -> None:
        options = gx.resolve_series_args(qd.build_parser().parse_args(["--model", "e4b"]))

        assert options["model_dir"] == sources.qat_checkpoint_dir("e4b")
        assert options["tokenizer_dir"] == sources.qat_tokenizer_dir("e4b")
        assert options["drafter_dir"] == sources.qat_drafter_dir("e4b")
        assert options["out"].name == gemma4_series_name("e4b", qd.QAT_DRAFTER_SUFFIX)
        assert options["product_dir"].name == gemma4_series_name("e4b", qat.QAT_PRODUCT_SUFFIX)
        assert options["device"] == "cpu"
        assert "sym_max" not in options

    def test_every_option_is_a_keyword_of_the_export(self) -> None:
        """`main` hands the options to `export_series` by name — a knob it does not take would
        fail only at the end of the parse of a real run."""
        options = gx.resolve_series_args(qd.build_parser().parse_args([]))
        options.pop("model_dir")
        options.pop("out")
        accepted = set(inspect.signature(qd.export_series).parameters)

        assert set(options) <= accepted


class TestTheRegularDrafterIsUnchanged:
    """What the QAT line generalised keeps its old value on the regular path."""

    def test_the_regular_shared_declaration_is_still_i8(self) -> None:
        assert drafter.drafter_shared_storage(None) == IrStorage(dtype="i8")

    def test_the_regular_drafter_still_loads_the_regular_e2b_assistant(self) -> None:
        source = inspect.signature(drafter.load_drafter).parameters["source"]

        assert source.default == drafter.ASSISTANT_REPO == "google/gemma-4-E2B-it-assistant"
        assert drafter.DEFAULT_OUT_DIR.name == "gemma4-e2b-drafter"

    def test_the_regular_lender_check_still_refuses_an_i4_table(self) -> None:
        with pytest.raises(AssertionError, match="'i4'"):
            drafter.lender_slots(
                _lender_document(),
                {TINY_SHARED_TENSOR: "i4"},
                SimpleNamespace(),
                _lender_target_config(),
            )

    def test_the_regular_shared_declaration_defaults_to_i8(self, tmp_path) -> None:
        wrapper = _tiny_wrapper()
        text = wrapper.drafter.config.get_text_config()
        lender = _tiny_lender()
        graph, tensors = drafter.export_module_with_attention(
            wrapper, drafter.example_inputs(wrapper, decode.rope_specs(text))
        )
        surgical = drafter.to_external_states_form(
            graph, drafter.external_states_plan(graph, text, lender)
        )

        shared_graph, _, name = drafter.share_target_embedding(
            surgical, tensors, lender, _target_config()
        )

        assert shared_graph.initializers[name].storage == IrStorage(dtype="i8")

    def test_the_regular_goldens_still_run_against_the_regular_product_rounding(
        self, monkeypatch
    ) -> None:
        calls: list[tuple[str, Any]] = []
        target = object()
        wrapper: Any = object()

        def record(name: str, result: Any = None) -> Callable[..., Any]:
            def call(*args: Any, **kwargs: Any) -> Any:
                calls.append((name, args))
                return result

            return call

        monkeypatch.setattr(drafter.product, "load_wrapper", record("load", target))
        monkeypatch.setattr(drafter.one_shot, "quantize_wrapper", record("i8"))
        monkeypatch.setattr(drafter, "goldens_against", record("goldens", {"ok": True}))

        summary = drafter.write_goldens(wrapper, Path("target"), (), Path("out"), 7, "cpu")

        assert summary == {"ok": True}
        assert [name for name, _ in calls] == ["load", "i8", "goldens"]
        assert calls[0][1] == (Path("target"),)
        assert calls[1][1] == (target,)
        assert calls[2][1] == (wrapper, target, (), Path("out"), 7, "cpu")
