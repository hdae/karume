"""`gemma4/export_moe.py` (26B A4B: the `gemma4_text` class with the expert block) on a tiny
random checkpoint (`moe_fixture` — two shards plus their index, every linear, the embedding and
every stacked expert weight on a Q4_0 grid).

What is fixed here are the properties that would otherwise pass silently:

- the experts run through `karume` after the model is built (not upstream's default), and match
  upstream's eager experts: the same routing ids, logits within F32 tolerance;
- the sharded checkpoint reads what one file would, a stray key or an index that disagrees with
  its shards fails, and a rank-3 expert weight imported in blocks of experts is its whole-tensor
  import as `[E·out, in]`;
- the shape-only trace is the real-weight trace, and the container written from it is the
  real-weight trace's **byte for byte** (a constant a handler derives from a parameter would ship
  the shape-only tensor's uninitialised bytes);
- every parameter of rank ≥ 2 is imported (an expert left out fails with its key, before any
  read), the experts are stored as i4f16 and decode to codes × f16(step), and the reference holds
  exactly what the container decodes to;
- the reference equals an F32 load whose weights are replaced by their imported values;
- the form check counts one router lookup per layer, holds every expert_linear to a topk over the
  experts, and keeps 12B's soft inputs;
- the cut reference's layers are the full reference's first layers bit for bit, the cut goldens
  carry what the GPU comparison reads, and the check against the full series' cut points catches a
  different model;
- the diagnostic routing graph adds each router's topk index after logits and hidden;
- the series run publishes the container, the goldens, the cut points and a record naming the
  experts implementation and both shards.

transformers is required (`importorskip`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors import safe_open
from torch import nn

from _shared import decode_series as shared
from _shared.container_read import read_stored
from gemma4 import experts as routed
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_moe as moe
from gemma4 import export_product as product
from gemma4 import export_unified as unified
from gemma4 import provenance, sources
from gemma4.tests import moe_fixture as fx
from karume.container import container_parts
from karume.emit import f16_step_table, unpack_int4
from karume.ir import IrNode
from karume.ops import EMBEDDING_OP, EXPERT_LINEAR_OP
from karume.quantize import DEFAULT_GROUP_SIZE, dequantize_int4

transformers = pytest.importorskip("transformers")

#: The cases a series run uses (names from 12B's set — the greedy filter and the first token
#: expectations key on them).
CASE_NAMES = ("capital-en", "capital-ja", "context-en", "window-en")
GREEDY_NAMES = ("capital-en", "capital-ja", "window-en")
#: The stand-in window case's length: past the tiny ring and past the traced chunk range.
WINDOW_CASE_LENGTH = fx.TINY_SYM_MAX + 6


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory) -> Path:
    return fx.write_checkpoint(tmp_path_factory.mktemp("tiny-26b"))


@pytest.fixture
def config(checkpoint_dir) -> Any:
    return moe.load_config(checkpoint_dir)


@pytest.fixture
def small_blocks(monkeypatch) -> None:
    """Blocks of 16 rows of a 64-wide row, so an expert weight spans several blocks of experts and
    every linear several blocks of rows."""
    monkeypatch.setattr(unified, "CHUNK_BYTES", 16 * fx.HIDDEN * 4)


def _ids(length: int, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, fx.VOCAB, (1, length), generator=generator, dtype=torch.int64)


def _call(wrapper: nn.Module, ids: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    specs = decode.rope_specs(wrapper.model.config)
    with torch.no_grad():
        return wrapper(
            ids,
            *unified.no_soft_inputs(ids, fx.HIDDEN),
            *decode.rope_args(specs, decode.positions_for(ids)),
            decode.last_rows_for(ids, rows),
        )


def naive_wrapper(checkpoint_dir: Path, config: Any) -> unified.UnifiedProductWrapper:
    """The F32-loaded wrapper (every checkpoint tensor widened; the experts as F32 parameters)."""
    model = unified.meta_model(config)
    checkpoint = moe.open_checkpoint(checkpoint_dir)
    state = {
        key: checkpoint.read(unified.checkpoint_key(f"model.{key}"))
        for key in model.state_dict()
        if key != "lm_head.weight"
    }
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, assign=True, strict=True)
    model.tie_weights()
    one_shot.assert_materialized(model)
    decode.swap_rope_inputs(model)
    return unified.UnifiedProductWrapper(model).eval()


def imported_wrapper(checkpoint_dir: Path, config: Any) -> unified.UnifiedProductWrapper:
    """The F32-loaded wrapper with every linear and the embedding replaced by codes · step of its
    whole-tensor grid import, and every expert weight by codes · f16(step) (the writer's rounding)
    of its whole-tensor `[E·out, in]` import — the model the container stores, built without row
    blocks, packed bytes or the reference's holders."""
    wrapper = naive_wrapper(checkpoint_dir, config)
    keys = unified.trace_keys(wrapper)
    for fqn in (unified.EMBED_KEY, *sorted(keys.int4)):
        module = wrapper.get_submodule(fqn.removesuffix(".weight"))
        codes, steps = unified.grid_import(module.weight.data, DEFAULT_GROUP_SIZE, fqn)
        module.weight.data = dequantize_int4(codes, steps)
    for fqn in sorted(keys.experts):
        owner, _, name = fqn.rpartition(".")
        parameter = getattr(wrapper.get_submodule(owner), name)
        experts, rows, cols = parameter.shape
        codes, steps = unified.grid_import(
            parameter.data.reshape(experts * rows, cols), DEFAULT_GROUP_SIZE, fqn
        )
        from karume.emit import pack_int4

        packed = pack_int4(codes).reshape(experts * rows, cols // 2)
        table = f16_step_table(packed, steps, fqn)[: steps.numel()].reshape(steps.shape)
        parameter.data = dequantize_int4(codes, table.to(torch.float32)).reshape(
            experts, rows, cols
        )
    return wrapper


@pytest.fixture
def traced(config):
    wrapper = unified.shape_only_model(config)
    graph, tensors = unified.trace(wrapper, _ids(fx.WINDOW + 7), fx.TINY_SYM_MAX)
    return SimpleNamespace(graph=graph, tensors=tensors, keys=unified.trace_keys(wrapper))


def _with_ins(node: IrNode, ins: list[str]) -> IrNode:
    return IrNode(op=node.op, ins=ins, outs=node.outs, attrs=node.attrs, states=node.states)


# ---------------------------------------------------------------------------
# Config and experts
# ---------------------------------------------------------------------------


class TestConfig:
    def test_it_reads_the_moe_text_config_with_karume_experts(self, config) -> None:
        assert config.model_type == unified.MOE_TEXT_MODEL_TYPE
        assert config.enable_moe_block
        assert config._attn_implementation == one_shot.ATTENTION_NAME
        assert config._experts_implementation == routed.EXPERTS_IMPLEMENTATION

    def test_the_built_model_keeps_the_karume_experts(self, config) -> None:
        """Upstream resolves the implementation at construction; the built model and every experts
        module still name `karume` (an unset one would have become `grouped_mm`)."""
        model = unified.meta_model(config)

        experts = [
            module for module in model.modules() if type(module).__name__ == "Gemma4TextExperts"
        ]
        assert len(experts) == fx.LAYERS
        assert all(
            module.config._experts_implementation == routed.EXPERTS_IMPLEMENTATION
            for module in experts
        )

    def test_a_config_without_the_karume_experts_is_refused_before_construction(
        self, checkpoint_dir
    ) -> None:
        config = one_shot.load_text_config(checkpoint_dir)
        config._attn_implementation = one_shot.ATTENTION_NAME

        with pytest.raises(routed.ExpertsError, match="not 'karume'"):
            unified.meta_model(config)

    def test_a_truncation_that_ends_on_a_sliding_layer_fails_loudly(self, checkpoint_dir) -> None:
        with pytest.raises(unified.UnifiedExportError, match="full attention"):
            moe.load_config(checkpoint_dir, layers=fx.CUT + 1)

    def test_a_config_without_the_expert_block_is_refused(self, tmp_path) -> None:
        text = fx.text_config(enable_moe_block=False)
        transformers.Gemma4Config(text_config=text.to_dict()).save_pretrained(tmp_path)

        with pytest.raises(moe.MoeExportError, match="expert block"):
            moe.load_config(tmp_path)

    def test_12b_s_script_does_not_take_26b(self) -> None:
        """Adding 26b to the line tables does not route it into `export_unified`'s CLI."""
        assert "26b" not in unified.MODELS
        assert moe.MODELS == ("26b",)


def upstream_model(checkpoint_dir: Path, config: Any, experts: str) -> nn.Module:
    """Upstream `Gemma4ForCausalLM` with the checkpoint widened to F32, its own masks, eager
    attention and the experts implementation `experts`."""
    upstream_config = copy.deepcopy(config)
    upstream_config._attn_implementation = "eager"
    upstream_config._experts_implementation = experts
    model = transformers.Gemma4ForCausalLM(upstream_config).eval()
    checkpoint = moe.open_checkpoint(checkpoint_dir)
    state = {
        key: checkpoint.read(unified.checkpoint_key(f"model.{key}"))
        for key in model.state_dict()
        if key != "lm_head.weight"
    }
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    return model


def _routing(model: nn.Module, ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    picks: list[torch.Tensor] = []
    handles = [
        layer.router.register_forward_hook(lambda _m, _a, out: picks.append(out[2]))
        for layer in model.model.layers
    ]
    try:
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    return logits, picks


class TestUpstreamExperts:
    #: F32 bound on the softcapped logits (|logits| ≤ 30); the sums over the picks differ in order
    #: only (upstream's eager adds in expert-id order — ADR 0145 decision 2).
    BOUND = 2e-5

    @pytest.mark.parametrize("length", [3, fx.WINDOW + 1, 3 * fx.WINDOW + 5])
    def test_karume_experts_match_upstream_s_eager_experts(
        self, checkpoint_dir, config, length
    ) -> None:
        """The same routing ids in every layer and the logits within F32 tolerance."""
        ids = _ids(length, seed=length)
        ours, our_picks = _routing(upstream_model(checkpoint_dir, config, "karume"), ids)
        theirs, their_picks = _routing(upstream_model(checkpoint_dir, config, "eager"), ids)

        assert len(our_picks) == fx.LAYERS
        for got, want in zip(our_picks, their_picks, strict=True):
            assert torch.equal(got, want)
        assert float((ours - theirs).abs().max()) <= self.BOUND

    def test_the_export_wrapper_matches_upstream_s_eager_forward(
        self, checkpoint_dir, config
    ) -> None:
        """The export's masks, host RoPE and K = V wiring against upstream's own, on both sides of
        the window."""
        length = fx.RING + 3
        ids = _ids(length, seed=5)
        logits, _ = _call(naive_wrapper(checkpoint_dir, config), ids, length)
        expected, _ = _routing(upstream_model(checkpoint_dir, config, "eager"), ids)

        assert float((logits - expected).abs().max()) <= self.BOUND

    def test_another_experts_layout_is_refused(self, checkpoint_dir, config) -> None:
        model = upstream_model(checkpoint_dir, config, "karume")
        model.model.layers[0].experts.is_transposed = True

        with pytest.raises(routed.ExpertsError, match="is_transposed"):
            _routing(model, _ids(4))


# ---------------------------------------------------------------------------
# Checkpoint and grid import
# ---------------------------------------------------------------------------


class TestShardedCheckpoint:
    def test_the_shards_read_what_one_file_would(self, checkpoint_dir, tmp_path) -> None:
        single = fx.write_checkpoint(tmp_path / "single", sharded=False)
        sharded = moe.open_checkpoint(checkpoint_dir)
        whole = moe.open_checkpoint(single)

        assert sharded.names() == whole.names()
        assert sharded.files == (
            unified.CHECKPOINT_INDEX,
            *fx.SHARDS,
        )
        assert whole.files == (one_shot.CHECKPOINT_FILE,)
        for key in sorted(sharded.names()):
            assert torch.equal(sharded.read(key, None), whole.read(key, None)), key

    def test_a_stray_key_fails_loudly(self, tmp_path) -> None:
        directory = fx.write_checkpoint(
            tmp_path, extra={"model.embed_audio.weight": torch.zeros(2, dtype=torch.bfloat16)}
        )

        with pytest.raises(unified.UnifiedExportError, match="outside the text decoder"):
            moe.open_checkpoint(directory)

    def test_12b_s_ignored_prefixes_do_not_cover_the_vision_tower(self, checkpoint_dir) -> None:
        with pytest.raises(unified.UnifiedExportError, match="vision_tower"):
            unified.Checkpoint(checkpoint_dir)

    def test_an_index_that_disagrees_with_its_shards_fails_loudly(
        self, checkpoint_dir, tmp_path
    ) -> None:
        directory = tmp_path / "bent"
        shutil.copytree(checkpoint_dir, directory)
        index = json.loads((directory / unified.CHECKPOINT_INDEX).read_text(encoding="utf-8"))
        key = next(key for key, name in index["weight_map"].items() if name == fx.SHARDS[0])
        index["weight_map"][key] = fx.SHARDS[1]
        (directory / unified.CHECKPOINT_INDEX).write_text(json.dumps(index), encoding="utf-8")

        with pytest.raises(unified.UnifiedExportError, match="maps"):
            moe.open_checkpoint(directory)


#: A stacked expert weight of the tiny checkpoint.
GATE_UP = "model.language_model.layers.1.experts.gate_up_proj"


class TestExpertGrid:
    def test_blocks_of_experts_are_the_whole_tensor_import(self, checkpoint_dir, small_blocks):
        """`[E, out, in]` imported a few experts at a time gives the bytes and steps of the whole
        tensor imported as `[E·out, in]` — and the packed rows are `[E·out, in/2]`."""
        checkpoint = moe.open_checkpoint(checkpoint_dir)
        view = unified.grid_view(checkpoint, GATE_UP, DEFAULT_GROUP_SIZE, experts=True)
        assert view.units > unified._block_rows(view.rows_per_unit * view.cols)

        fixed = unified.quantize_int4_rows(checkpoint, GATE_UP, experts=True)

        weight = checkpoint.read(GATE_UP).reshape(view.rows, view.cols)
        codes, steps = unified.grid_import(weight, DEFAULT_GROUP_SIZE, GATE_UP)
        assert fixed.dtype == "i4f16"
        assert list(fixed.packed.shape) == [fx.EXPERTS * 2 * fx.MOE_INTERMEDIATE, fx.HIDDEN // 2]
        assert torch.equal(unpack_int4(fixed.packed, codes.shape), codes)
        assert torch.equal(fixed.scale, steps)

    def test_a_rank_3_weight_is_refused_unless_it_is_an_expert_weight(self, checkpoint_dir) -> None:
        checkpoint = moe.open_checkpoint(checkpoint_dir)

        with pytest.raises(unified.UnifiedExportError, match="rank 3"):
            unified.quantize_int4_rows(checkpoint, GATE_UP)

    def test_an_expert_weight_off_the_grid_fails_naming_its_row(self, tmp_path) -> None:
        bent = fx.checkpoint_state()[GATE_UP].clone()
        bent[2, 5, 3] = bent[2, 5, 3] * 1.3 + 0.01
        directory = fx.write_checkpoint(tmp_path, extra={GATE_UP: bent})

        with pytest.raises(unified.UnifiedExportError, match=rf"row {2 * 64 + 5}\b"):
            unified.quantize_int4_rows(moe.open_checkpoint(directory), GATE_UP, experts=True)


# ---------------------------------------------------------------------------
# Trace, material and container
# ---------------------------------------------------------------------------


class TestShapeOnlyTrace:
    def test_the_graph_is_the_real_weight_trace(self, checkpoint_dir, config) -> None:
        ids = _ids(fx.WINDOW + 7)
        shape_only, shape_tensors = unified.trace(
            unified.shape_only_model(config), ids, fx.TINY_SYM_MAX
        )
        real, real_tensors = unified.trace(
            naive_wrapper(checkpoint_dir, config), ids, fx.TINY_SYM_MAX
        )

        assert shape_only.to_json() == real.to_json()
        assert sorted(shape_tensors) == sorted(real_tensors)

    def test_the_container_of_the_shape_only_trace_is_the_real_trace_s_byte_for_byte(
        self, checkpoint_dir, config, tmp_path
    ) -> None:
        """Every tensor the writer takes — handler-made constants included — is the same bytes
        whether the trace held the weights or uninitialised tensors of their shapes: a constant
        derived from a parameter (a folded reshape of the router's scale, say) would ship the
        shape-only tensor's garbage."""
        checkpoint = moe.open_checkpoint(checkpoint_dir)
        ids = _ids(fx.WINDOW + 7)

        def written(name: str, wrapper: nn.Module) -> list[bytes]:
            graph, tensors = unified.trace(wrapper, ids, fx.TINY_SYM_MAX)
            keys = unified.trace_keys(wrapper)
            material = unified.prepare_material(graph, tensors, keys, checkpoint)
            path = tmp_path / name / "model.krm"
            unified.write_container(graph, material, path, None)
            return [part.read_bytes() for part in container_parts(path)]

        shape_only = unified.shape_only_model(config)
        for parameter in shape_only.parameters():
            parameter.data.fill_(float("nan"))  # what uninitialised bytes may hold

        assert written("shape-only", shape_only) == written(
            "real", naive_wrapper(checkpoint_dir, config)
        )


class TestMaterial:
    def test_every_rank_2_or_3_parameter_is_imported(self, traced) -> None:
        assert len(traced.keys.int4) == fx.INT4_LINEARS
        assert len(traced.keys.experts) == fx.EXPERT_WEIGHTS
        unified.assert_fixed_matrices(traced.keys)

    def test_an_expert_left_out_of_the_imported_set_fails_with_its_key_before_any_read(
        self, checkpoint_dir, traced, monkeypatch
    ) -> None:
        left_out = sorted(traced.keys.experts)[0]
        keys = unified.TraceKeys(
            persistent=traced.keys.persistent,
            derived=traced.keys.derived,
            int4=traced.keys.int4,
            experts=traced.keys.experts - {left_out},
            matrices=traced.keys.matrices,
        )
        reads: list[str] = []
        monkeypatch.setattr(
            unified, "quantize_int4_rows", lambda *args, **kwargs: reads.append(args[1])
        )

        with pytest.raises(unified.UnifiedExportError, match=left_out):
            unified.prepare_material(
                traced.graph, traced.tensors, keys, moe.open_checkpoint(checkpoint_dir)
            )
        assert reads == []

    def test_the_experts_are_stored_as_i4f16_and_decode_to_codes_times_the_f16_step(
        self, checkpoint_dir, config, traced, small_blocks, tmp_path
    ) -> None:
        checkpoint = moe.open_checkpoint(checkpoint_dir)
        material = unified.prepare_material(traced.graph, traced.tensors, traced.keys, checkpoint)
        unified.write_container(traced.graph, material, tmp_path / "model.krm", None)
        stored = read_stored(tmp_path / "model.krm")
        reference = moe.reference_wrapper(config, checkpoint, traced.keys, material.fixed)

        storage = [entry.layout for entry in stored.values()]
        assert storage.count("i4f16") == fx.EXPERT_WEIGHTS
        assert storage.count("i4") == fx.INT4_LINEARS + 1
        for key in sorted(traced.keys.experts):
            entry = stored[key]
            experts, rows, cols = entry.shape
            codes = unpack_int4(
                torch.frombuffer(bytearray(entry.payload), dtype=torch.uint8),
                (experts * rows, cols),
            )
            halves = torch.frombuffer(bytearray(entry.scale), dtype=torch.float16)
            steps = halves[: experts * rows * cols // DEFAULT_GROUP_SIZE].to(torch.float32)
            decoded = dequantize_int4(codes, steps.reshape(experts * rows, -1))
            weight = checkpoint.read(unified.checkpoint_key(key)).reshape(experts * rows, cols)
            want_codes, want_steps = unified.grid_import(weight, DEFAULT_GROUP_SIZE, key)
            assert torch.equal(codes, want_codes), key
            assert torch.equal(steps.reshape(want_steps.shape), want_steps.half().float()), key
            owner, _, name = key.removeprefix("model.").rpartition(".")
            holder = getattr(reference.model.get_submodule(owner), name)
            held = torch.cat([holder.expert(expert) for expert in range(experts)])
            assert torch.equal(decoded, held), key
        for key in sorted(traced.keys.int4):
            entry = stored[key]
            codes = unpack_int4(
                torch.frombuffer(bytearray(entry.payload), dtype=torch.uint8), entry.shape
            )
            steps = torch.frombuffer(bytearray(entry.scale), dtype=torch.float32)
            values = dequantize_int4(codes, steps.reshape(entry.shape[0], -1))
            module = reference.get_submodule(key.removesuffix(".weight"))
            assert torch.equal(values, module.weight), key


class TestReference:
    def test_the_reference_equals_the_f32_load_with_imported_weights(
        self, checkpoint_dir, config, traced, small_blocks
    ) -> None:
        """Packed bytes dequantised at use (the experts one per pick, through the op's own body)
        give the outputs of an F32 load whose weights are the imported values, bit for bit."""
        checkpoint = moe.open_checkpoint(checkpoint_dir)
        material = unified.prepare_material(traced.graph, traced.tensors, traced.keys, checkpoint)
        reference = moe.reference_wrapper(config, checkpoint, traced.keys, material.fixed)
        loaded = imported_wrapper(checkpoint_dir, config)
        ids = _ids(fx.RING + 3)

        for got, want in zip(_call(reference, ids, 3), _call(loaded, ids, 3), strict=True):
            assert torch.equal(got, want)

    def test_the_reference_dequantises_only_the_picked_experts(
        self, checkpoint_dir, config, traced, monkeypatch
    ) -> None:
        checkpoint = moe.open_checkpoint(checkpoint_dir)
        material = unified.prepare_material(traced.graph, traced.tensors, traced.keys, checkpoint)
        reference = moe.reference_wrapper(config, checkpoint, traced.keys, material.fixed)
        read: list[int] = []
        original = routed.PackedExperts.expert
        monkeypatch.setattr(
            routed.PackedExperts,
            "expert",
            lambda self, expert: read.append(expert) or original(self, expert),
        )

        _call(reference, _ids(2, seed=9), 1)

        # Two tokens, two picks each, two projections per layer: at most 4 experts per projection.
        assert 0 < len(read) <= 2 * fx.LAYERS * 2 * fx.TOP_K

    def test_the_reference_refuses_material_without_an_expert(
        self, checkpoint_dir, config, traced
    ) -> None:
        checkpoint = moe.open_checkpoint(checkpoint_dir)
        material = unified.prepare_material(traced.graph, traced.tensors, traced.keys, checkpoint)
        fixed = dict(material.fixed)
        fixed.pop(sorted(traced.keys.experts)[0])

        with pytest.raises(moe.MoeExportError, match="imported weights"):
            moe.reference_wrapper(config, checkpoint, traced.keys, fixed)


class TestForm:
    def test_the_written_graph_passes_the_product_form(
        self, checkpoint_dir, config, traced, tmp_path
    ) -> None:
        checkpoint = moe.open_checkpoint(checkpoint_dir)
        material = unified.prepare_material(traced.graph, traced.tensors, traced.keys, checkpoint)
        verified = unified.write_container(traced.graph, material, tmp_path / "model.krm", None)

        form = product.assert_ir_form_product(verified, config, moe._storage(traced.keys), None)

        assert [spec.name for spec in verified.inputs] == [
            decode.INPUT_IDS,
            product.SOFT_EMBEDS,
            product.SOFT_MASK,
            *decode.ROPE_INPUTS,
            decode.TOKEN_ONLY_LAST_ROW,
        ]
        assert form["embedding_nodes"] == 2 + fx.LAYERS
        assert form["router_lookups"] == fx.LAYERS
        assert form["storage"]["i4f16"] == fx.EXPERT_WEIGHTS
        assert unified.assert_shared_kv_projection(verified, config) == {
            "shared_kv_layers": len(fx.FULL_LAYERS)
        }
        assert moe.assert_expert_wiring(verified, config) == {
            "expert_linear_nodes": 2 * fx.LAYERS,
            "router_topk_nodes": fx.LAYERS,
        }

    def test_a_dropped_router_lookup_is_caught(self, config, traced) -> None:
        nodes = [node for node in traced.graph.nodes if node.op == EMBEDDING_OP]
        producers = unified._producers(traced.graph)
        lookup = next(
            node
            for node in nodes
            if producers.get(node.ins[0]) is not None
            and producers[node.ins[0]].op == "reshape"
            and producers[node.ins[0]].ins[0] in traced.graph.initializers
        )
        bent = SimpleNamespace(
            nodes=[node for node in traced.graph.nodes if node is not lookup],
            inputs=traced.graph.inputs,
            initializers=traced.graph.initializers,
            values=traced.graph.values,
        )

        with pytest.raises(AssertionError, match="router lookups"):
            product.router_scale_lookups(
                bent, config, [node for node in nodes if node is not lookup]
            )

    def test_a_model_without_the_expert_block_has_no_router_lookup(self, traced) -> None:
        plain = SimpleNamespace(enable_moe_block=False)
        embeddings = [node for node in traced.graph.nodes if node.op == EMBEDDING_OP]

        with pytest.raises(AssertionError, match="without the expert block"):
            product.router_scale_lookups(traced.graph, plain, embeddings)

    def test_a_value_that_is_the_rotated_key_is_caught(self, config, traced) -> None:
        """12B's K = V wiring check on the MoE graph: the second full layer's V given its rotated
        key (shapes and dtypes agree, so only the wiring tells them apart)."""
        nodes = list(traced.graph.nodes)
        index = [i for i, node in enumerate(nodes) if node.op == "attention"][fx.FULL_LAYERS[-1]]
        node = nodes[index]
        nodes[index] = _with_ins(node, [node.ins[0], node.ins[1], node.ins[1]])
        bent = SimpleNamespace(
            nodes=nodes, inputs=traced.graph.inputs, initializers=traced.graph.initializers
        )

        with pytest.raises(unified.UnifiedExportError, match="rotated"):
            unified.assert_shared_kv_projection(bent, config)

    def test_an_expert_index_that_is_not_a_topk_is_caught(self, config, traced) -> None:
        nodes = list(traced.graph.nodes)
        index = next(i for i, node in enumerate(nodes) if node.op == EXPERT_LINEAR_OP)
        node = nodes[index]
        nodes[index] = _with_ins(node, [node.ins[0], decode.TOKEN_ONLY_LAST_ROW, node.ins[2]])
        bent = SimpleNamespace(
            nodes=nodes,
            inputs=traced.graph.inputs,
            values=traced.graph.values,
            initializers=traced.graph.initializers,
        )

        with pytest.raises(moe.MoeExportError, match="not a topk index"):
            moe.assert_expert_wiring(bent, config)


# ---------------------------------------------------------------------------
# The cut model
# ---------------------------------------------------------------------------


def _reference(checkpoint_dir: Path, layers: int | None) -> tuple[Any, Any]:
    config = moe.load_config(checkpoint_dir, layers)
    wrapper = unified.shape_only_model(config)
    graph, tensors = unified.trace(wrapper, _ids(fx.WINDOW + 7), fx.TINY_SYM_MAX)
    keys = unified.trace_keys(wrapper)
    checkpoint = moe.open_checkpoint(checkpoint_dir)
    material = unified.prepare_material(graph, tensors, keys, checkpoint)
    return config, moe.reference_wrapper(config, checkpoint, keys, material.fixed)


class TestCut:
    def test_the_cut_layers_are_the_full_reference_s_first_layers_bit_for_bit(
        self, checkpoint_dir
    ) -> None:
        _, full = _reference(checkpoint_dir, None)
        _, cut = _reference(checkpoint_dir, fx.CUT)
        ids = _ids(fx.RING + 2)

        with moe.LayerRecorder(full, range(fx.CUT), routes=True) as whole:
            moe.teacher_forced(full, ids, 2)
        with moe.LayerRecorder(cut, range(fx.CUT), routes=True) as part:
            moe.teacher_forced(cut, ids, 2)

        for layer in range(fx.CUT):
            assert torch.equal(part.hidden[layer], whole.hidden[layer]), layer
            assert torch.equal(part.route_ids[layer], whole.route_ids[layer]), layer
            assert torch.equal(part.route_probs[layer], whole.route_probs[layer]), layer

    def test_the_cut_layers_a_series_records_end_on_full_layers(self, config) -> None:
        assert moe.cut_layers(config) == [layer for layer in fx.FULL_LAYERS[:-1]]

    def test_the_cut_goldens_carry_what_the_gpu_comparison_reads(
        self, checkpoint_dir, tmp_path
    ) -> None:
        config, cut = _reference(checkpoint_dir, fx.CUT)
        cases = (("capital-en", _ids(fx.WINDOW + 3)), ("window-en", _ids(fx.RING + 2, seed=4)))

        written, last, gaps = moe.write_cut_goldens(cut, cases, config, tmp_path)

        assert written == [f"{moe.CUT_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name, _ in cases]
        for (name, ids), file in zip(cases, written, strict=True):
            length = int(ids.shape[1])
            logits, hidden = moe.teacher_forced(cut, ids, moe.CUT_ROWS)
            with safe_open(str(tmp_path / file), framework="pt") as handle:
                assert handle.get_tensor(moe.ROWS_KEY).tolist() == list(
                    range(length - moe.CUT_ROWS, length)
                )
                assert torch.equal(handle.get_tensor(moe.LOGITS_KEY), logits)
                assert torch.equal(handle.get_tensor(moe.HIDDEN_KEY), hidden)
                layer_hidden = handle.get_tensor(moe.LAYER_HIDDEN_KEY)
                ids_ = handle.get_tensor(moe.ROUTE_IDS_KEY)
                probs = handle.get_tensor(moe.ROUTE_PROBS_KEY)
            assert list(layer_hidden.shape) == [fx.CUT, moe.CUT_ROWS, fx.HIDDEN]
            assert ids_.dtype is torch.int32
            assert list(ids_.shape) == [fx.CUT, length, fx.TOP_K + 1]
            assert list(probs.shape) == [fx.CUT, length, fx.TOP_K + 1]
            assert torch.equal(layer_hidden[-1], last[name][-moe.CUT_ROWS :])
            assert gaps[name] == pytest.approx(
                float((probs[..., fx.TOP_K - 1] - probs[..., fx.TOP_K]).min())
            )

    def test_the_check_against_the_full_series_catches_another_model(
        self, checkpoint_dir, tmp_path
    ) -> None:
        full_config, full = _reference(checkpoint_dir, None)
        config, cut = _reference(checkpoint_dir, fx.CUT)
        cases = (("capital-en", _ids(fx.WINDOW + 3)),)
        series = tmp_path / "gemma4-26b-product"
        series.mkdir()
        moe.write_cut_points(full, cases, full_config, series)
        _, last, _ = moe.write_cut_goldens(cut, cases, config, tmp_path)

        assert moe.assert_cut_consistency(last, cases, series, fx.CUT - 1)["rule"] == "torch.equal"
        bent = {name: hidden.clone() for name, hidden in last.items()}
        bent["capital-en"][0, 0] += 1e-6
        with pytest.raises(moe.MoeExportError, match="differs from the full reference"):
            moe.assert_cut_consistency(bent, cases, series, fx.CUT - 1)
        with pytest.raises(moe.MoeExportError, match=r"has no 'layer\.0'"):
            moe.assert_cut_consistency(last, cases, series, 0)


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True):
        return SimpleNamespace(ids=[len(text) % fx.VOCAB])

    def id_to_token(self, token: int) -> str:
        return f"<{token}>"

    def decode(self, tokens) -> str:
        return " ".join(f"<{token}>" for token in tokens)


@pytest.fixture
def tiny_series(monkeypatch):
    """Stand-ins for what the tiny random model cannot provide: the cases, the tokenizer, the
    margin floor and the first-token check."""
    cases = tuple(
        (
            name,
            _ids(
                WINDOW_CASE_LENGTH if name == unified.WINDOW_CASE else fx.WINDOW + 3 + 2 * index,
                seed=index,
            ),
        )
        for index, name in enumerate(CASE_NAMES)
    )
    assert [name for name, _ in moe.GOLDEN_CASES] == list(CASE_NAMES)
    monkeypatch.setattr(one_shot, "encode_cases", lambda model_dir, texts: cases)
    monkeypatch.setattr(one_shot, "load_tokenizer", lambda model_dir: _StubTokenizer())
    monkeypatch.setattr(decode, "MARGIN_FLOOR", 0.0)
    checked: list[dict[str, int]] = []

    def sanity(greedy, expected, labels):
        checked.append(dict(greedy))
        return {"stub": "ok"}

    monkeypatch.setattr(one_shot, "_sanity", sanity)
    return SimpleNamespace(cases=cases, checked=checked)


def _pin(repo: str, revision: str, directory: Path, names: tuple[str, ...]) -> sources.HubPin:
    return sources.HubPin(
        repo=repo,
        revision=revision,
        sha256={
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names
        },
    )


WEIGHTS_REVISION = "0" * 40
TOKENIZER_REVISION = "1" * 40
#: The weights' fingerprinted files: the index, both shards and the config.
WEIGHTS_FILES = (unified.CHECKPOINT_INDEX, *fx.SHARDS, "config.json")


@pytest.fixture
def two_sources(checkpoint_dir, tmp_path) -> SimpleNamespace:
    """26B's layout: the weights (two shards and their index) and their config in one directory,
    the tokenizer and the generation defaults in another, and pins for both."""
    weights = tmp_path / "sources" / "weights"
    tokenizer = tmp_path / "sources" / "tokenizer"
    for directory, names in (
        (weights, WEIGHTS_FILES),
        (tokenizer, ("tokenizer.json", "generation_config.json")),
    ):
        directory.mkdir(parents=True)
        for name in names:
            shutil.copy(checkpoint_dir / name, directory / name)
    pins = sources.ModelPins(
        weights=_pin("test/weights", WEIGHTS_REVISION, weights, WEIGHTS_FILES),
        tokenizer=_pin("test/tokenizer", TOKENIZER_REVISION, tokenizer, provenance.TOKENIZER_FILES),
    )
    return SimpleNamespace(weights=weights, tokenizer=tokenizer, pins=pins)


class TestExportSeries:
    def test_it_publishes_the_container_the_goldens_the_cut_points_and_the_record(
        self, tiny_series, two_sources, checkpoint_dir, config, small_blocks, tmp_path
    ) -> None:
        out_dir = tmp_path / "series" / "gemma4-26b-product"
        steps = 3

        summary = moe.export_series(
            two_sources.weights,
            out_dir,
            tokenizer_dir=two_sources.tokenizer,
            pins=two_sources.pins,
            sym_max=fx.TINY_SYM_MAX,
            steps=steps,
            checkpoint_label="weights and tokenizer",
        )

        names = sorted(path.name for path in out_dir.iterdir())
        assert names == sorted(
            [
                *(path.name for path in container_parts(out_dir / one_shot.MODEL_FILE)),
                *(f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name in GREEDY_NAMES),
                *(f"{moe.CUT_POINTS_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name in CASE_NAMES),
                provenance.REFERENCE_FILE,
            ]
        )
        record = json.loads((out_dir / provenance.REFERENCE_FILE).read_text(encoding="utf-8"))
        assert record["reference"]["experts"] == routed.EXPERTS_RECORD
        assert record["reference"]["suppressTokens"] == []
        assert record["reference"]["cutPoints"]["layers"] == list(fx.FULL_LAYERS[:-1])
        assert record["checkpoint"]["sources"]["weights"]["files"] == list(WEIGHTS_FILES)
        for pin in two_sources.pins.by_role().values():
            for name, digest in pin.sha256.items():
                assert record["checkpoint"]["files"][name]["sha256"] == digest, name
        assert summary["quantized"] == {
            "i4": fx.INT4_LINEARS + 1,
            "i4f16": fx.EXPERT_WEIGHTS,
            "i8": 0,
        }
        assert summary["kv"] == {"shared_kv_layers": len(fx.FULL_LAYERS)}
        assert [stage["name"] for stage in summary["stages"]] == [
            "trace",
            "import",
            "goldens",
            "write",
            "provenance",
        ]
        # The goldens are an independent greedy run of the F32 load with the imported weights.
        view = unified.GreedyView(
            imported_wrapper(checkpoint_dir, config), decode.rope_specs(config)
        )
        for name, ids in tiny_series.cases:
            expected, margins = shared.greedy_continuation(view, ids, steps)
            if name not in GREEDY_NAMES:
                assert tiny_series.checked[0][name] == expected[0], name
                continue
            golden = out_dir / f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}"
            with safe_open(str(golden), framework="pt") as handle:
                assert handle.get_tensor(shared.EXPECTED_KEY).tolist() == expected
                assert handle.get_tensor(shared.MARGIN_KEY).tolist() == pytest.approx(margins)

    def test_a_wrong_pin_of_a_shard_stops_the_run_before_anything_is_written(
        self, tiny_series, two_sources, tmp_path
    ) -> None:
        pins = two_sources.pins
        wrong = sources.ModelPins(
            weights=sources.HubPin(
                repo=pins.weights.repo,
                revision=pins.weights.revision,
                sha256={**pins.weights.sha256, fx.SHARDS[1]: "f" * 64},
            ),
            tokenizer=pins.tokenizer,
        )

        with pytest.raises(sources.SourcePinError, match=fx.SHARDS[1]):
            moe.export_series(
                two_sources.weights,
                tmp_path / "series" / "gemma4-26b-product",
                tokenizer_dir=two_sources.tokenizer,
                pins=wrong,
                sym_max=fx.TINY_SYM_MAX,
                steps=1,
            )
        assert not (tmp_path / "series").exists()


class TestExportCut:
    def test_it_publishes_the_cut_container_goldens_routing_graph_and_record(
        self, tiny_series, two_sources, small_blocks, tmp_path
    ) -> None:
        full = tmp_path / "series" / "gemma4-26b-product"
        moe.export_series(
            two_sources.weights,
            full,
            tokenizer_dir=two_sources.tokenizer,
            pins=two_sources.pins,
            sym_max=fx.TINY_SYM_MAX,
            steps=1,
        )
        out_dir = tmp_path / "series" / f"gemma4-26b-cut{fx.CUT}"

        summary = moe.export_cut(
            two_sources.weights,
            out_dir,
            tokenizer_dir=two_sources.tokenizer,
            pins=two_sources.pins,
            layers=fx.CUT,
            full_series=full,
            routing=True,
            sym_max=fx.TINY_SYM_MAX,
        )

        assert summary["layers"] == fx.CUT
        assert summary["cut"]["consistency"] == {
            "fullSeries": full.name,
            "layer": fx.CUT - 1,
            "rule": "torch.equal",
        }
        assert summary["routing"]["route_outputs"] == fx.CUT
        assert sorted(path.name for path in out_dir.glob(f"{moe.CUT_PREFIX}*")) == sorted(
            f"{moe.CUT_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name in CASE_NAMES
        )
        assert container_parts(out_dir / moe.ROUTING_DIR / one_shot.MODEL_FILE)
        record = json.loads((out_dir / provenance.REFERENCE_FILE).read_text(encoding="utf-8"))
        assert record["reference"]["cut"]["layers"] == fx.CUT
        assert record["reference"]["experts"] == routed.EXPERTS_RECORD
        assert sorted(record["reference"]["goldens"]) == sorted(
            f"{moe.CUT_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name in CASE_NAMES
        )

    def test_without_the_full_series_the_cut_fails_unless_the_check_is_skipped(
        self, tiny_series, checkpoint_dir, small_blocks, tmp_path
    ) -> None:
        with pytest.raises(moe.MoeExportError, match="write the product series first"):
            moe.export_cut(
                checkpoint_dir,
                tmp_path / "cut",
                tokenizer_dir=checkpoint_dir,
                pins=None,
                layers=fx.CUT,
                full_series=tmp_path / "missing",
                sym_max=fx.TINY_SYM_MAX,
            )
        assert not (tmp_path / "cut").exists()

        summary = moe.export_cut(
            checkpoint_dir,
            tmp_path / "cut",
            tokenizer_dir=checkpoint_dir,
            pins=None,
            layers=fx.CUT,
            full_series=None,
            sym_max=fx.TINY_SYM_MAX,
        )
        assert summary["cut"]["consistency"] == {"fullSeries": None, "rule": "skipped"}


class TestPlan:
    def test_the_plan_reads_no_weight_and_checks_the_form(
        self, tiny_series, checkpoint_dir, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            unified.Checkpoint,
            "read_rows",
            lambda *args: pytest.fail("the plan read a weight"),
        )

        summary = moe.plan(
            checkpoint_dir,
            tokenizer_dir=checkpoint_dir,
            pins=None,
            layers=None,
            sym_max=fx.TINY_SYM_MAX,
        )

        assert summary["experts_implementation"] == routed.EXPERTS_IMPLEMENTATION
        assert summary["i4_targets"] == fx.INT4_LINEARS
        assert summary["i4f16_targets"] == fx.EXPERT_WEIGHTS
        assert summary["ops"][EXPERT_LINEAR_OP] == 2 * fx.LAYERS
        assert summary["form"]["router_lookups"] == fx.LAYERS
        assert summary["kv"] == {"shared_kv_layers": len(fx.FULL_LAYERS)}
        assert summary["suppress_tokens"] == []


def test_the_routing_wrapper_returns_each_router_s_topk_index(checkpoint_dir, config) -> None:
    wrapper = naive_wrapper(checkpoint_dir, config)
    routing = moe.RoutingWrapper(wrapper.model).eval()
    ids = _ids(fx.WINDOW + 2)
    specs = decode.rope_specs(config)
    args = (
        ids,
        *unified.no_soft_inputs(ids, fx.HIDDEN),
        *decode.rope_args(specs, decode.positions_for(ids)),
        decode.last_rows_for(ids, 1),
    )

    with torch.no_grad():
        outputs = routing(*args)
        logits, hidden = wrapper(*args)
    with moe.LayerRecorder(wrapper, (), routes=True) as recorder, torch.no_grad():
        wrapper(*args)

    assert len(outputs) == 2 + fx.LAYERS
    assert torch.equal(outputs[0], logits) and torch.equal(outputs[1], hidden)
    for layer, picks in enumerate(outputs[2:]):
        assert torch.equal(picks, recorder.route_ids[layer][:, : fx.TOP_K]), layer


class TestRequiredLimits:
    """The distribution declares `requiredLimits.maxStorageBufferBindingSize` for the largest expert
    weight (ADR 0145 decision 1): assembly derives it from the container (`karume.dist.
    component_demand_bytes` — a hand-written value is refused), so what is pinned here is that the
    derivation counts a stacked expert weight as one buffer, and that 26B's is over the default."""

    def test_the_demand_counts_a_stacked_expert_weight_as_one_buffer(
        self, checkpoint_dir, traced, tmp_path
    ) -> None:
        from karume.dist import component_demand_bytes

        checkpoint = moe.open_checkpoint(checkpoint_dir)
        material = unified.prepare_material(traced.graph, traced.tensors, traced.keys, checkpoint)
        unified.write_container(traced.graph, material, tmp_path / "model.krm", None)

        demand = component_demand_bytes(tmp_path / "model.krm", {"maxPosition": 1}, {})

        gate_up = fx.EXPERTS * 2 * fx.MOE_INTERMEDIATE * fx.HIDDEN // 2
        assert demand >= gate_up
        assert demand >= max(fixed.packed.numel() for fixed in material.fixed.values())

    def test_26b_s_gate_up_payload_bakes_the_binding_limit(self) -> None:
        """From the pinned config (needs the snapshot): one layer's `gate_up` is 253.8 MB, over
        WebGPU's default 128 MiB binding, so assembly bakes the limit and a smaller adapter fails
        before the download."""
        from karume.limits import WEBGPU_DEFAULT_LIMITS, required_limits

        path = sources.checkpoint_dir("26b") / "config.json"
        if not path.is_file():
            pytest.skip(f"26B's pinned config is not in the hub cache ({path})")
        text = json.loads(path.read_text(encoding="utf-8"))["text_config"]
        payload = (
            int(text["num_experts"])
            * 2
            * int(text["moe_intermediate_size"])
            * int(text["hidden_size"])
            // 2
        )

        assert payload == 253_755_392
        assert payload > WEBGPU_DEFAULT_LIMITS["maxStorageBufferBindingSize"]
        assert required_limits(payload)["maxStorageBufferBindingSize"] >= payload
