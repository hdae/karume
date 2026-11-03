"""`gemma4/export_unified.py` (12B: the `gemma4_unified` class) on a tiny random checkpoint.

The tiny model has the 12B structure in small numbers: 7 layers (5 sliding + 2 full — a full
layer after a full layer, and the last layer full as upstream requires), K = V on the full
layers with 1 KV head against 2 on the sliding layers, head widths 16 / 32, no per-layer
embeddings, no KV sharing, a tied head, and a BF16 single-file checkpoint keyed like 12B's
(`model.language_model.*` plus the vision / audio embedders the export never reads) whose
linears lie on a Q4_0 grid as the QAT checkpoint's do ({@link grid_weight}). What is fixed here
are the properties that would otherwise pass silently:

- the shape-only trace is the real-weight trace (graph JSON and tensor keys);
- the linears are imported from their 4-bit grid (codes in [−8, 7], one step per group) and a
  weight off the grid fails loudly; the row-block container is the whole-tensor import, byte for
  byte;
- each full layer's V slot receives `v_norm(k_proj x)` unrotated and its K slot the rotated, normed
  key — against upstream's own modules — and the written graph's wiring check catches a V that
  is the rotated key or the unrotated, `k_norm`-scaled one;
- the export wrapper's logits are upstream `Gemma4UnifiedForCausalLM`'s (eager, F32) at lengths
  on both sides of the sliding window;
- the PLE-free product graph has no `per_layer_inputs`;
- the R4 reference (resident i4 codes, dequantized at use) equals an F32 load whose linears are
  codes · steps, weight for weight and output for output, and holds exactly what the container's
  bytes decode to;
- the goldens are taken over logits with the suppressed ids at −inf, and some golden case runs
  past the sliding window;
- the series run reads the weights and the tokenizer from two directories and publishes the
  container, goldens equal to an independent greedy run, and a provenance record naming both
  sources, the series itself and the suppressed ids — and a wrong pin of either source stops it
  before anything is written.

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
from safetensors.torch import save_file
from torch import nn

from _shared import decode_series as shared
from _shared.container_read import read_stored
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import export_unified as unified
from gemma4 import provenance, sources
from karume.container import container_parts
from karume.dist import DistError
from karume.emit import unpack_int4
from karume.ir import IrNode
from karume.ops import ATTENTION_OP, STATE_APPEND_OP
from karume.quantize import DEFAULT_GROUP_SIZE, dequantize_int4, fake_quant_int8

transformers = pytest.importorskip("transformers")

VOCAB = 64
HIDDEN = 64
SLIDING_DEPTH = 16
FULL_DEPTH = 32
HEADS = 4
SLIDING_KV = 2
FULL_KV = 1
WINDOW = 4
S, F = "sliding_attention", "full_attention"
LAYER_TYPES = (S, S, S, F, S, S, F)
LAYERS = len(LAYER_TYPES)
FULL_LAYERS = tuple(index for index, kind in enumerate(LAYER_TYPES) if kind == F)
#: The first full layer (a truncation may end on it).
FULL_LAYER = FULL_LAYERS[0]
#: The linears stored as i4: q / k / v / o + 3 MLP on a sliding layer, no `v_proj` on a full one.
INT4_LINEARS = 7 * (LAYERS - len(FULL_LAYERS)) + 6 * len(FULL_LAYERS)
TINY_SYM_MAX = 24
#: The tiny sliding ring (`window + SLIDING_SLACK_ROWS`).
RING = WINDOW + 8

#: The cases a series run uses (names from 12B's set — the greedy filter and the first token
#: expectations key on them).
CASE_NAMES = ("capital-en", "capital-ja", "context-en", "window-en")
#: The cases with greedy goldens (`context-en` is a near tie on the real 12B —
#: `UNIFIED_GREEDY_CASES`).
GREEDY_NAMES = ("capital-en", "capital-ja", "window-en")

#: The ids the tiny checkpoint's `generation_config.json` suppresses (upstream order, unsorted).
SUPPRESSED = (VOCAB - 1, VOCAB - 3)


def _text_config(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 128,
        "num_hidden_layers": LAYERS,
        "layer_types": list(LAYER_TYPES),
        "num_attention_heads": HEADS,
        "num_key_value_heads": SLIDING_KV,
        "num_global_key_value_heads": FULL_KV,
        "head_dim": SLIDING_DEPTH,
        "global_head_dim": FULL_DEPTH,
        "attention_k_eq_v": True,
        "sliding_window": WINDOW,
        "final_logit_softcapping": 30.0,
        "max_position_embeddings": 256,
        "hidden_size_per_layer_input": 0,
        "tie_word_embeddings": True,
        **overrides,
    }
    return transformers.Gemma4UnifiedTextConfig(**fields)


def grid_weight(rows: int, cols: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A weight on a Q4_0 grid as the QAT checkpoint stores one: `bf16(q·d)` per group of 32
    along the row, codes `q` in [−8, 7] and a learned-looking step `d` per group (log-normal, not
    `amax / 8`). Every third group's codes stay in [−3, 3] (reaching neither end), group 0 of row 0
    is all zero and group 1 of row 0 holds a −8. Returns `(weight BF16, codes int8, steps F32)`."""
    generator = torch.Generator().manual_seed(seed)
    groups = cols // DEFAULT_GROUP_SIZE
    shape = (rows, groups, DEFAULT_GROUP_SIZE)
    codes = torch.randint(-8, 8, shape, generator=generator)
    narrow = torch.randint(-3, 4, shape, generator=generator)
    codes[:, ::3] = narrow[:, ::3]
    codes[0, 0] = 0
    codes[0, 1, 0] = -8
    steps = 0.004 * torch.exp(0.3 * torch.randn((rows, groups), generator=generator))
    weight = (codes * steps.unsqueeze(-1)).reshape(rows, cols).to(torch.bfloat16)
    return weight, codes.reshape(rows, cols).to(torch.int8), steps


def write_checkpoint(directory: Path, *, extra: dict[str, torch.Tensor] | None = None) -> Path:
    """A tiny 12B-shaped checkpoint (BF16 `model.safetensors` + `config.json` + a stand-in
    `tokenizer.json` for the fingerprint); every text linear is a {@link grid_weight}."""
    torch.manual_seed(0)
    text = _text_config()
    model = transformers.Gemma4UnifiedForCausalLM(text).eval()
    state: dict[str, torch.Tensor] = {}
    for index, (key, value) in enumerate(model.state_dict().items()):
        if key == "lm_head.weight":
            continue
        tensor = value
        if key.endswith("_proj.weight"):
            tensor, _, _ = grid_weight(*value.shape, seed=index)
        elif key.endswith("layer_scalar"):
            tensor = torch.full_like(value, 0.75)
        elif key.endswith("norm.weight"):
            tensor = 1 + 0.1 * torch.randn_like(value)
        state[f"model.language_model.{key.removeprefix('model.')}"] = tensor.to(torch.bfloat16)
    state["model.vision_embedder.patch_dense.weight"] = torch.randn(HIDDEN, 8).to(torch.bfloat16)
    state["model.embed_vision.embedding_projection.weight"] = torch.randn(HIDDEN, HIDDEN).to(
        torch.bfloat16
    )
    state["model.embed_audio.embedding_projection.weight"] = torch.randn(HIDDEN, 4).to(
        torch.bfloat16
    )
    state.update(extra or {})
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.contiguous() for key, value in state.items()},
        str(directory / "model.safetensors"),
    )
    transformers.Gemma4UnifiedConfig(text_config=text.to_dict()).save_pretrained(directory)
    (directory / "tokenizer.json").write_text('{"stand-in": true}', encoding="utf-8")
    (directory / "generation_config.json").write_text(
        json.dumps(
            {
                "do_sample": True,
                "suppress_tokens": list(SUPPRESSED),
                "temperature": 1.0,
                "top_k": 8,
                "top_p": 0.9,
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory) -> Path:
    return write_checkpoint(tmp_path_factory.mktemp("tiny-12b"))


@pytest.fixture
def config(checkpoint_dir) -> Any:
    return unified.load_config(checkpoint_dir)


def naive_wrapper(checkpoint_dir: Path, config: Any) -> unified.UnifiedProductWrapper:
    """The F32-loaded wrapper (every checkpoint tensor widened, as E2B / E4B load theirs)."""
    model = unified.meta_model(config)
    checkpoint = unified.Checkpoint(checkpoint_dir)
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


def grid_loaded_wrapper(
    checkpoint_dir: Path, config: Any, int4: frozenset[str]
) -> unified.UnifiedProductWrapper:
    """The F32-loaded wrapper with the embedding rounded per row to i8 (`fake_quant_int8`, the
    E2B / E4B rounding) and every linear of `int4` replaced by codes · steps of its whole-tensor
    grid import — the model the container stores, built without row blocks or the reference's
    parametrization."""
    wrapper = naive_wrapper(checkpoint_dir, config)
    fake_quant_int8(wrapper, include=one_shot.is_int8_module)
    for fqn in sorted(int4):
        module = wrapper.get_submodule(fqn.removesuffix(".weight"))
        codes, steps = unified.grid_import(module.weight.data, DEFAULT_GROUP_SIZE, fqn)
        module.weight.data = dequantize_int4(codes, steps)
    return wrapper


def _ids(length: int, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, length), generator=generator, dtype=torch.int64)


def _call(wrapper: nn.Module, ids: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    specs = decode.rope_specs(wrapper.model.config)
    with torch.no_grad():
        return wrapper(
            ids,
            *decode.rope_args(specs, decode.positions_for(ids)),
            decode.last_rows_for(ids, rows),
        )


@pytest.fixture
def small_blocks(monkeypatch) -> None:
    """Row blocks of 16 rows (a 64-wide F32 row is 256 B), so every weight spans several."""
    monkeypatch.setattr(unified, "CHUNK_BYTES", 16 * HIDDEN * 4)


class TestConfig:
    def test_it_reads_the_unified_text_config_without_ple(self, config) -> None:
        assert config.model_type == unified.UNIFIED_TEXT_MODEL_TYPE
        assert config.layer_types[FULL_LAYER] == one_shot.FULL_ATTENTION
        assert config._attn_implementation == one_shot.ATTENTION_NAME

    def test_a_truncation_that_ends_on_a_sliding_layer_fails_loudly(self, checkpoint_dir) -> None:
        with pytest.raises(unified.UnifiedExportError, match="full attention"):
            unified.load_config(checkpoint_dir, layers=3)

    def test_a_config_with_per_layer_embeddings_fails_loudly(self, tmp_path) -> None:
        text = _text_config(hidden_size_per_layer_input=8)
        transformers.Gemma4UnifiedConfig(text_config=text.to_dict()).save_pretrained(tmp_path)

        with pytest.raises(unified.UnifiedExportError, match="per-layer"):
            unified.load_config(tmp_path)

    def test_the_kv_heads_follow_the_layer_type(self, config) -> None:
        assert decode.kv_heads(config, one_shot.SLIDING_ATTENTION) == SLIDING_KV
        assert decode.kv_heads(config, one_shot.FULL_ATTENTION) == FULL_KV


class TestShapeOnlyTrace:
    def test_the_graph_is_the_real_weight_trace(self, checkpoint_dir, config) -> None:
        """The trace follows shapes only, so data-less parameters give the same graph."""
        ids = _ids(WINDOW + 7)
        shape_only, shape_tensors = unified.trace(
            unified.shape_only_model(config), ids, TINY_SYM_MAX
        )
        real, real_tensors = unified.trace(naive_wrapper(checkpoint_dir, config), ids, TINY_SYM_MAX)

        assert shape_only.to_json() == real.to_json()
        assert sorted(shape_tensors) == sorted(real_tensors)

    def test_the_parameters_hold_no_data_and_keep_the_tie(self, config) -> None:
        wrapper = unified.shape_only_model(config)

        assert not any(parameter.is_meta for parameter in wrapper.parameters())
        assert wrapper.model.lm_head.weight is wrapper.model.model.embed_tokens.weight
        # The config-derived buffer is upstream's value, not an empty one.
        assert float(wrapper.model.model.embed_tokens.embed_scale) == pytest.approx(HIDDEN**0.5)


@pytest.fixture
def traced(config):
    wrapper = unified.shape_only_model(config)
    graph, tensors = unified.trace(wrapper, _ids(WINDOW + 7), TINY_SYM_MAX)
    return SimpleNamespace(graph=graph, tensors=tensors, keys=unified.trace_keys(wrapper))


class TestContainer:
    def test_the_row_block_container_is_the_whole_tensor_import_byte_for_byte(
        self, checkpoint_dir, traced, small_blocks, monkeypatch, tmp_path
    ) -> None:
        """Row blocks (16 rows — every weight spans several) and one block per tensor write the
        same parts: a group never crosses a row, so the grid import closes per row."""
        checkpoint = unified.Checkpoint(checkpoint_dir)

        def written(name: str) -> tuple[Any, list[bytes]]:
            material = unified.prepare_material(
                traced.graph, traced.tensors, traced.keys, checkpoint
            )
            path = tmp_path / name / "model.krm"
            verified = unified.write_container(traced.graph, material, path, None)
            return verified, [part.read_bytes() for part in container_parts(path)]

        verified, in_blocks = written("blocks")
        monkeypatch.setattr(unified, "CHUNK_BYTES", 1 << 40)
        quantized = [
            unified.checkpoint_key(key) for key in (*sorted(traced.keys.int4), unified.EMBED_KEY)
        ]
        assert all(
            checkpoint.shape(key)[0] <= unified._block_rows(checkpoint.shape(key)[1])
            for key in quantized
        )
        _, whole = written("whole")

        assert in_blocks == whole
        storage = [init.storage.dtype for init in verified.initializers.values()]
        assert storage.count("i4") == len(traced.keys.int4) == INT4_LINEARS
        assert storage.count("i8") == 1

    def test_the_container_decodes_to_the_reference_weights(
        self, checkpoint_dir, config, traced, small_blocks, tmp_path
    ) -> None:
        """R4: the goldens' reference holds exactly what the container's bytes decode to — every
        imported linear through `karume.emit.unpack_int4` + `dequantize_int4`, and the i8
        embedding — so the goldens describe the stored model, −8 codes included."""
        checkpoint = unified.Checkpoint(checkpoint_dir)
        material = unified.prepare_material(traced.graph, traced.tensors, traced.keys, checkpoint)
        unified.write_container(traced.graph, material, tmp_path / "model.krm", None)
        stored = read_stored(tmp_path / "model.krm")
        reference = unified.reference_wrapper(config, checkpoint, traced.keys.int4)

        lowest = 0
        for fqn in sorted(traced.keys.int4):
            entry = stored[fqn]
            rows, cols = entry.shape
            assert entry.layout == "i4", fqn
            assert entry.scale is not None and entry.group_size == DEFAULT_GROUP_SIZE, fqn
            codes = unpack_int4(
                torch.frombuffer(bytearray(entry.payload), dtype=torch.uint8), entry.shape
            )
            steps = torch.frombuffer(bytearray(entry.scale), dtype=torch.float32).reshape(
                rows, cols // DEFAULT_GROUP_SIZE
            )
            weight = reference.get_submodule(fqn.removesuffix(".weight")).weight
            assert torch.equal(dequantize_int4(codes, steps), weight), fqn
            lowest = min(lowest, int(codes.min()))
        assert lowest == -8

        (embedding,) = (key for key in stored if key in (unified.EMBED_KEY, unified.TIED_HEAD_KEY))
        entry = stored[embedding]
        assert entry.layout == "i8" and entry.scale is not None
        values = torch.frombuffer(bytearray(entry.payload), dtype=torch.int8).reshape(entry.shape)
        scale = torch.frombuffer(bytearray(entry.scale), dtype=torch.float32).reshape(-1, 1)
        assert torch.equal(
            values.to(torch.float32) * scale, reference.model.model.embed_tokens.weight
        )

    def test_the_row_blocks_split_every_weight(self, checkpoint_dir, traced, small_blocks) -> None:
        """The byte identity above is measured across block boundaries (not one block) — for every
        quantized linear as for the embedding."""
        checkpoint = unified.Checkpoint(checkpoint_dir)
        quantized = [
            unified.checkpoint_key(key) for key in (*sorted(traced.keys.int4), unified.EMBED_KEY)
        ]

        assert len(quantized) == INT4_LINEARS + 1
        for key in quantized:
            rows, cols = checkpoint.shape(key)
            assert rows > unified._block_rows(cols), key

    def test_the_product_form_has_no_per_layer_inputs(
        self, checkpoint_dir, config, traced, tmp_path
    ) -> None:
        checkpoint = unified.Checkpoint(checkpoint_dir)
        material = unified.prepare_material(traced.graph, traced.tensors, traced.keys, checkpoint)
        verified = unified.write_container(traced.graph, material, tmp_path / "model.krm", None)

        form = product.assert_ir_form_product(
            verified, config, {"i8": 1, "i4": len(traced.keys.int4)}, None
        )

        assert [spec.name for spec in verified.inputs] == [
            decode.INPUT_IDS,
            *decode.ROPE_INPUTS,
            decode.TOKEN_ONLY_LAST_ROW,
        ]
        assert form["slots"] == 2 * LAYERS
        assert form["heads"] == {
            one_shot.SLIDING_ATTENTION: [HEADS, SLIDING_KV, SLIDING_KV],
            one_shot.FULL_ATTENTION: [HEADS, FULL_KV, FULL_KV],
        }
        sliding = verified.states["l0.k"]
        full = verified.states[f"l{FULL_LAYER}.v"]
        assert sliding.shape == [1, SLIDING_KV, WINDOW + decode.SLIDING_SLACK_ROWS, SLIDING_DEPTH]
        assert full.shape == [1, FULL_KV, decode.CAPACITY_SYMBOL, FULL_DEPTH]

    def test_the_form_check_refuses_ple_rows_for_a_graph_without_per_layer_inputs(
        self, config, traced
    ) -> None:
        with pytest.raises(AssertionError, match="ple_rows"):
            product.assert_ir_form_product(traced.graph, config, {}, 8)

    def test_a_per_layer_tensor_in_the_checkpoint_fails_loudly(self, tmp_path) -> None:
        directory = write_checkpoint(
            tmp_path,
            extra={
                "model.language_model.embed_tokens_per_layer.weight": torch.zeros(
                    VOCAB, 8, dtype=torch.bfloat16
                )
            },
        )
        config = unified.load_config(directory)
        keys = unified.trace_keys(unified.shape_only_model(config))

        with pytest.raises(unified.UnifiedExportError, match="per-layer"):
            unified.assert_checkpoint_keys(
                keys.persistent, unified.Checkpoint(directory).names(), LAYERS
            )

    def test_a_key_outside_the_text_decoder_fails_loudly(self, tmp_path) -> None:
        directory = write_checkpoint(
            tmp_path, extra={"model.audio_tower.layer.weight": torch.zeros(2, dtype=torch.bfloat16)}
        )

        with pytest.raises(unified.UnifiedExportError, match="outside the text decoder"):
            unified.Checkpoint(directory)

    def test_a_truncated_model_leaves_the_later_layers_unread(self, checkpoint_dir) -> None:
        config = unified.load_config(checkpoint_dir)
        config.num_hidden_layers = 1
        keys = {key for key in unified.trace_keys(unified.shape_only_model(config)).persistent}
        available = unified.Checkpoint(checkpoint_dir).names()

        unified.assert_checkpoint_keys(keys, available, 1)
        with pytest.raises(unified.UnifiedExportError, match="does not consume"):
            unified.assert_checkpoint_keys(keys, available, 2)


#: A linear of the tiny checkpoint (the first layer's query projection).
Q_PROJ = "model.language_model.layers.0.self_attn.q_proj.weight"


class TestGridImport:
    def test_codes_and_steps_reproduce_a_grid_weight_within_the_tolerance(self) -> None:
        """Groups that reach −8, groups that reach neither end and an all-zero group: every one
        is reconstructed within `2^-7` of its max |w|, on its own learned step."""
        weight, built_codes, built_steps = grid_weight(8, 4 * DEFAULT_GROUP_SIZE, seed=11)
        widened = weight.to(torch.float32)

        codes, steps = unified.grid_import(widened, DEFAULT_GROUP_SIZE, "'w'")

        assert codes.dtype is torch.int8 and steps.dtype is torch.float32
        assert codes.shape == widened.shape and steps.shape == (8, 4)
        assert int(codes.min()) == -8 and int(codes.max()) <= 7
        groups = widened.reshape(8, 4, DEFAULT_GROUP_SIZE)
        error = (dequantize_int4(codes, steps) - widened).reshape(8, 4, DEFAULT_GROUP_SIZE)
        bound = unified.GRID_TOLERANCE * groups.abs().amax(dim=-1)
        assert bool((error.abs().amax(dim=-1) <= bound).all())
        # The groups whose codes stay in [−3, 3] keep their own step (not one stretched to ±7).
        narrow = built_codes.reshape(8, 4, DEFAULT_GROUP_SIZE)[:, ::3].abs().amax(dim=-1) <= 3
        assert bool(narrow.all())
        nonzero = built_steps.new_ones(built_steps.shape, dtype=torch.bool)
        nonzero[0, 0] = False
        assert torch.allclose(steps[nonzero], built_steps[nonzero], rtol=2.0**-7)
        # The all-zero group: codes 0, step 1.
        assert not bool(codes[0, :DEFAULT_GROUP_SIZE].any()) and float(steps[0, 0]) == 1.0

    def test_a_weight_off_the_grid_fails_loudly_naming_the_key(self, tmp_path) -> None:
        directory = write_checkpoint(
            tmp_path, extra={Q_PROJ: torch.randn(HEADS * SLIDING_DEPTH, HIDDEN).to(torch.bfloat16)}
        )
        checkpoint = unified.Checkpoint(directory)

        with pytest.raises(unified.UnifiedExportError, match=rf"'{Q_PROJ}' rows 0\.\..*group"):
            unified.int4_codes(checkpoint, Q_PROJ)
        with pytest.raises(unified.UnifiedExportError, match="not on a 16-level grid"):
            unified.quantize_int4_rows(checkpoint, Q_PROJ)

    def test_the_failing_group_is_named_by_row_and_index(self) -> None:
        weight, _, _ = grid_weight(4, 2 * DEFAULT_GROUP_SIZE, seed=5)
        widened = weight.to(torch.float32)
        widened[2, DEFAULT_GROUP_SIZE + 3] *= 1.3  # off the grid of row 2's group 1

        with pytest.raises(unified.UnifiedExportError, match="group 1 of row 18"):
            unified.grid_import(widened, DEFAULT_GROUP_SIZE, "'w'", first_row=16)

    def test_a_nan_never_passes_as_an_all_zero_group(self) -> None:
        weight = torch.zeros(1, DEFAULT_GROUP_SIZE)
        weight[0, 0] = float("nan")

        with pytest.raises(unified.UnifiedExportError, match="not on a 16-level grid"):
            unified.grid_import(weight, DEFAULT_GROUP_SIZE, "'w'")


class TestSharedKeyValue:
    def test_the_full_layer_stores_v_norm_of_the_raw_key_projection(
        self, checkpoint_dir, config, monkeypatch
    ) -> None:
        """The k / v the wrapper hands to attention (= what the state slots receive) against
        upstream's own modules: V = v_norm(k_proj x) unrotated, K = RoPE(k_norm(k_proj x))."""
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            apply_rotary_pos_emb,
        )

        wrapper = naive_wrapper(checkpoint_dir, config)
        seen: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        entering: dict[int, torch.Tensor] = {}

        def spy(module, query, key, value, attention_mask, **kwargs):
            seen[module.layer_idx] = (key, value)
            return one_shot.gqa_sdpa_attention(module, query, key, value, attention_mask, **kwargs)

        transformers.AttentionInterface.register(one_shot.ATTENTION_NAME, spy)
        handles = [
            wrapper.model.model.layers[layer].self_attn.register_forward_pre_hook(
                lambda module, _args, kwargs: entering.__setitem__(
                    module.layer_idx, kwargs["hidden_states"]
                ),
                with_kwargs=True,
            )
            for layer in FULL_LAYERS
        ]
        ids = _ids(WINDOW + 5)
        try:
            _call(wrapper, ids, 1)
        finally:
            for handle in handles:
                handle.remove()
            one_shot.register_attention()

        specs = decode.rope_specs(config)
        cos, sin = decode.rope_args(specs, decode.positions_for(ids))[2:4]
        for layer in FULL_LAYERS:
            attention = wrapper.model.model.layers[layer].self_attn
            with torch.no_grad():
                raw = attention.k_proj(entering[layer]).view(1, ids.shape[1], FULL_KV, FULL_DEPTH)
                value = attention.v_norm(raw).transpose(1, 2)
                key = apply_rotary_pos_emb(
                    attention.k_norm(raw), cos, sin, unsqueeze_dim=2
                ).transpose(1, 2)
            assert attention.v_proj is None
            assert torch.equal(seen[layer][1], value), layer
            assert torch.equal(seen[layer][0], key), layer
            assert not torch.equal(seen[layer][0], seen[layer][1]), layer

    def test_the_slots_append_the_attention_inputs(self, traced) -> None:
        """The surgery appends exactly the k / v the attention reads — so the eager check above is
        the slots' content."""
        attentions = [node for node in traced.graph.nodes if node.op == ATTENTION_OP]
        appends = {
            node.states["slot"]: node.ins[0]
            for node in traced.graph.nodes
            if node.op == STATE_APPEND_OP
        }
        full = attentions[FULL_LAYER]

        assert appends[f"l{FULL_LAYER}.k"] == full.ins[1]
        assert appends[f"l{FULL_LAYER}.v"] == full.ins[2]

    def test_the_wiring_check_passes_the_traced_graph(self, config, traced) -> None:
        assert unified.assert_shared_kv_projection(traced.graph, config) == {
            "shared_kv_layers": len(FULL_LAYERS)
        }

    def test_a_value_that_is_the_rotated_key_is_caught(self, config, traced) -> None:
        nodes = list(traced.graph.nodes)
        # The second full layer: its input already went through the first one's full RoPE, so
        # only the chain from its own projection can tell whether V is rotated.
        index = [i for i, node in enumerate(nodes) if node.op == ATTENTION_OP][FULL_LAYERS[-1]]
        node = nodes[index]
        nodes[index] = _with_ins(node, [node.ins[0], node.ins[1], node.ins[1]])
        bent = SimpleNamespace(
            nodes=nodes, inputs=traced.graph.inputs, initializers=traced.graph.initializers
        )

        with pytest.raises(unified.UnifiedExportError, match="rotated"):
            unified.assert_shared_kv_projection(bent, config)

    def test_a_value_normed_by_k_norm_is_caught(self, config, traced) -> None:
        """V = k_norm(k_proj x): unrotated and from the shared projection, but scaled by `k_norm`'s
        learned weight instead of the scale-free `v_norm` — the norm in V's chain is given K's."""
        graph = traced.graph
        producers = unified._producers(graph)
        attention = [node for node in graph.nodes if node.op == ATTENTION_OP][FULL_LAYERS[-1]]
        _, key_chain = unified._projection_of(producers, attention.ins[1], "K")
        _, value_chain = unified._projection_of(producers, attention.ins[2], "V")
        key_norm = next(node for node in key_chain if node.op == "rms_norm")
        value_norm = next(node for node in value_chain if node.op == "rms_norm")
        assert graph.initializers[key_norm.ins[1]].tensor.endswith("k_norm.weight")
        nodes = [
            _with_ins(node, [node.ins[0], key_norm.ins[1]]) if node is value_norm else node
            for node in graph.nodes
        ]
        bent = SimpleNamespace(nodes=nodes, inputs=graph.inputs, initializers=graph.initializers)

        with pytest.raises(unified.UnifiedExportError, match="v_norm has no scale"):
            unified.assert_shared_kv_projection(bent, config)


def _with_ins(node: IrNode, ins: list[str]) -> IrNode:
    return IrNode(op=node.op, ins=ins, outs=node.outs, attrs=node.attrs, states=node.states)


def upstream_model(checkpoint_dir: Path, config: Any) -> nn.Module:
    """Upstream `Gemma4UnifiedForCausalLM` with the checkpoint widened to F32, its own rotary
    embedding, its own mask construction and eager attention."""
    upstream_config = copy.deepcopy(config)
    upstream_config._attn_implementation = "eager"
    model = transformers.Gemma4UnifiedForCausalLM(upstream_config).eval()
    checkpoint = unified.Checkpoint(checkpoint_dir)
    state = {
        key: checkpoint.read(unified.checkpoint_key(f"model.{key}"))
        for key in model.state_dict()
        if key != "lm_head.weight"
    }
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    return model


class TestUpstreamParity:
    #: F32 bound on the softcapped logits (|logits| ≤ 30); measured max difference 3.3e-6.
    BOUND = 2e-5

    @pytest.mark.parametrize("length", [3, WINDOW, WINDOW + 1, 3 * WINDOW + 5, RING + 3])
    def test_the_wrapper_logits_are_upstream_s_eager_forward(
        self, checkpoint_dir, config, length
    ) -> None:
        """Two full layers (one right after another) and lengths on both sides of the sliding
        window: the export's masks, host RoPE and K = V wiring against upstream's own."""
        wrapper = naive_wrapper(checkpoint_dir, config)
        upstream = upstream_model(checkpoint_dir, config)
        ids = _ids(length, seed=length)

        logits, _ = _call(wrapper, ids, length)
        with torch.no_grad():
            expected = upstream(input_ids=ids, use_cache=False).logits

        assert len(FULL_LAYERS) >= 2
        assert logits.shape == expected.shape
        assert float((logits - expected).abs().max()) <= self.BOUND


class TestReference:
    def test_the_r4_reference_equals_the_f32_load_with_imported_linears(
        self, checkpoint_dir, config, traced, small_blocks
    ) -> None:
        """Resident i4 codes dequantized at use give the outputs of an F32 load whose linears are
        codes · steps (and whose embedding is `fake_quant_int8`'s), bit for bit (R4: an F32
        reference without an F32 copy of the weights)."""
        reference = unified.reference_wrapper(
            config, unified.Checkpoint(checkpoint_dir), traced.keys.int4
        )
        loaded = grid_loaded_wrapper(checkpoint_dir, config, traced.keys.int4)
        ids = _ids(WINDOW + 9)

        for got, want in zip(_call(reference, ids, 3), _call(loaded, ids, 3), strict=True):
            assert torch.equal(got, want)

    def test_each_dequantized_weight_is_the_whole_tensor_import(
        self, checkpoint_dir, config, traced, small_blocks
    ) -> None:
        """Weight for weight, bit for bit: `dequantize_int4` of the resident codes (taken in row
        blocks) is codes · steps of the whole tensor's grid import."""
        reference = unified.reference_wrapper(
            config, unified.Checkpoint(checkpoint_dir), traced.keys.int4
        )
        naive = grid_loaded_wrapper(checkpoint_dir, config, traced.keys.int4)

        for fqn in sorted(traced.keys.int4):
            module = fqn.removesuffix(".weight")
            got = reference.get_submodule(module).weight
            want = naive.get_submodule(module).weight
            assert got.dtype is torch.float32, fqn
            assert torch.equal(got, want), fqn

    def test_the_linear_weights_stay_resident_as_i4_values(
        self, checkpoint_dir, config, traced
    ) -> None:
        reference = unified.reference_wrapper(
            config, unified.Checkpoint(checkpoint_dir), traced.keys.int4
        )
        attention = reference.model.model.layers[0].self_attn
        codes = attention.q_proj.parametrizations.weight.original

        assert codes.dtype is torch.int8
        assert int(codes.min()) >= -8 and int(codes.max()) <= 7
        assert attention.q_proj.weight.dtype is torch.float32
        assert reference.model.lm_head.weight is reference.model.model.embed_tokens.weight


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True):
        return SimpleNamespace(ids=[len(text) % VOCAB])

    def id_to_token(self, token: int) -> str:
        return f"<{token}>"

    def decode(self, tokens) -> str:
        return " ".join(f"<{token}>" for token in tokens)


#: The stand-in window case's length: past the tiny ring and past the traced chunk range, as
#: 12B's `window-en` is past both (1212 against 1032 and 768).
WINDOW_CASE_LENGTH = TINY_SYM_MAX + 6


@pytest.fixture
def tiny_series(monkeypatch, checkpoint_dir):
    """Stand-ins for what the tiny random model cannot provide: the cases, the tokenizer, the
    margin floor and the first-token expectations (`test_export_decode.tiny_series`' set)."""
    cases = tuple(
        (
            name,
            _ids(
                WINDOW_CASE_LENGTH if name == unified.WINDOW_CASE else WINDOW + 3 + 2 * index,
                seed=index,
            ),
        )
        for index, name in enumerate(CASE_NAMES)
    )
    assert [name for name, _ in unified.UNIFIED_CASES] == list(CASE_NAMES)
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


#: The stand-in pinned sources' revisions (distinct, so a record or a container naming the wrong
#: one is caught).
WEIGHTS_REVISION = "0" * 40
TOKENIZER_REVISION = "1" * 40


@pytest.fixture
def two_sources(checkpoint_dir, tmp_path) -> SimpleNamespace:
    """12B's layout: the weights and their config in one directory, the tokenizer and the
    generation defaults in another — neither holds the other's files, so a read from the wrong one
    fails — and pins for both."""
    weights = tmp_path / "sources" / "weights"
    tokenizer = tmp_path / "sources" / "tokenizer"
    for directory, names in (
        (weights, ("model.safetensors", "config.json")),
        (tokenizer, ("tokenizer.json", "generation_config.json")),
    ):
        directory.mkdir(parents=True)
        for name in names:
            shutil.copy(checkpoint_dir / name, directory / name)
    pins = sources.ModelPins(
        weights=_pin("test/weights", WEIGHTS_REVISION, weights, provenance.WEIGHTS_FILES),
        tokenizer=_pin("test/tokenizer", TOKENIZER_REVISION, tokenizer, provenance.TOKENIZER_FILES),
    )
    return SimpleNamespace(weights=weights, tokenizer=tokenizer, pins=pins)


class TestExportSeries:
    def test_it_publishes_the_container_the_goldens_and_the_record(
        self, tiny_series, two_sources, checkpoint_dir, config, small_blocks, tmp_path
    ) -> None:
        out_dir = tmp_path / "series" / "gemma4-12b-product"
        steps = 3

        summary = unified.export_series(
            two_sources.weights,
            out_dir,
            tokenizer_dir=two_sources.tokenizer,
            pins=two_sources.pins,
            sym_max=TINY_SYM_MAX,
            steps=steps,
            checkpoint_label="weights and tokenizer",
        )

        names = sorted(path.name for path in out_dir.iterdir())
        assert names == sorted(
            [
                *(path.name for path in container_parts(out_dir / one_shot.MODEL_FILE)),
                *(f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name in GREEDY_NAMES),
                provenance.REFERENCE_FILE,
            ]
        )
        assert [path.name for path in out_dir.parent.iterdir()] == [out_dir.name]
        record = json.loads((out_dir / provenance.REFERENCE_FILE).read_text(encoding="utf-8"))
        assert record["series"] == record["reference"]["series"] == out_dir.name
        assert sorted(record["reference"]["goldens"]) == sorted(
            f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name in GREEDY_NAMES
        )
        # A case without a golden still has its first token checked.
        assert [sorted(first) for first in tiny_series.checked] == [
            sorted(unified.FIRST_TOKEN_EXPECTATIONS)
        ]
        assert "context-en" in tiny_series.checked[0]
        assert record["reference"]["suppressTokens"] == sorted(SUPPRESSED)
        assert record["checkpoint"]["dir"] == "weights and tokenizer"
        assert record["checkpoint"]["sources"] == {
            "weights": {
                "repo": "test/weights",
                "revision": WEIGHTS_REVISION,
                "files": list(provenance.WEIGHTS_FILES),
            },
            "tokenizer": {
                "repo": "test/tokenizer",
                "revision": TOKENIZER_REVISION,
                "files": list(provenance.TOKENIZER_FILES),
            },
        }
        files = record["checkpoint"]["files"]
        for pin in two_sources.pins.by_role().values():
            for name, digest in pin.sha256.items():
                assert files[name]["sha256"] == digest, name
        assert summary["window"] == {
            "sliding_window": WINDOW,
            "ring_rows": RING,
            "longest_forward": WINDOW_CASE_LENGTH + steps - 1,
        }
        assert summary["suppress_tokens"] == sorted(SUPPRESSED)
        assert summary["kv"] == {"shared_kv_layers": len(FULL_LAYERS)}
        assert [stage["name"] for stage in summary["stages"]] == [
            "trace",
            "goldens",
            "quantize",
            "write",
            "provenance",
        ]

        # The goldens are an independent greedy run of the F32 load with imported linears, over
        # the logits with the suppressed ids masked.
        loaded = grid_loaded_wrapper(
            checkpoint_dir, config, unified.trace_keys(naive_wrapper(checkpoint_dir, config)).int4
        )
        view = unified.GreedyView(loaded, decode.rope_specs(config), sorted(SUPPRESSED))
        for name, ids in tiny_series.cases:
            expected, margins = shared.greedy_continuation(view, ids, steps)
            if name not in GREEDY_NAMES:
                assert tiny_series.checked[0][name] == expected[0], name
                continue
            golden = out_dir / f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}"
            with safe_open(str(golden), framework="pt") as handle:
                assert handle.get_tensor(shared.EXPECTED_KEY).tolist() == expected
                assert handle.get_tensor(shared.MARGIN_KEY).tolist() == pytest.approx(margins)
                assert torch.equal(handle.get_tensor(shared.PROMPT_KEY), ids[0].to(torch.int32))
            assert not set(expected) & set(SUPPRESSED), name

    def test_the_container_names_the_weights_revision(
        self, tiny_series, two_sources, small_blocks, tmp_path
    ) -> None:
        from _shared.container_read import read_provenance

        out_dir = tmp_path / "gemma4-12b-product"
        unified.export_series(
            two_sources.weights,
            out_dir,
            tokenizer_dir=two_sources.tokenizer,
            pins=two_sources.pins,
            sym_max=TINY_SYM_MAX,
            steps=1,
        )

        assert read_provenance(out_dir / one_shot.MODEL_FILE).upstream_revision == WEIGHTS_REVISION

    @pytest.mark.parametrize(
        ("role", "name"),
        [
            ("weights", "model.safetensors"),
            ("weights", "config.json"),
            ("tokenizer", "tokenizer.json"),
        ],
    )
    def test_a_wrong_pin_of_either_source_stops_the_run_before_anything_is_written(
        self, tiny_series, two_sources, tmp_path, role, name
    ) -> None:
        pins = two_sources.pins
        pin = getattr(pins, role)
        wrong = sources.ModelPins(
            **{
                **pins.by_role(),
                role: sources.HubPin(
                    repo=pin.repo, revision=pin.revision, sha256={**pin.sha256, name: "f" * 64}
                ),
            }
        )
        out_dir = tmp_path / "series" / "gemma4-12b-product"

        with pytest.raises(sources.SourcePinError, match=rf"{name}.*{pin.repo}@"):
            unified.export_series(
                two_sources.weights,
                out_dir,
                tokenizer_dir=two_sources.tokenizer,
                pins=wrong,
                sym_max=TINY_SYM_MAX,
                steps=1,
            )
        assert not (tmp_path / "series").exists()

    def test_a_truncated_run_writes_the_container_alone(
        self, tiny_series, checkpoint_dir, small_blocks, tmp_path
    ) -> None:
        out_dir = tmp_path / "truncated"

        summary = unified.export_series(
            checkpoint_dir,
            out_dir,
            tokenizer_dir=checkpoint_dir,
            pins=None,
            sym_max=TINY_SYM_MAX,
            steps=1,
            layers=FULL_LAYER + 1,
        )

        assert summary["layers"] == FULL_LAYER + 1
        assert "greedy" not in summary
        assert not list(out_dir.glob(f"{shared.GREEDY_PREFIX}*"))
        assert not (out_dir / provenance.REFERENCE_FILE).exists()


class TestPlan:
    def test_the_plan_reads_no_weight_and_checks_the_form(
        self, tiny_series, checkpoint_dir
    ) -> None:
        summary = unified.plan(
            checkpoint_dir,
            tokenizer_dir=checkpoint_dir,
            pins=None,
            layers=None,
            sym_max=TINY_SYM_MAX,
        )

        assert [name for name, _, _ in summary["inputs"]] == [
            decode.INPUT_IDS,
            *decode.ROPE_INPUTS,
            decode.TOKEN_ONLY_LAST_ROW,
        ]
        assert summary["i4_targets"] == INT4_LINEARS
        assert summary["kv"] == {"shared_kv_layers": len(FULL_LAYERS)}
        assert summary["case_lengths"][unified.WINDOW_CASE] == WINDOW_CASE_LENGTH
        assert summary["window"]["ring_rows"] == RING
        assert summary["suppress_tokens"] == sorted(SUPPRESSED)

    def test_the_plan_checks_both_pins_first(self, tiny_series, two_sources) -> None:
        pins = two_sources.pins
        wrong = sources.ModelPins(
            weights=pins.weights,
            tokenizer=sources.HubPin(
                repo=pins.tokenizer.repo,
                revision=pins.tokenizer.revision,
                sha256={"tokenizer.json": "f" * 64},
            ),
        )

        unified.plan(
            two_sources.weights,
            tokenizer_dir=two_sources.tokenizer,
            pins=pins,
            layers=None,
            sym_max=TINY_SYM_MAX,
        )
        with pytest.raises(sources.SourcePinError, match=r"tokenizer\.json"):
            unified.plan(
                two_sources.weights,
                tokenizer_dir=two_sources.tokenizer,
                pins=wrong,
                layers=None,
                sym_max=TINY_SYM_MAX,
            )


class TestSuppressedTokens:
    def test_the_greedy_view_never_selects_a_suppressed_id(self, checkpoint_dir, config) -> None:
        """Suppressing the unmasked winner moves the greedy token to the runner-up, and the margin
        is taken over the masked logits."""
        naive = naive_wrapper(checkpoint_dir, config)
        specs = decode.rope_specs(config)
        ids = _ids(WINDOW + 5)
        with torch.no_grad():
            plain, _ = unified.GreedyView(naive, specs)(ids, decode.positions_for(ids))
            masked, _ = unified.GreedyView(naive, specs, [int(plain[0, -1].argmax())])(
                ids, decode.positions_for(ids)
            )
        top = torch.topk(plain[0, -1], 3)
        winner = int(top.indices[0])

        tokens, margins = shared.greedy_continuation(
            unified.GreedyView(naive, specs, [winner]), ids, 1
        )

        assert masked[0, -1, winner] == float("-inf")
        assert tokens == [int(top.indices[1])]
        assert margins[0] == pytest.approx(float(top.values[1] - top.values[2]))

    def test_the_ids_come_from_the_generation_config_sorted(self, checkpoint_dir, config) -> None:
        assert unified.suppress_tokens(checkpoint_dir, config) == tuple(sorted(SUPPRESSED))

    def test_an_id_outside_the_vocabulary_fails_loudly(self, tmp_path, config) -> None:
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"suppress_tokens": [VOCAB], "temperature": 1.0, "top_k": 8, "top_p": 1.0}),
            encoding="utf-8",
        )

        with pytest.raises(DistError, match="outside the graph's vocabulary"):
            unified.suppress_tokens(tmp_path, config)


class TestWindowExercised:
    """The 12B equivalent of `gemma4.export.assert_case_lengths`."""

    def _cases(self, window_length: int, other: int = 5) -> list[tuple[str, torch.Tensor]]:
        return [("context-en", _ids(other)), (unified.WINDOW_CASE, _ids(window_length))]

    def test_the_longest_forward_is_the_prompt_plus_the_steps_before_the_last(self, config) -> None:
        window = unified.assert_window_exercised(self._cases(RING + 1), 3, config)

        assert window["longest_forward"] == RING + 3

    def test_goldens_that_stay_within_the_window_fail_loudly(self, config) -> None:
        """T + steps − 1 = window: the last re-forward is still inside the band (off by one from
        the T + steps a naive count gives); one step more passes the window rule (and reaches the
        next check)."""
        cases = [("context-en", _ids(WINDOW - 2))]

        with pytest.raises(unified.UnifiedExportError, match="within the sliding window"):
            unified.assert_window_exercised(cases, 3, config)
        with pytest.raises(unified.UnifiedExportError, match="lack"):
            unified.assert_window_exercised(cases, 4, config)

    def test_a_window_case_that_fits_the_ring_fails_loudly(self, config) -> None:
        with pytest.raises(unified.UnifiedExportError, match="does not exceed the sliding ring"):
            unified.assert_window_exercised(self._cases(RING), 1, config)

    def test_context_en_has_no_greedy_golden_but_keeps_its_first_token_check(self) -> None:
        assert unified.UNIFIED_GREEDY_CASES == GREEDY_NAMES
        assert "context-en" in unified.FIRST_TOKEN_EXPECTATIONS

    def test_the_window_case_must_be_there(self, config) -> None:
        with pytest.raises(unified.UnifiedExportError, match="lack"):
            unified.assert_window_exercised([("context-en", _ids(RING + 4))], 1, config)

    def test_only_the_window_case_may_leave_the_traced_range(self) -> None:
        unified.assert_trace_lengths(self._cases(TINY_SYM_MAX + 1), TINY_SYM_MAX)
        with pytest.raises(unified.UnifiedExportError, match="traced range"):
            unified.assert_trace_lengths([("context-en", _ids(TINY_SYM_MAX + 1))], TINY_SYM_MAX)

    def test_12b_s_window_case_is_past_the_ring(self) -> None:
        """The real text against the real tokenizer (needs the pinned tokenizer snapshot)."""
        tokenizer_dir = sources.tokenizer_dir("12b")
        if not (tokenizer_dir / "tokenizer.json").is_file():
            pytest.skip(f"12B's tokenizer snapshot is not in the hub cache ({tokenizer_dir})")
        cases = dict(one_shot.encode_cases(tokenizer_dir, unified.UNIFIED_CASES))

        assert int(cases[unified.WINDOW_CASE].shape[1]) > 1024 + decode.SLIDING_SLACK_ROWS
        assert all(
            int(ids.shape[1]) <= one_shot.SYM_MAX
            for name, ids in cases.items()
            if name != unified.WINDOW_CASE
        )
