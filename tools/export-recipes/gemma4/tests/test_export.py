"""Script-level contract of `gemma4/export.py` (the part that needs no real weights).

Emitting with real weights is manual (the same discipline as the tests of `minicpm5` /
`embeddinggemma`). What is pinned here is only the discipline whose breakage would give a **false
PASS**:

- the boundaries of the 2 additive masks (causal / band) and **prefix commutation** (the condition
  for the folding of ADR 0010 to hold)
- the 35-way PLE split is **bit-identical** to upstream `get_per_layer_inputs` (a wrong column
  assignment still matches in shape and type, so nobody sees it until the golden comparison)
- the quantization target assignment is **exclusive and exhaustive** (no double rounding of the
  tied `lm_head` — ADR 0019 / 0069)
- `assert_ir_form` **actually detects** a materialized repeat_kv, a mask left as a graph input, a
  head_dim mixed up between layer types, causal and band sharing one constant, and a wrong count of
  storage dtypes
- the golden length discipline (at least one case longer than the window) and the sanity check
  actually fail
- the loading path built on meta (`load_model_and_tables`) matches the previous loading built on
  the CPU and copied in, in every parameter, every buffer and the ties (with a fake checkpoint; the
  KV-shared layers of the tiny config have no full owning layer and the upstream forward does not
  run, so every slot's value is compared instead of the output)

Only the cases that need transformers SKIP via `importorskip` (the default-sync CI does not install
the model dependencies — the 2-job setup of ADR 0065).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import export_token as token
from gemma4 import ple
from gemma4.distribution import gemma4_series_name
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue

#: Shape of the tiny subject. head_dim gets **different values** per layer type (to spring the same
#: trap as the real E2B's 256 / 512 — `hidden_size / num_attention_heads` differs from both).
HEADS = 4
KV_HEADS = 1
SLIDING_DEPTH = 16
FULL_DEPTH = 32
HIDDEN = 64
PLE_DIM = 32
VOCAB = 64
WINDOW = 4
TINY_SYM_MAX = 32
LAYER_TYPES = (
    gx.SLIDING_ATTENTION,
    gx.SLIDING_ATTENTION,
    gx.FULL_ATTENTION,
)
DEPTHS = {gx.SLIDING_ATTENTION: SLIDING_DEPTH, gx.FULL_ATTENTION: FULL_DEPTH}


# ---- Additive masks ------------------------------------------------------


class TestAdditiveCausalMask:
    def test_the_band_is_zero_below_the_diagonal_and_neg_inf_above(self):
        mask = gx.additive_causal_mask(4)

        assert tuple(mask.shape) == (1, 1, 4, 4)
        assert mask.dtype is torch.float32
        assert torch.equal(mask[0, 0].tril(), torch.zeros(4, 4))
        assert bool(torch.isneginf(mask[0, 0]).sum() == 6)

    def test_the_prefix_of_a_longer_mask_is_the_shorter_mask(self):
        """MUST: commutes with the prefix (ADR 0010) — broken, the Tmax folding silently gives
        another value."""
        long_mask = gx.additive_causal_mask(TINY_SYM_MAX)

        prefix = long_mask[:, :, :5, :5]

        assert torch.equal(prefix, gx.additive_causal_mask(5))


class TestAdditiveSlidingMask:
    def test_the_window_includes_self_and_excludes_the_row_one_step_beyond(self):
        """Equal to upstream `sliding_window_mask_function`: `0 <= q - kv < window`."""
        mask = gx.additive_sliding_mask(8, WINDOW)[0, 0]

        assert float(mask[5, 5]) == 0.0
        assert float(mask[5, 5 - (WINDOW - 1)]) == 0.0
        assert bool(torch.isneginf(mask[5, 5 - WINDOW]))

    def test_the_future_is_blocked(self):
        mask = gx.additive_sliding_mask(8, WINDOW)[0, 0]

        assert bool(torch.isneginf(mask[3, 4]))

    def test_a_short_sequence_makes_the_band_identical_to_causal(self):
        """For T <= window the band = causal. Exactly why the goldens need a case longer than the
        window."""
        assert torch.equal(
            gx.additive_sliding_mask(WINDOW, WINDOW), gx.additive_causal_mask(WINDOW)
        )

    def test_a_long_sequence_makes_the_band_differ_from_causal(self):
        assert not torch.equal(
            gx.additive_sliding_mask(WINDOW + 1, WINDOW), gx.additive_causal_mask(WINDOW + 1)
        )

    def test_the_prefix_of_a_longer_mask_is_the_shorter_mask(self):
        """MUST: the band also commutes with the prefix (the same ADR 0010 condition as causal)."""
        long_mask = gx.additive_sliding_mask(TINY_SYM_MAX, WINDOW)

        prefix = long_mask[:, :, :7, :7]

        assert torch.equal(prefix, gx.additive_sliding_mask(7, WINDOW))


# ---- The 35-way PLE split ------------------------------------------------


def _packed_table() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(VOCAB, len(LAYER_TYPES) * PLE_DIM)


def _split_tables(packed: torch.Tensor) -> nn.ModuleList:
    return nn.ModuleList(
        [
            nn.Embedding.from_pretrained(
                packed[:, index * PLE_DIM : (index + 1) * PLE_DIM].contiguous(), freeze=True
            )
            for index in range(len(LAYER_TYPES))
        ]
    )


class TestPerLayerInputs:
    """The column assignment (layer i ↔ columns `[i*D, (i+1)*D)`) matches upstream's reshape."""

    @staticmethod
    def _reference(packed: torch.Tensor, ids: torch.Tensor, scale: float) -> torch.Tensor:
        """Upstream `get_per_layer_inputs`'s formula as is (single lookup → scale → reshape)."""
        looked_up = nn.functional.embedding(ids, packed) * scale
        return looked_up.reshape(*ids.shape, len(LAYER_TYPES), PLE_DIM)

    def test_the_split_reproduces_the_packed_lookup(self):
        packed = _packed_table()
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)
        scale = float(PLE_DIM) ** 0.5

        rebuilt = ple.per_layer_inputs(_split_tables(packed), ids, scale)

        assert torch.equal(rebuilt, self._reference(packed, ids, scale))

    def test_a_wrong_layer_order_would_be_visible(self):
        """Fault injection: swapping the layer order breaks the comparison above (evidence that it
        is not a tautology)."""
        packed = _packed_table()
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)
        scale = float(PLE_DIM) ** 0.5
        tables = _split_tables(packed)
        swapped = nn.ModuleList([tables[1], tables[0], tables[2]])

        rebuilt = ple.per_layer_inputs(swapped, ids, scale)

        assert not torch.equal(rebuilt, self._reference(packed, ids, scale))


def _tiny_text_config():
    """A tiny text config whose PLE table is the probe model's ({@link ple.PLE_PROBE_ROWS} rows)."""
    transformers = pytest.importorskip("transformers")
    return transformers.Gemma4TextConfig(
        vocab_size=VOCAB,
        vocab_size_per_layer_input=ple.PLE_PROBE_ROWS,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=SLIDING_DEPTH,
        global_head_dim=FULL_DEPTH,
        hidden_size_per_layer_input=PLE_DIM,
        layer_types=list(LAYER_TYPES),
        num_kv_shared_layers=1,
        sliding_window=WINDOW,
        use_double_wide_mlp=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        max_position_embeddings=256,
    )


@pytest.fixture
def tiny_model_and_tables():
    """`(model, tables, probe)` — the probe model loaded with the **scattered** probe rows of the
    split source."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.Gemma4ForCausalLM(_tiny_text_config()).to(torch.float32).eval()
    packed = _packed_table()
    tables = _split_tables(packed)
    probe = ple.probe_rows(VOCAB)
    with torch.no_grad():
        model.model.embed_tokens_per_layer.weight.copy_(
            packed[torch.tensor(probe, dtype=torch.int64)]
        )
    return model, tables, probe


class TestProbeRows:
    def test_the_probe_scatters_across_both_ends_and_the_middle(self):
        """The rows are scattered (Codex wave H finding H-05 — 8 consecutive rows touch only one
        block)."""
        probe = ple.probe_rows(262144)

        assert len(probe) == ple.PLE_PROBE_ROWS
        assert len(set(probe)) == ple.PLE_PROBE_ROWS
        assert 0 in probe and 262143 in probe
        # Both sides of the first block boundary and a middle pair.
        assert ple.PLE_ROW_BLOCK - 1 in probe and ple.PLE_ROW_BLOCK in probe
        assert 262144 // 2 in probe

    def test_a_tiny_table_still_yields_the_full_count(self):
        """Even a tiny table without block boundaries has the fixed count (the probe model's row
        count is baked into its config)."""
        probe = ple.probe_rows(VOCAB)

        assert len(probe) == ple.PLE_PROBE_ROWS
        assert len(set(probe)) == ple.PLE_PROBE_ROWS
        assert all(0 <= row < VOCAB for row in probe)
        assert 0 in probe and VOCAB - 1 in probe

    def test_a_table_shorter_than_the_probe_fails_loudly(self):
        with pytest.raises(ValueError, match="fewer than the"):
            ple.probe_rows(ple.PLE_PROBE_ROWS - 1)


class TestAssertPerLayerSplit:
    def test_a_faithful_split_passes(self, tiny_model_and_tables):
        model, tables, probe = tiny_model_and_tables

        ple.assert_per_layer_split(model, tables, probe)

    def test_a_shuffled_split_fails_loudly(self, tiny_model_and_tables):
        """MUST: a wrong column assignment fails here (it matches in shape and type, so later
        stages cannot see it)."""
        model, tables, probe = tiny_model_and_tables
        shuffled = nn.ModuleList([tables[2], tables[0], tables[1]])

        with pytest.raises(AssertionError, match="not bit-identical"):
            ple.assert_per_layer_split(model, shuffled, probe)

    def test_a_split_of_the_wrong_width_fails_loudly(self, tiny_model_and_tables):
        model, tables, probe = tiny_model_and_tables
        short = nn.ModuleList([tables[0], tables[1]])

        with pytest.raises(AssertionError, match="split shape"):
            ple.assert_per_layer_split(model, short, probe)


# ---- Loading path (meta construction + assign) ---------------------------


#: The KV-shared layer of the fake checkpoint (`num_kv_shared_layers=1` of `_tiny_text_config` → the
#: last layer).
SHARED_LAYER = len(LAYER_TYPES) - 1


def _full_table_config():
    """A tiny config with the full PLE table ({@link VOCAB} rows) — the checkpoint side's shape."""
    config = _tiny_text_config()
    config.vocab_size_per_layer_input = VOCAB
    return config


def _write_checkpoint(directory: Path) -> Path:
    """Write a bf16 fake checkpoint with the multimodal spelling (`model.language_model.*`).

    Like upstream's official checkpoint, the tied `lm_head.weight` is not included, the KV-shared
    layer gets k/v residue, and one vision tower key is mixed in besides the text (so the skipping
    path is exercised too).
    """
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    source = transformers.Gemma4ForCausalLM(_full_table_config())
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in source.state_dict().items():
        if name == gx.LM_HEAD_KEY:
            continue
        # Left at random initialization, norm / layer_scalar would be constant 1s and a mix-up
        # would not show in the values.
        noisy = tensor + 0.01 * torch.randn(tensor.shape)
        key = gx.CHECKPOINT_TEXT_PREFIX + name[len(gx.MODEL_TEXT_PREFIX) :]
        tensors[key] = noisy.to(torch.bfloat16).contiguous()
    owner = f"{gx.CHECKPOINT_TEXT_PREFIX}layers.0.self_attn"
    residue = f"{gx.CHECKPOINT_TEXT_PREFIX}layers.{SHARED_LAYER}.self_attn"
    for leaf in ("k_proj.weight", "v_proj.weight", "k_norm.weight"):
        tensors[f"{residue}.{leaf}"] = tensors[f"{owner}.{leaf}"].clone()
    tensors["model.vision_tower.stray.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    path = directory / gx.CHECKPOINT_FILE
    save_file(tensors, str(path))
    return path


def _conventional_load(model_file: Path) -> nn.Module:
    """The loading before the change (random initialization on the CPU, then copying in with
    `load_state_dict`) — the control for identity."""
    transformers = pytest.importorskip("transformers")
    gx.register_attention()
    config = _full_table_config()
    probe = ple.probe_rows(VOCAB)
    config.vocab_size_per_layer_input = ple.PLE_PROBE_ROWS
    config._attn_implementation = gx.ATTENTION_NAME
    model = transformers.Gemma4ForCausalLM(config).eval()
    state, _ = gx.renamed_state(model, model_file, probe)
    model.load_state_dict(state)
    return model


def _slots(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ]


class TestLoadModelAndTables:
    """Loading with meta construction + `assign=True` is identical to the previous loading in every
    slot of the model."""

    @pytest.fixture
    def loaded(self, tmp_path, monkeypatch):
        model_file = _write_checkpoint(tmp_path)
        monkeypatch.setattr(gx, "load_text_config", lambda _: _full_table_config())
        model, tables = gx.load_model_and_tables(tmp_path)
        return model, tables, _conventional_load(model_file)

    def test_every_slot_is_real_and_equal_to_the_conventional_load(self, loaded):
        model, _, reference = loaded
        expected = _slots(reference)

        actual = _slots(model)

        assert [name for name, _ in actual] == [name for name, _ in expected]
        for (name, tensor), (_, want) in zip(actual, expected, strict=True):
            assert tensor.device.type == "cpu", name
            assert tensor.dtype == want.dtype, name
            assert tensor.requires_grad == want.requires_grad, name
            assert tensor.is_contiguous(), name
            assert torch.equal(tensor, want), name

    def test_the_non_persistent_buffers_keep_their_kind(self, loaded):
        """If RoPE / embed_scale turned into persistent state, the state_dict's shape would
        change."""
        model, _, reference = loaded

        names = [name for name, _ in model.named_non_persistent_buffers()]

        assert names == [name for name, _ in reference.named_non_persistent_buffers()]
        assert any(name.endswith("inv_freq") for name in names)

    def test_the_head_stays_tied_to_the_embedding(self, loaded):
        """MUST: split, torch.export would emit the tied data as 2 initializers."""
        model, _, _ = loaded

        assert model.lm_head.weight is model.model.embed_tokens.weight


class TestAssertMaterialized:
    """The check after loading actually rejects missing data, uninitialized buffers and an untied
    tie."""

    @pytest.fixture
    def model(self):
        transformers = pytest.importorskip("transformers")
        return transformers.Gemma4ForCausalLM(_tiny_text_config()).eval()

    def test_a_faithfully_built_model_passes(self, model):
        gx.assert_materialized(model)

    def test_a_slot_left_on_meta_fails_loudly(self, model):
        model.model.norm.weight = nn.Parameter(torch.empty(HIDDEN, device="meta"))

        with pytest.raises(AssertionError, match="meta"):
            gx.assert_materialized(model)

    def test_a_buffer_the_upstream_init_did_not_fill_fails_loudly(self, model):
        model.model.embed_tokens.embed_scale.fill_(float("nan"))

        with pytest.raises(AssertionError, match="not filled by upstream's initialization"):
            gx.assert_materialized(model)

    def test_a_split_tie_fails_loudly(self, model):
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.detach())

        with pytest.raises(AssertionError, match="is split into a Parameter"):
            gx.assert_materialized(model)


# ---- Quantization target assignment --------------------------------------


class TestQuantizationTargets:
    """The assignment of i8 (the embeddings) and i4 (the linears) is exclusive and exhaustive.

    The only modules that can become weight slots are `nn.Embedding` and `nn.Linear` (a norm's
    weight is an input of `rms_norm`, outside the eligible set of
    `emit.eligible_compressed_initializers`).
    """

    @staticmethod
    def _targets(wrapper: nn.Module) -> tuple[set[str], set[str], set[str]]:
        modules = dict(wrapper.named_modules())
        int8 = {
            name
            for name, module in modules.items()
            if gx.is_int8_module(name) and isinstance(module, nn.Embedding)
        }
        int4 = {
            name
            for name, module in modules.items()
            if gx.is_int4_module(name) and isinstance(module, nn.Linear)
        }
        weighted = {
            name for name, module in modules.items() if isinstance(module, nn.Embedding | nn.Linear)
        }
        return int8, int4, weighted

    def test_the_two_target_sets_are_disjoint(self, tiny_model_and_tables):
        """MUST: exclusive (running the tied `lm_head` through both would round it twice and drift
        the scale ledger from the real values)."""
        model, tables, _ = tiny_model_and_tables
        wrapper = gx.build_wrapper(model, tables)

        int8, int4, _ = self._targets(wrapper)

        assert int8 & int4 == set()

    def test_every_weight_slot_module_but_the_tied_head_is_covered(self, tiny_model_and_tables):
        """Exhaustive: the only gap is the tied `lm_head` (the rounding on the main embedding side
        applies to the data)."""
        model, tables, _ = tiny_model_and_tables
        wrapper = gx.build_wrapper(model, tables)

        int8, int4, weighted = self._targets(wrapper)

        assert gx.LM_HEAD_MODULE in weighted
        assert int8 | int4 == weighted - {gx.LM_HEAD_MODULE}

    def test_the_int8_side_is_the_embeddings(self, tiny_model_and_tables):
        model, tables, _ = tiny_model_and_tables
        wrapper = gx.build_wrapper(model, tables)

        int8, _, _ = self._targets(wrapper)

        assert int8 == {gx.EMBED_TOKENS_MODULE} | {
            f"{gx.PER_LAYER_PREFIX}{index}" for index in range(len(LAYER_TYPES))
        }

    def test_the_probe_table_is_gone_from_the_wrapper(self, tiny_model_and_tables):
        """MUST: leaving the probe model's PLE table in makes "an embedding that hits neither
        target"."""
        model, tables, _ = tiny_model_and_tables

        wrapper = gx.build_wrapper(model, tables)

        assert not hasattr(wrapper.model.model, "embed_tokens_per_layer")


# ---- assert_ir_form --------------------------------------------------------


TINY_IR_CONFIG = SimpleNamespace(
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=SLIDING_DEPTH,
    global_head_dim=FULL_DEPTH,
    num_hidden_layers=len(LAYER_TYPES),
    layer_types=list(LAYER_TYPES),
)

#: Storage breakdown of the positive case (i8 = the embeddings / i4 = the number of dummy linears).
STORAGE_COUNTS: Mapping[str, int] = {"i8": 4, "i4": 9}


def _ir_graph(
    *,
    kv_heads: int = KV_HEADS,
    layer_types: Sequence[str] = LAYER_TYPES,
    depth_override: Mapping[int, int] | None = None,
    single_mask: bool = False,
    storage_counts: Mapping[str, int] = STORAGE_COUNTS,
) -> IrGraph:
    """Minimal IR with `len(layer_types)` attentions (the mask is a per-layer-type Tmax constant + a
    slice)."""
    graph = IrGraph(symbols=["T"])
    graph.inputs.append(IrInput(name=gx.INPUT_IDS, dtype="i32", shape=[1, "T"]))
    kinds = {kind: "shared" if single_mask else kind for kind in set(layer_types)}
    for source in sorted(set(kinds.values())):
        constant = f"mask_const_{source}"
        graph.initializers[constant] = IrInitializer(
            tensor=f"const.{source}", storage=IrStorage(dtype="f32")
        )
        graph.values[constant] = IrValue(dtype="f32", shape=[1, 1, TINY_SYM_MAX, TINY_SYM_MAX])
        graph.values[f"mask_{source}"] = IrValue(dtype="f32", shape=[1, 1, "T", "T"])
        graph.nodes.append(
            IrNode(
                op=gx.SYM_PREFIX_SLICE_OP,
                ins=[constant],
                outs=[f"mask_{source}"],
                attrs={"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
            )
        )
    for index, layer_type in enumerate(layer_types):
        depth = (depth_override or {}).get(index, DEPTHS[layer_type])
        for slot, count in (("q", HEADS), ("k", kv_heads), ("v", kv_heads)):
            graph.values[f"{slot}{index}"] = IrValue(dtype="f32", shape=[1, count, "T", depth])
        graph.values[f"attn{index}"] = IrValue(dtype="f32", shape=[1, HEADS, "T", depth])
        graph.nodes.append(
            IrNode(
                op=gx.ATTENTION_OP,
                ins=[f"q{index}", f"k{index}", f"v{index}", f"mask_{kinds[layer_type]}"],
                outs=[f"attn{index}"],
                attrs={"scale": 1.0},
            )
        )
    for dtype, count in storage_counts.items():
        for slot in range(count):
            name = f"weight_{dtype}_{slot}"
            graph.initializers[name] = IrInitializer(
                tensor=f"w.{dtype}.{slot}", storage=IrStorage(dtype=dtype)
            )
    graph.outputs.append(f"attn{len(layer_types) - 1}")
    return graph


class TestAssertIrForm:
    def test_the_grouped_shape_and_two_masks_pass(self):
        form = gx.assert_ir_form(_ir_graph(), TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)

        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]
        assert form["attention_nodes"] == len(LAYER_TYPES)
        assert form["head_dim"] == {
            gx.SLIDING_ATTENTION: SLIDING_DEPTH,
            gx.FULL_ATTENTION: FULL_DEPTH,
        }
        assert form["storage"]["i4"] == STORAGE_COUNTS["i4"]

    def test_a_materialized_kv_shape_is_rejected(self):
        """MUST: a form where Hkv turned into H is an asset whose "numbers match but the acceptance
        lost its meaning"."""
        with pytest.raises(AssertionError, match="repeat_kv"):
            gx.assert_ir_form(
                _ir_graph(kv_heads=HEADS), TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS
            )

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _ir_graph()
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "T", "T"]))

        with pytest.raises(AssertionError, match="graph inputs are"):
            gx.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)

    def test_a_mask_constant_of_the_wrong_extent_is_rejected(self):
        """A constant shorter than Tmax is the form "silently too short once T grows to the
        limit"."""
        graph = _ir_graph()
        graph.values[f"mask_const_{gx.FULL_ATTENTION}"] = IrValue(dtype="f32", shape=[1, 1, 8, 8])

        with pytest.raises(AssertionError, match="Tmax form"):
            gx.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)

    def test_a_missing_layer_is_rejected(self):
        with pytest.raises(AssertionError, match=r"not matching the \d+ layers"):
            gx.assert_ir_form(
                _ir_graph(layer_types=LAYER_TYPES[:2]),
                TINY_IR_CONFIG,
                TINY_SYM_MAX,
                STORAGE_COUNTS,
            )

    def test_a_head_dim_from_the_wrong_layer_type_is_rejected(self):
        """MUST: D is taken per layer type (fails if global_head_dim reaches a sliding layer)."""
        graph = _ir_graph(depth_override={0: FULL_DEPTH})

        with pytest.raises(AssertionError, match="head_dim"):
            gx.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)

    def test_one_mask_constant_shared_by_both_layer_types_is_rejected(self):
        """Causal and band folded into one constant = the window has no effect (for T <= 512 the
        numbers even match)."""
        with pytest.raises(AssertionError, match="share the mask constants"):
            gx.assert_ir_form(
                _ir_graph(single_mask=True), TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS
            )

    def test_a_storage_census_that_misses_a_compressed_weight_is_rejected(self):
        """MUST: a weight that falls outside eligibility **silently stays f32**, so the count
        catches it."""
        graph = _ir_graph(storage_counts={"i8": STORAGE_COUNTS["i8"], "i4": 1})

        with pytest.raises(AssertionError, match="storage dtype counts"):
            gx.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)


# ---- Golden length discipline / sanity -----------------------------------


def _case(name: str, length: int) -> tuple[str, torch.Tensor]:
    return name, torch.zeros(1, length, dtype=torch.int64)


class TestAssertCaseLengths:
    def test_a_case_set_with_a_long_case_passes(self):
        gx.assert_case_lengths((_case("short", 6), _case("long", WINDOW + 1)), TINY_SYM_MAX, WINDOW)

    def test_a_case_beyond_the_symbolic_range_fails_loudly(self):
        with pytest.raises(ValueError, match="outside the symbolic dimension's range"):
            gx.assert_case_lengths((_case("long", TINY_SYM_MAX + 1),), TINY_SYM_MAX, WINDOW)

    def test_a_case_set_that_never_exceeds_the_window_fails_loudly(self):
        """MUST: without going past the window the band mask equals causal and the sliding layers
        go unverified."""
        with pytest.raises(ValueError, match="sliding_window"):
            gx.assert_case_lengths((_case("short", WINDOW),), TINY_SYM_MAX, WINDOW)


#: The subjects of `_sanity` (case name → expected token id) and their display names.
SANITY_EXPECTED = {"a": 10, "b": 20}
SANITY_LABELS = {10: "▁x", 20: "▁y", 30: "▁z"}


class TestSanity:
    def test_matching_greedy_tokens_pass(self):
        result = gx._sanity({"a": 10, "b": 20}, SANITY_EXPECTED, SANITY_LABELS)

        assert result == {"a": "10(▁x)", "b": "20(▁y)"}

    def test_a_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="differs from the expected continuation"):
            gx._sanity({"a": 30, "b": 20}, SANITY_EXPECTED, SANITY_LABELS)

    def test_a_constant_output_fails_loudly(self):
        """MUST: the detection line that remains even with the expectation table relaxed (both
        cases have the same top-1 = constant output)."""
        with pytest.raises(AssertionError, match="constant output"):
            gx._sanity({"a": 10, "b": 10}, {"a": 10, "b": 10}, SANITY_LABELS)

    def test_a_case_set_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="case sets disagree"):
            gx._sanity({"a": 10}, SANITY_EXPECTED, SANITY_LABELS)


class TestGreedyTokens:
    def test_the_last_position_is_the_one_that_is_read(self):
        """MUST: the last position. Looking at another position would compare a quantity that is
        not "the prediction of the continuation"."""
        logits = torch.zeros(1, 3, 5)
        logits[0, 0, 4] = 1.0
        logits[0, -1, 2] = 1.0

        assert gx.greedy_tokens({"case": logits}) == {"case": 2}


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
        assert set(gx.expected_token_ids(_StubTokenizer([7])).values()) == {7}

    def test_a_multi_token_continuation_fails_loudly(self):
        """MUST: with several tokens "agreement of the top-1" has no defined meaning (do not
        silently look at the first one only)."""
        with pytest.raises(AssertionError, match="unless it is a single token"):
            gx.expected_token_ids(_StubTokenizer([7, 8]))


class TestGoldenCases:
    def test_every_case_has_a_greedy_expectation(self):
        """MUST: do not mix in cases without an expectation (sanity must not look at only some of
        the cases)."""
        assert {name for name, _ in gx.GOLDEN_CASES} == set(gx.GREEDY_EXPECTATIONS)

    def test_the_expectations_are_not_all_the_same_token(self):
        """The condition under which the constant-output detection line is not a tautology (with
        only one kind of expectation the later stage of `_sanity` dies)."""
        assert len(set(gx.GREEDY_EXPECTATIONS.values())) > 1


class TestModelAxis:
    """`--model e2b|e4b` decides only the default location (the shape is derived from the config —
    the script has no branch)."""

    @staticmethod
    def _resolve(parser, argv):
        return gx.resolve_series_args(parser.parse_args(argv))

    def test_the_default_model_keeps_the_e2b_paths(self):
        """Without `--model` the default is the same location as when only E2B existed."""
        resolved = self._resolve(gx.series_parser("one-shot", gx.series_dir), [])

        assert resolved["model_dir"] == gx.DEFAULT_MODEL_DIR
        assert resolved["out"] == gx.DEFAULT_OUT_DIR
        assert gx.DEFAULT_MODEL_DIR.name == "gemma-4-E2B-it"
        assert gx.DEFAULT_OUT_DIR.name == "gemma4-e2b"

    def test_e4b_reads_its_own_checkpoint_and_writes_its_own_series(self):
        resolved = self._resolve(gx.series_parser("one-shot", gx.series_dir), ["--model", "e4b"])

        assert resolved["model_dir"].name == "gemma-4-E4B-it"
        assert resolved["out"].name == "gemma4-e4b"
        # The model name only decides the location and is not passed to `export_series`.
        assert "model" not in resolved

    def test_explicit_paths_win_over_the_model_default(self, tmp_path):
        resolved = self._resolve(
            gx.series_parser("one-shot", gx.series_dir),
            ["--model", "e4b", "--model-dir", str(tmp_path / "m"), "--out", str(tmp_path / "o")],
        )

        assert (resolved["model_dir"], resolved["out"]) == (tmp_path / "m", tmp_path / "o")

    @pytest.mark.parametrize(("argv", "steps"), [([], 16), (["--model", "e4b"], 10)])
    def test_the_greedy_steps_default_follows_the_model(self, monkeypatch, argv, steps):
        """E4B's golden stops before the step whose margin falls under the floor (ruling 47)."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(decode, "export_series", lambda *a, **o: captured.update(o) or {})
        decode.main(argv)

        assert captured["steps"] == steps

    def test_an_explicit_steps_wins_over_the_model_default(self, monkeypatch):
        captured: dict[str, object] = {}
        monkeypatch.setattr(decode, "export_series", lambda *a, **o: captured.update(o) or {})
        decode.main(["--model", "e4b", "--steps", "3"])

        assert captured["steps"] == 3

    def test_an_unknown_model_is_refused(self, capsys):
        with pytest.raises(SystemExit):
            gx.series_parser("one-shot", gx.series_dir).parse_args(["--model", "e8b"])

        assert "--model" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("script", "suffix"),
        [(decode, "decode"), (token, "decode-token"), (product, "product-bf16")],
    )
    def test_the_chunk_series_names_follow_the_model(self, script, suffix):
        """A chunk series is `gemma4-<model>-<suffix>` (the same spelling as the distribution
        recipe's `gemma4_series_name`)."""
        assert script.DEFAULT_OUT_DIR.name == gemma4_series_name("e2b", suffix)
        assert gx.series_dir("e4b", suffix).name == gemma4_series_name("e4b", suffix)

    def test_the_product_series_follows_the_model_and_takes_its_own_goldens(
        self, monkeypatch, capsys
    ):
        """`--model e4b` reads E4B's checkpoint, writes E4B's bf16 product series and takes E4B's
        greedy steps — the series borrows no goldens, so no `reference` reaches the export."""
        captured: dict[str, object] = {}

        def run(*args, **options):
            captured.update(options, args=args)
            return {}

        monkeypatch.setattr(product, "export_series", run)
        product.main(["--model", "e4b"])
        capsys.readouterr()

        *_, model_dir, out = captured["args"]
        assert model_dir.name == "gemma-4-E4B-it"
        assert out.name == "gemma4-e4b-product-bf16"
        assert captured["steps"] == decode.GREEDY_STEPS["e4b"]
        assert "reference" not in captured

    @pytest.mark.parametrize(("script", "series"), [(token, "gemma4-e4b-decode-token")])
    def test_the_borrowed_goldens_default_to_the_same_models_decode_series(
        self, monkeypatch, capsys, script, series
    ):
        """For a series that takes no goldens, the `--reference` default is the logits opt-in
        series of **the same model**.

        The tokenizer is identical, so reusing E2B's goldens for E4B would still pass the prompt
        comparison — a mix-up of the default does not fail at export time.
        """
        captured: dict[str, object] = {}

        def run(*args, **options):
            captured.update(options, args=args)
            return {}

        # Each CLI looks up the module's `export_series` at call time (token-only goes through
        # `export_decode.run_variant_cli`).
        monkeypatch.setattr(decode if script is token else script, "export_series", run)
        script.main(["--model", "e4b"])
        capsys.readouterr()

        *_, model_dir, out = captured["args"]
        assert model_dir.name == "gemma-4-E4B-it"
        assert out.name == series
        assert captured["reference"].name == "gemma4-e4b-decode"
