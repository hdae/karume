"""Contract of the wiring of calibrated i4 (GPTQ) (the part that needs no real weights).

Calibration with real weights is manual (see the README). What is pinned here is only what turns
into a **false PASS** when it breaks:

- the stage split is bit-identical to the wrapper's forward (rounding on a drifted path still
  produces normal numbers)
- the i4-eligible weights split **exclusively and completely** into "inside the stages =
  calibrated" and "outside the stages = plain RTN"
- the scale ledger's keys live in the wrapper's FQN space (= the safetensors tensor keys)
- the calibration **actually produces a different rounding** (if it passed through untouched, the
  storage form would be the same, so the assets could not tell)

The model is a tiny skeleton that copies **the same FQNs as the real one**
(`model.embeddings.word_embeddings` / `model.encoder.rel_embeddings` /
`model.encoder.layer.<i>.…`) and **the same call shape** (a layer receives hidden and mask as
positional arguments, and only the first layer's output has the ConvLayer on it). The names are
copied too so that the checks can be tested without replacing the constants meant for the real
model ({@link deberta.export.NON_STAGE_I4_WEIGHTS} / {@link deberta.export.NON_LOOKUP_EMBEDDINGS}).
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch
from torch import nn

from deberta import calib
from deberta import export as export_deberta
from deberta.calib_texts import CALIB_TEXTS

#: The quantization axis (= the group size). i4 never creates a partial group (ADR 0069 decision 2).
HIDDEN = 32
VOCAB = 16
#: Rows of the relative-position table (the `rel_embeddings` container — never looked up, so not
#: eligible for i4).
REL_ROWS = 8


class TinyLayer(nn.Module):
    """A layer that copies only **the call shape** of `DebertaV2Layer` (receives hidden and mask as
    positional arguments)."""

    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(HIDDEN, HIDDEN)
        self.dense = nn.Linear(HIDDEN, HIDDEN)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        relative_pos: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        attended = torch.tanh(self.query(hidden) + rel_embeddings[relative_pos]) * mask
        return (self.dense(attended) + hidden, None)


class TinyConv(nn.Module):
    """Stand-in for `ConvLayer` (only on the first layer, blends the stage's input hidden back in as
    a residual).

    Like the real one, it holds no `nn.Linear` at all (`Conv1d` + `LayerNorm`) — the model side
    also keeps the stage scan set from moving because of the ConvLayer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blend = nn.Parameter(torch.full((HIDDEN,), 0.5))

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor, input_mask: torch.Tensor
    ) -> torch.Tensor:
        return residual + hidden * self.blend * input_mask.unsqueeze(-1).to(hidden.dtype)


class TinyEncoder(nn.Module):
    """An encoder that copies the loop of `DebertaV2Encoder.forward`."""

    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList(TinyLayer() for _ in range(layers))
        self.rel_embeddings = nn.Embedding(REL_ROWS, HIDDEN)
        self.conv = TinyConv()

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor, relative_pos: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        rel_embeddings = self.rel_embeddings.weight
        states = [hidden]
        next_kv = hidden
        for index, layer in enumerate(self.layer):
            output = layer(
                next_kv,
                attention_mask,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )[0]
            if index == 0:
                output = self.conv(hidden, output, attention_mask)
            states.append(output)
            next_kv = output
        return tuple(states)


class TinyEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB, HIDDEN)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class TinyModel(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.embeddings = TinyEmbeddings()
        self.encoder = TinyEncoder(layers)


class TinyWrapper(nn.Module):
    """Skeleton of `HiddenStatesWrapper` (returns every layer's hidden as a tuple)."""

    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.model = TinyModel(layers)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, relative_pos: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self.model.encoder(self.model.embeddings(input_ids), attention_mask, relative_pos)


def make_wrapper(layers: int = 2, seed: int = 0) -> TinyWrapper:
    torch.manual_seed(seed)
    wrapper = TinyWrapper(layers)
    wrapper.eval()
    return wrapper


def make_args(count: int = 3, seed: int = 7) -> tuple[tuple[torch.Tensor, ...], ...]:
    """A few calibration inputs (the wrapper's positional arguments) of different lengths (the real
    48-sentence corpus is not run)."""
    generator = torch.Generator().manual_seed(seed)
    args = []
    for index in range(count):
        length = 4 + index
        ids = torch.randint(0, VOCAB, (1, length), generator=generator)
        args.append(
            (
                ids,
                torch.ones_like(ids),
                torch.randint(0, REL_ROWS, (length,), generator=generator),
            )
        )
    return tuple(args)


def i4_weight_keys(wrapper: nn.Module) -> set[str]:
    return {f"{name}.weight" for name in export_deberta._i4_module_names(wrapper)}


class _StubTokenizer:
    """A tokenizer that only returns one id per character (the entry shape of
    `deberta.export.encode_text`)."""

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": list(range(1, len(text) + 1))}


class _StubModel(nn.Module):
    """A model with only the attributes from which `build_graph_inputs` reads the bucket size."""

    def __init__(self) -> None:
        super().__init__()
        attention = nn.Module()
        attention.position_buckets = 4
        attention.max_relative_positions = 8
        layer = nn.Module()
        layer.attention = nn.Module()
        layer.attention.self = attention
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([layer])


class TestStageSplit:
    def test_the_stage_chain_reproduces_the_wrapper_output_bit_exactly(self):
        """MUST: the split does not differ from the real path by a single bit (otherwise GPTQ of a
        different path ships)."""
        wrapper = make_wrapper()
        args = make_args()
        stages = calib.encoder_stages(wrapper)
        batches = calib.capture_stage_batches(wrapper, args)

        calib.assert_stage_split(wrapper, args[0], batches[0], stages)

    def test_a_dropped_stage_is_caught(self):
        """Dropping one stage changes the final hidden (a measurement that the check does not let
        it through)."""
        wrapper = make_wrapper()
        args = make_args()
        stages = calib.encoder_stages(wrapper)
        batches = calib.capture_stage_batches(wrapper, args)

        with pytest.raises(AssertionError, match="is not bit-identical to the wrapper output"):
            calib.assert_stage_split(wrapper, args[0], batches[0], stages[:-1])

    def test_a_stage_chain_without_the_conv_residual_is_caught(self):
        """Dropping the ConvLayer on the first layer makes the second stage onwards see a different
        hidden."""
        wrapper = make_wrapper()
        args = make_args()
        stages = calib.encoder_stages(wrapper)
        batches = calib.capture_stage_batches(wrapper, args)
        without_conv = tuple(
            (prefix, calib.EncoderStage(index, getattr(stage, stage.child), None))
            for index, (prefix, stage) in enumerate(stages)
        )

        with pytest.raises(AssertionError, match="is not bit-identical to the wrapper output"):
            calib.assert_stage_split(wrapper, args[0], batches[0], without_conv)

    def test_the_stage_prefix_lands_in_the_wrapper_fqn_space(self):
        """The prefixed local FQNs equal the wrapper's actual FQNs (where the space of the ledger
        keys is decided)."""
        wrapper = make_wrapper()
        stages = calib.encoder_stages(wrapper)

        names = calib.stage_linear_names(stages)

        assert names == {
            f"model.encoder.layer.{index}.{child}"
            for index in range(2)
            for child in ("query", "dense")
        }
        assert names <= {name for name, _module in wrapper.named_modules()}

    def test_a_forward_that_never_reaches_the_conv_fails_loudly(self):
        """A structure that has a ConvLayer the loop never calls = the stage spelling and the
        model have drifted apart.

        The sentinel never fires and the forward completes, so the inputs are not all captured.
        Carrying on silently would only end in a "zero calibration inputs" diagnostic on the core
        side, so the capture side fails.
        """

        class ConvlessEncoder(TinyEncoder):
            def forward(self, hidden, attention_mask, relative_pos):
                rel_embeddings = self.rel_embeddings.weight
                next_kv = hidden
                for layer in self.layer:
                    next_kv = layer(
                        next_kv,
                        attention_mask,
                        relative_pos=relative_pos,
                        rel_embeddings=rel_embeddings,
                    )[0]
                return (hidden, next_kv)

        wrapper = make_wrapper()
        wrapper.model.encoder = ConvlessEncoder(2)

        with pytest.raises(AssertionError, match="did not capture the first stage's inputs"):
            calib.capture_stage_batches(wrapper, make_args(count=1))

    def test_zero_calibration_inputs_fail_loudly(self):
        """MUST: no branch silently drops to plain RTN."""
        wrapper = make_wrapper()

        with pytest.raises(AssertionError, match="no calibration inputs at all"):
            calib.build_rig(wrapper, calib.encoder_stages(wrapper), ())


class TestCalibratedI4:
    def test_the_ledger_keys_live_in_the_wrapper_fqn_space(self):
        """MUST: ledger keys = safetensors tensor keys (emit's matching is decided here)."""
        wrapper = make_wrapper()

        scales, overrides = export_deberta._fake_quant("i4", wrapper, calib_args=make_args())

        parameters = dict(wrapper.named_parameters())
        assert set(overrides) <= set(parameters), "the i4 keys are not in the wrapper's FQN space"
        assert set(scales) <= set(parameters)
        assert set(overrides) == i4_weight_keys(wrapper)

    def test_the_eligible_set_is_split_exclusively_between_the_two_paths(self):
        """**Exclusive and complete** between inside the stages (calibrated) and outside (RTN) (the
        mixed MUST of `quantize.py`)."""
        wrapper = make_wrapper()
        stages = calib.encoder_stages(wrapper)
        stage_names = calib.stage_linear_names(stages)
        i4_names = export_deberta._i4_module_names(wrapper)

        assert i4_names - stage_names == export_deberta.NON_STAGE_I4_WEIGHTS
        assert stage_names <= i4_names, "linears inside the stages are missing from the i4 set"
        # The relative-position table that is never looked up stays out of the i4 side (i8 keeps it)
        assert export_deberta.NON_LOOKUP_EMBEDDINGS & i4_names == frozenset()

    def test_calibration_produces_a_different_rounding_from_plain_rtn(self):
        """If the calibration passed through untouched, the storage form would be the same and the
        assets could not tell — measured by the value difference."""
        calibrated = make_wrapper()
        plain = make_wrapper()

        export_deberta._fake_quant("i4", calibrated, calib_args=make_args())
        export_deberta._fake_quant("i4", plain)

        after = dict(calibrated.named_parameters())
        differing = sorted(
            name
            for name, weight in plain.named_parameters()
            if not torch.equal(weight, after[name])
        )
        assert differing, "GPTQ produced the same values as RTN (the calibration had no effect)"
        assert all(name.startswith("model.encoder.layer.") for name in differing)

    def test_the_lookup_table_is_rounded_by_the_plain_path_in_both_modes(self):
        """The vocabulary table is outside the stages, so both modes use plain RTN (the rounded
        values are bit-identical)."""
        calibrated = make_wrapper()
        plain = make_wrapper()

        export_deberta._fake_quant("i4", calibrated, calib_args=make_args())
        export_deberta._fake_quant("i4", plain)

        assert torch.equal(
            calibrated.model.embeddings.word_embeddings.weight,
            plain.model.embeddings.word_embeddings.weight,
        )

    def test_an_unexpected_non_stage_eligible_fails_loudly(self, monkeypatch):
        """Eligible weights outside the stages that differ from the declaration = a signal that
        the upstream structure moved (no silent reclassification)."""
        wrapper = make_wrapper()
        monkeypatch.setattr(export_deberta, "NON_STAGE_I4_WEIGHTS", frozenset())

        with pytest.raises(AssertionError, match="i4-eligible weights outside the encoder stages"):
            export_deberta._fake_quant("i4", wrapper, calib_args=make_args())

    def test_a_ledger_short_of_the_include_set_fails_loudly(self, monkeypatch):
        """Rounded set ≠ stored set means "green with i8 weights mixed into the i4 storage" — the
        last check fails it."""
        wrapper = make_wrapper()
        genuine = calib.calibrate_i4

        def short(rig, include):
            report, ledger = genuine(rig, include)
            trimmed = dict(ledger.scales)
            trimmed.pop(sorted(trimmed)[0])
            return report, replace(ledger, scales=trimmed)

        monkeypatch.setattr(calib, "calibrate_i4", short)

        with pytest.raises(AssertionError, match="i4-eligible weights, but"):
            export_deberta._fake_quant("i4", wrapper, calib_args=make_args())

    def test_a_ledger_that_overlaps_the_plain_path_fails_loudly(self, monkeypatch):
        """Rounding the same weight on 2 paths quietly corrupts only the values (the storage form
        and the counts look right)."""
        wrapper = make_wrapper()
        genuine = calib.calibrate_i4
        table = f"{sorted(export_deberta.NON_STAGE_I4_WEIGHTS)[0]}.weight"

        def overlapping(rig, include):
            report, ledger = genuine(rig, include)
            merged = dict(ledger.scales) | {table: torch.ones(VOCAB, 1)}
            return report, replace(ledger, scales=merged)

        monkeypatch.setattr(calib, "calibrate_i4", overlapping)

        with pytest.raises(AssertionError, match="double rounding"):
            export_deberta._fake_quant("i4", wrapper, calib_args=make_args())


class TestCorpusWiring:
    def test_the_i4_series_is_calibrated_by_default(self):
        """MUST: the CLI's `--dtype i4` is calibrated without any flag (the only opt-out is an
        argument for tests)."""
        default = inspect.signature(export_deberta.export_variant).parameters["calib_texts"].default

        assert default is CALIB_TEXTS

    def test_the_calibration_inputs_go_through_the_golden_input_path(self, monkeypatch):
        """MUST: the calibration inputs go through the same input-building path as the goldens
        (built with a different spelling, the activations would silently diverge)."""
        lengths: list[int] = []

        def fake_tables(length, *, position_buckets, max_position):
            lengths.append(length)
            table = torch.zeros(length, length, dtype=torch.int64)
            return table, table

        monkeypatch.setattr(export_deberta.patch, "build_rel_pos_tables", fake_tables)

        args = export_deberta.build_calib_args(_StubTokenizer(), _StubModel(), ("あい", "うえお"))

        assert lengths == [2, 3], "the index tables are not built at the actual length"
        assert [tuple(item.shape) for item in args[0]] == [(1, 2), (1, 2), (2, 2), (2, 2)]
        # The positional argument order is INPUT_ORDER itself (they are passed to the wrapper by
        # position, so a swap would go unnoticed)
        assert torch.equal(args[1][0], torch.tensor([[1, 2, 3]]))
        assert torch.equal(args[1][1], torch.ones(1, 3, dtype=torch.int64))
        assert export_deberta.INPUT_ORDER == ("input_ids", "attention_mask", "c2p_pos", "p2c_pos")

    def test_an_empty_corpus_fails_loudly(self):
        with pytest.raises(ValueError, match="calibration corpus is empty"):
            export_deberta.build_calib_args(_StubTokenizer(), _StubModel(), ())
