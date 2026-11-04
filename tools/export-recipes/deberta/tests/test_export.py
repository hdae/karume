"""Script-level contract of `deberta/export.py` (the part that needs no real weights).

The real-weight emit is manual (see the README). What is pinned here is only the discipline that
turns into a **false PASS** when it breaks:

- the series are separate per dtype (the f32 net does not silently fall on i8 assets)
- `--act-quant` does not pass without `--dtype i8` (a mirror of f32 assets cannot be reproduced)
- the prefix of the mirror io is **not caught** by the Deno-side enumeration of plain cases
  (startsWith `io.`)
- the plain golden io is taken **without hooks** (with them attached, the expected values of the
  w8 E2E are contaminated)
- the provenance record of the symbolic dimension is written with the same spelling and the same
  fields as the reader (`sbv2.distribution`) (DeBERTa's Tmax cannot be read from the artifact, so
  the record is its only carrier)
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from deberta import export as export_deberta
from karume.pipeline import export_to_file


class TinyText(nn.Module):
    """The smallest skeleton of `HiddenStatesWrapper` (`(input_ids, attention_mask) → tuple`).

    It has exactly one eligible linear (`in_features % 4 == 0`), so the presence of the activation
    hooks shows in the output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x = input_ids.to(torch.float32) * attention_mask.to(torch.float32)
        return (self.fc(x),)


class TinyHybrid(nn.Module):
    """The smallest skeleton of mixed i4 (eligible linear / embedding = i4 group32, the rest = i8).

    The quantization axis is 64 because of the MUST that i4 never creates a partial group (ADR 0069
    decision 2). `narrow` is an ineligible linear whose quantization axis 16 (< group 32) is **not
    divisible**, and it falls to the i8 side — with no i8 targets `fake_quant_int8` fails loudly
    and "is the split exclusive" cannot be observed, so this one also serves as the i8 side's
    occupant.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 64)
        self.fc = nn.Linear(64, 4)
        self.narrow = nn.Linear(16, 4)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        embedded = self.embed(input_ids)
        return (self.fc(embedded) + self.narrow(embedded[..., :16]),)


#: `_write_io` receives cases as `(name, graph input name → tensor)` (the real model has 4 inputs,
#: but the lookup order comes from `graph.inputs`, so a tiny 2-input model takes the same path).
CASES = (
    (
        "case0",
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            "attention_mask": torch.ones(1, 4, dtype=torch.int64),
        },
    ),
)


@pytest.fixture
def exported(tmp_path):
    """Export one tiny wrapper and return `(wrapper, graph, out_dir)`."""
    torch.manual_seed(0)
    wrapper = TinyText()
    example = tuple(CASES[0][1].values())
    graph = export_to_file(
        wrapper,
        example,
        tmp_path / export_deberta.MODEL_FILE,
        provenance=export_deberta.PROVENANCE,
        graph_name="tiny",
    )
    return wrapper, graph, tmp_path


class TestSeries:
    def test_the_default_output_root_is_a_separate_series_per_dtype(self):
        """MUST: compressed series get their own directories (ADR 0019) — sharing one would remove
        the f32 net."""
        roots = export_deberta.DEFAULT_OUT_ROOTS

        assert set(roots) == set(export_deberta.WEIGHT_DTYPES) == {"f32", "i8", "i4"}
        assert len(set(roots.values())) == len(roots)

    def test_f16_is_not_offered(self):
        """f16 is decided together with moving the SBV2 series to f16 (not filed yet) — not
        anticipated here."""
        assert "f16" not in export_deberta.WEIGHT_DTYPES

    def test_the_i4_series_is_stored_as_i8_by_default(self):
        """i4 is **mixed** (only the eligible linear / embedding are i4, the rest i8) — a
        single-dtype i4 series cannot be built.

        Since the i4 execution path is limited to the weight slots of linear / embedding (ADR 0069
        decision 5), defaulting to i4 would silently leave conv weights and weights the group size
        does not divide in f32. The default is i8, and assigning only the eligible ones with
        per-tensor overrides is the only form.
        """
        assert set(export_deberta.BASE_WEIGHT_DTYPES) == set(export_deberta.WEIGHT_DTYPES)
        assert export_deberta.BASE_WEIGHT_DTYPES["i4"] == "i8"


class TestExportProvenance:
    """The provenance record of the symbolic dimension — DeBERTa's Tmax **cannot be read from the
    artifact**, so only the record carries it.

    front / voice hold the bound as a baked constant of `sym_prefix_slice`, but DeBERTa promoted
    the relative-position index tables to graph inputs in ADR 0045 wave 3, so not one constant
    carrying the bound remains in the graph (measured on the 4 shipped containers — 0
    `sym_prefix_slice`). The distribution side's `maxTokens` is baked as a constant, so a series
    taken with a non-default `--sym-max` keeps export and distribution green while only the
    declaration becomes a lie.
    """

    def test_the_spelling_matches_the_reader(self):
        """MUST: the same spelling as the reader (`sbv2.distribution`).

        `deberta` is not a consumer of SBV2, so it does not import that side (the distribution's
        text_encoder component is a shared asset, not a coupling). With two copies, they must not
        be able to move independently.
        """
        from sbv2.distribution import EXPORT_PROVENANCE_FILE

        assert export_deberta.EXPORT_PROVENANCE_FILE == EXPORT_PROVENANCE_FILE

    def test_the_license_matches_the_card_that_redistributes_this_encoder(self):
        """MUST: the license identifier of the provenance baked into the container equals the
        source of truth on the card that redistributes these weights.

        `deberta` has no card of its own (it only bakes SBV2's `text_encoder` component), so the
        identifier's source of truth is on the consuming side. With two copies, they must not be
        able to move independently.
        """
        from sbv2.card import SBV2_TEXT_ENCODER_LICENSE

        assert export_deberta.PROVENANCE.license == SBV2_TEXT_ENCODER_LICENSE

    def test_the_weights_are_fetched_at_the_revision_the_container_names(self, monkeypatch):
        """MUST: fetching is pinned to a revision, the same single point the container claims.

        Without `revision=`, HF's main is pulled as of that moment, and on the day upstream main
        moves the `text_encoder` component is silently baked from a different revision (while the
        container keeps claiming the same provenance).
        """
        transformers = pytest.importorskip("transformers")

        seen: dict[str, object] = {}

        def fetch(model_id, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("the fetch stops here")

        monkeypatch.setattr(transformers.DebertaV2Model, "from_pretrained", fetch)

        with pytest.raises(RuntimeError, match="the fetch stops here"):
            export_deberta.load_model(export_deberta.MODEL_ID, 1)

        assert seen["revision"] == export_deberta.MODEL_REVISION
        assert export_deberta.PROVENANCE.upstream_revision == export_deberta.MODEL_REVISION

    def test_it_records_the_variant_and_the_symbolic_maximum(self, tmp_path):
        """The record's fields are exactly the two the reader compares (`target` / `sym_max`)."""
        from sbv2.distribution import SBV2_MAX_TOKENS, SBV2_TEXT_ENCODER_VARIANT

        export_deberta._write_export_provenance(
            SBV2_TEXT_ENCODER_VARIANT, export_deberta.SYM_MAX, tmp_path
        )

        record = json.loads(
            (tmp_path / export_deberta.EXPORT_PROVENANCE_FILE).read_text(encoding="utf-8")
        )
        assert record == {"target": SBV2_TEXT_ENCODER_VARIANT, "sym_max": SBV2_MAX_TOKENS}

    def test_it_carries_a_non_default_symbolic_maximum(self, tmp_path):
        """A deviating `--sym-max` stays in the record (the distribution side's comparison looks at
        this value)."""
        export_deberta._write_export_provenance("sbv2-22layer", 256, tmp_path)

        record = json.loads(
            (tmp_path / export_deberta.EXPORT_PROVENANCE_FILE).read_text(encoding="utf-8")
        )
        assert record["sym_max"] == 256

    def test_the_default_symbolic_maximum_agrees_with_the_distribution_declaration(self):
        """The script's default and the distribution's `maxTokens` are the same number (the premise
        for the record to be able to claim the default)."""
        from sbv2.distribution import SBV2_MAX_TOKENS

        assert export_deberta.SYM_MAX == SBV2_MAX_TOKENS


class TestActQuantCli:
    @pytest.mark.parametrize("dtype", ["f32", "i4"])
    def test_act_quant_requires_i8_weights(self, monkeypatch, dtype):
        """Activation i8 applies only to linears with i8-resident weights (ADR 0025 decision 1).

        i4 is rejected too — the linears of the mixed series are group32 i4 resident, so a
        per-token i8 mirror would be expected values the runtime never reproduces.
        """
        monkeypatch.setattr(
            "sys.argv", ["export_deberta.py", "--dtype", dtype, "--act-quant", "--layers", "2"]
        )

        with pytest.raises(SystemExit, match="--dtype i8"):
            export_deberta.main()

    def test_act_quant_passes_the_gate_with_i8_weights(self, monkeypatch):
        """The accepting side — confirms the check is not of the "always fails" form (no real
        weights are fetched)."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            export_deberta,
            "export_variant",
            lambda *args, **kwargs: calls.append(kwargs) or {"variant": "stub"},
        )
        monkeypatch.setattr(
            "sys.argv", ["export_deberta.py", "--dtype", "i8", "--act-quant", "--layers", "2"]
        )

        export_deberta.main()

        assert [call["dtype"] for call in calls] == ["i8"]
        assert calls[0]["act_quant"] is True


class TestMirrorIoPrefix:
    def test_the_mirror_prefix_does_not_match_the_plain_enumeration(self):
        """MUST: an `io.` prefix would pick the mirror up as a w8 golden (the Deno-side enumeration
        rule)."""
        mirror = f"{export_deberta.ACT_IO_PREFIX}case0{export_deberta.IO_SUFFIX}"

        assert not mirror.startswith(export_deberta.IO_PREFIX)
        assert mirror.endswith(export_deberta.IO_SUFFIX)


class TestFakeQuant:
    def test_f32_leaves_the_weights_untouched(self):
        torch.manual_seed(0)
        wrapper = TinyText()
        before = wrapper.fc.weight.clone()

        assert export_deberta._fake_quant("f32", wrapper) == ({}, {})
        assert torch.equal(wrapper.fc.weight, before)

    def test_i8_rounds_to_per_channel_representable_values_keyed_by_fqn(self):
        """MUST: the ledger keys are FQNs as seen from the exported module (emit's matching is
        decided here)."""
        torch.manual_seed(0)
        wrapper = TinyText()

        scales, overrides = export_deberta._fake_quant("i8", wrapper)

        assert overrides == {}, "a single-dtype i8 series needs no per-tensor storage assignments"
        assert "fc.weight" in scales, "the key is not in the FQN space of the exported module"
        scale = scales["fc.weight"]
        assert list(scale.shape) == [4, 1]
        assert torch.equal(torch.round(wrapper.fc.weight / scale) * scale, wrapper.fc.weight)

    def test_i4_splits_the_eligible_modules_and_the_rest_exclusively(self):
        """MUST: the i8 / i4 targets are exclusive (the mixed MUST of `quantize.py` — double
        rounding is a silent wrong value).

        It also checks the hole of weights left in neither (the merged ledger covers every
        weight). The **shape** of a scale tells where it went: i4 has the group shape
        `[channels, groups]`, i8 the keepdim shape of the same rank as the weight.
        """
        torch.manual_seed(0)
        wrapper = TinyHybrid()

        scales, overrides = export_deberta._fake_quant("i4", wrapper)

        # embedding is i4 too (the embedding supplement to ADR 0069 decision 5) — grouped along the
        # D axis of the vocabulary table `[V,D]`.
        assert overrides == {"fc.weight": "i4", "embed.weight": "i4"}
        assert set(scales) == {"fc.weight", "embed.weight", "narrow.weight"}
        # Quantization axis 64 / group 32 gives 2 groups — distinguishable from the keepdim shape
        # ([4, 1]) by shape.
        assert list(scales["fc.weight"].shape) == [4, 2]
        assert list(scales["embed.weight"].shape) == [8, 2]
        # The one that is not divisible stays i8 (excluded from the targets rather than failing
        # the whole configuration)
        assert list(scales["narrow.weight"].shape) == [4, 1]

    def test_a_weight_that_the_group_size_does_not_divide_stays_in_the_i8_side(self):
        """The source of truth of i4 eligibility is the emit-side rule (type × divisibility) —
        derived from measurement, not from a spelling."""
        torch.manual_seed(0)
        wrapper = TinyHybrid()

        assert export_deberta._i4_module_names(wrapper) == {"fc", "embed"}

    def test_a_table_that_is_never_looked_up_is_excluded_from_the_i4_side(self, monkeypatch):
        """A weight in an `nn.Embedding` container that is never looked up is consumed by no
        weight slot = not eligible for compression.

        Assigning it to i4 makes emit's check of explicit assignments fail with "not eligible", so
        it is excluded on the target-set side (in the real model, DeBERTa's relative-position table
        — `deberta.patch` slices it and passes it through linears).
        """
        torch.manual_seed(0)
        wrapper = TinyHybrid()
        monkeypatch.setattr(export_deberta, "NON_LOOKUP_EMBEDDINGS", frozenset({"embed"}))

        assert export_deberta._i4_module_names(wrapper) == {"fc"}
        # The excluded table is on the i8 side (the same side rounds it = the numbers stay as
        # before)
        scales, overrides = export_deberta._fake_quant("i4", wrapper)
        assert overrides == {"fc.weight": "i4"}
        assert list(scales["embed.weight"].shape) == [8, 1]


class TestMirrorIo:
    def test_mirror_io_is_written_under_its_own_prefix(self, exported):
        wrapper, graph, out_dir = exported

        written, attached = export_deberta._write_mirror_io(wrapper, graph, CASES, out_dir)

        assert attached == 1, "there is one eligible linear (in_features=4)"
        assert written == [f"{export_deberta.ACT_IO_PREFIX}case0{export_deberta.IO_SUFFIX}"]

    def test_mirror_io_differs_from_the_plain_golden(self, exported):
        """If the mirror has the same numbers as the plain io, the hooks did nothing (a path that
        raises nothing even with 0 hooks)."""
        wrapper, graph, out_dir = exported
        export_deberta._write_io(wrapper, graph, CASES, out_dir)
        export_deberta._write_mirror_io(wrapper, graph, CASES, out_dir)

        plain = load_file(out_dir / f"{export_deberta.IO_PREFIX}case0{export_deberta.IO_SUFFIX}")
        mirror = load_file(
            out_dir / f"{export_deberta.ACT_IO_PREFIX}case0{export_deberta.IO_SUFFIX}"
        )

        assert torch.equal(plain["input.input_ids"], mirror["input.input_ids"])
        assert not torch.equal(plain["output.0"], mirror["output.0"])

    def test_the_hooks_are_detached_so_the_plain_golden_stays_clean(self, exported):
        """MUST: taking the plain io with the hooks attached contaminates the expected values of
        the w8 E2E with activation quantization."""
        wrapper, graph, out_dir = exported
        export_deberta._write_io(wrapper, graph, CASES, out_dir)
        before = load_file(out_dir / f"{export_deberta.IO_PREFIX}case0{export_deberta.IO_SUFFIX}")

        export_deberta._write_mirror_io(wrapper, graph, CASES, out_dir)
        export_deberta._write_io(wrapper, graph, CASES, out_dir)
        after = load_file(out_dir / f"{export_deberta.IO_PREFIX}case0{export_deberta.IO_SUFFIX}")

        assert torch.equal(before["output.0"], after["output.0"])

    def test_zero_eligible_linears_fails_loudly(self, tmp_path):
        """Taking the mirror with 0 hooks gives "w8 numbers while meaning w8a8" (ADR 0006's
        always-on diagnostics)."""

        class NoLinear(nn.Module):
            def forward(
                self, input_ids: torch.Tensor, attention_mask: torch.Tensor
            ) -> tuple[torch.Tensor, ...]:
                return (input_ids.to(torch.float32) + attention_mask.to(torch.float32),)

        wrapper = NoLinear()
        example = tuple(CASES[0][1].values())
        graph = export_to_file(
            wrapper,
            example,
            tmp_path / export_deberta.MODEL_FILE,
            provenance=export_deberta.PROVENANCE,
            graph_name="tiny",
        )

        with pytest.raises(SystemExit, match="0 eligible linears"):
            export_deberta._write_mirror_io(wrapper, graph, CASES, tmp_path)
