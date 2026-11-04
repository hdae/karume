"""Script-level contracts of `irodori/export.py` (the part that needs no real weights).

Emitting from the real weights is manual (the same discipline as the tests of the existing
`deberta/export.py` / `embeddinggemma/export.py`). Pinned here is only the discipline whose
breakage would give a **false pass**:

- The reference (the golden's expectation) can only be taken **before patching** (if it could be
  taken after, the equivalence check would become tautological)
- The check of the static scheme (no runtime attention_mask) really fails when the scheme breaks
- Mixing up the text / caption projectors (reading the same weights twice) fails in `_sanity`
- `_write_io` writes no io that disagrees with the IR's input names / no io with a different
  number of outputs
- The second output of `caption-proj` is the sequence with `caption_norm` applied, and the
  first output stays the plain projector output (the same formula as `text-proj`)
- Mixing up `text_norm` / `caption_norm` (reading the same weights twice) fails in
  `_norm_divergence`
- The check that the speaker output without a reference (mask all 0) is **exactly 0** fails when
  it is not 0
- The speaker cases from the real latent fail on a missing asset or a shape mismatch **instead
  of turning into synthetic ones**
- The check that `duration` does not depend on `aux_features` fails if it does
- The check that the cond / three uncond variants of `dit` **differ from each other** fails when
  they are the same
- Right-padding a condition state beyond its declared length fails
- The pair from splitting the DiT into `dit-context` / `dit` (ADR 0114) is **bit-identical** to
  upstream's `forward_with_encoded_conditions`, the two split ownership into "the
  condition-side projections / the rest", and the IR's boundary names follow the contract's
  spelling
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from irodori import export as ir
from irodori import patch as patch_irodori
from irodori import pipeline_ref as ip
from irodori.distribution import _assert_irodori_dit_boundary, irodori_context_kv_names
from karume.dist import ir_graph
from karume.ir import IrGraph, IrInput, IrNode, IrValue
from karume.pipeline import export_to_file, publish_model
from karume.quantize import quantize_to_int8

#: A minimal config of what `_static_scheme_evidence` / `build_cases` read.
TEXT_CONFIG = {"pad_token_id": 3, "bos_token_id": 1}
MODEL_CONFIG = {"max_text_len": 16, "max_caption_len": 32}

CASES = (
    ("short", "text", torch.tensor([[1, 5, 6]], dtype=torch.int64)),
    ("cap", "caption", torch.tensor([[1, 7, 8, 9]], dtype=torch.int64)),
)


class MaskRespectingBackbone(nn.Module):
    """A backbone stand-in that **correctly ignores** padding (the side where the static scheme
    holds).

    The output depends only on "the embedding with the masked positions set to 0", so adding
    padding leaves the first T rows unchanged — the same property the real bidirectional mask
    has.
    """

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4)
        return hidden * mask.unsqueeze(-1).to(torch.float32)


class PadLeakingBackbone(nn.Module):
    """A backbone stand-in where **the amount of padding leaks into the output** (fault injection
    for the side where the static scheme fails)."""

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        base = MaskRespectingBackbone()(input_ids, mask)
        return base + float(input_ids.shape[1])


class ScalingNorm(nn.Module):
    """A stand-in for `caption_norm` (a constant factor — the value shows whether it was
    applied)."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((4,), scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class DoublingProjector(nn.Module):
    """A stand-in for the real projector `_pristine_outputs` calls (it takes
    `(backbone, ids, mask)`)."""

    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, backbone: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return backbone(ids, mask) * self.gain


class TestPristineReferenceOrdering:
    """MUST: the reference can only be taken before patching (otherwise the equivalence check
    becomes tautological and gives a false pass)."""

    def test_taking_a_reference_after_patching_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", True)

        with pytest.raises(
            AssertionError, match="Tried to take a reference after the patches were applied"
        ):
            ir._pristine_outputs(MaskRespectingBackbone(), {}, ScalingNorm(2.0), CASES)

    def test_taking_a_reference_before_patching_is_allowed(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)

        outputs = ir._pristine_outputs(MaskRespectingBackbone(), {}, ScalingNorm(2.0), CASES)

        assert set(outputs[ir.TARGET_BACKBONE]) == {"short", "cap"}
        assert tuple(outputs[ir.TARGET_BACKBONE]["short"][0].shape) == (1, 3, 4)

    def test_only_the_caption_projector_gets_a_second_output(self, monkeypatch):
        """MUST: only the caption side gets a second output (one on the text side would change
        the `duration` chain)."""
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        projectors = {
            ir.TARGET_TEXT_PROJ: DoublingProjector(1.0),
            ir.TARGET_CAPTION_PROJ: DoublingProjector(3.0),
        }

        outputs = ir._pristine_outputs(
            MaskRespectingBackbone(), projectors, ScalingNorm(2.0), CASES
        )

        assert len(outputs[ir.TARGET_TEXT_PROJ]["short"]) == 1
        caption = outputs[ir.TARGET_CAPTION_PROJ]["short"]
        assert len(caption) == 2
        # Second output = norm(first output). The first output itself stays before the norm.
        assert torch.equal(caption[1], caption[0] * 2.0)


def _backbone_first(backbone: nn.Module, cases) -> dict[str, torch.Tensor]:
    """Take only the first output from the backbone side of `_pristine_outputs` (the form the
    downstream of the chain consumes)."""
    return ir._first(
        ir._pristine_outputs(backbone, {}, ScalingNorm(2.0), cases)[ir.TARGET_BACKBONE]
    )


class TestStaticSchemeEvidence:
    """Checks that the check of the static scheme (the host removes the right padding) is not
    tautological."""

    def test_a_mask_respecting_backbone_measures_zero(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        backbone = MaskRespectingBackbone()
        pristine = _backbone_first(backbone, CASES)

        evidence = ir._static_scheme_evidence(backbone, TEXT_CONFIG, MODEL_CONFIG, CASES, pristine)

        assert evidence == {"short": 0.0, "cap": 0.0}

    def test_a_pad_leaking_backbone_fails_loudly(self, monkeypatch):
        """MUST: fails when the scheme is broken — a check that stops instead of working around
        it."""
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        backbone = PadLeakingBackbone()
        pristine = _backbone_first(backbone, CASES)

        with pytest.raises(AssertionError, match="the static scheme"):
            ir._static_scheme_evidence(backbone, TEXT_CONFIG, MODEL_CONFIG, CASES, pristine)

    def test_a_case_longer_than_its_family_cap_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        long_case = (("long", "text", torch.arange(1, 40, dtype=torch.int64).unsqueeze(0)),)
        backbone = MaskRespectingBackbone()
        pristine = _backbone_first(backbone, long_case)

        with pytest.raises(SystemExit, match="upper bound"):
            ir._static_scheme_evidence(backbone, TEXT_CONFIG, MODEL_CONFIG, long_case, pristine)


class MaskZeroingEncoder(nn.Module):
    """A stand-in for the reference encoder that sets masked positions to **exactly 0** (the same
    property as the real one)."""

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = latent.sum(dim=-1, keepdim=True).expand(-1, -1, 3) + 1.0
        return hidden * mask.unsqueeze(-1).to(torch.float32)


class MaskLeakingEncoder(nn.Module):
    """A stand-in that returns non-zero even with an all-zero mask (fault injection — the side
    where the host's zero input does not hold)."""

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return MaskZeroingEncoder()(latent, mask) + 1e-3


class TestNoReferenceEvidence:
    """MUST: do not leave "no reference gives 0" as a claim (the only basis for the host's zero
    input)."""

    def test_a_mask_zeroing_encoder_measures_zero(self):
        assert ir._no_reference_evidence(MaskZeroingEncoder(), 4, 4) == 0.0

    def test_a_leaking_encoder_fails_loudly(self):
        with pytest.raises(AssertionError, match="without a reference"):
            ir._no_reference_evidence(MaskLeakingEncoder(), 4, 4)


class AuxReadingPredictor(nn.Module):
    """A duration stand-in that reads `aux_features` (fault injection)."""

    def forward(self, text_state: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return text_state.sum(dim=(1, 2)) + kwargs["aux_features"].sum(dim=1)


class AuxIgnoringPredictor(nn.Module):
    """A duration stand-in that does not read `aux_features` (the same property as the token-sum
    form)."""

    def forward(self, text_state: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return text_state.sum(dim=(1, 2))


DURATION_REFERENCE = {
    "a": {
        "text_state": torch.ones(1, 3, 2),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
        "speaker_state": torch.zeros(1, 2, 2),
        "speaker_mask": torch.ones(1, 2, dtype=torch.bool),
        "has_speaker": torch.ones(1, dtype=torch.bool),
        "caption_state": torch.zeros(1, 2, 2),
        "caption_mask": torch.ones(1, 2, dtype=torch.bool),
        "has_caption": torch.ones(1, dtype=torch.bool),
    }
}


class TestDurationAuxInertness:
    """MUST: dropping `aux_features` from the graph inputs is backed by a check."""

    def test_an_aux_ignoring_predictor_measures_zero(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        predictor = AuxIgnoringPredictor()
        pristine = ir._pristine_duration_outputs(predictor, DURATION_REFERENCE, 4)

        assert ir._duration_aux_is_inert(predictor, DURATION_REFERENCE, 4, pristine) == {"a": 0.0}

    def test_an_aux_reading_predictor_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        predictor = AuxReadingPredictor()
        pristine = ir._pristine_duration_outputs(predictor, DURATION_REFERENCE, 4)

        with pytest.raises(AssertionError, match="aux_features"):
            ir._duration_aux_is_inert(predictor, DURATION_REFERENCE, 4, pristine)

    def test_taking_a_reference_after_patching_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", True)

        with pytest.raises(
            AssertionError, match="Tried to take a reference after the patches were applied"
        ):
            ir._pristine_duration_outputs(AuxIgnoringPredictor(), DURATION_REFERENCE, 4)


class TestSpeakerCases:
    def test_the_declared_lengths_are_deterministic_and_in_range(self):
        first = ir.build_speaker_cases(4, 750)
        second = ir.build_speaker_cases(4, 750)

        assert [name for name, _ in first] == [name for name, _, _ in ir.SPEAKER_CASES]
        for (_name, lhs), (_same, rhs) in zip(first, second, strict=True):
            assert torch.equal(lhs, rhs)

    def test_a_case_over_the_symbolic_cap_fails_loudly(self):
        with pytest.raises(SystemExit, match="outside the symbolic dimension's range"):
            ir.build_speaker_cases(4, 8)


def _fake_patch(seq, mask, patch_size):
    """A stand-in for `patch_sequence_with_mask` (drops the remainder and groups `patch_size`
    rows at a time).

    The real one is injected from the upstream implementation
    (`IrodoriSource.patch_sequence_with_mask`), so all this looks at is **the behavior of
    `build_real_speaker_cases` itself** — the correctness of the grouping is upstream's
    responsibility, and copying it into a stand-in would mean nothing.
    """
    usable = (seq.shape[1] // patch_size) * patch_size
    bundles = usable // patch_size
    bundled = seq[:, :usable].reshape(seq.shape[0], bundles, seq.shape[2] * patch_size)
    bundled_mask = mask[:, :usable].reshape(mask.shape[0], bundles, patch_size).all(-1)
    return bundled, bundled_mask


def _write_reference_latent(latent_dir: Path, latent: torch.Tensor) -> None:
    latent_dir.mkdir(parents=True, exist_ok=True)
    path = latent_dir / f"{ir.REFERENCE_LATENT_PREFIX}{ir.REFERENCE_LATENT_CASE}{ir.IO_SUFFIX}"
    save_file({ir.REFERENCE_LATENT_KEY: latent}, str(path))


class TestRealSpeakerCases:
    """MUST: the cases from the real latent are **not replaced by synthetic ones** (the tolerance
    rests on the value range)."""

    def test_the_cases_follow_the_table_and_share_one_prefix(self, tmp_path):
        # 190 frames (7.6 s of reference audio) → 47 rows with patch 4, the same shape as the real
        # asset.
        latent = torch.arange(190 * 32, dtype=torch.float32).reshape(1, 190, 32)
        _write_reference_latent(tmp_path, latent)

        cases = ir.build_real_speaker_cases(_fake_patch, 32, 4, 128, 750, latent_dir=tmp_path)

        assert [name for name, _ in cases] == [name for name, _ in ir.SPEAKER_REAL_CASES]
        by_name = dict(cases)
        assert by_name["ref-real-full"].shape == (1, 47, 128)
        assert by_name["ref-real-short"].shape == (1, 6, 128)
        # The short one is cut from the **start** of the full length (not other random values or
        # another range).
        assert torch.equal(by_name["ref-real-short"], by_name["ref-real-full"][:, :6])

    def test_a_missing_latent_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="Real latent missing"):
            ir.build_real_speaker_cases(_fake_patch, 32, 4, 128, 750, latent_dir=tmp_path)

    def test_a_latent_of_the_wrong_width_fails_loudly(self, tmp_path):
        _write_reference_latent(tmp_path, torch.zeros((1, 190, 16)))

        with pytest.raises(SystemExit, match=r"is not \[1,S,32\]"):
            ir.build_real_speaker_cases(_fake_patch, 32, 4, 128, 750, latent_dir=tmp_path)

    def test_a_patch_size_that_misses_the_speaker_input_width_fails_loudly(self, tmp_path):
        _write_reference_latent(tmp_path, torch.zeros((1, 190, 32)))

        # With patch 2 the grouped width is 64, which disagrees with the speaker graph's input
        # 128.
        with pytest.raises(SystemExit, match="speaker's input dimension"):
            ir.build_real_speaker_cases(_fake_patch, 32, 2, 128, 750, latent_dir=tmp_path)

    def test_a_latent_shorter_than_the_table_fails_loudly(self, tmp_path):
        # 20 frames → 5 rows after patching, short of the table's 6 short rows.
        _write_reference_latent(tmp_path, torch.zeros((1, 20, 32)))

        with pytest.raises(SystemExit, match="real latent's length after patching"):
            ir.build_real_speaker_cases(_fake_patch, 32, 4, 128, 750, latent_dir=tmp_path)


def _dit_pristine(distinct: bool) -> dict[str, torch.Tensor]:
    """Dummy outputs for all `DIT_CASES` (a different value each when `distinct`).

    The grouping is left to `_dit_uncond_divergence` itself (copying it into the test would let a
    grouping mistake break both sides the same way and pass unnoticed).
    """
    return {
        name: torch.full((1, 2, 3), float(index) if distinct else 0.0)
        for index, (name, *_rest) in enumerate(ir.DIT_CASES)
    }


class TestDitUncondDivergence:
    """MUST: do not leave "the mask takes effect" as a claim (the basis for expressing uncond with
    the mask)."""

    def test_distinct_outputs_are_reported_pairwise(self):
        pairs = ir._dit_uncond_divergence(_dit_pristine(distinct=True))

        # All pairs of cond + the three uncond variants = 6 pairs.
        assert len(pairs) == 6
        assert min(pairs.values()) >= ir.DIT_UNCOND_DIVERGENCE_MIN

    def test_identical_outputs_fail_loudly(self):
        with pytest.raises(AssertionError, match="segment layout of the mask"):
            ir._dit_uncond_divergence(_dit_pristine(distinct=False))


class TestPristineDitReferenceOrdering:
    def test_taking_a_reference_after_patching_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", True)

        with pytest.raises(
            AssertionError, match="Tried to take a reference after the patches were applied"
        ):
            ir._pristine_dit_outputs(AuxIgnoringPredictor(), {})


class TestRightPad:
    """Right padding of the condition states (kept on the host, ADR 0047)."""

    def test_pads_with_zeros_and_keeps_the_head(self):
        padded = ir._right_pad(torch.ones(1, 2, 3), 5, "test")

        assert tuple(padded.shape) == (1, 5, 3)
        assert torch.equal(padded[:, :2], torch.ones(1, 2, 3))
        assert float(padded[:, 2:].abs().max()) == 0.0

    def test_a_state_longer_than_the_declared_length_fails_loudly(self):
        with pytest.raises(SystemExit, match="condition's declared length"):
            ir._right_pad(torch.ones(1, 6, 3), 5, "test")


class TestGoldenCaseBody:
    """MUST: the preprocessing of the golden cases **differs by kind** (as spelled in upstream's
    `_synthesize`).

    text gets `normalize_text` + strip, the caption **strip only**. Normalizing the caption would
    silently change the conditioning by the outer-bracket stripping, NFKC and symbol removal.
    """

    #: A spelling sensitive to normalization (outer brackets + a symbol — both removed by
    #: `normalize_text`).
    BODY = " 「①明るい声」 "

    @staticmethod
    def _spy():
        """A stand-in for `normalize_text` that records the call and returns **something
        else**."""
        calls: list[str] = []

        def normalize_text(body: str) -> str:
            calls.append(body)
            return "normalized"

        return calls, normalize_text

    def test_the_text_kind_goes_through_normalization(self):
        calls, normalize_text = self._spy()

        assert ir.golden_case_body("t", "text", self.BODY, normalize_text) == "normalized"
        assert calls == [self.BODY]

    def test_the_caption_kind_never_calls_normalization(self):
        calls, normalize_text = self._spy()

        assert ir.golden_case_body("c", "caption", self.BODY, normalize_text) == "「①明るい声」"
        assert calls == []

    def test_an_unknown_kind_fails_loudly(self):
        """If the spellings of the kinds diverge, it silently falls back to one of the two
        preprocessings."""
        _calls, normalize_text = self._spy()

        with pytest.raises(SystemExit, match="is neither text nor caption"):
            ir.golden_case_body("x", "Caption", self.BODY, normalize_text)

    def test_an_empty_body_fails_loudly(self):
        _calls, normalize_text = self._spy()

        with pytest.raises(SystemExit, match="the text is empty after preprocessing"):
            ir.golden_case_body("c", "caption", "  \n ", normalize_text)

    def test_the_existing_caption_goldens_are_insensitive_to_normalization(self):
        """MUST: checks that the existing golden does not move (if it does, that is the signal to
        re-export).

        The very premise of the change that moved the caption to strip-only. Adding a caption case
        that is sensitive to normalization makes this fail, so the need to regenerate the golden
        shows.
        """
        pytest.importorskip("irodori_tts")
        from irodori_tts.text_normalization import normalize_text

        captions = [(name, body) for name, kind, body in ir.GOLDEN_CASES if kind == "caption"]

        assert captions, "no golden case of the caption kind"
        for name, body in captions:
            assert normalize_text(body).strip() == body.strip(), name


class TestSanityCatchesProjectorMixups:
    """MUST: a mix-up that reads the same weights twice matches in shape and dtype, so it shows
    only here."""

    def test_identical_projector_outputs_fail_loudly(self):
        shared = {"a": (torch.ones(1, 3, 4),)}
        pristine = {ir.TARGET_TEXT_PROJ: shared, ir.TARGET_CAPTION_PROJ: dict(shared)}

        with pytest.raises(AssertionError, match="same weights are suspected to be read twice"):
            ir._sanity(pristine)

    def test_diverging_projector_outputs_are_reported(self):
        """MUST: the first outputs are compared (the caption side's second output is normed and
        has a different value range)."""
        pristine = {
            ir.TARGET_TEXT_PROJ: {"a": (torch.zeros(1, 3, 4),)},
            ir.TARGET_CAPTION_PROJ: {"a": (torch.full((1, 3, 4), 2.0), torch.full((1, 3, 4), 9.0))},
        }

        assert ir._sanity(pristine) == {"a": 2.0}


class TinyResidualProjector(nn.Module):
    """A residual_mlp stand-in with only the attributes `ProjectorGraph` reads."""

    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Linear(4, 4, bias=False)
        self.residual_norm = ScalingNorm(1.25)
        self.residual_up = nn.Linear(4, 4, bias=False)
        self.residual_down = nn.Linear(4, 4, bias=False)


class TestCaptionProjectorGraph:
    """MUST: the first output keeps the same formula as `text-proj` (the basis for bit identity
    with the existing golden)."""

    def test_the_first_output_is_the_plain_projection(self):
        torch.manual_seed(0)
        projector = TinyResidualProjector()
        hidden = torch.randn(1, 3, 4)

        with torch.no_grad():
            head, normed = ir.CaptionProjectorGraph(projector, ScalingNorm(2.0))(hidden)
            plain = ir.ProjectorGraph(projector)(hidden)

        assert torch.equal(head, plain)
        assert torch.equal(normed, head * 2.0)


class TestWrapperEquivalence:
    def test_a_tuple_output_is_compared_position_by_position(self):
        wrapper = ir.CaptionProjectorGraph(TinyResidualProjector(), ScalingNorm(2.0))
        hidden = torch.zeros(1, 3, 4)

        diff = ir._check_wrapper_equivalence(
            wrapper, (hidden,), (torch.zeros(1, 3, 4), torch.zeros(1, 3, 4)), "test", 0.0
        )

        assert diff == 0.0

    def test_a_missing_output_fails_loudly(self):
        """MUST: do not silently truncate with `zip` when the counts differ (an unchecked output
        would remain)."""
        wrapper = ir.CaptionProjectorGraph(TinyResidualProjector(), ScalingNorm(2.0))

        with pytest.raises(AssertionError, match="the wrapper has"):
            ir._check_wrapper_equivalence(
                wrapper, (torch.zeros(1, 3, 4),), (torch.zeros(1, 3, 4),), "test", 0.0
            )


class TestNormDivergence:
    """MUST: the only check guarding the contract of `caption-proj`'s second output (the sequence
    with `caption_norm` applied)."""

    def test_identical_norm_weights_fail_loudly(self):
        shared = ScalingNorm(1.5)

        with pytest.raises(AssertionError, match="same weights are suspected to be read twice"):
            ir._norm_divergence(shared, ScalingNorm(1.5))

    def test_diverging_norm_weights_are_reported(self):
        assert ir._norm_divergence(ScalingNorm(1.0), ScalingNorm(1.5)) == pytest.approx(0.5)


class TestOutputTupleHelpers:
    def test_single_wraps_and_first_unwraps(self):
        value = torch.ones(1, 2)

        wrapped = ir._single({"a": value})

        assert wrapped == {"a": (value,)}
        assert ir._first(wrapped) == {"a": value}

    def test_first_takes_only_the_leading_output(self):
        head, tail = torch.zeros(1, 2), torch.ones(1, 2)

        assert ir._first({"a": (head, tail)}) == {"a": head}


class TinyProjector(nn.Module):
    """A minimal graph to run `_write_io` (one input named `hidden`, one output)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden)


class TestWriteIo:
    @pytest.fixture
    def exported(self, tmp_path):
        torch.manual_seed(0)
        module = TinyProjector()
        graph = export_to_file(
            module,
            (torch.randn(1, 3, 4),),
            tmp_path / ir.MODEL_FILE,
            provenance=ir.PROVENANCE,
            graph_name="tiny",
        )
        return module, graph, tmp_path

    def test_writes_one_file_per_case(self, exported):
        module, graph, out_dir = exported
        hidden = torch.randn(1, 3, 4)
        with torch.no_grad():
            expected = {"a": (module(hidden),)}

        written = ir._write_io(graph, {"a": {"hidden": hidden}}, expected, out_dir)

        assert written == [f"{ir.IO_PREFIX}a{ir.IO_SUFFIX}"]

    def test_writes_every_output_position(self, exported):
        """The two outputs of `caption-proj` both appear as `output.0` / `output.1`."""
        module, graph, out_dir = exported
        hidden = torch.randn(1, 3, 4)
        with torch.no_grad():
            head = module(hidden)
        two = IrGraph(
            inputs=graph.inputs,
            outputs=[*graph.outputs, "second"],
            nodes=graph.nodes,
            values={**graph.values, "second": IrValue(dtype="f32", shape=[1, 3, 2])},
            initializers=graph.initializers,
            symbols=graph.symbols,
        )

        ir._write_io(two, {"a": {"hidden": hidden}}, {"a": (head, head * 2.0)}, out_dir)

        written = load_file(str(out_dir / f"{ir.IO_PREFIX}a{ir.IO_SUFFIX}"))
        assert sorted(written) == ["input.hidden", "output.0", "output.1"]
        assert torch.equal(written["output.1"], head * 2.0)

    def test_an_input_name_mismatch_fails_loudly(self, exported):
        """MUST: an io with shifted names passes silently as "readable, but fed into another
        input"."""
        module, graph, out_dir = exported
        hidden = torch.randn(1, 3, 4)
        with torch.no_grad():
            expected = {"a": (module(hidden),)}

        with pytest.raises(AssertionError, match="the input names"):
            ir._write_io(graph, {"a": {"state": hidden}}, expected, out_dir)

    def test_an_output_count_mismatch_fails_loudly(self, exported):
        """MUST: an io with a different count passes silently as "one output goes unchecked"."""
        module, graph, out_dir = exported
        hidden = torch.randn(1, 3, 4)
        with torch.no_grad():
            expected = {"a": (module(hidden),)}
        two = IrGraph(
            inputs=graph.inputs,
            outputs=[*graph.outputs, "second"],
            nodes=graph.nodes,
            values={**graph.values, "second": IrValue(dtype="f32", shape=[1])},
            initializers=graph.initializers,
            symbols=graph.symbols,
        )

        with pytest.raises(AssertionError, match="expected outputs"):
            ir._write_io(two, {"a": {"hidden": hidden}}, expected, out_dir)


class TestTargetCli:
    def test_all_targets_by_default(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(ir, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ir.main([])

        assert seen["targets"] == ir.TARGETS

    def test_a_single_target_is_forwarded(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(ir, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ir.main(["--target", ir.TARGET_BACKBONE])

        assert seen["targets"] == (ir.TARGET_BACKBONE,)

    def test_the_default_out_root_is_derived_from_the_weight_directory(self, tmp_path):
        assert ir.default_out_root(tmp_path / "v4-small").name == "irodori-v4-small"

    @pytest.mark.parametrize("dtype", ["f16", "i8"])
    def test_the_dtype_is_forwarded(self, monkeypatch, dtype):
        seen: dict[str, object] = {}
        monkeypatch.setattr(ir, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ir.main(["--dtype", dtype])

        assert seen["dtype"] == dtype

    @pytest.mark.parametrize("steps", ["0", "41"])
    def test_a_calib_step_count_outside_the_reference_loop_is_refused_at_the_entrance(
        self, monkeypatch, capsys, steps
    ):
        """MUST: the limit is checked **at the entrance** (an excess would fail only after a full
        run, with a diagnosis about something else).

        The cutoff sentinel fires only on `len(batches) >= steps`, so a value beyond the full
        reference loop never fires, and after running a whole real-weight case's denoise it fails
        with "upstream's spelling disagrees with what the script expects" — a diagnosis with the
        wrong cause, and the most expensive one.
        """
        monkeypatch.setattr(ir, "export_series", lambda *_a, **_kw: {"dir": "x"})

        with pytest.raises(SystemExit):
            ir.main(["--dtype", "i4", "--calib-steps", steps])

        assert f"--calib-steps must be 1 to {ip.NUM_STEPS}" in capsys.readouterr().err

    def test_a_calib_step_count_inside_the_reference_loop_is_forwarded(self, monkeypatch):
        """The window for a reduced smoke run stays open (for what goes into the distribution,
        the assembly's budget check refuses it)."""
        seen: dict[str, object] = {}
        monkeypatch.setattr(ir, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ir.main(["--dtype", "i4", "--calib-steps", str(ip.NUM_STEPS)])

        assert seen["calib_steps"] == ip.NUM_STEPS


def _is_f16_exact(tensor: torch.Tensor) -> bool:
    """Whether it is on the f16 grid (the same predicate as emit's eligibility check)."""
    return bool(torch.equal(tensor, tensor.to(torch.float16).to(torch.float32)))


class TestWeightDtypeSeries:
    """Series per storage dtype (ADR 0018 / 0019 / 0027 / 0050) — only the part whose breakage
    gives a **false pass**.

    No real weights (the wiring discipline does not depend on the model). The numbers themselves
    are checked by the E2E on the Deno side and by emit's eligibility check.
    """

    def test_each_dtype_gets_its_own_series_root(self, tmp_path):
        """MUST: a compressed series gets its own directory (sharing one would put the compressed
        assets under the f32 checks)."""
        roots = {
            dtype: ir.default_out_root(tmp_path / "v4-small", dtype) for dtype in ir.WEIGHT_DTYPES
        }

        assert set(roots) == {"f32", "f16", "i8", "i4"}
        assert len(set(roots.values())) == len(roots)
        assert roots["f32"].name == "irodori-v4-small"
        assert roots["f16"].name == "irodori-v4-small-f16"
        assert roots["i8"].name == "irodori-v4-small-i8"
        assert roots["i4"].name == "irodori-v4-small-i4"

    def test_the_series_name_carries_the_weights_directory_name(self, tmp_path):
        other = ir.default_out_root(tmp_path / "v9-large", "f16")

        assert other.name == "irodori-v9-large-f16"

    def test_f32_leaves_the_weights_untouched(self):
        # MUST: restore the RNG after seeding (`fork_rng`) — leaving the global RNG changed makes
        # later test files build their modules from **different random values**, and an addition
        # here would move a check far away (the truncation bit identity of
        # `irodori.dacvae.export`).
        with torch.random.fork_rng():
            torch.manual_seed(0)
            module = TinyResidualProjector()
            before = module.projector.weight.clone()

            quantized = ir.fake_quant("f32", {"text-proj": module})

            assert quantized.reports == {}
            assert quantized.scales == {}
            assert torch.equal(module.projector.weight, before)

    def test_f16_rounds_every_module_it_is_handed(self):
        """MUST: missing even one would put values computed with the original weights into the
        golden for that one."""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            modules = {name: TinyResidualProjector() for name in "abc"}
            assert not any(_is_f16_exact(module.projector.weight) for module in modules.values()), (
                "already on the grid before rounding, so the test would detect nothing"
            )

            quantized = ir.fake_quant("f16", modules)

            assert sorted(quantized.reports) == ["a", "b", "c"]
            assert quantized.scales == {}
            for module in modules.values():
                assert all(_is_f16_exact(tensor) for tensor in module.parameters())

    def test_it_refuses_an_empty_set_of_modules(self):
        """Do not stay silent when `--dtype f16` is given but there is nothing to round."""
        with pytest.raises(SystemExit, match="there is no module to round"):
            ir.fake_quant("f16", {})

    def test_f16_refuses_a_set_where_nothing_was_actually_rounded(self):
        """A different hole from "zero modules" at the entrance — **the modules exist but zero
        weights were rounded**.

        The "a total of 0 fails" the docstring claims was a check for i8 only, so f16 could pass
        without rounding anything (unreachable today, since the current 9 modules always have f32
        parameters, but the check was not in place).
        """
        with pytest.raises(SystemExit, match="not a single weight was rounded"):
            ir.fake_quant("f16", {"text_norm": nn.Identity()})

    def test_i8_quantizes_only_the_per_channel_types(self):
        """MUST: i8 touches only the `weight` of the types in `QUANT_CHANNEL_AXES` (ADR 0019).

        Rounding the norms too would disagree with emit's eligibility check (only weight slots
        are compressed), and only the golden would be computed with different weights.
        """
        with torch.random.fork_rng():
            torch.manual_seed(0)
            module = TinyResidualProjector()
            norm_before = module.residual_norm.weight.clone()

            quantized = ir.fake_quant("i8", {"text-proj": module})

            assert sorted(quantized.scales["text-proj"]) == [
                "projector.weight",
                "residual_down.weight",
                "residual_up.weight",
            ]
            assert torch.equal(module.residual_norm.weight, norm_before)
            for key, scale in quantized.scales["text-proj"].items():
                weight = module.get_parameter(key)
                restored = quantize_to_int8(weight, scale).to(torch.float32) * scale
                assert torch.equal(restored, weight), "not on the i8 grid"

    def test_i8_lets_a_norm_only_role_through(self):
        """MUST: a norm-only role passes through under i8 (unlike `f16`, "zero" is the correct
        state)."""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            modules = {"text-proj": TinyResidualProjector(), "text_norm": ScalingNorm(1.5)}

            quantized = ir.fake_quant("i8", modules)

            assert sorted(quantized.reports) == ["text-proj", "text_norm"]
            assert sorted(quantized.scales) == ["text-proj"]

    def test_i8_refuses_a_set_with_nothing_quantizable(self):
        """Do not stay silent when `--dtype i8` is given but there are zero per-channel targets
        overall."""
        with pytest.raises(SystemExit, match="no module could be quantized per channel"):
            ir.fake_quant("i8", {"text_norm": ScalingNorm(1.5)})

    def test_every_loaded_module_reaches_the_fake_quant(self):
        """What gets passed is everything `export_series` builds with `load_*` (a gap would make
        the golden disagree).

        Captures **the actual arguments of the call** instead of listing names — a copied table
        would stay green and stale when `export_series` adds one more.
        """
        source = inspect.getsource(ir.export_series)
        handed = source[source.index("quantized = fake_quant(") :]
        handed = handed[: handed.index("\n    )")]
        for name in ("backbone", "projectors[", "speaker_encoder", "duration", "dit"):
            assert name in handed
        for norm in ("speaker_norm", "text_norm", "caption_norm"):
            assert f'"{norm}": {norm}' in handed


class TinyRopeBody(nn.Module):
    """A stand-in for the backbone body with a buffer `assert_rope_lifted` can demote."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Linear(4, 4, bias=False)
        self.register_buffer("inv_freq", torch.ones(2))


class TinyBackboneHolder(nn.Module):
    """The shape `BackboneGraph` reads (the body is inside the `backbone` attribute)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = TinyRopeBody()


class TinyRopeEncoder(nn.Module):
    """An encoder stand-in with the attribute `SpeakerGraph` reads in `__init__` (`head_dim`)."""

    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 4
        self.in_proj = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])


class TinyJointAttention(nn.Module):
    """A stand-in that copies only **the set of children** of `JointAttention` (no computation —
    it only serves to check the rebasing).

    The order matches the real one too (3 on the self side → 6 on the condition side → gate / wo
    → the q/k norms).
    """

    def __init__(self) -> None:
        super().__init__()
        for name in ("wq", "wk", "wv", *ir.CONTEXT_KV_PROJECTIONS, "gate", "wo"):
            setattr(self, name, nn.Linear(4, 4, bias=False))
        self.q_norm = ScalingNorm(1.0)
        self.k_norm = ScalingNorm(1.0)


class TinyDitBlock(nn.Module):
    """The children of `DiffusionBlock` (attention + mlp + two adaLNs)."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = TinyJointAttention()
        self.mlp = nn.Linear(4, 4, bias=False)
        self.attention_adaln = nn.Linear(4, 4, bias=False)
        self.mlp_adaln = nn.Linear(4, 4, bias=False)


class TinyDit(nn.Module):
    """The subtrees the two DiT wrappers hold + branches they **do not** hold (like the inner copy
    `load_dit` builds whole).

    The block has the real set of children (the DiT is split into `dit-context` / `dit` — ADR
    0114 — so checking with the table how one block's weights divide between the two wrappers
    needs the block's contents).
    """

    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 4
        self.cond_module = nn.Linear(4, 4, bias=False)
        self.in_proj = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([TinyDitBlock()])
        self.out_norm = ScalingNorm(1.0)
        self.out_proj = nn.Linear(4, 4, bias=False)
        self.text_norm = ScalingNorm(1.0)
        self.caption_norm = ScalingNorm(1.0)
        self.pretrained_text_backbone = nn.Linear(4, 4, bias=False)


class TestTargetScales:
    """The table that rebases the i8 scale ledger onto **the FQNs inside the wrapper**
    (`TARGET_SCALE_SOURCES`).

    MUST: if this breaks, emit fails with "eligible but no scale" (it does not pass with broken
    values). That failure is far away, so the mapping between the table and the wrappers is
    pinned here.
    """

    def test_the_table_covers_every_target(self):
        assert sorted(ir.TARGET_SCALE_SOURCES) == sorted(ir.TARGETS)

    def _rebased(self, target: str, wrapper: nn.Module, module: nn.Module) -> list[str]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            scales = ir.fake_quant("i8", {target: module}).scales
        return sorted(ir.target_scales(target, wrapper, scales))

    def test_every_wrapper_receives_a_scale_for_each_of_its_weights(self):
        """MUST: **every** per-channel target weight a wrapper holds finds its rebase target.

        `pytest.importorskip` is there because `SpeakerGraph` / `DitGraph` pull in the upstream
        implementation to build the real-pair RoPE table (the table itself is not used for the
        rebasing, but `__init__` calls it).
        """
        pytest.importorskip("irodori_tts")
        cases = {
            ir.TARGET_BACKBONE: (TinyBackboneHolder(), ir.BackboneGraph),
            ir.TARGET_TEXT_PROJ: (TinyResidualProjector(), ir.ProjectorGraph),
        }
        for target, (module, wrap) in cases.items():
            wrapper = wrap(module)
            expected = sorted(
                f"{module_name}.weight"
                for module_name, child in wrapper.named_modules()
                if isinstance(child, nn.Linear)
            )

            assert self._rebased(target, wrapper, module) == expected

    def test_the_caption_projector_is_rebased_two_levels_deep(self):
        """`CaptionProjectorGraph` contains `ProjectorGraph` (a two-level prefix)."""
        module = TinyResidualProjector()
        wrapper = ir.CaptionProjectorGraph(module, ScalingNorm(2.0))

        assert self._rebased(ir.TARGET_CAPTION_PROJ, wrapper, module) == [
            "projection.projector.projector.weight",
            "projection.projector.residual_down.weight",
            "projection.projector.residual_up.weight",
        ]

    def test_the_duration_predictor_is_rebased(self):
        module = TinyResidualProjector()
        wrapper = ir.DurationGraph(module, ScalingNorm(1.0))

        assert self._rebased(ir.TARGET_DURATION, wrapper, module) == [
            "predictor.projector.weight",
            "predictor.residual_down.weight",
            "predictor.residual_up.weight",
        ]

    def test_the_speaker_encoder_is_rebased(self):
        pytest.importorskip("irodori_tts")
        module = TinyRopeEncoder()
        wrapper = ir.SpeakerGraph(module, ScalingNorm(1.0), 8)

        assert self._rebased(ir.TARGET_SPEAKER, wrapper, module) == [
            "encoder.blocks.0.weight",
            "encoder.in_proj.weight",
        ]

    def test_weights_the_dit_wrapper_does_not_hold_are_dropped(self):
        """`load_dit` builds the **whole** DiT, so the ledger also holds scales of unused
        branches.

        The DiT is split in two (ADR 0114), so `dit` receives the block linears except the
        condition-side projections plus what is outside the blocks — the 6 condition-side ones go
        to `dit-context` (the expectation moved exactly as far as the graph changed).
        """
        pytest.importorskip("irodori_tts")
        module = TinyDit()
        wrapper = ir.DitGraph(module, 8)

        rebased = self._rebased(ir.TARGET_DIT, wrapper, module)

        assert rebased == [
            "blocks.0.attention.gate.weight",
            "blocks.0.attention.wk.weight",
            "blocks.0.attention.wo.weight",
            "blocks.0.attention.wq.weight",
            "blocks.0.attention.wv.weight",
            "blocks.0.attention_adaln.weight",
            "blocks.0.mlp.weight",
            "blocks.0.mlp_adaln.weight",
            "cond_module.weight",
            "in_proj.weight",
            "out_proj.weight",
        ]
        assert "pretrained_text_backbone.weight" not in rebased

    def test_the_context_wrapper_receives_only_the_conditioning_projections(self):
        """`dit-context` picks only the 6 condition-side projections × blocks from the same ledger
        of the `dit` role.

        As in `export_series`, the rounding is applied once to the `dit` role (the whole
        `TextToLatentRFDiT`) — there is no ledger of a `dit-context` role.
        """
        module = TinyDit()
        wrapper = ir.DitContextGraph(module)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            scales = ir.fake_quant("i8", {ir.TARGET_DIT: module}).scales

        rebased = sorted(ir.target_scales(ir.TARGET_DIT_CONTEXT, wrapper, scales))

        assert rebased == sorted(
            f"blocks.0.attention.{name}.weight" for name in ir.CONTEXT_KV_PROJECTIONS
        )

    def test_a_stale_prefix_fails_loudly(self, monkeypatch):
        """MUST: a rebasing that matches nothing does not silently return empty (it would go
        unnoticed until emit)."""
        module = TinyResidualProjector()
        wrapper = ir.ProjectorGraph(module)
        monkeypatch.setattr(
            ir,
            "TARGET_SCALE_SOURCES",
            {ir.TARGET_TEXT_PROJ: ((ir.TARGET_TEXT_PROJ, "", "stale."),)},
        )

        with pytest.raises(SystemExit, match="could be remapped to an FQN in the wrapper"):
            self._rebased(ir.TARGET_TEXT_PROJ, wrapper, module)

    def test_f32_and_f16_hand_no_scales_to_emit(self):
        """f16 has no scales (an empty ledger rebases to empty — emit does not look them up for
        f16 either)."""
        wrapper = ir.ProjectorGraph(TinyResidualProjector())

        assert ir.target_scales(ir.TARGET_TEXT_PROJ, wrapper, {}) == {}


#: The declared condition lengths of the small real DiT (what `_dit_cases` right-pads to) and the
#: speaker's limit after patching.
TINY_MODEL_CONFIG = {"max_text_len": 6, "max_caption_len": 7}
TINY_SPEAKER_MAX = 4
#: The total condition-side length of the `dit` mask (text 6 + speaker 4+1 + caption 7).
TINY_CONTEXT_TOTAL = 6 + (TINY_SPEAKER_MAX + 1) + 7


def _tiny_real_dit() -> nn.Module:
    """Upstream's `TextToLatentRFDiT` itself (a small config, condition encoders from scratch).

    head_dim is 64 as in the real weights (the bit identity of real-pair RoPE depends on the
    shape — the NOTE of `irodori.patch`; 0 at 64), heads is 2 (`_apply_rotary_half` splits the
    heads axis in half). Every weight is overwritten with deterministic random values — upstream
    initializes `out_proj` to 0, so as is the output would be identically 0 and the comparison
    tautological.
    """
    irodori_model = pytest.importorskip("irodori_tts.model")
    from irodori_tts.config import ModelConfig

    config = ModelConfig(
        latent_dim=4,
        model_dim=128,
        num_layers=2,
        num_heads=2,
        mlp_ratio=1.0,
        text_vocab_size=16,
        text_dim=16,
        text_layers=1,
        text_heads=1,
        text_mlp_ratio=1.0,
        use_caption_condition=True,
        use_speaker_condition=True,
        caption_vocab_size=16,
        caption_dim=16,
        caption_layers=1,
        caption_heads=1,
        caption_mlp_ratio=1.0,
        speaker_dim=16,
        speaker_layers=1,
        speaker_heads=1,
        speaker_mlp_ratio=1.0,
        timestep_embed_dim=16,
        adaln_rank=4,
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = irodori_model.TextToLatentRFDiT(config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0.0, 0.2)
    return model


class _TinySplit:
    """The result of running the DiT stage of `export_series` (cases → pre-patch reference →
    assembling the inputs) with the same functions on a small real DiT."""

    def __init__(self, model: nn.Module) -> None:
        from irodori_tts.model import TextToLatentRFDiT, get_timestep_embedding

        source = SimpleNamespace(
            prepend_masked_mean_token=TextToLatentRFDiT._prepend_masked_mean_token,
            timestep_embedding=get_timestep_embedding,
        )
        generator = torch.Generator().manual_seed(5)
        self.sym_max = ir.dit_sym_max(model.cfg)
        self.inputs, reference = ir._dit_cases(
            source,
            model.cfg,
            TINY_MODEL_CONFIG,
            model.text_norm,
            model.caption_norm,
            TINY_SPEAKER_MAX,
            self.sym_max,
            {ir.DIT_TEXT_SOURCE: torch.randn(1, 5, 16, generator=generator)},
            {ir.DIT_CAPTION_SOURCE: torch.randn(1, 3, 16, generator=generator)},
            {ir.DIT_SPEAKER_SOURCE: torch.randn(1, 3, 16, generator=generator)},
        )
        self.pristine = ir._pristine_dit_outputs(model, reference)
        self.context_inputs, self.owner = ir._dit_context_cases(self.inputs)
        self.context = ir._pristine_dit_context_outputs(model, self.context_inputs)
        self.dit_inputs = ir._dit_graph_inputs(
            self.inputs,
            self.owner,
            self.context,
            irodori_context_kv_names(len(model.blocks)),
        )


@pytest.fixture
def tiny_split(restore_forward) -> tuple[nn.Module, _TinySplit]:
    """Takes the reference before patching and returns after applying the patches (the ①→②
    order of `export_series`)."""
    model = _tiny_real_dit()
    split = _TinySplit(model)
    patch_irodori.apply_patches()
    return model, split


class TestDitSplit:
    """Splits the DiT into `dit-context` (condition-side K/V, once per generation) and `dit`
    (ADR 0114).

    MUST: the split does not move the numbers by a single bit — the sha checks of the reference
    layer (WAV / latent) are expected to stay unchanged.
    """

    def test_the_composed_pair_reproduces_the_upstream_forward_bit_for_bit(self, tiny_split):
        """The split's equivalence check itself (every case, including the three uncond variants
        and both ends of S, 2 / 750; atol 0)."""
        model, split = tiny_split
        wrappers = ir.dit_wrappers(model, split.sym_max)
        composed = ir.ComposedDitGraph(wrappers.context, wrappers.dit)

        for name, args in split.inputs.items():
            diff = ir._check_wrapper_equivalence(
                composed, tuple(args.values()), (split.pristine[name],), name, 0.0
            )
            assert diff == 0.0, name

    def test_the_context_graph_matches_the_upstream_kv_cache(self, tiny_split):
        """The golden of `dit-context` is the value of upstream's `build_context_kv_cache` (not
        of the wrapper itself)."""
        model, split = tiny_split
        wrappers = ir.dit_wrappers(model, split.sym_max)

        for representative, states in split.context_inputs.items():
            assert (
                ir._check_wrapper_equivalence(
                    wrappers.context,
                    tuple(states.values()),
                    split.context[representative],
                    representative,
                    0.0,
                )
                == 0.0
            )

    def test_the_dit_graph_fed_with_the_context_golden_matches_upstream(self, tiny_split):
        """With the golden inputs of `dit` (K/V = the expectation of `dit-context`) it is
        bit-identical to upstream's output."""
        model, split = tiny_split
        wrappers = ir.dit_wrappers(model, split.sym_max)

        for name, args in split.dit_inputs.items():
            assert list(args) == list(wrappers.dit.input_names), (
                "the order of the golden inputs differs from the IR"
            )
            assert (
                ir._check_wrapper_equivalence(
                    wrappers.dit, tuple(args.values()), (split.pristine[name],), name, 0.0
                )
                == 0.0
            )

    def test_the_context_outputs_are_one_k_and_one_v_per_block_in_order(self, tiny_split):
        """There are 2 × the number of blocks outputs, with the K / V of block b (the first-level
        concatenation) in ascending b."""
        model, split = tiny_split
        wrappers = ir.dit_wrappers(model, split.sym_max)
        states = next(iter(split.context_inputs.values()))

        with torch.no_grad():
            outputs = wrappers.context(*states.values())
            text = model.text_norm(states["text_state"])
            caption = model.caption_norm(states["caption_state"])

        assert len(outputs) == 2 * len(model.blocks)
        assert wrappers.context.output_names == (
            "context_k_0",
            "context_v_0",
            "context_k_1",
            "context_v_1",
        )
        for index, block in enumerate(model.blocks):
            key, value = ir.DitContextGraph.block_context(
                block.attention, text, states["speaker_state"], caption
            )
            assert torch.equal(outputs[2 * index], key)
            assert torch.equal(outputs[2 * index + 1], value)
            assert tuple(key.shape) == (1, TINY_CONTEXT_TOTAL, 2, 64)
        # Power to detect a mix-up: the values differ per block (if they were equal, a swapped
        # order would pass unnoticed).
        assert not torch.equal(outputs[0], outputs[2])

    def test_a_swapped_block_order_breaks_the_composition(self, tiny_split):
        """Fault injection: passing the K/V of blocks 0 and 1 swapped breaks the bit identity with
        upstream."""
        model, split = tiny_split
        wrappers = ir.dit_wrappers(model, split.sym_max)
        args = split.dit_inputs["dit-cond-1s"]
        values = list(args.values())
        values[3:5], values[5:7] = values[5:7], values[3:5]

        with pytest.raises(AssertionError, match="eager equivalence broke"):
            ir._check_wrapper_equivalence(
                wrappers.dit, tuple(values), (split.pristine["dit-cond-1s"],), "swap", 0.0
            )

    def test_the_two_wrappers_split_the_old_ownership_and_share_only_k_norm(self):
        """The union of the ownership = every weight of the subtrees the unsplit `DitGraph` held;
        the only overlap is `k_norm`.

        Ownership = the container's initializers (the premise of the rebasing, the read-back and
        the i4 eligibility check), so if this breaks, the storage overrides fail on unknown keys or
        a weight lands in neither container.
        """
        model = _tiny_real_dit()
        wrappers = ir.dit_wrappers(model, ir.dit_sym_max(model.cfg))
        context = {name for name, _ in wrappers.context.named_parameters()}
        rest = {name for name, _ in wrappers.dit.named_parameters()}
        before = {
            f"{prefix}.{name}"
            for prefix in (
                "cond_module",
                "in_proj",
                "blocks",
                "out_norm",
                "out_proj",
                "text_norm",
                "caption_norm",
            )
            for name, _ in model.get_submodule(prefix).named_parameters()
        }

        assert context | rest == before
        assert context & rest == {
            f"blocks.{index}.attention.k_norm.weight" for index in range(len(model.blocks))
        }
        assert {name for name in context if ".attention.w" in name} == {
            f"blocks.{index}.attention.{projection}.weight"
            for index in range(len(model.blocks))
            for projection in ir.CONTEXT_KV_PROJECTIONS
        }
        assert not {name for name in rest if name.startswith(("text_norm", "caption_norm"))}

    def test_every_case_with_the_same_states_shares_one_context_golden(self, tiny_split):
        """The three uncond variants keep the cond states — one set of `dit-context` goldens is
        enough."""
        _model, split = tiny_split

        assert list(split.context_inputs) == [ir.DIT_CASES[0][0]]
        assert set(split.owner) == {name for name, *_rest in ir.DIT_CASES}
        assert set(split.owner.values()) == {ir.DIT_CASES[0][0]}

    def test_the_exported_boundary_carries_the_contract_names(self, tiny_split, tmp_path):
        """The IR's boundary names (the outputs of `dit-context` / the inputs of `dit` from the
        fourth on) follow the contract's spelling, and the assembly check
        (`irodori.distribution`) accepts them as they are.

        torch.export names variadic arguments `context_<i>` and outputs after FX nodes — if the
        renaming is lost, the runtime cannot bind the resident tensors by name.
        """
        model, split = tiny_split
        wrappers = ir.dit_wrappers(model, split.sym_max)
        paths = {
            ir.TARGET_DIT_CONTEXT: tmp_path / ir.TARGET_DIT_CONTEXT / ir.MODEL_FILE,
            ir.TARGET_DIT: tmp_path / ir.TARGET_DIT / ir.MODEL_FILE,
        }
        context_graph = publish_model(
            paths[ir.TARGET_DIT_CONTEXT],
            *ir.export_ir(
                wrappers.context,
                tuple(next(iter(split.context_inputs.values())).values()),
                axis=ir.TargetAxis(None, None, 0, {}, ir.PRESERVED_OP_PREFIXES),
                output_names=wrappers.context.output_names,
            ),
            provenance=ir.PROVENANCE,
            graph_name="dit_context",
        )
        dit_graph = publish_model(
            paths[ir.TARGET_DIT],
            *ir.export_ir(
                wrappers.dit,
                tuple(split.dit_inputs["dit-cond-1s"].values()),
                axis=ir.TargetAxis(
                    ir.DIT_TORCH_DIM,
                    ir.DIT_SYMBOL,
                    split.sym_max,
                    {0: (1, 0), 2: (3, TINY_CONTEXT_TOTAL)},
                    ir.PRESERVED_OP_PREFIXES,
                ),
                input_names=wrappers.dit.input_names,
            ),
            provenance=ir.PROVENANCE,
            graph_name="dit",
        )

        assert [spec.name for spec in context_graph.inputs] == [
            "text_state",
            "speaker_state",
            "caption_state",
        ]
        assert tuple(context_graph.outputs) == wrappers.context.output_names
        assert context_graph.symbols == []
        assert tuple(spec.name for spec in dit_graph.inputs) == wrappers.dit.input_names
        _assert_irodori_dit_boundary(
            {
                role: ir_graph(path)
                for role, path in (
                    ("dit_context", paths[ir.TARGET_DIT_CONTEXT]),
                    ("dit", paths[ir.TARGET_DIT]),
                )
            },
            {"dit_context": paths[ir.TARGET_DIT_CONTEXT], "dit": paths[ir.TARGET_DIT]},
            {"maxTextLen": 6, "speakerRows": TINY_SPEAKER_MAX + 1, "maxCaptionLen": 7},
        )
        # A golden io can be written only when it matches the IR's input names (the check of
        # `_write_io`).
        written = ir._write_io(
            dit_graph,
            {"dit-cond-1s": split.dit_inputs["dit-cond-1s"]},
            {"dit-cond-1s": (split.pristine["dit-cond-1s"],)},
            tmp_path,
        )
        assert written == [f"{ir.IO_PREFIX}dit-cond-1s{ir.IO_SUFFIX}"]


class TestPristineDitContextOrdering:
    def test_taking_the_context_reference_after_patching_fails_loudly(self, monkeypatch):
        """MUST: taken after patching, the value comes from the same path as the wrapper and the
        equivalence check becomes tautological."""
        monkeypatch.setattr(patch_irodori, "_APPLIED", True)

        with pytest.raises(
            AssertionError, match="Tried to take a reference after the patches were applied"
        ):
            ir._pristine_dit_context_outputs(TinyDit(), {})


class TestDitContextCases:
    """One `dit-context` golden per set of condition states (sets are compared by value)."""

    @staticmethod
    def _states(seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return {
            name: torch.randn(1, 3, 2, generator=generator)
            for name in ("text_state", "speaker_state", "caption_state")
        }

    def test_cases_with_equal_states_share_the_first_case_as_representative(self):
        shared = self._states(0)
        cases = {
            "a": {"x_t": torch.zeros(1), **shared},
            "b": {"x_t": torch.ones(1), **{key: value.clone() for key, value in shared.items()}},
        }

        groups, owner = ir._dit_context_cases(cases)

        assert list(groups) == ["a"]
        assert owner == {"a": "a", "b": "a"}
        assert list(groups["a"]) == ["text_state", "speaker_state", "caption_state"]

    def test_a_case_with_another_state_gets_its_own_golden(self):
        shared = self._states(0)
        other = {**shared, "speaker_state": shared["speaker_state"] + 1.0}
        cases = {"a": {"x_t": torch.zeros(1), **shared}, "b": {"x_t": torch.zeros(1), **other}}

        groups, owner = ir._dit_context_cases(cases)

        assert list(groups) == ["a", "b"]
        assert owner == {"a": "a", "b": "b"}


def _boundary_graph() -> IrGraph:
    """A minimal graph with two inputs and two outputs (node outputs) (the subject of the
    boundary renaming)."""
    return IrGraph(
        inputs=[
            IrInput(name="x", dtype="f32", shape=[1, 2]),
            IrInput(name="context_0", dtype="f32", shape=[1, 2]),
        ],
        outputs=["add", "add_1"],
        values={
            "add": IrValue(dtype="f32", shape=[1, 2]),
            "add_1": IrValue(dtype="f32", shape=[1, 2]),
        },
        nodes=[
            IrNode(op="add", ins=["x", "context_0"], outs=["add"], attrs={}),
            IrNode(op="add", ins=["add", "context_0"], outs=["add_1"], attrs={}),
        ],
    )


class TestNameBoundary:
    def test_inputs_and_outputs_are_renamed_by_position_everywhere(self):
        renamed = ir.name_boundary(
            _boundary_graph(), inputs=("x", "context_k_0"), outputs=("first", "second")
        )

        assert [spec.name for spec in renamed.inputs] == ["x", "context_k_0"]
        assert renamed.outputs == ["first", "second"]
        assert set(renamed.values) == {"first", "second"}
        assert [node.ins for node in renamed.nodes] == [
            ["x", "context_k_0"],
            ["first", "context_k_0"],
        ]
        assert [node.outs for node in renamed.nodes] == [["first"], ["second"]]

    def test_the_original_graph_is_left_untouched(self):
        graph = _boundary_graph()

        ir.name_boundary(graph, outputs=("first", "second"))

        assert graph.outputs == ["add", "add_1"]

    def test_a_count_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="with 1 names"):
            ir.name_boundary(_boundary_graph(), outputs=("only",))

    def test_a_name_that_collides_with_a_remaining_value_fails_loudly(self):
        """A collision would silently drop one declaration."""
        with pytest.raises(AssertionError, match="collide"):
            ir.name_boundary(_boundary_graph(), outputs=("add", "x"))

    def test_an_output_that_is_not_produced_by_a_node_fails_loudly(self):
        """Renaming an output that returns an input as is would move the input's name too."""
        graph = _boundary_graph()
        graph.outputs = ["x", "add_1"]

        with pytest.raises(AssertionError, match="is not a node output"):
            ir.name_boundary(graph, outputs=("first", "second"))


class TestProvenance:
    def test_the_container_points_at_the_notice(self) -> None:
        """As in the other families that ship a NOTICE, the location of the modification notice
        can be traced from a single krm."""
        from karume.dist import NOTICE_FILENAME

        assert ir.PROVENANCE.notice == NOTICE_FILENAME
