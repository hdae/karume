"""Script-level contracts of `irodori/pipeline_ref.py` (the part that needs no real weights).

Emitting from the real weights is manual (the same discipline as `irodori/export.py`). Pinned
here are the details of **the two stages the host copies** (deciding S and assembling Euler),
whose breakage would silently change the golden:

- The t schedule follows upstream's formula (`(1 - linspace) * 0.999`) and decreases
  monotonically
- Banker's rounding has not turned into "round half up" (the handling of 0.5)
- The clamp of S takes effect at the floor 13 / ceiling 750
- How the segment mask is split (the order and offsets of self / text / speaker / caption), and
  that an uncond variant drops **only its own segment**
- The CFG scale table is one-to-one with the spellings of the uncond variants
- The token preprocessing differs by kind (text gets `normalize_text` + strip, the caption strip
  only), and the caption passed to the upstream comparison is built **from upstream's entry
  point**
- Every check of the i4 preset's (`--dtype i4`) **read-back from the shipped bytes**
  (provenance / shape / count / whether the preset takes effect) fires — if one passes
  unnoticed, the accident of "calling an i8 golden an i8+dit4 golden" goes through with
  matching numbers and shapes. The i4 preset is **a mix of I4 + I8 + F32** (i4 inside the
  blocks except adaLN, i8 for adaLN and outside the blocks — listening ruling 2026-08-23), so the
  inverse of i8 and "the takes-effect check counts i4 only" are pinned here as well. The DiT is
  split into the two containers `dit` / `dit-context` (ADR 0114), so the consistency between
  the two (matching calibration records, identical shared tensors) is pinned here too
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch
from torch import nn

from irodori import export as ex
from irodori import pipeline_ref as ip
from irodori.distribution import (
    CALIB_PROVENANCE_FILE,
    IRODORI_FLOW_MEANFLOW,
    IRODORI_FLOW_RF,
    irodori_calib_floor,
)
from karume.container import BLOCK_MAX_BYTES, container_parts
from karume.emit import stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import publish_container
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)
from karume.verify import verify_container


class TestTSchedule:
    def test_it_starts_at_the_init_scale_and_ends_at_zero(self):
        schedule = ip.t_schedule(40)

        assert schedule.shape == (41,)
        assert float(schedule[0]) == pytest.approx(ip.INIT_SCALE)
        assert float(schedule[-1]) == 0.0

    def test_it_is_strictly_decreasing(self):
        """Upstream's `sample_euler_rf_cfg` requires `t_schedule[:-1] > t_schedule[1:]`."""
        schedule = ip.t_schedule(40)

        assert bool(torch.all(schedule[:-1] > schedule[1:]))

    def test_the_closed_form_is_within_one_ulp(self):
        """The TS side is expected to use the closed form — pins the difference at about 1
        ulp."""
        assert ip.closed_form_matches(ip.t_schedule(2), 2) == 0.0
        assert ip.closed_form_matches(ip.t_schedule(40), 40) < 1e-7

    def test_the_cfg_window_covers_the_first_half_of_the_steps(self):
        """CFG runs only on the steps with `t ∈ [0.5, 1.0]` — the first 20 of 40 steps."""
        schedule = ip.t_schedule(40)
        inside = [
            index for index in range(40) if ip.CFG_MIN_T <= float(schedule[index]) <= ip.CFG_MAX_T
        ]

        assert inside == list(range(20))


class TestBankerRounding:
    """MUST: do not let it turn into "round half up" (TS's `Math.round` always goes up at
    0.5)."""

    @pytest.mark.parametrize(
        ("frames", "want"),
        [(12.4, 12), (12.5, 12), (12.6, 13), (13.5, 14), (14.5, 14), (160.9389, 161)],
    )
    def test_half_way_values_round_to_even(self, frames, want):
        assert ip.banker_round_frames(frames) == want


class TestSequenceLength:
    def test_a_typical_prediction_is_used_as_is(self):
        # Feed the inverse of expm1 (log1p) to get the intended frames.
        log_frames = torch.log1p(torch.tensor([160.9389]))

        steps, meta = ip.sequence_length(log_frames, ex.CODEC_FRAME_RATE)

        assert steps == 161
        assert meta["minFrames"] == math.ceil(ip.MIN_SECONDS * ex.CODEC_FRAME_RATE)
        assert meta["maxFrames"] == math.floor(ip.MAX_SECONDS * ex.CODEC_FRAME_RATE)

    def test_a_short_prediction_is_clamped_up(self):
        steps, meta = ip.sequence_length(torch.log1p(torch.tensor([1.0])), ex.CODEC_FRAME_RATE)

        assert steps == meta["minFrames"] == 13

    def test_a_long_prediction_is_clamped_down(self):
        steps, meta = ip.sequence_length(torch.log1p(torch.tensor([5000.0])), ex.CODEC_FRAME_RATE)

        assert steps == meta["maxFrames"] == 750

    def test_the_mean_is_taken_over_the_batch(self):
        """Upstream uses `expm1(...).float().mean()` (the mean when there is more than one
        candidate)."""
        log_frames = torch.log1p(torch.tensor([100.0, 200.0]))

        steps, _meta = ip.sequence_length(log_frames, ex.CODEC_FRAME_RATE)

        assert steps == 150


CAPS = {"text": 4, "speaker": 5, "caption": 6}
USED = {"text": 2, "speaker": 3, "caption": 4}


class TestSegmentMasks:
    """MUST: pin how the segments are split (order and offsets) — a shift would still match the
    shape."""

    def test_the_cond_mask_marks_every_used_prefix(self):
        mask = ip._segment_masks(3, USED, CAPS, None)

        assert tuple(mask.shape) == (1, 1, 1, 3 + 4 + 5 + 6)
        flat = mask[0, 0, 0].tolist()
        assert flat[:3] == [True] * 3
        assert flat[3:7] == [True, True, False, False]
        assert flat[7:12] == [True, True, True, False, False]
        assert flat[12:] == [True, True, True, True, False, False]

    @pytest.mark.parametrize("uncond", ["text", "speaker", "caption"])
    def test_an_uncond_variant_clears_only_its_own_segment(self, uncond):
        cond = ip._segment_masks(3, USED, CAPS, None)[0, 0, 0]
        got = ip._segment_masks(3, USED, CAPS, uncond)[0, 0, 0]

        offsets = {"text": 3, "speaker": 7, "caption": 12}
        start = offsets[uncond]
        end = start + CAPS[uncond]
        assert not bool(got[start:end].any())
        assert torch.equal(got[:start], cond[:start])
        assert torch.equal(got[end:], cond[end:])

    def test_the_variant_names_match_the_export_script(self):
        """If the spellings diverge, an uncond segment silently drops a different condition."""
        assert tuple(ip.CFG_SCALES) == ex.DIT_UNCOND_VARIANTS


class TestRightPadIds:
    def test_it_pads_to_the_declared_length_and_marks_the_head(self):
        ids = torch.tensor([[1, 5, 6]], dtype=torch.int64)

        padded, mask = ip._right_pad_ids(ids, 6, 3)

        assert padded[0].tolist() == [1, 5, 6, 3, 3, 3]
        assert mask[0].tolist() == [True, True, True, False, False, False]

    def test_a_missing_sequence_becomes_all_pad_with_an_empty_mask(self):
        """The empty-caption path (upstream's `caption_mask.zero_()`)."""
        padded, mask = ip._right_pad_ids(None, 4, 3)

        assert padded[0].tolist() == [3, 3, 3, 3]
        assert not bool(mask.any())


class _RecordingTokenizer:
    """A minimal stand-in for `tokenizers.Tokenizer` (**records the strings it is given**).

    The ids are the code points themselves — all this needs to see is what was passed, so it has
    no vocabulary.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> Any:
        assert add_special_tokens is False, "upstream calls it without special tokens"
        self.seen.append(text)
        return SimpleNamespace(ids=[ord(char) for char in text])


class TestPackedIds:
    """MUST: the preprocessing differs by kind (text = `normalize_text` + strip / caption =
    strip only)."""

    #: A caption sensitive to normalization (outer brackets, a symbol — both removed by
    #: `normalize_text`).
    CAPTION = " 「①明るい声」 "

    @staticmethod
    def _bracket_stripper(body: str) -> str:
        """A stand-in for `normalize_text` (copies only stripping the outer brackets and removing
        ①)."""
        return body.strip().removeprefix("「").removesuffix("」").replace("①", "")

    def test_the_text_side_goes_through_normalization(self):
        tokenizer = _RecordingTokenizer()

        ip._packed_ids(tokenizer, self.CAPTION, 1, 64, self._bracket_stripper)

        assert tokenizer.seen == ["明るい声"]

    def test_the_caption_side_only_strips(self):
        """Upstream's `_synthesize` applies only `str(...).strip()` to the caption."""
        tokenizer = _RecordingTokenizer()

        ids = ip._packed_caption_ids(tokenizer, self.CAPTION, 1, 64)

        assert tokenizer.seen == ["「①明るい声」"]
        assert ids[0].tolist() == [1, *(ord(char) for char in "「①明るい声」")]

    def test_the_caption_body_budget_leaves_room_for_the_bos(self):
        ids = ip._packed_caption_ids(_RecordingTokenizer(), "あいうえお", 1, 3)

        assert ids[0].tolist() == [1, ord("あ"), ord("い")]

    def test_an_empty_caption_fails_loudly(self):
        """Conditioning on a BOS-only sequence would be yet another condition, different from no
        caption."""
        with pytest.raises(SystemExit, match="the caption is empty after strip"):
            ip._packed_caption_ids(_RecordingTokenizer(), "  \n ", 1, 64)


class _FakeCaptionTokenizer:
    """A stand-in that copies only **how** upstream's `PretrainedTextTokenizer.batch_encode` is
    called.

    The returned list is fixed (the first two mask entries set) — all this needs to see is "is
    the string passed in already stripped" and "does an empty caption zero the whole mask,
    BOS included".
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def batch_encode(self, texts, max_length: int):
        self.seen.extend(texts)
        ids = torch.zeros((1, max_length), dtype=torch.int64)
        mask = torch.zeros((1, max_length), dtype=torch.bool)
        mask[0, :2] = True
        return ids, mask


class TestUpstreamCaptionCondition:
    def test_the_caption_is_stripped_before_the_upstream_entrance(self):
        fake = _FakeCaptionTokenizer()

        _ids, mask = ip.upstream_caption_condition(fake, "  明るい声  ", 8)

        assert fake.seen == ["明るい声"]
        assert int(mask.sum()) == 2

    def test_an_empty_caption_zeroes_the_whole_mask(self):
        """Upstream `_synthesize`'s `caption_mask.zero_()` — it drops the BOS entry too."""
        fake = _FakeCaptionTokenizer()

        _ids, mask = ip.upstream_caption_condition(fake, "   ", 8)

        assert fake.seen == [""]
        assert not bool(mask.any())


class TestCaptionGoldenStability:
    def test_the_pipeline_captions_are_insensitive_to_normalization(self):
        """MUST: checks that the existing golden does not move (if it does, that is the signal to
        re-emit).

        The very premise of the change that moved the caption to strip-only. If this fails,
        `outputs/series/irodori-*/pipeline/` must be regenerated, or the TS-side integration
        check breaks.
        """
        pytest.importorskip("irodori_tts")
        from irodori_tts.text_normalization import normalize_text

        for case in ip.PIPELINE_CASES:
            assert normalize_text(case.caption).strip() == case.caption.strip(), case.name


#: The group length of the synthetic container (i4 makes no partial group — ADR 0069 decision 2).
GROUP = 32

#: The entries stored as i4 and as i8 (the listening ruling of 2026-08-23 took the 5 outside the
#: blocks and the 144 adaLN tensors out of i4). The read-back is driven by the container's
#: declarations, so one of each is enough for the synthetic container, but the expectation for
#: the real weights shows in `i4Source` of `meta.json`: **`dit` int4Tensors 96 / int8Tensors 149,
#: `dit-context` int4Tensors 72 / int8Tensors 0** (before the split, `dit` alone had 168 / 149 —
#: ADR 0114).
I4_KEY = "in_proj.weight"
SCALE_KEY = f"karume.scale.{I4_KEY}"
I8_KEY = "out_proj.weight"
I8_SCALE_KEY = f"karume.scale.{I8_KEY}"
#: A lifted-constant entry (spelled as in `karume.convert`) — present to check that it is left out
#: of the overwrite targets.
CONST_KEY = "const.0f0f0f0f0f0f0f0f"


class _DitWrapper(nn.Module):
    """A stand-in for `irodori.export.DitGraph` (copies only **which parameters it owns**).

    One i4 entry (`in_proj.weight` — its quantization axis divides by g32), one i8 entry
    (`out_proj.weight` — outside the blocks) and four f32 entries (bias / norm). The names are
    spelled as in the real one to test directly that the read-back works in the wrapper's FQN
    space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(2 * GROUP, 4)
        self.out_proj = nn.Linear(2 * GROUP, 4)
        self.out_norm = nn.LayerNorm(4)


def _shipped(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The shipped i4 form of `weight`, `(group scale, f32 restored from the shipped bytes)`.

    The expectation is restored **from the bytes written** (the same direction as the
    bit-identity check of `karume.emit` at storage time).
    """
    scale = group_scale(weight, GROUP)
    return scale, dequantize_int4(quantize_to_int4(weight, scale), scale)


def _shipped_i8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The same for an i8 entry (`(per-channel scale, f32 restored from the shipped bytes)`)."""
    scale = channel_scale(weight, 0)
    return scale, quantize_to_int8(weight, scale).to(torch.float32) * scale


def _init_name(key: str) -> str:
    """Tensor key → IR v1 initializer name (the spelling torch.export normalizes the FQN to)."""
    return "const_0f0f0f0f0f0f0f0f" if key == CONST_KEY else f"p_{key.replace('.', '_')}"


def _graph(tensors: Mapping[str, torch.Tensor]) -> IrGraph:
    """The graph of a synthetic `dit` (the i4 / i8 entries are consumed by **the weight slot of
    a linear**).

    Compressed storage is eligible only where "the tensor is consumed by a weight slot", so in a
    shell that only lists declarations `--dtype i4` takes effect nowhere (`karume.emit` fails
    loudly).

    The chain of adds at the exit is built only from the entries **present in the material** (the
    two norm tensors and the lifted constant) — material with one entry removed still makes a
    valid IR that references no undefined value (the publish check applies the IR's acceptance
    rules).
    """
    initializers = {
        _init_name(key): IrInitializer(tensor=key, storage=IrStorage(dtype="f32"))
        for key in tensors
    }
    values = {
        _init_name(key): IrValue(dtype="f32", shape=list(tensor.shape))
        for key, tensor in tensors.items()
    }
    chain = [key for key in ("out_norm.weight", "out_norm.bias", CONST_KEY) if key in tensors]
    links = [f"h{index + 2}" for index in range(len(chain) - 1)] + ["out"]
    values.update({name: IrValue(dtype="f32", shape=[1, 4]) for name in ("h0", "h1", *links)})
    nodes = [
        IrNode(
            op="linear",
            ins=["x", _init_name(I4_KEY), _init_name("in_proj.bias")],
            outs=["h0"],
            attrs={},
        ),
        IrNode(
            op="linear",
            ins=["x", _init_name(I8_KEY), _init_name("out_proj.bias")],
            outs=["h1"],
            attrs={},
        ),
        *(
            IrNode(op="add", ins=[previous, _init_name(key)], outs=[link], attrs={})
            for previous, key, link in zip(["h0", *links[:-1]], chain, links, strict=True)
        ),
    ]
    return IrGraph(
        inputs=[IrInput(name="x", dtype="f32", shape=[1, 2 * GROUP])],
        outputs=["out", "h1"],
        initializers=initializers,
        values=values,
        nodes=nodes,
    )


class _Material(NamedTuple):
    """The material of a synthetic container (each check breaks exactly one thing before
    passing it to {@link _write_series})."""

    tensors: dict[str, torch.Tensor]
    scales: dict[str, torch.Tensor]
    overrides: dict[str, str]
    #: Name of a parameter the module owns → the f32 value the read-back should put there.
    shipped: dict[str, torch.Tensor]


def _material(seed: int = 0) -> _Material:
    """Material of one i4 + one i8 + four f32 tensors + one lifted constant."""
    generator = torch.Generator().manual_seed(seed)
    i4_scale, i4_stored = _shipped(torch.randn(4, 2 * GROUP, generator=generator))
    i8_scale, i8_stored = _shipped_i8(torch.randn(4, 2 * GROUP, generator=generator))
    plain = {
        "in_proj.bias": torch.randn(4, generator=generator),
        "out_proj.bias": torch.randn(4, generator=generator),
        "out_norm.weight": torch.randn(4, generator=generator),
        "out_norm.bias": torch.randn(4, generator=generator),
    }
    tensors = {I4_KEY: i4_stored, I8_KEY: i8_stored, **plain, CONST_KEY: torch.zeros(1, 4)}
    return _Material(
        tensors=tensors,
        scales={I4_KEY: i4_scale, I8_KEY: i8_scale},
        overrides={I4_KEY: "i4", I8_KEY: "i8"},
        shipped={I4_KEY: i4_stored, I8_KEY: i8_stored, **plain},
    )


def _write_series(
    directory: Path,
    material: _Material,
    method: str | None = "gptq",
    budget: Mapping[str, int] | None = None,
    block_bytes: int = BLOCK_MAX_BYTES,
    graph: Callable[[Mapping[str, torch.Tensor]], IrGraph] | None = None,
) -> Path:
    """Write a synthetic i4 series (the `krm` parts + the calibration record; `method=None`
    drops the record).

    Writing goes through the same single path as the real one (`karume.emit.stored_model` →
    `karume.publish.publish_container`) — keeping a copy of the rules would let the fixture alone
    go stale the day the rules move, giving "tests green, only the real thing fails".

    `budget` is the two fields of the calibration budget (the default is exactly the
    distribution floor = the same shape as a shipped series). This is where the axis of trying
    to bake a golden from a series built with a smoke budget is varied. `block_bytes` injects a
    size that forces the piece split.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stored = stored_model(
        (_graph if graph is None else graph)(material.tensors),
        material.tensors,
        weight_dtype="f32",
        weight_scales=material.scales,
        weight_dtype_overrides=material.overrides,
    )
    publish_container(
        directory / ex.MODEL_FILE,
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name="dit",
        provenance=ex.PROVENANCE,
        block_bytes=block_bytes,
    )
    if method is not None:
        record = {
            "method": method,
            "grid": "rtn",
            "group_size": GROUP,
            **(irodori_calib_floor() if budget is None else budget),
        }
        (directory / CALIB_PROVENANCE_FILE).write_text(json.dumps(record), encoding="utf-8")
    return directory


def _restore(wrapper: nn.Module, series: Path) -> ip.RestoredDit:
    """Read back one container (calls {@link irodori.pipeline_ref.restore_dit_from_i4_series}
    with one target — `series` is `<target>/` directly under the series root).

    The per-container checks do not depend on the number of targets, so the tests of the
    existing checks try them one at a time in this form.
    """
    return ip.restore_dit_from_i4_series({series.name: wrapper}, series.parent)[series.name]


class TestRestoreDitFromI4Series:
    """MUST: the golden is baked from the shipped bytes (do not run calibration twice and bet on
    the results agreeing)."""

    def test_every_owned_parameter_comes_from_the_shipped_bytes(self, tmp_path):
        module = _DitWrapper()
        material = _material()
        series = _write_series(tmp_path / "dit", material)

        record = _restore(module, series)

        assert (record.int4, record.int8, record.plain, record.changed) == (1, 1, 4, 1)
        assert record.calib["method"] == "gptq"
        owned = dict(module.named_parameters())
        # The lifted constant (`const.*`) is not an entry — present or not, it is not overwritten.
        assert set(owned) == set(material.shipped)
        for key, value in material.shipped.items():
            assert torch.equal(owned[key].detach(), value), key

    def test_an_uncalibrated_series_is_refused(self, tmp_path):
        """The output of `--no-calib` has the same storage form = the assets cannot tell it
        apart (only the audio degrades)."""
        series = _write_series(tmp_path / "dit", _material(), method="rtn")

        with pytest.raises(SystemExit, match="rounding method allowed for distribution"):
            _restore(_DitWrapper(), series)

    def test_a_missing_provenance_record_is_refused(self, tmp_path):
        series = _write_series(tmp_path / "dit", _material(), method=None)

        with pytest.raises(SystemExit, match="no record of the i4 series' calibration conditions"):
            _restore(_DitWrapper(), series)

    def test_a_smoke_budget_series_is_refused(self, tmp_path):
        """`--calib-steps 1` leaves `method` at `gptq` — it passes unless the budget fields are
        checked too.

        If only the golden is baked with the smoke rounding, the comparison with the distributed
        assets stays green with "different weights on the two sides" (neither the storage form
        nor the count moves at all).
        """
        series = _write_series(
            tmp_path / "dit", _material(), budget={**irodori_calib_floor(), "steps": 1}
        )

        with pytest.raises(
            SystemExit, match="calibration budget 'steps' is below the distribution floor"
        ):
            _restore(_DitWrapper(), series)

    def test_a_tensor_the_module_does_not_own_fails_loudly(self, tmp_path):
        """An entry in the container but not in the module (other than a lifted constant) is an
        immediate error."""
        material = _material()
        material.tensors["extra.weight"] = torch.zeros(1, 4)
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="missing from the module"):
            _restore(_DitWrapper(), series)

    def test_a_parameter_missing_from_the_container_fails_loudly(self, tmp_path):
        """The opposite direction (in the module but not in the container) — an entry would be
        left un-overwritten."""
        material = _material()
        del material.tensors["out_norm.bias"]
        del material.shipped["out_norm.bias"]
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="missing from the container"):
            _restore(_DitWrapper(), series)

    def test_a_container_without_any_i4_tensor_fails_loudly(self, tmp_path):
        """Pointing at a directory that is not an i4 series (an f16 / i8 series)."""
        material = _material()
        material.overrides[I4_KEY] = "i8"
        material.scales[I4_KEY] = channel_scale(material.tensors[I4_KEY], 0)
        material.tensors[I4_KEY] = (
            quantize_to_int8(material.tensors[I4_KEY], material.scales[I4_KEY]).to(torch.float32)
            * material.scales[I4_KEY]
        )
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="no tensor is stored as i4"):
            _restore(_DitWrapper(), series)

    def test_a_restore_that_changes_nothing_fails_loudly(self, tmp_path):
        """The takes-effect check: if everything equals the stage-1 (i8 rounding) values, the
        read-back had no effect."""
        module = _DitWrapper()
        material = _material()
        with torch.no_grad():
            module.in_proj.weight.copy_(material.shipped[I4_KEY])
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="the i4 read-back has no effect"):
            _restore(module, series)

    def test_a_shape_mismatch_fails_loudly(self, tmp_path):
        """The container's `[1, 4]` against the module's `[4]` — a shape the graph's add accepts
        through broadcasting."""
        material = _material()
        material.tensors["out_norm.bias"] = torch.zeros(1, 4)
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="disagrees: container"):
            _restore(_DitWrapper(), series)

    def test_a_storage_outside_the_seat_fails_loudly(self, tmp_path):
        """The dit of the i8+dit4 preset holds no f16 (if it did, the mix would have shipped in an
        unexpected form).

        Only f32 / i8 / i4 are accepted ({@link irodori.pipeline_ref._RESTORE_STORAGE}).
        """
        material = _material()
        material.overrides[I8_KEY] = "f16"
        del material.scales[I8_KEY]
        material.tensors[I8_KEY] = material.tensors[I8_KEY].to(torch.float16).to(torch.float32)
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="cannot be read back"):
            _restore(_DitWrapper(), series)

    def test_the_i8_seat_does_not_count_as_the_i4_seat_working(self, tmp_path):
        """MUST: the takes-effect check counts **i4 only**.

        The i8 outside the blocks is on the same grid as stage 1 (i8 rounding), so its values not
        moving is normal, but they are not guaranteed to match on every run (stage 1 is the whole
        `dit`, stage 2 the container's bytes). Counting them in the takes-effect check would let
        the i8 differences cover up the accident of no i4 tensor taking effect.
        """
        module = _DitWrapper()
        material = _material()
        with torch.no_grad():
            module.in_proj.weight.copy_(material.shipped[I4_KEY])
        series = _write_series(tmp_path / "dit", material)
        assert not torch.equal(module.out_proj.weight.detach(), material.shipped[I8_KEY]), (
            "with the i8 entry equal to stage 1, the test cannot show that the takes-effect check"
            " counts i4 only"
        )

        with pytest.raises(SystemExit, match="the i4 read-back has no effect"):
            _restore(module, series)

    def test_a_missing_container_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="no container of the i4 series"):
            _restore(_DitWrapper(), tmp_path / "dit")


class TestRestoreDitFromASplitSeries:
    """The read-back gives the same result for a series with tensors split into pieces
    (container-v1 §4.2).

    A reader that does not join them would count the i4 entry piece by piece and overwrite it with
    a value cut short (= this one test decides whether joining happens).
    """

    def test_every_owned_parameter_still_comes_from_the_shipped_bytes(self, tmp_path):
        module = _DitWrapper()
        material = _material()
        # Lower the block limit to one row (32 bytes, since i4 packs 2 elements per byte) so the
        # 4-row weight is split into pieces.
        series = _write_series(tmp_path / "dit", material, block_bytes=64)
        # Observation point for the premise — with no piece at all this test would test nothing.
        verified = verify_container(container_parts(series / ex.MODEL_FILE))
        split = [
            name
            for bound in verified.graphs.values()
            for name, supply in bound.supplies.items()
            if len(supply.blocks) > 1
        ]
        assert split

        record = _restore(module, series)

        assert (record.int4, record.int8, record.plain, record.changed) == (1, 1, 4, 1)
        owned = dict(module.named_parameters())
        for key, value in material.shipped.items():
            assert torch.equal(owned[key].detach(), value), key


#: A tensor both DiT containers hold (the real `blocks.<b>.attention.k_norm.weight` — it applies to
#: the K of both the condition side and the self side, so both wrappers carry it. ADR 0114).
SHARED_KEY = "blocks.0.attention.k_norm.weight"
#: The i4 entry of `dit-context` (spelled like the real condition-side projection).
CONTEXT_I4_KEY = "blocks.0.attention.wk_text.weight"
CONTEXT_BIAS_KEY = "blocks.0.attention.wk_text.bias"


class _KeyNorm(nn.Module):
    """A stand-in for the shared `k_norm` (one weight of a type that is not quantized)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))


def _holder(entries: Mapping[str, nn.Module]) -> nn.Module:
    """A shell that places modules at FQNs (`blocks.0.attention.k_norm`) (a miniature of what the
    real wrappers own)."""
    root = nn.Module()
    for fqn, module in entries.items():
        *parents, leaf = fqn.split(".")
        node = root
        for part in parents:
            if not hasattr(node, part):
                node.add_module(part, nn.Module())
            node = getattr(node, part)
        node.add_module(leaf, module)
    return root


def _pair_graph(weight_key: str, bias_key: str) -> Callable[[Mapping[str, torch.Tensor]], IrGraph]:
    """The shell of a container that consumes one i4 linear + the shared tensor (a miniature of
    the two DiT containers)."""

    def build(tensors: Mapping[str, torch.Tensor]) -> IrGraph:
        keys = (weight_key, bias_key, SHARED_KEY)
        values = {
            _init_name(key): IrValue(dtype="f32", shape=list(tensors[key].shape)) for key in keys
        }
        values.update({name: IrValue(dtype="f32", shape=[1, 4]) for name in ("h", "out")})
        return IrGraph(
            inputs=[IrInput(name="x", dtype="f32", shape=[1, 2 * GROUP])],
            outputs=["out"],
            initializers={
                _init_name(key): IrInitializer(tensor=key, storage=IrStorage(dtype="f32"))
                for key in keys
            },
            values=values,
            nodes=[
                IrNode(
                    op="linear",
                    ins=["x", _init_name(weight_key), _init_name(bias_key)],
                    outs=["h"],
                    attrs={},
                ),
                IrNode(op="add", ins=["h", _init_name(SHARED_KEY)], outs=["out"], attrs={}),
            ],
        )

    return build


def _pair_material(weight_key: str, bias_key: str, shared: torch.Tensor, seed: int) -> _Material:
    """Material of one i4 tensor + bias + the shared tensor."""
    generator = torch.Generator().manual_seed(seed)
    scale, stored = _shipped(torch.randn(4, 2 * GROUP, generator=generator))
    bias = torch.randn(4, generator=generator)
    tensors = {weight_key: stored, bias_key: bias, SHARED_KEY: shared}
    return _Material(
        tensors=tensors,
        scales={weight_key: scale},
        overrides={weight_key: "i4"},
        shipped=dict(tensors),
    )


class TestRestoreBothDitSeries:
    """Reads back the two DiT containers (`dit` / `dit-context` — ADR 0114) in one call.

    The two are written from the same rounding by one export. The per-container checks live in
    {@link TestRestoreDitFromI4Series} above; this only looks at the consistency between the
    two.
    """

    @staticmethod
    def _wrappers() -> dict[str, nn.Module]:
        shared = _KeyNorm()
        return {
            ex.TARGET_DIT: _holder(
                {"in_proj": nn.Linear(2 * GROUP, 4), "blocks.0.attention.k_norm": shared}
            ),
            ex.TARGET_DIT_CONTEXT: _holder(
                {
                    "blocks.0.attention.wk_text": nn.Linear(2 * GROUP, 4),
                    "blocks.0.attention.k_norm": shared,
                }
            ),
        }

    @staticmethod
    def _series(
        root: Path,
        *,
        context_shared: torch.Tensor | None = None,
        context_budget: Mapping[str, int] | None = None,
    ) -> tuple[_Material, _Material]:
        shared = torch.full((4,), 1.5)
        dit = _pair_material("in_proj.weight", "in_proj.bias", shared, seed=1)
        context = _pair_material(
            CONTEXT_I4_KEY,
            CONTEXT_BIAS_KEY,
            shared if context_shared is None else context_shared,
            seed=2,
        )
        _write_series(
            root / ex.TARGET_DIT, dit, graph=_pair_graph("in_proj.weight", "in_proj.bias")
        )
        _write_series(
            root / ex.TARGET_DIT_CONTEXT,
            context,
            budget=context_budget,
            graph=_pair_graph(CONTEXT_I4_KEY, CONTEXT_BIAS_KEY),
        )
        return dit, context

    def test_both_containers_overwrite_the_parameters_their_wrapper_owns(self, tmp_path):
        wrappers = self._wrappers()
        dit, context = self._series(tmp_path)

        records = ip.restore_dit_from_i4_series(wrappers, tmp_path)

        assert set(records) == {ex.TARGET_DIT, ex.TARGET_DIT_CONTEXT}
        for target, material in ((ex.TARGET_DIT, dit), (ex.TARGET_DIT_CONTEXT, context)):
            record = records[target]
            # The i4 takes-effect check holds per container (each has one i4 tensor, one that
            # moved).
            assert (record.int4, record.changed) == (1, 1), target
            owned = dict(wrappers[target].named_parameters())
            assert set(owned) == set(material.shipped)
            for key, value in material.shipped.items():
                assert torch.equal(owned[key].detach(), value), (target, key)

    def test_a_shared_tensor_that_differs_between_the_containers_fails_loudly(self, tmp_path):
        """A container read later must not silently overwrite the earlier values (a mix of
        separate export runs)."""
        self._series(tmp_path, context_shared=torch.full((4,), 2.5))

        with pytest.raises(SystemExit, match="shared tensors should be the same bytes"):
            ip.restore_dit_from_i4_series(self._wrappers(), tmp_path)

    def test_calibration_records_that_differ_between_the_containers_fail_loudly(self, tmp_path):
        """Each record meets the distribution conditions on its own (a budget at or above the
        floor) — one at a time they pass."""
        self._series(tmp_path, context_budget={**irodori_calib_floor(), "steps": ip.NUM_STEPS + 1})

        with pytest.raises(SystemExit, match="records differ per target"):
            ip.restore_dit_from_i4_series(self._wrappers(), tmp_path)

    def test_a_missing_context_container_fails_loudly(self, tmp_path):
        """The `dit` container alone does not make the DiT of the i8+dit4 preset (the 72
        condition-side tensors stay i8)."""
        _write_series(
            tmp_path / ex.TARGET_DIT,
            _pair_material("in_proj.weight", "in_proj.bias", torch.ones(4), seed=1),
            graph=_pair_graph("in_proj.weight", "in_proj.bias"),
        )

        with pytest.raises(SystemExit, match="no container of the i4 series"):
            ip.restore_dit_from_i4_series(self._wrappers(), tmp_path)


class TestEulerSharesOneContext:
    """The condition-side K/V is built once per generation, and cond and uncond use the same
    values (ADR 0114 / 0047 decision 1)."""

    def test_every_dit_forward_receives_the_same_context_and_only_dit_is_counted(self):
        context = (torch.ones(1, 2), torch.full((1, 2), 2.0))
        received: list[tuple[torch.Tensor, ...]] = []

        def dit(x_t: torch.Tensor, _t_embed, _mask, *kv: torch.Tensor) -> torch.Tensor:
            received.append(kv)
            return torch.zeros_like(x_t)

        graphs = ip.HostGraphs(
            backbone=nn.Identity(),
            text_proj=nn.Identity(),
            caption_proj=nn.Identity(),
            speaker=nn.Identity(),
            duration=nn.Identity(),
            dit_context=nn.Identity(),
            dit=dit,  # type: ignore[arg-type]
        )
        source = SimpleNamespace(timestep_embedding=lambda t, dim: torch.zeros(1, dim))
        segments = dict.fromkeys(ex.DIT_UNCOND_VARIANTS, 2)

        _z, forwards = ip._euler(
            graphs,
            source,  # type: ignore[arg-type]
            torch.zeros(1, 3, 4),
            context,
            {name: 1 for name in segments},
            segments,
            ["text", "caption"],
            4,
            ip.t_schedule(4),
        )

        # Steps inside the CFG window (t ∈ [0.5, 1.0]) run cond + two uncond, the rest cond only.
        assert forwards == len(received) > 4
        assert all(
            len(kv) == len(context) and all(a is b for a, b in zip(kv, context, strict=True))
            for kv in received
        )


class TestMeanFlowSchedule:
    """The schedule of a MeanFlow checkpoint is upstream's own (ADR 0150)."""

    @pytest.mark.parametrize("steps", [1, 2, 3, 4, 7, 16, 40])
    def test_it_is_upstreams_float32_linspace_from_one_to_zero(self, steps):
        schedule = ip.meanflow_schedule(steps)

        assert schedule.dtype == torch.float32
        assert torch.equal(schedule, torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32))
        assert float(schedule[0]) == 1.0
        assert float(schedule[-1]) == 0.0

    def test_the_reference_loop_length_follows_the_flow(self):
        assert ip.reference_steps(IRODORI_FLOW_RF) == ip.NUM_STEPS == 40
        assert ip.reference_steps(IRODORI_FLOW_MEANFLOW) == ip.MEANFLOW_NUM_STEPS == 4

    def test_an_unknown_flow_fails_loudly(self):
        with pytest.raises(SystemExit, match="Unknown flow"):
            ip.reference_steps("ddpm")

    def test_the_calibration_floor_is_the_whole_meanflow_loop(self):
        assert irodori_calib_floor(IRODORI_FLOW_MEANFLOW)["steps"] == ip.MEANFLOW_NUM_STEPS
        assert irodori_calib_floor()["steps"] == ip.NUM_STEPS


class TestMeanFlowLoop:
    """One conditional forward per step with the interval, and `x += v (t_next − t)` (upstream's
    `sample_euler_meanflow`)."""

    @staticmethod
    def _run(
        *, zero_interval: bool = False
    ) -> tuple[torch.Tensor, int, list[tuple[float, float, torch.Tensor]], torch.Tensor]:
        context = (torch.ones(1, 2), torch.full((1, 2), 2.0))
        received: list[tuple[float, float, torch.Tensor]] = []

        def dit(x_t, t_embed, delta_embed, mask, *kv):
            assert all(a is b for a, b in zip(kv, context, strict=True))
            received.append((float(t_embed[0, 0]), float(delta_embed[0, 0]), mask))
            # A velocity that depends on both scalars, so the update is checked numerically.
            return torch.full_like(x_t, float(t_embed[0, 0]) + 10.0 * float(delta_embed[0, 0]))

        graphs = ip.HostGraphs(
            backbone=nn.Identity(),
            text_proj=nn.Identity(),
            caption_proj=nn.Identity(),
            speaker=nn.Identity(),
            duration=nn.Identity(),
            dit_context=nn.Identity(),
            dit=dit,  # type: ignore[arg-type]
        )
        # The embedding writes its scalar into the first column (the loop must pass t and delta).
        source = SimpleNamespace(
            timestep_embedding=lambda value, dim: value.reshape(1, 1).expand(1, dim).clone()
        )
        segments = dict.fromkeys(ex.DIT_UNCOND_VARIANTS, 2)
        schedule = ip.meanflow_schedule(4)
        noise = torch.zeros(1, 3, 4)
        z, forwards = ip._meanflow(
            graphs,
            source,  # type: ignore[arg-type]
            noise,
            context,
            {name: 1 for name in segments},
            segments,
            4,
            schedule,
            zero_interval=zero_interval,
        )
        return z, forwards, received, schedule

    def test_every_step_runs_exactly_one_conditional_forward(self):
        _z, forwards, received, _schedule = self._run()
        cond = ip._segment_masks(
            3,
            dict.fromkeys(ex.DIT_UNCOND_VARIANTS, 1),
            dict.fromkeys(ex.DIT_UNCOND_VARIANTS, 2),
            None,
        )

        assert forwards == len(received) == 4
        assert all(torch.equal(mask, cond) for _t, _d, mask in received)

    def test_each_forward_gets_t_and_the_interval_to_the_next_t(self):
        _z, _forwards, received, schedule = self._run()

        for index, (t, delta, _mask) in enumerate(received):
            assert t == float(schedule[index])
            assert delta == float(schedule[index] - schedule[index + 1])

    def test_the_update_is_x_plus_v_times_t_next_minus_t(self):
        z, _forwards, received, schedule = self._run()
        x = torch.zeros(1, 3, 4)
        for index, (t, delta, _mask) in enumerate(received):
            velocity = torch.full_like(x, t + 10.0 * delta)
            x = x + velocity * (schedule[index + 1] - schedule[index])

        assert torch.equal(z, x)

    def test_the_zero_interval_run_embeds_every_interval_as_zero(self):
        _z, _forwards, received, _schedule = self._run(zero_interval=True)

        assert [delta for _t, delta, _mask in received] == [0.0] * 4


class TestCli:
    def test_the_default_out_dir_is_derived_from_the_weight_directory(self, tmp_path):
        out = ip.default_out_dir(tmp_path / "v4-small")

        assert out.name == "pipeline"
        assert out.parent.name == "irodori-v4-small"

    def test_each_dtype_writes_into_its_own_series(self, tmp_path):
        """MUST: goldens are per series too (comparing f16 assets with the f32 golden mixes the
        quantization error into the tolerance, and only the detection power drops while it stays
        green)."""
        dirs = {
            dtype: ip.default_out_dir(tmp_path / "v4-small", dtype) for dtype in ex.WEIGHT_DTYPES
        }

        assert len(set(dirs.values())) == len(dirs)
        assert dirs["f16"].parent.name == "irodori-v4-small-f16"
        # The spelling of the series root is shared with the export script (not copied).
        for dtype, path in dirs.items():
            assert path.parent == ex.default_out_root(tmp_path / "v4-small", dtype)

    def test_the_dtype_is_forwarded(self, monkeypatch):
        seen: list[object] = []
        monkeypatch.setattr(ip, "emit", lambda *args: seen.append(args) or {"dir": "x"})
        ip.main(["--dtype", "f16"])

        assert seen[0][-1] == "f16"

    def test_the_case_names_are_unique(self):
        names = [case.name for case in ip.PIPELINE_CASES]

        assert len(names) == len(set(names))

    def test_the_reference_frames_divide_by_the_speaker_patch_size(self):
        """Without exact division, upstream's `patch_sequence_with_mask` drops the remainder and
        diverges from the saved input."""
        for case in ip.PIPELINE_CASES:
            if case.reference is not None:
                assert case.reference.frames % 4 == 0


class TestEulerReferenceSensitivity:
    """The pass/fail function of the second stage (judging the implementation difference
    normalized by the amplification factor — the form added on 2026-09-01)."""

    def test_the_measured_v41_f16_pair_passes(self):
        """The measured pair that motivated it (worst 1.33e-2 / amp 37,107) passes the second
        stage."""
        assert ip.euler_reference_within_sensitivity(1.33e-2, 37_107.0)

    def test_a_low_amplification_trajectory_keeps_the_tight_bound(self):
        """On a trajectory with mild amplification (v4-small class, amp≈353) a difference of 1e-2
        fails as a regression."""
        assert not ip.euler_reference_within_sensitivity(1.0e-2, 353.0)

    def test_the_absolute_ceiling_stops_gross_breakage(self):
        """An error the size of a wrong formula (O(1)) fails however large the amplification
        is."""
        assert not ip.euler_reference_within_sensitivity(0.5, 1.0e9)

    def test_the_ceiling_binds_even_when_normalized_noise_is_small(self):
        """The absolute ceiling binds first, even for a size the normalization would pass."""
        amp = 1.0e5  # amp × EULER_NOISE_PER_AMP = 0.5 > the ceiling 5e-2
        worst = 1.0e-1
        assert worst <= amp * ip.EULER_NOISE_PER_AMP
        assert not ip.euler_reference_within_sensitivity(worst, amp)
