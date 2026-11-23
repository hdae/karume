"""The pure part of `embeddinggemma2/measure_quant_towers.py` (what needs no real weights).

The measurement itself is manual (the real weights on CPU). What is pinned here is the side whose
breakage makes the tables silently lie:

- the quality set's manifest is refused unless it is exactly the format the script measures
- the resampler keeps a band-limited tone and the length rule, and the audio limits are the
  pipeline's
- the retrieval metrics count a shared transcript as a hit for any of its clips
- the bf16 set splits into module weights / tables / structural constants exactly once each
- the rounding reaches exactly the intended weights (a table through its holder too), the tower
  payload arithmetic
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from embeddinggemma2 import measure_quant as mq
from embeddinggemma2 import measure_quant_towers as mt

# ---- The quality set ---------------------------------------------------------------------------


def _write_png(path: Path, mode: str = "RGB") -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    channels = {"RGB": 3, "RGBA": 4}[mode]
    Image.fromarray(np.full((8, 8, channels), 100, dtype=np.uint8), mode).save(path)


def _write_wav(path: Path, rate: int, samples: np.ndarray, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.repeat(np.round(samples * 32767).astype("<i2")[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _manifest(root: Path) -> dict[str, Any]:
    for name in ("a", "b"):
        _write_png(root / "images" / f"{name}.png")
        _write_wav(root / "audio" / f"{name}.wav", 16_000, np.zeros(1_000))
    return {
        "format": mt.MANIFEST_FORMAT,
        "images": [
            {
                "file": f"images/{name}.png",
                "caption": f"caption {name}",
                "generator": "g",
                "prompt": "p",
                "seed": index,
            }
            for index, name in enumerate("ab")
        ],
        "audio": [
            {
                "file": "audio/a.wav",
                "transcript": "one",
                "generator": "sbv2-jvnv/x",
                "sampleRate": 16_000,
            },
            {
                "file": "audio/b.wav",
                "transcript": "two",
                "generator": "sbv2-jvnv/y",
                "sampleRate": 16_000,
            },
        ],
    }


class TestManifest:
    def test_a_well_formed_manifest_gives_every_entry(self, tmp_path: Path) -> None:
        (tmp_path / mt.MANIFEST_FILE).write_text(json.dumps(_manifest(tmp_path)))

        quality = mt.load_quality_set(tmp_path)

        assert [entry.file for entry in quality.images] == ["images/a.png", "images/b.png"]
        assert [(entry.transcript, entry.sample_rate) for entry in quality.audio] == [
            ("one", 16_000),
            ("two", 16_000),
        ]

    def test_a_missing_manifest_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(mt.QualitySetError, match="does not exist"):
            mt.load_quality_set(tmp_path)

    def test_another_format_is_refused(self, tmp_path: Path) -> None:
        document = _manifest(tmp_path) | {"format": "karume-embeddinggemma2-quality-set/2"}

        with pytest.raises(mt.QualitySetError, match="format"):
            mt.validate_manifest(document, tmp_path)

    @pytest.mark.parametrize(
        ("kind", "index", "change", "message"),
        [
            ("images", 0, {"seed": True}, "is not a int"),
            ("images", 0, {"caption": "  "}, "empty"),
            ("images", 1, {"extra": 1}, "keys"),
            ("audio", 0, {"sampleRate": 0}, "is not > 0"),
            ("audio", 1, {"file": "audio/../a.wav"}, "is not audio/<name>.wav"),
            ("audio", 1, {"file": "audio/missing.wav"}, "does not exist"),
            ("images", 1, {"file": "images/a.png"}, "listed twice"),
        ],
    )
    def test_a_malformed_entry_is_refused(
        self, tmp_path: Path, kind: str, index: int, change: dict[str, Any], message: str
    ) -> None:
        document = _manifest(tmp_path)
        document[kind][index] = document[kind][index] | change

        with pytest.raises(mt.QualitySetError, match=message):
            mt.validate_manifest(document, tmp_path)

    def test_a_modality_with_one_distinct_text_is_refused(self, tmp_path: Path) -> None:
        """Retrieval among one candidate has no ranking to compare."""
        document = _manifest(tmp_path)
        document["audio"][1]["transcript"] = "one"

        with pytest.raises(mt.QualitySetError, match="1 distinct texts"):
            mt.validate_manifest(document, tmp_path)


class TestReading:
    def test_an_rgba_png_is_refused(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "x.png", "RGBA")

        with pytest.raises(mt.QualitySetError, match="not an 8-bit RGB PNG"):
            mt.read_png_rgb8(tmp_path / "x.png")

    def test_a_stereo_clip_is_averaged_to_mono_at_the_pcm_scale(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "x.wav", 16_000, np.full(4, 0.5), channels=2)

        pcm, rate = mt.read_wav_pcm16(tmp_path / "x.wav")

        assert rate == 16_000
        assert pcm.shape == (4, 2)
        assert mt.pcm_to_mono(pcm).tolist() == [16384 / 32768] * 4

    def test_the_declared_rate_must_be_the_files(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "x.wav", 22_050, np.zeros(2_000))

        with pytest.raises(mt.QualitySetError, match="22050 Hz, the manifest says 44100"):
            mt.clip_samples(tmp_path / "x.wav", 44_100)


class TestResample:
    def test_an_equal_rate_is_the_signal_itself(self) -> None:
        values = np.random.default_rng(0).standard_normal(500)

        assert np.array_equal(mt.resample(values, 16_000, 16_000), values)

    def test_a_tone_in_band_survives_44100_to_16000(self) -> None:
        rate_in, rate_out, frequency = 44_100, 16_000, 1_000.0
        values = np.sin(2 * np.pi * frequency * np.arange(44_100) / rate_in)

        output = mt.resample(values, rate_in, rate_out)

        expected = np.sin(2 * np.pi * frequency * np.arange(output.shape[0]) / rate_out)
        # Away from the edges, where the zero padding enters the window.
        assert np.abs(output - expected)[200:-200].max() < 1e-3

    def test_a_tone_above_the_new_nyquist_is_removed(self) -> None:
        values = np.sin(2 * np.pi * 12_000.0 * np.arange(44_100) / 44_100)

        output = mt.resample(values, 44_100, 16_000)

        assert np.abs(output[200:-200]).max() < 1e-3

    def test_the_length_is_the_count_of_output_instants(self) -> None:
        assert mt.resampled_length(44_100, 44_100, 16_000) == 16_000
        assert mt.resampled_length(441, 44_100, 16_000) == math.ceil(441 * 16_000 / 44_100)
        assert mt.resample(np.zeros(442), 44_100, 16_000).shape == (161,)

    @pytest.mark.parametrize(
        ("samples", "message"), [(160, "no mel frame"), (480_001, "more than 480000")]
    )
    def test_the_pipeline_limits_are_errors(self, samples: int, message: str) -> None:
        with pytest.raises(mt.QualitySetError, match=message):
            mt.check_audio_length(samples, "x")

    @pytest.mark.parametrize("samples", [161, 480_000])
    def test_the_limits_themselves_pass(self, samples: int) -> None:
        mt.check_audio_length(samples, "x")


# ---- Metrics -----------------------------------------------------------------------------------


class TestRetrievalMetrics:
    def test_distinct_texts_keep_the_first_order(self) -> None:
        assert mt.distinct_texts(["b", "a", "b", "c"]) == (("b", "a", "c"), (0, 1, 0, 2))

    def test_a_shared_transcript_counts_any_of_its_clips(self) -> None:
        """Clips 0 and 1 speak text 0, clip 2 speaks text 1. Text 0 ranks clip 1 first (a hit),
        text 1 ranks clip 0 first (a miss)."""
        item_text = (0, 0, 1)
        text_by_item = torch.tensor([[0.2, 0.9, 0.1], [0.8, 0.3, 0.5]])

        assert mt.text_to_item_hits(text_by_item, item_text) == 1

    def test_item_to_text_needs_the_items_own_text_first(self) -> None:
        item_text = (0, 0, 1)
        item_by_text = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.1, 0.7]])

        assert mt.item_to_text_hits(item_by_text, item_text) == 2

    def test_retrieval_metrics_of_a_swap(self) -> None:
        reference = torch.tensor([[0.9, 0.5, 0.1], [0.2, 0.8, 0.3]])
        swapped = torch.tensor([[0.5, 0.9, 0.1], [0.2, 0.8, 0.3]])

        metrics = mt.retrieval_metrics(swapped, reference, mt.item_to_text_hits, (0, 1))

        assert metrics == {
            "queries": 2,
            "candidates": 3,
            "referenceTop1": 2,
            "top1": 1,
            "top1Agreement": 1,
            "rankInversionRate": pytest.approx(1 / 6),
        }

    def test_measure_width_reads_the_quality_rows_only_for_retrieval(self) -> None:
        items = mt.ModalityItems(
            names=("q0", "q1", "synthetic:s"),
            kinds=(mt.QUALITY, mt.QUALITY, mt.SYNTHETIC),
            item_text=(0, 1, None),
            texts=("t0", "t1"),
        )
        vectors = torch.eye(3, 4)
        texts = torch.eye(2, 4)

        metrics = mt.measure_width(items, vectors, vectors, texts, texts, 4)

        assert metrics["itemToText"]["queries"] == 2
        assert metrics["textToItem"]["candidates"] == 2
        assert metrics["itemToText"]["top1"] == 2
        assert metrics["cosine"]["synthetic"]["items"] == 1
        assert metrics["itemTextCosineMaxAbsChange"] == 0.0

    def test_the_worst_item_is_named(self) -> None:
        cosines = torch.tensor([0.99, 0.95, 0.999], dtype=torch.float64)

        summary = mt.cosine_groups(cosines, ["a", "b", "c"], [mt.QUALITY, mt.SYNTHETIC, mt.QUALITY])

        assert summary["worst"] == "b"
        assert summary[mt.QUALITY] == {"items": 2, "min": 0.99, "mean": pytest.approx(0.9945)}

    def test_soft_tokens_are_compared_per_token(self) -> None:
        reference = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
        values = [torch.tensor([[1.0, 0.0], [1.0, 1.0]])]

        summary = mt.soft_token_summary(values, reference, ["x"])

        assert summary["tokens"] == 2
        assert summary["min"] == pytest.approx(1 / math.sqrt(2))
        assert summary["mean"] == pytest.approx((1 + 1 / math.sqrt(2)) / 2)


class TestItemIds:
    def test_one_block_alone_between_bos_and_eos(self) -> None:
        assert mt.item_ids(2, 1, (10, 11, 12), 3) == [2, 10, 11, 11, 11, 12, 1]
        assert mt.item_ids(2, 1, (10, 11, 12), 3)[mt.SOFT_START] == 11


# ---- The rounded sets and the payload ------------------------------------------------------------


class TestClassifyBf16Set:
    def test_every_kind_is_split_out(self) -> None:
        groups = mt.classify_bf16_set(
            ["a.weight", "pos_x", "sel", "b.weight"],
            {"a.weight", "b.weight", "unused.weight"},
            {"pos_x"},
            {"sel"},
        )

        assert groups == (["a.weight", "b.weight"], ["pos_x"], ["sel"])

    def test_a_bf16_tensor_of_no_kind_fails_loudly(self) -> None:
        with pytest.raises(AssertionError, match="'rope' is 0 kinds"):
            mt.classify_bf16_set(["rope"], {"a.weight"}, set(), set())

    def test_a_bf16_tensor_of_two_kinds_fails_loudly(self) -> None:
        with pytest.raises(AssertionError, match="is 2 kinds"):
            mt.classify_bf16_set(["t"], {"t"}, {"t"}, set())


class TinyTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 4, bias=False)
        self.conv = nn.Conv1d(4, 4, 5, groups=4)
        self.pos_x = nn.Parameter(torch.randn(6, 32), requires_grad=False)
        self.register_buffer("rope", torch.randn(6, 8))


def _tiny_weights(model: TinyTower) -> mt.TowerWeights:
    return mt.TowerWeights(
        name="vision",
        module_weights=["fc.weight", "conv.weight"],
        tables=["pos_x"],
        kept_bf16=[],
        rounded_stats=mq.GroupStats(3, 0, 0),
        kept_stats=mq.GroupStats(0, 0, 0),
        remainder_elements=0,
        remainder_tensors=0,
        live={
            "vision:fc.weight": model.fc.weight,
            "vision:conv.weight": model.conv.weight,
            "vision:pos_x": model.pos_x,
        },
    )


class TestRoundTower:
    def test_exactly_the_set_is_rounded_and_a_restore_gives_it_back(self) -> None:
        torch.manual_seed(0)
        model = TinyTower()
        weights = _tiny_weights(model)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        pristine = {key: tensor.detach().clone() for key, tensor in weights.live.items()}

        rounded = mt.round_tower(model, weights)

        mq.assert_rounded(rounded, set(weights.keys), "tiny")
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, before[name]) == (name in {"rope", "conv.bias"})
        mt.restore_all({"vision": weights}, pristine)
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, before[name])

    def test_a_table_is_rounded_per_row_like_an_embedding(self) -> None:
        torch.manual_seed(1)
        model = TinyTower()
        embedding = nn.Embedding(6, 32)
        with torch.no_grad():
            embedding.weight.copy_(model.pos_x)
        mq_holder = nn.Module()
        mq_holder.add_module("table", embedding)

        mt.round_tower(model, _tiny_weights(model))
        mt.fake_quant_int8(mq_holder)

        assert torch.equal(model.pos_x, embedding.weight)


class TestTowerPayload:
    def test_bf16_and_i8_add_the_kept_set_and_the_remainder(self) -> None:
        rounded = mq.GroupStats(tensors=2, elements=64, rows=6)
        kept = mq.GroupStats(tensors=1, elements=10, rows=1)

        payload = mt.tower_payload(rounded, kept, remainder_elements=5)

        assert payload["bf16"]["bytes"] == (16 * 64 + 16 * 10 + 32 * 5) // 8
        assert payload["i8"]["bytes"] == (8 * 64 + 32 * 6 + 16 * 10 + 32 * 5) // 8


class TestConfigs:
    def test_text_i8_and_all_i8_measure_both_modalities_and_each_tower_its_own(self) -> None:
        by_name = {config.name: config for config in mt.CONFIGS}

        assert by_name["vision-i8"].modalities == (mt.IMAGE,)
        assert by_name["audio-i8"].modalities == (mt.AUDIO,)
        assert by_name["text-i8"].towers == {mt.TEXT_TOWER}
        assert by_name["all-i8"].towers == set(mt.TOWERS)
