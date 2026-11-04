"""The script-level rules of `sbv2/demo.py` (only pure functions that need neither the upstream
package nor real weights).

Running the demo and preparing the assets are manual (they need `style_bert_vits2`). What is pinned
here is only the discipline whose breakage would **silently turn the distribution or the comparison
of the 3 WAVs into something else**:

- `wav_pcm16` rounds with `floor(x + 0.5)` (the same rule as JS `Math.round`) — slipping to
  Python's built-in `round` (round half to even) gives a WAV 1 LSB off from the same waveform, and
  "listening to out / reference / official side by side" becomes a comparison mixed with an
  implementation difference
- the postconditions of the folding in `clean_text_ranges` (ascending, not adjacent) and which
  list the boundary code points go into — the output becomes the distribution's tokenizer asset as
  is
- a count mismatch in `tile_bert` fails loudly
- `dump_metadata` does not accept a safetensors without the `demo` field
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import struct
import types

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from sbv2 import demo


def _wav_fields(blob: bytes) -> dict[str, int]:
    """The numeric fields of the WAV header (the 16 bytes of the `fmt ` chunk + the `data`
    length)."""
    (
        _size,
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
    ) = struct.unpack("<IHHIIHH", blob[16:36])
    (data_length,) = struct.unpack("<I", blob[40:44])
    return {
        "audioFormat": audio_format,
        "channels": channels,
        "sampleRate": sample_rate,
        "byteRate": byte_rate,
        "blockAlign": block_align,
        "bits": bits,
        "dataLength": data_length,
    }


class TestWavPcm16:
    def test_the_header_declares_16bit_mono_pcm(self):
        blob = demo.wav_pcm16(np.zeros(8, dtype=np.float32), 44100)

        assert blob[:4] == b"RIFF"
        assert blob[8:12] == b"WAVE"
        assert blob[12:16] == b"fmt "
        assert blob[36:40] == b"data"
        assert _wav_fields(blob) == {
            "audioFormat": 1,
            "channels": 1,
            "sampleRate": 44100,
            "byteRate": 44100 * 2,
            "blockAlign": 2,
            "bits": 16,
            "dataLength": 16,
        }
        # The RIFF size field is "every byte after it" = 36 + the data length.
        assert struct.unpack("<I", blob[4:8])[0] == 36 + 16
        assert len(blob) == 44 + 16

    def test_the_rounding_is_floor_of_x_plus_half(self):
        """MUST: the same rule as JS `Math.round` (round half to even would give [0, 2, 0, -2])."""
        samples = np.array([0.5, 1.5, -0.5, -1.5], dtype=np.float64) / 32767.0

        pcm = np.frombuffer(demo.wav_pcm16(samples, 48000)[44:], dtype="<i2")

        assert list(pcm) == [1, 2, 0, -1]

    def test_it_clips_to_unity_without_using_the_lower_end(self):
        """Clipping is at ±1.0 and the low end −32768 is not used (an amplitude symmetric in
        sign)."""
        pcm = np.frombuffer(demo.wav_pcm16(np.array([2.0, -2.0]), 48000)[44:], dtype="<i2")

        assert list(pcm) == [32767, -32767]


def _in_ranges(ranges: list[list[int]], cp: int) -> bool:
    """Whether a code point is in a list of closed intervals (the test, not the script, owns the
    scan)."""
    return any(start <= cp <= end for start, end in ranges)


class TestCleanTextRanges:
    """The very result of making Python the single source of truth for the classification (no
    Unicode classification table is brought into TS)."""

    @pytest.fixture(scope="class")
    @staticmethod
    def ranges() -> dict[str, list[list[int]]]:
        return demo.clean_text_ranges()

    def test_both_lists_are_folded_ascending_and_never_adjacent(self, ranges):
        """The postcondition of the folding — 2 adjacent intervals left over mean the folding did
        not work."""
        for name, table in ranges.items():
            assert table, name
            for start, end in table:
                assert start <= end, (name, start, end)
            for current, following in itertools.pairwise(table):
                assert current[1] + 1 < following[0], (name, current, following)

    def test_the_removed_side_holds_the_null_the_replacement_and_the_surrogates(self, ranges):
        for cp in (0x0000, 0xFFFD, 0xD800, 0xDFFF):
            assert _in_ranges(ranges["removed"], cp), hex(cp)
            assert not _in_ranges(ranges["spaced"], cp), hex(cp)

    def test_the_spaced_side_holds_the_whitespace_including_the_three_controls(self, ranges):
        """`\\t` `\\n` `\\r` are **taken out of** the control class and put on the whitespace side
        (the reference implementation's branch)."""
        for cp in (0x0020, 0x00A0, ord("\t"), ord("\n"), ord("\r")):
            assert _in_ranges(ranges["spaced"], cp), hex(cp)
            assert not _in_ranges(ranges["removed"], cp), hex(cp)

    def test_ordinary_characters_are_in_neither_list(self, ranges):
        for cp in (ord("あ"), ord("A"), ord("1"), ord("。")):
            assert not _in_ranges(ranges["removed"], cp), hex(cp)
            assert not _in_ranges(ranges["spaced"], cp), hex(cp)


class TestTileBert:
    def test_it_repeats_each_token_and_returns_the_transposed_matrix(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        tiled = demo.tile_bert(hidden, [1, 3])

        assert tiled.shape == (2, 4)
        assert torch.equal(tiled, torch.tensor([[1.0, 3.0, 3.0, 3.0], [2.0, 4.0, 4.0, 4.0]]))

    def test_a_token_count_that_disagrees_with_word2ph_fails_loudly(self):
        with pytest.raises(AssertionError, match="differs from the word2ph length"):
            demo.tile_bert(torch.zeros(3, 2), [1, 1])


class TestDumpMetadata:
    def test_it_reads_the_demo_record(self, tmp_path):
        path = tmp_path / "dump.safetensors"
        record = {"text": "テスト", "knobs": {"sdpRatio": 0.2}}
        save_file({"audio": torch.zeros(4)}, str(path), metadata={"demo": json.dumps(record)})

        assert demo.dump_metadata(path) == record

    def test_a_dump_without_the_demo_field_fails_loudly(self, tmp_path):
        path = tmp_path / "other.safetensors"
        save_file({"audio": torch.zeros(4)}, str(path), metadata={"other": "1"})

        with pytest.raises(ValueError, match=r"has no __metadata__\.demo"):
            demo.dump_metadata(path)


@pytest.mark.skipif(
    importlib.util.find_spec("style_bert_vits2") is None,
    reason="the sbv2 dependency group is missing (`uv sync --group sbv2` is required)",
)
class TestSharedSymbolsCarryOnlyTheCommonRules:
    """`symbols.json` is a **shared slot** of the distribution (the same single file for every
    speaker) — it carries no model-specific facts.

    Carrying the baked model's style / speaker / config `version` would claim the wrong model name
    for another speaker. Knob defaults are not carried either (the source of truth is the manifest's
    `pipelineConfig.defaults`, and the TS side does not read the knobs here — W-RC5-1).
    """

    def test_the_rules_have_no_model_specific_fields(self, monkeypatch) -> None:
        # The other 2 rules read the source and the HF config, so they are injected for this
        # check.
        monkeypatch.setattr(demo, "blank_id_from_source", lambda: 0)
        monkeypatch.setattr(demo, "bert_rel_pos_rule", lambda: {"positionBuckets": 1})
        hps = types.SimpleNamespace(
            version="2.0-JP-Extra",
            data=types.SimpleNamespace(add_blank=True, sampling_rate=44100, hop_length=512),
        )

        rules = demo.jp_extra_rules(hps)

        assert not {"defaults", "style", "speaker"} & set(rules)
        assert "modelVersion" not in rules["source"]
        assert "2.0-JP-Extra" not in json.dumps(rules)
