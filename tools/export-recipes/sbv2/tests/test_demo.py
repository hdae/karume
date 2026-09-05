"""`sbv2/demo.py` の台本レベルの約束事（上流パッケージも実重みも要らない純関数だけ）。

デモの実行と資産 prep は手動（`style_bert_vits2` が要る）。ここで固定するのは、壊れると
**配布形か 3 本の WAV 比較が黙って別物になる**側の規律だけ:

- `wav_pcm16` の丸めが `floor(x + 0.5)`（JS の `Math.round` と同じ規則）であること —
  Python 組み込みの `round`（偶数丸め）へ滑ると、同じ波形から 1 LSB 違う WAV が出て
  「out / reference / official の聴き比べ」が実装差の混入した比較になる
- `clean_text_ranges` の畳み込みの後条件（昇順・隣接しない）と、境界のコードポイントが
  どちらの列に入るか — 出力はそのまま配布形の tokenizer 資産になる
- `tile_bert` の本数不一致が fail loudly すること
- `dump_metadata` が `demo` 欄の無い safetensors を受け取らないこと
"""

from __future__ import annotations

import itertools
import json
import struct

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from sbv2 import demo


def _wav_fields(blob: bytes) -> dict[str, int]:
    """WAV ヘッダの数値欄（`fmt ` チャンクの 16 バイト + `data` 長）。"""
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
        # RIFF のサイズ欄は「以降の全バイト」= 36 + データ長。
        assert struct.unpack("<I", blob[4:8])[0] == 36 + 16
        assert len(blob) == 44 + 16

    def test_the_rounding_is_floor_of_x_plus_half(self):
        """MUST: JS の `Math.round` と同じ規則（偶数丸めなら [0, 2, 0, -2] になる）。"""
        samples = np.array([0.5, 1.5, -0.5, -1.5], dtype=np.float64) / 32767.0

        pcm = np.frombuffer(demo.wav_pcm16(samples, 48000)[44:], dtype="<i2")

        assert list(pcm) == [1, 2, 0, -1]

    def test_it_clips_to_unity_without_using_the_lower_end(self):
        """クリップは ±1.0 で、下端 −32768 は使わない（正負で対称な振幅）。"""
        pcm = np.frombuffer(demo.wav_pcm16(np.array([2.0, -2.0]), 48000)[44:], dtype="<i2")

        assert list(pcm) == [32767, -32767]


def _in_ranges(ranges: list[list[int]], cp: int) -> bool:
    """コードポイントが閉区間の列に含まれるか（走査は台本ではなくテスト側が持つ）。"""
    return any(start <= cp <= end for start, end in ranges)


class TestCleanTextRanges:
    """判定の正本を Python に一本化した結果そのもの（TS へ Unicode 分類表を持ち込まない）。"""

    @pytest.fixture(scope="class")
    @staticmethod
    def ranges() -> dict[str, list[list[int]]]:
        return demo.clean_text_ranges()

    def test_both_lists_are_folded_ascending_and_never_adjacent(self, ranges):
        """畳み込みの後条件 — 隣接した 2 区間が残っていたら畳み込みが効いていない。"""
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
        """`\\t` `\\n` `\\r` は control 判定から**外して**空白側へ置く（参照実装の分岐）。"""
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
        with pytest.raises(AssertionError, match="word2ph 長"):
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

        with pytest.raises(ValueError, match=r"__metadata__\.demo が無い"):
            demo.dump_metadata(path)
