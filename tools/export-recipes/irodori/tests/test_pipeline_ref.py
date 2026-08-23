"""`irodori/pipeline_ref.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（`irodori/export.py` と同じ規律）。ここで固定するのは、**ホストが
写している 2 段**（S 決定と Euler の組み立て）の細部で、壊れると golden が静かに別物になる側:

- t スケジュールが上流の式（`(1 - linspace) * 0.999`）で、単調減少であること
- 銀行家丸めが「四捨五入」に化けていないこと（0.5 の扱い）
- S の clamp が下限 13 / 上限 750 で効くこと
- 区間マスクの割り方（self / text / speaker / caption の順とオフセット）と、uncond が
  **その区間だけ**を落とすこと
- CFG のスケール表が uncond 変種の綴りと 1 対 1 であること
- token 列の前処理が種別で分かれていること（text は `normalize_text` + strip・caption は
  strip のみ）と、上流突合へ渡す caption が**上流の入口から**作られること
- i4 席（`--dtype i4`）の**出荷バイトからの読み戻し**の門（provenance / 形 / 本数 / 席の効き）が
  1 つ残らず発火すること — ここが素通りすると「w8 の golden を w4 の golden と呼ぶ」事故が
  数値も形も合ったまま通る。i4 席は **I4 + I8 + F32 の混成**（block 内 linear が i4・block 外が
  i8 — 聴感裁定 2026-08-23）なので、i8 の逆変換と「効き門は i4 だけで数える」もここで固定する
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from irodori import export as ex
from irodori import pipeline_ref as ip
from irodori.distribution import CALIB_PROVENANCE_FILE
from karume.emit import pack_int4, unpack_int4
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)


class TestTSchedule:
    def test_it_starts_at_the_init_scale_and_ends_at_zero(self):
        schedule = ip.t_schedule(40)

        assert schedule.shape == (41,)
        assert float(schedule[0]) == pytest.approx(ip.INIT_SCALE)
        assert float(schedule[-1]) == 0.0

    def test_it_is_strictly_decreasing(self):
        """上流 `sample_euler_rf_cfg` が `t_schedule[:-1] > t_schedule[1:]` を要求する。"""
        schedule = ip.t_schedule(40)

        assert bool(torch.all(schedule[:-1] > schedule[1:]))

    def test_the_closed_form_is_within_one_ulp(self):
        """TS 側は閉形式で作る想定 — 差が 1 ulp 級であることを固定する。"""
        assert ip.closed_form_matches(ip.t_schedule(2), 2) == 0.0
        assert ip.closed_form_matches(ip.t_schedule(40), 40) < 1e-7

    def test_the_cfg_window_covers_the_first_half_of_the_steps(self):
        """CFG は `t ∈ [0.5, 1.0]` の step だけ — 40 step では前半 20 本。"""
        schedule = ip.t_schedule(40)
        inside = [
            index for index in range(40) if ip.CFG_MIN_T <= float(schedule[index]) <= ip.CFG_MAX_T
        ]

        assert inside == list(range(20))


class TestBankerRounding:
    """MUST: 「四捨五入」に化けさせない（TS の `Math.round` は 0.5 で常に上へ行く）。"""

    @pytest.mark.parametrize(
        ("frames", "want"),
        [(12.4, 12), (12.5, 12), (12.6, 13), (13.5, 14), (14.5, 14), (160.9389, 161)],
    )
    def test_half_way_values_round_to_even(self, frames, want):
        assert ip.banker_round_frames(frames) == want


class TestSequenceLength:
    def test_a_typical_prediction_is_used_as_is(self):
        # expm1 の逆（log1p）を入れて狙った frames を作る。
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
        """上流は `expm1(...).float().mean()`（候補数 > 1 のときの平均）。"""
        log_frames = torch.log1p(torch.tensor([100.0, 200.0]))

        steps, _meta = ip.sequence_length(log_frames, ex.CODEC_FRAME_RATE)

        assert steps == 150


CAPS = {"text": 4, "speaker": 5, "caption": 6}
USED = {"text": 2, "speaker": 3, "caption": 4}


class TestSegmentMasks:
    """MUST: 区間の割り方（順序とオフセット）を固定する — ずれても shape は合ってしまう。"""

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
        """綴りが割れると uncond の区間が黙って別の条件を落とす。"""
        assert tuple(ip.CFG_SCALES) == ex.DIT_UNCOND_VARIANTS


class TestRightPadIds:
    def test_it_pads_to_the_declared_length_and_marks_the_head(self):
        ids = torch.tensor([[1, 5, 6]], dtype=torch.int64)

        padded, mask = ip._right_pad_ids(ids, 6, 3)

        assert padded[0].tolist() == [1, 5, 6, 3, 3, 3]
        assert mask[0].tolist() == [True, True, True, False, False, False]

    def test_a_missing_sequence_becomes_all_pad_with_an_empty_mask(self):
        """caption 空の経路（上流 `caption_mask.zero_()`）。"""
        padded, mask = ip._right_pad_ids(None, 4, 3)

        assert padded[0].tolist() == [3, 3, 3, 3]
        assert not bool(mask.any())


class _RecordingTokenizer:
    """`tokenizers.Tokenizer` の最小の身代わり（**渡された文字列を記録する**）。

    id はコードポイントそのもの — 何が渡ったかだけを見たいので、語彙は持たない。
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> Any:
        assert add_special_tokens is False, "上流は特殊トークン無しで呼ぶ"
        self.seen.append(text)
        return SimpleNamespace(ids=[ord(char) for char in text])


class TestPackedIds:
    """MUST: 前処理は種別で違う（text = `normalize_text` + strip / caption = strip のみ）。"""

    #: 正規化に感受する caption（外側括弧・記号 — どちらも `normalize_text` の削除対象）。
    CAPTION = " 「①明るい声」 "

    @staticmethod
    def _bracket_stripper(body: str) -> str:
        """`normalize_text` の身代わり（外側括弧を剥がし ① を消すぶんだけを写す）。"""
        return body.strip().removeprefix("「").removesuffix("」").replace("①", "")

    def test_the_text_side_goes_through_normalization(self):
        tokenizer = _RecordingTokenizer()

        ip._packed_ids(tokenizer, self.CAPTION, 1, 64, self._bracket_stripper)

        assert tokenizer.seen == ["明るい声"]

    def test_the_caption_side_only_strips(self):
        """上流 `_synthesize` は caption に `str(...).strip()` しか掛けない。"""
        tokenizer = _RecordingTokenizer()

        ids = ip._packed_caption_ids(tokenizer, self.CAPTION, 1, 64)

        assert tokenizer.seen == ["「①明るい声」"]
        assert ids[0].tolist() == [1, *(ord(char) for char in "「①明るい声」")]

    def test_the_caption_body_budget_leaves_room_for_the_bos(self):
        ids = ip._packed_caption_ids(_RecordingTokenizer(), "あいうえお", 1, 3)

        assert ids[0].tolist() == [1, ord("あ"), ord("い")]

    def test_an_empty_caption_fails_loudly(self):
        """BOS だけの列を条件に載せると、caption 無しとも違う別の条件になる。"""
        with pytest.raises(SystemExit, match="strip 後の caption が空"):
            ip._packed_caption_ids(_RecordingTokenizer(), "  \n ", 1, 64)


class _FakeCaptionTokenizer:
    """上流 `PretrainedTextTokenizer.batch_encode` の**呼ばれ方**だけを写した身代わり。

    返す列は固定（先頭 2 本のマスクが立つ形）— ここで見たいのは「渡る文字列が strip 済みか」と
    「空 caption でマスクが BOS ごと全 0 になるか」の 2 点だけ。
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
        """上流 `_synthesize` の `caption_mask.zero_()` — BOS の列も落とす。"""
        fake = _FakeCaptionTokenizer()

        _ids, mask = ip.upstream_caption_condition(fake, "   ", 8)

        assert fake.seen == [""]
        assert not bool(mask.any())


class TestCaptionGoldenStability:
    def test_the_pipeline_captions_are_insensitive_to_normalization(self):
        """MUST: 既存 golden が動かないことの実測（動くなら再 emit が要る合図）。

        caption を strip-only へ直した波の前提そのもの。ここが落ちたら
        `outputs/series/irodori-*/pipeline/` を採り直さないと TS 側の統合門が割れる。
        """
        pytest.importorskip("irodori_tts")
        from irodori_tts.text_normalization import normalize_text

        for case in ip.PIPELINE_CASES:
            assert normalize_text(case.caption).strip() == case.caption.strip(), case.name


#: 合成コンテナの group 長（i4 は端数 group を作らない — ADR 0069 決定 2）。
GROUP = 32

#: block 内の席（i4 格納）と block 外の席（i8 格納 — 聴感裁定 2026-08-23 で i4 から外した）。
I4_KEY = "in_proj.weight"
SCALE_KEY = f"karume.scale.{I4_KEY}"
I8_KEY = "out_proj.weight"
I8_SCALE_KEY = f"karume.scale.{I8_KEY}"
#: 持ち上げ定数の席（`karume.convert` の綴り）— 上書き対象から外れることを見るために置く。
CONST_KEY = "const.0f0f0f0f0f0f0f0f"


class _DitWrapper(nn.Module):
    """`irodori.export.DitGraph` の身代わり（**所有パラメタの顔ぶれ**だけを写す）。

    i4 席 1 本（`in_proj.weight` — 量子化軸が g32 で割り切れる）と i8 席 1 本
    （`out_proj.weight` — block 外）と f32 席 4 本（bias / norm）。名前を実物と同じ綴りに
    するのは、読み戻しがラッパ内 FQN 空間で動くことをそのまま試すため。
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(2 * GROUP, 4)
        self.out_proj = nn.Linear(2 * GROUP, 4)
        self.out_norm = nn.LayerNorm(4)


def _raw(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().numpy().tobytes()


def _shipped(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`weight` の出荷形 `(packed 4bit, group scale, 出荷バイトから戻した f32)`。

    期待値は**書いたバイトから**戻す（`karume.emit` の格納時ビット一致門と同じ向き）。
    """
    scale = group_scale(weight, GROUP)
    packed = pack_int4(quantize_to_int4(weight, scale))
    return packed, scale, dequantize_int4(unpack_int4(packed, weight.shape), scale)


def _shipped_i8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """同・i8 席（`(int8, per-channel scale, 出荷バイトから戻した f32)`）。"""
    scale = channel_scale(weight, 0)
    quantized = quantize_to_int8(weight, scale)
    return quantized, scale, quantized.to(torch.float32) * scale


def _material(seed: int = 0) -> tuple[list[list[Any]], dict[str, Any], dict[str, torch.Tensor]]:
    """合成コンテナの素材 `(ヘッダ項目, IR の initializer 宣言, 期待値)`。

    門ごとに 1 箇所だけ壊せるよう、テスト側で書き換えてから {@link _write_series} へ渡す。
    """
    generator = torch.Generator().manual_seed(seed)
    packed, scale, restored = _shipped(torch.randn(4, 2 * GROUP, generator=generator))
    stored8, scale8, restored8 = _shipped_i8(torch.randn(4, 2 * GROUP, generator=generator))
    plain = {
        "in_proj.bias": torch.randn(4, generator=generator),
        "out_proj.bias": torch.randn(4, generator=generator),
        "out_norm.weight": torch.randn(4, generator=generator),
        "out_norm.bias": torch.randn(4, generator=generator),
    }
    entries: list[list[Any]] = [
        [I4_KEY, "I4", [4, 2 * GROUP], _raw(packed)],
        [SCALE_KEY, "F32", list(scale.shape), _raw(scale)],
        [I8_KEY, "I8", [4, 2 * GROUP], _raw(stored8)],
        [I8_SCALE_KEY, "F32", list(scale8.shape), _raw(scale8)],
        *([key, "F32", [4], _raw(value)] for key, value in plain.items()),
        [CONST_KEY, "F32", [2], _raw(torch.zeros(2))],
    ]
    initializers: dict[str, Any] = {
        "p_in_proj_weight": {
            "tensor": I4_KEY,
            "storage": {"dtype": "i4", "scale": SCALE_KEY, "group_size": GROUP},
        },
        "p_out_proj_weight": {
            "tensor": I8_KEY,
            "storage": {"dtype": "i8", "scale": I8_SCALE_KEY},
        },
        **{
            f"p_{key.replace('.', '_')}": {"tensor": key, "storage": {"dtype": "f32"}}
            for key in plain
        },
        "const_0f0f0f0f0f0f0f0f": {"tensor": CONST_KEY, "storage": {"dtype": "f32"}},
    }
    return entries, initializers, {I4_KEY: restored, I8_KEY: restored8, **plain}


def _write_series(
    directory: Path,
    entries: list[list[Any]],
    initializers: dict[str, Any],
    method: str | None = "gptq",
) -> Path:
    """合成の i4 系列（コンテナ 1 本 + 校正記録）を書く（`method=None` で記録を落とす）。

    `safetensors` のライタは `I4` を知らないので、ヘッダ JSON とデータ節を素で組む
    （読み手も自前 — `irodori.pipeline_ref._read_stored` の MUST と対）。
    """
    directory.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {
        "__metadata__": {"karume_ir": json.dumps({"initializers": initializers})}
    }
    blob = bytearray()
    for key, dtype, shape, payload in entries:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(blob), len(blob) + len(payload)],
        }
        blob += payload
    raw = json.dumps(header).encode()
    (directory / ex.MODEL_FILE).write_bytes(
        len(raw).to_bytes(8, "little") + raw + bytes(blob),
    )
    if method is not None:
        (directory / CALIB_PROVENANCE_FILE).write_text(
            json.dumps({"method": method, "grid": "rtn", "group_size": GROUP, "cases": 4}),
            encoding="utf-8",
        )
    return directory


class TestRestoreDitFromI4Series:
    """MUST: golden は出荷バイトから焼く（校正を 2 度走らせて一致に賭けない）。"""

    def test_every_owned_parameter_comes_from_the_shipped_bytes(self, tmp_path):
        module = _DitWrapper()
        entries, initializers, shipped = _material()
        series = _write_series(tmp_path / "dit", entries, initializers)

        record = ip.restore_dit_from_i4_series(module, series)

        assert (record.int4, record.int8, record.plain, record.changed) == (1, 1, 4, 1)
        assert record.calib["method"] == "gptq"
        owned = dict(module.named_parameters())
        # 持ち上げ定数（`const.*`）は席から外れる — 在っても上書き対象にならない。
        assert set(owned) == set(shipped)
        for key, value in shipped.items():
            assert torch.equal(owned[key].detach(), value), key

    def test_an_uncalibrated_series_is_refused(self, tmp_path):
        """`--no-calib` の生成物は格納形が同じ = 資産から判別できない（音だけが劣化する）。"""
        entries, initializers, _shipped_values = _material()
        series = _write_series(tmp_path / "dit", entries, initializers, method="rtn")

        with pytest.raises(SystemExit, match="配布して良い丸め方式"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_missing_provenance_record_is_refused(self, tmp_path):
        entries, initializers, _shipped_values = _material()
        series = _write_series(tmp_path / "dit", entries, initializers, method=None)

        with pytest.raises(SystemExit, match="校正条件の記録が無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_tensor_the_module_does_not_own_fails_loudly(self, tmp_path):
        """コンテナに在るのにモジュールに無い席（持ち上げ定数以外）は即エラー。"""
        entries, initializers, _shipped_values = _material()
        entries.append(["extra.weight", "F32", [4], _raw(torch.zeros(4))])
        initializers["p_extra_weight"] = {"tensor": "extra.weight", "storage": {"dtype": "f32"}}
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match="モジュールに無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_parameter_missing_from_the_container_fails_loudly(self, tmp_path):
        """逆向き（モジュールに在るのにコンテナに無い）— 上書きされない席が残る。"""
        entries, initializers, _shipped_values = _material()
        del initializers["p_out_norm_bias"]
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match="コンテナに無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_an_i4_tensor_without_a_scale_declaration_fails_loudly(self, tmp_path):
        entries, initializers, _shipped_values = _material()
        del initializers["p_in_proj_weight"]["storage"]["scale"]
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match="scale の宣言が無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_scale_missing_from_the_container_fails_loudly(self, tmp_path):
        entries, initializers, _shipped_values = _material()
        entries = [entry for entry in entries if entry[0] != SCALE_KEY]
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match=f"'{SCALE_KEY}' がコンテナに無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_an_i4_tensor_outside_the_declared_set_fails_loudly(self, tmp_path):
        """本数門: I4 で書かれているのに f32 として読み戻す宣言、を通さない。"""
        entries, initializers, _shipped_values = _material()
        for entry in entries:
            if entry[0] == "in_proj.bias":
                entry[1] = "I4"
                entry[2] = [8]
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match="I4 格納のテンソル 2 本"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_container_without_any_i4_tensor_fails_loudly(self, tmp_path):
        """i4 系列でないディレクトリ（f16 / i8 の系列）を指した形。"""
        entries, initializers, _shipped_values = _material()
        weight = torch.zeros(4, 2 * GROUP)
        entries[0] = [I4_KEY, "F32", [4, 2 * GROUP], _raw(weight)]
        initializers["p_in_proj_weight"]["storage"] = {"dtype": "f32"}
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match="I4 格納のテンソルが 1 本も無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_restore_that_changes_nothing_fails_loudly(self, tmp_path):
        """席の効き門: 段 1（i8 丸め）の値と全て同じなら、読み戻しが効いていない。"""
        module = _DitWrapper()
        entries, initializers, shipped = _material()
        with torch.no_grad():
            module.in_proj.weight.copy_(shipped[I4_KEY])
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match="i4 の読み戻しが効いていない"):
            ip.restore_dit_from_i4_series(module, series)

    def test_a_shape_mismatch_fails_loudly(self, tmp_path):
        entries, initializers, _shipped_values = _material()
        for entry in entries:
            if entry[0] == "out_norm.bias":
                entry[2] = [2]
                entry[3] = _raw(torch.zeros(2))
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match="の形が コンテナ"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_declaration_that_disagrees_with_the_header_fails_loudly(self, tmp_path):
        """宣言 f32 / 実体 F16 — 宣言と実体は 2 面で突き合わせる。"""
        entries, initializers, _shipped_values = _material()
        for entry in entries:
            if entry[0] == "out_norm.weight":
                entry[1] = "F16"
                entry[3] = _raw(torch.zeros(4, dtype=torch.float16))
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match="ヘッダは 'F16'"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_storage_dtype_outside_the_seat_fails_loudly(self, tmp_path):
        """w4 席の dit に f16 は並ばない（並んだら混成が想定と違う形で出荷されている）。

        受理するのは f32 / i8 / i4 の 3 つだけ（{@link irodori.pipeline_ref._RESTORE_STORAGE}）。
        """
        entries, initializers, _shipped_values = _material()
        initializers["p_out_norm_weight"]["storage"] = {"dtype": "f16"}
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match="は読み戻せない"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_an_i8_tensor_without_a_scale_declaration_fails_loudly(self, tmp_path):
        """i8 も逆変換に per-channel scale が要る（宣言が無ければ読み戻せない）。"""
        entries, initializers, _shipped_values = _material()
        del initializers["p_out_proj_weight"]["storage"]["scale"]
        series = _write_series(tmp_path / "dit", entries, initializers)

        with pytest.raises(SystemExit, match=r"i8 格納の .* に scale の宣言が無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_the_i8_seat_does_not_count_as_the_i4_seat_working(self, tmp_path):
        """MUST: 席の効き門は **i4 だけ**で数える。

        block 外の i8 は段 1（i8 丸め）と同じ格子なので値が動かないのが正常だが、実行のたびに
        必ず一致するとは限らない（段 1 は `dit` 丸ごと・段 2 はコンテナのバイト）。そこを効き門に
        数えると、i4 が 1 本も効いていない事故を i8 の差が埋め合わせて隠す。
        """
        module = _DitWrapper()
        entries, initializers, shipped = _material()
        with torch.no_grad():
            module.in_proj.weight.copy_(shipped[I4_KEY])
        series = _write_series(tmp_path / "dit", entries, initializers)
        assert not torch.equal(module.out_proj.weight.detach(), shipped[I8_KEY]), (
            "i8 席が段 1 と同値では、効き門が i4 だけを数えていることを試せない"
        )

        with pytest.raises(SystemExit, match="i4 の読み戻しが効いていない"):
            ip.restore_dit_from_i4_series(module, series)

    def test_a_missing_container_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="i4 系列のコンテナが無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), tmp_path / "dit")


class TestCli:
    def test_the_default_out_dir_is_derived_from_the_weight_directory(self, tmp_path):
        out = ip.default_out_dir(tmp_path / "v4-small")

        assert out.name == "pipeline"
        assert out.parent.name == "irodori-v4-small"

    def test_each_dtype_writes_into_its_own_series(self, tmp_path):
        """MUST: golden も系列ごと（f32 の golden で f16 資産を突き合わせると量子化誤差が
        tolerance に混ざり、緑のまま検出力だけが落ちる）。"""
        dirs = {
            dtype: ip.default_out_dir(tmp_path / "v4-small", dtype) for dtype in ex.WEIGHT_DTYPES
        }

        assert len(set(dirs.values())) == len(dirs)
        assert dirs["f16"].parent.name == "irodori-v4-small-f16"
        # 系列 root の綴りは export 台本と共有する 1 語（写経しない）。
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
        """割り切れないと上流 `patch_sequence_with_mask` が端を捨て、保存した入力とずれる。"""
        for case in ip.PIPELINE_CASES:
            if case.reference is not None:
                assert case.reference.frames % 4 == 0
