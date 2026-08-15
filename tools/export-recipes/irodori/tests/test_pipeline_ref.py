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
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from irodori import export as ex
from irodori import pipeline_ref as ip


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
