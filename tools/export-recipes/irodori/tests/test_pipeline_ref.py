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
  1 つ残らず発火すること — ここが素通りすると「i8 の golden を i8+dit4 の golden と呼ぶ」事故が
  数値も形も合ったまま通る。i4 席は **I4 + I8 + F32 の混成**（block 内の adaLN 以外が i4・
  adaLN と block 外が i8 — 聴感裁定 2026-08-23）なので、i8 の逆変換と「効き門は i4 だけで
  数える」もここで固定する
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch
from torch import nn

from irodori import export as ex
from irodori import pipeline_ref as ip
from irodori.distribution import CALIB_PROVENANCE_FILE, irodori_calib_floor
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

#: i4 格納の席と i8 格納の席（聴感裁定 2026-08-23 で block 外 5 本と adaLN 144 本を i4 から
#: 外した）。読み戻しはコンテナの宣言駆動なので合成コンテナは各 1 本で足りるが、実重みの
#: 期待値は `meta.json` の `i4Source` に出る **int4Tensors 168 / int8Tensors 149**。
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


def _shipped(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`weight` の i4 出荷形 `(group scale, 出荷バイトから戻した f32)`。

    期待値は**書いたバイトから**戻す（`karume.emit` の格納時ビット一致門と同じ向き）。
    """
    scale = group_scale(weight, GROUP)
    return scale, dequantize_int4(quantize_to_int4(weight, scale), scale)


def _shipped_i8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """同・i8 席（`(per-channel scale, 出荷バイトから戻した f32)`）。"""
    scale = channel_scale(weight, 0)
    return scale, quantize_to_int8(weight, scale).to(torch.float32) * scale


def _init_name(key: str) -> str:
    """テンソルキー → IR v1 の initializer 名（torch.export が FQN を正規化した綴り）。"""
    return "const_0f0f0f0f0f0f0f0f" if key == CONST_KEY else f"p_{key.replace('.', '_')}"


def _graph(tensors: Mapping[str, torch.Tensor]) -> IrGraph:
    """合成 `dit` のグラフ（i4 / i8 の席が **linear の重みスロット**で消費される形）。

    圧縮格納の適格判定は「重みスロットで消費されているか」なので、宣言だけ並べた器では
    `--dtype i4` が 1 本も効かない（`karume.emit` が fail loudly する）。
    """
    initializers = {
        _init_name(key): IrInitializer(tensor=key, storage=IrStorage(dtype="f32"))
        for key in tensors
    }
    values = {
        _init_name(key): IrValue(dtype="f32", shape=list(tensor.shape))
        for key, tensor in tensors.items()
    }
    values.update(
        {name: IrValue(dtype="f32", shape=[1, 4]) for name in ("h0", "h1", "h2", "h3", "out")}
    )
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
        IrNode(op="add", ins=["h0", _init_name("out_norm.weight")], outs=["h2"], attrs={}),
        IrNode(op="add", ins=["h2", _init_name("out_norm.bias")], outs=["h3"], attrs={}),
        IrNode(op="add", ins=["h3", _init_name(CONST_KEY)], outs=["out"], attrs={}),
    ]
    return IrGraph(
        inputs=[IrInput(name="x", dtype="f32", shape=[1, 2 * GROUP])],
        outputs=["out", "h1"],
        initializers=initializers,
        values=values,
        nodes=nodes,
    )


class _Material(NamedTuple):
    """合成コンテナの素材（門ごとに 1 箇所だけ壊してから {@link _write_series} へ渡す）。"""

    tensors: dict[str, torch.Tensor]
    scales: dict[str, torch.Tensor]
    overrides: dict[str, str]
    #: モジュールの所有パラメタ名 → 読み戻しで入るはずの f32 実体。
    shipped: dict[str, torch.Tensor]


def _material(seed: int = 0) -> _Material:
    """i4 1 本 + i8 1 本 + f32 4 本 + 持ち上げ定数 1 本の素材。"""
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
) -> Path:
    """合成の i4 系列（`krm` の part 列 + 校正記録）を書く（`method=None` で記録を落とす）。

    書き出しは実物と同じ 1 本道（`karume.emit.stored_model` →
    `karume.publish.publish_container`）を通す — 規則の写しを持つと、規則が動いた日に
    フィクスチャだけが古びて「テストは緑・実物だけ落ちる」になる。

    `budget` は校正予算の 2 欄（既定は配布の下限ちょうど = 出荷済み系列と同じ形）。smoke 予算で
    焼いた系列から golden を焼こうとする軸をここで振る。`block_bytes` は piece 分割を踏ませる
    ための寸法の差し込み。
    """
    directory.mkdir(parents=True, exist_ok=True)
    stored = stored_model(
        _graph(material.tensors),
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


class TestRestoreDitFromI4Series:
    """MUST: golden は出荷バイトから焼く（校正を 2 度走らせて一致に賭けない）。"""

    def test_every_owned_parameter_comes_from_the_shipped_bytes(self, tmp_path):
        module = _DitWrapper()
        material = _material()
        series = _write_series(tmp_path / "dit", material)

        record = ip.restore_dit_from_i4_series(module, series)

        assert (record.int4, record.int8, record.plain, record.changed) == (1, 1, 4, 1)
        assert record.calib["method"] == "gptq"
        owned = dict(module.named_parameters())
        # 持ち上げ定数（`const.*`）は席から外れる — 在っても上書き対象にならない。
        assert set(owned) == set(material.shipped)
        for key, value in material.shipped.items():
            assert torch.equal(owned[key].detach(), value), key

    def test_an_uncalibrated_series_is_refused(self, tmp_path):
        """`--no-calib` の生成物は格納形が同じ = 資産から判別できない（音だけが劣化する）。"""
        series = _write_series(tmp_path / "dit", _material(), method="rtn")

        with pytest.raises(SystemExit, match="配布して良い丸め方式"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_missing_provenance_record_is_refused(self, tmp_path):
        series = _write_series(tmp_path / "dit", _material(), method=None)

        with pytest.raises(SystemExit, match="校正条件の記録が無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_smoke_budget_series_is_refused(self, tmp_path):
        """`--calib-steps 1` は `method` を `gptq` のまま残す — 予算欄まで見ないと通る。

        golden だけが smoke 用の丸めで焼かれると、配布資産との突合が「両辺が違う重み」の
        まま緑になる（格納形も本数も 1 つも動かない）。
        """
        series = _write_series(
            tmp_path / "dit", _material(), budget={**irodori_calib_floor(), "steps": 1}
        )

        with pytest.raises(SystemExit, match="校正予算 'steps' が配布の下限を下回る"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_tensor_the_module_does_not_own_fails_loudly(self, tmp_path):
        """コンテナに在るのにモジュールに無い席（持ち上げ定数以外）は即エラー。"""
        material = _material()
        material.tensors["extra.weight"] = torch.zeros(1, 4)
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="モジュールに無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_parameter_missing_from_the_container_fails_loudly(self, tmp_path):
        """逆向き（モジュールに在るのにコンテナに無い）— 上書きされない席が残る。"""
        material = _material()
        del material.tensors["out_norm.bias"]
        del material.shipped["out_norm.bias"]
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="コンテナに無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_container_without_any_i4_tensor_fails_loudly(self, tmp_path):
        """i4 系列でないディレクトリ（f16 / i8 の系列）を指した形。"""
        material = _material()
        material.overrides[I4_KEY] = "i8"
        material.scales[I4_KEY] = channel_scale(material.tensors[I4_KEY], 0)
        material.tensors[I4_KEY] = (
            quantize_to_int8(material.tensors[I4_KEY], material.scales[I4_KEY]).to(torch.float32)
            * material.scales[I4_KEY]
        )
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="i4 格納のテンソルが 1 本も無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_restore_that_changes_nothing_fails_loudly(self, tmp_path):
        """席の効き門: 段 1（i8 丸め）の値と全て同じなら、読み戻しが効いていない。"""
        module = _DitWrapper()
        material = _material()
        with torch.no_grad():
            module.in_proj.weight.copy_(material.shipped[I4_KEY])
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="i4 の読み戻しが効いていない"):
            ip.restore_dit_from_i4_series(module, series)

    def test_a_shape_mismatch_fails_loudly(self, tmp_path):
        material = _material()
        material.tensors["out_norm.bias"] = torch.zeros(2)
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="の形が コンテナ"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_a_storage_outside_the_seat_fails_loudly(self, tmp_path):
        """i8+dit4 席の dit に f16 は並ばない（並んだら混成が想定と違う形で出荷されている）。

        受理するのは f32 / i8 / i4 の 3 つだけ（{@link irodori.pipeline_ref._RESTORE_STORAGE}）。
        """
        material = _material()
        material.overrides[I8_KEY] = "f16"
        del material.scales[I8_KEY]
        material.tensors[I8_KEY] = material.tensors[I8_KEY].to(torch.float16).to(torch.float32)
        series = _write_series(tmp_path / "dit", material)

        with pytest.raises(SystemExit, match="は読み戻せない"):
            ip.restore_dit_from_i4_series(_DitWrapper(), series)

    def test_the_i8_seat_does_not_count_as_the_i4_seat_working(self, tmp_path):
        """MUST: 席の効き門は **i4 だけ**で数える。

        block 外の i8 は段 1（i8 丸め）と同じ格子なので値が動かないのが正常だが、実行のたびに
        必ず一致するとは限らない（段 1 は `dit` 丸ごと・段 2 はコンテナのバイト）。そこを効き門に
        数えると、i4 が 1 本も効いていない事故を i8 の差が埋め合わせて隠す。
        """
        module = _DitWrapper()
        material = _material()
        with torch.no_grad():
            module.in_proj.weight.copy_(material.shipped[I4_KEY])
        series = _write_series(tmp_path / "dit", material)
        assert not torch.equal(module.out_proj.weight.detach(), material.shipped[I8_KEY]), (
            "i8 席が段 1 と同値では、効き門が i4 だけを数えていることを試せない"
        )

        with pytest.raises(SystemExit, match="i4 の読み戻しが効いていない"):
            ip.restore_dit_from_i4_series(module, series)

    def test_a_missing_container_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="i4 系列のコンテナが無い"):
            ip.restore_dit_from_i4_series(_DitWrapper(), tmp_path / "dit")


class TestRestoreDitFromASplitSeries:
    """piece へ割れたテンソル（container-v1 §4.2）を含む系列でも読み戻しは同じ結果になる。

    畳まない読み手だと i4 席が piece のまま数えられ、実体が途中で切れたまま上書きされる
    （= この 1 本が畳みの有無を決める）。
    """

    def test_every_owned_parameter_still_comes_from_the_shipped_bytes(self, tmp_path):
        module = _DitWrapper()
        material = _material()
        # block 上限を 1 行（i4 は 2 要素 1 バイトなので 32 バイト）まで下げて、4 行の重みを
        # piece 列へ割らせる。
        series = _write_series(tmp_path / "dit", material, block_bytes=64)
        # 前提の観測点 — piece が 1 本も無ければこのテストは空振りになる。
        verified = verify_container(container_parts(series / ex.MODEL_FILE))
        split = [
            name
            for bound in verified.graphs.values()
            for name, supply in bound.supplies.items()
            if len(supply.blocks) > 1
        ]
        assert split

        record = ip.restore_dit_from_i4_series(module, series)

        assert (record.int4, record.int8, record.plain, record.changed) == (1, 1, 4, 1)
        owned = dict(module.named_parameters())
        for key, value in material.shipped.items():
            assert torch.equal(owned[key].detach(), value), key


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


class TestEulerReferenceSensitivity:
    """2 段目の合否関数（増幅率で正規化した実装差の判定 — 2026-09-01 追記の型）。"""

    def test_the_measured_v41_f16_pair_passes(self):
        """導入の動機になった実測ペア（worst 1.33e-2 / amp 37,107）は 2 段目を通る。"""
        assert ip.euler_reference_within_sensitivity(1.33e-2, 37_107.0)

    def test_a_low_amplification_trajectory_keeps_the_tight_bound(self):
        """増幅がおとなしい軌道（v4-small 級 amp≈353）では 1e-2 の差は退行として落ちる。"""
        assert not ip.euler_reference_within_sensitivity(1.0e-2, 353.0)

    def test_the_absolute_ceiling_stops_gross_breakage(self):
        """式の取り違え級（O(1)）は増幅率がいくら大きくても落ちる。"""
        assert not ip.euler_reference_within_sensitivity(0.5, 1.0e9)

    def test_the_ceiling_binds_even_when_normalized_noise_is_small(self):
        """正規化では通る大きさでも絶対上限が先に効く。"""
        amp = 1.0e5  # amp × EULER_NOISE_PER_AMP = 0.5 > 上限 5e-2
        worst = 1.0e-1
        assert worst <= amp * ip.EULER_NOISE_PER_AMP
        assert not ip.euler_reference_within_sensitivity(worst, amp)
