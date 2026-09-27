"""`irodori/export.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（既存 `deberta/export.py` / `embeddinggemma/export.py` のテストと同じ
規律）。ここで固定するのは、壊れると**偽 PASS** になる側の規律だけ:

- 参照（golden の期待値）は**パッチ前**にしか採れない（採れてしまうと同値検証が恒真化する）
- 静的方式（実行時 attention_mask 非対応）の実測が、方式が崩れたときに実際に落ちる
- text / caption projector の取り違え（同じ重みを 2 回読む）が `_sanity` で落ちる
- `_write_io` が IR の入力名と食い違う io を書かない / 出力の本数がずれた io を書かない
- `caption-proj` の第 2 出力が `caption_norm` を掛けた系列であり、第 1 出力は素の projector 出力
  のまま（`text-proj` と同じ式）であること
- `text_norm` / `caption_norm` の取り違え（同じ重みを 2 回読む）が `_norm_divergence` で落ちる
- 参照なし（マスク全 0）の speaker 出力が**厳密に 0** であることの実測が、0 でなければ落ちる
- 実 latent 由来の speaker ケースが、資産の欠け・形の食い違いで**合成に化けずに**落ちる
- `duration` の `aux_features` 非依存の実測が、依存していたら落ちる
- `dit` の cond / uncond 3 変種が**互いに違う**ことの実測が、同じなら落ちる
- 条件 state の右 pad が宣言長を超えたら落ちる
- DiT を `dit-context` / `dit` に割った対（ADR 0114）が、上流 `forward_with_encoded_conditions`
  と**ビット一致**し、2 本の所有が「条件側の射影 / 残り」に割れ、IR の境界名が契約の綴りになる
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

#: `_static_scheme_evidence` / `build_cases` が読む config の最小形。
TEXT_CONFIG = {"pad_token_id": 3, "bos_token_id": 1}
MODEL_CONFIG = {"max_text_len": 16, "max_caption_len": 32}

CASES = (
    ("short", "text", torch.tensor([[1, 5, 6]], dtype=torch.int64)),
    ("cap", "caption", torch.tensor([[1, 7, 8, 9]], dtype=torch.int64)),
)


class MaskRespectingBackbone(nn.Module):
    """pad を**正しく無視する** backbone の代役（静的方式が成立する側）。

    出力は「マスクされた位置を 0 にした埋め込み」だけで決まるので、pad を足しても先頭 T 行は
    変わらない — 実物の双方向マスクが持つ性質と同じ。
    """

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4)
        return hidden * mask.unsqueeze(-1).to(torch.float32)


class PadLeakingBackbone(nn.Module):
    """pad の**本数が出力に漏れる** backbone の代役（静的方式が成立しない側の故障注入）。"""

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        base = MaskRespectingBackbone()(input_ids, mask)
        return base + float(input_ids.shape[1])


class ScalingNorm(nn.Module):
    """`caption_norm` の代役（定数倍 — 掛かったかどうかが値で判る）。"""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((4,), scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class DoublingProjector(nn.Module):
    """`_pristine_outputs` が呼ぶ実 projector の代役（`(backbone, ids, mask)` を受ける形）。"""

    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, backbone: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return backbone(ids, mask) * self.gain


class TestPristineReferenceOrdering:
    """MUST: 参照はパッチ前にしか採れない（採れると同値検証が恒真化して偽 PASS になる）。"""

    def test_taking_a_reference_after_patching_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", True)

        with pytest.raises(AssertionError, match="パッチ適用後に参照を採ろうとした"):
            ir._pristine_outputs(MaskRespectingBackbone(), {}, ScalingNorm(2.0), CASES)

    def test_taking_a_reference_before_patching_is_allowed(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)

        outputs = ir._pristine_outputs(MaskRespectingBackbone(), {}, ScalingNorm(2.0), CASES)

        assert set(outputs[ir.TARGET_BACKBONE]) == {"short", "cap"}
        assert tuple(outputs[ir.TARGET_BACKBONE]["short"][0].shape) == (1, 3, 4)

    def test_only_the_caption_projector_gets_a_second_output(self, monkeypatch):
        """MUST: 第 2 出力は caption 側だけ（text 側に生えたら `duration` の鎖が変わる）。"""
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
        # 第 2 出力 = norm(第 1 出力)。第 1 出力そのものは norm 前のまま。
        assert torch.equal(caption[1], caption[0] * 2.0)


def _backbone_first(backbone: nn.Module, cases) -> dict[str, torch.Tensor]:
    """`_pristine_outputs` の backbone 側から第 1 出力だけを取り出す（鎖の下流が食う形）。"""
    return ir._first(
        ir._pristine_outputs(backbone, {}, ScalingNorm(2.0), cases)[ir.TARGET_BACKBONE]
    )


class TestStaticSchemeEvidence:
    """静的方式（右詰め pad をホストで消す）の実測が、恒真でないことを確かめる。"""

    def test_a_mask_respecting_backbone_measures_zero(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        backbone = MaskRespectingBackbone()
        pristine = _backbone_first(backbone, CASES)

        evidence = ir._static_scheme_evidence(backbone, TEXT_CONFIG, MODEL_CONFIG, CASES, pristine)

        assert evidence == {"short": 0.0, "cap": 0.0}

    def test_a_pad_leaking_backbone_fails_loudly(self, monkeypatch):
        """MUST: 方式が崩れていたら落ちる — 回避せずに止めるための門。"""
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        backbone = PadLeakingBackbone()
        pristine = _backbone_first(backbone, CASES)

        with pytest.raises(AssertionError, match="静的方式"):
            ir._static_scheme_evidence(backbone, TEXT_CONFIG, MODEL_CONFIG, CASES, pristine)

    def test_a_case_longer_than_its_family_cap_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        long_case = (("long", "text", torch.arange(1, 40, dtype=torch.int64).unsqueeze(0)),)
        backbone = MaskRespectingBackbone()
        pristine = _backbone_first(backbone, long_case)

        with pytest.raises(SystemExit, match="上限"):
            ir._static_scheme_evidence(backbone, TEXT_CONFIG, MODEL_CONFIG, long_case, pristine)


class MaskZeroingEncoder(nn.Module):
    """マスクされた位置を**厳密に 0** にする参照エンコーダの代役（実物と同じ性質）。"""

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = latent.sum(dim=-1, keepdim=True).expand(-1, -1, 3) + 1.0
        return hidden * mask.unsqueeze(-1).to(torch.float32)


class MaskLeakingEncoder(nn.Module):
    """マスク全 0 でも非ゼロを返す代役（故障注入 — ホストのゼロ供給が成立しない側）。"""

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return MaskZeroingEncoder()(latent, mask) + 1e-3


class TestNoReferenceEvidence:
    """MUST: 「参照なしは 0」を主張のままにしない（ホストのゼロ供給の唯一の根拠）。"""

    def test_a_mask_zeroing_encoder_measures_zero(self):
        assert ir._no_reference_evidence(MaskZeroingEncoder(), 4, 4) == 0.0

    def test_a_leaking_encoder_fails_loudly(self):
        with pytest.raises(AssertionError, match="参照なし"):
            ir._no_reference_evidence(MaskLeakingEncoder(), 4, 4)


class AuxReadingPredictor(nn.Module):
    """`aux_features` を読む duration の代役（故障注入）。"""

    def forward(self, text_state: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return text_state.sum(dim=(1, 2)) + kwargs["aux_features"].sum(dim=1)


class AuxIgnoringPredictor(nn.Module):
    """`aux_features` を読まない duration の代役（token-sum 形と同じ性質）。"""

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
    """MUST: `aux_features` をグラフ入力から落とす根拠を実測で持つ。"""

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

        with pytest.raises(AssertionError, match="パッチ適用後に参照を採ろうとした"):
            ir._pristine_duration_outputs(AuxIgnoringPredictor(), DURATION_REFERENCE, 4)


class TestSpeakerCases:
    def test_the_declared_lengths_are_deterministic_and_in_range(self):
        first = ir.build_speaker_cases(4, 750)
        second = ir.build_speaker_cases(4, 750)

        assert [name for name, _ in first] == [name for name, _, _ in ir.SPEAKER_CASES]
        for (_name, lhs), (_same, rhs) in zip(first, second, strict=True):
            assert torch.equal(lhs, rhs)

    def test_a_case_over_the_symbolic_cap_fails_loudly(self):
        with pytest.raises(SystemExit, match="記号次元の範囲"):
            ir.build_speaker_cases(4, 8)


def _fake_patch(seq, mask, patch_size):
    """`patch_sequence_with_mask` の代役（端を捨てて `patch_size` 本ずつ束ねる）。

    実物は上流実装から注入される（`IrodoriSource.patch_sequence_with_mask`）ので、ここで
    見るのは **`build_real_speaker_cases` 自身の振る舞い**だけ — 束ね方の正しさは上流の責任で、
    代役で写しても意味が無い。
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
    """MUST: 実 latent 由来のケースは**合成で代替しない**（tolerance の根拠が値域に立つ）。"""

    def test_the_cases_follow_the_table_and_share_one_prefix(self, tmp_path):
        # 190 フレーム（参照音声 7.6 秒）→ patch 4 で 47 行、という実資産と同じ形。
        latent = torch.arange(190 * 32, dtype=torch.float32).reshape(1, 190, 32)
        _write_reference_latent(tmp_path, latent)

        cases = ir.build_real_speaker_cases(_fake_patch, 32, 4, 128, 750, latent_dir=tmp_path)

        assert [name for name, _ in cases] == [name for name, _ in ir.SPEAKER_REAL_CASES]
        by_name = dict(cases)
        assert by_name["ref-real-full"].shape == (1, 47, 128)
        assert by_name["ref-real-short"].shape == (1, 6, 128)
        # 短尺は全長の**先頭**を切り出したもの（別の乱数や別の区間に化けていない）。
        assert torch.equal(by_name["ref-real-short"], by_name["ref-real-full"][:, :6])

    def test_a_missing_latent_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="実 latent が無い"):
            ir.build_real_speaker_cases(_fake_patch, 32, 4, 128, 750, latent_dir=tmp_path)

    def test_a_latent_of_the_wrong_width_fails_loudly(self, tmp_path):
        _write_reference_latent(tmp_path, torch.zeros((1, 190, 16)))

        with pytest.raises(SystemExit, match=r"\[1,S,32\] でない"):
            ir.build_real_speaker_cases(_fake_patch, 32, 4, 128, 750, latent_dir=tmp_path)

    def test_a_patch_size_that_misses_the_speaker_input_width_fails_loudly(self, tmp_path):
        _write_reference_latent(tmp_path, torch.zeros((1, 190, 32)))

        # patch 2 なら束ねた幅は 64 で、speaker グラフの入力 128 と食い違う。
        with pytest.raises(SystemExit, match="speaker の入力次元"):
            ir.build_real_speaker_cases(_fake_patch, 32, 2, 128, 750, latent_dir=tmp_path)

    def test_a_latent_shorter_than_the_table_fails_loudly(self, tmp_path):
        # 20 フレーム → patch 後 5 行で、表の短尺 6 行に足りない。
        _write_reference_latent(tmp_path, torch.zeros((1, 20, 32)))

        with pytest.raises(SystemExit, match="実 latent の patch 後の長さ"):
            ir.build_real_speaker_cases(_fake_patch, 32, 4, 128, 750, latent_dir=tmp_path)


def _dit_pristine(distinct: bool) -> dict[str, torch.Tensor]:
    """全 `DIT_CASES` ぶんのダミー出力（`distinct` なら 1 本ずつ違う値）。

    グループの取り方は `_dit_uncond_divergence` 自身に任せる（テスト側で写すと、
    グループ分けの誤りが両側で同じように壊れて素通りする）。
    """
    return {
        name: torch.full((1, 2, 3), float(index) if distinct else 0.0)
        for index, (name, *_rest) in enumerate(ir.DIT_CASES)
    }


class TestDitUncondDivergence:
    """MUST: 「マスクが効いている」を主張のままにしない（uncond をマスクで表す根拠）。"""

    def test_distinct_outputs_are_reported_pairwise(self):
        pairs = ir._dit_uncond_divergence(_dit_pristine(distinct=True))

        # cond + uncond 3 変種の総当たり = 6 組。
        assert len(pairs) == 6
        assert min(pairs.values()) >= ir.DIT_UNCOND_DIVERGENCE_MIN

    def test_identical_outputs_fail_loudly(self):
        with pytest.raises(AssertionError, match="マスクの区間割り"):
            ir._dit_uncond_divergence(_dit_pristine(distinct=False))


class TestPristineDitReferenceOrdering:
    def test_taking_a_reference_after_patching_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", True)

        with pytest.raises(AssertionError, match="パッチ適用後に参照を採ろうとした"):
            ir._pristine_dit_outputs(AuxIgnoringPredictor(), {})


class TestRightPad:
    """条件 state の右 pad（ADR 0047 のホスト残置）。"""

    def test_pads_with_zeros_and_keeps_the_head(self):
        padded = ir._right_pad(torch.ones(1, 2, 3), 5, "テスト")

        assert tuple(padded.shape) == (1, 5, 3)
        assert torch.equal(padded[:, :2], torch.ones(1, 2, 3))
        assert float(padded[:, 2:].abs().max()) == 0.0

    def test_a_state_longer_than_the_declared_length_fails_loudly(self):
        with pytest.raises(SystemExit, match="条件の宣言長"):
            ir._right_pad(torch.ones(1, 6, 3), 5, "テスト")


class TestGoldenCaseBody:
    """MUST: golden ケースの前処理は**種別で違う**（上流 `_synthesize` の綴り）。

    text は `normalize_text` + strip、caption は **strip のみ**。caption へ正規化を掛けると
    外側括弧の剥がし・NFKC・記号削除のぶんだけ conditioning が黙って別物になる。
    """

    #: 正規化に感受する綴り（外側括弧 + 記号 — どちらも `normalize_text` の削除対象）。
    BODY = " 「①明るい声」 "

    @staticmethod
    def _spy():
        """呼ばれたら記録して**別物**を返す `normalize_text` の身代わり。"""
        calls: list[str] = []

        def normalize_text(body: str) -> str:
            calls.append(body)
            return "正規化された"

        return calls, normalize_text

    def test_the_text_kind_goes_through_normalization(self):
        calls, normalize_text = self._spy()

        assert ir.golden_case_body("t", "text", self.BODY, normalize_text) == "正規化された"
        assert calls == [self.BODY]

    def test_the_caption_kind_never_calls_normalization(self):
        calls, normalize_text = self._spy()

        assert ir.golden_case_body("c", "caption", self.BODY, normalize_text) == "「①明るい声」"
        assert calls == []

    def test_an_unknown_kind_fails_loudly(self):
        """種別の綴りが割れると、黙ってどちらかの前処理へ倒れる。"""
        _calls, normalize_text = self._spy()

        with pytest.raises(SystemExit, match="text / caption のどちらでもない"):
            ir.golden_case_body("x", "Caption", self.BODY, normalize_text)

    def test_an_empty_body_fails_loudly(self):
        _calls, normalize_text = self._spy()

        with pytest.raises(SystemExit, match="前処理後の本文が空"):
            ir.golden_case_body("c", "caption", "  \n ", normalize_text)

    def test_the_existing_caption_goldens_are_insensitive_to_normalization(self):
        """MUST: 既存 golden が動かないことの実測（動くなら再 export が要る合図）。

        caption を strip-only へ直した波の前提そのもの。正規化に感受する caption ケースを
        足したときはここが落ちるので、golden の採り直しに気づける。
        """
        pytest.importorskip("irodori_tts")
        from irodori_tts.text_normalization import normalize_text

        captions = [(name, body) for name, kind, body in ir.GOLDEN_CASES if kind == "caption"]

        assert captions, "caption 種別の golden ケースが 1 本も無い"
        for name, body in captions:
            assert normalize_text(body).strip() == body.strip(), name


class TestSanityCatchesProjectorMixups:
    """MUST: 同じ重みを 2 回読む取り違えは shape も dtype も一致するので、ここでしか出ない。"""

    def test_identical_projector_outputs_fail_loudly(self):
        shared = {"a": (torch.ones(1, 3, 4),)}
        pristine = {ir.TARGET_TEXT_PROJ: shared, ir.TARGET_CAPTION_PROJ: dict(shared)}

        with pytest.raises(AssertionError, match="同じ重みを 2 回読んでいる疑い"):
            ir._sanity(pristine)

    def test_diverging_projector_outputs_are_reported(self):
        """MUST: 比べるのは第 1 出力（caption 側の第 2 出力は norm 済みで値域が違う）。"""
        pristine = {
            ir.TARGET_TEXT_PROJ: {"a": (torch.zeros(1, 3, 4),)},
            ir.TARGET_CAPTION_PROJ: {"a": (torch.full((1, 3, 4), 2.0), torch.full((1, 3, 4), 9.0))},
        }

        assert ir._sanity(pristine) == {"a": 2.0}


class TinyResidualProjector(nn.Module):
    """`ProjectorGraph` が読む属性だけを持つ residual_mlp の代役。"""

    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Linear(4, 4, bias=False)
        self.residual_norm = ScalingNorm(1.25)
        self.residual_up = nn.Linear(4, 4, bias=False)
        self.residual_down = nn.Linear(4, 4, bias=False)


class TestCaptionProjectorGraph:
    """MUST: 第 1 出力は `text-proj` と同じ式のまま（既存 golden とビット一致する根拠）。"""

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
            wrapper, (hidden,), (torch.zeros(1, 3, 4), torch.zeros(1, 3, 4)), "テスト", 0.0
        )

        assert diff == 0.0

    def test_a_missing_output_fails_loudly(self):
        """MUST: 本数がずれたまま `zip` で黙って切り捨てない（未検証の出力が残る）。"""
        wrapper = ir.CaptionProjectorGraph(TinyResidualProjector(), ScalingNorm(2.0))

        with pytest.raises(AssertionError, match="ラッパの出力が"):
            ir._check_wrapper_equivalence(
                wrapper, (torch.zeros(1, 3, 4),), (torch.zeros(1, 3, 4),), "テスト", 0.0
            )


class TestNormDivergence:
    """MUST: `caption-proj` 第 2 出力の契約（`caption_norm` を掛けた系列）を守る唯一の門。"""

    def test_identical_norm_weights_fail_loudly(self):
        shared = ScalingNorm(1.5)

        with pytest.raises(AssertionError, match="同じ重みを 2 回読んでいる疑い"):
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
    """`_write_io` を回すための最小グラフ（入力名 `hidden` の 1 入力 1 出力）。"""

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
        """`caption-proj` の 2 出力が `output.0` / `output.1` として揃うこと。"""
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
        """MUST: 名前がずれた io は「読めるが別の入力へ入る」形で静かに通ってしまう。"""
        module, graph, out_dir = exported
        hidden = torch.randn(1, 3, 4)
        with torch.no_grad():
            expected = {"a": (module(hidden),)}

        with pytest.raises(AssertionError, match="入力名"):
            ir._write_io(graph, {"a": {"state": hidden}}, expected, out_dir)

    def test_an_output_count_mismatch_fails_loudly(self, exported):
        """MUST: 本数がずれた io は「1 本ぶん検証されない」形で静かに通ってしまう。"""
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

        with pytest.raises(AssertionError, match="期待出力"):
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
        """MUST: 上限は**入口**で見る（超過は完走してから主語違いの診断で落ちる）。

        打ち切り番兵は `len(batches) >= steps` でしか飛ばないので、参照ループ全長を超えた
        指定は 1 度も飛ばず、実重み 1 ケースぶんの denoise を回し切った後に「上流の綴りが
        台本の想定と食い違っている」で落ちる — 原因の違う診断で、しかも一番高くつく。
        """
        monkeypatch.setattr(ir, "export_series", lambda *_a, **_kw: {"dir": "x"})

        with pytest.raises(SystemExit):
            ir.main(["--dtype", "i4", "--calib-steps", steps])

        assert f"--calib-steps は 1〜{ip.NUM_STEPS}" in capsys.readouterr().err

    def test_a_calib_step_count_inside_the_reference_loop_is_forwarded(self, monkeypatch):
        """縮小 smoke の窓は塞がない（配布へ載せる側は組み立ての予算門が拒否する）。"""
        seen: dict[str, object] = {}
        monkeypatch.setattr(ir, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ir.main(["--dtype", "i4", "--calib-steps", str(ip.NUM_STEPS)])

        assert seen["calib_steps"] == ip.NUM_STEPS


def _is_f16_exact(tensor: torch.Tensor) -> bool:
    """f16 の格子に乗っているか（emit の適格判定と同じ述語）。"""
    return bool(torch.equal(tensor, tensor.to(torch.float16).to(torch.float32)))


class TestWeightDtypeSeries:
    """格納 dtype の系列（ADR 0018 / 0019 / 0027 / 0050）— 壊れると**偽 PASS** になる側だけ。

    実重みは使わない（配線の規律はモデルに依らない）。数値そのものの検証は Deno 側の E2E と
    emit の適格判定が持つ。
    """

    def test_each_dtype_gets_its_own_series_root(self, tmp_path):
        """MUST: 圧縮系列は別ディレクトリ（同居させると f32 の網が圧縮資産へ掛かる）。"""
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
        # MUST: 種を蒔いたら元へ戻す（`fork_rng`）— 大域 RNG を置き去りにすると、後続の
        # テストファイルが**別の乱数**でモジュールを組むことになり、こちらの追加が離れた
        # 場所の実測門（`irodori.dacvae.export` の切り詰めビット一致）を動かす。
        with torch.random.fork_rng():
            torch.manual_seed(0)
            module = TinyResidualProjector()
            before = module.projector.weight.clone()

            quantized = ir.fake_quant("f32", {"text-proj": module})

            assert quantized.reports == {}
            assert quantized.scales == {}
            assert torch.equal(module.projector.weight, before)

    def test_f16_rounds_every_module_it_is_handed(self):
        """MUST: 1 本でも漏らすと、漏れた側だけが元の重みで計算した値を golden に載せる。"""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            modules = {name: TinyResidualProjector() for name in "abc"}
            assert not any(_is_f16_exact(module.projector.weight) for module in modules.values()), (
                "丸め前から格子に乗っていては検出力が無い"
            )

            quantized = ir.fake_quant("f16", modules)

            assert sorted(quantized.reports) == ["a", "b", "c"]
            assert quantized.scales == {}
            for module in modules.values():
                assert all(_is_f16_exact(tensor) for tensor in module.parameters())

    def test_it_refuses_an_empty_set_of_modules(self):
        """`--dtype f16` を指定したのに丸める相手が 0 本、を沈黙させない。"""
        with pytest.raises(SystemExit, match="1 本も無い"):
            ir.fake_quant("f16", {})

    def test_f16_refuses_a_set_where_nothing_was_actually_rounded(self):
        """入口の「モジュール 0 本」とは別の穴 — **モジュールは在るが丸めた本数が 0**。

        docstring が名乗る「総数 0 は落とす」は i8 だけの門だったので、f16 は 1 本も丸めずに
        素通りできた（現行の 9 モジュールは必ず f32 パラメータを持つので今は到達しないが、
        門としては立っていなかった）。
        """
        with pytest.raises(SystemExit, match="丸めた重みが 1 本も無い"):
            ir.fake_quant("f16", {"text_norm": nn.Identity()})

    def test_i8_quantizes_only_the_per_channel_types(self):
        """MUST: i8 が触るのは `QUANT_CHANNEL_AXES` の型の `weight` だけ（ADR 0019）。

        norm 系まで丸めると emit の適格判定（重みスロットだけ圧縮）と食い違い、golden だけが
        別の重みで計算した値になる。
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
                assert torch.equal(restored, weight), "i8 の格子に乗っていない"

    def test_i8_lets_a_norm_only_role_through(self):
        """MUST: norm 単体の役割は i8 で素通り（`f16` と違い「0 本」が正しい状態）。"""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            modules = {"text-proj": TinyResidualProjector(), "text_norm": ScalingNorm(1.5)}

            quantized = ir.fake_quant("i8", modules)

            assert sorted(quantized.reports) == ["text-proj", "text_norm"]
            assert sorted(quantized.scales) == ["text-proj"]

    def test_i8_refuses_a_set_with_nothing_quantizable(self):
        """`--dtype i8` を指定したのに per-channel の対象が全体で 0 本、を沈黙させない。"""
        with pytest.raises(SystemExit, match="per-channel 量子化できたモジュールが 1 本も無い"):
            ir.fake_quant("i8", {"text_norm": ScalingNorm(1.5)})

    def test_every_loaded_module_reaches_the_fake_quant(self):
        """束ねる相手は `export_series` が `load_*` で組む全部（欠けは golden の食い違いになる）。

        名前を数え上げるのではなく**呼び出しの実引数**を捕まえる — 表を書き写すと、
        `export_series` 側が 1 本増やしたときに写しだけが古いまま緑になる。
        """
        source = inspect.getsource(ir.export_series)
        handed = source[source.index("quantized = fake_quant(") :]
        handed = handed[: handed.index("\n    )")]
        for name in ("backbone", "projectors[", "speaker_encoder", "duration", "dit"):
            assert name in handed
        for norm in ("speaker_norm", "text_norm", "caption_norm"):
            assert f'"{norm}": {norm}' in handed


class TinyRopeBody(nn.Module):
    """`assert_rope_lifted` が降格できる buffer を持つ backbone 本体の代役。"""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Linear(4, 4, bias=False)
        self.register_buffer("inv_freq", torch.ones(2))


class TinyBackboneHolder(nn.Module):
    """`BackboneGraph` が読む形（本体は `backbone` 属性の内側）。"""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = TinyRopeBody()


class TinyRopeEncoder(nn.Module):
    """`SpeakerGraph` が `__init__` で読む属性（`head_dim`）を持つ encoder の代役。"""

    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 4
        self.in_proj = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])


class TinyJointAttention(nn.Module):
    """`JointAttention` の**子の顔ぶれ**だけを写した代役（計算は持たない — 張り替えを見るだけ）。

    並びも実物と同じ（self 側 3 本 → 条件側 6 本 → gate / wo → q/k ノルム）。
    """

    def __init__(self) -> None:
        super().__init__()
        for name in ("wq", "wk", "wv", *ir.CONTEXT_KV_PROJECTIONS, "gate", "wo"):
            setattr(self, name, nn.Linear(4, 4, bias=False))
        self.q_norm = ScalingNorm(1.0)
        self.k_norm = ScalingNorm(1.0)


class TinyDitBlock(nn.Module):
    """`DiffusionBlock` の子の顔ぶれ（attention + mlp + 2 本の adaLN）。"""

    def __init__(self) -> None:
        super().__init__()
        self.attention = TinyJointAttention()
        self.mlp = nn.Linear(4, 4, bias=False)
        self.attention_adaln = nn.Linear(4, 4, bias=False)
        self.mlp_adaln = nn.Linear(4, 4, bias=False)


class TinyDit(nn.Module):
    """DiT のラッパ 2 本が抱える部分木 + **抱えない**枝（`load_dit` が丸ごと組む内側のコピー相当）。

    block は実物の子の顔ぶれを持つ（DiT を `dit-context` / `dit` に割った — ADR 0114 — ので、
    1 つの block の重みが 2 本のラッパへどう分かれるかを表で見るには block の中身が要る）。
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
    """i8 の scale 台帳を**ラッパ内 FQN**へ張り替える表（`TARGET_SCALE_SOURCES`）。

    MUST: ここが崩れると emit が「適格なのに scale が無い」で落ちる（値が壊れる形では
    通らない）。落ちる場所が遠いので、表とラッパの対応はここで固定する。
    """

    def test_the_table_covers_every_target(self):
        assert sorted(ir.TARGET_SCALE_SOURCES) == sorted(ir.TARGETS)

    def _rebased(self, target: str, wrapper: nn.Module, module: nn.Module) -> list[str]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            scales = ir.fake_quant("i8", {target: module}).scales
        return sorted(ir.target_scales(target, wrapper, scales))

    def test_every_wrapper_receives_a_scale_for_each_of_its_weights(self):
        """MUST: ラッパが持つ per-channel 対象の重みは**全て**張り替え先が見つかること。

        `pytest.importorskip` は `SpeakerGraph` / `DitGraph` が実数形 RoPE 表を作るのに
        上流実装を引くため（表そのものは張り替えに使わないが `__init__` が呼ぶ）。
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
        """`CaptionProjectorGraph` は `ProjectorGraph` を内包する（接頭辞が 2 段）。"""
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
        """`load_dit` は DiT **丸ごと**を組むので、台帳には使わない枝の scale まで載る。

        DiT を 2 本に割った（ADR 0114）ので、`dit` が受け取るのは条件側の射影を除いた block の
        linear と block 外 — 条件側の 6 本は `dit-context` へ行く（期待値はグラフが変わったぶん
        だけ動いた）。
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
        """`dit-context` は同じ `dit` 役の台帳から、条件側の射影 6 本 × ブロックだけを拾う。

        丸めは `export_series` と同じく `dit` 役（`TextToLatentRFDiT` 丸ごと）に 1 回だけ当てる —
        `dit-context` 役の台帳というものは無い。
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
        """MUST: 1 本も当たらない張り替えを黙って空で返さない（emit まで気づけなくなる）。"""
        module = TinyResidualProjector()
        wrapper = ir.ProjectorGraph(module)
        monkeypatch.setattr(
            ir,
            "TARGET_SCALE_SOURCES",
            {ir.TARGET_TEXT_PROJ: ((ir.TARGET_TEXT_PROJ, "", "stale."),)},
        )

        with pytest.raises(SystemExit, match="張り替えられなかった"):
            self._rebased(ir.TARGET_TEXT_PROJ, wrapper, module)

    def test_f32_and_f16_hand_no_scales_to_emit(self):
        """f16 は scale を持たない（台帳が空なら張り替えも空 — emit も f16 では引かない）。"""
        wrapper = ir.ProjectorGraph(TinyResidualProjector())

        assert ir.target_scales(ir.TARGET_TEXT_PROJ, wrapper, {}) == {}


#: 小さな実物 DiT の条件の宣言長（`_dit_cases` が右 pad する長さ）と speaker の patch 後上限。
TINY_MODEL_CONFIG = {"max_text_len": 6, "max_caption_len": 7}
TINY_SPEAKER_MAX = 4
#: `dit` の mask の条件側の総長（text 6 + speaker 4+1 + caption 7）。
TINY_CONTEXT_TOTAL = 6 + (TINY_SPEAKER_MAX + 1) + 7


def _tiny_real_dit() -> nn.Module:
    """上流の `TextToLatentRFDiT` そのもの（小さな config・scratch の条件エンコーダ）。

    head_dim は実重みと同じ 64（RoPE の実数化のビット一致は形依存 — `irodori.patch` の NOTE。
    64 では 0）、heads は 2（`_apply_rotary_half` が heads 軸を半分に割る）。重みは全部を
    決定的な乱数で上書きする — 上流は `out_proj` を 0 で初期化するので、そのままだと出力が
    恒等的に 0 になり比較が恒真になる。
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
    """小さな実物 DiT で、`export_series` の DiT の段（ケース → パッチ前の参照 → 入力の組み）を
    同じ関数で踏んだ結果。"""

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
    """パッチ前に参照を採り、パッチを当ててから返す（`export_series` の ①→② の順序）。"""
    model = _tiny_real_dit()
    split = _TinySplit(model)
    patch_irodori.apply_patches()
    return model, split


class TestDitSplit:
    """DiT を `dit-context`（条件側 K/V・生成 1 回）と `dit` に割る（ADR 0114）。

    MUST: 割っても数値は 1 ビットも動かない — 参照層の sha 門（WAV / latent）は不変が期待値。
    """

    def test_the_composed_pair_reproduces_the_upstream_forward_bit_for_bit(self, tiny_split):
        """分割の同値門そのもの（uncond 3 変種と S の両端 2 / 750 を含む全ケース・atol 0）。"""
        model, split = tiny_split
        wrappers = ir.dit_wrappers(model, split.sym_max)
        composed = ir.ComposedDitGraph(wrappers.context, wrappers.dit)

        for name, args in split.inputs.items():
            diff = ir._check_wrapper_equivalence(
                composed, tuple(args.values()), (split.pristine[name],), name, 0.0
            )
            assert diff == 0.0, name

    def test_the_context_graph_matches_the_upstream_kv_cache(self, tiny_split):
        """`dit-context` の golden は上流 `build_context_kv_cache` の値（ラッパ自身ではない）。"""
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
        """`dit` の golden 入力（K/V = `dit-context` の期待値）で上流の出力にビット一致する。"""
        model, split = tiny_split
        wrappers = ir.dit_wrappers(model, split.sym_max)

        for name, args in split.dit_inputs.items():
            assert list(args) == list(wrappers.dit.input_names), "golden 入力の並びが IR と違う"
            assert (
                ir._check_wrapper_equivalence(
                    wrappers.dit, tuple(args.values()), (split.pristine[name],), name, 0.0
                )
                == 0.0
            )

    def test_the_context_outputs_are_one_k_and_one_v_per_block_in_order(self, tiny_split):
        """出力は 2 × ブロック数本で、ブロック b の K / V（1 段目の連結）が b の昇順に並ぶ。"""
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
        # 取り違えの検出力: ブロックごとに値が違う（同じなら順序の入れ替えが素通りする）。
        assert not torch.equal(outputs[0], outputs[2])

    def test_a_swapped_block_order_breaks_the_composition(self, tiny_split):
        """故障注入: ブロック 0 と 1 の K/V を入れ替えて渡すと、上流とのビット一致が崩れる。"""
        model, split = tiny_split
        wrappers = ir.dit_wrappers(model, split.sym_max)
        args = split.dit_inputs["dit-cond-1s"]
        values = list(args.values())
        values[3:5], values[5:7] = values[5:7], values[3:5]

        with pytest.raises(AssertionError, match="eager 同値が崩れた"):
            ir._check_wrapper_equivalence(
                wrappers.dit, tuple(values), (split.pristine["dit-cond-1s"],), "swap", 0.0
            )

    def test_the_two_wrappers_split_the_old_ownership_and_share_only_k_norm(self):
        """所有の和 = 割る前の `DitGraph` が抱えていた部分木の全重み・重なりは `k_norm` だけ。

        所有 = 容器の initializer（張り替え・読み戻し・i4 適格の判定の前提）なので、ここが崩れると
        格納指定が未知キーで落ちるか、重みがどちらの容器にも載らない。
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
        """uncond 3 変種は cond の state のまま — `dit-context` の golden は 1 組で足りる。"""
        _model, split = tiny_split

        assert list(split.context_inputs) == [ir.DIT_CASES[0][0]]
        assert set(split.owner) == {name for name, *_rest in ir.DIT_CASES}
        assert set(split.owner.values()) == {ir.DIT_CASES[0][0]}

    def test_the_exported_boundary_carries_the_contract_names(self, tiny_split, tmp_path):
        """IR の境界名（`dit-context` の出力 / `dit` の 4 本目以降の入力）が契約の綴りになり、
        組み立ての門（`irodori.distribution`）がそのまま受け取る。

        torch.export は可変長引数を `context_<i>`・出力を FX ノード名で名乗る — 付け替えが
        外れるとランタイムが常駐テンソルを名前で束ねられない。
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
        # golden io は IR の入力名と一致して初めて書ける（`_write_io` の門）。
        written = ir._write_io(
            dit_graph,
            {"dit-cond-1s": split.dit_inputs["dit-cond-1s"]},
            {"dit-cond-1s": (split.pristine["dit-cond-1s"],)},
            tmp_path,
        )
        assert written == [f"{ir.IO_PREFIX}dit-cond-1s{ir.IO_SUFFIX}"]


class TestPristineDitContextOrdering:
    def test_taking_the_context_reference_after_patching_fails_loudly(self, monkeypatch):
        """MUST: パッチ後に採るとラッパと同じ経路の値になり、同値門が恒真化する。"""
        monkeypatch.setattr(patch_irodori, "_APPLIED", True)

        with pytest.raises(AssertionError, match="パッチ適用後に参照を採ろうとした"):
            ir._pristine_dit_context_outputs(TinyDit(), {})


class TestDitContextCases:
    """`dit-context` の golden は条件 state の組ごとに 1 本（組は値で判定する）。"""

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
    """入力 2 本・出力 2 本（ノードの出力）の最小グラフ（境界名の付け替えの被験体）。"""
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
        with pytest.raises(AssertionError, match="名前 1 本"):
            ir.name_boundary(_boundary_graph(), outputs=("only",))

    def test_a_name_that_collides_with_a_remaining_value_fails_loudly(self):
        """潰すと宣言が黙って 1 本消える。"""
        with pytest.raises(AssertionError, match="衝突"):
            ir.name_boundary(_boundary_graph(), outputs=("add", "x"))

    def test_an_output_that_is_not_produced_by_a_node_fails_loudly(self):
        """入力をそのまま返す出力を付け替えると、入力名まで動く。"""
        graph = _boundary_graph()
        graph.outputs = ["x", "add_1"]

        with pytest.raises(AssertionError, match="ノードの出力でない"):
            ir.name_boundary(graph, outputs=("first", "second"))


class TestProvenance:
    def test_the_container_points_at_the_notice(self) -> None:
        """NOTICE を配る他の family と同じく、単一の krm から改変告知の在処が辿れる。"""
        from karume.dist import NOTICE_FILENAME

        assert ir.PROVENANCE.notice == NOTICE_FILENAME
