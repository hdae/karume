"""`export_irodori.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（既存 `export_deberta.py` / `export_embeddinggemma.py` のテストと同じ
規律）。ここで固定するのは、壊れると**偽 PASS** になる側の規律だけ:

- 参照（golden の期待値）は**パッチ前**にしか採れない（採れてしまうと同値検証が恒真化する）
- 静的方式（実行時 attention_mask 非対応）の実測が、方式が崩れたときに実際に落ちる
- text / caption projector の取り違え（同じ重みを 2 回読む）が `_sanity` で落ちる
- `_write_io` が IR の入力名と食い違う io を書かない / 出力の本数がずれた io を書かない
- `caption-proj` の第 2 出力が `caption_norm` を掛けた系列であり、第 1 出力は素の projector 出力
  のまま（`text-proj` と同じ式）であること
- `text_norm` / `caption_norm` の取り違え（同じ重みを 2 回読む）が `_norm_divergence` で落ちる
- 参照なし（マスク全 0）の speaker 出力が**厳密に 0** であることの実測が、0 でなければ落ちる
- `duration` の `aux_features` 非依存の実測が、依存していたら落ちる
- `dit` の cond / uncond 3 変種が**互いに違う**ことの実測が、同じなら落ちる
- 条件 state の右 pad が宣言長を超えたら落ちる
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

import export_irodori as ir
from karume import patch_irodori
from karume.ir import IrGraph, IrValue
from karume.pipeline import export_to_file

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
        graph = export_to_file(module, (torch.randn(1, 3, 4),), tmp_path / ir.MODEL_FILE)
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
