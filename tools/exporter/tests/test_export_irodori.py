"""`export_irodori.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（既存 `export_deberta.py` / `export_embeddinggemma.py` のテストと同じ
規律）。ここで固定するのは、壊れると**偽 PASS** になる側の規律だけ:

- 参照（golden の期待値）は**パッチ前**にしか採れない（採れてしまうと同値検証が恒真化する）
- 静的方式（実行時 attention_mask 非対応）の実測が、方式が崩れたときに実際に落ちる
- text / caption projector の取り違え（同じ重みを 2 回読む）が `_sanity` で落ちる
- `_write_io` が IR の入力名と食い違う io を書かない
"""

from __future__ import annotations

import pytest
import torch
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


class TestPristineReferenceOrdering:
    """MUST: 参照はパッチ前にしか採れない（採れると同値検証が恒真化して偽 PASS になる）。"""

    def test_taking_a_reference_after_patching_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", True)

        with pytest.raises(AssertionError, match="パッチ適用後に参照を採ろうとした"):
            ir._pristine_outputs(MaskRespectingBackbone(), {}, CASES)

    def test_taking_a_reference_before_patching_is_allowed(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)

        outputs = ir._pristine_outputs(MaskRespectingBackbone(), {}, CASES)

        assert set(outputs[ir.TARGET_BACKBONE]) == {"short", "cap"}
        assert tuple(outputs[ir.TARGET_BACKBONE]["short"].shape) == (1, 3, 4)


class TestStaticSchemeEvidence:
    """静的方式（右詰め pad をホストで消す）の実測が、恒真でないことを確かめる。"""

    def test_a_mask_respecting_backbone_measures_zero(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        backbone = MaskRespectingBackbone()
        pristine = ir._pristine_outputs(backbone, {}, CASES)[ir.TARGET_BACKBONE]

        evidence = ir._static_scheme_evidence(backbone, TEXT_CONFIG, MODEL_CONFIG, CASES, pristine)

        assert evidence == {"short": 0.0, "cap": 0.0}

    def test_a_pad_leaking_backbone_fails_loudly(self, monkeypatch):
        """MUST: 方式が崩れていたら落ちる — 回避せずに止めるための門。"""
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        backbone = PadLeakingBackbone()
        pristine = ir._pristine_outputs(backbone, {}, CASES)[ir.TARGET_BACKBONE]

        with pytest.raises(AssertionError, match="静的方式"):
            ir._static_scheme_evidence(backbone, TEXT_CONFIG, MODEL_CONFIG, CASES, pristine)

    def test_a_case_longer_than_its_family_cap_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(patch_irodori, "_APPLIED", False)
        long_case = (("long", "text", torch.arange(1, 40, dtype=torch.int64).unsqueeze(0)),)
        backbone = MaskRespectingBackbone()
        pristine = ir._pristine_outputs(backbone, {}, long_case)[ir.TARGET_BACKBONE]

        with pytest.raises(SystemExit, match="上限"):
            ir._static_scheme_evidence(backbone, TEXT_CONFIG, MODEL_CONFIG, long_case, pristine)


class TestSanityCatchesProjectorMixups:
    """MUST: 同じ重みを 2 回読む取り違えは shape も dtype も一致するので、ここでしか出ない。"""

    def test_identical_projector_outputs_fail_loudly(self):
        shared = {"a": torch.ones(1, 3, 4)}
        pristine = {ir.TARGET_TEXT_PROJ: shared, ir.TARGET_CAPTION_PROJ: dict(shared)}

        with pytest.raises(AssertionError, match="同じ重みを 2 回読んでいる疑い"):
            ir._sanity(pristine)

    def test_diverging_projector_outputs_are_reported(self):
        pristine = {
            ir.TARGET_TEXT_PROJ: {"a": torch.zeros(1, 3, 4)},
            ir.TARGET_CAPTION_PROJ: {"a": torch.full((1, 3, 4), 2.0)},
        }

        assert ir._sanity(pristine) == {"a": 2.0}


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
            expected = {"a": module(hidden)}

        written = ir._write_io(graph, {"a": {"hidden": hidden}}, expected, out_dir)

        assert written == [f"{ir.IO_PREFIX}a{ir.IO_SUFFIX}"]

    def test_an_input_name_mismatch_fails_loudly(self, exported):
        """MUST: 名前がずれた io は「読めるが別の入力へ入る」形で静かに通ってしまう。"""
        module, graph, out_dir = exported
        hidden = torch.randn(1, 3, 4)
        with torch.no_grad():
            expected = {"a": module(hidden)}

        with pytest.raises(AssertionError, match="入力名"):
            ir._write_io(graph, {"a": {"state": hidden}}, expected, out_dir)

    def test_a_multi_output_graph_fails_loudly(self, exported):
        _module, graph, out_dir = exported
        extra = IrGraph(
            inputs=graph.inputs,
            outputs=[*graph.outputs, "second"],
            nodes=graph.nodes,
            values={**graph.values, "second": IrValue(dtype="f32", shape=[1])},
            initializers=graph.initializers,
            symbols=graph.symbols,
        )

        with pytest.raises(AssertionError, match="IR 出力が"):
            ir._write_io(extra, {}, {}, out_dir)


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
