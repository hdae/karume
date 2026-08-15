"""`anima/export.py` の台本レベルの約束事（実重み不要分）。

実重みを要する emit / verify は手動（README 参照）。ここで固定するのは、壊れると**偽 PASS**
になる側の規律だけ:

- `--verify` と `--target` の同一プロセス併用を CLI が機械的に拒否すること
- 参照採取が VAE パッチ適用**後**にならないこと（恒真化の門）
- ターゲット表と builder / LoRA 前置きの対応が抜けないこと
"""

from __future__ import annotations

import argparse

import pytest
import torch
from torch import nn

from anima import export as export_anima
from anima import patch as patch_anima


class TestTargets:
    def test_every_target_has_a_builder(self):
        assert set(export_anima.BUILDERS) == set(export_anima.TARGETS)

    def test_lora_prefixes_only_name_real_targets(self):
        """LoRA の対象は DiT と conditioner だけ（recon §1）— 綴り誤りは黙って空振りする。"""
        assert set(export_anima.LORA_PREFIXES) <= set(export_anima.TARGETS)
        assert set(export_anima.LORA_PREFIXES) == {"transformer", "text_conditioner"}


class TestFakeQuant:
    """`--dtype f16` の丸め（ADR 0006）— 既定 f32 の挙動を 1 ビットも動かさないこと。"""

    def test_f32_leaves_the_weights_untouched(self):
        model = nn.Linear(4, 3)
        before = model.weight.clone()

        export_anima._fake_quant(argparse.Namespace(dtype="f32"), model, "t")

        assert torch.equal(model.weight, before)

    def test_f16_rounds_the_weights_to_representable_values(self):
        model = nn.Linear(4, 3)

        export_anima._fake_quant(argparse.Namespace(dtype="f16"), model, "t")

        assert torch.equal(model.weight, model.weight.to(torch.float16).to(torch.float32))

    def test_i8_rounds_to_per_channel_representable_values(self):
        model = nn.Linear(4, 3)

        scales = export_anima._fake_quant(argparse.Namespace(dtype="i8"), model, "t")

        scale = scales["weight"]
        assert list(scale.shape) == [3, 1]
        assert torch.equal(torch.round(model.weight / scale) * scale, model.weight)

    def test_the_default_output_root_is_a_separate_series_per_dtype(self):
        """MUST: 圧縮系列は別ディレクトリ（ADR 0018 / 0019）— 同居させると f32 の網が消える。"""
        roots = export_anima.DEFAULT_OUT_ROOTS

        assert set(roots) == set(export_anima.WEIGHT_DTYPES) == {"f32", "f16", "i8"}
        assert len(set(roots.values())) == len(roots)


class TestDtypeTargets:
    """MUST: i8 系列は transformer のみ（ADR 0019）— 表と CLI の両方で固定する。"""

    def test_every_dtype_declares_its_targets(self):
        assert set(export_anima.DTYPE_TARGETS) == set(export_anima.WEIGHT_DTYPES)
        for dtype, targets in export_anima.DTYPE_TARGETS.items():
            assert set(targets) <= set(export_anima.TARGETS), dtype

    def test_i8_covers_the_transformer_only(self):
        assert export_anima.DTYPE_TARGETS["i8"] == ("transformer",)

    @pytest.mark.parametrize("flag", ["--target", "--verify"])
    def test_an_i8_target_outside_the_table_is_refused(self, monkeypatch, flag):
        """既定を絞るだけだと明示指定が通ってしまう（排除したはずの資産が黙って生える）。"""
        monkeypatch.setattr("sys.argv", ["export_anima.py", "--dtype", "i8", flag, "vae_decoder"])

        with pytest.raises(SystemExit):
            export_anima.main()


class TestCliExclusion:
    def test_verify_and_target_cannot_share_a_process(self, monkeypatch):
        """MUST: emit がパッチを当てた後では「パッチ前の参照」が採れない（ADR 0013 / 0016）。"""
        monkeypatch.setattr(
            "sys.argv",
            ["export_anima.py", "--verify", "vae_decoder", "--target", "vae_decoder"],
        )

        with pytest.raises(SystemExit):
            export_anima.main()


class TestOrderGuard:
    def test_taking_a_reference_after_the_vae_patch_is_rejected(self, monkeypatch):
        """恒真化の門 — パッチ後の「参照」は差が常に 0 になるので緑でも証拠にならない。"""
        monkeypatch.setattr(patch_anima, "vae_patches_applied", lambda: True)

        with pytest.raises(RuntimeError, match="恒真化"):
            export_anima._assert_vae_unpatched("テスト")

    def test_it_passes_before_the_patch(self, monkeypatch):
        monkeypatch.setattr(patch_anima, "vae_patches_applied", lambda: False)

        export_anima._assert_vae_unpatched("テスト")


class TestIoNaming:
    def test_graph_input_names_must_match_the_declared_order(self, tmp_path):
        """IR の入力名 = forward の引数名。ずれたまま位置で書くと golden が別の入力を指す。"""

        class Two(nn.Module):
            def forward(self, first, second):
                return first + second

        component = export_anima.Component(
            module=Two(),
            dynamic_shapes=None,
            input_names=("first", "third"),
            cases=(("case0", (torch.zeros(2), torch.ones(2))),),
            reference=None,
        )
        graph = export_anima.export_to_file(
            Two(), component.cases[0][1], tmp_path / "model.safetensors"
        )

        with pytest.raises(AssertionError, match="グラフ入力名が宣言と不一致"):
            export_anima._write_io(component, graph, tmp_path)


class TestDitGraph:
    """`--dit-graph dyn`（S 形の追加系列 — #21 波 T2）の CLI 規律。"""

    def test_the_dyn_series_lands_in_its_own_directory(self, monkeypatch):
        """MUST: 追加系列は**別ディレクトリ**（静的系列を上書きしたら波 T2 の前提が消える）。"""
        captured: dict[str, object] = {}

        def stub(target, args, out_dir):
            captured[target] = out_dir
            return {"target": target}

        monkeypatch.setattr("sys.argv", ["export_anima.py", "--dtype", "f16", "--dit-graph", "dyn"])
        monkeypatch.setattr(export_anima, "emit_target", stub)

        export_anima.main()

        assert set(captured) == {"transformer"}
        assert captured["transformer"].parent.name == (
            f"{export_anima.DEFAULT_OUT_ROOTS['f16'].name}{export_anima.DYN_SUFFIX}"
        )

    @pytest.mark.parametrize("flag", ["--target", "--verify"])
    def test_a_non_transformer_target_is_refused(self, monkeypatch, flag):
        """S 形は transformer 専用（他 3 ターゲットは解像度に依らないので共有のまま）。"""
        monkeypatch.setattr(
            "sys.argv", ["export_anima.py", "--dit-graph", "dyn", flag, "vae_decoder"]
        )

        with pytest.raises(SystemExit):
            export_anima.main()

    def test_resolution_is_refused_because_it_does_not_apply(self, monkeypatch):
        """効かないノブを黙って受けない — S 形のグラフは解像度を 1 つも持たない。"""
        monkeypatch.setattr(
            "sys.argv", ["export_anima.py", "--dit-graph", "dyn", "--resolution", "1024"]
        )

        with pytest.raises(SystemExit):
            export_anima.main()

    def test_the_golden_cases_use_two_different_token_lengths(self):
        """2 点評価の中身: 解像度が 2 つとも違う（同じ S を並べると束縛の穴が数に出ない）。"""
        assert len(set(export_anima.DIT_DYN_RESOLUTIONS)) == len(export_anima.DIT_DYN_RESOLUTIONS)
        assert export_anima.DIT_DYN_RESOLUTIONS[0] == export_anima.RESOLUTION
