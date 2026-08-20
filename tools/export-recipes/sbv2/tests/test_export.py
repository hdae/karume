"""SBV2 export 台本（dp / front / flow / dec / voice）の約束事の固定（M1-P3 波 1・6・7）。

前半（golden 入力の作りと CLI の排他）は実重み不要で常に走る。後半は `inputs/sbv2/FN4/` の
実重みと `sbv2` dependency-group が揃っている環境でだけ走り、無ければ SKIP する — 重みは
251MB 級でリポジトリ管理外、依存も既定の `uv sync` には入らないため。

NOTE: front / flow / voice の export はパッチをプロセス全域に当て、dec / voice は
`remove_weight_norm` で重みを畳む。dp のテストはどちらの影響も受けない（MHA / FFN / spline
を通らず dec でもない）ので順序依存は無いが、**「前の参照」を採る検証だけは別プロセス**
（`--verify <target>` のサブプロセス実行）で行う — ADR 0013 の排他規律そのもの。
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

from _shared.paths import REPO_ROOT, SERIES_ROOT
from karume.dims import parse_dim
from karume.ops import EMITTABLE_OPS
from karume.quantize import (
    QUANT_CHANNEL_AXES,
    channel_scale,
    fake_quant_int8,
    quantize_to_int8,
)
from karume.verify import verify_model
from sbv2 import export as export_sbv2
from sbv2 import patch as patch_sbv2

MODEL_DIR = export_sbv2.DEFAULT_MODEL_DIR

_WEIGHTS_PRESENT = (
    MODEL_DIR.is_dir()
    and any(MODEL_DIR.glob("*.safetensors"))
    and (MODEL_DIR / export_sbv2.CONFIG_FILE).is_file()
)
_PACKAGE_PRESENT = importlib.util.find_spec("style_bert_vits2") is not None

requires_weights = pytest.mark.skipif(
    not (_WEIGHTS_PRESENT and _PACKAGE_PRESENT),
    reason=(
        f"実重み（{MODEL_DIR}）か sbv2 dependency-group が無い"
        "（`uv sync --all-groups` と重みの配置が前提 — tools/export-recipes/sbv2/README.md）"
    ),
)


class TestGoldenInputs:
    """golden の入力生成は実重みに依らない — ここは常に走る。"""

    def test_hidden_has_the_encoder_output_shape(self):
        assert export_sbv2.make_hidden(37).shape == (1, 192, 37)
        assert export_sbv2.make_hidden(37).dtype == torch.float32

    def test_hidden_is_reproducible_for_the_same_length(self):
        # 生成物をバイト一致で再現できることが golden の前提（グローバル seed に依存しない）。
        assert torch.equal(export_sbv2.make_hidden(37), export_sbv2.make_hidden(37))

    def test_hidden_differs_between_lengths(self):
        # 長さごとに seed を派生させている（同じ列の prefix を使い回すと、P を変えた
        # ケースが「同じ入力の切り出し」になり、P 依存のバグに対する検出力が落ちる）。
        short = export_sbv2.make_hidden(37)[..., :16]
        assert not torch.equal(short, export_sbv2.make_hidden(16))

    def test_mask_is_all_ones_without_padding(self):
        assert torch.equal(export_sbv2.make_mask(16, 0), torch.ones(1, 1, 16))

    def test_mask_zeroes_exactly_the_trailing_padding(self):
        mask = export_sbv2.make_mask(16, 5)
        assert mask.shape == (1, 1, 16)
        assert torch.equal(mask[..., :11], torch.ones(1, 1, 11))
        assert torch.equal(mask[..., 11:], torch.zeros(1, 1, 5))

    @pytest.mark.parametrize("pad", [-1, 16, 20])
    def test_out_of_range_padding_is_rejected(self, pad):
        with pytest.raises(ValueError):
            export_sbv2.make_mask(16, pad)

    def test_case_table_covers_the_declared_range_and_the_mask_path(self):
        lengths = [length for _, length, _ in export_sbv2.GOLDEN_CASES]
        padded = [name for name, _, pad in export_sbv2.GOLDEN_CASES if pad > 0]

        # 下限 2（0/1 特殊化の回避線）と宣言上限 SYM_MAX の両端を踏む。
        assert min(lengths) == 2
        assert max(lengths) == export_sbv2.SYM_MAX
        assert len(set(lengths)) == len(lengths)
        # x_mask に 0 を含むケースが必ず 1 本（マスク経路の唯一の検出器）。
        assert padded

    def test_case_inputs_are_keyed_by_the_graph_input_names(self):
        g = torch.zeros(1, 512, 1)
        built = export_sbv2.build_cases(g)

        assert [name for name, _ in built] == [name for name, _, _ in export_sbv2.GOLDEN_CASES]
        for name, args in built:
            assert set(args) == {"h", "x_mask", "g"}, name
            assert args["g"] is g, name


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """一時ディレクトリへ dp を export する（リポジトリの成果物には触らない）。"""
    out = tmp_path_factory.mktemp("sbv2") / export_sbv2.TARGET_DP
    return out, export_sbv2.export_dp(MODEL_DIR, out)


@pytest.fixture(scope="module")
def net_g():
    return export_sbv2.load_net_g(MODEL_DIR)[0]


@requires_weights
class TestDurationPredictorExport:
    def test_the_container_passes_the_full_verification(self, exported):
        out, _ = exported

        verify_model(out / export_sbv2.MODEL_FILE)

    def test_the_graph_declares_the_wrapper_argument_names(self, exported):
        out, summary = exported
        graph = verify_model(out / export_sbv2.MODEL_FILE)

        assert [spec.name for spec in graph.inputs] == ["h", "x_mask", "g"]
        assert [spec.shape for spec in graph.inputs] == [[1, 192, "P"], [1, 1, "P"], [1, 512, 1]]
        assert summary["symbols"] == ["P"]

    def test_the_graph_stays_inside_the_current_contract(self, exported):
        _, summary = exported

        # 波 1 の前提そのもの: dp は語彙を広げずに通る（recon §2）。増えたら fail loudly。
        assert set(summary["ops"]) <= set(EMITTABLE_OPS)
        assert set(summary["ops"]) == {"add", "conv1d", "layer_norm", "mul", "permute", "relu"}

    def test_the_io_files_follow_the_naming_convention(self, exported):
        out, summary = exported

        expected = [
            f"{export_sbv2.IO_PREFIX}{name}{export_sbv2.IO_SUFFIX}"
            for name, _, _ in export_sbv2.GOLDEN_CASES
        ]
        assert summary["io"] == expected
        for name in expected:
            with safe_open(str(out / name), framework="pt") as handle:
                keys = set(handle.keys())
            assert keys == {"input.h", "input.x_mask", "input.g", "output.0"}

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported, case):
        out, _ = exported
        name, length, _ = case
        graph = verify_model(out / export_sbv2.MODEL_FILE)
        tensors = _io_tensors(out, name)

        for spec in graph.inputs:
            actual = list(tensors[f"input.{spec.name}"].shape)
            declared = [
                length if isinstance(dim, str) and parse_dim(dim).sym == "P" else dim
                for dim in spec.shape
            ]
            assert actual == declared, spec.name
        assert list(tensors["output.0"].shape) == [1, 1, length]
        assert tensors["output.0"].dtype == torch.float32

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_output_reproduces_a_fresh_eager_forward(self, exported, net_g, case):
        """格納された期待値が、同じ入力に対する torch の出力とビット一致する。

        `_write_io` がケースを取り違える / 入力と出力の対応がずれる形はここで落ちる
        （ケースをまたいで P が違うので、ずれると shape から壊れる）。
        """
        out, _ = exported
        name, _, _ = case
        module = export_sbv2.DurationPredictorGraph(net_g.dp)
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = module(tensors["input.h"], tensors["input.x_mask"], tensors["input.g"])

        assert torch.equal(fresh, tensors["output.0"])

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported, net_g, case):
        """故障注入 — 期待値を 1 要素だけ壊すと上のテストが赤になることを固定する。

        壊れた値を見逃す形（キーの引き違いで別テンソルを読む / 自分自身と比較する）は
        上のテストだけでは検出できない（恒真のまま緑になる）。ここが恒真化の門番。
        """
        out, _ = exported
        name, _, _ = case
        module = export_sbv2.DurationPredictorGraph(net_g.dp)
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = module(tensors["input.h"], tensors["input.x_mask"], tensors["input.g"])
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh, corrupted)

    def test_the_padded_case_zeroes_the_masked_tail(self, exported):
        """`x * x_mask` 4 箇所が効いていれば、パディング列の出力は厳密に 0 になる。"""
        out, _ = exported
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        output = _io_tensors(out, name)["output.0"]

        assert torch.equal(output[..., length - pad :], torch.zeros(1, 1, pad))

    def test_the_masked_tail_assertion_is_not_vacuous(self, exported, net_g):
        """故障注入 — マスクを全 1 に差し替えると末尾が 0 でなくなる。

        h はパディング列にも値が入っているので、マスクが外れれば値が漏れる。これが
        無いと上のテストは「そもそも常に 0 になる形」でも緑になりうる。
        """
        out, _ = exported
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        module = export_sbv2.DurationPredictorGraph(net_g.dp)
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            leaked = module(tensors["input.h"], torch.ones(1, 1, length), tensors["input.g"])

        assert not torch.equal(leaked[..., length - pad :], torch.zeros(1, 1, pad))

    def test_regeneration_is_byte_identical(self, exported, tmp_path):
        """固定 seed なので同じ環境なら生成物がバイト一致する（golden の前提）。"""
        out, _ = exported
        again = tmp_path / export_sbv2.TARGET_DP
        export_sbv2.export_dp(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


class TestWindowSizeGate:
    """窓幅の門は net_g **全体**を走査する（実重み不要 — 合成モジュールで層別に踏む）。

    実重みは全層が同じ窓幅なので、「先頭 1 層だけ見る門」でも実重みのテストは緑になる。
    層ごとに違う窓幅を持つモデルを作れる合成側でしか、走査範囲の縮みは検出できない。
    """

    @staticmethod
    def _layered(sizes) -> torch.nn.Module:
        """`attn_layers[i].window_size = sizes[i]` を持つだけの入れ子モジュール。"""
        root = torch.nn.Module()
        layers = torch.nn.ModuleList()
        for size in sizes:
            layer = torch.nn.Module()
            layer.window_size = size
            layers.append(layer)
        root.attn_layers = layers
        return root

    def test_a_uniform_expected_window_size_passes(self):
        export_sbv2._assert_window_size(self._layered([export_sbv2.EXPECTED_WINDOW_SIZE] * 3))

    def test_a_mismatch_in_a_later_layer_is_rejected(self):
        """先頭が前提どおりでも、深い層の食い違いで落ちる（走査範囲そのものの検出器）。

        `FlowReverse` は全 coupling × 全層に**同じ表**を配るので、1 層でも窓幅が違えば
        その層は幅の違う埋め込みを黙って読む。先頭 1 枚見の門はこの形を素通りする。
        """
        sizes = [export_sbv2.EXPECTED_WINDOW_SIZE] * 3
        sizes[-1] += 1

        with pytest.raises(ValueError, match="窓幅"):
            export_sbv2._assert_window_size(self._layered(sizes))

    def test_layers_without_relative_attention_are_ignored(self):
        """`window_size = None` の層は対象外（相対位置注意を使わない層がこの形）。"""
        export_sbv2._assert_window_size(self._layered([export_sbv2.EXPECTED_WINDOW_SIZE, None]))

    def test_a_model_without_any_window_size_is_rejected(self):
        """恒真化の門 — 属性名が変わって走査が空振りしたら落ちる。"""
        with pytest.raises(ValueError, match="恒真化"):
            export_sbv2._assert_window_size(torch.nn.Module())


@requires_weights
class TestLoadAsserts:
    def test_a_window_size_mismatch_is_rejected(self):
        """故障注入 — 窓幅が前提とずれた net_g は load 時に落ちる（ADR 0013）。

        窓幅の食い違いは shape エラーにならず黙って誤った埋め込みを読む「沈黙誤値」
        クラスなので、assert が本当に発火することを固定する。
        """
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
        attn = net_g.flow.flows[0].enc.attn_layers[0]
        original = attn.window_size
        attn.window_size = original + 1
        try:
            with pytest.raises(ValueError, match="窓幅"):
                export_sbv2._assert_window_size(net_g)
        finally:
            attn.window_size = original

    def test_a_residual_weight_norm_parameter_is_rejected(self):
        """故障注入 — weight_norm 由来のパラメータが残っていれば落ちる。"""
        module = torch.nn.Module()
        module.register_parameter("weight_v", torch.nn.Parameter(torch.zeros(1)))

        with pytest.raises(ValueError, match="weight_norm"):
            export_sbv2._assert_no_weight_norm(module)

    def test_the_speaker_id_is_range_checked(self):
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]

        with pytest.raises(ValueError, match="話者"):
            export_sbv2.speaker_embedding(net_g, net_g.emb_g.num_embeddings)

    def test_the_speaker_embedding_comes_from_the_real_weights(self):
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
        g = export_sbv2.speaker_embedding(net_g)

        assert g.shape == (1, 512, 1)
        # 実重みから引けていれば零ベクトルではない（未初期化・取り違えの検出）。
        assert float(g.abs().max()) > 0


class TestVerifyExclusivity:
    """パッチのプロセス汚染に対する門（実重み不要 — 引数解析と適用済みフラグだけを見る）。"""

    def test_the_cli_refuses_verify_together_with_an_emit_target(self, monkeypatch):
        """`--verify` と emit の併用は引数解析の時点で拒否される（ADR 0013）。

        併用できると emit 側が先にパッチを当て、`--verify` の「パッチ前の参照」が既に
        パッチ後の値になる — 差が常に 0 になって**偽 PASS** する。
        """
        monkeypatch.setattr(
            sys, "argv", ["export_sbv2.py", "--verify", "front", "--target", "front"]
        )

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2

    def test_the_cli_takes_exactly_one_verify_target(self, monkeypatch):
        """検証は 1 プロセス 1 ターゲット — 併記は引数解析が構造的に受け付けない。

        「MHA パッチ系どうしは排他 / dec の remove は voice とだけ排他 / 丸めは全てと排他」
        という対ごとの排他表を持つ形にすると、表の穴がそのまま偽 PASS になる。値を 1 つ
        取る `--verify` なら**汚染の組み合わせが存在しない**（ADR 0013 の排他規律の実装）。
        """
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--verify", "flow", "voice"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2

    def test_the_cli_refuses_an_unknown_verify_target(self, monkeypatch):
        """dp は検証対象に無い（パッチも前処理も通らず、参照と対象が同一 = 恒真）。"""
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--verify", "dp"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2
        assert set(export_sbv2.VERIFIERS) == set(export_sbv2.TARGETS) - {export_sbv2.TARGET_DP}

    def test_taking_a_reference_after_patching_is_rejected(self, monkeypatch):
        """順序違反そのものの検出（CLI を通らない経路からの誤用も落とす）。"""
        monkeypatch.setattr(patch_sbv2, "_APPLIED", True)

        with pytest.raises(RuntimeError, match="恒真化"):
            export_sbv2.reference_front_outputs(object(), {})

    @pytest.mark.parametrize("where", ["front", "flow", "voice"])
    def test_the_patch_gate_is_shared_by_every_mha_target(self, monkeypatch, where):
        """パッチ汚染の門は 4 ターゲットで**同じ 1 本**（front だけの特権にしない）。"""
        monkeypatch.setattr(patch_sbv2, "_APPLIED", True)

        with pytest.raises(RuntimeError, match="恒真化"):
            export_sbv2._assert_patches_not_applied(f"{where} の参照採取")

    def test_the_dec_gate_points_the_other_way(self):
        """dec の門だけは「導出パラメータが**残っている**こと」を要求する。

        汚染源が `remove_weight_norm`（破壊的に畳む）なので、`patches_applied()` を
        そのまま流用すると remove 済み net_g での参照採取が素通りし、remove 後どうしの
        比較（差 0 が自明）になる。
        """
        removed = torch.nn.Module()
        removed.register_parameter("weight", torch.nn.Parameter(torch.zeros(1)))
        with pytest.raises(RuntimeError, match="恒真化"):
            export_sbv2._assert_weight_norm_present(removed, "dec の参照採取")

        intact = torch.nn.Module()
        intact.register_parameter("weight_v", torch.nn.Parameter(torch.zeros(1)))
        export_sbv2._assert_weight_norm_present(intact, "dec の参照採取")


@requires_weights
class TestVerifyGatePlacement:
    """門が**参照採取の経路上に居る**ことの実測（関数を素通りしないこと）。

    ヘルパ単体のテストは「門が発火すること」しか言えない。検証関数から呼ばれていなければ
    順序違反はそのまま偽 PASS になるので、実際に `verify_*` を汚染状態で呼んで落ちるかを見る。
    """

    @pytest.mark.parametrize(
        "verifier", [export_sbv2.verify_flow, export_sbv2.verify_voice], ids=["flow", "voice"]
    )
    def test_a_patched_process_cannot_take_references(self, monkeypatch, verifier):
        monkeypatch.setattr(patch_sbv2, "_APPLIED", True)

        # 門を特定して照合する（どちらの門で落ちたか分からない `恒真化` 一致だと、
        # dec 側の門で落ちても緑になる）。
        with pytest.raises(RuntimeError, match="パッチ適用後"):
            verifier(MODEL_DIR, cases=[(2, 0)])

    @pytest.mark.parametrize(
        "verifier", [export_sbv2.verify_dec, export_sbv2.verify_voice], ids=["dec", "voice"]
    )
    def test_an_already_removed_dec_cannot_be_the_reference(self, monkeypatch, verifier):
        """1 プロセスで 2 度目の検証を回した形（remove は冪等なので黙って通ってしまう）。"""
        net_g, hps = export_sbv2.load_net_g(MODEL_DIR)
        export_sbv2.ensure_dec_plain(net_g)
        monkeypatch.setattr(export_sbv2, "load_net_g", lambda _model_dir: (net_g, hps))
        # 汚染源を remove だけに絞る（voice はパッチの門も持つので、先にそちらで落ちると
        # 「dec の門が経路上に居ること」を見たことにならない）。
        monkeypatch.setattr(patch_sbv2, "_APPLIED", False)

        with pytest.raises(RuntimeError, match="remove_weight_norm 済み"):
            verifier(MODEL_DIR, cases=[(2, 0)])


class TestSymMaxPerTarget:
    """sym_max のターゲット別既定（ADR 0013 — 取り違えは沈黙する）。"""

    def test_every_target_declares_its_upper_bound(self):
        assert set(export_sbv2.TARGET_SYM_MAX) == set(export_sbv2.TARGETS)

    def test_the_front_family_and_the_flow_family_differ_by_an_order(self):
        """front=512（音素数）と flow=4096（フレーム数）— 桁が違うから取り違えが致命的。"""
        assert export_sbv2.TARGET_SYM_MAX[export_sbv2.TARGET_FRONT] == export_sbv2.SYM_MAX
        assert export_sbv2.TARGET_SYM_MAX[export_sbv2.TARGET_VOICE] == export_sbv2.FLOW_SYM_MAX
        assert export_sbv2.SYM_MAX < export_sbv2.FLOW_SYM_MAX

    def test_a_shared_sym_max_across_targets_is_refused(self, monkeypatch):
        """1 つの `--sym-max` を複数ターゲットへ配る形は拒否する。

        既定が桁違いに違うので、配れば必ずどちらかが誤値になる（しかも沈黙する）。
        """
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--sym-max", "1024"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2


class _TinyDp(nn.Module):
    """`DurationPredictor` の最小の骨格（`forward(h, x_mask, g=…)` と 192/512 チャネル）。

    `DurationPredictorGraph` がそのまま被せられる形なので、実重み無しで
    「dtype ノブが export の端まで通っているか」を回せる。
    """

    def __init__(self) -> None:
        super().__init__()
        self.cond = nn.Conv1d(512, 192, 1)
        self.conv = nn.Conv1d(192, 1, 3, padding=1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert g is not None
        return self.conv((x + self.cond(g)) * x_mask) * x_mask


class _TinyHybrid(nn.Module):
    """i4 混成（適格な linear / conv1d = i4 group32・残り = i8）の最小の骨格。

    `fc` の in 軸を 64 にするのは i4 が端数 group を作らない MUST（ADR 0069 決定 2）のため。
    `odd`（in 軸 48）は**割り切れない linear**で、適格から外れて i8 側へ落ちる枝の検出器 —
    実 net_g では 6 本とも割り切れる（FN4 実測）ので、ここでしか踏めない。`conv` の行長
    （`Cin·K` = 12）も割り切れないので i8 側に残り、「排他に割れているか」が観測できる
    （i8 側が空だと `fake_quant_int8` の対象 0 本になって枝が見えない）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(4, 4, 3)
        self.fc = nn.Linear(64, 4)
        self.odd = nn.Linear(48, 4)


class _TinyI4Census(nn.Module):
    """i4 の適格集合を**過不足なしで数え上げる**ための骨格（枝ごとに落ちる理由を 1 つに絞る）。

    適格 3 本（`fc` / `wide` / `deep`）に対し、非適格 4 本はそれぞれ**理由が 1 つだけ**成立
    するように寸法を選んである（他の条件は全て満たす）— 落ちる理由が重なっていると、
    どれか 1 つの門が消えても集合が変わらず、テストが門を守らなくなる。
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 4)  # 適格（既存の linear 枝・行長 64）
        self.wide = nn.Conv1d(32, 4, kernel_size=1)  # 適格（行長 Cin·K = 32）
        self.deep = nn.Conv1d(16, 4, kernel_size=2)  # 適格（行長 = 16·2 = 32）
        self.narrow = nn.Linear(48, 4)  # 行長 48 が group32 で割り切れない
        self.short = nn.Conv1d(4, 4, kernel_size=3)  # 行長 12 が割り切れない
        self.depthwise = nn.Conv1d(64, 32, kernel_size=1, groups=2)  # 行長 32・groups > 1
        self.up = nn.ConvTranspose1d(4, 32, kernel_size=1)  # 行長 32・型が違う
        self.table = nn.Embedding(4, 32)  # 行長 32・今回のスコープ外（i8 のまま）


class _TinyAllI4(nn.Module):
    """量子化対象が**全て** i4 適格な骨格（実物では `flow` がこの形 — 全部 conv1d）。"""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Conv1d(32, 4, kernel_size=1)
        self.post = nn.Conv1d(16, 4, kernel_size=2)


class _TinyNetG(nn.Module):
    """`export_dp` が触る net_g の面だけ（`dp` と話者埋め込み）。"""

    def __init__(self) -> None:
        super().__init__()
        self.dp = _TinyDp()
        self.emb_g = nn.Embedding(1, 512)


#: dtype 配線テスト用の golden ケース（実重みの 5 本は要らない — 短い 2 本で足りる）。
_TINY_CASES = (("p2", 2, 0), ("padded", 16, 5))


def _weights_are_f16_exact(module: nn.Module) -> bool:
    """全 f32 パラメータが f16 の格子に乗っているか（emit の適格判定と同じ述語）。"""
    return all(
        torch.equal(tensor, tensor.to(torch.float16).to(torch.float32))
        for tensor in module.parameters()
        if tensor.dtype is torch.float32
    )


def _weights_are_int8_exact(module: nn.Module) -> bool:
    """全 weight スロットが `q8 · scale` でビット一致して戻るか（emit の逆変換の門の述語）。

    `scale` は**現在の重みから引き直す** — 量子化後の重みからの scale 再計算が f32 の不動点に
    なる（ADR 0019 の ±127 に閉じた量子化）ので成立する述語で、量子化していない重みはここを
    通らない（掛け忘れの検出器になる）。対象が 1 本も無ければ False（恒真化の防止）。
    """
    checked = 0
    for sub in module.modules():
        axes = {axis for cls, axis in QUANT_CHANNEL_AXES.items() if isinstance(sub, cls)}
        weight = getattr(sub, "weight", None) if axes else None
        if weight is None:
            continue
        scale = channel_scale(weight, next(iter(axes)))
        if not torch.equal(quantize_to_int8(weight, scale).to(torch.float32) * scale, weight):
            return False
        checked += 1
    return checked > 0


class TestWeightDtypeSeries:
    """格納 dtype の系列（ADR 0018 / 0019）— 壊れると**偽 PASS** になる側の規律だけを固定する。

    実重みは使わない（配線の規律はモデルに依らない）。数値そのものの検証は Deno 側の
    E2E（`packages/runtime/tests/e2e_sbv2_test.ts` の f16 / i8 系列）と emit の往復ビット
    一致の門が持つ。
    """

    def test_the_default_output_root_is_a_separate_series_per_dtype(self):
        """MUST: 圧縮系列は別ディレクトリ（ADR 0018 / 0019）— 同居させると f32 の網が掛かる。"""
        roots = {
            dtype: export_sbv2.default_out_root(export_sbv2.DEFAULT_MODEL_DIR, dtype)
            for dtype in export_sbv2.WEIGHT_DTYPES
        }

        assert set(roots) == {"f32", "f16", "i8", "i4"}
        assert len(set(roots.values())) == len(roots)
        # 系列は `outputs/series/` 側（`models/` は配布形だけの場所 — ADR 0037）。
        assert roots == {
            "f32": SERIES_ROOT / "sbv2-FN4",
            "f16": SERIES_ROOT / "sbv2-FN4-f16",
            "i8": SERIES_ROOT / "sbv2-FN4-i8",
            "i4": SERIES_ROOT / "sbv2-FN4-i4",
        }

    def test_the_i4_series_is_stored_as_i8_by_default(self):
        """i4 は**混成**（適格な linear / conv1d だけ i4・残りは i8）— 単一 dtype にはできない。

        i4 の実行経路が `emit.I4_WEIGHT_OPS` 限定（ADR 0069 決定 5 とその追補）である以上、
        既定を i4 にすると適格外（embedding / `ConvTranspose1d` / depthwise conv）が黙って
        f32 で残る。既定は i8 で、適格だけ 1 本単位の override で振るのが唯一の形。
        """
        assert set(export_sbv2.BASE_WEIGHT_DTYPES) == set(export_sbv2.WEIGHT_DTYPES)
        assert export_sbv2.BASE_WEIGHT_DTYPES["i4"] == "i8"

    def test_the_series_name_carries_the_weights_directory_name(self):
        """話者ごとに別系列（綴りを共有すると別話者の資産を黙って上書きする）。"""
        other = export_sbv2.default_out_root(Path("inputs/sbv2/OTHER"), "f32")

        assert other == SERIES_ROOT / "sbv2-OTHER"
        assert other != export_sbv2.default_out_root(export_sbv2.DEFAULT_MODEL_DIR, "f32")

    def test_every_target_takes_the_dtype_knob(self):
        """5 ターゲット全部が系列ノブを持つ（1 本でも欠けると f32 資産が f16 系列に混ざる）。"""
        for target, exporter in export_sbv2.EXPORTERS.items():
            parameter = inspect.signature(exporter).parameters.get("dtype")
            assert parameter is not None, target
            assert parameter.default == "f32", target

    def test_the_cli_refuses_a_compressed_dtype_together_with_verify(self, monkeypatch):
        """MUST: `--dtype` は emit 専用（`--sym-max` と同じ扱い）。

        検証は格納形式を見ない eager 比較で、しかも dec / voice では丸めを
        `remove_weight_norm` の**後**にしか当てられないのに参照は remove の**前**に採る。
        併用を許すと「丸めた側 vs 丸めていない側」の比較になり、`bit_exact` が資産とは
        無関係に False へ落ちて門の意味が消える。
        """
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--verify", "dec", "--dtype", "f16"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2

    def test_the_cli_refuses_i8_together_with_verify(self, monkeypatch):
        """i8 も同じ扱い（圧縮 dtype が増えるたびに例外表を持たない — 一律 emit 専用）。"""
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--verify", "dec", "--dtype", "i8"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2

    def test_f32_leaves_the_weights_untouched(self):
        torch.manual_seed(0)
        module = _TinyDp()
        before = module.conv.weight.clone()

        export_sbv2._fake_quant("f32", module, export_sbv2.TARGET_DP)

        assert torch.equal(module.conv.weight, before)

    def test_f16_rounds_the_module_that_gets_exported(self):
        torch.manual_seed(0)
        module = _TinyDp()
        assert not _weights_are_f16_exact(module), "丸め前から格子に乗っていては検出力が無い"

        export_sbv2._fake_quant("f16", module, export_sbv2.TARGET_DP)

        assert _weights_are_f16_exact(module)

    def test_i8_quantizes_the_module_that_gets_exported(self):
        """i8 は per-channel symmetric（ADR 0019）。台帳のキーは FQN（emit の突合の空間）。"""
        torch.manual_seed(0)
        module = _TinyDp()
        assert not _weights_are_int8_exact(module), "量子化前から格子に乗っていては検出力が無い"

        scales, overrides = export_sbv2._fake_quant("i8", module, export_sbv2.TARGET_DP)

        assert overrides == {}, "i8 単一系列に 1 本単位の格納指定は要らない"
        assert _weights_are_int8_exact(module)
        assert set(scales) == {"cond.weight", "conv.weight"}

    def test_i4_splits_the_eligible_linears_and_the_rest_exclusively(self):
        """MUST: i8 / i4 の対象は排他（`quantize.py` の混成 MUST — 二重丸めは沈黙誤値）。

        どちらにも入らない重みが残る穴も同時に見る（合流台帳が全ての重みを覆っていること）。
        振り分け先は scale の**形**で読める: i4 は group 形 `[チャネル, group 数]`、i8 は
        重みと同 rank の keepdim 形。`odd`（in 軸 48）は i4 の適格から外れて i8 側へ落ちる —
        ここが黙って落ちると「i4 系列なのに対象が痩せた」が観測できない。
        """
        torch.manual_seed(0)
        module = _TinyHybrid()

        scales, overrides = export_sbv2._fake_quant("i4", module, export_sbv2.TARGET_FRONT)

        assert overrides == {"fc.weight": "i4"}
        assert set(scales) == {"fc.weight", "odd.weight", "conv.weight"}
        # in 軸 64 / group 32 なので group 数 2 — keepdim 形（[4, 1]）とは形で区別できる。
        assert list(scales["fc.weight"].shape) == [4, 2]
        assert list(scales["odd.weight"].shape) == [4, 1]
        assert list(scales["conv.weight"].shape) == [4, 1, 1]

    def test_the_i4_target_set_is_linears_plus_dense_conv1ds(self):
        """適格 = 型 × `groups == 1` × 行長の整除（波 J-5b の conv1d 追補）。

        集合を**丸ごと**突き合わせる（部分集合ではなく）— 過剰に拾えば emit が
        fail loudly するが、取りこぼしは「i4 系列なのに対象が痩せた」として黙って通る。
        """
        assert export_sbv2._i4_module_names(_TinyI4Census()) == {"fc", "wide", "deep"}

    def test_i4_rounds_a_conv1d_weight_with_a_rank2_group_scale(self):
        """conv1d の scale は rank2（`[Cout, (Cin·K)/g]`）— i8 の keepdim 形 `[4,1,1]` と別形。

        振り分け先が形で読めるのは linear と同じ流儀で、rank3 の重みに rank3 の scale が
        付いていたら i8 側へ落ちている。
        """
        torch.manual_seed(0)
        module = _TinyI4Census()

        scales, overrides = export_sbv2._fake_quant("i4", module, export_sbv2.TARGET_VOICE)

        assert set(overrides) == {"fc.weight", "wide.weight", "deep.weight"}
        assert list(scales["deep.weight"].shape) == [4, 1]
        assert list(scales["short.weight"].shape) == [4, 1, 1], "非適格 conv は i8 の keepdim 形"
        assert list(scales["up.weight"].shape) == [1, 32, 1], "転置レイアウトの i8 軸は 1"
        assert list(scales["table.weight"].shape) == [4, 1]  # embedding は i8（スコープ外）

    def test_i4_skips_the_i8_pass_when_every_target_is_eligible(self):
        """全対象が i4 適格な系列でも落ちない（`fake_quant_int8` の「対象 0 本」を踏まない）。

        0 本が**数え上げの結果**であることを呼ぶ前に確かめる設計で、i8 側を「i4 に居ない」の
        否定だけで書いていると conv1d だけで構成された系列（実物の `flow`）で export が死ぬ。
        """
        torch.manual_seed(0)
        module = _TinyAllI4()

        scales, overrides = export_sbv2._fake_quant("i4", module, export_sbv2.TARGET_FLOW)

        assert set(scales) == {"pre.weight", "post.weight"}
        assert set(overrides) == set(scales), "全本が i4 明示指定へ載る"

    def test_rounding_before_removing_the_weight_norm_leaves_the_grid(self):
        """MUST の裏付け: 丸めは `remove_weight_norm` の**後**（`ensure_dec_plain` の予告）。

        weight_norm が有効な間の `weight` は導出値で、丸めるべき実効重みは
        `g · v / ‖v‖`。先に `weight_g` / `weight_v` を丸めても実効重みは f16 の格子に
        乗らない — 配信サイズだけ減って数値は別物になる。検出器は emit の往復ビット一致の門
        （`test_emit.py::test_an_unrounded_eligible_weight_fails_loudly`）で、この述語が
        その門が見ているものそのもの。
        """
        torch.manual_seed(0)
        conv = torch.nn.utils.weight_norm(nn.Conv1d(4, 4, 3))
        rounded = export_sbv2.round_weights_to_f16(conv)
        assert rounded.parameters >= 2, "weight_g / weight_v を丸めたことの確認"

        torch.nn.utils.remove_weight_norm(conv)

        assert not _weights_are_f16_exact(conv)

    @pytest.mark.parametrize("dtype", ["f32", "f16", "i8"])
    def test_the_dtype_reaches_the_container_and_the_golden(self, monkeypatch, tmp_path, dtype):
        """dtype が export の端（格納宣言）まで通り、golden が**丸め後**の重みで採れている。

        golden を丸めより前に採ると「golden は元値・実行は丸め値」になり、E2E の差が
        量子化誤差と実装誤差の合成に化ける（tolerance を緩める方向にしか効かないので、
        緑のまま検出力だけが落ちる）。ここでは export 後のモジュール（= 丸め済み）を
        そのまま流し直して golden とビット一致することで、その順序を固定する。
        """
        torch.manual_seed(0)
        net_g = _TinyNetG()
        hps = SimpleNamespace(version="tiny")
        monkeypatch.setattr(export_sbv2, "load_net_g", lambda _model_dir: (net_g, hps))
        out = tmp_path / export_sbv2.TARGET_DP

        summary = export_sbv2.export_dp(tmp_path, out, cases=_TINY_CASES, dtype=dtype)

        graph = verify_model(out / export_sbv2.MODEL_FILE)
        stored = {name: init.storage.dtype for name, init in graph.initializers.items()}
        compressed = {name for name, storage in stored.items() if storage in ("f16", "i8")}
        if dtype == "f32":
            assert compressed == set(), stored
            assert summary["compressed_tensors"] == 0
        else:
            # 適格 = conv の重み 2 本だけ（bias は常に f32 — ADR 0006）。
            assert {stored[name] for name in compressed} == {dtype}, stored
            assert len(compressed) == 2, stored
            assert summary["compressed_tensors"] == 2
        if dtype == "i8":
            # companion scale の宣言（ADR 0019）— 無いと値が復元できない。
            scale_keys = {graph.initializers[name].storage.scale for name in compressed}
            assert None not in scale_keys, stored
            assert len(scale_keys) == len(compressed), "scale キーが重みごとに分かれていない"
            assert summary["scale_bytes"] > 0
        else:
            assert summary["scale_bytes"] == 0

        assert summary["dtype"] == dtype

        module = export_sbv2.DurationPredictorGraph(net_g.dp)
        assert _weights_are_f16_exact(module) == (dtype == "f16")
        assert _weights_are_int8_exact(module) == (dtype == "i8")
        for name, _length, _pad in _TINY_CASES:
            tensors = _io_tensors(out, name)
            with torch.no_grad():
                fresh = module(tensors["input.h"], tensors["input.x_mask"], tensors["input.g"])
            assert torch.equal(fresh, tensors["output.0"]), name


@requires_weights
class TestInt8IdempotenceOnRealWeights:
    """i8 fake-quant が**実重み**でも冪等（ADR 0019 の「±127 に閉じた量子化は f32 の不動点」）。

    `_fake_quant` は「実効重みが確定した後・golden の採取前」という順序にだけ依存する設計だが、
    dec / voice は同じ net_g 系統から採り（voice のラッパは flow と dec を 1 度に丸める）、
    ターゲットを跨いだ再適用が起こりうる形をしている。合成モデルの冪等テスト
    （`test_quantize.py`）は重みの分布が違うので、**実重みの amax 分布での不動点性**は別に
    押さえる — 破れていれば「1 度目の丸めで採った golden」と「2 度目の丸めで書いた格納値」が
    食い違い、emit の逆変換ビット一致の門が落ちる（= 沈黙はしないが、原因が分からない赤になる）。
    """

    @pytest.mark.parametrize("target", ["dec", "voice"])
    def test_reapplying_the_quantization_is_bit_identical(self, target):
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
        export_sbv2.ensure_dec_plain(net_g)
        # 重みしか見ないのでパッチは当てない（当てるとプロセス全域が汚れ、同じプロセスで
        # 参照を採る他のテストを巻き込む — ADR 0013 の排他規律）。
        module = net_g.dec if target == "dec" else patch_sbv2.Sbv2Voice(net_g)

        first = fake_quant_int8(module)
        snapshot = {name: module.get_parameter(name).detach().clone() for name in first.scales}
        second = fake_quant_int8(module)

        assert first.modules == len(snapshot) > 0
        for name, before in snapshot.items():
            assert torch.equal(module.get_parameter(name), before), name
            assert torch.equal(second.scales[name], first.scales[name]), f"{name} の scale"


#: 検証レポートの 1 ケース行（`P=   2 pad=0: ... bit_exact=True`）。
_VERIFY_CASE_LINE = re.compile(r"^[PT]=\s*\d+ pad=\d+:")


@pytest.fixture(scope="module")
def verify_report():
    """`--verify <target>` を**別プロセス**で 1 回だけ回し、要約を返す（ターゲットごと）。

    MUST: 別プロセスで回す — パッチと `remove_weight_norm` はプロセス全域を汚すので、
    同じプロセスで参照を採る他のテスト（export フィクスチャ）と混ぜると同値検証が
    恒真化して偽 PASS する。これは ADR 0013 の排他規律そのものの実行。
    """
    cache: dict[str, dict] = {}

    def run(target: str) -> dict:
        if target not in cache:
            completed = subprocess.run(
                [sys.executable, "-m", "sbv2.export", "--verify", target],
                cwd=str(REPO_ROOT / "tools" / "export-recipes"),
                capture_output=True,
                text=True,
                check=True,
            )
            lines = completed.stdout.splitlines()
            worst = [line for line in lines if line.startswith("worst maxdiff = ")]
            exact = [line for line in lines if line.startswith("bit_exact all = ")]
            assert len(worst) == 1 and len(exact) == 1, completed.stdout
            cache[target] = {
                "cases": [line for line in lines if _VERIFY_CASE_LINE.match(line)],
                "worst": float(worst[0].split("=", 1)[1]),
                "bit_exact": exact[0].split("=", 1)[1].strip() == "True",
                "stdout": completed.stdout,
            }
        return cache[target]

    return run


@requires_weights
class TestEquivalenceVerification:
    """参照実装との eager 同値（**別プロセス**で実測する — 排他規律の実行）。"""

    @pytest.mark.parametrize(
        ("target", "expected_cases"),
        [
            (export_sbv2.TARGET_FRONT, export_sbv2.VERIFY_CASES),
            (export_sbv2.TARGET_FLOW, export_sbv2.FLOW_VERIFY_CASES),
            (export_sbv2.TARGET_DEC, export_sbv2.FLOW_VERIFY_CASES),
            (export_sbv2.TARGET_VOICE, export_sbv2.FLOW_VERIFY_CASES),
        ],
    )
    def test_every_verify_case_is_reported(self, verify_report, target, expected_cases):
        """検証ケースが黙って減っていない（表を縮めると同値の主張の範囲が縮む）。"""
        report = verify_report(target)

        assert len(report["cases"]) == len(expected_cases), report["stdout"]

    def test_the_front_patch_is_equivalent_to_the_reference(self, verify_report):
        # 実測 2.02e-5（P=512 の logw_sdp）。閾値はその 5 倍で、パッチが「同じ式を別の順序で
        # 組む」以上のことをしたら（= 実装が変わったら）O(1) の差になってここで落ちる。
        report = verify_report(export_sbv2.TARGET_FRONT)

        assert report["worst"] < 1e-4, report["stdout"]

    def test_the_flow_patch_is_equivalent_to_the_reference(self, verify_report):
        """FlowReverse（表入力・split→slice・exp(−logs) 畳み込み）が原 reverse 経路と同値。

        実測 worst 1.43e-6（T=512）。front より 1 桁小さいのは spline を通らないから。
        閾値はその約 7 倍。
        """
        report = verify_report(export_sbv2.TARGET_FLOW)

        assert report["worst"] < 1e-5, report["stdout"]

    def test_removing_the_weight_norm_is_bit_exact_in_every_case(self, verify_report):
        """**recon §6 の未検証事項を閉じる**: remove_weight_norm のビット一致。

        recon 時点では 1 ケース（z=(1,192,50)）の実測しか無く、実効重み `g·v/‖v‖` が f32 で
        厳密に再現されることはスペック保証ではない、と明記されていた。ここで golden より
        広い全 10 ケースを突合し、**maxdiff 0 かつビット一致**であることを確定させる。
        ビット一致まで要求するのは、`0.0` と `-0.0` の取り違え（差 0 だがビットは別）を
        通さないため。
        """
        report = verify_report(export_sbv2.TARGET_DEC)

        assert report["bit_exact"], report["stdout"]
        assert report["worst"] == 0.0, report["stdout"]

    def test_the_voice_fusion_is_equivalent_to_the_reference_chain(self, verify_report):
        """融合 voice が「未パッチ flow + weight_norm 有効 dec」の合成と同値。

        実測 worst 1.25e-6（T=203）。**融合しても flow 単体の差（1.43e-6）を超えない** —
        dec は conv の縮約に入力の微小差を埋もれさせる側で、誤差を増幅しない。
        """
        report = verify_report(export_sbv2.TARGET_VOICE)

        assert report["worst"] < 1e-5, report["stdout"]


@pytest.fixture(scope="module")
def exported_front(tmp_path_factory):
    """一時ディレクトリへ front を export する（リポジトリの成果物には触らない）。

    MUST: このフィクスチャはパッチをプロセス全域に当てる。パッチ前の参照が要る検証は
    別プロセス（`TestFrontVerify`）で回すこと。
    """
    out = tmp_path_factory.mktemp("sbv2-front") / export_sbv2.TARGET_FRONT
    return out, export_sbv2.export_front(MODEL_DIR, out)


@requires_weights
class TestFrontExport:
    def test_the_container_passes_the_full_verification(self, exported_front):
        out, _ = exported_front

        verify_model(out / export_sbv2.MODEL_FILE)

    def test_the_graph_declares_the_wrapper_argument_names(self, exported_front):
        out, summary = exported_front
        graph = verify_model(out / export_sbv2.MODEL_FILE)

        assert [spec.name for spec in graph.inputs] == list(export_sbv2.FRONT_INPUT_ORDER)
        assert [spec.shape for spec in graph.inputs] == [
            [1, "P"],
            [1, 1, "P"],
            [1, "P"],
            [1, "P"],
            [1, 1024, "P"],
            [1, 256],
            [1, 512, 1],
            [1, 2, "P"],
        ]
        assert summary["symbols"] == ["P"]
        assert summary["outputs"] == len(export_sbv2.FRONT_OUTPUT_NAMES)

    def test_the_graph_stays_inside_the_current_contract(self, exported_front):
        _, summary = exported_front

        # 波 6 の前提: front は**語彙を広げずに**通る（波 3〜5 で足した op で足りる）。
        assert set(summary["ops"]) <= set(EMITTABLE_OPS)

    def test_the_relative_position_tables_are_baked_at_pmax(self, exported_front):
        """相対位置注意の表が Pmax 焼き込み + `sym_prefix_slice` になっている（ADR 0013）。

        表が焼き込まれていないと、二次 shape 式（2P−1 等）が残るか、実行時に P 依存の
        添字計算が生き残る。**表は 3 本**（key 側の idx_k / valid、value 側の idx_v）で、
        いずれも入力は記号を含まない静的形でなければならない。
        """
        out, _ = exported_front
        graph = verify_model(out / export_sbv2.MODEL_FILE)
        sources = []
        for node in graph.nodes:
            if node.op != "sym_prefix_slice":
                continue
            assert node.attrs["sym"] == "P"
            shape = graph.values[node.ins[0]].shape
            assert all(isinstance(dim, int) for dim in shape), shape
            sources.append((tuple(shape), graph.values[node.ins[0]].dtype))

        pmax = export_sbv2.SYM_MAX
        window = 2 * export_sbv2.EXPECTED_WINDOW_SIZE + 1
        assert sorted(sources) == sorted(
            [
                ((1, 1, pmax, pmax), "i32"),  # idx_k = clamp(w + j − i, 0, 2w)
                ((1, 1, pmax, pmax), "f32"),  # valid = [|j − i| ≤ w]
                ((1, 1, pmax, window), "i32"),  # idx_v = i + c（value 側）
            ]
        )

    def test_the_baked_tables_stay_within_the_expected_budget(self, exported_front):
        """焼き込み定数は約 2MB（ADR 0013 の「front は実害なし」の根拠を数で固定する）。

        flow（Tmax=4096）で同じ方式を採ると O(Tmax²) で 134MB になる — front の
        Pmax=512 だからこそ焼き込みで済む、という裁定の前提がここで壊れれば見える。
        """
        out, _ = exported_front
        graph = verify_model(out / export_sbv2.MODEL_FILE)
        baked = sum(
            math.prod(graph.values[name].shape) * 4
            for name in graph.initializers
            if name.startswith("const_")
        )

        assert baked < 3 * 1024 * 1024, baked

    def test_the_io_files_follow_the_naming_convention(self, exported_front):
        out, summary = exported_front

        expected = [
            f"{export_sbv2.IO_PREFIX}{name}{export_sbv2.IO_SUFFIX}"
            for name, _, _ in export_sbv2.GOLDEN_CASES
        ]
        assert summary["io"] == expected
        for name in expected:
            with safe_open(str(out / name), framework="pt") as handle:
                keys = set(handle.keys())
            assert keys == {
                *(f"input.{declared}" for declared in export_sbv2.FRONT_INPUT_ORDER),
                *(f"output.{index}" for index in range(len(export_sbv2.FRONT_OUTPUT_NAMES))),
            }

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported_front, case):
        out, _ = exported_front
        name, length, _ = case
        graph = verify_model(out / export_sbv2.MODEL_FILE)
        tensors = _io_tensors(out, name)

        for spec in graph.inputs:
            actual = list(tensors[f"input.{spec.name}"].shape)
            declared = [
                length if isinstance(dim, str) and parse_dim(dim).sym == "P" else dim
                for dim in spec.shape
            ]
            assert actual == declared, spec.name
        # logw_sdp / logw_dp は [1,1,P]、m_p / logs_p は [1,192,P]。
        assert [list(tensors[f"output.{index}"].shape) for index in range(4)] == [
            [1, 1, length],
            [1, 1, length],
            [1, 192, length],
            [1, 192, length],
        ]

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_outputs_reproduce_a_fresh_eager_forward(
        self, exported_front, front_module, case
    ):
        """格納された期待値が、同じ入力に対する（パッチ後の）torch 出力とビット一致する。"""
        out, _ = exported_front
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = front_module(
                *(tensors[f"input.{declared}"] for declared in export_sbv2.FRONT_INPUT_ORDER)
            )

        for index, value in enumerate(fresh):
            assert torch.equal(value, tensors[f"output.{index}"]), index

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported_front, front_module, case):
        """故障注入 — 期待値を 1 要素だけ壊すと上のテストが赤になる（恒真化の門番）。"""
        out, _ = exported_front
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = front_module(
                *(tensors[f"input.{declared}"] for declared in export_sbv2.FRONT_INPUT_ORDER)
            )
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh[0], corrupted)

    def test_the_padded_case_zeroes_the_masked_tail_of_every_output(self, exported_front):
        """マスク乗算が 4 本の出力すべてに効いている（パディング列は厳密に 0）。"""
        out, _ = exported_front
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        tensors = _io_tensors(out, name)

        for index in range(len(export_sbv2.FRONT_OUTPUT_NAMES)):
            tail = tensors[f"output.{index}"][..., length - pad :]
            assert torch.equal(tail, torch.zeros_like(tail)), index

    def test_the_masked_tail_assertion_is_not_vacuous(self, exported_front, front_module):
        """故障注入 — マスクを全 1 に差し替えると末尾が 0 でなくなる。

        x / tone / bert はパディング列にも値が入っているので、マスクが外れれば値が漏れる。
        """
        out, _ = exported_front
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        tensors = _io_tensors(out, name)
        args = [tensors[f"input.{declared}"] for declared in export_sbv2.FRONT_INPUT_ORDER]
        args[export_sbv2.FRONT_INPUT_ORDER.index("x_mask")] = torch.ones(1, 1, length)
        with torch.no_grad():
            leaked = front_module(*args)

        for index, value in enumerate(leaked):
            tail = value[..., length - pad :]
            assert not torch.equal(tail, torch.zeros_like(tail)), index

    def test_regeneration_is_byte_identical(self, exported_front, tmp_path):
        """固定 seed なので同じ環境なら生成物がバイト一致する（golden の前提）。

        z_noise を含む乱数がグローバル seed に依存していると、ここで落ちる。
        """
        out, _ = exported_front
        again = tmp_path / export_sbv2.TARGET_FRONT
        export_sbv2.export_front(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


@pytest.fixture(scope="module")
def front_module(exported_front):
    """golden と同じ計算をする eager モジュール（パッチは export 側で適用済み）。"""
    net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
    patch_sbv2.apply_all_patches()
    return patch_sbv2.Sbv2Front(net_g)


@requires_weights
class TestFrontInputs:
    """front の golden 入力（実重み由来の資産を使うので requires_weights）。"""

    def test_the_noise_matches_what_the_reference_draws_internally(self):
        """`make_noise` が参照実装の内部 randn と**同じ列**（同値検証の前提そのもの）。

        参照側は `torch.manual_seed(...)` 経由のグローバル生成器、golden 側は `Generator`
        経由。両者が同じ列を返すことは torch の実装事情なので、ここで明示的に固定する
        （割れると同値検証が「別のノイズどうしの比較」になり、差が大きく出て気づく側では
        あるが、原因が分からない形で赤くなる）。
        """
        for length in (2, 37):
            torch.manual_seed(export_sbv2._noise_seed(length))
            reference = torch.randn(1, 2, length) * export_sbv2.NOISE_SCALE

            assert torch.equal(export_sbv2.make_noise(length), reference), length

    def test_the_style_vector_comes_from_the_real_asset(self):
        style = export_sbv2.style_vector(MODEL_DIR)

        assert style.shape == (1, 256)
        assert style.dtype == torch.float32
        assert float(style.abs().max()) > 0

    def test_an_out_of_range_style_id_is_rejected(self):
        with pytest.raises(ValueError, match="スタイル"):
            export_sbv2.style_vector(MODEL_DIR, 10_000)

    def test_the_case_inputs_are_keyed_by_the_graph_input_names(self):
        g = torch.zeros(1, 512, 1)
        style = torch.zeros(1, 256)
        built = export_sbv2.build_front_cases(g, style)

        assert [name for name, _ in built] == [name for name, _, _ in export_sbv2.GOLDEN_CASES]
        for name, args in built:
            # x_lengths は参照実装専用で golden には載せない（IR の入力ではない）。
            assert set(args) == set(export_sbv2.FRONT_INPUT_ORDER), name

    def test_the_padded_case_carries_values_in_the_masked_tail(self):
        """マスク経路の検出器になる形（パディング列が全 0 だと漏れを検出できない）。"""
        _, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        inputs = export_sbv2.front_inputs(length, pad, torch.zeros(1, 512, 1), torch.zeros(1, 256))

        assert int(inputs["x"][..., length - pad :].abs().sum()) > 0
        assert float(inputs["bert"][..., length - pad :].abs().sum()) > 0
        assert float(inputs["z_noise"][..., length - pad :].abs().sum()) > 0


@pytest.fixture(scope="module")
def exported_flow(tmp_path_factory):
    """一時ディレクトリへ flow を export する（パッチをプロセス全域に当てる）。"""
    out = tmp_path_factory.mktemp("sbv2-flow") / export_sbv2.TARGET_FLOW
    return out, export_sbv2.export_flow(MODEL_DIR, out)


@pytest.fixture(scope="module")
def exported_dec(tmp_path_factory):
    out = tmp_path_factory.mktemp("sbv2-dec") / export_sbv2.TARGET_DEC
    return out, export_sbv2.export_dec(MODEL_DIR, out)


@pytest.fixture(scope="module")
def exported_voice(tmp_path_factory):
    out = tmp_path_factory.mktemp("sbv2-voice") / export_sbv2.TARGET_VOICE
    return out, export_sbv2.export_voice(MODEL_DIR, out)


@pytest.fixture(scope="module")
def flow_module(exported_flow):
    """golden と同じ計算をする eager モジュール（パッチは export 側で適用済み）。"""
    net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
    patch_sbv2.apply_all_patches()
    return patch_sbv2.FlowReverse(net_g)


@pytest.fixture(scope="module")
def dec_module(exported_dec):
    net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
    export_sbv2.ensure_dec_plain(net_g)
    return net_g.dec


@pytest.fixture(scope="module")
def voice_module(exported_voice):
    net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
    patch_sbv2.apply_all_patches()
    export_sbv2.ensure_dec_plain(net_g)
    return patch_sbv2.Sbv2Voice(net_g)


@requires_weights
class TestFlowExport:
    """flow — 相対位置表を**グラフ入力**へ昇格した唯一の系統（voice と共通）。"""

    def test_the_container_passes_the_full_verification(self, exported_flow):
        out, _ = exported_flow

        verify_model(out / export_sbv2.MODEL_FILE)

    def test_the_graph_declares_the_wrapper_argument_names(self, exported_flow):
        out, summary = exported_flow
        graph = verify_model(out / export_sbv2.MODEL_FILE)

        assert [spec.name for spec in graph.inputs] == list(export_sbv2.FLOW_INPUT_ORDER)
        assert [spec.shape for spec in graph.inputs] == [
            [1, 192, "T"],
            [1, 1, "T"],
            [1, 512, 1],
            ["T", "T"],
            ["T", "T"],
        ]
        # 表は i64 → i32（境界正規化 — ADR 0009）と f32。
        assert [spec.dtype for spec in graph.inputs][3:] == ["i32", "f32"]
        assert summary["symbols"] == ["T"]

    def test_the_graph_stays_inside_the_current_contract(self, exported_flow):
        _, summary = exported_flow

        # 波 7 の前提: flow も**語彙を広げずに**通る。rank-4 matmul が core 分解で
        # view→bmm→view に落ちるか（recon §6 の未検証事項）はここが答え — bmm が出て
        # matmul が出なければ落ちている。
        assert set(summary["ops"]) <= set(EMITTABLE_OPS)
        assert "bmm" in summary["ops"]
        assert "matmul" not in summary["ops"]

    def test_the_relative_position_tables_are_inputs_not_constants(self, exported_flow):
        """`(T,T)` 表が焼き込まれていない（ADR 0013 の裁定そのもの）。

        焼き込むと sym_max=4096 で `idx_k` + `valid` だけで 134MB。残ってよい焼き込みは
        value 側の `idx_v` `(Tmax, 2w+1)` 1 本だけで、こちらは 4096×9 で 150KB 級。
        """
        out, _ = exported_flow
        graph = verify_model(out / export_sbv2.MODEL_FILE)
        sources = [
            tuple(graph.values[node.ins[0]].shape)
            for node in graph.nodes
            if node.op == "sym_prefix_slice"
        ]

        tmax = export_sbv2.FLOW_SYM_MAX
        window = 2 * export_sbv2.EXPECTED_WINDOW_SIZE + 1
        assert sources == [(1, 1, tmax, window)]

    def test_the_baked_constants_stay_far_below_the_promoted_tables(self, exported_flow):
        """焼き込み定数の総量が 1MB 未満（「昇格しないと 134MB」の対照）。"""
        out, _ = exported_flow
        graph = verify_model(out / export_sbv2.MODEL_FILE)
        baked = sum(
            math.prod(graph.values[name].shape) * 4
            for name in graph.initializers
            if name.startswith("const_")
        )

        assert baked < 1024 * 1024, baked

    def test_the_io_files_follow_the_naming_convention(self, exported_flow):
        out, summary = exported_flow

        expected = [
            f"{export_sbv2.IO_PREFIX}{name}{export_sbv2.IO_SUFFIX}"
            for name, _, _ in export_sbv2.GOLDEN_CASES
        ]
        assert summary["io"] == expected
        for name in expected:
            with safe_open(str(out / name), framework="pt") as handle:
                keys = set(handle.keys())
            assert keys == {
                *(f"input.{declared}" for declared in export_sbv2.FLOW_INPUT_ORDER),
                "output.0",
            }

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported_flow, case):
        out, _ = exported_flow
        name, length, _ = case
        graph = verify_model(out / export_sbv2.MODEL_FILE)
        tensors = _io_tensors(out, name)

        for spec in graph.inputs:
            actual = list(tensors[f"input.{spec.name}"].shape)
            declared = [
                length if isinstance(dim, str) and parse_dim(dim).sym == "T" else dim
                for dim in spec.shape
            ]
            assert actual == declared, spec.name
        assert list(tensors["output.0"].shape) == [1, 192, length]

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_output_reproduces_a_fresh_eager_forward(
        self, exported_flow, flow_module, case
    ):
        out, _ = exported_flow
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = flow_module(*(tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER))

        assert torch.equal(fresh, tensors["output.0"])

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported_flow, flow_module, case):
        """故障注入 — 期待値を 1 要素だけ壊すと上のテストが赤になる（恒真化の門番）。"""
        out, _ = exported_flow
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = flow_module(*(tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER))
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh, corrupted)

    def test_a_shifted_table_changes_the_golden(self, exported_flow, flow_module):
        """故障注入 — 表を 1 ずらすと golden と食い違う（実重み・全 24 層での確認）。

        合成注意 1 層での検出は `test_patch.py` が持つ。ここは「グラフ入力として
        運ばれた表が本当に 24 層すべての注意で使われている」ことの実測 — 表が途中で
        捨てられて in-graph 構築に落ちていたら、ずらしても値が動かない。
        """
        out, _ = exported_flow
        name, _, _ = next(case for case in export_sbv2.GOLDEN_CASES if case[1] > 2 * 4)
        tensors = _io_tensors(out, name)
        args = [tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER]
        index = export_sbv2.FLOW_INPUT_ORDER.index("idx_k")
        args[index] = torch.clamp(args[index].to(torch.int64) + 1, 0, 2 * 4)
        with torch.no_grad():
            shifted = flow_module(*args)

        assert not torch.equal(shifted, tensors["output.0"])

    def test_the_padded_case_zeroes_the_masked_tail(self, exported_flow):
        """`y_mask` 乗算が効いていればパディング列の出力は厳密に 0。

        coupling は 2 段目以降で両半分とも masked 側を通るので、最終出力は 192 チャネル
        全部が末尾 0 になる（片側だけ 0 なら結合の順序を取り違えている）。
        """
        out, _ = exported_flow
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        output = _io_tensors(out, name)["output.0"]
        tail = output[..., length - pad :]

        assert torch.equal(tail, torch.zeros_like(tail))

    def test_the_masked_tail_assertion_is_not_vacuous(self, exported_flow, flow_module):
        """故障注入 — マスクを全 1 に差し替えると末尾が 0 でなくなる。"""
        out, _ = exported_flow
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        tensors = _io_tensors(out, name)
        args = [tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER]
        args[export_sbv2.FLOW_INPUT_ORDER.index("y_mask")] = torch.ones(1, 1, length)
        with torch.no_grad():
            leaked = flow_module(*args)
        tail = leaked[..., length - pad :]

        assert not torch.equal(tail, torch.zeros_like(tail))

    def test_regeneration_is_byte_identical(self, exported_flow, tmp_path):
        out, _ = exported_flow
        again = tmp_path / export_sbv2.TARGET_FLOW
        export_sbv2.export_flow(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


@requires_weights
class TestDecExport:
    """dec — パッチ不要で、前処理は `remove_weight_norm` だけ。"""

    def test_the_container_passes_the_full_verification(self, exported_dec):
        out, _ = exported_dec

        verify_model(out / export_sbv2.MODEL_FILE)

    def test_the_graph_declares_the_generator_argument_names(self, exported_dec):
        out, summary = exported_dec
        graph = verify_model(out / export_sbv2.MODEL_FILE)

        # ラッパを置いていないので `Generator.forward(x, g)` の引数名がそのまま出る。
        assert [spec.name for spec in graph.inputs] == list(export_sbv2.DEC_INPUT_ORDER)
        assert [spec.shape for spec in graph.inputs] == [[1, 192, "T"], [1, 512, 1]]
        assert summary["symbols"] == ["T"]

    def test_the_graph_stays_inside_the_current_contract(self, exported_dec):
        _, summary = exported_dec

        # 波 5（conv 族拡張）が dec のために足した op がそのまま出る。ops が増えたら
        # 「知らないうちに別の経路が生えた」ということなので fail loudly。
        assert set(summary["ops"]) <= set(EMITTABLE_OPS)
        assert set(summary["ops"]) == {
            "add",
            "conv1d",
            "conv_transpose1d",
            "div",
            "leaky_relu",
            "tanh",
        }

    def test_the_output_length_is_the_upsampling_product(self, exported_dec):
        """出力長が厳密に 512·T（ConvTranspose の `pad=(k−u)//2` が効いている）。"""
        out, _ = exported_dec
        graph = verify_model(out / export_sbv2.MODEL_FILE)
        shape = graph.values[graph.outputs[0]].shape

        assert shape[:2] == [1, 1]
        assert parse_dim(shape[2]).coeff == 512
        assert parse_dim(shape[2]).offset == 0

    def test_the_io_files_follow_the_naming_convention(self, exported_dec):
        out, summary = exported_dec

        for name in summary["io"]:
            with safe_open(str(out / name), framework="pt") as handle:
                keys = set(handle.keys())
            assert keys == {"input.x", "input.g", "output.0"}

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported_dec, case):
        out, _ = exported_dec
        name, length, _ = case
        tensors = _io_tensors(out, name)

        assert list(tensors["input.x"].shape) == [1, 192, length]
        assert list(tensors["output.0"].shape) == [1, 1, 512 * length]

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_output_reproduces_a_fresh_eager_forward(
        self, exported_dec, dec_module, case
    ):
        out, _ = exported_dec
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = dec_module(tensors["input.x"], g=tensors["input.g"])

        assert torch.equal(fresh, tensors["output.0"])

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported_dec, dec_module, case):
        out, _ = exported_dec
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = dec_module(tensors["input.x"], g=tensors["input.g"])
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh, corrupted)

    def test_removing_the_weight_norm_is_idempotent(self):
        """`ensure_dec_plain` の冪等性（融合 voice が同じ net_g で二度通る形）。"""
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
        export_sbv2.ensure_dec_plain(net_g)
        weight = net_g.dec.conv_pre.weight.clone()

        export_sbv2.ensure_dec_plain(net_g)

        assert torch.equal(net_g.dec.conv_pre.weight, weight)

    def test_a_dec_with_weight_norm_is_refused_by_the_assertion(self):
        """故障注入 — remove を通さないまま assert すれば落ちる（順序制約の門番）。"""
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]

        with pytest.raises(ValueError, match="weight_norm"):
            export_sbv2._assert_no_weight_norm(net_g.dec)

    def test_regeneration_is_byte_identical(self, exported_dec, tmp_path):
        out, _ = exported_dec
        again = tmp_path / export_sbv2.TARGET_DEC
        export_sbv2.export_dec(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


@requires_weights
class TestVoiceExport:
    """voice — flow + dec の融合。**このターゲットが通ると SBV2 全チェーンが揃う**。"""

    def test_the_container_passes_the_full_verification(self, exported_voice):
        out, _ = exported_voice

        verify_model(out / export_sbv2.MODEL_FILE)

    def test_the_graph_takes_the_flow_inputs_and_returns_audio(self, exported_voice):
        out, summary = exported_voice
        graph = verify_model(out / export_sbv2.MODEL_FILE)

        assert [spec.name for spec in graph.inputs] == list(export_sbv2.FLOW_INPUT_ORDER)
        assert summary["outputs"] == 1
        assert parse_dim(graph.values[graph.outputs[0]].shape[2]).coeff == 512

    def test_the_fusion_covers_both_halves(self, exported_voice, exported_flow, exported_dec):
        """融合グラフが flow と dec の op を**両方**持つ（片方が落ちていない）。

        ノード数も両者の和に近いはず — 大きく足りなければどちらかの段が消えている。
        """
        _, voice = exported_voice
        _, flow = exported_flow
        _, dec = exported_dec

        assert set(voice["ops"]) == set(flow["ops"]) | set(dec["ops"])
        assert voice["nodes"] >= flow["nodes"] + dec["nodes"] - 10

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported_voice, case):
        out, _ = exported_voice
        name, length, _ = case
        tensors = _io_tensors(out, name)

        assert list(tensors["input.z_p"].shape) == [1, 192, length]
        assert list(tensors["input.idx_k"].shape) == [length, length]
        assert list(tensors["output.0"].shape) == [1, 1, 512 * length]

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_output_reproduces_a_fresh_eager_forward(
        self, exported_voice, voice_module, case
    ):
        out, _ = exported_voice
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = voice_module(*(tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER))

        assert torch.equal(fresh, tensors["output.0"])

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported_voice, voice_module, case):
        out, _ = exported_voice
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = voice_module(*(tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER))
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh, corrupted)

    def test_the_fused_output_matches_the_two_stage_goldens(
        self, exported_voice, exported_flow, dec_module
    ):
        """融合の golden が「flow の golden → dec」と一致する（融合の意味の実測）。

        入力は flow と voice で共通（`build_flow_cases`）なので、flow の期待出力 z に
        `y_mask` を掛けて dec に通せば voice の期待出力になるはず。段の順序取り違えや
        `z * y_mask` の抜けはここで落ちる。
        """
        voice_out, _ = exported_voice
        flow_out, _ = exported_flow
        name, _, _ = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        flow_tensors = _io_tensors(flow_out, name)
        voice_tensors = _io_tensors(voice_out, name)
        with torch.no_grad():
            staged = dec_module(
                flow_tensors["output.0"] * flow_tensors["input.y_mask"],
                g=flow_tensors["input.g"],
            )

        assert torch.equal(staged, voice_tensors["output.0"])

    def test_regeneration_is_byte_identical(self, exported_voice, tmp_path):
        out, _ = exported_voice
        again = tmp_path / export_sbv2.TARGET_VOICE
        export_sbv2.export_voice(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


def _io_tensors(out_dir, case_name) -> dict[str, torch.Tensor]:
    path = out_dir / f"{export_sbv2.IO_PREFIX}{case_name}{export_sbv2.IO_SUFFIX}"
    with safe_open(str(path), framework="pt") as handle:
        # safe_open は Mapping ではないので keys() が唯一の列挙手段。
        return {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
