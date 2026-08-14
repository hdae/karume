"""`export_vowel_detector.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（既存 `export_siglip2.py` / `export_birefnet.py` のテストと同じ規律）。
ここは **tiny な合成重み**（{@link tiny_module} — 寸法だけ小さくした同じ `Crnn`）で同じ経路を
一周し、壊れると**偽 PASS** になる側の規律を固定する:

- 写したモデル定義が上流の state_dict と同じ鍵・同じ形を持つこと（実重みが無い環境で
  「写しのずれ」を落とせる唯一の門 — {@link TestTranscription}）
- emit した initializer が state_dict と**バイト一致**すること、および重み由来でない
  initializer が紛れないこと（{@link TestCheckpointBytes}）
- 長さが 20ms 格子に乗ること（奇数長は末尾に半端フレームを作る）
- golden の合成ケースが**互いに違う**こと（同じ入力を 2 度使うと sanity が恒真）
- `_sanity` が確率質量の**順序**を見ること（値域検査も自己一致も恒真）
- `--verify` が emit しないこと
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import export_vowel_detector as vd
from karume.paths import SERIES_ROOT
from karume.pipeline import export_to_file

#: tiny な合成重みの寸法（特徴 83 次元だけは {@link vd.build_cases} と揃える）。
TINY_HIDDEN = 4
TINY_GRU_HIDDEN = 3

#: tiny な export で使う長さ（`MIN_LENGTH` より上で、GRU の展開が数十ノードに収まる）。
TINY_LENGTH = 8

#: 上流 `training/src/vowel_detector/crnn.py` + 学習済み `.pt` が持つ 22 本の重み
#: （鍵と形）。**写しの契約表**で、写しが構造ごとずれたらここで落ちる。
UPSTREAM_PARAMETERS: dict[str, tuple[int, ...]] = {
    "conv.0.weight": (160, 83, 5),
    "conv.0.bias": (160,),
    "conv.2.weight": (160, 160, 3),
    "conv.2.bias": (160,),
    "gru.weight_ih_l0": (384, 160),
    "gru.weight_hh_l0": (384, 128),
    "gru.bias_ih_l0": (384,),
    "gru.bias_hh_l0": (384,),
    "gru.weight_ih_l0_reverse": (384, 160),
    "gru.weight_hh_l0_reverse": (384, 128),
    "gru.bias_ih_l0_reverse": (384,),
    "gru.bias_hh_l0_reverse": (384,),
    "gru.weight_ih_l1": (384, 256),
    "gru.weight_hh_l1": (384, 128),
    "gru.bias_ih_l1": (384,),
    "gru.bias_hh_l1": (384,),
    "gru.weight_ih_l1_reverse": (384, 256),
    "gru.weight_hh_l1_reverse": (384, 128),
    "gru.bias_ih_l1_reverse": (384,),
    "gru.bias_hh_l1_reverse": (384,),
    "head.weight": (8, 256),
    "head.bias": (8,),
}

#: 上流のパラメタ総数（`crnn_epoch3.pt` の実測）。
UPSTREAM_PARAMETER_COUNT = 664_744


def _probabilities(pau: float, vowel: float) -> torch.Tensor:
    """平均 P(pau) と平均母音質量が指定値になる `[1, 2, 8]` のロジット。

    softmax を通した後の質量で `_sanity` が判定するので、log を取って戻す
    （残りは N / cons の 2 クラスへ均等に配る）。
    """
    rest = (1.0 - pau - vowel) / 2.0
    row = [vowel / vd.VOWEL_COUNT] * vd.VOWEL_COUNT + [rest, pau, rest]
    return torch.log(torch.tensor([[row, row]]))


def _pooled(silence_pau: float, voiced_vowel: float) -> dict[str, torch.Tensor]:
    """4 ケース分のロジット（無音の P(pau) と有声の母音質量だけを動かす）。"""
    return {
        vd.SILENCE_CASE: _probabilities(silence_pau, 0.3),
        vd.VOICED_CASE: _probabilities(0.02, voiced_vowel),
        "noise": _probabilities(0.03, 0.1),
        "ramp": _probabilities(0.01, 0.2),
    }


@pytest.fixture
def tiny_module() -> vd.Crnn:
    """寸法だけ小さくした同じ `Crnn`（実重みは要らない）。"""
    torch.manual_seed(0)
    return vd.Crnn(hidden=TINY_HIDDEN, gru_hidden=TINY_GRU_HIDDEN).eval()


@pytest.fixture
def exported(tmp_path: Path, tiny_module: vd.Crnn):
    """tiny なモジュールを 1 本 export して `(module, graph, out_dir)` を返す。"""
    example = vd.build_cases(TINY_LENGTH)[0][1]
    graph = export_to_file(tiny_module, (example,), tmp_path / vd.MODEL_FILE, symbol_names=())
    return tiny_module, graph, tmp_path


class TestTranscription:
    def test_the_transcribed_definition_has_the_upstream_parameters(self) -> None:
        """MUST: 写しの鍵と形が上流と一致する（ずれると実重みが読めない）。"""
        state_dict = vd.Crnn().state_dict()

        assert {name: tuple(tensor.shape) for name, tensor in state_dict.items()} == (
            UPSTREAM_PARAMETERS
        )
        assert sum(tensor.numel() for tensor in state_dict.values()) == UPSTREAM_PARAMETER_COUNT

    def test_the_output_grid_is_half_the_input(self, tiny_module: vd.Crnn) -> None:
        """conv の stride 2 が 10ms を 20ms へ畳む（後処理の frame_sec 0.02 の根拠）。"""
        features = vd.build_cases(TINY_LENGTH)[0][1]

        with torch.no_grad():
            logits = tiny_module(features)

        assert tuple(logits.shape) == (1, TINY_LENGTH // 2, len(vd.CLASSES))

    def test_the_class_table_matches_the_upstream_order(self) -> None:
        """クラスの並びは後処理と共有の規約（pau の位置がずれると sanity が別物を見る）。"""
        assert vd.CLASSES == ("a", "i", "u", "e", "o", "N", "pau", "cons")
        assert vd.CLASSES[vd.PAU_INDEX] == "pau"
        assert vd.CLASSES[: vd.VOWEL_COUNT] == ("a", "i", "u", "e", "o")


class TestSeriesLayout:
    def test_the_default_output_dir_is_a_series(self) -> None:
        """系列出力は `outputs/series/` 配下（配布形の `models/` ではない — karume.paths）。"""
        assert vd.default_out_dir(vd.DEFAULT_CKPT, vd.DEFAULT_LENGTH).parent == SERIES_ROOT

    def test_each_length_gets_its_own_series(self) -> None:
        """MUST: 長さごとに別系列 — 綴りを共有すると先の資産が黙って上書きされる。"""
        first = vd.default_out_dir(vd.DEFAULT_CKPT, 200)
        second = vd.default_out_dir(vd.DEFAULT_CKPT, 500)

        assert first != second
        assert first.name.endswith("-t200")
        assert second.name.endswith("-t500")

    def test_each_checkpoint_gets_its_own_series(self) -> None:
        """MUST: epoch 違いも別系列（同じ席へ書くと先の重みが消える）。"""
        other = vd.MODELS_ROOT / "crnn_epoch2.pt"

        assert vd.default_out_dir(other, 200) != vd.default_out_dir(vd.DEFAULT_CKPT, 200)


class TestLength:
    @pytest.mark.parametrize("length", [201, 3, 0, -2])
    def test_a_length_off_the_grid_fails_loudly(self, length: int) -> None:
        """奇数長は末尾に 10ms 分しか持たない半端フレームを作る（20ms 格子が崩れる）。"""
        with pytest.raises(SystemExit, match="20ms 格子"):
            vd.assert_length(length)

    @pytest.mark.parametrize("length", [vd.MIN_LENGTH, 200, 1000])
    def test_a_length_on_the_grid_passes(self, length: int) -> None:
        vd.assert_length(length)


class TestCheckpoint:
    def test_a_missing_checkpoint_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="チェックポイントが見つからない"):
            vd.load_checkpoint(tmp_path / "absent.pt")

    def test_an_unexpected_wrapper_fails_loudly(self, tmp_path: Path) -> None:
        """MUST: 外側の形が変わったら止める（黙って別の鍵を state_dict と見なさない）。"""
        path = tmp_path / "ckpt.pt"
        torch.save({"state_dict": {}, "epoch": 0}, path)

        with pytest.raises(SystemExit, match="外側ラッパの鍵"):
            vd.load_checkpoint(path)

    def test_it_loads_on_cpu(self, tmp_path: Path, tiny_module: vd.Crnn) -> None:
        """MUST: `map_location="cpu"`（上流の学習台本は CUDA で保存する）。"""
        path = tmp_path / "ckpt.pt"
        torch.save({vd.STATE_DICT_KEY: tiny_module.state_dict(), "epoch": 3}, path)

        state_dict = vd.load_checkpoint(path)

        assert set(state_dict) == set(tiny_module.state_dict())
        assert all(tensor.device.type == "cpu" for tensor in state_dict.values())

    def test_a_shape_mismatch_is_caught_by_strict_loading(self, tiny_module: vd.Crnn) -> None:
        """MUST: 写しのずれは `strict=True` が落とす（黙って一部だけ読まない）。"""
        with pytest.raises(RuntimeError, match=r"size mismatch|Missing key"):
            vd.load_module(tiny_module.state_dict())


class TestCheckpointBytes:
    def test_every_checkpoint_tensor_is_byte_identical(self, exported) -> None:
        module, _graph, out_dir = exported

        matched = vd.assert_checkpoint_bytes(out_dir / vd.MODEL_FILE, module.state_dict())

        assert matched == len(module.state_dict())

    def test_a_changed_byte_fails_loudly(self, exported) -> None:
        """MUST: 値が変わったら落とす（変換は「読んで書くだけ」— 値は変わらない）。"""
        module, _graph, out_dir = exported
        state_dict = dict(module.state_dict())
        tampered = state_dict["head.bias"].clone()
        tampered[0] += 1.0
        state_dict["head.bias"] = tampered

        with pytest.raises(AssertionError, match="バイト列が一致しない"):
            vd.assert_checkpoint_bytes(out_dir / vd.MODEL_FILE, state_dict)

    def test_a_missing_initializer_fails_loudly(self, exported, tmp_path: Path) -> None:
        module, _graph, _out_dir = exported
        stripped = tmp_path / "stripped.safetensors"
        save_file({"conv.0.bias": module.state_dict()["conv.0.bias"]}, str(stripped))

        with pytest.raises(AssertionError, match="initializer に無い"):
            vd.assert_checkpoint_bytes(stripped, module.state_dict())

    def test_a_weight_that_is_not_in_the_checkpoint_fails_loudly(self, exported) -> None:
        """MUST: 重みが別名で入る形を落とす（畳み込み定数だけが `const.` を名乗れる）。"""
        module, _graph, out_dir = exported
        state_dict = dict(module.state_dict())
        del state_dict["head.bias"]

        with pytest.raises(AssertionError, match="重み由来でない initializer"):
            vd.assert_checkpoint_bytes(out_dir / vd.MODEL_FILE, state_dict)


class TestGoldenCases:
    def test_cases_have_the_declared_shape_and_dtype(self) -> None:
        for name, features in vd.build_cases(TINY_LENGTH):
            assert tuple(features.shape) == (1, TINY_LENGTH, vd.FEATURE_DIM), name
            assert features.dtype is torch.float32, name

    def test_the_dsp_dimensions_stay_in_the_measured_range(self) -> None:
        """MUST: 実発話のレンジを外れた特徴で golden を採らない（別の分布の突合になる）。"""
        for name, features in vd.build_cases(TINY_LENGTH):
            voicing, log_energy, zcr = (features[0, 0, vd.N_MELS + i].item() for i in range(3))
            assert 0.0 <= voicing <= 1.0, name
            assert -1.0 <= log_energy <= 0.0, name
            assert 0.0 <= zcr <= 1.0, name

    def test_every_case_is_a_different_input(self) -> None:
        """MUST: 同じ特徴を 2 度使わない — sanity の順序が恒真になる。"""
        cases = vd.build_cases(TINY_LENGTH)

        for index, (name, features) in enumerate(cases):
            for other_name, other in cases[index + 1 :]:
                assert not torch.equal(features, other), f"{name} と {other_name} が同一"

    def test_the_sanity_cases_exist(self) -> None:
        names = {name for name, _ in vd.build_cases(TINY_LENGTH)}

        assert {vd.SILENCE_CASE, vd.VOICED_CASE} <= names


class TestWriteIo:
    def test_writes_one_file_per_case_with_the_declared_keys(self, exported) -> None:
        module, graph, out_dir = exported
        cases = vd.build_cases(TINY_LENGTH)

        written, logits = vd._write_io(module, graph, cases, out_dir)

        assert written == [f"{vd.IO_PREFIX}{name}{vd.IO_SUFFIX}" for name, _ in cases]
        tensors = load_file(str(out_dir / written[0]))
        assert set(tensors) == {f"{vd.INPUT_PREFIX}{vd.INPUT_NAME}", f"{vd.OUTPUT_PREFIX}0"}
        assert tuple(logits[vd.SILENCE_CASE].shape) == (1, TINY_LENGTH // 2, len(vd.CLASSES))

    def test_more_than_one_graph_output_fails_loudly(self, tmp_path: Path) -> None:
        """MUST: 出力はロジット 1 本 — 2 本目が生えると io の位置規約が黙ってずれる。"""

        class TwoOutputs(vd.Crnn):
            def forward(self, features: torch.Tensor):  # type: ignore[override]
                logits = super().forward(features)
                return logits, logits * 2.0

        torch.manual_seed(0)
        module = TwoOutputs(hidden=TINY_HIDDEN, gru_hidden=TINY_GRU_HIDDEN).eval()
        cases = vd.build_cases(TINY_LENGTH)
        graph = export_to_file(module, (cases[0][1],), tmp_path / vd.MODEL_FILE, symbol_names=())

        with pytest.raises(AssertionError, match="ロジットは 1 本"):
            vd._write_io(module, graph, cases, tmp_path)


class TestSanity:
    def test_passes_when_both_orders_hold(self) -> None:
        result = vd._sanity(_pooled(silence_pau=0.2, voiced_vowel=0.7))

        assert result["pau_mean"][vd.SILENCE_CASE] > result["pau_mean"][vd.VOICED_CASE]
        assert result["vowel_mass_mean"][vd.VOICED_CASE] > result["vowel_mass_mean"]["noise"]

    def test_fails_loudly_when_silence_is_not_the_most_paused(self) -> None:
        """MUST: 無音で pau が立たないなら、重みのレイアウトか入力の届き方が壊れている。"""
        with pytest.raises(AssertionError, match=r"平均 P\(pau\) が最大でない"):
            vd._sanity(_pooled(silence_pau=0.01, voiced_vowel=0.7))

    def test_fails_loudly_when_voiced_is_not_the_most_vowel_like(self) -> None:
        with pytest.raises(AssertionError, match="平均母音質量が最大でない"):
            vd._sanity(_pooled(silence_pau=0.2, voiced_vowel=0.05))

    def test_fails_loudly_when_every_case_collapses_to_one_output(self) -> None:
        """MUST: 入力が効いていない形を落とす（4 本が同じ値なら何も検証していない）。"""
        same = _probabilities(0.2, 0.3)
        pooled = dict.fromkeys((vd.SILENCE_CASE, vd.VOICED_CASE, "noise", "ramp"), same)

        with pytest.raises(AssertionError, match="入力が効いていない"):
            vd._sanity(pooled)


class TestCli:
    def test_verify_does_not_emit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            vd,
            "verify_decomposition",
            lambda _ckpt, _length: (
                seen.append("verify")
                or [
                    {
                        "stage": "gru-decomposition",
                        "claim": "bit-exact",
                        "bit_exact": True,
                        "maxdiff": {},
                    }
                ]
            ),
        )
        monkeypatch.setattr(
            vd, "export_series", lambda *_a, **_kw: pytest.fail("--verify で emit された")
        )

        vd.main(["--verify"])

        assert seen == ["verify"]

    def test_without_verify_it_emits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            vd, "export_series", lambda *_a, **_kw: seen.append("emit") or {"dir": "x"}
        )
        monkeypatch.setattr(
            vd, "verify_decomposition", lambda *_a: pytest.fail("emit で --verify が走った")
        )

        vd.main([])

        assert seen == ["emit"]

    def test_the_output_dir_follows_the_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MUST: `--out` 未指定なら系列は長さに追随する（固定だと上書きになる）。"""
        seen: list[Path] = []
        monkeypatch.setattr(
            vd, "export_series", lambda _ckpt, out, _length: seen.append(out) or {"dir": str(out)}
        )

        vd.main(["--length", "500"])

        assert seen == [vd.default_out_dir(vd.DEFAULT_CKPT, 500)]
