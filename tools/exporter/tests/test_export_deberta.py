"""`export_deberta.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（README 参照）。ここで固定するのは、壊れると**偽 PASS** になる側の
規律だけ:

- 系列が dtype ごとに分かれること（f32 の網が i8 資産へ黙って掛からない）
- `--act-quant` が `--dtype i8` 無しでは通らないこと（f32 資産の鏡像は再現不能）
- 鏡像 io の prefix が Deno 側の通常ケース列挙（`io.` の startsWith）に**引っかからない**こと
- 通常の golden io が**フックなし**で採られること（掛けたままだと w8 E2E の期待値が汚染される）
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

import export_deberta
from karume.pipeline import export_to_file


class TinyText(nn.Module):
    """`HiddenStatesWrapper` の最小の骨格（`(input_ids, attention_mask) → タプル`）。

    適格 linear（`in_features % 4 == 0`）を 1 本だけ持つので、活性フックの有無が出力に出る。
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x = input_ids.to(torch.float32) * attention_mask.to(torch.float32)
        return (self.fc(x),)


#: `_write_io` はケースを `(名前, グラフ入力名 → テンソル)` で受ける（実物は 4 入力だが、
#: 引く順は `graph.inputs` から来るので tiny な 2 入力でも同じ経路が通る）。
CASES = (
    (
        "case0",
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            "attention_mask": torch.ones(1, 4, dtype=torch.int64),
        },
    ),
)


@pytest.fixture
def exported(tmp_path):
    """tiny なラッパを 1 本 export して `(wrapper, graph, out_dir)` を返す。"""
    torch.manual_seed(0)
    wrapper = TinyText()
    example = tuple(CASES[0][1].values())
    graph = export_to_file(wrapper, example, tmp_path / export_deberta.MODEL_FILE)
    return wrapper, graph, tmp_path


class TestSeries:
    def test_the_default_output_root_is_a_separate_series_per_dtype(self):
        """MUST: 圧縮系列は別ディレクトリ（ADR 0019）— 同居させると f32 の網が消える。"""
        roots = export_deberta.DEFAULT_OUT_ROOTS

        assert set(roots) == set(export_deberta.WEIGHT_DTYPES) == {"f32", "i8"}
        assert len(set(roots.values())) == len(roots)

    def test_f16_is_not_offered(self):
        """f16 は SBV2 系列と一体で決める（タスク #30）— ここで先取りしない。"""
        assert "f16" not in export_deberta.WEIGHT_DTYPES


class TestActQuantCli:
    @pytest.mark.parametrize("dtype", ["f32"])
    def test_act_quant_requires_i8_weights(self, monkeypatch, dtype):
        """活性 i8 は i8 常駐重みの linear にしか効かない（ADR 0025 決定 1）。"""
        monkeypatch.setattr(
            "sys.argv", ["export_deberta.py", "--dtype", dtype, "--act-quant", "--layers", "2"]
        )

        with pytest.raises(SystemExit, match="--dtype i8"):
            export_deberta.main()


class TestMirrorIoPrefix:
    def test_the_mirror_prefix_does_not_match_the_plain_enumeration(self):
        """MUST: `io.` 始まりにすると鏡像が w8 の golden として拾われる（Deno 側の列挙規則）。"""
        mirror = f"{export_deberta.ACT_IO_PREFIX}case0{export_deberta.IO_SUFFIX}"

        assert not mirror.startswith(export_deberta.IO_PREFIX)
        assert mirror.endswith(export_deberta.IO_SUFFIX)


class TestFakeQuant:
    def test_f32_leaves_the_weights_untouched(self):
        torch.manual_seed(0)
        wrapper = TinyText()
        before = wrapper.fc.weight.clone()

        assert export_deberta._fake_quant("f32", wrapper) == {}
        assert torch.equal(wrapper.fc.weight, before)

    def test_i8_rounds_to_per_channel_representable_values_keyed_by_fqn(self):
        """MUST: 台帳のキーは export する module から見た FQN（emit の突合はここで決まる）。"""
        torch.manual_seed(0)
        wrapper = TinyText()

        scales = export_deberta._fake_quant("i8", wrapper)

        assert "fc.weight" in scales, "キーが export 対象の FQN 空間に無い"
        scale = scales["fc.weight"]
        assert list(scale.shape) == [4, 1]
        assert torch.equal(torch.round(wrapper.fc.weight / scale) * scale, wrapper.fc.weight)


class TestMirrorIo:
    def test_mirror_io_is_written_under_its_own_prefix(self, exported):
        wrapper, graph, out_dir = exported

        written, attached = export_deberta._write_mirror_io(wrapper, graph, CASES, out_dir)

        assert attached == 1, "適格 linear は 1 本（in_features=4）"
        assert written == [f"{export_deberta.ACT_IO_PREFIX}case0{export_deberta.IO_SUFFIX}"]

    def test_mirror_io_differs_from_the_plain_golden(self, exported):
        """鏡像が通常 io と同じ数なら、フックが空振りしている（0 本でも例外にならない経路）。"""
        wrapper, graph, out_dir = exported
        export_deberta._write_io(wrapper, graph, CASES, out_dir)
        export_deberta._write_mirror_io(wrapper, graph, CASES, out_dir)

        plain = load_file(out_dir / f"{export_deberta.IO_PREFIX}case0{export_deberta.IO_SUFFIX}")
        mirror = load_file(
            out_dir / f"{export_deberta.ACT_IO_PREFIX}case0{export_deberta.IO_SUFFIX}"
        )

        assert torch.equal(plain["input.input_ids"], mirror["input.input_ids"])
        assert not torch.equal(plain["output.0"], mirror["output.0"])

    def test_the_hooks_are_detached_so_the_plain_golden_stays_clean(self, exported):
        """MUST: 掛けたまま通常 io を採ると w8 E2E の期待値が活性量子化ごと汚染される。"""
        wrapper, graph, out_dir = exported
        export_deberta._write_io(wrapper, graph, CASES, out_dir)
        before = load_file(out_dir / f"{export_deberta.IO_PREFIX}case0{export_deberta.IO_SUFFIX}")

        export_deberta._write_mirror_io(wrapper, graph, CASES, out_dir)
        export_deberta._write_io(wrapper, graph, CASES, out_dir)
        after = load_file(out_dir / f"{export_deberta.IO_PREFIX}case0{export_deberta.IO_SUFFIX}")

        assert torch.equal(before["output.0"], after["output.0"])

    def test_zero_eligible_linears_fails_loudly(self, tmp_path):
        """0 本のまま鏡像を採ると「w8a8 のつもりで w8 の数」になる（ADR 0006 の診断常設）。"""

        class NoLinear(nn.Module):
            def forward(
                self, input_ids: torch.Tensor, attention_mask: torch.Tensor
            ) -> tuple[torch.Tensor, ...]:
                return (input_ids.to(torch.float32) + attention_mask.to(torch.float32),)

        wrapper = NoLinear()
        example = tuple(CASES[0][1].values())
        graph = export_to_file(wrapper, example, tmp_path / export_deberta.MODEL_FILE)

        with pytest.raises(SystemExit, match="適格 linear が 0 本"):
            export_deberta._write_mirror_io(wrapper, graph, CASES, tmp_path)
