"""`deberta/export.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（README 参照）。ここで固定するのは、壊れると**偽 PASS** になる側の
規律だけ:

- 系列が dtype ごとに分かれること（f32 の網が i8 資産へ黙って掛からない）
- `--act-quant` が `--dtype i8` 無しでは通らないこと（f32 資産の鏡像は再現不能）
- 鏡像 io の prefix が Deno 側の通常ケース列挙（`io.` の startsWith）に**引っかからない**こと
- 通常の golden io が**フックなし**で採られること（掛けたままだと w8 E2E の期待値が汚染される）
- 記号次元の出所記録が読み手（`sbv2.distribution`）と同じ綴り・同じ欄で書かれること
  （DeBERTa の Tmax は artifact から読めないので、記録が唯一の運び手）
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from deberta import export as export_deberta
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


class TinyHybrid(nn.Module):
    """i4 混成（適格な linear / embedding = i4 group32・残り = i8）の最小の骨格。

    量子化軸を 64 にするのは i4 が端数 group を作らない MUST（ADR 0069 決定 2）のため。
    `narrow` は量子化軸 16（< group 32）で**割り切れない**適格外の linear で、i8 側へ落ちる —
    i8 側の対象が空だと `fake_quant_int8` が fail loudly して「排他に割れているか」を
    観測できないので、この 1 本が i8 側の住人も兼ねる。
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 64)
        self.fc = nn.Linear(64, 4)
        self.narrow = nn.Linear(16, 4)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        embedded = self.embed(input_ids)
        return (self.fc(embedded) + self.narrow(embedded[..., :16]),)


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

        assert set(roots) == set(export_deberta.WEIGHT_DTYPES) == {"f32", "i8", "i4"}
        assert len(set(roots.values())) == len(roots)

    def test_f16_is_not_offered(self):
        """f16 は SBV2 系列と一体で決める（タスク #30）— ここで先取りしない。"""
        assert "f16" not in export_deberta.WEIGHT_DTYPES

    def test_the_i4_series_is_stored_as_i8_by_default(self):
        """i4 は**混成**（適格な linear / embedding だけ i4・残りは i8）— 単一 dtype の i4 系列は
        作れない。

        i4 の実行経路が linear / embedding の重みスロット限定（ADR 0069 決定 5）である以上、
        既定を i4 にすると conv や group 長で割り切れない重みが黙って f32 で残る。既定は i8 で、
        適格分だけ 1 本単位の override で振るのが唯一の形。
        """
        assert set(export_deberta.BASE_WEIGHT_DTYPES) == set(export_deberta.WEIGHT_DTYPES)
        assert export_deberta.BASE_WEIGHT_DTYPES["i4"] == "i8"


class TestExportProvenance:
    """記号次元の出所記録 — DeBERTa の Tmax は **artifact から読めない**ので記録だけが運ぶ。

    front / voice は上限を `sym_prefix_slice` の焼き込み定数として持つが、DeBERTa は相対位置の
    添字表を ADR 0045 波 3 でグラフ入力へ昇格させたので、上限を運ぶ定数がグラフに 1 本も残って
    いない（出荷 4 コンテナの実測 — `sym_prefix_slice` 0 本）。配布側の `maxTokens` は定数で
    焼かれるため、非既定の `--sym-max` で採った系列は export も配布も緑のまま宣言だけが嘘になる。
    """

    def test_the_spelling_matches_the_reader(self):
        """MUST: 読み手（`sbv2.distribution`）と同じ綴り。

        `deberta` は SBV2 の消費者ではないので向こうを import しない（配布形の text_encoder 席は
        資産の共有であって結合ではない）。写しを 2 つ持つ以上、独立に動く形にはしない。
        """
        from sbv2.distribution import EXPORT_PROVENANCE_FILE

        assert export_deberta.EXPORT_PROVENANCE_FILE == EXPORT_PROVENANCE_FILE

    def test_it_records_the_variant_and_the_symbolic_maximum(self, tmp_path):
        """記録の欄は読み手が突き合わせる 2 つ（`target` / `sym_max`）そのもの。"""
        from sbv2.distribution import SBV2_MAX_TOKENS, SBV2_TEXT_ENCODER_VARIANT

        export_deberta._write_export_provenance(
            SBV2_TEXT_ENCODER_VARIANT, export_deberta.SYM_MAX, tmp_path
        )

        record = json.loads(
            (tmp_path / export_deberta.EXPORT_PROVENANCE_FILE).read_text(encoding="utf-8")
        )
        assert record == {"target": SBV2_TEXT_ENCODER_VARIANT, "sym_max": SBV2_MAX_TOKENS}

    def test_it_carries_a_non_default_symbolic_maximum(self, tmp_path):
        """`--sym-max` の逸脱が記録に残る（配布側の突合はこの値を見る）。"""
        export_deberta._write_export_provenance("sbv2-22layer", 256, tmp_path)

        record = json.loads(
            (tmp_path / export_deberta.EXPORT_PROVENANCE_FILE).read_text(encoding="utf-8")
        )
        assert record["sym_max"] == 256

    def test_the_default_symbolic_maximum_agrees_with_the_distribution_declaration(self):
        """台本の既定と配布形の `maxTokens` は同じ数（記録が既定を名乗れる前提）。"""
        from sbv2.distribution import SBV2_MAX_TOKENS

        assert export_deberta.SYM_MAX == SBV2_MAX_TOKENS


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

        assert export_deberta._fake_quant("f32", wrapper) == ({}, {})
        assert torch.equal(wrapper.fc.weight, before)

    def test_i8_rounds_to_per_channel_representable_values_keyed_by_fqn(self):
        """MUST: 台帳のキーは export する module から見た FQN（emit の突合はここで決まる）。"""
        torch.manual_seed(0)
        wrapper = TinyText()

        scales, overrides = export_deberta._fake_quant("i8", wrapper)

        assert overrides == {}, "i8 単一系列に 1 本単位の格納指定は要らない"
        assert "fc.weight" in scales, "キーが export 対象の FQN 空間に無い"
        scale = scales["fc.weight"]
        assert list(scale.shape) == [4, 1]
        assert torch.equal(torch.round(wrapper.fc.weight / scale) * scale, wrapper.fc.weight)

    def test_i4_splits_the_eligible_modules_and_the_rest_exclusively(self):
        """MUST: i8 / i4 の対象は排他（`quantize.py` の混成 MUST — 二重丸めは沈黙誤値）。

        どちらにも入らない重みが残る穴も同時に見る（合流台帳が全ての重みを覆っていること）。
        scale の**形**で振り分け先が読める: i4 は group 形 `[チャネル, group 数]`、i8 は
        重みと同 rank の keepdim 形。
        """
        torch.manual_seed(0)
        wrapper = TinyHybrid()

        scales, overrides = export_deberta._fake_quant("i4", wrapper)

        # embedding も i4 席（ADR 0069 決定 5 の embedding 追補）— 語彙表 `[V,D]` の D 軸で group。
        assert overrides == {"fc.weight": "i4", "embed.weight": "i4"}
        assert set(scales) == {"fc.weight", "embed.weight", "narrow.weight"}
        # 量子化軸 64 / group 32 なので group 数 2 — keepdim 形（[4, 1]）とは形で区別できる。
        assert list(scales["fc.weight"].shape) == [4, 2]
        assert list(scales["embed.weight"].shape) == [8, 2]
        # 割り切れない 1 本は i8 のまま（構成ごと落とさず対象から外す）
        assert list(scales["narrow.weight"].shape) == [4, 1]

    def test_a_weight_that_the_group_size_does_not_divide_stays_in_the_i8_side(self):
        """i4 適格の正本は emit 側の規則（型 × 整除）— 綴りではなく実測から引く。"""
        torch.manual_seed(0)
        wrapper = TinyHybrid()

        assert export_deberta._i4_module_names(wrapper) == {"fc", "embed"}

    def test_a_table_that_is_never_looked_up_is_excluded_from_the_i4_side(self, monkeypatch):
        """`nn.Embedding` の器でも表引きされない重みは重みスロットの消費がゼロ = 圧縮の適格外。

        i4 に振ると emit の明示指定の門が「適格でない」で落ちるので、対象集合の側で外す
        （実物は DeBERTa の相対位置表 — `deberta.patch` が切り出して linear に通す）。
        """
        torch.manual_seed(0)
        wrapper = TinyHybrid()
        monkeypatch.setattr(export_deberta, "NON_LOOKUP_EMBEDDINGS", frozenset({"embed"}))

        assert export_deberta._i4_module_names(wrapper) == {"fc"}
        # 外した表は i8 側（丸めの担い手が変わらない = 従来の数値のまま）
        scales, overrides = export_deberta._fake_quant("i4", wrapper)
        assert overrides == {"fc.weight": "i4"}
        assert list(scales["embed.weight"].shape) == [8, 1]


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
