"""BiRefNet 系配布形のモデルカード描画（`birefnet.card`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。**例外は解像度**（`imageWidth` / `imageHeight`）と**席**: 実行資源の
実測（checkpoint ごとの `BirefnetCheckpoint.resources`）は解像度で引き、測った席が manifest に
無ければ描かないので、偽 manifest も実測のある解像度と席を持つ。

manifest からの導出（表・数・使い方）は `birefnet/tests/test_distribution.py` の
`TestBirefnetModelCard` / `TestBirefnetResolutionFamily` が**組み立て 1 周ぶん**で見る。ここが
持つのはカード側にしか無い門 — テンプレートの pipeline 固有性と、帰属表に無い checkpoint を
描かないこと、実測していない解像度を描かないこと、そして案内するロード入口。
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest

from _shared.paths import REPO_ROOT
from birefnet.card import (
    BIREFNET_CHECKPOINTS,
    BIREFNET_MATTE_WEIGHTS,
    BIREFNET_POSTPROCESS_DESCRIPTIONS,
    BIREFNET_SUPPORTED_PIPELINE,
    BIREFNET_UPSTREAM,
    render_birefnet_model_card,
)

#: 使い方スニペットに綴られるリポ ID（pipeline の宣言から dist が渡す）。
REPO = "hdae/fake-repo"

#: このリポが配る重み（pipeline 席が渡す軸 — manifest には無い）。
CHECKPOINT = "hr"

#: 既定のモデル名 = 既定の解像度。
MODEL = "1024"

#: checkpoint → 実物の配布形の席（先頭が既定 — `birefnet.distribution.birefnet_seat_dtypes` の値。
#: 規則そのものは `birefnet/tests/test_distribution.py` が見る）。偽 manifest もこの席で組む —
#: カードは測った席が無いモデルも、上流の格納より広い席も描かない。
SEATS: dict[str, tuple[str, ...]] = {"hr": ("f16",), "lucida": ("f32", "f16")}

#: 別 family の pipeline 契約（**綴りをそのまま持つ** — recipe 間のコード結合を作らないため）。
#: 2 つの実在テンプレートが互いの manifest を拒むことは `sbv2/tests/test_card.py` が
#: 両向きで見る。ここが要るのは「自分の契約以外を拒む」という 1 方向だけ。
FOREIGN_PIPELINE = "siglip2/1"

#: ロード側が受理する後処理の手順の閉じた集合の宣言（TS の `MATTE_PROCEDURES`）。
MATTE_SOURCE = REPO_ROOT / "packages/models/src/birefnet/matte.ts"

#: `MATTE_PROCEDURES` の宣言の 1 行（`export const MATTE_PROCEDURES = [...] as const;`）。
MATTE_PROCEDURES_DECLARATION = re.compile(
    r"^export const MATTE_PROCEDURES = \[(?P<body>[^\]]*)\] as const;$", re.MULTILINE
)


def _ts_matte_procedures(source: str) -> tuple[str, ...]:
    """TS の宣言から手順の名前を読む（宣言の形が動いたら黙って空にせず落とす）。"""
    declarations = MATTE_PROCEDURES_DECLARATION.findall(source)
    assert len(declarations) == 1, f"{MATTE_SOURCE} の MATTE_PROCEDURES の宣言が読めない"
    (body,) = declarations
    names = tuple(re.findall(r'"([^"]+)"', body))
    rest = re.sub(r'"[^"]+"|,|\s', "", body)
    assert names and rest == "", f"MATTE_PROCEDURES に文字列の名前以外の要素がある: {body!r}"
    return names


#: TS 側の宣言を読んだ値（手書きの写しにすると、片側だけ動いても緑のままになる）。カードが説明を
#: 持たない手順も、ロード側が読めない手順を説明するカードも作らない。
TS_MATTE_PROCEDURES = _ts_matte_procedures(MATTE_SOURCE.read_text(encoding="utf-8"))


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """weights の 1 dtype ぶん（`karume/5` の `container` — ADR 0109 決定 3）。

    `descriptor` はカードが読まない欄なので、形だけ実物どおりに置く（2 文書の長さと sha256）。
    """
    return {
        "container": {
            "descriptor": {
                "graph": {"length": 40, "sha256": "0" * 64},
                "model": {"length": 24, "sha256": "1" * 64},
            },
            "parts": list(refs),
        }
    }


def _birefnet_manifest(
    model: str = MODEL, seats: tuple[str, ...] = SEATS[CHECKPOINT]
) -> dict[str, Any]:
    """BiRefNet 系の最小 manifest（解像度と席以外は実物と重ならない偽値）。

    `seats` は quant 席の格納 dtype（先頭が既定）。既定は {@link CHECKPOINT}（HR）の実物の席。
    """
    side = int(model)
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": BIREFNET_SUPPORTED_PIPELINE,
                "weights": {
                    "matte": {
                        dtype: _container(_ref(f"m/model.{dtype}-00001-of-00003.krm", 13, "d"))
                        for dtype in seats
                    }
                },
                "assets": {},
                "quants": {dtype: {"weights": {"matte": dtype}, "session": {}} for dtype in seats},
                "defaultQuant": seats[0],
                "pipelineConfig": {
                    "imageWidth": side,
                    "imageHeight": side,
                    "imageMean": [0.1, 0.2, 0.3],
                    "imageStd": [0.4, 0.5, 0.6],
                    "interpolation": "bilinear",
                    "postprocess": "u8-pil-bicubic",
                },
            }
        },
    }


class TestTheSeatTable:
    def test_it_is_the_distributions_seats_with_the_default_first(self) -> None:
        """{@link SEATS} は配布の席の写し — 規則（`birefnet_seat_dtypes` /
        `birefnet_default_quant`）が動いたのに偽 manifest だけ古い席で組んで緑、を作らない。"""
        from birefnet.distribution import birefnet_default_quant, birefnet_seat_dtypes

        assert set(SEATS) == set(BIREFNET_CHECKPOINTS)
        for checkpoint, seats in SEATS.items():
            assert seats == birefnet_seat_dtypes(checkpoint)
            assert seats[0] == birefnet_default_quant(checkpoint)


class TestBirefnetPostprocess:
    """後処理の手順（`pipelineConfig.postprocess`）の説明 — 手順の名前はモデルの宣言から引く。"""

    def test_it_describes_exactly_the_procedures_the_loader_accepts(self) -> None:
        assert sorted(BIREFNET_POSTPROCESS_DESCRIPTIONS) == sorted(TS_MATTE_PROCEDURES)

    def test_every_checkpoint_declares_a_procedure_the_loader_accepts(self) -> None:
        """組み立て段が書く手順は、ロード側の閉じた集合の中（外なら配布形が読めない）。"""
        from birefnet.distribution import BIREFNET_POSTPROCESS

        assert set(BIREFNET_POSTPROCESS.values()) <= set(TS_MATTE_PROCEDURES)

    @pytest.mark.parametrize("procedure", TS_MATTE_PROCEDURES)
    def test_it_describes_the_declared_procedure_only(self, procedure: str) -> None:
        manifest = _birefnet_manifest()
        manifest["models"][MODEL]["pipelineConfig"]["postprocess"] = procedure
        card = render_birefnet_model_card(manifest, REPO, CHECKPOINT)
        assert "\n".join(BIREFNET_POSTPROCESS_DESCRIPTIONS[procedure]) in card
        (other,) = (name for name in TS_MATTE_PROCEDURES if name != procedure)
        assert f"`{other}`" not in card

    def test_it_refuses_a_procedure_it_cannot_describe(self) -> None:
        """推し量った説明を載せない — 配布形の実際の後処理とカードが食い違ったまま公開される。"""
        manifest = _birefnet_manifest()
        manifest["models"][MODEL]["pipelineConfig"]["postprocess"] = "nearest"
        with pytest.raises(ValueError, match="nearest"):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)

    def test_it_refuses_a_model_without_a_declared_procedure(self) -> None:
        """欄の欠けも、説明を持たない手順と同じ文言で落とす（素の `KeyError` にしない）。"""
        manifest = _birefnet_manifest()
        del manifest["models"][MODEL]["pipelineConfig"]["postprocess"]
        with pytest.raises(ValueError, match="後処理の手順 None の説明を持たない"):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)


class TestBirefnetCardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _birefnet_manifest()
        manifest["models"][MODEL]["pipeline"] = FOREIGN_PIPELINE
        with pytest.raises(ValueError, match=BIREFNET_SUPPORTED_PIPELINE):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)

    def test_it_refuses_a_checkpoint_it_cannot_attribute(self) -> None:
        """帰属表に無い checkpoint で描くと、`base_model` の無いカード（= 出所を名乗って
        いない再配布）が黙って出る。既知の一覧を添えて落とす。
        """
        with pytest.raises(ValueError, match="tiny"):
            render_birefnet_model_card(_birefnet_manifest(), REPO, "tiny")

    def test_it_refuses_a_resolution_it_has_not_measured(self) -> None:
        """カードが名乗る資源の数は 1 つ残らず実測に紐づく（実測の無い解像度では描かない）。"""
        manifest = _birefnet_manifest()
        manifest["models"][MODEL]["pipelineConfig"]["imageWidth"] = 512
        with pytest.raises(ValueError, match="512"):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _birefnet_manifest()
        before = copy.deepcopy(manifest)
        assert render_birefnet_model_card(manifest, REPO, CHECKPOINT) == (
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)
        )
        assert manifest == before

    def test_the_title_follows_the_checkpoint(self) -> None:
        """上流が名前で売っているモデルはその名前で呼ぶ（見出しは帰属表 1 つから来る）。"""
        assert "# Lucida (BiRefNet) — Karume" in render_birefnet_model_card(
            _birefnet_manifest(seats=SEATS["lucida"]), REPO, "lucida"
        )
        assert "# BiRefNet HR — Karume" in render_birefnet_model_card(
            _birefnet_manifest(), REPO, CHECKPOINT
        )

    def test_the_attribution_does_not_follow_the_model_name(self) -> None:
        """モデル名は解像度で、帰属の軸ではない — 同じモデル名でも席（checkpoint）が違えば別の
        上流を名乗る。"""
        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            card = render_birefnet_model_card(
                _birefnet_manifest(seats=SEATS[checkpoint]), REPO, checkpoint
            )
            assert f"base_model: {repo}" in card

    def test_the_upstream_table_is_the_only_source_of_the_repository_ids(self) -> None:
        """`BIREFNET_UPSTREAM` は帰属表からの導出（2 表にすると片方だけ動ける）。"""
        assert {
            name: entry.repo for name, entry in BIREFNET_CHECKPOINTS.items()
        } == BIREFNET_UPSTREAM


class TestBirefnetEntryPoint:
    """カードが案内するロード入口 — `fromPretrained` の 1 本だけ。"""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` は案内しない（2026-08-29 裁定）。

        分割配布形も読めるようになった（X2-101）が、あちらはバイト列を自分で持っている前提の
        ローカルデバッグ向けの面で、HF から使う読者の普通の入口は `fromPretrained`。両方を
        並べると「どちらを使うのか」を読者に判断させることになる。`fromPretrained` 側も併せて
        見るのは、Usage ごと消えても通る門にしないため。
        """
        card = render_birefnet_model_card(_birefnet_manifest(), REPO, CHECKPOINT)
        assert "fromAssets" not in card
        assert "BirefnetPipeline.fromPretrained" in card


#: checkpoint × 実測した解像度（実行資源の注記のパラメータ）。
_MEASURED = [
    (checkpoint, model)
    for checkpoint, entry in sorted(BIREFNET_CHECKPOINTS.items())
    for model in sorted(entry.resources.by_resolution)
]


class TestBirefnetResourceNote:
    """実行資源の注記（利用者が「渡す前に知りたい事実」— manifest に無いので定数）。"""

    @pytest.mark.parametrize(("checkpoint", "model"), _MEASURED)
    def test_the_card_names_the_gpu_memory_that_resolution_needs(
        self, checkpoint: str, model: str
    ) -> None:
        """MUST: 総確保と、128MiB 既定を超える binding の両方を**そのモデルの実測で**名乗る。

        WebGPU の `maxStorageBufferBindingSize` の仕様既定は 128MiB なので、それを超える
        binding は「端末によっては要求自体が通らない」制約。カードが黙っていると、読み手は
        `requiredLimits` が空なこと（= 常駐分は既定内）を「既定スペックで動く」と読む。
        """
        card = render_birefnet_model_card(
            _birefnet_manifest(model, SEATS[checkpoint]), REPO, checkpoint
        )
        measurement = BIREFNET_CHECKPOINTS[checkpoint].resources
        resources = measurement.by_resolution[model]

        assert resources.total in card
        assert resources.binding in card
        assert resources.run in card
        assert measurement.conditions in card
        assert "maxStorageBufferBindingSize" in card

    def test_each_checkpoint_measured_the_seat_it_names(self) -> None:
        """実測した席は checkpoint ごと — HR は f16 席・Lucida は f32 席（どちらも
        2026-10-11）。"""
        assert {name: entry.resources.quant for name, entry in BIREFNET_CHECKPOINTS.items()} == {
            "hr": "f16",
            "lucida": "f32",
        }

    @pytest.mark.parametrize("checkpoint", sorted(BIREFNET_CHECKPOINTS))
    def test_the_total_is_named_as_the_measured_seats_number(self, checkpoint: str) -> None:
        """総確保がどの席の実測かを言う（席が 2 つ並ぶ Lucida のカードでも、他の席の総確保は
        推し量らない — 2026-09-26 レビュー F2）。"""
        card = render_birefnet_model_card(
            _birefnet_manifest(seats=SEATS[checkpoint]), REPO, checkpoint
        )
        prose = " ".join(card.split())
        measurement = BIREFNET_CHECKPOINTS[checkpoint].resources

        assert (
            f"{measurement.by_resolution[MODEL].total} allocated in total with the"
            f" `{measurement.quant}` quant" in prose
        )

    def test_it_refuses_a_model_without_the_measured_seat(self) -> None:
        """実測した席が無い配布形では、総確保を名乗れない（別の席の数として読まれる）。

        Lucida（f32 席で実測）の f16 席だけの manifest。HR は f16 席の他に描ける席が無い
        （f32 席は上流の格納より広く、帰属節が落とす — {@link
        TestBirefnetStorageAttribution.test_a_seat_wider_than_the_checkpoint_is_refused}）。
        """
        with pytest.raises(ValueError, match="実測した quant 'f32'"):
            render_birefnet_model_card(_birefnet_manifest(seats=("f16",)), REPO, "lucida")

    def test_it_does_not_carry_another_resolutions_measurement(self) -> None:
        """1 つの実測を全モデルへ写すと、片方のカードが**測っていない数**を名乗る。"""
        card = render_birefnet_model_card(_birefnet_manifest(MODEL), REPO, CHECKPOINT)

        assert BIREFNET_CHECKPOINTS[CHECKPOINT].resources.by_resolution["2048"].total not in card

    def test_it_does_not_carry_another_checkpoints_measurement(self) -> None:
        """実測は checkpoint ごと — HR のカードは Lucida の総確保も、それを測った席も名乗らない。

        測った条件（日付と機種）は 2 つの checkpoint で同じになりうるので、区別には使わない。
        """
        card = render_birefnet_model_card(_birefnet_manifest(MODEL), REPO, CHECKPOINT)
        prose = " ".join(card.split())
        other = BIREFNET_CHECKPOINTS["lucida"].resources

        assert other.by_resolution[MODEL].total not in card
        assert f"allocated in total with the `{other.quant}` quant" not in prose


def _with_seat(manifest: dict[str, Any], dtype: str, *, default: bool = False) -> dict[str, Any]:
    """既定モデルへ `dtype` の weights と同名の quant 席を足す（`default` なら既定席にする）。"""
    model = manifest["models"][manifest["defaultModel"]]
    model["weights"]["matte"][dtype] = _container(
        _ref(f"m/model.{dtype}-00001-of-00002.krm", 7, "e")
    )
    model["quants"][dtype] = {"weights": {"matte": dtype}, "session": {}}
    if default:
        model["defaultQuant"] = dtype
    return manifest


class TestBirefnetStorageAttribution:
    """席ごとの格納の説明と `base_model_relation`（ADR 0113 — checkpoint の格納から決まる）。"""

    def test_the_weights_key_is_the_distributions_role(self) -> None:
        """カードは循環 import を避けて weights キーを自分で綴る — 配布 recipe と一致させる。"""
        from birefnet.distribution import BIREFNET_ROLE

        assert BIREFNET_MATTE_WEIGHTS == BIREFNET_ROLE

    @pytest.mark.parametrize("checkpoint", sorted(BIREFNET_CHECKPOINTS))
    def test_the_default_seat_declares_no_relation(self, checkpoint: str) -> None:
        """既定席（上流の格納と同じ型 — HR は f16・Lucida は f32）は checkpoint の値を変えない —
        4 値のどれでもないので置かない。"""
        card = render_birefnet_model_card(
            _birefnet_manifest(seats=SEATS[checkpoint]), REPO, checkpoint
        )
        assert "base_model_relation" not in card

    def test_an_f16_default_for_the_f32_checkpoint_is_quantized(self) -> None:
        card = render_birefnet_model_card(
            _with_seat(_birefnet_manifest(seats=("f32",)), "f16", default=True), REPO, "lucida"
        )
        assert "base_model_relation: quantized" in card

    def test_the_storage_lines_follow_the_seats_in_the_manifest(self) -> None:
        """dtype ラベルは manifest から — f32 だけの配布形は f16 席を名乗らない。"""
        plain = render_birefnet_model_card(_birefnet_manifest(seats=("f32",)), REPO, "lucida")
        both = render_birefnet_model_card(_birefnet_manifest(seats=SEATS["lucida"]), REPO, "lucida")

        assert "  - `f32`:" in plain and "  - `f16`:" not in plain
        assert "  - `f32`:" in both and "  - `f16`:" in both

    def test_the_hr_card_names_its_f16_checkpoint(self) -> None:
        """HR の唯一の席（f16）は上流の値そのもの — 席が 1 つなので単数で名乗る。

        旧文面「source checkpoint's own f32 values」は HR では事実と違った（上流は f16）。
        """
        card = render_birefnet_model_card(_birefnet_manifest(), REPO, "hr")
        prose = " ".join(card.split())

        assert (
            "**No quantization**: the quant seat stores the source checkpoint's own values" in prose
        )
        assert "  - `f16`: the checkpoint's own f16 values" in card
        assert "`f32`:" not in card
        assert "widened exactly from f16 to f32" not in prose
        assert "own f32 values" not in card

    def test_a_seat_wider_than_the_checkpoint_is_refused(self) -> None:
        """上流の格納より広い席（HR の f32 席）の文面は持たない — 値を正確に広げるだけの席は
        配らない（ADR 0113 追記 2026-10-11）ので、その席を名乗るカードは描かない。"""
        with pytest.raises(ValueError, match="格納 f32 の席を f16 の checkpoint"):
            render_birefnet_model_card(_with_seat(_birefnet_manifest(), "f32"), REPO, "hr")

    def test_the_lucida_card_declares_its_f16_seat_quantized(self) -> None:
        card = render_birefnet_model_card(_birefnet_manifest(seats=SEATS["lucida"]), REPO, "lucida")

        assert "**Quantization**" in card
        assert "  - `f16`: **quantized**" in card
        assert "  - `f32`: the checkpoint's own f32 values, unchanged." in card

    def test_a_seat_it_has_no_wording_for_is_refused(self) -> None:
        """段 2 の i8 は文面と一緒に足す — 推し量った文面を帰属節へ載せない。"""
        with pytest.raises(ValueError, match="i8"):
            render_birefnet_model_card(_with_seat(_birefnet_manifest(), "i8"), REPO, "hr")
