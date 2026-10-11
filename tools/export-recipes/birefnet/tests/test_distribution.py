"""BiRefNet 系の配布 recipe（`birefnet.distribution`）— 組み立て 1 周ぶんの単体テスト。

実資産は使わない。組み立てへ届く入力は数 KB の**正当な最小コンテナ**（`ir_fixtures`）で、
門に落とされることを見るケースも同じ器で作る（{@link _birefnet_container}）— 組み立ては入力を
コンテナとして開くので、手で綴った偽物は門の手前で別の理由で落ちる。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

core だけで観測できる層（合成計画で足りる規模上限・quant 完全写像・staging/swap の不変条件・
帰属プロファイルの解決規則）は `tools/exporter/tests/test_dist.py` が持つ（ADR 0065 段 3+4 の
分割）。テンプレート単位の門は `birefnet/tests/test_card.py`。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from container_series import i2_container, placed_paths, replace_component, write_component
from ir_fixtures import Shape, ir_container
from upstream_fixture import OTHER_REVISION, stamp_fixture_provenance

from _shared.licenses import mit_license
from birefnet.card import (
    BIREFNET_CHECKPOINTS,
    BIREFNET_LICENSE,
    BIREFNET_POSTPROCESS_DESCRIPTIONS,
    BIREFNET_UPSTREAM,
)
from birefnet.distribution import (
    BIREFNET_COPYRIGHTS,
    BIREFNET_DEFAULT_MODEL,
    BIREFNET_HR_CHECKPOINT,
    BIREFNET_IMAGE_MEAN,
    BIREFNET_IMAGE_STD,
    BIREFNET_LUCIDA_CHECKPOINT,
    BIREFNET_MODELS,
    BIREFNET_OUTPUT_PATHS,
    BIREFNET_PLAIN_DTYPE,
    BIREFNET_POSTPROCESS,
    BIREFNET_QUANT_SEATS,
    BIREFNET_ROLE,
    BIREFNET_STORAGE_FORBIDDEN,
    BIREFNET_STORAGE_REQUIREMENTS,
    BIREFNET_WEIGHT_DTYPES,
    LUCIDA_PIPELINE,
    PIPELINE,
    BirefnetSources,
    birefnet_default_quant,
    birefnet_plan,
    birefnet_postprocess,
    birefnet_quants,
    birefnet_repo_name,
    birefnet_role,
    birefnet_seat_dtypes,
    birefnet_series_name,
    birefnet_sources,
    birefnet_weights,
)
from dist import default_out_dir, main
from karume.container import CODEC_LEDGER, Provenance
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NOTICE_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)


def _baked_provenance(checkpoint: str, revision: str | None = None) -> Provenance:
    """台本がその checkpoint の取得記録から焼く出所（出所の門の正常形）。

    `revision` を省くと帰属表の固定値（= 組み立てが突き合わせる値）を名乗る。
    """
    pinned = BIREFNET_CHECKPOINTS[checkpoint].revision
    return Provenance(
        license=BIREFNET_LICENSE,
        notice=NOTICE_FILENAME,
        upstream_revision=pinned if revision is None else revision,
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """モデルサブツリー内の期待 path（ADR 0041 §9 の一様レイアウト）。"""
    return [f"{model}/{rel}" for rel in paths]


def _placed_paths(checkpoint: str = BIREFNET_HR_CHECKPOINT) -> list[str]:
    """配布形に現れる相対 path（weights の席は part 連番に展開される — container-v1 §8）。

    出力 path 表は dtype の語彙の全体の役割を持つので、その checkpoint の席の役割だけに絞る。
    """
    roles = {birefnet_role(dtype) for dtype in birefnet_seat_dtypes(checkpoint)}
    paths = {role: path for role, path in BIREFNET_OUTPUT_PATHS.items() if role in roles}
    return placed_paths(paths, birefnet_weights(checkpoint))


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


# ---- BiRefNet 系（image-segmentation）----------------------------------------
#
# 偽資産は実物と同じ解像度（モデル名 = 解像度なので、SigLIP2 のように小さな非正方へは
# 寄せられない）。代わりに「モデル名と寸法が食い違う」「非正方」「受理集合外のモデル名」の
# 3 つを**落ちる側**のケースとして持つ。

#: 2 モデル目（`--model 2048`）。1 リポ 2 モデルの器（ADR 0092 決定 1 / 8）で既定と同居する。
_BIREFNET_SECOND_MODEL = "2048"

#: `pipelineConfig` の欄名（TS 側 `packages/models/src/birefnet/config.ts` の `ROOT_KEYS` の
#: 写し）。**ロード側は未知キーも欠落も parse 時に落とす**ので、焼く側とロード側の欄名は
#: 完全一致が要る。
_BIREFNET_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "interpolation",
    "postprocess",
)


#: 既定モデルの一辺（= 既定のモデル名そのもの）。偽資産の shape はここから組む。
_DEFAULT_SIDE = int(BIREFNET_DEFAULT_MODEL)


def _birefnet_container(
    *,
    storage: str = "f32",
    side: int = _DEFAULT_SIDE,
    input_name: str = "pixel_values",
    input_shape: Sequence[Any] | None = None,
    matte_shape: Sequence[Any] | None = None,
    outputs: int = 1,
    symbols: Sequence[str] = (),
) -> list[bytes]:
    """系列に置くマット推定グラフの中身（part 列 — 先頭が part 0）。

    組み立てへ届く形は**正当なコンテナ**でなければならない（組み立ては入力を開いて宣言の全
    規則で見る — `karume.dist.assert_weight_components_verified`）ので、門に落とす側も同じ器で
    作り、**宣言だけを実物とずらす**。

    `storage` は実物どおりの混成コンポーネントの格納形（f16 / i8 / i4 は適格外の重みと bias が
    f32 で残り、i4 はさらに i8 が混ざる）— 禁止表（{@link BIREFNET_STORAGE_FORBIDDEN}）の門は
    この形でしか試せない。残りの引数はグラフ宣言の各軸（入力の名前 / shape・マットの shape・
    出力の本数・記号次元）で、既定は実物どおり。
    """
    shape: Shape = list(input_shape) if input_shape is not None else [1, 3, side, side]
    matte: Shape = list(matte_shape) if matte_shape is not None else [1, 1, shape[2], shape[3]]
    return ir_container(
        mark="birefnet-matte",
        # 疑似系列も**部品名で名乗る**（容器のグラフ名 = manifest の weights のキー
        # MUST・container-v1 §2.1 — 組み立ての門がこの一致を見る）。
        named=BIREFNET_ROLE,
        storage=storage,
        inputs=((input_name, shape),),
        outputs=[matte] * outputs,
        symbols=symbols,
    )


def _build_birefnet_sources(
    root: Path,
    *,
    checkpoint: str = BIREFNET_HR_CHECKPOINT,
    model: str = BIREFNET_DEFAULT_MODEL,
    side: int | None = None,
    storage: str = BIREFNET_PLAIN_DTYPE,
    f16_storage: str = "f16",
    f16_side: int | None = None,
    provenance: Provenance | None = None,
    **container: Any,
) -> BirefnetSources:
    """系列を偽資産で再現する（配布しない `io.*` / `quality.json` の混入込み）。

    並びは `_shared.paths` の実レイアウト（`outputs/series/`）に揃える — CLI 経路のテストが
    root を差し替えるだけで同じ木を指せる形。系列は checkpoint × 解像度 × 格納 dtype で 1 本ずつ
    （その checkpoint の席 — `birefnet_seat_dtypes` — の全部を焼く）なので、偽資産の寸法は既定で
    **モデル名から**引く。`side` / `f16_side` はその対応を**故意にずらす**ための席（系列を 1 本
    掴み違えた形をそのまま再現する）。

    `storage` / `side` は **f32 席の系列**の形（禁止表の門のケース — f32 席を持つのは Lucida）、
    `f16_storage` / `f16_side` は f16 席の系列の形（要求表の門のケース）。その checkpoint に無い
    席の形を指定したら落とす（黙って焼かれない故障注入は、何も試さずに緑になる）。残りのグラフ
    宣言の軸（`**container`）は全 dtype の系列に掛ける。

    容器が名乗る出所は `provenance`（省くとその checkpoint の固定 revision — {@link
    _baked_provenance}）で、出所の門のケースだけが差し替える。
    """
    sources = birefnet_sources(root / "outputs" / "series", checkpoint, model)
    shapes = {BIREFNET_PLAIN_DTYPE: (storage, side), "f16": (f16_storage, f16_side)}
    for dtype, shape in shapes.items():
        if dtype not in sources.series_by_dtype and shape != (dtype, None):
            raise AssertionError(
                f"checkpoint '{checkpoint}' に {dtype} 席は無い — その席の系列の形 {shape} は"
                " 焼かれない"
            )
    with pytest.MonkeyPatch.context() as patch:
        stamp_fixture_provenance(
            patch, _baked_provenance(checkpoint) if provenance is None else provenance
        )
        for dtype, series in sources.series_by_dtype.items():
            seat_storage, seat_side = shapes[dtype]
            write_component(
                series / "model.krm",
                _birefnet_container(
                    storage=seat_storage,
                    side=seat_side if seat_side is not None else int(model),
                    **container,
                ),
            )
            # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
            _write(series / "io.ramp.safetensors", b"io-fixture")
            if dtype != BIREFNET_PLAIN_DTYPE:
                # 圧縮系列の品質計測の記録（`birefnet/export.py` の QUALITY_FILE）も配布に入らない。
                _write(series / "quality.json", b"{}")
    return sources


#: checkpoint → その配布リポを組む pipeline の CLI 名（1 リポ 1 Pipeline — ADR 0092 決定 1）。
_PIPELINE_NAMES = {BIREFNET_HR_CHECKPOINT: "birefnet", BIREFNET_LUCIDA_CHECKPOINT: "lucida"}


@pytest.fixture
def birefnet_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_birefnet_sources(tmp_path)
    out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_HR_CHECKPOINT)
    manifest = assemble_family(
        [birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)],
        out_dir,
        BIREFNET_DEFAULT_MODEL,
        root_files=PIPELINE.root_files,
    )
    return out_dir, manifest


def _birefnet_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][BIREFNET_DEFAULT_MODEL]


class TestBirefnetLayout:
    def test_it_places_the_single_graph_under_the_model_subtree(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        expected = _in_subtree(BIREFNET_DEFAULT_MODEL, _placed_paths())
        # 法的テキスト 2 本（MIT の著作権 / 許諾表示と改変告知）は manifest が宣言しないメタ席。
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, "LICENSE.md", "NOTICE.md"]
        )

    def test_it_never_carries_the_io_fixtures(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, birefnet_assembled) -> None:
        """実行に要るのはグラフ 1 本だけ（tokenizer も表も無い = `assets` は空）。"""
        _, manifest = birefnet_assembled
        model = _birefnet_model(manifest)
        assert model["pipeline"] == "birefnet/2"
        assert list(model["weights"]) == [BIREFNET_ROLE]
        assert model["assets"] == {}
        # HR の席は f16 だけ — 上流の格納が f16（ADR 0113 追記 2026-10-11）。
        assert list(model["weights"][BIREFNET_ROLE]) == ["f16"]
        assert list(model["quants"]) == ["f16"]
        assert model["defaultQuant"] == "f16"
        assert model["quants"]["f16"]["weights"] == {BIREFNET_ROLE: "f16"}
        assert model["quants"]["f16"]["session"] == {}

    def test_lucida_keeps_both_seats_with_f32_as_the_default(self, tmp_path: Path) -> None:
        """Lucida の上流は f32 — f32 席（既定）と、丸めた f16 席の 2 つ。"""
        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        plan = birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)

        assert list(plan.weights[BIREFNET_ROLE]) == ["f32", "f16"]
        assert list(plan.quants) == ["f32", "f16"]
        assert plan.default_quant == "f32"
        assert plan.quants["f32"]["weights"] == {BIREFNET_ROLE: "f32"}
        assert plan.quants["f16"]["weights"] == {BIREFNET_ROLE: "f16"}
        assert plan.quants["f32"]["session"] == {}
        assert plan.quants["f16"]["session"] == {}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(tmp_path)
        out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_HR_CHECKPOINT)
        plans = [birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)]
        first = assemble_family(plans, out_dir, BIREFNET_DEFAULT_MODEL)
        assert first == assemble_family(plans, out_dir, BIREFNET_DEFAULT_MODEL)
        assert verify_dist(out_dir)

    @pytest.mark.parametrize(("storage", "intruder"), [("f16", "f16"), ("i8", "i8"), ("i4", "i4")])
    def test_it_refuses_a_real_compressed_series_in_the_f32_seat(
        self, tmp_path: Path, storage: str, intruder: str
    ) -> None:
        """実物どおりの圧縮系列は**要求検査を満たす** — 禁止表だけが系列 root の取り違えを見る。

        圧縮系列も適格外の重みと bias を f32 で持つので「f32 を含む」は真になり、要求検査では
        1 バイトも落ちない。落ちるべき理由は「f32 席に別系列の資産が居る」で、数値の門では
        原理的に検出できない（ADR 0027 / 0029）。
        """
        sources = _build_birefnet_sources(
            tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT, storage=storage
        )
        with pytest.raises(DistError, match=rf"{birefnet_role('f32')}: .* {intruder} がある"):
            birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)

    def test_the_plain_f32_series_passes_both_storage_gates(self, tmp_path: Path) -> None:
        """正常対 — 素の f32 系列は要求検査も禁止表も通る（禁止表が恒真に倒れていない）。"""
        sources = _build_birefnet_sources(
            tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT, storage="f32"
        )

        assert birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT).name == BIREFNET_DEFAULT_MODEL

    def test_the_forbidden_table_names_every_compressed_layout_in_the_codec_ledger(self) -> None:
        """禁止表は格納検査の語彙（codec 台帳の layout）から f32 / i32 を除いた**全部**を持つ。

        1 つでも抜けると、抜けた格納形の系列 root だけが f32 席を黙って通る。i32 は素の格納
        （添字表）なので禁じない。
        """
        compressed = {entry.layout for entry in CODEC_LEDGER.values()} - {"f32", "i32"}

        assert set(BIREFNET_STORAGE_FORBIDDEN[birefnet_role("f32")]) == compressed

    def test_it_refuses_an_i2_series_in_the_f32_seat(self, tmp_path: Path) -> None:
        """`ir_container` では書けない i2（QAT 系列の固定 packed 値）も、束縛表の現物で落ちる。"""
        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        replace_component(
            sources.series_by_dtype["f32"] / "model.krm", i2_container(named=BIREFNET_ROLE)
        )

        with pytest.raises(DistError, match=rf"{birefnet_role('f32')}: .* i2 がある"):
            birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)


class TestBirefnetContainerProvenance:
    """容器が名乗る出所を帰属表の固定値（ライセンスと上流 revision）へ突き合わせる。"""

    def test_it_refuses_a_container_that_names_another_license(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path,
            provenance=Provenance(
                license="apache-2.0",
                notice=NOTICE_FILENAME,
                upstream_revision=BIREFNET_CHECKPOINTS[BIREFNET_HR_CHECKPOINT].revision,
            ),
        )

        with pytest.raises(DistError, match=r"provenance\.license が 'apache-2.0'"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_container_without_an_upstream_revision(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path, provenance=Provenance(license=BIREFNET_LICENSE, notice=NOTICE_FILENAME)
        )

        with pytest.raises(DistError, match="upstreamRevision を持たない"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_every_series_naming_the_pinned_revision_is_assembled(
        self, tmp_path: Path, checkpoint: str
    ) -> None:
        """正常形: 全席の系列がその checkpoint の固定 revision を名乗れば組める（下の故障注入の
        対照）。"""
        sources = _build_birefnet_sources(tmp_path, checkpoint=checkpoint)

        birefnet_plan(sources, checkpoint)

    def test_the_pinned_revisions_tell_the_checkpoints_apart(self) -> None:
        """固定値が checkpoint を区別すること — 同じ値なら下の取り違えが素通りする。"""
        revisions = [entry.revision for entry in BIREFNET_CHECKPOINTS.values()]

        assert len(set(revisions)) == len(revisions)

    def test_it_refuses_a_series_baked_from_another_revision(self, tmp_path: Path) -> None:
        """同じ checkpoint でも固定値と違う revision から焼いた系列は組まない（上流の更新を
        取り込むなら固定値を動かす — 手元の取得だけ先に進んだ系列を黙って配らない）。"""
        sources = _build_birefnet_sources(
            tmp_path, provenance=_baked_provenance(BIREFNET_HR_CHECKPOINT, OTHER_REVISION)
        )

        with pytest.raises(DistError, match="upstreamRevision が") as raised:
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)
        assert OTHER_REVISION in str(raised.value)

    def test_it_refuses_the_f16_series_of_another_checkpoint(self, tmp_path: Path) -> None:
        """故障注入: Lucida の f32 系列 + HR の f16 系列（ライセンスはどちらも MIT・寸法も同じ）。

        出所の門がライセンスと revision の有無だけを見ていると、別 checkpoint の系列が 1 モデルの
        2 席として組み上がる（2026-09-26 レビュー A-1）。全容器を Lucida の固定値で見るので、HR の
        revision を名乗る f16 系列がそれを落とす。
        """
        lucida = _build_birefnet_sources(tmp_path / "lucida", checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        hr = _build_birefnet_sources(tmp_path / "hr")
        mixed = BirefnetSources(
            series_by_dtype={
                "f32": lucida.series_by_dtype["f32"],
                "f16": hr.series_by_dtype["f16"],
            }
        )

        with pytest.raises(DistError, match="upstreamRevision が") as raised:
            birefnet_plan(mixed, BIREFNET_LUCIDA_CHECKPOINT)
        assert str(hr.series_by_dtype["f16"]) in str(raised.value)
        assert BIREFNET_CHECKPOINTS[BIREFNET_HR_CHECKPOINT].revision in str(raised.value)
        assert BIREFNET_CHECKPOINTS[BIREFNET_LUCIDA_CHECKPOINT].revision in str(raised.value)

    def test_the_single_hr_seat_refuses_another_checkpoints_series(self, tmp_path: Path) -> None:
        """故障注入: HR の唯一の席（f16）に Lucida の f16 系列。

        席が 1 つなので、席どうしの revision の一致では何も比べられない。固定値で見るので、HR の
        リポには HR の重みだけが入る（帰属・著作権行・後処理は HR のまま中身だけ Lucida、という
        配布形を作らない）。
        """
        lucida = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        foreign = BirefnetSources(series_by_dtype={"f16": lucida.series_by_dtype["f16"]})

        with pytest.raises(DistError, match="upstreamRevision が") as raised:
            birefnet_plan(foreign, BIREFNET_HR_CHECKPOINT)
        assert BIREFNET_CHECKPOINTS[BIREFNET_LUCIDA_CHECKPOINT].revision in str(raised.value)
        assert BIREFNET_CHECKPOINTS[BIREFNET_HR_CHECKPOINT].revision in str(raised.value)


class TestBirefnetPipelineConfig:
    """`pipelineConfig` はロード側（`src/birefnet/config.ts`）のスキーマと欄名まで一致する。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, birefnet_assembled) -> None:
        _, manifest = birefnet_assembled
        assert tuple(_birefnet_model(manifest)["pipelineConfig"]) == _BIREFNET_CONFIG_KEYS

    def test_it_derives_the_resize_target_from_the_exported_graph(self, birefnet_assembled) -> None:
        """resize 先の出どころは焼かれたグラフ 1 つきり（上流に前処理 config が無い）。"""
        _, manifest = birefnet_assembled
        config = _birefnet_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == _DEFAULT_SIDE
        assert config["imageHeight"] == _DEFAULT_SIDE

    def test_it_declares_the_upstream_normalization_constants(self, birefnet_assembled) -> None:
        """正規化定数は機械可読な出どころが無いので dist が宣言として持つ（節の冒頭）。"""
        _, manifest = birefnet_assembled
        config = _birefnet_model(manifest)["pipelineConfig"]
        assert config["imageMean"] == list(BIREFNET_IMAGE_MEAN)
        assert config["imageStd"] == list(BIREFNET_IMAGE_STD)
        assert config["interpolation"] == "bilinear"

    def test_the_hr_repository_declares_the_upstream_postprocess(self, birefnet_assembled) -> None:
        """HR の上流は σ → 8bit → Pillow の BICUBIC（ADR 0125 — A/B の後に上流を採った）。"""
        _, manifest = birefnet_assembled
        assert _birefnet_model(manifest)["pipelineConfig"]["postprocess"] == "u8-pil-bicubic"

    def test_the_lucida_repository_declares_its_own_upstream_postprocess(
        self, tmp_path: Path
    ) -> None:
        """Lucida の上流は σ → 浮動小数の antialias bilinear → 8bit。手順は checkpoint から決まる
        （同じ解像度・同じ前処理でも、HR の手順を写していれば Lucida は自分の上流から外れる）。
        """
        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        plan = birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)
        assert plan.pipeline_config["postprocess"] == "float-bilinear"

    def test_every_checkpoint_has_a_postprocess_the_card_can_describe(self) -> None:
        """手順の表は帰属表と同じ checkpoint を持ち、値はカードが説明できる手順だけ。"""
        assert sorted(BIREFNET_POSTPROCESS) == sorted(BIREFNET_UPSTREAM)
        assert set(BIREFNET_POSTPROCESS.values()) <= set(BIREFNET_POSTPROCESS_DESCRIPTIONS)

    def test_it_refuses_a_checkpoint_without_a_postprocess(self) -> None:
        """既定の手順で埋めない — 上流と違う後処理を宣言した配布形が黙って成立する。"""
        with pytest.raises(DistError, match="tiny"):
            birefnet_postprocess("tiny")


class TestBirefnetGraphGate:
    """組み立て門 — ずれても配布形としては成立してしまう組み合わせを、配置の前に落とす。"""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """モデル名は利用者が `model: "1024"` と綴る値そのもの — 中身とずれたら配れない。"""
        sources = _build_birefnet_sources(tmp_path, input_shape=(1, 3, 512, 512))
        with pytest.raises(DistError, match=rf"モデル '{BIREFNET_DEFAULT_MODEL}' が名乗る"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_the_other_models_graph_in_this_models_seat(self, tmp_path: Path) -> None:
        """同居する 2 モデルの取り違え — 系列を 1 本掴み違えるだけで起こる、この器の固有形。

        `2048` の席へ `1024` のグラフが入ると、`pipelineConfig` は 1024² で整合したままなので
        実行も `verify_dist` も通り、**食い違うのはモデル名だけ**になる。
        """
        sources = _build_birefnet_sources(
            tmp_path, model=_BIREFNET_SECOND_MODEL, f16_side=_DEFAULT_SIDE
        )
        with pytest.raises(DistError, match=rf"モデル '{_BIREFNET_SECOND_MODEL}' が名乗る"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT, _BIREFNET_SECOND_MODEL)

    def test_it_refuses_a_model_name_outside_the_published_set(self, tmp_path: Path) -> None:
        """受理集合は {@link BIREFNET_MODELS} 1 つきり — 系列を探す前に落とす。"""
        with pytest.raises(DistError, match="は配らない"):
            birefnet_sources(tmp_path, BIREFNET_HR_CHECKPOINT, "512")

    def test_it_refuses_a_graph_whose_input_is_not_square(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(tmp_path, input_shape=(1, 3, _DEFAULT_SIDE, 512))
        with pytest.raises(DistError, match=rf"モデル '{BIREFNET_DEFAULT_MODEL}' が名乗る"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_graph_with_another_channel_count(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path, input_shape=(1, 4, _DEFAULT_SIDE, _DEFAULT_SIDE)
        )
        with pytest.raises(DistError, match="batch もチャネル数も静的"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """multi-scale supervision 込みの export は、位置で引く後段が別の値を α として読む。"""
        sources = _build_birefnet_sources(tmp_path, outputs=2)
        with pytest.raises(DistError, match="マット 1 本"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_matte_that_is_not_one_channel(self, tmp_path: Path) -> None:
        """要素数だけ見る実装なら通ってしまう形（`[1, 3, S, S]` = 中間予測 3 枚）。"""
        sources = _build_birefnet_sources(
            tmp_path, matte_shape=[1, 3, _DEFAULT_SIDE, _DEFAULT_SIDE]
        )
        with pytest.raises(DistError, match="入力と同じ寸法の 1 チャネル"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """解像度も窓マスクも定数として焼かれているので、動かす軸は 1 本も無い。"""
        sources = _build_birefnet_sources(tmp_path, symbols=("T",))
        with pytest.raises(DistError, match="記号次元"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """実行側は名前で束ねるので、綴りが変われば束ねられない。"""
        sources = _build_birefnet_sources(tmp_path, input_name="pixels")
        with pytest.raises(DistError, match="グラフ入力"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)


class TestBirefnetModelCard:
    def _run(self, tmp_path: Path, checkpoint: str, *models: str) -> Path:
        """偽資産だけで CLI を 1 周回す（`--model` は解像度・先頭が `defaultModel`）。"""
        names = models or (BIREFNET_DEFAULT_MODEL,)
        series_dir = tmp_path / "outputs" / "series"
        for model in names:
            _build_birefnet_sources(tmp_path, checkpoint=checkpoint, model=model)
        out_dir = tmp_path / "dist"
        argv = ["--pipeline", _PIPELINE_NAMES[checkpoint]]
        for model in names:
            argv += ["--model", model]
        main([*argv, "--series", str(series_dir), "--out", str(out_dir)])
        return out_dir

    def test_it_describes_the_background_removal_distribution(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path, BIREFNET_HR_CHECKPOINT)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: image-segmentation" in card
        assert "license: mit" in card
        # 格納形を変えない配布形は 4 値のどれでもない（Hub の推論に任せる）。
        assert "base_model_relation" not in card
        # 合成をこちらでしない事実は、利用者が最初に確かめたい制約そのもの。
        assert "**Compositing is yours.**" in card
        assert f"resizes to {_DEFAULT_SIDE} × {_DEFAULT_SIDE}" in card
        assert "one alpha byte per pixel" in card
        # 後処理の手順はモデルの宣言から描く（HR = u8-pil-bicubic）。
        assert "- **post-processing** (`u8-pil-bicubic`" in card
        assert "`float-bilinear`" not in card
        # 単一ファイル配布形は廃止済み（ADR 0081 決定 3）— 器の説明が現物と食い違わない。
        assert "a single safetensors file" not in card
        assert "`.krm` part sequence" in card
        # カードは**検証を通った**配布形から描かれる。
        assert verify_dist(out_dir)

    def test_the_usage_snippet_enumerates_the_choices_from_the_manifest(
        self, tmp_path: Path
    ) -> None:
        """裁定 2026-08-12 の形: optional はコメント併記で、選べる値は manifest 由来の列挙。

        `segment()` に実行時ノブは無いので、この pipeline の optional は model / quant だけ。
        1 席しか無くても「available:」で綴るのは、席が増えたときに文面が自動で追従するため
        （「the only one this repository ships」のような焼き込みは、増えた瞬間に嘘になる）。
        """
        card = (self._run(tmp_path / "hr", BIREFNET_HR_CHECKPOINT) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert (
            f'  // model: "{BIREFNET_DEFAULT_MODEL}",'
            f" // default — available: {BIREFNET_DEFAULT_MODEL}" in card
        )
        # HR の席は f16 だけ（1 席でも「available:」で綴る）・Lucida は f32（既定）/ f16。
        assert '  // quant: "f16", // default — available: f16' in card
        assert "the only one this repository ships" not in card
        lucida = (
            self._run(tmp_path / "lucida", BIREFNET_LUCIDA_CHECKPOINT) / MODEL_CARD_FILENAME
        ).read_text(encoding="utf-8")
        assert '  // quant: "f32", // default — available: f16 / f32' in lucida

    def test_the_attribution_follows_the_checkpoint_of_the_repository(self, tmp_path: Path) -> None:
        """帰属は pipeline 席の checkpoint から一意に決まる（選ばせる軸にしない）。"""
        card = (self._run(tmp_path, BIREFNET_LUCIDA_CHECKPOINT) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {BIREFNET_UPSTREAM['lucida']}" in card
        # fine-tune の元と学習データのライセンスは lucida 側だけの事実。
        assert BIREFNET_UPSTREAM["hr"] in card
        assert "ToonOut" in card
        # 前処理を焼き込んだ変種を配らないことと、その理由がカードに残る。
        assert "lucida-m35-comfy.safetensors" in card
        # 後処理の手順も checkpoint の上流のもの（HR の手順を写さない）。
        assert "- **post-processing** (`float-bilinear`" in card
        assert "`u8-pil-bicubic`" not in card

    def test_the_default_model_card_does_not_carry_another_models_attribution(
        self, tmp_path: Path
    ) -> None:
        card = (self._run(tmp_path, BIREFNET_HR_CHECKPOINT) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {BIREFNET_UPSTREAM['hr']}" in card
        assert "ToonOut" not in card


class TestBirefnetResolutionFamily:
    """1 リポ 2 モデル（ADR 0092 決定 1 / 8）— 1024² と 2048² が同じリポに同居する。"""

    def _run(self, tmp_path: Path) -> Path:
        for model in BIREFNET_MODELS:
            _build_birefnet_sources(tmp_path, model=model)
        out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_HR_CHECKPOINT)
        main(
            [
                "--pipeline",
                "birefnet",
                "--model",
                BIREFNET_DEFAULT_MODEL,
                "--model",
                _BIREFNET_SECOND_MODEL,
                "--series",
                str(tmp_path / "outputs" / "series"),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_both_resolutions_live_in_one_repository_with_1024_as_the_default(
        self, tmp_path: Path
    ) -> None:
        """既定は**先頭の `--model`**（`karume.dist.main`）— 綴りの順序が既定を決める。"""
        out_dir = self._run(tmp_path)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert sorted(manifest["models"]) == sorted(BIREFNET_MODELS)
        assert manifest["defaultModel"] == BIREFNET_DEFAULT_MODEL
        assert out_dir.name == "karume-birefnet-hr"

    def test_each_model_declares_its_own_resize_target(self, tmp_path: Path) -> None:
        """同居しても寸法はモデルごと — 1 つの宣言を共有していれば片方が別の数になる。"""
        out_dir = self._run(tmp_path)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        for model in BIREFNET_MODELS:
            config = manifest["models"][model]["pipelineConfig"]
            assert (config["imageWidth"], config["imageHeight"]) == (int(model), int(model))

    def test_the_layout_keeps_each_models_graph_in_its_own_subtree(self, tmp_path: Path) -> None:
        """part 0（グラフ記述を持つ part）は寸法の宣言なので、共有席へ畳まれてはいけない。"""
        placed = sorted(verify_dist(self._run(tmp_path)))
        graph_part = _placed_paths()[0]
        for model in BIREFNET_MODELS:
            assert f"{model}/{graph_part}" in placed
        assert {path.split("/", 1)[0] for path in placed} <= {*BIREFNET_MODELS, "shared"}

    def test_the_card_enumerates_both_models_and_their_own_resources(self, tmp_path: Path) -> None:
        """Usage の列挙も資源の実測も manifest のモデル 2 つに追従する。"""
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        listing = " / ".join(sorted(BIREFNET_MODELS))
        assert f'  // model: "{BIREFNET_DEFAULT_MODEL}", // default — available: {listing}' in card
        # 資源はモデルごとの実測（1 つを両方の節へ写していれば片方が嘘になる）。HR の実測は f16 席。
        assert "about 1.2 GiB allocated in total with the `f16` quant" in card
        assert "about 3.6 GiB allocated in total with the `f16` quant" in card
        # 上流は 1 本（同居しているのは同じ checkpoint の解像度違い）。
        assert card.count(f"base_model: {BIREFNET_UPSTREAM['hr']}") == 1
        assert BIREFNET_UPSTREAM["lucida"] not in card


class TestBirefnetCli:
    def test_the_default_output_directory_is_named_per_repository(self) -> None:
        """リポ名は導出しない（`karume-lucida` はモデル名からは決まらない綴り）。"""
        assert default_out_dir(PIPELINE, [BIREFNET_DEFAULT_MODEL]).name == "karume-birefnet-hr"
        assert default_out_dir(LUCIDA_PIPELINE, [BIREFNET_DEFAULT_MODEL]).name == "karume-lucida"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    @pytest.mark.parametrize("model", BIREFNET_MODELS)
    def test_each_pipeline_looks_only_at_its_own_checkpoints_series(
        self, tmp_path: Path, model: str
    ) -> None:
        """checkpoint は席が固定で持つ — 席を跨いだ取り違えは綴りとして作れない。

        `LICENSE.md` の著作権行は Pipeline に固定で載る 1 組なので、HR の席が Lucida の系列を
        掴むと「自分のものでない著作権を名乗る配布」になる。どちらも散文としては妥当なまま
        `verify_dist` を通るため、**系列 path が席の checkpoint からしか決まらない**ことが
        唯一の防波堤になる（`--model` は解像度で、リポの中身を選べない）。
        """
        for pipeline, checkpoint in (
            (PIPELINE, BIREFNET_HR_CHECKPOINT),
            (LUCIDA_PIPELINE, BIREFNET_LUCIDA_CHECKPOINT),
        ):
            with pytest.raises(DistError) as raised:
                pipeline.plan(tmp_path, model)
            # 最初に探すのは既定席の系列（HR = f16 系列・Lucida = f32 系列）。
            expected = birefnet_series_name(checkpoint, model, birefnet_default_quant(checkpoint))
            assert expected in str(raised.value)

    @pytest.mark.parametrize("pipeline", [PIPELINE, LUCIDA_PIPELINE])
    def test_each_pipeline_refuses_a_model_outside_the_published_set(self, tmp_path, pipeline):
        with pytest.raises(DistError, match="は配らない"):
            pipeline.plan(tmp_path, "512")

    def test_the_series_name_carries_the_upstream_name_and_the_resolution(
        self, tmp_path: Path
    ) -> None:
        """系列名は上流リポ名 + 解像度（`birefnet.export.default_out_dir` と同じ式）。"""
        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            name = repo.split("/", 1)[1].lower().replace("_", "-")
            for model in BIREFNET_MODELS:
                expected = f"{name}-{model}"
                assert birefnet_series_name(checkpoint, model) == expected
                sources = birefnet_sources(tmp_path, checkpoint, model)
                assert {dtype: path.name for dtype, path in sources.series_by_dtype.items()} == {
                    dtype: expected if dtype == BIREFNET_PLAIN_DTYPE else f"{expected}-{dtype}"
                    for dtype in birefnet_seat_dtypes(checkpoint)
                }

    def test_it_refuses_a_checkpoint_it_has_no_attribution_for(self, tmp_path: Path) -> None:
        """帰属表に無い checkpoint は「出所を名乗れない」ので、系列を探す前に落とす。"""
        with pytest.raises(DistError, match="知らない"):
            birefnet_sources(tmp_path, "tiny")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist

        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")

        main(["--pipeline", "lucida", "--series", str(sources.series_by_dtype["f32"].parent)])

        out_dir = tmp_path / "models" / "karume-lucida"
        expected = _in_subtree(BIREFNET_DEFAULT_MODEL, _placed_paths(BIREFNET_LUCIDA_CHECKPOINT))
        assert sorted(verify_dist(out_dir)) == sorted(expected)


class TestBirefnetLegalText:
    """配布リポ直下の MIT 原文と改変告知（ADR 0092 決定 7）。"""

    def _legal(self, tmp_path: Path, checkpoint: str) -> tuple[str, str]:
        _build_birefnet_sources(tmp_path, checkpoint=checkpoint)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                _PIPELINE_NAMES[checkpoint],
                "--series",
                str(tmp_path / "outputs" / "series"),
                "--out",
                str(out_dir),
            ]
        )
        return (
            # LICENSE はバイト列を decode して返す（read_text の改行変換で CRLF が畳まれると
            # バイト同一の門が素通りする — 他家族の read_bytes 比較と同じ網の粗さにする）。
            (out_dir / "LICENSE.md").read_bytes().decode("utf-8"),
            (out_dir / "NOTICE.md").read_text(encoding="utf-8"),
        )

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_it_ships_the_license_text_byte_identical(
        self, tmp_path: Path, checkpoint: str
    ) -> None:
        """MIT §「著作権表示と許諾表示を含めること」— 差し込み口以外は本文テンプレそのもの。

        原本は `_shared/licenses/mit.txt` に {@link BIREFNET_COPYRIGHTS} を差し込んだ結果。
        組み立ての経路のどこかで整形や改行変換が入ると 1 バイト動くが、散文としては妥当な
        ままなので他の門は素通りする。
        """
        license_text, _ = self._legal(tmp_path, checkpoint)
        assert license_text == mit_license(BIREFNET_COPYRIGHTS[checkpoint])

    def test_the_hr_repository_carries_only_the_upstream_copyright(self, tmp_path: Path) -> None:
        license_text, notice = self._legal(tmp_path, BIREFNET_HR_CHECKPOINT)
        assert "MIT License" in license_text
        assert "Copyright (c) 2024 ZhengPeng" in license_text
        # 自分のものでない著作権を名乗らない（fine-tune 側は別リポの事実）。
        assert "egeorcun" not in license_text
        assert (
            "The above copyright notice and this permission notice shall be included in all"
            in license_text
        )
        assert BIREFNET_UPSTREAM[BIREFNET_HR_CHECKPOINT] in notice
        assert "no quantization" in notice.lower()

    def test_the_lucida_repository_carries_both_copyrights(self, tmp_path: Path) -> None:
        """MIT は派生でも上流の著作権表示を落とせない（上流カードの自己申告と同じ向き）。"""
        license_text, notice = self._legal(tmp_path, BIREFNET_LUCIDA_CHECKPOINT)
        assert "Copyright (c) 2026 egeorcun" in license_text
        assert "Copyright (c) 2024 ZhengPeng" in license_text
        assert BIREFNET_UPSTREAM[BIREFNET_LUCIDA_CHECKPOINT] in notice


class TestBirefnetWeightDtypes:
    """格納 dtype の軸（ADR 0113 — 席は checkpoint ごとに上流の格納から導く・系列は dtype ごとに
    別 root）。"""

    def test_the_seats_follow_the_upstream_storage(self) -> None:
        """上流の格納より広い席は出さず、既定は格納と同じ幅（ADR 0113 追記 2026-10-11）。

        HR の上流は f16 だけ — f32 席は同じ値を倍の常駐で持つだけなので出さない。Lucida の上流は
        f32 — f32 席（既定）と、丸めた f16 席。
        """
        assert birefnet_seat_dtypes(BIREFNET_HR_CHECKPOINT) == ("f16",)
        assert birefnet_default_quant(BIREFNET_HR_CHECKPOINT) == "f16"
        assert birefnet_seat_dtypes(BIREFNET_LUCIDA_CHECKPOINT) == ("f32", "f16")
        assert birefnet_default_quant(BIREFNET_LUCIDA_CHECKPOINT) == "f32"

    def test_each_dtype_has_its_own_output_path(self) -> None:
        """格納 dtype をファイル名に出す（1 ディレクトリに 2 系列が並んでも取り違えない綴り）。"""
        assert {
            birefnet_role("f32"): f"{BIREFNET_ROLE}/model.f32.krm",
            birefnet_role("f16"): f"{BIREFNET_ROLE}/model.f16.krm",
        } == BIREFNET_OUTPUT_PATHS

    def test_every_seat_requires_its_own_storage(self) -> None:
        assert {
            birefnet_role(dtype): dtype for dtype in BIREFNET_WEIGHT_DTYPES
        } == BIREFNET_STORAGE_REQUIREMENTS

    def test_only_the_plain_seat_has_a_forbidden_table(self) -> None:
        """f16 席の取り違え（f32 系列）は要求検査で落ちるので、禁止表は f32 席だけ。"""
        assert set(BIREFNET_STORAGE_FORBIDDEN) == {birefnet_role(BIREFNET_PLAIN_DTYPE)}

    def test_the_series_root_carries_the_dtype_suffix(self, tmp_path: Path) -> None:
        """f32 は接尾なし（既存系列の綴り）・f16 は `-f16`（`birefnet.export.default_out_dir`）。"""
        sources = birefnet_sources(tmp_path, BIREFNET_LUCIDA_CHECKPOINT, BIREFNET_DEFAULT_MODEL)
        assert {dtype: path.name for dtype, path in sources.series_by_dtype.items()} == {
            "f32": "lucida-1024",
            "f16": "lucida-1024-f16",
        }

    def test_hr_looks_only_at_its_f16_series(self, tmp_path: Path) -> None:
        """HR の席は f16 だけ — 手元に f32 系列（`birefnet-hr-1024`）があっても探さない。"""
        sources = birefnet_sources(tmp_path, BIREFNET_HR_CHECKPOINT, BIREFNET_DEFAULT_MODEL)
        assert {dtype: path.name for dtype, path in sources.series_by_dtype.items()} == {
            "f16": "birefnet-hr-1024-f16",
        }

    def test_hr_does_not_take_an_f32_series(self, tmp_path: Path) -> None:
        """HR の計画に f32 系列を渡しても組まない（値を広げただけの席を配らない）。

        出所も寸法も HR のものでも、席に無い dtype の系列は落とす — 落とさないと weights の宣言に
        無い配置が紛れ込むか、席の規則を迂回した f32 席ができる。
        """
        hr = _build_birefnet_sources(tmp_path / "hr")
        widened = _build_birefnet_sources(
            tmp_path / "widened",
            checkpoint=BIREFNET_LUCIDA_CHECKPOINT,
            provenance=_baked_provenance(BIREFNET_HR_CHECKPOINT),
        )
        sources = BirefnetSources(
            series_by_dtype={
                "f32": widened.series_by_dtype["f32"],
                "f16": hr.series_by_dtype["f16"],
            }
        )

        with pytest.raises(DistError, match=r"の席 \['f16'\] と違う"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_dtype_it_does_not_ship(self) -> None:
        with pytest.raises(DistError, match="は配らない"):
            birefnet_series_name(BIREFNET_HR_CHECKPOINT, BIREFNET_DEFAULT_MODEL, "i8")

    def test_it_refuses_the_plain_series_in_the_f16_seat(self, tmp_path: Path) -> None:
        """f16 席に f32 系列（`--dtype f16` を付け忘れた書き出し）— 要求検査が落とす。"""
        sources = _build_birefnet_sources(tmp_path, f16_storage="f32")
        with pytest.raises(DistError, match=rf"{birefnet_role('f16')}: .* f16 が無い"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_when_the_f16_series_is_missing(self, tmp_path: Path) -> None:
        """片方の系列だけでは組まない（f16 席が宣言だけあって中身の無い配布形を作らない）。"""
        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        for path in sources.series_by_dtype["f16"].iterdir():
            path.unlink()
        sources.series_by_dtype["f16"].rmdir()

        with pytest.raises(DistError) as raised:
            birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)
        assert sources.series_by_dtype["f16"].name in str(raised.value)

    def test_it_checks_the_graph_of_every_dtype_series(self, tmp_path: Path) -> None:
        """MUST: f16 系列も寸法を見る（別解像度で焼いた f16 系列が同じモデル名で同居しない）。"""
        sources = _build_birefnet_sources(
            tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT, f16_side=512
        )
        with pytest.raises(DistError, match=rf"モデル '{BIREFNET_DEFAULT_MODEL}' が名乗る"):
            birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_every_seat_has_its_display_fields(self, checkpoint: str) -> None:
        """quant 席の表示欄は（席, 上流の格納）の組で引く別表 — 席の集合と鍵がずれると、weights
        だけ宣言して席が無い（または中身の無い席がある）配布形になる（2026-09-26 レビュー F8）。"""
        stored = BIREFNET_CHECKPOINTS[checkpoint].stored_dtype
        seats = birefnet_seat_dtypes(checkpoint)

        assert all((dtype, stored) in BIREFNET_QUANT_SEATS for dtype in seats)
        assert tuple(birefnet_quants(checkpoint)) == seats

    def test_the_display_table_has_no_seat_that_no_repository_ships(self) -> None:
        """表示欄の表に、どの checkpoint も配らない組（HR の f32 席など）を残さない。"""
        shipped = {
            (dtype, entry.stored_dtype)
            for name, entry in BIREFNET_CHECKPOINTS.items()
            for dtype in birefnet_seat_dtypes(name)
        }

        assert set(BIREFNET_QUANT_SEATS) == shipped

    @pytest.mark.parametrize(
        ("checkpoint", "default"),
        [(BIREFNET_HR_CHECKPOINT, "f16"), (BIREFNET_LUCIDA_CHECKPOINT, "f32")],
    )
    def test_the_quant_seats_carry_display_fields_and_the_storage_width_is_the_default(
        self, tmp_path: Path, checkpoint: str, default: str
    ) -> None:
        """表示欄（ADR 0075）は席の表から manifest へ同じ文字列で入る。既定は上流の格納と同じ幅。"""
        sources = _build_birefnet_sources(tmp_path, checkpoint=checkpoint)
        out_dir = tmp_path / "models" / birefnet_repo_name(checkpoint)
        manifest = assemble_family(
            [birefnet_plan(sources, checkpoint)], out_dir, BIREFNET_DEFAULT_MODEL
        )
        model = _birefnet_model(manifest)
        stored = BIREFNET_CHECKPOINTS[checkpoint].stored_dtype

        for name, quant in model["quants"].items():
            seat = BIREFNET_QUANT_SEATS[(name, stored)]
            assert quant["label"] == seat.label
            assert quant["description"] == seat.description
        assert model["defaultQuant"] == default

    def test_a_single_seat_does_not_compare_itself_with_a_seat_that_is_not_there(self) -> None:
        """HR の席は f16 1 つ — 表示欄が「smaller download」「half-size」のように f32 席との比較を
        名乗ると、選択 UI に比べる相手の無い文が出る。"""
        (quant,) = birefnet_quants(BIREFNET_HR_CHECKPOINT).values()
        text = f"{quant['label']} {quant['description']}".lower()

        for word in ("smaller", "larger", "largest", "less", "more", "most", "half"):
            assert re.search(rf"\b{word}\b", text) is None, word

    def test_the_card_measured_a_seat_each_repository_ships(self) -> None:
        """カードの実行資源（測った席の数として名乗る）は、そのリポが配る席で測ったもの。"""
        for name, entry in BIREFNET_CHECKPOINTS.items():
            assert entry.resources.quant in birefnet_seat_dtypes(name)

    def test_the_quality_record_is_not_distributed(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        assert list(out_dir.rglob("quality.json")) == []


class TestBirefnetStorageNotice:
    """NOTICE の格納の説明は配る席と checkpoint の格納から組む（中身と食い違わない）。"""

    def _notice(self, tmp_path: Path, checkpoint: str) -> str:
        return TestBirefnetLegalText()._legal(tmp_path, checkpoint)[1]

    @staticmethod
    def _prose(notice: str) -> str:
        """折り返しを畳んだ本文（文言の検査は改行の位置に依らない）。"""
        return " ".join(notice.split())

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_the_storage_lines_are_wrapped_like_the_rest_of_the_template(
        self, tmp_path: Path, checkpoint: str
    ) -> None:
        notice = self._notice(tmp_path, checkpoint)
        assert max(len(line) for line in notice.splitlines()) <= 100

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_it_names_what_stays_f32_in_the_f16_seat(self, tmp_path: Path, checkpoint: str) -> None:
        """f16 格納は linear / conv / embedding（相対位置表）。f32 に残るのは deform の sampling
        weights（`regular_conv` — offset / modulator conv は f16）と bias / norm で、BN の α / β は
        統計から導いた値（2026-09-26 レビュー A-3 / F4）。"""
        prose = self._prose(self._notice(tmp_path, checkpoint))
        assert "linear, convolution and embedding (relative-position) weights" in prose
        assert "sampling weights (`regular_conv`)" in prose
        assert "derived per-channel affine in f32" in prose
        assert "the deformable convolutions' weights," not in prose

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_every_seat_is_described_and_no_other(self, tmp_path: Path, checkpoint: str) -> None:
        """NOTICE はそのリポの席だけを説明する（HR は f16 だけ — 配らない f32 席を名乗らない）。"""
        notice = self._notice(tmp_path, checkpoint)
        seats = birefnet_seat_dtypes(checkpoint)
        for dtype in BIREFNET_WEIGHT_DTYPES:
            assert (f"  - `{dtype}`:" in notice) is (dtype in seats), dtype

    def test_the_hr_f16_seat_is_lossless_because_the_checkpoint_is_f16(
        self, tmp_path: Path
    ) -> None:
        """HR の checkpoint 自身が f16 — 唯一の席（f16）は値を変えない。"""
        notice = self._prose(self._notice(tmp_path, BIREFNET_HR_CHECKPOINT))
        assert "the checkpoint itself is stored in f16" in notice
        # 席が 1 つなので単数で名乗る（「every quant seat」は席が 1 つのリポで読み手を迷わせる）。
        assert (
            "**No quantization**: the quant seat stores the source checkpoint's own values"
            in notice
        )
        assert "`f16`: the checkpoint's own f16 values" in notice
        assert "**quantized**" not in notice
        # f32 席は配らない（値を正確に広げるだけの席 — ADR 0113 追記 2026-10-11）。
        assert "widened exactly from f16 to f32" not in notice
        # 旧文面「source checkpoint's own f32 values」は HR では事実と違った（上流は f16）。
        assert "own f32 values" not in notice

    def test_the_lucida_f16_seat_is_declared_quantized(self, tmp_path: Path) -> None:
        """Lucida は f32 の checkpoint — f16 席は丸めた値で、そう名乗る（無損失を装わない）。"""
        notice = self._prose(self._notice(tmp_path, BIREFNET_LUCIDA_CHECKPOINT))
        assert "no quantization" not in notice.lower()
        assert "`f16`: **quantized**" in notice
        assert "`f32`: the checkpoint's own f32 values, unchanged." in notice
