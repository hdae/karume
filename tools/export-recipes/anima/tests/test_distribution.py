"""Anima の配布 recipe（`anima.distribution`）— 系列 → 配布形の組み立て。

実資産は使わない。weights の席へ挿すのは数 KB の**正当な最小 IR コンテナ**（`ir_fixtures` —
{@link _weights_container}）で、rope 素表のような extras の席と、門に落とされることを見る
ケースだけが従来の偽資産のまま。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

汎用エンジン側（合成計画だけで観測できる層 — 規模上限・quant 完全写像・staging/swap の
不変条件）は `tools/exporter/tests/test_dist.py` に残っている。ここに居るのは **Anima の
recipe を通さないと作れないケース**だけ。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ir_fixtures import ir_container
from shard_series import (
    placed_paths,
    read_component,
    replace_component,
    shard_paths,
    write_component,
)

from anima.card import ATTRIBUTION_NOTICE
from anima.distribution import (
    ANIMA_AESTHETIC_MODEL_NAME,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_DEFAULT_QUANT,
    ANIMA_MODELS,
    ANIMA_QUANT_ABBREVIATIONS,
    ANIMA_QUANTS,
    ANIMA_STORAGE_FORBIDDEN,
    ANIMA_TURBO_MODEL_NAME,
    ANIMA_TURBO_PIPELINE_CONFIG,
    ANIMA_WEIGHTS,
    CALIB_PROVENANCE_FILE,
    EXTRA_MODELS,
    EXTRA_NOTICE_MARKDOWN,
    EXTRA_PIPELINE,
    LICENSE_SOURCE_PATH,
    LORA_PROVENANCE_FILE,
    OFFICIAL_MODELS,
    OFFICIAL_NOTICE_MARKDOWN,
    OFFICIAL_PIPELINE,
    OUTPUT_PATHS,
    STORAGE_REQUIREMENTS,
    AnimaModel,
    AnimaSources,
    anima_dist_plan,
    anima_model,
    anima_plan,
    anima_quants,
    anima_sources,
    anima_weights,
)
from dist import main
from karume.artifacts import STAGING_SUFFIX
from karume.dist import (
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MODEL_CARD_FILENAME,
    SHARED_DIRNAME,
    Artifact,
    DistError,
    assemble_family,
    assert_model_name,
    assert_quant_presentation,
    materialize,
    resolve_card_renderer,
    verify_dist,
)
from karume.shards import resolve_shards


def _fake_safetensors(
    dtype: str, payload: bytes, metadata: Mapping[str, str] | None = None
) -> bytes:
    """格納 dtype の門を通る最小の safetensors（8 バイト長 + ヘッダ JSON + データ節）。

    `metadata` を渡すと `__metadata__` 節が付く（IR コンテナを要求する門のため）。
    """
    header: dict[str, Any] = {
        "w": {"dtype": dtype, "shape": [len(payload)], "data_offsets": [0, len(payload)]}
    }
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + payload


def _mixed_safetensors(dtypes: tuple[str, ...], payload: bytes) -> bytes:
    """複数の格納 dtype が同居するヘッダ（混成系列 = i4 の実物の形）。

    i4 系列は「適格な重みが I4・残りが I8・適格外と scale が F32」の 3 種が並ぶので、単一 dtype
    の偽資産では**圧縮席どうしの取り違え**（i4 系列 → i8 席）を再現できない。
    """
    header: dict[str, Any] = {}
    for index, dtype in enumerate(dtypes):
        start = index * len(payload)
        header[f"w{index}"] = {
            "dtype": dtype,
            "shape": [len(payload)],
            "data_offsets": [start, start + len(payload)],
        }
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + payload * len(dtypes)


def _weights_container(role: str, storage: str) -> list[bytes]:
    """weights の席へ挿す**正当な IR コンテナ**（役割ごとに違うバイト列）。

    組み立ては入力コンテナを IR v1 の全規則で見る
    （`karume.dist.assert_weight_components_verified`）ので、weights の席は本物でなければ
    ならない。格納 dtype の集合は実物と同じ形（適格な重みだけが圧縮・bias / 定数 / scale は
    F32・i4 は I4 + I8 + F32 の混成）になるので、{@link ANIMA_STORAGE_FORBIDDEN} の
    不在検査もこの形に掛かる。
    """
    return ir_container(mark=role, storage=storage)


#: 偽資産の中身（役割ごとに違うバイト列 — 取り違えがハッシュで見える）。モデル 5 役は
#: `STORAGE_REQUIREMENTS` が要求する dtype をヘッダに持つ。rope_base は weights ではなく
#: extras の席（IR コンテナではない）なので、ヘッダだけの偽資産のままでよい。
_PAYLOADS = {
    "text_encoder": _weights_container("text-encoder", "f16"),
    "text_conditioner": _weights_container("text-conditioner", "f16"),
    "transformer_f16": _weights_container("transformer-f16", "f16"),
    "transformer_i8": _weights_container("transformer-i8", "i8"),
    "transformer_i4": _weights_container("transformer-i4", "i4"),
    "rope_base": _fake_safetensors("F32", b"rope-base-table"),
    "vae_decoder": _weights_container("vae-decoder", "f16"),
    "tokenizer": b'{"qwen2": true}',
    "tokenizer_2": b'{"t5": true}',
}


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


#: 「焼いた記録」の偽値（実在の LoRA と重ならない sha256）。**どのモデルも LoRA を焼かなく
#: なった**（2026-09-01 — 公式 Turbo checkpoint 化で焼き込みが消えた）ので、この記録は
#: 「在ってはならないものが在る」side を作るためだけに書く。
FAKE_LORA_SHA256 = "9" * 64


def _bake_lora_record(series: Path, sha256: str = FAKE_LORA_SHA256) -> Path:
    """旧 fused 系列が持っていた帰属の記録を 1 本の系列へ置く（実物と同じ形）。

    旧 `anima-turbo`（LoRA 焼き込み）の系列は `outputs/series/` にまだ残っているので、
    新しい席へ挿し込む取り違えは**実際に起こしうる**。融合済みと素の資産は形が 1 バイトも
    変わらないので、記録の**不在**を見る門だけがこれを捕まえる。
    """
    path = series / "transformer" / LORA_PROVENANCE_FILE
    _write(
        path,
        json.dumps({"file": "anima-turbo-lora-v0.2.safetensors", "sha256": sha256}).encode("utf-8"),
    )
    return path


def _calib_record(method: str, model: str) -> bytes:
    """`anima/export.py` が i4 系列へ残す校正条件の記録（実物と同じ形）。

    step / CFG は**モデル別**（`anima.calib.calib_conditions` が `pipeline_config` から導く）
    ので、身代わりも同じ 1 箇所から引く — 値を写すと「どのモデルの条件で焼いたか」を見る門を
    試せない（`--model` を取り違えて焼いた系列がここでも作れなくなる）。
    """
    defaults = anima_model(model).pipeline_config["defaults"]
    return json.dumps(
        {
            "method": method,
            "group_size": 32,
            "grid": "rtn",
            "prompts": 4,
            "resolution": 512,
            "steps": int(defaults["steps"]),
            "guidance": float(defaults["guidanceScale"]),
            "text_dtype": "f16",
        }
    ).encode("utf-8")


def _build_series(
    series_dir: Path,
    *,
    model: str = ANIMA_TURBO_MODEL_NAME,
    with_i4: bool = False,
    i8_rope: bytes | None = None,
    i4_rope: bytes | None = None,
    mark: bytes = b"",
    calib_method: str = "gptq",
    calib_model: str | None = None,
) -> AnimaSources:
    """系列レイアウト（`outputs/series/` 相当）を偽資産で再現する（`io.*` の混入込み）。

    `mark` は transformer 系列だけに混ぜる差分 — ファミリー組み立てで「モデルごとに違う重み」と
    「モデル間で同一の base 資産」を作り分けるための軸。`i8_rope` / `i4_rope` は rope 素表を
    f16 系列からずらす軸（系列ごとに独立に振れる — 網が全系列に掛かっていることを見るため）。
    `calib_method` は i4 系列の丸め方式をずらす軸（既定は配布可の `gptq`）。`calib_model` は
    **校正条件だけ**を別モデルのものへずらす軸（`--model` を取り違えて焼いた系列 — 重みも
    格納形も正しいまま条件だけが別）。

    `with_i4` は**受理集合に無い 3 本目の格納系列**を足す軸（{@link _i4_seat} の相方 —
    i4 の機構テスト専用）。綴りは `anima_sources` と同じ `<model>-i4-dyn` で、系列を作るのは
    ここ・席を宣言するのは spec 側という分担にしてある。

    LoRA 帰属の記録は**どのモデルでも書かない**（焼き込みが消えた 2026-09-01 以降の実物と
    同じ状態）— 記録が在る側は {@link _bake_lora_record} でテストが明示的に作る。
    """
    sources = anima_sources(series_dir, model)
    if with_i4:
        sources = replace(
            sources,
            transformer={**sources.transformer, "i4": series_dir / f"{model}-i4-dyn"},
        )
    write_component(sources.base / "text_encoder" / "model.safetensors", _PAYLOADS["text_encoder"])
    write_component(
        sources.text_conditioner / "text_conditioner" / "model.safetensors",
        _PAYLOADS["text_conditioner"],
    )
    write_component(sources.base / "vae_decoder" / "model.safetensors", _PAYLOADS["vae_decoder"])
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.base / "text_encoder" / "io.t005.safetensors", b"io-fixture")
    _write(sources.base / "vae_decoder" / "io.case0.safetensors", b"io-fixture")
    ropes = {"f16": None, "i8": i8_rope, "i4": i4_rope}
    for storage, series in sources.transformer.items():
        role = f"transformer_{storage}"
        payload = (
            _PAYLOADS[role]
            if not mark
            else _weights_container(f"{role}{mark.decode('utf-8')}", storage)
        )
        write_component(series / "transformer" / "model.safetensors", payload)
        _write(series / "transformer" / "io.s01024t0699.safetensors", b"io-fixture")
        rope = ropes[storage]
        _write(
            series / "transformer" / "rope_base.safetensors",
            _PAYLOADS["rope_base"] if rope is None else rope,
        )
    # 校正条件は i4 系列だけが持つ（f16 / i8 は校正の対象外）。
    if "i4" in sources.transformer:
        _write(
            sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE,
            _calib_record(calib_method, model if calib_model is None else calib_model),
        )
    _write(sources.tokenizers / "qwen2-tokenizer.json", _PAYLOADS["tokenizer"])
    _write(sources.tokenizers / "t5-tokenizer.json", _PAYLOADS["tokenizer_2"])
    return sources


#: ファミリー組み立てのフィクスチャが並べる 2 モデル（公式リポの素版 + Aesthetic）。
#: **同じリポに実際に同居する組**を使う — 受理集合が別リポへ分けた組み合わせで組み立てると、
#: フィクスチャと {@link TestPipelineMembership} が別のことを言う。
FAMILY_MODELS = (ANIMA_BASE_MODEL_NAME, ANIMA_AESTHETIC_MODEL_NAME)


#: 配布の i4 席は**全モデルから消えた**（2026-09-01 ユーザー裁定 — 旧 fused turbo が持って
#: いた最後の 1 席も公式 checkpoint 化で持ち越さない）。それでも i4 の**機構**は
#: 復活レバーとしてコードに残っている（3 本目の格納系列の配置・rope 素表の系列横断突合・
#: 校正条件の門・格納 dtype の要求と禁止・quant 席の導出）ので、席を注入した spec で
#: 機構だけを守る。
#:
#: MUST: 注入は `anima_plan(..., spec=…)` の口 1 つに閉じる — {@link ANIMA_MODELS} は 1 行も
#: 動かさない（動かすと「配布の受理集合」が実験のために揺れ、席の有無を見る門〈全モデル i4
#: なし〉が自分のフィクスチャを検査するだけになる）。`anima.eval_dist` が視認評価で使うのと
#: **同じ口**なので、機構が外れれば向こうも一緒に落ちる。
def _i4_seat(model: str = ANIMA_TURBO_MODEL_NAME) -> AnimaModel:
    spec = anima_model(model)
    assert "i4" not in spec.storages, "配布の受理集合に i4 席が戻っている（注入が要らない）"
    return replace(spec, storages=(*spec.storages, "i4"))


def _assemble_anima(
    sources: AnimaSources,
    out_dir: Path,
    model: str = ANIMA_TURBO_MODEL_NAME,
    *,
    spec: AnimaModel | None = None,
) -> dict[str, Any]:
    """単一モデルの組み立て（計画 → 実体化）を 1 行で回すテスト用の糊。

    `spec` は席を差し替える口（省略時はモデル名から引く = 配布経路）— 渡すのは i4 の機構を
    見るテストだけで、{@link _i4_seat} が作った「i4 席を足した spec」が入る。
    """
    return assemble_family([anima_plan(sources, model, spec=spec)], out_dir, model)


def _in_subtree(
    model: str, paths: Iterable[str] | None = None, *, storages: tuple[str, ...] | None = None
) -> list[str]:
    """モデルサブツリー内の期待 path（ADR 0041 §9 の一様レイアウト）。

    省略時はそのモデルが**宣言した格納形だけ**（i4 席を持たないモデルに i4 のファイルは出ない）
    を、配布形に現れる形へ展開する — weights の 5 役は shard 連番になり（ADR 0081）、
    rope_base（extras）と tokenizer（assets）は 1 ファイルのまま。`storages` は席を注入して
    組んだ木を見るときだけ渡す（{@link _i4_seat}）。
    """
    if paths is None:
        storages = anima_model(model).storages if storages is None else storages
        declared = {
            role: rel
            for role, rel in OUTPUT_PATHS.items()
            if not role.startswith("transformer_") or role.removeprefix("transformer_") in storages
        }
        paths = placed_paths(declared, ANIMA_WEIGHTS)
    return [f"{model}/{rel}" for rel in paths]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_series(tmp_path / "series")
    out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
    manifest = _assemble_anima(sources, out_dir)
    return out_dir, manifest


#: 席を注入して組んだときの格納形（{@link _i4_seat}）。
I4_SEAT_STORAGES = ("f16", "i8", "i4")


@pytest.fixture
def assembled_with_i4(tmp_path: Path) -> tuple[Path, dict]:
    """**受理集合には無い** i4 席を注入して組んだ配布形（機構の保存 — {@link _i4_seat}）。"""
    sources = _build_series(tmp_path / "series", with_i4=True)
    out_dir = tmp_path / "models" / "i4-seat"
    manifest = _assemble_anima(sources, out_dir, spec=_i4_seat())
    return out_dir, manifest


class TestLayout:
    def test_it_places_every_declared_path_under_the_model_subtree(self, assembled) -> None:
        out_dir, _ = assembled
        assert _present(out_dir) == sorted(
            [*_in_subtree(ANIMA_TURBO_MODEL_NAME), MANIFEST_FILENAME]
        )

    def test_it_never_carries_io_fixtures_into_the_distribution(self, assembled) -> None:
        out_dir, _ = assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_renames_the_two_transformer_series_into_dtype_files(self, assembled) -> None:
        out_dir, _ = assembled
        subtree = out_dir / ANIMA_TURBO_MODEL_NAME / "transformer"
        assert read_component(subtree / "model.f16.safetensors") == _PAYLOADS["transformer_f16"]
        assert read_component(subtree / "model.i8.safetensors") == _PAYLOADS["transformer_i8"]

    def test_no_shipped_model_places_an_i4_file(self, assembled) -> None:
        """配布の i4 席は全モデルから消えた（2026-09-01 裁定）— ファイルも 1 本も出ない。"""
        out_dir, _ = assembled
        assert not (out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["transformer_i4"]).exists()
        assert not any("i4" in name for name in _present(out_dir))

    def test_an_injected_i4_seat_gets_its_own_dtype_file(self, assembled_with_i4) -> None:
        """i4 は f16 / i8 と並ぶ 3 本目の格納席（同じ path へ載せると席が 1 つ消える）。

        席そのものは配布から降りたが、3 本目を別 path へ載せる**機構**は復活レバーとして
        残っている（`anima.eval_dist` が今まさに使う経路でもある）ので、席を注入して守る。
        """
        out_dir, _ = assembled_with_i4
        placed = out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["transformer_i4"]
        assert placed.name == "model.i4.safetensors"
        assert read_component(placed) == _PAYLOADS["transformer_i4"]
        assert _present(out_dir) == sorted(
            [
                *_in_subtree(ANIMA_TURBO_MODEL_NAME, storages=I4_SEAT_STORAGES),
                MANIFEST_FILENAME,
            ]
        )

    def test_a_single_model_repository_has_no_shared_directory(self, assembled) -> None:
        """`shared/` は 2 モデル以上が同じ中身を持ったときだけ現れる席（ADR 0041 §5）。"""
        out_dir, _ = assembled
        assert not (out_dir / SHARED_DIRNAME).exists()

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        _assemble_anima(sources, out_dir)
        _assemble_anima(sources, out_dir)  # 既存リンクがあっても落ちない
        assert verify_dist(out_dir)

    def test_the_model_name_moves_the_whole_subtree(self, tmp_path: Path) -> None:
        """`--model` はサブツリー名・系列名・manifest のキーを 1 語で動かす。"""
        sources = _build_series(tmp_path / "series", model="anima-wai-v1.0")
        out_dir = tmp_path / "models" / "anima-wai-v1.0"
        manifest = _assemble_anima(sources, out_dir, "anima-wai-v1.0")
        assert list(manifest["models"]) == ["anima-wai-v1.0"]
        assert manifest["defaultModel"] == "anima-wai-v1.0"
        assert _present(out_dir) == sorted([*_in_subtree("anima-wai-v1.0"), MANIFEST_FILENAME])


class TestPlacementStrategy:
    def test_it_places_independent_copies(self, assembled) -> None:
        """配布形はハードリンクを持たない（系列から独立した自己完結スナップショット）。"""
        out_dir, _ = assembled
        placed = out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["text_encoder"]
        assert read_component(placed) == _PAYLOADS["text_encoder"]
        # 独立コピーであることは shard 1 本ずつに掛かる（分割で漏れる席を作らない）。
        assert [shard.stat().st_nlink for shard in resolve_shards(placed)] == [1, 1]

    def test_a_series_rewrite_does_not_reach_the_dist(self, tmp_path: Path) -> None:
        """系列の再 export（truncate 上書き）が組み立て済み配布形へ波及しないこと。

        リンク方式ではここが破れていた — 同じ inode を共有するため、系列の書き直しが
        manifest の sha256 と現物を黙って食い違わせる。
        """
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        _assemble_anima(sources, out_dir)
        # 書き直すのは系列の**グラフ shard**（再 export は shard ごとに truncate 上書きする）。
        source = resolve_shards(sources.base / "text_encoder" / "model.safetensors")[0]
        with source.open("wb") as handle:
            handle.write(_fake_safetensors("F16", b"rewritten-after-assembly"))
        placed = out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["text_encoder"]
        assert read_component(placed) == _PAYLOADS["text_encoder"]

    def test_it_stops_when_an_input_is_missing(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        for shard in resolve_shards(sources.base / "vae_decoder" / "model.safetensors"):
            shard.unlink()
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            _assemble_anima(sources, tmp_path / "models" / ANIMA_TURBO_MODEL_NAME)


class TestRopeBase:
    def test_it_collapses_the_two_series_into_one_file(self, assembled) -> None:
        out_dir, manifest = assembled
        entry = manifest["models"][ANIMA_TURBO_MODEL_NAME]["weights"]["transformer"]
        f16_extra = entry["f16"]["extras"]["rope_base"]
        i8_extra = entry["i8"]["extras"]["rope_base"]
        assert f16_extra == i8_extra
        assert f16_extra["path"] == f"{ANIMA_TURBO_MODEL_NAME}/{OUTPUT_PATHS['rope_base']}"
        assert (out_dir / f16_extra["path"]).read_bytes() == _PAYLOADS["rope_base"]

    def test_it_refuses_to_pick_a_side_when_the_series_disagree(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series", i8_rope=b"rope-base-table-DIFFERENT")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        with pytest.raises(DistError, match="バイト同一でない"):
            _assemble_anima(sources, out_dir)
        # 止めた以上、途中の配布形を残さない（片方だけ入った出力を後段に見せない）。
        assert not (out_dir / MANIFEST_FILENAME).exists()

    def test_the_check_reaches_a_third_series_too(self, tmp_path: Path) -> None:
        """網は**宣言された全系列**に掛かる（2 本目までで止まっていると 3 本目が漏れる）。

        漏れた系列の quant は「別の幾何の rope 表」で走り、ロードも実行も通って絵だけが
        静かに壊れる。3 本目は配布から降りた i4 席を注入して踏む（{@link _i4_seat}）。
        """
        sources = _build_series(
            tmp_path / "series", with_i4=True, i4_rope=b"rope-base-table-DIFFERENT"
        )
        out_dir = tmp_path / "models" / "i4-seat"
        with pytest.raises(DistError, match="バイト同一でない"):
            _assemble_anima(sources, out_dir, spec=_i4_seat())
        assert not (out_dir / MANIFEST_FILENAME).exists()


class TestLoraProvenance:
    """**どのモデルも LoRA を焼いていない**ことを、系列に記録が無いことで確かめる。

    2026-09-01 の再構造で公式 Turbo が checkpoint 配布になり、焼き込みは受理集合から消えた
    （`ANIMA_MODELS` の `lora_sha256` は全て `None`）。融合済みと素の資産は形が 1 バイトも
    変わらないので、旧 fused 系列（`outputs/series/` に現存する）を新しい席へ挿し込む取り違えは
    他のどの検査にも掛からない — 「記録が無いこと」の検査だけが唯一の網。
    """

    def test_no_model_declares_a_baked_lora_any_more(self) -> None:
        """受理集合の側の事実（`None` = 記録の不在を積極検査する側へ全員が回る）。"""
        assert [spec.lora_sha256 for spec in ANIMA_MODELS.values()] == [None] * len(ANIMA_MODELS)

    def test_it_stops_when_the_official_turbo_gets_a_series_with_a_baked_lora(
        self, tmp_path: Path
    ) -> None:
        """旧 `anima-turbo`（LoRA 焼き込み）の系列を公式 Turbo の席へ挿す取り違え。"""
        sources = _build_series(tmp_path / "series")
        _bake_lora_record(sources.transformer["f16"])
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME

        with pytest.raises(DistError, match="焼いた記録のある系列が来ている"):
            _assemble_anima(sources, out_dir)

        # 計画段の検査なので配布形は 1 ファイルも生えない。
        assert not out_dir.exists()

    def test_it_stops_when_a_plain_model_gets_a_series_with_a_baked_lora(
        self, tmp_path: Path
    ) -> None:
        """素版の席でも同じ 1 実装が掛かる（網はモデルではなく系列に掛かる）。"""
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        _bake_lora_record(sources.transformer["f16"])
        out_dir = tmp_path / "models" / ANIMA_BASE_MODEL_NAME

        with pytest.raises(DistError, match="焼いた記録のある系列が来ている"):
            _assemble_anima(sources, out_dir, ANIMA_BASE_MODEL_NAME)

        assert not out_dir.exists()

    def test_the_check_reaches_every_transformer_series(self, tmp_path: Path) -> None:
        """網が f16 の 1 本に留まっていると、他の格納席に紛れた記録が素通りする。

        旧 fused 系列は f16 / i8 / i4 の 3 本が並んでいるので、配布側の格納席（今は f16 /
        i8 の 2 本）へどれか 1 本だけを差し替える取り違えは実際に起こしうる形。列挙元を
        宣言側へ縛るのは、席が増えた日に自動で網が広がるようにするため。
        """
        for storage in ANIMA_MODELS[ANIMA_TURBO_MODEL_NAME].storages:
            sources = _build_series(tmp_path / f"series-{storage}")
            _bake_lora_record(sources.transformer[storage])

            with pytest.raises(DistError, match="焼いた記録のある系列が来ている"):
                _assemble_anima(sources, tmp_path / "models" / storage)

    # NOTE: 旧「記録は配布形へ持ち出さない」テストはこの波で削除した。記録を持つ系列は
    # 上の門で計画段に落ちるので、組み立て済みの木に記録が無いのは**恒真**になった
    # （系列側の記録が配布へ出ないことは、校正記録の同名テストと `io.*` の混入テストが
    # 実在のファイルで見ている）。


class TestModelVariants:
    """モデルごとに違う事実（席の範囲・text_conditioner の出所・既定の step / CFG）。

    公式 3 変種（Turbo / base / Aesthetic）と第三者 fine-tune 2 本が同じ 1 実装を通る
    （ADR 0087）— 違いは {@link ANIMA_MODELS} の宣言だけで、組み立ての経路は 1 本。
    """

    def test_a_plain_model_excludes_the_i4_seat(self, tmp_path: Path) -> None:
        """素版は i4 の席もファイルも**持たない**（0.5.0 の除外裁定 — `ANIMA_MODELS` の NOTE）。

        視認裁定（2026-08-24 配布スキップ → 2026-08-25 除外）の反転。「持たない」は quant 表と
        ファイル一覧の**両方**で見る — 片方だけだと、席は消えたがファイルだけ配られる（無駄な
        GB）/ ファイルは消えたが席が残る（選ぶと 404）のどちらかが緑になる。turbo が i4 を
        持ち続けることは TestQuant 系の既存門が見ている（あちらは別裁定で公開済み）。
        rtn 方式の校正門も turbo 側の既存門（丸め方式の受理検査）が被覆を保つ — 素版の i4 は
        構成として組めなくなったので、素版側の校正門テストはこの裁定で削除した。
        """
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        out_dir = tmp_path / "models" / ANIMA_BASE_MODEL_NAME
        manifest = _assemble_anima(sources, out_dir, ANIMA_BASE_MODEL_NAME)

        entry = manifest["models"][ANIMA_BASE_MODEL_NAME]
        assert sorted(entry["weights"]["transformer"]) == ["f16", "i8"]
        assert "f16+dit4" not in entry["quants"]
        assert "f16+dit4-attn8-s16" not in entry["quants"]
        assert not (out_dir / ANIMA_BASE_MODEL_NAME / OUTPUT_PATHS["transformer_i4"]).exists()

    def test_the_official_aesthetic_carries_the_same_two_storages(self, tmp_path: Path) -> None:
        """Aesthetic も f16 / i8 の 2 席（既定は視認裁定の 30 step / CFG 4 — 2026-09-01）。"""
        sources = _build_series(tmp_path / "series", model=ANIMA_AESTHETIC_MODEL_NAME)
        out_dir = tmp_path / "models" / ANIMA_AESTHETIC_MODEL_NAME
        manifest = _assemble_anima(sources, out_dir, ANIMA_AESTHETIC_MODEL_NAME)

        entry = manifest["models"][ANIMA_AESTHETIC_MODEL_NAME]
        assert sorted(entry["weights"]["transformer"]) == ["f16", "i8"]
        defaults = entry["pipelineConfig"]["defaults"]
        assert (defaults["steps"], defaults["guidanceScale"]) == (30, 4)

    def test_no_shipped_model_declares_an_i4_seat(self) -> None:
        """MUST: 配布の i4 席は**全モデルから**消えた（2026-09-01 ユーザー裁定 — ADR 0087）。

        旧 fused turbo だけが持っていた最後の 1 席も公式 checkpoint 化で持ち越さない裁定。
        席の取捨は宣言した格納形から導く（`anima_quants` / `anima_weights`）ので、
        ここを 1 モデルでも戻すと quant 表・ファイル・NOTICE の int4 記載が連動して動く —
        逆に言えば、宣言側のこの 1 行が席の唯一の正本。
        """
        assert [name for name, spec in ANIMA_MODELS.items() if "i4" in spec.storages] == []
        for name, spec in ANIMA_MODELS.items():
            assert spec.storages == ("f16", "i8"), name

    def test_the_official_variants_split_on_the_text_conditioner(self, tmp_path: Path) -> None:
        """text_conditioner を共有するのは実測でビット同一だった変種だけ（2026-09-01）。

        Turbo は base と f16 丸め後ビット同一なので共有系列を読み、Aesthetic は 64 テンソル
        差が残る実測なので自前で持つ — 取り違えると**別のモデルのテキスト条件付け**で走り、
        ロードも実行も通って絵だけが静かにずれる。
        """
        turbo = anima_sources(tmp_path / "series", ANIMA_TURBO_MODEL_NAME)
        aesthetic = anima_sources(tmp_path / "series", ANIMA_AESTHETIC_MODEL_NAME)

        assert turbo.text_conditioner == turbo.base
        assert aesthetic.text_conditioner != aesthetic.base
        assert aesthetic.text_conditioner.name == f"{ANIMA_AESTHETIC_MODEL_NAME}-f16"

    def test_a_fine_tune_reads_its_own_text_conditioner(self, tmp_path: Path) -> None:
        """第三者 fine-tune は llm_adapter も焼き直しているので、共有系列を読ませない。"""
        base = anima_sources(tmp_path / "series", ANIMA_BASE_MODEL_NAME)
        wai = anima_sources(tmp_path / "series", "anima-wai-v1.0")

        assert base.text_conditioner == base.base
        assert wai.text_conditioner != wai.base
        assert wai.text_conditioner.name == "anima-wai-v1.0-f16"

    def test_the_default_quant_stays_the_same_seat(self, tmp_path: Path) -> None:
        """既定席は turbo と揃える（利用者が model を変えても既定の意味が動かない）。"""
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        manifest = _assemble_anima(sources, tmp_path / "out", ANIMA_BASE_MODEL_NAME)

        assert manifest["models"][ANIMA_BASE_MODEL_NAME]["defaultQuant"] == ANIMA_DEFAULT_QUANT

    def test_a_plain_model_defaults_to_many_steps_with_guidance(self, tmp_path: Path) -> None:
        """CFG を使う既定であること — negative prompt が効くのはこの経路だけ。"""
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        manifest = _assemble_anima(sources, tmp_path / "out", ANIMA_BASE_MODEL_NAME)

        defaults = manifest["models"][ANIMA_BASE_MODEL_NAME]["pipelineConfig"]["defaults"]
        assert defaults["guidanceScale"] != 1
        assert defaults["steps"] > 8
        assert defaults["negativePrompt"]

    def test_the_default_model_is_the_official_turbo(self, tmp_path: Path) -> None:
        """公式リポの既定は Turbo（上流 README の「まず Turbo を」に合わせた 2026-09-01 裁定）。

        既定は `pipelineConfig` ごと動く軸（8 step / CFG 1 = 負プロンプトが効かない席）なので、
        既定モデルと既定の中身は**同じ 1 本のテスト**で押さえる。
        """
        assert OFFICIAL_PIPELINE.default_model == ANIMA_TURBO_MODEL_NAME
        assert OFFICIAL_MODELS[0] == ANIMA_TURBO_MODEL_NAME

        sources = _build_series(tmp_path / "series")
        manifest = _assemble_anima(sources, tmp_path / "out")

        defaults = manifest["models"][ANIMA_TURBO_MODEL_NAME]["pipelineConfig"]["defaults"]
        assert (defaults["steps"], defaults["guidanceScale"]) == (8, 1)


class TestPipelineMembership:
    """リポ直下の改変告知は Pipeline に固定で載る 1 組 — 取り違えて組めないようにする。

    分割の軸は**公式 / 追加学習**（2026-09-01 裁定 — ADR 0087）。公式リポは CircleStone の
    ライセンス 1 本だけ、追加学習リポは出所ページごとの許諾が重なるので、告知も出所節も違う。
    """

    def test_the_official_pipeline_refuses_a_community_fine_tune(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="この pipeline のリポに入らない"):
            anima_dist_plan(tmp_path / "series", "anima-wai-v1.0", OFFICIAL_MODELS)

    def test_the_extra_pipeline_refuses_an_official_model(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="この pipeline のリポに入らない"):
            anima_dist_plan(tmp_path / "series", ANIMA_TURBO_MODEL_NAME, EXTRA_MODELS)

    def test_the_two_lists_partition_the_accepted_models(self) -> None:
        """MUST: 全モデルがどちらか一方**だけ**に居る。

        重なると同じモデルが 2 つの告知の下で配られ、漏れると受理集合に居るのに
        どのリポからも組めないモデルが黙って生まれる（`--pipeline` の選択肢からは読めない）。
        """
        official, extra = set(OFFICIAL_MODELS), set(EXTRA_MODELS)

        assert official | extra == set(ANIMA_MODELS)
        assert official & extra == set()

    def test_the_two_repositories_declare_different_modifications(self) -> None:
        """告知が 1 本に畳まれていたら（= 同じ文面なら）どちらかが嘘になる。"""
        assert OFFICIAL_NOTICE_MARKDOWN != EXTRA_NOTICE_MARKDOWN
        assert "community fine-tune" in EXTRA_NOTICE_MARKDOWN
        assert "community fine-tune" not in OFFICIAL_NOTICE_MARKDOWN

    def test_neither_notice_claims_a_baked_lora(self) -> None:
        """焼き込みは 2026-09-01 に消えた — 告知に残ると改変内容の記載が事実と食い違う。

        §3(d)(i) が求めるのは「改変内容の告知」なので、していない改変を挙げるのは
        余計な文言ではなく**誤った告知**（値としては妥当な散文なので配ってから露見する）。
        """
        for notice in (OFFICIAL_NOTICE_MARKDOWN, EXTRA_NOTICE_MARKDOWN):
            assert "LoRA" not in notice

    def test_neither_notice_claims_an_int4_series(self) -> None:
        """配布に i4 席が 1 つも無い以上、「int4 系列も足した」は**していない改変の告知**。

        §3(d)(i) が求めるのは改変内容の告知なので、余分な 1 行ではなく誤りになる（旧 base
        告知が実際にこの状態だった — 2026-09-01 に是正）。告知は Pipeline に固定で載って
        manifest を見られないので、席の有無と文面は**別々に**動く = 突き合わせが要る。
        """
        for notice in (OFFICIAL_NOTICE_MARKDOWN, EXTRA_NOTICE_MARKDOWN):
            assert "int4" not in notice
            assert "int8-quantized series of the transformer was added" in notice
        assert not any("i4" in spec.storages for spec in ANIMA_MODELS.values())

    def test_an_unknown_model_name_is_refused_with_the_choices(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="知らない Anima のモデル名"):
            anima_sources(tmp_path / "series", "anima-nope")


class TestCalibProvenance:
    """i4 系列が配布して良い丸め（GPTQ 校正付き × 下限以上の予算 × このモデルの条件）で
    作られたことを、組み立て時に突き合わせる。

    校正の方式も予算も条件も**格納形を 1 バイトも変えない**（research 2026-08-21 §6 —
    ファイルサイズもバイト単位で同じ）ので、ヘッダ dtype の門も `verify_dist` の構造検査も
    素通りする。`--no-calib` / `--calib-prompts 1` は smoke 用の opt-out・`--model` は条件を
    引くだけのノブなのに、その生成物が配布へ紛れても資産からは読めず、出るのは
    「全体的にぼやけた」絵だけ — LoRA 帰属と同じ規律をここにも敷く。

    2026-09-01 の裁定で配布の i4 席は全モデルから消えたので、実在の spec ではこの門を 1 度も
    踏めない。門は復活レバー（席が戻った日に真っ先に効く安全網）であり、`anima.eval_dist` が
    今まさに使っている経路でもあるので、**席を注入した spec**（{@link _i4_seat}）で機構を守る。
    """

    def _plan(self, tmp_path: Path, **kwargs) -> tuple[AnimaSources, Path]:
        """i4 席を注入した系列と出力先（この class の全テストが同じ 1 経路を踏む）。"""
        return _build_series(tmp_path / "series", with_i4=True, **kwargs), (
            tmp_path / "models" / "i4-seat"
        )

    def _assemble(self, sources: AnimaSources, out_dir: Path) -> dict[str, Any]:
        return _assemble_anima(sources, out_dir, spec=_i4_seat())

    def test_it_stops_when_the_i4_series_was_rounded_without_calibration(
        self, tmp_path: Path
    ) -> None:
        """`--no-calib` の生成物（method = rtn）を名指しで拒否する。"""
        sources, out_dir = self._plan(tmp_path, calib_method="rtn")

        with pytest.raises(DistError, match="配布して良い丸め方式で作られていない"):
            self._assemble(sources, out_dir)

        # 計画段の検査なので配布形は 1 ファイルも生えない。
        assert not out_dir.exists()

    def test_it_stops_when_the_i4_series_carries_no_record_at_all(self, tmp_path: Path) -> None:
        """記録の無い系列（校正条件を突き合わせられない）も緑にしない。"""
        sources, out_dir = self._plan(tmp_path)
        (sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE).unlink()

        with pytest.raises(DistError, match="校正条件の記録が無い"):
            self._assemble(sources, out_dir)

    def test_it_stops_when_the_record_is_not_readable_json(self, tmp_path: Path) -> None:
        sources, out_dir = self._plan(tmp_path)
        (sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE).write_bytes(b"{oops")

        with pytest.raises(DistError, match="校正条件の記録を解析できない"):
            self._assemble(sources, out_dir)

    def test_a_record_written_before_the_guidance_field_is_still_accepted(
        self, tmp_path: Path
    ) -> None:
        """後方互換 MUST: 欄を足しても**既存の系列を作り直させない**。

        `guidance` は校正条件をモデル別化した 2026-08-23 に足した欄で、それ以前に採った
        turbo の i4 系列（当時 HF へ上げた現物）には無い。読み手が欄の存在を要求すると、この
        1 行の追加が「丸め時間ぶんの再 export」を既存系列へ課すことになる — 見るのは**在る欄
        だけ**に留める（`_shared.calib_provenance` の同 MUST）。
        """
        sources, out_dir = self._plan(tmp_path)
        path = sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE
        legacy = json.loads(path.read_text(encoding="utf-8"))
        del legacy["guidance"]
        path.write_text(json.dumps(legacy), encoding="utf-8")

        manifest = self._assemble(sources, out_dir)

        assert "f16+dit4" in manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]

    def test_it_stops_when_the_calibration_ran_on_a_smaller_budget(self, tmp_path: Path) -> None:
        """`--calib-prompts 1` は `method` を `gptq` のまま残す — 予算欄まで見ないと通る。

        丸めの格子は動かないのでファイルサイズは 1 バイトも変わらず、絵がぼやけるだけ。
        """
        sources, out_dir = self._plan(tmp_path)
        path = sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE
        record = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**record, "prompts": 1}), encoding="utf-8")

        with pytest.raises(DistError, match="校正予算 'prompts' が配布の下限を下回る"):
            self._assemble(sources, out_dir)

    def test_it_stops_when_the_series_was_calibrated_under_another_models_conditions(
        self, tmp_path: Path
    ) -> None:
        """`--model` の取り違えで焼いた i4 を名指しで拒否する。

        `anima.export` の `--model` は**校正条件を引くためだけ**のノブなので、turbo の重みを
        `--model anima-v1.0` で焼いた資産は「正しい turbo i4」に見える — 格納形も本数も
        LoRA 記録も全て正しく、素版の多 step・CFG で校正されていることだけが違う。
        （席の持ち主が入れ替わっても条件は `pipeline_config` 1 箇所から導くので、
        取り違えの向きは turbo→素版のまま同じ門を踏む。）
        """
        sources, out_dir = self._plan(tmp_path, calib_model=ANIMA_BASE_MODEL_NAME)

        with pytest.raises(DistError, match="校正条件 'steps' がこのモデルの既定と違う"):
            self._assemble(sources, out_dir)

    def test_it_stops_when_only_the_guidance_came_from_another_model(self, tmp_path: Path) -> None:
        """step が一致していても CFG がずれていれば落ちる（2 欄とも門に載っている）。"""
        sources, out_dir = self._plan(tmp_path)
        path = sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE
        record = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**record, "guidance": 4.0}), encoding="utf-8")

        with pytest.raises(DistError, match="校正条件 'guidance' がこのモデルの既定と違う"):
            self._assemble(sources, out_dir)

    def test_the_record_never_reaches_the_distribution(self, assembled_with_i4) -> None:
        """記録は系列側の事実 — 配布形（HF リポ）には持ち出さない。

        記録が実在する系列（= i4 席を注入した組み立て）で見る。配布経路には i4 系列自体が
        無いので、そちらで見ると「無いものが出てこない」を確かめるだけの恒真になる。
        """
        out_dir, _ = assembled_with_i4

        assert not any(name.endswith(CALIB_PROVENANCE_FILE) for name in _present(out_dir))


class TestStorageGate:
    """格納 dtype の門（実測の事故が根拠 — `--dtype` 付け忘れの素 F32 は PNG 門まで沈黙した）。"""

    def test_it_stops_when_an_f16_component_is_stored_as_raw_f32(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        replace_component(
            sources.base / "text_encoder" / "model.safetensors",
            _fake_safetensors("F32", b"text-encoder-weights"),
        )
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        with pytest.raises(DistError, match=r"text_encoder: .* F16 が無い"):
            _assemble_anima(sources, out_dir)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない（rope 不一致と同じ規律）。
        assert not out_dir.exists()

    def test_it_stops_when_the_i8_transformer_lacks_i8_storage(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        replace_component(
            sources.transformer["i8"] / "transformer" / "model.safetensors",
            _fake_safetensors("F16", b"transformer-i8-weights"),
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* I8 が無い"):
            _assemble_anima(sources, tmp_path / "models" / ANIMA_TURBO_MODEL_NAME)

    def test_it_stops_when_the_i4_transformer_lacks_i4_storage(self, tmp_path: Path) -> None:
        """i4 席へ i8 系列が入る取り違え — 要求が I8 のままだと素通りして沈黙する。

        席は配布から降りたが要求表（`STORAGE_REQUIREMENTS`）には残っているので、注入した
        席で門を守る（{@link _i4_seat}）。
        """
        sources = _build_series(tmp_path / "series", with_i4=True)
        replace_component(
            sources.transformer["i4"] / "transformer" / "model.safetensors",
            _fake_safetensors("I8", b"transformer-i4-weights"),
        )
        with pytest.raises(DistError, match=r"transformer_i4: .* I4 が無い"):
            _assemble_anima(sources, tmp_path / "models" / "i4-seat", spec=_i4_seat())

    def test_it_stops_when_the_i4_series_lands_in_the_i8_seat(self, tmp_path: Path) -> None:
        """逆向きの取り違え（i4 系列 → i8 席）— 存在検査だけでは**素通りする**。

        i4 系列は混成で既定格納が i8 なので必ず I8 を含み、「I8 を含む」を満たしてしまう。
        既定 quant `f16+dit8-a8-attn8-s16` が i4 常駐を掴むと、`c285f97` 以降の `a8` の述語は
        i4 も受ける
        （ADR 0076）ので fail loudly せず w4a8 の数値契約で走る — ADR 0076 決定 6 が席に
        載せないと決めた構成が既定席で沈黙して出る。禁止表（`ANIMA_STORAGE_FORBIDDEN`）が
        唯一の検出器。
        """
        sources = _build_series(tmp_path / "series")
        replace_component(
            sources.transformer["i8"] / "transformer" / "model.safetensors",
            _mixed_safetensors(("I4", "I8", "F32"), b"transformer-i4-weights"),
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* I4 がある"):
            _assemble_anima(sources, tmp_path / "models" / ANIMA_TURBO_MODEL_NAME)

    def test_no_transformer_series_slips_into_another_series_seat(self) -> None:
        """3 席 × 他 2 系列の**全ての**取り違えが、要求か禁止のどちらかで落ちる。

        席が増えた日に片方の表だけ更新されると、網から漏れた組み合わせが黙って配布形に並ぶ
        （系列 root の取り違えは数値の門では原理的に検出できない — ADR 0027 / 0029）。
        """
        #: 系列 → そのヘッダが**必ず含む**格納 dtype（i4 は混成で既定格納が i8 なので I8 も含む）。
        headers = {
            "transformer_f16": {"F32", "F16"},
            "transformer_i8": {"F32", "I8"},
            "transformer_i4": {"F32", "I8", "I4"},
        }
        # MUST: 列挙元を production の表へ縛る。ここをテスト内 dict のままにすると、4 本目の
        # 系列が `STORAGE_REQUIREMENTS` に生えて `headers` に足されなかったとき、docstring が
        # 名指しする失敗モードそのものを一度も見ないまま緑が残る。
        assert set(headers) == {
            role for role in STORAGE_REQUIREMENTS if role.startswith("transformer")
        }

        for seat in headers:
            for series, found in headers.items():
                caught = STORAGE_REQUIREMENTS[seat] not in found or any(
                    dtype in found for dtype in ANIMA_STORAGE_FORBIDDEN.get(seat, ())
                )
                assert caught is (series != seat), f"{series} → {seat} 席"

    def test_every_series_seat_mix_up_is_refused_by_the_real_gates(self, tmp_path: Path) -> None:
        """上の表ではなく**実 gate**（`assert_storage` / `assert_storage_absent`）で 3×3 を回す。

        上のテストは述語を再実装しているので、`anima_plan` から
        `assert_storage_absent` の呼びが 1 行消えても落ちない。ここは組み立てを実際に通すので、
        呼びが外れた瞬間に非対角が緑になって落ちる。i4 席は配布から降りたので、3 席そろった
        盤面を作るために席を注入する（{@link _i4_seat} — 表は 3 席のまま残っている）。
        """
        # 挿し込むのは**正当な IR コンテナ**（対角は組み立てまで通るので本物が要る）。格納 dtype
        # の集合は系列そのままで、f16 = {F32, F16} / i8 = {F32, I8} / i4 = {F32, I8, I4}。
        seats = [f"transformer_{storage}" for storage in I4_SEAT_STORAGES]
        for seat in seats:
            for series in seats:
                sources = _build_series(tmp_path / f"series-{seat}-{series}", with_i4=True)
                storage = seat.removeprefix("transformer_")
                target = sources.transformer[storage] / "transformer" / "model.safetensors"
                replace_component(
                    target,
                    _weights_container("swapped-series", series.removeprefix("transformer_")),
                )
                out_dir = tmp_path / "models" / f"{seat}-{series}"

                if series == seat:
                    # 対角は通る（同じ系列を同じ席へ）
                    _assemble_anima(sources, out_dir, spec=_i4_seat())
                else:
                    with pytest.raises(DistError):
                        _assemble_anima(sources, out_dir, spec=_i4_seat())

    def test_it_stops_when_a_header_is_not_safetensors(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        replace_component(sources.base / "vae_decoder" / "model.safetensors", b"not-a-safetensors")
        with pytest.raises(DistError, match="ヘッダが読めない"):
            _assemble_anima(sources, tmp_path / "models" / ANIMA_TURBO_MODEL_NAME)


class TestPlanGates:
    """`plans` だけから決まる検査は**最初の 1 バイトを書く前**（dist.py 冒頭の MUST）。"""

    def test_it_refuses_a_missing_input_the_storage_gate_never_looks_at(
        self, tmp_path: Path
    ) -> None:
        """格納 dtype を要求しない役割（tokenizer）の欠落も**配置の前**に落ちる。

        要求表に載る役割は計画段の {@link assert_storage} が実在まで見るので、素通しされる
        役割だけが「配置の途中で落ちる」経路として残っていた。
        """
        sources = _build_series(tmp_path / "series")
        (sources.tokenizers / "qwen2-tokenizer.json").unlink()
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            _assemble_anima(sources, out_dir)
        assert not out_dir.exists()


class TestManifest:
    def test_it_writes_the_envelope_of_adr_0041(self, assembled) -> None:
        out_dir, manifest = assembled
        on_disk = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert manifest["format"] == MANIFEST_FORMAT
        assert manifest["generator"].startswith("karume/")
        assert manifest["defaultModel"] == ANIMA_TURBO_MODEL_NAME
        assert list(manifest["models"]) == [ANIMA_TURBO_MODEL_NAME]
        assert manifest["models"][ANIMA_TURBO_MODEL_NAME]["pipeline"] == "anima/1"

    def test_every_weights_entry_is_keyed_by_dtype(self, assembled) -> None:
        """v1 の `{file}` / `{variants}` の 2 形は消えた — i8 単体も dtype キーを持つ（§3）。

        期待値は**モデルが宣言した格納形**から導く（`anima_weights`）— 語彙の全量
        （{@link ANIMA_WEIGHTS}）をそのまま期待すると、席を降ろしたモデルで落ちる。語彙との
        繋がりは「役割は全量・ラベルは部分集合」で保つ。
        """
        _, manifest = assembled
        weights = manifest["models"][ANIMA_TURBO_MODEL_NAME]["weights"]
        declared = anima_weights(anima_model(ANIMA_TURBO_MODEL_NAME))

        assert set(declared) == set(ANIMA_WEIGHTS)
        assert set(declared["transformer"]) < set(ANIMA_WEIGHTS["transformer"]), "i4 席が復活した"
        assert sorted(weights) == sorted(declared)
        for name, entry in weights.items():
            assert sorted(entry) == sorted(declared[name]), name
            for files in entry.values():
                assert sorted(files) in (["shards"], ["extras", "shards"])

    def test_the_unconditional_files_live_in_assets(self, assembled) -> None:
        _, manifest = assembled
        assets = manifest["models"][ANIMA_TURBO_MODEL_NAME]["assets"]
        assert sorted(assets) == ["tokenizer", "tokenizer_2"]
        for ref in assets.values():
            assert sorted(ref) == ["path", "sha256", "size"]

    def test_it_derives_size_and_sha256_from_the_placed_files(self, assembled) -> None:
        out_dir, manifest = assembled
        shards = manifest["models"][ANIMA_TURBO_MODEL_NAME]["weights"]["text_encoder"]["f16"][
            "shards"
        ]
        # 3 点セットは shard 1 本ずつに掛かる（列のどこかだけ古い、を作れない）。
        for ref, payload in zip(shards, _PAYLOADS["text_encoder"], strict=True):
            assert ref["size"] == len(payload)
            assert ref["sha256"] == hashlib.sha256(payload).hexdigest()
            assert (out_dir / ref["path"]).read_bytes() == payload

    def test_it_carries_the_quant_table_and_pipeline_config(self, assembled) -> None:
        """quant 表は**宣言した格納形で成立する席だけ**（`anima_quants` の導出）。

        席の取捨を quant 名の直書きリストで持つと、格納形を降ろした日に席だけが残る
        （選ぶと 404）— 期待値も同じ導出から引いて、表と現物が 1 箇所から動くようにする。
        """
        _, manifest = assembled
        model = manifest["models"][ANIMA_TURBO_MODEL_NAME]
        declared = anima_quants(anima_model(ANIMA_TURBO_MODEL_NAME))

        assert sorted(model["quants"]) == sorted(declared)
        assert set(declared) < set(ANIMA_QUANTS), "i4 席が quant 表から落ちていない"
        assert model["defaultQuant"] == ANIMA_DEFAULT_QUANT
        assert model["defaultQuant"] in model["quants"]
        assert model["pipelineConfig"] == dict(ANIMA_TURBO_PIPELINE_CONFIG)

    def test_every_quant_maps_every_weights_entry(self, assembled) -> None:
        """hub は写像の完全性を実行時にも検査する — 埋め漏れは配布してから落ちる。"""
        _, manifest = assembled
        model = manifest["models"][ANIMA_TURBO_MODEL_NAME]
        for name, quant in model["quants"].items():
            assert set(quant["weights"]) == set(model["weights"]), name
            for weight, label in quant["weights"].items():
                assert label in model["weights"][weight]


class TestQuantNaming:
    """席名は ADR 0074 の文法（`<格納>[+<部品><ビット>]…[-<ノブ>]…`）と表示欄（ADR 0075）。"""

    def test_every_seat_starts_from_the_shared_base_storage(self) -> None:
        """Anima の基底格納は `f16` — text 経路 3 役が f16 固定なので、圧縮は `+dit…` で綴る。

        `<格納>` を字句どおり全役割共通の基底に取る規則（ADR 0074 決定 1・決定 6）が破れると、
        席名が「transformer の格納だけ」を名乗る旧綴りへ戻る。
        """
        assert all(name.startswith("f16") for name in ANIMA_QUANTS), sorted(ANIMA_QUANTS)

    def test_the_component_override_token_spells_the_transformer(self) -> None:
        """略称は recipe が定める（ADR 0074 決定 4）— 対応はカードにも出る。"""
        assert ANIMA_QUANT_ABBREVIATIONS == {"dit": "transformer"}
        assert set(ANIMA_QUANT_ABBREVIATIONS.values()) <= set(ANIMA_WEIGHTS)

    def test_every_seat_carries_a_label_and_a_description_within_the_limits(self) -> None:
        """表示欄は optional だが、Anima は 8 席とも書く（選択 UI に出す family なので）。"""
        for name, quant in ANIMA_QUANTS.items():
            assert quant["label"] and quant["description"], name
            assert_quant_presentation(f"anima.quants.{name}", quant)

    def test_no_description_repeats_which_seat_is_the_default(self) -> None:
        """既定は `defaultQuant` が指している（ADR 0075 決定 3）— 散文に書くと二重持ちになる。"""
        for name, quant in ANIMA_QUANTS.items():
            assert "default" not in quant["description"].lower(), name


class TestI4Quants:
    """i4 常駐の 2 席（`f16+dit4` = 格納だけ / `f16+dit4-attn8-s16` = **低 VRAM 席**）。

    2026-09-01 の裁定で**どの配布モデルもこの 2 席を持たない**（`storages` から i4 が消え、
    `anima_quants` の導出で席ごと落ちる）。席の定義は quant 語彙（{@link ANIMA_QUANTS}）に
    残っており、`storages` に i4 を足せばそのまま生える復活レバー — ここが見るのは
    「語彙側の定義が壊れていないこと」と「配布へは 1 席も出ないこと」の 2 つ。
    """

    @staticmethod
    def _i4_seats() -> dict[str, Any]:
        return {
            name: quant
            for name, quant in ANIMA_QUANTS.items()
            if quant["weights"].get("transformer") == "i4"
        }

    def test_it_declares_exactly_the_two_i4_seats(self) -> None:
        assert sorted(self._i4_seats()) == ["f16+dit4", "f16+dit4-attn8-s16"]

    def test_the_plain_seat_leaves_the_session_untouched(self) -> None:
        """`f16+dit4` は格納だけを動かす席（計算経路は f32 のまま）。"""
        assert ANIMA_QUANTS["f16+dit4"]["session"] == {}

    def test_the_attention_seat_declares_only_the_attention_knobs(self) -> None:
        assert ANIMA_QUANTS["f16+dit4-attn8-s16"]["session"] == {
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        }

    def test_no_i4_seat_declares_linear_compute(self) -> None:
        """MUST: i4 席に `linearCompute` を宣言しない。**この不変条件の理由は 2026-08-21 に
        入れ替わっている** — 旧: 「`a8` の述語が i8 常駐を要求するので宣言しても効かない」/
        新: w4a8（ADR 0076）で効くようになったが、**掛けると画の細部が荒れる**という視認裁定
        （research 2026-08-21 §6）。速度は戻る（1,640 → 955 ms/step）が、この席は
        サイズ・VRAM のための席で、速度が要るなら既定の `f16+dit8-a8-attn8-s16` が上。
        """
        for name, quant in self._i4_seats().items():
            assert "linearCompute" not in quant["session"], name

    def test_the_default_quant_stays_on_the_i8_seat(self) -> None:
        """席が増えても既定は動かさない（既定の変更は品質裁定を要する別の判断）。"""
        assert ANIMA_DEFAULT_QUANT == "f16+dit8-a8-attn8-s16"

    def test_no_shipped_model_carries_an_i4_seat(self, assembled) -> None:
        """MUST: 配布形の quant 表に i4 席が 1 つも出ない（2026-09-01 裁定）。

        「席は消えたがファイルだけ配られる（無駄な GB）」と「ファイルは消えたが席が残る
        （選ぶと 404）」は別の壊れ方なので、quant 表と現物の**両方**で見る。
        """
        _, manifest = assembled
        quants = manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]

        assert set(quants) & set(self._i4_seats()) == set()
        for entry in quants.values():
            assert entry["weights"]["transformer"] != "i4"
        for spec in ANIMA_MODELS.values():
            assert set(anima_quants(spec)) & set(self._i4_seats()) == set()

    def test_the_seats_reach_the_manifest_with_their_session_knobs(self, assembled_with_i4) -> None:
        """席が戻れば表に足しただけで配布形へ出ること（quant 表は manifest 由来 — §3）。

        導出（`anima_quants`）が壊れると復活レバーが引けなくなるので、席を注入した spec で
        「宣言 → quant 表 → manifest」の 1 本を通しておく。
        """
        _, manifest = assembled_with_i4
        quants = manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]

        assert set(self._i4_seats()) <= set(quants)
        for name, quant in self._i4_seats().items():
            assert quants[name]["weights"]["transformer"] == "i4"
            assert quants[name]["session"] == dict(quant["session"])


class TestVerifyDist:
    def test_it_passes_on_a_freshly_assembled_tree(self, assembled) -> None:
        out_dir, _ = assembled
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))

    def test_it_catches_a_file_that_no_longer_matches_its_declared_size(self, assembled) -> None:
        out_dir, _ = assembled
        # 分割された役割は shard 1 本の破損で落ちる（列の中の 1 本も突合の対象）。
        target = resolve_shards(out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["vae_decoder"])[1]
        target.unlink()  # ハードリンクを外してから書く（源の系列を壊さない）
        target.write_bytes(b"shorter")
        with pytest.raises(DistError, match="size が manifest と違う"):
            verify_dist(out_dir)

    def test_it_catches_a_missing_file(self, assembled) -> None:
        out_dir, _ = assembled
        (out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["tokenizer"]).unlink()
        with pytest.raises(DistError, match="参照するファイルが無い"):
            verify_dist(out_dir)

    def test_it_catches_an_undeclared_file(self, assembled) -> None:
        out_dir, _ = assembled
        (
            out_dir / ANIMA_TURBO_MODEL_NAME / "transformer" / "io.s01024t0699.safetensors"
        ).write_bytes(b"stale")
        with pytest.raises(DistError, match="宣言していないファイル"):
            verify_dist(out_dir)

    def test_it_admits_the_model_card_as_a_meta_file(self, assembled) -> None:
        """`README.md` は karume.json と同格のメタファイル（前回の組み立ての残りでも通す）。"""
        out_dir, _ = assembled
        (out_dir / MODEL_CARD_FILENAME).write_text("前回のモデルカード", encoding="utf-8")
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))

    def test_it_still_refuses_a_meta_name_in_a_subdirectory(self, assembled) -> None:
        """例外は直下の 2 つだけ — 下位ディレクトリの同名は宣言外のまま。"""
        out_dir, _ = assembled
        (out_dir / ANIMA_TURBO_MODEL_NAME / MODEL_CARD_FILENAME).write_text(
            "紛れ込み", encoding="utf-8"
        )
        with pytest.raises(DistError, match="宣言していないファイル"):
            verify_dist(out_dir)


def _rewrite(out_dir: Path, mutate) -> None:
    """`karume.json` を書き換える（verify_dist の反例を作るための唯一の手段）。"""
    path = out_dir / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class TestVerifyDistStructure:
    """v2 manifest が**自分で閉じている**ことの検査（hub が受理する形かを焼いた側でも見る）。"""

    def test_it_refuses_a_manifest_that_is_not_karume_2(self, assembled) -> None:
        out_dir, _ = assembled
        _rewrite(out_dir, lambda m: m.update(format="karume/1"))
        with pytest.raises(DistError, match=MANIFEST_FORMAT):
            verify_dist(out_dir)

    def test_it_refuses_a_default_model_that_is_not_declared(self, assembled) -> None:
        out_dir, _ = assembled
        _rewrite(out_dir, lambda m: m.update(defaultModel="anima-lite"))
        with pytest.raises(DistError, match="defaultModel 'anima-lite'"):
            verify_dist(out_dir)

    def test_it_refuses_a_default_quant_that_is_not_declared(self, assembled) -> None:
        out_dir, _ = assembled

        def drop_default(manifest: dict) -> None:
            manifest["models"][ANIMA_TURBO_MODEL_NAME]["defaultQuant"] = "nonexistent"

        _rewrite(out_dir, drop_default)
        with pytest.raises(DistError, match="defaultQuant 'nonexistent'"):
            verify_dist(out_dir)

    def test_it_refuses_a_quant_that_leaves_a_weights_entry_unmapped(self, assembled) -> None:
        out_dir, _ = assembled

        def unmap(manifest: dict) -> None:
            del manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]["f16"]["weights"][
                "vae_decoder"
            ]

        _rewrite(out_dir, unmap)
        with pytest.raises(DistError, match="完全写像でない"):
            verify_dist(out_dir)

    def test_it_refuses_a_quant_that_names_a_dtype_the_weights_lack(self, assembled) -> None:
        out_dir, _ = assembled

        def retype(manifest: dict) -> None:
            manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]["f16"]["weights"][
                "vae_decoder"
            ] = "i8"

        _rewrite(out_dir, retype)
        with pytest.raises(DistError, match="dtype 'i8' が無い"):
            verify_dist(out_dir)

    def test_it_refuses_a_path_outside_the_model_subtree_and_shared(self, assembled) -> None:
        """レイアウトは一律「モデル別サブツリー + shared/」（ADR 0041 §9）。"""
        out_dir, _ = assembled

        def flatten(manifest: dict) -> None:
            ref = manifest["models"][ANIMA_TURBO_MODEL_NAME]["assets"]["tokenizer"]
            ref["path"] = OUTPUT_PATHS["tokenizer"]

        _rewrite(out_dir, flatten)
        with pytest.raises(DistError, match="レイアウトはモデル別サブツリー"):
            verify_dist(out_dir)

    def test_it_refuses_two_references_to_one_path_that_disagree(self, assembled) -> None:
        """同一 path の共有は合法だが、{size, sha256} の食い違いは取得層を振動させる。"""
        out_dir, _ = assembled

        def bend(manifest: dict) -> None:
            entry = manifest["models"][ANIMA_TURBO_MODEL_NAME]["weights"]["transformer"]
            entry["i8"]["extras"]["rope_base"]["size"] += 1

        _rewrite(out_dir, bend)
        with pytest.raises(DistError, match="食い違う"):
            verify_dist(out_dir)


class TestFamilyAssembly:
    """1 リポに複数モデル（ADR 0041 §2）+ 共有ファイルは `shared/` に 1 回だけ（§5）。

    並べるのは**実際に同居する組**（公式リポの素版 + Aesthetic）— 受理集合が別リポへ分けた
    組み合わせをフィクスチャで組むと、テストと {@link TestPipelineMembership} が別のことを言う。
    """

    @pytest.fixture
    def family(self, tmp_path: Path) -> tuple[Path, dict]:
        series = tmp_path / "series"
        # base（text 経路 / VAE / tokenizer）は共通、transformer だけモデルごとに違う中身。
        first = _build_series(series, model=FAMILY_MODELS[0], mark=b"-base")
        second = _build_series(series, model=FAMILY_MODELS[1], mark=b"-aesthetic")
        out_dir = tmp_path / "models" / "anima-family"
        manifest = assemble_family(
            [anima_plan(first, FAMILY_MODELS[0]), anima_plan(second, FAMILY_MODELS[1])],
            out_dir,
            FAMILY_MODELS[0],
        )
        return out_dir, manifest

    def test_it_declares_every_model_with_the_first_as_default(self, family) -> None:
        _, manifest = family
        assert list(manifest["models"]) == list(FAMILY_MODELS)
        assert manifest["defaultModel"] == FAMILY_MODELS[0]

    def test_it_places_a_byte_identical_file_once_under_shared(self, family) -> None:
        out_dir, manifest = family
        # 畳まれるのは**コンポーネント丸ごと**（shard 列がそのまま shared/ の下へ移る）。
        shared = [f"{SHARED_DIRNAME}/{rel}" for rel in shard_paths(OUTPUT_PATHS["text_encoder"])]
        for name in manifest["models"]:
            refs = manifest["models"][name]["weights"]["text_encoder"]["f16"]["shards"]
            assert [ref["path"] for ref in refs] == shared
        for rel, payload in zip(shared, _PAYLOADS["text_encoder"], strict=True):
            assert (out_dir / rel).read_bytes() == payload
        # 各モデルのサブツリーには残らない（1 回だけ置く = 重複を配らない）。
        for name in manifest["models"]:
            for rel in shard_paths(OUTPUT_PATHS["text_encoder"]):
                assert not (out_dir / name / rel).exists()

    def test_it_keeps_the_files_that_differ_inside_each_model_subtree(self, family) -> None:
        out_dir, manifest = family
        paths = {
            name: [ref["path"] for ref in entry["weights"]["transformer"]["i8"]["shards"]]
            for name, entry in manifest["models"].items()
        }
        assert paths == {
            name: [f"{name}/{rel}" for rel in shard_paths(OUTPUT_PATHS["transformer_i8"])]
            for name in FAMILY_MODELS
        }
        assert [(out_dir / rel).read_bytes() for rel in paths[FAMILY_MODELS[0]]] != [
            (out_dir / rel).read_bytes() for rel in paths[FAMILY_MODELS[1]]
        ]

    def test_the_shared_and_the_private_files_together_cover_the_tree(self, family) -> None:
        out_dir, _ = family
        # rope 素表は幾何だけで決まるのでモデル間でもバイト同一 — 付帯資産（extras）も
        # 同じ規則で畳まれる。
        shared_roles = (
            "text_encoder",
            "text_conditioner",
            "vae_decoder",
            "tokenizer",
            "tokenizer_2",
            "rope_base",
        )
        # weights の席は shard 連番へ展開される（ADR 0081）— 畳む単位はコンポーネント丸ごと。
        expected = [
            f"{SHARED_DIRNAME}/{rel}"
            for rel in placed_paths(
                {role: OUTPUT_PATHS[role] for role in shared_roles}, ANIMA_WEIGHTS
            )
        ]
        expected += [
            f"{model}/{rel}"
            for model in FAMILY_MODELS
            for storage in anima_model(model).storages
            for rel in shard_paths(OUTPUT_PATHS[f"transformer_{storage}"])
        ]
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_a_shared_extra_is_referenced_from_every_dtype_of_every_model(self, family) -> None:
        """付帯資産（extras）も同じ規則で畳まれる — 参照側は 4 箇所とも同じ path を書く。"""
        _, manifest = family
        paths = {
            entry["extras"]["rope_base"]["path"]
            for model in manifest["models"].values()
            for entry in model["weights"]["transformer"].values()
        }
        assert paths == {f"{SHARED_DIRNAME}/{OUTPUT_PATHS['rope_base']}"}

    def test_it_leaves_no_empty_directory_behind_after_folding(self, family) -> None:
        out_dir, _ = family
        empty = [
            str(path.relative_to(out_dir))
            for path in out_dir.rglob("*")
            if path.is_dir() and not any(path.iterdir())
        ]
        assert empty == []

    def test_the_assembled_family_verifies(self, family) -> None:
        out_dir, manifest = family
        declared = verify_dist(out_dir)
        # 共有ぶんは 1 本に畳まれるので、宣言 path は「モデル数 × 役割数」より少ない。
        assert len(declared) < len(OUTPUT_PATHS) * len(manifest["models"])

    def test_it_reassembles_a_family_over_a_previous_run(self, tmp_path: Path) -> None:
        """畳んだ後の木をもう一度組んでも落ちない（`shared/` の既存ファイルを踏み直す経路）。"""
        series = tmp_path / "series"
        first = _build_series(series, model=FAMILY_MODELS[0], mark=b"-base")
        second = _build_series(series, model=FAMILY_MODELS[1], mark=b"-aesthetic")
        out_dir = tmp_path / "models" / "anima-family"
        plans = [anima_plan(first, FAMILY_MODELS[0]), anima_plan(second, FAMILY_MODELS[1])]
        before = assemble_family(plans, out_dir, FAMILY_MODELS[0])
        after = assemble_family(plans, out_dir, FAMILY_MODELS[0])
        assert before == after
        assert verify_dist(out_dir)

    def test_it_refuses_a_duplicated_model_name(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        plan = anima_plan(sources, ANIMA_TURBO_MODEL_NAME)
        with pytest.raises(DistError, match="モデル名が重複"):
            assemble_family([plan, plan], tmp_path / "out", ANIMA_TURBO_MODEL_NAME)

    def test_it_refuses_a_default_model_it_is_not_assembling(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        with pytest.raises(DistError, match="既定モデル 'anima-lite'"):
            assemble_family(
                [anima_plan(sources, ANIMA_TURBO_MODEL_NAME)], tmp_path / "out", "anima-lite"
            )


class TestAtomicReplacement:
    """組み立ては staging へ作って rename で据える — 既存の配布形に中途の形を一度も晒さない。"""

    def _snapshot(self, out_dir: Path) -> dict[str, bytes]:
        return {rel_path: (out_dir / rel_path).read_bytes() for rel_path in _present(out_dir)}

    def _siblings(self, out_dir: Path) -> list[str]:
        """出力先の隣に残った作業ディレクトリ（staging / 退避先）— 成功でも失敗でも空。"""
        return sorted(path.name for path in out_dir.parent.iterdir() if path.name != out_dir.name)

    def _fail_at(self, monkeypatch: pytest.MonkeyPatch, nth: int) -> None:
        """`nth` 回目の配置だけを I/O 故障にする（数 GB の途中で落ちる形の注入）。"""
        calls = 0

        def failing(artifact: Artifact, dest: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == nth:
                raise OSError("配置の途中で落ちた")
            materialize(artifact, dest)

        monkeypatch.setattr("karume.dist.materialize", failing)

    def test_a_failure_midway_leaves_no_distribution_on_a_fresh_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        self._fail_at(monkeypatch, nth=3)
        with pytest.raises(OSError, match="配置の途中で落ちた"):
            _assemble_anima(sources, out_dir)

        assert not out_dir.exists()
        assert list(out_dir.parent.iterdir()) == []

    def test_it_writes_the_card_into_the_tree_it_swaps_in(self, tmp_path: Path) -> None:
        """`render_card` は組み立て済みの manifest を受け取り、据わる木の中に書かれる。"""
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        assemble_family(
            [anima_plan(sources, ANIMA_TURBO_MODEL_NAME)],
            out_dir,
            ANIMA_TURBO_MODEL_NAME,
            render_card=lambda manifest: f"{manifest['defaultModel']}\n",
        )
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert card == f"{ANIMA_TURBO_MODEL_NAME}\n"
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))
        assert self._siblings(out_dir) == []

    def test_reassembling_a_subset_replaces_the_whole_repository(self, tmp_path: Path) -> None:
        """再組み立ては plan に無いモデルごと丸ごと置き換える（前回の残骸が生き残らない）。"""
        series = tmp_path / "series"
        first = _build_series(series, model=FAMILY_MODELS[0], mark=b"-base")
        second = _build_series(series, model=FAMILY_MODELS[1], mark=b"-aesthetic")
        out_dir = tmp_path / "models" / "anima-family"
        assemble_family(
            [anima_plan(first, FAMILY_MODELS[0]), anima_plan(second, FAMILY_MODELS[1])],
            out_dir,
            FAMILY_MODELS[0],
        )

        manifest = assemble_family([anima_plan(first, FAMILY_MODELS[0])], out_dir, FAMILY_MODELS[0])

        assert list(manifest["models"]) == [FAMILY_MODELS[0]]
        assert not (out_dir / FAMILY_MODELS[1]).exists()
        # 1 モデルだけなら畳む相手が居ない = `shared/` も残らない（ADR 0041 §5）。
        assert not (out_dir / SHARED_DIRNAME).exists()
        assert _present(out_dir) == sorted([*_in_subtree(FAMILY_MODELS[0]), MANIFEST_FILENAME])
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(FAMILY_MODELS[0]))

    def test_it_discards_a_working_directory_left_by_an_interrupted_run(
        self, tmp_path: Path
    ) -> None:
        """中断が残した staging は踏み直さずに捨てる（plan に無いファイルを混ぜない）。"""
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        leftover = out_dir.with_name(out_dir.name + STAGING_SUFFIX)
        _write(
            leftover / "anima-wai-v1.0" / "transformer" / "model.f16.safetensors", b"interrupted"
        )

        _assemble_anima(sources, out_dir)

        assert _present(out_dir) == sorted(
            [*_in_subtree(ANIMA_TURBO_MODEL_NAME), MANIFEST_FILENAME]
        )
        assert self._siblings(out_dir) == []

    def test_a_successful_reassembly_lands_the_same_tree_without_leftovers(
        self, tmp_path: Path
    ) -> None:
        """成功経路は据え替えても同値 — manifest も現物も 1 回目と同じで、作業跡も残らない。"""
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        first = _assemble_anima(sources, out_dir)
        snapshot = self._snapshot(out_dir)

        again = _assemble_anima(sources, out_dir)

        assert again == first
        assert self._snapshot(out_dir) == snapshot
        assert self._siblings(out_dir) == []


class TestModelCard:
    """`karume dist` は組み立て + 検証の**後**にモデルカードを書く。"""

    def _run(self, tmp_path: Path, *argv: str) -> Path:
        _build_series(tmp_path / "series")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "anima",
                "--series",
                str(tmp_path / "series"),
                "--out",
                str(out_dir),
                *argv,
            ]
        )
        return out_dir

    def test_it_writes_a_model_card_next_to_the_manifest(self, tmp_path: Path) -> None:
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert card.startswith("---\n")
        assert "base_model: circlestone-labs/Anima-Base-v1.0-Diffusers" in card

    def test_it_derives_the_file_table_from_the_assembled_tree(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        for rel_path in _in_subtree(ANIMA_TURBO_MODEL_NAME):
            assert f"`{rel_path}`" in card
        # 表が引くのは shard 1 本ずつのサイズ（分割された役割は行が本数ぶん並ぶ）。
        for shard in resolve_shards(
            out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["transformer_i8"]
        ):
            assert f"{shard.stat().st_size:,} B" in card

    def test_it_names_the_repository_after_the_assembled_directory(self, tmp_path: Path) -> None:
        """ファミリーリポの ID は pipeline の定数にできない — 組み立て先から引く。"""
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert '  repo: "hdae/dist",' in card

    def test_it_leaves_the_tree_verifiable_after_writing_the_card(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))

    def test_it_reassembles_over_a_previous_card(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        first = (out_dir / MODEL_CARD_FILENAME).read_bytes()
        main(
            [
                "--pipeline",
                "anima",
                "--series",
                str(tmp_path / "series"),
                "--out",
                str(out_dir),
            ]
        )
        assert (out_dir / MODEL_CARD_FILENAME).read_bytes() == first


class TestLegalTexts:
    """上流ライセンス（CircleStone Non-Commercial License）§3 の再配布条件を配布リポで満たす。

    条件は「ライセンス文のコピーを提供する」(a)・「Attribution Notice を目立つように掲示する」
    (b)・「改変した旨を Notice に含める」(d)(i)・「公式製品と誤認させない」(d)(iii) の 4 つで、
    どれも**配布リポ 1 つ**に掛かる（モデルの資産ではない）ので直下の 2 ファイルが受け持つ。
    """

    #: 上流から取得した原文の sha256（2026-08-20 実測）。ここを固定するのは、整形や改行変換で
    #: 1 バイトでも動けば「このライセンスのコピー」でなくなるため（§3(a)）。
    LICENSE_SHA256 = "ee956174133d7c2cdcf220440c7726187eaf4b50e8e48ee32194353a22164d15"

    def _run(
        self, tmp_path: Path, *, pipeline: str = "anima", model: str = ANIMA_TURBO_MODEL_NAME
    ) -> Path:
        """`dist.py` の CLI を 1 周させる（既定は公式リポ = `--pipeline anima`）。

        リポは 2 つに分かれ、直下の `NOTICE.md` は Pipeline ごとに違う 1 組なので、
        法的テキストの検査は**どちらのリポで組んだか**まで含めて回す。
        """
        _build_series(tmp_path / "series", model=model)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                pipeline,
                "--model",
                model,
                "--series",
                str(tmp_path / "series"),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    @staticmethod
    def _flat(text: str) -> str:
        """改行の折り位置に依存せず**文**を見るための平坦化（法的文言は文が単位）。"""
        return " ".join(text.split())

    def test_the_recipe_carries_the_license_verbatim(self) -> None:
        source = LICENSE_SOURCE_PATH.read_bytes()
        assert hashlib.sha256(source).hexdigest() == self.LICENSE_SHA256

    def test_it_ships_the_license_text_byte_identical(self, tmp_path: Path) -> None:
        """§3(a) — 提供するのは**このライセンスのコピー**（要約でも書き換えでもない）。"""
        out_dir = self._run(tmp_path)
        assert (out_dir / "LICENSE.md").read_bytes() == LICENSE_SOURCE_PATH.read_bytes()

    def test_the_notice_displays_the_attribution_verbatim(self, tmp_path: Path) -> None:
        """§3(b) — 掲示する文言は逐語（2 文とも）。"""
        notice = (self._run(tmp_path) / "NOTICE.md").read_text(encoding="utf-8")
        assert notice == OFFICIAL_NOTICE_MARKDOWN
        for sentence in ATTRIBUTION_NOTICE.split("\n"):
            assert sentence in notice

    def test_the_extra_repository_ships_its_own_notice(self, tmp_path: Path) -> None:
        """追加学習リポの直下には**そのリポの**改変告知が入る（取り違えると告知が嘘になる）。"""
        out_dir = self._run(tmp_path, pipeline="anima-extra", model="anima-wai-v1.0")

        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")
        assert notice == EXTRA_NOTICE_MARKDOWN
        assert (out_dir / "LICENSE.md").read_bytes() == LICENSE_SOURCE_PATH.read_bytes()
        assert ATTRIBUTION_NOTICE in notice

    def test_the_notice_states_the_modifications_inside_the_attribution_notice(
        self, tmp_path: Path
    ) -> None:
        """§3(d)(i) — 改変した旨を **Attribution Notice の中に**含める。"""
        text = (self._run(tmp_path) / "NOTICE.md").read_text(encoding="utf-8")
        # 改変記載は独立節ではなく Attribution Notice 節の内側（次見出しの前）に居ること。
        notice_section = text.split("## Not an official product")[0]
        assert "## Attribution Notice" in notice_section
        flat = self._flat(notice_section)
        assert flat.count("this Attribution Notice also states that") == 1
        assert flat.count("modified as follows:") == 1
        # 実際にした改変（コンテナ形式への変換と量子化系列の追加）が列挙されていること。
        assert "converted into the container format" in flat
        assert "split across numbered shards when a component is too large for one file" in flat
        assert "An int8-quantized series of the transformer was added alongside the f16 one" in flat

    def test_the_notice_disclaims_any_official_standing(self, tmp_path: Path) -> None:
        """§3(d)(iii) — 公式製品・承認済みと誤認させない。"""
        text = (self._run(tmp_path) / "NOTICE.md").read_text(encoding="utf-8")
        assert (
            "This is not an official product of CircleStone Labs LLC, and it is not endorsed,"
            " approved or validated by CircleStone Labs LLC." in self._flat(text)
        )
        assert text.endswith(
            "The full license text is distributed alongside this repository as LICENSE.md.\n"
        )

    def test_the_assembled_card_displays_the_attribution_too(self, tmp_path: Path) -> None:
        """§3(b) の掲示は**据わった配布形**で成立する必要がある（節の中身は test_card が見る）。"""
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert ATTRIBUTION_NOTICE in card

    def test_the_tree_still_verifies_with_the_legal_texts_in_place(self, tmp_path: Path) -> None:
        """直下の 2 つは manifest が宣言しない — 宣言外ファイル検査の例外側に居る。"""
        out_dir = self._run(tmp_path)
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))
        assert (out_dir / "LICENSE.md").is_file()
        assert (out_dir / "NOTICE.md").is_file()

    def test_both_pipelines_declare_exactly_the_two_legal_seats(self) -> None:
        for pipeline in (OFFICIAL_PIPELINE, EXTRA_PIPELINE):
            assert sorted(pipeline.root_files) == ["LICENSE.md", "NOTICE.md"]

    def test_both_pipelines_ship_the_same_license_text(self) -> None:
        """上流ライセンスは 1 本しか掛からない — リポで違うのは改変告知だけ（§3(a) は同文）。"""
        assert OFFICIAL_PIPELINE.root_files["LICENSE.md"] == EXTRA_PIPELINE.root_files["LICENSE.md"]


class TestPipelineEntry:
    """ドライバへ差す 2 行（{@link OFFICIAL_PIPELINE} / {@link EXTRA_PIPELINE}）。

    名前・リポ名・帰属の 3 点を、リポごとに独立に見る（1 つに畳めない理由は
    {@link TestPipelineMembership}）。
    """

    def test_the_model_name_is_a_legal_path_segment(self) -> None:
        """モデル名は manifest のキーであると同時にリポ内のディレクトリ名（ADR 0041 §6 / §9）。"""
        for model in (*OFFICIAL_MODELS, *EXTRA_MODELS):
            assert assert_model_name(model) == model

    def test_it_names_the_default_model_and_its_repository(self) -> None:
        assert OFFICIAL_PIPELINE.default_model == ANIMA_TURBO_MODEL_NAME
        assert EXTRA_PIPELINE.default_model == EXTRA_MODELS[0]
        for pipeline in (OFFICIAL_PIPELINE, EXTRA_PIPELINE):
            model = pipeline.default_model
            assert pipeline.repo_name(model) == f"karume-{model}"

    def test_one_attribution_needs_no_choice(self) -> None:
        """帰属は 1 リポにつき 1 通りしかない（選びようがないものを聞かない）。"""
        for pipeline, profile in ((OFFICIAL_PIPELINE, "anima"), (EXTRA_PIPELINE, "anima-extra")):
            assert len(pipeline.card_profiles) == 1
            assert resolve_card_renderer(pipeline, None) is pipeline.card_profiles[profile]

    def test_the_two_repositories_render_different_cards(self) -> None:
        """公式 / 追加学習で題も出所節の導入も違う — 描き手を取り違えると帰属が入れ替わる。"""
        assert (
            OFFICIAL_PIPELINE.card_profiles["anima"]
            is not EXTRA_PIPELINE.card_profiles["anima-extra"]
        )

    def test_their_cards_refuse_another_pipelines_manifest(self) -> None:
        manifest = {"models": {"m": {"pipeline": "anima/0"}}}
        for pipeline in (OFFICIAL_PIPELINE, EXTRA_PIPELINE):
            for render_card in pipeline.card_profiles.values():
                with pytest.raises(ValueError):
                    render_card(manifest, "hdae/x")
