"""配布ディレクトリ組み立て（`karume.dist`）の単体テスト。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合だけ**（格納形の門）
なので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。SBV2 は加えて
`config.json` と `style_vectors.npy` を読むが、どちらも数行 / 数 KB の合成物で足りる
（**合成 config の style2id は実重み FN4 と別の並び**にしてある — 表を焼き込んでいれば落ちる）。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from karume.dist import (
    ANIMA_DEFAULT_QUANT,
    ANIMA_MODEL_NAME,
    ANIMA_PIPELINE_CONFIG,
    ANIMA_QUANTS,
    ANIMA_WEIGHTS,
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MODEL_CARD_FILENAME,
    OUTPUT_PATHS,
    PIPELINES,
    SBV2_DEFAULT_MODEL,
    SBV2_DEFAULT_QUANT,
    SBV2_KNOB_KEYS,
    SBV2_OUTPUT_PATHS,
    SBV2_QUANTS,
    SBV2_SPEAKER_KEY,
    SBV2_SPEAKER_TENSOR,
    SBV2_STYLE_KEY,
    SBV2_WEIGHTS,
    SHARED_DIRNAME,
    AnimaSources,
    DistError,
    Sbv2Sources,
    WeightFiles,
    anima_plan,
    anima_sources,
    assemble_family,
    assert_manifest_limits,
    assert_model_name,
    build_parser,
    complete_quant_weights,
    default_out_dir,
    main,
    resolve_card_renderer,
    sbv2_knob_defaults,
    sbv2_pipeline_config,
    sbv2_plan,
    sbv2_repo_name,
    sbv2_series_name,
    sbv2_sources,
    sbv2_speaker_embeddings,
    sbv2_style_vectors,
    verify_dist,
)
from karume.ir import IR_METADATA_KEY


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


#: 偽資産の中身（役割ごとに違うバイト列 — 取り違えがハッシュで見える）。モデル 5 役は
#: `STORAGE_REQUIREMENTS` が要求する dtype をヘッダに持つ。rope_base はヘッダ検査の対象外
#: だが、実物同様に正規形にしておく。
_PAYLOADS = {
    "text_encoder": _fake_safetensors("F16", b"text-encoder-weights"),
    "text_conditioner": _fake_safetensors("F16", b"text-conditioner-weights"),
    "transformer_f16": _fake_safetensors("F16", b"transformer-f16-weights"),
    "transformer_i8": _fake_safetensors("I8", b"transformer-i8-weights"),
    "rope_base": _fake_safetensors("F32", b"rope-base-table"),
    "vae_decoder": _fake_safetensors("F16", b"vae-decoder-weights"),
    "tokenizer": b'{"qwen2": true}',
    "tokenizer_2": b'{"t5": true}',
}


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _build_series(
    series_dir: Path,
    *,
    model: str = ANIMA_MODEL_NAME,
    i8_rope: bytes | None = None,
    mark: bytes = b"",
) -> AnimaSources:
    """系列レイアウト（`outputs/series/` 相当）を偽資産で再現する（`io.*` の混入込み）。

    `mark` は transformer 系列だけに混ぜる差分 — ファミリー組み立てで「モデルごとに違う重み」と
    「モデル間で同一の base 資産」を作り分けるための軸。
    """
    sources = anima_sources(series_dir, model)
    _write(sources.base / "text_encoder" / "model.safetensors", _PAYLOADS["text_encoder"])
    _write(sources.base / "text_conditioner" / "model.safetensors", _PAYLOADS["text_conditioner"])
    _write(sources.base / "vae_decoder" / "model.safetensors", _PAYLOADS["vae_decoder"])
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.base / "text_encoder" / "io.t005.safetensors", b"io-fixture")
    _write(sources.base / "vae_decoder" / "io.case0.safetensors", b"io-fixture")
    for series, role, dtype in (
        (sources.transformer_f16, "transformer_f16", "F16"),
        (sources.transformer_i8, "transformer_i8", "I8"),
    ):
        payload = (
            _PAYLOADS[role] if not mark else _fake_safetensors(dtype, role.encode("utf-8") + mark)
        )
        _write(series / "transformer" / "model.safetensors", payload)
        _write(series / "transformer" / "io.s01024t0699.safetensors", b"io-fixture")
    _write(
        sources.transformer_f16 / "transformer" / "rope_base.safetensors", _PAYLOADS["rope_base"]
    )
    _write(
        sources.transformer_i8 / "transformer" / "rope_base.safetensors",
        _PAYLOADS["rope_base"] if i8_rope is None else i8_rope,
    )
    _write(sources.tokenizers / "qwen2-tokenizer.json", _PAYLOADS["tokenizer"])
    _write(sources.tokenizers / "t5-tokenizer.json", _PAYLOADS["tokenizer_2"])
    return sources


def _assemble_anima(
    sources: AnimaSources, out_dir: Path, model: str = ANIMA_MODEL_NAME
) -> dict[str, Any]:
    """単一モデルの組み立て（計画 → 実体化）を 1 行で回すテスト用の糊。"""
    return assemble_family([anima_plan(sources, model)], out_dir, model)


def _in_subtree(model: str, paths: Iterable[str] | None = None) -> list[str]:
    """モデルサブツリー内の期待 path（ADR 0041 §9 の一様レイアウト）。"""
    return [f"{model}/{rel}" for rel in (OUTPUT_PATHS.values() if paths is None else paths)]


def _ref(path: str) -> dict[str, Any]:
    """3 点セットの偽値（規模上限の検査は件数しか見ないので中身は形だけで足りる）。"""
    return {"path": path, "size": 1, "sha256": "0" * 64}


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_series(tmp_path / "series")
    out_dir = tmp_path / "models" / "anima-turbo"
    manifest = _assemble_anima(sources, out_dir)
    return out_dir, manifest


class TestLayout:
    def test_it_places_every_declared_path_under_the_model_subtree(self, assembled) -> None:
        out_dir, _ = assembled
        assert _present(out_dir) == sorted([*_in_subtree(ANIMA_MODEL_NAME), MANIFEST_FILENAME])

    def test_it_never_carries_io_fixtures_into_the_distribution(self, assembled) -> None:
        out_dir, _ = assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_renames_the_two_transformer_series_into_dtype_files(self, assembled) -> None:
        out_dir, _ = assembled
        subtree = out_dir / ANIMA_MODEL_NAME / "transformer"
        assert (subtree / "model.f16.safetensors").read_bytes() == _PAYLOADS["transformer_f16"]
        assert (subtree / "model.i8.safetensors").read_bytes() == _PAYLOADS["transformer_i8"]

    def test_a_single_model_repository_has_no_shared_directory(self, assembled) -> None:
        """`shared/` は 2 モデル以上が同じ中身を持ったときだけ現れる席（ADR 0041 §5）。"""
        out_dir, _ = assembled
        assert not (out_dir / SHARED_DIRNAME).exists()

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / "anima-turbo"
        _assemble_anima(sources, out_dir)
        _assemble_anima(sources, out_dir)  # 既存リンクがあっても落ちない
        assert verify_dist(out_dir)

    def test_the_model_name_moves_the_whole_subtree(self, tmp_path: Path) -> None:
        """`--model` はサブツリー名・系列名・manifest のキーを 1 語で動かす。"""
        sources = _build_series(tmp_path / "series", model="anima-lite")
        out_dir = tmp_path / "models" / "anima-lite"
        manifest = _assemble_anima(sources, out_dir, "anima-lite")
        assert list(manifest["models"]) == ["anima-lite"]
        assert manifest["defaultModel"] == "anima-lite"
        assert _present(out_dir) == sorted([*_in_subtree("anima-lite"), MANIFEST_FILENAME])


class TestPlacementStrategy:
    def test_it_places_independent_copies(self, assembled) -> None:
        """配布形はハードリンクを持たない（系列から独立した自己完結スナップショット）。"""
        out_dir, _ = assembled
        placed = out_dir / ANIMA_MODEL_NAME / OUTPUT_PATHS["text_encoder"]
        assert placed.read_bytes() == _PAYLOADS["text_encoder"]
        assert placed.stat().st_nlink == 1

    def test_a_series_rewrite_does_not_reach_the_dist(self, tmp_path: Path) -> None:
        """系列の再 export（truncate 上書き）が組み立て済み配布形へ波及しないこと。

        リンク方式ではここが破れていた — 同じ inode を共有するため、系列の書き直しが
        manifest の sha256 と現物を黙って食い違わせる。
        """
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / "anima-turbo"
        _assemble_anima(sources, out_dir)
        source = sources.base / "text_encoder" / "model.safetensors"
        with source.open("wb") as handle:
            handle.write(_fake_safetensors("F16", b"rewritten-after-assembly"))
        placed = out_dir / ANIMA_MODEL_NAME / OUTPUT_PATHS["text_encoder"]
        assert placed.read_bytes() == _PAYLOADS["text_encoder"]

    def test_it_stops_when_an_input_is_missing(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        (sources.base / "vae_decoder" / "model.safetensors").unlink()
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            _assemble_anima(sources, tmp_path / "models" / "anima-turbo")


class TestRopeBase:
    def test_it_collapses_the_two_series_into_one_file(self, assembled) -> None:
        out_dir, manifest = assembled
        entry = manifest["models"][ANIMA_MODEL_NAME]["weights"]["transformer"]
        f16_extra = entry["f16"]["extras"]["rope_base"]
        i8_extra = entry["i8"]["extras"]["rope_base"]
        assert f16_extra == i8_extra
        assert f16_extra["path"] == f"{ANIMA_MODEL_NAME}/{OUTPUT_PATHS['rope_base']}"
        assert (out_dir / f16_extra["path"]).read_bytes() == _PAYLOADS["rope_base"]

    def test_it_refuses_to_pick_a_side_when_the_series_disagree(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series", i8_rope=b"rope-base-table-DIFFERENT")
        out_dir = tmp_path / "models" / "anima-turbo"
        with pytest.raises(DistError, match="バイト同一でない"):
            _assemble_anima(sources, out_dir)
        # 止めた以上、途中の配布形を残さない（片方だけ入った出力を後段に見せない）。
        assert not (out_dir / MANIFEST_FILENAME).exists()


class TestStorageGate:
    """格納 dtype の門（実測の事故が根拠 — `--dtype` 付け忘れの素 F32 は PNG 門まで沈黙した）。"""

    def test_it_stops_when_an_f16_component_is_stored_as_raw_f32(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        (sources.base / "text_encoder" / "model.safetensors").write_bytes(
            _fake_safetensors("F32", b"text-encoder-weights")
        )
        out_dir = tmp_path / "models" / "anima-turbo"
        with pytest.raises(DistError, match=r"text_encoder: .* F16 が無い"):
            _assemble_anima(sources, out_dir)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない（rope 不一致と同じ規律）。
        assert not out_dir.exists()

    def test_it_stops_when_the_i8_transformer_lacks_i8_storage(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        (sources.transformer_i8 / "transformer" / "model.safetensors").write_bytes(
            _fake_safetensors("F16", b"transformer-i8-weights")
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* I8 が無い"):
            _assemble_anima(sources, tmp_path / "models" / "anima-turbo")

    def test_it_stops_when_a_header_is_not_safetensors(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        (sources.base / "vae_decoder" / "model.safetensors").write_bytes(b"not-a-safetensors")
        with pytest.raises(DistError, match="ヘッダが読めない"):
            _assemble_anima(sources, tmp_path / "models" / "anima-turbo")


class TestModelName:
    """モデル名は manifest のキーであると同時にリポ内のディレクトリ名（ADR 0041 §6 / §9）。"""

    @pytest.mark.parametrize("name", ["../escape", "with space", ".hidden", "a/b", ""])
    def test_it_refuses_a_name_that_is_not_a_path_segment(self, name: str) -> None:
        with pytest.raises(DistError, match="許可文字"):
            assert_model_name(name)

    def test_it_refuses_the_name_reserved_for_shared_files(self) -> None:
        with pytest.raises(DistError, match=SHARED_DIRNAME):
            assert_model_name(SHARED_DIRNAME)

    def test_it_accepts_the_names_the_distributions_actually_use(self) -> None:
        for name in (ANIMA_MODEL_NAME, SBV2_DEFAULT_MODEL, "jvnv-F1"):
            assert assert_model_name(name) == name


class TestQuantCompletion:
    """quant の `weights` は hub が**完全写像**として検査する（ADR 0041 §3）。"""

    _weights: ClassVar[dict] = {
        "text_encoder": {"i8": WeightFiles("text_encoder")},
        "front": {"f16": WeightFiles("front_f16"), "i8": WeightFiles("front_i8")},
    }

    def test_it_fills_in_the_weights_that_have_a_single_dtype(self) -> None:
        completed = complete_quant_weights(
            self._weights, {"w8": {"weights": {"front": "i8"}, "session": {}}}
        )
        assert completed["w8"]["weights"] == {"front": "i8", "text_encoder": "i8"}

    def test_it_orders_the_mapping_by_the_weights_declaration(self) -> None:
        """埋めた席が末尾に溜まると weights 節と quant 節で同じ役割が別の順に並ぶ。"""
        completed = complete_quant_weights(
            self._weights, {"w8": {"weights": {"front": "i8"}, "session": {}}}
        )
        assert list(completed["w8"]["weights"]) == ["text_encoder", "front"]

    def test_it_keeps_the_rest_of_the_quant_untouched(self) -> None:
        quants = {
            "w8a8": {
                "weights": {"front": "i8"},
                "session": {"linearCompute": "i8a8"},
                "gpuFeatures": {"shaderF16": True},
            }
        }
        completed = complete_quant_weights(self._weights, quants)
        assert completed["w8a8"]["session"] == {"linearCompute": "i8a8"}
        assert completed["w8a8"]["gpuFeatures"] == {"shaderF16": True}

    def test_it_refuses_to_pick_a_dtype_when_the_weights_offer_a_choice(self) -> None:
        """埋めてよいのは選択肢が 1 つの席だけ — 既定を勝手に選ぶと別の格納形が黙って配られる。"""
        with pytest.raises(DistError, match="複数あるのに指定が無い"):
            complete_quant_weights(self._weights, {"w8": {"weights": {}, "session": {}}})

    def test_it_refuses_an_unknown_weights_name(self) -> None:
        with pytest.raises(DistError, match="未知の weights 'voice'"):
            complete_quant_weights(
                self._weights, {"w8": {"weights": {"voice": "i8"}, "session": {}}}
            )

    def test_it_refuses_a_dtype_the_weights_do_not_carry(self) -> None:
        with pytest.raises(DistError, match="dtype 'bf16' が無い"):
            complete_quant_weights(
                self._weights, {"w8": {"weights": {"front": "bf16"}, "session": {}}}
            )


class TestManifestLimits:
    """規模上限（ADR 0041 §7）は hub も同じ値で弾く — 配ってから落ちる形にしない。"""

    @staticmethod
    def _model(config: Any = None) -> dict[str, Any]:
        return {
            "pipeline": "anima/1",
            "weights": {"w": {"f16": {"file": _ref("m/w.safetensors")}}},
            "assets": {},
            "quants": {"f16": {"weights": {"w": "f16"}, "session": {}}},
            "defaultQuant": "f16",
            "pipelineConfig": {} if config is None else config,
        }

    def _manifest(self, models: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "format": MANIFEST_FORMAT,
            "generator": "karume/9.9.9",
            "defaultModel": next(iter(models)),
            "models": dict(models),
        }

    def test_it_accepts_the_largest_repository_the_hub_reads(self) -> None:
        models = {f"m{index}": self._model() for index in range(32)}
        assert_manifest_limits(self._manifest(models))

    def test_it_refuses_one_model_beyond_the_limit(self) -> None:
        models = {f"m{index}": self._model() for index in range(33)}
        with pytest.raises(DistError, match="models が 33 件"):
            assert_manifest_limits(self._manifest(models))

    def test_it_refuses_a_pipeline_config_beyond_the_limit(self) -> None:
        oversized = {"table": {f"k{index}": index for index in range(40_000)}}
        with pytest.raises(DistError, match="pipelineConfig"):
            assert_manifest_limits(self._manifest({"m": self._model(oversized)}))


class TestManifest:
    def test_it_writes_the_envelope_of_adr_0041(self, assembled) -> None:
        out_dir, manifest = assembled
        on_disk = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert manifest["format"] == MANIFEST_FORMAT
        assert manifest["generator"].startswith("karume/")
        assert manifest["defaultModel"] == ANIMA_MODEL_NAME
        assert list(manifest["models"]) == [ANIMA_MODEL_NAME]
        assert manifest["models"][ANIMA_MODEL_NAME]["pipeline"] == "anima/1"

    def test_every_weights_entry_is_keyed_by_dtype(self, assembled) -> None:
        """v1 の `{file}` / `{variants}` の 2 形は消えた — i8 単体も dtype キーを持つ（§3）。"""
        _, manifest = assembled
        weights = manifest["models"][ANIMA_MODEL_NAME]["weights"]
        assert sorted(weights) == sorted(ANIMA_WEIGHTS)
        for name, entry in weights.items():
            assert sorted(entry) == sorted(ANIMA_WEIGHTS[name]), name
            for files in entry.values():
                assert sorted(files) in (["file"], ["extras", "file"])

    def test_the_unconditional_files_live_in_assets(self, assembled) -> None:
        _, manifest = assembled
        assets = manifest["models"][ANIMA_MODEL_NAME]["assets"]
        assert sorted(assets) == ["tokenizer", "tokenizer_2"]
        for ref in assets.values():
            assert sorted(ref) == ["path", "sha256", "size"]

    def test_it_derives_size_and_sha256_from_the_placed_files(self, assembled) -> None:
        out_dir, manifest = assembled
        ref = manifest["models"][ANIMA_MODEL_NAME]["weights"]["text_encoder"]["f16"]["file"]
        payload = _PAYLOADS["text_encoder"]
        assert ref["size"] == len(payload)
        assert ref["sha256"] == hashlib.sha256(payload).hexdigest()
        assert (out_dir / ref["path"]).read_bytes() == payload

    def test_it_carries_the_quant_table_and_pipeline_config(self, assembled) -> None:
        _, manifest = assembled
        model = manifest["models"][ANIMA_MODEL_NAME]
        assert sorted(model["quants"]) == sorted(ANIMA_QUANTS)
        assert model["defaultQuant"] == ANIMA_DEFAULT_QUANT
        assert model["defaultQuant"] in model["quants"]
        assert model["pipelineConfig"] == dict(ANIMA_PIPELINE_CONFIG)

    def test_every_quant_maps_every_weights_entry(self, assembled) -> None:
        """hub は写像の完全性を実行時にも検査する — 埋め漏れは配布してから落ちる。"""
        _, manifest = assembled
        model = manifest["models"][ANIMA_MODEL_NAME]
        for name, quant in model["quants"].items():
            assert set(quant["weights"]) == set(model["weights"]), name
            for weight, label in quant["weights"].items():
                assert label in model["weights"][weight]


class TestVerifyDist:
    def test_it_passes_on_a_freshly_assembled_tree(self, assembled) -> None:
        out_dir, _ = assembled
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_MODEL_NAME))

    def test_it_catches_a_file_that_no_longer_matches_its_declared_size(self, assembled) -> None:
        out_dir, _ = assembled
        target = out_dir / ANIMA_MODEL_NAME / OUTPUT_PATHS["vae_decoder"]
        target.unlink()  # ハードリンクを外してから書く（源の系列を壊さない）
        target.write_bytes(b"shorter")
        with pytest.raises(DistError, match="size が manifest と違う"):
            verify_dist(out_dir)

    def test_it_catches_a_missing_file(self, assembled) -> None:
        out_dir, _ = assembled
        (out_dir / ANIMA_MODEL_NAME / OUTPUT_PATHS["tokenizer"]).unlink()
        with pytest.raises(DistError, match="参照するファイルが無い"):
            verify_dist(out_dir)

    def test_it_catches_an_undeclared_file(self, assembled) -> None:
        out_dir, _ = assembled
        (out_dir / ANIMA_MODEL_NAME / "transformer" / "io.s01024t0699.safetensors").write_bytes(
            b"stale"
        )
        with pytest.raises(DistError, match="宣言していないファイル"):
            verify_dist(out_dir)

    def test_it_admits_the_model_card_as_a_meta_file(self, assembled) -> None:
        """`README.md` は karume.json と同格のメタファイル（前回の組み立ての残りでも通す）。"""
        out_dir, _ = assembled
        (out_dir / MODEL_CARD_FILENAME).write_text("前回のモデルカード", encoding="utf-8")
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_MODEL_NAME))

    def test_it_still_refuses_a_meta_name_in_a_subdirectory(self, assembled) -> None:
        """例外は直下の 2 つだけ — 下位ディレクトリの同名は宣言外のまま。"""
        out_dir, _ = assembled
        (out_dir / ANIMA_MODEL_NAME / MODEL_CARD_FILENAME).write_text("紛れ込み", encoding="utf-8")
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
            manifest["models"][ANIMA_MODEL_NAME]["defaultQuant"] = "nonexistent"

        _rewrite(out_dir, drop_default)
        with pytest.raises(DistError, match="defaultQuant 'nonexistent'"):
            verify_dist(out_dir)

    def test_it_refuses_a_quant_that_leaves_a_weights_entry_unmapped(self, assembled) -> None:
        out_dir, _ = assembled

        def unmap(manifest: dict) -> None:
            del manifest["models"][ANIMA_MODEL_NAME]["quants"]["f16"]["weights"]["vae_decoder"]

        _rewrite(out_dir, unmap)
        with pytest.raises(DistError, match="完全写像でない"):
            verify_dist(out_dir)

    def test_it_refuses_a_quant_that_names_a_dtype_the_weights_lack(self, assembled) -> None:
        out_dir, _ = assembled

        def retype(manifest: dict) -> None:
            manifest["models"][ANIMA_MODEL_NAME]["quants"]["f16"]["weights"]["vae_decoder"] = "i8"

        _rewrite(out_dir, retype)
        with pytest.raises(DistError, match="dtype 'i8' が無い"):
            verify_dist(out_dir)

    def test_it_refuses_a_path_outside_the_model_subtree_and_shared(self, assembled) -> None:
        """レイアウトは一律「モデル別サブツリー + shared/」（ADR 0041 §9）。"""
        out_dir, _ = assembled

        def flatten(manifest: dict) -> None:
            ref = manifest["models"][ANIMA_MODEL_NAME]["assets"]["tokenizer"]
            ref["path"] = OUTPUT_PATHS["tokenizer"]

        _rewrite(out_dir, flatten)
        with pytest.raises(DistError, match="レイアウトはモデル別サブツリー"):
            verify_dist(out_dir)

    def test_it_refuses_two_references_to_one_path_that_disagree(self, assembled) -> None:
        """同一 path の共有は合法だが、{size, sha256} の食い違いは取得層を振動させる。"""
        out_dir, _ = assembled

        def bend(manifest: dict) -> None:
            entry = manifest["models"][ANIMA_MODEL_NAME]["weights"]["transformer"]
            entry["i8"]["extras"]["rope_base"]["size"] += 1

        _rewrite(out_dir, bend)
        with pytest.raises(DistError, match="食い違う"):
            verify_dist(out_dir)


class TestFamilyAssembly:
    """1 リポに複数モデル（ADR 0041 §2）+ 共有ファイルは `shared/` に 1 回だけ（§5）。"""

    @pytest.fixture
    def family(self, tmp_path: Path) -> tuple[Path, dict]:
        series = tmp_path / "series"
        # base（text 経路 / VAE / tokenizer）は共通、transformer だけモデルごとに違う中身。
        first = _build_series(series, model="anima-turbo", mark=b"-turbo")
        second = _build_series(series, model="anima-lite", mark=b"-lite")
        out_dir = tmp_path / "models" / "anima-family"
        manifest = assemble_family(
            [anima_plan(first, "anima-turbo"), anima_plan(second, "anima-lite")],
            out_dir,
            "anima-turbo",
        )
        return out_dir, manifest

    def test_it_declares_every_model_with_the_first_as_default(self, family) -> None:
        _, manifest = family
        assert list(manifest["models"]) == ["anima-turbo", "anima-lite"]
        assert manifest["defaultModel"] == "anima-turbo"

    def test_it_places_a_byte_identical_file_once_under_shared(self, family) -> None:
        out_dir, manifest = family
        shared_path = f"{SHARED_DIRNAME}/{OUTPUT_PATHS['text_encoder']}"
        for name in manifest["models"]:
            ref = manifest["models"][name]["weights"]["text_encoder"]["f16"]["file"]
            assert ref["path"] == shared_path
        assert (out_dir / shared_path).read_bytes() == _PAYLOADS["text_encoder"]
        # 各モデルのサブツリーには残らない（1 回だけ置く = 重複を配らない）。
        for name in manifest["models"]:
            assert not (out_dir / name / OUTPUT_PATHS["text_encoder"]).exists()

    def test_it_keeps_the_files_that_differ_inside_each_model_subtree(self, family) -> None:
        out_dir, manifest = family
        paths = {
            name: manifest["models"][name]["weights"]["transformer"]["i8"]["file"]["path"]
            for name in manifest["models"]
        }
        assert paths == {
            "anima-turbo": f"anima-turbo/{OUTPUT_PATHS['transformer_i8']}",
            "anima-lite": f"anima-lite/{OUTPUT_PATHS['transformer_i8']}",
        }
        assert (out_dir / paths["anima-turbo"]).read_bytes() != (
            out_dir / paths["anima-lite"]
        ).read_bytes()

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
        private_roles = ("transformer_f16", "transformer_i8")
        expected = [f"{SHARED_DIRNAME}/{OUTPUT_PATHS[role]}" for role in shared_roles]
        expected += [
            f"{model}/{OUTPUT_PATHS[role]}"
            for model in ("anima-turbo", "anima-lite")
            for role in private_roles
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
        first = _build_series(series, model="anima-turbo", mark=b"-turbo")
        second = _build_series(series, model="anima-lite", mark=b"-lite")
        out_dir = tmp_path / "models" / "anima-family"
        plans = [anima_plan(first, "anima-turbo"), anima_plan(second, "anima-lite")]
        before = assemble_family(plans, out_dir, "anima-turbo")
        after = assemble_family(plans, out_dir, "anima-turbo")
        assert before == after
        assert verify_dist(out_dir)

    def test_it_refuses_a_duplicated_model_name(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        plan = anima_plan(sources, ANIMA_MODEL_NAME)
        with pytest.raises(DistError, match="モデル名が重複"):
            assemble_family([plan, plan], tmp_path / "out", ANIMA_MODEL_NAME)

    def test_it_refuses_a_default_model_it_is_not_assembling(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        with pytest.raises(DistError, match="既定モデル 'anima-lite'"):
            assemble_family([anima_plan(sources, ANIMA_MODEL_NAME)], tmp_path / "out", "anima-lite")


class TestModelCard:
    """`karume dist` は組み立て + 検証の**後**にモデルカードを書く。"""

    def _run(self, tmp_path: Path, *argv: str) -> Path:
        _build_series(tmp_path / "series")
        out_dir = tmp_path / "dist"
        main(["--series", str(tmp_path / "series"), "--out", str(out_dir), *argv])
        return out_dir

    def test_it_writes_a_model_card_next_to_the_manifest(self, tmp_path: Path) -> None:
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert card.startswith("---\n")
        assert "base_model: circlestone-labs/Anima-Base-v1.0-Diffusers" in card

    def test_it_derives_the_file_table_from_the_assembled_tree(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        for rel_path in _in_subtree(ANIMA_MODEL_NAME):
            assert f"`{rel_path}`" in card
        size = (out_dir / ANIMA_MODEL_NAME / OUTPUT_PATHS["transformer_i8"]).stat().st_size
        assert f"{size:,} B" in card

    def test_it_names_the_repository_after_the_assembled_directory(self, tmp_path: Path) -> None:
        """ファミリーリポの ID は pipeline の定数にできない — 組み立て先から引く。"""
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert 'fromPretrained("hdae/dist"' in card

    def test_it_leaves_the_tree_verifiable_after_writing_the_card(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_MODEL_NAME))

    def test_it_reassembles_over_a_previous_card(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        first = (out_dir / MODEL_CARD_FILENAME).read_bytes()
        main(["--series", str(tmp_path / "series"), "--out", str(out_dir)])
        assert (out_dir / MODEL_CARD_FILENAME).read_bytes() == first


class TestCli:
    def test_it_defaults_to_anima_with_its_default_model(self) -> None:
        args = build_parser().parse_args([])
        assert args.pipeline == "anima"
        assert args.models is None  # 解決は main（pipeline ごとの既定を引くため）

    def test_the_model_flag_accumulates_into_a_family(self) -> None:
        args = build_parser().parse_args(["--model", "F1", "--model", "F2"])
        assert args.models == ["F1", "F2"]

    def test_it_knows_both_pipelines(self) -> None:
        assert sorted(PIPELINES) == ["anima", "sbv2"]
        assert PIPELINES["anima"].default_model == ANIMA_MODEL_NAME
        assert PIPELINES["sbv2"].default_model == SBV2_DEFAULT_MODEL

    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert default_out_dir(PIPELINES["anima"], ["anima-turbo"]).name == "karume-anima-turbo"
        assert default_out_dir(PIPELINES["sbv2"], ["FN4"]).name == "karume-sbv2-FN4"

    def test_it_refuses_to_invent_a_family_repository_name(self) -> None:
        """`karume-sbv2-jvnv` のようなファミリー名はモデル名の並びからは決まらない。"""
        with pytest.raises(DistError, match="--out"):
            default_out_dir(PIPELINES["sbv2"], ["F1", "F2"])

    def test_every_pipeline_renders_its_own_model_card(self) -> None:
        """カードは pipeline ごとのテンプレート — 描き手が他 pipeline の manifest を拒む。"""
        for name, spec in PIPELINES.items():
            manifest = {"models": {"m": {"pipeline": f"{name}/0"}}}
            for render_card in spec.card_profiles.values():
                with pytest.raises(ValueError):
                    render_card(manifest, "hdae/x")


class TestCardProfile:
    """帰属プロファイルの選択（`--card-profile`）— 誤帰属は配ってからでないと気づけない。"""

    def test_it_is_unset_until_asked_for(self) -> None:
        assert build_parser().parse_args([]).card_profile is None
        assert build_parser().parse_args(["--card-profile", "jvnv"]).card_profile == "jvnv"

    def test_a_pipeline_with_one_profile_needs_no_choice(self) -> None:
        """anima の帰属は 1 通りしかない（選びようがないものを聞かない）。"""
        profiles = PIPELINES["anima"].card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINES["anima"], None) is next(iter(profiles.values()))

    def test_it_refuses_to_pick_an_attribution_when_several_exist(self) -> None:
        """既定を黙って選ぶと、新しいファミリーへ前のファミリーの帰属がそのまま残る。"""
        with pytest.raises(DistError, match="--card-profile") as error:
            resolve_card_renderer(PIPELINES["sbv2"], None)
        assert "fn" in str(error.value)
        assert "jvnv" in str(error.value)

    def test_it_refuses_a_profile_it_does_not_have(self) -> None:
        with pytest.raises(DistError, match="jvnv"):
            resolve_card_renderer(PIPELINES["sbv2"], "FN9")

    def test_it_resolves_each_name_to_its_own_renderer(self) -> None:
        """名前ごとに別の描き手（束ね違いなら 2 つのファミリーが同じカードを描く）。"""
        profiles = PIPELINES["sbv2"].card_profiles
        assert sorted(profiles) == ["fn", "jvnv"]
        assert resolve_card_renderer(PIPELINES["sbv2"], "jvnv") is profiles["jvnv"]
        assert profiles["fn"] is not profiles["jvnv"]


# ---- SBV2（text-to-speech）---------------------------------------------------

#: `style_bert_vits2` は optional な `sbv2` dependency-group。定数から既定ノブを引く層だけが
#: これを要る（組み立て本体はノブを引数で受けるので、無い環境でも全経路を観測できる）。
requires_sbv2_package = pytest.mark.skipif(
    importlib.util.find_spec("style_bert_vits2") is None,
    reason="sbv2 dependency-group が無い（`uv sync --group sbv2` が前提）",
)

#: 偽 text_encoder の IR グラフ出力の本数（22 層 + embedding 出力）と、`symbols.json` が持つ
#: 取り出し位置。門 `assert_bert_hidden` が見るのは「本数 − 位置 == 22」なので、**この 2 つは
#: 対でしか動かせない**（実資産の 23 本 / 1 と同じ組み合わせ）。
_SBV2_GRAPH_OUTPUTS = 23
_SBV2_BERT_FROM_END = 1

#: 偽資産（役割ごとに違うバイト列 — 取り違えがハッシュで見える）。モデル 5 役は
#: `SBV2_STORAGE_REQUIREMENTS` が要求する dtype をヘッダに持つ。text_encoder だけは
#: IR コンテナとしても読まれる（出力本数の門）ので `__metadata__` を持つ。
_SBV2_PAYLOADS = {
    "text_encoder": _fake_safetensors(
        "I8",
        b"deberta-i8-weights",
        {
            IR_METADATA_KEY: json.dumps(
                {"outputs": [f"layer_norm_{index}" for index in range(_SBV2_GRAPH_OUTPUTS)]}
            )
        },
    ),
    "front_f16": _fake_safetensors("F16", b"front-f16-weights"),
    "front_i8": _fake_safetensors("I8", b"front-i8-weights"),
    "voice_f16": _fake_safetensors("F16", b"voice-f16-weights"),
    "voice_i8": _fake_safetensors("I8", b"voice-i8-weights"),
    "tokenizer": b'{"deberta": true}',
}

#: 実行時ノブの既定。**実際の `style_bert_vits2` の値とは別の数**にしてある — 組み立てが
#: 引数のノブを配るのか自分の表を配るのかを、値そのもので見分けるため。
_SBV2_KNOBS: Mapping[str, Any] = {
    "style": "Neutral",
    "styleWeight": 0.5,
    "sdpRatio": 0.25,
    "noiseScale": 0.55,
    "noiseScaleW": 0.75,
    "lengthScale": 1.25,
}

_SBV2_SYMBOLS = json.dumps(
    {"defaults": dict(_SBV2_KNOBS), "bertHiddenFromEnd": _SBV2_BERT_FROM_END},
    ensure_ascii=False,
).encode("utf-8")

#: 話者埋め込みの幅（合成 config の `model.gin_channels`。実重み FN4 は 512 だが、偽資産は
#: 幅を縛れていることが見えれば足りる — 実値との一致は config から導出するので固定しない）。
_SBV2_GIN_CHANNELS = 8

#: 合成 config。**実重み FN4 の style2id（Neutral / high / low / NSFW）とは別の並び**で、
#: 話者も 2 人。焼き込んだ表を配っていればここで落ちる。
_SBV2_CONFIG: Mapping[str, Any] = {
    "version": "2.6.1-JP-Extra",
    "data": {
        "spk2id": {"kappa": 0, "sigma": 1},
        "n_speakers": 2,
        "num_styles": 3,
        "style2id": {"Neutral": 0, "Sleepy": 1, "Shout": 2},
    },
    "model": {"gin_channels": _SBV2_GIN_CHANNELS},
}


def _style_table(rows: int, offset: float = 0.0) -> np.ndarray:
    """行ごとに違う値を持つ `[rows, 256]` の f32（配布形へ移した後の同一性を見るため）。"""
    return np.arange(rows * 256, dtype=np.float32).reshape(rows, 256) + offset


def _speaker_table(rows: int, cols: int = _SBV2_GIN_CHANNELS, offset: float = 0.0) -> np.ndarray:
    """ckpt の `emb_g.weight` に相当する `[rows, cols]` の f32（値は style 表と別系列）。"""
    return (np.arange(rows * cols, dtype=np.float32) * 0.5 - 1.0).reshape(rows, cols) + offset


def _write_ckpt(model_dir: Path, tensors: Mapping[str, np.ndarray]) -> None:
    """実重み ckpt を偽物で置く（`*.safetensors` の一意存在が dist の要求）。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(model_dir / "ckpt.safetensors"))


def _build_sbv2_sources(
    root: Path,
    *,
    model: str = SBV2_DEFAULT_MODEL,
    config: Mapping[str, Any] = _SBV2_CONFIG,
    style_rows: int = 3,
    speaker_rows: int = 2,
    speaker_cols: int = _SBV2_GIN_CHANNELS,
    symbols: bytes = _SBV2_SYMBOLS,
    offset: float = 0.0,
) -> Sbv2Sources:
    """系列 + デモ資産 + 実重みの置き場を偽資産で再現する（配布しないものの混入込み）。

    3 つの置き場の並びは `karume.paths` の実レイアウト（`outputs/series/` / `outputs/` 直下 /
    `inputs/`）に揃える — CLI 経路のテストが root を差し替えるだけで同じ木を指せる形。
    `offset` は表と重みにモデルごとの差を入れる軸（ファミリー組み立ての共有判定を見るため）。
    """
    series = root / "outputs" / "series"
    stem = sbv2_series_name(model)
    sources = Sbv2Sources(
        series_f16=series / f"{stem}-f16",
        series_i8=series / f"{stem}-i8",
        text_encoder=series / "deberta-i8" / "sbv2-22layer",
        demo=root / "outputs" / "sbv2-demo",
        model=root / "inputs" / "sbv2" / model,
    )
    _write(sources.text_encoder / "model.safetensors", _SBV2_PAYLOADS["text_encoder"])
    _write(sources.text_encoder / "io.case0.safetensors", b"io-fixture")
    for series_dir, label in ((sources.series_f16, "f16"), (sources.series_i8, "i8")):
        for target in ("front", "voice"):
            role = f"{target}_{label}"
            payload = _SBV2_PAYLOADS[role]
            if offset:
                dtype = "F16" if label == "f16" else "I8"
                payload = _fake_safetensors(dtype, f"{role}-{model}".encode())
            _write(series_dir / target / "model.safetensors", payload)
            _write(series_dir / target / "io.p2.safetensors", b"io-fixture")
        # 配布しない単体グラフ（golden 検証専用）も系列には並ぶ。
        for target in ("dp", "flow", "dec"):
            _write(series_dir / target / "model.safetensors", b"not-distributed")
    _write(sources.demo / "deberta-tokenizer.json", _SBV2_PAYLOADS["tokenizer"])
    _write(sources.demo / "symbols.json", symbols)
    # デモ専用資産（配布形には入らない）。
    _write(sources.demo / "assets.safetensors", b"demo-only")
    _write(sources.model / "config.json", json.dumps(config, ensure_ascii=False).encode("utf-8"))
    sources.model.mkdir(parents=True, exist_ok=True)
    np.save(sources.model / "style_vectors.npy", _style_table(style_rows, offset))
    # ckpt には話者埋め込み以外も並ぶ（名前で 1 本だけ引けていることが見える形）。
    _write_ckpt(
        sources.model,
        {
            "emb_g.weight": _speaker_table(speaker_rows, speaker_cols, offset),
            "dec.conv_pre.weight": np.zeros((2, 2), dtype=np.float32),
        },
    )
    return sources


def _assemble_sbv2(
    sources: Sbv2Sources, out_dir: Path, model: str = SBV2_DEFAULT_MODEL
) -> dict[str, Any]:
    return assemble_family([sbv2_plan(sources, _SBV2_KNOBS, model)], out_dir, model)


@pytest.fixture
def sbv2_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_sbv2_sources(tmp_path)
    out_dir = tmp_path / "models" / sbv2_repo_name(SBV2_DEFAULT_MODEL)
    manifest = _assemble_sbv2(sources, out_dir)
    return out_dir, manifest


def _sbv2_model(manifest: Mapping[str, Any], name: str = SBV2_DEFAULT_MODEL) -> Mapping[str, Any]:
    return manifest["models"][name]


class TestSbv2Layout:
    def test_it_places_every_declared_path_under_the_model_subtree(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        expected = _in_subtree(SBV2_DEFAULT_MODEL, SBV2_OUTPUT_PATHS.values())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_io_fixtures_into_the_distribution(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_leaves_the_single_graphs_out_of_the_distribution(self, sbv2_assembled) -> None:
        """dp / flow / dec は golden 検証専用（front / voice の融合が実行経路の全て）。"""
        out_dir, manifest = sbv2_assembled
        model = _sbv2_model(manifest)
        assert sorted(model["weights"]) == ["front", "text_encoder", "voice"]
        assert sorted(model["assets"]) == [
            "speaker_embeddings",
            "style_vectors",
            "symbols",
            "tokenizer",
        ]
        for target in ("dp", "flow", "dec", "decoder", "duration_predictor"):
            assert not (out_dir / SBV2_DEFAULT_MODEL / target).exists()

    def test_it_renames_the_two_series_into_dtype_files(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        for role in ("front_f16", "front_i8", "voice_f16", "voice_i8"):
            placed = out_dir / SBV2_DEFAULT_MODEL / SBV2_OUTPUT_PATHS[role]
            assert placed.read_bytes() == _SBV2_PAYLOADS[role]

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        out_dir = tmp_path / "models" / sbv2_repo_name(SBV2_DEFAULT_MODEL)
        _assemble_sbv2(sources, out_dir)
        _assemble_sbv2(sources, out_dir)  # 既存リンクがあっても落ちない
        assert verify_dist(out_dir)

    def test_it_rewrites_a_generated_table_without_touching_the_source(
        self, tmp_path: Path
    ) -> None:
        """生成物（表）は前回のリンクを外してから書く — 開いて書くと源の資産が壊れる。"""
        sources = _build_sbv2_sources(tmp_path)
        out_dir = tmp_path / "models" / sbv2_repo_name(SBV2_DEFAULT_MODEL)
        _assemble_sbv2(sources, out_dir)
        _assemble_sbv2(sources, out_dir)
        table = load_file(str(out_dir / SBV2_DEFAULT_MODEL / SBV2_OUTPUT_PATHS["style_vectors"]))[
            SBV2_STYLE_KEY
        ]
        assert np.array_equal(table, _style_table(3))
        assert np.array_equal(np.load(sources.model / "style_vectors.npy"), _style_table(3))


class TestSbv2StyleVectors:
    def test_it_converts_the_npy_into_a_single_f32_tensor(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        path = _sbv2_model(manifest)["assets"]["style_vectors"]["path"]
        tensors = load_file(str(out_dir / path))
        assert list(tensors) == [SBV2_STYLE_KEY]
        table = tensors[SBV2_STYLE_KEY]
        assert table.dtype == np.float32
        assert table.shape == (3, 256)
        assert np.array_equal(table, _style_table(3))

    def test_it_stops_when_the_row_count_disagrees_with_num_styles(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path, style_rows=2)
        out_dir = tmp_path / "models" / sbv2_repo_name(SBV2_DEFAULT_MODEL)
        with pytest.raises(DistError, match="行数 2 が config の num_styles 3"):
            _assemble_sbv2(sources, out_dir)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない。
        assert not out_dir.exists()

    def test_it_stops_when_the_row_count_disagrees_with_the_style_names(
        self, tmp_path: Path
    ) -> None:
        """`num_styles` とは合うのに名前が足りない形（行と名前の対応が崩れている）。"""
        data = {**_SBV2_CONFIG["data"], "style2id": {"Neutral": 0, "Sleepy": 1}}
        config = {**_SBV2_CONFIG, "data": data}
        sources = _build_sbv2_sources(tmp_path, config=config)
        with pytest.raises(DistError, match=r"style2id 2 件と一致しない"):
            sbv2_style_vectors(sources.model, config)

    def test_it_stops_when_the_table_is_not_two_dimensional(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        np.save(sources.model / "style_vectors.npy", np.zeros(256, dtype=np.float32))
        with pytest.raises(DistError, match=r"\[スタイル数, 256\] でない"):
            sbv2_style_vectors(sources.model, _SBV2_CONFIG)


class TestSbv2SpeakerEmbeddings:
    """`front` / `voice` はどちらも `g` をグラフ入力に取る — この表が無いと実行できない。"""

    def test_it_converts_the_ckpt_tensor_into_a_single_f32_tensor(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        path = _sbv2_model(manifest)["assets"]["speaker_embeddings"]["path"]
        tensors = load_file(str(out_dir / path))
        assert list(tensors) == [SBV2_SPEAKER_KEY]
        table = tensors[SBV2_SPEAKER_KEY]
        assert table.dtype == np.float32
        assert table.shape == (2, _SBV2_GIN_CHANNELS)
        assert np.array_equal(table, _speaker_table(2))

    def test_it_picks_the_speaker_tensor_by_name(self, tmp_path: Path) -> None:
        """ckpt には他のテンソルも並ぶ（名前で引けていなければ別の重みが配られる）。"""
        sources = _build_sbv2_sources(tmp_path)
        table = sbv2_speaker_embeddings(sources.model, _SBV2_CONFIG)
        assert np.array_equal(table, _speaker_table(2))

    def test_it_stops_when_the_row_count_disagrees_with_n_speakers(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path, speaker_rows=3)
        out_dir = tmp_path / "models" / sbv2_repo_name(SBV2_DEFAULT_MODEL)
        with pytest.raises(DistError, match="行数 3 が config の n_speakers 2"):
            _assemble_sbv2(sources, out_dir)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない。
        assert not out_dir.exists()

    def test_it_stops_when_the_row_count_disagrees_with_the_speaker_names(
        self, tmp_path: Path
    ) -> None:
        """`n_speakers` とは合うのに名前が足りない形（行と名前の対応が崩れている）。"""
        data = {**_SBV2_CONFIG["data"], "spk2id": {"kappa": 0}}
        config = {**_SBV2_CONFIG, "data": data}
        sources = _build_sbv2_sources(tmp_path, config=config)
        with pytest.raises(DistError, match=r"spk2id 1 件と一致しない"):
            sbv2_speaker_embeddings(sources.model, config)

    def test_it_stops_when_the_column_count_disagrees_with_gin_channels(
        self, tmp_path: Path
    ) -> None:
        """列数は config から導出できる（グラフ入力 g の幅そのもの）ので shape ごと縛る。"""
        sources = _build_sbv2_sources(tmp_path, speaker_cols=_SBV2_GIN_CHANNELS + 1)
        with pytest.raises(DistError, match="列数 9 が config の gin_channels 8"):
            sbv2_speaker_embeddings(sources.model, _SBV2_CONFIG)

    def test_it_stops_when_the_ckpt_has_no_speaker_embedding(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        _write_ckpt(sources.model, {"dec.conv_pre.weight": np.zeros((2, 2), dtype=np.float32)})
        with pytest.raises(DistError, match=SBV2_SPEAKER_TENSOR.replace(".", r"\.")):
            sbv2_speaker_embeddings(sources.model, _SBV2_CONFIG)

    def test_it_stops_when_the_ckpt_is_not_unique(self, tmp_path: Path) -> None:
        """どの ckpt から引いたかが黙って変わる形を塞ぐ（`load_net_g` と同じ要求）。"""
        sources = _build_sbv2_sources(tmp_path)
        save_file({"emb_g.weight": _speaker_table(2)}, str(sources.model / "another.safetensors"))
        with pytest.raises(DistError, match="ckpt が一意でない"):
            sbv2_speaker_embeddings(sources.model, _SBV2_CONFIG)

    def test_it_stops_when_the_config_has_no_model_section(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        with pytest.raises(DistError, match="'model' 節が無い"):
            sbv2_speaker_embeddings(sources.model, {"data": _SBV2_CONFIG["data"]})

    def test_the_speaker_map_resolves_into_the_shipped_table(self, sbv2_assembled) -> None:
        """`pipelineConfig.speakers` の値が話者表の行番号として閉じていること。"""
        out_dir, manifest = sbv2_assembled
        model = _sbv2_model(manifest)
        speakers = model["pipelineConfig"]["speakers"]
        table = load_file(str(out_dir / model["assets"]["speaker_embeddings"]["path"]))
        assert sorted(speakers.values()) == list(range(table[SBV2_SPEAKER_KEY].shape[0]))
        styles = model["pipelineConfig"]["styles"]
        style_table = load_file(str(out_dir / model["assets"]["style_vectors"]["path"]))
        assert sorted(styles.values()) == list(range(style_table[SBV2_STYLE_KEY].shape[0]))


class TestSbv2PipelineConfig:
    def test_it_reads_the_tables_from_the_config_instead_of_baking_them(self) -> None:
        """**同じ経路が別の表を返すこと**が「焼き込んでいない」の観測可能な形。"""
        first = sbv2_pipeline_config(
            {"data": {"style2id": {"Neutral": 0, "Sleepy": 1}, "spk2id": {"kappa": 0}}},
            _SBV2_KNOBS,
        )
        second = sbv2_pipeline_config(
            {
                "data": {
                    "style2id": {"Neutral": 0, "Angry": 1, "Sad": 2},
                    "spk2id": {"omega": 0, "psi": 1},
                }
            },
            _SBV2_KNOBS,
        )
        assert first["styles"] == {"Neutral": 0, "Sleepy": 1}
        assert second["styles"] == {"Neutral": 0, "Angry": 1, "Sad": 2}
        assert first["speakers"] == {"kappa": 0}
        assert second["speakers"] == {"omega": 0, "psi": 1}

    def test_the_assembled_manifest_carries_the_config_tables(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        config = _sbv2_model(manifest)["pipelineConfig"]
        assert config["styles"] == _SBV2_CONFIG["data"]["style2id"]
        assert config["speakers"] == _SBV2_CONFIG["data"]["spk2id"]

    def test_it_defaults_to_the_first_speaker_of_the_config(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        assert _sbv2_model(manifest)["pipelineConfig"]["defaults"]["speaker"] == "kappa"

    def test_it_fills_the_knob_defaults_from_the_given_values(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        defaults = _sbv2_model(manifest)["pipelineConfig"]["defaults"]
        assert {key: defaults[key] for key in SBV2_KNOB_KEYS} == dict(_SBV2_KNOBS)

    def test_it_stops_when_the_default_style_is_absent_from_the_config(self) -> None:
        knobs = {**_SBV2_KNOBS, "style": "Whisper"}
        with pytest.raises(DistError, match="既定スタイル 'Whisper'"):
            sbv2_pipeline_config(_SBV2_CONFIG, knobs)

    def test_it_stops_when_a_knob_default_is_missing(self) -> None:
        knobs = {key: value for key, value in _SBV2_KNOBS.items() if key != "noiseScaleW"}
        with pytest.raises(DistError, match="実行時ノブの既定が足りない"):
            sbv2_pipeline_config(_SBV2_CONFIG, knobs)

    def test_it_stops_when_the_config_has_no_style_table(self) -> None:
        with pytest.raises(DistError, match=r"data\.style2id"):
            sbv2_pipeline_config({"data": {"spk2id": {"kappa": 0}}}, _SBV2_KNOBS)

    def test_it_stops_when_the_config_has_no_data_section(self) -> None:
        with pytest.raises(DistError, match="'data' 節が無い"):
            sbv2_pipeline_config({"model": {"gin_channels": 512}}, _SBV2_KNOBS)


class TestSbv2StorageGate:
    """格納 dtype の門（Anima と同じ実測事故が根拠 — 素 F32 は参照一致の門まで沈黙した）。"""

    def test_it_stops_when_an_f16_series_holds_raw_f32(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        (sources.series_f16 / "front" / "model.safetensors").write_bytes(
            _fake_safetensors("F32", b"front-f16-weights")
        )
        out_dir = tmp_path / "models" / sbv2_repo_name(SBV2_DEFAULT_MODEL)
        with pytest.raises(DistError, match=r"front_f16: .* F16 が無い"):
            _assemble_sbv2(sources, out_dir)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない。
        assert not out_dir.exists()

    def test_it_stops_when_an_i8_series_lacks_i8_storage(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        (sources.series_i8 / "voice" / "model.safetensors").write_bytes(
            _fake_safetensors("F16", b"voice-i8-weights")
        )
        with pytest.raises(DistError, match=r"voice_i8: .* I8 が無い"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_text_encoder_is_not_i8(self, tmp_path: Path) -> None:
        """DeBERTa は i8 系列 1 本だけを配る（f32 の 1.32GB は配布に非現実的）。"""
        sources = _build_sbv2_sources(tmp_path)
        (sources.text_encoder / "model.safetensors").write_bytes(
            _fake_safetensors("F32", b"deberta-f32-weights")
        )
        with pytest.raises(DistError, match=r"text_encoder: .* I8 が無い"):
            _assemble_sbv2(sources, tmp_path / "out")


class TestSbv2BertHiddenGate:
    """層数と取り出し位置の対応（片方だけ動くと**別の層の声**が出るのに実行は通る）。

    層数は `export_deberta.py` の variant が、位置は `sbv2_demo.py` の定数が持つ別々の台本
    なので、対で動かし忘れた配布形が普通に組み上がってしまう。不変量は「出力本数 − 位置 == 22」。
    """

    def test_it_stops_when_only_the_layer_count_moved(self, tmp_path: Path) -> None:
        """層を 22 に削ったのに `symbols.json` が旧 24 層向けの 3 を主張したまま。"""
        symbols = json.dumps({"defaults": dict(_SBV2_KNOBS), "bertHiddenFromEnd": 3})
        sources = _build_sbv2_sources(tmp_path, symbols=symbols.encode("utf-8"))
        out_dir = tmp_path / "out"
        with pytest.raises(DistError, match=r"bertHiddenFromEnd=3 は先頭から 20 番目"):
            _assemble_sbv2(sources, out_dir)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない。
        assert not out_dir.exists()

    def test_it_stops_when_only_the_extraction_position_moved(self, tmp_path: Path) -> None:
        """`symbols.json` を 1 にしたのに text_encoder が全 25 本出しの 24 層のまま。"""
        sources = _build_sbv2_sources(tmp_path)
        (sources.text_encoder / "model.safetensors").write_bytes(
            _fake_safetensors(
                "I8",
                b"deberta-i8-weights",
                {IR_METADATA_KEY: json.dumps({"outputs": [f"h{index}" for index in range(25)]})},
            )
        )
        with pytest.raises(DistError, match=r"bertHiddenFromEnd=1 は先頭から 24 番目"):
            _assemble_sbv2(sources, tmp_path / "out")


class TestSbv2Manifest:
    def test_it_writes_the_envelope_of_adr_0041(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        on_disk = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert manifest["format"] == MANIFEST_FORMAT
        assert manifest["defaultModel"] == SBV2_DEFAULT_MODEL
        assert _sbv2_model(manifest)["pipeline"] == "sbv2/1"
        assert manifest["generator"].startswith("karume/")

    def test_it_derives_size_and_sha256_from_the_placed_files(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        ref = _sbv2_model(manifest)["weights"]["front"]["i8"]["file"]
        payload = _SBV2_PAYLOADS["front_i8"]
        assert ref["size"] == len(payload)
        assert ref["sha256"] == hashlib.sha256(payload).hexdigest()
        assert (out_dir / ref["path"]).read_bytes() == payload

    def test_the_single_dtype_text_encoder_still_has_a_dtype_key(self, sbv2_assembled) -> None:
        """i8 単体でも `{ "i8": … }` の統一形（ADR 0041 §3 — 2 形パースを消した）。"""
        _, manifest = sbv2_assembled
        entry = _sbv2_model(manifest)["weights"]["text_encoder"]
        assert list(entry) == ["i8"]
        assert entry["i8"]["file"]["path"].endswith("model.i8.safetensors")

    def test_it_carries_the_quant_table(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        model = _sbv2_model(manifest)
        assert sorted(model["quants"]) == sorted(SBV2_QUANTS)
        assert model["defaultQuant"] == SBV2_DEFAULT_QUANT
        assert model["defaultQuant"] in model["quants"]

    def test_every_quant_names_the_text_encoder_too(self, sbv2_assembled) -> None:
        """v1 で `weights` に書けなかった単一ファイル役も、v2 では完全写像の一部（§3）。"""
        _, manifest = sbv2_assembled
        model = _sbv2_model(manifest)
        for name, quant in model["quants"].items():
            assert set(quant["weights"]) == set(SBV2_WEIGHTS), name
            assert quant["weights"]["text_encoder"] == "i8"


class TestSbv2VerifyDist:
    def test_it_passes_on_a_freshly_assembled_tree(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        expected = _in_subtree(SBV2_DEFAULT_MODEL, SBV2_OUTPUT_PATHS.values())
        assert sorted(verify_dist(out_dir)) == sorted(expected)

    def test_it_passes_without_a_model_card(self, sbv2_assembled) -> None:
        """検証は**カードを書く前**に走る。無いことは宣言外ファイル検査の障害にならない。"""
        out_dir, _ = sbv2_assembled
        assert not (out_dir / MODEL_CARD_FILENAME).exists()
        assert verify_dist(out_dir)

    def test_it_catches_an_undeclared_file(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        (out_dir / SBV2_DEFAULT_MODEL / "voice" / "io.p2.safetensors").write_bytes(b"stale")
        with pytest.raises(DistError, match="宣言していないファイル"):
            verify_dist(out_dir)


class TestSbv2Family:
    """JVNV 4 モデルのような「1 リポ複数話者」— text_encoder / tokenizer が共有になる。"""

    @pytest.fixture
    def family(self, tmp_path: Path) -> tuple[Path, dict]:
        first = _build_sbv2_sources(tmp_path, model="F1")
        second = _build_sbv2_sources(tmp_path, model="F2", offset=7.0)
        out_dir = tmp_path / "models" / "karume-sbv2-family"
        manifest = assemble_family(
            [sbv2_plan(first, _SBV2_KNOBS, "F1"), sbv2_plan(second, _SBV2_KNOBS, "F2")],
            out_dir,
            "F1",
        )
        return out_dir, manifest

    def test_the_text_encoder_is_fetched_once_for_the_whole_family(self, family) -> None:
        """319MB の DeBERTa がモデルごとに複製されるのが v1 の実害（ADR 0041 Context ②）。"""
        _, manifest = family
        paths = {
            name: model["weights"]["text_encoder"]["i8"]["file"]["path"]
            for name, model in manifest["models"].items()
        }
        assert set(paths.values()) == {f"{SHARED_DIRNAME}/{SBV2_OUTPUT_PATHS['text_encoder']}"}

    def test_the_tables_that_differ_stay_per_model(self, family) -> None:
        out_dir, manifest = family
        first = manifest["models"]["F1"]["assets"]["style_vectors"]["path"]
        second = manifest["models"]["F2"]["assets"]["style_vectors"]["path"]
        assert first == f"F1/{SBV2_OUTPUT_PATHS['style_vectors']}"
        assert second == f"F2/{SBV2_OUTPUT_PATHS['style_vectors']}"
        assert np.array_equal(load_file(str(out_dir / first))[SBV2_STYLE_KEY], _style_table(3))
        assert np.array_equal(
            load_file(str(out_dir / second))[SBV2_STYLE_KEY], _style_table(3, 7.0)
        )

    def test_the_shared_assets_are_the_ones_both_models_produce_identically(self, family) -> None:
        _, manifest = family
        shared = {
            name
            for name, ref in manifest["models"]["F1"]["assets"].items()
            if ref["path"].startswith(f"{SHARED_DIRNAME}/")
        }
        assert shared == {"tokenizer", "symbols"}

    def test_the_assembled_family_verifies(self, family) -> None:
        out_dir, _ = family
        assert verify_dist(out_dir)


class TestSbv2Sources:
    def test_it_derives_the_two_series_from_the_model_name(self, tmp_path: Path) -> None:
        sources = sbv2_sources(tmp_path, "FN7")
        assert sources.series_f16 == tmp_path / "sbv2-FN7-f16"
        assert sources.series_i8 == tmp_path / "sbv2-FN7-i8"
        assert sources.model.name == "FN7"
        # DeBERTa はモデル名に依らない（ファミリーでは shared/ へ 1 本化される）。
        assert sources.text_encoder == tmp_path / "deberta-i8" / "sbv2-22layer"

    def test_it_looks_outside_the_series_root_for_the_host_assets(self, tmp_path: Path) -> None:
        """デモ資産（`outputs/` 直下）と実重み（`inputs/`）は系列ではない — 綴りは paths.py。"""
        sources = sbv2_sources(tmp_path)
        assert tmp_path not in sources.demo.parents
        assert tmp_path not in sources.model.parents
        assert sources.demo.name == "sbv2-demo"
        assert sources.model.name == SBV2_DEFAULT_MODEL


@requires_sbv2_package
class TestSbv2KnobDefaults:
    """既定ノブの出所（`style_bert_vits2.constants`）と配布資産の突合。"""

    @staticmethod
    def _package_knobs() -> dict[str, Any]:
        from style_bert_vits2.constants import (
            DEFAULT_LENGTH,
            DEFAULT_NOISE,
            DEFAULT_NOISEW,
            DEFAULT_SDP_RATIO,
            DEFAULT_STYLE,
            DEFAULT_STYLE_WEIGHT,
        )

        return {
            "style": DEFAULT_STYLE,
            "styleWeight": DEFAULT_STYLE_WEIGHT,
            "sdpRatio": DEFAULT_SDP_RATIO,
            "noiseScale": DEFAULT_NOISE,
            "noiseScaleW": DEFAULT_NOISEW,
            "lengthScale": DEFAULT_LENGTH,
        }

    def test_it_pulls_every_knob_from_the_package(self, tmp_path: Path) -> None:
        knobs = self._package_knobs()
        path = tmp_path / "symbols.json"
        path.write_text(json.dumps({"defaults": knobs}), encoding="utf-8")
        assert sbv2_knob_defaults(path) == knobs

    def test_it_stops_when_the_shipped_asset_disagrees(self, tmp_path: Path) -> None:
        """`symbols.json` と `karume.json` に同じ値が並ぶので、版ずれをここで落とす。"""
        knobs = {**self._package_knobs(), "noiseScaleW": 0.123}
        path = tmp_path / "symbols.json"
        path.write_text(json.dumps({"defaults": knobs}), encoding="utf-8")
        with pytest.raises(DistError, match="noiseScaleW"):
            sbv2_knob_defaults(path)

    def test_it_stops_when_the_shipped_asset_has_no_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "symbols.json"
        path.write_text(json.dumps({"symbols": ["_"]}), encoding="utf-8")
        with pytest.raises(DistError, match="'defaults' 節が無い"):
            sbv2_knob_defaults(path)


@requires_sbv2_package
class TestSbv2Cli:
    """`karume dist --pipeline sbv2` の経路（既定の出力先・`--model` 軸・ファミリー）。"""

    @staticmethod
    def _sources(tmp_path: Path, model: str, offset: float = 0.0):
        knobs = TestSbv2KnobDefaults._package_knobs()
        symbols = json.dumps(
            {"defaults": knobs, "bertHiddenFromEnd": _SBV2_BERT_FROM_END}, ensure_ascii=False
        ).encode("utf-8")
        return knobs, _build_sbv2_sources(tmp_path, model=model, symbols=symbols, offset=offset)

    @staticmethod
    def _reroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from karume import dist

        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(dist, "OUTPUTS_ROOT", tmp_path / "outputs")
        monkeypatch.setattr(dist, "INPUTS_ROOT", tmp_path / "inputs")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        knobs, sources = self._sources(tmp_path, SBV2_DEFAULT_MODEL)
        self._reroot(tmp_path, monkeypatch)

        main(
            [
                "--pipeline",
                "sbv2",
                "--card-profile",
                "fn",
                "--series",
                str(sources.series_f16.parent),
            ]
        )

        out_dir = tmp_path / "models" / sbv2_repo_name(SBV2_DEFAULT_MODEL)
        expected = _in_subtree(SBV2_DEFAULT_MODEL, SBV2_OUTPUT_PATHS.values())
        assert sorted(verify_dist(out_dir)) == sorted(expected)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        defaults = _sbv2_model(manifest)["pipelineConfig"]["defaults"]
        assert defaults["style"] == knobs["style"]
        # カードは検証を通った manifest から描かれる（スタイル表がそのまま本文に出る）。
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: text-to-speech" in card
        for style in _sbv2_model(manifest)["pipelineConfig"]["styles"]:
            assert f"| `{style}` |" in card

    def test_the_model_flag_moves_the_series_and_the_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, sources = self._sources(tmp_path, "FN7")
        self._reroot(tmp_path, monkeypatch)

        main(
            [
                "--pipeline",
                "sbv2",
                "--card-profile",
                "fn",
                "--model",
                "FN7",
                "--series",
                str(sources.series_f16.parent),
            ]
        )

        out_dir = tmp_path / "models" / "karume-sbv2-FN7"
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert list(manifest["models"]) == ["FN7"]
        assert verify_dist(out_dir)

    def test_repeating_the_model_flag_assembles_one_family_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, first = self._sources(tmp_path, "F1")
        _, second = self._sources(tmp_path, "F2", offset=3.0)
        assert second.demo == first.demo  # デモ資産は共通の置き場
        self._reroot(tmp_path, monkeypatch)
        out_dir = tmp_path / "models" / "karume-sbv2-jvnv"

        main(
            [
                "--pipeline",
                "sbv2",
                "--card-profile",
                "jvnv",
                "--model",
                "F1",
                "--model",
                "F2",
                "--series",
                str(first.series_f16.parent),
                "--out",
                str(out_dir),
            ]
        )

        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert list(manifest["models"]) == ["F1", "F2"]
        assert manifest["defaultModel"] == "F1"
        assert verify_dist(out_dir)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "## Model: F1" in card
        assert "## Model: F2" in card
        assert 'fromPretrained("hdae/karume-sbv2-jvnv"' in card
        # 帰属は選んだファミリーのもの（FN の出所が 1 語も混ざらない）。
        assert "base_model:\n  - litagin/style_bert_vits2_jvnv\n" in card
        assert "license: cc-by-sa-4.0" in card
        assert "rufflet17" not in card

    def test_it_refuses_to_assemble_sbv2_without_an_attribution_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """カードの帰属を選ばせる — しかも**組み立てる前**に落ちる（残骸を作らない）。"""
        _, sources = self._sources(tmp_path, SBV2_DEFAULT_MODEL)
        self._reroot(tmp_path, monkeypatch)

        with pytest.raises(DistError, match="--card-profile"):
            main(["--pipeline", "sbv2", "--series", str(sources.series_f16.parent)])

        assert not (tmp_path / "models").exists()
