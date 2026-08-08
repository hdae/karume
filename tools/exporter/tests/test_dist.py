"""配布ディレクトリ組み立て（`karume.dist`）の単体テスト。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合だけ**（格納形の門）
なので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。SBV2 は加えて
`config.json` と `style_vectors.npy` を読むが、どちらも数行 / 数 KB の合成物で足りる
（**合成 config の style2id は実重み FN4 と別の並び**にしてある — 表を焼き込んでいれば落ちる）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from karume.dist import (
    ANIMA_DEFAULT_PRESET,
    ANIMA_PIPELINE_CONFIG,
    ANIMA_PRESETS,
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    OUTPUT_PATHS,
    PIPELINES,
    SBV2_DEFAULT_PRESET,
    SBV2_DIST_NAME,
    SBV2_KNOB_KEYS,
    SBV2_MODEL_NAME,
    SBV2_OUTPUT_PATHS,
    SBV2_PRESETS,
    SBV2_SPEAKER_KEY,
    SBV2_SPEAKER_TENSOR,
    SBV2_STYLE_KEY,
    AnimaSources,
    DistError,
    Sbv2Sources,
    anima_sources,
    assemble_anima,
    assemble_sbv2,
    build_parser,
    main,
    sbv2_knob_defaults,
    sbv2_pipeline_config,
    sbv2_sources,
    sbv2_speaker_embeddings,
    sbv2_style_vectors,
    verify_dist,
)


def _fake_safetensors(dtype: str, payload: bytes) -> bytes:
    """格納 dtype の門を通る最小の safetensors（8 バイト長 + ヘッダ JSON + データ節）。"""
    header = json.dumps(
        {"w": {"dtype": dtype, "shape": [len(payload)], "data_offsets": [0, len(payload)]}}
    ).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + payload


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


def _build_series(series_dir: Path, *, i8_rope: bytes | None = None) -> AnimaSources:
    """系列レイアウト（`outputs/series/` 相当）を偽資産で再現する（`io.*` の混入込み）。"""
    sources = anima_sources(series_dir)
    _write(sources.base / "text_encoder" / "model.safetensors", _PAYLOADS["text_encoder"])
    _write(sources.base / "text_conditioner" / "model.safetensors", _PAYLOADS["text_conditioner"])
    _write(sources.base / "vae_decoder" / "model.safetensors", _PAYLOADS["vae_decoder"])
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.base / "text_encoder" / "io.t005.safetensors", b"io-fixture")
    _write(sources.base / "vae_decoder" / "io.case0.safetensors", b"io-fixture")
    for series, role in (
        (sources.transformer_f16, "transformer_f16"),
        (sources.transformer_i8, "transformer_i8"),
    ):
        _write(series / "transformer" / "model.safetensors", _PAYLOADS[role])
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


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_series(tmp_path / "series")
    out_dir = tmp_path / "models" / "anima-turbo"
    manifest = assemble_anima(sources, out_dir)
    return out_dir, manifest


class TestLayout:
    def test_it_places_every_declared_path_and_nothing_else(self, assembled) -> None:
        out_dir, _ = assembled
        present = sorted(
            str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file()
        )
        assert present == sorted([*OUTPUT_PATHS.values(), MANIFEST_FILENAME])

    def test_it_never_carries_io_fixtures_into_the_distribution(self, assembled) -> None:
        out_dir, _ = assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_renames_the_two_transformer_series_into_variant_files(self, assembled) -> None:
        out_dir, _ = assembled
        f16 = out_dir / "transformer" / "model.f16.safetensors"
        i8 = out_dir / "transformer" / "model.i8.safetensors"
        assert f16.read_bytes() == _PAYLOADS["transformer_f16"]
        assert i8.read_bytes() == _PAYLOADS["transformer_i8"]

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / "anima-turbo"
        assemble_anima(sources, out_dir)
        assemble_anima(sources, out_dir)  # 既存リンクがあっても落ちない
        assert verify_dist(out_dir)


class TestPlacementStrategy:
    def test_it_hardlinks_when_the_filesystem_allows_it(self, assembled) -> None:
        out_dir, _ = assembled
        placed = out_dir / OUTPUT_PATHS["text_encoder"]
        assert placed.stat().st_nlink >= 2

    def test_it_falls_back_to_copy_when_linking_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(source: Path, dest: Path) -> None:
            raise OSError(f"cross-device link: {source} → {dest}")

        monkeypatch.setattr(os, "link", refuse)
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / "anima-turbo"
        assemble_anima(sources, out_dir)
        placed = out_dir / OUTPUT_PATHS["text_encoder"]
        assert placed.read_bytes() == _PAYLOADS["text_encoder"]
        assert placed.stat().st_nlink == 1

    def test_it_stops_when_an_input_is_missing(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        (sources.base / "vae_decoder" / "model.safetensors").unlink()
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            assemble_anima(sources, tmp_path / "models" / "anima-turbo")


class TestRopeBase:
    def test_it_collapses_the_two_series_into_one_file(self, assembled) -> None:
        out_dir, manifest = assembled
        variants = manifest["components"]["transformer"]["variants"]
        f16_extra = variants["f16"]["extras"]["rope_base"]
        i8_extra = variants["i8"]["extras"]["rope_base"]
        assert f16_extra == i8_extra
        assert f16_extra["path"] == OUTPUT_PATHS["rope_base"]
        assert (out_dir / OUTPUT_PATHS["rope_base"]).read_bytes() == _PAYLOADS["rope_base"]

    def test_it_refuses_to_pick_a_side_when_the_series_disagree(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series", i8_rope=b"rope-base-table-DIFFERENT")
        out_dir = tmp_path / "models" / "anima-turbo"
        with pytest.raises(DistError, match="バイト同一でない"):
            assemble_anima(sources, out_dir)
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
            assemble_anima(sources, out_dir)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない（rope 不一致と同じ規律）。
        assert not out_dir.exists()

    def test_it_stops_when_the_i8_transformer_lacks_i8_storage(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        (sources.transformer_i8 / "transformer" / "model.safetensors").write_bytes(
            _fake_safetensors("F16", b"transformer-i8-weights")
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* I8 が無い"):
            assemble_anima(sources, tmp_path / "models" / "anima-turbo")

    def test_it_stops_when_a_header_is_not_safetensors(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        (sources.base / "vae_decoder" / "model.safetensors").write_bytes(b"not-a-safetensors")
        with pytest.raises(DistError, match="ヘッダが読めない"):
            assemble_anima(sources, tmp_path / "models" / "anima-turbo")


class TestManifest:
    def test_it_writes_the_envelope_of_adr_0038(self, assembled) -> None:
        out_dir, manifest = assembled
        on_disk = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert manifest["format"] == "karume/1"
        assert manifest["pipeline"] == "anima/1"
        assert manifest["generator"].startswith("karume/")

    def test_it_derives_size_and_sha256_from_the_placed_files(self, assembled) -> None:
        out_dir, manifest = assembled
        ref = manifest["components"]["text_encoder"]["file"]
        payload = _PAYLOADS["text_encoder"]
        assert ref["size"] == len(payload)
        assert ref["sha256"] == hashlib.sha256(payload).hexdigest()
        assert (out_dir / ref["path"]).read_bytes() == payload

    def test_it_carries_the_preset_table_and_pipeline_config(self, assembled) -> None:
        _, manifest = assembled
        assert manifest["presets"] == dict(ANIMA_PRESETS)
        assert manifest["defaultPreset"] == ANIMA_DEFAULT_PRESET
        assert manifest["defaultPreset"] in manifest["presets"]
        assert manifest["pipelineConfig"] == dict(ANIMA_PIPELINE_CONFIG)

    def test_it_maps_weights_for_every_component_that_has_variants(self, assembled) -> None:
        _, manifest = assembled
        with_variants = {
            name for name, spec in manifest["components"].items() if "variants" in spec
        }
        for name, preset in manifest["presets"].items():
            assert set(preset["weights"]) == with_variants, name
            for component, label in preset["weights"].items():
                assert label in manifest["components"][component]["variants"]


class TestVerifyDist:
    def test_it_passes_on_a_freshly_assembled_tree(self, assembled) -> None:
        out_dir, _ = assembled
        assert sorted(verify_dist(out_dir)) == sorted(OUTPUT_PATHS.values())

    def test_it_catches_a_file_that_no_longer_matches_its_declared_size(self, assembled) -> None:
        out_dir, _ = assembled
        target = out_dir / OUTPUT_PATHS["vae_decoder"]
        target.unlink()  # ハードリンクを外してから書く（源の系列を壊さない）
        target.write_bytes(b"shorter")
        with pytest.raises(DistError, match="size が manifest と違う"):
            verify_dist(out_dir)

    def test_it_catches_a_missing_file(self, assembled) -> None:
        out_dir, _ = assembled
        (out_dir / OUTPUT_PATHS["tokenizer"]).unlink()
        with pytest.raises(DistError, match="参照するファイルが無い"):
            verify_dist(out_dir)

    def test_it_catches_an_undeclared_file(self, assembled) -> None:
        out_dir, _ = assembled
        (out_dir / "transformer" / "io.s01024t0699.safetensors").write_bytes(b"stale")
        with pytest.raises(DistError, match="宣言していないファイル"):
            verify_dist(out_dir)

    def test_it_admits_the_model_card_as_a_meta_file(self, assembled) -> None:
        """`README.md` は karume.json と同格のメタファイル（前回の組み立ての残りでも通す）。"""
        out_dir, _ = assembled
        (out_dir / MODEL_CARD_FILENAME).write_text("前回のモデルカード", encoding="utf-8")
        assert sorted(verify_dist(out_dir)) == sorted(OUTPUT_PATHS.values())

    def test_it_still_refuses_a_meta_name_in_a_subdirectory(self, assembled) -> None:
        """例外は直下の 2 つだけ — 下位ディレクトリの同名は宣言外のまま。"""
        out_dir, _ = assembled
        (out_dir / "transformer" / MODEL_CARD_FILENAME).write_text("紛れ込み", encoding="utf-8")
        with pytest.raises(DistError, match="宣言していないファイル"):
            verify_dist(out_dir)


class TestModelCard:
    """`karume dist` は組み立て + 検証の**後**にモデルカードを書く。"""

    def _run(self, tmp_path: Path) -> Path:
        _build_series(tmp_path / "series")
        out_dir = tmp_path / "dist"
        main(["--series", str(tmp_path / "series"), "--out", str(out_dir)])
        return out_dir

    def test_it_writes_a_model_card_next_to_the_manifest(self, tmp_path: Path) -> None:
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert card.startswith("---\n")
        assert "base_model: circlestone-labs/Anima-Base-v1.0-Diffusers" in card

    def test_it_derives_the_file_table_from_the_assembled_tree(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        for rel_path in OUTPUT_PATHS.values():
            assert f"`{rel_path}`" in card
        size = (out_dir / OUTPUT_PATHS["transformer_i8"]).stat().st_size
        assert f"{size:,} B" in card

    def test_it_leaves_the_tree_verifiable_after_writing_the_card(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        assert sorted(verify_dist(out_dir)) == sorted(OUTPUT_PATHS.values())

    def test_it_reassembles_over_a_previous_card(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        first = (out_dir / MODEL_CARD_FILENAME).read_bytes()
        main(["--series", str(tmp_path / "series"), "--out", str(out_dir)])
        assert (out_dir / MODEL_CARD_FILENAME).read_bytes() == first


# ---- SBV2（text-to-speech）---------------------------------------------------

#: `style_bert_vits2` は optional な `sbv2` dependency-group。定数から既定ノブを引く層だけが
#: これを要る（組み立て本体はノブを引数で受けるので、無い環境でも全経路を観測できる）。
requires_sbv2_package = pytest.mark.skipif(
    importlib.util.find_spec("style_bert_vits2") is None,
    reason="sbv2 dependency-group が無い（`uv sync --group sbv2` が前提）",
)

#: 偽資産（役割ごとに違うバイト列 — 取り違えがハッシュで見える）。モデル 5 役は
#: `SBV2_STORAGE_REQUIREMENTS` が要求する dtype をヘッダに持つ。
_SBV2_PAYLOADS = {
    "text_encoder": _fake_safetensors("I8", b"deberta-i8-weights"),
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

_SBV2_SYMBOLS = json.dumps({"defaults": dict(_SBV2_KNOBS)}, ensure_ascii=False).encode("utf-8")

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


def _style_table(rows: int) -> np.ndarray:
    """行ごとに違う値を持つ `[rows, 256]` の f32（配布形へ移した後の同一性を見るため）。"""
    return np.arange(rows * 256, dtype=np.float32).reshape(rows, 256)


def _speaker_table(rows: int, cols: int = _SBV2_GIN_CHANNELS) -> np.ndarray:
    """ckpt の `emb_g.weight` に相当する `[rows, cols]` の f32（値は style 表と別系列）。"""
    return (np.arange(rows * cols, dtype=np.float32) * 0.5 - 1.0).reshape(rows, cols)


def _write_ckpt(model_dir: Path, tensors: Mapping[str, np.ndarray]) -> None:
    """実重み ckpt を偽物で置く（`*.safetensors` の一意存在が dist の要求）。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(model_dir / "FN4.safetensors"))


def _build_sbv2_sources(
    root: Path,
    *,
    config: Mapping[str, Any] = _SBV2_CONFIG,
    style_rows: int = 3,
    speaker_rows: int = 2,
    speaker_cols: int = _SBV2_GIN_CHANNELS,
    symbols: bytes = _SBV2_SYMBOLS,
) -> Sbv2Sources:
    """系列 + デモ資産 + 実重みの置き場を偽資産で再現する（配布しないものの混入込み）。

    3 つの置き場の並びは `karume.paths` の実レイアウト（`outputs/series/` / `outputs/` 直下 /
    `inputs/`）に揃える — CLI 経路のテストが root を差し替えるだけで同じ木を指せる形。
    """
    series = root / "outputs" / "series"
    sources = Sbv2Sources(
        series_f16=series / f"{SBV2_DIST_NAME}-f16",
        series_i8=series / f"{SBV2_DIST_NAME}-i8",
        text_encoder=series / "deberta-i8" / "full-24layer",
        demo=root / "outputs" / "sbv2-demo",
        model=root / "inputs" / "sbv2" / SBV2_MODEL_NAME,
    )
    _write(sources.text_encoder / "model.safetensors", _SBV2_PAYLOADS["text_encoder"])
    _write(sources.text_encoder / "io.case0.safetensors", b"io-fixture")
    for series_dir, label in ((sources.series_f16, "f16"), (sources.series_i8, "i8")):
        for target in ("front", "voice"):
            _write(series_dir / target / "model.safetensors", _SBV2_PAYLOADS[f"{target}_{label}"])
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
    np.save(sources.model / "style_vectors.npy", _style_table(style_rows))
    # ckpt には話者埋め込み以外も並ぶ（名前で 1 本だけ引けていることが見える形）。
    _write_ckpt(
        sources.model,
        {
            "emb_g.weight": _speaker_table(speaker_rows, speaker_cols),
            "dec.conv_pre.weight": np.zeros((2, 2), dtype=np.float32),
        },
    )
    return sources


@pytest.fixture
def sbv2_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_sbv2_sources(tmp_path)
    out_dir = tmp_path / "models" / SBV2_DIST_NAME
    manifest = assemble_sbv2(sources, out_dir, _SBV2_KNOBS)
    return out_dir, manifest


class TestSbv2Layout:
    def test_it_places_every_declared_path_and_nothing_else(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        present = sorted(
            str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file()
        )
        assert present == sorted([*SBV2_OUTPUT_PATHS.values(), MANIFEST_FILENAME])

    def test_it_never_carries_io_fixtures_into_the_distribution(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_leaves_the_single_graphs_out_of_the_distribution(self, sbv2_assembled) -> None:
        """dp / flow / dec は golden 検証専用（front / voice の融合が実行経路の全て）。"""
        out_dir, manifest = sbv2_assembled
        assert sorted(manifest["components"]) == sorted(
            [
                "text_encoder",
                "front",
                "voice",
                "tokenizer",
                "symbols",
                "style_vectors",
                "speaker_embeddings",
            ]
        )
        for target in ("dp", "flow", "dec", "decoder", "duration_predictor"):
            assert not (out_dir / target).exists()

    def test_it_renames_the_two_series_into_variant_files(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        for role in ("front_f16", "front_i8", "voice_f16", "voice_i8"):
            assert (out_dir / SBV2_OUTPUT_PATHS[role]).read_bytes() == _SBV2_PAYLOADS[role]

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        out_dir = tmp_path / "models" / SBV2_DIST_NAME
        assemble_sbv2(sources, out_dir, _SBV2_KNOBS)
        assemble_sbv2(sources, out_dir, _SBV2_KNOBS)  # 既存リンクがあっても落ちない
        assert verify_dist(out_dir)


class TestSbv2StyleVectors:
    def test_it_converts_the_npy_into_a_single_f32_tensor(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        tensors = load_file(str(out_dir / SBV2_OUTPUT_PATHS["style_vectors"]))
        assert list(tensors) == [SBV2_STYLE_KEY]
        table = tensors[SBV2_STYLE_KEY]
        assert table.dtype == np.float32
        assert table.shape == (3, 256)
        assert np.array_equal(table, _style_table(3))

    def test_it_stops_when_the_row_count_disagrees_with_num_styles(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path, style_rows=2)
        out_dir = tmp_path / "models" / SBV2_DIST_NAME
        with pytest.raises(DistError, match="行数 2 が config の num_styles 3"):
            assemble_sbv2(sources, out_dir, _SBV2_KNOBS)
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
        out_dir, _ = sbv2_assembled
        tensors = load_file(str(out_dir / SBV2_OUTPUT_PATHS["speaker_embeddings"]))
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
        out_dir = tmp_path / "models" / SBV2_DIST_NAME
        with pytest.raises(DistError, match="行数 3 が config の n_speakers 2"):
            assemble_sbv2(sources, out_dir, _SBV2_KNOBS)
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
        speakers = manifest["pipelineConfig"]["speakers"]
        table = load_file(str(out_dir / SBV2_OUTPUT_PATHS["speaker_embeddings"]))[SBV2_SPEAKER_KEY]
        assert sorted(speakers.values()) == list(range(table.shape[0]))
        styles = manifest["pipelineConfig"]["styles"]
        style_table = load_file(str(out_dir / SBV2_OUTPUT_PATHS["style_vectors"]))[SBV2_STYLE_KEY]
        assert sorted(styles.values()) == list(range(style_table.shape[0]))


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
        config = manifest["pipelineConfig"]
        assert config["styles"] == _SBV2_CONFIG["data"]["style2id"]
        assert config["speakers"] == _SBV2_CONFIG["data"]["spk2id"]

    def test_it_defaults_to_the_first_speaker_of_the_config(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        assert manifest["pipelineConfig"]["defaults"]["speaker"] == "kappa"

    def test_it_fills_the_knob_defaults_from_the_given_values(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        defaults = manifest["pipelineConfig"]["defaults"]
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
        out_dir = tmp_path / "models" / SBV2_DIST_NAME
        with pytest.raises(DistError, match=r"front_f16: .* F16 が無い"):
            assemble_sbv2(sources, out_dir, _SBV2_KNOBS)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない。
        assert not out_dir.exists()

    def test_it_stops_when_an_i8_series_lacks_i8_storage(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        (sources.series_i8 / "voice" / "model.safetensors").write_bytes(
            _fake_safetensors("F16", b"voice-i8-weights")
        )
        with pytest.raises(DistError, match=r"voice_i8: .* I8 が無い"):
            assemble_sbv2(sources, tmp_path / "models" / SBV2_DIST_NAME, _SBV2_KNOBS)

    def test_it_stops_when_the_text_encoder_is_not_i8(self, tmp_path: Path) -> None:
        """DeBERTa は i8 系列 1 本だけを配る（f32 の 1.32GB は配布に非現実的）。"""
        sources = _build_sbv2_sources(tmp_path)
        (sources.text_encoder / "model.safetensors").write_bytes(
            _fake_safetensors("F32", b"deberta-f32-weights")
        )
        with pytest.raises(DistError, match=r"text_encoder: .* I8 が無い"):
            assemble_sbv2(sources, tmp_path / "models" / SBV2_DIST_NAME, _SBV2_KNOBS)


class TestSbv2Manifest:
    def test_it_writes_the_envelope_of_adr_0038(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        on_disk = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert manifest["format"] == "karume/1"
        assert manifest["pipeline"] == "sbv2/1"
        assert manifest["generator"].startswith("karume/")

    def test_it_derives_size_and_sha256_from_the_placed_files(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        ref = manifest["components"]["front"]["variants"]["i8"]["file"]
        payload = _SBV2_PAYLOADS["front_i8"]
        assert ref["size"] == len(payload)
        assert ref["sha256"] == hashlib.sha256(payload).hexdigest()
        assert (out_dir / ref["path"]).read_bytes() == payload

    def test_it_carries_the_preset_table(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        assert manifest["presets"] == dict(SBV2_PRESETS)
        assert manifest["defaultPreset"] == SBV2_DEFAULT_PRESET
        assert manifest["defaultPreset"] in manifest["presets"]

    def test_it_maps_weights_for_every_component_that_has_variants(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        with_variants = {
            name for name, spec in manifest["components"].items() if "variants" in spec
        }
        assert with_variants == {"front", "voice"}
        for name, preset in manifest["presets"].items():
            assert set(preset["weights"]) == with_variants, name
            for component, label in preset["weights"].items():
                assert label in manifest["components"][component]["variants"]

    def test_it_never_names_a_single_file_component_in_weights(self, sbv2_assembled) -> None:
        """`{file}` 形を `weights` に書くと hub が `ManifestReferenceError` で弾く（§3）。"""
        _, manifest = sbv2_assembled
        single = {name for name, spec in manifest["components"].items() if "file" in spec}
        assert single == {
            "text_encoder",
            "tokenizer",
            "symbols",
            "style_vectors",
            "speaker_embeddings",
        }
        for name, preset in manifest["presets"].items():
            assert single.isdisjoint(preset["weights"]), name


class TestSbv2VerifyDist:
    def test_it_passes_on_a_freshly_assembled_tree(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        assert sorted(verify_dist(out_dir)) == sorted(SBV2_OUTPUT_PATHS.values())

    def test_it_passes_without_a_model_card(self, sbv2_assembled) -> None:
        """モデルカードは `anima/1` 専用。無いことは宣言外ファイル検査の障害にならない。"""
        out_dir, _ = sbv2_assembled
        assert not (out_dir / MODEL_CARD_FILENAME).exists()
        assert verify_dist(out_dir)

    def test_it_catches_an_undeclared_file(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        (out_dir / "voice" / "io.p2.safetensors").write_bytes(b"stale")
        with pytest.raises(DistError, match="宣言していないファイル"):
            verify_dist(out_dir)


class TestSbv2Sources:
    def test_it_derives_the_two_series_from_the_distribution_name(self, tmp_path: Path) -> None:
        sources = sbv2_sources(tmp_path)
        assert sources.series_f16 == tmp_path / f"{SBV2_DIST_NAME}-f16"
        assert sources.series_i8 == tmp_path / f"{SBV2_DIST_NAME}-i8"
        assert sources.text_encoder == tmp_path / "deberta-i8" / "full-24layer"

    def test_it_looks_outside_the_series_root_for_the_host_assets(self, tmp_path: Path) -> None:
        """デモ資産（`outputs/` 直下）と実重み（`inputs/`）は系列ではない — 綴りは paths.py。"""
        sources = sbv2_sources(tmp_path)
        assert tmp_path not in sources.demo.parents
        assert tmp_path not in sources.model.parents
        assert sources.demo.name == "sbv2-demo"
        assert sources.model.name == "FN4"


class TestPipelineDispatch:
    def test_it_defaults_to_anima(self) -> None:
        assert build_parser().parse_args([]).pipeline == "anima"

    def test_it_knows_both_pipelines(self) -> None:
        assert sorted(PIPELINES) == ["anima", "sbv2"]
        assert PIPELINES["anima"].dist_name == "anima-turbo"
        assert PIPELINES["sbv2"].dist_name == SBV2_DIST_NAME

    def test_only_anima_renders_a_model_card(self) -> None:
        assert PIPELINES["anima"].render_card is not None
        assert PIPELINES["sbv2"].render_card is None


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
    """`karume dist --pipeline sbv2` の経路（既定の出力先・モデルカード無しの検証）。"""

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from karume import dist

        knobs = TestSbv2KnobDefaults._package_knobs()
        symbols = json.dumps({"defaults": knobs}, ensure_ascii=False).encode("utf-8")
        sources = _build_sbv2_sources(tmp_path, symbols=symbols)
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(dist, "OUTPUTS_ROOT", tmp_path / "outputs")
        monkeypatch.setattr(dist, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "sbv2", "--series", str(sources.series_f16.parent)])

        out_dir = tmp_path / "models" / SBV2_DIST_NAME
        assert sorted(verify_dist(out_dir)) == sorted(SBV2_OUTPUT_PATHS.values())
        assert not (out_dir / MODEL_CARD_FILENAME).exists()
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["pipelineConfig"]["defaults"]["style"] == knobs["style"]
