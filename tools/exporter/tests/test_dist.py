"""配布ディレクトリ組み立て（`karume.dist`）の単体テスト。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合だけ**（格納形の門）
なので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from karume.dist import (
    ANIMA_DEFAULT_PRESET,
    ANIMA_PIPELINE_CONFIG,
    ANIMA_PRESETS,
    MANIFEST_FILENAME,
    OUTPUT_PATHS,
    AnimaSources,
    DistError,
    anima_sources,
    assemble_anima,
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


def _build_series(models_dir: Path, *, i8_rope: bytes | None = None) -> AnimaSources:
    """`models/` の系列レイアウトを偽資産で再現する（`io.*` の混入込み）。"""
    sources = anima_sources(models_dir)
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
    sources = _build_series(tmp_path / "models")
    out_dir = tmp_path / "models" / "anima"
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
        sources = _build_series(tmp_path / "models")
        out_dir = tmp_path / "models" / "anima"
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
        sources = _build_series(tmp_path / "models")
        out_dir = tmp_path / "models" / "anima"
        assemble_anima(sources, out_dir)
        placed = out_dir / OUTPUT_PATHS["text_encoder"]
        assert placed.read_bytes() == _PAYLOADS["text_encoder"]
        assert placed.stat().st_nlink == 1

    def test_it_stops_when_an_input_is_missing(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "models")
        (sources.base / "vae_decoder" / "model.safetensors").unlink()
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            assemble_anima(sources, tmp_path / "models" / "anima")


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
        sources = _build_series(tmp_path / "models", i8_rope=b"rope-base-table-DIFFERENT")
        out_dir = tmp_path / "models" / "anima"
        with pytest.raises(DistError, match="バイト同一でない"):
            assemble_anima(sources, out_dir)
        # 止めた以上、途中の配布形を残さない（片方だけ入った出力を後段に見せない）。
        assert not (out_dir / MANIFEST_FILENAME).exists()


class TestStorageGate:
    """格納 dtype の門（実測の事故が根拠 — `--dtype` 付け忘れの素 F32 は PNG 門まで沈黙した）。"""

    def test_it_stops_when_an_f16_component_is_stored_as_raw_f32(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "models")
        (sources.base / "text_encoder" / "model.safetensors").write_bytes(
            _fake_safetensors("F32", b"text-encoder-weights")
        )
        out_dir = tmp_path / "models" / "anima"
        with pytest.raises(DistError, match=r"text_encoder: .* F16 が無い"):
            assemble_anima(sources, out_dir)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない（rope 不一致と同じ規律）。
        assert not out_dir.exists()

    def test_it_stops_when_the_i8_transformer_lacks_i8_storage(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "models")
        (sources.transformer_i8 / "transformer" / "model.safetensors").write_bytes(
            _fake_safetensors("F16", b"transformer-i8-weights")
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* I8 が無い"):
            assemble_anima(sources, tmp_path / "models" / "anima")

    def test_it_stops_when_a_header_is_not_safetensors(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "models")
        (sources.base / "vae_decoder" / "model.safetensors").write_bytes(b"not-a-safetensors")
        with pytest.raises(DistError, match="ヘッダが読めない"):
            assemble_anima(sources, tmp_path / "models" / "anima")


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
