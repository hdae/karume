"""Anima の配布 recipe（`anima.distribution`）— 系列 → 配布形の組み立て。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合だけ**（格納形の門）
なので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。

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
from pathlib import Path
from typing import Any

import pytest

from anima.card import ATTRIBUTION_NOTICE, LORA_NAME, LORA_SHA256, LORA_SOURCE
from anima.distribution import (
    ANIMA_DEFAULT_QUANT,
    ANIMA_MODEL_NAME,
    ANIMA_PIPELINE_CONFIG,
    ANIMA_QUANTS,
    ANIMA_STORAGE_FORBIDDEN,
    ANIMA_WEIGHTS,
    LICENSE_SOURCE_PATH,
    LORA_PROVENANCE_FILE,
    NOTICE_MARKDOWN,
    OUTPUT_PATHS,
    PIPELINE,
    STORAGE_REQUIREMENTS,
    AnimaSources,
    anima_plan,
    anima_sources,
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
    materialize,
    resolve_card_renderer,
    verify_dist,
)


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


#: 偽資産の中身（役割ごとに違うバイト列 — 取り違えがハッシュで見える）。モデル 5 役は
#: `STORAGE_REQUIREMENTS` が要求する dtype をヘッダに持つ。rope_base はヘッダ検査の対象外
#: だが、実物同様に正規形にしておく。
_PAYLOADS = {
    "text_encoder": _fake_safetensors("F16", b"text-encoder-weights"),
    "text_conditioner": _fake_safetensors("F16", b"text-conditioner-weights"),
    "transformer_f16": _fake_safetensors("F16", b"transformer-f16-weights"),
    "transformer_i8": _fake_safetensors("I8", b"transformer-i8-weights"),
    "transformer_i4": _fake_safetensors("I4", b"transformer-i4-weights"),
    "rope_base": _fake_safetensors("F32", b"rope-base-table"),
    "vae_decoder": _fake_safetensors("F16", b"vae-decoder-weights"),
    "tokenizer": b'{"qwen2": true}',
    "tokenizer_2": b'{"t5": true}',
}


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _lora_record(sha256: str) -> bytes:
    """`anima/export.py` が焼き込み時に残す帰属の記録（実物と同じ形）。"""
    return json.dumps({"file": "anima-turbo-lora-v0.2.safetensors", "sha256": sha256}).encode(
        "utf-8"
    )


def _build_series(
    series_dir: Path,
    *,
    model: str = ANIMA_MODEL_NAME,
    i8_rope: bytes | None = None,
    i4_rope: bytes | None = None,
    mark: bytes = b"",
    lora_sha256: str | None = None,
) -> AnimaSources:
    """系列レイアウト（`outputs/series/` 相当）を偽資産で再現する（`io.*` の混入込み）。

    `mark` は transformer 系列だけに混ぜる差分 — ファミリー組み立てで「モデルごとに違う重み」と
    「モデル間で同一の base 資産」を作り分けるための軸。`lora_sha256` は帰属の記録を
    カードの宣言からずらす軸（既定は一致する値）。`i8_rope` / `i4_rope` は rope 素表を
    f16 系列からずらす軸（系列ごとに独立に振れる — 網が全系列に掛かっていることを見るため）。
    """
    sources = anima_sources(series_dir, model)
    _write(sources.base / "text_encoder" / "model.safetensors", _PAYLOADS["text_encoder"])
    _write(sources.base / "text_conditioner" / "model.safetensors", _PAYLOADS["text_conditioner"])
    _write(sources.base / "vae_decoder" / "model.safetensors", _PAYLOADS["vae_decoder"])
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.base / "text_encoder" / "io.t005.safetensors", b"io-fixture")
    _write(sources.base / "vae_decoder" / "io.case0.safetensors", b"io-fixture")
    for series, role, dtype, rope in (
        (sources.transformer_f16, "transformer_f16", "F16", None),
        (sources.transformer_i8, "transformer_i8", "I8", i8_rope),
        (sources.transformer_i4, "transformer_i4", "I4", i4_rope),
    ):
        payload = (
            _PAYLOADS[role] if not mark else _fake_safetensors(dtype, role.encode("utf-8") + mark)
        )
        _write(series / "transformer" / "model.safetensors", payload)
        _write(series / "transformer" / "io.s01024t0699.safetensors", b"io-fixture")
        _write(
            series / "transformer" / LORA_PROVENANCE_FILE,
            _lora_record(LORA_SHA256 if lora_sha256 is None else lora_sha256),
        )
        _write(
            series / "transformer" / "rope_base.safetensors",
            _PAYLOADS["rope_base"] if rope is None else rope,
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

    def test_it_gives_the_i4_series_its_own_dtype_file(self, assembled) -> None:
        """i4 は f16 / i8 と並ぶ 3 本目の格納席（同じ path へ載せると席が 1 つ消える）。"""
        out_dir, _ = assembled
        placed = out_dir / ANIMA_MODEL_NAME / OUTPUT_PATHS["transformer_i4"]
        assert placed.name == "model.i4.safetensors"
        assert placed.read_bytes() == _PAYLOADS["transformer_i4"]

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

    def test_the_check_reaches_the_i4_series_too(self, tmp_path: Path) -> None:
        """網が f16 / i8 の 2 系列に留まっていると、i4 席だけ別の幾何で走って絵が静かに壊れる。"""
        sources = _build_series(tmp_path / "series", i4_rope=b"rope-base-table-DIFFERENT")
        out_dir = tmp_path / "models" / "anima-turbo"
        with pytest.raises(DistError, match="バイト同一でない"):
            _assemble_anima(sources, out_dir)
        assert not (out_dir / MANIFEST_FILENAME).exists()


class TestLoraProvenance:
    """カードが印字する LoRA の帰属は、系列に残った記録と組み立て時に突き合わせる。

    融合後の重みからは焼いた LoRA を復元できないので、突き合わせが無いと差し替え後も古い
    sha256 が公開される — 値は 64 桁 hex として形式が妥当なので `verify_dist` の構造検査も
    通り、配布 README の帰属だけが黙って嘘になる。
    """

    def test_it_stops_when_the_recorded_lora_is_not_the_one_the_card_declares(
        self, tmp_path: Path
    ) -> None:
        sources = _build_series(tmp_path / "series", lora_sha256="9" * 64)
        out_dir = tmp_path / "models" / "anima-turbo"

        with pytest.raises(DistError, match="カードの宣言と違う"):
            _assemble_anima(sources, out_dir)

        # 計画段の検査なので配布形は 1 ファイルも生えない。
        assert not out_dir.exists()

    def test_it_stops_when_the_series_carries_no_record_at_all(self, tmp_path: Path) -> None:
        """記録の無い系列（帰属を突き合わせられない）も緑にしない。"""
        sources = _build_series(tmp_path / "series")
        (sources.transformer_i8 / "transformer" / LORA_PROVENANCE_FILE).unlink()

        with pytest.raises(DistError, match="焼き込んだ LoRA の記録が無い"):
            _assemble_anima(sources, tmp_path / "models" / "anima-turbo")

    def test_it_stops_when_the_record_is_not_readable_json(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        (sources.transformer_f16 / "transformer" / LORA_PROVENANCE_FILE).write_bytes(b"{oops")

        with pytest.raises(DistError, match="LoRA の記録を解析できない"):
            _assemble_anima(sources, tmp_path / "models" / "anima-turbo")

    def test_the_record_never_reaches_the_distribution(self, assembled) -> None:
        """記録は系列側の事実 — 配布形（HF リポ）には持ち出さない。"""
        out_dir, _ = assembled

        assert not any(name.endswith(LORA_PROVENANCE_FILE) for name in _present(out_dir))


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

    def test_it_stops_when_the_i4_transformer_lacks_i4_storage(self, tmp_path: Path) -> None:
        """i4 席へ i8 系列が入る取り違え — 要求が I8 のままだと素通りして沈黙する。"""
        sources = _build_series(tmp_path / "series")
        (sources.transformer_i4 / "transformer" / "model.safetensors").write_bytes(
            _fake_safetensors("I8", b"transformer-i4-weights")
        )
        with pytest.raises(DistError, match=r"transformer_i4: .* I4 が無い"):
            _assemble_anima(sources, tmp_path / "models" / "anima-turbo")

    def test_it_stops_when_the_i4_series_lands_in_the_i8_seat(self, tmp_path: Path) -> None:
        """逆向きの取り違え（i4 系列 → i8 席）— 存在検査だけでは**素通りする**。

        i4 系列は混成で既定格納が i8 なので必ず I8 を含み、「I8 を含む」を満たしてしまう。
        既定 quant `w8a8-s16` が i4 常駐を掴むと、i8a8 の述語が外れて fail loudly せず
        f32 計算経路へ黙って落ちる — 禁止表（`ANIMA_STORAGE_FORBIDDEN`）が唯一の検出器。
        """
        sources = _build_series(tmp_path / "series")
        (sources.transformer_i8 / "transformer" / "model.safetensors").write_bytes(
            _mixed_safetensors(("I4", "I8", "F32"), b"transformer-i4-weights")
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* I4 がある"):
            _assemble_anima(sources, tmp_path / "models" / "anima-turbo")

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

        for seat in headers:
            for series, found in headers.items():
                caught = STORAGE_REQUIREMENTS[seat] not in found or any(
                    dtype in found for dtype in ANIMA_STORAGE_FORBIDDEN.get(seat, ())
                )
                assert caught is (series != seat), f"{series} → {seat} 席"

    def test_it_stops_when_a_header_is_not_safetensors(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        (sources.base / "vae_decoder" / "model.safetensors").write_bytes(b"not-a-safetensors")
        with pytest.raises(DistError, match="ヘッダが読めない"):
            _assemble_anima(sources, tmp_path / "models" / "anima-turbo")


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
        out_dir = tmp_path / "models" / "anima-turbo"
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
                assert sorted(files) in (["shards"], ["extras", "shards"])

    def test_the_unconditional_files_live_in_assets(self, assembled) -> None:
        _, manifest = assembled
        assets = manifest["models"][ANIMA_MODEL_NAME]["assets"]
        assert sorted(assets) == ["tokenizer", "tokenizer_2"]
        for ref in assets.values():
            assert sorted(ref) == ["path", "sha256", "size"]

    def test_it_derives_size_and_sha256_from_the_placed_files(self, assembled) -> None:
        out_dir, manifest = assembled
        ref = manifest["models"][ANIMA_MODEL_NAME]["weights"]["text_encoder"]["f16"]["shards"][0]
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


class TestI4Quants:
    """i4 常駐の 2 席（`w4` / `w4-a8-s16`）— 波 J-4 先行の速度実測用。"""

    @staticmethod
    def _i4_seats() -> dict[str, Any]:
        return {
            name: quant
            for name, quant in ANIMA_QUANTS.items()
            if quant["weights"].get("transformer") == "i4"
        }

    def test_it_declares_exactly_the_two_i4_seats(self) -> None:
        assert sorted(self._i4_seats()) == ["w4", "w4-a8-s16"]

    def test_the_plain_seat_leaves_the_session_untouched(self) -> None:
        """`w4` は格納だけを動かす席（計算経路は f32 のまま）。"""
        assert ANIMA_QUANTS["w4"]["session"] == {}

    def test_the_attention_seat_declares_only_the_attention_knobs(self) -> None:
        assert ANIMA_QUANTS["w4-a8-s16"]["session"] == {
            "attentionCompute": "i8a8",
            "attentionScoreStorage": "f16",
        }

    def test_no_i4_seat_declares_linear_compute(self) -> None:
        """MUST: i4 席に `linearCompute` を宣言しない。**この不変条件の理由は 2026-08-21 に
        入れ替わっている** — 旧: 「i8a8 の述語が i8 常駐を要求するので宣言しても効かない」/
        新: w4a8（ADR 0076）で効くようになったが、**掛けると画の細部が荒れる**という視認裁定
        （research 2026-08-21 §6）。速度は戻る（1,640 → 955 ms/step）が、この席は
        サイズ・VRAM のための席で、速度が要るなら既定の `w8a8-s16` が上。
        """
        for name, quant in self._i4_seats().items():
            assert "linearCompute" not in quant["session"], name

    def test_the_default_quant_stays_on_the_i8_seat(self) -> None:
        """席が増えても既定は動かさない（既定の変更は品質裁定を要する別の判断）。"""
        assert ANIMA_DEFAULT_QUANT == "w8a8-s16"

    def test_the_seats_reach_the_manifest_with_their_session_knobs(self, assembled) -> None:
        """表に足しただけで配布形へ出ること（quant 表は manifest 由来 — ADR 0041 §3）。"""
        _, manifest = assembled
        quants = manifest["models"][ANIMA_MODEL_NAME]["quants"]

        for name, quant in self._i4_seats().items():
            assert quants[name]["weights"]["transformer"] == "i4"
            assert quants[name]["session"] == dict(quant["session"])


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
            ref = manifest["models"][name]["weights"]["text_encoder"]["f16"]["shards"][0]
            assert ref["path"] == shared_path
        assert (out_dir / shared_path).read_bytes() == _PAYLOADS["text_encoder"]
        # 各モデルのサブツリーには残らない（1 回だけ置く = 重複を配らない）。
        for name in manifest["models"]:
            assert not (out_dir / name / OUTPUT_PATHS["text_encoder"]).exists()

    def test_it_keeps_the_files_that_differ_inside_each_model_subtree(self, family) -> None:
        out_dir, manifest = family
        paths = {
            name: manifest["models"][name]["weights"]["transformer"]["i8"]["shards"][0]["path"]
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
        private_roles = ("transformer_f16", "transformer_i8", "transformer_i4")
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
        out_dir = tmp_path / "models" / ANIMA_MODEL_NAME
        self._fail_at(monkeypatch, nth=3)
        with pytest.raises(OSError, match="配置の途中で落ちた"):
            _assemble_anima(sources, out_dir)

        assert not out_dir.exists()
        assert list(out_dir.parent.iterdir()) == []

    def test_it_writes_the_card_into_the_tree_it_swaps_in(self, tmp_path: Path) -> None:
        """`render_card` は組み立て済みの manifest を受け取り、据わる木の中に書かれる。"""
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_MODEL_NAME
        assemble_family(
            [anima_plan(sources, ANIMA_MODEL_NAME)],
            out_dir,
            ANIMA_MODEL_NAME,
            render_card=lambda manifest: f"{manifest['defaultModel']}\n",
        )
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert card == f"{ANIMA_MODEL_NAME}\n"
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_MODEL_NAME))
        assert self._siblings(out_dir) == []

    def test_reassembling_a_subset_replaces_the_whole_repository(self, tmp_path: Path) -> None:
        """再組み立ては plan に無いモデルごと丸ごと置き換える（前回の残骸が生き残らない）。"""
        series = tmp_path / "series"
        first = _build_series(series, model="anima-turbo", mark=b"-turbo")
        second = _build_series(series, model="anima-lite", mark=b"-lite")
        out_dir = tmp_path / "models" / "anima-family"
        assemble_family(
            [anima_plan(first, "anima-turbo"), anima_plan(second, "anima-lite")],
            out_dir,
            "anima-turbo",
        )

        manifest = assemble_family([anima_plan(first, "anima-turbo")], out_dir, "anima-turbo")

        assert list(manifest["models"]) == ["anima-turbo"]
        assert not (out_dir / "anima-lite").exists()
        # 1 モデルだけなら畳む相手が居ない = `shared/` も残らない（ADR 0041 §5）。
        assert not (out_dir / SHARED_DIRNAME).exists()
        assert _present(out_dir) == sorted([*_in_subtree("anima-turbo"), MANIFEST_FILENAME])
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree("anima-turbo"))

    def test_it_discards_a_working_directory_left_by_an_interrupted_run(
        self, tmp_path: Path
    ) -> None:
        """中断が残した staging は踏み直さずに捨てる（plan に無いファイルを混ぜない）。"""
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_MODEL_NAME
        leftover = out_dir.with_name(out_dir.name + STAGING_SUFFIX)
        _write(leftover / "anima-lite" / "transformer" / "model.f16.safetensors", b"interrupted")

        _assemble_anima(sources, out_dir)

        assert _present(out_dir) == sorted([*_in_subtree(ANIMA_MODEL_NAME), MANIFEST_FILENAME])
        assert self._siblings(out_dir) == []

    def test_a_successful_reassembly_lands_the_same_tree_without_leftovers(
        self, tmp_path: Path
    ) -> None:
        """成功経路は据え替えても同値 — manifest も現物も 1 回目と同じで、作業跡も残らない。"""
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_MODEL_NAME
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


class TestLegalTexts:
    """上流ライセンス（CircleStone Non-Commercial License）§3 の再配布条件を配布リポで満たす。

    条件は「ライセンス文のコピーを提供する」(a)・「Attribution Notice を目立つように掲示する」
    (b)・「改変した旨を Notice に含める」(d)(i)・「公式製品と誤認させない」(d)(iii) の 4 つで、
    どれも**配布リポ 1 つ**に掛かる（モデルの資産ではない）ので直下の 2 ファイルが受け持つ。
    """

    #: 上流から取得した原文の sha256（2026-08-20 実測）。ここを固定するのは、整形や改行変換で
    #: 1 バイトでも動けば「このライセンスのコピー」でなくなるため（§3(a)）。
    LICENSE_SHA256 = "ee956174133d7c2cdcf220440c7726187eaf4b50e8e48ee32194353a22164d15"

    def _run(self, tmp_path: Path) -> Path:
        _build_series(tmp_path / "series")
        out_dir = tmp_path / "dist"
        main(["--series", str(tmp_path / "series"), "--out", str(out_dir)])
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
        assert notice == NOTICE_MARKDOWN
        for sentence in ATTRIBUTION_NOTICE.split("\n"):
            assert sentence in notice

    def test_the_notice_states_the_modifications_with_the_lora_source(self, tmp_path: Path) -> None:
        """§3(d)(i) — 改変した旨を **Attribution Notice の中に**含める（出所つき）。"""
        text = (self._run(tmp_path) / "NOTICE.md").read_text(encoding="utf-8")
        # 改変記載は独立節ではなく Attribution Notice 節の内側（次見出しの前）に居ること。
        notice_section = text.split("## Not an official product")[0]
        assert "## Attribution Notice" in notice_section
        flat = self._flat(notice_section)
        assert flat.count("this Attribution Notice also states that") == 1
        assert flat.count("modified as follows:") == 1
        assert f"The official {LORA_NAME} ({LORA_SOURCE}) was baked into the weights" in text
        assert "https://civitai.com/models/2560840" in text

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
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_MODEL_NAME))
        assert (out_dir / "LICENSE.md").is_file()
        assert (out_dir / "NOTICE.md").is_file()

    def test_the_pipeline_declares_exactly_the_two_legal_seats(self) -> None:
        assert sorted(PIPELINE.root_files) == ["LICENSE.md", "NOTICE.md"]


class TestPipelineEntry:
    """ドライバへ差す 1 行（{@link PIPELINE}）— 名前・リポ名・帰属の 3 点。"""

    def test_the_model_name_is_a_legal_path_segment(self) -> None:
        """モデル名は manifest のキーであると同時にリポ内のディレクトリ名（ADR 0041 §6 / §9）。"""
        assert assert_model_name(ANIMA_MODEL_NAME) == ANIMA_MODEL_NAME

    def test_it_names_the_default_model_and_its_repository(self) -> None:
        assert PIPELINE.default_model == ANIMA_MODEL_NAME
        assert PIPELINE.repo_name(ANIMA_MODEL_NAME) == f"karume-{ANIMA_MODEL_NAME}"

    def test_one_attribution_needs_no_choice(self) -> None:
        """帰属は 1 通りしかない（選びようがないものを聞かない）。"""
        assert len(PIPELINE.card_profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is PIPELINE.card_profiles["anima"]

    def test_its_card_refuses_another_pipelines_manifest(self) -> None:
        manifest = {"models": {"m": {"pipeline": "anima/0"}}}
        with pytest.raises(ValueError):
            PIPELINE.card_profiles["anima"](manifest, "hdae/x")
