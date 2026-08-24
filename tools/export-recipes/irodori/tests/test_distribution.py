"""Irodori の配布 recipe（`irodori.distribution`）— 組み立て 1 周ぶんの単体テスト。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合**と IR メタデータ
（`__metadata__`）だけなので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。
Irodori は加えてチェックポイントの `config_json` とコーデックの `metadata.json` を読むが、
どちらも合成 JSON で足りる（**合成 config は実重み v4-small と全ての数が違う** — 数を焼き
込んでいれば落ちる）。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

core だけで観測できる層（合成計画で足りる規模上限・quant 完全写像・staging/swap の不変条件・
帰属プロファイルの解決規則）は `tools/exporter/tests/test_dist.py` が持つ（ADR 0065 段 3+4 の
分割）。カードの描画も**組み立て 1 周ぶん**でここが見る — Irodori のテンプレートは core の
`tests/test_modelcard.py` に固有の節を持っていなかったので、`test_card.py` は作らない。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from dist import default_out_dir, main
from irodori.distribution import (
    CALIB_PROVENANCE_FILE,
    CALIB_SHIPPABLE_METHOD,
    IRODORI_CODEC_DIRS,
    IRODORI_CODEC_HALO_FRAMES,
    IRODORI_CODEC_NAME,
    IRODORI_DEFAULT_MODEL,
    IRODORI_DTYPE_ROLES,
    IRODORI_GRAPH_ROLES,
    IRODORI_OUTPUT_PATHS,
    IRODORI_QUANT_ABBREVIATIONS,
    IRODORI_QUANT_SEATS,
    IRODORI_SAMPLING_DEFAULTS,
    IRODORI_SERIES_DIRS,
    IRODORI_STORAGE_FORBIDDEN,
    IRODORI_STORAGE_REQUIREMENTS,
    IRODORI_WEIGHT_DTYPES,
    IRODORI_WEIGHTS,
    PIPELINE,
    IrodoriSources,
    irodori_calib_floor,
    irodori_plan,
    irodori_repo_name,
    irodori_series_name,
    irodori_sources,
)
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    DistError,
    assemble_family,
    assert_quant_presentation,
    resolve_card_renderer,
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


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """モデルサブツリー内の期待 path（ADR 0041 §9 の一様レイアウト）。"""
    return [f"{model}/{rel}" for rel in paths]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


#: 合成チェックポイント config。**実重み v4-small とは全ての数が違う**（latent 32 / speaker 768
#: / 参照 120s …）— pipelineConfig を焼き込んでいれば、この config で組んだ manifest が実重みの
#: 数を名乗って落ちる。`latent_patch_size` だけは 1（TS 側の `latentDim` が 2 つの役割を兼ねる
#: 唯一の成立条件で、値そのものが門になっている）。
_IRODORI_CONFIG: Mapping[str, Any] = {
    "latent_dim": 8,
    "latent_patch_size": 1,
    "speaker_patch_size": 2,
    "speaker_dim": 24,
    "text_dim": 12,
    "caption_dim": 16,
    "timestep_embed_dim": 6,
    "max_text_len": 10,
    "max_caption_len": 14,
    "ref_max_seconds": 8.0,
}

#: 上の config から導出される数（テストが式を写さないための 1 箇所）。
_IRODORI_SPEAKER_ROWS = 101  # int(8.0 × 25) // 2 + 1
_IRODORI_DIT_SYM_MAX = 750  # int(30.0 × 25) // 1
_IRODORI_MASK_TOTAL = 10 + _IRODORI_SPEAKER_ROWS + 14

#: backbone の hidden 幅（projector の入力 — pipelineConfig には現れない数）。
_IRODORI_HIDDEN = 32

#: 偽コーデックの `metadata.json`（`irodori/dacvae/convert.py` が書く形）。**実物とは違う数**
#: （48kHz / hop 1920 ではない）にして、`sampleRate` / `hopLength` を焼き込んでいれば落ちるように
#: する。`frameRate` 25 と噛み合う組み合わせを選ぶ（12,000 = 25 × 480）。
_IRODORI_CODEC_METADATA: Mapping[str, Any] = {
    "kwargs": {"sample_rate": 12_000, "encoder_rates": [2, 4, 60]}
}
_IRODORI_HOP_LENGTH = 480

#: i4 系列の校正条件の記録（`irodori.export._write_calib_provenance` が書く形）。
#:
#: 予算の 2 欄は**出荷済みの系列の現物と同じ値**（`outputs/series/irodori-v4-small-i4/` の
#: `calib_provenance.json` = 12 件 × 40 step）。門が引く下限との一致は
#: {@link TestIrodoriCalibProvenance.test_the_floor_is_the_condition_the_export_defaults_to} が
#: 正本から見る — ここに写した値が古びたら、そちらが落ちる。
_IRODORI_CALIB_PROVENANCE: Mapping[str, Any] = {
    "method": CALIB_SHIPPABLE_METHOD,
    "grid": "rtn",
    "group_size": 32,
    "cases": 12,
    "steps": 40,
}

#: `pipelineConfig` の欄名（TS 側 `packages/models/src/irodori/config.ts` の `ROOT_KEYS` の写し）。
#: **ロード側は未知キーも欠落も parse 時に落とす**ので、焼く側とロード側の欄名は完全一致が要る。
#: 写しをテストが持つのは、片方だけが動いたときに落ちる席がここしか無いため。
_IRODORI_CONFIG_KEYS = (
    "maxTextLen",
    "maxCaptionLen",
    "speakerRows",
    "ditSymMax",
    "frameRate",
    "sampleRate",
    "hopLength",
    "codecHaloFrames",
    "latentDim",
    "speakerPatchSize",
    "speakerDim",
    "textDim",
    "captionDim",
    "timestepEmbedDim",
    "steps",
    "initScale",
    "cfgMinT",
    "cfgMaxT",
    "cfgScales",
    "minSeconds",
    "maxSeconds",
    "speakerUncondMode",
    "cfgGuidanceMode",
)


def _irodori_graph(inputs: Sequence[tuple[str, list[Any]]], outputs: int, symbol: str = "T") -> str:
    """門が読む最小の IR メタデータ（入力の名前と形・出力名・記号次元）。"""
    return json.dumps(
        {
            "inputs": [{"name": name, "shape": shape} for name, shape in inputs],
            "outputs": [f"out_{index}" for index in range(outputs)],
            "symbols": [symbol],
        }
    )


def _irodori_graphs() -> dict[str, str]:
    """8 グラフの IR メタデータ（{@link _IRODORI_CONFIG} と噛み合う形）。"""
    latent_dim = _IRODORI_CONFIG["latent_dim"]
    speaker_dim = _IRODORI_CONFIG["speaker_dim"]
    text_dim = _IRODORI_CONFIG["text_dim"]
    caption_dim = _IRODORI_CONFIG["caption_dim"]
    return {
        "backbone": _irodori_graph([("input_ids", [1, "T"])], 1),
        "text_proj": _irodori_graph([("hidden", [1, "T", _IRODORI_HIDDEN])], 1),
        "caption_proj": _irodori_graph([("hidden", [1, "T", _IRODORI_HIDDEN])], 2),
        "speaker": _irodori_graph(
            [("latent", [1, "S", latent_dim * _IRODORI_CONFIG["speaker_patch_size"]])], 1, "S"
        ),
        "duration": _irodori_graph(
            [
                ("text_state", [1, "T", text_dim]),
                ("speaker_vec", [1, speaker_dim]),
                ("has_speaker", [1, 1]),
                ("caption_vec", [1, caption_dim]),
                ("has_caption", [1, 1]),
            ],
            1,
        ),
        "dit": _irodori_graph(
            [
                ("x_t", [1, "S", latent_dim]),
                ("t_embed", [1, _IRODORI_CONFIG["timestep_embed_dim"]]),
                ("mask", [1, 1, 1, f"S+{_IRODORI_MASK_TOTAL}"]),
                ("text_state", [1, _IRODORI_CONFIG["max_text_len"], text_dim]),
                ("speaker_state", [1, _IRODORI_SPEAKER_ROWS, speaker_dim]),
                ("caption_state", [1, _IRODORI_CONFIG["max_caption_len"], caption_dim]),
            ],
            1,
            "S",
        ),
        # コーデック 2 本（別系列・純畳み込み）。入力幅が latentDim / hopLength と噛み合う。
        "codec_decoder": _irodori_graph([("latent", [1, "S", latent_dim])], 1, "S"),
        "codec_encoder": _irodori_graph([("wav", [1, "T", _IRODORI_HOP_LENGTH])], 1),
    }


def _irodori_container(dtype: str, role: str, graph: str, extra: tuple[str, ...] = ()) -> bytes:
    """1 ターゲットぶんの偽コンテナ（格納 dtype の集合が系列と対応する形）。

    圧縮系列は**適格スロットだけ**が F16 / I8 / I4 になり、bias / norm / グラフ定数（i8 なら
    per-channel scale・i4 なら group scale も）は F32 のまま残る（実物の圧縮コンテナのヘッダは
    この 2 つが並ぶ）。この形にしておかないと、f32 席へ圧縮資産を挿し込む取り違えが
    「F32 が無い」で落ちてしまい、**圧縮 dtype の不在**を見る門（`assert_storage_absent`）が
    一度も試されない。

    `extra` は混成コンテナを作る口。**実物の i4 系列がこの形**（block 内の adaLN 以外が I4・
    adaLN と block 外が I8 — 聴感裁定 2026-08-23）で、{@link IRODORI_STORAGE_FORBIDDEN} の MUST
    が名指しする
    「I8 を含むという要求検査では i8 席と i4 系列を分けられない」失敗モードそのもの。
    """
    payload = f"{role}-{dtype}-weights".encode()
    dtypes = ("F32",) if dtype == "f32" else (dtype.upper(), "F32")
    dtypes = (*dtypes, *extra)
    header: dict[str, Any] = {"__metadata__": {IR_METADATA_KEY: graph}}
    for index, stored in enumerate(dtypes):
        start = index * len(payload)
        header[f"w{index}"] = {
            "dtype": stored,
            "shape": [len(payload)],
            "data_offsets": [start, start + len(payload)],
        }
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + payload * len(dtypes)


def _build_irodori_sources(
    root: Path,
    *,
    model: str = IRODORI_DEFAULT_MODEL,
    config: Mapping[str, Any] = _IRODORI_CONFIG,
    graphs: Mapping[str, str] | None = None,
    codec_metadata: Mapping[str, Any] | None = _IRODORI_CODEC_METADATA,
    calib_provenance: Mapping[str, Any] | None = _IRODORI_CALIB_PROVENANCE,
) -> IrodoriSources:
    """系列 + チェックポイントの置き場を偽資産で再現する（配布しないものの混入込み）。

    並びは `_shared.paths` の実レイアウト（`outputs/series/` と `inputs/`）に揃える — CLI 経路の
    テストが root を差し替えるだけで同じ木を指せる形。コーデックは**別系列・別入力素材**
    （`dacvae-32dim`）なので、Irodori 本体とは別の 2 ディレクトリへ置く。系列は格納 dtype
    ごとに 1 本ずつ（`IRODORI_DTYPE_ROLES` — i4 だけは `dit` 1 役なので系列も 1 ディレクトリ）。
    """
    series_root = root / "outputs" / "series"
    suffix = {dtype: "" if dtype == "f32" else f"-{dtype}" for dtype in IRODORI_WEIGHT_DTYPES}
    by_dtype = {
        dtype: series_root / f"{irodori_series_name(model)}{tail}" for dtype, tail in suffix.items()
    }
    codec_by_dtype = {
        dtype: series_root / f"{IRODORI_CODEC_NAME}{tail}"
        for dtype, tail in suffix.items()
        if not set(IRODORI_DTYPE_ROLES[dtype]).isdisjoint(IRODORI_CODEC_DIRS)
    }
    sources = IrodoriSources(
        model=root / "inputs" / "irodori" / model,
        codec_model=root / "inputs" / "irodori" / IRODORI_CODEC_NAME,
        series_by_dtype=by_dtype,
        codec_series_by_dtype=codec_by_dtype,
    )
    resolved = graphs or _irodori_graphs()
    for dtype, roles in IRODORI_DTYPE_ROLES.items():
        for role in roles:
            in_codec = role in IRODORI_CODEC_DIRS
            series = (
                sources.codec_series_by_dtype[dtype] if in_codec else sources.series_by_dtype[dtype]
            )
            directory = (IRODORI_SERIES_DIRS | IRODORI_CODEC_DIRS)[role]
            _write(
                series / directory / "model.safetensors",
                _irodori_container(dtype, role, resolved[role]),
            )
            # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
            _write(series / directory / "io.case0.safetensors", b"io-fixture")
    if calib_provenance is not None:
        # 校正条件の記録（`irodori.export._write_calib_provenance` が i4 の `dit` 直下へ書く）。
        _write(
            sources.series_by_dtype["i4"] / IRODORI_SERIES_DIRS["dit"] / CALIB_PROVENANCE_FILE,
            json.dumps(calib_provenance, ensure_ascii=False).encode("utf-8"),
        )
    if codec_metadata is not None:
        _write(
            sources.codec_model / "metadata.json",
            json.dumps(codec_metadata, ensure_ascii=False).encode("utf-8"),
        )
    _write(sources.series / "tokenizer" / "tokenizer.json", b'{"vocabText": "a"}')
    # tokenizer の golden 3 本は検証用（実行に要らないので配布形には入らない）。
    for name in ("golden.encode.json", "golden.normalize.json", "nfkc-diff.json"):
        _write(sources.series / "tokenizer" / name, b'{"golden": true}')
    _write(
        sources.model / "model.safetensors",
        _fake_safetensors(
            "F32", b"checkpoint", {"config_json": json.dumps(config, ensure_ascii=False)}
        ),
    )
    return sources


def _assemble_irodori(
    sources: IrodoriSources, out_dir: Path, model: str = IRODORI_DEFAULT_MODEL
) -> dict[str, Any]:
    return assemble_family([irodori_plan(sources, model)], out_dir, model)


@pytest.fixture
def irodori_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_irodori_sources(tmp_path)
    out_dir = tmp_path / "models" / irodori_repo_name(IRODORI_DEFAULT_MODEL)
    manifest = _assemble_irodori(sources, out_dir)
    return out_dir, manifest


def _irodori_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][IRODORI_DEFAULT_MODEL]


class TestIrodoriLayout:
    def test_it_places_every_declared_path_under_the_model_subtree(self, irodori_assembled) -> None:
        out_dir, _ = irodori_assembled
        expected = _in_subtree(IRODORI_DEFAULT_MODEL, IRODORI_OUTPUT_PATHS.values())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_io_fixtures_or_tokenizer_goldens(self, irodori_assembled) -> None:
        """配布へ入るのは実行に要る 7 本だけ（golden は検証用の資産）。"""
        out_dir, _ = irodori_assembled
        assert list(out_dir.rglob("io.*")) == []
        assert list(out_dir.rglob("golden.*")) == []
        assert list(out_dir.rglob("nfkc-diff.json")) == []

    def test_it_declares_the_eight_graphs_and_the_tokenizer(self, irodori_assembled) -> None:
        _, manifest = irodori_assembled
        model = _irodori_model(manifest)
        assert model["pipeline"] == "irodori/1"
        assert sorted(model["weights"]) == [
            "backbone",
            "caption_proj",
            "codec_decoder",
            "codec_encoder",
            "dit",
            "duration",
            "speaker",
            "text_proj",
        ]
        assert sorted(model["assets"]) == ["tokenizer"]

    def test_the_quants_are_the_five_seats_the_seat_table_spells(self, irodori_assembled) -> None:
        """席は 5 つ（格納 3 系列 + `-a8` の実行形ノブ + `dit` だけ i4 の `i8+dit4`）。
        既定は `i8-a8`。

        `i8` / `i8-a8` は**同じ i8 バイトを共有**し、違うのは `session` だけ — 席が増えても
        配布形のファイルは増えない、が席表の要点。`i8+dit4` はその例外で、`dit` の 1 本だけが
        i4 系列を指す。
        """
        _, manifest = irodori_assembled
        model = _irodori_model(manifest)
        assert model["defaultQuant"] == "i8-a8"
        assert list(model["quants"]) == ["f32", "f16", "i8", "i8-a8", "i8+dit4"]
        for name, seat in IRODORI_QUANT_SEATS.items():
            assert model["quants"][name]["session"] == seat.session
            assert "gpuFeatures" not in model["quants"][name]
            # 完全写像（hub の受理要件）— 8 役全部が dtype ラベルを 1 つずつ指す。
            assert model["quants"][name]["weights"] == {
                role: seat.roles.get(role, seat.dtype) for role in model["weights"]
            }
        assert model["quants"]["i8"]["session"] == {}
        assert model["quants"]["i8-a8"]["session"] == {"linearCompute": "a8"}
        # MUST: `i8+dit4` は `linearCompute` を宣言しない（irodori では w4a8 経路が未測定 —
        # 席表の同 MUST）。ここが緩むと「速いが荒い」構成が int4 の席名のまま出る。
        assert model["quants"]["i8+dit4"]["session"] == {}
        assert model["quants"]["i8+dit4"]["weights"]["dit"] == "i4"

    def test_every_quant_points_at_its_own_storage_series(self, irodori_assembled) -> None:
        """席と現物の対応（圧縮席のファイルが実際に F16 / I8 / I4 格納であることは組み立て門が
        見るが、**宣言の側も**系列を跨がないことをここで固定する — 片方だけ動くと配布形の中で
        「f16 と名乗る f32」が並ぶ）。"""
        out_dir, manifest = irodori_assembled
        model = _irodori_model(manifest)
        for name, seat in IRODORI_QUANT_SEATS.items():
            for role, label in model["quants"][name]["weights"].items():
                dtype = seat.roles.get(role, seat.dtype)
                path = model["weights"][role][label]["shards"][0]["path"]
                assert path.endswith(f"model.{dtype}.safetensors"), (role, name, path)
                assert (out_dir / path).is_file()

    def test_the_two_i8_seats_share_one_set_of_bytes(self, irodori_assembled) -> None:
        """`i8-a8` は席を 1 行足すだけ（ADR 0050 波 2 — 配布サイズは 1 バイトも増えない）。"""
        _, manifest = irodori_assembled
        model = _irodori_model(manifest)
        assert model["quants"]["i8"]["weights"] == model["quants"]["i8-a8"]["weights"]

    def test_the_dit4_seat_shares_the_i8_bytes_for_the_other_seven_roles(
        self, irodori_assembled
    ) -> None:
        """`i8+dit4` が新しく足すファイルは `dit` の 1 本だけ（他 7 役は `i8` とバイト共有）。"""
        _, manifest = irodori_assembled
        model = _irodori_model(manifest)
        differing = {
            role
            for role, label in model["quants"]["i8+dit4"]["weights"].items()
            if label != model["quants"]["i8"]["weights"][role]
        }

        assert differing == {"dit"}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        out_dir = tmp_path / "models" / irodori_repo_name(IRODORI_DEFAULT_MODEL)
        assert _assemble_irodori(sources, out_dir) == _assemble_irodori(sources, out_dir)
        assert verify_dist(out_dir)


class TestIrodoriPipelineConfig:
    """`pipelineConfig` はロード側（`src/irodori/config.ts`）のスキーマと欄名まで一致する。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, irodori_assembled) -> None:
        _, manifest = irodori_assembled
        assert tuple(_irodori_model(manifest)["pipelineConfig"]) == _IRODORI_CONFIG_KEYS

    def test_it_derives_the_model_specific_numbers_from_the_checkpoint(
        self, irodori_assembled
    ) -> None:
        """焼き込んでいれば実重み（latent 32 / speaker 768 / 参照 120s）の数が出てくる。"""
        _, manifest = irodori_assembled
        config = _irodori_model(manifest)["pipelineConfig"]
        assert config["maxTextLen"] == _IRODORI_CONFIG["max_text_len"]
        assert config["maxCaptionLen"] == _IRODORI_CONFIG["max_caption_len"]
        assert config["latentDim"] == _IRODORI_CONFIG["latent_dim"]
        assert config["speakerPatchSize"] == _IRODORI_CONFIG["speaker_patch_size"]
        assert config["speakerDim"] == _IRODORI_CONFIG["speaker_dim"]
        assert config["textDim"] == _IRODORI_CONFIG["text_dim"]
        assert config["captionDim"] == _IRODORI_CONFIG["caption_dim"]
        assert config["timestepEmbedDim"] == _IRODORI_CONFIG["timestep_embed_dim"]
        # 参照 latent の patch 後の上限 + 平均トークン 1 本 / 30s × 25Hz ÷ latent patch。
        assert config["speakerRows"] == _IRODORI_SPEAKER_ROWS
        assert config["ditSymMax"] == _IRODORI_DIT_SYM_MAX

    def test_it_derives_the_codec_numbers_from_the_codec_metadata(self, irodori_assembled) -> None:
        """焼き込んでいれば実物（48kHz / hop 1920）の数が出てくる。halo だけが直書き。"""
        _, manifest = irodori_assembled
        config = _irodori_model(manifest)["pipelineConfig"]
        assert config["sampleRate"] == _IRODORI_CODEC_METADATA["kwargs"]["sample_rate"]
        # hop_length = prod(encoder_rates)（`DACVAE.__init__` の綴り）。
        assert config["hopLength"] == _IRODORI_HOP_LENGTH
        assert config["codecHaloFrames"] == IRODORI_CODEC_HALO_FRAMES
        # ロード側は 3 者の整合（sampleRate == frameRate × hopLength）を parse 時に見る。
        assert config["sampleRate"] == config["frameRate"] * config["hopLength"]

    def test_it_refuses_a_codec_whose_frame_rate_does_not_match(self, tmp_path: Path) -> None:
        """秒 → フレームと 秒 → サンプル → フレームの 2 系統が独立に動く形を作らない。"""
        sources = _build_irodori_sources(
            tmp_path, codec_metadata={"kwargs": {"sample_rate": 12_000, "encoder_rates": [500]}}
        )
        with pytest.raises(DistError, match="hop_length"):
            irodori_plan(sources)

    def test_it_refuses_a_missing_codec_metadata(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path, codec_metadata=None)
        with pytest.raises(DistError, match=r"metadata\.json"):
            irodori_plan(sources)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sample_rate": 0, "encoder_rates": [2, 4, 60]},
            {"sample_rate": 12_000, "encoder_rates": []},
            {"sample_rate": 12_000, "encoder_rates": [2, "4", 60]},
        ],
    )
    def test_it_refuses_codec_numbers_that_are_not_positive_integers(
        self, tmp_path: Path, kwargs: Mapping[str, Any]
    ) -> None:
        sources = _build_irodori_sources(tmp_path, codec_metadata={"kwargs": kwargs})
        with pytest.raises(DistError):
            irodori_plan(sources)

    def test_it_carries_the_sampler_defaults_the_upstream_declares(self, irodori_assembled) -> None:
        _, manifest = irodori_assembled
        config = _irodori_model(manifest)["pipelineConfig"]
        for key, value in IRODORI_SAMPLING_DEFAULTS.items():
            assert config[key] == value
        # ADR 0047 決定 1 — ロード側はこの 2 値以外を parse 時に拒否する。
        assert config["speakerUncondMode"] == "mask"
        assert config["cfgGuidanceMode"] == "independent"

    def test_it_refuses_a_checkpoint_without_the_config_metadata(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        _write(sources.model / "model.safetensors", _fake_safetensors("F32", b"checkpoint"))
        with pytest.raises(DistError, match="config_json"):
            irodori_plan(sources)

    def test_it_refuses_a_checkpoint_whose_config_is_not_json(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.model / "model.safetensors",
            _fake_safetensors("F32", b"checkpoint", {"config_json": "{"}),
        )
        with pytest.raises(DistError, match="JSON として読めない"):
            irodori_plan(sources)

    @pytest.mark.parametrize("value", [0, "32", True, None])
    def test_it_refuses_a_dimension_that_is_not_a_positive_integer(
        self, tmp_path: Path, value: Any
    ) -> None:
        sources = _build_irodori_sources(tmp_path, config={**_IRODORI_CONFIG, "latent_dim": value})
        with pytest.raises(DistError, match="latent_dim"):
            irodori_plan(sources)

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf"), -1.0, 0.0, "8.0", True, None]
    )
    def test_it_refuses_a_reference_length_that_is_not_a_finite_positive_number(
        self, tmp_path: Path, value: Any
    ) -> None:
        """NaN は比較が全て False で `<= 0` を素通りする — 下流の秒 → フレーム換算まで運ばない。"""
        sources = _build_irodori_sources(
            tmp_path, config={**_IRODORI_CONFIG, "ref_max_seconds": value}
        )
        with pytest.raises(DistError, match="ref_max_seconds"):
            irodori_plan(sources)

    def test_it_refuses_a_latent_patch_size_the_loader_schema_cannot_express(
        self, tmp_path: Path
    ) -> None:
        """`latentDim` は x_t の幅と参照 latent の 1 フレーム幅を兼ねる（1 でしか両立しない）。"""
        sources = _build_irodori_sources(
            tmp_path, config={**_IRODORI_CONFIG, "latent_patch_size": 2}
        )
        with pytest.raises(DistError, match="latent_patch_size"):
            irodori_plan(sources)


class TestIrodoriGraphGate:
    """組み立て門 — ずれても shape が合ったまま通る組み合わせを、配置の**前**に落とす。"""

    def _sources(self, tmp_path: Path, role: str, graph: str) -> IrodoriSources:
        return _build_irodori_sources(tmp_path, graphs={**_irodori_graphs(), role: graph})

    def test_it_refuses_a_caption_projector_with_a_single_output(self, tmp_path: Path) -> None:
        """第 2 出力（`caption_norm` 済み系列）が無いと `caption_vec` が別のベクトルになる。"""
        sources = self._sources(
            tmp_path,
            "caption_proj",
            _irodori_graph([("hidden", [1, "T", _IRODORI_HIDDEN])], 1),
        )
        with pytest.raises(DistError, match="グラフ出力が 1 本"):
            irodori_plan(sources)

    def test_it_refuses_a_dit_that_lost_an_input(self, tmp_path: Path) -> None:
        graphs = _irodori_graphs()
        trimmed = json.loads(graphs["dit"])
        trimmed["inputs"] = trimmed["inputs"][:5]
        sources = self._sources(tmp_path, "dit", json.dumps(trimmed))
        with pytest.raises(DistError, match="グラフ入力"):
            irodori_plan(sources)

    def test_it_refuses_a_dit_whose_inputs_are_reordered(self, tmp_path: Path) -> None:
        graphs = _irodori_graphs()
        swapped = json.loads(graphs["dit"])
        swapped["inputs"][3], swapped["inputs"][5] = swapped["inputs"][5], swapped["inputs"][3]
        sources = self._sources(tmp_path, "dit", json.dumps(swapped))
        with pytest.raises(DistError, match="グラフ入力"):
            irodori_plan(sources)

    def test_it_refuses_a_graph_that_declares_another_conditioning_length(
        self, tmp_path: Path
    ) -> None:
        """条件 state の宣言長がずれても右 pad は通る（別の位置の条件を読んで沈黙する）。"""
        graphs = _irodori_graphs()
        stretched = json.loads(graphs["dit"])
        stretched["inputs"][3]["shape"][1] = _IRODORI_CONFIG["max_text_len"] + 1
        sources = self._sources(tmp_path, "dit", json.dumps(stretched))
        with pytest.raises(DistError, match="maxTextLen"):
            irodori_plan(sources)

    def test_it_refuses_a_speaker_encoder_with_another_patch_width(self, tmp_path: Path) -> None:
        sources = self._sources(
            tmp_path, "speaker", _irodori_graph([("latent", [1, "S", 999])], 1, "S")
        )
        with pytest.raises(DistError, match="speakerPatchSize"):
            irodori_plan(sources)

    def test_it_refuses_a_mask_whose_segments_do_not_add_up(self, tmp_path: Path) -> None:
        """区間の合計がずれると、マスクの区間割りだけが黙って別の位置を指す。"""
        graphs = _irodori_graphs()
        bent = json.loads(graphs["dit"])
        bent["inputs"][2]["shape"][3] = f"S+{_IRODORI_MASK_TOTAL + 1}"
        sources = self._sources(tmp_path, "dit", json.dumps(bent))
        with pytest.raises(DistError, match="mask"):
            irodori_plan(sources)

    def test_it_refuses_a_codec_decoder_for_another_latent_width(self, tmp_path: Path) -> None:
        """別次元の DACVAE を混ぜると shape は合ったまま別の声になる。"""
        sources = self._sources(
            tmp_path, "codec_decoder", _irodori_graph([("latent", [1, "S", 999])], 1, "S")
        )
        with pytest.raises(DistError, match="latentDim"):
            irodori_plan(sources)

    def test_it_refuses_a_codec_encoder_with_another_hop(self, tmp_path: Path) -> None:
        """入力幅 = hopLength がずれると、波形のフレーム分割だけが黙って別の格子になる。"""
        sources = self._sources(
            tmp_path, "codec_encoder", _irodori_graph([("wav", [1, "T", 999])], 1)
        )
        with pytest.raises(DistError, match="hopLength"):
            irodori_plan(sources)

    def test_it_refuses_a_codec_decoder_that_lost_its_input_name(self, tmp_path: Path) -> None:
        sources = self._sources(
            tmp_path, "codec_decoder", _irodori_graph([("z", [1, "S", 8])], 1, "S")
        )
        with pytest.raises(DistError, match="グラフ入力"):
            irodori_plan(sources)

    def test_it_refuses_a_codec_decoder_with_two_outputs(self, tmp_path: Path) -> None:
        """検証用の別ターゲット（中間値つき）が紛れ込んでいないことの証跡。"""
        sources = self._sources(
            tmp_path, "codec_decoder", _irodori_graph([("latent", [1, "S", 8])], 2, "S")
        )
        with pytest.raises(DistError, match="グラフ出力が 2 本"):
            irodori_plan(sources)

    def test_it_refuses_a_container_without_ir_metadata(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series / "dit" / "model.safetensors", _fake_safetensors("F32", b"dit-weights")
        )
        with pytest.raises(DistError, match="IR メタデータ"):
            irodori_plan(sources)

    def test_it_refuses_an_f32_seat_whose_container_has_no_f32(self, tmp_path: Path) -> None:
        """格納形は series 名でなくヘッダが正（要求 dtype の存在検査）。"""
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series / "dit" / "model.safetensors",
            _fake_safetensors("F16", b"dit-weights", {IR_METADATA_KEY: _irodori_graphs()["dit"]}),
        )
        with pytest.raises(DistError, match="F32"):
            irodori_plan(sources)


class TestIrodoriStorageSeries:
    """系列 × 格納 dtype の**集合等値** — 系列 root の取り違えを掴む唯一の検出器。

    MUST: 数値の門（E2E の tolerance）では原理的に検出できない（ADR 0027 / 0029 の検出限界 —
    f32 系列と f16 系列は実測が同桁なので互いの閾値を素通りする）。存在検査と不在検査を
    両側から掛けて初めて、どちらの席にどちらの資産が来ても落ちる。
    """

    def test_it_refuses_an_f16_series_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """圧縮コンテナも適格外の重みを F32 で持つので、**存在検査だけでは素通りする**形。"""
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series / "dit" / "model.safetensors",
            _irodori_container("f16", "dit", _irodori_graphs()["dit"]),
        )

        with pytest.raises(DistError, match="F16 がある"):
            irodori_plan(sources)

    def test_it_refuses_an_f32_series_asset_in_the_f16_seat(self, tmp_path: Path) -> None:
        """逆向き（丸め忘れ = `--dtype f16` のつもりが素の f32）は F16 の不在で落ちる。"""
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series_by_dtype["f16"] / "dit" / "model.safetensors",
            _irodori_container("f32", "dit", _irodori_graphs()["dit"]),
        )

        with pytest.raises(DistError, match="F16 が無い"):
            irodori_plan(sources)

    def test_it_refuses_an_i8_series_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """i8 資産は F32（bias / norm / per-channel scale）を持つので、**存在検査は真になる**。

        MUST: 禁止表を「f32 席は F16 だけ禁止」のまま i8 系列を足すと、この取り違えが素通りする
        （波 1 で f16 について同じ穴を塞いだのと同じ機序 — 禁止は集合で持つ）。
        """
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series / "dit" / "model.safetensors",
            _irodori_container("i8", "dit", _irodori_graphs()["dit"]),
        )

        with pytest.raises(DistError, match="I8 がある"):
            irodori_plan(sources)

    def test_it_refuses_an_f32_series_asset_in_the_i8_seat(self, tmp_path: Path) -> None:
        """逆向き（丸め忘れ = `--dtype i8` のつもりが素の f32）は I8 の不在で落ちる。"""
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series_by_dtype["i8"] / "dit" / "model.safetensors",
            _irodori_container("f32", "dit", _irodori_graphs()["dit"]),
        )

        with pytest.raises(DistError, match="I8 が無い"):
            irodori_plan(sources)

    def test_it_refuses_an_i8_series_asset_in_the_f16_seat(self, tmp_path: Path) -> None:
        """圧縮系列どうしの取り違えは、**要求 dtype の不在**が落とす（禁止表は要らない）。"""
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series_by_dtype["f16"] / "dit" / "model.safetensors",
            _irodori_container("i8", "dit", _irodori_graphs()["dit"]),
        )

        with pytest.raises(DistError, match="F16 が無い"):
            irodori_plan(sources)

    def test_it_refuses_an_f16_series_asset_in_the_i8_seat(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series_by_dtype["i8"] / "dit" / "model.safetensors",
            _irodori_container("f16", "dit", _irodori_graphs()["dit"]),
        )

        with pytest.raises(DistError, match="I8 が無い"):
            irodori_plan(sources)

    def test_it_refuses_an_i4_series_asset_in_the_i8_seat(self, tmp_path: Path) -> None:
        """MUST: i8 席は **I4 の不在**で締める（要求検査だけでは分けられない）。

        写すのは**実物の i4 系列**（block 内の adaLN 以外が I4・adaLN と block 外が I8）—
        そのコンテナは
        「I8 を含む」を満たすので、禁止表が無いと i8 席へ挿し込めてしまう。実害は既定席
        `i8-a8` に出る（宣言した `linearCompute: "a8"` の述語は i4 常駐も受けるので、
        ADR 0076 の w4a8 経路が int8 を名乗る席のまま黙って走る）。
        """
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series_by_dtype["i8"] / "dit" / "model.safetensors",
            _irodori_container("i4", "dit", _irodori_graphs()["dit"], extra=("I8",)),
        )

        with pytest.raises(DistError, match="I4 がある"):
            irodori_plan(sources)

    def test_it_refuses_an_i8_series_asset_in_the_i4_seat(self, tmp_path: Path) -> None:
        """逆向き（`--dtype i4` のつもりが i8 系列）は I4 の不在で落ちる。"""
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series_by_dtype["i4"] / "dit" / "model.safetensors",
            _irodori_container("i8", "dit", _irodori_graphs()["dit"]),
        )

        with pytest.raises(DistError, match="I4 が無い"):
            irodori_plan(sources)

    def test_it_refuses_an_i4_series_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """i4 資産も F32（bias / norm / group scale）を持つので、**存在検査は真になる**。"""
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series / "dit" / "model.safetensors",
            _irodori_container("i4", "dit", _irodori_graphs()["dit"]),
        )

        with pytest.raises(DistError, match="I4 がある"):
            irodori_plan(sources)

    def test_it_refuses_a_codec_series_mixup_too(self, tmp_path: Path) -> None:
        """コーデックは別系列（`dacvae-32dim{,-f16,-i8}`）— 同じ門が 8 役全部に掛かる。"""
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.codec_series_by_dtype["f16"] / "decoder" / "model.safetensors",
            _irodori_container("f32", "codec_decoder", _irodori_graphs()["codec_decoder"]),
        )

        with pytest.raises(DistError, match="F16 が無い"):
            irodori_plan(sources)

    def test_every_graph_role_carries_every_seat_its_series_declares(self) -> None:
        """要求表 / 禁止表 / 宣言が同じ席（8 役 × f32/f16/i8 + `dit` の i4）を指す。

        片方だけ席が増えると、増えたほうが黙って無検査のまま配布形に並ぶ。
        """
        expected = {
            f"{role}_{dtype}" for dtype, roles in IRODORI_DTYPE_ROLES.items() for role in roles
        }

        assert set(IRODORI_STORAGE_REQUIREMENTS) == expected
        assert set(IRODORI_STORAGE_FORBIDDEN) == {
            *(f"{role}_f32" for role in IRODORI_GRAPH_ROLES),
            "dit_i8",
        }
        # 禁止は**圧縮系列ぶん全部**（1 つでも抜けると、抜けたほうの資産が f32 席を素通りする）。
        assert set(IRODORI_STORAGE_FORBIDDEN[f"{IRODORI_GRAPH_ROLES[0]}_f32"]) == {
            "F16",
            "I8",
            "I4",
        }
        # i8 席は I4 の不在で締める（i4 系列も I8 を含むので、要求検査だけでは塞がらない）。
        assert set(IRODORI_STORAGE_FORBIDDEN["dit_i8"]) == {"I4"}
        assert {
            files.file for labels in IRODORI_WEIGHTS.values() for files in labels.values()
        } == expected

    def test_every_seat_starts_from_a_storage_dtype_of_the_vocabulary(self) -> None:
        """席名は ADR 0074 の文法 — `<格納>` は資産ヘッダの語彙（決定 2）。`w8` / `w4` は廃した。"""
        for name, seat in IRODORI_QUANT_SEATS.items():
            assert name.split("+")[0].split("-")[0] == seat.dtype, name

    def test_it_needs_no_abbreviation_legend(self) -> None:
        """`i8+dit4` のトークン `dit` は weights 名そのもの（略称ではない — ADR 0074 決定 4）。

        対応表を空でないものにすると「`dit` は `dit` です」の行がカードに生える。
        """
        assert IRODORI_QUANT_ABBREVIATIONS == {}

    def test_every_seat_carries_a_label_and_a_description_within_the_limits(self) -> None:
        for name, seat in IRODORI_QUANT_SEATS.items():
            assert seat.label and seat.description, name
            assert_quant_presentation(
                f"irodori.quants.{name}",
                {"label": seat.label, "description": seat.description},
            )

    def test_every_quant_seat_names_a_storage_series_that_exists(self) -> None:
        """席表（`IRODORI_QUANT_SEATS`）が指す dtype は必ず系列として焼かれている側にある。

        席名（`i8`）と系列 root（`-i8`）の対応を綴る箇所はここ 1 つきり — 2 箇所に分かれると、
        片方だけ動いたときに「存在しない系列を指す席」か「誰も指さない系列」が生える。
        MUST: 役割ごとの例外（`i8+dit4` の `dit`）も同じ検査に載せる。
        """
        named = {
            seat.roles.get(role, seat.dtype)
            for seat in IRODORI_QUANT_SEATS.values()
            for role in IRODORI_GRAPH_ROLES
        }

        assert named == set(IRODORI_WEIGHT_DTYPES)
        for seat in IRODORI_QUANT_SEATS.values():
            for role, dtype in seat.roles.items():
                assert role in IRODORI_DTYPE_ROLES[dtype], (role, dtype)

    def test_the_series_roots_match_the_exporter_spelling(self, tmp_path: Path) -> None:
        """系列 root の綴りは書き手（`export_*.default_out_root`）と 1 文字も違わない。"""
        import irodori.dacvae.export
        import irodori.export

        sources = irodori_sources(tmp_path)
        for dtype in IRODORI_WEIGHT_DTYPES:
            assert (
                sources.series_by_dtype[dtype].name
                == irodori.export.default_out_root(Path(IRODORI_DEFAULT_MODEL), dtype).name
            )
        for dtype in sources.codec_series_by_dtype:
            assert (
                sources.codec_series_by_dtype[dtype].name
                == irodori.dacvae.export.default_out_root(Path(IRODORI_CODEC_NAME), dtype).name
            )
        # コーデックは i4 系列を持たない（`IRODORI_DTYPE_ROLES` の i4 は `dit` だけ）。
        assert set(sources.codec_series_by_dtype) == {"f32", "f16", "i8"}


class TestIrodoriCalibProvenance:
    """i4 系列の丸め条件の突合 — **資産からは判別できない**唯一の事実を機械で見る席。

    MUST: 校正の方式も予算も格納形を 1 バイトも変えないので、要求 dtype 検査もグラフ検査も
    `verify_dist` も素通りする。`--no-calib` も `--calib-steps 1` も生成物が配布へ紛れれば
    出るのは音の劣化だけ。
    """

    def test_a_calibrated_series_passes(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)

        assert irodori_plan(sources).quants["i8+dit4"]["weights"]["dit"] == "i4"

    def test_a_missing_record_is_refused(self, tmp_path: Path) -> None:
        """記録の不在は「古い export」— 名指しで再エクスポートを促す。"""
        sources = _build_irodori_sources(tmp_path, calib_provenance=None)

        with pytest.raises(DistError, match="校正条件の記録が無い"):
            irodori_plan(sources)

    def test_an_uncalibrated_series_is_refused(self, tmp_path: Path) -> None:
        """`--no-calib` は `rtn` と記録される — 組み立てはそれを名指しで拒否する。"""
        sources = _build_irodori_sources(
            tmp_path,
            calib_provenance={**_IRODORI_CALIB_PROVENANCE, "method": "rtn", "cases": 0, "steps": 0},
        )

        with pytest.raises(DistError, match="配布して良い丸め方式で作られていない"):
            irodori_plan(sources)

    def test_an_unparsable_record_is_refused(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series_by_dtype["i4"] / IRODORI_SERIES_DIRS["dit"] / CALIB_PROVENANCE_FILE,
            b"{ not json",
        )

        with pytest.raises(DistError, match="解析できない"):
            irodori_plan(sources)

    def test_a_smoke_budget_is_refused(self, tmp_path: Path) -> None:
        """`--calib-steps 1` は `method` を `gptq` のまま残す — 予算の欄まで見ないと通る。

        格納形は 1 バイトも変わらない（格子は RTN i4 g32 のまま）ので、ヘッダ検査も
        `verify_dist` もこの資産を「正しい w4 席」と読む。
        """
        sources = _build_irodori_sources(
            tmp_path, calib_provenance={**_IRODORI_CALIB_PROVENANCE, "steps": 1}
        )

        with pytest.raises(DistError, match="校正予算 'steps' が配布の下限を下回る"):
            irodori_plan(sources)

    def test_a_smaller_corpus_is_refused(self, tmp_path: Path) -> None:
        """コーパスを削って焼いた系列も同じ席で落ちる（4 件 → 12 件は聴感裁定 2026-08-23）。"""
        sources = _build_irodori_sources(
            tmp_path, calib_provenance={**_IRODORI_CALIB_PROVENANCE, "cases": 4}
        )

        with pytest.raises(DistError, match="校正予算 'cases' が配布の下限を下回る"):
            irodori_plan(sources)

    def test_a_record_written_before_the_budget_fields_is_still_accepted(
        self, tmp_path: Path
    ) -> None:
        """後方互換 MUST: 欄の**不在**は受理する（記録の作り直し = 丸め時間ぶんの再 export）。"""
        legacy = {
            key: value
            for key, value in _IRODORI_CALIB_PROVENANCE.items()
            if key not in {"cases", "steps"}
        }
        sources = _build_irodori_sources(tmp_path, calib_provenance=legacy)

        assert irodori_plan(sources).quants["i8+dit4"]["weights"]["dit"] == "i4"

    def test_the_floor_is_the_condition_the_export_defaults_to(self) -> None:
        """下限は写しではなく正本から引く（コーパスの本数と参照ループ全長）。"""
        from irodori.calib_cases import CALIB_CASES
        from irodori.pipeline_ref import NUM_STEPS

        assert irodori_calib_floor() == {"cases": len(CALIB_CASES), "steps": NUM_STEPS}
        # 出荷済みの系列（12 件 × 40 step）はこの下限をちょうど満たす形で焼かれている。
        assert _IRODORI_CALIB_PROVENANCE["cases"] == len(CALIB_CASES)
        assert _IRODORI_CALIB_PROVENANCE["steps"] == NUM_STEPS


class TestIrodoriModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """偽資産だけで CLI を 1 周回す（チェックポイントの置き場も tmp へ寄せる）。"""
        from irodori import distribution

        sources = _build_irodori_sources(tmp_path)
        # `INPUTS_ROOT`（系列の外にあるチェックポイント）を引くのは recipe 側 — ドライバの
        # `dist` ではなくこちらの束縛を外す（`DIST_ROOT` は逆でドライバ側）。
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            ["--pipeline", "irodori", "--series", str(sources.series.parent), "--out", str(out_dir)]
        )
        return out_dir

    def test_it_describes_the_text_to_audio_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: text-to-speech" in card
        assert "license: mit" in card
        # 既定 quant（`i8-a8`）の DL 実体が int8 系列なので `quantized` を宣言する（sbv2 / anima と
        # 同型。旧「f32 のまま・宣言しない」は i8 系列同梱前の陳腐化した前提だった）。
        assert "base_model_relation: quantized" in card
        assert "Text in, waveform out" in card
        # 同梱したコーデックは帰属にも `base_model` にも並ぶ（再配布しているため）。
        assert "Semantic-DACVAE-Japanese-32dim" in card
        # 参照話者は「音声」でも「latent」でも渡せる（voice cloning は配線済み）。
        assert "Voice cloning is wired up both ways" in card
        # 周波数は配布形と一致必須（リサンプルを持たない = 不一致は fail loudly）。この数は
        # コーデックの metadata から manifest 経由で降りてくる。
        rate = _IRODORI_CODEC_METADATA["kwargs"]["sample_rate"]
        assert f"distribution's own {rate} Hz" in card
        assert "there is no resampler" in card
        # 非タイルの encoder は長尺参照で落ちうる（limitations 起票済みの by-design 制約）。
        assert "`codec_encoder` is not tiled" in card
        assert '  repo: "hdae/dist",' in card
        # revision は object ref 形の中にコメントアウトで置く（外すだけで pin できる）。
        assert '  // revision: "<full commit sha>",' in card
        # Usage は「コメントを外すだけで次の一歩へ進める」形（裁定 2026-08-12）: voice cloning の
        # 両形（audio / latent）と optional ノブがコメントアウトで併記され、選べる値は manifest
        # から機械導出される（席が増えれば列挙も既定も追従する — ここでは 5 席・既定 `i8-a8`）。
        assert '// speaker: { audio: decodeWav(await Deno.readFile("reference.wav")) },' in card
        assert "// speaker: { latent: savedLatent }," in card
        assert '// quant: "i8-a8", // default — available: f16 / f32 / i8 / i8+dit4 / i8-a8' in card
        assert "// durationSeconds: 5," in card

    def test_it_derives_the_shape_section_from_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = self._run(tmp_path, monkeypatch)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert f"up to {_IRODORI_CONFIG['max_text_len']} tokens" in card
        assert f"up to {_IRODORI_DIT_SYM_MAX} frames" in card
        # カードは**検証を通った**配布形から描かれる（表と現物が食い違ったまま説明が生えない）。
        assert verify_dist(out_dir)


class TestIrodoriCli:
    @staticmethod
    def _reroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import dist
        from irodori import distribution

        # `DIST_ROOT` は既定の出力先を決めるドライバ側（`dist.default_out_dir`）、`INPUTS_ROOT` は
        # 系列の外の入力を引く recipe 側 — 別モジュールの束縛を別々に外す。
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert default_out_dir(PIPELINE, [IRODORI_DEFAULT_MODEL]).name == "karume-irodori-v4-small"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        """上流 1 リポの重みを移しただけなので帰属は 1 通り（選びようがない）。"""
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sources = _build_irodori_sources(tmp_path)
        self._reroot(tmp_path, monkeypatch)

        main(["--pipeline", "irodori", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / irodori_repo_name(IRODORI_DEFAULT_MODEL)
        expected = _in_subtree(IRODORI_DEFAULT_MODEL, IRODORI_OUTPUT_PATHS.values())
        assert sorted(verify_dist(out_dir)) == sorted(expected)

    def test_the_model_flag_moves_the_series_and_the_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sources = _build_irodori_sources(tmp_path, model="v9-large")
        self._reroot(tmp_path, monkeypatch)

        main(
            ["--pipeline", "irodori", "--model", "v9-large", "--series", str(sources.series.parent)]
        )

        out_dir = tmp_path / "models" / "karume-irodori-v9-large"
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert list(manifest["models"]) == ["v9-large"]
        assert verify_dist(out_dir)
