"""SBV2 の配布 recipe（`sbv2.distribution`）— 組み立て 1 周ぶんの単体テスト。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合だけ**（格納形の門）
なので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。SBV2 は加えて
`config.json` と `style_vectors.npy` を読むが、どちらも数行 / 数 KB の合成物で足りる
（**合成 config の style2id は実重み FN4 と別の並び**にしてある — 表を焼き込んでいれば落ちる）。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

core だけで観測できる層（合成計画で足りる規模上限・quant 完全写像・staging/swap の不変条件）は
`tools/exporter/tests/test_dist.py` が持つ（ADR 0065 段 3+4 の分割）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from dist import main
from karume.dist import (
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MODEL_CARD_FILENAME,
    SHARED_DIRNAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)
from karume.ir import IR_METADATA_KEY
from sbv2.distribution import (
    PIPELINE,
    SBV2_DEFAULT_MODEL,
    SBV2_DEFAULT_QUANT,
    SBV2_KNOB_KEYS,
    SBV2_MAX_FRAMES,
    SBV2_MAX_TOKENS,
    SBV2_OUTPUT_PATHS,
    SBV2_QUANTS,
    SBV2_SPEAKER_KEY,
    SBV2_SPEAKER_TENSOR,
    SBV2_STYLE_KEY,
    SBV2_TEXT_ENCODER_INPUTS,
    SBV2_WEIGHTS,
    Sbv2Sources,
    sbv2_knob_defaults,
    sbv2_pipeline_config,
    sbv2_plan,
    sbv2_repo_name,
    sbv2_series_name,
    sbv2_sources,
    sbv2_speaker_embeddings,
    sbv2_style_vectors,
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


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """モデルサブツリー内の期待 path（ADR 0041 §9 の一様レイアウト）。"""
    return [f"{model}/{rel}" for rel in paths]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


#: `style_bert_vits2` は optional な `sbv2` dependency-group。定数から既定ノブを引く層だけが
#: これを要る（組み立て本体はノブを引数で受けるので、無い環境でも全経路を観測できる）。
requires_sbv2_package = pytest.mark.skipif(
    importlib.util.find_spec("style_bert_vits2") is None,
    reason="sbv2 dependency-group が無い（`uv sync --group sbv2` が前提）",
)

#: 偽 text_encoder の IR の形と `symbols.json` の取り出し位置。門 `assert_bert_hidden` が
#: 通すのは **22 層 × 出力 1 本 × 位置 1** の組み合わせだけ（実資産と同じ）。
_SBV2_GRAPH_LAYERS = 22
_SBV2_GRAPH_OUTPUTS = 1
_SBV2_BERT_FROM_END = 1


def _fake_ir(
    layers: int = _SBV2_GRAPH_LAYERS,
    outputs: int = _SBV2_GRAPH_OUTPUTS,
    inputs: Iterable[str] = SBV2_TEXT_ENCODER_INPUTS,
) -> str:
    """門が読む最小の IR メタデータ（層番号つき initializer 名・出力名・入力名だけ）。"""
    return json.dumps(
        {
            "initializers": {
                f"p_model_encoder_layer_{index}_attention_self_query_proj_weight": {}
                for index in range(layers)
            },
            "outputs": [f"layer_norm_{index}" for index in range(outputs)],
            "inputs": [{"name": name} for name in inputs],
        }
    )


#: 偽資産（役割ごとに違うバイト列 — 取り違えがハッシュで見える）。モデル 5 役は
#: `SBV2_STORAGE_REQUIREMENTS` が要求する dtype をヘッダに持つ。text_encoder だけは
#: IR コンテナとしても読まれる（層数と出力本数の門）ので `__metadata__` を持つ。
_SBV2_PAYLOADS = {
    "text_encoder": _fake_safetensors("I8", b"deberta-i8-weights", {IR_METADATA_KEY: _fake_ir()}),
    "text_encoder_i4": _fake_safetensors(
        "I4", b"deberta-i4-weights", {IR_METADATA_KEY: _fake_ir()}
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

    3 つの置き場の並びは `_shared.paths` の実レイアウト（`outputs/series/` / `outputs/` 直下 /
    `inputs/`）に揃える — CLI 経路のテストが root を差し替えるだけで同じ木を指せる形。
    `offset` は表と重みにモデルごとの差を入れる軸（ファミリー組み立ての共有判定を見るため）。
    """
    series = root / "outputs" / "series"
    stem = sbv2_series_name(model)
    sources = Sbv2Sources(
        series_f16=series / f"{stem}-f16",
        series_i8=series / f"{stem}-i8",
        text_encoder=series / "deberta-i8" / "sbv2-22layer",
        text_encoder_i4=series / "deberta-i4" / "sbv2-22layer",
        demo=root / "outputs" / "sbv2-demo",
        model=root / "inputs" / "sbv2" / model,
    )
    _write(sources.text_encoder / "model.safetensors", _SBV2_PAYLOADS["text_encoder"])
    _write(sources.text_encoder / "io.case0.safetensors", b"io-fixture")
    _write(sources.text_encoder_i4 / "model.safetensors", _SBV2_PAYLOADS["text_encoder_i4"])
    _write(sources.text_encoder_i4 / "io.case0.safetensors", b"io-fixture")
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

    def test_it_declares_the_operating_limits_the_loader_requires(self, sbv2_assembled) -> None:
        """上限 2 欄は配布形にしか無い（ロード側は定数を持たず、欠けていれば読めない）。"""
        _, manifest = sbv2_assembled
        config = _sbv2_model(manifest)["pipelineConfig"]
        assert config["maxTokens"] == SBV2_MAX_TOKENS
        assert config["maxFrames"] == SBV2_MAX_FRAMES

    def test_the_limits_are_the_symbolic_maxima_the_export_scripts_baked(self) -> None:
        """配る上限と焼いた記号次元の上限が同じ 1 組であること。

        ずれると「宣言は通るのにグラフの相対位置表が足りない」形になり、**配ってから利用者の
        手元でしか出ない**（front / voice は記号次元の上限を実行時に自己申告しない）。
        トークン列 T は DeBERTa の記号次元で、音素次元 P の上限（front）と同値であることも
        ここで固定する — 片方だけ動いたら上限の意味が割れる。
        """
        from deberta import export as export_deberta
        from sbv2 import export as export_sbv2

        assert SBV2_MAX_TOKENS == export_deberta.SYM_MAX
        assert SBV2_MAX_TOKENS == export_sbv2.SYM_MAX
        assert SBV2_MAX_FRAMES == export_sbv2.FLOW_SYM_MAX


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

    def test_it_stops_when_the_i4_seat_holds_the_i8_series(self, tmp_path: Path) -> None:
        """i4 席に i8 系列を挿すと、サイズだけが元へ戻った配布形が黙って組み上がる。

        2 つの席は同じ 22 層 DeBERTa なので、層数・出力・入力の門は**両方とも通る** —
        格納 dtype の要求だけが席を区別できる。
        """
        sources = _build_sbv2_sources(tmp_path)
        (sources.text_encoder_i4 / "model.safetensors").write_bytes(_SBV2_PAYLOADS["text_encoder"])
        with pytest.raises(DistError, match=r"text_encoder_i4: .* I4 が無い"):
            _assemble_sbv2(sources, tmp_path / "out")


class TestSbv2BertHiddenGate:
    """22 層 × 出力 1 本 × 位置 1 の組み合わせだけを通す門。

    層数と出力形は `deberta/export.py` の variant が、位置は `sbv2/demo.py` の定数が持つ別々の
    台本なので、対で動かし忘れた配布形が普通に組み上がる。ずれても shape は合ったまま実行が
    通り、**別の層の BERT 特徴で音が出る**だけで沈黙する。
    """

    def test_it_stops_when_the_encoder_was_not_truncated(self, tmp_path: Path) -> None:
        """切り詰め忘れの 24 層資産（出力 1 本）は、最終出力が layer 23 なので別の層になる。"""
        sources = _build_sbv2_sources(tmp_path)
        (sources.text_encoder / "model.safetensors").write_bytes(
            _fake_safetensors("I8", b"deberta-i8-weights", {IR_METADATA_KEY: _fake_ir(layers=24)})
        )
        out_dir = tmp_path / "out"
        with pytest.raises(DistError, match=r"encoder は 24 層で、期待の 22 層でない"):
            _assemble_sbv2(sources, out_dir)
        # 検査は配置の前 — 途中の配布形を 1 ファイルも残さない。
        assert not out_dir.exists()

    def test_it_stops_when_the_verification_variant_slipped_in(self, tmp_path: Path) -> None:
        """全層出し（検証用）の資産が配布経路に混ざると、readback も取り出し位置も変わる。"""
        sources = _build_sbv2_sources(tmp_path)
        (sources.text_encoder / "model.safetensors").write_bytes(
            _fake_safetensors("I8", b"deberta-i8-weights", {IR_METADATA_KEY: _fake_ir(outputs=23)})
        )
        with pytest.raises(DistError, match=r"グラフ出力が 23 本で、配布形が要求する 1 本でない"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_only_the_symbols_are_stale(self, tmp_path: Path) -> None:
        """資産は 1 本出しなのに `symbols.json` が旧・全層出し向けの 3 を主張したまま。"""
        symbols = json.dumps({"defaults": dict(_SBV2_KNOBS), "bertHiddenFromEnd": 3})
        sources = _build_sbv2_sources(tmp_path, symbols=symbols.encode("utf-8"))
        with pytest.raises(DistError, match=r"bertHiddenFromEnd=3 が、出力 1 本のグラフで"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_position_tables_were_baked_back_in(self, tmp_path: Path) -> None:
        """添字表が入力から外れた資産（= 2MiB の定数が焼き戻った形）を配布経路で止める。"""
        sources = _build_sbv2_sources(tmp_path)
        (sources.text_encoder / "model.safetensors").write_bytes(
            _fake_safetensors(
                "I8",
                b"deberta-i8-weights",
                {IR_METADATA_KEY: _fake_ir(inputs=("input_ids", "attention_mask"))},
            )
        )
        with pytest.raises(DistError, match=r"グラフ入力が \['input_ids', 'attention_mask'\]"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_checks_every_text_encoder_seat(self, tmp_path: Path) -> None:
        """席ごとに掛ける門 — i8 席だけ見ていると i4 席の切り詰め忘れが素通りする。

        2 本は別々の `deberta/export.py` 実行の産物なので、対で動かし忘れる形は普通に起きる。
        """
        sources = _build_sbv2_sources(tmp_path)
        (sources.text_encoder_i4 / "model.safetensors").write_bytes(
            _fake_safetensors("I4", b"deberta-i4-weights", {IR_METADATA_KEY: _fake_ir(layers=24)})
        )
        with pytest.raises(DistError, match=r"encoder は 24 層で、期待の 22 層でない"):
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

    def test_the_text_encoder_declares_both_storage_forms(self, sbv2_assembled) -> None:
        """dtype キーは常に要る統一形（ADR 0041 §3 — 2 形パースを消した）。

        並びは {@link SBV2_WEIGHTS} の宣言順そのもの（manifest の weights 節と quant 節で
        同じ役割が別の順に並ばない — `complete_quant_weights` の MUST と対）。
        """
        _, manifest = sbv2_assembled
        entry = _sbv2_model(manifest)["weights"]["text_encoder"]
        assert list(entry) == ["i8", "i4"]
        assert entry["i8"]["file"]["path"].endswith("model.i8.safetensors")
        assert entry["i4"]["file"]["path"].endswith("model.i4.safetensors")

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
            assert quant["weights"]["text_encoder"] == ("i4" if name == "w8-bert4" else "i8"), name

    def test_the_bert4_quant_is_w8_with_only_the_text_encoder_swapped(self, sbv2_assembled) -> None:
        """`w8-bert4` の意味は「`w8` と同構成で BERT だけ i4」— 差分が 1 席であることを固定する。

        session ノブまで見るのは、`w8a8` のような**別軸**の変更が紛れ込んだまま「格納形だけの
        席」を名乗ると、聴感で採った裁定（perf-ledger Q-1）の対象が黙って変わるから。
        """
        _, manifest = sbv2_assembled
        quants = _sbv2_model(manifest)["quants"]
        base, variant = quants["w8"], quants["w8-bert4"]

        assert variant["session"] == base["session"]
        differing = {
            role for role in base["weights"] if base["weights"][role] != variant["weights"][role]
        }
        assert differing == {"text_encoder"}
        assert variant["weights"]["text_encoder"] == "i4"


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
        # 格納形ごとに**別系列**（ADR 0019）— variant 名だけが両者で同じ。
        assert sources.text_encoder_i4 == tmp_path / "deberta-i4" / "sbv2-22layer"

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
        import dist
        from sbv2 import distribution

        # `DIST_ROOT` は既定の出力先を決めるドライバ側（`dist.default_out_dir`）、`OUTPUTS_ROOT` /
        # `INPUTS_ROOT` は系列の外の入力を引く recipe 側 — 別モジュールの束縛を別々に外す。
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "OUTPUTS_ROOT", tmp_path / "outputs")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

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


class TestSbv2CardProfile:
    """帰属プロファイルの選択（`--card-profile`）— 誤帰属は配ってからでないと気づけない。

    engine 側の規則（1 つなら省略で通る / 2 つ以上なら明示必須）は
    `tools/exporter/tests/test_dist.py` の `TestCardProfile` が合成 pipeline で持つ。ここは
    **SBV2 の表が実際に 2 つの帰属を持つ**ことと、名前ごとに別の描き手へ解けることを見る。
    """

    def test_it_refuses_to_pick_an_attribution_when_several_exist(self) -> None:
        """既定を黙って選ぶと、新しいファミリーへ前のファミリーの帰属がそのまま残る。"""
        with pytest.raises(DistError, match="--card-profile") as error:
            resolve_card_renderer(PIPELINE, None)
        assert "fn" in str(error.value)
        assert "jvnv" in str(error.value)

    def test_it_refuses_a_profile_it_does_not_have(self) -> None:
        with pytest.raises(DistError, match="jvnv"):
            resolve_card_renderer(PIPELINE, "FN9")

    def test_it_resolves_each_name_to_its_own_renderer(self) -> None:
        """名前ごとに別の描き手（束ね違いなら 2 つのファミリーが同じカードを描く）。"""
        profiles = PIPELINE.card_profiles
        assert sorted(profiles) == ["fn", "jvnv"]
        assert resolve_card_renderer(PIPELINE, "jvnv") is profiles["jvnv"]
        assert profiles["fn"] is not profiles["jvnv"]


class TestSbv2PipelineEntry:
    """`--pipeline sbv2` の 1 行が指す先（ドライバが core の表へ合成する席）。"""

    def test_it_carries_the_default_model_and_the_repo_name(self) -> None:
        assert PIPELINE.default_model == SBV2_DEFAULT_MODEL
        assert PIPELINE.repo_name(SBV2_DEFAULT_MODEL) == f"karume-sbv2-{SBV2_DEFAULT_MODEL}"

    def test_its_card_refuses_another_pipelines_manifest(self) -> None:
        manifest = {"models": {"m": {"pipeline": "sbv2/0"}}}
        for render_card in PIPELINE.card_profiles.values():
            with pytest.raises(ValueError):
                render_card(manifest, "hdae/x")
