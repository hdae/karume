"""母音検出の配布 recipe（`vowel_detector.distribution`）— 組み立て 1 周ぶんの単体テスト。

実資産は使わない。組み立てへ届く入力は数 KB の**正当な最小 IR コンテナ**（`ir_fixtures`）で、
門に落とされることを見るケースだけが従来の偽資産のまま（{@link _vowel_detector_container}）。
特徴の契約と mel 基底の出どころ（`feature_config.json`）も合成 JSON で足りる。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

core だけで観測できる層（合成計画で足りる規模上限・quant 完全写像・staging/swap の不変条件・
帰属プロファイルの解決規則）は `tools/exporter/tests/test_dist.py` が持つ（ADR 0065 段 3+4 の
分割）。カードの観測は**組み立て 1 周ぶん**をここが持ち、テンプレート単位の門（案内するロード
入口）は `vowel_detector/tests/test_card.py`。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ir_fixtures import ir_container
from safetensors.numpy import load_file
from shard_series import placed_paths, write_component

from dist import default_out_dir, main
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)
from karume.ir import IR_METADATA_KEY
from vowel_detector.distribution import (
    PIPELINE,
    VOWEL_DETECTOR_DEFAULT_MODEL,
    VOWEL_DETECTOR_GRAPH_ROLE,
    VOWEL_DETECTOR_MAX_FRAMES,
    VOWEL_DETECTOR_OUTPUT_PATHS,
    VOWEL_DETECTOR_WEIGHTS,
    VowelDetectorSources,
    vowel_detector_plan,
    vowel_detector_repo_name,
    vowel_detector_series_name,
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


def _placed_paths() -> list[str]:
    """配布形に現れる相対 path — **weights の席だけ**が shard 連番に展開される（ADR 0081）。

    mel 基底は assets の席（1 ファイル参照）なので分割されない — 2 種の席が並ぶこの family では、
    展開が weights にだけ掛かることがそのまま期待値に出る。
    """
    return placed_paths(VOWEL_DETECTOR_OUTPUT_PATHS, VOWEL_DETECTOR_WEIGHTS)


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


# ---- 母音検出（音声 → リップシンク用の母音系列）------------------------------
#
# 偽資産は**記号長で焼かれた CRNN 1 本**（入力 `2T` / 出力 `T`）+ 上流 `feature_config.json`。
# ここが見るのは「1 本だけが並ぶ」「時間軸が記号のまま」「`pipelineConfig` の 4 欄が上流 config
# と台本の宣言から組まれる」の 3 つ。長さバケット 4 本の時代は ADR 0056 / 0057 で終わった。

#: `pipelineConfig` の欄名（TS 側 `packages/models/src/vowel-detector/config.ts` の `ROOT_KEYS`
#: の写し）。**ロード側は未知キーも欠落も parse 時に落とす**ので、焼く側とロード側の欄名は
#: 完全一致が要る。
_VOWEL_DETECTOR_CONFIG_KEYS = ("sampleRate", "featureDim", "classes", "maxFrames")

_VOWEL_DETECTOR_N_MELS = 80
_VOWEL_DETECTOR_N_FFT = 512
_VOWEL_DETECTOR_CLASSES = ("a", "i", "u", "e", "o", "N", "pau", "cons")


def _vowel_detector_feature_config(**patch: Any) -> dict[str, Any]:
    """上流 `assets/feature_config.json` の骨格（門が読む欄だけ・mel 基底は一様な三角窓もどき）。"""
    bins = _VOWEL_DETECTOR_N_FFT // 2 + 1
    basis = np.zeros((_VOWEL_DETECTOR_N_MELS, bins), dtype=np.float32)
    # 行ごとに 1 本だけ立てる（空の mel チャネルの門を通す最小の形）。
    for row in range(_VOWEL_DETECTOR_N_MELS):
        basis[row, row + 1] = 1.0
    return {
        "sample_rate": 16000,
        "n_fft": _VOWEL_DETECTOR_N_FFT,
        "n_mels": _VOWEL_DETECTOR_N_MELS,
        "feature_dim": _VOWEL_DETECTOR_N_MELS + 3,
        "classes": list(_VOWEL_DETECTOR_CLASSES),
        "mel_basis": basis.tolist(),
        **patch,
    }


def _vowel_detector_graph(
    *,
    feature_dim: int = _VOWEL_DETECTOR_N_MELS + 3,
    name: str = "features",
    outputs: int = 1,
    in_shape: Sequence[Any] | None = None,
    out_shape: Sequence[Any] | None = None,
    symbols: Sequence[str] = ("T",),
) -> str:
    """門が読む最小の IR メタデータ（入力 1 本・出力の宣言・記号次元）。"""
    names = [f"out_{index}" for index in range(outputs)]
    logits = list(out_shape) if out_shape is not None else [1, "T", len(_VOWEL_DETECTOR_CLASSES)]
    shape = list(in_shape) if in_shape is not None else [1, "2T", feature_dim]
    return json.dumps(
        {
            "inputs": [{"name": name, "shape": shape}],
            "outputs": names,
            "values": {output: {"dtype": "f32", "shape": logits} for output in names},
            "symbols": list(symbols),
        }
    )


def _vowel_detector_container(
    dtype: str, graph: str | None, storage: str | None
) -> bytes | list[bytes]:
    """系列に置く母音検出グラフの中身。

    組み立てへ届く既定の形は**正当な IR コンポーネント**でなければならない（組み立ては入力を
    IR v1 の全規則で見る — `karume.dist.assert_weight_components_verified`）。正当な側は
    shard 列（`list[bytes]`・先頭がグラフ shard — ADR 0081）で、偽コンテナは代表 path 1 本の
    `bytes` のまま（計画の門で止まるので分割の側まで届かない）。

    軸は 2 本ある:

    - `dtype` は**単一 dtype の偽コンテナ**が名乗るヘッダ dtype。要求検査（「F32 が無い」）を
      見るケースはこちらでしか作れない — 実物どおりの f16 系列は F32 も含むので、要求検査は
      通ってしまう。
    - `storage` は**実物どおりの混成コンテナ**の格納形（f16 / i8 / i4 は適格外の重みと bias が
      F32 で残り、i4 はさらに I8 が混ざる）。禁止表（{@link VOWEL_DETECTOR_STORAGE_FORBIDDEN}）の
      門はこの形でしか試せない。
    """
    if storage is not None or (graph is None and dtype == "F32"):
        return ir_container(
            mark="vowel-detector",
            storage=storage if storage is not None else dtype.lower(),
            inputs=(("features", (1, "2T", _VOWEL_DETECTOR_N_MELS + 3)),),
            outputs=([1, "T", len(_VOWEL_DETECTOR_CLASSES)],),
        )
    return _fake_safetensors(
        dtype, b"vowel-detector", {IR_METADATA_KEY: graph or _vowel_detector_graph()}
    )


def _build_vowel_detector_sources(
    root: Path,
    *,
    model: str = VOWEL_DETECTOR_DEFAULT_MODEL,
    graph: str | None = None,
    feature_config: Mapping[str, Any] | None = None,
    omit_feature_config: bool = False,
    omit_graph: bool = False,
    dtype: str = "F32",
    storage: str | None = None,
) -> VowelDetectorSources:
    """系列 1 本と上流素材を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `_shared.paths` の実レイアウト（`outputs/series/` と `inputs/`）に揃える — CLI 経路の
    テストが root を差し替えるだけで同じ木を指せる形。
    """
    sources = VowelDetectorSources(
        series_dir=root / "outputs" / "series",
        model=root / "inputs" / "vowel-detector",
        model_name=model,
    )
    if not omit_graph:
        write_component(
            sources.series / "model.safetensors",
            _vowel_detector_container(dtype, graph, storage),
        )
        # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
        _write(sources.series / "io.silence.safetensors", b"io-fixture")
    if not omit_feature_config:
        raw = feature_config if feature_config is not None else _vowel_detector_feature_config()
        _write(sources.model / "feature_config.json", json.dumps(raw).encode("utf-8"))
    return sources


@pytest.fixture
def vowel_detector_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_vowel_detector_sources(tmp_path)
    out_dir = tmp_path / "models" / vowel_detector_repo_name(VOWEL_DETECTOR_DEFAULT_MODEL)
    manifest = assemble_family(
        [vowel_detector_plan(sources, VOWEL_DETECTOR_DEFAULT_MODEL)],
        out_dir,
        VOWEL_DETECTOR_DEFAULT_MODEL,
    )
    return out_dir, manifest


def _vowel_detector_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][VOWEL_DETECTOR_DEFAULT_MODEL]


class TestVowelDetectorLayout:
    def test_it_places_the_graph_under_the_model_subtree(self, vowel_detector_assembled) -> None:
        out_dir, _ = vowel_detector_assembled
        expected = _in_subtree(VOWEL_DETECTOR_DEFAULT_MODEL, _placed_paths())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_the_io_fixtures(self, vowel_detector_assembled) -> None:
        out_dir, _ = vowel_detector_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_the_graph_is_one_role_and_the_mel_basis_is_an_asset(
        self, vowel_detector_assembled
    ) -> None:
        """記号長になったので CRNN は 1 役割きり（長さバケットの役割軸は消えた）。"""
        _, manifest = vowel_detector_assembled
        model = _vowel_detector_model(manifest)
        assert model["pipeline"] == "vowel-detector/1"
        assert list(model["weights"]) == ["crnn"]
        assert list(model["assets"]) == ["mel_basis"]
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {"crnn": "f32"}

    def test_it_writes_the_mel_basis_as_one_f32_tensor(self, vowel_detector_assembled) -> None:
        """特徴抽出はグラフの外なので、mel 基底は資産として配らないと再現できない。"""
        out_dir, manifest = vowel_detector_assembled
        ref = _vowel_detector_model(manifest)["assets"]["mel_basis"]
        tensors = load_file(str(out_dir / ref["path"]))
        assert list(tensors) == ["mel_basis"]
        basis = tensors["mel_basis"]
        assert basis.dtype == np.float32
        assert basis.shape == (_VOWEL_DETECTOR_N_MELS, _VOWEL_DETECTOR_N_FFT // 2 + 1)

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(tmp_path)
        out_dir = tmp_path / "models" / vowel_detector_repo_name(VOWEL_DETECTOR_DEFAULT_MODEL)
        first = assemble_family(
            [vowel_detector_plan(sources)], out_dir, VOWEL_DETECTOR_DEFAULT_MODEL
        )
        assert first == assemble_family(
            [vowel_detector_plan(sources)], out_dir, VOWEL_DETECTOR_DEFAULT_MODEL
        )
        assert verify_dist(out_dir)

    def test_it_refuses_a_compressed_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """格納形は系列ディレクトリ名でなくヘッダが正（`--dtype` 付け忘れの逆向き）。"""
        sources = _build_vowel_detector_sources(tmp_path, dtype="F16")
        with pytest.raises(DistError, match="F32 が無い"):
            vowel_detector_plan(sources)

    @pytest.mark.parametrize(("storage", "intruder"), [("f16", "F16"), ("i8", "I8"), ("i4", "I4")])
    def test_it_refuses_a_real_compressed_series_in_the_f32_seat(
        self, tmp_path: Path, storage: str, intruder: str
    ) -> None:
        """実物どおりの圧縮系列は**要求検査を満たす** — 禁止表だけが系列 root の取り違えを見る。

        圧縮系列も適格外の重みと bias を F32 で持つので「F32 を含む」は真になり、上の
        単一 dtype の偽資産と違って要求検査では 1 バイトも落ちない。落ちるべき理由は
        「f32 席に別系列の資産が居る」で、数値の門では原理的に検出できない
        （ADR 0027 / 0029）。
        """
        sources = _build_vowel_detector_sources(tmp_path, storage=storage)
        with pytest.raises(DistError, match=rf"{VOWEL_DETECTOR_GRAPH_ROLE}: .* {intruder} がある"):
            vowel_detector_plan(sources)

    def test_the_plain_f32_series_passes_both_storage_gates(self, tmp_path: Path) -> None:
        """正常対 — 素の f32 系列は要求検査も禁止表も通る（禁止表が恒真に倒れていない）。"""
        sources = _build_vowel_detector_sources(tmp_path, storage="f32")

        assert vowel_detector_plan(sources).name == VOWEL_DETECTOR_DEFAULT_MODEL


class TestVowelDetectorPipelineConfig:
    """`pipelineConfig` はロード側（`src/vowel-detector/config.ts`）のスキーマと欄名まで一致。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(
        self, vowel_detector_assembled
    ) -> None:
        _, manifest = vowel_detector_assembled
        assert (
            tuple(_vowel_detector_model(manifest)["pipelineConfig"]) == _VOWEL_DETECTOR_CONFIG_KEYS
        )

    def test_it_takes_the_feature_contract_from_the_upstream_config(
        self, vowel_detector_assembled
    ) -> None:
        _, manifest = vowel_detector_assembled
        config = _vowel_detector_model(manifest)["pipelineConfig"]
        assert config["sampleRate"] == 16000
        assert config["featureDim"] == _VOWEL_DETECTOR_N_MELS + 3
        assert config["classes"] == list(_VOWEL_DETECTOR_CLASSES)

    def test_it_declares_the_operating_limit_the_loader_requires(
        self, vowel_detector_assembled
    ) -> None:
        """上限は配布形にしか無い（IR は記号の値域を持たず、ロード側は定数を持たない）。"""
        _, manifest = vowel_detector_assembled
        config = _vowel_detector_model(manifest)["pipelineConfig"]
        assert config["maxFrames"] == VOWEL_DETECTOR_MAX_FRAMES

    def test_the_limit_is_the_symbolic_maximum_the_export_script_baked(self) -> None:
        """配る上限と焼いた記号次元の上限が同じ 1 組であること。

        `SYM_MAX` は 20ms 格子の本数、配る `maxFrames` は 10ms 格子の本数（入力は `2T`）。
        ずれると「宣言は通るのにグラフが受けていない長さ」を利用者の手元で踏む。
        """
        from vowel_detector import export

        assert VOWEL_DETECTOR_MAX_FRAMES == 2 * export.SYM_MAX

    def test_it_refuses_a_feature_dim_that_is_not_mel_plus_dsp(self, tmp_path: Path) -> None:
        """内訳が上流と食い違う config は、グラフと形が合っていても受理しない。"""
        sources = _build_vowel_detector_sources(
            tmp_path, feature_config=_vowel_detector_feature_config(feature_dim=90)
        )
        with pytest.raises(DistError, match="特徴の内訳が上流と食い違っている"):
            vowel_detector_plan(sources)

    def test_it_needs_the_upstream_feature_config(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(tmp_path, omit_feature_config=True)
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            vowel_detector_plan(sources)


class TestVowelDetectorMelBasis:
    """mel 基底は数値経路の一部そのもの — 形も中身もここでしか検査されない。"""

    def test_it_refuses_a_basis_shaped_for_another_n_fft(self, tmp_path: Path) -> None:
        config = _vowel_detector_feature_config()
        config["mel_basis"] = [row[:-1] for row in config["mel_basis"]]
        sources = _build_vowel_detector_sources(tmp_path, feature_config=config)
        with pytest.raises(DistError, match="n_mels / n_fft から組んだ期待"):
            vowel_detector_plan(sources)

    def test_it_refuses_an_empty_mel_channel(self, tmp_path: Path) -> None:
        """帯域外へ落ちた三角窓は、その 1 列を常に log の下駄へ張り付かせる（沈黙劣化）。"""
        config = _vowel_detector_feature_config()
        config["mel_basis"][7] = [0.0] * len(config["mel_basis"][7])
        sources = _build_vowel_detector_sources(tmp_path, feature_config=config)
        with pytest.raises(DistError, match="空の mel チャネル"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_non_finite_basis(self, tmp_path: Path) -> None:
        config = _vowel_detector_feature_config()
        config["mel_basis"][3][5] = float("nan")
        sources = _build_vowel_detector_sources(tmp_path, feature_config=config)
        with pytest.raises(DistError, match="有限でない要素"):
            vowel_detector_plan(sources)


class TestVowelDetectorGraphGate:
    """組み立て門 — ずれても配布形としては成立してしまう組み合わせを、配置の前に落とす。"""

    def test_it_refuses_a_graph_baked_for_a_fixed_length(self, tmp_path: Path) -> None:
        """長さ固定のグラフは名前も階数も同じ = 載せても manifest は成立してしまう。

        利用者の手元では**その 1 長以外の全ての音声**が実行時に落ちる（配布してから出る）。
        """
        sources = _build_vowel_detector_sources(
            tmp_path,
            graph=_vowel_detector_graph(
                in_shape=[1, 500, _VOWEL_DETECTOR_N_MELS + 3],
                out_shape=[1, 250, len(_VOWEL_DETECTOR_CLASSES)],
                symbols=(),
            ),
        )
        with pytest.raises(DistError, match="記号次元"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_whose_input_is_not_twice_the_symbol(self, tmp_path: Path) -> None:
        """入力が `T` のままだと、実行時の束縛が 2 倍ずれて `.lab` の時間が伸びる。"""
        sources = _build_vowel_detector_sources(
            tmp_path,
            graph=_vowel_detector_graph(in_shape=[1, "T", _VOWEL_DETECTOR_N_MELS + 3]),
        )
        with pytest.raises(DistError, match="10ms 格子の長さは記号 T の 2 倍"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_with_another_feature_dim(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(
            tmp_path, graph=_vowel_detector_graph(feature_dim=80)
        )
        with pytest.raises(DistError, match="期待は"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_whose_output_is_not_the_20ms_grid(self, tmp_path: Path) -> None:
        """出力が 10ms 格子のまま焼かれていると、`.lab` の時間が 2 倍に伸びる。"""
        sources = _build_vowel_detector_sources(
            tmp_path,
            graph=_vowel_detector_graph(out_shape=[1, "2T", len(_VOWEL_DETECTOR_CLASSES)]),
        )
        with pytest.raises(DistError, match="20ms 格子"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(tmp_path, graph=_vowel_detector_graph(outputs=2))
        with pytest.raises(DistError, match="ロジット 1 本だけ"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_with_a_second_symbol(self, tmp_path: Path) -> None:
        """記号は時間軸 1 本きり（2 本目はホストが束縛を渡せない）。"""
        sources = _build_vowel_detector_sources(
            tmp_path, graph=_vowel_detector_graph(symbols=("T", "S"))
        )
        with pytest.raises(DistError, match="記号次元"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """実行側は名前で束ねるので、綴りが変われば束ねられない。"""
        sources = _build_vowel_detector_sources(tmp_path, graph=_vowel_detector_graph(name="mel"))
        with pytest.raises(DistError, match="グラフ入力"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_missing_graph(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(tmp_path, omit_graph=True)
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            vowel_detector_plan(sources)


class TestVowelDetectorModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """偽資産だけで CLI を 1 周回す（上流素材の置き場も tmp へ寄せる）。"""
        from vowel_detector import distribution

        sources = _build_vowel_detector_sources(tmp_path)
        # `INPUTS_ROOT`（系列の外にある上流素材）を引くのは recipe 側 — ドライバの `dist`
        # ではなくこちらの束縛を外す（`DIST_ROOT` は逆でドライバ側）。
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "vowel-detector",
                "--series",
                str(sources.series_dir),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_lip_sync_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: audio-classification" in card
        assert "license: mit" in card
        # 格納形を変えない配布形は 4 値のどれでもない（Hub の推論に任せる）。
        assert "base_model_relation" not in card
        # 「任意長 1 本」と上限は利用者が最初に確かめたい制約そのもの（数は manifest から）。
        assert "**One graph, any length.**" in card
        assert "longer than 600.0 s is rejected rather than silently truncated" in card
        assert f"**{VOWEL_DETECTOR_MAX_FRAMES} frames of 10 ms** (600.0 s)" in card

    def test_the_usage_snippet_enumerates_the_choices_from_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """裁定 2026-08-12 の形: optional はコメント併記で、選べる値は manifest 由来の列挙。

        `detect()` に実行時ノブは無いので、この pipeline の optional は model / quant だけ。
        1 席しか無くても「available:」で綴るのは、席が増えたときに文面が自動で追従するため
        （「the only one this repository ships」のような焼き込みは、増えた瞬間に嘘になる）。
        """
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert (
            f'  // model: "{VOWEL_DETECTOR_DEFAULT_MODEL}",'
            f" // default — available: {VOWEL_DETECTOR_DEFAULT_MODEL}" in card
        )
        assert '  // quant: "f32", // default — available: f32' in card
        assert "the only one this repository ships" not in card

    def test_it_carries_the_upstream_training_attributions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """上流 NOTICE.txt が「配布するなら帰属を残してほしい」と明記している分。"""
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        for source in (
            "reazon-research/japanese-hubert-base-k2",
            "ROHAN4600",
            "ITA corpus",
            "Common Voice ja",
            "Style-Bert-VITS2",
            "AivisHub",
            "ACML 1.0",
            "JSUT basic5000",
        ):
            assert source in card, source
        # 教師モデルと合成エンジンは**同梱していない**ことまで書く（AGPL-3.0 の誤解を防ぐ）。
        assert "the teacher is **not** part of these weights" in card
        assert "not distributed with, and not part of, these weights" in card

    def test_the_card_comes_from_a_verified_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert verify_dist(self._run(tmp_path, monkeypatch))


class TestVowelDetectorCli:
    def test_the_repository_name_does_not_carry_the_checkpoint_generation(self) -> None:
        """世代は manifest のキーが綴る事実で、世代が上がるたびにリポが増える形にしない。"""
        assert (
            default_out_dir(PIPELINE, [VOWEL_DETECTOR_DEFAULT_MODEL]).name
            == "karume-vowel-detector"
        )

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    def test_the_series_name_follows_the_exporter(self) -> None:
        """系列名は `vowel_detector.export.default_out_dir` と同じ式（表を 2 つ持たない）。"""
        assert (
            vowel_detector_series_name(VOWEL_DETECTOR_DEFAULT_MODEL) == "vowel-detector-crnn-epoch3"
        )

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist
        from vowel_detector import distribution

        sources = _build_vowel_detector_sources(tmp_path)
        # `DIST_ROOT` は既定の出力先を決めるドライバ側（`dist.default_out_dir`）、`INPUTS_ROOT` は
        # 系列の外の入力を引く recipe 側 — 別モジュールの束縛を別々に外す。
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "vowel-detector", "--series", str(sources.series_dir)])

        out_dir = tmp_path / "models" / "karume-vowel-detector"
        expected = _in_subtree(VOWEL_DETECTOR_DEFAULT_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)
