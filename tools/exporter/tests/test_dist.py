"""配布ディレクトリ組み立て（`karume.dist`）の単体テスト。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合だけ**（格納形の門）
なので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

NOTE: Anima / SBV2 / Irodori の配布 recipe は wheel の外へ出た（ADR 0065 段 3+4）ので、その依存
ケースは `tools/export-recipes/<family>/tests/test_distribution.py` に居る。ここに残るのは core
だけで観測できる層（合成計画で足りる規模上限・quant 完全写像・staging/swap の不変条件・帰属
プロファイルの解決規則）と、まだ core に居る family の recipe。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from safetensors.numpy import load_file

from karume.dist import (
    BIREFNET_DEFAULT_MODEL,
    BIREFNET_IMAGE_MEAN,
    BIREFNET_IMAGE_STD,
    BIREFNET_OUTPUT_PATHS,
    BIREFNET_RESOLUTION,
    BIREFNET_ROLE,
    DEPTH_ANYTHING_DEFAULT_MODEL,
    DEPTH_ANYTHING_OUTPUT_PATHS,
    DEPTH_ANYTHING_ROLE,
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MODEL_CARD_FILENAME,
    PIPELINES,
    SHARED_DIRNAME,
    SIGLIP2_DEFAULT_MODEL,
    SIGLIP2_OUTPUT_PATHS,
    SIGLIP2_ROLE,
    SUPERSEDED_SUFFIX,
    VOWEL_DETECTOR_DEFAULT_MODEL,
    VOWEL_DETECTOR_MAX_FRAMES,
    VOWEL_DETECTOR_OUTPUT_PATHS,
    Artifact,
    BirefnetSources,
    DepthAnythingSources,
    DistError,
    ModelPlan,
    Pipeline,
    Siglip2Sources,
    VowelDetectorSources,
    WeightFiles,
    assemble_family,
    assert_manifest_limits,
    assert_model_name,
    birefnet_plan,
    birefnet_repo_name,
    birefnet_series_name,
    birefnet_sources,
    build_parser,
    complete_quant_weights,
    default_out_dir,
    depth_anything_checkpoint,
    depth_anything_plan,
    depth_anything_repo_name,
    depth_anything_series_name,
    depth_anything_sources,
    main,
    materialize,
    resolve_card_renderer,
    siglip2_checkpoint,
    siglip2_plan,
    siglip2_repo_name,
    siglip2_sources,
    verify_dist,
    vowel_detector_plan,
    vowel_detector_repo_name,
    vowel_detector_series_name,
)
from karume.ir import IR_METADATA_KEY
from karume.modelcard import (
    BIREFNET_UPSTREAM,
    DEPTH_ANYTHING_LICENSE,
    DEPTH_ANYTHING_UPSTREAM,
    SIGLIP2_UPSTREAM,
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


def _ref(path: str) -> dict[str, Any]:
    """3 点セットの偽値（規模上限の検査は件数しか見ないので中身は形だけで足りる）。"""
    return {"path": path, "size": 1, "sha256": "0" * 64}


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


def _synthetic_plan(name: str, rel_path: str, payload: bytes) -> ModelPlan:
    """1 役だけを持つ最小の計画（配置と共有の畳み込みだけを観測するための合成）。

    中身は生成物（`payload`）で持つ — 系列の偽資産を組まずに「どのモデルがどのバイト列を
    主張するか」だけを作り分けられる。
    """
    return ModelPlan(
        name=name,
        pipeline="anima/1",
        artifacts={"w": Artifact(rel_path=rel_path, payload=payload)},
        weights={"w": {"f16": WeightFiles(file="w")}},
        assets={},
        quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
        default_quant="f16",
        pipeline_config={},
    )


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
        for name in (DEPTH_ANYTHING_DEFAULT_MODEL, SIGLIP2_DEFAULT_MODEL, "jvnv-F1"):
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


class TestPlanGates:
    """`plans` だけから決まる検査は**最初の 1 バイトを書く前**（dist.py 冒頭の MUST）。

    数 GB を並べ切ってから落ちると、`karume.json` の無い / 前回のままの中途半端なツリーが残る。
    """

    def test_it_refuses_too_many_models_before_writing_anything(self, tmp_path: Path) -> None:
        plans = [
            _synthetic_plan(f"m{index}", "w/model.safetensors", f"weights-{index}".encode())
            for index in range(33)
        ]
        out_dir = tmp_path / "models" / "too-many"
        with pytest.raises(DistError, match="models が 33 件"):
            assemble_family(plans, out_dir, "m0")
        assert not out_dir.exists()


class TestFamilyAssembly:
    """1 リポに複数モデル（ADR 0041 §2）+ 共有ファイルは `shared/` に 1 回だけ（§5）。"""

    def test_it_never_folds_a_path_that_two_byte_strings_both_claim(self, tmp_path: Path) -> None:
        """同じ相対 path に畳める組が 2 つあるとき、どの組も畳まない（席は path で 1 つだけ）。

        畳むと後の組が先の組の `shared/` 実体を上書きし、先の組のモデルは自分の manifest が
        宣言する sha256 とは**別のバイト列**を読む配布形になる。長さを揃えてあるので
        `verify_dist` の size 突合も重複 path の 3 点セット突合も緑のまま = 沈黙する経路。
        """
        rel_path = "text_encoder/model.safetensors"
        first, second = b"weights-D1", b"weights-D2"
        assert len(first) == len(second)
        plans = [
            _synthetic_plan(name, rel_path, payload)
            for name, payload in (("A", first), ("B", first), ("C", second), ("D", second))
        ]
        out_dir = tmp_path / "models" / "contested"
        manifest = assemble_family(plans, out_dir, "A")

        for name, payload in (("A", first), ("B", first), ("C", second), ("D", second)):
            ref = manifest["models"][name]["weights"]["w"]["f16"]["file"]
            assert (out_dir / ref["path"]).read_bytes() == payload, name
            assert ref["sha256"] == hashlib.sha256(payload).hexdigest(), name
            assert ref["size"] == len(payload), name
        assert not (out_dir / SHARED_DIRNAME).exists()
        assert sorted(verify_dist(out_dir)) == sorted(f"{name}/{rel_path}" for name in "ABCD")

    def test_it_refuses_a_shared_copy_that_did_not_land_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """共有席は 1 本しか置かないので、コピーが壊れると**全モデルが揃って**壊れた実体を指す。

        宣言する sha256 は置いた現物から採るため、壊れたバイト列と宣言は一致したままになり
        `verify_dist` は沈黙する（採り直さない）。出所の sha256 と突き合わせられるのは、
        共有判定のために既に出所を読んでいる配置の側だけ。
        """

        def corrupt(artifact: Artifact, dest: Path) -> None:
            materialize(artifact, dest)
            if SHARED_DIRNAME in dest.parts:
                dest.write_bytes(b"corrupted")

        monkeypatch.setattr("karume.dist.materialize", corrupt)
        plans = [_synthetic_plan(name, "w/model.safetensors", b"same-bytes") for name in ("A", "B")]
        with pytest.raises(DistError, match="出所と食い違う"):
            assemble_family(plans, tmp_path / "models" / "family", "A")


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

    def _fail_replace_at(self, monkeypatch: pytest.MonkeyPatch, nth: int) -> None:
        """`nth` 回目の `os.replace` だけを I/O 故障にする（据え替えの途中で落ちる形の注入）。"""
        real = os.replace
        calls = 0

        def failing(src: Path, dst: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == nth:
                raise OSError("据え替えの途中で落ちた")
            real(src, dst)

        monkeypatch.setattr("karume.dist.os.replace", failing)

    def _versioned_plans(self, version: str) -> list[ModelPlan]:
        """3 モデル × 1 役の最小計画。版ごとに**中身だけ**が変わる（長さは同じ）。

        同じ中身で組み直すと、途中まで上書きされた木も前回の木とバイト単位で見分けられない
        （不変の主張が空振りする）— 故障注入の観測には版の違いが要る。
        """
        return [
            _synthetic_plan(name, "w/model.safetensors", f"{version}-{name}".encode())
            for name in ("A", "B", "C")
        ]

    def test_a_failure_midway_leaves_the_previous_distribution_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """in-place で更新していた頃はここで「旧 manifest + 新旧混在ツリー」が残っていた。"""
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans("v1"), out_dir, "A")
        before = self._snapshot(out_dir)

        self._fail_at(monkeypatch, nth=2)
        with pytest.raises(OSError, match="配置の途中で落ちた"):
            assemble_family(self._versioned_plans("v2"), out_dir, "A")

        assert self._snapshot(out_dir) == before
        assert self._siblings(out_dir) == []

    def test_a_failing_card_renderer_leaves_the_previous_distribution_byte_identical(
        self, tmp_path: Path
    ) -> None:
        """カードも差し替えの内側 — 描けなければ据え替えごと起きない。"""
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans("v1"), out_dir, "A")
        before = self._snapshot(out_dir)

        def explode(manifest: Mapping[str, Any]) -> str:
            raise DistError("カードが描けない")

        with pytest.raises(DistError, match="カードが描けない"):
            assemble_family(self._versioned_plans("v2"), out_dir, "A", render_card=explode)

        assert self._snapshot(out_dir) == before
        assert self._siblings(out_dir) == []

    def test_a_failing_swap_puts_the_previous_distribution_back_at_the_canonical_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """据え替えの 2 回目（staging → 出力先）で落ちても、正規 path から配布形が消えない。"""
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans("v1"), out_dir, "A")
        before = self._snapshot(out_dir)

        self._fail_replace_at(monkeypatch, nth=2)
        with pytest.raises(DistError, match="据え替え") as failure:
            assemble_family(self._versioned_plans("v2"), out_dir, "A")

        # 原因（I/O 故障）は連鎖で残す — 据え替えの失敗と組み立ての失敗を取り違えない。
        assert isinstance(failure.value.__cause__, OSError)
        assert self._snapshot(out_dir) == before
        assert self._siblings(out_dir) == []

    def test_it_restores_a_distribution_left_only_in_the_superseded_slot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """前回が rename 2 回の**間**で落ちた形 — 退避先の last-known-good を捨てずに戻す。

        次の組み立てが失敗しても、正常な配布形は正規 path に戻っている（捨てていれば
        「出力先も退避先も無い」= 手元の配布形が完全消失した状態で止まる）。
        """
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans("v1"), out_dir, "A")
        before = self._snapshot(out_dir)
        os.replace(out_dir, out_dir.with_name(out_dir.name + SUPERSEDED_SUFFIX))

        self._fail_at(monkeypatch, nth=2)
        with pytest.raises(OSError, match="配置の途中で落ちた"):
            assemble_family(self._versioned_plans("v2"), out_dir, "A")

        assert self._snapshot(out_dir) == before
        assert self._siblings(out_dir) == []


class TestCli:
    def test_it_has_no_default_pipeline_of_its_own(self) -> None:
        """既定を core に焼かない（受理集合は呼び出し側が渡す — ADR 0065 決定 2）。"""
        args = build_parser().parse_args([])
        assert args.pipeline is None
        assert args.models is None  # 解決は main（pipeline ごとの既定を引くため）

    def test_it_lists_the_choices_when_the_pipeline_is_omitted(self) -> None:
        """黙って 1 つ選ばない — 表が呼び出し側で変わる以上、既定は当てずっぽうになる。"""
        with pytest.raises(DistError, match="--pipeline が要る") as error:
            main([])
        for name in PIPELINES:
            assert name in str(error.value)

    def test_a_caller_supplied_registry_carries_its_own_default(self) -> None:
        """リポの dist ドライバは recipe を足した表と既定を渡す（受理集合はそちらが正）。"""
        args = build_parser(PIPELINES, "siglip2").parse_args([])
        assert args.pipeline == "siglip2"

    def test_the_model_flag_accumulates_into_a_family(self) -> None:
        args = build_parser().parse_args(["--model", "F1", "--model", "F2"])
        assert args.models == ["F1", "F2"]

    def test_it_knows_every_pipeline(self) -> None:
        assert sorted(PIPELINES) == [
            "birefnet",
            "depth-anything",
            "siglip2",
            "vowel-detector",
        ]
        assert PIPELINES["siglip2"].default_model == SIGLIP2_DEFAULT_MODEL
        assert PIPELINES["birefnet"].default_model == BIREFNET_DEFAULT_MODEL
        assert PIPELINES["vowel-detector"].default_model == VOWEL_DETECTOR_DEFAULT_MODEL
        assert PIPELINES["depth-anything"].default_model == DEPTH_ANYTHING_DEFAULT_MODEL

    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert default_out_dir(PIPELINES["siglip2"], ["so400m"]).name == "karume-siglip2-so400m"

    def test_it_refuses_to_invent_a_family_repository_name(self) -> None:
        """ファミリーリポの名前（例 `karume-sbv2-jvnv`）はモデル名の並びからは決まらない。"""
        with pytest.raises(DistError, match="--out"):
            default_out_dir(PIPELINES["siglip2"], ["base", "so400m"])

    def test_every_pipeline_renders_its_own_model_card(self) -> None:
        """カードは pipeline ごとのテンプレート — 描き手が他 pipeline の manifest を拒む。"""
        for name, spec in PIPELINES.items():
            manifest = {"models": {"m": {"pipeline": f"{name}/0"}}}
            for render_card in spec.card_profiles.values():
                with pytest.raises(ValueError):
                    render_card(manifest, "hdae/x")


#: 帰属が 2 通りある合成 pipeline（描き手は名前ごとに別の関数オブジェクト）。
#:
#: NOTE: 実在の 2 プロファイル family（SBV2）は wheel の外へ出たが、**選ばせる規則は core の
#: 責務**（{@link resolve_card_renderer}）なので、規則の側は表の形だけで観測する。実物の表が
#: 2 つ持っていることは `tools/export-recipes/sbv2/tests/test_distribution.py` が見る。
_TWO_PROFILE_PIPELINE = Pipeline(
    default_model="m",
    repo_name=lambda model: f"karume-{model}",
    plan=lambda series_dir, model: _synthetic_plan(model, "w/model.safetensors", b"w"),
    card_profiles={
        "fn": lambda manifest, repo: "fn",
        "jvnv": lambda manifest, repo: "jvnv",
    },
)


class TestCardProfile:
    """帰属プロファイルの選択（`--card-profile`）— 誤帰属は配ってからでないと気づけない。"""

    def test_it_is_unset_until_asked_for(self) -> None:
        assert build_parser().parse_args([]).card_profile is None
        assert build_parser().parse_args(["--card-profile", "jvnv"]).card_profile == "jvnv"

    def test_a_pipeline_with_one_profile_needs_no_choice(self) -> None:
        """siglip2 の帰属は 1 通りしかない（選びようがないものを聞かない）。"""
        profiles = PIPELINES["siglip2"].card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINES["siglip2"], None) is next(iter(profiles.values()))

    def test_it_refuses_to_pick_an_attribution_when_several_exist(self) -> None:
        """既定を黙って選ぶと、新しいファミリーへ前のファミリーの帰属がそのまま残る。"""
        with pytest.raises(DistError, match="--card-profile") as error:
            resolve_card_renderer(_TWO_PROFILE_PIPELINE, None)
        assert "fn" in str(error.value)
        assert "jvnv" in str(error.value)

    def test_it_refuses_a_profile_it_does_not_have(self) -> None:
        with pytest.raises(DistError, match="jvnv"):
            resolve_card_renderer(_TWO_PROFILE_PIPELINE, "FN9")

    def test_it_resolves_each_name_to_its_own_renderer(self) -> None:
        """名前ごとに別の描き手（束ね違いなら 2 つのファミリーが同じカードを描く）。"""
        profiles = _TWO_PROFILE_PIPELINE.card_profiles
        assert sorted(profiles) == ["fn", "jvnv"]
        assert resolve_card_renderer(_TWO_PROFILE_PIPELINE, "jvnv") is profiles["jvnv"]
        assert profiles["fn"] is not profiles["jvnv"]


# ---- SigLIP2（image-feature-extraction）--------------------------------------
#
# 偽資産は**実物と違う数**にする（96×64 の非正方・mean/std は 0.5 でない・hidden 7）— 前処理
# 定数や寸法を焼き込んでいれば落ちる。非正方なのは、`imageWidth` / `imageHeight` の取り違えが
# 正方形では原理的に検出できないため。

#: `pipelineConfig` の欄名（TS 側 `packages/models/src/siglip2/config.ts` の `ROOT_KEYS` の写し）。
#: **ロード側は未知キーも欠落も parse 時に落とす**ので、焼く側とロード側の欄名は完全一致が要る。
_SIGLIP2_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "hiddenDim",
    "interpolation",
)

_SIGLIP2_HEIGHT = 96
_SIGLIP2_WIDTH = 64
_SIGLIP2_HIDDEN = 7

#: 偽の `preprocessor_config.json`（実物と同じ欄・違う値）。
_SIGLIP2_PREPROCESSOR: Mapping[str, Any] = {
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.1, 0.2, 0.3],
    "image_std": [0.4, 0.5, 0.6],
    "image_processor_type": "SiglipImageProcessor",
    "resample": 2,
    "rescale_factor": 1.0 / 255.0,
    "size": {"height": _SIGLIP2_HEIGHT, "width": _SIGLIP2_WIDTH},
}


def _siglip2_graph(
    *,
    shape: Sequence[Any] = (1, 3, _SIGLIP2_HEIGHT, _SIGLIP2_WIDTH),
    hidden: int = _SIGLIP2_HIDDEN,
    name: str = "pixel_values",
    outputs: int = 1,
    symbols: Sequence[str] = (),
) -> str:
    """門が読む最小の IR メタデータ（入力 1 本・出力の宣言・記号次元）。"""
    names = [f"out_{index}" for index in range(outputs)]
    return json.dumps(
        {
            "inputs": [{"name": name, "shape": list(shape)}],
            "outputs": names,
            "values": {output: {"dtype": "f32", "shape": [1, hidden]} for output in names},
            "symbols": list(symbols),
        }
    )


def _build_siglip2_sources(
    root: Path,
    *,
    model: str = SIGLIP2_DEFAULT_MODEL,
    graph: str | None = None,
    dtype: str = "F32",
    preprocessor: Mapping[str, Any] | None = _SIGLIP2_PREPROCESSOR,
) -> Siglip2Sources:
    """系列 + 実重みの置き場を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `karume.paths` の実レイアウト（`outputs/series/` と `inputs/`）に揃える — CLI 経路の
    テストが root を差し替えるだけで同じ木を指せる形。
    """
    checkpoint = siglip2_checkpoint(model)
    sources = Siglip2Sources(
        series=root / "outputs" / "series" / checkpoint,
        model=root / "inputs" / "siglip2" / checkpoint,
    )
    _write(
        sources.series / "model.safetensors",
        _fake_safetensors(
            dtype,
            b"siglip2-vision-weights",
            {IR_METADATA_KEY: graph if graph is not None else _siglip2_graph()},
        ),
    )
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    if preprocessor is not None:
        _write(
            sources.model / "preprocessor_config.json",
            json.dumps(preprocessor, ensure_ascii=False).encode("utf-8"),
        )
    return sources


@pytest.fixture
def siglip2_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_siglip2_sources(tmp_path)
    out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
    manifest = assemble_family(
        [siglip2_plan(sources, SIGLIP2_DEFAULT_MODEL)], out_dir, SIGLIP2_DEFAULT_MODEL
    )
    return out_dir, manifest


def _siglip2_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][SIGLIP2_DEFAULT_MODEL]


class TestSiglip2Layout:
    def test_it_places_the_single_graph_under_the_model_subtree(self, siglip2_assembled) -> None:
        out_dir, _ = siglip2_assembled
        expected = _in_subtree(SIGLIP2_DEFAULT_MODEL, SIGLIP2_OUTPUT_PATHS.values())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_the_io_fixtures(self, siglip2_assembled) -> None:
        out_dir, _ = siglip2_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, siglip2_assembled) -> None:
        """実行に要るのはグラフ 1 本だけ（tokenizer も表も無い = `assets` は空）。"""
        _, manifest = siglip2_assembled
        model = _siglip2_model(manifest)
        assert model["pipeline"] == "siglip2/1"
        assert list(model["weights"]) == [SIGLIP2_ROLE]
        assert model["assets"] == {}
        # 席は 1 つ（f32 系列 1 本きり）。dtype ラベルは自動補完で完全写像になる。
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {SIGLIP2_ROLE: "f32"}
        assert model["quants"]["f32"]["session"] == {}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_siglip2_sources(tmp_path)
        out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
        first = assemble_family([siglip2_plan(sources)], out_dir, SIGLIP2_DEFAULT_MODEL)
        assert first == assemble_family([siglip2_plan(sources)], out_dir, SIGLIP2_DEFAULT_MODEL)
        assert verify_dist(out_dir)

    def test_it_refuses_a_compressed_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """格納形は系列ディレクトリ名でなくヘッダが正（`--dtype` 付け忘れの逆向き）。"""
        sources = _build_siglip2_sources(tmp_path, dtype="F16")
        with pytest.raises(DistError, match="F32 が無い"):
            siglip2_plan(sources)


class TestSiglip2PipelineConfig:
    """`pipelineConfig` はロード側（`src/siglip2/config.ts`）のスキーマと欄名まで一致する。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, siglip2_assembled) -> None:
        _, manifest = siglip2_assembled
        assert tuple(_siglip2_model(manifest)["pipelineConfig"]) == _SIGLIP2_CONFIG_KEYS

    def test_it_derives_the_preprocessing_constants_from_the_checkpoint(
        self, siglip2_assembled
    ) -> None:
        """焼き込んでいれば実物（224 / 384・mean = std = 0.5）の数が出てくる。"""
        _, manifest = siglip2_assembled
        config = _siglip2_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == _SIGLIP2_WIDTH
        assert config["imageHeight"] == _SIGLIP2_HEIGHT
        assert config["imageMean"] == _SIGLIP2_PREPROCESSOR["image_mean"]
        assert config["imageStd"] == _SIGLIP2_PREPROCESSOR["image_std"]
        assert config["interpolation"] == "bilinear"

    def test_it_derives_the_hidden_width_from_the_exported_graph(self, siglip2_assembled) -> None:
        """`hiddenDim` の出どころはグラフの出力宣言 1 つきり（config.json は幅を持たない）。"""
        _, manifest = siglip2_assembled
        assert _siglip2_model(manifest)["pipelineConfig"]["hiddenDim"] == _SIGLIP2_HIDDEN

    def test_it_refuses_a_missing_preprocessor_config(self, tmp_path: Path) -> None:
        sources = _build_siglip2_sources(tmp_path, preprocessor=None)
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            siglip2_plan(sources)

    def test_it_refuses_a_checkpoint_that_asks_for_another_interpolation(
        self, tmp_path: Path
    ) -> None:
        """`resample` 3（BICUBIC）は TS 側に実装が無い — 黙って bilinear で通さない。"""
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, "resample": 3}
        )
        with pytest.raises(DistError, match="resample"):
            siglip2_plan(sources)

    def test_it_refuses_a_rescale_factor_the_host_cannot_express(self, tmp_path: Path) -> None:
        """TS 側は 8bit 画素を 255 で割る形で閉じている（除数は宣言に無い）。"""
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, "rescale_factor": 1.0 / 127.5}
        )
        with pytest.raises(DistError, match="rescale_factor"):
            siglip2_plan(sources)

    @pytest.mark.parametrize("flag", ["do_resize", "do_rescale", "do_normalize"])
    def test_it_refuses_a_checkpoint_that_skips_one_of_the_three_stages(
        self, tmp_path: Path, flag: str
    ) -> None:
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, flag: False}
        )
        with pytest.raises(DistError, match=flag):
            siglip2_plan(sources)

    @pytest.mark.parametrize("flag", ["do_center_crop", "do_pad"])
    def test_it_refuses_a_checkpoint_that_wants_a_crop_or_a_pad(
        self, tmp_path: Path, flag: str
    ) -> None:
        """この前処理層は crop も pad も持たない（アスペクト比を保たない伸縮 1 本）。"""
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, flag: True}
        )
        with pytest.raises(DistError, match=flag):
            siglip2_plan(sources)

    @pytest.mark.parametrize(
        "patch",
        [
            {"image_mean": [0.1, 0.2]},
            {"image_std": [0.4, 0.0, 0.6]},
            {"image_std": [0.4, "0.5", 0.6]},
            {"size": {"height": 0, "width": _SIGLIP2_WIDTH}},
        ],
    )
    def test_it_refuses_constants_outside_the_loader_value_range(
        self, tmp_path: Path, patch: Mapping[str, Any]
    ) -> None:
        """値域はロード側（`config.ts`）と同じ — std 0 は 0 除算で ±Infinity を静かに作る。"""
        sources = _build_siglip2_sources(tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, **patch})
        with pytest.raises(DistError):
            siglip2_plan(sources)


class TestSiglip2GraphGate:
    """組み立て門 — ずれても配布形としては成立してしまう組み合わせを、配置の前に落とす。"""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """前処理の寸法（preprocessor_config）と焼かれた解像度は別々に決まる。

        base の前処理 config と so400m のグラフを組み合わせても、ここが無ければ配布形は
        成立し、利用者の手元で Session の shape 検査が「どちらが正か」を伝えないまま落ちる。
        """
        sources = _build_siglip2_sources(tmp_path, graph=_siglip2_graph(shape=(1, 3, 384, 384)))
        with pytest.raises(DistError, match="前処理の寸法と焼かれた解像度が別の版"):
            siglip2_plan(sources)

    def test_it_refuses_a_graph_whose_axes_are_transposed(self, tmp_path: Path) -> None:
        """非正方の寸法だけが検出できる取り違え（`[1,3,W,H]`）。"""
        sources = _build_siglip2_sources(
            tmp_path, graph=_siglip2_graph(shape=(1, 3, _SIGLIP2_WIDTH, _SIGLIP2_HEIGHT))
        )
        with pytest.raises(DistError, match="前処理の寸法と焼かれた解像度が別の版"):
            siglip2_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """`last_hidden_state` 込みの別 export は `hiddenDim` の出どころごと別物になる。"""
        sources = _build_siglip2_sources(tmp_path, graph=_siglip2_graph(outputs=2))
        with pytest.raises(DistError, match="pooler_output 1 本だけ"):
            siglip2_plan(sources)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """解像度もパッチ数も固定なので、動かす軸は 1 本も無い。"""
        sources = _build_siglip2_sources(tmp_path, graph=_siglip2_graph(symbols=("T",)))
        with pytest.raises(DistError, match="記号次元"):
            siglip2_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """実行側は名前で束ねるので、綴りが変われば束ねられない。"""
        sources = _build_siglip2_sources(tmp_path, graph=_siglip2_graph(name="pixels"))
        with pytest.raises(DistError, match="グラフ入力"):
            siglip2_plan(sources)


class TestSiglip2ModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str) -> Path:
        """偽資産だけで CLI を 1 周回す（実重みの置き場も tmp へ寄せる）。"""
        from karume import dist

        sources = _build_siglip2_sources(tmp_path, model=model)
        monkeypatch.setattr(dist, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "siglip2",
                "--model",
                model,
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_image_embedding_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = self._run(tmp_path, monkeypatch, SIGLIP2_DEFAULT_MODEL)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: image-feature-extraction" in card
        assert "license: apache-2.0" in card
        # 格納形を変えない配布形は 4 値のどれでもない（Hub の推論に任せる）。
        assert "base_model_relation" not in card
        # text tower を持たない事実は、利用者が最初に確かめたい制約そのもの。
        assert "The text tower is not here." in card
        # 前処理は同梱（利用者が渡すのは生の RGB8）。数は manifest から降りてくる。
        assert f"resizes to {_SIGLIP2_WIDTH} × {_SIGLIP2_HEIGHT}" in card
        assert f"`pooler_output`, {_SIGLIP2_HIDDEN} f32 values, not L2-normalized" in card
        # カードは**検証を通った**配布形から描かれる。
        assert verify_dist(out_dir)

    def test_the_attribution_follows_the_model_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """帰属はモデル名から一意に決まる（選ばせる軸にしない = 取り違えようがない）。"""
        card = (self._run(tmp_path, monkeypatch, "so400m") / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {SIGLIP2_UPSTREAM['so400m']}" in card
        assert SIGLIP2_UPSTREAM["base"] not in card


class TestSiglip2Cli:
    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert (
            default_out_dir(PIPELINES["siglip2"], [SIGLIP2_DEFAULT_MODEL]).name
            == "karume-siglip2-base"
        )
        assert default_out_dir(PIPELINES["siglip2"], ["so400m"]).name == "karume-siglip2-so400m"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINES["siglip2"].card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINES["siglip2"], None) is next(iter(profiles.values()))

    def test_the_series_name_is_the_upstream_repository_name(self, tmp_path: Path) -> None:
        """系列名 / 入力素材のディレクトリ名は上流リポ名そのもの（表を 2 つ持たない）。"""
        for model, repo in SIGLIP2_UPSTREAM.items():
            sources = siglip2_sources(tmp_path, model)
            assert sources.series.name == repo.split("/", 1)[1]
            assert sources.model.name == sources.series.name

    def test_it_refuses_a_model_it_has_no_attribution_for(self, tmp_path: Path) -> None:
        """帰属表に無いモデル名は「出所を名乗れない」ので、系列を探す前に落とす。"""
        with pytest.raises(DistError, match="知らない"):
            siglip2_sources(tmp_path, "large")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from karume import dist

        sources = _build_siglip2_sources(tmp_path)
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(dist, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "siglip2", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
        expected = _in_subtree(SIGLIP2_DEFAULT_MODEL, SIGLIP2_OUTPUT_PATHS.values())
        assert sorted(verify_dist(out_dir)) == sorted(expected)


# ---- BiRefNet 系（image-segmentation）----------------------------------------
#
# 偽資産は実物と同じ 1024²（{@link BIREFNET_RESOLUTION} が他の解像度を配布から締め出すので、
# SigLIP2 のように小さな非正方へ寄せられない）。代わりに「寸法が違う」「非正方」「軸が
# 転置された」の 3 つを**落ちる側**のケースとして持つ。

#: `pipelineConfig` の欄名（TS 側 `packages/models/src/birefnet/config.ts` の `ROOT_KEYS` の
#: 写し）。**ロード側は未知キーも欠落も parse 時に落とす**ので、焼く側とロード側の欄名は
#: 完全一致が要る。
_BIREFNET_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "interpolation",
)


def _birefnet_graph(
    *,
    shape: Sequence[Any] = (1, 3, BIREFNET_RESOLUTION, BIREFNET_RESOLUTION),
    out_shape: Sequence[Any] | None = None,
    name: str = "pixel_values",
    outputs: int = 1,
    symbols: Sequence[str] = (),
) -> str:
    """門が読む最小の IR メタデータ（入力 1 本・出力の宣言・記号次元）。"""
    names = [f"out_{index}" for index in range(outputs)]
    matte = list(out_shape) if out_shape is not None else [1, 1, shape[2], shape[3]]
    return json.dumps(
        {
            "inputs": [{"name": name, "shape": list(shape)}],
            "outputs": names,
            "values": {output: {"dtype": "f32", "shape": matte} for output in names},
            "symbols": list(symbols),
        }
    )


def _build_birefnet_sources(
    root: Path,
    *,
    model: str = BIREFNET_DEFAULT_MODEL,
    graph: str | None = None,
    dtype: str = "F32",
) -> BirefnetSources:
    """系列を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `karume.paths` の実レイアウト（`outputs/series/`）に揃える — CLI 経路のテストが
    root を差し替えるだけで同じ木を指せる形。
    """
    sources = BirefnetSources(series=root / "outputs" / "series" / birefnet_series_name(model))
    _write(
        sources.series / "model.safetensors",
        _fake_safetensors(
            dtype,
            b"birefnet-matte-weights",
            {IR_METADATA_KEY: graph if graph is not None else _birefnet_graph()},
        ),
    )
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    return sources


@pytest.fixture
def birefnet_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_birefnet_sources(tmp_path)
    out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_DEFAULT_MODEL)
    manifest = assemble_family(
        [birefnet_plan(sources, BIREFNET_DEFAULT_MODEL)], out_dir, BIREFNET_DEFAULT_MODEL
    )
    return out_dir, manifest


def _birefnet_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][BIREFNET_DEFAULT_MODEL]


class TestBirefnetLayout:
    def test_it_places_the_single_graph_under_the_model_subtree(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        expected = _in_subtree(BIREFNET_DEFAULT_MODEL, BIREFNET_OUTPUT_PATHS.values())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_the_io_fixtures(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, birefnet_assembled) -> None:
        """実行に要るのはグラフ 1 本だけ（tokenizer も表も無い = `assets` は空）。"""
        _, manifest = birefnet_assembled
        model = _birefnet_model(manifest)
        assert model["pipeline"] == "birefnet/1"
        assert list(model["weights"]) == [BIREFNET_ROLE]
        assert model["assets"] == {}
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {BIREFNET_ROLE: "f32"}
        assert model["quants"]["f32"]["session"] == {}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(tmp_path)
        out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_DEFAULT_MODEL)
        first = assemble_family([birefnet_plan(sources)], out_dir, BIREFNET_DEFAULT_MODEL)
        assert first == assemble_family([birefnet_plan(sources)], out_dir, BIREFNET_DEFAULT_MODEL)
        assert verify_dist(out_dir)

    def test_it_refuses_a_compressed_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """格納形は系列ディレクトリ名でなくヘッダが正（`--dtype` 付け忘れの逆向き）。"""
        sources = _build_birefnet_sources(tmp_path, dtype="F16")
        with pytest.raises(DistError, match="F32 が無い"):
            birefnet_plan(sources)


class TestBirefnetPipelineConfig:
    """`pipelineConfig` はロード側（`src/birefnet/config.ts`）のスキーマと欄名まで一致する。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, birefnet_assembled) -> None:
        _, manifest = birefnet_assembled
        assert tuple(_birefnet_model(manifest)["pipelineConfig"]) == _BIREFNET_CONFIG_KEYS

    def test_it_derives_the_resize_target_from_the_exported_graph(self, birefnet_assembled) -> None:
        """resize 先の出どころは焼かれたグラフ 1 つきり（上流に前処理 config が無い）。"""
        _, manifest = birefnet_assembled
        config = _birefnet_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == BIREFNET_RESOLUTION
        assert config["imageHeight"] == BIREFNET_RESOLUTION

    def test_it_declares_the_upstream_normalization_constants(self, birefnet_assembled) -> None:
        """正規化定数は機械可読な出どころが無いので dist が宣言として持つ（節の冒頭）。"""
        _, manifest = birefnet_assembled
        config = _birefnet_model(manifest)["pipelineConfig"]
        assert config["imageMean"] == list(BIREFNET_IMAGE_MEAN)
        assert config["imageStd"] == list(BIREFNET_IMAGE_STD)
        assert config["interpolation"] == "bilinear"


class TestBirefnetGraphGate:
    """組み立て門 — ずれても配布形としては成立してしまう組み合わせを、配置の前に落とす。"""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """配るのは 1024² だけ（2048² は実行段が未実測 — docs/limitations.md）。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(shape=(1, 3, 512, 512)))
        with pytest.raises(DistError, match="配るのは"):
            birefnet_plan(sources)

    def test_it_refuses_a_graph_whose_input_is_not_square(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path, graph=_birefnet_graph(shape=(1, 3, BIREFNET_RESOLUTION, 512))
        )
        with pytest.raises(DistError, match="配るのは"):
            birefnet_plan(sources)

    def test_it_refuses_a_graph_with_another_channel_count(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path,
            graph=_birefnet_graph(shape=(1, 4, BIREFNET_RESOLUTION, BIREFNET_RESOLUTION)),
        )
        with pytest.raises(DistError, match="batch もチャネル数も静的"):
            birefnet_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """multi-scale supervision 込みの export は、位置で引く後段が別の値を α として読む。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(outputs=2))
        with pytest.raises(DistError, match="マット 1 本"):
            birefnet_plan(sources)

    def test_it_refuses_a_matte_that_is_not_one_channel(self, tmp_path: Path) -> None:
        """要素数だけ見る実装なら通ってしまう形（`[1, 3, S, S]` = 中間予測 3 枚）。"""
        sources = _build_birefnet_sources(
            tmp_path,
            graph=_birefnet_graph(out_shape=[1, 3, BIREFNET_RESOLUTION, BIREFNET_RESOLUTION]),
        )
        with pytest.raises(DistError, match="入力と同じ寸法の 1 チャネル"):
            birefnet_plan(sources)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """解像度も窓マスクも定数として焼かれているので、動かす軸は 1 本も無い。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(symbols=("T",)))
        with pytest.raises(DistError, match="記号次元"):
            birefnet_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """実行側は名前で束ねるので、綴りが変われば束ねられない。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(name="pixels"))
        with pytest.raises(DistError, match="グラフ入力"):
            birefnet_plan(sources)


class TestBirefnetModelCard:
    def _run(self, tmp_path: Path, model: str) -> Path:
        """偽資産だけで CLI を 1 周回す。"""
        sources = _build_birefnet_sources(tmp_path, model=model)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "birefnet",
                "--model",
                model,
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_background_removal_distribution(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path, BIREFNET_DEFAULT_MODEL)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: image-segmentation" in card
        assert "license: mit" in card
        # 格納形を変えない配布形は 4 値のどれでもない（Hub の推論に任せる）。
        assert "base_model_relation" not in card
        # 合成をこちらでしない事実は、利用者が最初に確かめたい制約そのもの。
        assert "**Compositing is yours.**" in card
        assert f"resizes to {BIREFNET_RESOLUTION} × {BIREFNET_RESOLUTION}" in card
        assert "one alpha byte per pixel" in card
        # カードは**検証を通った**配布形から描かれる。
        assert verify_dist(out_dir)

    def test_the_attribution_follows_the_model_name(self, tmp_path: Path) -> None:
        """帰属はモデル名から一意に決まる（選ばせる軸にしない = 取り違えようがない）。"""
        card = (self._run(tmp_path, "lucida") / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert f"base_model: {BIREFNET_UPSTREAM['lucida']}" in card
        # fine-tune の元と学習データのライセンスは lucida 側だけの事実。
        assert BIREFNET_UPSTREAM["hr"] in card
        assert "ToonOut" in card
        # 前処理を焼き込んだ変種を配らないことと、その理由がカードに残る。
        assert "lucida-m35-comfy.safetensors" in card

    def test_the_default_model_card_does_not_carry_another_models_attribution(
        self, tmp_path: Path
    ) -> None:
        card = (self._run(tmp_path, BIREFNET_DEFAULT_MODEL) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {BIREFNET_UPSTREAM['hr']}" in card
        assert "ToonOut" not in card


class TestBirefnetCli:
    def test_the_default_output_directory_is_named_per_model(self) -> None:
        """リポ名は導出しない（`karume-lucida` はモデル名からは決まらない綴り）。"""
        assert (
            default_out_dir(PIPELINES["birefnet"], [BIREFNET_DEFAULT_MODEL]).name
            == "karume-birefnet-hr"
        )
        assert default_out_dir(PIPELINES["birefnet"], ["lucida"]).name == "karume-lucida"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINES["birefnet"].card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINES["birefnet"], None) is next(iter(profiles.values()))

    def test_the_series_name_carries_the_upstream_name_and_the_resolution(
        self, tmp_path: Path
    ) -> None:
        """系列名は上流リポ名 + 解像度（`export_birefnet.default_out_dir` と同じ式）。"""
        for model, repo in BIREFNET_UPSTREAM.items():
            checkpoint = repo.split("/", 1)[1].lower().replace("_", "-")
            expected = f"{checkpoint}-{BIREFNET_RESOLUTION}"
            assert birefnet_series_name(model) == expected
            assert birefnet_sources(tmp_path, model).series.name == expected

    def test_it_refuses_a_model_it_has_no_attribution_for(self, tmp_path: Path) -> None:
        """帰属表に無いモデル名は「出所を名乗れない」ので、系列を探す前に落とす。"""
        with pytest.raises(DistError, match="知らない"):
            birefnet_sources(tmp_path, "tiny")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from karume import dist

        sources = _build_birefnet_sources(tmp_path, model="lucida")
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")

        main(
            ["--pipeline", "birefnet", "--model", "lucida", "--series", str(sources.series.parent)]
        )

        out_dir = tmp_path / "models" / "karume-lucida"
        expected = _in_subtree("lucida", BIREFNET_OUTPUT_PATHS.values())
        assert sorted(verify_dist(out_dir)) == sorted(expected)


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


def _build_vowel_detector_sources(
    root: Path,
    *,
    model: str = VOWEL_DETECTOR_DEFAULT_MODEL,
    graph: str | None = None,
    feature_config: Mapping[str, Any] | None = None,
    omit_feature_config: bool = False,
    omit_graph: bool = False,
    dtype: str = "F32",
) -> VowelDetectorSources:
    """系列 1 本と上流素材を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `karume.paths` の実レイアウト（`outputs/series/` と `inputs/`）に揃える — CLI 経路の
    テストが root を差し替えるだけで同じ木を指せる形。
    """
    sources = VowelDetectorSources(
        series_dir=root / "outputs" / "series",
        model=root / "inputs" / "vowel-detector",
        model_name=model,
    )
    if not omit_graph:
        _write(
            sources.series / "model.safetensors",
            _fake_safetensors(
                dtype,
                b"vowel-detector",
                {IR_METADATA_KEY: graph if graph is not None else _vowel_detector_graph()},
            ),
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
        expected = _in_subtree(VOWEL_DETECTOR_DEFAULT_MODEL, VOWEL_DETECTOR_OUTPUT_PATHS.values())
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
        import export_vowel_detector

        assert VOWEL_DETECTOR_MAX_FRAMES == 2 * export_vowel_detector.SYM_MAX

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
        from karume import dist

        sources = _build_vowel_detector_sources(tmp_path)
        monkeypatch.setattr(dist, "INPUTS_ROOT", tmp_path / "inputs")
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
            default_out_dir(PIPELINES["vowel-detector"], [VOWEL_DETECTOR_DEFAULT_MODEL]).name
            == "karume-vowel-detector"
        )

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINES["vowel-detector"].card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINES["vowel-detector"], None) is next(
            iter(profiles.values())
        )

    def test_the_series_name_follows_the_exporter(self) -> None:
        """系列名は `export_vowel_detector.default_out_dir` と同じ式（表を 2 つ持たない）。"""
        assert (
            vowel_detector_series_name(VOWEL_DETECTOR_DEFAULT_MODEL) == "vowel-detector-crnn-epoch3"
        )

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from karume import dist

        sources = _build_vowel_detector_sources(tmp_path)
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(dist, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "vowel-detector", "--series", str(sources.series_dir)])

        out_dir = tmp_path / "models" / "karume-vowel-detector"
        expected = _in_subtree(VOWEL_DETECTOR_DEFAULT_MODEL, VOWEL_DETECTOR_OUTPUT_PATHS.values())
        assert sorted(verify_dist(out_dir)) == sorted(expected)


# ---- Depth Anything V2（depth-estimation）------------------------------------
#
# 偽資産は**実物と違う数**にする（96×64 の非正方・mean/std は ImageNet 統計でない）— 前処理
# 定数や寸法を焼き込んでいれば落ちる。非正方なのは、`imageWidth` / `imageHeight` の取り違えが
# 正方形では原理的に検出できないため。

#: `pipelineConfig` の欄名（TS 側 `packages/models/src/depth-anything/config.ts` の `ROOT_KEYS` の
#: 写し）。**ロード側は未知キーも欠落も parse 時に落とす**ので、焼く側とロード側の欄名は
#: 完全一致が要る。
_DEPTH_ANYTHING_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "interpolation",
)

_DEPTH_ANYTHING_HEIGHT = 96
_DEPTH_ANYTHING_WIDTH = 64

#: 偽の `preprocessor_config.json`（実物と同じ欄・違う値。`keep_aspect_ratio` /
#: `ensure_multiple_of` は実物どおり載せておく — 宣言へ写していないことを見るため）。
_DEPTH_ANYTHING_PREPROCESSOR: Mapping[str, Any] = {
    "do_normalize": True,
    "do_pad": False,
    "do_rescale": True,
    "do_resize": True,
    "ensure_multiple_of": 14,
    "image_mean": [0.1, 0.2, 0.3],
    "image_processor_type": "DPTImageProcessor",
    "image_std": [0.4, 0.5, 0.6],
    "keep_aspect_ratio": True,
    "resample": 3,
    "rescale_factor": 1.0 / 255.0,
    "size": {"height": _DEPTH_ANYTHING_HEIGHT, "width": _DEPTH_ANYTHING_WIDTH},
}


def _depth_anything_graph(
    *,
    shape: Sequence[Any] = (1, 3, _DEPTH_ANYTHING_HEIGHT, _DEPTH_ANYTHING_WIDTH),
    out_shape: Sequence[Any] | None = None,
    name: str = "pixel_values",
    outputs: int = 1,
    symbols: Sequence[str] = (),
) -> str:
    """門が読む最小の IR メタデータ（入力 1 本・出力の宣言・記号次元）。"""
    names = [f"out_{index}" for index in range(outputs)]
    depth = list(out_shape) if out_shape is not None else [1, shape[2], shape[3]]
    return json.dumps(
        {
            "inputs": [{"name": name, "shape": list(shape)}],
            "outputs": names,
            "values": {output: {"dtype": "f32", "shape": depth} for output in names},
            "symbols": list(symbols),
        }
    )


def _build_depth_anything_sources(
    root: Path,
    *,
    model: str = DEPTH_ANYTHING_DEFAULT_MODEL,
    graph: str | None = None,
    dtype: str = "F32",
    preprocessor: Mapping[str, Any] | None = _DEPTH_ANYTHING_PREPROCESSOR,
) -> DepthAnythingSources:
    """系列 + 実重みの置き場を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `karume.paths` の実レイアウト（`outputs/series/` と `inputs/`）に揃える — CLI 経路の
    テストが root を差し替えるだけで同じ木を指せる形。
    """
    sources = DepthAnythingSources(
        series=root / "outputs" / "series" / depth_anything_series_name(model),
        model=root / "inputs" / "depth-anything" / depth_anything_checkpoint(model),
    )
    _write(
        sources.series / "model.safetensors",
        _fake_safetensors(
            dtype,
            b"depth-anything-weights",
            {IR_METADATA_KEY: graph if graph is not None else _depth_anything_graph()},
        ),
    )
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    if preprocessor is not None:
        _write(
            sources.model / "preprocessor_config.json",
            json.dumps(preprocessor, ensure_ascii=False).encode("utf-8"),
        )
    return sources


@pytest.fixture
def depth_anything_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_depth_anything_sources(tmp_path)
    out_dir = tmp_path / "models" / depth_anything_repo_name(DEPTH_ANYTHING_DEFAULT_MODEL)
    manifest = assemble_family(
        [depth_anything_plan(sources, DEPTH_ANYTHING_DEFAULT_MODEL)],
        out_dir,
        DEPTH_ANYTHING_DEFAULT_MODEL,
    )
    return out_dir, manifest


def _depth_anything_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][DEPTH_ANYTHING_DEFAULT_MODEL]


class TestDepthAnythingLayout:
    def test_it_places_the_single_graph_under_the_model_subtree(
        self, depth_anything_assembled
    ) -> None:
        out_dir, _ = depth_anything_assembled
        expected = _in_subtree(DEPTH_ANYTHING_DEFAULT_MODEL, DEPTH_ANYTHING_OUTPUT_PATHS.values())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_the_io_fixtures(self, depth_anything_assembled) -> None:
        out_dir, _ = depth_anything_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, depth_anything_assembled) -> None:
        """実行に要るのはグラフ 1 本だけ（tokenizer も表も無い = `assets` は空）。"""
        _, manifest = depth_anything_assembled
        model = _depth_anything_model(manifest)
        assert model["pipeline"] == "depth-anything/1"
        assert list(model["weights"]) == [DEPTH_ANYTHING_ROLE]
        assert model["assets"] == {}
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {DEPTH_ANYTHING_ROLE: "f32"}
        assert model["quants"]["f32"]["session"] == {}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_depth_anything_sources(tmp_path)
        out_dir = tmp_path / "models" / depth_anything_repo_name(DEPTH_ANYTHING_DEFAULT_MODEL)
        first = assemble_family(
            [depth_anything_plan(sources)], out_dir, DEPTH_ANYTHING_DEFAULT_MODEL
        )
        assert first == assemble_family(
            [depth_anything_plan(sources)], out_dir, DEPTH_ANYTHING_DEFAULT_MODEL
        )
        assert verify_dist(out_dir)

    def test_it_refuses_a_compressed_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """格納形は系列ディレクトリ名でなくヘッダが正（`--dtype` 付け忘れの逆向き）。"""
        sources = _build_depth_anything_sources(tmp_path, dtype="F16")
        with pytest.raises(DistError, match="F32 が無い"):
            depth_anything_plan(sources)


class TestDepthAnythingPipelineConfig:
    """`pipelineConfig` はロード側（`src/depth-anything/config.ts`）のスキーマと欄名まで一致。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(
        self, depth_anything_assembled
    ) -> None:
        _, manifest = depth_anything_assembled
        assert tuple(_depth_anything_model(manifest)["pipelineConfig"]) == (
            _DEPTH_ANYTHING_CONFIG_KEYS
        )

    def test_it_derives_the_preprocessing_constants_from_the_checkpoint(
        self, depth_anything_assembled
    ) -> None:
        """焼き込んでいれば実物（518² / ImageNet 統計）の数が出てくる。"""
        _, manifest = depth_anything_assembled
        config = _depth_anything_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == _DEPTH_ANYTHING_WIDTH
        assert config["imageHeight"] == _DEPTH_ANYTHING_HEIGHT
        assert config["imageMean"] == _DEPTH_ANYTHING_PREPROCESSOR["image_mean"]
        assert config["imageStd"] == _DEPTH_ANYTHING_PREPROCESSOR["image_std"]

    def test_it_declares_bicubic_not_the_bilinear_the_other_towers_use(
        self, depth_anything_assembled
    ) -> None:
        """`resample: 3` = PIL の BICUBIC。SigLIP2 / BiRefNet の bilinear と取り違えると、
        `pixel_values` が uint8 1 LSB の 34 倍ずれたままロードも実行も通る。"""
        _, manifest = depth_anything_assembled
        assert _depth_anything_model(manifest)["pipelineConfig"]["interpolation"] == "bicubic"

    def test_it_does_not_carry_the_aspect_ratio_knobs_into_the_declaration(
        self, depth_anything_assembled
    ) -> None:
        """上流 config の `keep_aspect_ratio` / `ensure_multiple_of` は宣言へ写さない。

        焼かれたグラフは正方 1 点でしか受け取らないので、保つ経路には行き先が無い。素通しで
        写すと TS 側の parse が未知キーで落ちる（= 配布形はできるのにロードできない）。
        """
        _, manifest = depth_anything_assembled
        config = _depth_anything_model(manifest)["pipelineConfig"]
        assert "keepAspectRatio" not in config
        assert "ensureMultipleOf" not in config

    def test_it_refuses_a_missing_preprocessor_config(self, tmp_path: Path) -> None:
        sources = _build_depth_anything_sources(tmp_path, preprocessor=None)
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            depth_anything_plan(sources)

    def test_it_refuses_a_checkpoint_that_asks_for_another_interpolation(
        self, tmp_path: Path
    ) -> None:
        """`resample` 2（BILINEAR）を黙って bicubic として宣言しない。"""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, "resample": 2}
        )
        with pytest.raises(DistError, match="resample"):
            depth_anything_plan(sources)

    def test_it_refuses_a_rescale_factor_the_host_cannot_express(self, tmp_path: Path) -> None:
        """TS 側は 8bit 画素を 255 で割る形で閉じている（除数は宣言に無い）。"""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, "rescale_factor": 1.0 / 127.5}
        )
        with pytest.raises(DistError, match="rescale_factor"):
            depth_anything_plan(sources)

    @pytest.mark.parametrize("flag", ["do_resize", "do_rescale", "do_normalize"])
    def test_it_refuses_a_checkpoint_that_skips_one_of_the_three_stages(
        self, tmp_path: Path, flag: str
    ) -> None:
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, flag: False}
        )
        with pytest.raises(DistError, match=flag):
            depth_anything_plan(sources)

    @pytest.mark.parametrize("flag", ["do_center_crop", "do_pad"])
    def test_it_refuses_a_checkpoint_that_wants_a_crop_or_a_pad(
        self, tmp_path: Path, flag: str
    ) -> None:
        """この前処理層は crop も pad も持たない（伸縮 1 本）。"""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, flag: True}
        )
        with pytest.raises(DistError, match=flag):
            depth_anything_plan(sources)

    @pytest.mark.parametrize(
        "patch",
        [
            {"image_mean": [0.1, 0.2]},
            {"image_std": [0.4, 0.0, 0.6]},
            {"image_std": [0.4, "0.5", 0.6]},
            {"size": {"height": 0, "width": _DEPTH_ANYTHING_WIDTH}},
        ],
    )
    def test_it_refuses_constants_outside_the_loader_value_range(
        self, tmp_path: Path, patch: Mapping[str, Any]
    ) -> None:
        """値域はロード側（`config.ts`）と同じ — std 0 は 0 除算で ±Infinity を静かに作る。"""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, **patch}
        )
        with pytest.raises(DistError):
            depth_anything_plan(sources)


class TestDepthAnythingGraphGate:
    """組み立て門 — ずれても配布形としては成立してしまう組み合わせを、配置の前に落とす。"""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """サイズが違えば事前学習解像度も違う（Small 518² / 別サイズの組み合わせ）。"""
        sources = _build_depth_anything_sources(
            tmp_path, graph=_depth_anything_graph(shape=(1, 3, 518, 518))
        )
        with pytest.raises(DistError, match="前処理の寸法と焼かれた解像度が別の版"):
            depth_anything_plan(sources)

    def test_it_refuses_a_graph_whose_axes_are_transposed(self, tmp_path: Path) -> None:
        """非正方の寸法だけが検出できる取り違え（`[1,3,W,H]`）。"""
        sources = _build_depth_anything_sources(
            tmp_path,
            graph=_depth_anything_graph(
                shape=(1, 3, _DEPTH_ANYTHING_WIDTH, _DEPTH_ANYTHING_HEIGHT)
            ),
        )
        with pytest.raises(DistError, match="前処理の寸法と焼かれた解像度が別の版"):
            depth_anything_plan(sources)

    def test_it_refuses_a_depth_map_with_a_channel_axis(self, tmp_path: Path) -> None:
        """`[1, 1, H, W]` は要素数では `[1, H, W]` と区別できない（階数まで見る）。"""
        sources = _build_depth_anything_sources(
            tmp_path,
            graph=_depth_anything_graph(
                out_shape=(1, 1, _DEPTH_ANYTHING_HEIGHT, _DEPTH_ANYTHING_WIDTH)
            ),
        )
        with pytest.raises(DistError, match="チャネル軸なし"):
            depth_anything_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """中間段まで出す別 export は、位置で引く後段が別の値を深度として読む。"""
        sources = _build_depth_anything_sources(tmp_path, graph=_depth_anything_graph(outputs=2))
        with pytest.raises(DistError, match="深度マップ 1 本だけ"):
            depth_anything_plan(sources)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """解像度もパッチ数も固定なので、動かす軸は 1 本も無い。"""
        sources = _build_depth_anything_sources(
            tmp_path, graph=_depth_anything_graph(symbols=("T",))
        )
        with pytest.raises(DistError, match="記号次元"):
            depth_anything_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """実行側は名前で束ねるので、綴りが変われば束ねられない。"""
        sources = _build_depth_anything_sources(tmp_path, graph=_depth_anything_graph(name="px"))
        with pytest.raises(DistError, match="グラフ入力"):
            depth_anything_plan(sources)


class TestDepthAnythingModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """偽資産だけで CLI を 1 周回す（実重みの置き場も tmp へ寄せる）。"""
        from karume import dist

        sources = _build_depth_anything_sources(tmp_path)
        monkeypatch.setattr(dist, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "depth-anything",
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_relative_depth_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = self._run(tmp_path, monkeypatch)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: depth-estimation" in card
        assert f"license: {DEPTH_ANYTHING_LICENSE}" in card
        assert f"base_model: {DEPTH_ANYTHING_UPSTREAM['small']}" in card
        # 格納形を変えない配布形は 4 値のどれでもない（Hub の推論に任せる）。
        assert "base_model_relation" not in card
        # 相対深度であること（metric depth でないこと）は、利用者が最初に確かめる制約そのもの。
        assert "**Relative depth has no unit and no origin**" in card
        # 前処理は同梱（利用者が渡すのは生の RGB8）。数は manifest から降りてくる。
        assert f"resized to {_DEPTH_ANYTHING_WIDTH} × {_DEPTH_ANYTHING_HEIGHT} (bicubic" in card
        # カードは**検証を通った**配布形から描かれる。
        assert verify_dist(out_dir)

    def test_it_states_that_only_the_apache_licensed_size_is_redistributed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Base / Large は CC BY-NC 4.0。「無いこと」は読み手が最初に探す事実。"""
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "Only the Small checkpoint is Apache-2.0." in card
        assert "CC BY-NC 4.0" in card

    def test_the_title_comes_from_the_upstream_checkpoint_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """サイズの綴りは帰属表 1 本から導く（2 表にすると別サイズとして売れてしまう）。"""
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "# Depth Anything V2 Small — Karume" in card


class TestDepthAnythingCli:
    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert (
            default_out_dir(PIPELINES["depth-anything"], [DEPTH_ANYTHING_DEFAULT_MODEL]).name
            == "karume-depth-anything-v2-small"
        )

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINES["depth-anything"].card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINES["depth-anything"], None) is next(
            iter(profiles.values())
        )

    def test_the_series_name_is_the_lowercased_upstream_repository_name(
        self, tmp_path: Path
    ) -> None:
        """系列名 / 入力素材のディレクトリ名は上流リポ名そのもの（表を 2 つ持たない）。"""
        for model, repo in DEPTH_ANYTHING_UPSTREAM.items():
            checkpoint = repo.split("/", 1)[1]
            sources = depth_anything_sources(tmp_path, model)
            assert sources.series.name == checkpoint.lower()
            assert sources.model.name == checkpoint

    @pytest.mark.parametrize("model", ["base", "large"])
    def test_it_refuses_a_size_it_may_not_redistribute(self, tmp_path: Path, model: str) -> None:
        """上流で CC BY-NC 4.0 のサイズは帰属表に載っていない = 系列を探す前に落ちる。

        台本は `--model-dir` でどのサイズでも焼けるので、「焼けたものは配れる」と読める形に
        しないための唯一の門（`karume.dist` の Depth Anything 節の MUST）。
        """
        with pytest.raises(DistError, match="知らない"):
            depth_anything_sources(tmp_path, model)

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from karume import dist

        sources = _build_depth_anything_sources(tmp_path)
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(dist, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "depth-anything", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / depth_anything_repo_name(DEPTH_ANYTHING_DEFAULT_MODEL)
        expected = _in_subtree(DEPTH_ANYTHING_DEFAULT_MODEL, DEPTH_ANYTHING_OUTPUT_PATHS.values())
        assert sorted(verify_dist(out_dir)) == sorted(expected)
