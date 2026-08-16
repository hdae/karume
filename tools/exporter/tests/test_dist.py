"""配布ディレクトリ組み立て（`karume.dist`）の単体テスト。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合だけ**（格納形の門）
なので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

NOTE: family 別の配布 recipe は 1 つ残らず wheel の外へ出た（ADR 0065 段 3+4 完了）ので、その
依存ケースは `tools/export-recipes/<family>/tests/test_distribution.py` に居る。ここに残るのは
core だけで観測できる層だけ — 合成計画で足りる規模上限・quant 完全写像・staging/swap の
不変条件・帰属プロファイルの解決規則・受理集合を呼び出し側から受ける規約。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from karume import dist
from karume.dist import (
    MANIFEST_FORMAT,
    PIPELINES,
    SHARED_DIRNAME,
    SUPERSEDED_SUFFIX,
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assemble_family,
    assert_manifest_limits,
    assert_model_name,
    build_parser,
    complete_quant_weights,
    main,
    materialize,
    resolve_card_renderer,
    verify_dist,
)


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
        """実在の recipe が使う綴りの形（英小文字 / 数字 / ハイフン / ドット）は通す。

        NOTE: 定数を recipe から import してこない — core のテストが family へ逆流する形に
        しない（ADR 0065 決定 3）。綴りが実物と噛み合っていることは各 recipe の
        `tests/test_distribution.py` が自分の定数で見る。
        """
        for name in ("small", "base", "so400m", "v4-small", "crnn-epoch3", "jvnv-F1"):
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

    @staticmethod
    def _record_source_digests(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """出所 hash を採った席の `(モデル名, 相対 path)` を記録する。

        「読まれなかった」は現物を並べただけでは観測できない（結果のツリーは読んでも読まなくても
        同じ）ので、共有判定が使う唯一の読み口を計装して回数で見る。
        """
        real = dist._source_digest
        taken: list[str] = []

        def recording(artifact: Artifact, memo: dict[Path, str]) -> str:
            taken.append(artifact.rel_path)
            return real(artifact, memo)

        monkeypatch.setattr("karume.dist._source_digest", recording)
        return taken

    def test_it_settles_a_path_two_sizes_claim_without_reading_either(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """同じ相対 path でもサイズが違えば中身も必ず違う — hash を採る前に非共有が決まる。"""
        rel_path = "text_encoder/model.safetensors"
        short, long = b"weights-A", b"weights-BB"
        assert len(short) != len(long)
        taken = self._record_source_digests(monkeypatch)
        plans = [
            _synthetic_plan(name, rel_path, payload)
            for name, payload in (("A", short), ("B", long))
        ]
        out_dir = tmp_path / "models" / "sized"
        manifest = assemble_family(plans, out_dir, "A")

        assert taken == []
        for name, payload in (("A", short), ("B", long)):
            ref = manifest["models"][name]["weights"]["w"]["f16"]["file"]
            assert ref["path"] == f"{name}/{rel_path}", name
            assert (out_dir / ref["path"]).read_bytes() == payload, name
        assert not (out_dir / SHARED_DIRNAME).exists()

    def test_it_still_reads_the_source_to_separate_two_equal_sized_byte_strings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """サイズが同じ中身違いは前置では落とせない — 従来どおり出所 sha256 が弁別する。"""
        rel_path = "text_encoder/model.safetensors"
        first, second = b"weights-D1", b"weights-D2"
        assert len(first) == len(second)
        taken = self._record_source_digests(monkeypatch)
        plans = [
            _synthetic_plan(name, rel_path, payload)
            for name, payload in (("A", first), ("B", second))
        ]
        out_dir = tmp_path / "models" / "equal-sized"
        manifest = assemble_family(plans, out_dir, "A")

        assert taken == [rel_path, rel_path]
        for name, payload in (("A", first), ("B", second)):
            ref = manifest["models"][name]["weights"]["w"]["f16"]["file"]
            assert ref["path"] == f"{name}/{rel_path}", name
            assert (out_dir / ref["path"]).read_bytes() == payload, name
        assert not (out_dir / SHARED_DIRNAME).exists()

    def test_it_folds_the_matching_pair_while_a_differently_sized_seat_looks_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """前置フィルタは**畳める組**に一切触らない — 落ちるのはサイズ違いの単独だけ。

        同じ相対 path に「畳める組」と「サイズ違いの単独」が同居する形が、前置を相対 path
        ごとに掛けてしまった場合（= 組まで巻き添えに落ちる）と結果が割れる唯一の形。
        """
        rel_path = "text_encoder/model.safetensors"
        shared_bytes, odd = b"same-bytes", b"a-longer-byte-string"
        assert len(shared_bytes) != len(odd)
        taken = self._record_source_digests(monkeypatch)
        plans = [
            _synthetic_plan(name, rel_path, payload)
            for name, payload in (("A", shared_bytes), ("B", shared_bytes), ("C", odd))
        ]
        out_dir = tmp_path / "models" / "mixed"
        manifest = assemble_family(plans, out_dir, "A")

        assert taken == [rel_path, rel_path]
        target = f"{SHARED_DIRNAME}/{rel_path}"
        for name in ("A", "B"):
            ref = manifest["models"][name]["weights"]["w"]["f16"]["file"]
            assert ref["path"] == target, name
            assert ref["sha256"] == hashlib.sha256(shared_bytes).hexdigest(), name
        loner = manifest["models"]["C"]["weights"]["w"]["f16"]["file"]
        assert loner["path"] == f"C/{rel_path}"
        assert (out_dir / loner["path"]).read_bytes() == odd
        assert (out_dir / target).read_bytes() == shared_bytes
        assert sorted(verify_dist(out_dir)) == sorted([target, f"C/{rel_path}"])

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

    def test_a_failing_evacuation_leaves_the_previous_distribution_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """据え替えの 1 回目（出力先 → 退避先）で落ちても、os.replace は原子的なので出力先は
        無傷のまま残る。"""
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans("v1"), out_dir, "A")
        before = self._snapshot(out_dir)

        self._fail_replace_at(monkeypatch, nth=1)
        with pytest.raises(DistError, match="退避") as failure:
            assemble_family(self._versioned_plans("v2"), out_dir, "A")

        # 原因（I/O 故障）は連鎖で残す — 退避の失敗と組み立ての失敗を取り違えない。
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


#: 帰属が 1 通りだけの合成 pipeline（実在 family の代わりに「表の形」だけを持つ被験体）。
#:
#: NOTE: 実物の recipe は 1 つ残らず wheel の外なので、**core が recipe を import して被験体に
#: する形は取らない**（ADR 0065 決定 3 — 依存は recipe → core の一方向）。ここで観測するのは
#: 「渡された表をどう扱うか」という core 側の規則だけで、実物の表の中身は各 recipe の
#: `tests/test_distribution.py` と repo driver の `tests/test_dist_driver.py` が見る。
_ONE_PROFILE_PIPELINE = Pipeline(
    default_model="m",
    repo_name=lambda model: f"karume-solo-{model}",
    plan=lambda series_dir, model: _synthetic_plan(model, "w/model.safetensors", b"w"),
    card_profiles={"solo": lambda manifest, repo: "solo"},
)

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

#: 呼び出し側が渡す表の代わり（repo driver が渡すものの合成版）。
_CALLER_REGISTRY: Mapping[str, Pipeline] = {
    "solo": _ONE_PROFILE_PIPELINE,
    "duo": _TWO_PROFILE_PIPELINE,
}


class TestCli:
    def test_it_has_no_default_pipeline_of_its_own(self) -> None:
        """既定を core に焼かない（受理集合は呼び出し側が渡す — ADR 0065 決定 2）。"""
        args = build_parser().parse_args([])
        assert args.pipeline is None
        assert args.models is None  # 解決は main（pipeline ごとの既定を引くため）

    def test_the_core_table_is_empty(self) -> None:
        """family 別 recipe は 1 つ残らず wheel の外（ADR 0065 段 3+4 完了）。

        `PIPELINES` が空でないなら、core wheel が family 知識を持ち直したということ。
        """
        assert PIPELINES == {}

    def test_it_still_fails_loudly_with_an_empty_table(self) -> None:
        """既定 = 空の表でも文言が壊れない（選択肢が 0 個でも「要る」と言い切る）。

        MUST: 空欄で黙って通らない — 表を渡し忘れた呼び出しは、資産を 1 バイトも触る前に
        ここで落ちるのが唯一の検出点。
        """
        with pytest.raises(DistError, match="--pipeline が要る") as error:
            main([])
        assert "選択肢: " in str(error.value)
        assert "tools/export-recipes/dist.py" in str(error.value)

    @pytest.mark.parametrize("name", ["siglip2", "anima", ""])
    def test_an_empty_table_accepts_no_pipeline_name(self, name: str) -> None:
        """受理集合が空なら**どの名前も**通らない（argparse の choices が空集合）。"""
        with pytest.raises(SystemExit) as raised:
            build_parser().parse_args(["--pipeline", name])
        assert raised.value.code == 2

    def test_a_caller_supplied_registry_carries_its_own_default(self) -> None:
        """リポの dist ドライバは recipe を足した表と既定を渡す（受理集合はそちらが正）。"""
        args = build_parser(_CALLER_REGISTRY, "solo").parse_args([])
        assert args.pipeline == "solo"

    def test_a_caller_supplied_registry_widens_the_accepted_names(self) -> None:
        """渡された表の名前だけが通る（core の空の表とは独立）。"""
        args = build_parser(_CALLER_REGISTRY, "solo").parse_args(["--pipeline", "duo"])
        assert args.pipeline == "duo"

    def test_the_model_flag_accumulates_into_a_family(self) -> None:
        args = build_parser().parse_args(["--model", "F1", "--model", "F2"])
        assert args.models == ["F1", "F2"]

    def test_it_has_no_place_of_its_own_to_write_into(self) -> None:
        """置き場の既定も呼び出し側が渡す（`models/` / `outputs/series/` は repo topology）。"""
        args = build_parser().parse_args([])
        assert args.out is None
        assert args.series is None

    def test_it_refuses_to_assemble_without_a_series_root(self) -> None:
        """MUST: 渡されなければ落ちる — 黙って cwd 相対のどこかを系列として読まない。"""
        with pytest.raises(DistError, match="--series が要る"):
            main(["--pipeline", "solo"], pipelines=_CALLER_REGISTRY)

    def test_it_refuses_to_assemble_without_an_output_directory(self, tmp_path: Path) -> None:
        """`--out` の既定を作る hook を渡さない呼び出しは、資産を触る前にここで落ちる。"""
        with pytest.raises(DistError, match="--out が要る"):
            main(["--pipeline", "solo", "--series", str(tmp_path)], pipelines=_CALLER_REGISTRY)

    def test_a_caller_supplied_hook_decides_where_the_default_lands(self, tmp_path: Path) -> None:
        """hook を渡した呼び出しは `--out` 省略で通り、**hook が返した場所**へ組み上がる。

        hook が受け取るのは解決済みの pipeline とモデル名の並び（リポ名を綴るのに要る全部）。
        """
        seen: list[tuple[Pipeline, list[str]]] = []

        def hook(pipeline: Pipeline, models: Sequence[str]) -> Path:
            seen.append((pipeline, list(models)))
            return tmp_path / "models" / "karume-solo-m"

        main(
            ["--pipeline", "solo", "--series", str(tmp_path)],
            pipelines=_CALLER_REGISTRY,
            default_out_dir=hook,
        )

        assert seen == [(_ONE_PROFILE_PIPELINE, ["m"])]
        assert verify_dist(tmp_path / "models" / "karume-solo-m")


class TestCardProfile:
    """帰属プロファイルの選択（`--card-profile`）— 誤帰属は配ってからでないと気づけない。"""

    def test_it_is_unset_until_asked_for(self) -> None:
        assert build_parser().parse_args([]).card_profile is None
        assert build_parser().parse_args(["--card-profile", "jvnv"]).card_profile == "jvnv"

    def test_a_pipeline_with_one_profile_needs_no_choice(self) -> None:
        """帰属が 1 通りしかない pipeline には聞かない（選びようがない）。"""
        profiles = _ONE_PROFILE_PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(_ONE_PROFILE_PIPELINE, None) is next(iter(profiles.values()))

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
