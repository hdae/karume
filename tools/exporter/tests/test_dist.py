"""配布ディレクトリ組み立て（`karume.dist`）の単体テスト。

実資産は使わない。ただし weights の席へ挿す入力は**正当な IR コンテナ**でなければならない —
組み立ては配置の前に入力コンテナを IR v1 の全規則で見る
（`dist.assert_weight_components_verified`）。その合成は {@link ir_fixtures}（数 KB のグラフ
1 本ぶん）で、生成物（`payload`）で持つ席は IR コンテナではないので従来どおり生バイト列でよい。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

NOTE: family 別の配布 recipe は 1 つ残らず wheel の外へ出た（ADR 0065 段 3+4 完了）ので、その
依存ケースは `tools/export-recipes/<family>/tests/test_distribution.py` に居る。ここに残るのは
core だけで観測できる層だけ — 合成計画で足りる規模上限・quant 完全写像・staging/swap の
不変条件・帰属プロファイルの解決規則・受理集合を呼び出し側から受ける規約。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest
from ir_fixtures import ir_container, ir_shards
from safetensors.numpy import load, save

from karume import dist, limits
from karume.artifacts import SUPERSEDED_SUFFIX
from karume.dist import (
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MAX_QUANT_DESCRIPTION_CHARS,
    MAX_QUANT_LABEL_CHARS,
    MAX_SHARDS,
    PIPELINES,
    SHARED_DIRNAME,
    Artifact,
    DistError,
    ExternalComponents,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assemble_family,
    assert_manifest_limits,
    assert_model_name,
    assert_quant_presentation,
    build_parser,
    complete_quant_weights,
    is_external_ref,
    main,
    manifest_text,
    materialize,
    preprocessor_channels,
    resolve_card_renderer,
    resolve_external_components,
    safetensors_header,
    verify_dist,
)
from karume.ir import IR_METADATA_KEY
from karume.shards import parse_piece_key, shard_name


def _write_series(root: Path, payloads: Sequence[bytes], name: str = "model.safetensors") -> Path:
    """shard 列を系列ディレクトリへ連番で書き、**代表 path** を返す。

    配布形は常に分割される（ADR 0081）ので、系列側の現物も常に連番 — 代表 path 自身は
    書かない（書くと「単一ファイルと連番の同居」になり、組み立てが fail loudly する）。
    """
    root.mkdir(parents=True, exist_ok=True)
    total = len(payloads)
    for index, payload in enumerate(payloads, start=1):
        (root / shard_name(name, index, total)).write_bytes(payload)
    return root / name


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
                "session": {"linearCompute": "a8"},
                "gpuFeatures": {"shaderF16": True},
            }
        }
        completed = complete_quant_weights(self._weights, quants)
        assert completed["w8a8"]["session"] == {"linearCompute": "a8"}
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
            "weights": {"w": {"f16": {"shards": [_ref("m/w.safetensors")]}}},
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


class TestShardDeclaration:
    """`karume/4` の weights は dtype ごとに **shard 列**を持つ（ADR 0070 決定 1 の欄）。

    ここが見るのは**単一コンテナ**（= 上限以下で分割されなかった資産）の側 — 列は 1 要素で、
    `karume_ir` を持つコンテナそのもの = 先頭のグラフ shard。複数要素になる側は
    {@link TestShardExpansion}。
    """

    _PAYLOAD = b"weights-A"

    def _assemble(self, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
        out_dir = tmp_path / "models" / "sharded"
        plans = [_synthetic_plan("A", "w/model.safetensors", self._PAYLOAD)]
        return out_dir, assemble_family(plans, out_dir, "A")

    def _rewrite_weights(self, out_dir: Path, entry: Any) -> None:
        """据わった配布形の manifest だけを別形に差し替える（現物はそのまま）。"""
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        manifest["models"]["A"]["weights"]["w"]["f16"] = entry
        (out_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def test_the_format_identifier_names_the_fourth_manifest_version(self) -> None:
        """形式識別子は hub と 1 文字も違えられない（hub は 1 形しか読まない）。

        `karume/4` は quant の表示欄（ADR 0075 決定 1）を足した形 — 欄は optional でも hub の
        allowlist が未知キーを拒否するので、major を上げずに足すと旧クライアントが黙って
        読めなくなる（ADR 0075 決定 4）。
        """
        assert MANIFEST_FORMAT == "karume/4"

    def test_it_declares_the_container_as_a_one_element_shard_list(self, tmp_path: Path) -> None:
        out_dir, manifest = self._assemble(tmp_path)
        entry = manifest["models"]["A"]["weights"]["w"]["f16"]

        # v2 の `file` は残っていない（2 形が同居すると hub が「どちらを読むか」を持つ）。
        assert list(entry) == ["shards"]
        assert entry["shards"] == [
            {
                "path": "A/w/model.safetensors",
                "size": len(self._PAYLOAD),
                "sha256": hashlib.sha256(self._PAYLOAD).hexdigest(),
            }
        ]
        assert (out_dir / entry["shards"][0]["path"]).read_bytes() == self._PAYLOAD

    def test_the_shard_is_covered_by_the_declaration_check(self, tmp_path: Path) -> None:
        """突合は shard 列を辿って現物へ届く（列に移して素通りしはじめると宣言外扱いになる）。"""
        out_dir, _ = self._assemble(tmp_path)
        assert verify_dist(out_dir) == {"A/w/model.safetensors": len(self._PAYLOAD)}

    def test_it_refuses_a_manifest_that_kept_the_v2_single_file_form(self, tmp_path: Path) -> None:
        """形式識別子だけ v3 で中身が `{file}` の manifest は、hub が読めないのでここで落とす。"""
        out_dir, manifest = self._assemble(tmp_path)
        ref = manifest["models"]["A"]["weights"]["w"]["f16"]["shards"][0]
        self._rewrite_weights(out_dir, {"file": ref})
        with pytest.raises(DistError, match="shards が"):
            verify_dist(out_dir)

    @pytest.mark.parametrize("shards", [[], "A/w/model.safetensors"])
    def test_it_refuses_a_shard_list_that_is_not_a_non_empty_array(
        self, tmp_path: Path, shards: Any
    ) -> None:
        """空の列（= 重みを 1 本も指さない dtype 席）も、列ですらない値も受理しない。"""
        out_dir, _ = self._assemble(tmp_path)
        self._rewrite_weights(out_dir, {"shards": shards})
        with pytest.raises(DistError, match=f"1〜{MAX_SHARDS} 要素の配列でない"):
            verify_dist(out_dir)


class TestDtypeLabelVocabulary:
    """weights の dtype ラベルは runtime の**格納 dtype 語彙**の内側
    （{@link karume.dist.STORAGE_DTYPE_LABELS}）。

    ラベルは manifest の自由キーで hub も文字列としか見ないが、モデルカードは
    「`I4` は公式 safetensors ライブラリで開けない」注記を**ラベルの綴り**で出すか出さないか
    決める（`modelcard.quants`）。語彙を縛らないと、i4 を含む配布形をラベル `w4` で綴った
    家族で注記が黙って消え、カード自身の「ラベルは格納 dtype 語彙」という主張も裏づけを失う。
    """

    def _distribution(self, tmp_path: Path, label: str) -> Path:
        """据わった配布形の manifest だけを `label` の dtype 席へ書き換えて返す。"""
        out_dir = tmp_path / "models" / "labelled"
        assemble_family([_synthetic_plan("A", "w/model.safetensors", b"weights-A")], out_dir, "A")
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        model = manifest["models"]["A"]
        model["weights"]["w"] = {label: model["weights"]["w"]["f16"]}
        model["quants"]["f16"]["weights"]["w"] = label
        (out_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return out_dir

    def test_it_refuses_a_label_outside_the_storage_dtype_vocabulary(self, tmp_path: Path) -> None:
        """席名を dtype ラベルへ持ち込む綴り（`w4`）は fail loudly — 許可語彙を文言に出す。"""
        out_dir = self._distribution(tmp_path, "w4")

        with pytest.raises(DistError, match="dtype ラベル 'w4'") as error:
            verify_dist(out_dir)
        assert "'i4'" in str(error.value)

    def test_it_accepts_the_packed_int4_label(self, tmp_path: Path) -> None:
        """`i4` は語彙の内側（カード側の i4 注記の pin と対になる綴り）。"""
        out_dir = self._distribution(tmp_path, "i4")

        assert verify_dist(out_dir) == {"A/w/model.safetensors": len(b"weights-A")}


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

    def test_it_refuses_a_rel_path_that_climbs_out_of_the_model_subtree(
        self, tmp_path: Path
    ) -> None:
        """`A/../../victim` は staging の外の既存ファイルを truncate で潰す。

        書いた後の門は 1 つも鳴らない — manifest の root 検査は先頭セグメント（`A`）しか
        見ず、宣言外ファイル検査は staging の中しか見ない。`out_dir / rel_path` は
        `is_file()` も size も宣言と一致するので、`verify_dist` は最後まで緑で通る。
        """
        victim = tmp_path / "models" / "victim.bin"
        victim.parent.mkdir(parents=True)
        victim.write_bytes(b"someone-elses-bytes")
        plans = [_synthetic_plan("A", "../../victim.bin", b"weights")]
        out_dir = tmp_path / "models" / "escape"

        with pytest.raises(DistError, match="先頭ドットのセグメント"):
            assemble_family(plans, out_dir, "A")

        assert victim.read_bytes() == b"someone-elses-bytes"
        assert not out_dir.exists()

    def test_it_refuses_a_model_name_that_is_not_a_path_segment(self, tmp_path: Path) -> None:
        """名前検査は recipe 側にしか無かった（core の組み立て経路は素通しだった）。"""
        plans = [_synthetic_plan("../escape", "w/model.safetensors", b"weights")]
        out_dir = tmp_path / "models" / "bad-name"

        with pytest.raises(DistError, match="許可文字"):
            assemble_family(plans, out_dir, "../escape")
        assert not out_dir.exists()

    def test_it_refuses_two_roles_of_one_model_claiming_the_same_path(self, tmp_path: Path) -> None:
        """同一モデル内の `rel_path` 衝突は後勝ちで沈黙する（モデル間には同じ MUST がある）。

        後の役が先の役の実体を上書きし、manifest は**両役を後の digest** で宣言するので、
        現物と表は一致したまま `encoder` のセッションが `decoder` のグラフを読む。
        """
        plan = ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={
                "encoder": Artifact(rel_path="model.safetensors", payload=b"encoder-bytes"),
                "decoder": Artifact(rel_path="model.safetensors", payload=b"decoder-bytes"),
            },
            weights={
                "enc": {"f16": WeightFiles(file="encoder")},
                "dec": {"f16": WeightFiles(file="decoder")},
            },
            assets={},
            quants={"f16": {"weights": {"enc": "f16", "dec": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )
        out_dir = tmp_path / "models" / "collided"

        with pytest.raises(DistError, match="両方主張している"):
            assemble_family([plan], out_dir, "A")
        assert not out_dir.exists()


class TestQuantPresentation:
    """quant の表示欄（ADR 0075 決定 1 の `label` / `description`）— optional・上限つき 1 行。

    hub は「manifest は外部入力」の前提で同じ境界検査を持つ（決定 2）ので、緩いまま焼くと
    「組み上がったのに読めない配布形」ができる。
    """

    def test_a_seat_may_leave_both_fields_out(self) -> None:
        assert_quant_presentation("m.quants.f16", {"weights": {}, "session": {}})

    @pytest.mark.parametrize(
        ("key", "limit"),
        [("label", MAX_QUANT_LABEL_CHARS), ("description", MAX_QUANT_DESCRIPTION_CHARS)],
    )
    def test_it_accepts_the_longest_value_the_hub_reads(self, key: str, limit: int) -> None:
        assert_quant_presentation("m.quants.f16", {key: "x" * limit})

    @pytest.mark.parametrize(
        ("key", "limit"),
        [("label", MAX_QUANT_LABEL_CHARS), ("description", MAX_QUANT_DESCRIPTION_CHARS)],
    )
    def test_it_refuses_one_character_beyond_the_limit(self, key: str, limit: int) -> None:
        with pytest.raises(DistError, match=f"{limit} を超えた"):
            assert_quant_presentation("m.quants.f16", {key: "x" * (limit + 1)})

    @pytest.mark.parametrize("value", ["", "   ", 42, ["a label"]])
    def test_it_refuses_a_value_that_is_not_a_non_empty_string(self, value: Any) -> None:
        with pytest.raises(DistError, match="非空の文字列でない"):
            assert_quant_presentation("m.quants.f16", {"label": value})

    def test_it_refuses_a_description_that_spans_two_lines(self) -> None:
        """1 行の説明 — 選択 UI の 1 行にもカードの表の 1 セルにも改行は入れられない。"""
        with pytest.raises(DistError, match="1 行でない"):
            assert_quant_presentation("m.quants.f16", {"description": "first\nsecond"})

    def test_the_plan_gate_catches_it_before_writing_anything(self, tmp_path: Path) -> None:
        """計画段の門（ADR 0041 §7 の規模上限と同じ席）— 数 GB を並べてから落とさない。"""
        plan = _synthetic_plan("A", "w/model.safetensors", b"weights")
        oversized = {
            "f16": {**plan.quants["f16"], "label": "x" * (MAX_QUANT_LABEL_CHARS + 1)},
        }
        out_dir = tmp_path / "models" / "loud"

        with pytest.raises(DistError, match=r"A\.quants\.f16\.label"):
            assemble_family([replace(plan, quants=oversized)], out_dir, "A")
        assert not out_dir.exists()


class TestManifestJsonLiterals:
    """manifest に載る綴りは**標準 JSON だけ** — hub の読み口はブラウザの `JSON.parse`。

    Python の `json` は `NaN` / `Infinity` を既定で書き、既定で読み返す（`verify_dist` の
    読み返しも素通しする）ので、焼く側で落とさないと配布してから利用者の手元で初めて壊れる。
    同じ MUST は IR 側（`karume.ir.IrGraph.to_json`）に既にある。
    """

    def test_it_refuses_a_non_finite_float_in_the_manifest(self) -> None:
        with pytest.raises(ValueError, match="JSON compliant"):
            manifest_text({"format": MANIFEST_FORMAT, "models": {"a": {"scale": float("inf")}}})

    def test_it_writes_a_finite_manifest_with_a_trailing_newline(self) -> None:
        assert manifest_text({"format": MANIFEST_FORMAT}).endswith('"karume/4"\n}\n')

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_it_refuses_a_non_finite_preprocessor_channel(self, bad: float) -> None:
        """`NaN` は `<= 0` が偽なので、正数検査だけでは `image_std` すらすり抜ける。"""
        with pytest.raises(DistError, match="有限でない"):
            preprocessor_channels(
                {"image_std": [0.5, bad, 0.5]},
                "image_std",
                "preprocessor_config.json",
                channels=3,
                positive=True,
            )

    def test_it_still_reads_the_finite_channels(self) -> None:
        assert preprocessor_channels(
            {"image_mean": [0.5, 0.25, 1]},
            "image_mean",
            "preprocessor_config.json",
            channels=3,
            positive=False,
        ) == [0.5, 0.25, 1.0]


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
            ref = manifest["models"][name]["weights"]["w"]["f16"]["shards"][0]
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
            ref = manifest["models"][name]["weights"]["w"]["f16"]["shards"][0]
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
            ref = manifest["models"][name]["weights"]["w"]["f16"]["shards"][0]
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
            ref = manifest["models"][name]["weights"]["w"]["f16"]["shards"][0]
            assert ref["path"] == target, name
            assert ref["sha256"] == hashlib.sha256(shared_bytes).hexdigest(), name
        loner = manifest["models"]["C"]["weights"]["w"]["f16"]["shards"][0]
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


class TestRootFiles:
    """配布リポ直下の法的テキスト（上流ライセンスが再配布の条件として要求するファイル）。

    manifest はこれを宣言しない（どのモデルにも属さず、配布リポそのものに掛かる）ので、
    宣言外ファイル検査の例外として名指しで通す席になる。
    """

    def test_it_writes_them_at_the_repository_root_and_verifies(self, tmp_path: Path) -> None:
        """直下へ UTF-8 で置かれ、その状態のまま `verify_dist` を通る。"""
        out_dir = tmp_path / "models" / "legal"
        texts = {"LICENSE.md": "ライセンス原文\n", "NOTICE.md": "notice\n"}
        assemble_family(
            [_synthetic_plan("A", "w/model.safetensors", b"weights")],
            out_dir,
            "A",
            root_files=texts,
        )

        for name, text in texts.items():
            assert (out_dir / name).read_text(encoding="utf-8") == text
        # manifest は 1 つも宣言していない（資産ではない）が、検査は通る。
        assert sorted(verify_dist(out_dir)) == ["A/w/model.safetensors"]

    def test_it_refuses_a_name_outside_the_legal_seat_before_writing_anything(
        self, tmp_path: Path
    ) -> None:
        """席は法的テキストの 2 つだけ — 直下は「宣言外ファイルの不在」の網が届かない場所。"""
        out_dir = tmp_path / "models" / "smuggled"
        with pytest.raises(DistError, match="法的テキスト専用の席"):
            assemble_family(
                [_synthetic_plan("A", "w/model.safetensors", b"weights")],
                out_dir,
                "A",
                root_files={"LICENSE.md": "ok", "config.json": "{}"},
            )
        assert not out_dir.exists()

    def test_a_family_without_them_still_refuses_an_undeclared_root_file(
        self, tmp_path: Path
    ) -> None:
        """例外は**その 2 つの名前**だけに掛かる — 席を使わない family の網は緩まない。"""
        out_dir = tmp_path / "models" / "plain"
        assemble_family([_synthetic_plan("A", "w/model.safetensors", b"weights")], out_dir, "A")
        (out_dir / "leftover.safetensors").write_bytes(b"stale")

        with pytest.raises(DistError, match=r"leftover\.safetensors"):
            verify_dist(out_dir)

    def test_the_exemption_does_not_reach_a_subdirectory(self, tmp_path: Path) -> None:
        """例外は名前でなく**相対 path** — モデルサブツリーに紛れた同名は従来どおり落ちる。"""
        out_dir = tmp_path / "models" / "nested"
        assemble_family(
            [_synthetic_plan("A", "w/model.safetensors", b"weights")],
            out_dir,
            "A",
            root_files={"NOTICE.md": "notice\n"},
        )
        (out_dir / "A" / "NOTICE.md").write_text("notice\n", encoding="utf-8")

        with pytest.raises(DistError, match=r"A/NOTICE\.md"):
            verify_dist(out_dir)


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

        monkeypatch.setattr("karume.artifacts.os.replace", failing)

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


class TestExternalComponents:
    """越境コンポーネント参照（ADR 0038 §7 の `repo` / `revision` 席）— **opt-in**。

    用途は「同じバイト列を 2 つのリポへ二重に上げない」こと 1 つで、指定が無ければ配布形は
    完全に自己完結のまま（{@link test_a_plain_assembly_stays_self_contained}）。
    """

    _SHARED = b"text-encoder-bytes"
    _OWN = b"transformer-bytes"
    _REVISION = "0123456789abcdef0123456789abcdef01234567"

    def _plan(self, name: str) -> ModelPlan:
        """2 役のモデル（片方は他リポと同一バイト・片方はこのリポ固有）。"""
        return ModelPlan(
            name=name,
            pipeline="anima/1",
            artifacts={
                "text_encoder": Artifact("text_encoder/model.safetensors", payload=self._SHARED),
                "transformer": Artifact("transformer/model.safetensors", payload=self._OWN),
            },
            weights={
                "text_encoder": {"f16": WeightFiles("text_encoder")},
                "transformer": {"f16": WeightFiles("transformer")},
            },
            assets={},
            quants={
                "f16": {"weights": {"text_encoder": "f16", "transformer": "f16"}, "session": {}}
            },
            default_quant="f16",
            pipeline_config={},
        )

    def _source_dist(self, tmp_path: Path) -> Path:
        """参照元（既に組み上がっている別リポの配布形）。"""
        source = tmp_path / "models" / "karume-source"
        assemble_family([self._plan("source")], source, "source")
        return source

    def _components(self, source: Path, **overrides: Any) -> ExternalComponents:
        return ExternalComponents(
            **{
                "repo": "hdae/karume-source",
                "revision": self._REVISION,
                "dist": source,
                "model": "source",
                "roles": ("text_encoder",),
                **overrides,
            }
        )

    def test_a_plain_assembly_stays_self_contained(self, tmp_path: Path) -> None:
        """指定なしの組み立ては 1 バイトも変わらない — 参照は 1 つも生えず現物が全部揃う。"""
        out_dir = tmp_path / "models" / "karume-plain"
        manifest = assemble_family([self._plan("plain")], out_dir, "plain")

        assert not any(is_external_ref(ref) for _, ref in dist._declared_refs(manifest))
        assert sorted(verify_dist(out_dir)) == [
            "plain/text_encoder/model.safetensors",
            "plain/transformer/model.safetensors",
        ]

    def test_it_declares_the_named_role_as_a_pinned_reference(self, tmp_path: Path) -> None:
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-borrower"

        manifest = assemble_family(
            [self._plan("borrower")],
            out_dir,
            "borrower",
            external=self._components(source),
        )

        entry = manifest["models"]["borrower"]["weights"]["text_encoder"]["f16"]
        assert entry["shards"] == [
            {
                "repo": "hdae/karume-source",
                "revision": self._REVISION,
                "path": "source/text_encoder/model.safetensors",
                "size": len(self._SHARED),
                "sha256": hashlib.sha256(self._SHARED).hexdigest(),
            }
        ]

    def test_the_referenced_bytes_are_not_stored_here_a_second_time(self, tmp_path: Path) -> None:
        """参照の存在理由そのもの — 置いた上で参照を書くと利得がゼロになる。"""
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-borrower"

        assemble_family(
            [self._plan("borrower")], out_dir, "borrower", external=self._components(source)
        )

        assert not (out_dir / "borrower" / "text_encoder").exists()
        # 越境参照は実在検査の対象外で、自リポ固有の役割だけが現物として残る。
        assert sorted(verify_dist(out_dir)) == ["borrower/transformer/model.safetensors"]

    @pytest.mark.parametrize("revision", ["main", "v0.4.3", "0123456789abcdef", "A" * 40, "0" * 41])
    def test_it_refuses_a_revision_that_is_not_a_full_commit_sha(
        self, tmp_path: Path, revision: str
    ) -> None:
        """ブランチ名やタグは pin にならない — 指し先が動けば size / sha256 と現物が食い違う。"""
        with pytest.raises(DistError, match="40 桁の commit sha でない"):
            self._components(tmp_path, revision=revision)

    def test_it_refuses_a_role_the_source_distribution_does_not_declare(
        self, tmp_path: Path
    ) -> None:
        """参照先に無いものは参照できない（**指定した役割**が向こうで欠けている形）。"""
        source = tmp_path / "models" / "karume-source"
        # 参照元は `transformer` しか持たない配布形。
        assemble_family(
            [_synthetic_plan("source", "transformer/model.safetensors", self._OWN)],
            source,
            "source",
        )
        out_dir = tmp_path / "models" / "karume-borrower"

        with pytest.raises(DistError, match="役割 'text_encoder' のファイルが参照元"):
            assemble_family(
                [self._plan("borrower")], out_dir, "borrower", external=self._components(source)
            )
        assert not out_dir.exists()

    def test_it_refuses_a_reference_whose_bytes_differ_from_the_local_ones(
        self, tmp_path: Path
    ) -> None:
        """中身違いの参照は「別のモデルの重み」を自分のものとして配る形になる。

        shape も manifest も正しいままなので、突き合わせをここで切ると沈黙する。
        """
        source = tmp_path / "models" / "karume-source"
        other = replace(
            self._plan("source"),
            artifacts={
                "text_encoder": Artifact("text_encoder/model.safetensors", payload=b"other-bytes!"),
                "transformer": Artifact("transformer/model.safetensors", payload=self._OWN),
            },
        )
        assemble_family([other], source, "source")
        out_dir = tmp_path / "models" / "karume-borrower"

        with pytest.raises(DistError, match="自分で組むバイト列と違う"):
            assemble_family(
                [self._plan("borrower")], out_dir, "borrower", external=self._components(source)
            )
        assert not out_dir.exists()

    def test_it_applies_one_reference_to_every_model_of_a_family(self, tmp_path: Path) -> None:
        """複数モデル同居リポでも越境参照は掛けられる（ADR 0087 の extra リポがこの形）。

        成立条件は「指定役割の現物が全モデルで参照先とバイト同一」— 満たしていれば全モデルの
        同名役割が**同一の参照**を宣言し、そのバイト列は自リポのどこにも置かれない。
        """
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-family"

        manifest = assemble_family(
            [self._plan("A"), self._plan("B")],
            out_dir,
            "A",
            external=self._components(source),
        )

        expected = [
            {
                "repo": "hdae/karume-source",
                "revision": self._REVISION,
                "path": "source/text_encoder/model.safetensors",
                "size": len(self._SHARED),
                "sha256": hashlib.sha256(self._SHARED).hexdigest(),
            }
        ]
        assert manifest["models"]["A"]["weights"]["text_encoder"]["f16"]["shards"] == expected
        assert manifest["models"]["B"]["weights"]["text_encoder"]["f16"]["shards"] == expected
        # 参照したバイト列は自リポのどの席（モデル別サブツリー・`shared/`）にも無い。
        assert not (out_dir / "A" / "text_encoder").exists()
        assert not (out_dir / "B" / "text_encoder").exists()
        assert not (out_dir / SHARED_DIRNAME / "text_encoder").exists()
        # 自リポ固有の役割は 2 モデルで同一バイトなので従来どおり `shared/` へ畳まれる。
        assert sorted(verify_dist(out_dir)) == [f"{SHARED_DIRNAME}/transformer/model.safetensors"]

    def test_it_refuses_a_family_where_one_model_holds_different_bytes(
        self, tmp_path: Path
    ) -> None:
        """1 モデルだけ現物が参照先と違う形 — 「同じ役割名が別バイトを指す」の実体。

        通してしまうと、そのモデルには**別のモデルの重み**を指す manifest が配られる。
        """
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-family"
        divergent = replace(
            self._plan("B"),
            artifacts={
                "text_encoder": Artifact("text_encoder/model.safetensors", payload=b"other-bytes!"),
                "transformer": Artifact("transformer/model.safetensors", payload=self._OWN),
            },
        )

        with pytest.raises(DistError, match="自分で組むバイト列と違う"):
            assemble_family(
                [self._plan("A"), divergent],
                out_dir,
                "A",
                external=self._components(source),
            )
        assert not out_dir.exists()

    def test_it_refuses_a_family_whose_models_resolve_to_different_references(
        self, tmp_path: Path
    ) -> None:
        """バイト列が同じでも**指し先が 1 つに定まらない**なら落とす。

        参照は全モデルへ同じ 1 つを書く形なので、モデルごとに別の path へ解決する組み合わせは
        そもそも宣言できない（食い違った役割名と両モデル名を綴る）。
        """
        source = tmp_path / "models" / "karume-source"
        # 同じバイト列を 2 つの相対 path で宣言する参照元（借り手ごとに別の席が当たる）。
        assemble_family(
            [
                replace(
                    self._plan("source"),
                    artifacts={
                        "text_encoder": Artifact(
                            "text_encoder/model.safetensors", payload=self._SHARED
                        ),
                        "transformer": Artifact("transformer/model.safetensors", payload=self._OWN),
                        "alt_encoder": Artifact("alt/model.safetensors", payload=self._SHARED),
                    },
                    weights={
                        "text_encoder": {"f16": WeightFiles("text_encoder")},
                        "transformer": {"f16": WeightFiles("transformer")},
                        "alt_encoder": {"f16": WeightFiles("alt_encoder")},
                    },
                    quants={
                        "f16": {
                            "weights": {
                                "text_encoder": "f16",
                                "transformer": "f16",
                                "alt_encoder": "f16",
                            },
                            "session": {},
                        }
                    },
                )
            ],
            source,
            "source",
        )
        out_dir = tmp_path / "models" / "karume-family"
        elsewhere = replace(
            self._plan("B"),
            artifacts={
                "text_encoder": Artifact("alt/model.safetensors", payload=self._SHARED),
                "transformer": Artifact("transformer/model.safetensors", payload=self._OWN),
            },
        )

        with pytest.raises(DistError, match=r"指し先がモデル間で食い違う.*text_encoder.*'A'.*'B'"):
            assemble_family(
                [self._plan("A"), elsewhere],
                out_dir,
                "A",
                external=self._components(source),
            )
        assert not out_dir.exists()

    def test_the_cli_stays_inert_until_every_flag_is_given(self, tmp_path: Path) -> None:
        """部分指定を黙って無視すると、参照するつもりの組み立てが自己完結形で静かに出来上がる。"""
        assert (
            resolve_external_components(repo=None, revision=None, dist=None, model=None, roles=None)
            is None
        )
        with pytest.raises(DistError, match="--ref-model, --ref-revision, --ref-role"):
            resolve_external_components(
                repo="hdae/karume-source", revision=None, dist=tmp_path, model=None, roles=None
            )

    def test_the_cli_wires_the_five_flags_through(self, tmp_path: Path) -> None:
        """`main` 経由の一本通し（合成 pipeline は 1 役 `w` の計画を返す）。"""
        source = tmp_path / "models" / "karume-source"
        assemble_family([_synthetic_plan("m", "w/model.safetensors", b"w")], source, "m")
        out_dir = tmp_path / "models" / "karume-borrower"

        main(
            [
                "--pipeline",
                "solo",
                "--series",
                str(tmp_path),
                "--out",
                str(out_dir),
                "--ref-repo",
                "hdae/karume-source",
                "--ref-revision",
                self._REVISION,
                "--ref-dist",
                str(source),
                "--ref-model",
                "m",
                "--ref-role",
                "w",
            ],
            pipelines=_CALLER_REGISTRY,
        )

        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        ref = manifest["models"]["m"]["weights"]["w"]["f16"]["shards"][0]
        assert ref["repo"] == "hdae/karume-source"
        assert ref["path"] == "m/w/model.safetensors"
        assert verify_dist(out_dir) == {}


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


class TestShardExpansion:
    """分割されたコンポーネントは、現物から解決した**複数要素の shard 列**として宣言される。

    何本に割れるかは書いたバイト数で決まる（`karume.shards`）ので、pipeline の表には書けない
    — 組み立ては代表 path から現物を辿る（{@link karume.dist.expand_weight_shards}）。
    """

    def _plan(self, series: Path, sources: Sequence[bytes]) -> ModelPlan:
        """1 役だけの計画。系列側は shard 列（先頭 = グラフ shard）を連番で持つ。"""
        source = _write_series(series, sources)
        return ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={"w": Artifact(rel_path="w/model.safetensors", source=source)},
            weights={"w": {"f16": WeightFiles(file="w")}},
            assets={},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )

    def test_a_split_component_is_declared_as_an_ordered_shard_list(self, tmp_path: Path) -> None:
        payloads = ir_shards(3, mark="w")
        plan = self._plan(tmp_path / "series", payloads)
        out_dir = tmp_path / "models" / "sharded"

        manifest = assemble_family([plan], out_dir, "A")

        entry = manifest["models"]["A"]["weights"]["w"]["f16"]
        assert [ref["path"] for ref in entry["shards"]] == [
            "A/w/model-00001-of-00003.safetensors",
            "A/w/model-00002-of-00003.safetensors",
            "A/w/model-00003-of-00003.safetensors",
        ]
        # 並びは shard 番号順 MUST（先頭 = グラフ shard）。中身も番号どおりに置かれている。
        assert [ref["size"] for ref in entry["shards"]] == [len(item) for item in payloads]
        assert [ref["sha256"] for ref in entry["shards"]] == [
            hashlib.sha256(item).hexdigest() for item in payloads
        ]
        for ref, payload in zip(entry["shards"], payloads, strict=True):
            assert (out_dir / ref["path"]).read_bytes() == payload

    def test_the_shards_are_covered_by_the_declaration_check(self, tmp_path: Path) -> None:
        """突合は列を辿って現物へ届く（宣言外ファイル検査に落ちない）。"""
        payloads = ir_shards(2, mark="w")
        plan = self._plan(tmp_path / "series", payloads)
        out_dir = tmp_path / "models" / "sharded"

        assemble_family([plan], out_dir, "A")

        assert verify_dist(out_dir) == {
            "A/w/model-00001-of-00002.safetensors": len(payloads[0]),
            "A/w/model-00002-of-00002.safetensors": len(payloads[1]),
        }

    def test_the_smallest_component_still_declares_two_elements(self, tmp_path: Path) -> None:
        """数 KB の資産でも「グラフ shard + weight shard」の 2 要素（常時分割 — ADR 0081）。"""
        plan = self._plan(tmp_path / "series", ir_container(mark="w"))
        out_dir = tmp_path / "models" / "whole"

        manifest = assemble_family([plan], out_dir, "A")

        entry = manifest["models"]["A"]["weights"]["w"]["f16"]
        assert [ref["path"] for ref in entry["shards"]] == [
            "A/w/model-00001-of-00002.safetensors",
            "A/w/model-00002-of-00002.safetensors",
        ]

    def test_the_leftovers_of_a_previous_export_fail_loudly(self, tmp_path: Path) -> None:
        """単一ファイルと連番の同居は「どちらを配るか」が一意に決まらない。"""
        series = tmp_path / "series"
        plan = self._plan(series, ir_shards(2, mark="w"))
        (series / "model.safetensors").write_bytes(b"stale")

        with pytest.raises(DistError, match="同居"):
            assemble_family([plan], tmp_path / "models" / "mixed", "A")

    def test_a_split_component_that_another_seat_also_names_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """assets / extras の席は 1 ファイル参照 — 複数 shard になった役割は指せない。"""
        plan = self._plan(tmp_path / "series", ir_shards(2, mark="w"))
        plan = replace(plan, assets={"table": "w"})

        with pytest.raises(DistError, match="assets / extras も"):
            assemble_family([plan], tmp_path / "models" / "aliased", "A")


class TestRequiredLimits:
    """quant の `requiredLimits`（ADR 0038 §7）は組み立てが**現物から**導いて焼く。

    導出規則そのもの（何を需要と数え、どこから焼くか）の門は `test_limits.py` — ここが見るのは
    結線だけ: 現物のどこを入口にするか・quant の席にどう載るか・計画側が同じ欄を書いたら
    落ちるか。既定を差し替える 2 本は、合成資産（数 KB）では 128MiB の帯へ届かないため
    （実配布資産を作らずに「導出が現物のどこを見ているか」を観測する）。
    """

    #: フィクスチャの最大テンソル = linear 重み 4×32 の f32 = 512 バイト（`ir_fixtures`）。
    _LARGEST_TENSOR: ClassVar[int] = 512

    #: 上の 512 バイトの重み（1 行 128 バイト）を **2 piece** へ割る書き手容量（ADR 0090）。
    _PIECE_CAPACITY: ClassVar[int] = 256

    def _plan(
        self,
        series: Path,
        *,
        pipeline_config: Mapping[str, Any] | None = None,
        quant: Mapping[str, Any] | None = None,
        capacity: int | None = None,
    ) -> ModelPlan:
        """1 役だけの計画（weights の席は正当な IR コンテナ = 導出の入口）。"""
        source = _write_series(series, ir_container(mark="w", capacity=capacity))
        return ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={"w": Artifact(rel_path="w/model.safetensors", source=source)},
            weights={"w": {"f16": WeightFiles(file="w")}},
            assets={},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}, **(quant or {})}},
            default_quant="f16",
            pipeline_config=dict(pipeline_config or {}),
        )

    def _assemble(self, tmp_path: Path, plan: ModelPlan) -> dict[str, Any]:
        return assemble_family([plan], tmp_path / "models" / "limits", "A")

    def test_a_distribution_within_the_defaults_declares_nothing(self, tmp_path: Path) -> None:
        """既定スペックで動く資産に欄は生えない（「欄が無い = 既定で動く」の意味論）。"""
        manifest = self._assemble(tmp_path, self._plan(tmp_path / "series"))

        assert manifest["models"]["A"]["quants"]["f16"] == {
            "weights": {"w": "f16"},
            "session": {},
        }

    def test_it_bakes_the_largest_tensor_of_the_container(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """需要の出所は**配布に入る現物**のテンソル寸法（モデル定義の再計算ではない）。"""
        monkeypatch.setattr(
            limits,
            "WEBGPU_DEFAULT_LIMITS",
            {"maxBufferSize": 400, "maxStorageBufferBindingSize": 200},
        )

        manifest = self._assemble(tmp_path, self._plan(tmp_path / "series"))

        assert manifest["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxBufferSize": self._LARGEST_TENSOR,
            "maxStorageBufferBindingSize": self._LARGEST_TENSOR,
        }
        # 焼いた欄は据わった `karume.json` にも居る（返り値だけの飾りではない）。
        written = json.loads(
            (tmp_path / "models" / "limits" / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert written["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxBufferSize": self._LARGEST_TENSOR,
            "maxStorageBufferBindingSize": self._LARGEST_TENSOR,
        }

    def test_a_tensor_split_across_shards_is_baked_as_its_whole_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """分割テンソル（ADR 0090 決定 1）の需要は**親の全長**（GPU は piece を知らない）。

        導出は全 shard のヘッダを 1 枚へ畳んでから最大を採る（`dist.component_demand_bytes`）。
        shard ごとに最大を採る形へ退化すると、ここは断片 1 本ぶん（256 バイト）になり
        `maxBufferSize` の欄が消える — 「宣言は満たすのに `createSession` で落ちる」形。
        対照は直上の {@link test_it_bakes_the_largest_tensor_of_the_container}（piece ゼロの
        同じ系列でも同じ 512 が出る）。
        """
        monkeypatch.setattr(
            limits,
            "WEBGPU_DEFAULT_LIMITS",
            {"maxBufferSize": 400, "maxStorageBufferBindingSize": 200},
        )
        out_dir = tmp_path / "models" / "limits"
        plan = self._plan(tmp_path / "series", capacity=self._PIECE_CAPACITY)

        manifest = self._assemble(tmp_path, plan)

        # 被験体が本当に割れていること — 割れなかった日に主張が黙って恒真化しないための観測。
        entry = manifest["models"]["A"]["weights"]["w"]["f16"]
        pieces = [
            name
            for ref in entry["shards"]
            for name in safetensors_header(out_dir / ref["path"])
            if parse_piece_key(name) is not None
        ]
        assert len(pieces) == 2
        assert manifest["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxBufferSize": self._LARGEST_TENSOR,
            "maxStorageBufferBindingSize": self._LARGEST_TENSOR,
        }

    def test_it_declares_only_the_limit_the_demand_actually_exceeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """既定の違う 2 つなので、片方だけを超える帯では欄も片方だけ（実配布 irodori f16 の形）。"""
        monkeypatch.setattr(
            limits,
            "WEBGPU_DEFAULT_LIMITS",
            {"maxBufferSize": 4096, "maxStorageBufferBindingSize": 200},
        )

        manifest = self._assemble(tmp_path, self._plan(tmp_path / "series"))

        assert manifest["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxStorageBufferBindingSize": self._LARGEST_TENSOR
        }

    def test_a_state_slot_can_be_the_largest_resident_buffer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KV 容量の大きい系列では**最大テンソルより state スロットの方が大きい**。

        states 形のコンテナは合成できない（手術済み IR = attention の契約ごと組む必要がある）
        ので、読み口だけを差し替えて「state も需要に入る」ことを観測する。容量は
        `pipelineConfig` の席から来る（束縛点は `createGenerationContext` — ADR 0066 追記 7）。
        """
        monkeypatch.setattr(
            dist,
            "ir_graph",
            lambda path: {"states": {"kv": {"dtype": "f32", "shape": [1, 1, "C", 512]}}},
        )
        plan = self._plan(tmp_path / "series", pipeline_config={"maxPosition": 200_000})

        manifest = self._assemble(tmp_path, plan)

        slot_bytes = 200_000 * 512 * 4
        assert slot_bytes > self._LARGEST_TENSOR
        assert manifest["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxBufferSize": slot_bytes,
            "maxStorageBufferBindingSize": slot_bytes,
        }

    def test_a_state_slot_the_config_cannot_bind_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """容量席が無ければ落とす — 黙って state を外すと「宣言があるのに足りない」欄が焼ける。"""
        monkeypatch.setattr(
            dist,
            "ir_graph",
            lambda path: {"states": {"kv": {"dtype": "f32", "shape": [1, 1, "C", 512]}}},
        )
        plan = self._plan(tmp_path / "series")

        with pytest.raises(DistError, match="maxPosition"):
            self._assemble(tmp_path, plan)

    def test_it_refuses_a_plan_that_writes_the_field_itself(self, tmp_path: Path) -> None:
        """導出できる値を表にも持たせない（表だけが古い下限を名乗る形を作らせない）。"""
        plan = self._plan(
            tmp_path / "series", quant={"requiredLimits": {"maxBufferSize": 1_073_741_824}}
        )

        with pytest.raises(DistError, match="requiredLimits"):
            self._assemble(tmp_path, plan)

    def test_nothing_is_written_when_the_derivation_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """導出は**1 バイトも書く前**（落ちるなら途中の配布形を 1 ファイルも残さない）。"""
        monkeypatch.setattr(
            dist,
            "ir_graph",
            lambda path: {"states": {"kv": {"dtype": "f32", "shape": [1, 1, "C", 512]}}},
        )
        out_dir = tmp_path / "models" / "limits"

        with pytest.raises(DistError):
            assemble_family([self._plan(tmp_path / "series")], out_dir, "A")

        assert not out_dir.exists()


class TestInputContainerVerification:
    """組み立ては入力コンテナを「過去に検証済み」と信頼しない（CG4-1）。

    系列ディレクトリは truncate で上書きされる可変な場所（`dist` のモジュール doc）なので、
    **古いエクスポータで焼いた系列を `--series` で指す**運用事故が現実にありうる。組み立てが
    自前で読むのは格納 dtype と IR メタデータの一部だけなので、node / op / storage の壊れは
    family 固有の門をすり抜けて配布形に据わり、利用者の `createSession` で初めて落ちる。
    """

    #: 語彙に無い op（差し替え先）。`requires.ops` ごと差し替えるので、宣言の整合は保たれる
    #: — 落ちる先は**ランタイム支援の突合**であって JSON の形ではない。
    _UNSUPPORTED: ClassVar[str] = "linear_v2"

    def _tampered(self, payloads: Sequence[bytes]) -> list[bytes]:
        """weight shard はそのままで、**グラフ shard** の `karume_ir` の op だけを差し替えた列。

        4.md の失敗シナリオそのもの（手作業の改竄 / 別実装の出力）— 格納 dtype の門も
        宣言の門も通り、`verify_dist` も緑になる形。
        """
        graph_shard, *rest = payloads
        header_length = int.from_bytes(graph_shard[:8], "little")
        header = json.loads(graph_shard[8 : 8 + header_length])
        graph = json.loads(header["__metadata__"][IR_METADATA_KEY])
        swapped = {node["op"] for node in graph["nodes"]}
        for node in graph["nodes"]:
            node["op"] = self._UNSUPPORTED
        graph["requires"]["ops"] = sorted(
            {self._UNSUPPORTED if op in swapped else op for op in graph["requires"]["ops"]}
        )
        tensors = load(graph_shard)
        return [save(tensors, {IR_METADATA_KEY: json.dumps(graph)}), *rest]

    def _plan(self, series: Path, payloads: Sequence[bytes]) -> ModelPlan:
        source = _write_series(series, payloads)
        return ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={"w": Artifact(rel_path="w/model.safetensors", source=source)},
            weights={"w": {"f16": WeightFiles(file="w")}},
            assets={},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )

    def test_a_weights_container_the_runtime_cannot_execute_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        plan = self._plan(tmp_path / "series", self._tampered(ir_container()))
        out_dir = tmp_path / "models" / "tampered"

        # 落ちる層まで見る — 「JSON が読めない」ではなく**ランタイム支援の突合**で落ちること
        # そのものが、この門が閉じている穴（形も宣言も整った非実行グラフ）の定義。
        with pytest.raises(
            DistError, match="IR v1 の規則を満たさない: ランタイムの capability 不足 — 非対応 op"
        ):
            assemble_family([plan], out_dir, "A")
        assert not out_dir.exists()

    def test_without_the_preflight_the_same_distribution_is_assembled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """故障注入 — 門を外すと壊れたコンテナがそのまま据わる（恒真化していないことの証明）。

        他の門（格納 dtype・宣言と現物の突合・宣言外ファイル検査）はどれもこの改竄を見ない、
        が上の 1 本の意味そのもの。
        """
        monkeypatch.setattr(dist, "assert_weight_components_verified", lambda sharded: None)
        plan = self._plan(tmp_path / "series", self._tampered(ir_container()))
        out_dir = tmp_path / "models" / "tampered"

        assemble_family([plan], out_dir, "A")

        assert (out_dir / MANIFEST_FILENAME).is_file()

    def test_a_generated_payload_is_not_read_as_a_container(self, tmp_path: Path) -> None:
        """生成物（`payload`）の席は IR コンテナではない — 門の対象外（rope 素表・表 2 本）。"""
        plan = ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={"w": Artifact(rel_path="w/model.safetensors", payload=b"not-a-container")},
            weights={"w": {"f16": WeightFiles(file="w")}},
            assets={},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )

        manifest = assemble_family([plan], tmp_path / "models" / "generated", "A")

        entry = manifest["models"]["A"]["weights"]["w"]["f16"]
        assert [ref["path"] for ref in entry["shards"]] == ["A/w/model.safetensors"]


class TestExternalShardedComponents:
    """**分割されたコンポーネントへの越境参照** — `shards` 配列の各要素が 1 つの参照になる
    （ADR 0038 §7 / ADR 0071 決定 2「各要素は従来の FileRef 検査をそのまま通す」）。

    実需は turbo リポの共有 text_encoder（1GiB 超で複数本へ割れる）で、1 役 = 1 参照しか
    書けないと「同じバイト列を 2 つのリポへ上げない」という参照の存在理由そのものが、
    分割された資産にだけ届かなくなる。
    """

    _REVISION = "0123456789abcdef0123456789abcdef01234567"
    _REPO = "hdae/karume-source"
    _SHARDS = tuple(ir_shards(3, mark="text-encoder"))
    _OWN = b"transformer-bytes"

    def _series(self, root: Path, payloads: Sequence[bytes]) -> Path:
        """系列出力を書いて**代表 path** を返す（連番 = 現物が決める shard 列）。"""
        return _write_series(root, payloads)

    def _plan(self, name: str, series: Path) -> ModelPlan:
        """2 役のモデル（`text_encoder` は他リポと同一バイト・`transformer` はこのリポ固有）。"""
        return ModelPlan(
            name=name,
            pipeline="anima/1",
            artifacts={
                "text_encoder": Artifact("text_encoder/model.safetensors", source=series),
                "transformer": Artifact("transformer/model.safetensors", payload=self._OWN),
            },
            weights={
                "text_encoder": {"f16": WeightFiles("text_encoder")},
                "transformer": {"f16": WeightFiles("transformer")},
            },
            assets={},
            quants={
                "f16": {"weights": {"text_encoder": "f16", "transformer": "f16"}, "session": {}}
            },
            default_quant="f16",
            pipeline_config={},
        )

    def _source_dist(self, tmp_path: Path, series: Path) -> Path:
        """参照元（既に組み上がっている別リポの配布形 — 向こうでも同じ本数に割れている）。"""
        source = tmp_path / "models" / "karume-source"
        assemble_family([self._plan("source", series)], source, "source")
        return source

    def _components(self, source: Path, **overrides: Any) -> ExternalComponents:
        return ExternalComponents(
            **{
                "repo": self._REPO,
                "revision": self._REVISION,
                "dist": source,
                "model": "source",
                "roles": ("text_encoder",),
                **overrides,
            }
        )

    def test_every_shard_becomes_its_own_pinned_reference(self, tmp_path: Path) -> None:
        """repo / revision は全要素同一・path / size / sha256 は shard ごと・並びは番号順。"""
        series = self._series(tmp_path / "series", self._SHARDS)
        source = self._source_dist(tmp_path, series)
        out_dir = tmp_path / "models" / "karume-borrower"

        manifest = assemble_family(
            [self._plan("borrower", series)],
            out_dir,
            "borrower",
            external=self._components(source),
        )

        entry = manifest["models"]["borrower"]["weights"]["text_encoder"]["f16"]
        assert entry["shards"] == [
            {
                "repo": self._REPO,
                "revision": self._REVISION,
                "path": f"source/text_encoder/model-{index:05d}-of-00003.safetensors",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for index, payload in enumerate(self._SHARDS, start=1)
        ]

    def test_the_referenced_shards_are_not_stored_here_a_second_time(self, tmp_path: Path) -> None:
        """参照の存在理由そのもの — 全 shard が向こう側に残り、自リポは固有の役割だけ持つ。"""
        series = self._series(tmp_path / "series", self._SHARDS)
        source = self._source_dist(tmp_path, series)
        out_dir = tmp_path / "models" / "karume-borrower"

        assemble_family(
            [self._plan("borrower", series)],
            out_dir,
            "borrower",
            external=self._components(source),
        )

        assert not (out_dir / "borrower" / "text_encoder").exists()
        assert sorted(verify_dist(out_dir)) == ["borrower/transformer/model.safetensors"]

    @pytest.mark.parametrize("victim", [0, 1, 2])
    def test_it_checks_every_shard_against_the_source_distribution(
        self, tmp_path: Path, victim: int
    ) -> None:
        """突合は shard 列の**全要素**へ届く（先頭だけ見る形なら後続の改竄が沈黙する）。"""
        series = self._series(tmp_path / "series", self._SHARDS)
        source = self._source_dist(tmp_path, series)
        # 参照元の現物だけを、長さを保ったまま書き換える（size ではなく sha256 の門を踏む）。
        tampered = (
            source / "source" / "text_encoder" / (f"model-{victim + 1:05d}-of-00003.safetensors")
        )
        tampered.write_bytes(bytes(len(self._SHARDS[victim])))
        out_dir = tmp_path / "models" / "karume-borrower"

        with pytest.raises(DistError, match="自分で組むバイト列と違う"):
            assemble_family(
                [self._plan("borrower", series)],
                out_dir,
                "borrower",
                external=self._components(source),
            )
        assert not out_dir.exists()

    def test_the_smallest_component_is_referenced_shard_by_shard(self, tmp_path: Path) -> None:
        """数 KB の役割でも参照は shard ごと（常時分割 — 代表 path 1 本の参照は無い）。"""
        whole = tuple(ir_container(mark="whole-text-encoder"))
        series = self._series(tmp_path / "series", whole)
        source = self._source_dist(tmp_path, series)
        out_dir = tmp_path / "models" / "karume-borrower"

        manifest = assemble_family(
            [self._plan("borrower", series)],
            out_dir,
            "borrower",
            external=self._components(source),
        )

        entry = manifest["models"]["borrower"]["weights"]["text_encoder"]["f16"]
        assert entry["shards"] == [
            {
                "repo": self._REPO,
                "revision": self._REVISION,
                "path": f"source/text_encoder/model-{index:05d}-of-00002.safetensors",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for index, payload in enumerate(whole, start=1)
        ]

    def test_an_asset_seat_cannot_point_at_a_split_component(self, tmp_path: Path) -> None:
        """assets / extras は 1 ファイル参照しか書けない席 — 先頭 shard だけを黙って指さない。

        参照元では weights の役割として分割されているコンポーネントを、こちら側では assets の
        席から指した形（席が違えば同じ綴りの役割が別の意味を持つ）。
        """
        series = self._series(tmp_path / "series", self._SHARDS)
        source = self._source_dist(tmp_path, series)
        out_dir = tmp_path / "models" / "karume-borrower"
        # 自リポ側は単一ファイル（分割は参照先の事実）。参照は assets の席から掛ける。
        borrower = replace(
            self._plan("borrower", series),
            artifacts={
                "text_encoder": Artifact("text_encoder/model.safetensors", payload=b"local-copy"),
                "transformer": Artifact("transformer/model.safetensors", payload=self._OWN),
            },
            weights={"transformer": {"f16": WeightFiles("transformer")}},
            assets={"encoder": "text_encoder"},
            quants={"f16": {"weights": {"transformer": "f16"}, "session": {}}},
        )

        with pytest.raises(DistError, match="越境参照は分割されたコンポーネントに掛けられない"):
            assemble_family([borrower], out_dir, "borrower", external=self._components(source))
        assert not out_dir.exists()
