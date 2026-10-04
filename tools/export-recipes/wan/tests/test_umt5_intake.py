"""第三者の umT5 互換 encoder の取り込み（ADR 0122 段 b — 決定 5 / 6）の約束事。

- 取り込み（`wan.umt5_intake`）: ファイルを名前も中身もそのまま置き、sha256 が API の値と
  一致・`intake.json` が元の dtype・config の出所（上流のもの / 本家の写し）・ライセンス（未宣言
  なら印）を名乗る。sha256・キー集合・別名の対の値・FP8・混在・`.bin`・旧い綴り・構成の欄の
  不一致は fail loudly。対の両方を持つ形と片方だけの形は受ける。`unknown` など再配布の条件を
  識別しないライセンスの値は未宣言の印になる。取り込み先が `models/` の下なら API を引く前に、
  取り込み先に記録の外の重みのファイル（索引を含む）があれば記録を書く前に落ちる。
- 読み口: 記録が名指すファイルだけを、sha256 を照合した記述子越しに読む（索引を置いても、照合の
  後に path を差し替えても、読むのは照合した内容）。
- 書き手（`wan.umt5_export --intake`）: BF16 の取り込みから書いた容器が、同じ値を F32 へ広げた
  safetensors から同じ記録（dtype と、中身に従うファイルの sha256 / バイト数だけが違う）で書いた
  容器と全 part の sha256 で一致する（読みの時点の拡幅が丸めの出発点を変えない）。記録の dtype と
  ファイルの dtype の食い違い・取り込んだ後のファイルの差し替えは fail loudly。未宣言は明示なしの
  `write` が何も読まず・書かずに落ち、明示ありなら容器のライセンス欄に印が入る。
- 組み立て（`dist.py --pipeline umt5 --intake`）: 取り込み由来のモデルは、ライセンスの宣言の有無に
  よらず出力先が `models/` の下なら落ちる（symlink・`..` を含む相対でも）。未宣言は明示なしで書く
  前に落ちる。ルートのファイルは記録から導き、本家向けの LICENSE と「No retraining」の NOTICE を
  付けない。`karume-umt5-xxl` を名乗らない。宣言済みなら記録と容器の provenance が照合される。

リポ名は架空（ADR 0122 — 第三者のリポ名を追跡されるファイルに書かない）。重みは乱数初期化の小さな
umT5（`wan.umt5_probe.TINY_CONFIG`）— 実重みも HF へのアクセスも要らない。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from container_series import write_component
from safetensors.torch import load_file, save_file
from upstream_fixture import FIXTURE_REVISION, OTHER_REVISION

import dist
from _shared.container_read import read_provenance
from _shared.licenses import apache_license_2_0
from karume.container import Provenance
from karume.dist import (
    LICENSE_FILENAME,
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NOTICE_FILENAME,
    DistError,
)
from wan import umt5_export as ue
from wan import umt5_intake as ui
from wan import umt5_probe as probe
from wan.sources import UMT5_CONFIG, UMT5_SOURCES
from wan.tests import umt5_fixture
from wan.umt5_distribution import (
    PIPELINE,
    UMT5_DEFAULT_MODEL,
    UMT5_REPO_NAME,
    intake_container,
    intake_pipeline,
    intake_repo_name,
)

#: 架空の上流（実在のどのリポとも一致しない綴り）。
REPO = "fixture-org/umt5-compatible-fixture"
FILE = "umt5-compatible-fixture_bf16.safetensors"
NAME = "fixture"

#: 本家の行（構成の照合と写しの基準）。
BASE = UMT5_SOURCES[ui.BASE_UPSTREAM].source

EMBED = "encoder.embed_tokens.weight"
SHARED = "shared.weight"

#: 塊の行数（語彙 384・d_ff 160・d_model 64 のどれも割り切らない）。
CHUNK_ROWS = 7


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FakeHub:
    """HF API と取得の模擬（`wan.umt5_intake.Hub`）。

    上流は {@link REPO} の `FIXTURE_REVISION` だけ、本家は config だけを持つ。`sha256` は LFS の値の
    上書き（不一致の故障注入）、`card` は `cardData`（ライセンス・ベース）。
    """

    def __init__(
        self,
        files: Mapping[str, Path],
        base_config: Path,
        *,
        license: str | None = None,
        base_model: tuple[str, ...] | None = None,
        sha256: Mapping[str, str] | None = None,
        api_revision: str = FIXTURE_REVISION,
    ) -> None:
        self.files = dict(files)
        self.base_config = base_config
        self.license = license
        self.base_model = base_model
        self.sha256 = dict(sha256 or {})
        self.api_revision = api_revision
        self.downloads: list[tuple[str, str, str]] = []

    def revision(self, repo: str, revision: str) -> ui.HubRevision:
        assert (repo, revision) == (REPO, FIXTURE_REVISION)
        return ui.HubRevision(
            sha=self.api_revision,
            files={
                name: ui.HubFile(
                    size=path.stat().st_size,
                    sha256=None if name == UMT5_CONFIG else self.sha256.get(name, _sha256(path)),
                )
                for name, path in self.files.items()
            },
            license=self.license,
            base_model=self.base_model,
        )

    def download(self, repo: str, revision: str, filename: str) -> Path:
        self.downloads.append((repo, revision, filename))
        if (repo, revision) == (BASE.repo, BASE.revision):
            assert filename == UMT5_CONFIG
            return self.base_config
        assert (repo, revision) == (REPO, FIXTURE_REVISION)
        return self.files[filename]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """小さな umT5 の `save_pretrained`（config + F32 の単一 safetensors — tied な重みは
    片方だけ）。"""
    pytest.importorskip("transformers")
    directory = tmp_path_factory.mktemp("umt5-tiny")
    probe.tiny_model().save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def f32_tensors(tiny_dir: Path) -> dict[str, torch.Tensor]:
    return load_file(str(tiny_dir / ue.CHECKPOINT_SINGLE))


@pytest.fixture(scope="module")
def bf16_tensors(f32_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """追加学習版の形（同じキー・同じ形の BF16 — f32 を RNE で丸めた値）。"""
    return {key: value.to(torch.bfloat16) for key, value in f32_tensors.items()}


def _upstream(root: Path, tensors: Mapping[str, torch.Tensor], name: str = FILE) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(root / name))
    return root / name


def _intake(
    tmp_path: Path,
    tiny_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    name: str = NAME,
    config: Path | None = None,
    **hub: Any,
) -> ui.Umt5Intake:
    files = {FILE: _upstream(tmp_path / "upstream", tensors)}
    if config is not None:
        files[UMT5_CONFIG] = config
    return ui.intake(
        REPO,
        FIXTURE_REVISION,
        FILE,
        name,
        hub=FakeHub(files, tiny_dir / UMT5_CONFIG, **hub),
        out_root=tmp_path / "inputs" / "umt5",
    )


def _config(tmp_path: Path, tiny_dir: Path, **changes: Any) -> Path:
    """上流が持つ config（本家の config からの差だけを `changes` で入れる）。"""
    document = json.loads((tiny_dir / UMT5_CONFIG).read_text(encoding="utf-8"))
    document.update(changes)
    path = tmp_path / "upstream-config" / UMT5_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 取り込み
# ---------------------------------------------------------------------------


class TestTheIntake:
    def test_it_places_the_file_unchanged_and_records_the_upstream(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        placed = tmp_path / "inputs" / "umt5" / NAME / FILE
        assert intake.checkpoint == placed
        assert placed.read_bytes() == (tmp_path / "upstream" / FILE).read_bytes()
        assert _sha256(placed) == intake.file.sha256
        assert not placed.with_name(f"{FILE}.part").exists()
        record = json.loads((placed.parent / ui.INTAKE_FILE).read_text(encoding="utf-8"))
        assert record["format"] == ui.INTAKE_FORMAT
        assert (record["repo"], record["revision"]) == (REPO, FIXTURE_REVISION)
        assert record["file"] == {
            "name": FILE,
            "size": placed.stat().st_size,
            "sha256": _sha256(placed),
        }
        assert record["dtype"] == "BF16"

    def test_an_undeclared_upstream_is_marked_and_its_configuration_is_assumed(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        """cardData も config も無い上流（今回の追加学習版の形）: 未宣言の印・ベースは null・構成は
        本家の config の写し。"""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        assert intake.license == ui.UNDECLARED_LICENSE == "NOASSERTION"
        assert intake.undeclared
        assert intake.base_model is None
        assert intake.config == ui.ConfigOrigin("base-copy", BASE.repo, BASE.revision)
        assert intake.config_assumed
        assert (intake.directory / UMT5_CONFIG).read_bytes() == (
            tiny_dir / UMT5_CONFIG
        ).read_bytes()

    def test_a_declared_upstream_with_its_own_config_is_recorded_as_such(
        self, tmp_path, tiny_dir, f32_tensors
    ):
        config = _config(tmp_path, tiny_dir, architectures=["UMT5EncoderModel"])

        intake = _intake(
            tmp_path,
            tiny_dir,
            f32_tensors,
            config=config,
            license="apache-2.0",
            base_model=(BASE.repo,),
        )

        assert (intake.license, intake.base_model, intake.dtype) == (
            "apache-2.0",
            (BASE.repo,),
            "F32",
        )
        assert intake.config.source == "upstream"
        assert not intake.config_assumed
        assert (intake.directory / UMT5_CONFIG).read_bytes() == config.read_bytes()

    def test_rerunning_the_same_intake_is_safe(self, tmp_path, tiny_dir, bf16_tensors):
        first = _intake(tmp_path, tiny_dir, bf16_tensors)
        second = _intake(tmp_path, tiny_dir, bf16_tensors)

        assert second.file == first.file

    def test_a_sha256_that_differs_from_the_api_fails_and_records_nothing(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        with pytest.raises(ui.Umt5IntakeError, match="API の値と違う"):
            _intake(tmp_path, tiny_dir, bf16_tensors, sha256={FILE: "0" * 64})

        destination = tmp_path / "inputs" / "umt5" / NAME
        assert not (destination / FILE).exists()
        assert not (destination / ui.INTAKE_FILE).exists()

    def test_an_api_commit_other_than_the_pin_fails(self, tmp_path, tiny_dir, bf16_tensors):
        with pytest.raises(ui.Umt5IntakeError, match="指定の revision"):
            _intake(tmp_path, tiny_dir, bf16_tensors, api_revision=OTHER_REVISION)

    @pytest.mark.parametrize(
        ("file", "revision", "match"),
        [
            ("pytorch_model.bin", FIXTURE_REVISION, "safetensors でない"),
            ("sub/" + FILE, FIXTURE_REVISION, "basename でない"),
            (FILE, "main", "40 桁"),
        ],
    )
    def test_a_pickle_a_nested_file_or_a_branch_fails_before_the_api(
        self, tmp_path, file, revision, match
    ):
        class Unreachable:
            def revision(self, *_: Any) -> Any:
                raise AssertionError("API を引く前に落ちていない")

            def download(self, *_: Any) -> Any:
                raise AssertionError("取得の前に落ちていない")

        with pytest.raises(ui.Umt5IntakeError, match=match):
            ui.intake(REPO, revision, file, NAME, hub=Unreachable(), out_root=tmp_path)

    def test_the_name_of_the_base_row_is_refused(self):
        with pytest.raises(ui.Umt5IntakeError, match="本家の行"):
            ui.assert_intake_name(UMT5_DEFAULT_MODEL)

    @pytest.mark.parametrize("license", ["unknown", "UNKNOWN", "", "  ", "NoAssertion", "none"])
    def test_a_license_that_names_no_terms_is_recorded_as_undeclared(
        self, tmp_path, tiny_dir, bf16_tensors, license
    ):
        """HF の `license: unknown` などは宣言が無いのと同じ（明示なしで容器を書かせない）。"""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors, license=license)

        assert intake.license == ui.UNDECLARED_LICENSE
        assert intake.undeclared

    def test_a_foreign_weight_file_in_the_destination_fails_and_records_nothing(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        destination = tmp_path / "inputs" / "umt5" / NAME
        destination.mkdir(parents=True)
        (destination / ue.CHECKPOINT_INDEX).write_text(
            json.dumps({"weight_map": {SHARED: "evil.safetensors"}}), encoding="utf-8"
        )

        with pytest.raises(ue.Umt5ExportError, match="記録の外の重みのファイル"):
            _intake(tmp_path, tiny_dir, bf16_tensors)
        assert not (destination / ui.INTAKE_FILE).exists()


class TestTheIntakeDestination:
    """取り込み先（`--out` の下の `<名前>`）が `models/` の下なら、API を引く前に落ちる。"""

    class Unreachable:
        def revision(self, *_: Any) -> Any:
            raise AssertionError("API を引く前に落ちていない")

        def download(self, *_: Any) -> Any:
            raise AssertionError("取得の前に落ちていない")

    @pytest.fixture
    def models(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(ui, "DIST_ROOT", tmp_path / "models")
        (tmp_path / "models").mkdir()
        return tmp_path / "models"

    def _refused(self, out_root: Path, models: Path) -> None:
        with pytest.raises(ui.Umt5IntakeError, match="models/ に置かない"):
            ui.intake(REPO, FIXTURE_REVISION, FILE, NAME, hub=self.Unreachable(), out_root=out_root)
        assert not (models / NAME).exists()

    def test_models_is_refused(self, models):
        self._refused(models, models)

    def test_a_symlink_into_models_is_refused(self, tmp_path, models):
        (tmp_path / "staging").symlink_to(models)

        self._refused(tmp_path / "staging", models)

    def test_a_relative_path_climbing_into_models_is_refused(self, tmp_path, models, monkeypatch):
        (tmp_path / "work").mkdir()
        monkeypatch.chdir(tmp_path / "work")

        self._refused(Path("..", "inputs", "..", "models"), models)

    def test_the_default_destination_is_outside_models(self):
        """既定の置き場（`inputs/umt5/`）は判定を通る（判定がどこでも落ちる形でない — 対）。"""
        ui.assert_outside_distribution_root(ui.INTAKE_ROOT / NAME, root=ui.DIST_ROOT)


class TestTheCardLicense:
    """`cardData` の `license` / `license_name` → 記録する識別子（`None` は未宣言の印になる）。"""

    @pytest.mark.parametrize(
        ("license_id", "license_name", "expected"),
        [
            (None, None, None),
            ("unknown", None, None),
            ("Unknown", None, None),
            ("", None, None),
            ("other", None, None),
            ("other", "", None),
            ("other", "unknown", None),
            ("other", "vendor-license", "vendor-license"),
            ("apache-2.0", None, "apache-2.0"),
        ],
    )
    def test_values_that_name_no_terms_become_undeclared(self, license_id, license_name, expected):
        assert ui._card_license(license_id, license_name, REPO) == expected

    @pytest.mark.parametrize(("license_id", "license_name"), [(["mit"], None), ("other", 1)])
    def test_a_value_that_is_not_a_string_fails(self, license_id, license_name):
        with pytest.raises(ui.Umt5IntakeError, match="文字列でない"):
            ui._card_license(license_id, license_name, REPO)


class TestTheCheckpointGate:
    """取り込みの手順 4（ヘッダ・キー集合・形・別名の対）— どれが違っても記録を書かない。"""

    def _refused(self, tmp_path, tiny_dir, tensors, match: str) -> None:
        with pytest.raises(ue.Umt5ExportError, match=match):
            _intake(tmp_path, tiny_dir, tensors)
        assert not (tmp_path / "inputs" / "umt5" / NAME / ui.INTAKE_FILE).exists()

    def test_a_missing_key_fails(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = dict(bf16_tensors)
        del tensors["encoder.final_layer_norm.weight"]
        self._refused(tmp_path, tiny_dir, tensors, "キー集合")

    def test_a_decoder_key_fails(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = {**bf16_tensors, "decoder.final_layer_norm.weight": torch.ones(64).bfloat16()}
        self._refused(tmp_path, tiny_dir, tensors, "余り 1 本")

    def test_the_old_spelling_fails(self, tmp_path, tiny_dir, bf16_tensors):
        """Wan の原典の綴り（`blocks.N.attn.q` 等）は変換表を持たずに拒む。"""
        tensors = {
            key.replace("encoder.block.", "blocks.").replace(".layer.0.SelfAttention.", ".attn."): (
                value
            )
            for key, value in bf16_tensors.items()
        }
        self._refused(tmp_path, tiny_dir, tensors, "キー集合")

    def test_a_wrong_shape_fails(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = {**bf16_tensors, "encoder.final_layer_norm.weight": torch.ones(65).bfloat16()}
        self._refused(tmp_path, tiny_dir, tensors, "形が")

    def test_fp8_fails(self, tmp_path, tiny_dir, f32_tensors):
        tensors = {key: value.to(torch.float8_e4m3fn) for key, value in f32_tensors.items()}
        self._refused(tmp_path, tiny_dir, tensors, "F8_E4M3 は受けない")

    def test_mixed_bf16_and_f32_fails(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = {**bf16_tensors, SHARED: bf16_tensors[SHARED].float()}
        self._refused(tmp_path, tiny_dir, tensors, "混在")

    def test_both_names_of_the_tied_pair_are_accepted_when_bit_identical(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        """本家の形（対の両方 — 243 本の側）。"""
        tensors = {**bf16_tensors, EMBED: bf16_tensors[SHARED].clone()}

        assert _intake(tmp_path, tiny_dir, tensors).dtype == "BF16"

    def test_the_tied_pair_that_differs_in_one_element_fails(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        other = bf16_tensors[SHARED].clone()
        other[3, 5] += 1
        self._refused(
            tmp_path, tiny_dir, {**bf16_tensors, EMBED: other}, "値が checkpoint の中で違う"
        )

    def test_only_the_other_name_of_the_pair_is_accepted(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = dict(bf16_tensors)
        tensors[EMBED] = tensors.pop(SHARED)

        assert _intake(tmp_path, tiny_dir, tensors).dtype == "BF16"


class TestTheConfigGate:
    @pytest.mark.parametrize(
        "change",
        [
            {"relative_attention_max_distance": 64},
            {"relative_attention_num_buckets": 16},
            {"d_ff": 192},
        ],
    )
    def test_an_upstream_config_that_differs_from_the_base_fails(
        self, tmp_path, tiny_dir, bf16_tensors, change
    ):
        """バケットの構成はグラフ記述に入らない（容器がバイト同一のまま出力だけが変わりうる）ので、
        上流が config を持てば本家と突き合わせる。"""
        config = _config(tmp_path, tiny_dir, **change)

        with pytest.raises(ui.Umt5IntakeError, match="構成の"):
            _intake(tmp_path, tiny_dir, bf16_tensors, config=config)


class TestTheRecord:
    def _document(self, tmp_path, tiny_dir, bf16_tensors) -> tuple[Path, dict[str, Any]]:
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        path = intake.directory / ui.INTAKE_FILE
        return path, json.loads(path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"dtype": "F16"}, "dtype"),
            ({"format": "karume-umt5-intake/0"}, "format"),
            ({"extra": 1}, "余剰"),
            ({"revision": "main"}, "40 桁"),
            ({"license": ""}, "license が空"),
            ({"license": "unknown"}, "識別しない値"),
            ({"license": "noassertion"}, "識別しない値"),
            ({"license": "Other"}, "識別しない値"),
        ],
    )
    def test_a_broken_record_is_refused(self, tmp_path, tiny_dir, bf16_tensors, change, match):
        path, document = self._document(tmp_path, tiny_dir, bf16_tensors)
        path.write_text(json.dumps({**document, **change}), encoding="utf-8")

        with pytest.raises(ui.Umt5IntakeError, match=match):
            ui.load_intake(path.parent)

    def test_the_base_row_key_is_the_distribution_model_name(self):
        assert ui.BASE_UPSTREAM == UMT5_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 書き手（--intake）
# ---------------------------------------------------------------------------


def _widened_copy(intake: ui.Umt5Intake, root: Path, tensors: Mapping[str, torch.Tensor]) -> Path:
    """同じ取り込みの写しで、ファイルを F32 の `tensors` に替え、記録の dtype を F32 にしたもの
    （ファイルの sha256 / バイト数は中身に従って変わる — 他の欄は同じ）。"""
    shutil.copytree(intake.directory, root)
    save_file(dict(tensors), str(root / FILE))
    document = json.loads((root / ui.INTAKE_FILE).read_text(encoding="utf-8"))
    document["dtype"] = "F32"
    document["file"] = {
        "name": FILE,
        "size": (root / FILE).stat().st_size,
        "sha256": _sha256(root / FILE),
    }
    (root / ui.INTAKE_FILE).write_text(json.dumps(document), encoding="utf-8")
    return root


def _container(intake: ui.Umt5Intake, out: Path) -> list[str]:
    export = ue.prepare(
        intake.directory, checkpoint=ue.intake_checkpoint(intake), chunk_rows=CHUNK_ROWS
    )
    ue.write_container(export, out / ue.MODEL_FILE, ue.intake_provenance(intake))
    return ue.part_digests(out / ue.MODEL_FILE)


class TestTheWriter:
    def test_the_bf16_intake_writes_the_container_of_its_values_widened_to_f32(
        self, tmp_path, tiny_dir, bf16_tensors, f32_tensors
    ):
        """読みの時点の拡幅は丸めの出発点を変えない（ADR 0122 段 b の検収）。対照: 丸める前の F32
        から書くと別の容器になる（同じ層でも bf16 起点の i8 はバイトが違う）。"""
        bf16 = _intake(tmp_path, tiny_dir, bf16_tensors)
        widened = ui.load_intake(
            _widened_copy(
                bf16,
                tmp_path / "widened" / NAME,
                {key: value.float() for key, value in bf16_tensors.items()},
            )
        )
        unrounded = ui.load_intake(_widened_copy(bf16, tmp_path / "unrounded" / NAME, f32_tensors))

        digests = _container(bf16, tmp_path / "out-bf16")

        assert widened.dtype == "F32"
        assert _container(widened, tmp_path / "out-widened") == digests
        assert _container(unrounded, tmp_path / "out-unrounded") != digests

    def test_a_record_that_names_another_dtype_fails(self, tmp_path, tiny_dir, bf16_tensors):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        path = intake.directory / ui.INTAKE_FILE
        path.write_text(
            json.dumps({**json.loads(path.read_text(encoding="utf-8")), "dtype": "F32"}),
            encoding="utf-8",
        )

        with pytest.raises(ue.Umt5ExportError, match="F32 だけを受ける"):
            ue.prepare(
                intake.directory,
                checkpoint=ue.intake_checkpoint(ui.load_intake(intake.directory)),
            )

    def test_an_f32_file_under_a_bf16_record_fails(self, tmp_path, tiny_dir, bf16_tensors):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        widened = _widened_copy(
            intake, tmp_path / "w" / NAME, {k: v.float() for k, v in bf16_tensors.items()}
        )
        path = widened / ui.INTAKE_FILE
        path.write_text(
            json.dumps({**json.loads(path.read_text(encoding="utf-8")), "dtype": "BF16"}),
            encoding="utf-8",
        )

        with pytest.raises(ue.Umt5ExportError, match=r"intake\.json が名乗る BF16 だけを受ける"):
            ue.prepare(widened, checkpoint=ue.intake_checkpoint(ui.load_intake(widened)))

    def test_a_file_replaced_after_the_intake_fails(self, tmp_path, tiny_dir, bf16_tensors):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        other = dict(bf16_tensors)
        other[SHARED] = other[SHARED] + 1
        save_file(other, str(intake.checkpoint))

        with pytest.raises(ue.Umt5ExportError, match="sha256"):
            ue.intake_checkpoint(intake)

    def test_the_reader_ignores_an_index_and_reads_only_the_recorded_file(
        self, tmp_path, tiny_dir, bf16_tensors, monkeypatch
    ):
        """故障注入: 取り込み先に別のファイル（値 +1）とそれを指す索引を置き、記録の外の重みの門を
        外しても、読み口は照合したファイルだけを読む（索引の分岐を通らない）。"""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        evil = {key: value + 1 for key, value in bf16_tensors.items()}
        save_file(evil, str(intake.directory / "evil.safetensors"))
        (intake.directory / ue.CHECKPOINT_INDEX).write_text(
            json.dumps({"weight_map": dict.fromkeys(evil, "evil.safetensors")}), encoding="utf-8"
        )
        monkeypatch.setattr(ue, "assert_only_recorded_weights", lambda *_: None)

        checkpoint = ue.intake_checkpoint(intake)

        assert torch.equal(checkpoint.read(SHARED), bf16_tensors[SHARED].float())

    @pytest.mark.parametrize("foreign", [ue.CHECKPOINT_INDEX, "evil.safetensors", "sub/x.bin"])
    def test_a_foreign_weight_file_next_to_the_record_fails(
        self, tmp_path, tiny_dir, bf16_tensors, foreign
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        (intake.directory / foreign).parent.mkdir(parents=True, exist_ok=True)
        (intake.directory / foreign).write_bytes(b"{}")

        with pytest.raises(ue.Umt5ExportError, match="記録の外の重みのファイル"):
            ue.intake_checkpoint(intake)

    def test_a_file_swapped_after_the_check_is_not_read(self, tmp_path, tiny_dir, bf16_tensors):
        """照合と読みが同じ内容: 照合の後に path の先を別のファイルへ差し替えても（rename）、
        読むのは握った記述子の指す照合済みの内容。"""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        checkpoint = ue.intake_checkpoint(intake)
        swapped = tmp_path / "swapped.safetensors"
        save_file({key: value + 1 for key, value in bf16_tensors.items()}, str(swapped))
        os.replace(swapped, intake.checkpoint)

        assert torch.equal(checkpoint.read(SHARED), bf16_tensors[SHARED].float())
        assert torch.equal(checkpoint.read_rows(SHARED, 2, 5), bf16_tensors[SHARED][2:5].float())

    def test_the_upstream_table_path_still_takes_f32_only(self, tmp_path, bf16_tensors):
        save_file(dict(bf16_tensors), str(tmp_path / ue.CHECKPOINT_SINGLE))

        with pytest.raises(ue.Umt5ExportError, match="F32 だけを受ける"):
            ue.Checkpoint(tmp_path).read(SHARED)


class TestTheWriteCommand:
    @pytest.fixture
    def series_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        root = tmp_path / "series"
        monkeypatch.setattr(ue, "SERIES_ROOT", root)
        return root

    def test_an_undeclared_intake_fails_before_reading_or_writing(
        self, tmp_path, tiny_dir, bf16_tensors, series_root, monkeypatch
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        def unreachable(*_: Any, **__: Any) -> Any:
            raise AssertionError("明示なしで重みを読んだ")

        monkeypatch.setattr(ue, "prepare", unreachable)

        with pytest.raises(ue.Umt5ExportError, match=ui.ALLOW_UNDECLARED_LICENSE_FLAG):
            ue.main(["write", "--intake", str(intake.directory)])
        assert not series_root.exists()

    def test_an_unknown_license_needs_the_declaration_too(
        self, tmp_path, tiny_dir, bf16_tensors, series_root
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors, license="unknown")

        with pytest.raises(ue.Umt5ExportError, match=ui.ALLOW_UNDECLARED_LICENSE_FLAG):
            ue.main(["write", "--intake", str(intake.directory)])
        assert not series_root.exists()

    def test_the_declaration_writes_the_mark_into_the_container(
        self, tmp_path, tiny_dir, bf16_tensors, series_root, capsys
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        ue.main(["write", "--intake", str(intake.directory), ui.ALLOW_UNDECLARED_LICENSE_FLAG])

        container = series_root / f"umt5-xxl-{NAME}-i8-dyn" / ue.COMPONENT_DIR / ue.MODEL_FILE
        assert read_provenance(container) == Provenance(
            license=ui.UNDECLARED_LICENSE,
            notice=NOTICE_FILENAME,
            upstream_revision=FIXTURE_REVISION,
        )

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["write", ui.ALLOW_UNDECLARED_LICENSE_FLAG], "--intake の write にだけ効く"),
            (["write", "--intake", "seat", "--upstream", "xxl"], "同じ軸"),
            (["check-mask", "--intake", "seat"], "--intake は check-mask に効かない"),
            (
                ["reference", "--intake", "seat", ui.ALLOW_UNDECLARED_LICENSE_FLAG],
                "--allow-undeclared-license は reference に効かない",
            ),
        ],
    )
    def test_misplaced_arguments_are_refused(self, argv, message, capsys):
        with pytest.raises(SystemExit) as exited:
            ue.main(argv)

        assert exited.value.code == 2
        assert message in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 組み立て（dist.py --intake）
# ---------------------------------------------------------------------------


def _record(
    directory: Path,
    *,
    license: str = ui.UNDECLARED_LICENSE,
    config_source: str = "base-copy",
    base_model: list[str] | None = None,
) -> Path:
    """組み立てが読む記録だけを置く（組み立ては重みのファイルを読まない）。"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ui.INTAKE_FILE).write_text(
        json.dumps(
            {
                "format": ui.INTAKE_FORMAT,
                "name": directory.name,
                "repo": REPO,
                "revision": FIXTURE_REVISION,
                "file": {"name": FILE, "size": 1234, "sha256": "ab" * 32},
                "dtype": "BF16",
                "license": license,
                "base_model": base_model,
                "config": {
                    "source": config_source,
                    "base": {"repo": BASE.repo, "revision": BASE.revision},
                },
                "fetched_at": "2026-10-04T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return directory


class TestTheMirror:
    @pytest.fixture
    def roots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """`models/` と実験用ミラーの既定の置き場を一時ディレクトリへ向ける。"""
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(dist, "LOCAL_DIST_ROOT", tmp_path / "local-dist")
        (tmp_path / "models").mkdir()
        return tmp_path

    def _series(
        self, root: Path, *, license: str = ui.UNDECLARED_LICENSE, revision: str = FIXTURE_REVISION
    ) -> None:
        provenance = Provenance(license=license, notice=NOTICE_FILENAME, upstream_revision=revision)
        write_component(
            intake_container(root / "series", NAME),
            umt5_fixture.umt5_container(provenance=provenance),
        )

    def _argv(self, root: Path, *extra: str) -> list[str]:
        return [
            "--pipeline",
            "umt5",
            "--intake",
            str(root / "inputs" / "umt5" / NAME),
            "--series",
            str(root / "series"),
            *extra,
        ]

    def test_an_undeclared_intake_needs_the_declaration_before_anything_is_written(self, roots):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)

        with pytest.raises(DistError, match=ui.ALLOW_UNDECLARED_LICENSE_FLAG):
            dist.main(self._argv(roots))
        assert not (roots / "local-dist").exists()

    def test_the_undeclared_mirror_carries_no_license_and_says_so(self, roots, capsys):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)

        dist.main(self._argv(roots, ui.ALLOW_UNDECLARED_LICENSE_FLAG))

        out_dir = roots / "local-dist" / NAME
        assert not (out_dir / LICENSE_FILENAME).exists()
        notice = (out_dir / NOTICE_FILENAME).read_text(encoding="utf-8")
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "declares no license (recorded as `NOASSERTION`)" in notice
        assert f"`{REPO}` at commit `{FIXTURE_REVISION}`" in notice
        assert "bfloat16" in notice and "widened to float32 without loss" in notice
        for claim in ("No retraining", "Wan", "Google", "google/"):
            assert claim not in notice
        assert "license: NOASSERTION" in card
        assert "base_model" not in card.split("---")[1]
        assert "> **Local experimental conversion — do not redistribute.**" in card
        assert "**Configuration**: assumed" in card
        assert "The upstream file stores bfloat16 values." in card
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert list(manifest["models"]) == [NAME]

    def test_a_declared_mirror_ships_the_verbatim_license(self, roots, capsys):
        _record(
            roots / "inputs" / "umt5" / NAME,
            license="apache-2.0",
            config_source="upstream",
            base_model=["fixture-org/base"],
        )
        self._series(roots, license="apache-2.0")

        dist.main(self._argv(roots))

        out_dir = roots / "local-dist" / NAME
        assert (out_dir / LICENSE_FILENAME).read_text(encoding="utf-8") == apache_license_2_0()
        notice = (out_dir / NOTICE_FILENAME).read_text(encoding="utf-8")
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "declares the license `apache-2.0`" in notice
        assert "It declares the base model `fixture-org/base`." in notice
        for claim in ("No retraining", "Wan", "Google", "google/"):
            assert claim not in notice
        assert "license: apache-2.0" in card and "base_model: fixture-org/base" in card
        assert "> **Local experimental conversion — not published.**" in card
        assert "**Configuration**: the upstream `config.json`" in card

    @pytest.mark.parametrize("license", [ui.UNDECLARED_LICENSE, "apache-2.0"])
    def test_models_is_refused_whatever_the_license(self, roots, license):
        _record(roots / "inputs" / "umt5" / NAME, license=license)
        self._series(roots, license=license)

        with pytest.raises(DistError, match="models/ に置かない"):
            dist.main(
                self._argv(
                    roots, "--out", str(roots / "models" / NAME), ui.ALLOW_UNDECLARED_LICENSE_FLAG
                )
            )
        assert not (roots / "models" / NAME).exists()

    def test_a_symlink_into_models_is_refused(self, roots):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)
        (roots / "staging").symlink_to(roots / "models")

        with pytest.raises(DistError, match="models/ に置かない"):
            dist.main(
                self._argv(
                    roots, "--out", str(roots / "staging" / NAME), ui.ALLOW_UNDECLARED_LICENSE_FLAG
                )
            )
        assert not (roots / "models" / NAME).exists()

    def test_a_relative_path_climbing_into_models_is_refused(self, roots, monkeypatch):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)
        (roots / "work").mkdir()
        monkeypatch.chdir(roots / "work")
        out = os.path.join("..", "local-dist", "..", "models", NAME)

        with pytest.raises(DistError, match="models/ に置かない"):
            dist.main(self._argv(roots, "--out", out, ui.ALLOW_UNDECLARED_LICENSE_FLAG))
        assert not (roots / "models" / NAME).exists()

    @pytest.mark.parametrize(
        ("license", "revision", "match"),
        [
            ("cc-by-nc-4.0", FIXTURE_REVISION, r"provenance\.license"),
            ("apache-2.0", OTHER_REVISION, "別の revision"),
        ],
    )
    def test_the_container_provenance_is_checked_against_the_record(
        self, roots, license, revision, match
    ):
        _record(roots / "inputs" / "umt5" / NAME, license="apache-2.0")
        self._series(roots, license=license, revision=revision)

        with pytest.raises(DistError, match=match):
            dist.main(self._argv(roots))
        assert not (roots / "local-dist").exists()

    def test_a_marked_container_does_not_pass_as_a_declared_record(self, roots):
        """記録を宣言済みに書き換えても、印を焼いた容器は照合で落ちる（容器の中身で閉じる）。"""
        _record(roots / "inputs" / "umt5" / NAME, license="apache-2.0")
        self._series(roots, license=ui.UNDECLARED_LICENSE)

        with pytest.raises(DistError, match="NOASSERTION"):
            dist.main(self._argv(roots))

    def test_a_declared_license_without_a_text_is_refused_before_writing(self, roots):
        _record(roots / "inputs" / "umt5" / NAME, license="mit")
        self._series(roots, license="mit")

        with pytest.raises(DistError, match="本文が無い"):
            dist.main(self._argv(roots))
        assert not (roots / "local-dist").exists()

    @pytest.mark.parametrize(
        ("extra", "match"),
        [
            (["--model", UMT5_DEFAULT_MODEL], "取り込みの名前"),
            (["--pipeline", "wan"], "--pipeline umt5 にだけ効く"),
        ],
    )
    def test_mixing_with_other_models_or_pipelines_is_refused(self, roots, extra, match):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)

        with pytest.raises(DistError, match=match):
            dist.main([*self._argv(roots, ui.ALLOW_UNDECLARED_LICENSE_FLAG), *extra])

    def test_the_declaration_alone_is_refused(self, roots):
        with pytest.raises(DistError, match="--intake と組むときだけ"):
            dist.main(["--pipeline", "umt5", ui.ALLOW_UNDECLARED_LICENSE_FLAG])

    def test_it_never_names_the_base_repository(self, tmp_path):
        intake = ui.load_intake(_record(tmp_path / NAME))
        pipeline = intake_pipeline(intake, allow_undeclared_license=True)

        assert pipeline.repo_name(NAME) == intake_repo_name(NAME) != UMT5_REPO_NAME
        assert not intake_repo_name(NAME).startswith("karume-")
        assert pipeline.root_files != PIPELINE.root_files

    def test_the_base_pipeline_is_unchanged(self):
        assert dist.PIPELINES["umt5"] is PIPELINE
        assert set(PIPELINE.root_files) == {LICENSE_FILENAME, NOTICE_FILENAME}
        assert "No retraining and no fine-tuning" in PIPELINE.root_files[NOTICE_FILENAME]


class TestTheWholeChain:
    def test_intake_write_and_mirror(self, tmp_path, tiny_dir, bf16_tensors, monkeypatch, capsys):
        """取り込み → 書き手（明示あり）→ 実験用ミラー（明示あり）を実物の書き口で通す。"""
        monkeypatch.setattr(ue, "SERIES_ROOT", tmp_path / "series")
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(dist, "LOCAL_DIST_ROOT", tmp_path / "local-dist")
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        ue.main(["write", "--intake", str(intake.directory), ui.ALLOW_UNDECLARED_LICENSE_FLAG])
        dist.main(
            [
                "--pipeline",
                "umt5",
                "--intake",
                str(intake.directory),
                "--series",
                str(tmp_path / "series"),
                ui.ALLOW_UNDECLARED_LICENSE_FLAG,
            ]
        )

        out_dir = tmp_path / "local-dist" / NAME
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["models"][NAME]["pipeline"] == "umt5-encoder/1"
        assert sorted(path.name for path in out_dir.iterdir() if path.is_file()) == sorted(
            [MODEL_CARD_FILENAME, NOTICE_FILENAME, MANIFEST_FILENAME]
        )
