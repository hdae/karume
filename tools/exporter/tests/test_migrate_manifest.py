"""移行 CLI のリポ丸ごとモード（`karume migrate --manifest`）— ADR 0109 決定 3 / 4 / 8。

被験体は**合成の `karume/4` リポ**（実ミラーは読まない）。グラフ 1 本ぶんの変換そのものは
`test_migrate.py` の担当なので、ここが見るのはリポ単位でしか現れない事実である:

- shard 列を manifest の宣言から組む（ディレクトリを跨ぐ列・同じ列を指す複数モデルの畳み込み）
- `extras` と PLE sidecar が容器の資産になり、manifest の `assets` から消える
- 越境参照は変換せず `--cross-repo` の変換済みディレクトリから引き写す
- 変換しなかったファイルがそのまま複写され、`karume/5` の karume.json が出る
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from legacy_writer import Entry, legacy_fill_shards, legacy_shards, order, write_safetensors

from karume import migrate, publish
from karume.container import AssetRecord, Provenance, numbered_name
from karume.migrate import MigrateError, migrate_repository, parse_cross_repo

PROVENANCE = Provenance(license="apache-2.0", writer="karume/test")

#: 合成資産（数 KiB）で part またぎと資産の block 分割を踏むために下げた寸法。
SMALL_PART_BYTES = 1024
SMALL_BLOCK_BYTES = 512

#: 越境参照の pin（40 桁 hex 小文字 — `dist.REVISION_RE` の受理形）。
REVISION = "0" * 39 + "1"

#: PLE の合成寸法（`values` の 1 行 = layers × dim / 2 = 16 B・
#: `scales` の 1 行 = layers × 4 = 8 B）。
PLE_TOKENS = 64
PLE_LAYERS = 2
PLE_DIM = 16
PLE_RANGES: tuple[tuple[int, int], ...] = ((0, 20), (20, 45), (45, 64))


# ---------------------------------------------------------------------------
# 合成の `karume/4` リポ
# ---------------------------------------------------------------------------


def place(root: Path, rel: str, blob: bytes) -> dict[str, Any]:
    """1 ファイルを置いて 3 点セットを返す。"""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    return {"path": rel, "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def stage_component(
    root: Path, directories: Sequence[str], stem: str, shards: Sequence[bytes]
) -> list[dict[str, Any]]:
    """shard 列を**shard ごとの置き場**へ並べて FileRef 列を返す（列がディレクトリを跨げる）。"""
    total = len(shards)
    return [
        place(root, numbered_name(f"{directory}/{stem}.safetensors", index, total), blob)
        for index, (directory, blob) in enumerate(zip(directories, shards, strict=True), start=1)
    ]


def quant(weights: Mapping[str, str]) -> dict[str, Any]:
    return {"weights": dict(weights), "session": {}}


def model_entry(
    weights: Mapping[str, Mapping[str, Any]],
    assets: Mapping[str, Any],
    quants: Mapping[str, Any],
    *,
    pipeline: str = "synthetic",
) -> dict[str, Any]:
    return {
        "pipeline": f"{pipeline}/1",
        "weights": {name: dict(labels) for name, labels in weights.items()},
        "assets": dict(assets),
        "quants": dict(quants),
        "defaultQuant": next(iter(quants)),
        "pipelineConfig": {"kind": "synthetic"},
    }


def write_manifest(root: Path, models: Mapping[str, Any], default_model: str) -> Path:
    manifest = {
        "format": "karume/4",
        "generator": "karume/0.0.0",
        "defaultModel": default_model,
        "models": dict(models),
    }
    path = root / "karume.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """2 モデルが `encoder` を共有し、`decoder` の列がディレクトリを跨ぐリポ。"""
    root = tmp_path / "src"
    encoder = stage_component(
        root, ["shared/encoder"] * 2, "model.f32", legacy_shards(mark="enc", storage="f32")
    )
    # 列がディレクトリを跨ぐ形（グラフ shard は共有ディレクトリ・重み shard は話者ディレクトリ）。
    decoder = stage_component(
        root, ["shared/decoder", "alpha/decoder"], "model.f32", legacy_shards(mark="dec")
    )
    rope = place(root, "alpha/decoder/rope_base.safetensors", b"rope-base-table\n")
    tokenizer = place(root, "tokenizer.json", b'{"model":"synthetic"}\n')
    (root / "README.md").write_text("# synthetic\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    alpha = model_entry(
        {
            "encoder": {"f32": {"shards": encoder}},
            "decoder": {"f32": {"shards": decoder, "extras": {"rope_base": rope}}},
        },
        {"tokenizer": tokenizer},
        {"full": quant({"encoder": "f32", "decoder": "f32"})},
    )
    beta = model_entry(
        {"encoder": {"f32": {"shards": encoder}}},
        {"tokenizer": tokenizer},
        {"full": quant({"encoder": "f32"})},
    )
    write_manifest(root, {"alpha": alpha, "beta": beta}, "alpha")
    return root


def migrated(repo: Path, out: Path, **overrides: Any) -> migrate.RepositoryResult:
    return migrate_repository(
        repo / "karume.json",
        out,
        provenance=PROVENANCE,
        _part_bytes=SMALL_PART_BYTES,
        _block_bytes=SMALL_BLOCK_BYTES,
        **overrides,
    )


def manifest_of(out: Path) -> dict[str, Any]:
    return json.loads((out / "karume.json").read_text(encoding="utf-8"))


def container_of(out: Path, model: str, component: str, dtype: str) -> dict[str, Any]:
    entry = manifest_of(out)["models"][model]["weights"][component][dtype]
    assert set(entry) == {"container"}
    return entry["container"]


def opened(out: Path, container: Mapping[str, Any]) -> Any:
    from karume.container import read_container

    return read_container([out / ref["path"] for ref in container["parts"]])


def asset_payload(read: Any, name: str) -> bytes:
    """資産 1 本の payload（宣言の論理長で切る — 末尾の 0x00 を推測で剥がない）。"""
    record = read.model.assets[name]
    return read.block(record.block)[: record.length]


# ---------------------------------------------------------------------------


class TestTheRepositoryManifest:
    def test_it_writes_karume_5_with_the_old_declarations_kept(
        self, repo: Path, tmp_path: Path
    ) -> None:
        result = migrated(repo, tmp_path / "out")
        manifest = manifest_of(tmp_path / "out")

        assert result.manifest == tmp_path / "out" / "karume.json"
        assert manifest["format"] == "karume/5"
        assert manifest["defaultModel"] == "alpha"
        assert list(manifest["models"]) == ["alpha", "beta"]
        alpha = manifest["models"]["alpha"]
        assert list(alpha) == [
            "pipeline",
            "weights",
            "assets",
            "quants",
            "defaultQuant",
            "pipelineConfig",
        ]
        assert alpha["quants"] == {"full": quant({"encoder": "f32", "decoder": "f32"})}
        assert alpha["pipelineConfig"] == {"kind": "synthetic"}
        assert set(alpha["assets"]) == {"tokenizer"}

    def test_the_container_entry_has_the_declared_shape(self, repo: Path, tmp_path: Path) -> None:
        """ADR 0109 決定 3 — 2 文書それぞれの期待値 + part 0 を含む全 part の FileRef。"""
        out = tmp_path / "out"
        migrated(repo, out)
        container = container_of(out, "alpha", "encoder", "f32")

        assert list(container) == ["descriptor", "parts"]
        assert list(container["descriptor"]) == ["graph", "model"]
        for document in container["descriptor"].values():
            assert list(document) == ["length", "sha256"]
        read = opened(out, container)
        assert container["descriptor"]["graph"] == {
            "length": len(read.graph_descriptor_bytes),
            "sha256": hashlib.sha256(read.graph_descriptor_bytes).hexdigest(),
        }
        assert container["descriptor"]["model"] == {
            "length": len(read.model_descriptor_bytes),
            "sha256": hashlib.sha256(read.model_descriptor_bytes).hexdigest(),
        }
        assert len(container["parts"]) == len(read.model.parts) + 1
        for ref in container["parts"]:
            blob = (out / ref["path"]).read_bytes()
            assert ref["size"] == len(blob)
            assert ref["sha256"] == hashlib.sha256(blob).hexdigest()

    def test_a_zero_length_part_is_written_and_declared_with_size_zero(
        self, tmp_path: Path
    ) -> None:
        """const が空でも part 1 は 0 バイトのファイルとして並ぶ（ADR 0109 決定 3）。

        被験体は**定数を 1 本も持たない**コンポーネント（`legacy_fill_shards`）— const を
        持つ合成では part 1 が埋まってしまい、0 バイトの席そのものを踏めない。
        """
        root = tmp_path / "empty-const"
        plain = stage_component(
            root, ["alpha/encoder"] * 2, "model.f32", legacy_fill_shards(2, mark="plain")
        )
        write_manifest(
            root,
            {
                "alpha": model_entry(
                    {"encoder": {"f32": {"shards": plain}}},
                    {},
                    {"full": quant({"encoder": "f32"})},
                )
            },
            "alpha",
        )
        out = tmp_path / "out"
        migrated(root, out)
        parts = container_of(out, "alpha", "encoder", "f32")["parts"]

        assert parts[1]["size"] == 0
        assert (out / parts[1]["path"]).stat().st_size == 0
        assert parts[1]["sha256"] == hashlib.sha256(b"").hexdigest()

    def test_the_output_sits_beside_the_old_weight_shards(self, repo: Path, tmp_path: Path) -> None:
        """置き場は**重み shard の親**（列がディレクトリを跨いでも推測しない）。"""
        out = tmp_path / "out"
        migrated(repo, out)

        assert container_of(out, "alpha", "encoder", "f32")["parts"][0]["path"].startswith(
            "shared/encoder/model.f32-"
        )
        # decoder はグラフ shard が shared/・重み shard が alpha/ に居る。
        assert container_of(out, "alpha", "decoder", "f32")["parts"][0]["path"].startswith(
            "alpha/decoder/model.f32-"
        )

    def test_a_shared_component_is_converted_once(self, repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = migrated(repo, out)

        # encoder（共有）+ decoder の 2 本だけが変換される（席は 3 つ）。
        assert len(result.converted) == 2
        assert container_of(out, "alpha", "encoder", "f32") == container_of(
            out, "beta", "encoder", "f32"
        )

    def test_the_same_input_produces_the_same_bytes(self, repo: Path, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        migrated(repo, first)
        migrated(repo, second)

        def tree(root: Path) -> dict[str, bytes]:
            return {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }

        assert tree(first) == tree(second)

    def test_the_old_repository_is_read_only(self, repo: Path, tmp_path: Path) -> None:
        before = {
            path.relative_to(repo).as_posix(): path.read_bytes()
            for path in sorted(repo.rglob("*"))
            if path.is_file()
        }
        migrated(repo, tmp_path / "out")

        assert {
            path.relative_to(repo).as_posix(): path.read_bytes()
            for path in sorted(repo.rglob("*"))
            if path.is_file()
        } == before

    def test_a_manifest_that_is_not_karume_4_fails_loudly(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        root.mkdir()
        (root / "karume.json").write_text('{"format":"karume/5"}\n', encoding="utf-8")

        with pytest.raises(MigrateError, match="移行できるのは 'karume/4' だけ"):
            migrated(root, tmp_path / "out")

    def test_an_output_inside_the_input_fails_loudly(self, repo: Path) -> None:
        with pytest.raises(MigrateError, match="入れ子"):
            migrated(repo, repo / "out")

    def test_weight_shards_spread_over_directories_fail_loudly(self, tmp_path: Path) -> None:
        """置き場は重み shard の親 1 つ（散っていたら推測せずに止まる）。"""
        root = tmp_path / "src"
        spread = stage_component(
            root,
            ["shared/multi", "alpha/multi", "beta/multi"],
            "model.f32",
            legacy_fill_shards(3, mark="multi"),
        )
        write_manifest(
            root,
            {
                "alpha": model_entry(
                    {"multi": {"f32": {"shards": spread}}}, {}, {"full": quant({"multi": "f32"})}
                )
            },
            "alpha",
        )

        with pytest.raises(MigrateError, match="重み shard の親ディレクトリが揃っていない"):
            migrated(root, tmp_path / "out")

    def test_a_shard_row_with_mixed_stems_fails_loudly(self, tmp_path: Path) -> None:
        """出力の stem は旧 shard 名から取る（揃っていなければ推測せずに止まる）。"""
        root = tmp_path / "src"
        blobs = legacy_shards(mark="mix")
        mixed = [
            place(root, numbered_name("shared/mix/model.f32.safetensors", 1, 2), blobs[0]),
            place(root, numbered_name("shared/mix/other.f32.safetensors", 2, 2), blobs[1]),
        ]
        write_manifest(
            root,
            {
                "alpha": model_entry(
                    {"mix": {"f32": {"shards": mixed}}}, {}, {"full": quant({"mix": "f32"})}
                )
            },
            "alpha",
        )

        with pytest.raises(MigrateError, match="stem が揃っていない"):
            migrated(root, tmp_path / "out")

    def test_the_single_form_is_refused(self, repo: Path, tmp_path: Path) -> None:
        """karume/5 の container.parts は 2 要素以上 MUST（ADR 0109 決定 3）。"""
        with pytest.raises(MigrateError, match="--single はリポ丸ごとモードでは使えない"):
            migrate.main(
                [
                    "--manifest",
                    str(repo / "karume.json"),
                    "--out",
                    str(tmp_path / "out"),
                    "--license",
                    "apache-2.0",
                    "--single",
                ]
            )

    def test_a_failure_midway_through_the_swap_leaves_nothing_behind(
        self, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """落ちた回は何も残さない（半分だけ本番名で公開された容器を残さない — §12）。"""
        out = tmp_path / "out"
        real = os.replace
        seen = 0

        def flaky(source: Any, target: Any) -> None:
            nonlocal seen
            seen += 1
            if seen == 2:
                raise OSError("据え替えの途中で落ちた回")
            real(source, target)

        monkeypatch.setattr(publish.os, "replace", flaky)
        with pytest.raises(OSError, match="据え替えの途中"):
            migrated(repo, out)

        assert not list(out.rglob("*.krm"))


class TestTheCopiedFiles:
    def test_plain_files_are_copied_and_converted_ones_are_not(
        self, repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        result = migrated(repo, out)
        copied = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}

        assert "README.md" in copied
        assert "tokenizer.json" in copied
        # `.` で始まるディレクトリは入らない。
        assert not any(name.startswith(".git/") for name in copied)
        # 変換した shard・容器の資産になった extras・旧 karume.json は複写されない
        # （新しい karume.json は書かれる）。
        assert not any(name.endswith(".safetensors") for name in copied)
        assert not (out / "alpha" / "decoder" / "rope_base.safetensors").exists()
        assert result.copied == 2  # README.md / tokenizer.json
        assert (out / "karume.json").is_file()

    def test_it_refuses_to_write_over_a_previous_output(self, repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        migrated(repo, out)

        with pytest.raises(MigrateError, match=r"既に在る|前回の成果物"):
            migrated(repo, out)


class TestTheExtras:
    def test_rope_base_becomes_a_container_asset(self, repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        migrated(repo, out)
        read = opened(out, container_of(out, "alpha", "decoder", "f32"))

        expected = (repo / "alpha/decoder/rope_base.safetensors").read_bytes()
        assert read.model.assets == {
            "rope_base": AssetRecord(block="a.0", role="rope-base", length=len(expected))
        }
        raw = read.block("a.0")
        assert raw[: len(expected)] == expected
        assert raw[len(expected) :] == b"\x00" * (len(raw) - len(expected))
        # extras は manifest の席から消える（容器が持つ）。
        assert "extras" not in container_of(out, "alpha", "decoder", "f32")

    def test_an_unknown_extra_fails_loudly(self, repo: Path, tmp_path: Path) -> None:
        manifest = json.loads((repo / "karume.json").read_text(encoding="utf-8"))
        entry = manifest["models"]["alpha"]["weights"]["decoder"]["f32"]
        entry["extras"] = {"mystery": entry["extras"]["rope_base"]}
        (repo / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match="extras 'mystery' の写し先が無い"):
            migrated(repo, repo.parent / "out")


# ---------------------------------------------------------------------------
# 越境参照
# ---------------------------------------------------------------------------


@pytest.fixture
def borrower(tmp_path: Path, repo: Path) -> Path:
    """`hdae/lender` の `alpha.encoder.f32` を借りるリポ（自前の重みは 1 本）。"""
    root = tmp_path / "borrower"
    own = stage_component(root, ["gamma/decoder"] * 2, "model.f32", legacy_shards(mark="own"))
    lender = json.loads((repo / "karume.json").read_text(encoding="utf-8"))
    borrowed = [
        {**ref, "repo": "hdae/lender", "revision": "a" * 40}
        for ref in lender["models"]["alpha"]["weights"]["encoder"]["f32"]["shards"]
    ]
    alpha = model_entry(
        {
            "encoder": {"f32": {"shards": borrowed}},
            "decoder": {"f32": {"shards": own}},
        },
        {},
        {"full": quant({"encoder": "f32", "decoder": "f32"})},
    )
    write_manifest(root, {"alpha": alpha}, "alpha")
    return root


class TestTheCrossRepoReferences:
    def test_an_undeclared_repo_is_listed_with_its_count(
        self, borrower: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(MigrateError, match=r"hdae/lender（1 席）"):
            migrated(borrower, tmp_path / "out")

    def test_the_container_is_copied_with_the_new_pin(
        self, borrower: Path, repo: Path, tmp_path: Path
    ) -> None:
        lender_out = tmp_path / "lender"
        migrated(repo, lender_out)
        out = tmp_path / "out"
        migrated(
            borrower,
            out,
            cross_repos=[parse_cross_repo(f"hdae/lender={lender_out}@{REVISION}")],
        )
        crossed = container_of(out, "alpha", "encoder", "f32")

        assert (
            crossed["descriptor"]
            == container_of(lender_out, "alpha", "encoder", "f32")["descriptor"]
        )
        for ref, source in zip(
            crossed["parts"],
            container_of(lender_out, "alpha", "encoder", "f32")["parts"],
            strict=True,
        ):
            assert ref == {**source, "repo": "hdae/lender", "revision": REVISION}
        # 越境の列は自リポへ置かない。
        assert not (out / crossed["parts"][0]["path"]).exists()

    def test_the_lookup_follows_the_output_path_not_the_seat_name(
        self, borrower: Path, repo: Path, tmp_path: Path
    ) -> None:
        """引き写す鍵は part 0 の置き場（貸し手と借り手で席の名前が揃う保証は無い）。"""
        lender_out = tmp_path / "lender"
        migrated(repo, lender_out)
        manifest = json.loads((lender_out / "karume.json").read_text(encoding="utf-8"))
        # 貸し手の席を全部別名のモデルへ移す（容器の path は変えない）。
        manifest["models"] = {"renamed": manifest["models"]["alpha"]}
        manifest["defaultModel"] = "renamed"
        (lender_out / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")
        out = tmp_path / "out"
        migrated(
            borrower,
            out,
            cross_repos=[parse_cross_repo(f"hdae/lender={lender_out}@{REVISION}")],
        )
        crossed = container_of(out, "alpha", "encoder", "f32")

        assert crossed["parts"][0]["path"].startswith("shared/encoder/model.f32-")
        assert all(ref["repo"] == "hdae/lender" for ref in crossed["parts"])

    def test_a_container_that_is_not_in_the_converted_directory_lists_the_candidates(
        self, borrower: Path, repo: Path, tmp_path: Path
    ) -> None:
        lender_out = tmp_path / "lender"
        migrated(repo, lender_out)
        manifest = json.loads((lender_out / "karume.json").read_text(encoding="utf-8"))
        # 借り手が指す列（shared/encoder）が参照先から消えた形。
        for model in manifest["models"].values():
            model["weights"].pop("encoder", None)
        (lender_out / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(
            MigrateError,
            match=r"容器 .shared/encoder/model\.f32\.krm. が無い.*alpha/decoder/model\.f32\.krm",
        ):
            migrated(
                borrower,
                tmp_path / "out",
                cross_repos=[parse_cross_repo(f"hdae/lender={lender_out}@{REVISION}")],
            )

    def test_a_mixed_entry_fails_loudly(self, borrower: Path, tmp_path: Path) -> None:
        manifest = json.loads((borrower / "karume.json").read_text(encoding="utf-8"))
        shards = manifest["models"]["alpha"]["weights"]["encoder"]["f32"]["shards"]
        shards[1] = {key: shards[1][key] for key in ("path", "size", "sha256")}
        (borrower / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match="越境参照が混在"):
            migrated(borrower, tmp_path / "out")

    def test_the_same_repo_twice_fails_loudly(self, borrower: Path, tmp_path: Path) -> None:
        """どちらの変換済みディレクトリから引くかが決まらない。"""
        spec = parse_cross_repo(f"hdae/lender={tmp_path / 'lender'}@{REVISION}")

        with pytest.raises(MigrateError, match="同じ repo が 2 度"):
            migrated(borrower, tmp_path / "out", cross_repos=[spec, spec])

    @pytest.mark.parametrize(
        "spec",
        [
            "hdae/lender",
            "hdae/lender=/tmp/x",
            "lender=/tmp/x@" + "a" * 40,
            "hdae/lender=/tmp/x@abc",
        ],
    )
    def test_a_malformed_cross_repo_option_fails_loudly(self, spec: str) -> None:
        with pytest.raises(MigrateError, match="--cross-repo"):
            parse_cross_repo(spec)


# ---------------------------------------------------------------------------
# PLE sidecar → 容器の資産（索引 schema 3）
# ---------------------------------------------------------------------------


def ple_rows(start: int, stop: int, width: int, seed: int) -> bytes:
    """token 行そのもの（`start` から決まるので新旧の突合が行単位で効く）。"""
    return bytes(
        ((token * 31 + offset * 7 + seed) & 0xFF)
        for token in range(start, stop)
        for offset in range(width)
    )


def ple_common(layers: int = PLE_LAYERS, dim: int = PLE_DIM, *, schema: int = 2) -> dict[str, Any]:
    """旧索引 / shard メタデータの共通欄。schema 1（I8）は `storage` 欄を持たない（実資産の形）。"""
    return {
        "schema": schema,
        **({"storage": "i4"} if schema == 2 else {}),
        "tokens": PLE_TOKENS,
        "layers": layers,
        "dim": dim,
        "embedScale": 4.0,
    }


def stage_ple(
    root: Path,
    directory: str,
    *,
    layers: int = PLE_LAYERS,
    dim: int = PLE_DIM,
    schema: int = 2,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """旧 PLE sidecar（索引 + token 範囲 shard）を置き、`(索引の FileRef, assets)` を返す。

    `layers` / `dim` を動かせるのは、行バイト数から決まる門（4 の倍数 MUST）を踏むためである。
    """
    values_row = layers * dim // (2 if schema == 2 else 1)
    values_dtype = "I4" if schema == 2 else "I8"
    scales_row = layers * 4
    assets: dict[str, dict[str, Any]] = {}
    shards: list[dict[str, Any]] = []
    for position, (start, stop) in enumerate(PLE_RANGES, start=1):
        rows = stop - start
        name = numbered_name("ple.safetensors", position, len(PLE_RANGES))
        path = root / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payloads = {
            "values": ple_rows(start, stop, values_row, 11),
            "scales": ple_rows(start, stop, scales_row, 23),
        }
        blob = write_safetensors(
            order(
                [
                    Entry("values", values_dtype, (rows, layers, dim), payloads["values"]),
                    Entry("scales", "F32", (rows, layers), payloads["scales"]),
                ]
            ),
            {
                "karume_ple": json.dumps(
                    {**ple_common(layers, dim, schema=schema), "start": start, "stop": stop}
                )
            },
        )
        path.write_bytes(blob)
        assets[name] = {
            "path": f"{directory}/{name}",
            "size": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        shards.append({"file": name, "start": start, "stop": stop})
    index = place(
        root,
        f"{directory}/ple.json",
        (
            json.dumps({**ple_common(layers, dim, schema=schema), "shards": shards}, indent=2)
            + "\n"
        ).encode("utf-8"),
    )
    return index, assets


@pytest.fixture
def gemma_repo(tmp_path: Path) -> Path:
    return stage_gemma_repo(tmp_path / "gem")


def stage_gemma_repo(
    root: Path, *, layers: int = PLE_LAYERS, dim: int = PLE_DIM, schema: int = 2
) -> Path:
    """`gemma4` の PLE を持つリポ（部品 `model` に f32 / f16 の 2 席）。"""
    weights = {
        dtype: {
            "shards": stage_component(
                root, ["e2b/model"] * 2, f"model.{dtype}", legacy_shards(mark="g", storage=dtype)
            )
        }
        for dtype in ("f32", "f16")
    }
    index, ple = stage_ple(root, "e2b/ple", layers=layers, dim=dim, schema=schema)
    tokenizer = place(root, "tokenizer.json", b"{}\n")
    entry = model_entry(
        {"model": weights},
        {"tokenizer": tokenizer, "ple_index": index, **ple},
        {"full": quant({"model": "f32"}), "half": quant({"model": "f16"})},
        pipeline="gemma4",
    )
    write_manifest(root, {"e2b": entry}, "e2b")
    return root


def source_rows(repo: Path, directory: str, key: str) -> bytes:
    """旧 shard の `key` を token 順に連結した生バイト（突合の相手を別経路で取る）。"""
    out = bytearray()
    for position in range(1, len(PLE_RANGES) + 1):
        blob = (
            repo / directory / numbered_name("ple.safetensors", position, len(PLE_RANGES))
        ).read_bytes()
        length = struct.unpack("<Q", blob[:8])[0]
        spec = json.loads(blob[8 : 8 + length])[key]
        begin, end = spec["data_offsets"]
        out += blob[8 + length + begin : 8 + length + end]
    return bytes(out)


class TestThePleFold:
    def test_the_index_becomes_schema_3_over_asset_blocks(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        migrated(gemma_repo, out)
        read = opened(out, container_of(out, "e2b", "model", "f32"))
        index = json.loads(asset_payload(read, "ple_index"))

        assert list(index) == [
            "schema",
            "storage",
            "tokens",
            "layers",
            "dim",
            "embedScale",
            "values",
            "scales",
        ]
        assert index["schema"] == 3
        assert index["storage"] == "i4"
        assert index["tokens"] == PLE_TOKENS
        assert index["values"]["rowBytes"] == PLE_LAYERS * PLE_DIM // 2
        assert index["scales"]["rowBytes"] == PLE_LAYERS * 4
        # 旧 shard の境界（20 / 45）は消え、block 上限（512 B）だけが刻みを決める。
        assert [(b["start"], b["stop"]) for b in index["values"]["blocks"]] == [(0, 32), (32, 64)]
        assert [(b["start"], b["stop"]) for b in index["scales"]["blocks"]] == [(0, 64)]
        for key in ("values", "scales"):
            for block in index[key]["blocks"]:
                assert read.model.assets[block["asset"]].role == f"ple-{key}"

    def test_every_token_row_survives_byte_for_byte(self, gemma_repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        migrated(gemma_repo, out)
        read = opened(out, container_of(out, "e2b", "model", "f32"))
        index = json.loads(asset_payload(read, "ple_index"))

        for key in ("values", "scales"):
            joined = b"".join(asset_payload(read, block["asset"]) for block in index[key]["blocks"])
            assert joined == source_rows(gemma_repo, "e2b/ple", key)

    def test_a_range_read_block_sits_alone_in_its_part(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """区間読みを要する block は 1 block = 1 part（ADR 0109 決定 4 / container-v1 §4.2）。

        走査型の取得元（Deno の既定）は part を先頭から切るので、同居させると 1 行引くのに
        part の先頭から読み飛ばすことになる — §4.2 がこの MUST を置いた当の失敗形である。
        """
        out = tmp_path / "out"
        migrated(gemma_repo, out)
        read = opened(out, container_of(out, "e2b", "model", "f32"))
        index = json.loads(asset_payload(read, "ple_index"))
        occupants = Counter(block.part for block in read.model.blocks)
        part_of = {block.id: block.part for block in read.model.blocks}

        ranged = [
            read.model.assets[block["asset"]].block
            for key in ("values", "scales")
            for block in index[key]["blocks"]
        ]
        assert len(ranged) == 3  # values 2 本 + scales 1 本（空振りしない形で固定する）
        for block_id in ranged:
            assert occupants[part_of[block_id]] == 1

        # 全量読みの索引は専用 part を取らないが、重み block とは同居しない。
        index_part = part_of[read.model.assets["ple_index"].block]
        assert index_part not in {
            part_of[block.id] for block in read.model.blocks if block.role != "asset"
        }

    def test_a_schema_1_i8_sidecar_folds_with_storage_i8(self, tmp_path: Path) -> None:
        # karume-gemma4（非 QAT）の実資産は schema 1（I8・storage 欄なし）。新索引は格納を必ず綴る。
        repo = stage_gemma_repo(tmp_path / "gem1", schema=1)
        out = tmp_path / "out"
        migrated(repo, out)
        read = opened(out, container_of(out, "e2b", "model", "f32"))
        index = json.loads(asset_payload(read, "ple_index"))

        assert index["schema"] == 3
        assert index["storage"] == "i8"
        assert index["values"]["rowBytes"] == PLE_LAYERS * PLE_DIM
        joined = b"".join(
            asset_payload(read, block["asset"]) for block in index["values"]["blocks"]
        )
        assert joined == source_rows(repo, "e2b/ple", "values")

    def test_the_ple_assets_leave_the_manifest_and_the_files_are_not_copied(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        migrated(gemma_repo, out)
        manifest = manifest_of(out)

        assert set(manifest["models"]["e2b"]["assets"]) == {"tokenizer"}
        assert not (out / "e2b" / "ple").exists()

    def test_the_fold_reaches_every_dtype_of_the_owning_component(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        migrated(gemma_repo, out)

        for dtype in ("f32", "f16"):
            read = opened(out, container_of(out, "e2b", "model", dtype))
            assert "ple_index" in read.model.assets

    def test_a_ple_index_on_a_pipeline_without_an_owner_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        manifest = json.loads((gemma_repo / "karume.json").read_text(encoding="utf-8"))
        manifest["models"]["e2b"]["pipeline"] = "synthetic/1"
        (gemma_repo / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match="PLE の持ち主でない"):
            migrated(gemma_repo, tmp_path / "out")

    def test_a_shard_whose_metadata_disagrees_with_the_index_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """索引と `karume_ple` の食い違いは沈黙誤値の種（行の持ち主が 2 通りになる）。"""
        manifest = json.loads((gemma_repo / "karume.json").read_text(encoding="utf-8"))
        index_path = gemma_repo / manifest["models"]["e2b"]["assets"]["ple_index"]["path"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["shards"][1]["stop"] = 44
        index["shards"][2]["start"] = 44
        index_path.write_text(json.dumps(index), encoding="utf-8")

        with pytest.raises(MigrateError, match=r"karume_ple\.stop"):
            migrated(gemma_repo, tmp_path / "out")

    def test_a_dim_that_the_packing_cannot_divide_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """i4 は 1 バイトに 2 要素（割り切れない dim は行バイト数が決まらない）。"""
        manifest = json.loads((gemma_repo / "karume.json").read_text(encoding="utf-8"))
        index_path = gemma_repo / manifest["models"]["e2b"]["assets"]["ple_index"]["path"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["dim"] = PLE_DIM - 1
        index_path.write_text(json.dumps(index), encoding="utf-8")

        with pytest.raises(MigrateError, match=r"dim 15 が格納 .i4. の詰め数 2 で割り切れない"):
            migrated(gemma_repo, tmp_path / "out")

    def test_a_row_that_is_not_a_multiple_of_four_bytes_fails_loudly(self, tmp_path: Path) -> None:
        """block は行の倍数で切る（行が 4 の倍数でなければ block 長の整列と両立しない）。"""
        root = stage_gemma_repo(tmp_path / "thin", layers=1, dim=4)  # values の 1 行 = 2 バイト

        with pytest.raises(MigrateError, match=r"values の 1 行 2 バイトが 4 の倍数でない"):
            migrated(root, tmp_path / "out")

    def test_an_owner_without_a_local_seat_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """畳み先が 1 つも無いまま assets から消すと PLE が黙って配布物から落ちる。"""
        lender_out = tmp_path / "lender"
        migrated(gemma_repo, lender_out)
        lender = json.loads((gemma_repo / "karume.json").read_text(encoding="utf-8"))
        root = tmp_path / "borrower"
        index, ple = stage_ple(root, "e2b/ple")
        weights = {
            dtype: {
                "shards": [
                    {**ref, "repo": "hdae/lender", "revision": "a" * 40}
                    for ref in lender["models"]["e2b"]["weights"]["model"][dtype]["shards"]
                ]
            }
            for dtype in ("f32", "f16")
        }
        write_manifest(
            root,
            {
                "e2b": model_entry(
                    {"model": weights},
                    {"ple_index": index, **ple},
                    {"full": quant({"model": "f32"})},
                    pipeline="gemma4",
                )
            },
            "e2b",
        )

        with pytest.raises(MigrateError, match="自リポで変換する席が 1 つも無い"):
            migrated(
                root,
                tmp_path / "out",
                cross_repos=[parse_cross_repo(f"hdae/lender={lender_out}@{REVISION}")],
            )


class TestTheCli:
    def test_it_migrates_the_repository_and_prints_a_summary(
        self, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        migrate.main(
            ["--manifest", str(repo / "karume.json"), "--out", str(out), "--license", "apache-2.0"]
        )
        printed = capsys.readouterr().out

        assert (out / "karume.json").is_file()
        assert "containers=2" in printed and "crossed=0" in printed

    def test_a_positional_path_cannot_be_combined_with_the_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(MigrateError, match="--manifest と位置引数は併用できない"):
            migrate.main(
                [
                    "a/model.safetensors",
                    "--manifest",
                    "a/karume.json",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                ]
            )

    @pytest.mark.parametrize("extra", [["--graph-name", "x"], ["--graph"]])
    def test_the_component_only_options_are_refused(self, tmp_path: Path, extra: list[str]) -> None:
        with pytest.raises(MigrateError, match="部品単位モードの席"):
            migrate.main(
                [
                    "--manifest",
                    "a/karume.json",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                    *extra,
                ]
            )

    def test_cross_repo_is_refused_in_component_mode(self, tmp_path: Path) -> None:
        with pytest.raises(MigrateError, match="--cross-repo はリポ丸ごとモード"):
            migrate.main(
                [
                    "a/model.safetensors",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                    "--cross-repo",
                    f"hdae/x=/tmp/y@{REVISION}",
                ]
            )

    def test_it_requires_either_a_component_or_a_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(MigrateError, match="代表 path か --manifest"):
            migrate.main(["--out", str(tmp_path), "--license", "mit"])
