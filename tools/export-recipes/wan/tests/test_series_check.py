"""`wan/series_check.py`（書き直した成果物を既存の系列と比べる CLI — ADR 0121 段 4）の約束事。

門なので守るのは「違うのに一致と言わない」こと:

- 同じ系列は一致（終了コード 0）。safetensors の `__metadata__` のキー順だけの違いは一致
  （safetensors 0.8.0 はキー順がプロセスごとに変わる — ADR 0118 追記）
- テンソルの 1 ビット・dtype・形・名前の集合・メタの値、krm 風のバイナリの 1 バイトの違いは
  食い違いとして名指す（-0.0 と 0.0・payload の違う NaN も — 値の比較なら等しくなる組）
- どちらかの側に無い項目・必須ディレクトリの中の余分なファイル・リンク・FIFO・種類の違い・
  自分自身との比較・ファイルを 1 つも比べない必須ディレクトリも食い違い（skip しない）
- データ域の隙間・重なり・重複したキー・知らない dtype・収まらないヘッダ長は読めない
  safetensors として食い違い（テンソルの外のバイトを見落とさない）
- `--require` 無し・根の外を指す名前・重複・入れ子・NUL・ディレクトリでない根は引数の誤り（2）、
  読めないファイルは 3（食い違いの 1 と分ける）

小さな合成ファイルだけで回す（実物の照合のコマンドは `wan/README.md`）。
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

from wan import series_check

#: 合成の系列の項目（krm の part 風のディレクトリ 1 つ + 系列の根の safetensors 1 本）。
GRAPH = "graph"
FIXTURE = "fixture.safetensors"


def _raw(dtype: str, values: list[float]) -> bytes:
    return np.array(values, dtype=dtype).tobytes()


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    metadata: dict[str, str] | None = None,
) -> Path:
    """safetensors を手で書く（メタのキー順・テンソルの生バイトを指定どおりに置く）。"""
    header: dict[str, Any] = {} if metadata is None else {"__metadata__": metadata}
    data = b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    return _write_raw_safetensors(path, json.dumps(header), data)


def _write_raw_safetensors(path: Path, header: str, data: bytes) -> Path:
    """ヘッダの JSON を文字列のまま置く（重複したキー・壊れた data_offsets を作るため）。"""
    encoded = header.encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return path


def _series(root: Path) -> Path:
    """合成の系列（容器の part 2 つ + safetensors 本体が書いたフィクスチャ）。"""
    graph = root / GRAPH
    graph.mkdir(parents=True)
    (graph / "model-00001-of-00002.krm").write_bytes(b"KRM2" + bytes(range(64)))
    (graph / "model-00002-of-00002.krm").write_bytes(bytes(255 - i for i in range(200)))
    save_file(
        {
            "latents": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "frames": np.linspace(-1, 1, 30, dtype=np.float16).reshape(5, 6),
        },
        str(root / FIXTURE),
        metadata={"seed": "20261002", "chunks": "9", "role": "band"},
    )
    return root


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    """同じ中身を別のファイルとして持つ（書いた側, 既存）の組。"""
    existing = _series(tmp_path / "existing")
    written = tmp_path / "written"
    shutil.copytree(existing, written)
    return written, existing


def _run(
    capsys: pytest.CaptureFixture[str], written: Path, existing: Path, *required: str
) -> tuple[int, dict[str, Any]]:
    argv = ["--written", str(written), "--existing", str(existing)]
    for name in required:
        argv += ["--require", name]
    code = series_check.main(argv)
    return code, json.loads(capsys.readouterr().out)


def _reasons(summary: dict[str, Any]) -> list[tuple[str, str]]:
    return [(entry["path"], entry["reason"]) for entry in summary["mismatches"]]


def _tensor_pair(
    tmp_path: Path,
    written: dict[str, tuple[str, list[int], bytes]],
    existing: dict[str, tuple[str, list[int], bytes]],
    *,
    written_metadata: dict[str, str] | None = None,
    existing_metadata: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    roots = (tmp_path / "written", tmp_path / "existing")
    for root, tensors, metadata in zip(
        roots, (written, existing), (written_metadata, existing_metadata), strict=True
    ):
        root.mkdir()
        _write_safetensors(root / FIXTURE, tensors, metadata)
    return roots


class TestMatchingSeries:
    def test_an_identical_series_compares_every_file_and_exits_zero(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, summary = _run(capsys, *pair, GRAPH, FIXTURE)
        assert code == 0
        assert summary["match"] is True
        assert summary["mismatches"] == []
        assert [(entry["path"], entry["compare"]) for entry in summary["files"]] == [
            ("graph/model-00001-of-00002.krm", "sha256"),
            ("graph/model-00002-of-00002.krm", "sha256"),
            (FIXTURE, "safetensors"),
        ]
        assert summary["files"][2]["tensors"] == 2
        assert [(item["name"], item["files"]) for item in summary["items"]] == [
            (GRAPH, 2),
            (FIXTURE, 1),
        ]

    def test_metadata_key_order_alone_is_not_a_mismatch(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tensors = {"frames": ("F32", [2], _raw("float32", [1.0, 2.0]))}
        written, existing = _tensor_pair(
            tmp_path,
            tensors,
            tensors,
            written_metadata={"seed": "1", "role": "band"},
            existing_metadata={"role": "band", "seed": "1"},
        )
        assert (written / FIXTURE).read_bytes() != (existing / FIXTURE).read_bytes()
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 0
        assert summary["match"] is True


class TestTensorDifferences:
    def test_one_flipped_bit_names_the_tensor_its_first_offset_and_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = _raw("float32", [float(i) for i in range(12)])
        flipped = bytearray(base)
        flipped[5 * 4 + 1] ^= 0x01
        written, existing = _tensor_pair(
            tmp_path,
            {"latents": ("F32", [3, 4], base), "frames": ("F32", [12], bytes(flipped))},
            {"latents": ("F32", [3, 4], base), "frames": ("F32", [12], base)},
        )
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert mismatch["reason"] == "値のビットが違う"
        assert mismatch["tensor"] == "frames"
        assert mismatch["first_differing_byte_in_tensor"] == 21
        assert mismatch["first_differing_element"] == 5
        assert mismatch["differing_elements"] == 1
        assert mismatch["elements"] == 12

    def test_differences_across_chunks_count_elements_not_bytes(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 8 バイトずつ読ませる。先頭の chunk は同じ・要素 2 は 3 バイト・要素 6 は後の chunk で
        # 2 バイト違う（バイト数で数えると 5・chunk 内の位置だけなら先頭は 1）。
        monkeypatch.setattr(series_check, "CHUNK_BYTES", 8)
        base = _raw("float32", [float(i) for i in range(8)])
        changed = bytearray(base)
        for offset in (9, 10, 11, 24, 26):
            changed[offset] ^= 0x10
        written, existing = _tensor_pair(
            tmp_path,
            {"frames": ("F32", [8], bytes(changed))},
            {"frames": ("F32", [8], base)},
        )
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert mismatch["first_differing_byte_in_tensor"] == 9
        assert mismatch["first_differing_element"] == 2
        assert mismatch["differing_elements"] == 2

    def test_a_different_dtype_mismatches_even_with_equal_bytes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        raw = _raw("float32", [1.0, 2.0])
        written, existing = _tensor_pair(
            tmp_path, {"frames": ("I32", [2], raw)}, {"frames": ("F32", [2], raw)}
        )
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert (mismatch["reason"], mismatch["tensor"]) == ("dtype が違う", "frames")
        assert (mismatch["written"], mismatch["existing"]) == ("I32", "F32")

    def test_a_different_shape_mismatches_even_with_equal_bytes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        raw = _raw("float32", [float(i) for i in range(12)])
        written, existing = _tensor_pair(
            tmp_path, {"frames": ("F32", [3, 4], raw)}, {"frames": ("F32", [4, 3], raw)}
        )
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert (mismatch["reason"], mismatch["tensor"]) == ("形が違う", "frames")
        assert (mismatch["written"], mismatch["existing"]) == ([3, 4], [4, 3])

    def test_tensor_name_set_differences_are_named_both_ways(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        raw = _raw("float32", [1.0])
        written, existing = _tensor_pair(
            tmp_path,
            {"frames": ("F32", [1], raw), "extra": ("F32", [1], raw)},
            {"frames": ("F32", [1], raw), "frames_full": ("F32", [1], raw)},
        )
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        assert [(entry["reason"], entry["tensor"]) for entry in summary["mismatches"]] == [
            ("テンソルが既存に無い", "extra"),
            ("テンソルが書いた側に無い", "frames_full"),
        ]

    @pytest.mark.parametrize(
        ("written_bits", "existing_bits"),
        [
            pytest.param(0x80000000, 0x00000000, id="-0.0と0.0"),
            pytest.param(0x7FC00001, 0x7FC00000, id="payloadの違うNaN"),
        ],
    )
    def test_values_equal_by_comparison_still_mismatch_by_bits(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        written_bits: int,
        existing_bits: int,
    ) -> None:
        left = np.array([1.0, 0.0], dtype=np.float32)
        right = left.copy()
        left.view(np.uint32)[1] = written_bits
        right.view(np.uint32)[1] = existing_bits
        assert np.array_equal(left, right, equal_nan=True)
        written, existing = _tensor_pair(
            tmp_path,
            {"frames": ("F32", [2], left.tobytes())},
            {"frames": ("F32", [2], right.tobytes())},
        )
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert (mismatch["tensor"], mismatch["first_differing_element"]) == ("frames", 1)

    def test_bytes_outside_every_tensor_make_the_file_unreadable(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        written, existing = pair
        with (written / FIXTURE).open("ab") as stream:
            stream.write(b"\x00" * 8)
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert mismatch["reason"].startswith("書いた側が safetensors として読めない")


def _entry(dtype: str, shape: list[int], begin: int, end: int) -> str:
    return json.dumps({"dtype": dtype, "shape": shape, "data_offsets": [begin, end]})


class TestUnreadableSafetensors:
    """既存は正しいファイル・書いた側だけ取り決めから外れる（検査を外すと中身は一致してしまう組）。"""

    DATA = bytes(range(12))

    @pytest.mark.parametrize(
        ("header", "data", "existing", "phrase"),
        [
            pytest.param(
                f'{{"a": {_entry("F32", [2], 0, 8)}, "b": {_entry("F32", [2], 16, 24)}}}',
                DATA[:8] + b"\xee" * 8 + DATA[4:12],
                {"a": DATA[:8], "b": DATA[4:12]},
                "隙間か重なり",
                id="データ域の隙間",
            ),
            pytest.param(
                f'{{"a": {_entry("F32", [2], 0, 8)}, "b": {_entry("F32", [2], 4, 12)}}}',
                DATA,
                {"a": DATA[:8], "b": DATA[4:12]},
                "隙間か重なり",
                id="データ域の重なり",
            ),
            pytest.param(
                f'{{"a": {_entry("F32", [1], 0, 4)}, "a": {_entry("F32", [1], 0, 4)}}}',
                DATA[:4],
                {"a": DATA[:4]},
                "重複したキー",
                id="重複したキー",
            ),
            pytest.param(
                f'{{"a": {_entry("F4", [8], 0, 4)}}}',
                DATA[:4],
                {"a": DATA[:4]},
                "知らない",
                id="知らないdtype",
            ),
        ],
    )
    def test_a_header_that_breaks_the_layout_rules_is_unreadable(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        header: str,
        data: bytes,
        existing: dict[str, bytes],
        phrase: str,
    ) -> None:
        written_root, existing_root = tmp_path / "written", tmp_path / "existing"
        written_root.mkdir()
        existing_root.mkdir()
        _write_raw_safetensors(written_root / FIXTURE, header, data)
        _write_safetensors(
            existing_root / FIXTURE,
            {name: ("F32", [len(raw) // 4], raw) for name, raw in existing.items()},
        )
        code, summary = _run(capsys, written_root, existing_root, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert mismatch["reason"].startswith("書いた側が safetensors として読めない")
        assert phrase in mismatch["reason"]

    def test_a_header_length_beyond_the_file_is_unreadable(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        written, existing = pair
        data = bytearray((written / FIXTURE).read_bytes())
        data[:8] = struct.pack("<Q", len(data))
        (written / FIXTURE).write_bytes(bytes(data))
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert "収まらない" in mismatch["reason"]


class TestMetadataDifferences:
    def test_a_metadata_value_difference_names_the_key(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tensors = {"frames": ("F32", [1], _raw("float32", [1.0]))}
        written, existing = _tensor_pair(
            tmp_path,
            tensors,
            tensors,
            written_metadata={"seed": "1", "tile": "32"},
            existing_metadata={"seed": "1", "tile": "24"},
        )
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert (mismatch["reason"], mismatch["key"]) == ("メタの値が違う", "tile")
        assert (mismatch["written"], mismatch["existing"]) == ("32", "24")

    @pytest.mark.parametrize(
        ("written_metadata", "existing_metadata", "expected"),
        [
            pytest.param(
                {"seed": "1", "tile": "32"},
                {"seed": "1"},
                ("メタのキーが既存に無い", "tile", "written", "32"),
                id="書いた側だけのキー",
            ),
            pytest.param(
                {"seed": "1"},
                {"seed": "1", "tile": "32"},
                ("メタのキーが書いた側に無い", "tile", "existing", "32"),
                id="既存だけのキー",
            ),
        ],
    )
    def test_a_key_on_one_side_only_names_the_key(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        written_metadata: dict[str, str],
        existing_metadata: dict[str, str],
        expected: tuple[str, str, str, str],
    ) -> None:
        tensors = {"frames": ("F32", [1], _raw("float32", [1.0]))}
        written, existing = _tensor_pair(
            tmp_path,
            tensors,
            tensors,
            written_metadata=written_metadata,
            existing_metadata=existing_metadata,
        )
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        [mismatch] = summary["mismatches"]
        reason, key, side, value = expected
        assert (mismatch["reason"], mismatch["key"], mismatch[side]) == (reason, key, value)

    @pytest.mark.parametrize(
        ("written_metadata", "existing_metadata", "reason"),
        [
            pytest.param(None, {}, "__metadata__ が書いた側に無い", id="書いた側に無い"),
            pytest.param({}, None, "__metadata__ が既存に無い", id="既存に無い"),
        ],
    )
    def test_absent_metadata_is_not_the_same_as_an_empty_one(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        written_metadata: dict[str, str] | None,
        existing_metadata: dict[str, str] | None,
        reason: str,
    ) -> None:
        tensors = {"frames": ("F32", [1], _raw("float32", [1.0]))}
        written, existing = _tensor_pair(
            tmp_path,
            tensors,
            tensors,
            written_metadata=written_metadata,
            existing_metadata=existing_metadata,
        )
        code, summary = _run(capsys, written, existing, FIXTURE)
        assert code == 1
        assert _reasons(summary) == [(FIXTURE, reason)]


class TestBinaryDifferences:
    def test_one_byte_in_a_container_like_binary_is_named(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        written, existing = pair
        part = written / GRAPH / "model-00002-of-00002.krm"
        data = bytearray(part.read_bytes())
        data[100] ^= 0xFF
        part.write_bytes(bytes(data))
        code, summary = _run(capsys, written, existing, GRAPH)
        assert code == 1
        assert _reasons(summary) == [("graph/model-00002-of-00002.krm", "sha256 が違う")]


class TestPresence:
    @pytest.mark.parametrize(
        ("side", "reason"), [("written", "書いた側に無い"), ("existing", "既存に無い")]
    )
    def test_an_item_missing_on_either_side_mismatches(
        self,
        pair: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
        side: str,
        reason: str,
    ) -> None:
        written, existing = pair
        (written if side == "written" else existing).joinpath(FIXTURE).unlink()
        code, summary = _run(capsys, written, existing, GRAPH, FIXTURE)
        assert code == 1
        assert _reasons(summary) == [(FIXTURE, reason)]

    def test_an_extra_file_inside_a_required_directory_mismatches_when_nested(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        written, existing = pair
        (written / GRAPH / "sub").mkdir()
        (written / GRAPH / "sub" / "extra.krm").write_bytes(b"x")
        code, summary = _run(capsys, written, existing, GRAPH)
        assert code == 1
        assert _reasons(summary) == [
            ("graph/sub", "既存に無い"),
            ("graph/sub/extra.krm", "既存に無い"),
        ]

    def test_a_symlink_is_not_followed_and_mismatches(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        written, existing = pair
        part = written / GRAPH / "model-00001-of-00002.krm"
        part.unlink()
        part.symlink_to(existing / GRAPH / "model-00001-of-00002.krm")
        code, summary = _run(capsys, written, existing, GRAPH)
        assert code == 1
        assert _reasons(summary) == [
            ("graph/model-00001-of-00002.krm", "書いた側がシンボリックリンク（辿らない）")
        ]

    @pytest.mark.parametrize(
        ("link", "required"),
        [
            pytest.param(FIXTURE, FIXTURE, id="項目そのもの"),
            pytest.param(GRAPH, f"{GRAPH}/model-00001-of-00002.krm", id="途中の成分"),
        ],
    )
    def test_a_symlink_in_the_required_path_is_not_followed(
        self,
        tmp_path: Path,
        pair: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
        link: str,
        required: str,
    ) -> None:
        # リンク先は既存とは別の実体の同じ中身 — 辿れば一致になってしまう組。
        written, existing = pair
        target = tmp_path / "elsewhere" / link
        target.parent.mkdir()
        if (written / link).is_dir():
            shutil.copytree(written / link, target)
            shutil.rmtree(written / link)
        else:
            shutil.copy2(written / link, target)
            (written / link).unlink()
        (written / link).symlink_to(target)
        code, summary = _run(capsys, written, existing, required)
        assert code == 1
        assert _reasons(summary) == [(link, "書いた側がシンボリックリンク（辿らない）")]

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(FIXTURE, id="項目"),
            pytest.param(f"{GRAPH}/model-00001-of-00002.krm", id="ディレクトリの中"),
        ],
    )
    def test_a_directory_where_the_other_side_has_a_file_mismatches(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str], path: str
    ) -> None:
        written, existing = pair
        (written / path).unlink()
        (written / path).mkdir()
        code, summary = _run(capsys, written, existing, path.split("/")[0])
        assert code == 1
        [mismatch] = summary["mismatches"]
        assert (mismatch["path"], mismatch["reason"]) == (path, "種類が違う")
        assert (mismatch["written"], mismatch["existing"]) == ("dir", "file")

    def test_a_fifo_on_both_sides_is_not_compared_and_mismatches(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        written, existing = pair
        for root in pair:
            os.mkfifo(root / GRAPH / "pipe")
        code, summary = _run(capsys, written, existing, GRAPH)
        assert code == 1
        assert _reasons(summary) == [
            ("graph/pipe", "書いた側が通常ファイルでもディレクトリでもない"),
            ("graph/pipe", "既存が通常ファイルでもディレクトリでもない"),
        ]

    def test_a_required_directory_without_any_file_mismatches(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        written, existing = pair
        for root in pair:
            (root / "empty" / "sub").mkdir(parents=True)
        code, summary = _run(capsys, written, existing, "empty")
        assert code == 1
        assert _reasons(summary) == [("empty", "必須ディレクトリに比べるファイルが無い")]
        assert summary["items"][0]["files"] == 0

    def test_the_same_root_on_both_sides_mismatches_as_a_self_comparison(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, existing = pair
        code, summary = _run(capsys, existing, existing, GRAPH, FIXTURE)
        assert code == 1
        assert [path for path, _ in _reasons(summary)] == [GRAPH, FIXTURE]


class TestArguments:
    def test_no_require_is_a_usage_error(self, pair: tuple[Path, Path]) -> None:
        written, existing = pair
        with pytest.raises(SystemExit) as raised:
            series_check.main(["--written", str(written), "--existing", str(existing)])
        assert raised.value.code == 2

    @pytest.mark.parametrize("name", ["/etc/passwd", "../existing/graph", "graph/../..", "."])
    def test_a_require_outside_the_root_or_the_root_itself_is_a_usage_error(
        self, pair: tuple[Path, Path], name: str
    ) -> None:
        written, existing = pair
        with pytest.raises(SystemExit) as raised:
            series_check.main(
                ["--written", str(written), "--existing", str(existing), "--require", name]
            )
        assert raised.value.code == 2

    @pytest.mark.parametrize(
        "required",
        [
            pytest.param([FIXTURE, FIXTURE], id="重複"),
            pytest.param([GRAPH, f"{GRAPH}/model-00001-of-00002.krm"], id="入れ子"),
            pytest.param([f"{GRAPH}\0x"], id="NUL"),
        ],
    )
    def test_duplicate_nested_or_nul_requires_are_usage_errors(
        self, pair: tuple[Path, Path], required: list[str]
    ) -> None:
        written, existing = pair
        argv = ["--written", str(written), "--existing", str(existing)]
        for name in required:
            argv += ["--require", name]
        with pytest.raises(SystemExit) as raised:
            series_check.main(argv)
        assert raised.value.code == 2

    def test_a_root_that_is_not_a_directory_is_a_usage_error(self, pair: tuple[Path, Path]) -> None:
        written, existing = pair
        with pytest.raises(SystemExit) as raised:
            series_check.main(
                [
                    "--written",
                    str(written / FIXTURE),
                    "--existing",
                    str(existing),
                    "--require",
                    FIXTURE,
                ]
            )
        assert raised.value.code == 2

    @pytest.mark.skipif(os.geteuid() == 0, reason="root は権限を無視して読める")
    def test_an_unreadable_file_exits_three_with_no_summary(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        written, existing = pair
        part = written / GRAPH / "model-00001-of-00002.krm"
        part.chmod(0)
        try:
            code = series_check.main(
                ["--written", str(written), "--existing", str(existing), "--require", GRAPH]
            )
        finally:
            part.chmod(0o644)
        captured = capsys.readouterr()
        assert code == 3
        assert captured.out == ""
        assert "読めない" in captured.err
