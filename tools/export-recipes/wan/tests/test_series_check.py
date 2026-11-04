"""The contract of `wan/series_check.py` (the CLI that compares rewritten outputs with an existing
series — ADR 0121 step 4).

Because it is a check, what it guards is never saying "match" when the two differ:

- The same series matches (exit code 0). A difference only in the key order of a safetensors
  `__metadata__` matches (in safetensors 0.8.0 the key order changes from process to process — ADR
  0118 addendum)
- One bit of a tensor, a dtype, a shape, the set of names, a metadata value, or one byte of a
  krm-like binary is named as a mismatch (-0.0 vs 0.0 and NaNs with different payloads too — pairs
  that a value comparison would call equal)
- An item missing on either side, an extra file inside a required directory, a link, a FIFO, a kind
  difference, a self-comparison and a required directory with no file to compare are mismatches
  too (not skipped)
- A gap or an overlap in the data area, duplicate keys, an unknown dtype or a header length that
  does not fit make the file unreadable as safetensors, which is a mismatch (bytes outside the
  tensors are not missed)
- No `--require`, a name pointing outside the root, a duplicate, nesting, NUL and a root that is
  not a directory are argument errors (2), and an unreadable file is 3 (kept apart from the 1 of a
  mismatch)

Runs only on small synthetic files (the commands for checking the real outputs are in
`wan/README.md`).
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

#: The items of the synthetic series (one directory like a krm part + one safetensors file at the
#: root of the series).
GRAPH = "graph"
FIXTURE = "fixture.safetensors"


def _raw(dtype: str, values: list[float]) -> bytes:
    return np.array(values, dtype=dtype).tobytes()


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a safetensors file by hand (the metadata key order and the raw tensor bytes are placed
    exactly as given)."""
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
    """Write the header JSON as a raw string (to produce duplicate keys and broken data_offsets)."""
    encoded = header.encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return path


def _series(root: Path) -> Path:
    """A synthetic series (two container parts + a fixture written by safetensors itself)."""
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
    """A (written side, existing) pair holding the same contents as separate files."""
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
        assert mismatch["reason"] == "value bits differ"
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
        # Read 8 bytes at a time. The first chunk is the same; element 2 differs in 3 bytes and
        # element 6, in a later chunk, in 2 bytes (5 when counted in bytes; the first position
        # would be 1 if only the position within the chunk were counted).
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
        assert (mismatch["reason"], mismatch["tensor"]) == ("dtype differs", "frames")
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
        assert (mismatch["reason"], mismatch["tensor"]) == ("shape differs", "frames")
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
            ("tensor missing on the existing side", "extra"),
            ("tensor missing on the written side", "frames_full"),
        ]

    @pytest.mark.parametrize(
        ("written_bits", "existing_bits"),
        [
            pytest.param(0x80000000, 0x00000000, id="-0.0 vs 0.0"),
            pytest.param(0x7FC00001, 0x7FC00000, id="NaN with a different payload"),
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
        assert mismatch["reason"].startswith("written side is not readable as safetensors")


def _entry(dtype: str, shape: list[int], begin: int, end: int) -> str:
    return json.dumps({"dtype": dtype, "shape": shape, "data_offsets": [begin, end]})


class TestUnreadableSafetensors:
    """The existing side is a valid file and only the written side breaks the format (pairs whose
    contents would match if the checks were removed)."""

    DATA = bytes(range(12))

    @pytest.mark.parametrize(
        ("header", "data", "existing", "phrase"),
        [
            pytest.param(
                f'{{"a": {_entry("F32", [2], 0, 8)}, "b": {_entry("F32", [2], 16, 24)}}}',
                DATA[:8] + b"\xee" * 8 + DATA[4:12],
                {"a": DATA[:8], "b": DATA[4:12]},
                "gap or overlap",
                id="gap in the data area",
            ),
            pytest.param(
                f'{{"a": {_entry("F32", [2], 0, 8)}, "b": {_entry("F32", [2], 4, 12)}}}',
                DATA,
                {"a": DATA[:8], "b": DATA[4:12]},
                "gap or overlap",
                id="overlap in the data area",
            ),
            pytest.param(
                f'{{"a": {_entry("F32", [1], 0, 4)}, "a": {_entry("F32", [1], 0, 4)}}}',
                DATA[:4],
                {"a": DATA[:4]},
                "duplicate keys",
                id="duplicate keys",
            ),
            pytest.param(
                f'{{"a": {_entry("F4", [8], 0, 4)}}}',
                DATA[:4],
                {"a": DATA[:4]},
                "unknown",
                id="unknown dtype",
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
        assert mismatch["reason"].startswith("written side is not readable as safetensors")
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
        assert "does not fit" in mismatch["reason"]


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
        assert (mismatch["reason"], mismatch["key"]) == ("metadata value differs", "tile")
        assert (mismatch["written"], mismatch["existing"]) == ("32", "24")

    @pytest.mark.parametrize(
        ("written_metadata", "existing_metadata", "expected"),
        [
            pytest.param(
                {"seed": "1", "tile": "32"},
                {"seed": "1"},
                ("metadata key missing on the existing side", "tile", "written", "32"),
                id="key only on the written side",
            ),
            pytest.param(
                {"seed": "1"},
                {"seed": "1", "tile": "32"},
                ("metadata key missing on the written side", "tile", "existing", "32"),
                id="key only on the existing side",
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
            pytest.param(
                None, {}, "__metadata__ missing on the written side", id="missing on written"
            ),
            pytest.param(
                {}, None, "__metadata__ missing on the existing side", id="missing on existing"
            ),
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
        assert _reasons(summary) == [("graph/model-00002-of-00002.krm", "sha256 differs")]


class TestPresence:
    @pytest.mark.parametrize(
        ("side", "reason"),
        [("written", "missing on the written side"), ("existing", "missing on the existing side")],
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
            ("graph/sub", "missing on the existing side"),
            ("graph/sub/extra.krm", "missing on the existing side"),
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
            ("graph/model-00001-of-00002.krm", "written side is a symbolic link (not followed)")
        ]

    @pytest.mark.parametrize(
        ("link", "required"),
        [
            pytest.param(FIXTURE, FIXTURE, id="the item itself"),
            pytest.param(
                GRAPH, f"{GRAPH}/model-00001-of-00002.krm", id="an intermediate component"
            ),
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
        # The link target is a different file from the existing side with the same contents — a
        # pair that would match if the link were followed.
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
        assert _reasons(summary) == [(link, "written side is a symbolic link (not followed)")]

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(FIXTURE, id="the item"),
            pytest.param(f"{GRAPH}/model-00001-of-00002.krm", id="inside a directory"),
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
        assert (mismatch["path"], mismatch["reason"]) == (path, "kind differs")
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
            ("graph/pipe", "written side is neither a regular file nor a directory"),
            ("graph/pipe", "existing side is neither a regular file nor a directory"),
        ]

    def test_a_required_directory_without_any_file_mismatches(
        self, pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        written, existing = pair
        for root in pair:
            (root / "empty" / "sub").mkdir(parents=True)
        code, summary = _run(capsys, written, existing, "empty")
        assert code == 1
        assert _reasons(summary) == [("empty", "required directory has no file to compare")]
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
            pytest.param([FIXTURE, FIXTURE], id="duplicate"),
            pytest.param([GRAPH, f"{GRAPH}/model-00001-of-00002.krm"], id="nested"),
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

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads regardless of permissions")
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
        assert "cannot read" in captured.err
