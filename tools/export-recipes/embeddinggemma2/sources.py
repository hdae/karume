"""Where EmbeddingGemma 2's upstream snapshot is read from, and its pin.

The snapshot is `google/embeddinggemma-2` at one commit ({@link PIN}), fetched with

    hf download google/embeddinggemma-2 --revision <PIN.revision> \\
        --local-dir inputs/embeddinggemma2/embeddinggemma-2

into `inputs/embeddinggemma2/embeddinggemma-2/` (`inputs/<family>/<name>/` — assets-layout; the
name is the tail of the repository id, as gemma4 names its hand-placed checkpoints). The
repository is still being updated, so following `main` would read whatever the branch points at
on the day of the run; the pin names the commit and the SHA-256 of every file a script of this
family reads.

A `--local-dir` snapshot holds plain files (not links named by their hash), so a check hashes the
bytes. {@link assert_pinned} checks only the files a script names — the tokenizer script does not
hash the 1.49 GB `model.safetensors` it never reads.

MUST: torch-free and transformers-free — the pure tests import it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from _shared.paths import INPUTS_ROOT


class SourcePinError(ValueError):
    """A file on disk is not the pinned one (missing, or other bytes)."""


@dataclass(frozen=True)
class SourcePin:
    """One repository's commit and the SHA-256 of the files read from it."""

    #: The HF repository id.
    repo: str
    #: The HF commit SHA (40 lower-case hex digits) — never a branch or a tag.
    revision: str
    #: Snapshot-relative path → SHA-256 of its bytes at {@link revision}.
    sha256: Mapping[str, str]

    @property
    def label(self) -> str:
        """`<repo>@<revision>` — how a git-tracked record names this source."""
        return f"{self.repo}@{self.revision}"


#: `google/embeddinggemma-2` (Apache-2.0 per the README front matter; not gated), fetched on
#: 2026-11-21. `model.safetensors` is 1,488,915,288 B and holds all three towers; the text tower is
#: its 413 `language_model.*` tensors (271,002,648 bf16 parameters, bytes
#: [612,180,776, 1,154,186,072) of the file). Each digest matched the fetch record of
#: `hf download` (the LFS oid for the two large files, the git blob id for the others).
#:
#: Moving the pin needs the licence review again.
PIN = SourcePin(
    repo="google/embeddinggemma-2",
    revision="f9972ac22e696a105c482202646979e33e4b5a1e",
    sha256={
        "model.safetensors": "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
        "tokenizer.json": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
        "config.json": "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
        "modules.json": "3d02572a0455b832de67fb8e63a54981bc7e8b46e337c95e917bd8122a533bfd",
        "1_Pooling/config.json": (
            "8759bdf7c77efc7df7723f64856a593c8943b71ee38baf2a88771fbaf78438f9"
        ),
        "2_Normalize/config.json": (
            "cdb09dfca347a56aa2d691744e38d5ad3c7cbc2834e7181272b9a15328b82524"
        ),
        "config_sentence_transformers.json": (
            "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5"
        ),
        "sentence_bert_config.json": (
            "b1bcd9f2dce3ae863b359e87d0710b5dbc3314a59ecb4e2f97c7778fc8e4b228"
        ),
        "tokenizer_config.json": (
            "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874"
        ),
    },
)

#: The hand-placed snapshot ({@link PIN}).
SNAPSHOT_DIR = INPUTS_ROOT / "embeddinggemma2" / PIN.repo.split("/", 1)[1]


def file_sha256(path: Path) -> str:
    """The SHA-256 of `path`'s bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_pinned(directory: Path, names: Iterable[str], pin: SourcePin = PIN) -> None:
    """Every file of `names` is in `directory` with the SHA-256 `pin` gives it.

    MUST: run before a script reads any of them — another revision placed at the same path
    otherwise changes the outputs without a trace. A name the pin does not know fails too (a
    script reading an unpinned file is the slip this check exists for).
    """
    for name in names:
        expected = pin.sha256.get(name)
        if expected is None:
            raise SourcePinError(f"{name} is not pinned in {pin.label}")
        path = directory / name
        if not path.is_file():
            raise SourcePinError(
                f"{path} is missing — fetch the pinned snapshot first: hf download {pin.repo}"
                f" --revision {pin.revision} --local-dir {directory}"
            )
        actual = file_sha256(path)
        if actual != expected:
            raise SourcePinError(f"{path}: SHA-256 {actual}, pin {expected} ({pin.label})")
