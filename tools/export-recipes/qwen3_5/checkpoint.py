"""The checkpoint the Qwen3.5 text tower is exported from: its pin, and a reader that hands out
one tensor, or one block of rows, at a time.

{@link SOURCES} is the source of truth for where the weights come from. The only row today is
MOSS-Image's bundled text encoder (`OpenMOSS-Team/MOSS-Image-SFT`, subfolder `text_encoder`, a
`Qwen3_5Model` — the vision tower and the language model, no `lm_head`): the revision is the commit
SHA MOSS-Image's own recipe pins, and every shard is pinned by its LFS SHA-256. The weights are
read from the local Hugging Face cache; this module never downloads (the MOSS recipe's
`--fetch --text` fetches them).

Why the MOSS bundle and not `Qwen/Qwen3.5-9B`: it is what MOSS-Image ran when its DiT was trained
and its references were made. The two agree on every tensor (full comparison 2026-11-01 —
`moss.sources.TEXT_ENCODER_ORIGIN`): upstream Qwen stores `linear_attn.A_log` and
`linear_attn.norm.weight` in F32, which the bundle rounds to BF16.

The reader ({@link Checkpoint}) widens BF16 to F32 when it reads (lossless). It accepts only the
text tower's keys (`language_model.*`); the vision tower's are ignored by design (MOSS-Image never
runs it), and anything else fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

#: The checkpoint's index (transformers' `save_pretrained` spelling).
CHECKPOINT_INDEX = "model.safetensors.index.json"

#: Storage dtypes the reader widens to F32 (both are exact in F32).
ACCEPTED_DTYPES = frozenset({"BF16", "F32"})


class CheckpointError(RuntimeError):
    """The checkpoint is not the pinned one, or it does not hold what the graph needs."""


@dataclass(frozen=True)
class TextTowerSource:
    """One pinned checkpoint of a Qwen3.5 text tower."""

    #: The label in series names (`qwen3.5-9b-text-<name>-<storage>-dyn`).
    name: str
    repo: str
    #: The HF commit SHA (40 lower-case hex digits) — never a branch or a tag.
    revision: str
    subfolder: str
    #: `license:` of the repository's README front matter.
    license: str
    #: The prefix of the text tower's keys in the checkpoint (`Qwen3_5Model.language_model`).
    prefix: str
    #: Keys outside {@link prefix} that the checkpoint holds and the export ignores (by prefix).
    ignored_prefixes: tuple[str, ...]
    #: Shard file → LFS SHA-256 (the HF cache names the blob of an LFS file by it).
    shards: Mapping[str, str]


#: MOSS-Image's text encoder (Qwen3.5-9B, BF16, 4 shards / 16.79 GB in all, vision tower
#: included). The revision is `main` of 2026-10-27 (the MOSS recipe's pin); the shard digests
#: are the blob names of that snapshot, and shard 4's was recomputed on 2026-10-31.
MOSS_IMAGE_SFT = TextTowerSource(
    name="moss-image-sft",
    repo="OpenMOSS-Team/MOSS-Image-SFT",
    revision="f972bd4b4b9e3852133cc5da954d87bc32cd8fb9",
    subfolder="text_encoder",
    license="apache-2.0",
    prefix="language_model.",
    ignored_prefixes=("visual.",),
    shards={
        "model-00001-of-00004.safetensors": (
            "abfaa3d3a8612508c85ee8f6da9c734b862f10298b903765c4ad384177c32b0b"
        ),
        "model-00002-of-00004.safetensors": (
            "9112637fa6bf4d07bafd6ae268852b18d3d0c1242f4ee901c5579006e5bd4894"
        ),
        "model-00003-of-00004.safetensors": (
            "956f1ef51d0cfe610b0d22987c0b7821c36e3ce228abceccef523518bcf079dc"
        ),
        "model-00004-of-00004.safetensors": (
            "9e21d20e8adde88c3e0b65d4a0d251f143d98ee9d52db36b4348361091e4e208"
        ),
    },
)

#: The pinned checkpoints by label.
SOURCES: Mapping[str, TextTowerSource] = {MOSS_IMAGE_SFT.name: MOSS_IMAGE_SFT}
DEFAULT_SOURCE = MOSS_IMAGE_SFT.name


def hub_cache() -> Path:
    """The Hugging Face hub cache (`HF_HUB_CACHE`, else `HF_HOME/hub`, else
    `~/.cache/huggingface/hub` — huggingface_hub's own resolution order)."""
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def snapshot_dir(source: TextTowerSource, cache: Path | None = None) -> Path:
    """The pinned snapshot's subfolder in the local cache (fails loudly when it is not there)."""
    root = (cache or hub_cache()) / f"models--{source.repo.replace('/', '--')}"
    directory = root / "snapshots" / source.revision / source.subfolder
    if not (directory / CHECKPOINT_INDEX).is_file():
        raise CheckpointError(
            f"{directory} has no {CHECKPOINT_INDEX} — fetch {source.repo}@{source.revision}"
            f" ({source.subfolder}) first"
        )
    return directory


def assert_pinned_shards(
    directory: Path, source: TextTowerSource, *, hash_contents: bool = False
) -> None:
    """The index names exactly the pinned shards, and each one is the pinned blob.

    By default the check reads the cache's blob name (the LFS SHA-256 — cheap); `hash_contents`
    hashes the bytes too (about 10 s per 5 GB shard).
    """
    index = json.loads((directory / CHECKPOINT_INDEX).read_text(encoding="utf-8"))
    named = set(index["weight_map"].values())
    if named != set(source.shards):
        raise CheckpointError(
            f"{directory}: the index names shards {sorted(named)}, the pin {sorted(source.shards)}"
        )
    for name, expected in sorted(source.shards.items()):
        path = directory / name
        if not path.is_file():
            raise CheckpointError(f"{path} is missing")
        if path.is_symlink() and path.resolve().name != expected:
            raise CheckpointError(f"{path} resolves to blob {path.resolve().name}, pin {expected}")
        if hash_contents:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise CheckpointError(f"{path}: SHA-256 {actual}, pin {expected}")


class Checkpoint:
    """Reads the text tower's tensors from a sharded safetensors checkpoint, widened to F32.

    Keys are in the wrapper's name space (`embed_tokens.weight`, `layers.<n>.…`, `norm.weight`):
    the reader adds `prefix` (`language_model.` for a `Qwen3_5Model` checkpoint; empty for a
    `Qwen3_5TextModel` one). MUST: a key outside `prefix` that is not under one of
    `ignored_prefixes` fails loudly — reading only what the graph asks for would silently accept
    a checkpoint of another structure.
    """

    def __init__(
        self, directory: Path, *, prefix: str, ignored_prefixes: tuple[str, ...] = ()
    ) -> None:
        index = directory / CHECKPOINT_INDEX
        if not index.is_file():
            raise CheckpointError(f"{directory} has no {CHECKPOINT_INDEX}")
        weight_map: Mapping[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        stray = sorted(
            key
            for key in weight_map
            if not key.startswith(prefix) and not key.startswith(ignored_prefixes)
        )
        if stray:
            raise CheckpointError(
                f"{directory}: {len(stray)} keys outside '{prefix}' and the ignored prefixes"
                f" {list(ignored_prefixes)}: {stray[:4]}"
            )
        self.directory = directory
        self.prefix = prefix
        self._files = {
            key.removeprefix(prefix): directory / name
            for key, name in weight_map.items()
            if key.startswith(prefix)
        }

    @classmethod
    def pinned(cls, source: TextTowerSource, directory: Path | None = None) -> Checkpoint:
        """The pinned checkpoint (the cache's snapshot unless `directory` is given), after
        {@link assert_pinned_shards}."""
        directory = directory if directory is not None else snapshot_dir(source)
        assert_pinned_shards(directory, source)
        return cls(directory, prefix=source.prefix, ignored_prefixes=source.ignored_prefixes)

    def names(self) -> frozenset[str]:
        return frozenset(self._files)

    def _path(self, key: str) -> Path:
        if key not in self._files:
            raise CheckpointError(f"'{self.prefix}{key}' is not in the checkpoint")
        return self._files[key]

    def _assert_dtype(self, key: str, dtype: str) -> None:
        if dtype not in ACCEPTED_DTYPES:
            raise CheckpointError(
                f"'{self.prefix}{key}' is {dtype} (only {sorted(ACCEPTED_DTYPES)})"
            )

    def shape(self, key: str) -> list[int]:
        with safe_open(str(self._path(key)), framework="pt") as handle:
            view = handle.get_slice(f"{self.prefix}{key}")
            self._assert_dtype(key, view.get_dtype())
            return list(view.get_shape())

    def read(self, key: str) -> torch.Tensor:
        """One whole tensor as F32."""
        with safe_open(str(self._path(key)), framework="pt") as handle:
            view = handle.get_slice(f"{self.prefix}{key}")
            self._assert_dtype(key, view.get_dtype())
            return view[:].to(torch.float32)

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        """Rows `[start, stop)` of the leading axis as F32."""
        with safe_open(str(self._path(key)), framework="pt") as handle:
            view = handle.get_slice(f"{self.prefix}{key}")
            self._assert_dtype(key, view.get_dtype())
            return view[start:stop].to(torch.float32)
