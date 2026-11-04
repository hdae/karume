"""Wan's upstream sources (HF repositories with pinned revisions) and component fetching (ADR 0118
decisions 5 / 7, 0121 decision 1).

`SOURCES` is the source of truth for the table of upstream sources. Each revision is a commit SHA
(40 digits) resolved through the HF API — ADR 0092 decision 3 (the practice of baking the revision
of a distribution repository) applied to the upstream side as well. The recipe scripts read the
upstream through this module — if each script spelled `from_pretrained(repo)` itself, the day the
head of the default branch moved, the reference and the export would silently read a different
checkpoint.

Whether a component is fetched ({@link COMPONENTS}) and the expected parameter counts
({@link EXPECTED_PARAMETERS}) are tables keyed by model name (ADR 0121 decision 1 — the generations
treat components differently). Every model fetches only the DiT (`transformer`), the VAE (`vae`),
the scheduler config, the `README.md` that carries the license front matter, and the component
index `model_index.json`.

- Wan2.1 (`t2v-1.3b`): umT5 (`text_encoder`, about 22.7 GB in f32) and the tokenizer are not
  fetched with the DiT / VAE — the separate text-embedding process of step 6 (`wan.text_embeds`)
  fetches them with {@link text_snapshot} (umT5 and the DiT are not held together in the 31 GiB of
  host RAM of the development machine — ADR 0118 decision 4).
- Wan2.2 (`ti2v-5b`): `text_encoder` and `tokenizer` are fetched on no path (ADR 0121 decision 9 —
  umT5 is a cross-repository reference to `karume-umt5-xxl`, and the tokenizer assets are copied
  from the Wan2.1 series).

The upstream of the umT5 container is not Wan but the original `google/umt5-xxl`
({@link UMT5_SOURCES} — ADR 0122 decision 1). The original only has the sharded pickle `.bin` form,
so the fetch entry point ({@link umt5_snapshot}) derives from the index the shards the encoder
needs, matches them against the shards of the table (pinned by sha256) and fetches only those
(decision 2). The tokenizer assets that `karume-umt5-xxl` distributes on its own are baked
(`wan.umt5_xxl_tokenizer`) from the original tokenizer files (pinned by sha256 in the same table —
{@link pinned_umt5_tokenizer}). The Wan snapshot ({@link text_snapshot}) stays for the Wan tokenizer
assets (with the preprocessing tables), check-mask, the precomputation and the golden id lists
(decision 3).

    uv run --group wan python -m wan.sources --fetch     # fetch (into the default HF cache)
    uv run --group wan python -m wan.sources             # check what was fetched (parameter counts)
    uv run --group wan python -m wan.sources --model ti2v-5b --fetch   # TI2V-5B (~22.8 GB)
    uv run --group wan python -m wan.sources --umt5 xxl --fetch        # original umT5 (~29.8 GB)

MUST: `huggingface_hub` is imported inside the functions (the `wan` group is not in the default
sync — `tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream HF repository (repo + pinned revision + license identifier)."""

    repo: str
    #: HF commit SHA (40 lowercase hex digits). No branch names or tags.
    revision: str
    #: `license:` in the front matter of the upstream `README.md` (checked with the HF API on
    #: 2026-10-02).
    license: str


@dataclass(frozen=True)
class Component:
    """One component of an upstream repository (`subfolder`) and whether it is fetched."""

    subfolder: str
    fetch: bool
    #: Why it is or is not fetched (for readers of the scripts and the reports).
    why: str
    #: Whether this is a component left out of the DiT / VAE fetch (`fetch=False`) and fetched by
    #: the separate text-embedding process with {@link text_snapshot} (Wan2.1's umT5 and tokenizer).
    text: bool = False


#: The table of upstream sources. Keys are the model names of the distribution (model `t2v-1.3b`
#: of `models/karume-wan2.1/` — ADR 0118 decision 7; model `ti2v-5b` of `models/karume-wan2.2/` —
#: ADR 0121 decisions 1 / 10).
SOURCES: dict[str, UpstreamSource] = {
    "t2v-1.3b": UpstreamSource(
        repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        revision="0fad780a534b6463e45facd96134c9f345acfa5b",
        license="apache-2.0",
    ),
    "ti2v-5b": UpstreamSource(
        repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        revision="b8fff7315c768468a5333511427288870b2e9635",
        license="apache-2.0",
    ),
}

#: The default model (the default of the Wan2.1 scripts — {@link WAN21_MODELS}).
DEFAULT_MODEL = "t2v-1.3b"

#: The models that the Wan2.1 scripts (series location, case table and text step fixed to Wan2.1)
#: accept with `--model`. Using `sorted(SOURCES)` as the choices would open a way to write the 5B
#: into the Wan2.1 series location when `ti2v-5b` is passed (closed at step 0 of ADR 0121, when
#: `ti2v-5b` was added to the table).
WAN21_MODELS: tuple[str, ...] = ("t2v-1.3b",)

#: The models that the Wan2.2 distribution (`karume-wan2.2`) distributes — one set shared by the
#: plan check of the distribution recipe (`wan.distribution.WAN22`) and the attribution check of
#: the card (`wan.card.WAN22_CARD`). As with {@link WAN21_MODELS}, being in the table
#: ({@link SOURCES}) does not let a model through (the table has rows for both generations).
WAN22_MODELS: tuple[str, ...] = ("ti2v-5b",)

#: The model the two text assets of the Wan2.2 distribution (embeddings, tokenizer) come from = the
#: Wan2.1 checkpoint (ADR 0121 decision 9 — the same umT5 and the same tokenizer, so the very files
#: of the 2.1 series are distributed). Shared by the provenance check of the distribution recipe
#: (`text_model` of `wan.distribution.WAN22`) and the text of the card.
WAN22_TEXT_MODEL = "t2v-1.3b"

#: The components of the upstream repository (model name → components). A component with
#: `fetch=False` is not downloaded by the DiT / VAE fetch.
COMPONENTS: dict[str, tuple[Component, ...]] = {
    "t2v-1.3b": (
        Component(
            "transformer",
            fetch=True,
            why="DiT (steps 2 / 3) — fp32 safetensors in 2 shards, about 5.7 GB",
        ),
        Component("vae", fetch=True, why="video VAE (steps 4 / 5) — fp32, about 0.5 GB"),
        Component("scheduler", fetch=True, why="UniPC config (shift 3.0 — decision 5)"),
        Component(
            "text_encoder",
            fetch=False,
            text=True,
            why="umT5-XXL f32, about 22.7 GB — fetched by the separate process of step 6",
        ),
        Component(
            "tokenizer",
            fetch=False,
            text=True,
            why="umT5 tokenizer — fetched by the separate process of step 6",
        ),
    ),
    "ti2v-5b": (
        Component(
            "transformer",
            fetch=True,
            why="DiT — fp32 safetensors in 5 shards, about 19.99 GB (ADR 0121 decision 1)",
        ),
        Component(
            "vae",
            fetch=True,
            why="Wan2.2-VAE — fp32, about 2.82 GB (encoder included, ADR 0121 decision 1)",
        ),
        Component(
            "scheduler",
            fetch=True,
            why="UniPC config (flow_shift is written into the golden metadata"
            " — ADR 0121 decisions 1 / 8)",
        ),
        Component(
            "text_encoder",
            fetch=False,
            why="umT5 is a cross-repository reference to karume-umt5-xxl — not fetched"
            " (ADR 0121 decision 9)",
        ),
        Component(
            "tokenizer",
            fetch=False,
            why="the tokenizer assets are copied from the Wan2.1 series — not fetched"
            " (ADR 0121 decision 9)",
        ),
    ),
}

#: Files directly under the repository root fetched outside the components (the license front
#: matter and the component index).
ROOT_FILES: tuple[str, ...] = ("README.md", "model_index.json")


@dataclass(frozen=True)
class Umt5Source:
    """One umT5 upstream (the pinned HF repository, and the sha256 of the pickle shards the encoder
    needs and of the tokenizer files — ADR 0122 decisions 1 / 2). The encoder and the tokenizer are
    directly under the repository root (no subfolder)."""

    source: UpstreamSource
    #: The shards the encoder needs (file name → the LFS sha256 from the HF API, 64 lowercase hex
    #: digits). The fetch and read entry points check that the set of shards derived from the index
    #: equals this table (an upstream index that moved or a stale table is not silently let
    #: through). The read entry point checks the sha256 of each shard before unpickling — reading a
    #: pickle can run arbitrary code, so the first line of defence is the content pin (the second is
    #: `weights_only=True`).
    shards: Mapping[str, str]
    #: The files the tokenizer source of truth (`AutoTokenizer` of transformers 5.14.1) reads (file
    #: name → sha256). `tokenizer.json` and `tokenizer_config.json` alone give the same backend as
    #: reading the whole repository (`special_tokens_map.json` and `spiece.model` do not change the
    #: backend — measured on 2026-10-15, see the docstring of `wan.umt5_xxl_tokenizer`). The fetch
    #: entry point ({@link pinned_umt5_tokenizer}) copies only the verified bytes to the
    #: destination.
    tokenizer: Mapping[str, str]


#: The umT5 upstream table (keys are the model names of the umT5 distribution —
#: `wan.umt5_distribution.UMT5_DEFAULT_MODEL`). The commit is on main of the original repository
#: (no tags); the license comes from `cardData` of the API and the README front matter; the shard
#: sha256 values are the LFS values of `/api/models/google/umt5-xxl/revision/<commit>?blobs=true`
#: (checked on 2026-10-04 — research `2026-10-04-umt5-upstream-provenance` §1). For the tokenizer,
#: `tokenizer.json` uses the LFS value of the same API, and `tokenizer_config.json`, which is not in
#: LFS, uses the sha256 of the local bytes (checked on 2026-10-15 that its git blob id
#: `c1a69656…` equals `blobId` of the same API).
UMT5_SOURCES: dict[str, Umt5Source] = {
    "xxl": Umt5Source(
        source=UpstreamSource(
            repo="google/umt5-xxl",
            revision="66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
            license="apache-2.0",
        ),
        shards={
            "pytorch_model-00001-of-00006.bin": (
                "382094214dfe74d782769f61ad95cfe32fdd297ae51f16f9208afa180b355e61"
            ),
            "pytorch_model-00002-of-00006.bin": (
                "b49efce006c907ea93eb38658577d5c8d4e85b4bab398a0c6ba25141927529d4"
            ),
            "pytorch_model-00003-of-00006.bin": (
                "da3d39fffe6464247531c20696715860ccaabdaaad3d5a2979dc2e2fbb7789fc"
            ),
        },
        tokenizer={
            "tokenizer.json": "af904105ce1071b1202bba0059a841f4a7b85b48b6ec179c4948e3483476e0dd",
            "tokenizer_config.json": (
                "616015e2b6c26f9a98ae3171634518d440a46260ebc1115c2596792d33827bdc"
            ),
        },
    ),
}

#: The index and the config of the original sharded pickle form (spelled as transformers'
#: `save_pretrained` writes them).
UMT5_BIN_INDEX = "pytorch_model.bin.index.json"
UMT5_CONFIG = "config.json"

#: Key prefixes of the weights the encoder uses (the vocabulary embedding `shared.weight` and
#: `encoder.*` — the decoder and `lm_head` are not fetched).
UMT5_ENCODER_PREFIXES: tuple[str, ...] = ("shared.", "encoder.")

#: The parameter counts a fetched checkpoint must have (model name → component name → sum of the
#: element counts in the safetensors headers). The VAE includes the encoder.
#: - `t2v-1.3b`: the values from the survey (2026-10-02), also confirmed on the files of the pinned
#:   revision.
#: - `ti2v-5b`: the values of ADR 0121 decision 1 (summed from the safetensors headers of the
#:   pinned revision — the vae is decoder + `post_quant_conv` 555,051,580 and encoder +
#:   `quant_conv` 149,637,088).
EXPECTED_PARAMETERS: dict[str, dict[str, int]] = {
    "t2v-1.3b": {
        "transformer": 1_418_996_800,
        "vae": 126_892_531,
    },
    "ti2v-5b": {
        "transformer": 4_999_787_712,
        "vae": 704_688_668,
    },
}

#: safetensors header dtype → bytes per element (for the header consistency check).
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1}


class WanSourceError(RuntimeError):
    """The upstream checkpoint is not available locally in its pinned form, or a header cannot be
    read."""


def _components(model: str) -> tuple[Component, ...]:
    """The component table of a model (a model not in the table fails loudly — another model's
    table is never used silently)."""
    parts = COMPONENTS.get(model)
    if parts is None:
        raise WanSourceError(
            f"no component table for model {model!r} (known: {sorted(COMPONENTS)})"
        )
    return parts


def text_components(model: str = DEFAULT_MODEL) -> tuple[str, ...]:
    """Components fetched only by the separate text-embedding process (`wan.text_embeds`) (empty
    for a model without any)."""
    return tuple(part.subfolder for part in _components(model) if part.text)


def allow_patterns(model: str = DEFAULT_MODEL) -> list[str]:
    """The fetch targets passed to `snapshot_download` (under each fetched component + the two
    files at the repository root)."""
    return [*ROOT_FILES, *(f"{part.subfolder}/*" for part in _components(model) if part.fetch)]


def fetch(model: str = DEFAULT_MODEL) -> Path:
    """Download the fetch targets of the pinned revision into the default HF cache and return the
    snapshot directory."""
    from huggingface_hub import snapshot_download

    source = SOURCES[model]
    return Path(
        snapshot_download(
            source.repo, revision=source.revision, allow_patterns=allow_patterns(model)
        )
    )


def local_snapshot(model: str = DEFAULT_MODEL) -> Path:
    """The directory of an already fetched snapshot (no network access).

    Fails loudly when it is absent — if a script silently started fetching, a 5.7 GB download would
    slip into the middle of building a reference.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = SOURCES[model]
    try:
        snapshot = Path(
            snapshot_download(
                source.repo,
                revision=source.revision,
                allow_patterns=allow_patterns(model),
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError as error:
        raise WanSourceError(
            f"{source.repo}@{source.revision} is not in the HF cache — fetch it first with"
            f" `uv run --group wan python -m wan.sources --model {model} --fetch`"
        ) from error
    missing = [part.subfolder for part in _components(model) if part.fetch]
    missing = [name for name in missing if not (snapshot / name).is_dir()]
    if missing:
        raise WanSourceError(
            f"{snapshot} is missing components {missing} — fetch again with `--fetch`"
        )
    return snapshot


def text_snapshot(model: str = DEFAULT_MODEL, *, fetch: bool = False) -> Path:
    """The snapshot of umT5 and the tokenizer ({@link text_components}) at the pinned revision.

    With `fetch=False` there is no network access, and an absent snapshot fails loudly (the same
    reason as {@link local_snapshot} — a download of about 23 GB does not slip into generating the
    embeddings). It is a separate entry point from the DiT / VAE fetch targets, so that umT5 never
    mixes into the fetch of the reference pipeline.
    MUST: a model that does not fetch the text-step components from upstream (`ti2v-5b` — ADR 0121
    decision 9) fails loudly.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = SOURCES[model]
    names = text_components(model)
    if not names:
        raise WanSourceError(
            f"model {model!r} does not fetch the text-step components (umT5, tokenizer)"
            " from upstream (ADR 0121 decision 9 — umT5 is a cross-repository reference and the"
            " tokenizer assets are copied from the Wan2.1 series)"
        )
    patterns = [f"{name}/*" for name in names]
    try:
        snapshot = Path(
            snapshot_download(
                source.repo,
                revision=source.revision,
                allow_patterns=patterns,
                local_files_only=not fetch,
            )
        )
    except LocalEntryNotFoundError as error:
        raise WanSourceError(
            f"{source.repo}@{source.revision}: {list(names)} not in the HF cache —"
            " fetch them first with"
            " `uv run --group wan --inexact python -m wan.text_embeds --fetch`"
        ) from error
    missing = [name for name in names if not (snapshot / name).is_dir()]
    if missing:
        raise WanSourceError(
            f"{snapshot} is missing components {missing} — fetch again with `--fetch`"
        )
    return snapshot


def umt5_encoder_shards(weight_map: Mapping[str, str]) -> frozenset[str]:
    """Derive from the `weight_map` of the index (key → shard file name) the shards that hold the
    encoder weights ({@link UMT5_ENCODER_PREFIXES})."""
    return frozenset(
        shard for key, shard in weight_map.items() if key.startswith(UMT5_ENCODER_PREFIXES)
    )


def pinned_umt5_shards(index: Path, pinned: Mapping[str, str]) -> dict[str, str]:
    """Check that the set of shards derived from the index equals the table (`pinned`), and return
    the keys of the index's `weight_map` that live in the table's shards (key → shard file name).

    MUST: if the sets differ, fail loudly — the upstream index moved or the table is stale. Either
    way, a shard the table's sha256 does not cover would be read, or a needed shard would be missed.
    """
    weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    derived = umt5_encoder_shards(weight_map)
    if derived != frozenset(pinned):
        raise WanSourceError(
            f"{index}: encoder shards derived from the index {sorted(derived)} differ from the"
            f" table's shards {sorted(pinned)} — the upstream index moved or the table"
            " (wan.sources.UMT5_SOURCES) is stale"
        )
    return {key: shard for key, shard in weight_map.items() if shard in pinned}


def umt5_snapshot(model: str, *, fetch: bool = False) -> Path:
    """The snapshot of the umT5 upstream pin ({@link UMT5_SOURCES}) (config, index, encoder shards).

    Fetches the config and the index first, then the shards derived from the index
    ({@link pinned_umt5_shards} — fails if they differ from the table). Shards with only decoder
    weights are not fetched (of the original's 6 shards, about 51.9 GB, the encoder needs about
    29.8 GB). With `fetch=False` there is no network access, and an absent snapshot fails loudly
    (the same reason as {@link local_snapshot}). The sha256 of the shards is checked by the read
    entry point (`wan.umt5_export.Checkpoint`) before unpickling — the check closes on the files
    actually read, not on a record of the fetch.
    """
    row = _umt5_row(model)
    hint = f"uv run --group wan python -m wan.sources --umt5 {model} --fetch"
    snapshot = _umt5_download(row, [UMT5_CONFIG, UMT5_BIN_INDEX], fetch=fetch, hint=hint)
    index = snapshot / UMT5_BIN_INDEX
    if not index.is_file() or not (snapshot / UMT5_CONFIG).is_file():
        raise WanSourceError(f"{snapshot} has no {UMT5_CONFIG} / {UMT5_BIN_INDEX} — `--fetch`")
    shards = sorted(set(pinned_umt5_shards(index, row.shards).values()))
    snapshot = _umt5_download(row, shards, fetch=fetch, hint=hint)
    missing = [name for name in shards if not (snapshot / name).is_file()]
    if missing:
        raise WanSourceError(f"{snapshot} is missing shards {missing} — fetch again with `--fetch`")
    return snapshot


def pinned_umt5_tokenizer(model: str, destination: Path, *, fetch: bool = False) -> None:
    """Fetch the tokenizer files of the umT5 upstream pin ({@link Umt5Source.tokenizer}) and copy
    only the sha256-verified bytes to `destination` (the source of truth reads from there).

    With `fetch=False` there is no network access, and absent files fail loudly (the same reason as
    {@link local_snapshot}).

    MUST: the check runs on the very bytes that are copied (read, verify, then write — even if the
    file in the HF cache is swapped after the check, the copy keeps the verified bytes). Files not
    in the table are not copied — the snapshot directory also holds `config.json` and other files
    downloaded for the encoder, so reading directly from there would make the source of truth depend
    on files that are not pinned.
    """
    row = _umt5_row(model)
    names = sorted(row.tokenizer)
    hint = f"uv run --group wan --inexact python -m wan.umt5_xxl_tokenizer --model {model} --fetch"
    snapshot = _umt5_download(row, names, fetch=fetch, hint=hint)
    verified: dict[str, bytes] = {}
    for name in names:
        path = snapshot / name
        if not path.is_file():
            raise WanSourceError(f"{snapshot} has no {name} — fetch again with `--fetch`")
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != row.tokenizer[name]:
            raise WanSourceError(
                f"{path}: sha256 {actual} differs from {row.tokenizer[name]} in the table — no"
                " tokenizer other than the upstream pin is read (the table wan.sources.UMT5_SOURCES"
                " or the HF cache is broken)"
            )
        verified[name] = data
    # Copy only after verifying all of them (if even one differs, none is copied).
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in verified.items():
        (destination / name).write_bytes(data)


def _umt5_row(model: str) -> Umt5Source:
    """The row of the umT5 upstream table (a model not in the table fails loudly)."""
    row = UMT5_SOURCES.get(model)
    if row is None:
        raise WanSourceError(
            f"umT5 model {model!r} is not in the upstream table (known: {sorted(UMT5_SOURCES)})"
        )
    return row


def _umt5_download(row: Umt5Source, patterns: list[str], *, fetch: bool, hint: str) -> Path:
    """Fetch only `patterns` from the umT5 upstream pin (with `fetch=False`, only the cache is
    consulted)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = row.source
    try:
        return Path(
            snapshot_download(
                source.repo,
                revision=source.revision,
                allow_patterns=patterns,
                local_files_only=not fetch,
            )
        )
    except LocalEntryNotFoundError as error:
        raise WanSourceError(
            f"{source.repo}@{source.revision}: {patterns} not in the HF cache — fetch them first"
            f" with `{hint}`"
        ) from error


def read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    """Read only the safetensors header (the 8-byte length at the start + JSON; the body is not
    read)."""
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    return header


def count_parameters(component_dir: Path) -> int:
    """The sum of the element counts of all tensors in the safetensors files of a component
    directory.

    A row whose `data_offsets` width disagrees with `dtype × shape` fails loudly (rules out a
    misread header that happens to give the right count).
    """
    files = sorted(component_dir.glob("*.safetensors"))
    if not files:
        raise WanSourceError(f"{component_dir} has no safetensors")
    total = 0
    for file in files:
        for name, entry in read_safetensors_header(file).items():
            shape = entry["shape"]
            dtype = entry["dtype"]
            offsets = entry["data_offsets"]
            assert isinstance(shape, list) and isinstance(dtype, str) and isinstance(offsets, list)
            count = 1
            for size in shape:
                count *= int(size)
            if count * _DTYPE_BYTES[dtype] != int(offsets[1]) - int(offsets[0]):
                raise WanSourceError(
                    f"{file}: {name}: dtype {dtype} × {shape} does not match the width"
                )
            total += count
    return total


def check_snapshot(snapshot: Path, model: str = DEFAULT_MODEL) -> dict[str, int]:
    """Count the parameters of the fetched components and match them against the model's row of
    {@link EXPECTED_PARAMETERS}."""
    expected_counts = EXPECTED_PARAMETERS.get(model)
    if expected_counts is None:
        raise WanSourceError(
            f"no expected parameter counts for model {model!r}"
            f" (known: {sorted(EXPECTED_PARAMETERS)})"
        )
    counts = {name: count_parameters(snapshot / name) for name in expected_counts}
    wrong = {
        name: (counts[name], expected)
        for name, expected in expected_counts.items()
        if counts[name] != expected
    }
    if wrong:
        raise WanSourceError(
            f"{model}: parameter counts differ from the survey (actual, expected): {wrong}"
        )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument("--fetch", action="store_true", help="fetch at the pinned revision")
    parser.add_argument(
        "--umt5",
        choices=sorted(UMT5_SOURCES),
        help="fetch / check the umT5 upstream (the encoder shards of the original) instead of Wan",
    )
    args = parser.parse_args(argv)

    if args.umt5 is not None:
        snapshot = umt5_snapshot(args.umt5, fetch=args.fetch)
        print(f"snapshot: {snapshot}")
        for name in sorted(UMT5_SOURCES[args.umt5].shards):
            print(f"{name}: {(snapshot / name).stat().st_size:,} bytes")
        return 0
    snapshot = fetch(args.model) if args.fetch else local_snapshot(args.model)
    counts = check_snapshot(snapshot, args.model)
    print(f"snapshot: {snapshot}")
    for name, count in counts.items():
        print(f"{name}: {count:,} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
