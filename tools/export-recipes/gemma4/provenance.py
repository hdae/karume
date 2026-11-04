"""Provenance record (`reference.json`) of **a series that takes no goldens of its own** — the
writer side's source of truth.

The token-only series ({@link gemma4.export_token}) does not take its greedy expected sequences
itself; it reuses `greedy.<case>.safetensors` of the logits opt-in series
({@link gemma4.export_decode}) in the acceptance gate
(`packages/models/tests/e2e_gemma4_token_exit_test.ts`). This form is cheap, but the one premise
that **the two series come from the same checkpoint** was not machine-readable (full review
CX-2.3) — there was only a check that the asset directories exist, so the gate stays green even
when only one of them is stale.

So the token-only export writes this record next to the assets it places:

- `checkpoint` — **the fingerprint of the source checkpoint** (sha256 and byte count of
  {@link FINGERPRINT_FILES}). It names which weights, config and tokenizer produced this container.
- `reference` — **the identity of the reused golden series** (the series directory name, and the
  sha256 and byte count of each referenced `greedy.<case>.safetensors`).

Binding semantics: "this container comes from checkpoint X and is to be accepted against **exactly
the goldens with the digests written here**". Retaking the goldens (= rebuilding the logits series
from another checkpoint or with another rounding) moves the digests, and the gate on the TS side
**fails loudly** — that forces a re-export, which updates `checkpoint` along with it. Conversely,
when only the token-only side is rebuilt, the record is written in the same swap as the container,
so a record older than its container cannot arise.

MUST: **do not require a re-export of the logits series** (retaking the greedy expected sequences
takes tens of minutes per run). This module reads only the bytes of the existing goldens, and
neither adds to them nor overwrites them.

MUST: this record is a **claim** of "from the same checkpoint", not a proof. The numeric identity
is checked by the cross-series parity on the TS side (an exact match of 3 cases × 16 steps) — this
module's part is to pin, machine-readably, "which combination of assets parity should be checked
on". At export time {@link assert_reference_goldens} also compares each golden's `prompt` with the
cases built this run, so a combination whose tokenizer or case definitions disagree fails before
the record is written.

12B's product series (`gemma4.export_unified`) writes its own goldens next to its container, and
its record uses the same schema with `reference` naming that series itself. Its checkpoint is two
pinned hub snapshots — the weights and their config from one repository, the tokenizer from
another (`gemma4.sources`) — so the fingerprint reads each file from its own source, adds
`sources` (role → repository, revision and the files read from it) and is also checked against
the pins ({@link assert_pinned_fingerprint}).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from _shared.decode_series import GREEDY_PREFIX, GREEDY_SUFFIX, PROMPT_KEY
from gemma4.sources import ModelPins

#: File name of the record (directly under the series directory; the acceptance gate on the TS side
#: reads the same spelling).
REFERENCE_FILE = "reference.json"

#: Version of the record (a field so that a reader does not silently read a version it does not
#: know).
SCHEMA = 1

#: The fingerprinted files read from the weights' source and from the tokenizer's — one directory
#: for a hand-placed checkpoint, two pinned snapshots for 12B (`gemma4.sources.ModelPins`).
WEIGHTS_FILES = ("model.safetensors", "config.json")
TOKENIZER_FILES = ("tokenizer.json",)

#: The checkpoint files to fingerprint. Not only the weights but also the config and the tokenizer —
#: the same weights with a different layer layout in `config.json` or a different tokenizer are a
#: different asset.
FINGERPRINT_FILES = (*WEIGHTS_FILES, *TOKENIZER_FILES)

#: The fingerprinted files of each pinned source (`gemma4.sources.ModelPins.by_role`).
FILES_BY_ROLE: dict[str, tuple[str, ...]] = {"weights": WEIGHTS_FILES, "tokenizer": TOKENIZER_FILES}

#: Read size when taking a sha256 (a 10GB checkpoint is never loaded into memory whole).
DIGEST_BLOCK = 8 << 20


def file_digest(path: Path) -> dict[str, Any]:
    """`{bytes, sha256}` of `path` (read in blocks — the data is never held in memory)."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(DIGEST_BLOCK)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def checkpoint_fingerprint(
    model_dir: Path,
    label: str | None = None,
    *,
    tokenizer_dir: Path | None = None,
    pins: ModelPins | None = None,
) -> dict[str, Any]:
    """Fingerprint of the source checkpoint (each file of {@link FINGERPRINT_FILES}).

    `dir` names the checkpoint: `label` when given — the repository ids of hub snapshots, whose
    directory names are only the commit SHAs — else the directory name (a hand-placed checkpoint
    under `inputs/`, named after its repository).

    {@link WEIGHTS_FILES} are read from `model_dir` and {@link TOKENIZER_FILES} from
    `tokenizer_dir` (`None`: `model_dir` — one checkpoint holds both). `pins` — the sources of a
    pinned model — adds `sources`: role → `{repo, revision, files}`, naming which source each
    fingerprinted file was read from.
    """
    tokenizer_root = model_dir if tokenizer_dir is None else tokenizer_dir
    files: dict[str, Any] = {}
    for name in FINGERPRINT_FILES:
        directory = model_dir if name in WEIGHTS_FILES else tokenizer_root
        path = directory / name
        if not path.is_file():
            raise AssertionError(
                f"the checkpoint lacks the fingerprinted file '{name}': {directory}"
            )
        files[name] = file_digest(path)
    fingerprint: dict[str, Any] = {
        "dir": model_dir.name if label is None else label,
        "files": files,
    }
    if pins is not None:
        fingerprint["sources"] = {
            role: {"repo": pin.repo, "revision": pin.revision, "files": list(FILES_BY_ROLE[role])}
            for role, pin in pins.by_role().items()
        }
    return fingerprint


def assert_pinned_fingerprint(fingerprint: Mapping[str, Any], pins: ModelPins) -> None:
    """The fingerprint {@link checkpoint_fingerprint} took (every byte hashed) names the pinned
    files exactly (`gemma4.sources.HUB_PINS`), each source's against its own pin.

    The cheap pre-check (`gemma4.sources.assert_model_pins`) trusts the cache's blob names; this
    one runs on the bytes the export actually read, right before the record that names them is
    written. MUST: each pin covers exactly the files the fingerprint reads from its source
    ({@link FILES_BY_ROLE}) — a file the record names but no pin does would go unchecked, and a
    file pinned in the other source's role would be checked against bytes of another snapshot.
    """
    for role, pin in pins.by_role().items():
        if sorted(pin.sha256) != sorted(FILES_BY_ROLE[role]):
            raise AssertionError(
                f"the {role} pin ({pin.label}) names {sorted(pin.sha256)}, the fingerprint reads"
                f" {sorted(FILES_BY_ROLE[role])} from that source"
            )
    files = fingerprint["files"]
    wrong = {
        name: (files[name]["sha256"], expected, pin.label)
        for pin in pins.by_role().values()
        for name, expected in sorted(pin.sha256.items())
        if files[name]["sha256"] != expected
    }
    if wrong:
        raise AssertionError(
            f"the checkpoint read is not the pinned one (actual, pin, source): {wrong}"
        )


def _golden_prompt(path: Path) -> torch.Tensor:
    """Read the golden's `prompt` (1-D i32)."""
    with safe_open(str(path), framework="pt") as handle:
        stored = handle.keys()
        if PROMPT_KEY not in stored:
            raise AssertionError(f"golden {path.name} has no '{PROMPT_KEY}' ({sorted(stored)})")
        return handle.get_tensor(PROMPT_KEY)


def assert_reference_goldens(
    reference_dir: Path, cases: Sequence[tuple[str, torch.Tensor]]
) -> dict[str, Any]:
    """Check each reused golden and return `{file name: {bytes, sha256}}`.

    MUST: check that `prompt` is **identical to the case built this run** (a combination where the
    tokenizer, the BOS handling or the case text disagrees fails before a digest is written). The
    golden's `expected` is not read — comparing with the expected sequence is the job of the
    acceptance gate on a real GPU, and redoing part of it in torch here would leave room for "two
    things wrong in the same direction".
    """
    if not reference_dir.is_dir():
        raise AssertionError(
            f"reference golden series {reference_dir} is missing"
            " — write the logits opt-in series first (python -m gemma4.export_decode)"
        )
    goldens: dict[str, Any] = {}
    for name, ids in cases:
        path = reference_dir / f"{GREEDY_PREFIX}{name}{GREEDY_SUFFIX}"
        if not path.is_file():
            raise AssertionError(
                f"reference golden {path} is missing (the reused series lacks {name})"
            )
        prompt = _golden_prompt(path).to(torch.int64)
        if not torch.equal(prompt, ids[0].to(torch.int64)):
            raise AssertionError(
                f"reference golden {path.name}: its prompt differs from this run's case '{name}'"
                f" (golden {tuple(prompt.shape)} / this run {tuple(ids[0].shape)})"
                " — trying to reuse a golden taken with another tokenizer or case text"
            )
        goldens[path.name] = file_digest(path)
    return {"series": reference_dir.name, "goldens": goldens}


def build_record(
    series_dir: Path,
    model_dir: Path,
    reference: Mapping[str, Any],
    *,
    checkpoint_label: str | None = None,
    tokenizer_dir: Path | None = None,
    pins: ModelPins | None = None,
) -> dict[str, Any]:
    """Contents of `reference.json` (binds the result of {@link assert_reference_goldens} to the
    fingerprint).

    `checkpoint_label` is the fingerprint's `dir`; `tokenizer_dir` and `pins` are passed on as they
    are ({@link checkpoint_fingerprint}).
    """
    return {
        "schema": SCHEMA,
        "series": series_dir.name,
        "checkpoint": checkpoint_fingerprint(
            model_dir, checkpoint_label, tokenizer_dir=tokenizer_dir, pins=pins
        ),
        "reference": dict(reference),
    }


def write_record(series_dir: Path, record: Mapping[str, Any]) -> None:
    """Write the record into the series directory.

    MUST: the caller writes it into **the same swap unit as the container** (the staging directory
    of `staged_publication`) — placing them separately allows "new container + old record" to exist
    even for a moment, and the binding loses its meaning.
    """
    path = series_dir / REFERENCE_FILE
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
