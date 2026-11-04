"""Script that converts the DACVAE PyTorch checkpoint (`weights.pth`) to safetensors **1:1**.

    uv run python -m irodori.dacvae.convert

A one-off utility that only repacks the real weights placed by hand in `inputs/<family>/<name>/`
(not a generated artifact — docs/assets-layout.md) into a form usable as input material for the IR
export, **without changing their contents**. The default input and output are
`inputs/irodori/dacvae-32dim/weights.pth` →
`inputs/irodori/dacvae-32dim/weights.safetensors` (+ `metadata.json`).

## What 1:1 means (what this script does not do)

**Key names, dtypes and value bytes stay unchanged.** Concretely, none of the following is done:

- Baking in `weight_norm` (this checkpoint holds the old-style `weight_g` / `weight_v` pairs).
  Composing `w = g · v/‖v‖` is **the job of the export stage, which depends on the model's
  shape**; baking it here would freeze the "input material" into one interpretation already.
- Normalizing key names, re-prefixing, or folding Snake's `.alpha` and the like.
- Converting dtypes (f16 / i8 storage is a matter of the distribution — handled by `karume.emit`
  under ADR 0018 / 0019).

## Handling of metadata

The outside of the `.pth` is a wrapper dict `{state_dict, metadata}`, and `metadata` is plain
configuration values without tensors (`kwargs`). safetensors' `__metadata__` **can only hold
strings**, so the structured source of truth is written to `metadata.json` in the same directory,
and `__metadata__` carries the origin (original file name, sha256, configuration values) folded into
a canonical JSON under **a single key** (so the provenance can be traced even when the safetensors
file is taken out on its own). Both are written **from the same source in this one pass**; there
is no path that updates only one of them.

MUST: `__metadata__` has exactly one key ({@link SOURCE_KEY}). `safetensors` holds `__metadata__`
in a Rust-side `HashMap` and **neither preserves nor sorts its order** — measured: the same input
produced three different orders (back when there were three keys). With two or more keys the
header bytes move from run to run, and "the same input gives the same bytes" below does not hold
(the data section is identical but only the file's sha256 moves, which reads as a diff saying "the
asset changed").

## Write path

Written plainly with `safetensors.torch.save_file`. This is not the distribution but **an input
placed next to a hand-placed asset** (`inputs/irodori/dacvae-32dim/weights.safetensors`), a
different matter from the distribution's container (`krm`).
Karume's reader requires the data section to be covered "without gaps and aligned to the element
size" (docs/limitations.md), so right after writing, the same rule is checked with
`verify.assert_reader_layout`. This checkpoint is all F32, so the alignment constraint holds
trivially, but the check runs unconditionally.
The tensor order is fixed to ascending key order, and `__metadata__` is one key whose value is a
canonical JSON with ascending keys too (the same input gives **the whole file** as the same bytes
— regeneration produces no diff). The check is
`dacvae/tests/test_convert.py::TestConvert::test_two_runs_write_the_same_bytes`.

Writes go to **a staging area** and are moved to the canonical path only after every check passes
(ADR 0052 — the same discipline as the other writers). The output directory is shared with
hand-placed assets, so the move uses the **per-file** form of
{@link karume.artifacts.staged_publication} twice (`weights.safetensors` / `metadata.json`) —
moving a whole directory would also replace the neighbouring `weights.pth`.

## Self-verification

Converting the real weights is manual (a one-off utility), but the parts that do not depend on
assets are pinned by `dacvae/tests/test_convert.py`. In a real run, right after the conversion
**every tensor is read back from the safetensors side and checked byte for byte**
(`safetensors.safe_open` — a reader implementation separate from the one that wrote it), and the
match count and the output's sha256 go into the summary. If this fails, the whole staging area is
discarded and the canonical path does not move by a single byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from _shared.paths import INPUTS_ROOT
from karume.artifacts import staged_publication
from karume.verify import assert_reader_layout

#: Default input (the hand-placed real weights — `inputs/<family>/<name>/`).
DEFAULT_CKPT = INPUTS_ROOT / "irodori" / "dacvae-32dim" / "weights.pth"

#: Keys expected in the outer wrapper of the `.pth`. Any addition or removal is a change on the
#: model publisher's side, so it fails loudly.
WRAPPER_KEYS = ("state_dict", "metadata")

#: The **only** key of `__metadata__` (its value is one canonical JSON string — the MUST of the
#: module doc).
SOURCE_KEY = "karume_source"

#: Keys inside that canonical JSON (by the safetensors spec, `__metadata__` values are strings
#: only, so the structure is held here).
SOURCE_FILE_KEY = "source_file"
SOURCE_SHA256_KEY = "source_sha256"
SOURCE_METADATA_KEY = "source_metadata"

_SHA_CHUNK = 1 << 22


class ConvertError(ValueError):
    """The checkpoint's shape differs from what is assumed / the conversion result is not
    byte-identical to the original."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_SHA_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], object]:
    """Load with `weights_only=True`, check the outer shape and the state_dict contents, and
    return them."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ConvertError(f"{path}: the top level is not a dict ({type(obj).__name__})")
    if set(obj) != set(WRAPPER_KEYS):
        raise ConvertError(
            f"{path}: the outer wrapper's keys are not the expected ones"
            f" (expected {list(WRAPPER_KEYS)} / actual {list(obj)})"
        )
    state_dict = obj["state_dict"]
    if not isinstance(state_dict, dict):
        raise ConvertError(f"{path}: state_dict is not a dict ({type(state_dict).__name__})")
    non_tensor = [key for key, value in state_dict.items() if not torch.is_tensor(value)]
    if non_tensor:
        raise ConvertError(f"{path}: state_dict holds non-tensor values: {sorted(non_tensor)}")
    return state_dict, obj["metadata"]


def _metadata_json(metadata: object) -> str:
    """Turn the non-tensor metadata into a JSON string (fails loudly unless it round-trips to an
    equal value).

    There is no `default=` escape hatch — a type that fails serialization means the premise of
    "configuration values that fit into JSON" is broken, and silently writing a `repr` would
    misstate the provenance.
    """
    try:
        text = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    except TypeError as cause:
        raise ConvertError(f"metadata cannot be serialized to JSON: {cause}") from cause
    if json.loads(text) != metadata:
        raise ConvertError(
            "metadata does not survive a JSON round trip unchanged (a tuple / non-string key)"
        )
    return text


def _source_document(ckpt: Path, source_sha256: str, metadata: object) -> str:
    """The only value of `__metadata__` (the three origin fields folded into a canonical JSON with
    ascending keys).

    Before folding, the value goes through the round-trip check of {@link _metadata_json}, and
    **the value that survived that round trip** is embedded as is (the checked value and the
    written value cannot differ). `sort_keys` is one half of determinism; the other half (keeping
    `__metadata__` to one key) is the MUST of the module doc.
    """
    document = {
        SOURCE_FILE_KEY: ckpt.name,
        SOURCE_SHA256_KEY: source_sha256,
        SOURCE_METADATA_KEY: json.loads(_metadata_json(metadata)),
    }
    return json.dumps(document, ensure_ascii=False, sort_keys=True)


def _assert_byte_identical(path: Path, tensors: Mapping[str, torch.Tensor]) -> int:
    """Read the written safetensors back with **a separate reader implementation** and check that
    every tensor is byte-identical.

    Compares the raw bytes, not just dtype and shape (to claim "the values did not change" down to
    NaN bit patterns and -0.0). Returns the number of tensors that matched.
    """
    matched = 0
    with safe_open(str(path), framework="pt") as handle:
        keys = set(handle.keys())
        expected = set(tensors)
        if keys != expected:
            raise ConvertError(
                f"{path}: key sets differ (missing {sorted(expected - keys)} / "
                f"extra {sorted(keys - expected)})"
            )
        for name in sorted(expected):
            source = tensors[name]
            restored = handle.get_tensor(name)
            if restored.dtype != source.dtype:
                raise ConvertError(
                    f"tensor '{name}': dtype mismatch (original {source.dtype} / read back "
                    f"{restored.dtype})"
                )
            if tuple(restored.shape) != tuple(source.shape):
                raise ConvertError(
                    f"tensor '{name}': shape mismatch (original {tuple(source.shape)} / read back "
                    f"{tuple(restored.shape)})"
                )
            if restored.numpy().tobytes() != source.numpy().tobytes():
                raise ConvertError(f"tensor '{name}': bytes differ")
            matched += 1
    return matched


def convert(ckpt: Path, out: Path | None = None) -> dict[str, object]:
    """Convert `ckpt` to safetensors 1:1 and return a summary (`metadata.json` goes to the same
    directory)."""
    if not ckpt.is_file():
        raise ConvertError(f"checkpoint not found: {ckpt}")
    target = out if out is not None else ckpt.with_suffix(".safetensors")
    metadata_path = target.with_name("metadata.json")

    source_sha256 = _sha256(ckpt)
    state_dict, metadata = _load_checkpoint(ckpt)
    source_text = _source_document(ckpt, source_sha256, metadata)

    # detach is a safeguard (a weights_only=True load carries no requires_grad). contiguous is
    # what the writer assumes, as it writes raw bytes through numpy. Neither changes the values.
    tensors = {key: value.detach().contiguous() for key, value in state_dict.items()}
    # MUST: move into place only after the checks pass (ADR 0052). The output directory
    # `inputs/<family>/<name>/` is **shared with hand-placed assets**, so the move happens twice
    # **per file**, not per directory (moving the whole directory would also replace the
    # neighbouring `weights.pth`).
    with (
        staged_publication(target) as staged_weights,
        staged_publication(metadata_path) as staged_metadata,
    ):
        save_file(
            {key: tensors[key] for key in sorted(tensors)},
            str(staged_weights),
            # MUST: one key (module doc) — the order of `__metadata__` is not preserved.
            metadata={SOURCE_KEY: source_text},
        )
        assert_reader_layout(staged_weights)
        matched = _assert_byte_identical(staged_weights, tensors)
        staged_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "source": str(ckpt),
        "source_sha256": source_sha256,
        "out": str(target),
        "out_sha256": _sha256(target),
        "out_bytes": target.stat().st_size,
        "metadata_json": str(metadata_path),
        "tensors": len(tensors),
        "byte_identical": matched,
        "parameters": sum(tensor.numel() for tensor in tensors.values()),
        "dtypes": sorted({str(tensor.dtype) for tensor in tensors.values()}),
        "prefixes": {
            prefix: sum(1 for key in tensors if key.split(".")[0] == prefix)
            for prefix in sorted({key.split(".")[0] for key in tensors})
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output safetensors (default: --ckpt with the extension changed). metadata.json goes"
        " to the same directory.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.ckpt, args.out), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
