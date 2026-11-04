"""Rebuild a civitai single-file checkpoint into the diffusers layout (the entry for variants).

    uv run --group anima python -m anima.single_file \
        --checkpoint ../../inputs/anima/waiANIMA_v10Base10.safetensors \
        --out ../../outputs/misc/anima-diffusers/anima-wai

`--repo` of `anima.export` is a string passed as is to `from_pretrained(repo, subfolder=...)`,
so it accepts only **a repository / directory in the diffusers layout** (there is no path that
reads a single file). This module provides the bridge.

The resulting directory is "a copy of base with only transformer and text_conditioner replaced":

- **The checkpoint holds only two parts: the DiT (567 tensors) and llm_adapter (118 tensors)**;
  text_encoder / VAE / tokenizer are not in it (measured 2026-08-22). So those are taken from base
  as they are (**symlinks** — several GB are not duplicated per variant).
- The key names use the ComfyUI-style spelling (a `net.` or `model.diffusion_model.` prefix), so
  they go through diffusers' conversion table for Cosmos. Measured, they cover the keys of base's
  transformer **exactly, with nothing missing or extra**, and with 0 shape mismatches.

MUST: compare the converted key set with base's actual files before writing. If the table
changes upstream and "weights only partly carried over" are written out silently, both the
export and the distribution pass and **only the images break, silently**.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# MUST: import `diffusers` / `huggingface_hub` **inside functions** (the same practice as the
# other modules of this recipe — none of `export.py` / `patch.py` / `lora.py` /
# `rope_tables.py` / `pipeline_ref.py` imports them at module level). Both belong to the `anima`
# dependency group and are **not part of the default `uv sync`** (the NOTE of the same name in
# pyproject). Importing them at module level makes merely reading this file raise
# `ModuleNotFoundError`, and in an environment without the group synced **the tests fail at
# collection** (hit on 2026-08-22). They are needed only when the conversion runs; renaming and
# comparing the keys (the pure functions below) depend on neither.

#: The upstream diffusers repository (the same one as `DEFAULT_REPO` in `anima/export.py`).
BASE_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"

#: The prefix the conversion table expects. The civitai side wraps keys in either `net.` or
#: `model.diffusion_model.`.
CANONICAL_PREFIX = "net."
KNOWN_PREFIXES = ("model.diffusion_model.", "net.")

#: The subtree on the checkpoint side that corresponds to text_conditioner.
CONDITIONER_PREFIX = "llm_adapter."

#: The parts taken from base by symlink (what the checkpoint does not hold).
SHARED_ENTRIES = ("text_encoder", "vae", "tokenizer", "t5_tokenizer", "scheduler")

#: The two parts that are replaced, and their weight file name (the diffusers default).
WEIGHT_FILENAME = "diffusion_pytorch_model.safetensors"

#: The record of where the conversion came from (for people to trace — the distribution's
#: attribution is held by `anima/card.py`).
SOURCE_PROVENANCE_FILE = "source_provenance.json"

#: The intake record (a machine-owned file that `anima.civitai` writes next to the checkpoint).
CIVITAI_PROVENANCE_FILE = "civitai.json"

#: The layout the intake creates (`civitai-<versionId>/`, made by `fetch_checkpoint` of
#: `anima.civitai`). A checkpoint there is known to have come through the intake, so a missing
#: record can be told apart from a hand-placed file.
CIVITAI_INTAKE_DIR = re.compile(r"civitai-[0-9]+")

#: The fields carried over from the record above (the HTML of the description is not carried —
#: it is not needed to trace where the conversion came from). Only `sha256` sits under `file`, so
#: it is handled separately.
CIVITAI_CARRIED_FIELDS = (
    "air",
    "model_id",
    "version_id",
    "model_name",
    "version_name",
    "derived_name",
    "permissions",
)


def _strip_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip a known prefix and normalize to the `net.` the conversion table expects."""
    for prefix in KNOWN_PREFIXES:
        if all(name.startswith(prefix) for name in state):
            return {
                f"{CANONICAL_PREFIX}{name.removeprefix(prefix)}": value
                for name, value in state.items()
            }
    heads = sorted({name.split(".")[0] for name in state})
    raise SystemExit(f"unknown key prefix: {heads} (expected: {', '.join(KNOWN_PREFIXES)})")


def _assert_covers(converted: Mapping[str, torch.Tensor], reference: Path, label: str) -> None:
    """Check that the converted state covers base's actual files exactly (key set and shapes)."""
    expected = load_file(reference)
    missing = sorted(set(expected) - set(converted))
    extra = sorted(set(converted) - set(expected))
    if missing or extra:
        raise SystemExit(
            f"{label}: the converted keys do not match base — {len(missing)} missing"
            f" {missing[:3]} / {len(extra)} extra {extra[:3]}"
        )
    mismatched = [
        name for name in expected if tuple(expected[name].shape) != tuple(converted[name].shape)
    ]
    if mismatched:
        raise SystemExit(
            f"{label}: shape differs from base in {len(mismatched)} keys {mismatched[:3]}"
        )


def _link(target: Path, link: Path) -> None:
    """Remove what exists, then link again (idempotent — a rerun leaves no stale target).

    A real directory sitting there is removed too — real contents end up where a symlink was
    meant to be after someone copies the contents (`cp -r` follows symlinks). `unlink` fails
    with `IsADirectoryError`, so the cases are split by kind.
    """
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    link.symlink_to(target)


def _civitai_provenance(checkpoint: Path) -> dict[str, object] | None:
    """Narrow the `civitai.json` next to the checkpoint (the record of `anima.civitai`) to the
    carried fields.

    A hand-placed checkpoint has none, so a missing file gives `None` (a consequence of ADR 0088
    — a source other than Civitai simply has no `civitai.json` and gets the conventional
    provenance). **But if the parent is the intake layout (`civitai-<versionId>/`), the missing
    record itself means a broken intake, so it fails** — unlike a hand-placed file, a gap here is
    the trace of a broken provenance chain, and silently treating it as hand-placed would let
    weights of unknown origin through. **It also fails when the record exists but cannot be read
    or lacks fields** — silently dropping the section would present weights of unknown origin as
    "with provenance".
    """
    record_path = checkpoint.parent / CIVITAI_PROVENANCE_FILE
    if not record_path.exists():
        if CIVITAI_INTAKE_DIR.fullmatch(checkpoint.parent.name):
            raise SystemExit(
                f"{checkpoint.parent} is in the intake layout but has no {CIVITAI_PROVENANCE_FILE}"
                " (the intake broke midway, or the provenance was deleted) — "
                "redo the intake with `python -m anima.civitai`"
            )
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read {record_path}: {error}") from error
    if not isinstance(record, dict):
        raise SystemExit(f"{record_path} is not a JSON object")

    file = record.get("file")
    missing = [field for field in CIVITAI_CARRIED_FIELDS if field not in record]
    if not isinstance(file, dict) or "sha256" not in file:
        missing.append("file.sha256")
    if missing:
        raise SystemExit(
            f"{record_path} is missing fields: {missing} (the shape `anima.civitai` writes)"
        )

    return {
        "air": record["air"],
        "model_id": record["model_id"],
        "version_id": record["version_id"],
        "model_name": record["model_name"],
        "version_name": record["version_name"],
        "derived_name": record["derived_name"],
        "sha256": file["sha256"],
        "permissions": record["permissions"],
    }


def _source_provenance(checkpoint: Path, base_repo: str) -> dict[str, object]:
    """The contents of `source_provenance.json` (with a `civitai` section when a civitai intake
    record exists)."""
    provenance: dict[str, object] = {"file": checkpoint.name, "base_repo": base_repo}
    civitai = _civitai_provenance(checkpoint)
    if civitai is not None:
        provenance["civitai"] = civitai
    return provenance


def convert(checkpoint: Path, out: Path, base_repo: str = BASE_REPO) -> None:
    from diffusers.loaders.single_file_utils import (
        convert_cosmos_transformer_checkpoint_to_diffusers,
    )
    from huggingface_hub import snapshot_download

    # Read the record **before the heavy work** (if it is broken, fail before minutes of
    # conversion).
    provenance = _source_provenance(checkpoint, base_repo)
    base = Path(snapshot_download(base_repo))
    stripped = _strip_prefix(load_file(checkpoint))
    # MUST: text_conditioner **does not go through the conversion table**, and **is taken out
    # before the table is called**. (1) The Cosmos table rewrites even the inside of llm_adapter
    # to diffusers names (`transformer_blocks.0.attn1.*`), but base's text_conditioner keeps the
    # original spelling (`blocks.0.cross_attn.*`), and its reader (`AnimaTextConditioner`)
    # expects that. The raw checkpoint keys map 1:1 to base once the prefix is stripped
    # (measured 2026-08-22). (2) The table consumes its input dict with pop, so looking up the
    # original keys after calling it finds nothing (hit the same day).
    conditioner_prefix = f"{CANONICAL_PREFIX}{CONDITIONER_PREFIX}"
    conditioner = {
        name.removeprefix(conditioner_prefix): value
        for name, value in stripped.items()
        if name.startswith(conditioner_prefix)
    }
    converted = convert_cosmos_transformer_checkpoint_to_diffusers(stripped)
    transformer = {
        name: value for name, value in converted.items() if not name.startswith(CONDITIONER_PREFIX)
    }
    _assert_covers(transformer, base / "transformer" / WEIGHT_FILENAME, "transformer")
    _assert_covers(conditioner, base / "text_conditioner" / WEIGHT_FILENAME, "text_conditioner")

    out.mkdir(parents=True, exist_ok=True)
    for name, state in (("transformer", transformer), ("text_conditioner", conditioner)):
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        # The checkpoint has no config (the architecture is identical to base), so copy it from
        # base.
        (target / "config.json").write_bytes((base / name / "config.json").read_bytes())
        save_file(state, target / WEIGHT_FILENAME)
        print(f"[single-file] {name}: {len(state)} tensors", flush=True)

    for entry in SHARED_ENTRIES:
        _link(base / entry, out / entry)
    print(f"[single-file] symlinked from base: {', '.join(SHARED_ENTRIES)}", flush=True)

    (out / SOURCE_PROVENANCE_FILE).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[single-file] {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="civitai single-file checkpoint"
    )
    parser.add_argument("--out", type=Path, required=True, help="directory to rebuild into")
    parser.add_argument(
        "--base-repo", default=BASE_REPO, help="diffusers repository to take the missing parts from"
    )
    args = parser.parse_args()
    convert(args.checkpoint, args.out, args.base_repo)


if __name__ == "__main__":
    main()
