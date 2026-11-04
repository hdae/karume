r"""Cross-check the post-processing of the A/B tool (`tools/birefnet-ab/`) with the same upstream
functions as the golden generator (ruling 4, ADR 0125 — verification of the tool's dry run).

    cd tools/export-recipes
    uv run --group birefnet python -m birefnet.matte_ab_check <the tool's --out>/work

Reads the logits the tool wrote per image (the graph output itself) and the 8-bit α per procedure
(listed by `work/manifest.json`), and checks them under the same contract as the golden
(`birefnet.matte_fixture`):

- `u8-pil-bicubic`: **0 mismatches** against `matte_fixture.hr_alpha` (f64 σ → round half up →
  Pillow's BICUBIC). For samples where 255σ is less than `TIE_MARGIN` from `k + 0.5` (ties — the
  same boundary at which the generator rejects fixture inputs), the rounding direction is left to
  the exp implementation, so only the output pixels such a sample reaches through the BICUBIC
  support are excluded, and the remaining pixels are compared for 0 mismatches (the number of
  excluded pixels is `excludedTiePixels` in the summary).
- `float-bilinear`: matches the round-half-up of `matte_fixture.lucida_scaled` (torch f32 σ →
  TF.resize → × 255), or may be off by one step only where the golden's `255α` is within
  `LUCIDA_EPSILON` of `k + 0.5`.

Fails when the logit shape does not match the manifest's `graph` dimensions
(`[1, 1, height, width]`). Any violation gives exit code 1 (the summary goes to stdout as JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from birefnet import matte_fixture

#: Names of the procedures the tool writes (`MATTE_PROCEDURES` of
#: `packages/models/src/birefnet/matte.ts`).
UPSTREAM = "u8-pil-bicubic"
CURRENT = "float-bilinear"


#: Pillow's BICUBIC support (the support of `bicubic_filter` in `Resample.c`).
BICUBIC_SUPPORT = 2.0


def bicubic_reach(in_size: int, out_size: int) -> np.ndarray:
    """Boolean table `[output, input]` for one axis — whether the BICUBIC support of an output index
    reaches an input index.

    `xmin` / `xmax` of Pillow's `precompute_coeffs` (clamped to `[0, inSize]` after the `(int)`
    truncation — the same formula as `fixedTaps` in `packages/models/src/image/pillow-resample.ts`).
    On an axis whose size does not change Pillow skips the pass, so each index reaches only itself.
    """
    if in_size == out_size:
        return np.eye(in_size, dtype=bool)
    scale = in_size / out_size
    support = BICUBIC_SUPPORT * max(scale, 1.0)
    reach = np.zeros((out_size, in_size), dtype=bool)
    for index in range(out_size):
        center = (index + 0.5) * scale
        start = max(0, int(center - support + 0.5))
        stop = min(in_size, int(center + support + 0.5))
        reach[index, start:stop] = True
    return reach


def tie_samples(logits: np.ndarray) -> np.ndarray:
    """Tie samples (255σ less than `TIE_MARGIN` from `k + 0.5`).

    The boundary is the same as `matte_fixture.assert_no_ties`.
    """
    scaled = matte_fixture.sigmoid_f64(logits) * 255
    return np.abs(scaled - (np.floor(scaled) + 0.5)) < matte_fixture.TIE_MARGIN


def tie_reach(ties: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    """Output pixels that some tie sample reaches through the BICUBIC support (booleans
    `[out_height, out_width]`).

    The filter is separable, so output `(Y, X)` reaches sample `(r, c)` when the vertical support
    reaches `r` and the horizontal support reaches `c`.
    """
    in_height, in_width = ties.shape
    rows, columns = np.nonzero(ties)
    vertical = bicubic_reach(in_height, out_height)[:, rows].astype(np.int64)
    horizontal = bicubic_reach(in_width, out_width)[:, columns].astype(np.int64)
    return (vertical @ horizontal.T) > 0


def check_upstream(logits: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    """Check the `u8-pil-bicubic` α against HR's upstream procedure.

    The contract is 0 mismatches, excluding the pixels tie samples reach ({@link tie_reach}).
    """
    height, width = alpha.shape
    excluded = tie_reach(tie_samples(logits), width, height)
    # The excluded pixels depend on how the tie samples round, but the rest do not, so either
    # rounding gives the same result.
    expected = matte_fixture.hr_alpha(logits, width, height, allow_ties=True)
    mismatches = int(np.count_nonzero((expected != alpha) & ~excluded))
    return {
        "mismatches": mismatches,
        "excludedTiePixels": int(np.count_nonzero(excluded)),
        "ok": mismatches == 0,
    }


def check_current(logits: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    """Check the `float-bilinear` α against Lucida's upstream procedure (only one step within ε
    is allowed)."""
    height, width = alpha.shape
    scaled = matte_fixture.lucida_scaled(logits, width, height).astype(np.float64)
    expected = np.floor(scaled + 0.5)
    actual = alpha.astype(np.float64)
    differs = actual != expected
    boundary = np.minimum(actual, expected) + 0.5
    allowed = (
        differs
        & (np.abs(actual - expected) == 1)
        & (np.abs(scaled - boundary) <= matte_fixture.LUCIDA_EPSILON)
    )
    outside = int(np.count_nonzero(differs & ~allowed))
    return {
        "withinEpsilon": int(np.count_nonzero(allowed)),
        "outsideEpsilon": outside,
        "ok": outside == 0,
    }


def check_entry(logits: np.ndarray, alphas: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """One image (logits `[h, w]` and the α per procedure `[H, W]`)."""
    missing = {UPSTREAM, CURRENT} - set(alphas)
    if missing:
        raise ValueError(f"no α for the procedures: {sorted(missing)}")
    return {
        UPSTREAM: check_upstream(logits, alphas[UPSTREAM]),
        CURRENT: check_current(logits, alphas[CURRENT]),
    }


def check_work(work: Path) -> dict[str, Any]:
    """Cross-check the tool's `work/` (every image in the manifest)."""
    from safetensors.numpy import load_file

    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    graph = manifest["graph"]
    # MUST: check the logit shape against the graph's dimensions — with mixed-up logits (another
    # model, transposed) the α of both procedures can still be made at the right size, and the
    # cross-check could only say "disagrees with upstream".
    logits_shape = (1, 1, graph["height"], graph["width"])
    report: dict[str, Any] = {}
    for entry in manifest["entries"]:
        tensors = load_file(work / entry["logits"])
        if len(tensors) != 1:
            raise ValueError(
                f"{entry['logits']}: {len(tensors)} tensors (expected the one graph output)"
            )
        (logits,) = tensors.values()
        if logits.shape != logits_shape:
            raise ValueError(
                f"{entry['logits']}: logit shape {list(logits.shape)} differs from the manifest's"
                f" graph {list(logits_shape)}"
            )
        alphas = load_file(work / entry["alpha"])
        for name, alpha in alphas.items():
            if alpha.dtype != np.uint8 or alpha.shape != (entry["height"], entry["width"]):
                raise ValueError(
                    f"{entry['alpha']}: {name} is not U8 [{entry['height']}, {entry['width']}]"
                )
        report[entry["name"]] = check_entry(logits[0, 0], alphas)
    return report


def violations(report: Mapping[str, Any]) -> list[str]:
    """The (image, procedure) pairs that break the contract."""
    return [
        f"{name}/{procedure}"
        for name, entry in report.items()
        for procedure, result in entry.items()
        if not result["ok"]
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "work", type=Path, help="the work/ of the tool's output (where manifest.json is)"
    )
    args = parser.parse_args(argv)
    report = check_work(args.work)
    failed = violations(report)
    print(json.dumps({"report": report, "violations": failed}, indent=1, ensure_ascii=False))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
