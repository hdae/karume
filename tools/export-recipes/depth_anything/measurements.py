"""Numbers measured during the Depth Anything V2 export (kept on the **side that does not load
torch**).

They live apart from the patch layer for the same reason as `siglip2.measurements`: the wording
that states the measured width is used by the distribution side (the model card in
`depth_anything.card` and the modification notice in `depth_anything.distribution`), and those
sit on the import chain of the `dist.py` driver. If `depth_anything.patch` held the number, the
patch layer (which depends on torch) would enter the distribution side's import chain. The
exporter core's `karume/__init__` resolves its public surface on first access (PEP 562), so
`import dist` does not load torch (measured 2026-09-24). Unless the patch layer is kept off the
distribution path's chain here, that property breaks through this file.

MUST: the measured width is spelled in this one file only. It used to be split between the patch
layer's docstring (`1.4e-06`) and the card's constant (`1.4e-6`), so **the same measured value
appeared under two spellings** — and since the prose stays plausible when only one of them moves,
nobody can notice until after it has shipped.
"""

from __future__ import annotations

#: Measured upper bound of the depth difference introduced by replacing the DPT reassemble
#: `ConvTranspose2d` with a 1×1 conv + pixel shuffle (step ③ of `depth_anything.patch`). It is the
#: least significant bits moving because of a different reduction order; the RMS of the depth
#: values is about 1.0. The source is `depth_anything/export.py --verify` (the maximum over the 4
#: synthetic cases — measured 2026-09-05: checker 1.67e-06 / disc 1.55e-06 / noise 1.67e-06 /
#: ramp 6.08e-06. The earlier spelling 1.4e-06 was the value of one case, not the upper bound of
#: the 4). `verify_patches` holds this value as a **check** and fails when a re-measurement
#: exceeds it (W-P6-3).
CONVT_MAXDIFF: float = 6.1e-06


def convt_diff_text() -> str:
    """The spelling of {@link CONVT_MAXDIFF} embedded in prose (`"6.1e-06"`).

    The format is a function so that the model card and `NOTICE.md` state **the same string**.
    """
    return f"{CONVT_MAXDIFF:g}"
