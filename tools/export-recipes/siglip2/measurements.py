"""Numbers measured during the SigLIP2 export (kept on the **side that does not load torch**).

They live apart from the patch layer because of the import chain. The text that states the
measured range is pulled in by the distribution side (the model card in `siglip2.card` and the
modification notice in `siglip2.distribution`), and those sit on the import chain of the `dist.py`
driver. If `siglip2.patch` held the numbers, the patch layer (which depends on torch) would join the
distribution side's import chain. The exporter core's `karume/__init__` resolves its public surface
on first access (PEP 562), so `import dist` does not load torch (measured 2026-09-24). Unless the
patch layer is kept off the distribution path's chain here, that property breaks through this file.
The other aim is to spell the measured range in one place.
This is a leaf module of constants and their string forms only, in the same direction as
`gemma4.distribution` copying `SYM_MAX` to the side that does not load torch
(there a test checks that the copy holds the same value; here **the source of truth itself** lives
here, so there is no copy).

MUST: the text that states the measured range exists only once, in this file. The model card and
`NOTICE.md` (the modification notice) state the same number, so keeping it in two places silently
allows the day when only one moves and "only the notice claims a stale number" — it still reads as
valid prose, so nobody can notice until after it has shipped.
"""

from __future__ import annotations

#: Measured range (lower, upper) of the pooler_output difference that making the MAP head's q/k/v
#: explicit (step ③ of `siglip2.patch`) introduces. Source: the 4 golden cases × 2 series of
#: `siglip2/export.py --verify`.
MAP_HEAD_MAXDIFF: tuple[float, float] = (7.75e-07, 2.38e-06)


#: Scale the difference is divided by = the measured range (lower, upper) of the pooled vector's
#: L2 norm. Taken as a pair from the same `--verify` run; **without this number the size of the
#: maxdiff cannot be read** (the basis of the relative ~1e-7).
MAP_HEAD_VECTOR_NORM: tuple[float, float] = (12.7, 13.1)


def map_head_diff_text() -> str:
    """{@link MAP_HEAD_MAXDIFF} as text to embed in prose (`"7.75e-07 to 2.38e-06"`).

    The format is a function because the model card and `NOTICE.md` state **the same string**
    (writing `f"{low:g} to {high:g}"` in each place means nothing fails when only one side's format
    moves).
    """
    low, high = MAP_HEAD_MAXDIFF
    return f"{low:g} to {high:g}"


def map_head_norm_text() -> str:
    """{@link MAP_HEAD_VECTOR_NORM} as text to embed in the distributed text (`"around 13"`).

    Internally it is kept as a range (a pair); the distributed text states a rounded scalar —
    printing the range would attach precision that the claim "the difference is negligible" does
    not need. It has the same two layers as {@link map_head_diff_text} (constant + string form), so
    the card and `NOTICE.md` pull the same single word.
    """
    low, high = MAP_HEAD_VECTOR_NORM
    return f"around {round((low + high) / 2):g}"
