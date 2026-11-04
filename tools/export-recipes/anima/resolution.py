"""Resolution spelling (`WxH` and the square shorthand `N`) — the one place the reference scripts
share (#23).

The reference scripts use the same spelling as the demo's `--resolution`
(`examples/anima/main.ts`). If the fixture and the demo cannot be named by the same string, a
mix-up of the form "a 768x1344 demo was checked against a 1344x768 reference" becomes invisible
in the names.

MUST: **the TS side is the source of truth for the accepted set**
(`packages/models/src/anima/resolution.ts`). This module checks only what a reference fixture
needs to be structurally valid (a positive integer, a multiple of {@link GRANULARITY}); it checks
neither the asset-side lower bound (each side ≥ 512 = VAE tile latent 64) nor the upper bound of
`Dim("S")` — the reference scripts sometimes run shapes the demo does not accept (the synthetic
cases of the unit tests). Writing the same accepted set on both sides would let shapes that "the
reference can produce but the demo does not accept" grow silently when only one side changes.
"""

from __future__ import annotations

#: Resolution step (= spatial compression 8 × patch 2). The smallest unit at which each latent
#: side divides by patch 2; off it, patchify cannot be built (the host-side `latentSides` rejects
#: it).
GRANULARITY = 16


def parse_resolution(text: str) -> tuple[int, int]:
    """`"1344x768"` → `(1344, 768)`, `"512"` → `(512, 512)` (square shorthand).

    MUST: do not silently accept spellings other than the shorthand and `WxH`. Rejecting
    `1344*768` or `1344,768` as "not a number" is a spelling check at the entry; the validity of
    the dimensions is not checked here.
    """
    parts = text.lower().split("x")
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"resolution {text!r} is neither WxH nor the square shorthand (e.g. 1344x768 / 512)"
        )
    width, height = (int(part) for part in parts)
    for side, label in ((width, "width"), (height, "height")):
        if side <= 0 or side % GRANULARITY != 0:
            raise ValueError(
                f"resolution {text!r}: {label} {side} is not a positive multiple of {GRANULARITY}"
            )
    return width, height


def format_resolution(width: int, height: int) -> str:
    """`(1344, 768)` → `"1344x768"`; a square uses the shorthand (`(512, 512)` → `"512"`).

    The inverse of {@link parse_resolution}; it returns a spelling whose round trip is the
    identity (it ends up in directory names, so the paths of the existing square fixtures must
    not change by a single character).
    """
    return str(width) if width == height else f"{width}x{height}"


def resolution_meta(width: int, height: int) -> dict[str, int | str]:
    """The resolution fields of a fixture JSON (`width` / `height` are the source of truth for the
    dimensions).

    MUST: for a square run, emit `resolution` **as the int side length, as before**. There is a
    reader that reads this field of the output
    (`outputs/series/anima-tiling-<dtype>-<resolution>/tiling.json`) as a number (the instance is
    `TestResolutionMeta` in `anima/tests/test_resolution.py`); changing its type breaks silently,
    and only on regeneration. A non-square shape has no single side, so it gets the spelling
    (`"1344x768"`) — readers should look at `width` / `height`.
    """
    return {
        "resolution": width if width == height else format_resolution(width, height),
        "width": width,
        "height": height,
    }
