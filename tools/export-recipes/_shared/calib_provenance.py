"""Decide whether the calibration conditions of an i4 series (`calib_provenance.json`) are good
enough for distribution (ADR 0065 decision 2).

Neither the calibration method nor its budget **changes the storage by a single byte** (the grid
stays RTN i4 g32 — only the rounded values and the scale ledger change), so both the structural
check of `verify_dist` and the storage check of the binding table (`assert_storage`) let them
through.
The only fact that tells them apart is "the record the writer left", and three places read that
record: `irodori.distribution` (assembling the distribution), `irodori.pipeline_ref` (rebuilding
the golden) and `anima.distribution` (assembling likewise).

MUST: **this module is the single source of truth for the decision** (readers call it and add
only their family-specific follow-up). With three copies, the day one check is tightened the
others silently become weaker — in fact, while only the method field was examined, "assets
built with the smoke budget (`--calib-steps 1`)" passed all three (review Ca17 of 2026-08-24).
Same reason as `_shared.decode_series`.

MUST: **a record without a field is accepted**. Fields can be added later (anima's `guidance` was
added on 2026-08-23), so requiring their presence would make a one-line addition impose "a
re-export costing the rounding time" on every existing series. Only the fields present are
examined: **budgets against a lower bound, model conditions by equality**.

The dependency direction is recipe → core only (`_shared` does not import families) — callers
take both the bounds and the conditions from their source of truth and pass them in.
"""

from __future__ import annotations

from collections.abc import Mapping


def _number(value: object) -> float | None:
    """Read a record field as a number (`None` if it cannot be read).

    `bool` is rejected so that `True` does not pass a lower bound as 1 (the record writer never
    writes a JSON `true`, so one appearing means the record itself is broken).
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def calib_complaint(
    record: object,
    *,
    method: str,
    at_least: Mapping[str, float] | None = None,
    exactly: Mapping[str, float] | None = None,
) -> str | None:
    """Return in one sentence why the record fails the distribution conditions (`None` if it
    meets them).

    `at_least` holds the **budget** (fields whose reduction only lowers quality — the number of
    capture steps, of calibration prompts, of corpus entries); `exactly` holds **the conditions it
    was built under** (fields fixed per model — the step count, CFG). Budgets are checked against
    a lower bound and conditions by equality because "bigger is better" does not hold for the
    latter (putting an asset built with turbo's 8 steps into the base model's slot is as wrong as
    the reverse).

    A record that is not a mapping is treated as having an unreadable `method` (it fails as a
    method mismatch) — a broken record is never accepted as "just missing fields".
    """
    fields = record if isinstance(record, Mapping) else {}
    if fields.get("method") != method:
        return (
            "not built with a rounding method allowed for distribution"
            f" ({fields.get('method')!r}; allowed: {method!r})"
        )
    for name, floor in (at_least or {}).items():
        if name not in fields:
            continue
        value = _number(fields[name])
        if value is None:
            return f"calibration budget {name!r} is not a number ({fields[name]!r})"
        if value < floor:
            return (
                f"calibration budget {name!r} is below the distribution floor"
                f" ({value:g} < {floor:g})"
            )
    for name, want in (exactly or {}).items():
        if name not in fields:
            continue
        value = _number(fields[name])
        if value is None:
            return f"calibration condition {name!r} is not a number ({fields[name]!r})"
        if value != want:
            return (
                f"calibration condition {name!r} differs from this model's default"
                f" ({value:g}; default {want:g})"
            )
    return None
