"""Transactional publication of artifacts — build in staging, swap into final once it passes
(ADR 0052).

Whatever the writer (assembling a distribution, writing a series directory, a single export), the
discipline is one: **final is not touched by a single byte until the last rename**. Whether the
write fails midway, a verification check rejects it, or Ctrl-C hits, the canonical path keeps the
previous last-known-good, and neither a "written but unreadable" asset nor a "mix of old and new"
asset ever appears.

There are only two surfaces:

- {@link staged_publication} — the whole thing, from preparing the staging path to the swap. The
  body of `with` is "write + verification check", and the swap happens only when it exits without
  an exception. The caller owns neither the path spelling nor the cleanup.
- {@link swap_into_place} — the swap only. For callers that already hold a filled staging path.

The staging path (`<final name>.staging`) and the superseded path (likewise `.old`) are **always**
created in the same parent as final — rename is atomic only within one file system, and moving them
to `/tmp` or the like turns the swap into a cross-device copy and loses atomicity altogether. Both
have **fixed names**, and a path left behind by an interruption is silently discarded by the next
run (scattering unique names would pile up debris that can never be picked up again, one per run).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: The path that holds the artifact being assembled (same parent as final, fixed name).
STAGING_SUFFIX = ".staging"

#: The path that keeps the last-known-good until just before the swap (likewise).
SUPERSEDED_SUFFIX = ".old"


class ArtifactSwapError(OSError):
    """The rename of a swap failed (the causing OSError is chained in `__cause__`).

    It inherits OSError because, for the caller, this is the I/O failure itself — an `except
    OSError` from the days of calling bare `os.replace` keeps working without changing meaning.
    """


def _discard(path: Path) -> None:
    """Remove `path` entirely (if present) — cleanup of the staging and superseded paths.

    Reusing a path left behind by an interruption would silently mix files the writer did not plan
    into the new artifact.
    """
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _reclaim_superseded(final: Path, superseded: Path) -> None:
    """Empty the superseded path — but if it is the last-known-good, move it back to final.

    A superseded path present with no final is the shape of a previous run that failed
    **between** the two renames. That superseded path is the last-known-good itself, so discarding
    it would start the next run from a state where "the local artifact is completely gone" (a
    superseded path next to an existing final is just debris a run that finished the swap failed
    to remove).
    """
    if superseded.exists() and not final.exists():
        os.replace(superseded, final)
    _discard(superseded)


def swap_into_place(staged: Path, final: Path) -> None:
    """Put the filled staging path `staged` in place of `final` **as a whole** (nothing of the old
    content is carried over).

    A rename cannot land on a non-empty directory, so swapping a directory takes two steps: move
    aside, then promote. A failure between the two leaves final **absent**, but a reader detects
    absence reliably — what must not be created is "a silently readable mix of old and new", and
    this procedure cannot create it by construction. If the promotion fails, the superseded path is
    moved back, so the caller keeps the last-known-good.

    A file (and an absent final) is put in place by one `os.replace` — stronger, since no
    superseded path is created, and there is no moment at which final is absent.

    The staging path is not removed — its owner is the caller (through
    {@link staged_publication}, that side discards it).
    """
    superseded = final.with_name(final.name + SUPERSEDED_SUFFIX)
    _reclaim_superseded(final, superseded)
    if not final.is_dir():
        try:
            os.replace(staged, final)
        except OSError as error:
            raise ArtifactSwapError(f"swap (rename) into {final} failed: {error}") from error
        return
    try:
        os.replace(final, superseded)
    except OSError as error:
        # os.replace is atomic — final stays intact even on failure.
        raise ArtifactSwapError(f"moving {final} aside (rename) failed: {error}") from error
    try:
        os.replace(staged, final)
    except OSError as error:
        # Do not stop in a state where the only valid artifact sits in the superseded path.
        os.replace(superseded, final)
        raise ArtifactSwapError(f"swap (rename) into {final} failed: {error}") from error
    _discard(superseded)


@contextmanager
def staged_publication(final: Path) -> Iterator[Path]:
    """Hand out the staging path, and put it in place of `final` only when `with` exits **without
    an exception**.

    The body is all of "write + verification check"; wherever it fails (Ctrl-C included), only the
    staging path is removed and final does not change by a single byte. Since the check can sit
    inside the body, the order "write → verify → write more from the values that passed that
    verification" can also be built within the same staging path.

    MUST: checks that apply outside the staging path (the overall preconditions the caller
    prepares) are done **before** entering `with` — failing inside means discarding only after
    laying out several GB, even if discarding is all it does.

    The staging path itself is **not created** — a writer that needs a directory runs mkdir itself,
    and a writer of one file writes straight to that path (only the writer knows which it needs).
    """
    staging = final.with_name(final.name + STAGING_SUFFIX)
    superseded = final.with_name(final.name + SUPERSEDED_SUFFIX)
    _discard(staging)
    _reclaim_superseded(final, superseded)
    try:
        yield staging
        swap_into_place(staging, final)
    except BaseException:
        _discard(staging)
        raise
