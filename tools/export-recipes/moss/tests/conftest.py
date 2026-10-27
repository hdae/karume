"""Fixtures shared by the MOSS-Image recipe tests.

Tests that need the pinned upstream files take {@link moss_snapshot} (the DiT, VAE, scheduler and
root files) or {@link moss_text_snapshot} (the tokenizer and text encoder). On a machine without
them in the HF cache, or without the `moss` group synced, those tests SKIP with the reason, so the
default sync never fails at collection (the direction of `tests/test_optional_group_imports.py`).
Fetching: `uv run --group moss --inexact python -m moss.sources --fetch [--text]`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _snapshot(*, text: bool) -> Path:
    pytest.importorskip("huggingface_hub")
    from moss import sources

    try:
        return sources.text_snapshot() if text else sources.local_snapshot()
    except sources.MossSourceError as error:
        pytest.skip(f"the pinned MOSS-Image snapshot is not cached: {error}")


@pytest.fixture(scope="session")
def moss_snapshot() -> Path:
    return _snapshot(text=False)


@pytest.fixture(scope="session")
def moss_text_snapshot() -> Path:
    return _snapshot(text=True)
