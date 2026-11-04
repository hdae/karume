"""Make the clone of Irodori's model implementation importable (the default `--source-dir` of
`irodori.export`).

It is put on `sys.path` **when present** — the Irodori-side patches of {@link irodori.patch}
replace parts of `irodori_tts`, so the equivalence tests need this clone. It is not tracked by
git, so in an environment without it the tests skip (`pytest.importorskip("irodori_tts")`).

It lives in `irodori/tests/` because **only these 4 tests need it** (pytest reads the conftest
files from the rootdir down to a test file's directory before importing the tests of that
directory). Placed at the top of export-recipes, the presence of the clone would also affect
collection of the other families — a dependency's reach stays next to what requires it.

NOTE: this used to live in the core's `tools/exporter/tests/conftest.py` (one of the reverse
dependencies the Context of ADR 0065 names). It moved down here together with the family (step
4 of the same ADR).
"""

from __future__ import annotations

import sys

import pytest

from _shared.paths import REPO_ROOT

#: The same location as `irodori.export.DEFAULT_SOURCE_DIR` (if the spellings diverge, one of
#: them silently finds nothing).
IRODORI_SOURCE_DIR = REPO_ROOT / "inputs" / "irodori" / "Irodori-TTS"
if IRODORI_SOURCE_DIR.is_dir() and str(IRODORI_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(IRODORI_SOURCE_DIR))


@pytest.fixture
def restore_forward():
    """Undo the class-attribute replacements of `irodori.patch.apply_patches` after the test.

    The replacement is process-wide, so without undoing it the following tests run on the
    patched implementation.

    `apply_patches` also replaces parts of `irodori_tts` (a clone not tracked by git — added to
    `sys.path` above), so in an environment without it only the tests that use this fixture skip.
    It lives here because both the patch equivalence (`test_patch.py`) and the DiT split
    equivalence (`test_export.py`) use it.
    """
    from irodori import patch as patch_irodori

    modernbert = pytest.importorskip("transformers.models.modernbert.modeling_modernbert")
    irodori_model = pytest.importorskip("irodori_tts.model")
    original = modernbert.ModernBertAttention.forward
    original_rope = irodori_model.apply_rotary_emb
    original_norm = irodori_model.RMSNorm.forward
    original_adaln = irodori_model.LowRankAdaLN.forward
    applied = patch_irodori._APPLIED
    try:
        yield irodori_model
    finally:
        modernbert.ModernBertAttention.forward = original
        irodori_model.apply_rotary_emb = original_rope
        irodori_model.RMSNorm.forward = original_norm
        irodori_model.LowRankAdaLN.forward = original_adaln
        patch_irodori._APPLIED = applied
        patch_irodori._ORIGINAL_APPLY_ROTARY_EMB = None
