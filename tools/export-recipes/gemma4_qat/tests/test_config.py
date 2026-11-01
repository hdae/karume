"""QAT の小さな正本（`gemma4_qat.config`）が、torch を避けるために持つ写しの定数。

`config.py` は dist から torch を import しないために、通常 Gemma の製品形が持つ値を
写しで綴る。写しは正本を動かした日に片方だけ古びるので、同値をここで縛る。
"""

from __future__ import annotations

import re

from gemma4 import export_product
from gemma4.provenance import FINGERPRINT_FILES
from gemma4_qat.config import CHECKPOINTS, MAX_SELECTED_ROWS, SOURCES


class TestCopiedConstants:
    def test_the_selected_rows_ceiling_follows_the_product_form(self) -> None:
        """trace の記号 R の上限は、製品形の `ROW_SYM_MAX`（sliding ring の余裕 + 1）と同じ。

        食い違うと、QAT の trace 上限と配布形が宣言する `slidingSlack` がずれる（余裕を動かした
        日に QAT 側だけ古い上限で焼かれる）。
        """
        assert MAX_SELECTED_ROWS == export_product.ROW_SYM_MAX


class TestPin:
    def test_every_revision_is_a_full_commit_sha(self) -> None:
        """A branch name or a short SHA reads other contents once the branch moves."""
        for source in SOURCES.values():
            assert re.fullmatch(r"[0-9a-f]{40}", source.revision), source.revision

    def test_every_model_is_pinned(self) -> None:
        assert set(SOURCES) == set(CHECKPOINTS)

    def test_each_pin_covers_every_file_the_export_fingerprints(self) -> None:
        """The distribution compares the pin with `reference.json`; a file left out of the pin
        would go unchecked."""
        for source in SOURCES.values():
            assert tuple(source.sha256) == FINGERPRINT_FILES
            for sha256 in source.sha256.values():
                assert re.fullmatch(r"[0-9a-f]{64}", sha256), sha256
