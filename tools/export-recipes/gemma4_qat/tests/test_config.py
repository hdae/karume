"""QAT の小さな正本（`gemma4_qat.config`）が、torch を避けるために持つ写しの定数。

`config.py` は dist から torch を import しないために、通常 Gemma の製品形が持つ値を
写しで綴る。写しは正本を動かした日に片方だけ古びるので、同値をここで縛る。
"""

from __future__ import annotations

import re

from gemma4 import export_product
from gemma4.provenance import FILES_BY_ROLE
from gemma4.sources import REGULAR_TOKENIZER_PINS
from gemma4_qat.config import CHECKPOINTS, MAX_SELECTED_ROWS, SOURCES, checkpoint_repo


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
            for pin in source.pins.by_role().values():
                assert re.fullmatch(r"[0-9a-f]{40}", pin.revision), pin.revision

    def test_every_model_is_pinned(self) -> None:
        assert set(SOURCES) == set(CHECKPOINTS)

    def test_each_pin_covers_every_file_the_export_fingerprints(self) -> None:
        """The distribution compares the pins with `reference.json`; a file left out of a pin
        would go unchecked, and one pinned in the other source's role would be checked against
        bytes of another snapshot."""
        for source in SOURCES.values():
            for role, pin in source.pins.by_role().items():
                assert tuple(pin.sha256) == FILES_BY_ROLE[role]
                for sha256 in pin.sha256.values():
                    assert re.fullmatch(r"[0-9a-f]{64}", sha256), sha256

    def test_the_weights_come_from_the_mobile_repository(self) -> None:
        for model, source in SOURCES.items():
            assert source.pins.weights.repo == checkpoint_repo(model)

    def test_the_tokenizer_is_the_regular_lines_fixed_one(self) -> None:
        """The mobile repositories ship the `tokenizer.json` from before Google's glitch-token
        fixes; the line reads the regular `-it` repository at the pins the q4_0 QAT line reads
        (one spelling — `gemma4.sources.REGULAR_TOKENIZER_PINS`)."""
        for model, source in SOURCES.items():
            assert source.pins.tokenizer == REGULAR_TOKENIZER_PINS[model]
