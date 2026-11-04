"""The copied constants the small QAT source of truth (`gemma4_qat.config`) keeps to avoid torch.

So that dist does not import torch, `config.py` spells copies of values the regular Gemma's
product form holds. A copy goes stale on one side the day the source of truth moves, so the
equality is pinned here.
"""

from __future__ import annotations

import re

from gemma4 import export_product
from gemma4.provenance import FILES_BY_ROLE
from gemma4.sources import REGULAR_TOKENIZER_PINS
from gemma4_qat.config import CHECKPOINTS, MAX_SELECTED_ROWS, SOURCES, checkpoint_repo


class TestCopiedConstants:
    def test_the_selected_rows_ceiling_follows_the_product_form(self) -> None:
        """The upper bound of the trace symbol R equals the product form's `ROW_SYM_MAX` (the
        sliding ring's slack + 1).

        If they disagree, the QAT trace bound and the `slidingSlack` the distribution declares
        drift apart (the day the slack moves, only the QAT side is built with the old bound).
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
