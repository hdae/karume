"""`moss/prompts.py` — the fixed prompts and their sources."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from moss import prompts, sources


class TestFixedPrompts:
    def test_three_positives_and_the_empty_negative(self):
        roles = [prompt.role for prompt in prompts.FIXED_PROMPTS]
        names = [prompt.name for prompt in prompts.FIXED_PROMPTS]
        negatives = [p for p in prompts.FIXED_PROMPTS if p.role == prompts.NEGATIVE]

        assert roles.count(prompts.POSITIVE) == 3
        assert [p.text for p in negatives] == [""]
        assert len(set(names)) == len(names)

    def test_every_source_url_pins_a_commit(self):
        """A branch URL can point at another text once the branch moves."""
        for prompt in prompts.FIXED_PROMPTS:
            assert re.search(r"/blob/[0-9a-f]{40}/", prompt.url), prompt.url

    def test_an_unknown_name_fails_loudly(self):
        with pytest.raises(prompts.FixedPromptError):
            prompts.prompt_by_name("no-such-prompt")

    def test_the_card_example_is_the_one_in_the_pinned_readme(self, moss_snapshot):
        """`fridge-note` is the quick start's `prompt = '''…'''` of the pinned model card."""
        readme = (moss_snapshot / "README.md").read_text(encoding="utf-8")
        start = readme.index("prompt = '''") + len("prompt = '''")
        quoted = readme[start : readme.index("'''", start)]

        assert prompts.prompt_by_name("fridge-note").text == quoted

    def test_the_guide_note_differs_from_the_card_note_only_in_its_quotes(self):
        """The guide's second example is the card's with typographic quotes (kept as quoted)."""
        card = prompts.prompt_by_name("fridge-note").text
        guide = prompts.prompt_by_name("guide-fridge-note").text

        assert guide != card
        assert guide.replace("“", '"').replace("”", '"') == card

    def test_the_guide_examples_are_in_the_pinned_user_guide(self):
        """The two user-guide prompts are quoted verbatim from `USERGUIDE.md` at the pinned commit
        (SKIP until `python -m moss.sources --fetch --guide` has kept it)."""
        try:
            guide = sources.local_user_guide()
        except sources.MossSourceError as error:
            pytest.skip(str(error))
        quoted = [p for p in prompts.FIXED_PROMPTS if p.url == sources.USER_GUIDE.url]

        assert [p.name for p in quoted] == ["guide-whiteboard", "guide-fridge-note"]
        for prompt in quoted:
            assert prompt.text in guide, prompt.name


class TestImportCost:
    def test_importing_prompts_and_sources_does_not_load_torch(self):
        """The readers that must not load torch (the card, the assembly checks) import these two.

        A fresh interpreter: the test session has torch loaded already.
        """
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\nimport moss.prompts, moss.sources\nprint('torch' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(prompts.__file__).resolve().parent.parent,
        )

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "False"
