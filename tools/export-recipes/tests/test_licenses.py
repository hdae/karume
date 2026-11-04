"""The original license texts bundled at the root of the distribution repositories
(`_shared.licenses`).

Being "a verbatim copy" is the requirement itself (Apache 2.0 §4(a) / the MIT permission notice /
CC BY-SA 4.0 §3(a)(1)(C)), so
2 points are pinned here: **it has not turned into a summary** (the clauses are actually there),
and the MIT placeholder changes **only** the copyright lines. A rewording still reads as valid
prose, and the mismatch would go unnoticed until it is distributed.

NOTE: "is the `LICENSE.md` of the assembled distribution byte-identical to the original" is
checked by the **family side**'s `Test*LegalText` (the original → distribution path does not go
through here, so matching this helper against itself could only be vacuous).
"""

from __future__ import annotations

import hashlib
import re

import pytest

from _shared.licenses import (
    APACHE_LICENSE_2_0_PATH,
    CC_BY_SA_4_0_PATH,
    MIT_COPYRIGHT_PLACEHOLDER,
    MIT_LICENSE_PATH,
    apache_license_2_0,
    cc_by_sa_4_0,
    mit_license,
)

#: The digests that freeze the verbatim content of `_shared/licenses/*.txt`.
#:
#: MUST: a commit that updates these constants records that "**the original was deliberately
#: replaced**" — write the source in its body (whose distributed spelling, and why). A substring
#: check alone lets a rewrapping, a dropped clause or an appended text all pass green.
#:
#: On the MIT side the target is not the output of `mit_license()` but the bytes of the **raw
#: template** with `{copyright}` in it (this check's claim is that everything outside the
#: placeholder is verbatim).
APACHE_LICENSE_2_0_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
MIT_LICENSE_SHA256 = "c43f56ab1c4a9c8366b6843ba495421ed57950827d5e0d82bc3b9021b8bc0444"
#: The source is <https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt> (the plain-text
#: version CC itself distributes — transcribing the HTML version changes the wrapping and the
#: symbols with whoever does it).
CC_BY_SA_4_0_SHA256 = "28a9529c7d0bb4dc51f4bf5c116a3d16ef247a052f7591466768ddf563fd1cf5"

#: The section headings of Apache 2.0 (the 9 from `   1. ` to `   9. `). An aid for reading "what
#: changed" when the digest moves — unlike a line count, an addition and a deletion that cancel out
#: do not pass.
APACHE_SECTION_HEADING = re.compile(r"^   \d\. ", re.MULTILINE)

#: The section headings of CC BY-SA 4.0 (the 8 from `Section 1 -- ` to `Section 8 -- `). Same use
#: as above.
CC_BY_SA_SECTION_HEADING = re.compile(r"^Section \d -- ", re.MULTILINE)


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestApache:
    def test_it_is_the_license_text_and_not_a_summary(self) -> None:
        text = apache_license_2_0()
        assert "Apache License" in text
        assert "Version 2.0, January 2004" in text
        # §4(a) / §4(b) are where the requirement comes from, so check that the original
        # actually contains those clauses.
        assert "You must give any other recipients of the Work" in text

    def test_the_file_is_byte_for_byte_the_original(self) -> None:
        """MUST: the original is not reformatted — a rewrapping, a dropped clause and an
        addition all fail here."""
        assert _digest(APACHE_LICENSE_2_0_PATH) == APACHE_LICENSE_2_0_SHA256

    def test_the_nine_sections_and_the_line_count_are_intact(self) -> None:
        """An auxiliary check for reading "what changed" on the day the digest moves."""
        text = apache_license_2_0()

        assert len(APACHE_SECTION_HEADING.findall(text)) == 9
        assert "   9. Accepting Warranty or Additional Liability." in text
        assert text.count("\n") == 201


class TestCcBySa:
    def test_it_is_the_license_text_and_not_a_summary(self) -> None:
        text = cc_by_sa_4_0()
        assert text.startswith("Attribution-ShareAlike 4.0 International\n")
        # §3(a) (attribution and the modification notice) and §3(b) (ShareAlike) are where the
        # bundling requirement comes from.
        assert "b. indicate if You modified the Licensed Material and" in text
        assert "The Adapter's License You apply must be a Creative Commons" in text

    def test_the_file_is_byte_for_byte_the_original(self) -> None:
        """MUST: the original is not reformatted — a rewrapping, a dropped clause and an
        addition all fail here."""
        assert _digest(CC_BY_SA_4_0_PATH) == CC_BY_SA_4_0_SHA256

    def test_the_eight_sections_and_the_line_count_are_intact(self) -> None:
        """An auxiliary check for reading "what changed" on the day the digest moves."""
        text = cc_by_sa_4_0()

        assert len(CC_BY_SA_SECTION_HEADING.findall(text)) == 8
        assert "Section 8 -- Interpretation." in text
        assert text.count("\n") == 428


class TestMit:
    def test_it_substitutes_only_the_copyright_block(self) -> None:
        """The body is verbatim — putting the placeholder back gives the template."""
        template = MIT_LICENSE_PATH.read_text(encoding="utf-8")
        rendered = mit_license(("Copyright (c) 2026 Someone",))

        assert MIT_COPYRIGHT_PLACEHOLDER not in rendered
        assert rendered.replace("Copyright (c) 2026 Someone", MIT_COPYRIGHT_PLACEHOLDER) == template

    def test_it_keeps_several_copyright_holders_on_their_own_lines(self) -> None:
        """Redistributing a derivative cannot drop the upstream notice either, so there are 2 or
        more lines."""
        rendered = mit_license(("Copyright (c) 2026 Downstream", "Copyright (c) 2024 Upstream"))

        assert "Copyright (c) 2026 Downstream\nCopyright (c) 2024 Upstream" in rendered

    def test_the_template_is_byte_for_byte_the_original(self) -> None:
        """MUST: everything outside the placeholder is verbatim (the target is the raw template
        with `{copyright}` in it)."""
        assert _digest(MIT_LICENSE_PATH) == MIT_LICENSE_SHA256

    def test_it_refuses_to_render_without_a_copyright_holder(self) -> None:
        """An MIT text without copyright lines has nothing for "the above copyright notice" to
        point at."""
        with pytest.raises(ValueError, match="copyright lines are empty"):
            mit_license(())

    @pytest.mark.parametrize("copyright_lines", [("",), ("   ",), ("Copyright (c) 2026 X", "")])
    def test_it_refuses_a_blank_copyright_line(self, copyright_lines: tuple[str, ...]) -> None:
        """A copyright line that turned blank "names no rights holder" either — yet it still
        reads as valid prose."""
        with pytest.raises(ValueError, match="copyright lines are empty"):
            mit_license(copyright_lines)
