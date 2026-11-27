"""License texts bundled at the root of a distribution repository (the `LICENSE.md` of
`karume.dist.Pipeline.root_files`).

SPEC: docs/spec/distribution-cards.md#licensemd

ADR [0092](../../../../docs/decisions/0092-distribution-repos-and-sources.md) decision 7 —
Apache-2.0 puts "the full text + the §4(b) modification notice", MIT "the full text + the
copyright lines" and CC BY-SA 4.0 "the legal code + the §3(a) attribution and modification
notice" at the root of the distribution repository.
The texts do not differ per family, so each family's recipe does not keep its own copy
(every new family would add another verbatim copy, and no one could say which is the original).

MUST: **the texts are never reformatted**. The contents of the `.txt` files are passed byte for
byte — the moment they are reformatted, rewrapped or given Markdown headings they are no longer
"a copy of this license". Only MIT's copyright lines differ per rights holder, so that one block
alone is inserted ({@link mit_license}). The text outside the insertion point is not touched. In
`LICENSE.md` every text sits verbatim inside a fence ({@link license_markdown}), so that Markdown
does not read its indentation, numbering and underlines as formatting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Where the texts live (relative to `Path(__file__)` — independent of the cwd and of where
#: series live).
LICENSES_DIR = Path(__file__).parent

APACHE_LICENSE_2_0_PATH = LICENSES_DIR / "apache_license_2_0.txt"

#: The CC BY-SA 4.0 legal code (the plain-text version of the creativecommons.org legalcode,
#: verbatim). §3(a)(1)(C) asks for "the legal code itself or its URI" — the URI alone would do,
#: but the full text is stored, as with Apache, so it travels with the distribution repository
#: even when it is copied off HF.
CC_BY_SA_4_0_PATH = LICENSES_DIR / "cc_by_sa_4_0.txt"

#: The MIT text template. Only the block of copyright lines is an insertion point
#: ({@link MIT_COPYRIGHT_PLACEHOLDER}); the rest is verbatim.
MIT_LICENSE_PATH = LICENSES_DIR / "mit.txt"

#: The spelling of the insertion point. The MIT text has no braces at all, so the replacement
#: cannot hit the wrong place.
MIT_COPYRIGHT_PLACEHOLDER = "{copyright}"


def apache_license_2_0() -> str:
    """The text of the Apache License 2.0 (verbatim)."""
    return APACHE_LICENSE_2_0_PATH.read_text(encoding="utf-8")


def cc_by_sa_4_0() -> str:
    """The legal code of Creative Commons Attribution-ShareAlike 4.0 International (verbatim)."""
    return CC_BY_SA_4_0_PATH.read_text(encoding="utf-8")


def mit_license(copyright_lines: Sequence[str]) -> str:
    """Insert the copyright lines into the MIT text.

    `copyright_lines` are `Copyright (c) <year> <holder>` lines. When there are several holders,
    as in a redistributed fine-tune, they are ordered **closest to the distributed work first**
    (that version's copyright holder → the inherited upstream) — the usual order for MIT
    derivatives, so the first line names the holder of this repository's contents.

    MUST: never build it empty — an MIT text without copyright lines has nothing for "the above
    copyright notice" to point at and fails the requirement to "include the copyright notice and
    the permission notice". It still reads as valid prose, so this is the only detector.
    """
    if not copyright_lines or any(not line.strip() for line in copyright_lines):
        raise ValueError(
            "the MIT copyright lines are empty — a license text that names no rights holder is"
            " not built"
        )
    template = MIT_LICENSE_PATH.read_text(encoding="utf-8")
    return template.replace(MIT_COPYRIGHT_PLACEHOLDER, "\n".join(copyright_lines))


#: The fence a license text sits in (a plain-text code block: Markdown would otherwise read a
#: text's indentation, numbering and underlines as lists, code and headings).
FENCE_OPEN = "```text\n"
FENCE_CLOSE = "```\n"


def fenced(text: str) -> str:
    """A license text inside a ```` ```text ```` fence, byte for byte.

    A text without a final line break gets one, so that the closing fence stands on its own line
    (the line break ends the text's last line; it is not part of the text).

    MUST: a text holding three backticks in a row fails — the fence would close inside it, and the
    rest of the license would be read as Markdown.
    """
    if "```" in text:
        raise ValueError("the license text holds ``` — it cannot be fenced verbatim")
    return FENCE_OPEN + text + ("" if text.endswith("\n") else "\n") + FENCE_CLOSE


@dataclass(frozen=True)
class FurtherLicense:
    """A license text of `LICENSE.md` after the primary one: what it is and which components it
    covers (its heading `## <license> — <scope>`), a paragraph saying so, and the text."""

    license: str
    scope: str
    paragraph: str
    text: str


# SPEC: docs/spec/distribution-cards.md#licensemd
def license_markdown(primary: str, further: Sequence[FurtherLicense] = ()) -> str:
    """`LICENSE.md`: the primary license text fenced from the first byte, then each further text
    after a `---` line, its heading and its paragraph, fenced the same way.

    MUST: no text is touched ({@link fenced}) — the module doc's MUST.
    """
    return fenced(primary) + "".join(
        f"\n---\n\n## {section.license} — {section.scope}\n\n{section.paragraph}\n\n"
        + fenced(section.text)
        for section in further
    )
