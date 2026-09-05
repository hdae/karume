"""配布リポ直下へ同梱するライセンス原文（`karume.dist.Pipeline.root_files` の `LICENSE.md`）。

ADR [0092](../../../../docs/decisions/0092-distribution-repos-and-sources.md) 決定 7 —
Apache-2.0 は「全文 + §4(b) の改変告知」、MIT は「全文 + 著作権行」を配布リポ直下へ置く。
原文は family ごとに違わないので、家族の recipe が各自 1 部ずつ持つ形にはしない
（家族が増えるたびに逐語コピーが増え、どれが原本か言えなくなる）。

MUST: **原文は整形しない**。`.txt` の中身をバイトのまま渡す — 整形・改行幅の調整・
見出しの Markdown 化をした瞬間に「このライセンスのコピー」ではなくなる。MIT だけは
著作権行が権利者ごとに違うので、その 1 ブロックだけを差し込む
（{@link mit_license}）。差し込み口以外の本文には触らない。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

#: 原文の置き場（`Path(__file__)` 基準 — cwd にも系列の置き場にも依存しない）。
LICENSES_DIR = Path(__file__).parent

APACHE_LICENSE_2_0_PATH = LICENSES_DIR / "apache_license_2_0.txt"

#: MIT の本文テンプレート。著作権行のブロックだけが差し込み口
#: （{@link MIT_COPYRIGHT_PLACEHOLDER}）で、残りは逐語。
MIT_LICENSE_PATH = LICENSES_DIR / "mit.txt"

#: 差し込み口の綴り。MIT 本文に波括弧は 1 つも現れないので、置換で取り違えようがない。
MIT_COPYRIGHT_PLACEHOLDER = "{copyright}"


def apache_license_2_0() -> str:
    """Apache License 2.0 の原文（逐語）。"""
    return APACHE_LICENSE_2_0_PATH.read_text(encoding="utf-8")


def mit_license(copyright_lines: Sequence[str]) -> str:
    """MIT の原文に著作権行を差し込む。

    `copyright_lines` は `Copyright (c) <年> <権利者>` の行。fine-tune の再配布のように
    権利者が複数居るときは、**配布物に近い順**（その版の著作権者 → 継承した上流）に並べる
    — MIT の派生物で慣例の並びで、先頭がこのリポの中身の権利者になる。

    MUST: 空では組まない — 著作権行の無い MIT は「上記の著作権表示」が指す先を持たず、
    §「著作権表示と許諾表示を含めること」を満たさない。散文としては成立してしまうので、
    ここが唯一の検出器になる。
    """
    if not copyright_lines or any(not line.strip() for line in copyright_lines):
        raise ValueError("MIT の著作権行が空 — 権利者を名乗らないライセンス文は組まない")
    template = MIT_LICENSE_PATH.read_text(encoding="utf-8")
    return template.replace(MIT_COPYRIGHT_PLACEHOLDER, "\n".join(copyright_lines))
