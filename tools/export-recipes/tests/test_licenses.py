"""配布リポ直下へ同梱するライセンス原文（`_shared.licenses`）。

原文は「逐語のコピー」であることが要件そのもの（Apache 2.0 §4(a) / MIT の許諾表示）なので、
ここで固定するのは**要約に化けていないこと**（条項が現物として居る）と、MIT の差し込み口が
著作権行**だけ**を動かすことの 2 点。文面の言い換えは値としては妥当な散文になってしまい、
配ってからでないと食い違いに気づけない。

NOTE: 「組み立てた配布形の `LICENSE.md` が原本とバイト同一か」は**家族側**の
`Test*LegalText` が見る（原本 → 配布形の経路はここを通らないので、この helper を自分自身と
突き合わせても恒真にしかならない）。
"""

from __future__ import annotations

import hashlib
import re

import pytest

from _shared.licenses import (
    APACHE_LICENSE_2_0_PATH,
    MIT_COPYRIGHT_PLACEHOLDER,
    MIT_LICENSE_PATH,
    apache_license_2_0,
    mit_license,
)

#: `_shared/licenses/*.txt` の逐語性を凍結する digest。
#:
#: MUST: この定数を更新するコミットは「**原本を意図的に差し替えた**」ことの記録になる —
#: 本文に出所（どの配布元の綴りを、なぜ）を書くこと。部分文字列の検査だけでは、折り返しの
#: 変更・条項の欠落・末尾への追記のいずれも緑を通る。
#:
#: MIT 側の対象は `mit_license()` の出力ではなく、`{copyright}` を含む**生テンプレート**の
#: バイト列（差し込み口より外は逐語でなければならない、というのがこの門の主張）。
APACHE_LICENSE_2_0_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
MIT_LICENSE_SHA256 = "c43f56ab1c4a9c8366b6843ba495421ed57950827d5e0d82bc3b9021b8bc0444"

#: Apache 2.0 の節見出し（`   1. ` 〜 `   9. ` の 9 個）。digest が動いたとき「何が変わったか」
#: を読める形にするための補助 — 行数と違い、追記と削除が相殺しても通らない。
APACHE_SECTION_HEADING = re.compile(r"^   \d\. ", re.MULTILINE)


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestApache:
    def test_it_is_the_license_text_and_not_a_summary(self) -> None:
        text = apache_license_2_0()
        assert "Apache License" in text
        assert "Version 2.0, January 2004" in text
        # §4(a) / §4(b) が要求の出どころなので、原文にその条項が居ることまで見る。
        assert "You must give any other recipients of the Work" in text

    def test_the_file_is_byte_for_byte_the_original(self) -> None:
        """MUST: 原文は整形しない — 折り返しの変更も条項の欠落も追記もここで落ちる。"""
        assert _digest(APACHE_LICENSE_2_0_PATH) == APACHE_LICENSE_2_0_SHA256

    def test_the_nine_sections_and_the_line_count_are_intact(self) -> None:
        """digest が動いた日に「何が変わったか」を読むための補助検査。"""
        text = apache_license_2_0()

        assert len(APACHE_SECTION_HEADING.findall(text)) == 9
        assert "   9. Accepting Warranty or Additional Liability." in text
        assert text.count("\n") == 201


class TestMit:
    def test_it_substitutes_only_the_copyright_block(self) -> None:
        """本文は逐語 — 差し込み口を戻せばテンプレートに一致する。"""
        template = MIT_LICENSE_PATH.read_text(encoding="utf-8")
        rendered = mit_license(("Copyright (c) 2026 Someone",))

        assert MIT_COPYRIGHT_PLACEHOLDER not in rendered
        assert rendered.replace("Copyright (c) 2026 Someone", MIT_COPYRIGHT_PLACEHOLDER) == template

    def test_it_keeps_several_copyright_holders_on_their_own_lines(self) -> None:
        """派生の再配布は上流の表示も落とせないので、行が 2 本以上になる。"""
        rendered = mit_license(("Copyright (c) 2026 Downstream", "Copyright (c) 2024 Upstream"))

        assert "Copyright (c) 2026 Downstream\nCopyright (c) 2024 Upstream" in rendered

    def test_the_template_is_byte_for_byte_the_original(self) -> None:
        """MUST: 差し込み口より外は逐語（対象は `{copyright}` を含む生テンプレート）。"""
        assert _digest(MIT_LICENSE_PATH) == MIT_LICENSE_SHA256

    def test_it_refuses_to_render_without_a_copyright_holder(self) -> None:
        """著作権行の無い MIT は「上記の著作権表示」が指す先を持たない。"""
        with pytest.raises(ValueError, match="著作権行が空"):
            mit_license(())

    @pytest.mark.parametrize("copyright_lines", [("",), ("   ",), ("Copyright (c) 2026 X", "")])
    def test_it_refuses_a_blank_copyright_line(self, copyright_lines: tuple[str, ...]) -> None:
        """空行に化けた著作権行も「権利者を名乗っていない」— 散文としては成立してしまう。"""
        with pytest.raises(ValueError, match="著作権行が空"):
            mit_license(copyright_lines)
