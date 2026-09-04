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

import pytest

from _shared.licenses import (
    MIT_COPYRIGHT_PLACEHOLDER,
    MIT_LICENSE_PATH,
    apache_license_2_0,
    mit_license,
)


class TestApache:
    def test_it_is_the_license_text_and_not_a_summary(self) -> None:
        text = apache_license_2_0()
        assert "Apache License" in text
        assert "Version 2.0, January 2004" in text
        # §4(a) / §4(b) が要求の出どころなので、原文にその条項が居ることまで見る。
        assert "You must give any other recipients of the Work" in text


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

    def test_it_refuses_to_render_without_a_copyright_holder(self) -> None:
        """著作権行の無い MIT は「上記の著作権表示」が指す先を持たない。"""
        with pytest.raises(ValueError, match="著作権行が空"):
            mit_license(())
