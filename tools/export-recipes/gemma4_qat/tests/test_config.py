"""QAT の小さな正本（`gemma4_qat.config`）が、torch を避けるために持つ写しの定数。

`config.py` は dist から torch を import しないために、通常 Gemma の製品形が持つ値を
写しで綴る。写しは正本を動かした日に片方だけ古びるので、同値をここで縛る。
"""

from __future__ import annotations

from gemma4 import export_product
from gemma4_qat.config import MAX_SELECTED_ROWS


class TestCopiedConstants:
    def test_the_selected_rows_ceiling_follows_the_product_form(self) -> None:
        """trace の記号 R の上限は、製品形の `ROW_SYM_MAX`（sliding ring の余裕 + 1）と同じ。

        食い違うと、QAT の trace 上限と配布形が宣言する `slidingSlack` がずれる（余裕を動かした
        日に QAT 側だけ古い上限で焼かれる）。
        """
        assert MAX_SELECTED_ROWS == export_product.ROW_SYM_MAX
