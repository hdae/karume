"""SBV2（Style-Bert-VITS2 JP-Extra / text-to-speech）の export recipe（ADR 0065 決定 2）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: 上流モデル由来のパッチ層
（{@link sbv2.patch}）・export 台本・ホスト資産とデモ（{@link sbv2.demo}）・quant 計測・
配布 recipe（{@link sbv2.distribution}）とカードテンプレート（{@link sbv2.card}）。
依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

配布形の `text_encoder` は DeBERTa 系列（`deberta` recipe が書く）の成果物を **path で**
拾うだけで、コードの import は持たない（shared 席は資産の共有であって結合ではない）。

MUST: **再輸出しない**（`from sbv2 import X` で family の中身が芋づるに import される形に
しない）。台本はどれも重い上流 import を持つので、`import sbv2` が style_bert_vits2 /
transformers を引き込む形になると、配布 recipe を触るだけの経路まで巻き添えになる。
呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例: `uv run python -m sbv2.export --dtype i8`）。
"""
