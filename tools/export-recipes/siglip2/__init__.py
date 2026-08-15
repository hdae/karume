"""SigLIP2 vision tower（image-feature-extraction）の export recipe（ADR 0065 決定 2）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: 上流モデル由来のパッチ層
（{@link siglip2.patch}）・export 台本・前処理パリティの参照実装（{@link siglip2.preprocess}）・
配布 recipe（{@link siglip2.distribution}）とカードテンプレート（{@link siglip2.card}）。
依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

MUST: **再輸出しない**（`from siglip2 import X` で family の中身が芋づるに import される形に
しない）。台本は重い上流 import を持つので、`import siglip2` が transformers を引き込む形に
なると、配布 recipe を触るだけの経路まで巻き添えになる。呼び出し側はサブモジュールを名指しで
import すること。

台本の起動は export-recipes ルートから（例: `uv run --group siglip2 python -m siglip2.export`）。
"""
