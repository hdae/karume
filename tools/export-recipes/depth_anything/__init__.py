"""Depth Anything V2（depth-estimation）の export recipe（ADR 0065 決定 2 — wheel 外・リポ専用）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: 上流モデル由来のパッチ層
（{@link depth_anything.patch}）・export 台本・配布 recipe
（{@link depth_anything.distribution}）とカードテンプレート（{@link depth_anything.card}）。
依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

パッケージ名がハイフンでなく `depth_anything` なのは Python の識別子制約 — pipeline 名
（`--pipeline depth-anything`）と系列名・リポ名の綴りは従来どおりハイフンで、そちらは
{@link depth_anything.distribution} が持つ。

MUST: **再輸出しない**（`from depth_anything import X` で family の中身が芋づるに import
される形にしない）。台本は重い上流 import を持つので、`import depth_anything` が transformers を
引き込む形になると、配布 recipe を触るだけの経路まで巻き添えになる。呼び出し側はサブモジュールを
名指しで import すること。

台本の起動は export-recipes ルートから（例:
`uv run --group depth-anything python -m depth_anything.export`）。
"""
