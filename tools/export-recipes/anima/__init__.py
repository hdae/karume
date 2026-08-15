"""Anima（text-to-image）の export recipe（ADR 0065 決定 2 — wheel 外・リポ専用）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: 上流モデル由来のパッチ層
（{@link anima.patch}）・トークナイザの参照実装（{@link anima.text}）・参照 pipeline /
タイリング / rope 表の台本・quant 計測・配布 recipe（{@link anima.distribution}）と
カードテンプレート（{@link anima.card}）。依存方向は **recipe → core の一方向だけ**
（`tools/exporter/tests/test_architecture_boundary.py` が機械で守る）。

MUST: **再輸出しない**（`from anima import X` で family の中身が芋づるに import される形に
しない）。台本はどれも重い上流 import を持つので、`import anima` が diffusers /
transformers を引き込む形になると `anima.resolution` を触るだけの経路まで巻き添えになる。
呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例: `uv run python -m anima.export --dtype f16`）。
"""
