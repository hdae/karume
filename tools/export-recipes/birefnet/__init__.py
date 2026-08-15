"""BiRefNet 系（image-segmentation — BiRefNet_HR / Lucida）の export recipe（ADR 0065 決定 2）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: 上流モデル由来のパッチ層
（{@link birefnet.patch}）・export 台本・配布 recipe（{@link birefnet.distribution}）と
カードテンプレート（{@link birefnet.card}）。依存方向は **recipe → core の一方向だけ**
（`tools/exporter/tests/test_architecture_boundary.py` が機械で守る）。

Lucida は独立 recipe にしない — 構造は BiRefNet_HR と同一で、違うのは重みの出所（と、それに
紐づく帰属・リポ名）だけなので、`--model` の 1 軸として同じ表が持つ。

MUST: **再輸出しない**（`from birefnet import X` で family の中身が芋づるに import される形に
しない）。台本は重い上流 import を持つので、`import birefnet` が transformers / timm / kornia を
引き込む形になると、配布 recipe を触るだけの経路まで巻き添えになる。呼び出し側はサブモジュールを
名指しで import すること。

台本の起動は export-recipes ルートから（例: `uv run --group birefnet python -m birefnet.export`）。
"""
