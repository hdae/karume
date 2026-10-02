"""Wan2.1（text-to-video）の export recipe（ADR 0118 / ADR 0065 決定 2 — wheel 外・リポ専用）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: 上流の取得元の pin
（{@link wan.sources}）・diffusers の参照パイプライン（{@link wan.pipeline_ref}）・DiT と
動画 VAE のパッチと export 台本・タイルの参照・テキスト埋め込みの生成・配布 recipe。依存方向は
**recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py` が機械で
守る）。

MUST: **再輸出しない**（`from wan import X` で family の中身が芋づるに import される形に
しない）。台本はどれも重い上流 import を持つので、`import wan` が diffusers を引き込む形に
なると `wan.sources` の pin を読むだけの経路まで巻き添えになる。呼び出し側はサブモジュールを
名指しで import すること。

台本の起動は export-recipes ルートから（例: `uv run --group wan python -m wan.sources --fetch`）。
"""
