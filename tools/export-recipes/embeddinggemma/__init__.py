"""EmbeddingGemma-300m（文埋め込み）の export recipe（ADR 0065 決定 2 — wheel 外・リポ専用）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: export 台本
（{@link embeddinggemma.export}）1 本と、トークナイザ資産の compile
（{@link embeddinggemma.tokenizer} — gemma4 と同じ台本系で回る。実装は共用・**資産は別**）。
パッチ層は持たない（上流のモデリングコードを差し替えずに export できる）し、配布 recipe も
持たない — この系列は golden io と性能計測のための資産で、HF へ配る形は今のところ無い。

依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

MUST: **再輸出しない**（`from embeddinggemma import X` で transformers が芋づるに入る形に
しない）。呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例:
`uv run --with 'transformers==5.14.1' python -m embeddinggemma.export`）。
"""
