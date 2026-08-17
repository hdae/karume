"""MiniCPM5-1B（causal LM・1-shot 形）の export recipe（ADR 0065 決定 2 — wheel 外・リポ専用）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: export 台本
（{@link minicpm5.export}）1 本。独立したパッチ層は持たない — 上流のモデリングコードは
差し替えず、transformers の公開拡張点（`AttentionInterface.register`）へ GQA を保つ
attention 実装を 1 本足すだけで済む（台本内に同居 — `minicpm5.export.gqa_sdpa_attention`）。
配布 recipe も持たない: この系列は ADR 0067 決定 1（GQA 整除 broadcast）を実重みで検収する
ための資産で、HF へ配る形は今のところ無い。

依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

MUST: **再輸出しない**（`from minicpm5 import X` で transformers が芋づるに入る形にしない）。
呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例:
`uv run --with 'transformers==5.14.1' python -m minicpm5.export`）。
"""
