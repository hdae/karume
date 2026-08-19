"""Gemma 4 E2B（causal LM・1-shot 形）の export recipe（ADR 0065 決定 2 — wheel 外・リポ専用）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: export 台本 3 本
（1-shot の {@link gemma4.export}・states 形 decode の {@link gemma4.export_decode}・
token-only 出口の {@link gemma4.export_token}）。独立したパッチ層は持たない — 上流のモデリング
コードは差し替えず、transformers の公開拡張点（`AttentionInterface.register`）へ MQA を保つ
attention 実装を 1 本足すだけで済む（台本内に同居 — `gemma4.export.gqa_sdpa_attention`）。
配布 recipe も持たない: この系列は 層種別 2 種の帯マスクと**混成量子化**（embedding i8 ×
linear i4 — ADR 0069）を実重みで検収するための資産で、HF へ配る形は今のところ無い。

台本 3 本が共有する裏方が 2 つ: PLE 1 枚表の 35 分割（{@link gemma4.ple}）と、golden を採らない
系列の出所記録（{@link gemma4.provenance}）。どちらも「同じ規律を 2 箇所に書かない」ための
分離で、台本を跨いで効く不変条件を持つものだけがここに増える。

依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

MUST: **再輸出しない**（`from gemma4 import X` で transformers が芋づるに入る形にしない）。
呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例:
`uv run --with 'transformers==5.14.1' python -m gemma4.export`）。
"""
