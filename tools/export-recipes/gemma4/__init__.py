"""Gemma 4 E2B（causal LM・1-shot 形）の export recipe（ADR 0065 決定 2 — wheel 外・リポ専用）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: export 台本 4 本
（1-shot の {@link gemma4.export}・states 形 decode の {@link gemma4.export_decode}・
token-only 出口の {@link gemma4.export_token}・PLE 外出し + 最終行 logits 出口の
{@link gemma4.export_product}）、トークナイザ資産の compile
（{@link gemma4.tokenizer} — ADR 0084 決定 1。グラフには触らず「文字列 ⇄ id 列」だけを扱う）と、
chat フォーマットの parity フィクスチャ採取（{@link gemma4.chat} — ADR 0084 決定 5 / 決定 7）。
独立したパッチ層は持たない — 上流のモデリングコードは差し替えず、transformers の公開拡張点
（`AttentionInterface.register`）へ MQA を保つ attention 実装を 1 本足すだけで済む
（台本内に同居 — `gemma4.export.gqa_sdpa_attention`）。
配布 recipe（{@link gemma4.distribution} / {@link gemma4.card}）も持つ: 製品コンテナと PLE
sidecar・トークナイザ資産を 1 リポ `karume-gemma4` へ畳む（ADR 0092 決定 1 — 家族 1 リポ）。

台本と配布側が共有する裏方は 3 本: PLE 1 枚表の 35 分割（{@link gemma4.ple}）、golden を
採らない系列の出所記録（{@link gemma4.provenance}）、ホスト生成 RoPE の式
（{@link gemma4.rope} — ADR 0091）。どれも「同じ規律を 2 箇所に書かない」ための分離で、
読み手の本数はモジュールごとに違う（`rope` は配布側も読む）。
MUST: `rope` は torch を import しない（配布側の import を重くしないため — 当のモジュールが
理由ごと自己申告している。これは MUST の再掲であって検査ではない）。トークナイザ側の同じ役は
`_shared/gemma_tokenizer.py`（EmbeddingGemma と共有 — ADR 0084 決定 6）。

依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

MUST: **再輸出しない**（`from gemma4 import X` で transformers が芋づるに入る形にしない）。
呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例:
`uv run --with 'transformers==5.14.1' python -m gemma4.export`）。
"""
