# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-08-19

## Now

- **0.3.0 リリース済み**（2026-08-16・JSR 3 + PyPI・CI 緑）。ポジショニングの正本 =
  [research/2026-08-16-runtime-landscape.md](../docs/research/2026-08-16-runtime-landscape.md)。
- **autoregressive-ready 実装波 A〜H: 全消化（2026-08-17〜19）** — GQA 整除 broadcast・
  多出力 + argmax / topk・GenerationContext（第 5 寿命クラス）・states 形 attention +
  state_append・decode 台本 + greedy 検収・w4（i4 g32・linear 限定）・shard ロード +
  admission estimator・Gemma 4 E2B 検収（PLE 35 分割・KV 共有 30 slot・混成 i8+i4）・
  **token-only 既定出口**。設計の正本 = ADR
  [0066](../docs/decisions/0066-generation-context-state-slots.md)〜
  [0070](../docs/decisions/0070-shard-loading-admission.md)（各追記）、波の経緯と送り =
  [backlog](../docs/backlog.md) now 節。検収 = MiniCPM5-1B / Gemma 4 E2B とも実 GPU で
  greedy K=16 厳密一致（ring エビクト越え込み）・w4 混成の完全常駐（hostExpandedBytes 0）・
  chat デモ ~11 tok/s。
- **全体レビュー（2026-08-19）は修正波込みで全消化** — E/C = 0・W 19 → 修正 11 コミット・
  verify 1620/0/5。Codex 提案の波割りは backlog へ反映済み（R1 同席 4 件・生成 API 波・
  recipe 基盤同席）。
- **w4 横展開 + 量子化方式スクリーニング波（2026-08-19）: 実装・実測は全消化** — 実測正本 =
  [research/2026-08-19-w4-method-screening.md](../docs/research/2026-08-19-w4-method-screening.md)・
  採否台帳 = perf-ledger Q-1〜Q-5。要旨: 方式序列はモデル系統で割れる（LLM/埋め込み =
  kmeans:shared 最良 / TTS = nf4 相対最良 / 画像 = 方式差消失の全滅帯）・SBV2 BERT linear i4 =
  配布 −9.76%（既存格納形・聴感待ち）。recipe 基盤 4 件（CX-1.4/1.1/1.3/2.3）も同波で消化。
  **聴感/視認も消化（同日ユーザー実施）**: RTN/NF4 全ファミリ一次通過・**Q-4（mxfp4）不採用
  確定・Q-5 は配布席候補へ改訂**（正本 = research §6 / perf-ledger Q 節）。既定化は速度と
  細かい品質のバランスで別途。次波 = 量子化探索・第 2 段（**冒頭に Q-1 実装** = SBV2 BERT
  linear i4 配布形 + 実 GPU 検収 — backlog next 節）。

## Open decisions

- MiniMax-H3（動画生成・オープンウェイト 33.1B/42.5GB 級）は遠期の関心として記録のみ —
  ブラウザ実行はメモリ規模的に現行スコープ外（レビュー DS-4）。

## Pitfalls（現役のみ）

- **Metal**: threadgroup `vec4` への動的インデックス書きは黙って捨てられる（`gemm.ts` の
  `storeBTransposed` の switch 展開を新しい箇所で崩さない）。attention i8a8 / conv2d の
  Metal 数値差は known-issues・Metal は gpuTiming 不可（limitations）。
- **融合 matcher は実測形 exact-match** — exporter の発行順・形が変わると黙って外れ、値は
  正しいまま性能だけ落ちる。観測 = `Diagnostics.lastRunFusions` +
  `assets_fusion_counts_test.ts`。**row-block だけは外れ方が性能でなく資源** — 128MiB 級
  device で resource-limit failure に戻る（**分解経路の matcher だけの話** — 保存 attention の
  states 形は行ブロックを op 内蔵で持つ・ADR 0067 決定 7・波 D 済）。
- **RoPE / SiLU 融合の丸め障壁（workgroup memory 往復）は実測依存** — バックエンド更新で
  PNG 門が割れたらまずここを疑う。
- **`deno task verify` はリポ内に worktree を置くと worktree 側まで test を拾う** — worktree は
  リポ外に作る（CLAUDE.md 検証コマンド節。deno.json に exclude は設けない — 2026-08-16 裁定）。
- **Session 構築の重みアップロード後 submit 1 回は瞬間ピーク +2.7GiB を抑えている** — 消さない。
- **資産の置き場**: `models/` = HF へそのまま上げる配布形のみ・系列出力は `outputs/series/`・
  入力素材は `inputs/<ファミリ>/<名前>/` — 綴りの正本は
  `tools/export-recipes/_shared/paths.py` と [assets-layout](../docs/assets-layout.md)。
  格納 dtype はヘッダが正（dist の門が検査）。旧識別子以前の資産は開けない（互換シム無し）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提。JSR npm 互換層の
  `sideEffects: false` 出力は未検証（backlog release）。
