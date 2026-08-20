# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-08-20

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
- **波 I（w4 横展開 + 方式スクリーニング・2026-08-19）: 聴感/視認込みで完全クローズ** —
  実測正本 =
  [research/2026-08-19-w4-method-screening.md](../docs/research/2026-08-19-w4-method-screening.md)・
  採否 = perf-ledger Q 節（RTN/NF4 全ファミリ一次通過・Q-4 mxfp4 不採用確定・Q-2 kmeans は
  LLM/埋め込み限定）。方式序列はモデル系統で割れる — 1 回の実測で一般化しない。既定化は
  速度と細かい品質のバランスで別途。
- **波 J（量子化探索・第 2 段・2026-08-20 着手）が現行**: J-1（Q-1 = SBV2 BERT linear i4 の
  配布配線 + 実 GPU WAV 門）は**消化済み** — deberta-i4 混成系列 + quant `w8-bert4`（既定は
  w8 のまま・perf-ledger Q-1 ✅）。J-2（GPTQ/AWQ 校正付き丸め）・J-3（g 軸）・J-4（格納席の
  実装裁定）は波割り裁定待ち（骨子 = backlog now 節）。

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
