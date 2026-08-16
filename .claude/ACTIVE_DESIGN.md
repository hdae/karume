# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-08-16

## Now

- **全体レビュー波（2026-08-16）消化済み**: sop-review 網羅 + Codex 独立レンズ + 反証 7 レッグ
  → blocker 0 → 修正 35 項目 + onEvent コールバック（anima/irodori — 観測の副作用ゼロを
  sha 門で常設証明）を 7 コミットで消化。**残り = 4 パッケージ lockstep 0.2.2 → 0.3.0 の
  bump とリリース（裁定済み）**。勢力図とポジショニング検証（フットプリント 56〜93 倍・
  i64 は「境界 1 箇所集約」が正確な主張）=
  [research/2026-08-16-runtime-landscape.md](../docs/research/2026-08-16-runtime-landscape.md)。
- 次の大波 = **autoregressive-ready 基盤波**（backlog next 節）。ADR 5 本を実装より先行・
  着手前に sop-research で参照資料収集。R2（shape 不変条件 = 静的物理格納 + 有界論理 extent の
  席予約）/ R3（KV は名前付き state スロット）/ G3（GQA kv_heads>1 — ADR 0023 reopen）を
  ADR ①⑤ へ編入済み（backlog が正本）。検収 = Gemma 4 E2B / MiniCPM5-1B（M3 は非目標裁定）。

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
  device で resource-limit failure に戻る（正面解決は backlog next の attention ADR）。
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
