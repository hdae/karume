# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-08-15

## Now

- **整理整頓波（2026-08-14〜）**: 外部レビュー 6 本（ADR 棚卸し / docs 事実相違 / 汎用性監査 /
  構造・ライセンス / 予定整理 / 量子化資料批評）の TRIAGE 消化 → docs 事実修正・ADR 注記・
  planning SoT 再編（backlog.md 新設）まで完了。残り = **exporter 構造再編（案 A — PyPI core と
  `tools/export-recipes/` の分離）** + housekeeping 一括（backlog now 節）。
- 次の大波 = **autoregressive-ready 基盤波**（backlog next 節）。主語は「Gemma 対応」ではなく
  IR / loader / state 実行モデルの器づくり — ADR 5 本を実装より先行させる。検収モデル =
  Gemma 4 E2B / MiniCPM5-1B。

## Open decisions

- exporter 再編の段階実施中の個別裁定（family 移動順・共有 utility の置き場）— 都度。

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
- **`deno task verify` は `.claude/worktrees/` が存在すると worktree 側まで test を拾う**
  （現在は撤去済み・再設置で再発 — exclude 裁定は backlog now）。
- **Session 構築の重みアップロード後 submit 1 回は瞬間ピーク +2.7GiB を抑えている** — 消さない。
- **資産の置き場**: `models/` = HF へそのまま上げる配布形のみ・系列出力は `outputs/series/`・
  入力素材は `inputs/<ファミリ>/<名前>/` — 綴りの正本は `karume/paths.py` と
  [assets-layout](../docs/assets-layout.md)。格納 dtype はヘッダが正（dist の門が検査）。
  旧識別子以前の資産は開けない（互換シム無し）。turbo LoRA だけ未移行（backlog now）。
- **birefnet / depth / vowel の series 資産は未再生成** = 当該 e2e 門 SKIP 中（backlog now）。
  DeBERTa 実重み e2e も未移植 — 回帰網は WAV sha256 門 + rel-pos parity（backlog now）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提。JSR npm 互換層の
  `sideEffects: false` 出力は未検証（backlog release）。
