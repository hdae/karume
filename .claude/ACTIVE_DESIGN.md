# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
>
> Last updated: 2026-08-05

## Active redesigns (in flight)

- **立ち上げロードマップ（ADR 0037）**: P0 scaffold・P1 runtime 移植・P1.5 IR 識別子確定
  （`karume_ir` / `karume-ir`）+ golden 再生成・**P2 hub — 完了**（manifest v1 = ADR 0038
  〈pre-mortem 44 指摘反映・記録は research/2026-08-05-manifest-premortem.md〉+ `@karume/hub`
  実装 429 テスト緑・取得層は `@hdae/fetch-cache`）。
  次 = **P3 models/anima**: `AnimaPipeline` 再編（barrel + サブパス・副作用ゼロ）。
  門 = 生成 PNG sha256 の参照一致。S 形 + タイル VAE 資産の再エミット（新識別子）と
  モデル e2e の復帰を含む。以降: **P4 exporter CLI 化**（PyPI `karume`・サブコマンド式・
  manifest 自動生成 + モデルカード README・HF アップ可能ディレクトリを直接出力）→
  **P5 HF 実網通し + 公開準備**（gated リポの Authorization 実網確認・Cache Storage 数 GB
  quota・sideEffects: false の実測もここ）。

## Pitfalls

- **現行識別子（`karume_ir` / `karume-ir`）以前に焼かれた資産は開けない**（互換シム無し —
  fail loudly）。`models/` の大型資産は P3 でエクスポータから再エミットするまで使えない。
  `models/` は untracked。
- モデル e2e（anima / sbv2 / deberta の実重み系）は P3 まで不在 — カバレッジはランタイム核
  のみ（意図的な過渡状態）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提 —
  崩れると barrel 経由の shake が静かに死ぬ。
- JSR npm 互換層が package.json に `sideEffects: false` を出すかは**未検証**（P2〜P3 で実測）。
- tokenizer パリティ資産 `anima-text/parity.json` は暫定で `packages/runtime/tests/fixtures/`
  に置いた（exporter のテストが参照）— 最終の置き場は P3 で models 側へ見直し。
