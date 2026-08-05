# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
>
> Last updated: 2026-08-05

## Active redesigns (in flight)

- **立ち上げロードマップ（ADR 0037）**: P0 scaffold・P1 runtime 移植（404 テスト緑・publish
  dry-run 緑）・P1.5 IR 識別子確定（`karume_ir` / `karume-ir`）+ golden 再生成 — **完了**。
  次 = **P2 hub**: 配布 manifest v1 の ADR 起票 + 実装（variant 解決表 =「quantization ノブ →
  グラフ/重みファイル選択 + 実行ノブ」が本体。HF アップロード習慣準拠のリポレイアウト
  〈モデルカード README.md の YAML frontmatter 込み〉の実地確認を含む）。
  以降: **P3 models/anima**（`AnimaPipeline` 再編。門 = 生成 PNG sha256 の参照一致。
  大型資産の再エミット込み）→ **P4 exporter CLI 化**（PyPI `karume`・サブコマンド式・
  HF アップ可能ディレクトリを直接出力）→ **P5 HF 実網通し + 公開準備**。

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
