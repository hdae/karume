# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
>
> Last updated: 2026-08-05

## Active redesigns (in flight)

- **旧リポからの分離移植（P0〜P5・経緯と裁定は ADR 0037）**: P0 scaffold 完了。
  P1 = runtime 移植（`packages/runtime`）+ docs/exporter 移設（並列レッグ委任）。
  P1.5 = IR 識別子の改名（旧識別子 → `karume_ir` / `karume-ir`）+ fixture 再生成（メイン）。
  P2 = hub（manifest v1 の ADR 起票 + 実装 — variant 解決表が本体）。
  P3 = models/anima（`AnimaPipeline` 再編。**移植の門 = PNG sha256 が旧デモと完全一致**。
  大型資産の再エミット含む）。P4 = exporter CLI 化（PyPI `karume`・サブコマンド式）。
  P5 = HF 実網通し + 公開準備（ユーザーの HF リポ作成が前提）。

## Pitfalls

- **IR 識別子改名後、旧資産は開けない**（互換シム無し — 未リリース原則）。fixture・資産は
  exporter で再生成する。`models/` は untracked（履歴 bundle にも入っていない）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提 —
  崩れると barrel 経由の shake が静かに死ぬ。
- JSR npm 互換層が package.json に `sideEffects: false` を出すかは**未検証**（要実測 —
  無くても ESM 副作用解析でおおむね shake されるが、フラグ有りが確実）。
- モデル e2e（anima / sbv2 / deberta 実重み系）は P3 まで旧リポ側に残る —
  P1〜P2 の間、新リポのカバレッジはランタイム核のみ（意図的な過渡状態）。
