# Karume — プロジェクト規約（1画面）

WebGPU 汎用 NN 推論スタックの monorepo（Deno + ブラウザ両対応・純 TS + WGSL・ランタイム
依存ゼロ）。JSR `@karume/runtime`（IR 実行）/ `@karume/hub`（HF revision resolve・DL・cache・
model / quant 解決）/ `@karume/models`（パイプライン・tokenizer）+ PyPI `karume`（tools/exporter —
torch.export → IR v1・uv 管理）。設計の正本は [docs/decisions/](docs/decisions/)。現在の
焦点と落とし穴は [.claude/ACTIVE_DESIGN.md](.claude/ACTIVE_DESIGN.md)
（レビュー・計画タスクは必読）。

## 用語

- 「プロトタイプ」= 参考元の先行実験プロジェクト。**固有名は成果物（コード・コメント・
  コミットメッセージ含む）に書かない**。

## 言語

- **README は全て英語で書く** — リポ直下 / `models/` / `tools/exporter/`（PyPI パッケージの
  ドキュメント）/ 将来の `packages/*/`、およびモデルカード生成物（`karume dist` が書く
  `README.md`）。コード内コメント・docstring・`docs/`・ACTIVE_DESIGN・コミットメッセージの
  説明部は従来どおり日本語。
- 例外: コードが実際に投げるエラー文言の逐語引用と、日本語 TTS のデモ入力は日本語のまま
  （引用であって散文ではない — 訳すと現物と食い違う）。

## 構成

- `packages/runtime/` — 公開 API は mod.ts の**薄い面のみ**（ADR 0008）。`src/` が本体
  （gpu / codegen / kernels / format / ops / reference / runtime）
- `packages/hub/` — manifest 解決・fetch・cache（仕様の正本は ADR 0038）
- `packages/models/` — **barrel（mod.ts）+ ファミリ別サブパス export の両建て**
- `tools/exporter/` — PyPI `karume` = **汎用 core のみ**（src layout・境界は machine gate —
  ADR 0065）/ `tools/export-recipes/` — モデル別 recipe（wheel 外・uv workspace 共有 venv・
  起動は `python -m <family>.<mod>`・dist は `export-recipes/dist.py`）。`examples/` は
  README 整備予定（実装は 4 ファミリ済み・`deno task verify` の対象）

## 検証コマンド（変更後は全て）

- `deno task verify`（= fmt --check + lint + check + test）。GPU テストは実 GPU で実行、
  アダプタ無し環境は明示 SKIP（リリース判定は緑必須 — ADR 0005）
- exporter: `uv run pytest`（**tools/exporter と tools/export-recipes の両方で** — ADR 0065）
- git worktree は**リポ外**（例 `~/workspace/karume-wt-<名前>`）に作る — リポ内に置くと
  fmt / lint / test が worktree 側のファイルを誤拾いする（deno.json に exclude は設けない —
  2026-08-16 裁定）

## 横断の不変条件

- **全モジュール副作用ゼロ**（top-level 登録・グローバル可変状態・import 時実行の禁止）—
  barrel 経由 tree-shaking の成立条件 MUST
- codegen 決定性: 同一キー → バイト単位同一 WGSL（スナップショットで固定）
- elementwise / 行 reduce は grid-stride 前提。requiredLimits は compute 系まで明示 +
  `pushErrorScope('validation')` 常設
- 未対応・想定外は fail loudly（黙って近似しない）。シンボル束縛は `Object.hasOwn`
- flush-before-destroy。バッファプールの不変条件は ADR 0004
- ランタイム依存は Web 標準 API のみ（依存パッケージも Web 標準 API のみで構成されたものに
  限る — 例 `@hdae/fetch-cache`。ADR 0038）。相対 import は `.ts` 付き。WGSL はテンプレート
  リテラル。TypedArray は `<ArrayBuffer>` 明示

## docs の置き場（what-goes-where）

- `docs/decisions/` — ADR（MADR-lite）。インライン `DECIDED:` の指し先
- `docs/backlog.md` — **波順・作業項目・状態の正本**（now / next / later / release / parked）
- `docs/ir-v1.md` — IR フォーマット仕様 / `docs/op-vocabulary.md` — op 語彙台帳
- `docs/assets-layout.md` — ローカル資産 3 根（models / outputs / inputs — 全て git 追跡外）の規約
- `docs/research/` — 調査・実測記録（**時点スナップショット** — 冒頭に性格を示す注記 1 行）
- `docs/perf-ledger.md` — 性能候補の起票・採否・kill 基準の台帳（数値の正本は research・
  全体の波順は backlog）
- `docs/known-issues.md` / `docs/limitations.md` — 未解決バグ / by-design 制約
