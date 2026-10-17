# Karume — プロジェクト規約（1画面）

WebGPU 汎用 NN 推論スタックの monorepo（開発は Node 24 + Dawn・主対象はブラウザ・Deno は副次〈ADR 0127〉・
純 TS + WGSL・ランタイム依存ゼロ）。JSR `@karume/runtime`（IR 実行）/ `@karume/hub`（HF revision resolve・
DL・cache・model / quant 解決）/ `@karume/models`（パイプライン・tokenizer）+ PyPI `karume`（tools/exporter —
torch.export → コンテナ `krm`〈IR v2〉・uv 管理）。pnpm workspace + Vite+（`vp`）。設計の正本は
[docs/decisions/](docs/decisions/)。現在の焦点と落とし穴は [.claude/ACTIVE_DESIGN.md](.claude/ACTIVE_DESIGN.md)
（レビュー・計画タスクは必読）。

## 用語

- 「プロトタイプ」= 参考元の先行実験プロジェクト。**固有名は成果物（コード・コメント・
  コミットメッセージ含む）に書かない**。
- 対話の日本語の用語は [docs/GLOSSARY.ja.md](docs/GLOSSARY.ja.md)（意味から決める・直訳しない）。

## 言語（移行中 — ADR 0130〈proposed〉）

- **新しく書くもの（コード・コメント・エラー文言・コミットメッセージ・docs・ADR）は英語**。
  既存の日本語（コメント・docs・ADR・テストの文言）はフェーズ 2 で整理しながら英訳する —
  移植中は同じファイルに両言語が混ざってよい。対話は日本語。
- README・モデルカード生成物は従来どおり英語。例外: コードが実際に投げるエラー文言の逐語引用と、
  日本語 TTS のデモ入力は日本語のまま（引用であって散文ではない）。
- **日本語で対話するときは [docs/GLOSSARY.ja.md](docs/GLOSSARY.ja.md)（対話用の用語表）を先に読み、
  その語で話す**（英語で対話するときは不要）。

## 構成

- `packages/runtime/` — 公開 API は mod.ts の**薄い面のみ**（ADR 0008）。`src/` が本体
  （gpu / codegen / kernels / format / ops / reference / runtime / tune）。Node から使うときは
  `acquireGpu({ gpu })` に WebGPU 実装（Dawn）を渡す
- `packages/hub/` — manifest 解決・fetch・cache（仕様の正本は ADR 0038 §4 + 0041 +
  0075〈quant の表示名〉+ 0109〈manifest は `karume/5`〉+ 0080〈取得層〉+ 0086〈取得元抽象〉）。
  ランタイム固有のサブパスは `hub/node`（`nodeDirectory`）と `hub/deno`（`denoDirectory`）だけ
- `packages/models/` — **barrel（mod.ts）+ ファミリ別サブパス export の両建て**
- `packages/e2e/`（私的）— 実重みの e2e（ローカル資産 + GPU）。家族ごとのレーンでリリース前に回す（ADR 0129）
- `packages/test-support/`（私的）— テストのハーネス（Dawn の注入・fs helper・
  レーン表・移植の codemod）。移植の規則は同ディレクトリの README
- `tools/exporter/` — PyPI `karume` = **汎用 core のみ**（src layout・境界は machine gate —
  ADR 0065）/ `tools/export-recipes/` — モデル別 recipe（wheel 外・uv workspace 共有 venv・
  起動は `python -m <family>.<mod>`・dist は `export-recipes/dist.py`）。
  `tools/{lint,published-smoke,verify-diff,release}` はプロセス用の道具（workspace package）。
  計測の道具と CLI の例（0.14.0 の `tools/*` / `examples/*`）は削除済みで、ブラウザアプリとして
  作り直す（ADR 0127）

## 検証コマンド（変更後は全て）

- `vp check`（= fmt〈Oxfmt〉+ lint〈Oxlint・型認識〉+ 型検査〈TypeScript 7〉）と `vp test`
  （全レーン = フル verify）。GPU テストは実 GPU（Dawn）で実行、アダプタ無し環境は明示 SKIP
  （リリース判定は緑必須 — ADR 0005）
- **レーン実行**: 変更が 1 系列に閉じるならその系列だけ回す（`vp test --project models:<系列>`）。
  コア（runtime / hub / 共通層）の大きめ変更・横断変更・リリース前はフル `vp test`。
  コアだけなら `vp test --project core`（門番 3 本を含む）。レーンの表は
  `packages/test-support/src/lanes.ts`、被覆漏れは `verify_lanes_test.ts` が落とす（ADR 0129）
- **上流との一致**（torch / diffusers の golden と許容差で比べるケース）は opt-in: `KARUME_UPSTREAM_PARITY=1`。
  既定のレーンでは明示 SKIP。一致は実装時に取り、以後の変化は sha256 の回帰チェックが捕まえる（裁定 2026-10-17・
  フラグは `packages/e2e/tests/helpers/upstream-parity.ts`）
- **実ネットワークを使うケース**（huggingface.co から小さい資産を取る `node_cache_storage_test.ts` の 1 本）も
  opt-in: `KARUME_NETWORK=1`。既定のレーンでは明示 SKIP
- **既定で走らせないファミリー**（minicpm5 / siglip2 — 保守はするが毎回は走らせない・レーン表の `LOW_PRIORITY_FAMILIES`）:
  `KARUME_ALL_FAMILIES=1 vp test --project models:siglip2`。フル verify とリリース判定は既定の集合で回す
- 1 ファイルなら `vp test <path>`。GPU を使う走行は他セッションと `flock /tmp/karume-gpu.lock` で
  直列化する。GPU を使わずに回すときは `KARUME_NO_DAWN=1`（CI と同じ構成 — 全 SKIP ガードは
  `KARUME_ALLOW_NO_GPU=1` 等で明示的に開ける）
- **sha256 参照値は環境ごとの行**（ADR 0106）: 環境キーは Node では `node-<アダプタの device>`
  （例 `node-nvidia-geforce-rtx-3080-ti`）。行が無い機はそのケースが明示 SKIP + 参照門が赤。
  `KARUME_REFERENCE=write vp test --project <レーン>` で自分の機の行を作る（既存の行は触らない・
  焼き直しは `rewrite`）。結果と実物は `outputs/verify/<環境キー>/<日付>_<系列>/`（消して安全）。
  **sha 門の tolerance 化は禁止**（緩めた時点で移植の門の意味が消える）
- exporter: `uv run pytest`（**tools/exporter と tools/export-recipes の両方で** — ADR 0065）
- `models/` の配布形ミラーが無い機では `KARUME_ALLOW_NO_DISTRIBUTION=1` で門番を外せる
  （配布形を根にする実資産 e2e と融合ヒット数の検査は走らない）
- git worktree は**リポ外**（例 `~/workspace/karume-wt-<名前>`）に作る — リポ内に置くと
  lint / test が worktree 側のファイルを誤拾いする（アプリが作る `.claude/worktrees/` も同じ理由で使わない）

## 横断の不変条件

- **全モジュール副作用ゼロ**（top-level 登録・グローバル可変状態・import 時実行の禁止）—
  barrel 経由 tree-shaking の成立条件 MUST（門: `hub_barrel_surface_test`）
- WebGPU グローバル（`GPUBufferUsage` 等）はモジュールスコープで読まない — runtime は
  `src/gpu/webgpu-constants.ts` 経由・裸参照禁止 MUST（import 安全性。Node にこれらのグローバルは
  無く、テストのハーネスが Dawn の写しを入れているだけ）
- codegen 決定性: 同一キー → バイト単位同一 WGSL（スナップショットで固定）
- elementwise / 行 reduce は grid-stride 前提。requiredLimits は compute 系まで明示 +
  `pushErrorScope('validation')` 常設
- 未対応・想定外は fail loudly（黙って近似しない）。シンボル束縛は `Object.hasOwn`
- flush-before-destroy。バッファプールの不変条件は ADR 0004
- ランタイム依存は Web 標準 API のみ（依存パッケージも Web 標準 API のみで構成されたものに
  限る — 例 `@hdae/fetch-cache`。ADR 0002・門は `dependency_graph_test`。`node:` を読めるのは
  `hub/node.ts` だけ）。相対 import は `.ts` 付き。WGSL はテンプレート
  リテラル。TypedArray は `<ArrayBuffer>` 明示（lint 規則 `karume/typed-array-buffer` —
  `tools/lint/oxlint-plugin.ts`・例外は ADR 0002 の例外表）
- 上流との一致は数値（またはその意図）の水準で取り、上流のバグは写さない。上流に問題があれば回避策を
  オプションで用意する（優れていれば既定に）。一致が非現実的なら別の方法で正しさを示す（ADR 0005 追記 2026-10-11）

## docs の置き場（what-goes-where）

- `docs/decisions/` — ADR（MADR-lite）。インライン `DECIDED:` の指し先
- `docs/backlog.md` — **波順・作業項目・状態の正本**（now / next / later / release / parked）
- `docs/ir-v2.md` — IR フォーマット仕様 / `docs/op-vocabulary.md` — op 語彙台帳
- `docs/glossary.md` — 用語 → 意味の要約 → 正本へのポインタ（**意味の正本は参照先** — ここは索引。
  docs に新しい用語を入れたら 1 行足す）/ `docs/GLOSSARY.ja.md` — 対話で使う日本語の用語
- `docs/quantization.md` — 量子化方式の索引（格納型 / 実行ノブ / quant プリセット / exporter 側の方式と
  その状態。数値と意味論は複製せず正本を指す）
- `docs/assets-layout.md` — ローカル資産 3 根（models / outputs / inputs — 全て git 追跡外）の規約
- `docs/release-runbook.md` — リリース・HF 公開の恒久手順（pin 焼き込み・断片化対策・publish 順序）
- `docs/research/` — 調査・実測記録（**時点スナップショット** — 冒頭に性格を示す注記 1 行）
- `docs/perf-ledger.md` — 性能候補の起票・採否・kill 基準の台帳（数値の正本は research・
  全体の波順は backlog）
- `docs/known-issues.md` / `docs/limitations.md` — 未解決バグ / by-design 制約
