# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
>
> Last updated: 2026-08-05

## Active redesigns (in flight)

- **立ち上げロードマップ（ADR 0037）**: P0〜P2 完了（runtime 移植 / IR 識別子確定 / hub =
  ADR 0038）。**P3 models/anima — 完了**: `AnimaPipeline`（fromPretrained / fromAssets・
  段ごと Session・S 形 + 常時タイル）+ 共通 image 層 + `models/anima-turbo/` 配布形（dist.py・
  karume.json 実 hash・**格納 dtype 門**付き）。**移植の門 = PNG sha256 ビット一致 ×4 が全緑**
  （ローカル / 取得層 + integrity / example CLI の 3 経路とも参照値と同一）。example は 90 行
  1 画面（移行元 1,111 行から縮退）。
  **P4 進行中**: 配布形を **anima-turbo** へ改名、`karume` サブコマンド CLI + モデルカード
  README 自動生成（**英語** — `karume.modelcard`・数値は全て manifest 導出）を実装済み。
  rope_base の読みは裁定 a で解消（runtime 公開面に `parseSafetensors` / `tensorBytes` —
  ADR 0008 追記）。`AnimaPipeline` は `using` 対応（`[Symbol.dispose]`）。
  base_model_relation = quantized・license 3 値継承はユーザー裁定で確定。宿題②は解消済み
  （`AnimaFromPretrainedOptions` に `caches` / `fetch` 注入席 + hub 公開面に `clearHubCache` —
  ADR 0038 追記）。残る宿題: ③波 1 積み残しの参照フィクスチャ系テスト（timestepsProj atol
  突合等 — anima-pipeline 系列の再エミットが前提）④tokenizer parity fixture の models 側への
  移設。
  以降: **P5 HF 実網通し + 公開準備**（gated リポの Authorization 実網確認・Cache Storage
  数 GB quota・sideEffects: false 実測・使い方スニペットの実リポ ID 化〈`--repo`〉・
  公開 README 群〈リポ直下 + packages/*〉の英語起草もここ）。

## Pitfalls

- **Metal（Apple GPU）は WGSL の受け取り方が Vulkan と違う** — threadgroup の `vec4` へ
  **動的インデックスで成分を書く**（`sb[i][wsl] = v`）と `wsl != 0` が黙って捨てられる。
  GEMM の B タイル充填は静的成分への `switch` 展開で回避済み（`gemm.ts` の
  `storeBTransposed`）で、**同じ形を新しく書かないこと**。残る誤値（attention i8a8 /
  conv2d の 2 経路一致）は known-issues.md、**性能**（Linux の 31〜41 倍 — 帯域は健全なのに
  GEMM のタイリングが 1.21x しか効かない = Apple GPU 向け未最適化）は
  [research/2026-08-06](../docs/research/2026-08-06-metal-silent-miscompute.md) §3。
  Metal では `gpuTiming` が使えない（dispatch 数がサンプル上限を超える）。

- **`models/` に置くのは HF へそのまま上げられる配布形だけ**（1 ディレクトリ = 1 HF リポ）。
  エクスポータの系列出力は `outputs/series/` — 綴りの正本は `karume/paths.py`
  （`DIST_ROOT` / `SERIES_ROOT`）で、台本と `karume dist` がそこを共有する。**ADR と
  docs/research 内の `models/anima-*` 表記は当時の記録**（時点スナップショットなので直さない）。
- **現行識別子（`karume_ir` / `karume-ir`）以前に焼かれた資産は開けない**（互換シム無し —
  fail loudly）。`models/` と `outputs/` はどちらも untracked。
- モデル e2e は anima の PNG 門 4 本が本リポに常駐（`models/anima-turbo/` 資産が前提・無ければ明示
  SKIP）。deberta / sbv2 の実重み e2e は移行元リポに残置のまま（SBV2 の取り込み時に復帰）。
- 入力素材（SBV2 の実重み ckpt・turbo LoRA ファイル）の置き場は**未裁定** — 現状の綴りは
  `models/` のままで、SBV2 取り込み時に配布形と分ける必要がある。
- **配布資産の格納形は series ディレクトリ名でなくヘッダが正** — dist.py の格納 dtype 門が
  組み立て時に検査する（`--dtype` 付け忘れの素 F32 が PNG 門まで沈黙した実測事故が根拠）。
- 配布形の宣言外ファイル検査は直下の `karume.json` / `README.md` だけを例外にする（それ以外が
  混ざると `verify_dist` が止まる — 前回残骸・`io.*` の混入を後段に見せない）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提 —
  崩れると barrel 経由の shake が静かに死ぬ。
- JSR npm 互換層が package.json に `sideEffects: false` を出すかは**未検証**（P2〜P3 で実測）。
- tokenizer パリティ資産 `anima-text/parity.json` は暫定で `packages/runtime/tests/fixtures/`
  に置いた（exporter のテストが参照）— 最終の置き場は P3 で models 側へ見直し。
