# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
>
> Last updated: 2026-08-05

## Active redesigns (in flight)

- **立ち上げロードマップ（ADR 0037）**: P0〜P2 完了（runtime 移植 / IR 識別子確定 / hub =
  ADR 0038）。**P3 models/anima — 完了**: `AnimaPipeline`（fromPretrained / fromAssets・
  段ごと Session・S 形 + 常時タイル）+ 共通 image 層 + `models/anima/` 配布形（dist.py・
  karume.json 実 hash・**格納 dtype 門**付き）。**移植の門 = PNG sha256 ビット一致 ×4 が全緑**
  （ローカル / 取得層 + integrity / example CLI の 3 経路とも参照値と同一）。example は 90 行
  1 画面（移行元 1,111 行から縮退）。
  次 = **P4 exporter CLI 化**: PyPI `karume` のサブコマンド式（export / dist / verify を包む）・
  モデルカード README 自動生成（variant 表は manifest から導出）。同時裁定・回収する宿題:
  ①rope_base の読み（models 側の parseSafetensors 再実装 130 行 — runtime 公開面に載せるか
  IR コンテナ化か）②`AnimaFromPretrainedOptions` に `caches` 注入席（テスト用）③波 1 積み残しの
  参照フィクスチャ系テスト（timestepsProj atol 突合等 — anima-pipeline 系列の再エミットが前提）
  ④tokenizer parity fixture の models 側への移設。
  以降: **P5 HF 実網通し + 公開準備**（gated リポの Authorization 実網確認・Cache Storage
  数 GB quota・sideEffects: false の実測もここ）。

## Pitfalls

- **現行識別子（`karume_ir` / `karume-ir`）以前に焼かれた資産は開けない**（互換シム無し —
  fail loudly）。`models/` の大型資産は P3 でエクスポータから再エミットするまで使えない。
  `models/` は untracked。
- モデル e2e は anima の PNG 門 4 本が本リポに常駐（`models/anima/` 資産が前提・無ければ明示
  SKIP）。deberta / sbv2 の実重み e2e は移行元リポに残置のまま（SBV2 の取り込み時に復帰）。
- **配布資産の格納形は series ディレクトリ名でなくヘッダが正** — dist.py の格納 dtype 門が
  組み立て時に検査する（`--dtype` 付け忘れの素 F32 が PNG 門まで沈黙した実測事故が根拠）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提 —
  崩れると barrel 経由の shake が静かに死ぬ。
- JSR npm 互換層が package.json に `sideEffects: false` を出すかは**未検証**（P2〜P3 で実測）。
- tokenizer パリティ資産 `anima-text/parity.json` は暫定で `packages/runtime/tests/fixtures/`
  に置いた（exporter のテストが参照）— 最終の置き場は P3 で models 側へ見直し。
