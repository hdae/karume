# 0042: 実行計画は 2 相分離し、導出済みレシピを bindings キーで Session 常駐する

- Status: accepted（PNG sha256 門 4 本 + WAV 門が不変・fixture 0 diff・verify 749/0）
- Date: 2026-08-10
- 関連: ADR [0004](0004-execution-model.md)（アリーナ不変条件・flush-before-destroy — 本 ADR の
  実行相はこの簿記を 1 箇所へ引き継ぐ）/ [0040](0040-fusion-pass.md)（融合の純関数パスと常設
  カウンタ — ヒット run の報告義務は §3 を継承）/ [0021](0021-gpu-timing-diagnostics.md)
  （常設診断の流儀）/ [0022](0022-gemm-register-blocking.md)（f32 実行時オートチューン禁止 —
  本キャッシュは変種選択を一切変えない）
- recon と実測: [research/2026-08-10-op-timing-stats.md](../research/2026-08-10-op-timing-stats.md)
  §7〜§8・先行設計 = [research/2026-08-06-kernel-triage/large-designs.md](../research/2026-08-06-kernel-triage/large-designs.md) C/D/E

## Context

HOST-006 第 1 波（params 内容アドレスキャッシュ + layout 保持）後も、定常 run のホスト露出は
≈1.1s ≈ 壁の 11%（w8a8-1024）残った。内訳は createBindGroup 44.4ms/step・plan/fusion の毎 run
再計算・WGSL 文字列の毎 dispatch 再生成・params キー文字列・encode ループの JS 残差。

recon（3 レッグ・2026-08-10）の中核事実:

- 計画（bindSymbols → planGraph → planFusions）とレシピ導出（params 内容・パイプラインキー・
  workgroups・束縛構成）は **graph と解決済み bindings だけの純関数**。graph は Session 構築時に
  固定され、run 経路に外部状態への参照は無い。
- 計画の生成物は GPUBuffer 参照を一切含まない（名前 → バッファは env 経由で解決）。
- 入力 shape は bindSymbols の 2 巡目で bindings との全一致が強制される — つまり
  **解決済み bindings がキャッシュキーとして必要十分**。

## Decision

### 1. エンコード層は「導出相 / 実行相」の 2 相（`src/runtime/recipe.ts`）

導出相は GPU コマンドを出さず run 寿命の状態（RunArena・env）に触れない。成果物は
StepRecipe 列 — dispatch ごとに pipeline / layout / params（いずれも Session 常駐の直参照）と
束縛の**手順**（値名 / ノード内一時 id）、ステップごとに出力確保仕様・一時の寿命
（dispatch 境界の添字・同一境界は確保の逆順で解放）・解放する値名の延べ列を持つ。
実行相（executeStepRecipe）は簿記 1 本でこれを再生する（ADR 0004 の不変条件を引き継ぐ）。

MUST: レシピは GPUBindGroup と run 寿命バッファを持たない。持てるのは Session 常駐の実体と
「どの位置に何を束ねるか」だけ（これが破れると本 ADR §2 のキャッシュが成立しない）。

### 2. 導出済み計画（PreparedPlan）は解決済み bindings をキーに Session 常駐（LRU 4）

- キー = graph.symbols 宣言順の bindings 値の連結。シンボル無しグラフはキー "" の 1 本。
- 器は SessionState（モジュールスコープ禁止 — 副作用ゼロ不変条件）。持つのは後段が実際に
  読む 3 欄のみ（shapes / recipes / fusions）。
- **登録は導出相の完走後のみ**。途中 throw した run の部分レシピは載せない（載せると次の
  同一 bindings run が欠けたステップ列を沈黙実行する）。
- ヒット run は planGraph / planFusions / レシピ導出を丸ごと飛ばす。**bindSymbols
  （入力 shape 検証）は毎 run 走らせる**。契約検査を飛ばせる根拠は「キーが解決済み bindings の
  完全一致なら、同じ入力に対する同じ検査の再実行を省くだけ」— fail loudly は緩まない。
- LRU 上限 4 は定数（設定ノブにしない）。追い出しはホストオブジェクトのみ（GPU 資源は
  paramsCache / PipelineCache が所有 — ここで destroy すると別計画の直参照が破棄済みを掴む）。

### 3. 常設診断 `lastRunPrepared {hit, cachedPlans}`

キャッシュが外れても値は正しいまま性能だけ静かに戻る — ここが唯一の観測点（lastRunFusions と
同格）。ヒット run でも lastRunFusions はキャッシュ済み counts を報告する（ADR 0040 §3）。
lastRunParams はヒット run で {0,0} になる（導出相が走らない事実の報告 — 値の意味は不変）。

## Consequences

- **WGSL バイト突合の被覆縮小（承認済みトレードオフ）**: ヒット run は WGSL を再生成しないため
  PipelineCache の「同一キー → バイト同一」常設突合はミス経路（初回導出）でのみ走る。決定性の
  担保は codegen スナップショット（fixture 109 本）+ 初回突合に寄せる。
- **失敗 run の discard 統合カバレッジ縮小**: dispatch 発行が実行相に一本化された結果、
  「エンコード途中で落ちて pending が残る」を外から誘発する経路が消えた（導出相の throw は
  1 件も積む前に落ちる）。SubmitScheduler.discard 単体テストは残存。
- params キャッシュの再利用観測は「同一 bindings の 2 run 目」から「同一 run 内の重複 +
  prepared 追い出し後の再導出 run」へ移設（gpu_params_cache_test — 2 キャッシュの寿命独立性を
  固定）。
- 初回 run はパイプライン生成が最初の submit より前に直列化するため、cold Session の 1 run 目
  だけ僅かに伸びうる（2 run 目以降は無関係）。
- 波 2（transient slot 固定 + bind group のレシピ焼き込み）は recipes に欄を足す形で載る。
  ヒット run の残存ホスト費用は createBindGroup とアリーナ簿記に一本化された。

## 実測（検収 ABBA・回文・冷却規約・PNG 門込み・2026-08-10）

`d7626c8`（波前）vs `b2e6ce0`（段 A+B）、e2e_anima 全 4 本 = 1 走・各 2 走:

- f16-1024: 19.9/19.9 → 19.9/20.0s、w8a8-1024: 10.4/10.3 → 10.4/10.5s — **壁時計は中立**
  （差はノイズ幅 ±0.1s の内側）。PNG 門は 4 走とも緑。
- **帰属の確定（見積りの訂正）**: 露出 gap ≈1.1s のうち、導出相（plan/fusion 走査・WGSL
  再生成・params キー文字列・契約検査）の寄与は**壁に出ない水準**だった — これらは GPU 実行と
  ほぼ完全に重畳していた。露出費用の本体は createBindGroup（44.4ms/step）・アリーナ簿記
  （allocStorage/retain/release ×~3,280/run）・転送系に絞られた。**波 2（transient slot 固定 +
  bind group のレシピ焼き込み）が壁時計の本丸**であり、本 ADR の 2 相分離とキャッシュは
  その前提条件（バッファ同一性を固定できる形）として機能する。
