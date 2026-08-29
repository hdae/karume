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

## 波 2 の追記（同日）: transient slot 固定と bind group 焼き込み

### 4. transient slot の GPU backing（容量 1・初ヒットで遅延構築）

`derivePlanSlots` がレシピ列から **RunArena のサイズクラス LIFO を仮想再生**して slot 表を導く
（独自パッキング禁止 — footprint 一致がテストで固定され、常駐化しても VRAM の新ピークは
生まれない）。backing（slot + 入力バッファ + 焼き込み bind group）は**初ヒット run で遅延構築**
し、容量は活性 1 signature のみ（DiT で ~1GiB 規模のため本数倍にしない）。単発 run は slot
メモリを一切払わない。破棄は退役キュー + flush 後の 1 箇所（切替 / LRU 追い出し / dispose
相乗り）。新規構築した run が失敗したら backing を退役（一過性 OOM から次ヒットで回復・
既存 backing は据え置き = スラッシング回避）。

### 5. bind group 焼き込みと入力固定

束縛先が全て run 跨ぎで固定（resident / slot / 常駐入力）なので、構築時に全 dispatch の
GPUBindGroup を焼き込み、backed run は dispatch を積むだけ。入力は backing 所有バッファへ
毎 run writeBuffer — 追い越し安全性は「#chain 直列化 + run は flush/readback 完了後にしか
返らない + run 内では入力書き込みが全エンコードに先行」で論証（並行 run テストで固定）。
残骸上書きの正しさは full-write（ADR 0014）1 本。診断 `planBacking {residentBytes, buildCount}`
（signature 交互切替による毎 run 再構築が唯一の沈黙劣化 — buildCount が観測点）。

## 実測（検収 ABBA・回文・冷却規約・PNG 門込み・2026-08-10）

段ごとにコミットを切り替えた ABBA（e2e_anima 全 4 本 = 1 走・各側 2 走・代表値 min）:

| 区間                                                           | w8a8-1024                                | f16-1024                       | 判定     |
| -------------------------------------------------------------- | ---------------------------------------- | ------------------------------ | -------- |
| `d7626c8` → `b2e6ce0`（波 1 = 2 相分離 + 計画キャッシュ）      | 10.3-10.4 → 10.4-10.5s                   | 19.9 → 19.9-20.0s              | **中立** |
| `b2e6ce0` → `339fc0c`（段 C = slot 固定）                      | 10.4-10.5 → **10.1/10.1s（×1.03-1.04）** | 20.0 → **19.6/19.6s（×1.02）** | **利得** |
| `339fc0c` → `1751f3c`（段 D = bind group 焼き込み + 入力固定） | 10.1 → 10.1s（4 走同値）                 | 19.6 → 19.6s                   | **中立** |

PNG 門は全 12 走緑。**帰属の確定（2 度の見積り訂正）**: 露出していたホスト費用は
アリーナ簿記 + createBuffer/destroy（段 C が削除）だけで、導出相（波 1）も
createBindGroup 44.4ms/step（段 D）も **GPU 実行と完全に重畳しており壁に出ていなかった**。
段 D の価値は将来の dispatch 数増加への耐性と、E 系（prepared 値の常駐）の土台に限られる。

**staged execution（large-designs D/E）への含意**: dispatch 削減系の候補（E = 392/predict・
timestep-only stage = 772/predict）が狙っていたホスト費用は重畳側にあり、GPU 側の利得も
E ≈ 0.60% / timestep ≈ 0.12% しかない。この環境では**壁時計の利得はほぼゼロ**と結論できる
（採否の裁定は research 側の記録とともにユーザーへ）。

## 追記（2026-08-18・波 D — 焼き込み単位の分離と backing 世代識別子）

ADR [0066](0066-generation-context-state-slots.md) 決定 5 の実装（`22b5f64`）で backing 節が
次の 2 点の改訂を受けた（本文の機構は不変 — generation を伴わない run は 1 バイトも変わらない）:

1. **焼き込みの単位は束ねる相手の所有者で分ける**: `bakeBindGroups` が焼くのは Session 所有の
   実体だけを束ねる dispatch で、GenerationContext 所有の実体（state スロット・論理長 uniform）を
   束ねる dispatch は **context 側**（`bakeGenerationBindGroups`）が (context, backing 実体) の
   組ごとに焼く。判別は `bindsGeneration` の 1 本・歩きは `bakeGroups` の 1 本を両側で共有。
2. **ActiveBacking は世代識別子（`build`）を持つ**: 同じ計画鍵で作り直した backing も必ず別値に
   なる単調採番。context 側の束はこの識別子で照合してから使う MUST — **退役した backing の
   バッファは当該 run の後始末（flush 後の destroy）まで生存する**ため、照合を欠いた古い束は
   validation を通ったまま**値だけが静かに変わる**（波 D-4 の故障注入で実測: 出力全要素不一致・
   例外ゼロ）。診断は `stateBacking.rebindCount`（run 数に比例しないことが分離の観測点）。

## 追記（2026-08-29）— PipelineCache の所有者変更

決定 2 の「GPU 資源は paramsCache / PipelineCache が所有」のうち PipelineCache は
**GpuContext 所有（device 寿命・device 1 個につき 1 本）**へ移した。同一 device 上の Session が
キャッシュを共有し、`PipelineKeyConflictError`（同一キー・異 WGSL の即死）は Session を跨いで
効くようになる。paramsCache は従来どおり Session 常駐。診断は `pipelineCount` = その Session の
**使用**キー本数 / `devicePipelineCount` = device 合計、の 2 席に分離（外部レビュー消化波 —
経緯は backlog 2026-08-29）。
