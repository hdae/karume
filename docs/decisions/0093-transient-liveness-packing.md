# 0093: 中間バッファ（transient）の静的 liveness パッキング — サイズ別プールから区間 + offset 配置へ

- Status: implemented（2026-09-05 — 決定 1〜8 を runtime へ結線: `transient-plan.ts` の計画を
  `recipe.ts` の焼き込み / ミス run・`executor.ts` の backing・`estimate.ts` の見積りが共有し、
  `RunArena` のプールと `derivePlanSlots` は退役。dispatch の読み書きは `PipelineCache` が WGSL 宣言
  から採る。裁定は網羅レビュー 2026-09-04_62bdbeb の要判断 2 をユーザーが推奨案 a
  〈A 代数書き換え → B 本 ADR → C 上限 preflight〉で下したもの。実測の正本 =
  `.claude/reviews/2026-09-04_62bdbeb/B1-design-draft.md` / `findings/module-B1-birefnet-2048.md`）
- Date: 2026-09-05
- 関連: ADR [0004](0004-execution-model.md)（バッファ管理「per-run アリーナ（サイズ別プール +
  最終消費者解放）」— **本 ADR が置換**。不変条件 ①②③④⑥ は据え置き）/
  [0014](0014-layout-ops-full-write.md)（full-write — 据え置き・本 ADR の再利用安全性の根拠）/
  [0054](0054-resident-loop-and-fence.md)（フェンスと寿命 — 不変）/
  [0066](0066-generation-context-state-slots.md)（state スロットは context 所有 — 本 ADR の対象外）/
  [0070](0070-shard-loading-admission.md) 決定 5・[0089](0089-memory-limits-preflight.md)
  （見積りと上限検査 — 決定 4 / 5 が中間へ拡張される）/
  [0022](0022-gemm-register-blocking.md)（実行時オートチューン禁止 — 計画は純関数のまま）

## Context

中間バッファ（ノード出力とノード内一時）の確保は **サイズクラス厳密一致の LIFO プール**
（`RunArena.allocStorage` と、その鏡写しである `derivePlanSlots` / `estimate.transientSlotBytes`）で
行われている。サイズが揃わない列では解放済みが再利用されず slot が累積するため、**必要な
GPU メモリが同時生存バイトの最大（生存ピーク）の約 3.2 倍に膨らむ**。BiRefNet HR の実 IR で
計算した値（2026-09-04・`estimateGraphMemory` と純関数の再生）:

| 解像度 | workspace（現行プール） | 生存ピーク |  膨張 |
| ------ | ----------------------: | ---------: | ----: |
| 1024²  |                6.14 GiB |   1.88 GiB | 3.27× |
| 2048²  |               23.76 GiB |   7.50 GiB | 3.17× |

2048² は 12 GiB 級 GPU でグラフの 5.4% 地点で総確保が天井に当たる（limitations「Deno では
GPUBuffer の総確保がドライバ申告予算の 97% で頭打ち」）。同じ膨張率は他家族にも掛かっており、
1024² の BiRefNet が「デスクトップ級 GPU 限定」なのはこの規則の帰結である。

厳密一致の根拠は「要求より大きいバッファを配ると runtime-sized array の `arrayLength()` が
静かに変わる」（`arena.ts`）だったが、①生成 WGSL に `arrayLength` は 1 度も現れず（要素数は
params の `n` で渡す）②bind group entry は `{ buffer, offset, size }` で**束縛範囲を実寸に切れる**
ので、実寸の保証は「バッファの大きさ」ではなく「束縛範囲」の性質として保てる。

## Decision

### 1. 計画 = 区間 + offset（純関数・実行時オートチューン無し）

- レシピ列（`StepRecipe[]`）から、確保・解放の**イベント順**をそのまま再生して各論理テンソル
  （ノード出力・一時）の**生存区間** `[alloc, release)` と実寸（`toSizeClass`）を取り、
  時間区間が重ならないテンソル同士が同じバイトを共有できるように **first-fit（大きい順）で
  領域バッファの offset へ配置**する。順序は確保 → retain → 定義ぶんの解放が現行と同一
  （`executeStepRecipe` / `derivePlanSlots` の MUST の順）。別名（reshape / 恒等 expand）は根の
  区間へ合算、pinned（グラフ出力）は末尾まで生存。
- 領域は `maxBufferSize` を bin にして複数本になりうる。offset は `minStorageBufferOffsetAlignment`
  （granted 値・仕様既定 256 B）に整列する。
- **usage scope の制約**（WebGPU: 1 dispatch の中で同じ GPUBuffer を `read`（read-only-storage）
  と `read_write`（storage）の両方で束ねることはできない — 判定はバッファ単位で範囲は見ない）を
  計画に組み込む: 同じ dispatch で片方が読まれ他方が書かれる 2 テンソルは**別の領域**に置く。
  dispatch ごとの読み / 書きの役割は、**その dispatch が使うカーネルの WGSL 宣言**
  （`@binding(N) var<storage, read>` / `read_write`）から機械的に採る（手書きの表を持たない）。
- 計画は bindings だけの純関数で、`PreparedPlan`（導出済み計画）と同じ寿命で持つ。**GPU の
  granted limits（`maxBufferSize` / `maxStorageBufferBindingSize` / offset 整列）は値として渡す**
  （ADR 0022 — 融合の行ブロック枚数と同じ扱い）。

### 2. 束縛は実寸で切る

`createBindGroup` のエントリは、計画が配った slot については `{ buffer: 領域, offset, size }` を
渡す（Session 常駐の重み / scale / params・入力・state スロットは従来どおりバッファ全体）。
「要求より大きいバッファを配らない」（ADR 0004 / `arena.ts` の MUST）は「束縛範囲を実寸に切る」
へ言い換える。

### 3. ミス run とヒット run は同じ計画を使う（アリーナの出力プールは退役）

- ミス run（初回）は計画の領域を run 寿命で確保して回し、run の末尾で返す。ヒット run は同じ
  計画の領域を Session 常駐（backing）として保持し、bind group を焼き込む — 現行の 2 経路の
  区別（プール確保 vs 常駐 slot）はそのまま、**確保の中身だけ**が「slot 1 本ずつ」から
  「領域 1〜数本」へ変わる。
- `RunArena` の役割は「host が書くバッファ / readback staging / 領域の所有と flush-before-destroy」
  に縮む。出力ストレージの参照計数・プール・`assertDrained` は計画側の再生へ移る。

### 4. 見積りは同じパッカーを共有する

`estimateGraphMemory` の `transientSlotBytes` は、融合前ノード列から同じ形の「確保プログラム」を組み
**同じ関数**でパッキングした領域総和を返す（ADR 0070 決定 5 の「実行と同じ規則」を関数共有で
満たす — 3 箇所に同じ規則を写す現状を 1 箇所へ）。融合が畳む中間と融合ルールの一時は従来どおり
`unaccounted` の側（融合前の列で数える近似は不変）。見積り側の限界値は WebGPU core 既定
（`maxBufferSize` 256 MiB / 整列 256 B）を使い、device が渡されればその granted 値を使う。

### 5. 中間の上限 preflight（ADR 0089 決定 1 の中間への拡張・レビュー C 案）

計画時に「slot 実寸 > `maxStorageBufferBindingSize`」「領域 > `maxBufferSize`」を**確保の前に全件列挙して
落とす**（`assertWeightsWithinLimits` と同じ文言の型）。ADR 0089 Consequences「run 時 transient は
errorScope 頼み」の起票を閉じる。

### 6. 診断欄の意味

- `ArenaStats.peakTransientBytes` = **生存ピーク**（計画の同時生存バイトの最大 —
  `TransientPlan.peakLiveBytes`）。`transientBytes` = run 末尾に生存している中間 = pin されたグラフ出力の
  総バイト数。`allocatedBytes` = 領域の総和 + host / staging。`reuseCount` = 先に置かれた別の slot と
  バイト範囲を共有した slot の本数（`TransientPlan.sharedSlots` — 旧プールの「配り直し回数」に相当し、
  full-write の故障注入テストが配り直しの発生を確かめる観測点として残す。実装時 2026-09-05 に「0 固定」
  から改めた）。`PlanBackingStats.residentBytes` = 領域の総和。
- 「slot の総バイト数 = 非 backed run のプール確保」（footprint 不変の門）は「backed の領域総和 =
  ミス run の領域総和」へ読み替える（同じ計画を使うので恒等）。

### 7. 据え置くもの

full-write（ADR 0014 — 全ノードは自分の束縛範囲の全バイトを書く。前 run / 別テンソルの残骸を読まない
根拠はそのまま）・flush-before-destroy・writeBuffer 対象はプール外（入力は領域に入れない）・
state スロットは context 所有（ADR 0066）・codegen 決定性。

### 8. 退役する MUST

- ADR 0004 バッファ管理「per-run アリーナ（**サイズ別プール** + 最終消費者解放）」→ 区間 + offset 配置。
- `recipe.ts` `derivePlanSlots` の「**独自パッキング禁止**（RunArena のサイズクラス LIFO を仮想的に
  再生して導く）」→ 計画は 1 本の純関数（`transient-plan.ts`）で、実行・焼き込み・見積りがそれを共有する
  （「別の値が同じ実体を掴んでよいか」の判定が 2 実装に分かれる懸念は、判定が 1 箇所になることで解ける）。
- `arena.ts` の「要求より大きいバッファを配らない」→ 決定 2 へ言い換え。

## Consequences

- 見込み（実 IR の模擬・重なり検査済み・first-fit は生存ピークちょうどに到達）: BiRefNet 2048² の
  workspace 23.76 → 7.50 GiB（decoder 末尾の代数書き換え〈レビュー A 案〉と併せて 2.50 GiB）、1024² は
  6.14 → 1.88 GiB（併せて 0.63 GiB）。他家族も同じ規則で縮む（perf ではなくメモリ管理の是正）。
- 実測（2026-09-05・結線後・RTX 3080 Ti / Deno wgpu Vulkan）: BiRefNet 1024² の中間の領域総和
  2,020 MiB（生存ピーク 1,920 MiB・領域 5 本・共有 slot 2,060 本・dispatch 2,308）で、見込み
  1.88 GiB とほぼ一致（差は 256 バイト整列と別領域規則のぶん）。run 時間 1.5〜1.7 s。2048² は
  計画時 preflight が `'upsample_bilinear2d_20' 3221225472B / 'cat_212' 4026531840B` の 2 本を
  列挙して落とす（A 案の代数書き換えで消える 2 本 — そのとおり）。
- 実測（2026-09-05・A = recipe パッチ ⑨ と併せて・同じ機）: 1024² の中間 749 MiB（見込み 0.63 GiB）・
  run 1.8 s / 2048² の中間 2,948 MiB（生存ピーク 2,560 MiB・見込み 2.50 GiB）・重み 1,116 MiB・
  GPU 総確保 ≈ 4.1 GiB・run 7.5〜8.6 s（RTX 3080 Ti 12 GiB）。2048² は preflight を通り、runtime /
  models の e2e（1024² golden）は緑のまま。
- 触る面: `packages/runtime/src/runtime/transient-plan.ts`（新設・パッカー）/ `recipe.ts`（`derivePlanSlots`
  → 計画・`createBindGroup` の offset/size・`executeStepRecipe` の確保経路）/ `executor.ts`
  （`#activateBacking` の領域確保・ミス run の領域確保・readback の offset）/ `estimate.ts`
  （`transientSlotBytes` の共有化）/ `arena.ts`（出力プールの退役）/ `recipe-builder.ts`（dispatch の
  読み書き役割を WGSL から採る）。公開面（`mod.ts`）は変えない。
- 回帰門: 全家族の e2e golden（sha256）・`gpu_plan_backing_test`（footprint 不変の読み替え）・
  `estimate_test` の GPU 突合・`gpu_full_write_test`（故障注入）・`runtime_executor_test` の
  `peakTransientBytes` 期待値（生存ピークへ更新 — 数値が変わるのは意味が変わるため。旧値との
  対応を doc に残す）。
- 未検証: 記号次元を持つ家族（gemma4 / sbv2）でも区間は bindings ごとに計画時に確定するので同じ形が
  成立する見込み。Metal / ブラウザ（Dawn）での offset 束縛の validation 差は実機で確認する。
