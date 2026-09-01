# 0089: GPU メモリ適合は「絶対上限との決定論的比較」へ昇格する（メモリ管理波 Phase A）

- Status: accepted（2026-09-01 — ユーザー裁定「OK、進めてください」・設計裁定 1a / 2a / 記録承認）
- Date: 2026-09-01
- 関連: ADR [0070](0070-shard-loading-admission.md)（決定 4 = errorScope 全面依存・決定 5 =
  空き側と比較しない・決定 6 = limits preflight の席予約 — 本 ADR がその席の実装）/
  [0066](0066-generation-context-state-slots.md)（追記 5 = state の binding 上限検査）/
  [0038](0038-manifest-v1.md)（§7 追記 = `requiredLimits` 欄の導入）/
  [0081](0081-shard-spec-v2.md)（shard 1GiB 上限 — 本 ADR が読み手側の門を実装）

## Context

確保失敗の検出は shard 単位の out-of-memory / validation errorScope に全面依存していた
（ADR 0070 決定 4）。この網には 3 つの弱点がある: ①**実装依存** — M2 実機で out-of-memory
errorScope が沈黙する実測（known-issues・wgpu-hal metal の `check_if_oom()` は no-op）が
あり、沈黙環境では「確保失敗 = 無効バッファへの no-op writeBuffer = ゴミを読む」が通り得る
②**遅い** — 検出は数 GiB を転送した後 ③**粗い** — 名乗れるのは失敗 shard までで、どの重みが
何バイト超えたかは出ない。

一方で確保寸法は**宣言だけで確定している**（常駐計画 `planWeightResidency` は prepare 相の
純関数・state スロットも束縛で決まる）。また manifest の `requiredLimits` 欄は「exporter が
書かない（gemma4 recipe だけが手書き）・hub が parse するだけ・読み手ゼロ」の三重に空で、
shard 1GiB 上限（ADR 0081）の門は書き手側にしか無かった。

## Decision

**決定論的に比較できるものは全て、確保・取得の前の明示検査に昇格する。** 比較の相手は
デバイスの**絶対上限**（`maxBufferSize` / `maxStorageBufferBindingSize`）だけで、空き VRAM・
確保合計とは比較しない（ADR 0070 決定 5 の維持 — WebGPU は総量を露出せず、当て推量の閾値は
健全な環境で誤拒否を作る）。総量の最終検出は errorScope のままで、Metal の沈黙環境では
それが残る — [limitations](../limitations.md) の「GPU メモリの事前検査は絶対上限まで」節が
この限界の記録。

1. **重み経路の明示検査**（runtime）: 席 → 確保バッファの写像を
   `planWeightBuffers`（weight-residency.ts）へ 1 本化し（適格 = payload・適格外 = f32
   展開後・i8 / i4 は companion scale をもう 1 本・`toSizeClass` 整列後 = createBuffer 実寸）、
   `assertWeightsWithinLimits` が Session 構築の入口（shard ループ前 = 1 バイトも上げる前）で
   全エントリを両上限と突合する。超過は**全件列挙して 1 回で**落とす。見積り
   （estimate.ts）も同じ写像を消費する — 検査・見積り・実ロードが別の寸法を主張する形を
   作らない。
2. **state 検査の補完**（runtime）: 既存の binding 上限検査（ADR 0066 追記 5）に
   `maxBufferSize` も加える。`binding ≤ buffer` は device を自前計画する側の関係であって、
   外から渡された GpuContext への保証ではない。
3. **`requiredLimits` の書き手は core の組み立て一括導出**（exporter `karume/limits.py` +
   `dist.py` の `bake_required_limits`）: 需要 = quant が選ぶコンポーネントの最大テンソル
   payload と最大 state スロット（容量は `pipelineConfig` で数値化・記号 2 本以上や束縛欠落は
   fail loudly）。**常駐前提の寸法のみ** — f32 展開のワーストを要求に書くと本来動く環境を
   DL 前に誤拒否する（展開時の実寸は決定 1 の実行時検査が守る）。**焼くのは WebGPU 保証既定
   （256MiB / 128MiB）を超える席だけ**（「欄なし = 既定スペックで動く」の意味論）。
   workgroup 系はカーネル設計依存なので焼かない。**計画側の手書きは二重管理として拒否**
   （gemma4 recipe の手書き席は退役 — core 導出値と完全一致を確認済み）。
4. **shard ≤ 1GiB の読み手検査**（hub）: manifest parse 時に `shards` の各 `size` を検査
   （閉区間・Python 正本 `shards.py` と同値）。`assets` / `extras` は対象外 — 上限は shard
   分割の契約で、1GiB 超の単一付帯資産は合法。
5. **席（次段 = 波 2）**: manifest `requiredLimits` の読み手（models の admission で GPU
   limits と比較して DL 前拒否・adapter 取得の先行）と、`estimateSessionMemory` のロード面
   結線。ADR 0070 決定 6 の残り半分。

## Consequences

- dist 再生成で `requiredLimits` 欄が新設される: anima 全 30 quant（296.8MiB）・irodori
  f32（300MiB 両上限）/ f16（150MiB・binding のみ — 片側だけ超過する帯の実例）。gemma4 は
  手書きと導出が同値なのでバイト不変・sbv2 は既定内で無風。再生成は models 読み手（波 2）と
  同じ回で行う。
- 検査が新たに弾く既存資産は無い: bind group は常にバッファ全体を束縛するので、binding 上限を
  超える重みが正常動作していた経路は存在しない。
- 残る同型の弱点（起票のみ・本 ADR 対象外）: `GpuContext.createResident` と run 時 transient
  の確保は依然 errorScope 頼み。gemma4 PLE sidecar は shard 連番形だが `extras` 席なので
  決定 4 の門の外（現物は上限内）。
