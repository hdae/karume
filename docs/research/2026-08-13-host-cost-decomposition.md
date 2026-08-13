# ホスト固定費の分解実測（H-2）— フェンス待ちが支配・K-7 按分の置換

> NOTE: 時点スナップショット。数値は 2026-08-13 の実測（RTX 3080 Ti / Linux / Vulkan /
> Deno 2.9.4・main `34a3e18` = K-4a 後）。手法・台本は scratchpad（揮発）— 恒等式分解 +
> 遅延注入。perf-ledger H-2 の消化記録で、H-1/H-3/H-5 の採否根拠と K-7 の裁定材料を持つ。

**結論を先に**:

1. **EG bare の per-run 固定費 ~38ms の正体はフェンス待ち**（§2）: 1 run = 壁 54.0ms 中、
   onSubmittedWorkDone ×2（同期部 8.66ms + 待ち 11.3ms each）+ mapAsync（待ち 11.3ms）で
   **フェンス関連 51.3ms（95%）**。素の WebGPU でも待ちの床は ≈11ms/本（§1）— GPU 実時間
   ~15ms はこの待ちの中に隠れており、純粋なフェンスオーバーヘッド ≈36ms が「謎の固定費」
   だった。
2. **H-1 の素の価格差 = 12ms/run を直接実測**（§1）: 同一カーネルで二段待ち（現行形）
   24.4ms vs 単一待ち 12.5ms。EG の kill 基準（53.8 → 43.8ms 未満 = −10ms）に**この 1 手で
   到達する**。
3. **irodori の素の生成壁は 8.6s**（gpuTiming OFF 初取得・§3）。run window 8.2s のうち
   フェンス関連 ≈7.5s（onSubmittedWorkDone sync 3.8s + 待ち 2.5s + mapAsync 1.2s）。
   GPU 4.2s を差し引いた純オーバーヘッド ≈3.3s。run 境界の露出傾き **0.82**（注入実測）。
   従来の「計測装置代 ≈2.1s」外挿は過小 — 実測 ≈3.1s（restats §3.3 の露出 5.3s は
   旧 HEAD + 外挿の値。現 HEAD の素の露出は ≈4.4s）。
4. **per-dispatch ホスト代は実測 0.62µs/dispatch・binding 3〜7 本でほぼ平坦**（§4）。
   K-7 の均一按分（≈1.1s）は **50 倍の過大評価**と確定 — dispatch −33,600 のホスト利得 =
   21ms × dispatch 地点の露出傾き 0.37 ≈ **8ms**。

## 1. 素の WebGPU 原価（karume 非経由・micro）

| API                                                  |                                              実測 |
| :--------------------------------------------------- | ------------------------------------------------: |
| createBuffer(4KiB) / writeBuffer(4KiB)               |                                     11.0 / 11.1µs |
| createBindGroup（binding 2/4/6/8）                   |                        6.3 / 10.1 / 14.0 / 18.6µs |
| pass.setPipeline / setBindGroup / dispatchWorkgroups |                              0.05 / 0.09 / 0.07µs |
| encoder.finish（1 / 64 / 256 pass）                  |                               0.34 / 6.3 / 21.1ms |
| queue.submit（1 pass）                               |                                             151µs |
| **onSubmittedWorkDone**（trivial 作業）              |            同期部 0.07ms + **待ち 11.07ms（床）** |
| **H-1 素の価格差**（wg 1/256/4096 で不変）           | 二段 24.4〜25.1ms / 単一 12.5〜12.6ms = **Δ12ms** |

フェンス待ちの床 ≈11ms は GPU 負荷に依存しない（wg1 でも同じ）— Deno/wgpu の
フェンス解決レイテンシ。encoder.finish の pass 数線形（host-overhead-recon §4.2 の再確認）。

## 2. EG bare の恒等式分解（20 run・gpuTiming OFF・装置代 2.2ms = 4.2%）

1 run の壁 54.0ms = 同期 API 19.2ms + ブロック待ち 34.0ms + JS 残差 0.8ms。

- 同期の 9 割 = **onSubmittedWorkDone.sync 17.3ms/run**（2 本 × 8.66ms — Deno の同実装は
  呼び出し自体が同期ブロックしうる: 2026-08-04 recon §4.1 の現行 HEAD 再確認）
- 待ち = onSubmittedWorkDone 22.7ms/run（2 本 × 11.3ms）+ mapAsync 11.3ms/run
- run あたりフェンス 3 本（flush 内 2 + readback mapAsync 1）× 実効 11〜20ms が全て。
  encode（pass 3 呼び出し × 958 dispatch）は 0.7ms/run で**無視できる**
- 注入傾き: dispatch / run 地点とも ≈1.0（完全露出 — GPU 15ms に隠れる余地がない）

## 3. irodori voice-clone の恒等式分解（108 run・gpuTiming OFF・装置代 176ms = 2.0%）

素の生成壁 **8.59s**（gpuTiming ON の従来値 11.7s との差 ≈3.1s が計測装置代の実測 —
外挿 2.1s は過小だった）。run 壁合計 8.22s の内訳（level 1）:

| 項                                 |    値 | 備考                                                             |
| :--------------------------------- | ----: | :--------------------------------------------------------------- |
| onSubmittedWorkDone.sync           | 3.80s | 224 本 × **17.0ms**（EG の 8.7ms から倍増 — 保留仕事量で伸びる） |
| onSubmittedWorkDone 待ち           | 2.50s | 224 本 × 11.1ms                                                  |
| mapAsync 待ち                      | 1.22s | 109 本 × 11.2ms                                                  |
| encoder.finish（encode+flush）     | 0.29s | チャンク 284 + flush 108                                         |
| outside writeBuffer + createBuffer | 0.13s | run 間の入力再アップロード（H-3 の対象）                         |
| createBindGroup ほか run-other     | 0.18s | backing 構築ほか                                                 |
| pass 3 呼び出し（encode）          | 0.01s | 161,639 dispatch × 0.2µs — **無視できる**                        |

- **フェンス関連 ≈7.5s / run 壁 8.2s**。GPU 4.2s はこの待ちに部分的に隠れ、純オーバー
  ヘッド ≈3.3s（= 現 HEAD の露出ホストの主部）
- 露出傾き（注入実測）: **run 境界 0.82**・dispatch 地点 0.27〜0.42・チャンク地点は
  注入が小さすぎ判定不能（±50ms の壁揺れに埋没 — 負値はノイズ）
- WAV sha256 は全 12 走で `e7846ac1…` 一致（計測・注入の計算経路無影響を毎走証明）

## 4. per-dispatch ホスト代（K-7 均一按分の置換）

帰属済み 161,639 dispatch の実測: **平均 0.617µs/dispatch**。binding 3 本 0.611 / 4 本
0.621 / 7 本 0.612µs — **binding 本数依存なし**（bind group は焼き込み済みで、encode は
pass 3 呼び出しだけのため）。EG 側も 0.673µs で同水準。

**K-7 の裁定式への代入**: 33,600 dispatch 削減 × 0.617µs = 20.7ms（ホスト raw）×
露出傾き 0.37 ≈ **8ms/生成**。GPU 側上限 143ms（restats §7）と合算しても ≈150ms =
素の壁 8.6s の **1.8%** — 均一按分の見込み ≈1.1s は 50 倍の過大評価だった。

## 5. 帰結（採否根拠）

- **H-1（二段待ち → 単一化）**: 素の価格差 12ms/run 実測。EG 54.0 → 42ms 級 = kill 基準
  到達。irodori は 108 run × 12ms × 傾き 0.82 ≈ **1.1s**。TDR 予算推定の窓と timestamp
  回収が同じ待ちに相乗りしている再設計制約は不変（perf-ledger H-1 行）。
- **H-5（反復状態 GPU 常駐）**: run 境界そのものを 100 本消す — フェンス 3 本 × 100 run
  ぶんの構造が対象で、H-1 とセットなら irodori 壁 8.6s → GPU 律速 ≈5s 級（×1.7）の
  見込み。**波②の本命**。
- **H-3（入力の内容アドレスキャッシュ）**: outside writeBuffer + createBuffer 実測
  0.13s × 傾き 0.8 ≈ 0.1s — 単独では小粒。H-5 の常駐化に併合するのが妥当。
- **K-7**: 上記 §4 — 棄却水準（裁定はユーザー — 台帳 K-7 行参照）。
