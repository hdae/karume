# skinny-M GEMM 幾何の掃引と第 1 波の効果実測

> 時点スナップショット（2026-08-11・RTX 3080 Ti / Vulkan / Deno 2.9.4）。EmbeddingGemma
> フロア最適化第 1 波（skinny-M 幾何 + rope 融合一般化）の設計材料と検収実測。幾何の規約は
> ADR 0022 追記、rope は ADR 0040 追記が正本。掃引の生 JSON とハーネスは scratchpad（揮発）。

## 1. 掃引（幾何候補 × EmbeddingGemma 実形状）

- 方法: カーネル単体の ABBA 回文対計測（1 パス ~80ms へ reps を倍々較正・メモリクロック
  張り付け・代表値 min・**比のみ有効** — 2026-08-10 kernel-variant-sweep §5 の規約）。
  候補 11 種 × linear 実形状（K,N ∈ {768,1152,3072}×{256,768,1152,3072} の実在 6 種）×
  M ∈ {1,4,64,318,512}。**全 330 点で既定幾何とビット同一**（bitEqual 違反 0 — f32 幾何の
  ビット同一命題に 330 点を追加）。
- 集計: EmbeddingGemma 1 run の実形状構成（q/o 48・gate/up 48・down 24・k/v 48 + M=1 の
  Dense 2 本)で重み付けした linear 合計時間の対既定比:

| M   | 1 位             | 2 位         | 旧暫定 2 段目 M64N32 | 採否                      |
| --- | ---------------- | ------------ | -------------------- | ------------------------- |
| 1   | M16N8 ×3.20      | M16N16 ×3.18 | ×2.50                | M16N16（1% 未満差の同着） |
| 4   | M16N8 ×3.19      | M16N16 ×3.16 | ×2.51                | 同上                      |
| 64  | M16N8 ×3.03      | M16N16 ×3.03 | ×2.49                | 同上                      |
| 318 | **M64N32 ×1.67** | M32N32 ×1.54 | —                    | **未採用**（下の §3）     |
| 512 | **M64N32 ×1.28** | M64N64 ×1.26 | —                    | 同上                      |

- 確定: **M ≤ 64 → M16N16（r1×4 wg4×16）・M ≥ 65 → 既定 M128N128**。M=64 でも行タイル
  4 枚（重み 4 回読み直し）を払って workgroup 数を採る方が勝つ — 「tileM を M の直上に置いて
  読み直しを避ける」という暫定表の仮説は実測で棄却。M16N8 は 3 点とも僅差で上だが規約上
  同値（比 1% 未満）につき、実装・検査済みの M16N16 を保持。

## 2. 第 1 波の効果（EmbeddingGemma-300m・warm）

| 指標                       | 第 1 波前 | 第 1 波後       | 帰属                          |
| -------------------------- | --------- | --------------- | ----------------------------- |
| 実 dispatch / run          | 1,294     | **958**（−26%） | rope 融合 48 箇所 × 7 本      |
| GPU（gpuTiming on・bare）  | 47.3ms    | **15.5ms**      | linear 41.3 → 12.4ms が主     |
| linear µs/dispatch（M=4）  | 242.6     | **72.6**        | 幾何（掃引比 ×3.2 と整合）    |
| wall bare (T=4)            | 76.4ms    | **52.7ms**      | ×1.45                         |
| wall query-en (T=16)       | 76.4ms    | **52.5ms**      | ×1.46                         |
| wall long-document (T=318) | 81.1ms    | 79.5ms          | M=318 は境界の外（by design） |

- gpuTiming on の値は 1 dispatch = 1 pass の計測費込み。run 間変動 ±2ms 程度
  （15.5〜17.6ms を観測）。PNG/WAV 門・e2e 門（atol 1e-6）は全て緑のまま。
- 対 ORT Web WebGPU（同一 GPU・bare 21.4ms）: 3.6 倍差 → **2.5 倍差**へ縮小。

## 3. 帰属と残り（open）

- **bare の残り 52.7ms はホスト支配へ移行**: GPU 実時間は計測費抜きで 15.5ms 未満なので、
  ホスト側（958 dispatch の encode + プラン走査 + 同期）が **~40ms ≈ 75%**。dispatch 数を
  減らす次の道具は融合 attention の mask 対応（ADR 0023 改訂 — 約 −250 本の見込み）で、
  per-dispatch / per-run のホスト固定費の分解はその後の計測課題。
- **M 65〜512 のバケット追加は未裁定**: M64N32 で T=318 の linear に ×1.67 の余地
  （wall 79.5 → 60ms 台の見込み）。ただしこの範囲は Anima / SBV2 の実形状も踏むため、
  採用するなら両モデルの掃引 or E2E A/B とセット（EmbeddingGemma の形状だけで裁定しない）。
- **バッチ変種（B=32）は export 段でブロック**: transformers の packed-sequence 分岐が
  B>1 で eq/ne/index を IR に残す + bool 定数を IR v1 initializer にできない制約
  （known-issues 起票）。occupancy 検証としては本掃引の M=318/512 が代替を果たした。
- 掃引は linear のみ（GPU 時間の 87% を占めた律速族）。bmm は M=4 で 39.8 → 11.8µs/disp と
  同傾向の改善を run 実測で確認済み — 単体掃引は未実施。

## 4. 追記（同日・波①）: 中 M バケット 65〜512 → M64N32 の採用

§3 で未裁定だった中 M バケットを、採用条件（両モデルの E2E A/B とセット）を満たして採用:

- **Anima/SBV2 の ABBA**（e2e テスト時間・A=無し/B=有り の A,B,B,A）: w8a8-1024
  14/14/14/15s・w8a8-512 7/7/7/7s・f16-1024 26/28/28/29s・fromPretrained 11×4・SBV2 1s×4。
  f16 の単調増加は A/B と無相関の熱ドリフト（最遅が最後の A）— **バケット起因の退行なし**。
  静的にも Anima の GEMM 3 op に M∈[65,512] の実形状はほぼ無い（text encoder は M=64 で
  小 M バケット・DiT は S ≥ 1024）。
- **PNG/WAV 門 sha256 全一致・verify 779/0**（ビット同一の実測命題に中 M バケットの点を追加）。
- **EmbeddingGemma long-document (T=318): wall 79.2 → 63.8ms（×1.24）**。bare はホスト
  固定費（§3）律速のため不変。M ≥ 513 は既定のまま（DiT の実測選定領域 — 補間しない）。

最適化 3 波（skinny-M/rope 融合 → attention mask → 中 M）後の 5 ケース確定値と ORT 比較は
[2026-08-11-embeddinggemma-ort-comparison.md](2026-08-11-embeddinggemma-ort-comparison.md) §6。
