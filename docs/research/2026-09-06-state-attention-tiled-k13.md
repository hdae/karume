# K-13: states 形 attention の prefill を GEMM 骨格のタイル経路へ（①ₜ / ③ₜ）— K/V を行タイルごとに 1 回読む（2026-09-06）

> 時点スナップショット（RTX 3080 Ti / Vulkan・Deno 2.9.6）。設計の正本は ADR
> [0067](../decisions/0067-autoregressive-attention-vocabulary.md)（追記 2026-09-06「幾何表」）、採否は
> [perf-ledger](../perf-ledger.md) K-13。前段 = K-14（[research](2026-09-06-state-qk-parallel-k14.md) §3 の
> 「M でバケットする幾何表」）。生データは `outputs/bench/karume-gemma4/2026-09-06_ab-k13*-graph-*` と scratchpad の
> 壁ログ（揮発 — 数値は本文に写す）。

## §0 要約

- prefill 計画（M = chunk 行数 768）の ①QK / ③PV は 1 invocation = 1 要素で K / V 行を M 行ぶん読み直す traffic
  律速（14.7K token の chunk 1 本で attention が GPU の 79%）。融合 attention（anima 系）が持つ GEMM 骨格
  （共有タイル・レジスタブロック・M バケット幾何・K タイル 16 昇順）に states 用の断片（ring / ins の 2 源の
  行基点・live で切る境界・述語 → −inf・pad 行）を差した ①ₜ / ③ₜ を足し、M ≥ 16 の計画で席に依らず選ぶ。
- **①ₜ（段 1・`ad8a4b9`）**: 1 項の式（半スケールを q / k 双方）と加算順（d 昇順）が ① と同じで **ビット同一**
  （S バッファ全語の u32 一致 17 ケース）。実測: prefill ①QK **14.1〜17.6 s → 1.08〜1.19 s（×13〜15）**・chunk 18
  単体 1,416〜1,878 → 92〜105 ms・**P=16K の prefill 壁 55.7 / 64.7 s → 38.3 / 35.6 s（−31〜45%）**・decode 不変。
- **③ₜ（段 2・`39d5e4e`）**: A = P（S と行統計から充填時に `exp(S − m)·inv`・非実体化）・B = V（2 源・転置なし）で
  ③ と **ビット同一**（O バッファ全語の u32 一致 19 ケース・pad 行は select で厳密 +0.0・V に +Inf を注入しても 0）。
  段 1 + 段 2 の実測: prefill 20 chunk の GPU **35.2 / 40.9 s → 10.3 / 11.0 s（×3.4〜3.7）**・③PV 13.7〜15.5 →
  1.24〜1.32 s（×11〜12）・chunk 18 単体 3,092〜3,942 → 602〜660 ms（×5〜6）・**P=16K の prefill 壁 53.6 / 56.8 s →
  19.5 / 20.6 s（−64%）**・decode 不変（33.3〜33.9 ms/token）・token 列一致。残る prefill の GPU は linear が 72%。
- 帰結: ③′（K-12 の KV 並列縮約）の適用は M < 16 の計画（実質 decode）に狭まる — prefill 計画は席に依らず
  参照経路とビット同一の ③ₜ になる（K-12 の prefill 利得は誤差内だった）。

## §1 実装

- 骨格は gemm.ts が所有し、`GemmSpec` に op `attention_state_qk`（/ `attention_state_pv`）を足して融合 attention と
  同じ組み立て関数の隣に生成器を置く。再利用したのは数値契約の乗る部分（`skeleton` = Dims / 共有タイル /
  K タイルループ / barrier、`accumulatorUpdate` = `acc = acc + a * b`、`fillA` の半スケール、`storeBTransposed`）。
- states 固有の断片: 列 → K 行基点と出どころ（`col < past` → スロット `(kv_plane·C + slot_row(col))·D` /
  以降 → ins `(kv_plane·M + (col − past))·D`）を K タイルループ不変として prologue で畳む・B ローダは `cl < live`
  で切る（`dims.n` = col_cap は静的上界）・store は `local_row < effective_rows` と `cl < live` だけ書き、述語外は
  `neg_inf` を書く（① と同じ残骸の残り方）。述語・幾何の断片（`slot_row` / `live_columns` / `in_window` /
  `effective_rows`）は ① と同じ生成器（uniform 名を引数化・既存生成物はバイト不変）。
- Dims: 骨格の先頭 3 語に `m = rows_block` / `n = col_cap` / `k = depth` を割り当て、states の欄（row_offset /
  chunk_rows / kv_repeat / window / capacity / neg_inf / scale）を続ける。①③ 共有の `Params` の語順は不変。
- 幾何 = `gemmGeometryForRows(chunkRows)`（M ≤ 64 = M16N16 / ≤ 512 = M64N32 / それ以上 = M128N128・共有メモリ
  16,384 B = core 既定ちょうど・融合 attention と同条件）。キー = `attention_state_qk:v1:f32:reg128x128r8x8w16[:sliding][:gqa]`
  （gemm の幾何断片 `gemmGeometryTileKeyPart` を共有）。v4 経路は持たない（実効 N が live で quad 書きが残骸を潰す）。
- 切替（recipe-builder）: M=1 かつ席 parallel → ①′ / M ≥ 16 → ①ₜ（席に依らない既定）/ それ以外 → ①
  （純関数 `stateQkTiledEligible` = `chunkRows >= 16`・2 条件は重ならない）。

## §2 実測（段 1・A = `e309e84` の worktree・B = `ad8a4b9`・A B B A・同一セッション）

内訳 = `opbench graph --capacity 16384 --prompt <14,709 token>`（timing on・prefill 20 chunk + decode 8）、
壁 = ctx-sweep 台本の写し（公開面・capacity 16,384・new-tokens 32）。

| 量                                 | A（①）           | B（①ₜ）               | 比       |
| ---------------------------------- | ---------------- | --------------------- | -------- |
| prefill 20 chunk の ①QK（内訳）    | 14.1 / 17.6 s    | **1.08 / 1.19 s**     | ×13〜15  |
| chunk 18（14K 文脈・768 行）の ①QK | 1,416 / 1,878 ms | **92 / 105 ms**       | ×15〜18  |
| prefill 20 chunk の全 GPU（内訳）  | 35.1 / 45.8 s    | 23.6 / 26.6 s         |          |
| 〃 うち ③PV                        | 13.6 / 18.8 s    | 14.5 / 16.5 s（不変） |          |
| prefill 壁（P=16K）                | 55.7 / 64.7 s    | **38.3 / 35.6 s**     | −31〜45% |
| decode GPU / token（①′ のまま）    | 20.45 / 20.87 ms | 20.48 / 20.66 ms      | 不変     |
| decode 壁 中央値（P=16K）          | 35.27 / 36.84 ms | 33.96 / 33.56 ms      | 誤差内   |

読み: ①QK は K 読みが `M` → `M / 128`（tileM）になり、chunk 単体で ×15。段 1 の後の prefill は ③PV が 6 割。

## §3 段 2（③ₜ・A = `e309e84` の worktree・B = `39d5e4e`・A B B A・同一セッション）

| 量                                   | A（①・③′）                     | B（①ₜ・③ₜ）              | 比                     |
| ------------------------------------ | ------------------------------ | ------------------------ | ---------------------- |
| prefill 20 chunk の全 GPU（内訳）    | 35.2 / 40.9 s                  | **10.3 / 11.0 s**        | ×3.4〜3.7              |
| 〃 うち ①QK                          | 14.2 / 16.6 s                  | 1.10 / 1.16 s            | ×13〜14                |
| 〃 うち ③PV                          | 13.7 / 15.5 s                  | **1.24 / 1.32 s**        | ×11〜12                |
| 〃 うち linear                       | 6.8 / 8.2 s                    | 7.4 / 7.9 s（不変・72%） |                        |
| chunk 18（14K 文脈・768 行）の全 GPU | 3,092 / 3,942 ms               | **602 / 660 ms**         | ×5〜6                  |
| prefill 壁（P=16K）                  | 53.6 / 56.8 s                  | **19.5 / 20.6 s**        | −64%                   |
| decode 壁 中央値（P=16K）            | 33.32 / 33.73 ms               | 33.89 / 33.88 ms         | 誤差内                 |
| 生成 token 列（先頭 6）              | 1024,506,31770,4799,236761,669 | 同一                     | ビット同一の実走裏付け |

③ₜ の設計上の差（段 1 と違う点）: B は V 行がそのまま vec4 に載るので転置なし（`fillBDense` 系）・行頭の 2 源分岐は
充填側・K ループの上限は実効 live（`k_live` — 有効行を含まない行タイルは 0 周 = 仕事量 ∝ Q の契約）・pad 行は
`select` で厳密 +0.0（A を 0 埋めするだけでは非有限な V で NaN 化する）・dispatch の行軸は rows_block（full-write）。

## §3.1 幾何表（この波の帰結）

| 計画の M        | ①QK                  | ③PV                  | 数値                                 |
| --------------- | -------------------- | -------------------- | ------------------------------------ |
| 1（decode）     | ①′（席 parallel）/ ① | ③′（席 parallel）/ ③ | 席で選ぶ（並列縮約は帯門）           |
| 2〜15           | ①                    | ③′（席 parallel）/ ③ | 同上                                 |
| ≥ 16（prefill） | ①ₜ                   | ③ₜ                   | 参照経路とビット同一（席に依らない） |

## §4 未検証

- Metal / DirectX でのビット同一（① との差だけを見る門なので fma の判断が揃えば動かないが、実機未確認）。
- `maxComputeWorkgroupStorageSize` が 16,384 B 未満の device（既定幾何 M128N128 は M ≥ 513 の計画でだけ出る —
  融合 attention と同条件）。
