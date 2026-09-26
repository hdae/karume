> 2026-09-26 時点の実測スナップショット。Intel Arc B570（BMG G21）/ Deno 2.9.6 / `tools/flag-bench`（HEAD `9142e37d`〜`5442b5c2`）。壁時計は Deno のフェンス床込み、GPU 時間は timestamp-query の tick 未換算（B570 は 1 tick = 52.08 ns）なので**基準比 % だけを読む**。

# gemma の実行フラグ単体ベンチ — `-fast` 束の中身を実測で決める

利用者の決め方（2026-09-26）: 「用意されているフラグをそれぞれ単体でベンチし、速度低下するものが無ければ
全部入り。全部入り（低下なし）と全部入り（低下含む）も比べ、勝った方を採用」。この機に Chrome が無いので、
`tools/flag-bench`（Deno・set ごとに fresh な pipeline・固定 prompt 2 本 × 暖機 1 + 計測 2・ABBA 2 round・
基準 = quant `i4` + `stateAttentionReduce: "sequential"` = runtime の参照経路）で測った。主指標は計測 ON の
走行の GPU decode ms/step、副指標は計測 OFF の走行の壁時計 decode ms/token。

## 結果（基準比 %・負 = 速い）

| フラグ                                        | E2B 通常 GPU | E2B 通常 壁 | QAT E2B GPU | QAT E2B 壁 | QAT E4B GPU | QAT E4B 壁 |
| --------------------------------------------- | -----------: | ----------: | ----------: | ---------: | ----------: | ---------: |
| `linearGemvReduce: "parallel"`                |        −34.1 |       −19.1 |       −17.5 |      −16.7 |       −26.6 |      −17.4 |
| `fuseRmsNormAdd`                              |         −8.3 |        −1.7 |        −1.4 |       −1.3 |        −1.1 |       −0.9 |
| `stateAttentionReduce: "parallel"`            |        −16.0 |        −7.0 |        −7.1 |       −5.7 |        −7.1 |       −5.1 |
| `stateAttentionReduce: "parallel-fused"`      |        −16.2 |        −6.8 |        −7.4 |       −7.2 |        −6.5 |       −5.8 |
| `fuseLinearStaticQuantize`（+ GEMV parallel） |            — |           — |       −34.1 |      −20.8 |       −30.9 |      −20.0 |
| `packedStaticQuantize`（+ GEMV parallel）     |            — |           — |       −18.8 |      −18.4 |       −29.7 |      −19.0 |
| 全部入り（attention parallel）                |        −46.9 |       −28.0 |       −48.6 |      −29.6 |       −41.6 |      −29.2 |
| **全部入り（attention parallel-fused）**      |    **−47.9** |   **−28.7** |   **−49.3** |  **−29.6** |   **−41.7** |  **−29.5** |

絶対値（計測 OFF の壁時計 decode ms/token・参照 → 全部入り parallel-fused）: E2B 通常 36.6 → 26.1・QAT E2B 36.1 → 25.4・
QAT E4B 48.8 → 34.4。dispatch/step（計画上・GPU 計測 ON の走行）: E2B 通常 1,028 → 894・QAT E2B 1,513 → 1,104・QAT E4B 1,867 → 1,363。

## 読み取り

- 速度低下するフラグは 3 バリアントとも無い → 全部入り。attention は `parallel-fused` が GPU で僅かに速く dispatch も少ない
  ので採る（M2 では利得無し〈ADR 0102〉— この束は B570 の実測で決めたもの）。
- token 列は全 set で訪問間同一（決定性）。E2B 通常は参照席とも同一。QAT は参照席と生成 token が異なる（既知）。
- QAT E4B は並列 GEMV の適格表に E4B の形を足した後の値（97bb7e05）。足す前は GEMV 系 3 フラグが全 linear で
  no-op だった（Fable レビュー i4 F02）。flag-bench の適用キー census（`noKeyChange`）は全 set で false。
- E4B 通常は export が OOM（PLE 表 f32 10.5 GiB を含む定常 ≈ 27.8 GiB が 31 GiB 機に収まらない）で未計測。

## 束の確定

- E2B 通常 `i4-fast` = {linearGemvReduce: parallel, fuseRmsNormAdd: true, stateAttentionReduce: parallel-fused}（5442b5c2）。
- QAT E2B / E4B `i4-fast` = {linearGemvReduce: parallel, fuseRmsNormAdd: true, fuseLinearStaticQuantize: true,
  packedStaticQuantize: true, stateAttentionReduce: parallel-fused}（E4B QAT は新設・既定へ）。
- 結果の JSON: `outputs/bench/flag-bench/2026-09-26/*.stdout.json`（git 追跡外）。
