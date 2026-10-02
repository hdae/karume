# フラグ無しの Chrome（M2）でのアプリ内掃引 — ADR 0117 段 7 の記録（2026-10-02・時点スナップショット）

> 性格: 2026-10-02 時点の観測記録。裁定は含まない — 正本は ADR [0117](../decisions/0117-app-geometry-tuning.md) と backlog。

## 1. 目的

ADR 0117 検収 段 7（フラグ無しの Chrome で、アプリの流れ〈掃引 → 生成 → 保存 → 再起動 → 照合 → コールバックで注入〉が
通る）の最初の段 = 掃引の記録を読む。残りの段（生成 → 保存 → 再読み込み → 「保存した表（照合して注入）」→ 1 枚生成）の
JSON は未着。

## 2. 記録

`outputs/bench-browser/geometry-sweep-browser-2026-10-02T08-21-22.303Z.json`（gpu-lab 掃引タブ・checkout `a0ff1702`・
checkoutDirty false・Chrome 154・macOS・開発者フラグ無し）。

| 項目                      | 値                                                                                                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| adapter の 4 欄           | `apple` / `metal-3` / `""` / `""`（フラグ無しなので device と description が空）                                                                                                     |
| 候補集合                  | `quick+`・57 ケース・521 行                                                                                                                                                          |
| 所要（cases[].elapsedMs） | 合計 872 s（startedAt 08:06:50 → date 08:21:22）                                                                                                                                     |
| 失敗 / 出力不一致         | 0 / 0・deviceLost null                                                                                                                                                               |
| 既定の再測定比            | 57 ケースとも 0.969〜1.053（0.9〜1.1 の範囲内・除外なし）                                                                                                                            |
| `gpuTiming`               | `{ feature: true, unit: "ns", quantized: false }`                                                                                                                                    |
| 新しい欄（段 6）          | `startedAt` / `aborted: false` / `cases[].elapsedMs` / `caseSet c1060d8829f3b9de` / `defaultKernels ea5c10c2bfb5047d`（`candidateKernels` は bundle が `b9245574` より前なので無い） |

## 3. 観察 1: フラグ無しでも timestamp は 100 µs に丸められていなかった

2,605 round のうち 100 µs の倍数は 0（1 µs の倍数も 17 だけ）。値の例 `91357184, 91553792, 91619328`（ns）。
リポにある掃引記録 14 本すべて（Chrome 153 / 154・M2 / M5 / RTX 5070 Ti / Mali-G715・フラグ有り 13 本 + 無し 1 本）で
`quantized: false` で、100 µs の倍数の round はほぼ 0（RTX で 1〜2 個 = 偶然）。

- 事実: この M2（macOS・Chrome 154・フラグ無し）では timestamp-query の値は丸められていない。
- （推測）ADR 0117 Context・gpu-lab README「Before measuring」が前提にした「フラグ無しの Chrome は 100 µs に丸める」は、
  少なくとも macOS の Chrome 154 では成り立たない（Chrome 側の緩和か、Metal バックエンドの扱いか、記録からは読めない）。
  Windows / Android のフラグ無しは未観測。
- 含意: 段 3 の「丸め誤差の上界 E ≤ 1%」の門は、丸められた記録が来ても受けられる保険として残る。前提の文言は ADR 0117 の
  追記と README で「丸められうる（記録の `quantized` が示す）」に弱めた。

## 4. 観察 2: フラグ無しの quick+ 1 本から作った表 vs 埋め込み `apple-metal-3`

`deno run -A tools/geometry-sweep/main.ts profile --from <記録> --id m2-flagless --opt-in` で生成（scratchpad・登録しない）。

| 欄                    | フラグ無し quick+ 1 本（2026-10-02）                 | 埋め込み `apple-metal-3`（full 5 本・`c358832b`）  | 一致 |
| --------------------- | ---------------------------------------------------- | -------------------------------------------------- | ---- |
| gemmRows ≤ 16         | 既定                                                 | 既定                                               | =    |
| gemmRows 17〜32       | `reg32x32r2x4w8` ×1.290                              | `reg32x32r2x4w8` ×1.31                             | =    |
| gemmRows 33〜64       | `reg64x32r4x4w8` ×1.478                              | `reg64x32r4x4w8` ×1.41                             | =    |
| gemmRows 65〜512      | 既定（3 段）                                         | 既定（3 段）                                       | =    |
| gemmRows > 512        | `reg128x32r8x4w8` ×1.186                             | `reg128x32r8x4w8` ×1.51（Chrome 153 の材料を含む） | =    |
| attention.qk          | `reg128x32r8x4w8` ×1.155                             | 同じ                                               | =    |
| attention.pv          | `reg64x64r4x8w8` ×1.745                              | 同じ ×1.66                                         | =    |
| conv2d.rows64         | `igemm64x64:wg8x16` ×1.322                           | `igemm128x64:wg16x16`                              | ≠    |
| conv2d.rows32         | `igemm128x64:wg16x16` ×1.460                         | 同じ                                               | =    |
| i8a8.linear           | `tile32x64r4x8w8x8k16` ×1.106                        | `tile64x64r4x8w8x16k16`                            | ≠    |
| i8a8.attentionQk / Pv | `tile32x64r4x8w8x8k16` ×1.158 / `tile16x128…` ×1.129 | 同じ                                               | =    |

- gemmRows と attention は 7 段 + 2 欄とも同じ幾何。比（> 512 ×1.19・①QK ×1.16・③PV ×1.75）は Chrome 154 の 10-01 の
  掃引単独の値（research K-70 §15: ×1.18 / ×1.16 / ×1.75）と一致 — フラグの有無で計測は変わっていない。
- 違うのは conv2d.rows64 と i8a8.linear の 2 欄。どちらも候補どうしの差が小さい欄（conv2d.rows64 の候補は ×1.27〜1.32 に
  3 つ・i8a8.linear は ×1.10〜1.11 に 4 つ）で、quick+ 1 本の揺れで入れ替わる範囲（推測）。full 5 本の表が正。
- 利用者アプリの観点: フラグ無しの quick+ 1 本（約 15 分）で、DiT が使う欄（> 512・①QK・③PV・i8a8 の attention）は
  登録表と同じ幾何に到達する。

## 5. 残り（段 7）

プロファイルタブで「表を作る」→ 再読み込み → GPU 設定「保存した表（照合して注入 — 一致しなければ自動）」→ 適用 → Anima で
N = 1 → JSON と状態行の文言。照合は adapter 4 欄（空文字も値）・指紋・caseSet で、保存時と同じ runtime なら一致するはず。
`a0ff1702` 以降のコミット（`b9245574`・`f483f612` 等）はカーネルの指紋と caseSet を変えていない（2 表の `--check` と
`tune_fingerprint_test` が緑）ので、サーバを再起動して最新の bundle で続けてよい。

## 6. 参照

- ADR 0117 検収表 段 7・追記（2026-10-02・独立レビュー）
- research [K-70 §15](2026-09-27-k70-metal-per-op.md)（M2 の自己 A/B・Chrome 153 → 154 の既定幾何の変化）
- 生成物（scratchpad・登録しない）: `m2-flagless.ts`
