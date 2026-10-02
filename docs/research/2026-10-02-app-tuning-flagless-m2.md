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

## 7. 追記: フラグ無しの M5（2026-10-02・quick+ 掃引と Anima の A/B）

利用者が M5（macOS・Chrome 154・フラグ無し）で quick+ の掃引（`geometry-sweep-browser-2026-10-02T11-02-17.340Z.json`・checkout
`de0c24a8`）と Anima タブの A/B ボタン（`anima-residency-browser-f16+dit8-a8-attn8-s16-2026-10-02T11-34-41.119Z.json`・既定 quant +
f16 × default / auto × 3）を回した記録。段 7 の「保存した表（照合して注入）」の経路はまだ通していない（A/B ボタンは別の経路）。

- 掃引: 57 ケース・521 行・677 s・失敗 0・不一致 0・**timestamp は非量子化**（2,605 round 中 100 µs の倍数 0 — M2 と同じ）。
  `candidateKernels` が載っている（bundle が `b9245574` 以降）。既定の再測定比は 5 ケースが 0.9〜1.1 の外（0.874〜1.179）で、
  生成器がその掃引の比の材料から外した（M5 は熱の揺れが大きい — research K-70 §14 と同じ観察）。
- この 1 本から作った表（opt-in・登録しない）: **conv2d.rows32 だけ採用**（`igemm128x128:wg16x16` ×1.116）で他の 13 欄は既定。
  full 2 本からの試走（research K-70 §15: 17〜32 ×1.246・33〜64 ×1.538・conv2d rows32 ×1.097 以外は既定）より採用が少ないのは、
  quick+ 1 本で揺れが大きいため（推測）。「M5 は既定が最良」（ADR 0115 追記決定 7）と整合。
- A/B（自動 = 既定・表は当たらない — M5 に登録表は無く、フラグ無しでは description も空）: 区間 A と B は同じ既定で走り、PNG sha は
  quant ごとに 6 / 6 一致（既定 quant `0df85f770e9a`・f16 `dbad691e97ac`）。DiT 段 GPU は既定 quant 22.10 s・f16 17.17 s
  （512²・常駐 2 回目以降の中央値）— **M5 でも f16 が既定 quant（a8）より 1.29 倍速い**（M2 は 1.58 倍・research K-70 §12 の
  「Metal では a8 が損」の第 2 の実例）。VAE 段 GPU 0.87 s・text_encoder の壁時計 17〜19 s はネットワーク越しの重み取得。

## 8. 追記: 段 7 の本体 — フラグ無し M2 で「保存した表（照合して注入）」が通った（2026-10-02）

`outputs/bench-browser/anima-residency-browser-f16+dit8-a8-attn8-s16-2026-10-02T12-14-04.178Z.json`（M2・Chrome 154・フラグ無し・
checkout `de0c24a8`・2 行 = 既定 quant と f16 を 1 枚ずつ）。

| 項目                                   | 値                                                                                                         |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `geometryProfileRequested`             | `saved:apple-metal-3`（利用者がプロファイルタブで付けた id — 埋め込み表と同名・下の注意）                  |
| `geometryProfileInjected`              | あり（provenance.sweep = §2 の quick+ 掃引・candidateSet quick+・adapter 4 欄は空の device / description） |
| 各段の `geometryProfile`               | `apple-metal-3`（注入された表で走った）                                                                    |
| 今の runtime での照合（Deno で再計算） | caseSet `c1060d8829f3b9de` = ・kernels `cb8b4a773580567c` = ・`geometryProfileMismatch` → 一致             |
| PNG sha256                             | f16 `041027e63559`・既定 quant `0a5695470e4a` — **フラグ有りで埋め込み表を当てた K-70 §15 の記録と同一**   |
| DiT 段 GPU                             | f16 42.10 s・既定 quant 68.25 s（§15 のフラグ有り + 埋め込み表の 42.02 / 68.42 s と同じ）                  |

- **M5 も同じ流れが通った**（`…2026-10-02T12-05-02.121Z.json`・2 行）: `saved:apple-metal-3`（利用者が M5 の表にも同じ id を付けた）・
  provenance.sweep = §7 の M5 の quick+ 掃引・各段はその表で走り、PNG sha は f16 `dbad691e97ac`・既定 quant `0df85f770e9a` で
  §7 の既定幾何の A/B と**同一**（M5 の表は conv2d.rows32 以外が既定なので期待どおり）・DiT 段 GPU 17.1 / 22.1 s も同じ。
  M2 と M5 の記録は adapter の 4 欄が同じ（description 空）で見分けられず、provenance.sweep と DiT の時間で判別した。
- アプリの流れ（掃引 → 生成 → 保存 → 再読み込み → 照合 → コールバックで注入）は、description が空のフラグ無し Chrome でも通った。
  描画結果は埋め込み表（full 5 本）と**ビット同一**で、quick+ 1 本の表（conv2d.rows64 / i8a8.linear が違う）でも DiT が使う欄は同じ
  幾何なので時間も同じ。
- 注意（隣接・未修正）: 生成した表の id を利用者が `apple-metal-3` と付けたので、行の `geometryProfile` 列だけでは埋め込み表と
  見分けられない（provenance で分かる）。gpu-lab の id の既定値が埋め込み表の id と衝突しない形（例 `<vendor>-<arch>-generated`）に
  するのが筋 — backlog に起票。
- ADR 0117 検収 段 7 の残り = フル verify（段 3 の収束後に回す）。

## 6. 参照

- ADR 0117 検収表 段 7・追記（2026-10-02・独立レビュー）
- research [K-70 §15](2026-09-27-k70-metal-per-op.md)（M2 の自己 A/B・Chrome 153 → 154 の既定幾何の変化）
- 生成物（scratchpad・登録しない）: `m2-flagless.ts`
