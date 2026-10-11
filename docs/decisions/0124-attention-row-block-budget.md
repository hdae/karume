# 0124: attention の行ブロックの一時に容量の予算（`attentionRowBlockMaxBytes`）を置く

- Status: accepted（利用者裁定 2026-10-11 — 裁定 6 = a〈行ブロック分割を利用者の予算でも細かく割れるようにする・結果のビットは
  変えない〉、設計 v2 の問い 33〜35 = 推奨案 a。設計の正本は `.claude/reviews/2026-10-04_de127de1/DESIGN-4-6-8-v2-2026-10-11.md`
  の d6〈git 追跡外〉）
- Date: 2026-10-11
- 関連: ADR [0060](0060-row-block-attention.md)（分解 attention の行ブロック — ビット同一の根拠）/
  [0023](0023-fused-attention.md)（融合 attention）/ [0067](0067-autoregressive-attention-vocabulary.md) 決定 7（states 形の行ブロック）/
  [0093](0093-transient-liveness-packing.md)（中間の配置と見積り — 追記 2026-10-10）/
  [0070](0070-shard-loading-admission.md) 決定 5（見積りは空き VRAM と比べない）/
  [0022](0022-gemm-register-blocking.md)（実行時オートチューン禁止）/ [0058](0058-numerics-opt-in-contract.md)（数値の opt-in 席）/
  [0121](0121-wan22-ti2v-5b.md)（Wan2.2 — 開発機の 1280×704×121 の実測）

## Context

### 行ブロックの今の決め方

スコア行列 S を実体化する attention は 3 つの形がある。融合 attention（ADR 0023）・states 形（ADR 0067）・分解 attention の
`rowBlockAttention`（ADR 0060）である。どれも S をクエリ行のかたまり（行ブロック）に分けて順に計算する。枚数は
「S 1 枚と行統計 1 枚のそれぞれが device の `maxStorageBufferBindingSize` に収まる最小枚数」で、device の束縛上限だけで決まる。

束縛上限がほぼ 2 GiB の機（RTX 3080 Ti・Arc B570）では、Wan2.2 の 1280×704×121（S = 27,280）の実用席で S 1 枚が約 1.96 GiB
（17 枚）になる。開発機（RTX 3080 Ti・12,288 MiB）の DiT の段の VRAM の山は 11,361 MiB で、余裕は約 0.9 GiB だった（ADR 0121
追記「開発機の 1280×704×121」）。

### 何を変えたいか

行ブロックを束縛上限より細かく割れば、attention の瞬間に生きている一時が縮み、VRAM の山が下がる。効き目は「入らない形を入れる」
ことではなく、「10〜12 GB 級の機に余裕を作る」ことである。832×480 は受理の外で、121 フレームは開発機で完走済みである。

### 決め手になる事実

- **WebGPU は空き VRAM を見せない**。見積りは空き側と比べない（ADR 0070 決定 5）。実行時オートチューンは禁止（ADR 0022）。
  いつ速度を捨てて容量を取るかを runtime や pipeline は判断できない。判断できるのは利用者（アプリ）だけである。
- **行ブロックはビット同一**。行窓は担当行の割り当てだけを変える（ADR 0060 決定 3）。したがって予算は数値の席ではなく、
  `submitPolicy` と同じ資源の政策になる。
- **速度の代価は Wan の規模では未測定**。小さい形では後退の実測がある（Anima の形 `[1,16,4096,128]` の a8 / s16 で 1 分割
  6.46 / 6.98 ms・8 分割 8.42 / 10.90 ms — [research 2026-09-10](../research/2026-09-10-codex-mtp-optimization.md)）。分割の方が
  速かった例もある（ADR 0060 の検収・S = 750 の分解経路で 2 枚 46.8 ms 対 1 枚 48.9 ms）。

## Decision

### 1. ノブの意味 — 行ブロックの一時 1 本ごとの上限（合計ではない）

- `SessionOptions.attentionRowBlockMaxBytes`（任意・正の安全な整数・バイト）を足す。
- 行ブロックの上限は **min(device の granted な `maxStorageBufferBindingSize`, 予算)** にする。
- 上限は**一時 1 本ごと**で、合計ではない。S 1 枚と行統計 1 枚はそれぞれ上限以下で、同時に生きる合計は上限を超えうる。
  i8a8 の量子化の一時（行ブロックのループの外）と他の中間は、この値で縛らない。
- 導出は純関数 1 本（`rowBlockLimit` — `packages/runtime/src/runtime/fusion-rules/row-block-attention.ts`）にまとめる。
  融合 attention・states 形・readonly の states 形・分解の `rowBlockAttention`・見積りは、全てこの 1 本を通す。
- 中間バッファの配置の上限検査（ADR 0093 決定 5）は granted 値のままにする。FFN の中間や埋め込みの束縛は縛らない。

### 2. 既定は今と同一

欄を省くと上限は束縛上限のままになる。枚数・パイプラインキー・WGSL・dispatch 列・backing は、欄を足す前と 1 バイトも変わらない。
既存の sha 行・WGSL スナップショット・census はそのまま通る。

### 3. 資源の政策で、数値の席ではない

quant 席・manifest の `session`・hub には入れない。出力のビットを変えないからである。census の除外リスト `NonNumericField`
（`packages/runtime/tests/helpers/pipeline-census.ts`）に名前を足し、「数値を変えない欄」と型の上で宣言する。

### 4. ビット同一の根拠

構造の根拠は 3 つある。

1. 行窓は担当行の割り当てだけを変える（ADR 0060 決定 3）。1 出力要素の K 縮約順と丸めの並びは動かない。
2. 行統計と softmax は行の中で閉じる。行を切っても 1 行の演算列は変わらない。
3. i8a8 の量子化は行ブロックのループの外にある。量子化の値は枚数に依らない。

n = 1 から n ≥ 2 への切り替えは行窓の別キー・別 WGSL に替わる。この境目のビット同一は、テストで縛っている変種の列挙で担保する。

| 形                           | 変種                                                                    | テスト（強制分割と予算分割の両方で 1 枚実行と Uint32 一致） |
| ---------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------- |
| 融合 attention               | f32・f32 + mask・s16・a8・a8 + s16（GQA 形を含む）                      | `packages/runtime/tests/gpu_attention_row_block_test.ts`    |
| 融合 attention（f16 計算）   | f16・f16 + mask（GQA 形を含む・`shader-f16` を要る — 別の `Deno.test`） | 同上（本 ADR で追加）                                       |
| 分解 attention（9 ノード窓） | f32                                                                     | `packages/runtime/tests/gpu_row_block_attention_test.ts`    |
| states 形                    | f32 だけ（数値変種と組むと fail loudly — ADR 0058 決定 3）              | `packages/runtime/tests/gpu_state_execution_test.ts`        |

予算で割った走行では、枚数が 2 以上であることと、中間のピーク（`peakTransientBytes`）が予算なしより厳密に小さいことを同時に見る
（予算が効かないまま同じ経路を 2 回撃つ形を緑にしない）。

### 5. 落ちる時点

- クエリ 1 行ぶんの一時が予算を超える形は、**その形の計画を最初に導く run** で、予算を名指して fail loudly にする。
  Session の構築時には形が記号のままで、枚数が決まらないからである。Wan では重みの読み込みの後に落ちる。readonly の states 形
  （1 行 = decode）でも、長い容量に小さい予算を渡すと decode で落ちる。
- 構築時の値域検査（`sessionOptionsViolation`）は、構築時に分かる不正値（0・負・非整数・NaN・数値でない値）だけを落とす。
- 予算違反を重みの読み込みの前に知る手段は、見積り（決定 7）に同じ値を渡すことである。

### 6. 強制枚数との優先

テスト専用の `ROW_BLOCK_SPLIT` で枚数を強制したときは、その枚数が min(束縛上限, 予算) に収まるときだけ通す。収まらなければ
予算を名指して fail loudly にする（強制で予算を緩めない）。予算を渡さない既存の強制テストの意味は変わらない。

### 7. 見積り

- `EstimateOptions.attentionRowBlockMaxBytes` を足す。値域検査は Session 構築と同じ文言で、行ブロックを持たないグラフでも見る。
- 既存の勘定（states 形の S と行統計・分解の行ブロック attention の窓の一時）は、実行と同じ導出（決定 1）で枚数を決める。
  1 行ぶんに満たない予算は、実行の計画導出と同じ文言で落ちる。
- **融合 attention op のノード内一時（S・行統計・i8a8 の量子化の中間）も見積りに数える**（裁定 34 = a）。実行と同じ純関数
  （`planAttentionBlocks`・`attentionRoute`・`attentionScoreElementBytes`・`attentionI8a8TempBytes`）で数える。これは予算の
  有無に関わらず、融合 attention op を持つ全系列の見積りの値を増やしうる既定の挙動の変更である。そこで、先に既存の呼び手の
  前後の表を [research 2026-10-11](../research/2026-10-11-fused-attention-estimate-before-after.md) に書き、勘定の変更と
  `unaccounted` の文言の変更を別のコミットにした（表の結果: Anima の段ごとの見積りと Gemma の見積りは前後で不変）。
- 実行が拒否する組（mask × a8・GQA × a8・i8a8 の i32 縮約の門・states 形 × f32 以外の席）には見積りも数を返さない。判定は
  実行と同じ純関数（`fusedAttentionViolation` / `stateAttentionVariantViolation` —
  `packages/runtime/src/runtime/recipe-builders/attention.ts`）で、文言も同じ。
- `EstimateOptions` は `attentionCompute` / `attentionScoreStorage` も受ける。省略は runtime の既定と同じ f32 で数える
  （`stateAttentionReduce` と同じ前例）。

### 8. ビット同一の門は 2 層

- **層 1**: DiT の e2e（`packages/models/tests/e2e_wan_ti2v_dit_test.ts`）で `SessionOptions` を直接渡す u32 の A/B。同じ入力で
  予算なし / 512 MiB / 256 MiB を回し、出力のビット列を比べる。形は 17 フレーム（既定のレーン）と 121 フレーム（opt-in
  `KARUME_WAN_TI2V_121F=1`）で、実用席と参照席の両方を回す。行ブロックの期待枚数は self / cross 別・席の格納幅つきの式で
  runtime と独立に導き、行窓の ①QK の dispatch 本数がそれと一致し、予算なしと違うことを同時に見る。models の口に依らない。
- **層 2**: pipeline の sha 行との照合。予算を pipeline に届ける口（models）ができたときだけ行う。口を出さないと決まった場合、
  pipeline に配線が無いので出力は変わりえず、層 2 は要らない。

### 9. online softmax（S を作らない opt-in の高速パス）との境界

この ADR の外に置く。決めるのは境界の 2 点だけである。

1. このノブは資源の政策で、quant 席に入れない。
2. S を作らない経路が来たときに、このノブを無視するか拒否するかは、その経路の ADR で決める。

その経路の前提になる段別の GPU 時間（QK・行統計・PV）は、段 4b の計測モードで同じ走行のついでに採る。

### 10. 公開面

- runtime の `SessionOptions` / `EstimateOptions` に任意欄を 1 つずつ足すだけを先に行う。`mod.ts` の輸出名は増えない。
- models（`WanPipelineOptions`・gpu-lab の Wan タブ）に出すかは、段 4b の実測（減り幅と速度の代価）を見てから決める（裁定 33 = a）。
- Wan の既定に予算を入れるかは、段 4b の数字を添えて後で問う。
- 10 GB 級の機で「61 フレーム以上は非対応」（limitations）は変えない（裁定 35 = a — 確かめる機が手元に無い）。

## 採らなかった案

- **runtime の既定を固定の予算にする（例 1 GiB・全系列）**: 空き VRAM が見えないので、余裕のある機でも一律に分割の代価を払う。
  代価は Wan の規模では未測定で、小さい形では後退し、分解経路では速くなった例もある。どちらに転ぶか分からないまま、Anima・Irodori
  など全系列の既定の性能が黙って変わる（ADR 0058 決定 3 の精神に反する）。固定値の根拠は 1 機の実測にしか置けない。
- **総 VRAM の予算から行ブロックの上限を逆算する（pipeline が見積りで決める）**: 融合 attention の一時を見積りが数えるまで
  成立しない。backing の実測は同じ 121 フレーム・同じ 17 枚で Deno 5.97 GiB 対 Chrome 4.80 GiB と 1.17 GiB 違い、原因が
  分かっていない（ADR 0121）。逆算した値は機によって外れうる。外れると out-of-memory で落ち、最上段の公開面の意味が内部の近似の
  精度に縛られる。
- **`planBackingBudgetBytes` の流用**: backing の保持本数という別の意味と混ざる。
- **device の束縛上限そのものを絞る**（`acquireGpu` の requiredLimits を下げる）: FFN の中間や埋め込みの束縛まで縛る。
- **`ROW_BLOCK_SPLIT`（枚数）を公開のノブに昇格する**: 同じ枚数でも形ごとに 1 枚の大きさが変わる。バイトの上限に替える。

## Consequences

- runtime の `SessionOptions` / `EstimateOptions` に任意欄が 1 つずつ増える（追加だけで、既存の呼び手は壊れない）。hub・manifest・
  quant 席は変わらない。
- 予算をかけると dispatch が増える（融合 attention は 1 枚あたり 3 本）。Wan2.2 の 121 フレームの実用席で 512 MiB なら、1 forward
  あたり約 4,590 本増える（計算値: self (67 − 17) × 3 × 30 層 + cross (2 − 1) × 3 × 30 層）。速度の代価は段 4b で測る。
- 予算をかけると cross-attention（テキスト 512 列）も割れる（121 フレームの実用席で 512 MiB なら 2 枚 — 計算値）。予算を
  指定した利用者の環境では、他の系列（Anima・Irodori など）の n = 1 から n ≥ 2 への境目が新たに踏まれる。強制分割の parity
  （決定 4）が形を横断して担保する。
- 効き目は推測の幅である（段 4b の実測で置き換える）。backing の式 `max(Sblk + k·A, 2·S·ffn·4 + 3·A)`
  （[research 2026-10-04](../research/2026-10-04-wan22-ti2v-5b-karume-gap.md) §3.2）に当てると、予算 512 MiB の減り幅は
  0.95〜1.46 GiB になる（推測 — 下端が Chrome の点、上端が Deno の点）。Chrome の点では FFN の項が下限を決め、1 GiB でも
  512 MiB と同じだけ減る（推測）。
- 落ちる時点が遅い（決定 5）。予算違反は Wan では重みの読み込みの後に落ちる。
- 見積りの拡張（決定 7）で、既存の全系列の `workspaceBytes` / `peakAccountedBytes` が増える。Anima の先回りの退避（常駐 DiT を
  畳む判断）が早まりうる（推測 — 出力のビットは変わらず、速度だけが変わりうる）。

## 検収（門の形）

- ホスト（GPU 不要）: `rowBlockLimit` の 3 分岐・Wan2.2 の形の枚数（121 フレームの実用席の self で予算なし / 512 MiB / 256 MiB が
  17 / 67 / 134 枚 ほか）・1 行に満たない予算の fail loudly（予算を名指す）・強制枚数との併用・値域検査・見積りの手計算の定数
  （ADR 0093 追記の例に予算 64 MiB を掛けると 3 枚・`workspaceBytes` 92,057,472 B）。既存の WGSL スナップショットと census は不変。
- 実 GPU（決定 4 の表）: 強制分割と予算分割の両方で 1 枚実行と Uint32 一致・予算つきの枚数が 2 以上・ピークが厳密に小さい。
- 層 1（決定 8）: DiT の u32 A/B。
- 見積りの拡張: 他の融合が成立しない融合 attention 1 ノードのグラフで、`workspaceBytes` が実行の `planBacking.residentBytes` と
  予算あり・なしの両方で厳密に一致する（`packages/runtime/tests/gpu_estimate_test.ts`）。

## 結果

（未記入 — 段 4b の実測〈121 フレームの実用席で予算なし / 1 GiB / 512 MiB / 256 MiB の backing・VRAM の山・1 forward の壁時計・
段別の GPU 時間〉の後で追記する）
