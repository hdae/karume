# op 別 GPU 時間内訳の実測 — 最適化台帳の裏取り（統計波）

> NOTE: 時点スナップショット。数値は 2026-08-10 の実測（RTX 3080 Ti / Linux / Vulkan /
> Deno 2.9.4・main `5c5a76b` 相当）。手法は §5。台帳の出所は
> [2026-08-06-kernel-triage/](2026-08-06-kernel-triage/triage.md)、壁時計の基準は
> [2026-08-08-branch-adoption-perf.md](2026-08-08-branch-adoption-perf.md)。

発端は「最適化候補（HOST-006 / OP-008 / PLAN-011 / PLAN-012 / PIPE-013）の期待利得を
実測で順位づけしたい」。gpuTiming（ADR 0021/0032）の op 別内訳を実運用経路そのままで採り、
台帳エントリへ帰属させた。

**結論を先に**:

1. anima w8a8-1024（壁 13.945s / GPU 11.37s = 81.5%）の GPU 内訳は **DiT linear i8a8
   40.0% / DiT attention 3 本 23.2% / VAE conv2d 19.1%** で、この 3 群が 82.4% を占める。
   台帳の設計候補 4 件は**合計しても壁時計の 7% 台**（OP-008 ≈ −1.2% / PLAN-012 ≈ −0.8% /
   HOST-006 上限 −5.1% / PLAN-011 ≈ 0）。**桁を動かすには linear + attention のカーネル
   そのものに手を入れるしかない。**
2. **PLAN-011（timestep-only 部分グラフの CFG 共有）は既定構成で利得ゼロ** — 既定
   guidance 1 では uncond を 1 度も計算しない。ただし「772 dispatch/predict（DiT の
   26.6%）が GPU 時間の 0.12%」という**ホスト:GPU = 33:1** の部分グラフであることが
   分かった。dispatch 削減の候補としては生きている。
3. SBV2（FN4/w8）は **voice Session が GPU の 86.9%**、conv 族（conv1d +
   conv_transpose1d）が全体の 86.5%・front+voice に限れば **96.7%**。ただし
   **SBV2 の律速はホスト側**（壁 1.08s に対し GPU 0.42s・ホスト固定費 0.27s）。
4. 計測の装置代は **+41ms/step**（w8a8-1024 実測）で、ADR 0021/0032 が記録する
   370〜375ms/step より 9 倍小さい（当時からコード側が変わっている — 新実測として記録）。

## 1. anima の段別・op 別内訳

### 1.1 段別（GPU timestamp の ns 合計。壁時計ではない）

**w8a8-s16 / 1024×1024 / 8 step / seed 42 / guidance 1（manifest 既定）** — A/B 2 回、
再現性は DiT −0.04%・全体 +0.26%:

| 段               | run 数 |      GPU (ms) | dispatch | 全体比 |
| ---------------- | -----: | ------------: | -------: | -----: |
| text_encoder     |      1 |        194.88 |      908 |   1.7% |
| text_conditioner |      1 |         13.05 |      469 |   0.1% |
| transformer(DiT) |      8 |      8,697.15 |   23,256 |  76.5% |
| vae_decoder      |      9 |      2,464.52 |    2,727 |  21.7% |
| **合計**         |     19 | **11,369.59** |   27,360 |   100% |

壁時計（gpuTiming 無し）13.945s — 既存記録 13.9s と一致。wall − GPU の gap は DiT で
**89ms/step**（ホスト固定費 170.3ms/step の約 52% は GPU 実行と重なっている — §2 HOST-006）。

**f16 / 1024**: DiT 24,627ms（90.2%・w8a8 の 2.83 倍）/ VAE 2,518 / 合計 27,294ms・
壁 30.90s。dispatch は 23,256 → 17,832（quantize_rows 622/predict と attention permute
56/predict が消えるぶん）。

### 1.2 DiT の op キー別（w8a8-1024・8 step 合計）

| ms        | 構成比 | dispatch | キー                                            |
| --------- | -----: | -------: | ----------------------------------------------- |
| 4,551.529 | 52.33% |    3,632 | `linear:v3:i8a8:reg64x64v4:dp4a`                |
| 1,131.982 | 13.02% |      448 | `attention_pv:v2:i8a8:reg64x64v4:dp4a:s16`      |
| 1,035.749 | 11.91% |      448 | `attention_qk:v2:i8a8:reg64x64v4:dp4a:s16`      |
| 474.862   |  5.46% |      448 | `attention_stats:v1:f32:lastdim:safe:wg256:s16` |
| 452.763   |  5.21% |    4,976 | `quantize_rows:v1:f32>i8:pertoken:wg256`        |
| 343.766   |  3.95% |    6,968 | `strided:v1:f32:r4:wg256`                       |
| 228.712   |  2.63% |      904 | `rms_norm:v1:f32:lastdim:wg256`                 |
| 143.419   |  1.65% |    2,032 | `ew:v2:add:f32>f32:r3:wg256`                    |
| 118.131   |  1.36% |    1,352 | `ew:v2:mul:f32>f32:r3:wg256`                    |
| 73.007    |  0.84% |      224 | `ew:v2:gelu:f32>f32:r3:wg256`                   |
| 70.673    |  0.81% |      448 | `rope:v1:half:f32:wg256`（融合）                |
| 70.153    |  0.81% |      680 | `layer_norm:v1:f32:lastdim:wg256`               |
| 2.353     |  0.03% |      680 | `ew:v2:add:f32>f32:r2:wg256`                    |
| 0.046     |  0.00% |       16 | `silu:v1:x-sigmoid:f32:wg256`（融合）           |

**linear + attention 3 本で DiT の 82.7% = 全生成 GPU の 63.3%**。f16 版も同型
（linear wf16 62.9% / attention 3 本 32.9%）。

### 1.3 VAE decoder（9 タイル合計）

conv2d 2 本（igemm64x64 1,332.0ms + igemm32x64 834.3ms）で **VAE の 87.9% = 全体の
19.1%**。OP-009（channel L2 = mul→sum→sqrt→clamp_min→div の 30 鎖）相当は 196.0ms =
VAE の 7.95%・全体の 1.72%。text_encoder / conditioner は linear が 92.7% / 85.5%。

### 1.4 融合の確認

`lastRunFusions` 全 run 合計（anima 1024）: rope 503 / silu 305 / upsample2x 27 /
identityExpand 160 — ADR 0040 §4 の表と完全一致。SBV2 は identityExpand 210 / 他 0。
**融合は実運用どおり効いた状態の計測**である。

## 2. 台帳エントリへの帰属と期待利得

帰属は実行計画（`planGraph` を実 shape で組む）から各ノードの入力依存集合で段を分類し、
ノード→計測キー→仕事量モデルで実測 ns を按分した。**モデルは全キーで dispatch 数が実測と
一致**（w8a8 14 キー・f16 13 キー、合計 23,256 = 23,256）。PLAN-011 の 856 node /
PLAN-012 の 308 node という分類結果も台帳の静的集計と完全一致。

| エントリ                               | 実測（w8a8-1024・8 step）                                                                                                                                                              | 期待利得（壁時計比）                                                                                                                                                                 | 確度                             |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| **OP-008**（adaLN 85 鎖）              | `layer_norm→mul→add` はちょうど 85 鎖（全て `[1,4096,2048]`）。**201.7ms = 全 GPU の 1.77%**・dispatch 255/predict                                                                     | 帯域 3:1 で GPU −134.5ms + dispatch −1,360 で CPU −79.7ms（露出 ≈ −33ms）→ **≈ −0.17s = −1.2%**                                                                                      | GPU は実測按分・利得は帯域モデル |
| **PLAN-012**（encoder-only 308 node）  | dispatch 308/predict・**GPU 68.0ms = 全体の 0.60%**。切り出し面 56 値 / 224.00MiB（f32）                                                                                               | 7/8 削減で **≈ −0.11s = −0.8%**（f16 −0.6%）。**VRAM 代償 +224MiB/condition**                                                                                                        | 実測按分・利得は上界             |
| **PLAN-011**（timestep-only 856 node） | dispatch **772/predict = DiT の 26.6%** だが **GPU 10.8ms/8step = 0.12%**。ホスト ≈45ms/step vs GPU 1.35ms/step = **33:1**                                                             | **既定 guidance 1 では利得ゼロ**（uncond 非実行 — DiT run は 8 本だけ）。guidance 2 でも GPU 0.06% / 露出 CPU ≈ −150ms。**CFG 共有ではなく dispatch 削減の候補として読み替えるべき** | 実測（guidance 2 も実走）        |
| **PIPE-013**（cond/uncond 直列）       | guidance 2 実測: DiT run 16 本・GPU 2.0135 倍で完全直列。`#chain` 直列は text 系の `Promise.all` で直接観測（wall 333→501ms の逐次）                                                   | CFG batch の利得はホスト固定費（guidance 2 で ≈ −170ms CPU/step）。**text 系の並列化は GPU の 1.8% しか触れず費用対効果なし**                                                        | 実測                             |
| **HOST-006**（params/bind group）      | 定常 step: createBuffer 63.9 + writeBuffer 40.0 + createBindGroup 44.4 + destroy 19.7 + submit 2.0 = **170.3ms/step（58.6µs/dispatch）**。台帳の旧実測（139/52ms）は現行でも同桁で再現 | ホスト費用の約 52% は GPU と重なっており、露出 gap は **89ms/step が天井 → 上限 ≈ −5.1%**。executable plan 単位の再利用で params 系 123.6ms/step が主対象                            | 実測（device メソッドラップ）    |

初回 predict は +287ms（i8 重み 1.9GiB のアップロード）。DiT の pipeline はグラフ全体で
**14 本**しかなく、pipeline 並列 prewarm（PIPE-015）の余地は 7.6ms — 対象外にしてよい。

> 訂正（2026-08-10・C 波の実 IR 精査）: ①OP-008 の鎖は隣接 3 ノードではなく **窓 7 の
> 非隣接形**（layer_norm と mul の間に shift/scale/gate の reshape 3 本 + `1+scale` の add が
> 挟まる）。鎖の実 dispatch は `1+scale` の add を含め **340/predict**（表の 255 は
> layer_norm/mul/add の 3 op のみの計数）。②表の下で触れた `ew:v2:add:f32>f32:r2`
> （680 dispatch）は adaLN 鎖の一部ではなく、変調ベクトル生成 `add[1,6144]` ×85。GPU 時間
> 201.7ms の帰属自体は変わらない。設計評価の正本は C 波レポート（実装波の ADR に統合予定）。

## 3. SBV2 の支配 op（FN4 / w8 / seed 0 — WAV 門と同条件・sha256 参照一致）

| Session                 |   GPU (ms) | dispatch |    全体比 |
| ----------------------- | ---------: | -------: | --------: |
| text_encoder（DeBERTa） |      45.31 |      887 |     10.7% |
| front                   |      10.00 |      758 |      2.4% |
| **voice**               | **366.28** |    1,258 | **86.9%** |
| 合計                    |     421.59 |    2,903 |      100% |

- op ファミリ別: **conv1d 70.2% / conv_transpose1d 16.2% = conv 族 86.5%**（front+voice
  に限れば **96.7%**）。linear は全体 9.5% だが、その 99% は DeBERTa 内（front+voice では
  0.10%）。
- ADR 0039 §5 の「conv1d が 86〜90%」は **conv 族の全体比としては成立・conv1d 単体
  （70.2%）としては過大**。「linear は実質 0」は front+voice では成立するが、DeBERTa を
  含むパイプライン全体では成立しない（GFLOP 集計由来の主張 — TS 実測では読み替えが要る）。
- 単発最重量: `conv_transpose1d:v1:f32:gather:wg256:wi8` が voice で **5 dispatch /
  68.43ms = 13.7ms/dispatch**（conv1d direct は 1.18ms/dispatch）。
- **SBV2 の律速はホスト側**: 壁 1.08s に対し GPU 0.42s・ホスト固定費 0.27s（front は
  GPU 10ms に対し wall 125ms）。pipeline 生成 89 本 13.4ms + shader module 21.8ms も
  初回コストとして無視できない。**anima と逆の形**で、SBV2 の体感短縮はホスト側
  （dispatch 数・構築費）が効く。

## 4. 付随して確定した事実

- **gpuTiming の装置代は 41ms/step**（w8a8-1024: 壁 13.945 → 14.62〜14.66s）。
  ADR 0021/0032 の記録 370〜375ms/step と 9 倍差（当時の実測も正 — コード側が変わった。
  submit は 24 → 43/step に増える）。既定 OFF の裁定を覆す差ではない。
- **gpuTiming on でも出力ビット不変**（PNG 門 3 構成 + WAV 門の sha256 が全て参照一致 —
  ADR 0021 検証節の主張を実運用経路で追認）。
- timestamp の非単調（clampedNegativeSamples）は全 run で 0。再現性は A/B で ±0.5% 以内
  （例外は初回 GPU 仕事の text_encoder ±8% — クロック立ち上がり）。
- **読み方の罠**: `strided` のキーは rank 固定（`codegen/strided.ts` の `STRIDED_RANK`）
  なので `:r4` に rank 2 の slice も混ざる — キー名から rank を読まない。ホスト計測を
  run 単位で差分すると Session 構築時の重みアップロードが「次の run」に落ちる — 定常値は
  初回 run を除いて取る。

## 5. 手法（再現用）

- **production 無変更**。駆動台本側で `packages/runtime/src/runtime/executor.ts` の
  `Session` を直 import し、`Session.prototype.run` をラップして解決直後に
  `this.diagnostics()` を読む（`#enqueue` の `#chain` 登録順により、次 run の
  `resetTiming()` より先に読めることが保証される — 実測でも全 run で
  timingDispatchCount = submitDispatchDelta）。パイプラインは `generate` をそのまま
  呼ぶ（経路の乖離ゼロ・門 sha256 一致で証明）。
- GPU は `acquireGpu({ gpuTiming: true })` を `options.gpu` へ注入。ホスト費用は
  `gpu.device` のメソッド（createBuffer / createBindGroup / queue.writeBuffer 等）を
  台本側でラップ（ラッパ有無の壁時計差はノイズ内）。
- クールダウン規約: 各 run 前にアイドル +5°C 以下まで待機。GPU 併走ジョブなし。
- 台本と生データ JSON は scratchpad（揮発）。本ドキュメントの表が写し。

## 6. 未解決（openQuestions）

- OP-008 の利得（−1.2%）は帯域モデルの推定 — 実装するなら融合カーネルの実測で確定する。
- HOST-006 の「executable plan 単位の再利用」は生成数の実測（2,949 createBuffer/step）
  まで — 寿命・再利用可否の設計はこれから。
- linear i8a8（52.3%）と attention（30.4%）のカーネル最適化は本記録のスコープ外 —
  別途マイクロベンチ波が要る（attention_stats の 29.4M workgroup/step という幾何は
  最初の観察対象候補）。
- 観測面の恒久化（パイプラインから `Session.diagnostics()` へ到達する席が無い）は裁定待ち。

## 7. 追記（同日・HOST-006 の第 1 波着地と残余の再実測）

カーネル 3 波後（w8a8-1024 壁 13.9 → 10.6s）にホスト露出が相対拡大したため再実測し、
HOST-006 の第 1 波を実装・着地した（`9f47a04` = params の内容アドレスキャッシュ・
`2d1dad1` = bind group layout の PipelineCache 保持）。

- **着地前の露出 gap（run 壁 − GPU・計測込み 1 走）**: w8a8-1024 で 2.26s / run 壁 10.32s。
  §2 の「上限 −5.1%」は壁 13.9s 時点の比率であり、この時点で失効していた。
- **params キャッシュ**: params の全バイトは (グラフ, attrs, 解決済み shape) の純関数という
  実読結論に基づき、usage + 全要素連結をキーに Session 常駐（weights アリーナ所有・一度
  書いたら不変）で共有。定常 step の新規生成は 0 本（診断 `lastRunParams` と挙動テスト
  2 本で固定）。
- **検収 ABBA（回文・門緑・各 2 走同値）**: w8a8-1024 10.6 → **10.3s（×1.03）**・
  f16-1024 20.2 → **20.0s（×1.01）**。§2 の params 系 124ms/step に対し壁へ出たのは
  ≈37ms/step — **約半分は GPU 実行と重なっていた**ため（overlap 分は消しても壁に出ない）。
- **着地後の残余 gap**: 1.72s / run 壁 9.99s（計測込み）。装置代 ≈0.33s + 初回アップロード
  ≈0.29s を引いた実質 ≈**1.1s ≈ 壁の 11%**。残余の内訳候補は createBindGroup 44.4ms/step・
  plan/fusion の毎 run 再計算・入力アップロード・encode ループの JS 残差。
- **帰結**: 残余はバッファ同一性の固定（= PreparedExecutionPlan）なしには取れない。
  次波は plan/fusion の Session キャッシュ + transient slot 固定 + bind group 再利用を
  土台に、staged execution（large-designs D — 特に E の prepared cross-attention K/V を
  量子化形 +57MiB で）を段階導入する（ユーザー裁定 2026-08-10: b と D は同一設計波）。

## 8. 追記（同日・PreparedExecutionPlan 波 1 の着地 — 導出相は壁に出ていなかった）

エンコード層を 2 相分離し（`32dad19`）、導出済み計画を bindings キーで Session 常駐させた
（`b2e6ce0` — 設計の正本は ADR 0042）。同一 bindings の 2 run 目以降は planGraph / planFusions /
WGSL 生成 / params キー構築 / 契約検査を丸ごと飛ばす。

- **検収 ABBA（回文・門緑・各 2 走）: 壁時計は中立** — f16-1024 19.9/19.9 → 19.9/20.0s・
  w8a8-1024 10.4/10.3 → 10.4/10.5s（差はノイズ幅内）。
- **§7 の残余内訳候補の帰属が確定**: 「plan/fusion の毎 run 再計算・JS 残差」は露出 gap に
  ほぼ寄与していなかった（GPU と完全重畳）。露出の本体は **createBindGroup 44.4ms/step +
  アリーナ簿記（allocStorage/retain/release ×~3,280/run）+ 転送系**。波 2（transient slot 固定 +
  bind group のレシピ焼き込み）がこの全てを対象にする。
- **観測点の移設**: 定常 step（prepared ヒット）では導出相が走らないため `lastRunParams` は
  {0,0} になる（§7 の「診断 lastRunParams で固定」は初回導出 run にのみ当てはまる）。params
  キャッシュの門は gpu_params_cache_test が「run 内重複 + prepared 追い出し後の再導出 run」で
  観測する形へ移設済み。ヒット/ミスの観測点は `lastRunPrepared`。
- **VRAM 実測（設計材料・onRunDiagnostics 経由）**: DiT の run 中実確保は w8a8-1024 で
  1062.5MiB / f16-1024 で 1461.7MiB（live ピーク 712.5 / 1184.5MiB）・VAE タイル 861.9MiB。
  この量は現行でも毎 run 中に存在するため、slot を Session 寿命へ昇格（活性 1 signature ぶん）
  しても新しい VRAM ピークは生まれない。
