# 大規模最適化・互換性設計候補（2026-08-06）

> NOTE: 時点スナップショット。ここにある項目は原則未実装であり、採用決定ではなく検証可能な設計候補である。
>
> NOTE: 本ファイルは**参照実装ブランチ（`codex/kernel-quick-fixes`）での設計候補記録を原文のまま持ち込んだもの**で、本リポ main の実装状態とは一致しない（実測は Apple M2 / Metal と RTX 3080 Ti / Vulkan）。
> main への採否と本リポでの再実測は [2026-08-08-branch-adoption-perf.md](../2026-08-08-branch-adoption-perf.md) が正本。

## 優先順位

1. **沈黙誤値をカナリアで封じる**: dp4a と共有メモリを feature 列挙だけで信用しない。
2. **Metal 向け GEMM の形を増やす**: 1 種類のタイルを全 GPU に強制しない。
3. **dispatch と全 tensor pass を減らす**: RoPE、norm、再量子化、permute。
4. **denoise 間・CFG 間の不変計算を再利用する**。
5. **ホスト固定費を executable plan へ畳む**: pipeline、params、bind group。
6. 最後に subgroup など optional feature の高速変種を足し、常に portable fallback を残す。

## A. 実走カナリアとバックエンド能力表

### 問題

WebGPU feature / limit の列挙は「その命令列が正しく実行される」証明ではない。既存の
Metal 調査では WGSL として合法な動的 vector component write が MSL 経路で沈黙 no-op になった。
同じ種類のリスクが packed dot、workgroup memory、f16 計算にもある。

### 提案

`GpuContext` に「列挙された能力」と「実走確認済み能力」を分けて持たせる。

- `dot4I8Packed`: 既知の正負・飽和しない境界値を dp4a と scalar emulation で比較。
- `workgroupVectorComponentWrite`: 現在は静的 switch 回避を常用する。再導入せず診断用途だけ。
- `shaderF16`: 既存カナリアを維持。
- 大きな workgroup / shared tile: pipeline validation だけでなく既知解の短い GEMM を実走。
- 結果は device lifetime で 1 回だけ評価し、adapter 名の文字列分岐には使わない。

失敗時は feature を無効化して portable kernel へ落とす。高速変種しか実装されていない場合は
loud な例外にし、沈黙継続しない。

### 検証ゲート

| バックエンド            | 必須                                             |
| ----------------------- | ------------------------------------------------ |
| Dawn + D3D12            | dp4a / emulation の既知解、f32 / f16 storage     |
| Dawn + Metal            | 同上 + workgroup memory の境界ケース             |
| wgpu + Vulkan（NVIDIA） | dp4a / emulation、現行 golden parity             |
| wgpu + Vulkan（AMD）    | wave 幅を仮定しない fallback、境界 shape         |
| wgpu + Metal            | Apple Silicon 実機で canary と end-to-end の両方 |

## B. Metal-first GEMM ポートフォリオ

### 現状の疑い

現行の主変種は 64×64 tile、16×16 = 256 invocation、1 thread あたり 4×4 出力、
`array<vec4<f32>, 4>` accumulator、K tile 16、約 8KiB workgroup memory を使う。
Linux / NVIDIA では tiled が naive の約 5.40 倍だが、M2 では約 1.21 倍に留まった。

これは「Metal が遅い」とだけ見るより、次を疑うべき信号である。

- 256 thread と accumulator 配列の組合せによる register pressure / spill。
- 動的な accumulator index が MSL で静的展開されず、local memory 化する。
- 64×64 が Apple GPU の occupancy と合わない。
- vec4 load の整列条件を満たしても、変換後 MSL の access pattern が合わない。
- K tile 16 と barrier 頻度の釣り合いが backend ごとに違う。

### 候補行列

同じ数値式を保ち、幾何だけを独立軸にする。

| 変種             |   workgroup | thread 出力 | 狙い                                                           |
| ---------------- | ----------: | ----------: | -------------------------------------------------------------- |
| 現行             | 16×16 = 256 |         4×4 | NVIDIA の基準                                                  |
| 64×32            |  16×8 = 128 |         4×4 | thread 数と B tile を半減                                      |
| 32×64            |  8×16 = 128 |         4×4 | M が小さい / N が大きい場合                                    |
| 32×32            |    8×8 = 64 |         4×4 | register pressure と小 shape                                   |
| 64×64 static-acc |       16×16 |         4×4 | 名前付き4 vec4（16 scalar）へ展開。通常linearは`:u4`で実装済み |
| 32×32 scalar     | 16×16 = 256 |         2×2 | occupancy 優先の対照                                           |

linear、matmul/bmm、conv2d implicit GEMM、attention QK/PV を最初から全て増やさず、
まず共通 GEMM generator の microbenchmark で 2–3 変種に絞る。その後、各 op のロード規則へ
移植する。

### 部分実装（2026-08-06）

通常のlinear実行経路に `64×64 static-acc` の最小片を `:u4` 変種として導入した。
`acc0`〜`acc3` の名前付き `vec4<f32>` 4本（合計16 scalar）へ展開し、内積とstoreから
動的配列添字を除いた。旧変種は既定引数で残し、既存WGSLスナップショットも維持している。

Animaの既定presetは `w8a8-s16` であり、上記の通常linear実行経路を通らない。そこで
i8a8生成器にも同じ `:u4` 変種を追加し、実モデル454本中449本を代表する6形状の直接A/B
harness（`deno task bench:linear-i8a8`）を用意した。生成APIの既定は旧WGSLを維持し、
executorだけが検証済みの `:u4` を選ぶ。計測はpipeline compileとreadbackを除外し、
timestamp-query / wall fallback、ABBA / BAAB順序、全出力のbit比較を含む。

Linux / RTX 3080 Ti / wgpu-Vulkanの予備計測では、dp4aの全6形状がbit一致し、node数で
重み付けしたkernel時間は現行比1.429倍だった。Apple M2 / wgpu-Metalでも全6形状がbit一致し、
全shapeが1.158〜1.662倍、重み付き1.175倍だった。2 backendで退行shapeが無かったため、
backend分岐を増やさずproduction executorを `:u4` へ切り替えた。

これはタイルportfolioやautotuneの採用を意味しない。matmul / bmm / attention /
conv2dへの適用、64/128 thread変種は引き続き本節の検証対象。

### K方向の部分実装（2026-08-07）

出力M64×N64、256 thread、4×4静的accumulatorを固定し、Kだけを16→32へ広げるbenchmark候補を
追加した。共有A/Wは各512語を2パスで埋め、workgroup memoryを2→4KiBに増やす代わりに
barrier epochを半減する。既定WGSL・key・executorはK16のままで、K32とM/N縮小は組み合わせない。

RTXの全454本ではbit一致したがpaired加重0.974倍で、全代表shapeが退行した。M2も全454本で
bit一致し、pairedは1.044倍だったがp10 / p90が0.823〜1.106倍、独立中央値の加重実時間比は
0.919倍だった。複数backendで安定して勝たず、K32を不採用としてK拡大を終了する。
backend selectorやautotuneは、選ぶ価値のある候補が1本も残らない段階では導入しない。

### 選択方法

adapter 名による決め打ちは最後の手段にする。推奨は device 初期化時または model load 時の
小さな autotune である。

- 代表 shape を小・中・大 K、細長い M/N に分類。
- warmup 後に timestamp-query が検証済みなら GPU 時間、無ければ同期回数を抑えた wall time。
- 結果は device lifetime と kernel family / dtype / shape bucket でキャッシュ。
- autotune 自体の上限時間を設け、タイムアウト時は portable 既定へ戻す。
- pipeline key には tile、workgroup、accumulator 形式を全て含める。

### 採用条件

- Metal M2 で現行比 1.25 倍以上、または同速で大幅に低い一時メモリ。
- NVIDIA の既定を悪化させない。AMD / D3D12 は portable fallback 比で退行しない。
- 非 4 倍数、非常に小さい M/N/K、grid-stride 上限超過を含む。
- f32、w=f16 storage、i8、w8a8、dp4a off の全経路で CPU reference と比較。
- exact bit parity を要求する箇所と、浮動小数 tolerance を要求する箇所を分離する。

## C. PreparedExecutionPlan とホスト固定費

### 問題

現在は node ごとに params buffer を作り、bind group を作り、必要になった pipeline を逐次
await してから encode する。既存の 1024 条件の調査では params buffer 約 139ms/step、
bind group 約 52ms/step が観測された。さらに同じ Session の run は `#chain` で直列化される。

### 提案する層

`Session.prepare(inputShapes, bindings)` 相当で shape-specialized な executable plan を作る。

1. `planGraph` の結果から必要な pipeline key / WGSL を全列挙。
2. 異なるキーの pipeline Promise をまとめて開始し `Promise.all` で prewarm。
3. shape と attrs だけで決まる params を packed persistent buffer に配置。
4. binding が run 間で固定の weight / constant は bind group を再利用。
5. arena の transient buffer が変わる binding は、buffer slot が安定する設計にしてから再利用。
6. device loss では plan 全体を破棄し、古い GPU object を持ち越さない。

単純に bind group を Map キャッシュするだけでは、transient buffer の寿命と address が変わるため
危険である。先に arena の slot assignment を plan に固定するか、dynamic offset を使える
binding layout へ寄せる必要がある。

### 段階導入

- C1: pipeline の全列挙と並列 prewarm。GPU buffer の寿命を変えない。
- C2: params を 1 packed storage buffer / run にまとめる。
- C3: weight-only bind group を再利用。
- C4: transient slot 固定後に node bind group を再利用。

C1 は初回 latency、C2–C4 は毎 step の CPU 固定費に効く。別指標として測る。

## D. 入力依存性による staged execution

### 静的シグナル

- timestep-only 部分グラフ: 856 node、推定 774 dispatch。
- encoder-hidden-only 部分グラフ: linear 56、reshape 56、permute 168、RMS 28、推定 308 dispatch。
- 現在の pipeline の CFG は cond / uncond を逐次 predict する。
- `Promise.all` で同一 Session の run を呼んでも `#chain` により GPU 実行は直列であり、
  共通計算は自動共有されない。

### 提案

グラフ node ごとに入力依存集合を計算し、次の stage に分ける。

- model weight のみ: load 時に準備。
- condition のみ: prompt / negative prompt ごとに 1 回。
- timestep のみ: denoise step ごとに 1 回。
- latent + condition + timestep: predict ごと。
- latent のみ: CFG batch 化後にも共有できない本体。

stage 出力は CPU readback せず GPU resident tensor として保持する。CFG の cond / uncond で
timestep stage を共有し、全 denoise step で condition stage を共有する。

### 難所

- arena の liveness は 1 run 内だけでなく prepared tensor の寿命を表現する必要がある。
- condition cache の key は tensor identity ではなく model / device / dtype / shape / 内容の
  世代を含める。
- VRAM 増加を計測し、再計算より cache が不利な小 shape には使わない。
- device loss と Session dispose で prepared tensor を確実に解放する。
- RNG や副作用 op が追加された場合は依存集合だけで移動してはいけない。

## E. Prepared cross-attention K/V

encoder hidden state は denoise step 間で不変なのに、各 block / step で K/V projection、
reshape、permute、場合によっては量子化を繰り返す。

提案する内部値:

```text
PreparedCrossAttention {
  k: GPUBuffer
  v: GPUBuffer
  kScale?: GPUBuffer
  vScale?: GPUBuffer
  shape/dtype/layout/deviceGeneration
}
```

condition ごとに一度作り、attention kernel は prepared K/V を直接読む。量子化済み K/V なら
初回準備後に約 392 dispatch/run の削減候補で、保持量の試算は condition あたり約 57MiB
（f32 なら約 224MiB）。メモリ節約と dispatch 削減が同じ方向になる可能性が高い。

ただし public IR に直ちに prepared 型を出さず、まず runtime 内部の plan value として試す。
通常の attention 入力経路を fallback として残し、数値比較と device loss を固定してから
exporter 側の明示 op を検討する。

## F. 高頻度分解 OP の融合

### F1. RoPE

transformerを再集計すると、対象は28 block × Q/Kの **56鎖** だった。各鎖は7 nodeだがcatが
2 dispatchを使うため、現状448 dispatch。専用kernel後は56 dispatchで、削減は392となる。
旧見積りの112鎖 / 896 dispatchはQ/Kを二重計上していた。text encoderにもdirect-mul-first順の
strict鎖が **55本**あり、同じkernelで440→55、385 dispatch / generateを追加で削減できる。

第一段階は実装済み。public rope opを増やさず、executor内でslice-first / direct-mul-firstの
連続half-split鎖だけをexact matchする。内部値の外部consumer、graph output、shape / dtype / attrsの
差があれば従来primitiveへfallbackする。

- x=[1,H,S,128]、table [1,1,S,128]だけを受理。
- rotate_half(x) = cat(-x[...,64:128], x[...,0:64])。偶奇layoutは受理しない。
- Q / Kは別dispatch。Q/K同時融合や前後permuteとの融合は別段階。
- 256 thread、1D grid-stride、5 bindings、workgroup memory 2KiB。optional featureなし。
- 2本の積をworkgroup u32へmaterializeし、barrier後に加算してprimitiveのf32丸めを保つ。

S=4096のslice-firstはRTX/Vulkanで全出力bit一致と3.756倍、Apple M2 / Metalで全出力bit一致と
5.066倍のpaired改善を確認した。T=29のdirect-firstもH8 / H16でbit一致し、RTXは7.557 / 7.160倍、
Apple M2は8.709 / 5.295倍のpaired改善だった。Anima 512 / 8step goldenも一致し、残る門は他の
OP-017 / 018とまとめたAnima 1024 E2Eだけである。

conditionerの24鎖はD64かつ`[1,S,16,64]`、tableも`[1,S,1,64]`で、既存BHSD kernelの
`token = row % sequence`をそのまま広げると沈黙誤値になる。うち連続する22鎖はparamsへtoken行strideを
追加し、BSHDをshapeで明示受理できた場合だけ176→22の次候補とする。`sym_prefix_slice`が途中に入る
残り2鎖はbuffer生成順まで変わるため別設計とし、matcherのscan-aheadだけでは扱わない。

### F2. adaptive norm

`norm(x) * scale + bias` を 1 kernel にする。scale / bias の broadcast 軸を attrs で自由化せず、
実モデルに現れる末尾 channel 形から始める。NaN、epsilon、variance の丸め順を既存 norm と
比較し、単なる「式として同じ」ではなく許容差を固定する。

### F3. VAE channel RMS

非最終軸sumと前後permuteの除去はaxis-reduceとして実装済みで、VAE dispatchは395→335になった。
現在残る30鎖は物理的に `mul(x,x)→sum(axis=1)→sqrt→clamp_min→div` の5 dispatchで、専用
channel L2なら150→30の候補になる。ただしmulのstorage書込みが作るf32丸め境界を消すと、後段sumが
FMAや再結合されbit一致を失い得る。約1.13GiBの反復tensor passは強い信号だが、実装前に積を
u32 stagingして既存縮約順を保つprototypeと、特殊値・Anima goldenのcanaryを先に通す。

### F4. SiLU / linear epilogue

第一段階はOP-018として実装した。public IR / exporterを変えず、隣接する
`sigmoid(x)→mul(x,sigmoid)`だけを2→1 dispatchへ畳むstrict peepholeである。u32 workgroup
stagingとbarrierでprimitiveの中間f32 materialization点に対応させ、broadcast、alias、extra
consumerはfallbackする。既定1024 / 8stepでは610→305 dispatch、512では146→73 dispatchの試算で、
RTX / Apple M2直接A/BとAnima 512 goldenを通過した。1024 E2Eを残す。

`linear→silu` / `linear→gelu` のGEMM epilogue融合は別のL設計として保留する。kernel familyと
key軸を増やし、線形出力のstorage materialization境界も変えるため、GEMM portfolioが固まる前には
入れない。

## G. 量子化結果の共有

同じ activation を複数の i8 linear が読む Q/K/V や timestep projection で、row quantization を
各 linear 内に繰り返さない。

内部表現を `QuantizedRows { data, scale, rows, cols, deviceGeneration }` とし、consumer 数が
2 以上のときだけ materialize する。候補は 195 dispatch の削減だが、追加 buffer と liveness が
増えるため次をゲートにする。

- 1 consumer は現行 fused / inline 経路。
- 2 consumer 以上で共有。
- QKV 3 consumer を最初の実装対象。
- cross-attention と timestep は staged execution と同じ cache ownership を使う。
- dp4a canary failure 時も scalar emulation が同じ prepared value を読める。

### 実装（2026-08-07）

最初の段階として、public IRやSessionを越えるprepared tensorを増やさず、run-localの共有を実装した。
同じIR値名を直接読む適格i8a8 linearを実行計画から数え、最初のconsumerが作ったpacked dataと
row scaleを最後のconsumerまで `RunArena` で保持する。reshape aliasは行境界を変え得るため対象外。

実配布transformerでは454本のlinear量子化が259本になり、195 dispatch/predictを削減する。
共有30値のfan-outは3×28、56×1、85×1。S=4096の静的な同時保持上限は約8.52MiB、
RTX実測のrun全体のtransient peak増加は約0.50MiBだった。cross-attentionのdenoise step間共有は
依然としてD/Eのownership設計が必要であり、このrun-local cacheをそのまま永続化しない。

## H. CFG batch 化と並列性

同一 GPU queue では、独立 command buffer を JavaScript の `Promise.all` で投げるだけでは
カーネルが同時実行される保証はなく、現在は Session の `#chain` が run 自体を直列化する。
狙うべきは API 呼び出しの並列化より、**同じ dispatch 数で GPU を大きく使うこと**である。

候補:

- latent を batch 2 にし、cond / uncond の condition を batch 軸に積んで 1 predict。
- timestep embedding は batch へ broadcast。
- 最後に batch を 2 分割し CFG combine。
- VRAM は activation がほぼ 2 倍になるため、1024/4096 と VAE 共存条件で peak を測る。
- 小 batch で occupancy が低い Metal には効く可能性があるが、メモリ帯域飽和済みの NVIDIA では
  退行し得る。backend 名ではなく shape / VRAM budget と autotune で選ぶ。

## 実装順の提案

1. COMPAT-004: dp4a 実走カナリア + emulation fallback。
2. OP-017: VAE nearest x2 exact fusion（Metal直接A/B通過）の1024 E2E。
3. OP-018: SiLU exact peephole（Metal直接A/B通過）の1024 E2E。
4. HOST-006: immutable params / bind group生成数を再計測し、executable plan単位の再利用を試す。
5. F2: adaptive normのexact matcherと丸め境界canary。
6. F3: VAE channel L2はu32 stagingでbit一致を証明できた場合だけprototypeする。
7. C1: pipeline 並列 prewarm。
8. D + E: staged execution と prepared cross-attention K/V。
9. C2–C4: params / bind group / transient slot の executable plan 化。
10. CFG batch は VRAM と Metal 実測を見て opt-in から開始。

各段で NVIDIA / AMD / Metal / D3D12 の最低 1 経路を CI または予約実機で回し、portable fallback
を削除しない。
