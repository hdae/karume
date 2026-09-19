> 2026-09-19 時点の実測と調査のスナップショット。RTX 3080 Ti / Deno 2.9.6 の値が主で、Chrome と Apple M2 の値は過去 research からの引用。

# gemma4-qat の decode 速度の帰属と次に試すこと

QAT レビューの裁定を実装した後（main `6f57f14`・既定 capacity 4096 / chunk 768）、長文脈の出力は崩れなかったが
速度が期待に届かない、という利用者の観察を受け、「計算や KV を i8 にすれば速くなる」を含む高速化候補を
Opus の並列調査で集め、反証を掛けた。手順は §12。数値の正本は本文の表（測定はスクラッチで採り、値をここへ写した）。

## 1. 結論（先に読む）

1. **Deno の decode 23.8 ms/token のうち 10 ms は Deno 固有の待ち**である。`deno_webgpu` の `mapAsync` /
   `onSubmittedWorkDone` が GPU 完了後にも必ず 10 ms 寝る実装で（§3.2・ソース確認済み）、karume のコードとは
   無関係。同じ資産を Chrome で回した過去実測は壁 11.2 ms/token（89 tok/s）で、「ブラウザならもう少し速い」は
   当たり（2.1 倍）。
2. 出発点の「QAT の GEMV は通常 Gemma の同形状より 2〜4 倍遅い」は**取り下げる**（§4）。diagnostics のキーは
   担当形状が違い、計測モードの歪みも大きい。QAT が通常より遅いこと自体は Chrome の実測で確かだが
   （GPU 7.47 vs 6.28 ms）、その差はカーネル整備の不足 4 点（§6）で説明できる。
3. decode は**帯域律速でも演算律速でもない**。重み 825 MB を読む帯域時間は 0.9 ms で、GPU 実測 7.5 ms は
   帯域の 11.5%。律速は「発行命令数 × 低い占有率」と「dispatch 本数（1,132/token）」。
4. 利用者の仮説「計算を i8 に」は**方向として正しい**が、効く理由はバイト数でなく命令数（INT2 の f32 復元が
   1 要素 6 命令超）。実装点は既存の linear→SRQ 融合カーネルの内側 1 箇所に絞れる（§7）。
   「KV を i8 に」は**速度には効かない**（今の文脈長で KV は転送量の 0.45%・16k 文脈でも attention は遅延律速）。
   価値はメモリと容量天井（§7.2）。
5. WebML（同じ checkpoint・専用カーネル・285 tok/s）が速い理由も i8 ではない。KV は f32、整数内積命令は 0 箇所。
   差は実行構造 4 点 — decode 1 token を 1 pass / 1 submit、次 token を GPU 内で供給して深さ 4 で先行投入、
   SRQ を生産側へ畳んで単独 dispatch 0 本、subgroup 縮約の GEMV（§8）。

## 2. 今日の実測（Deno・RTX 3080 Ti・capacity 4096 / chunk 768・同一 prompt 41 token・warmup あり）

| 条件                                                                           | decode tok/s | ms/token | 備考                                                                      |
| ------------------------------------------------------------------------------ | -----------: | -------: | ------------------------------------------------------------------------- |
| QAT E2B `i4-fast`（既定）・64 token                                            |         42.0 |     23.8 | capacity 1024 でも 42.0（容量の影響なし）                                 |
| QAT E2B `i4-gemvpar`（並列 GEMV・融合なし）・64 token                          |         41.0 |     24.4 | linear→SRQ 融合は速くしている（遅くしていない）                           |
| QAT E2B `i4`（逐次参照）・64 token                                             |         35.1 |     28.5 |                                                                           |
| 通常 E2B `i4-fast`・64 token                                                   |         43.3 |     23.1 |                                                                           |
| QAT E2B `i4-fast`・256 token                                                   |         38.1 |     26.2 | 定常                                                                      |
| 通常 E2B `i4-fast`・256 token                                                  |         39.5 |     25.3 |                                                                           |
| 参照: Chrome・同じ資産・並列 GEMV 後（2026-09-12）                             |         89.1 |     11.2 | GPU 7.87 ms（[chrome-gemv-parallel](2026-09-12-chrome-gemv-parallel.md)） |
| 参照: Chrome・温度管理下（2026-09-15）                                         |         94.6 |     10.6 | [held-combinations](2026-09-15-held-combinations.md)                      |
| 参照: WebML Space（同じ公式 QAT E2B・専用 WebGPU カーネル・同じ RTX / Chrome） |      **285** |  **3.5** | [webml-browser-speed](2026-09-12-webml-browser-speed.md)                  |

GPU 側の内訳（`--diagnostics` = timestamp 計測モード。1 dispatch = 1 pass に開くので decode は 8〜10 tok/s に
落ち GPU 時間も膨らむ — **相対値と dispatch 数だけを読む**。run をまたいだ 1 dispatch あたりの比較は成立しない）:

| 条件             | 直近 run の GPU 時間 | dispatch/token | 上位 5                                                                                                                                  |
| ---------------- | -------------------: | -------------: | --------------------------------------------------------------------------------------------------------------------------------------- |
| QAT `i4-fast`    |              33.2 ms |          1,132 | wi4g512:l4+SRQ 4.57 ms（65）/ wi2:l2+SRQ 3.88（40）/ static_quantize 3.33（210）/ wi4g2048:l32+SRQ 2.40（43）/ rms_norm_add 2.37（105） |
| 通常 `i4-fast`   |              12.8 ms |            922 | wi4g32:l4 3.49（141）/ wi4g32:l32 3.16（135）/ rms_norm 1.15（136）/ rms_norm_add 1.10（105）/ wi8:l16 0.88（1）                        |
| QAT `i4`（逐次） |              45.5 ms |          1,513 | wi2 11.6（61）/ wi4g512 7.06（95）/ wi4g2048 5.89（43）/ static_quantize 5.71（485）/ rms_norm 3.75（242）                              |

## 3. 1 token の費用モデル（4 区分）

### 3.1 Deno の壁 23.8 ms の内訳

| 区分                               |     ms/token | 根拠                                                                                                                   |
| ---------------------------------- | -----------: | ---------------------------------------------------------------------------------------------------------------------- |
| GPU 実行                           |     6.8〜7.9 | Chrome 実測 7.47（cap 128）〜7.87（cap 8192）。今日の build は linear→SRQ 融合ぶん下がっているはず                     |
| dispatch 固定費（ホスト側 encode） |     0.7〜3.2 | 1,132 × 0.62 µs = 0.70 ms（[host-cost](2026-08-13-host-cost-decomposition.md) §4）+ writeBuffer 8 本 + bind group 構築 |
| ホスト（PLE gather・sampling・JS） |     0.3〜1.5 | PLE gather の CPU 実測 0.154 ms/token・RoPE 入力 0.029 ms（本調査 S5 の計測）                                          |
| **Deno 固有のフェンス待ち**        | **10〜13.5** | 壁 23.8 − Chrome 相当 10.3〜11.2。根因は §3.2                                                                          |

Chrome 側の同じ分解は 壁 11.2 = GPU 7.87 + 非 GPU 3.34。**GPU 時間を 0 にしても Chrome の karume は 3.34 ms/token
= 300 tok/s** で WebML の 1 token 全部（3.5 ms）と同じ桁 — WebML に並ぶには GPU も非 GPU も両方削る必要がある。

### 3.2 Deno のフェンス床の根因（ソース確認済み）

Deno 2.9.6 の `ext/webgpu/buffer.rs:241-261`（`mapAsync`）と `ext/webgpu/queue.rs:103-125`（`onSubmittedWorkDone`）は
同じ形をしている:

```rust
let device_poll_fut = async move {
  while !*done.borrow() {
    self.instance.device_poll(self.device, wgpu_types::PollType::wait_indefinitely())?;
    tokio::time::sleep(Duration::from_millis(10)).await;
  }
  Ok(())
};
let receiver_fut = async move { receiver.await??; *done_.borrow_mut() = true; Ok(()) };
tokio::try_join!(device_poll_fut, receiver_fut)?;
```

`device_poll(wait_indefinitely)` は GPU 完了まで同期ブロックし、その中でコールバックが `receiver` へ結果を送る。
しかし `done` を立てる `receiver_fut` が走れるのは `sleep(10 ms)` で yield した後なので、**GPU が何 ms で
終わっても毎回 1 回 10 ms 寝る**。decode は 1 token = フェンス 1 本（greedy 出口の 8 B 読み戻し）なので、
そのまま 10 ms/token の定数になる。過去実測と整合: フェンス床 ≈ 11.07 ms・workgroup 数に不依存
（[host-cost](2026-08-13-host-cost-decomposition.md) §1）、Deno の map 同期発行 11.98 ms vs Chrome 0.005 ms
（[codex-mtp-optimization](2026-09-10-codex-mtp-optimization.md)「ブラウザでの run 内訳」）。
`pop_error_scope` は同期でこの床を持たない。

含意: **Deno CLI の tok/s は製品の性能指標に使えない**。採否判定の物差しはブラウザの壁と GPU 時間にする。
Deno 側で床を消すには、Deno にパッチを当てる（poll ループを完了通知で即抜ける形に）か、1 フェンスあたりの
token 数を増やす（§10 のクラスタ B）しかない。

## 4. 取り下げた前提

| 前提                                                         | 判定         | 根拠                                                                                                                                                                                                                           |
| ------------------------------------------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| QAT の GEMV は通常の同形状より 1 dispatch あたり 2〜4 倍遅い | **取り下げ** | diagnostics のキー wi4g32:l4（通常・141 本）には最大 MLP 12288×1536 が 40 本入るが、QAT の wi4g512:l4（65 本）には 1 本も入らない（同じ場所は INT2 の別キー）。同一キー rms_norm_add の run 間比 2.16 倍で割り戻すと差は誤差内 |
| PLE をホストで gather しているから遅い                       | **取り下げ** | CPU 実測 0.154 ms/token（壁の 0.6%）。GPU 常駐 PLE の価値は速度ではなく「次 token を GPU 内で供給する」前提条件                                                                                                                |
| 重み帯域律速                                                 | **否**       | 下の roofline                                                                                                                                                                                                                  |
| SRQ 融合エピローグが GEMV を遅くしている                     | **否**       | i4-gemvpar 41.0 → i4-fast 42.0 tok/s、Chrome でも 7.351 → 6.775 ms/token                                                                                                                                                       |

roofline（decode 1 token で読む重み + scale・帯域 912 GB/s）:

| 資産                                                   | 読むバイト | 帯域時間 | 実測 GPU（Chrome） | 帯域利用率 |
| ------------------------------------------------------ | ---------: | -------: | -----------------: | ---------: |
| QAT（i2 / i4g512 / i4g2048 / i4g4096 / i8 / f32 混在） |     825 MB |  0.91 ms |            7.47 ms |  **11.5%** |
| 通常（i4g32 + i8 head）                                |   1,577 MB |  1.73 ms |            6.28 ms |      25.5% |

QAT は通常の 52% のバイトしか読まないのに 1.19 倍遅い。MAC 数は両者 2,279.5 M で同一。演算側も
0.19 TFLOP/s（FP32 ピークの 0.56%）。

## 5. dispatch の解剖（QAT `i4-fast`・M=1・1,132 本/token）

IR（1,985 ノード・35 層・自前 KV 15 層 / KV 共有 20 層）と recipe-builder の対応から:

| 実行単位                        | 本数 | 出どころ                                                                |
| ------------------------------- | ---: | ----------------------------------------------------------------------- |
| linear + SRQ 融合               |  275 | 7 GEMV/層 × 35 + k/v 2 × 15                                             |
| SRQ 単体（GEMV の入力側）       |  210 | 6/層 × 35 — rms_norm 後 70・mul 後 70・add 後 35・reshape←permute 後 35 |
| rms_norm 単体                   |  137 | 242 − 融合 105                                                          |
| rms_norm + add 融合             |  105 | 3/層                                                                    |
| states 形 attention             |  105 | 35 × 3 dispatch（`parallel`）                                           |
| mul                             |  109 | gelu×up 35 + gelu×PLE 35 + layer_scalar 35 + 4                          |
| gelu_tanh                       |   70 | MLP 35 + PLE ゲート 35                                                  |
| RoPE 融合                       |   50 | q 35 + k 15                                                             |
| slice（PLE 入力の層別切り出し） |   35 |                                                                         |
| state_append                    |   30 | k/v × 15 層                                                             |
| 入口・出口                      |    7 |                                                                         |
| reshape / permute / expand      |    0 | 別名化（K-41）                                                          |

検算: 逐次参照 1,513 − 1,132 = 381 ≒ 275 + 105、QAT 1,132 − 通常 922 = 210 = 生き残った SRQ ちょうど。
両 IR は static_quantize 485 本の差だけで、他 15 種の op は本数まで一致する。1 層あたり約 32 本
（WebML は約 9 本・§8）。dispatch 1 本の値段は GPU 0.5〜2.0 µs + ホスト 0.62 µs なので、融合で削れる
上限約 360 本は 0.4〜0.9 ms/token（壁の 2〜4%）— **dispatch ダイエット単独では WebML との差は埋まらない**。

## 6. GPU 側の構図 — 「深い量子化に合わせたカーネル整備」が通常の i4g32 より遅れている

並列 GEMV は 1 workgroup = 128 スレッドで出力 1 列を `lanes` 本で分担するので、起動スレッド数 ≈ n × lanes。
GA102 の同時常駐スレッド 122,880 に対する占有率:

| 格納     | n × k                                         |    lanes | 占有率 | 本数/token | 重み/token |
| -------- | --------------------------------------------- | -------: | -----: | ---------: | ---------: |
| i2       | 12288 × 1536（gate / up）                     |        2 |    20% |         40 | **191 MB** |
| i4 g512  | 6144 × 1536                                   |        4 |    20% |         30 |     144 MB |
| i2       | 1536 × 12288（down）                          |       32 |    40% |         20 |      94 MB |
| i4 g2048 | 1536 × 6144                                   |       32 |    40% |         15 |      71 MB |
| **f32**  | **8960 × 1536**（per_layer_model_projection） | **逐次** |     7% |          1 |  **55 MB** |
| **i2**   | **262144 × 1536**（lm_head）                  | **逐次** |      — |          1 | **102 MB** |

通常 Gemma の同形状 12288 × 1536 は i4g32 で lanes 4（40%）、lm_head は i8 で lanes 16。QAT が遅い 4 点:

1. SRQ 210 本の単独 dispatch と 1 token あたり 10.7 M 回の境界二分探索（通常にはこの op が無い）。
2. INT2 の最重量ブロック 40 本が lanes 2 = 占有率 20%（通常の同形の半分）。INT2 は「バイトは半分・算術は同じ以上」で、
   f32 復元の命令数は 1 要素あたり i2 6.25 / i4 5.10 / i8 3.25（生成 WGSL の実数え）= 1 token 13.3 G 命令。
3. `per_layer_model_projection` が QAT だけ f32 格納（通常は i4・6.9 MB）で、55 MB を占有率 7% の逐次 1 dispatch で読む。
4. INT2 の lm_head が並列 GEMV の対象表（`PARALLEL_SHAPES`）に無く逐次カーネル。通常の i8 head は wi8:l16 で 456 GB/s 出ている。

## 7. 利用者の仮説「計算や KV を i8 に」の評価

### 7.1 計算の i8 化（K-45）— 当たり。理由は命令数

- 活性は既に int8 の値になっている: linear 277 本のうち 275 本が SRQ 出力を読み、SRQ は定義上 q·scale を返す
  （int8 コード q を計算して捨てている）。w8a8 が要した動的 amax（quantize_rows）は QAT には不要。
- 現行の `linearCompute: "a8"` では実現できない — decode は GEMV 分岐より手前で GEMM 骨格へ落ち（GEMV の 8 倍遅い）、
  INT2 は対象外、SRQ 融合と排他。正しい形は K-45 で、linear→SRQ 融合（ADR 0103）で量子化器と内積が同じ
  シェーダに来た今、**エピローグの出力型と内積ループの型を変える差分**に縮んでいる。
- 見積り: dp4a 形で命令 13.3 G → 約 2.2 G。GPU 時間 1.6〜2.4 ms/token（Chrome 壁 11.2 → 8.8〜9.6 ms）。
  Deno の壁では 10 ms の床の陰で見えにくい。反証結果は §9 クラスタ C。
- 副産物: 整数累算は縮約順に依らないので、ADR 0097 追記 8 が「残る」とした CPU / GPU の token 列分岐の根本原因が消える
  （忠実度のレバー — 速度とは別勘定）。

### 7.2 KV の i8 化（K-46）— 速度には効かない。メモリと容量天井のレバー

KV の実寸（E2B・f32・full 3 スロット [C,512] × k,v・sliding 12 スロット [520,256] × k,v）:

|    文脈長 L | KV 読み/token | 重み 825 MB に対する比 | int8 化で縮むバイト |
| ----------: | ------------: | ---------------------: | ------------------: |
| 100（今日） |        3.7 MB |              **0.45%** |              2.8 MB |
|       4,096 |       62.9 MB |                   7.1% |             47.2 MB |
|      16,000 |        209 MB |                  20.2% |              157 MB |
|      32,768 |        415 MB |                  33.5% |              311 MB |
|     131,072 |      1,623 MB |                  66.3% |            1,217 MB |

attention は帯域律速ではない — P=16,000 で実測 49 ms vs 帯域換算 0.4 ms（123 倍・[context-length-sweep](2026-09-03-gemma4-context-length-sweep.md)）、
K-12 後も約 48 倍（[k12-sweep](2026-09-03-gemma4-chunklength-k12-sweep.md)）。perf-ledger L-10 の着手条件（帯域律速）は不成立。
価値はメモリ: capacity 4096 で KV 60 → 24 MiB、131,072 で 1,548 → 396 MiB、full スロットが `maxStorageBufferBindingSize`
に張り付く制約が 4 倍緩む。新規に確定した事実: 上流 checkpoint の `k/v_cache_scale` 70 本を直読すると、
full 層 = 4 / 9 / 14 / 19 / 24 / 29 / 34、自前 KV は層 0〜14（full 3 + sliding 12）、**KV 共有層の scale は共有元と完全一致**
（層 19 / 24 / 29 / 34 は層 14 の値・共有 sliding 16 層は層 13 の値）。スロット共有で scale が衝突する障害は無い。

## 8. WebML との対照（同じ checkpoint・ローカル参照 `158f16a` を実読）

| 観点            | WebML                                                                                                                           | karume（今日）                                                          |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| decode の構築   | ロード時に 1 度 prepared step 列へ焼き、毎 token は enqueue だけ                                                                | 毎 run で plan を実行（保存グラフ）                                     |
| submit / pass   | **1 encoder・1 pass・1 submit**（copy が挟まる時だけ pass を切る）                                                              | submit 4 / pass 3                                                       |
| 次 token の供給 | **GPU 内**（argmax 出力 = 次 step の埋め込み入力 buffer）・深さ 4 の先行投入                                                    | 8 B を読み戻してホストが次 run へ書く（1 token = 1 フェンス）           |
| SRQ             | 生産側カーネルへ畳み、行和 sum_a を一緒に出す（presrq）— 単独 dispatch 0 本                                                     | linear 出力側 275 本は融合済・入力側 210 本が単独                       |
| GEMV            | workgroup 32（= 1 subgroup）が K を coalesced 分担・subgroupAdd で barrier 0・活性 vec4 を N_ROWS 行で共有・scale は行ごと 1 個 | 128 スレッド・列を lanes 分担・木縮約・i4 は group scale を語ごとに引く |
| 整数内積        | **使っていない**（dot4I8Packed 0 箇所・unpack4xU8 + f32 dot。MLP だけ f16 dot + f32 累算）                                      | f32 復元 + f32 積和                                                     |
| KV cache        | **f32**                                                                                                                         | f32                                                                     |
| dispatch/token  | 約 316（コード見積り）                                                                                                          | 1,132                                                                   |
| PLE             | GPU 常駐の量子化表・gather + 初段 norm + SRQ を 1 dispatch                                                                      | ホスト行キャッシュ（0.15 ms）                                           |
| 融合粒度        | 1 層 7〜11 dispatch（QKV 1・attention 1・o_proj+残差+norm 1・gate/up+gelu+mul 1・down+残差+norm 1 …）                           | 1 層 約 32                                                              |
| 出口            | GPU 内 argmax・token id 4 B だけ読む                                                                                            | GPU 内 topk・8 B 読み戻し                                               |

帯域の物差し: 配布形 790 MiB / 912 GB/s = 0.91 ms が下限。WebML の 3.5 ms でも 3.9 倍、karume Chrome 11.2 ms は
12 倍、Deno 23.8 ms は 26 倍 — **どちらも帯域律速ではなく、karume はオーバーヘッド律速**。WebML の
`DecodeDownNormAdd` は atomic チケットで最後の workgroup に後段を実行させる形（karume が WGSL メモリモデル上
「可搬に証明できない」と判定した手筋）を使っている点は、そのまま真似できない。

## 9. 反証結果（候補 56 件 → 11 クラスタ・利得は二重計上を除いた値）

候補は 6 レッグから 56 件出た。重複を畳んで 11 クラスタにし、クラスタごとに実現性 / 数字 / 全体への効きの 3 レンズで
反証したうえで、会計レッグが費用モデル（§3）との整合と二重計上を点検した。**表の利得は「併用時の限界値」**
（単独実施の値は各クラスタの findings に残る）。Chrome の現況は ADR 0103 融合後の **壁 10.03 ms / GPU 6.79 ms /
1,136 dispatch**（[linear-static-quantize-fusion](2026-09-15-linear-static-quantize-fusion.md)・99.7 tok/s）を基準にした。

| #        | クラスタ                                                                            | 判定                         |       Deno ms/token | Chrome ms/token | 決め手                                                                                                                                                                                                                                                                                    |
| -------- | ----------------------------------------------------------------------------------- | ---------------------------- | ------------------: | --------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1a       | Deno の poll ループ修正（10 ms sleep 除去）                                         | holds 3/3                    |            **10.0** |               0 | §3.2。karume 側の変更ゼロ・Deno のパッチか上流修正                                                                                                                                                                                                                                        |
| B        | decode の先行投入（次 token を GPU 内で供給し深さ d で run を先行発行）             | holds                        | **5.0**（単独 8.0） |         **2.2** | WebML の `submitStep` / `depth 4` と同形。karume の部品（常駐 topk index・常駐入力の bind group・位置の先読み）は揃い、欠けるのは PLE の GPU 常駐と ADR 0066 の「同じ context に未確定 run を積まない」契約の opt-in 例外。sleep は 10/d に割れる                                         |
| B の前提 | PLE を GPU 常駐の i4 表に（S6-8）                                                   | holds                        |                0.25 |            0.25 | 既存 embedding カーネル（w=i4・group 256）が sidecar の並びとそのまま一致しビット一致で組める。寸法 1.13 GiB の単一束縛 → manifest の requiredLimits（現在 256 MiB）引き上げ・shard 遅延ロードの放棄                                                                                      |
| C        | K-45 = QAT 専用の整数内積 GEMV（活性は SRQ の i8 コード・dp4a・i32 累算）           | holds（条件付き）            |             **0.9** |         **0.9** | 起票の 1.6〜2.4 は過大（GEMV の GPU 総時間 4.0〜4.9 ms が母数・H1 / H2 を内包）。**段 0 の kill 判定**（新カーネル不要・半日）で「命令削減が時間になるか」を先に切り分ける — 同一命令数の i2 カーネルが n=12288 で 769・n=262144 で 1,574 G要素/s と 2 倍差（主要形は命令律速でない疑い） |
| E        | 入力側 SRQ の融合（rms_norm→SRQ 70・mul→SRQ 70・gelu→mul→SRQ 35）                   | holds                        | 0.19（B 併用 0.08） |    0.19（0.08） | 既存 rms_norm エピローグ規則の拡張（段 1・不変条件無改訂）+ 新規則（段 2・「heads 互いに素」の緩和が要る）。attention 後 35・残差 add 後 35 は契約（windowTouchesState MUST・単一出力）で不可                                                                                             |
| F        | その他 dispatch 削減（slice のオフセット別名化 35・parallel-fused 28・rms→RoPE 50） | holds / uncertain            |        0.17（0.06） |    0.14（0.06） | slice 別名化はビット同一・再 export 不要（ADR 0093 の offset 束縛を alias に通すだけ）。parallel-fused は capacity 4096 で 28/35 層適格・Deno に指定口が無く未実測。既定変更は ADR 0102 が M2 実測で却下済み                                                                              |
| G        | 重みの連結 GEMV（k+v 15 層・gate/up 35 層）                                         | holds（gate/up は段 0 待ち） |                0.16 |            0.13 | k+v は同形・同 lanes・出力 SRQ scale 15/15 層一致 → ビット同一で −15 dispatch（実測差 2.85 µs/層）。q/k/v 3 本連結は lanes と scale が違い不可                                                                                                                                            |
| I        | `per_layer_model_projection`（f32 55 MB）を並列 GEMV の対象表へ                     | holds → **§13 で kill**      |                0.14 |            0.14 | 同形 i4 の逐次→lanes 4 が 1.63 倍（実測）。再 export 不要。i4 化（段 2）は再 export が別の理由で起きる時だけ相乗り                                                                                                                                                                        |
| J        | submit / pass 構造                                                                  | 反転                         |       0（B 採用時） |   0（B 採用時） | 「submit 4→1」は refuted（同方向の H-25 が Chrome で −7.3%）。生きるのは逆向きの **first-chunk ramp**（先頭チャンクを小さくして GPU を早く走らせる — Deno 0.45 / Chrome 1.0）で、B が Deno 専用と判明した時だけ復活                                                                       |
| H1       | INT2 カーネルの lane / LUT / 語粒度                                                 | refuted（lane）/ C に吸収    |                   0 |               0 | 同形の lane 全掃引が l2 → l32 で単調悪化（実測）。残るのは bitcast 復元の命令ダイエット 0.2（C を落とした時だけ独立に採る）                                                                                                                                                               |
| H2       | GEMV 再構成（subgroup K 分担・per-row scale K-47・f16 活性・subgroup matrix）       | refuted / C に吸収           |                   0 |               0 | subgroup matrix は WebML でも prefill（M ≥ 64）限定。K-47 は i4 語 = 32 要素で scale 読みは 48 回/列に過ぎず速度案として refuted。ZP と scale をエピローグへ畳む変種は C と同一作業                                                                                                       |
| D        | lm_head（i2 262144×1536）の並列化・整数化・block-major                              | **refuted（kill）**          |                   0 |               0 | lane 2〜32 が全て逐次より遅い（実測・l16 で 3.0 倍悪化）。天井 0.24 ms/token = GPU の 3.3%。前後に SRQ が無く整数内積の適用先としても最悪                                                                                                                                                 |
| K        | int8 KV cache（K-46）・attention の整数化                                           | 速度 0（holds）              |                   0 |               0 | L≈100 で KV 経路 0.15 ms/token（限界傾き 0.52 µs/位置）。メモリ項目として再起票（sliding も同じ土台で・「full だけ」の段階案は捨てる）。長文脈でロード粒度（4 要素/語）が効くかは M-K2 で判定                                                                                             |

refuted で閉じたもの（理由つき・各 findings に詳細）: 投機デコードによるフェンス償却（**QAT 資産に drafter が無い**・
自由文帯ではフェンス数/token が増える）、複数会話の 1 フェンス束ね（×1.27 が上限）、q/k/v 3 本連結、attention の 1 dispatch 化
（last-arriver merge を可搬に証明できない）、state_append の消去（ADR 0066 決定 9 の順序不変条件）、f16 活性、
`linearCompute: "a8"` の流用（M=1 の GEMM 骨格は GEMV の 8.45 倍遅い）、prefill の subgroup matrix（decode 0 ms）。

### 9.1 会計 — 併用したときに到達できる壁

| 環境   |                  今日 |                併用後の見込み | 内訳（限界値）                                                     |
| ------ | --------------------: | ----------------------------: | ------------------------------------------------------------------ |
| Deno   |   23.8 ms（42 tok/s） | **約 7.5 ms（約 130 tok/s）** | 1a 10.0 + B 5.0 + C 0.9 + E 0.08 + F 0.06 + G 0.16 + I 0.14 = 16.3 |
| Chrome | 10.03 ms（100 tok/s） | **約 6.5 ms（約 150 tok/s）** | B 2.2 + C 0.9 + E 0.08 + F 0.06 + G 0.13 + I 0.14 = 3.5            |

WebML の 3.5 ms には Chrome でなお 3.0 ms 足りない。値段の付いていないレバーが 2 つ残る:

1. **層内の大融合が作る約 480 dispatch の差**（WebML 約 316 本 vs karume 1,132 本。E + F + G を全部入れても 794 本）。
   karume では 3 つの契約が同時に塞いでいる — `windowTouchesState` の MUST（ADR 0067）、FusedStep 単一出力の MUST（ADR 0068 決定 1）、
   atomic last-arriver merge の可搬性判定。個別候補が全て refuted になった結果、この 480 本に値段を付けたクラスタは無い（単価 1.1 µs なら 0.5 ms 相当）。
   契約の改訂は設計裁定であって速度候補ではない。
2. **Chrome の非 GPU 3.2 ms のうち driver 固有の分**（pass 境界・submit 遅延）。B の kill 基準そのもので、大半が driver 側なら B の 2.2 は過大で
   どのレバーでも取れない。

順序依存: **先行投入（B）を入れると壁が GPU 律速になり、ホスト側の節約（J・E/F の 0.62 µs/本）は壁に出なくなる一方、GPU 側の節約は 1:1 で壁に出る。**
GPU 系を B の前に Deno で測ると過小に見える — **GPU 系の採否は Chrome の GPU 時間で決める**。

## 10. 次に試すこと（順位・期待利得・測り方）

| 順           | 作業                                                                                                                                                                                                                                              | 種別                             | 期待利得                                                | 前提・費用                                                                                                                                                                | 採否の門                                                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 0（済 §13）  | **物差しを直す**: 採否判定は Chrome の壁と GPU 時間（`deno task bench:llm-browser`）。Deno CLI の tok/s は 10 ms の定数込みなので比較にだけ使う。Deno の上流へ poll ループの修正を報告するかは別途                                                | 計測衛生                         | Deno 10 ms（パッチ時）                                  | コード変更ゼロ                                                                                                                                                            | —                                                                                                                                  |
| 0'（済 §13） | **1 セッションで確定する事実**（GPU 1 回・実装なし）: opbench graph の per-key GPU 時間で GEMV 総時間（4.0〜4.9 の見積り）・未帰属 49 dispatch（C-9）・`per_layer_model_projection` の現状費用（0.2〜0.57 の幅）・lm_head 1 dispatch の値を採る   | 計測                             | —                                                       | `tools/opbench` 既存                                                                                                                                                      | —                                                                                                                                  |
| 1            | **C 段 0 — K-45 の kill 判定**: 現行 GEMV の積和展開だけを「数値は壊れるが構造は同一」な最小命令形に差し替えた比較用シェーダを opbench single（heater 付き・ABBA×4）で 2 形（i2 262144×1536・i2 12288×1536 l2）A/B                                | スパイク（半日・新カーネル不要） | C の 0.9 ms 全額が懸かる                                | 無し                                                                                                                                                                      | ① が 25% 以上速くならなければ **C は refuted**。① だけ速く ② が 5% 以内なら対象形を絞る                                            |
| 2            | **B 段 ① — PLE の GPU 常駐（depth 1 のまま）**: 既存 embedding カーネルで sidecar を gather、input_ids を selector の常駐 index に                                                                                                                | 実装（前提工事）                 | 0.25 ms + B の前提                                      | manifest requiredLimits 引き上げ（256 MiB → 1.13 GiB・M2 の VRAM 可否は利用者実機）・ADR 0085 決定 3 の shard 遅延ロード放棄・段階席（full 予算が取れる機だけ常駐）       | **速度を見ない**: `ple.probe` のビット一致 + 64 / 256 token の id 列完全一致。割れたら進まない                                     |
| 3            | **B 段 ② — 先行投入 depth 2 → 4**: ADR 0066 に「K 本の decode run を同一 context へ先行発行してよい（位置は発行側が先に宣言・論理長は成功で K 進む）」の opt-in 例外・EOS 超過は既存 rewind・greedy 限定席                                        | 実装（契約改訂）                 | Deno ×1.5〜1.7・Chrome 最大 −2.2 ms                     | ADR 0066 / 0054 決定 2（gpuTiming と非両立 → 診断経路は非先行投入）                                                                                                       | depth 1/2/4/8 の ABBA（Deno と Chrome）。**Chrome の壁−GPU 差が 3.2 → 1 ms 未満へ落ちなければ Deno 専用と判定**し J の ramp を復活 |
| 4            | **C 段 1 / 段 2 — 整数内積 GEMV 族**（段 0 が通った形だけ）: Route A = 重み語を `unpack4xU8 → −z → pack4xI8 → dot4I8Packed`（行和不要・再 export 不要）→ Route B = 符号なし内積 + 活性の偏り `a^0x80` + 補正 2 項をエピローグへ                   | 実装（新カーネル族）             | 0.9 ms（GPU）                                           | ADR 0058 の数値 opt-in 席 + manifest の 4 つ目の宣言可能ノブ・`src/reference` の整数経路（i32 厳密参照・atol=0）・PARALLEL_SHAPES の穴埋め・ADR 0097 追記 8 / 0098 の改訂 | 単体 A/B → lane 2〜32 で u32 完全一致（縮約一意性の検出器）→ Chrome ABBA                                                           |
| 5            | **小物の波**（ビット同一・再 export 不要・各 ≤0.2 ms）: E 段 1（rms_norm→SRQ 70）、F の slice オフセット別名化、G の k+v 連結、I 段 1（f32 8960×1536 を対象表へ・lane 掃引）、F の `--state-attention-reduce` CLI フラグ追加と capacity 4096 実測 | 実装（小）                       | 合計 0.5〜0.6 ms（Chrome）                              | 融合カウンタ（ADR 0040）・SessionSpec の席                                                                                                                                | 各 dispatch 数の減少を `--diagnostics` で確認 → Chrome ABBA。E 段 2（mul→SRQ・gelu 連鎖）は「heads 互いに素」の緩和裁定後          |
| 6            | **K-46 の再起票**（速度の波から外す）: メモリ・容量天井の項目として full + sliding 両方。長文脈の品質検収（512 超・未検収）が立ってから                                                                                                           | 台帳整理                         | 速度 0・KV 60 → 15 MiB（4096）・1,548 → 387 MiB（131k） | ADR 0066 決定 2 / STATE_DTYPES・ADR 0058 席                                                                                                                               | M-K2: attention_state_qk の D_LANES 掃引で「遅延律速か発行律速か」を判定 → 発行律速なら 4 要素/語のロードを長文脈の波へ            |
| —            | **契約の裁定（値段の付いていない 480 dispatch）**: windowTouchesState MUST / 単一出力 MUST / last-arriver の可搬性を緩めるか                                                                                                                      | 設計裁定                         | 最大 0.5 ms + WebML 級融合の前提                        | ADR 0067 / 0068 の改訂                                                                                                                                                    | 裁定前に候補化しない                                                                                                               |

kill として台帳に残すもの: lm_head の並列化 / block-major / argmax エピローグ（D）、INT2 の lane 変更（H1）、q/k/v 3 本連結、
submit 4→1（S6-1）、K-47 の速度案、f16 活性、decode の subgroup matrix、QAT の投機（drafter 無し）、複数会話束ね。

## 11. 実測でしか確定しない主張（外れたときに崩れる金額の大きい順）

1. GEMV の GPU 総時間 4.0〜4.9 ms/token — GPU 区分の全判定の母数だが直接測られたことが無い（0' で確定）。**→ §13: 約 4.0 ms で確定**。
2. C 段 0: 占有率 20% 以下の形で命令削減が時間になるか — 反証材料あり（同一命令数で n 依存の 2 倍差）。
3. dispatch 単価 1.1 µs/本（GPU 0.48 + ホスト 0.62）— E / F / G / J の全額がこの係数に線形。温度管理下では上下 2 倍ぶれうる。
4. Chrome の壁−GPU 差 3.2 ms のうち driver 固有の割合 — B と J の存否。J の陽性対照（H-25 の 2048 アーム再走で −7% が再現するか）が最短。**→ §13: 差は 3.3〜4.1 ms（割合は未分解）**。
5. B: depth d の d 本の mapAsync が 1 回の 10 ms sleep 窓でまとめて解決するか（推測: `done` を receiver 側で立てる形なので解決するはず）。
6. parallel-fused を capacity 4096・Deno で測った値（指定口が無く一度も測られていない）。
7. `per_layer_model_projection` の現状費用（上界 0.2〜0.57 の幅）。0.3 ms 未満なら I はクラスタごと消える。**→ §13: 0.13 ms・kill**。
8. G: i2 n=24576 l2 の単体時間（n 方向の実測点が 1 つ）。
9. Deno で 0.1 ms 級の GPU 短縮が poll の 10 ms 刻みに吸われて壁が動かない可能性 — GPU 系は Chrome で先に測る根拠。
10. M2 の Metal で `dot4I8Packed` が native 命令に落ちるか（落ちなければ C は M2 で無効）。
11. `v_cache_scale` が層ごとに較正されていない疑い（full 7 層一律 0.2857・sliding 28 層一律 0.0472 — 固定クリップ幅の可能性・int8 V の品質リスク）。

## 12. 方法の記録

- 調査 Workflow（6 レッグ・Opus・読み取り専用・GPU 不使用）: S1 費用モデル / S2 GEMV カーネル / S3 融合と dispatch 削減 /
  S4 i8 計算と KV / S5 ホスト・ランタイム / S6 外部実装（WebML のソースはローカル参照を実読）。findings は
  `.claude/reviews/2026-09-19_qat-speed-recon/findings/`（git 追跡外・判断に要る数値は本文へ転記した）。
- 反証: 1 回目は候補ごと × 3 レンズの直積（168 レッグ）で組み、S5 の 7 候補分（21 レッグ）を終えたところで
  サブスクリプションの週間制限に当たり残りが失敗した。2 回目は候補 49 件を重複で 11 クラスタに畳み、
  クラスタごとに 1 レッグで 3 レンズを順に適用（11 レッグ）+ 費用モデルとの整合・二重計上を点検する会計 1 レッグ。
  **1 Workflow の総レッグ数は 100 未満に抑える**（候補 × レンズの直積展開は組まない — 重複クラスタ化が二重計上の検出にもなる）。
- Deno の根因（§3.2）はオーケストレータが Deno v2.9.6 のソースを取得して行番号まで確認した。
- GPU 実測は全てオーケストレータが採り、レッグには渡したログだけを使わせた。

## 付録 — 今日の実測ログ（Deno・逐語）

```text
=== gemma4 e2b (i4-fast, capacity 4096, no diagnostics)
[gemma4] loaded（1.9s） / capacity 4096 / maxPosition 131072 / chunk 768
[max-tokens · 64 tok · TTFT 123 ms · decode 43.3 tok/s · total 1.58s]
=== gemma4-qat e2b --quant i4 (sequential, no diagnostics)
[max-tokens · 64 tok · TTFT 161 ms · decode 35.1 tok/s · total 1.95s]
=== gemma4-qat e2b --quant i4-gemvpar (no fusion)
[max-tokens · 64 tok · TTFT 164 ms · decode 41.0 tok/s · total 1.70s]
=== gemma4-qat e2b i4-fast capacity 1024
[max-tokens · 64 tok · TTFT 168 ms · decode 42.0 tok/s · total 1.67s]
=== gemma4-qat e2b i4-fast capacity 4096
[max-tokens · 64 tok · TTFT 159 ms · decode 41.9 tok/s · total 1.66s]
=== gemma4-qat e2b i4-fast 256 tokens
[max-tokens · 256 tok · TTFT 158 ms · decode 38.1 tok/s · total 6.85s]
=== gemma4 e2b i4-fast 256 tokens
[max-tokens · 256 tok · TTFT 124 ms · decode 39.5 tok/s · total 6.58s]
=== gemma4-qat e2b (i4-fast, capacity 4096) --diagnostics
[max-tokens · 64 tok · TTFT 226 ms · decode 8.6 tok/s · total 7.58s]
[diagnostics] run 64 本 · 直近 run 33.162 ms / dispatch 1132
  linear_gemv_parallel:wi4g512:l4:static-quantize:v1 4.569 ms（13.8%） · 65 dispatch
  linear_gemv_parallel:wi2:l2:static-quantize:v1 3.882 ms（11.7%） · 40 dispatch
  static_quantize:v1:f32:wg128 3.332 ms（10.0%） · 210 dispatch
  linear_gemv_parallel:wi4g2048:l32:static-quantize:v1 2.404 ms（7.3%） · 43 dispatch
  rms_norm_add:v1:rms_norm:v1:f32:lastdim:wg256:residual-norm:xor-round 2.374 ms（7.2%） · 105 dispatch
=== gemma4 e2b (i4-fast, capacity 4096) --diagnostics
[max-tokens · 64 tok · TTFT 189 ms · decode 9.9 tok/s · total 6.52s]
[diagnostics] run 64 本 · 直近 run 12.776 ms / dispatch 922
  linear_gemv_parallel:wi4g32:l4 3.491 ms（27.3%） · 141 dispatch
  linear_gemv_parallel:wi4g32:l32 3.156 ms（24.7%） · 135 dispatch
  rms_norm:v1:f32:lastdim:wg256 1.153 ms（9.0%） · 136 dispatch
  rms_norm_add:v1:rms_norm:v1:f32:lastdim:wg256:residual-norm:xor-round 1.101 ms（8.6%） · 105 dispatch
  linear_gemv_parallel:wi8:l16 0.883 ms（6.9%） · 1 dispatch
=== gemma4-qat e2b --quant i4 (sequential reference) --diagnostics
[max-tokens · 64 tok · TTFT 221 ms · decode 8.1 tok/s · total 8.04s]
[diagnostics] run 64 本 · 直近 run 45.527 ms / dispatch 1513
  linear_gemv:v1:f32:c32u4:wi2 11.608 ms（25.5%） · 61 dispatch
  linear_gemv:v1:f32:c32u4:wi4g512 7.061 ms（15.5%） · 95 dispatch
  linear_gemv:v1:f32:c32u4:wi4g2048 5.886 ms（12.9%） · 43 dispatch
  static_quantize:v1:f32:wg128 5.710 ms（12.5%） · 485 dispatch
  rms_norm:v1:f32:lastdim:wg256 3.751 ms（8.2%） · 242 dispatch
```

## 13. 計測セッション（2026-09-19 夜・RTX 3080 Ti・main `debb4a1`・§10 の ⓪ と 0'）

条件: capacity 4096 / chunk 768・prompt は `tools/llm-speed/browser/cases.json` の 2 本（英 31 token / 日 37 token）・64 token・
warmup あり・各ロード前 75℃ 以下かつ thermal slowdown 無効。生データは [結果 JSON](2026-09-19-qat-speed-recon-results.json) と
bundle `outputs/bench/karume/2026-09-19_18-11-40_speed-baseline-k9/`（git 追跡外）。Chrome は Xvfb 上の Chrome 153（NVIDIA f16 実験フラグ付き・
[browser-llm-speed](2026-09-12-browser-llm-speed.md) と同じ起動条件）、ロードごとに fresh page・2 暖機 + 3 本計測。

### 13.1 壁時計（decode）

| 環境                                             |                        QAT `i4-fast` |                 通常 `i4-fast` |
| ------------------------------------------------ | -----------------------------------: | -----------------------------: |
| Deno CLI・64 token（3 回の中央値・範囲 ±0.1）    |                42.1 tok/s = 23.75 ms |          43.6 tok/s = 22.94 ms |
| Deno CLI・256 token                              |                 38.6 tok/s = 25.9 ms |           39.7 tok/s = 25.2 ms |
| Chrome・64 token（2 ロード × 2 prompt の中央値） | 92.7〜93.9 tok/s = **10.6〜10.8 ms** | 99.4〜99.8 tok/s = **10.0 ms** |
| Deno − Chrome                                    |                              13.0 ms |                        12.9 ms |

Deno − Chrome の 13 ms は §3 の「Deno 固有のフェンス待ち 12.6〜13.5」の内側（sleep 10 ms + Deno 側ホストの残り約 3 ms）。
token 列は 4 ロード × 2 prompt とも 3 本の計測内で完全一致（結果 JSON に保存）。

### 13.2 GPU 時間（Chrome・pass 境界の外部 timestamp = 製品の pass 構成のまま）

| ロード | GPU ms/token 中央値 [min–max] | 同ロードの壁 | 非 GPU（壁 − GPU） | dispatch / pass per token |
| ------ | ----------------------------: | -----------: | -----------------: | ------------------------- |
| QAT 1  |              6.98 [6.79–7.13] |   10.6〜11.1 |           3.3〜4.1 | 1,134 / 3                 |
| QAT 2  |              7.49 [6.92–8.40] |         10.8 |                3.3 | 1,134 / 3                 |
| 通常   |              6.21 [6.05–6.49] |         9.95 |                3.7 | 924 / 2                   |

先行投入（H-27）の kill 基準「Chrome の壁 − GPU が 1 ms 未満へ落ちるか」の基線は **3.3〜4.1 ms**。

### 13.3 per-key の GPU 配分（dispatch 分割 timestamp を pass 総時間へ補正）

dispatch ごとに pass を割ると総 GPU 時間が QAT 11.61 ms / 通常 10.29 ms に膨らむ（pass 総時間 6.98 / 6.21 の 1.66 倍）。
膨張を **dispatch あたり一定（QAT 4.1 µs・通常 4.4 µs）と仮定**して各キーから引いた値が「補正」。仮定は粗いが、
dispatch 本数の多い小カーネル（SRQ・rms_norm・elementwise）ほど生値が過大になる向きは正しい。

| 群（QAT）                                   |   本数/token |            生値 ms |            補正 ms |  通常 Gemma 補正 ms |
| ------------------------------------------- | -----------: | -----------------: | -----------------: | ------------------: |
| GEMV（並列 275 + 逐次 2）                   |          277 |               5.09 |    **3.95（57%）** | 4.06（65%・277 本） |
| static_quantize                             |          210 |               1.52 |           **0.66** |                   0 |
| rms_norm + rms_norm_add                     |          242 |               2.04 |               1.05 |                0.95 |
| attention_state_*（qk / stats / pv）        |          105 |               1.17 |               0.74 |                0.73 |
| elementwise（mul 108・gelu 70・…）          |          181 |               1.08 |               0.34 |                0.26 |
| RoPE 融合 / slice（strided） / state_append | 50 / 35 / 30 | 0.29 / 0.22 / 0.18 | 0.09 / 0.08 / 0.05 |  0.10 / 0.07 / 0.04 |
| 出口（argmax / topk）・embedding            |            4 |               0.03 |               0.01 |                0.01 |

QAT の GEMV 形別（補正 ms）: wi4g512:l4 65 本 0.88 / **wi2:l2（gate・up）40 本 0.80**（191 MB → 239 GB/s） /
**wi2:l32（down）20 本 0.73**（94 MB → 129 GB/s） / wi4g2048:l32 43 本 0.57 / wi8 70 本 0.35 / wi4g512:l32 30 本 0.15 /
lm_head（逐次 wi2）1 本 **0.24** / `per_layer_model_projection`（逐次 f32）1 本 **0.13**。

### 13.4 確定したこと（§11 の未確定事項への回答）

1. **GEMV の GPU 総時間 = 約 4.0 ms/token**（§11 の 1・見積り 4.0〜4.9 の下端）。QAT は通常の半分のバイトしか読まないのに
   GEMV 時間は同じ（3.95 vs 4.06）— 帯域律速でないことの直接の証拠。**QAT が通常より遅い 0.77 ms は SRQ 0.66 + f32 射影 0.13 でほぼ全部**。
2. **未帰属 dispatch は無い**（§11 の 4・C-9）: 1,134 本すべてが既知キーで、内訳は §5 の表と一致（1,132 + greedy 出口の argmax / topk 2 本）。
   §5 の census 由来モデルの +49 は slice の実体化 35 本と RoPE の k 側 15 本を census が数えていなかった帳簿差で、未知のカーネルではない。
3. **K-51（`per_layer_model_projection` の並列化）は kill**（§11 の 7）: 現状費用 0.13 ms/token は kill 線 0.3 ms 未満。0 にしても GPU の 1.9%。
4. **K-52（lm_head）の kill を実モデルでも確認**: 0.24 ms/token（単体実測と同じ値）。
5. static_quantize 210 本の実時間は 0.66 ms（補正）。クラスタ E の見積り 0.19 ms は「dispatch 固定費」だけを数えた値で、
   融合で消える時間はその間（0.19〜0.66）。E の kill 線を再設定する材料。
6. Chrome の非 GPU は 3.3〜4.1 ms（旧基準 3.2〜3.3 とほぼ同じ）。

M2（利用者実機・Chrome・ブラウザ計測ページの既定往復）は利用者が実行中で、JSON 受領後に本節へ追記する。

### 13.5 Deno 側の per-op 配分（opbench graph・QAT 対応を追加して実走）

`tools/opbench` の `graph` を QAT 家族でも駆動できるようにし（`--family` 省略時は manifest から推定・census の既定シナリオも共有）、
同じ prompt・64 token・capacity 4096 で 1 回実走した（timing モード = dispatch ごとの pass 分割なので絶対値は膨らむ。1 run 25.9 ms）。
census との突合で **unmapped_keys は空**（1,137 dispatch/run = 1,132 + 出口・prefill 込み）。op 別の配分は Chrome の補正値と同じ順位:
linear 57% / rms_norm 14% / attention 10% / static_quantize 9.5% / mul 3% / gelu 2% / RoPE 融合 1.5% / slice 1.2% / state_append 0.8%。
census の素ノード数と dispatch 数の食い違い（attention 35 → 105・static_quantize 485 → 210・add 106 → 0・slice 35 → 0）は、
census が session ノブの融合（rms_norm→add・linear→SRQ）と attention の 3 段展開・slice の実体化を数えないため。

### 13.6 M2（利用者実機・Chrome 153・apple / metal-3・ブラウザ計測ページの既定往復）

条件はページ既定（QAT E2B・`i4-fast` の並列 GEMV・rms→add 融合・**capacity 128 / chunk 64**・投入上限 768・linear→SRQ 融合を
off → on → on → off の 4 ロード・2 prompt × 各 5 生成 = 40 生成・checkout `debb4a1` clean）。要約は
[M2 結果 JSON](2026-09-19-m2-browser-speed-results.json)（元ファイルの SHA256 つき）。

| ロード | linear→SRQ 融合 | 英語 decode tok/s（中央値 [範囲]） | 日本語 decode tok/s | TTFT 英 / 日 |
| ------ | --------------- | ---------------------------------: | ------------------: | -----------: |
| 1      | off             |                   33.5 [32.2–35.3] |    32.1 [30.8–35.1] | 386 / 487 ms |
| 2      | on              |                   34.8 [33.2–35.2] |    35.1 [31.6–35.2] |    386 / 486 |
| 3      | on              |                   35.2 [29.9–35.4] |    35.1 [30.7–35.2] |    386 / 477 |
| 4      | off             |                   34.4 [32.1–35.3] |    34.5 [30.2–35.2] |    386 / 484 |

中央値の中央値: 融合 off 33.9 / on **35.1 tok/s**（+3.5%）— 09-15 の M2 実測（34.1 / 35.1・
[m2-linear-srq-adoption](2026-09-15-m2-linear-srq-adoption.md)）と同じ値で、現行 main に M2 の退行は無い。
token 列は 8 系列とも計測内で決定的。M2 の壁 28.5 ms/token は RTX Chrome の 10.7 ms の 2.7 倍で、
RTX で決めた採否を M2 へ持ち込むときは同ページの往復比較で追試する（従来どおり）。

## 14. K-45 段 0 — 「現行の INT2 / INT4 GEMV は ALU 命令数が律速か」の切り分け（2026-09-19・RTX・Deno 単体ハーネス）

新カーネルを書く前の kill 判定（§10 row 1）。製品の生成器が吐く実キーの WGSL（SRQ 融合込み）を出発点に、**構造（ロード列・
ループ回数・lane・縮約木・エピローグ）を全部保ったまま積和だけを差し替えた 3 変種**を単体で A/B した。数値は壊れる前提。
bundle は `outputs/bench/karume/2026-09-19_18-31-28_k45-stage0-dceb0046/`（git 追跡外・README に手順と全 WGSL）。

| 変種     | 活性のロード（1 語あたり）          | 積和                                                                                         |
| -------- | ----------------------------------- | -------------------------------------------------------------------------------------------- |
| current  | `vec4<f32>` を 16 本（i4 は 8 本）  | f32 復元 + 要素ごと FMA（製品そのまま・i2 は 1 要素 6.25 命令）                              |
| minimal  | packed int8 の `vec4<u32>` **1 本** | `dot4I8Packed` × 4 + 語あたり scale 乗算 1 回（1 語 16 要素しか消費しない = **利得の上界**） |
| noalu    | current と同じ 16 本                | XOR 累算だけ（= 現行のロード列の床）                                                         |
| noaluMin | minimal と同じ 1 本                 | XOR 累算だけ（= 重み転送の床）                                                               |

手順: `acquireGpu({gpuTiming:true})` で製品と同じ device・合成重み（L2 に乗らないよう重みバッファ 4 本を回す）・opbench の heater で
クロック固定（サンプルごと）・1 pass に reps 本の dispatch を積み timestamp で ns/dispatch・回文順 ABBA × 4（各変種 8 サンプル）・
ブロック前に 75℃ 以下を待つ。本走はヒーターで 89〜90℃ / sw thermal slowdown が混ざったため、門に掛かる ② と ③ は 62℃ 待ちで再走して一致を確認した。

| 形（QAT decode の実キー）            | current | minimal | noalu | noaluMin | 算術ぶん | 活性ロードぶん |  床（実効帯域） |
| ------------------------------------ | ------: | ------: | ----: | -------: | -------: | -------------: | --------------: |
| ① i2 12288×1536 lanes 2（gate / up） | 25.1 µs |     8.8 |  19.3 |      9.0 |  **23%** |            41% | 36%（527 GB/s） |
| ② i2 1536×12288 lanes 32（down）     |    49.2 |     7.9 |  49.1 |      8.1 | **0.3%** |        **83%** | 17%（581 GB/s） |
| ③ i2 262144×1536 逐次（lm_head）     |   291.9 |   140.4 | 185.9 |    139.9 |  **36%** |            16% | 48%（719 GB/s） |
| ④ i4 g512 6144×1536 lanes 4          |    19.7 |     8.5 |  18.1 |      8.8 |   **8%** |            48% | 45%（536 GB/s） |

（min ベース。算術ぶん = current − noalu、活性ロードぶん = noalu − noaluMin。`minimal − noaluMin` は 4 形とも ≤ 3% =
**`dot4I8Packed` の算術自体はタダ**。本ハーネスの current は §13.3 の補正値の 1.2〜1.35 倍で系統的に大きい — 比だけを読む。）

### 14.1 判定 — 門は通った（holds）。ただし効く理由は当初の帰属と違う

- ③（占有率飽和の対照）は minimal で 2.1 倍（−52%）、① も 2.7〜2.9 倍で、門の 25% を大きく超える → **K-45 の速度動機は refuted されない**。
- しかし **「INT2 の f32 復元が 6.25 命令/要素だから遅い」は ② と ④ で成り立たない**（語あたり ALU を 83% 削っても 0.3% / 8%）。
  時間を食っているのは**活性のロード本数**（1 語あたり 16 本 / i4 8 本）で、これを 1 本にすると ② は 6 倍・④ は 2.3 倍。
  算術が効くのは ③（36%）と ①（23%）だけ。
- 忠実な K-45（活性 packed int8 を 64 要素 = 4 本読み、i2 を int8 レーンへ展開する命令を払う）を線形モデルで外挿すると
  **② 約 3 倍・③ 1.7〜2.0 倍（推測）**。minimal の 6 倍 / 2 倍を期待値として扱ってはいけない。

### 14.2 設計への含意（段 1 の形を変える — 要判断）

1. **効くレバーは「活性を packed int8 で持つこと」そのもの**（= クラスタ C の S4-2・E の presrq と同じ部品）。②④ では算術がタダなので、
   `unpack4xI8` で int8 活性を f32 へ戻し**今の f32 FMA のまま**積和してもロードは 16 → 4 本に落ちる。`dot4I8Packed` の可搬性
   （M2 の Metal で native に落ちるか・§11 の 10）に賭けずに利得の大半が取れる可能性がある。
2. ③（lm_head・逐次）だけは算術が 36% で、ここは整数内積が効く。形ごとに効くレバーが違う。
3. 床は 527〜719 GB/s（ピーク 912 の 58〜79%）で、活性ロードを潰した後の GEMV は重み転送の床に近づく。③ の残り代は 2.1 倍が上限。

## 15. H-28 段 ① — PLE の GPU 常駐（opt-in 席・depth 1）の A/B（2026-09-19・RTX・main `118b2c3`）

実装は `d1c848e`（`pleResidency: "gpu"`・ADR 0085 追記）。GPU 内 gather は golden ともホスト gather とも u32 一致、
生成文は全対で同一。条件は §13 と同じ（QAT E2B `i4-fast`・capacity 4096 / chunk 768・64 token）。
生値は [結果 JSON](2026-09-19-h28-ple-gpu-results.json)。

| 経路                                          |                    host |                                              gpu |                                         差 |
| --------------------------------------------- | ----------------------: | -----------------------------------------------: | -----------------------------------------: |
| Chrome 壁（greedy・2 ロード × 2 prompt）      |        94.2〜95.4 tok/s |                                 87.6〜95.3 tok/s | 0〜−7%（英語 prompt で悪化・日本語は同等） |
| Chrome GPU（pass 境界）                       |           6.94 ms/token | 7.02 ms/token（+ gather 2 dispatch・pass 3 → 4） |                                   +0.08 ms |
| Chrome TTFT 英 / 日                           |          40 / 61〜65 ms |                               42〜43 / 89〜96 ms |            日本語 prompt で **+25〜30 ms** |
| Deno 壁・温度 0（greedy・64 / 256 token）     | 43.0〜43.6 / 39.5〜39.8 |                          43.1〜43.3 / 39.5〜39.6 |                                       中立 |
| Deno 壁・CLI 既定サンプラー（64 / 256 token） | 42.0〜42.3 / 38.5〜38.6 |                      **26.8〜27.1 / 24.5〜25.0** |         **+13 ms/token**（フェンス +1 本） |

判定: **単独では効かない**（クラスタ B の見積り 0.25 ms は出ない）。ホストの gather + writeBuffer はもともと GPU の陰に
隠れていて臨界路に無く、代わりに token ごとの gather pass + copy が GPU 側に 1 段増える。greedy 以外の run
（サンプリング・prefill・診断）は gather 用 batch を 1 本先行させるのでフェンスが 1 本増え、Deno では 10 ms 床がそのまま乗る。
席は opt-in・既定 host のまま据え置き、**価値は先行投入（H-27 段 ②）の前提**としてだけ残す。H-27 に進む前の残件:
①非 greedy 経路でも gather を target と同じ batch に積む（フェンス +1 を消す）②日本語 prompt の TTFT +25〜30 ms の帰属
（gather Session の行数別 plan / 常駐の初回コンパイルか）③通常 Gemma 4 E2B の i8 sidecar（2.19 GiB）は参照機の束縛上限に
載らない（QAT 専用のまま）。
