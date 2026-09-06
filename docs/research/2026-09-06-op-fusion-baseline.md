# OP / Fusion 作業前の実走ベースライン — 4 家族の op 別 GPU 時間と壁時計（2026-09-06）

> 時点スナップショット（RTX 3080 Ti / Vulkan・Deno 2.9.6・karume `3b7084d`）。道具は
> `tools/opbench/main.ts graph`（この日に siglip2 / irodori へ広げた — `3b7084d`）。性能候補の採否は
> [perf-ledger](../perf-ledger.md)、波順は [backlog](../backlog.md)。生データは
> `outputs/bench/<資産>/2026-09-06_baseline-*`（git 追跡外）。op 別の静的本数は
> [op-census 2026-09-03](2026-09-03-op-census-fusion-hints.md)、単体計測と torch 対照は
> [opbench-stage2 2026-09-04](2026-09-04-opbench-stage2.md)。

## §0 要約

- 在庫 3 件（P-5 / K-15 / K-7 再評価）の効き先 4 家族（anima / gemma4 / siglip2 / irodori）を、実走 1 回の
  op 別 GPU 時間（timing）と壁時計（wall・別プロセス）で各 2 回採った。dispatch 数は 4 家族とも census の
  素ノード本数と 1:1 で突合できた（unmapped は irodori の行ブロック attention だけ — §2.4）。
- **在庫 3 件はどれも上限が全 GPU の 2% 以下**（§3）: P-5 は anima の strided 族まるごとで 1.8〜2.1%
  （消費側の strided 読みを無料と置いた理論上限。判定線 −1.3% に対し余裕がない）、K-15 は gemma4 decode の
  `gelu_tanh` 70 本 = GPU 1.2% / 壁 ≈0.8%（kill 線 壁 −1% に届かない）、K-7 の `mul,add` 対融合は anima 0.5% /
  irodori 0.5%。
- ledger の P-5 上限（391ms = 8.4%）は `permute` の実効帯域を 120GB/s と置いた外挿だったが、実走の strided
  族は 1,468M 要素の読み書きを 20.7ms でこなしていて **≥567GB/s**。上限は 5 分の 1 に縮む。
- 時間の大所は linear（anima transformer step の 57% / gemma4 decode の 69% / siglip2 の 79〜87% / irodori DiT
  step の 63%）と attention（anima 27%）、anima では VAE decoder（全 GPU の 29%）。

## §1 方法

- 1 家族 = 実走 1 回を `graph` で 2 回ずつ、timing（timestamp・1 dispatch = 1 pass）と wall（計測無効・
  `generate` 等の壁だけ）を別プロセスで。代表値は載せず 2 回とも書く（差が再現性の目安）。
- **絶対値は同一リグ・同一セッション内でしか比較しない**（opbench README の規約）。採否の A/B は実装後に
  on / off を同一セッションで ABBA する（P-1 の流儀）。この記録は op 別の内訳表と回帰の基準値。
- 入力: anima = turbo-v1.1・既定 quant・1024²・4 step・seed 1・既定プロンプト / gemma4 = e2b i4・chat 1 ターン・
  既定プロンプト（短い prefill）・new-tokens 16（decode 15 run の平均）/ siglip2 = base と so400m・f32・
  合成画像 1 枚 / irodori = v4.1-small i8-a8・既定文 1 文・seed 0・発話長はモデル任せ（DiT 60 step）。
- irodori の timing は DiT がホスト経路で回るので壁が倍近くなる（pipeline の観測席の doc）— GPU 時間は
  読める。irodori の既定実走は 8 段のうち 5 段（backbone / text_proj / duration / dit / codec_decoder）—
  speaker / codec_encoder / caption_proj は参照音声・caption を渡したときだけ回る。

```bash
# timing（op 別 GPU 時間・census と突合）
deno run -A tools/opbench/main.ts graph --source models/karume-anima --family anima --steps 4 --size 1024 --census outputs/bench/karume-anima/2026-09-04_op-census --scenario 1024px --out outputs/bench/karume-anima/2026-09-06_baseline-timing-1
deno run -A tools/opbench/main.ts graph --source models/karume-gemma4 --family gemma4 --new-tokens 16 --census outputs/bench/karume-gemma4-e2b/2026-09-04_op-census --scenario decode --out outputs/bench/karume-gemma4/2026-09-06_baseline-timing-1
deno run -A tools/opbench/main.ts graph --source models/karume-siglip2 --family siglip2 --census outputs/bench/karume-siglip2/2026-09-06_op-census-base --scenario native --out outputs/bench/karume-siglip2/2026-09-06_baseline-base-timing-1
deno run -A tools/opbench/main.ts graph --source models/karume-siglip2 --family siglip2 --model so400m --census outputs/bench/karume-siglip2/2026-09-06_op-census-so400m --scenario native --out outputs/bench/karume-siglip2/2026-09-06_baseline-so400m-timing-1
deno run -A tools/opbench/main.ts graph --source models/karume-irodori-v4.1-small --family irodori --census outputs/bench/karume-irodori-v4.1-small/2026-09-04_op-census --scenario representative --out outputs/bench/karume-irodori-v4.1-small/2026-09-06_baseline-timing-1
# wall（同じ引数に --mode wall・--census / --scenario 無し・--out の末尾を wall-1 に）
```

siglip2 の census は配布形ミラー `models/karume-siglip2` から採り直した（component 名 `vision`。09-04 の
census は series 出力からで `model`）。

## §2 結果

### §2.1 anima 1024² 4 step

| 量                              |        1 回目 |        2 回目 |
| ------------------------------- | ------------: | ------------: |
| 全 GPU（timing・15 run）        |    4,625.5 ms |    4,501.4 ms |
| transformer step ×4             |    3,298.4 ms |    3,158.6 ms |
| vae_decoder タイル ×9           |    1,285.8 ms |    1,301.4 ms |
| text_encoder / text_conditioner | 31.7 / 9.6 ms | 31.8 / 9.6 ms |
| dispatch（全 run）              |        13,309 |        13,309 |
| 壁（wall・generate・load 除く） |      8,266 ms |      8,197 ms |

transformer 1 step（2,316 dispatch・census 素ノード 1,383 + attention 3 本 / ノード + quantize_rows）:

| キー先頭語      | dispatch | 1 回目 ms | 2 回目 ms |
| --------------- | -------: | --------: | --------: |
| linear          |      454 |     458.9 |     451.1 |
| attention_pv    |       56 |      99.1 |      93.1 |
| attention_qk    |       56 |      75.2 |      72.2 |
| attention_stats |       56 |      48.0 |      44.3 |
| quantize_rows   |      622 |      37.6 |      33.3 |
| ew              |      281 |      30.1 |      27.0 |
| rms_norm        |      113 |      29.3 |      28.5 |
| strided         |      535 |      23.8 |      20.7 |
| adaln_norm      |       85 |      11.7 |      10.6 |
| rope            |       56 |      11.0 |       8.9 |

ew の内訳（2 回目・step あたり）: `add` r3 84 本 10.24ms / `gelu`（erf 型）28 本 9.17ms / `mul` r3 84 本
7.28ms / `add` r2 85 本 0.30ms。strided はキー 1 種（`strided:v1:f32:r4:wg256`）で、census の実体化
`permute` 224 本 / 1,468.0M 要素を含む。

### §2.2 gemma4 e2b i4・chat 1 ターン（new-tokens 16）

| 量                                            |   1 回目 |   2 回目 |
| --------------------------------------------- | -------: | -------: |
| decode GPU / token（15 run 平均）             | 20.37 ms | 21.88 ms |
| decode dispatch / token                       |    1,373 |    1,373 |
| prefill GPU（短いプロンプト・1,478 dispatch） | 473.9 ms | 372.8 ms |
| 壁（wall・ターン全体・load 除く）             | 2,647 ms | 2,478 ms |

decode 1 token:

| キー先頭語            | dispatch | 1 回目 ms | 2 回目 ms |
| --------------------- | -------: | --------: | --------: |
| linear_gemv           |      276 |      9.31 |      9.88 |
| linear（lm_head i8）  |        1 |      4.85 |      5.27 |
| attention_state_qk    |       35 |      1.93 |      2.09 |
| rms_norm              |      242 |      1.45 |      1.57 |
| ew                    |      427 |      1.44 |      1.56 |
| strided               |      205 |      0.69 |      0.75 |
| strided_write         |       70 |      0.21 |      0.23 |
| attention_state_stats |       35 |      0.18 |      0.20 |
| attention_state_pv    |       35 |      0.16 |      0.18 |
| state_append          |       30 |      0.10 |      0.11 |
| rope                  |       15 |      0.05 |      0.05 |

ew の内訳（2 回目・token あたり）: `mul` r3 108 本 0.408ms / `add` r3 105 本 0.370ms / `mul` r4 71 本
0.261ms / **`gelu_tanh` r3 70 本 0.251ms** / `add` r4 36 本 0.131ms / `neg` r4 35 本 0.126ms。1 dispatch ≈
3.6µs で、M=1 の要素ごとは launch 律速。prefill（2 回目）: linear 277 本 345.3ms / ew 487 本 13.4ms / rms_norm
4.63ms / attention_state_pv 3.67ms / strided 235 本 3.07ms + strided_write 100 本 1.0ms。

### §2.3 siglip2（f32・合成画像 1 枚・run 1 本）

| 量                           | base 1 回目 | base 2 回目 | so400m 1 回目 | so400m 2 回目 |
| ---------------------------- | ----------: | ----------: | ------------: | ------------: |
| GPU                          |    17.52 ms |    14.80 ms |     125.39 ms |     124.26 ms |
| dispatch                     |         274 |         274 |           589 |           589 |
| 壁（wall・embed・load 除く） |       79 ms |       79 ms |        231 ms |        227 ms |

| キー先頭語 | base dispatch | base ms（1 / 2 回目） | so400m dispatch | so400m ms（1 / 2 回目） |
| ---------- | ------------: | --------------------: | --------------: | ----------------------: |
| linear     |            78 |         15.22 / 12.85 |             168 |           98.64 / 97.73 |
| bmm        |            26 |           0.70 / 0.59 |              56 |           17.23 / 17.09 |
| softmax    |            13 |           0.32 / 0.27 |              28 |             3.95 / 3.91 |
| ew         |            65 |           0.39 / 0.33 |             140 |             2.46 / 2.45 |
| strided    |            65 |           0.33 / 0.28 |             140 |             1.88 / 1.86 |
| layer_norm |            26 |           0.21 / 0.18 |              56 |             0.93 / 0.93 |
| conv2d     |             1 |           0.35 / 0.31 |               1 |             0.30 / 0.30 |

融合ヒットは identityExpand（別名化）53 / 113 だけ。分解 attention（`bmm` + `softmax` + `bmm`）は so400m で
21.2ms = 17%。

### §2.4 irodori v4.1-small i8-a8・発話 1 本（DiT 60 step）

| 量                                        |              1 回目 |              2 回目 |
| ----------------------------------------- | ------------------: | ------------------: |
| 全 GPU（timing・64 run）                  |          2,332.6 ms |          2,315.0 ms |
| dit 1 step（60 run 平均・1,596 dispatch） |            36.29 ms |            36.15 ms |
| codec_decoder                             |            139.9 ms |            131.4 ms |
| backbone / duration / text_proj           | 13.6 / 1.4 / 0.2 ms | 13.5 / 1.4 / 0.2 ms |
| 壁（wall・generate・load 除く）           |            3,443 ms |            3,333 ms |
| 壁（timing 有効 = DiT ホスト経路）        |            8,215 ms |            8,016 ms |

dit 1 step:

| キー先頭語    | dispatch | 1 回目 ms | 2 回目 ms |
| ------------- | -------: | --------: | --------: |
| linear        |      317 |     22.72 |     22.62 |
| bmm           |       24 |      3.72 |      3.71 |
| ew            |      445 |      2.41 |      2.40 |
| rms_norm      |       87 |      2.24 |      2.23 |
| quantize_rows |      317 |      1.80 |      1.79 |
| strided_write |      216 |      1.61 |      1.61 |
| strided       |      161 |      1.26 |      1.25 |
| safe_softmax  |       12 |      0.47 |      0.46 |
| silu          |       17 |      0.07 |      0.07 |

ew の内訳（2 回目・step あたり）: `mul` r4 132 本 0.95ms / `add` r3 144 本 0.51ms / `add` r4 36 本 0.43ms /
`mul` r3 72 本 0.31ms / `neg` 24 本 0.08ms / `tanh` 24 本 0.08ms / `sigmoid` 12 本 0.04ms。

突合の `unmapped_keys` に `bmm` と `safe_softmax` が出る: dit の行ブロック attention（融合 12 ヒット）は
census 側で `fused_by = rowBlockAttention` の吸収ノードだが、実行はその融合ステップが `bmm` /
`safe_softmax` のカーネルキーで走るので、`compareWithCensus` の op 写像が census の素ノードに当てられない。
数値は正しく、写像の欄が無いだけ（道具の宿題 — §4）。

## §3 在庫 3 件の上限（この実測から）

| 候補 | 対象の実測                                                                                                            | 上限（消えうる最大）                                                                                             | 判定線                      |
| ---- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------- |
| P-5  | anima transformer の strided 族 535 本 20.7〜23.8ms / step（うち `permute` 224 本 1,468M 要素）                       | 83〜95ms / 4 step = **全 GPU の 1.8〜2.1%**                                                                      | 全 GPU −1.3%（= 59〜60ms）  |
| K-15 | gemma4 decode の `gelu_tanh` 70 本 0.251ms / token（消費側 `mul` r4 71 本 0.261ms）                                   | 0.25ms = **decode GPU の 1.2% / 壁 ≈0.8%**                                                                       | 壁 −1%                      |
| K-7  | anima `mul` r3 84 本 7.28 + `add` r3 84 本 10.24ms / step・irodori `mul` r3 72 本 0.31 + `add` 72 本分 ≈0.25ms / step | 対融合 = 中間 1 本の書き + 読み ≈ 1/3 → anima ≈5.8ms / step = **全 GPU 0.5%**・irodori ≈0.19ms / step = **0.5%** | 未設定（前回棄却は壁 1.8%） |

- **P-5**: `permute` の実効帯域は実走で ≥ 1,468.0M × 4B × 2 ÷ 20.7ms = **567GB/s**（strided 族の時間に
  slice / cat も含むので下限）。ledger の 391ms（120GB/s）は外挿で、実走はその 5 分の 1。上限 1.8〜2.1% は
  「消費側（attention / linear）が strided のまま読んで追加費用ゼロ」という置き方で、GEMM のタイル充填へ
  strided 読みを持ち込むと逆行しうる（ledger の注記）ので実効はこれより小さい。siglip2 の strided は
  base 1.9% / so400m 1.5%、gemma4 prefill（短文）は strided + strided_write 4.1ms = 1.1%。irodori DiT は
  strided + strided_write 2.86ms = **step の 7.9%** だが `cat`（strided_write 216 本）が主で、`permute`
  限定の P-5 の射程外。
- **K-15**: 中間は M=1 × 12,288 の f32 = 48KiB で帯域は無関係、消えるのは dispatch 70 本ぶんの launch
  （≈3.6µs × 70 = 0.25ms）+ ホスト 0.62µs × 70 = 0.04ms。壁 32ms / token（[decode-wallclock](2026-08-30-gemma4-decode-wallclock.md)
  §7.2）に対し 0.9% が最大で、kill 線（壁 −1%）に届かない。ledger の着手条件「70 dispatch ぶんの往復を
  単体で測る」はこの実走値で代替できる。
- **K-7**: 対融合（`mul` + `add` を 1 カーネル）の利得は中間 1 本の往復ぶんで、両家族とも 0.5%。linear の
  epilogue へ 3 つとも畳む形なら anima で ew 17.5ms / step が消える（全 GPU 1.6%）代わりに GEMM が gate と
  residual を読む 2 本（84 × 67MB ≈ 5.6GB ≈ 10ms / step）が乗り、正味 ≈0.7%。

## §4 含意と道具の宿題

1. 在庫 3 件は上限がそれぞれ 2% 以下で、P-5 / K-15 は自分の判定線に紙の上で届かない。着手するなら
   「dispatch ダイエットの積み上げ」として複数を束ねる前提で、単独の kill 線は満たせない。
2. 時間の大所は linear と attention。gemma4 decode は GEMV（K-14 / K-16）、anima は linear 57% +
   attention 27% + VAE 29%（全 GPU）、siglip2 は linear 79〜87% + 分解 attention 17%（so400m）、
   irodori DiT は linear 63%。
3. 道具の宿題: ①wall モードの `dispatch_count` は `submit.dispatchCount`（累計）を写していて run 別に
   ならない（`graph.ts` の `recordRun` — timing 無効時の値。壁の数値は無関係）②行ブロック attention の
   融合ステップは `bmm` / `safe_softmax` のキーで走るので `compareWithCensus` で unmapped になる
   （fused バケットへの写像が要る）③opbench の USAGE 1 行目が `<census|single>` のまま（graph / torch
   が抜けている）④irodori の `graph` は参照音声 / caption の受け口が無く 8 段中 5 段。
