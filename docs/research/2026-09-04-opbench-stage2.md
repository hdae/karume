# OP マイクロベンチ 2 段目 — `opbench single` / `graph` / `torch` と Fusion 2 段目 `inductor`（2026-09-04）

> 時点スナップショット（RTX 3080 Ti / Vulkan・Deno 2.9.x・torch 2.13.0+cu130 / CUDA 13.0）。道具の正本は
> `tools/opbench/`・`tools/fusion-hints/`（README が使い方）、性能候補の採否は
> [perf-ledger](../perf-ledger.md)、波順は [backlog](../backlog.md)。生データは `outputs/bench/<資産>/2026-09-04_*`
> （git 追跡外）。1 段目（静的 census / 候補列挙）は [2026-09-03](2026-09-03-op-census-fusion-hints.md)。

## §0 要約

- **計測規約を実装として内蔵した**（`tools/opbench/bench.ts`）: 1 パス ≈80ms まで反復を積む・代表値は min・
  timing（timestamp・1 dispatch = 1 pass）と wall（フェンス 1 本の区間に enqueue を束ねる）は別欄で倍率を
  作らない。**新規の規約 = クロック張り付けの filler**（§1.1）。
- **合格線**: K-11 の census 加重（gemma4 decode の linear f32+i4g32・12 形 / 276 本）は単体で **9.05ms** —
  ADR 0082 の 7.38ms に +22.6%（±25% 帯内）。実グラフ内の同 op との比 **single / graph = 1.01**（lm_head 込み）。
  P-1 の `quantize_rows` 小 D 変種（`:r8w32` / `:r4w64` / `:r2w128`）は anima 1024² の transformer step で
  **226 dispatch = research 2026-09-03 と同数**で立ち、短い行の合計 10.9ms/step（同 research 7.4ms/step の 1.45×・
  timing ON・同桁）。
- **census = 実行 1 回の dispatch** が実資産で確認できた: gemma4 decode は linear 277 / rms_norm 242 /
  state_append 30 が 1:1、attention は 35 ノード = 105 dispatch（3 / ノード）。anima transformer step は
  linear 454 / rms_norm 113 / add 169 / gelu 28 が 1:1、attention 56 ノード = 168 dispatch。
- **列 B（torch の到達点）**: §3。
- **Inductor の融合決定 × 候補表**: §4。

## §1 `single` — 加重行の単体計測

### §1.1 クロックは軽い dispatch の反復では張り付かない（新規の規約）

M=1 の GEMV（48 workgroup）を 1 パス 80ms 積んで反復しても、RTX 3080 Ti は **P8 / P5**（SM 210〜900MHz・
mem 405〜810MHz）に留まった（`nvidia-smi -l 1` を並走させて実測）。同じ 12 形の加重合計は **35.9〜38.8ms**
で、k=12288 の GEMV は 827µs / dispatch。f32 linear [2048,4096]×[4096,4096] を 8 本流す **filler（heater）**
を「計測前に連続 3 回が min の +5% 以内に収まるまで」+「round ごとに 1 回」挟むと **P3**（SM ≈1.35GHz /
mem 5GHz）で安定し、同じ形が **157µs**（5.3 倍）、加重合計 **9.05ms** になった。研究 2026-08-10 §5 の
「対の前にメモリクロックが張り付くまで空回し」は重い kernel でしか成立しない、が実測の中身。P0
（1665MHz / 9501MHz）には届いていない — filler をさらに重くする余地はあるが、合格線は P3 で満たした。

### §1.2 K-11 の照合（gemma4 decode・linear f32+i4g32）

| n     | k     | reps | ns / node | count |  加重 ms |
| ----- | ----- | ---: | --------: | ----: | -------: |
| 1536  | 12288 |  324 |   156,672 |    20 |    3.133 |
| 12288 | 1536  | 1024 |    36,725 |    40 |    1.469 |
| 1536  | 6144  |  228 |    65,699 |    15 |    0.985 |
| 256   | 1536  | 1024 |    14,083 |    59 |    0.830 |
| 6144  | 1536  | 1024 |    25,050 |    30 |    0.751 |
| 1536  | 2048  | 1024 |    24,600 |    28 |    0.688 |
| 2048  | 1536  | 1024 |    15,247 |    28 |    0.426 |
| 1536  | 256   | 1024 |     7,169 |    35 |    0.250 |
| 1536  | 4096  | 1024 |    31,801 |     7 |    0.222 |
| 4096  | 1536  | 1024 |    21,150 |     7 |    0.148 |
| 512   | 1536  | 1024 |    16,674 |     6 |    0.100 |
| 8960  | 1536  | 1024 |    40,016 |     1 |    0.040 |
| 合計  |       |      |           |   276 | **9.05** |

全形が `linear_gemv:v1:f32:c32u4:wi4g32`。lm_head（i8 [262144,1536]・1 本）は 4.67ms / node（K-16 の decode
研究値 5.41ms と同桁）。k が大きく n が小さい形（down-proj）ほど遅い = GEMV の workgroup 数が n/32 で決まり
k 方向が 1 スレッドの逐次になる（K-14 と同じ構造）。

### §1.3 gemma4 decode の全 op

88 行を測定、除外 24（rope 融合 12・別名化 10・i32 活性 2）・state を触る op 3（除外理由つき）。加重合計の
上位: linear/i4g32 10.6ms・linear/i8 4.6ms・rms_norm 1.66ms・add 0.62・mul 0.56・permute 0.43・slice 0.42・
attention（融合 3 入力）0.30・gelu_tanh 0.28。

### §1.4 anima の linear（session `a8`）

census の `session`（quant 宣言）を写すと `linear:v4:i8a8:tile128x64r8x8w8x16k16v4:dp4a` + `quantize_rows`
の 2 dispatch / node で立つ（session を渡さないと f32 の linear を測ってしまう — 1 段目レビューの指摘②が
実害になる場面）。[1,4096,2048]×[2048,2048] = 1.54ms / node × 168 = 259ms。

## §2 `graph` — 実資産 1 run の突合

### §2.1 gemma4（chat 1 ターン・new-tokens 8）

prefill 1,478 dispatch 371.6ms・decode 1,373 dispatch **≈19.4ms / run** × 7（ADR 0082 の「decode GPU 20ms」と
一致）。decode の内訳: linear 277 dispatch 13.5ms・attention 105 dispatch 2.0ms・rms_norm 242 1.37ms・
要素ごと 427 1.36ms・strided 275 0.85ms・state_append 30 0.09ms・rope（融合）15 0.05ms。

### §2.2 anima 1024²（2 step）

text_encoder 873 dispatch 31.7ms・text_conditioner 445 9.5ms・**transformer 2,316 dispatch 969 / 783ms**・
vae_decoder タイル 303 dispatch ≈142ms × 8。transformer step の内訳: linear 454 = 449ms・attention 168
dispatch = 210ms（pv 107 / qk 79 / stats 52）・quantize_rows 622 = 33.5ms・rms_norm 113 = 28ms・strided 535 =
21ms・fused 143（rope 56 + adaln 85 + silu 2）= 20ms・add 169 = 10.5ms・gelu 28 = 9.1ms。

### §2.3 単体 × 加重と実グラフ内の関係

gemma4 decode の linear: single の加重合計（lm_head 込み 13.66ms）/ graph の同 op（13.54ms）= **1.01**。
K-11 当時の「単体は実グラフ内より 15〜25% 小さい」（62.38 vs 73.28）は再現せず、filler で張り付けた
クロックの下では単体と実グラフが一致した — 当時の差はクロック状態の差だった可能性が高い（推測・
当時の台本は残っていないので確定できない）。

## §3 `torch` — 列 B（PyTorch eager・CUDA）

gemma4 decode の 88 行を torch eager で組み直して測った（`tools/opbench/torch_bench.py`・列 = f32〈TF32 off〉/
f32_tf32 / f16 / bf16・heater と min は karume 側と同じ規約・skipped 0）。比は case ごとの
karume ms / torch ms の中央値（>1 = karume が遅い）、ms は census 加重の合計。

| op        | cases | karume ms | torch f16 ms | 比 f16 | 比 bf16 | 読み                                                                            |
| --------- | ----: | --------: | -----------: | -----: | ------: | ------------------------------------------------------------------------------- |
| linear    |    13 |     14.39 |         7.75 |   1.18 |    1.18 | i4 GEMV + i8 lm_head 対 cuBLAS f16。総和は 1.86×（down-proj と lm_head が支配） |
| rms_norm  |     6 |      1.69 |         1.70 |   0.76 |    0.73 | karume が速い（1 行 = 1 workgroup の reduce）                                   |
| attention |     1 |      1.19 |         1.47 |   0.81 |    0.81 | 融合 attention（f32）対 SDPA f16                                                |
| mul       |    10 |      0.70 |         0.78 |   0.96 |    0.94 | 要素ごとは同等                                                                  |
| add       |     4 |      0.59 |         0.59 |   1.01 |    1.01 |                                                                                 |
| gelu_tanh |     3 |      0.29 |         0.30 |   1.04 |    1.01 |                                                                                 |
| neg       |     2 |      0.15 |         0.15 |   1.03 |    1.04 |                                                                                 |
| slice     |    39 |      0.43 |         0.66 |   2.50 |    2.49 | 加重合計は karume が小さいが case 比は 2.5×（小さい形の固定費）                 |
| permute   |     6 |      0.43 |         0.12 |   3.60 |    3.98 | strided 実体化 — P-5 の対象                                                     |
| cat       |     2 |      0.29 |         0.17 |   1.70 |    1.69 | 同上（strided_write）                                                           |

含意: gemma4 decode で列 B に**負けているのは linear（1.18 / 総和 1.86×）と strided 族（permute 3.6× / cat 1.7× /
slice 2.5×）**、要素ごと・rms_norm・attention は同等以上。linear は「4-bit の重みを読んでいるのに f16 cuBLAS より
遅い」= 帯域ではなく GEMV の並列度（n/32 workgroup・k 逐次）が律速で、K-14 と同じ構造の改善余地。strided 族は
P-5（permute の consumer 畳み込み）の根拠が torch 対照でも裏付いた。列 compile_f16（Inductor）は `--compile true`
で採れる（今回は eager のみ — gcc は home-manager で導入済み）。

## §4 `inductor` — Inductor の融合決定 × 候補表

`tools/fusion-hints/main.ts inductor` で、exporter の golden 台帳 31 本 + 鎖モデル 5 本（残差 rms_norm・
gated gelu・RoPE 半回転・softmax 鎖・adaLN 変調）を CUDA 上で export し、Inductor の Scheduler が
融合後に持つノード群を IR の op 名列へ写して、11 資産の候補表（4,182 行・相異なる鎖 2,263）と突き合わせた
（生データ = `outputs/bench/inductor/2026-09-04_probe/`）。

| 判定       | 相異なる鎖 | 意味                                                                      |
| ---------- | ---------: | ------------------------------------------------------------------------- |
| fused      |         20 | Inductor がその op 列を 1 カーネルに畳んだ群がある（witness = モデル#群） |
| split      |        420 | 全 op は観測したが 1 群には入らない                                       |
| unobserved |      1,823 | 鎖の op のどれかが probe のどの群にも現れない                             |
| trivial    |          — | reshape だけの鎖（0 dispatch）                                            |

- **fused**: adaLN 変調の `mul,add` / `add,mul,add` / `slice,slice,add` 系（witness = chain_adaln#2）。Irodori DiT
  の `mul,add` 72 本・anima transformer の `mul,add` 84 本がここに当たる = **K-7 の「rms_norm ベース adaLN」側は
  Inductor も同じ鎖を畳む**（再評価の材料）。
- **split**: RoPE 半回転の `slice,slice,cat` / `slice,cat`（96 本）、分解 attention の `expand,reshape,bmm`
  （96 本）、`slice,reshape,permute`（144 本）。Inductor も cat の実体化と bmm の境界で切る = 汎用の
  pointwise 融合では取れない鎖で、karume の専用カーネル（rope / 行ブロック attention）と同じ判断。
- **unobserved が 8 割を占める限界**: probe の群に現れた IR op は要素ごと・layout 系（add / mul / slice /
  permute / cat / expand / bmm …）だけで、**normalize が合成する op（`linear` / `rms_norm` / `rope` / `attention` /
  `silu`）は 1 つも現れない**。Inductor 側は分解のまま（mm + add・pow / mean / rsqrt …）で fx 名が別物になり、
  IR 側の名前（normalize が付け直す）と join できないため。`rms_norm,add` 106 本や `linear,rms_norm,add` 105 本が
  unobserved なのはこの機序で、Inductor が畳まないことを意味しない。**改善 = normalize が畳んだ元ノード名の
  出自を IR ノードへ残し、それで join する**（backlog 起票 — exporter の normalize に 1 欄）。
- 失敗 1 本 = golden `broadcast_binary`（forward 内の CPU 定数と CUDA 入力の混在・probe 側の device 寄せで
  拾えない形）。

## §5 含意（波 c / Fusion 実装への入口）

1. **K-14 / K-16（decode の GEMV）**: 列 B 対照で linear だけが負ける（case 比 1.18・加重合計 1.86×）。単体の形別
   表（§1.2）で加重の 35% を占めるのは down-proj（k=12288 → n=1536・157µs）と lm_head（i8・4.67ms）で、どちらも
   「workgroup 数が n/32・k 方向が 1 スレッド逐次」の GEMV 形 — K-14 の①QK 並列化と同じ構造の改善が linear
   側にも効く見込み（perf-ledger K-14 / K-16 の根拠欄へ）。
2. **P-5（permute の consumer 畳み込み）**: karume の permute は torch の `.permute().contiguous()` の 3.6〜4.0×、
   cat 1.7×。実グラフ内でも gemma4 decode の strided 族は 275 dispatch 0.85ms、anima transformer step は
   535 dispatch 21ms。実装スパイクへ進む根拠が対照側で揃った。
3. **rms_norm / attention / 要素ごとは列 B と同等以上** — ここに性能候補を立てる理由は現時点で無い。
4. **Fusion**: RoPE と分解 attention は Inductor でも汎用融合の外（専用カーネル路線の裏付け）。adaLN 型の
   `mul,add` 鎖は Inductor が畳む = K-7 再評価の対象は「偶奇 RoPE」ではなく「rms_norm ベース adaLN」側。
5. **道具の宿題**: ①probe の join を normalize の出自で行う ②`single` の Metal 対応（wall モードは実装済み・
   timing は Metal の timestamp 不能で使えない）③`graph` の他家族（現状 gemma4 / anima）。
