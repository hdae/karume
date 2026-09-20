# 0105: 固定SRQの活性をpacked int8で並列GEMVへ渡す

- Status: accepted（2026-09-19・K-45 段 1a として利用者が「a を優先」と裁定した範囲）。速度の採否は実測後に判断し、
  quant宣言・manifest語彙への昇格はそこから先。
- 関連: [0097](0097-gemma4-qat-integration.md)、[0098](0098-linear-gemv-parallel.md)、
  [0103](0103-linear-static-quantize-fusion.md)、[0104](0104-gemma-fast-quant.md)、
  [0040](0040-fusion-pass.md)、[0058](0058-numeric-opt-in.md)、[0022](0022-no-runtime-autotune.md)
- 根拠: [QAT decode 速度の帰属](../research/2026-09-19-qat-speed-recon.md) §14（段 0 の切り分け）

## 問題

QAT E2B の decode で時間を食っているのは**活性のロード本数**であって算術ではない。
段 0 の単体 A/B（research §14）は、重み 1 語あたりの活性ロードを 16 本から 1 本へ落とすと
down 形が 6 倍・i4 g512 形が 2.3 倍になる一方、語あたり ALU を 83% 削っても down 形は 0.3% しか動かない、
と測った。現在の並列GEMVは重み 1 語（i2 = 64 要素 / i4 = 32 要素 / i8 = 16 要素）に対して
活性を `vec4<f32>` で 16 本 / 8 本 / 4 本読んでいる。

活性の生産元は固定SRQ（`static_quantize`）で、値はもともと int8 の 256 段しか取らない。
f32 で受け渡しているぶんだけ、読む語数が 4 倍になっている。

## 決定

### 1. 席

`SessionOptions.packedStaticQuantize?: boolean`（既定 false）。
true は `linearGemvReduce: "parallel"` と `linearCompute: "f32"` の組合せだけを受理し、
未対応の組合せと boolean 以外の値は Session 構築時に拒否する（ADR 0103 と同じ流儀で、
黙って f32 経路へ落とさない）。`@karume/models` の Gemma 共通 pipeline オプションから
target / drafter の Session へ同じ値を渡し、CLI は `--packed-static-quantize <true|false>`。

manifest 所有の語彙（hub の `SessionSpec`）には席を作らない。採用が決まるまでは**呼び手の明示指定だけ**で入る。

### 2. 対付け（プラン時の判断・IR は不変）

席が立っているとき、次を全て満たす `static_quantize` ノードだけを packed で受け渡す。

1. 融合されず**素のノードとして残った**もの。linear→SRQ 融合（ADR 0103）に飲まれた SRQ は
   出力側エピローグで、その packed 出力は本段の範囲外。
2. 出力が graph output でなく、入出力とも f32、最終次元 k が 16 の倍数、scale が正・有限。
3. 消費先が 1 本以上あり、**その全てが**並列GEMV（`linear_gemv_parallel:*`・SRQ 融合エピローグの
   有無を問わない）へ落ちる linear の**活性スロット**（`ins[0]`）であること。
   GEMM・逐次GEMV・subgroup変種・他 op が 1 本でも混ざれば f32 のまま。
4. その値を消費する融合ステップが linear→SRQ 融合以外に無いこと。packed で読む綴りを持つのは
   その融合だけなので、別のルールが `linear` を窓へ入れた瞬間、その融合カーネルは packed の語を
   f32 として読む。現行 7 ルールの窓に `linear` は 1 本も無いので今は発火しないが、
   受理側に置かないと**追加したルールだけ**が黙って壊れる。

「並列GEMVへ落ちるか」は `linearGemvParallelEligible`（格納・`gemmUsesVec4`・`k % 刻み`・
i4 の group 長・実測形の lane 表）1 本で決め、対付け（fusion.ts）と recipe-builder の門が同じ述語を読む。
別々に持つと、片方だけ広いときに `vec4<u32>` 束縛へ f32 の語が流れる = 例外なしの沈黙誤値になる。

判定の入力に「素のノードとして残ったか」が要るので、`planFusions` は席が立っているときだけ**走査を 2 度**回す。
1 度目は対付けの入力を得るため、2 度目が結果。packed の有無はどのルールの match 条件にも入らない
（変わるのは linear→SRQ 融合が組む dispatch のキー・WGSL・params だけ）ので、2 度の走査は同じ窓を同じルールが掴む。

融合カウンタ（ADR 0040）に `packedStaticQuantize` を足す。QAT E2B の decode 計画は **210**、
prefill 計画（M ≥ 32・GEMM）は 0 で、linear→SRQ 融合の 275 とは重ならない
（あちらの出力側の消費先には linear 以外が混ざる）。融合の on / off でこの数は動かない。

### 3. packed の並びと復元（ビット同一の根拠）

出力は u32 1 語 = int8 コード 4 個で、要素 i のコードは語 `i >> 2` のバイト `i & 3`
（`pack4xI8` / `unpack4xI8` と同じリトルエンディアン順）。コードは**現行の `quantize` が確定する
level / sign そのもの**で、境界表と二分探索の字面は f32 経路と共有する。
出力テンソルの実体は宣言 shape のまま確保し（実際に書くのはその 1/4）、アリーナのバケットを動かさない。

消費側の復元は **`vec4<f32>(unpack4xI8(語)) * x_scale`**（要素ごとの f32 1 乗算）。
これは生産側の出力値表（`staticQuantizeParams` の `params[129 + level]` = `Math.fround(level * scale)`）と
要素ごとに u32 一致する:

- `level` は 0..128 で f32 に厳密。`scale` は f32 厳密（params 側の門）。
- 両者の積は仮数 31 bit 以下なので **f64 で厳密**。表側（f64 の積 → f32 へ丸め）と
  WGSL 側（f32 乗算 = 正しく丸めた積）は同じ「厳密な積を正しく丸めた f32」になる。
- 符号は積の符号 = 表引き側の `| sign` と一致（level ≥ 1）。

したがって**消費側へ表（uniform 129 語）を渡す案は採らない**。表引きは構造的にビット同一だが、
並列GEMVの Dims に 2 本目の 65×vec4 を積むうえ、引き方も 1 命令では済まない。
上の証明で乗算 1 個に落ちるなら、そちらが安い。

`x_scale` は **Dims の最終メンバ**に 1 語だけ足す（非融合 packed は語 3、SRQ 融合エピローグ付き packed は
既存の 264 語の後ろ = 語 264）。既存キーの params の並びは 1 語も動かない。

**int8 に席の無い 2 値だけは f32 経路と挙動が違う**（ADR 0058 の数値 opt-in の範囲で受け入れる）。

- `-0.0` はコード 0 = `+0.0` へ落ちる。GEMV の積和は `acc` が `+0.0` 始まりで、`±0.0` の加算では
  `acc` が動かない（`+0.0 + (-0.0) = +0.0`・非零の `acc` は不変）ので、**出力は動かない**。
  これは論証ではなく実測命題として扱い、`-0.0` を含む活性での u32 完全一致を門にする。
- `NaN` は境界表の外側として ±127 / -128 へ**飽和**する（f32 経路は NaN をそのまま流す）。
  `±Inf` は f32 経路も同じ表で飽和するので一致する。

### 4. GEMV の packed 活性変種

並列GEMV（i2 / i4 / i8 × lane 2〜32 × SRQ 融合エピローグあり / なし）に、活性束縛を
`array<vec4<u32>>` に替えた変種を足す。**幾何・lane 表・K の巡回配分・workgroup の加算木・
bias の加算順・出力側エピローグは 1 バイトも変えない**。変わるのは

- 行頭の quad 添字が `dims.k / 4u` から `dims.k / 16u` になること、
- 語を 4 quad に 1 度だけ読み、quad ごとに上の式で f32 へ戻すこと、

の 2 点だけ。結果、重み 1 語あたりの活性ロードは **i2 16 → 4 本・i4 8 → 2 本・i8 4 → 1 本**。
キーは既存キーの末尾に `:packed-x-i8` を足した形で、診断・census が読む格納判別子の位置は動かない。
逐次GEMV・行ブロックGEMV・subgroup変種・GEMM は変えない。

## 検討した代替

- **最初から整数内積（`dot4I8Packed`）**（段 1b）。段 0 は「算術はタダ」と測った（down 形 0.3% / i4 g512 形 8%）ので、
  利得の大半は活性ロードだけで取れる。整数内積は `dot4I8Packed` の可搬性（M2 の Metal で native に落ちるか）に
  賭ける必要があり、しかも縮約が i32 累算になるので**現行とビット同一ではなくなる**。効くのは算術が 36% の
  lm_head 形（③）だけなので、形を絞って別段でやる。
- **消費側にも出力値表を渡して `table[|code|] | sign` で引く**。構造的にビット同一だが、決定 3 の証明で
  乗算 1 個と等しいと分かる以上、uniform 1040 バイトと追加命令のぶんだけ損。
- **出力側 SRQ 融合エピローグも packed で書く**。消費先が GEMV でない（= f32 を期待する）ので本段の範囲外。
- **活性を `array<u32>`（1 語 4 要素）で束縛する**。バイト数は 1/4 になるが**ロード本数は減らない**
  （i2 は 16 本のまま）ので、段 0 が測った律速に当たらない。

## 影響

- 新キー 7 本: `static_quantize:v1:packed-i8:wg128` と、並列GEMV × {融合なし, SRQ 融合} の packed 変種。
  WGSL スナップショットを追加する。**既存キーの生成物は 1 バイトも変わらない**。
- prefill 計画（M ≥ 32）は並列GEMVに落ちないので自然に対象外。
- IR・重み・manifest・公開 source pin・reference 経路は変更しない。既定 false で従来どおり。
- `linearGemvParallelEligible` の導入で、linear→SRQ 融合（ADR 0103）の受理判定も同じ述語を通るようになった。
  受理集合は変わらない（`PARALLEL_SHAPES` の全形が追加条件を満たす）。

## 検収

- CPU: 席の受理と受け渡し、対付けの受理条件（消費先が全て並列GEMVの活性のときだけ・
  linear 以外 / 非並列 linear / 重みスロット / graph output / M=9 / scale 0 は対象外・
  融合に飲まれた SRQ は対象外）、融合カウンタ、生成 WGSL のスナップショット、活性ロード本数
  （i2 4 / i4 2 / i8 1）、実配布 QAT E2B の 210 / 0。
- GPU: 全 256 コード × 代表 scale（QAT 配布形の実 scale を含む）で復元が現行 SRQ 出力と u32 一致
  （`-0.0` の落ちる本数まで固定）。並列GEMV × i2/i4/i8 × lane 2〜32 × M 1/4/8 × 融合あり / なし ×
  活性 3 種（素直 / int8 全域 / `±Inf`・`-0.0`）で出力が **u32 完全一致**（540 件）。
  QAT E2B の 64 token greedy id 列が席 on / off で完全一致。
- 全GPUでの浮動小数点丸めの仕様保証とはしない（ADR 0103 と同じ立場 — 実機での一致は検収結果）。

## 追記 1（2026-09-19）: 形ごとの採否と SRQ カーネルの再設計

決定 1〜4 の席をそのまま実測に掛けたら、QAT E2B decode（Chrome・pass 境界 timestamp）の
GPU 時間は **+0.25 ms/token（悪化）**だった。dispatch 分割の per-key 実測
（`outputs/bench/karume/2026-09-19_21-25-39_k45-1a-packed-aa2db539/per-key-on-off.json`）が
悪化の出どころを 2 つに割った。

| キー（packed 変種 vs 現行）                                 | 本数/token | on/off の生値比 | 補正 Δ ms/token |
| ----------------------------------------------------------- | ---------: | --------------: | --------------: |
| `linear_gemv_parallel:wi2:l32:static-quantize:v1`           |         20 |            0.62 |           −0.31 |
| `linear_gemv_parallel:wi4g2048:l32:static-quantize:v1`      |         43 |            0.81 |           −0.13 |
| `linear_gemv_parallel:wi4g4096:l32:static-quantize:v1`      |          7 |            0.77 |           −0.03 |
| `linear_gemv_parallel:wi4g512:l4:static-quantize:v1`        |         65 |            0.98 |           +0.01 |
| `linear_gemv_parallel:wi4g512:l32:static-quantize:v1`       |         30 |            1.01 |           +0.02 |
| `linear_gemv_parallel:wi8:l4:static-quantize:v1`            |         35 |            1.05 |           +0.03 |
| `linear_gemv_parallel:wi8:l32:static-quantize:v1`           |         35 |            1.05 |           +0.03 |
| `linear_gemv_parallel:wi2:l2:static-quantize:v1`            |         40 |            1.23 |           +0.25 |
| 素の SRQ（席 on = `:packed-i8:wg128` / off = `:f32:wg128`） |        210 |            1.19 |           +0.39 |

読み:

1. 効くのは **K が長く lanes 32 の形**（1 スレッドが 1 重み語あたりに読む活性が多い形）だけ。
   K=1536 の lanes 2 / 4 と i8 では、減らしたロード本数より追加の unpack / 変換 / 乗算が勝つ。
2. **packed SRQ カーネル自体が f32 版より 19% 遅い**。1 スレッドが 4 要素を直列に量子化する
   幾何ではスレッド数が要素数の 1/4 に落ち、k=1536 なら 384 スレッド = workgroup 3 個で
   占有率が足りない。

### 追記決定 1: packed SRQ カーネルの幾何を f32 経路に揃える

`STATIC_QUANTIZE_PACKED_WGSL` を**1 スレッド 1 要素**へ戻す。各スレッドが自分の要素のコードを
求めて workgroup 共有メモリ（`array<u32, 128>`）へ置き、`workgroupBarrier()` の後に下位
32 スレッドが 4 コードずつ 1 語へ詰めて書く。dispatch 数は f32 版と同じ `ceil(count / 128)`。
atomic OR は使わない（書き込みが 1 語 1 スレッドに閉じるので要らない）。

- **grid-stride の端**: タイル（128 要素）の選択を `workgroup_id` だけで回す。`count` が 128 の
  倍数でなくても barrier は workgroup 全体で一様に通る（`local_invocation_index` でループを
  回すと端のタイルで barrier が非一様になり、WGSL の一様性解析で落ちる）。範囲外の要素は
  コード 0 を共有メモリへ置くだけで、**語は書かない** — `count` は 4 の倍数（params の門）なので
  1 語は「全要素が範囲内」か「全要素が範囲外」のどちらかにしかならない。
- **コードの確定**（境界表・二分探索・level / sign）は 1 バイトも動かさない。変わったのは
  担当割りと書き出しの経路だけで、決定 3 のビット同一の根拠はそのまま。
- タイルごとに barrier を 2 回通る（2 本目は次のタイルが共有メモリを上書きする前の WAR 障壁）。

### 追記決定 2: 形ごとの採否を実測表に載せる

`PARALLEL_SHAPES`（ADR 0098 / 0022 の「実測した形だけ」の流儀）の各行に
`packedActivations: boolean` を足し、上の実測で効いた 4 行だけ true にする。

| 行                             | lanes | per-key        | 採否  |
| ------------------------------ | ----: | -------------- | ----- |
| i2 n=1536 k=12288              |    32 | `wi2:l32`      | true  |
| i4 g2048 n=1536 k=2048         |    32 | `wi4g2048:l32` | true  |
| i4 g2048 n=1536 k=6144         |    32 | `wi4g2048:l32` | true  |
| i4 g4096 n=1536 k=4096         |    32 | `wi4g4096:l32` | true  |
| 他 21 行（g32 の 12 行を含む） |     — | —              | false |

対付け（決定 2）の消費側条件はこの採否まで含む: **消費先の全てが true の行へ落ちる**
`static_quantize` だけを packed にする。判定は `linearGemvPackedEligible`
（`linearGemvParallelEligible` に行の採否を重ねた述語）1 本で、fusion.ts の対付けと
recipe-builder の門が同じ述語を読む MUST は決定 2 のまま。

false の行にも packed 変種の WGSL・params は生成できる（テストとスナップショットは全形を
維持する）が、**製品の plan では選ばれない**。

融合カウンタ `packedStaticQuantize` は QAT E2B decode 計画で **210 → 70**
（内訳 = per-key の本数そのもの: 20 + 43 + 7）。prefill 計画は 0 のまま。

### 検収（追記ぶん）

- 新カーネルが f32 経路と同じコード列を出すこと（コード巡回 × `count` = 4 / 16 / 128 / 132 /
  1536 / 12288 で u32 一致 + 範囲外の語を書かない番兵）。既存の 540 件 u32 完全一致と
  256 コード × 代表 scale 6 本はそのまま緑。
- 採否の gating（true の行だけ対付けが成立・false の行が 1 本でも混ざる SRQ は f32 のまま）。
  同じ (格納, n, k) でも group 長が違えば別の行 = 別の採否になることを i4 1536×2048 の
  g2048 / g32 の対で固定する。
- 実配布 QAT E2B の decode 70 / prefill 0。
- 故障注入（共有メモリの詰め順を 1 バイトずらす）で 3 つの GPU 門が赤になることを確認して復元。

## 追記 2（2026-09-20）: 語彙への昇格と `i4-fast` の宣言・段 1b の棄却

決定 1 の「manifest 所有の語彙には席を作らない」を解く。research §16 の A/B
（Chrome +8.3% / GPU −0.57 ms/token・Deno +1.5〜2.9%・64 token greedy の id 列一致）で
採用が決まったので、席を配布形が宣言できる語彙へ昇格する。

### 追記決定 3: hub の `SessionSpec` に `packedStaticQuantize?: boolean` を足す

- true / false だけを受理し、null・数値・文字列は拒否する。false を省略へ畳まず、`@karume/models` の
  共通写像（`WRITERS`）も同じ欄へ明示して転送する（[0104](0104-gemma-fast-quant.md) の融合 2 欄と
  同じ流儀）。manifest は `karume/4` のまま。旧 reader は未知キーとして拒否するので、
  宣言した配布形には対応する hub / models が要る。
- Gemma が quant.session から受理する欄に加える。優先順位は**明示指定 → quant 宣言 → 未指定
  （runtime 既定 false）**で、型の正しい明示 false は quant の true に勝つ。決定 1 の
  「`linearGemvReduce: "parallel"` 必須」の拒否は quant 由来でも同じ。

### 追記決定 4: QAT E2B の `i4-fast` が宣言する

- recipe（`gemma4_qat/distribution.py` の `qat_quants`）の E2B `i4-fast` に
  `packedStaticQuantize: true` を足す。E4B は `i4` のまま。
- 通常 Gemma 4 は `static_quantize` ノードを持たず対付けが 0 本なので宣言しない
  （宣言しても no-op だが、意味の無い欄を配布形に載せない）。
- 既定の数値: 決定 3 の「`-0.0` はコード 0・NaN は飽和」が QAT E2B の既定経路の挙動になる
  （ADR 0058 の数値 opt-in の範囲）。参照 golden・reference 経路・`i4` / `i4-gemvpar` は不変。
- ブラウザ計測ページ（`tools/llm-speed/browser/`）に明示 off / on の軸を足し、M2 の追試は
  そこでの往復比較で行う（linear→SRQ 融合と同じ器）。

### 段 1b（整数内積）の棄却

「検討した代替」で lm_head 形（③・算術 36%）に絞って別段とした整数内積は実装しない。

- research §14 の「③は算術が効く」は単体ハーネスの読みで、実モデルの lm_head には **int8 活性が
  存在しない**。公式 checkpoint の lm_head は SRQ の scale が入出力とも 0（未較正 = 恒等）で、
  recipe は恒等 SRQ を IR に挟まない（`packages/models/src/gemma/qat.ts` の共有 head の門・
  ADR 0097）。整数内積に要る int8 活性を作るには公式にも無い動的量子化を新設することになり、
  公式 mobile とも現行ともビット同一でない数値契約が増える。
- 上限は lm_head 1 dispatch の 0.24 ms/token（research §13.4）の算術 36% ≈ 0.09 ms/token
  （GPU 6.63 ms の 1.3%）。K-45 の目的「公式 mobile と同じ計算形」に合わず利得も小さいので、
  perf-ledger K-52 と同じ判定で棄却する。

## 追記 3（2026-09-20）: Metal での分岐と活性復元の丸め障壁

追記 2 の宣言後、利用者の M2（Chrome・apple / metal-3）で計測ページの往復を回すと、速度は
中立（off 35.3 / on 35.2 tok/s）で、**英語 prompt の id 列が off / on で 64 token 中の位置 56 から
分岐**した（[research §16.1](../research/2026-09-19-qat-speed-recon.md)）。M2 で単体の u32 一致門
（`gpu_packed_static_quantize_test.ts`）を回すと、SRQ の符号列と復元 `f32(code) × scale` は一致し、
**並列 GEMV の packed 変種だけ**が最初の組（i2・lanes 2・非融合・M=1）から不一致だった。

WGSL の差は 1 点で、f32 経路の活性はロード値（丸めが確定した値）として積和 `acc + x × d` に入るのに
対し、packed 経路は `vec4<f32>(unpack4xI8(w)) × x_scale` という**積の式**のまま積和に入る。RTX / Vulkan
では両者が u32 同一だが、Metal のコンパイラは `(c × s) × d + acc` に再結合・縮約の自由度を持ち、
そこで丸めが変わったと読む（WGSL は fusion を許し再結合を禁じるが、実装の実測事実は別 —
[0099](0099-rms-norm-add-fusion.md) の丸め障壁と同じ立場）。

### 追記決定 5: 復元した quad に丸め障壁を通す

`activationQuad` の packed 分岐を
`bitcast<vec4<f32>>(bitcast<vec4<u32>>(vec4<f32>(unpack4xI8(…)) × dims.x_scale) ^ vec4<u32>(dims.rounding_mask))`
にする（[0099](0099-rms-norm-add-fusion.md) の rms→add 融合と同じ書き方・実行時 0 との XOR）。

- SRQ 融合ありの変種は Dims に `rounding_mask`（語 3）が既にあるのでそれを読む。融合なし packed 変種は
  Dims を `m / n / k / x_scale / rounding_mask` にし、params を 8 語（uniform の 16 B 整列）へ広げる。
- XOR 0 は恒等なので RTX の数値は動かない（u32 一致門 540 件・QAT E2B の id 列一致は緑のまま）。
  変わるのは packed 変種 6 本の WGSL スナップショットと params の語 4。
- 決定 3 の「乗算 1 個は正しく丸めた積」は乗算単体の性質で、積和に inline したときの
  コンパイラの変換までは縛れない。障壁はその変換を式の外へ出さないための実装事実で、仕様保証ではない
  （M2 の再走で緑になることを検収条件にする）。

検収（M2・利用者実走）: 同じ u32 一致門 1 コマンドが緑 → 計測ページの往復で英語 prompt の id 列が
off / on で一致。赤なら次の仮説（重み側 `f32(q) × wscale` との組合せ）へ進む。
