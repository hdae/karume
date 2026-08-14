# 0056 — 語彙拡張: gru_scan / gru_scan_reverse（GRU 隠れ側スキャン・第 2 層）

- Status: accepted（2026-08-14）
- 対象: `gru_scan` / `gru_scan_reverse` op 新設（契約 56・57 op 目）+
  packages/runtime/src/kernels/gru-scan.ts + tools/exporter/karume/custom_ops.py +
  エクスポータハンドラ 1 本
- 関連: ADR [0043](0043-op-addition-layers.md)（層の定義と判定手順 — 本 ADR は第 2 層）/
  [0023](0023-fused-attention.md)（ビット同一を設計の核に据える形と、決定 4「欄の不存在が
  語彙に無いことを構造で表す」）/ [0055](0055-deform-conv2d.md)（直近の op 追加。あちらは
  A/B の相手がいない新規原子なので**退化オラクル**に逃げた — 本 op は逃げない）/
  [0046](0046-cat-symbolic-axis.md)（記号軸の緩和は `cat` だけ・`slice` / `flip` は静的専業
  据え置き）/ [0014](0014-layout-ops-full-write.md)（`flip` の静的軸）/
  [0022](0022-gemm-register-blocking.md)（実行時オートチューン禁止 = codegen 決定性）/
  [0024](0024-conv2d-implicit-gemm.md)（conv 系の bias-first と GEMM の bias-last の分岐点）
- 需要の実測: vowel-detector（`Crnn` = conv 2 段 → **2 層 BiGRU** → linear）の
  `aten.gru.input` は `run_decompositions` が**時間方向へ完全展開**する。分解ノード数は
  T10 = 20 / 40 / 200 で **874 / 1,714 / 8,434**（call_function・T に線形）で、`torch.export`
  は長さ軸を specialize する（`Specializations unexpectedly required (T)`）ため
  **長さごとに別のグラフ**が要る。本 op を通すと **T に依存しない 19 ノード**になる。
  再出現先として Kokoro-82M の双方向 LSTM 6 本があるが、**本 ADR では LSTM を入れない**
  （決定 8）。

## Context

`aten.gru.input` は Core ATen の分解表にあり、`curated_decompositions()` を通すと Python
ループで T 回展開される。展開後のグラフは**語彙内の既存 op（`linear` / `slice` / `sigmoid` /
`tanh` / `mul` / `sub` / `add` / `cat`）だけで厳密に書けている** — つまり GRU は原子ではなく
**分子**で、ADR 0043 追記 決定 1 の判定軸（「語彙内の他 op の合成で厳密同値に書けるか」）は
第 1 層／第 1' 層のどちらでもないと答える。

分子でありながら語彙へ入れる根拠は、`docs/op-vocabulary.md` の第 2 層入場条件のうち 2 つ:

- **① 分解形からの復元で意味情報が落ちる**（該当）。T 回展開されたグラフから「これは GRU
  だ」を第 3 層 matcher で復元するのは、T ごとに違うノード列を exact-match することになり
  成立しない。落ちる意味情報は「**時間軸は実行時に決まる**」という事実そのもので、これは
  ADR 0046 が「Irodori DiT が export できない」を解いたのと同型の、機能の開通にあたる。
- **③ 再出現率が恒久コミットを正当化する**（該当）。vowel-detector の BiGRU に加え、
  RNN 系列モデルの隠れ側スキャンは構造がこの 1 形しかない。

② 中間実体化の容量は部分的にしか当たらない（T=200 で 8,434 ノードぶんの transient と
dispatch は重いが、本件の主要因ではない）。④ attr 変種は非該当。

なお `aten.gru.input` は Core ATen 外だが、**層の判定軸は「Core ATen 由来かどうか」ではない**
（ADR 0043 追記 決定 1 — Core ATen 由来は第 1 層の母集団の説明であって判定軸ではない）。

## 決定

### 1. 入力側 GEMM は op に含めない — 契約は「隠れ側スキャンだけ」（案 S）

torch の分解は既に **入力側 linear（時間一括・出力 `[T,N,3H]`）+ 隠れ側 linear（毎ステップ・
出力 `[N,3H]`）** に分かれている（実測）。この切れ目をそのまま契約の境界にする:

`gru_scan` / `gru_scan_reverse`・**アリティ 4 固定**・uniform f32・**attrs 空**。

| 形     | 契約                                           |
| ------ | ---------------------------------------------- |
| `gi`   | `[T, N, 3H]`（入力側 `x·W_ihᵀ + b_ih` の結果） |
| `h0`   | `[N, H]`                                       |
| `w_hh` | `[3H, H]`                                      |
| `b_hh` | `[3H]`                                         |
| 出力   | `[T, N, H]`（全ステップの `h`）                |

ゲートの並びは **r / z / n**（torch の `weight_hh_l{k}` の並びの逐語）。

入力側を内包する案（op 名 `gru`・アリティ 6）を却下した理由は 4 軸すべてで劣るため:

- **契約の大きさ**: 案 S はアリティ 4・attrs 空で、「入力側は語彙に無い」が構造で表れる
  （ADR 0023 決定 4）。案 G は将来 `has_biases=False` を足すと**スロットの有無**が 2 軸になり、
  可変アリティか合成 bias が要る。
- **既存 linear の最適化が効く**: 案 S では入力側が素の `linear` ノードなので
  `WEIGHT_SLOTS` に載り、f16 / i8 格納も `linearCompute: "i8a8"` もそのまま効く。
  vowel-detector の `w_ih` は 2 層双方向で `[384,160]`×2 + `[384,256]`×2 = 重みの過半。
  案 G ではそれが op 内スロットになり適格から外れる。
- **ビット同一の証明対象が半分になる**: 入力側 linear が分解形と同一ノードなのでビットは
  自明に一致し、証明が要るのは隠れ側の逐次だけ（決定 3）。
- **LSTM への拡張が素直**: ゲート数 3 → 4 と cell state の追加は「op 内部の式」だけの差で、
  入力側は LSTM でも同じ `linear`（`[T,N,4H]`）。

**op 名が `gru` ではなく `gru_scan` になる**（IR ファイルに焼かれる恒久の公開コミットメント）
のはこの帰結で、「これは GRU そのものではなく GRU の隠れ側スキャン」という意味が名前に出る。

- **`w_hh` は低精度格納の適格外**（`WEIGHT_SLOTS` / `WEIGHT_CHANNEL_AXES` に載せない）。
  op 内スロットの i8 は scale 軸の取り違えが沈黙誤値になる（ADR 0024 決定 6 が記録した
  バグクラス）ので、需要が出るまで足さない — 入力側の重みは呼び手の `linear` が持つので、
  配布サイズの主成分はそちらで縮む。

### 2. 逆方向は attr ではなく**別 op 名**（`gru_scan_reverse`）

双方向 GRU の逆方向を IR で表す形は 3 つあり、採ったのは 3 番目:

- **`flip` を挟む形は成立しない**。torch の分解は `linear(x)` の結果を `flip(·,[0])` してから
  逐次し、出力は `cat` のリスト順を逆にする（出力側 `flip` は無い）。しかし karume の `flip` は
  **記号軸を両側で拒否する**（Python: `shapes.py` の `_flip` が `_static_axis` /
  TS: `plan.ts` の `assertStaticLayoutAxis`）。ADR 0046 が「`slice` / `flip` は静的専業のまま
  据え置く」と明示的に裁定しており、それを覆すのは独立の ADR 1 本ぶんの判断になる。しかも
  `flip` は `[T,N,3H]` の実体化コピーを 1 本増やす（T=200・H=128 で 307KB × 4 本）。
  → **走査方向は op が知る**しかない。
- **attrs の bool（`reverse: true`）は採らない**。karume の既存 attrs は int / int リスト /
  f32 スカラ / 文字列だけで、**bool を載せた前例が無い**。それどころか Python 側の
  `_assert_integer_attr` は「Python の bool は int の派生で `True` が 1 として素通りする」と
  明記して bool を**明示的に弾いて**おり、受け入れるには両側に bool 用の検証関数を新設する
  ことになる（= 契約検査機構の拡張）。
- **採用: `gelu` / `gelu_tanh` と同じ「attr 変種は別 op」の手筋**（ADR 0043 の初適用例）。
  恒久コミットが 1 本増える代わりに、契約検査機構は 1 行も増えない。

**MUST: 逆方向 op も出力は順方向の時間順で書く**（走査順だけが逆）。出力側に `flip` を
要求する形にすると、`flip` が記号軸を拒否するという上の事実に正面から戻る。

### 3. ビット同一を**恒久の門**にする（分解形との Uint32 完全一致）

ADR 0055（deform_conv2d）は新規原子で A/B の相手がいないため退化オラクル（offset 全 0・
mask 全 1 → conv2d）という弱い形に逃げた。**GRU は逃げる必要がない** — T を固定すれば
torch の分解グラフがそのまま完全な参照実装になるので、ADR 0023 決定 2 と同じ強い形が採れる。

門: `packages/runtime/tests/gpu_gru_scan_parity_test.ts`。同一入力に対し ①既存語彙で組んだ
分解グラフ（`linear → slice → add → sigmoid` / `mul → add → tanh` / `sub → mul → add` を
T 回 + `cat`）と ②`gru_scan` 1 ノードを**同じ実 GPU**で流し、出力の **f32 ビット列**が
完全一致することを assert する。

成立根拠は 4 点。どれか 1 つでも崩すと丸め列が変わる:

**(a) 隠れ側縮約が `linear` の GEMM と同一順** — 縮約は `k` 昇順の逐次で、字面も
`acc = acc + w * h`（`gemm.ts` の `accumulatorUpdate` と同じ形）。GEMM は K タイル 16 で
分割するが 1 出力要素あたりの加算順は k 昇順のままで、端タイルのゼロ詰めは `acc + a·0` が
f32 で厳密恒等。**実測で成立を確認**（H = 19 = タイル 16 の倍数でないケースを含む）。

**(b) bias は last** — 分解経路の隠れ側は `linear` = GEMM = **store で最後に足す**形なので、
カーネルも `(Σ W·h) + b` と書く。conv 系の bias-first（ADR 0024 の MUST ①）を流用すると
`bias + Σ` の順になり丸めの並びが変わる（`gemm.ts` が「本設計で最大の分岐点」と記録している
のと同じ罠）。

**(c) `sigmoid` の WGSL 本体を素の elementwise と共有する** — `SIGMOID_STABLE_WGSL` を
`codegen/elementwise.ts` から import する（silu.ts と同じ規律。書き写すと primitive と
融合版で丸め列が割れうる）。`tanh` は WGSL 組込なので共有の問題は無い。

**(d) 演算の並びと括り方を分解形の逐語にする** — 更新式は **`h' = (h − n)·z + n`**。
数学的に同値な `(1 − z)·n + z·h` は**別の丸め列**で、f32 の 10 万要素中 **44,345 件が
ビット不一致**（maxdiff 4.77e-07）という実測がある。ゲートの足し順は隠れ側が第 1 引数
（`gh + gi`）、`n` の積は `gh_n · r`（**reset ゲートは b_hh 込みの隠れ側積に掛かる** —
bias を reset の外へ出す誤り形は maxdiff 0.196 で外れる）、`n` の和は入力側が第 1 引数
（`gi_n + (gh_n·r)`）。有限 f32 では加算・乗算は可換だがこの列が丸め列の正本で、
NaN payload はバックエンド差がありうる（`silu.ts` が `SiluMulOrder` をキーに残しているのと
同型の理由）。

### 4. 丸め障壁（workgroup memory 往復）は**この op でも必要**だった — 実測で確定

`a * b + c` を 1 式で書くと **fma へ縮約される**。本ワークツリーの実 GPU で測った結果:

| 形                                             | mul / add 2 dispatch とのビット不一致         |
| ---------------------------------------------- | --------------------------------------------- |
| `o[i] = a[i] * b[i] + c[i]`（1 式）            | **15,371 / 65,536**                           |
| `let p = a[i] * b[i]; o[i] = p + c[i];`        | **15,371 / 65,536**（名前を付けても同じ）     |
| `bitcast<f32>(bitcast<u32>(a[i]*b[i])) + c`    | **15,371 / 65,536**（bitcast では止まらない） |
| workgroup memory へ書いて barrier 後に読み戻す | **0 / 65,536**                                |

分解経路は `mul` と `add` が**別 dispatch** = storage 往復なので縮約されない。したがって
**積を workgroup memory へ書いて barrier 後に読み戻す**（silu.ts の技法）で明示的な
materialization 点を残す。必要な箇所は 2 つ:

1. `n = tanh(gi_n + gh_n·r)` の積 `gh_n · r`
2. `h' = (h − n)·z + n` の積 `(h − n)·z`

**どちらも門にとって load-bearing であることを故障注入で確認した**（barrier を外して
融合式へ書き換えると、①は T4/N1/H128 で 205/512 要素・②は 294/512 要素がビット不一致に
なり門が赤くなる）。

**この丸め障壁は WGSL 仕様の保証ではない** — 仕様は fusion を許すだけで、workgroup memory
往復を最適化障壁として尊重することは要求していない。有限値のビット一致は**バックエンドごとの
実 GPU A/B で採用門にした実測事実**であり、`silu.ts` / RoPE 融合と同じ扱い（バックエンド更新で
門が割れたら、まずここを疑う）。Metal では conv2d parity が既に割れた実績がある
（[known-issues.md](../known-issues.md)）ので、この門も参照環境（Vulkan）で成立した事実として
読む。

### 5. カーネルは「1 workgroup = 1 バッチ要素 / 1 lane = 1 隠れユニット」— `H ≤ 256`

- **workgroup サイズは `256` のコンパイル時定数**。device limits を読んで幾何を変える
  カーネルは karume に 1 本も無く、読むと「同一キー → バイト単位同一 WGSL」（codegen 決定性・
  ADR 0022 が実行時オートチューンを禁止した帰結）が崩れる。
- バッチ要素どうしの再帰は独立なので **1 workgroup = 1 バッチ要素**。バッチ方向は
  layer-norm と同じ grid-stride で dispatch 上限（既定 65,535）を跨ぐ。workgroup 間の同期は不要。
- **1 lane = 1 隠れユニット**に固定し、`GRU_SCAN_MAX_HIDDEN = 256`（= workgroup サイズ）を
  超える形は `gruScanParams` が `CodegenError` で落とす（黙って縮退させない — 縮退させると
  `h_shared` の範囲外書き込みで別ユニットの状態が例外なしに壊れる）。
  実測に出ている形は **H = 128**（vowel-detector）だけで、上限を上げるには workgroup 内
  grid-stride と `h` の二重化（読み終わる前に書けないため）が要る = 別の設計判断。
  「実測に出た形だけ」（ADR 0023 決定 4）に従い、需要が出てから広げる。
- workgroup 共有は **`h` の H 要素**と**丸め障壁の中継 256 語**だけ（`W_hh` は `[3H,H]` =
  H=128 で 196KB なので仕様既定の 16,384 B に載らず、毎ステップ storage から読む）。
  時間ループと barrier 群は**workgroup 一様な制御流**の中だけに置く（`dims` は uniform —
  layer-norm.ts の MUST と同じ。storage からのロードは一様性解析で保証されない）。

### 6. エクスポータは `torch.library.custom_op` で記号 T を保つ

`karume::gru_scan` / `karume::gru_scan_reverse` を `torch.library.custom_op` +
`register_fake` で定義する（`tools/exporter/karume/custom_ops.py`）。本体が fake の裏へ
隠れるのでトレースが Python ループへ入らず、**時間軸 T は記号のまま単一ノードで残る**。
`karume::` は Core ATen の分解表に無いので `run_decompositions(curated_decompositions())` を
通しても展開されない（`torchvision::deform_conv2d` と同じ理由）。実測で確認済み:
`karume.gru_scan.default (s77, 1, 5)`。

- **eager 実装は torch の GRU 分解の逐語**にする。golden の期待値はこの実装を eager で回して
  採るので、ここがずれると「エクスポータとランタイムが一致して両方間違っている」状態が緑に
  なる。単方向 1 層の `nn.GRU` と **`torch.equal` でビット完全一致**することを実測で確認した
  （順方向・逆方向とも maxdiff 0.0）。
- ハンドラのキー（`torch.ops.karume.gru_scan.default`）を書くには登録済みである必要があるので、
  `convert.py` が `karume.custom_ops` を先頭で import する。**この import はプロセス全域の
  グローバル副作用**だが、CLAUDE.md の「全モジュール副作用ゼロ」は `packages/*`（TS 側の
  tree-shaking 成立条件）の不変条件で、Python 側 exporter には掛からない（`convert.py` が
  `torchvision` を同じ理由で素に import しているのと同型 — ADR 0055 決定 7）。
- **多層・双方向は exporter 側の正規化でノードを並べて表す**（決定 7 の帰結）。`aten.gru` の
  `Tensor[16]`（16 本の重み）は IR に載らないので、層と方向ごとに 1 ノードずつ発行し、層間の
  結合は既存 `cat`（dim 2・静的軸）で行う。本 ADR はこの op の契約だけを定義し、
  **`nn.GRU` を割るパッチ層はモデル移植側の仕事**として扱う（範囲外）。

### 7. 契約から**欄を落とす**もの（欄の不存在が「語彙に無い」を表す）

| 軸                   | 扱い                                                                                                                                                                 |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 多層（`num_layers`） | 欄を作らない。層ごとに 1 ノード（`num_layers` を持つと op が層ループと層間 `cat` を内包する）                                                                        |
| 双方向               | 欄を作らない。方向ごとに別 op 名（決定 2）                                                                                                                           |
| `has_biases=False`   | 欄を作らない = **bias 必須のアリティ 4 固定**。既存 `linear` の「bias は常時あり」と同じ手筋                                                                         |
| `batch_first`        | 欄を作らない。IR 上の `gi` は `[T,N,3H]`（time-first）固定 — torch の分解自身が `permute` で正規化している                                                           |
| `dropout` / `train`  | 欄を作らない（推論専業 — ADR 0023 決定 4 の dropout と同じ）                                                                                                         |
| `h0` の省略          | 欄を作らない = **入力スロットで必須**。ゼロ `h0` は `aten.full` が第 0 層の定数畳み込みで initializer にする（実測）                                                 |
| 出力 `h_n`           | **返さない**。IR v1 は実質単一出力（`convert.py` は常に `outs=[out_name]` を書き、TS 側 `plan.ts` も `node.outs[0]` しか読まない）。vowel-detector は `h_n` を捨てる |

### 8. LSTM は入れない（対称性のための追加をしない）

`docs/op-vocabulary.md` の「やらない方がよいこと」と ADR 0043 Consequences が優先する。
将来 `lstm_scan` を足すときの棚卸しを記録しておく:

- **そのまま再利用できる**: 入力側 linear が時間一括という構造 / 時間ループ + barrier の
  WGSL 骨格 / 走査方向を op 名で分ける形 / ビット同一門の作り方（T 固定の分解形との Uint32
  突合）/ 決定 3 の (b)(c)(d) と決定 4 の丸め障壁。
- **別物になる**: cell state `c` が増えるので workgroup 共有が 2 本になり、決定 5 の
  `GRU_SCAN_MAX_HIDDEN` に相当する上限が半分になる。入力スロットも `c0` が増えてアリティ 5。
- **正面衝突する論点**: LSTM の `(y, h_n, c_n)` は 3 出力で、**決定 7 の単一出力制約と
  正面から衝突する**。`y` だけで足りるモデルなら本 ADR の形がそのまま写せるが、`h_n` / `c_n` を
  消費するモデルは別の設計（IR v1 の多出力化、または状態返し版の別 op）が要る。
  Kokoro-82M の双方向 LSTM が `h_n` / `c_n` を消費するかは**未検証**なので、LSTM に着手する
  前にそこを確かめること（契約の形が手戻りする唯一の分岐）。
- ゲート数を attrs にして GRU と兼ねる形は採れない（`gelu` / `gelu_tanh` と同じ attr 変種問題 —
  決定 2 と同じ理由で別 op が正）。

## Consequences

- 契約 1 セット（TS `OP_CONTRACTS` + `karume/ops.py` の `OP_CONTRACTS` + `shapes.py` の
  shape 規則 + `fixtures/op-contracts.json` + CPU 参照 + golden `gru_scan_block`）+
  カーネル 1 本（2 方向）+ エクスポータハンドラ 1 本 + custom op 定義。
  **既存 IR・既存資産への影響はゼロ**（新しい op 名が 2 つ増えるだけ）。
- 契約表は 55 → **57 op**。固定アリティ 4 は `deform_conv2d` の 5 に次ぐ多さ。
- **`gru_scan` は attrs 空の第 2 層 op** で、走査方向という唯一の自由度を op 名が持つ。
  この形が increment されるたびに op 名が増えることは承知の上の対価（決定 2）。
- 性能は**一切測っていない**（ユーザー方針が「正しい結果が目的・速度は後回し」）。
  `W_hh` が毎ステップ storage 読みになる帯域影響、1 workgroup = 1 バッチ要素で N=1 のとき
  GPU が 1 workgroup しか使わない占有率、H=128 で 256 lane の半分が遊ぶ点はいずれも
  性能候補であって本 ADR の範囲ではない（要求が出たら [perf-ledger](../perf-ledger.md) 起票）。
- **vowel-detector の移植そのものは本 ADR の範囲外**。長さ軸を記号にするには先頭
  `Conv1d(stride=2)` のせいで `2*Dim("T")` の派生次元宣言が要る（素の `Dim("T")` だと出力
  extent が `((T−1)//2)+1` の床除算になり karume の次元言語 `coeff·sym+offset` に載らない —
  実測）。現行の `LENGTH_MULTIPLE = 2` 制約と要求が一致するので実運用の制約は増えないが、
  「グラフ入力の長さは常に偶数」がランタイム側の契約になる点は移植時に裁定が要る
  （**右ゼロ pad による回避は禁じ手** — 逆方向 GRU が pad から状態を持ち帰る）。

## 追記（2026-08-14）— vowel-detector の移植で着地

Consequences 末尾の「移植時に裁定が要る」2 点はどちらも決着した。

- **長さ軸は `2*Dim("T")`**（記号は出力の 20ms 格子側）。派生次元だけを持つ入力からシンボルを
  束縛できるようにする IR v1 の緩和が要り、**ADR [0057](0057-derived-dim-binding.md)** で
  裁定した（`solveDim` は一意・割り切れない実寸は fail loudly）。
- **「10ms フレーム数は偶数」はランタイム契約**になった。奇数フレームは**末尾 1 本を切り捨て**
  （既存の実重み E2E とホスト後処理が元から採っていた規約）で、右ゼロ pad は入れていない。
- 差し替え層は `karume/patch_vowel_detector.py`（`nn.GRU` → 単方向 1 層 × 4 + `cat` 2 本）。
  **`nn.GRU` とビット一致**することは pytest（5 長 × 多層 × N>1）と、実重みの emit ごとの
  常設門（期待値は参照経路から採り、差し替え経路と `torch.equal`）と `--verify`（5 長）で
  三重に踏む。
- 実測: IR ノード **8,434（T10=200 の展開列）→ 18（T 非依存）**・`model.safetensors`
  3,892,256 → 2,668,608 B・export 8.1 → 0.6 s。**実音声 4 本の `.lab` は展開列経路と
  ビット完全一致**（f32 全要素・4 本とも差 0）なので、実重み E2E の期待 `.lab` は 1 行も
  変えていない。配布形は長さバケット 4 本 34,088,454 B → 1 本 2,759,461 B（−91.9%）。
- 副産物: バケット + 右ゼロ pad をやめたことで、**配布形経路と実長経路の `.lab` の差が消えた**
  （実測: 旧経路は 4 本中 3 本で割れていた — 20ms の境界ずれ・末尾 40ms の `pau`・発話中間の
  40ms `pau`）。一致は `packages/models/tests/e2e_vowel_detector_lab_test.ts` が毎回踏む。
