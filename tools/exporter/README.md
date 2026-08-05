# Karume exporter

`torch.export` 済みモデルを Karume の **IR v1**（[../../docs/ir-v1.md](../../docs/ir-v1.md)）へ
落とす Python ツール。uv 管理・CPU 版 torch のみ（GPU 不要）。

配布形は safetensors 1 ファイル — テンソル（重み・定数）と、`__metadata__` の
キー `karume_ir` にグラフ JSON を持つ。

## セットアップ

```sh
uv sync            # tools/exporter/ で実行（CPU 版 torch を pytorch-cpu index から取る）
```

## 検証コマンド（変更後は全て）

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## golden fixtures の再生成

```sh
uv run python -m karume.goldens          # 既定の出力先: ../../packages/runtime/tests/fixtures/golden/
uv run python -m karume.goldens --out /tmp/golden   # 出力先を変える
```

固定 seed（`karume.goldens.SEED`）なので、同じ環境なら**バイト単位で同一**の
ファイルが出る（`tests/test_goldens.py::TestDeterminism` が再生成と突合する）。
契約表の全 op（EMITTABLE_OPS）を全モデル合計で被覆していない場合は生成が失敗する —
op を足したら golden も足す、が実装契約（ADR 0005）。

### golden レイアウト

```
packages/runtime/tests/fixtures/golden/<model>/model.safetensors   重み・定数 + __metadata__.karume_ir
packages/runtime/tests/fixtures/golden/<model>/io.safetensors      入力テンソルと torch CPU での期待出力
```

`io.safetensors` のテンソルキー命名規約:

| キー           | 内容                                                    |
| -------------- | ------------------------------------------------------- |
| `input.<name>` | `<name>` は **グラフ入力名**（`graph.inputs[].name`）。 |
| `output.<i>`   | `<i>` は **`graph.outputs` の位置**（0 始まり）。       |

記号次元の束縛は別に持たない — 入力 shape の次元位置から取る（IR v1 の束縛規則と同じ。
`"T"` のように係数 1・オフセット 0 で現れる次元の実長がその束縛値）。現在の golden は
記号次元をすべて `GOLDEN_T` で焼いている。

`io.safetensors` の格納 dtype は**意味論 dtype の実表現**（ADR 0009 の境界正規化）:
f32 → `F32` / i32（torch の i64 を含む）→ `I32` / bool → `U32` の 0 / 1。値域外の i64 は
fail loudly（`convert.normalize_boundary_tensor`）。

現在のモデルと被覆:

| モデル             | 記号次元 | 踏む IR op                                                                                                 |
| ------------------ | -------- | ---------------------------------------------------------------------------------------------------------- |
| `unary_chain`      | なし     | neg, abs, sqrt, log, exp                                                                                   |
| `activations`      | なし     | tanh, sigmoid, relu, gelu                                                                                  |
| `broadcast_binary` | `T`      | add, sub, mul, div（右詰め broadcast + lifted 定数）                                                       |
| `mlp`              | `T`      | matmul, add, relu（重み initializer 経由の rank-2 MLP）                                                    |
| `row_reduce`       | `T`      | sum, amax, amin                                                                                            |
| `mask_chain`       | `T`      | mul(i32), cast, bitwise_not（bool 出力あり）                                                               |
| `int_cast`         | `T`      | cast(f32→i32 切り捨て), sub(i32), mul(i32)（i32 出力）                                                     |
| `layout_chain`     | `T`      | permute(3 巡回), reshape ×3（別名の連鎖 + 係数次元 4T）                                                    |
| `expand_mask`      | `T`      | expand(bool / i32), cast, mul, bitwise_not                                                                 |
| `batch_matmul`     | `T`      | bmm(B/M/K/N 全て別長), permute（rank-3 バッチ matmul）                                                     |
| `gather_last_dim`  | `T`      | gather(最終次元 / 添字は i32 入力), sum                                                                    |
| `attention_block`  | `T`      | linear ×4, softmax ×2, layer_norm, bmm, permute, reshape, add                                              |
| `fused_attention`  | なし     | attention（SDPA 保存の 1 ノード。B/H/M/N/D 全て別長・最終 query 行は logit −190 級）                       |
| `embedding_lookup` | `T`      | embedding(padding_idx=0 は forward 不活性), sum                                                            |
| `masked_scores`    | `T`      | masked_fill(−3.4e38 broadcast / 0 同形), softmax, cast, bitwise_not                                        |
| `conv_block`       | `T`      | conv1d(kernel 3 / stride 1 / padding 1), permute                                                           |
| `dilated_conv`     | `T`      | conv1d(depthwise g=C・dilation 1/3/9 / 中間 groups / 残差), leaky_relu, add                                |
| `conv_transpose`   | `T`      | conv_transpose1d(up 2 と up 8 / 非対称チャネル), conv1d(bias 無し), tanh                                   |
| `symbolic_table`   | `T`      | sym_prefix_slice(i32 2 軸 / f32 1 軸), gather, add（Tmax 畳み込み）                                        |
| `scalar_operands`  | `T`      | add, sub, mul, div, cast（スカラ昇格 + 逆順 `1 − mask` を重みに使う）                                      |
| `spline_pieces`    | `T`      | ge_scalar, le_scalar, gt_scalar, ge, bitwise_and, cumsum, sum(bool→i32), clamp, exp, log1p, where, reshape |
| `coupling_split`   | `T`      | slice(split 分解 + pad 後の切り出し), cat, flip(軸長 3), pad, tanh, mul                                    |
| `decoder_tail`     | `T`      | leaky_relu(slope 0.1 と既定 0.01), expand(f32), conv1d, tanh, mul                                          |
| `i8_weights`       | `T`      | **i8 格納**で linear, conv1d, conv2d, conv_transpose1d, embedding（`WEIGHT_SLOTS` 全 5 op）, tanh          |

`attention_block` の 2 本目の出力は **大きい負値（−205..−180）の softmax** で、素朴形
（amax を引かない softmax）なら f32 で `exp` が 0 に潰れて `0/0 = NaN` になる領域。
safe-softmax が外れた瞬間にこの golden が赤くなる。`masked_scores` の mask は 1 行を
**全マスク**にしてあり、全要素が −3.4e38 になった行の softmax（一様分布）まで踏む。

`dilated_conv` / `conv_transpose` は ADR 0015 の conv 族拡張の被覆。前者は depthwise
（`groups = C`）・中間 groups（`1 < g < C`）・dilation 1/3/9 を 1 本の経路で踏み、後者は
**チャネル数を全段で非対称**（5 → 3 → 2 → 1）にしてある — `conv_transpose1d` の重み
`[Cin, Cout, K]` を `[Cout, Cin, K]` と読む取り違えは Cin == Cout だと要素数が合ってしまい、
shape 検査も golden も素通りするため（recon §4）。最終段の `Conv1d(..., bias=False)` は
**エクスポータのゼロ bias 合成**（アリティ 3 への正規化）を貫通させる唯一の golden。

`spline_pieces` は波 3 の数理 op 群を **sdp の spline と同じ並び**（区間内判定 → 区間境界の
cumsum → searchsorted-free の `sum(x[…,None] >= bl)` = bool の行 sum → 区間外復帰の where と
softplus の log1p）で 1 本に通す。入力は ±TAIL と clamp の両端を**跨ぐ**列でなければならない
（全要素が片側に寄ると where の分岐が片方しか踏まれない）。`coupling_split` は
split → 片側だけ変換 → cat → flip の順序が値に出る形で、**チャネル数 6**（flip の軸長 3 —
2ch の反転は off-by-one が対称に消える）。`decoder_tail` は leaky_relu の **slope 2 種**
（0.1 と位置引数省略の torch 既定 0.01）を 1 グラフに混ぜ、attrs に slope を持たない設計だと
片方が黙って誤ることを golden で塞ぐ。

`symbolic_table` は **T（= 6）< Tmax（= 24）** で焼くのが要点 — 読み出し stride を
束縛後の shape から組む誤りは T = Tmax でしか一致しないので、実長の短い golden だけが
検出器になる。`scalar_operands` の `1 − mask` は非可換で、定数を右に置く誤り（`mask − 1` に
なる符号反転）がここで値に出る — そのためには**結果を値として下流に流す**必要がある
（mask ∈ {0,1} では `(1 − mask) · mask` が逆順でも恒等 0 で、恒真な期待値になる）。生成側は
「2 要素以上あるのに全要素が同じ値」の出力を書き出す前に落とす（`_assert_not_trivial`）。

`i8_weights` は**唯一の圧縮格納 golden**（`GoldenSpec.weight_dtype = "i8"` — 生成時に
`fake_quant_int8` が export と期待値採取の**両方より前**に掛かる）。要点は 2 つ:

- **重みの行長も総要素数も 4 の倍数にしない**（`[7,5]` / `[3,5]` / `[5,3,3]` / `[5,2,3]` /
  `[3,2,3,1]`）。i8 は 4 要素を 1 u32 へ詰めるので、「語とレーンを行内相対添字から割り出す」
  誤りは**行長が 4 の倍数のときだけ偶然一致する**（f16 の偶奇と同型の罠 — ADR 0019）。
- **embedding の 1 行を全ゼロ**にする（`amax == 0` のチャネル）。scale の下限 clamp が外れると
  `0/0 = NaN` になり、この 1 行だけが golden を赤くする。

`conv_transpose1d` を混ぜてあるのは per-channel 軸が **1**（`[Cin,Cout,K]` の転置レイアウト）の
唯一の op だから — 軸表の取り違えは他 4 op では値に出ない。

## 実重み DeBERTa の export と E2E（M1-P2 波 5）

tiny golden が「op 契約の被覆」を受け持つのに対し、`export_deberta.py` は**実重み・実トークン
列での数値一致**を受け持つ。対象は SBV2 text front が使う BERT そのもの
（HF `ku-nlp/deberta-v2-large-japanese-char-wwm`）。

```sh
# 1. 生成（重み・トークナイザの取得は transformers が HF から行う。初回のみ約 1.3GB のDL）
cd tools/exporter
uv run --with 'transformers==5.14.1' python export_deberta.py            # 2 層 + 24 層
uv run --with 'transformers==5.14.1' python export_deberta.py --layers 2 # 2 層だけ（開発用）

# 1b. i8 系列（ADR 0019 の格納 + ADR 0025 の w8a8 鏡像 golden）
uv run --with 'transformers==5.14.1' python export_deberta.py --dtype i8 --act-quant

# 2. 実 GPU 突合（資産が無ければ全ケース SKIP）
cd ../.. && deno test -A packages/runtime/tests/e2e_deberta_test.ts packages/runtime/tests/e2e_deberta_w8a8_test.ts
```

- **transformers は 5.14.1 でピン**する（recon §6-5 — モデリングコードが変わるとグラフ形が
  変わる）。`pyproject.toml` / `uv.lock` には入れず `--with` で一時的に足す。
- 生成物は **`models/deberta/<variant>/`**（リポジトリ直下 `.gitignore` の `models/` で
  コミット対象外 — 24 層の重みは 1.3GB）。`--dtype i8` は**別系列**
  `models/deberta-i8/<variant>/`（24 層 319MB = f32 比 25.4%）。

```
models/deberta/dev-2layer/model.safetensors      2 層（130 ノード / 208MB）
models/deberta/dev-2layer/io.<case>.safetensors
models/deberta/full-24layer/model.safetensors    24 層（1230 ノード / 1.32GB / 出力 25 本）
models/deberta/full-24layer/io.<case>.safetensors

models/deberta-i8/full-24layer/model.safetensors      24 層 i8 格納（319MB）
models/deberta-i8/full-24layer/io.<case>.safetensors       w8 の golden（活性は f32）
models/deberta-i8/full-24layer/io-i8a8.<case>.safetensors  w8a8 の鏡像（--act-quant）
```

io のテンソルキー命名は tiny golden と同じ（`input.<グラフ入力名>` / `output.<位置>`）。
1 モデルに対して io が**ケースごとに複数**ある点だけが違う。ケースは 4 つ:

| ケース   | 内容                             | T  |
| -------- | -------------------------------- | -- |
| `case0`  | 短文                             | 11 |
| `case1`  | 長め                             | 26 |
| `case2`  | 記号混じりの長文                 | 35 |
| `padded` | `case0` + `[PAD]`×5（mask に 0） | 16 |

ラッパは `forward(input_ids, attention_mask) -> hidden_states`（全層のタプル）。層ごとに
突合できるので、**誤差が層数でどう伸びるか**が golden から直接読める（tolerance の根拠が
実測になる）。`padded` は `attention_mask=0` を混ぜてマスク経路（mul → cast →
bitwise_not → masked_fill、および conv 経路の 0 埋め）を踏む唯一のケース。

Deno 側は `packages/runtime/tests/e2e_deberta_test.ts`（1 ケース = 1 テスト）。**資産が 1 件も無ければ全 SKIP**
（生成コマンドを警告に出す）で、ADR 0005 の「全 SKIP は明示 FAIL」門番
（`packages/runtime/tests/gpu_gate_test.ts`）とは独立 — 門番は GPU アダプタの有無だけを見ており、資産が
無くても tiny golden の実 GPU テストは走る。一方、資産が**中途半端に**ある場合
（片方の variant だけ / ケース欠け）は SKIP ではなく FAIL にする。tolerance は 24 層の誤差
蓄積に合わせた専用値で、tiny golden の `GOLDEN_TOLERANCE` とは別（導出根拠は
`packages/runtime/tests/e2e_deberta_test.ts` の `DEBERTA_TOLERANCE` のコメントが正本）。**f32 / i8 の 2 系列を
同じ構造で回し、tolerance だけ系列ごとに実測導出する**（系列間の流用禁止）。

`--act-quant` が書く `io-i8a8.<case>` は **w8a8**（`linearCompute: "i8a8"`）の鏡像で、
`packages/runtime/tests/e2e_deberta_w8a8_test.ts` が使う。通常の `io.<case>` は**フックなし**で採る MUST
（掛けたまま採ると w8 側 E2E の期待値が活性量子化ごと汚染される）。prefix を `io.` から
分けてあるのは、Deno 側の通常ケース列挙（`io.` の startsWith）が鏡像を拾わないため。
w8a8 の E2E は**数値パリティの網ではない**（活性量子化が不連続なので数層で GPU と torch が
「同じ分布の別標本」になる）— 検出力の設計は同テスト冒頭のコメントが正本。

## 実重み SBV2 の export と E2E（M1-P3 波 1: dp / 波 6: front / 波 7: flow・dec・voice）

音響チェーン側の実重み。ADR [0013](../../docs/decisions/0013-sbv2-chain-export.md) の emit
ターゲット 5 本が**全て揃っている**（`voice` の E2E が緑 = SBV2 全チェーン成立）:

| ターゲット | 中身                                      | sym_max | 備考                                                |
| ---------- | ----------------------------------------- | ------- | --------------------------------------------------- |
| `dp`       | DurationPredictor 単体                    | 512 (P) | 波 1。パッチ層も語彙拡張も不要（貫通の足場）        |
| `front`    | enc_p + dp + sdp(reverse) の融合 1 グラフ | 512 (P) | 波 6。**パッチ層必須**（下の「パッチ層」を参照）    |
| `flow`     | TransformerCouplingBlock の reverse       | 4096(T) | 波 7。相対位置表を**グラフ入力**へ昇格              |
| `dec`      | HiFi-GAN Generator                        | 4096(T) | 波 7。パッチ不要・`remove_weight_norm` だけが前処理 |
| `voice`    | flow + dec の融合 1 グラフ                | 4096(T) | 波 7。**これが通ると全チェーンが揃う**              |

### 依存

```sh
cd tools/exporter
uv sync --group sbv2   # style-bert-vits2==2.5.0 / huggingface-hub
```

- **`style-bert-vits2` は `==` でピン**する。後続の波のエクスポータはパッケージ内部
  （モデリングコードのクラス属性）を monkeypatch する前提で組むので、マイナー更新で
  差し替え先の名前や forward の形が変わるとグラフ形ごと黙って変わる。理由は
  `pyproject.toml` の当該行にもコメントで残してある。
- 既定の `uv sync` には**入れない**（base deps だけで tiny golden と pytest が回る状態を
  保つ）。`transformers==5.14.1` はこのグループの推移的依存として入る。
- **ビルド依存に注意**: 推移的依存の `pyopenjtalk-dict` は wheel が無く sdist から
  ビルドされるため、**C / C++ コンパイラと cmake / make が要る**。無い環境では
  `uv sync --group sbv2` が cmake のエラーで落ちる（dp の export 自体はこのパッケージを
  一切使わないが、依存解決は素通りできない）。

### 重みの入手

**実重みはリポジトリに含まれない。ローカルに持っている資産を `models/sbv2/` へ手で配置する
（配布元・取得手順は未特定 — 現時点でコードベースからも特定できていない）。**

```
models/sbv2/config.json                        HyperParameters（`version` が JP-Extra 判定に効く）
models/sbv2/<任意名>.safetensors               ckpt（このディレクトリ直下に **1 本だけ**）
models/sbv2/style_vectors.npy                  スタイルベクトル（front の style_vec に使う）
```

ckpt は `models/sbv2/*.safetensors` の**一意存在**を要求する（複数あると「どれを読んだか」が
黙って変わる）。生成物は 1 段下の `models/sbv2/<target>/` に置くので、この glob には掛からない。

### 生成と突合

```sh
# 1. 生成（重み 251MB 級のロード込みで 5 本合計およそ 40 秒）
cd tools/exporter
uv run --group sbv2 python export_sbv2.py                # 全ターゲット
uv run --group sbv2 python export_sbv2.py --target front # 1 本だけ
uv run --group sbv2 python export_sbv2.py --dtype f16    # f16 系列 → models/sbv2-f16/
uv run --group sbv2 python export_sbv2.py --dtype i8     # i8 系列  → models/sbv2-i8/

# 2. 参照実装との eager 同値検証（**1 プロセス 1 ターゲット**。下の「パッチ層」を参照）
uv run --group sbv2 python export_sbv2.py --verify front
uv run --group sbv2 python export_sbv2.py --verify flow
uv run --group sbv2 python export_sbv2.py --verify dec    # remove_weight_norm 前後
uv run --group sbv2 python export_sbv2.py --verify voice

# 3. 実 GPU 突合（資産が無ければ全ケース SKIP）
cd ../.. && deno test -A packages/runtime/tests/e2e_sbv2_test.ts packages/runtime/tests/sbv2_relattn_parity_test.ts
```

```
models/sbv2/dp/model.safetensors       IR   17 ノード /  12 initializer /   1.78MB
models/sbv2/front/model.safetensors    IR  911 ノード / 263 initializer /  33.4MB（うち焼き込み表 2.1MB）
models/sbv2/flow/model.safetensors     IR 1589 ノード / 458 initializer / 158.9MB（焼き込み表 0.15MB）
models/sbv2/dec/model.safetensors      IR  246 ノード / 197 initializer /  58.7MB
models/sbv2/voice/model.safetensors    IR 1836 ノード / 655 initializer / 217.6MB
models/sbv2/<target>/io.<case>.safetensors  入力と torch CPU 期待出力
```

#### 格納 dtype の系列（`--dtype f16` / `--dtype i8` — ADR 0018 / 0019）

`--dtype f16` / `--dtype i8` はそれぞれ**別系列** `models/sbv2-f16/<target>/` /
`models/sbv2-i8/<target>/` へ書く（f32 系列と同居させると既存 E2E の f32 tolerance が黙って
圧縮資産に掛かる）。丸め（fake-quant）は共有の `quantize.round_weights_to_f16` /
`quantize.fake_quant_int8` を **`remove_weight_norm` / パッチ適用の後・参照と golden の
採取の前**に、export する各ターゲットのモジュールへ当てる。i8 は **w8a8 の受け皿ではない**
（SBV2 は 5 ターゲットとも conv1d が 86〜90% で linear が実質 0 GFLOP — ADR 0025 決定⑤）—
狙いは資産サイズとロード時間で、計算は f32 のまま（w8a32）。

| ターゲット | f32 格納 | f16 格納 |    比 | i8 格納 |    比 | 適格（圧縮常駐） |
| ---------- | -------- | -------- | ----: | ------- | ----: | ---------------- |
| `dp`       | 1.78MB   | 0.90MB   | 50.4% | 0.46MB  | 25.7% | 4 本 / 0.44MB    |
| `front`    | 33.39MB  | 17.96MB  | 53.8% | 10.33MB | 30.9% | 80 本 / 7.72MB   |
| `flow`     | 158.86MB | 79.92MB  | 50.3% | 40.65MB | 25.6% | 156 本 / 39.47MB |
| `dec`      | 58.71MB  | 29.43MB  | 50.1% | 14.84MB | 25.3% | 98 本 / 14.64MB  |
| `voice`    | 217.60MB | 109.37MB | 50.3% | 55.53MB | 25.5% | 254 本 / 54.11MB |

（適格の本数は 3 系列で同じ。バイト数は i8 系列のもの — 圧縮対象の集合は格納 dtype に依らず
`WEIGHT_SLOTS` の重みスロットだけで決まる。）front だけ比が高いのは、焼き込んだ相対位置表
（2.1MB の i32 / f32 定数）が重みスロット適格外で f32 のまま残るため。合計は f32 470.34MB →
f16 237.57MB（50.5%）→ **i8 121.81MB（25.9%）**で、i8 の companion scale は 505,576 B
（圧縮バイトの 0.42%）。

MUST: `--dtype` は **emit 専用**（`--sym-max` と同じ）。`--verify` との併用は CLI が拒否する
— 検証は格納形式を見ない eager 比較で、しかも dec / voice では丸めを `remove_weight_norm` の
**後**にしか当てられないのに参照は remove の**前**に採るため、併用すると「丸めた側 vs
丸めていない側」の比較になって `bit_exact` の主張が壊れる。

dp のラッパは `forward(h, x_mask, g) -> logw`。素の `DurationPredictor.forward` は `g` が
`Optional` で分岐を持つので、必須にして分岐を消し、入力名を recon の呼び名に揃えている
（IR の入力名は forward の引数名がそのまま出る）。動的軸は `Dim("P", min=2, max=512)` —
**sym_max はターゲット別で、front 系が 512（P = 音素数）、flow/dec/voice は 4096
（T = フレーム数）**。機械的な強制が無く誤値は沈黙するので、既定値は台本側に持たせて
`--sym-max` で逸脱するときだけ明示させる（ADR 0013）。

front のラッパは `forward(x, x_mask, tone, language, bert, style_vec, g, z_noise)
-> (logw_sdp, logw_dp, m_p, logs_p)`。`x_mask` は**外部入力**（原実装は `x_lengths` から
内部生成するが、長さを「値として」使う形は畳み込みに載らない）で、sdp reverse の乱数も
外部入力 `z_noise` に昇格している（`noise_scale` の乗算はホスト側 — 実行時ノブをグラフに
焼かない）。`sdp_ratio` の混合と durations 化も同じくホスト側。

flow / voice のラッパは `forward(z_p, y_mask, g, idx_k, valid) -> z / audio`、dec は素の
`Generator.forward(x, g) -> audio [1,1,512T]`（ラッパを置いていない）。`idx_k` / `valid` は
相対位置注意の `(T,T)` 表で、**front と違ってグラフ入力**（下の「パッチ層」を参照）。

golden ケースは**全ターゲット共通の 5 本**（Deno 側が 1 本の表でターゲット横断に等値検査
するため）。dp の `h`、front の `x` / `tone` / `bert`、flow / dec の `z_p` / `x` は長さごとの
固定 seed の randn で、`g` は**実重みの話者埋め込み（`emb_g`）**、`style_vec` は**実資産
（`style_vectors.npy`）**から引く（合成乱数だと値域が実運用と対応せず、tolerance の根拠が
浮く）。

| ケース   | 長さ | 内容                                                           |
| -------- | ---- | -------------------------------------------------------------- |
| `p2`     | 2    | 下限（`torch.export` の 0/1 特殊化を避ける最小値）             |
| `p37`    | 37   | 短め                                                           |
| `p203`   | 203  | 中                                                             |
| `p512`   | 512  | front 系の宣言上限ちょうど                                     |
| `padded` | 16   | 末尾 5 列をマスク 0（front / flow のマスク経路の唯一の検出器） |

- ケース名の `p<n>` は front 系の **P（音素数）**由来だが、flow 系では **T（フレーム数）**を
  指す。1 本の表を全ターゲットで共有する（＝どのターゲットもケースを欠かせない）ことを
  優先して名前は据え置いてある。
- **flow 系の宣言上限 4096 を踏む golden は無い**。表と注意スコアが O(T²)、dec の出力が
  512·T なので、T=4096 では io 1 ケースで 134MB / 出力 210 万点になり golden 資産としても
  実 GPU テストとしても割に合わない。上限近傍は「宣言上限に依存した実装が無いこと」を
  長さの散らばり（2 → 512 で 256 倍）で踏む側に寄せている。

`padded` の入力は**パディング列にも値が入っている** — マスク乗算が効いていれば出力の末尾は
厳密に 0 になり、外れれば値が漏れる。この形でないとマスク経路の検出器にならない
（`tests/test_export_sbv2.py` が「マスクを全 1 に差し替えると末尾が 0 でなくなる」ことまで
固定して、恒真化を塞いでいる）。

ロード時に固定する assert が 2 つ:

- **相対位置注意の窓幅 == 4**（`enc_p` と `flow` の両方）。gather 化パッチの添字表は
  `clamp(rel + 4, 0, 8)` を焼き込むので、窓幅が違うと**幅の違う埋め込みを黙って読む**
  （要素数は合うので shape エラーにならない「沈黙誤値」クラス）。しかも golden も同じ誤りで
  生成されるため数値突合もすり抜ける。ckpt を差し替えた瞬間に落とすのが最小の対処なので
  ローダ側に置く（ADR 0013）。
- **weight_norm 由来のパラメータが `enc_p` / `sdp` / `dp` / `flow` に無いこと**。残っていると
  `weight` が実効重みではなくなり、そのまま書き出すと別のモデルになる（実測 0 件だが、
  無い前提で何もしていない側なので fail loudly で固定する）。**`dec` はここに入らない** —
  有効な weight_norm（95 モジュール / 190 パラメータ）を持って出荷される側で、除去は
  `ensure_dec_plain` が export の直前に行う。

`ensure_dec_plain`（冪等）は dec の `remove_weight_norm` を通してから上の assert を掛ける。
**重みの丸め（`--dtype f16` / `--dtype i8`）は remove 後の実効重みに当てる**順序制約があり
（remove より先に丸めると `weight_g` / `weight_v` を丸めることになって実効重みが丸めの格子に
乗らない。i8 では捨てられる要素が amax に効いて per-channel scale ごとずれる）、
`ensure_dec_plain` と `_fake_quant` の docstring に MUST として書いてある
（ADR 0013 / 0018 / 0019）。

Deno 側は `packages/runtime/tests/e2e_sbv2_test.ts`（1 ケース = 1 テスト）と
`packages/runtime/tests/sbv2_relattn_parity_test.ts`（表のバイト一致）。DeBERTa と同じ二段構えで、
`models/sbv2/` に**ターゲットのディレクトリが 1 つも無ければ全 SKIP**（生の重みだけ置いて
export をまだ流していない環境がこれ）、**中途半端にある場合**（ターゲット欠け / ケース欠け）
は FAIL。**系列（f32 / f16 / i8）でパラメタ化**してあり、tolerance は**系列 × ターゲットごとに
実測から導く**（系列間の流用禁止 — 片方の再導出がもう片方を黙って動かす）:

| ターゲット | f32 atol | f32 rtol | 判定の主役   | f32 実測 maxAbs | f16 atol | f16 rtol | f16 実測 maxAbs | i8 atol | i8 rtol | i8 実測 maxAbs |
| ---------- | -------- | -------- | ------------ | --------------- | -------- | -------- | --------------- | ------- | ------- | -------------- |
| `dp`       | 1e-6     | 1e-5     | rtol         | 2.62e-6         | 1e-6     | 1e-5     | 3.58e-6         | 1e-6    | 2e-5    | 2.38e-6        |
| `front`    | 1e-4     | 1e-5     | atol         | 1.75e-5         | 3e-4     | 1e-5     | 3.62e-5         | 2e-4    | 1e-5    | 2.37e-5        |
| `flow`     | 2e-5     | 1e-6     | atol         | 2.74e-6         | 2e-5     | 1e-6     | 2.15e-6         | 2e-5    | 1e-6    | 2.62e-6        |
| `dec`      | 3e-5     | 1e-6     | atol（単独） | 4.00e-6         | 2e-5     | 1e-6     | 2.37e-6         | 5e-5    | 1e-6    | 6.07e-6        |
| `voice`    | 1e-5     | 1e-6     | atol（単独） | 1.60e-6         | 1.5e-5   | 1e-6     | 1.71e-6         | 1.5e-5  | 1e-6    | 1.74e-6        |

f16 系列で有意に増えたのは front の `logw_sdp` だけ（1.75e-5 → 3.62e-5）。他の 4 ターゲットは
f32 系列と同桁で、丸めた重みが golden 側にも入っている（= 量子化誤差が差に入らない）ことの
裏取りになっている — 掛け忘れていれば差は重みの相対 5e-4 級に化けて 3 桁上に出る。i8 系列も
同じ構造で、5 ターゲットとも f32 系列と同桁に収まる（front が 1.4 倍・dec が 1.5 倍で最大 —
掛け忘れなら per-channel の量子化誤差 4e-3 級が桁で上に出る）。判定の主役はどの系列でも
ターゲットごとに同じ（値域の形で決まるので格納 dtype では動かない）。

flow / dec / voice で rtol を主役に据えられないのは、出力（潜在変数 z と波形）が 0 を跨ぐ
値域で、`|ref|` の最小非ゼロが 1e-8 級まで落ちるため — そこで相対誤差が発散する（dec の
実測 maxRel は 0.44 に達するが、その要素の絶対誤差は 1e-8 級）。導出根拠は同ファイルの
`SBV2_*_TOLERANCE` のコメントが正本。**dec の最終段 `tanh` は WGSL 実装依存で torch と
ビット一致しない**ので、この突合は原理的に許容誤差込みでしか成立しない。

pytest 側は `tests/test_export_sbv2.py`（台本の約束事）と `tests/test_patch_sbv2.py`
（パッチ層の単体）。golden 入力の作りと CLI の排他は実重み不要で常に走り、export 本体は
**実重みと `sbv2` グループが揃っている環境でだけ**走る（無ければ SKIP）。

### パッチ層（`karume/patch_sbv2.py`）

front と flow / voice は無改造では export できない（dec は不要）。パッチは import 済み
クラスの属性差し替え（monkeypatch）とラッパで行い、`style_bert_vits2` パッケージ本体には
触れない。

| パッチ                        | 何をするか                                            | なぜ必要か                                                                                          |
| ----------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| spline の分岐フリー・非破壊化 | boolean-mask indexing → clamp + where、in-place → cat | 区間内要素だけの抽出は**要素数がデータ依存**で `GuardOnDataDependentSymNode` になる                 |
| 相対位置注意の gather 化      | rel⇄abs シフト → 添字表 + 最終次元 gather             | シフトが `2P−1` / `2P²` / `P(2P−1)` の**二次 shape 式**を作る（次元言語のアフィン拡張でも救えない） |
| FFN の pad 畳み込み           | 明示 `F.pad` → `conv1d(padding=…)`                    | `constant_pad_nd` を減らす。**奇数 kernel かつ非 causal のときだけ**厳密に等価（assert で固定）     |

#### 相対位置表の 2 方式（ADR 0013）

添字表（`idx_k` / `valid` / `idx_v`）は `i` と `j` だけで決まり **長さに依存しない**ので、
エクスポータの定数畳み込みが焼き込み + `sym_prefix_slice` に落とせる。ただし焼き込み量は
O(sym_max²) なので、**ターゲットで方式を変える**:

| 系統          | key 側 `idx_k` / `valid` | value 側 `idx_v`    | 焼き込み量  |
| ------------- | ------------------------ | ------------------- | ----------- |
| front (P≤512) | 焼き込み `(512,512)`     | 焼き込み            | 実測 2.1MB  |
| flow / voice  | **グラフ入力**           | 焼き込み `(4096,9)` | 実測 0.15MB |

flow で焼き込むと `(4096,4096)` × 2 本 = **134MB** になるため入力へ昇格した。式の正本は
`patch_sbv2.build_relattn_tables` で、**front の in-graph 構築も同じ関数を呼ぶ**（式を 2 箇所
に書くと片方だけ直したとき両者が黙って別の表になる）。ホスト側の鏡像は
`packages/runtime/tests/helpers/relattn-tables.ts`（SBV2 固有なので `packages/runtime/src/` には置かない — 将来
`examples/` へ昇格する）で、**バイト一致は `packages/runtime/tests/sbv2_relattn_parity_test.ts` が golden の
実データで固定**する。窓幅 4 の食い違いは shape エラーにならない沈黙誤値クラスなので、
Python 側は `_assert_window_size`（ckpt ロード時）、TS 側はパリティテストが**コンテナに
焼き込まれた `idx_v` の幅 `2w+1`** と突き合わせて落とす（両側に門を置くのは、片側だけだと
ホストと golden が同じ誤りを共有してすり抜ける経路が残るため）。

#### flow / voice のラッパ

- `FlowReverse` は `[TransformerCouplingLayer(mean_only) + Flip] × 4` を reversed 順に適用。
  coupling reverse は `torch.split` → 明示スライス（96/96）に置き換え、**`exp(−logs)` の
  乗算を畳む**（`mean_only=True` なので `logs = zeros_like(m)`、`exp(−0.0)` は IEEE 754 で
  厳密に 1.0 なのでビット一致 — `zeros_like` を IR に持ち込まないための定型手筋）。
- `Sbv2Voice` は `FlowReverse` → `z * y_mask` → `dec`。参照 infer 末尾と同順で、`max_len`
  スライス（推論経路では常に `None` = 恒等）は持ち込まない。

#### プロセス汚染に対する門

- **`--verify` と emit は同一プロセスで併用できない**（CLI が `parser.error` で拒否する）。
  パッチはクラス属性の**プロセス全域**差し替えで、`remove_weight_norm` は重みを破壊的に
  畳むので、emit が先に走ると「前の参照」が採れなくなり同値検証が**恒真化して偽 PASS**
  する。検証自体も「全ケースの参照を確定 → 変更 → 比較」の順序で、順序が破れたら
  参照採取の直前で `RuntimeError` にする（ADR 0013）— front / flow / voice は
  `patch_sbv2.patches_applied()`、**dec は逆向き**に「weight_norm 由来パラメータが
  まだ残っていること」を見る（汚染源が remove だから。voice は両方の門を持つ）。
- **`--verify` はターゲットを 1 つだけ取る**。「MHA パッチ系どうしは排他 / dec の remove は
  voice とだけ排他 / 丸めは全てと排他」という対ごとの排他表を CLI に持たせると、表の穴が
  そのまま偽 PASS になる。値を 1 つ取る形なら**汚染の組み合わせが構造的に存在しない**。

#### 同値の実測

| `--verify` | ケース | worst maxdiff               | 備考                                          |
| ---------- | ------ | --------------------------- | --------------------------------------------- |
| `front`    | 9      | 2.02e-5 @P=512              | `P ≤ 5` はビット一致                          |
| `flow`     | 10     | 1.43e-6 @T=512              | `T ≤ 5` はビット一致                          |
| `dec`      | 10     | **0（全ケースビット一致）** | remove_weight_norm 前後                       |
| `voice`    | 10     | 1.25e-6 @T=203              | 未パッチ flow + weight_norm 有効 dec との比較 |

- 差の出所は value 側の縮約長が変わることによる BLAS の順序差で、実 GPU golden の誤差と
  同じ桁に収まる（front 1.75e-5 / flow 2.74e-6 / voice 1.60e-6）。
- **dec の全ケースビット一致は recon §6 の未検証事項を閉じたもの**（それまでは 1 ケース
  `z=(1,192,50)` の実測しか無く、実効重み `g·v/‖v‖` の f32 再現はスペック保証ではないと
  記録されていた）。`torch.equal` まで要求しているので `0.0` / `-0.0` の取り違えも通らない。

## 実重み Anima の export

画像生成側の実重み。ADR [0016](../../docs/decisions/0016-anima-chain-export.md) の emit
ターゲット 4 本を `export_anima.py` が書き出す。フル emit（`models/anima/`）と Deno 側 E2E
（`packages/runtime/tests/e2e_anima_test.ts`）まで M1-P4 で完了済み。

### 依存と重みの入手

```bash
uv sync --group anima   # accelerate / diffusers==0.39.0 / torchvision / transformers==5.14.1
```

重みは HF Hub の `circlestone-labs/Anima-Base-v1.0-Diffusers`（初回だけ自動 DL・5.3GB）。
`diffusers` を `==` でピンするのは、`patch_anima` が `QwenImageRMS_norm` /
`QwenImageResample` / `QwenImageUpsample` / `QwenImageAttentionBlock` の forward をクラス属性
ごと差し替え、`AnimaTextConditioner` / `CosmosTransformer3DModel` の forward を逐語で書き下した
ラッパを持つため（マイナー更新でグラフ形や eager 同値の前提が黙って変わる）。

### 生成と突合

```bash
# emit（IR + golden io を <out>/<target>/ へ）
uv run --group anima python export_anima.py --out /path/to/out
uv run --group anima python export_anima.py --target vae_decoder --out /path/to/out
uv run --group anima python export_anima.py --target transformer --num-layers 2 --out ...

# パッチ前後の eager 同値（**1 プロセス 1 ターゲット** — CLI が併用を拒否する）
uv run --group anima python export_anima.py --verify text_encoder
uv run --group anima python export_anima.py --verify vae_decoder

# LoRA を焼き込んでから emit（transformer / text_conditioner に効く）
uv run --group anima python export_anima.py --target transformer --lora turbo.safetensors

# S 形（トークン長 1 シンボル）の追加系列 — transformer 専用・既定 out に -dyn が付く
uv run --group anima python export_anima.py --dtype f16 --dit-graph dyn --lora turbo.safetensors \
  --out ../../models/anima-turbo-f16-dyn
uv run --group anima python export_anima.py --dtype f16 --dit-graph dyn --verify transformer \
  --lora turbo.safetensors
```

- **`--verify` と `--target` は同一プロセスで併用できない**。VAE パッチはクラス属性の
  プロセス全域差し替えなので、emit 側が先に当てると「パッチ前の参照」が汚染されて同値検証が
  恒真化する（差が常に 0 になる = 緑が証拠にならない壊れ方 — ADR 0013 の規律）。
- **`--lora` は `--num-layers` より前に焼く**。後にすると切った層に対応する LoRA が「対象が
  無い」まま黙って捨てられ、`fuse_lora` の取りこぼし検査が縮小モデルで効かなくなる。

### `--dit-graph dyn`（DiT の S 化 — #21 波 T2）

グラフの入口を patchify の**後ろ**へずらし、rope の cos / sin 表をグラフ入力へ昇格した
**追加系列**（静的系列は 1 バイトも動かさない）。入口 `tokens [1,S,68]` / rope 表
`[1,1,S,128]` ×2、出口は unpatchify 前の `[1,S,64]`。**解像度依存の焼き込みが 0 本**になる
（静的形は padding channel と rope 表の 3 本）。設計の根拠は
[dynres-vae-tiling](../../docs/research/2026-08-03-dynres-vae-tiling.md) §2.2。

- **transformer 専用**。他 3 ターゲットは解像度に依らないので静的系列と共有する（CLI が
  他ターゲットの指定を拒否する）。`--resolution` も**効かないので拒否する** — golden の
  解像度は `DIT_DYN_RESOLUTIONS = (512, 1024)` 固定で、グラフ自体は解像度を持たない。
- 系列ディレクトリには `model.safetensors` / `io.*` に加えて
  **`rope_base.safetensors`**（64KiB）が並ぶ。ホスト（`examples/anima/host/dit-tokens.ts`）が
  rope 表を組むための**軸別素表**で、`model.rope` の出力から切り出したもの。
  **これが要る理由**: `torch` の f32 三角関数は正しく丸めた値と 1 ulp ずれることがあり
  （実測: 位置 × 周波数 8,192 通りで cos 472 件 / sin 231 件）、JS の `Math.cos` では
  再現できない。静的グラフには torch の値が焼かれているので、素表を並べ替える形でしか
  ビット同一にならない。素表は**解像度に依らず**、行数（= 上流の
  `seq = arange(max(max_size))` の長さ）がモデル側の対応上限になる（Anima では 128 =
  latent 256 = 2048px 相当）。
- `--verify transformer --dit-graph dyn` は「ホスト patchify → S 形 → ホスト unpatchify」を
  **パッチ前の diffusers 経路**と突き合わせる。実測（turbo LoRA 焼き込み・f16 丸め後）:
  **2 ケースとも `bit_exact=True` / maxdiff 0.000e+00**（S=1,024 と S=4,096）。
- 実 GPU の主門は `packages/runtime/tests/e2e_anima_dyn_test.ts`（S 形 ≡ 静的グラフの Uint32 完全一致）。

### 実測（波 2 時点。DiT / Qwen3 は `--num-layers 2`、他はフル）

| ターゲット         | IR ノード | model.safetensors | symbols       | max rank |
| ------------------ | --------- | ----------------- | ------------- | -------- |
| `text_encoder`     | 131       | 749.8MB           | `T`           | 4        |
| `text_conditioner` | 613       | 539.1MB           | `Tsrc` `Ttgt` | 4        |
| `transformer`      | 316       | 629.4MB           | （静的）      | 4        |
| `vae_decoder`      | 455       | 101.3MB           | （静的）      | 4        |

**未対応 aten op は 4/4 とも 0 種**（波 2 着手時点は和集合 19 種 — recon §2）。層を切っていない
`text_conditioner` / `vae_decoder` はこの表がそのままフル深度、`text_encoder` はフル 28 層でも
別途 0 種を実測済み（正規化後の FX ノード 2467）。DiT のフル 28 層は f32 で 7.29GiB なので
波 3 で測る — 層数は未対応 op の**種類**を変えない（同じブロックの繰り返し）。

### 実測（波 3・**フル深度**。`models/anima/` へ全 4 本）

各ターゲットを**別プロセス**で回した実測（ホスト RAM のピークを重ねないため。`--target` を
並べて 1 プロセスで回すこともできるが、DiT のピークが他の 3 本の常駐に積み上がる）。

| ターゲット         | IR ノード | initializer | model.safetensors | emit 時間 | ピーク RAM |
| ------------------ | --------- | ----------- | ----------------- | --------- | ---------- |
| `text_encoder`     | 1769      | 317         | 2,386,195,204 B   | 17.5s     | 3,794MiB   |
| `text_conditioner` | 613       | 122         | 539,060,388 B     | 5.7s      | 1,152MiB   |
| `transformer`      | 3904      | 579         | 7,827,646,080 B   | 36.5s     | 11,593MiB  |
| `vae_decoder`      | 455       | 108         | 101,279,604 B     | 10.0s     | 1,546MiB   |

`model.safetensors` の合計 10,854,181,276 B、`io.<case>.safetensors` 込みの `models/anima/`
全体で 10,868,931,292 B（10.12GiB）。**`models/` は `.gitignore` 配下**（git に入れない）。
op 語彙は 4 本の和で **23 種**（`add bmm cat clamp clamp_min conv2d div embedding expand gelu
layer_norm linear mul neg permute reshape rms_norm sigmoid slice softmax sqrt sum
sym_prefix_slice`）で、**新 op はゼロ**（波 1 で足した 3 種で足りた）。

**再生成の決定性**: `vae_decoder` を別ディレクトリへもう 1 度 emit し、`model.safetensors` /
`io.case0` / `io.case1` の 3 本とも **sha256 一致**を確認した（乱数は `SEED` からの派生で
グローバル seed に依存しない）。大物 3 本は emit 時間とディスクの都合で未確認 — 乱数経路は
4 本とも同一実装なので、確認したのは `vae_decoder` の範囲であることを明示しておく。

### 実測（波 3・`--verify` 済みの eager 同値はフル深度でも据え置き）

Deno 側の実 GPU 突合の結果は `packages/runtime/tests/e2e_anima_test.ts` の tolerance コメントが正本。要約:

| ターゲット         | 実 GPU golden              | 備考                                        |
| ------------------ | -------------------------- | ------------------------------------------- |
| `text_encoder`     | maxAbs 5.22e-4（3 ケース） | 折り込み -inf 130,816 個で **NaN 0 個**     |
| `text_conditioner` | maxAbs 2.23e-6（2 ケース） | Tsrc/Ttgt を別々に振って取り違えを潰す      |
| `transformer`      | **未実行**                 | 重み 7,465MiB が本機の GPU 上限 7,280MiB 超 |
| `vae_decoder`      | maxAbs 1.01e-5（2 ケース） | `--verify` の 9.34e-6 と同じ桁              |

`transformer` が未実行なのは**グラフの問題ではない**: `--num-layers 20`（重み 5,613MiB）で
emit したものは実 GPU で完走し torch と一致する（maxAbs 9.63e-5）。詳細は
`docs/known-issues.md`。

パッチ前後の eager 同値（`--verify`）:

| ターゲット         | ケース | worst maxdiff       | 備考                                 |
| ------------------ | ------ | ------------------- | ------------------------------------ |
| `text_encoder`     | 3      | **0（ビット一致）** | 全 1 マスク落とし                    |
| `text_conditioner` | 2      | **0（ビット一致）** | 全 1 マスク 2 本落とし               |
| `transformer`      | 2      | **0（ビット一致）** | timestep 昇格 / padding ゼロ定数化   |
| `vae_decoder`      | 2      | 9.34e-6             | conv3d → conv2d の縮約順序差（下記） |

VAE だけビット一致にならないのは、CausalConv3d(T=1) → conv2d が**同じ数の別の足し方**に
なるため。書き換えごとの同値は `tests/test_patch_anima.py` が f64 で `< 1e-14` に固定して
いる（conv3d↔conv2d / チャネル L2↔`F.normalize` / RMS_norm の rank4 化）。データ移動だけの
nearest-exact ×2 は f32 で**ビット一致**。decoder 全体の増幅率は f64 実測で約 5.4e3 なので、
f32 の丸め（1.19e-7）が 9e-6 級に育つのは整合する。

> NOTE: 上の f64 単体検査と違い、`--verify` を f64 で回しても残差は 0 にならない
> （実測 9.1e-8）。原因は `QwenImageUpsample.forward` が原実装で `x.float()` と f32 へ落として
> いるため — f32 経路（= export する経路）では `.float()` が no-op なのでビット一致する。

### f16 格納の emit（`--dtype f16` — ADR 0018）

`--dtype f16` は**重みを f16 表現可能値へ丸めてから**（fake-quant）参照と golden を採り、
**適格な重みスロットだけ**を f16 で格納する。出力先は f32 系列と別の `models/anima-f16/`
（`--out` 省略時の既定が `--dtype` で切り替わる）。

```sh
uv run --group anima python export_anima.py --dtype f16                    # 4 本まとめて
uv run --group anima python export_anima.py --dtype f16 --target transformer
uv run --group anima python anima_pipeline.py --dtype f16                  # フィクスチャ
```

**MUST: 丸めは参照・golden の採取より前**（ADR 0006）。各 builder がモデルを組んだ直後
（`--lora` の焼き込みより**後**）に `_fake_quant` が掛かる。後ろへ動かすと参照だけが元の重みで
計算され、E2E の差が「量子化誤差 + 実装誤差」の合成になる — tolerance を緩める方向にしか
効かないので、**緑のまま検出力だけが落ちる**壊れ方になる。

**適格判定は 2 条件の AND**（`karume/emit.py`）:

1. その initializer の消費が `WEIGHT_SLOTS`（`linear` / `conv1d` / `conv2d` /
   `conv_transpose1d` の重み = スロット 1、`embedding` = スロット 0）**だけ**である。
   ランタイム側 `packages/runtime/src/runtime/plan.ts` の鏡像で、ずれは適合表
   （`packages/runtime/tests/fixtures/op-contracts.json` の `weight_slot`）が TS / Python 双方から落とす。
   **bias は絶対に載せない** — プロトタイプの f16 降格バグ（bias の f32 定数が weight を
   道連れにして適格 0MB）の根治形。
2. f32 → f16 → f32 の往復が**ビット一致**する。適格なのに一致しないものは fail loudly
   （丸めの掛け忘れ / 順序の誤り / 畳み込み定数が重みスロットへ流れた、のいずれか）。

適格 0 本のまま f16 を指定した場合も `EmitError` で落ちる（ADR 0006 の「適格 0MB を
沈黙させない」の書き出し側）。

**safetensors の並び順**（`docs/limitations.md`）: Karume のリーダはデータ節を「隙間なく・
要素サイズに整列して」覆うことを要求するので、**要素数が奇数の F16**（バイト長 ≡ 2 mod 4）の
直後に F32 / I32 を置くとロードが整列違反で落ちる。並べ替えはエクスポータの責務なので、
`save_file` は使わず自前で順序を決めて書く（順序を外部ライブラリの実装詳細に預けない）:

    F32（名前昇順）→ I32（名前昇順）→ 偶数要素 F16 → **奇数要素 F16（末尾）**

奇数 F16 より前は全て 4 の倍数長なので累積 offset は 4 の倍数を保ち、奇数 F16 どうしは
2 バイト整列で足りる。書いた直後に `verify_model` が **Karume のリーダ規則を写した検査**
（`assert_reader_layout`）を通す — HF の `safe_open` は整列違反のファイルを**読めてしまう**
ので、そちらを通すだけでは検出できない（`tests/test_emit.py` の故障注入がこれを実証）。
f16 / i8 を 1 本も含まないファイルではこの並びは `save_file` の出力と**バイト一致**する
（tiny golden の f32 系 24 本で確認済み — f32 系列の資産は writer の差し替えで 1 バイトも
動かない。25 本目の `i8_weights` だけが圧縮格納）。

#### 実測（2026-08-03・`--dtype f16` でフル深度・別プロセス）

| ターゲット         | model.safetensors | f32 比 | 適格（f16 格納）   | 適格外（f32 格納） | emit 時間 | ピーク RAM |
| ------------------ | ----------------- | ------ | ------------------ | ------------------ | --------- | ---------- |
| `text_encoder`     | 1,194,225,580 B   | 50.0%  | 197 本 / 1,192.0MB | 120 本 / 1.86MB    | 19.0s     | 4,292MiB   |
| `text_conditioner` | 269,838,164 B     | 50.1%  | 62 本 / 269.2MB    | 60 本 / 0.48MB     | 6.2s      | 1,360MiB   |
| `transformer`      | 3,914,867,592 B   | 50.0%  | 454 本 / 3,912.8MB | 125 本 / 1.22MB    | 42.2s     | 11,807MiB  |
| `vae_decoder`      | 50,732,956 B      | 50.1%  | 37 本 / 50.5MB     | 71 本 / 0.075MB    | 10.4s     | 該当なし   |

`models/anima-f16/` 全体で 5,444,414,308 B（5.07GiB。f32 系列は 10.12GiB）。丸めた重みは
text_encoder 5.96 億 / text_conditioner 1.35 億 / transformer 19.56 億 / vae 1.27 億要素。
**適格外のバイトは全ターゲットで 0.5% 未満**（bias・norm 系 weight・畳み込み定数）で、
「適格外は VRAM 削減ゼロ」という制約は Anima では実害にならない。

`Session.diagnostics().storage` の実測（Deno 側）— `residentCompressedBytes` は GPU に
圧縮のまま載ったバイト数、`hostExpandedBytes` は f16 宣言のうちロード時に f32 展開したぶん:

| ターゲット         | resident   | hostExpanded |
| ------------------ | ---------- | ------------ |
| `text_encoder`     | 1,136.8MiB | 0.0MiB       |
| `text_conditioner` | 256.8MiB   | 0.0MiB       |
| `transformer`      | 3,731.5MiB | 0.0MiB       |
| `vae_decoder`      | 48.2MiB    | 0.0MiB       |

`hostExpanded` が全て 0 なのは**設計どおり**: エクスポータは適格判定を通ったものだけを f16 と
**宣言する**ので、ランタイム側の「f16 宣言だが適格外 → CPU 展開」経路には入らない
（この経路自体は手書き IR で踏めるので `packages/runtime/tests/gpu_f16_weights_test.ts` が単体で固定している）。

**これで DiT フル 28 層が実 GPU に載った**: f32 の 7,465MiB は GPUBuffer 天井 7,280MiB
（`docs/known-issues.md`）を超えて load できなかったが、f16 では 3,731.5MiB で天井の半分。
実 GPU golden と通しチェーン段②の実測値は `packages/runtime/tests/e2e_anima_test.ts` の tolerance コメントが
正本（DiT golden maxAbs 6.68e-5・段② 生の DiT 出力 3.03e-5・段③ 通し 6.41e-6）。

`anima_pipeline.py --dtype f16` は 4 コンポーネント**全て**を fake-quant してから参照を採る
（1 つでも素のまま残すとその段だけ別のモデルの数になる）。出力は
`models/anima-pipeline-f16/`（21 テンソル・9.4MB）で、実測 44s / DiT 1 step あたり 13.5s。

#### 実測: Anima の重みは元から BF16 なので、f16 丸めはほぼ恒等（2026-08-03）

HF の `circlestone-labs/Anima-Base-v1.0-Diffusers` は **4 コンポーネントとも safetensors 上が
`BF16`**（`text_encoder/config.json` の `torch_dtype` も `bfloat16`）。BF16 の仮数 8bit は f16 の
11bit に収まるため、f32 へ上げてから f16 へ丸めても値が動くのは **f16 の非正規化数域へ落ちる
極小値だけ**:

| コンポーネント     | 丸めで値が動く要素             | 最大変化 |
| ------------------ | ------------------------------ | -------- |
| `vae`              | 163,271 / 1.27 億 = **0.129%** | 2.98e-8  |
| `text_conditioner` | 16,718 / 1.35 億 = **0.0124%** | 1.59e-4  |

**この事実は f16 系列の読み方を縛る**: f16 系列と f32 系列の誤差が同じ桁だったのは
「量子化が無害だから」ではなく「元が BF16 のモデルに f16 を掛けたから」で、f32 学習の重みでは
別の話になる。

**故障注入の結果（重要 — 期待と違った）**: 「fake-quant を参照採取より後ろへ動かす」を実際に
注入した結果は次のとおり。

| 注入                                        | 結果                                                                                                                                     |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| ① builder の `_fake_quant` を外す           | **emit が fail loudly**（`EmitError: initializer 'p_vae_decoder_conv_in_weight' … fake-quant が未適用か、参照採取より後に掛かっている`） |
| ② ①に加えて適格判定（往復ビット一致）も外す | 資産は書けるが **E2E は緑のまま**（vae_decoder maxAbs 6.85e-6 / text_encoder 4.40e-4 — どちらも tolerance の内側）                       |

②で赤くならないのは上の BF16 の事実の帰結で、丸め前後の torch 出力の差そのものが小さい
（実測: text_encoder t024 で 2.64e-3 = 値域 68.7 に対し相対 3.8e-5、vae_decoder で 1.72e-5）。
つまり **Anima では E2E は「丸めの掛け忘れ」の検出器にならない** — 検出しているのは emit の
適格判定 1 本だけである。この門を「冗長」と見なして外さないこと。

> NOTE（2026-08-03・SBV2 f16 で対照実測）: この「E2E に映らない」は**モデル（配布 dtype）
> 依存の性質**で、f16 格納一般の性質ではない。ckpt が真の f32 の SBV2 では同じ注入で
> emit の門と E2E の**両方**が赤くなる（atol の 31 倍超過 — ADR 0027）。

### i8 格納の emit（`--dtype i8` — ADR 0019）

`--dtype i8` は **per-channel symmetric int8**（zero-point なし）で fake-quant してから参照と
golden を採り、適格な重みスロットだけを i8 + companion scale で格納する。出力先は
`models/anima-i8/`。

```sh
uv run --group anima python export_anima.py --dtype i8              # transformer だけ
uv run --group anima python export_anima.py --verify transformer --dtype i8
uv run --group anima python anima_pipeline.py --dtype i8            # フィクスチャ
```

**MUST: `--dtype i8` は transformer 専用**（`DTYPE_TARGETS` — 他ターゲットは `--target` /
`--verify` で明示しても CLI が拒否する）。理由は 2 つ:

1. **系列設計**（ADR 0019）: DiT の −1.87GiB が支配項で、text / cond / VAE は
   `models/anima-f16/` を共有する。VAE の i8 化はプロトタイプ実測で 2 桁小さい。
2. **VAE は丸めの順序制約を満たせない**: `patch_anima` は CausalConv3d の重みを
   **時間方向の最終スライス**へ差し替えるが、これはパッチ適用時（= 参照採取より後）に起きる。
   f16 の要素ごとの丸めはスライスと可換なので害が無いが、i8 の per-channel scale は
   **捨てられる要素まで amax に数えてしまう**（scale がずれると全要素の値が動く）。

**量子化の定義**（`karume/quantize.py`）:

- `scale = clamp(amax / 127, f32 tiny)` を出力チャネルごとに、`q = clamp(round(w/scale), ±127)`。
  **−128 は使わない** — 最大絶対値要素が `q = ±127` に乗って `q·scale` で厳密に復元され、
  fake-quant が**冪等**になる（再適用でビット不変）。下限 clamp は全ゼロチャネルの `0/0` 回避。
- チャネル軸はモジュール型の表（`QUANT_CHANNEL_AXES`）。`ConvTranspose1d` だけ重みが
  `[Cin, Cout, K]` の転置レイアウトなので **1**、他 4 つは 0。op 名で引く鏡像
  （`ops.WEIGHT_CHANNEL_AXES` / TS 側 `packages/runtime/src/ops.ts`）とは適合表
  （`packages/runtime/tests/fixtures/op-contracts.json` の `channel_axis`）が両側から突き合わせる。
- 突合は **FQN**（`<module>.weight`）— `id(tensor)` は使わない（ADR 0006）。`convert.py` が
  safetensors のテンソルキーに FQN をそのまま使うので、同じ空間で emit 側と噛み合う。
  **`--dtype i8` は export するラッパ（`AnimaDit`）に丸めを当てる**のがそのため
  （内側の `model` に当てると FQN の接頭辞が食い違う）。
- 対象 0 本は `QuantizeError` で fail loudly（`--dtype i8` を指定したのに実質 f32 で書けた、を
  沈黙させない）。

**適格判定は f16 と同じ 2 条件の AND**（`eligible_compressed_initializers` を共用）。2 つ目の
「逆変換ビット一致」が i8 では `torch.equal(q8.to(f32) · scale, t)` になる。**scale は fake-quant が
使った値をそのまま**書く（再計算しない）。

**companion scale**: `karume.scale.<重みキー>` という F32 テンソルを同じファイルに入れ、IR 側は
`storage.scale` でそのキーを明示宣言する（`i8` では**必須** — 既定 1.0 で補完すると書き忘れが
「全チャネル 1.0 で dequant した重み」に化けてロードも実行も通ってしまう）。実テンソルとの
名前衝突は書き出し前に検査する。

**並び順**: I8 は要素サイズ 1 で整列制約が無いかわり**任意のバイト長**を作るので、既存の F16
規則の後ろ = **末尾**に置く。前に置くと後続の絶対 offset が要素サイズの倍数から外れる
（`test_emit.py` の故障注入がこれを実証 — HF の `safe_open` は読めてしまう）。

    F32（名前昇順）→ I32 → 偶数要素 F16 → 奇数要素 F16 → **I8（末尾）**

#### 実測（2026-08-03・`--dtype i8` でフル 28 層・別プロセス）

| 指標                            | 実測                                                      |
| ------------------------------- | --------------------------------------------------------- |
| `model.safetensors`             | 1,963,762,200 B（**1,872.8MiB**）                         |
| f32 比 / f16 比                 | **25.09%** / **50.16%**（f32 7,465.0MiB・f16 3,733.5MiB） |
| 適格（i8 格納）                 | 454 本 / 1,956.4MB（19.56 億要素）                        |
| companion scale                 | 5.19MB = 適格バイトの **0.265%**                          |
| 適格外（f32 格納）              | 125 本 / 1.22MB                                           |
| emit 時間 / ピーク RSS          | 44.3s / 11,593MiB                                         |
| `--verify transformer` 2 ケース | maxdiff 0.000e+00・**全ケースビット一致**                 |

`Session.diagnostics().storage`（Deno 側）は `residentCompressedBytes` **1,961,579,776 B
（1,870.7MiB）**・`hostExpandedBytes` **0**。前者はエクスポータの
`compressed_bytes + scale_bytes`（1,956,388,864 + 5,190,912）と**バイト単位で一致**する —
「診断に scale を足す」がエクスポータとランタイムで同じ意味になっていることの実測。

scale のオーバヘッドは ADR 0019 の試算（0.4〜0.9%）より小さい **0.265%**。DiT の Linear が
`[Cout, Cin]` で Cin が 1024〜4096 と大きいため（scale は Cout 本しか要らない）。

`anima_pipeline.py --dtype i8` は **DiT だけ i8・他 3 コンポーネントは f16** で丸めてから参照を
採る（`COMPONENT_DTYPES`）。資産系列（`models/anima-i8/transformer` +
`models/anima-f16/` の他 3 つ）と 1 対 1 に対応させるためで、全部を i8 にすると text 経路の
参照だけが実行される資産と別のモデルの数になる。出力は `models/anima-pipeline-i8/`
（21 テンソル・9,873,808 B）で、実測 DiT 1 step あたり 14.1s / 14.4s。

実 GPU E2E の実測値は `packages/runtime/tests/e2e_anima_i8_test.ts` の tolerance コメントが正本
（DiT golden maxAbs 8.59e-5 / 段② 生の DiT 出力 5.34e-5 / 段② latents 1.19e-6）。**f16 系列の
値は流用していない** — i8 は丸め後の重みが別の数なので、縮約順序に由来する実装誤差も別物になる。

#### 故障注入の結果（2026-08-03・pytest + tiny golden 再生成）

1 点だけ壊して `uv run pytest tests/` と `python -m karume.goldens` を回した
（ハーネスが必ず復元し、復元後のベースライン 1,602 passed も再確認済み）。

| 注入                                            | 結果                                                                                                                 |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| ① 丸め順序の破れ（fake-quant を export の後へ） | pytest **208 errors** / golden 生成が `EmitError: … per-channel scale が無い` で停止                                 |
| ② scale の再計算（軸 0 固定）                   | pytest **2 failed** / golden 生成が `EmitError: … 逆変換してもビット一致しない`（落ちたのは `up.weight` = 転置軸 1） |
| ③ scale の再計算（**正しい軸**）                | pytest **2 failed**（門そのものが消えたことを負のテストが検出）。**golden はバイト不変で再生成できる**               |
| ④ −128 混入（`scale = amax/128`・負側 −128）    | pytest **5 failed**（±127 / 冪等 / scale 定義 / golden 決定性 / scale 不動点）                                       |
| ⑤ 書き出し順の破れ（I8 を先頭群へ）             | pytest **3 failed** / golden 生成が `ContainerError: … 絶対 offset 4175 が F32 の要素サイズ 4 に整列していない`      |

**③ が重要（期待と違った）**: 「fake-quant 済みの重みから正しい軸で scale を引き直す」は
**データとしては何も変えない**。`q` を ±127 に閉じているので最大絶対値要素は必ず `q = ±127` に
乗り、`amax(|q·s|)/127 = fl(fl(127·s)/127) = s` が f32 の**不動点**になる（乱数 8.9e7 サンプルで
反例ゼロ）。つまり ADR 0019 の「再計算禁止」は逆変換ゲートでは検出できない規律で、守るべき
理由は「emit の時点の重みが実効重みでない / 軸が違う / 式が違う」ときに黙って別の scale に
なることのほうにある（②がその実例）。この不動点性は
`test_emit.py::test_recomputing_the_scale_from_a_quantized_weight_is_a_fixed_point` が固定している。

### パッチ層（`karume/patch_anima.py`）

`patch_sbv2` と違い **export 可否ではなく IR の質**が目的（ADR 0016 / recon §5）。素のままでも
4/4 export は通るが、そのままだと conv3d / `linalg_vector_norm` / `upsample_nearest2d` が語彙に
要り、rank が 8 まで上がり、timestep 埋め込みがグラフに焼き込まれる。

| 対象        | 書き換え                                                                                                                                |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3       | `attention_mask` を渡さない（全 1 マスク ⇒ 因果マスクのみ・加算バイアス同一）                                                           |
| Conditioner | 全 1 マスク 2 本を渡さない。512 パディングと出力マスク乗算はホスト側                                                                    |
| DiT         | timestep 埋め込みをグラフ入力へ昇格 / `padding_mask` をゼロ定数チャネル化 / rank4 化                                                    |
| VAE decoder | CausalConv3d(T=1) → conv2d / nearest-exact ×2 → reshape+expand / チャネル L2 → 最終次元 sum + `clamp_min` / `feat_cache` は fail loudly |
| 共通        | RoPE の `inv_freq` をバッファ → 素の属性へ降格（定数畳み込みの葉にする）                                                                |

`--verify` の参照は**パッチ前**に採る（VAE は `vae_patches_applied()` が門番）。参照は
`vae.decode` そのもの — 「conv3d ⇒ conv2d 等価」と「T=1 では feat_cache が結果に効かない」を
1 度の突合で同時に実測する（片方を別経路で確かめると、もう片方の前提が黙って崩れる余地が残る）。

### LoRA（`karume/lora.py`）

ΔW=(B@A)·scale を **f32 で計算**して元 dtype へ in-place 加算する。IR は 1 ノードも変わらず
ランタイム実装は要らない（ADR 0016）。命名変換は diffusers 同梱の
`_convert_non_diffusers_anima_lora_to_diffusers` を呼ぶ（対応表を自前で持たない）。
fail loudly は 4 つ: ファイル全体の `lora_B` が全 0 / 対象 0 件 / `lora_A`・`lora_B` の片方欠け /
形状不一致。成分ごとの ΔW=0 は正常でありうるので、そちらは `FuseReport.is_noop` で知らせる
（例外にしない）。

## Anima のホストパイプライン参照フィクスチャ（`anima_pipeline.py`）

IR に載るのは 4 グラフだけで、その外側 — トークナイズ / sigma スケジュール / timestep 埋め込み表 /
CFG / Euler 更新 / latent 逆正規化 / 512 パディング — は全てホストコードになる。**その「数の正」**を
1 本のフィクスチャに落とす台本。正本は diffusers 0.39 の `modular_pipelines/anima/` 4 ブロックで、
**パッチ層を通さない素の diffusers 経路**で採る（パッチ層の同値は `export_anima.py --verify` が
別に測る — 検証網を独立に保つ。ここでもパッチを通すと、パッチのバグが参照とテスト対象の両方に
同じ形で乗って差 0 のまま素通りする）。

```sh
uv run --group anima python anima_pipeline.py                    # models/anima-pipeline/ へ
uv run --group anima python anima_pipeline.py --steps 32 --ref-steps 2
uv run --group anima python anima_pipeline.py --resolution 1344x768 …   # 非正方（#23）
```

- 出力は `models/anima-pipeline/pipeline.safetensors`（21 テンソル・9.4MB）と `pipeline.json`
  （プロンプト・step 数・shift・CFG 係数・各テンソルの役割と shape）。
- **配布形 `models/anima-turbo/` 直下に置かない** — あちらは manifest が宣言したファイルだけを
  並べてそのまま HF へ上げる木で、宣言外のファイルが混ざると `verify_dist` が止まる
  （`models/sbv2-demo/` を分けたのと同じ理由）。
- プロンプトは英語 1 本の固定値（danbooru 系タグ）。**ネガティブを空文字列にしない** — 空の
  T5 id 列は長さ 1 になり conditioner の受理集合 `Dim("Ttgt", min=2)` から外れる。
- `latents_init` は `SEED = 20260802` 固定（グローバル seed に依存しない）。
- **`--resolution` は `WxH`**（正方は略記 — 綴りはデモの `--resolution` と同じで、正本は
  `karume/resolution.py`）。非正方では `latents_init [1,16,H/8,W/8]` と
  `padding_mask [1,1,H,W]` の**軸の順**が唯一の落とし穴で、入れ替えても要素数は合う。
  メタには綴り（`resolution`）と `width` / `height` の両方が載る — **読み手は後者を見る**
  （`resolution` は正方のとき int のままで、既存フィクスチャを読むテストとの互換のため）。
- 生の DiT 出力（`noise_cond_*` / `noise_uncond_*`）も残す。これが無いと Deno 側で CFG と Euler の
  ホストグルーを**単体でパリティ検査できず**、DiT の誤差と混ざった形でしか見られない。しかも
  σ の刻みが `sigmas[1] − sigmas[0] = −1.064e-2` しかないので、DiT の誤差は Euler 更新で
  約 1/100 に薄まる（latent だけを見る検出器は構造的に鈍い）。
- `image` は `vae.decode` の戻り値そのもの。`AutoencoderKLQwenImage._decode` が最後に
  `clamp(-1, 1)` を掛けており（`AnimaVaeDecoder` が焼き込んでいるのはこの clamp で、
  postprocess 由来ではない）、フィクスチャ側と IR 側で clamp の位置が揃う。

実測（`--steps 32 --ref-steps 2` / 512px）: **44.0s・ピーク RAM 12,918MiB**（DiT を CPU f32 で
4 回 = 2 step × cond/uncond。1 step あたり 13.5s）。テキスト側は qwen 29 / t5 30 トークン
（ネガティブは 13 / 21）。

Deno 側のホストグルー実装（`packages/runtime/tests/e2e_anima_test.ts` の `sigmaSchedule` / `cfgEulerStep` /
`denormalizeLatents` / `padSequence`）は、このフィクスチャと **4 本ともビット一致**する
（`Math.fround` で 1 演算ずつ f32 に丸める）。

### Turbo LoRA の焼き込みと turbo 参照フィクスチャ

`--lora` に少ステップ蒸留の Turbo LoRA（例: `models/anima-turbo-lora-v0.2.safetensors`。
実重みはリポジトリに含まれない — 手動配置）を渡すと export 前に重みへ焼き込まれる。この
LoRA は text_conditioner 側の `lora_B` が**全ゼロ（noop）と実測済み**なので、**transformer
ターゲットだけを emit すれば足りる**（他 3 ターゲットは既存 `models/anima-f16/` を共有）:

```sh
uv run --group anima python export_anima.py --dtype f16 --target transformer \
  --lora ../../models/anima-turbo-lora-v0.2.safetensors --out ../../models/anima-turbo-f16
```

`--verify transformer --lora <path>` は LoRA 適用後の重みに対する eager 同値検証になる
（`_apply_lora` が fake-quant・参照採取より前に効くため、追加のコード変更なしで両側が同じ
LoRA 適用済みモデルを見る。実測: 全ケースビット一致）。

Turbo 運用（steps=10 / CFG=1）の参照フィクスチャ:

```sh
uv run --group anima python anima_pipeline.py --dtype f16 --steps 10 --ref-steps 10 \
  --guidance-scale 1.0 --lora ../../models/anima-turbo-lora-v0.2.safetensors \
  --out ../../models/anima-pipeline-turbo-f16
```

`--guidance-scale 1.0` は **uncond 分岐の DiT 呼び出し自体を省略する**（実運用の turbo
デプロイと同じ形）ため、出力フィクスチャに `noise_uncond_stepNNNN` キーが**存在しない**。
guidance_scale != 1.0 の base 系列 fixture とキー集合が異なる点に注意（ホストグルー側は
このキー欠落を「uncond を計算しない」分岐として扱う）。

## VAE decode の固定タイル化 参照フィクスチャ（`anima_tiling.py`）

`examples/anima` の `--vae-tiling`（VAE decode を latent 64×64 の固定タイルに割る）の
「ホスト側の数の正」。VAE decoder のグラフは**解像度に対して構造不変**なので（512px 用と
1024px 用の `model.safetensors` はノード列・重みバイトまで完全一致 — recon
`docs/research/2026-08-03-dynres-vae-tiling.md` §1.2）、512px 用資産をそのままタイル
decoder として使える。**IR にもランタイムにも追加はゼロ**で、切り出し / ブレンド / 貼り付け
だけがホスト（`examples/anima/host/tiling.ts`）に載る。

```sh
uv run --group anima python anima_tiling.py     # models/anima-tiling-f16-1024/ へ
uv run --group anima python anima_tiling.py --resolution 1344x768 \
  --latents ../../models/anima-pipeline-turbo-f16-1344x768/pipeline.safetensors
```

- 入力 latent は `models/anima-pipeline-turbo-f16-1024/pipeline.safetensors` の
  `latents_denorm`（逆正規化済み = VAE decoder の入力そのもの）を借りる。**先にそちらを
  生成しておく**（無ければ名指しで落ちる）。randn ではなく実パイプラインの latent を使うのは、
  継ぎ目の出方が値の中身に依るから。
- 出力は `tiling.safetensors`（`latents_denorm` / `image_tiled`・13.0MB）と `tiling.json`
  （**タイル幾何** = 軸ごとの開始位置列 / stride / ブレンド幅、および非タイル decode との
  差の観測）。Deno 側（`packages/runtime/tests/e2e_anima_tiling_test.ts`）は数値だけでなく**この幾何メタとも
  突き合わせる** — 数値だけだと「別の幾何でも tolerance の内側」を排除できない。
- **`vae.enable_tiling()` は使わない**。上流の `tiled_decode` は `range(0, H, stride)` で
  走査するので最後のタイルが短くなり、固定形のタイル decoder では食えない。走査は
  「最後のタイルの開始位置を `extent − tile` へスナップする等間隔配置」に変えてある
  （recon §4.2 が予告した意図的逸脱）。台本は `vae.use_tiling` が真なら fail loudly する。
- **ブレンドの式は上流の逐語移植**（`blend_v` / `blend_h`）。同値は
  `tests/test_anima_tiling.py` が**本物のメソッドとのビット一致**で固定する — 走査を自前に
  した以上、式の同型はここでしか担保できない。
- 重みは資産系列と同じ dtype へ fake-quant してから参照を採る（ADR 0006 —
  `anima_pipeline.py` と同じ規律）。既定 `--dtype f16` は TS 側が開く `models/anima-f16/
  vae_decoder` に対応する。

- **`--resolution` は `WxH`**（非正方は #23）。幾何は入力 latent の shape から組むので、
  この引数は「借りた latent が意図した解像度か」の門と既定 `--out` の名前を決めるだけ。
  1344×768（latent 96×168）は **2×3 = 6 タイル・stride 32/52・ブレンド 256/96px**。

実測（1024px・CPU f32・9 タイル）: 非タイル decode との差は **maxAbs 5.07e-2 /
mean 9.82e-4**。タイル化は decoder の attention の受容野をタイル内に閉じる**近似**（上流の
`tiled_decode` と同じ近似）なので、**この差は 0 にならないのが正常**。実装誤差はこれとは別で、
実 GPU 対 torch は maxAbs 1.642e-5（`packages/runtime/tests/e2e_anima_tiling_test.ts` の tolerance 導出）。
1344×768（6 タイル）の同じ観測は **maxAbs 9.97e-2**、実 GPU 対 torch は **8.02e-6**
（`packages/runtime/tests/e2e_anima_nonsquare_test.ts`）。

## 非正方の rope 表 参照フィクスチャ（`anima_rope.py`）

S 形 DiT（`--dit-graph dyn`）のホストは rope の cos / sin を**軸別素表の並べ替え**で組む。
**正方ではこの並べ替えの h ↔ w 取り違えが原理的に検出できない** — Anima の `rope_scale` は
h と w が同値（`[1.0, 4.0, 4.0]`）なので `cos_h` と `cos_w` がバイト単位で一致し、H'=W' なら
表そのものが同じ値になる（ADR 0034 の検出限界 1）。**非正方では位置の取り違えが割れる**ので、
上流 `model.rope` の表を 4 幾何ぶん焼いて TS 側の再構成と Uint32 完全一致で突き合わせる。

```sh
uv run --group anima python anima_rope.py       # models/anima-rope-nonsquare/ へ（数秒）
```

- 幾何は `GEOMETRIES` の固定 4 本（**16:9 と 3:4 の縦横** = 1344×768 / 768×1344 / 1152×896 /
  896×1152・どれも S=4,032）。**縦横の対を必ず両方入れる** — 片方だけだと「h と w を
  入れ替えた実装」がもう一方の幾何の表と一致してしまう。台本は正方が混ざったら落ちる。
- **重みを 1 バイトも読まない**。`CosmosRotaryPosEmbed` はパラメータもバッファも持たない
  純計算なので、モデルは `meta` デバイス上に config から組む（7.3GiB のロードが要らない）。
- 出力は `rope.safetensors`（幾何ごとの `cos_<WxH>` / `sin_<WxH>`・各 `[1,1,S,128]`・16.5MB）と
  `rope.json`（latent 寸法・トークン格子・S・素表の行数）。
- Python 側の鏡像（素表からの再構成 ≡ 上流の表）は `tests/test_anima_rope.py` が**実重み無しの
  合成 rope**（`rope_scale` の h と w を別の値にして取り違えが数に出る構成）で固定する。

## 画像デモのプロンプト層（`anima_demo.py`）

`examples/anima/text/`（プロンプト文字列 → トークン id 列の Deno 実装）が要る**実行時資産**と、
その**パリティ用フィクスチャ**を出す台本。モデルグラフには触らない。

```sh
# 資産（models/anima-demo/text/ へ 2 本）+ フィクスチャ（packages/runtime/tests/fixtures/anima-text/）
cd tools/exporter
uv run --group anima python anima_demo.py
# 生成後は必ず整形する（commit 形はフォーマッタが正 — verify の fmt --check が fixtures も見る）
cd ../.. && deno fmt packages/runtime/tests/fixtures/anima-text/parity.json
```

**1 回の実行で必ず両方**出す（同じ表から作らないと、実行時資産とフィクスチャが別々に古びて
「テストは緑だがデモだけ別の id 列」になる）。実測 **約 3 分**（支配項は下の網羅検査）。

| 出力                                                     | サイズ（実測）            | 中身                                                                    |
| -------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------- |
| `models/anima-demo/text/qwen2-tokenizer.json`            | 3,514,619 B               | 語彙 151,643 / merges 151,387 / 文字クラス表 / NFC 分節表 / 追加語彙 26 |
| `models/anima-demo/text/t5-tokenizer.json`               | 1,093,419 B               | 語彙 32,100 + スコア / 正規化表 / 追加語彙 103                          |
| `packages/runtime/tests/fixtures/anima-text/parity.json` | 474KB（整形後・git 管理） | 28 ケースの参照 id 列 + NFC 251 対 + 語彙の**部分集合**                 |

- 実行時資産（計 4.6MB）は **`models/` = `.gitignore` 配下**。生の `tokenizer.json` 計 13.8MB
  から実行に要る情報だけを抜いた形で、**ライセンス物をリポジトリに抱えない**。
- **MUST: 配布形 `models/anima-turbo/` 直下に置かない** — あちらは manifest が宣言したファイル
  だけを並べてそのまま HF へ上げる木（`models/anima-pipeline/` を分けたのと同じ理由）。
- フィクスチャは語彙の**部分集合**（Qwen2 218 語 / merges 375 / T5 125 語）だけを持つので、
  151k / 32k を commit せずに全ケースを再現できる。正規化表と文字クラス表は**畳み込みの成果物
  そのもの**（= 検証対象）なので全体を載せる。

### なぜ表を焼くのか（`karume/anima_text.py`）

**Unicode 判定を TS で再実装しない / 標準 API に委ねない**（`sbv2_demo.clean_text_ranges` と同じ
規律）。判定の正本は Rust（`tokenizers` / その正規表現エンジン / `unicode-normalization`）側の
Unicode 表で、JS エンジンの ICU 版とずれた瞬間に pre-token の切れ目や正規化結果が変わり、
**例外も警告も出さずに id 列だけが別物**になる。Python から全コードポイントを実評価して
閉区間表・写像表へ畳み、TS は二分探索で引くだけにする。

焼くのは 5 種:

| 表                              | 焼き方（正本への問い合わせ）                                   | 実測                          |
| ------------------------------- | -------------------------------------------------------------- | ----------------------------- |
| 文字クラス `\p{L}` `\p{N}` `\s` | `Split(Regex(…), behavior="removed")` に 1 文字ずつ投げる      | 677 / 144 / 10 区間           |
| `(?i:)` の同一視                | `Split(Regex("(?i:s)"), …)` を接尾辞 8 文字ぶん                | 9 組                          |
| `NFC` の分節                    | `normalizers.NFC` と素の NFC が割れる cp を 8 文脈の探り針で   | 123 cp / 43 区間              |
| `Precompiled` 正規化            | charsmap の DARTS 復号 + **発火しうる規則だけ**へ最小化        | 5,512 規則                    |
| クラスタ境界（3 表）            | 「6 バイト以上へ押し上げると規則が不発火」を利用した**探り針** | extend / breakAfter / prepend |

**`NFC` も標準 API では足りない**（2026-08-03 実測）: 正本の `unicode-normalization` は Unicode 表が
古く、**123 コードポイント**で `String.prototype.normalize("NFC")` / `unicodedata` と出力が割れる
（120 cp は結合クラスを 0 と見なして並べ替えない・3 cp は新しい canonical composition を持たない）。
畳み方は「割れる cp で文字列を分節し、各節だけを素の NFC に掛けて連結する（分節 cp 自身は正規化に
参加させない）」。実害は id 列の沈黙不一致で、`PROMPT_CASES` に該当文字が 1 つも無かったため
28 ケースの門を素通りしていた（乱択 1,200 プロンプトで Qwen2 6 件の不一致を実測 → 修正後 0 件）。

**`(?i:)` は ASCII 相当ではない**（2026-08-03 実測）: Rust の `(?i:)` は Unicode の simple case
folding で、接尾辞の 8 文字（s t r e v m l d）では **U+017F（ſ）が `s` と同一視される**。
`.lower()` / `toLowerCase()` / ASCII の大小反転はいずれも正本ではなく、`it'ſs` の切れ目が
`'ſ` + `s` ではなく `'ſs` になる（fixture ケース `apostrophe_fold` がこの境界）。

**`Precompiled` は NFKC でも最長一致でもない**: 書記素クラスタ単位の丸ごと置換 + **最短接頭辞
勝ち**で、`A`+U+0301+U+0301 → `Á`（3 文字目が黙って消える）/ `A`+U+0302+U+0301 → `Â`
（3cp 規則 `Ấ` ではなく 2cp 接頭辞が勝つ）。UTF-8 6 バイト以上のクラスタは丸ごと置換の経路に
入らず、そのおかげでハングル / 地域表示子 / 絵文字 ZWJ 列は境界規則を実装せずに済んでいる。

### 検証の三段（どれか 1 つでも外れたら emit しない）

| 段                            | 場所                                             | 規模（実測）                                                                                                                                                |
| ----------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ① 畳み込みの同値              | `anima_demo.py`（emit の門）                     | 全 1,112,064 cp + 規則鍵 5,512 + 乱択 200,000（seed 固定）/ pre-token 走査と NFC 分節はそれぞれ **全 cp × 11 文脈 = 12,232,704 件**                         |
| ② 参照実装 vs `AutoTokenizer` | `anima_demo.py` + `tests/test_anima_demo.py`     | 28 ケース（`padding="longest"` / `max_length=512` / `truncation=True` — `anima_pipeline.encode_text` と同じ呼び方）+ **乱択 2,000 プロンプト**（seed 固定） |
| ③ TS 実装 vs フィクスチャ     | `packages/runtime/tests/anima_tokenizer_test.ts` | 28 ケース × 2 トークナイザ + NFC 251 対 + 性質テスト + パイプライン参照とのクロスチェック                                                                   |

- ①は**再生成時にしか走らない**（数分かかる）。pytest 側（②）は commit 済みフィクスチャに対して
  同じ突合をやり直すので、**再生成しなくても上流の tokenizer.json の変化に気づける**。
- pre-token 走査の探り針は 11 文脈。`'{}` の 1 文字だけでは**足りない** — 短縮形の選択肢が
  発火してもしなくても同じ 1 断片になり、大小無視の取り違えが素通りする（実測でこの穴を踏んだ）。
  NFC 分節も同じ形で、検出用（8 文脈）と検証用（11 文脈）を**別に**持つ。
- ②の乱択 2,000 件は「人手の台本が思いつかなかった組み合わせ」を機械的に踏ませる恒久の門
  （NFC のエンジン差は 28 ケースの隙間を通って id 列を割った）。絵文字・結合文字・分節 cp・
  各種空白を混ぜた alphabet から seed 固定で引く。
- ③は**実資産なしで走る**（フィクスチャに語彙の部分集合が入っている）。資産がある環境では
  実語彙 151k / 32k での再現と、`models/anima-pipeline{,-f16,-i8,-turbo-f16}/` の
  `qwen_input_ids` / `t5_input_ids`（torch 側が採った id 列）とのクロスチェックが追加で走る。
  後者は**フィクスチャ生成器とは別経路**なので、フィクスチャ自体が誤っている場合を捕まえる。
- 上流 `tokenizer.json` の構造前提（`normalizer.type` / `pre_tokenizer` 先頭 / ByteLevel の
  `add_prefix_space` / post_processor が足す特殊トークン / 追加語彙のフラグ / T5 の
  `byte_fallback`）は `anima_demo.check_upstream_shape` が emit のたびに検査する。①〜③は
  **叩いた範囲でしか**拾えないので、前提そのものは構造で固定する。

## 音声デモの資産 prep と torch 参照（`sbv2_demo.py`）

`examples/sbv2/`（実テキスト → WAV の Deno デモ）が要る**ホスト側資産**と、その出力の
**数値パリティ**を受け持つ台本。モデルグラフには触らない（emit 経路も golden も変えない）。

```sh
# ① 実行時資産（models/sbv2-demo/ へ 3 本）
uv run --group sbv2 python sbv2_demo.py assets

# ② デモを走らせる（リポジトリ直下）→ out.wav と dump.safetensors
cd ../.. && deno task demo:sbv2 --text "こんにちは、これはテストです。" && cd tools/exporter

# ③ torch 参照（dump の離散入力・乱数列で同じチェーンを再実行）→ reference.wav + 数値
uv run --group sbv2 python sbv2_demo.py reference --dump ../../models/sbv2-demo/out/dump.safetensors

# ④ 公式 infer（pyopenjtalk 経路）→ official.wav（アクセントの聴き比べ用）
uv run --group sbv2 python sbv2_demo.py official --text "こんにちは、これはテストです。"
```

### `assets` が出すもの

| ファイル                 | 内容                                                                      |
| ------------------------ | ------------------------------------------------------------------------- |
| `symbols.json`           | JP-Extra の ID 化規則とモデル定数・ノブ既定値（**全て実物から引いた値**） |
| `deberta-tokenizer.json` | DeBERTa 文字トークナイザの語彙・特殊 ID・`_clean_text` 判定表             |
| `assets.safetensors`     | `style_vec` `[1,256]` と話者埋め込み `g` `[1,512,1]`                      |

**MUST: 定数を手で写さない。** 記号表・tone 基点（`LANGUAGE_TONE_START_MAP["JP"]`）・言語 ID
（`LANGUAGE_ID_MAP["JP"]`）・add_blank の挿入値は `style_bert_vits2` から引く。多言語版と
JP-Extra で同じに見えて、ずれても **shape は合ったまま音だけが壊れる**（沈黙誤値クラス）。
挿入値だけはソースにリテラルで書かれているので、`blank_id_from_source` が
`infer.get_text` のソースから正規表現で 3 系列ぶん抜き、値が 1 種類に揃うことまで確認する。

`_clean_text` の判定表を焼くのも同じ動機で、`unicodedata.category` の分類を TS へ移すと
ICU の版差が静かな不一致になる — 全コードポイントを Python で実評価して閉区間に畳む。

> NOTE: **`language` は JP-Extra でも全 0 ではない**。`infer.get_text` は
> `cleaned_text_to_sequence(..., JP)` を通すので実音素位置は 1、add_blank の挿入位置だけが 0。
> `export_sbv2.make_language` が全 0 なのは *golden の合成入力*としての選択（どんな値でも
> golden は成立する）で、推論規則ではない。

### `reference` が主張すること

`patch_sbv2` のモジュール群（`Sbv2Front` / `Sbv2Voice`）と**デモと同じホストグルー**を
torch CPU で回し、dump に載った Karume の波形と突き合わせる。つまり測っているのは
「同じ計算グラフを実 GPU で走らせた値 vs torch CPU で走らせた値」で、パッチ前の原実装との
同値は `export_sbv2.py --verify` が別に持つ（層を混ぜない）。合わせて 2 つの門を通す:

- **トークナイズのパリティ** — dump の `bertText` を Python のトークナイザに食わせ、
  dump の `input_ids` と完全一致することを波形突合の**手前で**要求する。ここが割れると
  BERT 特徴が別の音素へ配られ、「音は出るが崩れる」形で沈黙する。
- **`w_ceil` の整数一致** — 継続長は `ceil` なので、front の出力が閾値の直上にいると
  GPU/CPU の 1e-5 差で 1 フレーム飛ぶ。食い違った位置は `w` の値ごとレポートに出す
  （フレークか実装差かを読み手が判定できる形）。

### `official` を別サブコマンドにする理由

`patch_sbv2` のパッチはクラス属性の**プロセス全域**差し替えで、`reference` はそれを当てる。
`official` の主張は「原実装の g2p（pyopenjtalk）・原実装の注意 / spline を通した音」なので、
同一プロセスに同居させると黙ってパッチ後の経路になる。argparse のサブパーサは 1 プロセスに
つき 1 つしか選べないため、**1 プロセス 1 サブコマンド**が構造的に成立する（`--verify` が
対ごとの排他表を持たないのと同じ理由づけ）。

3 本の wav（`out.wav` / `reference.wav` / `official.wav`）は同じ PCM16 変換規則
（クリップ → `floor(x·32767 + 0.5)`）で書く。Python 組み込みの `round` は偶数丸めなので、
そこを揃えないと聴き比べに実装差が混ざる。

## 使い方（台本から）

```python
from karume import export_to_file

graph = export_to_file(module, (x,), "model.safetensors", dynamic_shapes=({0: dim},))
```

`export_to_file` は export → 正規化 → 変換 → 書き出し → **検証**まで通す。書けたが
ランタイムが読めないファイルを配布物として残さないための門なので、経路を分岐させない。

## モジュール構成

| モジュール    | 役割                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------- |
| `dims`        | 次元言語 `coeff·sym+offset`。文法の正本は `packages/runtime/tests/fixtures/dim-grammar.json` |
| `ir`          | IR v1 のグラフ表現と JSON 直列化（`allow_nan=False`）                                        |
| `ops`         | op 契約表（TS 側 `packages/runtime/src/ops.ts` の同義物）                                    |
| `shapes`      | 出力 shape 規則（TS 側 `computeOutputShape` の同義物。宣言 shape と全ノード突合）            |
| `convert`     | ExportedProgram → IR グラフ（定数畳み込み・aten 対応表・CSE）                                |
| `normalize`   | 語彙を増やさない FX 同値書き換え（パス登録制）                                               |
| `emit`        | safetensors への書き出し                                                                     |
| `verify`      | IR v1 の全規則 + 配布形の突合 + ランタイム capability 突合                                   |
| `pipeline`    | 上記を一本道に並べた `export_module` / `export_to_file`                                      |
| `goldens`     | tiny golden fixtures の定義と生成                                                            |
| `patch_sbv2`  | モデル別 — SBV2 を **export 可能にする** monkeypatch 層とラッパ                              |
| `patch_anima` | モデル別 — Anima の **IR の質**を上げる monkeypatch 層とラッパ（動機が違う）                 |
| `lora`        | export 前の LoRA 重み焼き込み（IR は 1 ノードも変わらない — ADR 0016）                       |
| `resolution`  | 解像度の綴り `WxH`（参照台本 2 本が共有。**受理集合の正本はデモ側**）                        |

モデル固有なのは `patch_sbv2` / `patch_anima` の 2 本だけで、`style_bert_vits2` / `diffusers`
の import は**関数内**に閉じてある（パッケージが無い環境でも `import` は通る = 他モジュールの
テストが依存グループに引きずられない）。

**2 つのパッチ層は動機が違う**（ADR 0016）: `patch_sbv2` は「export 可否そのもの」が目的
（分岐フリー化しないと `torch.export` が data-dependent guard で落ちる）。`patch_anima` は
**素のままでも 4/4 export が通る**うえでの品質層 — 新 op を増やさない / rank ≤ 4 に収める /
実行時ノブをグラフに焼かない、の 3 点のために置いている。

### TS 側との契約の同期

`ops` と `shapes` は TS 側（`packages/runtime/src/ops.ts`）と**同じ契約の別実装**で、一致は人手の規律ではなく
適合ケース表 `packages/runtime/tests/fixtures/op-contracts.json` が担保する（`dims` と
`packages/runtime/tests/fixtures/dim-grammar.json` の関係と同じ）。両側のテスト
（`tests/test_ops_conformance.py` / `packages/runtime/tests/ops_conformance_test.ts`）が**自分の実装から導いた
表**をこのファイルへ突き合わせるので、片側だけ動かすと両方が赤になる。載せているのは
op 名の全集合 / アリティ / スロット dtype / attrs キー集合 / attrs の値域 / 出力 shape 規則
（strided コピー族の rank 上限を含む）。

`shapes` は torch の meta が付けた宣言 shape を**正としない** — 契約の規則から独立に計算した
shape と全ノードで突き合わせ、食い違えば export 時に落とす（`convert` の出口と
`verify_model` の両方を通る）。束縛が要る判定（長さ 0 の軸・Tmax 超過）だけはランタイム側の層
が持つ。

## 対応範囲（perf-a 時点）

**op 数と契約の正本は `packages/runtime/tests/fixtures/op-contracts.json`**（TS 側 `packages/runtime/src/ops.ts` と Python 側
`ops.py` が両方ここへ突き合わせる）。以下はその写しなので、食い違ったら適合表の側が正しい。

- **意味論 dtype は f32 / i32 / bool**（ADR 0009）。torch の i64 はエクスポータ境界で i32 へ
  正規化する（値域外は fail loudly）。**格納 dtype は f32 と i32**（i32 は生の int32 —
  ADR 0010 の明示的な例外）。initializer の意味論 dtype は f32 / i32 で、意味論と格納の組は
  `f32 × {f32,f16,bf16,i8}` と `i32 × i32` の 2 通りだけ（交差は fail loudly）。
- IR op は **50 個**（ADR 0017 で `rms_norm` / `conv2d` / `clamp_min`、ADR 0023 で
  `attention` を追加）:
  - unary `neg abs exp log log1p sqrt tanh sigmoid relu gelu`（f32）/ `bitwise_not`（bool）/
    attrs 付き unary `clamp` / `clamp_min` / `leaky_relu`（f32）
  - スカラ比較 `ge_scalar le_scalar gt_scalar`（f32 → bool）
  - binary `add div`（f32）/ `mul sub`（f32・i32）/ `ge`（f32 → bool）/
    `bitwise_and`（bool）と三項 `where`（cond は bool）— いずれも torch 右詰め broadcast
  - `cast`（f32 / i32 / bool 間）/ `matmul`（rank-2）/ `bmm`（rank-3）/
    `gather`（最終次元固定）/ reduce `sum amax amin`（**1 軸**・attrs `dim` 宣言必須・
    keepdim 無し。`sum` は bool 入力 → i32 も受理）/ `cumsum`（最終次元）
  - レイアウト（ADR 0011 / 0014）: `reshape` / `permute` / `expand`（f32 も解禁）/
    `slice` / `cat`（**IR v1 で唯一の可変アリティ op**）/ `pad` / `flip`
  - 記号 prefix スライス（ADR 0010）: `sym_prefix_slice`
  - 融合 op（ADR 0012 / 0015 / 0017 / 0023）: `linear` / `layer_norm` / **`rms_norm`** /
    `softmax` / **`attention`** / `embedding` / `masked_fill` / `conv1d` / **`conv2d`** /
    `conv_transpose1d`
- **attrs を持つのは 26 op**（`sum.dim` / `amax.dim` / `amin.dim` /
  `attention.scale` / `clamp.{min,max}` / `clamp_min.min` / `rms_norm.eps` /
  `conv2d.{stride,padding,dilation,groups}` / `leaky_relu.negative_slope` /
  `ge_scalar.value` / `le_scalar.value` / `gt_scalar.value` / `cumsum.dim` / `cast.to` /
  `permute.dims` / `slice.{dim,start,end}` / `cat.dim` / `pad.{left,right}` / `flip.dim` /
  `sym_prefix_slice.{sym,slices}` / `layer_norm.{normalized_shape,eps}` / `softmax.dim` /
  `embedding.padding_idx` / `masked_fill.value` / `conv1d.{stride,padding,dilation,groups}` /
  `conv_transpose1d.{stride,padding}`）。宣言キーは全て必須で、宣言外のキーと値域外は
  fail loudly（ADR 0012）。**既定値の補完はしない** — `conv1d` の `dilation` / `groups` を
  省略できると depthwise の IR が黙って通常畳み込みになる（ADR 0015）。
  `gelu(approximate="tanh")` は載せる欄が無いので受理しない（黙って別の式で近似しない）。
- 分解停止（preserved）の既定は **11 op**（`PRESERVED_OP_PREFIXES` — ADR 0007 の 9 op に
  ADR 0015 で `leaky_relu`、ADR 0017 で `rms_norm` を追加）: linear / layer_norm / rms_norm /
  softmax / gelu / leaky_relu / conv1d / conv2d / conv_transpose1d / embedding / masked_fill。
  - **12 本目 `scaled_dot_product_attention` は既定に載らない**（ADR 0023）。SDPA は mask /
    causal / GQA を引数で表せてしまい、既定へ足すと Anima text_encoder（−inf 折り込み因果
    マスク）が `_h_attention` の fail loudly に当たって **export できなくなる**。有効化は
    `export_module(…, preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION)` の**ターゲット別
    opt-in**（`export_anima.TARGET_PRESERVED` — 現状は transformer と vae_decoder のみ）。
  - **`rms_norm` の供給ルートは 2 系統**（ADR 0017）: diffusers `nn.RMSNorm` 由来の
    `aten.rms_norm` は保存で残り、手書き分解形（Qwen3 / DiT）は保存では畳めないので
    `normalize._fold_rms_norm` が受け持つ。
- 融合 op のカーネル契約の狭さは IR ではなくランタイム capability 側に置く（ADR 0007）。
  エクスポータ境界で落とすのは**表現が存在しない軸**だけ: 多軸 `normalized_shape`、
  最終次元以外の softmax、非有限の `masked_fill` 埋め値、上限だけの clamp（`clamp_max` が
  語彙に無い）、`output_padding` / `groups` / `dilation` が既定でない conv_transpose1d、
  `2·padding ≠ K − stride`（出力長が `L·stride` にならない形）の conv_transpose1d。
  - `conv1d` / `conv2d` の `groups` / `dilation` は**受理する**（ADR 0015 / 0017）。
  - **省略可能なスロットは合成でアリティを固定する**（`Emitted.synth_consts`）— bias 無しの
    conv / linear はゼロ bias、affine 無しの layer_norm は ones/zeros、weight 無しの rms_norm は
    ones。カーネルと契約に arity 分岐を持ち込まないため（`+0` / `×1` の厳密恒等 —
    ADR 0015 / 0016）。実測の重み: Anima の linear 711 本中 698 本が bias 無し、DiT の
    layer_norm 85 本が全て affine 無し。
  - 下限だけの `clamp(min=eps)` は**別 op**（`clamp_min`）へ落ちる（ADR 0017）。欠けた側を
    ±有限最大値で補うのは「表現が無い形を黙って別の形で実行する」ことになるので採らない。
- 未対応 aten op は**全件列挙**して落とす（1 件目で打ち切らない）。

### 定数畳み込み（Tmax 畳み込みと 2 点評価 — ADR 0010）

定数と shape シンボルだけに依存する部分木は、**各シンボルの上限 Tmax で実評価**して
initializer に焼き、実行時は `sym_prefix_slice` で先頭を切り出す。相対位置バケット表
（`arange / sign / log / ceil / clamp / 比較 / where`）が丸ごとこの経路で消え、バケット境界の
1ulp 差が gather 添字 1 ずれになるバグクラスを構造的に排除する。

- **Tmax の出どころは `ExportedProgram.range_constraints`**（= `dynamic_shapes` の
  `Dim(min=…, max=…)` が torch に登録した制約そのもの）。呼び出し側から別引数で受けない —
  `dynamic_shapes` との二重管理になり、食い違ったときに「宣言より短い定数を焼いて実行時に
  範囲外」が黙って通る。`Dim(max=…)` 未設定（`int_oo`）は fail loudly。
- **適格判定は allowlist ではなく 2 点評価の実測**（`_check_prefix_commutes`）。
  第 1 点 = Tmax、第 2 点 = **Tmax − 1**（0/1 特殊化を避けるため 2 以上でなければならない。
  `Tmax − 1 < max(下限, 2)` の値域は「検査が恒真化する」として fail loudly）。第 2 点の
  評価結果と、Tmax で焼いた定数の prefix（長さ `coeff·sym+offset`）をバイト比較し、
  一致しなければ**畳まずに落とす**。`arange(T)/T` や `full((T,), T)` のように T を「値」
  として使う形は allowlist 掲載 op だけで組めてしまうので、この実測だけが止められる。
- 焼いた定数の dtype は f32 / i32（i64 は境界正規化で i32 へ、値域外は fail loudly）。
  bool の定数は initializer の語彙が無いので落とす。
- `expand` / `repeat` は allowlist に載せない（畳むと B·H 倍に実体化する）— frontier で
  止まり、実行時の strided コピーで済む。

### 正規化パス（`normalize.py`）

登録順に走る。IR 語彙も attrs も増やさない同値書き換えだけを置く。

| パス                       | 書き換え                                             | 発火条件（外れたら触らない）                        |
| -------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| `_drop_metadata_asserts`   | `_assert_tensor_metadata` を削除                     | 常時                                                |
| `_fold_rms_norm`           | `x·rsqrt(mean(x²)+eps)·w` → `rms_norm`               | weight が最終次元長の rank1・eps が有限の正数       |
| `_drop_safe_softmax_guard` | SDPA の safe-softmax ガードを除去                    | **不活性の証明が立つときだけ**（立たなければ例外）  |
| `_lower_unit_expand`       | `unsqueeze→expand→view` → rank ≤ 3 の expand         | rank > `STRIDED_RANK`・複製軸が静的                 |
| `_lower_split_unbind`      | 最終次元 split の幅 1 slice+squeeze → 最終次元 slice | rank > `STRIDED_RANK`・分割が連続で静的             |
| `_lower_reshape_permute`   | rank ≥ 5 の reshape→permute→reshape → rank4 転置列   | rank > `STRIDED_RANK`・端点が rank ≤ 4              |
| `_collect_dead_code`       | 途中 1 回の DCE（**位置に意味がある**）              | 常時                                                |
| `_pow2_to_mul`             | `pow(x, 2)` → `mul(x, x)`                            | 指数がちょうど 2                                    |
| `_drop_identity_repeat`    | `repeat(x, [1,…,1])` → `x`                           | **引数が全て 1 かつ本数 = rank**（rank 上げは別物） |
| `_drop_identity_add`       | `add(x, 0)` → `x`                                    | Python スカラの 0 かつ dtype 不変                   |
| `_promote_scalar_operands` | `add/sub/mul/div(tensor, スカラ)` → 二項 op + 定数   | dtype 不変・rank ≥ 1・`alpha=1`・有限スカラ         |
| `_eq_zero_to_not_bool`     | `eq(x, 0)` → `bitwise_not(cast(x, bool))`            | 右辺がちょうど 0・出力が bool                       |
| `_select_to_squeeze`       | `select.int(長さ 1 の軸, 0)` → `squeeze`             | **静的に**長さ 1（記号軸は例外）                    |
| `_split_to_slices`         | `split_with_sizes` + `getitem` → `slice` 列          | 消費者が getitem のみ・分割軸が静的                 |

- **順序が載荷（3 本）**:
  - `_fold_rms_norm` は `_pow2_to_mul` / `_promote_scalar_operands` より**先**（ADR 0016）。
    eps が rank-1 定数へ昇格するとスカラ照合が外れて畳めなくなる。
  - `_drop_identity_add` は `_promote_scalar_operands` より先。逆順だと `+0` が rank-1 定数に
    なり、恒等除去の対象でなくなる。
  - `_collect_dead_code` は rank 下げ 3 パスの**直後**。畳んだ元パターン（pow/mean/rsqrt・
    ガードの eq/any/logical_not）をここで消さないと、後続パスが死んだ部分木を書き換えて
    統計だけが膨らむ（IR は変わらないので緑のまま気づけない）。
- **rank 下げ 3 パスの発火は `rank > STRIDED_RANK` 限定**（ADR 0016 の安全線）。既存グラフ
  （SBV2 / DeBERTa は全て rank ≤ 4）へ誤爆しない — 新しい定数は作らず、「strided カーネルが
  実行できない rank」を発火条件そのものにしている。実測: この波の追加で **tiny golden 22 本と
  SBV2 / DeBERTa の IR はバイト単位で無変更**。
- **ガードの除去は証明付き**（ADR 0016）: ①依存錐に -inf 源が無い ②-inf が加算マスク 1 本
  由来で、2 つの記号長（5 / 9）の実評価で全行に有限要素が残る — のどちらかが立つときだけ。
  立たなければ `NotImplementedError`（消すと NaN が下流へ流れる）。
- 昇格した定数は `aten.full.default([1], value, dtype=…)` として挿入する。畳み込み
  allowlist に載っている op なので、消費側が畳み込み対象なら定数ごと畳まれ、そうでなければ
  rank-1 の initializer（i32 なら i32 initializer）になり broadcast で吸収される。
- **定数は元のスカラが居た側に置く**。`sub` / `div` は非可換で、実測には
  `sub.Tensor(1, mask)` のようにスカラが左に来る形がある（`1 − attention_mask`）。
- 型昇格を伴う形（`div(i64, 128)` は f32 を返す）と rank-0 テンソル（`[] × [1] → [1]` で
  rank が上がる）は**書き換えない** — 畳み込みか未対応 op の列挙に回す。

## まだ無いもの（後続フェーズ）

- IR の Python 解釈オラクル（数値突合）。数値検証は Deno 側 E2E が受け持つ。
- **bf16 格納**（IR の語彙にはあるが実行経路が無い — ADR 0006）。f16 は ADR 0018、
  i8 + per-channel scale は ADR 0019 で実装済み。w4（group 量子化）は不採用確定。
- **混成格納**（1 ターゲット内で i8 と f16 を混ぜる）。`_apply_weight_dtype` は
  `weight_dtype` 1 個を全体に当てる。
- Anima の `transformer` **f32 格納のフル 28 層の実 GPU 突合**（重み 7,465MiB が本機の
  GPU バッファ上限 7,280MiB を超えて load できない — `docs/known-issues.md`）。f16 格納
  （3,733.5MiB）と i8 格納（1,872.8MiB）では実 GPU E2E が走っている。
- 実行時パイプライン（SBV2: テキスト → durations → y_mask 組み立て → voice / Anima:
  トークナイズ → scheduler → CFG → 逆正規化）のホスト実装。emit したターゲットを繋ぐ層は
  エクスポータの範囲外（Anima は `packages/runtime/tests/e2e_anima_test.ts` にテスト用の TS 実装がある）。
  （Anima の**トークナイザ TS 移植**は `anima_demo.py` + `examples/anima/text/` で完了。
  `packages/runtime/tests/e2e_anima_test.ts` は引き続きフィクスチャの `input_ids` を使う — あちらが測るのは
  NN の数値で、トークナイズのパリティは `packages/runtime/tests/anima_tokenizer_test.ts` が別に持つ。）
