# IR v1 仕様

Karume のモデルフォーマット。ADR [0003](decisions/0003-ir-v1.md) の具体化。
プロトタイプ IR（v0）の実証済み構造を土台に、格納メタの明示化・バージョニング・
capability 宣言を加えた非互換改訂。確定範囲を定義し、拡張は本書を改訂して行う。

改訂履歴（未リリースにつきシムも移行も作らない — ADR 0003 の改訂手順）:

- M0: 初版。
- M1-P2: 意味論 dtype に i32 / bool（ADR [0009](decisions/0009-dtype-i32-bool.md)）、
  レイアウト op（ADR [0011](decisions/0011-layout-strategy.md)）、attrs 語彙と融合 op
  （ADR [0012](decisions/0012-attrs-and-fused-ops.md)）、**i32 initializer / 格納 `i32` /
  `sym_prefix_slice`**（ADR [0010](decisions/0010-symbolic-constant-folding.md)）。
- M1-P3（波 3）: 数理 op 群 10 種 — `where` / `clamp` / `ge_scalar` / `le_scalar` /
  `gt_scalar` / `ge` / `bitwise_and` / `log1p` / `cumsum` / `leaky_relu`（sdp の spline と
  dec が要求。`leaky_relu` の設計裁定は ADR
  [0015](decisions/0015-conv-family-extension.md)）。あわせて **`sum` の bool 入力 → i32**
  と **`expand` の f32** を解禁し、契約表に**出力 dtype 写像**（スロット 0 → 出力、恒等が
  既定）の欄を追加した。
- M1-P3（波 4・5）: レイアウト第 2 群 `slice` / `cat`（**IR v1 で唯一の可変アリティ op**）/ `pad` /
  `flip`（ADR [0014](decisions/0014-layout-ops-full-write.md)）。conv 族の拡張 —
  `conv1d` の attrs に `dilation` / `groups` を追加（**宣言必須・既定値補完なし**）、
  `conv_transpose1d` を新設（ADR [0015](decisions/0015-conv-family-extension.md)）。
- M1-P4（波 1）: `rms_norm` / `conv2d` / `clamp_min` を追加（ADR
  [0017](decisions/0017-rms-norm-conv2d-clamp-min.md)）。`conv2d` は保存リストにあったが
  カーネルが無かった op で、Anima の VAE decoder で実測に出たので**実行できるようになった**。
  attrs に **`[H, W]` の 2 成分**（`conv2d` の `stride` / `padding` / `dilation`）という
  新しい値の形が入る。
- perf-a: 融合 `attention` を追加（ADR [0023](decisions/0023-fused-attention.md)）。
  **アリティ 3 固定・rank-4 head-first・attrs `scale` は半スケール**（q と k の両方に掛かる
  `√scale_factor`）。mask / causal / dropout / GQA は語彙に無く、エクスポータ境界で全件
  fail loudly。SDPA の保存は**ターゲット別**なので、既存の分解形 IR はそのまま有効。

## コンテナ

- 配布形は **safetensors 1 ファイル**。テンソル（重み・定数）と、`__metadata__` の
  キー **`karume_ir`** にグラフ JSON（文字列）を持つ。
- JSON に NaN / Infinity リテラルを含めてはならない（エクスポータは `allow_nan=False`
  相当で書く。パーサは検出したら fail loudly）。

## グラフ JSON

```jsonc
{
  "format": "karume-ir",
  "version": 1,
  "requires": { "ops": ["matmul"] },
  "symbols": ["T"],
  "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 128] }],
  "outputs": ["h"],
  "initializers": {
    "w": { "tensor": "enc.w", "storage": { "dtype": "f32" } }
  },
  "values": {
    "w": { "dtype": "f32", "shape": [128, 64] },
    "h": { "dtype": "f32", "shape": ["T", 64] }
  },
  "nodes": [{ "op": "matmul", "ins": ["x", "w"], "outs": ["h"], "attrs": {} }]
}
```

- `format` は固定文字列 `"karume-ir"`、`version` は整数 `1`。不一致は fail loudly。
- **未知のトップレベルキーは fail loudly**（未リリースにつき前方互換チャネルは持たない。
  必要になったら本書の改訂で導入する）。
- `requires.ops` は nodes で実際に使われる op 名の集合と一致しなければならない（パーサが
  検証）。ランタイムは自分の対応表と突合し、非対応 op を**列挙して** fail loudly する。

## 値と型

- **意味論 dtype**（`inputs[].dtype` / `values[].dtype`）: `"f32" | "i32" | "bool"`。
  計算は常にこの型で行う（f16/bf16/i8 は意味論 dtype に存在しない）。
  実行時格納は i32 = 32bit、bool = **u32 の 0 / 1**（ストレージバッファに 1bit 型が無いため
  — ADR [0009](decisions/0009-dtype-i32-bool.md)）。要素は 3 型とも 4 バイト。
- **op ごとに実行できる意味論 dtype は違う**（一括ではない）。受理集合の正本は契約テーブル
  （`packages/runtime/src/ops.ts` / `karume/ops.py`）で、ランタイムは非対応を列挙して fail loudly。
  グラフ入力の転送は 3 型とも可能。torch 既定の整数 i64 は **エクスポータ境界で i32 へ
  正規化**する（値域外は fail loudly）— IR に i64 は無い。
- **格納 dtype**（`initializers[].storage.dtype`）: `"f32" | "f16" | "bf16" | "i8" | "i32"`。
  前 4 つは**意味論 f32 の符号化**で、`i32` だけが**生の int32**（記号依存定数の焼き込み先 —
  ADR [0010](decisions/0010-symbolic-constant-folding.md)。「格納語彙は f32 の符号化」の
  明示的な例外）。量子化格納は `storage.scale`（scale テンソルの safetensors キー）・
  `storage.group_size` を持てる。**ランタイムが実行できるのは `f32` / `f16` / `i8` /
  `i32`** — bf16 だけが「宣言としては valid、実行は fail loudly（capability 不足の診断付き）」。
  `group_size` は語彙としては残るが**実行経路が無い**（group 量子化 = w4 は ADR 0019 で
  不採用確定）— 付いていれば capability 不足で落ちる。
- **`f16` の実行**（ADR [0018](decisions/0018-f16-weight-execution.md)）: 意味論はあくまで f32
  （「格納のみ量子化・計算は f32」— ADR 0006）で、経路が**適格判定で 2 つに分かれる**。
  - **適格**（その initializer の消費が `linear` / `conv1d` / `conv2d` / `conv_transpose1d` /
    `embedding` の**重みスロットだけ**）: 生の f16 ペイロードのまま GPU 常駐し、dequant は
    カーネル内（`unpack2x16float` で 2 要素/語を展開 — 添字 `i` の値は
    `unpack2x16float(w[i / 2])[i % 2]`）。VRAM は f32 比 ≈ 1/2。**要素数が奇数のときは末尾
    2 バイトをゼロ詰め**して 4 バイト整列させる（読み出しは要素数で打ち切るので値に影響しない）。
  - **適格外**（bias / norm 系の weight / その他の op / 重みスロットと他スロットの混在消費 /
    消費ゼロ）: **ロード時に CPU で f32 へ展開**する。正しさは保たれるが VRAM 削減はゼロで、
    縮むのは配信サイズだけ。bias が適格にならないのは ADR 0006 の「bias は常に f32」規則
    そのもの（低精度適格判定に bias を含めない）。
  - 内訳は `Session.diagnostics().storage`（GPU 常駐圧縮バイト数 / CPU 展開バイト数）で
    取得できる — 「f16 指定なのに適格 0MB」を沈黙させないための常設診断（ADR 0006）。
- **`i8` の実行**（ADR [0019](decisions/0019-i8-weight-execution.md)）: 方式は
  **per-channel symmetric int8**（zero-point なし）。適格判定・2 経路・診断の枠組みは f16 と
  **同じ 1 本**で、違うのは格納の詰め方と scale の扱いだけ。
  - **パッキング**: 1 要素 = 符号付き 8bit。GPU 常駐時は **4 要素を 1 語（u32）へリトル
    エンディアン順**に詰めた並びとして `array<u32>` で束縛し、`unpack4xI8` で展開する
    （語 = `w[i / 4]`、レーン = `i % 4` を**平坦添字**から割り出す）。safetensors 上の
    バイト列は素の I8 のままで、**要素数が 4 の倍数でないときだけ GPU バッファ側で末尾を
    4 バイト境界までゼロ詰め**する（読み出しは要素数で打ち切るので値に影響しない）。
    1 バイト要素なのでリーダの整列制約は無く、ファイル内の並び順は末尾側で構わない。
  - **scale は companion テンソル**（`storage.scale` で**必須**宣言。無ければロード時に
    fail loudly — 既定 1.0 で補完しない）。F32 で、**重みと同 rank の keepdim broadcast 形**
    （`torch.amax(..., keepdim=True)` の出力そのもの — 例: linear `[out,in]` に対し
    `[out,1]`）。実テンソル（他の initializer の `tensor` キー）との名前衝突は拒否する。
  - **チャネル軸は出力チャネル**: linear / conv1d / conv2d / embedding は **0**、
    **conv_transpose1d だけ 1**（重み `[Cin,Cout,K]` の転置レイアウト）。GPU 常駐経路では
    scale を平坦に `wscale[出力チャネル]` と引くので、チャネル軸以外が 1 でない形は
    Session 構築で落ちる。
  - **scale の適用位置は「要素ごと」**（読み出し時 dequant — `out = Σ x·(q·s) + bias`）。
    縮約の外で掛ける形（`(Σ x·q)·s`）は乗算が減るが、CPU 展開とのビット一致を失うので
    採らない。この形のおかげで**適格経路（GPU）と適格外経路（CPU 展開）はビット単位で
    同じ値**を出す（`q·s` の f32 丸めが両側とも 1 回）。
  - VRAM は f32 比 ≈ 1/4（+ scale のオーバヘッド 1% 未満）。診断の
    `residentCompressedBytes` には**scale バッファのバイト数も加算**する。
- `initializers[].tensor` は safetensors のテンソルキー。safetensors 側 dtype は
  `storage.dtype` と一致し（`i32` ↔ safetensors `I32`）、shape は宣言 shape と一致しなければ
  ならない（ロード時検証）。
- **宣言完全性**: 全ての値（inputs・initializers・全ノード出力）はちょうど 1 箇所で宣言される
  — inputs は `inputs[]` で、**initializer と中間値・出力は `values{}` で**。実行時に毎ノード、
  宣言 shape/dtype と照合する。
- initializer の宣言は**数値次元のみ**（記号次元不可）。意味論 dtype は **f32 / i32** で、
  **意味論と格納の組は次の 2 通りだけ**（交差は fail loudly — i32 宣言が f16 のビット列として
  読まれる沈黙誤値を塞ぐ）:
  - 意味論 `f32` × 格納 `f32` / `f16` / `bf16` / `i8`
  - 意味論 `i32` × 格納 `i32`

  bool の initializer は語彙に無い（実測に無く、safetensors の `BOOL` は 1 バイト格納で
  4 バイト前提の転送と噛み合わない）。
- `storage.scale` / `storage.group_size` は `storage.dtype: "i8"` のときのみ許可。
  `i8` では `scale` は**必須**（ADR 0019）。
- 非有限数の拒否はリテラルだけでなく**値レベル**で行う（`1e999` は JSON として構文有効だが
  Infinity に丸まるため、パース時 reviver で拒否）。safetensors ヘッダ側の未知キーは許容する
  （外部フォーマット）— 未知キー拒否はグラフ JSON のみの規則。

## shape と次元言語

- shape 要素は「非負整数」または「次元式文字列」。
- 次元式の正準文法: **`coeff·sym + offset` の一次式・1 次元 1 シンボル**。
  文字列表現は `"T"` / `"8T"` / `"T+8"` / `"8T+2"`（coeff は 2 以上のとき前置、
  offset は 1 以上のとき `+` 後置。`"1T"` や `"+0"` は非正準として拒否。負の係数・
  負のオフセット・複数シンボルは拒否）。
- シンボル名は `[A-Za-z_][A-Za-z0-9_]*`。`symbols` に列挙され、かつ**少なくとも 1 つの
  入力 shape に係数 1・オフセット 0 の素の形（`"T"`）で出現**しなければならない
  （束縛は入力 shape の次元位置から直接取る。要素数からの逆算はしない）。
- 文法の正本は 1 箇所: 適合ケース表 `packages/runtime/tests/fixtures/dim-grammar.json`（valid / invalid /
  束縛評価の 3 節）。TS 実装と（M1 の）Python 実装は同じ表で検証する。

## ノード

- SSA・単一代入: 各値名は inputs / initializers / いずれかのノードの `outs` のうち
  **ちょうど 1 箇所**で定義される。`ins` は定義済みの名前のみ参照。`outputs` は定義済み
  名の部分集合。`outs` は長さ 1 以上の配列（複数出力はスキーマ上有効。M0 の op は全て
  単一出力）。
- `nodes` は**トポロジカル順**で格納される（パーサが検証。前方参照は fail loudly）。
- `attrs` は op ごとの契約テーブルで検証する。未知の attr・契約外の値は fail loudly
  （近似実行しない）。

## op セット（契約は実装の契約テーブルが正本、ここは一覧のみ）

> 実装は 2 つある（TS `packages/runtime/src/ops.ts` とエクスポータ `karume/ops.py` +
> `karume/shapes.py`）。両者が同じ契約を持つことの正本は適合ケース表
> `packages/runtime/tests/fixtures/op-contracts.json` で、op 名の全集合 / アリティ / スロット dtype /
> attrs キー集合と値域 / 出力 shape 規則（rank 上限を含む）を両側のテストが同じ表へ
> 突き合わせる（次元文法と `dim-grammar.json` の関係と同じ）。

- unary elementwise: `neg, abs, exp, log, log1p, sqrt, tanh, sigmoid, relu, gelu`（f32）/
  `bitwise_not`（bool）
  - `clamp`（f32、attrs `min` / `max`）— 両端とも**必須**の有限 f32 で `min <= max`
    （逆転は WGSL の `clamp` が未定義になる向き）
  - `clamp_min`（f32、attrs `min`）— **片側 clamp**（ADR
    [0017](decisions/0017-rms-norm-conv2d-clamp-min.md)）。上限の欄は持たない（「欠けた側を
    ±有限最大値で補う」近似を構造的に潰す）。既存 `clamp` の attrs を optional にして
    兼ねる案は「宣言済み attrs の既定値補完はしない」（ADR 0012）を崩すため採らない
  - `leaky_relu`（f32、attrs `negative_slope`）— **既定値補完はしない**（ADR
    [0015](decisions/0015-conv-family-extension.md)。実測は 0.1 と 0.01 の混在で、既定に
    頼ると片方が黙って誤る）
  - `ge_scalar` / `le_scalar` / `gt_scalar`（f32 → **bool**、attrs `value`）— 有限 f32 との比較
- binary elementwise: `add, div`（f32）/ `mul, sub`（f32・i32）/ `ge`（f32 → **bool**）/
  `bitwise_and`（bool）— いずれも torch 準拠の右詰め broadcast
- `where`（**三項**、attrs 無し）— `out = cond ? a : b`。スロット別 dtype
  （cond: bool / a, b: f32 → 出力 f32）で、**3 者とも右詰め broadcast**（条件も値と同じ規則で
  広がる）。出力は条件ではなく**値の側**と同型
- `cumsum`（f32、attrs `dim`）— 最終次元の前縁和 `out[…, j] = Σ_{i ≤ j} x[…, i]`。
  **最終次元のみ**受理（`dim` は非負表記 — softmax と同じ絞り方）。長さ 0 の軸は素通し
- `cast`（f32 / i32 / bool 間。attrs `to` が変換先）
- `matmul`（rank-2 × rank-2、f32）
- `bmm`（**rank-3 × rank-3** のバッチ matmul `[B,M,K] × [B,K,N] → [B,M,N]`、f32、attrs 無し）
  — バッチ次元は完全一致（broadcast も stride 0 も無い）。rank-2 は `matmul` の担当で、
  兼用しない
- `gather`（**最終次元固定**、attrs 無し）— `out[..., j] = src[..., index[..., j]]`。
  src と index は同じ rank で先行次元が完全一致し、最終次元だけが自由。**入力スロットごとに
  dtype が違う唯一の op**（src: f32 / index: i32 → 出力 f32）。添字の値域 `0 <= index < D`
  は実行時データ依存なので shape 契約では見ない（範囲外の扱いは実装契約 —
  GPU は該当要素のみ NaN 汚染 / CPU 参照は throw）
- reduce（**1 軸**、attrs `dim`、keepdim 無し）: `sum`（f32 → f32 / **bool → i32** の真の個数）/
  `amax, amin`（f32）— `dim` は**宣言必須**の非負軸番号（既定値補完をしない。省略を許すと
  チャネル軸の縮約を書いたつもりの IR が黙って最終次元を畳んだ別の計算として実行される）。
  最終次元は行カーネル、それ以外は軸変種で実行するが、**縮約順序は両変種で厳密に一致**する
  （設計と記号検証は research/2026-08-04-vae-axis-reduce-recon.md §5.2）
- レイアウト（ADR [0011](decisions/0011-layout-strategy.md)）— いずれも単項:
  - `reshape`（f32 / i32 / bool）— **出力の宣言 shape が目標形**。契約は要素数一致のみで、
    要素順は変えない。attrs 無し（目標形を attrs に持たせると宣言と二重管理になる）
  - `permute`（f32、attrs `dims`）— `dims[d]` = 出力の次元 d が取る入力の次元番号。
    **非負整数の全単射**のみ（負の軸表記はエクスポータ境界で正規化する）
  - `expand`（f32 / i32 / bool）— 出力の宣言 shape へ右詰め broadcast。**拡張できるのは
    長さ 1 の次元だけ**（長さ n → m の複製は語彙に無い）。attrs 無し
- レイアウト第 2 群（ADR [0014](decisions/0014-layout-ops-full-write.md)）— 全カーネルが
  **出力の全バイトを書く**（full-write）形だけを語彙に入れる:
  - `slice`（f32、attrs `dim` / `start` / `end`）— **静的軸・静的範囲**の切り出し
    `x[…, start:end, …]`。3 つとも非負整数（負の軸・負の添字表記はエクスポータ境界で正規化）、
    `start ≤ end ≤ 入力の軸長`。**対象軸は宣言レベルで静的**（記号軸の切り出しは
    `sym_prefix_slice` の担当 — 重複させない）。実行は strided 読みコピー族の流用で、
    可変点は params の offset 1 語
  - `cat`（f32、attrs `dim`）— 静的軸の連結。**唯一の可変アリティ op**（入力 2 本以上）で、
    連結軸以外の次元は全入力で一致、出力の軸長は入力の軸長の総和。実行は strided **書き**
    コピー族で、入力ごとに出力の部分領域へ書く（全入力で出力全域を覆う）
  - `pad`（f32、attrs `left` / `right`）— **最終次元の定数 0 埋め**。幅は非負整数（負幅の
    切り詰めは語彙に無い）、埋め値の欄は持たない（実測は 0 のみ）。専用カーネル 1 本が
    範囲内を転写し**範囲外にも 0 を書く**（ゼロ初期化されたバッファを前提にしない）
  - `flip`（f32、attrs `dim`）— 静的軸の添字反転（shape は恒等）。対象軸は宣言レベルで静的
- **記号 prefix スライス**（ADR [0010](decisions/0010-symbolic-constant-folding.md)）:
  - `sym_prefix_slice`（f32 / i32、attrs `sym` / `slices`）— エクスポータが記号 T 依存の
    部分木を **Tmax で実評価して焼いた定数**から、束縛後の長さの**先頭**を切り出す単項 op。
    `slices` は `[{ dim, coeff, offset }, …]` で、`dim` 軸の出力長は次元言語と同じ
    `coeff·sym+offset`（`coeff ≥ 1` / `offset ≥ 0` / 同じ `dim` の重複は不可 / 空配列は不可）。
    `slices` に無い軸は入力のまま残る。
    - **入力は記号を含まない静的 shape**（= Tmax 形。実際はエクスポータが焼いた initializer）。
      入力側に記号次元があると「Tmax 形」の前提が崩れ、読み出し stride が実行ごとに変わる。
    - `sym` は `symbols` に宣言済みでなければならない（束縛が取れないと prefix 長が決まらない）。
    - 出力 shape は attrs と束縛から**計算**し、宣言と照合する（reshape / expand のような
      「宣言が目標形」ではない）。`coeff·sym+offset` が入力の当該次元を超えたら fail loudly。
    - 実行はレイアウト op と同じ strided 実体化コピー（開始位置 0・入力側の連続 stride）
- 融合 op（ADR [0012](decisions/0012-attrs-and-fused-ops.md) / ADR
  [0015](decisions/0015-conv-family-extension.md) / ADR
  [0017](decisions/0017-rms-norm-conv2d-clamp-min.md)）— エクスポータが分解を止めて 1 ノードの
  まま運ぶ高位 op。ADR [0007](decisions/0007-op-vocabulary.md) の保存 op は M1-P3 で
  `leaky_relu`、M1-P4 で `conv2d` / `rms_norm` を足して **11 本**（ADR 0017）で、いずれも
  **カーネルを持つ**（`conv2d` は Anima の VAE decoder で実測に出た）。`rms_norm` だけは
  **供給ルートが 2 系統**ある: ①diffusers `nn.RMSNorm` 由来の `aten.rms_norm` は保存リスト
  経由でそのまま 1 ノードになる ②Qwen3 / DiT の手書き分解形は保存では畳めないので、
  エクスポータの畳み込みパス（`_fold_rms_norm`）が 1 ノードへ合成する（ADR 0016 / 0017）:
  - `linear`（f32、attrs 無し、**アリティ 3 固定**）— `x[…,in] × W[out,in] + b[out]`。
    重みは `[out, in]` の転置レイアウトのまま。bias 無しの形は語彙に無い
  - `layer_norm`（f32、attrs `normalized_shape` / `eps`、**アリティ 3 固定**）— 最終次元のみ
    （`normalized_shape` は長さ 1）、affine 常時あり。分散は**母分散（correction = 0）**、
    `eps` は有限の正数
  - `rms_norm`（f32、attrs `eps`、**アリティ 2 固定**）—
    `y = x · rsqrt(mean(x², 最終次元) + eps) · weight`。**bias が無く、平均も引かない**
    （layer_norm との差はこの 2 点）。正規化長の正本は **weight の長さ**で、
    `normalized_shape` の欄は持たない（同じ事実を attrs と入力で二重に持たない）。
    `weight` は最終次元長の rank1・`eps` は有限の正数。weight 無しの形はエクスポータが
    ones を合成してアリティ 2 へ正規化する
  - `softmax`（f32、attrs `dim`）— **最終次元のみ**（`dim` は非負表記）。**safe-softmax**
    （行の最大値を引く）で計算する
  - `embedding`（attrs `padding_idx`）— `weight f32[V,H] × index i32[…] → f32[…,H]` の行
    gather。**入力スロットごとに dtype が違う**（weight: f32 / index: i32 → 出力 f32）。
    `padding_idx` は**受理するが forward には効かない**（勾配側の欄）。範囲外添字の扱いは
    `gather` と同じ（GPU は該当行のみ NaN 汚染 / CPU 参照は throw）
  - `masked_fill`（attrs `value`）— `out = mask ? value : x`。スロット別 dtype
    （x: f32 / mask: bool → 出力 f32）で、**出力は常に x と同形**（mask だけが右詰め
    broadcast で読まれる）。`value` は有限の f32 スカラ
  - `conv1d`（f32、attrs `stride` / `padding` / `dilation` / `groups`、**アリティ 3 固定**）—
    `x[B,Cin,L] * W[Cout,Cin/groups,K] + b[Cout]`。出力長は
    `floor((L + 2·padding − dilation·(K−1) − 1) / stride) + 1`。**4 つの attrs は全て宣言必須で
    既定値補完をしない**（省略を許すと depthwise の IR が黙って通常畳み込みとして実行される）。
    `groups` は `Cin` / `Cout` の両方を割り切ること・重みの第 2 軸が `Cin/groups` である
    ことが契約
  - `conv2d`（f32、attrs `stride` / `padding` / `dilation` / `groups`、**アリティ 3 固定**）—
    `x[B,Cin,H,W] * W[Cout,Cin/groups,Kh,Kw] + b[Cout]`。**空間 3 つの attrs は `[H, W]` の
    長さ 2 の配列**（スカラ表記は受理しない — 同じ畳み込みに 2 通りの IR ができる。正規化は
    エクスポータ境界の仕事）。`groups` だけがスカラ。出力長は H / W それぞれ
    `floor((L + 2·padding − dilation·(K−1) − 1) / stride) + 1`。`groups` が `Cin` / `Cout` の
    両方を割り切ること・重みの第 2 軸が `Cin/groups` であること・**`Kh` と `Kw` の順**が契約
    （4 つの attrs は conv1d と同じく全て宣言必須で既定値補完をしない）
  - `conv_transpose1d`（f32、attrs `stride` / `padding`、**アリティ 3 固定**）—
    `x[B,Cin,L] ⊛ᵀ W[Cin,Cout,K] + b[Cout]`。**重みは conv1d と転置の `[Cin, Cout, K]`**。
    `stride ≥ 1` は MUST（0 はカーネルのゼロ除算・ハング）。受理するのは出力長がちょうど
    `L·stride` になる形、すなわち `2·padding == K − stride` が成立する形**だけ**で、一般形
    `(L−1)·stride − 2·padding + K` は fail loudly（需要が出たら本書の改訂で広げる）。
    `output_padding` / `dilation` / `groups` は attrs に欄が無い = 0 / 1 / 1 固定

  bias 無しの conv（実測は dec の `conv_post` 1 本）は**エクスポータがゼロ bias initializer を
  合成**してアリティ 3 へ正規化する — IR にも契約にもカーネルにも arity 分岐は無い。

出力 dtype は「**スロット 0 の入力 dtype → 出力 dtype**」の写像で決まる（既定は恒等）。
恒等でないのは比較 4 本（→ bool）・bool 入力の `sum`（→ i32）・`where`（bool → f32）だけで、
`cast` だけが例外的に attrs.to で決まる。

dtype ごとの受理集合（**入力スロット別**の受理集合と出力 dtype の導出を含む）・attrs の値域
（`cast` の丸め規約・`permute` の `dims`・融合 op の各 attr を含む）は契約テーブルが正本。

拡張順序は [op-vocabulary.md](op-vocabulary.md) の実装順序に従う。
