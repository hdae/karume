# IR v2 仕様

Karume のグラフフォーマット。ADR [0003](decisions/0003-ir-v1.md)（v1）と
ADR [0108](decisions/0108-container-format.md)（v2 — 格納の宣言をコンテナの束縛表へ外出し）の
具体化。プロトタイプ IR（v0）の実証済み構造を土台に、バージョニング・capability 宣言を加えた
非互換改訂。確定範囲を定義し、拡張は本書を改訂して行う。グラフを載せる**物理形式**（`krm` /
`krg`・block / part・codec 台帳）は [container-v1](container-v1.md) が正本で、本書はグラフ JSON
だけを定める。

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
  **アリティ 3 か 4・rank-4 head-first・attrs `scale` は半スケール**（q と k の両方に掛かる
  `√scale_factor`）。省略可能な第 4 入力は**加算 mask**（f32・`[1,1,M,N]` ちょうど・B·H へ
  broadcast — 2026-08-11 の改訂・ADR 0023 追記）。bool mask / `[B,1,M,N]` 等・causal /
  dropout は語彙に無く、エクスポータ境界で全件 fail loudly（GQA / MQA は 2026-08-17 の改訂
  〈ADR 0067 決定 1〉で整除 broadcast として受理 — 下の `attention` 項）。SDPA の保存は
  **ターゲット別**なので、既存の分解形 IR はそのまま有効。
- モデル拡充（2026-08-13）: `upsample_bilinear2d` を**第 1 層**として追加（ADR 0043 の
  判定手順 3 — Core ATen 内の原子。当時の暫定運用は 2026-08-14 の入場門モデル
  〈ADR [0059](decisions/0059-op-vocabulary-entry-doors.md)・現 Core ATen 層〉で解消）。attrs は `output_size` の 1 本だけで、
  **`align_corners = True` 以外は欄を持たない**。既存 IR への影響はゼロ（新しい op 名が
  増えるだけ）。
- モデル拡充（2026-08-13）: `deform_conv2d` を**第 1' 層**として追加（ADR
  [0055](decisions/0055-deform-conv2d.md) — Core ATen 外の原子で、要求元は BiRefNet 一族）。
  **アリティ 5**（x / weight / offset / mask / bias）— 固定アリティでは最多
  （従来は `where` / conv 族の 3 と `attention` の 3〜4）。
  attrs は `padding` の 1 本だけで、`stride` / `dilation` / `groups` / `offset_groups` は
  欄を持たない（= 1 固定）。既存 IR への影響はゼロ。
- モデル拡充（2026-08-14）: `gru_scan` / `gru_scan_reverse` を**第 2 層**〈現 拡張分子層 —
  ADR 0059〉として追加（ADR
  [0056](decisions/0056-gru-scan.md) — 分子だが「時間軸が実行時に決まる」意味情報が分解形で
  失われる。要求元は vowel-detector の 2 層 BiGRU）。**アリティ 4 固定・attrs 空**で、
  op が受け持つのは**隠れ側の逐次だけ**（入力側 GEMM は呼び手の `linear`）。走査方向は
  attrs ではなく **op 名**が持つ（`gelu` / `gelu_tanh` と同じ手筋）。既存 IR への影響はゼロ。
- autoregressive（2026-08-17）: `argmax` を **Core ATen 層**として追加（ADR
  [0068](decisions/0068-decode-exit-multi-output.md) 決定 2 — greedy デコードの出口。
  `torch.Tag.core` の実測で core・台帳 NOTE のみが要件〈ADR 0059〉だが、決定 2 は
  多出力解禁と同じ ADR に載っている）。**attrs 空・アリティ 1・入力 f32 → 出力 i32**（添字）で、
  **rank 保存**（最終次元を 1 に潰す）は shape 規則が持つ。既存 IR への影響はゼロ。
- autoregressive（2026-08-17）: `topk` を **Core ATen 層**として追加（ADR
  [0068](decisions/0068-decode-exit-multi-output.md) 決定 3 — top-k sampling の出口。
  `torch.Tag.core` の実測で core・core decomposition にも載らない）。**ノードレベル
  多出力の最初の入居者**で、`outs` が 2 本（値 f32 + 添字 i32）の IR が初めて実在する
  — スキーマは元から `outs` 長さ 1 以上を許可しているので**仕様は無改訂**（ADR 0068
  決定 1。0 本の解禁は別波）。**attrs `k` 宣言必須・アリティ 1**で、受理領域
  `1 ≤ k ≤ 最終次元`。既存 IR への影響はゼロ。
- autoregressive（2026-08-17）: **省略可能なトップレベル節 `states{}`** を追加（ADR
  [0066](decisions/0066-generation-context-state-slots.md) 決定 2 — 生成 1 本ぶんの可変 state を
  持つ名前付きスロット）。**節を持たないグラフは無風**（既存 IR への影響はゼロ・エクスポータは
  まだ出さない）。**ノードからスロットを参照する欄（`states` 欄 / `state_append`）はまだ無い** —
  この改訂で入るのは宣言と検査だけ（下の「state スロット」節）。
- autoregressive（2026-08-18）: **ノードの省略可能な `states` 欄**と effect op
  `state_append`、省略可能な attrs `window` を追加（ADR
  [0067](decisions/0067-autoregressive-attention-vocabulary.md) 決定 4 / 5 / 5b）。同時に 3 点が
  緩む / 締まる: ①`outs` の**長さ 0 を許可**（値を定義しない effect op — 許すのは契約が
  そう宣言する op だけで、執行点はパーサでなく契約テーブル）②記号の**束縛点が 2 つ**になる
  （入力 shape ∪ states shape — ADR 0066 追記 7。states 専用記号が値 shape に現れたら
  fail loudly）③**参照完全性**（宣言したスロットは 1 つ以上のノードから参照される MUST）。
  `states` 欄を持たないグラフ（= 既存の全モデル）の受理集合は 1 バイトも動かない。
- MTP（2026-09-08）: **借り物の宣言 3 種**を追加（ADR
  [0096](decisions/0096-speculative-decoding.md) 段 2 — drafter が target の KV と重みを読む形）:
  ①`states[].external`（省略可能・`true` のみ — 実体は借り先 context にあり自分では確保しない）
  ②`attention` の省略可能 attrs **`readonly`**（`true` のみ — states 形が **ins を q 1 本**に
  絞り、今 step の k/v を取らない past-only 形になる）③`initializers[].shared`（`tensor` の
  代わりに書く借り物宣言 — バイトを配布形に持たない）。3 つとも**欄を持たないグラフは無風**で、
  既存 IR への影響はゼロ。
- **v2**（2026-09-22・ADR [0108](decisions/0108-container-format.md) 決定 11 / 17）:
  **非互換改訂**。①`initializers[].storage` と `initializers[].tensor` を IR から外し、格納の
  宣言はコンテナの束縛表へ移す（`initializers[name]` は `{}` か `{ "shared": true }`）
  ②initializer 名を**実体の鍵**にする（エクスポータは FQN / `const.<hash>` で付ける）
  ③scale は rank 2 group 形に統一し `rowAxis` を宣言に出す（束縛表側 —
  [container-v1](container-v1.md) §6.1）④**正準直列化**の規則を定める（下の節 — `krg` の
  同一性を内容ハッシュで判定する条件）。`version` は `2`。v1 との両読みは作らない（旧資産の
  移行は container-v1 §12 の CLI）。

## コンテナ

- グラフ JSON はコンテナ（`krm` / `krg`）の**グラフ記述**に `graphs[<グラフ名>]` として載る
  （[container-v1](container-v1.md) §2.1）。1 コンテナに複数グラフを持てる。実体（重み・定数）は
  同じコンテナの block にあり、グラフとの対応は束縛表（§5）が持つ。
- JSON に NaN / Infinity リテラルを含めてはならない（エクスポータは `allow_nan=False`
  相当で書く。パーサは検出したら fail loudly）。

## グラフ JSON

```jsonc
{
  "format": "karume-ir",
  "version": 2,
  "requires": { "ops": ["matmul"] },
  "symbols": ["T"],
  "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 128] }],
  "outputs": ["h"],
  "initializers": {
    "enc.w": {}
  },
  "values": {
    "enc.w": { "dtype": "f32", "shape": [128, 64] },
    "h": { "dtype": "f32", "shape": ["T", 64] }
  },
  "nodes": [{ "op": "matmul", "ins": ["x", "enc.w"], "outs": ["h"], "attrs": {} }]
}
```

- `format` は固定文字列 `"karume-ir"`、`version` は整数 `2`。不一致は fail loudly。
- **未知のトップレベルキーは fail loudly**（未リリースにつき前方互換チャネルは持たない。
  必要になったら本書の改訂で導入する）。上の 9 キーは**全て必須**で、省略可能な節は
  **`states` の 1 本だけ**（下の「state スロット」）。
- `requires.ops` は nodes で実際に使われる op 名の集合と一致しなければならない（パーサが
  検証）。ランタイムは自分の対応表と突合し、非対応 op を**列挙して** fail loudly する。

## 正準直列化

グラフ JSON は**バイト列として一意**でなければならない（同じグラフを再 export するとグラフ記述が
バイト同一になる — ADR 0108 決定 4・container-v1 §13.5。`krg` の同一性を内容ハッシュで判定する
条件）。書き手（Python の exporter・移行 CLI・TS のテスト補助）は次の規則で直列化し、読み手は
`parse → 正準化 → serialize` が入力バイト列と一致することをテストで確かめる。

- **UTF-8・空白なし**（`,` と `:` の後に空白を置かない・改行を置かない）。
- **オブジェクトのキー順**: 固定スキーマのオブジェクト（トップレベル・`inputs[]` の要素・
  `values{}` の値・`states{}` の値・ノード）は**本書の例に現れる順**（トップレベルは `format` /
  `version` / `requires` / `symbols` / `inputs` / `outputs` / `initializers` / `values` /
  `states` / `nodes`、ノードは `op` / `ins` / `outs` / `attrs` / `states`、`states{}` の値は
  `dtype` / `shape` / `external`）。**名前をキーに持つ map**（`initializers` / `values` /
  `states` / ノードの `attrs` / ノードの `states`）は**キーの code point 順**（UTF-16 単位の順では
  ない — 非 BMP 文字で順序が変わる。名前は実測で全て ASCII なので両者は一致する）。
- **配列は宣言順のまま**（`inputs` / `outputs` / `nodes` / `ins` / `outs` / `shape` は順序が意味を
  持つ）。順序に意味の無い集合（`requires.ops` / `symbols`）は **code point 順**に並べる。
- **数値の綴りは ECMAScript の `Number::toString`**（ECMA-262 §6.1.6.1.20）に固定する。整数値は
  小数点も指数も付けず（`1`・`10000` — `10000.0` とは書かない）、非整数は最短往復桁で、
  指数表記になるのは `|x| < 1e-6` または `|x| ≥ 1e21` のときだけ（`1e-7`・`1e+21` — 指数の
  ゼロ詰めをしない・`0.000001` は指数にしない）。`-0` は `0` と書く。NaN / Infinity は書けない
  （値レベルで拒否 — 上の節）。JS 側は `JSON.stringify` がこの綴りそのものなので、**Python 側が
  これを実装する**（`repr(1e-6)` = `1e-06`・`repr(10000.0)` = `10000.0` は正準ではない）。
- **文字列のエスケープ**は JSON の最小形（`"` / `\` / 制御文字 U+0000〜U+001F だけをエスケープし、
  `\b` `\f` `\n` `\r` `\t` は短縮形、他は `\u00XX` の小文字 hex。非 ASCII は生の UTF-8）。
  JS `JSON.stringify` と Python `json.dumps(ensure_ascii=False)` の共通部分で、両者は一致する。
- **省略可能な節**（`states` / ノードの `states` / `states[].external`）は**空・偽なら書かない**
  （欄の不存在がそのまま宣言 — 常に出すと同じグラフが 2 通りの綴りを持つ）。

読み手が空白や `\u` エスケープを含む**非正準の JSON を拒否するかどうか**は読み手の契約に属する
（TS の `parseIrDeclaration` は JSON として受理し、正準性は書き手側の規則と往復テストで守る）。

## 値と型

- **意味論 dtype**（`inputs[].dtype` / `values[].dtype`）: `"f32" | "i32" | "bool"`。
  計算は常にこの型で行う（f16/bf16/i8 は意味論 dtype に存在しない）。
  実行時格納は i32 = 32bit、bool = **u32 の 0 / 1**（ストレージバッファに 1bit 型が無いため
  — ADR [0009](decisions/0009-dtype-i32-bool.md)）。要素は 3 型とも 4 バイト。
- **op ごとに実行できる意味論 dtype は違う**（一括ではない）。受理集合の正本は契約テーブル
  （`packages/runtime/src/ops.ts` / `karume/ops.py`）で、ランタイムは非対応を列挙して fail loudly。
  グラフ入力の転送は 3 型とも可能。torch 既定の整数 i64 は **エクスポータ境界で i32 へ
  正規化**する（値域外は fail loudly）— IR に i64 は無い。
- **格納（codec）は IR に書かない**。initializer の実体の格納形はコンテナの**束縛表**
  （[container-v1](container-v1.md) §5 — `binding[<グラフ名>][<initializer 名>].encoding`）が
  持ち、IR が持つのは意味論 dtype と宣言 shape だけである（「1 アーキ・1 量子化方式・1 グラフ」—
  ADR [0108](decisions/0108-container-format.md) 決定 17）。codec の語彙は台帳（container-v1
  §6.3）: `f32` / `f16` / `bf16` / `i32` と量子化 4 種 `int8-sym` / `int4-sym-g` / `int2-off` /
  `ternary`。`i32` 以外は**意味論 f32 の符号化**で、`i32` だけが**生の int32**（記号依存定数の
  焼き込み先 — ADR [0010](decisions/0010-symbolic-constant-folding.md)。「格納語彙は f32 の
  符号化」の明示的な例外）。**ランタイムが実行できるのは bf16 以外** — bf16 だけが「宣言としては
  valid、実行は fail loudly（capability 不足の診断付き）」。
- **合流層**（container-v1 §13.3）: グラフと束縛表を**重み取得前に合流**し、合流後表現に対して
  格納の規則を掛ける（置き場は runtime `format/` の 1 箇所 — Python 側も同じ規則を鏡像で持ち、
  二重実装しない）:
  - 量子化 codec は `encoding.scale` が**必須**（既定 1.0 で補完しない — 書き忘れが
    「量子化前の 1/127 倍の重み」で静かに走るのを塞ぐ・ADR 0019 / 0069）
  - `int4-sym-g` の `groupSize` は **2 冪かつ 16 以上**で、行長（`numel / shape[rowAxis]`）を
    割り切る MUST（端数 group を作らない制約が、行境界・group 境界のバイト整列を保証する —
    ADR 0069 決定 2）。`int8-sym` / `int2-off` / `ternary` は per-channel（`groupSize` = 行長）
  - scale の形は **rank 2 group 形 `[shape[rowAxis], 行長 / groupSize]`** に統一する
    （container-v1 §6.1 — v1 の i8 が使っていた keepdim broadcast 形は廃止）。`rowAxis` は
    宣言に出る（`conv_transpose1d` の重み `[Cin,Cout,K]` だけ 1・他は 0）
  - 意味論と格納の組は**次の 2 通りだけ**（交差は fail loudly — i32 宣言が f16 のビット列として
    読まれる沈黙誤値を塞ぐ）: 意味論 `f32` × `f32` / `f16` / `bf16` / 量子化 4 種、
    意味論 `i32` × `i32`
- **適格判定と 2 経路**（ADR [0018](decisions/0018-f16-weight-execution.md) /
  [0019](decisions/0019-i8-weight-execution.md) / [0069](decisions/0069-packed-w4-storage.md)）:
  意味論はあくまで f32（「格納のみ量子化・計算は f32」— ADR 0006）で、圧縮格納の initializer は
  **適格判定**で経路が 2 つに分かれる。
  - **適格**（その initializer の消費が `linear` / `conv1d` / `conv2d` / `conv_transpose1d` /
    `embedding` の**重みスロットだけ**）: 圧縮 payload のまま GPU 常駐し、dequant はカーネル内。
    codec ごとの適格 op は台帳の `executableOps`（container-v1 §6.3）が正本 — `int4-sym-g` は
    linear / embedding / conv1d（`groups == 1`）、`int2-off` / `ternary` は linear / embedding。
  - **適格外**（bias / norm 系の weight / その他の op / 重みスロットと他スロットの混在消費 /
    消費ゼロ / `graph.outputs` に載った initializer）: **ロード時に CPU で f32 へ展開**する。
    正しさは保たれるが VRAM 削減はゼロで、縮むのは配信サイズだけ。bias が適格にならないのは
    ADR 0006 の「bias は常に f32」規則そのもの（低精度適格判定に bias を含めない）。
  - 内訳は `Session.diagnostics().storage`（GPU 常駐圧縮バイト数 / CPU 展開バイト数）で
    取得できる — 「圧縮指定なのに適格 0MB」を沈黙させないための常設診断（ADR 0006）。
- **`f16`**: GPU 常駐時は `unpack2x16float` で 2 要素/語を展開する（添字 `i` の値は
  `unpack2x16float(w[i / 2])[i % 2]`）。VRAM は f32 比 ≈ 1/2。要素数が奇数のときの末尾詰め物は
  **書き手が block に焼く**（container-v1 §4.1 — 読み手側の整列分岐は無い）。
- **`int8-sym`**（ADR 0019）: **per-channel symmetric int8**（zero-point なし・`q ∈ [−127, 127]`）。
  GPU 常駐時は **4 要素を 1 語（u32）へリトルエンディアン順**に詰めた並びとして `array<u32>` で
  束縛し、`unpack4xI8` で展開する（語 = `w[i / 4]`、レーン = `i % 4` を**平坦添字**から割り出す）。
  **scale の適用位置は「要素ごと」**（読み出し時 dequant — `out = Σ x·(q·s) + bias`）。縮約の外で
  掛ける形（`(Σ x·q)·s`）は乗算が減るが、CPU 展開とのビット一致を失うので採らない。この形の
  おかげで**適格経路（GPU）と適格外経路（CPU 展開）はビット単位で同じ値**を出す（`q·s` の f32
  丸めが両側とも 1 回）。VRAM は f32 比 ≈ 1/4（+ scale のオーバヘッド 1% 未満）。診断の
  `residentCompressedBytes` には**scale バッファのバイト数も加算**する。
- **`int4-sym-g`**（ADR 0069 決定 2 / 3）: 行方向 group の対称量子化を packed 4bit で持つ
  （`q ∈ [−7, 7]`・`u = q + 8`・バイト内で要素 `2i` = 下位 nibble）。宣言 shape は**論理形の
  まま**で、payload は `numel / 2` バイト（要素数が奇数の宣言は bit 総量が byte 境界に乗らないので
  fail loudly）。展開カーネルは `array<u32>` で束縛する。
- **`int2-off`**（ADR [0097](decisions/0097-gemma4-qat-integration.md#追記-1--int2-格納と実行の契約2026-09-11)）:
  論理形は正整数の rank 2 `[N,K]`、K は 16 の倍数。1 バイトに 4 要素を下位 2bit から `u = q + 2`
  で詰め、整数 `[−2, 1]` を保持する。復元は要素ごとの `fround(q·scale)`。linear の計算は f32
  のみで、a8 / f16 は明示拒否する。**`ternary`** は値域 `{−1, 0, +1}` を主張する別名 codec で、
  詰め方・復元・カーネルは `int2-off` と同一（container-v1 §6.3）。
- **initializer 名は実体の鍵**である。v1 の `initializers[].tensor`（safetensors のテンソル
  キー）は無く、束縛表が `(グラフ名, initializer 名)` で実体を指す（container-v1 §5）。
  `initializers[name]` の値は **`{}`**（実体を持つ）か **`{ "shared": true }`**（下）の 2 形だけで、
  他のキーは未知キーとして fail loudly。エクスポータは名前を**パラメータの FQN**
  （`model.layers.0.mlp.gate_proj.weight`）と **`const.<sha256 先頭 16 hex>`**（定数 — 内容で
  重複除去）で付ける。上流 checkpoint の取り込み（ADR 0108 決定 19）と LoRA の対象解決は、
  この名前を鍵にする。宣言 shape / 意味論 dtype と実体は束縛の突合で一致しなければならない
  （ロード時検証・不足も余剰も全件列挙で拒否）。
- **共有 initializer（`shared`）**（ADR [0096](decisions/0096-speculative-decoding.md) 段 2 §1.3）:
  `{ "shared": true }` と書くと、バイトをコンテナに持たない宣言になる（実体は貸し手 Session が
  既に GPU へ載せた重み）。

  ```jsonc
  "initializers": {
    "model.lm_head.weight": { "shared": true }
  }
  ```

  - **名前は貸し手グラフの initializer 名と同じ** MUST。借り手はこの名前で
    `SessionOptions.sharedWeights` から実体を受け取り、models 側は同名で貸し手の
    `exportWeight` を引く（v1 の `shared.tensor` は要らない — 名前が鍵なので）。
  - `values{}` の dtype / shape 宣言は従来どおり**必須**。格納（codec / scale / group）は
    **書かない** — 貸し手の常駐重みが正本で、写すと同じ事実が 2 箇所に生える。
  - 束縛表の突合集合から**外れる**（container-v1 §5 の借用形 `{ "shared": true }` と対）。実体の
    素性はロード側の門が見る（同一 device・宣言 shape・消費席・チャネル軸 — 貸し手の codec が
    借り手の消費 op で実行できること）。
- **宣言完全性**: 全ての値（inputs・initializers・全ノード出力）はちょうど 1 箇所で宣言される
  — inputs は `inputs[]` で、**initializer と中間値・出力は `values{}` で**。実行時に毎ノード、
  宣言 shape/dtype と照合する。
- initializer の宣言は**数値次元のみ**（記号次元不可）。意味論 dtype は **f32 / i32**
  （格納との組は上の合流層の規則）。bool の initializer は語彙に無い（実測に無く、1 バイト格納は
  4 バイト前提の転送と噛み合わない）。
- 非有限数の拒否はリテラルだけでなく**値レベル**で行う（`1e999` は JSON として構文有効だが
  Infinity に丸まるため、パース時 reviver で拒否）。descriptor 側（束縛表・block 目次）の規則は
  container-v1 §0 が持つ — こちらも未知キーは fail loudly。

## shape と次元言語

- shape 要素は「非負整数」または「次元式文字列」。
- 次元式の正準文法: **`coeff·sym + offset` の一次式・1 次元 1 シンボル**。
  文字列表現は `"T"` / `"8T"` / `"T+8"` / `"8T+2"`（coeff は 2 以上のとき前置、
  offset は 1 以上のとき `+` 後置。`"1T"` や `"+0"` は非正準として拒否。負の係数・
  負のオフセット・複数シンボルは拒否）。
- シンボル名は `[A-Za-z_][A-Za-z0-9_]*`。`symbols` に列挙され、かつ**少なくとも 1 つの
  入力 shape または states shape の次元位置に出現**しなければならない（束縛は shape の次元位置
  から直接取る。要素数からの逆算はしない）。束縛点が 2 つに分かれる規則と、states 専用記号を
  `values{}` に書けない制約は後述の「state スロット（`states{}`）」節が正本 — 通常値の解決に
  効くのは入力由来の束縛だけ。出現は**派生形でよい** — `"2T"` や `"T+8"` の実寸からは
  `(size − offset) / coeff` で解が一意に決まり、割り切れない実寸は fail loudly
  （[ADR 0057](decisions/0057-derived-dim-binding.md)。母音検出 CRNN は先頭 conv の stride 2
  のせいで `2T` でしか長さ軸を宣言できない）。
- 文法の正本は 1 箇所: 適合ケース表 `packages/runtime/tests/fixtures/dim-grammar.json`（valid / invalid /
  束縛評価の 3 節）。TS 実装と（M1 の）Python 実装は同じ表で検証する。

## state スロット（`states{}`）

生成 1 本ぶんの可変 state（自己回帰の KV など）を置く**名前付きスロット**の宣言
（ADR [0066](decisions/0066-generation-context-state-slots.md) 決定 2）。**省略可能な節**で、
持たないグラフ（= 既存の全モデル）の受理集合は一切変わらない。

```jsonc
{
  // …（他のトップレベル節は上と同じ）
  "symbols": ["T"],
  "states": {
    "layer0.k": { "dtype": "f32", "shape": [1, 2, 512, 128] },
    "layer0.v": { "dtype": "f32", "shape": [1, 2, 512, 128] }
  }
}
```

- 形は `name → { dtype, shape }` の 2 キー + **省略可能な `external`**（未知キーは fail loudly）。
  空オブジェクトも節の省略も同義（= スロット 0 本）。スロット名は**空文字列でない**（参照側の欄が
  受理しない名前を宣言できると、原理的に参照不能なスロットになる）。
- **`external`**（ADR [0096](decisions/0096-speculative-decoding.md) 段 2 §1.1・**`true` のみ**・
  欄の不存在 = 自分で確保する）: 実体を**借り先の context** が持つスロット。`false` は書けない
  （欄の不存在と同義の綴りを 2 つ持たない）。規律は 4 点で、どれも破れは例外ではなく
  「未初期化の過去を読む」「貸し手が確定した KV を上書きする」という別の値として出る:
  - `state_append` は **0 本**（借り物へ書けるのは貸し手だけ）
  - 読者は **`attrs.readonly` の attention だけ**（今 step の k/v を足す形は自分で書いた行を
    読む前提なので、書き手の居ないスロットでは必ず未初期化行を読む）
  - `readonly` の読者が参照するスロットは **external だけ**（上の対）
  - external が 1 本でもあれば**全スロットが external**（借り手 context は自前スロットを持たない）
- `shape` は**固定 rank の容量込み具体形**: rank は **1..4**（ADR 0066 決定 2 の「固定 rank
  （rank ≤ 4）」）、数値次元は**正整数**（`values` の非負とは違う — 容量 0 のスロットは実体を
  持てない）。次元式は値と同じ次元言語で、`symbols` に宣言済みの記号なら使える。**記号の
  束縛点は 2 つ**（ADR 0066 追記 7）: 入力 shape の次元位置（従来どおり — 値 shape の解決に
  効くのはこちらだけ）と、**states の shape に現れる記号は
  `createGenerationContext(spec.bindings)`**（KV 容量のように context 生成時に決める値）。
  束縛可能性検査は「入力 shape **∪** states shape のどこかに現れる」で、**states 専用記号
  （states にしか現れない記号）が `values{}` の宣言 shape に現れたら fail loudly** — 通常値の
  解決に効くのは入力由来の束縛だけなので、現れれば実行時に必ず束縛不能になる（宣言の時点で
  落とす）。実行時も同じ分担: states 専用記号は `run` / `enqueue` の bindings では
  **要求されず、渡したら fail loudly**（束縛点は `createGenerationContext` の 1 箇所 —
  context が持つ容量との二重簿記を作らない）。
- `dtype` は **`f32` のみ**。`f16` は**席の予約だけ**があり（ADR 0066 追記 5 — state 格納の
  低精度化は数値契約が変わるので ADR [0058](decisions/0058-numerics-opt-in-contract.md) 流儀の
  opt-in が要る）、宣言されたら「**未対応**」として fail loudly する（「語彙外」とは別の診断）。
  それ以外（`i32` / `bool` / 未知の名前）は語彙外。
- **スロットは値ではない**: `inputs` / `values` に宣言を持たず、`ins` / `outs` からも参照されない
  （state をグラフの通常 input / output にしないのが ADR 0066 決定 1 の MUST）。実体の確保・寿命・
  論理長は GenerationContext が持つ。
- 名前空間は値と別だが、**値名（inputs / initializers / ノード出力 / `values` の宣言）と同名の
  スロットは拒否する** — 別名前空間の同名は「スロット名の欄に値名を書いた / その逆」を検出
  できなくするだけで表現力を足さない（束縛表が scale の block を他 initializer の実体と衝突させ
  ない規則①〈container-v1 §5〉と同じ流儀）。
- **「層 × 均一 KV」の前提は無い**: スロットは「KV」を知らない汎用の器で、sliding 層と full 層は
  容量の違う別スロット、KV 共有層は**同一スロット名の参照**で表す（層数・レイアウトの欄は
  作らない）。
- **参照完全性**: 宣言したスロットは**少なくとも 1 つのノードの `states` 欄から参照される**
  MUST（`values` の孤立宣言と同じ穴 — 誰も読まないスロットは GenerationContext が確保だけして
  黙って容量を食う）。読者だけ / 書き手だけのスロットはどちらも正規（KV 共有層は
  `state_append` を持たない — ADR 0067 決定 5）。
- ノードからの参照は下の「state 参照ノード」節（`attention` の states 形 / `state_append`）。
  スロットの物理形（`[B, Hkv, C, D]`）と `window ≤ C` はそちらが持つ。

## ノード

- SSA・単一代入: 各値名は inputs / initializers / いずれかのノードの `outs` のうち
  **ちょうど 1 箇所**で定義される。`ins` は定義済みの名前のみ参照。`outputs` は定義済み
  名の部分集合。`outs` の本数は **op 契約が決める**（**複数出力**は `topk` の 2 本〈値 f32 +
  添字 i32・ADR [0068](decisions/0068-decode-exit-multi-output.md) 決定 3〉、**0 本**は
  `state_append`〈値を定義しない effect op・ADR
  [0067](decisions/0067-autoregressive-attention-vocabulary.md) 決定 5〉で、他は全て単一出力）。
  **0 本を許すのは契約が effect を宣言する op だけ** — パーサは本数に意味を与えず、執行点は
  契約テーブルの出力数突合（他の op に `outs: []` を書くと本数不一致で落ちる）。
- `nodes` は**トポロジカル順**で格納される（パーサが検証。前方参照は fail loudly）。
- `attrs` は op ごとの契約テーブルで検証する。未知の attr・契約外の値は fail loudly
  （近似実行しない）。**省略可能な attrs は `window` と `readonly` の 2 本だけ**（どちらも
  states 形専用 — 下の「state 参照ノード」）。欄の不存在それ自体が別の宣言（`window` 不在 =
  全 context / `readonly` 不在 = 今 step の k/v も読む形）になる欄で、他の attrs は従来どおり
  「宣言済みキーは全て必須・既定値補完なし」。
- **`states` 欄**（省略可能・下の「state 参照ノード」）: `ins` / `outs` と別の欄で state
  スロットを名前参照する。キーは op 契約が固定し、値は `states{}` で宣言済みのスロット名
  （未宣言参照は fail loudly）。

## state 参照ノード（`attention` の states 形 / `state_append`）

state スロットを読み書きするノードの契約（ADR
[0067](decisions/0067-autoregressive-attention-vocabulary.md) 決定 4 / 5 / 5b）。

```jsonc
{
  "nodes": [
    {
      "op": "attention",
      "ins": ["q", "k_cur", "v_cur"],
      "outs": ["o"],
      "attrs": { "scale": 0.088, "window": 512 },
      "states": { "k": "layer0.k", "v": "layer0.v" }
    },
    {
      "op": "state_append",
      "ins": ["k_cur"],
      "outs": [],
      "attrs": { "window": 512 },
      "states": { "slot": "layer0.k" }
    },
    {
      "op": "state_append",
      "ins": ["v_cur"],
      "outs": [],
      "attrs": { "window": 512 },
      "states": { "slot": "layer0.v" }
    }
  ]
}
```

- **欄の有無が形を判別する**: `states` 欄を持たない `attention` は従来契約そのまま。欄が
  `{ k, v }`（**キーはちょうどこの 2 つ**・k と v に**同じスロット名は書けない**）のとき
  states 形になり、`ins` の k / v は**今 step の chunk**（`[B,Hkv,M,D]`・M は q と共有）、
  過去分はスロットが持つ。states 形は **mask 入力を取らない**（causal 固定 — 述語で表し
  mask tensor は実体化しない）。
- `state_append`（f32・**アリティ 1**・**出力 0 本**・`states` 欄は `{ slot }` ちょうど・
  必須 attrs 無し）— 今 step の k / v をスロットへ書く単機能 effect op。入力は
  `[B,Hkv,M,D]`。KV 共有層は「このノードが無い」だけで表せる（ORT の kv_empty と同じ表現力を
  op の不在で得る）。
- **スロットの物理形は `[B, Hkv, C, D]` 固定**（C = 容量）。参照するノードの `ins` と
  **B / Hkv / D が一致**し、`attention` の k / v スロットは**互いに同形**でなければならない。
- **readonly 形**（ADR [0096](decisions/0096-speculative-decoding.md) 段 2 §1.2）: states 形の
  attention が省略可能 attrs **`readonly`**（`true` のみ）を宣言すると、`ins` は **q 1 本
  ちょうど**（今 step の k / v を取らない past-only 形）になる。

  ```jsonc
  {
    "op": "attention",
    "ins": ["q"],
    "outs": ["o"],
    "attrs": { "scale": 1.0, "window": 512, "readonly": true },
    "states": { "k": "l13.k", "v": "l13.v" }
  }
  ```

  - q は `[B, H, M, D]` で **M = 1 MUST**（読むのは論理位置 P−1 の 1 行だけ — M > 1 は
    2 行目以降が同じ列範囲を読む沈黙誤値になる）。スロット `[B, Hkv, C, D]` とは
    **B / D が一致**・`H % Hkv == 0` かつ `H ≥ Hkv`・sliding なら `window ≤ C`。出力は
    `[B, H, 1, D]`（q と同形）。
  - 意味論: 列 `[column_base, P)` への attention（`column_base` = sliding なら `P − min(P, W)` /
    full なら `0`）。今 step の `ins` は**無い**ので、live 列数は `min(P, W)` / `P`。
    softmax の数値契約（半スケール・−inf identity・②統計）は states 形と同じで、**P = 0
    （列 0 本）は空行 → 出力 0**。
  - `readonly` は **states 欄を持つノードでのみ**宣言でき、参照先は **external スロットだけ**
    （上の「state スロット」節）。`state_append` には書けない。
- 省略可能 attrs **`window`**（正の整数・欄の不存在 = 全 context）= sliding window の幅。
  **`window ≤ C`** MUST。**同一スロットに触れる全ノードで存在有無も値も一致**する MUST
  （論理 col → 物理 row の写像は読み書き同式 — 読み側だけ別式にすると沈黙誤読になる）。
  `window` は **states 欄を持つノードでのみ宣言できる**（従来形に書くと「受理はされるが
  誰も読まない attr」になる）。
- **順序は `nodes` 配列順**（決定 5b）: state 参照はテンソルのデータ辺を張らないので
  トポロジ順では決まらない。①`state_append` は 1 スロットにつき 1 本まで ②append があるなら
  **そのスロットに触れる最後のノード**（append より後に読者を置けない）。エクスポータは
  「全読者 → append」の順で発行する。
- 論理長（`pastLength` / `queryLength`）は attrs にも shape にも載らない — 毎 step 変わる
  実行時スカラで、GenerationContext 所有の可変 uniform が運ぶ（ADR 0066 決定 3・追記 4）。
  full スロットの `pastLength + queryLength ≤ C` は実行時検査（context 側）。

## op セット（契約は実装の契約テーブルが正本、ここは一覧のみ）

> 実装は 2 つある（TS `packages/runtime/src/ops.ts` とエクスポータ `karume/ops.py` +
> `karume/shapes.py`）。両者が同じ契約を持つことの正本は適合ケース表
> `packages/runtime/tests/fixtures/op-contracts.json` で、op 名の全集合 / アリティ / スロット dtype /
> attrs キー集合と値域 / 出力 shape 規則（rank 上限を含む）を両側のテストが同じ表へ
> 突き合わせる（次元文法と `dim-grammar.json` の関係と同じ）。

- unary elementwise: `neg, abs, exp, log, log1p, sqrt, sin, tanh, sigmoid, relu, gelu, gelu_tanh`
  （f32）/ `bitwise_not`（bool）
  - `gelu_tanh` は torch の `gelu(approximate="tanh")`。erf 形の `gelu` と数値が違うため
    **別 op** で運ぶ（attrs 空の契約に近似種別の欄は無い — ADR
    [0043](decisions/0043-op-addition-layers.md) 初適用・現行の門 = Core ATen 層の
    attr 変種〈ADR 0059〉）
  - `sin` は語彙で**唯一の三角関数**（Core ATen 層〈旧第 1 層〉）。RoPE のような定数表は
    エクスポータが initializer へ畳むので、語彙に要るのは実行時値を取る形だけ
    （初出は DACVAE の Snake 活性）。`cos` は実測に出るまで足さない
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
- `argmax`（f32 → **i32**、**attrs 無し** — ADR
  [0068](decisions/0068-decode-exit-multi-output.md) 決定 2）— 最終次元の argmax で、出力は
  **rank 保存**（最終次元を 1 に潰した固定形）。reduce 族とは別の契約: 軸は最終次元固定
  （`dim` の欄が無い）・`keepdim` の欄も無い（rank が下がる形は語彙に無い）・長さ 0 の最終
  次元は拒否。固定挙動は 3 点とも torch 準拠で、**タイブレークは最小 index** / **NaN は最大**
  （複数なら最小 index）/ **全 −inf 行は index 0**（行 max の identity は −inf MUST — 有限
  sentinel だと番兵 index が出力へ漏れる）。llama.cpp は GPU 側 = 最大 index・CPU sampler =
  最小 index で同一リポ内でも食い違っており、明文化しないと greedy の再現性が実装差で割れる
- `topk`（f32 → **値 f32 + 添字 i32 の 2 出力**、attrs `k` — ADR
  [0068](decisions/0068-decode-exit-multi-output.md) 決定 3）— 最終次元の top-k で、出力は
  2 本とも `[…, k]`（rank 保存）。**唯一の多出力 op**。`k` は**計画時定数**（attrs 宣言必須 =
  static-k。torch の schema は SymInt だが実行時に決まる k は静的形状の前提に載らない）で、
  受理領域は **`1 ≤ k ≤ 最終次元`** — k=0（torch は受理する）・k > 最終次元・記号 k は
  全て fail loudly。軸は最終次元固定（`dim` の欄が無い）・`largest` / `sorted` の欄も無い
  （**降順ソート済みの最大側**の 1 形だけが語彙）。固定挙動は **タイブレーク = 最小 index** /
  **NaN は最大**（複数なら最小 index）/ 全 −inf 行も最小 index から k 本。**値の列は torch と
  ビット一致**する一方、**添字の列は torch の未規定部分を karume が規定した**側（torch は
  同値要素の順序を保証せず、`topk([5,5,5,5],1)` = 2 に対し `argmax` = 0 で自己矛盾している —
  実測 2026-08-17。k=1 は karume では argmax と一致する）。実装は「レーン局所 top-k →
  トーナメント merge」で全語彙 argsort を経由せず、**k の実装上限**が workgroup storage の
  device limit から静的に決まる（WebGPU 既定 16384 バイトで **k ≤ 63**。超過は縮退させず
  上限値つきで `CodegenError`）
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
  - `cat`（f32、attrs `dim`）— **唯一の可変アリティ op**（入力 2 本以上）で、連結軸以外の
    次元は全入力で一致、出力の軸長は入力の軸長の総和。連結軸は〈定数〉または〈**同一**
    シンボルの一次式〉で、総和が次元言語 `coeff·sym+offset` に載るときに限り記号でよい
    （`S`+1519 → `S+1519`、`S`+`S` → `2S` — ADR
    [0046](decisions/0046-cat-symbolic-axis.md)）。異なるシンボルの混在は表現が無い。実行は
    strided **書き**コピー族で、入力ごとに出力の部分領域へ書く（全入力で出力全域を覆う）
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
  エクスポータの畳み込みパス（`_fold_rms_norm`）が 1 ノードへ合成する（ADR 0016 / 0017）。
  **融合 `attention`（perf-a）は既定の 11 本に入らない**: SDPA の保存はグローバルに掛けると
  契約外のマスクを持つ形まで拾って export できなくなるため、**ターゲット別の opt-in**にして
  ある（ADR 0023 追記）:
  - `linear`（f32、attrs 無し、**アリティ 3 固定**）— `x[…,in] × W[out,in] + b[out]`。
    重みは `[out, in]` の転置レイアウトのまま。bias 無しの形は語彙に無い
  - `layer_norm`（f32、attrs `normalized_shape` / `eps`、**アリティ 3 固定**）— 最終次元のみ
    （`normalized_shape` は長さ 1）、affine 常時あり。分散は**母分散（correction = 0）**、
    `eps` は有限の正数
  - `attention`（f32、attrs `scale`、**アリティ 3 か 4**）— `out = softmax_lastdim((q·scale) @
    (k·scale)ᵀ + mask) @ v`（ADR [0023](decisions/0023-fused-attention.md)）。入力は
    **rank-4 head-first**（`q[B,H,M,D]` / `k[B,Hkv,N,D]` / `v[B,Hkv,N,D]`・連続）で出力は
    `[B,H,M,D]`。**H と Hkv は整除 broadcast**（`H % Hkv == 0` かつ `H ≥ Hkv ≥ 1` — GQA / MQA・
    ADR [0067](decisions/0067-autoregressive-attention-vocabulary.md) 決定 1。
    `r = H / Hkv` は導出値で attrs 欄を持たない。B は完全一致・k/v 間の Hkv も完全一致）。
    **D は 3 者とも同じ**（v 側だけ別の長さを許すと「D を取り違えた IR」が
    shape 検査を素通りする）。**`scale` は半スケール** = q と k の**両方**に掛かる
    `√scale_factor`（torch の `_scaled_dot_product_attention_math` と同じ形。内積の後に
    1 度だけ掛ける形へ変えると丸め列が変わり、分解経路とのビット同一が失われる）。
    省略可能な第 4 入力は**加算 mask**（f32・rank-4・shape はちょうど `[1,1,M,N]`）で、
    B·H の全バッチへ broadcast する。`[B,1,M,N]` / `[1,H,M,N]` / bool / rank≠4 は受理せず、
    causal / dropout は語彙に無い。行が全て −inf になるマスクは**契約違反**（NaN 汚染 —
    検査は入れない。その形が正規なのは `safe_softmax` を使う分解経路だけ。ADR
    [0044](decisions/0044-runtime-attention-mask.md) 決定 3）。
    省略可能な **`states` 欄**を持つと autoregressive の states 形になる（上の
    「state 参照ノード」節 — 同一 op 名の契約拡張で、欄の有無が形を判別する）。
    states 形がさらに省略可能 attrs **`readonly`** を宣言すると **アリティ 1**（q だけの
    past-only 形 — ADR [0096](decisions/0096-speculative-decoding.md) 段 2 §1.2）になる
  - `state_append`（f32、**アリティ 1・出力 0 本**、省略可能 attrs `window`、`states` 欄
    `{ slot }` 必須 — ADR
    [0067](decisions/0067-autoregressive-attention-vocabulary.md) 決定 5）— 今 step の k / v を
    state スロットへ書く effect op（上の「state 参照ノード」節）。**値を定義しない唯一の op**で、
    torch から出す経路は持たない（aten に対応物が無く、発行するのは decode グラフ台本だけ）
  - `rms_norm`（f32、attrs `eps`、**アリティ 2 固定**）—
    `y = x · rsqrt(mean(x², 最終次元) + eps) · weight`。**bias が無く、平均も引かない**
    （layer_norm との差はこの 2 点）。正規化長の正本は **weight の長さ**で、
    `normalized_shape` の欄は持たない（同じ事実を attrs と入力で二重に持たない）。
    `weight` は最終次元長の rank1・`eps` は有限の正数。weight 無しの形はエクスポータが
    ones を合成してアリティ 2 へ正規化する
  - `softmax`（f32、attrs `dim`）— **最終次元のみ**（`dim` は非負表記）。**safe-softmax**
    （行の最大値を引く）で計算する
  - `safe_softmax`（f32、attrs `dim`）— `softmax` と**同一契約** + 「**行 max が −inf の行は
    全 0**」（ADR [0044](decisions/0044-runtime-attention-mask.md) — 拡張分子層
    〈旧第 2 層〉）。torch の SDPA
    分解が `softmax` に被せる safe-softmax ガードと同じ意味論で、マスクが実行時値でガードの
    不活性を証明できない形だけに現れる。有限要素を持つ行の値は `softmax` と**ビット同一**
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

- **変形畳み込み**（拡張原子層〈旧第 1' 層〉の原子 — `torchvision::deform_conv2d`。ADR
  [0055](decisions/0055-deform-conv2d.md)・門の定義は
  [0059](decisions/0059-op-vocabulary-entry-doors.md)）:
  - `deform_conv2d`（f32、attrs `padding`、**アリティ 5 固定**）—
    `x[B,Cin,H,W]` / `W[Cout,Cin,Kh,Kw]` / `offset[B,2·Kh·Kw,Hout,Wout]` /
    `mask[B,Kh·Kw,Hout,Wout]` / `b[Cout]` → `[B,Cout,Hout,Wout]` の DCNv2。
    出力空間は `Hout = H + 2·padH − (Kh−1)` / `Wout = W + 2·padW − (Kw−1)` で、`offset` /
    `mask` の空間 2 軸はこれと**一致することが契約**（出力形の正本は x + weight + attrs 側）。
    サンプル座標は `y = (oy − padH) + kh + off_y` / `x = (ox − padW) + kw + off_x` で、
    **offset のチャネル並びは `(kh, kw)` の入れ子・最内が偶数 = y / 奇数 = x**。
    `mask`（modulator）は**双線形補間の後**に掛かり、境界外は **border clamp ではなく
    ゼロ埋め**（中心が `(−1, in)` の外ならタップ全体 0・内側でも範囲外の隅はその隅だけ 0）。
    `stride` / `dilation` / `groups` / `offset_groups` は attrs に**欄が無い = 1 固定**、
    `mask` はスロットとして必須 = **DCNv2 専業**（`use_mask=False` の DCNv1 は表現を持たない
    = エクスポータ境界で fail loudly）。**offset の NaN は 0 に落とさず出力へ伝播する**
    （範囲外の 0 とは別扱い — ADR 0055 決定 5）

- **空間 resample**（Core ATen 層〈旧第 1 層〉の原子 — `aten.upsample_bilinear2d.vec`）:
  - `upsample_bilinear2d`（f32、attrs `output_size`、**アリティ 1 固定**）—
    `x[B,C,H,W] → [B,C,Hout,Wout]` の双線形補間。**`align_corners = True` 専業**で、
    `align_corners` / `mode` / `scale_factor` は attrs に**欄が無い**（`False` も nearest /
    bicubic / area / antialias も表現を持たない = エクスポータ境界で fail loudly。`False` の
    需要が出たら `gelu` / `gelu_tanh` と同じ手筋で別 op として足す）。`output_size` は conv2d と
    同じ `[Hout, Wout]` の長さ 2 の配列で、宣言必須・既定値補完なし。
    源座標は `((in−1)/(out−1)) · 出力添字`（出力長 1 の軸は倍率 0）、タップは
    `index0 = trunc(源座標)` と `index1 = index0 + (index0 < in−1 ? 1 : 0)`、重みは
    `λ1 = 源座標 − index0` / `λ0 = 1 − λ1` で、**式木は H が外・W が内**の入れ子
    （torch の `aten/src/ATen/native/UpSample.h` と同じ順）。倍率は非整数でよく、
    **縮小（`Hout < H`）も同じ op**（2 タップしか読まないのは torch と同じ仕様で `area` とは
    別物）。空間軸の長さ 0 は受理しない

- **固定活性量子化**（ADR [0097](decisions/0097-gemma4-qat-integration.md#追記-2--固定-srq-op-の契約2026-09-11)）:
  - `static_quantize`（f32、アリティ 1、attrs `scale`）— shape は不変。scale は非負・有限で厳密な f32 値。
    f32 除算 → 最近接の偶数への整数丸め → `[-128,127]` への飽和 → f32 乗算を行う。
    scale=0 は全ビットを保つ恒等動作。符号付きゼロと NaN の符号・payload を保持し、
    scale > 0 の NaN は quiet 化する。±Inf は飽和し、出力の overflow は ±Inf。
    GPU の除算精度や非正規化数の flush に依存せず、境界・出力表を整数で読む。

- **RNN スキャン**（拡張分子層〈旧第 2 層〉の分子 — ADR [0056](decisions/0056-gru-scan.md)）:
  - `gru_scan` / `gru_scan_reverse`（f32、**attrs 空**、**アリティ 4 固定**）—
    `gi[T,N,3H]` / `h0[N,H]` / `W_hh[3H,H]` / `b_hh[3H]` → `y[T,N,H]` の GRU 隠れ側スキャン。
    **時間軸 T は記号でよい**（この op の存在理由 — 分解形は T 回展開されて記号を失う）。
    op が受け持つのは隠れ側の逐次だけで、**入力側 GEMM（`x·W_ihᵀ + b_ih`）は呼び手が既存
    `linear` で用意する**（`gi` がその結果）。1 ステップの式は
    `gh = W_hh·h + b_hh` / `r = σ(gh_r + gi_r)` / `z = σ(gh_z + gi_z)` /
    `n = tanh(gi_n + gh_n·r)` / `h' = (h − n)·z + n` で、**この演算の並びと括り方が契約**
    （数学的に同値な `(1 − z)·n + z·h` は別の丸め列 — f32 の 10 万要素中 44,345 件が
    ビット不一致）。3H のゲート並びは **r / z / n**、`b_hh` は **last**（`(Σ W·h) + b`）、
    reset ゲートは **b_hh 込みの隠れ側積**に掛かる。
    `gru_scan_reverse` は**走査順だけ**が逆で、**出力は順方向の時間順**（`flip` を挟まない —
    `flip` は記号軸を受理しないため）。
    多層 / 双方向 / `has_biases=False` / `batch_first` / `dropout` は attrs に**欄が無い**
    （層と方向は**ノードを並べて**表す — `aten.gru` の `Tensor[16]` は IR に載らない）。
    出力は `y` だけで **`h_n` を返さない**（IR の単一出力前提）。`h0` は入力スロットで
    必須（ゼロ `h0` はエクスポータの定数畳み込みで initializer になる）。
    隠れ幅は **`H ≤ 256`**（1 lane = 1 隠れユニットの割り当て — 超過は `CodegenError`）

出力 dtype は「**スロット 0 の入力 dtype → 出力 dtype**」の写像で決まる（既定は恒等）。写像は
**出力 slot 別の列**で、列の長さがその op の出力数（ADR 0068 決定 1）。恒等でないのは比較 4 本
（→ bool）・bool 入力の `sum`（→ i32）・`where`（bool → f32）・`argmax`（→ 添字の i32）・
`topk` の slot 1（→ 添字の i32・slot 0 の値は恒等）だけで、`cast` だけが例外的に attrs.to で
決まる。

dtype ごとの受理集合（**入力スロット別**の受理集合と出力 dtype の導出を含む）・attrs の値域
（`cast` の丸め規約・`permute` の `dims`・融合 op の各 attr を含む）は契約テーブルが正本。

拡張順序は [op-vocabulary.md](op-vocabulary.md) の実装順序に従う。
