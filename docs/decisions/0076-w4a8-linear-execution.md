# 0076: w4a8 — i4 常駐の重みを整数内積の経路に載せる

- Status: accepted（2026-08-21 — ユーザー裁定「a の w4a8 も試してみたいので、お願いします」。
  実装・実測とも完了。**ただし anima の配布席では採らない** — 決定 6）
- Date: 2026-08-21
- 関連: ADR [0025](0025-w8a8-linear-execution.md)（w8a8 = i8 常駐の整数内積・本 ADR の親 —
  適格述語の拡張は同 ADR の 2026-08-21 追記）/
  [0069](0069-packed-w4-storage.md)（i4 格納・group scale — 決定 5 の scale 適用形は同 ADR の
  追記 8 で改訂）/ [0058](0058-numerics-opt-in-contract.md)（numerics opt-in の一般契約 —
  席は活性の i8 化 1 つで、格納形は別軸）/
  [0074](0074-quant-seat-naming.md)（0.5.0 で `linearCompute` の値を `"a8"` へ改名）/
  perf-ledger Q-8 / 実測 =
  [research/2026-08-21-anima-i4-seat-speed.md](../research/2026-08-21-anima-i4-seat-speed.md)

## Context

`linearCompute: "i8a8"` の述語は **i8 常駐**を必要条件に含んでいた（`weightStorage === "i8"`）。
i4 常駐の linear はこの述語を外れ、fail loudly せず通常の f32 計算経路へ流れる（従来はこれを
「縮退ではなく i4 の実装済み経路」と明記していた）。

2026-08-21 の実測で、この設計が anima の i4 席に高い値段を付けていたことが分かった — 取得量
−21.2% / VRAM −22.6% と引き換えに DiT が既定の **2.0 倍**（1,646 vs 822 ms/step — research §3 の
2 周中央値どうし）。しかも
i4 の in-kernel dequant 自体はほぼ無料（f32 計算どうしの比較で +2%）で、**失っていたのは
dp4a の利得だけ**だった。i8a8 既定を持つファミリ（anima `w8a8-s16` / irodori `w8a8`）では、
i4 を選ぶと速度が半減する構図になる。

## Decision

### 1. 述語に i4 を足す（値は増やさない）

`linearCompute === "i8a8" && (weightStorage === "i8" || weightStorage === "i4") && k > 0 &&
k % 4 === 0`。ノブの意味は「**活性を i8 にして整数内積で計算する**」で、重みの格納形は別軸
（i8 常駐 → w8a8 / i4 常駐 → w4a8）。`"i4a8"` のような値を増やす案は採らない — 格納形は資産
ヘッダが正なので manifest との二重持ちになり、かつ**混成資産で片方しか加速できない**
（anima の i4 系列は i4 453 本 + i8 1 本）。値の綴りは 0.5.0 で `"a8"` へ改名する（ADR 0074）。

### 2. 数値契約は w8a8 と**別**（group 部分縮約）

    accf = 0
    for gi in 0..k/g-1:
        acci = Σ_{i∈group gi} xq[row,i]·wq[col,i]      # i32 厳密（|acci| ≤ g·127·8）
        accf = fma(f32(acci), wscale[col, gi], accf)   # group 境界ちょうどで flush
    out = fma(accf, xs[row], bias[col])

丸めは厳密に `k/g + 1` 回。w8a8 の「`xs·wscale` を先に 1 つの f32 へ畳む」MUST は wscale が
group ごとに変わるので**成立しない** — `xs` は最終 fma へ回す。この非対称は公開 API の
docstring と両参照に明記する。

### 3. flush は group 境界ちょうど（タイル境界ではない）

これにより丸め回数が幾何に依存せず、「タイル幾何は数値契約の外」という既存の自由度
（ADR 0025 系）が w4a8 でも保たれる。CPU 参照は素の 4 重ループのままでよい。
K タイルが group 境界を跨ぐ幾何は **CodegenError** で落とす（跨ぐと 2 group の重みが 1 つの
i32 に混ざる沈黙誤値）。

### 4. オーバフロー門の軸を k から group へ

i8 は縮約全体が 1 つの i32 なので門は **k**（`LINEAR_I8A8_MAX_K`）。i4 は flush が group ごと
なので i32 に載るのは 1 group ぶんだけで、門は **group 長**（`LINEAR_W4A8_MAX_GROUP`）。
k の門を i4 にも適用するのは根拠のない過剰制約になる。

### 5. nibble → i8 レーンは整数のまま展開する

`weight-storage.ts` に整数レーン展開を新設し、nibble 順の MUST を 1 ファイルに閉じる
（scale は group 境界の flush が掛ける）。既存 `dequant4`（f32 展開）とは別関数で、
融合 5 カーネルの生成物はバイト不変。

### 6. anima の配布席では採らない（2026-08-21 視認裁定）

実 GPU の画（1024²・8 step・seed 42）で、`w4-a8-s16` に `linearCompute` を宣言した構成は
**「細部に破綻・線がラフ」**というユーザー評価になった（宣言しない構成は「高品質を維持」）。
速度は 1,640 → 955 ms/step と大きく戻るが、**この席の存在理由はサイズと VRAM であって速度では
ない**（速度が要るなら既定の `w8a8-s16` が 823 ms/step で上）。したがって anima の i4 席は
`linearCompute` を宣言しない。runtime の実装は残す — i8a8 既定を持つ他ファミリ（irodori）と、
低 VRAM で速度も要る将来の実需のための資産。

## Consequences

- **BREAKING**: `linearCompute: "i8a8"` を i4 常駐へ指定したときの出力ビットが変わる
  （f32 計算 → 整数内積）。公開中の manifest 3 リポにこの組み合わせを宣言する席は無いので
  実害は無いが、手で `SessionOptions` を組む利用者には効く。
- 既存 i8 経路の生成物は 4 変種ともバイト同一（fixture と直接突合で確認済み）。
- 実測（RTX 3080 Ti・1024²・8 step）: **955 ms/step**（既定 823 の +16% — research §6）。
  取得量 −21.2% は i4 席のまま。**VRAM は未測定** — 重み常駐は i4 席と同じだが活性量子化の
  一時領域（`xq` / `xs`）が増える（research §8-8）。
- 残る +16% の内訳は ①nibble 展開が内側ループに乗る ②group ごとの f32 flush が k/g 回
  ③accumulator が i32 16 本 + f32 16 本でレジスタ圧が i8 の 2 倍。**専用幾何**（`I8a8GemmOp` に
  `linear_w4a8` を足して regM を落とす）は未試行で、詰める余地として残る。
- 同じ `linearCompute: "i8a8"` の下に 2 つの数値契約が並ぶ（i8 = full-k 厳密 / i4 = group 部分
  縮約）。取り違えると atol=0 の主張が意味を失うので、経路の識別はパイプラインキーの
  `:wi4g32` サフィックスと診断が担う。
