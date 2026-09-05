# 0025 — w8a8 linear 実行経路（活性 per-token i8 × 整数内積・opt-in）

- Status: accepted（2026-08-03。品質はユーザー目視裁定済み —
  「別の画像になっているがプロンプトの趣旨は通っているので正しい結果」）
- 検出器の現況（2026-08-15 追記）: 決定 6 が正本に指定した
  `packages/runtime/tests/e2e_anima_w8a8_test.ts` は models 移行（旧構成の廃止）で**削除済み**。
  現行の検出器は `packages/models/tests/e2e_anima_test.ts` の PNG sha256 完全一致門 +
  `packages/runtime/tests/gpu_i8a8_test.ts`（atol=0）。**旧 E2E の tolerance / 判別帯は歴史値**
  として読むこと。
- ノブ値の改名（2026-09-05 記録）: ADR [0074](0074-quant-seat-naming.md)（0.5.0 で
  `linearCompute` の値を `"a8"` へ改名）— 本文 5 箇所の `"i8a8"` は**当時の綴り**で、現行の型は
  `ComputePrecision = "f32" | "f16" | "a8"`（`packages/runtime/src/runtime/session-types.ts`）。
  手書きの `SessionOptions` へここから綴りを写さない。
- 対象: `SessionOptions.linearCompute: "i8a8"`（packages/runtime/src/kernels/quantize-rows.ts /
  packages/runtime/src/kernels/linear-i8a8.ts / packages/runtime/src/reference/i8a8.ts / executor `#encodeLinearI8a8`）
- 需要の実測: GEMM 置換（ADR 0022）後も linear が DiT GPU の 69.8%（1024px wi8）/
  84.2%（512px wf16）。プロトタイプの「DP4a 4.73×」は旧 16×16 カーネル比で、現行比の
  正味は 1.543×（重み i8 の dequant 除去分は別途上乗せ）。
  設計書 = [research/2026-08-03-dp4a-w8a8-design.md](../research/2026-08-03-dp4a-w8a8-design.md)。

## 品質ゲート（Q0 — 実装前に torch CPU で採取）

`tools/exporter/measure_quant_anima.py`（f32 / w8 / w8a8 の 3 構成を同一プロセス比較）:

- step 毎の latent relRMS は w8a8 が w8 の **15.6×**（1 step 目 1.04e-2 vs 6.69e-4）。
  増幅率はほぼ同じ（1.54×/step vs 1.80×/step）— 劣化は「累積の仕方」ではなく毎 step の
  雑音量。最終画像 PSNR は w8 23.92dB → w8a8 **13.28dB**。
- **目視は「劣化」ではなく「別の絵」**: 構図・配色が変わるが、ノイズ・バンディング・
  解剖破綻・ぼけは無い（拡散の軌道分岐として現れる）。画像 3 枚 = models/anima-demo/q0-w8a8/。
  → **ユーザー目視裁定で受理**（ADR 0019 の w8 目視採用と同型の判断）。
- 層別誤差は一様量子化の理論値 `relRMS ≈ peak/(127·√12)` に中央値 1.8% で一致。外れ値は
  modulation 適用後と GELU 後（peak 32〜56）。**一部層 f32 残置は不成立**（in relRMS > 0.02
  が k·n の 62% を占め、利得が −18.9% → −7.3% に縮む）。SmoothQuant 級の平滑化は有望
  （peak/440 モデルが当たっているため）だが、w8 資産のバイト再利用という本設計の前提を
  崩すので**別タスク**。

## 決定

1. **opt-in の実行時ノブ**（資産系列は増やさない）: `models/anima-i8/` 等の既存 w8 資産を
   バイトそのまま使い、`SessionOptions.linearCompute: "i8a8"` で実行形だけ切り替える。
   **既定は `"f32"` MUST** — 自動適用すると既存の PNG sha256 門と E2E tolerance が黙って
   変わる。適格 = **i8 常駐重みの linear × k%4==0 ×（k ≤ 2^17）**。適格外は従来経路のまま。
   資産系列分離（案 B）は重複バイト・IR op 化（案 C）は「量子化はグラフ意味論に入らない」
   （ADR 0006）に反し却下。
2. **quantize_rows**（1 ノード = 2 dispatch の前段・1 行 = 1 workgroup 256・行方向
   grid-stride）: `s = max(amax·(1/127), tiny)`・`q = clamp(round(x/s), ±127)`・pack4xI8。
   **±127 に閉じる**（ADR 0019 と同じ規約・−128 不使用）。absmax の NaN は**ビット列判定で
   伝播**（ADR 0020 — 素の max は NaN を飲む）。
   **MUST: scale は 127 の除算ではなく 1/127 との乗算**で作る — WGSL の f32 除算は仕様上
   2.5 ULP まで許され、本機実測でも IEEE と 200,000 中 55,605 件で 1 ULP 割れた（乗算・
   加算・fma は 0 件）。除数がデータ依存の `x/s` は置換不能なので、**丸め境界近傍の要素は
   ±1 段揺れうる**（limitations 記載・atol=0 突合は境界マージンをテストが実測して門にする）。
3. **i8a8 GEMM**（`linear:v3:i8a8:reg64x64{v4}:dp4a{Emu}`）: タイル幾何は ADR 0022 と同じ
   64×64 / 16×16 / 1 スレッド 4×4 / K タイル 16 要素（= 4 パック）。共有タイルは
   **[pack][行] / [pack][列]** 配置（プロトタイプの [行][pack] から組み替え — バンク衝突
   8-way → 2-way）。縮約は `vec4<i32>` の**厳密整数加算**（順序非依存）。
   dequant は **`out = fma(f32(acc), xs[row]·wscale[col], bias[col])`** — 乗算順序
   （xs·wscale を先に 1 つの f32 へ）と **明示 `fma`**（素の `a*b+c` はドライバが融合するか
   否かが実装依存になる — 本機は実測で融合した）を codegen で固定。
   **k ≤ 131,072（2^17）のオーバフロー門**は fail loudly（|acc| ≤ k·127² と i32 上限から。
   DiT 最大 k=8,192 の 16 倍の余裕）。
4. **ADR 0019 の「scale は要素ごと dequant」との関係**: 同 ADR が `(Σ x·q)·s` 形を却下した
   理由は「f32 縮約が順序依存で GPU と CPU 展開のビット一致を失う」こと。**w8a8 の縮約は
   整数で順序非依存**なのでその前提が消え、`(Σ)·s` 形でも **GPU と CPU 参照が atol=0 で
   一致する**（数値契約はむしろ強くなる）。ADR 0019 は w8（f32 計算）の正本として不変。
5. **dp4a / エミュの 2 変種は数値完全一致**: `dot4I8Packed`（WGSL 言語拡張
   `packed_4x8_integer_dot_product`）と `dot(unpack4xI8, unpack4xI8)`（core WGSL）は同じ
   整数を返す（中間値最大 4·127² = 64,516 で巻き戻り不能）。**拡張の有無は速度のみ** —
   fail loudly も環境別 tolerance も不要（ADR 0002 と整合）。選択は
   `wgslLanguageFeatures` の列挙（数値が同一な変種の選択にだけ使う — 機能検出ではない）。
   検出器 = テスト専用の強制エミュノブ（`LINEAR_I8A8_DOT` unique symbol・packages/runtime/mod.ts 非公開）で
   両変種を実走し atol=0 突合。
6. **E2E は数値パリティの網にできない**（実装波で実測確定）: 活性量子化は不連続関数で、
   上流 1e-5 級の差が丸め境界の ±1 段飛びを起こし数層で飽和 — GPU と torch 鏡像は「同じ
   分布の別標本」になる（step1 noise 実測 5.687e-1 / |ref| 上端 4.840）。素直な「実測の
   5〜10 倍」は恒真化するため、検出力は ① テキスト段（w8a8 非適用）の厳しい tolerance
   ② step1 の判別帯（実測の 1.6 倍 — 「活性量子化なし」の 1.515e+0 に届かない = i8a8 経路が
   f32 へ静かに落ちたら赤）③ 走ったパイプラインキー検査（quantize_rows / i8a8 GEMM 各
   454 本・linear:v2 0 本）の 3 本に置き直した（packages/runtime/tests/e2e_anima_w8a8_test.ts 冒頭が正本）。
   **本当の数値契約は packages/runtime/tests/gpu_i8a8_test.ts の atol=0**（GPU vs TS 参照 / dp4a vs エミュ）。
   鏡像フィクスチャは `anima_pipeline.py --dtype i8 --act-quant`
   （`karume/act_quant.py` が数値仕様の Python 正本）。

## 検証（全て実測済み・2026-08-03）

- GPU vs TS 参照 **atol=0**（v4 / スカラ / タイル端 / k%16≠0 / NaN 行 / ゼロ行 / 負値 —
  丸め境界マージンをテストが実測して門にする）。dp4a vs エミュ **atol=0**。
  既定 `"f32"` では生成 WGSL・数値とも 1 バイト不変（スナップショット 70 本）。
- WGSL `round` は**偶数丸め**と実測確定（torch と一致・JS `Math.round` は不一致）—
  TS 参照は `roundTiesToEven` 1 本を正本に。
- 故障注入 6+1 件（dequant 逐次化 / xs 行→列 / unpack4xU8 / 門外し / 素の max / タイル 1 減
  → 各赤実測。**clamp±128 は構造的到達不能と判明** — `s = amax/127` から |x/s| ≤ 127 が
  常に成立。検出器は clamp ではなく scale の分母で、代替注入「分母 128」が 6 本赤）。
- verify 553 / 0 failed / 2 ignored・pytest 1,850・ruff 緑。既存 tolerance 変更ゼロ。

## 実測効果（2026-08-03・seed 42）

- カーネル単体（512px・DiT 1 step・timestamp-query）: linear 816.1 → 251.0ms（**3.25×**）、
  量子化 +11.8ms 込みで 3.11×。**DiT step 全体 947.2 → 420.8ms（2.25×）**。見積り 1.543× を
  大きく超えたのは、基準（i8 重み × f32 計算）が持っていた dequant コストも同時に消えるため。
- デモ実測（turbo + w8a8・`--linear-compute i8a8`・排他 GPU）: 512px **全体 15.6s**
  （DiT 9.2s = 925ms/step・DiT GPU 370ms/step・linear:i8a8:dp4a 234ms = 63%）/
  1024px **全体 36.5s**（DiT 28.0s = 2,805ms/step・DiT GPU 2,415ms/step・linear 1,024ms =
  42%・**attention 1,110ms = 46% が次の支配項**）。f32 実行比: 512px 23.9 → 15.6s /
  1024px 68.3 → 36.5s。perf マイルストーン累計: 512px 61.4 → 15.6s（3.9×）/
  1024px ~290 → 36.5s（**7.9×**）。step 別の生値は examples/anima/README.md の NOTE。
- 画像は f32/w8 実行と**別物になる**（軌道分岐 — 品質ゲートの節）。PNG sha256 門の対象外で、
  出力名も `-w8a8` 付きで分離。

## 検出限界・制約（記録）

- 非有限値: NaN は行 scale 経由で行全体へ伝播（f32 経路と同粒度）。**Inf の符号は f32 経路と
  一致しない**（limitations 記載）。行 scale が非有限のとき `xq` の中身は契約外。
- E2E は数値パリティ網ではない（決定 6）。
- SBV2 は w8a8 の検証対象として構造的に不成立（全 5 ターゲットが conv1d 86〜90% 支配・
  linear は実質 0 GFLOP — 設計書 §7 の census）。「他モデル検証」の受け皿は DeBERTa 24 層
  （linear 86.8%）と Anima text 系 — 展開は後続タスク。

## 参照

- 設計書: [research/2026-08-03-dp4a-w8a8-design.md](../research/2026-08-03-dp4a-w8a8-design.md) /
  Q0 記録: models/anima-demo/q0-w8a8/（git 外・画像 3 枚 + 層別 CSV）
- 実装: packages/runtime/src/kernels/quantize-rows.ts / packages/runtime/src/kernels/linear-i8a8.ts / packages/runtime/src/reference/i8a8.ts /
  executor `#encodeLinearI8a8` / tools/exporter `karume/act_quant.py`
- 将来: SmoothQuant 級平滑化（要エクスポータ改訂）/ w8a16（shader-f16 と合流）/
  conv1d・conv2d への整数内積拡張

## 追記（2026-08-21・適格述語の拡張 — ADR 0076 が i4 常駐を載せた）

決定 1 の「適格 = **i8 常駐重みの linear** × k%4==0 ×（k ≤ 2^17）」は、ADR
[0076](0076-w4a8-linear-execution.md)（w4a8）で**整数常駐一般（i8 / i4）**へ広がった。現行の述語は
`recipe-builder.ts` の `linearCompute === "i8a8" && (weightStorage === "i8" || weightStorage === "i4")
&& k > 0 && k % 4 === 0`。

- **k 門（k ≤ 2^17）は i8 変種にだけ掛かる**。i4 変種は flush が group ごとなので i32 に載るのは
  1 group ぶんで、門は group 長（`LINEAR_W4A8_MAX_GROUP`）へ置き換わる（0076 決定 4）。
- **数値契約は i8 と i4 で別**（i8 = full-k 厳密 / i4 = group 部分縮約）。同じ `linearCompute: "i8a8"`
  の下に 2 契約が並ぶので、経路の識別はパイプラインキーの `:wi4g32` サフィックスが担う。
- したがって「i4 資産に `"i8a8"` を宣言しても効かない」は**もう成り立たない**（0076 以前の性質）。
