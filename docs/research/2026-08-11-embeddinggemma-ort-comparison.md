# EmbeddingGemma-300m: karume vs ONNX Runtime Web/Node の実測比較

> 時点スナップショット（2026-08-11）。同一機（Ryzen 5 5600 / RTX 3080 Ti / Linux）・同一入力
> 5 ケースでの実測。karume = 本リポの export（IR v1・f32・Tmax=512）、ORT = onnx-community/
> embeddinggemma-300m-ONNX（fp32 / int8 — com.microsoft contrib op 入りの最適化済みグラフ）。
> 数値の正は各ベンチが書き出した JSON（scratchpad — 揮発）で、本書へ転記済み。

## 1. レイテンシ（median ms・warmup 3 + 計測 20 回〈int8 のみ 50〉・B=1）

| ケース (T)          | ORT Node fp32 CPU | ORT Node int8 CPU | ORT Web wasm (4thr) | **ORT Web WebGPU** | **karume (Deno/WebGPU)** |
| ------------------- | ----------------- | ----------------- | ------------------- | ------------------ | ------------------------ |
| bare (12)           | **14.5**          | 83.7              | 29.8                | 21.4               | 76.4                     |
| query-en (16)       | **15.7**          | 85.8              | 33.4                | 20.0               | 76.4                     |
| query-ja (16)       | **15.8**          | 85.8              | 32.8                | 19.5               | 76.5                     |
| document-en (19)    | **17.6**          | 87.9              | 37.4                | 19.5               | 76.1                     |
| long-document (318) | 203.9             | 287.3             | 554.7               | **28.7**           | 81.1                     |

セッション構築: ORT Node fp32 361ms（ファイル読み込み）/ ORT Web webgpu 2,893ms /
ORT Web wasm 2,369ms / karume 267ms（openModel+createSession。1.23GB の Deno.readFile は
別計上 1,901ms）。

環境: Node 24.18.0 + onnxruntime-node 1.27.0（CPU EP・スレッド既定）/ Chrome for Testing
151.0.7922.71 headless（`--enable-unsafe-webgpu --enable-features=Vulkan` 等・puppeteer）+
onnxruntime-web 1.27.0（COOP/COEP 付き同一オリジン配信・crossOriginIsolated=true）。
ORT **Node** の webgpu EP は「Failed to get a WebGPU adapter」で不発（同梱 Dawn の初期化）・
cuda EP はライブラリ不在 — Node での GPU は取れず、**GPU vs GPU はブラウザ版で成立**。

## 2. 数値忠実度（vs torch CPU 期待値 = 本リポの golden）

| 実装              | cos               | maxAbs          |
| ----------------- | ----------------- | --------------- |
| **karume**        | ≥ 0.9999999999987 | **2.3〜3.9e-7** |
| ORT fp32（全 EP） | 0.9975〜0.99999   | 5.0e-4〜7.9e-3  |
| ORT int8（Node）  | 0.981〜0.993      | 〜2.1e-2        |

- ORT の fp32 逸脱は **CPU / wasm / WebGPU の 3 EP でほぼ同一値** → EP ではなく
  **グラフ由来**（onnx-community の変換時最適化 = contrib op 融合による縮約順序・内部近似の
  変更が有力仮説。未特定 — graphOptimizationLevel: disabled での切り分けが open）。
  実用上は cos ≥ 0.9975 で同等だが「ORT fp32 = torch 忠実」ではない。
- karume は f32 丸め水準（e2e 門 atol 1e-6 の内側）。「数値を崩さず持ち込む」面では 3〜4 桁差。

## 3. 帰属と含意

- **karume の 76〜81ms は T 非依存の平坦フロア**（T 26 倍で +6%）。gpuTiming 追実測では
  計測装置込みでも GPU 44〜45ms・こちらも T 非依存 — 実演算ではなく **dispatch 1,681 本 +
  同期の固定費が支配**。DiT（10s 級）ではこの費用が GPU と完全重畳して見えなかったが
  （ADR 0042 の帰属実測）、**100ms 級の小型モデルでは裸で露出する** — 新しい実測点。
- 同一 GPU の ORT Web WebGPU（~20ms）との差 ≈ 3〜4 倍は、ORT 側が MultiHeadAttention 等へ
  融合した少 dispatch グラフである事実と整合。**縮める道具は既にある**: 融合 attention の
  mask 対応（ADR 0023 改訂が要る）・融合パス（ADR 0040）の transformer ブロック対応・
  バッチ化（B>1 は未計測 — dispatch 支配ならバッチで大きく効くはず）。
- CPU 系との比較: 短文は ORT Node fp32 CPU が最速（14〜18ms）・T=318 では karume が CPU 系の
  2.5〜6.8 倍速い。損益分岐 T ≈ 60〜80（T=19 と 318 の間は未計測）。
- int8（動的量子化）は**この CPU では fp32 より遅い**（短文 5.5〜5.8 倍・長文 1.4 倍）—
  「int8 = 速い」は一般化不可。
- ブラウザ（Chrome 151）の adapter は **subgroups / chromium-experimental-subgroup-matrix /
  maxImmediateSize 64 を列挙** — features recon（2026-08-10）の「ブラウザは Deno より先行」を
  実機確認。karume をブラウザで走らせる際の将来資産。

## 4. 公平性の注意（要旨）

基質が違う比較を含む（CPU vs GPU）。グラフの由来も違う（ORT = 変換時最適化済み / karume =
torch.export 直系）ため、差は「ランタイム実装の差」だけには帰属できない。ORT Node の
セッション構築はファイル読み込み込み・karume は別計上（初回起動総和なら karume ≈ 2.2s vs
ORT fp32 ≈ 0.36s）。int8 はスレッド既定 1 点のみ。B=1 のみ。OS スケジューリング・CPU 周波数は
未制御（分散は小さい）。ORT Web レッグはワークフローの構造化出力段で失敗したが、数値は
ベンチページが書いた成果物 JSON から転記（モデル出力の再構成ではない）。

## 5. open questions

- ORT fp32 の対 torch 逸脱の出所（最適化無効で開き直すと縮むか）。
- 損益分岐 T の実測（T=32/64/128 の追加）とバッチ B>1 の挙動。
- karume フロアの分解（真の GPU 実行時間 = 計測装置の pass 分割費用を除いた値・ホスト側の
  内訳）— dispatch 削減波の設計材料。
- ORT Node webgpu EP（Dawn）がヘッドレスで adapter を取れない機序（Chrome では取れる）。
