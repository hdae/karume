# 0018 — f16 格納の実行経路（GPU 常駐 + カーネル内 dequant）

- Status: accepted（2026-08-03）
- 前提: ADR [0006](0006-quantization.md)（意味論 =「格納のみ量子化・計算は f32」・
  fake-quant 正・bias/折り畳み定数は常に f32・診断常設）。格納スキーマは IR v1 で確定済み、
  本 ADR は**実行経路**の設計。動機の実測は
  [known-issues.md](../known-issues.md) — GPUBuffer 総確保量の天井 7,280MiB に対し Anima DiT
  フル 28 層の f32 重みが 7,465MiB で載らない（f16 なら約 3.7GiB）。

## 決定

- **適格重みは f16 のまま GPU 常駐**させ、dequant はカーネル内で行う（ADR 0006 の帰結 —
  VRAM ≈ 1/2 はこれで初めて成立する。CPU 展開してから上げる形は配信サイズしか縮まない）。
- **読み出しは `unpack2x16float`（core WGSL・feature 依存ゼロ）**。f16 ペイロードを
  `array<u32>` として束縛し 2 要素ずつ展開する。プロトタイプは `enable f16`（shader-f16
  feature）だったが、Karume は M0.1 で optional feature を全廃した設計（ADR 0002 の互換
  重視）であり、feature 配線の再導入は perf マイルストーンで実測してから個別に裁定する
  （shader-f16 は upgrade path として温存）。奇数要素長は末尾 2 バイトのゼロ詰めで
  4 バイト整列させる（読み出しは要素数で打ち切るので値に影響しない）。
- **適格 = 融合 op の weight スロットに限る**: `linear` / `conv1d` / `conv2d` /
  `conv_transpose1d` / `embedding` の重み（= 実測でサイズが支配的な 5 スロット）。
  各カーネルに **weight 格納の変種**（`w=f32` / `w=f16`）を持たせ、パイプラインキーに
  格納判別子を含める（決定性キーの規約どおり）。bias・norm 系の weight・その他の
  initializer は f16 宣言でも**ロード時に CPU で f32 展開**する（正しさは保たれ VRAM 削減
  ゼロ — ADR 0006 の適格外規則）。
- **診断の常設**（ADR 0006 の義務）: セッション初期化時に「GPU 常駐圧縮バイト数 /
  CPU 展開バイト数」を取得できるようにし、「f16 指定なのに適格 0MB」を沈黙させない。
- **検証は fake-quant 正**（ADR 0006）: エクスポータが重みを f16 表現可能値へ丸めてから
  参照・golden を生成する。GPU 側の dequant（unpack2x16float）と CPU 側の f16→f32 展開の
  **ビット一致**をユニットで固定し、E2E は「丸め済み重みでの torch」対「GPU」の実装誤差
  だけを見る。
- **Anima の f16 資産系列は別ディレクトリ**（`models/anima-f16/` / パイプライン
  フィクスチャも fake-quant 版を別に生成）。f32 系列（models/anima/）はそのまま残す —
  transformer 以外は f32 で緑であり、f16 系列は「量子化の実装誤差」を上乗せで検証する
  独立の網になる。**DiT フル 28 層の実 GPU golden と通しチェーン段②はこの系列で初めて
  実測できる** — e2e_anima_test.ts の暫定 tolerance（transformer / 段②）はここで採り直す。
- スコープ外: bf16 / i8 の実行経路（宣言 valid・実行 fail loudly のまま。w8 は conv2d
  変種の要否込みで次の裁定）・活性 f16（ADR 0006 で不採用）。

## 検討した代替案

- shader-f16（`enable f16` + `array<f16>` 直読み）: 読み出しが速いが optional feature の
  配線（requiredFeatures・非対応機のフォールバック分岐 = パイプライン二重化）が要る。
  正しさ優先の初期実装では unpack 形 1 本が面が狭い。perf で実測してから。却下（保留）。
- ロード時 CPU 展開のみ（GPU は常に f32）: VRAM が縮まず、本 ADR の動機（DiT の天井超え）を
  解決しない。適格外スロットの扱いとしてのみ採用。
- 全 initializer を一律 f16 適格にする: bias の f32 規則（ADR 0006 — プロトタイプの降格
  バグの根治形）と衝突。却下。
