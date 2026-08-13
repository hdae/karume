# 0053 — conv1d の implicit GEMM 置換（groups==1・ビット同一）

- Status: accepted（2026-08-13・性能波①〈perf-ledger K-4a〉のユーザー承認下でメイン裁定）
- 対象: `conv1d` の実行カーネル（packages/runtime/src/kernels/conv1d.ts + gemm.ts 断片共有 +
  executor の踏み分け）。IR・契約・エクスポータは無変更（純カーネル置換・再 emit 不要）
- 需要の実測: dacvae codec の conv1d direct が decoder 1,111.57ms（27 dispatch）/ encoder
  647.39ms（31 dispatch）— K-10 後の irodori 全 GPU 5.82s の 30% で新支配項
  （[research/2026-08-13-op-timing-restats.md](../research/2026-08-13-op-timing-restats.md) §3/§5）。
  実効 577 GFLOP/s で、律速は conv2d 置換時と同じ「共有メモリ無し・MAC 毎の添字計算 +
  unpack + 範囲分岐」（ADR 0024 と同診断）。
- 実装前実証（B′ スパイク）: conv1d を退化 conv2d（H=1）として既存 igemm へ流す executor
  20 行の一時変更で、**WAV sha256 不変（ビット同一）+ decoder 16.2 倍 / encoder 13.4 倍**を
  実装前に実測してから本実装へ入った（restats §8）。実装は Kh=1/H=1 の退化を 1D 専用断片
  として正規化したもの（死んだ 2D 欄を持ち込まない — 決定 3）。

## 決定

1. **ADR 0024 の 1D 版**: `C[Cout, N] = W[Cout, K] × Xcol[K, N]`（`M = Cout`・`N = Lout`・
   `K = Cin·K`）。A タイル（重み `[Cout,Cin,K]` = 平坦化で `[M,K]` 行優先そのもの）・
   bias-first・store・内積ループは 2D 版と**同じ断片**（`prologueAConv` / `fillAConv` /
   `convAccInit` へ改名して共有 — 既存 op の生成バイト列は 1 バイトも不変・スナップショットが
   検出器）。新規は B タイルのみ（`xcol1d` の暗黙 gather + 平坦 k → `(ic, k)` 分解）。
2. **ビット同一（MUST）**: 平坦 K 昇順 = 直接カーネルの `(ic, k)` 二重昇順と厳密一致・
   bias は `acc` の初期値・範囲外の x は 0（0024 決定 3 と同じ 3 点）。唯一の数値差分は
   符号付きゼロで、1D では到達条件を「`bias[oc]` が literal `−0.0`、かつ以降の実 tap 積が
   全て `−0.0`」まで精密化できる（RN では `x + (−x) = +0.0` のため部分和は途中から `−0.0` に
   なれない）。現行 5 グラフの conv1d bias 全 143,356 要素に `−0.0` は 0 件（実測）—
   機序自体は parity 負ケース（bias = −0.0 で direct `0x80000000` / igemm `0x00000000` に
   割れる）がテストとして固定し、将来モデルでの検出器を兼ねる。
3. **バッチは dispatch の z 軸**（0024 決定 2 の踏襲）。uniform は 1D 専用 6 語
   （channels_in / length_in / kernel / stride / padding / dilation）。`length_out` は
   載せない（= `dims.n` — 同じ事実を 2 語に割らない。死んだ欄の禁止 = 0022 決定 5 /
   0024 決定 2 の規律）。
4. **v4 判定 = `kFlat%4==0 && Lout%4==0 && stride==1`**（`conv1dUsesVec4` 1 箇所）。1D では
   出力平面が 1 行なので 2D の「`Wout%4` と `N%4` の区別」（0024 決定 4）は消える。
   stride>1 でも B 側 else 枝（xcol 4 回）で v4 を許す緩和は将来候補（未実施 — 実測形の
   encoder ダウンサンプル stride 2/8/10/12 に効きうる）。
5. **groups == 1 専用**。groups > 1（SBV2 front の depthwise 12 本）は直接カーネル
   （`conv1d:v2:…:direct` 温存）— **恒久の A/B ビット比較オラクル**を兼ねる
   （packages/runtime/tests/gpu_conv1d_parity_test.ts: 13 形状 × direct/igemm の Uint32
   完全一致 + 32 行 m タイル強制の全形状再走 + 符号付きゼロ負ケース + 踏み分けの実走キー
   観測）。igemm キーは `conv1d:v3:f32:igemm{64|32}x128{v4}:wg16x{8|4}{:w…}`（direct が
   v2 を名乗るため v3）。
6. **i8 の scale は行 = 出力チャネル**（0024 決定 6 の断片共有そのもの — 検出器は m タイル
   2 枚以上の parity 形状 Cout=70/96）。
7. **m タイル述語は `conv2dIgemmMTile` を共有**（M = Cout の純関数 — 次元非依存。1D 用に
   写すと同じ境界が 2 箇所に散る）。Cout < 32（SBV2 dec の 16 / 波形ヘッドの 1）のタイル
   量子化の無駄は既知・この波では未対処。dispatch 上限は fail loudly（tileN=128 で
   Lout ≤ 65535×128 ≈ 8.39M サンプル ≈ 175s @48kHz — limitations に記載）。

## 検収（2026-08-13・RTX 3080 Ti / Vulkan・irodori voice-clone S=170）

- `deno task verify` **1053 passed / 0 failed**（実 GPU・PNG 門 4 本 + WAV 門・parity 5 本 +
  codegen 3 本の新設込み）。故障注入: `conv1dKDecode` を `(k, ic)` 昇順へ差し替えると
  parity 2 本が即赤（tap 集合同一・加算順のみの差を検出 — tolerance 比較では不可視の欠陥）。
- 検収実測（A/B・クールダウン規約・数値の正本は restats §8）: conv1d 合計 1,758.96 →
  110.1ms（**16.0 倍** — decoder 64.2ms = 17.3 倍 / encoder 45.9ms = 14.1 倍）・irodori
  全 GPU 5,818 → 4,226ms（**−27.4%**）・壁 ×1.12。WAV sha256 は本日の全 6 走（K-7 計測 2 +
  スパイク 2 + 検収 2）で既存一致値 `e7846ac1…` と同一。実装コミット `7a725d7`。
