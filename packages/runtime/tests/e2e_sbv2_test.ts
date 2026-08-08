// SBV2 音響チェーンの実重み golden E2E — ADR 0013 の emit ターゲット 5 本が揃った状態
// （波 1: dp / 波 6: front / 波 7: flow・dec・voice）。**voice が緑 = SBV2 全チェーン成立**
// （front が durations を出し、ホストが z_p を組み、voice が波形を出す）。
//
// tiny golden（tests/e2e_golden_test.ts）が「op 契約の被覆」を、実重み DeBERTa
// （tests/e2e_deberta_test.ts）が「テキスト側 24 層の数値一致」を受け持つのに対し、こちらは
// **音響チェーン側の実重み**を受け持つ。対象は `outputs/series/sbv2-FN4/<target>/` 配下で、
// 重み 251MB 級のためリポジトリ管理外（`.gitignore` の `outputs/`）。生成は
// `tools/exporter/export_sbv2.py`（コマンドは下の GENERATE_COMMAND がそのまま正本）。
//
// **系列（格納 dtype）でパラメタ化**してある — f32 系列 / f16 系列（ADR 0018）/ i8 系列
// （ADR 0019 の per-channel w8 格納・計算は f32）を同じ構造で回し、**ターゲット別 tolerance
// だけを系列ごとに実測導出**する。系列間で tolerance を流用しない（片方の再導出がもう片方を
// 黙って動かすため）。圧縮系列の golden は **fake-quant 後の重み**で採ってあるので、
// どの系列の誤差も「実装差だけ」を見ている（量子化そのものの質は聴感ゲートの領分）。
//
// 資産が無い環境では**その系列を SKIP** する。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。
// 逆に、資産が**中途半端に**存在する場合（ターゲット欠け / ケース欠け）は SKIP ではなく
// FAIL にする（下の「資産の完全性」テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, createSession, openModel, parseSafetensors, type Tensor } from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * 実重み SBV2 dp の torch CPU 期待値との突合に使う許容誤差。
 *
 * **tiny golden の `GOLDEN_TOLERANCE`（atol 1e-6 / rtol 1e-5）と同じ値を採る** — 実測が
 * その内側に収まったので、独自定数を作る根拠が無い（DeBERTa は 24 層の誤差蓄積で外れた
 * ため `DEBERTA_TOLERANCE` を導出したが、dp は 17 ノード・conv 3 段の浅いグラフで
 * それには当たらない）。**値が同じでも据え置きの理由は別**なので定数を共有はしない。
 *
 * 実測（`atol=rtol=0` の素の突合、5 ケース × 出力 1 本）:
 *
 * | ケース   | P   | maxAbs  | maxRel  | \|logw\| 上端 |
 * | -------- | --- | ------- | ------- | ------------- |
 * | p2       | 2   | 5.96e-7 | 4.40e-7 | 1.46          |
 * | p37      | 37  | 2.03e-6 | 2.83e-6 | 2.73          |
 * | p203     | 203 | 1.67e-6 | 1.46e-6 | 2.68          |
 * | p512     | 512 | 2.62e-6 | 1.94e-6 | 3.35          |
 * | padded   | 16  | 8.34e-7 | 1.06e-6 | 1.61          |
 *
 * MUST（判定の主役は rtol）: maxAbs 2.62e-6 は atol 1e-6 を**超えている**。通っているのは
 * 要素ごとの判定式が `abs ≤ atol + rtol·|y|` で、出力が全域 O(1)（|logw| ≤ 3.35）のため
 * rtol 項 1e-5·|y| が効いているから。余裕率は maxRel 2.83e-6 に対して約 3.5 倍で、
 * atol 側には余裕が無い。値域が 0 近傍に寄るモデル（後続の波の m_p / logs_p 等）へ
 * この値を流用してはならない — 同じ手順で実測し直すこと。
 *
 * 誤差の出所は tiny golden と同じ（fma 融合・conv / layer_norm の縮約順序・超越関数の
 * 実装差）。P を 2 → 512 と 256 倍にしても maxAbs は 4 倍程度しか伸びない — dp の縮約は
 * conv の kernel 3 × 入力 192/256 チャネルで P に依らず一定で、P は独立な列方向にしか
 * 効かないため（層数で単調に伸びる DeBERTa とは伸び方の構造が違う）。実装バグ
 * （マスク経路の取りこぼし・軸取り違え）の誤差は出力の値域と同じ O(1) で、この閾値の
 * 5 桁以上上に出る。
 */
const SBV2_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 1e-5 };

/**
 * 実重み SBV2 front（enc_p + dp + sdp reverse の融合グラフ）の許容誤差。
 *
 * **上の `SBV2_TOLERANCE` は流用できない**（dp 側の MUST がそう書いている）。front の出力
 * `m_p` / `logs_p` は 0 を跨ぐ値域で、最小の非ゼロ |y| が 7.7e-7 まで下がる — rtol を判定の
 * 主役にすると、そこで相対誤差が発散して閾値が意味を失う（実測 maxRel は 3.6e-2 に達するが、
 * これは |y| → 0 の要素のアーティファクトで、その要素の絶対誤差は 2.8e-8 しかない）。
 * **判定の主役は atol** で、rtol は値域上端の比例項として残すだけ（dp とは主役が逆）。
 *
 * 実測（`atol=rtol=0` の素の突合、5 ケース × 出力 4 本）— 各セルは maxAbs:
 *
 * | ケース   | P   | logw_sdp | logw_dp | m_p     | logs_p  | \|ref\| 上端 |
 * | -------- | --- | -------- | ------- | ------- | ------- | ------------ |
 * | p2       | 2   | 2.38e-7  | 5.96e-8 | 1.79e-7 | 1.79e-7 | 1.46         |
 * | p37      | 37  | 7.99e-6  | 5.96e-6 | 1.26e-5 | 1.55e-6 | 2.77         |
 * | p203     | 203 | 8.05e-6  | 1.75e-5 | 1.57e-5 | 3.22e-6 | 3.32         |
 * | p512     | 512 | 1.55e-5  | 5.25e-6 | 1.06e-5 | 2.15e-6 | 3.61         |
 * | padded   | 16  | 7.33e-6  | 5.36e-6 | 1.20e-5 | 2.21e-6 | 2.07         |
 *
 * atol 1e-4 は実測最悪 1.75e-5 の約 5.7 倍。rtol 1e-5 の寄与は値域上端 |y| = 3.61 でも
 * 3.6e-5（atol の 1/3 弱）で、判定を主導しない。
 *
 * 誤差が dp 単体（maxAbs 2.62e-6）より 1 桁大きいのは、front が 6 層の相対位置注意 Encoder と
 * sdp の spline 4 段（softmax / cumsum / 逆二次解）を通す深いグラフだから。出所は tiny golden と
 * 同じ（fma 融合・縮約順序・超越関数の実装差）で、**同じモデルのパッチ前後 eager 同値差
 * （worst 2.02e-5 @P=512 — `export_sbv2.py --verify` の実測）と同じ桁**に収まっている
 * ことがこの見立ての裏付け: CPU 上で縮約順序だけを変えても同じ大きさの差が出る。
 * 実装バグ（添字ずれ・軸取り違え・マスク経路の取りこぼし・窓幅の取り違え）の誤差は出力の
 * 値域と同じ O(1) で、この閾値の 4 桁上に出る。
 */
const SBV2_FRONT_TOLERANCE: Tolerance = { atol: 1e-4, rtol: 1e-5 };

/**
 * 実重み SBV2 flow（TransformerCouplingBlock reverse）の許容誤差。
 *
 * front と同じく**判定の主役は atol**（出力 z は 0 を跨ぐ潜在変数で、最小の非ゼロ |y| が
 * 2.13e-5 まで下がる — そこで相対誤差が 9.78e-3 に達するが、その要素の絶対誤差は 2e-7 級で
 * しかない）。rtol は値域上端の比例項として残すだけで、|y| = 5.61 でも寄与は 5.6e-6 と
 * atol の 1/3 以下。判定を主導しない。
 *
 * 実測（`atol=rtol=0` の素の突合、5 ケース × 出力 1 本）:
 *
 * | ケース   | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | --- | ------- | ------- | ------------ | ------------------ |
 * | p2       | 2   | 5.36e-7 | 5.69e-5 | 3.45         | 7.85e-4            |
 * | p37      | 37  | 2.15e-6 | 1.55e-4 | 4.76         | 1.08e-3            |
 * | p203     | 203 | 2.15e-6 | 3.94e-3 | 5.47         | 5.27e-5            |
 * | p512     | 512 | 2.74e-6 | 9.78e-3 | 5.61         | 2.13e-5            |
 * | padded   | 16  | 1.31e-6 | 3.69e-4 | 3.49         | 4.84e-4            |
 *
 * atol 2e-5 は実測最悪 2.74e-6 の約 7.3 倍。**front（1e-4）より 1 桁厳しくできる**のは、
 * flow が 24 層の注意を通す割に誤差が T でほとんど伸びない（T を 2 → 512 と 256 倍にして
 * maxAbs は 5 倍）ため — 層方向の縮約長は T に依らず、T は独立な列方向にしか効かない。
 * front が 1 桁大きいのは sdp の spline（softmax / cumsum / 逆二次解）を通るからで、
 * flow にはその段が無い。誤差の出所は tiny golden と同じ（fma 融合・縮約順序・超越関数の
 * 実装差）で、同じモデルのパッチ前後 eager 同値差（worst 1.43e-6 @T=512 —
 * `export_sbv2.py --verify flow` の実測）と**同じ桁**に収まっているのが裏付け:
 * CPU 上で縮約順序だけを変えても同じ大きさの差が出る。実装バグ（添字ずれ・軸取り違え・表の食い違い）の
 * 誤差は出力の値域と同じ O(1) で、この閾値の 5 桁上に出る。
 */
const SBV2_FLOW_TOLERANCE: Tolerance = { atol: 2e-5, rtol: 1e-6 };

/**
 * 実重み SBV2 dec（HiFi-GAN Generator）の許容誤差。
 *
 * **rtol は判定に一切寄与しない**（出力は波形 = 零交差の連続で、|ref| の最小非ゼロが
 * 5.5e-8 まで落ちる。実測 maxRel は 0.44 に達するが、その要素の絶対誤差は 1e-8 級）。
 * 値域上端も 0.17 しかないので rtol 1e-6 の寄与は 1.7e-7 — atol の 1/100 以下で、
 * **実質 atol 単独判定**である。この形の出力に rtol を主役に据えてはならない。
 *
 * 実測（`atol=rtol=0` の素の突合、5 ケース × 出力 1 本）:
 *
 * | ケース   | T   | 出力長 | maxAbs  | maxRel  | \|ref\| 上端 |
 * | -------- | --- | ------ | ------- | ------- | ------------ |
 * | p2       | 2   | 1024   | 2.14e-7 | 2.31e-3 | 0.0737       |
 * | p37      | 37  | 18944  | 6.58e-7 | 6.62e-2 | 0.0729       |
 * | p203     | 203 | 103936 | 4.00e-6 | 4.22e-1 | 0.1691       |
 * | p512     | 512 | 262144 | 1.97e-6 | 4.36e-1 | 0.1281       |
 * | padded   | 16  | 8192   | 4.02e-7 | 1.43e-1 | 0.0677       |
 *
 * atol 3e-5 は実測最悪 4.00e-6 の約 7.5 倍。**最終段の `tanh` は WGSL の実装依存で
 * `Math.tanh` / torch とビット一致しない**（recon §4）ので、この突合は原理的に許容誤差
 * 込みでしか成立しない — atol を 0 に締めるという選択肢は存在しない。誤差の出所は他に
 * 5 段の conv_transpose1d と ResBlock 15 本（dilation 1/3/5）の縮約順序差で、いずれも
 * 数 ulp 級。実装バグ（conv_transpose の重みレイアウト取り違え・leaky_relu の slope
 * 取り違え）は波形の値域と同じ O(0.1) の誤差になり、この閾値の 4 桁上に出る。
 */
const SBV2_DEC_TOLERANCE: Tolerance = { atol: 3e-5, rtol: 1e-6 };

/**
 * 実重み SBV2 voice（flow + dec の融合）の許容誤差。**これが緑 = SBV2 全チェーン成立**。
 *
 * dec と同じ理由で atol 単独判定（出力は同じ波形）。
 *
 * 実測（`atol=rtol=0` の素の突合、5 ケース × 出力 1 本）:
 *
 * | ケース   | T   | 出力長 | maxAbs  | maxRel  | \|ref\| 上端 |
 * | -------- | --- | ------ | ------- | ------- | ------------ |
 * | p2       | 2   | 1024   | 3.96e-8 | 4.22e-3 | 0.0125       |
 * | p37      | 37  | 18944  | 1.34e-6 | 2.91e-1 | 0.0812       |
 * | p203     | 203 | 103936 | 1.31e-6 | 3.62e-1 | 0.1174       |
 * | p512     | 512 | 262144 | 1.60e-6 | 1.17e+1 | 0.1670       |
 * | padded   | 16  | 8192   | 1.58e-7 | 1.38e-2 | 0.0591       |
 *
 * atol 1e-5 は実測最悪 1.60e-6 の約 6.3 倍。**融合したのに dec 単体（4.00e-6）より誤差が
 * 小さいのは、dec に入る値が違うから**（単体は randn を直に食わせるが、融合は flow が
 * 出した z·y_mask で、隣接フレームの相関が強く波形が滑らか）。誤差が「flow の誤差 + dec の
 * 誤差」に積み上がる形にはなっていない — flow の出力誤差 2.7e-6 は dec の入力値域 O(1) に
 * 対して 6 桁下で、conv の縮約に埋もれる。この観測は「融合で誤差が累積して閾値の再導出が
 * 要る」という予想を**否定**した（tolerance は単体より厳しく採れている）。
 */
const SBV2_VOICE_TOLERANCE: Tolerance = { atol: 1e-5, rtol: 1e-6 };

/**
 * **f16 系列**（`--dtype f16` — 適格な重みスロットだけ f16 格納・計算は f32）の dp の許容誤差。
 *
 * MUST: f32 系列の値を流用しない。golden は fake-quant 後の重みで採ってあるので**量子化誤差は
 * 差に入らない**（ADR 0006 の方法論）が、丸め後の重みは値そのものが変わるため、縮約の丸め方も
 * 変わる — 誤差の大きさは f32 系列と同じにはならない。下は本波の実測（`atol=rtol=0`・
 * 5 ケース × 出力 1 本）:
 *
 * | ケース   | P   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | --- | ------- | ------- | ------------ | ------------------ |
 * | p2       | 2   | 2.38e-7 | 1.63e-7 | 1.46         | 1.36               |
 * | p37      | 37  | 1.07e-6 | 1.51e-6 | 2.73         | 4.11e-1            |
 * | p203     | 203 | 3.58e-6 | 1.82e-6 | 2.68         | 5.46e-1            |
 * | p512     | 512 | 2.15e-6 | 1.91e-6 | 3.35         | 3.86e-1            |
 * | padded   | 16  | 1.67e-6 | 1.06e-6 | 1.61         | 7.85e-1            |
 *
 * **f32 系列と同じく判定の主役は rtol**（実測が主役構造を支持している — 出力は全域 O(1) で、
 * 非ゼロ要素の \|y\| は 0.386 以上までしか下がらない。ゼロは padded の maxRel が 1.06e-6 に
 * 留まることから**厳密な 0 どうし**と分かる）。rtol 1e-5 は実測最悪 maxRel 1.91e-6 の約 5.2 倍。
 * atol 1e-6 は下限項で、**それ単独では足りない**（maxAbs 3.58e-6 を下回る）— f32 系列と同じ形。
 *
 * 実測最悪 3.58e-6 は f32 系列の 2.62e-6 と**同桁**で、これは「golden が丸め後の重みで
 * 採れている」ことの裏取りでもある。掛け忘れていれば差は f16 の量子化誤差そのもの
 * （重みの相対 5e-4 級）になり、3 桁上に出る。
 */
const SBV2_F16_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 1e-5 };

/**
 * **f16 系列**の front（enc_p + dp + sdp reverse の融合グラフ）の許容誤差。
 *
 * f32 系列と同じく**判定の主役は atol**（`m_p` / `logs_p` は 0 を跨ぐ値域で、実測の最小非ゼロ
 * \|ref\| は 2.35e-7 まで下がる — rtol を主役にすると発散する）。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 4 本）— 各セルは maxAbs:
 *
 * | ケース   | P   | logw_sdp | logw_dp | m_p     | logs_p  | \|ref\| 上端 |
 * | -------- | --- | -------- | ------- | ------- | ------- | ------------ |
 * | p2       | 2   | 1.85e-6  | 5.96e-7 | 2.68e-7 | 1.19e-7 | 1.46         |
 * | p37      | 37  | 4.05e-6  | 3.10e-6 | 8.76e-6 | 1.01e-6 | 2.78         |
 * | p203     | 203 | 7.27e-6  | 4.29e-6 | 1.06e-5 | 1.52e-6 | 3.32         |
 * | p512     | 512 | 3.62e-5  | 6.91e-6 | 7.73e-6 | 1.28e-6 | 3.61         |
 * | padded   | 16  | 2.27e-6  | 1.55e-6 | 3.64e-6 | 1.01e-6 | 2.07         |
 *
 * atol 3e-4 は実測最悪 3.62e-5（p512 の `logw_sdp`）の約 8.3 倍。rtol 1e-5 の寄与は値域上端
 * \|y\| = 3.61 でも 3.6e-5（atol の 1/8）で、判定を主導しない。
 *
 * **5 ターゲットで唯一、f32 系列より実測が有意に大きい**（1.75e-5 → 3.62e-5 の 2.1 倍）。
 * 伸びているのは `logw_sdp` = sdp の spline（softmax / cumsum / 逆二次解）を通る出力だけで、
 * 他 3 出力は f32 系列と同桁のまま。f16 で丸めた重みは spline の分点をわずかに動かすため、
 * 逆二次解の条件数が悪い領域で縮約順序差が増幅されると読める（誤差の**出所**は f32 系列と
 * 同じ fma 融合・縮約順序・超越関数の実装差で、増えたのは増幅率）。実装バグ（添字ずれ・
 * 軸取り違え・マスク経路の取りこぼし）の誤差は出力の値域と同じ O(1) で、この閾値の 3 桁上。
 */
const SBV2_F16_FRONT_TOLERANCE: Tolerance = { atol: 3e-4, rtol: 1e-5 };

/**
 * **f16 系列**の flow（TransformerCouplingBlock reverse）の許容誤差。
 *
 * f32 系列と同じく atol 主役（出力 z は 0 を跨ぐ潜在変数で、実測の最小非ゼロ \|ref\| は
 * 6.56e-6 まで下がる）。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 1 本）:
 *
 * | ケース   | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | --- | ------- | ------- | ------------ | ------------------ |
 * | p2       | 2   | 4.77e-7 | 5.72e-5 | 3.45         | 7.82e-4            |
 * | p37      | 37  | 2.03e-6 | 2.32e-4 | 4.76         | 9.25e-4            |
 * | p203     | 203 | 2.03e-6 | 9.98e-3 | 5.47         | 5.15e-5            |
 * | p512     | 512 | 2.15e-6 | 4.78e-3 | 5.61         | 6.56e-6            |
 * | padded   | 16  | 1.55e-6 | 4.14e-4 | 3.49         | 2.16e-4            |
 *
 * atol 2e-5 は実測最悪 2.15e-6 の約 9.3 倍。f32 系列（実測最悪 2.74e-6）と**同桁で、むしろ
 * わずかに小さい** — flow には spline が無く、T を 256 倍にしても誤差がほとんど伸びない
 * 構造（層方向の縮約長は T に依らない）が f16 でもそのまま保たれている。
 */
const SBV2_F16_FLOW_TOLERANCE: Tolerance = { atol: 2e-5, rtol: 1e-6 };

/**
 * **f16 系列**の dec（HiFi-GAN Generator）の許容誤差。
 *
 * f32 系列と同じく**実質 atol 単独判定**（出力は波形 = 零交差の連続で、最小非ゼロ \|ref\| が
 * 2.03e-8 まで落ちる。実測 maxRel は 1.90 に達するが、その要素の絶対誤差は 1e-8 級）。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 1 本）:
 *
 * | ケース   | T   | 出力長 | maxAbs  | maxRel  | \|ref\| 上端 |
 * | -------- | --- | ------ | ------- | ------- | ------------ |
 * | p2       | 2   | 1024   | 1.64e-7 | 1.12e-3 | 0.0737       |
 * | p37      | 37  | 18944  | 5.18e-7 | 4.84e-2 | 0.0730       |
 * | p203     | 203 | 103936 | 2.22e-6 | 1.72e-1 | 0.1692       |
 * | p512     | 512 | 262144 | 2.37e-6 | 1.90e+0 | 0.1281       |
 * | padded   | 16  | 8192   | 6.02e-7 | 1.05e-2 | 0.0677       |
 *
 * atol 2e-5 は実測最悪 2.37e-6 の約 8.4 倍。f32 系列（4.00e-6）と同桁で、最終段 `tanh` が
 * WGSL 実装依存で torch とビット一致しない事情も同じ（atol を 0 に締める選択肢は無い）。
 */
const SBV2_F16_DEC_TOLERANCE: Tolerance = { atol: 2e-5, rtol: 1e-6 };

/**
 * **f16 系列**の voice（flow + dec の融合）の許容誤差。**これが緑 = f16 系列の全チェーン成立**。
 *
 * dec と同じ理由で atol 単独判定（出力は同じ波形）。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 1 本）:
 *
 * | ケース   | T   | 出力長 | maxAbs  | maxRel  | \|ref\| 上端 |
 * | -------- | --- | ------ | ------- | ------- | ------------ |
 * | p2       | 2   | 1024   | 4.20e-8 | 1.01e-2 | 0.0125       |
 * | p37      | 37  | 18944  | 1.24e-6 | 8.79e-2 | 0.0809       |
 * | p203     | 203 | 103936 | 1.71e-6 | 1.41e-1 | 0.1172       |
 * | p512     | 512 | 262144 | 1.07e-6 | 3.43e-1 | 0.1671       |
 * | padded   | 16  | 8192   | 1.97e-7 | 1.43e-2 | 0.0591       |
 *
 * atol 1.5e-5 は実測最悪 1.71e-6 の約 8.8 倍。f32 系列と同じく**融合したのに dec 単体
 * （2.37e-6）より誤差が小さい**（dec に入る値が flow の出した滑らかな z·y_mask だから）—
 * この構造は格納 dtype を変えても保たれている。
 */
const SBV2_F16_VOICE_TOLERANCE: Tolerance = { atol: 1.5e-5, rtol: 1e-6 };

/**
 * **i8 系列**（`--dtype i8` — 適格な重みスロットだけ per-channel i8 格納・計算は f32）の dp の
 * 許容誤差。
 *
 * MUST: f32 / f16 系列の値を流用しない。golden は fake-quant 後の重みで採ってあるので
 * **量子化誤差は差に入らない**（ADR 0006 の方法論）が、量子化後の重みは値そのものが変わるため
 * 縮約の丸め方も変わる。下は本波の実測（`atol=rtol=0`・5 ケース × 出力 1 本）:
 *
 * | ケース   | P   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | --- | ------- | ------- | ------------ | ------------------ |
 * | p2       | 2   | 1.19e-7 | 8.82e-8 | 1.47         | 1.35               |
 * | p37      | 37  | 1.31e-6 | 1.13e-6 | 2.72         | 4.09e-1            |
 * | p203     | 203 | 1.67e-6 | 2.04e-6 | 2.68         | 5.47e-1            |
 * | p512     | 512 | 2.38e-6 | 2.72e-6 | 3.34         | 3.89e-1            |
 * | padded   | 16  | 1.49e-6 | 1.64e-6 | 1.61         | 7.84e-1            |
 *
 * **他 2 系列と同じく判定の主役は rtol**（出力は全域 O(1) で、非ゼロ要素の \|y\| は 0.389 までしか
 * 下がらない — 実測が主役構造を支持している）。rtol 2e-5 は実測最悪 maxRel 2.72e-6 の約 7.4 倍で、
 * atol 1e-6 は下限項（それ単独では maxAbs 2.38e-6 に足りない — f32 / f16 系列と同じ形）。
 *
 * 実測最悪 2.38e-6 は f32 系列（2.62e-6）・f16 系列（3.58e-6）と**同桁**で、これは
 * 「golden が量子化後の重みで採れている」ことの裏取りでもある。掛け忘れていれば差は i8 の
 * 量子化誤差そのもの（per-channel で重みの相対 4e-3 級）になり、桁で上に出る。
 */
const SBV2_I8_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 2e-5 };

/**
 * **i8 系列**の front（enc_p + dp + sdp reverse の融合グラフ）の許容誤差。
 *
 * 他 2 系列と同じく**判定の主役は atol**（`m_p` は 0 を跨ぐ値域で、実測の最小非ゼロ \|ref\| は
 * 1.12e-6 まで下がり、そこで maxRel が 9.6e-2 に達する — その要素の絶対誤差は 1e-7 級）。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 4 本）— 各セルは maxAbs:
 *
 * | ケース   | P   | logw_sdp | logw_dp | m_p     | logs_p  | \|ref\| 上端 |
 * | -------- | --- | -------- | ------- | ------- | ------- | ------------ |
 * | p2       | 2   | 1.43e-6  | 7.15e-7 | 3.73e-7 | 1.19e-7 | 1.45         |
 * | p37      | 37  | 6.59e-6  | 2.62e-6 | 4.77e-6 | 1.01e-6 | 2.76         |
 * | p203     | 203 | 5.90e-6  | 9.06e-6 | 8.46e-6 | 3.52e-6 | 3.30         |
 * | p512     | 512 | 2.37e-5  | 7.63e-6 | 2.16e-5 | 6.47e-6 | 3.65         |
 * | padded   | 16  | 4.29e-6  | 4.65e-6 | 3.31e-6 | 1.40e-6 | 2.12         |
 *
 * atol 2e-4 は実測最悪 2.37e-5（p512 の `logw_sdp`）の約 8.4 倍。rtol 1e-5 の寄与は値域上端
 * \|y\| = 3.65 でも 3.7e-5（atol の 1/5）で、判定を主導しない。
 *
 * **5 ターゲットで唯一 f32 系列より実測が有意に大きい**のは f16 系列と同じ構造（f32 1.75e-5 →
 * i8 2.37e-5 = 1.4 倍、f16 は 2.1 倍）で、伸びているのは `logw_sdp`（sdp の spline 経路）と
 * その入力になる `m_p` だけ。丸めた重みが spline の分点を動かし、逆二次解の条件数が悪い領域で
 * 縮約順序差の増幅率が上がる、という f16 系列で立てた読みと整合する（i8 のほうが丸めは粗いのに
 * 増幅は小さい — 丸め幅そのものではなく分点の動き方で決まることの傍証）。
 */
const SBV2_I8_FRONT_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 1e-5 };

/**
 * **i8 系列**の flow（TransformerCouplingBlock reverse）の許容誤差。
 *
 * 他 2 系列と同じく atol 主役（出力 z は 0 を跨ぐ潜在変数で、実測の最小非ゼロ \|ref\| は
 * 2.00e-6 まで下がる）。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 1 本）:
 *
 * | ケース   | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | --- | ------- | ------- | ------------ | ------------------ |
 * | p2       | 2   | 4.77e-7 | 2.05e-5 | 3.45         | 4.36e-3            |
 * | p37      | 37  | 1.67e-6 | 6.88e-4 | 4.75         | 2.98e-5            |
 * | p203     | 203 | 2.38e-6 | 1.47e-2 | 5.48         | 3.24e-5            |
 * | p512     | 512 | 2.62e-6 | 4.48e-2 | 5.60         | 2.00e-6            |
 * | padded   | 16  | 1.67e-6 | 1.47e-2 | 3.49         | 1.82e-5            |
 *
 * atol 2e-5 は実測最悪 2.62e-6 の約 7.6 倍。f32 系列（2.74e-6）・f16 系列（2.15e-6）と同桁で、
 * 「flow は T を 256 倍にしても誤差がほとんど伸びない」構造が格納 dtype を変えても保たれている。
 */
const SBV2_I8_FLOW_TOLERANCE: Tolerance = { atol: 2e-5, rtol: 1e-6 };

/**
 * **i8 系列**の dec（HiFi-GAN Generator）の許容誤差。
 *
 * 他 2 系列と同じく**実質 atol 単独判定**（出力は波形 = 零交差の連続で、最小非ゼロ \|ref\| が
 * 3.67e-8 まで落ちる。実測 maxRel は 0.478 に達するが、その要素の絶対誤差は 1e-8 級）。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 1 本）:
 *
 * | ケース   | T   | 出力長 | maxAbs  | maxRel  | \|ref\| 上端 |
 * | -------- | --- | ------ | ------- | ------- | ------------ |
 * | p2       | 2   | 1024   | 4.71e-7 | 6.36e-3 | 0.0736       |
 * | p37      | 37  | 18944  | 3.95e-7 | 9.39e-3 | 0.0745       |
 * | p203     | 203 | 103936 | 6.07e-6 | 4.78e-1 | 0.1707       |
 * | p512     | 512 | 262144 | 3.25e-6 | 3.34e-1 | 0.1182       |
 * | padded   | 16  | 8192   | 3.15e-7 | 1.00e-2 | 0.0667       |
 *
 * atol 5e-5 は実測最悪 6.07e-6 の約 8.2 倍。**3 系列で最も大きい**（f32 4.00e-6 / f16 2.37e-6）が
 * 同桁の範囲で、最終段 `tanh` が WGSL 実装依存で torch とビット一致しない事情も同じ
 * （atol を 0 に締める選択肢は無い）。
 */
const SBV2_I8_DEC_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 1e-6 };

/**
 * **i8 系列**の voice（flow + dec の融合）の許容誤差。**これが緑 = i8 系列の全チェーン成立**。
 *
 * dec と同じ理由で atol 単独判定（出力は同じ波形）。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 1 本）:
 *
 * | ケース   | T   | 出力長 | maxAbs  | maxRel  | \|ref\| 上端 |
 * | -------- | --- | ------ | ------- | ------- | ------------ |
 * | p2       | 2   | 1024   | 3.68e-8 | 1.35e-3 | 0.0126       |
 * | p37      | 37  | 18944  | 1.32e-6 | 7.03e-1 | 0.0820       |
 * | p203     | 203 | 103936 | 1.74e-6 | 9.51e-2 | 0.1159       |
 * | p512     | 512 | 262144 | 1.13e-6 | 3.95e-1 | 0.1675       |
 * | padded   | 16  | 8192   | 2.87e-7 | 4.70e-2 | 0.0591       |
 *
 * atol 1.5e-5 は実測最悪 1.74e-6 の約 8.6 倍。他 2 系列と同じく**融合したのに dec 単体
 * （6.07e-6）より誤差が小さい**（dec に入る値が flow の出した滑らかな z·y_mask だから）—
 * この構造は格納 dtype を i8 まで落としても保たれている。
 */
const SBV2_I8_VOICE_TOLERANCE: Tolerance = { atol: 1.5e-5, rtol: 1e-6 };

/**
 * ターゲット → 許容誤差。**ターゲットごとに実測から導く**（`SBV2_TOLERANCE` の MUST）ので、
 * 1 本の定数を共有しない。表の網羅性は下の「資産の完全性」テストが EXPECTED_TARGETS と
 * 突き合わせて固定する — 新しいターゲットを足して tolerance を導き忘れると赤になる。
 */
const TOLERANCES: Readonly<Record<string, Tolerance>> = {
  dp: SBV2_TOLERANCE,
  front: SBV2_FRONT_TOLERANCE,
  flow: SBV2_FLOW_TOLERANCE,
  dec: SBV2_DEC_TOLERANCE,
  voice: SBV2_VOICE_TOLERANCE,
};

/** f16 系列の表。**f32 の表を流用しない**（系列間の tolerance 流用禁止 — 上の MUST）。 */
const F16_TOLERANCES: Readonly<Record<string, Tolerance>> = {
  dp: SBV2_F16_TOLERANCE,
  front: SBV2_F16_FRONT_TOLERANCE,
  flow: SBV2_F16_FLOW_TOLERANCE,
  dec: SBV2_F16_DEC_TOLERANCE,
  voice: SBV2_F16_VOICE_TOLERANCE,
};

/** i8 系列の表。**他系列の表を流用しない**（同上）。 */
const I8_TOLERANCES: Readonly<Record<string, Tolerance>> = {
  dp: SBV2_I8_TOLERANCE,
  front: SBV2_I8_FRONT_TOLERANCE,
  flow: SBV2_I8_FLOW_TOLERANCE,
  dec: SBV2_I8_DEC_TOLERANCE,
  voice: SBV2_I8_VOICE_TOLERANCE,
};

const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド（tools/exporter/README.md と同じもの）。 */
const GENERATE_COMMAND = "cd tools/exporter && uv run --group sbv2 python export_sbv2.py";

/** 格納 dtype ごとの資産系列（ADR 0018 — 同居させない）。 */
type Sbv2Series = {
  /** テスト名に出る系列名。 */
  readonly name: string;
  readonly root: URL;
  /** ターゲット → 許容誤差（**系列ごとに実測導出**）。 */
  readonly tolerances: Readonly<Record<string, Tolerance>>;
  /**
   * この系列の資産が宣言しているべき圧縮格納 dtype（無ければ `undefined` = 全て f32）。
   *
   * MUST: この検査を落とさない。**系列 root の取り違えは数値では検出できない** — 3 系列とも
   * tolerance は実測が同桁なので互いの資産を素通りさせる（f32 の atol で f16 / i8 資産も、
   * i8 の atol で f32 資産も通る。f16 は本波の前・i8 は本波の故障注入で実測）。同じ理由で
   * 「`--dtype i8` のつもりが f32 で書けていた」も数値では見えない。格納宣言だけが区別する。
   */
  readonly compressedStorage?: "f16" | "i8";
  /** SKIP 時にそのまま貼れる生成コマンド。 */
  readonly generate: string;
};

/**
 * 系列 root。綴りの `sbv2-FN4` は `export_sbv2.py` の `default_out_root()` が `--model-dir` の
 * ディレクトリ名から導いたもの（話者ごとに系列を分ける — 綴りを共有すると別話者の書き出しが
 * 先の資産を黙って上書きする）。**当面この 1 話者を決め打ち**する。
 */
const SERIES: readonly Sbv2Series[] = [
  {
    name: "f32",
    root: new URL("../../../outputs/series/sbv2-FN4/", import.meta.url),
    tolerances: TOLERANCES,
    generate: GENERATE_COMMAND,
  },
  {
    name: "f16",
    root: new URL("../../../outputs/series/sbv2-FN4-f16/", import.meta.url),
    tolerances: F16_TOLERANCES,
    compressedStorage: "f16",
    generate: `${GENERATE_COMMAND} --dtype f16`,
  },
  {
    name: "i8",
    root: new URL("../../../outputs/series/sbv2-FN4-i8/", import.meta.url),
    tolerances: I8_TOLERANCES,
    compressedStorage: "i8",
    generate: `${GENERATE_COMMAND} --dtype i8`,
  },
];

/**
 * 生成されているはずのターゲットとケース。**列挙結果ではなくここで固定する** — 列挙だけに
 * 頼ると生成を一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/exporter/export_sbv2.py` の TARGET / GOLDEN_CASES。
 *
 * MUST: ターゲットを足すときはこの表も同時に伸ばす（増えたターゲットは等値検査で FAIL
 * するので、伸ばし忘れは黙って通らない）。**ADR 0013 の 5 本が揃った状態**。
 *
 * NOTE: ケース名の `p<n>` は front 系の P（音素数）由来だが、flow 系では **T（フレーム数）**
 * を指す。1 本の表を全ターゲットで共有する（＝どのターゲットもケースを欠かせない）ことを
 * 優先して名前は据え置いた。長さの正本は `tools/exporter/export_sbv2.py` の GOLDEN_CASES。
 */
const EXPECTED_TARGETS = ["dec", "dp", "flow", "front", "voice"] as const;
const EXPECTED_CASES = ["p2", "p203", "p37", "p512", "padded"] as const;

/**
 * 資産ディレクトリの列挙。存在しない場合だけ空に縮退する。
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、
 * 実行されていない検証が SKIP として静かに緑になる。
 */
const listDir = (url: URL): readonly Deno.DirEntry[] => {
  try {
    return [...Deno.readDirSync(url)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) {
      return [];
    }
    throw cause;
  }
};

type Sbv2Case = {
  readonly target: string;
  readonly caseName: string;
  readonly ioFile: string;
};

/**
 * 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。生の重み
 * （config.json / ckpt / style_vectors.npy）は `inputs/sbv2/FN4/` 側に分かれていて系列 root
 * には来ないが、ターゲットの実体はディレクトリなので **ディレクトリだけ**を見る形は据え置く。
 */
const discoverTargets = (root: URL): readonly string[] =>
  listDir(root).filter((entry) => entry.isDirectory).map((entry) => entry.name).sort();

const discoverCases = (root: URL, target: string): readonly Sbv2Case[] =>
  listDir(new URL(`${target}/`, root))
    .filter((entry) =>
      entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
    )
    .map((entry) => ({
      target,
      caseName: entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length),
      ioFile: entry.name,
    }))
    .sort((a, b) => a.caseName.localeCompare(b.caseName));

const readBuffer = async (root: URL, target: string, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(`${target}/${file}`, root));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

for (const series of SERIES) {
  const targets = discoverTargets(series.root);
  const cases = targets.flatMap((target) => discoverCases(series.root, target));
  /** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
  const available = targets.length > 0;

  if (!available) {
    console.warn(
      `[karume] ${series.root.pathname} に export 済みターゲットが無いため実重み SBV2 E2E ` +
        `（${series.name} 系列）を SKIP する（重み 251MB 級につきリポジトリ管理外）。` +
        `生成: ${series.generate}`,
    );
  }

  Deno.test({
    name: `SBV2 資産（${series.name}）: 期待するターゲットとケースが揃っている`,
    // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
    ignore: !available,
    fn: () => {
      assertEquals(targets, [...EXPECTED_TARGETS], `${series.root.pathname} のターゲット`);
      // tolerance はターゲットごとに実測から導く。表の穴は「別ターゲットの値で突合する」
      // 沈黙誤りになるので、ターゲット一覧との等値で塞ぐ（系列ごとに 1 枚ずつ）。
      assertEquals(
        Object.keys(series.tolerances).sort(),
        [...EXPECTED_TARGETS].sort(),
        `${series.name} 系列の tolerance の表`,
      );
      for (const target of targets) {
        assertEquals(
          discoverCases(series.root, target).map((entry) => entry.caseName),
          [...EXPECTED_CASES],
          `${target} の golden ケース`,
        );
        const model = new URL(`${target}/${MODEL_FILE}`, series.root);
        assert(Deno.statSync(model).isFile, `${target}/${MODEL_FILE} が無い`);
      }
    },
  });

  for (const { target, caseName, ioFile } of cases) {
    Deno.test({
      name:
        `SBV2 golden 突合: ${series.name} / ${target} / ${caseName}（実 GPU / torch CPU 期待値）`,
      ignore: !available || !GPU_AVAILABLE,
      fn: async () => {
        const [modelBytes, ioBytes] = await Promise.all([
          readBuffer(series.root, target, MODEL_FILE),
          readBuffer(series.root, target, ioFile),
        ]);
        const parsed = openModel(modelBytes);
        const io = parseSafetensors(ioBytes);
        // Object.hasOwn で見る（素の `tolerances[target]` はプロトタイプ由来のキーを拾う）。
        assert(
          Object.hasOwn(series.tolerances, target),
          `${series.name} 系列の ${target} の tolerance が無い`,
        );
        const tolerance = series.tolerances[target];

        // 系列と資産の格納 dtype が一致する（root 取り違え / 圧縮の掛け忘れの唯一の検出器
        // — 上の `compressedStorage` の MUST）。適格スロットは 5 ターゲットとも複数あるので、
        // 「現れた圧縮 dtype の集合」が系列の宣言とちょうど一致することを見る（本数ではなく
        // 集合で見るのは、f16 系列に i8 資産が混ざる形を「圧縮が 1 本以上ある」で通さないため）。
        // NOTE: `i32` 格納（記号依存定数 — ADR 0010）は圧縮ではないのでここでは数えない。
        const compressed = [
          ...new Set(
            Object.values(parsed.graph.initializers)
              .map((initializer) => initializer.storage.dtype)
              .filter((dtype) => dtype === "f16" || dtype === "i8"),
          ),
        ].sort();
        assertEquals(
          compressed,
          series.compressedStorage === undefined ? [] : [series.compressedStorage],
          `${series.name}/${target}: 圧縮格納 dtype の集合が系列と食い違う`,
        );
        // i8 は companion scale が無いと値が復元できない（ADR 0019）。宣言と実体の両方を見る
        // — 宣言だけならキーが実在しない形が、実体だけなら別の重みの scale を読む形が通る。
        if (series.compressedStorage === "i8") {
          for (const [name, initializer] of Object.entries(parsed.graph.initializers)) {
            if (initializer.storage.dtype !== "i8") continue;
            const scale = initializer.storage.scale;
            assert(scale !== undefined, `${series.name}/${target}: '${name}' に scale 宣言が無い`);
            assert(
              parsed.file.tensors.has(scale),
              `${series.name}/${target}: '${name}' の scale '${scale}' が資産に無い`,
            );
          }
        }

        // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
        const expectedKeys = [
          ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
          ...parsed.graph.outputs.map((_, index) => `output.${index}`),
        ].sort();
        assertEquals([...io.tensors.keys()].sort(), expectedKeys, `${ioFile} のテンソルキー`);

        // 記号次元 P は golden の入力 shape の実長から束縛される（明示 bindings を渡さない）。
        // ケースごとに P が違う（2 / 37 / 203 / 512 / 16）ので、宣言上限 Pmax = 512 に
        // 依存した実装（プランを Pmax で組む等）はここで値か shape が壊れる。
        const inputs: Record<string, Tensor> = {};
        for (const spec of parsed.graph.inputs) {
          const view = io.tensors.get(`input.${spec.name}`);
          assert(view !== undefined, `input.${spec.name} が ${ioFile} に無い`);
          inputs[spec.name] = ioTensor(io, view, spec.dtype);
        }

        const gpu = await acquireGpu();
        const session = await createSession(gpu, parsed);
        try {
          const outputs = await session.run(inputs);
          assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

          parsed.graph.outputs.forEach((name, index) => {
            const view = io.tensors.get(`output.${index}`);
            assert(view !== undefined, `output.${index} が ${ioFile} に無い`);
            const where = `${series.name}/${target}/${caseName} output.${index} ('${name}')`;
            const declared = parsed.graph.values[name].dtype;
            assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
            assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
            const report = compareTensors(outputs[name], ioTensor(io, view, declared), tolerance);
            assert(report.pass, `${where}: ${formatAllclose(report)}`);
          });
        } finally {
          await session.dispose();
          gpu.destroy();
        }
      },
    });
  }
}
