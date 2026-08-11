// 実重み Irodori-TTS v4-Small の**条件エンコーダ + DiT 1 step**の実 GPU golden E2E
// （ADR 0005 の段 3）。
//
// tiny golden（tests/e2e_golden_test.ts）が「op 契約の被覆」を、実重み SBV2
// （tests/e2e_sbv2_test.ts）が「音響チェーン側の実重み」を、EmbeddingGemma
// （tests/e2e_embeddinggemma_test.ts）が「単一ベクトル出力のテキスト系」を受け持つのに対し、
// こちらは **token 列を返すテキスト条件エンコーダ**（`[1,T,H]` の系列出力）を受け持つ。
// 対象は `outputs/series/irodori-v4-small/<target>/`（backbone の重みだけで 1.26GB のため
// リポジトリ管理外 — `.gitignore` の `outputs/`）。生成は `tools/exporter/export_irodori.py`
// （コマンドは下の GENERATE_COMMAND がそのまま正本）。
//
// ターゲットは 6 本（recon の G1 / G1a / G1b / G2 / G3 と、ADR 0047 の G5'）:
//
// - `backbone`     — 同梱 ModernBERT-ja-310m（25 層）。`[1,T]` ids → `[1,T,768]`
// - `text-proj`    — text 側 projector。`[1,T,768]` → `[1,T,512]`
// - `caption-proj` — caption 側 projector（同形・別重み）。**出力 2 本** — 第 2 出力は
//   `caption_norm` を掛けた系列で、`duration` の `caption_vec`（masked mean）をホストが
//   採るためだけに足してある（第 1 出力は `text-proj` と同じ生の projector 出力）
// - `speaker`      — 参照 latent エンコーダ（8 層）+ `speaker_norm`。`[1,S,128]` → `[1,S,768]`
// - `duration`     — `text_norm` + duration predictor。`[1,T,512]` ほか 4 本 → `[1]`
// - `dit`          — DiT 1 step（12 層・G4 畳み込み形）。6 本 → `[1,S,32]`
//
// `dit` だけは**実行時 bool マスク**（`[1,1,1,S+1519]`）を入力に取り、SDPA を分解経路 +
// `safe_softmax`（ADR 0044）で通す。K/V の連結軸が記号次元になる形（ADR 0046 の `S+1519`）を
// 実資産で踏む唯一のターゲットでもある。
//
// backbone を projector と融合していないのは **backbone が text / caption で共有**だから
// （融合すると 1.26GB の重みが 2 部できる）。ホストは backbone を 2 回回して各 projector へ
// 流すので、この E2E も projector の入力に **backbone の torch 期待値**をそのまま食わせる。
// `duration` も同じ鎖で、text 系列の入力は **`text-proj` の torch 期待値そのもの**、
// speaker / caption のベクトルは `speaker` / `caption-proj` の torch 期待値から実装の
// メソッドで作ったもの（生成は `export_irodori.py` の `_duration_cases`）。
//
// `dit` の 6 本も同じ鎖で、条件 state は `text-proj` / `caption-proj` / `speaker` の torch
// 期待値を Tmax へ右 pad したもの（生成は `export_irodori.py` の `_dit_cases`）。**uncond
// 3 変種は cond と x_t / t / 条件 state を共有し、違うのはマスクだけ**（ADR 0047 決定 1）。
//
// 記号次元はターゲットで違う（テキスト系と `duration` は T ≤ 512、`speaker` / `dit` は S ≤ 750）。
// ケース名もターゲットで違うので、期待表は**ターゲット別**に持つ（下の EXPECTED_CASES）。
//
// SBV2 と違い格納 dtype 系列は f32 の 1 本のみ（f16 / i8 は別系列で決める話）なので、
// 系列パラメタ化はしない。
//
// 資産が無い環境では**明示 SKIP**する。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。
// 逆に資産が**中途半端に**（ターゲット欠け / ケース欠け）存在する場合は SKIP ではなく
// FAIL にする（下の「資産の完全性」テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, createSession, openModel, parseSafetensors, type Tensor } from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * 実重み Irodori backbone（ModernBERT-ja-310m 25 層）の torch CPU 期待値との突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、6 ケース × 出力 1 本 `[1,T,768]`）:
 *
 * | ケース       | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------------ | --- | ------- | ------- | ------------ | ------------------ |
 * | text-short   | 7   | 2.43e-5 | 6.42e-3 | 34.55        | 5.31e-5            |
 * | text-formal  | 13  | 1.53e-5 | 9.80e-2 | 34.37        | 8.24e-7            |
 * | caption-ja   | 22  | 1.74e-5 | 1.62e-1 | 34.29        | 4.32e-6            |
 * | text-emoji   | 29  | 1.96e-5 | 9.98e-3 | 34.30        | 8.29e-6            |
 * | text-long    | 144 | 2.67e-5 | 8.84e-2 | 34.21        | 1.13e-5            |
 * | caption-long | 404 | 2.72e-5 | 8.66e-1 | 34.22        | 2.06e-6            |
 *
 * **判定の主役は atol**。出力は隠れ状態そのもの（0 を跨ぐ値域で、最小の非ゼロ \|ref\| が
 * 8.24e-7 まで下がる）なので、rtol を主役にすると 0 近傍で発散する — 実測 maxRel は 0.866 に
 * 達するが、その要素の絶対誤差は 2e-6 級でしかない。rtol 1e-6 は値域上端 \|ref\| = 34.55 でも
 * 寄与が 3.5e-5（atol の 1/6）で、判定を主導しない。
 *
 * atol 2e-4 は実測最悪 2.72e-5（caption-long）の約 7.4 倍。誤差の出所は他の実重み E2E と
 * 同じ（fma 融合・linear / attention の縮約順序が torch と違う・超越関数の実装差）で、
 * T を 7 → 404 と 58 倍にしても maxAbs は 1.1 倍にしかならない（層方向の縮約長は T に
 * 依らず、T は独立な列方向にしか効かない — SBV2 flow と同じ構造）。実装バグ（RoPE の
 * θ 2 系統の取り違え・sliding window の窓幅取り違え・qkv 分割の取り違え）の誤差は出力の
 * 値域と同じ O(10) で、この閾値の 4〜5 桁上に出る。
 *
 * NOTE: この値は **`|ref|` 上端が 34 という値域**の上に立っている。同じ 2e-4 を値域 O(1) の
 * 出力へ流用してはならない（同じ手順で実測し直すこと）。
 */
const BACKBONE_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 1e-6 };

/**
 * text 側 projector（`[1,T,768]` → `[1,T,512]` の residual_mlp）の許容誤差。
 *
 * 実測（`atol=rtol=0`、6 ケース × 出力 1 本）:
 *
 * | ケース       | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------------ | --- | ------- | ------- | ------------ | ------------------ |
 * | text-short   | 7   | 7.15e-6 | 2.68e-4 | 11.89        | 3.00e-3            |
 * | text-formal  | 13  | 8.11e-6 | 7.88e-4 | 12.76        | 2.00e-4            |
 * | caption-ja   | 22  | 8.11e-6 | 4.55e-3 | 12.97        | 1.96e-4            |
 * | text-emoji   | 29  | 9.06e-6 | 1.53e-3 | 12.74        | 7.85e-4            |
 * | text-long    | 144 | 1.14e-5 | 1.57e-1 | 14.95        | 7.21e-6            |
 * | caption-long | 404 | 1.29e-5 | 2.32e-1 | 13.35        | 1.65e-5            |
 *
 * backbone と同じく **atol 主役**（最小の非ゼロ \|ref\| が 7.21e-6 まで下がる）。atol 1e-4 は
 * 実測最悪 1.29e-5 の約 7.8 倍で、rtol 1e-6 の寄与は上端 \|ref\| = 14.95 でも 1.5e-5。
 *
 * **backbone より 1 桁近く小さい**のは、この graph が 3 本の linear + rms_norm + sigmoid の
 * 7 ノードしかなく、25 層ぶんの縮約誤差の蓄積が無いから（入力は torch が出した backbone の
 * 期待値そのもので、backbone 側の誤差はここには入らない）。
 */
const TEXT_PROJ_TOLERANCE: Tolerance = { atol: 1e-4, rtol: 1e-6 };

/**
 * caption 側 projector の許容誤差。**`TEXT_PROJ_TOLERANCE` は流用しない** — 同形の graph でも
 * 重みが別なので、縮約の丸まり方も出力の値域も同じにはならない（実測が実際に違う）。
 *
 * 実測（`atol=rtol=0`、6 ケース × 出力 1 本）:
 *
 * | ケース       | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------------ | --- | ------- | ------- | ------------ | ------------------ |
 * | text-short   | 7   | 1.14e-5 | 1.16e-3 | 15.82        | 1.12e-3            |
 * | text-formal  | 13  | 1.05e-5 | 3.39e-3 | 16.16        | 4.57e-4            |
 * | caption-ja   | 22  | 3.05e-5 | 4.21e-2 | 32.83        | 2.94e-5            |
 * | text-emoji   | 29  | 1.53e-5 | 3.25e-3 | 16.27        | 8.07e-4            |
 * | text-long    | 144 | 1.53e-5 | 1.37e-2 | 18.04        | 8.25e-5            |
 * | caption-long | 404 | 1.14e-5 | 7.85e-2 | 18.09        | 2.29e-5            |
 *
 * atol 2e-4 は実測最悪 3.05e-5（caption-ja）の約 6.6 倍。最悪ケースが caption-ja なのは、
 * **本物の caption を食わせたときだけ出力の値域が 2 倍近く伸びる**（\|ref\| 上端 32.8 対
 * 他の 16〜18）ためで、誤差は値域に比例して伸びている。text 側 projector が同じケースで
 * 8.11e-6 に留まるのと対照的 — 2 本の tolerance を分けている理由がここに出ている。
 */
const CAPTION_PROJ_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 1e-6 };

/**
 * caption 側 projector の**第 2 出力**（`caption_norm` = RMSNorm 512 を掛けた系列）の許容誤差。
 *
 * 実測（`atol=rtol=0`、6 ケース × 出力 1 本 `[1,T,512]`）:
 *
 * | ケース       | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------------ | --- | ------- | ------- | ------------ | ------------------ |
 * | text-short   | 7   | 4.77e-6 | 1.16e-3 | 4.37         | 3.87e-4            |
 * | text-formal  | 13  | 2.62e-6 | 3.39e-3 | 3.68         | 1.53e-4            |
 * | caption-ja   | 22  | 3.34e-6 | 4.21e-2 | 4.24         | 7.80e-6            |
 * | text-emoji   | 29  | 3.81e-6 | 3.25e-3 | 4.34         | 1.44e-4            |
 * | text-long    | 144 | 3.34e-6 | 1.37e-2 | 5.36         | 1.25e-5            |
 * | caption-long | 404 | 3.34e-6 | 7.85e-2 | 5.29         | 6.42e-6            |
 *
 * **第 1 出力（`CAPTION_PROJ_TOLERANCE`）の値を流用してはならない**。RMSNorm は行ごとに
 * `rsqrt(mean(x²))` で割るので値域が 1 桁縮み（\|ref\| 上端 32.8 → 4.24）、絶対誤差も
 * 同じ割合で縮む（3.05e-5 → 3.34e-6）。maxRel が 2 出力で**桁まで一致**している
 * （caption-ja 4.21e-2 / caption-long 7.85e-2）のがその証拠 — 誤差は第 1 出力から相対量として
 * 持ち越され、norm の縮約自体はほとんど誤差を足していない。
 *
 * atol 3e-5 は実測最悪 4.77e-6（text-short）の約 6.3 倍。rtol 1e-6 の寄与は上端
 * \|ref\| = 5.36 でも 5.4e-6（atol の 1/5.6）で、判定を主導しない。norm の掛け違い
 * （`text_norm` との取り違え・weight 無しの素の RMS）は値域と同じ O(1) で出る
 * （export 台本の `_norm_divergence` が実測する weight 差は 0.318）。
 */
const CAPTION_NORM_TOLERANCE: Tolerance = { atol: 3e-5, rtol: 1e-6 };

/**
 * 参照 latent エンコーダ（`ReferenceLatentEncoder` 8 層 + `speaker_norm`）の許容誤差。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 1 本 `[1,S,768]`）:
 *
 * | ケース  | S   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------- | --- | ------- | ------- | ------------ | ------------------ |
 * | ref-min | 2   | 6.14e-6 | 6.87e-3 | 3.60         | 1.09e-4            |
 * | ref-1s  | 6   | 9.80e-5 | 2.99e-2 | 3.85         | 1.22e-4            |
 * | ref-5s  | 31  | 2.24e-5 | 3.18e-2 | 4.42         | 1.81e-4            |
 * | ref-30s | 187 | 5.34e-5 | 1.11e-1 | 4.94         | 7.60e-6            |
 * | ref-max | 750 | 5.83e-5 | 5.20e-1 | 5.30         | 1.70e-6            |
 *
 * atol 1e-3 は実測最悪 9.80e-5（ref-1s）の約 10 倍。**backbone / projector より 1 桁緩い**
 * のは、値域が O(5) と小さいのに絶対誤差が同程度出るから — 入力の `in_proj` 出力を 6 で
 * 割ってから 8 段の RMSNorm 付き残差を通す構造で、正規化のたびに小さな中間値の相対差が
 * 拡大する。加えて **golden の参照 latent は合成（標準正規）**で、実音声の DACVAE latent とは
 * 統計が違う（実 latent での誤差はこれより小さい可能性が高いが、コーデック波が済むまで
 * 測れない — その時点で測り直す）。
 *
 * 誤差の伸びが S に単調でない（S=6 が最悪で S=750 が 5.8e-5）のは、層方向の縮約長が S に
 * 依らず、S は独立な列方向にしか効かないため（backbone と同じ構造）。実装バグ
 * （RoPE の実数化の取り違え・q/k ノルムの head ごと weight の取り違え・sigmoid ゲートの
 * 掛け違い）の誤差は値域と同じ O(5) で、この閾値の 3〜4 桁上に出る。
 */
const SPEAKER_TOLERANCE: Tolerance = { atol: 1e-3, rtol: 1e-6 };

/**
 * duration predictor（`text_norm` + token-sum 形）の許容誤差。出力は `[1]` の
 * **log frames** 1 要素だけ。
 *
 * 実測（`atol=rtol=0`、5 ケース × 出力 1 本 `[1]`）:
 *
 * | ケース            | T   | maxAbs  | maxRel  | \|ref\|  |
 * | ----------------- | --- | ------- | ------- | -------- |
 * | dur-both          | 7   | 0       | 0       | 4.658    |
 * | dur-speaker-only  | 13  | 0       | 0       | 5.312    |
 * | dur-neither       | 22  | 4.77e-7 | 9.31e-8 | 5.120    |
 * | dur-caption-only  | 29  | 0       | 0       | 5.365    |
 * | dur-long          | 144 | 0       | 0       | 6.998    |
 *
 * **5 ケース中 4 本がビット一致**で、残る 1 本の 4.77e-7 も値域 5.12 における
 * **1 ulp ちょうど**（f32）。`log1p(Σ softplus)` の縮約が T 個の非負値の和なので、
 * 桁落ちの起きようが無い形になっている。
 *
 * atol 5e-6 は実測最悪の約 10 倍（≈ 10 ulp）。ホストはこの値を `expm1` してフレーム数に
 * するので、意味のある単位に直すと `e^7 × 5e-6 ≈ 5.5e-3` フレーム — 四捨五入で
 * 整数フレームに落とす下流には 1 ミリも届かない。逆に条件ベクトルの取り違え
 * （speaker / caption の入れ替え・`null_*` 選択の反転）は、上の表で
 * `dur-both` と `dur-neither` が 0.46 も違うことから分かるとおり **O(0.1〜1)** で出る。
 */
const DURATION_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 1e-6 };

/**
 * DiT 1 step（12 層 JointAttention + SwiGLU + LowRankAdaLN）の許容誤差。出力は速度場
 * `v_pred[1,S,32]`。
 *
 * 実測（`atol=rtol=0`、7 ケース × 出力 1 本）:
 *
 * | ケース             | S   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------------------ | --- | ------- | ------- | ------------ | ------------------ |
 * | dit-cond-min       | 2   | 4.53e-6 | 6.57e-5 | 2.43         | 5.19e-3            |
 * | dit-cond-1s        | 25  | 4.77e-6 | 4.41e-3 | 3.49         | 1.60e-4            |
 * | dit-cond-late      | 25  | 7.69e-6 | 4.79e-4 | 2.87         | 1.87e-3            |
 * | dit-uncond-text    | 25  | 4.53e-6 | 1.74e-3 | 3.42         | 5.26e-4            |
 * | dit-uncond-speaker | 25  | 5.72e-6 | 1.79e-4 | 3.53         | 3.60e-3            |
 * | dit-uncond-caption | 25  | 6.32e-6 | 5.24e-4 | 3.37         | 2.20e-3            |
 * | dit-cond-max       | 750 | 8.34e-6 | 6.57e-2 | 4.56         | 1.65e-5            |
 *
 * 他の 5 本と同じく **atol 主役**（S=750 では最小の非ゼロ \|ref\| が 1.65e-5 まで下がり、
 * maxRel はそこで 6.6e-2 に跳ねるが、その要素の絶対誤差は 1e-6 級でしかない）。
 * atol 5e-5 は実測最悪 8.34e-6（dit-cond-max）の約 6.0 倍で、rtol 1e-6 の寄与は
 * 上端 \|ref\| = 4.56 でも 4.6e-6（atol の 1/11）。
 *
 * **値域は O(5) と小さいのに絶対誤差は speaker と同オーダー**で、12 層ぶんの縮約誤差に
 * 加えて条件側 1519 トークンの縮約が毎層乗る（K/V の縮約長は S + 1519 で、S=2 でも 1521）。
 * S を 2 → 750 と 375 倍にしても maxAbs が 1.8 倍にしか伸びないのはこのため（縮約長の
 * 支配項が条件側）。実装バグ（マスクの区間割りの取り違え・RoPE を掛ける先頭 10 head の
 * 取り違え・条件 KV の連結順の入れ替え）の誤差は値域と同じ O(1) で、この閾値の 4〜5 桁上に
 * 出る（export 台本の `_dit_uncond_divergence` が実測する 4 本の相互差は 0.75〜1.92）。
 *
 * NOTE: 他ターゲットの値を流用してはならない（同じ手順で実測し直すこと）。
 */
const DIT_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 1e-6 };

/**
 * ターゲット別 tolerance（**出力位置ごとの配列**）。表の穴は「別ターゲット / 別出力の値で
 * 突合する」沈黙誤りになるので、本数が IR の出力数と合っているかもケースごとに検査する。
 */
const TOLERANCES: Readonly<Record<string, readonly Tolerance[]>> = {
  "backbone": [BACKBONE_TOLERANCE],
  "text-proj": [TEXT_PROJ_TOLERANCE],
  "caption-proj": [CAPTION_PROJ_TOLERANCE, CAPTION_NORM_TOLERANCE],
  "speaker": [SPEAKER_TOLERANCE],
  "duration": [DURATION_TOLERANCE],
  "dit": [DIT_TOLERANCE],
};

const SERIES_ROOT = new URL("../../../outputs/series/irodori-v4-small/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド（tools/exporter/export_irodori.py の docstring）。 */
const GENERATE_COMMAND =
  "cd tools/exporter && uv run --with 'transformers==5.14.1' python export_irodori.py";

/**
 * 生成されているはずのターゲットとケース。**列挙結果ではなくここで固定する** — 列挙だけに
 * 頼ると生成を一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/exporter/export_irodori.py` の TARGETS / GOLDEN_CASES / SPEAKER_CASES /
 * DURATION_CASES / DIT_CASES。
 *
 * MUST: ターゲット（G6 / G7 = codec）を足すときはこの表と TOLERANCES を同時に伸ばす。
 */
const TEXT_CASES = [
  "caption-ja",
  "caption-long",
  "text-emoji",
  "text-formal",
  "text-long",
  "text-short",
] as const;

/** ターゲット別の期待ケース（ソート済み — 列挙結果と `assertEquals` で突き合わせる）。 */
const EXPECTED_CASES: Readonly<Record<string, readonly string[]>> = {
  "backbone": TEXT_CASES,
  "caption-proj": TEXT_CASES,
  "text-proj": TEXT_CASES,
  "speaker": ["ref-1s", "ref-30s", "ref-5s", "ref-max", "ref-min"],
  "duration": [
    "dur-both",
    "dur-caption-only",
    "dur-long",
    "dur-neither",
    "dur-speaker-only",
  ],
  "dit": [
    "dit-cond-1s",
    "dit-cond-late",
    "dit-cond-max",
    "dit-cond-min",
    "dit-uncond-caption",
    "dit-uncond-speaker",
    "dit-uncond-text",
  ],
};

const EXPECTED_TARGETS = Object.keys(EXPECTED_CASES).sort();

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

/**
 * 系列ディレクトリ直下にある**グラフ以外**の生成物。ここは IR ターゲットの列挙なので、
 * 別の台本が書くホスト側資産は除く（`irodori_tokenizer.py` のトークナイザ資産と
 * `irodori_pipeline.py` の full-loop latent golden）。
 *
 * MUST: 「知らないディレクトリは無視」にしない — 除外は名前で明示する。未知の名前が
 * 増えたらこのテストが落ちて、門の対象から漏れていることに気づける。
 */
const NON_GRAPH_DIRS: ReadonlySet<string> = new Set(["pipeline", "tokenizer"]);

const discoverTargets = (root: URL): readonly string[] =>
  listDir(root)
    .filter((entry) => entry.isDirectory && !NON_GRAPH_DIRS.has(entry.name))
    .map((entry) => entry.name)
    .sort();

const discoverCases = (root: URL, target: string): readonly string[] =>
  listDir(new URL(`${target}/`, root))
    .filter((entry) =>
      entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
    )
    .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
    .sort();

const readBuffer = async (root: URL, target: string, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(`${target}/${file}`, root));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
const TARGETS = discoverTargets(SERIES_ROOT);
/** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
const AVAILABLE = TARGETS.length > 0;

if (!AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に export 済みターゲットが無いため実重み Irodori ` +
      `テキスト系 E2E を SKIP する（backbone だけで 1.26GB につきリポジトリ管理外）。` +
      `生成: ${GENERATE_COMMAND}`,
  );
}

Deno.test({
  name: "Irodori 資産: 期待するターゲットとケースが揃っている",
  // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
  ignore: !AVAILABLE,
  fn: () => {
    assertEquals(TARGETS, EXPECTED_TARGETS, `${SERIES_ROOT.pathname} のターゲット`);
    assertEquals(
      Object.keys(TOLERANCES).sort(),
      EXPECTED_TARGETS,
      "ターゲット別 tolerance の表",
    );
    for (const target of TARGETS) {
      assert(Object.hasOwn(EXPECTED_CASES, target), `${target} の期待ケース表が無い`);
      assertEquals(
        discoverCases(SERIES_ROOT, target),
        [...EXPECTED_CASES[target]],
        `${target} の golden ケース`,
      );
      const model = new URL(`${target}/${MODEL_FILE}`, SERIES_ROOT);
      assert(Deno.statSync(model).isFile, `${target}/${MODEL_FILE} が無い`);
    }
  },
});

for (const target of TARGETS) {
  for (const caseName of discoverCases(SERIES_ROOT, target)) {
    Deno.test({
      name: `Irodori golden 突合: ${target} / ${caseName}（実 GPU / torch CPU 期待値）`,
      ignore: !AVAILABLE || !GPU_AVAILABLE,
      fn: async () => {
        const ioFile = `${IO_PREFIX}${caseName}${IO_SUFFIX}`;
        const [modelBytes, ioBytes] = await Promise.all([
          readBuffer(SERIES_ROOT, target, MODEL_FILE),
          readBuffer(SERIES_ROOT, target, ioFile),
        ]);
        const parsed = openModel(modelBytes);
        const io = parseSafetensors(ioBytes);
        // Object.hasOwn で見る（素の `TOLERANCES[target]` はプロトタイプ由来のキーを拾う）。
        assert(Object.hasOwn(TOLERANCES, target), `${target} の tolerance が無い`);
        const tolerances = TOLERANCES[target];
        // 出力ごとに値域が違う（caption-proj の第 2 出力は norm 済みで 1 桁小さい）ので、
        // 本数が合っていない表は「別の出力の閾値で突合する」形で静かに通ってしまう。
        assertEquals(
          tolerances.length,
          parsed.graph.outputs.length,
          `${target} の tolerance 本数が IR 出力数と違う`,
        );

        // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
        const expectedKeys = [
          ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
          ...parsed.graph.outputs.map((_, index) => `output.${index}`),
        ].sort();
        assertEquals([...io.tensors.keys()].sort(), expectedKeys, `${ioFile} のテンソルキー`);

        // 記号次元は golden の入力 shape の実長から束縛される（明示 bindings を渡さない）。
        // ケースごとに長さが違う（テキスト系と duration は T = 7 / 13 / 22 / 29 / 144 / 404、
        // speaker は S = 2 / 6 / 31 / 187 / 750、dit は S = 2 / 25 / 750）ので、宣言上限
        // （テキスト系 Tmax = 512 = export_irodori.py の SYM_MAX / speaker Smax = 750 =
        // speaker_sym_max / dit Smax = 750 = dit_sym_max — 帯マスクと RoPE 表の焼き付け点）に
        // 依存した実装はここで値か shape が壊れる。speaker と dit は**上限そのもの**を踏む
        // ケース（ref-max / dit-cond-max）を持たせてある（テキスト系の最長は 404）。
        //
        // dit の `mask` は**派生次元** `S+1519`（ADR 0046）で宣言されており、束縛源は
        // `x_t` の次元 1 だけ。bindSymbols の 2 巡目が `mask` の実 shape（S+1519）を
        // 評価し直して照合するので、派生形の評価が壊れればここで落ちる。
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
            const where = `${target}/${caseName} output.${index} ('${name}')`;
            const declared = parsed.graph.values[name].dtype;
            assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
            assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
            const report = compareTensors(
              outputs[name],
              ioTensor(io, view, declared),
              tolerances[index],
            );
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
