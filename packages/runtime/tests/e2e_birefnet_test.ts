// 実重みの BiRefNet 系（背景抜き / salient object segmentation）の実 GPU golden E2E
// （ADR 0005 の段 3）。
//
// tiny golden（tests/e2e_golden_test.ts）が「op 契約の被覆」を、SigLIP2
// （tests/e2e_siglip2_test.ts）が「単一ベクトル出力の画像系」を受け持つのに対し、こちらは
// **画素ごとの出力を持つ画像系**（`[1,1,S,S]` の 1,048,576 要素マップ）を受け持つ。対象は
// `outputs/series/<系列>/`（重み + 焼いた定数で 1 系列 964MB のためリポジトリ管理外 —
// `.gitignore` の `outputs/`）。生成は `tools/export-recipes/birefnet/export.py`（コマンドは
// 各系列の {@link Series.generate} がそのまま正本）。
//
// 系列は**モデル × 解像度ごとに 1 本**（下の {@link SERIES}）。BiRefNet_HR（上流の
// 高解像度チェックポイント）と Lucida（その fine-tune — 構造は完全に同一で重みだけが違う）を
// 別系列として実走する。shifted-window マスクも H/W padding のゼロ定数も解像度依存の定数と
// して焼かれるので、解像度が変われば別のグラフになる。解像度は **1024²** と **2048²**（本家
// handler の General-HR）の 2 通りで、どちらも {@link SERIES} に入り、tolerance は系列ごとに
// 実測済み（2048² は RTX 3080 Ti で全緑）。ただし 2048² は Intel Arc B570 の機では GPU を使う
// テストを止めている（{@link HELD_SERIES} — decoder 末尾の 1 dispatch がドライバのジョブ上限を
// 超えてプロセスごと落ちるため）。
//
// **格納 dtype も系列の軸**（ADR 0113）: f32 系列に加えて `--dtype f16` の系列
// （`<モデル>-<解像度>-f16`）を別系列として実走する。f16 系列の golden は丸めた重みで採った torch
// 出力なので、突合に出るのはここでも**ランタイムの数値誤差だけ**（量子化誤差は export 時の
// `quality.json` が持つ）。系列 root の取り違え（f32 の席に f16 資産・その逆）は数値では検出
// できない（tolerance が同桁なので互いの資産を通す — ADR 0027 / 0029）ので、容器の**圧縮格納
// dtype の集合**を系列の宣言と突き合わせる検査（GPU 不要）を別に持つ。
//
// **許容誤差は系列ごとに独立して実測する**（`SERIES` の各行が自分の tolerance を持つ）。
// 共有すると、片方を測り直したときにもう片方が黙って緩む — 2 系列は同じ構造でも logit の
// 値域が桁で違う（実測: BiRefNet_HR の \|ref\| 上端 64.1 に対し Lucida は 1078.1）。
//
// グラフは 1 本で、出力も **sigmoid 前の logit `[1,1,S,S]`** 1 本だけ（マットの α は
// `sigmoid` を掛けたホスト側の値）。入力は**正規化済みの** `pixel_values f32 [1,3,S,S]`
// 1 本で、記号次元は無い。
//
// ## golden の 2 群（合成画像 + 実画像）— どちらも残す
//
// 入力は**どちらの群も golden に焼かれた `pixel_values` そのもの**（ビット同一）なので、
// 突合に出るのは**ランタイムの数値誤差だけ**。2 群に分けて持つのは、踏む分布が違うから:
//
// - **合成画像 4 ケース**（`checker` / `disc` / `noise` / `ramp`）: 値域の端や勾配を踏むぶん
//   数値回帰の検出が鋭い。`disc` は暗い背景に明るい円を置いた顕著物体で、幾何の判別
//   （円内 logit 平均 > 円外）を実 GPU 出力に掛ける土台にもなっている。
// - **実画像 4 ケース**（`photo-*`）: 自然画像の分布点でのランタイム忠実度。tolerance は群
//   ごと・系列ごとに独立に実測から導く（{@link HR_REAL_TOLERANCE}）。
//
// ## TS 前処理を含む鎖は、2 つの門の**合成**で持つ
//
// 「PNG を渡したら Python と同じマットが返る」という鎖は、このテスト単独ではなく
// `packages/models/tests/e2e_birefnet_real_test.ts` の**入力側 parity 門**（同じ PNG から
// Python と同じ `pixel_values` が出る — 入力差 ≤ 1e-6）と、本テストの**golden 入力での
// 忠実度**の合成で持つ。前処理をここで通さないのは依存方向のため（runtime のテストから
// models の実装を相対 import するのは逆向き）。分けた副産物として、落ちたときに前処理と推論の
// どちらが動いたのかがテストの名前で分かる。
//
// 意味の判別のうち**実画像側**（顕著物体のある 2 枚の前景比が無い 2 枚を上回る）と、その
// ついでに書くマット PNG も同じ理由で models 側にある（`encodePng` が models の実装で、
// 入力も TS 前処理を通したものだから）。**合成画像側の幾何判別**（`disc`）は golden の入力
// だけで完結するのでここに残る。
//
// 資産が無い環境では**明示 SKIP** する（系列ごとに独立）。実画像の群も独立に SKIP する
// （`--real-images` を付けずに emit した資産では合成 4 ケースしか無い）。ADR 0005 の
// 「全 SKIP は明示 FAIL」門番（tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見て
// おり、この SKIP とは独立。逆に資産が**中途半端に**（ケース欠け）存在する場合は SKIP ではなく
// FAIL にする（下の「資産の完全性」テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  type CodecLayout,
  codecLayout,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type SafetensorsFile,
  type Tensor,
} from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { assertAdapterMatchesEnvironment, ENVIRONMENT } from "./helpers/environment.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { openResults, runRecordedCase } from "./helpers/results.ts";
import { modelPresent, openSeriesContainer } from "./helpers/container-files.ts";
import { seriesGraph } from "./helpers/series-graphs.ts";

/**
 * **BiRefNet_HR / 合成画像**ケース（入力が golden とビット同一）の突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,1024,1024]`）:
 *
 * | ケース  | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------- | ------- | ------- | ------------ | ------------------ |
 * | checker | 2.38e-5 | 3.04e-6 | 11.677       | 3.50               |
 * | disc    | 1.26e-4 | 7.69e-3 | 23.123       | 5.36e-4            |
 * | noise   | 8.30e-5 | 1.00e-5 | 15.302       | 3.16               |
 * | ramp    | 2.11e-4 | 6.73e-1 | 10.765       | 2.58e-4            |
 *
 * atol 1e-3 は実測最悪 2.11e-4（ramp）の約 4.7 倍。
 *
 * **rtol は 0**。この出力は 0 を跨ぐ logit の地図で、`disc` / `ramp` のように前景と背景の
 * 境界を持つケースでは \|ref\| が 2.6e-4 まで薄く落ちる（境界画素は定義上 logit ≈ 0）。
 * rtol を主役にすると境界のところで判定が発散する — 実測 maxRel 0.673（ramp）の要素も、
 * 絶対誤差は 1.7e-4 でしかない。
 *
 * 誤差の出所は SigLIP2 と同じ（fma 融合・linear / conv の縮約順序が torch と違う・超越関数の
 * 実装差）だが、値域が広い（\|ref\| 上端 23.1）ぶん絶対量は 1 桁大きい。相対量で見ると
 * 2.11e-4 / 10.8 ≈ 2.0e-5 で、実画像側（1.09e-3 / 64.1 ≈ 1.7e-5 — {@link HR_REAL_TOLERANCE}）と
 * 同じ桁に揃う。
 *
 * 実装バグ（deform_conv2d のオフセット取り違え・窓マスクの位相ずれ・upsample の軸違い・
 * BatchNorm の per-channel 定数の並び違い）の誤差は出力の値域と同じ O(1)〜O(20) で、この
 * 閾値の 4 桁以上上に出る。
 */
const HR_SYNTHETIC_TOLERANCE: Tolerance = { atol: 1e-3, rtol: 0 };

/**
 * **BiRefNet_HR / 実画像**ケース（`photo-*`）の突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,1024,1024]`）:
 *
 * | ケース          | TS 前処理入力の maxAbs | \|ref\| 上端 | **golden 入力の maxAbs** |
 * | --------------- | ---------------------- | ------------ | ------------------------ |
 * | photo-corridor  | 4.77e-5                | 12.729       | 4.01e-5                  |
 * | photo-landscape | 5.25e-5                | 12.662       | 3.24e-5                  |
 * | photo-portrait  | 1.78e-3                | 64.094       | 1.09e-3                  |
 * | photo-street    | 8.58e-5                | 16.069       | 9.63e-5                  |
 *
 * この門が入力に使うのは**右端の列と同じ golden の `pixel_values`**（TS 前処理を通す鎖は
 * models 側へ分かれた — モジュール docstring の「2 つの門の合成」）。実効の実測最悪は
 * 1.09e-3（photo-portrait）で、**atol 5e-3 は据え置き**（約 4.6 倍の margin。左の列で導いた
 * ときの 2.8 倍より広くなっただけで、締め直すには実資産のある環境で測り直す必要がある）。
 *
 * 左右の列がほぼ同じ桁なのが**帰属の証拠**: 実画像側で誤差が大きいのは前処理の差ではなく
 * **値域**で、photo-portrait は logit が 64.1 まで振れる（顕著物体が画面の 55% を占め、内部の
 * 飽和が深い）。相対量で見ると 1.09e-3 / 64.1 ≈ 1.7e-5 で、合成画像側（≈2.0e-5）と同じ桁。
 * **それでも定数を共有しない** — 片方を測り直したときにもう片方が黙って緩む。
 *
 * **rtol は 0**（理由は {@link HR_SYNTHETIC_TOLERANCE} と同じ — 0 を跨ぐ logit の地図）。
 *
 * **画像の差し替え**（生成台本を回し直して golden を採り直していない）はこの門では見えない —
 * その sha256 突合は `packages/models/tests/e2e_birefnet_real_test.ts` が名指しで持つ。
 */
const HR_REAL_TOLERANCE: Tolerance = { atol: 5e-3, rtol: 0 };

/**
 * **BiRefNet_HR 2048² / 合成画像**ケースの突合に使う許容誤差（系列 `birefnet-hr-2048` —
 * recipe パッチ ⑨ + ADR 0093 で実行段が通るようになった 2026-09-05 に新設）。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,2048,2048]`・RTX 3080 Ti）:
 *
 * | ケース  | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------- | ------- | ------- | ------------ | ------------------ |
 * | checker | 4.29e-5 | 4.92e-6 | 12.121       | 3.61               |
 * | disc    | 8.60e-4 | 6.33e-3 | 47.904       | 1.06e-3            |
 * | noise   | 5.72e-5 | 6.62e-6 | 16.170       | 2.51               |
 * | ramp    | 1.15e-4 | 6.51e-3 | 10.696       | 6.84e-4            |
 *
 * atol 4e-3 は実測最悪 8.60e-4（disc）の約 4.7 倍。1024² の 1e-3 より広いのは**値域**（disc の
 * \|ref\| 上端が 23.1 → 47.9 と 2 倍）で、相対量は 8.60e-4 / 47.9 ≈ 1.8e-5 と 1024² の 2.0e-5 と
 * 同じ桁。**rtol は 0**（理由は {@link HR_SYNTHETIC_TOLERANCE} と同じ）。二値マスクの不一致は
 * 4 ケースとも 0。実画像 golden は 2048² では採らない（コーパスが 1024² で resize が恒等に
 * ならない — 実画像の門は 1024² 系列が持つ）。
 */
const HR_2048_SYNTHETIC_TOLERANCE: Tolerance = { atol: 4e-3, rtol: 0 };

/**
 * **Lucida / 合成画像**ケースの突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,1024,1024]`）:
 *
 * | ケース  | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------- | ------- | ------- | ------------ | ------------------ |
 * | checker | 1.53e-5 | 2.68e-6 | 13.554       | 3.34               |
 * | disc    | 1.11e-4 | 2.83e-3 | 17.400       | 1.19e-3            |
 * | noise   | 5.29e-5 | 1.85e-2 | 13.574       | 1.76e-4            |
 * | ramp    | 4.41e-5 | 2.89e-3 | 13.210       | 3.04e-4            |
 *
 * atol 5e-4 は実測最悪 1.11e-4（disc）の約 4.5 倍。**rtol は 0**（理由は
 * {@link HR_SYNTHETIC_TOLERANCE} と同じ — 0 を跨ぐ logit の地図）。
 *
 * BiRefNet_HR より 1 段小さいのは、合成画像に対する Lucida の応答が浅い（顕著物体を見つけ
 * られず値域が広がらない）ため。**両系列で同じ定数を使わない**のはこの非対称のためでもある。
 */
const LUCIDA_SYNTHETIC_TOLERANCE: Tolerance = { atol: 5e-4, rtol: 0 };

/**
 * **Lucida / 実画像**ケース（`photo-*`）の突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,1024,1024]`）:
 *
 * | ケース          | TS 前処理入力の maxAbs | \|ref\| 上端 | **golden 入力の maxAbs** |
 * | --------------- | ---------------------- | ------------ | ------------------------ |
 * | photo-corridor  | 1.25e-4                | 12.950       | 5.34e-5                  |
 * | photo-landscape | 1.00e-4                | 13.711       | 8.77e-5                  |
 * | photo-portrait  | 9.80e-3                | **1078.080** | 4.30e-3                  |
 * | photo-street    | 8.77e-4                | 210.791      | 1.05e-3                  |
 *
 * この門が入力に使うのは右端の列と同じ golden の `pixel_values`。実効の実測最悪は 4.30e-3
 * （photo-portrait）で、**atol 3e-2 は据え置き**（約 7.0 倍の margin）。
 *
 * BiRefNet_HR の実画像門（5e-3）より 1 桁緩いのは**値域の差がそのまま出ている**だけで、
 * 精度が落ちているわけではない — 相対量は 4.30e-3 / 1078.1 ≈ 4.0e-6 で、BiRefNet_HR の
 * 1.7e-5 より**小さい**。fine-tune された Lucida は前景の確信が桁で強く出る（logit 1078 =
 * sigmoid で 1 との差が f64 でも表せない飽和）。
 */
const LUCIDA_REAL_TOLERANCE: Tolerance = { atol: 3e-2, rtol: 0 };

/**
 * **Lucida 2048² / 合成画像**ケースの突合に使う許容誤差（系列 `lucida-2048` — 2026-09-05 新設）。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,2048,2048]`・RTX 3080 Ti）:
 *
 * | ケース  | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------- | ------- | ------- | ------------ | ------------------ |
 * | checker | 3.48e-5 | 4.65e-6 | 13.710       | 2.86               |
 * | disc    | 2.86e-4 | 8.44e-1 | 16.182       | 3.05e-5            |
 * | noise   | 5.34e-5 | 1.06e-5 | 13.084       | 9.35e-1            |
 * | ramp    | 1.39e-4 | 3.89e-5 | 12.868       | 1.43               |
 *
 * atol 1.5e-3 は実測最悪 2.86e-4（disc）の約 5.2 倍。1024² の 5e-4 より広いのは 2048² で disc の
 * 境界画素が増え、\|ref\| が 3.05e-5 まで薄く落ちるところで絶対誤差が積むため（maxRel 0.844 は
 * その境界画素 — 絶対誤差は 2.9e-4 でしかない）。**rtol は 0**（理由は
 * {@link HR_SYNTHETIC_TOLERANCE} と同じ）。二値マスクの不一致は 4 ケースとも 0。実画像 golden は
 * 2048² では採らない（{@link HR_2048_SYNTHETIC_TOLERANCE} と同じ理由）。
 */
const LUCIDA_2048_SYNTHETIC_TOLERANCE: Tolerance = { atol: 1.5e-3, rtol: 0 };

/**
 * **BiRefNet_HR の f16 系列**（`birefnet-hr-{1024,2048}-f16` — ADR 0113）の許容誤差。値は f32 系列の
 * {@link HR_SYNTHETIC_TOLERANCE} / {@link HR_REAL_TOLERANCE} / {@link HR_2048_SYNTHETIC_TOLERANCE}
 * と**同じ**。
 *
 * 根拠: 上流 BiRefNet_HR の checkpoint は f16 なので f16 への丸めが恒等で、**golden は f32 系列と
 * ビット一致**する（export 時の品質の門が 8 ケースで強制し、1024² は 2026-09-26 に f32 系列の
 * golden とも入出力のビット一致を突合済み）。GPU 側も「重みの f32 値が同一・`unpack2x16float` は
 * 厳密・縮約順は格納で分岐しない」ので f32 系列と同じ出力になる見込み（推測・未実測 — 実 GPU で
 * 2 系列の出力を突き合わせて確かめる）。
 *
 * 値が同じでも**定数は分ける**（モジュール docstring の「系列ごとに独立」— 片方を測り直したときに
 * もう片方が黙って動かないように）。
 */
const HR_F16_SYNTHETIC_TOLERANCE: Tolerance = { atol: 1e-3, rtol: 0 };
const HR_F16_REAL_TOLERANCE: Tolerance = { atol: 5e-3, rtol: 0 };
const HR_F16_2048_SYNTHETIC_TOLERANCE: Tolerance = { atol: 4e-3, rtol: 0 };

/**
 * **Lucida の f16 系列**（`lucida-{1024,2048}-f16` — ADR 0113）の許容誤差。
 *
 * 1024² は実測から導いた（Intel Arc B570・2026-09-26・`atol=rtol=0` の素の突合・出力 1 本）:
 *
 * | ケース         | maxAbs   | maxRel   |
 * | -------------- | -------- | -------- |
 * | checker        | 2.29e-5  | 3.87e-6  |
 * | disc           | 8.97e-5  | 3.38e-3  |
 * | noise          | 3.77e-5  | 1.72e-1  |
 * | ramp           | 5.34e-4  | 2.35e-1  |
 * | photo-portrait | 7.02e-3  | 4.46e-2  |
 * | photo-street   | 1.28e-3  | 5.46e-2  |
 * | photo-corridor | 3.48e-5  | 4.78e-6  |
 * | photo-landscape| 4.72e-5  | 7.16e-3  |
 *
 * 合成の atol 2.5e-3 は最悪 5.34e-4（ramp）の約 4.7 倍、実画像の atol 3e-2 は最悪 7.02e-3（portrait）の
 * 約 4.3 倍（f32 系列と同じ導出法）。f32 系列（ramp 1.1e-4 級）より合成の最悪が 1 桁大きいのは、丸めた
 * 重みで採った golden との実装誤差が f16 の値の並びで別の丸め境界を踏むため（推測 — 量子化誤差そのものは
 * golden に含まれ、ここには出ない）。
 *
 * TODO: 2048² は **B570 では測れない**（`HELD_SERIES` — deform_conv2d のジョブ上限）。f32 系列の
 * {@link LUCIDA_2048_SYNTHETIC_TOLERANCE} と同じ値を仮置きし、走れる機で `atol=rtol=0` の素の突合から
 * 導出して置き換える。
 */
const LUCIDA_F16_SYNTHETIC_TOLERANCE: Tolerance = { atol: 2.5e-3, rtol: 0 };
const LUCIDA_F16_REAL_TOLERANCE: Tolerance = { atol: 3e-2, rtol: 0 };
const LUCIDA_F16_2048_SYNTHETIC_TOLERANCE: Tolerance = { atol: 1.5e-3, rtol: 0 };

/**
 * 二値マスク（`logit > 0` = `sigmoid > 0.5`）が torch と食い違ってよい画素の割合。
 *
 * この門が出力側の tolerance と別に要るのは、**成果物がマスクだから**。tolerance は「値が
 * 近い」しか言わず、境界の画素（logit ≈ 0）は近いままいくらでも符号が反転しうる。逆にここ
 * だけでは値の回帰を捉えられない（飽和域の誤差は符号を変えない）ので、両方を持つ。
 *
 * 実測は **f32 の 2 系列 × 8 ケースとも 0 / 1,048,576**（合成 4 + 実画像 4）。f16 の 4 系列
 * （ADR 0113）は実測待ちで、同じ上限を仮に掛けている。0 をそのまま門に
 * しないのは、境界画素の符号が別のバックエンドやドライバで動きうるため — 1e-4 は 1024² で
 * 104 画素に相当し、マットの見た目には出ない量。実装バグ側は数万〜数十万画素が反転するので、
 * この閾値の 3 桁以上上に出る。**系列で共有する**のは、これが数値の量ではなく「マスクとしての
 * 判断が一致する」という同じ 1 つの主張だから（両系列とも実測 0 で、緩める理由が片方にも
 * 無い）。
 */
const MASK_DISAGREEMENT_LIMIT = 1e-4;

/** 実走する 1 系列（モデル × 解像度）。 */
type Series = {
  /** `outputs/series/` 直下のディレクトリ名（`birefnet.export.default_out_dir` の綴り）。 */
  readonly name: string;
  /** SKIP 時にそのまま貼れる生成コマンド。 */
  readonly generate: string;
  /** 合成画像ケースの許容誤差（**系列ごとに独立実測** — モジュール docstring）。 */
  readonly tolerance: Tolerance;
  /** 実画像ケースの許容誤差（同上）。 */
  readonly realTolerance: Tolerance;
  /**
   * 容器の**圧縮格納 dtype の集合**として宣言されているべきもの（f32 系列は空 — ADR 0029 決定 2 の
   * 形）。系列 root の取り違えと `--dtype` の付け忘れは数値では見えないので、これだけが区別する。
   */
  readonly compressedStorage: readonly CompressedStorage[];
};

/** 圧縮格納の語彙のうち、この family の系列が持ちうるもの（段 1 は f16 まで — ADR 0113）。 */
type CompressedStorage = "f16";

/**
 * 格納検査で数える圧縮格納か（素の格納 = f32 と i32 の添字表**以外の全部**）。列挙ではなく除外で
 * 書くのは、codec の語彙に layout が増えたとき、新しい圧縮格納を黙って数え漏らさないため。
 */
const isCompressedLayout = (layout: CodecLayout): boolean => layout !== "f32" && layout !== "i32";

const EXPORT_PREFIX =
  "cd tools/export-recipes && uv run --group birefnet python -m birefnet.export";

/**
 * 実走する系列（モデル × 解像度）。**列挙結果ではなくここで固定する** — 列挙だけに頼ると
 * 生成を一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。2048² は実画像 golden を
 * 持たないので実画像ケースは SKIP になる（`realTolerance` は 1024² の値を形式上共有するが読まれない）。
 */
const SERIES: readonly Series[] = [
  {
    name: "birefnet-hr-1024",
    generate: `${EXPORT_PREFIX} --real-images`,
    tolerance: HR_SYNTHETIC_TOLERANCE,
    realTolerance: HR_REAL_TOLERANCE,
    compressedStorage: [],
  },
  {
    name: "birefnet-hr-2048",
    generate: `${EXPORT_PREFIX} --resolution 2048`,
    tolerance: HR_2048_SYNTHETIC_TOLERANCE,
    realTolerance: HR_REAL_TOLERANCE,
    compressedStorage: [],
  },
  {
    name: "lucida-1024",
    generate: `${EXPORT_PREFIX} --model-dir <リポ>/inputs/birefnet/lucida --real-images`,
    tolerance: LUCIDA_SYNTHETIC_TOLERANCE,
    realTolerance: LUCIDA_REAL_TOLERANCE,
    compressedStorage: [],
  },
  {
    name: "lucida-2048",
    generate: `${EXPORT_PREFIX} --model-dir <リポ>/inputs/birefnet/lucida --resolution 2048`,
    tolerance: LUCIDA_2048_SYNTHETIC_TOLERANCE,
    realTolerance: LUCIDA_REAL_TOLERANCE,
    compressedStorage: [],
  },
  {
    name: "birefnet-hr-1024-f16",
    generate: `${EXPORT_PREFIX} --dtype f16 --real-images`,
    tolerance: HR_F16_SYNTHETIC_TOLERANCE,
    realTolerance: HR_F16_REAL_TOLERANCE,
    compressedStorage: ["f16"],
  },
  {
    name: "birefnet-hr-2048-f16",
    generate: `${EXPORT_PREFIX} --dtype f16 --resolution 2048`,
    tolerance: HR_F16_2048_SYNTHETIC_TOLERANCE,
    realTolerance: HR_F16_REAL_TOLERANCE,
    compressedStorage: ["f16"],
  },
  {
    name: "lucida-1024-f16",
    generate:
      `${EXPORT_PREFIX} --model-dir <リポ>/inputs/birefnet/lucida --dtype f16 --real-images`,
    tolerance: LUCIDA_F16_SYNTHETIC_TOLERANCE,
    realTolerance: LUCIDA_F16_REAL_TOLERANCE,
    compressedStorage: ["f16"],
  },
  {
    name: "lucida-2048-f16",
    generate:
      `${EXPORT_PREFIX} --model-dir <リポ>/inputs/birefnet/lucida --dtype f16 --resolution 2048`,
    tolerance: LUCIDA_F16_2048_SYNTHETIC_TOLERANCE,
    realTolerance: LUCIDA_F16_REAL_TOLERANCE,
    compressedStorage: ["f16"],
  },
];

/** 2048² 系列を Intel Arc B570 で止める理由（{@link HELD_SERIES} の 2 行が共有する）。 */
const B570_JOB_TIMEOUT =
  "decoder 末尾の deform_conv2d 1 dispatch（出力 [1,256,1024,1024]・1024² の 1.66 s から ≈ 6.6 s の" +
  "見込み）が Linux xe ドライバの compute ジョブ上限 5 s を超えて device lost になり、Deno はそれを" +
  "例外にせずプロセスごと panic する（後続の検証まで止まる — docs/limitations.md「BiRefNet 系」節）。" +
  "deform_conv2d の分割か高速化で解消したらこの行を消す";

/**
 * 環境キーごとに **GPU を使うテストを止める**系列。外側のキーは {@link SERIES} の系列名、内側は
 * **環境キー**（`<ランタイム>-<アダプタ名 slug>` — ADR 0106 決定 2・参照値の行と同じ流儀）。
 *
 * 行を環境キーごとに持つのは、**止める理由がその機にしか無い**ため。全機共通で止めると、走れる
 * 機（RTX 3080 Ti では 2048² も全緑）の検証まで黙って消える。行がある機でも**資産の完全性
 * テスト（GPU 不要）は走らせる** — 止めるのは device を触るテスト（golden 突合・幾何判別）だけ。
 *
 * この SKIP は ADR 0005 の「全 SKIP は明示 FAIL」門番（tests/gpu_gate_test.ts — GPU アダプタの
 * 有無だけを見る）とは独立で、**行がある環境だけに効く**。GPU 無しの機は環境キーを持たないので
 * どの行にも当たらない（そこでは元から GPU テストが SKIP される）。止めた系列は登録時に
 * `console.warn` で 1 度名乗る（無音の SKIP にしない）。
 *
 * MUST: 行を足すのは、その機で**走らせるとプロセスごと落ちる**（= 後続の検証まで道連れにする）
 * 場合に限り、根拠と「解消したらこの行を消す」を理由文に書く。数値が合わない系列を止める
 * 場所ではない（それは赤のまま直す）。
 *
 * - `birefnet-hr-2048` / `lucida-2048` の `deno-intel-graphics-bmg-g21`（Intel Arc B570・Linux xe
 *   ドライバ）: docs/limitations.md「BiRefNet 系」節（2026-09-20 実測・裁定 2026-09-26）。
 * - 同じ 2 系列の f16 版（`-f16`）も同じ機で止める — 落ちる dispatch は deform_conv2d で、その重みは
 *   f16 系列でも f32 格納のまま（適格外）なので、格納 dtype を変えても同じ 1 dispatch が走る。
 */
const HELD_SERIES: Readonly<
  Record<string, Readonly<Record<string, { readonly reason: string }>>>
> = {
  "birefnet-hr-2048": { "deno-intel-graphics-bmg-g21": { reason: B570_JOB_TIMEOUT } },
  "lucida-2048": { "deno-intel-graphics-bmg-g21": { reason: B570_JOB_TIMEOUT } },
  "birefnet-hr-2048-f16": { "deno-intel-graphics-bmg-g21": { reason: B570_JOB_TIMEOUT } },
  "lucida-2048-f16": { "deno-intel-graphics-bmg-g21": { reason: B570_JOB_TIMEOUT } },
};

/** この走行の機で止める理由（無ければ `undefined` = 止めない）。 */
const heldReason = (series: Series): string | undefined => {
  const environment = ENVIRONMENT.key;
  if (environment === undefined) return undefined;
  if (!Object.hasOwn(HELD_SERIES, series.name)) return undefined;
  const byEnvironment = HELD_SERIES[series.name];
  if (!Object.hasOwn(byEnvironment, environment)) return undefined;
  return byEnvironment[environment].reason;
};

const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);
const MODEL_FILE = "model.krm";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

const seriesRoot = (series: Series): URL => new URL(`${series.name}/`, SERIES_PARENT);

/**
 * 生成されているはずの**合成画像**ケース。正本は `birefnet/export.py` の
 * `build_cases`（モデル軸に依らず同じ 4 枚）。
 */
const EXPECTED_CASES = ["checker", "disc", "noise", "ramp"] as const;

/**
 * **実画像**ケース（`--real-images` を付けた emit だけが持つ）。正本は `birefnet/export.py` の
 * `REAL_CASES`。ここが要るのは golden のケース名だけで、元になった PNG との対応は
 * `packages/models/tests/e2e_birefnet_real_test.ts` が持つ（このテストは PNG を読まない —
 * モジュール docstring の「2 つの門の合成」）。
 */
const REAL_CASES = ["photo-portrait", "photo-landscape", "photo-corridor", "photo-street"] as const;

/**
 * 幾何の判別に使う合成ケースと、その円の半径（画像の短辺を 1 とした比）。正本は
 * `birefnet/export.py` の `DISC_CASE` / `DISC_RADIUS`（あちらは torch 出力に同じ式を掛ける）。
 */
const DISC_CASE = "disc";
const DISC_RADIUS = 0.3;

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

const discoverCases = (root: URL): readonly string[] =>
  listDir(root)
    .filter((entry) =>
      entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
    )
    .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
    .sort();

const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/** golden の入力を宣言 dtype の view で組む（記号次元が無いので明示 bindings も不要）。 */
const goldenInputs = (parsed: PreparedModel, io: SafetensorsFile): Record<string, Tensor> => {
  const inputs: Record<string, Tensor> = {};
  for (const spec of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${spec.name}`);
    assert(view !== undefined, `input.${spec.name} が golden に無い`);
    inputs[spec.name] = ioTensor(io, view, spec.dtype);
  }
  return inputs;
};

/** グラフ入力の静的次元（記号次元は無い — `birefnet/export.py` の `symbol_names=()`）。 */
const staticDim = (parsed: PreparedModel, axis: number): number => {
  const dim = parsed.graph.inputs[0].shape[axis];
  assert(typeof dim === "number", `pixel_values の軸 ${axis} が記号次元 '${String(dim)}'`);
  return dim;
};

/** 二値マスク（前景 = `logit > 0`）が食い違う画素の割合（{@link MASK_DISAGREEMENT_LIMIT}）。 */
const maskDisagreement = (got: Float32Array, expected: Float32Array): number => {
  assertEquals(got.length, expected.length, "マット長");
  let differing = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (got[index] > 0 !== expected[index] > 0) differing += 1;
  }
  return differing / expected.length;
};

/**
 * `disc` ケースの円内（`[S, S]` の bool）。`birefnet/export.py` の `disc_mask` と**同じ式**
 * （あちらが画像を作り、こちらは実 GPU 出力を同じ円で切る）。
 */
const discMask = (size: number): Uint8Array => {
  const axis = new Float64Array(size);
  for (let index = 0; index < size; index += 1) axis[index] = ((index + 0.5) / size) * 2 - 1;
  const mask = new Uint8Array(size * size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      mask[y * size + x] = axis[y] ** 2 + axis[x] ** 2 <= DISC_RADIUS ** 2 ? 1 : 0;
    }
  }
  return mask;
};

/** 円の内 / 外それぞれの logit 平均（{@link discMask} で切る）。 */
const discMeans = (logits: Float32Array, size: number): { inside: number; outside: number } => {
  const mask = discMask(size);
  let inside = 0;
  let insideCount = 0;
  let outside = 0;
  let outsideCount = 0;
  for (let index = 0; index < logits.length; index += 1) {
    if (mask[index] === 1) {
      inside += logits[index];
      insideCount += 1;
    } else {
      outside += logits[index];
      outsideCount += 1;
    }
  }
  return { inside: inside / insideCount, outside: outside / outsideCount };
};

/** 1 系列ぶんの資産の状態（登録時点で必要なので同期列挙する）。 */
type Discovery = {
  readonly cases: readonly string[];
  readonly realCases: readonly string[];
  /** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
  readonly available: boolean;
  /** 実画像の群（`--real-images` を付けずに emit した資産には無い）。 */
  readonly realAvailable: boolean;
  /**
   * **何か 1 つでも**残っているか（完全性テストの SKIP 述語 — Codex 波 H 指摘 H-02）。
   * golden が全滅してモデルだけ残った欠損は `available` では偽になり、`ignore: !available`
   * だと完全性テスト自身が SKIP される — 欠損を FAIL にする述語は「完全に空」でだけ寝てよい。
   */
  readonly anyPresent: boolean;
};

const realNames = new Set<string>(REAL_CASES);

const discover = (series: Series): Discovery => {
  const root = seriesRoot(series);
  const discovered = discoverCases(root);
  const realCases = discovered.filter((name) => realNames.has(name));
  const available = discovered.length > 0;
  return {
    cases: discovered.filter((name) => !realNames.has(name)),
    realCases,
    available,
    realAvailable: available && realCases.length > 0,
    anyPresent: available || modelPresent(new URL(MODEL_FILE, root)),
  };
};

const DISCOVERED: ReadonlyMap<string, Discovery> = new Map(
  SERIES.map((series) => [series.name, discover(series)]),
);

const discoveryOf = (series: Series): Discovery => {
  const found = DISCOVERED.get(series.name);
  if (found === undefined) throw new Error(`系列 ${series.name} の列挙が無い`);
  return found;
};

for (const series of SERIES) {
  const found = discoveryOf(series);
  const held = heldReason(series);
  if (held !== undefined) {
    console.warn(`[karume] ${series.name} はこの環境（${ENVIRONMENT.key}）で SKIP: ${held}`);
  }
  if (!found.available) {
    console.warn(
      `[karume] ${seriesRoot(series).pathname} に export 済み資産が無いため実重み BiRefNet ` +
        `E2E（${series.name}）を SKIP する（重みがリポジトリ管理外）。生成: ${series.generate}`,
    );
  } else if (!found.realAvailable) {
    console.warn(
      `[karume] ${series.name} の実画像ケースを SKIP する（golden ${found.realCases.length}/` +
        `${REAL_CASES.length} 本）。生成: ${series.generate}`,
    );
  }
}

/**
 * 環境キーの書式（`helpers/environment.ts` の `environmentKey` が作る綴り — `<ランタイム>-<slug>`、
 * slug は小文字英数字を `-` 1 つで繋いだもの・両端に `-` なし）。
 */
const ENVIRONMENT_KEY_FORMAT = /^(deno|chrome)-[a-z0-9]+(-[a-z0-9]+)*$/;

Deno.test("BiRefNet 環境別の SKIP 表: 行は実走する系列を指し、内側のキーは環境キーの書式", () => {
  // 系列名の綴り違いは行を黙って効かなくする（止めたはずの系列が走ってプロセスごと落ちる）。
  const names = new Set(SERIES.map((series) => series.name));
  assertEquals(Object.keys(HELD_SERIES).filter((name) => !names.has(name)), []);
  // 環境キーの綴り違い（大文字・商標記号・空白の残り）も同じく行を黙って効かなくする —
  // `environmentKey` はこの書式の外を作らないので、外れた行はどの機にも当たらない。
  const malformed = Object.entries(HELD_SERIES).flatMap(([series, byEnvironment]) =>
    Object.keys(byEnvironment)
      .filter((key) => !ENVIRONMENT_KEY_FORMAT.test(key))
      .map((key) => `${series} / ${key}`)
  );
  assertEquals(malformed, []);
});

/**
 * 決着と実測の置き場（`outputs/verify/<環境キー>/<日付>_birefnet-golden/` — 消して安全）。
 *
 * 系列名を `<family>-golden` にするのは、models 側の e2e（`birefnet`）が同じ根へ書くため
 * （同じ席に 2 モジュールが書くと互いの `cases` を上書きする）。
 */
const results = openResults("birefnet-golden");

for (const series of SERIES) {
  const found = discoveryOf(series);
  const root = seriesRoot(series);
  /** 容器の中のグラフ名（表は helpers/series-graphs.ts の 1 本 — 門番と同じ正本から引く）。 */
  const graphName = seriesGraph(series.name);
  /** この機で GPU テストを止める系列か（{@link HELD_SERIES} — 資産の完全性テストには掛けない）。 */
  const held = heldReason(series) !== undefined;

  Deno.test({
    name: `BiRefNet 資産: ${series.name} — 期待するケースとモデル本体が揃っている`,
    // 完全に空の環境だけ「生成していない」として SKIP。**何か 1 つでも**あれば欠けは FAIL
    //（モデルだけ残って golden が全滅した欠損も拾う — `Discovery.anyPresent` の JSDoc）。
    ignore: !found.anyPresent,
    fn: () => {
      assertEquals(found.cases, [...EXPECTED_CASES], `${root.pathname} の合成画像 golden ケース`);
      // 実画像は**任意だが全部か 0 か**（`--real-images` を付けた emit は 4 本まとめて書く）。
      // 部分的な欠けを SKIP に丸めると、採り直しの途中で落ちた資産が黙って通る。
      assert(
        found.realCases.length === 0 || found.realCases.length === REAL_CASES.length,
        `${root.pathname} の実画像 golden が ${found.realCases.length}/${REAL_CASES.length} 本` +
          `（採り直す: ${series.generate}）`,
      );
      assert(modelPresent(new URL(MODEL_FILE, root)), `${MODEL_FILE} が無い`);
    },
  });

  Deno.test({
    name: `BiRefNet 格納: ${series.name} — 容器の圧縮格納 dtype の集合が系列の宣言と一致する`,
    // GPU 不要（容器を開いて束縛表を読むだけ）。held の機でも走らせる — 止める理由は dispatch。
    ignore: !modelPresent(new URL(MODEL_FILE, root)),
    fn: async () => {
      // 系列と資産の格納 dtype が一致する（root 取り違え / `--dtype` の付け忘れの唯一の検出器 —
      // {@link Series.compressedStorage}）。本数ではなく**集合**で見るのは、f16 系列に別の圧縮
      // 格納が混ざる形を「圧縮が 1 本以上ある」で通さないため（ADR 0029 決定 2・e2e_sbv2 と同じ形）。
      const parsed = prepareContainer(
        await openSeriesContainer(new URL(MODEL_FILE, root)),
        graphName,
      );
      const compressed = [
        ...new Set(
          Object.values(parsed.graph.initializers)
            .flatMap((initializer) =>
              initializer.storage === undefined ? [] : [codecLayout(initializer.storage.codec)]
            )
            .filter(isCompressedLayout),
        ),
      ].sort();
      assertEquals(
        compressed,
        [...series.compressedStorage].sort(),
        `${series.name}: 圧縮格納 dtype の集合が系列と食い違う（採り直す: ${series.generate}）`,
      );
    },
  });

  /**
   * 突合を回すケース。合成と実画像で**入力の作り方は同じ**（どちらも golden の
   * `pixel_values`）で、違うのは踏む分布と、そこから独立に導いた tolerance だけ。
   */
  const goldenCases = [
    ...found.cases.map((name) => ({
      name,
      label: "golden 突合",
      tolerance: series.tolerance,
      ignore: !found.available,
    })),
    ...found.realCases.map((name) => ({
      name,
      label: "実画像 golden 突合",
      tolerance: series.realTolerance,
      ignore: !found.realAvailable,
    })),
  ];

  for (const entry of goldenCases) {
    const caseName = entry.name;
    /** ケース ID（系列が違えば同じケース名があるので組で持つ）。 */
    const caseId = `${series.name}/${caseName}`;
    Deno.test({
      name: `BiRefNet ${entry.label}: ${series.name} / ${caseName}（実 GPU / torch CPU 期待値）`,
      ignore: entry.ignore || !GPU_AVAILABLE || held,
      fn: async () => {
        await runRecordedCase(results, { id: caseId }, async ({ measurements }) => {
          const [opened, ioBytes] = await Promise.all([
            openSeriesContainer(new URL(MODEL_FILE, root)),
            readBuffer(root, `${IO_PREFIX}${caseName}${IO_SUFFIX}`),
          ]);
          const parsed = prepareContainer(opened, graphName);
          const io = parseSafetensors(ioBytes);

          // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
          const expectedKeys = [
            ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
            ...parsed.graph.outputs.map((_, index) => `output.${index}`),
          ].sort();
          assertEquals(
            [...io.tensors.keys()].sort(),
            expectedKeys,
            "io.safetensors のテンソルキー",
          );

          const inputs = goldenInputs(parsed, io);

          const gpu = await acquireGpu();
          // MUST: device の破棄は adapter 検査と `createSession` の失敗も通す。取り逃がすと、
          // その走行の残りが破棄されない device を抱えたまま進み、後続が OOM で赤くなる
          // （known-issues の遅延解放）。
          try {
            // 参照値・結果をこの機の行として残す経路なので、キーを採ったアダプタと実行アダプタの
            // 同一性をここで見る（複数 GPU の機で取り違えると、別の機の帯で測ることになる）。
            assertAdapterMatchesEnvironment(gpu);
            const session = await parsed.createContainerSession(gpu);
            try {
              const outputs = await session.run(inputs);
              assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

              const [name] = parsed.graph.outputs;
              const view = io.tensors.get("output.0");
              assert(view !== undefined, "output.0 が golden に無い");
              const where = `${series.name} / ${caseName} output.0 ('${name}')`;
              const declared = parsed.graph.values[name].dtype;
              assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
              assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
              const expected = ioTensor(io, view, declared);
              const report = compareTensors(outputs[name], expected, entry.tolerance);
              measurements.push({
                output: name,
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance: entry.tolerance,
                stage: "karume",
              });
              assert(report.pass, `${where}: ${formatAllclose(report)}`);

              // 値の近さとは別に、**マスクとしての判断**が torch と一致すること。
              assert(
                outputs[name].dtype === "f32" && expected.dtype === "f32",
                `${where}: f32 でない`,
              );
              const disagreement = maskDisagreement(outputs[name].data, expected.data);
              assert(
                disagreement <= MASK_DISAGREEMENT_LIMIT,
                `${where}: 二値マスクの不一致 ${(disagreement * 100).toFixed(4)}%` +
                  `（上限 ${MASK_DISAGREEMENT_LIMIT * 100}%）`,
              );
            } finally {
              await session.dispose();
            }
          } finally {
            gpu.destroy();
          }
        });
      },
    });
  }

  Deno.test({
    name: `BiRefNet 幾何判別: ${series.name} — disc の円内 logit 平均が円外を上回る`,
    ignore: !found.available || !GPU_AVAILABLE || held,
    fn: async () => {
      // golden 突合だけだと「期待値と合っている」ことしか言えず、マットとして意味のある出力かは
      // 別問題（一様に潰れた出力は期待値も同じく潰れていれば通ってしまう）。ここは**画像の中の
      // 既知の幾何**（円）で切るので、出力が一様なら平均が並んで落ちる。閾値は置かない
      // （順序そのものが検査対象 — `birefnet/export.py` の `_sanity` と同じ形で、あちらは
      // torch 側に掛かっている）。実測は円内 / 円外が BiRefNet_HR で +10.96 / −9.64、
      // Lucida で +3.37 / −9.69（合成画像に対する応答の深さが系列で違う）。
      const opened = await openSeriesContainer(new URL(MODEL_FILE, root));
      const ioBytes = await readBuffer(root, `${IO_PREFIX}${DISC_CASE}${IO_SUFFIX}`);
      const parsed = prepareContainer(opened, graphName);
      const io = parseSafetensors(ioBytes);
      const size = staticDim(parsed, 3);
      assertEquals(size, staticDim(parsed, 2), "円の判別は正方形の入力を前提にする");

      const gpu = await acquireGpu();
      // MUST: device の破棄は `createSession` の失敗も通す。取り逃がすと、その走行の残りが
      // 破棄されない device を抱えたまま進み、後続が OOM で赤くなる（known-issues の遅延解放）。
      try {
        const session = await parsed.createContainerSession(gpu);
        try {
          const [name] = parsed.graph.outputs;
          const output = (await session.run(goldenInputs(parsed, io)))[name];
          assert(output.dtype === "f32", `disc のマットの dtype が ${output.dtype}`);
          assertEquals(output.shape, [1, 1, size, size], "マットの形");
          const { inside, outside } = discMeans(output.data, size);
          assert(
            inside > outside,
            `disc の円内 logit 平均 ${inside} が円外 ${outside} 以下 — 顕著物体を分離できていない`,
          );
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }
    },
  });
}
