// gpu_ops_test.ts の fixture（OpCase 型・決定的な入力列ジェネレータ・op 別のケース配列）。
// テスト本体（runCase / checkAll / Deno.test 群）は gpu_ops_test.ts に残し、ここは
// 「checkAll に渡すデータ」だけを持つ — 依存は helpers → src の一方向のみ。

import type { RefTensor } from "../../src/reference/ops.ts";
import { fill } from "./graph.ts";

export type OpCase = {
  readonly name: string;
  readonly op: string;
  readonly inputs: readonly RefTensor[];
  readonly outShape: readonly number[];
  /** 既定は入力と同型（cast だけ出力 dtype が変わる）。 */
  readonly outDtype?: "f32" | "i32" | "bool";
  readonly attrs?: Record<string, unknown>;
};

/** 負値・0・正値を跨ぐ決定的な列。 */
export const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
/** log / sqrt の定義域内に収める列。 */
export const POSITIVE = (i: number): number => 0.125 + (i % 17) * 0.5;
/** div の除数（0 を含まない）。 */
const NONZERO = (i: number): number => ((i % 7) + 1) * 0.5;

const UNARY_INPUTS: readonly (readonly [string, (index: number) => number])[] = [
  ["neg", SIGNED],
  ["abs", SIGNED],
  ["exp", SIGNED],
  ["log", POSITIVE],
  ["sqrt", POSITIVE],
  ["sin", SIGNED],
  ["tanh", SIGNED],
  ["sigmoid", SIGNED],
  ["relu", SIGNED],
  ["gelu", SIGNED],
  ["gelu_tanh", SIGNED],
];

export const UNARY_CASES: readonly OpCase[] = UNARY_INPUTS.map(([op, generator]) => ({
  name: `unary ${op} [3,5]`,
  op,
  inputs: [fill([3, 5], generator)],
  outShape: [3, 5],
}));

export const BINARY_CASES: readonly OpCase[] = [
  {
    name: "add 同形 [4,3]",
    op: "add",
    inputs: [fill([4, 3], SIGNED), fill([4, 3], POSITIVE)],
    outShape: [4, 3],
  },
  {
    name: "sub 右詰め broadcast [4,3] - [3]",
    op: "sub",
    inputs: [fill([4, 3], SIGNED), fill([3], POSITIVE)],
    outShape: [4, 3],
  },
  {
    name: "mul 両側 broadcast [4,1] * [1,3]",
    op: "mul",
    inputs: [fill([4, 1], SIGNED), fill([1, 3], POSITIVE)],
    outShape: [4, 3],
  },
  {
    name: "div rank3 broadcast [2,3,4] / [3,1]",
    op: "div",
    inputs: [fill([2, 3, 4], SIGNED), fill([3, 1], NONZERO)],
    outShape: [2, 3, 4],
  },
  {
    name: "add スカラ broadcast [5] + []",
    op: "add",
    inputs: [fill([5], SIGNED), fill([], () => 2.5)],
    outShape: [5],
  },
];

/** 0/1 を跨ぐ決定的な整数列（mask 経路の実測形に合わせる）。 */
const MASK = (i: number): number => (i % 3 === 0 ? 0 : 1);
/** 負値・0・正値を跨ぐ整数列。 */
export const INTEGERS = (i: number): number => (i % 11) - 5;
/** 切り捨てが round / floor と区別できる小数列（±.5 と ±.7 を含む）。 */
const FRACTIONS = (i: number): number => ((i % 9) - 4) * 0.6;

/**
 * i32 / bool の実行経路（ADR 0009）。解禁したのは実測グラフの mask 経路に出る形だけ:
 * mask 外積 `mul(i32, i32)` / `1 - mask` の `sub(i32)` / 真偽化 cast / `bitwise_not`。
 */
export const DTYPE_CASES: readonly OpCase[] = [
  {
    // mask 外積 [4,1] × [1,4]（実測は unsqueeze 経由の [1,1,T,1]×[1,1,1,T]）
    name: "mul i32 の外積 broadcast [4,1] * [1,4]",
    op: "mul",
    inputs: [fill([4, 1], MASK, "i32"), fill([1, 4], MASK, "i32")],
    outShape: [4, 4],
  },
  {
    name: "sub i32 同形 [3,5]",
    op: "sub",
    inputs: [fill([3, 5], INTEGERS, "i32"), fill([3, 5], MASK, "i32")],
    outShape: [3, 5],
  },
  {
    name: "bitwise_not bool [2,6]",
    op: "bitwise_not",
    inputs: [fill([2, 6], MASK, "bool")],
    outShape: [2, 6],
  },
  {
    name: "cast f32 → i32（0 方向切り捨て）[3,3]",
    op: "cast",
    inputs: [fill([3, 3], FRACTIONS)],
    outShape: [3, 3],
    outDtype: "i32",
    attrs: { to: "i32" },
  },
  {
    name: "cast i32 → bool（x != 0）[3,4]",
    op: "cast",
    inputs: [fill([3, 4], INTEGERS, "i32")],
    outShape: [3, 4],
    outDtype: "bool",
    attrs: { to: "bool" },
  },
  {
    name: "cast bool → f32（0/1 の重み化）[2,5]",
    op: "cast",
    inputs: [fill([2, 5], MASK, "bool")],
    outShape: [2, 5],
    outDtype: "f32",
    attrs: { to: "f32" },
  },
  {
    name: "cast i32 → f32 [7]",
    op: "cast",
    inputs: [fill([7], INTEGERS, "i32")],
    outShape: [7],
    outDtype: "f32",
    attrs: { to: "f32" },
  },
  {
    name: "cast f32 → bool（x != 0）[6]",
    op: "cast",
    inputs: [fill([6], FRACTIONS)],
    outShape: [6],
    outDtype: "bool",
    attrs: { to: "bool" },
  },
];

/**
 * M1-P3 波3 の数理 op 群（sdp の spline / dec の leaky_relu — recon §2）。
 *
 * MUST: 「向き」を持つ op は非対称な入力で踏む。where の分岐・比較の不等号・cumsum の
 * 累積方向はいずれも shape も dtype も変えずに誤れるので、値でしか検出できない。
 */
export const MATH_CASES: readonly OpCase[] = [
  {
    // 定義域 x > -1。小さい x（補正が効く領域）は専用テストが精度まで見る
    name: "log1p [3,5]",
    op: "log1p",
    inputs: [fill([3, 5], POSITIVE)],
    outShape: [3, 5],
  },
  {
    // SIGNED は ±4.5 を跨ぐので、下側 / 内側 / 上側の 3 分岐を全て踏む
    name: "clamp [3,5] min=-1.5 max=2.25",
    op: "clamp",
    inputs: [fill([3, 5], SIGNED)],
    outShape: [3, 5],
    attrs: { min: -1.5, max: 2.25 },
  },
  {
    name: "clamp 潰す形 min == max",
    op: "clamp",
    inputs: [fill([6], SIGNED)],
    outShape: [6],
    attrs: { min: 1.5, max: 1.5 },
  },
  {
    // 片側 clamp（ADR 0017）。SIGNED は ±4.5 を跨ぐので下側 / 上側の両分岐を踏む
    name: "clamp_min [3,5] min=-1.5",
    op: "clamp_min",
    inputs: [fill([3, 5], SIGNED)],
    outShape: [3, 5],
    attrs: { min: -1.5 },
  },
  {
    // 実測形（チャネル L2 正規化の clamp(min=eps)）— 正値だけの列で下限が効かない側も踏む
    name: "clamp_min [4] min=1e-12（実測の eps 形）",
    op: "clamp_min",
    inputs: [fill([4], (i) => [0, -1, 1e-20, 2][i])],
    outShape: [4],
    attrs: { min: 1e-12 },
  },
  {
    // 実測の slope 2 種（ADR 0015）。params 経由なので同じパイプラインで値だけが変わる
    name: "leaky_relu slope=0.1 [3,5]",
    op: "leaky_relu",
    inputs: [fill([3, 5], SIGNED)],
    outShape: [3, 5],
    attrs: { negative_slope: 0.1 },
  },
  {
    name: "leaky_relu slope=0.01 [3,5]",
    op: "leaky_relu",
    inputs: [fill([3, 5], SIGNED)],
    outShape: [3, 5],
    attrs: { negative_slope: 0.01 },
  },
  {
    // SIGNED は i % 13 == 6 でちょうど 0 を取る = ge と gt の分かれ目を必ず踏む
    name: "ge_scalar value=0（等値点を含む）",
    op: "ge_scalar",
    inputs: [fill([3, 5], SIGNED)],
    outShape: [3, 5],
    outDtype: "bool",
    attrs: { value: 0 },
  },
  {
    name: "gt_scalar value=0（等値点を含む）",
    op: "gt_scalar",
    inputs: [fill([3, 5], SIGNED)],
    outShape: [3, 5],
    outDtype: "bool",
    attrs: { value: 0 },
  },
  {
    name: "le_scalar value=0（等値点を含む）",
    op: "le_scalar",
    inputs: [fill([3, 5], SIGNED)],
    outShape: [3, 5],
    outDtype: "bool",
    attrs: { value: 0 },
  },
  {
    // searchsorted の形（inputs[…,None] >= bl）を縮めた右詰め broadcast
    name: "ge [4,1] >= [3]（右詰め broadcast）",
    op: "ge",
    inputs: [fill([4, 1], SIGNED), fill([3], (i) => (i - 1) * 0.75)],
    outShape: [4, 3],
    outDtype: "bool",
  },
  {
    name: "bitwise_and bool [2,6] & [6]",
    op: "bitwise_and",
    inputs: [fill([2, 6], MASK, "bool"), fill([6], (i) => (i % 2 === 0 ? 1 : 0), "bool")],
    outShape: [2, 6],
    outDtype: "bool",
  },
  {
    // 分岐の取り違えが値に出るよう、a は正・b は負の値域で埋める
    name: "where cond[1,4] ? a[3,4] : b[3,1]",
    op: "where",
    inputs: [
      fill([1, 4], MASK, "bool"),
      fill([3, 4], POSITIVE),
      fill([3, 1], (i) => -10 - i),
    ],
    outShape: [3, 4],
    // 出力は条件（スロット 0）ではなく**値の側**と同型（契約表の写像 bool → f32）
    outDtype: "f32",
  },
  {
    name: "where 同形 [2,3,4]",
    op: "where",
    inputs: [
      fill([2, 3, 4], (i) => (i % 3 === 0 ? 1 : 0), "bool"),
      fill([2, 3, 4], SIGNED),
      fill([2, 3, 4], POSITIVE),
    ],
    outShape: [2, 3, 4],
    outDtype: "f32",
  },
  {
    // bool の sum は**真の個数**（出力 i32 — 契約表の写像）
    name: "sum bool [6,10] → i32 のカウント",
    op: "sum",
    inputs: [fill([6, 10], MASK, "bool")],
    outShape: [6],
    outDtype: "i32",
    attrs: { dim: 1 },
  },
  {
    // 行長が workgroup サイズ（256）を超える → 1 スレッドが複数要素を畳む経路（bool 版）
    name: "sum bool [3,700] → i32",
    op: "sum",
    inputs: [fill([3, 700], MASK, "bool")],
    outShape: [3],
    outDtype: "i32",
    attrs: { dim: 1 },
  },
  {
    // 累積方向が値に出る非対称な列（等差なら逆向きでも同じ値になりうる形を避ける）
    name: "cumsum [4,7]",
    op: "cumsum",
    inputs: [fill([4, 7], SIGNED)],
    outShape: [4, 7],
    attrs: { dim: 1 },
  },
  {
    name: "cumsum rank1 [10]",
    op: "cumsum",
    inputs: [fill([10], POSITIVE)],
    outShape: [10],
    attrs: { dim: 0 },
  },
  {
    // 1 スレッドが 1 行を逐次で走る形なので、行長を伸ばして累算の長さも踏む
    name: "cumsum 行長 300 rank3 [2,3,300]",
    op: "cumsum",
    inputs: [fill([2, 3, 300], SIGNED)],
    outShape: [2, 3, 300],
    attrs: { dim: 2 },
  },
  {
    // 相対位置埋め込みの 4D 化（recon §2）— f32 の expand を解禁した形
    name: "expand f32 [1,4,1] → [1,4,3]",
    op: "expand",
    inputs: [fill([1, 4, 1], SIGNED)],
    outShape: [1, 4, 3],
  },
  {
    name: "expand f32 rank 増 [3] → [2,4,3]",
    op: "expand",
    inputs: [fill([3], POSITIVE)],
    outShape: [2, 4, 3],
  },
];

/**
 * matmul（64×64 レジスタタイル — src/kernels/gemm.ts）。
 *
 * MUST: タイル辺 64 を**跨ぐ**ケースを持つ。`[7,5]×[5,3]` のような小形は全て「1 タイル未満」に
 * 潰れ、タイル原点・quad ガード・K タイル端数のどれも踏まない。
 * MUST: v4（`k%4==0 && n%4==0`）とスカラの**両変種**を踏む。片方だけでは端数形状の
 * フォールバックが黙って壊れても気づけない。
 */
export const MATMUL_CASES: readonly OpCase[] = [
  {
    // タイル 16 の端数（m/n/k いずれもタイル境界に揃わない）
    name: "matmul [7,5] × [5,3]",
    op: "matmul",
    inputs: [fill([7, 5], SIGNED), fill([5, 3], POSITIVE)],
    outShape: [7, 3],
  },
  {
    name: "matmul [32,16] × [16,32]",
    op: "matmul",
    inputs: [fill([32, 16], SIGNED), fill([16, 32], POSITIVE)],
    outShape: [32, 32],
  },
  {
    name: "matmul [1,64] × [64,1]",
    op: "matmul",
    inputs: [fill([1, 64], SIGNED), fill([64, 1], POSITIVE)],
    outShape: [1, 1],
  },
  {
    // v4 経路のタイル境界: n=68 は最終タイルの有効 quad が 16 中 1（quad ガードを外すと
    // 隣接行を潰す）、k=20 は 4 の倍数だが 16 の倍数でない（K タイル端数の 0 埋め）、
    // m=65 は行タイルを 2 枚跨ぐ。m/n/k は互いに違う長さ。
    name: "matmul v4 タイル境界 [65,20] × [20,68]",
    op: "matmul",
    inputs: [fill([65, 20], SIGNED), fill([20, 68], POSITIVE)],
    outShape: [65, 68],
  },
  {
    // スカラ変種のタイル境界（k=19 / n=23 が 4 の倍数でない・m=70 は行タイル 2 枚）
    name: "matmul スカラ変種 タイル境界 [70,19] × [19,23]",
    op: "matmul",
    inputs: [fill([70, 19], SIGNED), fill([19, 23], NONZERO)],
    outShape: [70, 23],
  },
  {
    // MUST: v4 判定は k と n の**両方**を見る。片方だけの判定が素通りすると、vec4 束縛と
    // 実バイト数が食い違って例外なしの誤値になる。k のみ / n のみ 4 の倍数を対で持つ。
    name: "matmul k のみ 4 の倍数 [66,20] × [20,19]",
    op: "matmul",
    inputs: [fill([66, 20], SIGNED), fill([20, 19], POSITIVE)],
    outShape: [66, 19],
  },
  {
    name: "matmul n のみ 4 の倍数 [66,19] × [19,20]",
    op: "matmul",
    inputs: [fill([66, 19], SIGNED), fill([19, 20], POSITIVE)],
    outShape: [66, 20],
  },
];

/**
 * bmm（rank-3 バッチ matmul）。
 *
 * MUST: **B / M / K / N を全て違う長さ**にしたケースを持つ（ACTIVE_DESIGN の Pitfalls）。
 * バッチ stride を隣の次元の積で組む誤りは、正方形や 2 軸が同じ長さの形では数値に出ない。
 */
export const BMM_CASES: readonly OpCase[] = [
  {
    // 実測形 [16,T,64] × [16,64,T] を縮めた非対称形（B=3 / M=5 / K=7 / N=2）
    name: "bmm 非対称 [3,5,7] × [3,7,2]",
    op: "bmm",
    inputs: [fill([3, 5, 7], SIGNED), fill([3, 7, 2], POSITIVE)],
    outShape: [3, 5, 2],
  },
  {
    // タイル 16 の端数をバッチ 2 枚で踏む（M / N / K いずれもタイル境界に揃わない）
    name: "bmm タイル端数 [2,17,19] × [2,19,23]",
    op: "bmm",
    inputs: [fill([2, 17, 19], SIGNED), fill([2, 19, 23], NONZERO)],
    outShape: [2, 17, 23],
  },
  {
    // 1 タイルを跨ぐ K（縮約が複数タイルに割れる経路）とバッチ 4 枚
    name: "bmm 縮約が複数タイル [4,3,40] × [4,40,5]",
    op: "bmm",
    inputs: [fill([4, 3, 40], SIGNED), fill([4, 40, 5], POSITIVE)],
    outShape: [4, 3, 5],
  },
  {
    name: "bmm バッチ 1 枚 [1,4,6] × [1,6,3]",
    op: "bmm",
    inputs: [fill([1, 4, 6], SIGNED), fill([1, 6, 3], POSITIVE)],
    outShape: [1, 4, 3],
  },
  {
    // MUST: v4 経路のバッチ base は **quad 単位**（`m * k4`）。要素単位のまま組む誤りは
    // batch ≥ 2 でしか出ず、B/M/K/N が 1 つでも同じ長さだと数値に現れないことがある。
    // M=68 で行タイル 2 枚・K=20 で K タイル端数・N=12 で列タイル 1 枚未満。
    name: "bmm v4 タイル境界 [3,68,20] × [3,20,12]",
    op: "bmm",
    inputs: [fill([3, 68, 20], SIGNED), fill([3, 20, 12], POSITIVE)],
    outShape: [3, 68, 12],
  },
  {
    // スカラ変種でも同じ罠を踏む（base が要素単位なのは正しいが、3 本とも base 経由か）
    name: "bmm スカラ変種 タイル境界 [2,70,19] × [2,19,23]",
    op: "bmm",
    inputs: [fill([2, 70, 19], SIGNED), fill([2, 19, 23], NONZERO)],
    outShape: [2, 70, 23],
  },
];

/**
 * 融合 attention（ADR 0023）。契約は rank-4 head-first で、1 ノード = 3 dispatch。
 *
 * MUST: **B / H / M / N / D を全て違う長さ**にしたケースを持つ。カーネルは B と H を 1 本の
 * バッチ軸へ畳むので、実測形（B=1）では軸の取り違えが値に出ない（設計 recon §4.6 の検出限界）。
 * MUST: **D ≠ 128** を持つ（実測は DiT の 128 と VAE の 384 だけ）。D は WGSL に展開されない
 * ので変種は増えないが、`gemmUsesVec4` の踏み分けは D で決まる。
 * MUST: **端数 M / N** を持つ。実モデルは M,N ∈ {512,1024,4096} で全て 64 の倍数なので、
 * タイル端数の経路はユニットテストが唯一の検出器（ADR 0022 が記録した穴と同型）。
 */
const attentionCase = (
  name: string,
  [b, h, m, n, d]: readonly [number, number, number, number, number],
  options: {
    readonly query?: (index: number) => number;
    readonly key?: (index: number) => number;
    readonly scale?: number;
    /** k / v の head 数（GQA — 省略時は H = Hkv の従来形。ADR 0067 決定 1）。 */
    readonly kvHeads?: number;
  } = {},
): OpCase => {
  const kvHeads = options.kvHeads ?? h;
  return {
    name,
    op: "attention",
    inputs: [
      fill([b, h, m, d], options.query ?? SIGNED),
      fill([b, kvHeads, n, d], options.key ?? POSITIVE),
      fill([b, kvHeads, n, d], NONZERO),
    ],
    outShape: [b, h, m, d],
    // 既定は契約どおりの半スケール（torch の `√(1/√D)`）。
    attrs: { scale: options.scale ?? Math.fround(Math.sqrt(1 / Math.sqrt(d))) },
  };
};

export const ATTENTION_CASES: readonly OpCase[] = [
  // B=2 / H=3 / M=5 / N=11 / D=7 — 5 軸とも違う長さ（軸の取り違えが必ず値に出る）
  attentionCase("attention 全異 [2,3,5,7] × [2,3,11,7]", [2, 3, 5, 11, 7]),
  // タイル端数（M % 64 ≠ 0 / N % 64 ≠ 0）+ D % 4 ≠ 0 でスカラ変種
  attentionCase("attention 端数 [3,1,17,13] × [3,1,19,13]", [3, 1, 17, 19, 13]),
  // v4 経路で行タイルを跨ぐ（M=68 > 64）
  attentionCase("attention v4 タイル跨ぎ [1,2,68,12]", [1, 2, 68, 20, 12]),
  // ① が v4・③ がスカラに落ちる混成（変種の踏み分けが段ごとに違うことの固定）
  attentionCase("attention 段ごとに変種が違う [2,2,9,6]", [2, 2, 9, 8, 6]),
  // D=64（conditioner 級）と D=384（VAE decoder 級 — flash 型が載らない D も段階融合は通る）
  attentionCase("attention D=64 [1,3,7,64]", [1, 3, 7, 5, 64]),
  attentionCase("attention D=384 [1,1,3,384]", [1, 1, 3, 5, 384]),
  // N が 1（縮約が 1 要素 — softmax が恒等になる退化形）
  attentionCase("attention N=1 [2,2,3,4]", [2, 2, 3, 1, 4]),
  {
    // MUST: 行統計の amax 減算が外れた瞬間に赤くなる形。q を大きい負値・k を全て正にすると
    // S が −200 級に落ち、素朴 softmax なら exp が f32 で 0 に潰れて 0/0 = NaN になる。
    ...attentionCase("attention 大きい負値 [1,2,4,4]（素朴形は underflow）", [1, 2, 4, 6, 4], {
      query: () => -15,
      key: (i) => 6 + (i % 5) * 0.25,
    }),
    name: "attention 大きい負値 [1,2,4,4]（素朴形は underflow）",
  },
  // 明示 scale（SDPA の `scale` 引数を指定した形 — 既定 1/√D 以外も契約どおり通る）
  attentionCase("attention 明示 scale [2,2,5,4]", [2, 2, 5, 7, 4], { scale: 0.25 }),
  // GQA / MQA（整除 broadcast — ADR 0067 決定 1）。**CPU 参照との突合**で見るのがここの役目で、
  // repeat_kv 実体化との突合（tests/gpu_attention_gqa_test.ts）とは別の検出器
  // （実体化側とカーネルが同じ写像を共有しているぶん、参照実装が独立の証人になる）。
  // MUST: B ≥ 2 の GQA を持つ（`⌊(b·H + h)/r⌋ = b·Hkv + ⌊h/r⌋` の b 項は B=1 では検証されない）。
  attentionCase("attention GQA r=4 [2,8,5,4] × [2,2,7,4]", [2, 8, 5, 7, 4], { kvHeads: 2 }),
  attentionCase("attention MQA r=8 [1,8,3,8] × [1,1,5,8]", [1, 8, 3, 5, 8], { kvHeads: 1 }),
];

/** src の最終次元 D=9 に収まる決定的な添字（恒等でも単調でもない列）。 */
const PICKS = (i: number): number => (i * 5 + 3) % 9;
/** 最終次元 D=4 に収まる添字。 */
const PICKS4 = (i: number): number => (i * 3 + 1) % 4;

/**
 * gather（最終次元固定）。実測は src f32[16,T,512] / index i32[16,T,T]。
 * MUST: 添字は行ごとに違う列を引く列にする（恒等添字だと行オフセットの誤りが出ない）。
 */
export const GATHER_CASES: readonly OpCase[] = [
  {
    name: "gather rank3 src [4,5,9] / index [4,5,6]",
    op: "gather",
    inputs: [fill([4, 5, 9], SIGNED), fill([4, 5, 6], PICKS, "i32")],
    outShape: [4, 5, 6],
  },
  {
    // 出力の最終次元が src より長い（同じ添字を何度引いてもよい）
    name: "gather 列が増える src [3,4] / index [3,7]",
    op: "gather",
    inputs: [fill([3, 4], POSITIVE), fill([3, 7], PICKS4, "i32")],
    outShape: [3, 7],
  },
  {
    name: "gather rank1 src [9] / index [5]",
    op: "gather",
    inputs: [fill([9], SIGNED), fill([5], PICKS, "i32")],
    outShape: [5],
  },
  {
    // 1 workgroup（256 要素）では覆えない大きさ。**grid-stride の縮退はここでは踏まない**
    // （262144 要素 = 1024 workgroup で dispatch 上限の内側なので必要数がそのまま割り当たる）—
    // 縮退経路は tests/gpu_gridstride_test.ts が dispatch 数を絞って直接踏む。
    name: "gather 大きめ src [4096,9] / index [4096,64]",
    op: "gather",
    inputs: [fill([4096, 9], SIGNED), fill([4096, 64], PICKS, "i32")],
    outShape: [4096, 64],
  },
];

export const REDUCE_CASES: readonly OpCase[] = [
  {
    name: "sum [6,10]",
    op: "sum",
    inputs: [fill([6, 10], SIGNED)],
    outShape: [6],
    attrs: { dim: 1 },
  },
  {
    name: "amax [6,10]",
    op: "amax",
    inputs: [fill([6, 10], SIGNED)],
    outShape: [6],
    attrs: { dim: 1 },
  },
  {
    name: "amin [6,10]",
    op: "amin",
    inputs: [fill([6, 10], SIGNED)],
    outShape: [6],
    attrs: { dim: 1 },
  },
  // 行長が workgroup サイズ（256）を超える → 1 スレッドが複数要素を畳む経路
  {
    name: "sum [3,700]",
    op: "sum",
    inputs: [fill([3, 700], SIGNED)],
    outShape: [3],
    attrs: { dim: 1 },
  },
  {
    name: "amax rank3 [2,3,7]",
    op: "amax",
    inputs: [fill([2, 3, 7], SIGNED)],
    outShape: [2, 3],
    attrs: { dim: 2 },
  },
  // rank 1 → rank 0（スカラ出力）
  {
    name: "sum [4] → スカラ",
    op: "sum",
    inputs: [fill([4], SIGNED)],
    outShape: [],
    attrs: { dim: 0 },
  },
  // 最終次元以外の軸（軸 reduce 変種 — 実行カーネルが別物になる）
  {
    name: "sum 軸 1 [5,9,11]",
    op: "sum",
    inputs: [fill([5, 9, 11], SIGNED)],
    outShape: [5, 11],
    attrs: { dim: 1 },
  },
  {
    name: "amax 軸 0 [7,13]",
    op: "amax",
    inputs: [fill([7, 13], SIGNED)],
    outShape: [13],
    attrs: { dim: 0 },
  },
  {
    // 縮約長 > 256（軸変種の slot 走査が 2 周する経路）。中間軸なので inner も 1 でない。
    name: "sum 軸 1 の縮約長 300 [2,300,5]",
    op: "sum",
    inputs: [fill([2, 300, 5], SIGNED)],
    outShape: [2, 5],
    attrs: { dim: 1 },
  },
  {
    // bool の軸 sum（累算器 i32 + 真偽化が軸変種にも入っていること）
    name: "sum bool 軸 1 [3,13,5] → i32",
    op: "sum",
    inputs: [fill([3, 13, 5], MASK, "bool")],
    outShape: [3, 5],
    outDtype: "i32",
    attrs: { dim: 1 },
  },
];

/**
 * argmax（最終次元・rank 保存・出力 i32 — ADR 0068 決定 2）。突合は i32 の**厳密一致**
 * （添字に丸め差は無い）。タイブレーク / NaN / 全 −inf 行の固定挙動は期待値リテラルを持つ
 * 専用門（gpu_ops_test.ts）が見るので、ここは shape と経路の被覆を担う。
 */
export const ARGMAX_CASES: readonly OpCase[] = [
  {
    name: "argmax [6,10]",
    op: "argmax",
    inputs: [fill([6, 10], SIGNED)],
    outShape: [6, 1],
    outDtype: "i32",
  },
  {
    // 行長が workgroup サイズ（256）を超える → 1 スレッドが複数要素を畳み、レーンを跨いだ
    // 木の簡約で index が運ばれる経路。最大値は**行ごとに違う位置**の 2 周目（添字 ≥ 256）に
    // 置く（全行で同じ添字が正解だと「別の行を読む」誤りが値に出ない）。
    name: "argmax [3,700]（走査ループ 2 周目に最大値・行ごとに別位置）",
    op: "argmax",
    inputs: [fill([3, 700], (i) => (i % 700 === 300 + Math.floor(i / 700) * 100 ? 99 : SIGNED(i)))],
    outShape: [3, 1],
    outDtype: "i32",
  },
  {
    name: "argmax rank3 [2,3,7]",
    op: "argmax",
    inputs: [fill([2, 3, 7], SIGNED)],
    outShape: [2, 3, 1],
    outDtype: "i32",
  },
  {
    // rank 1 → [1]（reduce 族はここでスカラへ落ちる — rank 保存との違いが出る形）
    name: "argmax [4] → [1]",
    op: "argmax",
    inputs: [fill([4], (i) => [-1.5, 0.25, 3.5, 3.5][i])],
    outShape: [1],
    outDtype: "i32",
  },
  {
    name: "argmax 最終次元が 1 [5,1]",
    op: "argmax",
    inputs: [fill([5, 1], SIGNED)],
    outShape: [5, 1],
    outDtype: "i32",
  },
];

/**
 * レイアウト op（ADR 0011）。reshape はバッファ別名なので dispatch を 1 本も出さず、
 * permute / expand は strided 実体化コピー 1 カーネル族で実行する。
 * 解禁 dtype は実測どおり（permute は f32、expand は i32 / bool）。
 */
export const LAYOUT_CASES: readonly OpCase[] = [
  {
    name: "reshape [2,3,4] → [6,4]（別名）",
    op: "reshape",
    inputs: [fill([2, 3, 4], SIGNED)],
    outShape: [6, 4],
  },
  {
    name: "reshape i32 [6] → [1,6,1]（軸の挿入）",
    op: "reshape",
    inputs: [fill([6], INTEGERS, "i32")],
    outShape: [1, 6, 1],
  },
  {
    name: "reshape bool [2,3] → [6]（軸の削除）",
    op: "reshape",
    inputs: [fill([2, 3], MASK, "bool")],
    outShape: [6],
  },
  {
    // 実測形 [0,2,1,3]（attention の head 整形）
    name: "permute rank4 [1,5,3,4] dims=[0,2,1,3]",
    op: "permute",
    inputs: [fill([1, 5, 3, 4], SIGNED)],
    outShape: [1, 3, 5, 4],
    attrs: { dims: [0, 2, 1, 3] },
  },
  {
    // 実測形 [0,2,1]（scores の転置）
    name: "permute rank3 [3,5,7] dims=[0,2,1]",
    op: "permute",
    inputs: [fill([3, 5, 7], POSITIVE)],
    outShape: [3, 7, 5],
    attrs: { dims: [0, 2, 1] },
  },
  {
    name: "permute rank2 転置 [4,6] dims=[1,0]",
    op: "permute",
    inputs: [fill([4, 6], SIGNED)],
    outShape: [6, 4],
    attrs: { dims: [1, 0] },
  },
  {
    // MUST: 巡回長 3 以上の並べ替えを 1 本持つ。実測に出る形（[0,2,1,3] / [0,2,1] / [1,0]）は
    // 全て対合（自分自身が逆置換）で、stride 表を逆置換で組む誤りをどれも検出できない。
    name: "permute rank3 の 3 巡回 [2,3,5] dims=[1,2,0]",
    op: "permute",
    inputs: [fill([2, 3, 5], SIGNED)],
    outShape: [3, 5, 2],
    attrs: { dims: [1, 2, 0] },
  },
  {
    name: "permute rank4 の 4 巡回 [2,3,4,5] dims=[1,2,3,0]",
    op: "permute",
    inputs: [fill([2, 3, 4, 5], POSITIVE)],
    outShape: [3, 4, 5, 2],
    attrs: { dims: [1, 2, 3, 0] },
  },
  {
    // gather 添字の実測形 [1,T,T] → [16,T,T]
    name: "expand i32 [1,4,4] → [5,4,4]（先頭軸の複製）",
    op: "expand",
    inputs: [fill([1, 4, 4], INTEGERS, "i32")],
    outShape: [5, 4, 4],
  },
  {
    // conv 経路の bool マスク [1,T,1] → [1,T,C]
    name: "expand bool [1,6,1] → [1,6,9]（最終軸の複製）",
    op: "expand",
    inputs: [fill([1, 6, 1], MASK, "bool")],
    outShape: [1, 6, 9],
  },
  {
    name: "expand i32 rank 増 [3] → [2,4,3]",
    op: "expand",
    inputs: [fill([3], INTEGERS, "i32")],
    outShape: [2, 4, 3],
  },
];

/**
 * レイアウト第 2 群（ADR 0014）— slice / cat / pad / flip。
 *
 * MUST: **開始位置 0 でない slice** と **先頭でない cat の入力**を必ず持つ（offset が 0 の
 * 形だけだと「offset を載せ忘れる / 取り違える」誤りが値に出ない）。
 * MUST: 反転軸は**長さ 3 以上**を 1 本持つ（長さ 2 の反転は off-by-one が対称で消える）。
 * MUST: pad は左右非対称の形を持つ（[w,w] だけだと左右の取り違えが値に出ない）。
 */
export const LAYOUT2_CASES: readonly OpCase[] = [
  {
    // enc_p の stats[:, c:]（後半チャネル）と同型 — 開始位置が 0 でない中間軸の切り出し
    name: "slice 中間軸の後半 [1,6,5] dim=1 [3,6)",
    op: "slice",
    inputs: [fill([1, 6, 5], SIGNED)],
    outShape: [1, 3, 5],
    attrs: { dim: 1, start: 3, end: 6 },
  },
  {
    // spline の bin_locations[..., :-1] と同型（最終次元・末尾を落とす）
    name: "slice 最終次元 [4,7] dim=1 [1,6)",
    op: "slice",
    inputs: [fill([4, 7], POSITIVE)],
    outShape: [4, 5],
    attrs: { dim: 1, start: 1, end: 6 },
  },
  {
    // 先頭軸の切り出し（行の送り幅が変わらない形 — offset だけが効く）
    name: "slice 先頭軸 [5,3] dim=0 [2,4)",
    op: "slice",
    inputs: [fill([5, 3], SIGNED)],
    outShape: [2, 3],
    attrs: { dim: 0, start: 2, end: 4 },
  },
  {
    // rank4 の中間軸（strided params の左詰めを踏まない rank）
    name: "slice rank4 [2,5,3,4] dim=1 [1,4)",
    op: "slice",
    inputs: [fill([2, 5, 3, 4], SIGNED)],
    outShape: [2, 3, 3, 4],
    attrs: { dim: 1, start: 1, end: 4 },
  },
  {
    // coupling reverse の cat([x0,x1], 1) と同型（チャネル軸の連結 — 行が交互に並ぶ）
    name: "cat チャネル軸 [1,3,5] + [1,2,5] dim=1",
    op: "cat",
    inputs: [fill([1, 3, 5], SIGNED), fill([1, 2, 5], POSITIVE)],
    outShape: [1, 5, 5],
    attrs: { dim: 1 },
  },
  {
    // spline の cat 系（最終次元の連結 — 1 要素の帯を先頭に足す形）
    name: "cat 最終次元 [3,1] + [3,4] dim=1",
    op: "cat",
    inputs: [fill([3, 1], NONZERO), fill([3, 4], SIGNED)],
    outShape: [3, 5],
    attrs: { dim: 1 },
  },
  {
    // 3 入力（可変アリティ）— 真ん中の入力は offset も長さも 0 でない
    name: "cat 3 入力 [2,1]+[2,3]+[2,2] dim=1",
    op: "cat",
    inputs: [fill([2, 1], POSITIVE), fill([2, 3], SIGNED), fill([2, 2], NONZERO)],
    outShape: [2, 6],
    attrs: { dim: 1 },
  },
  {
    // 先頭軸の連結（行がそのまま積み上がる形）
    name: "cat 先頭軸 [2,4] + [3,4] dim=0",
    op: "cat",
    inputs: [fill([2, 4], SIGNED), fill([3, 4], POSITIVE)],
    outShape: [5, 4],
    attrs: { dim: 0 },
  },
  {
    // 相対位置 value 側の F.pad(p_attn, [w,w])（w=4）と同型
    name: "pad 対称 [1,2,3,5] [4,4]",
    op: "pad",
    inputs: [fill([1, 2, 3, 5], SIGNED)],
    outShape: [1, 2, 3, 13],
    attrs: { left: 4, right: 4 },
  },
  {
    name: "pad 非対称 [3,4] [2,1]",
    op: "pad",
    inputs: [fill([3, 4], POSITIVE)],
    outShape: [3, 7],
    attrs: { left: 2, right: 1 },
  },
  {
    name: "pad 片側 0 [2,3] [0,3]",
    op: "pad",
    inputs: [fill([2, 3], SIGNED)],
    outShape: [2, 6],
    attrs: { left: 0, right: 3 },
  },
  {
    // flow の Flip（192ch）を縮めた形 — 中間軸・長さ 5（奇数で中央が動かない形も踏む）
    name: "flip 中間軸 [2,5,3] dim=1",
    op: "flip",
    inputs: [fill([2, 5, 3], SIGNED)],
    outShape: [2, 5, 3],
    attrs: { dim: 1 },
  },
  {
    // sdp の Flip（2ch）と同型
    name: "flip 2ch [1,2,6] dim=1",
    op: "flip",
    inputs: [fill([1, 2, 6], POSITIVE)],
    outShape: [1, 2, 6],
    attrs: { dim: 1 },
  },
  {
    name: "flip 最終次元 [3,4] dim=1",
    op: "flip",
    inputs: [fill([3, 4], SIGNED)],
    outShape: [3, 4],
    attrs: { dim: 1 },
  },
  {
    name: "flip 先頭軸 rank4 [4,2,3,2] dim=0",
    op: "flip",
    inputs: [fill([4, 2, 3, 2], SIGNED)],
    outShape: [4, 2, 3, 2],
    attrs: { dim: 0 },
  },
];

/** masked_fill の実測埋め値（f32 の最小有限値 — ADR 0012）。 */
const NEG_F32_MAX = -3.4028234663852886e+38;
/** 素朴 softmax なら f32 で exp が 0 に潰れる領域（safe-softmax の必要性を踏む）。 */
const HUGE_NEGATIVE = (i: number): number => -200 + (i % 7) * 3;
/** 0/1 の bool マスク（3 要素に 1 つ真）。 */
const MASKED = (i: number): number => (i % 3 === 0 ? 1 : 0);
/**
 * safe_softmax 用の `[3,5]` スコア（ADR 0044）。**行 1 が全 -inf**（torch のガードが発火する
 * 行）で、行 0 / 2 は -inf を混ぜた通常行。加算マスク `where(mask, 0, -inf)` を通した後の
 * スコアがちょうどこの形になる。
 */
const EMPTY_ROW_SCORES = (i: number): number => {
  const row = Math.floor(i / 5);
  if (row === 1) return -Infinity;
  return i % 4 === 3 ? -Infinity : SIGNED(i);
};

/**
 * 融合 op 10 本（ADR 0012 / recon §5 + ADR 0015 の conv_transpose1d + ADR 0017 の rms_norm /
 * conv2d + ADR 0044 の safe_softmax）。
 *
 * MUST: 各 op に「軸の長さが全て違う」ケースを 1 本持つ。linear の m/n/k、conv1d の
 * B/Cin/Cout/L/K は積で添字を組むので、2 軸が同じ長さの形では取り違えが数値に出ない。
 */
export const FUSED_CASES: readonly OpCase[] = [
  {
    // 実測は [1,T,1024] × W[1024,1024]。ここは m/n/k を全て違う長さにした縮小形
    name: "linear rank2 [5,7] × W[3,7] + b[3]",
    op: "linear",
    inputs: [fill([5, 7], SIGNED), fill([3, 7], POSITIVE), fill([3], NONZERO)],
    outShape: [5, 3],
  },
  {
    // 先行次元が 2 本（平坦化して GEMM に落とす経路）
    name: "linear rank3 [2,4,6] × W[5,6] + b[5]",
    op: "linear",
    inputs: [fill([2, 4, 6], SIGNED), fill([5, 6], NONZERO), fill([5], SIGNED)],
    outShape: [2, 4, 5],
  },
  {
    // タイル 16 の端数（m/n/k いずれもタイル境界に揃わない）+ 縮約が複数タイルに割れる。
    // n=19 は 4 の倍数でないのでスカラ変種で、共有メモリ転置（列 quad = wc/4・成分 = wc%4）が
    // 1 タイル内で quad を跨ぐ形になる（列ごとに W が違うので入れ替えが数値に出る）。
    name: "linear タイル端数 [17,40] × W[19,40] + b[19]",
    op: "linear",
    inputs: [fill([17, 40], SIGNED), fill([19, 40], POSITIVE), fill([19], NONZERO)],
    outShape: [17, 19],
  },
  {
    // v4 経路のタイル境界: n=68（最終タイルの有効 quad が 16 中 1・共有転置も quad を跨ぐ）/
    // k=20（K タイル端数）/ m=65（行タイル 2 枚）。m/n/k は互いに違う長さ。
    name: "linear v4 タイル境界 [65,20] × W[68,20] + b[68]",
    op: "linear",
    inputs: [fill([65, 20], SIGNED), fill([68, 20], POSITIVE), fill([68], SIGNED)],
    outShape: [65, 68],
  },
  {
    // スカラ変種でタイル辺 64 を跨ぐ（k=37 は 4 の倍数でも 2 の倍数でもない）
    name: "linear スカラ変種 タイル境界 [70,37] × W[23,37] + b[23]",
    op: "linear",
    inputs: [fill([70, 37], SIGNED), fill([23, 37], NONZERO), fill([23], SIGNED)],
    outShape: [70, 23],
  },
  {
    name: "layer_norm [4,10] affine",
    op: "layer_norm",
    inputs: [fill([4, 10], SIGNED), fill([10], POSITIVE), fill([10], SIGNED)],
    outShape: [4, 10],
    attrs: { normalized_shape: [10], eps: 1e-5 },
  },
  {
    // 行長が workgroup サイズ（256）を超える → 1 スレッドが複数要素を畳む経路
    name: "layer_norm 行長 300 rank3 [2,3,300]",
    op: "layer_norm",
    inputs: [fill([2, 3, 300], SIGNED), fill([300], NONZERO), fill([300], SIGNED)],
    outShape: [2, 3, 300],
    attrs: { normalized_shape: [300], eps: 1e-7 },
  },
  {
    // 実測の eps（1e-07）と**分散ちょうど 0 の行**（行ごとに定数）。eps が無ければ
    // 1/sqrt(0) = inf で出力が NaN になる経路で、結果は bias そのものになる。
    name: "layer_norm 分散 0 の行 [3,8] eps=1e-7",
    op: "layer_norm",
    inputs: [
      fill([3, 8], (i) => Math.floor(i / 8) + 1),
      fill([8], POSITIVE),
      fill([8], SIGNED),
    ],
    outShape: [3, 8],
    attrs: { normalized_shape: [8], eps: 1e-7 },
  },
  {
    // ADR 0017: rms_norm は平均を引かない（アリティ 2）。SIGNED は 0 を跨ぐので、
    // layer_norm を写した実装（平均を引く）なら CPU 参照と必ず食い違う。
    name: "rms_norm [4,10] weight",
    op: "rms_norm",
    inputs: [fill([4, 10], SIGNED), fill([10], POSITIVE)],
    outShape: [4, 10],
    attrs: { eps: 1e-6 },
  },
  {
    // 行長が workgroup サイズ（256）を超える → 1 スレッドが複数要素を畳む経路
    name: "rms_norm 行長 300 rank3 [2,3,300]",
    op: "rms_norm",
    inputs: [fill([2, 3, 300], SIGNED), fill([300], NONZERO)],
    outShape: [2, 3, 300],
    attrs: { eps: 1e-6 },
  },
  {
    // **全要素 0 の行**（二乗和 0）— eps が平方根の外にあると 1/sqrt(0) = inf で NaN 化する。
    // 正しい実装ではちょうど 0 が返る（layer_norm の「分散 0 の行」と同型の押さえ）。
    name: "rms_norm 全要素 0 の行 [3,8] eps=1e-6",
    op: "rms_norm",
    inputs: [fill([3, 8], () => 0), fill([8], POSITIVE)],
    outShape: [3, 8],
    attrs: { eps: 1e-6 },
  },
  {
    name: "softmax 最終次元 [4,9]",
    op: "softmax",
    inputs: [fill([4, 9], SIGNED)],
    outShape: [4, 9],
    attrs: { dim: 1 },
  },
  {
    // MUST: 素朴形なら exp(-200) が f32 で 0 に潰れて 0/0 = NaN になる領域。
    // safe-softmax（amax 減算）が外れた瞬間にここが赤くなる。
    name: "softmax 大きい負値 [3,11]（素朴形は underflow）",
    op: "softmax",
    inputs: [fill([3, 11], HUGE_NEGATIVE)],
    outShape: [3, 11],
    attrs: { dim: 1 },
  },
  {
    // 全要素が同じ値の行（masked_fill で全マスクされた行と同じ形）— 一様分布になる
    name: "softmax 全要素同値 [2,6]",
    op: "softmax",
    inputs: [fill([2, 6], () => NEG_F32_MAX)],
    outShape: [2, 6],
    attrs: { dim: 1 },
  },
  {
    name: "softmax 行長 300 rank3 [2,3,300]",
    op: "softmax",
    inputs: [fill([2, 3, 300], SIGNED)],
    outShape: [2, 3, 300],
    attrs: { dim: 2 },
  },
  {
    // safe_softmax（ADR 0044）— -inf を含まない入力では素の softmax と同じ値になる。
    name: "safe_softmax 最終次元 [4,9]（-inf 無し）",
    op: "safe_softmax",
    inputs: [fill([4, 9], SIGNED)],
    outShape: [4, 9],
    attrs: { dim: 1 },
  },
  {
    // MUST: **全要素 -inf の行**を含める。この op の存在理由そのもので、素の softmax なら
    // 0/0 = NaN になる（allclose は NaN を不合格にするので、外れれば必ず赤くなる）。
    // 行 1 が全 -inf / 行 0 と 2 は一部だけ -inf。
    name: "safe_softmax 全 -inf 行 [3,5]",
    op: "safe_softmax",
    inputs: [fill([3, 5], EMPTY_ROW_SCORES)],
    outShape: [3, 5],
    attrs: { dim: 1 },
  },
  {
    // 行長が workgroup サイズ（256）を超える形でも空行判定が効く（① の identity は
    // grid-stride の走査とツリー縮約の両方を通る）。行 1 が全 -inf。
    name: "safe_softmax 行長 300 rank3 [2,3,300]（全 -inf 行つき）",
    op: "safe_softmax",
    inputs: [fill([2, 3, 300], (i) => (Math.floor(i / 300) === 1 ? -Infinity : SIGNED(i)))],
    outShape: [2, 3, 300],
    attrs: { dim: 2 },
  },
  {
    // masked_fill の埋め値（-F32_MAX）は**有限**なので空行ではない — 素の softmax と同じ
    // 一様分布になる（-inf と -F32_MAX を取り違えた実装がここで落ちる）。
    name: "safe_softmax 全 -F32_MAX 行 [2,6]（空行ではない）",
    op: "safe_softmax",
    inputs: [fill([2, 6], () => NEG_F32_MAX)],
    outShape: [2, 6],
    attrs: { dim: 1 },
  },
  {
    // 実測は weight f32[22012,1024] × index i32[1,T]。V/H/添字数を全て違う長さにした縮小形
    name: "embedding weight [7,4] × index [2,3]",
    op: "embedding",
    inputs: [fill([7, 4], SIGNED), fill([2, 3], (i) => (i * 3 + 1) % 7, "i32")],
    outShape: [2, 3, 4],
    attrs: { padding_idx: -1 },
  },
  {
    // padding_idx は forward に効かない（受理して不活性 — ADR 0012）。添字 0 を必ず引く列で
    // 踏み、padding 行を 0 で潰す実装なら CPU 参照と食い違う。
    name: "embedding padding_idx=0 は forward に効かない",
    op: "embedding",
    inputs: [fill([5, 3], POSITIVE), fill([4], (i) => i % 5, "i32")],
    outShape: [4, 3],
    attrs: { padding_idx: 0 },
  },
  {
    // 実測形 mask [1,1,T,T] × x [1,16,T,T]（右詰め broadcast）を縮小したもの
    name: "masked_fill rank4 broadcast x[1,3,4,5] mask[1,1,4,5]",
    op: "masked_fill",
    inputs: [fill([1, 3, 4, 5], SIGNED), fill([1, 1, 4, 5], MASKED, "bool")],
    outShape: [1, 3, 4, 5],
    attrs: { value: NEG_F32_MAX },
  },
  {
    // 実測形 mask [1,T,1024] × x [1,T,1024]（同形・埋め値 0）
    name: "masked_fill 同形 x[2,3,4] mask[2,3,4] value=0",
    op: "masked_fill",
    inputs: [fill([2, 3, 4], SIGNED), fill([2, 3, 4], MASKED, "bool")],
    outShape: [2, 3, 4],
    attrs: { value: 0 },
  },
  {
    // rank を跨ぐ右詰め broadcast（mask の rank が x より低い形）
    name: "masked_fill rank 違い x[3,4,5] mask[5]",
    op: "masked_fill",
    inputs: [fill([3, 4, 5], SIGNED), fill([5], MASKED, "bool")],
    outShape: [3, 4, 5],
    attrs: { value: -1.5 },
  },
  {
    // 実測形（kernel 3 / stride 1 / padding 1）を B/Cin/Cout/L 全て違う長さで踏む
    name: "conv1d [2,3,9] * W[4,3,3] stride=1 padding=1",
    op: "conv1d",
    inputs: [fill([2, 3, 9], SIGNED), fill([4, 3, 3], POSITIVE), fill([4], NONZERO)],
    outShape: [2, 4, 9],
    attrs: { stride: 1, padding: 1, dilation: 1, groups: 1 },
  },
  {
    // padding 無し（出力長が縮む）+ stride 2（params 経路を踏む）
    name: "conv1d [1,2,11] * W[3,2,4] stride=2 padding=0",
    op: "conv1d",
    inputs: [fill([1, 2, 11], SIGNED), fill([3, 2, 4], NONZERO), fill([3], SIGNED)],
    outShape: [1, 3, 4],
    attrs: { stride: 2, padding: 0, dilation: 1, groups: 1 },
  },
  {
    // padding がカーネル長を超える形（両端が padding 域だけを見る出力を持つ）
    name: "conv1d [1,1,3] * W[2,1,3] stride=1 padding=2",
    op: "conv1d",
    inputs: [fill([1, 1, 3], SIGNED), fill([2, 1, 3], POSITIVE), fill([2], NONZERO)],
    outShape: [1, 2, 5],
    attrs: { stride: 1, padding: 2, dilation: 1, groups: 1 },
  },
  // ---- conv 族の拡張（ADR 0015）--------------------------------------------
  {
    // sdp の DDSConv（depthwise groups=C・dilation 3・k=5）の縮小形。groups=Cin=Cout なので
    // 重みの第 2 軸は 1 — グループ跨ぎで読む誤りは「別チャネルの値が混ざる」形で値に出る。
    name: "conv1d depthwise [1,6,17] * W[6,1,5] dilation=3 groups=6",
    op: "conv1d",
    inputs: [fill([1, 6, 17], SIGNED), fill([6, 1, 5], POSITIVE), fill([6], NONZERO)],
    outShape: [1, 6, 17],
    attrs: { stride: 1, padding: 6, dilation: 3, groups: 6 },
  },
  {
    // depthwise の dilation 9（DDSConv の 3 層目）。padding も d·(K−1)/2 = 18 まで伸びる
    name: "conv1d depthwise [1,4,13] * W[4,1,5] dilation=9 groups=4",
    op: "conv1d",
    inputs: [fill([1, 4, 13], SIGNED), fill([4, 1, 5], NONZERO), fill([4], SIGNED)],
    outShape: [1, 4, 13],
    attrs: { stride: 1, padding: 18, dilation: 9, groups: 4 },
  },
  {
    // 中間の groups（1 < g < C）— Cin/Cout ともグループごとに 2ch 以上あり、帯の
    // 先頭オフセットを落とす誤りが値に出る。Cin ≠ Cout の非対称形。
    name: "conv1d grouped [1,6,11] * W[9,2,3] groups=3",
    op: "conv1d",
    inputs: [fill([1, 6, 11], SIGNED), fill([9, 2, 3], POSITIVE), fill([9], NONZERO)],
    outShape: [1, 9, 11],
    attrs: { stride: 1, padding: 1, dilation: 1, groups: 3 },
  },
  {
    // dec の ResBlock1（dilation 5・k=3・full チャネル）の縮小形
    name: "conv1d dilated [1,3,15] * W[5,3,3] dilation=5",
    op: "conv1d",
    inputs: [fill([1, 3, 15], SIGNED), fill([5, 3, 3], NONZERO), fill([5], SIGNED)],
    outShape: [1, 5, 15],
    attrs: { stride: 1, padding: 5, dilation: 5, groups: 1 },
  },
  // ---- conv2d（ADR 0017）------------------------------------------------------
  // MUST: 実装の取り違えは対称な形では数値に出ない。以下の 5 ケースで
  // (a) Cin/Cout とも 2 以上で互いに違う (b) Kh ≠ Kw (c) stride/padding の H/W 非対称
  // (d) dilation > 1 (e) groups > 1（depthwise 含む）を全て踏む。
  {
    // VAE の素の conv2d 実測形（k3 / pad1 / same）。B/Cin/Cout/H/W を全て違う長さにする
    name: "conv2d [2,3,7,9] * W[5,3,3,3] stride=1 padding=1",
    op: "conv2d",
    inputs: [fill([2, 3, 7, 9], SIGNED), fill([5, 3, 3, 3], POSITIVE), fill([5], NONZERO)],
    outShape: [2, 5, 7, 9],
    attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 },
  },
  {
    // Kh ≠ Kw かつ stride / padding が H/W 非対称
    // H: (9 + 2 − 2)/2 + 1 = 5 / W: (11 + 0 − 4)/3 + 1 = 3
    name: "conv2d [1,2,9,11] * W[4,2,2,4] stride=[2,3] padding=[1,0]",
    op: "conv2d",
    inputs: [fill([1, 2, 9, 11], SIGNED), fill([4, 2, 2, 4], NONZERO), fill([4], SIGNED)],
    outShape: [1, 4, 5, 3],
    attrs: { stride: [2, 3], padding: [1, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // dilation も H/W 非対称（H: 12 − 3·2 = 6 / W: 10 − 2·1 = 8）
    name: "conv2d [1,2,12,10] * W[3,2,3,2] dilation=[3,2]",
    op: "conv2d",
    inputs: [fill([1, 2, 12, 10], SIGNED), fill([3, 2, 3, 2], POSITIVE), fill([3], NONZERO)],
    outShape: [1, 3, 6, 8],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [3, 2], groups: 1 },
  },
  {
    // depthwise（groups = Cin = Cout → 重みの第 2 軸は 1）。グループ跨ぎで読む誤りは
    // 「別チャネルの値が混ざる」形で必ず値に出る
    name: "conv2d depthwise [1,6,5,7] * W[6,1,3,3] groups=6",
    op: "conv2d",
    inputs: [fill([1, 6, 5, 7], SIGNED), fill([6, 1, 3, 3], POSITIVE), fill([6], NONZERO)],
    outShape: [1, 6, 5, 7],
    attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 6 },
  },
  {
    // 中間の groups（1 < g < C）で Cin ≠ Cout・Kh ≠ Kw — 帯の先頭オフセットを落とす誤りと
    // カーネル 2 軸の取り違えを 1 本で同時に踏む
    name: "conv2d grouped [1,6,6,8] * W[9,2,3,1] groups=3",
    op: "conv2d",
    inputs: [fill([1, 6, 6, 8], SIGNED), fill([9, 2, 3, 1], NONZERO), fill([9], SIGNED)],
    outShape: [1, 9, 4, 8],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 3 },
  },
  {
    // implicit GEMM（groups == 1 — ADR 0024）の**タイル境界**: m = 70 で行タイル 2 枚、
    // n = Hout·Wout = 72 で列タイル 2 枚、K = 8·3·3 = 72 で K タイル端数（16·4 + 8）。
    // kFlat % 4 == 0 かつ Wout % 4 == 0 かつ stride_w == 1 なので v4 変種を踏む。
    name: "conv2d igemm v4 タイル境界 [1,8,9,8] * W[70,8,3,3] padding=1",
    op: "conv2d",
    inputs: [fill([1, 8, 9, 8], SIGNED), fill([70, 8, 3, 3], POSITIVE), fill([70], NONZERO)],
    outShape: [1, 70, 9, 8],
    attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 },
  },
  {
    // **v4 判定の取り違えの検出器**（ADR 0024 の MUST ②）: Hout = Wout = 2 で N = 4 なので
    // `N % 4` で判定すると v4 が選ばれるが、列 quad が出力行をまたぐため誤値になる。
    // kFlat = 4·1·5 = 20 と stride_w = 1 は v4 の他条件を満たすので、`Wout % 4` だけが
    // 分かれ目になる。実測形（Wout は全て 4 の倍数）では絶対に露見しない。
    name: "conv2d igemm Hout=Wout=2（N%4==0 だが Wout%4≠0）[1,4,2,6] * W[3,4,1,5]",
    op: "conv2d",
    inputs: [fill([1, 4, 2, 6], SIGNED), fill([3, 4, 1, 5], POSITIVE), fill([3], NONZERO)],
    outShape: [1, 3, 2, 2],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // padding がカーネル張りを超える形（両端が padding 域だけを見る出力を持つ）
    name: "conv2d [1,1,2,3] * W[2,1,3,1] padding=[2,0]",
    op: "conv2d",
    inputs: [fill([1, 1, 2, 3], SIGNED), fill([2, 1, 3, 1], POSITIVE), fill([2], NONZERO)],
    outShape: [1, 2, 4, 3],
    attrs: { stride: [1, 1], padding: [2, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // dec の ups 末尾 2 本（up 2 / k 2 / pad 0）。非対称チャネルで重みの転置を固定する
    name: "conv_transpose1d [1,5,7] * W[5,3,2] stride=2 padding=0",
    op: "conv_transpose1d",
    inputs: [fill([1, 5, 7], SIGNED), fill([5, 3, 2], POSITIVE), fill([3], NONZERO)],
    outShape: [1, 3, 14],
    attrs: { stride: 2, padding: 0 },
  },
  {
    // dec の ups 先頭（up 8 / k 16 / pad 4）の縮小形 — 1 出力に複数の k が寄与する形
    name: "conv_transpose1d [1,3,5] * W[3,2,16] stride=8 padding=4",
    op: "conv_transpose1d",
    inputs: [fill([1, 3, 5], SIGNED), fill([3, 2, 16], NONZERO), fill([2], SIGNED)],
    outShape: [1, 2, 40],
    attrs: { stride: 8, padding: 4 },
  },
  {
    // k = stride（pad 0）— 寄与する k がちょうど 1 本になる境界形
    name: "conv_transpose1d [1,2,6] * W[2,4,4] stride=4 padding=0",
    op: "conv_transpose1d",
    inputs: [fill([1, 2, 6], SIGNED), fill([2, 4, 4], POSITIVE), fill([4], NONZERO)],
    outShape: [1, 4, 24],
    attrs: { stride: 4, padding: 0 },
  },
];

// 双線形 resample（第 1 層 op）。新規原子なので A/B オラクルが無く、ここでの CPU 参照突合と
// golden の torch 突合が主門になる。MUST: 拡大 / 縮小 / 端の潰し / scale 0 の軸を全て踏み、
// B·C > 1 で**平面添字の畳み方**（N·C を 1 本へ潰す形）も踏む。
export const UPSAMPLE_CASES: readonly OpCase[] = [
  {
    // 非整数倍の拡大（H 3/6・W 4/8）。H≠W かつ Hout≠Wout で軸の取り違えが値に出る
    name: "upsample_bilinear2d 拡大 [2,3,4,5] → 7×9",
    op: "upsample_bilinear2d",
    inputs: [fill([2, 3, 4, 5], SIGNED)],
    outShape: [2, 3, 7, 9],
    attrs: { output_size: [7, 9] },
  },
  {
    // 縮小（BiRefNet の forward_enc / cxt と同型）— 2 タップしか読まないのが torch の仕様
    name: "upsample_bilinear2d 縮小 [1,3,8,10] → 3×4",
    op: "upsample_bilinear2d",
    inputs: [fill([1, 3, 8, 10], POSITIVE)],
    outShape: [1, 3, 3, 4],
    attrs: { output_size: [3, 4] },
  },
  {
    // 入力の高さ 1 → H の scale が 0（末尾特例 `index1 = index0` が全出力で発火する）。
    // W だけ縮小させて、2 軸で別の分岐を同時に踏む
    name: "upsample_bilinear2d 高さ 1 [1,2,1,6] → 4×3",
    op: "upsample_bilinear2d",
    inputs: [fill([1, 2, 1, 6], SIGNED)],
    outShape: [1, 2, 4, 3],
    attrs: { output_size: [4, 3] },
  },
  {
    // 1×1 → N×M（ASPP の GAP 枝）— 両軸とも scale 0 で実質 broadcast
    name: "upsample_bilinear2d 1×1 → 5×4 [2,3,1,1]",
    op: "upsample_bilinear2d",
    inputs: [fill([2, 3, 1, 1], NONZERO)],
    outShape: [2, 3, 5, 4],
    attrs: { output_size: [5, 4] },
  },
  {
    // 出力 1×1（scale の out == 1 特例 = 0）。torch は左上の 1 点だけを読む
    name: "upsample_bilinear2d 6×7 → 1×1 [1,2,6,7]",
    op: "upsample_bilinear2d",
    inputs: [fill([1, 2, 6, 7], SIGNED)],
    outShape: [1, 2, 1, 1],
    attrs: { output_size: [1, 1] },
  },
  {
    // 等倍（in == out）— λ が全て 0 で恒等コピーになる形（scale がちょうど 1.0）
    name: "upsample_bilinear2d 等倍 [1,2,5,4] → 5×4",
    op: "upsample_bilinear2d",
    inputs: [fill([1, 2, 5, 4], SIGNED)],
    outShape: [1, 2, 5, 4],
    attrs: { output_size: [5, 4] },
  },
];

/** deform の offset（±2.5 の非整数 — 入力平面の外まで振る）。 */
const OFFSET = (i: number): number => ((i % 11) - 5) * 0.5;
/** deform の modulator（BiRefNet の `2·sigmoid(...)` と同じ値域 [0,2]）。 */
const MODULATOR = (i: number): number => (i % 9) * 0.25;

// DCNv2（第 1' 層 op — ADR 0055）。新規原子なので A/B オラクルが無く、ここでの CPU 参照突合と
// golden の torch 突合が主門になる（退化ビット一致は gpu_deform_conv2d_test.ts）。
// MUST: k の 2 形（BiRefNet の 4 分岐は k ∈ {1,1,3,7}）・B > 1・Cin ≠ Cout・Kh ≠ Kw・
// padding の H≠W を踏む。
export const DEFORM_CASES: readonly OpCase[] = [
  {
    // Kh ≠ Kw / padding の H≠W（offset の y/x 順と重みの軸が同時に効く形）
    name: "deform_conv2d [1,2,4,5] * W[3,2,3,2] padding=[1,0]",
    op: "deform_conv2d",
    inputs: [
      fill([1, 2, 4, 5], SIGNED),
      fill([3, 2, 3, 2], POSITIVE),
      fill([1, 12, 4, 4], OFFSET),
      fill([1, 6, 4, 4], MODULATOR),
      fill([3], SIGNED),
    ],
    outShape: [1, 3, 4, 4],
    attrs: { padding: [1, 0] },
  },
  {
    // k=1（ASPP の aspp1 分岐）かつ **バッチ 2**（offset / mask のバッチ段が効く）
    name: "deform_conv2d k=1 バッチ 2 [2,3,3,4] * W[4,3,1,1]",
    op: "deform_conv2d",
    inputs: [
      fill([2, 3, 3, 4], SIGNED),
      fill([4, 3, 1, 1], POSITIVE),
      fill([2, 2, 3, 4], OFFSET),
      fill([2, 1, 3, 4], MODULATOR),
      fill([4], NONZERO),
    ],
    outShape: [2, 4, 3, 4],
    attrs: { padding: [0, 0] },
  },
  {
    // 同サイズ出力（BiRefNet の実形 — k=3 / padding 1）で縮約長を伸ばす
    name: "deform_conv2d [1,4,6,7] * W[5,4,3,3] padding=[1,1]",
    op: "deform_conv2d",
    inputs: [
      fill([1, 4, 6, 7], SIGNED),
      fill([5, 4, 3, 3], POSITIVE),
      fill([1, 18, 6, 7], OFFSET),
      fill([1, 9, 6, 7], MODULATOR),
      fill([5], SIGNED),
    ],
    outShape: [1, 5, 6, 7],
    attrs: { padding: [1, 1] },
  },
];

// GRU 隠れ側スキャン（第 2 層 op — ADR 0056）。ビット同一の A/B オラクルは
// gpu_gru_scan_parity_test.ts が持つので、ここは **CPU 参照との独立突合**（縮約の添字と
// 走査方向を、GEMM とも WGSL とも共有しない実装で確かめる）。
// MUST: 逆方向も同じ列で回す（順方向だけだと `t` の写像が恒等でも緑になる）。

/** GRU の隠れ側重み（H 本の縮約でゲートが飽和しない大きさに抑える）。 */
const SMALL_WEIGHT = (i: number): number => ((i % 19) - 9) * 0.031;

export const GRU_SCAN_CASES: readonly OpCase[] = ["gru_scan", "gru_scan_reverse"].flatMap((op) => [
  {
    // 最小形（T = 2 = 状態が 1 度は運ばれる最小）・N ≠ H で軸の取り違えが値に出る
    name: `${op} [4,1,9] h0[1,3]`,
    op,
    inputs: [
      fill([4, 1, 9], SIGNED),
      fill([1, 3], NONZERO),
      fill([9, 3], SMALL_WEIGHT),
      fill([9], SIGNED),
    ],
    outShape: [4, 1, 3],
  },
  {
    // バッチ > 1（1 workgroup = 1 バッチ要素の境界。状態が混ざると値に出る）
    name: `${op} バッチ 3 [3,3,15] h0[3,5]`,
    op,
    inputs: [
      fill([3, 3, 15], SIGNED),
      fill([3, 5], NONZERO),
      fill([15, 5], SMALL_WEIGHT),
      fill([15], SIGNED),
    ],
    outShape: [3, 3, 5],
  },
]);

// 境界ケース。MUST: 「大きめの入力」は grid-stride の**縮退**を踏まない — 要素数から
// 必要な workgroup 数がそのまま割り当たるためで、`stride` を定数にする誤りはここでは緑のまま
// 通る。縮退そのものは tests/gpu_gridstride_test.ts が dispatch 数を絞って直接検証する。
// ここが受け持つのは「1 workgroup を超える大きさで添字計算が破綻しないこと」。
export const BOUNDARY_CASES: readonly OpCase[] = [
  {
    // 4096 workgroup（dispatch 上限の内側）— 1 スレッド 1 要素で全域が覆われる大きさ
    name: "relu 大きめ 1 本 [1048576]",
    op: "relu",
    inputs: [fill([1 << 20], SIGNED)],
    outShape: [1 << 20],
  },
  {
    // strided copy も同じ大きさで踏む（出力 1048576 要素 = 4096 workgroup）
    name: "permute 大きめ [512,2048] dims=[1,0]",
    op: "permute",
    inputs: [fill([512, 2048], SIGNED)],
    outShape: [2048, 512],
    attrs: { dims: [1, 0] },
  },
  {
    // 行数 > maxComputeWorkgroupsPerDimension（既定 65535）— この 1 本だけは実測形のまま
    // 縮退を踏む（gridStrideWorkgroups が上限で頭打ちになり、1 workgroup が複数行を回す）
    name: "sum 行数多め [70000,4]",
    op: "sum",
    inputs: [fill([70000, 4], SIGNED)],
    outShape: [70000],
    attrs: { dim: 1 },
  },
];
