/**
 * linear の **GEMV 変種**（M=1・重み i4 格納 — ADR 0082）。`linear` の 2 本目のカーネル族で、
 * 出力・束縛・uniform は既定経路（src/kernels/gemm.ts の linear）と同じまま、**担当割りだけ**が
 * 「1 スレッド = 1 出力列」へ変わる。
 *
 * ## なぜ既定の GEMM 骨格では足りないのか（機序 — ADR 0082 / research 2026-08-30 §7）
 *
 * 既定は M のバケットで幾何を選ぶ（src/kernels/gemm-geometry.ts の `gemmGeometryForRows`）が、
 * どのバケットも「共有タイル + K タイル 16 ごとの二重 barrier」という 1 つの骨格を共有する。
 * M=1 に当たる `GEOMETRY_M16N16`（64 スレッド）では:
 *
 * - **出力を書くのは 64 スレッド中 4 本だけ**（`lid.y == 0` の行）。共有 A タイル（16 行 × K16）は
 *   1 行しか実体が無く、残り 15 行は 0 で埋めた死荷重になる。
 * - K タイルごとに `workgroupBarrier()` を 2 回通るため、**重み読みのレイテンシがタイル本数ぶん
 *   逐次に露出する**。実測は `k / 16 × ≈1.3µs` の k 比例で **n にほぼ非依存**、重みが L2 に
 *   収まる小形でも同じだけ遅い — つまり律速は帯域飢餓ではなく**発行の逐次化**。
 *
 * 本変種は共有メモリと barrier を丸ごと落とし、重み語を先読みしてメモリ並列度を作る。
 * `n` 本の独立した縮約が同時に走るので、遅延は列方向の並列で隠れる。
 *
 * ## 数値契約（ビット同一 MUST）
 *
 * MUST: 1 出力要素あたりの縮約は **k 昇順の逐次**・積和の字面は `acc = acc + a * b`・重みの
 * 復元は `f32(i32(u) − 8) * scale` の成分ごと f32 乗算・bias は最後に 1 度だけ加算。すべて
 * 既定経路（gemm.ts の `accumulatorUpdate` と weight-storage.ts の `dequant4`）と同一で、
 * 変わるのは ADR 0022 決定 3 が自由と認めた**担当割り**だけ = **既定経路とビット同一**。
 * MUST: 先読み（`unroll`）は「語をまとめて読む」だけで、**積和の順序は語の昇順のまま**。
 * 語をまたいで積和を混ぜると 1 出力要素あたりの加算順が動き、契約が割れる。
 * NOTE: f32 の縮約に順序非依存の理論保証は無いので、ビット同一は gemm-geometry と同じく
 * **実測命題**（門 = tests/gpu_linear_gemv_test.ts の u32 完全一致）。
 *
 * MUST NOT: k を分割して部分和を足し直す（split-K）形は縮約順が変わるため、この族には
 * 入れない。実測でも ADR 0058 の opt-in 席を切る価値が無かった（ADR 0082 §不採用）。
 *
 * ## 幾何との関係（gemm-geometry の「唯一の選択点」MUST の射程）
 *
 * gemm-geometry は「担当割りの選択点は `gemmGeometryForRows` 1 箇所」と書くが、その射程は
 * **GEMM 骨格の中**にとどまる（ADR 0082 決定 2）。本族は骨格を共有しない別カーネルで、
 * 選択は `#buildLinear` の 1 箇所・**プラン時 shape の純関数**であり、判別子は
 * パイプラインキーの族名 `linear_gemv` に載る — 実行時オートチューン禁止と
 * 「同一キー → バイト同一 WGSL」はどちらも保たれる。
 */

import { CodegenError } from "../codegen/errors.ts";
import { gemmParams } from "./gemm.ts";
import { i4GroupKeyPart, i4GroupShift, weightKeyPart } from "./weight-storage.ts";

/**
 * 重み 1 語（`vec4<u32>` = 16 B）が運ぶ i4 要素数 = **縮約の刻み**。
 *
 * 適格判定（src/runtime/recipe-builder.ts の `#buildLinear`）が k と group 長へ課す整除の
 * 単位でもあるので、門とカーネルが同じ 1 個の定数を読む。
 */
export const LINEAR_GEMV_UNIT = 32;

/** WebGPU core が保証する 1 workgroup のスレッド数上限（`cols` の上界）。 */
const MAX_THREADS = 256;

/**
 * 生成パラメタ（幾何にあたるもの）。**{@link defaultLinearGemvVariant} が唯一の選択点**で、
 * 実行時に選び直さない（ADR 0022 の実行時オートチューン禁止は本族にも掛かる）。
 */
export type LinearGemvVariant = {
  /** 1 workgroup が担当する出力列数（= workgroup のスレッド数 — 1 スレッド 1 列）。 */
  readonly cols: number;
  /** 1 反復で先読みする重み語数（メモリ並列度）。積和の順序には影響しない。 */
  readonly unroll: number;
};

/**
 * 既定の変種。**RTX 3080 Ti / gemma4 E2B decode の実 12 形 + 端数 4 形の掃引**で census 加重
 * 最良（`c32 u4`・対既定 ×8.45 — ADR 0082 / docs/research/2026-08-30-gemma4-decode-wallclock.md
 * §7）。MUST: 既定の変更はビット同一門（tests/gpu_linear_gemv_test.ts）の再実測とセット。
 */
export const defaultLinearGemvVariant = (): LinearGemvVariant => ({ cols: 32, unroll: 4 });

const assertVariant = (variant: LinearGemvVariant): void => {
  const { cols, unroll } = variant;
  if (!Number.isSafeInteger(cols) || cols < 1 || cols > MAX_THREADS) {
    throw new CodegenError(`linear_gemv: cols は 1..${MAX_THREADS} の整数（${cols}）`);
  }
  if (!Number.isSafeInteger(unroll) || unroll < 1) {
    throw new CodegenError(`linear_gemv: unroll は正整数（${unroll}）`);
  }
};

/**
 * group 長 → WGSL に焼く shift。
 *
 * 2 冪 ≥ 16 は {@link i4GroupShift}（宣言層と同じ導出点）が見る。本族はさらに
 * **group ≥ {@link LINEAR_GEMV_UNIT}** を要求する — 1 語 32 要素が group を跨ぐと
 * 語あたり 1 個の scale では足りず、黙って別の scale が掛かった値が出るため。
 */
const gemvGroupShift = (groupSize: number): number => {
  const shift = i4GroupShift("linear_gemv", "i4", groupSize);
  if (shift === undefined || groupSize < LINEAR_GEMV_UNIT) {
    throw new CodegenError(
      `linear_gemv: group_size ${groupSize} が ${LINEAR_GEMV_UNIT} 以上の 2 冪でない` +
        `（1 語 = ${LINEAR_GEMV_UNIT} 要素が group を跨ぐ）`,
    );
  }
  return shift;
};

/**
 * uniform の Dims（既定経路の `linearParams(1, n, k)` とバイト単位で同一 — 束縛レイアウトを
 * 分けない契約）。族固有なのは検査だけで、m / n / k の u32 域は {@link gemmParams} へ委譲する。
 *
 * MUST: k を **{@link LINEAR_GEMV_UNIT} の倍数**に限る。WGSL の `units = dims.k / 32u` は
 * 端数を切り捨てるので、外すと縮約が行の末尾を黙って落とした値を返す（例外は出ない）。
 * MUST: k を **group_size の倍数**にも限る（:210 の `scale_base = col * (k >> shift)` が
 * 行あたりの scale 本数を割り算で導くため）。宣言層（ADR 0069 決定 2）と recipe-builder の
 * 適格判定が同じ条件を保証しているが、カーネル直呼びはそこを通らない。
 */
export const linearGemvParams = (
  n: number,
  k: number,
  groupSize: number,
): Uint32Array<ArrayBuffer> => {
  gemvGroupShift(groupSize);
  if (!Number.isSafeInteger(k) || k < 0 || k % LINEAR_GEMV_UNIT !== 0) {
    throw new CodegenError(
      `linear_gemv params: k は ${LINEAR_GEMV_UNIT} の倍数の非負整数（${k}）`,
    );
  }
  if (k % groupSize !== 0) {
    throw new CodegenError(
      `linear_gemv params: k=${k} が group_size ${groupSize} で割り切れない`,
    );
  }
  return gemmParams("linear", 1, n, k);
};

/**
 * パイプラインキー。族名 `linear_gemv` が既定経路（`linear`）との判別子で、変種と group 長は
 * どちらも WGSL に焼かれるのでキーに載せる（同一キー → バイト同一 WGSL の codegen 決定性）。
 *
 * 格納判別子（`:wi4`）と group 断片（`g32`）は weight-storage.ts の綴りをそのまま使う —
 * 診断・census が `:wi4g32` で経路を識別する既存の読み方（ADR 0069 決定 5）に揃える。
 */
export const linearGemvKey = (
  groupSize: number,
  variant: LinearGemvVariant = defaultLinearGemvVariant(),
): string => {
  assertVariant(variant);
  gemvGroupShift(groupSize);
  return `linear_gemv:v1:f32:c${variant.cols}u${variant.unroll}${weightKeyPart("i4")}${
    i4GroupKeyPart(groupSize)
  }`;
};

/**
 * 束縛。**既定経路の linear と同じ番号・同じ意味**（0 dims / 1 x / 2 w / 3 bias / 4 out /
 * 5 wscale）で、`#buildLinear` が組む束縛列をそのまま受ける。
 *
 * 変わるのは要素型 2 つだけ: 重みは `vec4<u32>`（16 B = i4 32 要素を 1 度に読む）、出力は
 * `f32`（1 スレッド 1 列のスカラ書き — 既定 v4 経路の `vec4<f32>` と違い n の整除を要らない）。
 */
const BINDINGS = `@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// 行頭が 16 B 整列なのは k % ${LINEAR_GEMV_UNIT} == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;`;

/** 語 1 本ぶんの読み（重み語 + group scale + x の quad 先頭）。 */
const unitLoads = (slot: string, unitExpr: string, shift: number): string =>
  `    let unit${slot} = ${unitExpr};
    let pw${slot} = w[row_base + unit${slot}];
    let ws${slot} = wscale[scale_base + ((unit${slot} * ${LINEAR_GEMV_UNIT}u) >> ${shift}u)];
    let xq${slot} = unit${slot} * ${LINEAR_GEMV_UNIT / 4}u;`;

/**
 * 語 1 本（i4 {@link LINEAR_GEMV_UNIT} 要素）の積和展開。
 *
 * nibble の並びは weight-storage.ts の `dequant4` と同一（要素 2i = 下位 / 2i+1 = 上位・
 * 格納値 `u = q + 8` — 正本はエクスポータ `karume/emit.py: pack_int4`）。
 * MUST: 展開順は語内の要素昇順（成分 x→w × バイト x→w × 下位→上位 nibble）— これが
 * 「k 昇順の逐次」そのもので、崩すと既定経路とのビット同一が割れる。
 * MUST: x は `vec4<f32>` 束縛から**静的成分**で引く（動的成分添字は Metal でローカル領域へ
 * 落ちる — gemm.ts の `storeBTransposed` と同じ規律）。
 */
const unitMacs = (slot: string): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes.map((component, quad) => {
    const bytes = `b${slot}_${quad}`;
    const xa = `xa${slot}_${quad}`;
    const xb = `xb${slot}_${quad}`;
    // 語の成分 `quad` は要素 8·quad..8·quad+7 = x の quad 2 本ぶん。
    const macs = lanes.flatMap((byte, lane) => {
      const source = lane < 2 ? xa : xb;
      const low = lanes[(lane * 2) % 4];
      const high = lanes[(lane * 2 + 1) % 4];
      return [
        `    acc = acc + ${source}.${low} * (f32(i32(${bytes}.${byte} & 0xFu) - 8) * ws${slot});`,
        `    acc = acc + ${source}.${high} * (f32(i32(${bytes}.${byte} >> 4u) - 8) * ws${slot});`,
      ];
    }).join("\n");
    return `    let ${bytes} = unpack4xU8(pw${slot}.${component});
    let ${xa} = x[xq${slot} + ${quad * 2}u];
    let ${xb} = x[xq${slot} + ${quad * 2 + 1}u];
${macs}`;
  }).join("\n");
};

/**
 * GEMV の WGSL（`out[n] = x[k] · wᵀ[n,k] + bias[n]`・M=1・重み i4 格納）。
 *
 * 1 スレッドが 1 出力列の縮約を丸ごと持つので、並列度の上限は `n`。これはビット同一の代償
 * そのもので、k 方向へ割れば並列度は上がるが縮約順が動く（MUST NOT — モジュール doc）。
 */
export const linearGemvWgsl = (
  groupSize: number,
  variant: LinearGemvVariant = defaultLinearGemvVariant(),
): string => {
  assertVariant(variant);
  const shift = gemvGroupShift(groupSize);
  const { cols, unroll } = variant;
  const slots = Array.from({ length: unroll }, (_, slot) => `${slot}`);
  const loads = slots.map((slot) => unitLoads(slot, `unit + ${slot}u`, shift)).join("\n");
  const macs = slots.map((slot) => unitMacs(slot)).join("\n");
  return `// karume linear gemv (M=1: out[n] = x[k] · wᵀ[n,k] + bias[n], f32, 重み i4 格納, ${cols} 列 / wg, 語 ${unroll} 本先読み)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
${BINDINGS}

@compute @workgroup_size(${cols})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // 共有メモリも barrier も持たないので、端の workgroup は早期 return してよい
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / ${LINEAR_GEMV_UNIT}u;
  let row_base = col * units;
  let scale_base = col * (dims.k >> ${shift}u);
  var acc = 0.0;
  var unit = 0u;
  // 先読みぶんの重み語を**先に**全て発行してから積和へ入る（メモリ並列度）。語の処理順は
  // 昇順のままなので、1 出力要素あたりの加算順序は先読み本数によらず同じ
  for (; unit + ${unroll}u <= units; unit = unit + ${unroll}u) {
${loads}
${macs}
  }
  // 端数の語（units % ${unroll} 本）— 上と同じ順序を 1 語ずつ辿る
  for (; unit < units; unit = unit + 1u) {
${unitLoads("t", "unit", shift)}
${unitMacs("t")}
  }
  out[col] = acc + bias[col];
}
`;
};
