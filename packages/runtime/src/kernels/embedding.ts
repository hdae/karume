/**
 * embedding（`out[…, h] = weight[index[…], h]`、値 f32 / 添字 i32）の固定カーネル。
 *
 * 出力・weight とも連続レイアウトで、出力の平坦添字 `i` から `row = i / H`・`col = i % H` を
 * 作れば読み出しは `weight[index[row] * H + col]` の 1 本で足りる（H = weight の最終次元）。
 *
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える
 * （実測は V=22012 / H=1024）。
 *
 * ## `padding_idx` は forward に効かない（契約 — ADR 0012）
 *
 * torch の `padding_idx` は「勾配で padding 行を更新しない」ための欄で、順伝播は素の行
 * gather と完全に同じ（`F.embedding` の forward も参照しない）。したがってこのカーネルは
 * attrs を 1 つも受け取らない。契約表（src/ops.ts）が受理して検証するのは、無視するために
 * 落とすと「未知 attr は fail loudly」の規律に穴が開くため。
 *
 * ## 範囲外添字の扱い（gather と同じ裁定 — src/kernels/gather.ts）
 *
 * 契約は `0 <= index < V`。違反は**モデル側の誤り**だが、起きたときに黙って別の行を返さない
 * ことを優先する: **GPU は範囲外の要素にだけ NaN（{@link EMBEDDING_OOB_BITS}）を書き**、
 * **CPU 参照は throw** する。カーネルから throw はできず、host へ通知するには run ごとの
 * デバイス側フォールト旗 + readback（新しい診断チャネル）が要るのでこの波では採らない。
 *
 * MUST: NaN のビット列は params で運ぶ。WGSL には NaN リテラルが無く、`bitcast<f32>(...)` を
 * 定数式で書くと「const-expression が NaN」としてシェーダ生成エラーになりうる実装がある。
 *
 * 重みは格納の変種を持つ（`w=f32` / `w=f16` / `w=i8` / `w=i4` — ADR 0018 / 0019 / 0069）。
 * 差は行の読み出し 1 行と scale の引き方だけで、範囲外添字の裁定は共通
 * （src/kernels/weight-storage.ts）。scale の読み出しは範囲内分岐の**内側**に置く
 * （範囲外の `pick` で `wscale` を引かない）。
 *
 * ## i4 の group scale は**要素ごと**（ADR 0069 決定 3）
 *
 * i8 の scale は行（= 語彙エントリ）ごとで、行が決まればループ不変になる。i4 の group scale は
 * 量子化軸（重み `[V,D]` の D 軸）方向に `group_size` ごと変わるので、`col` から引き直す:
 * `wscale[pick * (D / group_size) + col / group_size]`。1 スレッドが 1 出力要素しか持たない
 * ので巻き上げる先が無く、linear（gemm.ts の充填が quad ごとに引く）と違いここで直に引く。
 * MUST: group 長は WGSL に shift として焼くので**キーにも入れる**（`:wi4g32`）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  WEIGHT_SCALE_VAR,
  weightArrayType,
  weightKeyPart,
  weightLoaderWgsl,
  weightNote,
  weightRead,
  weightScaleWgsl,
  type WeightStorage,
} from "./weight-storage.ts";

export const EMBEDDING_WORKGROUP_SIZE = 256;

/** i8 / i4 変種の scale 束縛（出力の次の番号 — executor の bind entries と対で使う）。 */
export const EMBEDDING_SCALE_BINDING = 4;

/** 範囲外添字の出力に書く quiet NaN のビット列（f32）。 */
export const EMBEDDING_OOB_BITS = 0x7fc00000;

/**
 * i4 の group scale を**この 1 スレッドぶんだけ**束ねる行（i8 は行 scale・他は空文字）。
 *
 * MUST: 束縛名は i8 と同じ {@link WEIGHT_SCALE_VAR} — {@link weightRead} へ渡す式を格納形で
 * 分岐させないため。行の位置（範囲内分岐の内側・字下げ）も i8 と揃える。
 */
const embeddingScaleWgsl = (weight: WeightStorage, groupShift: number | undefined): string =>
  weight === "i4"
    ? `
      // group scale は量子化軸（列）依存 — 1 スレッド 1 要素なので巻き上げず要素ごとに引く
      let ${WEIGHT_SCALE_VAR} = wscale[u32(pick) * (dims.hidden >> ${groupShift}u) + (col >> ${groupShift}u)];`
    : weightScaleWgsl(weight, "u32(pick)", "      ");

export const embeddingKey = (weight: WeightStorage, groupSize?: number): string => {
  i4GroupShift("embedding", weight, groupSize);
  return `embedding:v1:f32:i32:wg${EMBEDDING_WORKGROUP_SIZE}${weightKeyPart(weight)}${
    i4GroupKeyPart(groupSize)
  }`;
};

export const embeddingWgsl = (weight: WeightStorage, groupSize?: number): string =>
  `// karume embedding (out[..., h] = weight[index[...], h], f32 / 添字 i32${weightNote(weight)})
struct Dims {
  n: u32,
  hidden: u32,
  vocab: u32,
  oob: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> weight: array<${weightArrayType(weight)}>;
@group(0) @binding(2) var<storage, read> index: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
${weightLoaderWgsl("weight", weight, EMBEDDING_SCALE_BINDING)}
@compute @workgroup_size(${EMBEDDING_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${EMBEDDING_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < dims.n) {
    let row = i / dims.hidden;
    let col = i % dims.hidden;
    let pick = index[row];
    // 契約外の添字は別の行を返さず NaN で汚染する（カーネル doc の裁定）
    if (pick < 0 || u32(pick) >= dims.vocab) {
      out[i] = bitcast<f32>(dims.oob);
    } else {${embeddingScaleWgsl(weight, i4GroupShift("embedding", weight, groupSize))}
      out[i] = ${weightRead("weight", weight, "u32(pick) * dims.hidden + col", WEIGHT_SCALE_VAR)};
    }
    i = i + stride;
  }
}
`;

/**
 * uniform の Dims（ちょうど 4 語 = 16 バイト。uniform アドレス空間の整列要件を満たす）。
 * `hidden` は 0 でもよい（そのとき `n` も 0 でカーネルのループが 1 度も回らない）。
 */
export const embeddingParams = (
  count: number,
  hidden: number,
  vocab: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("embedding params", { count, hidden, vocab });
  if (hidden === 0 && count !== 0) {
    // 0 除算になる組み合わせ（要素数 > 0 なのに hidden が無い）は shape 契約上ありえない。
    throw new CodegenError(`embedding params: hidden 0 で要素数 ${count} の組み合わせは無い`);
  }
  const params = new Uint32Array(4);
  params[0] = count;
  params[1] = hidden;
  params[2] = vocab;
  params[3] = EMBEDDING_OOB_BITS;
  return params;
};
