/**
 * dispatch の params バッファの usage（uniform / storage の 2 通り）。
 *
 * 族別導出と {@link "../recipe-builder.ts"} の融合 replay が同じ値を使うので、**値の置き場は
 * ここ 1 つ**にする。族別モジュールから `recipe-builder.ts` を値で import すると
 * 「入口 → 族別」の逆辺ができるため（型だけの参照は消去されるので逆辺にならない）。
 */

import { BUFFER_USAGE } from "../../gpu/webgpu-constants.ts";

export const PARAMS_STORAGE_USAGE = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST;
export const PARAMS_UNIFORM_USAGE = BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST;
