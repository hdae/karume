/**
 * 融合ルール `upsample2x`（VAE nearest-exact x2 の 6 ノード）。全ルール共通の適格条件と
 * `defineRule` は {@link "../fusion-rule.ts"}、適用順を決める宣言表は
 * {@link "../fusion.ts"} の `FUSION_RULES`。
 */

import { numel } from "../../ops.ts";
import {
  upsample2xParams,
  UPSAMPLE_2X_KEY,
  UPSAMPLE_2X_WGSL,
  UPSAMPLE_2X_WORKGROUP_SIZE,
} from "../../kernels/upsample2x.ts";
import {
  allF32,
  defineRule,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";

type Upsample2xMatch = FusionMatch & {
  readonly inputName: string;
  readonly inputShape: readonly number[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly width: number;
};

/**
 * VAE nearest-exact x2: エクスポータが出す連続 6 ノード
 * `reshape → expand(width x2) → reshape → reshape → expand(height x2) → reshape`。
 *
 * 結線・解決済み shape・dtype・内部 use-count が**全て**一致した場合だけ 1 pass へ置換する。
 * MUST: f32 rank4 NCHW の各空間軸 2 倍だけ。一般の broadcast / resize へ広げない。
 */
export const UPSAMPLE_2X_RULE = defineRule<Upsample2xMatch>({
  name: "upsample2x",
  heads: ["reshape"],
  match: (nodes, index, context) => {
    const first = nodes[index];
    if (first?.node.op !== "reshape") return undefined;
    const horizontal = nodes[index + 1];
    const flatten = nodes[index + 2];
    const addHeightAxis = nodes[index + 3];
    const vertical = nodes[index + 4];
    const output = nodes[index + 5];
    if (
      horizontal === undefined || flatten === undefined || addHeightAxis === undefined ||
      vertical === undefined || output === undefined
    ) return undefined;
    const chain = [first, horizontal, flatten, addHeightAxis, vertical, output];
    if (
      horizontal.node.op !== "expand" || flatten.node.op !== "reshape" ||
      addHeightAxis.node.op !== "reshape" || vertical.node.op !== "expand" ||
      output.node.op !== "reshape"
    ) return undefined;
    if (!allF32(chain)) return undefined;

    const inputName = first.node.ins[0];
    if (
      horizontal.node.ins[0] !== first.outputs[0].name ||
      flatten.node.ins[0] !== horizontal.outputs[0].name ||
      addHeightAxis.node.ins[0] !== flatten.outputs[0].name ||
      vertical.node.ins[0] !== addHeightAxis.outputs[0].name ||
      output.node.ins[0] !== vertical.outputs[0].name
    ) return undefined;

    const inputShape = first.inputShapes[0];
    if (inputShape.length !== 4 || inputShape.some((dim) => dim < 1)) return undefined;
    const [batch, channels, height, width] = inputShape;
    const batchChannels = batch * channels;
    const flatRows = batchChannels * height;
    const outHeight = height * 2;
    const outWidth = width * 2;
    if (
      ![batchChannels, flatRows, outHeight, outWidth].every((value) => Number.isSafeInteger(value))
    ) return undefined;

    if (
      !sameShape(first.outputs[0].shape, [flatRows, width, 1]) ||
      !sameShape(horizontal.outputs[0].shape, [flatRows, width, 2]) ||
      !sameShape(flatten.outputs[0].shape, [batchChannels, height, outWidth]) ||
      !sameShape(addHeightAxis.outputs[0].shape, [batchChannels, height, 1, outWidth]) ||
      !sameShape(vertical.outputs[0].shape, [batchChannels, height, 2, outWidth]) ||
      !sameShape(output.outputs[0].shape, [batch, channels, outHeight, outWidth])
    ) return undefined;
    if (!internalsArePrivate(chain, context)) return undefined;

    return {
      window: chain,
      chain,
      inputName,
      inputShape,
      outputName: output.outputs[0].name,
      outputShape: output.outputs[0].shape,
      width,
    };
  },
  build: (matched) => {
    const sourceCount = numel(matched.inputShape);
    return {
      binds: [matched.inputName],
      outputName: matched.outputName,
      outputShape: matched.outputShape,
      temps: [],
      dispatches: [{
        key: UPSAMPLE_2X_KEY,
        wgsl: () => UPSAMPLE_2X_WGSL,
        // 巨大 shape は GPU 確保より先に u32 添字の契約で fail loudly させる。
        params: upsample2xParams(sourceCount, matched.width),
        workgroups: {
          kind: "gridStride",
          items: sourceCount,
          size: UPSAMPLE_2X_WORKGROUP_SIZE,
        },
      }],
    };
  },
});
