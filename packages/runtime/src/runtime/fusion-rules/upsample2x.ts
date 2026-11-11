/**
 * Fusion rule `upsample2x` (the 6 nodes of the VAE nearest-exact x2). The eligibility conditions
 * shared by every rule and `defineRule` are in {@link "../fusion-rule.ts"}; the declaration table
 * that fixes the application order is `FUSION_RULES` in {@link "../fusion.ts"}.
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
 * VAE nearest-exact x2: the 6 consecutive nodes the exporter emits
 * `reshape → expand(width x2) → reshape → reshape → expand(height x2) → reshape`.
 *
 * Replaced by one pass only when the wiring, resolved shapes, dtypes and internal use counts
 * **all** match.
 * MUST: only f32 rank-4 NCHW, x2 on each spatial axis. Do not widen to a general broadcast /
 * resize.
 */
export const UPSAMPLE_2X_RULE = defineRule<Upsample2xMatch, "upsample2x">({
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
      horizontal === undefined ||
      flatten === undefined ||
      addHeightAxis === undefined ||
      vertical === undefined ||
      output === undefined
    )
      return undefined;
    const chain = [first, horizontal, flatten, addHeightAxis, vertical, output];
    if (
      horizontal.node.op !== "expand" ||
      flatten.node.op !== "reshape" ||
      addHeightAxis.node.op !== "reshape" ||
      vertical.node.op !== "expand" ||
      output.node.op !== "reshape"
    )
      return undefined;
    if (!allF32(chain)) return undefined;

    const inputName = first.node.ins[0];
    if (
      horizontal.node.ins[0] !== first.outputs[0].name ||
      flatten.node.ins[0] !== horizontal.outputs[0].name ||
      addHeightAxis.node.ins[0] !== flatten.outputs[0].name ||
      vertical.node.ins[0] !== addHeightAxis.outputs[0].name ||
      output.node.ins[0] !== vertical.outputs[0].name
    )
      return undefined;

    const inputShape = first.inputShapes[0];
    if (inputShape.length !== 4 || inputShape.some((dim) => dim < 1)) return undefined;
    const [batch, channels, height, width] = inputShape;
    const batchChannels = batch * channels;
    const flatRows = batchChannels * height;
    const outHeight = height * 2;
    const outWidth = width * 2;
    if (
      ![batchChannels, flatRows, outHeight, outWidth].every((value) => Number.isSafeInteger(value))
    )
      return undefined;

    if (
      !sameShape(first.outputs[0].shape, [flatRows, width, 1]) ||
      !sameShape(horizontal.outputs[0].shape, [flatRows, width, 2]) ||
      !sameShape(flatten.outputs[0].shape, [batchChannels, height, outWidth]) ||
      !sameShape(addHeightAxis.outputs[0].shape, [batchChannels, height, 1, outWidth]) ||
      !sameShape(vertical.outputs[0].shape, [batchChannels, height, 2, outWidth]) ||
      !sameShape(output.outputs[0].shape, [batch, channels, outHeight, outWidth])
    )
      return undefined;
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
      dispatches: [
        {
          key: UPSAMPLE_2X_KEY,
          wgsl: () => UPSAMPLE_2X_WGSL,
          // A huge shape fails loudly on the u32 index contract before any GPU allocation.
          params: upsample2xParams(sourceCount, matched.width),
          workgroups: {
            kind: "gridStride",
            items: sourceCount,
            size: UPSAMPLE_2X_WORKGROUP_SIZE,
          },
        },
      ],
    };
  },
});
