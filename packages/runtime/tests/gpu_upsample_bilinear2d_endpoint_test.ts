/**
 * upsample_bilinear2d on a shape with one axis above 2^24: the endpoint index does not run over
 * into the next plane.
 *
 * With the f32 rounding of scale = `fl((in − 1) / (out − 1))` and of the source coordinate
 * `fl(scale · output index)`, the last source coordinate can exceed `in − 1` (in_w = 16777220 and
 * out_w = 2 give 16777220 = in_w). Without an upper bound on the integer index, plane 0's endpoint
 * reads the start of plane 1 (a wrong value with no exception). Like torch's
 * `guard_index_and_lambda`, the index is clamped to `in − 1`, so the endpoint is the input's last
 * element.
 *
 * The input is 2 planes × 16,777,220 elements (about 134 MB). The clamp only triggers on shapes
 * with one axis above roughly 2^23 (with in_w 8,650,802 and out_w 34 the last source coordinate
 * also reaches in_w).
 */
import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu, prepareContainer } from "../mod.ts";
import { referenceUpsampleBilinear2d } from "../src/reference/ops.ts";
import { GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const WIDTH_IN = 16_777_220;
const INPUT_SHAPE = [1, 2, 1, WIDTH_IN];
const OUTPUT_SHAPE = [1, 2, 1, 2];

/**
 * Plane 0 = [1, 1, …, 7], plane 1 = [100, 2, …, 9] (only the ends and starts get distinct values,
 * so the output shows where it read).
 */
const input = (): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(2 * WIDTH_IN);
  data.fill(1, 0, WIDTH_IN);
  data[WIDTH_IN - 1] = 7;
  data.fill(2, WIDTH_IN);
  data[WIDTH_IN] = 100;
  data[2 * WIDTH_IN - 1] = 9;
  return data;
};

// [first, last] of each plane (downscaling to out_w = 2 reads only the two end points).
const EXPECTED = [1, 7, 100, 9];

describe("upsample_bilinear2d: endpoints of a downscale with one axis above 2^24", () => {
  it("the CPU reference reads the last element of each plane (not the start of the next plane)", () => {
    const out = referenceUpsampleBilinear2d(
      { dtype: "f32", shape: INPUT_SHAPE, data: input() },
      { output_size: [1, 2] },
    );
    assertEquals([...out.data], EXPECTED);
  });

  it.skipIf(!GPU_AVAILABLE)(
    "the GPU also reads the last element of each plane (real GPU)",
    async function () {
      const gpu = await acquireGpu();
      try {
        const prepared = prepareContainer(
          await openModelBytes(
            singleOpDeclaration("upsample_bilinear2d", [INPUT_SHAPE], [OUTPUT_SHAPE], {
              attrs: { output_size: [1, 2] },
            }),
            [],
          ),
          GRAPH_NAME,
        );
        const session = await prepared.createContainerSession(gpu);
        try {
          const out = await session.run({
            x0: { dtype: "f32", shape: INPUT_SHAPE, data: input() },
          });
          assertEquals([...out.y.data], EXPECTED);
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }
    },
  );
});
