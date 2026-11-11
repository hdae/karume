/**
 * The memory estimate of the Wan2.2 TI2V-5B DiT against the slot backing a real Session builds for it,
 * on the online preset (`f16+dit8-a8-attn8-online` — linear and attention on a8, S never stored):
 * `workspaceBytes` equals the run's `planBacking.residentBytes` at S = 7,920 (33 frames at 704 × 1280)
 * and S = 27,280 (121 frames).
 *
 * The estimate replays the fused windows of rmsNormRope / adaLN / gatedResidual / silu as execution
 * allocates them (the same fusion plan's step list and the same device-free bookkeeping,
 * fused-step-layout.ts), and so the linear prologue and the attention's absorbed permutes, the
 * online attention's quantization temporaries, and the a8 linears' quantized activations and scales
 * (`linearI8a8TempBytes`). A plain node it approximates as one dispatch reading its inputs and
 * writing its outputs, which its kernels may not bind that way, so the equality is a measurement
 * on this graph, not a property of the construction. The estimate gets the Session's options and
 * the device's limits and facts, as a pipeline passes them.
 *
 * The inputs are deterministic fills of the declared shapes: the slot table depends on the shapes
 * only. The Session builds a backing on the second run of a signature (the first runs on the arena);
 * each S's backing is larger than the default keep budget, so it is kept alone (`retainedCount` 1 —
 * `residentBytes` is then that backing's regions).
 *
 * MUST: the asset is `models/karume-wan2.2/` (untracked local asset); without it or without a GPU
 * adapter the case SKIPs explicitly with a reason (ADR 0005).
 */

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { prepareContainer, type Tensor } from "@karume/runtime";
import { attentionDeviceFacts } from "../../runtime/src/runtime/attention-route.ts";
import { quantOf } from "./helpers/ab-gate.ts";
import { effectiveSessionOptions } from "../../models/tests/helpers/census-table.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  openTi2vTransformer,
  WAN_TI2V_COMPONENT,
  wanTi2vMirrorPresent,
} from "../../models/tests/helpers/wan-ti2v-dit.ts";
import {
  readWanTi2vDistributionManifestText,
  WAN_TI2V_DIST_MODEL,
  WAN_TI2V_DIST_ROOT,
  WAN_TI2V_ONLINE_QUANT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";

const manifestText = wanTi2vMirrorPresent()
  ? await readWanTi2vDistributionManifestText()
  : undefined;
if (manifestText === undefined) {
  console.warn(
    `[karume] SKIP the Wan2.2 DiT estimate against its slot backing: ${WAN_TI2V_DIST_ROOT.pathname} is missing`,
  );
}

/** The token counts of 33 and 121 frames at 704 × 1280 (9 / 31 latent frames × 22 × 40 patches). */
const SEQUENCES = [7920, 27280] as const;

const MIB = 2 ** 20;

/** Every input of the graph filled at its declared shape with `S` bound (bool inputs alternate by row). */
const inputsAt = (
  inputs: readonly {
    readonly name: string;
    readonly dtype: string;
    readonly shape: readonly (number | string)[];
  }[],
  sequence: number,
): Record<string, Tensor> =>
  Object.fromEntries(
    inputs.map((spec, slot) => {
      const shape = spec.shape.map((dim) => (typeof dim === "number" ? dim : sequence));
      const count = shape.reduce((product, dim) => product * dim, 1);
      if (spec.dtype === "bool") {
        const data = new Uint32Array(count);
        for (let index = 0; index < count; index += 1) data[index] = (index >> 7) & 1;
        return [spec.name, { dtype: "bool", shape, data }];
      }
      if (spec.dtype !== "f32") throw new Error(`input '${spec.name}' is ${spec.dtype}`);
      const data = new Float32Array(count);
      for (let index = 0; index < count; index += 1) data[index] = Math.sin(index * 0.013 + slot);
      return [spec.name, { dtype: "f32", shape, data }];
    }),
  );

test.skipIf(!GPU_AVAILABLE || manifestText === undefined)(
  "the Wan2.2 DiT's estimated workspace equals the slot backing of its run on the online preset at S 7,920 and 27,280",
  async () => {
    assert(manifestText !== undefined);
    const declared = quantOf(
      parseManifest(manifestText),
      WAN_TI2V_DIST_MODEL,
      WAN_TI2V_ONLINE_QUANT,
    ).session;
    const options = effectiveSessionOptions(
      "wan",
      declared,
      {},
      `wan-ti2v ${WAN_TI2V_DIST_MODEL}/${WAN_TI2V_ONLINE_QUANT}`,
    );
    const prepared = prepareContainer(await openTi2vTransformer(), WAN_TI2V_COMPONENT);
    // The online seat's kernel takes its subgroup form as in production (the Wan family acquires
    // with "available" — ADR 0132 decision 5).
    const gpu = await acquireTestGpu({ subgroups: "available" });
    try {
      const session = await prepared.createContainerSession(gpu, options);
      try {
        for (const sequence of SEQUENCES) {
          const report = prepared.estimate({
            ...options,
            bindings: { S: sequence },
            maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
            maxBufferSize: gpu.limits.maxBufferSize,
            maxComputeWorkgroupsPerDimension: gpu.limits.maxComputeWorkgroupsPerDimension,
            attentionDevice: attentionDeviceFacts(gpu),
          });
          const inputs = inputsAt(prepared.graph.inputs, sequence);
          for (let run = 0; run < 2; run += 1) await session.run(inputs);
          const { planBacking } = session.diagnostics();
          const [scenario] = report.scenarios;
          console.log(
            `[wan dit estimate] S=${sequence}: estimate workspace ${scenario.workspaceBytes} B` +
              ` (${(scenario.workspaceBytes / MIB).toFixed(1)} MiB), backing ${planBacking.residentBytes} B` +
              ` (retained ${planBacking.retainedCount})`,
          );
          assertEquals(planBacking.retainedCount, 1, `S=${sequence}: one kept backing`);
          assertEquals(
            planBacking.residentBytes,
            scenario.workspaceBytes,
            `S=${sequence}: slot backing vs estimate`,
          );
        }
      } finally {
        await session.dispose();
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
);
