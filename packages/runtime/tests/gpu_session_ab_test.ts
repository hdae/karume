// The session A/B on the GPU (`runSessionAb` with `sessionArm` of `@karume/runtime/tune`, through
// the public API only): the base arm is a Session of a small synthetic graph — `y = mul(gelu(gate),
// other)` — without rules, the second arm the same with the external test rule `test/gelu_mul` of
// the runtime's test tree (tests/helpers/external-gelu-mul-rule.ts, one dispatch for the pair). The
// record must show both arms u32-identical to the base's first outputs over every visit, the rule's
// hit on the second arm only, and one dispatch against the plain pair's two. Real times are not
// judged: a verdict on them would depend on the machine. No adapter = explicit SKIP.

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu, prepareContainer, type Tensor } from "../mod.ts";
import { runSessionAb, sessionArm, sessionAbStatistics } from "../tune.ts";
import {
  EXTERNAL_GELU_MUL_RULE,
  EXTERNAL_GELU_MUL_RULE_NAME,
} from "./helpers/external-gelu-mul-rule.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";

/** Rows × width: 771 elements leave a partial last workgroup of the rule's 128 threads. */
const SHAPE = [1, 3, 257];

const GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["gelu", "mul"] },
  symbols: [],
  inputs: [
    { name: "gate", dtype: "f32", shape: SHAPE },
    { name: "other", dtype: "f32", shape: SHAPE },
  ],
  outputs: ["y"],
  initializers: {},
  values: { h: { dtype: "f32", shape: SHAPE }, y: { dtype: "f32", shape: SHAPE } },
  nodes: [
    { op: "gelu", ins: ["gate"], outs: ["h"], attrs: {} },
    { op: "mul", ins: ["h", "other"], outs: ["y"], attrs: {} },
  ],
};

const FEEDS: Readonly<Record<string, Tensor>> = {
  gate: fill(SHAPE, (i) => Math.sin(i * 0.37) * 9.5 + i / 997),
  other: fill(SHAPE, (i) => Math.cos(i * 0.53) * 3.3 - i / 1201),
};

describe.skipIf(!GPU_AVAILABLE)("runSessionAb with sessionArm (real GPU)", () => {
  for (const openPerVisit of [false, true]) {
    it(`records an external rule's arm as identical to the base, with its hit and one dispatch fewer (openPerVisit ${openPerVisit})`, async () => {
      const gpu = await acquireGpu();
      try {
        const prepared = prepareContainer(await openModelBytes(GRAPH, []), GRAPH_NAME);
        const info = gpu.adapterInfo;
        const record = await runSessionAb({
          arms: [
            sessionArm({ name: "base", prepared, gpu, feeds: FEEDS }),
            sessionArm({
              name: "fused",
              prepared,
              gpu,
              options: { fusionRules: [EXTERNAL_GELU_MUL_RULE] },
              feeds: FEEDS,
            }),
          ],
          rounds: 3,
          openPerVisit,
          environment: {
            adapter: {
              vendor: info.vendor,
              architecture: info.architecture,
              device: info.device,
              description: info.description,
            },
            userAgent: navigator.userAgent,
            backend: "unknown",
            date: new Date().toISOString(),
          },
        });
        const [base, fused] = record.arms;
        for (const arm of record.arms) {
          assertEquals(
            arm.identity.identical,
            true,
            `${arm.name}: ${JSON.stringify(arm.identity)}`,
          );
          assertEquals(arm.identity.visits, 8, arm.name);
          assertEquals(arm.identity.worst.words, 3 * 257, arm.name);
          assertEquals(arm.timesMs.length, 3, arm.name);
          assert(
            arm.timesMs.flat().every((time) => Number.isFinite(time) && time >= 0),
            `${arm.name}: ${JSON.stringify(arm.timesMs)}`,
          );
        }
        assertEquals(base.fusions?.[EXTERNAL_GELU_MUL_RULE_NAME], undefined);
        assertEquals(fused.fusions?.[EXTERNAL_GELU_MUL_RULE_NAME], 1);
        assertEquals([base.dispatches, fused.dispatches], [2, 1]);
        // The statistics read the record (no verdict on real times).
        assertEquals(sessionAbStatistics(record).length, 1);
      } finally {
        gpu.destroy();
      }
    });
  }
});
