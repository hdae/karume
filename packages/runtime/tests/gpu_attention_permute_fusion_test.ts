// A fused attention on an a8 seat that absorbs the [0, 2, 1, 3] permutes of its q / k / v
// (perf-ledger F-3 A4 / F-8 — `attentionPermutesAt` of src/runtime/fusion.ts, honoured by
// `buildAttention` on every route an a8 seat can take) and, on the online route only, the permute of
// O after it (the kernel stores O token-major): bit parity with the same graph where the permutes
// stay nodes, the counter, and the dispatches each route saves, on the online route (sg32 at D 128
// and 64, wg at D 128 — each with N = 200 and with N = 203, the padded key axis of the ADR 0132
// addendum of 2026-10-27: tiled transpose → padded Vᵀ quantisation → the `…:obshd:kpad` kernel), the three-dispatch a8 route with S in f16 (s16), an online seat whose shape
// (D 96) falls back to the three dispatches with S in f32, and the two a8 shapes whose stage falls
// back to f32 (①QK at D % 4 != 0, ③PV at N % 4 != 0 — the head-major copy path). The f32 seat
// absorbs nothing. Off the online route the output permute stays a node.
//
// The reference is the same graph with a zero-dispatch reshape between each input permute and the
// attention, and between the attention and the output permute: the values and the physical kernels
// stay the graph's, only the exact match stops matching.
//
// The route table, the graphs and the twin cases of the routes any device takes and of the online
// sg32 routes live in @karume/kernel-checks (checkAttentionPermuteTwins,
// checkAttentionPermuteSg32Twins), which the playground's /kernel-check page runs on the browser's
// backend too. The online wg routes need a device acquired without subgroups and keep their twin
// here; the census, the non-constant output and the a8 × "f32" comparison of every route stay here.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  ATTENTION_PERMUTE_ROUTES,
  ATTENTION_PERMUTE_TWIN_CASES,
  type AttentionPermuteDispatches,
  attentionPermuteFeeds,
  attentionPermuteOnlineForm,
  type AttentionPermuteShape,
  attentionPermuteSg32Shortfall,
  checkAttentionPermuteSg32Twins,
  checkAttentionPermuteTwins,
  checkFusionTwin,
  permutedAttentionGraph,
} from "@karume/kernel-checks";
import { stridedKey } from "../src/codegen/strided.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { attentionOnlineKeyWords } from "../src/kernels/attention-online-i8a8.ts";
import { QUANTIZE_ROWS_KEY } from "../src/kernels/quantize-rows.ts";
import { TRANSPOSE_TILED_KEY } from "../src/kernels/transpose-tiled.ts";
import { attentionDeviceFacts, attentionRoute } from "../src/runtime/attention-route.ts";
import {
  createSessionFromContainer,
  type Session,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  assertCheckCase,
  assertCheckPart,
  assertWordIdentity,
} from "./helpers/kernel-check-assert.ts";
import { GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";
import { type CensusRow, countDispatches, requireCensus } from "./helpers/pipeline-census.ts";

const words = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

type Observed = {
  readonly output: Uint32Array<ArrayBuffer>;
  readonly census: readonly CensusRow[];
};

const openGraph = async (
  gpu: GpuContext,
  shape: AttentionPermuteShape,
  interpose: boolean,
  options: SessionOptions,
): Promise<Session> =>
  await createSessionFromContainer(
    gpu,
    await openGraphModel(permutedAttentionGraph(shape, interpose)),
    GRAPH_NAME,
    options,
  );

/** Runs `session` once on the shape's inputs: its output words and its census. */
const observe = async (
  session: Session,
  shape: AttentionPermuteShape,
  interpose: boolean,
): Promise<Observed> => {
  const outputs = await session.run(attentionPermuteFeeds(shape));
  const ot = outputs["ot"];
  assert(ot !== undefined, "the attention output is missing");
  return {
    output: words(ot),
    census: requireCensus(
      session.diagnostics().lastRunPipelines,
      interpose ? "reference" : "absorbed",
    ),
  };
};

const runGraph = async (
  gpu: GpuContext,
  shape: AttentionPermuteShape,
  interpose: boolean,
  options: SessionOptions,
): Promise<Observed> => {
  const session = await openGraph(gpu, shape, interpose, options);
  try {
    return await observe(session, shape, interpose);
  } finally {
    await session.dispose();
  }
};

const dispatchesOf = (census: readonly CensusRow[]): AttentionPermuteDispatches => ({
  strided: countDispatches(census, stridedKey({ dtype: "f32" })),
  transpose: countDispatches(census, TRANSPOSE_TILED_KEY),
  quantizeBshd: countDispatches(
    census,
    (key) => key.startsWith(QUANTIZE_ROWS_KEY) && key.endsWith(":bshd"),
  ),
  quantizeRows: countDispatches(
    census,
    (key) => key.startsWith(QUANTIZE_ROWS_KEY) && !key.endsWith(":bshd"),
  ),
  total: countDispatches(census, () => true),
});

describe.skipIf(!GPU_AVAILABLE)(
  "a fused attention that absorbs the permutes of its q / k / v (perf-ledger F-3 / F-8)",
  () => {
    it("the routes any device takes: bit-identical to the permutes as nodes, attentionPermute at the route's count", async () => {
      const gpu = await acquireGpu({ subgroups: "available" });
      try {
        const part = await checkAttentionPermuteTwins(gpu);
        assertEquals([part.id, part.cases.length], Object.entries(ATTENTION_PERMUTE_TWIN_CASES)[0]);
        assertCheckPart(part);
      } finally {
        gpu.destroy();
      }
    });

    it("the online routes in the sg32 form: bit-identical to the permutes as nodes, attentionPermute 4", async (context) => {
      const gpu = await acquireGpu({ subgroups: "available" });
      try {
        if (attentionPermuteSg32Shortfall(gpu) !== undefined) {
          // The device does not take the sg32 form (the wg routes and the other routes cover it).
          context.skip();
          return;
        }
        const part = await checkAttentionPermuteSg32Twins(gpu);
        assertEquals([part.id, part.cases.length], Object.entries(ATTENTION_PERMUTE_TWIN_CASES)[1]);
        assertCheckPart(part);
      } finally {
        gpu.destroy();
      }
    });

    for (const route of ATTENTION_PERMUTE_ROUTES) {
      // The wg routes need a device acquired without subgroups, so their twin stays here.
      const twinHere = route.reduce === "wg";
      it(`${route.name}: ${twinHere ? `bit-identical to the permutes as nodes, attentionPermute ${route.absorbedPermutes}, and ` : ""}the route's own dispatches`, async (context) => {
        const gpu = await acquireGpu(route.subgroups === false ? {} : { subgroups: "available" });
        try {
          const { d, n } = route.shape;
          if (
            route.reduce !== undefined &&
            attentionPermuteOnlineForm(gpu, route.shape) !== route.reduce
          ) {
            // The device does not take this online form (the other online cases and s16 cover it).
            context.skip();
            return;
          }
          if (route.fallsBack) {
            // Precondition (asserted, not skipped — the shape alone rules the online kernel out).
            assertEquals(
              attentionRoute(
                { attentionCompute: "a8", attentionScoreStorage: "none" },
                d,
                n,
                attentionDeviceFacts(gpu),
              ),
              { qk: "i8a8", pv: "i8a8", score: "f32" },
              `${route.name}: route`,
            );
          }
          const absorbedSession = await openGraph(gpu, route.shape, false, route.options);
          const referenceSession = await openGraph(gpu, route.shape, true, route.options);
          let absorbed: Observed;
          let reference: Observed;
          try {
            if (twinHere) {
              assertCheckCase(
                await checkFusionTwin(gpu, {
                  label: `${route.name}: absorbed vs reference`,
                  fused: absorbedSession,
                  plain: referenceSession,
                  feeds: attentionPermuteFeeds(route.shape),
                  counter: "attentionPermute",
                  hits: route.absorbedPermutes,
                  nan: "payload",
                }),
              );
            }
            // The output words and the census, from a run of each side of its own (the harness
            // returns a case).
            absorbed = await observe(absorbedSession, route.shape, false);
            reference = await observe(referenceSession, route.shape, true);
          } finally {
            await absorbedSession.dispose();
            await referenceSession.dispose();
          }
          assert(new Set(reference.output).size > 1, `${route.name}: the output is constant`);
          const keys = (census: readonly CensusRow[]): string =>
            census.map((row) => `${row.key}×${row.dispatchCount}`).join(" / ");
          assertEquals(
            dispatchesOf(absorbed.census),
            route.absorbed,
            `${route.name}: absorbed census — ${keys(absorbed.census)}`,
          );
          assertEquals(
            dispatchesOf(reference.census),
            route.reference,
            `${route.name}: reference census — ${keys(reference.census)}`,
          );
          if (route.fallsBack) {
            // The run is the three-dispatch route's: the same kernels and counts as a8 × "f32".
            const threeDispatch = await runGraph(gpu, route.shape, false, {
              attentionCompute: "a8",
              attentionScoreStorage: "f32",
            });
            assertEquals(
              keys(absorbed.census),
              keys(threeDispatch.census),
              `${route.name}: the a8 × "f32" census`,
            );
            assertWordIdentity(
              absorbed.output,
              threeDispatch.output,
              "payload",
              `${route.name}: absorbed vs a8 × "f32"`,
            );
          } else if (route.options.attentionScoreStorage === "none") {
            assert(
              absorbed.census.some((row) => row.key.startsWith("attention_online:")),
              `${route.name}: the online kernel runs — ${keys(absorbed.census)}`,
            );
            // The absorbed run stores O token-major; the reference stores it head-major.
            const tokenMajorStores = (census: readonly CensusRow[]): number =>
              countDispatches(
                census,
                (key) => key.startsWith("attention_online:") && key.includes(":obshd"),
              );
            assertEquals(tokenMajorStores(absorbed.census), 1, `${route.name}: absorbed O store`);
            assertEquals(tokenMajorStores(reference.census), 0, `${route.name}: reference O store`);
            // Both runs read Vᵀ in the key-axis form of N (`…:kpad` exactly when N % 4 != 0).
            const paddedReads = (census: readonly CensusRow[]): number =>
              countDispatches(
                census,
                (key) => key.startsWith("attention_online:") && key.endsWith(":kpad"),
              );
            const expectedPadded = attentionOnlineKeyWords(n) === "padded" ? 1 : 0;
            assertEquals(
              paddedReads(absorbed.census),
              expectedPadded,
              `${route.name}: absorbed key axis`,
            );
            assertEquals(
              paddedReads(reference.census),
              expectedPadded,
              `${route.name}: reference key axis`,
            );
          }
          console.log(
            `[attention permutes] ${route.name}: ${route.reference.total} → ${route.absorbed.total} dispatches, ${absorbed.output.length} words`,
          );
        } finally {
          gpu.destroy();
        }
      });
    }
  },
);
