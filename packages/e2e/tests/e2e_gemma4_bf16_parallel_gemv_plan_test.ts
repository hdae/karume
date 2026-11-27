/**
 * The parallel GEMV planned for every linear of the regular (bf16) Gemma 4 graphs at the preset
 * `bf16-fast` (no GPU — part 0 only). It lives in e2e because it reads the distribution mirror
 * `models/karume-gemma4/`.
 *
 * The preset declares `numerics: { level: "reorder" }`, which takes a linear to the parallel GEMV
 * only where the geometry profile's table has a row for its storage and shape (the default
 * profile's 17 bf16 rows — docs/spec/geometry-profiles.md); every other linear keeps the
 * sequential GEMV at one row and the GEMM above it. This pins, on the mirror's actual graphs and
 * with the same predicate as the recipe builder, which linear lands where under the default profile;
 * which lands elsewhere under `nvidia-ampere`, the table of the development machine's RTX 3080 Ti
 * (its rows on bf16 `1536 × 2048` and on int8 shapes of the drafters, which the default has none
 * for, and other lanes on bf16 `512 × 1536` and `2560 × 256`); and that the totals under that table
 * are the census table's `karume-gemma4` rows (`models/tests/helpers/census-table.ts`), which the real-GPU runs on
 * that machine compare with the dispatched kernels.
 *
 * Where the mirror is missing the cases SKIP explicitly.
 */
import { readTextFile } from "@karume/test-support/fs";
import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../../runtime/src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../../runtime/src/kernels/geometry-profiles/index.ts";
import {
  type LinearGemvChoices,
  linearGemvParallelEligible,
} from "../../runtime/src/kernels/linear-gemv.ts";
import type { WeightStorage } from "../../runtime/src/kernels/weight-storage.ts";
import { bindSymbols, planGraph } from "../../runtime/src/runtime/plan.ts";
import { planWeightResidency } from "../../runtime/src/runtime/weight-residency.ts";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "../../runtime/tests/helpers/container-graph.ts";
import { censusSeats } from "../../runtime/tests/helpers/pipeline-census.ts";
import { parseManifest } from "@karume/hub";
import { resolveGemmaSessionOptions } from "../../models/src/gemma/session-options.ts";
import { bundleKey, censusRowOf } from "../../models/tests/helpers/census-table.ts";
import { mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import { mirrorRoot } from "../../models/tests/helpers/mirror.ts";

const MIRROR_NAME = "karume-gemma4";
const MIRROR = mirrorRoot(MIRROR_NAME);
const PRESET = "bf16-fast";

const AVAILABLE = mirrorAvailable(MIRROR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRROR.pathname} is missing (or not karume/6), so the bf16 parallel GEMV plan check is SKIPPED`,
  );
}

/** The storage the GEMV selection reads for a weight's residency (a borrowed one fails loudly). */
const storageOf = (seat: string | undefined, where: string): WeightStorage => {
  switch (seat) {
    case "raw":
      return "f32";
    case "f16":
    case "bf16":
    case "i8":
    case "i4":
    case "i2":
      return seat;
    default:
      throw new Error(`${where}: a linear weight of residency '${seat}' is not counted here`);
  }
};

/** The GEMV choices of `nvidia-ampere`, the table the census rows were counted on. */
const AMPERE_GEMV = BUILTIN_GEOMETRY_PROFILES.find(({ id }) => id === "nvidia-ampere")?.linearGemv;

/**
 * Folds the linears of a plan at `rows` physical rows into `<storage> <n>×<k> l<lanes>` (the
 * parallel GEMV and its lanes) or `<storage> <n>×<k> sequential` → count, under `choices` (the
 * default profile's when absent).
 */
const planGemv = (
  graph: IrGraph,
  rows: number,
  choices: LinearGemvChoices = DEFAULT_GEOMETRY_PROFILE.linearGemv,
): Record<string, number> => {
  const residency = planWeightResidency(graph);
  const inputShapes = Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : rows)),
    ]),
  );
  const stateShapes = new Map(
    Object.entries(graph.states).map(([name, slot]) => [
      name,
      slot.shape.map((dim) => (typeof dim === "number" ? dim : 640)),
    ]),
  );
  const plan = planGraph(graph, bindSymbols(graph, inputShapes), stateShapes);
  const counts = new Map<string, number>();
  for (const step of plan.nodes) {
    if (step.node.op !== "linear") continue;
    const [x, weight] = step.inputShapes;
    const m = x.slice(0, -1).reduce((product, dim) => product * dim, 1);
    const seat = residency.get(step.node.ins[1]);
    const storage = storageOf(seat?.seat, step.node.ins[1]);
    const group = seat?.seat === "i4" ? seat.groupSize : undefined;
    const lanes = linearGemvParallelEligible(choices, storage, m, weight[0], weight[1], group);
    const key = `${storage} ${weight[0]}×${weight[1]} ${lanes === undefined ? "sequential" : `l${lanes}`}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Object.fromEntries([...counts].toSorted(([a], [b]) => (a < b ? -1 : 1)));
};

const sorted = (counts: Record<string, number>): Record<string, number> =>
  Object.fromEntries(Object.entries(counts).toSorted(([a], [b]) => (a < b ? -1 : 1)));

/** The parallel and sequential totals of a fold. */
const totals = (counts: Record<string, number>): { variant: number; reference: number } => {
  let variant = 0;
  let reference = 0;
  for (const [key, count] of Object.entries(counts)) {
    if (key.endsWith(" sequential")) reference += count;
    else variant += count;
  }
  return { variant, reference };
};

/**
 * Per component: the expected fold. The product graphs' linears are bf16; the drafters are int8
 * throughout (their one-row graph is planned at one row).
 */
const EXPECTED = {
  e2b: {
    model: {
      "bf16 1536×12288 l16": 20,
      "bf16 1536×256 l8": 35,
      "bf16 1536×4096 l32": 7,
      "bf16 1536×6144 l16": 15,
      "bf16 2048×1536 l16": 28,
      "bf16 256×1536 l32": 59,
      "bf16 4096×1536 l8": 7,
      "bf16 512×1536 l16": 6,
      "bf16 6144×1536 l8": 30,
      // No row: slower than the GEMM at 8 rows on the RTX 3080 Ti (8960 × 1536), or no candidate
      // measured at all three row counts on all three devices.
      "bf16 12288×1536 sequential": 40,
      "bf16 1536×2048 sequential": 28,
      "bf16 262144×1536 sequential": 1,
      "bf16 8960×1536 sequential": 1,
    },
    drafter: {
      "i8 1536×256 l4": 2,
      "i8 1024×256 sequential": 9,
      "i8 2048×256 sequential": 27,
      "i8 256×1024 sequential": 9,
      "i8 256×2048 sequential": 15,
      "i8 256×3072 sequential": 3,
      "i8 262144×256 sequential": 3,
    },
  },
  e4b: {
    model: {
      "bf16 1024×2560 l16": 8,
      "bf16 2048×2560 l32": 35,
      "bf16 2560×10240 l16": 42,
      "bf16 2560×2048 l8": 35,
      "bf16 2560×4096 l16": 7,
      "bf16 256×2560 l32": 42,
      "bf16 4096×2560 l8": 7,
      "bf16 512×2560 l32": 40,
      "bf16 10240×2560 sequential": 84,
      "bf16 10752×2560 sequential": 1,
      "bf16 2560×256 sequential": 42,
      "bf16 262144×2560 sequential": 1,
    },
    drafter: {
      "i8 2560×256 l4": 2,
      "i8 1024×256 sequential": 9,
      "i8 2048×256 sequential": 27,
      "i8 256×1024 sequential": 9,
      "i8 256×2048 sequential": 15,
      "i8 256×5120 sequential": 3,
      "i8 262144×256 sequential": 3,
    },
  },
} as const;

/**
 * Per component: the linears that `nvidia-ampere` runs elsewhere than the default profile's fold —
 * a shape the default has no row for, or a row with other lanes —, `<storage> <n>×<k>` → the lanes
 * of that table's row. The rest of its fold is the default's.
 */
const AMPERE_MOVED: Record<keyof typeof EXPECTED, Record<string, Record<string, number>>> = {
  e2b: {
    model: { "bf16 1536×2048": 32, "bf16 512×1536": 32 },
    drafter: {
      "i8 1024×256": 4,
      "i8 2048×256": 4,
      "i8 256×1024": 32,
      "i8 256×2048": 32,
      "i8 256×3072": 32,
    },
  },
  e4b: {
    model: { "bf16 2560×256": 8 },
    // nvidia-ampere has no row for the pre_projection 256×5120, which E2B's 256×3072 has.
    drafter: {
      "i8 1024×256": 4,
      "i8 2048×256": 4,
      "i8 256×1024": 32,
      "i8 256×2048": 32,
    },
  },
};

/** The default profile's fold with the linears `moved` names on the parallel GEMV at its lanes. */
const movedFold = (
  counts: Record<string, number>,
  moved: Record<string, number>,
): Record<string, number> =>
  sorted(
    Object.fromEntries(
      Object.entries(counts).map(([key, count]) => {
        const shape = key.slice(0, key.lastIndexOf(" "));
        const lanes = moved[shape];
        return [lanes === undefined ? key : `${shape} l${lanes}`, count];
      }),
    ),
  );

describe.skipIf(!AVAILABLE)(
  "parallel GEMV on the regular Gemma 4 bf16 graphs at bf16-fast (plan, no GPU needed)",
  () => {
    for (const model of ["e2b", "e4b"] as const) {
      it(`${model}: the linears with a bf16 row take the parallel GEMV at 1 to 8 rows, and the totals under nvidia-ampere are the census table's row`, async () => {
        assert(AMPERE_GEMV !== undefined, "nvidia-ampere carries no GEMV choices");
        const text = await readTextFile(new URL("karume.json", MIRROR));
        const entry = parseManifest(text).models[model];
        const effective = resolveGemmaSessionOptions(entry.quants[PRESET].session, {}, model);
        // The preset resolves to the parallel GEMV and the fused states-form attention.
        assertEquals(
          bundleKey(censusSeats(effective)),
          "linearGemvReduce=parallel,stateAttentionReduce=parallel-fused",
        );
        const row = censusRowOf("gemma4", model, effective, MIRROR_NAME);
        assert(
          row !== undefined && row.family === "gemma4",
          `no census row for ${MIRROR_NAME} ${model} ${PRESET}`,
        );

        const manifest: ContainerManifest = JSON.parse(text);
        const weights = entry.quants[PRESET].weights;
        const productStorage = weights.model;
        assert(productStorage !== undefined, `${model} ${PRESET}: no model weights`);
        const product = await readContainerGraph(
          new URL(containerPart0(manifest, model, "model", productStorage).path, MIRROR),
          "model",
        );
        const fold = EXPECTED[model];
        const moved = AMPERE_MOVED[model];
        const expected = sorted(fold.model);
        const ampere = movedFold(expected, moved.model ?? {});
        for (const rows of [1, 4, 8]) {
          assertEquals(planGemv(product, rows), expected, `${model} M=${rows}`);
          assertEquals(planGemv(product, rows, AMPERE_GEMV), ampere, `${model} M=${rows} ampere`);
        }
        // M > 8 is outside the parallel GEMV (the rows' range).
        assertEquals(totals(planGemv(product, 9)).variant, 0, `${model} M=9`);
        assertEquals(totals(planGemv(product, 9, AMPERE_GEMV)).variant, 0, `${model} M=9 ampere`);
        for (const phase of ["decode", "verify"] as const) {
          assertEquals(
            row.census.target?.[phase]?.linearGemvReduce,
            totals(ampere),
            `${model} ${phase}: the census row`,
          );
        }

        if (!("drafter" in fold)) {
          assertEquals(weights.drafter, undefined, `${model}: ships no drafter`);
          return;
        }
        const drafterStorage = weights.drafter;
        assert(drafterStorage !== undefined, `${model} ${PRESET}: no drafter weights`);
        const drafter = await readContainerGraph(
          new URL(containerPart0(manifest, model, "drafter", drafterStorage).path, MIRROR),
          "drafter",
        );
        const drafterExpected = sorted(fold.drafter);
        const drafterAmpere = movedFold(drafterExpected, moved.drafter ?? {});
        assertEquals(planGemv(drafter, 1), drafterExpected, `${model} drafter`);
        assertEquals(planGemv(drafter, 1, AMPERE_GEMV), drafterAmpere, `${model} drafter ampere`);
        assertEquals(
          row.census.drafter?.draft?.linearGemvReduce,
          totals(drafterAmpere),
          `${model} draft: the census row`,
        );
      });
    }
  },
);
