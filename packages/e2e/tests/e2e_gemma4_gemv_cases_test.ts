/**
 * The GEMV sweep's case table against the real Gemma 4 graphs (no GPU): every linear over int2 /
 * int4 / int8 weights that a decode token of a model or a run of its MTP drafter executes is a case
 * of the sweep, so every device's table can be derived for it. It lives in e2e because it reads the
 * distribution mirrors `models/karume-gemma4*`; only part 0 of each container (the graph and the
 * binding table) is read, never a weight block.
 */
import { isNotFound, readTextFile, stat } from "@karume/test-support/fs";
import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import { bindSymbols, planGraph } from "../../runtime/src/runtime/plan.ts";
import { planWeightResidency } from "../../runtime/src/runtime/weight-residency.ts";
import { GEMV_CASES, type GemvCase } from "../../runtime/src/tune/gemv-cases.ts";
import {
  type ContainerManifest,
  readContainerGraph,
} from "../../runtime/tests/helpers/container-graph.ts";

const MIRRORS = ["karume-gemma4", "karume-gemma4-qat", "karume-gemma4-qat-mobile"] as const;

/** The components whose graphs a decode runs: the model, and the MTP drafter where one ships. */
const COMPONENTS = ["model", "drafter"] as const;

const rootOf = (mirror: string): URL => new URL(`../../../models/${mirror}/`, import.meta.url);

/**
 * Whether the asset exists (anything but NotFound propagates — a mount fault does not become a
 * SKIP).
 */
const exists = async (url: URL): Promise<boolean> => {
  try {
    await stat(url);
    return true;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const PRESENT = await Promise.all(
  MIRRORS.map((mirror) => exists(new URL("karume.json", rootOf(mirror)))),
);
const AVAILABLE = PRESENT.every((present) => present);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRRORS.filter((_, index) => !PRESENT[index])
      .map((mirror) => `models/${mirror}/karume.json`)
      .join(
        ", ",
      )} missing, so the check of the GEMV sweep's cases against the Gemma 4 graphs is SKIPPED`,
  );
}

/** One integer-storage linear shape of a graph at one row. */
type IntegerLinear = {
  readonly storage: "i2" | "i4" | "i8";
  readonly group?: number;
  readonly n: number;
  readonly k: number;
};

/** The integer-storage linears of a graph planned at one physical row (a decode token, a drafter run). */
const integerLinears = (graph: IrGraph): IntegerLinear[] => {
  const residency = planWeightResidency(graph);
  const inputShapes = Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : 1)),
    ]),
  );
  // The state lengths do not reach a linear's shape; any length plans.
  const stateShapes = new Map(
    Object.entries(graph.states).map(([name, slot]) => [
      name,
      slot.shape.map((dim) => (typeof dim === "number" ? dim : 640)),
    ]),
  );
  const plan = planGraph(graph, bindSymbols(graph, inputShapes), stateShapes);
  const linears: IntegerLinear[] = [];
  for (const step of plan.nodes) {
    if (step.node.op !== "linear") continue;
    const [x, weight] = step.inputShapes;
    const m = x.slice(0, -1).reduce((product, dim) => product * dim, 1);
    assertEquals(m, 1, `${step.node.ins[1]} runs ${m} rows in a graph planned at one row`);
    const seat = residency.get(step.node.ins[1]);
    if (seat?.seat !== "i2" && seat?.seat !== "i4" && seat?.seat !== "i8") continue;
    linears.push({
      storage: seat.seat,
      ...(seat.seat === "i4" ? { group: seat.groupSize } : {}),
      n: weight[0],
      k: weight[1],
    });
  }
  return linears;
};

/** Every model, component and storage of the three mirrors, with the integer linears of its graph. */
const mirrorLinears = async (): Promise<
  { readonly where: string; readonly linears: readonly IntegerLinear[] }[]
> => {
  const found: { where: string; linears: IntegerLinear[] }[] = [];
  for (const mirror of MIRRORS) {
    const root = rootOf(mirror);
    const manifest: ContainerManifest = JSON.parse(
      await readTextFile(new URL("karume.json", root)),
    );
    for (const [model, entry] of Object.entries(manifest.models)) {
      for (const component of COMPONENTS) {
        for (const [dtype, weights] of Object.entries(entry.weights[component] ?? {})) {
          const [head] = weights.container.parts;
          assertEquals(head.repo, undefined, `${mirror} ${model} ${component} ${dtype}`);
          const graph = await readContainerGraph(new URL(head.path, root), component);
          found.push({
            where: `${mirror} ${model} ${component} ${dtype}`,
            linears: integerLinears(graph),
          });
        }
      }
    }
  }
  return found;
};

const spelled = ({ storage, group, n, k }: IntegerLinear): string =>
  `${storage}${group === undefined ? "" : ` g${group}`} ${n}×${k}`;

/** The shapes of `linears` the case table has no case of, each once, in the order met. */
const withoutCase = (linears: readonly IntegerLinear[], cases: readonly GemvCase[]): string[] => [
  ...new Set(
    linears
      .filter(
        (linear) =>
          !cases.some(
            (gemvCase) =>
              gemvCase.storage === linear.storage &&
              gemvCase.group === linear.group &&
              gemvCase.n === linear.n &&
              gemvCase.k === linear.k,
          ),
      )
      .map(spelled),
  ),
];

describe.skipIf(!AVAILABLE)(
  "the GEMV sweep's case table covers the Gemma 4 graphs (plan, no GPU needed)",
  () => {
    it("has a case for every int2 / int4 / int8 linear of a decode token and of a drafter run, in every mirror", async () => {
      const graphs = await mirrorLinears();
      // The three mirrors ship 9 such graphs with integer linears (the bf16 models have none).
      assert(graphs.filter(({ linears }) => linears.length > 0).length >= 9);
      for (const { where, linears } of graphs) {
        assertEquals(withoutCase(linears, GEMV_CASES), [], where);
      }
    });

    it("names the shape a case table without the q4_0 E2B output head misses (fault: one case removed)", async () => {
      const graphs = await mirrorLinears();
      const without = GEMV_CASES.filter(({ id }) => id !== "i4g32-n262144-k1536");
      const missed = graphs.flatMap(({ where, linears }) =>
        withoutCase(linears, without).map((shape) => `${where}: ${shape}`),
      );
      assertEquals(missed, ["karume-gemma4-qat e2b model i4: i4 g32 262144×1536"]);
    });
  },
);
