/**
 * The GEMV sweep's case table and the default profile's parallel rows against the real Gemma 4
 * graphs (no GPU): every linear over int2 / int4 / int8 weights that a decode token of a model or a
 * run of its MTP drafter executes is a case of the sweep, so every device's table can be derived for
 * it; and the linears that stay on the sequential GEMV at the level `reorder` with the default
 * profile are exactly the listed ones (the census of
 * docs/research/2026-11-18-gemma4-gemv-census.md). It lives in e2e because it reads the
 * distribution mirrors `models/karume-gemma4*`; only part 0 of each container (the graph and the
 * binding table) is read, never a weight block.
 *
 * What it covers, and what it does not:
 * - The components `model` and `drafter` only (what a decode token and a drafter run execute). A
 *   mirror's other components (`vision`) are not read.
 * - Both directions between the graphs and the case table. Graph → case is asked of the integer
 *   linears only: a graph's int2 / int4 / int8 linear without a case fails, a float-stored one does
 *   not (the regular models' bf16 linears all have a case, the mobile models' one f32 linear has
 *   none). Case → graph is asked of every case, the bf16 ones included: a case whose storage and
 *   shape no graph's linear has fails, unless it is listed with its reason
 *   ({@link CASES_WITHOUT_A_GRAPH}).
 * - Which linears take a parallel row is asked at M = 1 and of the default profile only: not at
 *   the four or eight rows of a verify run, and not of a per-adapter table that carries its own
 *   GEMV choices.
 * - It reads all three mirrors, the mobile QAT one included, and runs in the `models:gemma4` lane
 *   (its file name carries no `gemma4_qat_mobile`): a machine without one of the three mirrors skips
 *   the whole file.
 * - The rows are matched by shape, so "the linears on a row" is a statement about these graphs. A
 *   row's shape can occur in another family's graph (int4 g32 1024×4096 does, in the SBV2 text
 *   encoder), which this file does not read.
 */
import { isNotFound, readTextFile, stat } from "@karume/test-support/fs";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../../runtime/src/kernels/geometry-profile.ts";
import {
  type LinearGemvChoices,
  linearGemvParallelEligible,
} from "../../runtime/src/kernels/linear-gemv.ts";
import { bindSymbols, planGraph } from "../../runtime/src/runtime/plan.ts";
import {
  planWeightResidency,
  type WeightResidency,
} from "../../runtime/src/runtime/weight-residency.ts";
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

/** One float-storage linear shape of a graph at one row (the storage the GEMV sweep names it by). */
type FloatLinear = {
  readonly storage: "bf16" | "f16" | "f32";
  readonly n: number;
  readonly k: number;
};

/**
 * The seats whose weight a linear reads as float words, with the storage a GEMV sweep case names
 * such a weight by: bf16 and f16 are resident as stored, `raw` is f32 as stored and `expanded` is
 * decoded to f32 at load.
 */
const FLOAT_SEATS: Readonly<Record<string, FloatLinear["storage"]>> = {
  bf16: "bf16",
  f16: "f16",
  raw: "f32",
  expanded: "f32",
};

/**
 * The integer storage of a linear's weight seat, or `undefined` for a float seat. Any other seat
 * fails: a borrowed weight (`shared`) is resident in its lender's storage, which the borrower's
 * graph does not say, and `i4f16` is `expert_linear`'s — a linear over either would be counted as
 * "no integer weight" if it were skipped.
 */
const integerStorageOf = (
  seat: WeightResidency | undefined,
  where: string,
): Pick<IntegerLinear, "storage" | "group"> | undefined => {
  if (seat?.seat === "i4") return { storage: "i4", group: seat.groupSize };
  if (seat?.seat === "i2" || seat?.seat === "i8") return { storage: seat.seat };
  if (seat !== undefined && Object.hasOwn(FLOAT_SEATS, seat.seat)) return undefined;
  throw new Error(
    `${where}: the weight's seat is '${seat?.seat ?? "(none)"}', and this test does not know whether such a linear reads integer weights`,
  );
};

/**
 * The linears of a graph planned at one physical row (a decode token, a drafter run): the
 * integer-storage ones, and the float-storage ones apart.
 */
const graphLinears = (
  graph: IrGraph,
): { readonly linears: IntegerLinear[]; readonly floats: FloatLinear[] } => {
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
  const floats: FloatLinear[] = [];
  for (const step of plan.nodes) {
    if (step.node.op !== "linear") continue;
    const [x, weight] = step.inputShapes;
    const m = x.slice(0, -1).reduce((product, dim) => product * dim, 1);
    assertEquals(m, 1, `${step.node.ins[1]} runs ${m} rows in a graph planned at one row`);
    const seat = residency.get(step.node.ins[1]);
    const storage = integerStorageOf(seat, step.node.ins[1]);
    if (storage !== undefined) linears.push({ ...storage, n: weight[0], k: weight[1] });
    // A seat that is neither integer nor float has thrown above.
    else if (seat !== undefined) {
      floats.push({ storage: FLOAT_SEATS[seat.seat], n: weight[0], k: weight[1] });
    }
  }
  return { linears, floats };
};

/**
 * Every `model` and `drafter` graph of the three mirrors (`<mirror> <model> <component> <storage>`)
 * with the number of its integer linears at one row. The set is exact: a graph that appears or goes,
 * or whose count moves, fails here before any other case reads it (a graph with a row for every
 * linear has no entry in {@link SEQUENTIAL_AT_REORDER}, so nothing else would notice its loss).
 */
const INTEGER_LINEARS: Readonly<Record<string, number>> = {
  // The regular line is bf16; only its E2B ships a drafter (int8).
  "karume-gemma4 e2b model bf16": 0,
  "karume-gemma4 e2b drafter i8": 68,
  "karume-gemma4 e4b model bf16": 0,
  "karume-gemma4-qat e2b model i4": 277,
  "karume-gemma4-qat e2b drafter i4": 68,
  "karume-gemma4-qat e4b model i4": 344,
  "karume-gemma4-qat e4b drafter i4": 68,
  "karume-gemma4-qat 12b model i4": 329,
  "karume-gemma4-qat 12b drafter i4": 68,
  // 26B's 60 routed experts are `expert_linear`, not `linear`.
  "karume-gemma4-qat 26b model i4": 236,
  // One more linear per token is the f32 per_layer_model_projection.
  "karume-gemma4-qat-mobile e2b model i4": 276,
  "karume-gemma4-qat-mobile e4b model i4": 343,
};

/**
 * The graphs with float-stored linears, each with their number at one row: the regular line's models
 * in bf16, and the f32 per_layer_model_projection of the mobile models. A graph that is not listed
 * has none.
 */
const FLOAT_LINEARS: Readonly<Record<string, number>> = {
  "karume-gemma4 e2b model bf16": 277,
  "karume-gemma4 e4b model bf16": 344,
  "karume-gemma4-qat-mobile e2b model i4": 1,
  "karume-gemma4-qat-mobile e4b model i4": 1,
};

/**
 * The cases whose shape no graph of the mirrors has, each with why it is a case all the same. A
 * case that is not listed and matches no graph fails: the sweep would measure, on every device, a
 * shape nothing runs.
 */
const CASES_WITHOUT_A_GRAPH: Readonly<Record<string, string>> = {
  "i8-n262144-k1536":
    "the regular E2B's int8 output head: the line's presets are bf16 since 2026-11-04, but the default " +
    "profile still holds this shape's parallel row, and every row of the profile has a case",
  "i8-n262144-k2560":
    "the regular E4B's int8 output head: as E2B's — the default profile still holds its parallel row",
};

/**
 * The ids of the cases of `cases` whose storage and shape none of `linears` has, in table order (a
 * float-stored linear and a float case have no group length).
 */
const casesWithoutGraph = (
  linears: readonly (IntegerLinear | FloatLinear)[],
  cases: readonly Pick<GemvCase, "id" | "storage" | "group" | "n" | "k">[],
): string[] =>
  cases
    .filter(
      (gemvCase) =>
        !linears.some(
          (linear) =>
            linear.storage === gemvCase.storage &&
            ("group" in linear ? linear.group : undefined) === gemvCase.group &&
            linear.n === gemvCase.n &&
            linear.k === gemvCase.k,
        ),
    )
    .map(({ id }) => id);

/**
 * Every model, component and storage of the three mirrors, with the integer linears of its graph
 * (`linears`) and its float-stored ones (`floats`).
 */
const mirrorLinears = async (): Promise<
  {
    readonly where: string;
    readonly linears: readonly IntegerLinear[];
    readonly floats: readonly FloatLinear[];
  }[]
> => {
  const found: { where: string; linears: IntegerLinear[]; floats: FloatLinear[] }[] = [];
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
            ...graphLinears(graph),
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

/**
 * The integer linears of `linears` that take no parallel row of `choices` at one row (the recipe
 * builder's predicate), as `<shape> ×<count>` in the order met: what a Session with linear at
 * `reorder` still runs on the sequential GEMV.
 */
const sequentialAtReorder = (
  linears: readonly IntegerLinear[],
  choices: LinearGemvChoices,
): string[] => {
  const counts = new Map<string, number>();
  for (const linear of linears) {
    const { storage, n, k, group } = linear;
    if (linearGemvParallelEligible(choices, storage, 1, n, k, group) !== undefined) continue;
    counts.set(spelled(linear), (counts.get(spelled(linear)) ?? 0) + 1);
  }
  return [...counts].map(([shape, count]) => `${shape} ×${count}`);
};

/**
 * Per graph, the integer linears the default profile gives no parallel row (a graph that is not
 * listed has none). Each is on the sequential GEMV at `reorder` for a measured reason, recorded in
 * docs/research/2026-11-18-gemma4-gemv-census.md.
 */
const SEQUENTIAL_AT_REORDER: Readonly<Record<string, readonly string[]>> = {
  "karume-gemma4 e2b drafter i8": [
    "i8 256×3072 ×3",
    "i8 1024×256 ×9",
    "i8 256×1024 ×9",
    "i8 2048×256 ×27",
    "i8 256×2048 ×15",
    "i8 262144×256 ×3",
  ],
  "karume-gemma4-qat e2b model i4": ["i4 g32 262144×1536 ×1"],
  "karume-gemma4-qat e2b drafter i4": [
    "i4 g32 1024×256 ×9",
    "i4 g32 2048×256 ×27",
    "i4 g32 256×2048 ×15",
    "i4 g32 262144×256 ×3",
  ],
  "karume-gemma4-qat e4b model i4": ["i4 g32 262144×2560 ×1"],
  "karume-gemma4-qat e4b drafter i4": [
    "i4 g32 256×5120 ×3",
    "i4 g32 1024×256 ×9",
    "i4 g32 2048×256 ×27",
    "i4 g32 256×2048 ×15",
    "i4 g32 262144×256 ×3",
  ],
  "karume-gemma4-qat 12b drafter i4": [
    "i4 g32 4096×1024 ×9",
    "i4 g32 8192×1024 ×27",
    "i4 g32 3840×1024 ×2",
    "i4 g32 262144×1024 ×3",
  ],
  "karume-gemma4-qat-mobile e2b model i4": ["i2 262144×1536 ×1"],
  "karume-gemma4-qat-mobile e4b model i4": ["i2 262144×2560 ×1"],
};

describe.skipIf(!AVAILABLE)(
  "the default profile's parallel rows on the Gemma 4 graphs (plan, no GPU needed)",
  () => {
    it("leaves exactly the listed int2 / int4 / int8 linears on the sequential GEMV at reorder", async () => {
      const graphs = await mirrorLinears();
      const listed = new Set(Object.keys(SEQUENTIAL_AT_REORDER));
      for (const { where, linears } of graphs) {
        assertEquals(
          sequentialAtReorder(linears, DEFAULT_GEOMETRY_PROFILE.linearGemv),
          SEQUENTIAL_AT_REORDER[where] ?? [],
          where,
        );
        listed.delete(where);
      }
      assertEquals([...listed], [], "a listed graph is not in the mirrors");
    });

    it("lists a linear whose row is taken away (fault: the 12B output head's row removed)", async () => {
      const graphs = await mirrorLinears();
      const defaults = DEFAULT_GEOMETRY_PROFILE.linearGemv;
      const without: LinearGemvChoices = {
        ...defaults,
        parallelShapes: defaults.parallelShapes.filter(
          ({ storage, n, k }) => !(storage === "i4" && n === 262144 && k === 3840),
        ),
      };
      const model = graphs.find(({ where }) => where === "karume-gemma4-qat 12b model i4");
      assert(model !== undefined);
      assertEquals(sequentialAtReorder(model.linears, defaults), []);
      assertEquals(sequentialAtReorder(model.linears, without), ["i4 g32 262144×3840 ×1"]);
    });
  },
);

describe.skipIf(!AVAILABLE)(
  "the GEMV sweep's case table covers the Gemma 4 graphs (plan, no GPU needed)",
  () => {
    it("reads exactly the model and drafter graphs the mirrors ship, each with its count of integer linears", async () => {
      const graphs = await mirrorLinears();
      assertEquals(
        Object.fromEntries(graphs.map(({ where, linears }) => [where, linears.length])),
        INTEGER_LINEARS,
      );
      assertEquals(
        Object.fromEntries(
          graphs
            .filter(({ floats }) => floats.length > 0)
            .map((graph) => [graph.where, graph.floats.length]),
        ),
        FLOAT_LINEARS,
      );
    });

    it("fails on a linear whose weight seat it cannot class as integer or float (fault: a borrowed weight, an expert weight)", () => {
      const borrowed: WeightResidency = {
        seat: "shared",
        eligible: true,
        i4Eligible: true,
        i2Eligible: false,
      };
      const expert: WeightResidency = {
        seat: "i4f16",
        payloadBytes: 16,
        scaleBytes: 4,
        groupSize: 32,
      };
      assertThrows(() => integerStorageOf(borrowed, "w"), Error, "the weight's seat is 'shared'");
      assertThrows(() => integerStorageOf(expert, "w"), Error, "the weight's seat is 'i4f16'");
      assertThrows(() => integerStorageOf(undefined, "w"), Error, "the weight's seat is '(none)'");
      assertEquals(integerStorageOf({ seat: "bf16", payloadBytes: 8 }, "w"), undefined);
      assertEquals(
        integerStorageOf({ seat: "i4", payloadBytes: 8, scaleBytes: 4, groupSize: 32 }, "w"),
        { storage: "i4", group: 32 },
      );
    });

    it("has a case for every int2 / int4 / int8 linear of a decode token and of a drafter run, in every mirror", async () => {
      const graphs = await mirrorLinears();
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

    it("holds no case — integer or bf16 — whose shape no graph has, but the listed ones", async () => {
      const linears = (await mirrorLinears()).flatMap((graph) => [
        ...graph.linears,
        ...graph.floats,
      ]);
      assertEquals(casesWithoutGraph(linears, GEMV_CASES), Object.keys(CASES_WITHOUT_A_GRAPH));
    });

    it("names a case whose shape no graph has (faults: an int4 and a bf16 case of shapes nothing runs)", async () => {
      const linears = (await mirrorLinears()).flatMap((graph) => [
        ...graph.linears,
        ...graph.floats,
      ]);
      const orphan = {
        id: "i4g32-n4096-k4096",
        storage: "i4",
        group: 32,
        n: 4096,
        k: 4096,
      } as const;
      // 12B's sliding k / v shape, which the mirrors hold in int4 g32 and not in bf16.
      const floatOrphan = { id: "bf16-n2048-k3840", storage: "bf16", n: 2048, k: 3840 } as const;
      assertEquals(casesWithoutGraph(linears, [...GEMV_CASES, orphan, floatOrphan]), [
        ...Object.keys(CASES_WITHOUT_A_GRAPH),
        orphan.id,
        floatOrphan.id,
      ]);
    });

    it("misses the bf16 cases when the float-stored linears are left out (fault: the integer linears alone)", async () => {
      const integer = (await mirrorLinears()).flatMap((graph) => graph.linears);
      const missed = casesWithoutGraph(integer, GEMV_CASES);
      assertEquals(
        missed.filter((id) => !id.startsWith("bf16-")),
        Object.keys(CASES_WITHOUT_A_GRAPH),
      );
      assertEquals(
        missed.filter((id) => id.startsWith("bf16-")),
        GEMV_CASES.filter(({ storage }) => storage === "bf16").map(({ id }) => id),
      );
    });
  },
);
