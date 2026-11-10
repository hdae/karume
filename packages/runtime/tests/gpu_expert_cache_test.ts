// The exact expert cache through a Session on a real GPU (ADR 0146): two routed layers of 16 experts
// (top-4, int4-sym-g-f16, group 32) built twice from the same container — every expert resident, and
// `expertCache: { slots: 8 }`. The cache runs a row, loads what it missed and runs it again until no
// pick misses; that run's output MUST be bit-identical to full residency (the bytes, not a
// tolerance). A second input that needs other experts evicts and converges the same way, and the
// counters count every run and load. With one slot count per layer (`slots: [8, 12]`), each layer's
// pools, slots and evictions are its own, and the exact runs are the same bytes.
//
// A missed run is never exact: its first run's output differs from full residency (the fault
// injection — committing a missed row would fail the comparison). The same protocol drives a
// states-form graph (attention over a full KV slot, then the expert layers): a deferred run that
// misses is discarded with commit(0), the experts load, the same position runs again and commits.
// On the same graph, self-speculation's pieces: drafts as chained deferred runs dropped by one
// commit(0), and a multi-row verify whose rows before its first missing row are committed alone —
// those rows are the resident Session's bytes, the first missing row is not.
//
// Both containers of a source are covered: the in-memory container (whole bytes, verified) and a
// `krm` cut into 2048-byte blocks (gate_up one expert a piece with its scale split along them, down
// two a piece; unverified, so every range is checked against its construction digest).
//
// Every load goes through the staging ring; the last describe adds the host tier (`hostBytes`) and
// loads larger than the ring (rounds).

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { openContainer } from "../src/format/container/open.ts";
import { parseIrDeclarationValue } from "../src/format/ir.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import {
  createSessionFromContainer,
  type ExpertCache,
  type ExpertRun,
  type GenerationRun,
  type RunInputs,
  type Session,
  type Tensor,
} from "../src/runtime/executor.ts";
import { EXPERT_RING_SLOTS } from "../src/runtime/expert-plan.ts";
import { writeModelContainer } from "./helpers/container-write.ts";
import {
  EXPERT_OPS,
  expertGraph,
  type ExpertGraphShape,
  expertLayers,
  expertTensors,
  waveValues,
} from "./helpers/expert-graph.ts";
import { fullStateGraph } from "./helpers/full-state-graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { type DeclarationJson, GRAPH_NAME, memoryModel } from "./helpers/model-fixture.ts";

const SHAPE: ExpertGraphShape = { experts: 16, picks: 4, width: 64, ffn: 32, layers: 2 };
const SLOTS = 8;
const ROWS = 2;

/** The u32 words of an f32 output (bit identity, not a tolerance). */
const bits = (tensor: Tensor | undefined): number[] => {
  if (tensor?.dtype !== "f32") throw new Error("not an f32 output");
  return [...new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length)];
};

/** The two containers a source can be (module doc). */
const KINDS = ["in-memory", "krm of 2048-byte blocks"] as const;

/** One container of a declaration and its weights. */
const containerOf = async (
  kind: (typeof KINDS)[number],
  declaration: DeclarationJson,
  tensors: ReturnType<typeof expertTensors>,
): Promise<BoundContainer> => {
  if (kind === "in-memory") return memoryModel(declaration, tensors);
  const written = await writeModelContainer(
    {
      graphs: { [GRAPH_NAME]: parseIrDeclarationValue(declaration) },
      consts: [],
      weights: tensors,
      assets: [],
      provenance: { license: "test" },
    },
    { blockBytes: 2048 },
  );
  return await openContainer({ kind: "parts", parts: written.parts });
};

/** Every cache run and load of one exact run, in order. */
type Converged = {
  readonly runs: readonly ExpertRun[];
  readonly loads: readonly {
    readonly loaded: number;
    readonly evicted: number;
    readonly bytes: number;
  }[];
};

/**
 * Runs until a run misses nothing (the routing of a later layer is only right once the earlier
 * layers are exact, so it can take one round per layer). `between` runs after a missed run, before
 * its load (the generation case discards the deferred rows there).
 */
const converge = async (
  cache: ExpertCache,
  run: () => Promise<ExpertRun>,
  between: () => void = () => undefined,
): Promise<Converged> => {
  const runs: ExpertRun[] = [];
  const loads: Converged["loads"][number][] = [];
  for (let round = 0; round <= cache.layers; round += 1) {
    const result = await run();
    runs.push(result);
    if (result.missing === 0) return { runs, loads };
    between();
    loads.push(await cache.load(result.routing));
  }
  throw new Error(`no exact run within ${cache.layers + 1} rounds`);
};

/** One expert's bytes: gate_up 2048 codes + 256 steps, down 1024 + 128. */
const EXPERT_BYTES = 2048 + 256 + 1024 + 128;

const withSessions = async (
  container: BoundContainer,
  body: (resident: Session, cached: Session, cache: ExpertCache) => Promise<void>,
  slots: number | readonly number[] = SLOTS,
): Promise<void> => {
  const gpu: GpuContext = await acquireGpu();
  try {
    const resident = await createSessionFromContainer(gpu, container, GRAPH_NAME);
    const cached = await createSessionFromContainer(gpu, container, GRAPH_NAME, {
      expertCache: { slots },
    });
    try {
      const cache = cached.expertCache;
      assert(cache !== undefined, "no expert cache");
      await body(resident, cached, cache);
    } finally {
      await cached.dispose();
      await resident.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

describe.skipIf(!GPU_AVAILABLE)("the exact expert cache through a Session (real GPU)", () => {
  const tensors = expertTensors(SHAPE, "i4f16");
  for (const kind of KINDS) {
    it(`runs bit-identical to full residency once nothing misses, through evictions (${kind})`, async () => {
      const container = await containerOf(kind, expertGraph(SHAPE), tensors);
      await withSessions(container, async (resident, cached, cache) => {
        assertEquals(
          [cache.slots, cache.experts, cache.layers, cache.picks],
          [[SLOTS, SLOTS], 16, 2, 4],
        );
        // The pools hold half the experts (per layer 16 · 3,456 bytes resident, 8 · 3,456 pooled).
        assertEquals(resident.diagnostics().storage.residentCompressedBytes, 2 * 16 * EXPERT_BYTES);
        assertEquals(
          cached.diagnostics().storage.residentCompressedBytes,
          2 * SLOTS * EXPERT_BYTES,
        );
        const loaded = { experts: 0, evicted: 0, bytes: 0, loads: 0, runs: 0, missed: 0 };
        const tally = (converged: Converged): void => {
          loaded.runs += converged.runs.length;
          loaded.missed += converged.runs.filter((run) => run.missing > 0).length;
          for (const load of converged.loads) {
            loaded.experts += load.loaded;
            loaded.evicted += load.evicted;
            loaded.bytes += load.bytes;
            if (load.loaded > 0) loaded.loads += 1;
          }
        };
        let evictedBefore = 0;
        for (const [label, seed] of [
          ["first input", 13],
          ["second input", 29],
        ] as const) {
          const inputs: RunInputs = {
            x: { dtype: "f32", shape: [ROWS, 64], data: waveValues(ROWS * 64, seed, 1.5) },
          };
          const expected = await resident.run(inputs);
          const converged = await converge(cache, () => cache.run(inputs));
          tally(converged);
          const [first] = converged.runs;
          const last = converged.runs.at(-1);
          // MUST (precondition): the input misses, so the exact run is a run again after loads.
          assert(first.missing > 0, `${label}: the first run missed nothing`);
          assert(last !== undefined && last.missing === 0);
          assertEquals(last.routing.rows, ROWS);
          // Fault injection: the missed run is not the resident one (committing it would be wrong).
          assertNotEquals(
            bits(first.outputs["out"]),
            bits(expected["out"]),
            `${label}: missed run`,
          );
          assertEquals(bits(last.outputs["out"]), bits(expected["out"]), `${label}: exact run`);
          // Only the declared outputs reach the caller (the index readback is the cache's).
          assertEquals(Object.keys(last.outputs), ["out"]);
          cache.touch(last.routing, ROWS);
          if (label === "second input") {
            // MUST (precondition): the second input needs experts the first left no free slot for.
            assert(loaded.evicted > evictedBefore, "the second input evicted nothing");
          }
          evictedBefore = loaded.evicted;
        }
        // Routing that is all resident loads nothing and counts nothing.
        const exact = await cache.run({
          x: { dtype: "f32", shape: [ROWS, 64], data: waveValues(ROWS * 64, 29, 1.5) },
        });
        assertEquals(exact.missing, 0);
        assertEquals((await cache.load(exact.routing)).loaded, 0);
        loaded.runs += 1;
        assertEquals(loaded.bytes, loaded.experts * EXPERT_BYTES);
        const stats = cache.stats();
        // Without a host tier every expert is read from the container and uploaded as read.
        assertEquals(stats, {
          runs: loaded.runs,
          missedRuns: loaded.missed,
          loads: loaded.loads,
          loadedExperts: loaded.experts,
          evictedExperts: loaded.evicted,
          hostExperts: 0,
          containerExperts: loaded.experts,
          bytesRead: loaded.bytes,
          bytesUploaded: loaded.bytes,
          loadMs: stats.loadMs,
          readMs: stats.readMs,
          uploadMs: stats.uploadMs,
          hostHeldExperts: 0,
          hostHeldBytes: 0,
        });
        // The cache's run chose the same kernels as full residency (marking the indices as outputs
        // changed no fusion): only expert_linear's key gains the pooled form.
        const census = (session: Session) =>
          session
            .diagnostics()
            .lastRunPipelines?.map(({ key, dispatchCount }) => [
              key.replace(/:pool$/, ""),
              dispatchCount,
            ]);
        await resident.run({
          x: { dtype: "f32", shape: [ROWS, 64], data: waveValues(ROWS * 64, 29, 1.5) },
        });
        assertEquals(census(cached), census(resident));
        assertEquals(cached.diagnostics().lastRunFusions, resident.diagnostics().lastRunFusions);
        assert(
          cached
            .diagnostics()
            .lastRunPipelines?.some(
              ({ key }) => key.startsWith("expert_linear") && key.endsWith(":pool"),
            ),
          "the cache did not run the pooled kernel",
        );
      });
    });
  }

  // One slot count per layer: 8 in layer 0, 12 in layer 1. Layer 1 holds more experts than layer
  // 0's count (its slots 8 to 11 are used — a pool or a kernel sized by layer 0's count would fail
  // the copies or write NaN there), layer 0 evicts within its own 8, and every exact run is the
  // resident bytes.
  const LAYER_SLOTS = [8, 12];
  for (const kind of KINDS) {
    it(`gives each layer its own slot count, bit-identical to full residency (${kind})`, async () => {
      const container = await containerOf(kind, expertGraph(SHAPE), tensors);
      await withSessions(
        container,
        async (resident, cached, cache) => {
          assertEquals(cache.slots, LAYER_SLOTS);
          assertEquals(
            cached.diagnostics().storage.residentCompressedBytes,
            (8 + 12) * EXPERT_BYTES,
          );
          /** Per layer, every expert an exact run used (each was loaded into the layer's pool). */
          const used = LAYER_SLOTS.map(() => new Set<number>());
          let evicted = 0;
          for (const seed of [13, 29, 47, 61]) {
            const inputs: RunInputs = {
              x: { dtype: "f32", shape: [ROWS, 64], data: waveValues(ROWS * 64, seed, 1.5) },
            };
            const expected = await resident.run(inputs);
            const converged = await converge(cache, () => cache.run(inputs));
            const last = converged.runs.at(-1);
            assert(last !== undefined && last.missing === 0);
            assertEquals(bits(last.outputs["out"]), bits(expected["out"]), `seed ${seed}`);
            cache.touch(last.routing, ROWS);
            for (const load of converged.loads) evicted += load.evicted;
            const span = ROWS * SHAPE.picks;
            used.forEach((experts, layer) => {
              for (const id of last.routing.ids.subarray(layer * span, (layer + 1) * span)) {
                experts.add(id);
              }
            });
          }
          // MUST (precondition): layer 0 used more experts than its 8 slots (it evicted), and layer
          // 1 more than layer 0's count (its free slots fill in ascending order, so it used slots
          // at 8 and above).
          assert(used[0].size > 8, `layer 0 used only ${used[0].size} experts`);
          assert(used[1].size > 8, `layer 1 used only ${used[1].size} experts`);
          assert(evicted > 0, "no expert was evicted");
        },
        LAYER_SLOTS,
      );
    });
  }

  // Attention over a full KV slot (one head of 64), then the two expert layers on its output.
  const states = fullStateGraph({ heads: 1, kvHeads: 1, depth: 64, slots: "full" });
  const layers = expertLayers(SHAPE, "ox", "out", "M");
  const stateful: DeclarationJson = {
    ...states,
    requires: { ops: [...new Set([...states.requires.ops, ...EXPERT_OPS])].sort() },
    outputs: ["out"],
    initializers: { ...states.initializers, ...layers.initializers },
    values: { ...states.values, ox: { dtype: "f32", shape: ["M", 64] }, ...layers.values },
    nodes: [
      ...states.nodes,
      { op: "reshape", ins: ["o"], outs: ["ox"], attrs: {} },
      ...layers.nodes,
    ],
  };
  for (const kind of KINDS) {
    it(`discards a missed deferred run with commit(0), loads, runs the position again and commits it (${kind})`, async () => {
      const container = await containerOf(kind, stateful, tensors);
      await withSessions(container, async (resident, cached, cache) => {
        const spec = { bindings: { C: 8 }, chunkLength: 1 };
        const plain = await resident.createGenerationContext(spec);
        const context = await cached.createGenerationContext(spec);
        try {
          for (let position = 0; position < 3; position += 1) {
            const row = (seed: number) => ({
              dtype: "f32" as const,
              shape: [1, 1, 1, 64],
              data: waveValues(64, 41 * position + seed, 0.8),
            });
            const [q, k, v] = [row(1), row(2), row(3)];
            const inputs: RunInputs = { q, k, v, append_k: k, append_v: v };
            const expected = await resident.run(inputs, {}, { context: plain, queryLength: 1 });
            const generation: GenerationRun = { context, queryLength: 1, commit: "deferred" };
            const converged = await converge(
              cache,
              () => cache.run(inputs, {}, generation),
              // A missed row is never committed: the same position runs again.
              () => context.commit(0),
            );
            const last = converged.runs.at(-1);
            assert(last !== undefined);
            assertEquals(context.pastLength, position, "a missed run moved the logical length");
            context.commit(1);
            cache.touch(last.routing, 1);
            if (position === 0) {
              // MUST (precondition) and fault injection: the first position misses, and its
              // missed run is not the resident one.
              assert(converged.runs[0].missing > 0, "the first position missed nothing");
              assertNotEquals(bits(converged.runs[0].outputs["out"]), bits(expected["out"]));
            }
            assertEquals(bits(last.outputs["out"]), bits(expected["out"]), `position ${position}`);
            assertEquals(last.routing.rows, 1);
          }
          assertEquals([context.pastLength, plain.pastLength], [3, 3]);
        } finally {
          await context.dispose();
          await plain.dispose();
        }
      });
    });
  }

  // Self-speculation's pieces (ADR 0146 decision 8) on the states-form graph with 32 experts per
  // layer and 12 slots, so that a run of three rows' picks fits and a row of other inputs still
  // misses: two drafts as chained deferred runs on the cache as it is, dropped by one commit(0);
  // row 0 alone until it misses nothing; a three-row verify at the same position whose row 0 hits
  // and whose later rows miss; then only the rows before its first missing row are committed, and the
  // rest of the rows run again after a load. Every committed row is the resident Session's, run
  // through the same shapes (the verify's three rows, then the rest) on a context of its own.
  const VERIFY_SLOTS = 12;
  const VERIFY_ROWS = 3;
  const VERIFY_SHAPE: ExpertGraphShape = { ...SHAPE, experts: 32 };
  const verifyLayers = expertLayers(VERIFY_SHAPE, "ox", "out", "M");
  const verifyGraph: DeclarationJson = {
    ...stateful,
    initializers: { ...states.initializers, ...verifyLayers.initializers },
    values: { ...states.values, ox: { dtype: "f32", shape: ["M", 64] }, ...verifyLayers.values },
    nodes: [
      ...states.nodes,
      { op: "reshape", ins: ["o"], outs: ["ox"], attrs: {} },
      ...verifyLayers.nodes,
    ],
  };
  const verifyTensors = expertTensors(VERIFY_SHAPE, "i4f16");
  /** The inputs of `rows` consecutive rows from row `first` (each row's q, k and v of its own). */
  const rowsOf = (first: number, rows: number): RunInputs => {
    const values = (seed: number) => {
      const data = new Float32Array(rows * 64);
      for (let row = 0; row < rows; row += 1) {
        data.set(waveValues(64, 53 * (first + row) + seed, 0.8), row * 64);
      }
      return { dtype: "f32" as const, shape: [1, 1, rows, 64], data };
    };
    const [q, k, v] = [values(1), values(2), values(3)];
    return { q, k, v, append_k: k, append_v: v };
  };
  /** Row `row` of an `[M, 64]` output, as u32 words. */
  const rowBits = (tensor: Tensor | undefined, row: number): number[] =>
    bits(tensor).slice(row * 64, (row + 1) * 64);
  for (const kind of KINDS) {
    it(`commits the rows before a verify's first missing row after chained drafts, and they are the resident rows (${kind})`, async () => {
      const container = await containerOf(kind, verifyGraph, verifyTensors);
      const gpu: GpuContext = await acquireGpu();
      try {
        const resident = await createSessionFromContainer(gpu, container, GRAPH_NAME);
        const cached = await createSessionFromContainer(gpu, container, GRAPH_NAME, {
          expertCache: { slots: VERIFY_SLOTS },
        });
        const spec = { bindings: { C: 8 }, chunkLength: VERIFY_ROWS, chunkBuckets: [2] };
        const plain = await resident.createGenerationContext(spec);
        const context = await cached.createGenerationContext(spec);
        try {
          const cache = cached.expertCache;
          assert(cache !== undefined, "no expert cache");
          // Two drafts, each one row: the second is placed after the first, both stay pending.
          const drafts: ExpertRun[] = [];
          for (const row of [0, 1]) {
            drafts.push(
              await cache.run(
                rowsOf(row, 1),
                {},
                {
                  context,
                  queryLength: 1,
                  commit: "deferred",
                  chain: true,
                },
              ),
            );
            assertEquals(context.pendingCommit, { pastLength: 0, queryLength: row + 1 });
          }
          for (const draft of drafts) {
            assertEquals(draft.routing.rows, 1);
            assertEquals(draft.firstMissingRow, draft.missing === 0 ? 1 : 0);
          }
          // MUST (precondition): the first draft ran on an empty cache, so it missed.
          assert(drafts[0].missing > 0, "the first draft missed nothing");
          context.commit(0);
          assertEquals([context.pastLength, context.pendingCommit], [0, undefined]);

          // Row 0 alone until it misses nothing (dropped each time): its experts, and only its, are
          // resident — 4 of 32 per layer, so rows of other inputs miss.
          const alone = { context, queryLength: 1, commit: "deferred" } as const;
          await converge(
            cache,
            () => cache.run(rowsOf(0, 1), {}, alone),
            () => context.commit(0),
          );
          context.commit(0);

          // The verify: three rows at position 0. Row 0 hits, so it is exact; rows 1 and 2 miss.
          const verifyInputs = rowsOf(0, VERIFY_ROWS);
          const verify = { context, queryLength: VERIFY_ROWS, commit: "deferred" } as const;
          const run = await cache.run(verifyInputs, {}, verify);
          const exact = run.firstMissingRow;
          // MUST (precondition): row 0 hit and a later row missed, so a part of the verify is dropped.
          assert(exact >= 1 && exact < VERIFY_ROWS, `the verify's first missing row is ${exact}`);
          const expected = await resident.run(verifyInputs, {}, { ...verify, context: plain });
          for (let row = 0; row < exact; row += 1) {
            assertEquals(
              rowBits(run.outputs["out"], row),
              rowBits(expected["out"], row),
              `row ${row}`,
            );
          }
          // Fault injection: the first missing row is not the resident one (committing it, or
          // reporting a later row as the first one, would commit a row that is not exact).
          assertNotEquals(rowBits(run.outputs["out"], exact), rowBits(expected["out"], exact));
          context.commit(exact);
          cache.touch(run.routing, exact);
          plain.commit(exact);
          await cache.load(run.routing);

          // The rest of the rows from the first missing one, until they miss nothing.
          const rest = VERIFY_ROWS - exact;
          const restInputs = rowsOf(exact, rest);
          const generation = { context, queryLength: rest, commit: "deferred" } as const;
          const converged = await converge(
            cache,
            () => cache.run(restInputs, {}, generation),
            () => context.commit(0),
          );
          const last = converged.runs.at(-1);
          assert(last !== undefined && last.missing === 0);
          assertEquals(last.firstMissingRow, rest);
          context.commit(rest);
          cache.touch(last.routing, rest);
          const reference = await resident.run(
            restInputs,
            {},
            { context: plain, queryLength: rest },
          );
          assertEquals(bits(last.outputs["out"]), bits(reference["out"]), "the rest of the rows");
          assertEquals([context.pastLength, plain.pastLength], [VERIFY_ROWS, VERIFY_ROWS]);
        } finally {
          await context.dispose();
          await plain.dispose();
          await cached.dispose();
          await resident.dispose();
        }
      } finally {
        gpu.destroy();
      }
    });
  }
});

// The host tier (`hostBytes`) and loads larger than the staging ring: three routed layers of 64
// experts (top-8) with 32 slots, four rows. The first run of an input misses in layer 0 (about 26
// distinct experts) and routes layers 1 and 2 from rows the misses zeroed (8 each), so its load
// needs more than the ring's 32 slots and runs in rounds. With a budget for
// every expert, construction holds them all and no load reads the container; with one of 20
// experts, construction holds the first 20 read, loads take what RAM holds and read (and keep) the
// rest. Either way every exact run is bit-identical to full residency.
const TIERED: ExpertGraphShape = { experts: 64, picks: 8, width: 64, ffn: 32, layers: 3 };
const TIERED_SLOTS = 32;
const TIERED_ROWS = 4;
const EVERY_EXPERT = 3 * 64 * EXPERT_BYTES;

describe.skipIf(!GPU_AVAILABLE)("the host tier and the staging ring's rounds (real GPU)", () => {
  const tensors = expertTensors(TIERED, "i4f16");
  for (const kind of KINDS) {
    for (const [budget, hostBytes] of [
      ["a budget for every expert", EVERY_EXPERT],
      ["a budget of 20 experts", 20 * EXPERT_BYTES],
    ] as const) {
      it(`runs bit-identical to full residency from RAM and the container, in rounds (${kind}, ${budget})`, async () => {
        const container = await containerOf(kind, expertGraph(TIERED), tensors);
        const gpu: GpuContext = await acquireGpu();
        try {
          const resident = await createSessionFromContainer(gpu, container, GRAPH_NAME);
          const cached = await createSessionFromContainer(gpu, container, GRAPH_NAME, {
            expertCache: { slots: TIERED_SLOTS, hostBytes },
          });
          try {
            const cache = cached.expertCache;
            assert(cache !== undefined, "no expert cache");
            const every = hostBytes === EVERY_EXPERT;
            const filled = cache.stats();
            // Construction filled the tier: every expert, or the first 20 read (layer 0's gate_up).
            assertEquals(
              [filled.hostHeldExperts, filled.hostHeldBytes],
              every ? [192, EVERY_EXPERT] : [20, 20 * EXPERT_BYTES],
            );
            let largest = 0;
            for (const seed of [13, 29, 47]) {
              const inputs: RunInputs = {
                x: {
                  dtype: "f32",
                  shape: [TIERED_ROWS, 64],
                  data: waveValues(TIERED_ROWS * 64, seed, 1.5),
                },
              };
              const expected = await resident.run(inputs);
              const converged = await converge(cache, () => cache.run(inputs));
              const last = converged.runs.at(-1);
              // MUST (precondition): the input misses, so the exact run follows loads.
              assert(converged.runs[0].missing > 0, `seed ${seed}: the first run missed nothing`);
              assert(last !== undefined && last.missing === 0);
              assertEquals(bits(last.outputs["out"]), bits(expected["out"]), `seed ${seed}`);
              cache.touch(last.routing, TIERED_ROWS);
              for (const load of converged.loads) largest = Math.max(largest, load.loaded);
            }
            // MUST (precondition): a load ran in more than one round of the ring.
            assert(largest > EXPERT_RING_SLOTS, `the largest load had only ${largest} experts`);
            const stats = cache.stats();
            assertEquals(stats.hostExperts + stats.containerExperts, stats.loadedExperts);
            assertEquals(stats.bytesUploaded, stats.loadedExperts * EXPERT_BYTES);
            assertEquals(stats.bytesRead, stats.containerExperts * EXPERT_BYTES);
            if (every) {
              assertEquals([stats.containerExperts, stats.hostHeldExperts], [0, 192]);
              assert(stats.hostExperts > 0);
            } else {
              // MUST (precondition): both sources served loads.
              assert(
                stats.hostExperts > 0 && stats.containerExperts > 0,
                `${stats.hostExperts} experts from RAM, ${stats.containerExperts} from the container`,
              );
              assertEquals(stats.hostHeldBytes, 20 * EXPERT_BYTES);
            }
          } finally {
            await cached.dispose();
            await resident.dispose();
          }
        } finally {
          gpu.destroy();
        }
      });
    }
  }
});
