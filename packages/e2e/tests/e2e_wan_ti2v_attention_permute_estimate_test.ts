/**
 * The transient backing of the Wan2.2 TI2V-5B DiT with and without the attention's absorbed permutes
 * (perf-ledger F-3 A4 / F-8 — no GPU: `estimateGraphMemory` on the distribution graph's part 0).
 *
 * The absorption must never make the backing larger. On the a8 attention seats (online and s16) the
 * region total of the graph as distributed — every attention absorbs the permutes of its q / k / v
 * and, on the online seat, the permute of O after it — is at most that of the same graph with a
 * zero-dispatch reshape between each of those permutes and its attention (the match broken, the
 * permutes back as nodes; the values and the kernels stay the graph's). The region total is what the
 * slot backing allocates (the estimate equals the slot table —
 * packages/runtime/tests/gpu_estimate_test.ts) and comes from the placement of `planTransients`
 * (ADR 0093 addendum 2026-10-26), which other live intervals can move either way while the live
 * peak stays put — under the first-fit placement it made the output permute's absorption grow the
 * backing (ADR 0040 addenda of 2026-10-25 and 2026-10-26). On the online seat the output permute's
 * absorption is also checked on its own: the region total with every permute absorbed is at most that
 * of the graph with q / k / v absorbed and only O's permute kept as a node.
 *
 * MUST: the asset is `models/karume-wan2.2/` (untracked local asset); where it is missing the case
 * SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import type { IrGraph, IrNode } from "../../runtime/src/format/ir.ts";
import { type EstimateOptions, estimateGraphMemory } from "../../runtime/src/runtime/estimate.ts";
import { planWeightResidency } from "../../runtime/src/runtime/weight-residency.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { exists, fusionCounts, readDistributionGraph } from "./helpers/fusion-counts.ts";

const WAN22_DIR = new URL("../../../models/karume-wan2.2/", import.meta.url);
const WAN22_AVAILABLE = await exists(new URL("karume.json", WAN22_DIR));
if (!WAN22_AVAILABLE) {
  console.warn(
    `[karume] SKIP the Wan2.2 estimate with and without the absorbed permutes: ${WAN22_DIR.pathname} is missing`,
  );
}

/** The DiT graph of the default model (`ti2v-5b`) on the i8 seat the attn8 presets use. */
const readDit = async (): Promise<IrGraph> => {
  const manifest: ContainerManifest = JSON.parse(
    await readTextFile(new URL("karume.json", WAN22_DIR)),
  );
  return await readDistributionGraph(
    WAN22_DIR,
    manifest,
    manifest.defaultModel,
    "transformer",
    "i8",
    new Map(),
  );
};

/**
 * The same graph with a zero-dispatch reshape between each `[0, 2, 1, 3]` permute and the fused
 * attention whose q / k / v it feeds (`inputs`), and between the attention and a permute that reads
 * its output (always): the exact match no longer holds, so each such permute stays a node, as when
 * it is not absorbed.
 */
const withPermutesAsNodes = (graph: IrGraph, inputs = true): IrGraph => {
  const producers = new Map(
    graph.nodes.flatMap((node) => node.outs.map((out) => [out, node] as const)),
  );
  const consumers = new Map<string, IrNode[]>();
  for (const node of graph.nodes) {
    for (const name of node.ins) consumers.set(name, [...(consumers.get(name) ?? []), node]);
  }
  const values: Record<string, IrGraph["values"][string]> = { ...graph.values };
  const nodes: IrNode[] = [];
  for (const node of graph.nodes) {
    if (node.op !== "attention" || Object.keys(node.states).length > 0) {
      nodes.push(node);
      continue;
    }
    const ins = node.ins.map((name, slot) => {
      if (!inputs || slot > 2 || producers.get(name)?.op !== "permute") return name;
      const reshaped = `${name}__as_node`;
      values[reshaped] = graph.values[name];
      nodes.push({ op: "reshape", ins: [name], outs: [reshaped], attrs: {}, states: {} });
      return reshaped;
    });
    // The output keeps its name for its consumers: the attention writes a new value that the
    // reshape turns back into it.
    const [output] = node.outs;
    if (!(consumers.get(output) ?? []).some((consumer) => consumer.op === "permute")) {
      nodes.push({ ...node, ins });
      continue;
    }
    const written = `${output}__as_node`;
    values[written] = graph.values[output];
    nodes.push({ ...node, ins, outs: [written] });
    nodes.push({ op: "reshape", ins: [written], outs: [output], attrs: {}, states: {} });
  }
  return {
    ...graph,
    requires: { ops: [...new Set([...graph.requires.ops, "reshape"])] },
    values,
    nodes,
  };
};

/**
 * The attn8 presets: linearCompute and attentionCompute "a8", S never stored or stored in f16, with
 * the permutes the distributed graph absorbs on each (q / k / v of the 60 attentions, and O too on
 * the online route).
 */
const SEATS: readonly (readonly [
  string,
  EstimateOptions & { attentionScoreStorage: "none" | "f16" },
  number,
])[] = [
  ["online", { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" }, 240],
  ["s16", { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" }, 180],
];

/**
 * Desktop-class device limits (2 GiB − 4 B per binding, 4 GiB per buffer, subgroups and 48 KiB of
 * workgroup storage, which hold the online kernel's sg32 form at D = 128).
 */
const DEVICE = {
  maxStorageBufferBindingSize: 2 ** 31 - 4,
  maxBufferSize: 2 ** 32,
  attentionDevice: { subgroups: true, maxComputeWorkgroupStorageSize: 48 * 1024 },
} as const;

const MiB = (bytes: number): string => (bytes / 2 ** 20).toFixed(1);

test.skipIf(!WAN22_AVAILABLE)(
  "the Wan2.2 DiT's transient backing with the absorbed permutes is no larger than with the permutes as nodes, on the online and s16 seats at 33 and 121 frames (S 7,920 / 27,280), and on the online seat no larger than with only the output permute as a node",
  async () => {
    const graph = await readDit();
    const reference = withPermutesAsNodes(graph);
    // q / k / v absorbed, O's permute a node: the online seat's absorption of O on its own.
    const outputAsNode = withPermutesAsNodes(graph, false);
    const residency = planWeightResidency(graph);
    for (const sequence of [7920, 27280]) {
      const shapes = (ir: IrGraph): Readonly<Record<string, readonly number[]>> =>
        Object.fromEntries(
          ir.inputs.map((spec) => [
            spec.name,
            spec.shape.map((dim) => (typeof dim === "number" ? dim : sequence)),
          ]),
        );
      for (const [seat, options, permutes] of SEATS) {
        // Preconditions: the distributed graph absorbs the seat's permutes, the reference nothing.
        const absorbed = (ir: IrGraph): number =>
          fusionCounts(
            ir,
            shapes(ir),
            undefined,
            false,
            false,
            false,
            "a8",
            "a8",
            options.attentionScoreStorage,
            DEVICE.attentionDevice,
          ).attentionPermute;
        assertEquals(
          absorbed(graph),
          permutes,
          `${seat} S=${sequence}: absorbed in the distributed graph`,
        );
        assertEquals(absorbed(reference), 0, `${seat} S=${sequence}: absorbed in the reference`);
        const workspace = (ir: IrGraph): number =>
          estimateGraphMemory(ir, residency, { bindings: { S: sequence }, ...DEVICE, ...options })
            .scenarios[0].workspaceBytes;
        const withAbsorption = workspace(graph);
        const withoutAbsorption = workspace(reference);
        console.log(
          `[attention permutes] Wan2.2 DiT ${seat} S=${sequence}: region total ${MiB(withoutAbsorption)} → ${MiB(withAbsorption)} MiB`,
        );
        assert(
          withAbsorption <= withoutAbsorption,
          `${seat} S=${sequence}: ${MiB(withAbsorption)} MiB with the absorption > ${MiB(withoutAbsorption)} MiB without`,
        );
        if (options.attentionScoreStorage !== "none") continue;
        // Precondition: the reference absorbs q / k / v of the 60 attentions and not O.
        assertEquals(
          absorbed(outputAsNode),
          180,
          `${seat} S=${sequence}: absorbed with O as a node`,
        );
        const withOutputAsNode = workspace(outputAsNode);
        console.log(
          `[attention permutes] Wan2.2 DiT ${seat} S=${sequence}: region total with O as a node ${MiB(withOutputAsNode)} → ${MiB(withAbsorption)} MiB`,
        );
        assert(
          withAbsorption <= withOutputAsNode,
          `${seat} S=${sequence}: ${MiB(withAbsorption)} MiB with O absorbed > ${MiB(withOutputAsNode)} MiB with O as a node`,
        );
      }
    }
  },
);
