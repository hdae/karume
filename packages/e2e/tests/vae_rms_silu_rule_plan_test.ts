/**
 * The candidate rule `candidate/vae_rms_silu_g8` / `_g4` (`@karume/candidate-rules`,
 * packages/candidate-rules/src/vae-rms-silu.ts — the VAE's channel RMS norm and the SiLU after it
 * in one dispatch) as the fusion pass plans it, on small synthetic graphs (no GPU, no model asset —
 * the core lane): planned through the path a Session takes (`planDistributionFusions` with
 * `fusionRules`, `./helpers/fusion-counts.ts`), so what is checked is the plan a Session given the
 * rule would run. It pins the chains the rule takes (the Wan layout `[C, T, H, W]` over axis 0, the
 * anima layout `[1, C, H, W]` over axis 1, the three gamma shapes the exported graphs have), the
 * 8-node form when no exact SiLU follows, one decline per pinned slot, that one form gives one key
 * and one WGSL text whatever the shapes, the params words and the workgroup declaration.
 *
 * Not checked: a non-f32 node in the window. No graph can hold one — `sqrt`, `div`, `clamp_min` and
 * `sigmoid` take f32 alone and `sum` f32 or bool (the op contracts), so the planner refuses such a
 * graph before any rule sees it, and the runtime declines a chain that is not all f32 by itself.
 *
 * What the rule computes is checked against the unfused plan on the GPU in
 * `e2e_vae_rms_silu_rule_test.ts`; its hits on the Wan2.1, Wan2.2 and anima VAE graphs are pinned
 * in those families' fusion-count tests.
 */

import { describe, it } from "vite-plus/test";
import { assertEquals, assertNotEquals } from "@std/assert";
import type { ExecStep } from "../../runtime/src/runtime/fusion-rule.ts";
import { mergeGraph } from "../../runtime/tests/helpers/merged-graph.ts";
import { vaeRmsSiluRule, vaeRmsSiluRuleName, type VaeRmsSiluGroups } from "@karume/candidate-rules";
import { type DistributionFusionPlan, planDistributionFusions } from "./helpers/fusion-counts.ts";
import {
  type NormGraph,
  type NormLayout,
  type NormNode,
  normChain,
  normDeclaration,
} from "./helpers/norm-chain.ts";

const G8 = vaeRmsSiluRuleName(8);

/** A Wan decoder block: `[C, T, H, W]` reduced over the channels on axis 0, gamma `[C, 1, 1, 1]`. */
const WAN: NormLayout = { x: [96, 2, 3, 5], axis: 0, gamma: [96, 1, 1, 1] };
/** An anima decoder block: `[1, C, H, W]` over axis 1, gamma `[1, C, 1, 1]`. */
const ANIMA: NormLayout = { x: [1, 384, 4, 6], axis: 1, gamma: [1, 384, 1, 1] };
/** The Wan decoder's mid-block norm (no SiLU after it): `[1, C, H, W]` over axis 1, gamma `[C, 1, 1]`. */
const MID: NormLayout = { x: [1, 1024, 2, 3], axis: 1, gamma: [1024, 1, 1] };

/** The f32 bits of the exported floor 1e-12. */
const FLOOR_BITS = 0x2b8cbccc;

/** The plan of `graph` with the rule of `groups` given, or (`"plain"`) without any rule. */
const planOf = (graph: NormGraph, groups: VaeRmsSiluGroups | "plain" = 8): DistributionFusionPlan =>
  planDistributionFusions(
    mergeGraph(normDeclaration(graph)),
    graph.inputs,
    groups === "plain" ? {} : { fusionRules: [vaeRmsSiluRule(groups)] },
  );

type FusedStepOf = Extract<ExecStep, { readonly kind: "fused" }>;

/** The rule's one step of a plan (fails when it has none or several). */
const onlyStep = (
  planned: DistributionFusionPlan,
  where: string,
  groups: VaeRmsSiluGroups = 8,
): FusedStepOf => {
  const name = vaeRmsSiluRuleName(groups);
  const steps = planned.plan.steps.flatMap((step) =>
    step.kind === "fused" && step.rule === name ? [step] : [],
  );
  assertEquals(steps.length, 1, `${where}: the rule's steps`);
  return steps[0];
};

/** The rule declines `graph`: no hit, the plan of the graph without the rule. */
const assertDeclined = (graph: NormGraph, where: string): void => {
  const fused = planOf(graph);
  const plain = planOf(graph, "plain");
  assertEquals(fused.plan.counts[G8], 0, `${where}: hits`);
  assertEquals(
    fused.plan.counts,
    { ...plain.plan.counts, [G8]: 0 },
    `${where}: the other counters`,
  );
  assertEquals(fused.plan.steps.length, plain.plan.steps.length, `${where}: steps`);
};

/** `graph` with the node that writes `out` replaced by `change(node)`. */
const changed = (
  graph: NormGraph,
  out: string,
  change: (node: NormNode) => NormNode,
): NormGraph => ({
  ...graph,
  nodes: graph.nodes.map((node) => (node.out === out ? change(node) : node)),
});

/** `graph` with `extra` nodes after its chain and the graph outputs `outputs`. */
const followed = (
  graph: NormGraph,
  extra: readonly NormNode[],
  outputs: readonly string[],
): NormGraph => ({ ...graph, nodes: [...graph.nodes, ...extra], outputs });

describe("candidate/vae_rms_silu: the chains it takes", () => {
  it("takes the Wan layout with the SiLU as one step of 10 nodes binding x, scale and gamma, and leaves silu at 0", () => {
    const planned = planOf(normChain(WAN, "silu"));
    const step = onlyStep(planned, "wan");
    assertEquals(
      [step.nodeCount, step.binds, step.outputName, step.outputShape],
      [10, ["x", "scale", "gamma"], "y", WAN.x],
    );
    assertEquals([planned.plan.counts[G8], planned.plan.counts.silu], [1, 0]);
    assertEquals(planned.plan.steps.length, 1, "the whole graph is the rule's step");
    // Without the rule the built-in silu takes the tail.
    assertEquals(planOf(normChain(WAN, "silu"), "plain").plan.counts.silu, 1);
  });

  it("takes the anima layout and the mid-block's rank-3 gamma, with and without the SiLU", () => {
    for (const [layout, tail, nodes] of [
      [ANIMA, "silu", 10],
      [ANIMA, "none", 8],
      [MID, "none", 8],
      [MID, "silu", 10],
    ] as const) {
      const where = `${JSON.stringify(layout.x)} gamma ${JSON.stringify(layout.gamma)} ${tail}`;
      const step = onlyStep(planOf(normChain(layout, tail)), where);
      assertEquals([step.nodeCount, step.binds], [nodes, ["x", "scale", "gamma"]], where);
    }
  });

  it("takes the 8-node form when no SiLU follows, when the SiLU's mul reads the sigmoid first, and when the mul after the sigmoid reads another value", () => {
    const silu = normChain(WAN, "silu");
    for (const [where, graph, silus] of [
      ["no tail", normChain(WAN, "none"), 0],
      ["mul(g, m2)", changed(silu, "y", (node) => ({ ...node, ins: ["g", "m2"] })), 1],
      [
        "mul(m2, z)",
        {
          ...changed(silu, "y", (node) => ({ ...node, ins: ["m2", "z"] })),
          inputs: { x: WAN.x, z: WAN.x },
          outputs: ["y", "g"],
        },
        0,
      ],
    ] as const) {
      const planned = planOf(graph);
      const step = onlyStep(planned, where);
      assertEquals([step.nodeCount, step.outputName], [8, "m2"], where);
      assertEquals(step.dispatches[0].key, "ext:candidate:vae_rms_silu:v1:g8:none:f32:wg256:xor");
      // The pair the rule leaves goes to the built-in silu when it is the exact form.
      assertEquals(planned.plan.counts.silu, silus, `${where}: silu`);
    }
  });
});

describe("candidate/vae_rms_silu: the chains it declines, one per pinned slot", () => {
  const silu = normChain(WAN, "silu");

  it("the div's operands swapped", () => {
    assertDeclined(
      changed(silu, "d", (node) => ({ ...node, ins: ["k", "x"] })),
      "div(k, x)",
    );
  });

  it("the scale first in its mul", () => {
    assertDeclined(
      changed(silu, "m1", (node) => ({ ...node, ins: ["scale", "d"] })),
      "mul(scale, d)",
    );
  });

  it("the gamma first in its mul", () => {
    assertDeclined(
      changed(silu, "m2", (node) => ({ ...node, ins: ["gamma", "m1"] })),
      "mul(gamma, m1)",
    );
  });

  it("a square of two different values", () => {
    assertDeclined(
      {
        ...changed(silu, "sq", (node) => ({ ...node, ins: ["x", "z"] })),
        inputs: { x: WAN.x, z: WAN.x },
      },
      "mul(x, z)",
    );
  });

  it("the div reading another value than the square's", () => {
    assertDeclined(
      {
        ...changed(silu, "d", (node) => ({ ...node, ins: ["z", "k"] })),
        inputs: { x: WAN.x, z: WAN.x },
      },
      "div(z, k)",
    );
  });

  it("a sum over another axis", () => {
    // [1, C, H, W] summed over H: the axes before it hold C.
    const x = [1, 8, 6, 5];
    assertDeclined(
      {
        inputs: { x },
        weights: { scale: [1], gamma: [1, 1, 6, 1] },
        nodes: [
          { op: "mul", ins: ["x", "x"], out: "sq", shape: x },
          { op: "sum", ins: ["sq"], out: "s", shape: [1, 8, 5], attrs: { dim: 2 } },
          { op: "sqrt", ins: ["s"], out: "r", shape: [1, 8, 5] },
          { op: "clamp_min", ins: ["r"], out: "n", shape: [1, 8, 5], attrs: { min: 1e-12 } },
          { op: "reshape", ins: ["n"], out: "k", shape: [1, 8, 1, 5] },
          { op: "div", ins: ["x", "k"], out: "d", shape: x },
          { op: "mul", ins: ["d", "scale"], out: "m1", shape: x },
          { op: "mul", ins: ["m1", "gamma"], out: "m2", shape: x },
        ],
      },
      "sum over H",
    );
  });

  it("a reshape to another shape than the kept-axis one", () => {
    // [T, H, W] kept at rank 3: the div still broadcasts it (right-aligned), but the shape is not
    // the keepdim form.
    assertDeclined(
      changed(silu, "k", (node) => ({ ...node, shape: [2, 3, 5] })),
      "reshape to [T, H, W]",
    );
  });

  it("a scale of more than one element", () => {
    assertDeclined(
      { ...silu, weights: { ...silu.weights, scale: [96, 1, 1, 1] } },
      "scale [C, 1, 1, 1]",
    );
  });

  it("a gamma that is not one element per channel on the channel axis", () => {
    assertDeclined({ ...silu, weights: { ...silu.weights, gamma: [1] } }, "gamma [1]");
    assertDeclined({ ...silu, weights: { ...silu.weights, gamma: [5] } }, "gamma [W]");
    assertDeclined(
      { ...silu, weights: { ...silu.weights, gamma: [1, 2, 1, 1] } },
      "gamma on the T axis",
    );
  });

  it("a leading axis longer than 1", () => {
    assertDeclined(
      normChain({ x: [2, 16, 3, 5], axis: 1, gamma: [1, 16, 1, 1] }, "silu"),
      "x [2, C, H, W]",
    );
  });

  it("an internal value read outside the chain", () => {
    for (const internal of ["sq", "s", "n", "d", "m1"]) {
      const shape = silu.nodes.find((node) => node.out === internal)?.shape ?? [];
      assertDeclined(
        followed(silu, [{ op: "neg", ins: [internal], out: "outside", shape }], ["y", "outside"]),
        `${internal} read by a node after the chain`,
      );
    }
  });

  it("an internal value that is a graph output", () => {
    for (const internal of ["sq", "r", "k", "m1"]) {
      assertDeclined({ ...silu, outputs: ["y", internal] }, `${internal} as a graph output`);
    }
    // Without the SiLU, m2 is the chain's output, which anyone may read.
    const none = normChain(WAN, "none");
    assertDeclined({ ...none, outputs: ["m2", "d"] }, "d as an output, no SiLU");
  });

  it("the whole window when m2, internal with the SiLU, has a third reader (the built-in silu takes the tail then)", () => {
    const graph = followed(
      silu,
      [{ op: "neg", ins: ["m2"], out: "outside", shape: WAN.x }],
      ["y", "outside"],
    );
    assertDeclined(graph, "m2 read after the chain");
    assertEquals(planOf(graph).plan.counts.silu, 1, "silu");
  });
});

describe("candidate/vae_rms_silu: keys, WGSL, params and workgroups", () => {
  it("gives one (groups, tail) one key and byte-identical WGSL whatever the layout and shapes, and every other pair another", () => {
    const dispatchOf = (layout: NormLayout, tail: "silu" | "none", groups: VaeRmsSiluGroups) =>
      onlyStep(planOf(normChain(layout, tail), groups), "key", groups).dispatches[0];
    const wan = dispatchOf(WAN, "silu", 8);
    const anima = dispatchOf(ANIMA, "silu", 8);
    assertEquals(wan.key, "ext:candidate:vae_rms_silu:v1:g8:silu:f32:wg256:xor");
    assertEquals(anima.key, wan.key);
    assertEquals(anima.wgsl(), wan.wgsl());
    assertNotEquals([...anima.params], [...wan.params]);
    const others = [
      dispatchOf(WAN, "none", 8),
      dispatchOf(WAN, "silu", 4),
      dispatchOf(WAN, "none", 4),
    ];
    assertEquals(
      others.map(({ key }) => key),
      [
        "ext:candidate:vae_rms_silu:v1:g8:none:f32:wg256:xor",
        "ext:candidate:vae_rms_silu:v1:g4:silu:f32:wg256:xor",
        "ext:candidate:vae_rms_silu:v1:g4:none:f32:wg256:xor",
      ],
    );
    const texts = new Set([wan, ...others].map((dispatch) => dispatch.wgsl()));
    assertEquals(texts.size, 4, "four pairs, four texts");
    // The same text from a second plan (no state between plans).
    assertEquals(dispatchOf(MID, "none", 4).wgsl(), others[2].wgsl());
  });

  it("writes the params P, C, the floor's bits and a zero mask, and the floor from the clamp's attribute", () => {
    const params = (graph: NormGraph) => [
      ...onlyStep(planOf(graph), "params").dispatches[0].params,
    ];
    // Wan [C, T, H, W]: P = T·H·W.
    assertEquals(params(normChain(WAN, "silu")), [2 * 3 * 5, 96, FLOOR_BITS, 0]);
    // anima [1, C, H, W]: P = H·W.
    assertEquals(params(normChain(ANIMA, "none")), [4 * 6, 384, FLOOR_BITS, 0]);
    // Another floor moves the params word and nothing else.
    const floored = changed(normChain(WAN, "silu"), "n", (node) => ({
      ...node,
      attrs: { min: 1e-6 },
    }));
    assertEquals(params(floored), [30, 96, 0x358637bd, 0]);
    const [base, other] = [normChain(WAN, "silu"), floored].map(
      (graph) => onlyStep(planOf(graph), "floor").dispatches[0],
    );
    assertEquals([other.key, other.wgsl()], [base.key, base.wgsl()]);
  });

  it("declares one grid-stride workgroup of 256 per tile of 32 (g8) or 64 (g4) pixels", () => {
    for (const [x, pixels] of [
      [[96, 2, 3, 5], 30],
      [[96, 1, 10, 10], 100],
      [[96, 2, 16, 16], 512],
    ] as const) {
      const layout: NormLayout = { x, axis: 0, gamma: [96, 1, 1, 1] };
      for (const [groups, perTile] of [
        [8, 32],
        [4, 64],
      ] as const) {
        const [dispatch] = onlyStep(
          planOf(normChain(layout, "silu"), groups),
          "wg",
          groups,
        ).dispatches;
        assertEquals(
          dispatch.workgroups,
          { kind: "gridStride", items: Math.ceil(pixels / perTile) * 256, size: 256 },
          `P ${pixels} g${groups}`,
        );
        const text = dispatch.wgsl();
        assertEquals(text.includes("@compute @workgroup_size(256)"), true, "workgroup size");
        assertEquals(text.includes(`const PIX: u32 = ${perTile}u;`), true, "pixels per tile");
        assertEquals(
          text.includes(`var<workgroup> nodes: array<f32, ${perTile * 33}>;`),
          true,
          "the stride-32 node rows",
        );
      }
    }
  });
});
