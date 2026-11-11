// The candidate rules `candidate/vae_rms_silu_g8` and `_g4` (`@karume/candidate-rules`,
// packages/candidate-rules/src/vae-rms-silu.ts — the VAE's channel RMS norm and the SiLU after it
// in one dispatch, written against `@karume/runtime/kernel` alone) on the GPU, against their
// unfused plan: the class E evidence of docs/kernel-rules.md (d). Each graph runs on a Session given
// the rule through `SessionOptions.fusionRules` and on one built from the same options with both
// rules disabled by the test-only `DISABLED_FUSION_RULES` (the plain side, the same graph without
// the rule). The output is compared as u32 words (`wordIdentity`, NaN by class); the rule's counter
// in `lastRunFusions` reads 1 on the fused side and 0 on the plain side, the built-in silu's reads 0
// on the fused side (the rule took the SiLU) and 1 on the plain side when the chain ends with one,
// and the fused side ran the whole graph as the rule's one dispatch, so the two sides ran the plans
// they stand for.
//
// The graphs are small and synthetic (no model asset, `./helpers/norm-chain.ts`): both layouts the
// exported graphs have (Wan's `[C, T, H, W]` reduced on axis 0 with gamma `[C, 1, 1, 1]`, anima's
// `[1, C, H, W]` on axis 1 with gamma `[1, C, 1, 1]`, and the Wan mid-block's `[1, C, H, W]` with
// gamma `[C, 1, 1]`), channel counts below, at and above the reduce tree's 256 leaves and not a
// multiple of the group count (96, 100, 256, 257, 384, 1024), pixel counts below one tile of 32 / 64,
// not a multiple of either and spanning several workgroups, with and without the SiLU, each run
// with both rules. Every input pixel is generated, and the first pixels are adversarial: one all ±0
// (the norm is the floor), one with +Inf, one with −Inf, two with NaN payloads, one with ±2⁶⁴ (the
// squares overflow), one all subnormal, one partly subnormal, one where every channel is 1 + 2⁻¹²
// (each square lies halfway between two f32, so a square contracted into the leaf's add would
// differ from the stored one), one all at the largest finite value, one dominated by a single
// channel (the others divide to tiny values) and one of ties of both signs. gamma holds 0, −0, 10³⁰
// (m2 overflows, so the SiLU sees ±Inf), a subnormal and a negative value before its generated
// ones; scale is √C, as the exported graphs have it.
//
// Fault injection, two test rules built here from the candidate's own planned step (its WGSL, its
// params, its workgroups) and run on three graphs with C > 256, where a leaf of the reduce tree
// folds more than one square: `unbarriered` — the same text with every XOR rounding barrier removed
// (each value reads its operand directly) — and `mask` — the same text with the mask word 1, so each
// barrier flips the last bit of the value it passes. Both are compared with the plain side and each
// one's count of differing words is logged; at least one must be red (asserted), or the check could
// not see a broken kernel. On Vulkan / RTX 3080 Ti (Dawn, 2026-11-11) both turned red on all three
// graphs: `unbarriered` 46,227 of 195,840 words (Wan C 384), 2,880 of 119,808 (Wan C 1024) and
// 69,693 of 262,911 (anima C 257); `mask` 152,361, 81,173 and 137,871 of the same — so on this
// backend the barriers are what keeps the fused words equal to the plain ones. They live here, not
// in the candidate package, because a switch in a rule only for its own check would be code no run
// uses. No adapter = explicit SKIP.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  createSessionFromContainer,
  type GpuContext,
  type RunOutputs,
  type Session,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import { defineRule, type FusionMatch } from "@karume/runtime/kernel";
import { wordIdentity } from "@karume/runtime/tune";
import type { ExecStep } from "../../runtime/src/runtime/fusion-rule.ts";
import { DISABLED_FUSION_RULES } from "../../runtime/src/runtime/session-types.ts";
import type { TensorInput } from "../../runtime/tests/helpers/container-write.ts";
import { mergeGraph } from "../../runtime/tests/helpers/merged-graph.ts";
import { f32Bytes, GRAPH_NAME, openModelBytes } from "../../runtime/tests/helpers/model-fixture.ts";
import { type VaeRmsSiluGroups, vaeRmsSiluRule, vaeRmsSiluRuleName } from "@karume/candidate-rules";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { planDistributionFusions } from "./helpers/fusion-counts.ts";
import {
  type NormGraph,
  type NormLayout,
  type NormTail,
  normChain,
  normDeclaration,
} from "./helpers/norm-chain.ts";

type Case = { readonly name: string; readonly layout: NormLayout; readonly tail: NormTail };

/** Wan's decoder layout: `[C, T, H, W]` over axis 0, gamma `[C, 1, 1, 1]`. */
const wan = (shape: readonly [number, number, number, number], tail: NormTail): Case => ({
  name: `wan [${shape.join(", ")}] ${tail}`,
  layout: { x: shape, axis: 0, gamma: [shape[0], 1, 1, 1] },
  tail,
});

/** anima's layout: `[1, C, H, W]` over axis 1, gamma `[1, C, 1, 1]` (or the given gamma shape). */
const anima = (
  shape: readonly [number, number, number],
  tail: NormTail,
  gamma: readonly number[] = [1, shape[0], 1, 1],
): Case => ({
  name: `anima [1, ${shape.join(", ")}] gamma [${gamma.join(", ")}] ${tail}`,
  layout: { x: [1, ...shape], axis: 1, gamma },
  tail,
});

/** Pixel counts P in the comments; a tile is 32 pixels (g8) or 64 (g4). */
const CASES: readonly Case[] = [
  wan([96, 1, 4, 5], "silu"), // P 20: below one tile for both
  wan([100, 2, 7, 7], "none"), // P 98
  wan([256, 3, 23, 25], "silu"), // P 1,725: 54 / 27 tiles, the last partial
  wan([257, 1, 10, 10], "silu"), // P 100
  wan([384, 2, 15, 17], "silu"), // P 510
  wan([1024, 1, 9, 13], "silu"), // P 117
  anima([96, 40, 41], "silu"), // P 1,640
  anima([100, 3, 7], "silu"), // P 21
  anima([256, 5, 13], "none"), // P 65: one past a g4 tile
  anima([257, 33, 31], "none"), // P 1,023
  anima([384, 9, 9], "silu"), // P 81
  anima([1024, 16, 16], "none", [1024, 1, 1]), // P 256: the Wan mid-block's form
];

/** The graphs the fault rules run on: C > 256, so a leaf of the tree folds more than one square. */
const FAULT_CASES: readonly Case[] = [
  wan([384, 2, 15, 17], "silu"),
  wan([1024, 1, 9, 13], "silu"),
  anima([257, 33, 31], "none"),
];

const GROUPS: readonly VaeRmsSiluGroups[] = [8, 4];
const RULES = { 8: vaeRmsSiluRule(8), 4: vaeRmsSiluRule(4) } as const;
const NAMES = GROUPS.map((groups) => vaeRmsSiluRuleName(groups));

const elements = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

const channelsOf = (layout: NormLayout): number => layout.x[layout.axis];
const pixelsOf = (layout: NormLayout): number => elements(layout.x.slice(layout.axis + 1));

/** The f32 bits of a number. */
const bitsOf = (value: number): number => new Uint32Array(new Float32Array([value]).buffer)[0];

/** 1 + 2⁻¹²: its square is a tie between two f32. */
const TIE = 0x3f800800;

/**
 * The adversarial pixels, in order from pixel 0: the word of channel `c` of `C`, or `undefined` to
 * keep the generated value.
 */
const SPECIAL_PIXELS: readonly ((c: number, channels: number) => number | undefined)[] = [
  (c) => (c % 2 === 0 ? 0x00000000 : 0x80000000), // ±0 everywhere: the norm is the floor
  (c) => (c === 1 ? 0x7f800000 : undefined), // +Inf
  (c, channels) => (c === channels - 1 ? 0xff800000 : undefined), // −Inf
  (c) => (c === 0 ? 0x7fc01234 : undefined), // a quiet NaN with a payload
  (c, channels) => (c === channels >> 1 ? 0xff812345 : undefined), // a negative NaN with a payload
  (c) => (c === 2 ? 0x5f800000 : c === 3 ? 0xdf800000 : undefined), // ±2⁶⁴: the squares overflow
  (c) => [0x00000001, 0x807fffff, 0x00400000, 0x80000010][c % 4], // every channel subnormal
  (c) => (c % 3 === 0 ? 0x00012345 : undefined), // a third of the channels subnormal
  () => TIE, // every square a tie
  () => 0x7f7fffff, // the largest finite everywhere
  (c) => (c === 5 ? bitsOf(1e18) : undefined), // one channel dominates the norm
  (c) => (c % 2 === 0 ? TIE : TIE | 0x80000000), // ties of both signs
];

/** x: generated values with the adversarial pixels first (`x[c·P + t]`, any leading 1 axes). */
const xOf = (layout: NormLayout): Tensor => {
  const [channels, pixels] = [channelsOf(layout), pixelsOf(layout)];
  const data = Float32Array.from(
    { length: channels * pixels },
    (_, i) => Math.sin(i * 0.37 + 1) * 3.1 + (i % 7) / 13,
  );
  const words = new Uint32Array(data.buffer);
  SPECIAL_PIXELS.slice(0, pixels).forEach((word, t) => {
    for (let c = 0; c < channels; c += 1) {
      const special = word(c, channels);
      if (special !== undefined) words[c * pixels + t] = special;
    }
  });
  return { dtype: "f32", shape: layout.x, data };
};

/** gamma: 0, −0, 10³⁰, a subnormal and −2.5 first, then generated values of both signs. */
const gammaOf = (channels: number): Float32Array<ArrayBuffer> => {
  const values = Float32Array.from(
    { length: channels },
    (_, c) => (c % 5 === 4 ? -1 : 1) * (1 + 0.5 * Math.sin(c * 1.3)),
  );
  const words = new Uint32Array(values.buffer);
  words.set([0x00000000, 0x80000000, bitsOf(1e30), 0x0001b4f6, bitsOf(-2.5)].slice(0, channels));
  return values;
};

const weightsOf = (layout: NormLayout): readonly TensorInput[] => {
  const channels = channelsOf(layout);
  const tensor = (initializer: string, values: ArrayLike<number>): TensorInput => ({
    graph: GRAPH_NAME,
    initializer,
    bytes: f32Bytes(values),
    encoding: { codec: "f32" },
  });
  return [tensor("scale", [Math.sqrt(channels)]), tensor("gamma", gammaOf(channels))];
};

/** What one run of a side did. */
type Side = {
  readonly outputs: RunOutputs;
  readonly hits: Readonly<Record<string, number | undefined>>;
  readonly pipelines: readonly { readonly key: string; readonly dispatchCount: number }[];
};

/** One run of a Session opened on `graph` with `options`. */
const runSide = async (
  gpu: GpuContext,
  graph: NormGraph,
  layout: NormLayout,
  options: SessionOptions,
): Promise<Side> => {
  const session: Session = await createSessionFromContainer(
    gpu,
    await openModelBytes(normDeclaration(graph), weightsOf(layout)),
    GRAPH_NAME,
    options,
  );
  try {
    const outputs = await session.run({ x: xOf(layout) });
    const diagnostics = session.diagnostics();
    return {
      outputs,
      hits: diagnostics.lastRunFusions ?? {},
      pipelines: diagnostics.lastRunPipelines ?? [],
    };
  } finally {
    await session.dispose();
  }
};

/** The words of the graph's only output. */
const wordsOf = (side: Side, where: string): Uint32Array<ArrayBuffer> => {
  const tensors = Object.values(side.outputs);
  assertEquals(tensors.length, 1, `${where}: one output`);
  const { data } = tensors[0];
  return new Uint32Array(data.buffer, data.byteOffset, data.length);
};

/** The plain side: both rules given and both disabled (the same graph without the rule). */
const PLAIN: SessionOptions = {
  fusionRules: [RULES[8], RULES[4]],
  [DISABLED_FUSION_RULES]: NAMES,
};

// ---- The faults: the candidate's own step with its barriers removed, or with a non-zero mask ----

type FusedStepOf = Extract<ExecStep, { readonly kind: "fused" }>;

type Fault = "unbarriered" | "mask";

const CANDIDATE_KEY_PREFIX = "ext:candidate:vae_rms_silu:";

/** The candidate's planned step on `graph` (the CPU plan a Session given the rule runs). */
const plannedStep = (graph: NormGraph, groups: VaeRmsSiluGroups): FusedStepOf => {
  const { plan } = planDistributionFusions(mergeGraph(normDeclaration(graph)), graph.inputs, {
    fusionRules: [RULES[groups]],
  });
  const steps = plan.steps.flatMap((step) => (step.kind === "fused" ? [step] : []));
  assertEquals(steps.length, 1, "the candidate's planned steps");
  return steps[0];
};

/** The candidate's text with every `bitcast<f32>(bitcast<u32>(v) ^ m)` read as `((v))`. */
const unbarriered = (text: string): string => {
  const stripped = text.replaceAll("bitcast<f32>(bitcast<u32>(", "((").replaceAll(") ^ m)", "))");
  assert(stripped !== text && !stripped.includes("^ m"), "the barriers were not all removed");
  return stripped;
};

/** A rule that takes the chain starting at `sq` and runs `step` with `fault` applied. */
const faultRule = (fault: Fault, step: FusedStepOf) => {
  const [dispatch] = step.dispatches;
  const name = `test/vae_rms_silu_${fault}` as const;
  return defineRule<FusionMatch, typeof name>({
    name,
    heads: ["mul"],
    contract: "E",
    internals: "stayInChain",
    match: (nodes, index) => {
      if (nodes[index]?.outputs[0]?.name !== "sq") return undefined;
      const chain = nodes.slice(index, index + step.nodeCount);
      return { window: chain, chain };
    },
    build: () => ({
      binds: step.binds,
      outputName: step.outputName,
      outputShape: step.outputShape,
      temps: [],
      dispatches: [
        {
          key: `ext:test:vae_rms_silu_${fault}:${dispatch.key.slice(CANDIDATE_KEY_PREFIX.length)}`,
          wgsl: () => (fault === "unbarriered" ? unbarriered(dispatch.wgsl()) : dispatch.wgsl()),
          params:
            fault === "mask"
              ? Uint32Array.from([...dispatch.params.slice(0, 3), 1])
              : Uint32Array.from(dispatch.params),
          workgroups: dispatch.workgroups,
        },
      ],
    }),
  });
};

describe.skipIf(!GPU_AVAILABLE)(
  "the candidate rules candidate/vae_rms_silu_g8 / _g4 against their unfused plan (real GPU)",
  () => {
    it("are u32-identical to the plain nodes on every layout, channel count, pixel count and tail, run each graph as one dispatch and count once", async () => {
      const gpu = await acquireGpu();
      try {
        for (const testCase of CASES) {
          const graph = normChain(testCase.layout, testCase.tail);
          const plain = await runSide(gpu, graph, testCase.layout, PLAIN);
          assertEquals(
            [...NAMES.map((name) => plain.hits[name]), plain.hits.silu],
            [0, 0, testCase.tail === "silu" ? 1 : 0],
            `${testCase.name}: the plain side's counters (g8, g4, silu)`,
          );
          for (const groups of GROUPS) {
            const where = `${testCase.name} g${groups}`;
            const name = vaeRmsSiluRuleName(groups);
            const fused = await runSide(gpu, graph, testCase.layout, {
              fusionRules: [RULES[groups]],
            });
            assertEquals(
              [fused.hits[name], fused.hits.silu],
              [1, 0],
              `${where}: the rule's counter and silu's`,
            );
            assertEquals(
              fused.pipelines.map(({ key, dispatchCount }) => [key, dispatchCount]),
              [[`ext:candidate:vae_rms_silu:v1:g${groups}:${testCase.tail}:f32:wg256:xor`, 1]],
              `${where}: the fused side's one dispatch`,
            );
            assertEquals(
              wordIdentity(wordsOf(fused, where), wordsOf(plain, where), "class"),
              { words: elements(testCase.layout.x), failures: 0 },
              `${where}: the fused words against the plain words`,
            );
          }
        }
      } finally {
        gpu.destroy();
      }
    });

    it("fault injection: the candidate's step without its barriers, or with a non-zero mask, differs from the plain nodes", async () => {
      const gpu = await acquireGpu();
      try {
        let failures = 0;
        for (const testCase of FAULT_CASES) {
          const graph = normChain(testCase.layout, testCase.tail);
          const plain = await runSide(gpu, graph, testCase.layout, PLAIN);
          for (const fault of ["unbarriered", "mask"] as const) {
            const where = `${testCase.name} ${fault}`;
            const rule = faultRule(fault, plannedStep(graph, 8));
            const faulty = await runSide(gpu, graph, testCase.layout, { fusionRules: [rule] });
            assertEquals(faulty.hits[rule.name], 1, `${where}: the fault's counter`);
            const identity = wordIdentity(wordsOf(faulty, where), wordsOf(plain, where), "class");
            console.log(
              `[vae_rms_silu] fault ${where}: ${identity.failures} of ${identity.words} words differ`,
            );
            failures += identity.failures;
          }
        }
        assert(failures > 0, "neither fault changed a word: the check cannot fail");
      } finally {
        gpu.destroy();
      }
    });
  },
);
