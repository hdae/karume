/**
 * The session A/B of `@karume/runtime/tune` on one MOSS-Image DiT forward (opt-in:
 * `KARUME_SESSION_AB=<dir>`): the `dit` component (the positive branch) from the distribution
 * mirror with the practical preset's knobs (`f32+dit8-a8-attn8-online`), two arms built with
 * `sessionArm` — `base` (the preset's knobs) and `rule` (the same plus the candidate rule
 * `candidate/ew_chain` of `@karume/candidate-rules`). A run is one forward at the sequence of a
 * 1024² request (S = 4,096 image tokens + 63 text tokens = 4,159) on fixed seeded feeds, and its
 * output is the DiT's output rows, so the A/B's identity is u32 identity of the output (NaN by
 * class) of every run of both arms against the base's first. Two DiT Sessions never live together:
 * every visit opens and closes its Session (`openPerVisit`), and a close awaits the dispose, which
 * resolves once the device has the memory back.
 *
 * It asserts that identity, the rule's hits (292 per forward — the count of
 * e2e_moss_fusion_counts_test.ts), that the built-in silu keeps its one hit in both arms, and that
 * the rule arm's dispatches are the base's less the nodes folded; never a verdict on times: the
 * verdict and its numbers go to the record and the log. The counts come from the runner's record
 * (`sessionArm` reports each run's fusion counts and dispatches, outside the timed span). The
 * record (`karume-session-ab/1`, inside an envelope with the request and the container's part-0
 * digest) is written under the directory the variable names.
 *
 * NOTE: the feeds (between the twin markers) are the same text as the playground's
 * apps/playground/src/models/session-ab/moss.ts, `export` aside: the playground's MOSS-Image DiT
 * target of `/session-ab` runs the same forward on the same feeds in a browser. No module both may
 * import fits the gates (the e2e tests cannot import the playground, `@karume/candidate-rules`
 * holds rules only), so the two copies are kept textually parallel: a change to one MUST be made to
 * the other (apps/playground/tests/session-ab_moss_test.ts compares them).
 *
 * The record's backend is inferred from the platform (linux → vulkan, darwin → metal, win32 →
 * d3d12: Dawn's backend there), and the envelope says so; WebGPU does not expose it.
 *
 * MUST: never run it while other work uses the GPU (the times would measure the contention); run it
 * under `flock /tmp/karume-gpu.lock`. Without the variable, without the mirror or without an
 * adapter, the case SKIPs with the reason (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { mkdir, writeTextFile } from "@karume/test-support/fs";
import { prepareContainer, type Tensor } from "@karume/runtime";
import { judgeSessionAb, runSessionAb, sessionArm } from "@karume/runtime/tune";
import { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "@karume/candidate-rules";
import {
  MOSS_PRACTICAL_PRESET,
  mossAcquireGpuOptions,
  mossPreset,
} from "../../models/src/moss/preset.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  MOSS_GET_MIRROR,
  MOSS_MIRROR,
  mossMirrorPresent,
  openMossMirrorContainer,
  part0Digest,
} from "../../models/tests/helpers/moss-dit.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";

const DIRECTORY = process.env["KARUME_SESSION_AB"];

const skipReason =
  DIRECTORY === undefined || DIRECTORY === ""
    ? "KARUME_SESSION_AB is not set (opt-in: KARUME_SESSION_AB=<dir> writes the record there)"
    : !mossMirrorPresent()
      ? `${MOSS_MIRROR.pathname} has no manifest (get it with: ${MOSS_GET_MIRROR})`
      : !GPU_AVAILABLE
        ? "no GPU adapter"
        : undefined;
if (skipReason !== undefined) {
  console.warn(`[karume] the session A/B on the MOSS-Image DiT SKIPs: ${skipReason}`);
}

/**
 * The rule's runs in one forward (42 FFN gates, 84 RoPE tails, 84 gated-residual tails, 82 adaLN
 * tails — e2e_moss_fusion_counts_test.ts) and the nodes they fold beyond one dispatch each.
 */
const RULE_HITS = 292;
const FOLDED_DISPATCHES = 42 * 8 + 84 + 84 + 82;

/** Dawn's backend on each platform Node runs it on (the record says the backend is inferred). */
const DAWN_BACKENDS: Readonly<Record<string, string>> = {
  linux: "vulkan",
  darwin: "metal",
  win32: "d3d12",
};

const inferredBackend = (platform: string): string => {
  if (!Object.hasOwn(DAWN_BACKENDS, platform)) {
    throw new Error(`No Dawn backend is known for the platform '${platform}'`);
  }
  return DAWN_BACKENDS[platform];
};

// ---- The MOSS DiT feeds: twin begin (the same text in e2e_moss_session_ab_test.ts and the playground's session-ab/moss.ts, `export` aside) ----

/**
 * The work of one forward: the sequence of a 1024 × 1024 request — 4,096 image tokens (64 × 64
 * latent cells) and the 63 text tokens of the fridge-note prompt, S = 4,159 — and the seed of the
 * feeds.
 */
const MOSS_DIT_WORK = { imageTokens: 4096, textTokens: 63, seed: 0x5eed } as const;

/** A graph input as the DiT declares it (`S` its one symbol). */
type DitInput = {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly (number | string)[];
};

/** mulberry32: the same seed gives the same feeds on every machine and runtime. */
const seededUniform = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/**
 * Fixed feeds of one DiT forward at `imageTokens + textTokens` tokens: every f32 input uniform in
 * [−1, 1) from the seeded generator, input by input in the graph's order (the latent tokens, the
 * text context, the RoPE tables and the timestep projection stand in for a real request's — the
 * times and the identity do not depend on them being real), and the bool `is_image` 1 for the image
 * tokens, which come first, and 0 for the text tokens. Refuses an input of another dtype or a
 * symbol other than `S`, so a changed graph does not run on feeds made for another.
 */
const mossDitFeeds = (
  inputs: readonly DitInput[],
  imageTokens: number,
  textTokens: number,
  seed: number,
): Readonly<Record<string, Tensor>> => {
  const sequence = imageTokens + textTokens;
  const next = seededUniform(seed);
  const feeds: Record<string, Tensor> = {};
  for (const input of inputs) {
    const shape = input.shape.map((dim) => {
      if (typeof dim === "number") return dim;
      if (dim !== "S") throw new Error(`MOSS DiT feeds: input '${input.name}' has symbol '${dim}'`);
      return sequence;
    });
    const count = shape.reduce((total, dim) => total * dim, 1);
    if (input.dtype === "f32") {
      feeds[input.name] = {
        dtype: "f32",
        shape,
        data: Float32Array.from({ length: count }, () => next() * 2 - 1),
      };
    } else if (input.dtype === "bool" && input.name === "is_image" && count === sequence) {
      const data = new Uint32Array(count);
      data.fill(1, 0, imageTokens);
      feeds[input.name] = { dtype: "bool", shape, data };
    } else {
      throw new Error(
        `MOSS DiT feeds: no feed for input '${input.name}' (${input.dtype} ${JSON.stringify(input.shape)})`,
      );
    }
  }
  return feeds;
};

// ---- The MOSS DiT feeds: twin end ----

/** The DiT component the A/B runs (its graph has the same name). */
const COMPONENT = "dit";

/** The request the A/B runs (written into the record's envelope). */
const REQUEST = {
  model: "sft",
  component: COMPONENT,
  quant: MOSS_PRACTICAL_PRESET,
  ...MOSS_DIT_WORK,
  rounds: 4,
  openPerVisit: true,
  minGain: 0.01,
} as const;

describe.skipIf(skipReason !== undefined)("the session A/B on the MOSS-Image DiT (opt-in)", () => {
  it("the candidate rule keeps every output word and takes its 292 runs per forward", async () => {
    const preset = mossPreset(REQUEST.quant);
    const gpu = await acquireTestGpu(mossAcquireGpuOptions(preset));
    try {
      const opened = await openMossMirrorContainer(COMPONENT);
      const prepared = prepareContainer(opened, COMPONENT);
      const feeds = mossDitFeeds(
        prepared.graph.inputs,
        REQUEST.imageTokens,
        REQUEST.textTokens,
        REQUEST.seed,
      );
      const arms = [
        { name: "base", options: { ...preset.dit } },
        { name: "rule", options: { ...preset.dit, fusionRules: [EW_CHAIN_RULE] } },
      ].map(({ name, options }) => sessionArm({ name, prepared, gpu, options, feeds }));
      const backend = inferredBackend(process.platform);
      const record = await runSessionAb({
        arms,
        rounds: REQUEST.rounds,
        openPerVisit: REQUEST.openPerVisit,
        environment: {
          adapter: gpu.adapterInfo,
          userAgent: navigator.userAgent,
          backend,
          date: new Date().toISOString(),
        },
      });
      const judgements = judgeSessionAb(record, { minGain: REQUEST.minGain });
      const directory = DIRECTORY ?? "";
      await mkdir(directory, { recursive: true });
      const file = `${directory}/session-ab-moss_${record.environment.date.replaceAll(":", "-")}.json`;
      await writeTextFile(
        file,
        `${JSON.stringify(
          {
            format: "karume-e2e-moss-session-ab/1",
            request: REQUEST,
            container: { component: COMPONENT, part0Sha256: await part0Digest(opened) },
            backendSource: `inferred from process.platform '${process.platform}' (Dawn's backend there)`,
            record,
            judgements,
          },
          null,
          2,
        )}\n`,
      );
      console.log(`[session-ab] record: ${file}`);
      for (const arm of record.arms) {
        console.log(
          `[session-ab] ${arm.name}: hits ${arm.fusions?.[EW_CHAIN_RULE_NAME] ?? 0}, dispatches ${arm.dispatches}, times ${JSON.stringify(arm.timesMs)}`,
        );
      }
      for (const judgement of judgements) {
        console.log(`[session-ab] ${judgement.arm}: ${judgement.verdict} — ${judgement.reason}`);
      }

      const [base, rule] = record.arms;
      assertEquals(
        [base.identity.worst.failures, rule.identity.worst.failures],
        [0, 0],
        "every run's output words equal the base's first",
      );
      assertEquals(
        [base.fusions?.[EW_CHAIN_RULE_NAME] ?? 0, rule.fusions?.[EW_CHAIN_RULE_NAME]],
        [0, RULE_HITS],
        "the rule's hits per forward",
      );
      assertEquals([base.fusions?.silu, rule.fusions?.silu], [1, 1], "silu keeps its hit");
      assertEquals(
        (base.dispatches ?? 0) - (rule.dispatches ?? 0),
        FOLDED_DISPATCHES,
        "the rule arm's dispatches are the base's less the nodes folded",
      );
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  });
});
