/**
 * The session A/B of `@karume/runtime/tune` on the VAE graphs the candidate rules
 * `candidate/vae_rms_silu_g8` / `_g4` target (opt-in: `KARUME_SESSION_AB=<dir>`), two targets:
 *
 * - **Wan2.2 VAE next chunk**: one forward of the `vae_decoder_next` chunk graph (tile 16: the
 *   latent frame `[48, 1, 16, 16]` and its 32 caches in, four frames and the 32 updated caches out)
 *   from the Wan2.2 distribution mirror at its default preset, with the Session options the
 *   pipeline gives the VAE there (`wanVaeSessionOptions` over the preset's resolved `session` — the
 *   conv knob alone, `convCompute: "a8"` on `i8-a8-attn8-online-conv8`, over the int8 weights).
 * - **anima VAE decode**: one forward of the `vae_decoder` tile graph (latents `[1, 16, 64, 64]` →
 *   a 512² tile) from the anima mirror at its default preset, with the options the pipeline gives
 *   the VAE (`{}` — the VAE takes none of the quant's session, `anima/pipeline.ts`). The graph's
 *   input is the tile's fixed shape, so a 1024² image is not one forward: its decode runs this
 *   forward 9 times (3 tiles per axis), each the same work as the one timed here.
 *
 * Three arms over one prepared model per target, built with `sessionArm`: `base` (the pipeline's
 * options), `g8` and `g4` (the same plus the rule of that geometry). A run is one forward on fixed
 * seeded feeds — every input, the caches included, uniform in [−1, 1) — and reads back every
 * output, so the A/B's identity is u32 identity (NaN by class) of every output of every run of every
 * arm against the base's first run. The timed span is `Session.run`, which writes the feeds and
 * reads every output back: on the Wan chunk graph that is about 0.9 GiB over the bus per run
 * (the caches both ways), the same in every arm. Every visit opens and closes its Session
 * (`openPerVisit`), so two VAE Sessions never live together.
 *
 * It asserts that identity, the rule's 30 hits per forward in its arm and none in the others (the
 * counts of the family fusion-count tests), that the built-in silu's 29 hits are the base's and the
 * rule arms' are 0, and that each rule arm's dispatches are the base's less the dispatches folded
 * (29 × 7 + 6 = 209: a site with the SiLU ran eight dispatches — the square, the sum, sqrt,
 * clamp_min, div, two muls and silu; the reshape is an alias — and the one without ran seven);
 * never a verdict on times: the verdicts and their numbers go to the record and the log. The counts
 * come from the runner's record (`sessionArm` reports each run's fusion counts and dispatches,
 * outside the timed span). Each target's record (`karume-session-ab/1`, inside an envelope with the
 * request and the container's part-0 digest) is written under the directory the variable names.
 *
 * NOTE: the feeds (between the twin markers) are the same text as the playground's
 * apps/playground/src/models/session-ab/vae.ts, `export` aside: the playground's two VAE targets of
 * `/session-ab` run the same forwards on the same feeds in a browser. No module both may import fits
 * the gates (the e2e tests cannot import the playground, `@karume/candidate-rules` holds rules
 * only), so the two copies are kept textually parallel: a change to one MUST be made to the other
 * (apps/playground/tests/session-ab_vae_test.ts compares them).
 *
 * The record's backend is inferred from the platform (linux → vulkan, darwin → metal, win32 →
 * d3d12: Dawn's backend there), and the envelope says so; WebGPU does not expose it. The rules MUST
 * NOT be passed on Metal (their XOR rounding barrier is measured not to hold there), so the case
 * refuses that backend.
 *
 * MUST: never run it while other work uses the GPU (the times would measure the contention); run it
 * under `flock /tmp/karume-gpu.lock`. Without the variable, without a target's mirror or without an
 * adapter, the case SKIPs with the reason (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { mkdir, writeTextFile } from "@karume/test-support/fs";
import {
  type GpuContext,
  type OpenedContainer,
  prepareContainer,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import { judgeSessionAb, runSessionAb, sessionArm } from "@karume/runtime/tune";
import { vaeRmsSiluRule, vaeRmsSiluRuleName } from "@karume/candidate-rules";
import { resolveSessionOptions } from "../../models/src/session/options.ts";
import { WAN_SESSION_POLICY } from "../../models/src/wan/family.ts";
import { wanVaeSessionOptions } from "../../models/src/wan/tile-decode.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  mirrorPresent,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { part0Digest } from "../../models/tests/helpers/moss-dit.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";

const DIRECTORY = process.env["KARUME_SESSION_AB"];

const WAN22_MIRROR = new URL("../../../models/karume-wan2.2/", import.meta.url);
const ANIMA_MIRROR = new URL("../../../models/karume-anima/", import.meta.url);

const suiteSkip =
  DIRECTORY === undefined || DIRECTORY === ""
    ? "KARUME_SESSION_AB is not set (opt-in: KARUME_SESSION_AB=<dir> writes the records there)"
    : !GPU_AVAILABLE
      ? "no GPU adapter"
      : undefined;
const mirrorSkip = (mirror: URL): string | undefined =>
  suiteSkip ??
  (mirrorPresent(mirror)
    ? undefined
    : `${mirror.pathname} has no readable manifest (assemble the distribution mirror)`);
const wanSkip = mirrorSkip(WAN22_MIRROR);
const animaSkip = mirrorSkip(ANIMA_MIRROR);
for (const [target, reason] of [
  ["Wan2.2 VAE next chunk", wanSkip],
  ["anima VAE decode", animaSkip],
] as const) {
  if (reason !== undefined)
    console.warn(`[karume] the session A/B on the ${target} SKIPs: ${reason}`);
}

/** The norms a forward holds (29 with the SiLU, 1 without) and the dispatches the rule folds. */
const RULE_HITS = 30;
const SILU_HITS = 29;
const FOLDED_DISPATCHES = 29 * 7 + 6;

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

// ---- The VAE feeds: twin begin (the same text in e2e_vae_session_ab_test.ts and the playground's session-ab/vae.ts, `export` aside) ----

/** The two targets: the graph a run forwards once, and the seed of its feeds. */
const VAE_AB_TARGETS = {
  "wan22-vae-next": { component: "vae_decoder_next", seed: 0x5eed },
  "anima-vae-decode": { component: "vae_decoder", seed: 0x5eed },
} as const;

/** A graph input as a VAE graph declares it (every shape static). */
type VaeInput = {
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
 * Fixed feeds of one VAE forward: every input uniform in [−1, 1) from the seeded generator, input by
 * input in the graph's order (the latent and, on a Wan chunk graph, every cache stand in for a
 * tile's — the times and the identity do not depend on them being real). Refuses an input that is
 * not f32 or has a symbolic dimension, so a changed graph does not run on feeds made for another.
 */
const vaeFeeds = (inputs: readonly VaeInput[], seed: number): Readonly<Record<string, Tensor>> => {
  const next = seededUniform(seed);
  const feeds: Record<string, Tensor> = {};
  for (const input of inputs) {
    const shape = input.shape.map((dim) => {
      if (typeof dim === "number") return dim;
      throw new Error(`VAE feeds: input '${input.name}' has symbol '${dim}'`);
    });
    if (input.dtype !== "f32") {
      throw new Error(`VAE feeds: no feed for input '${input.name}' (${input.dtype})`);
    }
    const count = shape.reduce((total, dim) => total * dim, 1);
    feeds[input.name] = {
      dtype: "f32",
      shape,
      data: Float32Array.from({ length: count }, () => next() * 2 - 1),
    };
  }
  return feeds;
};

// ---- The VAE feeds: twin end ----

type Target = keyof typeof VAE_AB_TARGETS;

/**
 * `component` at the distribution's default model and preset (the selection's defaults), with the
 * preset's declared `session`, the component's storage there and the opened container.
 */
const openDefault = async (mirror: URL, component: string) => {
  const chosen = await selectMirrorComponent(mirror, component);
  const { model, quant } = chosen.selection;
  const preset = chosen.loaded.manifest.models[model].quants[quant];
  return {
    model,
    quant,
    session: preset.session,
    storage: preset.weights[component],
    opened: await openSelectedContainer(chosen),
  };
};

const bytesOf = (tensors: Iterable<{ readonly shape: readonly (number | string)[] }>): number =>
  [...tensors].reduce(
    (total, { shape }) => total + 4 * shape.reduce<number>((count, dim) => count * Number(dim), 1),
    0,
  );

/** One target's A/B: open, run the three arms, write the record, assert identity and counts. */
const runTarget = async (
  target: Target,
  gpu: GpuContext,
  opened: OpenedContainer,
  request: Readonly<Record<string, unknown>>,
  options: SessionOptions,
): Promise<void> => {
  const { component, seed } = VAE_AB_TARGETS[target];
  const prepared = prepareContainer(opened, component);
  const feeds = vaeFeeds(prepared.graph.inputs, seed);
  const backend = inferredBackend(process.platform);
  if (backend === "metal") throw new Error("the rules' XOR barrier does not hold on Metal");
  const arms = [
    { name: "base", options },
    { name: "g8", options: { ...options, fusionRules: [vaeRmsSiluRule(8)] } },
    { name: "g4", options: { ...options, fusionRules: [vaeRmsSiluRule(4)] } },
  ].map((arm) => sessionArm({ ...arm, prepared, gpu, feeds }));
  const record = await runSessionAb({
    arms,
    rounds: 4,
    openPerVisit: true,
    environment: {
      adapter: gpu.adapterInfo,
      userAgent: navigator.userAgent,
      backend,
      date: new Date().toISOString(),
    },
  });
  const judgements = judgeSessionAb(record, { minGain: 0.01 });
  const directory = DIRECTORY ?? "";
  await mkdir(directory, { recursive: true });
  const file = `${directory}/session-ab-vae-${target}_${record.environment.date.replaceAll(":", "-")}.json`;
  await writeTextFile(
    file,
    `${JSON.stringify(
      {
        format: "karume-e2e-vae-session-ab/1",
        request: {
          target,
          component,
          seed,
          ...request,
          rounds: 4,
          openPerVisit: true,
          minGain: 0.01,
          arms: [
            { name: "base", fusionRules: [] },
            { name: "g8", fusionRules: [vaeRmsSiluRuleName(8)] },
            { name: "g4", fusionRules: [vaeRmsSiluRuleName(4)] },
          ],
          // What the timed span moves over the bus besides the forward (every feed in, every output out).
          feedBytes: bytesOf(prepared.graph.inputs),
          outputBytes: bytesOf(
            prepared.graph.outputs.map((name) => ({ shape: prepared.graph.values[name].shape })),
          ),
        },
        container: { component, part0Sha256: await part0Digest(opened) },
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
      `[session-ab] ${target} ${arm.name}: hits g8 ${arm.fusions?.[vaeRmsSiluRuleName(8)] ?? 0} / g4 ${arm.fusions?.[vaeRmsSiluRuleName(4)] ?? 0}, silu ${arm.fusions?.silu}, dispatches ${arm.dispatches}, times ${JSON.stringify(arm.timesMs)}`,
    );
  }
  for (const judgement of judgements) {
    console.log(
      `[session-ab] ${target} ${judgement.arm}: ${judgement.verdict} — ${judgement.reason}`,
    );
  }

  const [base, g8, g4] = record.arms;
  assertEquals(
    record.arms.map((arm) => arm.identity.worst.failures),
    [0, 0, 0],
    "every output word of every run equals the base's first",
  );
  assertEquals(
    record.arms.map((arm) => [
      arm.fusions?.[vaeRmsSiluRuleName(8)] ?? 0,
      arm.fusions?.[vaeRmsSiluRuleName(4)] ?? 0,
      arm.fusions?.silu,
    ]),
    [
      [0, 0, SILU_HITS],
      [RULE_HITS, 0, 0],
      [0, RULE_HITS, 0],
    ],
    "the rules' hits (g8, g4) and silu's per forward, per arm",
  );
  assertEquals(
    [g8, g4].map((arm) => (base.dispatches ?? 0) - (arm.dispatches ?? 0)),
    [FOLDED_DISPATCHES, FOLDED_DISPATCHES],
    "each rule arm's dispatches are the base's less the dispatches folded",
  );
};

describe.skipIf(suiteSkip !== undefined)(
  "the session A/B of candidate/vae_rms_silu_g8 / _g4 on the VAE graphs (opt-in)",
  () => {
    it.skipIf(wanSkip !== undefined)(
      "Wan2.2 VAE next chunk: both rules keep every output word and take the 30 norms of a forward",
      async () => {
        const target = "wan22-vae-next";
        const { component } = VAE_AB_TARGETS[target];
        const opened = await openDefault(WAN22_MIRROR, component);
        // The VAE's options as the pipeline derives them from the preset (tile-decode.ts).
        const options = wanVaeSessionOptions(
          resolveSessionOptions(
            WAN_SESSION_POLICY,
            opened.session,
            {},
            `Wan2.2 quant '${opened.quant}'`,
          ),
        );
        const gpu = await acquireTestGpu();
        try {
          await runTarget(
            target,
            gpu,
            opened.opened,
            {
              mirror: "karume-wan2.2",
              model: opened.model,
              quant: opened.quant,
              storage: opened.storage,
              sessionOptions: options,
            },
            options,
          );
        } finally {
          await settleReleases(gpu);
          gpu.destroy();
        }
      },
    );

    it.skipIf(animaSkip !== undefined)(
      "anima VAE decode: both rules keep every output word and take the 30 norms of a tile forward",
      async () => {
        const target = "anima-vae-decode";
        const { component } = VAE_AB_TARGETS[target];
        const opened = await openDefault(ANIMA_MIRROR, component);
        // The pipeline gives the VAE stage no option of the quant's session (anima/pipeline.ts).
        const options: SessionOptions = {};
        const gpu = await acquireTestGpu();
        try {
          await runTarget(
            target,
            gpu,
            opened.opened,
            {
              mirror: "karume-anima",
              model: opened.model,
              quant: opened.quant,
              storage: opened.storage,
              sessionOptions: options,
              tilesPer1024Image: 9,
            },
            options,
          );
        } finally {
          await settleReleases(gpu);
          gpu.destroy();
        }
      },
    );
  },
);
