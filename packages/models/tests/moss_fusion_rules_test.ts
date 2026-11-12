// The fusion rules of the MOSS-Image image graphs: the family's rule `models/ew_chain`, which the
// pipeline passes by default, then the experimental `fusionRules` the caller gives
// (`moss.PipelineOptions.fusionRules` → `MossGenerateState.fusionRules`), in their order. That one
// list reaches the Sessions of the DiT stage (`dit_text`, `dit`, `dit_skip22`) and of the VAE stage
// (the three decoder graphs), and the headroom estimates of the same six graphs with the device's
// `maxComputeWorkgroupsPerDimension`; without caller rules it is the family's rule alone, and every
// other option is what it was. Stand-in components record what each `createSession` and each
// `estimate` received; no GPU (a real `GpuContext` over a fake device whose trial allocations fit).

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import type { EstimateOptions, Session, SessionOptions } from "@karume/runtime";
import { GELU_MUL_RULE } from "@karume/candidate-rules";
import { EW_CHAIN_RULE } from "../moss.ts";
import type { GraphOwner, ModelComponent } from "../src/hub/components.ts";
import type { MossDitContract } from "../src/moss/dit-contract.ts";
import type { MossRopeBase } from "../src/moss/dit-rope.ts";
import {
  type MossDitStageState,
  planMossDit,
  runMossTextStage,
  withMossDitPair,
} from "../src/moss/dit-stage.ts";
import { generateMoss, type MossGenerateState } from "../src/moss/generate.ts";
import { MOSS_PRACTICAL_PRESET, mossPreset } from "../src/moss/preset.ts";
import { MOSS_TEXT_WIDTH, type MossTextEmbeddings } from "../src/moss/request.ts";
import type { MossVaeContract } from "../src/moss/vae-contract.ts";
import { decodeMossVae, type MossVaeStageState, planMossVae } from "../src/moss/vae-stage.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../../runtime/src/gpu/device.ts";
import { fakeDevice } from "../../runtime/tests/helpers/fake-gpu.ts";
import { syntheticTrainingGrid } from "./helpers/moss-dit.ts";

type Graph = GraphOwner["graph"];

/** Where a stand-in stops the stage once the Session under test has been asked for. */
const STOP = new Error("stand-in: stop after the recorded Session");

/** The limits the stages read; the dispatch limit is the one a rule's estimate must carry. */
const LIMITS: RequiredLimits = {
  maxBufferSize: 2_147_483_644,
  maxStorageBufferBindingSize: 2_147_483_644,
  maxUniformBufferBindingSize: 65_536,
  maxStorageBuffersPerShaderStage: 10,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 49_152,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxComputeWorkgroupSizeY: 1024,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65_535,
};

/** A real `GpuContext` whose trial allocations fit and whose residents are stand-ins. */
const roomyGpu = (): GpuContext => {
  const device = fakeDevice({
    // DOM types are not reproduced whole (the test-only boundary of `fake-gpu.ts`).
    createBuffer: () => ({ destroy: (): void => undefined }) as unknown as GPUBuffer,
  });
  const gpu = new GpuContext(device, readAdapterInfo({}), LIMITS, new Set());
  return Object.assign(gpu, {
    createResident: (byteLength: number, label: string) =>
      Promise.resolve({
        byteLength,
        label,
        write: (): void => undefined,
        clear: (): void => undefined,
        dispose: (): void => undefined,
      }),
  }) as GpuContext;
};

type Recorded = {
  /** Each `createSession`'s options, by component. */
  readonly sessions: { readonly component: string; readonly options: SessionOptions }[];
  /** Each `estimate`'s options, by component. */
  readonly estimates: { readonly component: string; readonly options: EstimateOptions }[];
};

const recorder = (): Recorded => ({ sessions: [], estimates: [] });

/**
 * A component's stand-in: records its `createSession` and `estimate` calls. `session` is what
 * `createSession` returns after recording (absent = it throws {@link STOP}).
 *
 * NOTE: the cast — the graph holds only the symbol the stages read.
 */
const component = (
  name: string,
  symbol: string,
  recorded: Recorded,
  session?: () => Session,
): ModelComponent => ({
  graph: { symbols: [symbol] } as unknown as Graph,
  createSession: (_gpu, options = {}) => {
    recorded.sessions.push({ component: name, options });
    return session === undefined ? Promise.reject(STOP) : Promise.resolve(session());
  },
  assets: {},
  asset: () => {
    throw new Error(`${name}: no asset`);
  },
  estimate: (options) => {
    recorded.estimates.push({ component: name, options });
    return { peakAccountedBytes: 2 ** 20 } as ReturnType<ModelComponent["estimate"]>;
  },
  maxPartBytes: 0,
});

/** The lender's stand-in Session (`dit` — exports weights, closes). NOTE: the cast as above. */
const lenderSession = (): Session =>
  ({
    exportWeight: (name: string) => ({ lentBy: "dit", name }),
    dispose: () => Promise.resolve(),
  }) as unknown as Session;

/** A base table every row of which the stages may read (one band, 8 text rows, 4 cells a side). */
const ROPE_BASE: MossRopeBase = {
  bands: 1,
  maxText: 8,
  maxSide: 4,
  sin: {
    r: new Float32Array(1),
    f: new Float32Array(9),
    h: new Float32Array(7),
    w: new Float32Array(7),
  },
  cos: {
    r: new Float32Array(1),
    f: new Float32Array(9),
    h: new Float32Array(7),
    w: new Float32Array(7),
  },
};

/** The DiT contract the stages read (width 4, the one borrowed initializer). NOTE: cast as above. */
const DIT_CONTRACT = {
  latentChannels: 128,
  width: 4,
  ropeWidth: 8,
  projWidth: 4,
  outputs: { dit: "out", ditSkip22: "out_skip", ditText: "text_out" },
  borrowed: ["w.0"],
  ropeBase: ROPE_BASE,
  trainingGrid: syntheticTrainingGrid(1000),
} as unknown as MossDitContract;

/** The VAE contract the stage reads (one 8-px cell per latent cell, a 128-channel BatchNorm). */
const VAE_CONTRACT: MossVaeContract = {
  pre: { input: "z", output: "h", height: "h", width: "w" },
  attn: { input: "h", output: "a", sequence: "S" },
  post: { input: "a", output: "image", height: "h", width: "w" },
  latentChannels: 32,
  hiddenChannels: 4,
  spatialScale: 8,
  bytesPerCell: 64,
  batchNorm: { mean: new Float32Array(128), std: new Float32Array(128).fill(1) },
  ditChannels: 128,
};

const embeddings = (tokens: number): MossTextEmbeddings => ({
  data: new Float32Array(tokens * MOSS_TEXT_WIDTH),
  shape: [tokens, MOSS_TEXT_WIDTH],
});

/** One 16×16 px plan with both branches (guidance ≠ 1), each text one token. */
const ditPlan = () =>
  planMossDit({ width: 16, height: 16, positive: embeddings(1), negative: embeddings(1) });

const refined = () => ({
  positive: { data: new Float32Array(DIT_CONTRACT.width), tokens: 1 },
  negative: { data: new Float32Array(DIT_CONTRACT.width), tokens: 1 },
});

const ditState = (
  recorded: Recorded,
  fusionRules?: SessionOptions["fusionRules"],
): MossDitStageState => ({
  gpu: roomyGpu(),
  preset: mossPreset(MOSS_PRACTICAL_PRESET),
  contract: DIT_CONTRACT,
  components: {
    dit: component("dit", "S", recorded, lenderSession),
    ditSkip22: component("dit_skip22", "S", recorded),
    ditText: component("dit_text", "T", recorded),
  },
  ...(fusionRules === undefined ? {} : { fusionRules }),
});

const vaeState = (
  recorded: Recorded,
  fusionRules?: SessionOptions["fusionRules"],
): MossVaeStageState => {
  const opened = (): Session => ({ dispose: () => Promise.resolve() }) as unknown as Session;
  return {
    gpu: roomyGpu(),
    contract: VAE_CONTRACT,
    components: {
      pre: component("vae_decoder_pre", "h", recorded, opened),
      attn: component("vae_decoder_attn", "S", recorded, opened),
      post: component("vae_decoder_post", "h", recorded),
    },
    ...(fusionRules === undefined ? {} : { fusionRules }),
  };
};

/** Runs the DiT pair's opening and the VAE's up to the last recorded Session. */
const openStages = async (recorded: Recorded, fusionRules?: SessionOptions["fusionRules"]) => {
  const state = ditState(recorded, fusionRules);
  await assertRejects(() => runMossTextStage(state, ditPlan()), Error, STOP.message);
  await assertRejects(
    () => withMossDitPair(state, ditPlan(), refined(), () => Promise.resolve()),
    Error,
    STOP.message,
  );
  const plan = planMossVae({ width: 16, height: 16 }, VAE_CONTRACT, LIMITS);
  await assertRejects(
    () => decodeMossVae(vaeState(recorded, fusionRules), plan, new Float32Array(128)),
    Error,
    STOP.message,
  );
};

/** The options without the `fusionRules` key and the dispatch limit a rule list needs. */
const stripped = (options: SessionOptions | EstimateOptions) =>
  Object.fromEntries(
    Object.entries(options).filter(
      ([key]) => key !== "fusionRules" && key !== "maxComputeWorkgroupsPerDimension",
    ),
  );

/** `actual` holds exactly the rules of `expected`, the same objects in the same order. */
const assertRuleList = (
  actual: SessionOptions["fusionRules"],
  expected: NonNullable<SessionOptions["fusionRules"]>,
  where: string,
): void => {
  assertEquals(
    actual?.map((rule) => rule.name),
    expected.map((rule) => rule.name),
    `${where}: the rule names`,
  );
  expected.forEach((rule, at) => assertStrictEquals(actual?.[at], rule, `${where}: rule ${at}`));
};

describe("MOSS-Image fusion rules (models/ew_chain by default, then the experimental fusionRules)", () => {
  const STAGE_SESSIONS = [
    "dit_text",
    "dit",
    "dit_skip22",
    "vae_decoder_pre",
    "vae_decoder_attn",
    "vae_decoder_post",
  ];

  it("the DiT stage's three Sessions and the VAE stage's three plan with models/ew_chain, then the caller's rules, beside the preset's knobs", async () => {
    const recorded = recorder();
    await openStages(recorded, [GELU_MUL_RULE]);
    assertEquals(
      recorded.sessions.map(({ component: name }) => name),
      STAGE_SESSIONS,
    );
    for (const { component: name, options } of recorded.sessions) {
      assertRuleList(options.fusionRules, [EW_CHAIN_RULE, GELU_MUL_RULE], name);
    }
    const [, dit, skip] = recorded.sessions;
    // The knobs and the borrowed weights are still there.
    assertEquals(dit.options.attentionScoreStorage, "none");
    assert(Object.hasOwn(skip.options, "sharedWeights"), "dit_skip22 lost its shared weights");
  });

  it("every headroom estimate of those graphs plans with the same list and the device's dispatch limit", async () => {
    const recorded = recorder();
    await openStages(recorded, [GELU_MUL_RULE]);
    assert(recorded.estimates.length > 0);
    for (const { component: name, options } of recorded.estimates) {
      assertRuleList(options.fusionRules, [EW_CHAIN_RULE, GELU_MUL_RULE], name);
      assertEquals(options.maxComputeWorkgroupsPerDimension, 65_535, name);
    }
    assertEquals(
      [...new Set(recorded.estimates.map(({ component: name }) => name))].sort(),
      [...STAGE_SESSIONS].sort(),
    );
  });

  it("without caller rules, every Session and estimate plans with models/ew_chain alone and the dispatch limit, and the rest is what the run with rules passed", async () => {
    const ruled = recorder();
    await openStages(ruled, [GELU_MUL_RULE]);
    const plain = recorder();
    await openStages(plain);
    for (const { component: name, options } of plain.sessions) {
      assertRuleList(options.fusionRules, [EW_CHAIN_RULE], name);
    }
    for (const { component: name, options } of plain.estimates) {
      assertRuleList(options.fusionRules, [EW_CHAIN_RULE], name);
      assertEquals(options.maxComputeWorkgroupsPerDimension, 65_535, name);
    }
    const shape = (entries: Recorded["sessions"] | Recorded["estimates"]) =>
      entries.map(({ component: name, options }) => ({ name, options: stripped(options) }));
    assertEquals(shape(plain.sessions), shape(ruled.sessions));
    assertEquals(shape(plain.estimates), shape(ruled.estimates));
  });

  it("generateMoss hands the state's rules, after models/ew_chain, to the first headroom check's estimates and to the DiT stage, not to the text encoder", async () => {
    const recorded = recorder();
    const dit = ditState(recorded);
    const vae = vaeState(recorded);
    const state: MossGenerateState = {
      gpu: dit.gpu,
      preset: dit.preset,
      dit: { components: dit.components, contract: dit.contract },
      vae: { components: vae.components, contract: vae.contract },
      fusionRules: [GELU_MUL_RULE],
    };
    await assertRejects(
      () =>
        generateMoss(state, {
          width: 16,
          height: 16,
          positive: embeddings(1),
          negative: embeddings(1),
          steps: 2,
          guidance: 3,
          shift: 5,
          seed: 0,
        }),
      Error,
      STOP.message,
    );
    // The first stage the embeddings form opens is dit_text, which stops it.
    assertEquals(
      recorded.sessions.map(({ component: name }) => name),
      ["dit_text"],
    );
    assertRuleList(
      recorded.sessions[0].options.fusionRules,
      [EW_CHAIN_RULE, GELU_MUL_RULE],
      "dit_text",
    );
    assertEquals(
      [...new Set(recorded.estimates.map(({ component: name }) => name))].sort(),
      [...STAGE_SESSIONS].sort(),
    );
    for (const { component: name, options } of recorded.estimates) {
      assertRuleList(options.fusionRules, [EW_CHAIN_RULE, GELU_MUL_RULE], name);
    }
  });
});
