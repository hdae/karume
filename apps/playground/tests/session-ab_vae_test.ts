import { readFileSync } from "node:fs";

import { vaeRmsSiluRuleName } from "@karume/candidate-rules";
import { parseManifest } from "@karume/hub";
import type { SessionAbArmRecord, SessionAbRecord } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import {
  buildSessionAbVaePageRecord,
  SESSION_AB_VAE_PAGE_FORMAT,
} from "../src/models/session-ab/record.ts";
import { INITIAL_SESSION_AB_FORM, parseSessionAbForm } from "../src/models/session-ab/request.ts";
import {
  isVaeTarget,
  sessionAbVaeRequest,
  VAE_AB_TARGETS,
  vaeAbModel,
  vaeArmOptions,
  vaeBackendRefusal,
  vaeFeeds,
  type VaeInput,
  vaeRowRule,
  vaeSessionOptions,
} from "../src/models/session-ab/vae.ts";
import { recordedCounts } from "../src/models/session-ab/view.ts";

import { abRecord, steady } from "./helpers/session-ab-records.ts";

const G8 = vaeRmsSiluRuleName(8);
const G4 = vaeRmsSiluRuleName(4);

/** A Wan chunk graph's inputs at small widths (the real latent is [48, 1, 16, 16]). */
const CHUNK_INPUTS: readonly VaeInput[] = [
  { name: "latent", dtype: "f32", shape: [3, 1, 2, 2] },
  { name: "cache_00", dtype: "f32", shape: [3, 2, 2, 2] },
  { name: "cache_01", dtype: "f32", shape: [5, 2, 2, 2] },
];

const wordsOf = (data: Float32Array | Int32Array | Uint32Array): readonly number[] => [
  ...new Uint32Array(data.buffer, data.byteOffset, data.length),
];

const settings = (form = {}) => {
  const parsed = parseSessionAbForm({ ...INITIAL_SESSION_AB_FORM, backend: "d3d12", ...form });
  if (!parsed.ok) throw new Error(parsed.errors.join(" "));
  return parsed.value;
};

const hex = (character: string): string => character.repeat(64);

const container = (component: string, dtype: string) => ({
  container: {
    descriptor: {
      graph: { length: 8, sha256: hex("d") },
      model: { length: 8, sha256: hex("e") },
    },
    parts: [
      { path: `m/${component}/model.${dtype}-00001-of-00002.krm`, size: 40, sha256: hex("a") },
      { path: `m/${component}/model.${dtype}-00002-of-00002.krm`, size: 8, sha256: hex("b") },
    ],
  },
});

/** A `karume/5` manifest with the Wan2.2 mirror's shape: the next chunk graph at two storages. */
const WAN_MANIFEST = parseManifest(
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "ti2v-5b",
    models: {
      "ti2v-5b": {
        pipeline: "wan-ti2v/1",
        assets: {},
        weights: {
          vae_decoder_next: {
            f16: container("vae_decoder_next", "f16"),
            i8: container("vae_decoder_next", "i8"),
          },
          transformer: { i8: container("transformer", "i8") },
        },
        quants: {
          "f16+dit8-a8-attn8-online": {
            weights: { vae_decoder_next: "f16", transformer: "i8" },
            session: {
              linearCompute: "a8",
              attentionCompute: "a8",
              attentionScoreStorage: "none",
            },
          },
          i8: { weights: { vae_decoder_next: "i8", transformer: "i8" }, session: {} },
          "i8-a8-attn8-online-conv8": {
            weights: { vae_decoder_next: "i8", transformer: "i8" },
            session: {
              linearCompute: "a8",
              attentionCompute: "a8",
              attentionScoreStorage: "none",
              convCompute: "a8",
            },
          },
        },
        defaultQuant: "i8-a8-attn8-online-conv8",
        pipelineConfig: {},
      },
    },
  }),
);

const CONV8 = vaeAbModel(WAN_MANIFEST, "wan22-vae-next", "i8-a8-attn8-online-conv8");

const ENVIRONMENT: RecordEnvironment = {
  recordedAt: "2026-11-11T12:00:00.000Z",
  userAgent: "Mozilla/5.0 Chrome/150.0.0.0",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "" },
  device: {
    features: [],
    wgslLanguageFeatures: [],
    subgroupLanesLinear: false,
    gpuTimingEnabled: false,
    limits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBufferBindingSize: 2 ** 31,
      maxBufferSize: 2 ** 31,
    },
  },
};

/** A three-arm runner record whose arms carry the counts a `sessionArm` reports. */
const countedRecord = (): SessionAbRecord => {
  const record = abRecord({ base: steady(584), rule: steady(564) });
  const [base, rule] = record.arms;
  const arm = (
    template: SessionAbArmRecord,
    name: string,
    fusions: Readonly<Record<string, number>>,
    dispatches: number,
  ): SessionAbArmRecord => ({ ...template, name, fusions, dispatches });
  return {
    ...record,
    arms: [
      arm(base, "base", { silu: 29 }, 457),
      arm(rule, "g8", { silu: 0, [G8]: 30 }, 248),
      arm(rule, "g4", { silu: 0, [G4]: 30 }, 248),
    ],
  };
};

const pageRecordOf = (
  record: SessionAbRecord,
  request = sessionAbVaeRequest("wan22-vae-next", CONV8, settings({ rounds: "3" })),
) =>
  buildSessionAbVaePageRecord({
    environment: ENVIRONMENT,
    versions: { runtime: "9.8.7", models: "6.5.4" },
    request,
    distribution: { directory: "models/karume-wan2.2", manifestSha256: hex("f") },
    source: { kind: "direct" },
    container: { component: "vae_decoder_next", part0Sha256: hex("c") },
    record,
    judgements: [],
  });

describe("the session A/B page's VAE targets", () => {
  it("are the two VAE targets and nothing else", () => {
    expect(Object.keys(VAE_AB_TARGETS)).toEqual(["wan22-vae-next", "anima-vae-decode"]);
    expect(["wan22-vae-next", "anima-vae-decode"].every(isVaeTarget)).toBe(true);
    expect(isVaeTarget("moss-dit-forward")).toBe(false);
    expect(isVaeTarget("gemma4-decode")).toBe(false);
  });

  describe("the feeds", () => {
    const feeds = vaeFeeds(CHUNK_INPUTS, VAE_AB_TARGETS["wan22-vae-next"].seed);

    it("feed every input, the caches included, at its declared shape", () => {
      expect(
        Object.fromEntries(Object.entries(feeds).map(([name, { shape }]) => [name, shape])),
      ).toEqual({ latent: [3, 1, 2, 2], cache_00: [3, 2, 2, 2], cache_01: [5, 2, 2, 2] });
    });

    it("draw every value from [−1, 1), and the same words again from the same seed", () => {
      for (const [name, tensor] of Object.entries(feeds)) {
        expect(tensor.dtype, name).toBe("f32");
        expect(
          [...tensor.data].every((value) => value >= -1 && value < 1),
          name,
        ).toBe(true);
      }
      const again = vaeFeeds(CHUNK_INPUTS, VAE_AB_TARGETS["wan22-vae-next"].seed);
      const other = vaeFeeds(CHUNK_INPUTS, VAE_AB_TARGETS["wan22-vae-next"].seed + 1);
      for (const name of Object.keys(feeds)) {
        expect(wordsOf(again[name].data), name).toEqual(wordsOf(feeds[name].data));
        expect(wordsOf(other[name].data), name).not.toEqual(wordsOf(feeds[name].data));
      }
      // One generator across the inputs: the second input does not restart the sequence.
      expect(wordsOf(feeds.cache_00.data).slice(0, 12)).not.toEqual(
        wordsOf(feeds.latent.data).slice(0, 12),
      );
    });

    it("refuse an input they were not made for: a symbolic dimension or another dtype", () => {
      expect(() =>
        vaeFeeds([{ name: "latents", dtype: "f32", shape: [1, 16, "h", "w"] }], 1),
      ).toThrow("input 'latents' has symbol 'h'");
      expect(() => vaeFeeds([{ name: "mask", dtype: "bool", shape: [4] }], 1)).toThrow(
        "no feed for input 'mask' (bool)",
      );
    });
  });

  describe("the VAE Session's options", () => {
    it("are the conv knob alone on Wan2.2, as wanVaeSessionOptions gives the VAE", () => {
      expect(vaeSessionOptions("wan22-vae-next", CONV8.session)).toEqual({ convCompute: "a8" });
      expect(
        vaeSessionOptions(
          "wan22-vae-next",
          vaeAbModel(WAN_MANIFEST, "wan22-vae-next", "f16+dit8-a8-attn8-online").session,
        ),
      ).toEqual({});
      expect(vaeSessionOptions("wan22-vae-next", {})).toEqual({});
    });

    it("are none on anima, whose VAE stage takes none of the quant's session", () => {
      expect(vaeSessionOptions("anima-vae-decode", { convCompute: "a8" })).toEqual({});
      expect(vaeSessionOptions("anima-vae-decode", { linearCompute: "a8" })).toEqual({});
    });
  });

  describe("the model and the request", () => {
    it("take the preset's declared session and the component's storage, and refuse a preset without it", () => {
      expect(CONV8).toEqual({
        model: "ti2v-5b",
        quant: "i8-a8-attn8-online-conv8",
        session: {
          linearCompute: "a8",
          attentionCompute: "a8",
          attentionScoreStorage: "none",
          convCompute: "a8",
        },
        storage: "i8",
      });
      expect(vaeAbModel(WAN_MANIFEST, "wan22-vae-next", "f16+dit8-a8-attn8-online").storage).toBe(
        "f16",
      );
      expect(() => vaeAbModel(WAN_MANIFEST, "wan22-vae-next", "f16")).toThrow(
        "no preset 'f16' of model 'ti2v-5b'",
      );
      // The anima target's component is not in a Wan2.2 manifest.
      expect(() => vaeAbModel(WAN_MANIFEST, "anima-vae-decode", "i8")).toThrow(
        "Preset 'i8' of model 'ti2v-5b' has no 'vae_decoder'",
      );
    });

    it("names the three arms, the options every arm gets and the seed, and opens per visit whatever the form says", () => {
      const request = sessionAbVaeRequest(
        "wan22-vae-next",
        CONV8,
        settings({ openPerVisit: false }),
      );
      expect(request).toEqual({
        target: "wan22-vae-next",
        mount: "wan22",
        component: "vae_decoder_next",
        ...CONV8,
        rounds: 4,
        minGain: 0.01,
        openPerVisit: true,
        backend: "d3d12",
        seed: 0x5eed,
        sessionOptions: { convCompute: "a8" },
        arms: [
          { name: "base", fusionRules: [] },
          { name: "g8", fusionRules: [G8] },
          { name: "g4", fusionRules: [G4] },
        ],
      });
      const anima = sessionAbVaeRequest(
        "anima-vae-decode",
        { model: "anima-turbo-v1.1", quant: "f16", session: {}, storage: "f16" },
        settings(),
      );
      expect([anima.mount, anima.component, anima.sessionOptions]).toEqual([
        "anima",
        "vae_decoder",
        {},
      ]);
    });

    it("gives each arm the request's options and the rule it names, the base none", () => {
      const arms = vaeArmOptions(sessionAbVaeRequest("wan22-vae-next", CONV8, settings()));
      expect(arms.map(({ name }) => name)).toEqual(["base", "g8", "g4"]);
      expect(arms[0].options).toEqual({ convCompute: "a8" });
      for (const [index, rule] of [
        [1, G8],
        [2, G4],
      ] as const) {
        const { options } = arms[index];
        expect(options.convCompute).toBe("a8");
        expect(options.fusionRules?.map(({ name }) => name)).toEqual([rule]);
      }
    });

    it("reads each arm's hits from its own rule, and the base's from the first rule", () => {
      const request = sessionAbVaeRequest("wan22-vae-next", CONV8, settings());
      expect(["base", "g8", "g4"].map((arm) => vaeRowRule(request, arm))).toEqual([G8, G8, G4]);
    });

    it("refuses the metal backend: the rules' rounding barrier does not hold there", () => {
      expect(vaeBackendRefusal("metal")).toContain("must not run on Metal");
      for (const backend of ["d3d12", "vulkan", "other"] as const) {
        expect(vaeBackendRefusal(backend)).toBeUndefined();
      }
    });
  });

  describe("the record", () => {
    it("is an envelope of its own around the runtime's record, with the container and both rules", () => {
      const record = countedRecord();
      const page = pageRecordOf(record);
      expect(page.format).toBe(SESSION_AB_VAE_PAGE_FORMAT);
      expect([page.page, page.target]).toEqual(["session-ab", "wan22-vae-next"]);
      expect(page.container).toEqual({ component: "vae_decoder_next", part0Sha256: hex("c") });
      expect(page.rules).toEqual([
        { name: G8, package: "@karume/candidate-rules", keyPrefix: "ext:candidate:" },
        { name: G4, package: "@karume/candidate-rules", keyPrefix: "ext:candidate:" },
      ]);
      expect(page.record).toBe(record);
      expect(JSON.parse(JSON.stringify(page))).toEqual(page);
      expect(
        page.record.arms.map(
          (arm) => recordedCounts(arm)?.fusions[vaeRowRule(page.request, arm.name)] ?? 0,
        ),
      ).toEqual([0, 30, 30]);
    });

    it("refuses a runtime record of other arms or rounds, or whose arms reported no counts", () => {
      const record = countedRecord();
      expect(() => pageRecordOf({ ...record, rounds: 4 })).toThrow("is not the request's");
      expect(() => pageRecordOf({ ...record, arms: [...record.arms].reverse() })).toThrow(
        "is not the request's",
      );
      const [base, g8, g4] = record.arms;
      const uncounted = { name: g4.name, timesMs: g4.timesMs, identity: g4.identity };
      expect(() => pageRecordOf({ ...record, arms: [base, g8, uncounted] })).toThrow(
        "arm(s) g4 reported no counts",
      );
    });
  });

  describe("its feeds' twin in the opt-in Node A/B", () => {
    /** The text between the twin markers, `export` aside. */
    const twinOf = (path: string): string => {
      const text = readFileSync(new URL(path, import.meta.url), "utf8");
      const begin = text.indexOf("// ---- The VAE feeds: twin begin");
      const endMarker = "// ---- The VAE feeds: twin end ----";
      const end = text.indexOf(endMarker);
      expect(begin, path).toBeGreaterThanOrEqual(0);
      expect(end, path).toBeGreaterThan(begin);
      return text.slice(begin, end + endMarker.length).replace(/^export /gm, "");
    };

    it("is the same text as the page's (a change to one is made to the other)", () => {
      const page = twinOf("../src/models/session-ab/vae.ts");
      expect(page).toContain("const vaeFeeds = (");
      expect(twinOf("../../../packages/e2e/tests/e2e_vae_session_ab_test.ts")).toBe(page);
    });
  });
});
