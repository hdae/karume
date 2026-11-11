import { judgeSessionAb } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import { readFileSync } from "node:fs";

import {
  buildSessionAbPageRecord,
  PACKAGE_VERSIONS,
  type SessionAbArmFacts,
} from "../src/models/session-ab/record.ts";
import {
  INITIAL_SESSION_AB_FORM,
  parseSessionAbForm,
  sessionAbRequest,
} from "../src/models/session-ab/request.ts";

import { abRecord, RULE, steady } from "./helpers/session-ab-records.ts";

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

const settings = (() => {
  const parsed = parseSessionAbForm({ ...INITIAL_SESSION_AB_FORM, rounds: "3", backend: "d3d12" });
  if (!parsed.ok) throw new Error(parsed.errors.join(" "));
  return parsed.value;
})();

const request = sessionAbRequest(
  { mount: "gemma4-qat", model: "26b", quant: "i4", expertCache: { slots: 64 } },
  settings,
  RULE,
);

const VERSIONS = { runtime: "9.8.7", models: "6.5.4" };

/** What the two arms learned: the cache as resolved (two layers here) and the rule's keys. */
const FACTS: readonly SessionAbArmFacts[] = [
  { name: "base", facts: { expertCache: { slots: [64, 64] }, ruleKeys: [] } },
  {
    name: "rule",
    facts: {
      expertCache: { slots: [64, 64] },
      ruleKeys: ["ext:candidate:gelu_mul:v1:gelu-other:f32:wg128"],
    },
  },
];

const build = (
  record = abRecord({ base: steady(100), rule: steady(90) }),
  arms: readonly SessionAbArmFacts[] = FACTS,
  ofRequest = request,
) =>
  buildSessionAbPageRecord({
    environment: ENVIRONMENT,
    versions: VERSIONS,
    request: ofRequest,
    distribution: { directory: "models/karume-gemma4-qat", manifestSha256: "ab".repeat(32) },
    source: { kind: "direct" },
    ruleName: RULE,
    arms,
    record,
    judgements: judgeSessionAb(record, { minGain: ofRequest.minGain }),
  });

describe("the session A/B page's record", () => {
  it("is the page's envelope around the runtime's record, kept as it is", () => {
    const record = abRecord({ base: steady(100), rule: steady(90) });
    const envelope = build(record);
    expect(envelope).toMatchObject({
      format: "karume-playground-session-ab/1",
      page: "session-ab",
      recordedAt: ENVIRONMENT.recordedAt,
      adapter: ENVIRONMENT.adapter,
      request,
      pipeline: "gemma.Pipeline",
      distribution: { directory: "models/karume-gemma4-qat" },
      source: { kind: "direct" },
      rule: { name: RULE, package: "@karume/candidate-rules" },
    });
    expect(envelope.record).toBe(record);
    expect(envelope.judgements.map(({ arm, verdict }) => [arm, verdict])).toEqual([
      ["rule", "adopt"],
    ]);
  });

  it("survives JSON: the stored record is judged again to the same judgements", () => {
    const envelope = build();
    const stored: typeof envelope = JSON.parse(JSON.stringify(envelope));
    expect(stored).toEqual(envelope);
    const { record, judgements } = stored;
    expect(judgeSessionAb(record, { minGain: request.minGain })).toEqual(judgements);
  });

  it("refuses a runtime record of other arms or another round count than the request's", () => {
    const fourRounds = abRecord({
      base: [...steady(100), [100, 100]],
      rule: [...steady(90), [90, 90]],
    });
    expect(() => build(fourRounds)).toThrow(/is not the request's/);
    const renamed = abRecord({ base: steady(100), rule: steady(90) });
    const swapped = { ...renamed, arms: renamed.arms.toReversed() };
    expect(() => build(swapped)).toThrow(/is not the request's/);
  });

  describe("what a reader needs to run it again", () => {
    it("keeps the versions it is given; the page's are the two packages' own", () => {
      expect(build().versions).toEqual(VERSIONS);
      const versionOf = (name: string): unknown =>
        JSON.parse(
          readFileSync(new URL(`../../../packages/${name}/package.json`, import.meta.url), "utf8"),
        ).version;
      expect(PACKAGE_VERSIONS).toEqual({
        runtime: versionOf("runtime"),
        models: versionOf("models"),
      });
    });

    it("names the rule's dispatch keys as the rule arm reported them", () => {
      expect(build().rule).toEqual({
        name: RULE,
        package: "@karume/candidate-rules",
        dispatchKeys: ["ext:candidate:gelu_mul:v1:gelu-other:f32:wg128"],
      });
    });

    it("names the rule's key prefix when no counts were read (no keys seen)", () => {
      const unread = FACTS.map(({ name, facts }) => ({
        name,
        facts: facts.expertCache === undefined ? {} : { expertCache: facts.expertCache },
      }));
      expect(
        build(abRecord({ base: steady(100), rule: steady(90), noCounts: true }), unread).rule,
      ).toEqual({
        name: RULE,
        package: "@karume/candidate-rules",
        keyPrefix: "ext:candidate:",
      });
    });

    it("keeps the expert cache as each arm resolved it, and none without the cache", () => {
      expect(build().expertCache).toEqual([
        { arm: "base", slots: [64, 64] },
        { arm: "rule", slots: [64, 64] },
      ]);
      const plain = sessionAbRequest(
        { mount: "gemma4-qat", model: "12b", quant: "i4" },
        settings,
        RULE,
      );
      const noCache = FACTS.map(({ name }) => ({ name, facts: { ruleKeys: [] } }));
      expect(Object.hasOwn(build(undefined, noCache, plain), "expertCache")).toBe(false);
    });

    it("refuses arm facts of other arms, or an arm without its resolved cache", () => {
      expect(() => build(undefined, FACTS.toReversed())).toThrow(/are not the request's arms/);
      const missing = [FACTS[0], { name: "rule", facts: { ruleKeys: [] } }];
      expect(() => build(undefined, missing)).toThrow(/rule reported no resolved expert cache/);
    });
  });
});
