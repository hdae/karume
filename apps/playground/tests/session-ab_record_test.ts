import { judgeSessionAb } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import { buildSessionAbPageRecord } from "../src/models/session-ab/record.ts";
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

const build = (record = abRecord({ base: steady(100), rule: steady(90) })) =>
  buildSessionAbPageRecord({
    environment: ENVIRONMENT,
    request,
    distribution: { directory: "models/karume-gemma4-qat", manifestSha256: "ab".repeat(32) },
    source: { kind: "direct" },
    ruleName: RULE,
    record,
    judgements: judgeSessionAb(record, { minGain: request.minGain }),
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
});
