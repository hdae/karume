import type { FromPretrainedOptions } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import { fakeDevice, fakeGpuContext } from "../../../packages/runtime/tests/helpers/fake-gpu.ts";
import {
  buildGemmaPipelineWith,
  type GemmaBuildInput,
  type GemmaEntryPoints,
  type PretrainedRef,
} from "../src/gemma4/pipeline.ts";

import { gemmaManifestText, memoryDistribution } from "./helpers/session-ab-manifests.ts";

/** What a fake entry point was called with. */
type Call = {
  readonly entryPoint: "regular" | "mobile";
  readonly ref: PretrainedRef;
  readonly options: FromPretrainedOptions;
};

/** Entry points that record their calls and return which one was called. */
const fakeEntryPoints = (calls: Call[]): GemmaEntryPoints<"regular", "mobile"> => ({
  regular: (ref, options) => {
    calls.push({ entryPoint: "regular", ref, options });
    return Promise.resolve("regular");
  },
  mobile: (ref, options) => {
    calls.push({ entryPoint: "mobile", ref, options });
    return Promise.resolve("mobile");
  },
});

/** The builder only passes the GPU context on: a real one over a fake device is enough. */
const GPU = fakeGpuContext(fakeDevice());
const REF = memoryDistribution(gemmaManifestText(["e2b"]));
const SIGNAL = new AbortController().signal;

const INPUT: GemmaBuildInput = {
  ref: REF,
  gpu: GPU,
  load: { mount: "gemma4-qat", model: "e2b", quant: "i4" },
  signal: SIGNAL,
};

const build = async (input: GemmaBuildInput): Promise<{ result: string; calls: Call[] }> => {
  const calls: Call[] = [];
  const result = await buildGemmaPipelineWith(fakeEntryPoints(calls), input);
  return { result, calls };
};

describe("the workbench's Gemma 4 pipeline builder", () => {
  it("builds the gemma4 and gemma4-qat mounts with gemma.Pipeline, the model and preset named, from the given reference", async () => {
    for (const mount of ["gemma4", "gemma4-qat"] as const) {
      const { result, calls } = await build({ ...INPUT, load: { ...INPUT.load, mount } });
      expect(result).toBe("regular");
      expect(calls).toHaveLength(1);
      expect(calls[0].ref).toBe(REF);
      expect(calls[0].options).toEqual({ gpu: GPU, quant: "i4", signal: SIGNAL, model: "e2b" });
    }
  });

  it("builds the mobile QAT mount with gemma4QatMobile.Pipeline", async () => {
    const { result, calls } = await build({
      ...INPUT,
      load: { mount: "gemma4-qat-mobile", model: "e4b", quant: "q4_0" },
    });
    expect(result).toBe("mobile");
    expect(calls.map(({ entryPoint, options }) => [entryPoint, options.model])).toEqual([
      ["mobile", "e4b"],
    ]);
  });

  it("refuses a mobile model the mobile entry point does not take, before any read", async () => {
    const calls: Call[] = [];
    await expect(
      buildGemmaPipelineWith(fakeEntryPoints(calls), {
        ...INPUT,
        load: { mount: "gemma4-qat-mobile", model: "12b", quant: "q4_0" },
      }),
    ).rejects.toThrow("gemma4QatMobile.Pipeline accepts the models e2b and e4b, not '12b'");
    expect(calls).toEqual([]);
  });

  it("passes the load options, the Session and pipeline options, the diagnostics callback and the progress as they are", async () => {
    const onRunDiagnostics = (): void => undefined;
    const onProgress = (): void => undefined;
    const { calls } = await build({
      ...INPUT,
      load: {
        mount: "gemma4-qat",
        model: "26b",
        quant: "i4",
        fullStateStorage: "f16",
        expertCache: { slots: [64, 32], hostBytes: 2 ** 30, depth: 2 },
      },
      onRunDiagnostics,
      onProgress,
      sessionOptions: { integerGemv: true, packedStaticQuantize: true },
      pipelineOptions: { decodeAhead: 2 },
    });
    expect(calls[0].options).toEqual({
      integerGemv: true,
      packedStaticQuantize: true,
      decodeAhead: 2,
      gpu: GPU,
      quant: "i4",
      fullStateStorage: "f16",
      expertCache: { slots: [64, 32], hostBytes: 2 ** 30, depth: 2 },
      signal: SIGNAL,
      onRunDiagnostics,
      onProgress,
      model: "26b",
    });
  });

  it("leaves out what the input does not name: no f16 KV, no expert cache, no callback, no drafter", async () => {
    const { calls } = await build(INPUT);
    for (const name of [
      "fullStateStorage",
      "expertCache",
      "onRunDiagnostics",
      "onProgress",
      "fusionRules",
      "speculative",
    ]) {
      expect(Object.hasOwn(calls[0].options, name), name).toBe(false);
    }
  });

  it("builds the drafter a load asks for on gemma.Pipeline, at the drafter's own draft count", async () => {
    for (const mount of ["gemma4", "gemma4-qat"] as const) {
      const { calls } = await build({ ...INPUT, load: { ...INPUT.load, mount, drafter: true } });
      expect(calls[0].options).toEqual({
        gpu: GPU,
        quant: "i4",
        signal: SIGNAL,
        model: "e2b",
        speculative: {},
      });
    }
  });

  it("refuses a mobile load with the drafter, before any read", async () => {
    const calls: Call[] = [];
    await expect(
      buildGemmaPipelineWith(fakeEntryPoints(calls), {
        ...INPUT,
        load: { mount: "gemma4-qat-mobile", model: "e2b", quant: "i4", drafter: true },
      }),
    ).rejects.toThrow("gemma4QatMobile.Pipeline has no MTP drafter");
    expect(calls).toEqual([]);
  });
});
