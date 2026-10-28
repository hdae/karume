import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  CHECKPOINTS,
  deviceFit,
  isCheckpointId,
  mebibytes,
  selectionBytes,
  settleSelection,
} from "../src/models/birefnet/settings.ts";

const hex = (character: string): string => character.repeat(64);

/** The limits a device reports (the runtime's `GpuContext.limits` keys), at WebGPU-ish values. */
const limits = {
  maxBufferSize: 2 ** 31,
  maxStorageBufferBindingSize: 2 ** 30,
  maxUniformBufferBindingSize: 65_536,
  maxStorageBuffersPerShaderStage: 10,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 32_768,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxComputeWorkgroupSizeY: 1024,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65_535,
};

/**
 * A two-model manifest shaped like the BiRefNet distributions: `1024` (default) and `2048` share
 * the big parts under `shared/`; `f32` is the default preset of both, `f16` declares a limit.
 */
const manifest = parseManifest(
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "1024",
    models: Object.fromEntries(
      ["1024", "2048"].map((name) => [
        name,
        {
          pipeline: "birefnet/2",
          assets: {},
          weights: {
            matte: Object.fromEntries(
              ["f32", "f16"].map((dtype) => [
                dtype,
                {
                  container: {
                    descriptor: {
                      graph: { length: 8, sha256: hex("d") },
                      model: { length: 8, sha256: hex("e") },
                    },
                    parts: [
                      {
                        path: `${name}/matte/model.${dtype}-00001-of-00002.krm`,
                        // container-v1 §8: header 24 + graph 8 + model descriptor 8.
                        size: 40,
                        sha256: hex(name === "1024" ? "a" : "b"),
                      },
                      {
                        path: `shared/matte/model.${dtype}-00002-of-00002.krm`,
                        size: dtype === "f32" ? 4000 : 2000,
                        sha256: hex(dtype === "f32" ? "c" : "f"),
                      },
                    ],
                  },
                },
              ]),
            ),
          },
          quants: {
            f32: { weights: { matte: "f32" }, session: {}, label: "Full precision (f32)" },
            f16: {
              weights: { matte: "f16" },
              session: {},
              requiredLimits: { maxStorageBufferBindingSize: 2 ** 29 },
            },
          },
          defaultQuant: "f32",
          pipelineConfig: {},
        },
      ]),
    ),
  }),
);

describe("the checkpoints", () => {
  it("are the two BiRefNet mounts, HR first", () => {
    expect(CHECKPOINTS.map(({ id }) => id)).toEqual(["birefnet-hr", "lucida"]);
  });

  it("recognise only their own ids", () => {
    expect(isCheckpointId("lucida")).toBe(true);
    expect(isCheckpointId("depth-anything")).toBe(false);
  });
});

describe("settleSelection", () => {
  it("takes the manifest's default model and that model's default preset for an empty form", () => {
    expect(settleSelection({ model: "", quant: "" }, manifest)).toEqual({
      model: "1024",
      quant: "f32",
    });
  });

  it("keeps a valid choice as the same object", () => {
    const form = { model: "2048", quant: "f16" };
    expect(settleSelection(form, manifest)).toBe(form);
  });

  it("replaces only the part the manifest does not have", () => {
    expect(settleSelection({ model: "2048", quant: "i8" }, manifest)).toEqual({
      model: "2048",
      quant: "f32",
    });
    expect(settleSelection({ model: "512", quant: "f16" }, manifest)).toEqual({
      model: "1024",
      quant: "f16",
    });
  });
});

describe("selectionBytes", () => {
  it("sums the parts of the chosen model and preset", () => {
    expect(selectionBytes(manifest, { model: "1024", quant: "f32" })).toBe(4040);
    expect(selectionBytes(manifest, { model: "2048", quant: "f16" })).toBe(2040);
  });
});

describe("deviceFit", () => {
  it("is undeclared when the preset declares no limits, or an empty table", () => {
    expect(deviceFit(undefined, limits)).toEqual({ kind: "undeclared" });
    expect(deviceFit({}, limits)).toEqual({ kind: "undeclared" });
  });

  it("fits when every declared limit is met, equality included", () => {
    expect(
      deviceFit({ maxStorageBufferBindingSize: 2 ** 30, maxBufferSize: 2 ** 20 }, limits),
    ).toEqual({ kind: "fits", declared: ["maxStorageBufferBindingSize", "maxBufferSize"] });
  });

  it("names every limit the device falls short of, with both values", () => {
    expect(
      deviceFit(
        {
          maxStorageBufferBindingSize: 2 ** 31,
          maxBufferSize: 2 ** 20,
          maxComputeWorkgroupStorageSize: 65_536,
        },
        limits,
      ),
    ).toEqual({
      kind: "short",
      shortfalls: [
        { name: "maxStorageBufferBindingSize", required: 2 ** 31, granted: 2 ** 30 },
        { name: "maxComputeWorkgroupStorageSize", required: 65_536, granted: 32_768 },
      ],
    });
  });

  it("reads the declaration of the manifest's preset", () => {
    const spec = manifest.models["1024"]?.quants["f16"]?.requiredLimits;
    expect(deviceFit(spec, { ...limits, maxStorageBufferBindingSize: 2 ** 28 })).toEqual({
      kind: "short",
      shortfalls: [{ name: "maxStorageBufferBindingSize", required: 2 ** 29, granted: 2 ** 28 }],
    });
  });
});

describe("mebibytes", () => {
  it("shows bytes in MiB with one decimal", () => {
    expect(mebibytes(335_544_320)).toBe("320.0 MiB");
    expect(mebibytes(1.5 * 2 ** 20)).toBe("1.5 MiB");
  });
});
