import { describe, expect, it } from "vite-plus/test";
import { dawnBackend, readEnvironment } from "@karume/sweep-app-support/environment";

import { fakeWebGpu } from "./helpers/fake-webgpu.ts";

const INFO = {
  vendor: "nvidia",
  architecture: "ampere",
  device: "NVIDIA GeForce RTX 3080 Ti",
  description: "NVIDIA GeForce RTX 3080 Ti (driver 1.2.3)",
};

/** Limits kept on the prototype, as GPUSupportedLimits keeps them, plus a member that is no limit. */
const prototypeLimits = () =>
  Object.create({ maxBindGroups: 4, maxBufferSize: 268_435_456, toString: "not a number" });

const gpuOf = () =>
  fakeWebGpu({
    adapterFeatures: ["timestamp-query", "shader-f16", "subgroups"],
    adapterLimits: prototypeLimits(),
    adapterInfo: { ...INFO, subgroupMinSize: 32 },
    wgslLanguageFeatures: [
      "readonly_and_readwrite_storage_textures",
      "packed_4x8_integer_dot_product",
    ],
  }).gpu;

describe("readEnvironment", () => {
  it("reads the adapter's info, its sorted features, every numeric limit by name and the sorted language features", async () => {
    expect(await readEnvironment(gpuOf(), { host: "node", userAgent: "node/24" })).toStrictEqual({
      host: "node",
      userAgent: "node/24",
      adapter: INFO,
      features: ["shader-f16", "subgroups", "timestamp-query"],
      limits: { maxBindGroups: 4, maxBufferSize: 268_435_456 },
      wgslLanguageFeatures: [
        "packed_4x8_integer_dot_product",
        "readonly_and_readwrite_storage_textures",
      ],
    });
  });

  it("carries crossOriginIsolated and the backend only when the host gives them", async () => {
    const environment = await readEnvironment(gpuOf(), {
      host: "browser",
      userAgent: "Chrome",
      crossOriginIsolated: false,
      backend: { name: "d3d12", source: "declared" },
    });
    expect(environment.crossOriginIsolated).toBe(false);
    expect(environment.backend).toEqual({ name: "d3d12", source: "declared" });
    const bare = await readEnvironment(gpuOf(), { host: "browser", userAgent: "Chrome" });
    expect(Object.keys(bare)).not.toContain("crossOriginIsolated");
    expect(Object.keys(bare)).not.toContain("backend");
  });
});

describe("dawnBackend", () => {
  it("takes a backend= flag, else Dawn's default for the platform", () => {
    expect(dawnBackend(["disable-dawn-features=x", "backend=opengl"], "linux")).toEqual({
      name: "opengl",
      source: "dawn-flag",
    });
    expect(dawnBackend([], "linux")).toEqual({ name: "vulkan", source: "dawn-default" });
    expect(dawnBackend([], "darwin")).toEqual({ name: "metal", source: "dawn-default" });
    expect(dawnBackend([], "win32")).toEqual({ name: "d3d12", source: "dawn-default" });
    expect(() => dawnBackend([], "aix")).toThrow("aix");
  });
});
