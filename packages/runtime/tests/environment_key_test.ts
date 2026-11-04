// Pins how an environment key (the ID that looks up per-device reference values) is spelled.
//
// If the key does not absorb the implementation differences (Deno uses description / Chrome uses
// vendor + architecture), reference values are either written silently into "another machine's row"
// or a new row appears every time the machine changes. Either way the check loses its meaning
// without going red, so the spelling rule itself is pinned here.

import { platformObject } from "@karume/test-support/webgpu";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  assertAdapterMatchesEnvironment,
  describeEnvironment,
  ENVIRONMENT,
  type Environment,
  environmentKey,
  verifyRunningGpu,
} from "./helpers/environment.ts";

/** A full `GPUAdapterInfo` (only 4 fields are read, but the type requires all of them). */
const adapterInfo = (fields: Partial<Omit<GPUAdapterInfo, "__brand">>): GPUAdapterInfo =>
  platformObject<GPUAdapterInfo>({
    vendor: "",
    architecture: "",
    device: "",
    description: "",
    subgroupMinSize: 0,
    subgroupMaxSize: 0,
    isFallbackAdapter: false,
    ...fields,
  });

test("environment key: Node (Dawn) builds it from device (description is the driver version, so it is not used)", () => {
  assertEquals(
    environmentKey(
      "node",
      adapterInfo({
        vendor: "nvidia",
        architecture: "ampere",
        device: "nvidia-geforce-rtx-3080-ti",
        description: "NVIDIA: 615.71.09 615.71.9.0",
      }),
    ),
    "node-nvidia-geforce-rtx-3080-ti",
  );
});

test("environment key: Deno builds it from description (trademark decorations are dropped)", () => {
  // Measured values (Intel Arc B570 / Deno 2.9 — architecture is empty and device is a numeric ID).
  assertEquals(
    environmentKey(
      "deno",
      adapterInfo({ vendor: "32902", device: "57868", description: "Intel(R) Graphics (BMG G21)" }),
    ),
    "deno-intel-graphics-bmg-g21",
  );
});

test("environment key: whitespace and upper case collapse (the recorded spelling of the RTX)", () => {
  assertEquals(
    environmentKey(
      "deno",
      adapterInfo({ vendor: "4318", description: "NVIDIA GeForce RTX 3080 Ti" }),
    ),
    "deno-nvidia-geforce-rtx-3080-ti",
  );
});

test("environment key: with an empty description it is built from vendor + architecture (the Chrome form)", () => {
  assertEquals(
    environmentKey("chrome", adapterInfo({ vendor: "apple", architecture: "metal-3" })),
    "chrome-apple-metal-3",
  );
});

test("environment key: throws when no name can be taken at all (no row with an empty key)", () => {
  assertThrows(
    () => environmentKey("deno", adapterInfo({ device: "57868" })),
    Error,
    "環境キーを作れない",
  );
});

test("environment key: throws when the adapter the run acquired differs from the environment key", () => {
  const environment: Environment = {
    key: "deno-intel-graphics-bmg-g21",
    runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
    adapter: {
      vendor: "32902",
      architecture: "",
      device: "57868",
      description: "Intel(R) Graphics (BMG G21)",
    },
    os: { platform: "linux", arch: "x86_64" },
  };
  const context = (description: string): { readonly adapterInfo: GPUAdapterInfo } => ({
    adapterInfo: adapterInfo({ description }),
  });
  // Control: the same adapter does nothing.
  assertAdapterMatchesEnvironment(context("Intel(R) Graphics (BMG G21)"), environment);
  assertThrows(
    () => assertAdapterMatchesEnvironment(context("NVIDIA GeForce RTX 3080 Ti"), environment),
    Error,
    "この走行が実際に取ったアダプタのキー",
  );
});

test("environment key: the description of this environment always names runtime / os", async () => {
  const environment = await describeEnvironment();
  assertEquals(environment.runtime.name, "node");
  assertEquals(environment.runtime.version, process.versions.node);
  assertEquals(environment.os, { platform: process.platform, arch: process.arch });
  // An environment without a GPU has no key (when it has one, it always starts with the runtime
  // name).
  assert(
    ENVIRONMENT.key === undefined || ENVIRONMENT.key.startsWith("node-"),
    `unexpected environment key spelling: ${String(ENVIRONMENT.key)}`,
  );
});

test("environment key: the GPU of a real run is checked on every acquisition, and a mismatching acquisition is destroyed before the throw", () => {
  // On a multi-GPU machine each acquisition may pick a different adapter (two requestAdapter calls
  // are not guaranteed to return the same physical adapter). If the check acquired once and cached
  // the result, in A→A→B the third acquisition B would be compared against A's row without being
  // checked (in write mode, B's output would go into A's row).
  const environment: Environment = {
    key: "deno-intel-graphics-bmg-g21",
    runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
    adapter: {
      vendor: "32902",
      architecture: "",
      device: "57868",
      description: "Intel(R) Graphics (BMG G21)",
    },
    os: { platform: "linux", arch: "x86_64" },
  };
  const destroyed: string[] = [];
  const acquire = (description: string) => ({
    adapterInfo: adapterInfo({ description }),
    destroy: (): void => {
      destroyed.push(description);
    },
  });
  const sequence = [
    "Intel(R) Graphics (BMG G21)",
    "Intel(R) Graphics (BMG G21)",
    "NVIDIA GeForce RTX 3080 Ti",
  ];
  const [first, second, third] = sequence.map(acquire);
  // An acquisition that passes the check is returned as is (the caller uses it and destroys it).
  assert(verifyRunningGpu(first, environment) === first);
  assert(verifyRunningGpu(second, environment) === second);
  assertEquals(destroyed, []);
  assertThrows(
    () => verifyRunningGpu(third, environment),
    Error,
    "この走行が実際に取ったアダプタのキー",
  );
  // So that it can be used before the caller's try, a mismatching acquisition is released here
  // before failing.
  assertEquals(destroyed, ["NVIDIA GeForce RTX 3080 Ti"]);
});
