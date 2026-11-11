// The entry `@karume/runtime/kernel` (ADR 0148): it imports where WebGPU does not exist, leaves the
// global object as it found it, exports the parts for writing a rule and nothing else as values,
// and its `defineRule` is the external define entry (a name without a namespace is refused). The
// names and kinds of every export, types included, are pinned by the public-surface snapshot
// (tests/public_surface_test.ts); this file states the intent for the values. No GPU needed.
//
// The import runs in a separate Node process started without the test harness's setup, so no
// WebGPU global exists there (the harness installs Dawn's in this one).

import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { describe, it } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { defineRule } from "../kernel.ts";
import { ExecutionError } from "../src/runtime/plan.ts";

const ENTRY = fileURLToPath(new URL("../kernel.ts", import.meta.url));

/** What the fresh process saw: WebGPU's presence before the import, the exports, the globals added. */
type Probe = {
  readonly webgpu: { readonly bufferUsage: string; readonly navigatorGpu: string };
  readonly values: readonly string[];
  readonly addedGlobals: readonly string[];
};

const PROBE = `
const before = new Set(Reflect.ownKeys(globalThis).map(String));
const webgpu = {
  bufferUsage: typeof globalThis.GPUBufferUsage,
  navigatorGpu: typeof globalThis.navigator?.gpu,
};
const entry = await import(process.argv[1]);
const addedGlobals = Reflect.ownKeys(globalThis).map(String).filter((name) => !before.has(name));
console.log(JSON.stringify({ webgpu, values: Object.keys(entry).sort(), addedGlobals }));
`;

const probe = (): Probe => {
  const output = execFileSync(process.execPath, ["--input-type=module", "-e", PROBE, ENTRY], {
    encoding: "utf8",
  });
  return JSON.parse(output) as Probe;
};

/** The values the entry exports (the types are the snapshot's). */
const VALUES = [
  "CodegenError",
  "IS_NAN_BITS_WGSL",
  "SIGMOID_STABLE_WGSL",
  "TANH_STABLE_WGSL",
  "allF32",
  "defineRule",
  "fusedBinaryWgsl",
  "gridStrideWorkgroups",
  "prologueUnaryWgsl",
  "sameShape",
  "stagedF32",
  "stagedLoopWgsl",
];

describe("@karume/runtime/kernel", () => {
  it("imports in a process without WebGPU, adds no global, and exports only the parts for writing a rule", () => {
    const seen = probe();
    // The precondition: the fresh process has no WebGPU, so the import below proves import safety.
    assertEquals(seen.webgpu, { bufferUsage: "undefined", navigatorGpu: "undefined" });
    assertEquals(seen.addedGlobals, []);
    assertEquals(seen.values, VALUES);
  });

  it("defineRule is the external entry: a name without '<namespace>/' is refused", () => {
    const definition = {
      heads: ["neg"],
      contract: "E",
      internals: "private",
      match: () => undefined,
      build: () => {
        throw new Error("never built");
      },
    } as const;
    assertEquals(defineRule({ ...definition, name: "test/neg" }).name, "test/neg");
    assertThrows(
      () => defineRule({ ...definition, name: "neg" as `${string}/${string}` }),
      ExecutionError,
      "'<namespace>/<rule>'",
    );
  });
});
