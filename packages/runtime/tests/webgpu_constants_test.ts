// The **spec-transcription check** of the WebGPU bit flags the runtime defines itself
// (src/gpu/webgpu-constants.ts).
//
// The runtime defines them itself for import safety (importing must not fail in an environment
// without WebGPU), so the values are copied from the spec by hand. A transcription mistake turns
// into silent misbehaviour (a missing usage bit is a createBuffer validation error, a surplus one
// passes quietly), so in an environment that has the namespaces every bit is compared with the
// real thing. The expected values **come from the platform namespaces** — writing the same numbers
// as the runtime's definitions a second time would make the check a tautology that catches no
// transcription mistake.
//
// The namespaces do not exist in an environment without WebGPU, so the test checks
// `Object.hasOwn(globalThis, ...)` (a bare identifier reference fails at evaluation time even
// inside this file). An environment without them is SKIPped explicitly.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { BUFFER_USAGE, MAP_MODE } from "../src/gpu/webgpu-constants.ts";

const BUFFER_USAGE_NAMESPACE = Object.hasOwn(globalThis, "GPUBufferUsage");
const MAP_MODE_NAMESPACE = Object.hasOwn(globalThis, "GPUMapMode");

if (!BUFFER_USAGE_NAMESPACE || !MAP_MODE_NAMESPACE) {
  console.warn(
    "[karume] SKIP the spec-transcription comparison: the WebGPU flag namespaces are missing" +
      " (whether the runtime's own constants match the spec is not verified)",
  );
}

test.skipIf(!BUFFER_USAGE_NAMESPACE)(
  "BUFFER_USAGE copies every bit of GPUBufferUsage as the spec defines it",
  () => {
    assertEquals(
      { ...BUFFER_USAGE },
      {
        MAP_READ: GPUBufferUsage.MAP_READ,
        MAP_WRITE: GPUBufferUsage.MAP_WRITE,
        COPY_SRC: GPUBufferUsage.COPY_SRC,
        COPY_DST: GPUBufferUsage.COPY_DST,
        INDEX: GPUBufferUsage.INDEX,
        VERTEX: GPUBufferUsage.VERTEX,
        UNIFORM: GPUBufferUsage.UNIFORM,
        STORAGE: GPUBufferUsage.STORAGE,
        INDIRECT: GPUBufferUsage.INDIRECT,
        QUERY_RESOLVE: GPUBufferUsage.QUERY_RESOLVE,
      },
    );
  },
);

test.skipIf(!MAP_MODE_NAMESPACE)(
  "MAP_MODE copies every bit of GPUMapMode as the spec defines it",
  () => {
    assertEquals(
      { ...MAP_MODE },
      {
        READ: GPUMapMode.READ,
        WRITE: GPUMapMode.WRITE,
      },
    );
  },
);
