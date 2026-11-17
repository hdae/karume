// The entry `@karume/runtime/container`: it imports where WebGPU does not exist, leaves the global
// object as it found it, and exports, as values, only the format's serializer, its sha256 and the
// 16-bit decoders. What a converter relies on holds: a model descriptor read from a container
// serializes back to the same bytes, and a sha256 replaced in it keeps the document's length. The
// names and kinds of every export, types included, are pinned by the public-surface snapshot
// (tests/public_surface_test.ts). No GPU needed.
//
// The import runs in a separate Node process started without the test harness's setup, so no
// WebGPU global exists there (the harness installs Dawn's in this one).

import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { asSha256Hex, serializeModelDescriptor } from "../container.ts";
import { openContainer } from "../mod.ts";
import { parseIrDeclaration } from "../src/format/ir.ts";
import { writeModelContainer } from "./helpers/container-write.ts";

const ENTRY = fileURLToPath(new URL("../container.ts", import.meta.url));

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
const VALUES = ["asSha256Hex", "decodeBf16", "decodeF16", "serializeModelDescriptor", "sha256Hex"];

describe("@karume/runtime/container", () => {
  it("imports in a process without WebGPU, adds no global, and exports only the format's parts for a writer", () => {
    const seen = probe();
    assertEquals(seen.webgpu, { bufferUsage: "undefined", navigatorGpu: "undefined" });
    assertEquals(seen.addedGlobals, []);
    assertEquals(seen.values, VALUES);
  });

  it("serializes a model descriptor read from a container back to its bytes, a replaced sha256 keeping the length", async () => {
    const written = await writeModelContainer({
      graphs: {
        g: parseIrDeclaration(
          JSON.stringify({
            format: "karume-ir",
            version: 2,
            requires: { ops: ["matmul"] },
            symbols: [],
            inputs: [{ name: "x", dtype: "f32", shape: [1, 4] }],
            outputs: ["y"],
            initializers: { w: {} },
            values: { w: { dtype: "f32", shape: [4, 4] }, y: { dtype: "f32", shape: [1, 4] } },
            nodes: [{ op: "matmul", ins: ["x", "w"], outs: ["y"], attrs: {} }],
          }),
        ),
      },
      consts: [],
      weights: [
        { graph: "g", initializer: "w", bytes: new Uint8Array(64), encoding: { codec: "f32" } },
      ],
      assets: [],
      provenance: { license: "apache-2.0" },
    });
    const opened = await openContainer({ kind: "parts", parts: written.parts });
    const model = opened.model;
    assert(model !== undefined);
    assertEquals(serializeModelDescriptor(model), written.modelDescriptorBytes);
    const zero = asSha256Hex("0".repeat(64), "test");
    const replaced = serializeModelDescriptor({
      ...model,
      blocks: model.blocks.map((block) => ({ ...block, sha256: zero })),
    });
    assertEquals(replaced.byteLength, written.modelDescriptorBytes.byteLength);
    assertThrows(() => asSha256Hex("0".repeat(63), "test"), Error, "is not a sha256");
  });
});
