// The read side of `helpers/safetensors.ts` (readBuffer / viewOf / floatsOf), which the golden and
// fixture tests of several families share. No GPU and no real assets: a file built by
// `buildSafetensors` is written to a temporary directory and read back.
//
// A wrong reader here does not fail the e2e checks that use it in an obvious way — it feeds them
// the wrong numbers, or lets a missing or mistyped golden tensor through — so its contract is
// pinned on its own.

import { makeTempDir, remove, writeFile } from "@karume/test-support/fs";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { assertEquals, assertThrows } from "@std/assert";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { buildSafetensors, f32Bytes, floatsOf, readBuffer, viewOf } from "./helpers/safetensors.ts";

describe("the safetensors read helpers", () => {
  let root: URL;
  let bytes: Uint8Array<ArrayBuffer>;
  let file: SafetensorsFile;
  beforeAll(async () => {
    root = new URL(`file://${await makeTempDir({ prefix: "karume-safetensors-read-" })}/`);
    bytes = new Uint8Array(
      buildSafetensors([
        { name: "floats", dtype: "F32", shape: [3], data: f32Bytes([1.5, -2, 3]) },
        { name: "count", dtype: "I32", shape: [1], data: new Uint8Array(Int32Array.of(7).buffer) },
      ]),
    );
    const url = new URL("case.safetensors", root);
    await writeFile(url, bytes);
    file = parseSafetensors(await readBuffer(url));
  });
  afterAll(async () => {
    await remove(root, { recursive: true });
  });

  describe("readBuffer", () => {
    it("returns exactly the bytes of the file", async () => {
      const buffer = await readBuffer(new URL("case.safetensors", root));
      assertEquals(new Uint8Array(buffer), bytes);
    });
  });

  describe("viewOf", () => {
    it("returns the tensor's dtype and shape", () => {
      const view = viewOf(file, "count", "case");
      assertEquals([view.dtype, view.shape], ["I32", [1]]);
    });
    it("throws, naming the place and the key, when the key is missing", () => {
      assertThrows(() => viewOf(file, "absent", "case"), Error, "case: no 'absent'");
    });
  });

  describe("floatsOf", () => {
    it("returns the F32 values of the tensor", () => {
      assertEquals([...floatsOf(file, "floats", "case")], [1.5, -2, 3]);
    });
    it("throws instead of reinterpreting a tensor of another dtype", () => {
      assertThrows(() => floatsOf(file, "count", "case"), Error, "case: 'count' is I32");
    });
  });
});
