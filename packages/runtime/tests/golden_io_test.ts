// Storage convention of golden io (tests/helpers/golden-io.ts — ADR 0009 / tools/exporter README).
//
// The MUST "reject when the storage dtype disagrees with the declaration" in `ioTensor` is the
// **only** detector across all e2e checks that catches "the storage form of the golden drifted from
// the IR's declared dtype" (every type has 4-byte elements, so a silent reinterpretation of the
// bits would pass). Yet that one line only runs on the path that passes a valid golden, and all the
// checks stayed green with the assert removed.
//
// Needs neither a GPU nor real assets (reads a minimal safetensors built with `buildSafetensors` of
// helpers/safetensors.ts through `parseSafetensors`).

import { test } from "vite-plus/test";
import { assertEquals, AssertionError, assertStrictEquals, assertThrows } from "@std/assert";
import { parseSafetensors, type TensorView } from "../src/format/safetensors.ts";
import { IO_ENCODING, ioTensor } from "./helpers/golden-io.ts";
import { buildSafetensors, f32Bytes } from "./helpers/safetensors.ts";

/** Bytes of 4-byte integers (the content of I32 / U32). */
const i32Bytes = (values: readonly number[]): Uint8Array =>
  new Uint8Array(Int32Array.from(values).buffer);

/**
 * A golden io holding one tensor of each of the 3 semantic dtypes (the only place with the valid
 * form).
 */
const goldenIo = () =>
  parseSafetensors(
    buildSafetensors([
      { name: "output.0", dtype: "F32", shape: [2], data: f32Bytes([1.5, -2.5]) },
      { name: "output.1", dtype: "I32", shape: [2], data: i32Bytes([7, -9]) },
      { name: "output.2", dtype: "U32", shape: [2], data: i32Bytes([0, 1]) },
    ]),
  );

const viewOf = (file: ReturnType<typeof goldenIo>, name: string): TensorView => {
  const view = file.tensors.get(name);
  if (view === undefined) throw new Error(`${name} is missing`);
  return view;
};

test("IO_ENCODING declares the storage form of all 3 semantic dtypes (a copy of ADR 0009)", () => {
  assertEquals(IO_ENCODING, { f32: "F32", i32: "I32", bool: "U32" });
});

test("ioTensor reads with the type of the declared dtype and does not copy the bytes", () => {
  const file = goldenIo();
  const f32 = ioTensor(file, viewOf(file, "output.0"), "f32");
  assertEquals([f32.dtype, f32.shape], ["f32", [2]]);
  assertEquals([...f32.data], [1.5, -2.5]);

  const i32 = ioTensor(file, viewOf(file, "output.1"), "i32");
  assertEquals([i32.dtype, i32.shape], ["i32", [2]]);
  assertEquals([...i32.data], [7, -9]);

  const bool = ioTensor(file, viewOf(file, "output.2"), "bool");
  assertEquals([bool.dtype, bool.shape], ["bool", [2]]);
  assertEquals([...bool.data], [0, 1]);

  // Being a view = rewriting the original bytes moves the value read too (detects a copy)
  new DataView(file.buffer).setFloat32(viewOf(file, "output.0").byteOffset, 42, true);
  assertStrictEquals(f32.data[0], 42);
});

test("ioTensor rejects pairs whose storage dtype disagrees with the declaration (no bit reinterpretation)", () => {
  const file = goldenIo();
  // Every type has 4-byte elements, so without this check all 3 would "read as numbers"
  const cases: readonly (readonly ["f32" | "i32" | "bool", string])[] = [
    ["f32", "output.1"], // I32 as f32
    ["i32", "output.0"], // F32 as i32
    ["bool", "output.0"], // F32 as bool
  ];
  for (const [dtype, name] of cases) {
    const error = assertThrows(
      () => ioTensor(file, viewOf(file, name), dtype),
      AssertionError,
      "dtype of golden tensor",
    );
    // MUST: the diagnostic names the tensor (without knowing which output drifted it cannot be
    // fixed)
    assertEquals(error.message.includes(`'${name}'`), true, error.message);
  }
});
