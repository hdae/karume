// Reads the expected bits from official Torch on CPU (fixtures/static-quantize-oracle.safetensors).
// The CPU reference side (static_quantize_test.ts) and the real-GPU side
// (gpu_static_quantize_test.ts) compare against **the same oracle**, so the reading lives in this
// one file.
import { assert } from "@std/assert";
import { readFile } from "@karume/test-support/fs";
import { parseSafetensors } from "../../src/format/safetensors.ts";

export const bits = (data: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

export type Case = { scale: number; x: Float32Array<ArrayBuffer>; y: Uint32Array<ArrayBuffer> };

export const cases = async (): Promise<readonly Case[]> => {
  const bytes = await readFile(
    new URL("../fixtures/static-quantize-oracle.safetensors", import.meta.url),
  );
  const file = parseSafetensors(bytes.buffer);
  const scales: unknown = JSON.parse(file.metadata.get("scale_bits") ?? "null");
  assert(
    Array.isArray(scales) &&
      scales.every((s: unknown) => typeof s === "number" && Number.isInteger(s)),
  );
  const read = (name: string): Float32Array<ArrayBuffer> => {
    const tensor = file.tensors.get(name);
    assert(tensor?.dtype === "F32");
    return new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4);
  };
  return scales.map((scale: number, i: number) => ({
    scale: new Float32Array(Uint32Array.of(scale).buffer)[0],
    x: read(`x${i}`),
    y: bits(read(`y${i}`)),
  }));
};
