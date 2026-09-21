// 公式 Torch CPU の期待ビット列（fixtures/static-quantize-oracle.safetensors）の読み出し。
// CPU 参照側（static_quantize_test.ts）と実 GPU 側（gpu_static_quantize_test.ts）が**同じ
// oracle**に突き合わせるため、読み出しはこの 1 本に置く。
import { assert } from "@std/assert";
import { parseSafetensors } from "../../src/format/safetensors.ts";

export const bits = (data: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

export type Case = { scale: number; x: Float32Array<ArrayBuffer>; y: Uint32Array<ArrayBuffer> };

export const cases = async (): Promise<readonly Case[]> => {
  const bytes = await Deno.readFile(
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
