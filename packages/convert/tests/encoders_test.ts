// The encoders against the exporter's own bytes. `fixtures/encoders.json` holds seeded tensors and
// what the exporter's Python stores for them (`fixtures/generate_encoders.py`: `fake_quant_int8`
// and `_convert_for_storage` of `tools/exporter`): int8-sym on both row axes with rows of all
// zeros, rounding ties, the smallest scale and ±0; bf16 from every source dtype with ±0, f32
// subnormals and ±inf, and the values the exporter refuses; f32 from every source dtype. No GPU,
// no local mirror.

import { describe, it } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { ConvertError } from "../mod.ts";
import { encodeValues } from "../src/encode.ts";
import { type TensorDtype, tensorValues } from "../src/tensors.ts";
import fixtures from "./fixtures/encoders.json" with { type: "json" };

const hexBytes = (hex: string): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length: hex.length / 2 }, (_, i) => parseInt(hex.slice(2 * i, 2 * i + 2), 16));

const asHex = (bytes: Uint8Array<ArrayBuffer> | undefined): string =>
  Array.from(bytes ?? new Uint8Array(new ArrayBuffer(0)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");

const DTYPES: readonly TensorDtype[] = ["F32", "F16", "BF16"];
const dtypeOf = (name: string): TensorDtype => {
  const dtype = DTYPES.find((candidate) => candidate === name);
  if (dtype === undefined) throw new Error(`fixture: unknown source dtype ${name}`);
  return dtype;
};

const caseNamed = (name: string): (typeof fixtures.cases)[number] => {
  const found = fixtures.cases.find((entry) => entry.name === name);
  if (found === undefined) throw new Error(`fixture: no case ${name}`);
  return found;
};

const encodeCase = (name: string): { readonly data: string; readonly scale: string } => {
  const entry = caseNamed(name);
  const encoded = encodeValues(
    entry.codec === "int8-sym" ? "int8-sym" : entry.codec === "bf16" ? "bf16" : "f32",
    tensorValues(dtypeOf(entry.sourceDtype), hexBytes(entry.source)),
    entry.shape,
    entry.rowAxis === 1 ? 1 : 0,
    `fixture '${name}'`,
  );
  return { data: asHex(encoded.data), scale: asHex(encoded.scale) };
};

const expectedOf = (name: string): { readonly data: string; readonly scale: string } => {
  const entry = caseNamed(name);
  return { data: entry.data, scale: entry.scale };
};

describe("int8-sym", () => {
  it("gives the exporter's codes and scales on row axis 0, zero rows, ties and the smallest scale included", () => {
    assertEquals(encodeCase("int8-rows"), expectedOf("int8-rows"));
  });

  it("gives the exporter's codes and scales on row axis 1 (the channel of a transposed layout)", () => {
    assertEquals(encodeCase("int8-axis1"), expectedOf("int8-axis1"));
  });

  it("reads f16 sources exactly", () => {
    assertEquals(encodeCase("int8-small"), expectedOf("int8-small"));
  });

  it("refuses a non-finite value, naming the element", () => {
    const values = new Float32Array([1, 2, Number.NaN, 4]);
    assertThrows(
      () => encodeValues("int8-sym", values, [2, 2], 0, "graph 'g' initializer 'w'"),
      ConvertError,
      "graph 'g' initializer 'w': element 2 is NaN",
    );
  });
});

describe("bf16", () => {
  for (const name of ["bf16-from-f32", "bf16-from-bf16", "bf16-from-f16"]) {
    it(`stores the exporter's bytes (${name}, ±0, subnormals and ±inf included)`, () => {
      assertEquals(encodeCase(name), expectedOf(name));
    });
  }

  for (const name of ["bf16-inexact", "bf16-nan"]) {
    it(`refuses what the exporter refuses, naming the element (${name})`, () => {
      const entry = caseNamed(name);
      assertEquals(entry.data, "");
      assertThrows(() => encodeCase(name), ConvertError, `element ${entry.refusedElement} `);
    });
  }
});

describe("f32", () => {
  for (const name of ["f32-from-bf16", "f32-from-f16", "f32-from-f32"]) {
    it(`stores the source values as f32 (${name})`, () => {
      assertEquals(encodeCase(name), expectedOf(name));
    });
  }
});
