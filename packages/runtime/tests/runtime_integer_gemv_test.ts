/**
 * The integer GEMV (`SessionOptions.integerGemv`) without a GPU: the kernel's key / WGSL / params
 * contract and eligibility, its CPU mirror against an independent computation, the option check,
 * and how the fusion plan pairs an integer GEMV linear with packed int8 activations. The kernel's
 * GPU identity with the mirror is gpu_linear_gemv_integer_test.ts.
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  INTEGER_GEMV_EXACT_LIMIT,
  integerGemvKey,
  type IntegerGemvLanes,
  integerGemvLaneBound,
  integerGemvParams,
  type IntegerGemvRows,
  type IntegerGemvVariant,
  integerGemvVariantFor,
  type IntegerGemvWeight,
  integerGemvWgsl,
} from "../src/kernels/linear-gemv-integer.ts";
import {
  linearGemvStaticQuantizeKey,
  linearGemvStaticQuantizePackedKey,
} from "../src/kernels/linear-gemv.ts";
import { staticQuantizeParams } from "../src/kernels/static-quantize.ts";
import { referenceIntegerGemv } from "../src/reference/linear-gemv-integer.ts";
import { referenceStaticQuantize } from "../src/reference/ops.ts";
import {
  type ExecStep,
  FUSION_RULES,
  type FusionPlan,
  type FusionWeightLayout,
  planFusions,
} from "../src/runtime/fusion.ts";
import { linearIntegerGemvVariant } from "../src/runtime/linear-route.ts";
import { countUses, type NodePlan, planGraph } from "../src/runtime/plan.ts";
import { sessionOptionsViolation } from "../src/runtime/session-options-check.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import { mergeGraph, type StorageMap, type StorageSpec } from "./helpers/merged-graph.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";

type Geometry = { readonly lanes: IntegerGemvLanes; readonly rows: IntegerGemvRows };

/** Every (lanes, rows) the generator accepts: rows ≤ lanes (one writer per row). */
const GEOMETRIES: readonly Geometry[] = ([4, 8, 16, 32] as const).flatMap((lanes) =>
  ([1, 2, 4, 8] as const).filter((rows) => rows <= lanes).map((rows) => ({ lanes, rows })),
);

const I2_WEIGHT = { storage: "i2" } as const satisfies IntegerGemvWeight;
const I8_WEIGHT = { storage: "i8" } as const satisfies IntegerGemvWeight;
const i4 = (groupSize: number, groups: number): IntegerGemvWeight => ({
  storage: "i4",
  groupSize,
  groups,
});

/** One form per storage, and i4 at one and three groups and two group lengths. */
const WEIGHTS: readonly IntegerGemvWeight[] = [
  I2_WEIGHT,
  I8_WEIGHT,
  i4(16, 1),
  i4(16, 3),
  i4(2048, 3),
];

const VARIANTS: readonly IntegerGemvVariant[] = WEIGHTS.flatMap((weight) =>
  GEOMETRIES.map((geometry) => ({ weight, ...geometry })),
);

/** The two measured i2 shapes (E2B QAT, the KV-shared layers' MLP): gate / up and down. */
const GATE_UP = {
  n: 12288,
  k: 1536,
  variant: { weight: I2_WEIGHT, lanes: 8, rows: 4 } satisfies IntegerGemvVariant,
} as const;
const DOWN = {
  n: 1536,
  k: 12288,
  variant: { weight: I2_WEIGHT, lanes: 16, rows: 4 } satisfies IntegerGemvVariant,
} as const;

type I4Shape = {
  readonly name: string;
  readonly n: number;
  readonly k: number;
  readonly groupSize: number;
  readonly variant: IntegerGemvVariant;
};

/** The three measured i4 shapes (E2B QAT): o (sliding / full) and the own-KV layers' down. */
const I4_SHAPES: readonly I4Shape[] = [
  {
    name: "o 1536×2048",
    n: 1536,
    k: 2048,
    groupSize: 2048,
    variant: { weight: i4(2048, 1), lanes: 16, rows: 4 },
  },
  {
    name: "o 1536×4096",
    n: 1536,
    k: 4096,
    groupSize: 4096,
    variant: { weight: i4(4096, 1), lanes: 16, rows: 2 },
  },
  {
    name: "down 1536×6144",
    n: 1536,
    k: 6144,
    groupSize: 2048,
    variant: { weight: i4(2048, 3), lanes: 8, rows: 2 },
  },
];

/** A small valid variant for the params checks that do not depend on the form. */
const SMALL: IntegerGemvVariant = { weight: I2_WEIGHT, lanes: 4, rows: 1 };

const X_SCALE = Math.fround(0.00071);
const OUTPUT_SCALE = Math.fround(0.0013);

const f32Bits = (value: number): number => new Uint32Array(Float32Array.of(value).buffer)[0];

/** A variant with values outside the type (what a JS caller can pass). */
const looseVariant = (lanes: number, rows: number): IntegerGemvVariant => {
  const variant: { weight: IntegerGemvWeight; lanes: IntegerGemvLanes; rows: IntegerGemvRows } = {
    weight: I2_WEIGHT,
    lanes: 4,
    rows: 1,
  };
  Reflect.set(variant, "lanes", lanes);
  Reflect.set(variant, "rows", rows);
  return variant;
};

describe("integer GEMV kernel: keys and WGSL", () => {
  it("gives byte-identical WGSL for the same variant and epilogue", () => {
    for (const variant of VARIANTS) {
      for (const quantize of [false, true]) {
        const copy = { ...variant, weight: { ...variant.weight } };
        assertEquals(integerGemvWgsl(variant, quantize), integerGemvWgsl(copy, quantize));
        assertEquals(integerGemvKey(variant, quantize), integerGemvKey(copy, quantize));
      }
    }
  });

  it("gives every weight form × variant × epilogue its own key and its own WGSL", () => {
    const keys = new Set<string>();
    const sources = new Set<string>();
    for (const variant of VARIANTS) {
      for (const quantize of [false, true]) {
        keys.add(integerGemvKey(variant, quantize));
        sources.add(integerGemvWgsl(variant, quantize));
      }
    }
    assertEquals(GEOMETRIES.length, 15);
    assertEquals(VARIANTS.length, 75);
    assertEquals(keys.size, 150);
    assertEquals(sources.size, 150);
    for (const key of keys) assert(key.startsWith("linear_gemv_integer:"), key);
  });

  it("names the weight form, the geometry and the epilogue in the key", () => {
    assertEquals(
      integerGemvKey({ weight: I2_WEIGHT, lanes: 8, rows: 4 }, false),
      "linear_gemv_integer:v2:wi2:l8r4",
    );
    assertEquals(
      integerGemvKey({ weight: I2_WEIGHT, lanes: 16, rows: 4 }, true),
      "linear_gemv_integer:v2:wi2:l16r4:static-quantize:v2",
    );
    assertEquals(
      integerGemvKey({ weight: i4(2048, 3), lanes: 8, rows: 2 }, true),
      "linear_gemv_integer:v2:wi4g2048x3:l8r2:static-quantize:v2",
    );
    assertEquals(
      integerGemvKey({ weight: i4(4096, 1), lanes: 16, rows: 2 }, false),
      "linear_gemv_integer:v2:wi4g4096x1:l16r2",
    );
    assertEquals(
      integerGemvKey({ weight: I8_WEIGHT, lanes: 32, rows: 8 }, false),
      "linear_gemv_integer:v2:wi8:l32r8",
    );
    assertEquals(
      integerGemvKey({ weight: I8_WEIGHT, lanes: 4, rows: 1 }, true),
      "linear_gemv_integer:v2:wi8:l4r1:static-quantize:v2",
    );
  });

  it("refuses a variant outside the generated table", () => {
    for (const variant of [looseVariant(3, 1), looseVariant(64, 1), looseVariant(8, 16)]) {
      assertThrows(() => integerGemvKey(variant, false), CodegenError, "invalid variant");
      assertThrows(() => integerGemvWgsl(variant, true), CodegenError, "invalid variant");
    }
    assertThrows(
      () => integerGemvKey({ weight: I2_WEIGHT, lanes: 4, rows: 8 }, false),
      CodegenError,
      "rows must not exceed lanes",
    );
  });

  it("refuses i4 groups outside 1..4 and a group length that is not a power of two of at least 16", () => {
    for (const weight of [i4(16, 0), i4(16, 5), i4(16, 1.5)]) {
      const variant: IntegerGemvVariant = { weight, lanes: 4, rows: 1 };
      assertThrows(() => integerGemvKey(variant, false), CodegenError, "is not in 1..4");
      assertThrows(() => integerGemvWgsl(variant, true), CodegenError, "is not in 1..4");
      assertThrows(
        () => integerGemvParams(variant, 1, 16, 16, X_SCALE),
        CodegenError,
        "is not in 1..4",
      );
    }
    for (const weight of [i4(8, 2), i4(24, 2), i4(0, 1)]) {
      const variant: IntegerGemvVariant = { weight, lanes: 4, rows: 1 };
      assertThrows(() => integerGemvKey(variant, false), CodegenError, "is not a power of two");
      assertThrows(() => integerGemvWgsl(variant, false), CodegenError, "is not a power of two");
    }
  });
});

describe("integer GEMV kernel: params", () => {
  it("lays out m, n, k, a zero rounding mask and the activation scale's bits", () => {
    const params = integerGemvParams(DOWN.variant, 2, DOWN.n, DOWN.k, X_SCALE);
    assertEquals([...params], [2, DOWN.n, DOWN.k, 0, f32Bits(X_SCALE), 0, 0, 0]);
  });

  it("appends the output-side SRQ table at word 8 with the epilogue", () => {
    const params = integerGemvParams(
      GATE_UP.variant,
      3,
      GATE_UP.n,
      GATE_UP.k,
      X_SCALE,
      OUTPUT_SCALE,
    );
    assertEquals(params.length, 268);
    assertEquals(
      [...params.subarray(0, 8)],
      [3, GATE_UP.n, GATE_UP.k, 0, f32Bits(X_SCALE), 0, 0, 0],
    );
    assertEquals([...params.subarray(8)], [...staticQuantizeParams(3 * GATE_UP.n, OUTPUT_SCALE)]);
  });

  it("refuses M outside 1..8 and a K that is not a positive multiple of 16", () => {
    for (const m of [0, 9, 1.5]) {
      assertThrows(
        () => integerGemvParams(SMALL, m, 16, 16, X_SCALE),
        CodegenError,
        "m must be in 1..8",
      );
    }
    for (const k of [0, 24, -16]) {
      assertThrows(
        () => integerGemvParams(SMALL, 1, 16, k, X_SCALE),
        CodegenError,
        "k must be a positive multiple of 16",
      );
    }
  });

  it("refuses the first K whose lane sums can pass 2^24, and accepts the last one that cannot", () => {
    // A lane holds ceil(K / 16 / lanes) steps of 16 products of at most |factor| · 128 (i2 factor
    // 3, i8 128): i2 at 4 lanes 2^24 / (16 · 3 · 128) = 2730 steps → K = 2730 · 64; i8 64 steps.
    for (const [variant, lastExact] of [
      [{ weight: I2_WEIGHT, lanes: 4, rows: 1 }, 174720],
      [{ weight: I8_WEIGHT, lanes: 4, rows: 1 }, 4096],
      [{ weight: I8_WEIGHT, lanes: 32, rows: 8 }, 32768],
    ] satisfies readonly (readonly [IntegerGemvVariant, number])[]) {
      const where = `${variant.weight.storage} l${variant.lanes}`;
      assert(integerGemvLaneBound(variant, lastExact) <= INTEGER_GEMV_EXACT_LIMIT, where);
      assert(integerGemvLaneBound(variant, lastExact + 16) > INTEGER_GEMV_EXACT_LIMIT, where);
      assertEquals(integerGemvParams(variant, 1, 16, lastExact, X_SCALE)[2], lastExact, where);
      assertThrows(
        () => integerGemvParams(variant, 1, 16, lastExact + 16, X_SCALE),
        CodegenError,
        "pass 2^24",
        where,
      );
    }
  });

  it("bounds an i4 lane sum per group, not per row", () => {
    // i4 factor 15: at 4 lanes a group of 32768 holds 512 steps per lane (15 · 2^20 ≤ 2^24), one
    // of 65536 holds 1024 (past it) — and 4 groups of 32768 stay within it.
    const within = { weight: i4(32768, 4), lanes: 4, rows: 1 } satisfies IntegerGemvVariant;
    assertEquals(integerGemvLaneBound(within, 131072), 15 * 2 ** 20);
    assertEquals(integerGemvParams(within, 1, 16, 131072, X_SCALE)[2], 131072);
    assertThrows(
      () => integerGemvParams({ weight: i4(65536, 1), lanes: 4, rows: 1 }, 1, 16, 65536, X_SCALE),
      CodegenError,
      "pass 2^24",
    );
    // Twice the lanes halve a lane's share of the group.
    const wider = { weight: i4(65536, 1), lanes: 8, rows: 1 } satisfies IntegerGemvVariant;
    assertEquals(integerGemvParams(wider, 1, 16, 65536, X_SCALE)[2], 65536);
  });

  it("refuses an i4 K that is not its groups × group length, and accepts the one that is", () => {
    const [, , down] = I4_SHAPES;
    assertEquals(
      [...integerGemvParams(down.variant, 1, down.n, 6144, X_SCALE)],
      [1, down.n, 6144, 0, f32Bits(X_SCALE), 0, 0, 0],
    );
    for (const k of [2048, 4096, 6160, 8192]) {
      assertThrows(
        () => integerGemvParams(down.variant, 1, down.n, k, X_SCALE),
        CodegenError,
        "is not 3 groups of 2048",
        `k=${k}`,
      );
    }
  });

  it("refuses an activation scale that is not positive, finite and exact in f32", () => {
    for (const scale of [0, -X_SCALE, Number.NaN, Number.POSITIVE_INFINITY, 0.1]) {
      assertThrows(
        () => integerGemvParams(SMALL, 1, 16, 16, scale),
        CodegenError,
        "activation scale",
      );
    }
  });
});

describe("integer GEMV kernel: eligibility (integerGemvVariantFor)", () => {
  it("names the measured variant of each measured i2 shape at M = 1..8", () => {
    for (let m = 1; m <= 8; m++) {
      assertEquals(
        integerGemvVariantFor("i2", undefined, m, GATE_UP.n, GATE_UP.k),
        GATE_UP.variant,
      );
      assertEquals(integerGemvVariantFor("i2", undefined, m, DOWN.n, DOWN.k), DOWN.variant);
    }
  });

  it("names the measured variant of each measured i4 shape, groups = K / group length, at M = 1..8", () => {
    for (const { name, n, k, groupSize, variant } of I4_SHAPES) {
      for (let m = 1; m <= 8; m++) {
        assertEquals(integerGemvVariantFor("i4", groupSize, m, n, k), variant, `${name} M=${m}`);
      }
    }
    assertEquals(integerGemvVariantFor("i4", 2048, 1, 1536, 6144)?.weight, i4(2048, 3));
  });

  it("names nothing for a measured i4 shape with another group length or none", () => {
    for (const { name, n, k, groupSize } of I4_SHAPES) {
      for (const other of [groupSize / 2, groupSize * 2, 32, undefined]) {
        assertEquals(integerGemvVariantFor("i4", other, 1, n, k), undefined, `${name} g${other}`);
      }
    }
  });

  it("names nothing for an i4 linear of another n or k, nor for the K = 1536 i4 shapes", () => {
    assertEquals(integerGemvVariantFor("i4", 2048, 1, 2048, 2048), undefined, "n");
    assertEquals(integerGemvVariantFor("i4", 2048, 1, 1536, 8192), undefined, "4 groups");
    assertEquals(integerGemvVariantFor("i4", 4096, 1, 1536, 8192), undefined, "2 groups");
    // q / k / v / gate / up measured slower or equal on the integer kernel.
    for (const n of [2048, 4096, 256, 512, 6144]) {
      assertEquals(integerGemvVariantFor("i4", 512, 1, n, 1536), undefined, `${n}×1536`);
    }
    for (const m of [0, 9]) {
      assertEquals(integerGemvVariantFor("i4", 2048, m, 1536, 2048), undefined, `M=${m}`);
    }
  });

  it("names nothing for i8, at the PLE shapes or at the measured i2 / i4 shapes", () => {
    for (const [n, k] of [
      [256, 1536],
      [1536, 256],
      [GATE_UP.n, GATE_UP.k],
      [DOWN.n, DOWN.k],
      ...I4_SHAPES.map(({ n, k }) => [n, k]),
    ]) {
      assertEquals(integerGemvVariantFor("i8", undefined, 1, n, k), undefined, `${n}×${k}`);
    }
  });

  it("names nothing for another storage, M outside 1..8 or an unmeasured shape", () => {
    for (const storage of ["i4", "i8", "f16", "bf16", "f32"] as const) {
      assertEquals(
        integerGemvVariantFor(storage, undefined, 1, DOWN.n, DOWN.k),
        undefined,
        storage,
      );
    }
    for (const m of [0, 9, 64]) {
      assertEquals(integerGemvVariantFor("i2", undefined, m, DOWN.n, DOWN.k), undefined, `M=${m}`);
    }
    assertEquals(integerGemvVariantFor("i2", undefined, 1, 1536, 1536), undefined);
    assertEquals(integerGemvVariantFor("i2", undefined, 1, 12288, 12288), undefined);
    assertEquals(
      integerGemvVariantFor("i2", undefined, 1, 12288, 1544),
      undefined,
      "k not a multiple of 16",
    );
    assertEquals(integerGemvVariantFor("i2", 32, 1, DOWN.n, DOWN.k), undefined, "i2 with a group");
  });
});

/** Four int8 codes per word, low byte first (written apart from the mirror's own reader). */
const packCodes = (codes: Int8Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const words = new Uint32Array(codes.length / 4);
  codes.forEach((code, index) => {
    words[Math.floor(index / 4)] += (code & 0xff) * 2 ** (8 * (index % 4));
  });
  return words;
};

/** 16 i2 values per word, two bits each, stored as value + 2, low bits first. */
const packI2 = (values: Int8Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const words = new Uint32Array(values.length / 16);
  values.forEach((value, index) => {
    words[Math.floor(index / 16)] += (value + 2) * 2 ** (2 * (index % 16));
  });
  return words;
};

/** 8 i4 values per word, four bits each, stored as value + 8, low bits first. */
const packI4 = (values: Int8Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const words = new Uint32Array(values.length / 8);
  values.forEach((value, index) => {
    words[Math.floor(index / 8)] += (value + 8) * 2 ** (4 * (index % 8));
  });
  return words;
};

/** 4 i8 values per word, one two's-complement byte each, low byte first. */
const packI8 = (values: Int8Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => packCodes(values);

/**
 * The output words computed straight from the unpacked codes: per group the integer dot product,
 * then the f32 roundings of the kernel's epilogue in order — each group's `S · w_scale`, their sum
 * in ascending group order, the `x_scale` multiply and the bias add.
 */
const straightforward = (input: {
  m: number;
  n: number;
  k: number;
  groups: number;
  codes: Int8Array<ArrayBuffer>;
  values: Int8Array<ArrayBuffer>;
  wscale: Float32Array<ArrayBuffer>;
  bias: Float32Array<ArrayBuffer>;
}): Uint32Array<ArrayBuffer> => {
  const { m, n, k, groups, codes, values, wscale, bias } = input;
  const length = k / groups;
  const out = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let total = 0;
      for (let g = 0; g < groups; g++) {
        let sum = 0;
        for (let e = g * length; e < (g + 1) * length; e++) {
          sum += codes[row * k + e] * values[col * k + e];
        }
        total = Math.fround(total + Math.fround(Math.fround(sum) * wscale[col * groups + g]));
      }
      out[row * n + col] = Math.fround(Math.fround(total * X_SCALE) + bias[col]);
    }
  }
  return new Uint32Array(out.buffer);
};

describe("referenceIntegerGemv (the CPU mirror)", () => {
  const m = 3,
    n = 5,
    k = 48;
  const codes = Int8Array.from({ length: m * k }, (_, i) => ((i * 37 + 11) % 256) - 128);
  const values = Int8Array.from({ length: n * k }, (_, i) => ((i * 7 + 3) % 4) - 2);
  const wscale = Float32Array.from({ length: n }, (_, i) => 0.0021 * (i + 1));
  const bias = Float32Array.from({ length: n }, (_, i) => (i - 2) * 0.37);
  const mirrorInput = {
    weight: I2_WEIGHT,
    m,
    n,
    k,
    codes: packCodes(codes),
    xScale: X_SCALE,
    weights: packI2(values),
    wscale,
    bias,
  };

  it("equals the straightforward integer dot product and epilogue for i2, word for word", () => {
    assertEquals(
      referenceIntegerGemv(mirrorInput),
      straightforward({ m, n, k, groups: 1, codes, values, wscale, bias }),
    );
  });

  it("equals the straightforward per-group sums and epilogue for i4, word for word", () => {
    const groups = 3,
      wide = 96;
    const wideCodes = Int8Array.from({ length: m * wide }, (_, i) => ((i * 37 + 11) % 256) - 128);
    const i4Values = Int8Array.from({ length: n * wide }, (_, i) => ((i * 5 + 3) % 16) - 8);
    const groupScales = Float32Array.from({ length: n * groups }, (_, i) => 0.0013 * (i + 1));
    assertEquals(
      referenceIntegerGemv({
        weight: i4(wide / groups, groups),
        m,
        n,
        k: wide,
        codes: packCodes(wideCodes),
        xScale: X_SCALE,
        weights: packI4(i4Values),
        wscale: groupScales,
        bias,
      }),
      straightforward({
        m,
        n,
        k: wide,
        groups,
        codes: wideCodes,
        values: i4Values,
        wscale: groupScales,
        bias,
      }),
    );
  });

  it("equals the straightforward integer dot product and epilogue for i8, word for word", () => {
    const i8Values = Int8Array.from({ length: n * k }, (_, i) => ((i * 53 + 7) % 256) - 128);
    assertEquals(
      referenceIntegerGemv({ ...mirrorInput, weight: I8_WEIGHT, weights: packI8(i8Values) }),
      straightforward({ m, n, k, groups: 1, codes, values: i8Values, wscale, bias }),
    );
  });

  it("stays exact on the saturated input at each storage's largest measured K", () => {
    // Every code −128 against the largest-magnitude value; i8's sum (1536 · 2^14) is past 2^24.
    for (const [weight, large, value, pack] of [
      [I2_WEIGHT, 12288, -2, packI2],
      [i4(2048, 3), 6144, -8, packI4],
      [I8_WEIGHT, 1536, -128, packI8],
    ] as const) {
      const groups = weight.storage === "i4" ? weight.groups : 1;
      const saturatedCodes = new Int8Array(large).fill(-128);
      const saturatedValues = new Int8Array(2 * large).fill(value);
      const scales = Float32Array.from({ length: 2 * groups }, (_, i) => 0.5 / (i + 1));
      const zero = new Float32Array(2);
      const words = referenceIntegerGemv({
        weight,
        m: 1,
        n: 2,
        k: large,
        codes: packCodes(saturatedCodes),
        xScale: X_SCALE,
        weights: pack(saturatedValues),
        wscale: scales,
        bias: zero,
      });
      assertEquals(
        words,
        straightforward({
          m: 1,
          n: 2,
          k: large,
          groups,
          codes: saturatedCodes,
          values: saturatedValues,
          wscale: scales,
          bias: zero,
        }),
        weight.storage,
      );
    }
  });

  it("adds the i4 groups in ascending order, rounding each term and each partial sum", () => {
    // K = 48 in 3 groups of 16. Channel 0: group sums (16384, 1, 1) with scales (1024, 1, 1) —
    // terms (2^24, 1, 1): ascending, each + 1 rounds back to 2^24 (ties to even); descending would
    // give 2^24 + 2. Channel 1: sums (−14336, −8, 1) with scales (1, 0.5, 0.25).
    const codes = Int8Array.from({ length: 48 }, (_, e) =>
      e < 16 ? -128 : e === 16 || e === 32 ? 1 : 77,
    );
    const values = Int8Array.from({ length: 96 }, (_, i) => {
      const [col, e] = [Math.floor(i / 48), i % 48];
      if (e < 16) return col === 0 ? -8 : 7;
      if (e === 16) return col === 0 ? 1 : -8;
      return e === 32 ? 1 : 0;
    });
    const wscale = Float32Array.of(1024, 1, 1, 1, 0.5, 0.25);
    const xScale = 2 ** -10;
    const bias = Float32Array.of(0, 1.5);
    const f = Math.fround;
    const channel = (sums: readonly number[], scales: readonly number[], add: number): number => {
      const terms = sums.map((sum, g) => f(sum * scales[g]));
      return f(f(f(f(terms[0] + terms[1]) + terms[2]) * xScale) + add);
    };
    const expected = [
      channel([16384, 1, 1], [1024, 1, 1], 0),
      channel([-14336, -8, 1], [1, 0.5, 0.25], 1.5),
    ];
    assertEquals(expected, [16384, -12.503662109375]);
    // The input tells the orders apart: descending would land 2 above 2^24 before the x scale.
    assertEquals(f(f(f(1 + 1) + 2 ** 24) * xScale), 16384.001953125);
    assertEquals(
      referenceIntegerGemv({
        weight: i4(16, 3),
        m: 1,
        n: 2,
        k: 48,
        codes: packCodes(codes),
        xScale,
        weights: packI4(values),
        wscale,
        bias,
      }),
      Uint32Array.from(expected, f32Bits),
    );
  });

  it("reads i8 weights as two's complement bytes (0x80 = −128, 0xff = −1)", () => {
    // K = 16. Channel 0: (−128)(−128) + (−1)(127) + (127)(−1) + (1)(1) = 16131; channel 1:
    // (127)(−128) + (−128)(127) + (−1)(−1) + (−1)(1) = −32512. The other codes meet zero weights.
    const codes = Int8Array.from({ length: 16 }, (_, e) => (e < 4 ? [-128, 127, -1, 1][e] : 9));
    const values = Int8Array.from({ length: 32 }, (_, i) =>
      i % 16 < 4 ? (i < 16 ? [-128, -1, 127, 1] : [127, -128, -1, -1])[i % 16] : 0,
    );
    const xScale = 2 ** -6;
    const f = Math.fround;
    const expected = [f(f(f(16131 * 0.25) * xScale) + 0.5), f(f(f(-32512 * 0.5) * xScale) - 0.25)];
    assertEquals(expected, [63.51171875, -254.25]);
    assertEquals(
      referenceIntegerGemv({
        weight: I8_WEIGHT,
        m: 1,
        n: 2,
        k: 16,
        codes: packCodes(codes),
        xScale,
        weights: packI8(values),
        wscale: Float32Array.of(0.25, 0.5),
        bias: Float32Array.of(0.5, -0.25),
      }),
      Uint32Array.from(expected, f32Bits),
    );
  });

  it("rounds a sum past 2^24 once, from the exact integer, at any K", () => {
    // i8 K = 1040: 1039 · (−128)(−128) + (−127)(−127) = 17039105, odd past 2^24 → f32 ties to
    // even: 17039104. The kernel bounds K per variant; the mirror has no bound to refuse at.
    const codes = new Int8Array(1040).fill(-128);
    codes[0] = -127;
    const values = codes.slice();
    assertEquals(
      referenceIntegerGemv({
        weight: I8_WEIGHT,
        m: 1,
        n: 1,
        k: 1040,
        codes: packCodes(codes),
        xScale: 1,
        weights: packI8(values),
        wscale: Float32Array.of(1),
        bias: Float32Array.of(0),
      }),
      Uint32Array.of(f32Bits(17039104)),
    );
  });

  it("applies the SRQ epilogue to the plain output", () => {
    const plain = new Float32Array(referenceIntegerGemv(mirrorInput).buffer);
    const quantized = referenceStaticQuantize(
      { dtype: "f32", shape: [m, n], data: plain },
      { scale: OUTPUT_SCALE },
    ).data;
    assertEquals(
      referenceIntegerGemv({ ...mirrorInput, outputScale: OUTPUT_SCALE }),
      new Uint32Array(Float32Array.from(quantized).buffer),
    );
  });
});

describe("integerGemv option check", () => {
  const packed = { linearGemvReduce: "parallel", packedStaticQuantize: true } as const;

  it("accepts integerGemv on top of the packed activations, and false anywhere", () => {
    assertEquals(sessionOptionsViolation({ ...packed, integerGemv: true }), undefined);
    assertEquals(
      sessionOptionsViolation({ ...packed, fuseLinearStaticQuantize: true, integerGemv: true }),
      undefined,
    );
    assertEquals(sessionOptionsViolation({ integerGemv: false }), undefined);
  });

  it("rejects integerGemv without the packed activations it reads", () => {
    const requires =
      "integerGemv requires packedStaticQuantize: true (with linearGemvReduce: parallel / linearCompute: f32)";
    assertEquals(sessionOptionsViolation({ integerGemv: true }), requires);
    assertEquals(
      sessionOptionsViolation({ linearGemvReduce: "parallel", integerGemv: true }),
      requires,
    );
    assertEquals(
      sessionOptionsViolation({ ...packed, packedStaticQuantize: false, integerGemv: true }),
      requires,
    );
    // The packed activations' own requirement is named first when it is the one broken.
    assertEquals(
      sessionOptionsViolation({ packedStaticQuantize: true, integerGemv: true }),
      "packedStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
    );
  });

  it("rejects a value that is not a boolean, without converting it", () => {
    let conversions = 0;
    const object = {
      [Symbol.toPrimitive](): boolean {
        conversions++;
        return true;
      },
    };
    for (const value of [null, 1, "true", [], object]) {
      const options: SessionOptions = { ...packed };
      Object.defineProperty(options, "integerGemv", { value, enumerable: true });
      assertEquals(sessionOptionsViolation(options), "options.integerGemv must be a boolean");
    }
    assertEquals(conversions, 0);
  });
});

// ---------------------------------------------------------------------------
// The plan: the eligibility function and the packed pairing
// ---------------------------------------------------------------------------

type Consumer = {
  readonly n: number;
  readonly storage: StorageSpec;
};

type GraphOptions = {
  readonly k: number;
  readonly m?: number;
  /** The linears that read the SRQ's output (gate and up share one SRQ in the product graph). */
  readonly consumers: readonly Consumer[];
  /** An output-side SRQ after the first linear (the linear→SRQ fusion's window). */
  readonly outputQuantize?: boolean;
};

const I2: StorageSpec = { codec: "int2-off" };
const I8: StorageSpec = { codec: "int8-sym" };

/** `src → static_quantize → linear × consumers` (+ an output-side SRQ on the first linear). */
const integerGraph = ({
  k,
  m = 1,
  consumers,
  outputQuantize = false,
}: GraphOptions): DeclarationJson => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear", "static_quantize"] },
    symbols: [],
    inputs: [{ name: "src", dtype: "f32", shape: [m, k] }],
    outputs: [],
    initializers: {},
    values: { xq: { dtype: "f32", shape: [m, k] } },
    nodes: [{ op: "static_quantize", ins: ["src"], outs: ["xq"], attrs: { scale: X_SCALE } }],
  };
  consumers.forEach(({ n }, index) => {
    graph.initializers[`w${index}`] = {};
    graph.initializers[`b${index}`] = {};
    graph.values[`w${index}`] = { dtype: "f32", shape: [n, k] };
    graph.values[`b${index}`] = { dtype: "f32", shape: [n] };
    graph.values[`y${index}`] = { dtype: "f32", shape: [m, n] };
    graph.nodes.push({
      op: "linear",
      ins: ["xq", `w${index}`, `b${index}`],
      outs: [`y${index}`],
      attrs: {},
    });
    if (index === 0 && outputQuantize) {
      graph.values.z = { dtype: "f32", shape: [m, n] };
      graph.nodes.push({
        op: "static_quantize",
        ins: ["y0"],
        outs: ["z"],
        attrs: { scale: OUTPUT_SCALE },
      });
      graph.outputs.push("z");
    } else {
      graph.outputs.push(`y${index}`);
    }
  });
  return graph;
};

const storageOf = (options: GraphOptions): StorageMap =>
  Object.fromEntries(options.consumers.map(({ storage }, index) => [`w${index}`, storage]));

const layoutOf = (storage: StorageSpec): FusionWeightLayout =>
  typeof storage === "object" && storage.codec === "int8-sym"
    ? { storage: "i8" }
    : { storage: "i2" };

type Context = Parameters<typeof planFusions>[1];

/** The parallel GEMV seat with packed activations (what integerGemv requires), plus `context`. */
const planOf = (options: GraphOptions, context: Partial<Context> = {}): FusionPlan => {
  const ir = mergeGraph(integerGraph(options), storageOf(options));
  return planFusions(planGraph(ir, {}).nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    packedStaticQuantize: true,
    linearGemvReduce: "parallel",
    linearCompute: "f32",
    weightLayouts: new Map(
      options.consumers.map(({ storage }, index) => [`w${index}`, layoutOf(storage)]),
    ),
    ...context,
  });
};

/** The packed roles of the plain node steps, by op. */
const roles = (steps: readonly ExecStep[]): readonly string[] =>
  steps.flatMap((step) =>
    step.kind === "node" && step.packedActivations !== undefined
      ? [`${step.plan.node.op}:${step.packedActivations.role}`]
      : [],
  );

/** The first linear's node plan of a graph (the eligibility function's input). */
const linearPlan = (options: GraphOptions): NodePlan => {
  const ir = mergeGraph(integerGraph(options), storageOf(options));
  const found = planGraph(ir, {}).nodes.find((plan) => plan.node.op === "linear");
  if (found === undefined) throw new Error("the graph has no linear");
  return found;
};

/** The whole plan as comparable text (each dispatch's WGSL and params written out). */
const planText = (plan: FusionPlan): string =>
  JSON.stringify(
    {
      counts: plan.counts,
      steps: plan.steps.map((step) =>
        step.kind === "fused"
          ? {
              ...step,
              dispatches: step.dispatches.map((dispatch) => ({
                ...dispatch,
                wgsl: dispatch.wgsl(),
                params: [...dispatch.params],
              })),
            }
          : step,
      ),
    },
    (_key, value: unknown) => (typeof value === "function" ? "[function]" : value),
  );

const gateUp = (overrides: Partial<GraphOptions> = {}): GraphOptions => ({
  k: GATE_UP.k,
  consumers: [{ n: GATE_UP.n, storage: I2 }],
  ...overrides,
});
const down = (overrides: Partial<GraphOptions> = {}): GraphOptions => ({
  k: DOWN.k,
  consumers: [{ n: DOWN.n, storage: I2 }],
  ...overrides,
});

/** One i4 linear of a measured i4 shape, its weight in the i4 codec at the shape's group length. */
const i4Linear = (shape: I4Shape, overrides: Partial<GraphOptions> = {}): GraphOptions => ({
  k: shape.k,
  consumers: [{ n: shape.n, storage: { codec: "int4-sym-g", groupSize: shape.groupSize } }],
  ...overrides,
});

describe("linearIntegerGemvVariant (the one eligibility function)", () => {
  it("names the variant of an i2 linear of a measured shape at M = 1..8 when the option is on", () => {
    assertEquals(
      linearIntegerGemvVariant(true, linearPlan(gateUp()), "i2", undefined),
      GATE_UP.variant,
    );
    assertEquals(
      linearIntegerGemvVariant(true, linearPlan(down({ m: 8 })), "i2", undefined),
      DOWN.variant,
    );
  });

  it("names the variant of an i4 linear of a measured shape with its group length, at M = 1..8", () => {
    for (const shape of I4_SHAPES) {
      for (const m of [1, 8]) {
        assertEquals(
          linearIntegerGemvVariant(true, linearPlan(i4Linear(shape, { m })), "i4", shape.groupSize),
          shape.variant,
          `${shape.name} M=${m}`,
        );
      }
    }
  });

  it("names nothing for an i4 linear with another group length or none", () => {
    for (const shape of I4_SHAPES) {
      const plan = linearPlan(i4Linear(shape));
      for (const other of [shape.groupSize / 2, undefined]) {
        assertEquals(
          linearIntegerGemvVariant(true, plan, "i4", other),
          undefined,
          `${shape.name} g${other}`,
        );
      }
      assertEquals(linearIntegerGemvVariant(false, plan, "i4", shape.groupSize), undefined);
    }
  });

  it("names nothing with the option off, another storage, M > 8 or a node that is not a linear", () => {
    const plan = linearPlan(gateUp());
    assertEquals(linearIntegerGemvVariant(false, plan, "i2", undefined), undefined);
    assertEquals(linearIntegerGemvVariant(undefined, plan, "i2", undefined), undefined);
    for (const storage of ["i4", "i8", "f16", "f32", undefined] as const) {
      assertEquals(
        linearIntegerGemvVariant(true, plan, storage, undefined),
        undefined,
        String(storage),
      );
    }
    assertEquals(
      linearIntegerGemvVariant(true, linearPlan(gateUp({ m: 9 })), "i2", undefined),
      undefined,
    );
    const ple = linearPlan({ k: 1536, consumers: [{ n: 256, storage: I8 }] });
    assertEquals(linearIntegerGemvVariant(true, ple, "i8", undefined), undefined, "i8 PLE");
    const ir = mergeGraph(integerGraph(gateUp()), storageOf(gateUp()));
    const quantize = planGraph(ir, {}).nodes.find((node) => node.node.op === "static_quantize");
    assert(quantize !== undefined);
    assertEquals(linearIntegerGemvVariant(true, quantize, "i2", undefined), undefined);
  });
});

describe("the packed pairing under integerGemv", () => {
  it("packs the SRQ of an i2 gate / up linear only with the option on", () => {
    // 12288×1536 lands on the parallel GEMV, whose table does not pack it (ADR 0105 addendum 1).
    const off = planOf(gateUp());
    assertEquals(off.counts.packedStaticQuantize, 0);
    assertEquals(roles(off.steps), []);
    const on = planOf(gateUp(), { integerGemv: true });
    assertEquals(on.counts.packedStaticQuantize, 1);
    assertEquals(roles(on.steps), ["static_quantize:write", "linear:read"]);
    for (const step of on.steps) {
      if (step.kind === "node" && step.packedActivations !== undefined) {
        assertEquals(
          step.packedActivations.scale,
          X_SCALE,
          "both sides carry the producer's scale",
        );
      }
    }
  });

  it("packs an SRQ read by two integer linears (gate and up), not one also read by another linear", () => {
    const twins = gateUp({
      consumers: [
        { n: GATE_UP.n, storage: I2 },
        { n: GATE_UP.n, storage: I2 },
      ],
    });
    assertEquals(roles(planOf(twins, { integerGemv: true }).steps), [
      "static_quantize:write",
      "linear:read",
      "linear:read",
    ]);
    // i8 256×1536 lands on the parallel GEMV without packing: every consumer must read packed.
    const mixed = gateUp({
      consumers: [
        { n: GATE_UP.n, storage: I2 },
        { n: 256, storage: I8 },
      ],
    });
    assertEquals(planOf(mixed, { integerGemv: true }).counts.packedStaticQuantize, 0);
  });

  it("keeps M > 8 and the other acceptance conditions of the pairing", () => {
    assertEquals(planOf(gateUp({ m: 9 }), { integerGemv: true }).counts.packedStaticQuantize, 0);
    for (const context of [
      { packedStaticQuantize: false },
      { linearGemvReduce: "sequential" },
      { linearCompute: "a8" },
    ] satisfies Partial<Context>[]) {
      const plan = planOf(gateUp(), { ...context, integerGemv: true });
      assertEquals(plan.counts.packedStaticQuantize, 0, JSON.stringify(context));
    }
  });

  it("builds the linear→SRQ window as the integer GEMV with the SRQ epilogue", () => {
    for (const [shape, offKey] of [
      [down, linearGemvStaticQuantizePackedKey("i2", undefined, 32)],
      [gateUp, linearGemvStaticQuantizeKey("i2", undefined, 2)],
    ] as const) {
      const options = shape({ outputQuantize: true });
      const { n, k } = shape === down ? DOWN : GATE_UP;
      const variant = shape === down ? DOWN.variant : GATE_UP.variant;
      const on = planOf(options, { fuseLinearStaticQuantize: true, integerGemv: true });
      const off = planOf(options, { fuseLinearStaticQuantize: true });
      // The same window is grabbed either way; only the dispatch it builds differs.
      assertEquals(on.counts.linearStaticQuantize, 1);
      assertEquals(off.counts.linearStaticQuantize, 1);
      const fused = on.steps.find((step) => step.kind === "fused");
      const reference = off.steps.find((step) => step.kind === "fused");
      if (fused?.kind !== "fused" || reference?.kind !== "fused") throw new Error("not fused");
      const [dispatch] = fused.dispatches;
      assertEquals(dispatch.key, integerGemvKey(variant, true));
      assertEquals(dispatch.wgsl(), integerGemvWgsl(variant, true));
      assertEquals(dispatch.params, integerGemvParams(variant, 1, n, k, X_SCALE, OUTPUT_SCALE));
      // gate / up: 12288 / (16 groups × 4 rows); down: 1536 / (8 groups × 4 rows).
      assertEquals(dispatch.workgroups, {
        kind: "tiled",
        counts: [shape === down ? 48 : 192, 1, 1],
      });
      assertEquals(dispatch.operands, reference.dispatches[0].operands);
      assertEquals(reference.dispatches[0].key, offKey);
    }
  });
});

describe("integerGemv off: the plan does not change", () => {
  const graphs: readonly (readonly [string, GraphOptions])[] = [
    ["gate / up", gateUp()],
    ["down", down()],
    ["down with an output SRQ", down({ outputQuantize: true })],
    ["gate / up with an output SRQ", gateUp({ outputQuantize: true })],
  ];
  const contexts: readonly Partial<Context>[] = [{}, { fuseLinearStaticQuantize: true }];

  it("plans exactly as without the field when the option is false", () => {
    for (const [label, options] of graphs) {
      for (const context of contexts) {
        assertEquals(
          planText(planOf(options, { ...context, integerGemv: false })),
          planText(planOf(options, context)),
          `${label} ${JSON.stringify(context)}`,
        );
      }
    }
  });

  it("plans exactly as with the option off when no linear is an integer linear", () => {
    const ineligible: readonly (readonly [string, GraphOptions])[] = [
      ["M = 9", gateUp({ m: 9 })],
      ["i8 weights", { k: 1536, consumers: [{ n: 256, storage: I8 }], outputQuantize: true }],
      ["an unmeasured i2 shape", { k: 1536, consumers: [{ n: 1536, storage: I2 }] }],
    ];
    for (const [label, options] of ineligible) {
      for (const context of contexts) {
        assertEquals(
          planText(planOf(options, { ...context, integerGemv: true })),
          planText(planOf(options, context)),
          `${label} ${JSON.stringify(context)}`,
        );
      }
    }
  });
});
