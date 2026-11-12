// The tiled key axis of the online int8 attention (perf-ledger K-106 — the online route's form where
// the geometry profile chooses it, ADR 0132 addendum 2026-11-12): `AttentionOnlineKeyWords` `tiled`
// (src/kernels/attention-online-i8a8.ts) and its producer, the tiled `quantize_rows`
// (src/kernels/quantize-rows.ts).
//
// - CPU: the keys (the existing ones unmoved), the row words and Vᵀ bytes by hand, the producer's
//   refusals, the host permutation that is the oracle of the GPU parts, and the production producer
//   building the form at any N.
// - GPU: the identity check of @karume/kernel-checks (one implementation with the playground's
//   /kernel-check page): the tiled producer against the host permutation of today's producer's bytes,
//   and the kernel reading the tiled Vᵀ against today's kernel, Uint32 identity of O on the shapes of
//   the four-point check (gpu_attention_online_i8a8_test.ts), both reduction forms, both output
//   layouts, dp4a and the emulated dot. Then the fault injections: a wrong word index in the producer
//   and a wrong lane in the kernel's load each turn their comparison red on every case.

import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import {
  ATTENTION_ONLINE_TILED_CASES,
  type CheckPart,
  checkAttentionOnlineTiled,
  tiledValueWords,
} from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  attentionOnlineI8a8Key,
  attentionOnlineI8a8Params,
  attentionOnlineI8a8Wgsl,
  attentionOnlineKeyWords,
  attentionOnlineTiledValueRowWords,
  attentionOnlineValueRowWords,
} from "../src/kernels/attention-online-i8a8.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsPaddedGeometry,
  quantizeRowsPaddedKey,
  quantizeRowsTiledGeometry,
  quantizeRowsTiledKey,
  quantizeRowsTiledParams,
  quantizeRowsTiledRowWords,
  quantizeRowsTiledWgsl,
} from "../src/kernels/quantize-rows.ts";
import { attentionValueQuantization } from "../src/runtime/recipe-builders/attention.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";

/** The key lengths the record and the scout name: the Wan2.2 cross / 33f / 121f keys, two odd ones. */
const KEY_LENGTHS = [512, 7920, 27280, 203, 65] as const;

describe("the tiled key axis (CPU)", () => {
  test("the existing keys are unmoved and the tiled form adds its own segment", () => {
    // Spelled out: a key that moved would re-key every cached pipeline and break the census
    assertEquals(
      attentionOnlineI8a8Key(true, 128, "sg32"),
      "attention_online:v2:i8a8:d128:r64k64:dp4a:sg32",
    );
    assertEquals(
      attentionOnlineI8a8Key(true, 128, "sg32", "bshd", "padded"),
      "attention_online:v2:i8a8:d128:r64k64:dp4a:sg32:obshd:kpad",
    );
    assertEquals(
      attentionOnlineI8a8Key(false, 64, "wg", "bhsd", "padded"),
      "attention_online:v2:i8a8:d64:r64k64:emu:wg:kpad",
    );
    assertEquals(
      attentionOnlineI8a8Key(true, 128, "sg32", "bhsd", "tiled"),
      "attention_online:v2:i8a8:d128:r64k64:dp4a:sg32:ktile",
    );
    assertEquals(
      attentionOnlineI8a8Key(false, 64, "wg", "bshd", "tiled"),
      "attention_online:v2:i8a8:d64:r64k64:emu:wg:obshd:ktile",
    );
    assertEquals(
      quantizeRowsKey(quantizeRowsGeometry(512)),
      "quantize_rows:v1:f32>i8:pertoken:wg256:r2w128",
    );
    assertEquals(
      quantizeRowsKey(quantizeRowsGeometry(7920)),
      "quantize_rows:v1:f32>i8:pertoken:wg256",
    );
    assertEquals(
      quantizeRowsPaddedKey(quantizeRowsPaddedGeometry(203)),
      "quantize_rows:v1:f32>i8:pertoken:wg256:r4w64:pad",
    );
    // The tiled producer always takes the lined-up generator (a row of whole tiles), so even one
    // row per workgroup carries its geometry segment
    assertEquals(
      KEY_LENGTHS.map((n) => quantizeRowsTiledKey(quantizeRowsTiledGeometry(n))),
      [
        "quantize_rows:v1:f32>i8:pertoken:wg256:r2w128:tile64",
        "quantize_rows:v1:f32>i8:pertoken:wg256:r1w256:tile64",
        "quantize_rows:v1:f32>i8:pertoken:wg256:r1w256:tile64",
        "quantize_rows:v1:f32>i8:pertoken:wg256:r4w64:tile64",
        "quantize_rows:v1:f32>i8:pertoken:wg256:r8w32:tile64",
      ],
    );
  });

  test("the row words and the Vᵀ bytes, by hand", () => {
    // 16 words per 64-key tile: 512 / 64 = 8 tiles, ceil(7,920 / 64) = 124, ceil(27,280 / 64) =
    // 427, ceil(203 / 64) = 4, ceil(65 / 64) = 2
    assertEquals(KEY_LENGTHS.map(attentionOnlineTiledValueRowWords), [128, 1984, 6832, 64, 32]);
    // The producer and the reader agree on the row length
    assertEquals(KEY_LENGTHS.map(quantizeRowsTiledRowWords), [128, 1984, 6832, 64, 32]);
    // Today's row lengths are unchanged: N / 4, or ceil(N / 4) for the padded axis
    assertEquals(KEY_LENGTHS.map(attentionOnlineValueRowWords), [128, 1980, 6820, 51, 17]);
    // Vᵀ bytes per head at D = 128 (words × 4 × D): equal at whole tiles, at most one tile more
    assertEquals(
      KEY_LENGTHS.map((n) => attentionOnlineTiledValueRowWords(n) * 4 * 128),
      [65536, 1015808, 3497984, 32768, 16384],
    );
  });

  test("the kernel takes any key length in the tiled form; N's own form is never tiled (only a profile chooses it)", () => {
    for (const n of KEY_LENGTHS) {
      assertEquals(attentionOnlineI8a8Params(64, n, 0, 0.5, "tiled")[1], n);
      assertNotEquals(attentionOnlineKeyWords(n), "tiled");
    }
    // The tiled kernel reads Vᵀ as vec4 and has no byte gather left
    const wgsl = attentionOnlineI8a8Wgsl(true, 128, "sg32", "bhsd", "tiled");
    assert(wgsl.includes("var<storage, read> vq: array<vec4<u32>>;"));
    assert(!wgsl.includes(">> shift"), "the tiled owner fill does not gather bytes");
    assert(attentionOnlineI8a8Wgsl(true, 128, "sg32").includes(">> shift"));
  });

  test("the tiled producer refuses rows that are not a multiple of 4 and an empty row", () => {
    assertEquals(Array.from(quantizeRowsTiledParams(8, 203).subarray(0, 2)), [8, 203]);
    for (const rows of [1, 2, 3, 6, 4 * 128 + 2]) {
      assertThrows(() => quantizeRowsTiledParams(rows, 512), Error, "multiple of 4");
    }
    assertThrows(() => quantizeRowsTiledParams(4, 0), Error, "positive integer");
    assertThrows(() => quantizeRowsTiledGeometry(0), Error, "positive integer");
    assertThrows(() => quantizeRowsTiledGeometry(2.5), Error, "positive integer");
  });

  test("the production producer builds the tiled form at any key length, and N's forms only for their own N", () => {
    for (const n of KEY_LENGTHS) {
      const launch = attentionValueQuantization(256, n, "tiled");
      const geometry = quantizeRowsTiledGeometry(n);
      assertEquals(launch.geometry, geometry, `N=${n}`);
      assertEquals(launch.key, quantizeRowsTiledKey(geometry), `N=${n}`);
      assertEquals(launch.wgsl, quantizeRowsTiledWgsl(geometry), `N=${n}`);
      assertEquals(Array.from(launch.params), Array.from(quantizeRowsTiledParams(256, n)));
    }
    // Four rows share each vec4 of the tiled Vᵀ, so a row count that is not a multiple of 4 fails
    assertThrows(() => attentionValueQuantization(254, 512, "tiled"), Error, "multiple of 4");
    // The other forms still refuse an N of the other form
    assertThrows(() => attentionValueQuantization(256, 203, "whole"), Error, "'padded'");
    assertThrows(() => attentionValueQuantization(256, 512, "padded"), Error, "'whole'");
  });

  test("the generated WGSL is deterministic and distinct from the other forms", () => {
    const geometry = quantizeRowsTiledGeometry(7920);
    assertEquals(quantizeRowsTiledWgsl(geometry), quantizeRowsTiledWgsl(geometry));
    assertNotEquals(
      attentionOnlineI8a8Wgsl(true, 128, "wg", "bhsd", "tiled"),
      attentionOnlineI8a8Wgsl(true, 128, "wg", "bhsd", "whole"),
    );
  });

  describe("the host permutation (the oracle of the GPU parts)", () => {
    /** Byte `(r·16 + i + 1) mod 256` at row r, key i, packed four keys per word. */
    const plainRows = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
      const words = attentionOnlineValueRowWords(dim);
      const out = new Uint32Array(rows * words);
      for (let row = 0; row < rows; row += 1) {
        for (let i = 0; i < dim; i += 1) {
          out[row * words + (i >> 2)] |= ((row * 16 + i + 1) & 0xff) << (8 * (i & 3));
        }
      }
      return out;
    };
    const byte = (row: number, key: number): number => (row * 16 + key + 1) & 0xff;
    const word = (...bytes: readonly number[]): number =>
      bytes.reduce((sum, value, i) => sum + value * 2 ** (8 * i), 0);

    test("N = 5: word q < 5 holds key q alone, the rest of the tile is zero, four rows interleaved", () => {
      const tiled = tiledValueWords(plainRows(4, 5), 4, 5);
      assertEquals(tiled.length, 4 * 16);
      for (let row = 0; row < 4; row += 1) {
        for (let q = 0; q < 16; q += 1) {
          assertEquals(tiled[q * 4 + row], q < 5 ? byte(row, q) : 0, `row ${row} word ${q}`);
        }
      }
    });

    test("N = 65: word 16·t + pk packs the keys 64·t + pk + 16·i; rows 4..7 are the second group", () => {
      const tiled = tiledValueWords(plainRows(8, 65), 8, 65);
      const n4p = 32;
      // Row 5 = group 1, lane 1
      const at = (row: number, q: number): number =>
        tiled[(Math.floor(row / 4) * n4p + q) * 4 + (row % 4)];
      assertEquals(at(5, 0), word(byte(5, 0), byte(5, 16), byte(5, 32), byte(5, 48)));
      assertEquals(at(5, 15), word(byte(5, 15), byte(5, 31), byte(5, 47), byte(5, 63)));
      assertEquals(at(5, 16), byte(5, 64));
      assertEquals(at(5, 17), 0);
    });

    test("rows that are not a multiple of 4 are refused", () => {
      assertThrows(() => tiledValueWords(plainRows(6, 8), 6, 8), Error, "multiple of 4");
    });
  });
});

/** The device's subgroup facts as the adapter lists them (the sg32 part needs all three). */
const subgroupsListed = async (): Promise<boolean> => {
  const adapter = await navigator.gpu?.requestAdapter();
  return (
    adapter !== null &&
    adapter !== undefined &&
    adapter.features.has("subgroups") &&
    adapter.features.has("subgroup-size-control") &&
    navigator.gpu.wgslLanguageFeatures.has("subgroup_id")
  );
};

/** Asserts the parts; the sg32 part only where the device has subgroups (else it is not run, with the reason). */
const assertParts = async (parts: readonly CheckPart[]): Promise<void> => {
  assertEquals(
    parts.map(({ id }) => id),
    Object.keys(ATTENTION_ONLINE_TILED_CASES),
  );
  const subgroups = await subgroupsListed();
  for (const part of parts) {
    if (part.id.endsWith("-sg32") && !subgroups) {
      assert(part.notRun !== undefined, `${part.id}: ran on a device without subgroups`);
      console.warn(`[tiled] ${part.id} not run: ${part.notRun}`);
      continue;
    }
    assertEquals(
      part.cases.length,
      ATTENTION_ONLINE_TILED_CASES[part.id as keyof typeof ATTENTION_ONLINE_TILED_CASES],
    );
    assertCheckPart(part);
  }
};

/** The failing words of a part's first criterion, per case (the comparison under test). */
const identicalCases = (part: CheckPart): readonly string[] =>
  part.cases
    .filter((checkCase) => (checkCase.criteria[0]?.failures ?? 0) === 0)
    .map(({ label }) => label);

describe.skipIf(!GPU_AVAILABLE)("the tiled key axis (real GPU)", () => {
  it("the producer equals the host permutation of today's producer, and the kernel's O equals today's word for word", async () => {
    const gpu = await acquireGpu({ subgroups: "available" });
    try {
      await assertParts(await checkAttentionOnlineTiled(gpu));
    } finally {
      gpu.destroy();
    }
  });

  it("④ a wrong word index in the producer turns every producer case red, and only those", async () => {
    const gpu = await acquireGpu({ subgroups: "available" });
    try {
      const anchor = "let e = (q / 16u) * 64u + q % 16u;";
      const [producer, ...kernels] = await checkAttentionOnlineTiled(gpu, {
        mutate: {
          producer: (wgsl) => wgsl.replace(anchor, "let e = (q / 16u) * 64u + ((q % 16u) ^ 1u);"),
        },
      });
      assert(producer !== undefined && producer.notRun === undefined);
      assertEquals(
        producer.cases.length,
        ATTENTION_ONLINE_TILED_CASES[producer.id as "attention-online-tiled-producer"],
      );
      assertEquals(identicalCases(producer), [], "producer cases the fault left identical");
      const red = producer.cases.reduce(
        (sum, { criteria }) => sum + (criteria[0]?.failures ?? 0),
        0,
      );
      console.log(
        `[tiled] producer fault: ${red} differing words over ${producer.cases.length} cases`,
      );
      // The kernel parts read host-built Vᵀ, so the producer's fault does not reach them
      for (const part of kernels) if (part.notRun === undefined) assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });

  it("④ a wrong lane in the kernel's Vᵀ load turns every kernel case red, and leaves the producer green", async () => {
    const gpu = await acquireGpu({ subgroups: "available" });
    try {
      const anchor = "+ tile * 16u + pk]";
      const [producer, ...kernels] = await checkAttentionOnlineTiled(gpu, {
        mutate: {
          kernel: (wgsl) => {
            assert(wgsl.includes(anchor), "the tiled load's anchor is gone");
            return wgsl.replaceAll(anchor, "+ tile * 16u + (pk ^ 1u)]");
          },
        },
      });
      assert(producer !== undefined);
      assertCheckPart(producer);
      let ran = 0;
      for (const part of kernels) {
        if (part.notRun !== undefined) continue;
        ran += 1;
        assertEquals(identicalCases(part), [], `${part.id}: kernel cases the fault left identical`);
        const red = part.cases.reduce((sum, { criteria }) => sum + (criteria[0]?.failures ?? 0), 0);
        console.log(
          `[tiled] kernel fault (${part.id}): ${red} differing words over ${part.cases.length} cases`,
        );
      }
      assert(ran > 0, "no kernel part ran");
    } finally {
      gpu.destroy();
    }
  });
});
