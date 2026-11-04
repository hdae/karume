/**
 * The route decision of fused attention (src/runtime/attention-route.ts) — the seats' request ×
 * the shape → the actual route per stage. No GPU needed. The spelling of the conditions (① is
 * `D % 4`, ③ is `N % 4`, f16 storage of S needs both) is canonical in ADR 0030 decision 5 / the doc
 * of score-storage.ts; this file pins that spec from the outside.
 */

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  type AttentionDeviceFacts,
  attentionDeviceFacts,
  attentionRoute,
  type AttentionSeats,
} from "../src/runtime/attention-route.ts";

const A8_S16: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "f16" };
/** The device facts for the seats that never read them (every seat but `"none"`). */
const DEVICE: AttentionDeviceFacts = { subgroups: true, maxComputeWorkgroupStorageSize: 32768 };

describe("attentionRoute: an a8 + s16 request", () => {
  it("when D and N are both multiples of 4, ①③ are i8a8 and S is stored as f16 (as declared)", () => {
    assertEquals(attentionRoute(A8_S16, 128, 4096, DEVICE), {
      qk: "i8a8",
      pv: "i8a8",
      score: "f16",
    });
  });

  it("N % 4 != 0 drops only ③PV and the S storage to f32; ①QK stays i8a8 (each stage is independent)", () => {
    for (const cols of [1089, 1122, 3969]) {
      assertEquals(
        attentionRoute(A8_S16, 128, cols, DEVICE),
        { qk: "i8a8", pv: "dense", score: "f32" },
        `N=${cols}`,
      );
    }
  });

  it("D % 4 != 0 drops only ①QK and the S storage; ③PV stays i8a8", () => {
    assertEquals(attentionRoute(A8_S16, 66, 512, DEVICE), {
      qk: "dense",
      pv: "i8a8",
      score: "f32",
    });
  });

  it("at every point of a small D·N grid, ① depends only on D % 4, ③ on N % 4 and S on both", () => {
    for (let depth = 1; depth <= 12; depth++) {
      for (let cols = 1; cols <= 12; cols++) {
        const qkOk = depth % 4 === 0;
        const pvOk = cols % 4 === 0;
        assertEquals(
          attentionRoute(A8_S16, depth, cols, DEVICE),
          {
            qk: qkOk ? "i8a8" : "dense",
            pv: pvOk ? "i8a8" : "dense",
            score: qkOk && pvOk ? "f16" : "f32",
          },
          `D=${depth} N=${cols}`,
        );
      }
    }
  });
});

describe("attentionRoute: seats that do not request a8 / s16", () => {
  it("f32 / f16 compute runs ①③ dense even at eligible shapes (i8a8 only when requested)", () => {
    for (const attentionCompute of ["f32", "f16"] as const) {
      assertEquals(
        attentionRoute({ attentionCompute, attentionScoreStorage: "f32" }, 128, 4096, DEVICE),
        { qk: "dense", pv: "dense", score: "f32" },
        attentionCompute,
      );
    }
  });

  it("an s16-only request keeps ①③ dense and stores only S as f16 (the two axes are orthogonal)", () => {
    assertEquals(
      attentionRoute({ attentionCompute: "f32", attentionScoreStorage: "f16" }, 128, 512, DEVICE),
      { qk: "dense", pv: "dense", score: "f16" },
    );
  });

  it("an a8-only request keeps S stored as f32 (with N % 4 != 0 only ③ drops)", () => {
    const seats: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "f32" };
    assertEquals(attentionRoute(seats, 128, 4096, DEVICE), {
      qk: "i8a8",
      pv: "i8a8",
      score: "f32",
    });
    assertEquals(attentionRoute(seats, 128, 1089, DEVICE), {
      qk: "i8a8",
      pv: "dense",
      score: "f32",
    });
  });
});

describe("attentionRoute: the online kernel (attentionScoreStorage 'none' — ADR 0132)", () => {
  const A8_NONE: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "none" };
  /** Dawn / Vulkan class: subgroups and 48 KiB of workgroup storage. */
  const SUBGROUPS_48K: AttentionDeviceFacts = {
    subgroups: true,
    maxComputeWorkgroupStorageSize: 49152,
  };
  /** A device without subgroups that still holds the workgroup-memory form (D = 128 needs 35,328 B). */
  const PORTABLE_48K: AttentionDeviceFacts = {
    subgroups: false,
    maxComputeWorkgroupStorageSize: 49152,
  };
  /** The WebGPU core default (16 KiB) without subgroups: neither form fits at D = 128. */
  const CORE_16K: AttentionDeviceFacts = {
    subgroups: false,
    maxComputeWorkgroupStorageSize: 16384,
  };
  /** The falls-back route: the three dispatches of a8 with S in f32 (never "none"). */
  const FALLBACK = { qk: "i8a8", pv: "i8a8", score: "f32" } as const;

  it("picks the subgroup form on a device with subgroups and the workgroup-memory form without", () => {
    for (const depth of [64, 128]) {
      assertEquals(
        attentionRoute(A8_NONE, depth, 27280, SUBGROUPS_48K),
        { qk: "i8a8", pv: "i8a8", score: "none", online: "sg32" },
        `D=${depth}`,
      );
      assertEquals(
        attentionRoute(A8_NONE, depth, 512, PORTABLE_48K),
        { qk: "i8a8", pv: "i8a8", score: "none", online: "wg" },
        `D=${depth}`,
      );
    }
  });

  it("decides the form from the workgroup-storage limit at its boundary (D = 128: sg32 31,232 B, wg 35,328 B)", () => {
    // Without subgroups, one byte below the workgroup-memory form: the three dispatches.
    // ((64·9 + 64·9 + 16·33 + 16·17) vec4 = 7,808 words, plus 64·16 words of scratch = 8,832 words
    // — the kernel test's arithmetic.)
    assertEquals(
      attentionRoute(A8_NONE, 128, 512, {
        subgroups: false,
        maxComputeWorkgroupStorageSize: 35327,
      }),
      FALLBACK,
    );
    assertEquals(
      attentionRoute(A8_NONE, 128, 512, {
        subgroups: false,
        maxComputeWorkgroupStorageSize: 35328,
      }),
      { qk: "i8a8", pv: "i8a8", score: "none", online: "wg" },
    );
    // With subgroups the scratch is not needed: 32 KiB holds the subgroup form, 31,231 B does not.
    assertEquals(
      attentionRoute(A8_NONE, 128, 512, { subgroups: true, maxComputeWorkgroupStorageSize: 31231 }),
      FALLBACK,
    );
    assertEquals(
      attentionRoute(A8_NONE, 128, 512, { subgroups: true, maxComputeWorkgroupStorageSize: 31232 }),
      { qk: "i8a8", pv: "i8a8", score: "none", online: "sg32" },
    );
    assertEquals(
      attentionRoute(A8_NONE, 128, 512, { subgroups: true, maxComputeWorkgroupStorageSize: 32768 }),
      { qk: "i8a8", pv: "i8a8", score: "none", online: "sg32" },
    );
    // Subgroups but a limit below the subgroup form's tiles: no form fits.
    assertEquals(attentionRoute(A8_NONE, 128, 512, { ...CORE_16K, subgroups: true }), FALLBACK);
  });

  it("falls back to the three a8 dispatches with S in f32 for ineligible shapes and devices", () => {
    // D = 96 (D % 64 != 0) and D = 256 (above the generator's maximum)
    assertEquals(attentionRoute(A8_NONE, 96, 512, SUBGROUPS_48K), FALLBACK);
    assertEquals(attentionRoute(A8_NONE, 256, 512, SUBGROUPS_48K), FALLBACK);
    // A 16 KiB device without subgroups
    assertEquals(attentionRoute(A8_NONE, 128, 512, CORE_16K), FALLBACK);
  });

  it("runs N % 4 != 0 on the online kernel (padded Vᵀ — ADR 0132 addendum 2026-10-27); the three-dispatch seats keep ③PV's N % 4", () => {
    // Before the addendum these shapes fell back to { qk: i8a8, pv: dense, score: f32 }.
    for (const cols of [65, 203, 1089, 4106]) {
      assertEquals(
        attentionRoute(A8_NONE, 128, cols, SUBGROUPS_48K),
        { qk: "i8a8", pv: "i8a8", score: "none", online: "sg32" },
        `N=${cols}`,
      );
      assertEquals(
        attentionRoute(A8_NONE, 64, cols, PORTABLE_48K),
        { qk: "i8a8", pv: "i8a8", score: "none", online: "wg" },
        `N=${cols}`,
      );
      // Only the online kernel learned the key valid length: a8 + s16 (Anima's preset) still runs
      // ③PV and S in f32 at these shapes.
      assertEquals(
        attentionRoute({ attentionCompute: "a8", attentionScoreStorage: "f16" }, 128, cols, DEVICE),
        { qk: "i8a8", pv: "dense", score: "f32" },
        `a8 + s16 N=${cols}`,
      );
    }
    // A device that holds neither form falls back as for N % 4 == 0 — ③PV then drops to dense
    assertEquals(attentionRoute(A8_NONE, 128, 1089, CORE_16K), {
      qk: "i8a8",
      pv: "dense",
      score: "f32",
    });
  });

  it("never returns 'none' without attentionCompute 'a8' (the option check rejects it earlier)", () => {
    for (const attentionCompute of ["f32", "f16"] as const) {
      assertEquals(
        attentionRoute(
          { attentionCompute, attentionScoreStorage: "none" },
          128,
          512,
          SUBGROUPS_48K,
        ),
        { qk: "dense", pv: "dense", score: "f32" },
        attentionCompute,
      );
    }
  });

  it("reads the device facts from the features the device enabled, the measured lane mapping and its granted limit", () => {
    const limits = { maxComputeWorkgroupStorageSize: 49152 };
    const all = new Set(["subgroups", "subgroup-size-control"]);
    const language = new Set(["subgroup_id"]);
    assertEquals(
      attentionDeviceFacts({
        features: all,
        wgslLanguageFeatures: language,
        subgroupLanesLinear: true,
        limits,
      }),
      { subgroups: true, maxComputeWorkgroupStorageSize: 49152 },
    );
    for (const [features, wgsl, linear] of [
      [new Set(["subgroups"]), language, true],
      [new Set(["subgroup-size-control"]), language, true],
      [all, new Set<string>(), true],
      // The features alone are not enough: the sg32 form addresses its tiles by the thread index
      // and needs the measured linear lane mapping (ADR 0132 decision 5).
      [all, language, false],
    ] as const) {
      assertEquals(
        attentionDeviceFacts({
          features,
          wgslLanguageFeatures: wgsl,
          subgroupLanesLinear: linear,
          limits,
        }).subgroups,
        false,
      );
    }
  });
});
