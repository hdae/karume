/**
 * 融合 attention の経路の判定（src/runtime/attention-route.ts）— 席の要求 × 形 → 段ごとの実経路。
 * GPU 不要。条件の綴り（① は `D % 4`・③ は `N % 4`・S の f16 格納は両方）は ADR 0030 決定 5 /
 * score-storage.ts の doc が正本で、ここはその仕様を外から固定する。
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

describe("attentionRoute: a8 + s16 の要求", () => {
  it("D と N がともに 4 の倍数なら、①③ が i8a8・S が f16 格納（宣言どおり）", () => {
    assertEquals(attentionRoute(A8_S16, 128, 4096, DEVICE), {
      qk: "i8a8",
      pv: "i8a8",
      score: "f16",
    });
  });

  it("N % 4 != 0 は ③PV と S の格納だけが f32 へ落ち、①QK は i8a8 のまま（段ごとに独立）", () => {
    for (const cols of [1089, 1122, 3969]) {
      assertEquals(
        attentionRoute(A8_S16, 128, cols, DEVICE),
        { qk: "i8a8", pv: "dense", score: "f32" },
        `N=${cols}`,
      );
    }
  });

  it("D % 4 != 0 は ①QK と S の格納だけが落ち、③PV は i8a8 のまま", () => {
    assertEquals(attentionRoute(A8_S16, 66, 512, DEVICE), {
      qk: "dense",
      pv: "i8a8",
      score: "f32",
    });
  });

  it("D・N の小さい格子の全点で、① は D % 4・③ は N % 4・S は両方の条件だけで決まる", () => {
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

describe("attentionRoute: a8 / s16 を要求しない席", () => {
  it("f32 / f16 の計算は、適格な形でも ①③ が dense（i8a8 は要求したときだけ）", () => {
    for (const attentionCompute of ["f32", "f16"] as const) {
      assertEquals(
        attentionRoute({ attentionCompute, attentionScoreStorage: "f32" }, 128, 4096, DEVICE),
        { qk: "dense", pv: "dense", score: "f32" },
        attentionCompute,
      );
    }
  });

  it("s16 だけの要求は、①③ が dense のまま S だけが f16 格納になる（2 つの軸は直交）", () => {
    assertEquals(
      attentionRoute({ attentionCompute: "f32", attentionScoreStorage: "f16" }, 128, 512, DEVICE),
      { qk: "dense", pv: "dense", score: "f16" },
    );
  });

  it("a8 だけの要求は、S が f32 格納のまま（N % 4 != 0 でも ③ だけが落ちる）", () => {
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
    // N % 4 != 0: ③PV falls back too (its own eligibility), S stays f32
    assertEquals(attentionRoute(A8_NONE, 128, 1089, SUBGROUPS_48K), {
      qk: "i8a8",
      pv: "dense",
      score: "f32",
    });
    // A 16 KiB device without subgroups
    assertEquals(attentionRoute(A8_NONE, 128, 512, CORE_16K), FALLBACK);
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
