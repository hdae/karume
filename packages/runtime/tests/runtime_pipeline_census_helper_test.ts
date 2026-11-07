// The behavior of the shared census helpers (tests/helpers/pipeline-census.ts) as pure functions.
// No GPU — the rows are fakes, and the predicates are matched against the real keys returned by
// src's key functions.

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  ATTENTION_STATS_KEY,
  attentionPvKey,
  attentionQkKey,
  attentionStatsKey,
} from "../src/kernels/attention.ts";
import { attentionPvI8a8Key, attentionQkI8a8Key } from "../src/kernels/attention-i8a8.ts";
import {
  attentionOnlineI8a8Key,
  attentionOnlineKeyWords,
} from "../src/kernels/attention-online-i8a8.ts";
import { linearKey } from "../src/kernels/linear.ts";
import {
  linearGemvKey,
  linearGemvParallelKey,
  linearGemvParallelPackedKey,
  linearGemvStaticQuantizeKey,
  linearGemvStaticQuantizePackedKey,
  linearGemvSubgroupKey,
} from "../src/kernels/linear-gemv.ts";
import { integerGemvKey } from "../src/kernels/linear-gemv-integer.ts";
import { planesGemvKey } from "../src/kernels/linear-gemv-planes.ts";
import { conv3dI8a8Key } from "../src/kernels/conv3d-i8a8.ts";
import { geluMulStaticQuantizeKey } from "../src/kernels/gelu-mul-static-quantize.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../src/kernels/geometry-profile.ts";
import { linearI8a8Key } from "../src/kernels/linear-i8a8.ts";
import type { MaterialisedScoreStorage } from "../src/kernels/score-storage.ts";
import { QUANTIZE_ROWS_KEY } from "../src/kernels/quantize-rows.ts";
import { RMS_NORM_KEY, rmsNormAddKey } from "../src/kernels/rms-norm.ts";
import { rmsNormSubgroupKey } from "../src/kernels/rms-norm-subgroup.ts";
import {
  statePvKey,
  statePvParallelKey,
  statePvParallelReadonlyKey,
  stateQkFoldedKey,
  stateQkKey,
  stateQkParallelKey,
  stateQkParallelReadonlyKey,
  stateStatsKey,
} from "../src/kernels/state-attention.ts";
import { stateAppendKey } from "../src/kernels/state-append.ts";
import { stateStatsPvKey } from "../src/kernels/state-attention-stats-pv.ts";
import { STATIC_QUANTIZE_KEY, STATIC_QUANTIZE_PACKED_KEY } from "../src/kernels/static-quantize.ts";
import {
  type AttentionDeviceFacts,
  type AttentionRoute,
  attentionRoute,
  type AttentionSeats,
} from "../src/runtime/attention-route.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import {
  assertAttentionSeatsComplete,
  assertPipelineCensus,
  assertSeatsApplied,
  type CensusRow,
  countDispatches,
  type KeyPredicate,
  mergeCensus,
  type NumericSeat,
  requireCensus,
  SEAT_SIGNATURES,
  type SeatSignatures,
} from "./helpers/pipeline-census.ts";

const rows = (...entries: readonly (readonly [string, number])[]): readonly CensusRow[] =>
  entries.map(([key, dispatchCount]) => ({ key, dispatchCount }));

/**
 * Representative keys per option × non-reference value (the real ones returned by src's key
 * functions). If a predicate does not match these, that option's census counts nothing on a real
 * machine (a tautological red, or a green that misses something).
 */
const REPRESENTATIVES: {
  readonly [Seat in NumericSeat]: {
    readonly [Value in keyof SeatSignatures[Seat]]: readonly string[];
  };
} = {
  linearCompute: {
    a8: [linearI8a8Key(true, true), linearI8a8Key(false, false, undefined, "i4", 32)],
    f16: [linearKey("f32", true, "f16"), linearKey("f16", false, "f16", 64)],
  },
  convCompute: {
    a8: [
      conv3dI8a8Key(true, true, DEFAULT_GEOMETRY_PROFILE.i8a8.conv3d),
      conv3dI8a8Key(false, false, DEFAULT_GEOMETRY_PROFILE.i8a8.conv3d),
    ],
  },
  attentionCompute: {
    f16: [attentionQkKey(true, "f16"), attentionPvKey(false, "f16"), attentionStatsKey("f16")],
    a8: [
      attentionQkI8a8Key(true, false),
      attentionPvI8a8Key(false, true, "f16"),
      attentionOnlineI8a8Key(false, 64, "wg"),
    ],
  },
  attentionScoreStorage: {
    f16: [
      attentionQkKey(true, "f32", "f16", true),
      attentionStatsKey("f32", "f16", 4),
      attentionPvI8a8Key(true, true, "f16"),
    ],
    none: [attentionOnlineI8a8Key(true, 128, "sg32"), attentionOnlineI8a8Key(false, 64, "wg")],
  },
  stateAttentionReduce: {
    parallel: [
      stateQkParallelKey(false, true),
      statePvParallelKey(true, false),
      stateQkParallelReadonlyKey(true, true),
      statePvParallelReadonlyKey(false, false),
      stateQkFoldedKey(4),
      stateQkFoldedKey(8),
      // On an f16 full slot the same forms still read as "parallel ran".
      stateQkParallelKey(false, true, "f16"),
      statePvParallelKey(false, false, "f16"),
      stateQkParallelReadonlyKey(false, false, "f16"),
      statePvParallelReadonlyKey(false, true, "f16"),
      stateQkFoldedKey(8, "f16"),
    ],
    "parallel-fused": [stateStatsPvKey(false, false), stateStatsPvKey(true, true)],
  },
  fullStateStorage: {
    f16: [
      stateQkKey(false, true, "f16"),
      stateQkParallelKey(false, false, "f16"),
      stateQkFoldedKey(8, "f16"),
      statePvParallelReadonlyKey(false, true, "f16"),
      stateAppendKey(false, "f16"),
    ],
  },
  linearGemvReduce: {
    parallel: [linearGemvParallelKey("i4", 32, 4), linearGemvStaticQuantizeKey("i2", undefined, 8)],
    "parallel-subgroup32": [linearGemvSubgroupKey("i8", undefined, 16)],
  },
  fuseRmsNormAdd: {
    true: [rmsNormAddKey("residual-norm", 1536), rmsNormSubgroupKey("norm-residual")],
  },
  fuseLinearStaticQuantize: {
    true: [
      linearGemvStaticQuantizeKey("i4", 512, 4),
      linearGemvStaticQuantizePackedKey("i8", undefined, 2),
    ],
  },
  fuseGeluMulStaticQuantize: {
    true: [geluMulStaticQuantizeKey("gelu-other"), geluMulStaticQuantizeKey("other-gelu")],
  },
  packedStaticQuantize: {
    true: [STATIC_QUANTIZE_PACKED_KEY, linearGemvParallelPackedKey("i4", 32, 8)],
  },
  integerGemv: {
    true: [
      integerGemvKey({ weight: { storage: "i2" }, lanes: 8, rows: 4 }, false),
      integerGemvKey({ weight: { storage: "i2" }, lanes: 16, rows: 4 }, true),
      integerGemvKey(
        { weight: { storage: "i4", groupSize: 2048, groups: 3 }, lanes: 8, rows: 2 },
        true,
      ),
      integerGemvKey({ weight: { storage: "i8" }, lanes: 4, rows: 1 }, false),
    ],
  },
  linearGemvPlanes: {
    true: [planesGemvKey({ lanes: 8, rows: 4 }), planesGemvKey({ lanes: 32, rows: 8 })],
  },
  rmsNormReduce: {
    subgroup32: [rmsNormSubgroupKey(), rmsNormSubgroupKey("residual-norm")],
  },
};

/**
 * Keys run by the reference path (every option at its default). They must match no option's
 * predicate.
 */
const REFERENCE_KEYS: readonly string[] = [
  linearKey("f32", true),
  linearKey("i8", true, "f32", 1),
  linearKey("i4", false, "f32", undefined, 32),
  linearGemvKey("i8"),
  linearGemvKey("i4", 64),
  // The sequential GEMV the byte-plane GEMV replaces (the QAT lm_head with the option off).
  linearGemvKey("i2"),
  attentionQkKey(true),
  attentionQkKey(false, "f32", "f32", true, true, true),
  attentionPvKey(true),
  ATTENTION_STATS_KEY,
  attentionStatsKey("f32", "f32", 16),
  stateQkKey(false, false),
  stateQkKey(true, true),
  statePvKey(true, true),
  stateStatsKey(false),
  RMS_NORM_KEY,
  STATIC_QUANTIZE_KEY,
  QUANTIZE_ROWS_KEY,
];

/**
 * Type exhaustiveness is held by the two tables above, so the traversal handles option and value
 * names as strings.
 */
const signatures: Readonly<Record<string, Readonly<Record<string, KeyPredicate>>>> =
  SEAT_SIGNATURES;
const representatives: Readonly<Record<string, Readonly<Record<string, readonly string[]>>>> =
  REPRESENTATIVES;

/** The coverage table flattened into (option, value, predicate). */
const signatureEntries = (): readonly (readonly [string, string, KeyPredicate])[] =>
  Object.entries(signatures).flatMap(([seat, variants]) =>
    Object.entries(variants).map(([value, predicate]) => [seat, value, predicate] as const),
  );

describe("SEAT_SIGNATURES", () => {
  it("each option × non-reference value predicate matches every representative key of that variant", () => {
    for (const [seat, value, predicate] of signatureEntries()) {
      const keys = representatives[seat]?.[value] ?? [];
      assertEquals(keys.length > 0, true, `${seat}=${value}: no representative keys`);
      for (const key of keys) {
        assertEquals(predicate(key), true, `${seat}=${value} does not match '${key}'`);
      }
    }
  });

  it("no predicate matches the reference path's keys (the premise for asserting zero for a reference-value option)", () => {
    for (const [seat, value, predicate] of signatureEntries()) {
      for (const key of REFERENCE_KEYS) {
        assertEquals(predicate(key), false, `${seat}=${value} matches the reference key '${key}'`);
      }
    }
  });

  it("non-reference values of the same option do not match each other's representative keys (per-value censuses do not mix)", () => {
    for (const [seat, value, predicate] of signatureEntries()) {
      for (const [other, keys] of Object.entries(representatives[seat] ?? {})) {
        if (other === value) continue;
        for (const key of keys) {
          assertEquals(predicate(key), false, `${seat}=${value} matches ${other}'s '${key}'`);
        }
      }
    }
  });
});

describe("requireCensus", () => {
  it("undefined (no latest run / it failed) fails loudly", () => {
    assertThrows(() => requireCensus(undefined, "where"), Error, "lastRunPipelines");
  });

  it("an empty table fails loudly (an assertion of zero is not made tautological)", () => {
    assertThrows(() => requireCensus([], "where"), Error, "the census is empty");
  });

  it("returns the rows as is when there are any", () => {
    const census = rows(["a", 1]);
    assertEquals(requireCensus(census, "where"), census);
  });
});

describe("countDispatches", () => {
  it("a string is an exact match, and a predicate sums the counts of the rows it matches", () => {
    const census = rows(["linear:a", 3], ["linear:b", 2], ["relu", 5]);
    assertEquals(countDispatches(census, "linear:a"), 3);
    assertEquals(countDispatches(census, "linear"), 0, "not a prefix match");
    assertEquals(
      countDispatches(census, (key) => key.startsWith("linear:")),
      5,
    );
  });
});

describe("mergeCensus", () => {
  it("sums counts per key across runs and orders them lexicographically", () => {
    assertEquals(
      mergeCensus([rows(["b", 1], ["a", 2]), rows(["a", 3], ["c", 1])], "where"),
      rows(["a", 5], ["b", 1], ["c", 1]),
    );
  });

  it("fails loudly if a run without a census is mixed in (that run's variants are not silently dropped)", () => {
    assertThrows(() => mergeCensus([rows(["a", 1]), undefined], "where"), Error, "run 1 has no");
  });
});

describe("assertPipelineCensus", () => {
  const census = rows(["linear:a", 3], ["relu", 1]);

  it("passes when both the exact count and the lower bound hold", () => {
    assertPipelineCensus(
      census,
      [
        { label: "linear:a", match: "linear:a", count: 3 },
        { label: "relu or more", match: "relu", atLeast: 1 },
        { label: "missing key", match: "gelu", count: 0 },
      ],
      "where",
    );
  });

  it("fails when the exact count is off", () => {
    assertThrows(
      () => assertPipelineCensus(census, [{ label: "L", match: "linear:a", count: 2 }], "where"),
      Error,
      "L ran 3 times (expected 2)",
    );
  });

  it("fails below the lower bound", () => {
    assertThrows(
      () => assertPipelineCensus(census, [{ label: "R", match: "relu", atLeast: 2 }], "where"),
      Error,
      "R ran 1 times (expected at least 2)",
    );
  });

  it("undefined / empty rows fail even when only zero is expected", () => {
    const zero = [{ label: "Z", match: "x", count: 0 }] as const;
    assertThrows(() => assertPipelineCensus(undefined, zero, "where"), Error, "lastRunPipelines");
    assertThrows(() => assertPipelineCensus([], zero, "where"), Error, "the census is empty");
  });
});

describe("assertSeatsApplied", () => {
  const parallel = linearGemvParallelKey("i4", 32, 4);
  const sequential = linearGemvKey("i4", 32);

  it("an option with a non-reference value passes if at least one dispatch of that variant ran", () => {
    assertSeatsApplied(
      rows([parallel, 7], [RMS_NORM_KEY, 2]),
      {
        linearGemvReduce: "parallel",
      },
      "where",
    );
  });

  it("an option with a non-reference value fails if no dispatch of the variant ran (detects silent fallback)", () => {
    assertThrows(
      () => assertSeatsApplied(rows([sequential, 7]), { linearGemvReduce: "parallel" }, "where"),
      Error,
      'no dispatch of the linearGemvReduce="parallel" variant ran',
    );
  });

  it("an option explicitly set to its reference value fails if a variant ran (the option leaked)", () => {
    assertThrows(
      () =>
        assertSeatsApplied(
          rows([sequential, 3], [rmsNormAddKey("norm-residual", 1536), 1]),
          { fuseRmsNormAdd: false },
          "where",
        ),
      Error,
      "fuseRmsNormAdd is the reference value false yet the true variant ran",
    );
  });

  it("an option explicitly set to its reference value passes if no non-reference variant ran", () => {
    assertSeatsApplied(
      rows([sequential, 3], [RMS_NORM_KEY, 1]),
      {
        linearGemvReduce: "sequential",
        fuseRmsNormAdd: false,
        rmsNormReduce: "workgroup",
      },
      "where",
    );
  });

  it("a reference-value option fails if any one non-reference variant ran (subgroup32 is also a variant of the parallel family)", () => {
    assertThrows(
      () =>
        assertSeatsApplied(
          rows([linearGemvSubgroupKey("i8", undefined, 4), 1]),
          { linearGemvReduce: "sequential" },
          "where",
        ),
      Error,
      "the parallel-subgroup32 variant ran",
    );
  });

  it("options not passed are not checked (the caller decides the effective value when omitted)", () => {
    assertSeatsApplied(rows([parallel, 1]), { fuseRmsNormAdd: false }, "where");
  });

  it("fails on an empty census even when only fields that do not change numerics are given", () => {
    assertThrows(
      () => assertSeatsApplied([], { planBackingBudgetBytes: 0 }, "where"),
      Error,
      "the census is empty",
    );
  });
});

/**
 * Pins the full-eligibility check of fused attention (`assertAttentionSeatsComplete` in
 * tests/helpers/pipeline-census.ts) with synthetic census rows (no GPU needed). The keys are built
 * from src's key functions — confirming that the check matches the same spelling execution emits,
 * without keeping the strings twice.
 *
 * Role of the fault injection: mixing one fallen-back dispatch into a census as declared turns it
 * red (the check is not tautological).
 */

const LAYERS = 30;
const A8_S16: SessionOptions = { attentionCompute: "a8", attentionScoreStorage: "f16" };
/** Device facts with subgroups and 48 KiB of workgroup storage (the online kernel's sg32 form). */
const DEVICE: AttentionDeviceFacts = { subgroups: true, maxComputeWorkgroupStorageSize: 49152 };

/**
 * Census rows for one fused-attention node × `blocks` row blocks (`blocks` each of ①②③). The path
 * uses `attentionRoute`'s return value as is — the rows come from the same decision as execution
 * (buildAttention).
 * `cols` (N) picks the online kernel's key-axis form, as execution does (`…:kpad` at N % 4 != 0).
 */
const attentionRows = (
  route: AttentionRoute,
  blocks: number,
  cols: number,
): readonly CensusRow[] => {
  // The online route (ADR 0132): one dispatch per row window, no ①②③.
  if (route.online !== undefined) {
    return [
      {
        key: attentionOnlineI8a8Key(true, 128, route.online, "bhsd", attentionOnlineKeyWords(cols)),
        dispatchCount: blocks,
      },
    ];
  }
  const score: MaterialisedScoreStorage = route.score;
  return [
    {
      key:
        route.qk === "i8a8"
          ? attentionQkI8a8Key(true, true, score)
          : attentionQkKey(true, "f32", score),
      dispatchCount: blocks,
    },
    { key: attentionStatsKey("f32", score, 16), dispatchCount: blocks },
    {
      key:
        route.pv === "i8a8"
          ? attentionPvI8a8Key(true, true, score)
          : attentionPvKey(true, "f32", score),
      dispatchCount: blocks,
    },
  ];
};

/**
 * A census of 30 transformer layers with the a8 + s16 options, D = 128 self (key length `selfCols`,
 * `blocks` row blocks) and cross (N = 512, 1 block) (linear i8a8 is mixed in too — confirming rows
 * outside the families are not counted).
 */
const ditCensus = (
  selfCols: number,
  blocks: number,
  seats: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "f16" },
  selfDepth = 128,
): readonly CensusRow[] => {
  const nodeRows = [
    ...attentionRows(attentionRoute(seats, selfDepth, selfCols, DEVICE), blocks, selfCols),
    ...attentionRows(attentionRoute(seats, 128, 512, DEVICE), 1, 512),
  ];
  // Grouped per family (as in a real census, identical keys fold into one row).
  const counts = new Map<string, number>();
  for (const { key, dispatchCount } of nodeRows) {
    counts.set(key, (counts.get(key) ?? 0) + dispatchCount * LAYERS);
  }
  return [
    { key: linearI8a8Key(true, true), dispatchCount: 300 },
    ...[...counts].map(([key, dispatchCount]) => ({ key, dispatchCount })),
  ];
};

describe("assertAttentionSeatsComplete: a census as declared", () => {
  it("shapes eligible for both self / cross (N = 4096 / 512) are green regardless of the row block count", () => {
    for (const blocks of [1, 3]) {
      assertAttentionSeatsComplete(ditCensus(4096, blocks), A8_S16, `${blocks} blocks`);
    }
  });

  it("options that do not declare a8 / s16 are not checked even with dense attention", () => {
    const census = attentionRows(
      attentionRoute({ attentionCompute: "f32", attentionScoreStorage: "f32" }, 128, 1089, DEVICE),
      1,
      1089,
    );
    assertAttentionSeatsComplete(census, {}, "no options");
    assertAttentionSeatsComplete(census, { attentionCompute: "f32" }, "reference value");
  });

  it("an a8-only option does not look at S's storage (green with i8a8 on f32 storage)", () => {
    const census = attentionRows(
      attentionRoute({ attentionCompute: "a8", attentionScoreStorage: "f32" }, 128, 4096, DEVICE),
      2,
      4096,
    );
    assertAttentionSeatsComplete(census, { attentionCompute: "a8" }, "a8 only");
  });
});

describe("assertAttentionSeatsComplete: censuses with fallbacks (fault injection)", () => {
  it("mixing in one dense attention_pv:v1:f32 turns red on missing i8a8 in the ③PV family", () => {
    const census: CensusRow[] = [
      ...ditCensus(4096, 1),
      { key: attentionPvKey(true, "f32", "f16"), dispatchCount: 1 },
    ];
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "mixed in"),
      Error,
      'attentionCompute="a8": of the attention_pv dispatches',
    );
  });

  it("one missing :s16 in ② row statistics turns red on missing s16 (a8 still satisfied)", () => {
    const census: CensusRow[] = [
      ...ditCensus(4096, 1),
      { key: attentionStatsKey("f32", "f32", 16), dispatchCount: 1 },
    ];
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "missing s16"),
      Error,
      'attentionScoreStorage="f16": of the attention_stats dispatches',
    );
  });

  it("a census where self has N % 4 != 0 (N = 1089 — Anima's 528×528) is red while assertSeatsApplied stays green", () => {
    const census = ditCensus(1089, 1);
    // The "at least one" check is satisfied by cross's i8a8 / s16 alone — this is the hole being
    // closed here.
    assertSeatsApplied(census, A8_S16, "at least one");
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "self falls back"),
      Error,
      "fell back to f32",
    );
  });

  it("a census with no fused attention at all is red as missing declared families (no green from 0 == 0)", () => {
    const census: CensusRow[] = [{ key: linearI8a8Key(true, true), dispatchCount: 4 }];
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "no attention"),
      Error,
      "was declared but there is no",
    );
  });
});

describe("the census helpers with attentionScoreStorage 'none' (the online kernel — ADR 0132)", () => {
  const A8_NONE: SessionOptions = { attentionCompute: "a8", attentionScoreStorage: "none" };
  const seats = { attentionCompute: "a8", attentionScoreStorage: "none" } as const;

  it("a fully eligible census (self N = 4096 in 3 windows, cross N = 512) is applied and complete", () => {
    const census = ditCensus(4096, 3, seats);
    assertSeatsApplied(census, A8_NONE, "online");
    assertAttentionSeatsComplete(census, A8_NONE, "online");
  });

  it("self N = 1089 (N % 4 != 0, the padded key axis — ADR 0132 addendum 2026-10-27) is complete too", () => {
    const census = ditCensus(1089, 1, seats);
    // The self node's row is the padded kernel's key, as execution spells it (not the whole form's)
    assertEquals(
      census.filter((row) => row.key.endsWith(":kpad")).length,
      1,
      "one padded online key",
    );
    assertSeatsApplied(census, A8_NONE, "online, N % 4 != 0");
    assertAttentionSeatsComplete(census, A8_NONE, "online, N % 4 != 0");
  });

  it("a node that fell back to the three dispatches (self D = 96) is applied but not complete", () => {
    // The cross attention runs the online kernel, the self attention the three a8 dispatches in f32.
    // Before the ADR 0132 addendum of 2026-10-27 the fallback came from self N = 1089 (N % 4 != 0).
    const census = ditCensus(4096, 1, seats, 96);
    assertSeatsApplied(census, A8_NONE, "partly online");
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_NONE, "partly online"),
      Error,
      "fell back to the three dispatches",
    );
  });

  it("a census without the online kernel is neither applied nor complete", () => {
    const census = ditCensus(4096, 1);
    assertThrows(() => assertSeatsApplied(census, A8_NONE, "no online"), Error, "none");
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_NONE, "no online"),
      Error,
      "no attention_online dispatch ran",
    );
  });
});
