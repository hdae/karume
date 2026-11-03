// census 共通ヘルパ（tests/helpers/pipeline-census.ts）の純関数としての振る舞い。GPU は使わない —
// 行は偽物で、述語の相手は src の key 関数が返す実物のキー。

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
 * 各席 × 非参照値の代表キー（src の key 関数が返す実物）。述語がこれに当たらなければ、
 * その席の census は実機で 1 本も数えない（恒真の赤か、見落としの緑になる）。
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
    ],
    "parallel-fused": [stateStatsPvKey(false, false), stateStatsPvKey(true, true)],
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
  rmsNormReduce: {
    subgroup32: [rmsNormSubgroupKey(), rmsNormSubgroupKey("residual-norm")],
  },
};

/** 参照経路（全席が省略値）で走るキー。どの席の述語にも当たってはならない。 */
const REFERENCE_KEYS: readonly string[] = [
  linearKey("f32", true),
  linearKey("i8", true, "f32", 1),
  linearKey("i4", false, "f32", undefined, 32),
  linearGemvKey("i8"),
  linearGemvKey("i4", 64),
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

/** 型の網羅は上の 2 表が持つので、走査は席名・値名を文字列で扱う。 */
const signatures: Readonly<Record<string, Readonly<Record<string, KeyPredicate>>>> =
  SEAT_SIGNATURES;
const representatives: Readonly<Record<string, Readonly<Record<string, readonly string[]>>>> =
  REPRESENTATIVES;

/** 網羅表を (席, 値, 述語) の平たい列に。 */
const signatureEntries = (): readonly (readonly [string, string, KeyPredicate])[] =>
  Object.entries(signatures).flatMap(([seat, variants]) =>
    Object.entries(variants).map(([value, predicate]) => [seat, value, predicate] as const),
  );

describe("SEAT_SIGNATURES", () => {
  it("各席 × 非参照値の述語は、その変種の代表キー全てに当たる", () => {
    for (const [seat, value, predicate] of signatureEntries()) {
      const keys = representatives[seat]?.[value] ?? [];
      assertEquals(keys.length > 0, true, `${seat}=${value}: 代表キーが無い`);
      for (const key of keys) {
        assertEquals(predicate(key), true, `${seat}=${value} が '${key}' に当たらない`);
      }
    }
  });

  it("どの述語も参照経路のキーには当たらない（参照値の席で 0 本を断言できる前提）", () => {
    for (const [seat, value, predicate] of signatureEntries()) {
      for (const key of REFERENCE_KEYS) {
        assertEquals(predicate(key), false, `${seat}=${value} が参照キー '${key}' に当たる`);
      }
    }
  });

  it("同じ席の非参照値どうしは互いの代表キーに当たらない（値ごとの census が混ざらない）", () => {
    for (const [seat, value, predicate] of signatureEntries()) {
      for (const [other, keys] of Object.entries(representatives[seat] ?? {})) {
        if (other === value) continue;
        for (const key of keys) {
          assertEquals(predicate(key), false, `${seat}=${value} が ${other} の '${key}' に当たる`);
        }
      }
    }
  });
});

describe("requireCensus", () => {
  it("undefined（直近 run が無い / 失敗）は fail loudly", () => {
    assertThrows(() => requireCensus(undefined, "where"), Error, "lastRunPipelines");
  });

  it("空の表は fail loudly（0 本の断言を恒真にしない）", () => {
    assertThrows(() => requireCensus([], "where"), Error, "census が空");
  });

  it("行があればそのまま返す", () => {
    const census = rows(["a", 1]);
    assertEquals(requireCensus(census, "where"), census);
  });
});

describe("countDispatches", () => {
  it("文字列は完全一致・述語は当たった行の本数を合計する", () => {
    const census = rows(["linear:a", 3], ["linear:b", 2], ["relu", 5]);
    assertEquals(countDispatches(census, "linear:a"), 3);
    assertEquals(countDispatches(census, "linear"), 0, "前方一致ではない");
    assertEquals(
      countDispatches(census, (key) => key.startsWith("linear:")),
      5,
    );
  });
});

describe("mergeCensus", () => {
  it("run を跨いでキー別に本数を足し、辞書順に並べる", () => {
    assertEquals(
      mergeCensus([rows(["b", 1], ["a", 2]), rows(["a", 3], ["c", 1])], "where"),
      rows(["a", 5], ["b", 1], ["c", 1]),
    );
  });

  it("census の無い run が混ざれば fail loudly（その run の変種を黙って落とさない）", () => {
    assertThrows(() => mergeCensus([rows(["a", 1]), undefined], "where"), Error, "1 本目");
  });
});

describe("assertPipelineCensus", () => {
  const census = rows(["linear:a", 3], ["relu", 1]);

  it("本数ちょうど・下限の両方を満たせば通る", () => {
    assertPipelineCensus(
      census,
      [
        { label: "linear:a", match: "linear:a", count: 3 },
        { label: "relu 以上", match: "relu", atLeast: 1 },
        { label: "無いキー", match: "gelu", count: 0 },
      ],
      "where",
    );
  });

  it("本数ちょうどが外れれば落ちる", () => {
    assertThrows(
      () => assertPipelineCensus(census, [{ label: "L", match: "linear:a", count: 2 }], "where"),
      Error,
      "L が 3 本（期待 2 本）",
    );
  });

  it("下限を割れば落ちる", () => {
    assertThrows(
      () => assertPipelineCensus(census, [{ label: "R", match: "relu", atLeast: 2 }], "where"),
      Error,
      "R が 1 本（期待 2 本以上）",
    );
  });

  it("行が undefined / 空なら期待が 0 本だけでも落ちる", () => {
    const zero = [{ label: "Z", match: "x", count: 0 }] as const;
    assertThrows(() => assertPipelineCensus(undefined, zero, "where"), Error, "lastRunPipelines");
    assertThrows(() => assertPipelineCensus([], zero, "where"), Error, "census が空");
  });
});

describe("assertSeatsApplied", () => {
  const parallel = linearGemvParallelKey("i4", 32, 4);
  const sequential = linearGemvKey("i4", 32);

  it("非参照値の席は、その変種が 1 本以上あれば通る", () => {
    assertSeatsApplied(
      rows([parallel, 7], [RMS_NORM_KEY, 2]),
      {
        linearGemvReduce: "parallel",
      },
      "where",
    );
  });

  it("非参照値の席で変種が 1 本も無ければ落ちる（沈黙縮退の検出）", () => {
    assertThrows(
      () => assertSeatsApplied(rows([sequential, 7]), { linearGemvReduce: "parallel" }, "where"),
      Error,
      'linearGemvReduce="parallel" の変種が 1 本も走っていない',
    );
  });

  it("参照値を明示した席で変種が走っていれば落ちる（席が漏れた形）", () => {
    assertThrows(
      () =>
        assertSeatsApplied(
          rows([sequential, 3], [rmsNormAddKey("norm-residual", 1536), 1]),
          { fuseRmsNormAdd: false },
          "where",
        ),
      Error,
      "fuseRmsNormAdd は参照値 false なのに true の変種が走った",
    );
  });

  it("参照値を明示した席は、どの非参照値の変種も無ければ通る", () => {
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

  it("参照値の席は非参照値のどれか 1 つでも走れば落ちる（subgroup32 も parallel 族の変種）", () => {
    assertThrows(
      () =>
        assertSeatsApplied(
          rows([linearGemvSubgroupKey("i8", undefined, 4), 1]),
          { linearGemvReduce: "sequential" },
          "where",
        ),
      Error,
      "parallel-subgroup32 の変種が走った",
    );
  });

  it("渡していない席は検査しない（省略時の実効値は呼び手が決める）", () => {
    assertSeatsApplied(rows([parallel, 1]), { fuseRmsNormAdd: false }, "where");
  });

  it("数値を変えない欄だけの指定でも census が空なら落ちる", () => {
    assertThrows(
      () => assertSeatsApplied([], { planBackingBudgetBytes: 0 }, "where"),
      Error,
      "census が空",
    );
  });
});

/**
 * 融合 attention の完全適格の検査（tests/helpers/pipeline-census.ts の `assertAttentionSeatsComplete`）を、
 * 合成した census の行で固定する（GPU 不要）。キーは src の key 関数から組む — 実行が出すキーと同じ綴りで
 * 検査が当たることの裏取りで、文字列を二重に持たない。
 *
 * 故障注入の役: 宣言どおりの census に縮退した dispatch を 1 本混ぜると赤になる（恒真の検査でないこと）。
 */

const LAYERS = 30;
const A8_S16: SessionOptions = { attentionCompute: "a8", attentionScoreStorage: "f16" };
/** Device facts with subgroups and 48 KiB of workgroup storage (the online kernel's sg32 form). */
const DEVICE: AttentionDeviceFacts = { subgroups: true, maxComputeWorkgroupStorageSize: 49152 };

/**
 * 融合 attention 1 ノード × 行ブロック `blocks` 枚の census の行（①②③ が各 `blocks` 本）。経路は
 * `attentionRoute` の戻りをそのまま使う — 実行（buildAttention）と同じ判定から行を作る。
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
 * a8 + s16 の席で、D = 128 の self（キー長 `selfCols`・行ブロック `blocks` 枚）と cross（N = 512・1 枚）を
 * transformer 30 層ぶん回した census（linear の i8a8 も混ぜる — 族の外の行を数えないことの確認）。
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
  // 族ごとに束ねる（実物の census と同じく、同じキーは 1 行に畳まれる）。
  const counts = new Map<string, number>();
  for (const { key, dispatchCount } of nodeRows) {
    counts.set(key, (counts.get(key) ?? 0) + dispatchCount * LAYERS);
  }
  return [
    { key: linearI8a8Key(true, true), dispatchCount: 300 },
    ...[...counts].map(([key, dispatchCount]) => ({ key, dispatchCount })),
  ];
};

describe("assertAttentionSeatsComplete: 宣言どおりの census", () => {
  it("self / cross とも適格な形（N = 4096 / 512）は、行ブロックの枚数に依らず緑", () => {
    for (const blocks of [1, 3]) {
      assertAttentionSeatsComplete(ditCensus(4096, blocks), A8_S16, `${blocks} 枚`);
    }
  });

  it("a8 / s16 を宣言しない席は、dense の attention があっても見ない", () => {
    const census = attentionRows(
      attentionRoute({ attentionCompute: "f32", attentionScoreStorage: "f32" }, 128, 1089, DEVICE),
      1,
      1089,
    );
    assertAttentionSeatsComplete(census, {}, "席なし");
    assertAttentionSeatsComplete(census, { attentionCompute: "f32" }, "参照値");
  });

  it("a8 だけの席は S の格納を見ない（f32 格納の i8a8 で緑）", () => {
    const census = attentionRows(
      attentionRoute({ attentionCompute: "a8", attentionScoreStorage: "f32" }, 128, 4096, DEVICE),
      2,
      4096,
    );
    assertAttentionSeatsComplete(census, { attentionCompute: "a8" }, "a8 だけ");
  });
});

describe("assertAttentionSeatsComplete: 縮退を含む census（故障注入）", () => {
  it("dense の attention_pv:v1:f32 を 1 本混ぜると、③PV 族の i8a8 の欠けで赤", () => {
    const census: CensusRow[] = [
      ...ditCensus(4096, 1),
      { key: attentionPvKey(true, "f32", "f16"), dispatchCount: 1 },
    ];
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "混入"),
      Error,
      'attentionCompute="a8" の attention_pv が',
    );
  });

  it("②行統計の :s16 が 1 本欠けると、s16 の欠けで赤（a8 は満たしたまま）", () => {
    const census: CensusRow[] = [
      ...ditCensus(4096, 1),
      { key: attentionStatsKey("f32", "f32", 16), dispatchCount: 1 },
    ];
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "s16 の欠け"),
      Error,
      'attentionScoreStorage="f16" の attention_stats が',
    );
  });

  it("self が N % 4 != 0（N = 1089 — Anima の 528×528）の census は赤、assertSeatsApplied は緑のまま", () => {
    const census = ditCensus(1089, 1);
    // 「1 本以上」の検査は cross の i8a8 / s16 だけで満たされる — これが今回塞ぐ穴。
    assertSeatsApplied(census, A8_S16, "1 本以上");
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "self の縮退"),
      Error,
      "f32 へ縮退",
    );
  });

  it("融合 attention が 1 本も無い census は、宣言した族の欠落として赤（0 == 0 で緑にしない）", () => {
    const census: CensusRow[] = [{ key: linearI8a8Key(true, true), dispatchCount: 4 }];
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "attention なし"),
      Error,
      "1 本も無い",
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
