// census 共通ヘルパ（tests/helpers/pipeline-census.ts）の純関数としての振る舞い。GPU は使わない —
// 行は偽物で、述語の相手は src の key 関数が返す実物のキー。

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  ATTENTION_STATS_KEY,
  attentionPvKey,
  attentionQkKey,
  attentionStatsKey,
} from "../src/kernels/attention.ts";
import { attentionPvI8a8Key, attentionQkI8a8Key } from "../src/kernels/attention-i8a8.ts";
import { linearKey } from "../src/kernels/linear.ts";
import {
  linearGemvKey,
  linearGemvParallelKey,
  linearGemvParallelPackedKey,
  linearGemvStaticQuantizeKey,
  linearGemvStaticQuantizePackedKey,
  linearGemvSubgroupKey,
} from "../src/kernels/linear-gemv.ts";
import { linearI8a8Key } from "../src/kernels/linear-i8a8.ts";
import { QUANTIZE_ROWS_KEY } from "../src/kernels/quantize-rows.ts";
import { RMS_NORM_KEY, rmsNormAddKey } from "../src/kernels/rms-norm.ts";
import { rmsNormSubgroupKey } from "../src/kernels/rms-norm-subgroup.ts";
import {
  statePvKey,
  statePvParallelKey,
  statePvParallelReadonlyKey,
  stateQkKey,
  stateQkParallelKey,
  stateQkParallelReadonlyKey,
  stateStatsKey,
} from "../src/kernels/state-attention.ts";
import { stateStatsPvKey } from "../src/kernels/state-attention-stats-pv.ts";
import { STATIC_QUANTIZE_KEY, STATIC_QUANTIZE_PACKED_KEY } from "../src/kernels/static-quantize.ts";
import {
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
  attentionCompute: {
    f16: [attentionQkKey(true, "f16"), attentionPvKey(false, "f16"), attentionStatsKey("f16")],
    a8: [attentionQkI8a8Key(true, false), attentionPvI8a8Key(false, true, "f16")],
  },
  attentionScoreStorage: {
    f16: [
      attentionQkKey(true, "f32", "f16", true),
      attentionStatsKey("f32", "f16", 4),
      attentionPvI8a8Key(true, true, "f16"),
    ],
  },
  stateAttentionReduce: {
    parallel: [
      stateQkParallelKey(false, true),
      statePvParallelKey(true, false),
      stateQkParallelReadonlyKey(true, true),
      statePvParallelReadonlyKey(false, false),
    ],
    "parallel-fused": [stateStatsPvKey(false, false), stateStatsPvKey(true, true)],
  },
  linearGemvReduce: {
    parallel: [linearGemvParallelKey("i4", 32, 4), linearGemvStaticQuantizeKey("i2", undefined, 8)],
    "parallel-subgroup32": [linearGemvSubgroupKey("i8", undefined, 16)],
  },
  fuseRmsNormAdd: {
    true: [rmsNormAddKey("residual-norm"), rmsNormSubgroupKey("norm-residual")],
  },
  fuseLinearStaticQuantize: {
    true: [
      linearGemvStaticQuantizeKey("i4", 512, 4),
      linearGemvStaticQuantizePackedKey("i8", undefined, 2),
    ],
  },
  packedStaticQuantize: {
    true: [STATIC_QUANTIZE_PACKED_KEY, linearGemvParallelPackedKey("i4", 32, 8)],
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
    Object.entries(variants).map(([value, predicate]) => [seat, value, predicate] as const)
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
    assertEquals(countDispatches(census, (key) => key.startsWith("linear:")), 5);
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
    assertPipelineCensus(census, [
      { label: "linear:a", match: "linear:a", count: 3 },
      { label: "relu 以上", match: "relu", atLeast: 1 },
      { label: "無いキー", match: "gelu", count: 0 },
    ], "where");
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
    assertSeatsApplied(rows([parallel, 7], [RMS_NORM_KEY, 2]), {
      linearGemvReduce: "parallel",
    }, "where");
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
          rows([sequential, 3], [rmsNormAddKey("norm-residual"), 1]),
          { fuseRmsNormAdd: false },
          "where",
        ),
      Error,
      "fuseRmsNormAdd は参照値 false なのに true の変種が走った",
    );
  });

  it("参照値を明示した席は、どの非参照値の変種も無ければ通る", () => {
    assertSeatsApplied(rows([sequential, 3], [RMS_NORM_KEY, 1]), {
      linearGemvReduce: "sequential",
      fuseRmsNormAdd: false,
      rmsNormReduce: "workgroup",
    }, "where");
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
