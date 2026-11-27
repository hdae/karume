// The GEMV sweep's `explore` set without a GPU: which cases are sentinels (src/tune/gemv-explore.ts
// — `gemvSentinels`, `gemvExploreOrder`), what another case measures at M = 1
// (`gemvExploreSelection`, `gemvGridNeighbours`), what M = 4 and 8 take from a case's M = 1 group
// (src/tune/gemv-candidates.ts — `gemvExploreCarriedFromM1`) and keep past the quick look
// (`gemvExploreQuickLookGoesOn`), how a sweep of the set is planned
// (src/tune/gemv-sweep.ts — `planGemvSweep`, `gemvSweepQuickLookMargin`, `gemvSweepExploreEvidence`)
// and the quick look's bar of a class among the kernels the derivation could adopt (src/tune/harness.ts —
// `quickLookGoesOn`; src/tune/gemv-sweep.ts — `gemvQuickLookAdoptable`).
//
// Faults injected, one per rule (each checked to turn its case red): the sentinel count moved from
// 12; a tie that goes to the later case; a nearer sentinel without a winner that blocks the farther
// one; a row whose check failed, that was screened, skipped or errored, or whose time is no finite
// number, and that still predicts; a sentinel of another storage that predicts; the parallel
// prediction taken from the fastest row alone, its other rows read against the baseline, or every
// row a winner without the reference's time; the sequential prediction taken from the fastest
// variant alone, its other variants read against the default's choice at reorder, or every variant
// a winner without the baseline's time; the fourth fastest row carried; the subgroup form
// carried; the row-blocked form of a lane outside the three fastest carried; the quick look at M = 4
// left to the margin alone, keeping a row whose row blocks were screened, or keeping the default's
// row; the sentinels left in the table's order; the option error dropped; the explore margin taken
// as ×1.25; a kernel whose check failed, or a subgroup kernel, setting its class's bar.

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  EXPLORE_QUICK_LOOK_MARGIN,
  GEMV_SENTINEL_COUNT,
  gemvCaseDistance,
  type GemvExploreEvidence,
  gemvExploreOrder,
  type GemvExploreRow,
  gemvExploreSelection,
  gemvGridNeighbours,
  gemvSentinels,
  gemvSweepExploreEvidence,
  gemvSweepQuickLookMargin,
  QUICK_LOOK_MARGIN,
} from "../tune.ts";
import {
  type GemvCandidate,
  gemvCandidates,
  gemvCarriedFromM1,
  gemvExploreCarriedFromM1,
  gemvExploreQuickLookGoesOn,
  type GemvGates,
  type GemvM1Group,
  gemvProfileChoice,
  gemvBaseline,
} from "../src/tune/gemv-candidates.ts";
import { GEMV_CASES, type GemvCase, gemvCaseOf } from "../src/tune/gemv-cases.ts";
import { gemvQuickLookAdoptable, planGemvSweep } from "../src/tune/gemv-sweep.ts";
import { quickLookGoesOn, type QuickLookKernel } from "../src/tune/harness.ts";
import { gemvRecord } from "./helpers/gemv-sweep-records.ts";

const caseOf = (id: string): GemvCase => {
  const found = gemvCaseOf(id);
  if (found === undefined) throw new Error(`no case ${id}`);
  return found;
};

/** The sentinels of the current case table, in the order chosen. */
const TABLE_SENTINELS = [
  "i4g32-n2112-k2816",
  "i8-n262144-k256",
  "bf16-n256-k2560",
  "i2-n12288-k1536",
  "i8-n256-k5120",
  "bf16-n262144-k1536",
  "i4g32-n262144-k256",
  "i8-n2048-k256",
  "bf16-n2560-k256",
  "i2-n1536-k12288",
  "bf16-n10752-k2560",
  "i2-n262144-k2560",
];

/** A synthetic i8 case of `n` columns and `k` (a shape only: the exploration reads no table row). */
const shape = (id: string, n: number, k = 1024): GemvCase => ({
  id,
  storage: "i8",
  n,
  k,
  rows: [1],
  source: "test",
});

describe("the sentinels", () => {
  it("of the current case table are these twelve, in this order", () => {
    assertEquals(GEMV_SENTINEL_COUNT, 12);
    assertEquals(gemvSentinels(GEMV_CASES), TABLE_SENTINELS);
  });

  it("are the same whatever a run is narrowed to: the whole table's sentinels among the narrowed cases go first, each part in the order given (fault: the table's order)", () => {
    // Table order: i4g32-n8960-k1536, i8-n256-k5120 (a sentinel), i8-n1024-k256, i8-n2048-k256 (one)
    const narrowed = GEMV_CASES.filter(({ id }) =>
      ["i8-n1024-k256", "i8-n2048-k256", "i4g32-n8960-k1536", "i8-n256-k5120"].includes(id),
    );
    assertEquals(
      narrowed.map(({ id }) => id),
      ["i4g32-n8960-k1536", "i8-n256-k5120", "i8-n1024-k256", "i8-n2048-k256"],
    );
    assertEquals(
      gemvExploreOrder(narrowed).map(({ id }) => id),
      ["i8-n256-k5120", "i8-n2048-k256", "i4g32-n8960-k1536", "i8-n1024-k256"],
    );
    // The sentinels of the narrowed list alone would be another choice: a run takes the table's
    assertEquals(gemvSentinels(narrowed, 2).includes("i8-n2048-k256"), false);
    // Computed from the list alone: the same in every call
    assertEquals(gemvSentinels(GEMV_CASES), gemvSentinels([...GEMV_CASES]));
  });

  it("are every case of a list of at most the count, in its order", () => {
    const list = [shape("a", 256), shape("b", 512)];
    assertEquals(gemvSentinels(list, 2), ["a", "b"]);
    assertEquals(gemvSentinels([], 3), []);
  });

  it("start at the case nearest to all, then take the farthest from those chosen; a tie goes to the earlier case (fault: the later one)", () => {
    const centre = shape("centre", 1024);
    const above = shape("above", 4096);
    const below = shape("below", 256);
    // above and below are both two octaves from the centre
    assertEquals(gemvCaseDistance(centre, above), gemvCaseDistance(centre, below));
    assertEquals(gemvSentinels([centre, above, below], 2), ["centre", "above"]);
    assertEquals(gemvSentinels([centre, below, above], 2), ["centre", "below"]);
    // Two cases equally near all: the earlier is the centre
    assertEquals(gemvSentinels([shape("first", 1024), shape("second", 1024), above], 2), [
      "first",
      "above",
    ]);
  });

  it("stop when only cases of a chosen shape are left", () => {
    const list = [shape("a", 1024), shape("twin", 1024), shape("far", 65536)];
    assertEquals(gemvSentinels([...list, shape("twin2", 1024)], 3), ["a", "far"]);
    assertEquals(gemvSentinels([shape("a", 1024), shape("b", 1024), shape("c", 1024)], 2), ["a"]);
  });

  it("refuse a count that is not an integer of at least 1", () => {
    assertThrows(() => gemvSentinels(GEMV_CASES, 0), Error, "integer of at least 1");
    assertThrows(() => gemvSentinels(GEMV_CASES, 1.5), Error, "integer of at least 1");
  });

  it("put storages further apart than any two shapes of one storage", () => {
    const i8 = shape("i8", 1024);
    const bf16: GemvCase = { ...i8, id: "bf16", storage: "bf16" };
    const farI8 = shape("far", 262144, 16);
    assert(gemvCaseDistance(i8, bf16) > gemvCaseDistance(i8, farI8));
  });
});

describe("the grid neighbours of a candidate", () => {
  it("of a sequential variant are one step of the columns and of the read-ahead, clipped at the grid's edges", () => {
    assertEquals(gemvGridNeighbours("c64u4"), ["c32u4", "c128u4", "c64u2", "c64u8"]);
    assertEquals(gemvGridNeighbours("c16u1"), ["c32u1", "c16u2"]);
    assertEquals(gemvGridNeighbours("c256u16"), ["c128u16", "c256u8"]);
  });

  it("of a parallel row are one step of the lanes and the other of plain and staged, clipped at the lanes' edges", () => {
    assertEquals(gemvGridNeighbours("parallel-l8"), ["parallel-l4", "parallel-l16", "staged-l8"]);
    assertEquals(gemvGridNeighbours("parallel-l2"), ["parallel-l4", "staged-l2"]);
    assertEquals(gemvGridNeighbours("staged-l32"), ["staged-l16", "parallel-l32"]);
  });

  it("are none for any other name", () => {
    for (const name of ["staged-l4r4", "subgroup32-l8", "gemm", "c32u4r2", "c512u4"]) {
      assertEquals(gemvGridNeighbours(name), [], name);
    }
  });
});

/** A timed row of an M = 1 group whose check held (identical class), with `extra` over it. */
const sequential = (
  candidate: string,
  perDispatch: number,
  extra: Partial<GemvExploreRow> = {},
): GemvExploreRow => ({
  candidate,
  class: "identical",
  perDispatch,
  identicalToBaseline: true,
  ...extra,
});

/** A timed row of an M = 1 group whose check held (reorder class), with `extra` over it. */
const parallel = (
  candidate: string,
  perDispatch: number,
  extra: Partial<GemvExploreRow> = {},
): GemvExploreRow => ({
  candidate,
  class: "reorder",
  perDispatch,
  withinBand: true,
  nonFiniteOutputs: 0,
  ...extra,
});

/** A row the quick look screened out: no time. */
const screenedOut = ({ perDispatch: _time, ...row }: GemvExploreRow): GemvExploreRow => ({
  ...row,
  screened: true,
});

/** A row the sweep skipped: no time. */
const skippedOut = ({ perDispatch: _time, ...row }: GemvExploreRow): GemvExploreRow => ({
  ...row,
  skipped: true,
});

describe("the candidates a case that is no sentinel measures at M = 1", () => {
  const target = shape("target", 2048);
  const near = shape("near", 1024);
  const far = shape("far", 16384);
  const rows: GemvExploreRow[] = [
    { candidate: "c32u4", class: "identical", isBaseline: true, perDispatch: 10 },
    sequential("c64u8", 8),
    parallel("parallel-l4", 9),
    parallel("staged-l16", 6),
  ];

  it("are the fastest sequential variant, the fastest plain or staged row and every other one ×1.10 faster than the default's choice at reorder, of the nearest sentinel, and their grid neighbours", () => {
    // The default's choice at reorder of the sentinel is the baseline c32u4: parallel-l4 is ×1.11
    assertEquals(gemvProfileChoice(near, 1, "reorder").name, "c32u4");
    assertEquals(
      gemvExploreSelection(target, [
        { gemvCase: far, rows: [] },
        { gemvCase: near, rows },
      ]),
      [
        "c64u8",
        ...gemvGridNeighbours("c64u8"),
        "staged-l16",
        ...gemvGridNeighbours("staged-l16"),
        "parallel-l4",
        ...gemvGridNeighbours("parallel-l4"),
      ],
    );
  });

  it("carry with the fastest plain or staged row every other one the derivation could adopt at the sentinel, and their neighbours: not one below ×1.10 over the default's choice, nor one whose check failed (fault: the fastest alone)", () => {
    // Against c32u4's 10: staged-l16 ×1.96 and staged-l32 ×2 are both winners; parallel-l2 ×1.05
    // is none, and staged-l2 is outside the band
    const selection = gemvExploreSelection(target, [
      {
        gemvCase: near,
        rows: [
          rows[0],
          sequential("c64u8", 8),
          parallel("staged-l32", 5),
          parallel("staged-l16", 5.1),
          parallel("parallel-l2", 9.5),
          parallel("staged-l2", 5.2, { withinBand: false }),
        ],
      },
    ]);
    assertEquals(selection, [
      "c64u8",
      ...gemvGridNeighbours("c64u8"),
      "staged-l32",
      "staged-l16",
      "parallel-l32",
      // staged-l16's neighbours: two lane steps from staged-l32
      "staged-l8",
      "parallel-l16",
    ]);
  });

  it("read the other plain or staged rows against the sentinel's default choice at reorder, not its baseline (fault: against the baseline)", () => {
    // E2B's per_layer_model_projection: the default's choice at reorder at M = 1 is parallel-l4
    const sentinel = caseOf("i4g32-n8960-k1536");
    assertEquals(gemvProfileChoice(sentinel, 1, "reorder").name, "parallel-l4");
    const i4: GemvCase = { ...shape("target-i4", 4096), storage: "i4", group: 32 };
    const selection = gemvExploreSelection(i4, [
      {
        gemvCase: sentinel,
        rows: [
          { candidate: "c32u4", class: "identical", isBaseline: true, perDispatch: 20 },
          parallel("parallel-l4", 10),
          parallel("staged-l32", 5),
          // ×1.18 over parallel-l4: a winner
          parallel("staged-l16", 8.5),
          // ×1.05 over parallel-l4 (×2.1 over the baseline): none
          parallel("parallel-l2", 9.5),
        ],
      },
    ]);
    assert(selection !== undefined);
    assert(selection.includes("staged-l8"), selection.join());
    assertEquals(
      ["parallel-l2", "staged-l2"].filter((name) => selection.includes(name)),
      [],
    );
  });

  it("carry the fastest plain or staged row alone when the default's choice at reorder has no time at the sentinel (fault: every row a winner)", () => {
    const selection = gemvExploreSelection(target, [
      {
        gemvCase: near,
        rows: [sequential("c64u8", 8), parallel("staged-l32", 5), parallel("staged-l16", 5.1)],
      },
    ]);
    assertEquals(selection, [
      "c64u8",
      ...gemvGridNeighbours("c64u8"),
      "staged-l32",
      ...gemvGridNeighbours("staged-l32"),
    ]);
  });

  it("carry with the fastest sequential variant every other one the derivation could adopt at the sentinel, and their neighbours (fault: the fastest alone)", () => {
    // The default's choice at exact of the sentinel is the baseline c32u4: against its 10, c64u8
    // ×1.25 and c16u16 ×1.18 are both winners
    assertEquals(gemvBaseline(near, 1).name, "c32u4");
    const selection = gemvExploreSelection(target, [
      {
        gemvCase: near,
        rows: [
          rows[0],
          sequential("c64u8", 8),
          sequential("c16u16", 8.5),
          parallel("staged-l16", 6),
        ],
      },
    ]);
    assertEquals(selection, [
      "c64u8",
      ...gemvGridNeighbours("c64u8"),
      "c16u16",
      ...gemvGridNeighbours("c16u16"),
      "staged-l16",
      ...gemvGridNeighbours("staged-l16"),
    ]);
  });

  it("keep a variant the derivation could adopt when one fast time leads it at the sentinel: an RTX 3080 Ti's explore run of 2026-11-27 (fault: the fastest alone)", () => {
    // The i4 sentinel's M = 1 times of that run, in µs: c16u8's 20.58 is one fast time (22.5 to
    // 22.7 in five other records); c32u16 and c16u16 are ×1.12 over the baseline, c32u8 ×1.05
    const sentinel = caseOf("i4g32-n2112-k2816");
    assertEquals(gemvBaseline(sentinel, 1).name, "c32u4");
    const selection = gemvExploreSelection(caseOf("i4g32-n8192-k3840"), [
      {
        gemvCase: sentinel,
        rows: [
          { candidate: "c32u4", class: "identical", isBaseline: true, perDispatch: 23.69 },
          sequential("c16u8", 20.58),
          sequential("c32u16", 21.17),
          sequential("c16u16", 21.21),
          sequential("c32u8", 22.58),
          parallel("staged-l32", 15),
        ],
      },
    ]);
    assert(selection !== undefined);
    assertEquals(
      selection.filter((name) => !name.includes("-l")),
      [
        "c16u8",
        // c16u8's neighbours: c32u8 is one, not a winner of its own
        "c32u8",
        "c16u4",
        "c16u16",
        "c32u16",
        // The one neighbour of c32u16 that no other winner has
        "c64u16",
      ],
    );
  });

  it("read the other sequential variants against the sentinel's default choice at exact, not at reorder (fault: against the choice at reorder)", () => {
    // E2B's per_layer_model_projection: the default's choice at reorder at M = 1 is parallel-l4,
    // at exact the baseline c32u4
    const sentinel = caseOf("i4g32-n8960-k1536");
    assertEquals(
      [gemvBaseline(sentinel, 1).name, gemvProfileChoice(sentinel, 1, "reorder").name],
      ["c32u4", "parallel-l4"],
    );
    const i4: GemvCase = { ...shape("target-i4", 4096), storage: "i4", group: 32 };
    const selection = gemvExploreSelection(i4, [
      {
        gemvCase: sentinel,
        rows: [
          { candidate: "c32u4", class: "identical", isBaseline: true, perDispatch: 20 },
          parallel("parallel-l4", 10),
          sequential("c64u8", 15),
          // ×1.18 over the baseline, slower than parallel-l4: a winner
          sequential("c16u16", 17),
          parallel("staged-l32", 5),
        ],
      },
    ]);
    assert(selection !== undefined);
    assert(selection.includes("c16u16") && selection.includes("c16u8"), selection.join());
  });

  it("carry the fastest sequential variant alone when the baseline has no time at the sentinel (fault: every variant a winner)", () => {
    for (const reference of [
      [],
      [{ ...rows[0], error: "Error: lost" }] satisfies GemvExploreRow[],
    ]) {
      const selection = gemvExploreSelection(target, [
        {
          gemvCase: near,
          rows: [
            ...reference,
            sequential("c64u8", 8),
            sequential("c16u16", 8.1),
            parallel("staged-l16", 6),
          ],
        },
      ]);
      assertEquals(selection, [
        "c64u8",
        ...gemvGridNeighbours("c64u8"),
        "staged-l16",
        ...gemvGridNeighbours("staged-l16"),
      ]);
    }
  });

  it("leave out a sequential variant less than ×1.10 faster than the baseline, or whose words differ, however near the fastest (fault: every variant a winner)", () => {
    // Against c32u4's 10: c16u2 is ×1.09, c128u1 differs from the baseline's words
    const selection = gemvExploreSelection(target, [
      {
        gemvCase: near,
        rows: [
          rows[0],
          sequential("c64u8", 8),
          sequential("c128u1", 8.2, { identicalToBaseline: false }),
          sequential("c16u2", 9.2),
          parallel("staged-l16", 6),
        ],
      },
    ]);
    assertEquals(selection, [
      "c64u8",
      ...gemvGridNeighbours("c64u8"),
      "staged-l16",
      ...gemvGridNeighbours("staged-l16"),
    ]);
  });

  it("take the baseline as a sequential winner: its words are the reference", () => {
    const selection = gemvExploreSelection(target, [
      { gemvCase: near, rows: [rows[0], parallel("parallel-l4", 9)] },
    ]);
    assertEquals(selection?.[0], "c32u4");
  });

  it("are predicted from the nearest sentinel that has a winner of the class (fault: a nearer one without any)", () => {
    const evidence: GemvExploreEvidence[] = [
      // Nearest, but every sequential row screened: it predicts the parallel row only
      { gemvCase: near, rows: [screenedOut(sequential("c64u8", 8)), parallel("staged-l8", 5)] },
      { gemvCase: far, rows: [sequential("c128u2", 7), parallel("parallel-l32", 3)] },
    ];
    const selection = gemvExploreSelection(target, evidence);
    assert(selection !== undefined);
    assertEquals(selection[0], "c128u2");
    assert(
      selection.includes("staged-l8") && !selection.includes("parallel-l32"),
      selection.join(),
    );
  });

  it("are predicted from the first of equally near sentinels", () => {
    const below = shape("below", 1024);
    const above = shape("above", 4096);
    const selection = gemvExploreSelection(target, [
      { gemvCase: below, rows: [sequential("c16u2", 5), parallel("parallel-l2", 5)] },
      { gemvCase: above, rows: [sequential("c256u16", 5), parallel("staged-l32", 5)] },
    ]);
    assertEquals(selection?.[0], "c16u2");
  });

  /** The selection when the group's fastest sequential row is `fastestSequential` and its fastest parallel row `fastestParallel`. */
  const withFastest = (fastestSequential: GemvExploreRow, fastestParallel: GemvExploreRow) => {
    const selection = gemvExploreSelection(target, [
      { gemvCase: near, rows: [...rows, fastestSequential, fastestParallel] },
    ]);
    assert(selection !== undefined);
    return selection;
  };

  it("take the fastest rows when their checks held (the control of the faults below)", () => {
    const selection = withFastest(sequential("c16u1", 1), parallel("parallel-l32", 1));
    assert(selection.includes("c16u1") && selection.includes("parallel-l32"), selection.join());
  });

  for (const [mark, row] of [
    [
      "whose words differ from the baseline's",
      sequential("c16u1", 1, { identicalToBaseline: false }),
    ],
    ["that failed", sequential("c16u1", 1, { error: "Error: lost" })],
    ["screened by the quick look", screenedOut(sequential("c16u1", 1))],
    ["skipped", skippedOut(sequential("c16u1", 1))],
  ] as const) {
    it(`leave a sequential row ${mark} out of the prediction: the next fastest predicts (fault: that row predicts)`, () => {
      const selection = withFastest(row, parallel("parallel-l32", 1));
      assertEquals(selection.includes("c16u1"), false, selection.join());
      assertEquals(selection[0], "c64u8");
    });
  }

  for (const [mark, row] of [
    ["outside the band", parallel("parallel-l32", 1, { withinBand: false })],
    ["with a non-finite output", parallel("parallel-l32", 1, { nonFiniteOutputs: 1 })],
    ["that failed", parallel("parallel-l32", 1, { error: "Error: lost" })],
    ["screened by the quick look", screenedOut(parallel("parallel-l32", 1))],
    ["skipped", skippedOut(parallel("parallel-l32", 1))],
  ] as const) {
    it(`leave a parallel row ${mark} out of the prediction: the next fastest predicts (fault: that row predicts)`, () => {
      const selection = withFastest(sequential("c16u1", 1), row);
      assertEquals(selection.includes("parallel-l32"), false, selection.join());
      assert(selection.includes("staged-l16"), selection.join());
    });
  }

  /** The same row through JSON, as a record read back holds it (a NaN time becomes null). */
  const roundTripped = (row: GemvExploreRow): GemvExploreRow => JSON.parse(JSON.stringify(row));

  for (const [mark, row] of [
    ["NaN", sequential("c16u1", Number.NaN)],
    ["null after a JSON round trip", roundTripped(sequential("c16u1", Number.NaN))],
    ["absent", (({ perDispatch: _time, ...rest }) => rest)(sequential("c16u1", 1))],
  ] as const) {
    it(`leave a row whose time is ${mark} out of the prediction, even as the first row read (fault: only an absent time is none)`, () => {
      const selection = gemvExploreSelection(target, [
        { gemvCase: near, rows: [row, sequential("c64u8", 8), parallel("staged-l16", 6)] },
      ]);
      assertEquals(selection?.[0], "c64u8");
    });
  }

  it("are predicted only from sentinels of the case's storage: a sentinel of another storage predicts nothing (fault: it predicts the class the nearer one lacks)", () => {
    const bf16: GemvCase = { ...shape("bf16", 2048), storage: "bf16" };
    const winners = [sequential("c128u2", 7), parallel("parallel-l32", 3)];
    assertEquals(gemvExploreSelection(target, [{ gemvCase: bf16, rows: winners }]), undefined);
    // The case's storage has a parallel winner only: the other storage does not fill in the rest
    assertEquals(
      gemvExploreSelection(target, [
        { gemvCase: bf16, rows: winners },
        { gemvCase: near, rows: [parallel("staged-l8", 5)] },
      ]),
      undefined,
    );
    // Control: the same rows on a sentinel of the case's storage predict
    assertEquals(gemvExploreSelection(target, [{ gemvCase: far, rows: winners }])?.[0], "c128u2");
  });

  it("are predicted from a sentinel of the same storage whatever its i4 group length", () => {
    const i4 = (id: string, group: number): GemvCase => ({
      ...shape(id, 2048),
      id,
      storage: "i4",
      group,
    });
    const selection = gemvExploreSelection(i4("target-g32", 32), [
      {
        gemvCase: i4("sentinel-g128", 128),
        rows: [sequential("c16u2", 5), parallel("parallel-l2", 5)],
      },
    ]);
    assertEquals(selection?.[0], "c16u2");
  });

  it("are undefined — every candidate — when no sentinel has a winner of one of the classes", () => {
    assertEquals(gemvExploreSelection(target, []), undefined);
    assertEquals(
      gemvExploreSelection(target, [
        { gemvCase: near, rows: rows.filter(({ class: one }) => one === "identical") },
      ]),
      undefined,
    );
    // A subgroup row is no parallel row a table holds: it predicts nothing
    assertEquals(
      gemvExploreSelection(target, [
        { gemvCase: near, rows: [sequential("c64u8", 8), parallel("subgroup32-l8", 1)] },
      ]),
      undefined,
    );
  });
});

/** E2B's per_layer_model_projection: the default's choice at reorder is parallel-l4 at M = 1, staged-l4r4 at M = 4. */
const PLE = caseOf("i4g32-n8960-k1536");
const GATES: GemvGates = { subgroups: true, maxComputeWorkgroupsPerDimension: 65535 };

describe("what the explore set carries from a case's M = 1 group to M = 4 and 8 (gemvExploreCarriedFromM1)", () => {
  const references = {
    exact: gemvBaseline(PLE, 4).name,
    reorder: gemvProfileChoice(PLE, 4, "reorder").name,
  };
  const row = (candidate: string, perDispatch: number) => ({
    candidate,
    perDispatch,
    withinBand: true,
    nonFiniteOutputs: 0,
  });
  // Steady; the default's choice at 10 µs: every row but staged-l2 (×1.05) is adoptable
  const group: GemvM1Group = {
    baselineSpread: 1.01,
    profile: { reorder: "parallel-l4" },
    rows: [
      row("parallel-l4", 10_000),
      row("parallel-l8", 5_000),
      row("staged-l8", 6_000),
      row("staged-l16", 7_000),
      row("parallel-l16", 8_000),
      row("subgroup32-l8", 4_000),
      row("staged-l2", 9_500),
    ],
  };
  const candidateAt = (m: number, name: string): GemvCandidate => {
    const found = gemvCandidates(PLE, m, "explore", GATES).find((one) => one.name === name);
    if (found === undefined) throw new Error(`no candidate ${name} at M=${m}`);
    return found;
  };
  const carried = (name: string, first: GemvM1Group = group): boolean =>
    gemvExploreCarriedFromM1(4, candidateAt(4, name), references, first);
  const carriedByFull = (name: string): boolean =>
    gemvCarriedFromM1(4, candidateAt(4, name), references, group);

  it("carries the three fastest plain or staged rows M = 1 leaves adoptable, not the fourth (fault: four)", () => {
    for (const name of ["parallel-l8", "staged-l8", "staged-l16"]) assert(carried(name), name);
    assert(carriedByFull("parallel-l16"), "the full set's rule carries the fourth");
    assertEquals(carried("parallel-l16"), false);
    // An unadoptable row is no candidate of the three, and a row M = 1 did not measure neither
    assertEquals([carried("staged-l2"), carried("parallel-l32")], [false, false]);
    assert(carriedByFull("parallel-l32"), "the full set's rule measures a row M = 1 has none of");
  });

  it("carries the row-blocked forms of their lanes and of the default's lanes, and no other (fault: lanes 2 and 32)", () => {
    for (const name of [
      "staged-l8r2",
      "staged-l8r4",
      "staged-l16r2",
      "staged-l16r4",
      "staged-l4r2",
    ]) {
      assert(carried(name), name);
    }
    for (const name of ["staged-l2r2", "staged-l2r4", "staged-l32r2"]) {
      assert(carriedByFull(name), `the full set's rule carries ${name}`);
      assertEquals(carried(name), false, name);
    }
  });

  it("does not carry the subgroup form, the fastest row of the group (fault: carried as a row)", () => {
    assert(carriedByFull("subgroup32-l8"));
    assertEquals(carried("subgroup32-l8"), false);
    assertEquals(carried("subgroup32-l4"), false);
  });

  it("always measures the references at the row count and the default's choice at M = 1 (fault: their M = 1 rows rule them out)", () => {
    const slow: GemvM1Group = {
      ...group,
      rows: group.rows.map((one) =>
        one.candidate === "parallel-l4" ? { ...one, withinBand: false } : one,
      ),
    };
    for (const name of [references.reorder, "parallel-l4"]) {
      assert(carried(name, slow), name);
    }
    assertEquals(references, { exact: "c32u4r2", reorder: "staged-l4r4" });
  });

  it("measures every candidate without the run's M = 1 group, and at a row count that is not read for a regression", () => {
    for (const name of ["parallel-l16", "subgroup32-l8", "staged-l32r2"]) {
      assert(gemvExploreCarriedFromM1(4, candidateAt(4, name), references, undefined), name);
    }
    const sixteen = gemvCandidates(PLE, 16, "row-blocks", GATES)[0];
    assert(sixteen !== undefined);
    assert(gemvExploreCarriedFromM1(16, sixteen, references, group));
  });
});

describe("what the explore set's quick look keeps at M = 4 and 8 besides its margin (gemvExploreQuickLookGoesOn)", () => {
  const row = (candidate: string, perDispatch: number) => ({
    candidate,
    perDispatch,
    withinBand: true,
    nonFiniteOutputs: 0,
  });
  // Steady; against the default's parallel-l4 the three fastest adoptable rows are parallel-l8,
  // staged-l8 and staged-l16
  const group: GemvM1Group = {
    baselineSpread: 1.01,
    profile: { reorder: "parallel-l4" },
    rows: [
      row("parallel-l4", 10_000),
      row("parallel-l8", 5_000),
      row("staged-l8", 6_000),
      row("staged-l16", 7_000),
      row("parallel-l16", 8_000),
    ],
  };
  const candidateAt = (m: number, name: string): GemvCandidate => {
    const found = gemvCandidates(PLE, m, "explore", GATES).find((one) => one.name === name);
    if (found === undefined) throw new Error(`no candidate ${name} at M=${m}`);
    return found;
  };
  // What the margin lets go on at M = 4: the reference staged-l4r4, staged-l8r4 and staged-l4r2;
  // every unblocked row is far slower than the reference, and staged-l16r4 is screened too
  const byMargin: readonly [string, boolean][] = [
    ["staged-l4r4", true],
    ["parallel-l4", false],
    ["parallel-l8", false],
    ["staged-l8", false],
    ["staged-l16", false],
    ["staged-l8r4", true],
    ["staged-l16r4", false],
    ["staged-l4r2", true],
  ];
  const goesOnAt = (m: number, first: GemvM1Group | undefined) =>
    Object.fromEntries(
      gemvExploreQuickLookGoesOn(
        m,
        byMargin.map(([name]) => candidateAt(m, name)),
        byMargin.map(([, goes]) => goes),
        first,
      ).map((goes, index) => [byMargin[index]?.[0], goes]),
    );

  it("keeps a row M = 1 carries for its own time whenever a row-blocked form of its lanes goes on: the row block's base (fault: the margin alone)", () => {
    const kept = goesOnAt(4, group);
    assertEquals([kept["parallel-l8"], kept["staged-l8"]], [true, true]);
    // The row-blocked forms and the reference keep what the margin said
    assertEquals(
      [kept["staged-l4r4"], kept["staged-l8r4"], kept["staged-l16r4"], kept["staged-l4r2"]],
      [true, true, false, true],
    );
  });

  it("does not keep a row whose row-blocked forms are screened out themselves: nothing to read it against (fault: kept for a carried row block)", () => {
    assertEquals(goesOnAt(4, group)["staged-l16"], false);
  });

  it("does not keep the default's choice at reorder at M = 1, even where M = 1 carries it among its three fastest: the derivation reads it as the reference, never as a candidate (fault: kept for the reference's row block)", () => {
    // An unsteady group rules nothing out by its times: the default's parallel-l4, the fastest
    // row, is one of the three M = 1 carries (in a steady group its gain over itself rules it out)
    const unsteady: GemvM1Group = {
      ...group,
      baselineSpread: 1.5,
      rows: group.rows.map((one) =>
        one.candidate === "parallel-l4" ? { ...one, perDispatch: 4_000 } : one,
      ),
    };
    const kept = goesOnAt(4, unsteady);
    assertEquals([kept["parallel-l4"], kept["parallel-l8"]], [false, true]);
  });

  it("keeps nothing beyond the margin at M = 1, or without the run's M = 1 group", () => {
    const margin = Object.fromEntries(byMargin);
    assertEquals(goesOnAt(8, group), { ...margin, "parallel-l8": true, "staged-l8": true });
    assertEquals(goesOnAt(4, undefined), margin);
    const one = ["parallel-l4", "parallel-l8", "staged-l8"].map((name) => candidateAt(1, name));
    assertEquals(gemvExploreQuickLookGoesOn(1, one, [true, false, false], group), [
      true,
      false,
      false,
    ]);
  });
});

describe("the explore set's measurement plans", () => {
  it("measure M = 1 with the whole grid and no identical row-block height at M = 4 and 8", () => {
    const one = gemvCandidates(PLE, 1, "explore", GATES).map(({ name }) => name);
    assertEquals(
      one,
      gemvCandidates(PLE, 1, "full", GATES).map(({ name }) => name),
    );
    const four = gemvCandidates(PLE, 4, "explore", GATES);
    assert(
      four.every((candidate) => candidate.class === "reorder"),
      "an identical candidate at M = 4",
    );
  });
});

describe("the planning of an explore sweep (planGemvSweep, before the GPU is touched)", () => {
  const SENTINEL = "i8-n2048-k256";
  const OTHER = "i8-n1536-k256";
  const LATER = "i4g32-n8960-k1536";

  it("measures the sentinels first, then the others, each part in the table's order (fault: the table's order)", () => {
    const plan = planGemvSweep({ candidateSet: "explore", cases: [OTHER, SENTINEL, LATER] });
    assertEquals(
      plan.cases.map(({ id }) => id),
      [SENTINEL, LATER, OTHER],
    );
    // The record's settings.cases: the measuring order, as the workbench's merged record holds them
    assertEquals(plan.caseIds, [SENTINEL, LATER, OTHER]);
    assertEquals(plan.sentinels, TABLE_SENTINELS);
    // Another set keeps the table's order and names no sentinel
    const full = planGemvSweep({ candidateSet: "full", cases: [OTHER, SENTINEL, LATER] });
    assertEquals(
      full.cases.map(({ id }) => id),
      [LATER, SENTINEL, OTHER],
    );
    assertEquals(full.caseIds, [LATER, SENTINEL, OTHER]);
    assertEquals(full.sentinels, undefined);
  });

  it("refuses candidates with the explore set (fault: the option taken)", () => {
    assertThrows(
      () => planGemvSweep({ candidateSet: "explore", candidates: ["c64u8"] }),
      Error,
      "The explore set chooses its candidates",
    );
    assertEquals(planGemvSweep({ candidateSet: "explore", candidates: [] }).selection, {});
  });

  it("refuses sentinel evidence with another set, for a case that is no sentinel, twice, for a case the sweep measures itself, or with another shape", () => {
    const evidenceOf = (id: string): GemvExploreEvidence => ({ gemvCase: caseOf(id), rows: [] });
    assertThrows(
      () => planGemvSweep({ candidateSet: "full", sentinelEvidence: [evidenceOf(SENTINEL)] }),
      Error,
      "read by the explore set only",
    );
    assertThrows(
      () =>
        planGemvSweep({
          candidateSet: "explore",
          cases: [OTHER],
          sentinelEvidence: [evidenceOf(LATER)],
        }),
      Error,
      "no sentinel of the case table",
    );
    assertThrows(
      () =>
        planGemvSweep({
          candidateSet: "explore",
          cases: [OTHER],
          sentinelEvidence: [evidenceOf(SENTINEL), evidenceOf(SENTINEL)],
        }),
      Error,
      "given twice",
    );
    assertThrows(
      () =>
        planGemvSweep({
          candidateSet: "explore",
          cases: [SENTINEL],
          sentinelEvidence: [evidenceOf(SENTINEL)],
        }),
      Error,
      "this sweep measures itself",
    );
    assertThrows(
      () =>
        planGemvSweep({
          candidateSet: "explore",
          cases: [OTHER],
          sentinelEvidence: [
            { gemvCase: { ...caseOf(SENTINEL), n: caseOf(SENTINEL).n * 2 }, rows: [] },
          ],
        }),
      Error,
      "another shape",
    );
    const plan = planGemvSweep({
      candidateSet: "explore",
      cases: [OTHER],
      sentinelEvidence: [evidenceOf(SENTINEL)],
    });
    assertEquals(
      plan.sentinelEvidence.map(({ gemvCase }) => gemvCase.id),
      [SENTINEL],
    );
  });

  it("takes ×1.10 as the explore set's default margin and ×1.25 as every other set's; a margin given wins (fault: ×1.25 for explore)", () => {
    assertEquals(EXPLORE_QUICK_LOOK_MARGIN, 1.1);
    assertEquals(
      planGemvSweep({ candidateSet: "explore" }).quickLookMargin,
      EXPLORE_QUICK_LOOK_MARGIN,
    );
    for (const set of ["quick", "full", "row-blocks"] as const) {
      assertEquals(planGemvSweep({ candidateSet: set }).quickLookMargin, QUICK_LOOK_MARGIN, set);
    }
    assertEquals(
      planGemvSweep({ candidateSet: "explore", quickLookMargin: 1.25 }).quickLookMargin,
      1.25,
    );
    assertEquals(gemvSweepQuickLookMargin("explore", 0), 0);
    assertThrows(() => gemvSweepQuickLookMargin("explore", 0.9), Error, "quickLookMargin");
  });
});

describe("the sentinel evidence of a record (gemvSweepExploreEvidence)", () => {
  it("is each of its sentinels with the rows of its M = 1 group, and nothing for a case that is no sentinel or a record of another set", () => {
    const record = gemvRecord("2026-11-26T00:00:00.000Z", [
      { caseId: "i8-n2048-k256", rows: [1, 4] },
      { caseId: "i8-n1536-k256", rows: [1] },
    ]);
    assertEquals(gemvSweepExploreEvidence(record), []);
    const explore = {
      ...record,
      settings: {
        ...record.settings,
        candidateSet: "explore" as const,
        sentinels: TABLE_SENTINELS,
      },
    };
    const evidence = gemvSweepExploreEvidence(explore);
    assertEquals(
      evidence.map(({ gemvCase }) => gemvCase.id),
      ["i8-n2048-k256"],
    );
    const [first] = evidence;
    assert(first !== undefined);
    const m1 = record.rows.filter((row) => row.caseId === "i8-n2048-k256" && row.m === 1);
    assert(m1.length > 0 && m1.length < record.rows.length);
    assertEquals(first.rows, m1);
    // Its case is the table's: a later sweep takes it as evidence
    assertEquals(
      planGemvSweep({
        candidateSet: "explore",
        cases: ["i8-n1536-k256"],
        sentinelEvidence: evidence,
      }).sentinelEvidence.length,
      1,
    );
  });
});

describe("the quick look's bar of a class (quickLookGoesOn)", () => {
  const kernel = (
    name: string,
    perDispatch: number,
    adoptable: boolean | undefined,
    reference = false,
  ): QuickLookKernel => ({
    name,
    class: "identical",
    perDispatch,
    reference,
    ...(adoptable === undefined ? {} : { adoptable }),
  });
  const fastest = () => ({ against: "fastest" }) as const;

  it("is the fastest of the kernels whose check held: a faster kernel whose check failed does not screen out the best that passed (fault: it sets the bar)", () => {
    const kernels = [
      kernel("baseline", 10, true, true),
      kernel("wrong", 5, false),
      kernel("best", 9, true),
      kernel("slow", 11.5, true),
    ];
    assertEquals(quickLookGoesOn(kernels, 1.25, fastest), [true, true, true, false]);
    // The control: the same times with every check held — the failing kernel's time would be the bar
    assertEquals(
      quickLookGoesOn(
        kernels.map((one) => ({ ...one, adoptable: true })),
        1.25,
        fastest,
      ),
      [true, true, false, false],
    );
  });

  it("is none for a class whose kernels with a time all failed their check: every kernel goes on", () => {
    const kernels = [kernel("a", 5, false), kernel("b", 50, false)];
    assertEquals(quickLookGoesOn(kernels, 1.25, fastest), [true, true]);
  });

  it("is the reference's time where the class is read against it, whatever the reference's check", () => {
    const kernels = [
      kernel("reference", 10, false, true),
      kernel("near", 12, true),
      kernel("far", 13, true),
    ];
    assertEquals(
      quickLookGoesOn(kernels, 1.25, () => ({ against: "reference", reference: "reference" })),
      [true, true, false],
    );
  });

  it("leaves the subgroup form out of the GEMV sweep's bar: a faster subgroup kernel does not screen out a plain row, and is itself screened against the bar (fault: the subgroup form adoptable)", () => {
    const named = (name: string): GemvCandidate => {
      const found = gemvCandidates(PLE, 1, "full", GATES).find((one) => one.name === name);
      if (found === undefined) throw new Error(`no candidate ${name}`);
      return found;
    };
    const reorderKernel = (
      candidate: GemvCandidate,
      perDispatch: number,
      reference = false,
    ): QuickLookKernel => ({
      name: candidate.name,
      class: candidate.class,
      perDispatch,
      reference,
      adoptable: gemvQuickLookAdoptable(candidate, true),
    });
    const kernels = [
      reorderKernel(named("parallel-l4"), 10, true),
      reorderKernel(named("subgroup32-l8"), 5),
      reorderKernel(named("parallel-l8"), 7),
      reorderKernel(named("subgroup32-l4"), 9),
      reorderKernel(named("staged-l16"), 9.5),
    ];
    assertEquals(
      kernels.map(({ adoptable }) => adoptable),
      [true, false, true, false, true],
    );
    // The bar is parallel-l8's 7: the slower subgroup kernel is screened, the faster goes on
    assertEquals(quickLookGoesOn(kernels, 1.25, fastest), [true, true, true, false, false]);
    // The control: with the subgroup form setting the bar (5), parallel-l8 would be screened out
    assertEquals(
      quickLookGoesOn(
        kernels.map((one) => ({ ...one, adoptable: true })),
        1.25,
        fastest,
      ),
      [true, true, false, false, false],
    );
    // A check that failed leaves any form out of the bar
    assertEquals(gemvQuickLookAdoptable(named("parallel-l8"), false), false);
  });

  it("reads every kernel's time where no check is read before the screening (the geometry sweep)", () => {
    const kernels = [
      kernel("default", 10, undefined, true),
      kernel("fast", 5, undefined),
      kernel("mid", 9, undefined),
    ];
    assertEquals(quickLookGoesOn(kernels, 1.25, fastest), [true, true, false]);
  });
});
