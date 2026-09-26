/**
 * census（席を指定したとき変種が**実際に**走った本数の検査 — ADR 0058 決定 4 ③ / ADR 0110 決定 5）の
 * 共通ヘルパ。
 *
 * 源は `SessionDiagnostics.lastRunPipelines`（直近 run の**計画上の** dispatch 本数 — 計測に依らず
 * 常駐経路でも埋まる）。計測表（`lastRunTiming`）を源にすると timestamp-query の無い機・計測が
 * device ごと落ちる機（Metal の query set 上限）で census が丸ごと SKIP になり、席の沈黙縮退を
 * 検出する門がその機では 1 本も無くなる。
 * NOTE: states 形（論理長から workgroup 数を算出する dispatch）は仕事量ゼロの step でも 1 本と
 * 数える（計画の census であって発行の census ではない — src/runtime/recipe.ts `pipelineCensus`）。
 *
 * 述語の綴りは src の key 関数 / 定数から組む（ここで文字列を二重に持つと、キーの改名で述語だけが
 * 空振りして census が恒真になる）。判別断片（`:c16` / `:subgroup32` / `i8a8` など）は複数の族で
 * 共有されるので、述語は必ず**族（キー先頭の `:` より前）との組**で書く。
 */

import { assert } from "@std/assert";
import {
  ATTENTION_STATS_KEY,
  attentionPvKey,
  attentionQkKey,
} from "../../src/kernels/attention.ts";
import { attentionPvI8a8Key, attentionQkI8a8Key } from "../../src/kernels/attention-i8a8.ts";
import { gemmComputeKeyPart } from "../../src/kernels/gemm.ts";
import { linearKey } from "../../src/kernels/linear.ts";
import {
  linearGemvParallelKey,
  linearGemvParallelPackedKey,
  linearGemvStaticQuantizeKey,
  linearGemvSubgroupKey,
} from "../../src/kernels/linear-gemv.ts";
import { linearI8a8Key } from "../../src/kernels/linear-i8a8.ts";
import { RMS_NORM_KEY, rmsNormAddKey } from "../../src/kernels/rms-norm.ts";
import { rmsNormSubgroupKey } from "../../src/kernels/rms-norm-subgroup.ts";
import { scoreKeyPart } from "../../src/kernels/score-storage.ts";
import {
  statePvParallelKey,
  statePvParallelReadonlyKey,
  stateQkParallelKey,
  stateQkParallelReadonlyKey,
} from "../../src/kernels/state-attention.ts";
import { stateStatsPvKey } from "../../src/kernels/state-attention-stats-pv.ts";
import { STATIC_QUANTIZE_PACKED_KEY } from "../../src/kernels/static-quantize.ts";
import type { SessionDiagnostics, SessionOptions } from "../../src/runtime/session-types.ts";

/** census の行（`lastRunPipelines` の要素）。 */
export type CensusRow = NonNullable<SessionDiagnostics["lastRunPipelines"]>[number];

/** パイプラインキーの判別述語。 */
export type KeyPredicate = (key: string) => boolean;

// ---------------------------------------------------------------------------
// 席の網羅表
// ---------------------------------------------------------------------------

/**
 * 数値を変えない `SessionOptions` の欄（census の対象外）。
 *
 * MUST: 除外リスト方式（許可リストにしない）。`SessionOptions` に文字列キーの欄を 1 つ足すと
 * {@link NumericSeat} に自動で入り、{@link SEAT_REFERENCE} / {@link SEAT_SIGNATURES} の網羅が
 * 型検査で落ちる — 足した人が「数値を変える席か」を必ず判断する。
 * NOTE: unique symbol のテスト専用ノブ（`I8A8_DOT` / `ROW_BLOCK_SPLIT`）は `Extract<…, string>` で
 * 外れる。どちらも参照と整数一致 / Uint32 一致が契約の変種選択で、数値を変えない。
 */
type NonNumericField =
  | "submitPolicy"
  | "sharedWeights"
  | "planBackingBudgetBytes"
  | "linearGemvRowsThreadTarget";

/** 数値を変える `SessionOptions` の席（文字列キーから {@link NonNumericField} を除いた集合）。 */
export type NumericSeat = Exclude<Extract<keyof SessionOptions, string>, NonNumericField>;

type SeatValue<Seat extends NumericSeat> = NonNullable<SessionOptions[Seat]>;

/**
 * 各席の参照値（= runtime の省略値 — src/runtime/session-build.ts の既定代入と同じ値）。
 * 型が各席の union へ縛るので、綴りを誤れば型検査で落ちる。
 */
export const SEAT_REFERENCE = {
  linearCompute: "f32",
  attentionCompute: "f32",
  attentionScoreStorage: "f32",
  stateAttentionReduce: "sequential",
  linearGemvReduce: "sequential",
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  packedStaticQuantize: false,
  rmsNormReduce: "workgroup",
} as const satisfies { readonly [Seat in NumericSeat]-?: SeatValue<Seat> };

type NonReference<Seat extends NumericSeat> = Exclude<
  SeatValue<Seat>,
  (typeof SEAT_REFERENCE)[Seat]
>;

/** 席 × 非参照値 → 判別述語（真偽の席は `"true"` をキーにする）。 */
export type SeatSignatures = {
  readonly [Seat in NumericSeat]: {
    readonly [Value in NonReference<Seat> as `${Value}`]: KeyPredicate;
  };
};

/** キーの族（先頭の `:` より前 — `linear` / `attention_qk` / `linear_gemv_parallel` …）。 */
const familyOf = (key: string): string => {
  const colon = key.indexOf(":");
  return colon < 0 ? key : key.slice(0, colon);
};

/** キーの dtype 欄（`族:世代:dtype:…` の 3 つ目 — i8a8 変種はここが `i8a8`）。 */
const dtypeOf = (key: string): string | undefined => key.split(":")[2];

/**
 * `:` 区切りの断片を含むか（部分文字列ではなく区切りの境界で見る — `:s16` が `:s16x` に
 * 当たらないように）。
 */
const hasFragment = (key: string, fragment: string): boolean => `${key}:`.includes(`${fragment}:`);

/** 変種キーが基準キーの末尾に足した断片（src の key 関数が「基準 + 断片」で組む族に使う）。 */
const suffixOf = (variant: string, base: string): string => {
  assert(variant.startsWith(base), `断片の導出: '${variant}' が '${base}' で始まらない`);
  return variant.slice(base.length);
};

/** `(sliding, gqa)` の 4 通り全てで key 関数を回した集合（states 形の変種ビットは 2 つだけ）。 */
const stateVariants = (
  ...keyOf: readonly ((sliding: boolean, gqa: boolean) => string)[]
): ReadonlySet<string> =>
  new Set(
    keyOf.flatMap((key) =>
      [false, true].flatMap((sliding) => [false, true].map((gqa) => key(sliding, gqa)))
    ),
  );

// 断片・族の導出に使う代表引数（どの引数でも断片は同じ — key 関数が「基準 + 断片」で組むため）。
const GEMV_PARALLEL_SAMPLE = linearGemvParallelKey("i8", undefined, 4);

const LINEAR_FAMILY = familyOf(linearKey("f32", true));
const I8A8_DTYPE = dtypeOf(linearI8a8Key(true, true));
const C16 = gemmComputeKeyPart("f16");
const S16 = scoreKeyPart("f16");
const ATTENTION_FAMILIES: ReadonlySet<string> = new Set(
  [attentionQkKey(true), attentionPvKey(true), ATTENTION_STATS_KEY].map(familyOf),
);
const ATTENTION_I8A8_FAMILIES: ReadonlySet<string> = new Set(
  [attentionQkI8a8Key(true, true), attentionPvI8a8Key(true, true)].map(familyOf),
);
const STATE_PARALLEL_KEYS = stateVariants(
  stateQkParallelKey,
  statePvParallelKey,
  stateQkParallelReadonlyKey,
  statePvParallelReadonlyKey,
);
const STATE_STATS_PV_KEYS = stateVariants(stateStatsPvKey);
const GEMV_PARALLEL_FAMILY = familyOf(GEMV_PARALLEL_SAMPLE);
const GEMV_SUBGROUP = suffixOf(
  linearGemvSubgroupKey("i8", undefined, 4),
  GEMV_PARALLEL_SAMPLE,
);
const GEMV_STATIC_QUANTIZE = suffixOf(
  linearGemvStaticQuantizeKey("i8", undefined, 4),
  GEMV_PARALLEL_SAMPLE,
);
const GEMV_PACKED = suffixOf(linearGemvParallelPackedKey("i8", undefined, 4), GEMV_PARALLEL_SAMPLE);
const RMS_NORM_ADD_FAMILY = familyOf(rmsNormAddKey("norm-residual"));
const RMS_FAMILIES: ReadonlySet<string> = new Set([familyOf(RMS_NORM_KEY), RMS_NORM_ADD_FAMILY]);
const RMS_SUBGROUP = suffixOf(rmsNormSubgroupKey(), RMS_NORM_KEY);

const isGemvParallel = (key: string): boolean => familyOf(key) === GEMV_PARALLEL_FAMILY;

/**
 * 数値を変える席 × 非参照値 → 「その変種が走った」ことを示すキーの述語。
 *
 * MUST: 同じ席の非参照値どうしの述語は互いに素（{@link assertSeatsApplied} が値ごとに数えるため）。
 * 席を跨いだ重なりは設計どおり（SRQ 融合・packed は並列 GEMV 族の上に立つ・subgroup32 の RMS→add
 * 融合キーは `fuseRmsNormAdd` と `rmsNormReduce` の両方の証拠）。
 */
export const SEAT_SIGNATURES: SeatSignatures = {
  linearCompute: {
    a8: (key) => familyOf(key) === LINEAR_FAMILY && dtypeOf(key) === I8A8_DTYPE,
    f16: (key) => familyOf(key) === LINEAR_FAMILY && hasFragment(key, C16),
  },
  attentionCompute: {
    f16: (key) => ATTENTION_FAMILIES.has(familyOf(key)) && hasFragment(key, C16),
    a8: (key) => ATTENTION_I8A8_FAMILIES.has(familyOf(key)) && dtypeOf(key) === I8A8_DTYPE,
  },
  attentionScoreStorage: {
    f16: (key) => ATTENTION_FAMILIES.has(familyOf(key)) && hasFragment(key, S16),
  },
  stateAttentionReduce: {
    parallel: (key) => STATE_PARALLEL_KEYS.has(key),
    "parallel-fused": (key) => STATE_STATS_PV_KEYS.has(key),
  },
  linearGemvReduce: {
    parallel: (key) => isGemvParallel(key) && !hasFragment(key, GEMV_SUBGROUP),
    "parallel-subgroup32": (key) => isGemvParallel(key) && hasFragment(key, GEMV_SUBGROUP),
  },
  fuseRmsNormAdd: {
    true: (key) => familyOf(key) === RMS_NORM_ADD_FAMILY,
  },
  fuseLinearStaticQuantize: {
    true: (key) => isGemvParallel(key) && hasFragment(key, GEMV_STATIC_QUANTIZE),
  },
  packedStaticQuantize: {
    // 生産側（単体 SRQ の packed 変種）と消費側（並列 GEMV の packed 変種）のどちらでも立つ。
    true: (key) =>
      key === STATIC_QUANTIZE_PACKED_KEY || (isGemvParallel(key) && hasFragment(key, GEMV_PACKED)),
  },
  rmsNormReduce: {
    subgroup32: (key) => RMS_FAMILIES.has(familyOf(key)) && hasFragment(key, RMS_SUBGROUP),
  },
};

// ---------------------------------------------------------------------------
// 検査
// ---------------------------------------------------------------------------

/**
 * census の行を取り出す。undefined / 空は fail loudly。
 *
 * MUST: 空を「何も見ない」で通さない — 空の表どうしの一致・0 本の断言は恒真になり、
 * census が 1 つも検査しないまま緑になる。
 */
export const requireCensus = (
  rows: SessionDiagnostics["lastRunPipelines"],
  where: string,
): readonly CensusRow[] => {
  assert(
    rows !== undefined,
    `${where}: lastRunPipelines が埋まっていない（直近 run が無い / 失敗）`,
  );
  assert(rows.length > 0, `${where}: census が空（数え損ねを 0 本の断言で隠す）`);
  return rows;
};

/** 述語（または完全一致のキー）に当たる行の dispatch 本数の合計。 */
export const countDispatches = (
  rows: readonly CensusRow[],
  match: string | KeyPredicate,
): number => {
  const test: KeyPredicate = typeof match === "string" ? (key) => key === match : match;
  return rows.reduce((total, row) => total + (test(row.key) ? row.dispatchCount : 0), 0);
};

/**
 * 複数 run の census を 1 つの表へ畳む（キー別に本数を足す・キーの辞書順）。
 *
 * MUST: 要素に undefined があれば fail loudly（成功した run の直後に census が無いのは
 * 源が壊れた形で、読み飛ばすと「その run の変種」が集計から黙って消える）。
 */
export const mergeCensus = (
  runs: readonly SessionDiagnostics["lastRunPipelines"][],
  where: string,
): readonly CensusRow[] => {
  const counts = new Map<string, number>();
  runs.forEach((rows, index) => {
    assert(rows !== undefined, `${where}: ${index} 本目の run の lastRunPipelines が無い`);
    for (const { key, dispatchCount } of rows) {
      counts.set(key, (counts.get(key) ?? 0) + dispatchCount);
    }
  });
  return [...counts]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([key, dispatchCount]) => ({ key, dispatchCount }));
};

/** census の期待 1 件（「本数ちょうど」か「下限」）。 */
export type CensusExpectation =
  & {
    /** 失敗文言に出す名前。 */
    readonly label: string;
    /** 判別述語（文字列はキーの完全一致）。 */
    readonly match: string | KeyPredicate;
  }
  & ({ readonly count: number } | { readonly atLeast: number });

/** 期待ごとの本数を検査する。rows が undefined / 空なら fail loudly（{@link requireCensus}）。 */
export const assertPipelineCensus = (
  rows: SessionDiagnostics["lastRunPipelines"],
  expected: readonly CensusExpectation[],
  where: string,
): void => {
  const census = requireCensus(rows, where);
  for (const expectation of expected) {
    const actual = countDispatches(census, expectation.match);
    const keys = census.map((row) => `${row.key}×${row.dispatchCount}`).join(" / ");
    if ("count" in expectation) {
      assert(
        actual === expectation.count,
        `${where}: ${expectation.label} が ${actual} 本（期待 ${expectation.count} 本）— ${keys}`,
      );
    } else {
      assert(
        actual >= expectation.atLeast,
        `${where}: ${expectation.label} が ${actual} 本（期待 ${expectation.atLeast} 本以上）— ${keys}`,
      );
    }
  }
};

/** 網羅表の席名（`Object.keys` は string[] を返すので、網羅表の型から取り直す）。 */
const NUMERIC_SEATS = Object.keys(SEAT_REFERENCE) as readonly NumericSeat[];

/**
 * 渡した席が実際に効いたことの両側検査。
 *
 * - 非参照値の席: その値の述語に当たる dispatch が 1 本以上。
 * - 参照値を**明示した**席: その席の非参照値の述語のどれにも当たる dispatch が 0 本。
 *
 * 渡していない（undefined の）席は検査しない — 省略時の実効値は呼び手（quant 席の宣言など）が
 * 決めるので、ここで参照値と決めつけない。
 * NOTE: 「1 本以上」は席が**どこかで**効いた証拠で、完全適格（全対象が変種へ落ちた）の証拠では
 * ない。対象ごとの本数は {@link assertPipelineCensus} で持つ。
 */
export const assertSeatsApplied = (
  rows: SessionDiagnostics["lastRunPipelines"],
  options: SessionOptions,
  where: string,
): void => {
  const census = requireCensus(rows, where);
  for (const seat of NUMERIC_SEATS) {
    const value = options[seat];
    if (value === undefined) continue;
    const variants: Readonly<Record<string, KeyPredicate>> = SEAT_SIGNATURES[seat];
    if (value === SEAT_REFERENCE[seat]) {
      for (const [variant, predicate] of Object.entries(variants)) {
        const hits = census.filter((row) => predicate(row.key)).map((row) => row.key);
        assert(
          hits.length === 0,
          `${where}: ${seat} は参照値 ${JSON.stringify(value)} なのに ${variant} の変種が走った — ${
            hits.join(" / ")
          }`,
        );
      }
      continue;
    }
    const predicate = variants[String(value)];
    assert(
      predicate !== undefined,
      `${where}: ${seat}=${JSON.stringify(value)} の述語が網羅表に無い`,
    );
    assert(
      countDispatches(census, predicate) > 0,
      `${where}: ${seat}=${JSON.stringify(value)} の変種が 1 本も走っていない — ${
        census.map((row) => row.key).join(" / ")
      }`,
    );
  }
};
