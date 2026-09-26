/**
 * flag-bench の走行順・GPU 時間の積算・要約（**純関数だけ** — GPU も I/O も時計も触らない）。
 *
 * MUST: 全モジュール副作用ゼロ（CLAUDE.md）。
 */

import type { GpuTimingStats } from "../../packages/runtime/mod.ts";

/** 1 visit = 1 つの set を fresh な device + pipeline で組んで回す 1 回。 */
export type Visit = {
  /** 1 始まりの通し番号。 */
  readonly visit: number;
  /** 1 始まりの round 番号。 */
  readonly round: number;
  /** `--set` の添字（0 = 基準）。 */
  readonly setIndex: number;
};

/**
 * ABBA の走行順 — 1 round = S0..Sk → Sk..S0。
 *
 * 往路と復路で順を裏返すのは、プロセスの時間とともに単調に進む変化（clock の暖まり・ページ
 * キャッシュの充填）が特定の set にだけ乗らないようにするため。各 set は 1 round に 2 回、
 * 前半と後半に対称な位置で現れるので、線形の漂流は set 間で打ち消し合う。
 */
export const abbaOrder = (setCount: number, rounds: number): Visit[] => {
  if (!Number.isSafeInteger(setCount) || setCount < 1) {
    throw new Error(`abbaOrder: set 数 ${setCount} が正の整数でない`);
  }
  if (!Number.isSafeInteger(rounds) || rounds < 1) {
    throw new Error(`abbaOrder: rounds ${rounds} が正の整数でない`);
  }
  const forward = Array.from({ length: setCount }, (_, index) => index);
  const visits: Visit[] = [];
  for (let round = 1; round <= rounds; round++) {
    for (const setIndex of [...forward, ...forward.toReversed()]) {
      visits.push({ visit: visits.length + 1, round, setIndex });
    }
  }
  return visits;
};

/** 非投機の生成が出す run の種別（`Gemma4RunPhase.kind` のうちこの台本が受けるもの）。 */
export type RunKind = "prefill" | "decode";

/** 1 走行（1 generate）ぶんの GPU 時間の器（種別ごと）。 */
export type KindTally = {
  runs: number;
  totalNs: number;
  dispatchCount: number;
  clampedNegativeSamples: number;
};
export type RunGpuTally = { readonly [K in RunKind]: KindTally } & {
  /**
   * decode run 1 本の**計画上の**キー別 dispatch 本数（`lastRunPipelines`）。1 生成の decode run は
   * 同じ計画を回すので、最初の decode run で確定し、以降の run は一致を検査するだけ。
   */
  decodePipelines?: PipelineCensus;
};

/**
 * パイプラインキー → 1 run の dispatch 本数（キーの辞書順）。どのカーネル変種が走ったかの記録で、
 * 並列フラグが適格表に無い形で no-op になった set を「速度中立」と読み違えないための証跡。
 */
export type PipelineCensus = Readonly<Record<string, number>>;

/** `SessionDiagnostics.lastRunPipelines` の行 → {@link PipelineCensus}。 */
export const pipelineCensus = (
  rows: readonly { readonly key: string; readonly dispatchCount: number }[],
): PipelineCensus => {
  const census: Record<string, number> = {};
  for (const row of rows.toSorted((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0))) {
    if (Object.hasOwn(census, row.key)) {
      throw new Error(`pipelineCensus: キー ${row.key} が 2 行ある`);
    }
    census[row.key] = row.dispatchCount;
  }
  return census;
};

/** キーと本数の両方が同じか。 */
export const sameCensus = (left: PipelineCensus, right: PipelineCensus): boolean => {
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length &&
    keys.every((key) => Object.hasOwn(right, key) && right[key] === left[key]);
};

/** キーの集合だけが同じか（本数は見ない）。 */
export const sameKeySet = (left: PipelineCensus, right: PipelineCensus): boolean => {
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length &&
    keys.every((key) => Object.hasOwn(right, key));
};

/**
 * run 1 本の `lastRunPipelines` を器へ記録する（decode run だけ — prefill は計画が prompt 長で
 * 変わり、フラグの効き目の比較には decode の 1 step が要る）。
 *
 * MUST: census が無い run・1 生成の中で decode の census が変わる run は落とす（欠けた / 混ざった
 * 記録で「キーが変わらない」と読ませない）。
 */
export const addRunPipelines = (
  tally: RunGpuTally,
  kind: RunKind,
  rows: readonly { readonly key: string; readonly dispatchCount: number }[] | undefined,
): void => {
  if (kind !== "decode") return;
  if (rows === undefined || rows.length === 0) {
    throw new Error("addRunPipelines: decode run の lastRunPipelines が空");
  }
  const census = pipelineCensus(rows);
  if (tally.decodePipelines === undefined) {
    tally.decodePipelines = census;
    return;
  }
  if (!sameCensus(tally.decodePipelines, census)) {
    throw new Error("addRunPipelines: 1 生成の中で decode run の census が変わった");
  }
};

export const emptyRunGpuTally = (): RunGpuTally => {
  const one = (): KindTally => ({
    runs: 0,
    totalNs: 0,
    dispatchCount: 0,
    clampedNegativeSamples: 0,
  });
  return { prefill: one(), decode: one() };
};

/** run 1 本の `lastRunTiming` を、その run の種別の器へ積む。 */
export const addRunTiming = (
  tally: RunGpuTally,
  kind: RunKind,
  stats: Pick<GpuTimingStats, "totalNs" | "dispatchCount" | "clampedNegativeSamples">,
): void => {
  const bucket = tally[kind];
  bucket.runs += 1;
  bucket.totalNs += stats.totalNs;
  bucket.dispatchCount += stats.dispatchCount;
  bucket.clampedNegativeSamples += stats.clampedNegativeSamples;
};

/** 1 走行の 1 種別ぶんの GPU 時間（分母はその走行のその種別の run 本数）。 */
export type KindRecord = {
  readonly runs: number;
  readonly msPerRun: number;
  readonly dispatchesPerRun: number;
  readonly clampedNegativeSamples: number;
};
export type RunGpuRecord = { readonly [K in RunKind]?: KindRecord };

/**
 * 器 → 行の `gpu`。run が 0 本の種別は**欄ごと無い**（0 ms と書くと「測ったら 0 だった」と
 * 読めてしまう — `tools/mtp-bench` の `timing.ts` と同じ原則）。
 */
export const gpuRecord = (tally: RunGpuTally): RunGpuRecord => {
  const record: { [K in RunKind]?: KindRecord } = {};
  for (const kind of ["prefill", "decode"] as const) {
    const bucket = tally[kind];
    if (bucket.runs === 0) continue;
    record[kind] = {
      runs: bucket.runs,
      msPerRun: bucket.totalNs / 1e6 / bucket.runs,
      dispatchesPerRun: bucket.dispatchCount / bucket.runs,
      clampedNegativeSamples: bucket.clampedNegativeSamples,
    };
  }
  return record;
};

/** JSONL の 1 走行 1 行。 */
export type RunRow = {
  readonly type: "run";
  readonly set: string;
  readonly visit: number;
  readonly round: number;
  /** `cases.json` の case 名。 */
  readonly prompt: string;
  /** 0 = 暖機・1 以上 = 計測の何本目か（visit × prompt の中で数える）。 */
  readonly rep: number;
  readonly warmup: boolean;
  /** この行を採った device で GPU 時間計測が有効だったか（ON と OFF を集計で混ぜないための印）。 */
  readonly gpuTiming: boolean;
  readonly promptTokens: number;
  /** 配送された token 数（停止 token を含む — `token` イベントの数）。 */
  readonly delivered: number;
  readonly stopReason: string;
  /** `generate` 呼び出し → 先頭 token の到着。 */
  readonly ttftMs: number;
  /** (完了 − 先頭 token) / (配送 − 1)。 */
  readonly decodeMsPerToken: number;
  /** token id 列（JSON の `[1,2,…]` 綴り・UTF-8）の SHA-256。 */
  readonly tokensSha256: string;
  /** `gpuTiming` のときだけ。 */
  readonly gpu?: RunGpuRecord;
  /** `gpuTiming` のときだけ: decode run 1 本のキー別 dispatch 本数（{@link addRunPipelines}）。 */
  readonly decodePipelines?: PipelineCensus;
};

/** 基準比（`(値 / 基準 − 1) × 100` — 正は遅い / 多い）。 */
export type DeltaPercent = {
  readonly decodeMsPerToken: number;
  readonly ttftMs: number;
  readonly gpuDecodeMsPerStep?: number;
  readonly gpuDispatchesPerStep?: number;
};

/** 1 set の要約（暖機を除く計測走行の中央値 — 平均は 1 本の跳ねを比へ持ち込む）。 */
export type SetSummary = {
  readonly label: string;
  readonly measuredRuns: number;
  readonly decodeMsPerToken: number;
  readonly ttftMs: number;
  /** `gpuTiming` のときだけ: 計測走行ごとの decode ms/run の中央値。 */
  readonly gpuDecodeMsPerStep?: number;
  readonly gpuDispatchesPerStep?: number;
  readonly gpuPrefillMsPerRun?: number;
  readonly deltaPercent: DeltaPercent;
  /** prompt ごとに、この set の全走行（暖機を含む・全 visit）の token 列が 1 種類だったか。 */
  readonly tokensIdenticalAcrossVisits: boolean;
  /** prompt ごとに、この set と基準の全走行の token 列が 1 種類だったか（フラグが出力を変えたか）。 */
  readonly tokensMatchReference: boolean;
  /** `gpuTiming` のときだけ: decode run 1 本のキー別 dispatch 本数（この set の全走行で同一）。 */
  readonly appliedKeys?: PipelineCensus;
  /**
   * `gpuTiming` のときだけ・基準以外の set だけ: decode のキー集合が基準と同じか。true は
   * 「カーネルの選択が 1 つも変わっていない」— カーネルを選び替えるはずのフラグなら no-op
   * （適格表に無い形での縮退など）で、GPU 時間の差は計測の揺れでしかない。
   */
  readonly noKeyChange?: boolean;
};

export const median = (values: readonly number[]): number => {
  if (values.length === 0) throw new Error("median: 空の列");
  const sorted = values.toSorted((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

const deltaPercent = (value: number, reference: number): number => (value / reference - 1) * 100;

/** prompt → その行群が出した token 列の sha の集合。 */
const shasByPrompt = (rows: readonly RunRow[]): Map<string, Set<string>> => {
  const map = new Map<string, Set<string>>();
  for (const row of rows) {
    const shas = map.get(row.prompt) ?? new Set<string>();
    shas.add(row.tokensSha256);
    map.set(row.prompt, shas);
  }
  return map;
};

type Measured = {
  readonly label: string;
  readonly rows: readonly RunRow[];
  readonly measured: readonly RunRow[];
  readonly decodeMsPerToken: number;
  readonly ttftMs: number;
  readonly gpuDecodeMsPerStep?: number;
  readonly gpuDispatchesPerStep?: number;
  readonly gpuPrefillMsPerRun?: number;
  readonly appliedKeys?: PipelineCensus;
};

/**
 * set の全走行（暖機を含む・全 visit）の decode census を 1 つに畳む。フラグは Session 構築時に
 * 固定される静的ノブなので、同じ set の走行は同じ計画を回す — 違えば記録が壊れている。
 *
 * census は全走行に在るか、どれにも無いか（census を記録する前の行）のどちらか。一部だけ在る
 * のは記録の破れなので落とす。どれにも無ければ `undefined`（`appliedKeys` を出さない）。
 */
const setCensus = (label: string, rows: readonly RunRow[]): PipelineCensus | undefined => {
  const recorded = rows.flatMap((row) =>
    row.decodePipelines === undefined ? [] : [row.decodePipelines]
  );
  if (recorded.length === 0) return undefined;
  const missing = rows.find((row) => row.decodePipelines === undefined);
  if (missing !== undefined) {
    throw new Error(
      `summarizeSets: set ${label} の ${missing.prompt} visit ${missing.visit} rep ${missing.rep} に decodePipelines が無い`,
    );
  }
  const [first] = recorded;
  if (!recorded.every((census) => sameCensus(first, census))) {
    throw new Error(`summarizeSets: set ${label} の decode census が走行ごとに違う`);
  }
  return first;
};

const gpuField = (
  label: string,
  measured: readonly RunRow[],
  kind: RunKind,
  pick: (record: KindRecord) => number,
): number => {
  const values = measured.map((row) => {
    const record = row.gpu?.[kind];
    if (record === undefined) {
      // 計測を有効にした走行で内訳が無い = 観測席が外れている（黙って欠けた中央値を出さない）。
      throw new Error(
        `summarizeSets: set ${label} の ${row.prompt} visit ${row.visit} rep ${row.rep} に gpu.${kind} が無い`,
      );
    }
    return pick(record);
  });
  return median(values);
};

const measure = (label: string, rows: readonly RunRow[], gpuTiming: boolean): Measured => {
  const measured = rows.filter((row) => !row.warmup);
  if (measured.length === 0) throw new Error(`summarizeSets: set ${label} に計測走行が無い`);
  return {
    label,
    rows,
    measured,
    decodeMsPerToken: median(measured.map((row) => row.decodeMsPerToken)),
    ttftMs: median(measured.map((row) => row.ttftMs)),
    ...(gpuTiming
      ? {
        gpuDecodeMsPerStep: gpuField(label, measured, "decode", (r) => r.msPerRun),
        gpuDispatchesPerStep: gpuField(label, measured, "decode", (r) => r.dispatchesPerRun),
        gpuPrefillMsPerRun: gpuField(label, measured, "prefill", (r) => r.msPerRun),
        appliedKeys: setCensus(label, rows),
      }
      : {}),
  };
};

/**
 * 行 → set ごとの要約（`labels[0]` が基準）。
 *
 * MUST: 計測 ON と OFF の行を 1 つの要約に混ぜない — ON の device は 1 dispatch = 1 pass に
 * 開くので壁時計が伸び、混ぜた中央値はどちらの条件の値でもなくなる。混ざっていたら落とす。
 */
export const summarizeSets = (
  rows: readonly RunRow[],
  labels: readonly string[],
): SetSummary[] => {
  if (labels.length === 0) throw new Error("summarizeSets: set が 1 つも無い");
  const timingFlags = new Set(rows.map((row) => row.gpuTiming));
  if (timingFlags.size > 1) {
    throw new Error(
      "summarizeSets: gpuTiming の ON と OFF の行が混ざっている（別々に集計すること）",
    );
  }
  const gpuTiming = timingFlags.has(true);
  for (const row of rows) {
    if (!labels.includes(row.set)) throw new Error(`summarizeSets: 未知の set ${row.set} の行`);
  }
  const all = labels.map((label) =>
    measure(label, rows.filter((row) => row.set === label), gpuTiming)
  );
  const reference = all[0];
  // census も set を跨いで全部在るか全部無いか（1 ファイル = 1 回の起動なので混ざらない）。
  if (new Set(all.map((one) => one.appliedKeys === undefined)).size > 1) {
    throw new Error("summarizeSets: decodePipelines を持つ set と持たない set が混ざっている");
  }
  const referenceShas = shasByPrompt(reference.rows);
  return all.map((one) => {
    const ownShas = shasByPrompt(one.rows);
    const identical = [...ownShas.values()].every((shas) => shas.size === 1);
    const prompts = new Set([...ownShas.keys(), ...referenceShas.keys()]);
    const matches = [...prompts].every((prompt) =>
      new Set([...(ownShas.get(prompt) ?? []), ...(referenceShas.get(prompt) ?? [])]).size === 1
    );
    const gpuDelta = one.gpuDecodeMsPerStep !== undefined &&
        reference.gpuDecodeMsPerStep !== undefined &&
        one.gpuDispatchesPerStep !== undefined &&
        reference.gpuDispatchesPerStep !== undefined
      ? {
        gpuDecodeMsPerStep: deltaPercent(one.gpuDecodeMsPerStep, reference.gpuDecodeMsPerStep),
        gpuDispatchesPerStep: deltaPercent(
          one.gpuDispatchesPerStep,
          reference.gpuDispatchesPerStep,
        ),
      }
      : {};
    return {
      label: one.label,
      measuredRuns: one.measured.length,
      decodeMsPerToken: one.decodeMsPerToken,
      ttftMs: one.ttftMs,
      ...(one.gpuDecodeMsPerStep === undefined
        ? {}
        : { gpuDecodeMsPerStep: one.gpuDecodeMsPerStep }),
      ...(one.gpuDispatchesPerStep === undefined
        ? {}
        : { gpuDispatchesPerStep: one.gpuDispatchesPerStep }),
      ...(one.gpuPrefillMsPerRun === undefined
        ? {}
        : { gpuPrefillMsPerRun: one.gpuPrefillMsPerRun }),
      deltaPercent: {
        decodeMsPerToken: deltaPercent(one.decodeMsPerToken, reference.decodeMsPerToken),
        ttftMs: deltaPercent(one.ttftMs, reference.ttftMs),
        ...gpuDelta,
      },
      tokensIdenticalAcrossVisits: identical,
      tokensMatchReference: matches,
      ...(one.appliedKeys === undefined ? {} : { appliedKeys: one.appliedKeys }),
      // 基準自身には立てない（自明に true で、警告の対象にならない）。
      ...(one === reference || one.appliedKeys === undefined || reference.appliedKeys === undefined
        ? {}
        : { noKeyChange: sameKeySet(one.appliedKeys, reference.appliedKeys) }),
    };
  });
};
