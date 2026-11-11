/**
 * **The fusion candidate search** (no GPU): the consecutive windows of plain nodes the fusion pass
 * left unfused in a plan, cut down to the maximal ones, grouped by form and ranked by the bytes of
 * the internal values a bit-identical fold would remove (the dispatch count is the second column).
 *
 * It exists to choose the next fusion rule to write; it accepts nothing. A row is a candidate, not
 * an acceptance: it passed only the eligibility conditions every rule shares
 * (`enumerateUnfusedWindows` in packages/runtime/src/runtime/fusion.ts), not any rule's spelling,
 * shape conditions or rounding points (ADR 0040 decision 2).
 *
 * MUST: the eligibility predicates stay in the fusion-pass layer. This module consumes the windows
 * the enumerator returns (their chains included) and never decides on its own which nodes may fold;
 * a copy here would keep calling a window foldable by an old criterion after the accepted set
 * changed.
 *
 * What a row measures, and why only these two:
 *
 * - **Bytes** (the ranking key): each internal value of the chain is written once and read once
 *   (an internal value has exactly one consumer), so a fold removes 2 × its elements × 4 B (chains
 *   are f32). A value written by a node that aliases its input (a reshape — zero dispatches,
 *   ADR 0011) is a view of its producer's buffer and counts 0; the producer's value carries the
 *   write and the read. Deterministic, GPU-free and reproducible per shape binding.
 * - **Dispatches** (the second column): the chain's dispatches minus the one of the fused kernel.
 *   Without a GPU this is an approximation: an aliasing node is 0, a `cat` is one per input, every
 *   other node is 1. It is the only clue for decode (M = 1), where the bytes say nothing.
 * - No GPU time: the per-kernel time of the timing mode (one pass per dispatch) also counts the
 *   per-pass fixed cost a run outside the timing mode never pays, so it overstates what removing a
 *   small dispatch saves (perf-ledger K-105, research 2026-11-10-gemma4-26b-fast §8.1: 60
 *   dispatches fewer moved the untimed forward by 0 to 0.1 ms on Node + Dawn, against 0.36 ms in
 *   the per-kernel times). Ranking by it would put folds that do not help at the top.
 *
 * A window is **maximal** when no other window contains both its node range and its chain; one
 * chain is then one row instance instead of all its n-gram prefixes. A window as wide as the
 * window limit may have been cut by the limit, so it carries the "hit the limit" mark.
 *
 * **The stable search** (`stableFusionCandidates`) takes the limit out of the reader's hands. A
 * fixed limit cuts a long chain into overlapping windows: the same chain becomes several rows
 * shifted by one node and its bytes are counted more than once (at 12, the MOSS-Image DiT had 16 of
 * 33 rows at the limit, a matte model 79 of 97). The mark alone is not enough either: a window that
 * ends one node short of the limit can still have its chain cut. So the search starts at the
 * default limit and doubles it until the ranked rows are the same as at the limit before
 * (everything compared but the mark, which follows the limit and not the chain), up to a ceiling.
 * A search that reaches the ceiling with the rows still changing says so (`stable: false`); its
 * rows keep their marks.
 *
 * The ceiling, 384, is the smallest doubling of the default limit at which every local mirror
 * stabilises (2026-11-11, Session defaults, the report's bindings): of 47 graphs, 17 stop at 24,
 * 20 at 48, 6 at 96, 2 at 192 (the MOSS-Image DiT pair, widest window 69 nodes) and 2 at 384 (the
 * two matte models: 32 rows at 96, 31 at 192 and 384, widest window 128 nodes). At 192 those two
 * would not stabilise. The cost grows about fourfold per doubling (a matte model takes 3.8 s at
 * 192 and 13 s at 384); the report over every mirror runs in 47 s at this ceiling, and its
 * searches would take about 21 s at 192.
 *
 * NOTE: "the same rows at two limits in a row" is a stopping rule, not a proof. A window whose first
 * node's value is read hundreds of nodes later appears only once the limit spans the whole
 * distance: the sbv2 voice graph gives the same rows at 48, 96 and 192 (the search stops at 96),
 * yet at 384 and 768 it has windows of 381 to 394 nodes (a `slice` read by a node at the end of a
 * flow layer, some 18 to 23 chain ops among hundreds of in-window passthroughs).
 */

import {
  enumerateUnfusedWindows,
  type ExecStep,
  type FusionContext,
  nodeStepSpan,
  type UnfusedWindow,
} from "../../../runtime/src/runtime/fusion.ts";
import { numel } from "../../../runtime/src/ops.ts";
import type { NodePlan } from "../../../runtime/src/runtime/plan.ts";

/** The default window limit (the old record recommended 10–12; its 9 cut long chains). */
export const DEFAULT_MAX_WINDOW = 12;

/** Bytes per element of a chain value (the enumerator admits f32 chains only). */
const F32_BYTES = 4;

/** One maximal window with its measures. */
export type CandidateWindow = {
  readonly window: UnfusedWindow;
  /** The aggregation key: the window's ops in order, in-window passthroughs in brackets. */
  readonly form: string;
  /** Bytes of internal values a fold removes (writes + reads). */
  readonly bytes: number;
  /** Dispatches a fold removes (the approximation of the module doc). */
  readonly dispatches: number;
  /** The window is as wide as the window limit, so the limit may have cut it. */
  readonly hitLimit: boolean;
};

/** The windows of one form, aggregated. */
export type CandidateRow = {
  readonly form: string;
  /** The chain's ops (passthroughs excluded). */
  readonly ops: readonly string[];
  readonly count: number;
  /** Total over the instances. */
  readonly bytes: number;
  /** Total over the instances. */
  readonly dispatches: number;
  /** Node indices of the instances, ascending (the first is the tie-break of the ranking). */
  readonly nodeIndices: readonly number[];
  /** The output name of the first instance's chain (the clue for finding it in the IR). */
  readonly outputName: string;
  /** Some instance hit the window limit: re-run with a larger limit. */
  readonly hitLimit: boolean;
};

/** Whether `outer` holds every node of `inner`. */
const holdsAll = (outer: readonly NodePlan[], inner: readonly NodePlan[]): boolean => {
  const nodes = new Set(outer);
  return inner.every((node) => nodes.has(node));
};

/**
 * The maximal windows of `windows`, in input order: a window is dropped when another window's node
 * range and chain both contain its own.
 */
export const maximalWindows = (windows: readonly UnfusedWindow[]): readonly UnfusedWindow[] => {
  const byStart = new Map<number, UnfusedWindow[]>();
  let widest = 0;
  for (const window of windows) {
    const found = byStart.get(window.nodeIndex);
    if (found === undefined) byStart.set(window.nodeIndex, [window]);
    else found.push(window);
    widest = Math.max(widest, window.windowSize);
  }
  const containedElsewhere = (inner: UnfusedWindow): boolean => {
    const end = inner.nodeIndex + inner.windowSize;
    for (let start = inner.nodeIndex - widest + 1; start <= inner.nodeIndex; start += 1) {
      for (const outer of byStart.get(start) ?? []) {
        if (outer === inner || start + outer.windowSize < end) continue;
        if (holdsAll(outer.chain, inner.chain)) return true;
      }
    }
    return false;
  };
  return windows.filter((window) => !containedElsewhere(window));
};

/**
 * The plain nodes of a step list by node index — the index `enumerateUnfusedWindows` gives a
 * window: a plain node step is one position, any other step advances by the nodes it stands for.
 */
const plainNodesByIndex = (steps: readonly ExecStep[]): ReadonlyMap<number, NodePlan> => {
  const nodes = new Map<number, NodePlan>();
  let index = 0;
  for (const step of steps) {
    if (step.kind === "fused") {
      index += step.nodeCount;
      continue;
    }
    const span = nodeStepSpan(step);
    if (span === 1) nodes.set(index, step.plan);
    index += span;
  }
  return nodes;
};

/**
 * The window's nodes in order. MUST: fail loudly when the nodes at the window's positions are not a
 * window around its chain (first node = the chain's first, every chain node inside) — the index rule
 * here and the enumerator's would have drifted apart, and every form would be read off the wrong
 * nodes.
 */
const windowNodes = (
  window: UnfusedWindow,
  nodes: ReadonlyMap<number, NodePlan>,
): readonly NodePlan[] => {
  const span = Array.from({ length: window.windowSize }, (_, offset) =>
    nodes.get(window.nodeIndex + offset),
  );
  const present = span.filter((node): node is NodePlan => node !== undefined);
  if (
    present.length !== span.length ||
    present[0] !== window.chain[0] ||
    !holdsAll(present, window.chain)
  ) {
    throw new Error(
      `fusion candidates: the nodes at ${window.nodeIndex}+${window.windowSize} are not the window` +
        ` of the chain '${window.ops.join(",")}' (the node index no longer matches the enumerator's)`,
    );
  }
  return present;
};

/** The aggregation key: the window's ops in order, the in-window passthroughs in brackets. */
const formOf = (window: UnfusedWindow, nodes: readonly NodePlan[]): string => {
  const chain = new Set(window.chain);
  return nodes.map((node) => (chain.has(node) ? node.node.op : `[${node.node.op}]`)).join(",");
};

/** The nodes of a step list that alias their input (zero dispatches — the plan's own decision). */
export const aliasedNodes = (steps: readonly ExecStep[]): ReadonlySet<NodePlan> =>
  new Set(steps.flatMap((step) => (step.kind === "node" && step.aliasesInput ? [step.plan] : [])));

/** Bytes of the internal values a fold of `chain` removes (module doc). */
export const removedBytes = (chain: readonly NodePlan[], aliased: ReadonlySet<NodePlan>): number =>
  chain
    .slice(0, -1)
    .filter((node) => !aliased.has(node))
    .reduce(
      (total, node) =>
        total + node.outputs.reduce((sum, out) => sum + 2 * numel(out.shape) * F32_BYTES, 0),
      0,
    );

/** Dispatches a fold of `chain` removes (the approximation of the module doc). */
export const removedDispatches = (
  chain: readonly NodePlan[],
  aliased: ReadonlySet<NodePlan>,
): number => {
  const before = chain.reduce((total, node) => {
    if (aliased.has(node)) return total;
    return total + (node.node.op === "cat" ? node.node.ins.length : 1);
  }, 0);
  return Math.max(0, before - 1);
};

/** The maximal windows of a plan with their measures, in node order. */
export const candidateWindows = (
  steps: readonly ExecStep[],
  context: FusionContext,
  maxWindow: number = DEFAULT_MAX_WINDOW,
): readonly CandidateWindow[] => {
  const aliased = aliasedNodes(steps);
  const nodes = plainNodesByIndex(steps);
  return maximalWindows(enumerateUnfusedWindows(steps, context, maxWindow)).map((window) => ({
    window,
    form: formOf(window, windowNodes(window, nodes)),
    bytes: removedBytes(window.chain, aliased),
    dispatches: removedDispatches(window.chain, aliased),
    hitLimit: window.windowSize === maxWindow,
  }));
};

/**
 * Windows grouped by form and ranked: bytes descending, then dispatches descending, then the first
 * instance's node index ascending (so two runs give the same order).
 */
export const candidateRows = (windows: readonly CandidateWindow[]): readonly CandidateRow[] => {
  const groups = new Map<string, CandidateWindow[]>();
  for (const window of windows) {
    const found = groups.get(window.form);
    if (found === undefined) groups.set(window.form, [window]);
    else found.push(window);
  }
  const rows = [...groups].map(([form, members]): CandidateRow => {
    const ordered = [...members].sort((a, b) => a.window.nodeIndex - b.window.nodeIndex);
    return {
      form,
      ops: ordered[0].window.ops,
      count: ordered.length,
      bytes: ordered.reduce((total, member) => total + member.bytes, 0),
      dispatches: ordered.reduce((total, member) => total + member.dispatches, 0),
      nodeIndices: ordered.map((member) => member.window.nodeIndex),
      outputName: ordered[0].window.outputName,
      hitLimit: ordered.some((member) => member.hitLimit),
    };
  });
  return rows.sort(
    (a, b) =>
      b.bytes - a.bytes || b.dispatches - a.dispatches || a.nodeIndices[0] - b.nodeIndices[0],
  );
};

/** The ranked candidate rows of a plan (the whole search in one call). */
export const fusionCandidates = (
  steps: readonly ExecStep[],
  context: FusionContext,
  maxWindow: number = DEFAULT_MAX_WINDOW,
): readonly CandidateRow[] => candidateRows(candidateWindows(steps, context, maxWindow));

/** The ceiling of the stable search's window limit (the module doc gives the measurement). */
export const MAX_WINDOW_CEILING = 384;

/** The result of {@link stableFusionCandidates}. */
export type StableCandidates = {
  /** The ranked rows at {@link StableCandidates.maxWindow}. */
  readonly rows: readonly CandidateRow[];
  /** The window limit the search stopped at. */
  readonly maxWindow: number;
  /**
   * The rows at `maxWindow` are the same as at the limit before it (limit marks aside). False: the
   * search reached the ceiling with the rows still changing, so a row marked limit may still be cut.
   */
  readonly stable: boolean;
};

/** A row without its limit mark (the mark follows the limit, not the chain). */
const rowWithoutMark = ({ hitLimit: _mark, ...row }: CandidateRow): string => JSON.stringify(row);

/** Whether two ranked row lists are the same, limit marks aside. */
const sameRowsBesidesMark = (a: readonly CandidateRow[], b: readonly CandidateRow[]): boolean =>
  a.length === b.length && a.every((row, rank) => rowWithoutMark(row) === rowWithoutMark(b[rank]));

/**
 * **The stable search**: the ranked rows of a plan, the window limit widened until the table stops
 * changing — from {@link DEFAULT_MAX_WINDOW}, doubled, until the rows are the same as at the limit
 * before (limit marks aside), up to {@link MAX_WINDOW_CEILING}. Reaching the ceiling with the rows
 * still changing is not an error: the result says `stable: false`, and the rows keep their marks.
 *
 * `ceiling` is there for the unit tests (a chain past the real ceiling costs seconds to search);
 * every real search takes the default.
 */
export const stableFusionCandidates = (
  steps: readonly ExecStep[],
  context: FusionContext,
  ceiling: number = MAX_WINDOW_CEILING,
): StableCandidates => {
  if (!Number.isSafeInteger(ceiling) || ceiling < DEFAULT_MAX_WINDOW) {
    throw new Error(
      `fusion candidates: the ceiling ${ceiling} must be an integer of at least ${DEFAULT_MAX_WINDOW}`,
    );
  }
  let maxWindow = DEFAULT_MAX_WINDOW;
  let rows = fusionCandidates(steps, context, maxWindow);
  while (maxWindow < ceiling) {
    const wider = Math.min(maxWindow * 2, ceiling);
    const widerRows = fusionCandidates(steps, context, wider);
    const stable = sameRowsBesidesMark(rows, widerRows);
    maxWindow = wider;
    rows = widerRows;
    if (stable) return { rows, maxWindow, stable };
  }
  return { rows, maxWindow, stable: false };
};

/**
 * How many times the op sequence `ops` occurs in the chains of `rows`: the non-overlapping
 * occurrences in each row's chain ops (passthroughs excluded), times the row's count. A check names
 * a known miss by its op sequence; one maximal window may hold several sites (two RoPE chains
 * feeding one attention).
 */
export const opSequenceCount = (rows: readonly CandidateRow[], ops: readonly string[]): number =>
  rows.reduce((total, row) => {
    let found = 0;
    for (let at = 0; at + ops.length <= row.ops.length;) {
      if (ops.every((op, offset) => row.ops[at + offset] === op)) {
        found += 1;
        at += ops.length;
      } else {
        at += 1;
      }
    }
    return total + found * row.count;
  }, 0);

/** What a table was searched under (printed at its head). */
export type CandidateScenario = {
  /** The graph (distribution, model, component, storage). */
  readonly graph: string;
  /** The shape binding (decode M = 1, prefill length, the DiT's S, …). */
  readonly shapes: Readonly<Record<string, readonly number[]>>;
  /** The Session options planned under (`{}` = the Session defaults). */
  readonly options: Readonly<Record<string, unknown>>;
  /** The window limit the stable search stopped at. */
  readonly maxWindow: number;
  /** Whether the stable search stabilised there ({@link StableCandidates.stable}). */
  readonly stable: boolean;
};

const MIB = 1024 * 1024;

/**
 * The statements every table carries (what the ranking promises, what it does not, and how a fold
 * is accepted instead).
 */
const TABLE_STATEMENTS = [
  "- **Promises**: folding a row's window bit-identically removes these bytes of internal values" +
    " (writes + reads) and about these dispatches, under this shape binding.",
  "- **Does not promise**: a shorter wall time. Decode is dominated by the launch and fence floor," +
    " and on Dawn removing small dispatches may not move the wall time at all; the cost of a small" +
    " dispatch on Chrome / D3D12 has not been measured.",
  "- **Acceptance**: not by this table. Each fold that gets built is measured on each backend by a" +
    " wall-clock A/B with the timing mode off (alternating rounds, per-position paired differences," +
    " the drift between rounds stated next to them) and decided by the existing kill criteria.",
  "- A row is a candidate that passed only the eligibility conditions every rule shares, not a" +
    " rule's spelling or rounding points. Dispatches are approximate (an aliasing node 0, `cat` one" +
    " per input, every other node 1). Forms list the window's ops; `[op]` is an in-window passthrough.",
  `- The window limit widens by itself: from ${DEFAULT_MAX_WINDOW}, doubled until the rows are the` +
    ` same as at the limit before (limit marks aside), up to ${MAX_WINDOW_CEILING}. A row marked` +
    " **limit** has a window as wide as the limit the search stopped at, so the limit may have cut" +
    " it; a search that stabilised leaves no such row.",
];

/** The window limit line of a table's head, saying in words whether the search stabilised. */
const windowLimitLine = ({ maxWindow, stable }: CandidateScenario): string =>
  stable
    ? `- Window limit: ${maxWindow} (the search stabilised: the rows are the same as at the limit before)`
    : `- Window limit: ${maxWindow} — **the search did not stabilise**: the rows still changed up to` +
      ` the ceiling, so a row marked limit may still be cut`;

/** The Markdown table of `rows` (deterministic: the same rows give the same bytes). */
export const candidatesMarkdown = (
  rows: readonly CandidateRow[],
  scenario: CandidateScenario,
): string => {
  const shapes = Object.entries(scenario.shapes)
    .map(([name, shape]) => `\`${name}\` [${shape.join(", ")}]`)
    .join(", ");
  const lines = [
    `# Fusion candidates — ${scenario.graph}`,
    "",
    `- Shape binding: ${shapes}`,
    `- Session options: \`${JSON.stringify(scenario.options)}\``,
    windowLimitLine(scenario),
    "",
    ...TABLE_STATEMENTS,
    "",
    "| Rank | Form | Count | Bytes removed | MiB | Dispatches removed | First node | Output | Limit |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ...rows.map(
      (row, rank) =>
        `| ${rank + 1} | \`${row.form}\` | ${row.count} | ${row.bytes} | ${(row.bytes / MIB).toFixed(2)}` +
        ` | ${row.dispatches} | ${row.nodeIndices[0]} | \`${row.outputName}\` | ${row.hitLimit ? "limit" : ""} |`,
    ),
    "",
  ];
  return lines.join("\n");
};

/** One JSON object per row, in rank order (deterministic). */
export const candidatesJsonl = (
  rows: readonly CandidateRow[],
  scenario: CandidateScenario,
): string =>
  rows
    .map((row, rank) =>
      JSON.stringify({
        graph: scenario.graph,
        rank: rank + 1,
        form: row.form,
        ops: row.ops,
        count: row.count,
        bytes: row.bytes,
        dispatches: row.dispatches,
        nodeIndices: row.nodeIndices,
        outputName: row.outputName,
        hitLimit: row.hitLimit,
      }),
    )
    .map((line) => `${line}\n`)
    .join("");
