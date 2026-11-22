/**
 * The host's garbage collections during an EmbeddingGemma preset A/B: an observer of the
 * Web-standard `PerformanceObserver` entry type `gc` (Node reports it; browsers do not), and the
 * attribution of the GCs to one run's span on the same timeline (`performance.now`, which the
 * entries' `startTime` share). A slow visit that held a major GC is the host stopping, not the
 * preset. The observer reads the global only when it is made, so importing this module does
 * nothing; its host is injected, so `vp test` drives it with a fake.
 */

import { nextTask } from "../next-task.ts";

import type { RunSpan } from "./arm.ts";

/**
 * The kind of a major GC in a Node `gc` entry's `detail.kind` (`NODE_PERFORMANCE_GC_MAJOR`: V8's
 * mark-sweep-compact). The other kinds are 1 (scavenge), 2 (minor mark-sweep), 8 (incremental
 * marking) and 16 (weak callbacks); Node 24 numbers them so.
 */
export const MAJOR_GC_KIND = 4;

/** One GC the host reported: when it began, how long it held the thread, and its kind. */
export type GcEntry = {
  readonly startTime: number;
  readonly duration: number;
  readonly kind: number;
};

/**
 * What an observer saw: every GC since it was made, or that the host cannot report GCs (so a
 * reader never takes an empty list for "no GC happened").
 */
export type GcReading =
  | { readonly observable: false }
  | { readonly observable: true; readonly entries: readonly GcEntry[] };

/** An observer of the host's GCs, made for the whole A/B. */
export type GcObserver = {
  /** Every GC that ended before the call, once the host has delivered it. */
  readonly read: () => Promise<GcReading>;
  readonly disconnect: () => void;
};

/** The part of a `PerformanceEntry` a `gc` entry is read from (Node puts the kind in `detail`). */
export type GcHostEntry = {
  readonly startTime: number;
  readonly duration: number;
  readonly detail?: unknown;
};

/** The part of `PerformanceObserver` the observer uses. */
export type GcObserverHost = {
  readonly supportedEntryTypes: readonly string[];
  new (callback: (list: { getEntries(): readonly GcHostEntry[] }) => void): {
    observe(options: { readonly type: string }): void;
    takeRecords(): readonly GcHostEntry[];
    disconnect(): void;
  };
};

const gcEntry = (entry: GcHostEntry): GcEntry => {
  const { detail } = entry;
  if (
    typeof detail !== "object" ||
    detail === null ||
    !("kind" in detail) ||
    typeof detail.kind !== "number"
  ) {
    throw new Error("A gc performance entry has no numeric detail.kind");
  }
  return { startTime: entry.startTime, duration: entry.duration, kind: detail.kind };
};

/** The global `PerformanceObserver`, read when called (`undefined` on a host without one). */
export const globalPerformanceObserver = (): GcObserverHost | undefined =>
  typeof PerformanceObserver === "undefined" ? undefined : PerformanceObserver;

/**
 * Starts observing the host's GCs through `host` ({@link globalPerformanceObserver} in the entry;
 * absent, or without the `gc` entry type: not observable). Every entry is kept for the life of the observer.
 *
 * NOTE: Node delivers a `gc` entry on a later task than the GC itself (observed under Node 24: not
 * synchronously, not after a microtask, not always after one task, always after two), so `read`
 * hands the event loop two tasks before it takes the records not yet delivered. That is Node's
 * behaviour, not a guarantee of the standard.
 */
export const observeGc = (host: GcObserverHost | undefined): GcObserver => {
  if (host === undefined || !host.supportedEntryTypes.includes("gc")) {
    return { read: () => Promise.resolve({ observable: false }), disconnect: () => {} };
  }
  const entries: GcEntry[] = [];
  const keep = (list: readonly GcHostEntry[]): void => {
    for (const entry of list) entries.push(gcEntry(entry));
  };
  const observer = new host((list) => keep(list.getEntries()));
  observer.observe({ type: "gc" });
  return {
    read: async () => {
      await nextTask();
      await nextTask();
      keep(observer.takeRecords());
      return { observable: true, entries: [...entries] };
    },
    disconnect: () => observer.disconnect(),
  };
};

/** The GCs that began inside one visit: the major ones, and every other kind together. */
export type VisitGc = {
  readonly major: number;
  /** The major GCs' durations summed (each holds the thread for its whole duration). */
  readonly majorPauseMs: number;
  readonly other: number;
  readonly otherPauseMs: number;
};

/** The GCs of `entries` whose `startTime` lies inside `span` (both ends included). */
export const visitGc = (span: RunSpan, entries: readonly GcEntry[]): VisitGc => {
  const inside = entries.filter(
    ({ startTime }) => startTime >= span.startMs && startTime <= span.endMs,
  );
  const sum = (list: readonly GcEntry[]): number =>
    list.reduce((total, { duration }) => total + duration, 0);
  const major = inside.filter(({ kind }) => kind === MAJOR_GC_KIND);
  const other = inside.filter(({ kind }) => kind !== MAJOR_GC_KIND);
  return {
    major: major.length,
    majorPauseMs: sum(major),
    other: other.length,
    otherPauseMs: sum(other),
  };
};
