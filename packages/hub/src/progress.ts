/**
 * Fetch progress (ADR 0038 §5, "the progress total is the sum of the manifest's `size`, not
 * content-length").
 *
 * Aggregation is **a convention of the common layer**, not a capability of the source, so every
 * surface (the whole-file and the warm-up surfaces) shares one implementation. Writing it per
 * surface would reproduce mismatches where the way `loaded` accumulates gets fixed on one side
 * only.
 */

import { type FileRef, fileRefKey } from "./manifest.ts";

/**
 * The naming of where a file is fetched from (the structured fields of `FetchContext.originOf` in
 * `context.ts`). Progress copies {@link AssetProgress.repo} / {@link AssetProgress.revision} from
 * it.
 */
type OriginIdentity = { readonly repo?: string; readonly revisionSha?: string };

/**
 * Progress phase. `complete` is the end of one file (the point where its bytes are final).
 *
 * MUST: the phase of one file only moves `downloading`* → `complete` and never goes back
 * (`complete` once per file; no notification for that file follows). The exception is self-heal of
 * a corrupt cache entry: the fetch layer rejects it and fetches again from the network, so this
 * cycle restarts from the beginning (`complete` stays the end).
 *
 * NOTE: there is no `verifying` phase for an ongoing comparison — asset verification is buried
 * inside the source (hashing while receiving / comparing the recorded hash) and cannot be observed
 * from the common layer. Naming an unobservable phase by guess would be a lie that shows a
 * comparison already finished as "in progress".
 */
export type AssetPhase = "downloading" | "complete";

export type AssetProgress = {
  readonly phase: AssetPhase;
  /**
   * The path of the file that raised the event (relative to the repo).
   *
   * MUST NOT: use the path alone as a file's identifier — with cross-repo references (ADR 0038 §7)
   * or component replacement, the same path from different repos appears in one load. To aggregate
   * per file, key on it together with {@link repo} / {@link revision}.
   */
  readonly path: string;
  /**
   * The repo the file was **actually fetched from** (`"owner/name"`). For a cross-repo reference
   * the declared target, otherwise the session's repo. For files of a source without the notion of
   * a repo (a local directory) the field does not appear at all (no synthetic name is given).
   */
  readonly repo?: string;
  /**
   * The generation of {@link repo} (the resolved commit SHA, 40 lowercase hex digits). For a
   * cross-repo reference the declared `revision`, otherwise the session's resolved SHA. For a
   * source without generations the field does not appear at all.
   */
  readonly revision?: string;
  /** Total bytes fetched (all files, after deduplicating identical files). */
  readonly loaded: number;
  /** Sum of the manifest's `size` (after deduplicating identical files). */
  readonly total: number;
  /**
   * Bytes received for **the file of `path` itself**. `loaded` is the total over all files, while
   * this is for one file, so per-file progress bars are drawn from this value and
   * {@link fileTotal}.
   *
   * `complete` is the point where everything has arrived, so `fileLoaded === fileTotal` always
   * holds there (a cache hit, which never emits `downloading`, emits only the single `complete`).
   */
  readonly fileLoaded: number;
  /**
   * The manifest size of the file of `path` itself (`FileRef.size`). `total` is the sum of these.
   */
  readonly fileTotal: number;
};

/** Progress aggregator for one fetch surface. Its state (bytes received) stays closed in here. */
export type ProgressEmitter = {
  /** Records the cumulative bytes received so far and emits `downloading`. */
  readonly downloading: (ref: FileRef, loaded: number) => void;
  /** The end of one file. */
  readonly complete: (ref: FileRef) => void;
};

/**
 * Looks up the manifest size that goes into the progress's `fileTotal`. The table is built from the
 * fetch targets themselves, so a ref being fetched is always found — a miss means an internal
 * invariant is broken, so it throws (filling in 0 would make per-file progress bars silently draw
 * broken values).
 */
const fileSizeOf = (refs: ReadonlyMap<string, FileRef>, key: string): number => {
  const ref = refs.get(key);
  if (ref === undefined) {
    throw new Error(`hub: ${key} の size が取得対象の表に無い（進捗集計の不変条件破れ）`);
  }
  return ref.size;
};

/**
 * Creates an aggregator from the table of fetch targets (keyed by {@link fileRefKey}). `total` is
 * fixed at the sum of this table's `size` and never changes afterwards.
 *
 * MUST: pass as `originOf` **the same function** that names errors (`FetchContext.originOf`) —
 * spelling "is it cross-repo" separately for progress would make diagnostics and progress name
 * different origins for the same file.
 *
 * MUST: record the bytes received (`received`) before notifying (`emit`) — `emit` reads its own
 * file's share from `received` as well, so in the reverse order the emitted value is one event
 * stale (the first downloading reports `fileLoaded` 0 and stays one value behind from then on).
 * `fileLoaded` and the overall `loaded` always count the same value (`emit` adds everything except
 * its own key), so they cannot disagree in either order.
 */
export const createProgressEmitter = (
  targets: ReadonlyMap<string, FileRef>,
  originOf: (ref: FileRef) => OriginIdentity,
  onProgress?: (progress: AssetProgress) => void,
): ProgressEmitter => {
  let total = 0;
  for (const ref of targets.values()) total += ref.size;

  const received = new Map<string, number>();
  const emit = (phase: AssetPhase, ref: FileRef): void => {
    if (onProgress === undefined) return;
    const refKey = fileRefKey(ref);
    const fileTotal = fileSizeOf(targets, refKey);
    // complete is the point where everything has arrived, so pass the size as is (a cache hit never
    // emits downloading and is not in `received`, so looking it up there would show 0).
    const fileLoaded = phase === "downloading" ? (received.get(refKey) ?? 0) : fileTotal;
    // MUST: add the same value to the overall `loaded` — taking this file's share from `received`
    // would contradict within one event: fileLoaded is the size while loaded does not count it (a
    // start with every file cached would line up completes with loaded stuck at 0). In downloading,
    // fileLoaded is the value in `received` itself, so nothing is counted twice.
    let sum = fileLoaded;
    for (const [other, bytes] of received) {
      if (other !== refKey) sum += bytes;
    }
    const origin = originOf(ref);
    const progress: AssetProgress = {
      phase,
      path: ref.path,
      ...(origin.repo === undefined ? {} : { repo: origin.repo }),
      ...(origin.revisionSha === undefined ? {} : { revision: origin.revisionSha }),
      loaded: sum,
      total,
      fileLoaded,
      fileTotal,
    };
    // MUST: isolate listener exceptions in this one place and treat every phase (downloading /
    // complete) the same — progress is incidental information, not a correctness requirement (the
    // same policy as the fetch layer isolating downloading). If the isolation differed per phase,
    // the same UI-side exception would change meaning: "harmless while receiving, a failure of the
    // whole load at the end (taking the other files' fetches down with it)". Only that one
    // notification is dropped, and not silently (it is reported with warn).
    try {
      onProgress(progress);
    } catch (error) {
      console.warn(
        `@karume/hub: onProgress リスナーが throw した（${phase} ${ref.path} — この通知だけを捨てて` +
          `取得は続行する）`,
        error,
      );
    }
  };

  return {
    downloading: (ref, loaded) => {
      received.set(fileRefKey(ref), loaded);
      emit("downloading", ref);
    },
    complete: (ref) => {
      // MUST: write the size into `received` too — emitting only complete would drop a file that
      // was a cache hit (no downloading ever emitted) from the `loaded` total of later events, and
      // the overall progress would appear to go backwards.
      received.set(fileRefKey(ref), ref.size);
      emit("complete", ref);
    },
  };
};
