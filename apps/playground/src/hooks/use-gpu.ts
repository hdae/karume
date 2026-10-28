import { acquireGpu, type AcquireGpuOptions, type GpuContext } from "@karume/runtime";
import { createContext, useCallback, useContext, useEffect, useState } from "react";

export type GpuState =
  | { readonly kind: "acquiring" }
  | { readonly kind: "ready"; readonly gpu: GpuContext }
  | { readonly kind: "lost"; readonly reason: string; readonly message: string }
  | { readonly kind: "error"; readonly error: unknown };

export type GpuOptions = Omit<AcquireGpuOptions, "onDeviceLost">;

/**
 * Holds the device until `work` settles and returns `work` as is (it does not handle a rejection:
 * a caller that drops the returned promise MUST catch it). A page MUST pass every pipeline /
 * Session disposal and every run or build it starts on the device through it, and MUST start a
 * disposal no later than the effects of the commit that releases the device (its cleanups, or an
 * effect keyed on the device): the device is destroyed (on leaving the page, on `reacquire`, or when
 * `options` change) only after all tracked work has settled, so GPU work never runs on a destroyed
 * device (flush before destroy).
 */
export type TrackGpuWork = <T>(work: Promise<T>) => Promise<T>;

/**
 * One acquisition of the page's GPU: the (`options`, attempt) it was made for. A new object whenever
 * either changes — also when the options return to earlier values — so a device released with an
 * older lease is never taken for the current one. `useRunTask` aborts a run when its lease ends.
 */
export type GpuLease = { readonly options: GpuOptions; readonly attempt: number };

export type GpuHandle = {
  readonly state: GpuState;
  readonly lease: GpuLease;
  readonly reacquire: () => void;
  readonly track: TrackGpuWork;
};

/**
 * The app's device queue: the destroy of the last released device, which every acquisition waits
 * for. One per app (the root layout provides it through {@link GpuQueueContext}), so the next page —
 * or a new instance of the same page — acquires only after the previous page's device is destroyed.
 */
export type GpuQueue = { released: Promise<void> };

export const createGpuQueue = (): GpuQueue => ({ released: Promise.resolve() });

export const GpuQueueContext = createContext<GpuQueue | undefined>(undefined);

/**
 * The device lifecycle behind {@link useGpu}, without React: `open` acquires a device (once the
 * queue's last released device is destroyed) and returns its release; `track` holds a release until
 * work settles. The tracked work is the lifecycle's own (one page instance), so a page's work never
 * delays another page's destroy.
 */
export type GpuLifecycle = {
  readonly open: (options: GpuOptions, onState: (state: GpuState) => void) => () => void;
  readonly track: TrackGpuWork;
};

/**
 * Waits until every tracked promise has settled, including ones tracked while waiting (a disposal a
 * finishing run starts).
 */
const settleAll = async (pending: ReadonlySet<Promise<unknown>>): Promise<void> => {
  while (pending.size > 0) await Promise.allSettled([...pending]);
};

const nextMacrotask = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * Teardown order: a release only marks the device for destruction. The destroy waits one macrotask
 * — by then React has run the remaining cleanups of the same commit (the page's disposals and
 * `useRunTask`'s abort), which start their work through `track` — and then until every tracked
 * promise has settled. The next `open` (of any page sharing `queue`) acquires only after that
 * destroy, so two devices that hold work never exist at once.
 *
 * A release that comes while still acquiring does not hold up the queue: that device has no work
 * yet and is destroyed as soon as it arrives, so the next acquisition does not wait for it (an
 * adapter or device request that never settles would otherwise block every later acquisition).
 */
export const createGpuLifecycle = (queue: GpuQueue): GpuLifecycle => {
  const pending = new Set<Promise<unknown>>();

  const open = (options: GpuOptions, onState: (state: GpuState) => void): (() => void) => {
    let disposed = false;
    let acquired: GpuContext | undefined;
    void queue.released
      .then(() => {
        if (disposed) return undefined;
        return acquireGpu({
          ...options,
          onDeviceLost: (info) => {
            if (!disposed) onState({ kind: "lost", reason: info.reason, message: info.message });
          },
        }).then((gpu) => {
          // Released (the page left, or React re-ran the effect) while acquiring.
          if (disposed) gpu.destroy();
          else {
            acquired = gpu;
            onState({ kind: "ready", gpu });
          }
        });
      })
      .catch((error: unknown) => {
        if (!disposed) onState({ kind: "error", error });
      });
    return () => {
      disposed = true;
      const gpu = acquired;
      if (gpu === undefined) return;
      // Chained on the previous value rather than replacing it: that value settled before this
      // device was acquired, unless a `runOnGpuQueue` task took the queue since then, and the next
      // acquisition must wait for that task too.
      const previous = queue.released;
      // A destroy that throws is reported here and still lets the queue move on: a rejected
      // `released` would fail every later acquisition of the app.
      const destroyed = nextMacrotask()
        .then(() => settleAll(pending))
        .then(() => gpu.destroy())
        .catch((error: unknown) => {
          console.error("Destroying the GPU device failed", error);
        });
      queue.released = Promise.all([previous, destroyed]).then(() => undefined);
    };
  };

  const track = <T>(work: Promise<T>): Promise<T> => {
    const settled = work.then(
      () => undefined,
      () => undefined,
    );
    pending.add(settled);
    void settled.then(() => pending.delete(settled));
    return work;
  };

  return { open, track };
};

/** A device state and the lease it was reported for. */
export type HeldGpuState = { readonly lease: GpuLease; readonly state: GpuState };

const ACQUIRING: GpuState = { kind: "acquiring" };

/** The lease of a render: `previous` while (`options`, `attempt`) are the same, else a new one. */
export const leaseFor = (previous: GpuLease, options: GpuOptions, attempt: number): GpuLease =>
  previous.options === options && previous.attempt === attempt ? previous : { options, attempt };

/** What a render shows: the held state when it was reported for this very lease, else "acquiring". */
export const leaseState = (held: HeldGpuState | undefined, lease: GpuLease): GpuState =>
  held !== undefined && held.lease === lease ? held.state : ACQUIRING;

/**
 * Runs `task` — work that acquires and destroys a device of its own, such as `runGeometrySweep` —
 * as the app's GPU user for as long as it runs: `task` starts once the queue's last released device
 * is destroyed, and the queue's next acquisition waits until `task` has settled (its device is
 * destroyed before its promise settles). Returns `task`'s promise as is (a caller that drops it MUST
 * catch it).
 *
 * The queue is taken synchronously, in the call itself: a page that calls this from an event
 * handler holds the queue before any later commit runs, so the next page's `useGpu` — whose `open`
 * reads `queue.released` in its effect, after the leaving page's cleanups — waits for `task`. The
 * caller aborts `task` when the page is left (its cleanup), and `task` MUST check its signal before
 * it acquires: a task that is still waiting for the previous destroy then settles without a device.
 * Unlike a `useGpu` release there is no macrotask before the queue moves on: `task` owns its device
 * alone, so no cleanup of the same commit can still start work on it.
 *
 * Known limitation: the queue moves on only when `task` settles. A `useGpu` release while acquiring
 * lets the queue go at once, but this function cannot see inside `task`: if `task` waits on an
 * adapter or device request that never settles — `runGeometrySweep` consults no signal while it
 * acquires — the app's GPU queue is held for good, and every later acquisition of any page waits.
 * Reloading the tab is the only recovery.
 *
 * MUST NOT be called while a `useGpu` device of the same queue is live (acquired and not released):
 * the queue's value already settled when that device was acquired, so `task` would start at once
 * and two devices would hold work together. (The two writers still compose: that device's release
 * chains on the queue, so the next acquisition waits for both.) Every `useGpu` caller is a page
 * route, so a page without `useGpu` meets this.
 */
export const runOnGpuQueue = <T>(queue: GpuQueue, task: () => Promise<T>): Promise<T> => {
  const run = queue.released.then(task);
  queue.released = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
};

/** The app's device queue (the root layout provides it through {@link GpuQueueContext}). */
export const useGpuQueue = (): GpuQueue => {
  const queue = useContext(GpuQueueContext);
  if (queue === undefined) {
    throw new Error("The GPU queue needs a GpuQueueContext provider (the root layout provides it)");
  }
  return queue;
};

/**
 * One GPU per page: acquired on mount, destroyed on unmount ({@link createGpuLifecycle} has the
 * order). `reacquire` destroys the current one and acquires again (after a device loss). `options`
 * MUST be stable (a module constant) — a new object would re-acquire on every render.
 *
 * The state belongs to the lease it was acquired for: in the render that changes the options or the
 * attempt, `state` is already "acquiring", so the page's effects keyed on the device clean up (and
 * start their disposals) in the same commit as this hook's release — before the destroy's macrotask.
 */
export const useGpu = (options: GpuOptions): GpuHandle => {
  const queue = useGpuQueue();
  const [lifecycle] = useState(() => createGpuLifecycle(queue));
  const [attempt, setAttempt] = useState(0);
  // The previous-render pattern: a new lease is stored in the render that makes it.
  const [storedLease, setStoredLease] = useState<GpuLease>(() => ({ options, attempt }));
  const lease = leaseFor(storedLease, options, attempt);
  if (lease !== storedLease) setStoredLease(lease);
  const [held, setHeld] = useState<HeldGpuState | undefined>(undefined);

  useEffect(
    () => lifecycle.open(lease.options, (state) => setHeld({ lease, state })),
    [lifecycle, lease],
  );

  const reacquire = useCallback(() => setAttempt((count) => count + 1), []);
  return { state: leaseState(held, lease), lease, reacquire, track: lifecycle.track };
};
