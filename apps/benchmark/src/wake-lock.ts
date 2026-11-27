/**
 * The screen wake lock a run holds where the browser has the API: asked when the run starts and
 * again whenever the page is shown and holds none (the browser releases it while the page is
 * hidden), and its state followed from the lock's `release` event, so the file says what was true
 * when the run ended. The API is injected, so the rules are tested without a browser.
 *
 * SPEC: docs/spec/benchmark.md#while-it-runs
 */

import type { WakeLockState } from "./bundle.ts";

/** The part of a `WakeLockSentinel` the holder reads. */
export type WakeLockLock = {
  readonly release: () => Promise<void>;
  readonly addEventListener: (type: "release", listener: () => void) => void;
};

/** The part of `navigator.wakeLock` the holder uses. */
export type WakeLockApi = { readonly request: (type: "screen") => Promise<WakeLockLock> };

export type WakeLockHold = {
  /** `held` while a lock is held; `not-held` while none is (released by the browser, or not granted yet); `failed` after a refused request; `unavailable` without the API. */
  readonly state: () => WakeLockState;
  /** Asks again when the page holds no lock and no request is pending (call it when the page is shown). */
  readonly renew: () => void;
  /** Ends the hold: releases the lock, and one granted later at once. */
  readonly release: () => void;
};

/** Holds a screen wake lock through `api` (undefined: the browser has none) until `release`. */
export const holdWakeLock = (api: WakeLockApi | undefined): WakeLockHold => {
  if (api === undefined) {
    return { state: () => "unavailable", renew: () => {}, release: () => {} };
  }
  let state: WakeLockState = "not-held";
  let lock: WakeLockLock | undefined;
  let pending = false;
  let ended = false;
  const request = (): void => {
    if (ended || pending || lock !== undefined) return;
    pending = true;
    api.request("screen").then(
      (granted) => {
        pending = false;
        if (ended) {
          void granted.release();
          return;
        }
        lock = granted;
        state = "held";
        granted.addEventListener("release", () => {
          if (lock !== granted) return;
          lock = undefined;
          // The hold's own release at the end leaves the state the run ended with
          if (!ended) state = "not-held";
        });
      },
      () => {
        pending = false;
        if (!ended) state = "failed";
      },
    );
  };
  request();
  return {
    state: () => state,
    renew: request,
    release: () => {
      ended = true;
      void lock?.release();
    },
  };
};
