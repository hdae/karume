/**
 * How long the page was hidden while a run went (another tab in front, a minimised window): a
 * browser may slow a hidden page's timers and its GPU work, so the run's file says so. Pure over the
 * moments it is told (`performance.now()` milliseconds).
 *
 * SPEC: docs/spec/benchmark.md#while-it-runs
 */

export type HiddenLog = {
  /** The page became hidden (`true`) or visible (`false`) at `at`; a repeat of the state is ignored. */
  readonly note: (hidden: boolean, at: number) => void;
  /** How often the page was hidden and for how long in all, up to `at`. */
  readonly summary: (at: number) => {
    readonly hiddenCount: number;
    readonly hiddenSeconds: number;
  };
};

/** A log that starts at `at` with the page hidden or not. */
export const hiddenLog = (hidden: boolean, at: number): HiddenLog => {
  let since: number | undefined = hidden ? at : undefined;
  let count = hidden ? 1 : 0;
  let total = 0;
  return {
    note: (nowHidden, when) => {
      if (nowHidden && since === undefined) {
        since = when;
        count += 1;
      } else if (!nowHidden && since !== undefined) {
        total += when - since;
        since = undefined;
      }
    },
    summary: (when) => ({
      hiddenCount: count,
      hiddenSeconds: Number(((total + (since === undefined ? 0 : when - since)) / 1000).toFixed(3)),
    }),
  };
};
