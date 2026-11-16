/**
 * The real session A/B runner with a test clock: every read of the clock opens or closes the span
 * it times, so a fake arm can tell whether it is inside a timed span.
 */

import { runSessionAb, type SessionAbArm, type SessionAbRecord } from "@karume/runtime/tune";

/** The environment the runner's record names in these tests. */
export const ENVIRONMENT = {
  adapter: { vendor: "test", architecture: "", device: "", description: "" },
  userAgent: "test",
  backend: "other",
  date: "2026-11-11T00:00:00.000Z",
};

export type SpanRunner = {
  /** Whether the runner's clock is open now (inside a timed span). */
  readonly inSpan: () => boolean;
  /** The runner's clock: every read opens or closes the span. */
  readonly clock: () => number;
  readonly run: (
    arms: readonly SessionAbArm[],
    rounds: number,
    openPerVisit: boolean,
  ) => Promise<SessionAbRecord>;
};

export const spanRunner = (): SpanRunner => {
  let open = false;
  let tick = 0;
  const clock = () => {
    open = !open;
    tick += 1;
    return tick;
  };
  return {
    inSpan: () => open,
    clock,
    run: (arms, rounds, openPerVisit) =>
      runSessionAb({ arms, rounds, openPerVisit, environment: ENVIRONMENT, clock }),
  };
};
