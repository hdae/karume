# Generation: what a decode step does on the host

What the generation loop of `@karume/models` (`packages/models/src/generation/`) does now between two decode runs of a
language model: where it checks for an abort. The rest of the
generation surface — the sequence, its turns, the sampler's processing order, speculation, the decode run-ahead — is
held by the ADRs that decided it ([0083](../decisions/0083-generation-api-surface.md),
[0096](../decisions/0096-speculative-decoding.md), [0142](../decisions/0142-decode-run-ahead.md)) until it has a spec
here.

The rule below changes where a decode token's time goes and does not change a token: the same request with the same
seed generates the same token ids with and without it.

## Abort checks

A generation given a `signal` checks it at every stage boundary — right before a run is issued — and throws
`signal.reason` as it is.

A check sees an abort only if the task that delivers it has run, so the event loop has to have turned since the check
before it:

- **A prefill chunk** yields to the event loop at its own boundary: it waits for a `setTimeout(0)` and then checks
  (`settleAbort`).
- **A decode boundary** — of the plain decode, of a speculative cycle, of a self-speculation cycle over the expert
  cache, of a run-ahead batch — takes that yield while the run before it is awaited (`createAbortLap`). Each boundary
  arms a `setTimeout(0)` for the next one and does not wait for it. The next boundary:
  - checks at once when that timer was delivered in the meantime (the run was awaited across a turn of the event
    loop's timers);
  - otherwise drops it and yields as a prefill chunk does, with a timer armed at that moment. That is the case at the
    first decode boundary of a turn, after a run that settled without the event loop turning, and after a run shorter
    than the timer's delay.
- A generation without a `signal` arms no timer and never yields.
- The timer armed at a turn's last boundary is dropped when the turn ends, however it ends.

The evidence that the event loop turned is the delivered timer, not the fact that a run was awaited. The timer is a
`setTimeout(0)` in both forms, so an abort that a timer armed earlier delivers is never overtaken by the yield.

What this means for an abort during decode:

| The abort | Is seen |
| --- | --- |
| `abort()` called by the consumer between two tokens | at the next boundary, before the next run |
| delivered by a task that ran while a decode run was awaited | at the next boundary: that run's token is delivered, no run follows |
| delivered by a task still queued when a decode run returned (it became ready after the event loop's last turn inside that run) | one boundary later: one more run and one more token |
| delivered by a timer, during a loop whose runs settle without the event loop turning | at the next boundary, before the next run |

A decode token pays no timer of its own in a visible tab or under Node. In a hidden browser tab timers are throttled,
none is delivered within a run, and every boundary waits for one ([limitations](../limitations.md)).
