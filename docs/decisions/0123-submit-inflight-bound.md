# 0123: Bound how far `queue.submit` runs ahead of the GPU

- Status: accepted (2026-10-07, the maintainer: "both as recommended, go ahead"); implemented
  2026-10-07 (`7fefb2ad`). The values 64 / 2 rest on Deno 2.9.6 measurements; the need and behaviour
  on Node + Dawn (ADR 0127) are not measured (noted 2026-10-18)
- Date: 2026-10-07
- Related: ADR [0004](0004-execution-model.md) (execution model — submit time budgets; this ADR adds the
  in-flight bound as revision 5) / [0032](0032-deserialize-submit-and-optin-timing.md) (no
  `onSubmittedWorkDone` per submit — kept; this ADR adds only spaced completion markers) /
  [0054](0054-resident-loop-and-fence.md) (one `mapAsync` fence per normal run, BatchScope, the
  ordering invariant) / [0121](0121-wan22-ti2v-5b.md) addendum (2026-10-07), step 10 (the device loss
  while writing the reference preset's 121-frame row) / research
  [2026-08-04-host-overhead-recon](../research/2026-08-04-host-overhead-recon.md) §4.1 (Deno's
  `onSubmittedWorkDone` blocks the host until completion in its synchronous part)

## Context

On the development machine (RTX 3080 Ti, NVIDIA 615.71.09, Deno 2.9.6, Vulkan, GPU also driving the
display), Wan2.2's reference preset `f16+dit8` at 1280×704×121 (S = 27,280) lost the device in the
second DiT forward in 4 of 5 runs, and Deno panicked (`ext/webgpu/queue.rs:109`). The investigation
of 2026-10-07 (`outputs/diag/wan22-121f-device-lost-2026-10-07/`) ruled out long submits (longest
0.67 s), heat and kernel-side recovery (no NVRM Xid); every failure was preceded by the
1,084th–1,100th `queue.submit` of that forward blocking for about 11 s. The host had no bound on its
lead: no per-submit wait (ADR 0032) and one final `mapAsync` fence per run (ADR 0054). A
reference-preset forward queues about 1,926 submits in 0.4 s for about 80 s of GPU work; a
practical-preset forward queues about 573 and never failed. Speculation: something in the driver
overflows and `vkQueueSubmit`'s internal wait times out. The mechanism cannot be observed; bounding
the lead removes the overflow whatever it is.

## Decision

1. **Completion markers**: while a run queues dispatches, it calls `SubmitScheduler.backpressure()`
   between steps; when the submits in the measurement window since the previous marker reach
   **`IN_FLIGHT_MARKER_INTERVAL` (64)**, it places one marker with `queue.onSubmittedWorkDone()`. There
   is no marker per submit (ADR 0032's MUST NOT stands). A marker is not a fence (it does not close
   the measurement window).
2. **Counting unit is the measurement window** (per Session scheduler). The window closes right after
   a fence (flush, a single-fence run's `mapAsync`, a batch's settlement), so its chunk count is "the
   submits since the GPU's completion was last confirmed by a fence". Closing or discarding the window
   resets the marker position to 0. Runs with fewer than 64 submits between fences (short repeated
   runs such as decode) place no marker and never wait.
3. **Waiting**: when more than **`IN_FLIGHT_MAX_MARKERS` (2)** markers are pending, the run waits for
   the oldest (one marker is placed at a time, so 3 are pending at that moment and 2 after the oldest
   settles; the lead is at most about 3 × 64 = 192 submits). Within the bound nothing waits — no
   Promise, no microtask; a run that never reaches the bound makes `executeBakedPlanPaced` return no
   Promise and queue synchronously as before. Before waiting, unsubmitted encoders are flushed so that
   a `queue.writeBuffer` issued during the wait cannot overtake unsubmitted dispatches (ADR 0004
   invariant 4 holds during the wait). Device loss is turned into `GpuDeviceLostError` through
   `raceDeviceLost` (an implementation may never resolve `onSubmittedWorkDone` after loss).
4. **Only a run places markers and waits (between steps, in `executeBakedPlanPaced`).** A run queues
   inside the errorScope section lock, which excludes other runs and batches, so nobody's dispatch can
   interleave during the wait.
   **MUST NOT: an enqueue neither waits nor places markers** (unchanged behaviour). Enqueues of
   different Sessions in one batch are not serialised by the section lock; their order rests on each
   Session's serialisation (`#chain`) having the same depth. If an enqueue waited, a later enqueue of
   another Session would run its body and reach the queue first: waiting mid-body breaks the order in
   the first round, waiting at the end in the second (A1 → B1 → A2 → B2 issued without `await`: A2 is
   held behind A1's wait while B2 runs ahead). Across a resident tensor that reads stale data silently
   (confirmed by fault injection on a real GPU and in an independent review's model). A marker without
   a wait is not placed either: where `onSubmittedWorkDone` is asynchronous it bounds nothing, and on
   Deno 2.9.6 it stalled the host in batch generation loops instead (see the alternatives).
5. **How it behaves per runtime** (a description, not a rule): on Deno 2.9.6
   `onSubmittedWorkDone` blocks the host in its synchronous part until all submitted work completes
   (research 2026-08-04 §4.1), so a Deno run leads by at most 64 submits and the wait only awaits a
   settled Promise; the GPU idles only while the next chunk is queued. In browsers (asynchronous
   `onSubmittedWorkDone`) a run leads by at most about 192. Enqueue leads are unbounded on both, as
   before. Node + Dawn has not been measured.
6. **Numbers do not change**: the content and order of queued commands are unchanged; only a run's
   chunk boundaries may move (flushing before a wait), and command buffer boundaries do not change
   execution semantics. sha256 reference values stayed unchanged for every series (full verify at
   `7fefb2ad`).

### Basis of the values

- The failure came at a lead of about 1,080 submits; one measurement variant started to stall at
  about 300 without failing; the practical preset's 573 per forward never failed. The bound is far
  below these, and large enough that waits do not cost speed: one wait per 64 submits (on Deno about
  30 waits per reference-preset forward, about 9 per practical-preset forward).
- These values come from Deno 2.9.6 measurements only; whether Node + Dawn needs the bound, and at
  which values, is not measured. The bound is kept because it costs little and protects on any
  runtime (force triage of 2026-10-18).

## Alternatives not taken

- **Count per device** (a counter on `GpuContext`): nothing resets it at fences, so short decode runs
  would hit 64 cumulatively and stall the host (on Deno), and fence-counting tests would depend on
  what ran before. Waiting runs are serialised by the section lock, so a per-scheduler count is
  already device-wide.
- **Let enqueue wait too (mid-body or at the end)**: breaks call order (decision 4). Keeping order
  while waiting needs a batch-wide FIFO for enqueue bodies — a new serialisation that changes the
  BatchScope contract and needs a deadlock review. Not taken here (the failure was on the run path).
  The same FIFO would also close an existing case (a first enqueue with an `await` in its body being
  overtaken by a later enqueue of another Session) — a separate decision.
- **Markers without waiting on enqueue**: bounds the lead only where `onSubmittedWorkDone` blocks
  synchronously (Deno) — an implementation quirk, not a specification guarantee — and changes nothing
  in browsers, while adding a GPU-idle stall per marker to Deno batch loops (irodori's DiT loop, Wan's
  VAE chunks). Long enqueue batches (about 20,000 submits for the 121-frame VAE) never failed.
  Implemented once and removed after the independent review.
- **Bound by time** (estimated GPU time in flight): no estimate exists for a run's first execution,
  window measurements include host time, and it is unknown whether count or time overflowed. A count
  is deterministic and testable.
- **Move the display to the integrated GPU, or change driver settings**: would stabilise the
  development machine but leaves the problem for users whose NVIDIA GPU also drives the display.
- **Drop the reference preset's 121 frames from acceptance**: the missing bound does not depend on
  preset or length; the same failure can return with another large run.

## Consequences

- `SubmitScheduler.backpressure()` exists (it returns a Promise only above the bound;
  `packages/runtime/src/gpu/submit.ts`), called only from `executeBakedPlanPaced`
  (`packages/runtime/src/runtime/recipe.ts`). Only run and enqueue baked execution
  (`dispatchWithWork`) queue dispatches on the scheduler; weight uploads and GenerationContext writes
  use `queue.writeBuffer` and bypass it.
- The bound is per Session scheduler (device-wide one bound's worth, since runs are serialised by the
  section lock). Waits happen between steps (one node, fusion included), so the submits of one step
  can exceed the bound in browsers.
- On Deno 2.9.6 the JS thread is blocked during a marker's synchronous part (64 submits, about 2.7 s
  for the reference preset's DiT), delaying timers and progress callbacks; progress is reported
  between steps, so what users see does not change. A device loss inside that synchronous part
  panics Deno instead of raising `GpuDeviceLostError` (as with flush — `docs/known-issues.md`, Deno
  device-loss item). The behaviour on Node + Dawn is not measured.
- During a run's wait the errorScope stays pushed (inside the section lock). Layers that issue GPU
  operations outside the lock use only scopes that complete synchronously (the `GpuContext` errorScope
  invariant), so attribution is unchanged. The window in which an unscoped operation (a user's
  `ResidentTensor.write`) can be attributed to this run already existed for miss runs (awaiting
  pipeline creation) and now extends to hit runs.
- Fence-counting tests whose runs submit 64 or more times between fences see more
  `onSubmittedWorkDone` calls (enqueue sees none); no small-graph test in the repository reaches 64.

## Acceptance

Done 2026-10-07: host tests, the real-GPU test `packages/runtime/tests/gpu_submit_backpressure_test.ts`
(now under Vitest — ADR 0129) and 10 fault injections; the 1280×704×121 reference preset completed
5 of 5 runs; full `deno task verify` at `7fefb2ad` green (4,003 passed, sha256 references unchanged;
per-file time 7,298 → 7,057 s). Records: `.claude/reviews/2026-10-07_submit-inflight/` (untracked).
