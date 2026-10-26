# 0054: GPU-resident execution loop and fence consolidation (resident tensors / batch enqueue / single-fence run)

- Status: accepted (2026-08-13 — main-session decision under the maintainer's approval of
  performance phase 2, H-5 option A + H-1); the decode-only scope of the 2026-09-20 addendum
  re-graded to SHOULD on 2026-10-18.
- Date: 2026-08-13
- Scope: runtime (`gpu/context.ts`, `gpu/submit.ts`, `runtime/executor.ts`) and models (Irodori,
  `packages/models/src/irodori/dit-loop.ts`). IR, exporter and assets unchanged (no re-export).
- Related: [0004](0004-execution-model.md) (invariant 4) / [0021](0021-gpu-timing-diagnostics.md).

## Context

[research/2026-08-13-host-cost-decomposition.md](../research/2026-08-13-host-cost-decomposition.md)
showed that **waiting on fences at run boundaries dominated wall time**: about 11 ms per wait, 51 of
the 54 ms of a bare EG run, 7.5 s of an 8.2 s Irodori run, and 12 ms per run between a two-stage and a
single wait.

## Decision

1. **ResidentTensor — a fourth lifetime class, owned by GpuContext.** The other three (weight arena,
   slot backing, run arena) are closed inside a Session, so passing data between Sessions went
   through the host at two fences. A resident tensor **holds only bytes and a size** (no dtype or
   shape it would have to invent). It is built only by `GpuContext.createResident` (async, inside an
   error scope — a failed allocation is otherwise a silently invalid buffer) and is destroyed only
   when no baked bind group refers to it (fail loudly — a silent destroy corrupts unrelated dispatches).
2. **BatchScope + Session.enqueue — a fence-free span.** One pair of error scopes (out-of-memory +
   validation) wraps the span; the only fence is `finish()`. **The no-overtaking invariant in a new form
   (extends ADR 0004 invariant 4)**: enqueue MUST end with an eager submit; `queue.writeBuffer` lands on
   the queue timeline in issue order, so it cannot overtake a submitted dispatch. Accepted trade-offs:
   failures are attributed per batch, and device loss is seen only at `finish`. Incompatible with
   gpuTiming (fails loudly — per-dispatch timestamps would pile up unread).
3. **Writes into a resident tensor are GPU copies** (`copyOutputs`: output slot → resident, FIFO in the
   dispatch's command stream). Output redirection is rejected: it needs ping-pong buffers, then
   alternating binding identities, then several backings. A copy keeps the binding identity stable and
   **a single backing hitting**, with no extra mechanism (microseconds of GPU time).
4. **The Irodori DiT loop runs resident**: latent, velocity, CFG intermediates and the three conditions
   are resident, and the loop (40 steps × 1+V forwards) is one batch. CFG combination and the Euler
   update are **small IR graphs built on the host** (3 + 2 nodes,
   `packages/models/src/irodori/host/sampler-graph.ts`) on a separate Session. **Their association,
   variant and argument order match the TS source of truth (`host/sampler.ts`) op by op** (f32
   addition is not associative; a mismatch splits the WAV sha256). This works because the loop is fully
   static (CFG window and variant set depend only on the step index). A device with gpuTiming takes the
   host loop (decision 2); the digest is the same.
5. **Numerics: the reference path** (the branch the maintainer approved). A parity probe over 135,936
   elements (denormals, signed zeros, fma-sensitive cases) found no fma contraction (one op per
   dispatch), identical signed zeros, and one difference: flush-to-zero of 5,670 elements **whose final
   output is denormal** (practically unreachable with real data). The **two WAV sha256 checks of the
   real pipeline match** — bit identity on the reference cases. The denormal caveat is in
   `docs/limitations.md`; the WAV checks are its permanent detector.
6. **H-1 — one fence for an ordinary run**: with gpuTiming off and at least one graph output, the
   readback copy goes into the run's own command stream (FIFO: `mapAsync` resolving implies the copy
   and every earlier dispatch finished), and `mapAsync` is the only fence. The fences of flush
   `onSubmittedWorkDone` and of the re-flush in `arena.destroy` go away; arena cleanup only submits
   (flush-before-destroy = no unsubmitted encoding left). gpuTiming on (ADR 0021 — timestamp readback
   needs the flush) and graphs with no outputs (nothing to wait on) keep the two-stage wait
   (`singleFence` in `packages/runtime/src/runtime/executor.ts`).
7. **Reduced observability (by design, `docs/limitations.md`)**: resident-path enqueue creates no run
   arena and no timing window, so `lastRun` / `lastRunTiming` are `undefined`. A per-op breakdown needs
   gpuTiming, which runs the host loop (about 2× wall time). Pipeline keys are readable without timing
   through `lastRunPipelines`.

## Acceptance (2026-08-13)

RTX 3080 Ti / Vulkan, full suite green with every digest unchanged; 15 new checks for fence count,
overtaking and lifetime, proven by fault injection. Bare EG run 52.54 → 28.15 ms (×1.86); Irodori
voice-clone 8.593 → 4.896 s (×1.76), WAV sha256 unchanged. Numbers: host-cost research §6; commits
`c90bd43` (runtime), `8ab141a` (Irodori loop), `e339cc0` (single-fence run).

## Addendum (2026-08-16) — in-flight lease

`BatchScope` takes an **in-flight lease** in the synchronous part of `Session.enqueue` (before
`#chain`) and returns it in the body's `finally`. `finish()` rejects new enqueues, waits for every
lease, submits everything pending, then waits on one fence. Without it, `finish()` called without
awaiting the enqueue promises (1) rejected every enqueue whose body had not yet run (the body runs one
microtask later), settling a zero-dispatch span as a success with unhandled rejections — reproduced on
a real GPU (review pass 2, V-1) — and (2) let running enqueues submit after the fence, returning with
GPU work in flight. Like the repository's other ordering contracts (`Session.#chain`, the error-scope
span lock, baked reference counts) this is closed by a mechanism, not prose. The surface is unchanged,
and with no outstanding lease no microtask is added; the Irodori WAV sha256 is unchanged.

## Addendum (2026-08-16) — onEvent takes the host path

Generation events (`onEvent`) cannot be observed inside one batch with one fence, so a subscription
switches to the host loop, as gpuTiming does. Output is bit-identical; wall time 7.2 → 8.6 s (S 170).
See [limitations](../limitations.md).

## Addendum (2026-08-30) — failure attribution includes one host-side failure

The lease did not cover an enqueue body that **ran and threw** (host-side checks, recipe building):
the lease returned in `finally`, the error scope saw only GPU failures, and the span settled **as a
success** while the throw escaped as an unhandled rejection — exactly the unawaited use this ADR is
for. `BatchScope` now records the **first** host-side failure of the span and `finish()` throws it.
When the error scope also failed, that failure is thrown (not rewrapped) with the record as `cause`.
Later failures are dropped: derived failures hide the root cause (as the error scope prefers internal /
out-of-memory). `finish()` throwing `ExecutionError` and similar is a **breaking change** relative to
0.7.0 ([limitations](../limitations.md)). The failure still appears on the enqueue's promise as well.

## Addendum (2026-09-12) — batch-end readback of resident tensors

- `BatchScope.finishAndRead(outputs)` (to cut Gemma / QAT output transfers) takes resident tensors of
  the same device by name and returns caller-owned `ArrayBuffer`s. `finish(): Promise<void>` and
  `ResidentTensor.read()` are unchanged.
- The member set is fixed at the call and use reservations are taken synchronously; the data is
  borrowed and nothing writes it until settlement. Aliases of one tensor are allowed. An invalid,
  destroyed or foreign-device member, or a total over the limit, releases the partial reservations and
  rejects the call without ending the batch.
- After every enqueue has encoded, pending work is submitted and the outputs are copied, 4-byte
  aligned, into one staging buffer; the error scope is checked after the copy's submit, the map is the
  fence, and the result returns only after merging host-side failures. An empty set closes with the
  ordinary queue fence. A total above `maxBufferSize` is rejected; there is no silent switch to split
  staging or several maps.
- A readback can be requested once, on a batch neither finished nor settling; `finish` /
  asyncDispose then wait for the same settlement; a request after `finish` or a second one is rejected.
  Reservations are returned before `finish` settles; the staging buffer is destroyed on every outcome.
- A copy or allocation GPU error behind a map rejection takes precedence and is checked before the map;
  enqueue failures stay as `cause`; a map pending at device loss races the loss notification (a failure,
  not a hang).
- The later stages (generation-context wiring, the greedy path of models) were accepted separately;
  Gemma's greedy output reads through `finishAndRead` (`packages/models/src/gemma/greedy-output.ts:202`).

## Addendum (2026-09-20) — batch readback of graph outputs

- An **ordinary run** sharing a span with a GPU-resident side input (the GPU gather of PLE —
  [0085](0085-ple-host-gather.md), addendum of 2026-09-19), such as decode with temperature > 0 or with
  diagnostics, paid an extra fence: `run` takes the device's error-scope span lock, which an open batch
  holds until `finish` (self-deadlock). The root cause — a batch could not hand graph outputs to the
  host — is fixed by extending the readback above to **graph outputs**.
- `Session.enqueueRead(inputs, options): { admitted, outputs }` queues what `enqueue` queues, registers
  the graph outputs' slot backing with the span, and at settlement copies them into **the same staging
  buffer** as `finishAndRead`'s residents, read with one map, under the same rules (one staging / one
  map, `maxBufferSize`, failure precedence, reservations). `admitted` means what `enqueue` returns;
  `outputs` resolves only after an error-free settlement and otherwise rejects with the same reason.
- The copy reads at settlement, so a batch holding an enqueueRead **rejects further enqueues on the
  same Session until it settles** (a later enqueue overwriting the slot would be a silent wrong value);
  the settlement finalizer lifts this on any outcome. Disposing the Session or context during the span
  stays forbidden. Logical-length advance / defer happen at
  settlement and `lastRun` is undefined, as with `enqueue`. gpuTiming devices cannot open a batch, so
  this surface sits behind the same check (ADR 0021 unchanged).
- The surface SHOULD be used for the decode shape (M = 1) only; prefill keeps the route that closes the
  gather batch first, because enqueue would pay a chunk-shaped slot backing from the first call (re-graded
  2026-10-18 to SHOULD: a cost judgement, not a correctness constraint, which a new measurement may widen;
  ruling: accepted as proposed).
- Checks: `packages/runtime/tests/gpu_batch_enqueue_read_test.ts`,
  `packages/runtime/tests/gpu_batch_generation_test.ts`, `packages/e2e/tests/e2e_gemma4_ple_gpu_test.ts`
  (the sampler route's id sequence matches the host route; the only extra fences are prefill's).

## Addendum (2026-10-26) — `ResidentTensor.clear()`: zeroing a resident on the GPU

- **The maintainer's ruling (2026-10-26)**: a new resident operation, `ResidentTensor.clear(): void`, on the
  already-exported type (no new export — the thin surface of [0008](0008-public-api.md) reviewed by
  hand; the `public_surface_test.ts` snapshot lists symbol names only and does not see it). It replaces host
  writes of zero arrays for resetting state between runs (perf-ledger K-72 — the Wan VAE causal caches at each
  tile head, [0118](0118-wan21-video-generation.md) addendum of the same date).
- **Form**: synchronous; its own command encoder → `clearBuffer(buffer, 0, byteLength)` → `queue.submit`. One
  submit per resident. A batched `GpuContext.clearResidents(list)` was not taken because the most it could save
  is small. Measured on the RTX 3080 Ti (Node + Dawn, Wan2.2 tile 16, one submit per cache resident), one `zero()`
  call took 2.2 ms of host time (median) at the production call site, the head of each tile. Called in an
  isolated loop right after a queue drain, it took 8.3 ms (median, 3.2–10.5 ms); the gap is not explained. Even
  the isolated figure is under 0.4% of a ~2.9 s tile and far below the 110 ms of the host write it replaced
  ([0118](0118-wan21-video-generation.md) addendum of the same date). Not part of a `BatchScope` (a batch has no
  encoder of its own, so it would still be a separate submit). **Unscoped** like `write` (no error-scope pair):
  the usage always has `COPY_DST` (`STORAGE_USAGE`) and `createResident` keeps the size a positive multiple of
  4, so no validation failure is reachable behind the guards, and a pop per call would add an await per resident.
  As with `write`, the submit can land while another operation's error scope is pushed (a run waiting inside its
  section lock, or an open batch), and a failure would be attributed to it — the window
  [0123](0123-submit-inflight-bound.md) names for unscoped operations. The submits are outside the Session
  scheduler's in-flight window (0123 bounds only run / enqueue dispatches): one small submit per resident.
- **Ordering** (the same claim as `write` in decision 2): the clear lands on the queue timeline in issue order,
  after every dispatch already submitted and before anything submitted later; only unsubmitted encodes could be
  overtaken, and enqueue's eager submit leaves none once it settles.
- **Guards**: a disposed resident and a lost or destroy-requested device fail loudly, as for `write`. **It also
  fails loudly (`ResidentTensorError`) while a use reservation is held** (`useReferences > 0`). The counter holds
  two kinds: the reservation an issued but unsettled run / enqueue takes on its resident inputs and `copyOutputs`
  destinations in the synchronous part of the issue (`retainUsedResidents` in
  `packages/runtime/src/runtime/executor.ts`, rationale in the `ResidentInternals.retainUse` doc in
  `packages/runtime/src/gpu/context.ts`; it came with commit 212b3ad4 and no addendum here records it), and the
  member reservations of a pending `finishAndRead` (the 2026-09-12 addendum). This is a deliberate
  asymmetry with `write`, which leaves the same window — a same-tick write landing before the issued run's
  dispatches — as a caller contract: a clear carries no host data, so the guard is an O(1) counter check. Adding
  the guard to `write` is a separate question (not decided here).
- Checks: `packages/runtime/tests/gpu_resident_batch_test.ts` (the whole range reads back as zero after
  non-zero data and the size is kept; a clear between two settled enqueues lands after the earlier copy and
  before the later dispatch; refused when disposed, after `destroy()` and after the loss is recorded, in the
  same tick as an issued run or enqueue that uses the resident, and while a `finishAndRead` holds it, then
  accepted after settlement).
