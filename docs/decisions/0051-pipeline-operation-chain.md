# 0051: Serialising the pipeline public API and asynchronous dispose

- Status: accepted (2026-08-13)
- Date: 2026-08-13
- Related: [0004](0004-execution-model.md) (flush-before-destroy) /
  [0048](0048-irodori-host-port.md). Raised by external review P1-1
  (`.claude/reviews/2026-08-13_chatgpt-reviews/TRIAGE.md`, untracked) together with the in-house
  exhaustive review M3-W-5 (dispose waiting for in-flight work).

## Context

The pipelines of the three families of the time (anima / sbv2 / irodori) build and tear down heavy
graphs one at a time, but only **inside** one `generate()`. The runtime serialises per Session
(`Session#chain`) and per run, so `Promise.all([generate(a), generate(b)])` kept heavy Sessions
resident together (anima: text 1.4 GB + DiT 3.7 GB) with no throughput gain, breaking the VRAM
design. A synchronous `dispose()` also destroyed owned GPU objects under an in-flight generation,
breaking flush-before-destroy (ADR 0004).

## Decision

1. **A serialisation chain shared inside models, `createOperationChain`**
   (`packages/models/src/concurrency/serial.ts`, not exported from the barrel); each pipeline owns
   one. `generate()` (and irodori's `generateLatent`) run on the chain, so concurrent calls
   **wait and run in order**. The chain itself never rejects: it passes on only the settlement, so
   one failure does not take later calls down with it.
2. **`dispose()` is async** (`(): Promise<void>`; `Symbol.dispose` → `Symbol.asyncDispose`).
   Disposal is queued on the same chain, which fixes the order "mark disposed → wait for earlier
   operations to settle → destroy owned GPU objects" in one place. Later calls return the same
   completion. "Disposed" is the single test `#disposal !== undefined` (no derived boolean).
3. `generate()` after dispose throws immediately, as before. The check runs **at call time**:
   checking inside the chain would also fail generations accepted before the dispose.

## Consequences

- Breaking change to a public type (allowed while unreleased): `dispose(): void` →
  `Promise<void>`; `using` becomes `await using`. Examples and e2e tests were updated.
- Concurrent `generate()` means "run serially", not "reject": callers need no retry branch; the
  only purpose is the one-at-a-time resource guarantee (order = call order).
- Every family's pipeline now uses the same chain (e.g. `birefnet/pipeline.ts`, `wan/pipeline.ts`).
- The runtime `Session` keeps `dispose(): Promise<void>` without `Symbol.asyncDispose`
  (`packages/runtime/src/runtime/executor.ts:1244`); this asymmetry remains a separate candidate.
