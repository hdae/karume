# 0061: Out-of-range indices in gather / embedding surface as quiet-NaN contamination

- Status: accepted (2026-08-15 — records the behaviour in place since 2026-08-02, until then
  spread over `docs/limitations.md` and kernel docs)
- Date: 2026-08-15
- Related: [0020](0020-nan-propagation-bitwise.md) (NaN propagation; left this case to a separate
  ruling) / [0010](0010-symbolic-constant-folding.md) / [0107](0107-model-input-error.md) / the
  doc comment of `packages/runtime/src/kernels/gather.ts`

## Context

- The contract is "indices are in range". On a violation, WebGPU's bounds-checked access quietly
  returns 0 or another valid value, leaving no trace — against the fail-loudly invariant.
- Raising an error from a kernel needs a new channel (a per-run fault flag plus readback), a
  constant cost on the dispatch path of every op.

## Decision

- **The GPU kernel writes quiet NaN only to the out-of-range element (for embedding, the row) and
  keeps running.** The kernel's output becomes NaN, which reaches later arithmetic through NaN
  propagation (ADR 0020); stages that drop NaN (comparisons, `where`, cast to int) can erase it —
  comparison semantics follow torch and do not change. There is no exception, and no other element
  is returned silently.
- **The CPU reference throws on out-of-range indices** (`packages/runtime/src/reference/ops.ts`).
  The asymmetry is intended: the reference is a diagnostic tool with no need to keep running.
- **No "fault flag plus readback" channel**; if ever needed, it is designed on its own. (The
  original premise — indices come from constants clamped at export, ADR 0010 — no longer holds for
  user input; the addendum replaces it with a host check.)

## Consequences

- An out-of-range index shows as NaN output, not an exception; when debugging, suspect the index
  tensor first.
- Dynamic indices (KV positions, routing) re-examine this contract in their own ADR (done for user
  token ids in the addendum). [limitations](../limitations.md) points here.

## Addendum (2026-09-25) — dynamic indices from user input

With the LLM work (ADR [0066](0066-generation-context-state-slots.md) /
[0083](0083-generation-api-surface.md)) a user's run-time input (prompt token ids) became an
embedding index, and the premise "a violation can only be a model error" broke. **The GPU contract
does not change** (out of range = quiet NaN on that row; no fault flag). Instead, **an index that
comes from the user MUST be range-checked against `0..vocabSize−1` at the models entry point (host
boundary), and an out-of-range index fails with `ModelInputError`** (ADR 0107) — the prompt and
`stopTokens` checks in `packages/models/src/generation/sequence.ts`. Internally generated indices
(speculation drafts in `generation/speculation.ts`, the greedy choice) get the same check with a
plain `Error` (a broken invariant, not user input). Gemma 4's host-side PLE gather (`gemma/ple.ts`)
checks its index row count (there an out-of-range id would hit another token's valid row). **An
ADR that adds a dynamic index (MoE routing and the like) MUST require the same host-boundary check
and MUST NOT use GPU NaN contamination to detect bad user input.**
