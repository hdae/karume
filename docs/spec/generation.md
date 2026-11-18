# Generation: what a decode step does on the host

What the generation loop of `@karume/models` (`packages/models/src/generation/`) does now between two decode runs of a
language model: where it checks for an abort, and where a sampled token's candidates come from. The rest of the
generation surface — the sequence, its turns, the sampler's processing order, speculation, the decode run-ahead — is
held by the ADRs that decided it ([0083](../decisions/0083-generation-api-surface.md),
[0096](../decisions/0096-speculative-decoding.md), [0142](../decisions/0142-decode-run-ahead.md)) until it has a spec
here.

Both rules below change where a decode token's time goes, and neither changes a token: the same request with the same
seed generates the same token ids with and without them.

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

## Candidate selection

A decode step that samples from the top-k of the logits reads back the **leading entries** of the row instead of the
row. The pipeline selects them on the GPU, the host merges them and the sampler draws from them; the sampler ranks,
normalizes and draws exactly as it does from the whole row.

### When a step uses it

- The pipeline has the selection: a Gemma 4 pipeline (regular, QAT, mobile QAT) that builds its GPU selection surface —
  not under the expert cache; and on a device without the runtime's timing mode and without `onRunDiagnostics`, unless
  the per-layer embeddings are on the GPU or the decode run-ahead is on — and whose vocabulary divides into 256 parts of
  more than 8 ids. The E2B, E4B and 12B vocabularies (262,144) do.
- The turn decodes one run per token and is not observed run by run: no drafter, no self-speculation, no
  `onRunDiagnostics`.
- The sampler has a temperature above 0 and a `topK`, no repetition penalty other than 1 and no logit bias, and asks
  for at most 256 entries (`topK` + the distribution's suppressed tokens + 1).

A turn that does not meet these decodes as before: the whole row is read back and sampled on the host, or, at
temperature 0, the GPU selects the maximum. The first token of a turn is always drawn on the host from the prefill's
logits.

### The selection on the GPU

- The decode run and the selection are one batch with one fence, as the temperature-0 selection is: the run's logits
  are copied to a resident, a small graph without weights reads it, and only the selection is read back.
- The graph cuts the row into 256 parts **by residue**: part `p` holds the ids `≡ p (mod 256)`. It is a `reshape` to
  `[width, 256]`, a `permute` and one `topk` of 8 over the parts — existing ops and their kernels.
- Each part returns its first 8 entries in the ranking `topk` defines: value descending, the smaller id first on a tie,
  +0 and −0 equal, NaN above every number. The readback is 16 KiB (2,048 logits and their places) in place of 1 MiB.
- The suppressed tokens of a distribution are not masked on the GPU. The host's draw masks them.

### The merge and the draw on the host

- The first `n` entries of the whole row's ranking are merged from the parts' lists (`leadingEntries`), `n` =
  `topK` + the number of suppressed tokens + 1.
- The merge **does not decide** when a part's whole list was taken before the last of the `n` entries: that part may
  hold an entry that ranks above the ones merged after it. A part whose last returned entry is the `n`-th hides
  nothing.
- The sampler builds its draw target from the `n` entries (`Sampler.topCandidates`): it leaves out the suppressed ids,
  applies the temperature as the whole-row path does, orders the candidates by processed value and id, takes the first
  `topK`, and computes the weights, their sums and the top-p cut with the function the whole-row path uses.
- The sampler **does not decide** when the entries cannot settle the candidates: a NaN at an id that is not
  suppressed, a maximum that is not finite, a `topK`-th candidate at −∞, or — at a temperature other than 1 — a
  `topK`-th candidate whose processed value equals that of the last entry (a token outside the entries could tie with
  it and rank first by its id).
- Entries that are not a ranking — an id outside the vocabulary, an order the ranking does not allow, a wrong count —
  do not decide either.
- A step that does not decide reads the whole row back from the resident, inside the same serialized operation as the
  run, and the sampler draws from it as without the selection — or throws what the whole row makes it throw. The
  random number generator is untouched until then, so the draw is the one the whole-row path makes.

### What a turn reports

The stop of a turn that samples (a temperature above 0) on a pipeline that has the selection carries
`candidateSelection`:

| Field | Meaning |
| --- | --- |
| `count` | The leading entries a step asked for; 0 = the turn did not use the selection |
| `steps` | Decode steps drawn from the selected entries |
| `fallbacks` | Decode steps that did not decide and read the whole row |

A turn at temperature 0 draws nothing and reports no `candidateSelection`: its stop is what it was without the
selection. Neither does a pipeline without the selection. The chat stop and the workbench's stop record copy it, and
the decode bench's summary line says `65 candidates: <steps> steps, <fallbacks> whole-row`.

### The other side

`WHOLE_ROW_SAMPLING` (a symbol exported by `packages/models/src/gemma/pipeline.ts`, not on the public surface) builds
a pipeline without the selection. Tests and measurements use it as the reference: the token ids of the two pipelines
are equal for the same request.
