# limitations — by-design constraints

A ledger of intentional design constraints (bugs are in known-issues.md, the history of design decisions in decisions/).
Where "when it can be lifted" is decided, the planned lifting is noted.

## Runs of several Sessions on the same device execute serially

WebGPU's errorScope is a per-device LIFO stack with no notion of "whose scope it is";
overlapping scopes cause misattributed failures and silent all-zero results (measurement record in
[research/2026-08-01-m0-review.md](research/2026-08-01-m0-review.md)). For this reason the whole
GPU work of one run (encode through readback through arena destruction) is serialized by a
per-device lock (the doc of `GpuContext.withScopeLock` gives the rationale for the trade-off). If
concurrent throughput is needed, use separate devices (= `acquireGpu()`). No lifting is planned
unless WebGPU gains something like a "scoped submit".

## Runs on the same `GenerationContext` go one at a time (issuing a second while the first is unsettled is rejected)

`Session.run` has "you may issue the next run without awaiting the returned Promise" as part of its
public contract, but **runs with the same `context` in the third argument `generation` are the
exception**: a second run issued while a first one is unsettled fails with `ExecutionError` (the run
lease of a `GenerationContext` is limited to one). The reason is a silent wrong value: the second run
builds its uniforms and dispatches from the logical length advanced by the first, while the inputs that
carry positions (for gemma4, the RoPE cos / sin derived inputs — the host builds them from the position)
are **ordinary graph inputs that the caller built at issue time**, and the runtime does not look at
their contents — the KV logical length stays correct while only the positions silently shift. For
generations you want to run concurrently, **use separate contexts** (concurrent issue between
different contexts, and of 1-shot runs without a generation surface, works as before).

Note also that the two fields of the third argument (`context` / `queryLength`), like `inputs` /
`bindings` / `copyOutputs`, are **copied during the synchronous part of the issue**, so rewriting that
object after issuing has no effect on the run in flight.

## `BatchScope.finish()` also throws the first **host-side** failure of the scope

In a scope that queues `Session.enqueue` without awaiting, a failure in the enqueue body (host-side
checks, recipe building) showed up only in the returned Promise — if nobody held it, it escaped as an
unhandled rejection, and the scope settled as "succeeded with one dispatch missing". Now
`BatchScope` records the **first one** of the scope and `finish()` throws it (when there is also an
errorScope-side failure, the errorScope one is thrown with the record in `cause`. The second and later
ones are dropped — a row of derived failures makes the root cause unreadable).

The same failure still also shows up in the enqueue's returned Promise as before (one fact visible
through two paths, the same as `run`). `ExecutionError` and the like coming out of `finish()` is a
**breaking behaviour change** relative to 0.7.0 and earlier.

## `generateGreedy` left the public surface (demoted to an internal helper for parity acceptance)

`generateGreedy` and `GreedySpec`, which the `@karume/models` barrel (`mod.ts`) exported, were
**removed** (ADR [0083](decisions/0083-generation-api-surface.md) decision 9 — a **breaking change**
to the public surface of 0.7.0 and earlier). The implementation remains in
`src/generation/greedy.ts`, and the parity check that compares the token sequences of the existing
goldens lives on as a path, but **it cannot be imported from outside the package**.

The reason is the nature of the surface: this function is the one deterministic path for "acceptance
against a fixed token id sequence" (there is a MUST that it carries **neither** sampling nor EOS
stopping). Static wiring and the request live in one type, and the lifetime of the
`GenerationContext` leaks to the caller, so building a multi-turn chat on top of this surface
reproduces on the consumer side the accident of "the last token of the previous assistant turn is
dropped" (ADR 0083 Context / decision 4). "Keep it public for now and remove it in the next breaking
wave" was not adopted.

The replacement is `GenerationSequence` from the generation API wave (ADR 0083 decisions 1–5 —
`AsyncIterable<GenerationEvent>` / `AbortSignal` / multi-turn), and host-side sampling is
`src/generation/sampler.ts` (the same ADR, decisions 7–8). The `@karume/models` barrel exports
`GenerationSequence` as a type, and the implementation is returned by `gemma.Pipeline.sequence()`.

## The pin constants `*_CURRENT` are gone from the public surface (replaced by per-family source tables, 0.9.0)

`ANIMA_CURRENT` / `ANIMA_EXTRA_CURRENT` / `IRODORI_V4_SMALL_CURRENT` /
`IRODORI_V4_1_SMALL_CURRENT` / `SBV2_JVNV_CURRENT` / `GEMMA4_CURRENT`, which `@karume/models`
exported, were **removed** (a **breaking change** to the public surface of 0.8.0 and earlier — ADR
[0092](decisions/0092-distribution-repos-and-sources.md). No compatibility re-export is kept). The
replacement is a per-family table whose key is the basename of the HF repository name with `karume-`
dropped:

| Up to 0.8.0                  | 0.9.0 – 0.14.x                          | From 0.15.0 (ADR 0136)                  |
| ---------------------------- | --------------------------------------- | --------------------------------------- |
| `ANIMA_CURRENT`              | `ANIMA_SOURCES["anima"]`                | `anima.SOURCES["anima"]`                |
| `ANIMA_EXTRA_CURRENT`        | `ANIMA_SOURCES["anima-extra"]`          | `anima.SOURCES["anima-extra"]`          |
| `IRODORI_V4_SMALL_CURRENT`   | `IRODORI_SOURCES["irodori-v4-small"]`   | `irodori.SOURCES["irodori-v4-small"]`   |
| `IRODORI_V4_1_SMALL_CURRENT` | `IRODORI_SOURCES["irodori-v4.1-small"]` | `irodori.SOURCES["irodori-v4.1-small"]` |
| `SBV2_JVNV_CURRENT`          | `SBV2_SOURCES["sbv2-jvnv"]`             | `sbv2.SOURCES["sbv2-jvnv"]`             |
| `GEMMA4_CURRENT`             | `GEMMA4_SOURCES["gemma4"]`              | `gemma.SOURCES["gemma4"]`               |

The gemma4 entry **also renamed the repository** — `karume-gemma4-e2b` → `karume-gemma4` (to make it
a container that E4B / 12B can share — ADR 0092 decision 1. Done at the 0.9.0 publication). HF's
rename redirects the old name (HTTP 307 for both the API and resolve), so pins baked into
`GEMMA4_CURRENT` up to 0.8.0 keep resolving.

The value (`{ repo, revision }`) has the same shape as before and can be passed as is as the first
argument of `fromPretrained`. The barrel (`@karume/models`) additionally exports `KARUME_SOURCES`,
which folds all families together (**same keys, same values**) — a surface for callers that walk all
public repositories; a consumer that uses only one family takes the family table of its subpath.

`*_CURRENT` was "one constant = one repository": every new repository required inventing a constant
name (version spellings like `_V4_1_` leaked into identifiers), and there was no way to get "the list
of repositories this family publishes" from code. As a table, the list comes from `Object.entries`,
and the key derives mechanically from the repository name (`"karume-" + key` gives back the repo
basename — checked by `packages/models/tests/sources_test.ts`).
There is still **no** default entry (the caller spells out which entry to use — the 2026-08-25
withdrawal in ADR 0073).

## The generation API changed in 7 points before publication (breaking changes to the first-edition text of ADR 0083 / 0084)

The generation API (`GenerationSequence` / `gemma.Pipeline`) reached JSR **for the first time in
0.9.0** (`@karume/models` of 0.8.0 does not export `./gemma`). The ADRs, examples and demo scripts
were written against the surface mid-wave, so the 7 points that changed **before publication** while
working through the public surface review (2026-08-31 / second wave 2026-09-02) are kept here.
The canonical record of the addenda is ADR
[0083](decisions/0083-generation-api-surface.md) / [0084](decisions/0084-gemma-tokenizer-chat.md).

- **`generate` after `dispose()` throws synchronously** (previously the first iteration produced a
  generic message from the `GenerationContext` side, and the real cause (you disposed it yourself)
  could not be read). Only the lifetime check is synchronous; the budget checks (position table,
  capacity) still become `GenerationCapacityError` on the iteration side as before.
- **`GenerationStop` gained a `tokens` field** (the number of tokens that turn generated; the stop
  token of `eos` counts as 1). Code that wrote equality assertions on `done` has to follow the added
  field.
- **The constructor of `GenerationCapacityError` takes 2 arguments** and carries structured fields
  (`constraint` / `pastLength` / `promptLength` / `requestedNewTokens` / `limit` / `maxNewTokens`).
- **The wiring fields left the public `GenerationProgram`** (graph input / output names, symbol
  bindings and `derivedInputs` were split off into the internal type `GenerationWiring`). What remains
  is `chunkLength` / `capacity` / `maxPosition` / `vocabSize` / `stopTokens`; the surface is frozen
  and `stopTokens` is a frozen copy (this closed the hole through which a consumer's `sort()` /
  `length = 0` rewrote the stop set of the generation loop).
- **`SamplerSpec.logitBias` went from a `Map` to an array of tuples** (`readonly [token, bias][]` —
  2026-09-02). Code that wrote `new Map([[id, bias]])` is rewritten to `[[id, bias]]`. `JSON.stringify`
  collapses a `Map` to `{}`, so a setting saved and restored as configuration had **silently become an
  empty bias**. In addition, **writing the same token twice is a `ModelInputError`** (the
  last-one-wins folding of `Map` is gone).
- **`gemma.Pipeline.sampler` was renamed to `defaultSampler`**, and the type of
  `gemma.PipelineConfig.sampler` shrank from `SamplerSpec` to `gemma.DefaultSampler` (the 3 fields
  `temperature` / `topK` / `topP` required). In addition, `fromAssets` now runs the same checks as
  `fromPretrained` (unknown keys, value ranges, `chunkLength ≤ maxChunkLength`,
  `chunkLength ≤ capacity ≤ maxPosition`) before opening the bytes, so **an invalid declaration
  fails before the 3.7GiB load finishes** (previously it first became a `RangeError` at the first
  `generate`).
- **The union of stop reasons gained `stop-token`, and the type of `gemma.ChatStream.done` widened to
  `gemma.ChatStop`** (2026-09-02 — per-request stop conditions). Code that wrote
  `switch (stop.reason)` exhaustively has to follow the added branch (the `chat` side additionally
  gains `stop-string`). In addition, `gemma.ChatStream` gained `text()`, so **one stream can be
  consumed only one way** (mixing iteration with `text()`, or iterating twice, throws synchronously —
  previously the second `for await` silently ended after 0 iterations).

## Only `chat` can stop on a stop **string** (`sequence()` stops on tokens)

`gemma.ChatOptions.stopStrings` (stop when it appears in the decoded text) is **a knob of the chat
layer**. The low-level surface (`gemma.Pipeline.sequence()` → `GenerationSequence.generate`) does not
have it; to stop there, use `GenerationRequest.stopTokens` (token ids, unioned with the
distribution's EOS set). `GenerationSequence` handles only token ids (it knows neither the tokenizer
nor the distribution), so string matching can live only in a layer that owns the decoder — the two
layers of the ADR [0083](decisions/0083-generation-api-surface.md) addendum of 2026-09-02. If you
drive `sequence()` yourself and want to stop on a string, look at the decoded text yourself and
`break` (the conversation advances "only by the successful runs", and the last token stays in the
uncommitted frontier = the same state as a `chat` stop string).

## sampler: top-k differs from HF in its candidate set at a tie on the boundary (by-design)

`SamplerSpec.topK` (`packages/models/src/generation/sampler.ts`) narrows to **exactly k entries**,
keeping the smaller id on a tie. HF's `TopKLogitsWarper` keeps every candidate equal to the k-th
value, so for logits with ties lined up on the k-th boundary the candidate set and the probabilities
differ. Keeping the smaller id is the same tie-break as greedy (first wins in `argmax`), chosen so
that `topK: 1` matches greedy regardless of temperature. The RNG differs from HF's sampling, and
token-sequence parity with it is not pursued in the first place (ADR
[0083](decisions/0083-generation-api-surface.md) decision 7), so no change is made to match HF.

## The params cache is unbounded for the lifetime of the Session (by-design)

The content-addressed cache of params buffers (`RecipeBuilder.#writeParams`) has no eviction; they
are released only by `Session.dispose()`. Driving a graph with symbolic dimensions with "different
bindings on every run" allocates new small buffers (one per node kind) on each run, which never hit
again and pile up for the whole lifetime of the Session (no exception and no warning;
`diagnostics().weights.allocCount` just grows monotonically). Each is a few tens of bytes, so there is
no recorded actual harm even for variable-length TTS or embeddings with variable sequence length,
but there is no upper bound.

- Why eviction is not made LRU: the params buffers are **referenced directly and folded in** by live
  derived plans (prepared plans), so making eviction = destruction would leave them holding destroyed
  buffers. Doing it safely needs separate bookkeeping, reference counting (not built until there is a
  real need).
- Workaround: for use cases that drive variable shapes for a long time, watch the growth of
  `diagnostics().weights.allocCount` and recreate the Session.
- Params newly uploaded by a run / batch that settled with a GPU-side failure are dropped from the
  cache (the allocation may have been invalid because of a non-fatal OOM — ADR
  [0095](decisions/0095-plan-backing-budget.md) addendum 2026-10-10), but the buffers themselves are
  not destroyed and remain until `Session.dispose()`. A retry allocates that content anew, so each
  repetition of such a failure piles up the small buffers of the failed execution. Host-side failures
  (wrong input, a failed pre-run check and so on — for both run and enqueue) do not drop them, so
  nothing piles up.
- Being unbounded is itself pinned by a check in `packages/runtime/tests/gpu_params_cache_test.ts`
  (adding eviction turns this check red).

## The name `__proto__` cannot be passed in an object literal (a constraint of JS syntax)

Karume's Records use a null prototype and preserve a `"__proto__"` key, but when **the caller** writes
`{ __proto__: tensor }`, JS grammar makes it a [[Prototype]] specification rather than a property
definition, and the key never reaches Karume because it is not created. For models that use
`"__proto__"` as an input / output name or symbol name, use a computed key `{ ["__proto__"]: tensor }`
or `Object.assign(Object.create(null), ...)`. Karume cannot detect this (it cannot know what did not
arrive).

## Non-finite values (NaN / ±Infinity) are not checked

Non-finite values in inputs and weights are not checked, for cost reasons, and flow straight into
the computation. **Matching propagation between the GPU and the CPU reference is guaranteed only for
the ops listed in the NOTE below**; it is not a universal guarantee. Relatedly, the reduction identity
of `amax` / `amin` is ±F32_MAX **on both the GPU and the CPU reference**, and `amax` of a row whose
elements are all -Infinity returns -F32_MAX, not -Infinity (torch returns -Infinity, so that is a
by-design mismatch). When handling models that contain non-finite values, check beforehand on the
caller side.

NOTE: matching propagation is guaranteed for `clamp` / `clamp_min` / `relu` / `amax` / `amin`, and
also, by bitwise NaN detection (`nan_max`), for **softmax / safe_softmax / the row statistics of
attention (both fused and states form)** (unified in v2 on 2026-08-31 — before that, WGSL's `max` drops
NaN at the spec level, so a row of all-NaN elements was mistaken for an empty row by the safe variants,
became exactly 0, and the NaN silently disappeared). Mechanism and ruling:
[decisions/0020](decisions/0020-nan-propagation-bitwise.md) and the
[decisions/0044](decisions/0044-runtime-attention-mask.md) addendum of 2026-08-31.

NOTE: the PV of states-form attention treats the columns outside the predicate (causal, window)
differently on the GPU and on the CPU reference. The GPU also adds weight 0 × V for the columns outside
the predicate within the live range, while the CPU reference does not read V for those columns (as the
formula says). With finite V both contribute 0 outside the predicate and the value does not change, but
if V outside the predicate contains NaN / ±Infinity, only the GPU row becomes NaN.

## The additive mask of fused attention carries −inf **as a value** (relies on the Finite Math Assumption)

The additive mask constants that the exporter bakes represent out-of-band positions with a literal
−Infinity (a different scheme from the finite sentinel −3.4028e38 of `masked_fill`), and the f32 path
of fused attention adds it as is and drops it to 0 through `exp(S−m)`. Under WGSL's Finite Math
Assumption (an implementation may assume that ∞ / NaN do not appear during execution — §15.7.2),
carrying ∞ as a value is itself exposed to the implementation's latitude, but in measurements
(shipped assets on Vulkan / Metal) it behaves as expected. A row whose columns are all −inf (a fully
masked row) is **a regular input that yields output 0 through the empty-row guard of
attention_stats v2** (2026-08-31 — before that, it was a contract violation of
`1/0` = indeterminate. [decisions/0044](decisions/0044-runtime-attention-mask.md)
addendum).

## The PV of `attentionCompute: "a8"` quantizes attention weights on a 1/127 grid (1:254 cut-off)

The ③PV of a8 is the fixed grid `qP = round(127·exp(S−m))` (no clamp, relative to the row maximum),
and scores with `S − m < −ln 254 ≈ −5.537` drop to **exactly 0** — attention weights below 1/254 of
the row maximum vanish (the f32 path keeps them). By design, the reciprocal sum `inv` does not
compensate for the dropped mass (the contract family of [decisions/0044](decisions/0044-runtime-attention-mask.md),
an opt-in option). The CPU reference uses the same grid, so comparison does not detect this cut-off —
the ruling on the accuracy impact rests on the measured tolerance of the E2E tests.

## The online attention (`attentionScoreStorage: "none"`) runs only for head widths 64 and 128 and key lengths that are multiples of 4, and needs 31 KiB of workgroup storage

The online kernel (ADR [0132](decisions/0132-online-int8-attention.md)) holds its q, k, Vᵀ and P̃
tiles in workgroup storage: 31,232 B in the subgroup form and 35,328 B in the portable form at
D = 128 (a head width of 256 would need 55 KiB and is not built). Shapes outside D ∈ {64, 128} or
with `N % 4 != 0`, and devices that offer neither form (no subgroups and a 32 KiB limit — today
Firefox and Safari on D3D12 / Metal; any device at the 16 KiB core default) run the three
dispatches of `attentionCompute: "a8"` with S in f32 instead — the route shows it in the
pipeline keys and the full-eligibility check of the census turns red. The mask and GQA are rejected
as for a8. The weights are quantised against a running row maximum, so the output is not the
three-pass a8 output bit for bit (class Q: within the P̃-lattice error model — the same 1/127 grid
as the ③PV above).

## Fused attention a8 / s16 store ③PV and S in f32 for shapes whose key length N is not a multiple of 4 (in Anima, resolutions whose sides are not multiples of 32 px)

`attentionCompute: "a8"` and `attentionScoreStorage: "f16"` (half-precision storage of S) run as
declared only in the stages that meet the shape conditions. The conditions are: the i8a8 of ①QK needs
`D % 4 == 0`, the i8a8 of ③PV needs `N % 4 == 0`, and f16 storage of S needs
`D % 4 == 0 && N % 4 == 0`. A stage that does not meet them falls back to the f32 path — with no
exception and no warning (the fallback that [ADR 0058](decisions/0058-numerics-opt-in-contract.md)
decision 3 permits). A stage that fell back is not on the declared path: ③PV does not use the integer
dot product, and S in VRAM has f32 width. The canonical source of the decision is one pure function
(`attentionRoute` in `packages/runtime/src/runtime/attention-route.ts`).

- **Anima**: with the default preset `f16+dit8-a8-attn8-s16` (and `f16+dit4-attn8-s16`), at
  resolutions where the self-attention N = (W/16)·(H/16) is not a multiple of 4, the self ③PV and the
  storage of S become f32. Of the 9,409 accepted shapes, 4,608 are affected (e.g. 528×528 = N 1,089),
  and **it does not happen when both sides are multiples of 32 px**. With the `f16+dit8-a8-attn8`
  preset only ③PV falls back. Cross-attention (N = 512) and ①QK (D = 128) are always as declared.
  The accepted set is not narrowed.
- **Wan2.1 / Wan2.2**: for every accepted shape (dimensions × frame count), both self and cross run as
  declared.
- **Detection**: each family's A/B check applies the full-eligibility check
  (`assertAttentionSeatsComplete` — every dispatch of fused attention is an i8a8 / `:s16` variant;
  it looks at ratios, so it does not depend on the device) to the census. Host tests
  (`packages/runtime/tests/attention_route_test.ts` /
  `packages/models/tests/wan_attention_route_test.ts`) pass every shape of the accepted set through
  `attentionRoute` and pin the result. When using the runtime directly, tell them apart by whether the
  diagnostic pipeline keys contain dense ones such as `attention_pv:v1:f32`.
- A change that handles the remainders on the declared path is in the later section of the backlog.

## conv3d w8a8 (`convCompute: "a8"`) is decided per node; whether its 1×1×1 nodes beat f32 is not measured

- The activation quantisation of ADR [0133](decisions/0133-conv3d-w8a8.md) (addendum of
  2026-10-26) is one channel-major quantiser dispatch per conv3d node: it reads `x[Cin, T·H·W]` in
  place and writes the per-pixel words and scales, so a node is two dispatches (quantiser, integer
  GEMM) and there is no transposed copy. On the RTX 3080 Ti the quantiser of a Wan2.2 next chunk at
  tile 16 takes 4.8 ms in all against 28.8 ms for the strided transpose + `quantize_rows` it
  replaced. The 1×1×1 nodes, which ran slower than on the f32 path with the copy (3.2 ms against
  2.5 ms), are estimated at about 2.0 ms now; no per-node A/B against f32 has been run.
- The eligibility is per node (int8 weight, `groups == 1`, `Cin % 4 == 0`); a node outside it runs
  the f32 implicit GEMM and shows it by its pipeline key — the census of the decoder's A/B
  (`e2e_wan_ti2v_vae_a8_test.ts`) pins that every conv3d of the Wan2.2 decoder qualifies.

## Wan2.2's 32-latent VAE tile (`vaeTile: 32`) is chosen explicitly and needs about 9.4 GiB during the decoder stage; the 10 GB class keeps 16

- ADR [0134](decisions/0134-wan22-vae-tile32-shared.md): `wan.GenerateRequest.vaeTile` defaults to 16
  and is never chosen automatically — WebGPU exposes neither the total nor the free device memory
  (ADR 0070 decision 5), so the application decides. At 32 the decoder stage holds the 16-tile pair
  as lenders plus the 32-tile pair, four times the causal caches and frame backings: on the RTX
  3080 Ti the GPU-wide peak during the stage is 9.4 GiB with the int8 decoder at 1280×704 (4.2 GiB
  at tile 16). A 10 GB device cannot hold it; a request with `vaeTile: 32` there fails in the VAE
  stage, after the DiT steps.
- The gain is the overlap factor (4×7 → 2×3 tiles at 1280×704), so it grows with the clip: the
  VAE stage with `i8-a8-attn8-online-conv8` on the RTX 3080 Ti is 0.81–0.83× of tile 16 at 17
  frames and 0.70× at 33 frames (the ADR's kill line of 0.8× was ruled on the 33-frame clip).
- The pair is accepted only when the preset's weights carry `vae_decoder_first_32` /
  `vae_decoder_next_32` (Wan2.2's presets do; Wan2.1 has none, so `vaeTile: 32` is a
  `ModelInputError` there). The output differs from tile 16's in the blended seams, so the sha256
  rows carry the tile in the case id.

## On devices without dp4a, a8 does not get faster (a warning only, no fallback)

The integer dot products of int8 activations (`linearCompute` / `attentionCompute: "a8"`) run emulated
on devices without the WGSL language feature `packed_4x8_integer_dot_product` (dp4a). The numbers are
identical to the dp4a version (ADR [0025](decisions/0025-w8a8-linear-execution.md)), but it gets no
faster than the f32 path, and on Apple's Metal it is slower (ADR
[0115](decisions/0115-geometry-profiles.md) Context). The runtime silently drops neither to f32 nor to
the sibling preset without a8; it reports once per device with `console.warn` at Session construction
(what happened, and the ways out = the preset with the same weights and no a8 (every distribution
has one — checked by the distribution gate test) or `"f32"`) — ADR
[0128](decisions/0128-default-preset-original-weights.md) decision 5. Using a distribution that has a8
in its default (anima / irodori / wan2.1 / wan2.2) with its default on a GPU without dp4a produces
this warning.

## With w8a8 (`linearCompute: "a8"`) the propagation granularity of non-finite values is the same, but the sign of Inf does not match the f32 path

On the per-token i8 activation execution path (opt-in — design in
[research/2026-08-03-dp4a-w8a8-design.md](research/2026-08-03-dp4a-w8a8-design.md)), the integer dot
product itself has no notion of non-finite values, and propagation happens **through the row scale
`xs[row]`**:

- **NaN**: a single NaN in a row makes `xs[row] = NaN` (propagated by bitwise detection — ADR 0020),
  and `out = f32(acc)·(xs·wscale) + bias` becomes NaN in every column of the row. The f32 path also
  reduces over the whole row, so **the propagation granularity does not degrade**.
- **±Inf**: an Inf in a row makes `xs[row] = +Inf` (it is the largest absolute value, so **the sign
  is not kept**). The output is `f32(acc)·(+Inf·wscale) + bias`, and the signs of `acc` and `wscale`
  decide between ±Inf / NaN — **it does not match the sign of the Inf the f32 path produces**. The
  invariant of ADR 0020 ("NaN does not silently disappear") holds, but sameness down to the sign is
  not guaranteed.

The quantized value `xq` itself is also **outside the contract** when the row scale is non-finite,
because `vec4<i32>(NaN)` is an indeterminate value (only the row scale and the final output are
compared). For models that contain non-finite values, this path too assumes a check beforehand on the
caller side (the same position as "non-finite values are not checked" above).

In addition, **WGSL's f32 division is not correctly rounded** (the spec allows 2.5 ULP; on this
machine `a / b` was also measured to differ from IEEE division by 1 ULP in 55,605 of 200,000 samples).
`q = round(x / s)` **can wobble by ±1 step** for elements where `x/s` comes close to a half-integer —
an atol=0 comparison between the GPU and the CPU reference is a contract that holds only on "data
sufficiently far from the rounding boundary", and on real models this ±1 step adds up as a numeric
difference (one of the reasons the E2E tolerance is re-measured for the w8a8 series). The scale side
is built by the **multiplication** `amax · (1/127)`, so it matches exactly.

## conv2d (groups==1) fails loudly at the dispatch limit (Hout·Wout > 8,388,480)

Implicit GEMM ([decisions/0024](decisions/0024-conv2d-implicit-gemm.md)) cannot degrade because
1 workgroup = 1 output tile, so shapes whose tile count over n = Hout·Wout (tileN=128) exceeds the
one-dimensional dispatch limit (65,535) — **Hout·Wout > 8,388,480** (2,897² or more for a square
output) — become `DispatchLimitError`. The old direct kernel (grid-stride) could run these shapes, so
this is an intentional narrowing of capability, but it stops with an exception rather than a silent
wrong value. **This threshold derives from the default GEMM geometry** (the n side =
`gemmTileN` = regN·wgX), so it moves when the default moves — the current value dates from `d0afc22`
(2026-08-10), when the default became M128N128, and the "2048² exceeds the limit by 1" that ADR 0024
wrote is the old value that assumed a side of 64.
The value above is for the default profile; the limit depends on the tileN of the geometry profile
the Session selected ([decisions/0115](decisions/0115-geometry-profiles.md)). With `apple-metal-3`
(M2 — automatic on Chrome with the flag; without it, when injected through
`acquireGpu({ geometryProfile })` (addendum decision 7)) conv2d has tileN 64, so the limit becomes
**Hout·Wout ≤ 4,194,240** (about 2047×2048).
The resolution is expected on the side of the "fixed-tile VAE" of the dynamic-resolution recon
(research record [2026-08-03-dynres-vae-tiling](research/2026-08-03-dynres-vae-tiling.md)).
The counts are pinned by `packages/runtime/tests/codegen_dispatch_test.ts`.

`conv3d` (ADR [0118](decisions/0118-wan21-video-generation.md) decision 1) has the same skeleton and
becomes `DispatchLimitError` at `Tout·Hout·Wout > 8,388,480` (the default tileN 128 × 65,535). The
maximum of Wan2.1's VAE chunk graphs (tile 32) is 4 × 256 × 256 = 262,144, which does not reach it.

GPU-side coordinate arithmetic is i32, so `conv1d` / `conv2d` / `conv3d` that break
`input length + 2·padding ≤ 2^31 − 1` on any axis become `CodegenError` at Session construction (a wrapped coordinate would
slip through the out-of-range-is-0 check and produce a wrong value — ADR 0118 addendum 2026-10-03. For
conv1d / conv2d the direct and the implicit GEMM versions go through the same check). The semantics of
the CPU reference are unchanged. Real assets never hit this (only IR with padding / dilation of about
2^30 or more). `conv_transpose1d`, for the same reason, becomes `CodegenError` for shapes where any of
the kernel, the input length or `output length + padding` exceeds 2^31 − 1 (to close the range of
valid taps in i32).

## Geometry profiles: on Chrome without the flag, tables matched by `description` do not apply automatically (external constraint)

Without the developer flag (`chrome://flags/#enable-webgpu-developer-features`), Chrome leaves the
adapter's `description` empty, so `apple-metal-3` (matched by `description: "Apple M2"`) does not
apply automatically and Apple gets the default geometry (the M2 DiT loses its ×1.75; on M5 the default
is the best). Applications can take it from `BUILTIN_GEOMETRY_PROFILES` and inject it with
`acquireGpu({ geometryProfile })` ([decisions/0115](decisions/0115-geometry-profiles.md) addendum decision 7).

## The BiRefNet family's distribution has 2 models, 1024² and 2048² (model name = resolution, default 1024²)

`karume dist --pipeline birefnet` folds the 2 models `--model 1024` / `--model 2048` into 1 repository
(ADR [0092](decisions/0092-distribution-repos-and-sources.md) decision 9 — if the resolution the
model name claims and the input size of the series disagree, `DistError`). Other resolutions are not
distributed (the export step `python -m birefnet.export --resolution <multiple of 64>` passes, so
making a series is possible — distributing it needs an addition to the accepted set
`BIREFNET_MODELS` and measurements).

2048² **started passing the execution stage on 2026-09-05** (until then, the 2 outputs of the
bilinear upsample at the end of the decoder, `[1,192,2048,2048]` 3.2GB, and of cat,
`[1,240,2048,2048]` 4.03GB, exceeded NVIDIA's binding limit of 2GiB, and the Session's plan-time
preflight (ADR 0093 decision 5) failed them with the node name). What resolved it was the combination
of recipe-side patch ⑨ (`birefnet.patch` — swapping the order of the 1×1 conv and the bilinear
upsample removes `cat`) and runtime-side static liveness packing (ADR 0093); the measurement for the
f32 preset (Lucida's default) (RTX 3080 Ti 12 GiB, 2026-10-16; the memory numbers re-measured
2026-10-26 from the runtime's estimate after the transient placement moved from first-fit to least
region growth — perf-ledger H-42) is weights 843MiB + intermediate
region 2,725MiB (live peak 2,560MiB) + input and output readback 64MiB =
**total GPU allocation 3,632MiB (about 3.5GiB — 1 GiB = 1,024 MiB), run 6.7–6.9 s**. 2048² entered the
distribution on 2026-09-05, alongside 1024² (default 1024²).
(It used to also be listed as a reason that "the decoder's 1×1 conv hits the conv2d dispatch limit
above", but since `d0afc22`, when the default geometry became M128N128, the n tiles of 2048² are
32,768, inside the limit — the only remaining reason was on the resource side.)
The distributed 1024² is not light either: the measurement for the f32 preset (Lucida's default) needs
weights 843MiB + intermediate region 689MiB (after ADR 0093's packing and patch ⑨; 6,283MiB with the
old pool + old tail) + input and output readback 16MiB = **total GPU allocation
1,547MiB (about 1.5GiB), run about 1.4 s** (memory measured 2026-10-26, run time 2026-10-16),
and the largest binding is the decoder's `cat_211` `[1,1280,256,256]` 320MiB (followed by
`[1,64,1024,1024]` 256MiB inside the ipt branch and the attention score S at 230MiB), so it is in
practice **limited to desktop-class GPUs**. **Even when the distribution's `requiredLimits` field is
empty, it does not run on a device with default specs** — the field states only the resident part
(weights, state), and the intermediates exceed 128MiB
(for the semantics of the declaration see the section "the GPU fit check before download" below).
**Row-block fusion of decomposed attention misses at all 96 sites in this family** (the shapes of the
mask `[1,6,144,144]` and of `softmax` do not match the measured shapes of the matcher). The score S
therefore becomes a single binding, requiring 230MiB at 1024² and 878MiB at 2048². Even so, the
largest binding is not that but the decoder's `cat_211` (320MiB at 1024² and 1280MiB at 2048² — the
fit check before download is built from this value) — **devices with the 128MiB default are outside
this family's scope** (at 1024², 24 intermediates besides S exceed 128MiB).
The inference resolution of the original (General-HR in the bundled `handler.py`) is 2048², so the
fact that this is **not the same setting as upstream** is stated explicitly as a constraint of the
distribution. No workaround is included (a matter for whoever measures and decides).

**Weight residency drops with the quant preset `f16`, but not by half** (ADR
[0113](decisions/0113-birefnet-weight-series.md), 2026-09-26). The resident weights of the f16 series
are **431 MiB** (f32 is 843 MiB — both nearly the same at 1024² and 2048²). Only eligible weights are
stored in f16 (linear / conv / relative-position tables — 205 tensors, 820 MiB → 410 MiB, plus 8
window-mask tables); the weights of deform_conv2d (19 MiB — they have no weight slot), bias / norm /
α β stay f32. The window masks and the zero pads are not part of the resident weights (since
2026-10-16 — the window masks are intermediates built at run time from tables for the 4 window kinds
and per-window indices, and the zero pad is an `expand` of a single zero element. Until then the
window masks (65 MiB at 1024², 248 MiB at 2048²) and the zero-pad constants (14 / 28 MiB) were kept
resident in f32 — ADR 0113 addendum 2026-10-16). At 2048² the intermediate region (2,725 MiB) is the
dominant term, so the effect on the total allocation is small. The quant presets are derived per
checkpoint from the upstream storage type (ADR 0113 addendum 2026-10-11): BiRefNet_HR has only the
`f16` preset (the upstream weights are f16 — an f32 preset would only hold the same values at twice
the residency, so it is not offered), and Lucida has the `f32` default + `f16`. The total GPU
allocation of HR's `f16` preset (weights + intermediate region + input + output readback) is
1,136 MiB (about 1.1 GiB) at 1024² and 3,220 MiB (about 3.1 GiB) at 2048² (measured 2026-10-26
from the runtime's estimate under the H-42 placement, 1 GiB = 1,024 MiB. The intermediate region and the
largest binding are the same as for the f32 preset — activations are f32 even with the f16 preset).

**2048² does not run on the Intel Arc B570 (Linux xe driver) — one dispatch exceeds the 5-second job
limit** (measured 2026-09-20, ruling 2026-09-26 = explicit SKIP under that machine's environment key).
The xe driver puts a limit of `job_timeout_ms` = 5,000 on a compute job (= one `queue.submit`) (not
changeable by non-root; at most 10,000 even then), kills jobs that exceed it and turns them into
device lost. In BiRefNet 2048², the single `deform_conv2d` at the end of the decoder (dispatch #2276,
output `[1,256,1024,1024]`) hits this — pinned down to the dispatch by bisection, and it reproduces
even with 1 dispatch = 1 submit (splitting submits by time budget (ADR 0004) does not help because
"a dispatch that exceeds the budget on its own cannot be split"). The same dispatch takes 1.66 s at
1024² (output `[1,256,512,512]`), so 2048² with 4 times the elements is expected at ≈ 6.6 s. The two
1024² series (birefnet-hr / lucida) are all green even on the B570. Deno does not turn this device
lost into an exception but panics, stopping the following tests along with the whole process
(known-issues, "Intel Arc B570" section). For this reason, `HELD_SERIES` in
`packages/e2e/tests/e2e_birefnet_test.ts` has rows for the 3 series of 2048² (birefnet-hr's f16,
lucida's f32 / f16) × the B570 environment key (`deno-intel-graphics-bmg-g21`), and on this machine
the tests that use the GPU (golden comparison, geometry discrimination) are explicitly SKIPped
(a warn at registration; the asset integrity tests still run; no effect on other machines — the held
rows of ADR [0106](decisions/0106-device-keyed-references.md) addendum 2026-09-26, part 3). The
condition for lifting it is to split `deform_conv2d` into several dispatches on the kernel side, or
to speed it up so that one dispatch fits within the limit ([perf-ledger](perf-ledger.md) K-63);
the rows are removed then.

## The BiRefNet family's α matte: it does not match the output of HR's upstream half-precision model, and upstream's truncation is not copied

Post-processing (logit → 8-bit α at the size of the input image) copies each series' upstream
**interpolation kernel and order of steps** (the distribution's `pipelineConfig.postprocess` —
HR = `u8-pil-bicubic`, Lucida = `float-bilinear`. ADR
[0125](decisions/0125-birefnet-matte-postprocess.md)). The contract goes as far as "the value obtained
by applying each series' upstream procedure to karume's logit" (the golden check —
`packages/models/tests/birefnet_matte_test.ts`), and the following 3 points are by-design.

- **It does not match the output of the upstream half-precision model**: HR's upstream (the bundled
  `handler.py`) runs the whole model in half precision (both weights and activations f16). Both of
  karume's HR presets compute in f32, so the logit differs from upstream from the start. Imitating half
  precision only in the last post-processing step would not bring it closer to upstream, so it is not
  copied (ADR 0125 decision 5).
- **Upstream's truncation is not copied**: quantization to 8 bits rounds half up (0.5 rounds up) in
  every procedure. Upstream truncates in both series, and f32 truncation turns 60–85% of the
  foreground with confidence 1 into 254 instead of 255 (measured on real images). In karume, pixels
  with logit 6.23 or more become 255 (ADR 0125 decision 4).
- Applications cannot choose the procedure (only the distribution declares it). The upstream
  procedure had no problem that needed a workaround, so no option to choose it was added (the
  "Consequences" section of ADR 0125, the ADR [0005](decisions/0005-verification.md) addendum
  2026-10-11).

## conv1d (groups==1) also fails loudly at the same dispatch limit (Lout ≈ 8.39M)

conv1d's implicit GEMM ([decisions/0053](decisions/0053-conv1d-implicit-gemm.md)) is also
1 workgroup = 1 output tile, and shapes whose tile count over n = Lout (tileN=128) exceeds 65,535 —
**Lout > 8,388,480 samples ≈ 175 seconds @48kHz** — become `DispatchLimitError`. In real use the
dacvae decoder is tiled (halo 8) and SBV2 is cut first by its operating limit (pipelineConfig), so
this is not reached, but long shapes calling the kernel directly stop with an exception (not a silent
wrong value).

## On Deno, the total GPUBuffer allocation tops out at 97% of the driver-reported budget (external constraint)

Deno's WebGPU implementation hard-codes wgpu's **memory budget thresholds**
(`MemoryBudgetThresholds { for_resource_creation: 97,
for_device_loss: 99 }` in `ext/webgpu/lib.rs` — wgpu's own default is "no thresholds"). The test is
"`heap usage + requested size ≥ heapBudget × 97%`" (`error_if_would_oom_on_resource_allocation` in
wgpu-hal vulkan), and `heapBudget` is a dynamic value the driver reports through
`VK_EXT_memory_budget`. **No change to the number of buffers, the upload path or the
flush frequency moves the ceiling** (only the total and the requested size enter the test —
consistent with the measurement where 6 ways of writing it gave the same value). A `createBuffer`
that crosses the 97% line returns OOM, and the 99% line is checked on every submit / poll and becomes
**device loss** (this also explains the measurement where a loss in the middle of Session
construction took the rest of the same process down with it).

- Measurement (RTX 3080 Ti 12,288MiB, Deno 2.9.4 / wgpu 29.0.1, as of 2026-08-03): tops out at a total
  allocation of 7,280MiB (= 0.97 × back-computed budget ≈ 7.5GiB. **Why the budget is reported as only
  61% of the total is not identified** — most likely it depends on the driver implementation). The f32
  Anima DiT (weights 7,465MiB) therefore does not fit, and the 2 E2E tests concerned are ignored (the
  f16 3.7GiB / i8 1.9GiB series are unaffected).
- **The ceiling is a point-in-time value, not a fixed one**: the `heapBudget` in the test is reported
  dynamically by the driver, so it moves with every measurement even on the same machine. The
  re-measurement on the same machine on 2026-08-08 gave **11,136–11,264MiB**
  (allocating in 256MiB steps with `tools/diag/hold-vram.ts` (removed in 0.15) → `not enough memory left`;
  two samples on the same day showed this spread). This does not contradict the 7,280MiB above; both
  simply reflect 97% of the budget reported at that time.
- Measurement (Intel Arc B570 10,172MiB = 9.93GiB, Linux xe, Deno 2.9.6, as of 2026-10-02): ceiling
  **9,600–9,631MiB** (no wobble across 3 series). The reported budget (ceiling ÷ 0.97) is 9,897–9,930MiB,
  97.3–97.6% of the total, and the ceiling is about 94.4–94.7% of the total. The report is much higher
  than the 61% precedent of the RTX 3080 Ti. Procedure and table in
  [research/2026-10-02-b570-vram-budget.md](research/2026-10-02-b570-vram-budget.md) §3.
- **Near the ceiling there is a boundary where the result is device loss, not OOM**: the 97% line is
  the OOM of `createBuffer`; the 99% line is **device lost**, checked on every submit / poll. Slightly
  higher pressure turns the symptom into "intermittent device loss", so watching only allocation
  failures misses it.
- **There are also 2 kinds of OOM at the 97% line, and JS cannot tell them apart** (investigation of
  2026-09-27 — [research/2026-09-27](research/2026-09-27-h35-oom-device-lost.md)): the OOM of
  `createBuffer` is non-fatal and the device stays alive. An OOM in the staging that
  `queue.writeBuffer` creates internally (the same for `writeTexture` and `mappedAtCreation` without
  MAP_WRITE), in submit or in bind group creation makes wgpu lose the device on the spot. Both reach
  the out-of-memory errorScope as the same `GPUOutOfMemoryError` (message `not enough memory left`).
- **On Deno, `device.lost` resolves late**: Deno does not register wgpu's device-lost callback, so
  `device.lost` does not resolve until the next call that goes through validation (submit / writeBuffer
  / createBuffer / mapAsync) (it does not resolve on `onSubmittedWorkDone`). At the moment an OOM is
  caught by the errorScope, the device may already be dead.
- **On a ReBAR GPU (Arc B570), staging lands on the VRAM heap**: the VRAM heap doubles as
  host-visible and the allocator (gpu-allocator) puts staging there, so when VRAM runs out it shows up
  as an OOM on the staging side (= the side that loses the device).
- **Chrome (Dawn) follows the same rule** (only createBuffer / createTexture / createQuerySet pass OOM
  to the errorScope; an OOM in staging / submit is device lost), but places staging on the largest heap
  (system RAM on this machine) (inferred from the source, not measured).
- **Consequence: a design of "hit the OOM, then release and retry" cannot be relied on under Deno**.
  karume measures the headroom first with a non-fatal trial allocation through `createBuffer`
  (the runtime's `fitsHeadroom`) and lets go before hitting the OOM if it does not fit (anima's DiT
  residency — ADR [0112](decisions/0112-anima-transformer-residency.md) addendum of 2026-09-27).
- The f16-1024 PNG check **briefly reached 8,391MiB on the first run** (steady state 5,723MiB),
  leaving only 2.7GiB of margin against the ceiling of 11,136MiB. The transient peak was the
  double counting of weight staging, resolved by inserting one submit at Session creation (measured
  peak 5,723MiB, margin 5.4GiB. Mechanism and measurements in
  [research/2026-08-08](research/2026-08-08-vram-oom-misreport.md)).
- **It cannot be worked around on Karume's side**: wgpu has an API that changes the thresholds, but
  Deno hard-codes them, and there is neither an environment variable nor a CLI flag. The only real
  options are a patched build of Deno or running on a machine with more VRAM. The limit is
  proportional to the "driver-reported budget", not to the total VRAM, so a 24GiB machine is expected
  to fit the f32 DiT (not demonstrated).
- The file:line sources, the known upstream reports (denoland/deno#35195 and others) and the basis of
  the back-computation are in
  [research/2026-08-03-wgpu-memory-ceiling.md](research/2026-08-03-wgpu-memory-ceiling.md).

## Decode on the Deno CLI includes a 10 ms wait per token (external constraint, deno_webgpu's poll loop)

`ext/webgpu/buffer.rs` (`mapAsync`) and `queue.rs` (`onSubmittedWorkDone`) of Deno 2.9.6 wait for
completion in the form `while !done { device_poll(wait_indefinitely); sleep(10 ms) }`, and always
sleep 10 ms once even after the GPU has finished. Decode is 1 token = 1 fence, so a constant of
10 ms/token rides on Deno's wall time (10 ms of 23.8 ms/token on the RTX 3080 Ti). Running the same
assets on Chrome gives a wall time of 10–11 ms/token.

- **It cannot be worked around on Karume's side** (`pop_error_scope` does not have this wait, but
  reading back needs `mapAsync`). The options are a patched build of Deno, or run-ahead that increases
  the tokens per fence (perf-ledger H-27).
- **tok/s on the Deno CLI is not used as a product performance metric** — adoption decisions are made
  on Chrome's wall time and GPU time. Root cause and measurements in the
  [decode speed investigation](research/2026-09-19-qat-speed-recon.md) §3.2.

## Execution paths and eligibility of compressed storage — VRAM shrinks only for eligible weight slots

Every codec in the codec ledger can be executed (bf16 was "declaration accepted, execution fails
loudly" until 2026-10-17, but with [decisions/0131](decisions/0131-bf16-weight-execution.md) it moved
onto the same path as f16). The canonical source for what can be executed is
`RUNTIME_SUPPORT.storage` ([ir-v2](ir-v2.md) "Values and types").

Group quantization (w4) was **enabled, with its execution path implemented**, by
[decisions/0069](decisions/0069-packed-w4-storage.md) (2026-08-18) — codec `int4-sym-g` (storage
dtype `i4`, packed 4-bit, symmetric quantization with groups along K). The constraints are 3 points:
`groupSize` is a power of 2 and at least 16; the quantization axis (the flattened row length with the
leading dimension as rows — for rank 2, the last dimension itself) is divisible by `groupSize`; and a
scale companion (f32, rank 2 `[rows, row length/groupSize]`) is required. **Eligibility is narrower
than for f16 / i8: "only weight slots consumed by linear / embedding / conv1d (`groups == 1`)"**
(0069 decision 5 and addenda 6 / 7 — additions for conv2d / conv_transpose1d come op by op as demand
appears), and ineligible weights are expanded on the CPU at load time (zero VRAM reduction). For the
per-channel codecs (`int8-sym` / `int2-off` / `ternary`), `groupSize` MUST equal the row length, and if
it differs the merge layer rejects it (silently reading it as per-channel would be a silent wrong
value).

**f16 was enabled on 2026-08-03** ([decisions/0018](decisions/0018-f16-weight-execution.md)),
**i8 (per-channel symmetric int8) was enabled the same day**
([decisions/0019](decisions/0019-i8-weight-execution.md)), and **bf16 was enabled on 2026-10-17**
([decisions/0131](decisions/0131-bf16-weight-execution.md) — the same eligibility as f16, expanded by
bitcast) — but in every case VRAM shrinks **only for eligible weight slots**. Ineligible ones (bias /
norm-type weight / other ops / mixed consumption / no consumer) are expanded to f32 on the CPU at load
time, the **VRAM reduction is zero**, and only the delivery size shrinks.
The breakdown is observed with `Session.diagnostics().storage` (the standing diagnostics of ADR 0006.
For i8, `residentCompressedBytes` also includes the bytes of the scale buffer).

i8 **requires a declaration** of `encoding.scale` (rank 2 `[shape[rowAxis], 1]`, f32), and the row
axis `rowAxis` is the output channel (axis 1 only for `conv_transpose1d`). A missing scale, a wrong
dtype or a wrong length fails in the merge layer before the weights are fetched. Matching `rowAxis`
against the channel axis of the consuming op applies only to initializers in GPU-resident slots, and a
mismatch fails in the residency plan of `prepareContainer` before the weights are fetched.
Slots that are ineligible and expanded to f32 are restored with the declared `rowAxis`, so the values
do not shift even without the matching.

## hub: phase 1 (streaming prefetch) requires CacheStorage and fails loudly

Phase 1 of `prefetchAssets` is a surface that does not hold the bytes in hand, so in an environment
where CacheStorage is unavailable there is structurally no room to fall back to a plain fetch, and it
fails loudly (ADR [0070](decisions/0070-shard-loading-admission.md) addendum). The `onCacheError`
diagnostic arrives only in phase 2.
Node has no Cache Storage, so pass `nodeCacheStorage(dir)` from `@karume/hub/node` as `caches`
(if it is not passed, it fails loudly in the same way — 2026-10-17).

## hub: when separate loads warm the same file at the same time, both download it (by-design)

`prefetchAssets` (phase 1) is outside the merged concurrent fetches of the fetch layer (single-flight —
the mechanism that bundles concurrent fetches of the same key into one). Merging is a contract of
"hand the bytes the earlier fetch received to the later one", and warming, which does not hold the
bytes in hand, has nothing to hand over (by-design in the fetch layer `@hdae/fetch-cache` 0.8.0).
So when separate loads (2 loads starting from different `loadManifest` calls) warm the same file at
the same time, both go out to the network. Storing into the cache overwrites with the same content,
so correctness is not broken; only bandwidth increases. Within one load the fetch plan is
deduplicated, so nothing is duplicated. A mechanism to share warming across loads will be added once
there is a real need.

## The `workspaceBytes` of estimates (`PreparedModel.estimate` / `estimateGraphMemory`) is approximate (`unaccounted` lists what is not counted)

`scenarios[].workspaceBytes` is a live-interval simulation over the node sequence **before** fusion,
and the shapes that the real build folds / splits are not counted. The exception is **row-block
attention** (the 9-node window of decomposed attention — ADR
[0060](decisions/0060-row-block-attention.md)): windows caught by the same fusion plan as the
execution phase are counted in their post-fusion shape (the temporaries and output of one row block),
and the binding-limit check also applies only to that buffer (ADR [0093](decisions/0093-transient-liveness-packing.md)
addendum 2026-10-10). The `unaccounted` field lists what is not counted, verbatim — ① intermediates
that fusions other than row-block attention fold away, and the in-node temporaries their fusion rules
declare ② the quantization intermediates of linear i8a8 ③ params buffers
④ the implementation staging of `queue.writeBuffer` ⑤ the window of retirement (slot backings
retired by budget overrun / LRU eviction of plans stay alive until the cleanup after flush).
**The in-node temporaries of attention are counted** — the score S and the row statistics of the
states form (the row statistics only for plans that `stateAttentionReduce` requires; 0 bytes for
plans where the stats/PV fusion holds), and the score S, row statistics and i8a8 quantization
intermediates of the fused attention op (ADR
[0124](decisions/0124-attention-row-block-budget.md) decision 7). The options of the fused attention
op are passed through `attentionCompute` / `attentionScoreStorage` of `EstimateOptions`; if omitted,
they are counted with the runtime default f32 (Anima's per-stage estimate does not pass the options —
an overestimate for s16 of the practical preset, and an underestimate by the quantization temporaries
for a8).
The number of row blocks of the fused attention op is decided by the binding limit, so an estimate
that does not pass `EstimateOptions.maxStorageBufferBindingSize` counts only shapes whose block count
is the same on every device (shapes that fit in 1 block at the WebGPU core default of 128 MiB, and
shapes whose `attentionRowBlockMaxBytes` is at most the core default), and fails loudly naming the
limit for everything else (the same reason as for the states form — dividing by the core default would
let the estimate alone claim a block count that no device produces).
There is one more difference not listed in `unaccounted`: **padding at region boundaries** — the
estimate's placement drifts from the real build's packer by a few hundred bytes.
Where the estimate's allocation program differs from the execution's (one dispatch per plain node,
fusion temporaries left out), the placement can also keep a different one of its two orders, because
the second order compares lifetimes counted in allocation events (ADR 0093 addendum 2026-10-26); the
drift can then exceed a few hundred bytes (not measured on the shipped models).
`EstimateOptions.maxBufferSize` affects both the limit check of a single state slot and the limit of
the intermediate packing regions, and can move `workspaceBytes` (how the placement splits regions
changes with the limit). The remaining difference is offset alignment: there is no field that receives the
device value, and it packs with the WebGPU core default of 256 (the JSDoc of `estimate.ts`).
The final check of feasibility remains the out-of-memory errorScope as before, and
`peakAccountedBytes` is, as its name says, "the peak of what was counted" = not an upper-bound
guarantee.

**The slot backing budget appears as a lower bound**: `peakAccountedBytes` is
`weights + state + max(budget, largest scenario)` (ADR [0095](decisions/0095-plan-backing-budget.md)
decision 4; the largest scenario includes the bucket shapes of `generation.chunkBuckets` — the same
ADR, addendum 2026-10-10), so for small models where the default budget of 256 MiB is larger than the
scenarios, 256 MiB is counted even when the amount actually held is a few MiB (an overestimate). It
can be subtracted using the reported `planBackingBudgetBytes`; with `planBackingBudgetBytes: 0` it
returns to the former `max(scenarios)`.

## The attention row-block budget (`attentionRowBlockMaxBytes`) is a per-temporary limit, not a total, and a budget violation fails when the run's plan is derived

`SessionOptions.attentionRowBlockMaxBytes` (ADR [0124](decisions/0124-attention-row-block-budget.md))
is a knob that splits the row blocks of attention that materializes the score S (the fused attention
op, the states form, row blocks of decomposed attention) so that they fit in **the smaller** of the
device's binding limit (`maxStorageBufferBindingSize`) and this value. The output bits are the same as
without a budget, and omitting it is the same as before (split by the binding limit only). Its
by-design properties are as follows.

- **It is a per-temporary limit, not a limit on the total**: one S block and one row-statistics block
  are each at most this value, but the total of what is alive at the same time can exceed it. The
  i8a8 quantization temporaries (outside the row-block loop) and other intermediates such as the FFN
  intermediate are not bound by this value (the limit check of the intermediate placement stays at the
  binding limit). It is not a budget for total VRAM.
- **A budget smaller than one row fails when the run's plan is derived** (fails loudly, naming the
  budget): at Session construction the shapes are still symbolic and the block count is not known, so
  construction rejects only values out of range (0, negative, non-integer, NaN, non-numbers). For Wan's
  DiT, it fails after the weights are loaded. The way to know beforehand is the estimate (pass the same
  value to `EstimateOptions.attentionRowBlockMaxBytes`).
- **The peak can drop below the no-budget case, but a smaller budget does not necessarily lower it
  further**: what the budget shrinks is the temporaries of one row block, and the peak of the
  intermediates as a whole depends on how the intermediate placement splits (in the estimate's
  computed values, for Wan2.2's DiT with the practical preset at 121 frames, 256 MiB is larger than
  512 MiB — [research 2026-10-11 (estimate before and after)](research/2026-10-11-fused-attention-estimate-before-after.md).
  The real-size backing of the same shape decreased monotonically 1 GiB → 512 MiB → 256 MiB, and the
  shape of the estimate did not appear — measured on the development machine 2026-10-11,
  [research 2026-10-11 (measurements of step 4b)](research/2026-10-11-row-block-budget-stage-4b.md).
  Only this one shape was measured).
- **The speed cost depends on the shape and the machine**: dispatches increase by the added block
  count × the dispatches per block (fused attention 3, decomposed attention 4, states form 2–3).
  There are measurements where splitting made small shapes slower, and examples where splitting was
  faster (ADR 0124 Context). For Wan2.2's DiT with the practical preset at 121 frames (the development
  machine's RTX 3080 Ti, Deno), the total GPU time in timing mode was within the wobble of the
  no-budget case at budgets of 1 GiB / 512 MiB, and +24.9% at 256 MiB (measured 2026-10-11,
  [research 2026-10-11 (measurements of step 4b)](research/2026-10-11-row-block-budget-stage-4b.md)).
- **The models pipelines have no knob for it (nor did gpu-lab, removed in 0.15)**: it is a field of
  the runtime's `SessionOptions` only, and cannot be passed from `wan.Ti2vPipeline` and the like, or
  from the Wan tab of gpu-lab. It is not put into Wan's default either (ruling 39 = a, the maintainer
  2026-10-11 — the results section of step 4b in ADR 0124). A knob will be added once a problem that
  needs solving appears (the policy of the ADR 0005 addendum of 2026-10-11). Callers that use the
  runtime directly can use `SessionOptions.attentionRowBlockMaxBytes`.

## Budgeted retention of slot backings: a shape that does not fit in the budget is held alone, and disposing a resident input is rejected while it is held

Slot backings are kept several at a time within a budget (default 256 MiB,
`SessionOptions.planBackingBudgetBytes` / the options of `gemma.Pipeline`) (ADR
[0095](decisions/0095-plan-backing-budget.md)). 3 by-design constraints:

- **A shape larger than the budget is held alone** (held by retiring all the others = the former
  capacity of 1). gemma4's chunk-768 shape is ≈ 528 MiB at capacity 16K and 192 MiB at capacity 2K,
  so by default, turns that include a prompt of 768 tokens or more still rebuild
  768 shape ↔ bucket shape ↔ decode shape (≈ 40 ms each) as before. Only the round trips of short turns
  (bucket shape ↔ decode shape) disappear. If you feed a long prompt every turn, raise the budget.
- **While a backing that has a resident input (`ResidentTensor`) baked in is held, that input's
  `dispose()` fails loudly**. With capacity 1 the window could be closed by switching to another
  shape; within the budget it lasts until the Session's `dispose()` or until retirement by budget
  overrun. `planBackingBudgetBytes: 0` returns to the former behaviour.
- **Not yet passed through for other families**: families other than gemma4 build their Session
  options from the manifest's `session`, so there is no way to change the budget and the default of
  256 MiB applies (families whose shapes are smaller than the budget are on the multiple-retention
  side; the maximum of +256 MiB appears in the estimate).

## prefill buckets (`chunkBuckets`): the uncounted window of the estimate widens, and buckets below 16 do not take the tiled path

gemma4's prefill runs each chunk with the physical row count of "the smallest bucket ≥ `queryLength`"
(ADR [0066](decisions/0066-generation-context-state-slots.md) addendum 10, default
`gemma.CHUNK_BUCKETS`). 2 by-design constraints:

- **The transient VRAM peak widens outside the estimate**: slot backings (the bundle of a run's
  intermediate buffers) are built only for hit runs, so the **first** bucket run of the last chunk
  (a miss run) allocates its temporaries in the arena while the held backings are still loaded.
  Backings are kept several at a time under a byte budget (ADR [0095](decisions/0095-plan-backing-budget.md),
  default 256 MiB), so the counted side is `max(budget, largest scenario)`, and the windows on the
  uncounted side are "from retirement (budget overrun / LRU eviction) to destroy" and "the arena
  temporaries of a miss run (at most one bucket shape)". The estimate's scenarios themselves remain the
  2 of prefill / decode. From the second turn on, both the bucket shapes and the decode shape are held,
  so no rebuild happens (within the budget).
  When `generation.chunkBuckets` is passed to the estimate, the bucket shapes **enter the maximum of
  the peak without being listed as scenarios** (with row-block equal splitting, a shape with fewer rows
  at a block-count boundary uses more temporaries, so the prefill shape is not necessarily the
  maximum — ADR [0095](decisions/0095-plan-backing-budget.md) addendum 2026-10-10). gemma4's
  `estimateSessionMemory` passes the buckets.
- **Buckets below 16 do not take K-13's tiled path**: ①ₜ / ③ₜ of states-form attention are chosen
  only for plans with M ≥ 16, and below that they fall to the reference path ① / ③ (with
  `stateAttentionReduce: "parallel"`, ③′, and for M ≤ 8 also ①′ — K-14). The values are correct but
  slow, and for `parallel` × M ∈ [2, 16) the only measured shapes are the speculative verify shapes
  (M = 4 / 8). Of gemma4's default buckets (`gemma.CHUNK_BUCKETS` = 4 / 8 / 32 / 64 / 128 / 256),
  4 and 8 fall into this range.

## The linear GEMV family (1 ≤ M ≤ 64): the row-block height is chosen with constants of the reference device, and the first turn pays the shader parsing cost

linear with i4 / i8 storage × f32 compute is taken by the GEMV family for 1 ≤ M ≤ 64 (ADR
[0082](decisions/0082-linear-gemv-decode.md) addendum 5), which stays a full u32 match with the default
GEMM skeleton while being ×6 faster at M=8 and ×3 faster at M=32. 2 by-design constraints:

- **The row-block height `rows` is a pure function of (storage, m, n), and the parallelism target of
  16384 threads is the saturation point of the reference device (RTX 3080 Ti) baked in**. On GPUs whose
  saturation point is an order of magnitude smaller (integrated GPUs, the Apple M series) it buys too
  many y tiles and re-reads the weights more than is optimal, but the values are correct, and the
  choice does not change per device — choosing by a pure function is the premise of "same key →
  byte-identical WGSL" and of the ban on run-time autotuning (ADR 0022). For other devices, replace it
  with `SessionOptions.linearGemvRowsThreadTarget` (static, fixed at Session creation).
  There is no automatic selection per device.
- **The first turn pays the shader parsing cost**: a row-block variant is WGSL unrolled over its rows,
  and the parse and validation cost is superlinear in the amount of text. Even at the ceiling
  (256 elements/word), i4 with 8 rows / i8 with 16 rows cost ≈ 55–60 ms each, and the 5 pipelines that
  the M=32 bucket prefill of gemma4 E2B creates on the first turn total ≈ 150 ms (measured on
  Deno / naga). Pipelines are created only once per device, so this does not apply from the second turn
  on. The parsing cost in the browser (Tint) is not measured.

## Scaffolding for speculative decoding (MTP stage 1, ADR 0096): deferred runs are limited to the sliding slack, and gemma4's distribution has to be rebuilt

- A run issued with `GenerationRun.commit: "deferred"` is limited to
  **`queryLength ≤ context.slidingSlack`** (gemma4's distribution has a slack of 8, so up to 8 rows —
  tightened by one column from `+ 1` at stage 3: a borrower (readonly reader) reads up to column
  `P−W`, one column older than the lender, so the condition under which rejected rows never enter the
  window in every case, including `commit(0)`, is `Q ≤ slack`. It does not branch on whether a
  borrower exists). An issue beyond it fails loudly in the synchronous part — it is the condition under
  which rejected rows would clobber the live window of the sliding ring, and letting it through would
  corrupt past KV with neither an exception nor a NaN.
  Immediate runs (prefill / decode) commit all rows, so their limit stays `chunkLength`.
  While a pending commit (`pendingCommit`) remains, the next run and `rewind` are rejected (fold it with
  `commit(rows)`).
- gemma4's distribution (the product series) changed to **a physical row count of `window + 8` for the
  sliding slots, 2 outputs (output 0 = logits `[1,R,V]`, output 1 = the hidden state after the final
  norm `[1,R,H]`), and `last_row` of `[R]`**. An old distribution with 1 output makes `gemma.Pipeline`
  fail loudly (no compatibility branch is written — unreleased).
  The local mirror (`models/karume-gemma4`) is rebuilt with `dist.py --pipeline gemma4`. The decode /
  token acceptance series are unchanged (the runtime still reads assets with sliding capacity = window
  as before — the modulus is the capacity, so the values do not change).
- Each decode step adds a readback of the hidden state `[1,1,1536]` f32 (6 KB) (the drafter reads it at
  stage 3 — the generation loop of stage 1 does not read it). The readback of an R-row verify is logits
  R MiB + hidden R × 6 KB.
- The PreparedPlan LRU holds 12. gemma4's 6 default buckets (4 / 8 / 32 / 64 / 128 / 256) + the prefill shape
  - the decode shape = **8 shapes per capacity**. Alternating sequences with different capacities
    overflows at 16 shapes, and decode silently falls back to re-derivation (no exception — the
    observation point is `SessionDiagnostics.lastRunPrepared.hit`).
- The attention of the verify shapes (M = 4 / 8) is ①′ + ③′ (①′ is for plans with M ≤ 8 = decode and
  verify; ①ₜ / ③ₜ stay at M ≥ 16). T(4) = 27.6 / T(8) = 37.2 ms (P ≈ 14.7K, GPU wall time) are the
  values in this band, and tightening them is stage 4.
- The torch-side trace limit of R is 9 (draft 8 + bonus 1). The IR does not carry symbol ranges, so it
  does not constrain the runtime's bindings — the runtime-side limit is the `slidingSlack` above.

## The entry to speculative decoding (MTP stage 2, ADR 0096): the drafter is a borrower and is folded before the lender, read-only sliding is 512 columns, k is 1..3

- A run of a **borrowing context** (`createGenerationContext({ chunkLength: 1, borrow: lender })`)
  **takes the lender's run lease** — if the lender has a run in progress, an uncommitted deferred run
  or poison, it fails loudly in the synchronous part. A borrower does not advance the logical length
  (an external slot is only read and cannot carry an append), and cannot issue deferred runs either.
  Several borrowers can attach to the same context.
- **The lifetime order borrower → lender is a MUST**: `dispose()` of a lender context while a borrowing
  context is alive, and `dispose()` of a lender Session while a borrowing Session is alive, are a
  **Promise.reject** (an intentional departure from the idempotence contract of dispose — the
  borrower's bind groups hold the lender's buffers).
  `gemma.Pipeline.dispose()` folds the drafter Session first and then the target Session. Weight lending
  on the lender side is counted by the borrow count of `Session.exportWeight`.
- **The sliding columns that readonly attention (the drafter's cross-attention) reads are the 512
  columns `[P−W, P)`**. HF's drafter sees 513 columns because its bidirectional SWA is inclusive, but
  the 513th column (position P−513) is a row that can be clobbered in the ring with a slack of 8 when a
  rejection with kmax = 8 happens, so it is not taken. The impact on the acceptance rate is not measured
  (comparing drafts on identical inputs matched 100% over 3 cases × 200 cycles × 3 steps).
- **`speculative.k` is 1..3** (the distribution's drafter is baked with a 3-step unroll — increasing k
  means re-exporting; a dynamic k is stage 4). `fromAssets` does not accept `speculative` (`gemma.Assets`
  has no field for the drafter).
- **`estimateSessionMemory`** adds the verify shape (M = R = k+1) to the target's prefill / decode, and
  sums in the drafter's resident weights (excluding the borrowed embedding table) and the 8 bytes of
  lengths of the borrowing context (stage 3).
- **The drafter's storage is i8 only** (linear too is i8 per-channel). Assets that drop linear to
  i4 g32 are rejected by `dist.py` (the number of accepted tokens E[a] drops −15 to −33% against
  the unrounded ratio — ADR 0096 addendum 2026-09-08).
  However, this check (`tools/export-recipes/gemma4/distribution.py`) is **contamination detection**
  that looks at the storage in the container's binding table as "i8 is present / f16 and i4 are
  absent", not a guarantee that "every linear is i8"
  (norm needs f32, so banning f32 entirely cannot close it, and a single i8 tensor satisfies the
  presence check — assets that baked only some linears in f32 pass).
- **Admission matches only the structure** (`packages/models/src/gemma/speculative.ts`) — it compares
  the drafter graph's inputs, outputs, external slots and shared initializers by shape, name and storage
  dtype, but does not check whether the target and the drafter are of **the same asset generation**
  (baked from the same checkpoint). A drafter of a different checkpoint with the same shapes passes
  admission; only the acceptance rate drops and no wrong token sequence comes out (what decides is the
  target's logits and `sampler.next`). Pairing the generations is the responsibility of whoever
  assembles the distribution (mirror).
- **hub's `ResolveOptions.weights: []` resolves only the assets** (a form that fetches no weights at
  all). It has no meaningful use, but reading empty as "all" would let a regression in the narrowing
  of `speculative` pass silently, so empty stays empty.

## Speculative decoding (MTP stage 3, ADR 0096): speculation runs regardless of temperature, M=1 / M=4 identity is guaranteed per device by checks, one drafter Session per pipeline

- **Speculation runs regardless of temperature, and the token sequence is identical to non-speculative
  decoding** (the acceptance draw calls `sampler.next` once per row with the same logits, history and
  order as non-speculative decoding, so even the consumed RNG sequence matches). The acceptance rate
  varies with temperature — the value with the distribution's recommended sampler (temperature 1.0) is
  not measured (stage 4).
- **The exact match "with speculation = without speculation" is pinned by e2e checks, including the
  default preset** (on 2026-09-26 the claim that "it can rarely split with the default preset" was
  withdrawn). Even with gemma4's default `"parallel"`, the reduction order of attention is the same for
  M=1 and M=4 (①′ applies for M ≤ 8 and ③′ for M < 16, so both decode and verify are ①′ + ③′).
  Identity including the other paths whose shape changes with M (the row blocks of the GEMV family and
  so on) is checked in 2 tiers by `e2e_gemma4_speculative_test.ts` — speculation ⑤ = u32 equality of
  verify row 0 and the decode logits (the combinations of 3 attention variants ×
  `linearGemvReduce: "parallel"`, attention `"sequential"` / `"parallel"` × GEMV `"sequential"`, and
  RMS→add fusion (with the linear→SRQ setting)), and speculation ① / ② = equality of the generated
  token sequences (① = the `"sequential"` setting, ② = 3 cases × 200 tokens with the default preset
  `i4-fast` of the regular E2B distribution, gated = always. The attention at the time of the B570
  measurement below was `"parallel"` (the old family default), and on the same day the declaration of
  `i4-fast` moved to `"parallel-fused"` — re-measurement with the new declaration is not done). The
  Intel Arc B570 measurement of 2026-09-26 matched for every combination, with 0/200 × 3 cases
  differing for the default preset
  ([research 2026-09-26](research/2026-09-26-gemma4-speculative-determinism.md)). The u32 check covers
  all 12 combinations of 3 attention variants × 2 GEMV kinds × RMS→add fusion (+ the linear→SRQ
  setting) (widened to all combinations on 2026-09-26).
- **The guarantee is per device**. The checks confirm only the device of the machine that ran the
  gemma4 lane, and on a machine where the check is red, differences between M=1 and M=4 really exist
  (the execution order of reductions changes because the grid-stride split depends on the device's
  limits, so no claim is made across machines).
  If exact reproducibility is needed on such a machine, choose `speculative: "always"` (no gate) or
  `stateAttentionReduce: "sequential"` (the runtime's reference path, where M=1 and M=4 use the same
  reduction kernel). There is no canary that confirms identity at run time on the user's device.
- **The self-tuning gate is on by default, and what switches it is wall-clock time** (stage 4-B ④, the
  default of `speculative: true`). The gate measures the wall time of a cycle and of one decode step at
  run time, and while speculation is losing it falls back to the M=1 decode shape (it decides after
  seeing 2 consecutive block aggregates of 16 cycles, and returns on a burst aggregate of 8 cycles
  (a burst that loses badly is cut off from the 4th cycle on) —
  a single sample loses to the variance of the accepted count. The gain depends on the task and the
  host, and the break-even threshold also moves, RTX 1.72–1.88 / M2 2.17–2.6, so it cannot be baked as a
  fixed value — research 2026-09-09). The wall clock wobbles per run, but the 2 paths that can be chosen
  (M=1 for a step that fell back, M=4 for a speculative cycle) are bit-identical by the checks of the
  item above, so **on a device where the checks are green, the wall-clock choice does not change the
  output** (general rule 3 of the ADR 0058 addendum of 2026-08-29, ADR 0096 addendum 2026-09-26).
  Speculation ② of the e2e also checks, with the default preset, that the gated sequence matches
  always. The handling of devices where the check is red is as in the item above.
  The number of steps that fell back is `GenerationStop.speculation.plainSteps` and the number of
  switches is `switches` (with `"always"`, neither field exists).
- **`stateAttentionReduce` does not affect the drafter's readonly attention (①′ ③′) — it is fixed to
  parallel** (the option affects only the target's states-form attention). The exact match above is a
  check of the target's identity, so this does not break it: the drafter's reduction order affects only
  the content of the draft (= the acceptance rate = speed), and the committed token sequence is decided
  by the target's per-row logits and `sampler.next`.
- **The commit of a verify goes "up to the frontier of the delivered tokens"**. If the consumer
  `break`s, tokens that were accepted but not delivered do not enter the conversation (only one
  frontier remains — the same shape as non-speculative decoding). If an exception occurs in the
  synchronous part between the verify returning and delivery, it is `commit(0)` (the frontier stays
  unfed; the next run overwrites the rejected rows), and the sequence remains usable afterwards.
- **`k + 1 ≤ slidingSlack`** (gemma4: 4 ≤ 8) is checked at sequence creation. The `last_row` of a verify
  is padded to R = k+1 even when k′ shrinks (so as not to add PreparedPlan shapes).
- **There is one drafter Session per pipeline**, and running 2 conversations of the same pipeline at
  the same time serializes their draft runs (head-of-line waiting). A borrowing context is opened and
  closed per sequence (per turn for `chat()`), so `stateBacking.rebindCount` of the drafter Session
  grows in proportion to the number of turns.
- **The public type `gemma.RunPhase` went from 2 → 4 branches** (`draft` / `verify` — an unreleased
  breaking change). `onRunDiagnostics` still delivers 1 message per run (draft carries the drafter
  Session's diagnostics; the others the target's).
- **Delivery granularity**: one verify delivers up to k+1 `token` events in a row (position is
  `pastLength + 1 + i`). `GenerationStop.tokens` is the number delivered (including the stop token), and
  the number of runs is `speculation.cycles + draftRuns`.

## gemma4 `fromAssets`: `gemma.Assets` has the 3 fields `config` / `model` (part sequence) / `tokenizer` — no PLE reader is passed (breaking change in 0.13.0)

The PLE moved, index (`ple_index`, schema 3) and matrix blocks alike, into the assets of the `model`
container (ADR [0109](decisions/0109-manifest-v5-container.md) decision 4), so there is no longer a
place for the caller of `fromAssets` to pass a PLE reader separately (`pleIndex` / `readPleShard` of
the published `gemma.Assets` were removed — the Breaking entries of 0.13.0 in the
[CHANGELOG](../CHANGELOG.md)). `model` is passed as a part sequence in index order from part 0 — parts
of length 0 are not skipped either (the index is the id of the part itself, and skipping one makes the
following ones read as different parts — `gemma.Assets` in
`packages/models/src/gemma/pipeline.ts`). Decode's PLE fetches rows by range reads of the container's
assets (`AssetReader.read`) (`packages/models/src/gemma/ple.ts`).

## Passing a split-form container to `fromAssets` (the whole-bytes surface) keeps every part resident in host RAM at once

A distribution readable with `fromPretrained` is also readable with `fromAssets` (a part sequence whose
fetch keys are `<role>[i]` is not concatenated but opened as a container as a part sequence —
`assetComponentOpener` in `packages/models/src/hub/components.ts`). However, the entry of the
whole-bytes surface is **a Record of already-fetched bytes**, so by the time the caller has assembled
the Record, **every part is in host RAM at once**. The property that holds the host RAM peak to a limit
per source type (the section below) holds only for the surface that goes through the fetch layer
(`fromPretrained`); what the whole-bytes surface shrinks is **the size of a single ArrayBuffer**
(Chromium's wall — below), not the total resident amount. Its positioning as a surface meant for local
debugging and `examples/` (removed in 0.15) is unchanged.

A sequence whose indices do not start at `[0]` or are not contiguous, or a mix of plain keys
(`transformer`) and split keys (`transformer[0]`), fails loudly **with a diagnostic that names the
component and the part index** (it does not silently take one of them).

## The host RAM peak at load time is decided by the source type (seek type: one item + 2 blocks read ahead; scan type: up to 3 parts near a part boundary + waiting for GC)

Session construction uploads one block at a time to the GPU and lets it go, reading at most 2 blocks
ahead of it across part boundaries (`queue.writeBuffer` copies the bytes at the moment it is called, so
the CPU-side release does not wait for a fence — ADR 0108 addendum 14). For **seek-type** sources that
can read by position (the local `denoDirectory` / `nodeDirectory`, the browser's lazy Blob — Chrome not
measured), the weights alive on the JS side are one item (a block of 32 MiB or less + its accompanying
scale + the f32 expansion result of an expanded slot and a copy of the carried-over scale) + the 2
blocks read ahead, independent of the part length. For **scan-type** sources
that cannot read by position (via HF on Deno / Node), hub reads the whole part into a holding buffer and cuts
blocks out as views of that buffer, so the peak grows roughly with the part length. While the read-ahead
crosses a part boundary, the parts its blocks fall in are alive at once (up to 3 when parts hold one block
each; a part of at most 2 blocks can briefly hold the next part twice — ADR 0108 addendum 14,
"Residual"), and the number of buffers waiting for GC is not decided by the declaration (in
measurements, the maximum of external was about 1.5–4.4 times the largest part).
Which of the two an externally implemented `DistributionSource` becomes is decided by whether it has
`openFile` and by the `cost` (`seek` / `scan`) of the reader it returns
(`packages/hub/src/container.ts`). The canonical counting is [container-v1](container-v1.md) §11;
the measurements are ADR [0108](decisions/0108-container-format.md) addendum 5 and the
[research record](research/2026-09-24-part-length-ram-peak.md).

## MoE assumes every expert is resident in VRAM (no per-expert dynamic loading / eviction)

By-design (ruling of 2026-08-31). The VRAM budget of an MoE model is built from **the total
parameters, not the active ones** — dynamic residency that "loads only the experts in use" is not
offered. The basis is a structural conflict across 3 layers (measurement record =
[research/2026-08-31-freetoken-moe-over-arraybuffer.md](research/2026-08-31-freetoken-moe-over-arraybuffer.md)):
① the container's binding table requires a supply for every declared initializer (for `krm`, the
comparison of the 2 documents `validateAgainstGraph`, and for the in-memory container, the
correspondence check of `openMemoryContainer`, reject both shortfall and surplus by failing loudly —
opening a hole would break the single accepted set) ② weight residency is assembled once at Session
construction and has no place for eviction / reload ③ the IR has no value-dependent execution choice
(dynamic transfer / skipping of experts) (an intentional hold in op-vocabulary; `topk` is static-k too).
The fixed-k selection itself can be computed with existing ops (expand → gather → permute / reshape →
bmm), multiplying and accumulating only k of them while keeping shapes static, but there is no kernel
that reads compressed-resident experts directly, and going through gather needs f32 expansion and
copies.
Furthermore, as an external factor, the "device-initiated host memory transfer" that prior expert-cache
methods (FreeToken and others) assume does not exist in WebGPU, so their sync-free mechanism cannot be
ported directly.
Host-controlled miss handling (the host reads the router's result and writes the missing experts) was
shown to work as a mechanism in a PoC outside the repository, but its performance and practicality are
not settled ([backlog](backlog.md) later, "MoE page-fault").
The revisit entry with its revival conditions is in [backlog](backlog.md) parked.

## Auxiliary-asset safetensors require coverage of the data section and element alignment (may be rejected even when the official `safe_open` reads them)

The strict reader `parseSafetensors`, which reads auxiliary assets that are not containers (sbv2's
style table, anima's rope base table, vowel-detector's tables and so on), requires that the data
section be "covered with no gaps, and the absolute offset of each tensor aligned to the element size of
its dtype", and rejects violations with `SafetensorsError` (the canonical ruling is ADR
[0063](decisions/0063-safetensors-physical-layout.md)). A file that places a 4-byte type right after
an f16 with an odd element count breaks this alignment. HF's `safe_open` can read alignment
violations, so passing through it alone does not detect them
(the exporter-side check of the same rule is `karume.verify.assert_reader_layout`). The weight
container (`krm`) does not have this constraint — block starts are 64 B aligned, and the order is
decided by the writer's placement ([container-v1](container-v1.md) §4.1).

## Quantized storage (`i4` / `i2`) is distributed only in Karume's container (general-purpose safetensors tools cannot read it)

packed int4 / int2 (ADR [0069](decisions/0069-packed-w4-storage.md) /
[0097](decisions/0097-gemma4-qat-integration.md)) are baked as codecs (`int4-sym-g` / `int2-off`) of
the `krm` container. The container is a Karume-specific format, so only Karume's runtime and the
exporter (`karume.container`) can read it (uploading to and downloading from HF are content-agnostic,
so they work). Auxiliary-asset safetensors accept only a subset of the official dtype vocabulary, and
reject the dialect dtypes `I4` / `I2` that the old distribution used
(`packages/runtime/src/format/safetensors.ts`).

INT2 execution uses a per-row scale, supports packed execution of linear / embedding, and limits
`linearCompute` to f32
([ADR 0097](decisions/0097-gemma4-qat-integration.md) addendum 1).

The upper limit on the logical element count (N·K, V·H) of linear weights `[N, K]` and embedding
weights `[V, H]` differs by path, according to the unit in which the GPU forms the start of a weight
row. A linear's path is decided by the execution shape (M) and the quantization settings, so whether
the same weights are rejected depends on the path.

- Paths that form the row start **by logical element index** reject shapes whose logical element count
  exceeds 2^32 − 1 with `CodegenError` at Session construction. They are the GEMM path of linear (any
  storage form), the w4a8 of i4 weights among the i8a8 paths, and embedding. The packed loader turns
  the logical element index into a word position, so if the product wraps it reads another row's
  weights without an exception. For i2 the storage reaches it at about 1 GiB, so the binding-limit
  check does not stop it.
- Paths that form the row start **by word index** (the GEMV family of linear (1 ≤ M ≤ 64) and the
  i8a8 of i8 weights) do not get this check. Wrapping would need 16 GiB or more of storage.

- No proposal to follow the official safetensors spec (a request upstream to add I4/U4) will be made —
  user ruling of 2026-09-01. Because the directions differ, the distribution moved to a dedicated
  container (ADR [0108](decisions/0108-container-format.md)).

## Out-of-range indices in gather / embedding contaminate with NaN on the GPU (no exception)

The canonical ruling is ADR [0061](decisions/0061-index-oob-semantics.md). The contract is "indices
are in range".
On a violation, the GPU kernel writes quiet NaN only to the affected elements (for embedding, the
affected rows) and continues, making the kernel's output NaN. Downstream arithmetic carries it through
NaN propagation (ADR 0020), but downstream steps that drop NaN, such as comparisons, where and casts to
int, can make it disappear. The CPU reference throws when out of range (an intentional asymmetry).

## f32 → i32 casts out of range or of NaN are undefined

The canonical ruling is ADR [0062](decisions/0062-f32-i32-cast-contract.md). Within range, truncation
following torch (rounding toward 0) is guaranteed to match; out of range and NaN are undefined.
A per-element range check is intentionally not added — clamp values that can go out of range
beforehand on the caller side or in the model.

## The IR's i32 arithmetic wraps as two's complement (narrowing of int64 intermediates is unguarded)

The exporter normalizes torch's int64 to i32 with a range check at the boundaries (graph inputs /
outputs, concrete boundary tensors) (ADR 0009), but **the intermediate values of emitted i32 operations
(mul / sub) are not guarded** — intermediates that cross 2³¹ do not raise but wrap as two's complement
(measured: `(x*x).float()` with int64 x=50000 is exported with zero diagnostics, and the reference
implementation's `Math.imul` returns −1794967296). The 99 i32 arithmetic ops in the 10 distributed
families are all structurally bounded values derived from masks, with zero affected (189 containers
scanned, 2026-08-16).
Statically proving intermediate value ranges is impossible in general, so the contract is boundary
checks + golden comparison (the de facto check on real inputs), and models that need i64-class
intermediate operations are handled by design at export time (a decomposition that keeps the range).

## exporter: removing the identity `x + 0` can diverge from torch for −0 inputs (downstream of div / sqrt)

`normalize._drop_identity_add` folds `add(x, 0)` (the addend is the Python scalar 0) into x. In f32 the
value changes only when x = −0.0, where `x + 0.0` returns +0.0. IR v2 **does have ops that distinguish
signed zeros** — `div` gives `1/(+0) = +∞` / `1/(−0) = −∞`, and `sqrt` gives `sqrt(±0) = ±0`
(the reference implementation too uses plain division and `Math.sqrt`) — so in shapes where the
downstream of the removed add reaches the denominator of div or the argument of sqrt, the sign can be
flipped relative to torch (a minimal counterexample was demonstrated in the review of 2026-08-16:
`num / (x + 0.0)` is torch +∞ / IR −∞). In the 10 measured families there is no path that reaches it,
and restricting the rewrite by looking at the consumers would trigger a re-export of every series, so
the possibility of divergence is **accepted** and the rewrite stays. When adding a shape where −0.0 can
reach a consumer, revisit the conditions under which this pass applies first (the same note is in the
docstring of `_drop_identity_add` in `karume/normalize.py`).

## Whether a semantic dtype can execute is decided per op (no blanket enabling)

The contract table (`packages/runtime/src/ops.ts` / `karume/ops.py`) is canonical. i32 / bool can
execute only in the shapes that appeared in measured graphs and in the ops newly created for them
(e.g. mul / sub are f32+i32, div is f32 only, bitwise_not is bool only). It is the dtype version of the
frozen vocabulary allowlist (ADR 0007), and an extension requires one set of measurements + the
contract table + golden.

## bool is represented as u32 0 / 1, and bool initializers are not in the vocabulary

WebGPU storage buffers have no 1-bit type, so bool is u32 0 / 1 both in GPU storage and in inputs /
outputs (`Tensor.data` = Uint32Array) (ADR 0009). safetensors' `BOOL` (1-byte storage) does not mesh
with transfers that assume 4 bytes, so bool initializers are not in the vocabulary of IR v2 (if they
become necessary, the storage convention is revised along with them).

## The strided copy family (permute / expand / slice / cat / sym_prefix_slice / the mask of masked_fill) is rank ≤ 4

`STRIDED_RANK = 4` is checked in the contract layer (all values of the DeBERTa front are rank ≤ 4 —
ADR 0011. slice / cat share this limit as members of the read family / write family — ADR 0014).
Lowering rank ≥ 5 is **the exporter's job**, and the chain of 3 passes `_lower_unit_expand` /
`_lower_split_unbind` / `_lower_reshape_permute` is **implemented**
(`karume/normalize.py`; it fires only for shapes that contain values of rank > `STRIDED_RANK` — ADR 0016).
Accepted patterns are lowered into sequences of rank ≤ 4, and high-rank shapes that these cannot
normalize fail loudly at export time (the execution limit itself is not relaxed).

## The second layout group accepts only measured shapes (slice / cat / pad / flip — ADR 0014)

To keep "an axis with no representation is not silently executed with a default", only shapes that
appeared in measurements are in the vocabulary.
Widening moves the 4-piece contract set (TS contract table / WGSL kernel / CPU reference / Python
contract + shape rule) together with the conformance table and golden as one set.

- `slice`: **step is fixed to 1** (strided reads cannot be expressed with the single variable word of
  the strided family). The sliced axis is **static** (symbolic axes are handled by
  `sym_prefix_slice` — no overlap). Negative indices and an omitted `end` are clamped to the axis
  length at the exporter boundary.
- `cat`: the concatenation axis is (a constant) or (**a linear expression in a single symbol**).
  Shapes whose sum fits the dimension language `coeff·sym+offset` (`S`+1 → `S+1`, `S`+`S` → `2S`) are
  accepted, and a mix of different symbols fails loudly
  (ADR [0046](decisions/0046-cat-symbolic-axis.md)). There are 2 or more inputs (a cat of one is an
  identity copy and is not in the vocabulary).
- `pad`: **last dimension, constant 0, non-negative widths** only. It has no field for the fill value,
  so there is structurally no path that "silently executes something other than 0 as 0". A negative
  width (truncation) means slice and is not accepted.
- `flip`: **one static axis** only (multi-axis flip is rejected at the exporter boundary. Flipping a
  dynamic axis can be written in the kernel, but it is not widened until a measured need appears).

## The conv family also accepts only measured shapes (ADR 0015)

- The attrs of `conv1d` — `stride` / `padding` / `dilation` / `groups` — **must all 4 be declared**,
  and **no default values are filled in**. Allowing omission would let the IR of a depthwise
  (`groups = C`) silently become an ordinary convolution, mixing in values of other channels while
  neither shape nor element count changes. For those writing IR by hand this is redundant, but it is an
  intentional redundancy to eliminate silent wrong values.
- `conv_transpose1d` accepts only shapes with **`2·padding == K − stride`** (= the output length is
  exactly `L·stride`). The general form could be written as a representation, but some combinations
  give output lengths that do not fit a linear expression of the dimension language
  (`coeff·sym+offset`), so it is not widened until there is demand. The weight is **`[Cin, Cout, K]`**
  (transposed from conv1d's `[Cout, Cin/groups, K]`), and `output_padding` / `groups` / `dilation`
  have no field in attrs; anything other than the default fails loudly.
- A conv without bias is **not rejected**; the exporter **synthesizes a zero bias** and normalizes to
  arity 3 (so as not to bring arity branches into the kernels and the contract). `linear` follows the
  same approach: **the IR's `linear` requires 3 inputs**, while **the export entry accepts no bias and
  synthesizes a zero bias** (`_h_linear` in `aten_handlers.py`; the synthesis is the exact identity
  `+0`). Bias-less is not assumed to be a rare exception — in the measurements of ADR
  [0016](decisions/0016-anima-chain-export.md), 698 of 711 linears have no bias.

## upsample_bilinear2d (align_corners=True) does not guarantee "exact match" at the end points

Even with `align_corners = True`, the edges of the output **do not necessarily match the edges of the
input bit for bit**. The source coordinate is made by multiplying the output index by
`scale = fl((in−1)/(out−1))` rounded to f32 on the host, so shapes really exist where
`fl(scale · (out−1))` falls slightly below `in−1` (measured: the last element for in=2 → out=42 is
0.9999999403953552).

This is by-design, and **torch itself produces the same value** (it evaluates
`area_pixel_compute_scale` in float and multiplies `scale · dst_index` in float). The numeric contract
of the kernel is "match torch's `UpSample.h`", so clamping the end points to make them exact would
instead split the bits from torch (not allowed on the default path in light of ADR 0058's opt-in
contract). Measured extent of occurrence (2026-08-16): of the 128,961 pairs `2 ≤ I ≤ 64` ×
`2 ≤ O ≤ 2048`, 11.6% are inexact. However, `O = 2I` / `O = 2I−1` (I ≤ 4096) have 0 cases, and the
47 real-shape sites of the distributed models are all on the exact side for both H and W.

## layer_norm / adaLN do not make constant rows (and nearly constant rows) exactly 0

The GPU layer_norm (and the adaLN fusion, which shares the same statistics code) builds the row sum
with an f32 sequential sum and a tree reduction, and rounds the mean only once as
`sum × (1/dim)`. The rounding of the sum, and of `1/dim` when dim is not a power of 2, shifts the mean
from the true value by a few ULP. For a constant row (variance 0) the deviation leans to the same sign
on every element, is amplified by a factor of `1/sqrt(eps)`, and remains in the output. torch returns
an exact 0 with Welford's method, so the disagreement is limited to this degenerate input (a "nearly
constant row" whose variance is on the order of ulp² of the mean).

Measured (RTX 3080 Ti, 2026-10-11, eps 1e-5, weight 1, bias 0): with dim 7, a row of all 3 gives
−7.54e-5, and a row of all 12288 gives −0.295. Rows of all 3 with dim 8 and dim 5120 give 0. However,
whether it becomes 0 depends not only on dim but also on the constant's value, and even real hidden
dimensions cannot be called safe. Simulating the GPU's reduction order on the CPU (the 2 values for
dim 7 match the GPU measurements above), even dim 4096 (a power of 2) gives non-zero results for the
constants 0.1 / 1.7 / 123.456 (−7.2e-3 for 123.456), dim 5120 is also non-zero for many constants, and
the magnitude depends on the constant (0.61 for the constant 7282.34).
The simulation procedure: each of 256 lanes sums elements i, i + 256, … sequentially with
`Math.fround`, the 256 partial sums are added by a tree that folds them in halves, and the mean is
made with `× fround(1/dim)` (the variance in the same order).

The fix would be to subtract the first element of each row before reducing (shifted data), but that
changes the reduction order and with it the output bits of layer_norm and of the adaLN fusion and the
sha reference rows. **The maintainer's ruling (2026-10-11): do not fix it; keep only the record in this
section** (Codex deep review R2B-001).

## The GRU scan supports hidden width up to 256, does not include the input-side GEMM, and does not return `h_n` (ADR 0056)

- The hidden width of `gru_scan` / `gru_scan_reverse` is **`H ≤ 256`** (1 lane = 1 hidden unit.
  Degrading silently would corrupt other units' states without an exception through out-of-range
  writes to workgroup-shared memory, so it fails loudly). The check is **double**: out-of-range values
  fail with `OpContractError` in the contract layer (`ops/shapes.ts`), and the path that calls the
  kernel directly fails with `CodegenError` from its params (`kernels/gru-scan.ts`) — the limit
  constant lives in one place, `GRU_SCAN_MAX_HIDDEN` in `codegen/limits.ts`. The only measured shape is
  H = 128, and raising the limit needs grid-stride within the workgroup and double-buffered state = a
  separate design decision.
- The op carries **only the sequential hidden-side part**. The input-side GEMM
  (`x·W_ihᵀ + b_ih`) **is prepared by the caller with the existing `linear`** (a separate node in the
  IR). Thanks to this split, the input-side weights stay eligible for f16 / i8 storage, but **`W_hh` is
  an in-op slot and is ineligible for low-precision storage** (it is not in `WEIGHT_SLOTS`).
- The output is **only `y[T,N,H]`; it does not return `h_n`** (the op contract has one output). Models
  that consume the final state cannot be expressed at present.
- **There are no fields for multi-layer / bidirectional / `has_biases=False` / `batch_first` /
  `dropout`**. Layers and directions are expressed by the exporter lining up nodes (the `Tensor[16]` of
  `aten.gru` does not fit in the IR).
- **LSTM is not in the vocabulary** (`aten.lstm` fails loudly as an unsupported op). The points for an
  extension are in ADR [0056](decisions/0056-gru-scan.md) decision 8.

## The gated delta rule scan runs head dims 128 only on the GPU, has no state slots, and leaves positive g unspecified (ADR 0135)

- `gated_delta_scan` (the value form) is defined for any head dims (the shape rule and the CPU
  reference hold the semantics), but **the GPU kernel implements `dk = dv = 128` only** (the only
  measured head dims — the kernel stages q / k rows of exactly this length). Other head dims fail
  loudly with `ExecutionError` when the Session is built (`validateGraphContracts` in
  `runtime/plan.ts`) and with `CodegenError` from the kernel's params; the constant lives in one
  place, `GATED_DELTA_SCAN_HEAD_DIM` in `codegen/limits.ts`. Generalising needs a measured geometry
  per head dim.
- **There is no initial-state input and no final-state output**: the value form starts from a zero
  state and returns only `o` (prefill). A carried state for decode is the states form, which is not
  in the vocabulary yet (ADR 0135 decision 4).
- **`g > 0` is outside the contract's numerics**: the accurate exp of the kernel is defined for
  `x ≤ 0` and its result above about 88.4 is unspecified (not guaranteed to be +∞), and a positive g
  makes the recurrence expand. The op does not check g (in Gated DeltaNet
  `g = −exp(A_log)·softplus(·) ≤ 0`). `g = −∞` gives a decay of exactly 0, `g = ±0` of exactly 1,
  and NaN propagates.
- The workgroup count is `B · Hv · 4`; above `maxComputeWorkgroupsPerDimension` (65,535 by default,
  that is `B · Hv > 16,383`) the dispatch fails loudly (there is no grid-stride).
- Measured only on the RTX 3080 Ti under Node + Dawn (Vulkan) at B = 1: other GPUs and backends,
  B > 1 on the GPU and the real MOSS `A_log` / `dt_bias` are not measured (ADR 0135 open point 10).

## Vowel detection: the input's 10ms frame count is **even**, and the limit is the distribution's declaration (ADR 0056 / 0057)

- The time axis of the graph input is `2T` (the symbol `T` is the 20ms grid of the output). **Odd
  lengths are not accepted** — `bindSymbols` fails with `actual 285 does not have the declared form '2T'`
  (no rounding). The caller **truncates** the last frame before passing it
  (`vowelDetector.Pipeline.detect` and the real-weight E2E do so).
  Matching the length with right zero padding is **forbidden** (the backward GRU carries state back
  from the pad, and the `.lab` changes anywhere in the utterance — measurements in the ADR 0056
  addendum).
- The length range is `pipelineConfig.minFrames` / `maxFrames` (the distribution's declaration = the
  range of the symbolic dimension `T` at bake time, converted to input-side 10ms frame units — **twice**
  the numbers of `Dim(min, max)`, not those numbers themselves). **The IR does not carry symbol
  ranges**, so this check in the pipeline is the only thing that can stop out-of-range values. Both
  exceeding the upper limit **and falling below the lower limit** fail loudly (no truncation — a
  `.lab` whose tail was silently dropped cannot be told apart from a normal result. Filling the lower
  side with right zero padding is forbidden for the same reason — the backward GRU carries state back
  from the pad). The current distribution is 60,000 frames = 10 minutes, based on the largest
  intermediate tensor = the gates on the GRU input side `[T,1,384]` f32 = 1,536 B/20ms frame → 46.1MB
  (a 2.9× margin against the spec default `maxStorageBufferBindingSize` of 128MiB = 134.2MB).
  **Longer lengths are not measured**.
- Even within the limit, execution time is proportional to the length (the time loop is sequential
  within one workgroup — performance not measured).

## The relative-position tables of flow / voice are graph inputs — generating them is the host's responsibility

SBV2's `flow` / `voice` require the `(T,T)` tables of relative-position attention (`idx_k` / `valid`)
**as graph inputs** (baking them at Tmax = 4096 would be 134MB — ADR 0013). Therefore **the caller
must generate and pass the tables for each T**, and the runtime does not check their correctness (as
values they are just i32 indices and an f32 mask, in the silent-wrong-value class where an off-by-one
does not cause a shape error).

- The canonical formula is `build_relattn_tables` in the Python-side `tools/export-recipes/sbv2/patch.py`
  (the in-graph construction of front calls the same function).
- The canonical host side is `packages/models/src/sbv2/relattn-tables.ts` (SBV2-specific, so it is
  not placed in `packages/runtime/src/` — the models package, which holds model-side knowledge, owns
  it, and `sbv2.Pipeline` calls it for each T). Byte equality with the Python side is pinned by
  `packages/models/tests/sbv2_relattn_parity_test.ts` with the golden's real data.
- A mismatch in the window width (measured 4) is in the same silent-wrong-value class, so the Python
  side rejects it with `_assert_window_size` at ckpt load (scanning all of net_g), and on the TS side
  the parity test compares it with the width `2w+1` of `idx_v` baked into the container.

## SBV2 host glue is evaluated in f64 — `w_ceil` is not bit-identical to torch

`durationsToFrames` / `buildZp` (`packages/models/src/sbv2/host/`) evaluate whole expressions in JS
f64 and round only once on assignment to `Int32Array` / `Float32Array` (the same house style as
`random.ts` in the same directory).
The reference side is sequential f32, so the following are known by-design differences:

- **`w_ceil` off by one frame**: when the f32 half ulp (relative 6e-8) between `f32(exp_f32(x))` and
  `exp_f64(x)` crosses the threshold of `ceil`, it moves by 1 frame (measured: 5.5e-7 per phoneme,
  P ≈ 1.3e-4 for one utterance of 229 phonemes). Even aligning to sequential `Math.fround` **does not
  make the torch match deterministic** — the GPU/CPU difference of 1e-5 in the front output itself
  causes the same threshold crossing at a rate more than 2 orders of magnitude higher, and the dominant
  term remains upstream (`tools/export-recipes/sbv2/README.md` allows it by design, and the positions
  that split are put in `w_ceil_diffs`).
- **1 ulp difference in `z_p`**: it always occurs in about 40% of the elements. It is 3 orders of
  magnitude below the measured maxAbs 5.16e-5 of the waveform comparison that measures this path, and
  since there is no discretization in between, it is not amplified in shape.

Neither affects the reproducibility of karume on its own (the WAV sha256 check, the match of the
stage 1 / stage 2 paths) — the f64 path is deterministic, and the differences appear only relative to
the torch reference.

## SBV2: utterance editing accepts only "the range that does not change the phoneme sequence" (edits that change the phoneme count are rejected)

Since 0.6.0 (ADR [0079](decisions/0079-sbv2-two-layer-input.md)), text analysis is **the caller's
responsibility**, and `generate` accepts one analysed utterance `sbv2.Utterance` (the mora layer). There
are 2 editing surfaces, and the check looks only at **whether the phoneme sequence assembled from
`moras` matches the phoneme sequence of `words` position by position**
(`assertWordPhones` in `packages/models/src/sbv2/text/model-input.ts`). Therefore:

- **Passes**: changes to `result.accentPhrases[i].accentNucleus` on the phrase layer `sbv2.Phrases`
  side and **re-splitting / joining phrases** (just convert again with `sbv2.toUtterance` — this is where
  VOICEVOX-compatible accent phrase editing goes), and **direct editing of `tone`** on the mora layer
  `sbv2.Utterance` (arbitrary 0/1 patterns that cannot be written with a nucleus).
  The check does not look at the phrase / mora boundaries themselves.
- **Rejected** (`sbv2.InputError`): edits that change the phoneme sequence, such as rereading a mora
  (rewriting `vowel` / `consonant`) or adding / removing symbols. Utterances that mix `moras` and
  `words` taken from different analyses also stop here.

NOTE: rearranging phrases / moras can produce sequences where **different tones ride** on the
consonant and the vowel of one mora (a shape that does not arise on the normal analysis path). No
invariant breaks (`sum(word2ph) === P` holds, and the BERT input and word2ph still derive from
`words`); the impact is on the output sound only.

To change a reading, **apply a correction dictionary in the caller's analyser and redo the analysis**
(`analyzeWithWords(dict, text, overlay?)` of `@hdae/yomi` and the like) — the BERT input text and
word2ph are rebuilt correctly, so the quality is higher than a path that swaps only phonemes. Karume
has no place for a correction dictionary (removed in 0.6.0 — both fetching the dictionary and analysis
are outside models). `adjust_word2ph` of the reference implementation (a compatibility hack that
redistributes word2ph by LCS) is not ported: there too the BERT features still derive from the original
text, so the inconsistency is not resolved (the principle of ADR 0072 decision 8 survives in 0079).

Only one kind of edit cannot be accepted — **replacing a reading in a way that does not match word
boundaries** (only some moras within a phrase to a different reading). The "replace the reading of a
whole phrase" operation equivalent to VOICEVOX's `/accent_phrases?is_kana=true` falls under this.

## SBV2: the interrogative "raise only the end" cannot be expressed (tone is binary 0/1)

SBV2's `tone` is **binary 0/1** representing high / low within an accent phrase, a discrete input that
just looks up a row of `front`'s tone embedding (`toneStart` / `numTones` in
`packages/models/src/sbv2/text/symbols.ts`).
VOICEVOX's `is_interrogative` / `enable_interrogative_upspeak` are operations that raise the mora f0 of
an AudioQuery afterwards, so this discrete tone has nothing to map them to. A value that goes outside
the JP tone base only looks up **a tone row of another language** and does not raise anything (which is
why `toSbv2PhoneTone` rejects `sbv2.Mora.tone` other than 0/1 with `sbv2.InputError` — the type is
`0 | 1`, but that has no effect on calls from JS).

There is only one path by which the interrogative shows in the sound — a `?` actually present in the
text **enters as one phoneme** (`toSbv2PhoneTone` turns real symbols into phonemes with tone 0). The
rising intonation is produced by the model as learned prosody. The reference implementation is the
same and has no raising mechanism (it only treats `?` as one symbol of the phoneme table).

Therefore:

- If the caller wants to add an interrogative, **put a `?` into the text to be read and redo the
  analysis** (the step that calls the analyser). A path that adds it later to the `punctuations` of an
  analysed utterance fails the check (`assertWordPhones`) because the phoneme sequence on the `words`
  side does not follow.
- In a layer that imitates the VOICEVOX-compatible API, **accepting and ignoring**
  `enable_interrogative_upspeak` matches reality (AivisSpeech Engine treats it the same way — source
  investigation of 2026-08-21). However, a normalization that "strips `？` from the moras and folds it
  into the `is_interrogative` flag" erases the interrogative information completely, so do not add
  that.

## Anima: `pipelineConfig.scheduler.shift` accepts only values for which the sigma ladder holds

The `shift` a distribution declares must not only pass "a positive finite number"; the sigma ladder
built from it must itself pass the structural check of `sigmaSchedule`, otherwise it fails loudly —
**values whose first entry exceeds 1 in f32** (even by 1 ulp), **values whose first entry is more than
4 ulp below 1**, and **values that are not strictly decreasing** are rejected (writing the accepted set
in 2 places makes one of them inevitably drift looser, so only the ladder side holds the check). The
reason is that the downstream `dpm-solver-multistep` has "first = 1, last = 0" as a stated premise, and
letting such values through would run NaN or a different trajectory with an unknown root cause. When
the first entry exceeds 1, DPM++ 2M's `log(1 − σ₀)` becomes NaN as the log of a negative number, so the
upper side is rejected including when used with Euler. For this reason decimals such as
`shift = 1.32` / `1.57` (55 of the 1,901 two-decimal values in `[1,20]`) are not accepted. Decimals that
shift 1–2 ulp downward (e.g. `1.18`) are accepted. The shift = 3 family of the real distribution has a
first entry of exactly 1, so it passes.

## Anima: steps accepts only integers from 2 to 1,000

`steps` of `generate` and `pipelineConfig.defaults.steps` of the manifest accept only integers from 2
to 1,000 (the check is the single `assertAcceptableSteps` in `packages/models/src/anima/sampler.ts`,
and the manifest check calls it too. In `generate`, exceeding it is a `ModelInputError` before the sigma
ladder is allocated; in the manifest, it is an `Error` as an asset inconsistency — the original
`ModelInputError` is kept in `cause`). The upper limit is a practical value aligned with Wan (1,000 of
`num_train_timesteps`), and Wan's rationale that "beyond it the integer timesteps repeat" does not apply
to Anima — Anima's timestep embedding uses the continuous value of σ as is. The purpose of the limit is
to stop huge steps at the entrance before they allocate the ladder and run a synchronous loop that
cannot be interrupted.

## Anima: preview images during generation are not available (intermediate results are raw latents only)

The only intermediate result the `denoise-step` of `anima.GenerateRequest.onEvent` can hand over is **a
copy of the latents** (`copyLatents()`); per-step preview images are not offered, by design. In the
default per-stage mode the VAE decoder can be loaded only **after** the DiT is released (keeping all 4
resident at once does not fit in VRAM — ADR 0016 / the module doc of `anima/pipeline.ts`), so there is
structurally no path that runs the VAE in the middle of the denoise loop. Consumers that need a preview
approximate it from the latents — the public helper for that is `approximatePreview`
(`@karume/models/anima`): passing the 2 fields of `copyLatents()` **as they are** (without
denormalizing) returns RGBA at latent resolution. The coefficients are calibrated **in the normalized
space** (measured 2026-08-24), so passing values denormalized with the mean / std of `anima.latentStats()`
gives a different, blown-out image. The coefficients themselves MUST NOT be `export`ed, so there is no
path for the consumer to copy them on its own
(it is a linear projection 16ch → 3ch, not an exact decode — the implementation doc is canonical for
the details).
The `stage` event (start / end of a stage — in per-stage mode, before Session construction / after
release) and the `vae-tile` event (once per tile) let you observe the progress of GB-class loads and
of tiled decode.

## Anima: DiT residency (`residency: "transformer"`) raises the VRAM ceiling and cannot be used on 4 GB-class GPUs

`"transformer"` in `anima.PipelineOptions.residency` / `anima.GenerateRequest.residency` is an opt-in that keeps the DiT
Session across generate calls (ADR [0112](decisions/0112-anima-transformer-residency.md)); its VRAM assumption differs from the
default `"per-stage"`. By-design constraints:

- **The chain peak goes up**. The text stage and the VAE stage sit **on top of** the resident DiT. With the default preset
  (`f16+dit8-a8-attn8-s16`) at 1024², the resident part is +2,646 MiB (weights 1,871 + backing 768 + inputs 7 MiB — runtime
  accounting of 2026-09-10), and the chain peak while resident is about 3.9–4.1 GiB (speculation — there is no runtime
  accounting of the text stage). That is about 1.3–1.5 GiB higher than the chain peak of per-stage operation (DiT stage
  2,647 MiB). **It cannot be used on 4 GB-class GPUs**.
- **The quant preset doubles it**. The DiT weights are 1,875 MiB for the default preset and 3,733 MiB for the `f16` preset
  (f32 compute — VRAM table of the 2026-08-05 final-perf-bench. The 1,871 MiB above is the runtime accounting of
  2026-09-10, the same quantity from a different source). Residency of the `f16` preset exceeds +3.7 GB in the text stage,
  so even an 8 GB-class GPU has little headroom (not measured).
- **When VRAM is short, the resident DiT is released and residency stops from then on** (downgrade). Before the text stage
  and before the VAE stage, the next stage's requirement is measured with a trial allocation, and if it does not fit the
  DiT is released before the stage is set up (`evicted` / `headroom`). If the measurement is wrong and the stage hits OOM,
  the DiT is evicted and the stage is re-run (`evicted` / `out-of-memory`). That it fell to the slow path is visible only
  through the `residency` event (`evicted` → then `released` / `downgraded`) —
  without subscribing, you notice it only from the speed. A downgrade does not revert for the lifetime of the pipeline
  (rebuilding it reverts it). In the DiT stage only a DiT carried over from an earlier generate is evicted; an OOM of a DiT
  created in the same generate has the same VRAM layout as per-stage operation, so it is not evicted and is thrown as is
  (ADR 0112 decision 3).
- **Eviction works only in environments that report OOM by type (`GpuOutOfMemoryError`)** (the pre-emptive trial
  allocation is judged by the same out-of-memory errorScope). In environments where Metal's out-of-memory errorScope is
  silent (the known-issues section on Metal's silent out-of-memory errorScope), neither eviction nor downgrade nor any
  event happens even when residency exceeds VRAM. Measured (2026-09-27, the maintainer's M2 24 GB / Chrome 153 / metal-3, check
  page): creating **80 (80 GiB)** STORAGE dummies of 1 GiB never produced an OOM, and on top of that a generate with a
  resident DiT (512²) also passed without eviction — Metal does not back allocations with physical memory, so both the
  trial allocation and the dummies only ever answer "it fits"
  ([research 2026-09-27](research/2026-09-27-h35-oom-device-lost.md) §6).
- **Eviction after hitting OOM (the second line) does not hold when the OOM hit is on the staging side of the weight
  upload**. On Deno that OOM has already invalidated the device, and the re-run stage fails with `GpuDeviceLostError` (it
  does not fail silently — the section of this file on Deno capping total GPUBuffer allocation at 97% of the driver-reported budget). The
  main line is pre-emptive eviction, and the Arc B570 reproduction (resident DiT + 6 GiB dummy) passes with pre-emptive
  eviction (ADR 0112 addendum of 2026-09-27). Other machines are not measured.
- **An evicted stage is re-run from the start**. For the DiT stage, denoise is re-run from step 1 (the pixels produced are
  the same), so that generate's `denoise-step` / `vae-tile` start over from 1 and the wall time grows by one stage.
- **Even when the resolution changes, the steady-state resident amount is at most "weights + the single largest backing"**
  (only one backing over the 256 MiB budget is retained — ADR 0095). In the run right after a switch, the retired old
  backing lives until after the flush, so the transient peak exceeds this. Up to 12 derived plans remain as host objects.
- **The gain appears only when generate is repeated on the same pipeline** (0 for one generation per process, like
  `examples/anima/main.ts` (removed in 0.15)). On the Arc B570 with 1024² turbo, 2.45 s per generation from the second one
  on (10.4% of wall time — [research 2026-09-26](research/2026-09-26-anima-residency-bench.md)). Other machines and Chrome
  are not measured.

## anima: resolutions above 1920px up to 2048px use rope positions outside the model's declaration (accepted; quality not measured)

The accepted upper limit of 2048px per side comes from the 128 rows (= latent 256) of the per-axis base table of the rope.
What sets the number of rows of the base table is the upper limit of the t axis, and the range of the h / w axes the model
declares (`rope.max_size` = 120 tokens = 1920px) is lower than that.
So a request above 1920px up to 2048px can be looked up as table rows, but uses **positions outside** the range the model
declared at training time. The position values follow the same formula as upstream, so bit identity with the static shape
is kept. Any difference would be **quality only**, stemming from positions outside the training distribution, and this is
not measured. The accepted set is not narrowed for this reason (ruling of 2026-09-04 — do not cut the distribution's
freedom without a measurement). In practice the upper limit of S and VRAM bind first. The history is detection limit 2 of
[ADR 0036](decisions/0036-freeform-resolution.md); the record on the code side is the NOTE on `MAX_LATENT_SIDE` in
`packages/models/src/anima/resolution.ts`.

## Wan2.1: accepted sizes are 832×480 / 480×832 only, and frame counts only 4n+1 from 5 to 81 (prompt acceptance depends on the text encoder route — ADR 0118 / 0119)

By-design constraints of `wan.Pipeline` in `@karume/models/wan`. Every input-caused rejection becomes a `ModelInputError` at
the entry of `generate` (`planWanGeneration` — before touching the GPU) (ADR [0118](decisions/0118-wan21-video-generation.md)
Consequences).

- **Accepted set**: sizes 832×480 / 480×832, frame counts 4n+1 from 5 to 81 (default 33). steps is an integer from 1 to
  1,000 (the upper limit is `num_train_timesteps` — integer timesteps are only `trunc(σ·1000)` from 0 to 999, so beyond it
  timesteps necessarily repeat. Rejected before the σ sequence is built. The manifest's `defaults.steps` has the same
  range), guidance is a finite number of at least 1 (1 means CFG is not run — the same as upstream's `guidance_scale > 1`.
  Passing `negativePrompt` with guidance 1 is rejected), shift is a positive finite number. Accepted were 33 / 81 frames at
  832×480 and the VAE at 480×832 (ADR 0118 steps 5 / 6 / 8). The model card and the fixture hold copies of the sizes and
  frame counts (the doc of `WAN21_GENERATION` in `packages/models/src/wan/descriptor.ts`).
- **Prompt acceptance depends on the route**: the default `textEncoder: "gpu"` accepts any sentence (umT5 runs on the GPU —
  the acceptance conditions are in "text encoder route" below). `"precomputed"` accepts only the 4 precomputed ones (3
  positive + 1 negative), and only a string that exactly matches the original text or the normalized string of the text
  embedding asset (listed by `wan.Pipeline.prompts`).
- **A timing (`gpuTiming`) device is rejected at construction**: the runtime does not open a batch on a timing device, so
  the VAE stage (1 tile = 1 batch) cannot run. GPU time does not appear in the diagnostics, so look at the elapsed time,
  VRAM (fdinfo and diagnostics) and the submit statistics.
- **The DiT stage and the VAE stage are not resident at the same time**: each stage sets up a Session and folds it, and the
  VAE is opened after the DiT is closed (the same as the default `"per-stage"` of ADR
  [0112](decisions/0112-anima-transformer-residency.md)). There is no opt-in like Anima's DiT residency, so the DiT weights
  are uploaded to the GPU again on every generate. The VRAM peak is the DiT stage (on the B570, 6.19 GiB at 33 frames /
  7.31 GiB at 81 frames — ADR 0118 steps 6 / 8).
- **The VAE always decodes in tiles** (fixed tiles of latent 32 — ADR 0118 ruling 1): there is an approximation difference
  from the official non-tiled decode (a ratio of 8.2e-2 observed at step 5).
- **The initial noise of a seed is a different sequence from torch's `randn`** (the same splitmix64 + Box–Muller as Anima —
  ADR 0118 decision 5): the same seed as the official implementation does not give the same video. To match a reference,
  pass `latents` from outside.
- Abort (`signal`): accepted both at construction (`wan.PipelineOptions.signal` — passed to the fetch layer) and in generation
  (`wan.GenerateRequest.signal` — checked at stage boundaries, at every DiT step and at every VAE tile. The open Session is
  folded and then `signal.reason` is thrown unwrapped) (ADR 0119 decision 9, 2026-10-03). Within one stage (one run) the
  work is indivisible, and an abort takes effect at the next boundary.
- **sha256 reference values are rows per environment** (ADR [0106](decisions/0106-device-keyed-references.md)): the rows
  that exist today are the B570 (`deno-intel-graphics-bmg-g21`) and the RTX 3080 Ti (`deno-nvidia-geforce-rtx-3080-ti` — written from the frozen copy `128b511e` by the ruling of the ADR 0121 addendum "the development machine changed"). Since 0.15 the tests run on Node and compare with the `node-nvidia-geforce-rtx-3080-ti` rows; the `deno-*` rows stay as history, neither written nor checked (ADR 0127 addendum of 2026-10-28). On a machine without a row the sha comparison SKIPs explicitly and the reference guard turns red
  (`KARUME_REFERENCE=write` creates your machine's rows). The 50-step rows are compared only with the opt-in
  (`KARUME_WAN_FULL_PIPELINE=1`).

- The DiT's i8 presets (ADR [0120](decisions/0120-wan-dit-w8a8-seat.md), 2026-10-03): the reference preset `f16+dit8` and the practical preset `f16+dit8-a8-attn8-s16` were added to the distribution. The GPU numeric checks (r check, self A/B check) and the visual A/B (steps 3–6) are done (2026-10-03 to 04). The default is the practical preset (ruling of 2026-10-04 — `f16` is chosen explicitly with `quant: "f16"`). The practical preset's a8 is not faster on Metal (M2) (measured in K-70 — it becomes a preset that only costs quality). (→ 2026-10-17: when a8 is requested on a device without dp4a, the runtime warns once per device — ADR 0128 decision 5)
- Chrome (step 9 ✅ — the range confirmed is the precomputed route (`textEncoder: "precomputed"`), the `f16` preset, 832×480, 81 frames, 50 steps. RTX 5070 Ti, ADR 0118 addendum 2026-10-04 "step 9: 81 frames × 50 steps complete in Chrome on the RTX 5070 Ti". As of the same addendum, the GPU route and the practical preset had not been run in Chrome): the default binding limit of 128 MiB is not enough even for the VAE's largest value of 192 MiB. `acquireGpu` requests the adapter values,
  so the actual limit depends on the adapter, and the required minimum is the FFN intermediate of 503,193,600 B (33 frames) / 1,174,118,400 B (81 frames). The environment key
  changes with the WebGPU Developer Features flag (separate sha rows). The check page is the Wan tab of tools/gpu-lab (removed in 0.15).
- Text encoder route (ADR 0119, 2026-10-03): the default `textEncoder: "gpu"` runs umT5-XXL on the GPU and accepts any prompt (2–512 tokens;
  out-of-vocabulary and in-text special tokens, HTML entity candidates (`R&D` → `R & D`), mojibake, C1 and characters unassigned in Unicode 16.0.0 are rejected with `ModelInputError`).
  `"precomputed"` accepts only the 4 of the embedding asset and does not fetch the umT5 component. The umT5 component is a cross-repository reference to the separate repository `karume-umt5-xxl`, and locally it
  needs the `crossRepo` mapping (not needed for a construction that replaces `text_encoder` — ADR
  [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md) addendum "step d: replacing a cross-repository slot needs no `crossRepo` mapping"). The `requiredLimits` of every quant preset declare a binding / buffer limit of
  1,050,148,864 B for the vocabulary embedding (the same for precomputed).

## Wan2.1: what the umT5 replacement (`components: { text_encoder }`) does not check (ADR 0122 decision 7)

The acceptance conditions of replacing with a compatible umT5 (component replacement — ADR [0108](decisions/0108-container-format.md) decision 19 /
[0122](decisions/0122-umt5-upstream-and-compatible-encoders.md) decision 7) are limited to those decided mechanically from the two manifests and the container
declarations: the replacement's manifest has the role and the quant; the sha256 of the graph descriptor equals the original manifest's declaration;
the container's capability and zero shortage / excess in the binding table (`packages/models/src/hub/components.ts:471-524`); and Wan's family check (the graph's input/output
contract, and storage of only i8 weights and f32 tables — `umt5Contract`; `packages/models/src/wan/text-stage.ts:150-162` adds the comparison with the DiT's context to
`umt5GraphContract` (`packages/models/src/umt5/graph-contract.ts:73-153`), which it shares with the standalone umT5). This is by-design, to keep the compatibility decision
to a single content hash, and the following 3 are not checked (the reasons for not adopting options B / C and the option of baking the bucket configuration into the graph,
and the conditions for reviving them, are in ADR 0122 "Alternatives not taken").

- **The prompt is silently misread**: plugging in a component trained with a different tokenizer of the same vocabulary size is accepted without an error, and generation proceeds with the prompt
  split into a wrong id sequence. Mechanism: the replacement takes only the container; the tokenizer asset stays the original manifest's
  (`packages/models/src/hub/components.ts:538-542`). Nobody checks the combination of component and tokenizer (no real case yet).
- **The output silently degrades**: plugging in a umT5 with a different relative-position bucket configuration is accepted without an error and encodes
  with different positional relations than intended. Mechanism: the bucket table configuration (`numBuckets` 32, `maxDistance` 128 — `packages/models/src/umt5/relative-position.ts:38`) and
  the valid length 512 (same file `:51`) are host constants not in the graph descriptor, so the containers of umT5s with different configurations can be byte-identical (measured in
  ADR 0122's independent check). Taking in an upstream without a `config.json` copies the original's config, so the intake comparison is empty too
  (`intake.json` records "configuration estimated" — ADR 0122 decision 5, item 5).
- **Weights of unknown licence run without warning**: the replacement's provenance and licence are neither checked nor recorded at run time
  (`packages/models/src/hub/components.ts:465-489` looks only at the manifest and the sha256 of the graph descriptor). The generated output does not reveal which weights
  were used either. The runtime does not know the provenance (ADR 0108 decision 19). Weights with an undeclared licence are stopped on the intake, conversion and
  local experimental mirror side (the explicit argument `--allow-undeclared-license`, outside `models/`, the pre-publication check — ADR 0122 decision 6), but plugging in a mirror
  assembled locally is not itself stopped.

## Wan2.2: accepted sizes are 1280×704 / 704×1280, frame counts 4n+1 from 5 to 121; I2V uses the same 2 sizes; 121 frames confirmed to complete on the 12 GiB class (the 10 GB class was confirmed up to 57 frames — ADR 0121)

By-design constraints of `wan.Ti2vPipeline` (Wan2.2 TI2V-5B) in `@karume/models/wan`. Input-caused rejections become `ModelInputError` at the same entry as Wan2.1 (before touching the GPU)
(outside the accepted set fails loudly).

- **Accepted set**: sizes 1280×704 / 704×1280, frame counts 4n+1 from 5 to 121 (default 1280×704, 33 frames — fixed on 2026-10-09). The sizes and frame counts are
  `WAN22_TI2V_GENERATION` in `packages/models/src/wan/descriptor.ts` (ADR [0121](decisions/0121-wan22-ti2v-5b.md) addendum (2026-10-05)
  "accepted sizes moved to the two official sizes"; the upper limit 121 is the addendum (2026-10-07) "stage 10: 121 frames accepted with development-machine sha rows").
- **The default shift was set to 5.0** (the pinned scheduler's `flow_shift` = the official 720P value). The code holds no default for shift and reads it from the manifest's
  `pipelineConfig.scheduler.shift`. 5.0 is the value the `pipelineConfig` of the distribution `karume-wan2.2` (stage 8a) declares; with `fromAssets` the manifest passed
  decides it.
- **832×480 / 480×832 are not accepted**: they are outside the official implementation's supported sizes (1280×704 / 704×1280). Generating at them on the B570 broke the picture even with other presets and shifts
  (thick outlines and flat fills). 1280×704 came out photorealistic with the same code and the same weights. Whether upstream (diffusers, CPU f32, the original weights) looks the same at 832×480
  is not confirmed.
- **The frame count upper limit of 121** (the official implementation's default) is the value at which 1280×704 with 50 steps completed with the practical preset on the development machine's RTX 3080 Ti (12,288 MiB, Deno 2.9.6)
  (2026-10-06, 0 non-finite values). The VRAM peak of the DiT stage was 11,361 MiB (whole GPU per `nvidia-smi`), about 0.9 GiB short of the physical capacity. The DiT's diagnostic values (weights + backing) are
  4.67 + 5.97 = 10.64 GiB. The same request also completed in Chrome on an RTX 5070 Ti (DiT diagnostic value 9.47 GiB). The measurements are the ADR 0121 addendum (2026-10-06)
  "RTX 5070 Ti Chrome: 1280×704 × 121 with the practical preset completed" and "RTX 3080 Ti lanes and full verify; 1280×704 × 121 on the development machine". The reference preset also completed 1280×704×121 (2 steps) on the development machine (VRAM peak
  10,233 MiB (whole GPU per `nvidia-smi`)) and has a development-machine sha row (ADR 0121 addendum (2026-10-07) "stage 10: 121 frames accepted with development-machine sha rows").
  Even at 12 GiB the headroom is thin, and when the GPU is shared with other processes it may not fit (speculation — not run).
  - **On Deno the actual headroom can be thinner than about 0.9 GiB** (speculation): 0.9 GiB is the difference between the whole-GPU peak and the physical capacity. What binds first on Deno is the ceiling on total allocation
    of the section above on Deno capping total GPUBuffer allocation (97% of the budget the driver reports; it moves over time), and on this machine 7,280 MiB and 11,136–11,264 MiB have been
    measured. The ceiling during the 121 run was not measured (it completed, so the allocation at that time was within the ceiling at that time). The DiT diagnostic value of 10.64 GiB
    (about 10,895 MiB) is only a few hundred MiB away from the 11,136 MiB ceiling (the diagnostic value is not wgpu's heap usage itself).
- **On the 10 GB class (e.g. the previous development machine, Intel Arc B570; Deno's total-allocation ceiling of about 9.4 GiB (9,600 MiB) — the section above on Deno capping total GPUBuffer allocation), the length
  confirmed is up to 57 frames**. The largest at which both presets completed 50 steps on the B570 is 49 frames (only the practical preset also completed 57 — headroom of 0.63 GiB from the ceiling minus the peak,
  the trial-run table of the ADR 0121 addendum (2026-10-05)). **Longer lengths (61–121 frames) are not supported on the 10 GB class**. None of them was
  run on the B570. They are unsupported because they are not confirmed, and the outlook for fitting differs by length (all speculation):
  - 81 frames and 121 frames: the 2 points estimated by the table in ADR 0121 Context "Capacity and time estimates". 1280×704×81 (S = 18,480) fits only on the optimistic side for the reference preset,
    and the practical preset does not fit. 1280×704×121 (S = 27,280) fits for neither preset. At 121, even the practical preset's DiT diagnostic value on the development machine (RTX 3080 Ti) of
    10.64 GiB exceeds the ceiling.
  - 61–77 frames and 85–117 frames: there is not even a capacity estimate, and whether they fit is unknown (the headroom at 57 frames is a thin 0.63 GiB).

  The accepted set is not per machine, so lengths over 57 frames can be requested from such a machine too. What it looks like then:
  - **admission does not reject it**. admission compares only against the device's absolute limits (`maxBufferSize` / `maxStorageBufferBindingSize`), not against free VRAM or
    the total of allocations (ADR 0089 decisions 1 / 3, ADR 0070 decision 5). Every binding of the 1280×704 family fits in 2 GiB even at 121 frames (FFN intermediate
    at most 1.46 GiB, score row block at most 1.96 GiB — ADR 0121 decision 8). On an adapter whose binding limit is smaller than that, the plan-time preflight (ADR 0093 decision 5) is designed to list every slot that exceeds it
    and fail before allocation (not confirmed at 1280×704×121).
  - **Exceeding the total fails in the middle of execution** (fail loudly — no silent shrinking or approximation). The final check is the out-of-memory errorScope
    (`GpuOutOfMemoryError`). Near the ceiling it can become a device loss rather than an allocation failure (the section above on Deno capping total GPUBuffer allocation).
    The DiT stage is where it is expected to exceed (speculation), and the time of the stages before it is wasted. Not confirmed on the B570.
  - **The concern about the time of one submit** (on the B570, the estimate that at S ≥ 18,480 (1280×704 at 81 frames or more) the 16 dispatches before backing can exceed 5 s —
    ADR 0121 Context "Capacity and time estimates") does not apply, because the 10 GB class does not support lengths over 57 frames (the concern is at 81 frames or more;
    what ran on the B570 is up to 57 frames (S = 13,200)). No runtime-side measure to shorten submits was adopted.
- **The row-block budget (ADR [0124](decisions/0124-attention-row-block-budget.md)) does not change this unsupported range**: the runtime's
  `SessionOptions.attentionRowBlockMaxBytes` shrinks attention's momentary temporaries, but `wan.Ti2vPipeline` has no option for it, and
  there is no machine to confirm whether 61 frames or more fit on the 10 GB class. With the practical preset at 1280×704×121 on the 12 GiB-class development machine (RTX 3080 Ti, Deno), a budget of
  512 MiB lowered the DiT's backing from 5.97 → 4.51 GiB, and the whole-GPU VRAM peak when running only the DiT from 11,308 → 9,811 MiB (measured
  2026-10-11; output bit-identical to no budget; [research 2026-10-11 (step 4b measurements)](research/2026-10-11-row-block-budget-stage-4b.md)).
- **There are 2 presets: the reference preset `f16+dit8` and the practical preset `f16+dit8-a8-attn8-s16` (default)** (there is no f16 preset — ADR 0121 decision 2). The default practical preset is
  the preset adopted from the maintainer's visual review (1280×704×33, 12 videos at 50 steps — 2026-10-06). In the self A/B check against the reference preset (ADR 0121 stage 7), the relRMS of the step-1 latent
  is 4.39e-2 at 33 frames and 4.92e-2 at 121 frames (bands 8.8e-2 / 9.9e-2; the 121-frame check is opt-in `KARUME_WAN_TI2V_121F=1`).
  The visual review (pairs of 12 reference-preset and 12 practical-preset videos — 2026-10-09) also shows no noticeable degradation.
- **sha256 reference values are rows per environment** (ADR [0106](decisions/0106-device-keyed-references.md)): the 6 cases of the 2-step end-to-end run (reference preset,
  1280×704 / 704×1280, 17 frames) have rows for the B570 (`deno-intel-graphics-bmg-g21`) and the RTX 3080 Ti (`deno-nvidia-geforce-rtx-3080-ti`). Since 0.15 the tests run on Node and compare with the `node-nvidia-geforce-rtx-3080-ti` rows; the `deno-*` rows stay as history, neither written nor checked (ADR 0127 addendum of 2026-10-28).
  The one practical-preset case of the same request (1280×704) has only an RTX 3080 Ti row. The 2-step × 121-frame case is opt-in (`KARUME_WAN_TI2V_121F=1`), with a row only for the RTX 3080 Ti. The 50-step end-to-end run (opt-in
  `KARUME_WAN_TI2V_FULL_PIPELINE=1`, 1280×704, 33 frames) has rows for both presets, only for the RTX 3080 Ti. The I2V 2-step × 9-frame
  end-to-end runs have only RTX 3080 Ti rows: default lane = boxing-cats (1280×704) for both presets; opt-in (`KARUME_WAN_TI2V_I2V_FULL=1`) = the reference preset's
  cat-dog-baking / ferret and 704×1280. The I2V 50-step end-to-end run (same opt-in, 1280×704, 33 frames, fit crop) has rows for both presets
  (the practical preset's row comes with the floor of differing by at least 1 bit from the reference preset on the same request), only for the RTX 3080 Ti.
- **The VAE always decodes in tiles** (tiles of latent 16, overlap of latent 4, blended in patchify space and then unpatchified — ADR 0121 decision 6;
  28 tiles at 1280×704): there is an approximation difference from the official non-tiled decode (a ratio of 1.2e-1 observed on the stage-5 comparison case 480×832×9).
- **Same as Wan2.1**: rejecting a timing (`gpuTiming`) device at construction; not keeping the DiT stage and the VAE stage resident at the same time; the seed's initial noise being
  a different sequence from torch's `randn`; how abort (`signal`) takes effect; the text encoder route (`"precomputed"` accepts the same 4 as 2.1 —
  the embedding asset is byte-identical to the 2.1 series; the `"gpu"` umT5 is a cross-repository reference to `karume-umt5-xxl`); replacing umT5 (`components: { text_encoder }`) and
  the 3 things it does not check are as in the Wan2.1 section above (the body of construction and generation is
  shared with 2.1 in `packages/models/src/wan/family.ts`).
- **Constraints of I2V (`image` / `fit` — ADR 0121 decision 11, stage 9)**:
  - The sizes are only the 2 sizes of the same accepted set as T2V. When `width` / `height` are omitted, the one closer to the image's aspect ratio is chosen (1280×704 if width ≥ height). The official implementation
    chooses a free size within an area limit (1248×704 for a 16:9 image), so with the same image `fit: "crop"` can cut more here.
  - Image files are not decoded (zero runtime dependencies). What is accepted is decoded RGB8, and no colour-space conversion is done either. The preprocessing (LANCZOS downscale and
    crop) is bit-identical to Pillow 12.3.0, but producing RGB8 from a file is the caller's job, and whether the input matches the reference is decided there
    (`examples/wan` (removed in 0.15) read PNG pixel-identically to Pillow, JPEG drifts slightly due to decoder differences, and the EXIF orientation is applied (deliberately different from upstream's `generate.py` — ADR 0121 addendum 2026-10-11) — `examples/wan/README.md` (removed in 0.15)).
  - The range confirmed on Deno is only the precomputed route (`textEncoder: "precomputed"`) on the RTX 3080 Ti: 2 steps × 9 frames (3 images at 1280×704,
    1 image at 704×1280) and 50 steps × 1280×704×33 (sha rows for both presets — the sha256 item above). It was not run on Deno on GPUs other than the RTX 3080 Ti.
    The combination with the GPU umT5 route is only the one that completed in Chrome on an RTX 5070 Ti (gpu-lab, 2026-10-10): condition image 896×1152 (portrait) → 704×1280, fit crop,
    practical preset, 20 steps, shift 7, free prompt, at 33 frames (wall 390.0 s, 5.99 s per step) and 121 frames (wall 1,346.5 s, 36.05 s
    per step), encoder stage 1.6–1.7 s. gpu-lab did not compare I2V rows with reference cases, so no claim is made about quality.
  - The band against upstream (diffusers on CPU f32) is wider than for T2V (3.5e-3 on the step-2 latent — T2V is 9.5e-4). This is because the DiT amplifies small differences at the encoder's exit
    by about 250× through the condition tokens (replacing with the reference condition latent brings it back to T2V's order — ADR 0121 addendum "stage 9b results").
  - The practical preset's quality (the A/B check and visual review above) is the value measured on T2V. The I2V self A/B check (opt-in `KARUME_WAN_TI2V_I2V_FULL=1`, case i2v-33f = 1280×704×33,
    S = 7,920, condition image boxing-cats, fit crop — `packages/e2e/tests/e2e_wan_ti2v_ab_test.ts`) gives a relRMS of 4.6146e-2 on the step-1 latent
    (2026-10-10, RTX 3080 Ti), band 9.3e-2 (the measurement × 2 rounded up to 2 significant digits; floor = differs by at least 1 bit), 1.05× T2V's 33 frames (4.39e-2 /
    band 8.8e-2). maxAbs (2.6× T2V) is in latent frame 0 before replacement by the condition (a frame that does not reach the output). The visual comparison of the 2 presets on I2V
    is stage 9d.
- The distribution `karume-wan2.2` is published on Hugging Face since 0.14.0 (`wan.SOURCES["wan2.2"]`; the recipe is `dist.py --pipeline wan-ti2v`, ADR 0121
  stage 8a). The entries are `wan.Ti2vPipeline.fromPretrained` (the published repository or a local mirror) and `wan.Ti2vPipeline.fromAssets`, which takes a
  manifest and assets assembled from the series containers. Only Wan2.1 (`karume-wan2.1`) stays unpublished.

## EmbeddingGemma: a run-time attention_mask (in-batch padding) is not supported — single sequence assumed

The exported graph (script `tools/export-recipes/embeddinggemma/export.py`) has no `attention_mask`
input. The bidirectional + sliding-window band mask is baked in as **a constant of Tmax=512**
(the first T is cut out with `sym_prefix_slice`), and there is no path that hides padding from attention.
For a sequence with padding, **the caller packs it to shorten T** (for a single sequence this is strictly
equivalent to torch eager).

Mechanism: ADR 0016's proof that the safe-softmax guard is inactive requires evaluating the actual mask values, which does
not hold for a run-time input placeholder (`_eval_static` rejects it — "single sequence assumed" is not a workaround
but a consequence of the design). The graph input `pool_mask` is **for pooling only** and is not wired to attention
(mixing in 0 would give a shape eager does not have, "the model sees it but the pooling discards it" —
always pass all ones).

Related: since `SYM_MAX = 512`, T > 512 fails at Session construction (the config's
max_position_embeddings is 2048). Raising it is to be done together with a ruling on the band mask constant growing with Tmax²
(512 → 2MB / 2048 → 32MB).

The design for lifting it (run-time mask support) is settled in
[decisions/0044](decisions/0044-runtime-attention-mask.md) (accepted), and **the mechanism is
implemented** (the `safe_softmax` op + the two-step `_drop_safe_softmax_guard` — 2026-08-11). What remains is
only the wiring of the EG script (taking a bool mask input, combining it additively with the band constant, and ruling whether SDPA
is kept as is or lowered to its decomposition — ADR 0044 Consequences).

## Irodori text side (backbone / projector): no run-time attention_mask; an empty caption and T=1 are not representable

The same static scheme as EmbeddingGemma (B=1, the caller packs the sequence). Equivalence with "right-padded + mask"
is measured on every emit by a standing check in the export script (`_static_scheme_evidence` in
`tools/export-recipes/irodori/export.py` — measured at most 8.3e-6, check 1e-3. If it breaks, the whole export fails). Related:

- **An empty caption (all-zero mask) is not represented in the graph** — in eager the projector output is strictly all
  zeros in that shape, so the host produces the zeros directly (the same treatment as CFG uncond — under ADR 0044).
- **T = 1 (BOS only) cannot be represented** (the symbolic dimension is `Dim(min=2)` — the default that avoids 0/1 specialization).
  No real input other than an empty caption gives T = 1.

## Irodori speaker / duration: no-reference and the mean-token prepend are on the host; the condition vectors are also host-supplied

A consequence of applying the same static scheme as the text side (B=1, the caller packs the sequence) to the speaker encoder (`[1,S,128]` →
`[1,S,768]`) and the duration predictor (`[1,T,512]` and others → `[1]`). Each is
a by-design line of "it does not go into the graph", and there is no value approximation or silent fallback:

- **No reference (`no_ref` — all-zero reference mask) does not call the graph**. The eager output is
  **strictly all zeros** in that shape (SDPA's safe-softmax returns 0 for fully masked rows, and the trailing `x * mask_f`
  makes everything 0), so the host placing a zero matrix is equivalent. **So that this does not become vacuous, `_no_reference_evidence` in the same export
  script measures it on every emit** (it passes a non-zero latent through
  an all-zero mask, and if the maximum absolute value of the output is not 0 the whole export fails).
- **The mean-token prepend (`_prepend_masked_mean_token`) is on the host in the current pipeline**. IR v2's
  `cat` accepts `1 + S → S+1` (ADR [0046](decisions/0046-cat-symbolic-axis.md)), so
  the reason for leaving it is not "`cat` on a symbolic axis is unsupported". The host-side work is only the mean over axis 1 and the concat, and
  no model computation (operations that use weights) remains. Whether to move it to the GPU is a separate design decision.
- **duration's `speaker_vec` / `caption_vec` are host-supplied**. The former is the slice of the mean token above,
  the latter is **a masked mean over the caption sequence after `caption_norm` (RMSNorm 512)**, and
  `caption_norm` comes out as the **second output** of the `caption-proj` graph (ADR
  [0048](decisions/0048-irodori-host-port.md) decision 1), so **only the row mean remains on the host**
  — not a single operation that uses weights remains. Making the caption sequence a graph input would give
  2 symbolic dimensions (T and the caption length), so it was not adopted (multi-symbol graphs are not measured — recon U3).
  **There is no 0/0 risk** (the implementation divides by `denom = clamp_min(sum, 1.0)` — the
  "masked_mean of an all-zero caption" that recon raised is already closed upstream).
- The choice of no reference / no caption (`null_speaker` / `null_caption`) is represented by **the bool inputs `has_speaker` /
  `has_caption` + a `where` inside the graph**. This avoids distributing the 2 learned vectors
  to the host (the same reason ADR 0010 rejected "host precomputation + extra input").
- **S = 1 and T = 1 cannot be represented** (the symbolic dimension is `Dim(min=2)`). The reference latent is
  after the patch of `speaker_patch_size` = 4, and the only shape where the real input becomes 1 token is `no_ref`
  (which does not call the graph, as above). **A reference that does not reach 2 tokens after the patch (8 codec frames) is not accepted**:
  less than 8 frames for a reference latent, or less than 13,441 samples at 48kHz for a reference audio (7 × hop 1920 + 1 — the tail is
  rounded up to a multiple of the hop by the reflect pad), becomes a `ModelInputError` at the entry of generation (before running the GPU)
  (`assertSpeakerReference` in `packages/models/src/irodori/conditioning.ts`). Silently zero-padding to 1 token
  is not done.
- The upper limit of the symbolic dimension is **S ≤ 750** (`ref_max_seconds` 120s × 25Hz ÷ patch 4 — derived from the checkpoint's
  config). A longer reference fails loudly at the binding check.

## Irodori DiT: the per-query-row limit that row blocks cannot split; CFG supports only the 2 default modes

One DiT step (the `dit` target) is exported in the execution form of ADR [0047](decisions/0047-irodori-dit-execution.md)
(B=1 × symbolic S × G4 convolution × uncond expressed by a mask). As a consequence,
the following 2 points are by-design constraints, with no approximation or silent fallback:

- **A shape whose scores for one query row (`H·C·4` bytes) exceed the binding limit fails loudly**. Decomposed
  attention runs the 9-node window with the minimum number of row blocks determined statically from the granted limit
  (ADR [0060](decisions/0060-row-block-attention.md) = row-block execution. Bit-identical, the default path
  — on a machine with room under the limit, 1 block = zero cost), so the former constraint of the intermediate `scores` at S=750 exceeding 128MiB
  is resolved, but a shape where not even one row fits the limit cannot be split by row blocks.
- **CFG represents nothing other than `speaker_uncond_mode="mask"` (default) and `cfg_guidance_mode="independent"` (default)**.
  uncond can be expressed by a mask alone because "the contribution of the context KV with state set to 0 is strictly 0
  through the mask", and `"noise"` (speaker's uncond as a random state) does not satisfy this
  equivalence. `joint` / `alternating` differ in how the variants are combined in the first place. **Non-default values
  fail loudly in the pipeline layer** (the graph side reduces the difference of the 4 variants to a single mask, so
  without rejecting here a different model would silently run).

## The GPU-resident path of the Irodori DiT loop: FTZ of denormal outputs, degraded diagnostics, host path when timing / onEvent

The DiT loop is GPU-resident by default ([ADR 0054](decisions/0054-resident-loop-and-fence.md) — CFG combination
and the Euler update run as GPU elementwise ops; one batch per fence). 3 by-design constraints:

- **Elements whose final output is exactly denormal (|x| < 2⁻¹²⁶) are kept denormal by the host implementation,
  whereas GPU shader arithmetic can flush them to ±0 of the same sign** (measured by the parity probe: the difference comes only from this mechanism,
  0 fma contractions, signed zeros match). Real latents / velocity fields (unit-variance class) practically never reach this
  region, and the 2 WAV sha256 checks of the reference cases match the digest exactly — the checks are the permanent detector.
- **On the resident path `lastRun` (run arena record) and `lastRunTiming` are `undefined`** (enqueue creates neither
  an arena nor a timing window). `planBacking` / `submit` / `lastRunPrepared` are as before, and the list of pipeline
  keys can be read from `lastRunPipelines` (the planned dispatch count per key of the latest run), which does not depend on timing.
- **A device with gpuTiming enabled, and generation that passes `onEvent` to `generate` / `generateLatent`, branch to
  the former host loop** (timing: incompatible with batch, so `beginBatch` rejects it / onEvent: in a section of 1 batch +
  a single fence the completion of each step cannot be observed from the host, and firing at `enqueue` time would be a lie
  as "progress"). The output has the same digest (the onEvent step of `e2e_irodori_wav_test.ts` makes it a standing check with the
  same sha256 as voice-clone), but the wall time grows (measured on the same case:
  7.2 → 8.6 s / S 170, reference environment 2026-08-16). Except when taking a per-op breakdown or reporting progress,
  use the default (resident path).

## Irodori pipeline (host layer): upstream optional knobs only at their default-equivalent values; reference audio is 48kHz

`irodori.Pipeline` covers the range of phases 3–4 (ADR [0048](decisions/0048-irodori-host-port.md) /
[0049](decisions/0049-irodori-codec-integration.md)), and the following are by-design constraints. There is no approximation
or silent fallback:

- **Upstream inference knobs that are dead at their default values are not ported**: dynamic LoRA loading /
  the speaker_kv_scale family / truncation_factor / temporal_score_rescale / the sway schedule /
  num_candidates>1 / decode_mode="batch". The CFG mode constraint (mask / independent only) is
  as in the DiT section above, and **the pipeline rejects it when parsing pipelineConfig**.
  The tail-trim thresholds (window 20 / std 0.05 / mean 0.1) and the reference audio target of −16 LUFS are also upstream default
  fixed values (not held as run-time knobs).
- **`cfgScales` accepts only values exactly representable in f32** (an inexact value fails loudly when parsing
  `pipelineConfig`). The DiT loop has 2 paths, the host path (multiplying in f64) and the GPU-resident path (rounding to f32
  before multiplying), and when the strength is not exact in f32 the last digit splits by 1–2 ulp on the same input
  (measured: diverges at s=1.3). To make the MUST "the outputs of the 2 paths are the same" hold unconditionally regardless of the distribution,
  it is rejected on the declaration side. The real distribution's 3.0 / 5.0 / 3.0 are all exact in f32 and unaffected.
- **Reference audio (`speaker: { audio }`) only at the distribution's `sampleRate` (48kHz)** — there is no resampling;
  it fails loudly (ADR 0049 decision 6. Conversion is the caller's responsibility). `decodeWav` accepts only
  PCM 16bit and IEEE float 32bit, and `WAVE_FORMAT_EXTENSIBLE` (0xFFFE) and the like are explicitly rejected.
- **`codec_encoder` is not split into tiles** (asymmetric with the decoder — ADR 0049 decision 1). A long reference
  (intermediates of 1.47GB×2 at 120 s) fails to allocate on a machine where `maxStorageBufferBindingSize` is the default
  128MiB. The decoder side is split into tiles with halo 8, so S=750 passes even on a machine with the default limit (with a bit-identity check).
- **Generated audio carries no watermark (SilentCipher)** (the wm branch is baked in its bypass form — ADR 0049 decision 2). An API
  option for one MAY be added when needed (re-graded 2026-10-18, ruling 8a: the earlier deferral to the pre-publication phase expired
  with publication on 2026-08-24; an added option is non-breaking).
- **The seed is not compatible with upstream** (bit reproduction of the torch generator is a non-goal — ADR 0048 decision 5).
  Only self-determinism, same seed → same waveform, is guaranteed; comparison with torch is done through the `initialNoise` injection point.
- The preprocessing `strip` is JS's `String.prototype.trim`, and its whitespace set differs from Python's `str.strip` at the
  edges (U+001C–1F and U+0085 are Python only / U+FEFF is JS only). When a caption consists only of these characters,
  even whether there is a caption (the caption condition and the CFG branch) is inverted against upstream (with only U+0085, upstream says
  "none" and karume says "present"; with only U+FEFF, the reverse). Accepted as a difference that does not occur with practical text input
  (the 33 normalize cases of the golden do not include this region).
- WAV reading (/32768) and writing (×32767) stay asymmetric (each is correct in matching the outside world —
  the MUST in `src/audio/wav.ts`. A round trip drifts at the 1 LSB level).
- **Request knobs (`codecTileFrames` / `seed` / `durationSeconds`) are checked at call time** (it does not fail after
  proceeding partway through the stages). As public-surface behaviour, ① even `generateLatent`, which does not run the codec,
  fails on an invalid `codecTileFrames` ② even generation that passes `initialNoise` has the range of `seed` checked
  (previously that path drew no random numbers, so it was silently ignored).
- **The distribution's declaration side gained 2 requirements**: `speakerRows` is at least 2 (1 mean token + the reference
  latent), and the latent length reachable from `maxSeconds` (the larger of the floor of the duration path and the ceil of the manual-seconds path)
  does not exceed `ditSymMax`. Both fail when parsing `pipelineConfig`.

## Additive mask of fused attention: static `[1,1,M,N]` only, not combined with a8, the bit-identity check is on the f32 path

The fourth input of `attention`, the mask (ADR 0023 addendum 2026-08-11), is deliberately narrow:

- Accepted is **f32, additive, exactly `[1,1,M,N]`** (broadcast to B·H). A run-time bool mask,
  `[B,1,M,N]` (per batch) and `[1,H,M,N]` (per head) fail loudly — to be widened in the phase of the Irodori CFG ruling
  "run-time bool mask (option a)", together with a redesign of ADR 0016's proof that the guard is inactive.
- **mask × `attentionCompute:'a8'` fails loudly** (①QK of the i8a8 family has no epilogue —
  it does not silently fall back to f32). Whether to support it is a design decision for a separate phase.
- The permanent bit-identity check (parity) against the decomposed path is **on the f32 path only**. s16 / c16 × mask is confirmed only up to WGSL
  generation, pipeline creation and execution on a real GPU (whether to add a check or keep stating f32-only in the ADR
  is decided when such a case appears in real assets).

## GQA × `attentionCompute:'a8'` in fused attention fails loudly (provisional)

The GQA / MQA shape (`H % Hkv == 0`, ADR
[0067](decisions/0067-autoregressive-attention-vocabulary.md) decision 1) is on the f32 / f16 paths only.
The a8 path has 5 head bases (`attention-i8a8.ts`), and the quantization and allocation of K / V also assume `B·H`, so
the GQA shape **fails loudly without fallback** (silently dropping to f32 would quietly change performance — ADR 0058
decision 3). **The rejection is provisional and support is planned later** (ADR 0067 decision 3 — the supplementary surface is settled: kv-head mapping
for the K/scale and V/scale bases only + Hkv in the recipe builder. Verification reuses the repeat_kv
parity assets of the f32 path).

## The implementation limit of topk's k depends on the device (k ≤ 63 with the WebGPU default limits)

The scratch of `topk` (ADR [0068](decisions/0068-decode-exit-multi-output.md) decision 3) stays within
workgroup storage, so the upper limit of k is set by `8·32·(k+1) ≤ maxComputeWorkgroupStorageSize`
(`topkMaxK` in `src/kernels/topk.ts`). With WebGPU's default of 16384B, **k ≤ 63**;
on this development machine (49152B), 191. Exceeding it fails loudly with the limit value (no fallback — ADR 0058
decision 3). The practical range of top-k sampling (k ≤ 50 class) fits even on a machine with the defaults. A design to raise the limit
(multi-stage merge and the like) is to be filed in the perf-ledger when there is real demand.

## WAVs with the streaming convention (riffSize placeholder) are not accepted

The scan boundary of `decodeWav` (`packages/models/src/audio/wav.ts`) is the
logical end that RIFF declares at offset 4, and physical bytes outside the declaration are not read (ignored as the spec says — the fields `encodeWav` writes are authoritative).
So the riffSize placeholder declaration, a convention of streaming writers, is by design
not accepted:

- `riffSize=0` puts the logical end right after the header, so not a single chunk is scanned, and it fails with
  `decodeWav: 'fmt ' chunk is missing`.
- `riffSize=0xFFFFFFFF` fails as a truncated container with
  `decodeWav: RIFF declares 4294967295 bytes, but only … bytes remain`.

The input is assumed to be a WAV as a complete file (both for reference audio and for detector input). If a real need arises to
feed streaming WAVs, the accepted form will be ruled on then (no silent fallback to the physical length — the cross-cutting
fail-loudly convention).

## hub: the cancellation granularity of concurrent fetches is the single-flight leader

In the single-flight of the fetch layer (`@hdae/fetch-cache`), the second and later calls for the same **content key**
(`["hf", kind, repo, path, sha256]`) join the earlier flight, and
an `AbortSignal` given by a joiner has no effect (aborting the leader takes the joiners down with it).
With several `fetchAssets` fetching the same asset concurrently, cancellation works only at this granularity
(ADR 0038 §5). The abort of a single call propagates correctly to all workers.

## gemm-family shapes with a zero-element dimension fail on the GPU binding's minimum size (unsupported degenerate region)

A zero-element dimension such as `in=0` of `linear` is valid under the op contract, but the lower bound on allocating a zero-element buffer
(4 bytes — `Math.max(4, …)` in the arena / executor) is below the minimum binding size of the vec4 variant (16 bytes),
so execution fails loudly with `GpuValidationError` (Binding size … less than minimum)
(independent of `linearCompute` — measured 2026-08-13). It does not become a silent wrong value. Lifting it would mean unifying the
allocation lower bound to 16 (a change to the allocation policy shared by all ops — a separate phase if demand appears).

NOTE: only `attention` is an exception: degenerate axes (B = 0 / M = 0 / N = 0) are **explicitly rejected in the contract layer**
(`OpContractError`). The reason is that the row-block planner (`planRowBlocks`) requires positive integers, so letting them through would give
an `ExecutionError` that does not name attention. That zero-element dimensions are contract-valid applies to
the remaining ops, such as the linear family.

## Public surface: 3 points where the rejected set grew after 0.9.0 (the accepted set is unchanged)

Each is a change that makes inputs which "until now silently fell back to something else or failed late" fail loudly at the
entry. The behaviour of correct calls does not change by a single bit.

- **The 4 execution-form knobs of `SessionOptions`** (`linearCompute` / `attentionCompute` /
  `attentionScoreStorage` / `stateAttentionReduce`) reject spellings outside the union at `Session` construction
  (one `ExecutionError` that lists them all). Previously they silently fell back to the default (f32 / sequential),
  so `linearCompute: "i8a8"` (the old spelling renamed to `"a8"` in 0.5.0) ran on f32 without any exception or warning.
- **The range check of `EstimateOptions.maxStorageBufferBindingSize`** (a positive safe integer) runs at the
  **entry** of `estimateGraphMemory` (`PreparedModel.estimate` is the same implementation). Previously it existed only inside the path that counts
  the temporaries of states-form attention, and in graphs without the states form −1 / 1.5 / NaN were silently
  accepted (they were not read, so the numbers did not change).
- **The effective upper limit of `SubmitPolicy.maxChunkSize` is 1,024 on a device with gpuTiming enabled** (query 2,048 =
  querySet capacity). A policy that exceeds it fails at Session construction (construction of the `SubmitScheduler`) with `SubmitPolicyError`
  (up to 0.9.0 it failed in the middle of a run).

## GitHub CI does not exercise tests that depend on local assets (`outputs/`) (a constraint on verification coverage)

`outputs/` is not tracked by git, so the tests that use real series assets as goldens — in addition to the GPU e2e tests,
**the CPU-only upstream parity tests** (irodori's codec / reference / t-embed, sbv2's rel-pos /
relattn / demo assets, wav's real-asset scale) — are skipped on GitHub Actions because the assets are absent.
These checks are carried by the local / self-hosted `vp test` (with assets), and a release requires
green on real assets + a real GPU (ADR 0005). Moving them to CI would need promoting the goldens to fixtures
(a trade-off with repository bloat) or fetching assets in a release gate — to be revisited in the release-preparation phase.

**`packages/e2e/tests/e2e_assets_gate_test.ts` turns an all-SKIP for missing assets into a FAIL** (an all-skip guard of the same
shape as `gpu_gate_test.ts` for a missing GPU). CI cannot hold the assets, so it sets `KARUME_ALLOW_NO_ASSETS=1`
as a statement of intent and passes. Green from an environment with this env set **cannot be used for a release decision**.

## `karume dist` peaks at about twice the distribution size on disk (the cost of staging→swap)

Assembly builds everything in a staging directory and then puts it in place with rename (ADR 0052 — by-design, so that a failure
midway does not break the existing distribution). The old and new trees coexist until the moment of the swap, so the destination
file system needs free space of about twice the distribution size. After the swap neither `.staging` nor `.old`
remains.

## exporter: per-model recipes are repository-only (only the general-purpose core goes into the wheel)

PyPI `karume` is the general-purpose exporter core only (ADR
[0065](decisions/0065-exporter-core-recipe-split.md) — by-design, to make the distribution boundary coincide with the licence
boundary). The export scripts, dist recipes and card templates of known models are in
`tools/export-recipes/` (uv workspace, outside the wheel), and the dependencies of real-weight export
(diffusers and the like) are held by the dependency-groups of that project. So the installed
`karume dist` has an empty pipeline table and fails loudly (the source of truth of the accepted set is the repository's
`tools/export-recipes/dist.py`). `karume verify` works with the installed version too.

## On Metal (Apple GPU), GPU-side timestamp measurement is not practical (external constraint)

`gpuTiming: true` (ADR 0021) opens 1 dispatch = 1 pass and takes timestamps at the pass boundaries, so
it creates one query set per chunk. On Metal the `createQuerySet` that creates that query set fails and

```
Failed to create counter sample buffer: Cannot allocate sample buffer (MTLCounterErrorDomain)
```

is returned. NSError code 0 = `MTLCounterSampleBufferError.outOfMemory` (the Metal driver cannot allocate the counter
sample buffer); wgpu-hal flattens this to `DeviceError::Unexpected`, and wgpu-core
treats it as fatal and loses the device. **It does not enter an errorScope** (errors of an already lost device are discarded at the entry of Deno's
error handler), so `pushErrorScope('validation')` does not catch it.

The axis of the failure is not "a single sample buffer is too large". The first run is fixed at 16 dispatches per chunk
(there is no wall-time measurement yet, so no time budget can be drawn), so with gemma4's prefill of ≈1,500 dispatches
about 100 query sets are **alive at the same time**. Each has 32 queries (even in total, tens of KB), smaller than the 4,096 samples that Chrome's
Dawn routinely uses on Metal. Moreover, Deno 2.9.x's
`GPUQuerySet.destroy()` is a **no-op** (wgpu 29.0.1 pinned; the implementation landed in wgpu v30), so even when karume
calls `destroy()` the real release is left to V8's GC and piles up. So the earlier attribution "anima's 3,301 dispatches are
too many" was wrong, and it also happens with gemma4, which has fewer dispatches.

"It is visible as `GpuDeviceLostError`, so it is not silent" is also **retracted**. In a script that grabs resources with a top-level
`await using`, the body's exception and the exception at release are folded into a `SuppressedError`, and Deno
prints only its outer shell, so neither the type nor the message can be read. **These 2 points were filled on 2026-09-03** — the example script
(`examples/gemma4/main.ts` (removed in 0.15)) wrapped its body in a function and at the top level recursively expanded `SuppressedError` (`.error` /
`.suppressed`), `AggregateError` and the `cause` chain to stderr in full, and exited 1.
`device.ts` puts the loss reason (`reason` / `message` of `GPUDeviceLostInfo`) into the message of
`GpuDeviceLostError`. However, **on this path the string of the real cause did not arrive** (measured on the M2, 2026-09-03) — what actually came out
was `GpuDeviceLostError: flush 中に device が失われた（再構築が必要） — reason: unknown /
device was lost`, and `message` was only Metal's generic wording. The expectation that a string like
`Device::create_query_set: …` reaches the caller does not hold, at least for this way of losing the device.

Observed on Apple M2 / Deno 2.9.x on 3 paths (anima's DiT 1 step = 3,301 dispatches, gemma4's interactive script, and
the census check of the full verify, `packages/e2e/tests/e2e_gemma4_pretrained_test.ts`). The third path was observed on
**a real macOS 26 machine, full verify of 2026-09-03** (1 red — the adapter enumerates `timestamp-query`, so it is not skipped
even on Metal). There is no record of the macOS version for the first 2 paths, so this version is not applied to all 3 paths.
Since the adapter enumerates it, **`KARUME_ALLOW_NO_TIMESTAMP_QUERY` (the opt-out that allows an all-SKIP on machines that do not enumerate it)
does not fire on Metal** — it fires for software adapters and machines without timestamp-query.

**On Deno today there is no workaround** (the idea of fixing the query set to one and reusing it is untested — see
[known-issues](known-issues.md)). Measurements that need a per-op breakdown are done on Linux / Vulkan machines.
Wall time alone can be measured with timing off (the default). **The reference Mac used for the observations in this section is confirmed
to be macOS 26 (Metal 4) (2026-09-03)**, which falls within the reach of another unfixed issue where timestamps are all zero even when allocation succeeds
([wgpu#9414](https://github.com/gfx-rs/wgpu/issues/9414)) — even with a change that folds the query sets into one,
a per-op breakdown may not become readable on this machine. Note that what is observed on this machine is
the failure of the allocation itself (device lost); the all-zero side is not observed.

For the same reason, the constraint of the section of this file on Deno capping total GPUBuffer allocation at 97% of the driver-reported budget
**does not apply to Metal** — wgpu's `MemoryBudgetThresholds` supports only D3D12 and Vulkan, and
wgpu-hal metal's `check_if_oom()` is a no-op that returns `Ok(())` (with a TODO, [wgpu#7460](https://github.com/gfx-rs/wgpu/issues/7460)).
On Metal exceeding the budget does not raise an exception; it just gets slower and proceeds.

## Unprotected `popErrorScope` waits are not raced against device loss (by-design — with an unverified range)

Some waits on `pushErrorScope` / `popErrorScope` are not connected to the device-loss subscription (`raceDeviceLost`).
Measured on Deno + wgpu / NVIDIA, an in-flight `popErrorScope` settles with `null` even after the loss, and
not a single hang was observed ([research 2026-08-16](research/2026-08-16-device-lost-wait-settlement.md)
— also consistent with the WebGPU spec's "`popErrorScope` of a lost device resolves with null").
**What remains unverified is loss through a real TDR / driver reset, and browsers (Dawn)**. Hitting an implementation that does not settle
would leave finish, use reservations, staging and the section lock stranded, so if a permanent pending is observed there, a wrapper that
races it against the loss is to be added.

## The GPU fit check before download covers the features and limits the quant declares (it does not look at totals or free memory)

Of the GPU prerequisites a quant declares, what is compared before fetching the weight parts (family admission) is
`gpuFeatures` plus the features that need an effective setting (f16 compute → shader-f16 — `sessionGpuFeatures`, all families —
ADR 0111 addendum 2026-09-26) (only on the path given a shared GPU — the path that acquires its own device passes them to `acquireGpu`
as a requirement) and `requiredLimits` (ADR 0089 decision 5, wired on 2026-09-01). The limits are compared against
`GpuContext.limits` for a shared GPU, and for the path that acquires its own, against the adapter values read just before
(`readAdapterLimits` — the adapter is read and discarded). The remaining limits:

- **The WebGPU spec does not guarantee that the adapter read before the download is the same as the adapter that creates the device**
  (`requestAdapter` returns a different object on every call and does not specify that the same physical adapter is chosen).
  The pre-download check is a prior judgement; the final check is the real buffer check at Session construction (ADR 0089 decision 1).
- `requiredLimits` are **the dimensions of the resident part** (only the excess over the defaults — ADR 0089 decision 3). They do not include run-time temporary allocations or
  the worst case of f32 expansion (construction-time and run-time checks carry those). In other words, **even when the field is empty, it does not
  guarantee "it runs on a device with the default spec"** — what it guarantees is only that the resident part (stored tensors and state
  slots) fits within the defaults. Intermediate tensors are not counted because their real demand after fusion depends on the device's
  granted limit (the row-block stride is decided there), so it cannot in principle be determined from the distribution, and baking the
  pre-fusion maximum would wrongly reject "a device that runs with row blocks" before the fetch. A real case whose intermediates exceed the defaults
  = BiRefNet 1024² (1 binding 320MiB, total GPU allocation about 1.6GiB (f32 preset) / about 1.2GiB (HR's f16 preset) —
  the BiRefNet section above).
  An intermediate over the limit (slot actual size > `maxStorageBufferBindingSize` / region > `maxBufferSize`) is caught by the plan-time preflight of ADR
  [0093](decisions/0093-transient-liveness-packing.md) decision 5, which **lists all of them with node names and fails before allocation, both at Session construction and in
  the estimate** (`estimateGraphMemory` — `PreparedModel.estimate` is the same implementation) (wired on 2026-09-05).
- gemma4's `fromAssets` does not take a manifest (`gemma.Assets` is only byte sequences and the config),
  so it cannot reach the declaration, and the only guard on this surface is the construction-time check. `fromAssets` in general has no download, so
  there is no place for a prior judgement at all (only when a shared GPU is passed does admission look at it).
- Totals vs. physical free capacity are not looked at (the section below, "GPU memory pre-checks go up to…").

## Browser: in a hidden tab, generation is pinned to the timer-throttling tick (external constraint, by-design choice)

The abort check placed at the stage boundaries of construction and generation (`settleAbort` in `packages/models/src/concurrency/abort.ts`)
**yields to the event loop once with `setTimeout(0)`**. Chrome's hidden tabs drop timers to 1-second intervals,
and to once a minute when hidden for 5 minutes or more, so processing that hits one timer per stage (one token for generation)
slows down to that tick.

Switching the yield target to `MessageChannel` (not subject to throttling) would restore the speed, but **it is not adopted** — message
tasks can be delivered before a timer set earlier, so the ordering with aborts arriving through a timer (`setTimeout(() =>
abort())` / `AbortSignal.timeout`) is lost, and the check "an abort that arrives after execution starts takes effect at the first stage
boundary" breaks depending on load (measured 2026-09-05). Being able to observe aborts reliably takes priority over speed.

## Browser: Chromium caps a single ArrayBuffer at 2,145,386,496 bytes (applies only to the entry that passes a single-form container in full)

Chromium (Chrome / Edge — common to all OSes, Mac included) always fails the allocation of a single ArrayBuffer over
2³¹ − 2MiB = 2,145,386,496 bytes, as a deliberate security design of PartitionAlloc
(cannot be relaxed with a flag; measured in the 2026-08-25 investigation — the history is in git).

The distributions fit inside this wall. HF's official distributions are split-form containers with a part length ceiling of 1024 MiB, and Session construction
handles a block (32 MiB or less) at a time (ADR [0108](decisions/0108-container-format.md) decisions 5 / 6).
The wall remains only for **the entry that passes a single-form container as a full ArrayBuffer** (`openContainer({ kind: "bytes" })`, the single
key of the full-read surface), and a single-form container over the limit fails loudly with `openContainer` prompting to "read it in split form"
(`MAX_SINGLE_CONTAINER_BYTES` in `packages/runtime/src/format/container/limits.ts`).

Since 2026-08-28 (following fetch-cache 0.5.0 — ADR 0080), this wall fails **before the download**: hub
passes `expectedBytes` (manifest `size`), so a size whose receive-buffer allocation would fail is thrown without receiving
a single byte (`cause` = RangeError — the accident of failing after wasting bandwidth and time is gone).

## Distribution: tensors with one row over the block limit (32 MiB) and consts over 32 MiB cannot be distributed (by-design — ADR 0108 decision 5)

A piece (a split of a tensor) is cut only at boundaries of the leading dimension (rows) — each piece's range is `pieces[].rows` in the binding table, so
reader, writer and checks can share the same row rule (rule ④ of [container-v1](container-v1.md) §5). So the following
shapes cannot be distributed, and export fails loudly (`karume.container`): a tensor where one row (the bytes of one entry of the leading dimension) exceeds
the block limit (e.g. an f32 of shape `[2, 2^24]` — 64 MiB per row), an initializer with `rowAxis` 1 over 32 MiB
(the scale's row range does not correspond to the piece's row range, so it cannot be split), and a const over 32 MiB (consts have no
piece mechanism — container-v1 §10). A quantized initializer whose scale held in 1 block exceeds 32 MiB cannot be
distributed either. What rejects this is not the writer but the block length check when publish re-reads the container it wrote
(`read_container` in `karume.container`), and both the recipes and `karume migrate` go through publish.
No real asset is affected.

## Distribution: the old distribution form (one safetensors file / a list of shards, manifest `karume/4` or earlier) cannot be read — migrate with `karume migrate` (`--manifest` for a whole repository for `karume/4`, positional arguments per component before that — ADR 0108 decision 18)

The current packages read only manifest `karume/5` and containers (`krm` / `krg`), with no dual reading. An old manifest
is rejected by hub as "unsupported major" (`packages/hub/src/manifest.ts`), and passing an old safetensors as a container makes the runtime
fail with "unknown magic" (`packages/runtime/src/format/container/header.ts`). The processing that reads old formats lives only in the migration CLI
(`uv run karume migrate` of `tools/exporter` — positional arguments per component, or the whole-repository mode of `--manifest`),
which only reads its input and does not delete it. Old package versions pin old revisions (40-digit SHA), so as long as the old versions are used,
the old distribution form keeps working. All 10 public HF repositories have been re-uploaded as `karume/5` (the 0.13.0 re-upload,
2026-09-24), and the pins of the current packages point to those revisions.

## hub: the cache is not isolated by credential (by-design — ruling of 2026-08-28)

The key of the asset cache is the content key (without `Authorization`), and the namespace is also the fetch layer's single fixed
one. A copy of a gated asset fetched with authentication is also hit by unauthenticated calls from the same device and the same
origin. There are no plans to operate gated assets, and per-credential namespace isolation was excessive defence, so it was
removed (ADR [0080](decisions/0080-hub-fetch-cache-050.md) decision 3).

If gated operation starts in the future, **a custom `caches` wrapper alone does not give isolation**. The fetch layer
joins concurrent fetches in the process by the content key alone (single-flight), and gives a joining later fetch the bytes of the fetch
that started first — the `caches`, authentication header and `fetch` passed by the later one are not used (by-design in the fetch layer
`@hdae/fetch-cache` 0.8.0). If the same file is requested without authentication while an authenticated fetch is running,
the authenticated bytes arrive even with separate caches. Isolation needs either joining keys separated per authentication
(a change to the fetch layer) or separate execution environments (origin, process) (ADR 0080 addendum
2026-10-10).

## hub: a local source does not verify the sha256 of whole files (it verifies the container descriptor and the blocks read)

When a local directory is the source via `localDirectory` / `nodeDirectory` (`@karume/hub/node`) / `denoDirectory` (`@karume/hub/deno`),
the host (hub) **does not verify the sha256 of whole files** (it trusts the record). The scope of verification splits into
3 by how the file is read:

1. **Files of the manifest's `assets` read in full** (tokenizer and the like — `fetchAssets`): the only check is
   exact equality with the `size` that `karume.json` declares. A copy truncated midway, a mix-up with a file of another quant / another model,
   and a missing file (if it cannot be read, it fails loudly naming the actual path) are detected; a rewrite that does not change the byte count
   (tampering, silent bit rot on disk) is not detected. A range read of `openAsset` looks only at the
   boundary of the declared `size`.
2. **The descriptor of a container (a `krm` part) and the blocks read**: a local source is treated as an unverified source
   (`verified` in `packages/hub/src/container.ts`), the runtime's `openContainer`
   verifies the 2 descriptor documents against the manifest's expected values (length + sha256), and verifies the sha256 of every block
   it reads (`readBlock` in `packages/runtime/src/format/container/open.ts`). Tampering with a block that is read
   is detected here.
3. **Not verified**: the regions of a container that are not read, and, for any file, the hash of the
   whole file.

The by-design reason is the difference between fetched data and assets: it rests on the premise that **a local distribution is managed by the user**
(not bytes fetched from the distributor). There is no threat worth hashing everything on every start (3.8GiB for gemma4 E2B),
and it would buy back locally the cost that ADR [0080](decisions/0080-hub-fetch-cache-050.md) decision 1 removed from the network
path. `size` is known for free once reading finishes, so it stays as a check.

**For operation that needs tamper detection of whole files, use the HF path (`HubRepoRef`)** — it verifies the sha256 of the
whole file at fetch time and burns in a record, and a mismatch with the record becomes an automatic evict + re-fetch (self-heal)
(silent bit rot after storage is not detected — ADR [0080](decisions/0080-hub-fetch-cache-050.md)
decision 1). The design's source of truth is ADR [0086](decisions/0086-distribution-source.md) decision 2 and its addenda.

## The sha256 reference check is for the reference environment only — cross-device bit identity is not guaranteed

> **Since 2026-09-20, "reference environment" = an environment whose row is in the reference value fixture** (ADR
> [0106](decisions/0106-device-keyed-references.md)). Reference values are no longer constants recorded on a single machine but
> **rows per environment key**. The section title is kept as is to preserve existing pointers (comments on the code side
> refer to it by name).

The PNG / WAV reference sha256 of the e2e tests (`e2e_anima_test` / `e2e_sbv2_wav_test` / `e2e_irodori_wav_test`) are
**values recorded in that environment**, stored in `packages/models/tests/fixtures/references/<series>.json` as rows per environment key
(`<runtime>-<adapter name slug>`, e.g. `deno-intel-graphics-bmg-g21`).
**An environment without a row SKIPs explicitly without comparing** (Metal and the like) — this is the specification, and the check
works as "a port and regression detector **on that machine**". So that a SKIP does not become a silent green, if even one registered
case lacks a row for the environment, **the reference guard of that series turns red** (an id without a row can be stated explicitly through the caller's held list — ADR 0106 addendum
2026-10-03; opt-out with `KARUME_ALLOW_NO_REFERENCE`).
**Even on the same Linux / Vulkan, different vendors do not match** (2026-09-20, full verify on an Intel Arc
B570 / Mesa ANV: against the then single constant, all 16 mismatched; the output PNGs looked normal =
tiny numeric differences). This measurement was the direct motivation for moving to rows per environment, and the B570 row
(`deno-intel-graphics-bmg-g21`) has been created. How rows are made, the reference guard (opt-out with `KARUME_ALLOW_NO_REFERENCE`) and the results
directory (`outputs/verify/`) are defined by ADR [0106](decisions/0106-device-keyed-references.md).

9 series have rows (2026-09-26 — ADR 0106 third addendum): besides the PNG / WAV of anima / sbv2 / irodori, the goldens of siglip2 /
gemma4 (runtime-side e2e — rows in `packages/runtime/tests/fixtures/references/<series>-golden.json`),
the real images of birefnet / depth-anything, the f32 output of the whole chain of vowel-detector (safetensors), and the
token id sequences (JSON) of the quant presets of gemma4 / gemma4-qat. The rows of anima, whose artifact is a PNG, are the sha of the pixels before compression (RGB8); the bytes of the PNG
(the IDAT deflate depends on the runtime's zlib implementation) are not in the rows (2026-10-17, ADR 0106 addendum).

Mechanism: IEEE 754 addition, subtraction, multiplication and division are exactly identical across devices too, but because of ① transcendental function (`exp` and the like) implementations
depending on the driver / compiler ② the shader compiler's fma fusion decisions (fusing a multiply-add into one instruction
removes one rounding) ③ differences in the compilation path (Tint in browsers / naga on Deno), cross-device identity does not hold
even when the kernels fix the reduction order. In addition, the i8a8 path of attention has a separate difference on Metal of
**exactly 1 ULP from the TS reference** (the integer stages match exactly; what drifts is the f32 epilogue shared by both
the dp4a / emulated variants — the Metal section of [known-issues.md](known-issues.md)).

What is guaranteed is the following 2 (both with measured data points on Vulkan and Metal — the Apple M2 measurements are
[research/2026-08-10-f32-geometry-probe.md](research/2026-08-10-f32-geometry-probe.md)
§Apple M2):

- **In-device determinism**: same key → byte-identical WGSL → identical output (shown on the M2 by matching output sha across 2 independent
  sessions).
- **Bit invariance under geometry changes**: tile geometry changes only the assignment of work (shown on the M2 by matching output sha across 2
  geometries).

Soundness checks on other backends are done not by matching the reference sha but by **self A/B** (whether the output sha of the same input with 2 geometries
or with 2 versions, old and new, match each other).

**The same principle applies on the CPU side**: the byte-equality check of the tiny golden io (torch CPU expected outputs)
(the regenerate-and-compare of `tools/exporter/tests/test_goldens.py`) is also for the reference environment only — oneDNN dispatches gemm / conv kernels
by the CPU's ISA, so the last bit of the results depends on the machine
(measured 2026-08-16: on the GitHub CI runner, only the 2 specs activations / conv2d_block of 30 specs differed by
±1–2 ulp). CI SKIPs the io byte comparison explicitly and requires only the byte comparison of the model (graph + weights from a fixed seed —
torch's CPU RNG is deterministic across machines). The TS-side real-GPU tests
that consume the goldens are tolerance-based to begin with, so they are unaffected. Reference environment = the development machine. When the CPU changes, re-record the io without
loosening the check (re-recorded on 2026-10-06 after switching to a Ryzen 5 7600 — the difference from the old reference was 2 specs, `activations` /
`dilated_conv`, absolute difference ≤ 2.4e-7).

The fixture of BiRefNet's α-matte post-processing (the regenerate-and-compare of `birefnet/tests/test_matte_fixture.py`) is treated
the same way: the last bit of the expected values of lucida (torch's CPU f32 σ and resize) changes with the SIMD kernel torch chooses
(measured 2026-10-15: on AVX2, 3 elements of `disc.up.lucida` with a maximum of 7.5e-9; without SIMD, 40 cases with a maximum of
4.6e-5 — no effect on the TS side's ε of 1e-2). CI SKIPs only the byte comparison of lucida explicitly, and requires byte equality of the JSON and hr
(the Pillow path — measured to match on 3 instruction sets) in every environment.

## states-form attention is f32-path only (cannot be combined with numeric variants); sliding rewind is always rejected

Deliberate constraints of attention with the `states` field / `state_append` (ADR
[0067](decisions/0067-autoregressive-attention-vocabulary.md) decisions 4 / 5):

- **`attentionCompute: 'a8' / 'f16'` × the states form fails loudly** (the kernels of the other families are f32 only —
  rejected without fallback, ADR 0058 decision 3. An opt-in option is to be raised in the ADR 0058 manner when there is real demand).
  **`attentionScoreStorage: 'f16'` × the states form likewise** (S is stored in f32 only).
- **`rewind` of a GenerationContext that contains even one sliding slot is always rejected** (ADR
  [0066](decisions/0066-generation-context-state-slots.md) addendum 2 — after eviction the ring's physical layout and
  logical range no longer coincide, and it has no left-packing compaction. Relaxing it goes together with implementing compaction).
- `enqueue.generation` keeps the context's use reservation until the batch's final settlement (added 2026-09-12;
  [ADR 0066](decisions/0066-generation-context-state-slots.md#addendum-2026-09-12--generationcontext-in-batched-execution)).
  Queuing several unsettled executions of the same context, and disposing a reserved context / its owning Session, are rejected.
  Proceed to the next execution or disposal after `finish` / `finishAndRead`. The dispose-wait contract of normal runs is unchanged.
- The dtype of state slots is f32 only (f16 is reserved — ADR 0066 addendum 5); multiple sequences /
  batch>1 generation and paged KV are outside the scope of ADR 0066 decision 8.

## gemma4: capacity / chunkLength are run-time knobs — changing their values can move the token sequence

The distribution's `pipelineConfig.capacity` / `chunkLength` are **defaults**, and the caller can override them with
`gemma.Pipeline.sequence({ capacity })` / the `capacity` of `gemma.ChatSession` (1 conversation = 1 capacity) and
`gemma.PipelineOptions.chunkLength` (per pipeline) (ADR
[0091](decisions/0091-gemma4-host-rope-variable-capacity.md) decision 4). There are 2 checks —
`chunkLength ≤ maxChunkLength` (the declaration of the trace limit of the symbol `M`; 768 for E2B) and
`chunkLength ≤ capacity ≤ maxPosition` (the declaration of the exclusive upper limit of positions; 131,072 for E2B).
`gemma.ChatSession` checks the relation of these 2 **at construction too** (this layer uses `capacity` as the yardstick for
overflow before the sequence allocates, so leaving it to the sequence side alone would truncate the history on a long turn
and then give a capacity error, hiding the real cause).
VRAM grows in proportion to the capacity (full-layer KV
12,288 B/token + the temporary S of states-form attention = `8 × row-block rows × capacity × 4 B`), so
`gemma.Pipeline.estimateSessionMemory({ capacity, chunkLength })` can estimate it before allocation (only the
required total — it does not compare with the free side).

- **Changing `chunkLength` can move the greedy token sequence**: the row statistics of the sliding layers assign the columns of S to 256
  lanes with an origin that depends on `past`, so when the chunk stride (= the sequence of `past` values) changes, the way the f32
  partial sums are folded changes (the full layers and linear are unchanged). Measured (RTX 3080 Ti, P=4,096, 24 tokens),
  32 / 256 / 512 / 768 matched, but a step with a small margin can flip. Acceptance goldens are taken with the distribution's
  declared values.
- **`capacity` does not affect the token sequence** (the work is cut by the logical length, and the values do not depend on the capacity — with a bit-identity check).
- **The reduction form of decode attention is decided by the quant preset's declaration** (on 2026-09-26 the models code default `"parallel"` was removed
  and promoted to the manifest vocabulary — [ADR 0104 addendum](decisions/0104-gemma-fast-quant.md)): `i4-gemvpar` and QAT E2B's `i4-fast` use
  the parallel reduction `"parallel"` (KV-parallel ③PV = perf-ledger K-12 (plans with M < 16), D-parallel ①QK = K-14 (plans with M ≤ 8 —
  decode and speculative verify; widened from M=1 on 2026-09-09)), regular gemma4's `i4-fast` uses `"parallel-fused"` (fuses the row statistics
  and PV in the same addition order — the "row-statistics/PV fusion" section below), and `i4` without a declaration and `fromAssets` use the runtime's reference path `"sequential"`.
  Prefill plans (M ≥ 16) use the tiled paths of the GEMM skeleton
  ①ₜ / ③ₜ regardless of the preset (K-13, bit-identical to the reference path — [research 2026-09-06](research/2026-09-06-state-attention-tiled-k13.md)). The band of ①′ is a difference from ①
  of 3.58e-7 / a difference from the f64 reference of 4.17e-7 (band 5e-6, [research 2026-09-06](research/2026-09-06-state-qk-parallel-k14.md)).
  About ③′: its reduction order differs from the runtime's reference path `"sequential"`
  (A/B band 5e-6; measured difference from ③ 2.4e-7, difference from the f64 reference 3.99e-7), and gemma4's golden is identical for both.
  The measured range of the band is reduction lengths live 1–16,384 (the check in `gpu_state_attention_parallel_test.ts` — the worst value appears on the side
  of small live) + 65,536 (measured by hand): at live 4,096 / 16,384 / 65,536, ③' vs f64 = 8.6e-9 / 1.5e-8 /
  1.8e-8 (1/100 of the band, measured in the 2026-09-03 review).
  `gemma.PipelineOptions.stateAttentionReduce: "sequential"` (or quant `i4`) returns to the reference path.
  The default of the low-level surface (consumers who call `createSessionFromContainer` / `createContainerSession` themselves) stays the runtime's `"sequential"`.
- The upper limit of `chunkLength` is the trace limit of the baked symbol `M`, which the distribution declares as `pipelineConfig.maxChunkLength`
  (768 for E2B). A larger value fails at the declaration check (measured 2026-09-03: before the declaration existed,
  `chunkLength: 1024` passed silently, the prefill was split into 2 chunks, and the token sequence was identical to 768).
- **Breaking change on an unpublished surface**: `gemma.PipelineConfig` gained `maxChunkLength` as a **required field**.
  A caller that writes the `config` passed to `fromAssets` by hand needs to add 1 field (the distribution mirror declares it through a
  re-issue by `dist.py`; the weight shards' sha is unchanged).
- It is not bit-identical to the upstream RoPE table (TS computes in f64 and rounds to f32 — upstream is f32 on every path).
  The difference is proportional to the position; the maximum over a full sweep of positions 0..131,071 is 9.4e-3 (the tolerance band is likewise proportional to the position, 1.57e-2 at the top
  position 131,071; the worst ratio to the band over all positions is 0.76). The goldens were re-taken with this table (decision 2 of the same ADR).

## Generation sequence: truncating the conversation is the host's responsibility on the low-level surface (exceeding capacity fails with a dedicated type)

`GenerationSequence` (`packages/models/src/generation/sequence.ts`, ADR
[0083](decisions/0083-generation-api-surface.md) decision 10, revised in the addendum 2026-09-02 of the same ADR) does **not truncate**
the conversation **automatically**.
A turn that exceeds either of 2 limits — the capacity of the full slots that the conversation allocated (`pastLength + queryLength ≤ C` — ADR
[0067](decisions/0067-autoregressive-attention-vocabulary.md) decision 4 ④; a run-time knob chosen when the sequence is created, defaulting to
the distribution's declared `capacity`) and the exclusive upper limit of positions the model declares (`maxPosition` —
ADR [0091](decisions/0091-gemma4-host-rope-variable-capacity.md)) — fails with `GenerationCapacityError` before
a single run is issued. Both use the same type because for the caller they are the same fact, "this
conversation no longer fits", and the remedy is the same.

The remedy is **on the host side**: drop old turns and replay the token transcript into a new context
(`rewind` is always rejected for a context containing sliding slots — ADR
[0066](decisions/0066-generation-context-state-slots.md) addendum 2), shorten the prompt, or
lower `maxNewTokens`. The runtime side also rejects the same overflow, but that is a generic message just before encoding the run,
and one would have to tell from the wording whether "truncating would let it pass" or "the wiring is broken" —
the dedicated type separates out that one case.

**This constraint applies only to the low-level surface (`gemma.Pipeline.sequence()` / `chat()`)**. The high-level
surface `gemma.ChatSession` **has the same remedy as its default policy** — **before** sending each turn it compares
`used + prompt + maxNewTokens - 1` with the limit, and if it exceeds it, rebuilds the history with `onOverflow` (default
`dropOldestTurns` = drop the oldest user / assistant pair, keep system) before
firing. The policy can be replaced wholesale, and passing a function that throws lets the host side handle it as before. When there is nothing
left to drop (and when the policy did not shrink the history), it fails with the same `GenerationCapacityError`.
**There is no compact that has the model summarize** (to be reconsidered after the window is widened — ADR 0083 addendum
2026-09-02).

There are 2 constraints on the same "host's responsibility" line (the session absorbs the first on the conversation side — an aborted
turn keeps only the text that came out in the history, and a sequence whose correspondence with the KV cannot be read is discarded. The second also applies to the session's
`send` with the same discipline):

- **An abort closes in "the state where the conversation advanced only by the successful runs"** (`break` / `return()` /
  `AbortSignal` alike). An abort before any token was received (= in the middle of prefill) leaves the prompt
  partway into the conversation, and the `chunk` of the `prefill` event gives the number of committed chunks. Whether to send the rest
  or discard the sequence is up to the caller.
  - **Even an abort that arrives after the final prefill delivers 1 token** (aborts are checked only at stage boundaries before a run is issued
    — ADR 0083 decision 5). The final prefill run has already returned logits, so even when aborted after that `prefill`
    event, the first token is sampled from the computed logits and delivered, and no new run
    runs. With `maxNewTokens: 1` it completes normally with `max-tokens` as is; if there is more, it becomes `aborted` before the next
    decode run (pinned by `generation_sequence_test.ts`).
- **A request runs on a snapshot taken at issue time** (`generate` / `gemma.Pipeline.chat` — ADR 0083 addendum
  2026-09-02). The `prompt` array, `maxNewTokens` / `signal`, and the sampler specification (down to the contents of `logitBias`) are
  copied in the synchronous part of issuing, so rewriting the request object after starting to draw from the returned sequence **does not change
  the generation in progress** — to affect the next turn, pass it as the next request. An abort takes effect only when a `signal` is
  passed at issue time (a signal plugged in later is not read).
- **The history that the repetition penalty looks at is the current turn only** (that turn's prompt + the generated tokens).
  The sequence does not hold the token transcript of the whole conversation (its mutable state is only the context and `pendingToken`
  — ADR 0083 decision 1), so the tokens of past turns are not penalised. To apply it to the whole
  conversation, including the range to be affected in that turn's `prompt` is currently the only way.

## Conversation fork (`fork`): each branch adds a whole KV; matching guarantees follow the numeric path's checks

`GenerationContext.fork` / `GenerationSequence.fork` / `gemma.ChatSession.fork` (ADR
[0066](decisions/0066-generation-context-state-slots.md) and
[0083](decisions/0083-generation-api-surface.md) addenda 2026-10-11, ruling 8) duplicate the KV of the same capacity
wholesale and return an independent continuation. Deliberate constraints and properties:

- **VRAM grows by one state per branch** (copies with reduced capacity, delta-only copies and prefix
  sharing are not provided — still outside the scope of ADR 0066 decision 8). The exact value per branch is the estimate
  (`resident.stateBytes` of `estimateSessionMemory`), and the number kept alive at once × that value is the caller's
  accounting. Neither the estimate nor admission knows the number of branches, and they do not compare with free VRAM, so a shortage
  is first noticed when the copy's allocation throws `GpuOutOfMemoryError` (the original stays intact).
  gemma4 E2B's full layers alone take 12,288 B/token, so at the default capacity of 4,096 the full-layer part is
  48 MiB, and at capacity 131,072 it is 1,536 MiB (values derived by multiplication — the sliding layers' part comes on top).
  A speculative borrowing context adds only the 8 bytes of the logical-length uniform per branch.
- **There are states that cannot be forked**: a run in progress (including while a borrower's run is running), during a batch's
  use reservation, while waiting for a deferred commit to settle, a borrowing context, after dispose / contamination / device loss, and
  a disposed Session. `GenerationSequence.fork` and `gemma.ChatSession.fork` throw synchronously when disposed
  (ChatSession also during a running turn), and a sequence's fork runs **after** the generation in progress
  (awaiting `fork()` without closing a stream that is being drained never returns).
- **Whether the token sequence after a fork matches "the same sequence of operations without fork" follows the determinism of the numeric path**.
  The copy duplicates the KV bytes wholesale, so the branch point is bit-identical, but matching afterwards is promised only
  without speculation and with `speculative: "always"`, and on devices where speculative check ⑤ is green with the self-tuning gate
  (the gate's observed state is copied to the branch, but the wall time fluctuates per branch — devices where the check is red are
  treated as in the speculative section above).
- **The GPU time of the copy does not show up in the scheduler's statistics (`submitCount`, time budget, GPU time breakdown)**.
  The copy is outside the Session's serialization chain and is submitted with its own encoder in the synchronous section that called `fork()`
  (in exchange for not waiting for long runs of other contexts of the same Session). A copy with a huge capacity can mix into the timing window of
  another run on the same device (not measured).
- **Editing (rewriting an utterance midway) and changing capacity cannot be expressed by forking** — as before, stream the history into a new
  sequence again (`rewind` of a context containing sliding slots stays always rejected).

## The strictness of cross-implementation parity checks is bounded by the trajectory's error amplification (by-design)

The "final-output comparison with the reference implementation" of diffusion pipelines (Irodori and others) compares values after
the iteration steps have accumulated and amplified floating-point operation-order differences (~1e-7 class, structurally unavoidable) by **a factor that depends on
model × input × storage rounding**. This factor swings by measurement from hundreds to tens of thousands (37,107× for case.full of v4.1-small f16
— [research](research/2026-09-01-irodori-v41-euler-sensitivity.md)), so on trajectories with
strong amplification a fixed strict tolerance cannot hold in principle. Irodori's full-loop check absorbs this with a two-stage judgement
(fixed 1e-3 → when exceeded, normalised by the measured amplification + an absolute upper limit) — the permanent logic is
`tools/export-recipes/irodori/pipeline_ref.py`. Bit identity of the distributed artifacts (same bytes + same implementation =
same output) holds regardless of this constraint.

## GPU memory pre-checks go up to "the device's absolute limits" — totals vs. physical free capacity are not checked (by-design)

What the pre-checks introduced in the memory-management phase (ruling of 2026-09-01) can see deterministically is **up to comparing each buffer's dimensions with
the device's absolute limits (`maxBufferSize` / `maxStorageBufferBindingSize`)**. Whether the **total** of allocations
exceeds physical memory is not checked — WebGPU does not expose the total or free capacity (ADR 0070
decision 5's "do not guess in environments where the budget cannot be obtained"), and a guessed threshold creates false rejections in healthy environments.
The final detection of exceeding the total remains the out-of-memory errorScope, and **on Metal there is a known environment where it is
silent** (the "does not apply to Metal" note above — wgpu-hal metal's `check_if_oom()` is a no-op;
the known-issues section on Metal's silent out-of-memory errorScope). So on Metal,
"single limits reliably fail in advance; physical overruns of the total can still proceed silently" remains. A mitigation candidate is passing the
pre-estimate of the requirement (`PreparedModel.estimate` / `estimateGraphMemory`, and `gemma.Pipeline.estimateSessionMemory` for gemma4)
to the caller, but the estimate needs every symbolic dimension of the graph inputs to be
bound, and gemma4 is the only family that has the bound values at load time, so the design of that option comes after the measurements of Phase B
(ADR 0089 addendum 2026-09-01 — not implemented).

## hub: cache inventory queries and per-selection deletion count only within a single manifest (by-design — ruling of 2026-09-05)

The reference accounting of `listCachedAssets` / `evictCachedAssets` (ADR
[0094](decisions/0094-hub-cache-inventory-and-eviction.md)) counts only the (model, quant) pairs of the single `LoadedManifest`
passed.

- **Cross-repository references are not deleted from the referring side** (`kept: "cross-repo"`). Conversely, deleting a selection in the referenced repo
  makes the selection of another repo that borrows it (e.g. anima-extra referring to anima's text stack) fall back to partial inventory
  — hub does not know the referring manifest, so it cannot protect it. Nothing breaks: the next load re-fetches only what is
  missing (the cache is an optimization, not a requirement for correctness).
- **Placing the same path with the same sha256 in another revision of the same cross-repository target gives 2 references but 1 cache
  entity** (confirmed 2026-10-11). Reference identity (`fileRefKey`) includes the revision, but HF's cache
  entity key (`[hf, model, repo, path, sha256]`) does not. This shape passes the manifest parse (the parse
  rejects only when the cross-repository target is the loaded repo itself — ADR [0038](decisions/0038-manifest-v1.md) addendum
  2026-10-11). Today's deletion does not delete cross-repository references from the referring side, so the protection and collateral (`alsoEvicted`) decisions
  do not drift with this shape. The only consequence today is on the fetch side: the same bytes are fetched as 2 separate entries (fetched at once, they may be
  downloaded twice). When matching is moved to the entity key, or deletion is changed to also delete cross-repository references, the same drift as the shape that points to
  another revision of one's own repository across repositories (deleting one deletes the other's entity) appears, so decide how this shape is treated
  first (the later section of the backlog).
- **A selection with partial inventory does not protect shared files**. Only selections whose references are all in inventory are protected (to match the selections an app
  shows as "downloaded").
- **Selections whose target and reference set are identical (same weights, different session settings only — quants) are not protected by default**
  (ADR 0094 addendum 2026-09-06). The cache's granularity cannot tell them apart, so deleting the target makes the siblings fall to partial inventory
  — which ones fell is named by `alsoEvicted`. An app that also wants to protect the siblings names them in `protect`
  (when given, only the list is candidates, and same-set exclusion is not applied).
- **A selection narrowed by role with `weights` does not share accounting with the rest of the same (model, quant)** (ADR
  [0096](decisions/0096-speculative-decoding.md) `ResolveOptions.weights`, added at stage 2). The protecting
  candidates are "selections whose label (`<model>/<quant>`) differs from the target", so another subset with the same label is not protected and
  does not appear in `alsoEvicted` either. 3 consequences:
  - **"Keep the main model, delete only the drafter" cannot be written as is**. `resolveSelection` expands all
    assets even when weights are narrowed, so the reference set of the narrowed selection includes common assets such as the tokenizer, and
    **only the main weights remain** (that (model, quant) falls to partial inventory — details and the workaround are in
    the relevant section of [known-issues](known-issues.md)).
  - **`protect` with the same label as the target is ignored** (because the protecting candidates are only "selections with a different label").
    `inventory_test.ts` pins "mixing the target itself into `protect` is ignored".
  - **What listing several subsets in `protect` can protect is only selections whose label differs from the target**. Whether the rest of the same label
    fell to partial inventory is seen by calling `listCachedAssets` again (there is no field in the label that names the subset).
- **A local source answers "everything in inventory" and refuses deletion with `HubError`**. A missing entity fails when it is
  read (the I/O of scanning the directory on every query is not paid). The directory's contents are not fetched data but
  the user's assets, so there is nothing hub may delete.
- **In an environment without `CacheStorage`, every source with a cache (HF) is `missing` and deletion is empty**
  (as the fetch layer's contract says). A local source does not go through the cache, as above, so it answers "everything in inventory" regardless of whether
  `CacheStorage` exists. On runtimes without `Cache.keys()` (Deno 2.8 and earlier) the
  fetch layer throws, failing loudly.

## hub: the receive limit of the HF source is the fetch layer's `expectedBytes` (only `karume.json` is judged after full receipt; content-length is not checked in advance — 2026-09-06)

Following fetch-cache 0.7.0 (ADR [0094](decisions/0094-hub-cache-inventory-and-eviction.md) decision 4),
hub's own received-byte check (the `fetch` wrapper) was removed. The checks on byte counts for the HF source are as follows:

- **Assets**: the fetch layer HF layer's `expectedBytes` (the manifest's `size`) is the receive limit, and receipt is cut off as soon as the declaration is
  exceeded. On the full-read surface (`fetchAssets`), exact equality after full receipt and the sha256 verification are as before. Either
  mismatch surfaces from hub as `HubFetchError` (with the fetch layer's error in `cause`) — only the local source throws
  `IntegrityError`.
- **The warm-up surface (`prefetchAssets`, and the warm-up that `openAsset` / the container surface insert for references not in inventory) does not detect shortfalls**:
  the fetch layer's prefetch uses `expectedBytes` only as the receive limit. An entity that "has the right sha256 but is shorter than the declared `size`"
  (a defective manifest on the writer's side) warms up successfully, and the progress reports complete at the declared `size`.
  What goes into the cache is the correct bytes per the sha256, and reading the part of the declared range outside the entity makes the later
  read (a container block, a range of `openAsset`) fail with an error worded differently from the warm-up (no silently zero-filled values
  are returned). Fetching the same reference on the full-read surface gives `HubFetchError`. Catching shortfalls on the warm-up surface needs a change in which the fetch layer
  checks at the end of receipt that the received length equals `expectedBytes` (symmetric with cutting off excess), and presupposes agreement with
  the fetch layer.
- **`karume.json`**: neither an expected sha256 nor a size can be known in advance, so the 1 MiB limit is checked by `parseManifest` **after full receipt**
  (the fetch layer has no "limit without exact equality"). Against a mirror that returns a response over the limit,
  the memory to receive that response in full is paid.
- **content-length is not checked in advance** (by-design): under the Fetch spec Content-Length cannot be trusted, and
  through Content-Encoding a mismatch between the declaration and the actual receipt is normal, so putting it in a general-purpose fetch layer would give false positives.
  A size mismatch is caught on receipt (an excess midway on every surface; a shortfall only on the full-read surface after full receipt — the warm-up surface's
  shortfall is the item above).

## Gemma 4 QAT: a fixed mobile format at the experimental stage

`gemma4-qat` is a separate family from regular `gemma4` and handles only the text generation of the official mobile Transformers format E2B / E4B.
It keeps fixed INT2 / INT4 / INT8, SRQ and packed PLE. MTP, vision, audio and a public source pin are not supported.
The local usage example is the QAT CLI (`examples/gemma4-qat/README.md` (removed in 0.15)), the numeric contract is [ADR 0097](decisions/0097-gemma4-qat-integration.md),
the terms are in the [glossary](glossary.md), and the source of the measurements is the [measurement record of the QAT review](research/2026-09-19-qat-review.md).

**Activations are "only rounded", not the official mobile integer dot product**.
The official mobile runtime passes the int8 rounded by SRQ as is to the integer dot product (int32 accumulation), but karume, like upstream
transformers, restores them to f32 and reduces in float ([ADR 0097](decisions/0097-gemma4-qat-integration.md) addenda 1 and 8).
This is the root of the token sequence diverging between CPU / GPU: when a small difference in reduction order crosses an SRQ rounding boundary, the sequence splits.
For the same reason, changing the chunk length can also change the sequence (there is a real case in the short-text comparison of E4B). "It changed = a bug" does not hold.
Implementing the integer dot product is outside this scope and was filed as [perf-ledger](perf-ledger.md) K-45.

**The KV cache is f32 and does not use the official quantization scales**.
The official checkpoint holds `k_cache_scale` / `v_cache_scale` per layer (70 in total for text), but
karume, like upstream transformers, does not read them.
The full-layer KV per position is 12,288 B for E2B / 32,768 B for E4B, and the sliding layers are fixed buffers with a window of 512, so they
do not depend on capacity. int8 conversion was filed as [perf-ledger](perf-ledger.md) K-46.

**The defaults are capacity 4096, chunkLength 768, trace limit 768, and `--max-new-tokens` 256 for the interactive CLI (removed in 0.15)** (the same as regular Gemma).
The prefill buckets also use the same ladder as regular Gemma (not a sequence re-measured specifically for QAT).
**Output quality in contexts over 512 is not accepted**, and this is the only substantive precondition when using up the capacity.
That it runs at all is confirmed: a run at capacity 8192 is in the
[prefill bucket investigation](research/2026-09-13-prefill-buckets.md) (though the longest input is 65 tokens), and
generation of length 1,024 is in the [speed bench of the WebML comparison](research/2026-09-12-webml-browser-speed.md).
Neither looked at quality. The short-text comparison where RTX Deno / Chrome matched is not a proof of broad quality or of equivalence on the M2 either.

**Thinking mode is not supported**. There is no path equivalent to `enable_thinking` of the official `chat_template.jinja`, and
the conversation's `reasoning` field is rejected, failing loudly (within the reach of [ADR 0084](decisions/0084-gemma-tokenizer-chat.md) decision 5).

**Not all 1,179 tensors of the checkpoint are used** (vision_tower 546, audio_tower 631,
embed_vision 1, embed_audio 1). This count is a guide to the size of the phase that takes on vision / audio.

## Parallel addition in GEMV (2026-09-12)

`linearGemvReduce: "parallel"` is an optional setting of [ADR 0098](decisions/0098-linear-gemv-parallel.md).
It covers the measured quantized matrices and physical M=1..8, with f32 arithmetic only. It is not bit-identical to the default sequential addition, and for QAT the generated sequence can change too.
M>8 and shapes outside the targets use the former path, and the diagnostic keys show the range it applies to. It does not mean an across-the-board speed-up of E4B or other models.
The table of target shapes is an allowlist, and shapes not on it silently fall back to sequential addition even when `parallel` is specified. The E4B rows (2026-09-26) are an unmeasured provisional copy of the lanes of the E2B rows of the same role, and QAT E4B's i2 lm_head is not on it ([ADR 0098 addendum 2026-09-26](decisions/0098-linear-gemv-parallel.md)).
[The speed improvement and short-text output on the M2](research/2026-09-13-m2-gemv-adoption.md) are confirmed. A broad quality evaluation remains.
The new distribution recipes of regular E2B and QAT E2B / E4B choose as the default quant `i4-fast`, which declares parallel and RMS→add fusion (QAT additionally linear→SRQ fusion and packed int8 activations) ([ADR 0104](decisions/0104-gemma-fast-quant.md)). The parallel-only `i4-gemvpar` is also kept.
The sequential default of the former `i4`, the runtime and fromAssets is kept. Existing distributions and public pins are not changed automatically.
The shapes of gemma4 E2B's drafter (i8, `[256,3072]` / `[262144,256]` and so on) are not in the table, and even with `i4-fast` 66 of its 68 linears run sequentially (the bundle's census table pins 66 for the reference — perf-ledger K-59).

## packed int8 activations (`packedStaticQuantize`, 2026-09-19)

`packedStaticQuantize` is an optional setting of [ADR 0105](decisions/0105-packed-static-quantize-activations.md) that packs 4 int8 codes produced by fixed SRQ into one u32 word and passes them to the parallel GEMV.
QAT E2B's `i4-fast` declares it by default (regular Gemma 4 has no `static_quantize`, so it does not declare it). 4 by-design constraints:

- **Only the 4 shapes where it measurably helped, and the 3 E4B shapes copied from them, become packed** (rows of `PARALLEL_SHAPES` whose `packedActivations` is true = shapes with a long K and lanes 32. The 3 E4B shapes are not measured). An SRQ whose consumers include even one row that is not true stays f32, and the other 40 rows (including the 23 g32 rows) are outside the targets.
- **The handling of non-finite values differs from the f32 path**. `NaN` saturates to ±127 / -128 as outside the boundary table (the f32 path passes NaN through as is), and `-0.0` becomes code 0 = `+0.0`. `±Inf` saturates with the same table on the f32 path too, so they match. Accepted as within the reach of the numeric opt-in ([ADR 0058](decisions/0058-numerics-opt-in-contract.md)).
- **The allocation does not become 1/4**. The output tensor's storage is allocated at the declared shape, and only 1/4 of it is actually written. This is a choice to keep the arena buckets unchanged, not a VRAM saving.
- **Only the combination of `linearGemvReduce: "parallel"` and `linearCompute: "f32"` is accepted**. Other combinations and non-boolean values are rejected at Session construction, without silently dropping to the f32 path.

## Scope of RMS fusion and the subgroup optimizations (organized 2026-09-14)

`fuseRmsNormAdd` defaults to false and is a setting independent of the submission limit ([ADR 0099](decisions/0099-rms-norm-add-fusion.md)).
That the comparison page's fusion + submission of 768 was used does not mean it became the default of the models, the CLI or the quants.

`rmsNormReduce: "subgroup32"` and `linearGemvReduce: "parallel-subgroup32"` are optional settings.
When an external GPU is passed, request the required features with `acquireGpu({ subgroups: true })`.
They need WebGPU's subgroups / subgroup-size-control, WGSL's subgroup_id and the 32-lane known-answer check, and are rejected when anything is missing.
Deno 2.9.6 did not provide the required features; on Node + Dawn (0.15) they are exposed and the variants run in the core lane ([research 2026-10-16](research/2026-10-16-node-port-results.md) item 12). Running in Chrome and the follow-up test on the M2 are done, but adopting them as the M2 default is deferred.
They are not included in the stored vocabulary of quant.session. The existing workgroup / parallel can be chosen again.
For the numbers, target shapes and the range not yet accepted, see [ADR 0100](decisions/0100-rms-subgroup-reduction.md) and [0101](decisions/0101-linear-gemv-subgroup.md).

## The real-execution checks of the subgroup variants do not run automatically (the playground's /kernel-check page is the manual check)

> Resolved in 0.15 (2026-10-16): on Node + Dawn the development machine exposes `subgroups`, so the real-execution cases below run in the core lane ([research 2026-10-16](research/2026-10-16-node-port-results.md) item 12). The text below records the Deno-era state; the manual check still applies to a machine whose adapter lacks the features.

> Manual procedure since 2026-10-28: the playground's `/kernel-check` page ([apps/playground/README.md](../apps/playground/README.md#the-kernel-check-page)). It runs the same checks as the node GPU tests (one implementation, the private package `@karume/kernel-checks`) on the browser's backend — D3D12 in Chrome, Metal — which the node lanes (Dawn / Vulkan) do not reach: the GEMV and RMS subgroup32 checks below (RMS now with its full test criteria rather than the round trip) and the K-55 / K-91 states-attention rewrites. A device without the subgroup features shows those two as "not available", never PASS. The `/check.html` and round-trip procedure below is the 0.14.0 record.

The real-execution checks of `rmsNormReduce: "subgroup32"` and `linearGemvReduce: "parallel-subgroup32"`
(the cases of `gpu_rms_norm_subgroup_test.ts` / `gpu_linear_gemv_subgroup_test.ts` that actually run the subgroup instructions) SKIP explicitly
on a device without the required features. Deno 2.9.6 did not provide `subgroups`, so in the Deno-era full verify and in CI they
**were not executed in any run**. There is no all-skip guard that turns the SKIP red either (something like `gpu_gate_test.ts`, which turns a missing GPU adapter red).
What the automatic checks see is only WGSL generation (snapshots) and the accept / reject branches; they do not look at the numbers from actually issuing subgroup instructions.
The real-execution check is manual: run the following 2 in Chrome (the procedure is `tools/llm-speed/browser/README.md` (removed in 0.15)).

- GEMV: the measurement page's `/check.html` checks u32 equality of parallel and `parallel-subgroup32` and outputs `PASS` / `FAIL`
  (the same check as `packages/runtime/tests/helpers/gemv-subgroup-check.ts` of 0.14.0 — now `checkGemvSubgroup` of `@karume/kernel-checks`).
- RMS: outside the scope of `/check.html`. Use the measurement page's "round-trip comparison of the 2 RMS paths" (`fused → subgroup32 → subgroup32 → fused`) to
  see that generation does not break. The reduction order changes, so the generated text can change.

When the WGSL or the selection conditions of the subgroup variants change, treat real execution on a backend as unverified until the `/kernel-check` page has run there (until 2026-10-28: these 2).

## The public-surface snapshot check looks only at exports added or removed — signature compatibility is not guaranteed

`packages/<pkg>/tests/public_surface_test.ts` takes only the set of `{ name, kind }` of the declarations exported by each entry (read through the TypeScript 7 synchronous API; the kind vocabulary carried over from `deno doc --json`) and
compares it with a fixture (the body is `packages/runtime/tests/helpers/public-surface.ts`). Overloads are folded into one pair.
Changes to arguments, return values and the contents of types pass as long as the name and kind are the same. The role of this check is "detecting exports added or removed"
(the 2026-09-20 addendum of ADR [0008](decisions/0008-public-api.md)), and signature compatibility is outside the check.
Only the range used by the code that `vp check` type-checks fails indirectly when broken.

## Row-statistics/PV fusion of states attention (2026-09-15)

`stateAttentionReduce: "parallel-fused"` is an optional setting of [ADR 0102](decisions/0102-state-attention-stats-pv-fusion.md).
It fuses only the states form with M<=8 and a static column limit <=1024; out-of-scope shapes and readonly keep the former parallel path.
Acceptance of RTX/Chrome and of the kernel numbers was done; performance acceptance on the M2 remains. On 2026-09-26 regular gemma4's `i4-fast` declared it
(B570 measurements show no slowdown against the reference, about the same as parallel — [ADR 0104 addendum](decisions/0104-gemma-fast-quant.md)). The M2 and the other presets are unchanged.

## gemma4: what the GPU-resident PLE option needs (2026-09-19, opt-in)

`pleResidency: "gpu"` puts the quantized bytes of PLE (an asset of the `model` container) on the GPU as **a single storage binding**
([ADR 0085 addendum](decisions/0085-ple-host-gather.md)). The default is `"host"`, and the former behaviour, numbers and
token sequences do not change at all.

The required binding limit is decided by the distribution. For E2B it is 1,174,405,120 B for QAT (i4) and 2,348,810,240 B for regular Gemma 4 (i8),
and 704,643,072 B for E4B QAT (i2); on a device where **either** `maxStorageBufferBindingSize` or
`maxBufferSize` is short, loading fails with the number of missing bytes.
It does not silently fall back to the host path. The reference machine's (RTX 3080 Ti / Vulkan) limit is 2,147,483,644 B, so
**regular Gemma 4 E2B cannot use this option on this machine** (QAT E2B / E4B fit).

There are 3 knobs that cannot be combined with it (all rejected at load time). `maxResidentPleBytes` is a budget that is never drawn on
with GPU residency, speculative decoding (`speculative`) has the drafter's per-layer inputs outside the range of this option, and
`acquireGpu({ gpuTiming: true })` cannot open a batch while timing.

On its own it is not faster ([research §15](research/2026-09-19-qat-speed-recon.md)): in Chrome's greedy decode, wall 0 to −7%,
GPU +0.08 ms/token; on Deno it is neutral at temperature 0. In Chrome, **prefill gets slower with each round from the 8th generation on in the same page** (TTFT 60 → 180 ms, then
levels off, [research §15.2](research/2026-09-19-qat-speed-recon.md)) — not gather or GPU time but the host side of the prefill run, a phenomenon
on the Chrome (Dawn) side that appears when VRAM occupancy is large (reproduced on the host path with a raised `planBackingBudgetBytes` too; does not appear on Deno). Decode with temperature > 0 or with diagnostics (normal runs)
has since 2026-09-20 been queued in the same batch as the gather and read back at the final fence (`Session.enqueueRead`, [ADR 0054 addendum](decisions/0054-resident-loop-and-fence.md#addendum-2026-09-20--batch-readback-of-graph-outputs)),
so there is one fence, the same as greedy. **Only prefill issues the batch for the gather one ahead per chunk** (on Deno
+10 ms of TTFT per chunk). The option was put in place as a prerequisite for run-ahead (perf-ledger H-27) and is not made the default.
