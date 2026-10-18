# 0091: gemma4 RoPE is supplied by the host per chunk — capacity and chunkLength become run-time options

- Status: accepted (2026-09-03); partly superseded by [0104](0104-gemma-fast-quant.md) (decision 7's
  pipeline default, by its addendum of 2026-09-26; marked 2026-10-18). Design material:
  [research/2026-09-03-gemma4-context-length-sweep.md](../research/2026-09-03-gemma4-context-length-sweep.md),
  [research/2026-09-03-gemma4-chunklength-k12-sweep.md](../research/2026-09-03-gemma4-chunklength-k12-sweep.md)
- Date: 2026-09-03
- Related: [0067](0067-autoregressive-attention-vocabulary.md) (replaces decision 4, "the exporter
  bakes per-layer-type RoPE"; "the attention op does not know about RoPE" MUST stays) /
  [0083](0083-generation-api-surface.md) (re-rules Consequences 4) / [0034](0034-dit-dynamic-tokens.md)
  (finding 2, f32 trigonometry) / [0085](0085-ple-host-gather.md) (per-chunk derived input) /
  [0066](0066-generation-context-state-slots.md) / [0089](0089-memory-limits-preflight.md) /
  [0070](0070-shard-loading-admission.md) / [0058](0058-numerics-opt-in-contract.md) (K-12) /
  [0038](0038-manifest-v1.md) (`pipelineConfig` needs no format bump)

## Context

The gemma4 E2B distribution baked the RoPE cos / sin tables (two layer types × cos / sin, f32,
6 KiB per row) as IR initializers, so the conversation length `capacity` was capped by the row count
`maxPosition` (1,024). Upstream allows 128K. Baking 128K rows would add 762 MiB to the distribution
((131,072 − 1,024) × 6,144 B) and need 768 MiB of resident VRAM (all rows), for every user.

The position sweep (research 2026-09-03) showed VRAM is no wall even at 128K (+2.3 GiB); time is
(prefill at chunkLength 32 grows with P², 5.4 minutes at 16K; decode's full-layer ③PV runs over the
KV length sequentially in one thread, 2.6× slower at 16K). Established facts:

1. In the shipped IR only the four RoPE gathers read `position_ids`. Their output `[1, M, headDim]`
   fits the place PLE uses: a derived input the host builds per chunk. A resident table of capacity
   rows would bring the capacity symbol `C` into an input shape and break `capacitySymbolOf` and the
   binding-point MUST.
2. Upstream RoPE is f32 throughout (ULP ≈ 0.008 rad at 131,071; SLEEF cos / sin within 1 ULP). TS
   in f64 rounded to f32 is **not bit-identical** (maximum 6.9e-5 over 0..1,024, 4.8e-3 at 131,071,
   9.4e-3 over 0..131,071). f64 is the exact side; gemma4's checks were already token equality plus
   logits atol 1e-2 (margin floor 2.5e-2).
3. chunkLength has no bound in the shipped IR, so a `karume.json` declaration alone raised it to 768:
   prefill 5–9× faster (16K: 326 → 60 s), token sequence unchanged. Since 2026-09-03 the declared
   `maxChunkLength` is the limit (decision 3); the declaration is the only source of the trace range.

## Decision

### 1. RoPE tables leave the distribution; the host passes cos / sin per chunk

Graph inputs are `input_ids` + `rope_{sliding_attention,full_attention}_{cos,sin}` (f32
`[1, M, headDim]`; sliding 256 / full 512) + `per_layer_inputs` + `last_row`. **The `position_ids`
input, the RoPE gathers and the `rotary_emb.*` initializers are gone.** The exporter replaces the
upstream rotary module with a pass-through of the cos / sin the wrapper received; leftovers of baked
tables are rejected by the graph check.

`packages/models/src/gemma/rope.ts` is the TS source. `DerivedRunInputs.derive(ids, positions, …)`
builds the four inputs where the PLE gather happens. Pad rows (position 0) use the same formula
(cos = 1 / sin = 0).

### 2. Numeric contract: TS f64 is the source; position-proportional parity; golden refreeze

- `invFreq[i] = theta^(−2i/headDim)` (`i < rotaryDim/2`, otherwise 0), `angle = position × invFreq`,
  row = `cat(freqs, freqs)`. The default and proportional rope_types share this formula
  (attention_scaling = 1; any other factor fails loudly at export).
- The parity check compares with the **real output of the upstream module** (fixture) within
  `2.5e-7 + position × 1.2e-7` (upper bound of upstream's f32 angle error), with fault injection
  (shifted position, swapped layer type, theta, rotaryDim) — `packages/models/tests/gemma4_rope_test.ts`.
  `tools/export-recipes/gemma4/rope.py` mirrors the formula and is checked against the same fixture.
- Bit identity is not claimed. f64 IEEE arithmetic is engine-independent; `Math.cos` / `Math.sin`
  differences ≤ 1 ULP survive f32 rounding with probability ≈ 2⁻²⁹ per element and are ignored.
- Re-export moves the shipped bytes (GPTQ rounding); the goldens were refrozen.

### 3. Distribution declarations (`pipelineConfig` — no format bump)

`chunkLength` (default 768 = the export Dim bound), `maxChunkLength` (trace bound of symbol `M`, 768;
added 2026-09-03), `capacity` (default 4,096, the measured practical line), `maxPosition` (upstream
`max_position_embeddings` 131,072 — **the position limit the model declares**), `rope` (per layer type
`{theta, headDim, rotaryDim}`, derived from the config) and `sampler`. TS checks
`chunkLength ≤ maxChunkLength` and `chunkLength ≤ capacity ≤ maxPosition`
(`parseGemma4PipelineConfig`, `packages/models/src/gemma/config.ts`).

### 4. capacity and chunkLength are run-time options

- **capacity** is set when `sequence()` / `Gemma4ChatSession` is created (one conversation = one
  context = one capacity); default = the declared value. The capacity symbol is bound only in
  `createGenerationContext` (ADR 0066 addendum 7). `Gemma4ChatSession`'s overflow handling uses the
  session's capacity.
- **chunkLength** is in `Gemma4PipelineOptions` (a plan-time constant, ADR 0066 decision 4; keeps the
  PreparedPlan LRU clean); default = the declared value.
- The estimate is `Gemma4Pipeline.estimateSessionMemory({capacity, chunkLength})`, after load and
  before context creation (ADR 0089 addendum of 2026-09-02), counting states-form attention's
  temporary S and row statistics (`c574aa7`).

### 5. `requiredLimits` is baked with `maxPosition`

The state symbol is bound to `maxPosition`, not `capacity`: baking the default capacity would fail on
a device that meets the declaration once a larger capacity is chosen. gemma4's value did not change
(main embedding i8 384 MiB > the full slot at 131,072, 256 MiB).

### 6. Prefill progress and diagnostics hooks

`Gemma4ChatOptions.onPrefill?: ({chunk, chunks}) => void` (chat / send, non-breaking) and
`Gemma4PipelineOptions.onRunDiagnostics?: (SessionDiagnostics) => void` (as in the other families).

Addendum (2026-09-07): the hook takes a **second argument `Gemma4RunPhase`** (`{kind:"prefill",
chunk, chunks}` / `{kind:"decode", step}`); only gemma4 runs one graph in two shapes, and counting
calls misread multi-chunk prompts (investigation F-01 of 2026-09-06). The last decode run (stop
token) emits no `token` event but still delivers diagnostics (notifications =
`GenerationStop.tokens − 1 + prefill chunk count`).

### 7. Relation to K-12 (KV-parallel reduction of ③PV)

The linear term of decode is outside this ADR (perf-ledger K-12; ADR 0067 addendum of 2026-09-03),
implemented as an opt-in (`8b76ef4`). Promotion to the default followed ADR 0058 decision 6 (quality
ruling and golden update in one commit) and came with this ADR's golden refreeze (ruling of
2026-09-03): the `Gemma4Pipeline` default became `"parallel"` (`"sequential"` returns to the
reference path); the runtime default stayed. (Superseded by ADR 0104's addendum on 2026-09-26:
`stateAttentionReduce` is a manifest `session` declaration, the family has no code default, and a
quant without a declaration or `fromAssets` runs the reference path `"sequential"` —
`packages/models/src/gemma/session-options.ts`; default presets: ADR
[0128](0128-default-preset-original-weights.md), proposed.)

## Consequences

- **Breaking** (unpublished surface): `position_ids` leaves the graph inputs (`fromAssets` and
  low-level consumers build the RoPE inputs; the `./gemma` subpath exports `gemma4RopeInputs`);
  `GenerationProgramSpec.positionIds` is removed; `DerivedRunInputs.derive` takes positions; `rope`
  is required in `pipelineConfig`. Distributions with baked tables (karume/4 gemma4) cannot be read.
- Time sets the practical limit: chunkLength 768 + K-12 gives prefill 60 s and decode 41 ms/token at
  P = 16K (next levers in perf-ledger).
- minicpm5's `export_decode` still bakes the table (separate ruling, open). Research 2026-09-02
  turboquant-recon §3 ("the context-length ceiling is the RoPE table") is overridden by this ADR.
