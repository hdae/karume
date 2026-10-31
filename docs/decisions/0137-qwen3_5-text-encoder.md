# 0137: Qwen3.5 text encoder — a shared module, host-built positions and RoPE, one i8 Session per generation

- Status: proposed (2026-10-31; decision 5's default ruled 2026-10-31 — the maintainer's ruling 51)
- Date: 2026-10-31
- Scope: `packages/models/src/qwen3_5/{rope,session-io,graph-contract,encoder}.ts` (next to the
  tokenizer and chat template already there), the GPU check
  `packages/e2e/tests/e2e_qwen3_5_encoder_test.ts`, the host tests
  `packages/models/tests/qwen3_5_{session_io,encoder}_test.ts`, the lane `models:qwen3_5` and the
  reference rows `packages/models/tests/fixtures/references/qwen3_5.json`. Not the MOSS-Image wiring
  (a later step on the unmerged MOSS branch) and not the recipe (`tools/export-recipes/qwen3_5`).
- Related: [0135](0135-gated-delta-rule-scan.md) (`gated_delta_scan`, the op 24 of the 32 layers
  run) / [0119](0119-wan-umt5-gpu-text-encoder.md) (umT5 on the GPU: the module shape this follows,
  the Session per generation — decision 11 — and the band rule of decision 8) /
  [0128](0128-default-preset-original-weights.md) (how a default preset is chosen; decision 4's
  original-weights table) / [0091](0091-gemma4-host-rope-variable-capacity.md) (host-built RoPE rows;
  decision 2's bound) / [0106](0106-device-keyed-references.md) (sha256 rows per environment) /
  [0129](0129-test-layout-and-lanes.md) (lanes; the kept oracle versus opt-in upstream parity) /
  [0136](0136-models-public-naming.md) (internal names keep their family prefix) / research
  [2026-10-31-qwen3_5-encoder-gpu](../research/2026-10-31-qwen3_5-encoder-gpu.md) (every GPU number
  below).

## Context

MOSS-Image conditions its DiT on the final hidden state of the **text tower** of Qwen3.5-9B (the
language model without `lm_head` and without the vision tower: 32 decoder layers — 24 Gated DeltaNet
and 8 full attention — hidden width 4,096, an embedding of 248,320 rows), read after the final norm,
for prompts of 10 to 4,096 tokens (the empty negative prompt renders to the bare chat template, 10
tokens).

The recipe `tools/export-recipes/qwen3_5` writes that tower as one IR v2 graph with one symbolic
length `T` in [1, 4096]: inputs `input_ids` i32 `[1, T]`, `positions` f32 `[T]`, `rope_cos` /
`rope_sin` f32 `[1, T, 64]`; one output f32 `[1, T, 4096]`. The causal mask is built in the graph
from `positions` (a run-time `ge`, no T² constant); upstream's rotary module is not traced. The only
series is `qwen3.5-9b-text-moss-image-sft-i8-dyn` (per-channel i8 linear and embedding weights, f32
norms and GDN parameters; 34 parts, 7.9 GB; 7.40 GiB uploaded). The container records neither the
RoPE theta nor the rotary width (it has no assets). Its error against the f32 upstream is the
recipe's record: about 3.6–4.4 % relative L2 for i8, while Karume's CPU reference execution of the
container agrees with the recipe's own i8 torch run to about 1e-6.

The tokenizer and the chat template are already on main (`./tokenizer.ts`, `./chat.ts`), with the
length policy of the maintainer's ruling 50: an over-length prompt is rejected by default, and
`overflow: "truncate"` reproduces upstream's silent truncation.

On the 12 GB development GPU the encoder (7.4 GiB of weights, about 7.7 GiB on the device while it
runs) cannot stay resident beside the MOSS DiT and VAE.

## Decision

1. **The encoder lives in `packages/models/src/qwen3_5/` and knows nothing of MOSS.**
   `Qwen3_5TextEncoder` takes token ids and returns the rows; the tokenizer and template stay
   separate steps a pipeline composes. The text tower is a Qwen3.5 part, not a MOSS part — any
   consumer of a Qwen3.5 text tower exported by the recipe reuses it, as Wan reuses `src/umt5/`. It
   is internal for now: no subpath export until its first published consumer (the rule the
   tokenizer's module doc states), so its names keep the family prefix (ADR 0136 decision 4).
2. **The host builds `positions` and the RoPE rows.** `positions` is `0 … T−1` as f32. The rows are
   the recipe's `rope_rows` formula: `invFreq[i] = theta^(−2i/64)`, `angle = position · invFreq[i]`,
   row = `cat(angle, angle)`, every step in f64, the cosine and sine rounded to f32 once. Why the
   host: the rotary module writes in place and calls `cos`, neither of which is IR vocabulary, and a
   folded table would be a `[Tmax, 64]` constant sliced on the symbolic axis (ADR 0091's precedent).
   The rows are bit-identical to the recipe's for every position 0 … 4095 (0 of 524,288 f32 words
   differ), pinned by a tracked fixture written by the recipe; they are not bit-identical to
   upstream's f32 rotary module, and that bound is ADR 0091 decision 2's position-proportional one.
   Theta (1e7) and the rotary width (64) are pinned from the checkpoint's config; the width is
   checked against the container's `rope_cos` / `rope_sin` declaration, theta cannot be (open
   point 1).
3. **Length and ids.** The encoder accepts `1 ≤ T ≤ 4096` (the recipe's bounds) and ids inside the
   embedding's rows (read from the graph declaration), and rejects anything else with a plain
   `Error` before the run: by then the input has passed the tokenizer, so a violation is an
   assembly error (ADR 0107 decision 2). User-facing length handling is the tokenizer's (ruling 50).
   The id check exists because the GPU embedding writes NaN for an out-of-range id, which would
   surface only as a non-finite output.
4. **One Session per instance; the consumer opens it per generation.**
   `Qwen3_5TextEncoder.open(component, gpu)` checks the graph contract and builds one Session on the
   caller's GPU; `encode` calls are serialised on one chain; `dispose` waits for an in-flight run and
   folds the Session, never the GPU. A pipeline that cannot keep it beside its other stages opens
   it, encodes the positive and the negative prompt, and disposes it before the DiT stage — Wan's
   umT5 text stage (ADR 0119 decision 11). Measured on the RTX 3080 Ti: building the Session takes
   8.3 s with the parts in the page cache; one encoding takes 60 ms at T = 10, 196 ms at T = 63 and
   1.40 s at T = 1,000 (warm medians). The MOSS wiring is a later step.
5. **Storage: i8 per-channel weights and f32 tensors only, and no execution knob.** The graph
   contract rejects any other storage (BF16, an all-f32 container, a shared declaration), and the
   session policy accepts no `session` key: that combination is what decision 6 verified, and the
   storage decides the form of execution (umT5's ADR 0119 decision 5). **i8 is the default**
   (the maintainer's ruling 51, 2026-10-31 — below); it is also the only series. A BF16 encoder (the checkpoint's own values — ADR 0128
   decision 4's original weights, about 15.9 GB) does not fit a 12 GB GPU in one Session, and its
   full export needs a fixed-BF16 entry in the writer (the recipe's `export.py` doc: the F32 set
   alone is 29.6 GiB); the gap is recorded here. The ruling followed a human image A/B on the RTX 3080 Ti: MOSS-Image
   with its default preset, 1024² × 50 steps, the three fixed positive prompts × seeds 42–45, the
   same seed and pipeline in both arms, differing only in the text rows — upstream's BF16 encoder
   output against this i8 encoder (PSNR between the pairs 19.9–29.6 dB). The maintainer judged the
   differences detail-level, with no quality problem. Against upstream in f32 the i8 rows differ by
   3.6–7.6 % rel_l2 (BF16 1.3–1.4 %); the largest case concentrates 87 % of its error in one
   numerically sensitive row (the newline after `assistant`, 63.5 % there against BF16's 4.8 %) —
   the place a mixed-precision option would act if a prompt ever shows it in an image (open point 6).
6. **Parity: the GPU against Karume's CPU reference of the same container, plus sha256 rows.** The
   oracle is the IR CPU reference execution: the runtime's per-op TypeScript references walk the
   container's graph on the CPU over the same decoded i8 weights (`outputs/diag/qwen3_5-export-2026-10-31/`,
   two recorded prompts: T = 10 and 63). It is Karume's own oracle, not a torch golden, so the check
   runs in the default lane (ADR 0129). The band is the relative RMS error `‖GPU − CPU‖₂ / ‖CPU‖₂`
   ≤ 2.6e-5: the worst measured error (5.09e-6) × 5, rounded up to two significant figures (ADR 0119
   decision 8's rule). Faults injected into the weights or the inputs land 65× to about 10,000× above it
   (one linear's scale × 1.01: 1.7e-3; one layer's MLP or GDN output zeroed: 0.15–0.27; RoPE rows
   swapped: 0.067–0.080). After the band, each case's rows are pinned as a sha256 row per environment
   (ADR 0106). The export's distance from upstream is not re-checked here; it is the recipe's record.

## Rejected

- **The encoder inside a MOSS module**: it would tie a Qwen3.5 part to one consumer, while the
  tokenizer and template already live in `src/qwen3_5/`.
- **Positions and RoPE inside the graph**: a folded mask is 64 MiB at T = 4096, and the rotary
  module is not exportable as IR; the recipe chose host inputs, and the host mirrors its formula.
- **A resident Session across generations**: it does not fit beside the DiT on a 12 GB GPU.
- **A manifest-level `fromPretrained` surface now** (the standalone `umt5.Encoder`'s shape): no
  distribution exists yet (its repository and pipeline name are open), and both the GPU check and a
  pipeline stage need a component-level entry; the hub surface comes with the distribution.
- **The band behind `KARUME_UPSTREAM_PARITY=1`**: the flag is for comparisons with torch / diffusers
  goldens; this check compares with Karume's own oracle.

## Consequences

- A new lane `models:qwen3_5` claims `*qwen3_5*_test.ts`; the tokenizer test moves into it from
  `core`. The lane needs the 7.9 GB series and the CPU reference under `outputs/`; without them its
  GPU cases skip explicitly with the commands that make them.
- A consumer that pairs the tokenizer with the encoder MUST check that the tokenizer asset's
  `maxLength` equals `QWEN3_5_MAX_TOKENS` (4,096), as Wan does for umT5.
- The CPU reference takes about 12 minutes per two prompts and its harness lives in `outputs/diag`
  (untracked); regenerating the references after a re-export needs that script (the check fails
  loudly on a re-export: it pins part 0's sha256).

## Open points

1. Theta is pinned, not read: the container (or a distribution's `pipelineConfig`) should record
   it so that the encoder can check it, as it checks the rotary width.
2. The GPU path is measured up to T = 1,000. T = 4,096 needs a `maxStorageBufferBindingSize` above
   the WebGPU default for the fused attention's temporaries (the estimate refuses it under the
   default; `acquireGpu` requests the adapter's limit); time and memory there are unmeasured.
3. ADR 0128 decision 4's table has no row for the Qwen3.5 text encoder yet (BF16: skipped —
   decision 5).
4. Other GPUs (the RTX 5070 Ti on D3D12) need their own sha256 rows and a run of the band check.
5. The IR CPU reference harness could move into a tracked tool so that references can be remade
   without the diag directory.
6. The i8 error's sensitive rows (decision 5): kept as is by ruling 51; a per-layer or per-row
   higher-precision option is the lever if an image ever shows them.
