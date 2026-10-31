> Point-in-time record (2026-10-31) of the first runs of MOSS-Image's text-encoder stage (the prompt form of `generateMoss` — the Qwen3.5 text tower of ADR 0137 wired into ADR 0138's internal entry) on one machine: the token ids against upstream's, the stage's rows against the encoder lane's and against upstream's bf16 embeddings, the stage's time and memory, and one prompt → image generation. Measurement material for the [ADR 0138](../decisions/0138-moss-image-dit-vae.md) addendum of 2026-10-31 and the [ADR 0137](../decisions/0137-qwen3_5-text-encoder.md) addendum of the same day (both proposed); not a design source of truth. The raw rows, the one-off diagnosis script and the comparison are under `outputs/diag/moss-text-stage-2026-10-31/` of the main checkout (not tracked); the e2e's results under `outputs/verify/node-nvidia-geforce-rtx-3080-ti/2026-10-31_moss-prompt/`.

# MOSS-Image's text-encoder stage: token ids, distance from upstream's embeddings, time and memory

Question: does the prompt form give upstream's token ids, does the stage hand the DiT exactly the
verified encoder's rows, how far are those rows from upstream's bf16 embeddings for the fixed
prompts, and what does the stage cost per generation on a 12 GB GPU?

Answer: the ids of all four fixed prompts are exactly upstream's. The stage's rows of `fridge-note`
and of the empty negative are the encoder lane's pinned rows bit for bit. Against upstream's bf16
embeddings the rows sit at 3.9–4.9 % relative L2 for three prompts and at 7.8 % for
`guide-fridge-note`, where one row (the `\n` after `assistant`) has a cosine of 0.804; Karume's CPU
reference execution of the same container gives the same 7.8 % and the same row, so that gap is the
i8 export's (or upstream's bf16 run's), not the GPU's. The stage takes about 9.9–10.1 s per
generation, of which 8.9–9.1 s is the Session build; it needs 8,603 MiB by its headroom check and
held at most 7,602 MiB of WebGPU buffers.

## 1. Setup

- RTX 3080 Ti (12 GB, driver 615.78.08), Linux 7.2.7, Node 24.22.2, Dawn (npm `webgpu` 0.6.2,
  Vulkan). Worktree of `wip/moss-image-rebased` at `7d093c81` plus the uncommitted text stage. The
  GPU was idle (`tools/diag/gpu-busy.zsh 5`: 0 %). The device was acquired without asking for a
  feature (as the encoder lane acquires it).
- The text series `qwen3.5-9b-text-moss-image-sft-i8-dyn` (34 parts, part 0 sha256 `e9b581dc…260ff`,
  exported 2026-10-31 09:02 UTC) and the tokenizer asset `outputs/series/qwen3_5-tokenizer/tokenizer.json`
  (`karume-qwen3_5-tokenizer/1`).
- Upstream: `outputs/series/moss-image-sft-text-embeds/text_embeds.safetensors` (`moss.text_embeds`:
  `MOSSImagePipeline.encode_prompt` on CPU in bf16, torch 2.13.0+cpu, transformers 5.14.1; its
  metadata records the token ids the encoder received, taken with a forward pre-hook).
- Metrics (f64 accumulation, the recipe's record's — `outputs/diag/qwen3_5-export-2026-10-31/full.json`):
  relative L2 `‖K − U‖₂ / ‖U‖₂` over all rows, max |Δ|, max |Δ| ÷ max |U| (ratio), and the smallest
  per-row cosine similarity.
- The checks: `packages/e2e/tests/e2e_moss_prompt_test.ts` (lane `models:moss`).

## 2. Token ids

The acceptance check (`planMossGenerate`, prompt form: the chat template, then the tokenizer) gives
exactly the ids of the asset's metadata for all four prompts: `fridge-note` 63, `guide-whiteboard`
134, `guide-fridge-note` 62 and the empty negative 10 tokens.

## 3. The stage's rows against the encoder lane

The rows `encodeMossPrompts` returned for `fridge-note` and for the negative hash to this
environment's rows of `e2e_qwen3_5_encoder_test.ts` (`fixtures/references/qwen3_5.json` — the rows
held within 2.6e-5 relative RMS of Karume's CPU reference). The negative was encoded three times,
once per stage run below, on three Sessions; every time its rows were those rows.

## 4. Distance from upstream's bf16 embeddings

| Prompt              | T   | rel L2    | max \|Δ\| | ratio     | min row cosine (row, id)          |
| ------------------- | --- | --------- | --------- | --------- | --------------------------------- |
| `fridge-note`       | 63  | 3.984e-2  | 2.385     | 3.290e-2  | 0.974377 (58, `<\|im_start\|>`)   |
| `guide-whiteboard`  | 134 | 3.883e-2  | 1.956     | 2.149e-2  | 0.979944 (105, id 12)             |
| `guide-fridge-note` | 62  | 7.788e-2  | 10.73     | 1.479e-1  | 0.803682 (59, `\n`)               |
| negative (`""`)     | 10  | 4.864e-2  | 1.325     | 2.664e-2  | 0.997514 (8, `<think>`)           |

For the two prompts the recipe's record covers, these are its CPU-reference figures
(`ir_i8_vs_asset_bf16`: 3.98382e-2 and 4.86367e-2) to four digits — the GPU adds nothing visible at
this scale. The record also puts the bf16 asset itself 1.38 % (`fridge-note`) and 2.09 % (negative)
from upstream's f32 run, and the i8 export 3.58 % and 4.36 % from it, so against upstream's f32 the
i8 rows are closer than these bf16 figures say.

### 4.1 `guide-fridge-note`: one row

`guide-fridge-note` is the `fridge-note` prompt with typographic quotes (“ ”) instead of ASCII ones.
Its gap is one row: row 59, the `\n` (id 198) after `assistant` in the template's tail, at a per-row
relative L2 of 0.643 and max |Δ| 10.73; every other row has a cosine ≥ 0.9954 (the next is row 60,
`<think>`).

Karume's CPU reference execution of the same container on the same ids
(`outputs/diag/qwen3_5-export-2026-10-31/ir_cpu_reference.ts`, 485 s on the CPU):

| Pair                               | rel L2    | max \|Δ\| | min row cosine (row) |
| ---------------------------------- | --------- | --------- | -------------------- |
| GPU vs CPU reference               | 2.914e-6  | 9.92e-5   | 0.99999999997 (57)   |
| CPU reference vs upstream bf16     | 7.788e-2  | 10.73     | 0.803681 (59)        |

The GPU is inside the encoder lane's band (2.6e-5) here too, so the gap is between the i8
container and upstream's bf16 run. Which side makes that row (the i8 rounding of the weights, or
upstream's own bf16 error on it) is not separated: there is no f32 upstream run of this prompt.

## 5. Time and memory of the stage

Three stage runs, each a positive prompt with the empty negative (one Session per run, as a
generation opens it):

| Positive            | Session build | positive encoding | negative encoding | dispose | whole stage |
| ------------------- | ------------- | ----------------- | ----------------- | ------- | ----------- |
| `fridge-note`       | 9,130 ms      | 562.3 ms          | 130.7 ms          | 0.8 ms  | 10.05 s     |
| `guide-whiteboard`  | 8,912 ms      | 634.1 ms          | 108.4 ms          | 0.7 ms  | 9.86 s      |
| `guide-fridge-note` | 9,002 ms      | 527.7 ms          | 118.8 ms          | 0.7 ms  | 9.86 s      |

The first encoding of a Session is cold (its pipelines are built then), so it is not comparable with
the encoder lane's warm medians (196 ms at T = 63). The whole stage includes its headroom check and
the release wait.

Memory, the same in all three runs except the peak:

- The stage's headroom need: 9,020,414,604 B (8,603 MiB) — the estimate at the longer prompt
  (8,215,108,236 B, the same at T = 62, 63 and 134: the bound of the kept plan backings, not the
  length, sets it there) + the largest weight part (268,435,456 B) + the margin (512 MiB).
- The WebGPU buffers held (non-probe, their own sizes): peak 7,958,092,908 / 7,971,037,628 /
  7,957,911,164 B (7,589–7,602 MiB), 0 B once the stage returned.
- The Session's diagnostics at dispose: weights 7,946,675,776 B; no plan backing built (the two runs
  have two lengths — a backing is built only on a repeated shape).

## 6. Prompt → image

`f32+dit8`, 256², 2 steps, `fridge-note` with `negativePrompt: ""`, guidance 3, shift 5, seed 42:
the Sessions opened and closed as text_encoder → dit_text → `dit` + `dit_skip22` → the three VAE
graphs, each stage's disposed before the next opened. 18.0 s for the generation, of which the text
encoder was 9.1 s of Session build and 0.6 s from its opening to its disposal. The image's sha256
row `f32+dit8-2step-fridge-note-qwen3_5-i8-seed42-256x256-g3-shift5` is `630b7894…12022` (a new row:
the embeddings are Karume's, not upstream's).

## 7. Not shown here

- Whether the distance of §4 shows in the images — the A/B of upstream's embeddings against
  Karume's (ADR 0137 decision 5, the maintainer's ruling 51). `guide-fridge-note` is the prompt to
  include there.
- The practical preset with the prompt form, other sizes and step counts.
- T = 4,096 (ADR 0137 open point 2) and other GPUs.
