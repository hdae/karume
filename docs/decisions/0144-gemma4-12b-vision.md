# 0144: Gemma 4 12B image input

- Status: proposed (2026-11-07 — the maintainer approved the plan; implemented for 12B on the development machine, unreleased)
- Date: 2026-11-07
- Related: ADR [0066](0066-generation-context-state-slots.md) (state slots) / [0067](0067-autoregressive-attention-vocabulary.md) (the
  sliding predicate) / [0084](0084-gemma-tokenizer-chat.md) (the chat layer; content parts were out of scope) /
  [0128](0128-default-preset-original-weights.md) (presets) / [0131](0131-bf16-weight-execution.md) (bf16 weights) /
  [0142](0142-decode-run-ahead.md) (`positionsOnly`)

## Context

Gemma 4 12B has no vision tower. An image becomes rows of raw pixels, and a small embedder (about 50 M parameters, bf16) projects
them to the hidden size. Those vectors overwrite the embedding of the image's soft tokens. Inside one image the sliding layers
attend in both directions; everything else is causal.

Three things did not exist: an input for image vectors in the product graph, a way for a run to say which rows are one image, and
preprocessing that matches upstream's.

E2B and E4B have a real tower and causal attention, and come later. They need the first of those three and not the second.

## Decision

1. **Upstream's masks of `create_masks_for_vision_model`.** Full layers are causal. Sliding layers are window AND (causal OR block).
   transformers has a second path in `Model.forward` that ORs the block onto full layers too, and plain `forward` and default
   `generate()` take it. Every outside reference (Google's JAX `gemma`, vLLM, llama.cpp) and the docstring of the method say the
   full layers stay causal, and the transformers change that moved Gemma 4 there left this one override behind, so the second path
   reads as a missed edit. This is an upstream defect not copied (ADR 0005 addendum of 2026-10-11). There is no Google reference for
   12B itself, so the confidence is medium to high. Goldens force this path.

2. **One product graph with two more inputs**: `soft_embeds` and `soft_mask`, with
   `embeds = where(soft_mask, soft_embeds, embed(ids))`. Decode and verify bind zeros and false. A second prefill graph would keep
   decode free of the `where`, at the cost of a second Session sharing the context and weights. Kill line: decode more than 1% slower
   on `i4-fast`.

3. **The block belongs to the run, not to the graph.** `GenerationRun.bidirectionalRows` carries one half-open row range. It reaches
   the kernel as two more words of the lengths uniform, as absolute columns, and the sliding ①ₜ has a variant whose predicate adds the
   block. Causality and the window are predicates over logical columns, which no tensor carries, and the block is the same kind
   of thing. Runs without a block keep the kernels, keys and bytes they had, so decode cannot move. The plan cache keys on the
   block's presence.
   - A run holds one block, so the host's chunk planner gives each image its own chunk and never splits one.
   - Only ①ₜ has the variant: soft tokens always outnumber its M ≥ 16, and a smaller plan fails loudly.
   - A block longer than the window is refused, so the block's half of the predicate needs no window. Upstream's AND is then
     implied.
   - The lengths uniform grew from 8 to 16 bytes. Kernels without a block read the first two words.

4. **Preprocessing matches byte for byte, not within a tolerance.** torch's uint8 antialiased bicubic has one answer: default, AVX2 and
   AVX-512 gave the same bytes. It differs from Pillow in 0.42–0.81% of samples (by at most 2) and from `resizeRgb8` in rounding. It
   is ported as its own path (`image/torch-resample.ts`), and the fixture holds upstream's SHA-256 over 16 processor cases and 7 resize
   cases, every budget and the rule for a side that rounds to 0 included. The existing paths do not move.

5. **The budget is 280 for now.** The embedder's row count is symbolic and the host drops upstream's padding, so 70, 140 and 560 need no
   re-export. 1120 does not fit the 768-row chunk.

## Acceptance (2026-11-07, RTX 3080 Ti, Node + Dawn)

| Check | Result |
| ----- | ------ |
| Preprocessing against the upstream processor | byte-identical: 16 synthetic cases, 7 resize cases, the 3 corpus images |
| Embedder against upstream's module (F32, bf16 weights) | max \|Δ\| / max \|x\| at most 9.8e-6 over 3 images; the e2e band is 5e-5 |
| Greedy with an image, 16 tokens, `i4` | token for token equal to the torch golden on 2 images (the third fell under the margin floor) |
| Text goldens over the new graph | the existing greedy goldens are byte-identical, and the same tokens come out |
| Bidirectional kernel | u32 identity against the always-true predicate with the host's mask, 33 cases, 4 fault injections that mismatch |
| Weight parts of the re-exported 12B container | 31 of 32 parts byte-identical; only part 1 (the descriptors) changed |
| Mask A against B on the first step, one image, real weights | max \|Δ logit\| 1.55, KL(A‖B) 8.8e-4, top-1 the same |
| Kill line of decision 2: 12B `i4-fast` decode, P = 973, 64 tokens, ABBA × 2, against the graph without the soft inputs | 22.63 against 22.70 ms per token (−0.3%, within the drift between runs); the same token ids in all 8 runs |

## Consequences

- Pipelines that never see an image run what they ran. The `where` is the one cost they pay.
- Every 12B container's graph changes, so a release re-uploads the descriptor part of each. The weight parts should not change.
- The history of a conversation holds image rows on the host (about 4.3 MiB per image), because a truncated history re-renders from
  the start.
- `pack2x16float` plays no part: S and the sliding KV of 12B stay f32 under the variant.
