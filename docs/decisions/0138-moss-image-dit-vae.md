# 0138: MOSS-Image's DiT and VAE without its text encoder — one S-shaped DiT graph, a separate text-refiner graph, a weight-borrowing skip-22 graph, an untiled three-graph FLUX.2 decoder

- Status: proposed (2026-10-27; the results of stages 2–5 recorded 2026-10-28). The design and
  decisions 8, 10 and 12 are ruled (see Rulings); the points settled during implementation were
  taken as the recommended option and reported to the maintainer, and are listed there apart from
  the rulings. The default preset is the practical preset `f32+dit8-a8-attn8-online` (the
  maintainer's visual review passed — Rulings), the `defaultQuant` of the distribution's manifest
  since 2026-11-01 (decision 13, Addendum of 2026-11-01). The text encoder (ADR 0137) is wired as
  the prompt form of decision 9 since 2026-10-31, in the narrow scope of the maintainer's ruling
  53 a (Addendum of 2026-10-31); the distribution step followed on 2026-11-01 (a local mirror and
  the subpath `@karume/models/moss`; not published on Hugging Face — Addendum of 2026-11-01).
- Date: 2026-10-27
- Scope: the family `packages/models/src/moss/` on `main`, with the subpath `@karume/models/moss`
  and the distribution recipe `tools/export-recipes/moss/distribution.py` since 2026-11-01 (the
  Addendum of 2026-11-01; before it, an internal family — decision 12). The work was developed on its own
  branch `wip/moss-image` and was not part of the 0.15.0 release (it had no text encoder then). It
  was rebased onto `main` after the 0.15.0 release (2026-10-31), and this ADR got its number then
  (two branches added ADRs in parallel, so a number taken earlier could have collided silently on a
  rebase); it was taken into `main` 2026-11-01.
- Related: ADR [0121](0121-wan22-ti2v-5b.md) (Wan2.2 — stage pattern, `where` with a condition mask,
  meta-trace export, layer-by-layer f64 reference, numeric baseline) /
  [0119](0119-wan-umt5-gpu-text-encoder.md) (big-model export) / [0118](0118-wan21-video-generation.md)
  (precomputed text embeddings, the upstream checks r and m) / [0034](0034-dit-dynamic-tokens.md)
  (S-shaped DiT, rope base tables) / [0033](0033-vae-fixed-tile-decode.md) + [0038](0038-manifest-v1.md)
  §4 (Anima's always-tiled distribution form — not followed here, decision 8) /
  [0124](0124-attention-row-block-budget.md) and [0060](0060-row-block-attention.md) (row-blocked
  attention) / [0132](0132-online-int8-attention.md) (online int8 attention — its addenda of
  2026-10-27 and 2026-10-28 are decision 10's runtime record) / [0096](0096-speculative-decoding.md)
  + [0134](0134-wan22-vae-tile32-shared.md) (shared initializers, a borrowing graph) /
  [0114](0114-irodori-dit-context-split.md) (a once-per-generation condition graph) /
  [0095](0095-plan-backing-budget.md) (plan backing budget) /
  [0112](0112-anima-transformer-residency.md) (`fitsHeadroom`) /
  [0128](0128-default-preset-original-weights.md) (default preset procedure, C1–C3, original-weights
  table) / [0131](0131-bf16-weight-execution.md) (bf16 storage) /
  [0110](0110-practical-tier-numerics-contract.md) / [0106](0106-device-keyed-references.md) /
  [0058](0058-numerics-opt-in-contract.md) / [0059](0059-op-vocabulary-entry-doors.md) /
  [0130](0130-project-language.md) (language gate) / [0135](0135-gated-delta-rule-scan.md) (the
  scan op the text encoder needs) / measurements: [research
  2026-10-28](../research/2026-10-28-moss-image-dit-vae.md)

## Context

### Request

The maintainer asked for MOSS-Image's DiT and VAE to be implemented first. MOSS-Image (OpenMOSS,
`OpenMOSS-Team/MOSS-Image-SFT`, Apache-2.0, not gated) is a 5B single-stream text-to-image DiT
with a Qwen3.5-9B text encoder and the FLUX.2 VAE (small decoder). The text encoder — 24 of its 32
layers are Gated DeltaNet — is out of scope: it needs a scan op (ADR 0135). The DiT is driven by
text embeddings precomputed with the upstream pipeline, as Wan's `"precomputed"` text route did
first (ADR 0118 decision 4).

### Terms introduced here

| Term                  | Meaning                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| joint sequence        | the DiT's token sequence `[image tokens (N = H/16·W/16), text tokens (T)]`, length S = N + T; full self-attention over it, no mask          |
| text refiner          | `text_embedder`: Linear(4096→3072) + two attention / FFN blocks over the text tokens only, without modulation                               |
| shared modulation     | `shared_block_modulation`: one Linear(3072→6·3072) of the time embedding, the same six vectors for all 42 blocks                            |
| situ                  | the FFN activation `(4·tanh(g/4)·sigmoid(g)) · (25·tanh(h/25))` over the two halves of `ffn_up_gate`                                        |
| skip 22               | the upstream default negative branch: the forward without block 22 (zero-based), `uncond_skip_layers=(22,)`                                 |
| text context          | the DiT graph input `[1,S,3072]` holding the refiner output in the text rows and zeros in the image rows                                    |
| training grid         | the scheduler's 1,000 σ points (and their timesteps σ × 1000); every requested σ is snapped to one of them                                  |
| BatchNorm restoration | the step before the VAE: `z·sqrt(running_var + eps) + running_mean` with the VAE's BatchNorm statistics over the 128 patchified channels    |
| key valid length      | the online attention kernel's key length N when it is not a multiple of 4: the packed Vᵀ rows are padded to `ceil(N/4)` words (decision 10) |
| three-dispatch route  | the existing fused attention: ①QK writes the scores S, ② row statistics (softmax), ③PV (P × V)                                              |
| P̃ / Vᵀ                | the softmax weights quantised to int8 on the 1/127 lattice / V transposed so that the key axis N runs along a row                           |

r, m, relRMS, the reference path, practical presets and the self A/B check are used as in ADR 0118,
0120 and 0121.

### Findings of the investigation

Upstream code citations are `pipeline_moss_image.py` at the pinned revision (decision 1).
diffusers / transformers citations are the shared recipe venv (diffusers 0.39.0, transformers
5.14.1, torch 2.13.0+cpu).

- **The DiT** (`transformer/config.json`): dim 3072, 24 heads × 128, FFN 8192, 42 layers, text_dim
  4096, latent 128 ch, rope base 1e4; 463 tensors, `total_size` 10,164,617,280 B (bf16).
  Parameter count by arithmetic: 5,082,308,608.
- **Single stream, no cross-attention** (`:243-259`): image tokens (`video_embedder`
  Linear(128→3072)) and refined text are concatenated as `[image, text]` and pass all 42 blocks; the
  head reads the first N rows only (`:256`). Batches are a loop of B = 1; text is never padded
  (`:244`). Only `video_embedder` and `final_linear_video` run on N rows upstream; the blocks and the
  refiner run on the same rows as in this design.
- **RoPE** (`:151-186`): four axes (r, f, h, w) × 16 bands, half-split rotation (`:77-82`). Image
  tokens sit at `(0, T, h − (H−1)/2, w − (W−1)/2)` — **their positions depend on the text length**;
  text tokens at `(0, i, 0, 0)`. Bands, phase and sin / cos are f32 on CPU (`:158-161`).
- **Time** (`:234-241`): the sinusoid in float64, then f32, then the MLP. The timestep is the snapped
  σ × 1000 in f32 (diffusers `FlowMatchEulerDiscreteScheduler.set_timesteps` with the shift held at 1,
  `:286`), so it takes one of the 1,000 training-grid values.
- **Sampler** (`:264-287`, README "Sampling settings"): flow-matching Euler, shift 5, 50 steps, CFG 3,
  empty negative prompt; the requested grid (`torch.linspace` in f32) is snapped to the nearest of
  the 1,000 training-grid values and de-duplicated. The negative branch skips block 22 in every step
  (`:344`, `:410`, `:416`). CFG and Euler are f32.
- **The negative text length is a constant**: the negative is `encode_prompt('')` (`:394-397`) under
  the chat template `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n`; the
  shipped `tokenizer.json` gives T₋ = 10 (a port of its BPE gave the same ten token ids in the
  investigation; `moss.text_embeds` records T₋ from upstream's own call).
  The encoder output is taken after Qwen3.5's final RMSNorm (`modeling_qwen3_5.py:1222-1225`); MOSS
  adds no normalisation.
- **VAE** (`vae/config.json`, diffusers `AutoencoderKLFlux2` / `Decoder` / `UNetMidBlock2D` /
  `ResnetBlock2D` / `Upsample2D`): 2D, latent 32 ch, patch 2×2, decoder widths `[96,192,384,384]` (up
  blocks 384→384, 384→384, 384→192 each followed by ×2, then 192→96 at full resolution), three resnets
  per up block, GroupNorm (32 groups) in every resnet and in the mid attention, a single-head mid
  attention (dim 384), `post_quant_conv` 1×1, f32 weights. `Upsample2D` calls
  `F.interpolate(scale_factor=2.0, mode="nearest")` directly (`upsampling.py:175`). Upstream decodes
  **untiled** (`:437`); diffusers has a `tiled_decode` (1,024 px tiles, 25 % overlap —
  `autoencoder_kl_flux2.py:217`, `:381`) that the pipeline does not call.
- **GroupNorm was new to Karume**: no op and no exporter handler. GroupNorm normalises over the
  whole spatial extent, so tiling changes every GroupNorm's statistics.
- **IR limits**: one symbol per dimension (`docs/ir-v2.md`, the dimension language); `cat` only along
  a constant or the same symbol; `slice` only on static axes. `cat(image[N], text[T])`, `x[:, :N]`
  and the GroupNorm reshape `[32, C/32·h·w]` (a product of symbols) have no representation.
- **Online int8 attention needed `N % 4 == 0`** before decision 10 (ADR 0132 decision 3; ③PV i8a8
  and s16 still do): P̃ and Vᵀ pack four keys per word along N (q / k pack along D). For a square
  image N = k² and k² mod 4 ∈ {0, 1}, so S₋ = N + 10 is never a multiple of 4: the negative half of
  all forwards fell back silently to the three dispatches with S in f32, and the positive half was
  eligible only when T₊ ≡ −N (mod 4). The complete-eligibility census check
  (`assertAttentionSeatsComplete`) is red by construction on such a run.
- **The same condition elsewhere**: only Wan declared the online route, and every shape Wan accepts
  has `N % 4 == 0`. Anima's default preset (a8 + s16) never takes the online route; it accepts shapes
  with `N % 4 != 0` and there, by design, runs ③PV and S in f32 (4,608 of its 9,409 accepted shapes —
  `docs/limitations.md`). The route is one pure function, `attentionRoute`, which decides the online
  kernel (`attentionOnlineI8a8Eligible`) and ③PV i8a8 (`attentionPvI8a8Eligible`) separately.
- **Two shapes per step**: S₊ and S₋ alternate. In one Session two DiT-sized backings beyond
  `planBackingBudgetBytes` (default 256 MiB) are rebuilt on every switch; the only observation point
  is `planBacking.buildCount`, not `lastRunPrepared.hit`.
- Arithmetic per forward (linears 9.513 GFLOP per token over the 42 blocks; attention
  42 × 4 × S² × 3,072): at 1024², T = 256 (S = 4,352) linears 41.40 and attention 9.77 TFLOP (19.1 %);
  at the area cap of decision 8 with T = 4,096 (N = 10,922, S = 15,018) 142.86 and 116.40 TFLOP
  (44.9 %).
- Arithmetic for the untiled VAE: the largest single tensor `[192,H,W]` f32 fits a 2 GiB binding
  (about 2 GiB on the RTX 3080 Ti — ADR 0124) up to 2³¹ / (192·4) ≈ 2,796,202 pixels.

### Constraints

- Cross-cutting invariants: fail loudly; no tolerance on sha rows (ADR 0106); runtime dependencies are
  Web standard APIs only; modules have no side effects; rope cos / sin are never computed in TS
  (ADR 0034 decision 3).
- Development machine: RTX 3080 Ti (12 GiB), Node + Dawn, 64 GB RAM; the GPU is shared with training
  and with other sessions (`flock /tmp/karume-gpu.lock`).
- Wan, Anima and every other family keep their values bit for bit; the runtime changes (decisions 7
  and 10) are guarded by the existing sha rows and codegen snapshots.
- Another session works on `wip/0.15` (runtime GEMV, submit, gemma PLE, Session build); this work
  stays on its own branch and touches none of those files.

## Decision

### 1. Upstream — `OpenMOSS-Team/MOSS-Image-SFT` pinned by commit SHA; transformer, VAE and scheduler fetched for the product; text encoder fetched only by the embedding script

- The revision is pinned by its 40-character SHA in the recipe's `sources.py` (`SOURCE`), with the
  sha256 of `pipeline_moss_image.py`, because it is `trust_remote_code` code that the recipe imports
  only after the hash is checked.
- The recipe dependency group `moss` uses the shared venv's existing pins (diffusers 0.39.0,
  transformers 5.14.1); they satisfy upstream's requirements (diffusers ≥ 0.38, transformers ≥ 5.3).
- The VAE is not MOSS's own: `sources.py --vae-origin` compares it tensor by tensor with the FLUX.2
  small decoder and the FLUX.2 VAE, and `tools/export-recipes/moss/THIRD_PARTY_NOTICES.md` records it
  as its own section beside MOSS-Image's Apache-2.0 (the licence review itself is the pre-release
  review of ADR 0065 decision 7).

Why: the same provenance discipline as ADR 0121 decision 1; one venv cannot hold two pins. Protects:
anyone rebuilds the same references from the same revision; Wan and Anima fixtures do not move.

### 2. The DiT is one S-shaped graph `dit`; concatenation by `where`, head over all S rows

- Inputs: `latent_tokens [1,S,128]` (text rows zero), `text_context [1,S,3072]` (image rows zero; a
  resident input set once per generation and branch), `is_image [1,S,1]` (bool), `rope_cos` /
  `rope_sin [1,1,S,128]`, `timesteps_proj [1,256]`. Output `[1,S,128]`; the host keeps the first N
  rows.
- In the graph: `x = where(is_image, video_embedder(latent_tokens), text_context)`, then the 42 blocks,
  the final norm, head modulation and `final_linear_video` over all S rows (all row-wise).
- The image RoPE's dependence on T is host-side: the table assembly picks row T of the f-axis base
  table for every image token.

Why: the IR cannot write `N + T` or slice a symbolic axis; `where` with a bool mask has the Wan2.2
precedent (ADR 0121 decision 3). The extra `video_embedder` / head rows cost below 1e-4 of a forward.
Alternatives: sum-of-symbols dimensions (a core change); text padded into buckets with a key mask
(upstream does not pad; a8 and online reject masks).

Protects: no IR change; one symbol per graph (ADR 0034).

### 3. The text refiner is a separate graph `dit_text` (symbol T), run once per branch per generation, always in f32 compute; the time MLP and both modulations stay in `dit` at M = 1

- `dit_text`: `text_embeds [1,T,4096]`, `rope_cos` / `rope_sin [1,1,T,128]` → `[1,T,3072]`
  (`text_embedder.proj` + two refiner blocks). The host reads the output back once (at most 50.3 MB at
  T = 4,096) and writes it into the text rows of `text_context`.
- Every preset runs `dit_text` with f32 compute (no a8): its input is an LLM hidden state after an
  RMSNorm, whose channel outliers per-token int8 does not tolerate well (the umT5 a8 precedent —
  inference for this model; the measured output has one outlier channel near 1,800 against a
  median channel maximum near 14), and it runs twice per generation (0.48 GFLOP per token against
  9.51 for the `dit` linears — arithmetic).
- The graph is traced with T ≥ 2 while the entry accepts T = 1, and no CPU path runs the IR, so the
  GPU check MUST include a `dit_text` case at T = 1. Its r gate is opt-in (upstream parity), so the
  default lane pins it with a sha row (`i8-dit_text-fridge-note-t1`, decision 11).

Why: the refiner has no modulation, so its output does not depend on the step (`:93-123`); the
irodori `dit_context` split is the precedent.

### 4. The negative branch is a second graph `dit_skip22` that borrows every weight of `dit`

- Traced from the same module without block 22; every initializer is a shared declaration
  (ADR 0096 decision 2, the ADR 0134 form), no weight bytes.
- Positive runs on the `dit` Session (lender), negative on the `dit_skip22` Session (borrower). Each
  Session sees one shape per generation; the check is `planBacking.buildCount` staying at one per
  Session over a generation.
- Accepted guidance settings: the upstream default (CFG with skip 22) and guidance 1 (no negative
  branch; only `dit` opens). Disabling the skip, arbitrary skip lists and SLG are not accepted.

Why: the shape alternation of every step disappears structurally, and the skip does not compute
block 22. A runtime gate in one graph would compute block 22, add a graph input and need
`planBackingBudgetBytes` sized for two backings.

### 5. Stage lifetimes — `dit_text`, then the DiT pair, then the VAE; each closed before the next opens

- Order: `dit_text` Session → close → `dit` + `dit_skip22` Sessions → close → the three VAE graphs.
  Each stage closes everything it opened, also when it fails, and waits for the release of the
  destroyed buffers before the next stage's check (Dawn returns them only when a later submit
  completes).
- Before a stage opens, `fitsHeadroom` (ADR 0112, a trial allocation) checks the device budget and
  fails loudly otherwise, naming the stage, the bytes and the size; generate also checks the largest
  of the three needs once at its start, with nothing open.
- The bytes follow Anima's form: the runtime estimate (`estimate` with the device limits, the
  stage's symbol bindings and the preset's knobs) → `peakAccountedBytes` + `maxPartBytes` + margin.
  `dit_text`: one estimate at T = max(T₊, T₋). The DiT pair: the sum of the `dit` (S₊) and
  `dit_skip22` (S₋) estimates — both are open at once — with the borrowed weights counted once
  (the borrower's estimate counts no weight), plus the two resident text contexts. The untiled VAE:
  the sum of its three graphs' estimates (h, w and S = h·w), the three Sessions being open together
  as in the Wan VAE encoder, plus the three hand-over residents. The added residents make the needs
  upper bounds.
- The estimate leaves the i8a8 linears' quantised intermediates out; the margin has to cover them,
  which stage 5 measured (Results).
- The largest area that fits a 12 GiB device is not a constant of the code: it is an observation
  recorded in `docs/limitations.md` (decision 9 for why).

Why: the untiled VAE peak (decision 8) and the DiT pair (4.52 GiB of weights — 4,849,617,932 B,
research §2.4 — plus two backings and two text contexts) do not both fit 12 GiB at the larger
accepted areas. The ADR 0114 pitfall (an open condition Session doubles the output slot) applies to
`dit_text`.

### 6. `situ` is written with existing ops; a fusion is a perf-ledger candidate

- `div` / `tanh` / `sigmoid` / `mul` as upstream writes them; `h / 25` stays a division so that the
  graph matches the CPU reference op by op (on the GPU, WGSL f32 division is not correctly rounded —
  2.5 ULP — so this is not a GPU bit-identity argument).
- No `situ` op (ADR 0059: exactly composable from the vocabulary). Its fusion is filed as perf-ledger
  F-10 from the stage-5 profile (about 8 % of a practical 1024² forward — research §6).

### 7. GroupNorm enters the vocabulary as a Core ATen layer op `group_norm`, two-pass per group

- Core ATen `native_group_norm` → op `group_norm` (a ledger NOTE in `docs/op-vocabulary.md`, like
  `upsample_bilinear2d`). The exporter takes the **preserved** `aten.group_norm`
  (`PRESERVED_OP_PREFIXES` 12 → 13): decomposed, `nn.GroupNorm` becomes `native_group_norm`'s three
  outputs + getitem with `HxW` the product of the spatial symbols — neither has an IR form (the
  `layer_norm` case). No existing export changes (a GroupNorm failed before as an unsupported op).
- Contract: `x[N,C,…]` rank ≥ 2 (C static, trailing axes may be symbolic, no axis of length 0),
  `weight[C]` / `bias[C]` (arity fixed at 3 — ones / zeros synthesized for a module without affine;
  bias-only fails), attrs `num_groups` (dividing C — a shape-rule check) and `eps` (the normalisation
  ops' rule), population variance, f32 only, not in `WEIGHT_SLOTS`.
- Kernel (`kernels/group-norm.ts`): a group is a contiguous row of `(C/G)·inner` elements; five
  dispatches — partial sums per 1,024-element chunk (each thread adds 4 elements at fixed offsets,
  then a 256 → 1 halving tree), the per-row total (strided per-thread sums of the partials, the same
  tree) → mean, the same two for the squared deviations → rstd, then the grid-stride affine
  `(x − mean)·rstd·w[c] + b[c]`. **Two passes** (mean, then the sum of squared deviations — the
  `layer_norm` discipline), with the tree fixed by the shape (determinism). `1/length` is rounded on
  the host (the shader does not divide). Two node temporaries (partials `rows·chunks` f32, stats
  `2·rows` f32), which the runtime estimate counts, as it counts argmax's split partials.
- Check (ADR 0110's four points): an f64 band, an A/B against the decomposition
  reshape → layer_norm → mul → add on small shapes, structural bit identities, and inverse fault
  injection — including a one-pass `E[x²] − E[x]²` on inputs with a large mean offset (+100), which
  the band MUST see, and a group near the cap (16.8M elements, past the 65,535-workgroup grid-stride
  boundary).
- GroupNorm–SiLU fusion is not added: it lowers the VAE's held peak, not its headroom need (the
  estimate counts no fusion), and the row-block budget of decision 8 already leaves the VAE stage
  room at the cap. It is filed as perf-ledger F-12.

Why: an exporter decomposition cannot write the reshape `[32, C/32·h·w]` (a product of symbols) in
the symbolic VAE graphs, and would stream every full-size intermediate several times. One group holds
up to about 17 million elements at the cap, where the one-pass form cancels catastrophically.

### 8. The FLUX.2 decoder runs untiled, as three symbolic-length graphs around the mid attention; the accepted area is capped

- A VAE-only series `moss-image-sft-f32-dyn` (f32 storage — upstream decodes in f32; the VAE does
  not change with the DiT's storage, and the DiT's writer replaces its series directory as a whole):
  `vae_decoder_pre` (symbols h, w: `post_quant_conv`, `conv_in`, mid resnet 1) → `vae_decoder_attn`
  (symbol S = h·w: mid attention) → `vae_decoder_post` (h, w: mid resnet 2, four up blocks,
  `conv_norm_out`, SiLU, `conv_out`), handed over through resident tensors read as `[C,h,w]` /
  `[C,S]` in one batch (the Wan2.2 VAE encoder form, ADR 0121 addendum 2026-10-09). The BatchNorm
  restoration's vectors are the asset `latent_batch_norm` of `vae_decoder_pre` (`mean` and
  `std = (running_var + eps).sqrt()` as upstream computes it; the host multiplies and adds).
- `Upsample2D` is rewritten in the recipe into the reshape / expand form of
  `_shared/vae_rank4.repeat_2x` (for ×2, `floor(o/2) = floor((o+0.5)/2)` for every integer o, so
  nearest and nearest-exact coincide), which the fusion rule `upsample2x` matches; the mid attention
  is `AttnProcessor2_0`'s statements on `[1,C,S]`.
- Host: BatchNorm restoration, unpatchify (128 → 32 ch, 2×), clamp to `[-1, 1]` (NaN kept; the stage
  then fails loudly on any non-finite value).
- **The area cap** is derived from the graphs: every symbolic value must be `K·h·w` (pre / post) or
  `K·S` (attn); the largest K is 49,152 B per VAE cell (`[1,192,8h,8w]` f32 — 768 B per pixel), so
  the cap is ⌊binding / 49,152⌋ · 64 px — 2,796,160 px at a binding of 2,147,483,644 B. The
  acceptance check refuses a larger area with a `ModelInputError` naming the limit and the binding.
  Whether the DiT pair (decision 4) and the VAE stage fit VRAM is not an acceptance limit: the
  per-stage check of decision 5 answers it per device and fails loudly.
- **The mid attention's row-block budget**: the `vae_decoder_attn` Session runs with
  `attentionRowBlockMaxBytes` 256 MiB, and the stage's need passes the same value to that graph's
  estimate (`MOSS_VAE_ATTENTION_ROW_BLOCK_BYTES`, `vae-stage.ts`; no runtime change). Values do not
  change with it (ADR 0124). Why: with the practical preset the VAE was the largest stage need at the
  cap (11.41 GiB, admitted by an idle 12 GiB device with about 0.5 GiB to spare); with the budget it
  is 9.88 GiB (Results). Below about 8,200 attention positions (about 0.52M px) the budget is not
  reached.
- Anima's distribution form is always tiled (ADR 0038 §4, the Anima section); this family does not
  follow it. The asset stays independent of resolution, which is what §4 protects. A tiled path above
  the cap is a later, separate decision.

Why: upstream decodes untiled, and GroupNorm makes tiles diverge in statistics across the whole image,
not only at seams; diffusers' `tiled_decode` exists but is not upstream's default path. Upstream's
recommended 1M / 1.5M / 2M pixels lie below the cap of a 2 GiB binding; its stated 4M-pixel
extrapolation does not. A fixed area constant measured on the 12 GiB machine would refuse feasible
areas on larger devices and admit infeasible ones on smaller ones, and free VRAM moves with the other
processes on a shared GPU. Rejected ways to lower the VAE's need: the three Sessions opened one at a
time (leaves decision 5's "open together" and the hand-over form), a GroupNorm–SiLU fusion (lowers
the held peak, not the need), fusion-aware estimates (a runtime change).

### 9. Accepted requests

One acceptance function (`planMossGenerate`, `packages/models/src/moss/generate.ts`) runs before
any GPU work; every rejection is a `ModelInputError` that names the limit or the missing input:

- Width and height: positive multiples of 16 (upstream's own check, `:369`), the area up to the cap
  of decision 8, and the longest side at most 65,536 px (4,096 latent cells) — an explicit acceptance
  limit: the rope base tables' `h` / `w` axes hold 2·4,096 − 1 rows, one per half-integer position of
  a side of up to 4,096 cells. Upstream has no ceiling, but with the area cap a longer side would
  mean a short side under about 43 px, which no use needs. A longer side MUST be rejected with an
  error that names the limit. Upstream's recommended and extrapolation ranges (1M / 1.5M / 2M
  pixels; 0.25M to 4M) are guidance in `docs/limitations.md`, not a check.
- Text: in one of two forms since the Addendum of 2026-10-31 — prompt strings (that addendum has
  their checks) or encoder outputs. The encoder outputs are f32 `[T, 4096]` with 1 ≤ T ≤ 4,096 (upstream
  requires at least one token, `:245-246`), checked for the array type (`Float32Array` — also at run
  time, for callers outside the types), shape, length and every value finite (a non-finite
  embedding would otherwise surface as non-finite latents after the whole DiT stage). The positive
  prompt's embeddings are required, the negative's are required while guidance ≠ 1 and refused at
  guidance 1, where they would have no effect (the Wan and Anima rule; upstream ignores them).
- Knobs: steps a positive integer (the effective count follows upstream's de-duplication), guidance
  finite and ≥ 0, shift finite and > 0 (both also finite in f32, since the host multiplies by their
  f32 roundings), and a shift not so small that the requested σ grid is not finite.
- Initial noise: exactly one of `seed` (a non-negative safe integer) and `latents` (a finite f32
  `[128, H/16, W/16]`).
- Nothing else is refused: no step cap (the GPU work is bounded by the snapped schedule — at most
  1,000 steps up to about 2·10⁶ requested steps, about 1,100 even at 10⁷; the rest is a host loop
  linear in the count, about 0.25 s at 10⁷), the whole guidance range, any area under the VAE's
  cap. Whether the stages fit the free VRAM is decision 5's check.
- **The tiny-shift refusal narrows upstream on purpose**: below about 2.98·10⁻⁸ `f32(shift − 1)` is
  −1 and the first requested σ is infinite; upstream snaps that infinity to the end of the grid and
  runs (2 steps at shift 10⁻⁹ give σ = [0.999, 0.001]), a schedule its formula did not produce.
  Upstream's degenerate behaviour is not copied (ADR 0005 addendum 2026-10-11), and no use needs it.
- **Past about 2·10⁶ requested steps the schedule is not monotonic** (at 5·10⁶ 989 snapped indices
  with 6 decreases, at 10⁷ 1,099 — more than the 1,000 training points — with 61; steps with
  dt > 0 — research §4.1): that is upstream's own `set_timesteps`, reproduced, not refused;
  `docs/limitations.md` records it.
- **No defaults at this entry**: upstream's (1024×1024, 50 steps, guidance 3, shift 5 — and its
  empty negative prompt, the Addendum of 2026-10-31) belong to a manifest's `pipelineConfig`, so
  every knob of `planMossGenerate` is explicit and the defaults keep one owner. Since 2026-11-01
  the distribution's manifest declares them and the public pipeline fills the fields a request
  leaves out (Addendum of 2026-11-01, B4). Rejected: upstream's `__call__` defaults as family
  constants — a second owner to retire.
- The seeded noise is splitmix64 + Box–Muller (`packages/models/src/moss/random.ts` — a deliberate
  copy of Wan's core, ADR 0118 decision 5), **not torch's `randn`**: a deliberate departure from
  upstream. The same seed does not give upstream's image; every comparison with upstream injects the
  reference's own initial latents through `latents`.

### 10. Online int8 attention and `N % 4` — a key valid length in the online kernel only

- The online kernel runs any key length N ≥ 1: for `N % 4 != 0` the packed Vᵀ rows are `ceil(N/4)`
  words, the padding bytes are 0 and keys ≥ N take the identity −3.0e38 already used for the last
  tile (ADR 0132 decision 4), so the padding never reaches O. q / k pack along D and need no
  padding. The runtime record is the ADR 0132 addendum of 2026-10-27 (`attentionOnlineKeyWords`,
  the padded `quantize_rows`, the `…:kpad` kernel).
- Scope: the Vᵀ producer, the online PV phase, `attentionOnlineI8a8Eligible` and `attentionRoute`.
  ③PV i8a8 and `attentionPvI8a8Eligible` are unchanged, so Anima's route — which never takes the
  online kernel — stays as it is at every shape, and every `N % 4 == 0` shape keeps its key, WGSL,
  buffers and dispatches byte for byte.
- Independent check (before the self A/B, which compares the kernel with itself and could absorb a
  masking bug): shapes with `N mod 4` ∈ {1, 2, 3} (65, 4,106 — MOSS's negative S at 1024² — and
  203), D 64 and 128, batch ≥ 2, against the f64 exact attention over exactly N keys of the
  dequantised inputs (an oracle with no padding at all) and the TS mirror's two-lattice-step band;
  atol 0 where it applies (padding bytes 0 and 127 give Uint32-identical outputs); faults that MUST
  turn red (the mask bound moved to `ceil(N/4)·4`, a wrong row length, a dropped last word — with
  spike-V inputs, since at N = 4,106 the dense inputs' band barely sees them).
- With it, every MOSS forward on a device with an online form is fully eligible, and the practical
  preset uses the existing complete-eligibility census check. A device with neither online form
  (no subgroups and a 32 KiB workgroup-storage limit) falls back to the three dispatches.
- The estimate of a `"none"` preset needs the device's online-route facts
  (`EstimateOptions.attentionDevice`). `@karume/runtime` exports `attentionDeviceFacts` /
  `AttentionDeviceFacts` — the derivation Session construction uses — so a family's headroom check
  counts the route its Sessions take (ADR 0132 addendum of 2026-10-28). Rejected: copying the
  condition into the models package (an estimate could count another route the moment the runtime's
  condition changes); estimating without `"none"` (counts the three-dispatch S — at the cap a need a
  12 GiB device refuses although the pair fits).

Alternatives: the key valid length in ③PV i8a8 too — Anima's default preset then changes route and
output at its 4,608 shapes with `N % 4 != 0`, which needs those sha rows rewritten and a visual
review, and MOSS does not need it; ship the defined fallback, declare the expected route per branch
and shape in a new census check, and file the key valid length in the perf-ledger — half of the
forwards would never run the declared route at square sizes.

### 11. Numeric baseline and acceptance classes

| Tier                       | Baseline                                                                                                   | Check |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- | ----- |
| Port (export)              | upstream `MOSSImageTransformer.forward` itself, with only `video_embedder` and `final_linear_video` called on S zero-padded rows (M aligned); the same f32 fake-quant weights in one process | eager equivalence, bit-identical (concatenation, slicing, RoPE and row order run as upstream's code); `dit_skip22` against `forward(..., skip_layers=(22,))` with the same substitution; `dit_text` against upstream's `text_embedder` output (`:251`, a forward hook); upstream's unmodified `forward` is an observation (last-bit M dependence only) |
| Host functions             | upstream outputs recorded in stage 0, at two torch thread counts                                           | Uint32-identical: rope tables (reordered base tables), time sinusoid and training-grid σ (upstream tables — the asset below), requested-grid σ (a TS transcription of `torch.linspace` in f32 with one rounding per element, the FMA the CPU kernel compiles to — an implementation fact, not a torch guarantee), snapping, de-duplication, timesteps, patchify / head rows, BatchNorm restoration, unpatchify, Euler, CFG; the requested grid at 202 step counts and at eleven shifts other than 5 × eight counts, CFG at nine guidance scales; the generate path's own sampler loop, fed upstream's recorded DiT outputs, reproduces upstream's latents of both steps |
| Port (GPU, reference path) | layer-by-layer CPU f64 reference (ADR 0121 decision 7) from the containers' dequantised weights            | r bands derived from measurement per group (`dit_text`; the pair at 256²; at 1024²), six band cases and four acceptance cases each; `dit_text` also a per-channel r (below); faults outside the band by a margin of 2; opt-in (upstream parity) |
| VAE                        | upstream f32 untiled decode                                                                                | ADR 0118 decision 8's upstream check of a VAE (f32 reference, m of the clamped image, worst of three band cases × 5 per size, acceptance cases separate); the GroupNorm kernel's four points (decision 7) hold what this band cannot see (a one-pass variance); the whole VAE at the cap (opt-in) |
| End to end (2 steps)       | upstream `pipe(...)` on CPU in f32 with the DiT's weights rounded to the series' i8 values (checked bit-identical to the containers) and upstream's f32 VAE (`moss.few_step_ref` — the Wan2.2 precedent) | m of the final latents and of the clamped image, per size (256², 384×256), the worst of three band cases × 5, held by two acceptance cases; faults (prompts swapped, guidance 3.3 and 3.003, shift 3, another seed, the first step at the next training-grid point) outside 2 × the band. Deviation from ADR 0118's addendum: the band cases and the acceptance cases are measured in one run; the rule leaves nothing to adjust |
| Practical preset           | the reference preset, same machine, same weights                                                           | self A/B check (ADR 0110 decision 5) per branch and per graph: the step-1 outputs of `dit` and `dit_skip22` and the latent after step 1 of upstream's 50-step request; ceilings = measured relRMS × 2 per size and point; floor 0 and at least one bit different; census exactly the i8a8 linears (258 / 252), exactly the online dispatches (attention nodes × query-row windows), none of the three dispatches, `:kpad` exactly where S % 4 ≠ 0; `dit_text` bit-identical between the presets; two practical runs bit-identical; faults: no knob → floor, another seed → determinism and ceiling, `attentionScoreStorage` dropped → census, the linear or attention node count ± 1 → census |
| Regression                 | —                                                                                                          | sha256 rows per environment key (ADR 0106) of the seed path's clamped f32 image, the row ID led by the preset's name: the default lane at 256² and 384×256 × 2 steps at guidance 3 for both presets, guidance 1 at 256² for the reference preset, and `dit_text` at T = 1; 1024² × 50 steps opt-in for both presets |
| Quality                    | —                                                                                                          | the maintainer's visual review of `f32+dit8` against `f32+dit8-a8-attn8-online` side by side (seeds 42–45 × 4 prompts, 1024², 50 steps, guidance 3, shift 5): passed — the practical preset's quality acceptable (Rulings; [research](../research/2026-10-28-moss-image-dit-vae.md) §5.9) |

- The two upstream tables of the host functions — the training-grid σ and the time sinusoid — are
  the asset `training_grid` of the `dit` container (role `training-grid`; safetensors with
  `sigmas` `[1000]`, `timesteps` `[1000]` and `time_sinusoid` `[1000, 256]`, all f32), next to
  `rope_base`. `moss.export_dit` takes the values from upstream code through `moss.host_fixture`'s
  own functions (one path), and its `write` checks the written asset bit-identical to the stage-0
  host record (a missing record fails the write). The host indexes the tables by the training-grid
  index and never evaluates the sinusoid: every timestep upstream can pass is one of these 1,000.
  Why an asset: the same reason as `rope_base` (ADR 0034 decision 3 — a TS transcription of the
  float64 sinusoid can split the last bit), and the TS side reads it with the container it already
  opens, so table and weights cannot come from different exports. Rejected: a TS transcription
  (last-bit risk), a separate file beside the series (a second artifact to pin and keep in step).
- The layer-by-layer reference stores its f64 output as two f32 parts (the f32 rounding and the f32
  rounding of the rest — their double sum is the f64 value within 2⁻⁴⁸ relative). Why:
  `dit_text`'s CPU f32 error is only 1–2 ULP at its outlier channel, so a reference rounded to f32
  is off by as much as the denominator of r. For the same outlier, `dit_text` also has a
  per-channel r (each channel's error divided by max(its max |value|, the median channel's)), banded
  the same way; both bands must hold. The tests refuse goldens of other containers (the part-0
  sha256 each golden records) and of other cases.
- The A/B floor stays 0 although one knob does give clearly less difference (without
  `linearCompute` the difference is about half of the bundle's): ADR 0120 decision 4 allows a floor
  above 0 and does not require one, and that fault is seen exactly by the census (no
  `linearCompute="a8"` variant — red), as for Anima and Wan2.2. A floor derived from a fault's
  configuration would add one more machine-dependent value.
- Agreement with the theoretical value (ADR 0110 decision 5-3), in Anima's form and to the order of
  magnitude only: the a8 row rounding (peak/rms)/(127·√12) gives per-layer relRMS of the 1e-2
  order, as the branches' measured 1.6–1.8e-2; MOSS's per-layer peak/rms is not measured.
- The 50-step latent after step 1 is a dependent observation point: its difference is the CFG of
  the branches' differences scaled by |dt| = 0.004, and another seed turns it red by the two noises'
  distance, not by a DiT error; the branch points carry that fault on the DiT. At 2 steps the latent
  after step 1 is about x₀ and another seed moves it less than its ceiling, hence the 50-step
  schedule for that point.
- sha rows MUST be written only by a run that also runs the upstream parity (`KARUME_UPSTREAM_PARITY=1`
  with its goldens — the tests refuse otherwise), and kept only when the whole run is green. A row's
  ID starts with the preset's name; `i8-dit_text-fridge-note-t1` is the exception, because no preset
  changes `dit_text` (decision 3), so its ID names the weights only.
- The official bf16 run is neither a baseline nor a standing record (ADR 0121 decision 5). A full f64
  comparison against upstream is not used: the f64 DiT alone is 40.7 GB (arithmetic). The stage-0
  environment (torch version, thread counts, MKL variables) is recorded (ADR 0121 decision 7 note on
  weight alignment).

### 12. Text embeddings — an asset of fixed prompts written by a CPU script over upstream's `encode_prompt`; no public surface until the text encoder

- `moss.text_embeds` calls `MOSSImagePipeline.encode_prompt` unchanged on CPU in bf16 and writes one
  F32 tensor `[T, 4096]` per prompt (`last_hidden_state`, not the refiner output) plus a single-key
  JSON metadata entry (the Wan rule for byte-stable safetensors).
- Prompts: the HF README example (a handwritten note in Chinese), two USERGUIDE examples and the empty
  negative. The Chinese text lives in one tracked file (`tools/export-recipes/moss/prompts.py`)
  covered by an exact-path entry with reason and `maxLines` in the language gate's allow-list
  (ADR 0130; `tools/lang/scan-japanese.ts` counts Han).
- No public subpath (`@karume/models` barrel and exports), no manifest and no HF publication until
  the text encoder exists: the family lives in `packages/models/src/moss/` behind an internal entry.
  Rejected: the Wan2.1 stage-1 form, a public string API restricted to the precomputed prompts —
  little user value, and a public promise made before the text encoder is designed.
- Since 2026-10-31 the text encoder exists and is wired (Addendum); publication still waits — it is
  the manifest step (the maintainer's ruling 53 a kept this decision in force). The manifest step
  was taken 2026-11-01 — a distribution, a local mirror and the subpath, without Hugging Face
  publication (Addendum of 2026-11-01).

### 13. Presets and the default

`packages/models/src/moss/preset.ts`; the names follow ADR 0074's grammar (the VAE's f32 is the base
storage and the DiT overrides it to i8 — the name `i8` alone would claim an i8 VAE):

| Preset                     | DiT weights | `dit` / `dit_skip22`                                                             | `dit_text`  | VAE |
| -------------------------- | ----------- | -------------------------------------------------------------------------------- | ----------- | --- |
| `f32+dit8` (reference)     | i8          | f32 compute (no knob)                                                            | f32 compute | f32 |
| `f32+dit8-a8-attn8-online` | the same i8 | `linearCompute: "a8"`, `attentionCompute: "a8"`, `attentionScoreStorage: "none"` | f32 compute | f32 |

- The preset is an explicit field of the stage and generate states (`MossDitStageState.preset`,
  `MossGenerateState.preset`); there is **no default in code** — the entries refuse a state without
  one. Since 2026-11-01 the default is the distribution manifest's `defaultQuant`, resolved by
  the public pipeline (Addendum of 2026-11-01, B3).
- The knobs go to the pair's two Sessions and to their estimates as one value, with the device's
  online-route facts (decision 10). `mossAcquireGpuOptions(preset)` asks for subgroups when available
  for the online preset (never required — ADR 0128 C1); without them the kernel takes its portable
  form (bit-identical).
- **The default departs from ADR 0128 decision 1 for now**: that procedure makes the reference
  preset `f32+dit8` the provisional default and the practical preset the default once the self A/B
  check and the visual review pass. This family holds no provisional default instead, because no
  manifest exists yet whose `defaultQuant` would own it (decision 12) — a default in code now would
  be a second owner to retire. The self A/B check passes (Results) and the visual review passed
  (Rulings), so under ADR 0128 decision 1 the family's default is the practical preset
  `f32+dit8-a8-attn8-online`; the manifest carries it as its `defaultQuant` when it comes with the
  text encoder and publication. Until then no default is in code — the standing departure, an
  implementation decision (Open points).
- ADR 0128's C1–C3 are checked from a manifest, and there is none, so they cannot be checked yet.
  What holds in `preset.ts` already: C2's sibling relation (`f32+dit8` has the practical preset's
  weights and no activation quantization) and C1's premise (neither preset requires a feature —
  above). C3 needs a `label` and a `description` per preset, which the family's internal table does
  not carry; they come with the manifest.
- No original-weights preset now: ADR 0128 decision 4's table carries a row for MOSS-Image (the DiT
  is bf16 upstream, the VAE f32; since 2026-11-01 the row records the distribution's state — the
  bf16 sets are not exported). Decision 4's rule (export practicality, no size cap) is unchanged,
  so the row names no size or run-feasibility reason.

### 14. Stages

| Stage | Goal                                                                                                                                            | Check                                                                                                                                         | State |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| 0     | branch; pinned fetch; VAE origin; embedding script (T₋ recorded); block references at 256² / 256×384; host-function fixtures at two thread counts; VAE references | hook-assembled run ≡ upstream `pipe(...)` with injected latents; RAM and time recorded                                                        | done  |
| 1     | DiT export: `dit`, `dit_text`, `dit_skip22`, `rope_base`, i8 series                                                                             | eager equivalence bit-identical against the M-aligned upstream `forward` (also with `skip_layers=(22,)`) and the upstream refiner output; no S-dependent constants; all-shared skip graph; census | done  |
| 2     | DiT stage in `packages/models/src/moss/`, lane `moss`, the asset `training_grid`                                                               | host functions Uint32-identical; r bands; `dit_text` at T = 1 on the GPU (MUST — decision 3); `planBacking.buildCount` one per Session per generation | done  |
| 3     | op inventory; `Upsample2D` rewrite; `group_norm` op; three VAE graphs; host BatchNorm restoration and unpatchify                                 | census of `upsample2x`; kernel four-point check with offset inputs; GroupNorm values at a group near the cap; VAE band vs upstream; VAE-stage peak at the cap | done  |
| 4     | sampler and end-to-end                                                                                                                          | σ columns bit-identical; host CFG + Euler bit-identical from recorded DiT outputs; 2-step parity bands; sha rows; 1024² × 50 steps completes (opt-in) | done  |
| 5     | presets, a8, key valid length (decision 10), VRAM and speed on the 3080 Ti                                                                     | key-valid-length check against f64 and the TS mirror, pad-content atol 0, fault injection; self A/B check per branch and graph with the existing census check; timing before and after decision 10; VRAM at 1024², 2M pixels and the cap; visual review | done  |
| 6     | docs; when the branch is taken in: rebase, ADR number                                                                                           | `vp check`, the language gate, the family and core lanes; after the rebase the full verification                                              | docs done; rebased onto `main` and numbered 2026-10-31; taken into `main` 2026-11-01; the full verification after the rebase pending |

## Results (stages 2–5, 2026-10-27 to 2026-10-28, RTX 3080 Ti 12 GiB, Node + Dawn)

The figures and their conditions are in the [research record](../research/2026-10-28-moss-image-dit-vae.md);
the ones the decisions rest on:

- **DiT port** (§2): bands of r 56 (`dit_text`; per-channel 37), 22 (256²), 25 (1024²); every
  acceptance case inside, every fault outside 2 × its band. The borrower's estimate counts no
  weight; the 1024² pair needs 10,070,775,568 B with the weights counted once.
- **VAE** (§3): GroupNorm band 5.5 in its error unit (theoretical depth bound ≈ 54), the one-pass
  fault at 226; the decoder against upstream m bands 2.0e-5 (256²) and 3.3e-5 (512×768), m 1.272e-5
  at the cap.
- **End to end** (§4): 2-step bands 5.3e-5 / 1.3e-4 (256², latents / image) and 1.2e-4 / 6.5e-5
  (384×256); the closest fault (guidance 3.003) at 8.0–18.4 × the band.
- **Self A/B** (§5.2): relRMS at 1024² `dit` 1.8482e-2, `dit_skip22` 1.5683e-2, latent 2.4677e-4
  (ceilings 3.7e-2 / 3.2e-2 / 5.0e-4); at the cap 1.7865e-2 / 1.6198e-2 / 2.3550e-4 — no growth with
  S. Census: all 42 + 41 online dispatches `:kpad` at 1024², the positive 42 at the cap, no
  three-dispatch attention.
- **Speed** (§5.3, §5.4): decision 10 takes one 1024² forward from 2,176–2,212 to 1,667–1,697 ms
  (`dit`, median) — about −22 %. Upstream's default request (1024² × 50 steps): `f32+dit8` 574.8 s,
  `f32+dit8-a8-attn8-online` 238.0 s right after it and 216.3 s from a cooled GPU.
- **VRAM** (§5.5): every stage held less than its need — e.g. at 1024² the reference pair 9.17 of
  10.07 GB, the practical pair 6.43 of 7.33 GB; at the cap the practical pair 9.04 of 10.11 GB, so the
  margin covers what the estimate leaves out. With the row-block budget the VAE at the cap needs
  10,608,156,072 B (was 12,252,394,824 B) and holds 7.49 GB; the image is Uint32-identical with and
  without it.
- **What binds on 12 GiB** (§5.6, observations of one device at one time): the practical preset's
  DiT pair passed squares up to 2064² (above the VAE's cap), so the cap binds it; the reference
  preset is refused at the cap and at some smaller squares (1616², 1632²) while others pass (1536²,
  1568², 1600², 1664²) — the need is not monotonic in S — so its DiT pair binds first. Upstream's
  2M-pixel example (1456², 2 steps) and its default 1024² × 50 steps completed with both presets.

## Alternatives considered

- **Sum-of-symbols dimensions in the IR** — a core change across both grammars for one family.
- **Text padded into buckets with a key mask** — upstream does not pad; a8 / online reject masks.
- **Refiner inside the DiT, every step** — needs `cat` / `slice` across symbols; recomputes a
  step-independent value 100 times.
- **A runtime gate for block 22 in one graph** — computes block 22, adds an input, and puts two DiT
  shapes into one Session.
- **Splitting the DiT at block 22 into three graphs** — three Sessions per forward and the weights in
  three containers.
- **Modulation on the host or in a per-step graph** — no measurable saving.
- **A `situ` op / a fusion rule before measurement** — not needed for correctness.
- **GroupNorm decomposed by the exporter** — the group reshape is a product of symbols; several
  full-size passes per norm.
- **One-pass GroupNorm statistics (`E[x²] − E[x]²`)** — cancellation over groups of millions of
  elements.
- **Always-tiled VAE** — GroupNorm statistics change per tile; upstream decodes untiled.
- **Tiled path above the cap now** — would admit areas above the cap, but is a second path with its
  own checks before it is needed, and the meaning of the image changes at the cap.
- **The key valid length in ③PV i8a8 too** — changes Anima's default-preset output at its
  `N % 4 != 0` shapes, which MOSS does not need.
- **Accepting the `N % 4` fallback with an expected-route census** — half of the forwards never run
  the declared route at square sizes.
- **Eager equivalence against a hand-written S-form wrapper** — the wrapper would re-transcribe the
  concatenation and slicing and so could not catch errors in them.
- **A full f64 comparison against upstream** — 40.7 GB of f64 weights per copy.
- **Rebuilding backings above the budget instead of capping the area** — an unmeasured cost per
  forward and a second memory regime.
- **Publishing with the precomputed route only** — little user value; a public promise made before
  the text encoder is designed.
- **A bf16 preset gated on whether it completes on 12 GiB** — not ADR 0128 decision 4's rule.
- **The official bf16 run as baseline** — bf16 sets the floor (ADR 0121 decision 5).

## Consequences

- New family `moss` (internal — no subpath, no manifest, no pipeline name yet), lane `models:moss`.
- Runtime: one new op (`group_norm`); the key valid length of the online kernel (decision 10); the
  export `attentionDeviceFacts` / `AttentionDeviceFacts`. No change to the IR, the executor,
  `session-build.ts` or the GEMV path.
- The VAE gains a second form (untiled, three symbolic graphs) beside Anima's and Wan's tiled forms.

### Affected files

| Area          | Files |
| ------------- | ----- |
| recipe        | `tools/export-recipes/moss/` (`sources.py`, `prompts.py`, `text_embeds.py`, `reference.py`, `host_fixture.py`, `records.py`, `dit_patch.py`, `export_dit.py`, `dit_reference.py`, `vae_patch.py` with the `Upsample2D` rewrite, `vae_reference.py`, `export_vae.py` (the VAE-only series `moss-image-sft-f32-dyn`), `pipeline_ref.py`, `few_step_ref.py` (the 2-step end-to-end reference with the series' weights), `THIRD_PARTY_NOTICES.md`, `tests/`), `tools/export-recipes/pyproject.toml` (group `moss`), `tools/uv.lock`, `_shared/vae_rank4.py` (`repeat_2x`, the body `nearest_exact_2x` delegates to), the shared registration tables `tests/test_graph_names.py` and `tests/test_staged_publication.py` |
| exporter      | `ops.py`, `shapes.py`, `aten_handlers.py` (the preserved `aten.group_norm`), `convert.py` (`PRESERVED_OP_PREFIXES`), `golden_models.py`, `goldens.py` (the spec), `pipeline.py` (a docstring count), `tests/test_aten_handlers.py`, `README.md` |
| runtime       | `group_norm`: `ops/{names,shapes,attrs,contracts}.ts`, `ops.ts` (re-export), `reference/ops.ts`, `runtime/recipe-builders/norm.ts`, `runtime/recipe-builder.ts` (the kind's case), `runtime/estimate.ts` (its two in-node temporaries), `kernels/group-norm.ts`; key valid length (the ADR 0132 addendum of 2026-10-27): `kernels/attention-online-i8a8.ts`, `kernels/quantize-rows.ts` (the padded form), `runtime/attention-route.ts`, `runtime/recipe-builders/attention.ts` (the Vᵀ producer and the online node), `runtime/session-types.ts` (JSDoc), `tune/attention-bench.ts`; the export (the addendum of 2026-10-28): `mod.ts`, `tests/fixtures/public-surface.json` |
| runtime tests | `group_norm`: `codegen_wgsl_test.ts` with `fixtures/wgsl/group_norm_{partial,total,apply}.wgsl`, `fixtures/op-contracts.json`, `fixtures/golden/group_norm_block/`, `ops_conformance_test.ts`, `reference_ops_test.ts`, `gpu_ops_test.ts`, `gpu_gridstride_test.ts`, `gpu_full_write_test.ts`, `gpu_group_norm_test.ts`, `helpers/{gpu_op_cases,gridstride_cases,op-tolerance}.ts`; key valid length: `gpu_attention_online_i8a8_test.ts`, `gpu_attention_online_session_test.ts`, `gpu_attention_permute_fusion_test.ts`, `gpu_attention_token_major_kernels_test.ts`, `gpu_i8a8_test.ts`, `attention_route_test.ts`, `runtime_fusion_test.ts`, `runtime_pipeline_census_helper_test.ts`, `tune_attention_bench_test.ts`, `codegen_wgsl_test.ts` with `fixtures/wgsl/attention_online_i8a8_{d128_sg32_obshd,d64_wg}_kpad.wgsl` and `quantize_rows_{r1w256,r8w32}_pad.wgsl`; `packages/e2e/tests/e2e_wan_ti2v_dit_test.ts` (a comment) |
| models        | `packages/models/src/moss/` (graph contracts, rope assembly, time table, text context, the DiT stage, sampler, seeded noise, VAE contract, host and stage, the acceptance check, the generate path, the presets and its README) — not in the barrel or exports (decision 12) |
| tests         | `packages/e2e/tests/e2e_moss_{dit,vae,generate,ab}_test.ts`, `packages/models/tests/moss_{dit_host,vae_host,sampler,generate_host,preset}_test.ts`, `packages/models/tests/helpers/{moss-dit,moss-vae,series-graphs}.ts`, `packages/models/tests/fixtures/references/moss.json`, `packages/test-support/src/lanes.ts` (lane `moss`) |
| tools         | `tools/lang/language-allow-list.ts` (the prompt file) |
| docs          | this ADR, the ADR 0132 addenda of 2026-10-27 and 2026-10-28, ADR 0128 decision 4's table, ADR 0040 (the online route's facts in its permute-absorption addendum), `docs/assets-layout.md`, `docs/research/2026-10-28-moss-image-dit-vae.md`, `docs/glossary.md`, `docs/GLOSSARY.ja.md`, `docs/ir-v2.md`, `docs/op-vocabulary.md`, `docs/quantization.md`, `docs/limitations.md`, `docs/backlog.md`, `docs/perf-ledger.md`, `CHANGELOG.md`, `.claude/ACTIVE_DESIGN.md` |

### Verification

As decision 11 and the stage table; plus the codegen snapshots and every other family's sha rows
unchanged, and the full verification after the rebase that takes the branch in.

### Added complexity and future obstacles

- **Added**: three DiT graphs (one bytes-less), three VAE graphs, a host-assembled text context, one op,
  a key valid length in the online attention kernel, one runtime export.
- **Held back**: no IR change; no mask path; no public surface; no tiled VAE path; no bf16 preset.
- **Possible obstacles**:
  1. The text encoder (the GDN scan of ADR 0135, D = 256 GQA attention) must produce `[T, 4096]`
     matching the entry of decision 9; embeddings from a GPU encoder will differ from the bf16
     upstream asset, so sha rows stay per route. (Met 2026-10-31 — the Addendum: the prompt form's
     sha row has its own ID.)
  2. Resolutions above the area cap need a tiled path (a separate ADR).
  3. Other skip settings or SLG need more graphs or a gate.
  4. Changes to the GEMV path on `wip/0.15` move the M = 1 linears' routes; MOSS's census counts by
     family, and its sha rows are rewritten only through `KARUME_REFERENCE`.

### Risks

1. The reference preset's DiT pair does not fit a 12 GiB device at the VAE's cap, and below it only
   some areas fit; the per-stage check refuses such areas per device (decision 5), and the largest
   areas tried are observations, not a promise.
2. A new GroupNorm kernel over groups of up to about 17 million elements (held by its four-point
   check and the large-group case).
3. a8 accuracy on this model is checked against the reference preset only (the self A/B check);
   the visual review found the practical preset's quality acceptable at 1024² × 50 steps for the
   four prompts reviewed; other sizes and step counts were not reviewed.
4. The requested-grid σ reproduces bit for bit with one rounding per linspace element — an
   implementation fact of the torch build the records were taken with, not a torch guarantee. A
   build without the contraction would still give upstream's snapped schedule for every count
   1–1,000.
5. `trust_remote_code` upstream code (pinned by revision and file sha256).

## Rulings

The maintainer ruled on these, each with its recommended option:

- **The design as first proposed** — decisions 1–7, 9, 11 and 13, the area cap of decision 8 and
  the prompt asset of decision 12 in their form before stage 2, approved as one bundle. What
  stages 2–5 settled on top of it is listed below, apart from the rulings.
- **Decision 10 — where the key valid length goes**: in the online kernel only; ③PV i8a8 and
  Anima's routes unchanged. Rejected: ③PV i8a8 too; the defined fallback with an expected-route
  census.
- **Decision 8 — how the VAE runs**: untiled, in three symbolic graphs with an area cap, and the
  VAE's need is lowered only when it decides what fits. Rejected: always tiled; a tiled path above
  the cap now.
- **Decision 12 — publication**: no public subpath and no HF publication before the text encoder;
  the family stays out of the models barrel and exports. Rejected: the Wan2.1 stage-1 form.
- **Decision 13 — the default preset** (2026-10-28): the visual review of `f32+dit8` against
  `f32+dit8-a8-attn8-online` (seeds 42–45 × 4 prompts, 1024², 50 steps, guidance 3, shift 5)
  found the practical preset's quality acceptable, so under ADR 0128 decision 1 the family's
  default is `f32+dit8-a8-attn8-online`. The manifest's `defaultQuant` applies it; no default is
  in code until then (the departure below).

Taken as the recommended option during implementation and reported to the maintainer (recorded in
the decisions; not rulings):

- the mid attention's row-block budget of 256 MiB (decision 8), applied under decision 8's rule
  for lowering the VAE's need: at the cap the VAE was the practical preset's largest stage need
  (11.41 GiB of 12);
- `@karume/runtime` exports `attentionDeviceFacts` / `AttentionDeviceFacts` (decision 10, ADR
  0132 addendum of 2026-10-28);
- a sha row's ID starts with its preset's name; `i8-dit_text-fridge-note-t1` is the exception,
  since no preset changes `dit_text` (decision 11);
- the runtime estimate counts `group_norm`'s two in-node temporaries (decision 7) — without them
  the VAE's need is short and the trial allocation too lenient;
- the estimate of a `"none"` preset takes the device's online-route facts (decision 10);
- the asset `training_grid` and the two-part f64 goldens with `dit_text`'s per-channel r
  (decision 11);
- the VAE's upstream band in ADR 0118 decision 8's form — three band cases per size, the ratio m
  (decision 11);
- the tiny-shift refusal, a negative embedding refused at guidance 1, no step cap and no request
  defaults (decision 9);
- no default preset in code until a manifest exists (decision 13 — the standing departure, Open
  points).

## Open points

- **No default in code until a manifest exists** (decision 13's departure from ADR 0128 decision
  1): closed 2026-11-01 — the distribution's manifest carries `f32+dit8-a8-attn8-online` as its
  `defaultQuant` and the distribution gate checks C1–C3 on the local mirror (Addendum of
  2026-11-01).
- Stage 6 after the rebase onto `main` (2026-10-31): the full verification. The rebase numbered
  this ADR (the placeholder replaced by a count-asserted script in every tracked file that cited
  it, in this file's name and in its title line; the recipe's placeholder check
  `tests/test_placeholders.py` runs from then on) and re-checked the perf-ledger IDs F-10 to F-12
  against `main` (no collision). Of the two texts both branches had rewritten, the online-route
  line of ADR 0040's addendum of 2026-10-26 merged as is; the online int8 attention entry of
  `CHANGELOG.md` had meanwhile been released unchanged in 0.15.0, so the key-length change went
  into [Unreleased] as a Changed entry.
- The text encoder: wired as the prompt form (Addendum of 2026-10-31). The manifest and its
  defaults followed (Addendum of 2026-11-01). Later, each its own decision: publication on Hugging
  Face, an original-weights preset, a tiled VAE path above the cap.

## Addendum (2026-10-31) — the prompt form: the text encoder wired (the maintainer's ruling 53 a)

The text encoder of ADR [0137](0137-qwen3_5-text-encoder.md) runs inside this family's internal
entry: `generateMoss` takes prompt strings. The maintainer ruled the narrow scope (53 a): **no
manifest, no distribution recipe, no local mirror, no public surface, no default preset** — the
caller opens the Qwen3.5 text series and its tokenizer asset as it opens the DiT and VAE series,
and decisions 9 (every knob explicit), 12 and 13 stay in force. Measurements:
[research 2026-10-31-moss-text-stage](../research/2026-10-31-moss-text-stage.md).

### A1. Two forms of the text, one per request

- **Prompt strings**: `prompt` and `negativePrompt`; the state carries the text series as `text`
  (the component, the tokenizer, the contract of A5). **Embeddings**: `positive` and `negative` as
  decision 9 has them — kept, because the parity checks and the A/B of upstream's embeddings against
  Karume's (ADR 0137 decision 5) feed the DiT through them.
- A request uses one form for both branches. Both forms (on one branch or across the two), or
  neither, is a `ModelInputError`; a prompt-form request whose state has no text series is the
  caller's assembly (a plain `Error`).
- The request type stays flat with optional fields and a run-time check, as `seed` / `latents` are:
  a JavaScript caller can pass anything, and the one acceptance function is the check.

### A2. No default negative prompt

Upstream's empty negative is one of its `__call__` defaults (`negative_prompt=""`), and decision 9
gives those to a manifest's `pipelineConfig` — a distribution owns its default negative (Anima's
manifest carries `pipelineConfig.defaults.negativePrompt`). A code default now would be the second
owner decision 9 rejects. So at guidance ≠ 1 the prompt form requires `negativePrompt`, as the
embeddings form requires `negative`; the message names upstream's default (`negativePrompt: ""`).
At guidance 1 a negative prompt is refused in both forms (it would have no effect). Rejected: `""`
as a family constant (Wan's in-code default negative) — a second owner to retire when the manifest
comes, and the two forms would then require different things.

### A3. Over-length prompts: the maintainer's ruling 50, through `overflow`

The tokenizer rejects a prompt over 4,096 tokens with its `ModelInputError`; the request's optional
`overflow: "truncate"` cuts both prompts as upstream does (dropping the template's tail). Exposed
because it is one field passed through to the tokenizer for both prompts, the tokenizer's own
rejection message names it, and without it upstream's behaviour for a long prompt would be
unreachable from this entry (the ruling's opt-in lost at its only consumer). `overflow` with the
embeddings form is refused (no tokenizer runs there).

### A4. The text-encoder stage (`packages/models/src/moss/text-stage.ts`)

- **Host, in the acceptance check**: the chat template (`renderQwen3_5ChatPrompt`) then the
  tokenizer, for each branch — before any GPU work, so a prompt the template or the bound refuses
  opens nothing.
- **The stage** (`encodeMossPrompts`, the first stage of a prompt-form generation): its own headroom
  check (decision 5's form: the encoder's estimate at T = max(T₊, T₋), no knob — ADR 0137 decision
  5 — plus the largest weight part and the margin), `Qwen3_5TextEncoder.open`, the positive then the
  negative encoding, and the encoder disposed and its released memory awaited **on every path**
  (success, a failed run, a cancellation) before `dit_text` opens — ADR 0137 decision 4, Wan's umT5
  text stage (ADR 0119 decision 11) the precedent. It takes no preset knob.
- **The first headroom check** of decision 5 takes the stage's need in the prompt form only (8,603
  MiB for the fixed prompts on the RTX 3080 Ti).
- The stage takes a `signal` of its own (the encoder's); `generateMoss` passes none — no stage of
  the DiT and VAE side has cancellation, and this addendum adds none.

### A5. Admission (`mossTextEncoderContract`)

Before a weight byte is uploaded: the encoder's graph contract (ADR 0137), its row width against
the DiT's text width (4,096), its one symbolic length (the estimate's binding), and the tokenizer
asset's `maxLength` against the encoder's bound and the DiT's (4,096 each — ADR 0137's MUST).

### A6. Checks

- Host (`packages/models/tests/moss_text_stage_host_test.ts`, stand-ins for the GPU and the
  Sessions): the two forms and their rejections, the ids through the template, `overflow`, the
  stage's order and its dispose and release wait on success, failure and cancellation, its headroom
  refusal, its need at the longer prompt, the encoder disposed before `dit_text` opens, and the
  first check taking the encoder's need in the prompt form only. Faults injected and caught: the
  dispose removed or not awaited, the release wait removed, the template skipped, the need at the
  positive prompt only, the mixing check removed, a default `""` negative, the stage's probe at 0 B,
  `overflow` dropped, the encoder's need left out of the first check.
- GPU (`packages/e2e/tests/e2e_moss_prompt_test.ts`): the ids of all four fixed prompts are exactly
  the ids upstream's `encode_prompt` fed its encoder (MUST); the stage's rows of `fridge-note` and
  the negative are the encoder lane's sha256 rows bit for bit (asserted); the distance from
  upstream's bf16 embeddings and the stage's time and memory are recorded, not banded; prompt →
  image at 256² × 2 steps (`f32+dit8`, guidance 3, shift 5, seed 42) with the Session order
  asserted and a new sha256 row `f32+dit8-2step-fridge-note-qwen3_5-i8-seed42-256x256-g3-shift5` —
  its own ID, since the embeddings are another route's (obstacle 1) — written only after the
  stage's assertion held in the same run.

### A7. Results (2026-10-31, RTX 3080 Ti, Node + Dawn)

- Relative L2 against upstream's bf16 embeddings: 3.98 % (`fridge-note`), 3.88 %
  (`guide-whiteboard`), 4.86 % (the negative) — the recipe's CPU-reference figures to four digits —
  and 7.79 % for `guide-fridge-note`, where one row (the `\n` after `assistant`) has a cosine of
  0.804. Karume's CPU reference of the same container gives the same 7.79 % and row (the GPU is
  2.9e-6 from it), so the gap is the i8 export's or upstream's bf16 run's, not the GPU's.
- The stage: 9.9–10.1 s, of which the Session build is 8.9–9.1 s; need 8,603 MiB, held peak
  7,589–7,602 MiB, 0 B held after it returns. The 256² × 2-step generation took 18.0 s.

### What still waits

The manifest step (decision 12, ruling 53 a): the text series and the tokenizer asset in a
distribution, its recipe and local mirror, `pipelineConfig` with upstream's defaults (the empty
negative prompt among them) and `defaultQuant`, and the public surface. Before it, the image A/B of
upstream's embeddings against Karume's (ADR 0137 decision 5, ruling 51) — `guide-fridge-note`
included. The encoder at T = 4,096 is unmeasured (ADR 0137 open point 2).

## Addendum (2026-11-01) — the distribution step: a recipe, a local mirror and the subpath `@karume/models/moss`

With the text encoder in place and its default storage ruled (i8 — the maintainer's ruling 51, ADR
0137 decision 5), the step decisions 12 and 13 deferred is taken: a distribution of the existing
series, its local mirror, and a public pipeline over the manifest. **Not done here**: Hugging Face
publication (a later maintainer decision — no repository, no revision pin, no `SOURCES` entry).

### B1. The distribution `karume-moss-image` (`tools/export-recipes/moss/distribution.py`)

- **One model, `sft`** (the upstream checkpoint `MOSS-Image-SFT` without the family name, as
  `ti2v-5b` is Wan2.2's), pipeline contract `moss-image/1`, built by
  `uv run python dist.py --pipeline moss-image` from the existing series, all self-contained:
  `dit` / `dit_skip22` / `dit_text` from `moss-image-sft-i8-dyn` (label `i8`; `dit_skip22` holds
  no weight bytes and runs on `dit`'s), `vae_decoder_pre` / `attn` / `post` from
  `moss-image-sft-f32-dyn` (label `f32`), `text_encoder` from
  `qwen3.5-9b-text-moss-image-sft-i8-dyn` (label `i8`), and the asset `tokenizer` from
  `qwen3_5-tokenizer/tokenizer.json`. The text tower is the checkpoint's own `text_encoder` folder,
  so it is stored here, not referenced from a Qwen3.5 repository. The host tables (`rope_base`,
  `training_grid`) and the BatchNorm statistics (`latent_batch_norm`) are container assets (ADR
  0109 decision 4). The precomputed text embeddings are not distributed (the public surface takes
  prompt strings — B4).
- **Assembly gates** (before a byte is written): each container's storage (required and forbidden
  layouts; the borrower's on its lender), its provenance (licence and revision of the pinned
  checkpoint — the text tower's too), the container assets its reader needs, that `dit_skip22`
  holds no weight bytes and borrows every initializer of `dit` with the lender's dtype and shape,
  that the text tower's rows are as wide as `dit_text` reads (4,096), and that the tokenizer asset
  has the TS reader's format, the pinned upstream folder and the bound 4,096.
- `LICENSE.md` (Apache 2.0) and `NOTICE.md` (the changes, the FLUX.2 VAE's origin) at the root;
  a model card from the manifest (`moss/card.py`).
- **The local mirror** `models/karume-moss-image/`: 13,160,009,476 B (about 12.3 GiB of weights,
  the text tower 7.95 GB of it). The core places independent copies (the ADR 0041 addendum's rule —
  no hard links); on this machine's btrfs the copy took seconds.

### B2. Presets, `defaultQuant` and ADR 0128's constraints

| Preset                                 | `session` (the DiT pair)                                                          | Label                                                 |
| -------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `f32+dit8`                             | —                                                                                | int8 transformer, f32 compute                         |
| `f32+dit8-a8-attn8-online` (default)   | `linearCompute: "a8"`, `attentionCompute: "a8"`, `attentionScoreStorage: "none"` | int8 transformer, int8 activations, online attention |

- Both take every component on its one label (the same weights map), each with a description
  (C3). **C1 holds**: neither declares `gpuFeatures`; the online kernel asks for subgroups only
  when the adapter has them and otherwise runs its portable form (ADR 0132 decision 5), so the
  practical preset can be the default. **C2 holds**: `f32+dit8` is the practical preset's sibling.
  The distribution gate (`packages/runtime/tests/distribution_gate_test.ts`) checks C1–C3 on the
  mirror; the mirror is in its roster (the pipeline e2e roots on it).
- `requiredLimits` (derived by the core): 1,017,118,720 B for `maxBufferSize` and
  `maxStorageBufferBindingSize` — the text tower's int8 embedding.
- **The manifest names a preset; `preset.ts` owns its knobs**: the pipeline refuses a quant whose
  `session` declares other knobs than the table's preset of that name, and a preset name the table
  does not know (`mossPresetOf`). Why: the checks that make a preset safe (the self A/B check, the
  census, the sha rows whose IDs start with the preset's name — decision 11) were run on these
  knobs; a manifest could otherwise run unchecked numerics under a checked name. A new preset is
  added to the table with its checks first. Rejected: knobs taken from the manifest alone (Wan's
  form) — two owners of a name's knobs.

### B3. `pipelineConfig` — upstream's defaults

`{ scheduler: { shift: 5.0 }, defaults: { width: 1024, height: 1024, steps: 50, guidance: 3.0,
negativePrompt: "" } }` — `MOSSImagePipeline.__call__`'s defaults and the pinned scheduler config's
`shift` (the recipe's test reads both from the pinned snapshot). The TS schema is
`packages/models/src/moss/config.ts` (unknown keys, missing fields and values outside the request
checks' ranges fail as plain `Error`s); the pipeline's admission also checks that the default steps
× shift give a σ schedule on the `dit` container's training grid.

### B4. The public surface `@karume/models/moss`

- `Pipeline.fromPretrained(ref, options)` (`model`, `quant`, `gpu`, `signal`, the fetch-layer
  knobs, `components`), `generate(request)`, `dispose()`; the types `GenerateRequest`,
  `GeneratedImage` (`[3, H, W]` f32 in `[-1, 1]`), `PipelineOptions`, `FromPretrainedOptions`;
  `imageToRgba`; `ModelInputError`. Bare names (ADR 0136); the barrel's namespace `moss` is the
  subpath's module (the barrel gate requires one per `exports` key). **No `SOURCES`** — the
  precedent of an unpublished family (vowel-detector; Wan2.1 has no key), since an entry before
  publication can only 404 (ADR 0073 decision 1). No `fromAssets` (the playground loads through
  `fromPretrained` with a source handle; it can be added in the house form).
- The request takes **prompt strings** only; the embeddings form stays on the internal entry
  (`generateMoss` — the parity checks). A field left out takes `pipelineConfig`'s default; the
  default negative prompt only while guidance ≠ 1 (at 1 a given one is refused as before); the
  seed defaults to 0, as in every family, and only when neither a seed nor latents is given.
- The admission runs before any weight byte is fetched: the pipeline name and major,
  `pipelineConfig`, the preset, a shared GPU's features and limits, the declared `requiredLimits`,
  the tokenizer asset's declaration, the text tower's graph contract, the DiT contract with its two
  assets and the VAE contract with its asset. The tokenizer's bound is checked when the asset
  arrives (after the weights). No Session is built at construction.
- **Removing the subpath** before a release (the maintainer's decision for 0.15.1): the `./moss`
  entries of `packages/models/package.json` and `deno.json` `exports`, the `export * as moss` block
  of `packages/models/mod.ts` (and its two mentions in the module doc), the file
  `packages/models/moss.ts`, the `./moss` block and the barrel's `moss` namespace in
  `packages/models/tests/fixtures/public-surface.json` (`KARUME_SURFACE=write`), and the
  `@karume/models/moss` row of `packages/models/README.md`. `src/moss/` and the e2e keep working
  (the e2e imports `moss.ts`; point it at `src/moss/pipeline.ts` instead).

### B5. Checks

- Recipe (`moss/tests/test_distribution.py`): the spellings against the writers, C1–C3 on the
  table, `pipelineConfig` against the pinned upstream code, the gates' refusals (synthetic
  containers), the plan over the real series. Faults injected and caught: the tokenizer's revision
  check removed, the borrower's byte check removed, the asset check reduced to presence, the default
  switched to `f32+dit8`, guidance 4.
- Host (`packages/models/tests/moss_pipeline_host_test.ts`): the preset from a synthetic manifest
  (default, named, another `defaultQuant`, unknown quant / model / preset name, knobs differing
  from the table, a knob the family does not accept), the request defaults, the schema, the
  contract. Faults injected and caught: `quant` overriding ignored, the knob comparison removed,
  the default negative at guidance 1, the seed default over given latents, the major check dropped,
  the `negativePrompt` type check dropped, the tokenizer-asset check dropped.
- GPU (`packages/e2e/tests/e2e_moss_pipeline_test.ts`, RTX 3080 Ti, Node + Dawn): the mirror's
  manifest resolves to the practical preset with upstream's defaults; `fromPretrained` from the
  mirror with `quant: "f32+dit8"`, the prompt `fridge-note`, 256² × 2 steps, seed 42 and guidance,
  shift and negative prompt **left to the manifest** gives exactly the existing sha row
  `f32+dit8-2step-fridge-note-qwen3_5-i8-seed42-256x256-g3-shift5` (written from the series with
  every knob explicit — Addendum of 2026-10-31, A6). The file never writes a row. Faults injected
  and caught: the default preset used in place of `quant`, a default guidance of 3.3.
- The gates that read every mirror under `models/` take the family: the distribution gate's roster,
  the session-options mirror check (`packages/e2e/tests/e2e_session_options_mirror_test.ts` — the
  family's acceptance table) and the census table (`packages/models/tests/helpers/census-table.ts` —
  the practical preset's row, 258 / 252 i8a8 linears for `dit` / `dit_skip22`, the e2e A/B's
  exact counts).

### What is still not done

- **Hugging Face publication**: the repository `hdae/karume-moss-image`, its revision pin and a
  `SOURCES` entry, the licence review of `THIRD_PARTY_NOTICES.md` (ADR 0065 decision 7), and the
  model card's review.
- **Other GPUs' sha rows** (the RTX 5070 Ti on D3D12 and others — ADR 0106): this family's rows
  exist for `node-nvidia-geforce-rtx-3080-ti` only; the pipeline e2e skips explicitly elsewhere.
- **T = 4,096**: the text encoder at its full bound is unmeasured (ADR 0137 open point 2), and the
  acceptance admits it.
- An original-weights preset (skipped — ADR 0128 decision 4's table records why), and a tiled VAE
  path above the cap.
