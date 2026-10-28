# MOSS-Image (host side)

MOSS-Image's DiT and its FLUX.2 decoder (ADR
[NNNN](../../../../docs/decisions/NNNN-moss-image-dit-vae.md), branch `wip/moss-image`):
`generateMoss` (`generate.ts`) turns precomputed text embeddings into an image `[3, H, W]` in
`[-1, 1]` with the DiT of `OpenMOSS-Team/MOSS-Image-SFT` and its untiled VAE decoder. The text
encoder (Qwen3.5-9B) is not ported, so a request carries upstream's encoder outputs instead of
strings. **There is no public surface**: the family is not a subpath of `@karume/models`, not in
the barrel and has no distribution or manifest until the text encoder exists (ADR NNNN decision 12).
The caller opens the two series (the DiT series `moss-image-sft-i8-dyn` and the VAE series
`moss-image-sft-f32-dyn`; the tests' helpers `openMossSeries` / `openMossVaeSeries` do it from
`outputs/series/`) and passes them with a device and a preset.

A generation runs three stages, one session set at a time, each closed before the next opens and
followed by a wait for the released GPU memory (Dawn returns destroyed buffers only when a later
submit completes). Before anything opens, one trial allocation (`fitsHeadroom`) checks the largest
of the three stage needs; before each stage opens, another checks its own need (the runtime
estimate with the preset's knobs, plus the largest weight part and a margin — ADR NNNN decision 5).
A refusal fails loudly with the stage, the bytes and the size:

1. **text** — the text refiner `dit_text` (symbol T) runs once for the positive and once for the
   negative embeddings, always with f32 compute; its outputs are read back once and placed in the
   text rows of each branch's text context.
2. **transformer** — the S-shaped DiT (S = N image tokens + T text tokens, one symbol): the `dit`
   Session (positive branch) and the `dit_skip22` Session (negative branch — upstream's forward
   without block 22, borrowing every weight of `dit`) are open together, each with one shape for the
   whole generation. Each step runs both forwards, the CFG combination and the Euler update on the
   host in f32. With guidance 1 only `dit` opens.
3. **vae_decoder** — the host undoes the latent BatchNorm and unpatchifies 128 → 32 channels at 2×;
   the three decoder graphs (`vae_decoder_pre` / `attn` / `post`, split around the mid attention)
   run in one batch and hand their tensors over through residents; the image is read back once and
   clamped to `[-1, 1]`. The decoder is untiled (upstream's default — every GroupNorm sees the whole
   image), and its mid attention runs in row blocks of at most 256 MiB.

The preset is part of the state and has **no default** — not even ADR 0128 decision 1's provisional
one, since no manifest owns a default yet (ADR NNNN decision 13); the family's default is
`f32+dit8-a8-attn8-online` (the maintainer's visual review passed), applied by the manifest when
it comes:

| Preset                     | DiT (`dit`, `dit_skip22`)                                                                   | `dit_text`  | VAE |
| -------------------------- | ------------------------------------------------------------------------------------------- | ----------- | --- |
| `f32+dit8` (reference)     | i8 weights, f32 compute                                                                     | f32 compute | f32 |
| `f32+dit8-a8-attn8-online` | the same i8 weights, `linearCompute: "a8"`, `attentionCompute: "a8"`, `attentionScoreStorage: "none"` | f32 compute | f32 |

`mossAcquireGpuOptions(preset)` gives the device options of a preset (subgroups when the adapter
has them, for the online kernel's subgroup form — never required).

| Files              | Stage | Contents |
| ------------------ | ----- | -------- |
| `dit-contract.ts`  | 2     | the DiT series' graph contract (`dit`, `dit_skip22`, `dit_text`) and its two assets, checked before the weight parts are fetched |
| `dit-rope.ts`      | 2     | the asset `rope_base` and the reordering of upstream's per-axis sin / cos tables into `rope_cos` / `rope_sin` (never computed in TS) |
| `dit-timestep.ts`  | 2     | the asset `training_grid` (σ, timesteps, time sinusoid of the 1,000 training-grid points) and the lookup of `timesteps_proj` by grid index |
| `dit-tokens.ts`    | 2     | the joint-sequence inputs (`latent_tokens`, `text_context`, `is_image`) and the head rows from the output |
| `request.ts`       | 2     | the size and text checks (multiples of 16, the longest side, the area against the VAE's cap, the embeddings) |
| `dit-stage.ts`     | 2     | the text stage, the DiT pair's Sessions and lifetimes, their headroom needs and the host CFG |
| `vae-contract.ts`  | 3     | the VAE series' graph contract and the asset `latent_batch_norm`; the area cap derived from the graphs and the device's binding limit |
| `vae-host.ts`      | 3     | the BatchNorm restoration, the unpatchify and the clamp |
| `vae-stage.ts`     | 3     | the VAE stage: the three Sessions and residents in one batch, its headroom need, the mid attention's row-block budget |
| `sampler.ts`       | 4     | upstream's σ schedule (the requested grid, the snapping to the training grid, de-duplication) and the Euler step |
| `random.ts`        | 4     | the seeded initial noise (splitmix64 + Box–Muller — not torch's `randn`) |
| `generate.ts`      | 4     | `planMossGenerate` (the one acceptance check of a request) and `generateMoss` (the stage order) |
| `preset.ts`        | 5     | the two presets, `mossPreset` / `MOSS_PRESET_NAMES`, `mossAcquireGpuOptions` |

(The stages are those of ADR NNNN decision 14.)

## Accepted requests

`planMossGenerate` rejects with a `ModelInputError`, before any GPU work: a side that is not a
positive multiple of 16, a side over 65,536 px, an area over this device's cap of the untiled VAE
(2,796,160 px at a 2 GiB storage binding — the cap is derived per device); embeddings that are not
`Float32Array` `[T, 4096]` with 1 ≤ T ≤ 4,096 and all finite, a missing positive prompt, a missing
negative prompt while guidance ≠ 1 or one given at guidance 1; steps that are not a positive integer,
guidance that is not finite and ≥ 0, a shift that is not finite and > 0 or so small that the σ grid
is not finite; and anything but exactly one of `seed` and `latents` (`[128, H/16, W/16]` f32,
finite). Every knob is explicit — upstream's defaults (1024², 50 steps, guidance 3, shift 5) wait
for a manifest. There is no step cap. Whether the stages fit the free VRAM is the stages' own check,
not an acceptance limit (`docs/limitations.md`, the MOSS-Image section, records what fit on a
12 GiB device).

## Numerics

- The host functions are Uint32-identical to upstream's recorded outputs: the RoPE tables, the time
  sinusoid and the training grid (upstream's own values, reordered or indexed), the requested σ grid
  (a transcription of `torch.linspace` with one rounding per element — the FMA torch's CPU kernel
  compiles to), the snapped schedule, CFG, Euler, the BatchNorm restoration and the unpatchify.
- From upstream's recorded DiT outputs, the host sampler loop reproduces upstream's latents of a
  two-step run bit for bit.
- The GPU is checked against layer-by-layer CPU references (r bands of the DiT), upstream's f32
  decode (m bands of the VAE) and upstream's two-step `pipe(...)` with the series' i8 weights (m
  bands of the latents and the image); these comparisons are opt-in (`KARUME_UPSTREAM_PARITY=1`),
  and the default lane pins the outputs with sha256 rows (ADR NNNN decision 11).
- The practical preset is checked against the reference preset on the same machine (the self A/B
  check of the step-1 DiT outputs and the latent after it) with the census of the variants that ran.
- The seeded noise is not torch's: the same seed does not give upstream's image; pass `latents` to
  start from upstream's initial noise.
