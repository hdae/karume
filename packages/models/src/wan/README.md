# Wan2.1 (host side)

The `./wan` subpath of `@karume/models` (ADR
[0118](../../../../docs/decisions/0118-wan21-video-generation.md) decision 7): `WanPipeline` turns a
prompt into a clip of `[3, F, H, W]` frames in `[-1, 1]`. The public surface is
[`wan.ts`](../../wan.ts) (also re-exported from the barrel); everything here is internal.

The pipeline runs three stages, one session set at a time, and waits for the released GPU memory to
settle after each stage (`destroy()` is released late on Intel / wgpu):

1. **text** — looks the prompt up in the precomputed umT5 embedding asset (no GPU). Only prompts in
   the asset are accepted, by their original or normalized text; anything else is a
   `ModelInputError`.
2. **transformer** — the S-shaped DiT, `steps` times; with guidance above 1 the uncond and cond
   passes run one after the other (B = 1), and the CFG combination and the UniPC update run on the
   host.
3. **vae_decoder** — the two chunk graphs with the resident causal cache, always tiled, then the clamp
   to `[-1, 1]`.

There is no distribution yet (stage 7): `WanPipeline.fromAssets` takes the series containers and the
embedding asset as bytes, and the scheduler config is the upstream value.

| Files            | Owner   | Contents                                                                                      |
| ---------------- | ------- | --------------------------------------------------------------------------------------------- |
| `dit*.ts`        | stage 2 | patchify / unpatchify, RoPE base-table reordering, `timesteps_proj`                           |
| `vae-chunks.ts`  | stage 4 | chunked decode of one tile with the resident causal cache                                     |
| `vae-tiles.ts`   | stage 5 | tiled decode over the full frame: tile plan from the asset's input shape, blend, paste, clamp |
| `scheduler.ts`   | stage 6 | flow-matching UniPC (σ / timestep columns, predictor and corrector) and the CFG combination   |
| `text-embeds.ts` | stage 6 | reading and checking the text-embedding asset, prompt lookup, zero padding to 512 rows        |
| `latents.ts`     | stage 6 | the VAE's per-channel latent mean / std and the de-normalization before decoding              |
| `random.ts`      | stage 6 | the seeded initial noise (splitmix64 + Box–Muller — not torch's `randn`)                      |
| `pipeline.ts`    | stage 6 | `WanPipeline`: input gates, the three stages, events and diagnostics                          |
| `frames.ts`      | stage 6 | one frame to 8-bit RGBA (`wanFrameToRgba` — the rule the reference hashes use)                |

## Accepted requests

832×480 or 480×832, 4n+1 frames from 5 to 33, `steps` ≥ 1, `guidance` ≥ 1 (1 turns CFG off),
`shift` > 0, and either a `seed` (default 0) or the initial noise as `latents`. Only 832×480 with 33
frames has been checked end to end on the GPU. Longer clips (up to 81 frames) are planned: holding the
DiT while the VAE stage starts would exceed the development GPU's VRAM by the current estimate.

## Numerics

- The σ column matches diffusers bit for bit and the timesteps exactly (σ computed in float64, then
  rounded to float32; the first timestep is 999).
- The UniPC trajectory matches within an absolute 2e-5 over 50 steps: torch's float32 `log` (MKL)
  differs from the correctly rounded value by one ULP at a few σ, which moves the step coefficients.
- The CFG combination matches bit for bit.
- From the DiT outputs the reference recorded, the host CFG + UniPC reproduces the reference latents
  of the two-step run bit for bit.
