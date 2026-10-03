# Wan2.1 (host side)

The `./wan` subpath of `@karume/models` (ADR
[0118](../../../../docs/decisions/0118-wan21-video-generation.md) decision 7): `WanPipeline` turns a
prompt into a clip of `[3, F, H, W]` frames in `[-1, 1]`. The public surface is
[`wan.ts`](../../wan.ts) (also re-exported from the barrel); everything here is internal.

The pipeline runs three stages, one session set at a time: the text-encoder session (GPU route) is
disposed before the transformer session is opened, and the transformer session before the VAE
sessions (ADR [0119](../../../../docs/decisions/0119-wan-umt5-gpu-text-encoder.md) decision 11: the
int8 umT5 at 5.30 GiB and the transformer stage do not fit the B570 together). There is no wait for
released GPU memory between the stages. Intel / wgpu can release `destroy()` late, but on the B570
(2026-10-02, fdinfo `drm-total-vram0`) the transformer stage's peak drops as soon as its session is
disposed and does not overlap the VAE stage's peak (the NOTE at the top of `pipeline.ts` has the
numbers); the same has not been measured after the text stage yet:

1. **text** — the route is chosen at construction (`textEncoder`, ADR 0119 decision 7):
   - `"gpu"` (default) — runs umT5 (int8 per-channel weights, float32 activations) once for the
     prompt and once for the negative prompt, then pads each `[1, L, 4096]` output with zero rows up
     to the transformer's 512 context rows. Any string is accepted that passes the `prompt_clean`
     mirror and the tokenizer (2 to 512 tokens); the rest is a `ModelInputError` whose message says
     how to fix the prompt. The default negative prompt is the official `sample_neg_prompt`, encoded
     on the GPU as well.
   - `"precomputed"` — looks the prompt up in the precomputed umT5 embedding asset (no GPU, umT5 is
     not downloaded). Only prompts in the asset are accepted, by their original or normalized text;
     anything else is a `ModelInputError`.
2. **transformer** — the S-shaped DiT, `steps` times; with guidance above 1 the uncond and cond
   passes run one after the other (B = 1), and the CFG combination and the UniPC update run on the
   host.
3. **vae_decoder** — the two chunk graphs with the resident causal cache, always tiled, then the clamp
   to `[-1, 1]`.

The distribution is `karume-wan2.1` (stage 7 — not published on Hugging Face yet, so there is no
`WAN_SOURCES` table): `WanPipeline.fromPretrained` loads it through `@karume/hub` (a local mirror is
passed as a `denoDirectory` source handle), and `fromAssets` takes the manifest and the bytes. Both
go through the same admission. The `text_encoder` component is a cross-repository reference to the
umT5 distribution (`karume-umt5-xxl`); a local mirror of it is passed through the source handle's
`crossRepo` mapping. The `"precomputed"` route never opens that component. The defaults for steps,
guidance and shift come from the manifest's `pipelineConfig` (`config.ts`); the UniPC structure is
the upstream value.

Both construction and generation take an `AbortSignal`. Generation checks it at the stage boundaries,
before every run (the two text-encoder runs and every transformer step) and between VAE tiles; an
abort disposes the open sessions and rethrows `signal.reason` unwrapped.

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
| `config.ts`      | stage 7 | `pipelineConfig` schema (pipeline `wan/1`): default steps, guidance and shift                 |
| `frames.ts`      | stage 6 | one frame to 8-bit RGBA (`wanFrameToRgba` — the rule the reference hashes use)                |
| `text/*.ts`      | 10a     | the `prompt_clean` mirror and the umT5 tokenizer (prompt → token ids, with the reject rules)  |
| `umt5/*.ts`      | 10c     | the relative-position bucket table and the umT5 session inputs / output padding               |

## Accepted requests

832×480 or 480×832, 4n+1 frames from 5 to 81 (default 33), `steps` ≥ 1, `guidance` ≥ 1 and
finite in float32 (1 turns CFG off), `shift` > 0 with a `steps` × `shift` pair whose σ column is
strictly decreasing, and either a `seed` (default 0) or the initial noise as `latents`. Only 832×480
with 33 and 81 frames have been checked end to end on the GPU. The transformer stage is closed before
the VAE stage opens, so the two are never resident together. A non-finite umT5 output, a non-finite
latent after any step, or a non-finite VAE output before the clamp fails the generation instead of
being returned.

## Numerics

- The σ column matches diffusers bit for bit and the timesteps exactly (σ computed in float64, then
  rounded to float32; the first timestep is 999).
- The UniPC trajectory matches within an absolute 2e-5 over 50 steps: torch's float32 `log` (MKL)
  differs from the correctly rounded value by one ULP at a few σ, which moves the step coefficients.
- The CFG combination matches bit for bit.
- From the DiT outputs the reference recorded, the host CFG + UniPC reproduces the reference latents
  of the two-step run bit for bit.
