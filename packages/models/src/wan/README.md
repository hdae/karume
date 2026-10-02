# Wan2.1 (host side)

Placeholder for the `./wan` subpath of `@karume/models` (ADR
[0118](../../../../docs/decisions/0118-wan21-video-generation.md) decision 7). Nothing here is
exported yet: the `./wan` entry in `deno.json` and the barrel re-export are added when the pipeline
is wired up (stage 6).

| Files           | Owner   | Contents                                                                                      |
| --------------- | ------- | --------------------------------------------------------------------------------------------- |
| `dit*.ts`       | stage 2 | patchify / unpatchify, RoPE base-table reordering, `timesteps_proj`                           |
| `vae-chunks.ts` | stage 4 | chunked decode of one tile with the resident causal cache                                     |
| `vae-tiles.ts`  | stage 5 | tiled decode over the full frame: tile plan from the asset's input shape, blend, paste, clamp |
| the rest        | stage 6 | `WanPipeline`, UniPC, CFG, text-embedding assets, input gates, `WAN_SOURCES`                  |
