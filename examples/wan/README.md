# Wan2.1 text-to-video demo

A one-shot command line for the Wan2.1 T2V 1.3B pipeline: one prompt in, one clip out as numbered
PNG frames. It is the worked example for `WanPipeline.fromAssets`, `prompts`, `generate` and
`wanFrameToRgba`.

```
deno task demo:wan
deno task demo:wan --prompt ferret --seed 7 --steps 20 --frames 17 --size 480x832
```

The script needs a WebGPU adapter. Run it from the repository root: the default inputs and the
output directory are relative paths.

## Where the model comes from

There is no distribution for this family yet (no `karume.json`, no `fromPretrained`, no pinned
source table — that is stage 7 of ADR 0118). The script reads the export series directly:

- `--series <dir>` (default `outputs/series/wan2.1-t2v-1.3b-f16-dyn`) holds the three containers
  `transformer/`, `vae_decoder_first/` and `vae_decoder_next/`, each as `model.krm` or as the
  numbered parts `model-0000N-of-0000M.krm`. They come from the Wan recipe
  (`python -m wan.export_dit` and `python -m wan.export_vae` under `tools/export-recipes`).
- `--embeds <file>` (default
  `outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors`) is the precomputed
  text-embedding asset (`python -m wan.text_embeds`).

All parts are read into host memory (about 3.1 GB) before the pipeline is built.

## Prompts

The first stage of the text encoder is a set of precomputed umT5 embeddings, so the pipeline accepts
only the prompts stored in the embedding asset. `--prompt` and `--negative` take the asset's names:

| Name             | Role     | Source                                                      |
| ---------------- | -------- | ----------------------------------------------------------- |
| `boxing-cats`    | positive | The t2v-1.3B example in the official README                 |
| `ferret`         | positive | The text-to-video example in the Diffusers documentation    |
| `cat-dog-baking` | positive | The `WanPipeline.__call__` example in Diffusers             |
| `negative`       | negative | The official `sample_neg_prompt` (the default `--negative`) |

An unknown name stops the script with the list of names the asset carries. The full prompt texts are
in the asset's metadata and in `pipeline.prompts`.

## Knobs

Every flag that is left out falls back to the pipeline default, which is the reference setting:
50 steps, guidance 5.0, flow shift 3.0, 832×480, 33 frames. `--seed` defaults to 42.

- `--size` accepts `832x480` or `480x832`.
- `--frames` accepts 4n+1 between 5 and 33. Only 33 frames has been verified end to end on the
  development GPU (Intel Arc B570); longer clips (up to 81 frames) are planned once the VRAM budget
  for holding the DiT next to the VAE is measured.
- `--guidance 1` turns classifier-free guidance off; `--negative` is then rejected.

Values outside these sets fail with `ModelInputError` before any weight reaches the GPU.

## Output

Frames go to
`outputs/examples/wan2.1-t2v-1.3b/wan-<prompt>-<W>x<H>-<frames>f-<steps>step-seed<seed>/frame-NN.png`
(`--out` changes the root). The 8-bit conversion is `wanFrameToRgba`, the same rule the reference
hashes use.

The default run takes about half an hour on the B570: two transformer forwards per step (about 17 s
each at 832×480 × 33 frames) and about two minutes of tiled VAE decoding.
