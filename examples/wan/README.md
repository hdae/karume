# Wan2.1 text-to-video demo

A one-shot command line for the Wan2.1 T2V 1.3B pipeline: one prompt in, one clip out as numbered
PNG frames. It is the worked example for `WanPipeline.fromPretrained`, `prompts`, `generate` and
`wanFrameToRgba`.

```
deno task demo:wan
deno task demo:wan --prompt ferret --seed 7 --steps 20 --frames 17 --size 480x832
```

The script needs a WebGPU adapter. Run it from the repository root: the default inputs and the
output directory are relative paths.

## Where the model comes from

The script loads the `karume-wan2.1` distribution with `WanPipeline.fromPretrained`. The
distribution is not published on Hugging Face yet, so there is no pinned source table
(`WAN_SOURCES`) and the default is the local mirror:

- Without `--source`, the script reads `models/karume-wan2.1` through a directory source handle
  (`denoDirectory`): nothing goes over the network or into the cache. Build the mirror first from
  the export series with `uv run python dist.py --pipeline wan` under `tools/export-recipes` (the
  series come from `python -m wan.export_dit`, `python -m wan.export_vae` and
  `python -m wan.text_embeds`). If the mirror is missing, the script stops and prints that command.
- `--source <path>` points at another local distribution (a directory with `karume.json`), and
  `--source <owner/name>` reads a Hugging Face repository (its `main` revision).

The distribution holds the transformer, the two VAE chunk graphs and the precomputed text-embedding
asset (about 2.9 GiB). Parts are read one at a time while the sessions are built, not loaded into
host memory up front.

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

Every flag that is left out falls back to the pipeline default. Steps, guidance and flow shift come
from the distribution's `pipelineConfig` (50, 5.0 and 3.0, the reference setting); the size and
frame count default to 832×480 and 33 frames. `--seed` defaults to 42.

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
