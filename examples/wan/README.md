# Wan2.1 text-to-video demo

A one-shot command line for the Wan2.1 T2V 1.3B pipeline: one prompt in, one clip out as numbered
PNG frames. It is the worked example for `WanPipeline.fromPretrained`, `prompts`, `generate` and
`wanFrameToRgba`.

```
deno task demo:wan
deno task demo:wan --prompt "A red fox trots through fresh snow at sunrise." --steps 20
deno task demo:wan --text-encoder precomputed --prompt ferret --seed 7 --frames 17 --size 480x832
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

The distribution holds the transformer, the two VAE chunk graphs, the precomputed text-embedding
asset and the umT5 tokenizer (about 2.9 GiB). The umT5 text encoder (int8, about 5.3 GiB) lives in a
separate distribution, `karume-umt5-xxl`, that the Wan manifest references across repositories:

- With a local Wan distribution and the GPU text encoder (the default), the script also needs the
  local umT5 distribution. `--umt5-source <path>` points at it (default `models/karume-umt5-xxl`,
  built with `uv run python dist.py --pipeline umt5` under `tools/export-recipes`). It is passed to
  the directory source as an explicit cross-repository mapping; the script reads the repository name
  from the Wan manifest and never guesses a neighboring directory. If the umT5 mirror is missing,
  the script stops and prints the command.
- With a Hugging Face `--source`, the cross-repository reference is resolved by the hub at the
  revision the manifest pins, and `--umt5-source` is rejected.
- With `--text-encoder precomputed`, umT5 is not read at all, and `--umt5-source` is rejected.

Parts are read one at a time while the sessions are built, not loaded into host memory up front.

## Text encoder and prompts

`--text-encoder` picks how the prompt becomes the transformer's context:

- `gpu` (default, the pipeline's own default) runs umT5 on the GPU for the prompt and the negative
  prompt in every run, then disposes it before the transformer stage. Any prompt is accepted that
  survives the upstream `prompt_clean` preprocessing and the tokenizer (2 to 512 tokens). Prompts
  the port cannot reproduce exactly are rejected with a `ModelInputError` that says how to fix
  them: characters outside the vocabulary or unassigned in Unicode 16.0.0, C1 control characters,
  text that looks like an HTML character reference (write `R & D` instead of `R&D`) or like
  mojibake, special tokens such as `</s>`, and empty prompts.
- `precomputed` looks the prompt up in the precomputed embedding asset and does not download or run
  umT5. Only the four prompts stored in the asset are accepted. Its embeddings come from the bf16
  umT5, so a clip differs from the `gpu` route's clip for the same prompt and seed.

`--prompt` and `--negative` take either a name from the embedding asset (its text is passed) or
any other string (passed as is; the `precomputed` route rejects it with the list of names):

| Name             | Role     | Source                                                      |
| ---------------- | -------- | ----------------------------------------------------------- |
| `boxing-cats`    | positive | The t2v-1.3B example in the official README                 |
| `ferret`         | positive | The text-to-video example in the Diffusers documentation    |
| `cat-dog-baking` | positive | The `WanPipeline.__call__` example in Diffusers             |
| `negative`       | negative | The official `sample_neg_prompt` (the default `--negative`) |

Without `--negative` the pipeline uses the official `sample_neg_prompt` on either route. The full
prompt texts are in the asset's metadata and in `pipeline.prompts`.

## Knobs

Every flag that is left out falls back to the pipeline default. Steps, guidance and flow shift come
from the distribution's `pipelineConfig` (50, 5.0 and 3.0, the reference setting); the size and
frame count default to 832×480 and 33 frames. `--seed` defaults to 42.

- `--size` accepts `832x480` or `480x832`.
- `--frames` accepts 4n+1 between 5 and 81. Only 33 and 81 frames have been verified end to end on
  the development GPU (Intel Arc B570).
- `--guidance 1` turns classifier-free guidance off; `--negative` is then rejected.

Values outside these sets fail with `ModelInputError` before any weight reaches the GPU.

## Output

Frames go to
`outputs/examples/wan2.1-t2v-1.3b/wan-<prompt>-<route>-<W>x<H>-<frames>f-<steps>step-seed<seed>/frame-NN.png`
(`--out` changes the root). `<prompt>` is the asset name, or `prompt-` and the first eight hex
digits of the prompt's SHA-256 for any other string; `<route>` is `gpu` or `precomputed`. `<steps>`
is the step count the run used, so a run without `--steps` is named after the distribution's
default. Guidance, flow shift and the negative prompt
are not part of the name: runs that differ only in those write to the same directory and overwrite
each other's frames, so give each one its own `--out`. The 8-bit conversion is `wanFrameToRgba`,
the same rule the reference hashes use.

The default run takes about half an hour on the B570: two transformer forwards per step (about 17 s
each at 832×480 × 33 frames) and about two minutes of tiled VAE decoding. The GPU text encoder adds a
umT5 stage per run (loading the 5.3 GiB of int8 weights and two forwards); its duration has not been
measured yet.

## In Chrome

The same pipeline runs in Chrome on the **4. Wan** tab of the GPU lab (`deno task bench:gpu-lab`;
see [tools/gpu-lab/README.md](../../tools/gpu-lab/README.md#4-wan)). The tab reads
`models/karume-wan2.1` from the lab's server, checks the adapter's limits against the selected frame
count before loading, plays the clip, and shows the SHA-256 of the frames for the reference cases.
The tab starts on the precomputed route; its GPU route also needs `models/karume-umt5-xxl`, which the
lab's server serves next to the Wan distribution (`--umt5-source`).
A clip needs a storage binding of at least 503,193,600 bytes at 33 frames and 1,174,118,400 bytes
at 81 frames (the transformer's FFN intermediate), far above the WebGPU default of 128 MiB;
`acquireGpu` requests the adapter's own limits. A complete run in Chrome has not been confirmed yet.
