# Wan text-to-video demo

A one-shot command line for two Wan pipelines: Wan2.1 T2V 1.3B (`WanPipeline`, the default) and
Wan2.2 TI2V 5B (`WanTi2vPipeline`, `--generation wan2.2`). One prompt in, one clip out as numbered
PNG frames. It is the worked example for `fromPretrained`, `prompts`, `generate` and
`wanFrameToRgba`. Both generations run text-to-video only; image-to-video is not supported yet.
Everything below describes Wan2.1 unless it says otherwise; [Wan2.2 TI2V 5B](#wan22-ti2v-5b) lists
what differs for the second generation.

```
deno task demo:wan
deno task demo:wan --prompt "A red fox trots through fresh snow at sunrise." --steps 20
deno task demo:wan --text-encoder precomputed --prompt ferret --seed 7 --frames 17 --size 480x832
deno task demo:wan --quant f16 --prompt boxing-cats --seed 42
deno task demo:wan --generation wan2.2
```

`--generation` takes `wan2.1` (default) or `wan2.2`; any other value is rejected. The script needs
a WebGPU adapter. Run it from the repository root: the default inputs and the output directory are
relative paths.

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

## Swapping the text encoder

`--swap-text-encoder` replaces umT5 with a compatible encoder from another distribution, for example
a local experiment mirror built from a third-party fine-tune (the intake and conversion steps are in
[tools/export-recipes/wan/README.md](../../tools/export-recipes/wan/README.md#third-party-compatible-encoders-adr-0122-stage-b)):

```
deno task demo:wan --swap-text-encoder outputs/misc/local-dist/<name> --prompt boxing-cats
deno task demo:wan --swap-text-encoder your-name/your-umt5-encoder@<40-hex commit> --prompt boxing-cats
```

The flag takes either a local distribution (a directory with `karume.json`) or a Hugging Face
repository with a commit, and the script passes it to `fromPretrained` as the component swap seat.
In code it is the same one option:

```ts
const pipeline = await WanPipeline.fromPretrained(wanSource, {
  components: {
    text_encoder: { source: { repo: "your-name/your-umt5-encoder", revision: "<40-hex commit>" } },
  },
});
```

- **Pin the replacement.** A Hugging Face source must be `{ repo, revision }` with a commit. A bare
  repository name (a string `source`) follows `main`, so the umT5 version that the Wan manifest pins
  through its cross-repository reference would silently move with every push to the replacement.
  The script therefore rejects `owner/name` without `@<commit>`, and also a branch name after `@`.
- **The swap needs the GPU text encoder.** With `--text-encoder precomputed` (`textEncoder:
  "precomputed"`) umT5 is never loaded, so a swap would have no effect; the script rejects the
  combination instead of ignoring the flag.
- **What is accepted.** The replacement is admitted only if its graph description has the same
  SHA-256 as the one the Wan manifest declares for `text_encoder`, its binding table has no missing
  or extra weights, and the Wan pipeline accepts its storage (int8 weights and float32 tables only —
  an int4 or f16 umT5 is refused). All of this is checked before any weight is downloaded. The
  model and quant are the replacement manifest's defaults (`model` and `quant` next to `source`
  pick others in code).
- **What is not swapped or checked.** The tokenizer and the prompt cleaning still come from the
  Wan distribution, and nothing checks that the replacement was trained with the same tokenizer or
  the same relative-position buckets, or under which license it may be used. See
  [docs/limitations.md](../../docs/limitations.md) for what each of these means for the output.
- **No umT5 mirror needed.** The swapped part is read from the replacement only, so a local Wan
  distribution does not need `models/karume-umt5-xxl` or `--umt5-source` (the script rejects
  `--umt5-source` together with a swap).

## Knobs

Every flag that is left out falls back to the pipeline default. Steps, guidance and flow shift come
from the distribution's `pipelineConfig` (50, 5.0 and 3.0, the reference setting); the size and
frame count default to 832×480 and 33 frames. `--seed` defaults to 42.

- `--quant` picks the quant, a key of the manifest's `quants` (`f16`, `f16+dit8` or
  `f16+dit8-a8-attn8-s16`). Without it the run uses the manifest's `defaultQuant`. The name
  is passed to `fromPretrained` as is; an unknown name fails with the list of available quants
  before any weight is read.
- `--size` accepts `832x480` or `480x832`.
- `--frames` accepts 4n+1 between 5 and 81. Only 33 and 81 frames have been verified end to end on
  the development GPU (Intel Arc B570).
- `--guidance 1` turns classifier-free guidance off; `--negative` is then rejected.

Values outside these sets fail with `ModelInputError` before any weight reaches the GPU.

## Output

Frames go to
`outputs/examples/wan2.1-t2v-1.3b/wan-<quant>-<prompt>-<route>-<W>x<H>-<frames>f-<steps>step-seed<seed>/frame-NN.png`
(`--out` changes the root). `<quant>` is the `--quant` value, or `default` without it.
`<prompt>` is the asset name, or `prompt-` and the first eight hex
digits of the prompt's SHA-256 for any other string; `<route>` is `gpu`, `precomputed`, or
`gpu-swap` with `--swap-text-encoder` (runs with different replacements share it, so give each its
own `--out`). `<steps>` is the step count the run used, so a run without `--steps` is named after
the distribution's default. Guidance, flow shift and the negative prompt
are not part of the name: runs that differ only in those write to the same directory and overwrite
each other's frames, so give each one its own `--out`. The 8-bit conversion is `wanFrameToRgba`,
the same rule the reference hashes use.

Without `--quant`, the script runs the distribution's default quant, `f16+dit8-a8-attn8-s16` (an
int8 transformer with
int8 activations — ADR 0120). With the precomputed embeddings, the default run took about 16 minutes
on the B570 (952 s): two transformer forwards per step (about 8.5 s each at 832×480 × 33 frames) and
about two minutes of tiled VAE decoding. The `f16` quant takes about half an hour (about 17 s per
forward). The GPU text encoder adds a
umT5 stage per run (loading the 5.3 GiB of int8 weights and two forwards); on the B570 it took
10.4 s, including building its session (measured in a 2-step run at 33 frames).

## In Chrome

The same pipeline runs in Chrome on the **4. Wan** tab of the GPU lab (`deno task bench:gpu-lab`;
see [tools/gpu-lab/README.md](../../tools/gpu-lab/README.md#4-wan)). The tab reads
`models/karume-wan2.1` from the lab's server, checks the adapter's limits against the selected frame
count before loading, plays the clip, and shows the SHA-256 of the frames for the reference cases.
The tab starts on the precomputed route; its GPU route also needs `models/karume-umt5-xxl`, which the
lab's server serves next to the Wan distribution (`--umt5-source`).
A clip needs a storage binding of at least 503,193,600 bytes at 33 frames and 1,174,118,400 bytes
at 81 frames (the transformer's FFN intermediate), far above the WebGPU default of 128 MiB;
`acquireGpu` requests the adapter's own limits. On an RTX 5070 Ti, a 50-step clip of 81 frames with
the `f16` quant and the precomputed embeddings completed in Chrome in 46.1 minutes. The GPU text
encoder and the int8 quants have not been run in Chrome yet for Wan2.1; for Wan2.2, see below.

## Wan2.2 TI2V 5B

`--generation wan2.2` runs `WanTi2vPipeline` instead of `WanPipeline`. The source, text encoder,
swap, quant and knob flags and the output naming work as described above; only the defaults and the
accepted values below differ. `--source` must point at a distribution of the chosen generation: the
pipeline rejects a manifest whose pipeline does not match (`wan/1` for Wan2.1, `wan-ti2v/1` for
Wan2.2).

```
deno task demo:wan --generation wan2.2
deno task demo:wan --generation wan2.2 --size 704x1280 --frames 49
deno task demo:wan --generation wan2.2 --quant f16+dit8
```

### Where the model comes from

The script loads the `karume-wan2.2` distribution (model `ti2v-5b`, pipeline `wan-ti2v/1`). It is
not published on Hugging Face yet either, so without `--source` the script reads the local mirror
`models/karume-wan2.2`. Build it under `tools/export-recipes` after the umT5 mirror (the script
prints this command when the mirror is missing):

```
uv run python dist.py --pipeline wan-ti2v \
    --ref-repo hdae/karume-umt5-xxl --ref-revision 0000000000000000000000000000000000000000 \
    --ref-dist ../../models/karume-umt5-xxl --ref-model xxl --ref-role text_encoder \
    --allow-placeholder-ref
```

The all-zero revision is a placeholder for the development mirror until `karume-umt5-xxl` is
published; a local reader resolves the reference through the explicit mapping and does not look at
the revision.

The distribution itself holds 6.714 GiB of weights, the int8 transformer (4.67 GiB) and the two f16
VAE graphs, plus 12 MiB of assets.
Its text encoder is the same cross-repository reference to the shared `karume-umt5-xxl`
distribution (5.296 GiB), so `--umt5-source` and `--text-encoder precomputed` behave as for
Wan2.1. The precomputed text-embedding asset and the tokenizer are byte-identical to Wan2.1's, so
the same four prompt names work.

### Quants

- `f16+dit8`: int8 transformer weights with float compute. This is the reference quant.
- `f16+dit8-a8-attn8-s16` (default): also int8 activations in the linear layers and in attention,
  with the attention scores stored in f16. It became the default after a visual check of 12 clips.

There is no `f16` quant for Wan2.2.

### Knobs

Steps, guidance and flow shift default to 50, 5.0 and 5.0 (the distribution's `pipelineConfig`).
`--size` accepts `1280x704` or `704x1280`, and `--frames` accepts 4n+1 between 5 and 49. The
default clip is 1280×704 × 33 frames, played at 24 fps.

### Output

Frames go to
`outputs/examples/wan2.2-ti2v-5b/wan-<quant>-<prompt>-<route>-<W>x<H>-<frames>f-<steps>step-seed<seed>/frame-NN.png`,
with the same naming rules as for Wan2.1.

On the development GPU, an NVIDIA GeForce RTX 3080 Ti, under Deno, with the default quant and the
precomputed embeddings at 1280×704 × 33 frames and 50 steps, a clip took 753 to 792 s (transformer
556 to 593 s, VAE 197 to 199 s). The whole GPU peaked at 7,935 to 8,047 MiB (`nvidia-smi`). These are
12 clips measured on 2026-10-06 with `WanTi2vPipeline` reading the export series directly, not with this
script.

### In Chrome

The GPU lab's Wan tab has a generation selector; the lab's server serves Wan2.2 from
`--wan22-source` (default `models/karume-wan2.2`; see
[tools/gpu-lab/README.md](../../tools/gpu-lab/README.md#7-wan22-ti2v-5b)). On an RTX 5070 Ti
(Chrome 154, Windows, 2026-10-06), a 50-step clip with the default quant and the GPU text encoder at
1280×704 and 121 frames completed in 40.3 minutes. The tab lifts the frame limit to 121 for this;
this script and the pipeline accept up to 49 frames.
