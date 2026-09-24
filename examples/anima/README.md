# Anima text-to-image demo

A one-shot command line for the Anima pipeline: one prompt in, one PNG out. It is the worked example
for `AnimaPipeline.fromPretrained` and `generate`. Where the assets come from is the only branching
part; everything else maps one flag to one `generate` knob.

```
deno task demo:anima
deno task demo:anima --prompt "1girl, solo, ..." --resolution 1344x768 --seed 42
```

The script needs a WebGPU adapter. Run it from the repository root: the default source and the
output directory are relative paths.

## Where the model comes from

`fromPretrained` has no built-in default, so the script picks one explicitly:

- Without `--source`, it reads the local distribution directory `models/karume-anima` when that
  directory exists. When it does not, the script fetches `ANIMA_SOURCES["anima"]` from Hugging Face —
  `hdae/karume-anima` at the commit this package version was verified against. A
  `models/karume-anima` directory without `karume.json` is an error, not a reason to fall back to
  the network.
- `--source <dir>` reads a local distribution directory (one that carries `karume.json`) directly.
  The layout is what `tools/export-recipes/anima` produces; see
  [`docs/assets-layout.md`](../../docs/assets-layout.md) for where the local mirrors live.
- `--source <owner/name>` — any value that is not such a directory — is treated as a Hugging Face
  repository name and fetched at `main`. Pin a commit yourself in code when you need a reproducible
  fetch.

One repository holds several models. `hdae/karume-anima` carries the official checkpoints and
defaults to Turbo; `hdae/karume-anima-extra` carries third-party fine-tunes and defaults to
`anima-wai-v1.0`. `--model` and `--quant` pick a different entry; the names are the ones the
repository's `karume.json` declares.

### Cross-repository references

The `karume-anima-extra` distribution does not ship its own text stack (text encoder, VAE decoder,
tokenizers). Its manifest points at files in `hdae/karume-anima` instead. When you read it from a
local directory, name the directory that stands in for that repository with `--source-map`:

```
deno task demo:anima --source models/karume-anima-extra \
  --source-map hdae/karume-anima=models/karume-anima
```

The flag repeats, one `owner/name=<dir>` per referenced repository. The script never guesses a
sibling directory: a reference without a mapping fails with a message that says which repository is
missing. Local reads check file sizes only, not sha256, so the mapped directory has to be the
revision the reference names. `--source-map` is rejected without an explicit local `--source`, and
with a Hugging Face `--source` (that path opens the declared repository and revision itself).

## Options

All options are `--key value` pairs. Unknown keys are rejected rather than silently ignored, so a
mistyped knob never runs on its default and passes for model variance.

| Option                          | Default                                        | What it does                                                              |
| ------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------- |
| `--source <dir\|owner/name>`    | `models/karume-anima`, else the pinned release | Where the distribution comes from (see above).                            |
| `--source-map <owner/name=dir>` | none                                           | Local stand-in for a cross-referenced repository. Repeatable.             |
| `--model <name>`                | manifest `defaultModel`                        | Which model of the repository to run.                                     |
| `--quant <name>`                | the model's `defaultQuant`                     | Which weight variant of that model to load.                               |
| `--prompt <text>`               | a built-in portrait prompt                     | The positive prompt.                                                      |
| `--negative <text>`             | manifest default                               | The negative prompt. Only valid when guidance is not `1` (see below).     |
| `--resolution <WxH\|N>`         | manifest default                               | Output size, such as `1344x768`; a single number means a square.          |
| `--seed <n>`                    | `42`                                           | Noise seed. A non-negative integer.                                       |
| `--steps <n>`                   | manifest default                               | Denoising steps. An integer of at least `2`.                              |
| `--guidance <x>`                | manifest default                               | Classifier-free guidance scale. `1` skips the unconditional branch.       |
| `--sampler <euler\|dpmpp-2m>`   | manifest `scheduler.type`                      | The denoising update rule. See [Comparing samplers](#comparing-samplers). |

"Manifest default" means the value the model's `pipelineConfig` declares. The options you leave out
keep that value; the ones you pass replace it for this run only.

A flag value may not start with `--`. The parser treats such a value as a swallowed flag and stops,
so a prompt whose text begins with `--` is rejected.

### Constraints the pipeline enforces

The resolution and the sampler are checked before any weights are loaded, so a bad value there
fails in seconds. The guidance / negative-prompt combination is checked later, inside `generate`,
after the weights have been fetched and loaded. The messages are in Japanese.

- **Resolution** (checked before loading). Each side must be a multiple of 16 and between 512 and
  2048 pixels. Sides above 1920 pixels are accepted but lie outside the position range the model
  declares, so quality there is unmeasured.
- **Sampler** (checked before loading). `--sampler` accepts exactly the values of
  `ANIMA_SAMPLER_TYPES`; the script checks the spelling before it touches the network or the GPU.
- **Guidance and the negative prompt** (checked inside `generate`, after loading). With
  `--guidance 1` the unconditional branch is never computed, so `--negative` would have no effect
  and is rejected. Any other guidance needs a negative prompt, from `--negative` or from the
  manifest default. The script passes your values through unchanged and lets the pipeline explain
  the conflict, so with a remote source a bad combination fails only after the multi-gigabyte
  download.

## Output

The image is written to `outputs/examples/<source>/`, where `<source>` is the last path element of
`--source`, or `karume-anima` when `--source` is omitted. The file name spells the knobs that change
the image:

```
anima-<quant>-<W>x<H>-<steps>step[-<sampler>]-seed<seed>.png
```

`<quant>` and `<steps>` read `default` and `defaultstep` when you did not pass them. The sampler
segment appears only when you pass `--sampler`, so a run without it keeps the name that
`eval-images.ts` (below) and the recipes that read its corpus expect. Because of that, a run that
leaves `--sampler` out and one that names the manifest's own sampler write the same image under two
different names. `outputs/` is untracked; the layout of its directories, and which of them are
safe to delete, is in [`docs/assets-layout.md`](../../docs/assets-layout.md).

The script prints one line before loading, a load-progress indicator on stderr, and the written path
with the elapsed time, loading included. The lines are shaped like this:

```
[anima] models/karume-anima / model （manifest の既定） / quant （manifest の既定） / sampler dpmpp-2m / seed 7
[anima] outputs/examples/karume-anima/anima-default-1024x1024-defaultstep-dpmpp-2m-seed7.png（<seconds>s）
```

## Comparing samplers

The distribution declares a default update rule in `pipelineConfig.scheduler.type`. The published
manifests declare `euler`, matching the upstream recommendation. `dpmpp-2m` (DPM++ 2M) is the other
choice. The sampler does not change which weights are loaded, and both rules run the model once per
step, so equal step counts mean equal compute.

To compare the two, hold every other knob fixed and name both samplers explicitly:

```sh
for seed in 42 43 44 45; do
  for sampler in euler dpmpp-2m; do
    deno task demo:anima --model anima-v1.0 --steps 20 --seed "$seed" --sampler "$sampler"
  done
done
```

Each pair differs only in the `-euler` / `-dpmpp-2m` segment of the file name, so the runs never
overwrite each other. Compare several seeds: a single seed can favor either sampler by chance.
The elapsed time on the last line includes loading, which every invocation repeats, so it is not a
speed comparison between samplers.

## Regenerating the evaluation images

`eval-images.ts` re-renders the four sample images (seeds 42 to 45) that several tests use as
real-image inputs: the depth and BiRefNet end-to-end tests, and on the SigLIP2 side the real-image
preprocessing test and its golden files. It runs `main.ts` once per image with a fixed prompt, seed
and resolution, then moves the results into a dated directory:

```
deno task demo:eval-images --source models/karume-anima
```

`--source` is required and is the only option. The images land in
`outputs/bench/<source>/<date>_eval-images/`. The tests read a frozen copy under
`outputs/misc/corpus/`, not this directory, so copying a new set there is a manual step, and the
golden files have to be regenerated afterwards. The script prints only the SigLIP2 golden command;
the depth and BiRefNet golden files need regenerating as well, and the script does not say how. It
never passes `--quant`, `--steps` or `--sampler`, because each of those would rename the output
files away from the names the recipes read.
