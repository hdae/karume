# Style-Bert-VITS2 text-to-speech demo

Turns one Japanese sentence into a WAV file with `Sbv2Pipeline`, in Deno + WebGPU. It is also the
worked example for the split the package draws: **text analysis belongs to the caller**. The script
runs [`@hdae/yomi`](https://jsr.io/@hdae/yomi) to get readings and accents, converts the result
with `toSbv2Utterance`, and hands `generate` that already-analyzed utterance. To fix a reading, put
yomi's overlay dictionary in front of the analysis; the model never sees raw text.

The script needs a WebGPU adapter and network access on the first run (see Constraints). Run it
from the repository root.

## Prerequisite: a distribution

`--source` is required and has no default. Local mirrors are generated per machine, so a default
path would turn into "directory not found" or an unintended repository fetch elsewhere.

The standard local mirror is `models/karume-sbv2-jvnv` — four JVNV speakers in one distribution. It
is assembled from exported series in `tools/export-recipes/`:

```sh
uv run python dist.py --pipeline sbv2 \
    --model F1 --model F2 --model M1 --model M2 --out ../../models/karume-sbv2-jvnv \
    --repo hdae/karume-sbv2-jvnv
```

The series come from the real checkpoints under `inputs/sbv2/<speaker>/`; how to obtain them and
export each storage form is in the [SBV2 export recipe](../../tools/export-recipes/sbv2/README.md).

`--source` accepts either form:

- a directory that carries `karume.json` — read in place, with nothing copied or cached;
- anything else — read as a Hugging Face repository name at `main` (a bare name is not pinned).
  This version reads only `karume/5` manifests; a repository whose `main` holds an older manifest
  fails with an unsupported-format error rather than loading.

`--source-map <owner/name=<dir>>` (repeatable) names a local directory for a repository that the
distribution references from outside itself. It is accepted only with a local `--source`.

## Running it

```sh
deno task demo:sbv2 --source models/karume-sbv2-jvnv --text "こんにちは、これはテストです。"
```

The first line echoes the source, model, quant, seed, and text. Load progress is overwritten on
stderr, and the run ends with a line of this shape:

```
[sbv2] outputs/examples/karume-sbv2-jvnv/sbv2-default-default-seed0.wav（<length>s / <elapsed>s）
```

`<length>` is the audio length in seconds and `<elapsed>` the wall-clock time from the start of text
analysis, including model loading.

Another speaker, quant, and style:

```sh
deno task demo:sbv2 --source models/karume-sbv2-jvnv --model F2 --quant f16+bert8 \
  --style Happy --seed 7 --text "今日はとてもいい天気ですね。" \
  --out outputs/examples/karume-sbv2-jvnv/F2-happy-seed7.wav
```

## Options

All options are `--key value` pairs. Unknown keys, and values that start with `--`, are rejected
rather than silently ignored, so a mistyped knob never runs on its default. The synthesis knobs
you leave out (`--style` through `--length-scale`) take the value the distribution declares in
`pipelineConfig.defaults` (shown for the standard mirror). The exceptions: `--model` and `--quant`
fall back to the manifest's `defaultModel` / `defaultQuant`, and `--seed` falls back to `0`, a
default of this script.

| Option                          | Default                          | What it does                                                                   |
| ------------------------------- | -------------------------------- | ------------------------------------------------------------------------------ |
| `--source <dir\|owner/name>`    | required                         | Distribution to read.                                                          |
| `--source-map <owner/name=dir>` | none                             | Local directory for a cross-repository reference (local `--source` only).      |
| `--text <text>`                 | `こんにちは、これはテストです。` | Text to speak.                                                                 |
| `--model <name>`                | manifest default (`F1`)          | Speaker model: `F1`, `F2`, `M1`, `M2` in the standard mirror.                  |
| `--quant <name>`                | manifest default (`i8+bert4`)    | Storage / execution variant (table below).                                     |
| `--style <name>`                | `Neutral`                        | `Neutral`, `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`.             |
| `--style-weight <x>`            | `1.0`                            | How strongly the style vector is applied.                                      |
| `--sdp-ratio <x>`               | `0.2`                            | Mix between the stochastic and the deterministic duration predictor.           |
| `--noise-scale <x>`             | `0.6`                            | Noise scale of the prior.                                                      |
| `--noise-scale-w <x>`           | `0.8`                            | Noise scale of the stochastic duration predictor.                              |
| `--length-scale <x>`            | `1.0`                            | Speaking-rate factor; larger is slower.                                        |
| `--seed <n>`                    | `0`                              | Sampling seed (non-negative integer). Same seed, same environment: same audio. |
| `--out <path>`                  | see Output                       | Output WAV path.                                                               |

The accepted style and speaker names come from the distribution's `pipelineConfig`, not from the
package. The quant names below are the ones the standard mirror declares; `karume.json` of the
distribution you read is the authority. `bert` refers to the text encoder.

| `--quant`   | Label in the manifest               | Summary                                                                            |
| ----------- | ----------------------------------- | ---------------------------------------------------------------------------------- |
| `f16+bert8` | Highest fidelity (f16 synthesis)    | Synthesis stored as f16, int8 text encoder — the largest download.                 |
| `i8`        | Half size (int8)                    | Every component stored as int8, computed in f32.                                   |
| `i8-a8`     | int8 with int8 linear activations   | The int8 weights plus int8 activations in the linear layers.                       |
| `i8+bert4`  | Balanced (int8 + int4 text encoder) | int8 synthesis, text encoder in GPTQ int4 (one scale per 32 weights). The default. |
| `i4`        | Smallest (int4)                     | Every component in int4 (one scale per 32 weights) — the smallest download.        |

## Output

A mono 16-bit PCM WAV at the model's sample rate (44.1 kHz for the JVNV speakers). Without `--out`
it goes to `outputs/examples/<source name>/sbv2-<quant>-<style>-seed<n>.wav`, where `<source name>`
is the last path element of `--source` and `<quant>` / `<style>` are `default` when not given. The
name does not include the speaker model or the text, so runs that differ only in those overwrite
each other — pass `--out` to keep both.

## Constraints

- **The first run downloads the yomi dictionary** (about 19 MB). Later runs read it from the Cache
  API.
- Both the text encoder's token count and the phoneme count are limited by the distribution's
  `maxTokens` (512 in the standard mirror). A longer utterance is rejected with `Sbv2InputError`
  rather than truncated; split the text and synthesize the parts separately.
- The total number of output frames is limited by the distribution's `maxFrames` (4096 in the
  standard mirror). Long text or a large `--length-scale` can exceed it; that is also rejected with
  `Sbv2InputError`. Split the text or lower `--length-scale`.
- A style name outside the distribution's `pipelineConfig` is rejected, not mapped to a default.
  (The CLI has no speaker option; API callers who pass a speaker name outside `pipelineConfig` get
  the same rejection.)

## Developer tool: the torch parity dump (`dump.ts`)

`deno task dump:sbv2 --source models/karume-sbv2-jvnv --text "…"` runs the same synthesis chain as
the pipeline and additionally writes its discrete inputs and random sequence to
`dump.safetensors` (plus `out.wav`) under `outputs/examples/<source name>/sbv2-dump/`. The torch
side of the comparison replays that dump; the full procedure is under "Asset prep and torch
reference for the voice demo" in the [SBV2 export recipe](../../tools/export-recipes/sbv2/README.md).
It reads local distributions only, and it reaches into the package's internal modules on purpose,
so it is not an example of the public API.
