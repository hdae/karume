# Irodori-TTS text-to-speech demo

Turns one Japanese sentence into a WAV file with `IrodoriPipeline`, in Deno + WebGPU. The only
branch in the script is where the distribution comes from; everything after that is the knobs of a
single `generate` call. Two optional inputs steer the voice: a reference recording (`--ref`, voice
cloning) and a free-text description of the voice (`--caption`, Voice Design).

The script needs a WebGPU adapter. Run it from the repository root.

## Quick start

With no `--source`, the script reads the local mirror `models/karume-irodori-v4.1-small` when that
directory exists, and otherwise fetches the revision this package version pinned in
`IRODORI_SOURCES["irodori-v4.1-small"]` from Hugging Face. Either way, no preparation is needed:

```sh
deno task demo:irodori --text "こんにちは、これはテストです。"
```

The first line names the source that was actually picked — the mirror path, or
`hdae/karume-irodori-v4.1-small@<commit SHA>` for the pinned fetch. Download and load progress is
overwritten on stderr, and the run ends with a line of this shape:

```
[irodori] outputs/examples/karume-irodori-v4.1-small/irodori-default-no-ref-seed0.wav（<length>s / S <frames> / dit <runs> forward / <elapsed>s）
```

`<length>` is the audio length in seconds, `S` the number of latent frames that were generated,
`dit` the number of DiT runs the sampler made, and `<elapsed>` the wall-clock time including model
loading.

That WAV is also a valid reference for voice cloning, which makes the second step self-contained:

```sh
deno task demo:irodori \
  --text "今日はとてもいい天気ですね。" \
  --ref outputs/examples/karume-irodori-v4.1-small/irodori-default-no-ref-seed0.wav
# → outputs/examples/karume-irodori-v4.1-small/irodori-default-cloned-seed0.wav
```

A caption describes the voice instead of copying one:

```sh
deno task demo:irodori --caption "落ち着いた女性の声で、ゆっくりと話している。" --seed 7 \
  --out outputs/examples/karume-irodori-v4.1-small/caption-seed7.wav
```

## Where the model comes from

| How you run it                | What is read                                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| no `--source`, mirror present | `models/karume-irodori-v4.1-small` (must carry `karume.json`, or the script stops instead of falling back to the network) |
| no `--source`, mirror absent  | `IRODORI_SOURCES["irodori-v4.1-small"]` — a pinned commit, so the fetch is reproducible and works offline once cached     |
| `--source <dir>`              | that local distribution directory (any directory that carries `karume.json`)                                              |
| `--source <owner/name>`       | that Hugging Face repository at `main` (a bare repository name follows `main`; it is not pinned)                          |

The local mirror is built from exported series with the
[Irodori export recipe](../../tools/export-recipes/irodori/README.md) — see its section on
`v4.1-small` — and `uv run python dist.py --pipeline irodori --model v4.1-small` in
`tools/export-recipes/`. This version reads only `karume/5` manifests; a distribution in an older
manifest format fails with an unsupported-format error rather than loading.

`--source-map <owner/name=<dir>>` (repeatable) names a local directory for a repository that the
distribution references from outside itself. It applies only together with `--source <dir>`; the
script rejects it otherwise instead of ignoring it.

## Options

All options are `--key value` pairs. Unknown keys, and values that start with `--`, are rejected
rather than silently ignored, so a mistyped knob never runs on its default.

| Option                          | Default                          | What it does                                                                                   |
| ------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------- |
| `--source <dir\|owner/name>`    | see above                        | Distribution to read.                                                                          |
| `--source-map <owner/name=dir>` | none                             | Local directory for a cross-repository reference (with `--source <dir>` only).                 |
| `--text <text>`                 | `こんにちは、これはテストです。` | Text to speak.                                                                                 |
| `--caption <text>`              | none                             | Voice description (Voice Design). An empty caption means no caption.                           |
| `--ref <wav>`                   | none                             | Reference recording to clone the voice from.                                                   |
| `--model <name>`                | manifest default                 | Model inside the distribution. The standard mirrors carry one each (`v4.1-small`, `v4-small`). |
| `--quant <name>`                | manifest default (`i8-a8`)       | Storage / execution variant (table below).                                                     |
| `--seconds <x>`                 | predicted                        | Utterance length in seconds. Skips the duration predictor.                                     |
| `--seed <n>`                    | `0`                              | Sampling seed (non-negative integer). Same seed, same environment: same audio.                 |
| `--out <path>`                  | see below                        | Output WAV path.                                                                               |

The quant names below are the ones the standard mirror declares; `karume.json` of the distribution
you read is the authority.

| `--quant` | Label in the manifest        | Summary                                                                     |
| --------- | ---------------------------- | --------------------------------------------------------------------------- |
| `f32`     | Full precision (f32)         | Every graph in f32 — the largest download.                                  |
| `f16`     | Half size (f16)              | Every graph stored as f16, computed in f32.                                 |
| `i8`      | Quarter size (int8)          | Every graph stored as int8, computed in f32.                                |
| `i8-a8`   | Balanced (int8, int8 linear) | int8 weights plus int8 activations in the DiT's linear layers. The default. |
| `i8+dit4` | Lowest memory (int4 DiT)     | The DiT in GPTQ int4 (one scale per 32 weights), the other graphs int8.     |

## Output

A mono 16-bit PCM WAV at the distribution's sample rate (48 kHz for the standard mirrors). Without
`--out` it goes to `outputs/examples/<source name>/irodori-<quant>-<no-ref|cloned>-seed<n>.wav`,
where `<source name>` is the last path element of `--source` (or of the pinned repository name) and
`<quant>` is `default` when `--quant` is not given. The name does not include the text or the
caption, so two runs that differ only in those overwrite each other — pass `--out` to keep both.

## Constraints

- **The reference must already be at the distribution's sample rate.** The pipeline has no
  resampler and fails loudly on a mismatch. The WAV reader accepts 16-bit PCM and 32-bit float;
  multi-channel audio is averaged to mono.
- **Sampler knobs are fixed by the distribution.** The step count and the CFG scales come from the
  manifest's `pipelineConfig`; only `--seed` and `--seconds` change a run.
- `--seconds` is clamped to the distribution's `minSeconds`..`maxSeconds` (0.5–30 s in the
  standard mirrors).
- Text longer than the distribution's `maxTextLen` (256 tokens, including the leading token, in the
  standard mirrors) is cut to fit, as upstream does. Text that is empty after normalization is
  rejected.

## Regenerating the vowel-detector evaluation audio (`eval-audio.ts`)

`deno task demo:eval-audio --source <dir|owner/name>` runs this CLI four times with fixed texts and
seeds, then downsamples each 48 kHz result to 16 kHz for the
[vowel-detector demo](../vowel-detector/README.md) and its end-to-end tests. `--source` is
required. The files go to `outputs/bench/vowel-detector/<UTC date>_eval-audio/`
(`vowel-<case>.wav` at 16 kHz, `vowel-<case>-48k.wav` as generated), with the SHA-256 of each
16 kHz file printed. The cases are `short`, `vowels`, `mid`, and `long`.
