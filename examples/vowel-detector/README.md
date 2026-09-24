# Vowel detector demo (speech to lip-sync `.lab`)

Runs `VowelDetectorPipeline` on one Japanese speech recording and writes the lip-sync timeline as a
`.lab` file, in Deno + WebGPU. Feature extraction and post-processing run on the host; the CRNN
itself runs on the GPU at the clip's own length (no padding, no length buckets).

The script needs a WebGPU adapter. Run it from the repository root.

## Prerequisite: a local distribution

**There is no published distribution for this family yet**, and therefore no pinned source
constant: `--source` is required and has no default. You build the distribution from the upstream
training checkpoint with the
[vowel-detector export recipe](../../tools/export-recipes/vowel_detector/README.md). From
`tools/export-recipes/`:

```sh
# one-time inputs: inputs/vowel-detector/crnn_epoch3.pt and the upstream feature_config.json
uv run python -m vowel_detector.export              # → outputs/series/vowel-detector-crnn-epoch3/
cp <upstream>/assets/feature_config.json ../../inputs/vowel-detector/
uv run python dist.py --pipeline vowel-detector     # → models/karume-vowel-detector/
```

The result carries one model (`crnn-epoch3`) in a single f32 storage form, so `--model` and
`--quant` can be left out.

`--source` also accepts a Hugging Face repository name (anything that is not a directory carrying
`karume.json` is read as one, at `main`), for a distribution you publish yourself.
`--source-map <owner/name=<dir>>` (repeatable) names a local directory for a repository that the
distribution references from outside itself; it is accepted only with a local `--source`.

## Prerequisite: a 16 kHz recording

**The input must be a 16 kHz WAV.** The pipeline has no resampler: audio at another rate would not
fail inside the model, it would produce a different, wrong vowel sequence. The script therefore
compares the WAV's rate with the distribution's declared rate right after loading and stops on a
mismatch. The WAV reader accepts 16-bit PCM and 32-bit float; multi-channel audio is averaged to
mono.

If you have no 16 kHz Japanese speech at hand, the
[Irodori demo](../irodori/README.md#regenerating-the-vowel-detector-evaluation-audio-eval-audiots)
generates four evaluation recordings, already downsampled to 16 kHz. The end-to-end tests do not
read this output directly: they read a copy frozen by hand under `outputs/misc/corpus/`, and a
fresh run can differ from that copy. One of the recordings reads the five vowels in order, so its
`.lab` can be checked by eye:

```sh
deno task demo:eval-audio --source models/karume-irodori-v4.1-small
# → outputs/bench/vowel-detector/<UTC date>_eval-audio/vowel-{short,vowels,mid,long}.wav
```

`--source` is required there and takes any Irodori distribution, local or on Hugging Face.

## Running it

```sh
deno task demo:vowel-detector --source models/karume-vowel-detector \
  --audio outputs/bench/vowel-detector/$(date -u +%F)_eval-audio/vowel-vowels.wav
```

`$(date -u +%F)` assumes you ran `eval-audio` today (UTC); otherwise put the UTC date of that run
in the path.

The first line echoes the source, model, and quant, followed by the input path, its length, and
its sample rate. Load progress is overwritten on stderr. The `.lab` body is then printed to stdout,
and the run ends with a line of this shape:

```
[vowel-detector] outputs/examples/karume-vowel-detector/vowel-vowels.wav.lab（<segments> 区間 / <elapsed>s）
```

`<segments>` is the number of `.lab` lines and `<elapsed>` the wall-clock time including model
loading. For the `vowels` recording, the vowel labels should appear in the order `a i u e o`.

## Options

All options are `--key value` pairs. Unknown keys, and values that start with `--`, are rejected
rather than silently ignored, so a mistyped knob never runs on its default.

| Option                          | Default          | What it does                                                              |
| ------------------------------- | ---------------- | ------------------------------------------------------------------------- |
| `--audio <wav>`                 | required         | 16 kHz speech recording.                                                  |
| `--source <dir\|owner/name>`    | required         | Distribution to read.                                                     |
| `--source-map <owner/name=dir>` | none             | Local directory for a cross-repository reference (local `--source` only). |
| `--model <name>`                | manifest default | Model inside the distribution.                                            |
| `--quant <name>`                | manifest default | Storage / execution variant.                                              |
| `--out <path>`                  | see Output       | Output `.lab` path.                                                       |

## Output

A `.lab` text file with one segment per line: start time, end time (both in seconds, seven decimal
places), and a label, separated by spaces. Labels are the five vowels `a` `i` `u` `e` `o`, plus
`N` (moraic nasal), `pau` (pause), and `cons` (consonant). Segment boundaries sit on a 20 ms grid.

Without `--out` the file goes to `outputs/examples/<source name>/<audio file name>.lab`, where
`<source name>` is the last path element of `--source`.

## Constraints

- The recording must be at least the distribution's `minFrames` and at most its `maxFrames` long
  (both counted in 10 ms frames, declared in `pipelineConfig`). Outside that range the pipeline
  throws `ModelInputError` naming the limit in seconds, instead of truncating or padding: truncation
  would silently drop the end of the timeline, and padding changes the result because the backward
  GRU carries state back from the padded tail. Split long recordings yourself.
- An odd trailing 10 ms frame is dropped, because the network outputs one frame per two input
  frames.
