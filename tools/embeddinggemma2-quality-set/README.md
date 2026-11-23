# embeddinggemma2-quality-set — generate the quality set of EmbeddingGemma 2

The set of captioned images and transcribed speech used to check the quantized image and audio towers of EmbeddingGemma 2
against its reference preset. Karume's own pipelines make it on Node + Dawn from the local mirrors under `models/`, so it
needs no outside material:

| Kind | Count | Generator | Files |
| ---- | ----- | --------- | ----- |
| images | 16 | Anima (`models/karume-anima`, `anima-turbo-v1.1`, its default preset) | `images/anima-<subject>.png` |
| images | 16 | MOSS-Image (`models/karume-moss-image`, `sft`, its default preset) | `images/moss-<subject>.png` |
| speech | 32 | SBV2 (`models/karume-sbv2-jvnv`, speakers F1 F2 M1 M2 × 8 sentences) | `audio/<speaker>-<nn>.wav` |

Both image generators draw the same 16 everyday subjects with the same seed and size (square, landscape and portrait
sizes, so the vision tower sees more than one aspect), so the set holds near-duplicates across two styles. The
sentences run from about 1 s to 12 s of speech. PNGs are 8-bit RGB; WAVs are 16-bit PCM mono at SBV2's own rate (not
resampled — a consumer that needs 16 kHz resamples). The subjects, prompts, seeds and sentences are in `set.ts`.

The set goes to `inputs/embeddinggemma2/quality-set/` (outside git) with its `manifest.json`:

```json
{
  "format": "karume-embeddinggemma2-quality-set/1",
  "images": [{ "file": "images/anima-fox.png", "caption": "A red fox sitting in fresh snow.", "generator": "karume-anima/anima-turbo-v1.1", "prompt": "…", "seed": 1000 }],
  "audio": [{ "file": "audio/F1-01.wav", "transcript": "…", "generator": "sbv2-jvnv/F1", "sampleRate": 44100 }]
}
```

## The command

From the repository root, on the GPU lock:

```sh
flock /tmp/karume-gpu.lock node tools/embeddinggemma2-quality-set/generate.ts [<output directory>] [--limit <n>]
```

- `--limit <n>` keeps the first `n` items of each generator — of Anima, of MOSS-Image and of each SBV2 speaker
  (`--limit 1` is a smoke run of 2 images and 4 clips).
- Every mirror is opened before the GPU is touched: a missing one fails at once.
- An earlier `manifest.json` is removed first and the new one is written last, so a run that fails leaves none.
- SBV2's reading dictionary is fetched by `@hdae/yomi` at the revision its version pins (on every run: Node has no
  Cache Storage).
