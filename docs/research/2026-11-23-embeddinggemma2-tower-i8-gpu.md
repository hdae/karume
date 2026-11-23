> A point-in-time measurement record (2026-11-23): the `i8` preset of EmbeddingGemma 2 with its int8 vision and
> audio towers, run on the GPU through the real pipeline and through the towers' series, against the `bf16` preset
> and against the torch-CPU screening of the same int8 forms
> ([research 2026-11-23](2026-11-23-embeddinggemma2-tower-quant-screen.md)). It is the GPU half of the towers' part
> of the `i8` preset (step 3 of the EmbeddingGemma 2 item in [backlog](../backlog.md)). Every number is copied from
> the records and test output named below. Nothing here is a design source of truth or a decision, and it is not
> edited to follow later events.

# EmbeddingGemma 2 int8 towers on the GPU

## What was measured and how

- **Machine**: the RTX 3080 Ti of the development machine (Node v24.22.3, Dawn / Vulkan, driver 620.70.06), the
  local mirror `models/karume-embeddinggemma-2` as assembled that day (the `i8` preset maps `model`, `vision` and
  `audio` to their i8 containers).
- **The preset comparison**:
  [`tools/embeddinggemma2-quality-set/compare-presets.ts`](../../tools/embeddinggemma2-quality-set/compare-presets.ts),
  run twice as `flock /tmp/karume-gpu.lock node tools/embeddinggemma2-quality-set/compare-presets.ts [<dir>]`. It
  builds `embeddinggemma.Pipeline` once per preset with `vision: true` and `audio: true` (both resident on one
  adapter) and embeds the 32 images of `inputs/embeddinggemma2/quality-set/` and the 32 clips of
  `inputs/embeddinggemma2/quality-set-16k/` (the 16 kHz, 16-bit WAVs the screening's re-run wrote, so the pipeline
  reads the screening's own samples), each alone and without a prompt, and the 16 distinct captions and 8 distinct
  transcripts with the `query` prompt. Every input runs twice per preset: a warm-up pass that gives the vectors, then
  a timed pass that must give the same bits, the two presets alternating which runs first. The raw records
  (untracked): `outputs/bench/embeddinggemma-2/2026-11-23_tower-i8-gpu/record.json` (run 1) and
  `…/2026-11-23_tower-i8-gpu-run2/record.json` (run 2). The two runs gave the same vectors bit for bit.
- **The towers' series**: the series e2e (`e2e_embeddinggemma2_vision_test.ts`, `e2e_embeddinggemma2_audio_test.ts`)
  over `outputs/series/embeddinggemma-2-vision-i8/` and `…-audio-i8/`, each case against its own goldens (upstream's
  tower on the rounded weights, torch CPU f32), and the bf16 and f32 series in the same run for the timings.

## Cosine of the `i8` vector to the `bf16` vector, per item

At 768 and 128 values (the shorter width is the leading slice re-normalised; its cosine is the slices' cosine). The
screening's column is its `all-i8` configuration against its reference on the same items (torch CPU, the quality
set's items only); the last column is the largest difference of one item's cosine between the two.

| Items | Width | GPU min | GPU mean | Worst item | Screening min / mean | Largest per-item difference |
| --- | --- | --- | --- | --- | --- | --- |
| 32 images | 768 | 0.999648 | 0.999812 | `anima-coffee` | 0.999648 / 0.999812 | 4.39e-8 (`anima-pancakes`) |
| 32 images | 128 | 0.999802 | 0.999892 | `anima-coffee` | 0.999802 / 0.999892 | 3.02e-8 (`anima-pancakes`) |
| 32 clips | 768 | 0.999493 | 0.999759 | `M2-03` | 0.999493 / 0.999759 | 2.40e-8 (`M2-03`) |
| 32 clips | 128 | 0.999651 | 0.999858 | `M2-03` | 0.999651 / 0.999858 | 1.41e-8 (`M2-03`) |

The texts, through the text tower alone: captions 0.999857 / 0.999909 at 768 and 0.999907 / 0.999944 at 128, the
transcripts 0.999813 / 0.999882 and 0.999881 / 0.999924 (min / mean).

## Cross-modal top-1

Item → text among the distinct texts, text → item among the items (a hit is any item of the text — four clips share
each transcript); "shared" is how many queries both presets answer with the same candidate.

| Modality | Width | item → text, `bf16` / `i8` / shared | text → item, `bf16` / `i8` / shared |
| --- | --- | --- | --- |
| images | 768 | 32 / 32 / 32 of 32 | 16 / 16 / 16 of 16 |
| images | 128 | 32 / 32 / 32 of 32 | 16 / 16 / 16 of 16 |
| clips | 768 | 26 / 26 / 32 of 32 | 8 / 8 / 7 of 8 |
| clips | 128 | 26 / 26 / 31 of 32 | 8 / 8 / 8 of 8 |

The screening's hits and agreements are the same numbers at both widths. At 768 one transcript's first clip
differs between the presets (both of that transcript); at 128 one clip's first transcript differs (both wrong in
`bf16` and `i8` alike, the hit counts being equal).

## Time per item

The whole `embed` call of the timed pass, in ms (the pipeline does not report its tower runs apart):

| Inputs | Preset | Run 1 mean / median | Run 2 mean / median |
| --- | --- | --- | --- |
| 32 images | `bf16` | 632.7 / 197.1 | 260.3 / 195.9 |
| 32 images | `i8` | 192.4 / 191.6 | 195.7 / 195.1 |
| 32 clips | `bf16` | 129.5 / 103.2 | 128.3 / 104.3 |
| 32 clips | `i8` | 101.6 / 103.9 | 101.4 / 102.1 |
| 16 captions | `bf16` | 30.6 / 29.6 | 66.7 / 38.8 |
| 16 captions | `i8` | 24.4 / 23.9 | 26.5 / 28.1 |
| 8 transcripts | `bf16` | 34.3 / 40.7 | 35.9 / 40.4 |
| 8 transcripts | `i8` | 30.4 / 36.4 | 31.1 / 36.0 |

The medians of the two presets are within 3% of each other for images and clips. The means are not, because of
single slow calls: a call 2.5 times or more slower than the other preset's call of the same input was a `bf16`
call every time in the timed passes — seven in run 1 (535 to 3,828 ms; four consecutive images at 3.1 to 3.8 s
against 176 to 193 ms in `i8`, and one short clip at 758 ms against 80 ms) and five in run 2 (575 to 990 ms, one of
them a caption). The warm-up passes had such calls in both presets (run 2: `bf16` 2.4 to 3.5 s on five images,
`i8` 3,244 ms on one), and took 23.6 s in run 1 and 45.3 s in run 2. The cause was not investigated, so the means
are no comparison of the presets. Building the pipeline with both towers took 1.92 / 1.90 s for `bf16` and 1.38 /
1.29 s for `i8`.

## The towers' Sessions

From the series e2e of the same day (the first and the second run of one case for the vision tower, the median of
five warm runs for the audio tower, in ms; weights as the Session holds them):

| Tower | Storage | Weights | Session build | Case runs |
| --- | --- | --- | --- | --- |
| vision | bf16 | 320.3 MiB | 324 to 385 ms | 134 to 149 (250 to 280 soft tokens), 75 (139) |
| vision | i8 | 161.1 MiB | 237 to 269 ms | 138 to 145 (250 to 280 soft tokens), 74 (139) |
| audio | bf16 | 560.1 MiB | 496 to 637 ms | 16.3 to 16.6 (25 and 26), 20.7 (63), 33.8 (250), 94.6 (750) |
| audio | i8 | 281.6 MiB | 285 to 414 ms | 10.6 to 10.8 (25 and 26), 13.9 (63), 34.0 (250), 93.0 (750) |

Against their own goldens the i8 towers sit where the f32 series sits from its goldens: the vision tower's soft
tokens within 5.15e-5 to 3.56e-4 (max abs, the worst token's cosine at least 1 − 5.55e-9), the audio tower's within
1.03e-5 to 2.49e-4 (at least 1 − 1.11e-9); the rounding itself moves them by 0.29 to 0.85 (vision) and 0.071 to 3.0
(audio). The 30 s golden on a device capped at 128 MiB of binding and buffer gives the uncapped bits for the i8
audio tower, and so do an image of 280 soft tokens and a 30 s audio through the `i8` pipeline on such a device: the
int8 towers add no `requiredLimits`.

## Through the pipeline e2e

The `i8` preset's vectors against upstream's SentenceTransformer vectors of the tracked multimodal fixture (the
whole model in bf16 upcast to f32): 0.999627 to 0.999881 over its five image inputs (0.999759 to 0.999908 with the
bf16 tower before), 0.999762 to 0.999907 over its four audio inputs (0.999819 to 0.999909 before). Each text through
the pipelines with towers keeps its text pipeline's sha256 row.
