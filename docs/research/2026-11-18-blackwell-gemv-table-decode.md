> A point-in-time measurement record (2026-11-18): Gemma 4 decode tokens on the RTX 5070 Ti in Chrome with the table `nvidia-blackwell`'s GEMV choices as registered at checkout `2fcacc4e`, against the default profile's.

# Gemma 4 decode with `nvidia-blackwell`'s GEMV choices (RTX 5070 Ti, Chrome)

The table `nvidia-blackwell` carries GEMV choices of its own since 2026-11-18
([spec](../spec/geometry-profiles.md#the-gemv-choices-of-linear)). This record is the first decode token measured
with them. It measured the table as registered at `2fcacc4e` (derived from 74 GEMV sweep records); the table was
derived again the same evening from 82 records, and that table's own effect is not measured (§4). The records and the
script are local and untracked: `outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemma4-bench_2026-11-18T16-27-38`,
`…T16-28-58`, `…T16-29-40`, `…T16-31-02` (batch `02-blackwell-gemv-table-decode-and-bf16`, steps 1 to 4) and
`outputs/diag/decode-optimization-2026-11-18/blackwell-regenerate/`.

## 1. Setup

- The RTX 5070 Ti, Chrome 157 on Windows, D3D12; checkout `2fcacc4e`, clean.
- The workbench's `gemma4-bench` entry: greedy, the prompt `few-ids`, 96 new tokens, 3 chats per visit, 4 rounds
  with the arms interleaved, one device per visit. The first chat of a visit is cold and left out; a visit's value is
  the mean over its warm chats, and an arm's value the median over its four visits. The difference is paired by
  round.
- Each record holds pairs of arms on one model and preset. The first arm runs the table's choices. The second runs:
  - for the QAT rows, the switch `default-gemv`: the default profile's GEMV choices (23 sequential exceptions and
    71 parallel rows, where the table has 65 and 87);
  - for regular E2B `bf16`, the switch `general-sequential`: the table's choices without any per-shape exception of
    the sequential variant.
- At `exact` (QAT `i4`, regular `bf16`) a Session runs no parallel row, so the difference is the sequential
  exceptions'. At `reorder` (`i4-fast`) it is the sequential exceptions' and the parallel rows' together.

## 2. Results

The interval between two tokens and the GPU time per token, ms (median over the visits); "faster" counts the rounds
in which the table's arm was faster.

| Model, preset | Second arm | Interval: table / second | Ratio | Faster | GPU: table / second | Ratio | Faster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QAT E2B `i4-fast` | `default-gemv` | 5.655 / 5.704 | ×1.009 | 4 of 4 | 4.500 / 4.569 | ×1.015 | 4 of 4 |
| QAT E4B `i4-fast` | `default-gemv` | 8.990 / 9.220 | ×1.026 | 4 of 4 | 7.608 / 7.944 | ×1.044 | 4 of 4 |
| QAT 12B `i4-fast` | `default-gemv` | 15.297 / 18.858 | ×1.233 | 4 of 4 | 13.993 / 17.459 | ×1.248 | 4 of 4 |
| QAT E2B `i4` | `default-gemv` | 12.363 / 12.601 | ×1.019 | 4 of 4 | 11.022 / 11.236 | ×1.020 | 4 of 4 |
| Regular E2B `bf16` | `general-sequential` | 19.142 / 22.867 | ×1.195 | 4 of 4 | 17.725 / 21.505 | ×1.213 | 4 of 4 |

- 12B `i4-fast` decodes 64.5 tokens per second with the table's choices and 52.5 with the default's (the records'
  `decodeTokensPerSecond`, median over the visits).
- The token ids are the same in both arms of every pair: one sha256 per pair.
- Regular E2B `bf16` at `exact` runs the 22 bf16 exceptions of the default profile, which the table inherits: this
  row is their decode token on this device (on the RTX 3080 Ti: 18.83 against 23.42 ms,
  [the sequential variants](2026-11-18-sequential-gemv-variants.md) §5).

## 3. How far the small differences go

- The batch ran no control (two arms with the same choices). On the RTX 3080 Ti a 4-round control read GPU ×1.033 in
  4 of 4 rounds and an 8-round one ×0.990 ([the sequential variants](2026-11-18-sequential-gemv-variants.md) §5).
  By that measure the E2B and E4B rows (×1.009 to ×1.044) are not resolved beyond the noise of 4 rounds; 12B
  (×1.233) and regular E2B (×1.195) are well past it.

## 4. What goes stale

- The table derived from 82 records the same evening differs from the one measured on two fields a measured Session
  reads: regular E2B's bf16 `1536 × 256` runs `c16u8` in place of `c32u16` at `exact`, and QAT E4B's int4
  `2048 × 2560` (group 32) runs 16 lanes, staged, in place of 4 at `reorder`. The regular E2B `bf16` row and the
  QAT E4B `i4-fast` row are of the table before that; the QAT E2B and 12B rows run the same choices under both. The
  22 bf16 parallel rows the new table adds are reached only at `reorder`, which no preset of regular Gemma 4
  declares.

## 5. Not measured

- The 26B model, a speculative decode with the drafter, a sampled decode, regular E4B.
- The table derived from 82 records.
- The mobile QAT line.
