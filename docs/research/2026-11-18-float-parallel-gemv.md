# The parallel GEMV on float-stored weights: regular Gemma 4 in bf16 (2026-11-18)

A point-in-time measurement record. It says how much of a decode token of regular Gemma 4 (`bf16`) is the sequential
GEMV, what the parallel GEMV does to each of its shapes at 1, 4 and 8 rows, why no shape has a row in the default
profile's table yet, and what a decode token costs with linear at `reorder` once the measured rows are in place.
Scripts, logs and tables: `outputs/diag/decode-optimization-2026-11-18/bf16-gemv/` (outside git).

## 1. Summary

- A decode token of regular E2B `bf16` is 277 linears on the sequential GEMV: 76.5 % of the token's GPU time
  (19.8 of 25.8 ms in a per-dispatch split; the token takes 25.6 ms of wall time unsplit). They read 4.56 GB of bf16
  weights per token (E4B: 344 linears, 9.29 GB).
- The parallel GEMV on bf16 words is ×1.4–5.6 faster than the sequential GEMV per dispatch at one row on every E2B
  shape, and ×1.3–5.0 on every E4B shape. On the large shapes it reaches 46 µs for 37.7 MB of weights, about
  820 GB/s against the card's nominal 912 GB/s, where the sequential kernel reaches 150–470 GB/s.
- The gain comes in two kinds. The reducing shapes (few outputs, long reductions: `o`, `down`) and the small expanding
  ones gain ×3–5.6: one thread walking a long row is what lanes remove. The wide expanding shapes (`gate` / `up`, the
  output head) gain ×1.7–1.9, and a sequential variant that reads eight words ahead instead of four already gives
  ×1.5–1.7 there with the reference bits (§5).
- A parallel row serves 1 to 8 rows. Past one row a float-stored weight has no sequential GEMV: an `exact` Session
  runs the GEMM, which reads the weight once for all rows, while the parallel GEMV reads it once per row. At 8 rows
  the parallel GEMV is slower than the GEMM on the widest shapes (E2B 12288 × 1536, 8960 × 1536 and the output head;
  E4B 10240 × 2560, 10752 × 2560 and the output head), so those cannot take a row under the adoption rule; every other
  shape holds at 4 and 8 rows (×1.3–14).
- No row is registered. The derivation's rule asks for two records with steady groups at every row count; one full
  record per model was taken before the GPU was rationed, and in it 14 of 39 (E2B) and 9 of 36 (E4B) groups are
  unsteady (§6). The rows the records point to are listed in §4.
- With E2B's ten rows applied for the run, a decode token with linear at `reorder` takes 17.2 ms against 27.2 ms at
  `exact` in the same interleaved run (×0.63; GPU time 12.7 against 20.6 ms), with the same tokens on the prompt
  measured (§7). E4B was not measured end to end.

## 2. Setup

- RTX 3080 Ti (12 GB), Node 24 + Dawn (Vulkan), the mirror `models/karume-gemma4` (E2B and E4B, preset `bf16`, the
  one preset of the regular line). The GPU was shared with other measurement runs through one lock, so every run
  started on a device another run had just loaded (70–90 °C). Nothing here compares two separate runs.
- Per dispatch: the workbench's `gemv-sweep` entry, the `full` set (1, 4 and 8 rows), 5 interleaved rounds, the quick
  look at its default margin, 10 s idle between cases. A kernel's time is the minimum of its rounds. Every timed
  parallel candidate is checked against the host's f64 oracle (the band of `linear`), every sequential variant against
  the baseline's words: 0 rows outside the band, 0 differing, in every record.
- The GEMM of the rows past one is `linear:v2:f32:reg16x16r1x4w4v4:wbf16`. These records were taken while the sweep
  built it with the default profile's tile geometry; the table this adapter selects (`nvidia-ampere`) keeps that
  geometry up to 16 rows, so it is the kernel a Session on this device runs, and the records stay current under the
  sweep that now builds the reference with the device's profile.
- Records (`outputs/workbench/node-nvidia-geforce-rtx-3080-ti/`): `gemv-sweep_2026-11-18T04-12-00.json` (E2B's 13
  cases, 735 rows), `gemv-sweep_2026-11-18T04-39-54.json` (E4B's 12 cases, 678 rows),
  `gemv-sweep_2026-11-18T03-35-23.json` (three E2B cases at one row, the `quick` set with the quick look off).
- A decode token: the workbench's `gemma4-bench` entry, greedy, one prompt, 96 new tokens, 3 chats per visit (the
  first left out), the arms visited in A B B A rounds, one pipeline resident at a time. The GPU time per token comes
  from the bench's tap (the product's own passes); the per-kernel shares from a `split` arm (one timed pass per
  dispatch — read its shares, not its total).

## 3. Where a decode token goes

Regular E2B `bf16`, the `split` arm (2 warm chats of 64 tokens, record `gemma4-bench_2026-11-18T03-23-29.json`):

| Family | ms per token | Share | Dispatches per token |
| --- | --- | --- | --- |
| linear (`linear_gemv:v1:f32:c32u4:wbf16`, every one) | 19.77 | 76.5 % | 277 |
| attention core | 2.83 | 11.0 % | 105 |
| norm | 1.56 | 6.0 % | 242 |
| elementwise | 1.01 | 3.9 % | 181 |
| attention other | 0.65 | 2.5 % | 115 |

The same token unsplit: 25.6 ms of wall time, 23.1 ms of GPU time (median of 2 warm chats; 24.7 ms at best).

The linears of one decode token, from the mirror's graphs:

| Model | Linears | bf16 weight bytes read | Largest by bytes |
| --- | --- | --- | --- |
| E2B | 277 | 4.56 GB | 12288 × 1536 ×40 (1.51 GB), the output head 262144 × 1536 (0.81 GB), 1536 × 12288 ×20 (0.76 GB), 6144 × 1536 ×30 (0.57 GB) |
| E4B | 344 | 9.29 GB | 10240 × 2560 ×84 (4.40 GB), 2560 × 10240 ×42 (2.20 GB), the output head 262144 × 2560 (1.34 GB) |

## 4. Per dispatch: the parallel GEMV against the sequential GEMV and the GEMM

"Parallel row" is the lane count of the plain form with the largest gain at one row among those not slower than the
GEMM at 4 and 8 rows; "none" names the fastest candidate at one row where no lane count holds at 8 rows. `screened`
means the quick look timed the candidate beyond ×1.25 of the GEMM, so it was not timed in full. The staged form was
never the fastest at one row. The spread is the baseline's largest round ÷ its smallest; above 1.25 the derivation
reads the group as unsteady.

E2B (one record):

| Shape (n × k) | Per token | Sequential GEMV at M = 1 | Fastest sequential variant | Parallel row | at M = 1 | against the GEMM at M = 4 | at M = 8 | GEMM at M = 4 / 8 | Baseline spread at M = 1 / 4 / 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12288 × 1536 | 40 | 80.4 µs | c16u8 ×1.55 | none (parallel-l16) | ×1.76 (45.8 µs) | ×1.34 | screened | 234 / 252 µs | 1.01 / 1.13 / 1.05 |
| 6144 × 1536 | 30 | 45.2 µs | c16u8 ×1.63 | 8 lanes | ×1.83 (24.7 µs) | ×2.31 | ×1.53 | 109 / 129 µs | 1.02 / 1.24 / 1.21 |
| 1536 × 12288 | 20 | 249 µs | c16u16 ×1.75 | 32 lanes | ×5.26 (47.3 µs) | ×6.81 | ×3.00 | 706 / 656 µs | 1.04 / 1.30 / 1.17 |
| 1536 × 6144 | 15 | 107 µs | c16u8 ×1.52 | 32 lanes | ×4.45 (24.0 µs) | ×6.31 | ×3.27 | 320 / 316 µs | 1.90 / 1.32 / 1.14 |
| 2048 × 1536 | 28 | 28.3 µs | c32u16 ×1.62 | 32 lanes | ×4.55 (6.2 µs) | ×4.64 | ×2.68 | 76.9 / 78.4 µs | 1.63 / 1.44 / 1.17 |
| 1536 × 2048 | 28 | 36.0 µs | c16u16 ×1.74 | 32 lanes | ×5.60 (6.4 µs) | ×6.18 | ×3.37 | 96.9 / 96.2 µs | 1.02 / 1.38 / 1.11 |
| 4096 × 1536 | 7 | 34.8 µs | c16u8 ×1.47 | 16 lanes | ×2.07 (16.8 µs) | ×2.74 | ×1.47 | 98.9 / 102 µs | 1.80 / 1.17 / 1.38 |
| 1536 × 4096 | 7 | 77.1 µs | c32u8 ×1.59 | 32 lanes | ×4.56 (16.9 µs) | ×6.47 | ×3.36 | 218 / 213 µs | 1.75 / 1.33 / 1.12 |
| 256 × 1536 | 59 | 14.8 µs | c16u16 ×1.47 | 32 lanes | ×3.94 (3.8 µs) | ×11.44 | ×8.71 | 52.9 / 52.9 µs | 2.46 / 1.47 / 1.05 |
| 512 × 1536 | 6 | 14.9 µs | c32u16 ×1.43 | 32 lanes | ×3.55 (4.2 µs) | ×8.86 | ×5.96 | 53.2 / 53.7 µs | 2.47 / 1.60 / 1.15 |
| 1536 × 256 | 35 | 4.0 µs | c16u16 ×1.27 | 16 lanes | ×1.42 (2.8 µs) | ×2.70 | ×1.77 | 11.4 / 11.4 µs | 1.13 / 1.10 / 1.22 |
| 8960 × 1536 | 1 | 60.5 µs | c16u8 ×1.65 | none (parallel-l32) | ×1.77 (34.2 µs) | ×1.28 | screened | 164 / 162 µs | 1.01 / 1.19 / 1.20 |
| 262144 × 1536 | 1 | 1531 µs | c16u4 ×0.93 | none (parallel-l16) | ×1.68 (914 µs) | ×1.20 | screened | 4367 / 4027 µs | 1.15 / 1.08 / 1.16 |

Summed over a decode token (minimum per dispatch × dispatches): the sequential GEMV 16.44 ms; the fastest sequential
variant per shape 10.72 ms; the ten rows above 7.79 ms; a parallel row on every shape 5.74 ms. The second record of the
three largest shapes at one row agrees within 3 % (12288 × 1536 ×1.71, 6144 × 1536 ×1.90, 1536 × 12288 ×5.15).

E4B (one record):

| Shape (n × k) | Per token | Sequential GEMV at M = 1 | Fastest sequential variant | Parallel row | at M = 1 | against the GEMM at M = 4 | at M = 8 | GEMM at M = 4 / 8 | Baseline spread at M = 1 / 4 / 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10240 × 2560 | 84 | 121 µs | c16u8 ×1.69 | none (parallel-l32) | ×1.94 (62.4 µs) | ×1.30 | screened | 312 / 338 µs | 1.01 / 1.13 / 1.05 |
| 2560 × 10240 | 42 | 222 µs | c16u8 ×1.66 | 32 lanes | ×3.58 (62.2 µs) | ×3.18 | ×1.50 | 583 / 555 µs | 1.08 / 1.05 / 1.11 |
| 2048 × 2560 | 35 | 55.1 µs | c16u16 ×1.69 | 32 lanes | ×3.82 (14.4 µs) | ×4.51 | ×2.68 | 138 / 151 µs | 1.05 / 1.14 / 1.25 |
| 2560 × 2048 | 35 | 39.1 µs | c32u8 ×1.44 | 16 lanes | ×2.73 (14.3 µs) | ×3.86 | ×2.56 | 107 / 114 µs | 1.69 / 1.37 / 1.04 |
| 4096 × 2560 | 7 | 59.2 µs | c16u8 ×1.54 | 16 lanes | ×2.24 (26.5 µs) | ×2.67 | ×1.47 | 161 / 170 µs | 1.90 / 1.17 / 1.08 |
| 2560 × 4096 | 7 | 84.0 µs | c16u16 ×1.69 | 32 lanes | ×3.20 (26.2 µs) | ×2.88 | ×1.50 | 214 / 215 µs | 1.07 / 1.29 / 1.15 |
| 512 × 2560 | 40 | 24.4 µs | c16u16 ×1.48 | 32 lanes | ×4.55 (5.4 µs) | ×10.64 | ×6.36 | 90.5 / 87.5 µs | 2.50 / 1.40 / 1.16 |
| 1024 × 2560 | 8 | 29.1 µs | c16u16 ×1.71 | 32 lanes | ×4.83 (6.0 µs) | ×6.26 | ×3.68 | 92.3 / 93.7 µs | 1.09 / 1.07 / 1.23 |
| 256 × 2560 | 42 | 23.8 µs | c32u16 ×1.46 | 32 lanes | ×5.04 (4.7 µs) | ×14.15 | ×9.98 | 87.7 / 86.5 µs | 2.51 / 1.00 / 1.04 |
| 2560 × 256 | 42 | 4.1 µs | c16u16 ×1.17 | 16 lanes | ×1.31 (3.1 µs) | ×2.22 | ×1.34 | 12.1 / 11.9 µs | 1.22 / 1.26 / 1.10 |
| 10752 × 2560 | 1 | 121 µs | c16u8 ×1.69 | none (parallel-l32) | ×1.86 (65.1 µs) | ×1.33 | screened | 333 / 327 µs | 1.00 / 1.23 / 1.17 |
| 262144 × 2560 | 1 | 2843 µs | c16u4 ×0.61 | none (parallel-l16) | ×1.88 (1515 µs) | ×1.06 | screened | 6410 / 6321 µs | 2.17 / 1.07 / 1.11 |

Summed over a decode token: the sequential GEMV 29.12 ms; the fastest sequential variant per shape 18.86 ms; the nine
rows above 17.68 ms; a parallel row on every shape 11.40 ms. E4B's token is dominated by the one shape that cannot
take a row: 10240 × 2560 is 84 dispatches, 10.1 of the 29.1 ms.

How far the outputs stand from the f64 oracle, as the largest share of the band of `linear` over every checked row: the
parallel candidates 0.23 (E2B, 129 rows) and 0.26 (E4B, 115 rows); the baselines — the reference order — 1.13 and
0.96 (§8).

## 5. Why the wide shapes lose at 8 rows, and what the sequential kernel leaves

- The parallel kernel takes its rows along the grid's y axis: every row is its own workgroups and reads the whole
  weight again. Its time grows with the row count (12288 × 1536: 46 µs at one row, 162 µs at 4). The GEMM reads the
  weight once into shared tiles for up to 16 rows, so its time barely moves between 4 and 8 rows (234 and 252 µs).
  Where one row is already at the memory bandwidth, 8 rows of the parallel GEMV cost eight weight reads and lose.
  A parallel GEMV that shares one weight read across the rows of a block (perf-ledger K-22, the addendum of
  2026-09-25) would remove this; it is not built here.
- At one row the sequential kernel is not at the bandwidth: reading 8 or 16 words ahead instead of 4
  (`c16u8`, `c16u16`) gives ×1.3–1.75 on every shape but the output heads, with the reference bits (0 differing
  words). A bf16 word holds 8 elements where an int4 word holds 32, so the default read-ahead of 4 words keeps a
  quarter of the bytes in flight. This is a per-shape exception of the sequential variant (`sequential.shapes`), a
  choice of the identical class that changes no number of any Session; the derivation adopts it from these records for
  the three shapes two records hold (6144 × 1536, 12288 × 1536, 1536 × 12288: `c16u8`, ×1.55–1.70). It is not
  registered here.

## 6. Why no row is registered

The derivation (`deriveLinearGemvChoices`) adopts a parallel row only from groups of two records that are steady
(baseline spread at most 1.25) at 1, 4 and 8 rows. On these records it adopts none:

- One full record per model exists. The second was queued when the GPU was rationed to one five-minute job per run;
  a `full` record of 13 cases takes about seven minutes.
- In the one record, 14 of E2B's 39 groups and 9 of E4B's 36 are unsteady, most of them at one row on the shapes with
  few outputs. The rounds show why: the sequential kernel and the GEMM run in two modes on this device (2048 × 1536,
  `c32u4`: 28.3, 29.9, 41.8, 46.0, 28.4 µs over the five rounds), while the parallel candidates of the same rounds do
  not move (256 × 1536, `parallel-l16`: 4.4 µs in every round). A candidate's gain is read against the baseline's
  fastest round, so the gains above are the conservative ones.
- Two more records would give most groups a steady pair; the generator's `--gemv-plan` writes the batch that
  measures what the records lack.

The rows of §4 (ten for E2B, nine for E4B) are in `outputs/diag/decode-optimization-2026-11-18/bf16-gemv/candidate-rows.mjs`,
which applies them to the default profile's table and reverts them.

## 7. A decode token with linear at `reorder`

Regular E2B `bf16`, the ten rows of §4 applied to the default profile's table for the run (the checkout is recorded as
dirty), one process, three arms in A B C C B A rounds, 4 rounds, 8 warm chats of 96 tokens per arm (record
`gemma4-bench_2026-11-18T06-23-35.json`):

| Arm | Decode, ms per token: median (min – max) | GPU, ms per token: median (min – max) | Against `exact` |
| --- | --- | --- | --- |
| `exact` (the preset as it ships) | 27.22 (22.79 – 32.78) | 20.64 (20.48 – 22.60) | |
| linear at `reorder` | 17.15 (14.96 – 19.55) | 12.73 (12.68 – 12.83) | ×0.630 wall, ×0.617 GPU |
| linear at `reorder`, split per dispatch | 20.93 (20.43 – 26.86) | 16.52 (16.47 – 16.63) | (read its shares) |

- The generated token ids are the same in every chat of the three arms (one sha256). The rows change the summation
  order, so that is an observation about this prompt, not a guarantee.
- The split arm's linears: 9.83 ms per token in place of 19.77 (§3). 235 of the 277 run the parallel GEMV — 163 at 32
  lanes (3.14 ms), 30 at 8 (0.90 ms), 42 at 16 (0.41 ms) — and the 42 without a row (the forty 12288 × 1536, 8960 × 1536
  and the output head) stay on the sequential GEMV and take 5.38 ms, a third of the token's GPU time.
- The gain is larger than the per-dispatch minima predict (16.4 → 7.8 ms of GEMV time, §4): inside a token the
  sequential kernels run slower than their fastest round of the sweep (19.8 ms in the split of §3 against 16.4 ms).
- Not measured: E4B end to end (its nine rows would cover 258 of 344 linears, 11.4 of 29.1 ms of sequential GEMV by
  §4's minima); Chrome on the RTX 5070 Ti; Metal; the second prompt of the bench; attention at `reorder` on top.

## 8. Noticed beside the topic

- The reference order is at the edge of the band of `linear` on long reductions. On the sweep's inputs (activations in
  [−1, 1), weights of magnitude in [2⁻⁴, 1)) the GEMM's output for 1536 × 12288 stands 3.8e-4 from the f64 oracle:
  1.13 of the band (absolute term 2e-4, derived at K = 72). The sequential GEMV of the same shape uses 0.82, E4B's
  2560 × 10240 0.85–0.96. The parallel forms sum in lanes and a tree and use at most 0.26. Nothing fails today: the
  sweep's verdict reads the candidates' band, and the GEMM appears as a baseline only for float-stored cases.
- The sequential kernels' two timing modes (§6) make a group unsteady however large the candidate's gain is. Whether
  the int storages' baselines show them too was not looked at here.

## 9. Commands

```sh
# Per dispatch (one record of E2B's cases; about 7 minutes)
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemv-sweep --candidateSet=full \
  --cases=bf16-n8960-k1536,bf16-n2048-k1536,bf16-n256-k1536,bf16-n1536-k2048,bf16-n6144-k1536,bf16-n1536-k6144,bf16-n1536-k256,bf16-n4096-k1536,bf16-n512-k1536,bf16-n1536-k4096,bf16-n12288-k1536,bf16-n1536-k12288,bf16-n262144-k1536

# What the records lack for the derivation, and the batch that measures it
node tools/geometry-profiles/generate.ts nvidia-ampere --gemv-plan --gemv-records=<record 1>,<record 2>

# The decode A/B (with rows of the float shapes in the profile)
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemma4-bench \
  --arms=gemma4/e2b,gemma4/e2b:linear-reorder,gemma4/e2b:linear-reorder:split --rounds=4 --chats=3 --newTokens=96 --prompts=few-ids
```
