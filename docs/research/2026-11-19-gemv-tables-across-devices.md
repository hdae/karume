> A point-in-time measurement record (2026-11-19): Gemma 4 decode tokens and per-dispatch GEMV readings on four
> devices — the RTX 3080 Ti, the RTX 5070 Ti, an Apple M2 and an Apple M5 — for the preset `bf16-fast`, the device
> tables' GEMV choices against the default profile's, the default's sequential exceptions, and the M2's table read on
> the M5's records. Every number below was computed from the untracked workbench records named beside it; nothing here
> is a design source of truth or a decision, and it is not edited to follow later events.

# GEMV choices across four devices: decode tokens and per-dispatch readings (2026-11-19)

The perf-ledger rows K-110 (the parallel GEMV on bf16 weights and the preset `bf16-fast`), K-112 (the per-shape
variants of the sequential GEMV) and K-113 (the derivation's evidence rules) quote measurements of 2026-11-19 that
were held only in local records. This file holds them. The records are local and untracked: they live under
`outputs/workbench/<environment>/`, and the scripts and their outputs under `outputs/diag/bf16-fast-2026-11-19/` and
`outputs/diag/m5-batch09-2026-11-19/`.

## 1. Devices, entries and method

| Device | Host, backend | Table the adapter selects | Records |
| --- | --- | --- | --- |
| RTX 3080 Ti (12 GB) | Node 24 + Dawn, Vulkan | `nvidia-ampere` | `outputs/workbench/node-nvidia-geforce-rtx-3080-ti/` |
| RTX 5070 Ti | Chrome 157 on Windows, D3D12 | `nvidia-blackwell` | `outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/` |
| Apple M2 | Chrome 157 on macOS, Metal | `apple-metal-3` (matched by `description: "Apple M2"`) | `outputs/workbench/browser-apple-m2/` |
| Apple M5 | Chrome 157 on macOS, Metal | none: the adapter reports `apple` / `metal-3` with an empty `device` and `description`, so no table matches and it runs the default profile | `outputs/workbench/browser-apple-metal-3/` |

Two workbench entries made the records ([spec](../spec/workbench.md)): `gemma4-bench` (decode tokens) and
`gemv-sweep` (per dispatch). A record's `entry` field holds its parameters and `checkout.commit` /
`checkout.dirty` the commit it ran and whether the checkout had uncommitted changes.

**Decode (`gemma4-bench`).** Every decode record used here has the same parameters: greedy, the prompt `few-ids`
(21 prompt tokens), 96 new tokens (95 decode tokens), 3 chats per visit, 4 rounds with the arms interleaved, cached
weights, subgroups on. The first chat of a visit is cold and left out, so each arm has 8 warm chats.

- The record's summary line prints, per arm, `decodeMsPerToken` = a chat's decode seconds ÷ its decode tokens, as
  the **median over the 8 warm chats** (`data.summary[].spreads.decodeMsPerToken.median`; recomputed here from
  `data.chats` with the same result to the printed digits). That is the statistic reported as "decode ms/token"
  below.
- Beside it: the median over the 8 warm chats of each chat's median interval between two tokens
  (`tokenIntervalMsMedian`), and of the GPU busy time per token (`tap.perToken.gpuBusyMs`).
- "Rounds" counts the rounds (of 4) in which the arm named was faster, by the mean `decodeMsPerToken` of its warm
  chats in that round.
- "Tokens: same" means every chat of every arm of the record, cold chats included, has one `tokenIdsSha256`.
- This is not the protocol of [the 2026-11-18 table decode record](2026-11-18-blackwell-gemv-table-decode.md)
  (median over visits of each visit's mean); figures from the two are not interchangeable.
- A bench in which an arm carries a profile switch (`default-gemv`, `general-sequential`) acquires a device for
  every visit; one without a switch runs every visit on one device. The spec says the two kinds' figures are not
  comparable ([profile switches](../spec/workbench.md#the-decode-benchs-profile-switches)).

**Per dispatch (`gemv-sweep`).** A kernel's time is the minimum of its rounds ÷ its dispatches per pass
(`perDispatch`). A kernel the quick look screened out has only its quick time (`quickPerDispatch`, one probe pass);
a kernel skipped at 4 or 8 rows (not adoptable from its M = 1 row) has no time. Packed-activation rows are not
measured by the sweep at all (`settings.notMeasured`). The protocol is in
[the spec](../spec/workbench.md#how-the-gemv-sweep-times-a-row-count).

The checkouts of the decode records:

| Device | Records | Commit | Checkout |
| --- | --- | --- | --- |
| RTX 3080 Ti | `gemma4-bench_2026-11-19T12-51-56` … `T12-58-41` (4) | `fb5c0e52` | uncommitted changes (the preset `bf16-fast` was committed later, at `11f05099`; the record does not say what the changes were) |
| RTX 5070 Ti | `gemma4-bench_2026-11-19T17-37-50` … `T17-43-26` (4; batch `08-bf16-fast-decode`, steps 1–4) | `1c6b4d91` | clean |
| Apple M2 | `gemma4-bench_2026-11-19T06-54-59`, `T07-47-33` (batch `03-m2-checks-gemv-and-decode`, steps 12 and 15) | `453921bc` | clean |
| Apple M2 | `gemma4-bench_2026-11-19T14-57-51` … `T15-12-31` (4; batch `08-bf16-fast-decode`, steps 1–4) | `1c6b4d91` | clean |
| Apple M5 | `gemma4-bench_2026-11-19T19-33-39`, `T19-36-27`, `T19-37-53`, `T21-43-51` (batch `09-m5-checks-decode-and-gemv-sweep`, steps 2, 3, 4 and 13) | `1c6b4d91` | clean |

`1c6b4d91` contains `8713ad4c`, the commit that derived the three device tables' GEMV choices from records against
the current default; `fb5c0e52` and `453921bc` precede it. At `453921bc` `apple-metal-3` carried no GEMV choices of
its own (the record's `gemvChoices.tableCarriesGemv` is `false`), so the M2 ran the default profile's.

## 2. Decode: regular Gemma 4 `bf16` against `bf16-fast` (E2B and E4B)

Records: the eight `gemma4-bench` records of 2026-11-19 whose arms are `gemma4/e2b@bf16,gemma4/e2b@bf16-fast` or
`gemma4/e4b@bf16,gemma4/e4b@bf16-fast`, under each device's directory. No arm carries a switch: each device runs its
own table (the M5 the default profile).

| Device | Model | Record | Decode ms/token, `bf16` / `bf16-fast` | Ratio | `bf16-fast` faster | Token interval, ms | Ratio | GPU ms/token | Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 3080 Ti | E2B | `T12-51-56` | 16.70 / 12.97 | ×0.777 | 4 of 4 | 16.46 / 10.88 | ×0.661 | 15.49 / 10.10 | same |
| RTX 3080 Ti | E2B | `T12-56-15` | 16.47 / 12.07 | ×0.733 | 4 of 4 | 16.14 / 10.69 | ×0.662 | 15.26 / 10.18 | same |
| RTX 3080 Ti | E4B | `T12-54-22` | 32.23 / 18.28 | ×0.567 | 4 of 4 | 26.28 / 17.83 | ×0.679 | 25.19 / 16.90 | same |
| RTX 3080 Ti | E4B | `T12-58-41` | 29.68 / 21.35 | ×0.720 | 4 of 4 | 26.15 / 17.97 | ×0.687 | 25.03 / 17.02 | same |
| RTX 5070 Ti | E2B | `T17-37-50` | 19.12 / 9.96 | ×0.521 | 4 of 4 | 19.06 / 9.72 | ×0.510 | 17.73 / 8.53 | same |
| RTX 5070 Ti | E4B | `T17-41-42` | 32.71 / 16.57 | ×0.507 | 4 of 4 | 32.66 / 16.40 | ×0.502 | 30.93 / 15.00 | same |
| Apple M2 | E2B | `T14-57-51` | 65.36 / 60.77 | ×0.930 | 4 of 4 | 65.62 / 60.79 | ×0.926 | 60.56 / 56.19 | same |
| Apple M2 | E4B | `T15-08-18` | 119.90 / 113.89 | ×0.950 | 4 of 4 | 119.73 / 113.78 | ×0.950 | 115.63 / 109.72 | same |
| Apple M5 | E2B | `T19-36-27` | 48.49 / 44.77 | ×0.923 | 4 of 4 | 48.41 / 44.48 | ×0.919 | 45.80 / 42.22 | same |
| Apple M5 | E4B | `T21-43-51` | 98.61 / 90.93 | ×0.922 | 4 of 4 | 98.52 / 90.76 | ×0.921 | 95.23 / 87.55 | same |

- On the RTX 3080 Ti the decode ms/token of a chat is bimodal while its token interval and GPU time are not. In
  `T12-54-22` the eight warm `bf16` chats read 32.0, 34.9, 26.3, 32.4, 26.5, 32.5, 26.6 and 33.1 ms/token, their
  token intervals 25.9 to 28.3 ms; the `bf16-fast` chats of `T12-58-41` alternate between about 18.2 and 24.5
  ms/token at a token interval of 17.9 to 18.1 ms. The two statistics therefore give different ratios on this device
  (E2B ×0.73 to ×0.78 by decode ms/token, ×0.66 by token interval; E4B ×0.57 to ×0.72 against ×0.68 to ×0.69). The
  records do not say where the extra time of the slow chats went. On the other three devices the two statistics agree
  within 0.03 of the ratio.
- On the RTX 3080 Ti the records ran at `fb5c0e52` with uncommitted changes, before `8713ad4c` re-derived
  `nvidia-ampere`'s GEMV choices; the table those Sessions ran is not recorded (no arm carries a switch, so the record
  has no `gemvChoices`).

## 3. Decode: a device table's GEMV choices against the default profile's (RTX 5070 Ti, Apple M2; E2B)

Records: `gemma4-bench_2026-11-19T17-42-31` and `T17-43-26` (RTX 5070 Ti), `T15-11-09` and `T15-12-31` (Apple M2),
all at `1c6b4d91`. The first arm runs the table as the adapter selects it; the second carries the switch
`default-gemv`. The choices each arm ran with, from `data.arms[].gemvChoices`:

| Device | Table's choices: sequential exceptions / parallel rows | `default-gemv`: sequential exceptions / parallel rows |
| --- | --- | --- |
| RTX 5070 Ti (`nvidia-blackwell`) | 64 / 108 | 22 / 88 |
| Apple M2 (`apple-metal-3`) | 86 / 101 | 22 / 88 |

| Device | Model, preset | Decode ms/token, table / default | Ratio (default ÷ table) | Table faster | Token interval, ms | GPU ms/token | Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | regular E2B `bf16-fast` | 10.03 / 11.57 | ×1.154 | 4 of 4 | 9.90 / 11.48 | 8.53 / 10.03 | same |
| RTX 5070 Ti | QAT E2B `i4-fast` | 5.74 / 5.76 | ×1.003 | 2 of 4 | 5.57 / 5.57 | 4.52 / 4.59 | same |
| Apple M2 | regular E2B `bf16-fast` | 60.41 / 60.47 | ×1.001 | 3 of 4 | 60.63 / 60.66 | 56.04 / 56.16 | same |
| Apple M2 | QAT E2B `i4-fast` | 24.88 / 27.34 | ×1.099 | 4 of 4 | 24.69 / 26.89 | 21.49 / 23.63 | same |

Put the other way, the table's time over the default's is ×0.867 (RTX 5070 Ti, `bf16-fast`), ×0.997 (RTX 5070 Ti,
`i4-fast`), ×0.999 (M2, `bf16-fast`) and ×0.910 (M2, `i4-fast`). No control run (two arms with the same choices) was
taken in these batches.

## 4. Decode: the default profile's bf16 sequential exceptions against the general variant (Apple M2, Apple M5; E2B `bf16`)

Records: `browser-apple-m2/gemma4-bench_2026-11-19T07-47-33` (at `453921bc`) and
`browser-apple-metal-3/gemma4-bench_2026-11-19T19-33-39` (at `1c6b4d91`). The second arm carries
`general-sequential`, so every sequential GEMV of one row runs the general variant `c32u4`.

- M2: `apple-metal-3` then carried no GEMV choices, so the first arm ran the default profile's 23 sequential
  exceptions of that commit — the 22 bf16 ones and the int8 output-head exception the default dropped later that day
  (`f0156ac9`) — and the second arm none.
- M5: the default profile of `1c6b4d91`, 22 exceptions, against none; 88 parallel rows in both arms (not reached at
  `exact`).

| Device | Decode ms/token, exceptions / general | Ratio (general ÷ exceptions) | Exceptions faster | Token interval, ms | GPU ms/token | Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Apple M2 | 68.24 / 71.05 | ×1.041 | 4 of 4 | 68.42 / 71.12 | 63.29 / 66.17 | same |
| Apple M5 | 48.54 / 53.59 | ×1.104 | 4 of 4 | 48.46 / 53.54 | 45.89 / 50.66 | same |

## 5. Decode on the Apple M5: QAT E2B `i4` against `i4-fast`

Record: `browser-apple-metal-3/gemma4-bench_2026-11-19T19-37-53` (at `1c6b4d91`, default profile, no switch).

| Arm | Decode ms/token | Token interval, ms | GPU ms/token |
| --- | --- | --- | --- |
| QAT E2B `i4` | 24.92 | 24.93 | 22.57 |
| QAT E2B `i4-fast` | 17.86 | 17.85 | 15.69 |

`i4-fast` ÷ `i4` = ×0.717 (×0.716 by token interval), `i4-fast` faster in 4 of 4 rounds, the same tokens. For
context only: the M2's record `browser-apple-m2/gemma4-bench_2026-11-19T06-54-59` (at `453921bc`, when
`apple-metal-3` carried no GEMV choices) read `i4-fast` 27.20 against `i4` 38.26 ms/token (×0.711), `i4-fast` faster
in 4 of 4 rounds, the same tokens.

## 6. Per dispatch: where a registered table's choice is slower than its alternative

Records: every `gemv-sweep` record dated 2026-11-17 or later under each device's directory — 44 of the RTX 3080 Ti
(`gemv-sweep_2026-11-17T08-06-12` to `…2026-11-19T12-42-12`, at twelve commits, the latest `fb5c0e52`, 12 records with
uncommitted changes), 86 of the RTX 5070 Ti (`…2026-11-17T18-26-52` to `…2026-11-19T12-55-39`, 72 of them at
`72f2ada9`, 1 with uncommitted changes), 14 of the Apple M2 (`…2026-11-18T23-15-21` to `…2026-11-19T12-58-40`, clean)
and 8 of the Apple M5 (`…2026-11-19T19-52-51` to `…T21-34-02`, clean, `1c6b4d91`).

**Method** (`outputs/diag/bf16-fast-2026-11-19/scripts/table-json.mts`, run from the repository root as
`node <script> <device label> <records dir> <table file name> <export name> table|default`; it writes
`outputs/diag/bf16-fast-2026-11-19/choice-vs-alternative-<label>-<which>.json`, read by `join.mjs`). Re-run tonight
into a scratch directory, every output was byte-identical to the stored one.

- The profile is the default profile with the table's `linearGemv` (`table`), or the default profile itself
  (`default`). For every case and row count of the sweep's case table that a record summarises, the script asks the
  sweep's own choice function which kernel that profile runs at `exact` and at `reorder`, by candidate name.
- Two comparisons per group: at one row, the choice at `exact` against the general variant `c32u4`, when the choice is
  not `c32u4` ("sequential"); at any row count, the choice at `reorder` against the choice at `exact`, when they
  differ ("parallel").
- Per record, the ratio of the two kernels' `perDispatch`; a group's value is the **median of its per-record
  ratios**, read as slower when above 1. A kernel without `perDispatch` in a record (screened, skipped or failed)
  leaves that record out for the group; a group in which no record timed both kernels is counted separately as never
  timed.
- A parallel group whose shape has a packed-activation row in the profile is not compared (the sweep does not measure
  that form): 21 groups in every run.
- Kernels are matched by candidate name across records of any checkout. The script does not apply the derivation's
  currency or steadiness rules (perf-ledger K-113) and does not check words.

| Table, on its device's records | Records | Parallel groups compared | Slower (median > 1) | Of them > ×1.05 | Row counts of the slower | Range of their medians | Sequential groups compared | Slower | Range | Never timed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nvidia-ampere` (RTX 3080 Ti) | 44 | 246 | 37 | 35 | 16 at 4 rows, 21 at 8 | ×1.014 to ×5.056 | 27 | 1 | ×1.001 | 0 |
| `nvidia-blackwell` (RTX 5070 Ti) | 86 | 303 | 6 | 6 | 3 at 4 rows, 3 at 8 | ×1.436 to ×2.382 | 72 | 0 | — | 0 |
| `apple-metal-3` (Apple M2) | 14 | 282 | 11 | 3 | 2 at 1 row, 3 at 4, 6 at 8 | ×1.000 to ×1.123 | 90 | 3 | ×1.024 to ×1.035 | 10 |

- `nvidia-blackwell`, the six slower parallel groups — each an output head, the choice at `reorder` a staged
  parallel kernel without a row block against the sequential kernel at `exact`, 3 records each:

  | Case | Rows | Choice at `reorder` | Choice at `exact` | Median (per-record range) |
  | --- | --- | --- | --- | --- |
  | int8 `262144 × 2560` | 8 | `staged-l16` | `c32u4r8` | ×2.382 (2.379–2.393) |
  | int8 `262144 × 2560` | 4 | `staged-l16` | `c32u4r4` | ×1.790 (1.789–1.793) |
  | int4 g32 `262144 × 2816` | 8 | `staged-l8` | `c32u4r8` | ×1.678 (1.665–1.737) |
  | int4 g32 `262144 × 3840` | 8 | `staged-l8` | `c32u4r8` | ×1.572 (1.555–1.592) |
  | int4 g32 `262144 × 2816` | 4 | `staged-l8` | `c32u4r4` | ×1.484 (1.479–1.528) |
  | int4 g32 `262144 × 3840` | 4 | `staged-l8` | `c32u4r4` | ×1.436 (1.425–1.458) |

- `nvidia-ampere`: the 37 slower parallel groups are int4 g32 (15), int4 g512 (16), int8 (4) and int2 (2), all at 4
  and 8 rows, every one a parallel kernel without a row block (`parallel-l<lanes>`, once `staged-l32`) against the
  sequential kernel the table runs at `exact` (`c32u4r1` to `c32u4r8`); the largest
  are the int4 g32 head `262144 × 3840` at 8 rows (×5.056, 6 records, 2.780–6.017) and `262144 × 2816` at 8 rows
  (×2.967). The one slower sequential group is E4B's bf16 head `262144 × 2560`, `c16u4` against `c32u4`: ×1.001 as
  the median of 8 records whose ratios run from 0.468 to 1.628.
- `apple-metal-3`: the 11 slower parallel groups are int8 `262144 × 1536` at 4 and 8 rows (×1.123, ×1.112), int4 g32
  `3840 × 4096` at 8 and 4 rows (×1.065, ×1.037), int4 g32 `262144 × 3840` at 4 rows (×1.035), int4 g32
  `12288 × 1536` at 8 rows (×1.026), bf16 `2048 × 1536` at 1 row (`parallel-l16` against `c256u8`, ×1.014, 4 records),
  int4 g32 `8960 × 1536`, int4 g512 `6144 × 1536` and int4 g32 `10752 × 2560` at 8 rows (×1.013 each), and int4 g512
  `4096 × 1536` at 1 row (×1.000); 10 of the 11 rest on 2 records. The three slower sequential groups are bf16
  `12288 × 1536` (×1.035), `6144 × 1536` (×1.029) and `10752 × 2560` (×1.024), `c16u8` against `c32u4`, 4 records
  each. The 10 never-timed groups are sequential groups of bf16 shapes at one row (`256 × 1536`, `1536 × 2048`,
  `1536 × 256`, `512 × 1536`, `1536 × 12288`, `512 × 2560`, `256 × 2560`, `2560 × 10240`, `2560 × 256`,
  `1024 × 2560`), in which no record timed both the table's choice and `c32u4`.

The same script on the Apple M5's 8 records, for reference (the M5 runs the default profile):

| Profile, on the M5's records | Parallel groups compared | Slower | Of them > ×1.05 | Range | Sequential groups compared | Slower | Range | Never timed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default profile | 174 | 43 (4 at 1 row, 14 at 4, 25 at 8) | 31 | ×1.019 to ×1.847 | 16 | 1 | ×1.003 | 6 |
| `apple-metal-3` | 153 | 9 (4 at 1 row, 1 at 4, 4 at 8) | 3 | ×1.007 to ×1.081 | 40 | 22 | ×1.002 to ×1.133 | 83 |

The M5's records were swept against the default profile, so a kernel only `apple-metal-3` chooses is often screened
or skipped there; that is the 83 never-timed groups.

## 7. The M2's table on the M5, per dispatch

Script: `outputs/diag/bf16-fast-2026-11-19/scripts/m5-table-vs-default.mjs <records dir> <choices A> <choices B>
<output>`, with the two choice dumps written by `choices-dump.mts <table file name> <export name> table|default
<output path>` (A = `apple-metal-3 APPLE_METAL_3 table`, B = the default profile). For every case, row count and
level at which A's and B's kernels differ, it takes in each M5 record the time of A's kernel over the time of B's
kernel — `perDispatch`, or `quickPerDispatch` where the kernel was screened — and reports the median over the records;
a group is marked "quick look" when any of its ratios used a quick time. Re-run tonight from the code at HEAD on the
M5's eight `gemv-sweep` records (`browser-apple-metal-3/gemv-sweep_2026-11-19T19-52-51` … `T21-34-02`: the
`full` candidate set over 86 of the case table's 115 cases — the bf16, int4 g32, int8 / int2 and int4 g512 / g2048 /
g4096 cases — in four parts, each taken twice; 5 rounds, quick-look margin ×1.25, 10 s idle),
the result equals the stored `outputs/diag/m5-batch09-2026-11-19/m5-table-over-default.json` entry for entry
(170 groups).

- Over the sweep's case table (525 groups), A and B choose different kernels in 95 groups at `exact` and 182 at
  `reorder`. The M5's records hold 67 and 125 of those groups; 67 and 103 have both kernels timed. The other 22 at
  `reorder` are at 4 and 8 rows, where one of the two kernels was a skipped row.
- 164 groups rest on 2 records, 6 on 1.

Ratio = A's time ÷ B's time (above 1: the M2's table is slower on the M5). "Faster" < 0.95, "same" 0.95–1.05,
"slower" > 1.05.

| Level | Weight storage or row count | Groups | Faster | Same | Slower | > ×1.25 | Min | Median | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `exact` (1 row only) | all | 67 | 2 | 51 | 14 | 3 | 0.86 | 1.01 | 1.78 |
| `exact` | bf16 | 17 | 0 | 9 | 8 | 3 | 0.99 | 1.04 | 1.78 |
| `exact` | int4 g32 | 27 | 2 | 24 | 1 | 0 | 0.86 | 1.01 | 1.08 |
| `exact` | int4 g512 | 10 | 0 | 10 | 0 | 0 | 1.00 | 1.01 | 1.03 |
| `exact` | int8 | 9 | 0 | 4 | 5 | 0 | 1.00 | 1.06 | 1.13 |
| `exact` | timed in full only | 36 | 0 | 30 | 6 | 0 | 0.99 | 1.01 | 1.19 |
| `reorder` | all | 103 | 92 | 9 | 2 | 0 | 0.14 | 0.71 | 1.09 |
| `reorder` | bf16 | 11 | 6 | 4 | 1 | 0 | 0.41 | 0.91 | 1.09 |
| `reorder` | int4 g32 | 44 | 40 | 3 | 1 | 0 | 0.14 | 0.71 | 1.07 |
| `reorder` | int4 g512 | 24 | 24 | 0 | 0 | 0 | 0.46 | 0.69 | 0.88 |
| `reorder` | int8 | 22 | 20 | 2 | 0 | 0 | 0.21 | 0.60 | 0.96 |
| `reorder` | int2 | 2 | 2 | 0 | 0 | 0 | 0.69 | 0.69 | 0.69 |
| `reorder` | 1 row | 35 | 24 | 9 | 2 | 0 | 0.14 | 0.88 | 1.09 |
| `reorder` | 4 rows | 34 | 34 | 0 | 0 | 0 | 0.38 | 0.66 | 0.89 |
| `reorder` | 8 rows | 34 | 34 | 0 | 0 | 0 | 0.41 | 0.62 | 0.84 |

(The script's quartiles are its own rank rule: element ⌊(n − 1) · p⌋ of the sorted medians.)

**Slower by more than ×1.25**: three groups, all at `exact`, one row, all bf16, each resting on a quick-look time:

| Case | M2 table's choice | Default's choice | Median | Records |
| --- | --- | --- | --- | --- |
| bf16 `256 × 2560` | `c128u8` | `c16u16` | ×1.78 | 2, quick look |
| bf16 `512 × 1536` | `c128u8` | `c16u16` | ×1.55 | 2, quick look |
| bf16 `1536 × 256` | `c64u2` | `c32u16` | ×1.44 | 2, quick look |

**Which numbers rest on quick-look times**: 31 of the 67 groups at `exact` and none at `reorder`. In every one of the
31 the screened kernel is the M2 table's choice; the default's choice is the M5 sweep's baseline and always has a full
time. So a quick-look group compares one probe pass of the M2's choice with the minimum of 5 rounds of the default's.
Among them are the three above and the two groups the M2's table reads faster at `exact` (int4 g32 `1536 × 12288`
×0.855 and `12288 × 1536` ×0.944). The 36 `exact` groups timed in full read 0.99 to 1.19, six above 1.05: bf16
`2560 × 256` (×1.185), `512 × 2560` (×1.182), `4096 × 1536` (×1.117), `4096 × 2560` (×1.106) and `8960 × 1536`
(×1.063), and int8 `1536 × 256` (×1.133). The two slower groups at `reorder` are timed in full: bf16 `512 × 1536` at one
row (`parallel-l8` against `parallel-l16`, ×1.094) and int4 g32 `2560 × 10240` at one row (`parallel-l8` against
`staged-l32`, ×1.072).

**Words on the M5** (same records): of the 2,749 sequential-variant rows the eight records time in full at one row
besides the baseline, 1,165 write other words than the baseline (`identicalToBaseline: false`; the summary lines
count 169 to 212 per integer record and 0 in both bf16 records). Every one is over an integer weight (int4 of groups
32, 512, 2048 and 4096, int8, int2) and has an unroll other than 4; 238 of them are 16-column variants. In the M2's
14 records since 2026-11-17, 423 of 2,768 such rows differ, the same kinds of weight and unroll, and none is a
16-column variant (the M2's case is in [known issues](../known-issues.md)). The 74 rows of an
`apple-metal-3` choice at `exact` that the M5's records timed in full (it differs from the default's choice there)
all wrote the baseline's words; the choices behind the 31 quick-look groups were screened, so their words were not
compared.

## 8. The kernel checks on the Apple M5 at `1c6b4d91`

Record: `browser-apple-metal-3/kernel-check_2026-11-19T19-29-08` (batch `09-m5-checks-decode-and-gemv-sweep`, clean
checkout, all 13 checks). Verdict `fail`: 10 passed, 2 failed, 1 incomplete.

| Check | Verdict | Failing parts on the M5 (failing cases of cases) | The Metal table of known-issues |
| --- | --- | --- | --- |
| `gemv-subgroup32`, `rms-subgroup32`, `rms-norm-prefetch`, `gemv-integer`, `gemv-planes`, `gemv-parallel-rows`, `score-f16-store`, `state-bidirectional`, `expert-linear`, `fusion-twins` | pass | — | — |
| `state-attention-rewrites` | fail | `state-qk-fold` 252 of 252; `state-pv-unroll` 532 of 546; `state-stats-pv-unroll` 476 of 490 | 252 of 252; 532 of 546 and 476 of 490 |
| `state-kv16` | fail | `state-kv16-readers` 31 of 246 (`state-kv16-append` 0 of 7) | 31 of 246 |
| `attention-online-tiled` | incomplete | `attention-online-tiled-wg` ran 0 cases: "the wg form at D = 128 declares 35328 B of workgroup storage, the device grants 32768 B"; `-producer` (18) and `-sg32` (28) pass | — |

- The failing counts equal the Metal table of [known issues](../known-issues.md) part for part.
- The first failures the summary prints are one unit in the last place: `state-qk-fold` word 0 expected `0x40a1f5cd`,
  actual `0x40a1f5cc`; `state-kv16-readers` expected `0xbf7014b5`, actual `0xbf7014b4`.
- The states-attention check's 5 mutants were all detected (each wrote differing words).
- The device's `maxComputeWorkgroupStorageSize` is 32768 in the M5's records.

## 9. Not measured

- E2B's speculative decode with the drafter under `bf16-fast` against `bf16`: the `gemma4-bench` entry has no drafter
  arm.
- `nvidia-ampere`'s table end to end against the default profile's choices: no record of 2026-11-19 at or after
  `8713ad4c` carries `default-gemv` on the RTX 3080 Ti. The three RTX 3080 Ti records at `79ab0f5c` with uncommitted
  changes (`gemma4-bench_2026-11-19T09-08-47`, `T09-11-52`, `T09-13-53`, the arms `linear-reorder` and
  `default-gemv`) precede the default's bf16 parallel rows (`10b322c0`) and are not read here.
- The RTX 3080 Ti's `bf16` / `bf16-fast` decode at a clean checkout, or with `nvidia-ampere` as re-derived at
  `8713ad4c`.
- The M2's table end to end on the M5: the M5 has only per-dispatch readings of it (§7), and those were swept against
  the default profile, so many of its choices are screened or skipped there. No table was derived from the M5's
  records.
- A table against the default (`default-gemv`) on E4B, 12B or 26B, or with QAT `i4` at `exact`, on any device.
- `bf16-fast` decode on any prompt but `few-ids`, with a sampler other than greedy, or past 96 new tokens.
- A control run (two arms with the same choices) in any of these batches, so the noise of 4 rounds on these devices
  that day is not measured; the differences of ×1.001 to ×1.003 in §3 are not resolved by these records.
- Words of the sequential variants that the M5's quick look screened, among them the M2 table's choices in 31 groups.
- Packed-activation rows: the sweep does not measure them, so §6 and §7 say nothing about them.
- Where the RTX 3080 Ti's slow chats spent their extra decode time (§2).
