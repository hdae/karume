> A point-in-time analysis record (2026-11-28): the error between GEMV sweep records of one device, read from the
> recorded sweeps of 2026-11-18 to 27, and the adjudication of a rule that adopts a GEMV choice from one steady record
> — the evidence of the rule the maintainer ruled on that day and of its numbers
> ([spec/geometry-profiles.md](../spec/geometry-profiles.md#deriving-the-gemv-choices)). Every number is copied from
> the survey's and the adjudication's outputs (local scripts and their text outputs under
> `outputs/misc/gemv-rule-2026-11-28/` and its `check/` directory, outside git). Nothing here is a design source of
> truth or a decision, and it is not edited to follow later events.

# The GEMV adoption rule of one steady record

## The question

The GEMV derivation adopted a candidate only when it was ×1.10 faster than the default's choice in every group of two
records or more, against the reference's fastest time repeated by a second record. An outside collaborator runs the
benchmark page once, so one record had to be enough where the device is steady, and the margin had to be the measured
error between records rather than a number set by hand.

## The data

- `karume-gemv-sweep/4` records that were not cut short and carry GPU timestamps in ns: RTX 5070 Ti 85 (Chrome,
  D3D12), Apple M2 14, Apple M5 2 plus 8 under `browser-apple-metal-3` (the same M5 without its adapter description,
  2026-11-19), RTX 3080 Ti 76 (Node, Vulkan). The "four steady devices" of the survey are three physical ones.
- When: the RTX 5070 Ti's on 2026-11-18 (82), 11-19 (2) and 11-27 (one `explore` run); the M5's on 11-19 (8) and
  11-23 (2); the M2's between 11-18 19:00 and 11-19 12:00. Most of the "error between runs" is error within one day.
- Two records' "same row": the same case fingerprint, case, row count, candidate, kernel fingerprint and baseline
  kernel fingerprint. The repeat error of a row is ln(gain in A ÷ gain in B).

## The error between records

| Device | Pairs | p50 | p95 | p99 | Largest |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 10,524 | ×1.004 | ×1.028 | ×1.081 | ×1.308 |
| Apple M5 | 4,263 | ×1.005 | ×1.040 | ×1.067 | ×1.102 |
| Apple M2 | 7,377 | ×1.008 | ×1.055 | ×1.099 | ×1.243 |
| Apple M5 (no description) | 3,127 | ×1.006 | ×1.051 | ×1.095 | ×1.151 |
| RTX 3080 Ti | 213,926 | ×1.022 | ×1.426 | ×1.918 | ×9.518 |

- The tails are heavy everywhere: 3.4 to 10.9 % of the pairs lie beyond five robust sigmas, so no "k sigma" margin
  rests on a normal shape. The error is multiplicative (on the RTX 5070 Ti the p50 stays ×1.003 to ×1.005 over three
  decades of dispatch time).
- The error is shared by a group: the group's median carries 72 % (RTX 5070 Ti), 52 %, 76 % and 52 % (Apple), 95 %
  (RTX 3080 Ti) of the robust variance, and it is mostly the reference's own movement. On the RTX 3080 Ti the reference
  kernel has two timing modes, which no quantity inside a record tells apart.
- Of the quantities inside a record the sum of the two rows' gaps (second-smallest round ÷ smallest) predicts the
  error best (Spearman 0.20 to 0.55). With it, ln bound = a + 1 · ln(gap × reference gap) covers 99 % of the steady
  devices' rows two-sided at a = 0.065 when fitted on all of them; at the form the rule reads (M = 1, the candidates a
  table can hold, the default's choice as reference, three physical devices) a = 0.0578, 0.0635 and 0.0562
  (n = 3,633, 4,414, 8,338), and without the gap term 0.0608, 0.0794 and 0.0774.

## The class of a record

- The share of a record's groups whose baseline spread is above ×1.05 separates the devices: of the records with 20
  groups or more, the 35 of the three steady devices read 0 to 15.2 % and the 36 of the RTX 3080 Ti 31.8 to 84.6 %
  (its `explore` runs 0.443 to 0.528). No record lies between: 20 % is a point inside an empty band, an interpolation.
- Below 20 groups the test fails both ways: six RTX 3080 Ti records of 12 or 13 groups read 0 to 16.7 % (the rule of
  one record then adopted 2 of 74 candidates that other records read slower, and missed its bound in 22 of 270 rows,
  20 of them overstated); four records of steady devices with 2 to 4 groups read 25 to 50 %. Over windows of
  consecutive groups no call is wrong from 36 groups on (12: M5 2.25 %, RTX 3080 Ti 1.47 %; 24: 2.95 %, 0.17 %).
- The share depends on the row count: on the Apple devices M = 1 (the first group after a case's warm start) reads
  9.4 to 10.0 %, M = 4 and 8 0.7 to 3.9 %; on the RTX 3080 Ti M = 1 reads 39.7 %, M = 4 and 8 61 to 63 %.

## Held out: what the fit does not cover

The margin's coverage, scored on records the fit did not use (`heldout_bound.ts`, M = 1, table candidates):

| Check | a fitted | Rows beyond the bound | Groups |
| --- | --- | --- | --- |
| In the fit, a = 0.065 (RTX 5070 Ti / M2 / M5) | – | 0.83 % / 0.86 % / 0.60 % | 17/439 · 18/288 · 19/402 |
| RTX 5070 Ti left out | 0.0635 | 0.85 % | 18/439 |
| Apple M2 left out | 0.0578 | 1.43 % (95 % upper 1.76 %) | 25/288 |
| Apple M5 left out | 0.0635 | 0.68 % | 19/402 |
| RTX 5070 Ti: 11-19 against 11-18 (a = 0.065) | within a day: 0.0185 | 3.81 % (upper 5.80 %), all overstated | 4/50 |
| RTX 5070 Ti: 11-27 `explore` against earlier | 0.0599 | 1.75 % (upper 3.03 %) | 9/115 |
| Apple M5: 11-23 against 11-19 | 0.0648 | 0.67 % (upper 0.94 %) | 10/172 |
| Apple M2: records 6–13 against 0–5 | 0.0438 | 2.54 % (upper 3.16 %) | 15/144 |

- So **×1.065 bounds the error between records within a session, and it is exceeded across days in 0.7 to 3.8 % of
  the rows**. On the RTX 5070 Ti the 99 % point within a day is a = 0.0185 and across days 0.1443.
- The misses are not random rows: single kernels move between sessions (with the shared movement of the case and row
  count taken out, their 99 % points are ×1.067, ×1.062 and ×1.054; beyond ×1.065 in 19 of 2,031, 19 of 2,251 and
  19 of 3,312 cases, beyond ×1.10 in 4, 4 and 3). On 2026-11-27 the RTX 5070 Ti's bf16 references `c16u16` and
  `c32u16` alone were 9 % slower in six cases, their gaps 1.0001 to 1.003: nothing inside the record shows it.
- At ×1.03 the bound is no bound: in the fit it is missed in 2.6, 6.1 and 3.6 % of the rows. ×1.03 is a margin chosen
  on purpose, not an error bound.

## The rule replayed

Harmful = an adopted candidate that the other records read slower than ×1/1.02. Units: device × case × class ×
reference kernel at M = 1 (`replay_m1.ts`).

| Situation | Rule | Adopted | Wrongly | Harmful | Clear winners refused | Time behind the best |
| --- | --- | --- | --- | --- | --- | --- |
| Reference = the default's choice, truth = all other records | one record, ×1.065 × gaps | 1,481 | 0 | 0 (upper 0.20 %) | 15 of 1,369 | 0.94 % |
| same | one record, ×1.03 × gaps | 1,679 | 0 | 0 (upper 0.18 %) | 3 of 1,369 | 0.46 % |
| same | one record, ×1.10 | 1,367 | 0 | 0 | 22 of 1,369 | 1.34 % |
| same | one record, ×1.03 | 1,726 | 3 | 1 | 2 of 1,369 | 0.39 % |
| same | two records, ×1.10 (the earlier rule) | 984 | 0 | 0 | 135 of 1,108 | 2.33 % |
| Truth from another day only | one record, ×1.065 / ×1.03 × gaps | 1,051 / 1,217 | 0 / 0 | 0 / 0 | – | – |
| Table already holds a winner (from held-out records), truth the rest | one record, ×1.065 × gaps | 19 of 3,160 decisions | 0 | 0 | – | 0.20 % |
| same | one record, ×1.03 × gaps | 35 | 4 | 0 | – | 0.18 % |
| same | one record, ×1.03 | 64 | 17 | 5 | – | 0.18 % |
| same, truth from another day only | one record, ×1.065 × gaps | 9 of 2,127 decisions | 2 | 2 | – | – |
| same | one record, ×1.03 × gaps | 21 | 5 | 3 | – | – |

- The two harmful adoptions at ×1.065 are one shape of the RTX 5070 Ti (bf16 `10240 × 2560`: ×1.14 to ×1.15 in the
  records of 11-18, ×0.958 and ×0.973 on 11-19 and 11-27, gaps 1.0002 to 1.006). The rule of two records makes the
  same adoption from two records of one day.
- "Below 1 % harmful" is shown for a first derivation only; with a table that already holds good choices the
  adoptions are too few (9 to 35) to state a rate per adoption.
- At M = 4 and 8 the bar of "not slower than the default's choice in the same group" (gain ≥ 1, no margin) passes 3 of
  1,299 rows that other records read slower than ×1/1.02 and refuses 3 of 10,684 they read ×1.02 faster or more;
  1 ÷ (margin × gaps) passes 37 to 51 % of the slower rows, margin × gaps refuses 5.5 to 7.6 % of the faster ones.
- The whole derivation replayed on pairs of complete records (a copy of the derivation with the rule as a switch,
  equal to it in the old mode): the parallel rows change in 40 of 864 decisions (4.6 %: harmful 6, helpful 17,
  neutral 15, unscored 2) and the sequential exceptions in 92 of 800 (11.5 %: 3, 52, 12, 25); no adopted row is slower
  than the default at any row count (0 of 359 and 0 of 268). At ×1.03 the adoptions grow from 457 to 465 and from 494
  to 560.
- An unsteady record alone (RTX 3080 Ti): one record at ×1.065 leaves 28 of 972 adopted rows slower than the default
  at a row count by records of another day; the rule in both of two records 5 of 467, the earlier rule 3 of 411.
- Picking a winner: two records of one device pick different winners in 5.2, 20.3 and 13.0 % of the pairs (6.9, 16.7
  and 9.9 % under the earlier rule), mostly near ties (median ×1.009, ×1.064, ×1.013 apart). Keeping the table's choice
  unless a challenger beats it by the bar moves 0 of 544, 2 of 321 and 0 of 439 choices, and the time behind the best
  falls from 0.12 / 0.88 / 0.66 % to 0.05 / 0.24 / 0.23 %.
- The reference's repeat across records is not needed on a steady device: the reference is more than ×1.10 slower
  than its fastest time elsewhere in 11 of 2,932 groups, and in 9 of those the whole group moved with it; the
  reference alone moved by more than ×1.065 in 7 of 2,932 groups, all at M = 1. No record repeats a kernel within a
  group (0 of 10,327 groups), so a repeat inside the group could not be measured — hence the sweep's re-timing of its
  references from 2026-11-28 on.
- Short passes: between 20 and 60 ms (Apple, the dispatch bound of 16,384 reached) the error grows (M5 p50 ×1.021
  against ×1.004), but the gaps grow with it and the bound is missed in 0.44 % against 0.40 %; below 20 ms there are
  3 and 5 rows (RTX 5070 Ti, RTX 3080 Ti), 3 of them misses — too few to set a wider margin, so such a group is no
  evidence.

## Refuted

- "×1.065 is a 99 % bound of any later run": refuted across days (above). It holds within a session.
- "The constants hold on devices and days the fit did not use": refuted for the bound's coverage (M2 left out 1.43 %,
  RTX 5070 Ti across days 3.81 % and 1.75 %); at the level of adoptions a first derivation stayed harmless.
- "The class separates any record": refuted below 20 groups (above).
- "Short passes need a wider margin": refuted — the gap term already absorbs them; a floor is safer.
- "The rule has no other path to adopt a slow kernel or refuse a fast one": refuted — single kernels' moves between
  sessions; the class of small records; a sequential key shared by several cases keeps the default when their winners
  differ (14 of 48 on the RTX 5070 Ti, where one candidate clears the bar in every case); the `explore` set's
  prediction leaves a clear winner unmeasured (RTX 5070 Ti 1 of 155, RTX 3080 Ti 137 of 1,699); the winner picked by
  M = 1 alone forgoes gains at M = 4 and 8 (the choice differs in 30 to 54 % of the cases with several qualifiers).
  The last three are properties of the earlier rule too.

## Limits

- Three steady devices, two of them Apple; one unsteady device (the RTX 3080 Ti under Node). No integrated GPU, no
  thermally limited laptop and no unsteady device in a browser: a device between steady and unsteady has no record.
- Few pairs across sessions: the RTX 5070 Ti over three days, the M5 over two, the M2 over 17 hours.
- The truth is the median gain of the other records, often one or two of them, so it moves too; binomial upper
  bounds assume independent rows, which a group's rows are not.
