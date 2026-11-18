> This document is a point-in-time snapshot (2026-11-18) of the per-shape variants of the sequential GEMV that the
> default geometry profile gained that day: what the GEMV sweep measured for the int4 output heads of the Gemma 4 q4_0
> QAT E2B / E4B models and for the bf16 linears of regular Gemma 4 E2B / E4B, what the derivation adopted, how the
> identity was checked and what a decode token gained, on the maintainer's RTX 3080 Ti under Node + Dawn. Chrome, the
> RTX 5070 Ti and Metal were not run. It is not a design source of truth. The scripts, logs and raw records are local
> and untracked (`outputs/diag/decode-optimization-2026-11-18/seq-variants/`,
> `outputs/workbench/node-nvidia-geforce-rtx-3080-ti/`).

# Per-shape variants of the sequential GEMV: the int4 output heads and the bf16 linears of Gemma 4

## 1. Summary

- The sequential GEMV of one row has a variant: the output columns one workgroup takes and the weight words it reads
  before it multiply-adds them (`c<columns>u<words>`). Every variant adds the same products in the same order, so its
  words are the same; the default profile ran `c32u4` on every shape but the int8 output head.
- Two findings of the same day's work said `c32u4` is slow in two places
  ([the census](2026-11-18-gemma4-gemv-census.md) §5, [the float GEMV](2026-11-18-float-parallel-gemv.md) §5). With
  ten more sweep records the derivation adopts another variant for 25 shapes (§3), and the default profile now carries
  them as per-shape exceptions:
  - the int4 output heads `262144 × 1536` and `262144 × 2560`: `c16u2`, ×1.68–2.52 per dispatch;
  - 23 bf16 linears of regular Gemma 4 E2B / E4B (every bf16 case but the two output heads): `c16u8`, `c16u16` or
    `c32u16`, ×1.17–1.89 per dispatch.
- A decode token of regular Gemma 4 `bf16` takes 18.8 ms in place of 23.4 ms (E2B) and 27.7 ms in place of 36.1 ms
  (E4B): the median over 8 visits of a visit's median interval between two tokens, the default faster in 8 of 8
  rounds on both (§5). The tokens are the same.
- The heads gain less end to end than their dispatch: 3.4 to 6.7 % of a decode token on the q4_0 QAT E2B `i4-fast`
  and E4B `i4` / `i4-fast` (7 or 8 of 8 rounds), and nothing that 20 rounds resolve on E2B `i4` (0.2 ms of 15.3, 13
  of 20 rounds).
- The identity: 642 timed rows of the identical class in the 15 records, none with a differing word; a GPU test that
  compares every exception with `c32u4` on its own shape; the sha256 rows of the corpus' outputs and of the models'
  token ids unmoved (§4).

## 2. Setup

- RTX 3080 Ti (12 GB), Node 24 + Dawn (Vulkan), the table `nvidia-ampere`. The GPU ran nothing else; a job started
  with the card at 38 to 84 °C, most of them at 64 to 77 °C.
- Per dispatch: the workbench's `gemv-sweep` entry, set `full` narrowed to one row (`--rows=1`), 5 interleaved rounds,
  the quick look at its default margin, 10 s idle between cases. A kernel's time is the minimum of its rounds.
- The records the derivation read (`seq-variants/records.txt`, 15): five of the earlier work that hold these cases
  (`gemv-sweep_2026-11-18T03-35-23`, `03-41-44`, `04-12-00`, `04-39-54`, `04-56-22`) and ten taken for this, at
  commit `72f2ada9` with a clean checkout: the whole grid of sequential variants over E2B's 13 bf16 cases (`11-32-42`),
  E4B's 12 (`11-41-59`) and the two int4 heads (`11-43-05`); a two-case probe (`11-36-30`); and three repetitions of
  the bf16 cases with the candidates narrowed to the four variants the whole grid left near the front, `c16u8`,
  `c16u16`, `c32u8` and `c32u16` (`11-46-29`, `11-49-44`, `11-53-13`, `11-56-28`, `11-59-54`, `12-03-07`).
- A decode token: the workbench's `gemma4-bench` entry, greedy, the prompt `few-ids`, 3 chats per visit (the first
  left out). The entry acquires one device for all its arms, and the geometry profile is the device's, so no arm of it
  can differ in the profile. The runs used a temporary patch of the entry
  (`seq-variants/bench-general-sequential.patch`, not committed): every visit acquires its own device, and an arm with
  the switch `general-sequential` acquires it with the table the adapter selects and GEMV choices without any
  per-shape exception of the sequential variant. Arm 1 is the default profile, arm 2 the same model and preset with
  the switch. No model of these runs has the int8 head's shape, so arm 2 is the profile as it was before.
- Arms are compared by the visit. A visit builds its device and its pipeline, and its two warm chats usually agree
  within 0.2 ms per token, while two visits of one arm differ by 0.3 to 1 ms and now and then by 5 ms (QAT E2B `i4`:
  13.8, 13.9, 14.7, 14.1, 14.2, 14.1, 14.6 and 18.6 ms of GPU time per token over the 8 visits of one arm). The figures
  below are the median and the minimum over an arm's visits, and the difference paired by round: the two visits of a
  round run back to back.

## 3. Per dispatch: what the derivation adopts

The rule ([spec](../spec/geometry-profiles.md#deriving-the-gemv-choices)): at one row, groups of at least two records
that are steady (baseline spread at most 1.25), the candidate at least ×1.10 faster than the default's choice in every
one, its words the baseline's in every current group. µs per dispatch over the groups that are evidence:

| Storage | Shape | Variant | `c32u4`, µs | Variant, µs | Gain | Records |
| --- | --- | --- | --- | --- | --- | --- |
| i4 | `262144 × 1536` | `c16u2` | 1171.3–1599.6 | 627.4–637.0 | ×1.84–2.52 | 3 |
| i4 | `262144 × 2560` | `c16u2` | 2256.9–2268.0 | 1320.9–1344.8 | ×1.68–1.71 | 3 |
| bf16 | `8960 × 1536` | `c16u8` | 60.5–60.6 | 36.6–36.8 | ×1.65–1.66 | 3 |
| bf16 | `2048 × 1536` | `c16u16` | 27.8–27.9 | 17.5 | ×1.59–1.60 | 3 |
| bf16 | `256 × 1536` | `c16u16` | 17.3 | 10.1 | ×1.72 | 3 |
| bf16 | `1536 × 2048` | `c16u8` | 33.0–36.0 | 21.3–22.7 | ×1.55–1.60 | 4 |
| bf16 | `6144 × 1536` | `c16u8` | 44.0–45.6 | 26.9–28.1 | ×1.62–1.64 | 5 |
| bf16 | `1536 × 6144` | `c16u16` | 117.9–118.8 | 71.0–71.5 | ×1.66 | 3 |
| bf16 | `1536 × 256` | `c32u16` | 3.94–4.04 | 3.16–3.20 | ×1.24–1.26 | 5 |
| bf16 | `4096 × 1536` | `c16u8` | 35.9 | 22.7 | ×1.58 | 3 |
| bf16 | `512 × 1536` | `c16u16` | 17.4–19.4 | 10.2 | ×1.70–1.89 | 4 |
| bf16 | `1536 × 4096` | `c16u16` | 75.3–75.5 | 46.4–46.7 | ×1.62 | 3 |
| bf16 | `12288 × 1536` | `c16u8` | 78.4–83.1 | 49.6–52.0 | ×1.55–1.64 | 5 |
| bf16 | `1536 × 12288` | `c16u16` | 238.5–249.2 | 138.8–142.3 | ×1.72–1.75 | 5 |
| bf16 | `10752 × 2560` | `c16u8` | 121.3–122.0 | 71.4–71.9 | ×1.69–1.71 | 5 |
| bf16 | `2048 × 2560` | `c16u8` | 52.4–55.1 | 31.6–33.5 | ×1.60–1.66 | 4 |
| bf16 | `512 × 2560` | `c16u16` | 28.2–28.3 | 16.1–16.2 | ×1.75 | 3 |
| bf16 | `256 × 2560` | `c16u16` | 28.2 | 16.0–16.1 | ×1.76 | 3 |
| bf16 | `2560 × 2048` | `c16u8` | 40.6–41.0 | 26.2–28.3 | ×1.45–1.55 | 4 |
| bf16 | `10240 × 2560` | `c16u8` | 114.7–120.7 | 67.8–71.5 | ×1.69–1.70 | 5 |
| bf16 | `2560 × 10240` | `c16u8` | 212.6–222.3 | 125.3–133.5 | ×1.67–1.73 | 5 |
| bf16 | `2560 × 256` | `c16u16` | 4.04–4.10 | 3.41–3.50 | ×1.17–1.19 | 5 |
| bf16 | `4096 × 2560` | `c16u8` | 62.3–64.0 | 37.1–37.7 | ×1.68–1.71 | 4 |
| bf16 | `1024 × 2560` | `c16u16` | 28.3–29.1 | 16.6–17.0 | ×1.70–1.71 | 5 |
| bf16 | `2560 × 4096` | `c16u8` | 80.1–84.0 | 49.4–50.2 | ×1.62–1.67 | 5 |

- Among several candidates the derivation takes the one whose smallest gain is largest. On most bf16 shapes `c16u8`
  and `c16u16` are within a few percent of each other, and which one is adopted moved with the records given
  (`1536 × 2048` was `c16u16` on 14 records and `c16u8` on 15). The shapes do not agree on one variant.
- Not adopted: the two bf16 output heads. `262144 × 1536`: `c16u8` takes about 1.71 ms in every record, while `c32u4`
  took 1.53 ms in one record and 2.55 to 2.56 ms in four, so the candidate is ×0.90 in one group and ×1.48–1.50 in the
  others; the rule asks for ×1.10 in every one. `262144 × 2560`: the baseline was unsteady in four of its five groups.
- Adopted by the derivation and not applied: the drafter shape int4 `256 × 2048` (`c16u16`, ×1.18 in two earlier
  records). It is outside the two items this work was given.
- Steadiness. With every candidate of the row count in the rotation, 9 of E2B's 13 groups and 3 of E4B's 12 were
  unsteady: in some rounds a sequential variant took about twice its fastest round (`6144 × 1536`, `c32u4`: 44.3,
  44.9, 82.9, 83.1 and 42.9 µs), while each parallel candidate of the same rounds stayed within 0.5 µs of itself.
  With the candidates narrowed to four sequential variants, 70 of 75 groups were steady; the five that were not are
  the E4B output head's twice and three other shapes' once each.

## 4. The identity

- The sweep compares every timed candidate of the identical class with the baseline word for word, on the case's real
  shape: 642 such rows in the 15 records, 0 with a differing word.
- `gpu_linear_gemv_variant_test.ts` runs, for every per-shape exception of the default profile (it reads the rows, so
  the int8 head's too), the general variant and the exception's on the same buffers and compares the outputs as u32:
  on the exception's own shape, on `68 × (3·u − 1)` words — a partial last workgroup and a remainder after the
  read-ahead loop of both variants — and on `4 ×` one word, which never enters the loop. A host reference reads every
  column of the small shapes and 256 columns of a real one.
- Fault injections on the 26 rows: the first two multiply-adds of every run swapped in the exception's WGSL alone — the
  u32 comparison fails on the real shape of all 26 rows and on 29 of the 52 small shapes, the host reference on none;
  the exception's kernel returning before it writes — both fail on all 78. The second passed until the test cleared
  the output buffer between the two kernels: both write one buffer, and it still held the general variant's words.
- The corpus (`gpu_corpus_test.ts`, 120 cases): every outputs row holds. Four census rows no longer match, all of the
  entry `gemma4-qat-e2b-cut4` (`prefill8` and `prefill64`, at `exact` and at `reorder`): in each step one dispatch of
  `linear_gemv:v1:f32:c32u4:wi4g32`, the head, is now `linear_gemv:v1:f32:c16u2:wi4g32`. They were not recorded again
  here.
- The token-id rows of the real models hold: `e2e_gemma4_reference_test.ts` (regular E2B and E4B `bf16`) and
  `e2e_gemma4_qat_reference_test.ts` (the q4_0 QAT E2B, E4B and 12B at `i4` and `i4-fast`, and E2B at a long
  position), 9 rows.
- Every chat of every A/B below gave the same token ids in both arms (one sha256 per record).

## 5. End to end

GPU time per decode token and the median interval between two tokens, ms; "default" is the profile with the
exceptions, "before" the same without; the difference is before − default, paired by round.

| Model, preset | Rounds | GPU: default / before (median; minimum) | Ratio | Paired difference | Interval: default / before (median) | Ratio | Paired difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Regular E2B `bf16` | 8 | 17.81 / 22.83; 15.76 / 20.48 | ×1.282 | 5.54, 8 of 8 | 18.83 / 23.42 | ×1.244 | 4.89, 8 of 8 |
| Regular E4B `bf16` | 8 | 26.79 / 35.12; 24.72 / 34.46 | ×1.311 | 8.18, 8 of 8 | 27.67 / 36.12 | ×1.306 | 8.95, 8 of 8 |
| QAT E2B `i4` | 20 | 14.50 / 14.49; 13.81 / 13.79 | ×0.999 | 0.11, 13 of 20 | 15.33 / 15.41 | ×1.005 | 0.20, 13 of 20 |
| QAT E2B `i4-fast` | 8 | 5.97 / 6.20; 5.91 / 5.97 | ×1.039 | 0.21, 8 of 8 | 6.42 / 6.66 | ×1.038 | 0.23, 8 of 8 |
| QAT E4B `i4` | 8 | 22.53 / 23.54; 21.93 / 23.15 | ×1.045 | 1.06, 7 of 8 | 23.61 / 24.44 | ×1.035 | 1.11, 7 of 8 |
| QAT E4B `i4-fast` | 8 | 10.56 / 11.00; 9.83 / 9.98 | ×1.042 | 0.51, 7 of 8 | 11.07 / 11.87 | ×1.072 | 0.80, 8 of 8 |
| Mobile QAT E2B `i4-fast` (no exception applies) | 12 | 5.91 / 5.87; 5.66 / 5.74 | ×0.994 | 0.00, 6 of 12 | 6.52 / 6.52 | ×1.000 | 0.03, 8 of 12 |

- The regular models: two records of 4 rounds and 96 new tokens each, pooled. The QAT rows: 8 rounds of 128 new
  tokens; E2B `i4` one record of 8 and one of 12. A first pass of 4 rounds and 96 tokens per QAT row read ×1.02 to
  ×1.06 on the medians of its chats and is not in the table. The last row is the control: two arms that run the same
  kernels, 4 and 8 rounds.
- With one timed pass per dispatch (the `split` switch, 2 rounds, 64 tokens), the linears of a token:

  | Model, preset | Linear, default / before | The kernels that moved |
  | --- | --- | --- |
  | QAT E2B `i4` | 11.83 / 12.36 | the head: 1.15 → 0.63 ms |
  | QAT E2B `i4-fast` | 5.07 / 5.69 | the head: 1.23 → 0.63 ms |
  | QAT E4B `i4` | 22.12 / 23.46 | the head: 3.23 → 1.89 ms (the first by difference: 344 dispatches of one key) |
  | QAT E4B `i4-fast` | 8.70 / 9.67 | the head: 2.33 → 1.37 ms |
  | Regular E2B `bf16` | 16.76 / 19.36 | 276 linears on `c16u16` (135), `c16u8` (106) and `c32u16` (35); the head stays on `c32u4`, 2.62 ms |
  | Regular E4B `bf16` | 26.84 / 33.21 | 343 linears on `c16u8` (211) and `c16u16` (132); the head stays on `c32u4`, 4.65 ms |

- The two views disagree in both directions. The bf16 models gain more in the product's own passes (5.5 and 8.2 ms of
  GPU time) than dispatch by dispatch (2.6 and 6.4 ms). The QAT E2B head saves 0.53 ms when it is timed alone, and the
  `i4` token, where it follows 276 other sequential linears, shows 0.1 to 0.2 ms that 20 rounds do not resolve. Why
  was not looked into; the float note saw the sequential kernels of a token run slower than their fastest round of the
  sweep ([§7](2026-11-18-float-parallel-gemv.md)).

## 6. What goes stale

- GEMV sweep records of the RTX 3080 Ti: the one-row groups of the 25 shapes, whose default choice moved — 126 groups
  in 15 of the 30 records of format `karume-gemv-sweep/4` (the generator's dry run on the 15 records of §2 reads
  their one-row groups of these cases as not current, and proposes no exception but the drafter shape of §3). Their
  groups at 4 and 8 rows stand: a float-stored case's baseline there is the GEMM and the heads' is a row block, which
  take no exception.
- No record of another device holds these cases.
- The four census rows of §4.

## 7. Noticed beside the topic

- A device and pipeline come up in a state that holds for the visit: a visit's GPU time per token differs from
  another's of the same arm by up to 35 %, far more than its own chats differ. The sweep shows the same on single
  kernels (the int4 E2B head on `c32u4`: 1171 µs in two records, 1600 µs in a third; the bf16 E2B head 1531 and
  2550 µs) and, inside a group, as the rounds of §3. The variants of 16 columns moved far less (the int4 E2B head on
  `c16u2`: 627 to 637 µs in all three records). An A/B of two resident pipelines reads this state as a difference
  between its arms.
- The i8 exception's own test (the first case of `gpu_linear_gemv_variant_test.ts`) does not clear its output between
  the two kernels either.

## 8. Not measured

- Any other device. Every embedded table runs with the default profile's GEMV choices, so Chrome on the RTX 5070 Ti
  and Metal run these exceptions too; whether they gain or lose there is not known. The identity was checked on Dawn /
  Vulkan only.
- The model lanes, beyond the two reference files of §4.
- A speculative decode: the verify run puts the head at several rows, a row block, which takes no exception; the
  draft steps run the drafter, whose shapes have none.
- The second prompt of the bench, a sampled decode, longer generations.

## 9. Commands

```sh
# Per dispatch: the whole grid of sequential variants at one row (E2B's bf16 cases; about 5 minutes)
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemv-sweep --candidateSet=full --rows=1 \
  --cases=bf16-n8960-k1536,bf16-n2048-k1536,bf16-n256-k1536,bf16-n1536-k2048,bf16-n6144-k1536,bf16-n1536-k6144,bf16-n1536-k256,bf16-n4096-k1536,bf16-n512-k1536,bf16-n1536-k4096,bf16-n12288-k1536,bf16-n1536-k12288,bf16-n262144-k1536
# The same, narrowed to four variants (steady groups; about 3 minutes)
#   … --candidates=c16u8,c16u16,c32u8,c32u16

# What the records adopt
node tools/geometry-profiles/generate.ts nvidia-ampere --gemv-records=<record>,<record>,… --dry-run

# The identity
flock /tmp/karume-gpu.lock vp test packages/runtime/tests/gpu_linear_gemv_variant_test.ts
flock /tmp/karume-gpu.lock vp test packages/runtime/tests/gpu_corpus_test.ts

# A decode token with and without the exceptions (the patch is temporary: revert it afterwards)
git apply outputs/diag/decode-optimization-2026-11-18/seq-variants/bench-general-sequential.patch
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemma4-bench \
  --arms=gemma4/e2b@bf16,gemma4/e2b@bf16:general-sequential --rounds=4 --chats=3 --newTokens=96 --prompts=few-ids
git apply -R outputs/diag/decode-optimization-2026-11-18/seq-variants/bench-general-sequential.patch
```
