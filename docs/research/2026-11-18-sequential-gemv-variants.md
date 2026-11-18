> This document is a point-in-time snapshot (2026-11-18) of the per-shape variants of the sequential GEMV that the
> default geometry profile and the table `nvidia-ampere` gained that day: what the GEMV sweep measured for the int4
> output heads of the Gemma 4 q4_0 QAT E2B / E4B models and for the bf16 linears of regular Gemma 4 E2B / E4B on the
> maintainer's RTX 3080 Ti (Node + Dawn) and RTX 5070 Ti (Chrome, D3D12), what the derivation adopted, where the rows
> went, how the identity was checked and what a decode token gained on the RTX 3080 Ti. Metal was not run, and no
> decode token was measured on the RTX 5070 Ti. It is not a design source of truth. The scripts, logs and raw records
> are local and untracked (`outputs/diag/decode-optimization-2026-11-18/seq-variants/`, `outputs/workbench/`).

# Per-shape variants of the sequential GEMV: the int4 output heads and the bf16 linears of Gemma 4

## 1. Summary

- The sequential GEMV of one row has a variant: the output columns one workgroup takes and the weight words it reads
  before it multiply-adds them (`c<columns>u<words>`). Every variant adds the same products in the same order, so its
  words are the same; the default profile ran `c32u4` on every shape but the int8 output head.
- Two findings of the same day's work said `c32u4` is slow in two places on the RTX 3080 Ti
  ([the census](2026-11-18-gemma4-gemv-census.md) §5, [the float GEMV](2026-11-18-float-parallel-gemv.md) §5). With
  ten more sweep records the derivation adopts another variant for 26 shapes there (§3): the int4 output heads
  `262144 × 1536` and `262144 × 2560` (`c16u2`, ×1.68–1.87 per dispatch, and ×2.52 in one record whose baseline ran
  in its slow state), 23 bf16 linears
  of regular Gemma 4 E2B / E4B — every bf16 case but the two output heads — (`c16u8`, `c16u16` or `c32u16`,
  ×1.17–1.89), and the MTP drafter shape int4 `256 × 2048` (`c16u16`, ×1.18).
- The RTX 5070 Ti measured the same shapes the same day. 22 of the bf16 shapes are faster there with the RTX
  3080 Ti's variant too (×1.015–1.495); the two int4 heads (×0.89–0.90), bf16 `10240 × 2560` (×0.945) and the drafter
  shape (×0.91) are slower. On D3D12 every one of those rows wrote the baseline's words.
- Where the rows are (the maintainer's ruling of 2026-11-18: the default keeps a row only where no measured device is
  slower with it; a device's own best lives in its table): the 22 bf16 rows are per-shape exceptions of the default
  profile; the other four are rows of `nvidia-ampere`, the RTX 3080 Ti's table, which now carries GEMV choices of its
  own.
- A decode token of regular Gemma 4 `bf16` on the RTX 3080 Ti takes 18.8 ms in place of 23.4 ms (E2B) and 27.7 ms in
  place of 36.1 ms (E4B) — the median over 8 visits, each visit the mean of its warm chats' median interval between
  two tokens —, the exceptions faster in 8 of 8 rounds on both (§5). The acceptance run read 18.7 in place of 23.2 ms
  and 28.1 in place of 36.2 ms. The tokens are the same.
- The heads gain less end to end than their dispatch. By the arms' medians a decode token of the q4_0 QAT line read
  3.7 % (E2B `i4-fast`), 3.4 % (E4B `i4`) and 6.7 % (E4B `i4-fast`) shorter, in 7 or 8 of 8 rounds, and nothing that
  20 rounds resolve on E2B `i4` (0.2 ms of 15.3, 13 of 20 rounds). The acceptance run repeated E4B `i4` (3.5 %, 7 of
  8 rounds) and did not repeat E2B `i4-fast` (0.6 %, 6 of 8 rounds).
- The identity: 642 timed rows of the identical class in the 15 records of the RTX 3080 Ti and 527 in four of the
  RTX 5070 Ti's, none with a differing word; a GPU test that compares every exception of the default profile and of
  `nvidia-ampere` with `c32u4` on its own shape; the sha256 rows of the corpus' outputs and of the models' token ids
  unmoved (§4).

## 2. Setup

- RTX 3080 Ti (12 GB), Node 24 + Dawn (Vulkan), the table `nvidia-ampere`. The GPU ran nothing else; a job started
  with the card at 38 to 84 °C, most of them at 64 to 77 °C.
- Per dispatch: the workbench's `gemv-sweep` entry, set `full` narrowed to one row (`--rows=1`), 5 interleaved rounds,
  the quick look at its default margin, 10 s idle between cases. A kernel's time is the minimum of its rounds.
- The records the derivation read (`seq-variants/records.txt`, 15): five of the earlier work that hold these cases
  (`gemv-sweep_2026-11-18T03-35-23`, `03-41-44`, `04-12-00`, `04-39-54`, `04-56-22`) and ten taken for this, at
  commit `72f2ada9` with a clean checkout: the whole grid of sequential variants over E2B's 13 bf16 cases (`11-32-42`),
  E4B's 12 (`11-41-59`) and the two int4 heads (`11-43-05`); a two-case probe (`11-36-30`); and three repetitions of
  the bf16 cases with the candidates narrowed to four variants, `c16u8`, `c16u16`, `c32u8` and `c32u16` (`11-46-29`,
  `11-49-44`, `11-53-13`, `11-56-28`, `11-59-54`, `12-03-07`).
  - Four of the five earlier records came from checkouts with uncommitted changes (`03-35-23`, `04-12-00` and
    `04-39-54` at `7834560b`, `04-56-22` at `a9daa291`; `03-41-44` at `a9daa291` was clean). The derivation reads a
    row by the fingerprint of its kernel, not by its checkout, and reads all of them as current.
  - The four candidates of the narrowed repetitions were chosen by hand: the variants the whole-grid records, most of
    their groups unsteady, left near the front. The adoption is therefore the best of the measured candidates. The
    measurement planner (`--gemv-plan`) on the 15 records still lists, for 12 bf16 cases at one row, 30 candidates
    that fewer than two records hold a steady group of — the 20 other sequential variants among them. The dry run
    names a narrowed record by its set, `full`.
- The RTX 5070 Ti (Chrome 157 on Windows, D3D12, the table `nvidia-blackwell`, which carried no GEMV choices then,
  checkout `72f2ada9`, clean — so its baseline is `c32u4` on every shape): `gemv-sweep_2026-11-18T12-22-44` and `13-53-17` (the int4 heads),
  `12-58-03` and `14-28-44` (the 23 bf16 shapes), `12-32-24` and `14-02-58` (the drafter shape), all at one row with
  the whole grid, every group steady. This work did not take them.
- A decode token: the workbench's `gemma4-bench` entry, greedy, the prompt `few-ids`, 3 chats per visit (the first
  left out). The geometry profile is the device's, and the entry ran every arm on one device, so the runs used a
  temporary patch of the entry (`seq-variants/bench-general-sequential.patch`): every visit acquires its own device,
  and an arm with the switch `general-sequential` acquires it with the table the adapter selects and GEMV choices
  without any per-shape exception of the sequential variant. The entry has had that switch since
  ([the spec](../spec/workbench.md#the-decode-benchs-profile-switches)). The runs were made at commit `57532ecd`
  (this work's first commit before a rebase: all 25 rows in the default profile), the checkout holding the patch and
  nothing else uncommitted. Arm 1 is the profile with the exceptions, arm 2 the same model and preset with the
  switch. No model of these runs has the int8 head's shape, so arm 2 is the profile as it was before.
- Arms are compared by the visit. A visit builds its device and its pipeline, and its two warm chats usually agree
  within 0.2 ms per token, while two visits of one arm differ by 0.3 to 1 ms and now and then by 5 ms (QAT E2B `i4`:
  13.8, 13.9, 14.7, 14.1, 14.2, 14.1, 14.6 and 18.6 ms of GPU time per token over the 8 visits of one arm). A visit's
  figure is the mean over its warm chats of the chat's GPU time per token, or of the chat's median interval between
  two tokens (`seq-variants/ab-visits.mjs`). The figures below are the median and the minimum of that over an arm's
  visits, and the difference paired by round: the two visits of a round run back to back. The workbench's A/B judge
  was not used.

## 3. Per dispatch: what the derivation adopts, and the other device

The rule ([spec](../spec/geometry-profiles.md#deriving-the-gemv-choices)): at one row, groups of at least two records
that are steady (baseline spread at most 1.25), the candidate at least ×1.10 faster than the default's choice in every
one, its words the baseline's in every current group. The RTX 3080 Ti's columns are µs per dispatch and the gain over
the groups that are evidence, from the dry run on the 15 records before any of these rows was in the default profile;
the RTX 5070 Ti's column is the same variant against `c32u4` in its two records.

| Storage | Shape | Variant | `c32u4`, µs | Variant, µs | Gain, RTX 3080 Ti | Records | Gain, RTX 5070 Ti | Row of |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i4 | `262144 × 1536` | `c16u2` | 1171.3–1599.6 | 627.4–637.0 | ×1.84–2.52 | 3 | ×0.886–0.897 | `nvidia-ampere` |
| i4 | `256 × 2048` | `c16u16` | 17.6 | 14.9 | ×1.18 | 2 | ×0.907–0.910 | `nvidia-ampere` |
| i4 | `262144 × 2560` | `c16u2` | 2256.9–2268.0 | 1320.9–1344.8 | ×1.68–1.71 | 3 | ×0.892–0.900 | `nvidia-ampere` |
| bf16 | `8960 × 1536` | `c16u8` | 60.5–60.6 | 36.6–36.8 | ×1.65–1.66 | 3 | ×1.049–1.051 | default |
| bf16 | `2048 × 1536` | `c16u16` | 27.8–27.9 | 17.5 | ×1.59–1.60 | 3 | ×1.285–1.288 | default |
| bf16 | `256 × 1536` | `c16u16` | 17.3 | 10.1 | ×1.72 | 3 | ×1.456–1.461 | default |
| bf16 | `1536 × 2048` | `c16u8` | 33.0–36.0 | 21.3–22.7 | ×1.55–1.60 | 4 | ×1.377–1.378 | default |
| bf16 | `6144 × 1536` | `c16u8` | 44.0–45.6 | 26.9–28.1 | ×1.62–1.64 | 5 | ×1.084–1.087 | default |
| bf16 | `1536 × 6144` | `c16u16` | 117.9–118.8 | 71.0–71.5 | ×1.66 | 3 | ×1.308–1.318 | default |
| bf16 | `1536 × 256` | `c32u16` | 3.94–4.04 | 3.16–3.20 | ×1.24–1.26 | 5 | ×1.031–1.034 | default |
| bf16 | `4096 × 1536` | `c16u8` | 35.9 | 22.7 | ×1.58 | 3 | ×1.142 | default |
| bf16 | `512 × 1536` | `c16u16` | 17.4–19.4 | 10.2 | ×1.70–1.89 | 4 | ×1.473–1.478 | default |
| bf16 | `1536 × 4096` | `c16u16` | 75.3–75.5 | 46.4–46.7 | ×1.62 | 3 | ×1.322–1.325 | default |
| bf16 | `12288 × 1536` | `c16u8` | 78.4–83.1 | 49.6–52.0 | ×1.55–1.64 | 5 | ×1.061 | default |
| bf16 | `1536 × 12288` | `c16u16` | 238.5–249.2 | 138.8–142.3 | ×1.72–1.75 | 5 | ×1.318–1.330 | default |
| bf16 | `10752 × 2560` | `c16u8` | 121.3–122.0 | 71.4–71.9 | ×1.69–1.71 | 5 | ×1.015–1.016 | default |
| bf16 | `2048 × 2560` | `c16u8` | 52.4–55.1 | 31.6–33.5 | ×1.60–1.66 | 4 | ×1.335 | default |
| bf16 | `512 × 2560` | `c16u16` | 28.2–28.3 | 16.1–16.2 | ×1.75 | 3 | ×1.485–1.494 | default |
| bf16 | `256 × 2560` | `c16u16` | 28.2 | 16.0–16.1 | ×1.76 | 3 | ×1.458–1.476 | default |
| bf16 | `2560 × 2048` | `c16u8` | 40.6–41.0 | 26.2–28.3 | ×1.45–1.55 | 4 | ×1.417–1.425 | default |
| bf16 | `10240 × 2560` | `c16u8` | 114.7–120.7 | 67.8–71.5 | ×1.69–1.70 | 5 | ×0.945 | `nvidia-ampere` |
| bf16 | `2560 × 10240` | `c16u8` | 212.6–222.3 | 125.3–133.5 | ×1.67–1.73 | 5 | ×1.461–1.466 | default |
| bf16 | `2560 × 256` | `c16u16` | 4.04–4.10 | 3.41–3.50 | ×1.17–1.19 | 5 | ×1.271–1.345 | default |
| bf16 | `4096 × 2560` | `c16u8` | 62.3–64.0 | 37.1–37.7 | ×1.68–1.71 | 4 | ×1.106 | default |
| bf16 | `1024 × 2560` | `c16u16` | 28.3–29.1 | 16.6–17.0 | ×1.70–1.71 | 5 | ×1.491–1.495 | default |
| bf16 | `2560 × 4096` | `c16u8` | 80.1–84.0 | 49.4–50.2 | ×1.62–1.67 | 5 | ×1.436–1.442 | default |

- The int4 E2B head's ×2.52 is against a baseline in its slow state: `c32u4` took 1600 µs in that record and 1171 and
  1173 µs in the two others, where the gain is ×1.84 and ×1.87 (§7).
- Among several candidates the derivation takes the one whose smallest gain is largest. Where two variants are within
  a few percent of each other the choice moves with the records given, each left out in turn: without `11-59-54`
  bf16 `1536 × 2048` is `c16u16`; without `04-39-54` bf16 `2560 × 10240` is `c16u16`; without `04-56-22` the int4
  E2B head is `c16u1`, and without `03-41-44` the E4B head is. The drafter shape rests on exactly two records
  (`03-41-44`, `04-56-22`). The words are the same whichever is chosen; the shapes do not agree on one variant.
- Not adopted: the two bf16 output heads. `262144 × 1536`: `c16u8` takes about 1.71 ms in every record, while `c32u4`
  took 1.53 ms in one record and 2.55 to 2.56 ms in four, so the candidate is ×0.90 in one group and ×1.48–1.50 in the
  others, and the rule asks for ×1.10 in every one. The refusal rests on that one record: without `04-12-00` the
  derivation adopts `c16u8` for it. `262144 × 2560`: the baseline was unsteady in four of its five groups.
- Steadiness. With every candidate of the row count in the rotation, 9 of E2B's 13 groups and 3 of E4B's 12 were
  unsteady on the RTX 3080 Ti: in some rounds a sequential variant took about twice its fastest round
  (`6144 × 1536`, `c32u4`: 44.3, 44.9, 82.9, 83.1 and 42.9 µs), while each parallel candidate of the same rounds
  stayed within 0.5 µs of itself. With the candidates narrowed to four sequential variants, 70 of 75 groups were
  steady; the five that were not are the E4B output head's twice and three other shapes' once each. Several bf16
  shapes have all their evidence from those narrowed records, taken one after another within 17 minutes.
- On the RTX 5070 Ti the rows that are faster gain less than on the RTX 3080 Ti, and another variant is often the
  fastest there (`c16u8` where the row says `c16u16`, and the reverse). No rule adopts from one record of each; its
  own choices wait for its own derivation.
- Later the same day the table `nvidia-blackwell` was derived from 74 records of that device
  ([spec](../spec/geometry-profiles.md#the-gemv-choices-of-linear)). It carries 42 exceptions of its own, all on
  int4 and int8 shapes, and none of the four rows of `nvidia-ampere`. For the bf16 shapes of this note it runs the
  22 rows it inherits from the default: its two records of them were taken against `c32u4`, the default of before
  those rows, so the derivation reads them as no evidence, and the variant its own records would choose is not
  registered.
- Later still that day eight more records of the device (`gemv-sweep_2026-11-18T16-31-11` to `16-45-19`, checkout
  `2fcacc4e`, clean) measured the 22 bf16 shapes at one row against these rows (six records, two of each shape)
  and int4 `2048 × 2560` at eight rows (two), and the table was derived again from 82 records. It keeps 20 of them; on bf16 `1536 × 256` it takes `c16u8` in place of `c32u16` (×1.200 and
  ×1.212) and on `4096 × 2560` `c16u16` in place of `c16u8` (×1.126 and ×1.124). On the other 20 the nearest
  variant reads ×0.95 to ×1.08 of the row's, below ×1.10.

## 4. The identity

- The sweep compares every timed candidate of the identical class with the baseline word for word, on the case's real
  shape: 642 such rows in the 15 records of the RTX 3080 Ti (Vulkan), 0 with a differing word; and on D3D12, in the
  RTX 5070 Ti's `12-22-44`, `12-58-03`, `13-53-17` and `14-28-44`, 527 rows, 0 differing — the two int4 heads' and
  the 23 bf16 shapes' variants of §3 among them, each in two records; the drafter shape's is identical in `12-32-24`
  and `14-02-58`. Metal has no such row.
- `gpu_linear_gemv_variant_test.ts` runs, for every per-shape exception of the default profile and of every
  registered table that carries GEMV choices (it reads the rows: the int8 head's and `nvidia-ampere`'s too), the
  general variant and the exception's on the same buffers and compares the outputs as u32: on the exception's own
  shape, on `68 × (3·u − 1)` words — a partial last workgroup and a remainder after the read-ahead loop of both
  variants — and on `4 ×` one word, which never enters the loop. An int4 row has no group length, so it is compared
  at groups of 32 and of 64. A host reference reads every column of the small shapes and 256 columns of a real one.
- Fault injections on the 27 rows (90 shapes): the first two multiply-adds of every run swapped in the exception's
  WGSL alone — the u32 comparison fails on all 30 real shapes and on 34 of the 60 small ones, the host reference on
  none; the exception's kernel returning before it writes — both fail on all 90. The second passed until the test
  cleared the output buffer between the two kernels: both write one buffer, and it still held the general variant's
  words.
- The corpus (`gpu_corpus_test.ts`, 120 cases): every outputs row holds. Four census rows no longer match, all of the
  entry `gemma4-qat-e2b-cut4` (`prefill8` and `prefill64`, at `exact` and at `reorder`): in each step one dispatch of
  `linear_gemv:v1:f32:c32u4:wi4g32`, the head, is now `linear_gemv:v1:f32:c16u2:wi4g32`. That is so on this machine
  with the head's row in `nvidia-ampere` as it was with the row in the default profile (the same four sha256); on a
  machine whose table has no such row the rows hold. They were not recorded again here.
- The token-id rows of the real models hold: `e2e_gemma4_reference_test.ts` (regular E2B and E4B `bf16`) and
  `e2e_gemma4_qat_reference_test.ts` (the q4_0 QAT E2B, E4B and 12B at `i4` and `i4-fast`, and E2B at a long
  position), 9 rows.
- Every chat of every A/B below gave the same token ids in both arms (one sha256 per record).

## 5. End to end on the RTX 3080 Ti

GPU time per decode token and the interval between two tokens, ms, by the visit (§2); "with" is the profile with the
exceptions, "before" the same without; the difference is before − with, paired by round.

| Model, preset | Rounds | GPU: with / before (median; minimum) | Ratio | Paired difference | Interval: with / before (median) | Ratio | Paired difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Regular E2B `bf16` | 8 | 17.81 / 22.83; 15.76 / 20.48 | ×1.282 | 5.54, 8 of 8 | 18.83 / 23.42 | ×1.244 | 4.89, 8 of 8 |
| Regular E4B `bf16` | 8 | 26.79 / 35.12; 24.72 / 34.46 | ×1.311 | 8.18, 8 of 8 | 27.67 / 36.12 | ×1.306 | 8.95, 8 of 8 |
| QAT E2B `i4` | 20 | 14.50 / 14.49; 13.81 / 13.79 | ×0.999 | 0.11, 13 of 20 | 15.33 / 15.41 | ×1.005 | 0.20, 13 of 20 |
| QAT E2B `i4-fast` | 8 | 5.97 / 6.20; 5.91 / 5.97 | ×1.039 | 0.21, 8 of 8 | 6.42 / 6.66 | ×1.038 | 0.23, 8 of 8 |
| QAT E4B `i4` | 8 | 22.53 / 23.54; 21.93 / 23.15 | ×1.045 | 1.06, 7 of 8 | 23.61 / 24.44 | ×1.035 | 1.11, 7 of 8 |
| QAT E4B `i4-fast` | 8 | 10.56 / 11.00; 9.83 / 9.98 | ×1.042 | 0.51, 7 of 8 | 11.07 / 11.87 | ×1.072 | 0.80, 8 of 8 |
| Control, 4 rounds: mobile QAT E2B `i4-fast` (no exception applies) | 4 | 5.86 / 6.06; 5.66 / 5.81 | ×1.033 | 0.19, 4 of 4 | 6.48 / 6.50 | ×1.003 | 0.03, 4 of 4 |
| Control, 8 rounds: the same | 8 | 5.92 / 5.86; 5.87 / 5.74 | ×0.990 | −0.07, 2 of 8 | 6.55 / 6.54 | ×0.999 | 0.01, 4 of 8 |

- The regular models: two records of 4 rounds and 96 new tokens each, pooled. The QAT rows: 8 rounds of 128 new
  tokens; E2B `i4` one record of 8 and one of 12.
- The control's two arms run the same kernels. Its 4-round record alone reads a difference of the size of the QAT
  rows' (GPU ×1.033, 4 of 4 rounds); its 8-round record reads the other way. A QAT row of 8 rounds is therefore one
  observation of an effect near the noise of the method, not a measured size.
- A first pass of 4 rounds and 96 new tokens per QAT row is not in the table. By the visit it read, GPU time: E2B
  `i4` ×0.989 (the exceptions faster in 1 of 4 rounds), E2B `i4-fast` ×1.027 (4 of 4), E4B `i4` ×1.032 (3 of 4),
  E4B `i4-fast` ×1.017 (2 of 4).
- The acceptance run (the orchestrator's, the same patch at `ce3a2c69`, by the visit, interval without → with):
  regular E2B 23.15 → 18.73 ms (4 of 4 rounds), regular E4B 36.25 → 28.12 ms (4 of 4), QAT E4B `i4` 24.52 → 23.67 ms
  (7 of 8), QAT E2B `i4-fast` 6.529 → 6.492 ms (6 of 8; GPU 6.066 → 5.963 ms). The bf16 models and E4B `i4` repeat;
  the 3.7 % of E2B `i4-fast` does not (0.6 % by the interval, 1.7 % by the GPU time). Records
  `gemma4-bench_2026-11-18T13-38-33`, `13-41-34`, `13-44-36`, `13-50-05`.
- With one timed pass per dispatch (the `split` switch, 2 rounds, 64 tokens), the linears of a token:

  | Model, preset | Linear, with / before | The kernels that moved |
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
- These runs were made with all 25 rows in the default profile. With three of them in `nvidia-ampere` instead, a
  device of this adapter runs the same kernels on the shapes of these models.

## 6. What goes stale

- GEMV sweep records, the one-row groups of the 22 shapes whose default choice moved: on the RTX 3080 Ti 115 groups
  in 12 of its 30 records of format `karume-gemv-sweep/4`; on the RTX 5070 Ti 44 groups in 2 of its records
  (`12-58-03`, `14-28-44`). The groups at 4 and 8 rows stand (a float-stored case's baseline there is the GEMM), and
  so do the groups of the four shapes that are rows of `nvidia-ampere`: the sweep's baseline is the default profile's
  choice, which is `c32u4` for them again.
- The four census rows of §4, on this machine.
- What the same day's notes read against `c32u4` at `exact` on the 22 shapes: the per-dispatch gains of the parallel
  GEMV on bf16 weights and its decode token against `exact` ([the float GEMV](2026-11-18-float-parallel-gemv.md) §4,
  §7; perf-ledger K-110, K-111). The `exact` they were read against ran `c32u4` on these shapes.

## 7. Noticed beside the topic

- A device and pipeline come up in a state that holds for the visit: a visit's GPU time per token differs from
  another's of the same arm by up to 35 %, far more than its own chats differ. The sweep shows the same on single
  kernels (the int4 E2B head on `c32u4`: 1171 and 1173 µs in two records, 1600 µs in a third; the bf16 E2B head 1531
  and 2550 µs) and, inside a group, as the rounds of §3. In the sweep the variants of 16 columns moved far less (the
  int4 E2B head on `c16u2`: 627 to 637 µs in all three records), but not always in a pipeline: in two later runs of
  one round with a pass per dispatch that head read 0.64 ms and 1.41 ms. An A/B of two resident pipelines reads this
  state as a difference between its arms.
- The i8 exception's own test (the first case of `gpu_linear_gemv_variant_test.ts`) does not clear its output between
  the two kernels either.
- The default profile's older row, the int8 output head `262144 × 1536` on `c16u4`, does not meet the rule the new
  rows were placed by: in the RTX 5070 Ti's two records that hold it (`gemv-sweep_2026-11-18T03-12-30`, `03-48-09`)
  the baseline is that row, 706 and 707 µs, and `c32u4` is ×1.037 and ×1.039 faster, with the same words. No shipped
  model has a linear of that shape and storage. The row was left as it is.
- The derivation judges a candidate against the default profile's choice for the shape. Where that choice is an
  exception, another device drops it only when its records show the general variant ×1.10 faster in every group; a
  device on which the exception is slower by less keeps it (filed in the backlog).

## 8. Not measured

- Metal: no GEMV sweep record of an Apple device exists. `apple-metal-3` carries no GEMV choices, so it runs the
  default profile's 22 bf16 rows; whether they gain there, and the identity of their words there, are not known.
- A decode token on the RTX 5070 Ti, with either placement of the rows or with the choices its table carries since
  that day.
- The model lanes, beyond the two reference files of §4.
- A speculative decode: the verify run puts the head at several rows, a row block, which takes no exception; the
  draft steps run the drafter, of whose shapes `nvidia-ampere` now has one row (`256 × 2048`), not measured end to end.
- The second prompt of the bench, a sampled decode, longer generations.

## 9. Commands

```sh
# Per dispatch: the whole grid of sequential variants at one row (E2B's bf16 cases; about 5 minutes)
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemv-sweep --candidateSet=full --rows=1 \
  --cases=bf16-n8960-k1536,bf16-n2048-k1536,bf16-n256-k1536,bf16-n1536-k2048,bf16-n6144-k1536,bf16-n1536-k6144,bf16-n1536-k256,bf16-n4096-k1536,bf16-n512-k1536,bf16-n1536-k4096,bf16-n12288-k1536,bf16-n1536-k12288,bf16-n262144-k1536
# The same, narrowed to four variants (steady groups; about 3 minutes)
#   … --candidates=c16u8,c16u16,c32u8,c32u16

# What the records adopt, and the table
node tools/geometry-profiles/generate.ts nvidia-ampere --gemv-records=<record>,<record>,… --dry-run
node tools/geometry-profiles/generate.ts nvidia-ampere --gemv-records=<record>,<record>,…

# The identity
flock /tmp/karume-gpu.lock vp test packages/runtime/tests/gpu_linear_gemv_variant_test.ts
flock /tmp/karume-gpu.lock vp test packages/runtime/tests/gpu_corpus_test.ts

# A decode token with the device's choices, with the default profile's, and without the sequential exceptions
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemma4-bench \
  --arms=gemma4/e2b@bf16,gemma4/e2b@bf16:general-sequential --rounds=4 --chats=3 --newTokens=96 --prompts=few-ids
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemma4-bench \
  --arms=gemma4-qat/e4b@i4,gemma4-qat/e4b@i4:default-gemv --rounds=8 --chats=3 --newTokens=128 --prompts=few-ids
```
