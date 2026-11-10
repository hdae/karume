# 0104: Declare the accepted fusions as a fast quant preset for Gemma E2B

- Status: accepted (2026-09-15, the maintainer's instruction to continue making it the default); partly
  superseded by [0109](0109-manifest-v5-container.md) (manifest format, 2026-09-22), by the addendum of
  2026-09-26 and [0111](0111-session-options-composition.md) (fields accepted from `quant.session`,
  2026-09-26), by [0128](0128-default-preset-original-weights.md) decision 1 (regular E4B default,
  2026-10-17), and by the 0128 addenda of 2026-11-04 (regular E2B / E4B default `bf16`; then the
  regular line ships `bf16` alone, so its `i4` / `i4-gemvpar` / `i4-fast` are gone — the same three
  declarations live on in the q4_0 QAT line `karume-gemma4-qat` (E2B / E4B) and the mobile QAT line
  `karume-gemma4-qat-mobile`). Automated acceptance: [integration acceptance](../research/2026-09-15-gemma-fast-quant.md);
  broad quality, long texts and E4B were not accepted at the time.
- Related: [0038](0038-manifest-v1.md), [0058](0058-numerics-opt-in-contract.md),
  [0098](0098-linear-gemv-parallel.md), [0099](0099-rms-norm-add-fusion.md),
  [0103](0103-linear-static-quantize-fusion.md); M2 evidence: [RMS fusion](../research/2026-09-13-rms-subgroup-reduction.md)
  (the maintainer's M2 results), [linear→SRQ](../research/2026-09-15-m2-linear-srq-adoption.md).

## The definition that changes

Regular Gemma 4 E2B and fixed mobile QAT E2B get a new preset `i4-fast`, which becomes `defaultQuant`.
The reference `i4` and `i4-gemvpar` (parallel GEMV only) keep their meaning. All three map the same
weights; no requantization or weight copy is needed.

| E2B     | `session` declaration of `i4-fast`                                                                                                                                                                                                       |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Regular | `linearGemvReduce: parallel`, `fuseRmsNormAdd: true` (+ `stateAttentionReduce` since the addendum of 2026-09-26)                                                                                                                         |
| QAT     | the above + `fuseLinearStaticQuantize: true` + `packedStaticQuantize: true` ([0105 addendum 2](0105-packed-static-quantize-activations.md#addendum-2-2026-09-20--promotion-to-the-manifest-vocabulary-the-i4-fast-declaration-and-the-rejection-of-stage-1b), 2026-09-20) |

E4B, outside the accepted scope, keeps only `i4`. (Superseded by ADR 0128 decision 1 on 2026-10-17: a
regular E4B distribution gets its default from the per-model procedure when one ships — no regular E4B
mirror exists, and `tools/export-recipes/gemma4/distribution.py` marks its `i4` as provisional; QAT E4B
got `i4-fast` in the addendum of 2026-09-26.)

The runtime default, the reference WGSL / golden and the published pins do not change. The default
changes when a distribution is rebuilt from the recipe; existing distributions are not rewritten.

## Stored vocabulary and compatibility

The hub's `SessionSpec` gains two optional booleans, `fuseRmsNormAdd` and `fuseLinearStaticQuantize`
(`packedStaticQuantize` was added the same way by 0105 addendum 2). Only true / false are accepted;
null, numbers and strings are rejected. false is not folded into an omission; the common mapping in
`@karume/models` forwards it explicitly.

The manifest stays `karume/4`: a new reader reads an old manifest with the same meaning, an old reader
rejects the new fields as unknown keys, and a new distribution is not treated as readable by the
published 0.12.0. (Superseded by ADR 0109 decision 2 on 2026-09-22: the manifest is `karume/5`; the
`quants[].session` fields carry over unchanged.)

Gemma accepts only `linearGemvReduce` and the fusion fields (`fuseRmsNormAdd` /
`fuseLinearStaticQuantize` / `packedStaticQuantize`) from `quant.session`. (Superseded by the addendum
of 2026-09-26 and ADR 0111 on 2026-09-26: `stateAttentionReduce` was added; `GEMMA_SESSION_POLICY` in
`packages/models/src/gemma/session-options.ts` accepts five fields; the composition rule is ADR 0111's.)

Admission, before the weights are fetched, checks the accepted fields and the effective settings. Each
key is chosen in the order explicit option → quant declaration → runtime reference default and passed
to both target and drafter. An unsupported declaration is not hidden by an override, and an invalid
explicit value such as null does not fall back to the quant's value.

SRQ fusion with an effective `linearGemvReduce` other than `parallel` is rejected — for an explicit
`sequential` or `parallel-subgroup32`, and also when nothing declares it (`i4` with only the fusion set
to true). To compare such a setting, also pass `fuseLinearStaticQuantize: false` and
`packedStaticQuantize: false`, or select `i4` / `i4-gemvpar` (packed has the same rejection —
`packages/runtime/src/runtime/session-options-check.ts`). A correctly typed false always wins over the
quant's true. `fromAssets` has no quant selection and uses only explicit options, as before.

## Host-side policy stays separate

The input limit and the prefill buckets are host policy and are not stored in `quant.session`. No
fusion or parallelism is added from the model or GPU name; the default is selected by the quant
declaration alone. The speed of the fusion alone is compared at the input limit of 1024, separately
from the gain of combining it with 768. This change does not switch the model's input policy or buckets.

## Acceptance items

`accepted` approves the decision, not the items (vocabulary, recipe, real-model runs, target / drafter
propagation, Chrome comparison, verify, pytest); their results are in the integration acceptance research.

## Addendum (2026-09-25) — the manifest after the container

The `quants[].session` fields carry over into `karume/5` ([0109](0109-manifest-v5-container.md) decision 2);
weight shards are container parts since [0108](0108-container-format.md) (decisions 12 / 13, [container-v1](../container-v1.md) §6).

## Addendum (2026-09-26) — `stateAttentionReduce` joins the manifest vocabulary; the models family default is removed

The maintainer's ruling of 2026-09-26. `stateAttentionReduce` (`"sequential"` / `"parallel"` /
`"parallel-fused"`, the runtime's values) joins the hub's `SessionSpec` under the same composition rule
as the other quant execution options (explicit > quant declaration > runtime default — ADR 0111).
Gemma's acceptance table has true; every other family's has false.

**Removed**: the code default `GEMMA4_STATE_ATTENTION_REDUCE = "parallel"` in `@karume/models`, injected
into every preset and `fromAssets`. It kept an `i4` with an empty `session` and `fromAssets` off the
runtime reference path, contradicting [ADR 0110](0110-practical-tier-numerics-contract.md) decision 1
(the practical bundle lives only in the manifest's presets). The effective value now comes from the
composition alone.

**Recipe declarations, regular distribution** (the former effective value kept as a declaration):

| Regular E2B / E4B | Effective before (attention) | Declaration (`session`)                                                                      |
| ----------------- | ---------------------------- | -------------------------------------------------------------------------------------------- |
| `i4`              | `parallel` (family default)  | empty = `sequential` (**behaviour change**)                                                  |
| `i4-gemvpar`      | `parallel` (family default)  | `linearGemvReduce: parallel`, `stateAttentionReduce: parallel`                               |
| `i4-fast`         | `parallel` (family default)  | `linearGemvReduce: parallel`, `fuseRmsNormAdd: true`, `stateAttentionReduce: parallel-fused` |

- Regular E2B `i4-fast` takes `parallel-fused` from a same-day measurement (Intel Arc B570, the former
  `tools/flag-bench`, removed by bb35fe2a): all three options cut GPU decode ms/step by 47.9% against
  `i4`, token sequences identical (`outputs/bench/flag-bench/2026-09-26/`). `parallel-fused` keeps the
  addition order of `parallel` ([ADR 0102](0102-state-attention-stats-pv-fusion.md)).
- Regular E4B has the same presets and declarations, default `i4`, and a description that says speed is
  not measured; bundle and default were to follow an E4B benchmark. (Superseded by ADR 0128 decision 1
  on 2026-10-17: decided by the per-model procedure when a regular E4B distribution ships. Decided on
  2026-11-04: regular E2B and E4B both default to `bf16` — ADR 0128 addendum 2026-11-04.)
- `fromAssets` runs `sequential` without an explicit option (formerly `parallel`). Distributions built
  before this change (including the published pins) run every preset `sequential` until rebuilt; an
  explicit `stateAttentionReduce` restores the old behaviour.
- **Compatibility**: the 0.13.x hub rejects unknown `session` keys, so a distribution declaring
  `stateAttentionReduce` needs a hub / models that know it (Breaking in the CHANGELOG). Published
  distributions and pins are re-uploaded with the next release ([ADR 0073](0073-models-source-pin.md)).

**QAT bundle** (same day; ruling: "benchmark each flag alone; if none slows down, take all; of the two
attention values, take the faster in the all-in bundle"). On the Intel Arc B570 against `i4`
(`outputs/bench/flag-bench/2026-09-26/qat-*.stdout.json`) no flag slowed down, and the all-in bundle with
`parallel-fused` cut GPU decode ms/step by 49.3% (QAT E2B) and 41.7% (QAT E4B). The recipe declares:

| QAT E2B / E4B       | Declaration (`session`)                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `i4`                | empty = reference path (no longer the QAT E4B default)                                                                                                       |
| `i4-gemvpar`        | `linearGemvReduce: parallel`, `stateAttentionReduce: parallel` (new for E4B)                                                                                 |
| `i4-fast` (default) | `linearGemvReduce: parallel`, `fuseRmsNormAdd: true`, `fuseLinearStaticQuantize: true`, `packedStaticQuantize: true`, `stateAttentionReduce: parallel-fused` |

QAT E4B was outside the original scope, but on this measurement it gets the same presets with `i4-fast`
as default (ruling: "build `i4-fast` for E4B too and decide by measurement"). Only its description carries
a measured figure, with the device: "E4B decode time -42% (B570)" (Intel Arc B570, 2026-09-26). ADR 0128
decision 1 keeps this default as an instance of its procedure.

## Addendum (2026-11-06) — 12B gets `i4-gemvpar` and `i4-fast`; its default stays `i4`

`karume-gemma4-qat` 12B now has the same three presets as E2B and E4B over one set of weights. The reason it had one, no
GEMV rows and a width outside the fusion, is gone (ADR 0098 and 0099 addenda of 2026-11-06).

- **Default**: `i4`, unchanged. ADR 0128 decision 1 applies to a change of default, which nobody has reviewed for 12B.
- **Speed** on the RTX 3080 Ti under Node + Dawn, median ms/token of a warm chat
  ([research 2026-11-06](../research/2026-11-06-gemma4-12b-fast.md)): 64.9 → 22.3 at P ≈ 1K, 121 → 36 at P ≈ 16K,
  181 → 50 at P ≈ 32K.
- **With the drafter `i4-fast` is slower than `i4`**: 38.2 against 32.6 on one prompt, 25.9 against 19.1 on the other.
  On E2B and E4B it is faster with the drafter. The cause is not known (perf-ledger K-102), so 12B's description says
  "except when speculative".
- **Not on Hugging Face yet.** The published `hdae/karume-gemma4-qat` still has `i4` alone for 12B; the local mirror has
  three. The manifest goes up and `SOURCES` is re-pinned at the next release.

## Addendum (2026-11-10) — 26B gets `i4-gemvpar` and `i4-fast`; its default stays `i4`

`karume-gemma4-qat` 26B A4B now declares the same three presets as the other models of the line. The reason it had one, no
GEMV rows and a width outside the fusion, is gone (ADR 0098 and 0099 addenda of 2026-11-10).

- **Default**: `i4`, unchanged, for the same reason as 12B (ADR 0128 decision 1).
- **What changes**: the dense linears, the attention and the RMS→add pairs. The routed experts (`expert_linear`) keep
  their sequential order in every preset, so the descriptions' "parallel GEMV" names the dense part only.
- **Speed** on the RTX 3080 Ti under Node + Dawn ([research 2026-11-10](../research/2026-11-10-gemma4-26b-fast.md)): one
  decode forward 24.6–25.7 → 17.5–17.9 ms. The card cannot hold 26B resident, so the measurement ran under the exact
  expert cache (ADR 0146); 26B has no drafter, so there is no speculative arm.
- **With the expert cache** the presets apply as they are: a run of the same row count takes the same kernels with
  every expert resident and with the cache, which is the cache's reference of exactness (ADR 0146 decision 7).
- **Not in the mirror or on Hugging Face yet.** The recipe declares the presets; the local mirror's manifest and the
  published `hdae/karume-gemma4-qat` still have `i4` alone for 26B.
