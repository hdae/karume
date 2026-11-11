# @karume/candidate-rules

Private workspace package (not published): fusion rules that are **candidates**, not adopted rules.

A candidate is a fusion rule written outside the runtime with `defineRule` of `@karume/runtime/kernel`
(ADR 0148) while it is being measured and checked: its effect on a real run (the wall-clock session A/B of
`@karume/runtime/tune`) and the evidence its contract class asks for (for a class E rule, the
fused-against-unfused u32 check on each backend — docs/kernel-rules.md). No pipeline passes a candidate by
default; a measurement passes it explicitly (the gemma pipeline's experimental `fusionRules`).

- **Nothing here is part of a release.** The package is private, no published package depends on it at
  run time (`@karume/models` lists it as a devDependency for its tests, which JSR does not publish), and
  no default, preset or manifest names a rule from it.
- **A rule moves into `packages/models` when it is adopted**, together with its evidence (the A/B records
  and the identity checks it was adopted on). A rule that is dropped is removed from here, and the
  measurement that dropped it stays in `docs/research/`.
- **Every file imports from `@karume/runtime/kernel` and nothing else of the runtime** — the entry a
  package outside the runtime writes a rule against. No runtime-internal module, no load hook, no type
  assertion around the entry's types; `packages/models/tests/external_fusion_rule_test.ts` pins this for
  the rules below.

| Rule | Export | What it folds | Contract | Checked by |
| --- | --- | --- | --- | --- |
| `candidate/gelu_mul` | `GELU_MUL_RULE`, `GELU_MUL_RULE_NAME` (`src/gelu-mul.ts`) | `gelu_tanh(gate)` then the `mul` that reads it, adjacent or with one independent `linear` between them (Gemma 4 26B A4B's routed experts and dense MLP), in one dispatch | E | `packages/e2e/tests/e2e_external_fusion_rule_test.ts` (fused vs unfused on the GPU), `e2e_gemma4_26b_external_rule_test.ts` (planned on 26B's graph), `e2e_gemma4_session_ab_test.ts` (the session A/B on 26B, opt-in: token identity and hits) |
| `candidate/ew_chain` | `EW_CHAIN_RULE`, `EW_CHAIN_RULE_NAME` (`src/ew-chain.ts`) | The longest run of consecutive f32 elementwise nodes (the scalar-free unary and binary ops, at most 16 nodes and 7 external inputs, which may broadcast to the last node's shape) whose values but the last are read only inside it, in one dispatch that stages every internal value; it leaves the exact `sigmoid(x) → mul(x, ·)` pair to the built-in silu rule (the MOSS-Image DiT's FFN gate and its RoPE, gated-residual and adaLN tails) | E | `packages/e2e/tests/e2e_ew_chain_rule_test.ts` (fused vs unfused on the GPU, with a fault injection), `ew_chain_rule_plan_test.ts` (the windows it takes and declines, its keys and params, no GPU), `e2e_moss_fusion_counts_test.ts` (planned on the DiT pair) and `e2e_moss_session_ab_test.ts` (the session A/B on the DiT, opt-in: u32 identity of the output, hits and dispatches) |

Who imports it (every importer; keep this list complete when one is added):

- `packages/models/tests/external_fusion_rule_test.ts` — pins the import list above.
- `packages/models/tests/gemma4_estimate_test.ts` — the gemma pipeline's `fusionRules` reaches the target
  and the drafter Session.
- `packages/models/tests/gemma4_vision_pipeline_test.ts` — and not the image embedder's Session.
- `packages/e2e/tests/e2e_external_fusion_rule_test.ts`, `e2e_gemma4_26b_external_rule_test.ts`,
  `e2e_gemma4_session_ab_test.ts`, `e2e_ew_chain_rule_test.ts`, `ew_chain_rule_plan_test.ts`,
  `e2e_moss_fusion_counts_test.ts` and `e2e_moss_session_ab_test.ts` — the e2e tests in the table above.
- `apps/playground/src/models/session-ab/page.tsx` — the playground's `/session-ab` page, and
  `apps/playground/tests/session-ab_arm_test.ts`, the test of its arms.
