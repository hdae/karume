# 0148: Fusion rules from outside the runtime are the one form of custom kernel (`@karume/runtime/kernel`)

- Status: accepted (2026-11-11 — the maintainer approved the design of the three extension mechanisms on 2026-11-11; this
  record writes down its rulings on the custom-kernel part)
- Date: 2026-11-11
- Amends: [0008](0008-public-api.md) (a third sub-surface — addendum of 2026-11-11) / [0040](0040-fusion-pass.md) (the rule
  list a plan uses comes from the Session and may hold rules defined outside the runtime — addendum of 2026-11-11)
- Related: [0147](0147-numerics-and-kernel-adoption.md) (contract classes, decision 2; default-on fusions and the test-only
  switch, decision 5; per-GPU evidence and the Session canary, decision 9 — its "extension point for model-specific
  kernels (a separate ADR later)" is this record) / [0004](0004-execution-model.md) (buffer pool invariants) /
  [0007](0007-op-vocabulary.md) (the op allowlist) / [0059](0059-op-vocabulary-entry-doors.md) (where a computation lives) /
  [0105](0105-packed-static-quantize-activations.md) (packed int8 activations) / [0108](0108-container-format.md) (the
  container has no extension point) / [kernel-rules](../kernel-rules.md) (how to write and check a rule, checklist (d))
- Evidence: [research 2026-11-10](../research/2026-11-10-gemma4-26b-fast.md) §8.1 (the `gelu_tanh` → `mul` prototype rule
  on Gemma 4 26B A4B, which reached the rule list through a module load hook)

## Context

A model-specific kernel can enter the runtime today in two ways: a new IR op (`expert_linear` touched 44 files across the
runtime, the exporter and the op contract table shared with Python), or a fusion rule built into the runtime
(`adaln`, `gatedResidual`, `rmsNormRope`, `upsample2x`). A package or an application outside the runtime had neither. The
prototype of 2026-11-10 showed what that costs: to try one rule on 26B, it patched `fusion.ts` as it loaded, imported
runtime internals by absolute path and cast its way past the types.

Three facts of the code make an outside fusion rule a small change. A fused step (`FusedStep`) is plain data, and the
executor runs any step's dispatches through one generic path (`#buildFused` in
`packages/runtime/src/runtime/recipe-builder.ts`) with no code per rule. Since the rule-list work of 2026-11-11 the list a
plan uses is an input (`FusionContext.rules`), the counters come from that list (`zeroFusionCounts`), and `defineRule`
applies each rule's privacy declaration. The graph a rule folds stays made of built-in ops, so the unfused plan of the same
graph always exists and is the reference the rule is checked against.

## Decision

### 1. The only custom kernel is a fusion rule passed from outside

A rule written outside the runtime replaces a run of the graph's built-in ops — one node or several — with its own kernel
dispatches. That is the only form of custom kernel the runtime takes. **No IR op enters from outside**:

- without an unfused sequence there is nothing to check the kernel against mechanically (the CPU reference would have to come
  from outside too);
- a `krm` that needs an extension to run would depend on code outside the distribution, and the container defines no
  extension point (`capabilities.features` MUST be empty — ADR 0108);
- an op touches the exporter and the op contract table shared with Python (ADR 0007, 0059), far beyond the runtime.

If such an op is ever needed, its entry is `prepareContainer`, where an unknown op is first refused (recorded only).

### 2. One public entry, `@karume/runtime/kernel`, experimental

The package gains a third entry (`packages/runtime/kernel.ts`; `exports` and `files` of `package.json`, `exports` of
`deno.json`). It exports only what a rule is written with:

- `defineRule` — the external define entry (decision 5);
- the types a rule author names: `FusionRule`, `FusionRuleDefinition` (the external definition), `FusionMatch`,
  `FusionMatchContext`, `FusionLimits`, `FusedBuild`, `FusedStep`, `FusedDispatch`, `FusedOperand`, `FusedTemp`,
  `FusedWorkgroups`, `FusionHit`, `ContractClass`, `ExternalRuleName`, `NodePlan`, and those of the WGSL parts
  (`PrologueUnaryOp`, `FusedBinaryOp`, `StagedLoop`, `StagedValue`). The IR-layer types `NodePlan` refers to are not
  exported; they are reachable as its indexed types, and so is the runtime's own scan context in `FusionRule.apply`;
- **a matcher's context is `FusionMatchContext`**: the graph's use counts and outputs (what the declared privacy reads)
  and the device limits the pass carries (`FusionLimits`, so a matcher can decline a window whose dispatch would exceed
  one), frozen and built once per scan. It holds no Session knob, no rule list and no packed-activation decision: a knob
  in it would make every new planning knob a change of this public type, the rule list would hand a matcher the built-in
  rules (and their WGSL), and the packed values would let a matcher take a different window in the second planning scan
  than in the first (decision 6). The two maps are read-only by type; the runtime does not copy them, and the type is the
  guard — as it is for the node list a matcher reads (decision 9);
- the predicates `allF32` and `sameShape`;
- the WGSL parts that keep the unfused rounding: `prologueUnaryWgsl` and `fusedBinaryWgsl` (the plain elementwise kernels'
  own text), `SIGMOID_STABLE_WGSL`, `TANH_STABLE_WGSL` and `IS_NAN_BITS_WGSL` (which `tanh_stable` calls), and the
  rounding barrier `stagedLoopWgsl` / `stagedF32`;
- `CodegenError`. A dispatch declares its workgroups and the runtime resolves and checks the counts, so the entry exports
  no workgroup-count helper.

**While the package is 0.x the entry is experimental: anything it exports may change in a minor version.** Its header, the
runtime README and the CHANGELOG say so. It has no side effects and reads no WebGPU global at import
(`packages/runtime/tests/kernel_entry_test.ts` imports it in a Node process without WebGPU).

`SessionOptions.fusionRules?: readonly FusionRule[]` takes the rules. `mod.ts` gains no export: a Session user names
`FusionRule` from the kernel entry. `FusionCounts` keeps every built-in counter and gains optional keys
`<namespace>/<rule>` (`number | undefined`); a Session given a rule shows it in `lastRunFusions`, 0 when it did not fire.
That is a breaking change of a public type (code that walks every value of the record now sees `undefined`), stated in the
CHANGELOG.

### 3. External rules are tried before the built-in rules

A Session plans with its given rules first, in the order given, then `FUSION_RULES` (`sessionFusionRules` in
`packages/runtime/src/runtime/fusion.ts`). Where an external rule and a built-in rule could both take a node, the external
rule — the caller's choice — takes it. **An external rule is tried only at a node whose op is one of its `heads`, and its
folded chain starts at that node**, so its declaration says at which nodes it can pre-empt a built-in rule (its chain may go on to fold nodes a built-in rule would have led: the heads name where it is tried, not everything it takes). The test that the
built-in rules' heads are disjoint (but for the one declared pair) stays as it is and covers the built-in list only.
Which rule hit is in `lastRunFusions`.

### 4. The model-specific built-in rules stay in the runtime

`adaln`, `gatedResidual`, `rmsNormRope` and `upsample2x` stay where they are. New model-specific rules are written in
`@karume/models` (or the application) against the kernel entry. No pipeline passes a rule yet.

### 5. What `defineRule` collects and checks

- **Every rule declares** `name`, `heads` (a non-empty array of op names), `contract` (`"E" | "C" | "R" | "Q"`, ADR 0147
  decision 2), `internals` (`"private"` or `"stayInChain"`, which the entry applies to the folded chain), `match` and
  `build` (functions). Both define entries check these when they run (and refuse a definition that is not an object), so
  a malformed JavaScript definition fails there as an `ExecutionError` naming the rule, not as a `TypeError` at the first
  run. The external entry also refuses a head that is not an op of the runtime's vocabulary (the op contract table): such
  a rule would never be tried. Every built-in rule declares `"E"`, the
  class the classification table of ADR 0147 decision 2 gives the fusion rules.
- **The definition is read once.** The entry copies each field when it runs and the rule reads only that frozen copy, so
  editing the caller's object afterwards (deriving a variant from a shared definition) changes nothing the rule does —
  not its name, its counter, its privacy, its matcher or its build. `match` and `build` are called as standalone functions,
  `this` undefined, so a method that reads `this` fails loudly at its first try instead of reading the frozen copy (in strict
  code, which every ES module is).
- **Only the return value of a define entry is a rule a Session takes.** The entry freezes the rule and brands it with a
  non-enumerable property keyed by a module-private symbol, recording whether it was defined inside the runtime or outside
  it. A Session refuses an entry of `fusionRules` without the external brand: a hand-written object, a copy of a rule (the
  brand does not travel with a spread) or a built-in rule. The brand is a value on the rule, not an entry in a module-level
  table, so the runtime keeps no global state. Because the symbol is private to the module, **a rule is bound to the copy of
  `@karume/runtime` that defined it**: a dependency tree with two copies (an application's pin beside the range
  `@karume/models` asks for, or a bundler that keeps symlinked copies apart) gets the rules of one refused by Sessions of the
  other. The refusal names both causes — an object `defineRule` did not make, or a rule from another copy of the runtime —
  and the remedy is one copy. The brand guards against accidents, not against a caller who reads the symbol off a rule on
  purpose and forges it: such a rule skips every check of an external rule, a documented limitation (no module-level
  registry replaces the brand, because the package keeps no global mutable state).
- **The external entry** (`defineRule` of the kernel entry; `defineExternalRule` inside) takes a name `<namespace>/<rule>`
  — two lowercase identifiers `[a-z][a-z0-9_]*` and one slash; built-in names have none, so the two never meet. The
  runtime's own entry keeps accepting the built-in names and is not on the kernel entry.
- **Each time an external rule builds a step** it meets the invariants the built-in rules keep by construction, and an
  `ExecutionError` names the rule otherwise:
  - its window is the run of nodes that starts where it was tried and holds at least one folded node (a window elsewhere
    would drop or repeat a node silently), its chain starts at the window's first node (no leading passthrough, so the op
    checked against `heads` is the chain's first op), and its chain follows the window's node order (the step's inputs and
    output are derived from the chain in that order);
  - it binds only distinct external inputs of its folded chain (the release bookkeeping rests on those inputs alone, so a
    bind outside them could be released and its buffer reused before the step reads it);
  - its output is the chain's last value with its shape, it declares at least one dispatch, and at least one dispatch binds
    the output (`operands` omitted, or holding the output);
  - no operand is a `weightScale` (an external kernel reads f32 values only, and a resident scale belongs to a weight
    stored in another form);
  - every dispatch key starts with `ext:<namespace>:` (the pipeline cache is one per device and shared by every Session).
    The keys are checked here, when a step is planned — in the first run that plans the binding — not at Session
    construction as the design of 2026-11-11 had it: a key exists only once a matcher has grabbed a window and the build
    has run, which needs the bindings.
- **A window is declined** for an external rule (decision 9) when its chain is not all f32 (decision 7), its output has
  no element (the binding sized it, and the recipe refuses a dispatch of zero groups), or its step binds a weight not
  resident as f32 — stored as f16 / bf16 / i8 / i4 / i2, borrowed from another Session, or held in the expert cache's
  pool (the weight's IR dtype is f32 whatever its storage, so `allF32` cannot see it).
- **When the step's recipe is built** (the WGSL is known once its pipeline is fetched), an external dispatch's storage
  roles and workgroup counts are checked, again failing loudly and naming the rule: every binding its WGSL declares
  `read_write` is the step's output or one of its temporaries (an input, a resident weight or scale, or a params buffer
  bound `read_write` is refused — each is a buffer the runtime shares beyond the step), at least one dispatch declares
  the output's binding `read_write` (binding is not writing: a kernel that only reads the output leaves the pool's stale
  data), and its counts, as the runtime resolves them, are three positive integers within the device limit (zero groups
  would leave the output with a reused buffer's data).
- **Any other failure** of the rule's code, or of the runtime reading what it returned — a `TypeError` on a result of the
  wrong shape, a `DispatchLimitError` of the count resolver — is rethrown as an `ExecutionError` that names the rule and
  keeps the original as `cause` (one wrapper at each call site, not a schema of every field).
- **Everything else is the built-in rules' checks**: a window with a multi-output node or a node that touches state is not
  taken, the chain is a subsequence of the window, the passthroughs and deferred nodes are independent of the chain, and
  the temporaries' lifetimes close inside the step.

### 6. The safety measures

| Concern | Measure |
| --- | --- |
| A pipeline key collides with a built-in kernel's or another extension's | The name `<namespace>/<rule>` and the key prefix `ext:<namespace>:` (decision 5). The last line of defence stays the pipeline cache's `PipelineKeyConflictError` (two WGSL bodies under one key), rethrown naming the rule with it as the cause |
| Registration at import, global state | None. The rules are a value of the Session options, fixed at construction (the prepared-plan key stays the bindings alone); the brand is on the value |
| Contention with the built-in rules | External first, in the order given (decision 3); the hit shows in `lastRunFusions` |
| Packed int8 words read as f32 | No fused step other than `linearStaticQuantize` may consume a packed value (`packedStaticQuantizeValues`, condition 5 — ADR 0105). The pairing is decided on the first planning scan, and an external matcher sees no packed value. On the second scan an external step that folds a node producing or reading a packed value fails loudly, naming the rule; after it the pairing is recomputed from the final steps, and a plan whose pairing changed between the scans (a matcher that is not a pure function can let a built-in rule take a paired value — the linear→SRQ fusion then writes f32 words a packed reader takes as codes) fails loudly, naming the external rules of the list |
| A rule that changes the numbers | `contract` is required. Class E claims u32 identity with the unfused plan; another class follows ADR 0147 (a named numerics opt-in with its evidence). Passing a rule is itself the opt-in: the runtime does not check its numbers |
| The shader compiler differs per device | The two layers of ADR 0147 decision 9. Before shipping, the rule's author runs a fused-against-unfused u32 check on each backend the rule ships on (the built-in rules' `/kernel-check` cases do not cover it). The Session canary, when it is built (ADR 0147 decision 9, backlog), covers external rules too |
| Buffers, bind groups, limits | The runtime's existing path: a rule supplies a match, WGSL, params and workgroup counts, and never allocates; the executor's generic fused-step path builds and releases everything (ADR 0004). An external kernel writes its output and nothing but its output and its temporaries, and its counts are checked against the limit (decision 5) |

### 7. What a rule cannot do

A rule cannot fold a node with several outputs or a node that touches state, cannot hand packed int8 words to another step,
and cannot add an op: those windows are not taken, and the restriction is not changed here. Every fused kernel of the
runtime is f32, and **an external rule folds only chains that are all f32**: the runtime checks `allF32` on the chain an
external matcher grabbed and does not take the window otherwise (the unfused plan is always correct). The check is not left
to the matcher, because a matcher that forgot it would fold an i32 or bool chain into an f32 kernel silently. The built-in
rules keep the dtype checks in their matchers (some read a bool condition — `isF32Where`). For the same reason a step that
binds a weight not resident as f32 is not taken (decision 5). This settles the question the
first version of this record left open (the design of 2026-11-11 listed non-f32 outputs among what `defineRule`
refuses).

### 8. The memory estimate is not changed

Replaced by the addendum of 2026-11-11 "the estimate plans with the rules" below; the text is kept as the state this
record was accepted in.

`estimate` plans the graph as if no rule were given, so a rule's step temporaries are not counted, and neither are the
values a built-in fold would have removed when an external rule takes its nodes first (the estimate counts row-block
attention, a linear's folded producer and an absorbed permute in their post-fusion form, so displacing one can make the
undercount larger than the rule's own temporaries — up to the full score matrix of an attention). Generalising the estimate
to every fused step, external ones included, is later work (the design's phase P4); no estimate option is added for the
rules. `SessionOptions.fusionRules` says so.

### 9. Decline or throw, and matchers are pure

The promise: a rule written outside the runtime cannot, by accident, make a Session silently compute wrong values or
corrupt buffers the runtime owns. A deliberately hostile caller is out of scope.

- **Decline** — the window stays unfused, its values stay correct, the rule's counter does not move — when the condition
  depends on the graph or the binding and the runtime can evaluate it without the rule being wrong: the node at the index
  is not one of the rule's heads; the chain is not all f32; the output has zero elements; a bind is a weight that is not
  resident as f32.
- **Throw an `ExecutionError` naming the rule** when what the rule returned is wrong whatever the data: a malformed match
  or build result, a window or chain that breaks the window rules, a bind outside the chain's inputs, a wrong output, a key
  outside its prefix, a kernel that writes what it must not or never writes its output, invalid workgroup counts, an
  operand an external step may not use (decision 5). Any other failure of the rule's code or of resolving its step is
  rethrown the same way with the original as the cause — except an out-of-memory error and a lost device, which are
  conditions of the device and pass through as they are (callers recover from them by class).
- **`match` and `build` are pure functions of their arguments (MUST)**, written as standalone functions. The pass scans a
  plan more than once (the packed pairing) and caches plans, so a matcher whose answer depends on anything else can make
  the scans disagree. Where that would break the packed pairing the plan fails loudly (decision 6); anywhere else it is
  the rule's bug, not one the runtime detects. The kernel entry's `defineRule` and `SessionOptions.fusionRules` say so.
- **The read-only types are the guard** of the node list a matcher reads (the scan's own list and its `NodePlan`s) and of
  the Session's maps in its context: the runtime does not copy or freeze them at run time. A matcher that writes them is a
  hostile caller in effect, outside the promise; freezing them is recorded, not done.

## Evidence (2026-11-11)

- The 2026-11-10 prototype, rewritten against the kernel entry alone (`packages/models/tests/helpers/gelu-mul-rule.ts`, the
  rule `test/gelu_mul`, class E — moved on 2026-11-11 to `packages/candidate-rules/src/gelu-mul.ts` as
  `candidate/gelu_mul`, see the addendum; `packages/models/tests/external_fusion_rule_test.ts` pins its only import to
  `@karume/runtime/kernel` and finds no type assertion in its code). The gelu value is staged by `stagedLoopWgsl` instead
  of the prototype's run-time XOR mask.
- Planned on Gemma 4 26B A4B's decode graph of the local mirror through the Session's own list composition, at M = 1 and
  M = 32: 60 hits per plan, 30 over the dense MLP's up projection and 30 adjacent in the routed experts, the prototype's
  count; every other counter as without the rule (`packages/e2e/tests/e2e_gemma4_26b_external_rule_test.ts`).
- On the GPU (Vulkan / RTX 3080 Ti): u32-identical to the same Session options with the rule disabled by name, in the
  three windows it takes, with the counter at 1 fused and 0 plain (`packages/e2e/tests/e2e_external_fusion_rule_test.ts`,
  which compares the words with `wordIdentity`; it sits in the e2e package because the runtime's tests import nothing from
  the models package). Multiplying the rule's product by 1.0001 turns it red at 478 of 771 words. The runtime keeps a twin
  of its own through `checkFusionTwin`, a `gelu` → `mul` rule defined in its test tree
  (`packages/runtime/tests/gpu_external_fusion_rule_test.ts`).
- Not produced: token identity of the rule on the real 26B weights. This machine's GPU (12 GB) cannot hold 26B, the 26B
  decode cases run through the gemma pipeline, which passes no rule (decision 4), and the only runtime-level 26B Session,
  the 12-layer cut's teacher-forced run (`KARUME_SERIES=1`), would need a second upload of the cut for one comparison.

## Addendum (2026-11-11) — a measurement opt-in on the gemma pipeline, and where candidate rules live

- **What changed**: `gemma.PipelineOptions.fusionRules` (experimental; the mobile QAT pipeline shares the options),
  passed unchanged as `SessionOptions.fusionRules` to the target Session (prefill and decode) and, with `speculative`,
  the drafter Session; the image embedder, the on-GPU PLE gather and the on-GPU greedy selection do not get it. The
  maintainer's ruling for this step: add it to gemma only, as an experiment, and decide the details later — so no other
  family, no manifest vocabulary and no preset name it, and it may change or go away in a minor version. Decision 4's
  "no pipeline passes a rule yet" still holds for every default: a pipeline passes a rule only when its caller gives
  one. The rule's author owns its evidence, and the memory estimate does not see it (as for `SessionOptions`).
- **Candidate rules**: rules that are measured and not adopted live in the private workspace package
  `packages/candidate-rules` (depends on `@karume/runtime` only; every file imports `@karume/runtime/kernel` and
  nothing else of the runtime, which `packages/models/tests/external_fusion_rule_test.ts` checks over the whole
  package). A rule moves into `packages/models` with its evidence when it is adopted; nothing there is part of a
  release. The prototype rule moved there and was renamed `test/gelu_mul` → `candidate/gelu_mul` (keys under
  `ext:candidate:`), since a page and a record now show it.
- **The evidence item "not produced" above** is now an opt-in check: `packages/e2e/tests/e2e_gemma4_session_ab_test.ts`
  (`KARUME_SESSION_AB=<dir>`) runs the session A/B of [0117](0117-app-geometry-tuning.md) (addendum of 2026-11-11) on
  Gemma 4 26B A4B under the expert cache, one pipeline per visit, and asserts token identity over every run of both
  arms and the rule's hits in the decode plan; the record goes to the directory, not to the docs. The arms are built
  without the diagnostics callback (its presence alone changes the gemma decode path), so the hits come from a count
  probe of each arm before the A/B (one build with the callback, one generation, one dispose). The playground's
  `/session-ab` page runs the same A/B in a browser.
- **Observed while building it**: on a device the caller keeps, a disposed pipeline's buffers are freed by Dawn only
  once a submit made after the destroy has completed (Dawn on Vulkan), so a second 26B build right after a dispose ran
  out of memory before its first submit. A disposal now ends with that release wait in the runtime
  ([0095](0095-plan-backing-budget.md), addendum of 2026-11-11), so the A/B arms only await the dispose.

## Addendum (2026-11-11) — the estimate plans with the rules (decision 8 replaced)

The later work decision 8 names is done (ADR [0093](0093-transient-liveness-packing.md) addendum 2026-11-11).

- `EstimateOptions` takes the fusion options of `SessionOptions`, `fusionRules` among them, and plans fusion as a
  Session built with the same options does. A fused step of an external rule is counted like a built-in one: its
  declared temporaries, its single output, the roles of its bindings from its WGSL.
- With `fusionRules` the estimate requires `maxStorageBufferBindingSize` and `maxComputeWorkgroupsPerDimension`, the
  limits a matcher may read; without them it fails loudly instead of planning under limits no device has.
- The estimate runs the checks on what an external dispatch writes and on the output being written (decision 5), so it
  returns no number for a step execution refuses. It does not run the checks recipe derivation makes on the dispatch itself
  (the workgroup counts, the params); such a step fails at the first run's plan derivation. Two WGSL texts under
  one pipeline key are refused by the estimate as by the pipeline cache.
- `gemma.Pipeline.estimateSessionMemory` passes the target and drafter estimates the fusion options their Sessions were
  built with, from one derivation. Only the target's scenarios enter its report (the drafter contributes its weights
  and state), so the rules move the reported workspace through the target, and decide whether the drafter's estimate
  refuses. The sentence "the memory estimate does not see it" of the addendum above and of the Consequences of ADR 0040's
  addendum no longer holds.
- A caller that gives a Session rules and the estimate none gets the count of the plan without them; `unaccounted`
  says so.

## Alternatives considered

1. **An IR op from outside** (a caller-supplied op contract and kernel) — rejected for this phase: decision 1.
2. **A registry of rules filled at import** — rejected: zero side effects in every module (MUST) and no global mutable
   state; the rule list stays a value.
3. **External rules after the built-in rules** — rejected: a caller who passes a rule for a form a built-in rule also takes
   wants their rule; the order is observable in `lastRunFusions` either way.
4. **Accept any object of the `FusionRule` shape** — rejected: a hand-written `apply` would bypass every window check of
   `defineRule` (decision 5).
5. **Move the model-specific built-in rules to `@karume/models` now** — rejected: decision 4.
6. **Count external rules in a separate diagnostic record** instead of optional keys of `FusionCounts` — rejected: one
   record keeps one place to look and one shape for the fused-against-unfused check.

## Consequences

- A package or an application can try a fusion on its own model without editing the runtime, and the same container runs
  without it.
- The kernel entry is a new public surface, experimental while 0.x; the surface snapshot pins its names and kinds.
- `FusionCounts` is a breaking change of a public type (CHANGELOG).
- The evidence for an external rule's numbers is its author's: the runtime does not run its checks and `/kernel-check` does
  not cover it.
- What a fused step may bind and write, its window, its chain's start, order and dtypes, and its workgroup counts are now
  checked for external rules, not only by construction; a condition of the graph or the binding declines the window, and a
  wrong result of the rule fails loudly, naming it (decision 9).
