# 0056: Vocabulary extension: gru_scan / gru_scan_reverse (the hidden-side GRU scan, an extended composite op)

- Status: accepted (2026-08-14); on 2026-10-18 the low-precision ineligibility of `w_hh` (decision 1),
  the `H ≤ 256` geometry (decision 5) and the single output without `h_n` (decision 7) were re-graded
  to MAY. The layer numbers used here were replaced by the entry doors of
  [0059](0059-op-vocabulary-entry-doors.md) on 2026-08-14 ("layer 2" = extended composite layer for an
  op outside Core ATen).
- Date: 2026-08-14
- Scope: the new ops `gru_scan` / `gru_scan_reverse` (the 56th and 57th contracts), the kernel
  `packages/runtime/src/kernels/gru-scan.ts`, the custom ops `tools/exporter/src/karume/custom_ops.py`
  and one exporter handler.
- Related: [0043](0043-op-addition-layers.md) (layers of the time) / [0023](0023-fused-attention.md)
  (bit identity at the core; decision 4) / [0055](0055-deform-conv2d.md) (the previous op) /
  [0046](0046-cat-symbolic-axis.md) (only `cat` got a symbolic axis) / [0014](0014-layout-ops-full-write.md)
  / [0022](0022-gemm-register-blocking.md) (no runtime autotuning) / [0024](0024-conv2d-implicit-gemm.md).
- Demand: in vowel-detector (`Crnn`: two convs → **two-layer BiGRU** → linear), `run_decompositions`
  **fully unrolls** `aten.gru.input` along time: 874 / 1,714 / 8,434 nodes for T10 = 20 / 40 / 200
  (linear in T), and `torch.export` specialises the length axis, so **every length needs its own
  graph**. With this op the graph has **19 nodes independent of T**. Kokoro-82M's six bidirectional
  LSTMs would be the next user, but **LSTM is not added here** (decision 8).

## Context

`aten.gru.input` is in the Core ATen decomposition table and `curated_decompositions()` unrolls it T
times in a Python loop. The unrolled graph is **written exactly with existing ops** (`linear` /
`slice` / `sigmoid` / `tanh` / `mul` / `sub` / `add` / `cat`), so GRU is a **molecule**, not an atom:
the test of ADR 0043 addendum decision 1 ("can it be written exactly as a composition?") places it in
neither layer 1 nor layer 1′.

It enters the vocabulary on two of the layer-2 entry conditions of `docs/op-vocabulary.md`:

- **(1) Semantic information is lost when restoring from the decomposition** (applies). Restoring "this
  is a GRU" with a layer-3 matcher would need an exact match of a node sequence that differs per T. The
  lost information is that **the time axis is decided at run time**; restoring it opens a capability,
  as ADR 0046 did for exporting the Irodori DiT.
- **(3) Recurrence justifies the permanent commitment** (applies). Besides vowel-detector's BiGRU, the
  hidden-side scan of RNN models has only this one form.

(2) The capacity of intermediates applies partly (8,434 nodes at T = 200 cost transients and
dispatches, but that is not the main reason). (4) Attr variants do not apply. `aten.gru.input` is
outside Core ATen, but **the layer test is not "does it come from Core ATen"** (ADR 0043 addendum
decision 1).

## Decision

### 1. The input-side GEMM is not part of the op — the contract is the hidden-side scan only (option S)

torch's decomposition already splits into an **input-side linear (all time steps at once, output
`[T,N,3H]`) and a hidden-side linear (per step, output `[N,3H]`)** (measured). That cut is the
contract boundary:

`gru_scan` / `gru_scan_reverse`, **fixed arity 4**, uniform f32, **empty attrs**.

| Tensor | Contract                                           |
| ------ | -------------------------------------------------- |
| `gi`   | `[T, N, 3H]` (the input side `x·W_ihᵀ + b_ih`)     |
| `h0`   | `[N, H]`                                           |
| `w_hh` | `[3H, H]`                                          |
| `b_hh` | `[3H]`                                             |
| output | `[T, N, H]` (`h` of every step)                    |

The gate order is **r / z / n** (torch's `weight_hh_l{k}` verbatim).

Including the input side (an op `gru` with arity 6, option G) is rejected on four counts:

- **Contract size**: option S has arity 4 and empty attrs, and "the input side is not in the
  vocabulary" shows in its structure (ADR 0023 decision 4). Option G would gain a second slot axis with
  `has_biases=False`, needing variable arity or a synthesised bias.
- **Existing linear optimisations apply**: in option S the input side is a plain `linear` node in
  `WEIGHT_SLOTS`, so f16 / i8 storage and `linearCompute: "i8a8"` work as is. vowel-detector's `w_ih`
  (`[384,160]`×2 + `[384,256]`×2) is most of its weights; in option G it would become an op-internal slot.
- **Half the bit-identity proof**: the input-side linear is the same node as in the decomposition, so
  its bits match trivially; only the hidden-side recurrence needs proof (decision 3).
- **LSTM extends naturally**: 4 gates and a cell state change only the op's internal formula; the input
  side stays the same `linear` (`[T,N,4H]`).

The op is named `gru_scan`, not `gru` (a permanent public commitment burnt into IR files): the name
says it is the hidden-side scan of a GRU, not the whole GRU.

- `w_hh` is **not enabled** for low-precision storage (not in `WEIGHT_SLOTS` / `WEIGHT_CHANNEL_AXES`);
  adding it MAY be done when there is demand. The input-side weights belong to the caller's `linear`,
  which is where the distribution size shrinks. (Re-graded 2026-10-18 to MAY: the reason was the
  silent i8 scale-axis mix-up of op-internal slots recorded in ADR 0024 decision 6, and the key match
  of the two tables is now checked (`packages/runtime/src/ops/names.ts`, `WEIGHT_CHANNEL_AXES`);
  ruling: accepted as proposed.)

### 2. The reverse direction is a **separate op name** (`gru_scan_reverse`), not an attr

Of the three ways to express the reverse direction of a bidirectional GRU, the third is taken:

- **Wrapping with `flip` does not work.** torch's decomposition flips `linear(x)` along `[0]` before
  the recurrence and reverses the `cat` order on output (no output flip). Karume's `flip` **rejects a
  symbolic axis on both sides** (Python: `_flip` → `_static_axis` in `shapes.py`; TS:
  `assertStaticLayoutAxis` in `plan.ts`), and ADR 0046 keeps `slice` / `flip` static; overturning that
  is a separate ADR. `flip` would also add a materialised `[T,N,3H]` copy (307 KB × 4 at T = 200,
  H = 128). → **The op must know the scan direction.**
- **A bool attr (`reverse: true`) is rejected.** Existing attrs are int / int list / f32 scalar /
  string; **there is no bool precedent**, and Python's `_assert_integer_attr` (`ops.py`) explicitly
  rejects bool ("Python's bool derives from int, so `True` would pass as 1"). Accepting it would mean
  new bool validators on both sides (an extension of the contract-check machinery).
- **Taken: "an attr variant is a separate op", as `gelu` / `gelu_tanh`** (first applied in ADR 0043).
  One more permanent op name, no added contract-check code.

**MUST: the reverse op also writes its output in forward time order** (only the scan order is
reversed). Requiring an output `flip` would run into the symbolic-axis rejection above.

### 3. Bit identity is a **permanent check** (exact Uint32 match with the decomposition)

ADR 0055 (deform_conv2d) had no A/B partner and fell back to a degenerate oracle. **GRU does not need
to**: with T fixed, torch's decomposed graph is a complete reference, so the strong form of ADR 0023
decision 2 applies.

Check: `packages/runtime/tests/gpu_gru_scan_parity_test.ts`. On the same input and the **same GPU** it
runs (1) the decomposed graph built from existing ops (`linear → slice → add → sigmoid` /
`mul → add → tanh` / `sub → mul → add` T times + `cat`) and (2) one `gru_scan` node, and asserts that
the **f32 bit patterns** match exactly.

Four conditions make it hold; breaking any one changes the rounding sequence:

**(a) The hidden-side reduction has the order of the `linear` GEMM** — sequential in ascending `k`,
written `acc = acc + w * h` (the form of `accumulatorUpdate` in `gemm.ts`). The GEMM splits K into
tiles of 16, but each output element still adds in ascending k, and zero padding of the edge tile is
`acc + a·0`, an exact identity in f32. **Confirmed by measurement**, including H = 19 (not a multiple
of 16).

**(b) bias last** — the hidden side of the decomposition is `linear` = GEMM, which **adds bias at the
store**, so the kernel writes `(Σ W·h) + b`. Reusing conv's bias-first (ADR 0024) would give
`bias + Σ` and a different rounding sequence (the same trap `gemm.ts` records as the design's largest
branch point).

**(c) The WGSL body of `sigmoid` is shared with the plain elementwise op** — `SIGMOID_STABLE_WGSL`
is imported from `codegen/numerics-wgsl.ts` (moved there from `codegen/elementwise.ts` on 2026-09-05;
same rule as `silu.ts`). A copy could split the rounding of the primitive and the fused version. `tanh`
is a WGSL builtin, so there is nothing to share.

**(d) The order and bracketing of operations follow the decomposition verbatim** — the update is
**`h' = (h − n)·z + n`**. The mathematically equal `(1 − z)·n + z·h` is **a different rounding
sequence**: **44,345 of 100,000** f32 elements differed (maxdiff 4.77e-07). The gate sums put the
hidden side first (`gh + gi`); the `n` product is `gh_n · r` (**the reset gate multiplies the hidden
product including b_hh** — moving the bias out of the reset is off by maxdiff 0.196); the `n` sum puts
the input side first (`gi_n + (gh_n·r)`). For finite f32, addition and multiplication commute, but this
sequence is the source of truth for rounding, and NaN payloads may differ by backend (the same reason
`silu.ts` keeps `SiluMulOrder` in its key).

### 4. A rounding barrier (a workgroup-memory round trip) is **needed here too** — measured

Written as one expression, `a * b + c` **is contracted into an fma**. On a real GPU:

| Form                                              | Bit mismatches against mul / add in 2 dispatches |
| ------------------------------------------------- | ------------------------------------------------ |
| `o[i] = a[i] * b[i] + c[i]` (one expression)      | **15,371 / 65,536**                              |
| `let p = a[i] * b[i]; o[i] = p + c[i];`           | **15,371 / 65,536** (naming does not help)       |
| `bitcast<f32>(bitcast<u32>(a[i]*b[i])) + c`       | **15,371 / 65,536** (bitcast does not stop it)   |
| write to workgroup memory, read back after barrier | **0 / 65,536**                                   |

The decomposition runs `mul` and `add` as **separate dispatches** (a storage round trip), so nothing is
contracted. The kernel therefore **writes the product to workgroup memory and reads it back after a
barrier** (the `silu.ts` technique) at two places:

1. the product `gh_n · r` in `n = tanh(gi_n + gh_n·r)`;
2. the product `(h − n)·z` in `h' = (h − n)·z + n`.

**Fault injection showed both are load-bearing** (removing the barrier and writing the fused form gives
205 / 512 and 294 / 512 mismatching elements at T4 / N1 / H128, and the check turns red).

**The barrier is not guaranteed by the WGSL spec** — the spec allows fusion and does not require a
workgroup-memory round trip to act as an optimisation barrier. Bit identity of finite values is **a
measured fact adopted as a check per backend**, like `silu.ts` and the RoPE fusion (if a backend update
splits the check, suspect this first). Metal has already split the conv2d parity
([known-issues.md](../known-issues.md)), so this check is a fact established on the reference
environment (Vulkan).

### 5. The kernel is "one workgroup = one batch element / one lane = one hidden unit" — `H ≤ 256`

- **The workgroup size is the compile-time constant `256`**, and the codegen key carries it, so the
  same key gives byte-identical WGSL (codegen determinism; ADR 0022 forbids runtime autotuning).
- Recurrences of different batch elements are independent, so **one workgroup = one batch element**;
  the batch direction crosses the dispatch limit (65,535 by default) with layer-norm's grid-stride. No
  synchronisation between workgroups.
- **One lane = one hidden unit**; shapes above `GRU_SCAN_MAX_HIDDEN = 256`
  (`packages/runtime/src/codegen/limits.ts`) fail loudly in the contract check (`OpContractError`) and
  in `gruScanParams` (`CodegenError`) (silently degrading would write `h_shared` out of range and corrupt other units'
  state). The only measured shape is **H = 128** (vowel-detector). Raising the limit needs a
  workgroup-internal grid-stride and a double-buffered `h`; it MAY be done when there is demand, with a
  geometry that stays in the key. (Re-graded 2026-10-18 to MAY: the original argument that no kernel
  reads device limits no longer holds — ADR [0115](0115-geometry-profiles.md) chooses GEMM
  geometry from a static shape × adapter profile and keeps determinism by putting the geometry in the
  key; the fail-loudly limit stays. Ruling: accepted as proposed.)
- Workgroup memory holds only **the H elements of `h`** and **a 256-word relay for the rounding
  barriers** (`W_hh` is `[3H,H]`, 196 KB at H = 128, more than the default 16,384 B, so it is read from
  storage every step). The time loop and the barriers sit only in **workgroup-uniform control flow**
  (`dims` is uniform — the MUST of `layer-norm.ts`; loads from storage are not proven uniform).

### 6. The exporter keeps T symbolic with `torch.library.custom_op`

`karume::gru_scan` / `karume::gru_scan_reverse` are defined with `torch.library.custom_op` +
`register_fake` (`tools/exporter/src/karume/custom_ops.py`). The body hides behind the fake, so tracing
does not enter the Python loop and **the time axis T stays symbolic in one node**. `karume::` is not in
the Core ATen decomposition table, so `run_decompositions(curated_decompositions())` does not unroll it
(as with `torchvision::deform_conv2d`). Measured: `karume.gru_scan.default (s77, 1, 5)`.

- **The eager implementation is torch's GRU decomposition verbatim.** Golden expectations come from it,
  so a drift would let "exporter and runtime agree and are both wrong" pass. It matches single-layer
  unidirectional `nn.GRU` **bit for bit under `torch.equal`** (forward and reverse, maxdiff 0.0).
- The handler key (`torch.ops.karume.gru_scan.default`) requires registration, so
  `tools/exporter/src/karume/aten_handlers.py` imports `karume.custom_ops` at the top. **The import is a
  process-wide side effect**, but the zero-side-effect rule of `CLAUDE.md` is an invariant of
  `packages/*` (TS tree-shaking) and does not cover the Python exporter (the same module imports
  `torchvision` for the same reason — ADR 0055 decision 7).
- **Several layers and directions are separate nodes placed by exporter normalisation** (decision 7).
  `aten.gru`'s `Tensor[16]` (16 weights) does not fit the IR, so one node is emitted per layer and
  direction, joined with the existing `cat` (dim 2, static axis). This ADR defines only the op
  contract; **the patch that splits `nn.GRU` belongs to the model port** (out of scope).

### 7. Fields **left out** of the contract (a missing field states that the form is not in the vocabulary)

| Axis                   | Treatment                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| layers (`num_layers`)  | No field. One node per layer (`num_layers` would make the op contain the layer loop and the `cat` between)   |
| bidirectional          | No field. A separate op name per direction (decision 2)                                                      |
| `has_biases=False`     | No field = **bias required, fixed arity 4**, as in `linear` ("bias always present")                         |
| `batch_first`          | No field. `gi` in the IR is fixed to `[T,N,3H]` (time first) — torch's decomposition normalises with `permute` |
| `dropout` / `train`    | No field (inference only — as dropout in ADR 0023 decision 4)                                               |
| omitted `h0`           | No field = **a required input slot**. A zero `h0` from `aten.full` becomes an initializer by constant folding (measured) |
| output `h_n`           | **Not returned** (see below)                                                                                 |

`h_n` is not returned: the contract is single-output with `y` only. Returning `h_n` MAY be added as a
revision of the op contract once a consumer appears (re-graded 2026-10-18 to MAY: the technical reason
— IR nodes were effectively single-output — is gone since node-level multiple outputs were allowed by
ADR [0068](0068-decode-exit-multi-output.md) decision 1; vowel-detector discards `h_n`; ruling:
accepted as proposed). The code comment in `packages/runtime/src/ops/names.ts` (`GRU_SCAN_OPS`) still
states the single-output rule as MUST.

### 8. LSTM is not added (no additions for symmetry)

"Things better not done" in `docs/op-vocabulary.md` and the Consequences of ADR 0043 take precedence.
The inventory for a future `lstm_scan`:

- **Reusable as is**: the all-at-once input-side linear; the WGSL skeleton of time loop + barriers;
  direction by op name; the bit-identity check (Uint32 against the T-fixed decomposition); (b)(c)(d) of
  decision 3 and the barrier of decision 4.
- **Different**: the cell state `c` adds a second workgroup buffer, halving the equivalent of
  `GRU_SCAN_MAX_HIDDEN`; `c0` adds an input slot (arity 5).
- **Open conflict**: LSTM's `(y, h_n, c_n)` has three outputs. A model that needs only `y` can copy this
  ADR's form; a model consuming `h_n` / `c_n` needs a multi-output contract. Node-level multiple outputs
  are allowed by ADR 0068 decision 1; the remaining wall is the exporter's `operator.getitem` wiring
  (ADR 0068 addendum 2, item 3). Whether Kokoro-82M's LSTMs consume `h_n` / `c_n` is **unverified** —
  check it before starting LSTM (the one branch that could rework the contract).
- Gate count as an attr shared with GRU is not possible (the attr-variant problem of `gelu` /
  `gelu_tanh`; a separate op, for the reason of decision 2).

## Consequences

- One contract set (TS `OP_CONTRACTS` + `OP_CONTRACTS` of `karume/ops.py` + the shape rule in
  `shapes.py` + `fixtures/op-contracts.json` + CPU reference + golden `gru_scan_block`), one kernel
  (both directions), one exporter handler and the custom op definitions. **No effect on existing IR or
  assets** (two new op names).
- The contract table went from 55 to **57 ops** at the time. Fixed arity 4 is second only to
  `deform_conv2d`'s 5.
- **`gru_scan` is an op with empty attrs** whose only freedom, the scan direction, lives in the op name;
  every such increment adds an op name (the accepted cost of decision 2).
- Performance was **not measured** (the maintainer's policy: correct results first, speed later). Reading
  `W_hh` from storage every step, one workgroup per batch element (one workgroup busy at N = 1) and half
  of the 256 lanes idle at H = 128 are performance candidates for [perf-ledger](../perf-ledger.md) when
  demand appears.
- Porting vowel-detector itself was out of scope; its two open points are settled in the addendum.
  **Right zero padding is not allowed as a workaround** (the reverse GRU would carry state back from the
  padding).

## Addendum (2026-08-14) — landed with the vowel-detector port

- **The length axis is `2*Dim("T")`** (the symbol is on the 20 ms output grid). Binding a symbol from an
  input that has only the derived dimension needed an IR relaxation, decided in
  **ADR [0057](0057-derived-dim-binding.md)** (`solveDim` is unique; a size that does not divide fails
  loudly).
- **"The number of 10 ms frames is even" is a runtime contract.** An odd frame count **drops the last
  frame** (the convention the real-weight E2E and the host post-processing already used); no right zero
  padding.
- Measured on landing: IR nodes 8,434 → 18, distribution 4 length buckets 34,088,454 B → one graph
  2,759,461 B (−91.9%), and the `.lab` output of the distribution route now matches the real-length
  route (`packages/e2e/tests/e2e_vowel_detector_chain_test.ts`). The `nn.GRU` split lives in
  `tools/export-recipes/vowel_detector/patch.py`.
