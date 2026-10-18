# 0055: Vocabulary extension: deform_conv2d (DCNv2, an extended atomic op)

- Status: accepted (2026-08-13); on 2026-10-18 the low-precision ineligibility of decision 1 and the
  single direct kernel of decision 2 were re-graded to MAY, and the no-hoisting rule of decision 2
  from MUST to SHOULD. The layer numbers used here were replaced by the entry doors of
  [0059](0059-op-vocabulary-entry-doors.md) on 2026-08-14 ("layer 1′" = extended atomic layer).
- Date: 2026-08-13
- Related: [0043](0043-op-addition-layers.md) (layers of the time) / [0023](0023-fused-attention.md) /
  [0024](0024-conv2d-implicit-gemm.md) / [0017](0017-rms-norm-conv2d-clamp-min.md) (conv2d contract) /
  [0020](0020-nan-propagation-bitwise.md) (NaN detected by bit pattern).
- Demand: every BiRefNet checkpoint (8 official plus the fine-tune Lucida) uses `ASPPDeformable` and
  calls `torchvision.ops.deform_conv2d` **20 times per forward** (5 instances × 4 branches, all
  `64 → 256`) — the one op that blocked BiRefNet_HR / lite / base / Lucida.

## Context

`torchvision::deform_conv2d` is a custom op (`TORCH_LIBRARY_FRAGMENT(torchvision, …)`) **outside Core
ATen**; after `run_decompositions(curated_decompositions())` it stays **one node** (measured).

BiRefNet's `DeformableConv2d.forward` is the DCNv2 form: `offset = offset_conv(x)` (the output of a
trained conv; the clamp is commented out) and `modulator = 2·sigmoid(modulator_conv(x))`. Offsets are
unbounded reals that can point outside the input plane and are **not export-time constants** (the
export-eliminated layer does not apply).

It cannot be composed from the vocabulary either: there is no `floor` (`cast` truncates, which is not
floor for negatives), and `gather` works on the last dimension only, so it cannot express a
value-dependent gather over two axes. An **atom** outside Core ATen needs an ADR as a permanent public
commitment (the interim rule of 2026-08-13, later the extended atomic layer of ADR 0059).

## Decision

### 1. The contract covers the measured form only — missing fields express the restriction (ADR 0023 decision 4)

`deform_conv2d`, **fixed arity 5** (`x` / `weight` / `offset` / `mask` / `bias`), uniform f32, and a
single attr, **`padding`** (`[H, W]`).

| Tensor   | Contract                                                         |
| -------- | ---------------------------------------------------------------- |
| `x`      | `[B, Cin, H, W]`                                                 |
| `weight` | `[Cout, Cin, Kh, Kw]` (no groups, so the second axis is Cin)     |
| `offset` | `[B, 2·Kh·Kw, Hout, Wout]` (**even channels = y / odd = x**)     |
| `mask`   | `[B, Kh·Kw, Hout, Wout]` (modulator — applied **after** sampling) |
| `bias`   | `[Cout]`                                                         |
| output   | `[B, Cout, Hout, Wout]`                                          |

- **No `stride` / `dilation` / `groups` / `offset_groups` fields — all fixed at 1.** All 20 measured
  sites use 1, and the missing field declares that other forms are not in the vocabulary (the approach
  of `conv_transpose1d`'s `output_padding` / `dilation` / `groups`, ADR 0015). "No additions for
  symmetry" (`docs/op-vocabulary.md`) applies to stride too. Whether observed subsets like this one
  are op semantics is due for a ruling (ADR 0064 axis A; `docs/backlog.md`, release section).
- **`mask` is a required input slot — DCNv2 only.** DCNv1 (`use_mask=False`) is expressed by the
  absence of a slot, so the fixed arity fails loudly. torchvision passes a `[1,1]` dummy and
  `use_mask=False` when `mask=None` (measured), so the exporter checks `use_mask` directly.
- `bias` is **not synthesised**, unlike the arity-3 conv family: with `bias=None` the torchvision
  wrapper inserts `aten.full([Cout], 0)`, which constant folding turns into an initializer (measured)
  — no second synthesis path in the exporter.
- Only `padding` varies (BiRefNet's four branches use `k//2` = 0 / 0 / 1 / 3). The output plane
  `Hout = H + 2·padH − (Kh − 1)` / `Wout = W + 2·padW − (Kw − 1)` is **derived from x, weight and
  attrs**, and the contract check compares it with the spatial axes of `offset` / `mask` (torchvision's
  meta kernel takes `offset.shape[-2:]`; reading one fact from two places lets an IR with a mismatched
  offset pass).
- Low-precision weight storage is **not enabled** (`deform_conv2d` is not in `WEIGHT_SLOTS` /
  `WEIGHT_CHANNEL_AXES`); adding it MAY be done when there is demand. (Re-graded 2026-10-18 to MAY:
  the original reason was an i8 scale-axis mix-up, a silent wrong value recorded in ADR 0024 decision
  6; the key match of the two tables is now checked mechanically (`packages/runtime/src/ops/names.ts`,
  `WEIGHT_CHANNEL_AXES`), f16 has no scale axis, and the BiRefNet f16 preset already keeps these 20
  weights in f32 (ADR 0113); ruling: accepted as proposed.)

### 2. One direct-convolution kernel

One invocation computes one output element with a grid-stride loop (full write — ADR 0014). The
implicit GEMM skeleton of ADR 0024 would carry over by replacing the B-tile gather (one read → four),
but this ADR does not write it: (1) a new atom has no existing route, so an igemm written first is no
A/B oracle (decision 4 sets the oracle); (2) performance was not measured, so there was no reason for a
second kernel. A second kernel MAY be added once it is measured and filed in the perf ledger (re-graded
2026-10-18 to MAY: the reason was the absence of a measurement; ruling: accepted as proposed). It is
now filed as perf-ledger K-63.

- The reduction is **sequential in ascending `(ic, kh, kw)`**, exactly the nesting of the direct conv2d
  kernel.
- The reads of `offset` / `mask` SHOULD NOT be hoisted out of the `ic` loop. With `offset_groups == 1`
  hoisting is semantically equivalent, but reordering the loops to `(kh, kw)` outside and `ic` inside
  **changes the reduction order and breaks the degenerate bit match of decision 4**. Hoisting is a
  performance candidate that goes together with a redesign of that oracle (a compiler hoist of the
  `ic`-independent address keeps the order and is harmless). (Re-graded 2026-10-18 from MUST to
  SHOULD: the invariant is the oracle, and the ban is only its means; ruling: accepted as proposed.)
- bias is the initial `acc` (bias-first — ADR 0024 decision 3).
- The four bilinear corners follow torchvision verbatim: if the centre is outside `(−1, in)` the **whole
  tap is 0**; inside, an out-of-range corner is **0 for that corner only** (no border clamp). Weights
  and addition order are fixed as `w1·v1 + w2·v2 + w3·v3 + w4·v4` (`w1 = λy0·λx0`, …).
- mask multiplies **after sampling and before the weight** (`(m · v) · w`), matching torchvision's
  im2col buffer (`mask · bilinear(…)`) and its GEMM `w × col` down to the bracketing.

### 3. Determinism holds (no atomics in the forward pass)

1. Each output element `(b, oc, oy, ox)` is owned by one thread as a pure gather. torchvision's forward
   (`deformable_im2col_kernel`) has no atomics either; only the backward `deformable_col2im_kernel` is
   non-deterministic. The reason to avoid `scatter_add` (WGSL has no `atomic<f32>`) does not apply.
2. Offsets are runtime values but decide only **which address is read**. The reduction order and the
   corner addition order are fixed by program order, so the rounding sequence depends only on inputs.
3. Neither condition of non-determinism (several threads adding to one address; a reduction order
   that changes at run time) is met.

### 4. The bit-identical oracle is the degenerate case = conv2d

There is no A/B partner for a new atom, so the **degenerate case falls back to an existing op**:

- **With `offset` all 0 and `mask` all 1, the output matches plain `conv2d` (same weight / bias /
  padding, stride 1, dilation 1, groups 1) bit for bit (Uint32).** Reasons: (1) λ is exactly 0, so
  `w1 = 1, w2 = w3 = w4 = 0` and `1·v + 0 + 0 + 0 = v` exactly in f32; (2) multiplying by `m = 1.0` is
  exact; (3) out-of-range taps **add** `0` in deform and are **skipped** in conv2d, but `fl(s + 0) = s`
  (they differ only in signed zero — the same single difference ADR 0024 decision 3 records); (4) the
  reduction order is the same (decision 2). GPU check: `packages/runtime/tests/gpu_deform_conv2d_test.ts`.
- **The main check is the torch (torchvision) golden** `deform_conv2d_block` (asymmetric: Cin ≠ Cout,
  Kh ≠ Kw, padding H ≠ W). Offsets are positive and negative, non-integer and cross the border, so both
  zero-fill stages (the early zero of the centre and the per-corner zero) are exercised. Whether
  op-level torch goldens stay as oracles is an open point of
  [0129](0129-test-layout-and-lanes.md).
- The CPU reference is **written in a different form from the kernel** (a four-deep loop iterating
  taps by coordinate), so an axis mix-up cannot cancel out on both sides.

### 5. NaN and huge offsets — positive range test, NaN propagates

`i32(floor(y))` is undefined for out-of-range values and NaN (`docs/limitations.md`). torch's C++ has
the same UB in `(int)floor(NaN)`, so "matching torch" is no escape.

1. **The range test is written positively**: `y > −1.0 && y < f32(H) && x > −1.0 && x < f32(W)`. For
   finite values it is exactly equivalent to torchvision's negated form (`h <= −1 || height <= h` gives
   an early 0), and huge values (`±1e30`) give a zero tap as in torch. After the test,
   `floor(y) ∈ [−1, H−1]` always fits in i32, so the undefined cast is gone **by construction**.
2. **NaN is propagated to the output, not turned into 0** (following ADR 0020). The positive test is
   false for NaN, which would make a NaN offset a silent zero contribution. Out-of-range and NaN are
   told apart by bit pattern (`(bitcast<u32>(v) & 0x7fffffff) > 0x7f800000`), and a NaN sets the output
   element to NaN. Routing through `clamp` is rejected: it repeats the driver `max` swallowing NaN that
   ADR 0020 fixed.
3. **An out-of-range index is not a contract violation.** Unlike gather / embedding, where out of range
   means NaN pollution (`docs/limitations.md`), deform's out-of-border samples are **normal semantics
   (0)**. The two conventions coexist in the vocabulary, so the op doc separates them explicitly.

### 6. Rejected: decomposition through `grid_sampler_2d`

Reshaping `offset` into a grid → `grid_sampler_2d` → multiply by mask → `conv2d(stride=(Kh,Kw))` is
exactly equivalent, and `aten.grid_sampler_2d` is **in Core ATen** (no ADR needed). It is still
rejected:

- **The intermediate is im2col itself** (`[B, Cin, Hout·Kh, Wout·Kw]`). In BiRefNet_HR's
  `decoder_block1` (512², Cin = 64) that is **3.29 GB for k = 7 and 604 MB for k = 3**, the level at
  which ADR 0024 rejected materialised im2col (3.62 GB); 822 MB for k = 7 even at 1024² operation.
- The node count grows from 20 to about 280 (per site: 2 permutes, 3 reshapes, 4 normalisation ops, 2
  for the body). Permutes materialise copies, so reshaping the offsets alone adds 102.8 MB × 2 at k = 7.
- **A hybrid does not reduce the commitment**: the two k = 1 branches of four could decompose without a
  capacity problem, but k = 3 / k = 7 still need deform itself.
- The round trip of normalised coordinates (`[−1,1]` ↔ pixels) adds one rounding (it could show in the
  torch check — not measured).

`grid_sampler_2d` itself is decided **independently** when STN or optical-flow warp demand appears (it
stays on hold as family 3 of `docs/op-vocabulary.md`; this ADR does not lift that).

### 7. torchvision becomes a base dependency of the exporter

The handler key (`torch.ops.torchvision.deform_conv2d.default`) and golden generation (rerun by every
`uv run pytest`) need the import. A conditional import would split environments into "op missing" and
"passes", so torchvision is in the base `dependencies` of `tools/exporter/pyproject.toml` (the
`anima` / `siglip2` group listings stay — harmless, and they record why those groups need it).

## Consequences

- One contract set (`OP_CONTRACTS` + `karume/ops.py` + `shapes.py` + `fixtures/op-contracts.json` +
  CPU reference + golden COVERAGE), one kernel and one exporter handler. No effect on existing IR or
  assets (one new op name).
- **What it buys is one architecture, the BiRefNet family** (8 official checkpoints + Lucida, all
  weight swaps of the `BiRefNet_HR` architecture). Narrow for a permanent commitment, but the family
  has no other way out (`lite` uses the same `dec_att`).
- BiRefNet_HR at 2048² needed work outside this op (conv2d dispatch limit, large intermediates,
  BatchNorm2d folding) and now ships ([0113](0113-birefnet-weight-series.md)); on Intel Arc B570 one
  2048² dispatch exceeds the driver's 5 s job limit (`docs/limitations.md`).
- One dispatch takes 1.66 s at 1024². Implicit GEMM or an output split is perf-ledger K-63; a candidate
  that changes the `(ic, kh, kw)` order of decisions 2 / 4 comes with a redesign of the oracle.
- The interim layer rule this ADR applied (an atom outside Core ATen coming from a model) was replaced
  by the entry-door model of ADR 0059 (2026-08-14), which closed the known-issues entry.
