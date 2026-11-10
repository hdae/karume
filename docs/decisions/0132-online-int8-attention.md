# 0132: Online int8 attention — `attentionScoreStorage: "none"` (S never materialised)

- Status: accepted (2026-10-23 — the Wan2.2 A/B check and the timing of decision 6 are below);
  decision 5's fallback to the three dispatches warns since [0147](0147-numerics-and-kernel-adoption.md) decision 8
  (2026-11-10)
- Date: 2026-10-23
- Related: ADR [0023](0023-fused-attention.md) (fused attention: the three dispatches ①QK / ②
  stats / ③PV and their bit identity with the decomposed path — unchanged, the default path) /
  [0030](0030-attention-a8-execution.md) (the a8 variant whose quantisation this kernel keeps) /
  [0058](0058-numerics-opt-in-contract.md) (one named option per numerics change; shape and limit
  fallback with the key showing it) / [0060](0060-row-block-attention.md) and
  [0124](0124-attention-row-block-budget.md) (row blocks: capacity of the S temporaries — this
  kernel has none) / [0100](0100-rms-subgroup-reduction.md) (the `subgroups` acquisition — amended
  below) / [0110](0110-practical-tier-numerics-contract.md) (contract classes; this variant is Q) /
  [0074](0074-quant-seat-naming.md) (preset naming — token `online` added) /
  [0128](0128-default-preset-original-weights.md) (C1: the default preset needs no feature) /
  [0121](0121-wan22-ti2v-5b.md) (Wan2.2 — the practical preset this replaces as default) /
  perf-ledger K-82 / research
  [2026-10-23-flash-attention-webgpu-recon](../research/2026-10-23-flash-attention-webgpu-recon.md)
  (the measurements behind every number here).

## Context

Wan2.2 at 1280×704×121 spends 64% of one DiT forward (24.2 s of GPU time on the RTX 3080 Ti) in
the three attention dispatches, and 16.4% in ② row statistics alone, which is pure traffic over
the f16 score matrix S (35.7 GB per layer at S = 27,280). ①QK and ③PV are compute-bound on dp4a
(32 and 19 TOPS against a measured ceiling of 65), so the arithmetic cannot be cut without matrix
units, which stable WebGPU does not expose; what can be cut is ② and the S round trip. A
FlashAttention-2-style kernel — one dispatch per block of query rows that streams int8 key / value
tiles, keeps the running row max and sum, and never writes S — was measured as a standalone spike
(recon §10.2): ×1.46 against the current three dispatches at S = 27,280 (×1.61 with the key order
of Vᵀ prepared on the host), ×1.04–1.15 at S = 7,920. The K-5b kill line (×1.3 on the attention
kernels) is passed at the long shape, which is the target; the maintainer ruled on 2026-10-23 to
productise it and to make it Wan2.2's default once the A/B check confirms the expected speed.

Two things made the spike win and shape this design: the shared-memory tile layouts must be free
of bank conflicts (the first version, with the natural layouts, was 2× slower than the three
dispatches), and the row reductions are 1.23× faster through subgroup shuffles than through
workgroup memory. The second collides with ADR 0128 C1 (a default preset declares no GPU feature)
and ADR 0100 decision 1 (`subgroups: true` rejects a device without them).

## Decision

1. **The option is a value of the existing score-storage axis: `attentionScoreStorage: "none"`.**
   "S is not stored" is the definition of the online kernel, so no new key enters the manifest
   vocabulary (ADR 0058 decision 1: one axis per knob). `"none"` is implemented only with
   `attentionCompute: "a8"`; any other combination fails loudly at Session construction (an
   unimplemented combination is not a fallback — ADR 0058 decision 3). Preset names spell it with
   the token `online` in the slot of `s16` (`f16+dit8-a8-attn8-online`; ADR 0074 §3 addendum).
2. **One kernel per node, dispatched over windows of query rows.** For every node the recipe runs
   the existing activation quantisation (`quantize_rows` for q and k; `strided` permute +
   `quantize_rows` for Vᵀ — unchanged, the same temporaries as a8) and then the online kernel
   `attention_online:v1:i8a8:…` once per window of query rows. The window is not a capacity
   device (there is no S) but a dispatch-size device: the planner (a pure function next to
   `planAttentionBlocks`) sizes the window from a work budget so that one dispatch stays well under
   the 1 s submit gate (`ATTENTION_ONLINE_WINDOW_MACS` = 2⁴⁰ multiply-accumulates: at the
   121-frame shape 6,528-row windows, 5 per node, about 70 ms each — smaller windows lose
   occupancy, 768-row windows measured 15% slower than one dispatch); the estimate (ADR 0124
   decision 7) counts no S temporaries for this route and takes the window count from the same
   function.
3. **Geometry**: 256 threads, 64 query rows per workgroup, 64 keys per tile, D ∈ {64, 128} as
   a generator parameter (D = 256 would need 58 KiB of tiles, above the 48 KiB of Dawn / Vulkan,
   and is not built; other depths and `N % 4 != 0` fall back to the three
   dispatches of a8 / f32 with the key showing it — shape eligibility, ADR 0058 decision 3). The
   `q` tile is loaded once; `k` rows are padded to 33 words and each lane owns keys 16 apart; the
   Vᵀ tile is stored `[pack][d]` (word `w` packs keys `{w, w+16, w+32, w+48}`, gathered byte-wise
   from the quantised Vᵀ); P̃ rows are padded to 17 words. These layouts are part of the
   definition (they are what makes the kernel faster than the three passes).
4. **Numerics**: per key tile, S by `dot4I8Packed` and the a8 dequant (`f32(acc) · (qs'·ks')`,
   ADR 0030), the tile's row max `tm`, `m' = max(m, tm)`, `α = exp(m − m')`, `l = l·α + Σ exp(S −
   m')`, P̃ = `round(127·exp(S − m'))` packed to int8, `O = O·α + f32(Σ P̃·Vᵀ)` with `O` in f32;
   at the end `O · (vs[d] · (1/127) / l)`. The row sum `l` is kept as a per-lane partial and
   reduced once at the end. Keys beyond N take the identity −3.0e38 (not −inf: `α` must not
   become `exp(−inf − (−inf))`). The mask and GQA are rejected exactly as for a8 (`fusedAttentionViolation`).
   Contract class **Q** (ADR 0110): the error model is the int8 lattice of P̃ against a running
   max (each weight carries at most `0.5/127` of the final `exp(S − m)`, since the later rescale
   by `α ≤ 1` only shrinks the step) plus f32 accumulation; the four-point check of ADR 0110
   decision 3 is the kernel test, with a TS mirror that walks the same 64-key tiles in the same
   order and treats `exp` statistically as ③PV does (a mismatch is ±1 step of P̃, never ±2).
5. **Subgroups are a speed-only variant, bit-identical to the portable one.** The kernel has two
   reduction forms with different keys: `:sg32` (`subgroupShuffleXor` across the 16 lanes of a row,
   `@subgroup_size(32)`, the lanes addressed by `local_invocation_index` under the linear lane
   mapping that `acquireGpu` measures on the device — `GpuContext.subgroupLanesLinear`; identifying
   them by the subgroup builtins instead costs 16%) and `:wg` (workgroup memory).
   The row max is exact under any order; the final sum of `l` uses the same butterfly pairing
   (`((l0+l8)+(l4+l12)) + …`) in both forms, so the bits are the same — a test runs both on one
   device and requires Uint32 equality (the same discipline as `dot4I8Packed` against its
   emulation). The build selects `:sg32` when the device has `subgroups`, `subgroup-size-control`,
   the WGSL feature `subgroup_id` and the measured linear lane mapping, otherwise `:wg` when the workgroup-storage limit allows it
   (at D = 128 `:sg32` declares 29,440 B and `:wg` 33,536 B — 32 KiB devices without subgroups
   fall back to the three dispatches,
   also keyed). Because the choice cannot change a bit, it is capability-driven without breaking
   ADR 0058 decision 2. (Amended 2026-11-10 by ADR [0147](0147-numerics-and-kernel-adoption.md)
   decisions 8 and 9: the fallback to the three dispatches also warns once per device; the
   `:sg32` / `:wg` choice stays capability-driven, gains a run-time canary and does not warn.)
   **ADR 0100 decision 1 is amended**: `AcquireGpuOptions.subgroups` accepts `"available"` —
   request the features when the adapter advertises them, silently run without otherwise; `true`
   keeps its rejecting meaning. The 32-lane known-answer check of ADR 0100 runs whenever the
   features were requested, in both modes: an adapter that advertises them and fails the check is
   rejected rather than silently demoted (a broken advertisement is a defect to surface). The Wan family acquires with `"available"` when the preset's
   `session` declares `"none"`; a default preset still declares no `gpuFeatures` (ADR 0128 C1
   holds — the feature is never required).
6. **Wan2.2 gets a new practical preset `f16+dit8-a8-attn8-online`** with the same `weights` map as
   `f16+dit8-a8-attn8-s16` (no new download) and `session` `{ linearCompute: "a8",
   attentionCompute: "a8", attentionScoreStorage: "none" }`. It becomes `defaultQuant` when the
   e2e A/B check against the reference preset stays inside the band derived for the a8 preset and
   the per-op timing shows the expected gain (the maintainer's condition of 2026-10-23); the old
   practical preset stays as a sibling (users who pass its name keep working). Wan2.1 is not
   changed (the maintainer uses Wan2.2; the kernel applies to it unchanged if wanted later). The
   practical sha256 rows of the new preset are written fresh; the reference rows do not move.
7. **Scope**: the fused attention op (`attention` — the DiT models). The states-form attention of
   the language models keeps its kernels; porting the online form there is a later candidate.

## Consequences

- Wan2.2's 121-frame forward loses ②'s 4.0 s and the S round trip; expected about −20% of the
  forward on the RTX 3080 Ti (more with the host-side Vᵀ key order, which is a separate, measured
  follow-up with its own kill line ×1.05).
- Four new surfaces: the kernel generator (`src/kernels/attention-online-i8a8.ts`), the route
  value in `attentionRoute`, the window planner, and `subgroups: "available"` in `acquireGpu`.
  The hub's `SCORE_STORAGE` allowlist grows by one value; an older runtime reading a manifest with
  `"none"` fails loudly, which only affects a revision it was never pinned to.
- The Wan2.2 manifest changes (a preset and, on acceptance, `defaultQuant`), so the 0.15 release
  re-uploads it (release runbook).
- Devices without subgroups and with 32 KiB of workgroup storage (today: Firefox and Safari on
  D3D12 / Metal) run the three dispatches for this preset — slower, same values as the a8 preset
  they already get.

## Alternatives considered

- A new key `attentionKernel: "online"`: adds an axis and leaves `attentionScoreStorage` meaningless
  under it ("silently ignored" — the thing ADR 0058 decision 3 distinguishes from eligibility).
- Requiring `subgroups: true` and declaring it in `gpuFeatures`: the preset could not be the default
  (C1), and a bit-identical portable form costs one summation-order discipline.
- A subgroup-matrix (tensor-core) kernel: 91 TOPS measured on a naive i8 GEMM, but the feature is
  experimental behind `allow_unsafe_apis` and its WGSL surface is changing — a Node-only opt-in,
  ruled separately (not in this ADR).
- Keeping the three dispatches and optimising ③PV's tile fill and ②'s reads (K-69): both are made
  redundant by the online kernel for this preset.

## Results (2026-10-23, RTX 3080 Ti, Node + Dawn)

- Kernel check (`gpu_attention_online_i8a8_test.ts`): the output uses at most 33% of the error
  model (a single 64-key tile), 8–20% on the other shapes; P̃ read out through an identity V matched
  the ③PV reference on every element; both reduction forms, dp4a and its emulation, repeated
  dispatches and row windows are bit-identical; both fault injections leave the model.
- Wan2.2 A/B at 1280×704×33 (`e2e_wan_ti2v_ab_test.ts`): relRMS 4.23e-2 against the reference
  preset (the s16 preset: 4.4e-2; band 8.8e-2), census 240 online dispatches and no three-pass
  kernel. The 121-frame A/B could not run: the reference preset's 121-frame step does not fit the
  12 GiB device under Node + Dawn (both practical presets fail in `beforeAll` — docs/known-issues.md).
- DiT measured mode at S = 27,280 (one forward, the second run of the same input, each seat in a
  fresh process after cooling the GPU to 60 °C): attention 18.4 → 12.5 s (×1.47; 180 dispatches of
  `attention_online:…:sg32`), forward 29.3 → 21.7 s. Against the morning's s16 baseline on the
  same machine (attention 15.6 s, forward 24.2 s) the gain is ×1.25 / −10%: the s16 run above was
  throttled (its linear took 8.9 s against 6.8–7.3 s). Two corrections were needed to get there:
  the subgroup form must address its tiles by `local_invocation_index` (16% faster than by the
  subgroup builtins — hence the lane-mapping probe), and the windows must be large enough for the
  machine (768-row windows ran 15% slower than one dispatch; 2⁴⁰ gives 6,528 rows).
- The default of Wan2.2 switched to `f16+dit8-a8-attn8-online` on this evidence (decision 6).

## Open points

- The visual check of the online preset (12 clips, 50 steps, 33 frames — the ruling of ADR 0128
  decision 1 for a default) was done for the s16 preset only; the online preset's quality is held
  by the kernel error model and the 33-frame A/B.
- The 121-frame A/B under Node (2026-10-24, after the release wait of ADR 0095's addendum of the same
  day): relRMS 4.675e-2 (s16 4.915e-2), inside the 9.9e-2 band.
- Bc = 128 on 48 KiB devices, double-buffered tiles and `exp2` are unmeasured headroom (the spike
  sits at 26–29 TOPS, under half the dp4a ceiling).

## Addendum (2026-10-25) — decision 3's layouts replaced by vec4 tiles (key `v2`, perf-ledger K-84)

Decision 3 names the tile layouts as part of the kernel's definition. They are replaced; everything
else in this ADR holds (the algorithm, the numerics of decision 4, the two reduction forms of
decision 5, the windows of decision 2, the route and the presets). Two values quoted above change
with the layouts: the key of decision 2 is now `attention_online:v2:i8a8:…`, and the workgroup
storage of decision 5 is given under "Workgroup storage" below.

- **Layouts** (`attention_online:v2:…`): the `q` and `k` tiles are `vec4<u32>` rows of `D/16`
  words padded by one (stride 9 at D = 128, 5 at D = 64). The Vᵀ tile is `[pack][d/4]` in `vec4`:
  word `w` of a column still packs the keys `{w, w+16, w+32, w+48}`. It is filled by owner threads:
  a thread owns one whole `vec4` (`pk`, `d4`), consecutive lanes take consecutive keys of the same
  four columns, and the row stride is `D/4 + 1`. The P̃ tile is transposed (`[word][row group]`,
  one `vec4` holding a word of the four rows of a thread). In the PV phase a lane owns the columns
  `{64·g + 4·tx + c}`, and the phase runs once per 64-column group. The q, k, O and vs bindings
  are `array<vec4<…>>`.
- **Why**: the speed recon of 2026-10-24/25
  ([research](../research/2026-10-25-speed-recon-3080ti.md) §4) found the v1 kernel at about 25 TOPS
  against a dp4a ceiling of about 65 TOPS on the RTX 3080 Ti. The Vᵀ fill with lanes along `d`
  touched 32 cache lines per load, and the scalar shared loads limited both dot-product loops. Measured
  at the switch on the RTX 3080 Ti (Node + Dawn, v1 and v2 interleaved in one lock hold): the
  production route bench (4 runs each) gives self-attention ×1.44 at 33 frames and ×1.46 over the
  121-frame windows, and cross-attention ×2.0. One 33-frame DiT forward of the default preset takes
  3,004 ms instead of 3,325 ms (median of 14, −9.7%). One 121-frame forward takes 16.0–16.1 s instead
  of 19.8–19.9 s as the first forward of a run; later forwards drift up in both kernels on this
  machine (median 18.9 s vs 24.6 s). On the RTX 5070 Ti (Chrome 155, D3D12, runbook step 02a) the
  recon's spike of these layouts (`bestC-split`) runs ×1.68 (self, 33 frames), ×2.20 (cross) and
  ×1.72 (121-frame window) faster than v1, with 0 differing words, and loses nothing with up to 48
  added registers (the RTX 3080 Ti meets its occupancy step at 16–23).
- **Bit identity**: v2 changes which thread computes which element, never the value. The integer
  sums are exact, and every f32 operation keeps v1's operands and order. Checked at the switch:
  0 differing u32 words between v1 and v2 over D ∈ {64, 128}, dp4a and its emulation, both reduction
  forms, partial key and row tiles, a row window, adversarial input regimes and the full Wan2.2 self
  and cross shapes. A fault injected into v2 (two rows of P̃ swapped) was caught at 6.1 M words.
  Every Wan sha256 row is unchanged. These checks ran on Vulkan / NVIDIA (RTX 3080 Ti). On D3D12
  only the recon's spike of these layouts was compared with v1 (0 differing words on the RTX 5070
  Ti). After the switch v2 itself was checked there: the playground's 17-frame quick-check hash
  of the default preset (runbook step 01) is unchanged, and runbook step 02 (run 3) times the spike
  against v2 at ×1.004–1.005 with 0 differing words. The f32 operations keep v1's
  order, but a driver compiler may contract a multiply and a later add into an FMA differently in
  the two kernels, so those end-to-end hashes are the check, not this argument.
- **MUST compliance**: every workgroup store is a whole `vec4` or a whole scalar. No component is
  written through a dynamic index (the naga/Metal silent drop of gemm.ts `storeBTransposed`). The
  register count drops by running the PV phase per column group: 118–121 on the proxy compile paths
  (Tint → SPIR-V → NVIDIA Vulkan, and DXC → dxil-spirv), below the 128-register occupancy step. The
  D3D12 driver's own count is unobserved.
- **Workgroup storage**: `sg32` 31,232 B at D = 128 (still under the 32 KiB of D3D12 and Metal),
  `wg` 35,328 B. The `wg` form already exceeded 32 KiB in v1 (33,536 B), so devices without
  subgroups and with 32 KiB run the three dispatches as before.
- **Landing condition** (the maintainer's ruling of 2026-10-25): v2 replaces v1 when the recon's
  spike of these layouts (`bestC-split`, runbook step 02a) runs at least as fast as v1 on the RTX
  5070 Ti (Chrome, D3D12); otherwise both stay, chosen per adapter. Met on 2026-10-25 with the
  figures under "Why" (two runs agree). After the landing the production route bench on the RTX
  5070 Ti reads online ×1.66 (self, 33 frames), ×1.69 (121 frames) and ×2.15 (cross) over v1, and
  online / s16 0.80 / 0.69 / 0.70 (v1: 1.31 / 1.18 / 1.50).

## Addendum (2026-10-27) — the key valid length: `N % 4 != 0` runs online (a padded Vᵀ)

Decision 3 made `N % 4 != 0` fall back to the three dispatches (Vᵀ and P̃ pack four keys per
word). That condition is lifted for the online kernel only; ③PV's i8a8 kernel and
`attentionPvI8a8Eligible` keep it, so the three-dispatch routes (a8, a8 + s16) do not change.

- **Why**: MOSS-Image's DiT attends over `S = N + T` with a constant negative-branch text length
  (T₋ = 10). On a square resolution `N = k²` is 0 or 1 mod 4, so `S₋` is 2 or 3 mod 4 and never
  eligible: half of all forwards fell back to the three dispatches with S in f32, and the
  practical preset's census check (`assertAttentionSeatsComplete`) could not hold.
- **What changed**: `attentionOnlineI8a8Eligible(D, N)` needs `D % 64 == 0`, `D ≤ 128` and
  `N ≥ 1`. `attentionOnlineKeyWords(N)` names the key-axis form — `whole` (`N % 4 == 0`) or
  `padded` — and the kernel, its uniform, the Vᵀ producer and the size of the quantised Vᵀ take it
  from that one function (the online node computes the form once and hands the same value to the
  producer, the kernel and `attentionOnlineI8a8Params`, which throws when the uniform's N is of the
  other form). In the `padded` form each Vᵀ row is `ceil(N/4)` words: the producer is a new form of
  `quantize_rows` (`…:pad`, rows of any length, the bytes from element N on forced to 0, the scale
  the amax over exactly N elements), and the kernel (`…:kpad`) differs only in its Vᵀ row length
  `n4 = ceil(N/4)`. The producer takes the form its reader reads: ③PV passes `whole`, so an
  `N % 4 != 0` that reached ③PV would still fail loudly at its producer, as before the addendum.
  The permute or tiled transpose to f32 Vᵀ is unchanged (its rows hold N elements). q and k pack
  along D, so they need no padding. Keys from N on already score −3.0e38 (decision 4), so their P̃
  is `round(127 · exp(−3.0e38 − m)) = 0` and the padding bytes never reach O: the kernel's output
  does not depend on them.
- **Invariant (MUST)**: the `whole` form keeps the key and the WGSL of every existing codegen key
  byte for byte, and the same buffers (`vq` stays `B·H·D·N` bytes) and dispatches. So every
  `N % 4 == 0` shape — every accepted Wan2.1 / Wan2.2 size, the only online users so far — is
  unchanged by construction; the codegen snapshots of the existing keys did not move. The one Wan
  shape with `N % 4 != 0` is the Wan2.2 DiT test's golden S = 8,190 (832×480 × 81 frames, outside
  the accepted sizes): on the online preset it now runs the online kernel (`…:obshd:kpad`) instead
  of the three dispatches, so the complete-eligibility census check applies to it (measured: 1
  submit at most 151.1 ms against the 1 s gate, non-finite 0, the check green). Anima does not use
  the online kernel and its three-dispatch routes are untouched (its `N % 4 != 0` shapes still run
  ③PV and S in f32 — docs/limitations.md).
- **Checks** (`gpu_attention_online_i8a8_test.ts`, RTX 3080 Ti, Node + Dawn, 2026-10-27): `N` =
  65, 4,106 and 203 (`N % 4` = 1, 2, 3; D = 64 and 128; batch 2–3) pass the four-point check — the
  band against the exact f64 attention over exactly N keys (a reference with no padding at all),
  the A/B against the three-pass a8 references, the TS mirror within two lattice steps, and the
  structural bit identities (dp4a / emulation, sg32 / wg, row windows, repeated dispatches) — on
  three inputs each. On the dense inputs of the `padded` shapes they use 34.6%, 3.6% and 20.4% of
  the error model (7.6–33.1% on the `whole` shapes). That band grows with Σ|ṽ| over all N keys, so
  at N = 4,106 it barely sees a wrong Vᵀ row length: the whole form's `floor(N/4)` left it at 1.33×
  the bound and the last word of each row dropped at 0.15×. The two spike-V inputs (one key per
  column, four columns on the last valid keys; a sharp softmax and a flat one) make the model one
  key's half lattice step, and see both faults at every padded shape (`floor(N/4)`: 613.4×,
  249.6×, 461.0×; last word dropped: 252.4×, 86.3×, 252.7×). They use 97.7–99.3% (sharp) and 7.4–82.9% (flat) of
  the model: with one key per column the worst row's rounding of that key is almost the whole
  bound. The half step is a proven bound on that rounding; the f32 term (a fixed 2e-5) is an
  allowance measured sufficient on this machine, not proven — an adapter whose `exp` or summation
  order differs may need it revisited before the band is read as a defect.
- **The mask bound**: the padding bytes filled with 0 and with 127 give Uint32-identical outputs —
  the primary detector of a mask bound past N (patched into the generator as `ceil(N/4)·4`, it
  turned red at all three shapes). The fault case of the f64 band — the same bound, padding bytes
  127, the flat spike (under the fault the padding keys score 0, and a sharp row max rounds their
  P̃ to 0) — leaves the model at 591.8×, 34.1× and 175.6×. With the producer's zero bytes a leaked
  padding key enters l only: the 0/127 check cannot see that, and the f64 band sees it only where it
  weighs (N = 65: 1.12× the bound; N = 4,106: 3.6% of it on the dense inputs, 8.1% on the flat
  spike — a bias inside the a8 error model there).
- **The producer and the Session path**: the padded `quantize_rows` writes word for word what the
  plain form writes over the rows extended with zeros (`gpu_i8a8_test.ts`); the Session path is
  bit-identical to the raw kernel on host-packed inputs (`gpu_attention_online_session_test.ts`
  case (v), N = 203, 65 and 4,106 — an undersized `vq` turns it red); and the token-major chain
  (tiled transpose → padded quantisation → `…:obshd:kpad`) is bit-identical to the head-major one
  on the online routes at N = 203 (`gpu_attention_permute_fusion_test.ts`, sg32 at D 128 and 64,
  wg at D 128). The codegen snapshots freeze the padded kernel at `obshd` sg32 D 128 and at `bhsd`
  wg D 64.
- **Regression** (same machine and day): the Wan2.2 sha256 rows of the 2-step seed path (reference
  seat at two sizes, `f16+dit8-a8-attn8-s16`, `f16+dit8-a8-attn8-online`,
  `i8-a8-attn8-online-conv8` with VAE tiles 16 and 32, the GPU text path) and every Anima sha256 row
  of `e2e_anima_test.ts` (the default preset `f16+dit8-a8-attn8-s16` at 1024² / 512² included)
  compare equal; the core lane is green.
- **Tests whose specification changed**: the assertions that a shape with `N % 4 != 0` is not
  online-eligible or falls back (the kernel test's eligibility case, `attention_route_test.ts`,
  `runtime_fusion_test.ts`, `runtime_pipeline_census_helper_test.ts`,
  `tune_attention_bench_test.ts`) now assert the online route there; their fallback cases use
  D = 96 instead. `attentionOnlineI8a8Params` takes the key-axis form, so its test callers pass
  it; the Session test matches the online key's reduce form by suffix with `:kpad` appended at
  `N % 4 != 0`.

## Addendum (2026-10-28) — `attentionDeviceFacts` on the runtime's public surface

- **What changed**: `@karume/runtime` exports `attentionDeviceFacts(gpu)` and its type
  `AttentionDeviceFacts` — the function Session construction already used to read decision 5's
  device facts (subgroups and subgroup size control enabled, WGSL `subgroup_id`, the measured
  linear lane mapping, the granted workgroup storage).
- **Why**: `EstimateOptions.attentionDevice` is public and required whenever
  `attentionScoreStorage` is `"none"`, but a caller outside the runtime could build it only by
  copying the condition (`GpuContext` exposes the features, the WGSL language features, the
  measured lane mapping and the limits it reads), and a copy would let an estimate count another
  route than the Session takes the moment the condition changes; the export keeps one derivation.
  The first caller is a model family's headroom check of a preset with the online kernel
  (MOSS-Image, `packages/models/src/moss/dit-stage.ts`). Given `"none"` without the facts, that
  estimate fails (it requires them); estimated without `"none"`, it counts the three-dispatch S:
  10,070,775,568 B instead of 7,333,404,976 B at 1024², and 12,519,585,664 B instead of
  10,113,779,296 B at its area cap — a need a 12 GiB device refuses although the pair it sizes
  fits there.
- **Surface**: the public-surface snapshot gains the two names (`KARUME_SURFACE=write`); no
  behaviour of the runtime changes.
