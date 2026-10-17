# 0131: bf16 weight storage — executed like f16, written when the round trip is exact

- Status: proposed (2026-10-17; to be accepted by the maintainer after review)
- Date: 2026-10-17
- Related: ADR [0006](0006-quantization.md) (storage-only quantization, compute in f32; bf16 was
  left "declared only") / [0018](0018-f16-weight-execution.md) (f16 storage: GPU-resident packed
  weights, unpack in the kernel, host expansion for the ineligible slots — the skeleton this ADR
  reuses) / [0019](0019-i8-weight-execution.md) and [0069](0069-packed-w4-storage.md) (the other
  storage paths; unchanged) / [0108](0108-container-format.md) (the codec ledger already registers
  `bf16`; addendum 12 made "declared but not executable" an explicit state) / [0128](0128-default-preset-original-weights.md)
  (the default preset question; its table listed only the DiT as BF16 — all four Anima components
  are) / [0127](0127-node-dawn-toolchain.md) (the Node port; the 2026-10-17 ruling that made
  bf16 support a phase-2 step) / [0074](0074-quant-seat-naming.md) (preset naming: the bf16 preset is
  the bare `<storage>` form `bf16`).

## Context

The container codec ledger has registered `bf16` since ADR 0108 (`packages/runtime/src/format/container/codecs.ts`,
mirrored in `tools/exporter/src/karume/container.py`), and the merge layer accepts `f32 × bf16`.
Nothing executed it: `RUNTIME_SUPPORT.storage` did not list it, so `prepareContainer` rejected a
bf16 initializer as a capability gap, and the exporter's `WEIGHT_DTYPES` could not write it.

Two facts changed the priority during the 0.15 migration:

- The maintainer's ruling of 2026-10-17 on the default preset (ADR 0128, rewritten in step
  2-7 the same day) asks for the "faithful" preset of every family to hold the values Karume did not quantize,
  and names bf16 explicitly: "support bf16 storage if feasible, computing in f32".
- The upstream Anima checkpoint (`circlestone-labs/Anima-Base-v1.0-Diffusers`) is BF16 in **all four
  components** — text encoder (310 tensors), text conditioner (118), transformer (567) and VAE
  (194), read from the safetensors headers on 2026-10-17. The shipped `f16` preset therefore rounds
  every component: bf16 and f16 share a 1-bit sign, but bf16 has 8 exponent bits and 7 mantissa
  bits where f16 has 5 and 10, so the conversion keeps the mantissa and loses range (values below
  2⁻¹⁴ become subnormal and lose precision, values above 65504 saturate — the exporter fails loudly
  on saturation). A bf16 preset is bit-identical to the upstream weights at the same size as f16
  (2 bytes per element, in download and in VRAM).

bf16 is the upper 16 bits of an f32. Expanding it is a bit copy (`bits << 16`), with no exponent
re-bias and no subnormal reinterpretation — exact on both the host and the GPU.

## Decision

### 1. The runtime executes bf16 on the f16 skeleton (ADR 0018)

- **Eligibility is the same single predicate as f16**: the weight slot of the fused kernels
  (`linear` / `conv1d` / `conv2d` / `conv3d` / `conv_transpose1d` / `embedding`), consumed nowhere
  else and not a graph output. Eligible weights stay resident as `array<u32>` (two elements per
  word, the same odd-length padding as f16 — `alignF16Payload` is shared); ineligible ones are
  expanded on the host at load (`decodeBf16`, `packages/runtime/src/format/bf16.ts`).
- **Unpack in the kernel by bit cast**: element `i` of word `w[i / 2]` is
  `bitcast<f32>(word << 16u)` for the low half and `bitcast<f32>(word & 0xFFFF0000u)` for the high
  half, selected by the parity of the **flat** index (the same trap as f16: a row-relative parity
  only shows for odd row lengths). The quad form for the GEMM v4 path builds a `vec4<f32>` from two
  words the same way. The GEMV family (M = 1 only, like f16) reads eight elements per 16-byte word.
- **Keys and snapshots**: the pipeline-key discriminator is `:wbf16`; the WGSL header note says
  "重み bf16 格納". The f32 / f16 / i8 generated text does not change by a byte (the existing
  snapshots are the detector); the bf16 variants have their own snapshots (`*_wbf16*.wgsl`, 21
  files).
- **Capability**: `RUNTIME_SUPPORT.storage` lists `bf16`; `capabilities().storage` reports it. The
  residency planner has a `bf16` seat next to `f16`; the estimate counts it as compressed resident
  bytes. The fusion rules treat bf16 like f16 (no activation quantization, no scale binding).
- Compute stays f32 (ADR 0006). The `f16` compute variant (ADR 0028) accepts bf16 weights the way
  it accepts f32 weights (the expanded value is rounded into the f16 tile).

### 2. The exporter writes bf16 under the f16 eligibility rule

`weight_dtype="bf16"` (and per-tensor overrides) stores a weight as bf16 only when the round trip
`f32 → bf16 → f32` is bit-exact, otherwise it stays f32 — the same rule as f16, so the stored bytes
are always the values the golden was computed with (ADR 0006, fake-quant first). Recipes round the
model with `round_weights_to_bf16` (round-to-nearest-even) before capturing references. The
rounding covers the parameters and the persistent buffers — what the checkpoint carries — and
leaves non-persistent buffers alone: those are computed in f32 at init (the RoPE `inv_freq`), the
upstream uses them in f32, and rounding one moved the Anima text conditioner's golden by 1.4e-4
while no parameter moved (measured 2026-10-17; the f16 rounding keeps touching them so the frozen
f16 series does not change). For a bf16 upstream checkpoint the rounding is therefore a no-op and
the asset holds the upstream bits. bf16 payloads are serialised through an `int16` view (numpy
has no bfloat16 dtype).

### 3. Anima ships a faithful preset `bf16`: all four components in bf16

The preset `bf16` (ADR 0074 grammar, bare storage name; label "Original weights (bf16)") stores
`text_encoder`, `text_conditioner`, `transformer` and `vae_decoder` as bf16 — the upstream bits. It
is opt-in; the default stays `f16+dit8-a8-attn8-s16` (ADR 0128 decision 1: the default is chosen per
model by procedure; the original-weights preset is best effort — decision 4). The preset was
shipped for the base model (`anima-v1.0`) first and extended to the other six models the same day
(addendum 2026-10-17): every Anima checkpoint — the official turbo / aesthetic variants and the two
fine-tunes in `anima-extra` — is a BF16 checkpoint of DiT + `llm_adapter` (no LoRA fusion is
involved since ADR 0087), so bf16 holds the upstream bits for all of them.

Under bf16 the recipe exports the `text_conditioner` of every model from its own checkpoint
(`text_conditioner_series`): the f16 preset of `anima-turbo-v1.1` shares the base conditioner
because the two are bit-identical after f16 rounding, but their upstream BF16 bits differ in 60 of
118 tensors (near-zero elements, at most 2.98e-8 — below f16's smallest step, 2^-24). The assembly
still folds byte-identical files into `shared/`: the text encoder and the VAE (untouched by every
fine-tune) are stored once, and so is the conditioner of the three official variants whose upstream
bits coincide (`anima-aesthetic-v1.1`, `anima-turbo-v1.0`, `anima-aesthetic-v1.0`).

### 4. Verification

- Host decode and GPU unpack agree on all 65,536 bit patterns (embedding of a `[256, 256]` bf16
  table): bit-exact except NaN (both sides NaN) and the patterns that are f32 subnormals (exponent
  0, mantissa ≠ 0 — 254 of them), which WGSL allows a GPU to flush to zero.
- The mirrored storage tests of `gpu_f16_i8_weights_test.ts` run for bf16 as a third entry
  (five fused kernels with odd row lengths, GEMM tile edges, pool poison, mixed consumption,
  diagnostics, graph output); the GEMV bit-identity and gate tests run for bf16 next to f16.
- The bf16 series of a recipe must produce the same goldens as its f32 series when the upstream is
  bf16 (the rounding is a no-op) — measured 2026-10-17 for Anima: the goldens of all four targets and
  of the S-form DiT are byte-identical to f32 references taken with the same code. The e2e sha rows
  of the bf16 preset are per environment like every other preset (ADR 0106).

## Consequences

- The runtime executes every codec in the ledger. `docs/limitations.md` loses "bf16 is declared
  only"; `docs/quantization.md` table 1 gains the row's status; `docs/ir-v2.md` and
  `docs/container-v1.md` drop the bf16 exception.
- Distribution size: an all-bf16 Anima preset downloads 5.06 GiB — the same as `f16`. The
  distribution repository gains a bf16 text stack and VAE (+1.41 GiB) and a bf16 DiT (+3.64 GiB) for
  the base model. With every model shipping bf16 (addendum 2026-10-17) the text encoder and the VAE
  sit under `shared/`, each model carries its own bf16 DiT, and the conditioner is folded only where
  two models hold the same upstream bits (local mirror: 35 → 50 GiB for `karume-anima`). VRAM
  equals the f16 preset.
- Geometry profiles (ADR 0115) are keyed by pipeline key, so `:wbf16` GEMMs fall back to the default
  geometry; the faithful preset is not tuned for speed.
- `CodecLayout` and `WeightStorage` both carry `bf16`; the residency and fusion types gained a
  member each, and the `never` switches surfaced every site that had to learn the new seat.

## Alternatives considered

- **Expand bf16 to f32 at load only** (no kernels): two files to change, but the DiT would occupy
  7.28 GiB of VRAM instead of 3.64 GiB, which excludes 8 GB cards from the faithful preset; adding
  the kernels later would mean writing the tests twice. Rejected.
- **Convert bf16 to f16 at load**: loses exponent range, so the preset would not be faithful.
  Rejected.
- **A DiT-only bf16 preset with the text stack left in f16**: smaller repository growth, but the
  text encoder, conditioner and VAE would still be rounded; the preset name grammar also has no
  place for a mixed 16-bit float storage. Rejected (decision 3).

## Open points

- gemma4 (upstream BF16, shipped as i4 only): a faithful preset also needs the PLE tables in bf16;
  deferred ("best effort", maintainer ruling 2026-10-17).
- Whether the geometry sweep should cover `:wbf16` GEMMs.
