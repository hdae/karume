> This document is a point-in-time snapshot (a read-only survey of 2026-11-16: the runtime at `6ec5679d`, llama.cpp at
> `ad5cbfa9`, ONNX Runtime at `7172420c`, and the headers and sampled blocks of twelve public GGUF files). Nothing here
> is decided and nothing was built or run on a GPU. It is not a design source of truth: GGUF is deferred and generating
> kernels is not a goal now (ADR [0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md),
> decisions 8 and 10). The six detailed reports are under `.claude/reviews/2026-11-16_kernel-generation/recon/` (local,
> untracked); sections 2 to 5 below are their facts, section 6 is inference.

# GGUF weights and the generation of quantized kernels: what each would take

## 1. What was asked

Whether reading GGUF model files needs many new quantized kernels, and so a generator in place of hand-written
kernels — or whether it can be done with the kernels the runtime has.

Short answer: for most of what public GGUF files hold, the weights can be converted at load into a storage form the
runtime already reads, without loss, and no kernel changes. What cannot is the family with a stored minimum per
sub-block (Q4_K and Q5_K, the bulk of every `Q4_K_M` file): the codec ledger forbids a zero point. A generator matters
only if those blocks are to be read in place.

## 2. What GGUF files hold

Read from ggml's block structs and `dequantize_row_*` routines
(<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c>) and from the headers of real files.

| Group | Types | Reconstruction of a weight |
| --- | --- | --- |
| Symmetric | Q4_0, Q5_0, Q8_0, Q3_K, Q6_K, Q1_0, Q2_0, TQ1_0, TQ2_0 | code × scale (the K types: code × sub-scale × f16 super-scale) |
| With a stored minimum | Q4_1, Q5_1, Q2_K, Q4_K, Q5_K | code × scale + minimum (Q4_K, Q5_K: − minimum) |
| Codebook | IQ4_NL, IQ4_XS, MXFP4, NVFP4 (16-entry tables); IQ1, IQ2, IQ3 (lattice grids of 1 to 16 KiB and a sign table) | table value × scale |

- Every multiply in these formulas is exact in f32 for every finite f16 scale and every code (about 1.4 × 10⁹ products
  enumerated, none inexact). The only rounding is the final add of the minimum, and it is the same with or without
  FMA. So a two-level scale flattens without loss into one f32 scale per group of 16 or 32.
- A `Q4_K_M` file of Llama 3.1 8B is Q4_K (193 tensors), Q6_K (33) and F32 (66); Qwen3 8B, Gemma 3 4B and Gemma 4 12B
  have the same pattern. Gemma 4 26B A4B's has Q8_0 (207), Q4_K (30) and Q5_1 (29): a row length that is not a multiple
  of 256 makes the quantizer fall back. Other mixes add IQ4_XS, IQ2, IQ3, Q2_K, Q3_K and Q5_K.
- Google's QAT file of Gemma 4 12B is Q4_0 with one Q6_K tensor (the token embedding). The code −8 is about 4.4 % of
  its codes.
- Blocks lie along the innermost axis and never cross a row. 17 of the 25 block types have a size that is not a
  multiple of four bytes (Q4_0 18, Q8_0 34, Q6_K 210), so reading them in place needs unaligned loads.
- The f16 super-scale is a subnormal in 9,983 of 9,986 sampled Q6_K blocks and 1,335 of 14,563 sampled Q4_K blocks.
  WGSL lets `unpack2x16float` flush subnormals to zero; whether Dawn / Vulkan, D3D12 or Metal does was not observed.
- llama.cpp's own CPU matmul quantizes the activations too (Q8_0, Q8_1 or Q8_K per weight type), so its outputs are
  not the product of dequantized weights with f32 activations. A parity target for GGUF weights can be defined exactly
  only as the latter.

## 3. What the runtime's storage allows

- The codec ledger has nine codecs (`packages/runtime/src/format/container/codecs.ts`): `f32`, `f16`, `bf16`, `i32`,
  `int8-sym` (one f32 scale per row), `int4-sym-g` and `int4-sym-g-f16` (one scale per group, a power of two of at
  least 16), `int2-off`, `ternary`. Every entry forbids a zero point. A file block is uploaded to the GPU as it is; the
  scale is its own block and buffer.
- The zero-point rule is ADR [0069](../decisions/0069-packed-w4-storage.md)'s ruling (symmetric only). Its stated reason
  is quality per byte on one sweep, not an obstacle in the format. ADR
  [0108](../decisions/0108-container-format.md) keeps two-level scales and codebooks out because they would break "one
  rounding, one f32 multiply" — a premise that section 2 shows does not apply to GGUF's two-level scales.
- GGUF Q4_0 stores element j in the low nibble and j + 16 in the high nibble and decodes (q − 8) × d; `int4-sym-g`
  stores 2i low and 2i + 1 high and decodes the same way. A nibble transcode at load reuses the existing decode.
- Adding a codec that only declares itself (`int4-sym-g-f16`) changed 29 files: both ledgers, descriptor, memory, bind,
  decode, residency, contracts, weight storage, emit, verify, dist, recipe tables, tests and docs — before any kernel
  reads it.

Where each GGUF type can go:

| Type | At load, into an existing form | Cost |
| --- | --- | --- |
| F32, F16, BF16 | As they are | None |
| Q4_0 | `int4-sym-g` with group 32 (nibble transcode, f16 scale widened to f32) | None; needs code −8 accepted (section 7) |
| Q8_0 | No existing form: `int8-sym` has one scale per row, not per 32 | A grouped int8 codec, or f16 at twice the size |
| Q3_K, Q6_K, Q5_0 | Codes widened to int8 with one flattened f32 scale per 16 or 32 — the same missing grouped int8 | 6.5 → about 10 bits per weight for Q6_K |
| Q4_K, Q5_K, Q4_1, Q5_1, Q2_K | None without a minimum: f16 / bf16, or a new codec that carries one | f16: 4.5 → 16 bits per weight |
| IQ and FP4 types | Expanded to int8 codes × one f32 scale per 16 or 32 (grouped int8), without loss | 2 → 9 bits per weight for IQ2_XXS |

## 4. The quantized kernels as they are

- WGSL comes from 55 files of the runtime (23,878 lines), two rules in `packages/models` and three probes. 19 builders
  compute an inner product on quantized operands; 12 of them carry a hand-written quantized inner loop (the GEMV at
  M = 1, by rows, in parallel, with subgroups; the expert linear; the byte-plane and the integer GEMV; the int8 linear
  with int8 or int4 weights; the int8 attention stages; the int8 conv3d; the online int8 attention). The other seven
  splice a shared fragment.
- Two "decode × skeleton" structures already exist. `weightLoaderWgsl`
  (`packages/runtime/src/kernels/weight-storage.ts`) emits a per-storage `dequant` that the GEMM tile fills, the
  embedding and the direct convs splice; the GEMM skeleton never sees the storage. `unitMacs`
  (`packages/runtime/src/kernels/linear-gemv.ts`) emits "decode and multiply-add one 16-byte word" for four GEMV
  skeletons and the expert linear. The int8-activation kernels share the integer dot, the accumulator and the inner
  loop, and write their fills, flushes and stores by hand.
- One storage form has several decode spellings, each tied to its own numeric contract: five for int4 and four for
  int2. Where the scale enters (fill, product, flush or store) differs by family, and a grouped scale changes the loop
  nest. The integer GEMV, the byte-plane GEMV and the int8-activation fills do not split into "one decode per storage".
- Both shared decodes take a flat element index and a scale handed in by the caller. A block with sub-block scales and
  a minimum does not fit that contract.

## 5. What a kernel must plug into

- Selection: `linear` is chosen by one ordered table of 12 strategy ids
  (`packages/runtime/src/runtime/linear-strategies.ts`); 11 are still built by device-bound functions. The
  device-free dispatch-program form has no operand for a weight scale or an activation scale, so no quantized kernel
  can use it yet.
- The estimate, the census and the verification cache do not read that table; the `linear` → `static_quantize` fusion
  rule keeps its own copy of the conditions.
- What pins a kernel: 357 WGSL byte fixtures in one enumerated list (of the 65 for `linear`, 17 are reference kernels),
  12 kernel identity checks registered by hand, per-family fingerprints in three geometry tables, and hand-written
  shape tables of 66, 5 and 1 rows that no sweep measures. The generated elementwise chain of
  `packages/models/src/moss/ew-chain.ts` is the one precedent for property tests in place of byte fixtures.
- Levels: a GEMM for a new storage would be that storage's reference kernel; a GEMV that gives the same bits is
  `exact`; a split reduction is `reorder` and needs the four-point check of the split-K linear; a dot product on
  quantized activations is a named approximation. The band of `linear` was derived at K = 72 and says to be derived
  again for a much larger K.

## 6. How others do it

- Every ggml GPU backend that reads GGUF blocks (Vulkan, Metal, WebGPU, SYCL, CUDA) uses one tiled matmul skeleton with
  a small decode per format that writes into a shared tile. Matrix-vector is different: only Q4_0 to Q8_0 share a
  generic body; the K and IQ types each have a hand-written one, and the speed-ups in their pull requests came from
  changing the load shape per format.
- ggml's WebGPU backend builds variants at run time with its own preprocessor, moved quantized buffers to flat u32
  (about 10 % decode cost on unaligned blocks), and tests against the CPU by normalised error, not by bits.
- ONNX Runtime's WebGPU `MatMulNBits` covers 2, 4 and 8 bits with build-time templates and a fixed selection chain.
  MLC compiles the decode as a tensor expression but handles a few uniform group formats and no GGUF. The Hugging Face
  WebGPU kernel collection makes variant selection data (predicates, priorities, tolerances) and has no GGUF format.
- Transcoding at load is common (ggml's OpenCL and SYCL backends, MLX: Q4_0, Q4_1 and Q8_0 stay quantized, everything
  else becomes f16).

## 7. Reading (inference)

- GGUF support does not need a kernel generator. It is a reader, a name map and a conversion at load, on the path of
  "weights from outside the exporter" (ADR 0152 decision 10), plus two storage decisions: a grouped int8 form, and
  whether a minimum is admitted for foreign formats. With both, every GGUF type has a lossless home at about 5 to 10 bits
  per weight; without the minimum, Q4_K and Q5_K fall back to f16.
- A new grouped storage form costs kernels where decode is hand-spelled: the GEMV family and the int8-activation
  family, not the GEMM. That is where one decode source per storage form would pay, GGUF or not.
- Reading K-type blocks in place is the only case that needs per-format kernels, and the stacks that did it wrote the
  fast matrix-vector kernels by hand per format.
- Byte fixtures of non-reference strategies make any regrouping of decode fragments a rewrite of fixtures; ADR 0152
  decision 8 already says those become property tests.

## 8. Found on the way (not acted on)

- `docs/container-v1.md` says `int4-sym-g` codes lie in [−7, 7] with −8 unused; the distributed Gemma 4 QAT q4_0
  containers store −8 (`tools/export-recipes/gemma4/export_unified.py`), and nothing checks the range.
- `docs/container-v1.md` still counts four codecs and three predicates.
- The f16 group scales of `int4-sym-g-f16` are read by the expert linear only; whether a backend flushes f16
  subnormals in `unpack2x16float` matters to it as well.
- `PARALLEL_SHAPES` has 66 rows (58 int4, 6 int8, 2 int2), 22 of them copied for another model size, not measured.
- No kernel identity check compares the sequential GEMV with the GEMM, and none covers the split-K linear.

## Not covered

The GGUF container's metadata and tensor name maps; tokenizers in GGUF; ggml's CUDA, Vulkan and Metal kernel bodies
beyond their structure; any measurement of speed or of subnormal handling on a GPU; the fused attention's selection;
the conv, norm and expert-linear selection; a design.
