# Anima export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and the authoritative model card
is the exemplary `README.md` that `dist` generates into the distribution directory.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Scripts are started as
modules from `tools/export-recipes/` (`uv run --group anima python -m anima.export …`); the generic emit, verify
and dist contracts live in [`../../exporter/README.md`](../../exporter/README.md).

## Real-weight Anima export

Real weights for the image generation side. `anima/export.py` writes out the 4 emit targets of ADR
[0016](../../../docs/decisions/0016-anima-chain-export.md). The full emit (`outputs/series/anima/`) and the
Deno-side E2E (`packages/models/tests/e2e_anima_test.ts`) were completed in M1-P4.

### Dependencies and obtaining the weights

```bash
uv sync --group anima   # accelerate / diffusers==0.39.0 / torchvision / transformers==5.14.1
```

The weights are `circlestone-labs/Anima-Base-v1.0-Diffusers` on the HF Hub (downloaded automatically
on the first run; 5.3GB). `diffusers` is pinned with `==` because `anima.patch` replaces the
forwards of `QwenImageRMS_norm` / `QwenImageResample` / `QwenImageUpsample` /
`QwenImageAttentionBlock` at the class-attribute level and carries wrappers that transcribe the
forwards of `AnimaTextConditioner` / `CosmosTransformer3DModel` line by line (a minor update would
silently change the graph shape or the premises of eager equivalence).

### Generation and comparison

```bash
# emit (IR + golden io into <out>/<target>/)
uv run --group anima python -m anima.export --out /path/to/out
uv run --group anima python -m anima.export --target vae_decoder --out /path/to/out
uv run --group anima python -m anima.export --target transformer --num-layers 2 --out ...

# eager equivalence across the patches (**one target per process** — the CLI rejects combining them)
uv run --group anima python -m anima.export --verify text_encoder
uv run --group anima python -m anima.export --verify vae_decoder

# fuse a LoRA before emitting (applies to transformer / text_conditioner)
uv run --group anima python -m anima.export --target transformer \
  --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors

# S form (one symbol for the token length), an additional series — transformer only; the default out gets -dyn
uv run --group anima python -m anima.export --dtype f16 --dit-graph dyn \
  --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors \
  --out ../../outputs/series/anima-turbo-f16-dyn
uv run --group anima python -m anima.export --dtype f16 --dit-graph dyn --verify transformer \
  --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors
```

- **`--verify` and `--target` cannot be combined in the same process.** The VAE patches replace
  class attributes process-wide, so if the emit side applies them first the "pre-patch reference" is
  contaminated and the equivalence check becomes vacuously true (the difference is always 0 = a
  breakage where green is no longer evidence — the discipline of ADR 0013).
- **`--lora` is fused before `--num-layers`.** The other way around, the LoRA belonging to the
  truncated layers would be silently discarded as "no target", and the leftover check of `fuse_lora`
  would stop working on the reduced model.

### `--dit-graph dyn` (the S form of the DiT — #21 wave T2)

An **additional series** that moves the graph entrance **after** the patchify and promotes the rope
cos / sin tables to graph inputs (the static series does not move by a single byte). The entrance is
`tokens [1,S,68]` plus two rope tables `[1,1,S,128]`, and the exit is `[1,S,64]` before the
unpatchify. **Zero resolution-dependent bakes remain** (the static form has 3: the padding channel
and the rope tables). The design rationale is in
[dynres-vae-tiling](../../../docs/research/2026-08-03-dynres-vae-tiling.md) §2.2.

- **transformer only.** The other 3 targets do not depend on the resolution and share the static
  series (the CLI rejects specifying them). `--resolution` **has no effect and is therefore
  rejected** too — the golden resolutions are fixed at `DIT_DYN_RESOLUTIONS = (512, 1024)` and the
  graph itself carries no resolution.
- Alongside `model.safetensors` / `io.*`, the series directory holds **`rope_base.safetensors`**
  (64KiB). It is the **per-axis base table** the host (`packages/models/src/anima/dit-tokens.ts`)
  uses to assemble the rope tables, cut out of the `model.rope` output. **Why it is needed**: torch's f32
  trigonometric functions can be 1 ulp off from the correctly rounded value (measured: over 8,192
  position × frequency combinations, cos 472 and sin 231 cases), and JS's `Math.cos` cannot
  reproduce them. The static graph has torch's values baked in, so bit identity is only achievable
  by permuting the base table. The base table is **independent of the resolution**, and its number
  of rows (= the length of the upstream `seq = arange(max(max_size))`) is the model-side upper bound
  (for Anima, 128 = latent 256 = the equivalent of 2048px).
- `--verify transformer --dit-graph dyn` compares "host patchify → S form → host unpatchify" against
  the **pre-patch diffusers path**. Measured (with the turbo LoRA fused and f16 rounding applied):
  **`bit_exact=True` / maxdiff 0.000e+00 in both cases** (S=1,024 and S=4,096).
- The "S form ≡ static graph, exact Uint32 equality" real-GPU gate was a dedicated E2E of the
  pre-migration tree; it has **not been ported into this repository yet**. The current in-repository
  Anima gate is `packages/models/tests/e2e_anima_test.ts` (exact PNG SHA-256 equality).

### Measurements (as of wave 2; DiT / Qwen3 with `--num-layers 2`, the rest full)

| Target             | IR nodes | model.safetensors | symbols       | max rank |
| ------------------ | -------- | ----------------- | ------------- | -------- |
| `text_encoder`     | 131      | 749.8MB           | `T`           | 4        |
| `text_conditioner` | 613      | 539.1MB           | `Tsrc` `Ttgt` | 4        |
| `transformer`      | 316      | 629.4MB           | (static)      | 4        |
| `vae_decoder`      | 455      | 101.3MB           | (static)      | 4        |

**Unsupported aten ops: 0 kinds in all 4** (at the start of wave 2 the union was 19 kinds — recon
§2). `text_conditioner` / `vae_decoder` are not truncated, so this table is already the full depth
for them; `text_encoder` was separately measured at 0 kinds for the full 28 layers as well (2467 FX
nodes after normalization). The full 28 layers of the DiT are 7.29GiB in f32, so they are measured
in wave 3 — the layer count does not change the **kinds** of unsupported ops (the same block
repeated).

### Measurements (wave 3, **full depth**; all 4 into `outputs/series/anima/`)

Measured with each target in a **separate process** (so that host RAM peaks do not overlap; listing
several `--target`s in one process also works, but the DiT peak would stack on top of the other 3
staying resident).

| Target             | IR nodes | initializers | model.safetensors | emit time | peak RAM  |
| ------------------ | -------- | ------------ | ----------------- | --------- | --------- |
| `text_encoder`     | 1769     | 317          | 2,386,195,204 B   | 17.5s     | 3,794MiB  |
| `text_conditioner` | 613      | 122          | 539,060,388 B     | 5.7s      | 1,152MiB  |
| `transformer`      | 3904     | 579          | 7,827,646,080 B   | 36.5s     | 11,593MiB |
| `vae_decoder`      | 455      | 108          | 101,279,604 B     | 10.0s     | 1,546MiB  |

`model.safetensors` totals 10,854,181,276 B, and `outputs/series/anima/` as a whole, including
`io.<case>.safetensors`, is 10,868,931,292 B (10.12GiB). **`outputs/` is under `.gitignore`** (never
committed). The op vocabulary is **23 kinds** across the 4 targets (`add bmm cat clamp clamp_min
conv2d div embedding expand gelu layer_norm linear mul neg permute reshape rms_norm sigmoid slice
softmax sqrt sum sym_prefix_slice`), with **zero new ops** (the 3 kinds added in wave 1 sufficed).

**Regeneration determinism**: `vae_decoder` was emitted a second time into a different directory and
all 3 files — `model.safetensors` / `io.case0` / `io.case1` — were confirmed to have **matching
sha256** (the randomness is derived from `SEED` and does not depend on the global seed). The 3 large
targets are unconfirmed for reasons of emit time and disk space — the randomness path is the same
implementation in all 4, so it is stated explicitly that the confirmation covers `vae_decoder` only.

### Measurements (wave 3; the `--verify` eager equivalence is unchanged at full depth)

The current in-repository Anima migration gate is `packages/models/tests/e2e_anima_test.ts`. It
requires **exact PNG SHA-256 equality and explicitly forbids tolerance-based acceptance** (replacing
the reference values is forbidden too). The per-target `maxAbs` figures below are **historical
measurements** from the pre-migration numerical E2E; they are not the current acceptance authority.

| Target             | Real-GPU golden          | Notes                                                               |
| ------------------ | ------------------------ | ------------------------------------------------------------------- |
| `text_encoder`     | maxAbs 5.22e-4 (3 cases) | 130,816 folded -inf values with **0 NaN**                           |
| `text_conditioner` | maxAbs 2.23e-6 (2 cases) | Tsrc/Ttgt varied separately to kill mix-ups                         |
| `transformer`      | **not run**              | the 7,465MiB of weights exceed this machine's GPU limit of 7,280MiB |
| `vae_decoder`      | maxAbs 1.01e-5 (2 cases) | same order as the 9.34e-6 of `--verify`                             |

`transformer` being unrun is **not a graph problem**: what is emitted with `--num-layers 20`
(5,613MiB of weights) runs to completion on a real GPU and agrees with torch (maxAbs 9.63e-5).
Details in `docs/known-issues.md`.

Eager equivalence across the patches (`--verify`):

| Target             | Cases | worst maxdiff         | Notes                                                    |
| ------------------ | ----- | --------------------- | -------------------------------------------------------- |
| `text_encoder`     | 3     | **0 (bit-identical)** | all-ones mask dropped                                    |
| `text_conditioner` | 2     | **0 (bit-identical)** | two all-ones masks dropped                               |
| `transformer`      | 2     | **0 (bit-identical)** | timestep promotion / padding turned into a zero constant |
| `vae_decoder`      | 2     | 9.34e-6               | conv3d → conv2d reduction order difference (below)       |

The VAE alone is not bit-identical because CausalConv3d(T=1) → conv2d becomes **the same number of
additions performed in a different order**. Equivalence per rewrite is pinned by
`tests/test_anima.patch.py` at `< 1e-14` in f64 (conv3d↔conv2d / channel L2↔`F.normalize` / rank-4
form of RMS_norm). The two nearest-exact resamples, which only move data, are **bit-identical** in
f32. The amplification factor of the whole decoder is about 5.4e3 as measured in f64, so f32
rounding (1.19e-7) growing into the 9e-6 class is consistent.

> NOTE: unlike the f64 unit checks above, running `--verify` in f64 does not drive the residual to 0
> (measured 9.1e-8). The cause is that `QwenImageUpsample.forward` in the original implementation
> drops to f32 with `x.float()` — on the f32 path (= the path that gets exported) `.float()` is a
> no-op, so it is bit-identical there.

### f16 storage emit (`--dtype f16` — ADR 0018)

`--dtype f16` takes the references and goldens **after rounding the weights to f16-representable
values** (fake-quant) and stores **only the eligible weight slots** in f16. The output goes to
`outputs/series/anima-f16/`, separate from the f32 series (the default for an omitted `--out` switches with
`--dtype`).

```sh
uv run --group anima python -m anima.export --dtype f16                    # all 4 at once
uv run --group anima python -m anima.export --dtype f16 --target transformer
uv run --group anima python -m anima.pipeline_ref --dtype f16                  # fixture
```

**MUST: the rounding happens before the reference and golden capture** (ADR 0006). `_fake_quant` is
applied right after each builder assembles the model (and **after** any `--lora` fusion). Moving it
later would leave only the reference computed on the original weights, making the E2E difference a
mixture of "quantization error + implementation error" — which only ever loosens the tolerance, i.e.
a breakage that **stays green while losing detection power**.

Which initializers are eligible for f16 storage, and the write order that keeps the container
readable, are the **core** emit contract — see "Compressed weight storage" in
[`../../exporter/README.md`](../../exporter/README.md). The measurements below are what that
contract was exercised against on Anima.

#### Measurements (2026-08-03, `--dtype f16` at full depth, separate processes)

| Target             | model.safetensors | vs f32 | Eligible (f16 storage)  | Ineligible (f32 storage) | emit time | peak RAM  |
| ------------------ | ----------------- | ------ | ----------------------- | ------------------------ | --------- | --------- |
| `text_encoder`     | 1,194,225,580 B   | 50.0%  | 197 tensors / 1,192.0MB | 120 tensors / 1.86MB     | 19.0s     | 4,292MiB  |
| `text_conditioner` | 269,838,164 B     | 50.1%  | 62 tensors / 269.2MB    | 60 tensors / 0.48MB      | 6.2s      | 1,360MiB  |
| `transformer`      | 3,914,867,592 B   | 50.0%  | 454 tensors / 3,912.8MB | 125 tensors / 1.22MB     | 42.2s     | 11,807MiB |
| `vae_decoder`      | 50,732,956 B      | 50.1%  | 37 tensors / 50.5MB     | 71 tensors / 0.075MB     | 10.4s     | n/a       |

`outputs/series/anima-f16/` as a whole is 5,444,414,308 B (5.07GiB; the f32 series is 10.12GiB). The rounded
weights amount to 5.96e8 elements for text_encoder, 1.35e8 for text_conditioner, 19.56e8 for
transformer and 1.27e8 for the vae. **The ineligible bytes are under 0.5% for every target**
(biases, norm weights, folded constants), so the constraint "ineligible means zero VRAM reduction"
does no real harm for Anima.

Measured `Session.diagnostics().storage` (Deno side) — `residentCompressedBytes` is the number of
bytes that stayed compressed on the GPU, `hostExpandedBytes` is the part of the f16 declarations
that was expanded to f32 at load time:

| Target             | resident   | hostExpanded |
| ------------------ | ---------- | ------------ |
| `text_encoder`     | 1,136.8MiB | 0.0MiB       |
| `text_conditioner` | 256.8MiB   | 0.0MiB       |
| `transformer`      | 3,731.5MiB | 0.0MiB       |
| `vae_decoder`      | 48.2MiB    | 0.0MiB       |

`hostExpanded` being 0 everywhere is **by design**: the exporter only **declares** f16 for what
passed the eligibility check, so the runtime's "declared f16 but ineligible → expand on the CPU"
path is never entered (that path itself is reachable with hand-written IR, and
`packages/runtime/tests/gpu_f16_weights_test.ts` pins it in isolation).

**This is what put the full 28-layer DiT on a real GPU**: the 7,465MiB of f32 exceeded the GPUBuffer
ceiling of 7,280MiB (`docs/known-issues.md`) and could not be loaded, whereas f16 is 3,731.5MiB,
half the ceiling. The real-GPU golden and the stage-② numbers of the end-to-end chain (DiT golden
maxAbs 6.68e-5, stage ② raw DiT output 3.03e-5, stage ③ end-to-end 6.41e-6) are **historical
measurements** from the pre-migration numerical E2E. The current gate is
`packages/models/tests/e2e_anima_test.ts` (exact PNG SHA-256 equality; tolerance-based acceptance
and reference-value replacement are forbidden).

`anima/pipeline_ref.py --dtype f16` fake-quants **all 4** components before taking the references (if
even one were left unrounded, that stage alone would be the numbers of a different model). The
output is `outputs/series/anima-pipeline-f16/` (21 tensors, 9.4MB), measured at 44s and 13.5s per DiT step.

#### Measurement: Anima's weights are already BF16, so f16 rounding is near-identity (2026-08-03)

In HF's `circlestone-labs/Anima-Base-v1.0-Diffusers`, **all 4 components are `BF16` in the
safetensors** (`torch_dtype` in `text_encoder/config.json` is `bfloat16` as well). BF16's 8-bit
mantissa fits inside f16's 11 bits, so promoting to f32 and rounding to f16 only moves values that
**fall into f16's subnormal range**:

| Component          | Elements moved by rounding    | Max change |
| ------------------ | ----------------------------- | ---------- |
| `vae`              | 163,271 / 1.27e8 = **0.129%** | 2.98e-8    |
| `text_conditioner` | 16,718 / 1.35e8 = **0.0124%** | 1.59e-4    |

**This fact constrains how the f16 series may be read**: the f16 and f32 series having errors of the
same order was not because "quantization is harmless" but because f16 was applied to a model that
was BF16 to begin with — for f32-trained weights the story is different.

**Fault injection results (important — not what was expected)**: actually injecting "move the
fake-quant after the reference capture" gave the following.

| Injection                                                         | Result                                                                                                                                |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| ① remove `_fake_quant` from the builder                           | **emit fails loudly** (`EmitError: initializer 'p_vae_decoder_conv_in_weight' … fake-quant が未適用か、参照採取より後に掛かっている`) |
| ② ① plus removing the eligibility check (round-trip bit equality) | the assets are written but **the E2E stays green** (vae_decoder maxAbs 6.85e-6 / text_encoder 4.40e-4 — both inside the tolerance)    |

② failing to turn red is a consequence of the BF16 fact above: the difference in torch's own output
across the rounding is itself small (measured: 2.64e-3 for text_encoder t024 = 3.8e-5 relative to a
range of 68.7, and 1.72e-5 for vae_decoder). In other words, **for Anima the E2E is not a detector
for a forgotten rounding** — the only thing detecting it is the emit-side eligibility check. Do not
treat this gate as redundant and remove it.

> NOTE (2026-08-03, measured against SBV2 f16 as a control): this "invisible to the E2E" property is
> **a property of the model (its distribution dtype)**, not of f16 storage in general. For SBV2,
> whose ckpt is genuinely f32, the same injection turns **both** the emit gate and the E2E red
> (exceeding atol by more than 31× — ADR 0027).

### i8 storage emit (`--dtype i8` — ADR 0019)

`--dtype i8` fake-quants with **per-channel symmetric int8** (no zero point) before taking the
references and goldens, and stores the eligible weight slots as i8 plus companion scales. The output
goes to `outputs/series/anima-i8/`.

```sh
uv run --group anima python -m anima.export --dtype i8              # transformer only
uv run --group anima python -m anima.export --verify transformer --dtype i8
uv run --group anima python -m anima.pipeline_ref --dtype i8            # fixture
```

**MUST: `--dtype i8` is transformer-only** (`DTYPE_TARGETS` — the CLI rejects other targets even
when spelled out with `--target` / `--verify`). Two reasons:

1. **Series design** (ADR 0019): the DiT's −1.87GiB is the dominant term, while text / cond / VAE
   share `outputs/series/anima-f16/`. Prototype measurements put i8 for the VAE two orders lower.
2. **The VAE cannot satisfy the rounding order constraint**: `anima.patch` replaces the CausalConv3d
   weights with **the last slice along the time axis**, which happens when the patches are applied
   (= after the reference capture). f16's element-wise rounding commutes with slicing and is
   therefore harmless, but i8's per-channel scale **counts even the discarded elements into amax**
   (a shifted scale moves the value of every element).

The definition of the per-channel quantization, the eligibility check, the companion scales and the
I8 write order are the **core** contract (same section of
[`../../exporter/README.md`](../../exporter/README.md)). One consequence is Anima-specific: since
matching is by FQN, **`--dtype i8` applies the rounding to the exported wrapper (`AnimaDit`)**
(applying it to the inner `model` would make the FQN prefixes disagree).

#### Measurements (2026-08-03, `--dtype i8` at the full 28 layers, separate process)

| Metric                          | Measured                                                 |
| ------------------------------- | -------------------------------------------------------- |
| `model.safetensors`             | 1,963,762,200 B (**1,872.8MiB**)                         |
| vs f32 / vs f16                 | **25.09%** / **50.16%** (f32 7,465.0MiB, f16 3,733.5MiB) |
| Eligible (i8 storage)           | 454 tensors / 1,956.4MB (19.56e8 elements)               |
| Companion scales                | 5.19MB = **0.265%** of the eligible bytes                |
| Ineligible (f32 storage)        | 125 tensors / 1.22MB                                     |
| emit time / peak RSS            | 44.3s / 11,593MiB                                        |
| `--verify transformer`, 2 cases | maxdiff 0.000e+00, **bit-identical in every case**       |

`Session.diagnostics().storage` (Deno side) reports `residentCompressedBytes` **1,961,579,776 B
(1,870.7MiB)** and `hostExpandedBytes` **0**. The former matches the exporter's `compressed_bytes +
scale_bytes` (1,956,388,864 + 5,190,912) **byte for byte** — a measurement that "adding the scales
into the diagnostics" means the same thing in the exporter and the runtime.

The scale overhead is **0.265%**, smaller than the 0.4–0.9% estimated in ADR 0019, because the DiT's
Linear weights are `[Cout, Cin]` with Cin as large as 1024–4096 (the scales only need Cout entries).

`anima/pipeline_ref.py --dtype i8` rounds **the DiT in i8 and the other 3 components in f16** before
taking the references (`COMPONENT_DTYPES`). This is to keep a one-to-one correspondence with the
asset series (`outputs/series/anima-i8/transformer` plus the other 3 from `outputs/series/anima-f16/`); making
everything i8 would make the text-path references the numbers of a different model than the assets
actually being executed. The output is `outputs/series/anima-pipeline-i8/` (21 tensors, 9,873,808 B),
measured at 14.1s / 14.4s per DiT step.

The real-GPU E2E numbers (DiT golden maxAbs 8.59e-5 / stage ② raw DiT output 5.34e-5 / stage ②
latents 1.19e-6) are **historical measurements** from the pre-migration numerical E2E; the i8
variant of that test has not been ported into this repository, and the current gate is
`packages/models/tests/e2e_anima_test.ts` (exact PNG SHA-256 equality; tolerance-based acceptance
and reference-value replacement are forbidden). **The f16-series values were not reused** — i8
rounding produces different weights, so even the implementation error stemming from reduction order
is a different quantity.

#### Fault injection results (2026-08-03, pytest + tiny golden regeneration)

Breaking exactly one thing and then running `uv run pytest` and `python -m karume.goldens` in `tools/exporter/`
(the harness always restores, and the post-restore baseline of 1,602 passed was re-confirmed).

| Injection                                                         | Result                                                                                                                                          |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| ① broken rounding order (fake-quant after the export)             | pytest **208 errors** / golden generation stops with `EmitError: … per-channel scale が無い`                                                    |
| ② recomputing the scale (axis fixed to 0)                         | pytest **2 failed** / golden generation stops with `EmitError: … 逆変換してもビット一致しない` (the failure is `up.weight` = transposed axis 1) |
| ③ recomputing the scale (**correct axis**)                        | pytest **2 failed** (the negative tests detect that the gate itself is gone). **The goldens regenerate byte-identically**                       |
| ④ letting −128 in (`scale = amax/128`, −128 on the negative side) | pytest **5 failed** (±127 / idempotence / scale definition / golden determinism / scale fixed point)                                            |
| ⑤ broken write order (I8 moved into the leading group)            | pytest **3 failed** / golden generation stops with `ContainerError: … 絶対 offset 4175 が F32 の要素サイズ 4 に整列していない`                  |

**③ is the important one (not what was expected)**: "re-deriving the scale from an
already-fake-quantized weight on the correct axis" **changes nothing in the data**. Because `q` is
closed to ±127, the element with the largest absolute value necessarily lands on `q = ±127`, and
`amax(|q·s|)/127 = fl(fl(127·s)/127) = s` is a **fixed point** in f32 (zero counterexamples over
8.9e7 random samples). So ADR 0019's "no recomputation" is a discipline the inverse-transform gate
cannot detect, and the reason to keep it lies elsewhere: when "the weight at emit time is not the
effective weight / the axis is wrong / the formula is wrong", the scale silently becomes a different
one (② is exactly that case). This fixed-point property is pinned by
`tools/exporter/tests/test_emit.py::test_recomputing_the_scale_from_a_quantized_weight_is_a_fixed_point`.

### Patch layer (`anima/patch.py`)

Unlike `sbv2.patch`, the goal is **not exportability but IR quality** (ADR 0016 / recon §5). All 4
targets export even unpatched, but as-is the vocabulary would need conv3d / `linalg_vector_norm` /
`upsample_nearest2d`, the rank would go up to 8, and the timestep embedding would be baked into the
graph.

| Subject     | Rewrite                                                                                                                              |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Qwen3       | do not pass `attention_mask` (an all-ones mask ⇒ the causal mask alone, with an identical additive bias)                             |
| Conditioner | do not pass the two all-ones masks. The 512 padding and the output mask multiplication are host-side                                 |
| DiT         | promote the timestep embedding to a graph input / turn `padding_mask` into a zero-constant channel / move to rank 4                  |
| VAE decoder | CausalConv3d(T=1) → conv2d / nearest-exact ×2 → reshape+expand / channel L2 → last-dim sum + `clamp_min` / `feat_cache` fails loudly |
| Shared      | demote RoPE's `inv_freq` from a buffer to a plain attribute (making it a leaf for constant folding)                                  |

The `--verify` references are taken **before the patches** (with `vae_patches_applied()` as the gate
for the VAE). The reference is `vae.decode` itself — one comparison measures both "conv3d ⇒ conv2d
equivalence" and "at T=1 the feat_cache does not affect the result" at the same time (checking one
of them through a separate path would leave room for the other's premise to break silently).

### LoRA (`anima/lora.py`)

ΔW=(B@A)·scale is **computed in f32** and added in place to the original dtype. The IR does not
change by a single node and no runtime implementation is required (ADR 0016). Name conversion calls
`_convert_non_diffusers_anima_lora_to_diffusers` bundled with diffusers (no hand-maintained mapping
table). There are 4 fail-loudly conditions: `lora_B` being all zeros for the whole file / 0 targets
matched / one of `lora_A`, `lora_B` missing / a shape mismatch. A per-component ΔW=0 can
legitimately happen, so that is reported through `FuseReport.is_noop` instead (not raised).

## Anima host pipeline reference fixture (`anima/pipeline_ref.py`)

Only 4 graphs go into the IR; everything outside them — tokenization / sigma schedule / timestep
embedding table / CFG / Euler update / latent denormalization / 512 padding — is host code. This
script pins **the authoritative numbers for that part** into a single fixture. The authority is the
4 blocks of `modular_pipelines/anima/` in diffusers 0.39, taken through the **plain diffusers path
without the patch layer** (patch-layer equivalence is measured separately by `anima/export.py
--verify` — keeping the verification nets independent; going through the patches here as well would
put a patch bug into both the reference and the subject under test in the same shape, letting it
pass with a difference of 0).

```sh
uv run --group anima python -m anima.pipeline_ref                    # into outputs/series/anima-pipeline/
uv run --group anima python -m anima.pipeline_ref --steps 32 --ref-steps 2
uv run --group anima python -m anima.pipeline_ref --resolution 1344x768 …   # non-square (#23)
```

- The outputs are `outputs/series/anima-pipeline/pipeline.safetensors` (21 tensors, 9.4MB) and
  `pipeline.json` (prompt, step count, shift, CFG coefficient, and the role and shape of every
  tensor).
- **Not placed directly under the distribution tree `models/anima-turbo/`** — that one holds exactly
  the files the manifest declares and is uploaded to HF as-is, and an undeclared file stops
  `verify_dist` (the same reason `outputs/sbv2-demo/` is kept separate).
- The prompt is a single fixed English string (danbooru-style tags). **The negative prompt is not
  the empty string** — an empty T5 id sequence has length 1 and falls outside the conditioner's
  accepted set `Dim("Ttgt", min=2)`.
- `latents_init` uses the fixed `SEED = 20260802` (independent of the global seed).
- **`--resolution` takes `WxH`** (square may be abbreviated; the spelling is the same as the demo's
  `--resolution`, and `anima/resolution.py` is authoritative). For non-square, the **axis order**
  of `latents_init [1,16,H/8,W/8]` and `padding_mask [1,1,H,W]` is the one pitfall, since swapping
  them still gives a matching element count. The metadata carries both the spelling (`resolution`)
  and `width` / `height` — **readers should use the latter** (`resolution` stays an int for square,
  for compatibility with tests that read existing fixtures).
- The raw DiT outputs (`noise_cond_*` / `noise_uncond_*`) are kept as well. Without them the Deno
  side **cannot parity-check the CFG and Euler host glue in isolation** and can only see it mixed
  with the DiT's error. On top of that, the σ step is only `sigmas[1] − sigmas[0] = −1.064e-2`, so
  the DiT error is diluted roughly 100× by the Euler update (a detector that only looks at the
  latents is structurally blunt).
- `image` is the return value of `vae.decode` itself. `AutoencoderKLQwenImage._decode` applies a
  final `clamp(-1, 1)` (this clamp is what `AnimaVaeDecoder` bakes in — it does not come from
  postprocessing), so the clamp sits at the same position on the fixture side and the IR side.

Measured (`--steps 32 --ref-steps 2`, 512px): **44.0s, peak RAM 12,918MiB** (4 DiT runs on CPU f32 =
2 steps × cond/uncond; 13.5s per step). On the text side, qwen 29 and t5 30 tokens (negative 13 /
21).

The Deno-side host glue implementation (`sigmaSchedule` / `cfgEulerStep` in
`packages/models/src/anima/sampler.ts`, `denormalizeLatents` / `padSequence` in
`packages/models/src/anima/latents.ts`) is **bit-identical to this fixture in all 4** (rounding to
f32 one operation at a time with `Math.fround`).

### Fusing the Turbo LoRA and the turbo reference fixture

Passing a few-step distilled Turbo LoRA to `--lora` (e.g.
`inputs/anima/anima-turbo-lora-v0.2.safetensors`; the real weights are not in the repository —
place them by hand) fuses it into the weights before the export. This LoRA has been **measured to
have an all-zero (noop) `lora_B` on the text_conditioner side**, so **emitting the transformer
target alone is enough** (the other 3 targets share the existing `outputs/series/anima-f16/`):

```sh
uv run --group anima python -m anima.export --dtype f16 --target transformer \
  --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors --out ../../outputs/series/anima-turbo-f16
```

`--verify transformer --lora <path>` becomes an eager equivalence check against the post-LoRA
weights (`_apply_lora` takes effect before the fake-quant and the reference capture, so both sides
see the same LoRA-applied model with no extra code changes; measured: bit-identical in every case).

Reference fixture for turbo operation (steps=10 / CFG=1):

```sh
uv run --group anima python -m anima.pipeline_ref --dtype f16 --steps 10 --ref-steps 10 \
  --guidance-scale 1.0 --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors \
  --out ../../outputs/series/anima-pipeline-turbo-f16
```

`--guidance-scale 1.0` **skips the DiT call for the uncond branch entirely** (the same shape as a
production turbo deployment), so the output fixture **has no `noise_uncond_stepNNNN` keys**. Note
that the key set differs from the base-series fixtures with guidance_scale != 1.0 (the host glue
treats the missing keys as the "do not compute uncond" branch).

## VAE decode fixed-tiling reference fixture (`anima/tiling.py`)

The **authoritative host-side numbers** for the host-side VAE decode tiling (splitting the VAE
decode into fixed latent 64×64 tiles). The VAE decoder graph is **structurally invariant with
respect to resolution** (the `model.safetensors` for 512px and for 1024px agree exactly, down to the
node sequence and the weight bytes — recon `docs/research/2026-08-03-dynres-vae-tiling.md` §1.2), so
the 512px assets can be used directly as the tile decoder. **Nothing is added to the IR or the
runtime**; only cutting, blending and pasting live on the host
(`packages/models/src/anima/tiling.ts`).

```sh
uv run --group anima python -m anima.tiling     # into outputs/series/anima-tiling-f16-1024/
uv run --group anima python -m anima.tiling --resolution 1344x768 \
  --latents ../../outputs/series/anima-pipeline-turbo-f16-1344x768/pipeline.safetensors
```

- The input latent is borrowed from `latents_denorm` in
  `outputs/series/anima-pipeline-turbo-f16-1024/pipeline.safetensors` (already denormalized = exactly the
  VAE decoder input). **Generate that first** (otherwise it fails naming the missing file). A real
  pipeline latent is used rather than randn because how the seams show depends on the actual values.
- The outputs are `tiling.safetensors` (`latents_denorm` / `image_tiled`, 13.0MB) and `tiling.json`
  (**the tile geometry** = per-axis start positions, stride and blend width, plus observations of
  the difference against a non-tiled decode). The pre-migration Deno-side tiling E2E checked
  **against this geometry metadata** as well as the numbers — numbers alone cannot rule out "a
  different geometry that also lands inside the tolerance". That test has **not been ported into
  this repository yet**.
- **`vae.enable_tiling()` is not used.** The upstream `tiled_decode` scans with `range(0, H,
  stride)`, which makes the last tile shorter — something a fixed-shape tile decoder cannot digest.
  The scan is changed to "evenly spaced placement that snaps the last tile's start to `extent −
  tile`" (the deliberate deviation announced in recon §4.2). The script fails loudly if
  `vae.use_tiling` is true.
- **The blend formulas are transcribed verbatim from upstream** (`blend_v` / `blend_h`). Equivalence
  is pinned by `tests/test_anima/tiling.py` as **bit equality against the real methods** — once the
  scan is our own, this is the only place the formulas' isomorphism can be guaranteed.
- The weights are fake-quantized to the same dtype as the asset series before the references are
  taken (ADR 0006 — the same discipline as `anima/pipeline_ref.py`). The default `--dtype f16`
  corresponds to the `outputs/series/anima-f16/vae_decoder` the TS side opens.

- **`--resolution` takes `WxH`** (non-square is #23). The geometry is built from the shape of the
  input latent, so this argument only gates "is the borrowed latent at the intended resolution" and
  determines the default `--out` name. 1344×768 (latent 96×168) is **2×3 = 6 tiles, stride 32/52,
  blend 256/96px**.

Measured (1024px, CPU f32, 9 tiles): the difference against a non-tiled decode is **maxAbs 5.07e-2 /
mean 9.82e-4**. Tiling is an **approximation** that confines the receptive field of the decoder's
attention to within a tile (the same approximation as upstream `tiled_decode`), so **this difference
being non-zero is correct**. The implementation error is a separate matter: real GPU vs torch was
maxAbs 1.642e-5, and the same observation for 1344×768 (6 tiles) is **maxAbs 9.97e-2** with real GPU
vs torch at **8.02e-6**. Both GPU-vs-torch figures are **historical measurements** from the
pre-migration numerical E2E; the current gate is `packages/models/tests/e2e_anima_test.ts` (exact
PNG SHA-256 equality; tolerance-based acceptance and reference-value replacement are forbidden).

## Non-square rope table reference fixture (`anima/rope_tables.py`)

The host for the S-form DiT (`--dit-graph dyn`) assembles the rope cos / sin by **permuting the
per-axis base tables**. **On square resolutions an h ↔ w mix-up in that permutation is in principle
undetectable** — Anima's `rope_scale` has the same value for h and w (`[1.0, 4.0, 4.0]`), so `cos_h`
and `cos_w` agree byte for byte and, with H'=W', the tables themselves hold the same values
(detection limit 1 of ADR 0034). **On non-square resolutions the positional mix-up splits them
apart**, so the upstream `model.rope` tables are baked for 4 geometries and compared against the TS
side's reconstruction with exact Uint32 equality.

```sh
uv run --group anima python -m anima.rope_tables       # into outputs/series/anima-rope-nonsquare/ (a few seconds)
```

- The geometries are the fixed 4 of `GEOMETRIES` (**16:9 and 3:4, both orientations** = 1344×768 /
  768×1344 / 1152×896 / 896×1152, all with S=4,032). **Both orientations of a pair must be
  included** — with only one, "an implementation that swapped h and w" would agree with the table of
  the other geometry. The script fails if a square resolution creeps in.
- **Not a single byte of weights is read.** `CosmosRotaryPosEmbed` has neither parameters nor
  buffers and is pure computation, so the model is built from the config on the `meta` device (no
  7.3GiB load required).
- The outputs are `rope.safetensors` (`cos_<WxH>` / `sin_<WxH>` per geometry, each `[1,1,S,128]`,
  16.5MB) and `rope.json` (latent dimensions, token grid, S, number of rows in the base tables).
- The Python-side mirror (reconstruction from the base tables ≡ the upstream tables) is pinned by
  `tests/test_anima/rope_tables.py` using a **synthetic rope without real weights** (a configuration where
  h and w of `rope_scale` differ so that a mix-up shows up in the numbers).

## Prompt layer of the image demo (`anima/demo.py`)

The script that produces the **runtime assets** required by `packages/models/src/anima/text/` (the
Deno implementation of prompt string → token id sequence) and the **parity fixture** for it. It does
not touch the model graphs.

```sh
# assets (2 files into outputs/series/anima-demo/text/) + fixture (packages/models/tests/fixtures/anima-text/)
cd tools/export-recipes
uv run --group anima python -m anima.demo
# always format afterwards (the committed form is what the formatter produces — verify's fmt --check covers fixtures too)
cd ../.. && deno fmt packages/models/tests/fixtures/anima-text/parity.json
```

**A single run always produces both** (if they are not built from the same table, the runtime
assets and the fixture would age apart into "the tests are green but only the demo emits a
different id sequence"). Measured at **about 3 minutes** (dominated by the exhaustive checks
below).

| Output                                                  | Size (measured)                | Content                                                                                                 |
| ------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `outputs/series/anima-demo/text/qwen2-tokenizer.json`   | 3,514,619 B                    | vocabulary 151,643 / merges 151,387 / character class tables / NFC segmentation table / 26 added tokens |
| `outputs/series/anima-demo/text/t5-tokenizer.json`      | 1,093,419 B                    | vocabulary 32,100 + scores / normalization table / 103 added tokens                                     |
| `packages/models/tests/fixtures/anima-text/parity.json` | 474KB (formatted, git-tracked) | reference id sequences for 28 cases + 251 NFC pairs + a **subset** of the vocabularies                  |

- The runtime assets (4.6MB in total) live under **`outputs/` = `.gitignore`**. They keep only the
  information execution needs out of the raw `tokenizer.json` files (13.8MB in total), so that
  **licensed material is not carried in the repository**.
- **MUST: do not place them directly under the distribution tree `models/anima-turbo/`** — that one
  holds exactly the files the manifest declares and is uploaded to HF as-is (the same reason
  `outputs/series/anima-pipeline/` is kept separate).
- The fixture only holds a **subset** of the vocabularies (Qwen2 218 tokens / 375 merges / T5 125
  tokens), so every case can be reproduced without committing the 151k / 32k entries. The
  normalization table and the character class tables are **the folding output itself** (= the
  subject under test), so they are included in full.

### Why the tables are baked (`anima/text.py`)

**Unicode decisions are neither reimplemented in TS nor delegated to standard APIs** (the same
discipline as `sbv2.demo.clean_text_ranges`). The authority for those decisions is the Unicode
tables on the Rust side (`tokenizers` / its regex engine / `unicode-normalization`), and the moment
they diverge from the JS engine's ICU version the pre-token boundaries or the normalization result
change and **the id sequence alone becomes a different thing, with no exception and no warning**.
Every code point is evaluated from Python and folded into closed-interval tables and mapping tables,
leaving TS with nothing but a binary search.

There are 5 kinds of baked tables:

| Table                                  | How it is baked (querying the authority)                                          | Measured                      |
| -------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------- |
| character classes `\p{L}` `\p{N}` `\s` | feed one character at a time into `Split(Regex(…), behavior="removed")`           | 677 / 144 / 10 intervals      |
| `(?i:)` case equivalence               | `Split(Regex("(?i:s)"), …)` for the 8 suffix characters                           | 9 pairs                       |
| `NFC` segmentation                     | code points where `normalizers.NFC` and plain NFC diverge, via 8 context probes   | 123 cp / 43 intervals         |
| `Precompiled` normalization            | DARTS decoding of the charsmap + minimization to **only the rules that can fire** | 5,512 rules                   |
| cluster boundaries (3 tables)          | **probes** exploiting "pushing to 6 bytes or more makes the rule not fire"        | extend / breakAfter / prepend |

**`NFC` is not covered by the standard API either** (measured 2026-08-03): the authoritative
`unicode-normalization` has old Unicode tables and diverges from `String.prototype.normalize("NFC")`
/ `unicodedata` on **123 code points** (120 cp treat the combining class as 0 and do not reorder; 3
cp lack a newer canonical composition). The folding is "segment the string at the diverging code
points, run plain NFC on each segment only, and concatenate (the segmenting code points themselves
do not take part in normalization)". The real damage is a silent id-sequence mismatch, and because
`PROMPT_CASES` contained not one of the affected characters, it slipped through the 28-case gate
(measured on 1,200 randomized prompts: 6 mismatches for Qwen2 → 0 after the fix).

**`(?i:)` is not the ASCII equivalent** (measured 2026-08-03): Rust's `(?i:)` is Unicode simple case
folding, and among the 8 suffix characters (s t r e v m l d) **U+017F (ſ) is equated with `s`**.
`.lower()` / `toLowerCase()` / flipping ASCII case are none of them authoritative, and the boundary
in `it'ſs` becomes `'ſs` rather than `'ſ` + `s` (the fixture case `apostrophe_fold` is exactly this
boundary).

**`Precompiled` is neither NFKC nor longest-match**: it is whole-cluster replacement at the grapheme
cluster level with **shortest prefix wins**, so `A`+U+0301+U+0301 → `Á` (the third character
silently disappears) and `A`+U+0302+U+0301 → `Â` (the 2cp prefix wins over the 3cp rule `Ấ`).
Clusters of 6 or more UTF-8 bytes never enter the whole-replacement path, which is why Hangul,
regional indicators and emoji ZWJ sequences need no boundary rules implemented.

### Three stages of verification (nothing is emitted if any one of them comes off)

| Stage                                         | Where                                           | Scale (measured)                                                                                                                                                          |
| --------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ① equivalence of the folding                  | `anima/demo.py` (the emit gate)                 | all 1,112,064 cp + 5,512 rule keys + 200,000 randomized (fixed seed) / the pre-token scan and the NFC segmentation each cover **all cp × 11 contexts = 12,232,704 cases** |
| ② reference implementation vs `AutoTokenizer` | `anima/demo.py` + `tests/test_anima/demo.py`    | 28 cases (`padding="longest"` / `max_length=512` / `truncation=True` — the same call as `anima.pipeline_ref.encode_text`) + **2,000 randomized prompts** (fixed seed)     |
| ③ TS implementation vs the fixture            | `packages/models/tests/anima_tokenizer_test.ts` | 28 cases × 2 tokenizers + 251 NFC pairs + property tests + a cross-check against the pipeline reference                                                                   |

- ① **only runs on regeneration** (it takes minutes). The pytest side (②) redoes the same comparison
  against the committed fixture, so **a change in the upstream tokenizer.json is noticed even
  without regenerating**.
- The pre-token scan uses 11 context probes. `'{}` alone as a single character is **not enough** —
  whether or not the contraction alternative fires, the result is the same single fragment, and a
  case-insensitivity mix-up slips through (this hole was actually hit in practice). NFC segmentation
  has the same shape, keeping the detection probes (8 contexts) and the verification probes (11
  contexts) **separate**.
- The 2,000 randomized cases of ② are a permanent gate that mechanically exercises "the combinations
  a hand-written script did not think of" (the engine difference in NFC went through a gap between
  the 28 cases and split the id sequence). They are drawn with a fixed seed from an alphabet mixing
  emoji, combining characters, segmenting code points and various kinds of whitespace.
- ③ **runs without the real assets** (the fixture contains a subset of the vocabularies). In an
  environment that has the assets, reproduction against the real 151k / 32k vocabularies and a
  cross-check against the `qwen_input_ids` / `t5_input_ids` (the id sequences captured by torch) of
  `outputs/series/anima-pipeline{,-f16,-i8,-turbo-f16}/` run in addition. The latter goes through a **path
  separate from the fixture generator**, so it catches the case where the fixture itself is wrong.
- The structural assumptions about the upstream `tokenizer.json` (`normalizer.type` / the head of
  `pre_tokenizer` / ByteLevel's `add_prefix_space` / the special tokens added by the post_processor
  / the flags of the added tokens / T5's `byte_fallback`) are checked by
  `anima.demo.check_upstream_shape` on every emit. ①–③ can only catch **what they hit**, so the
  assumptions themselves are pinned structurally.

## Not there yet

- **A real-GPU comparison of Anima's `transformer` in f32 storage at the full 28 layers** (the
  7,465MiB of weights exceed this machine's GPU buffer limit of 7,280MiB and cannot be loaded —
  `docs/known-issues.md`). The real-GPU E2E does run for f16 storage (3,733.5MiB) and i8 storage
  (1,872.8MiB).
