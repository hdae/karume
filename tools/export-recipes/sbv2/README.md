# SBV2 export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and the authoritative model card
is the exemplary `README.md` that `dist` generates into the distribution directory.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Scripts are started as
modules from `tools/export-recipes/` (`uv run --group sbv2 python -m sbv2.export …`); the
`text_encoder` seat of the distribution comes from the [DeBERTa recipe](../deberta/README.md), and
the generic emit, verify and dist contracts live in [`../../exporter/README.md`](../../exporter/README.md).

## Real-weight SBV2 export and E2E (M1-P3 wave 1: dp / wave 6: front / wave 7: flow, dec, voice)

Real weights for the acoustic chain. All 5 emit targets of ADR
[0013](../../../docs/decisions/0013-sbv2-chain-export.md) are **in place** (a green `voice` E2E = the
whole SBV2 chain holds):

| Target  | Contents                                       | sym_max | Notes                                                                                 |
| ------- | ---------------------------------------------- | ------- | ------------------------------------------------------------------------------------- |
| `dp`    | DurationPredictor alone                        | 512 (P) | Wave 1. Needs neither the patch layer nor vocabulary growth (the end-to-end scaffold) |
| `front` | enc_p + dp + sdp(reverse) fused into one graph | 512 (P) | Wave 6. **Patch layer required** (see "Patch layer" below)                            |
| `flow`  | reverse of TransformerCouplingBlock            | 4096(T) | Wave 7. Relative-position tables promoted to **graph inputs**                         |
| `dec`   | HiFi-GAN Generator                             | 4096(T) | Wave 7. No patches; `remove_weight_norm` is the only preprocessing                    |
| `voice` | flow + dec fused into one graph                | 4096(T) | Wave 7. **Once this passes, the whole chain is in place**                             |

### Dependencies

```sh
cd tools/export-recipes
uv sync --group sbv2   # style-bert-vits2==2.5.0 / huggingface-hub
```

- **`style-bert-vits2` is pinned with `==`.** The exporters of the later waves are built on
  monkeypatching package internals (class attributes of the modeling code), so a minor update that
  renames a replacement target or changes the shape of a forward silently changes the graph itself.
  The reason is also left as a comment on that line in `pyproject.toml`.
- It is **not** part of the default `uv sync` (so that the base deps alone keep the tiny goldens and
  pytest running). `transformers==5.14.1` comes in as a transitive dependency of this group.
- **Watch the build dependencies**: the transitive dependency `pyopenjtalk-dict` has no wheel and is
  built from an sdist, so it **requires a C / C++ compiler plus cmake and make**. Without them, `uv
  sync --group sbv2` fails with a cmake error (the dp export itself never uses this package, but
  dependency resolution cannot be skipped).

### Obtaining the weights

**The real weights are not part of the repository.** The default speaker is `FN4` of HF
`rufflet17/voice_models`, whose `FN/FN4/` directory carries `FN4.safetensors` / `config.json` /
`style_vectors.npy` / `style_settings.json` (the same content is also packed in `zip/FN_sbv2.zip`,
but the unpacked directory can be fetched file by file, so the zip is unnecessary). The publisher
declares the model free to modify. Place the 3 files the exporter reads under `inputs/sbv2/FN4/` —
the default `--model-dir` (`style_settings.json` is not read):

```
inputs/sbv2/FN4/config.json         HyperParameters (`version` drives the JP-Extra decision)
inputs/sbv2/FN4/FN4.safetensors     ckpt (**exactly one** `*.safetensors` directly under this dir)
inputs/sbv2/FN4/style_vectors.npy   style vectors (used for the front's style_vec)
```

`FN4` is a Style-Bert-VITS2 JP-Extra model (`version: 2.6.1-JP-Extra`) with `n_speakers: 1`, a
`sampling_rate` of 44100, and 4 styles (`data.style2id` = Neutral / high / low / NSFW).

The ckpt is required to be the **unique** match of `<model-dir>/*.safetensors` (with several, which
one was read would change silently), and the glob is non-recursive. Nothing generated is written
next to it: the exports go to a **separate root**, `outputs/series/`. Another speaker or another
model therefore needs nothing but `--model-dir` pointing at its directory — the series name is
derived from that directory's name (`inputs/sbv2/FN4/` → `outputs/series/sbv2-FN4/`), so two
speakers cannot silently overwrite each other's assets.

### Generation and comparison

```sh
# 1. generate (about 40 seconds for all 5, including loading the 251MB-class weights)
cd tools/export-recipes
uv run --group sbv2 python -m sbv2.export                # all targets
uv run --group sbv2 python -m sbv2.export --target front # one target only
uv run --group sbv2 python -m sbv2.export --dtype f16    # f16 series → outputs/series/sbv2-FN4-f16/
uv run --group sbv2 python -m sbv2.export --dtype i8     # i8 series  → outputs/series/sbv2-FN4-i8/
# mixed i4 series — only the two targets the distribution ships (dp / dec carry no linear at all)
uv run --group sbv2 python -m sbv2.export --dtype i4 --target front --target voice

# 2. eager equivalence against the reference implementation (**one target per process**; see "Patch layer" below)
uv run --group sbv2 python -m sbv2.export --verify front
uv run --group sbv2 python -m sbv2.export --verify flow
uv run --group sbv2 python -m sbv2.export --verify dec    # before/after remove_weight_norm
uv run --group sbv2 python -m sbv2.export --verify voice

# 3. real-GPU comparison (every case SKIPs when the assets are absent)
cd ../.. && deno test -A packages/runtime/tests/e2e_sbv2_test.ts packages/models/tests/sbv2_relattn_parity_test.ts
```

```
outputs/series/sbv2-FN4/dp/model.safetensors       IR   17 nodes /  12 initializers /   1.78MB
outputs/series/sbv2-FN4/front/model.safetensors    IR  911 nodes / 263 initializers /  33.4MB (2.1MB of baked tables)
outputs/series/sbv2-FN4/flow/model.safetensors     IR 1589 nodes / 458 initializers / 158.9MB (0.15MB of baked tables)
outputs/series/sbv2-FN4/dec/model.safetensors      IR  246 nodes / 197 initializers /  58.7MB
outputs/series/sbv2-FN4/voice/model.safetensors    IR 1836 nodes / 655 initializers / 217.6MB
outputs/series/sbv2-FN4/<target>/io.<case>.safetensors  inputs and expected torch CPU outputs
```

#### Storage dtype series (`--dtype f16` / `--dtype i8` / `--dtype i4` — ADR 0018 / 0019 / 0069)

`--dtype f16` / `--dtype i8` each write to a **separate series**,
`outputs/series/sbv2-FN4-f16/<target>/` / `outputs/series/sbv2-FN4-i8/<target>/` (keeping them next
to the f32 series would silently apply the f32 tolerance of the existing E2E to compressed assets).
The rounding (fake-quant) uses the shared `quantize.round_weights_to_f16` /
`quantize.fake_quant_int8` and is applied to the modules of each exported target **after
`remove_weight_norm` and the patches, and before the reference and golden capture**. i8 is **not a
vehicle for w8a8** (in all 5 SBV2 targets conv1d is 86–90% and linear is effectively 0 GFLOP — ADR
0025 decision ⑤) — the aim is asset size and load time, with the computation staying f32 (w8a32).

| Target  | f32 storage | f16 storage | Ratio | i8 storage | Ratio | Eligible (compressed resident) |
| ------- | ----------- | ----------- | ----: | ---------- | ----: | ------------------------------ |
| `dp`    | 1.78MB      | 0.90MB      | 50.4% | 0.46MB     | 25.7% | 4 tensors / 0.44MB             |
| `front` | 33.39MB     | 17.96MB     | 53.8% | 10.33MB    | 30.9% | 80 tensors / 7.72MB            |
| `flow`  | 158.86MB    | 79.92MB     | 50.3% | 40.65MB    | 25.6% | 156 tensors / 39.47MB          |
| `dec`   | 58.71MB     | 29.43MB     | 50.1% | 14.84MB    | 25.3% | 98 tensors / 14.64MB           |
| `voice` | 217.60MB    | 109.37MB    | 50.3% | 55.53MB    | 25.5% | 254 tensors / 54.11MB          |

(The eligible counts are the same across all 3 series. The byte figures are from the i8 series — the
set of compression candidates is determined solely by the `WEIGHT_SLOTS` weight slots and does not
depend on the storage dtype.) The ratio is higher for `front` alone because the baked
relative-position tables (2.1MB of i32 / f32 constants) are ineligible for weight slots and stay
f32. The totals are f32 470.34MB → f16 237.57MB (50.5%) → **i8 121.81MB (25.9%)**, and the i8
companion scales are 505,576 B (0.42% of the compressed bytes).

`--dtype i4` is a **mixed series** `outputs/series/sbv2-FN4-i4/<target>/`: eligible `nn.Linear`
and `nn.Conv1d` weights in group-32 i4 (ADR 0069 and its conv1d addendum — wave J-5b), everything
else in per-channel i8 exactly as in the i8 series. A conv1d is eligible when `groups == 1` and
its flattened row length `Cin·K` is a multiple of 32; depthwise convs, `ConvTranspose1d`
(`dec`'s `ups` — transposed layout, permuted pack not worth 2.6MiB), and weights with indivisible
rows stay i8, so a single-dtype i4 series cannot exist. Only `front` and `voice` are worth
writing — they are the seats of the distribution's `w4` quant, and no other consumer reads this
series.

The gain is negligible on purpose: net_g carries only 6 linears (`enc_p.style_proj` /
`enc_p.encoder.spk_emb_linear` in `front`, the 4 `flow_rev.flow.flows.<i>.enc.spk_emb_linear` in
`voice`) against 86–90% conv1d. The point of the series is to let the distribution ship a
whole-pipeline 4-bit seat next to the text encoder's, not to shrink net_g.

MUST: `--dtype` is **emit-only** (like `--sym-max`). The CLI rejects combining it with `--verify` —
verification is an eager comparison that does not look at the storage format, and for dec / voice
the rounding can only be applied **after** `remove_weight_norm` while the reference is taken
**before** the removal, so combining them would compare "the rounded side vs the unrounded side" and
break the `bit_exact` claim.

The dp wrapper is `forward(h, x_mask, g) -> logw`. The plain `DurationPredictor.forward` has `g` as
`Optional` and therefore a branch, so it is made mandatory to remove the branch, and the input names
are aligned with the names used in the recon (IR input names come straight from the forward argument
names). The dynamic axis is `Dim("P", min=2, max=512)` — **sym_max is per target: 512 for the front
family (P = number of phonemes) and 4096 for flow/dec/voice (T = number of frames)**. Nothing
enforces it mechanically and a wrong value stays silent, so the default lives in the script and
`--sym-max` only has to be spelled out when deviating from it (ADR 0013).

The front wrapper is `forward(x, x_mask, tone, language, bert, style_vec, g, z_noise) -> (logw_sdp,
logw_dp, m_p, logs_p)`. `x_mask` is an **external input** (the original implementation derives it
internally from `x_lengths`, but using a length "as a value" does not fit into the convolution), and
the randomness of the sdp reverse is likewise promoted to the external input `z_noise` (multiplying
by `noise_scale` is done on the host — runtime knobs are not baked into the graph). Mixing in
`sdp_ratio` and turning the result into durations is host-side as well.

The flow / voice wrapper is `forward(z_p, y_mask, g, idx_k, valid) -> z / audio`, and dec is the
plain `Generator.forward(x, g) -> audio [1,1,512T]` (no wrapper). `idx_k` / `valid` are the `(T,T)`
tables of relative-position attention and, **unlike front, are graph inputs** (see "Patch layer"
below).

The golden cases are **5 shared by every target** (so that the Deno side can check equality across
targets from a single table). The dp `h`, the front `x` / `tone` / `bert`, and the flow / dec `z_p`
/ `x` are randn with a fixed seed per length, `g` is the **real speaker embedding weight (`emb_g`)**
and `style_vec` is taken from the **real asset (`style_vectors.npy`)** (with synthetic randomness
the value ranges would not correspond to production and the tolerances would have no footing).

| Case     | Length | Content                                                                          |
| -------- | ------ | -------------------------------------------------------------------------------- |
| `p2`     | 2      | lower bound (the smallest value that avoids `torch.export`'s 0/1 specialization) |
| `p37`    | 37     | short                                                                            |
| `p203`   | 203    | medium                                                                           |
| `p512`   | 512    | exactly the declared upper bound of the front family                             |
| `padded` | 16     | last 5 columns masked to 0 (the only detector for the mask path of front / flow) |

- The `p<n>` in the case names comes from **P (number of phonemes)** in the front family, but in the
  flow family it means **T (number of frames)**. The names are left as they are, prioritizing one
  shared table across all targets (i.e. no target may skip a case).
- **No golden reaches the declared upper bound 4096 of the flow family.** The tables and attention
  scores are O(T²) and the dec output is 512·T, so at T=4096 a single io case would be 134MB with
  2.1e6 output points — not worth it as a golden asset nor as a real-GPU test. Coverage near the
  bound is instead shifted onto "no implementation depends on the declared upper bound", checked
  through the spread of lengths (2 → 512, a factor of 256).

The `padded` input **has values in the padded columns too** — if the mask multiplication works, the
tail of the output is exactly 0, and if it comes off, values leak. Without this shape it is not a
detector for the mask path (`tests/test_sbv2/export.py` goes as far as pinning down that "replacing
the mask with all ones makes the tail non-zero", closing off vacuous truth).

There are 2 asserts pinned at load time:

- **The relative-position attention window size == 4** (for both `enc_p` and `flow`). The index
  tables of the gather patch bake in `clamp(rel + 4, 0, 8)`, so a different window size would
  **silently read an embedding of a different width** (the element count still matches, so it is the
  "silent wrong value" class rather than a shape error). Worse, the goldens would be generated with
  the same mistake, so a numerical comparison would pass as well. Failing the moment the ckpt is
  swapped is the minimal remedy, so it lives in the loader (ADR 0013).
- **No weight_norm-derived parameters remain in `enc_p` / `sdp` / `dp` / `flow`.** If any remain,
  `weight` is no longer the effective weight and writing it out as-is produces a different model
  (measured 0 occurrences, but this is the side that does nothing on the assumption they are absent,
  so it is pinned with fail loudly). **`dec` is not on this list** — it ships with weight_norm
  active (95 modules / 190 parameters), and removal is done by `ensure_dec_plain` immediately before
  export.

`ensure_dec_plain` (idempotent) runs dec through `remove_weight_norm` and then applies the asserts
above. There is an ordering constraint that **weight rounding (`--dtype f16` / `--dtype i8`) is
applied to the effective weights after the removal** (rounding before the removal would round
`weight_g` / `weight_v`, leaving the effective weights off the rounding lattice; with i8 the
discarded elements feed into amax and shift the whole per-channel scale), and it is written as a
MUST in the docstrings of `ensure_dec_plain` and `_fake_quant` (ADR 0013 / 0018 / 0019).

On the Deno side: `packages/runtime/tests/e2e_sbv2_test.ts` (one case = one test) and
`packages/models/tests/sbv2_relattn_parity_test.ts` (byte equality of the tables). Same two-stage
structure as DeBERTa: **if `outputs/series/sbv2-FN4/` contains not a single target directory,
everything SKIPs** (this is the environment where only the raw weights are in place and export has
not been run yet), and when they are **partially** present (a missing target / a missing case) it is
a FAIL. It is **parameterized by series (f32 / f16 / i8)**, and the tolerances are **derived from
measurements per series × target** (no reuse across series — re-deriving one would silently move the
other):

| Target  | f32 atol | f32 rtol | Dominant check | f32 measured maxAbs | f16 atol | f16 rtol | f16 measured maxAbs | i8 atol | i8 rtol | i8 measured maxAbs |
| ------- | -------- | -------- | -------------- | ------------------- | -------- | -------- | ------------------- | ------- | ------- | ------------------ |
| `dp`    | 1e-6     | 1e-5     | rtol           | 2.62e-6             | 1e-6     | 1e-5     | 3.58e-6             | 1e-6    | 2e-5    | 2.38e-6            |
| `front` | 1e-4     | 1e-5     | atol           | 1.75e-5             | 3e-4     | 1e-5     | 3.62e-5             | 2e-4    | 1e-5    | 2.37e-5            |
| `flow`  | 2e-5     | 1e-6     | atol           | 2.74e-6             | 2e-5     | 1e-6     | 2.15e-6             | 2e-5    | 1e-6    | 2.62e-6            |
| `dec`   | 3e-5     | 1e-6     | atol (alone)   | 4.00e-6             | 2e-5     | 1e-6     | 2.37e-6             | 5e-5    | 1e-6    | 6.07e-6            |
| `voice` | 1e-5     | 1e-6     | atol (alone)   | 1.60e-6             | 1.5e-5   | 1e-6     | 1.71e-6             | 1.5e-5  | 1e-6    | 1.74e-6            |

In the f16 series the only significant increase is the front's `logw_sdp` (1.75e-5 → 3.62e-5). The
other 4 targets stay in the same order of magnitude as the f32 series, which corroborates that the
rounded weights also went into the goldens (i.e. the quantization error does not enter the
difference) — had the rounding been forgotten, the difference would have turned into the weights'
relative 5e-4 class and shown up 3 orders higher. The i8 series has the same structure, with all 5
targets landing in the same order of magnitude as the f32 series (front at 1.4× and dec at 1.5×
being the largest — a forgotten rounding would push the per-channel quantization error of the 4e-3
class up by an order of magnitude). Which check dominates is the same per target in every series (it
is determined by the shape of the value range, so the storage dtype does not move it).

rtol cannot be the dominant check for flow / dec / voice because the outputs (the latent z and the
waveform) span a range that crosses 0, where the smallest non-zero `|ref|` drops to the 1e-8 class —
the relative error diverges there (dec's measured maxRel reaches 0.44, but the absolute error of
that element is of the 1e-8 class). The derivations are authoritative in the `SBV2_*_TOLERANCE`
comments in the same file. **The final `tanh` of dec is WGSL-implementation dependent and does not
match torch bit for bit**, so this comparison can in principle only hold with a tolerance.

On the pytest side: `tests/test_sbv2/export.py` (the script's contract) and
`tests/test_sbv2.patch.py` (unit tests for the patch layer). Building the golden inputs and the CLI
exclusivity need no real weights and always run; the export body runs **only in an environment where
the real weights and the `sbv2` group are both present** (otherwise SKIP).

### Patch layer (`sbv2/patch.py`)

front and flow / voice cannot be exported unmodified (dec does not need it). The patches work by
replacing attributes of already-imported classes (monkeypatch) plus wrappers; the `style_bert_vits2`
package itself is left untouched.

| Patch                                      | What it does                                          | Why it is needed                                                                                                                                 |
| ------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Branch-free, non-destructive spline        | boolean-mask indexing → clamp + where, in-place → cat | Extracting only the in-interval elements makes the **element count data-dependent**, giving `GuardOnDataDependentSymNode`                        |
| Gather form of relative-position attention | rel⇄abs shift → index table + last-dim gather         | The shift creates **quadratic shape expressions** `2P−1` / `2P²` / `P(2P−1)` (not rescued even by an affine extension of the dimension language) |
| Folding the FFN pad into the convolution   | explicit `F.pad` → `conv1d(padding=…)`                | Reduces `constant_pad_nd`. Exactly equivalent **only for an odd kernel and non-causal** (pinned by an assert)                                    |

#### Two schemes for the relative-position tables (ADR 0013)

The index tables (`idx_k` / `valid` / `idx_v`) are determined by `i` and `j` alone and are
**independent of length**, so the exporter's constant folding can lower them to baked tables +
`sym_prefix_slice`. The baked volume is O(sym_max²), however, so **the scheme differs per target**:

| Family        | Key side `idx_k` / `valid` | Value side `idx_v` | Baked volume    |
| ------------- | -------------------------- | ------------------ | --------------- |
| front (P≤512) | baked `(512,512)`          | baked              | 2.1MB measured  |
| flow / voice  | **graph inputs**           | baked `(4096,9)`   | 0.15MB measured |

Baking them for flow would come to `(4096,4096)` × 2 tables = **134MB**, hence the promotion to
inputs. The formulas are authoritative in `sbv2.patch.build_relattn_tables`, and **the in-graph
construction for front calls the same function** (writing the formula in two places would silently
produce two different tables the moment only one is fixed). The host-side mirror is
`packages/models/src/sbv2/relattn-tables.ts` (SBV2-specific, so it is not placed under
`packages/runtime/src/` — the models package owns the model-side knowledge), and **byte equality is
pinned by `packages/models/tests/sbv2_relattn_parity_test.ts` against the goldens' real data**. A mismatch
in window size 4 is of the silent-wrong-value class rather than a shape error, so the Python side
has `_assert_window_size` (at ckpt load) and the TS side has the parity test comparing against **the
width `2w+1` of the `idx_v` baked into the container**; gates on both sides exist because with only
one side there remains a path where the host and the goldens share the same mistake and slip
through.

#### The flow / voice wrappers

- `FlowReverse` applies `[TransformerCouplingLayer(mean_only) + Flip] × 4` in reversed order. The
  coupling reverse replaces `torch.split` with explicit slices (96/96) and **folds away the
  multiplication by `exp(−logs)`** (with `mean_only=True`, `logs = zeros_like(m)` and `exp(−0.0)` is
  exactly 1.0 under IEEE 754, so it is bit-identical — the standard trick for keeping `zeros_like`
  out of the IR).
- `Sbv2Voice` is `FlowReverse` → `z * y_mask` → `dec`. Same order as the tail of the reference
  infer, without carrying in the `max_len` slice (always `None` = identity on the inference path).

#### Gates against process contamination

- **`--verify` and emit cannot be combined in the same process** (the CLI rejects it with
  `parser.error`). The patches replace class attributes **process-wide** and `remove_weight_norm`
  destructively folds the weights, so if emit runs first the "pre-patch reference" can no longer be
  taken and the equivalence check becomes **vacuously true, a false PASS**. The verification itself
  follows the order "fix the references for every case → mutate → compare", and if that order is
  broken it raises a `RuntimeError` right before the reference capture (ADR 0013) — front / flow /
  voice check `sbv2.patch.patches_applied()`, while **dec checks the opposite**, that
  weight_norm-derived parameters are still present (its contamination source is the removal; voice
  has both gates).
- **`--verify` takes exactly one target.** Giving the CLI a pairwise exclusivity table ("MHA-patch
  targets are mutually exclusive / dec's removal conflicts only with voice / rounding conflicts with
  everything") would turn every hole in the table into a false PASS. With a form that takes a single
  value, **contaminating combinations structurally do not exist**.

#### Measured equivalence

| `--verify` | Cases | worst maxdiff                      | Notes                                                         |
| ---------- | ----- | ---------------------------------- | ------------------------------------------------------------- |
| `front`    | 9     | 2.02e-5 @P=512                     | `P ≤ 5` is bit-identical                                      |
| `flow`     | 10    | 1.43e-6 @T=512                     | `T ≤ 5` is bit-identical                                      |
| `dec`      | 10    | **0 (bit-identical in all cases)** | before/after remove_weight_norm                               |
| `voice`    | 10    | 1.25e-6 @T=203                     | compared against unpatched flow + dec with weight_norm active |

- The differences come from BLAS ordering as the value-side reduction length changes, and stay in
  the same order of magnitude as the real-GPU golden error (front 1.75e-5 / flow 2.74e-6 / voice
  1.60e-6).
- **dec being bit-identical in all cases closes an open item from recon §6** (until then there were
  only measurements for the single case `z=(1,192,50)`, recorded with the note that f32
  reproducibility of the effective weight `g·v/‖v‖` is not a spec guarantee). Since it requires
  `torch.equal`, even a `0.0` / `-0.0` mix-up would not pass.

## Asset prep and torch reference for the voice demo (`sbv2/demo.py`)

The script that provides the **host-side assets** required by `examples/sbv2/` (the Deno demo from
real text to WAV) and takes on the **numerical parity** of its output. It does not touch the model
graphs (neither the emit path nor the goldens change).

```sh
# ① runtime assets (3 files into outputs/sbv2-demo/)
uv run --group sbv2 python -m sbv2.demo assets

# ② run the demo (from the repository root) → out.wav and dump.safetensors
cd ../.. && deno task demo:sbv2 --text "こんにちは、これはテストです。" && cd tools/export-recipes

# ③ torch reference (rerun the same chain on the dump's discrete inputs and random sequence) → reference.wav + numbers
uv run --group sbv2 python -m sbv2.demo reference --dump ../../outputs/demo/sbv2-dump/dump.safetensors

# ④ official infer (the pyopenjtalk path) → official.wav (for listening comparisons of the accent)
uv run --group sbv2 python -m sbv2.demo official --text "こんにちは、これはテストです。"
```

### What `assets` produces

| File                     | Content                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| `symbols.json`           | the JP-Extra id rules, model constants and knob defaults (**all read from the real thing**) |
| `deberta-tokenizer.json` | the DeBERTa character tokenizer's vocabulary, special ids and `_clean_text` decision table  |
| `assets.safetensors`     | `style_vec` `[1,256]` and the speaker embedding `g` `[1,512,1]`                             |

**MUST: never transcribe constants by hand.** The symbol table, the tone base
(`LANGUAGE_TONE_START_MAP["JP"]`), the language id (`LANGUAGE_ID_MAP["JP"]`) and the add_blank
insertion value are read from `style_bert_vits2`. The multilingual version and JP-Extra look the
same, yet a divergence **keeps the shapes matching and only breaks the sound** (the silent
wrong-value class). Only the insertion value is written as a literal in the source, so
`blank_id_from_source` extracts it from the source of `infer.get_text` with a regular expression for
all 3 series and additionally confirms that the value is the same across them.

Baking the `_clean_text` decision table has the same motivation: moving the `unicodedata.category`
classification into TS would turn ICU version differences into a silent mismatch — so every code
point is evaluated in Python and folded into closed intervals.

> NOTE: **`language` is not all zeros even for JP-Extra.** `infer.get_text` goes through
> `cleaned_text_to_sequence(..., JP)`, so the real phoneme positions are 1 and only the add_blank
> insertion positions are 0. `sbv2.export.make_language` being all zeros is a choice made for
> _synthetic golden inputs_ (any value makes the golden valid); it is not the inference rule.

### What `reference` claims

It runs the `sbv2.patch` modules (`Sbv2Front` / `Sbv2Voice`) together with **the same host glue as
the demo** on torch CPU and compares against the Karume waveform recorded in the dump. So what is
being measured is "the same computation graph run on a real GPU vs run on torch CPU"; equivalence
against the original unpatched implementation is held separately by `sbv2/export.py --verify` (the
layers are not mixed). Two gates are applied along with it:

- **Tokenization parity** — the dump's `bertText` is fed to the Python tokenizer and required to
  match the dump's `input_ids` exactly, **before** the waveform comparison. A divergence here
  distributes the BERT features to different phonemes and stays silent in the shape of "sound comes
  out, but distorted".
- **Integer equality of `w_ceil`** — durations use `ceil`, so if the front output sits just above a
  threshold, a 1e-5 GPU/CPU difference shifts a frame. Positions that disagree are reported together
  with the `w` values (in a form that lets the reader judge flake versus implementation difference).

### Why `official` is a separate subcommand

The `sbv2.patch` patches replace class attributes **process-wide**, and `reference` applies them.
What `official` claims is "the sound through the original implementation's g2p (pyopenjtalk) and the
original implementation's attention / spline", so co-hosting it in the same process would silently
put it on the patched path. Since argparse subparsers only allow one choice per process, **one
subcommand per process** holds structurally (the same rationale as `--verify` not carrying a
pairwise exclusivity table).

The 3 wav files (`out.wav` / `reference.wav` / `official.wav`) are written with the same PCM16
conversion rule (clip → `floor(x·32767 + 0.5)`). Python's built-in `round` is banker's rounding, so
without aligning that, an implementation difference would creep into the listening comparison.
