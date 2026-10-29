# 0016: Anima chain — export strategy and E2E layout

- Status: accepted (2026-08-02); partly superseded by [0129](0129-test-layout-and-lanes.md) (accepted 2026-10-29;
  model-exit tolerance bands opt-in since 2026-10-17); phase exclusions withdrawn 2026-10-18
- Date: 2026-08-02
- Related: [../research/2026-08-02-anima-recon.md](../research/2026-08-02-anima-recon.md) (recon with
  Karume's own exporter — 4/4 exported, 19 unsupported ops, arity gaps, VRAM and limits measured)

## Decision

- **Four export targets**: `text_encoder` (Qwen3, dynamic T, sym_max 512) / `text_conditioner` (two
  independent symbols Tsrc / Ttgt, sym_max 512) / `transformer` (DiT, static graph at a fixed
  resolution, default 512px) / `vae_decoder` (static, 512px). No fused target: host work (tokenization,
  CFG, scheduler, denormalization) sits between the components. Current fact: the static DiT graph is
  still the default; a symbolic token-length graph was added as an extra series (`--dit-graph dyn`,
  `DIT_SYM_MAX = 16384` — `tools/export-recipes/anima/export.py:274-283`).
- **Keep `STRIDED_RANK = 4`**; lower the rank in normalization passes. Raising it to 8 charges every
  rank-4 value, bumps snapshot and determinism keys and widens verification for nine patchify nodes
  (as ADR 0014 rejected negative-stride flip). Revisit only if many shapes fail structurally.
- **Six normalization passes** (recon §4, `tools/exporter/src/karume/normalize.py`): `_fold_rms_norm` /
  `_drop_safe_softmax_guard` / `_lower_unit_expand` / `_lower_split_unbind` / `_lower_reshape_permute`
  / `_select_to_squeeze`. Order MUST: the RMS fold runs before `_pow2_to_mul` and
  `_promote_scalar_operands`. The three rank-lowering passes fire only on rank ≥ 5 (guard against
  touching existing graphs; reuses `STRIDED_RANK`, no new constant). The SDPA guard is removed **only
  when proven inactive**, otherwise fail loudly (accepting the guard as an op is rejected: the IR JSON
  layer rejects non-finite values).
- **Anima patch layer** (`tools/export-recipes/anima/patch.py`, same approach as the SBV2 one; model
  packages untouched). Its purpose is **IR quality**, not exportability: no new ops, rank ≤ 4, no
  runtime knobs baked into the graph. Contents: (1) drop the all-ones masks of Qwen3 / conditioner
  (512 padding and mask multiplication on the host); (2) promote the DiT timestep embedding to a graph
  input, fold padding_mask into a constant zero channel, move the patchify rank; (3) replace VAE
  CausalConv3d(T=1) with conv2d at rank 4, reject feat_cache; (4) lift RoPE buffers into attributes.
  **The patched model is eager-equivalent** (`--verify` measures it; reference taken before patching;
  process-wide replacement order-protected), as in ADR 0013.
- **Extend FOLDABLE_OPS from measurement** (candidates sin / cos / reciprocal / pow.Scalar /
  index.Tensor / add.Scalar / mul.Scalar / ne.Scalar / eq.Tensor / le.Tensor; add only those seen on
  the folding frontier). Correction (2026-08-02): `--verify` cannot guard folding — it compares eager
  before and after patching, while folding happens inside `convert()`. The guard is: (1) symbolic
  graphs run the two-point evaluation of `_check_prefix_commutes`; (2) static graphs fold by
  recomputing in the same call, so "prefix does not commute" cannot occur; (3) golden E2E (IR on the
  GPU vs eager) covers the numbers of static graphs.
- **Linear without bias gets a zero bias; layer_norm without affine gets ones / zeros**, both to arity
  3 (ADR 0015's technique; no arity branch in kernels or contracts). Measured: 698 of 711 linears have
  no bias; all 85 DiT layer_norms have no affine.
- **LoRA is baked into the weights before export** (`tools/export-recipes/anima/lora.py`):
  ΔW = (B @ A) · scale in f32, added in place in the original dtype. IR unchanged, no runtime code.
  Fail loudly on an all-zero lora_B, no matching layer, or a shape mismatch. A turbo LoRA E2E is not a
  required check (turbo is now an official checkpoint, ADR [0087](0087-anima-official-extra-repos.md)).
- **E2E stages**: (1) per-component golden E2E (text 3 / conditioner 2 / DiT 2 / VAE 2 cases);
  (2) a few-step run against a pipeline reference fixture (a verbatim Python script emits sigmas, the
  timesteps_proj table, each step's latent and the final image; host glue in TS inside the test), with
  tolerances derived per stage from measurement (one DiT step / after the loop / decoded image), not
  borrowed from SBV2 or DeBERTa (superseded by ADR 0129 on 2026-10-17: model-exit parity runs only with
  `KARUME_UPSTREAM_PARITY=1`; Anima regressions are covered by the sha256 reference rows).
- Withdrawn 2026-10-18 (phase boundaries now past): "E2E stays f32 at 512px, no f16 storage in this
  phase" (f16 / i8 came in ADRs 0018 / 0019) and "no TS tokenizer port" (`@karume/models` has them).

## Alternatives considered

- Preserve SDPA and reimplement the guard in a fused kernel: one new op and a reimplemented guard
  semantics, nothing gained over the decomposition plus guard removal. Rejected.
- Add mean.dim / rsqrt and run RMSNorm decomposed: three intermediate buffers and five times the
  dispatches at 269 sites; one rms_norm kernel plus the FX fold is shortest (prototype). Rejected.
- All four components resident for the end-to-end run: f32 does not fit VRAM (10.11GiB plus
  activations); the hand-off is about 2MiB through the host, so sequential sessions suffice. Rejected.
