# 0150: Irodori MeanFlow — v4.1-small-mf, the interval embedding in the DiT step and the 4-step sampler

- Status: proposed (2026-11-14 — the recipe, the TS pipeline, the local mirror `models/karume-irodori-v4.1-small-mf/`
  and its e2e checks are in place; published with 0.21.0 on 2026-11-14 (`irodori.SOURCES["irodori-v4.1-small-mf"]`);
  the default preset is `i8` by the maintainer's listening of 2026-11-14, and the tokenizer asset is referenced at
  v4-small — decisions 1 and 6)
- Date: 2026-11-14
- Related: [0038](0038-manifest-v1.md) §7 and its addendum of 2026-10-28 (cross-repository references) /
  [0109](0109-manifest-v5-container.md) decision 3 (a reference covers every part of a container) /
  [0047](0047-irodori-dit-execution.md) (the DiT graph and its host loop) /
  [0054](0054-resident-loop-and-fence.md) (the GPU-resident loop) /
  [0114](0114-irodori-dit-context-split.md) (`dit_context` / `dit`) / [0069](0069-packed-w4-storage.md) (i4 storage) /
  [0027](0027-sbv2-f16-series.md) (tolerances derived per series) /
  [0106](0106-device-keyed-references.md) (sha256 rows per environment) /
  [0139](0139-tests-on-distributions-and-test-assets.md) (where goldens live) /
  [0149](0149-irodori-rare-token-split.md) (the rare-token split table of the tokenizer asset) /
  [0128](0128-default-preset-original-weights.md) (the default preset per model)

## Context

Upstream published [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF)
(revision `ccc78f5d480b6e51b69b2d5042a14c4da04fea6e`, 2026-09-12, MIT): a student of v4.1-Small distilled with
MeanFlow (upstream `docs/meanflow.md`, `irodori_tts/meanflow.py`). MeanFlow learns the average velocity over a sampling
interval instead of the instantaneous one, so the model takes a few large steps.

What changes against v4.1-Small, from upstream's code and checkpoint:

- The checkpoint's `config_json` says `flow_parameterization: "meanflow"` (absent in v4 / v4.1, where upstream's
  default is `"rf_velocity"`); nothing else in either config differs.
- The DiT gains `delta_cond_module` — three bias-free linears (`[1280,512]`, `[1280,1280]`, `[3840,1280]`), the shape of
  `cond_module` — and the condition embedding becomes
  `cond_module(t_embed) + delta_cond_module(timestep_embedding(delta_t))`
  (`TextToLatentRFDiT.forward_with_encoded_conditions`).
- The sampler (`sample_euler_meanflow`): `schedule = torch.linspace(1.0, 0.0, num_steps + 1, float32)`; per step
  `t = schedule[i]`, `next = schedule[i+1]`, `delta = t − next`, one conditional forward with `delta_t = delta`, and
  `x = x + v·(next − t)`. 4 steps by default. Guidance was applied to the teacher during distillation, so there is no
  CFG at run time, and upstream ignores CFG scales, guidance mode and CFG time bounds, sway sampling, initial-noise
  truncation, temporal score rescaling and speaker-KV scaling for such a checkpoint.
- Measured (2026-11-14): only the DiT-side tensors differ from v4.1-Small (the blocks' 408, `cond_module` 3, `in_proj`
  2, `out_norm` 1, `out_proj` 2) plus the 3 new ones. The condition encoders, the duration predictor, the text
  backbone and the norms are equal tensor by tensor, and `tokenizer/tokenizer.json` equals v4-small's.

## Decision

1. **Its own repository, borrowing what it does not change.** The model is `v4.1-small-mf` in the repository
   `karume-irodori-v4.1-small-mf`. It physically carries only the roles whose bytes are new — `dit` and `dit_context` in
   every series (f32 / f16 / i8 / i4); every other role is a cross-repository reference to the repository that
   physically holds the bytes (references are not followed further — `karume.dist.external_refs`): `duration_*` at
   `karume-irodori-v4.1-small`, the rest (backbone, the two projectors, speaker, the codec pair) and the tokenizer asset
   at `karume-irodori-v4-small`. The recipe declares the containers in `irodori.distribution.IRODORI_PUBLISHED` and the
   tokenizer in `IRODORI_TOKENIZER_HOLDERS`, and the existing mechanism expresses the two owners as two references (one
   per repository, each with its own pinned revision).

   A borrowed role is **the published container itself**, read from the owner's mirror — not a re-export. The published
   non-DiT containers of v4-small / v4.1-small predate the `provenance.notice` field of the containers (and carry the
   codec's old license string), so a fresh export of the same weights differs from them in part 0 and could never be
   reconciled byte for byte. What makes the borrowing hold is checked on the inputs instead
   (`assert_irodori_borrowed_weights`, failing loudly): every checkpoint tensor a borrowed graph is built from is
   byte-identical to the owner's, the text-encoder configs are equal, and the model configs differ only in
   `flow_parameterization`. Each borrowed container must also be the published bytes: the recipe pins part 0 of every
   published container by sha256 (`IRODORI_PUBLISHED_PART0`; part 0's descriptor pins the other parts) and fails
   otherwise (`assert_irodori_published_containers`). The core's reference check then confirms the pinned revision on
   the Hub holds those bytes (`--resolve-refs hf`).

   The tokenizer asset is referenced at v4-small (the maintainer's ruling of 2026-11-14): the student's own asset,
   written by the recipe from its checkpoint, is byte-identical to v4-small's once v4-small carries the rare-token split
   table (ADR 0149 — measured 2026-11-14), and the core's reference check compares the two on every build. v4-small's
   repository is therefore uploaded with the table before this one is built (the order is in
   `docs/release-runbook.md` §0; done for 0.21.0). An earlier draft carried its own copy, because the published v4-small
   asset predated the table.

2. **One more input to `dit`.** The step graph of a MeanFlow checkpoint takes `delta_embed` — the timestep embedding of
   the interval, the same width and form as `t_embed`, built on the host by the same function — right after `t_embed`:
   `x_t`, `t_embed`, `delta_embed`, `mask`, then the 24 condition-side K/V. `dit_context` is unchanged in form. The
   export wrapper (`irodori.export.MeanFlowDitGraph`) evaluates the two embeddings in upstream's order, and the eager
   equivalence against upstream's unpatched forward stays at atol 0. The seven other graphs are not exported for this
   model. Which wrapper, cases and loop apply is keyed off the checkpoint's own `flow_parameterization` (upstream builds
   `delta_cond_module` exactly then), never off the directory name.

3. **`pipelineConfig.flowParameterization`.** A new optional key with upstream's spellings, `"rf_velocity"` |
   `"meanflow"` (like the two existing mode keys). Absent means `"rf_velocity"`, because the published manifests have no
   such key. With `"meanflow"`, the RF-only keys (`initScale`, `cfgMinT`, `cfgMaxT`, `cfgScales`, `speakerUncondMode`,
   `cfgGuidanceMode`) must be absent, and their presence is a parse error: Karume does not accept a value it would
   ignore. The parsed TS type is a discriminated union on `flowParameterization` (common fields & (RF fields |
   MeanFlow)), so the CFG fields cannot be read on the MeanFlow branch. `steps` comes from the manifest as before (4).

4. **The loop.** The MeanFlow branch of both loop paths (host and GPU-resident) runs one conditional forward per step
   with the interval embedding, no CFG variant, and the existing Euler update `x + (t_next − t)·v`; the two paths stay
   bit-identical to each other. The schedule and the intervals reproduce upstream's f32 values for any step count
   (`meanflowSchedule` / `meanflowIntervals`): torch's f32 `linspace` is not the closed form `1 − i/steps` (they split in
   the last bit for 5 of the step counts 1–12), and the TS form copies the kernel's (`step = −1/steps` in f32, the first
   half `1 + step·i` and the second half `−step·(steps − i)` with one rounding each). The values are pinned against
   torch 2.13's CPU output for 15 step counts from 1 to 100.

5. **The recipe.** `export.py` (the MeanFlow step wrapper, golden cases with an interval each plus one case that differs
   from `dit-cond-1s` in the interval alone, and a non-vacuity check that the interval moves the output);
   `pipeline_ref.py` (the reference loop equals upstream's `sample_euler_meanflow` under the same two-stage judgement as
   the RF loop, plus a non-vacuity check that the intervals move z); `calib.py` (the i4 series is GPTQ-calibrated with
   activations captured from the MeanFlow loop, with the per-block K/V projections running inside the blocks as in the
   RF capture; `delta_cond_module` is stored like `cond_module`, i.e. as i8 outside the i4 set; the calibration floor is
   the whole 4-step loop); `distribution.py` (the borrowed roles, the head inputs, `pipelineConfig` derived from the
   checkpoint config, the references); `card.py` (the card names the student, its revision, its teacher, the 4 steps
   and the absence of runtime CFG); `THIRD_PARTY_NOTICES.md` (a weights block for v4.1-Small-MF).

6. **Quant series.** The same set as v4.1-small: `f32`, `f16`, `i8` (with `i8-a8`) and `i8+dit4`. The default is `i8`,
   by the maintainer's listening of 2026-11-14: `i8` was fine, and `i8-a8` showed no audible degradation either
   (`IRODORI_MODEL_DEFAULT_QUANTS`; the family's default stays `i8-a8`). The preset descriptions on this model's card
   say only that (`IRODORI_MODEL_SEAT_DESCRIPTIONS`): the family's text carries the v4 listening check of `f16` and the
   v4 speed measurement of `i8-a8`, neither of which was judged or measured for this model; `f32` and `i8+dit4` keep
   the family's text, which states facts of the storage only.

7. **Tests.** Unit tests for the declaration (both flows and each rejection), the schedule and intervals, the loop
   material and admission; real-asset e2e for the new mirror — per-graph parity of `dit` / `dit-context`, latent parity
   per series (both opt-in), the declaration-to-golden premise bit for bit, determinism and host/resident path identity,
   and WAV sha256 rows of this machine under new case IDs. The goldens are staged for the private test-asset dataset.

## Consequences

- No RF behaviour or byte moves: the RF branch of every changed function keeps its operations and order (the RF goldens,
  WAV sha rows and the export tests are unchanged), and the published v4 / v4.1 manifests parse as before.
- The MeanFlow distribution is small: 2.7 GB locally (the DiT pair in four series); its manifest pins the borrowed
  files at the owners' Hub revisions. The publication build follows the upload of v4-small with the table and the
  re-upload of v4.1-small against it, so it pins the revisions those uploads create.
- **Open measurement — quality.** A 4-step student leaves no later steps to wash out an error, and the i4 rounding was
  calibrated on 12 cases × 4 steps (7,856 hidden tokens, against 83,080 for the RF loop's 40 steps). Upstream's own
  benchmarks are in f32. The maintainer's listening of 2026-11-14 covered `i8` and `i8-a8` (decision 6); `f16` and
  `i8+dit4` have not been judged for this model, and the speed of `i8-a8` has not been measured for it.
- The i8-a8 bundle of this model has a census row derived from its graph (248 linears in `dit`, 72 in `dit_context`),
  but no self A/B band yet (the A/B check covers v4.1-small).
- Reproducing torch's `linspace` rests on the kernel's fused multiply-add, the compiled behaviour on CPU and CUDA alike,
  not a documented guarantee.
- Published with 0.21.0 (2026-11-14) in that order: the uploads of v4-small (with the table) and v4.1-small first, then
  the HF upload of this repository built with `--resolve-refs hf`; the goldens are in the test-asset dataset with the
  pin moved in the same commit as the tests, and `irodori.SOURCES["irodori-v4.1-small-mf"]` pins the uploaded
  revision.
