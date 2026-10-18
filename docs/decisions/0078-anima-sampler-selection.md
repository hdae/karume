# 0078: Anima's default sampler is Euler; DPM++ 2M is a request option

- Status: accepted (2026-08-25, the maintainer's ruling: "keep Euler as the default sampler; DPM++
  2M is one of the options"; implemented and published the same day)
- Date: 2026-08-25
- Related: [0041](0041-manifest-v2.md) (`pipelineConfig` passes through the hub — where the sampler
  is declared) / [0073](0073-models-source-pin.md) (pins follow the published revision) /
  [release-runbook](../release-runbook.md) §0 (re-upload order of cross-referencing repositories)

## Context

During the 0.5.0 release, a visual A/B of the shipped bytes (same step count, same compute cost) led
to declaring `pipelineConfig.scheduler.type: "dpmpp-2m"` in the base and turbo manifests
(`d139e50`). The A/B observation itself holds: seed 42 was about even, seed 7 favoured dpmpp-2m in
composition and mid-ground detail, and no new artefacts appeared.

The declaration, however, gives users who follow `revision: "main"` a different image from 0.4.x
even with a fixed seed. The implementation also fixed the sampler in the manifest and deliberately
had no override in the generation request (`d32fa5e`, to keep "the distributor's recommendation and
the update rule that actually ran" from silently diverging). A user who wanted DPM++ 2M had to write
their own manifest.

## Decision

1. **The distribution default stays Euler** (matching the upstream Anima recommendation, the
   maintainer's policy). The manifests declare `scheduler.type: "euler"` explicitly (omission also
   means Euler, but the explicit form shows the choice was ruled — `ANIMA_SCHEDULER` in
   `tools/export-recipes/anima/distribution.py`).
2. **DPM++ 2M is chosen on the request** as one of the options: `AnimaGenerateRequest.sampler?:
   AnimaSamplerType`, effective sampler = `request.sampler ?? the manifest's scheduler.type`. This
   reverses "no override" of `d32fa5e`: the distribution default is the distributor's
   recommendation, the update rule is the user's knob, and the divergence concern is met by the JSDoc
   and the visible default. An unknown spelling fails loudly before touching the GPU, with the
   expected and the actual values (the vocabulary has one source, `ANIMA_SAMPLER_TYPES` in
   `packages/models/src/anima/config.ts`).
3. The visual A/B observation is **not rejected**. It stays as a record of relative quality (the
   body of `d139e50` and this ADR); only the default sides with the upstream recommendation.

## Consequences

- Release history (2026-08-25): anima `2682441a` and turbo `88357344` restored Euler for `main`
  followers, pins followed in `7ea134d`, and 0.5.1 brought the `sampler` field to 0.5.0 users. The
  same upload fixed a wrong repository name in the turbo card's Usage; the card now derives it from
  `Pipeline.repo_name` (release-runbook §0). That turbo repository was later retired (ADR 0087).
- The request path is checked on the real GPU by two cases in `packages/e2e/tests/e2e_anima_test.ts`
  (the official `anima-turbo-v1.1` at 512², and the base model with CFG), which run
  `sampler: "dpmpp-2m"` and compare against the DPM++ 2M sha256 reference rows.

## Addendum (2026-09-24) — vocabulary constant renamed

`SAMPLER_TYPES` of decision 2 was renamed `ANIMA_SAMPLER_TYPES` and exported from `@karume/models`
(the barrel and `./anima`). It is still the single source in `config.ts`.
