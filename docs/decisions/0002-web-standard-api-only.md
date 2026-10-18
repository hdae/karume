# 0002: The runtime depends on Web standard APIs only

- Status: accepted (2026-08-01); kept unchanged by [0127](0127-node-dawn-toolchain.md) decision 2;
  0127 decision 5 moved only the lint mechanism (addendum of 2026-10-11).
- Date: 2026-08-01
- Related: recon §0 / §1 — restates the prototype's decision 0003 ("keep a Deno move possible"),
  which cut its port to 7 residual `deno check` errors.

## Decision

The library's runtime dependencies are **Web standard APIs only**, with zero external packages:
`navigator.gpu`, `performance.now`, `TextDecoder` and the `ArrayBuffer` family. File and network I/O
are not part of the library; the caller passes an `ArrayBuffer` (or a sequence of chunks).

Implementation rules (preventing the porting costs found in the prototype's Deno port):

- Relative imports always carry the `.ts` extension. Vite-specific syntax (`?raw`, `?url`,
  `import.meta.env`, glob imports) is forbidden. WGSL lives in TypeScript as template literals.
- TypedArrays name their buffer type, as in `Float32Array<ArrayBuffer>` (the fix for 6 of the
  prototype's 7 `deno check` residuals).
- No direct dependency on API surfaces where Dawn and wgpu differ (`wgslLanguageFeatures`, …).
  Features, limits and language features are normalised in **one capability-detection layer**; a
  missing one becomes the empty set.

## Consequences

- Any host with Web standard APIs and WebGPU works by construction (Deno and the browser then; Node
  with an injected WebGPU since ADR 0127 decision 2).
- Transport (range requests, the Cache API, concurrency control) stays outside the library, as in
  the prototype's decision 0011.

## Addendum (2026-09-25) — the allowed APIs are examples, not a closed list

The criterion is "is it a Web standard API". The runtime also uses `crypto.subtle.digest` (sha256 of
container blocks — `src/format/container/descriptor.ts`), `TextEncoder` (`header.ts`, `json.ts` in
the same directory) and `queueMicrotask` (`src/gpu/context.ts`); `Deno.` and `fetch(` do not appear in
its code. Transport moved to `@karume/hub` with the monorepo (ADR [0037](0037-karume-monorepo.md));
dependency packages built only on Web standard APIs are allowed (ADR [0038](0038-manifest-v1.md) §5).

## Addendum (2026-10-11) — machine check of `<ArrayBuffer>` and the exception table

The rule "TypedArrays name their buffer type" is the lint rule `karume/typed-array-buffer`
(`tools/lint/oxlint-plugin.ts`, registered in `lint.jsPlugins` of `vite.config.ts`, run by `vp lint`
inside `vp check`; tested by `tools/lint/tests/plugin_test.ts` in the core lane). It began as a
`deno lint` plugin; ADR 0127 decision 5 ported it to Oxlint.

- **Reports**: under `packages/<pkg>/src/`, one of the 12 TypedArrays without a type argument in a
  type position (annotation, type argument, return type, type alias, field, `satisfies`, interface
  `extends`, class `implements`); a qualified name is judged by its last segment. Value positions
  (`new Float32Array(n)`, `instanceof`, `typeof` queries) are not type references. Tests and tools
  are out of scope.
- **Misses**: forms that yield `ArrayBufferLike` without naming the TypedArray as a type reference
  (`InstanceType<typeof Float32Array>`, `ReturnType<typeof Uint8Array.from>`) — the check does not
  resolve types. None existed on 2026-10-11; `plugin_test.ts` pins both as not reported, so a rule
  that catches them updates this item and the test together.
- **Why**: since TypeScript 5.7 an omitted argument means `ArrayBufferLike` (including
  `SharedArrayBuffer`), silently widening values passed to `GPUQueue.writeBuffer` or
  `CompressionStream`. There were 148 such sites when the rule came in (hub 6, models 139, runtime 3).
- **Exceptions**: a boundary that deliberately accepts any buffer writes `<ArrayBufferLike>`
  explicitly, never a lint ignore. `rg -P '^(?!\s*(?://|/?\*)).*<ArrayBufferLike>' packages/*/src`
  (comment lines excluded) then lists them all: 20 lines with 21 sites on 2026-10-18 (one `png.ts`
  line has two), matching the table. A new exception adds a row.

| Boundary | Sites | Why |
| --- | --- | --- |
| hub fetch entry | `source.ts`: return of `PinnedSource.readFile`, argument of `ManifestReadOptions.parse` / `fetch.ts`: `isTightView`, `assertTightView`, argument of `decodeManifest`, `bytes` in `fetchAssetsWithBudget` | The external fetch layer (`@hdae/fetch-cache`) types its results and validation-hook arguments as `Uint8Array<ArrayBufferLike>`. `assertTightView` is the single point that checks at run time that the bytes are `ArrayBuffer`-backed and span the whole buffer, narrowing to `Uint8Array<ArrayBuffer>`; before it the wide type is accepted (`decodeManifest` only reads through `TextDecoder`). The public `DirectoryAdapter` (`sources/local.ts`) still requires `<ArrayBuffer>`. |
| models public API inputs (read on the CPU only) | `audio/wav.ts`: `encodeWav(samples)`, `decodeWav(bytes)` / `image/png.ts`: `encodePng(rgba)` / `image/preprocess.ts`: `Rgb8Image.data` (image input of birefnet, depth-anything, siglip2, wan) / `anima/latents.ts`: `denormalizeLatents(latents, mean, std)` / `anima/preview.ts`: `approximatePreview(latents)` / `vowel-detector/pipeline.ts`: `VowelDetectorPipeline.detect(audio)` | User values that the library only reads by index, `DataView` or `TextDecoder` (what reaches the GPU or `CompressionStream` is an array the library allocates). The wide type takes `SharedArrayBuffer`-backed arrays (an AudioWorklet ring buffer) and untyped results of other libraries without a copy. It equals the public type before the rule, so the public surface did not change. |
| internal read paths carrying those inputs | `vowel-detector/pipeline.ts`: `detectAudio(state, audio)` / `vowel-detector/features.ts`: `audio` of `extractFeatures(audio, …)` | Carries the input of `detect` to feature extraction without a copy (read only); `<ArrayBuffer>` here would force a full copy. |
| models internal CPU resampler | `image/pillow-resample.ts`: `plane` of `resizeU8PlaneBicubic` (BiRefNet matte post-processing) | Reads the 8-bit plane only; always returns a newly allocated array. |
| test-support file writes | `packages/test-support/src/fs.ts`: `data` of `writeFile`, `writeFileSync` | The write side accepts any buffer, like `Deno.writeFile` (private test harness, not published). |
