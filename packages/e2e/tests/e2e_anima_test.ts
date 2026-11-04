/**
 * Anima's **port check** (real GPU). It runs manifest → assets → `Pipeline` → `generate` →
 * `encodePng` end to end and checks only whether the sha256 of the output pixels (the RGB8 pixel
 * sequence `encodePng` writes into IDAT — {@link rgbPixels}) is **bit-identical** to the reference
 * value. The PNG is kept as an artifact, but the sha is taken over the pixels before compression —
 * the deflate of IDAT changes with the runtime's zlib implementation (Deno and Node produce
 * different bytes for identical pixels), so a row would capture the compressor instead of the
 * numbers (ADR 0106 addendum decision 9).
 *
 * The reference values sit in `fixtures/references/anima.json` as **rows per environment key**
 * (`helpers/reference.ts`). All this check can claim is "the numbers have not moved on that
 * machine"; cross-device bit identity is not guaranteed by the spec (docs/limitations.md). The RTX
 * 3080 Ti rows for turbo were **frozen at the first measurement of anima-turbo-v1.1 (the official
 * checkpoint — ADR 0087)** (the restructuring of 2026-09-01 retired the reference values of the old
 * fused turbo because the model itself was replaced. The rows of the plain base (anima-v1.0) are
 * still the 2026-08-22 measurement = they double as proof that the restructuring did not move a
 * single bit of v1.0's bytes).
 *
 * If the numbers move by even 1 bit, something in the port changed — **turning this into a
 * tolerance is forbidden**; the check stops red and prints what differs (PNG byte length / pixel
 * statistics / the PNG artifact). Loosening it would erase the meaning of "the port is done".
 *
 * NOTE: most cases (`f16+dit8-a8-attn8-s16` = the default preset) are **practical-preset rows**
 * baked on an opt-in preset; the only reference-path row is the one case `f16-1024` (ADR 0110
 * decision 7 / ADR 0106 addendum 2026-09-26 — a row's class is derived from whether the quant
 * preset's `session` is empty). When the practical-tier kernels are changed on purpose, the
 * practical-preset rows are `rewrite`n in the same commit. The reference-path rows are frozen; if
 * they move, the check stops red.
 *
 * - **Adding rows for another machine** is `KARUME_REFERENCE=write` (existing rows are untouched).
 * - **Re-baking your own machine's rows** is `KARUME_REFERENCE=rewrite` only, and only once you can
 *   say "what changed" first (old → new is logged). Other environments' rows are not touched.
 * - On a machine without reference values the case **SKIPs explicitly**, and the reference gate (at
 *   the end of this file) reports "every case SKIPped for want of references" in red.
 * - The checks that produce the same bytes through another route (`fromPretrained-512` /
 *   `onEvent-1024` / `base-cfg-fromAssets-shards`) hold **a row under their own case ID** and also
 *   check that they match the row of their twin case (so that splitting the rows does not loosen
 *   bit identity across routes).
 *
 * MUST: the assets are `models/karume-anima/` (untracked; local assets of the real-GPU machine —
 * the 5 official variants together, default = anima-turbo-v1.1). The turbo checks (8 steps / no
 * CFG) use the default model, and the **CFG≠1 check** (the plain base = `model: "anima-v1.0"` — the
 * only one that goes through the second text path and `cfgEulerStep`) runs by model selection in
 * the same repository. Without the assets, and without a GPU, the cases **SKIP explicitly** with a
 * reason (no deleting tests to go silently green — ADR 0005).
 *
 * NOTE: the checks on the real distribution basically all go **through the fetch layer** (local
 * HTTP + `fromPretrained`) (the same graph-first + sequential streaming route as production). The
 * part split itself can also be read through the whole-asset surface (`fromAssets`), so exactly one
 * check runs base through `Deno.readFile` + `fromAssets` and requires **the same reference sha256**
 * as the fetch-layer route (the "whole-asset surface" section below — X2-101). Only the test side
 * reads files with `Deno.open` / `Deno.readFile`; the package itself uses Web standard APIs only
 * (no fs — a cross-cutting invariant).
 *
 * NOTE: since the official repository is self-contained (ADR 0087), the real-asset check of
 * cross-repository references (a (repo, commit SHA) of another repository) is held by **the
 * distribution of the extra variants** `models/karume-anima-extra/` (baked at release with
 * cross-repository references to a commit SHA of the official repository — release-runbook) (the
 * "cross-repository references" section below). Where the extra mirror is absent, only that section
 * SKIPs explicitly.
 */

import { createReadStream } from "node:fs";
import { createServer, type ServerResponse } from "node:http";
import { readFile, stat, writeFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assertEquals, assertFalse, assertRejects, assertStrictEquals } from "@std/assert";
import { type FileRef, parseManifest, resolveSelection, selectionRefs } from "@karume/hub";
import { acquireGpu, type GpuContext, type SessionDiagnostics } from "@karume/runtime";
import type { Manifest } from "@karume/hub";
import {
  type GenerateEvent,
  Pipeline,
  type Residency,
  type SamplerType,
  formatResolution,
  type GeneratedImage,
  type ImageSize,
} from "../../models/anima.ts";
import { encodePng } from "../../models/mod.ts";
import { parseAnimaPipelineConfig } from "../../models/src/anima/config.ts";
import { sigmaSchedule } from "../../models/src/anima/sampler.ts";
import { ANIMA_SPATIAL_COMPRESSION } from "../../models/src/anima/dit-tokens.ts";
import { HEADROOM_MARGIN_BYTES } from "../../models/src/session/headroom-margin.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { MemoryCacheStorage } from "../../models/tests/helpers/memory-cache.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import {
  announceCheck,
  expectedOf,
  openReferences,
  registerReferenceGate,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults } from "../../runtime/tests/helpers/results.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";

/**
 * Where the assets live (`models/karume-anima/` at the repository root — the 5 official variants
 * together, default = turbo-v1.1).
 */
const ASSETS_DIR = new URL("../../../models/karume-anima/", import.meta.url);
/**
 * The model the CFG check selects (the plain base — the default turbo is CFG=1 and never goes
 * through the uncond side).
 */
const BASE_MODEL = "anima-v1.0";
/** Where the reference digests live (rows per environment key — tracked). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/anima.json", import.meta.url),
);
/**
 * Where the artifacts and outcomes go (`outputs/verify/<environment key>/<date>_anima/` — safe to
 * delete).
 */
const results = openResults("anima");

/**
 * The default prompt of the demo it was migrated from (the exact string the reference PNG was baked
 * with). The negative prompt is not passed: with `guidanceScale 1` not a single character of it is
 * used (passing it fails loudly).
 */
const PROMPT =
  "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, outdoors, " +
  "smile, upper body, masterpiece, best quality";
const STEPS = 8;
const SEED = 42;

/**
 * The conditions of the three checks of the default model (turbo). The case ID is
 * `<quant>-<resolution>` (it doubles as the log label, the key in `fixtures/references/anima.json`
 * and the artifact's file name).
 *
 * Internal consistency: fromPretrained-512 and 512 / onEvent-1024 and 1024 are each bit-identical
 * (the same byte sequence through another route, observation with NaN poisoning included) — this is
 * part of the check as the twin check.
 */
const REFERENCE = [
  { quant: "f16+dit8-a8-attn8-s16", resolution: { width: 1024, height: 1024 } },
  { quant: "f16+dit8-a8-attn8-s16", resolution: { width: 512, height: 512 } },
  { quant: "f16", resolution: { width: 1024, height: 1024 } },
  // The faithful preset (all 4 roles are upstream's bf16 bit patterns — ADR 0131 decision 3,
  // extended to every model on 2026-10-17). Only one turbo case holds a row (the bf16 of the other
  // models is covered by the byte comparison with the upstream checkpoint at export time).
  { quant: "bf16", resolution: { width: 1024, height: 1024 } },
] as const satisfies readonly { quant: string; resolution: ImageSize }[];

/** The case ID (log label / key of the reference value / file name of the artifact). */
const caseIdOf = (quant: string, resolution: ImageSize): string =>
  `${quant}-${formatResolution(resolution)}`;

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
const ASSETS_AVAILABLE = manifestText !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} has no karume.json, so Anima's e2e SKIPs` +
      " (the port check closes only on real assets — bake them with the exporter's dist.py)",
  );
}

const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

/**
 * The distribution of the extra variants (`models/karume-anima-extra/` at the repository root —
 * anima-wai-v1.0 / anima-copycat-20260610). Its shared components are cross-repository references
 * into the official repository, so it runs only together with the official mirror
 * ({@link ASSETS_DIR}).
 */
const EXTRA_ASSETS_DIR = new URL("../../../models/karume-anima-extra/", import.meta.url);
const extraManifestText = await readTextIfPresent(new URL("karume.json", EXTRA_ASSETS_DIR));
if (extraManifestText === undefined) {
  console.warn(
    `[karume] ${EXTRA_ASSETS_DIR.pathname} has no karume.json, so the cross-repository reference e2e SKIPs` +
      " (bake it to a commit SHA of the official repository with the cross-repository reference steps of the release-runbook)",
  );
}
const EXTRA_RUNNABLE = RUNNABLE && extraManifestText !== undefined;

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const readManifest = (): Manifest => parseManifest(manifestText as string);

// --- Local HTTP (a throwaway HF-shaped server) ------------------------------------
//
// Almost every real-asset check sits on this base — to run the part-split distribution through the
// same route as production (graph-first + sequential streaming). The whole-asset surface
// (`Deno.readFile` + `fromAssets`) is held by exactly one check in the "whole-asset surface"
// section below. It speaks only the two routes the hub actually calls (the revision-resolution API
// and the resolve URL).

const REPO = "karume-test/anima";
const REVISION_SHA = "1234567890abcdef1234567890abcdef12345678";
const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/**
 * The repo → local mirror map of cross-repository references. The official repository is
 * self-contained (ADR 0087); the only things fetched across repositories are the shared components
 * of extra ({@link EXTRA_ASSETS_DIR}) (text_encoder / vae_decoder / the 2 tokenizers). A mirror is
 * per repo; the revision stays in the source key ({@link originKey}) as the manifest declares it —
 * if the mirror's contents disagree with that commit, the fetch layer's integrity verification
 * fails.
 */
const CROSS_REPO_MIRRORS: ReadonlyMap<string, URL> = new Map([["hdae/karume-anima", ASSETS_DIR]]);

/** What one source (repo + revision) serves. */
type ServedOrigin = { readonly dir: URL; readonly paths: Set<string> };

/**
 * The key of a source. **The repo alone is not enough** — the same repo appearing at another
 * revision must not be mixed up (a cross-repository reference is fetched from its declared commit
 * SHA, unrelated to the session's SHA).
 */
const originKey = (repo: string, revision: string): string => `${repo}@${revision}`;

type AssetServer = { readonly addr: { readonly port: number }; shutdown(): Promise<void> };

/**
 * A temporary server that serves the source table in HF's URL shape (automatic port, bound to
 * 127.0.0.1).
 */
const serveAssets = (origins: ReadonlyMap<string, ServedOrigin>): Promise<AssetServer> =>
  new Promise((listening) => {
    const notFound = (response: ServerResponse): void => {
      response.writeHead(404, { "content-type": "text/plain" }).end("not found");
    };
    const server = createServer((request, response) => {
      void (async () => {
        const { pathname } = new URL(request.url ?? "/", "http://127.0.0.1");
        // Only the session repo asks for revision resolution (a cross-repository reference is
        // pinned to a commit SHA, so the fetch layer issues no resolution request for it).
        if (REVISION_RE.test(pathname)) {
          response
            .writeHead(200, { "content-type": "application/json" })
            .end(JSON.stringify({ sha: REVISION_SHA }));
          return;
        }
        const resolved = RESOLVE_RE.exec(pathname);
        if (resolved === null) return notFound(response);
        const [, repo, revision, rawPath] = resolved;
        const served = origins.get(originKey(repo, revision));
        const path = decodeURIComponent(rawPath);
        if (served === undefined || !served.paths.has(path)) return notFound(response);
        const file = new URL(path, served.dir);
        const { size } = await stat(file);
        response.writeHead(200, { "content-length": String(size) });
        createReadStream(file).pipe(response);
      })();
    });
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address !== null ? address.port : 0;
      listening({
        addr: { port },
        shutdown: () =>
          new Promise((done, fail) => {
            server.close((error) => (error === undefined ? done() : fail(error)));
            server.closeAllConnections();
          }),
      });
    });
  });

/**
 * Groups every path `fromPretrained` fetches for this distribution / quant (the manifest itself
 * included) **by the source it is fetched from**. The session repo is served from `dir`;
 * cross-repository references are served from the mirrors of {@link CROSS_REPO_MIRRORS}, under the
 * (repo, revision) as declared.
 */
const servedOrigins = (
  manifest: Manifest,
  quant: string,
  dir: URL,
  model?: string,
): Map<string, ServedOrigin> => {
  const origins = new Map<string, ServedOrigin>();
  const add = (repo: string, revision: string, mirror: URL, path: string): void => {
    const key = originKey(repo, revision);
    const served = origins.get(key) ?? { dir: mirror, paths: new Set<string>() };
    served.paths.add(path);
    origins.set(key, served);
  };
  add(REPO, REVISION_SHA, dir, "karume.json");
  const selection = resolveSelection(manifest, {
    quant,
    ...(model === undefined ? {} : { model }),
  });
  const refs: FileRef[] = [
    ...Object.keys(selection.containers).flatMap((name) => [...selection.containers[name].parts]),
    ...Object.keys(selection.assets).map((name) => selection.assets[name]),
  ];
  for (const ref of refs) {
    if (ref.repo === undefined || ref.revision === undefined) {
      add(REPO, REVISION_SHA, dir, ref.path);
      continue;
    }
    const mirror = CROSS_REPO_MIRRORS.get(ref.repo);
    if (mirror === undefined) {
      throw new Error(
        `no local mirror for the cross-repository reference '${ref.repo}'` +
          ` (known: ${[...CROSS_REPO_MIRRORS.keys()].join(" / ")})`,
      );
    }
    add(ref.repo, ref.revision, mirror, ref.path);
  }
  return origins;
};

/**
 * A summary of the pixels (to say "how different a picture it is" when the reference sha
 * disagrees).
 */
const describePixels = (image: GeneratedImage): string => {
  const channels = ["R", "G", "B"];
  return channels
    .map((name, offset) => {
      let sum = 0;
      let min = 255;
      let max = 0;
      for (let at = offset; at < image.data.length; at += 4) {
        const value = image.data[at];
        sum += value;
        if (value < min) min = value;
        if (value > max) max = value;
      }
      return `${name} mean ${(sum / (image.width * image.height)).toFixed(2)} / range ${min}–${max}`;
    })
    .join(" / ");
};

/**
 * The report when the reference sha disagrees.
 *
 * MUST: no escaping into a tolerance here. The reference is only a sha256 (the byte sequence of the
 * reference PNG is not kept), so **the position of the first difference cannot be given, in
 * principle** — instead the artifact (kept in the results directory every time), the byte length
 * and the pixel statistics are put together so that a person can compare them.
 */
const mismatchReport = (
  label: string,
  image: GeneratedImage,
  png: Uint8Array<ArrayBuffer>,
  expected: string,
  actual: string,
  dumped: URL,
): string =>
  `${label}: the sha256 of the output pixels does not match the reference\n` +
  `  expected ${expected}\n  actual   ${actual}\n` +
  `  PNG ${png.length} bytes / image ${image.width}×${image.height}\n` +
  `  pixels ${describePixels(image)}\n` +
  `  artifact ${dumped.pathname} (the reference is a sha256 only, not the bytes, so the position of the first difference cannot be given)`;

/**
 * The pixel sequence the sha is taken over: the RGB8 `encodePng` writes into IDAT (row-major, alpha
 * dropped). `GeneratedImage.data` is RGBA with alpha always 255 (`encodePng` checks it), so no
 * information is lost, and it is the same byte sequence as the pixels of the decoded PNG artifact —
 * a row can be recomputed from the artifact (ADR 0106 addendum decision 9).
 */
const rgbPixels = (image: GeneratedImage): Uint8Array<ArrayBuffer> => {
  const pixels = new Uint8Array(new ArrayBuffer(image.width * image.height * 3));
  for (let source = 0, target = 0; source < image.data.length; source += 4, target += 3) {
    pixels[target] = image.data[source];
    pixels[target + 1] = image.data[source + 1];
    pixels[target + 2] = image.data[source + 2];
  }
  return pixels;
};

/**
 * Keeps the artifact, compares it with the reference value and appends the outcome to the results
 * JSON (the common tail of every check).
 *
 * `twin` is the case ID of the partner that "should produce the same bytes through another route".
 * Since the reference values became rows per environment, bit identity across routes can no longer
 * be held as "passing the same constant", so **the twin's row and the measurement** are compared
 * directly (only when the partner's row does not exist in this environment yet is there no check).
 */
const settlePng = async (
  label: string,
  image: GeneratedImage,
  png: Uint8Array<ArrayBuffer>,
  actual: string,
  elapsedMs: number,
  twin?: string,
): Promise<void> => {
  // This route can write reference values, so first check that the adapter the key was taken from
  // is the running adapter.
  await assertRunningAdapter();
  const artifact = `${label}.png`;
  // Keep the artifact every time, pass or fail (the material for the A/B of the next breakage is
  // needed before it breaks).
  const dumped = results.artifact(artifact);
  await writeFile(dumped, png);
  // Only **the verdict** of the comparison with the twin's row is made first. Throwing here would
  // leave the artifact in the results directory without this case's outcome, and it would be read
  // together with the previous outcome of the same day (if the twin has the same value, it is this
  // case's mismatch too).
  const twinSha = twin === undefined ? undefined : references.lookup(twin);
  const twinMismatch =
    twinSha === undefined || twinSha === actual
      ? undefined
      : `is not bit-identical to ${twin} (${twin} ${twinSha} / measured ${actual}) — ` +
        "the other route does not produce the same bytes";
  const check = references.check(label, actual);
  announceCheck(label, check, actual);
  const expected = expectedOf(check);
  await results.record({
    id: label,
    status: twinMismatch === undefined ? check.status : "fail",
    ...(expected === undefined ? {} : { expected }),
    actual,
    artifact,
    elapsedMs,
    ...(twinMismatch === undefined ? {} : { note: twinMismatch }),
  });
  if (twinMismatch !== undefined) throw new Error(`${label}'s PNG ${twinMismatch}`);
  if (check.status === "fail") {
    throw new Error(mismatchReport(label, image, png, check.expected, actual, dumped));
  }
};

/**
 * Generate → PNG → sha comparison. On a mismatch it does not loosen; it fails with diagnostics.
 *
 * A call with `onEvent` is required to produce **the same byte sequence** too — direct evidence
 * that the observation hook did not touch a single bit of the numbers (the onEvent check below;
 * `twin` compares it with the case without observation). `sampler` is the request-side override
 * (when omitted, the manifest's `scheduler.type`).
 */
const assertReferencePng = async (
  label: string,
  pipeline: Pipeline,
  resolution: ImageSize,
  options: {
    readonly sampler?: SamplerType;
    readonly onEvent?: (event: GenerateEvent) => void;
    readonly twin?: string;
  } = {},
): Promise<void> => {
  const { sampler, onEvent } = options;
  const started = performance.now();
  const image = await pipeline.generate({
    prompt: PROMPT,
    resolution,
    steps: STEPS,
    seed: SEED,
    ...(sampler === undefined ? {} : { sampler }),
    ...(onEvent === undefined ? {} : { onEvent }),
  });
  const png = await encodePng(image.data, image.width, image.height);
  const actual = await sha256Hex(rgbPixels(image));
  const elapsedMs = performance.now() - started;
  console.log(
    `[e2e] ${label}: ${(elapsedMs / 1000).toFixed(1)}s / PNG ${png.length}B / pixels sha256 ${actual}`,
  );
  await settlePng(label, image, png, actual, elapsedMs, options.twin);
};

/** Injection seats for building the turbo distribution. */
type TurboOptions = {
  readonly caches?: CacheStorage;
  readonly onRunDiagnostics?: (component: string, diagnostics: SessionDiagnostics) => void;
  readonly residency?: Residency;
  /** A shared GPU (owned by the caller — the caller destroys it). */
  readonly gpu?: GpuContext;
};

/**
 * An opened turbo distribution. Disposing it releases the pipeline through its
 * [Symbol.asyncDispose] and then stops the server; later disposals return the first completion,
 * so a describe can close it inside an it and again in afterAll.
 */
type OpenedTurbo = AsyncDisposable & { readonly pipeline: Pipeline };

/**
 * Builds the turbo distribution **through the fetch layer** (local HTTP + `fromPretrained`).
 * Returns it open: {@link withTurbo} and the describes that share one pipeline across their its
 * (opened in beforeAll, closed in afterAll) build it through this one path.
 *
 * MUST: `caches` is passed through the public injection point (no writing several GB into the real
 * Cache Storage). When the caller passes its own instance, its contents can be inspected after
 * close.
 */
const openTurbo = async (quant: string, options: TurboOptions): Promise<OpenedTurbo> => {
  const manifest = readManifest();
  const server = await serveAssets(servedOrigins(manifest, quant, ASSETS_DIR));
  let pipeline: Pipeline;
  try {
    pipeline = await Pipeline.fromPretrained(
      { repo: REPO, hubUrl: `http://127.0.0.1:${server.addr.port}` },
      {
        quant,
        caches: options.caches ?? new MemoryCacheStorage(),
        ...(options.onRunDiagnostics === undefined
          ? {}
          : { onRunDiagnostics: options.onRunDiagnostics }),
        ...(options.residency === undefined ? {} : { residency: options.residency }),
        ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      },
    );
  } catch (error) {
    await server.shutdown();
    throw error;
  }
  let closing: Promise<void> | undefined;
  const close = async (): Promise<void> => {
    try {
      // Intended to also check the release through [Symbol.asyncDispose] on this real-GPU route.
      await pipeline[Symbol.asyncDispose]();
    } finally {
      await server.shutdown();
    }
  };
  return {
    pipeline,
    [Symbol.asyncDispose]: () => (closing ??= close()),
  };
};

/** Builds the turbo distribution with {@link openTurbo}, passes it to `body`, then disposes it. */
const withTurbo = async (
  quant: string,
  options: TurboOptions,
  body: (pipeline: Pipeline) => Promise<void>,
): Promise<void> => {
  await using opened = await openTurbo(quant, options);
  await body(opened.pipeline);
};

for (const { quant, resolution } of REFERENCE) {
  const label = caseIdOf(quant, resolution);
  // That an explicit request-side `sampler:"euler"` is bit-identical to the manifest default (= the
  // official distribution's declaration) is checked **without adding a single generation**, by
  // switching only the shortest case, 512, to the explicit form. The default route's check is held
  // by the remaining cases and fromPretrained (which runs the same quant/512 with the default and
  // requires the same sha), so neither route is left uncovered.
  const sampler: SamplerType | undefined = resolution.width === 512 ? "euler" : undefined;
  test.skipIf(!RUNNABLE || references.lacksReference(label))(
    `e2e(real GPU): quant ${quant} / ${formatResolution(resolution)} / ${STEPS}step / ` +
      `seed ${SEED} PNG matches the reference sha256` +
      (sampler === undefined ? "" : ` (sampler:"${sampler}" explicit)`),
    async () => {
      await withTurbo(quant, {}, (pipeline) =>
        assertReferencePng(label, pipeline, resolution, {
          ...(sampler === undefined ? {} : { sampler }),
        }),
      );
    },
  );
}

// --- Request-side sampler override (DPM++ 2M) -----------------------------------
//
// The official distribution's manifest declares `scheduler.type: "euler"` (re-ruling 2026-08-25).
// That this override **reaches the math** is shown by the output of a request with `"dpmpp-2m"`
// being **bit-identical** to the golden measured with the same update rule declared on the manifest
// side (d2e0484 — retired in daf86af). The override route and the declaration route produce the
// same bytes ⇒ only how the effective sampler is decided differs, and the spelling of the numbers
// does not branch by a single bit.
//
// MUST: no 1024² override is added (the GPU time budget — the update rule is a host-side formula
// that does not depend on the resolution).

/**
 * The case that runs turbo 512² (the same quant / resolution / steps / seed as `REFERENCE[1]`) with
 * DPM++ 2M.
 */
const TURBO_DPMPP_CASE = `${caseIdOf(REFERENCE[1].quant, REFERENCE[1].resolution)}-dpmpp`;

test.skipIf(!RUNNABLE || references.lacksReference(TURBO_DPMPP_CASE))(
  `e2e(real GPU): turbo ${formatResolution(REFERENCE[1].resolution)} run with the request's ` +
    `sampler:"dpmpp-2m" matches the DPM++ 2M reference sha256`,
  async () => {
    const { quant, resolution } = REFERENCE[1];
    await withTurbo(quant, {}, (pipeline) =>
      assertReferencePng(TURBO_DPMPP_CASE, pipeline, resolution, { sampler: "dpmpp-2m" }),
    );
  },
);

// --- CFG (the plain base distribution) --------------------------------------------------
//
// MUST: **the CFG≠1 route also gets a real-GPU check**. `guidanceScale === 1` never computes the
// uncond side (`needsUncond`), so the 3 cases above go through cond only — the CFG branch, the
// second text path and the `cfgEulerStep` combination had **never once run on a real GPU** (a hole
// noticed in iteration L). The plain base distribution defaults to CFG, so exactly one case pins it
// (the 512 resolution is for the run time — with CFG one step is two forwards).

/** The conditions of the CFG check (the RTX 3080 Ti rows are the 2026-08-22 measurement). */
const BASE_REFERENCE = {
  quant: "f16+dit8-a8-attn8-s16",
  resolution: { width: 512, height: 512 },
  steps: 20,
  guidanceScale: 4,
  negativePrompt: "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
} as const;

/** One distribution served as the session repo (location, manifest, model to select). */
type DistributionSource = {
  readonly dir: URL;
  readonly manifest: Manifest;
  readonly model: string;
};

/**
 * Bakes one image with the {@link BASE_REFERENCE} knobs on a built base pipeline and compares the
 * sha.
 */
const assertBasePng = async (
  label: string,
  pipeline: Pipeline,
  options: { readonly sampler?: SamplerType; readonly twin?: string } = {},
): Promise<void> => {
  const { resolution, steps, guidanceScale, negativePrompt } = BASE_REFERENCE;
  const started = performance.now();
  const image = await pipeline.generate({
    prompt: PROMPT,
    resolution,
    steps,
    guidanceScale,
    negativePrompt,
    seed: SEED,
    ...(options.sampler === undefined ? {} : { sampler: options.sampler }),
  });
  const png = await encodePng(image.data, image.width, image.height);
  const actual = await sha256Hex(rgbPixels(image));
  const elapsedMs = performance.now() - started;
  console.log(
    `[e2e] ${label}: ${(elapsedMs / 1000).toFixed(1)}s / PNG ${png.length}B / pixels sha256 ${actual}`,
  );
  await settlePng(label, image, png, actual, elapsedMs, options.twin);
};

/**
 * Bakes one image of the plain base with the {@link BASE_REFERENCE} knobs and compares it with the
 * reference sha256.
 *
 * The route is **through the fetch layer** (local HTTP + `fromPretrained`) — this distribution's
 * text_encoder and transformer are part-split, and only the fetch layer resolves them together with
 * the cross-repository references (the same shape as the demo's `--source` —
 * `examples/shared/local-dist-server.ts`). The split form can also be read through the whole-asset
 * surface (`fromAssets`), but that surface holds every part in host RAM at once; as a check it is
 * closed separately by the one check below (which requires the same reference sha).
 *
 * `source` is the distribution served as the session repo (when omitted, the plain base of the
 * official repository). The cross-repository reference check (the "cross-repository references"
 * section below) uses it to run the extra variants with the same knobs. `quant` overrides the
 * preset (when omitted, the practical preset of {@link BASE_REFERENCE} — the check of the faithful
 * preset `bf16` uses it).
 */
const assertBaseReferencePng = async (
  label: string,
  options: {
    readonly sampler?: SamplerType;
    readonly source?: DistributionSource;
    readonly quant?: string;
  } = {},
): Promise<void> => {
  const { dir, manifest, model } = options.source ?? {
    dir: ASSETS_DIR,
    manifest: readManifest(),
    model: BASE_MODEL,
  };
  const quant = options.quant ?? BASE_REFERENCE.quant;
  const server = await serveAssets(servedOrigins(manifest, quant, dir, model));
  try {
    // MUST: `caches` is passed through the public injection point (no writing several GB into the
    // real Cache Storage).
    await using pipeline = await Pipeline.fromPretrained(
      { repo: REPO, hubUrl: `http://127.0.0.1:${server.addr.port}` },
      { model, quant, caches: new MemoryCacheStorage() },
    );
    await assertBasePng(label, pipeline, options);
  } finally {
    await server.shutdown();
  }
};

test.skipIf(!RUNNABLE || references.lacksReference("base-cfg"))(
  `e2e(real GPU): plain base / CFG ${BASE_REFERENCE.guidanceScale} / ` +
    `${BASE_REFERENCE.steps}step PNG matches the reference sha256`,
  () => assertBaseReferencePng("base-cfg"),
);

// The faithful preset `bf16` (ADR 0131 decision 3 — all 4 roles hold the BF16 bit patterns of the
// upstream checkpoint; compute is f32). It bakes one image with the same knobs as the CFG check and
// compares it with this machine's row. Its rounding differs from the f16 preset (f16 has a narrower
// exponent range), so it holds its own row and is not a twin. The bf16 weights are expanded by a
// bitcast inside the kernel (ADR 0131 decision 1) — this check pins that this route works on the
// real assets of the distribution too.
test.skipIf(!RUNNABLE || references.lacksReference("base-cfg-bf16"))(
  `e2e(real GPU): the plain base's faithful preset bf16 / CFG ${BASE_REFERENCE.guidanceScale} / ` +
    `${BASE_REFERENCE.steps}step PNG matches the reference sha256`,
  () => assertBaseReferencePng("base-cfg-bf16", { quant: "bf16" }),
);

// --- Reading the split distribution through the whole-asset surface (fromAssets) ---------
//
// The check of "a distribution that `fromPretrained` reads, `fromAssets` reads too" (X2-101). In
// the real distribution, one container of a component is split into several parts, which are passed
// to the whole-asset surface spelled `transformer[0]` / `transformer[1]` / … — that Record goes to
// the whole-asset surface as is, and **the same reference sha256 as the CFG check above** is
// required. It only feeds the same bytes in through another surface, so not moving a single bit is
// correct (if it moves, the part order or the descriptor handling is broken).

/** Reads a local distribution whole (the same surface as `examples/shared/local-assets.ts`). */
const readLocalAssets = async (
  dir: URL,
  manifest: Manifest,
  quant: string,
): Promise<{ manifest: Manifest; assets: Record<string, Uint8Array<ArrayBuffer>> }> => {
  const selection = resolveSelection(manifest, { quant, model: BASE_MODEL });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  const read = async (key: string, ref: FileRef): Promise<void> => {
    // A cross-repository reference is fetchable only from (repo, commit SHA) — the whole-asset
    // surface cannot resolve it, so fail loudly.
    if (ref.repo !== undefined) {
      throw new Error(
        `the whole-asset surface cannot resolve the cross-repository reference '${ref.repo}' (key ${key})`,
      );
    }
    const bytes = byPath.get(ref.path) ?? (await readFile(new URL(ref.path, dir)));
    byPath.set(ref.path, bytes);
    assets = { ...assets, [key]: bytes };
  };
  // MUST: zero-length parts are listed too (the index is the id inside the container — skipping one
  // shifts every later one by one).
  for (const name of Object.keys(selection.containers)) {
    const { parts } = selection.containers[name];
    for (const [index, ref] of parts.entries()) await read(`${name}[${index}]`, ref);
  }
  for (const name of Object.keys(selection.assets)) await read(name, selection.assets[name]);
  return { manifest, assets };
};

test.skipIf(!RUNNABLE || references.lacksReference("base-cfg-fromAssets-shards"))(
  "e2e(real GPU): the split distribution built with fromAssets (whole-asset surface) still matches the plain base's reference sha256",
  async () => {
    const { quant } = BASE_REFERENCE;
    const input = await readLocalAssets(ASSETS_DIR, readManifest(), quant);
    // This check confirms the split form itself — the day the distribution goes back to the single
    // form, it fails with a reason instead of silently turning into "the whole-asset surface check"
    // (losing the check's meaning is the greater danger).
    const partKeys = Object.keys(input.assets).filter((key) => key.endsWith("]"));
    if (partKeys.length < 2) {
      throw new Error(
        `this distribution is not part-split (keys: ${Object.keys(input.assets).join(" / ")})`,
      );
    }
    await using pipeline = await Pipeline.fromAssets(input, { model: BASE_MODEL, quant });
    await assertBasePng("base-cfg-fromAssets-shards", pipeline, { twin: "base-cfg" });
  },
);

/**
 * The plain base switched to DPM++ 2M with the same knobs as {@link BASE_REFERENCE} (CFG 4 / 20step
 * / seed 42 / the same negative prompt) (the RTX 3080 Ti row is the golden of d2e0484 — retired in
 * daf86af).
 *
 * The CFG≠1 route also gets an override check — the second-order term of DPM++ 2M keeps in its
 * history the value **after** combining cond/uncond, so matching on turbo (no CFG) alone would
 * leave the interplay of combination and history uncovered.
 */
test.skipIf(!RUNNABLE || references.lacksReference("base-cfg-dpmpp"))(
  `e2e(real GPU): the plain base run with the request's sampler:"dpmpp-2m" ` +
    `matches the DPM++ 2M reference sha256`,
  () => assertBaseReferencePng("base-cfg-dpmpp", { sampler: "dpmpp-2m" }),
);

// --- Cross-repository references (the extra variants) ------------------------------------
//
// The extra variants hold only transformer / text_conditioner themselves and bake text_encoder /
// vae_decoder / the 2 tokenizers as cross-repository references to a (repo, commit SHA) of the
// official repository (ADR 0038 §7). That the fetch layer fetches them **under the (repo, revision)
// as declared** is run on the real assets by having the local HTTP server serve each source
// separately — not a single cross-repository path is put on the session repo, so if the fetch layer
// mixes them up with the session repo, it fails with a 404.
//
// The knobs are the same as the CFG check of the plain base (every extra variant is a fine-tune
// that defaults to CFG).

/** The extra variants (case ID = `extra-<model>-cfg`). */
const EXTRA_MODELS = ["anima-wai-v1.0", "anima-copycat-20260610"] as const;
const extraCaseIdOf = (model: string): string => `extra-${model}-cfg`;

for (const model of EXTRA_MODELS) {
  const label = extraCaseIdOf(model);
  test.skipIf(!EXTRA_RUNNABLE || references.lacksReference(label))(
    `e2e(real GPU): extra ${model} built with its cross-repository references, CFG ${BASE_REFERENCE.guidanceScale} / ` +
      `${BASE_REFERENCE.steps}step PNG matches the reference sha256`,
    async () => {
      const manifest = parseManifest(extraManifestText as string);
      const sessionOrigin = originKey(REPO, REVISION_SHA);
      const crossOrigins = [
        ...servedOrigins(manifest, BASE_REFERENCE.quant, EXTRA_ASSETS_DIR, model).keys(),
      ].filter((key) => key !== sessionOrigin);
      // Fail the day the distribution goes back to one with no cross-repository reference, instead
      // of silently turning into "the cross-repository check".
      if (crossOrigins.length === 0) {
        throw new Error(
          `the selection of ${model} contains no cross-repository reference (the check loses its meaning)`,
        );
      }
      await assertBaseReferencePng(label, {
        source: { dir: EXTRA_ASSETS_DIR, manifest, model },
      });
    },
  );
}

// --- fromPretrained (fetch layer included) --------------------------------------------
//
// The **real route** `loadManifest` → `resolveSelection` → fetch → build is now run by every turbo
// check, but this is the only one that compares **the injection points with the cache contents**.
// It is also the route that runs integrity verification with the manifest's real sha256, and it
// doubles as the check that the asset triples (path/size/sha256) match the files.
//
// The resolution is 512 for RAM reasons (the fetch layer's memory cache holds copies of the
// assets). As a bit-identity check it is on par with 1024 — the twin check requires it to produce
// **the same bytes** as the 512 case above.

test.skipIf(!RUNNABLE || references.lacksReference("fromPretrained-512"))(
  "e2e(real GPU): the fromPretrained (fetch layer + integrity verification) PNG matches the reference sha256",
  async () => {
    const quant = "f16+dit8-a8-attn8-s16";
    const resolution: ImageSize = { width: 512, height: 512 };
    const fetched = selectionRefs(resolveSelection(readManifest(), { quant }));
    const caches = new MemoryCacheStorage();
    // This real generation also pins the run counts of the observation hook (onRunDiagnostics) (the
    // DiT is 1 step = 1 run; each text component 1 run). An extra observation that has no effect on
    // the check's sha256 verdict.
    const observed: string[] = [];
    await withTurbo(
      quant,
      { caches, onRunDiagnostics: (component) => observed.push(component) },
      (pipeline) =>
        assertReferencePng("fromPretrained-512", pipeline, resolution, {
          twin: caseIdOf(REFERENCE[1].quant, REFERENCE[1].resolution),
        }),
    );
    const counts = new Map<string, number>();
    for (const component of observed) counts.set(component, (counts.get(component) ?? 0) + 1);
    const runsOf = (component: string): number => counts.get(component) ?? 0;
    if (
      runsOf("text_encoder") !== 1 ||
      runsOf("text_conditioner") !== 1 ||
      runsOf("transformer") !== STEPS ||
      runsOf("vae_decoder") < 1
    ) {
      throw new Error(
        "the observation hook's run counts do not match the execution structure: " +
          `text_encoder ${runsOf("text_encoder")} (expected 1) / ` +
          `text_conditioner ${runsOf("text_conditioner")} (expected 1) / ` +
          `transformer ${runsOf("transformer")} (expected ${STEPS}) / ` +
          `vae_decoder ${runsOf("vae_decoder")} (expected 1 or more)`,
      );
    }
    // The files being on the injected side shows that the injection point reaches the fetch layer
    // at the end (= the real cache was not polluted). The fetch layer owns the namespace names, so
    // every namespace is counted. The expectation is **1 manifest + the number of unique
    // (repo, path, sha256)** — an asset's key is the content key of the fetch layer's HF layer
    // `["hf", "model", repo, path, sha256]` (the hub's `sources/hf.ts`), which includes the path,
    // so identical bytes under another path are a separate entry (for cross-repository references
    // only the repo changes to the target).
    // NOTE: the hub does not expose how this key is built, so the same 3 elements are assembled here.
    let entries = 0;
    for (const namespace of caches.namespaces.values()) entries += namespace.entries.size;
    const contentKeys = new Set(
      fetched.map((ref) => JSON.stringify([ref.repo ?? REPO, ref.path, ref.sha256])),
    );
    const expected = contentKeys.size + 1;
    if (entries !== expected) {
      throw new Error(`the injected cache has ${entries} entries (expected ${expected})`);
    }
  },
);

// --- onEvent (the observation hook of generation events)
// ------------------------------------------
//
// The check of the observation hook itself. It pins 4 points:
//  ① The event sequence **matches the execution structure exactly** (4 stages × start/end / denoise
//     `steps` times / one event per VAE tile). The tile count is taken from the `vae_decoder` run
//     count of `onRunDiagnostics` (two independent observations are compared instead of
//     reimplementing the tile geometry).
//  ② The shape of `copyLatents()` matches the latent shape (resolution ÷ spatial compression), and
//     so does the element count.
//  ③ **The PNG's sha256 is identical to the reference value without observation.** The subscriber
//     receives a copy every step and even corrupts it with NaN — that the picture still does not
//     move by a single bit is direct evidence that "observation has no side effects" (had the
//     internal array been passed instead of a copy, this would always break).
//  ④ A throw in the callback brings the generation down, and the `dispose` after it still works (a
//     by-product of not swallowing exceptions = a way to abort at step granularity).

describe.skipIf(!RUNNABLE || references.lacksReference("onEvent-1024"))(
  "e2e(real GPU): onEvent observation does not touch the numbers (event sequence / intermediate latents / abort)",
  () => {
    const quant = "f16+dit8-a8-attn8-s16";
    const resolution: ImageSize = { width: 1024, height: 1024 };
    const runs: string[] = [];
    let scheduler: ReturnType<typeof parseAnimaPipelineConfig>["scheduler"];
    let pipeline: Pipeline;
    let turbo: OpenedTurbo | undefined;
    beforeAll(async () => {
      const manifest = readManifest();
      ({ scheduler } = parseAnimaPipelineConfig(
        manifest.models[manifest.defaultModel].pipelineConfig,
      ));
      turbo = await openTurbo(quant, {
        onRunDiagnostics: (component) => runs.push(component),
      });
      pipeline = turbo.pipeline;
    });
    // After both its: dispose, then check that the disposed pipeline refuses to generate. Releasing
    // still works after an aborted generation (the serialisation chain does not carry a failure to
    // the next one); that is exercised here.
    afterAll(async () => {
      if (turbo === undefined) return;
      await turbo[Symbol.asyncDispose]();
      await assertRejects(
        () => pipeline.generate({ prompt: PROMPT, steps: 2, seed: SEED }),
        Error,
        "after dispose",
      );
    });

    it("pins the event sequence and the intermediate latents, and the PNG keeps the reference sha256", async () => {
      const log: string[] = [];
      const shapes: string[] = [];
      const lengths: number[] = [];
      let firstStep: (() => { data: Float32Array<ArrayBuffer> }) | undefined;
      let firstStepData: Float32Array | undefined;
      await assertReferencePng("onEvent-1024", pipeline, resolution, {
        twin: caseIdOf(REFERENCE[0].quant, REFERENCE[0].resolution),
        onEvent: (event) => {
          if (event.kind === "stage") {
            log.push(`stage:${event.component}:${event.at}`);
            return;
          }
          if (event.kind === "vae-tile") {
            log.push(`tile:${event.tile}/${event.tiles}`);
            return;
          }
          // Not emitted by the default per-stage execution — if it is, it disagrees with the
          // expected sequence below and fails.
          if (event.kind === "residency") {
            log.push(`residency:${event.action}/${event.reason}`);
            return;
          }
          log.push(`step:${event.step}/${event.steps}@${event.sigma}`);
          const snapshot = event.copyLatents();
          shapes.push(snapshot.shape.join("x"));
          lengths.push(snapshot.data.length);
          // Corrupt the copy. Had the internal array been passed, later steps would be NaN-poisoned
          // and the PNG's sha would break.
          snapshot.data.fill(Number.NaN);
          if (event.step === 1) {
            firstStep = event.copyLatents;
            firstStepData = event.copyLatents().data;
          }
        },
      });

      // ① An exact match with the execution structure. The VAE tile count comes from the
      //   diagnostics' run count (an independent observation).
      const tiles = runs.filter((component) => component === "vae_decoder").length;
      const sigmas = sigmaSchedule(STEPS, scheduler.shift);
      assertEquals(
        log,
        [
          "stage:text_encoder:start",
          "stage:text_encoder:end",
          "stage:text_conditioner:start",
          "stage:text_conditioner:end",
          "stage:transformer:start",
          ...Array.from({ length: STEPS }, (_, at) => `step:${at + 1}/${STEPS}@${sigmas[at]}`),
          "stage:transformer:end",
          "stage:vae_decoder:start",
          ...Array.from({ length: tiles }, (_, at) => `tile:${at + 1}/${tiles}`),
          "stage:vae_decoder:end",
        ],
        "the event sequence does not match the execution structure (stage order / step count / tile count)",
      );

      // ② The shape of the intermediate latents. Derived from the resolution (no 128 written into
      //   the test).
      const latentHeight = resolution.height / ANIMA_SPATIAL_COMPRESSION;
      const latentWidth = resolution.width / ANIMA_SPATIAL_COMPRESSION;
      const channels = lengths[0] / (latentHeight * latentWidth);
      assertEquals(
        new Set(shapes),
        new Set([`1x${channels}x${latentHeight}x${latentWidth}`]),
        `the intermediate latent shape is not [1,C,${latentHeight},${latentWidth}]`,
      );
      assertEquals(
        new Set(lengths),
        new Set([channels * latentHeight * latentWidth]),
        "the intermediate latent element count does not match its shape",
      );

      // ③ Called after the generation finished, it still returns the step 1 values (lazy, but bound
      //   when created).
      assertEquals(
        Array.from((firstStep as () => { data: Float32Array<ArrayBuffer> })().data.slice(0, 8)),
        Array.from((firstStepData as Float32Array).slice(0, 8)),
        "copyLatents called after the generation does not return the step 1 values",
      );
    });

    it("a throw in the callback brings the whole generation down (abort at step granularity)", async () => {
      const seen: string[] = [];
      const abortAt = 2;
      await assertRejects(
        () =>
          pipeline.generate({
            prompt: PROMPT,
            resolution: { width: 512, height: 512 },
            steps: abortAt,
            seed: SEED,
            onEvent: (event) => {
              seen.push(
                event.kind === "stage"
                  ? `stage:${event.component}:${event.at}`
                  : event.kind === "denoise-step"
                    ? `step:${event.step}`
                    : event.kind === "residency"
                      ? `residency:${event.action}/${event.reason}`
                      : `tile:${event.tile}`,
              );
              if (event.kind === "denoise-step" && event.step === abortAt) {
                throw new Error("aborted by the subscriber");
              }
            },
          }),
        Error,
        "aborted by the subscriber",
      );
      // Exceptions are not swallowed, so it stops in the DiT stage — the VAE stage never opens.
      assertEquals(seen.at(-1), `step:${abortAt}`, "the aborted step is not the last event");
      assertFalse(
        seen.includes("stage:transformer:end"),
        "the stage ended normally despite the abort",
      );
      assertFalse(
        seen.some((entry) => entry.startsWith("stage:vae_decoder")),
        "it entered the VAE stage despite the abort",
      );
    });
  },
);

// --- Public contract (serialisation / dispose / ownership of a shared GPU)
// --------------------------
//
// `Pipeline` states 4 public contracts: ① concurrent `generate` calls wait and run in order
// ② `dispose` waits for the in-flight work to finish (flush-before-destroy) ③ the second and later
// `dispose` return the same completion ④ a passed `GpuContext` is not destroyed. The GPU-less
// `anima_pipeline_test.ts` cannot assemble the **wiring** because the constructor is private, and
// `serial_test.ts` only constrains `createOperationChain` on its own. The wiring itself is visible
// only on a real pipeline, so it lives here.
//
// MUST: generations run at the minimum size (512² / 2 steps) — what is checked here is order, not
// pixels.

describe.skipIf(!RUNNABLE)(
  "e2e(real GPU): concurrent generate calls run serially, and dispose waits for the in-flight work before destroying",
  () => {
    const { quant } = REFERENCE[1];
    const resolution: ImageSize = { width: 512, height: 512 };
    const request = { prompt: PROMPT, resolution, steps: 2, seed: SEED };
    let pipeline: Pipeline;
    let turbo: OpenedTurbo | undefined;
    beforeAll(async () => {
      turbo = await openTurbo(quant, {});
      pipeline = turbo.pipeline;
    });
    afterAll(() => turbo?.[Symbol.asyncDispose]());

    it("the stages of two concurrently issued calls do not interleave", async () => {
      const log: string[] = [];
      const trace = (tag: string) => (event: GenerateEvent) => {
        if (event.kind === "stage") log.push(`${tag}:${event.component}:${event.at}`);
      };
      // Issue two without awaiting (if the chain came off, the stages would appear interleaved).
      await Promise.all([
        pipeline.generate({ ...request, onEvent: trace("a") }),
        pipeline.generate({ ...request, onEvent: trace("b") }),
      ]);
      const firstEnd = log.indexOf("a:vae_decoder:end");
      const secondStart = log.indexOf("b:text_encoder:start");
      assertEquals(firstEnd >= 0 && secondStart >= 0, true, `the stage log is incomplete: ${log}`);
      assertEquals(
        firstEnd < secondStart,
        true,
        `the second call started before the first call's last stage (the chain came off): ${log}`,
      );
    });

    it("dispose resolves after the generation completes, and a second call returns the same completion", async () => {
      const order: string[] = [];
      const running = pipeline.generate(request).then(() => {
        order.push("generate");
      });
      const disposal = pipeline.dispose();
      assertStrictEquals(
        pipeline.dispose(),
        disposal,
        "the second dispose returns a different completion",
      );
      await Promise.all([running, disposal.then(() => order.push("dispose"))]);
      // If the disposal resolved first, the consumer would treat it as "released" and move on.
      assertEquals(order, ["generate", "dispose"]);
    });
  },
);

test.skipIf(!RUNNABLE)(
  "e2e(real GPU): a passed GpuContext is not destroyed by dispose (the passing side owns it)",
  async () => {
    const { quant } = REFERENCE[1];
    const server = await serveAssets(servedOrigins(readManifest(), quant, ASSETS_DIR));
    // A shared GPU belongs to the caller. Destroying it would drag down the other pipelines using
    // the same device (the examples pass one GpuContext around between families).
    const gpu = await acquireGpu();
    try {
      const hub = { repo: REPO, hubUrl: `http://127.0.0.1:${server.addr.port}` };
      const first = await Pipeline.fromPretrained(hub, {
        quant,
        gpu,
        caches: new MemoryCacheStorage(),
      });
      await first.dispose();
      // If the device had been destroyed, building the second one (the Session's weight upload)
      // would fail.
      const second = await Pipeline.fromPretrained(hub, {
        quant,
        gpu,
        caches: new MemoryCacheStorage(),
      });
      await second.dispose();
    } finally {
      gpu.destroy();
      await server.shutdown();
    }
  },
);

// --- DiT residency (residency — ADR 0112) ------------------------------------------
//
// Residency is designed not to change the numbers, so **not a single new reference row is added**
// (`fixtures/references/anima.json` staying unchanged is the check). Every case runs as a twin of
// an existing row (turbo 1024 / 512) and requires the same bytes whether it goes through residency
// or not, a resolution round trip, carrying over and releasing, rebuilding after an abort, or a
// pre-emptive eviction for lack of free memory.
//
// The lifetime is observed through the transformer diagnostics of `onRunDiagnostics` (when the same
// Session is reused, a derived plan is hit across generate calls and neither the backing build
// count nor the weight byte count grows — per-stage execution builds a new Session for every
// generate, so its first run always misses).
//
// MUST: the twin comparison does not go through `references.check` / `record` (doing so would write
// new rows in the recording modes). On a machine without the twin's row in this environment the
// case SKIPs (the row is created by running the existing case).

/** The existing rows used as residency twins (turbo 1024 / 512 — 0 and 1 of {@link REFERENCE}). */
const RESIDENCY_TWINS = {
  1024: caseIdOf(REFERENCE[0].quant, REFERENCE[0].resolution),
  512: caseIdOf(REFERENCE[1].quant, REFERENCE[1].resolution),
} as const;

const RESIDENCY_RUNNABLE =
  RUNNABLE &&
  references.lookup(RESIDENCY_TWINS[1024]) !== undefined &&
  references.lookup(RESIDENCY_TWINS[512]) !== undefined;

/** Turns one event into its log spelling (stages / residency only — counting steps and tiles is enough). */
const residencyLogOf = (event: GenerateEvent): string | undefined => {
  if (event.kind === "stage") return `stage:${event.component}:${event.at}`;
  if (event.kind === "residency") return `residency:${event.action}/${event.reason}`;
  return undefined;
};

/**
 * Bakes one image as a twin of an existing row and compares them (no row is written). Returns the
 * stage / residency event sequence of that generate. The artifact and the outcome go to the results
 * directory (`label` is the result id — not the key of a reference row). `onLogged` is called and
 * awaited each time an entry is appended to the event sequence (the hook that changes the GPU state
 * in the middle of a generate — `onEvent` is awaited, so the next stage does not start until it
 * returns).
 */
const assertResidencyTwin = async (
  label: string,
  pipeline: Pipeline,
  size: 512 | 1024,
  residency: Residency | undefined,
  onLogged?: (entry: string) => Promise<void>,
): Promise<string[]> => {
  const twin = RESIDENCY_TWINS[size];
  const log: string[] = [];
  const started = performance.now();
  const image = await pipeline.generate({
    prompt: PROMPT,
    resolution: { width: size, height: size },
    steps: STEPS,
    seed: SEED,
    ...(residency === undefined ? {} : { residency }),
    onEvent: async (event) => {
      const entry = residencyLogOf(event);
      if (entry === undefined) return;
      log.push(entry);
      await onLogged?.(entry);
    },
  });
  const png = await encodePng(image.data, image.width, image.height);
  const actual = await sha256Hex(rgbPixels(image));
  const elapsedMs = performance.now() - started;
  console.log(
    `[e2e] ${label}: ${(elapsedMs / 1000).toFixed(1)}s / PNG ${png.length}B / pixels sha256 ${actual}`,
  );
  await assertRunningAdapter();
  const expected = references.lookup(twin);
  if (expected === undefined) {
    throw new Error(`no row of ${twin} in this environment (contradicts the SKIP condition)`);
  }
  const artifact = `${label}.png`;
  const dumped = results.artifact(artifact);
  await writeFile(dumped, png);
  await results.record({
    id: label,
    status: actual === expected ? "pass" : "fail",
    expected,
    actual,
    artifact,
    elapsedMs,
    note: `twin of ${twin} (DiT residency — holds no reference row)`,
  });
  if (actual !== expected) {
    throw new Error(
      mismatchReport(`${label} (twin ${twin})`, image, png, expected, actual, dumped),
    );
  }
  return log;
};

/** An observation hook that groups the per-run transformer diagnostics by generate. */
const transformerRuns = () => {
  const perGenerate: SessionDiagnostics[][] = [];
  return {
    perGenerate,
    /** Starts the group of the next generate (call it right before the generate). */
    begin: () => perGenerate.push([]),
    observe: (component: string, diagnostics: SessionDiagnostics): void => {
      if (component === "transformer") perGenerate.at(-1)?.push(diagnostics);
    },
  };
};

/**
 * The expected sequence of each stage's `stage` start / end with the residency events. Residency
 * events appear in 3 places only: right before the end of the DiT stage (`residency`) and the 2
 * points where the pre-emptive eviction `evicted/headroom` appears — before the start of
 * text_encoder (`beforeText`) and between the end of transformer and the start of vae_decoder
 * (`beforeVae`). The last 2 are empty by default, so except in the cases that deliberately provoke
 * a pre-emptive eviction, it pins "nothing appears at either point" as well.
 */
const expectedStages = (
  residency: readonly string[],
  headroom: {
    readonly beforeText?: readonly string[];
    readonly beforeVae?: readonly string[];
  } = {},
): string[] => [
  ...(headroom.beforeText ?? []).map((entry) => `residency:${entry}`),
  "stage:text_encoder:start",
  "stage:text_encoder:end",
  "stage:text_conditioner:start",
  "stage:text_conditioner:end",
  "stage:transformer:start",
  ...residency.map((entry) => `residency:${entry}`),
  "stage:transformer:end",
  ...(headroom.beforeVae ?? []).map((entry) => `residency:${entry}`),
  "stage:vae_decoder:start",
  "stage:vae_decoder:end",
];

describe.skipIf(!RESIDENCY_RUNNABLE)(
  'e2e(real GPU): residency "transformer" matches the reference sha256 twins on a second run of the same case and on a resolution round trip,' +
    " without reloading the DiT",
  () => {
    const { quant } = REFERENCE[0];
    const runs = transformerRuns();
    let pipeline: Pipeline;
    let turbo: OpenedTurbo | undefined;
    beforeAll(async () => {
      turbo = await openTurbo(quant, { residency: "transformer", onRunDiagnostics: runs.observe });
      pipeline = turbo.pipeline;
    });
    // The checks over every generate of (a) and (b) run after both its, whatever their outcome,
    // and before the pipeline is disposed.
    afterAll(async () => {
      if (turbo === undefined) return;
      try {
        const all = runs.perGenerate.flat();
        for (const diagnostics of all) {
          assertEquals(
            diagnostics.planBacking.retainedCount <= 1,
            true,
            `${diagnostics.planBacking.retainedCount} backings are retained`,
          );
        }
        // The weight arena also holds the params cache (the doc of `SessionDiagnostics.weights`),
        // so a run that derives a new plan at another resolution grows the count and the bytes by
        // its params. The exact form of "the weights were not reloaded" is "the count added after
        // the first run = the params count each run allocated" — a single re-upload of an
        // initializer makes it larger than the right-hand side.
        const [first, ...later] = all;
        const paramsAllocated = later.reduce(
          (sum, diagnostics) => sum + (diagnostics.lastRunParams?.allocCount ?? 0),
          0,
        );
        assertEquals(
          (later.at(-1) ?? first).weights.allocCount - first.weights.allocCount,
          paramsAllocated,
          "across generate calls the weight arena gained allocations other than params (the weights were reloaded)",
        );
      } finally {
        await turbo[Symbol.asyncDispose]();
      }
    });

    const generate = async (label: string, size: 512 | 1024): Promise<void> => {
      runs.begin();
      const log = await assertResidencyTwin(label, pipeline, size, undefined);
      assertEquals(log, expectedStages(["retained/request"]), `event sequence of ${label}`);
    };
    const weightsOf = (diagnostics: SessionDiagnostics): number =>
      diagnostics.weights.allocatedBytes;

    it("(a) 1024² twice: both runs match the twin's row, and the second hits both the plan and the backing", async () => {
      await generate("residency-1024-first", 1024);
      await generate("residency-1024-second", 1024);
      const [first, second] = runs.perGenerate;
      assertEquals(first.length, STEPS);
      assertEquals(second.length, STEPS);
      // Under per-stage execution the second run's first run is a new Session and always misses
      // (direct evidence of residency).
      assertEquals(
        second[0].lastRunPrepared?.hit,
        true,
        "the first run of the second generate does not hit a derived plan",
      );
      assertEquals(
        second.at(-1)?.planBacking.buildCount,
        first.at(-1)?.planBacking.buildCount,
        "the second generate rebuilds the backing",
      );
      assertEquals(
        weightsOf(second[0]),
        weightsOf(first[0]),
        "the weight bytes grew on the second run",
      );
    });

    it("(b) a 512² → 1024² → 512² round trip: each matches its resolution's twin, and at most 1 backing is retained", async () => {
      await generate("residency-roundtrip-512-first", 512);
      await generate("residency-roundtrip-1024", 1024);
      await generate("residency-roundtrip-512-second", 512);
    });
  },
);

describe.skipIf(!RESIDENCY_RUNNABLE)(
  "e2e(real GPU): request override of residency (retain → release), rebuilding after an abort, and dispose with a resident DiT",
  () => {
    const { quant } = REFERENCE[1];
    const runs = transformerRuns();
    let pipeline: Pipeline;
    let turbo: OpenedTurbo | undefined;
    beforeAll(async () => {
      // The build options stay at the default (per-stage); only the request switches residency.
      turbo = await openTurbo(quant, { onRunDiagnostics: runs.observe });
      pipeline = turbo.pipeline;
    });
    afterAll(() => turbo?.[Symbol.asyncDispose]());

    it("(c) request 1 = transformer retains it, and request 2 = per-stage uses it and then releases it", async () => {
      runs.begin();
      assertEquals(
        await assertResidencyTwin("residency-request-retain", pipeline, 512, "transformer"),
        expectedStages(["retained/request"]),
      );
      runs.begin();
      assertEquals(
        await assertResidencyTwin("residency-request-release", pipeline, 512, "per-stage"),
        expectedStages(["released/request"]),
      );
      // Request 2 used the carried-over DiT (no reload).
      assertEquals(
        runs.perGenerate[1][0].lastRunPrepared?.hit,
        true,
        "the carried-over DiT was not used",
      );
    });

    it("(d) aborting the DiT stage with a throw in onEvent drops the resident DiT, and the next generate rebuilds it and matches", async () => {
      runs.begin();
      await assertResidencyTwin("residency-before-abort", pipeline, 512, "transformer");
      const seen: string[] = [];
      const abortAt = 2;
      await assertRejects(
        () =>
          pipeline.generate({
            prompt: PROMPT,
            resolution: { width: 512, height: 512 },
            steps: abortAt,
            seed: SEED,
            onEvent: (event) => {
              seen.push(
                event.kind === "denoise-step"
                  ? `step:${event.step}`
                  : (residencyLogOf(event) ?? `tile:${event.kind}`),
              );
              if (event.kind === "denoise-step" && event.step === abortAt) {
                throw new Error("aborted by the subscriber");
              }
            },
          }),
        Error,
        "aborted by the subscriber",
      );
      // Right after the abort it reports dropping the resident DiT, and the stage does not end
      // normally (no end).
      assertEquals(seen.slice(-2), [`step:${abortAt}`, "residency:evicted/failure"]);
      assertFalse(
        seen.includes("stage:transformer:end"),
        "the stage ended normally despite the abort",
      );

      runs.begin();
      assertEquals(
        await assertResidencyTwin("residency-after-abort", pipeline, 512, "transformer"),
        expectedStages(["retained/request"]),
      );
      // After the drop, the first run is a new Session (it does not hit a derived plan).
      assertEquals(
        runs.perGenerate.at(-1)?.[0].lastRunPrepared?.hit,
        false,
        "a DiT that should have been dropped is in use",
      );
    });
    // NOTE: what is pinned is only "a dispose with the residency held completes without rejecting
    // and accepts no later generation". That the resident DiT was actually torn down is not
    // observed — GpuContext / device have no live-byte diagnostics, and checking on a shared GPU
    // that "a DiT-sized allocation succeeds after dispose" passes even with the residency left on a
    // machine with VRAM to spare (it does not discriminate). The order "tear down the resident DiT,
    // then destroy the GPU" and the handling of failures are pinned without a GPU by
    // `disposeResidencyThenGpu` in `anima_residency_test.ts`.
    it("(e) a dispose with the residency held completes without throwing, and no generation is accepted afterwards", async () => {
      // The dispose happens here, with the DiT still retained by the last generate of (d)
      // (it ends retained); afterAll's dispose then returns this same completion.
      await turbo?.[Symbol.asyncDispose]();
      await assertRejects(
        () => pipeline.generate({ prompt: PROMPT, steps: 2, seed: SEED }),
        Error,
        "after dispose",
      );
    });
  },
);

// Pre-emptive eviction (H-35): actually shrink the free memory above the resident DiT and check
// that generate evicts the resident DiT before it lays out a stage and gets through (without
// hitting OOM). There are 2 measuring points (the "two lines of eviction" in `residency.ts`), so
// there are 2 tests. The dummies are not written to (`createBuffer` only — the check side does not
// walk a route where the staging of a write could lose the device).
//
// - Before the text stage: before the second generate, "stack 1 GiB dummies until OOM, then return
//   one" — the remainder is 1–2 GiB (just short of the allocation line), smaller than what
//   text_encoder needs (the estimated peak + the largest part + the margin — `stageMemoryNeed` in
//   `residency.ts`), and evicting the resident DiT lets every stage fit.
// - Before the VAE stage: at the transformer end of the first generate (right after the DiT is put
//   in residency), stack 256 MiB dummies until OOM after the 1 GiB ones — the remainder is under
//   256 MiB, smaller than what any stage needs (the 512 MiB margin included), so it always evicts
//   regardless of the size of the VAE estimate. Evicting frees the DiT's weights, and the VAE fits.

/** The size of one dummy (on a device whose `maxBufferSize` is smaller, that size is used instead). */
const HEADROOM_DUMMY_BYTES = 1024 * 1024 * 1024;
/** The size of the dummies that fill the remaining free memory (being smaller than {@link HEADROOM_MARGIN_BYTES} is what makes it deterministic). */
const HEADROOM_TIGHT_DUMMY_BYTES = 256 * 1024 * 1024;
/** The cap on the dummy count of one stacking (so that an implementation that never returns OOM does not loop forever). */
const HEADROOM_DUMMY_LIMIT = 32;

/** Stacks dummies of `size` into `dummies` until out-of-memory (returns how many were stacked). */
const fillUntilOom = async (
  gpu: GpuContext,
  size: number,
  dummies: GPUBuffer[],
): Promise<number> => {
  for (let count = 0; ; count += 1) {
    if (count >= HEADROOM_DUMMY_LIMIT) {
      throw new Error(`no out-of-memory after stacking ${count} dummies`);
    }
    gpu.device.pushErrorScope("out-of-memory");
    const buffer = gpu.device.createBuffer({
      label: `e2e-headroom-dummy-${dummies.length}`,
      size,
      usage: GPUBufferUsage.STORAGE,
    });
    const failure = await gpu.device.popErrorScope();
    if (failure !== null) {
      buffer.destroy();
      return count;
    }
    dummies.push(buffer);
  }
};

/**
 * Shrinks the free memory (`tight` = false: stack 1 GiB until OOM, then return one / true: also
 * stack 256 MiB until OOM). Returns after waiting for the release. The return value is the dummies
 * left stacked (the caller destroys them).
 */
const squeezeHeadroom = async (gpu: GpuContext, tight: boolean): Promise<GPUBuffer[]> => {
  const size = Math.min(HEADROOM_DUMMY_BYTES, gpu.limits.maxBufferSize);
  const dummies: GPUBuffer[] = [];
  try {
    const large = await fillUntilOom(gpu, size, dummies);
    if (large === 0) throw new Error("precondition: not a single dummy could be stacked");
    const small = tight ? await fillUntilOom(gpu, HEADROOM_TIGHT_DUMMY_BYTES, dummies) : 0;
    if (!tight) dummies.pop()?.destroy();
    await gpu.device.queue.onSubmittedWorkDone();
    const each = `${(size / 2 ** 30).toFixed(2)} GiB`;
    console.log(
      tight
        ? `[e2e] headroom: ${each} × ${large} + 256 MiB × ${small} filled the free memory down to under 256 MiB`
        : `[e2e] headroom: kept ${each} × ${large - 1}, leaving room for one`,
    );
    return dummies;
  } catch (error) {
    for (const dummy of dummies) dummy.destroy();
    throw error;
  }
};

test.skipIf(!RESIDENCY_RUNNABLE)(
  "e2e(real GPU): residency headroom — without enough free memory above the resident DiT, it evicts with evicted/headroom" +
    " before the text stage, and the second run also matches the twin",
  async () => {
    const { quant } = REFERENCE[1];
    const gpu = await acquireGpu();
    let dummies: GPUBuffer[] = [];
    try {
      await withTurbo(quant, { gpu, residency: "transformer" }, async (pipeline) => {
        assertEquals(
          await assertResidencyTwin("residency-headroom-first", pipeline, 512, undefined),
          expectedStages(["retained/request"]),
          "event sequence of the first run",
        );
        dummies = await squeezeHeadroom(gpu, false);
        // The second run evicts before the text stage (a downgrade), and the DiT stage rebuilds and
        // releases it under per-stage execution. The PNG matches the same twin row as the first run
        // (= bit-identical to the first run).
        assertEquals(
          await assertResidencyTwin("residency-headroom-second", pipeline, 512, undefined),
          expectedStages(["released/downgraded"], { beforeText: ["evicted/headroom"] }),
          "event sequence of the second run",
        );
      });
    } finally {
      for (const dummy of dummies) dummy.destroy();
      gpu.destroy();
    }
  },
);

test.skipIf(!RESIDENCY_RUNNABLE)(
  "e2e(real GPU): residency headroom — even on the first generate, without enough free memory after the DiT is put in residency," +
    " it evicts with evicted/headroom before the VAE stage and matches the twin",
  async () => {
    // The check side also pins what makes it deterministic (free memory after filling < 256 MiB <
    // margin ≤ what the VAE needs).
    assertEquals(HEADROOM_TIGHT_DUMMY_BYTES < HEADROOM_MARGIN_BYTES, true);
    const { quant } = REFERENCE[1];
    const gpu = await acquireGpu();
    let dummies: GPUBuffer[] = [];
    try {
      await withTurbo(quant, { gpu, residency: "transformer" }, async (pipeline) => {
        assertEquals(
          await assertResidencyTwin(
            "residency-headroom-vae",
            pipeline,
            512,
            undefined,
            async (entry) => {
              // The transformer end is after the DiT is put in residency, right before the
              // measurement before the VAE stage (pipeline.ts ③').
              if (entry === "stage:transformer:end") dummies = await squeezeHeadroom(gpu, true);
            },
          ),
          expectedStages(["retained/request"], { beforeVae: ["evicted/headroom"] }),
          "event sequence",
        );
      });
    } finally {
      for (const dummy of dummies) dummy.destroy();
      gpu.destroy();
    }
  },
);

/** Every case ID this check holds (the ones without a reference value are reported once at registration). */
const CASE_IDS: readonly string[] = [
  ...REFERENCE.map(({ quant, resolution }) => caseIdOf(quant, resolution)),
  TURBO_DPMPP_CASE,
  "base-cfg",
  "base-cfg-bf16",
  "base-cfg-fromAssets-shards",
  "base-cfg-dpmpp",
  "fromPretrained-512",
  "onEvent-1024",
  // On a machine without the extra mirror the whole section SKIPs — not counted as missing
  // reference values.
  ...(EXTRA_RUNNABLE ? EXTRA_MODELS.map(extraCaseIdOf) : []),
];
if (RUNNABLE) references.warnMissing(CASE_IDS);

// The gate that keeps "no reference value in this environment yet" from going silently green (the
// same approach as ADR 0005). Only the active cases above are counted (rows of retired cases that
// remain do not make it green).
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
