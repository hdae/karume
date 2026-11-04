// The check for the options that the `fromPretrained` of 9 families pass through to the hub
// (`src/hub/load-options.ts`). **Needs neither a GPU nor the real network.**
//
// When the 0.11.0 hub added `onRetry` (the retry notification for 429 / 503), the
// `FromPretrainedOptions` of 8 families (then `XFromPretrainedOptions`) duplicated the same 5
// fields by hand, so the field was missing in all 8 places — "a new knob silently does not arrive".
// The duplicates were folded into one ({@link hubLoadOptions}), and this file pins four things:
//  ① The 6 defined fields are copied **as the same references** (not re-wrapped on the way).
//  ② Undefined fields do not appear **at all, key included** — `key: undefined` is distinct from
//     "not specified" and can affect the fetch layer's branches (an explicit `undefined` `fetch`
//     overriding the default `globalThis.fetch`, and so on).
//  ③ `onProgress` is not copied (the family side puts progress into `loadContainerComponents`
//     separately).
//  ④ The type check — **every one** of the 9 families accepts `onRetry` (`deno task check` goes red
//     if one is missing).
// Finally, as the counterpart seen from the public surface, a mock fetch's 429 checks that "anima's
// `fromPretrained` carries `onRetry` down to `loadManifest`" (one family is enough — the carrying
// path is the same one for all 9 families).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import type { AssetProgress, CacheDiagnostic, RetryDiagnostic } from "@karume/hub";
import { hubLoadOptions } from "../src/hub/load-options.ts";
import * as anima from "../anima.ts";
import * as birefnet from "../birefnet.ts";
import * as depthAnything from "../depth-anything.ts";
import * as gemma from "../gemma.ts";
import * as irodori from "../irodori.ts";
import * as sbv2 from "../sbv2.ts";
import * as siglip2 from "../siglip2.ts";
import * as umt5 from "../umt5.ts";
import * as vowelDetector from "../vowel-detector.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

const noopRetry = (diagnostic: RetryDiagnostic): void => void diagnostic;

test("hubLoadOptions: the 6 defined fields are copied as the same references", () => {
  const signal = AbortSignal.abort();
  const headers = new Headers({ authorization: "Bearer test" });
  const onCacheError = (diagnostic: CacheDiagnostic): void => void diagnostic;
  const fetchStub: typeof globalThis.fetch = () => Promise.reject(new Error("test: not called"));
  const caches = new MemoryCacheStorage();

  const hubOptions = hubLoadOptions({
    signal,
    headers,
    onCacheError,
    onRetry: noopRetry,
    fetch: fetchStub,
    caches,
  });

  // Re-wrapping would make "a function other than the one the app passed get called", and neither
  // unsubscribing nor comparing would work.
  assertStrictEquals(hubOptions.signal, signal);
  assertStrictEquals(hubOptions.headers, headers);
  assertStrictEquals(hubOptions.onCacheError, onCacheError);
  assertStrictEquals(hubOptions.onRetry, noopRetry);
  assertStrictEquals(hubOptions.fetch, fetchStub);
  assertStrictEquals(hubOptions.caches, caches);
  assertEquals(Object.keys(hubOptions).length, 6, "the copied fields are not 6");
});

test("hubLoadOptions: undefined fields are absent, key included (no `key: undefined` is created)", () => {
  // An explicit `undefined` is rescued by defaulting like the fetch layer's
  // `options.fetch ?? globalThis.fetch`, but not by code that branches on `Object.hasOwn` or on a
  // destructuring default. The copy's contract is to deliver "not specified" and "specified as
  // undefined" as the distinct things they are.
  for (const options of [{}, { signal: undefined, headers: undefined, onRetry: undefined }]) {
    const hubOptions: Record<string, unknown> = hubLoadOptions(options);
    assertEquals(Object.keys(hubOptions), [], "undefined fields appear as keys");
    for (const key of ["signal", "headers", "onCacheError", "onRetry", "fetch", "caches"]) {
      assertEquals(Object.hasOwn(hubOptions, key), false, `the key ${key} was created`);
    }
  }
});

test("hubLoadOptions: onProgress is not copied (the family side adds progress separately)", () => {
  // Fetching the manifest has no progress, and asset-fetch progress is aggregated per family before
  // it is passed to `loadContainerComponents`. Copying it here would send the raw, unaggregated
  // progress twice.
  const onProgress = (progress: AssetProgress): void => void progress;
  const hubOptions: Record<string, unknown> = hubLoadOptions({ onProgress, onRetry: noopRetry });
  assertEquals(Object.hasOwn(hubOptions, "onProgress"), false, "onProgress was copied");
  assertEquals(Object.keys(hubOptions), ["onRetry"]);
});

test("type check: the fromPretrained options of 9 families accept onRetry", () => {
  // `satisfies` is the substance (`deno task check` is the check). Goes red if even one family
  // drifts from the shared type.
  const perFamily = [
    { onRetry: noopRetry } satisfies anima.FromPretrainedOptions,
    { onRetry: noopRetry } satisfies birefnet.FromPretrainedOptions,
    { onRetry: noopRetry } satisfies depthAnything.FromPretrainedOptions,
    { onRetry: noopRetry } satisfies gemma.FromPretrainedOptions,
    { onRetry: noopRetry } satisfies irodori.FromPretrainedOptions,
    { onRetry: noopRetry } satisfies sbv2.FromPretrainedOptions,
    { onRetry: noopRetry } satisfies siglip2.FromPretrainedOptions,
    { onRetry: noopRetry } satisfies umt5.FromPretrainedOptions,
    { onRetry: noopRetry } satisfies vowelDetector.FromPretrainedOptions,
  ];
  assertEquals(perFamily.length, 9, "the families listed are not 9");
  // The same list also checks that the copy's entry (`hubLoadOptions`) accepts the 9 families'
  // types **as they are**.
  for (const options of perFamily) assertStrictEquals(hubLoadOptions(options).onRetry, noopRetry);
});

// ---- Pass-through seen from the public surface (anima, one family)
// ----------------------------------------

const HUB_URL = "https://hub.test";
const REPO = "someone/karume-anima";
const SHA = "0123456789abcdef0123456789abcdef01234567";
const MANIFEST_URL = `${HUB_URL}/${REPO}/resolve/${SHA}/karume.json`;

/**
 * The dtype entry of one component (its contents never arrive, so only the spelling of the sha256
 * is made to match). The size of part 0 matches "header 24 + 2 documents" (the manifest's check —
 * container-v1 §8).
 */
const weightEntry = (stem: string) => ({
  f16: {
    container: {
      descriptor: {
        graph: { length: 10, sha256: "b".repeat(64) },
        model: { length: 8, sha256: "c".repeat(64) },
      },
      parts: [
        { path: `${stem}-00001-of-00002.krm`, size: 42, sha256: "a".repeat(64) },
        { path: `${stem}-00002-of-00002.krm`, size: 64, sha256: "d".repeat(64) },
      ],
    },
  },
});

const MANIFEST = {
  format: "karume/5",
  generator: "karume/0.11.0",
  defaultModel: "anima",
  models: {
    anima: {
      pipeline: "anima/1",
      weights: {
        text_encoder: weightEntry("text_encoder/model.f16"),
        text_conditioner: weightEntry("text_conditioner/model.f16"),
        transformer: weightEntry("transformer/model.f16"),
        vae_decoder: weightEntry("vae_decoder/model.f16"),
      },
      quants: {
        f16: {
          weights: {
            text_encoder: "f16",
            text_conditioner: "f16",
            transformer: "f16",
            vae_decoder: "f16",
          },
          session: {},
        },
      },
      defaultQuant: "f16",
      // The 2 fields the parse requires to exist (explicit even when empty — their contents are not
      // read in this test).
      assets: { tokenizer: { path: "tokenizer/tokenizer.json", size: 42, sha256: "b".repeat(64) } },
      pipelineConfig: {},
    },
  },
};

/**
 * A `fetch` that returns 429 (`retry-after: 0`) **only the first time** for `karume.json` and the
 * manifest the second time. Every other URL (= the container's parts) is 404 — only the path up to
 * the manifest fetch matters here, so what follows may fail.
 */
const createRateLimitedManifestFetch = (): { fetch: typeof globalThis.fetch; calls: string[] } => {
  const calls: string[] = [];
  let rateLimited = false;
  const fetch: typeof globalThis.fetch = (input) => {
    const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    calls.push(href);
    if (href !== MANIFEST_URL) {
      return Promise.resolve(new Response(null, { status: 404, statusText: "Not Found" }));
    }
    if (!rateLimited) {
      rateLimited = true;
      return Promise.resolve(
        new Response(null, {
          status: 429,
          statusText: "Too Many Requests",
          headers: { "retry-after": "0" },
        }),
      );
    }
    return Promise.resolve(
      new Response(JSON.stringify(MANIFEST), { headers: { "content-type": "application/json" } }),
    );
  };
  return { fetch, calls };
};

test("anima.Pipeline.fromPretrained: onRetry reaches loadManifest (once, on a 429)", async () => {
  const { fetch, calls } = createRateLimitedManifestFetch();
  const retries: RetryDiagnostic[] = [];
  // After the manifest (the container's part 0) it is a 404, so it fails. All this checks is that
  // "the family's `fromPretrained` carries `onRetry` to the fetch layer".
  await assertRejects(() =>
    anima.Pipeline.fromPretrained(
      { repo: REPO, revision: SHA, hubUrl: HUB_URL },
      {
        fetch,
        caches: new MemoryCacheStorage(),
        onRetry: (diagnostic) => retries.push(diagnostic),
      },
    ),
  );
  assertEquals(retries.length, 1, "the retry notification did not arrive exactly once");
  assertEquals(retries[0].status, 429);
  assertEquals(retries[0].url, MANIFEST_URL, "the notification names a URL other than karume.json");
  assertEquals(retries[0].attempt, 1);
  // It fetched again after the 429 (= not a form that only notifies and gives up).
  assertEquals(calls.filter((url) => url === MANIFEST_URL).length, 2);
  assert(
    calls.some((url) => url.endsWith("/text_encoder/model.f16-00001-of-00002.krm")),
    "the manifest was not read (it did not get as far as fetching the descriptor)",
  );
});
