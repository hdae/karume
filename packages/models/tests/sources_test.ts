// Checks of the tables of public distribution repositories (each family's `SOURCES` — ADR 0092; the
// pinned values are ADR 0073). **Never touches the network** — a recording stub is passed to the
// public surface's `fetch` injection option, and the request fails after only the URL the fetch
// layer built is observed.
//
// Three things are pinned here:
//  ① The shape of an entry — the owner is `hdae`, **the key can be derived from the repo name**
//     (`"karume-" + key` === the repo's basename), and the revision is a 40-digit hex commit SHA.
//     Without a SHA, hub goes out to the revision-resolution API (= offline start-up breaks, and it
//     falls back to a ref that can be moved). The key rule is checked because the table maps "key =
//     repository"; with free spelling, the keys of `KARUME_SOURCES` would stop pointing at
//     repositories.
//  ② `KARUME_SOURCES` is the **union** of the family tables. The merge is a spread, so overlapping
//   keys
//     let the last one win and one entry silently disappears (nothing happens, neither in the types
//     nor at run time).
//  ③ Passing an entry to `fromPretrained` puts its repo / revision into the fetch URL as is. If an
//   entry
//     slid to another repository reference, neither the types nor the shapes would change, so
//     without this it would go unnoticed.
//
// Only ① checks that the spelling is correct (③ derives the expected URL from the entry, so it
// passes even with wrong values). Together the two pin both "the right repo" and "it is actually
// used".
//
// A family without a public repository (vowel-detector) has no table at all and is outside this
// check (ADR 0073 decision 1).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertMatch, assertRejects } from "@std/assert";
import type { HubRepoRef } from "@karume/hub";
import * as anima from "../anima.ts";
import * as irodori from "../irodori.ts";
import * as sbv2 from "../sbv2.ts";
import * as gemma from "../gemma.ts";
import * as gemma4QatMobile from "../gemma4-qat-mobile.ts";
import * as siglip2 from "../siglip2.ts";
import * as depthAnything from "../depth-anything.ts";
import * as birefnet from "../birefnet.ts";
import * as umt5 from "../umt5.ts";
import * as wan from "../wan.ts";
import * as moss from "../moss.ts";
import { KARUME_SOURCES } from "../src/sources.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/**
 * The form that spares hub a resolution request (the same spelling as `isCommitSha` in
 * `@hdae/fetch-cache`).
 */
const COMMIT_SHA = /^[0-9a-f]{40}$/;

/** A fetch that records the URL and **fails on the spot** (not one byte reaches the network). */
const recordingFetch =
  (urls: string[]): typeof globalThis.fetch =>
  (input) => {
    urls.push(input instanceof Request ? input.url : String(input));
    return Promise.reject(new Error("sources_test: does not go any further"));
  };

type LoadOptions = { fetch: typeof globalThis.fetch; caches: CacheStorage };

/**
 * Returns the pathnames of the URLs `fromPretrained` fetches first.
 *
 * The host is left out (the default `hubUrl` moves with the mirror setting — what matters is the
 * repo and the revision).
 */
const requestedPaths = async (
  load: (options: LoadOptions) => Promise<unknown>,
): Promise<string[]> => {
  const urls: string[] = [];
  // MUST: inject `caches` too — touching the real Cache Storage could turn this into the hit path
  // through leftovers of other tests, checking a different story ("does not hit the network").
  await assertRejects(() =>
    load({ fetch: recordingFetch(urls), caches: new MemoryCacheStorage() }),
  );
  return urls.map((url) => new URL(url).pathname);
};

const manifestPath = (source: { repo: string; revision: string }): string =>
  `/${source.repo}/resolve/${source.revision}/karume.json`;

/**
 * What one family table consists of (the table itself + the `fromPretrained` that actually reads
 * its entries).
 */
type Family = {
  readonly name: string;
  readonly sources: Record<string, HubRepoRef>;
  /**
   * The expected key set (spelled out as **an external expectation** — see the `keys` check below).
   */
  readonly keys: readonly string[];
  readonly load: (ref: HubRepoRef, options: LoadOptions) => Promise<unknown>;
};

const FAMILIES: readonly Family[] = [
  {
    name: "anima.SOURCES",
    sources: anima.SOURCES,
    // The repository of the official models and the repository of third-party fine-tunes (ADR
    // 0087). extra takes its text stack from the official one by cross-repository reference, but
    // **the manifest itself comes from its own repository** (the cross-reference lives inside the
    // manifest).
    keys: ["anima", "anima-extra"],
    load: (ref, options) => anima.Pipeline.fromPretrained(ref, options),
  },
  {
    name: "irodori.SOURCES",
    sources: irodori.SOURCES,
    // A new generation or version gets its own repository (ADR 0092 decision 1) — v4 and v4.1 side
    // by side.
    keys: ["irodori-v4-small", "irodori-v4.1-small"],
    load: (ref, options) => irodori.Pipeline.fromPretrained(ref, options),
  },
  {
    name: "sbv2.SOURCES",
    sources: sbv2.SOURCES,
    keys: ["sbv2-jvnv"],
    load: (ref, options) => sbv2.Pipeline.fromPretrained(ref, options),
  },
  {
    name: "gemma.SOURCES",
    sources: gemma.SOURCES,
    // One entry per repository (ADR 0092 decision 3): the regular line and the q4_0 QAT line
    // (first published in 0.16.0), both read by `gemma.Pipeline`.
    keys: ["gemma4", "gemma4-qat"],
    load: (ref, options) => gemma.Pipeline.fromPretrained(ref, options),
  },
  {
    name: "gemma4QatMobile.SOURCES",
    sources: gemma4QatMobile.SOURCES,
    // The mobile QAT line, renamed from `karume-gemma4-qat` (0.15.1) and republished under its new
    // name in 0.16.0; E2B and E4B share one self-contained repository (default E2B).
    keys: ["gemma4-qat-mobile"],
    load: (ref, options) => gemma4QatMobile.Pipeline.fromPretrained(ref, options),
  },
  {
    name: "siglip2.SOURCES",
    sources: siglip2.SOURCES,
    // base and so400m are the same path differing only in size, so they share one repository
    // (default = base).
    keys: ["siglip2"],
    load: (ref, options) => siglip2.Pipeline.fromPretrained(ref, options),
  },
  {
    name: "depthAnything.SOURCES",
    sources: depthAnything.SOURCES,
    // The generation goes into the repository name (ADR 0092 decision 1) — V3 is a different
    // architecture, so it would sit alongside as a separate key.
    keys: ["depth-anything-v2"],
    load: (ref, options) => depthAnything.Pipeline.fromPretrained(ref, options),
  },
  {
    name: "birefnet.SOURCES",
    sources: birefnet.SOURCES,
    // One repository per checkpoint (the derivative Lucida has its own — ADR 0092 decision 1). Each
    // repository holds the per-resolution graphs as two models, "1024" / "2048" (decision 9).
    keys: ["birefnet-hr", "lucida"],
    load: (ref, options) => birefnet.Pipeline.fromPretrained(ref, options),
  },
  {
    name: "umt5.SOURCES",
    sources: umt5.SOURCES,
    // Also the target of Wan's cross-repository text_encoder reference, but one family with one
    // repository and its own entry point (`umt5.Encoder`).
    keys: ["umt5-xxl"],
    load: (ref, options) => umt5.Encoder.fromPretrained(ref, options),
  },
  {
    name: "wan.SOURCES",
    sources: wan.SOURCES,
    // A generation gets its own repository (ADR 0092 decision 1). Wan2.1 (`karume-wan2.1`) is not
    // published on HF, so it has no key (ADR 0118 addendum (2026-10-15)) — the table holds only
    // Wan2.2, which `wan.Ti2vPipeline` reads.
    keys: ["wan2.2"],
    load: (ref, options) => wan.Ti2vPipeline.fromPretrained(ref, options),
  },
  {
    name: "moss.SOURCES",
    sources: moss.SOURCES,
    // First published in 0.15.1; one self-contained repository (no cross-repository reference).
    keys: ["moss-image"],
    load: (ref, options) => moss.Pipeline.fromPretrained(ref, options),
  },
];

// Keys are derived mechanically from the repo name (`"karume-" + key` === basename), so check ①
// only sees that the key and the repo agree **with each other** — a rename that rewrites both at
// once (of the kind `karume-gemma4-e2b` → `karume-gemma4`) passes through. Pinning the set here as
// an external expectation pins the repo spellings indirectly too (a rename goes through only by
// failing this test = the point where the release procedure must be followed).
for (const { name, sources, keys } of FAMILIES) {
  test(`${name}: the key set is as expected`, () => {
    assertEquals(Object.keys(sources).toSorted(), [...keys].toSorted());
  });
}

// The totals of families and public repositories are spelled out too. A new repository (birefnet-hr
// / lucida in iteration b, umT5 / Wan2.2 in 0.14.0) fails this, so an entry missing from the merge
// into `KARUME_SOURCES` cannot pass silently.
test("11 families have a source table, with 15 published repositories", () => {
  assertEquals(
    FAMILIES.map(({ name }) => name),
    [
      "anima.SOURCES",
      "irodori.SOURCES",
      "sbv2.SOURCES",
      "gemma.SOURCES",
      "gemma4QatMobile.SOURCES",
      "siglip2.SOURCES",
      "depthAnything.SOURCES",
      "birefnet.SOURCES",
      "umt5.SOURCES",
      "wan.SOURCES",
      "moss.SOURCES",
    ],
  );
  assertEquals(Object.keys(KARUME_SOURCES).length, 15);
});

for (const { name, sources } of FAMILIES) {
  for (const [key, source] of Object.entries(sources)) {
    test(`${name}["${key}"]: the key derives from the repo + a 40-digit hex commit SHA`, () => {
      const segments = source.repo.split("/");
      assertEquals(segments.length, 2, `repo is not 'owner/name': ${source.repo}`);
      assertEquals(segments[0], "hdae");
      assertEquals(`karume-${key}`, segments[1]);
      assertMatch(source.revision ?? "", COMMIT_SHA);
    });
  }
}

// The merge is a spread, so overlapping keys let the last one win and one entry silently disappears
// (it cannot happen unless two families hold the same repository, but when it does there is no
// other way to notice).
test("KARUME_SOURCES: the union of the family tables, with no duplicate keys", () => {
  // With the literal types, comparing entries is a type error ("the value of another key"), so the
  // table is widened to a lookup shape (key = any spelling) before comparing the contents.
  const merged: Record<string, HubRepoRef> = KARUME_SOURCES;
  const entries = FAMILIES.flatMap(({ sources }) => Object.entries(sources));
  assertEquals(
    Object.keys(merged).length,
    entries.length,
    "an entry disappeared in the merge because keys overlapped",
  );
  // Positive control — do not let iterating over empty tables come out green.
  assert(entries.length >= 8, `the family tables have shrunk (${entries.length} entries)`);
  for (const [key, source] of entries) assertEquals(merged[key], source);
});

// Pinned to a SHA, so not one round trip to the revision-resolution API (= offline start-up works).
// If the fetch URL starts with `/api/models/.../revision/...`, the pin has gone back to a movable
// ref.
for (const { name, sources, load } of FAMILIES) {
  for (const [key, source] of Object.entries(sources)) {
    test(`${name}["${key}"]: the pin goes into the fetch URL as is`, async () => {
      const paths = await requestedPaths((options) => load(source, options));
      assertEquals(paths, [manifestPath({ repo: source.repo, revision: source.revision ?? "" })]);
    });
  }
}

// The meaning of a string ref (`{ repo }` = tracks main) did not change with the pins or with ref
// becoming required. main is not a commit SHA, so there is one round trip to the
// revision-resolution API **before** the manifest is fetched.
test("sbv2: a string ref still tracks main (through the revision-resolution API)", async () => {
  const paths = await requestedPaths((options) =>
    sbv2.Pipeline.fromPretrained("someone/karume-sbv2-fork", options),
  );
  assertEquals(paths, ["/api/models/someone/karume-sbv2-fork/revision/main"]);
});
