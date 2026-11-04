// Checks of the source argument of `fromPretrained` (`src/hub/repo-ref.ts`). **Never touches the
// network.**
//
// Since 0.5.0, `ref` is required for every family (the default source was removed). On the TS side
// the argument is required, so the type check covers it, but a call without arguments from untyped
// JS can only be stopped at run time. If it is not rejected here, `undefined` slides down to hub
// and turns into a failure **far from its cause**: "fetched a URL whose repo is undefined". Four
// things are pinned here:
//  ① A call without a source fails with "repo is required".
//  ② That message carries **a correct example** (two choices: the `{ repo, revision }` spelling,
//   and the
//     spelling that looks up the family's source table) — a diagnostic the reader cannot act on
//     directly says only "it failed" and is of no use.
//  ③ A family without a public distribution repository is not pointed at an identifier that does
//   not exist.
//  ④ **The spelling is an HF `owner/name`** (a string that can only be a path is not spelled into a
//   URL).
//     The spelling that cannot be rescued (a legal `owner/name` such as `models/karume-gemma4`) is
//     pinned alongside.
//  ⑤ **Source handles** (`localDirectory` / `nodeDirectory`) pass through without this check, and
//   adding
//     that pass-through did not loosen the check on the HF side.
// The normal cases (a string = tracks main / an object passes as is) go through the same function,
// so they are pinned too.

import { test } from "vite-plus/test";
import {
  assertEquals,
  assertMatch,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { localDirectory } from "@karume/hub";
import { toManifestSource, toRepoRef } from "../src/hub/repo-ref.ts";
import * as anima from "../anima.ts";
import * as birefnet from "../birefnet.ts";
import * as gemma4QatMobile from "../gemma4-qat-mobile.ts";
import * as umt5 from "../umt5.ts";
import * as wan from "../wan.ts";
import * as moss from "../moss.ts";

// Names what a fetch stub was asked for (a `Request` would otherwise print as "[object Request]").
const describeRequest = (input: RequestInfo | URL): string =>
  typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

test("toRepoRef: without a source it fails with 'repo is required' + two example spellings", () => {
  // The shape of a call without arguments from JS (in TS the type check rejects it first).
  const error = assertThrows(
    () => toRepoRef(undefined, "anima.Pipeline.fromPretrained", 'SOURCES["anima"]'),
    Error,
    "repo is required",
  );
  // Both the subject (which entry point failed) and the two example spellings are present.
  assertStringIncludes(error.message, "anima.Pipeline.fromPretrained");
  assertStringIncludes(error.message, '{ repo: "owner/name", revision: "<40-digit commit SHA>" }');
  // Anchored: a prefixed table name (`ANIMA_SOURCES[…]`, the 0.14 spelling) must not match.
  assertMatch(error.message, /(?<![\w.])SOURCES\["anima"\]/);
});

test("toRepoRef: a family without a public distribution repository is not pointed at a source table", () => {
  // Pointing at an identifier that does not exist sends the reader looking for something they
  // cannot import.
  const error = assertThrows(
    () => toRepoRef(undefined, "vowelDetector.Pipeline.fromPretrained"),
    Error,
    "repo is required",
  );
  assertStringIncludes(error.message, '{ repo: "owner/name", revision: "<40-digit commit SHA>" }');
  assertStringIncludes(error.message, "repo name string");
  assertEquals(error.message.includes("SOURCES"), false);
});

test("toRepoRef: an empty repo fails the same check (no URL is built from an empty string)", () => {
  assertThrows(() => toRepoRef("", "sbv2.Pipeline.fromPretrained"), Error, "repo is required");
  assertThrows(
    () => toRepoRef({ repo: "" }, "sbv2.Pipeline.fromPretrained"),
    Error,
    "repo is required",
  );
});

test("toRepoRef: a spelling that can only be a path fails before anything is sent to HF", () => {
  // Without this check, a call that passes a local directory has that string spelled into a URL,
  // and what comes back is 401 / 404 — a diagnostic far from its cause: "the source does not
  // exist".
  const paths = [
    "./models/karume-gemma4", // leading './'
    "../karume-gemma4", // leading '../'
    "/home/me/models/dist", // absolute path
    "hdae/karume-anima/", // trailing '/'
    "a/b/c", // two or more slashes
    "hdae//karume-anima", // empty segment
    "hdae/karume anima", // disallowed character
    ".hidden/name", // leading dot
    "karume-gemma4", // a single segment (not `owner/name`)
  ];
  for (const repo of paths) {
    assertThrows(
      () => toRepoRef(repo, "gemma.Pipeline.fromPretrained"),
      Error,
      "is not an HF 'owner/name'",
      repo,
    );
  }
});

test("toRepoRef: a local path that looks like `owner/name` cannot be rescued (by design)", () => {
  // It is a fully legal HF repo name, so the argument carries nothing to tell them apart.
  // Whether it exists is the fetch layer's job; this function looks only at the spelling.
  assertEquals(toRepoRef("models/karume-gemma4", "gemma.Pipeline.fromPretrained"), {
    repo: "models/karume-gemma4",
  });
});

test("toRepoRef: a string becomes `{ repo }` (= tracks main) and an object passes as is", () => {
  // Making ref required did not change what a string ref means (no revision = tracks main).
  assertEquals(toRepoRef("someone/karume-fork", "X.fromPretrained"), {
    repo: "someone/karume-fork",
  });
  // revision / hubUrl are hub's vocabulary, so nothing is removed or added here.
  const pinned = { repo: "someone/karume-fork", revision: "a".repeat(40), hubUrl: "https://m" };
  assertEquals(toRepoRef(pinned, "X.fromPretrained"), pinned);
});

test("toManifestSource: a source handle is passed through without the spelling check", () => {
  // A local distribution is not an HF repo, so it must not go through the check that requires
  // `owner/name` (it would fail with "repo is required" and not one byte could be read from the
  // local directory). Nothing is ever read, so an adapter that fails when read is enough.
  const source = localDirectory(
    {
      readFile: () =>
        Promise.reject(
          new Error("repo_ref_test: read the contents when only telling the kind apart"),
        ),
    },
    { label: "./models/karume-test" },
  );
  assertStrictEquals(toManifestSource(source, "gemma.Pipeline.fromPretrained"), source);
});

test("toManifestSource: a value that is not a source handle goes through the spelling check as before", () => {
  // Widening the union does not loosen the check on the HF side (a source handle is recognised by
  // identity, so "an object with a repo" cannot pass through as a source).
  assertEquals(toManifestSource("someone/karume-fork", "X.fromPretrained"), {
    repo: "someone/karume-fork",
  });
  assertThrows(
    () => toManifestSource("./models/karume-gemma4", "X.fromPretrained"),
    Error,
    "is not an HF 'owner/name'",
  );
  assertThrows(() => toManifestSource(undefined, "X.fromPretrained"), Error, "repo is required");
});

test("fromPretrained: an empty source fails without calling fetch even once", async () => {
  // The counterpart seen from the public surface. A stub that fails when called is put in the fetch
  // option to see that the check sits **before** the fetch layer (if it were called, the message
  // would turn into a network-side failure).
  let calls = 0;
  const fetchStub: typeof globalThis.fetch = (input) => {
    calls += 1;
    return Promise.reject(
      new Error(`repo_ref_test: reached the fetch layer (${describeRequest(input)})`),
    );
  };
  await assertRejects(
    () => anima.Pipeline.fromPretrained({ repo: "" }, { fetch: fetchStub }),
    Error,
    "anima.Pipeline.fromPretrained: repo is required",
  );
  assertEquals(calls, 0);
});

test("fromPretrained: birefnet also points at its source table (birefnet.SOURCES) when ref is missing", async () => {
  // 0.10.0 added the public pins (then `BIREFNET_SOURCES`, now `birefnet.SOURCES`), but the hint
  // stayed as it was when there was no public repository (a downstream report). This checks from
  // the public surface that the message matches the other families.
  const fetchStub: typeof globalThis.fetch = (input) =>
    Promise.reject(new Error(`repo_ref_test: reached the fetch layer (${describeRequest(input)})`));
  const error = await assertRejects(
    () => birefnet.Pipeline.fromPretrained({ repo: "" }, { fetch: fetchStub }),
    Error,
    "birefnet.Pipeline.fromPretrained: repo is required",
  );
  assertMatch(error.message, /(?<![\w.])SOURCES\["birefnet-hr"\] \(@karume\/models\/birefnet\)/);
});

test("fromPretrained: umT5 and Wan2.2 point at their source tables when ref is missing; the unpublished Wan2.1 does not", async () => {
  // 0.14.0 added the source tables (then `UMT5_SOURCES` / `WAN_SOURCES`) with the families' first
  // publication. Wan2.1 and 2.2 share the load body (`loadWanFromPretrained`), but the table has no
  // Wan2.1 key — this checks, through the public surface that goes through the shared body, that
  // 2.1 alone is not pointed at a key that does not exist (③).
  const fetchStub: typeof globalThis.fetch = (input) =>
    Promise.reject(new Error(`repo_ref_test: reached the fetch layer (${describeRequest(input)})`));
  const umt5Error = await assertRejects(
    () => umt5.Encoder.fromPretrained({ repo: "" }, { fetch: fetchStub }),
    Error,
    "umt5.Encoder.fromPretrained: repo is required",
  );
  assertMatch(umt5Error.message, /(?<![\w.])SOURCES\["umt5-xxl"\] \(@karume\/models\/umt5\)/);
  const wan22 = await assertRejects(
    () => wan.Ti2vPipeline.fromPretrained({ repo: "" }, { fetch: fetchStub }),
    Error,
    "wan.Ti2vPipeline.fromPretrained: repo is required",
  );
  assertMatch(wan22.message, /(?<![\w.])SOURCES\["wan2\.2"\] \(@karume\/models\/wan\)/);
  const wan21 = await assertRejects(
    () => wan.Pipeline.fromPretrained({ repo: "" }, { fetch: fetchStub }),
    Error,
    "wan.Pipeline.fromPretrained: repo is required",
  );
  assertStringIncludes(wan21.message, "pass a repo name string (= tracks main)");
  assertEquals(wan21.message.includes("SOURCES"), false, wan21.message);
});

test("fromPretrained: MOSS-Image points at its source table when ref is missing", async () => {
  // The table came with the family's first publication (0.15.1, like umT5 / Wan2.2 in 0.14.0): the
  // hint names the entry this package version verified, seen from the public surface.
  const fetchStub: typeof globalThis.fetch = (input) =>
    Promise.reject(new Error(`repo_ref_test: reached the fetch layer (${describeRequest(input)})`));
  const error = await assertRejects(
    () => moss.Pipeline.fromPretrained({ repo: "" }, { fetch: fetchStub }),
    Error,
    "moss.Pipeline.fromPretrained: repo is required",
  );
  assertMatch(error.message, /(?<![\w.])SOURCES\["moss-image"\] \(@karume\/models\/moss\)/);
});

test("fromPretrained: Gemma 4 mobile QAT points at its source table when ref is missing", async () => {
  // The table came back with the renamed distribution's publication (0.16.0). It shares the load
  // body with regular Gemma (`GemmaPipeline.loadPretrained`), so each family must name its own
  // table.
  const fetchStub: typeof globalThis.fetch = (input) =>
    Promise.reject(new Error(`repo_ref_test: reached the fetch layer (${describeRequest(input)})`));
  const error = await assertRejects(
    () => gemma4QatMobile.Pipeline.fromPretrained({ repo: "" }, { fetch: fetchStub }),
    Error,
    "gemma4QatMobile.Pipeline.fromPretrained: repo is required",
  );
  assertMatch(
    error.message,
    /(?<![\w.])SOURCES\["gemma4-qat-mobile"\] \(@karume\/models\/gemma4-qat-mobile\)/,
  );
});
