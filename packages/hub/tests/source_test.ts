/**
 * Unit tests of the branch point of the source contract (`sourceForRef`).
 *
 * The routing of cross-repository references lives in one place, because writing it per surface
 * invites mix-ups where only one surface lets a reference through. The public surface can observe
 * it only as "the bytes when a cross-repository reference is mixed in", so the branch itself is
 * pinned here (a resolution order of "explicit mapping → fail loudly" is to be added here later).
 */

import { assertEquals, assertStrictEquals } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { type FileRef, fileRefKey } from "../src/manifest.ts";
import { type PinnedSource, sourceForRef } from "../src/source.ts";

const SHA = "b".repeat(40);

/** A source that only records calls to `originFor` (its read methods are never called). */
const recordingSource = (): { source: PinnedSource; calls: [string, string][] } => {
  const calls: [string, string][] = [];
  const source: PinnedSource = {
    origin: { label: "repo someone/anima @ " + SHA, integrity: "network" },
    readManifest: () => Promise.reject(new Error("must not be called")),
    readFile: () => Promise.reject(new Error("must not be called")),
    prefetchFile: () => Promise.reject(new Error("must not be called")),
    originFor: (repo, revision) => {
      calls.push([repo, revision]);
      // The referenced repository is a different source (a different object, so identity is
      // visible).
      return { ...source, originFor: source.originFor };
    },
  };
  return { source, calls };
};

const ref = (extra: Partial<FileRef> = {}): FileRef => ({
  path: "transformer/model.safetensors",
  size: 4,
  sha256: "a".repeat(64),
  ...extra,
});

test("sourceForRef: a same-repository reference uses the session's source as is", () => {
  const { source, calls } = recordingSource();
  assertStrictEquals(sourceForRef(source, ref()), source);
  assertEquals(calls, [], "originFor was called for a reference that is not cross-repository");
});

test("sourceForRef: only a reference with both repo and revision goes to the declared source", () => {
  const { source, calls } = recordingSource();
  const crossed = sourceForRef(source, ref({ repo: "someone/other", revision: SHA }));
  assertEquals(calls, [["someone/other", SHA]]);
  assertStrictEquals(
    crossed === source,
    false,
    "the referenced source is the session's own source",
  );
});

describe("sourceForRef: declaring only one of the two does not cross repositories (they only mean something as a pair)", () => {
  // Parsing enforces the pair, so this does not appear through a manifest, but a branch that looks
  // at only one of them silently allows mix-ups such as "applying the other repository's revision
  // to this repository".
  it("repo only", () => {
    const { source, calls } = recordingSource();
    assertStrictEquals(sourceForRef(source, ref({ repo: "someone/other" })), source);
    assertEquals(calls, []);
  });
  it("revision only", () => {
    const { source, calls } = recordingSource();
    assertStrictEquals(sourceForRef(source, ref({ revision: SHA })), source);
    assertEquals(calls, []);
  });
});

// The dedupe key must detect cross-repository references with **the same predicate** (`crossRefOf`)
// as the source branch. If only the key side counted a one-sided declaration as cross-repository,
// one file would become two separate entries (the whole-file surface fetches the same URL twice and
// progress totals count it twice), and since the bytes are correct it would pass silently.

test("fileRefKey: only a cross-repository reference with the full pair becomes a key with the declared coordinates", () => {
  assertEquals(
    fileRefKey(ref({ repo: "someone/other", revision: SHA })),
    `someone/other@${SHA}/transformer/model.safetensors`,
  );
});

describe("fileRefKey: a one-sided declaration folds into a same-repository path key (the same rule as sourceForRef)", () => {
  it("repo only", () => {
    assertEquals(fileRefKey(ref({ repo: "someone/other" })), ref().path);
  });
  it("revision only", () => {
    assertEquals(fileRefKey(ref({ revision: SHA })), ref().path);
  });
});
