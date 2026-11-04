/**
 * Real-fs tests of `@karume/hub/node`. What the in-memory adapter (`local_test.ts`) cannot exercise
 * is "the runtime's read path itself" — how paths are joined (with or without a trailing `/`, a
 * `file:` URL), and whether the errors `node:fs` throws reach the shared layer **still naming the
 * actual path**.
 *
 * The semantics as a source (generations, verification, cross-repository references, progress)
 * belong to the main tests (`local_test.ts`), so this file does not duplicate them: it checks that
 * one end-to-end run passes, plus the properties confined to the runtime's read path itself.
 *
 * Of the latter, "whether the abort signal reaches the `node:fs` read" cannot be exercised from the
 * public surface (`fetchAssets`) (the abort is caught before the dispatch loop), so the source
 * driver is called directly through the internal `driverOf` — the same approach as
 * `inventory_test.ts` uses when it looks at a source's internal behaviour.
 */

import { makeTempDir, mkdir, readFile, remove, writeFile } from "@karume/test-support/fs";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { fetchAssets, HubFetchError, IntegrityError, loadManifest, openAsset } from "../mod.ts";
import { nodeDirectory } from "../node.ts";
import { driverOf } from "../src/source.ts";
import { buildLocalDist, FETCHED_PART_PATHS, PART_KEYS, TOKENIZER_PATH } from "./helpers/local.ts";
import { payloadFor } from "./helpers/mock.ts";
import { selectionFiles } from "./helpers/selection.ts";

/** Writes a synthesized distribution form to a real directory (the caller removes it). */
const materialize = async (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
): Promise<string> => {
  const root = await makeTempDir({ prefix: "karume-hub-local-" });
  for (const [path, bytes] of files) {
    const at = `${root}/${path}`;
    await mkdir(at.slice(0, at.lastIndexOf("/")), { recursive: true });
    await writeFile(at, bytes);
  }
  return root;
};

const withDistribution = async (
  body: (root: string, files: Map<string, Uint8Array<ArrayBuffer>>) => Promise<void>,
  mutate: (files: Map<string, Uint8Array<ArrayBuffer>>) => void = () => {},
): Promise<void> => {
  const dist = await buildLocalDist();
  mutate(dist.files);
  const root = await materialize(dist.files);
  try {
    await body(root, dist.files);
  } finally {
    await remove(root, { recursive: true });
  }
};

describe("nodeDirectory: a distribution form in a real directory reads with or without a trailing /", () => {
  // One materialized distribution shared by both spellings (the root path is only known once
  // beforeAll has run, so each it derives its spelling from the trailing-slash flag).
  let root: string | undefined;
  beforeAll(async () => {
    root = await materialize((await buildLocalDist()).files);
  });
  afterAll(async () => {
    if (root !== undefined) await remove(root, { recursive: true });
  });

  for (const trailing of [false, true]) {
    it(`root = ${trailing ? "with trailing /" : "without trailing /"}`, async () => {
      assert(root !== undefined, "the distribution form was not written");
      const spelling = trailing ? `${root}/` : root;
      const loaded = await loadManifest(nodeDirectory(spelling));
      const assets = await fetchAssets(loaded, selectionFiles(loaded.manifest));
      PART_KEYS.forEach((key, index) => {
        assertEquals(assets[key], payloadFor(FETCHED_PART_PATHS[index]));
      });
      assertEquals(assets["tokenizer"], payloadFor(TOKENIZER_PATH));
    });
  }
});

test("nodeDirectory: a file: URL root points at the same distribution form", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(nodeDirectory(new URL(`file://${root}`)));
    const assets = await fetchAssets(loaded, selectionFiles(loaded.manifest));
    assertEquals(assets["tokenizer"], payloadFor(TOKENIZER_PATH));
  });
});

test("nodeDirectory: does not read the neighbouring directory (does not leave root)", async () => {
  await withDistribution(async (root) => {
    // An unreadable karume.json is placed exactly where a join that drops the trailing `/` would
    // land (`<root>` + `karume.json`). If joining breaks, this is read and the manifest fails.
    const sibling = `${root}karume.json`;
    await writeFile(sibling, new TextEncoder().encode("{}"));
    try {
      const loaded = await loadManifest(nodeDirectory(root));
      assertEquals(loaded.manifest.available.models, ["m"], "the sibling karume.json was read");
    } finally {
      await remove(sibling);
    }
  });
});

test("nodeDirectory: a missing file fails loudly naming the actual path", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(nodeDirectory(root));
    // The file vanishes **after** the manifest was read (a directory missing part of the
    // distribution form).
    await remove(`${root}/${FETCHED_PART_PATHS[1]}`);

    const error = await assertRejects(
      () => fetchAssets(loaded, selectionFiles(loaded.manifest)),
      HubFetchError,
    );
    assertEquals(error.path, FETCHED_PART_PATHS[1]);
    assert(error.message.includes(root), `${error.message} does not name the directory`);
    assert(error.cause instanceof Error, "the fs error is not kept as cause");
    assert(
      error.cause.message.includes(`${root}/${FETCHED_PART_PATHS[1]}`),
      `${error.cause.message} does not name the absolute path of the file that could not be read`,
    );
  });
});

test("nodeDirectory: a missing karume.json (the whole root absent) fails on the same path", async () => {
  const root = await makeTempDir({ prefix: "karume-hub-empty-" });
  try {
    const error = await assertRejects(() => loadManifest(nodeDirectory(root)), HubFetchError);
    assertEquals(error.path, "karume.json");
    assertEquals(error.repo, undefined, "names a repo that does not exist");
    assert(error.message.includes(root), `${error.message} does not name the directory`);
  } finally {
    await remove(root, { recursive: true });
  }
});

test("nodeDirectory: IntegrityError when the actual size disagrees with the manifest", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(nodeDirectory(root));
    // A truncated copy (only the size is off — sha256 is trusted, so this is the only check).
    const truncated = payloadFor(TOKENIZER_PATH).subarray(0, 3);
    await writeFile(`${root}/${TOKENIZER_PATH}`, truncated);

    const error = await assertRejects(
      () => fetchAssets(loaded, selectionFiles(loaded.manifest)),
      IntegrityError,
    );
    assertEquals(error.path, TOKENIZER_PATH);
    assertEquals(error.source, "local");
    assertEquals(error.actual, "3");
  });
});

// ---- The size check of the whole-file surface (real-fs side). A file **longer** than declared is
// a
// shape the 3 tests above do not exercise: after `fs.readFile` returns the whole file, the source
// (`sources/local.ts`) fails naming the actual length.

test("nodeDirectory: a file longer than declared also fails the size check (naming the actual length)", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(nodeDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    // The shape of a mixed-up quant or a half-written copy (the manifest's size / sha256
    // unchanged).
    await writeFile(`${root}/${TOKENIZER_PATH}`, new Uint8Array(ref.size + 1));

    const error = await assertRejects(
      () => fetchAssets(loaded, selectionFiles(loaded.manifest)),
      IntegrityError,
    );
    assertEquals(error.path, TOKENIZER_PATH);
    assertEquals(error.actual, String(ref.size + 1));
    assertEquals(error.source, "local");
  });
});

test("nodeDirectory: a whole-file read with an already aborted signal returns not a single byte", async () => {
  await withDistribution(async (root) => {
    const source = nodeDirectory(root);
    const loaded = await loadManifest(source);
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const controller = new AbortController();
    const reason = new Error("test: aborted by the caller");
    controller.abort(reason);

    // Calls the source (`sources/local.ts` ③) **directly**. Through the shared layer
    // (`fetchAssets`), the abort is caught when the admission slot is acquired before the dispatch
    // loop and never reaches the real fs read (= whether the signal reaches `fs.readFile` cannot be
    // seen).
    const pinned = driverOf(source).pin("", {});
    const error = await assertRejects(
      () =>
        pinned.readFile(ref, {
          signal: controller.signal,
          // By contract the local source never calls `onProgress` (`FileReadOptions` in
          // `source.ts`), so an abort watch placed here would be trivially true — a no-op that only
          // fills the required field. `sizeViolation` checks that the abort did not turn into a
          // size
          // violation.
          onProgress: () => {},
          sizeViolation: (actual) => new Error(`the size check fired (actual ${actual})`),
        }),
      Error,
    );
    // The abort turns into neither a fetch failure nor a size violation — the reason the caller
    // passed comes up as is.
    assertStrictEquals(error, reason);
  });
});

// ---- Range reads (`openAsset` — `source.ts` ⑧). The `fs.open` → positional read loop exists only
// here on the real-fs side, so the fake adapter (`local_test.ts`) cannot stand in for it.

test("nodeDirectory: a range read matches the same range of the file (start / middle / exactly at the end)", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(nodeDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const whole = await readFile(`${root}/${ref.path}`);
    const reader = await openAsset(loaded, ref);
    assert(
      reader !== undefined,
      "the reader does not open although the adapter has positional reads",
    );
    // A directory's file is read positionally, so the cost does not depend on the offset.
    assertEquals(reader.cost, "seek");

    const middle = Math.floor(ref.size / 2);
    for (const [offset, length] of [
      [0, 4],
      [middle, 3],
      [ref.size - 5, 5],
    ]) {
      const bytes = await reader.read(offset, length);
      assertEquals(
        bytes,
        new Uint8Array(whole.subarray(offset, offset + length)),
        `the content of [${offset}, ${offset + length}) is corrupted`,
      );
      // The consumer reads it as a TypedArray as is, so it must occupy its whole buffer.
      assertEquals(bytes.byteOffset, 0);
      assertEquals(bytes.buffer.byteLength, length);
    }
  });
});

test("nodeDirectory: a zero-length range is an empty byte array (not a caller error)", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(nodeDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const reader = await openAsset(loaded, ref);
    assert(reader !== undefined, "the reader does not open");
    // A 0-byte request means "nothing to read", not an error (allowed exactly at the end as well).
    for (const offset of [0, ref.size]) {
      assertEquals((await reader.read(offset, 0)).byteLength, 0);
    }
  });
});

test("nodeDirectory: a range outside the declared size fails before reaching the adapter", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(nodeDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const reader = await openAsset(loaded, ref);
    assert(reader !== undefined, "the reader does not open");

    for (const [offset, length] of [
      [ref.size - 1, 2],
      [ref.size, 1],
      [-1, 2],
      [0, 1.5],
    ]) {
      const error = await assertRejects(() => reader.read(offset, length), Error);
      assert(error.message.includes(ref.path), `${error.message} does not name the path`);
      assert(
        error.message.includes(String(ref.size)),
        `${error.message} does not name the declared size`,
      );
    }
  });
});

test("nodeDirectory: a file shorter than declared fails on the range it cannot fill (does not return zero fill)", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(nodeDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const reader = await openAsset(loaded, ref);
    assert(reader !== undefined, "the reader does not open");
    // A truncated copy (the manifest's size unchanged = the same shape as the size check of the
    // whole-file read).
    await writeFile(`${root}/${ref.path}`, payloadFor(ref.path).subarray(0, 3));

    const error = await assertRejects(() => reader.read(0, ref.size), Error);
    assert(
      error.message.includes(`${root}/${ref.path}`),
      `${error.message} does not name the file that could not be read`,
    );
  });
});

test("nodeDirectory: a range read with an already aborted signal returns not a single byte", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(nodeDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const reader = await openAsset(loaded, ref);
    assert(reader !== undefined, "the reader does not open");
    const controller = new AbortController();
    const reason = new Error("test: aborted by the caller");
    controller.abort(reason);

    // The same for length 0 (even when the read loop never runs, the abort is checked before
    // touching the file).
    for (const length of [4, 0]) {
      // The abort does not turn into a fetch failure (the reason the caller passed comes up as is).
      assertStrictEquals(
        await assertRejects(() => reader.read(0, length, { signal: controller.signal }), Error),
        reason,
      );
    }
  });
});
