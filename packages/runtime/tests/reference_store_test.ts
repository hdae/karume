/**
 * The store of what a sha256 reference value was taken from (`helpers/reference-store.ts`), and its
 * use through `References.observe` / `settleReference`: a matching run keeps its artifact and its
 * kernels, a mismatch is told what changed, and the verdict never depends on the store.
 */

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  makeTempDirSync,
  readFileSync,
  readTextFileSync,
  removeSync,
  writeFileSync,
  writeTextFileSync,
} from "@karume/test-support/fs";
import type { KernelUse } from "@karume/test-support/kernel-record";
import type { Environment } from "./helpers/environment.ts";
import {
  kernelChanges,
  mismatchEvidence,
  openReferenceStore,
  type ReferenceStore,
} from "./helpers/reference-store.ts";
import {
  openReferences,
  referenceMismatchMessage,
  settleReference,
  sha256Hex,
} from "./helpers/reference.ts";
import { openResults } from "./helpers/results.ts";

const ENVIRONMENT_KEY = "node-test-adapter";

const environment: Environment = {
  key: ENVIRONMENT_KEY,
  runtime: { name: "node", version: "24.0.0", v8: "13.0", typescript: "7.0" },
  adapter: { vendor: "test", architecture: "", device: "test adapter", description: "test" },
  os: { platform: "linux", arch: "x86_64" },
};

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);

const kernel = (label: string, text: string): KernelUse => ({ label, wgslSha256: text.repeat(64) });

const GEMM_V1 = kernel("gemm", "1");
const GEMM_V2 = kernel("gemm", "2");
const ADD = kernel("add", "3");
const SILU = kernel("silu", "4");

/** A temporary directory holding a store, a fixture with the given rows and a place for artifacts. */
const withSeats = async (
  rows: Record<string, string>,
  body: (seats: {
    readonly dir: URL;
    readonly store: ReferenceStore;
    readonly storeRoot: URL;
    readonly fixtureUrl: URL;
    readonly artifact: (name: string, bytes: Uint8Array<ArrayBuffer>) => URL;
  }) => void | Promise<void>,
): Promise<void> => {
  const path = makeTempDirSync({ prefix: "karume-reference-store-" });
  try {
    const dir = new URL(`file://${path}/`);
    const storeRoot = new URL("references/", dir);
    const fixtureUrl = new URL("references.json", dir);
    const cases = Object.fromEntries(
      Object.entries(rows).map(([caseId, sha]) => [caseId, { [ENVIRONMENT_KEY]: sha }]),
    );
    writeTextFileSync(fixtureUrl, `${JSON.stringify({ schema: 1, kind: "sha256", cases })}\n`);
    await body({
      dir,
      store: openReferenceStore(storeRoot),
      storeRoot,
      fixtureUrl,
      artifact: (name, bytes) => {
        const url = new URL(name, dir);
        writeFileSync(url, bytes);
        return url;
      },
    });
  } finally {
    removeSync(path, { recursive: true });
  }
};

const bytesOf = (text: string): Uint8Array<ArrayBuffer> => new TextEncoder().encode(text);

describe("the reference store", () => {
  it("keeps the artifact under the reference value with its extension, and the kernels of the run", async () => {
    await withSeats({}, ({ store, artifact }) => {
      const artifactUrl = artifact("case-1.png", bytesOf("pixels"));
      store.keep({
        sha256: SHA_A,
        caseId: "case-1",
        environment: ENVIRONMENT_KEY,
        artifactUrl,
        kernels: [ADD, GEMM_V1],
      });
      const kept = store.read(SHA_A);
      assert(kept !== undefined);
      assertEquals(kept.record, {
        schema: 1,
        sha256: SHA_A,
        caseId: "case-1",
        environment: ENVIRONMENT_KEY,
        artifact: `${SHA_A}.png`,
        kernels: [ADD, GEMM_V1],
      });
      assertEquals(readFileSync(kept.artifactUrl), bytesOf("pixels"));
    });
  });

  it("on a later matching run keeps the first artifact and takes the latest kernels", async () => {
    await withSeats({}, ({ store, artifact }) => {
      const first = artifact("first.wav", bytesOf("first"));
      const later = artifact("later.wav", bytesOf("later"));
      const entry = { sha256: SHA_A, caseId: "case-1", environment: ENVIRONMENT_KEY };
      store.keep({ ...entry, artifactUrl: first, kernels: [GEMM_V1] });
      store.keep({ ...entry, artifactUrl: later, kernels: [GEMM_V2] });
      const kept = store.read(SHA_A);
      assert(kept !== undefined);
      assertEquals(readFileSync(kept.artifactUrl), bytesOf("first"));
      assertEquals(kept.record.kernels, [GEMM_V2]);
    });
  });

  it("answers undefined for a value it never kept, and throws on a record it cannot read", async () => {
    await withSeats({}, ({ store, storeRoot, artifact }) => {
      assertEquals(store.read(SHA_A), undefined);
      store.keep({
        sha256: SHA_A,
        caseId: "case-1",
        environment: ENVIRONMENT_KEY,
        artifactUrl: artifact("case-1.png", bytesOf("pixels")),
        kernels: [],
      });
      writeTextFileSync(new URL(`${SHA_A}.json`, storeRoot), JSON.stringify({ schema: 99 }));
      assertThrows(() => store.read(SHA_A), Error, "unexpected shape");
    });
  });
});

describe("kernelChanges", () => {
  it("reports nothing for the same set in another order", () => {
    assertEquals(kernelChanges([ADD, GEMM_V1], [GEMM_V1, ADD]), {
      changed: [],
      added: [],
      removed: [],
    });
  });

  it("separates a different WGSL text under a label from labels added and removed", () => {
    assertEquals(kernelChanges([ADD, GEMM_V1], [GEMM_V2, SILU]), {
      changed: ["gemm"],
      added: ["silu"],
      removed: ["add"],
    });
  });
});

describe("mismatchEvidence", () => {
  it("says that nothing is kept when the store has no record of the expected value", async () => {
    await withSeats({}, ({ store }) => {
      const lines = mismatchEvidence(store, SHA_A, { kernels: [GEMM_V1] });
      assertEquals(lines.length, 1);
      assertStringIncludes(lines[0], "nothing is kept for the reference value");
    });
  });

  it("with the same kernels points outside the WGSL, and counts the differing bytes", async () => {
    await withSeats({}, ({ store, artifact }) => {
      store.keep({
        sha256: SHA_A,
        caseId: "case-1",
        environment: ENVIRONMENT_KEY,
        artifactUrl: artifact("kept.bin", new Uint8Array([1, 2, 3, 4])),
        kernels: [ADD, GEMM_V1],
      });
      const lines = mismatchEvidence(store, SHA_A, {
        kernels: [GEMM_V1, ADD],
        artifactUrl: artifact("actual.bin", new Uint8Array([1, 9, 3, 9])),
      });
      assertStringIncludes(lines[0], "the same 2 WGSL texts");
      assertStringIncludes(lines[0], "outside the WGSL");
      assertStringIncludes(lines[1], "2 of 4 bytes differ, the first at offset 1");
    });
  });

  it("with changed kernels names them, and reports a different artifact size", async () => {
    await withSeats({}, ({ store, artifact }) => {
      store.keep({
        sha256: SHA_A,
        caseId: "case-1",
        environment: ENVIRONMENT_KEY,
        artifactUrl: artifact("kept.bin", new Uint8Array([1, 2, 3])),
        kernels: [ADD, GEMM_V1],
      });
      const lines = mismatchEvidence(store, SHA_A, {
        kernels: [GEMM_V2, SILU],
        artifactUrl: artifact("actual.bin", new Uint8Array([1, 2, 3, 4])),
      });
      assertStringIncludes(lines[0], "changed since the last matching run");
      assertStringIncludes(lines[0], "1 with a different WGSL text (gemm)");
      assertStringIncludes(lines[0], "1 new (silu)");
      assertStringIncludes(lines[0], "1 no longer dispatched (add)");
      assertStringIncludes(lines[1], "3 bytes against this run's 4");
    });
  });
});

describe("References.observe", () => {
  it("keeps the artifact and the kernels of a run that created the row", async () => {
    await withSeats({}, ({ store, fixtureUrl, artifact }) => {
      const references = openReferences(fixtureUrl, {
        environment,
        mode: "write",
        store,
        kernels: () => [GEMM_V1],
      });
      const check = references.check("case-1", SHA_A);
      assertEquals(check, { status: "written" });
      const text = references.observe("case-1", SHA_A, check, artifact("case-1.png", bytesOf("p")));
      assertEquals(text, "");
      assertEquals(store.read(SHA_A)?.record.kernels, [GEMM_V1]);
    });
  });

  it("on a mismatch keeps nothing for the measured value and returns the evidence, each line indented", async () => {
    await withSeats({ "case-1": SHA_A }, ({ store, fixtureUrl, artifact }) => {
      const matching = openReferences(fixtureUrl, {
        environment,
        mode: undefined,
        store,
        kernels: () => [GEMM_V1],
      });
      matching.observe(
        "case-1",
        SHA_A,
        matching.check("case-1", SHA_A),
        artifact("kept.bin", new Uint8Array([7, 7])),
      );

      const moved = openReferences(fixtureUrl, {
        environment,
        mode: undefined,
        store,
        kernels: () => [GEMM_V2],
      });
      const check = moved.check("case-1", SHA_B);
      assertEquals(check, { status: "fail", expected: SHA_A });
      const text = moved.observe(
        "case-1",
        SHA_B,
        check,
        artifact("moved.bin", new Uint8Array([7, 8])),
      );
      assertEquals(store.read(SHA_B), undefined);
      const lines = text.split("\n");
      assertEquals(lines[0], "");
      assertStringIncludes(lines[1], "  kernels: changed since the last matching run");
      assertStringIncludes(lines[2], "1 of 2 bytes differ");
    });
  });

  it("does nothing without a store: a caller that injects an environment gets none by default", async () => {
    await withSeats({ "case-1": SHA_A }, ({ fixtureUrl, artifact, storeRoot }) => {
      const references = openReferences(fixtureUrl, { environment, mode: undefined });
      const url = artifact("case-1.png", bytesOf("p"));
      assertEquals(references.observe("case-1", SHA_A, references.check("case-1", SHA_A), url), "");
      assertEquals(references.observe("case-1", SHA_B, references.check("case-1", SHA_B), url), "");
      assertThrows(() => readTextFileSync(new URL(`${SHA_A}.json`, storeRoot)));
    });
  });
});

describe("settleReference with a store", () => {
  it("carries the evidence of a mismatch into the diagnostic message and leaves the verdict alone", async () => {
    await withSeats({}, async ({ dir, store, fixtureUrl }) => {
      const results = openResults("unit", { root: new URL("verify/", dir), environment });
      const kept = bytesOf("matching output");
      const writing = openReferences(fixtureUrl, {
        environment,
        mode: "write",
        store,
        kernels: () => [GEMM_V1],
      });
      const written = await settleReference(writing, results, {
        id: "case-1",
        artifact: "case-1.bin",
        bytes: kept,
      });
      assertEquals(written.check, { status: "written" });
      assertEquals(written.evidence, "");
      assertEquals(store.read(await sha256Hex(kept))?.record.caseId, "case-1");

      const comparing = openReferences(fixtureUrl, {
        environment,
        mode: undefined,
        store,
        kernels: () => [GEMM_V2],
      });
      const moved = await settleReference(comparing, results, {
        id: "case-1",
        artifact: "case-1.bin",
        bytes: bytesOf("differing output"),
      });
      assertEquals(moved.check, { status: "fail", expected: await sha256Hex(kept) });
      const message = referenceMismatchMessage("case-1", moved, { fixtureUrl });
      assertStringIncludes(message, "does not match the reference");
      assertStringIncludes(message, "1 with a different WGSL text (gemm)");
    });
  });
});
