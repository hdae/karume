// Checks for `helpers/container-files.ts` (the surface that opens a series output's / mirror's
// `krm` as a part list). No GPU and no real assets are needed — a synthetic container
// (`helpers/container-write.ts`) is written to a temporary directory in split form and reopened
// from those files.
//
// The e2e tests that use this helper **pass only well-formed real assets**, so the fail-loudly
// branches (single form next to numbered parts / disagreeing `of` / a missing number / an
// out-of-range number / a different dtype mixed in), range reads and sha256 verification would be
// exercised only on machines that have the real assets. This is where they are exercised on the CPU
// alone.
//
// The Python mirror is `tools/exporter/tests/test_container.py` plus `TestPartExpansion` in
// `tools/exporter/tests/test_dist.py` (expanding the numbered part list, rejecting single form next
// to numbered parts).

import { makeTempDir, readFileSync, remove, writeFileSync } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import {
  fileBlockSource,
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "./helpers/container-files.ts";
import {
  type ModelInput,
  writeModelContainer,
  type WrittenContainer,
} from "./helpers/container-write.ts";
import { type IrDeclaration, parseIrDeclaration } from "../src/format/ir.ts";

// ---------------------------------------------------------------------------
// Synthetic container
// ---------------------------------------------------------------------------

/** Deterministic pseudo-random bytes (xorshift32 from seed). */
const bytesOf = (length: number, seed: number): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(new ArrayBuffer(length));
  let state = seed >>> 0 || 1;
  for (let i = 0; i < length; i += 1) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    out[i] = state & 0xff;
  }
  return out;
};

type InitializerSpec = { readonly shape: readonly number[]; readonly shared?: true };

/** The smallest IR v2 graph that consumes each initializer in one node. */
const declaration = (initializers: Readonly<Record<string, InitializerSpec>>): IrDeclaration => {
  const names = Object.keys(initializers);
  return parseIrDeclaration(
    JSON.stringify({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["matmul"] },
      symbols: ["T"],
      inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
      outputs: names.map((_, i) => `y${i}`),
      initializers: Object.fromEntries(
        names.map((name) => [name, initializers[name].shared ? { shared: true } : {}]),
      ),
      values: {
        ...Object.fromEntries(
          names.map((name) => [name, { dtype: "f32", shape: initializers[name].shape }]),
        ),
        ...Object.fromEntries(names.map((_, i) => [`y${i}`, { dtype: "f32", shape: ["T", 4] }])),
      },
      nodes: names.map((name, i) => ({
        op: "matmul",
        ins: ["x", name],
        outs: [`y${i}`],
        attrs: {},
      })),
    }),
  );
};

const F32 = 4;

/** The graph name spells the **family's weights key** (`model`), not the series component name. */
const GRAPH = "model";

const syntheticModel = (): ModelInput => ({
  graphs: {
    [GRAPH]: declaration({
      "enc.weight": { shape: [8, 16] },
      "proj.weight": { shape: [6, 16] },
      "const.rope": { shape: [16] },
      "head.weight": { shape: [4, 4], shared: true },
    }),
  },
  consts: [
    {
      graph: GRAPH,
      initializer: "const.rope",
      bytes: bytesOf(16 * F32, 11),
      encoding: { codec: "f32" },
    },
  ],
  weights: [
    {
      graph: GRAPH,
      initializer: "enc.weight",
      bytes: bytesOf(8 * 16 * 2, 21),
      encoding: { codec: "f16" },
    },
    {
      // i8 + companion scale (per-channel with row length 16).
      graph: GRAPH,
      initializer: "proj.weight",
      bytes: bytesOf(6 * 16, 22),
      encoding: {
        codec: "int8-sym",
        groupSize: 16,
        scale: { bytes: bytesOf(6 * F32, 23), dtype: "f32" },
      },
    },
  ],
  assets: [
    { name: "ple_index", role: "ple-index", bytes: bytesOf(200, 31) },
    { name: "style", role: "style-vectors", bytes: bytesOf(120, 32), dedicatedPart: true },
  ],
  provenance: { license: "apache-2.0", writer: "test" },
});

/** Tightens the part length / block limit so that even one component spans several parts. */
const OPTIONS = { partBytes: 320, blockBytes: 256 } as const;

const REPRESENTATIVE = "model.krm";

/** `BlockSource` part index (0-based) → split-form file name (the numbering is 1-based). */
const partName = (index: number, total: number): string =>
  `model-${String(index + 1).padStart(5, "0")}-of-${String(total).padStart(5, "0")}.krm`;

/** A temporary directory holding a zero-byte file (pass the URL with a trailing `/`). */
const withFiles = async (names: readonly string[], body: (dir: URL) => void): Promise<void> => {
  const path = await makeTempDir({ prefix: "karume-container-files-" });
  try {
    const dir = new URL(`file://${path}/`);
    for (const name of names) writeFileSync(new URL(name, dir), new Uint8Array(0));
    body(dir);
  } finally {
    await remove(path, { recursive: true });
  }
};

/** A temporary directory holding a synthetic container in split form. */
const withContainer = async (
  body: (dir: URL, written: WrittenContainer) => Promise<void>,
): Promise<void> => {
  const written = await writeModelContainer(syntheticModel(), OPTIONS);
  const path = await makeTempDir({ prefix: "karume-container-files-" });
  try {
    const dir = new URL(`file://${path}/`);
    for (const [index, bytes] of written.parts.entries()) {
      writeFileSync(new URL(partName(index, written.parts.length), dir), bytes);
    }
    await body(dir, written);
  } finally {
    await remove(path, { recursive: true });
  }
};

// ---------------------------------------------------------------------------
// resolveParts / modelPresent (how files are found)
// ---------------------------------------------------------------------------

test("resolveParts returns an unsplit container as a single representative path", async () => {
  await withFiles([REPRESENTATIVE], (dir) => {
    const representative = new URL(REPRESENTATIVE, dir);
    assertEquals(
      resolveParts(representative).map((url) => url.href),
      [representative.href],
    );
  });
});

test("resolveParts returns numbered parts in ascending order (guarantees the part index)", async () => {
  // They are placed out of order (a detector that it does not depend on `readDirSync`'s
  // listing order).
  const names = [
    "model-00003-of-00003.krm",
    "model-00001-of-00003.krm",
    "model-00002-of-00003.krm",
  ];
  await withFiles(names, (dir) => {
    const files = resolveParts(new URL(REPRESENTATIVE, dir)).map((url) => url.href);
    assertEquals(
      files,
      [...names].sort().map((name) => new URL(name, dir).href),
    );
  });
});

test("resolveParts fails loudly when single form and numbered parts sit together", async () => {
  await withFiles(
    [REPRESENTATIVE, "model-00001-of-00002.krm", "model-00002-of-00002.krm"],
    (dir) => {
      const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "sit together");
    },
  );
});

test("resolveParts fails loudly on numbered parts whose `of` totals disagree", async () => {
  await withFiles(["model-00001-of-00003.krm", "model-00002-of-00004.krm"], (dir) => {
    const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
    assertStringIncludes(error.message, "totals of the numbered parts disagree");
  });
});

test("resolveParts fails loudly on a missing number and names which one is missing", async () => {
  await withFiles(["model-00001-of-00003.krm", "model-00003-of-00003.krm"], (dir) => {
    const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
    assertStringIncludes(error.message, "part 2 is missing");
  });
});

test("resolveParts fails loudly on a number beyond the total", async () => {
  await withFiles(
    [
      "model-00001-of-00003.krm",
      "model-00002-of-00003.krm",
      "model-00003-of-00003.krm",
      "model-00004-of-00003.krm",
    ],
    (dir) => {
      const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "out-of-range numbers 4");
    },
  );
});

// Numbers at or below 0 (overrunning the 1-based numbering on the low side). Counting only the
// high side would fail with an empty list of out-of-range numbers (`[]`) — it does fail, but the
// diagnostic does not say what is extra.
test("resolveParts names part 0 as an extra number too", async () => {
  await withFiles(
    ["model-00000-of-00002.krm", "model-00001-of-00002.krm", "model-00002-of-00002.krm"],
    (dir) => {
      const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "out-of-range numbers 0");
    },
  );
});

// The stem is embedded into a regular expression, so without the escape `.` matches any one
// character and picks up the numbered parts of another dtype (picking them up would fail on the
// disagreeing totals {1,2}, so this check pins both the escape and the exact stem match by "exactly
// one path comes back").
test("resolveParts does not pick up a container of another dtype (exact stem match)", async () => {
  await withFiles(
    ["model.i8-00001-of-00001.krm", "modelXi8-00002-of-00002.krm", "other-00001-of-00002.krm"],
    (dir) => {
      const files = resolveParts(new URL("model.i8.krm", dir));
      assertEquals(
        files.map((url) => url.href),
        [new URL("model.i8-00001-of-00001.krm", dir).href],
      );
    },
  );
});

test("modelPresent treats a container with only numbered parts as present", async () => {
  await withFiles(["model-00001-of-00002.krm", "model-00002-of-00002.krm"], (dir) => {
    assert(modelPresent(new URL(REPRESENTATIVE, dir)));
  });
});

test("modelPresent treats a directory with no container at all as absent", async () => {
  await withFiles(["io.ramp.safetensors"], (dir) => {
    assertEquals(modelPresent(new URL(REPRESENTATIVE, dir)), false);
  });
});

// ---------------------------------------------------------------------------
// fileBlockSource (how ranges are read)
// ---------------------------------------------------------------------------

test("fileBlockSource hands out the part count and lengths from the files", async () => {
  await withContainer((dir, written) => {
    const source = fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir)));
    assertEquals(source.partCount, written.parts.length);
    assert(
      source.partCount >= 3,
      `${source.partCount} parts are not enough to check the split form`,
    );
    assertEquals(
      written.parts.map((_, index) => source.partLength(index)),
      written.parts.map((part) => part.byteLength),
    );
    return Promise.resolve();
  });
});

// MUST: if a source that nobody verified claimed `verified: true`, `openContainer` would stop
// checking the per-block sha256 (container-v1 §7) = tampering would slip through.
test("fileBlockSource does not claim to be verified", async () => {
  await withContainer((dir) => {
    assertEquals(fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir))).verified, false);
    return Promise.resolve();
  });
});

test("fileBlockSource returns only the requested range from the middle of a part", async () => {
  await withContainer(async (dir, written) => {
    const source = fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir)));
    // Read from the middle of the last part (the dedicated asset part) — without a working seek the
    // head would come back.
    const part = written.parts.length - 1;
    const bytes = written.parts[part];
    const offset = Math.floor(bytes.byteLength / 3);
    const length = Math.min(17, bytes.byteLength - offset);
    assertEquals(await source.read(part, offset, length), bytes.subarray(offset, offset + length));
  });
});

test("fileBlockSource fails loudly on a range that runs past the part length", async () => {
  await withContainer(async (dir, written) => {
    const source = fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir)));
    const error = await assertRejects(
      () => source.read(0, written.parts[0].byteLength - 1, 2),
      Error,
    );
    assertStringIncludes(error.message, "outside the part length");
  });
});

test("fileBlockSource fails loudly on a part index that does not exist", async () => {
  await withContainer(async (dir, written) => {
    const source = fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir)));
    const error = await assertRejects(() => source.read(written.parts.length, 0, 1), Error);
    assertStringIncludes(error.message, "does not exist");
  });
});

// ---------------------------------------------------------------------------
// openSeriesContainer (part list → opened container)
// ---------------------------------------------------------------------------

test("openSeriesContainer opens a container from a split-form part list", async () => {
  await withContainer(async (dir) => {
    const opened = await openSeriesContainer(new URL(REPRESENTATIVE, dir));
    assertEquals(opened.header.kind, "model");
    assertEquals(Object.keys(opened.graphs), [GRAPH]);
    assertEquals(opened.source.verified, false);
  });
});

test("a container opened with openSeriesContainer hands out blocks as the declared bytes", async () => {
  await withContainer(async (dir, written) => {
    const opened = await openSeriesContainer(new URL(REPRESENTATIVE, dir));
    for (const block of written.model.blocks) {
      assertEquals(
        await opened.readBlock(block.id),
        written.parts[block.part].subarray(block.offset, block.offset + block.length),
        `block '${block.id}'`,
      );
    }
  });
});

// Fault injection — corrupting one byte of a file makes the block's sha256 disagree. If this stays
// green, the `verified: false` path is not working (= verification is missing for range-reading
// sources).
test("openSeriesContainer rejects a tampered part through the block sha256", async () => {
  await withContainer(async (dir, written) => {
    const block = written.model.blocks[0];
    const file = new URL(partName(block.part, written.parts.length), dir);
    const bytes = readFileSync(file);
    bytes[block.offset] ^= 0xff;
    writeFileSync(file, bytes);
    const opened = await openSeriesContainer(new URL(REPRESENTATIVE, dir));
    const error = await assertRejects(() => opened.readBlock(block.id), Error);
    assertStringIncludes(error.message, "sha256");
  });
});

test("a container opened with openSeriesContainer can read assets by range", async () => {
  await withContainer(async (dir) => {
    const opened = await openSeriesContainer(new URL(REPRESENTATIVE, dir));
    const reader = opened.asset("ple_index");
    assertEquals(reader.role, "ple-index");
    assertEquals(reader.length, 200);
    const whole = syntheticModel().assets[0].bytes;
    assertEquals(await reader.read(0, reader.length), whole);
    assertEquals(await reader.read(64, 8), whole.subarray(64, 72));
  });
});

test("openSeriesContainer fails loudly on a series missing one part", async () => {
  await withContainer(async (dir, written) => {
    await remove(new URL(partName(1, written.parts.length), dir));
    const error = await assertRejects(
      () => openSeriesContainer(new URL(REPRESENTATIVE, dir)),
      Error,
    );
    assertStringIncludes(error.message, "part 2 is missing");
  });
});
