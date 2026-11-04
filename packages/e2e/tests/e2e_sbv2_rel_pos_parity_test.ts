// Pins, on real data, the **byte parity of the host TS generator and the Python generator** for
// DeBERTa's relative-position index tables (ADR 0045 wave 3 — the same responsibility as
// `sbv2_relattn_parity_test.ts` for flow / voice).
//
// text_encoder does not bake the tables in but takes them as **graph inputs**. So at run time the
// host builds the tables, while Python (`patch_deberta.build_rel_pos_tables`) builds the golden
// ones — if the formulas diverge, **the shape still matches and only the values are wrong** (it
// gathers different position embeddings). Worse, some kinds of error "are shared by the host and
// the golden and slip through the E2E", so the two are compared directly here, apart from the E2E.
//
// torch computes the original formula in **float32**, the host in float64. That the ceil at the
// boundaries agrees is an empirical claim (ADR 0045 wave 3), so this is the only place that pins it
// on real data.
//
// No GPU. Where the assets come from (ADR 0139): the golden tables are read from the test-asset
// root, `goldenDir("deberta-i8")` = `outputs/test-assets/goldens/deberta-i8/sbv2-22layer/io.*`
// (fetched with `node tools/test-assets/fetch.ts`); the bucket rule (`symbols`) and the
// container's graph come from the distribution `models/karume-sbv2-jvnv/` through its manifest
// (the shared int8 text encoder, byte-identical to the series' 22-layer container). Each case
// SKIPs explicitly when the assets it reads are absent — except goldens absent from a complete
// fetch of the pinned dataset (`testAssetsFetched()`), which FAIL: the dataset at the pin lacks
// them.
//
// Asset-free invariants of buildRelPosTables (argument gate, transpose, Toeplitz) stay in
// packages/models/tests/sbv2_rel_pos_parity_test.ts.

import { type DirEntry, isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseSafetensors, prepareContainer } from "../../runtime/mod.ts";
import {
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  readMirrorAsset,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { buildRelPosTables } from "../../models/src/sbv2/text/rel-pos-tables.ts";
import { parseJpExtraRules } from "../../models/src/sbv2/text/symbols.ts";

/**
 * The goldens of the text encoder that takes the tables as inputs (`deberta-i8/sbv2-22layer` —
 * the distribution's variant): `outputs/test-assets/goldens/deberta-i8/sbv2-22layer/`.
 */
const GOLDEN_ROOT = new URL("sbv2-22layer/", goldenDir("deberta-i8"));
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/** The distribution: the int8 text encoder and the bucket rule (`symbols`). */
const MIRROR_ROOT = mirrorRoot("karume-sbv2-jvnv");
/**
 * The selection: a speaker, and a quant whose text encoder is the int8 one (every speaker shares
 * the same text encoder and symbols under `shared/`).
 */
const MIRROR_CHOICE = { model: "F1", quant: "i8" } as const;
/** The graph name in the container = the manifest's weights key (container-v1 §2.1). */
const GRAPH = "text_encoder";

/** The tensor keys of the tables stored in io (= the graph input names). */
const C2P_KEY = "input.c2p_pos";
const P2C_KEY = "input.p2c_pos";

const listDir = (url: URL): readonly DirEntry[] => {
  try {
    return [...readDirSync(url)];
  } catch (cause) {
    if (isNotFound(cause)) {
      return [];
    }
    throw cause;
  }
};

/** The bucket rule, read from the distribution's `symbols` asset (one source — not copied into TS). */
const readRule = async (): Promise<{ positionBuckets: number; maxPosition: number }> => {
  const bytes = await readMirrorAsset(MIRROR_ROOT, "symbols", MIRROR_CHOICE);
  const raw = JSON.parse(new TextDecoder().decode(bytes));
  return parseJpExtraRules(raw, `${MIRROR_ROOT.pathname} (symbols)`).bertRelPos;
};

const CASES = listDir(GOLDEN_ROOT)
  .filter(
    (entry) => entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX),
  )
  .map((entry) => entry.name)
  .sort();
const MIRROR_PRESENT = mirrorPresent(MIRROR_ROOT);

// No golden at all: a FAIL after a complete fetch of the pin, a SKIP by name otherwise (never an
// empty registration).
if (CASES.length === 0) {
  if (testAssetsFetched()) {
    test("DeBERTa relative-position table parity: the goldens are present", () => {
      throw new Error(
        `${GOLDEN_ROOT.pathname} holds no io.* golden, but outputs/test-assets/ holds a complete ` +
          `fetch of ${TEST_ASSETS.repo}@${TEST_ASSETS.revision}: the dataset at the pin lacks them`,
      );
    });
  } else {
    test.skip(
      "DeBERTa relative-position table parity (skipped: no goldens under " +
        `outputs/test-assets/goldens/deberta-i8/sbv2-22layer/; fetch: ${FETCH_COMMAND})`,
      () => {},
    );
  }
}
if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] skipping the DeBERTa relative-position checks: the distribution mirror ` +
      `${MIRROR_ROOT.pathname} is absent (or not karume/5).`,
  );
}

/** Exact element-wise equality (the index tables are i32, so `Object.is` is strong enough). */
const assertSameElements = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: element count`);
  for (let index = 0; index < expected.length; index += 1) {
    if (!Object.is(actual[index], expected[index])) {
      throw new Error(
        `${where}: element ${index} differs (TS=${actual[index]} / Python=${expected[index]})`,
      );
    }
  }
};

for (const ioFile of CASES) {
  test.skipIf(!MIRROR_PRESENT)(
    `DeBERTa relative-position table parity: ${ioFile} (TS generator ↔ the golden's Python generator)`,
    async () => {
      const rule = await readRule();
      const bytes = await readFile(new URL(ioFile, GOLDEN_ROOT));
      const io = parseSafetensors(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      );
      const c2pView = io.tensors.get(C2P_KEY);
      const p2cView = io.tensors.get(P2C_KEY);
      assert(c2pView !== undefined && p2cView !== undefined, `${ioFile} has no index-table input`);
      assertEquals(c2pView.dtype, "I32", `${C2P_KEY} storage dtype`);
      assertEquals(p2cView.dtype, "I32", `${P2C_KEY} storage dtype`);

      // The actual length is taken from the golden's shape (fixing it on the TS side would, once
      // cases are added, "compare equal lengths with each other and always be green").
      assertEquals(c2pView.shape.length, 2, `${C2P_KEY} rank`);
      assertEquals(c2pView.shape, p2cView.shape, "shape of the two tables");
      const [rows, columns] = c2pView.shape;
      assertEquals(rows, columns, "the relative-position table is square (self-attention)");

      const built = buildRelPosTables(rows, rule.positionBuckets, rule.maxPosition);
      assertEquals(built.c2pPos.shape, [...c2pView.shape], `${ioFile}: c2p_pos shape`);
      assertSameElements(
        built.c2pPos.data,
        new Int32Array(io.buffer, c2pView.byteOffset, c2pView.byteLength / 4),
        `${ioFile}: c2p_pos`,
      );
      assertSameElements(
        built.p2cPos.data,
        new Int32Array(io.buffer, p2cView.byteOffset, p2cView.byteLength / 4),
        `${ioFile}: p2c_pos`,
      );
    },
  );
}

test.skipIf(!MIRROR_PRESENT)(
  "DeBERTa: no relative-position index table is baked into the container",
  async () => {
    // If the promotion took effect, no `sym_prefix_slice` that slices by T remains (one left
    // means still carrying Tmax=512's `[1,512,512]` = 2 MiB of dead weight). This kind of fault
    // only brings the size back while the values stay correct, so the E2E does not catch it.
    // The graph descriptor is in part 0 of the container (container-v1 §8) — no weight block is
    // read.
    const model = prepareContainer(
      await openSelectedContainer(await selectMirrorComponent(MIRROR_ROOT, GRAPH, MIRROR_CHOICE)),
      GRAPH,
    );
    const baked = model.graph.nodes.filter((node) => node.op === "sym_prefix_slice");
    assertEquals(baked.length, 0, "count of baked tables (sym_prefix_slice)");
    assertEquals(
      model.graph.inputs.map((input) => input.name),
      ["input_ids", "attention_mask", "c2p_pos", "p2c_pos"],
      "order of the graph inputs",
    );
  },
);
