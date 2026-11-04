// Acceptance check of the real-weight Gemma 4 E2B **product graph** (PLE moved out + last-row
// logits exit) — step 1b.
//
// What it accepts is the product form that puts ADR
// [0083](../../../docs/decisions/0083-generation-api-surface.md) decision 6 and ADR
// [0085](../../../docs/decisions/0085-ple-host-gather.md) **on the same re-export** (option α);
// the existing logits opt-in series remains as an acceptance fixture (the token-only exit series
// was removed on 2026-10-17).
//
// Four checks:
//
// ① Form premises (7 inputs = input_ids / **per_layer_inputs** / **4 rope inputs** / last_row[R];
//    2 outputs = the selected row's logits `[1,R,V]` + the hidden after the final norm `[1,R,H]`;
//    **no argmax in the graph**; no embedding that looks up the PLE table; the 30 states slots are
//    identical to the logits opt-in series (sliding is window + 8))
// ② **Bit identity of the PLE dequantization** (ADR 0085 decision 4): the host loader's gather is
//    **exactly identical** to `ple.probe.safetensors`, which the script took through torch's
//    35-table path (= exactly the values `embedding` + the `mul` right after it would have produced
//    had the PLE stayed in the graph). No GPU needed. It also measures "lazy load of only the
//    touched blocks + LRU" (decision 3).
//    The container read here is the `bf16` preset's (schema 5): its "dequantization" is the exact
//    widening of the bf16 bits to f32, then `× embedScale` — the policy checks walk whichever tables
//    the index has (`values` alone here; `values` + `scales` on a quantized index).
// ③ **Cross parity**: the greedy loop of host PLE gather + host `argmax(logits)` is exactly
//    identical to the logits opt-in series' `greedy.<case>.safetensors` (the expected sequences of
//    a torch full re-forward) over 3 cases × K=16. This proves that moving the PLE out and taking
//    argmax off the exit **leave the function unchanged**; a split means an error in the PLE
//    gather, the dequantization or the row selection.
// ④ **Provenance binding**: the `reference.json` the product series writes holds the fingerprint
//    of the source checkpoint and the digest of each golden ③ reuses (a combination where only one
//    side is stale could still turn ③ green).
//
// The host greedy loop is **built by this check itself**. `src/generation/greedy.ts` reads the
// graph-side argmax output (the token-only exit — that exit series was removed on 2026-10-17) and
// has a MUST of no host sampling, so it cannot be used
// (the product surface's `GenerationSequence` belongs to steps 2–3 — ADR 0083 decisions 1–8). Only
// the chunk split shares `planPrefillChunks` (so that a split error is not copied into the
// expectations, everything **except** the split is written out).
//
// ## Assets
//
// The container is the distribution mirror's `model` component (`models/karume-gemma4/`, model
// e2b, quant bf16 — the regular line's one preset since 2026-11-04: the graph and the checkpoint's
// bf16 weights next to 4.70 GB of bf16 PLE assets; byte-identical to the bf16 product series it was
// assembled from), opened through the manifest (ADR 0139 decision 1). The goldens come from the
// test-asset root (decision 2): `goldens/gemma4-e2b-product-bf16/` — the series' own
// `ple.probe.safetensors`, `reference.json` and greedy expected lists (the series binds its own
// goldens, where the int4 product borrowed the decode series'). Without the mirror the container
// cases SKIP explicitly. The goldens follow ADR
// 0139's fetched rule: a golden absent from a complete fetch of the pinned test-asset dataset FAILs
// the case that reads it (the dataset at the pin lacks it); without a complete fetch that case SKIPs
// with the missing files and `node tools/test-assets/fetch.ts` in its title. A mirror whose
// container lacks the PLE assets is a FAIL, not a SKIP.

import { isNotFound, readFile, readTextFile, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type SafetensorsFile,
  type Tensor,
} from "@karume/runtime";
import { createGemma4Ple, type Gemma4Ple } from "../../models/src/gemma/ple.ts";
import {
  defaultGemma4PleResidentBytes,
  gemma4PleBlockBytes,
  gemma4PleBlockOf,
  type Gemma4PleFloatIndex,
  type Gemma4PleIndex,
  type Gemma4PleTable,
  gemma4PleTables,
  PLE_INDEX_ASSET,
} from "../../models/src/gemma/ple-index.ts";
import { gemma4RopeInputNames, ropeInputs, type RopeSpec } from "../../models/src/gemma/rope.ts";
import { planPrefillChunks } from "../../models/src/generation/greedy.ts";
import {
  goldenDir,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
} from "@karume/test-support/test-assets";
import {
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";
// The helper is the source of truth for the fully resident budget (the same formula is not copied
// into each e2e).
import { allResidentBytes } from "../../models/tests/helpers/ple-budget.ts";
// The PLE index and block readers of the mirror's `model` container (ADR 0109 decision 4).
import { openGemma4Ple } from "../../models/tests/helpers/gemma-mirror.ts";

/** The distribution mirror that ships the product graph. */
const MIRROR = mirrorRoot("karume-gemma4");
/**
 * The mirror seat this gate reads: e2b with the bf16 weights. MUST: name the quant — the sessions
 * below are built with the options this gate always used (none, which the `bf16` preset declares
 * too), so the bytes it checks stay those of the named seat whatever the manifest's default becomes.
 */
const PRODUCT_CHOICE = { model: "e2b", quant: "bf16" } as const;
/** The component key, which is also the container's graph name (container-v1 §2.1). */
const MODEL_GRAPH = "model";
/** The bf16 product series' goldens (the PLE probe and the provenance record). */
const PRODUCT_ROOT = goldenDir("gemma4-e2b-product-bf16");
/**
 * The greedy goldens of ③ and of the provenance record — the bf16 series binds its own
 * (`reference.series` is the series itself), so this is the same folder.
 */
const GOLDEN_ROOT = goldenDir("gemma4-e2b-product-bf16");
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";

/**
 * The PLE dequantization reference (spelling source of truth: `gemma4/export_product.py`). The
 * index is on the container-asset side.
 */
const PLE_PROBE_FILE = "ple.probe.safetensors";
const PROBE_TOKENS_KEY = "tokens";
const PROBE_INPUTS_KEY = "per_layer_inputs";

/**
 * The file name and version of the provenance record (spelling source of truth:
 * `gemma4/provenance.py`).
 */
const REFERENCE_FILE = "reference.json";
const REFERENCE_SCHEMA = 1;

/** The command that restores the goldens (ADR 0139 decision 2). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/** The recipe commands that regenerate the product container and the goldens. */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m gemma4.export_product" +
  " --storage bf16";

/**
 * The source of truth of the expected sequences = the logits opt-in series' greedy goldens
 * (canonical list: export_decode.GREEDY_CASES).
 */
const EXPECTED_CASES = ["capital-en", "capital-ja", "context-en"] as const;
const GREEDY_STEPS = 16;

/**
 * The run conditions equal those of the two existing series' checks (the ruling for the same asset
 * generation is reused as is).
 */
const CHUNK_LENGTH = 768;
/**
 * The chunk length used only for the longest case (`context-en` = T:598). At 768 all three cases
 * fit in one chunk, and **the multi-chunk prefill specific to the product graph** (a `ple.gather`
 * per chunk + the LRU spanning them) never runs on the real weights. At 256 it splits into three
 * chunks, so whether the gather looks up the right token range at chunk boundaries (= the
 * dequantization and row selection across tables) shows up in the cross-parity values themselves.
 */
const LONG_CASE = "context-en";
const LONG_CASE_CHUNK_LENGTH = 256;
const CAPACITY_SYMBOL = "C";
const CAPACITY = 4096;
const MAX_POSITION = 131072;

/**
 * The RoPE parameters (the same values as the distribution's declaration — the asset has no table;
 * the host builds cos / sin).
 */
const ROPE: RopeSpec = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
};

/** Graph input names (source of truth: the constants of `export_decode` / `export_product`). */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";

/**
 * The real asset's shape (`num_hidden_layers` / `hidden_size_per_layer_input` / `vocab_size` from
 * the config).
 */
const LAYERS = 35;
const PLE_DIM = 256;
const VOCAB = 262144;
/**
 * The width of the hidden after the final norm (config `hidden_size` — the last axis of output 1).
 */
const HIDDEN = 1536;
/**
 * The row-count symbol (`last_row[R]` and the row axis of both outputs — the exporter's
 * `ROW_SYMBOL`).
 */
const ROW_SYMBOL = "R";
/** The number of states slots (15 owning layers × k/v — identical to the logits opt-in series). */
const SLOTS = 30;

/**
 * The ceiling of the resident compressed weights (ADR 0085's claim itself: a container with the
 * PLE moved out does not put the PLE on the GPU).
 *
 * Re-collected 2026-11-04 on the RTX 3080 Ti (Node + Dawn) when the gate moved from the int4 seat
 * (3.70 → 1.51 GiB, ceiling 2.0e9 with the decode series above 3.0e9) to `bf16` (a new target):
 * the bf16 container keeps 4,558,946,304 B resident, and kept in the graph its PLE would add the
 * 4.70e9 B of bf16 values (262,144 × 35 × 256 × 2). The line sits between the two with the old
 * line's relative headroom (about 1.2 × the measurement).
 */
const RESIDENT_CEILING = 5_500_000_000;

const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const MODEL_PRESENT = mirrorPresent(MIRROR);

if (!MODEL_PRESENT) {
  console.warn(
    `[karume] the distribution mirror ${MIRROR.pathname} is absent (or not karume/5), so the ` +
      "Gemma 4 E2B product-graph cases SKIP",
  );
}

const PLE_PROBE = new URL(PLE_PROBE_FILE, PRODUCT_ROOT);
const REFERENCE = new URL(REFERENCE_FILE, PRODUCT_ROOT);
const GREEDY_GOLDENS = EXPECTED_CASES.map(
  (name) => new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT),
);

/** The goldens of `files` that are absent, as paths under the test-asset root. */
const missingGoldens = (files: readonly URL[]): string[] =>
  files
    .filter((file) => !exists(file))
    .map((file) => file.href.slice(TEST_ASSETS_ROOT.href.length));

/**
 * ADR 0139's fetched rule for the goldens a case reads: absent with a complete fetch of the pin ⇒
 * the case runs and {@link assertGoldensPresent} FAILs naming them; absent without one ⇒ the case
 * SKIPs, and `title` says why (the assets gate is red then).
 */
const goldensGate = (files: readonly URL[]): { readonly skip: boolean; readonly title: string } => {
  const missing = missingGoldens(files);
  const skip = missing.length > 0 && !testAssetsFetched();
  const title = skip
    ? ` (SKIP: ${missing.join(" / ")} absent and outputs/test-assets/ is not a complete fetch of ` +
      `the pin — ${FETCH_COMMAND})`
    : "";
  return { skip, title };
};

/** FAILs naming each absent golden of `files` (the dataset at the pin lacks it). */
const assertGoldensPresent = (files: readonly URL[]): void => {
  const missing = missingGoldens(files);
  assert(
    missing.length === 0,
    `${missing.join(" / ")} absent although outputs/test-assets/ holds a complete fetch of ` +
      `${TEST_ASSETS.repo}@${TEST_ASSETS.revision} — the test-asset dataset at the pin lacks it`,
  );
};

const PROBE_GATE = goldensGate([PLE_PROBE]);
const PROVENANCE_GATE = goldensGate([REFERENCE, ...GREEDY_GOLDENS]);
const GOLDENS_PRESENT = GREEDY_GOLDENS.every((file) => exists(file));
const AVAILABLE = MODEL_PRESENT && GOLDENS_PRESENT;

/** Opens the mirror's `model` container (e2b / bf16) with range reads. */
const openProductContainer = async () =>
  await openSelectedContainer(await selectMirrorComponent(MIRROR, MODEL_GRAPH, PRODUCT_CHOICE));

/**
 * The container's completeness (a missing asset is not folded into a SKIP): a container that has
 * the graph but **no PLE assets** (the PLE was not moved out) is a defect, not "not generated", so
 * it FAILs. The goldens are not checked here — the default cases that read them (the PLE
 * bit-exactness case and the provenance case, which also reads the greedy expected lists of ③ and
 * of `e2e_gemma4_sequence_test.ts`) follow ADR 0139's fetched rule.
 */
test.skipIf(!MODEL_PRESENT)(
  "Gemma 4 E2B product assets: the mirror's model container carries the PLE assets",
  async () => {
    const opened = await openProductContainer();
    assert(
      Object.hasOwn(opened.model?.assets ?? {}, PLE_INDEX_ASSET),
      `${MODEL_GRAPH} has no asset '${PLE_INDEX_ASSET}' (the graph is there but the PLE was not` +
        ` moved out — per_layer_inputs cannot be built from this asset)`,
    );
  },
);

/** The name of the series directory (the side compared with the `series` the record names). */
const seriesName = (root: URL): string =>
  root.pathname
    .split("/")
    .filter((part) => part !== "")
    .at(-1) ?? "";

/**
 * Turns one level of JSON into "key → unchecked value" (an unknown shape fails with a clear
 * message).
 */
const objectAt = (value: unknown, where: string): Map<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${REFERENCE_FILE}: ${where} is not an object`);
  }
  return new Map(Object.entries(value));
};

const stringAt = (entries: Map<string, unknown>, key: string, where: string): string => {
  const value = entries.get(key);
  if (typeof value !== "string" || value === "") {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key} is not a non-empty string`);
  }
  return value;
};

/** One `{bytes, sha256}` entry (it also checks that sha256 is 64 lowercase hex digits). */
const digestAt = (
  entries: Map<string, unknown>,
  key: string,
  where: string,
): { bytes: number; sha256: string } => {
  const digest = objectAt(entries.get(key), `${where}.${key}`);
  const bytes = digest.get("bytes");
  if (typeof bytes !== "number" || !Number.isSafeInteger(bytes) || bytes <= 0) {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key}.bytes is not a positive integer`);
  }
  const sha256 = stringAt(digest, "sha256", `${where}.${key}`);
  if (!/^[0-9a-f]{64}$/.test(sha256)) {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key}.sha256 is not 64 hex digits`);
  }
  return { bytes, sha256 };
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> => {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
};

/**
 * ④ Provenance binding. The goldens ③ reuses must be "the ones whose digests the run that built
 * this container took on the spot" — re-taking the logits opt-in series moves the digests, this
 * check fails and forces a re-export of the product series (the token-only series' twin gate was
 * removed on 2026-10-17).
 */
test.skipIf(PROVENANCE_GATE.skip)(
  `Gemma 4 E2B product assets: the provenance record binds the reference goldens${PROVENANCE_GATE.title}`,
  async () => {
    const path = REFERENCE;
    assertGoldensPresent([REFERENCE, ...GREEDY_GOLDENS]);
    const record = objectAt(JSON.parse(await readTextFile(path)), "root");

    assertEquals(record.get("schema"), REFERENCE_SCHEMA, `schema of ${REFERENCE_FILE}`);
    assertEquals(
      stringAt(record, "series", "root"),
      seriesName(PRODUCT_ROOT),
      "the series the record names",
    );
    const checkpoint = objectAt(record.get("checkpoint"), "checkpoint");
    stringAt(checkpoint, "dir", "checkpoint");
    const fingerprint = objectAt(checkpoint.get("files"), "checkpoint.files");
    assert(fingerprint.size > 0, `${REFERENCE_FILE}: checkpoint.files is empty`);
    for (const file of fingerprint.keys()) digestAt(fingerprint, file, "checkpoint.files");

    const reference = objectAt(record.get("reference"), "reference");
    assertEquals(
      stringAt(reference, "series", "reference"),
      seriesName(GOLDEN_ROOT),
      "the bound golden series (the same series ③ reads)",
    );
    const goldens = objectAt(reference.get("goldens"), "reference.goldens");
    assertEquals(
      [...goldens.keys()].sort(),
      EXPECTED_CASES.map((name) => `${GREEDY_PREFIX}${name}${SUFFIX}`).sort(),
      "the set of bound goldens (exactly the 3 ③ reads)",
    );
    for (const name of EXPECTED_CASES) {
      const file = `${GREEDY_PREFIX}${name}${SUFFIX}`;
      const digest = digestAt(goldens, file, "reference.goldens");
      const bytes = await readFile(new URL(file, GOLDEN_ROOT));
      assertEquals(bytes.byteLength, digest.bytes, `${file}: byte count`);
      assertEquals(
        await sha256Hex(bytes),
        digest.sha256,
        `${file}: sha256（differs from the golden the product series bound — ` +
          `a combination where only one of the series was rebuilt — the test-asset dataset at the pin holds ` +
          `goldens of different runs: regenerate both (${GENERATE_COMMAND}), upload them to ` +
          `${TEST_ASSETS.repo} and move the pin; the container comes from the mirror）`,
      );
    }
  },
);

/**
 * Reads one file as an `ArrayBuffer`.
 * MUST: do not slice when the view covers the whole buffer — a golden read here can be several
 * hundred MB, so an unconditional `slice` would double the peak.
 */
const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await readFile(new URL(file, root));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `the golden has no '${name}'`);
  assertEquals(view.dtype, "I32", `storage dtype of golden '${name}'`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const goldenF32 = (file: SafetensorsFile, name: string): Float32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `the golden has no '${name}'`);
  assertEquals(view.dtype, "F32", `storage dtype of golden '${name}'`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** Reads the index and builds the loader (block reads go through the mirror's manifest source). */
const openPle = async (maxResidentBytes: number): Promise<Gemma4Ple> => {
  const { index, openBlock } = await openGemma4Ple(MIRROR, PRODUCT_CHOICE);
  return createGemma4Ple({ index, openBlock, vocabSize: VOCAB, maxResidentBytes });
};

const readPleIndex = async (): Promise<Gemma4PleFloatIndex> => {
  const { index } = await openGemma4Ple(MIRROR, PRODUCT_CHOICE);
  assert(
    index.storage === "bf16",
    `the bf16 preset's PLE index is ${index.storage} (schema 5 bf16 is read here)`,
  );
  return index;
};

/**
 * The minimum number of distinct rows that switches to a whole-block read (a copy of
 * `FULL_BLOCK_ROWS` in `src/gemma/ple.ts`).
 *
 * The implementation does not export it, so this is **a copy of the policy table**. If they drift,
 * the expectations below stop matching the measurement and fail, so a different policy is never
 * accepted silently.
 */
const FULL_BLOCK_ROWS = 32;

/** The policy for one table (what {@link planTable} returns). */
type TableExpectation = {
  /** The block indices the probe touched. */
  readonly touched: readonly number[];
  /** The block indices that switch to a whole-block read (distinct rows reached the minimum). */
  readonly full: readonly number[];
  /** The number of ranges read row by row (the sum of distinct rows of blocks not read whole). */
  readonly rowReads: number;
};

/**
 * Counts, **per table**, where the policy table of `src/gemma/ple.ts` sends the index and the ids
 * looked up.
 *
 * MUST: count values and scales separately. The block boundaries of the two tables are
 * independent (one row = 8,960 B of values / 140 B of scales, so under the same block limit the
 * number of tokens a block spans differs by orders of magnitude), and the **mix** "values read by
 * row, scales read whole because 32 or more rows land on the same block" is the default state of
 * the real asset.
 */
const planTable = (table: Gemma4PleTable, ids: readonly number[]): TableExpectation => {
  const grouped = new Map<number, Set<number>>();
  for (const id of ids) {
    const position = gemma4PleBlockOf(table, id);
    const rows = grouped.get(position);
    if (rows === undefined) grouped.set(position, new Set([id]));
    else rows.add(id);
  }
  const full: number[] = [];
  let rowReads = 0;
  for (const [position, rows] of grouped) {
    const block = table.blocks[position];
    // The minimum is capped by the block's row count (for a narrow block, all its rows).
    if (rows.size >= Math.min(FULL_BLOCK_ROWS, block.stop - block.start)) full.push(position);
    else rowReads += rows.size;
  }
  return { touched: [...grouped.keys()], full, rowReads };
};

/** The bytes taken by the blocks read whole (to check that they do not hit the budget cap). */
const fullBytesOf = (table: Gemma4PleTable, plan: TableExpectation): number =>
  plan.full.reduce((sum, position) => sum + gemma4PleBlockBytes(table, table.blocks[position]), 0);

/**
 * {@link planTable} over every table the index has (`values`, then `scales` for a quantized
 * index — a bf16 index has `values` alone), summed the way the loader's stats count the tables
 * together.
 */
const planTables = (
  index: Gemma4PleIndex,
  ids: readonly number[],
): { readonly full: number; readonly rowReads: number; readonly fullBytes: number } =>
  gemma4PleTables(index).reduce(
    (sum, [, table]) => {
      const plan = planTable(table, ids);
      return {
        full: sum.full + plan.full.length,
        rowReads: sum.rowReads + plan.rowReads,
        fullBytes: sum.fullBytes + fullBytesOf(table, plan),
      };
    },
    { full: 0, rowReads: 0, fullBytes: 0 },
  );

/**
 * ② Bit identity of the PLE dequantization (ADR 0085 decision 4) + lazy load / LRU (decision 3).
 *
 * The reference is `ple.probe.safetensors`, which the script took with **torch on the 35-table
 * path** — had the PLE stayed in the graph, `embedding` + the `mul` right after it would have
 * produced those values. The host side builds `f32(i8) × per-row scale` → `× embedScale` in the
 * same order (for the bf16 index read here: the exact widening of the bf16 bits, then
 * `× embedScale`), so **not a single bit may differ**. If this splits, ③'s token-sequence parity
 * splits too (but ③ cannot tell where the cause is — this is the line that isolates it).
 *
 * No GPU needed (the loader is closed over Web standard APIs only).
 */
test.skipIf(!MODEL_PRESENT || PROBE_GATE.skip)(
  `Gemma 4 E2B product acceptance: the PLE dequantization is bit-identical to the in-graph embedding (no GPU needed)${PROBE_GATE.title}`,
  async () => {
    assertGoldensPresent([PLE_PROBE]);
    const index = await readPleIndex();
    assertEquals(index.tokens, VOCAB, "the index's token rows (= the main embedding's vocab)");
    assertEquals(index.layers, LAYERS, "the index's layer count");
    assertEquals(index.dim, PLE_DIM, "the index's per-layer dimension");
    assertEquals(index.embedScale, Math.sqrt(PLE_DIM), "embed scale（hidden_per_layer ** 0.5）");
    assert(
      index.values.blocks.length > 1,
      `the PLE values have ${index.values.blocks.length} block (no split by vocab range)`,
    );

    const probeFile = parseSafetensors(await readBuffer(PRODUCT_ROOT, PLE_PROBE_FILE));
    const tokens = [...goldenI32(probeFile, PROBE_TOKENS_KEY)];
    const expected = goldenF32(probeFile, PROBE_INPUTS_KEY);
    assertEquals(
      expected.length,
      tokens.length * LAYERS * PLE_DIM,
      "element count of the reference per_layer_inputs",
    );
    // The probe touches both sides of the block boundaries, so it touches every block (otherwise
    // the policy-table observation would check nothing).
    const blockOf = (token: number): number =>
      index.values.blocks.findIndex((block) => token < block.stop);
    const touched = new Set(tokens.map(blockOf));
    assertEquals(
      touched.size,
      index.values.blocks.length,
      "number of values blocks the probe touches (it touches every block)",
    );

    // Looked up with the default budget (= two of the largest blocks). Checks that the cap is in
    // bytes, not in counts.
    const budget = defaultGemma4PleResidentBytes(index);
    /** Exact match with torch's 35-table path (no tolerance). */
    const assertProbeMatch = (data: Float32Array, where: string): void => {
      let mismatches = 0;
      let first = "";
      for (let element = 0; element < expected.length; element += 1) {
        if (data[element] === expected[element]) continue;
        mismatches += 1;
        if (first !== "") continue;
        const token = tokens[Math.floor(element / (LAYERS * PLE_DIM))];
        const layer = Math.floor(element / PLE_DIM) % LAYERS;
        first =
          `token ${token} / layer ${layer} / column ${element % PLE_DIM}: ` +
          `${data[element]} ≠ ${expected[element]}`;
      }
      assertEquals(
        mismatches,
        0,
        `${where}: the PLE dequantization differs from torch's 35-table path` +
          ` (${mismatches} elements / first disagreement ${first})`,
      );
    };

    // ②-a The row-read path — the policy table switches **per table** (the block boundaries of
    // the two tables are independent). Values have a few rows per block, so every block is read
    // by row. One scales block covers over a hundred thousand tokens, so most of the probe lands
    // on the same block, and only that block reaches the minimum (32 rows) and is read whole.
    // The expectations are computed from the index and the probe's ids (written as constant
    // counts, they would change meaning with the asset generation).
    // A bf16 index has no scale table, so its row path reads every probe row by range.
    const distinctIds = [...new Set(tokens)];
    const distinct = distinctIds.length;
    const valuePlan = planTable(index.values, distinctIds);
    const plans = planTables(index, distinctIds);
    const rows = await openPle(budget);
    const byRows = await rows.gather(tokens);
    assertEquals(byRows.dtype, "f32", "gather dtype");
    assertEquals(byRows.shape, [1, tokens.length, LAYERS, PLE_DIM], "gather shape");
    assert("data" in byRows && byRows.data instanceof Float32Array);
    assertProbeMatch(byRows.data, "row-read path");
    const rowStats = rows.stats();
    // Premise of the expectations: what is read whole fits in the default budget (if it
    // overflowed, the policy table would give up and go back to row reads).
    const plannedFullBytes = plans.fullBytes;
    assert(
      plannedFullBytes <= budget,
      `the ${plannedFullBytes} bytes read whole exceed the default budget of ${budget} bytes` +
        ` (this check's expectations assume "no giving up because of the budget")`,
    );
    assertEquals(
      valuePlan.full.length,
      0,
      "values touch only a few rows per block yet are read whole (the row-read path is gone)",
    );
    assertEquals(
      rowStats.loads,
      plans.full,
      "whole-read blocks (values 0; a scale table, where the index has one, adds its blocks with" +
        " 32 or more probe rows)",
    );
    assertEquals(
      rowStats.rowReads,
      plans.rowReads,
      "number of row-read ranges (per table, the sum of distinct rows of blocks not read whole)",
    );
    assertEquals(
      rowStats.resident,
      plans.full,
      "number of resident blocks (only those read whole stay resident)",
    );

    // ②-b The whole-read path — looking up the probe's tokens together with the first 32 rows
    // of their blocks makes every touched block reach the minimum (32 rows), so the policy table
    // reads them whole. The probe sits at the front of the sequence, so only the first part of
    // the same element count is compared. The check is that the values **do not move by a single
    // bit**.
    const filler = [...touched].flatMap((position) =>
      Array.from({ length: 32 }, (_row, row) => index.values.blocks[position].start + row),
    );
    const full = await openPle(allResidentBytes(index));
    const byFull = await full.gather([...tokens, ...filler]);
    assert("data" in byFull && byFull.data instanceof Float32Array);
    assertProbeMatch(byFull.data.subarray(0, expected.length), "whole-read path");
    assertEquals(
      [...new Uint32Array(byFull.data.buffer, 0, expected.length)],
      [...new Uint32Array(byRows.data.buffer)],
      "the whole-read values are not bit-identical to the row reads",
    );
    const fullStats = full.stats();
    const combined = [...new Set([...tokens, ...filler])];
    const fullPlans = planTables(index, combined);
    assertEquals(
      fullPlans.rowReads,
      0,
      "some block stays below the minimum even with the filler (②-b's premise is broken)",
    );
    assertEquals(
      fullStats.loads,
      fullPlans.full,
      "whole-read blocks (the touched blocks of every table the index has)",
    );
    assertEquals(fullStats.rowReads, 0, "row reads happened in a gather that should read whole");

    console.log(
      `[e2e] gemma4 product PLE: probe ${tokens.length} token (distinct ${distinct}) × ${LAYERS} layers ×` +
        ` ${PLE_DIM} dims bit-identical to torch / row-read path = whole ${rowStats.loads},` +
        ` rows ${rowStats.rowReads} ranges (budget ${(budget / 1024 / 1024).toFixed(0)} MiB) / ` +
        `whole-read path = ${fullStats.loads} blocks loaded, ${fullStats.resident} resident` +
        `（${(fullStats.residentBytes / 1024 / 1024).toFixed(0)} MiB）`,
    );
  },
);

/**
 * ① Form premises. The differences from the two existing series (+2 inputs, the output is the
 * last-row logits, no argmax, no PLE table) are checked here — the main checks of states / layer
 * kinds / storage belong to the writing side (`export_product.assert_ir_form_product`). If this
 * loosened, **a different graph would be accepted**.
 */
const assertProductForm = (parsed: PreparedModel): void => {
  const graph = parsed.graph;
  assertEquals(
    // The order does not depend on the writer's declaration order (compared by name — what was
    // added are the 4 rope inputs, and `position_ids` is gone: positions come from the host as
    // "the tables themselves", not as "indices into the tables").
    [...graph.inputs.map((spec) => spec.name)].sort(),
    [INPUT_IDS, PER_LAYER_INPUTS, LAST_ROW, ...gemma4RopeInputNames()].sort(),
    "graph inputs (the product form has per_layer_inputs / 4 rope inputs / last_row)",
  );
  const perLayer = graph.inputs.find((spec) => spec.name === PER_LAYER_INPUTS);
  assert(perLayer !== undefined, `'${PER_LAYER_INPUTS}' is missing`);
  assertEquals(perLayer.dtype, "f32", `dtype of '${PER_LAYER_INPUTS}'`);
  assertEquals(
    perLayer.shape,
    [1, "M", LAYERS, PLE_DIM],
    `shape of '${PER_LAYER_INPUTS}' (the [1,M,35,256] the host supplies)`,
  );

  // Two outputs, and the order is the contract (output 0 = the selected row's logits, output 1 =
  // the same row's hidden after the final norm — ADR 0068 addendum 7 / 0096 decision 5). The row
  // axis is the symbol R bound by `last_row[R]`; a normal decode has R=1.
  assertEquals(graph.outputs.length, 2, "number of graph.outputs (logits + hidden)");
  assertEquals(
    graph.values[graph.outputs[0]].shape,
    [1, ROW_SYMBOL, VOCAB],
    "shape of output 0 (the selected **rows** only — detects a regression to all-row logits)",
  );
  assertEquals(
    graph.values[graph.outputs[1]].shape,
    [1, ROW_SYMBOL, HIDDEN],
    "shape of output 1 (the selected rows' hidden after the final norm — not swapped with the logits)",
  );
  const lastRow = graph.inputs.find((spec) => spec.name === LAST_ROW);
  assert(lastRow !== undefined, `'${LAST_ROW}' is missing`);
  assertEquals(
    lastRow.shape,
    [ROW_SYMBOL],
    `shape of '${LAST_ROW}' (the only binding point of the symbol R)`,
  );
  const producer = new Map<string, (typeof graph.nodes)[number]>();
  for (const node of graph.nodes) {
    for (const out of node.outs) producer.set(out, node);
  }
  const exit = producer.get(graph.outputs[0]);
  assert(exit !== undefined, "output 0 is not a node output");
  assert(
    exit.op !== "argmax",
    "output 0 comes from argmax (the product exit is logits, and sampling is on the host — ADR 0083 decision 6)",
  );
  assertEquals(
    graph.nodes.filter((node) => node.op === "argmax").length,
    0,
    "number of argmax nodes (none in the graph)",
  );

  // The PLE table is not left in the graph (the form that would bring back 2,240 MiB resident).
  // The embedding's weight slot is ins[0] (`WEIGHT_SLOTS[embedding] = 0`).
  const embeddings = graph.nodes.filter((node) => node.op === "embedding");
  assertEquals(
    embeddings.filter((node) => {
      const shape = graph.values[node.ins[0]].shape;
      return shape.length === 2 && shape[0] === VOCAB && shape[1] === PLE_DIM;
    }).length,
    0,
    `an embedding that looks up the PLE table [${VOCAB},${PLE_DIM}] remains (not fully moved out to the host gather)`,
  );
  // The main embedding + the last-row selection (RoPE's cos / sin are derived inputs the host
  // supplies, not table lookups — ADR 0091 — so there is no gather for them).
  assertEquals(
    embeddings.length,
    2,
    "number of embedding nodes (the 35 PLE ones and the 4 RoPE tables are gone)",
  );

  assertEquals(
    Object.keys(graph.states).length,
    SLOTS,
    "number of states slots (identical to the two existing series)",
  );
  // Three symbols (chunk rows M / selected rows R / full-slot capacity C).
  assertEquals([...graph.symbols].sort(), ["C", "M", ROW_SYMBOL], "graph.symbols");
};

/** One i32 input tensor (both the token ids and the absolute positions are `[1, rows]`). */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/**
 * The row-selection input (an i32 of `[1]` — the shape the token-only exit, removed on 2026-10-17,
 * also used).
 */
const lastRowInput = (row: number): Tensor => ({
  dtype: "i32",
  shape: [1],
  data: Int32Array.of(row),
});

/**
 * Reads the next token from the last-row logits `[1,1,V]` with a host-side argmax.
 *
 * MUST: on ties take **the first one** (update on `>`) — the same tie-break as torch's `argmax`;
 * with `>=` the sequence would split on runs of equal values.
 * MUST: check the dtype and the shape (a mixed-up output name returns a different sequence with
 * neither an exception nor a warning).
 */
const argmaxToken = (tensor: Tensor, where: string): number => {
  assertEquals(tensor.dtype, "f32", `${where}: logits dtype`);
  assertEquals(tensor.shape, [1, 1, VOCAB], `${where}: logits shape`);
  const data = tensor.data;
  let best = 0;
  let bestValue = data[0];
  for (let index = 1; index < VOCAB; index += 1) {
    if (data[index] > bestValue) {
      bestValue = data[index];
      best = index;
    }
  }
  // A non-finite value (a gather outside the position table / a broken PLE) would fold into "a
  // plausible token id", so it fails here **before** folding.
  assert(Number.isFinite(bestValue), `${where}: the logits maximum is non-finite (${bestValue})`);
  return best;
};

/**
 * The measurements of one generation (fields that show, non-tautologically, that the lazy load
 * works).
 */
type GenerationRecord = { readonly tokens: number[]; readonly gathers: number };

/**
 * The greedy loop of host PLE gather + host argmax (a reduced form of the product surface).
 *
 * MUST: pad rows get the PLE of id 0 too, the same as `input_ids` — when the lookup was inside the
 * graph, pad rows looked up token 0's row, so zero-filling would make the pad rows' values
 * different (the value contract of ADR 0066 addendum 6 is "the host fills with 0", not "the PLE
 * is 0").
 */
const generate = async (
  session: Awaited<ReturnType<PreparedModel["createContainerSession"]>>,
  logitsName: string,
  ple: Gemma4Ple,
  prompt: readonly number[],
  maxNewTokens: number,
  chunkLength: number,
): Promise<GenerationRecord> => {
  const chunks = planPrefillChunks(prompt.length, chunkLength);
  const lastPosition = prompt.length + maxNewTokens - 2;
  assert(
    lastPosition < MAX_POSITION,
    `the last position ${lastPosition} is beyond the model's position limit`,
  );
  assert(prompt.length + maxNewTokens <= CAPACITY, `T + K exceeds the capacity ${CAPACITY}`);

  const context = await session.createGenerationContext({
    bindings: { [CAPACITY_SYMBOL]: CAPACITY },
    chunkLength,
  });
  let gathers = 0;
  try {
    let token = 0;
    for (const chunk of chunks) {
      const ids = new Int32Array(chunkLength);
      const positions = new Int32Array(chunkLength);
      for (let row = 0; row < chunk.queryLength; row += 1) {
        ids[row] = prompt[chunk.position + row];
        positions[row] = chunk.position + row;
      }
      const perLayer = await ple.gather([...ids]);
      gathers += 1;
      const outputs = await session.run(
        {
          [INPUT_IDS]: i32Row(chunkLength, ids),
          [PER_LAYER_INPUTS]: perLayer,
          ...ropeInputs(ROPE, positions),
          [LAST_ROW]: lastRowInput(chunk.queryLength - 1),
        },
        undefined,
        { context, queryLength: chunk.queryLength },
      );
      token = argmaxToken(outputs[logitsName], `prefill@${chunk.position}`);
    }

    const tokens = [token];
    for (let index = 0; index + 1 < maxNewTokens; index += 1) {
      const current = tokens[index];
      const perLayer = await ple.gather([current]);
      gathers += 1;
      const outputs = await session.run(
        {
          [INPUT_IDS]: i32Row(1, Int32Array.of(current)),
          [PER_LAYER_INPUTS]: perLayer,
          ...ropeInputs(ROPE, [prompt.length + index]),
          [LAST_ROW]: lastRowInput(0),
        },
        undefined,
        { context, queryLength: 1 },
      );
      tokens.push(argmaxToken(outputs[logitsName], `decode@${index}`));
    }
    return { tokens, gathers };
  } finally {
    // MUST: released even on a failure midway (do not hold on to the VRAM of the KV capacity).
    await context.dispose();
  }
};

/**
 * Default case: the graph form (①) and the residency ceiling (ADR 0085) are structure / gate
 * checks that do not compare against torch goldens, so they stay in the default run.
 */
describe.skipIf(!MODEL_PRESENT || !GPU_AVAILABLE)(
  "Gemma 4 E2B product acceptance: form premises and the residency ceiling (real GPU)",
  () => {
    let parsed: PreparedModel;
    beforeAll(async () => {
      const opened = await openProductContainer();
      parsed = prepareContainer(opened, MODEL_GRAPH);
    });

    it("① form premises: the PLE is moved out and the exit is the last-row logits", () => {
      assertProductForm(parsed);
    });

    // This section was the test body after the step (no step title of its own); it owns the GPU
    // session it opens, so the acquire / dispose pair stays inside this one case.
    it("residency ceiling: a container with the PLE moved out keeps half or less resident", async () => {
      const gpu = await acquireGpu();
      const session = await parsed.createContainerSession(gpu);
      try {
        const storage = session.diagnostics().storage;
        assert(storage !== undefined, "diagnostics.storage is missing");
        assertEquals(
          storage.hostExpandedBytes,
          0,
          "hostExpandedBytes (CPU expansion of ineligible weights)",
        );
        // ADR 0085's claim itself: a container with the PLE moved out keeps half or less resident.
        assert(
          storage.residentCompressedBytes < RESIDENT_CEILING,
          `residentCompressedBytes ${storage.residentCompressedBytes} is ${RESIDENT_CEILING} or more` +
            ` (reading a container with the PLE left in the graph — ADR 0085's residency reduction is not in effect)`,
        );
        console.log(
          `[e2e] gemma4 product residency: compressed ${storage.residentCompressedBytes} bytes` +
            ` (kept in the graph, the bf16 PLE would add 4.70e9 B)`,
        );
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    });
  },
);

/**
 * Opt-in case: ③ compares the host greedy loop with the torch greedy goldens (upstream parity).
 * The lazy-load assertions stay here because they observe this generation run; the GPU-free ②
 * case keeps the LRU policy check in the default run.
 */
describe.skipIf(!UPSTREAM_PARITY || !AVAILABLE || !GPU_AVAILABLE)(
  `Gemma 4 E2B product acceptance: cross parity (real GPU / host PLE + host argmax)${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    let logitsName: string;
    let index: Gemma4PleFloatIndex;
    let ple: Gemma4Ple;
    let gpu: GpuContext | undefined;
    let session: Awaited<ReturnType<PreparedModel["createContainerSession"]>> | undefined;
    beforeAll(async () => {
      const opened = await openProductContainer();
      const parsed = prepareContainer(opened, MODEL_GRAPH);
      logitsName = parsed.graph.outputs[0];

      index = await readPleIndex();
      // Every block resident (the LRU observation belongs to the GPU-free ② — this side only
      // checks that the generation round trips do not re-read blocks).
      ple = await openPle(allResidentBytes(index));
      gpu = await acquireGpu();
      session = await parsed.createContainerSession(gpu);
    });
    afterAll(async () => {
      await session?.dispose();
      gpu?.destroy();
    });

    it("③ 3 cases × 16 steps exactly match the opt-in series' expected sequences", async () => {
      assert(session !== undefined, "session was not created in beforeAll");
      let totalGathers = 0;
      for (const caseName of EXPECTED_CASES) {
        const golden = parseSafetensors(
          await readBuffer(GOLDEN_ROOT, `${GREEDY_PREFIX}${caseName}${SUFFIX}`),
        );
        const prompt = [...goldenI32(golden, "prompt")];
        const expected = [...goldenI32(golden, "expected")];
        assertEquals(expected.length, GREEDY_STEPS, `${caseName}: golden step count`);

        // Only the longest case runs with a finer chunk length (multi-chunk prefill = the only
        // path that runs a PLE gather across chunks on the real weights — the doc of
        // `LONG_CASE_CHUNK_LENGTH`).
        const chunkLength = caseName === LONG_CASE ? LONG_CASE_CHUNK_LENGTH : CHUNK_LENGTH;
        const chunks = planPrefillChunks(prompt.length, chunkLength).length;
        if (caseName === LONG_CASE) {
          assert(
            chunks > 1,
            `${caseName}: fits in one chunk at length ${chunkLength} (no gather across chunks runs)`,
          );
        }

        const started = performance.now();
        const generated = await generate(
          session,
          logitsName,
          ple,
          prompt,
          GREEDY_STEPS,
          chunkLength,
        );
        totalGathers += generated.gathers;
        assertEquals(
          generated.tokens,
          expected,
          `${caseName}: generated token sequence (vs the logits opt-in series golden)`,
        );
        console.log(
          `[e2e] gemma4 product ${caseName}: T=${prompt.length} (chunk length ${chunkLength} = ` +
            `${chunks} chunks) + K=${GREEDY_STEPS} steps / PLE gather ${generated.gathers} times / ` +
            `${(performance.now() - started).toFixed(0)}ms`,
        );
      }
      // The lazy load is in effect (not a tautology): the gather runs dozens of times, yet the
      // blocks are fetched at most once each. An implementation that re-reads every time would
      // grow this up to the gather count.
      const stats = ple.stats();
      assert(
        totalGathers > stats.loads,
        `${totalGathers} PLE gathers but ${stats.loads} block fetches` +
          ` (the cache is not working)`,
      );
      const blocks = gemma4PleTables(index).reduce(
        (sum, [, table]) => sum + table.blocks.length,
        0,
      );
      assert(
        stats.loads <= blocks,
        `${stats.loads} block fetches exceed the ${blocks} blocks (re-reads are happening)`,
      );
      console.log(
        `[e2e] gemma4 product PLE lazy load: ${totalGathers} gathers / ` +
          `${stats.loads} block fetches (of ${blocks} blocks)`,
      );
    });
  },
);
