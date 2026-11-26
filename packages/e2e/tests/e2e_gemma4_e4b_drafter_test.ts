// Implementation-time checks of regular Gemma 4 E4B's **MTP drafter** (`google/gemma-4-E4B-it-assistant`
// — ADR [0096](../../../docs/decisions/0096-speculative-decoding.md)) before it is published: the
// drafter series and the next `karume-gemma4` staged from the series (ADR 0139 decision 6 — a
// graph that is not shipped yet, so every case runs only with KARUME_SERIES=1).
//
// E4B's drafter is E2B's graph at E4B's widths (backbone 2560, two KV heads read from the target's
// l22 / l23). The cases, on the staged distribution's E4B entry:
//
// ① the manifest entry (no GPU): weights `model` + `drafter`, the drafter int8 alone, both seats
//    (`bf16`, `bf16-fast`) naming both roles; the staged drafter parts are the series' bytes, and
//    the staged E2B entry is the published mirror's (the next distribution changes E4B only)
// ② the drafter container (no GPU): the admission the pipeline runs pairs it with its target —
//    the four external slots are the target's l22 / l23 with the target's shapes, the one shared
//    initializer is the target's `model.lm_head.weight`, the `hidden` input is 2560 wide — and the
//    series record names the pinned assistant and the lender's checkpoint
// ③ speculation per seat (real GPU): a greedy chat turn of the gemma4 reference prompt with
//    `speculative: "always"` gives the same token ids, stop and token count as the plain turn, with
//    drafts accepted; the speculative ids are pinned as a sha256 row per seat (`e4b/<seat>+drafter`
//    in `fixtures/references/gemma4.json` — ADR 0106), and the plain ids are the ones the seat's
//    own row (`e4b/<seat>`, `e2e_gemma4_reference_test.ts`) pins
// ④ identity on the reference seat (real GPU): 200 tokens from each drafter golden's prompt,
//    speculative (`"always"`) and plain, give the same id and position lists; their common prefix
//    with the torch continuation reaches a floor, and the tokens per cycle reach theirs
// ⑤ draft parity (real GPU, upstream parity — opt-in KARUME_UPSTREAM_PARITY=1 like
//    `e2e_gemma4_drafter_test.ts`): the drafts, teacher-forced along the golden continuation
//    (3 cases × 200 cycles × 3 steps), match the torch drafter's in at least 99% of the positions
//    (E2B's floor — MUST NOT be moved to the measurement), and are not stuck
//
// ## Assets
//
// - the drafter series `outputs/series/gemma4-e4b-drafter/` (`gemma4.export_drafter --model e4b
//   --device cpu`): its goldens and `reference.json`;
// - the lender series `outputs/series/gemma4-e4b-product-bf16/` (its `reference.json` only — ②);
// - the next `karume-gemma4`, staged from the series outside `models/`:
//   `cd tools/export-recipes && uv run python dist.py --pipeline gemma4 --model e2b --model e4b
//   --out ../../outputs/misc/local-dist/karume-gemma4` (the published mirror `models/karume-gemma4/`
//   is what Hugging Face holds and is never rebuilt for this).
//
// Without KARUME_SERIES=1 every case SKIPs explicitly and nothing under `outputs/` is read. With
// it, an absent asset SKIPs with the command that makes it.

import { isNotFound, readFile, readTextFileSync, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { type Manifest, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import {
  acquireGpu,
  type GenerationContext,
  parseSafetensors,
  type SafetensorsFile,
  type Session,
  type Tensor,
} from "@karume/runtime";
import {
  type ChatStop,
  type GenerationStop,
  Pipeline,
  type PipelineConfig,
} from "../../models/gemma.ts";
import { parsePipelineConfig } from "../../models/src/gemma/config.ts";
import { createGemma4Ple, type Gemma4Ple } from "../../models/src/gemma/ple.ts";
import { gemma4RopeInputNames, ropeInputs } from "../../models/src/gemma/rope.ts";
import {
  admitGemma4Drafter,
  draftOnce,
  GEMMA4_DRAFT_STEPS,
  type Gemma4Drafter,
  openDrafterContext,
} from "../../models/src/gemma/speculative.ts";
import { planPrefillChunks } from "../../models/src/generation/greedy.ts";
import { openGemma4Ple } from "../../models/tests/helpers/gemma-mirror.ts";
import {
  REFERENCE_MAX_NEW_TOKENS,
  REFERENCE_PROMPT,
  seatArtifactBytes,
} from "../../models/tests/helpers/gemma-reference.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  mirrorDirectory,
  mirrorHasModel,
  mirrorRoot,
  openMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import {
  allResidentBytes,
  allResidentPleBytesOfMirror,
} from "../../models/tests/helpers/ple-budget.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { REORDER_STRATEGIES } from "../../runtime/src/runtime/executor.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import {
  openReferences,
  referenceEntryFields,
  referenceMismatchMessage,
  type ReferenceSettlement,
  registerReferenceGate,
  settleReference,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

const SERIES = new URL("../../../outputs/series/", import.meta.url);
const DRAFTER_SERIES = new URL("gemma4-e4b-drafter/", SERIES);
const LENDER_SERIES = new URL("gemma4-e4b-product-bf16/", SERIES);
/** The next `karume-gemma4`, staged from the series (module comment). */
const STAGED = new URL("../../../outputs/misc/local-dist/karume-gemma4/", import.meta.url);
/** The published mirror (read only, for ①'s comparison of the E2B entry). */
const PUBLISHED = mirrorRoot("karume-gemma4");

const EXPORT_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m" +
  " gemma4.export_drafter --model e4b --device cpu";
const STAGE_COMMAND =
  "cd tools/export-recipes && uv run python dist.py --pipeline gemma4 --model e2b --model e4b" +
  " --out ../../outputs/misc/local-dist/karume-gemma4";

const MODEL_NAME = "e4b";
/** The roles of the manifest (`gemma4/distribution.py`). */
const MODEL = "model";
const DRAFTER = "drafter";
/** The seats of the regular line (`GEMMA4_QUANTS` / `GEMMA4_PARALLEL_QUANTS`). */
const SEATS = ["bf16", "bf16-fast"] as const;
type Seat = (typeof SEATS)[number];
const SEAT_SESSIONS: Record<Seat, Readonly<Record<string, unknown>>> = {
  bf16: {},
  "bf16-fast": { numerics: { level: "reorder" } },
};
/** The reference seat (session {} — decode M=1 and verify M=4 run the same reduction kernels). */
const REFERENCE_SEAT: Seat = "bf16";

/** E4B's widths: the target's hidden, and the slots its drafter borrows (l22 sliding / l23 full). */
const HIDDEN = 2560;
const VOCAB = 262144;
const CAPACITY_SYMBOL = "C";
const SLOTS: Record<string, readonly (number | string)[]> = {
  "l22.k": [1, 2, 520, 256],
  "l22.v": [1, 2, 520, 256],
  "l23.k": [1, 2, CAPACITY_SYMBOL, 512],
  "l23.v": [1, 2, CAPACITY_SYMBOL, 512],
};
/** The table the drafter borrows (its name is the lender's initializer name — docs/ir-v2.md). */
const SHARED_WEIGHT = "model.lm_head.weight";
/** The pinned assistant (`gemma4.sources.REGULAR_DRAFTER_PINS["e4b"]`). */
const ASSISTANT = {
  repo: "google/gemma-4-E4B-it-assistant",
  revision: "8d0031ea8c2109e2b1e86bb9368a4539b537f80a",
} as const;
/** The target files the drafter record and the lender record must agree on (`export_drafter`). */
const LENDER_CHECKPOINT_FILES = ["model.safetensors", "config.json"] as const;

/** Graph inputs of the product (`gemma4/export_product.py`). */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";

/** The drafter golden cases (`gemma4/export_drafter.py`). */
const CASES = ["short-en", "readme-recipes", "readme-exporter"] as const;
type CaseName = (typeof CASES)[number];
const GOLDEN_PREFIX = "drafter-golden.";
const SUFFIX = ".safetensors";
const REFERENCE_FILE = "reference.json";

/** The draft-match floor (`e2e_gemma4_drafter_test.ts` — the denominator is cases × cycles × k). */
const MATCH_FLOOR = 0.99;
/** ④'s generated tokens per case (the length of the speculative test's ①). */
const MAX_NEW_TOKENS = 200;
/**
 * ④'s floor on the common prefix with the torch continuation (`e2e_gemma4_speculative_test.ts`'s
 * `GOLDEN_PREFIX_FLOOR`): it catches only the "switched to another list" class.
 */
const GOLDEN_PREFIX_FLOOR = 16;
/**
 * ④'s floor on the tokens committed per cycle (`delivered / cycles`) on the two README cases.
 *
 * MUST: do not fit it to a later measurement. Placed by the rule of
 * `e2e_gemma4_speculative_test.ts`'s `ACCEPTANCE_FLOOR` — below the smaller measurement when the
 * drafter landed, rounded down to one decimal: 2.314 / 2.314 tokens per cycle (readme-recipes /
 * readme-exporter, RTX 3080 Ti, Node + Dawn, 2026-11-26) give 2.2. `short-en` (25 prompt tokens,
 * 1.567) varies too much for one case and is only logged. A measurement on the RTX 3080 Ti / Vulkan
 * backend: on another GPU the accepted lists can shift (an environment difference, not a
 * regression).
 */
const ACCEPTANCE_FLOOR: Partial<Record<CaseName, number>> = {
  "readme-recipes": 2.2,
  "readme-exporter": 2.2,
};

const isFile = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const goldenUrl = (name: string): URL =>
  new URL(`${GOLDEN_PREFIX}${name}${SUFFIX}`, DRAFTER_SERIES);

/** Every asset is present (read only with KARUME_SERIES=1 — the default lanes touch nothing). */
const SERIES_PRESENT =
  SERIES_CHECKS &&
  isFile(new URL(REFERENCE_FILE, DRAFTER_SERIES)) &&
  CASES.every((name) => isFile(goldenUrl(name)));
const STAGED_PRESENT =
  SERIES_CHECKS &&
  mirrorHasModel(STAGED, MODEL_NAME) &&
  Object.hasOwn(
    parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, STAGED))).models[MODEL_NAME].weights,
    DRAFTER,
  );
const PRESENT = SERIES_PRESENT && STAGED_PRESENT;
if (SERIES_CHECKS && !SERIES_PRESENT) {
  console.warn(
    `[karume] ${DRAFTER_SERIES.pathname} is absent or incomplete, so the E4B drafter cases SKIP.` +
      ` Export it: ${EXPORT_COMMAND}`,
  );
}
if (SERIES_CHECKS && !STAGED_PRESENT) {
  console.warn(
    `[karume] ${STAGED.pathname} (the staged next karume-gemma4 with E4B's drafter) is absent, so` +
      ` the E4B drafter cases SKIP. Stage it: ${STAGE_COMMAND}`,
  );
}

const manifest: Manifest | undefined = STAGED_PRESENT
  ? parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, STAGED)))
  : undefined;
const entry = () => {
  assert(manifest !== undefined, "the staged manifest is not read");
  return manifest.models[MODEL_NAME];
};
const configOf = (): PipelineConfig => parsePipelineConfig(entry().pipelineConfig);

type ObjectRecord = Record<string, unknown>;
const objectAt = (value: unknown, where: string): ObjectRecord => {
  assert(
    typeof value === "object" && value !== null && !Array.isArray(value),
    `${where} is not an object`,
  );
  return value as ObjectRecord;
};
const readRecord = (root: URL): ObjectRecord =>
  objectAt(JSON.parse(readTextFileSync(new URL(REFERENCE_FILE, root))), `${root.pathname}`);

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const bufferOf = (bytes: Uint8Array<ArrayBuffer>): ArrayBuffer =>
  bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

// ---------------------------------------------------------------------------
// ① ② the manifest entry and the drafter container (no GPU)
// ---------------------------------------------------------------------------

test.skipIf(!PRESENT)(
  `gemma4 e4b drafter: the staged entry ships the int8 drafter under both seats, from the series${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const weights = entry().weights;
    assertEquals(Object.keys(weights).sort(), [DRAFTER, MODEL], "weights roles");
    assertEquals(Object.keys(weights[DRAFTER]), ["i8"], "the drafter: int8 alone");
    assertEquals(Object.keys(entry().quants), [...SEATS], "seats");
    assertEquals(entry().defaultQuant, REFERENCE_SEAT, "default seat");
    for (const seat of SEATS) {
      const quant = entry().quants[seat];
      assertEquals(quant.weights, { [MODEL]: "bf16", [DRAFTER]: "i8" }, `${seat}: weights`);
      assertEquals(quant.session ?? {}, SEAT_SESSIONS[seat], `${seat}: session`);
    }
    // The staged drafter parts are the series' container parts, byte for byte (`model-N-of-M.krm`
    // in the series, renamed by the distribution's layout).
    const parts = weights[DRAFTER]["i8"].container.parts;
    for (const [index, part] of parts.entries()) {
      const name = `model-${String(index + 1).padStart(5, "0")}-of-${String(parts.length).padStart(5, "0")}.krm`;
      const bytes = await readFile(new URL(name, DRAFTER_SERIES));
      assertEquals(await sha256Hex(bytes), part.sha256, `${part.path} against the series' ${name}`);
    }
    // The next distribution changes E4B only: its E2B entry is the published mirror's. A mirror
    // without E2B leaves nothing to compare against, so the case fails rather than pass unchecked.
    assert(
      mirrorHasModel(PUBLISHED, "e2b"),
      `the published mirror ${PUBLISHED.pathname} has no e2b entry to compare the staged one with`,
    );
    const published = parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, PUBLISHED)));
    assert(manifest !== undefined, "the staged manifest is not read");
    assertEquals(manifest.models["e2b"], published.models["e2b"], "the E2B entry");
  },
);

test.skipIf(!PRESENT)(
  `gemma4 e4b drafter: the drafter pairs with its target — l22 / l23, shared table, hidden 2560 (no GPU)${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const target = (await openMirrorComponent(STAGED, MODEL, { model: MODEL_NAME })).graph;
    const drafter = (await openMirrorComponent(STAGED, DRAFTER, { model: MODEL_NAME })).graph;
    assertEquals(
      drafter.inputs.map((input) => input.name),
      ["token", "hidden", ...gemma4RopeInputNames()],
      "drafter inputs (order included)",
    );
    assertEquals(drafter.outputs.length, GEMMA4_DRAFT_STEPS, "draft exits");
    assertEquals(drafter.symbols, [CAPACITY_SYMBOL], "the one symbol, inherited from the lender");
    assertEquals(Object.keys(drafter.states).sort(), Object.keys(SLOTS).sort(), "borrowed slots");
    for (const [name, shape] of Object.entries(SLOTS)) {
      assert(drafter.states[name].external === true, `${name}: not external`);
      assertEquals(drafter.states[name].shape, shape, `${name}: the drafter's shape`);
      assertEquals(target.states[name].shape, shape, `${name}: the target's shape`);
    }
    const admitted = admitGemma4Drafter("test", drafter, {
      graph: target,
      rope: configOf().rope,
      hiddenSize: HIDDEN,
      capacitySymbol: CAPACITY_SYMBOL,
    });
    assertEquals(admitted.hiddenSize, HIDDEN, "hidden width");
    assertEquals(admitted.sharedWeights, { [SHARED_WEIGHT]: SHARED_WEIGHT }, "shared weights");
    assertEquals(drafter.initializers[SHARED_WEIGHT].storage, undefined, "no bytes borrowed");

    // The record: the pinned assistant, and the target checkpoint the lender was exported from.
    const record = readRecord(DRAFTER_SERIES);
    const assistant = objectAt(record.drafter, "drafter");
    assertEquals(
      [assistant.repo, assistant.revision],
      [ASSISTANT.repo, ASSISTANT.revision],
      "the assistant",
    );
    // Without the lender's record there is no checkpoint to compare the goldens' target with, so
    // the case fails rather than pass unchecked.
    assert(
      isFile(new URL(REFERENCE_FILE, LENDER_SERIES)),
      `the lender series' ${REFERENCE_FILE} is missing (${new URL(REFERENCE_FILE, LENDER_SERIES).pathname}) — the goldens' target checkpoint cannot be compared with the lender's`,
    );
    const drafterFiles = objectAt(objectAt(record.checkpoint, "checkpoint").files, "files");
    const lenderFiles = objectAt(
      objectAt(readRecord(LENDER_SERIES).checkpoint, "checkpoint").files,
      "files",
    );
    for (const name of LENDER_CHECKPOINT_FILES) {
      assertEquals(
        objectAt(drafterFiles[name], name).sha256,
        objectAt(lenderFiles[name], name).sha256,
        `${name}: the goldens' target is the lender's checkpoint`,
      );
    }
  },
);

// ---------------------------------------------------------------------------
// ③ speculation per seat, with its sha256 row (real GPU)
// ---------------------------------------------------------------------------

/** sha256 rows of the token ids (rows per environment key — `runtime/tests/helpers/reference.ts`). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/gemma4.json", import.meta.url),
);
/** Artifacts and settlements (`outputs/verify/<environment key>/<date>_gemma4-e4b-drafter/`). */
const results = openResults("gemma4-e4b-drafter");
const RUNNABLE = PRESENT && GPU_AVAILABLE;
/** The row of a seat's speculative ids (`<model>/<seat>+drafter`). */
const rowId = (seat: Seat): string => `${MODEL_NAME}/${seat}+drafter`;

/** One greedy chat turn of the reference prompt, its ids collected (stop token not included). */
const chatTurn = async (
  pipeline: Pipeline,
  speculative: "always" | false,
): Promise<{ ids: number[]; text: string; stop: ChatStop; ms: number }> => {
  const ids: number[] = [];
  const started = performance.now();
  const stream = pipeline.chat([{ role: "user", content: REFERENCE_PROMPT }], {
    maxNewTokens: REFERENCE_MAX_NEW_TOKENS,
    sampler: { temperature: 0 },
    speculative,
    onToken: (id) => ids.push(id),
  });
  const text = await stream.text();
  return { ids, text, stop: await stream.done, ms: performance.now() - started };
};

/** Opens the staged E4B with its drafter on `seat`, the whole PLE resident on the host. */
const openSeat = async (seat: Seat): Promise<Pipeline> =>
  await Pipeline.fromPretrained(mirrorDirectory(STAGED), {
    model: MODEL_NAME,
    quant: seat,
    speculative: { k: GEMMA4_DRAFT_STEPS },
    maxResidentPleBytes: await allResidentPleBytesOfMirror(STAGED, {
      model: MODEL_NAME,
      quant: seat,
    }),
  });

/** One golden case's prompt and torch continuation. */
const readGolden = async (
  name: string,
): Promise<{
  prompt: Int32Array<ArrayBuffer>;
  tokens: Int32Array<ArrayBuffer>;
  draft: Int32Array<ArrayBuffer>;
  cycles: number;
}> => {
  const file: SafetensorsFile = parseSafetensors(bufferOf(await readFile(goldenUrl(name))));
  const i32 = (key: string): Int32Array<ArrayBuffer> => {
    const view = file.tensors.get(key);
    assert(view !== undefined && view.dtype === "I32", `${name}: golden '${key}' is not I32`);
    return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
  };
  const draft = file.tensors.get("draft");
  assert(draft !== undefined, `${name}: no 'draft'`);
  assertEquals(draft.shape[1], GEMMA4_DRAFT_STEPS, `${name}: draft steps`);
  const tokens = i32("tokens");
  assertEquals(draft.shape[0] + GEMMA4_DRAFT_STEPS, tokens.length, `${name}: cycles`);
  return { prompt: i32("prompt"), tokens, draft: i32("draft"), cycles: draft.shape[0] };
};

/** The common prefix length of two lists. */
const commonPrefix = (left: readonly number[], right: readonly number[]): number => {
  let index = 0;
  while (index < left.length && index < right.length && left[index] === right[index]) index += 1;
  return index;
};

/** One turn of `tokens` ids from `prompt` (ids and absolute positions). */
const runTurn = async (
  pipeline: Pipeline,
  prompt: readonly number[],
  speculative: "always" | false,
): Promise<{ ids: number[]; positions: number[]; stop: GenerationStop; ms: number }> => {
  const capacity = Math.max(pipeline.program.chunkLength, prompt.length + MAX_NEW_TOKENS);
  const sequence = await pipeline.sequence({ speculative, capacity });
  try {
    const started = performance.now();
    const stream = sequence.generate({ prompt: [...prompt], maxNewTokens: MAX_NEW_TOKENS });
    const ids: number[] = [];
    const positions: number[] = [];
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      ids.push(event.id);
      positions.push(event.position);
    }
    return { ids, positions, stop: await stream.done, ms: performance.now() - started };
  } finally {
    await sequence.dispose();
  }
};

for (const seat of SEATS) {
  describe.skipIf(!RUNNABLE)(
    `gemma4 e4b drafter ${seat}: speculation through the staged distribution (real GPU)${SERIES_CHECKS_OPT_IN}`,
    () => {
      let pipeline: Pipeline | undefined;
      beforeAll(async () => {
        const started = performance.now();
        pipeline = await openSeat(seat);
        console.log(
          `[e2e] gemma4 e4b drafter ${seat}: load with the drafter` +
            ` ${(performance.now() - started).toFixed(0)} ms`,
        );
      });
      afterAll(async () => {
        await pipeline?.dispose();
      });

      it(`③ ${seat}: the speculative chat turn equals the plain one, and its ids match this machine's row`, async () => {
        assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
        const opened = pipeline;
        const id = rowId(seat);
        // Without a row (and outside a writing mode) the sha settlement is left out — the
        // reference gate below is red for it — and the identity still runs.
        const settles = !references.lacksReference(id);
        let settlement: ReferenceSettlement | undefined;
        await runRecordedCase(
          results,
          {
            id,
            failureNote: (cause) => (cause instanceof Error ? cause.message : String(cause)),
          },
          async () => {
            await assertRunningAdapter();
            const speculative = await chatTurn(opened, "always");
            const plain = await chatTurn(opened, false);
            // Speculation's one contract: it never changes the output.
            assertEquals(speculative.ids, plain.ids, "the token ids");
            assertEquals(speculative.text, plain.text, "the text");
            assertEquals(speculative.stop.reason, plain.stop.reason, "the stop reason");
            assertEquals(speculative.stop.tokens, plain.stop.tokens, "the token count");
            assertEquals(plain.stop.speculation, undefined, "a plain turn carries no speculation");
            const speculation = speculative.stop.speculation;
            assert(speculation !== undefined, "the speculative turn did not speculate");
            assert(speculation.cycles > 0, "no verify run");
            assert(
              speculation.accepted > 0,
              `no draft accepted (${speculation.cycles} cycles, ${speculation.drafted} drafted)`,
            );
            // The row's strength: the turn ran to the limit (a turn closed early by EOS pins
            // fewer ids).
            assert(
              speculative.stop.reason === "max-tokens",
              `${id}: the turn closed after ${speculative.ids.length} tokens`,
            );
            const note =
              `${speculative.stop.tokens} tokens / ${speculation.cycles} cycles / accepted` +
              ` ${speculation.accepted}/${speculation.drafted} /` +
              ` ${(speculation.delivered / speculation.cycles).toFixed(3)} tokens per cycle /` +
              ` speculative ${speculative.ms.toFixed(0)} ms vs plain ${plain.ms.toFixed(0)} ms`;
            console.log(`[e2e] gemma4 e4b drafter ${seat} chat: ${note}`);
            const artifact = `${MODEL_NAME}-${seat}+drafter.json`;
            if (!settles) return { status: "pass", note };
            settlement = await settleReference(references, results, {
              id,
              artifact,
              bytes: seatArtifactBytes(speculative.ids),
            });
            return { ...referenceEntryFields(settlement, artifact), note };
          },
        );
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(id, settlement, references));
        }
      });

      if (seat !== REFERENCE_SEAT) return;

      // -----------------------------------------------------------------------
      // ④ identity on the reference seat over the drafter goldens' prompts
      // -----------------------------------------------------------------------
      for (const name of CASES) {
        it(`④ ${name}: ${MAX_NEW_TOKENS} tokens speculative / plain`, async () => {
          assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
          const golden = await readGolden(name);
          const prompt = [...golden.prompt];
          const speculative = await runTurn(pipeline, prompt, "always");
          const plain = await runTurn(pipeline, prompt, false);
          assertEquals(speculative.ids, plain.ids, `${name}: token ids`);
          assertEquals(speculative.positions, plain.positions, `${name}: positions`);
          const prefix = commonPrefix(speculative.ids, [...golden.tokens]);
          assert(
            prefix >= Math.min(GOLDEN_PREFIX_FLOOR, speculative.ids.length),
            `${name}: the common prefix with the torch continuation is ${prefix} tokens` +
              ` (floor ${GOLDEN_PREFIX_FLOOR})`,
          );
          const speculation = speculative.stop.speculation;
          assert(speculation !== undefined && speculation.cycles > 0, `${name}: no cycle ran`);
          const perCycle = speculation.delivered / speculation.cycles;
          console.log(
            `[e2e] gemma4 e4b drafter ④ ${name}: T=${prompt.length} / ${speculative.ids.length}` +
              ` tokens / torch prefix ${prefix} / ${perCycle.toFixed(3)} tokens per cycle` +
              ` (${speculation.cycles} cycles, accepted ${speculation.accepted}/${speculation.drafted})` +
              ` / speculative ${speculative.ms.toFixed(0)} ms vs plain ${plain.ms.toFixed(0)} ms`,
          );
          const floor = ACCEPTANCE_FLOOR[name];
          if (floor !== undefined) {
            assert(
              perCycle >= floor,
              `${name}: ${perCycle.toFixed(3)} tokens per cycle is under the floor ${floor}`,
            );
          }
        });
      }
    },
  );
}

const ROW_IDS: readonly string[] = SEATS.map(rowId);
// The rows are implementation-time checks: registered only on a run that can run them
// (`referenceGatePasses` counts the registered cases alone).
if (RUNNABLE) {
  references.warnMissing(ROW_IDS);
  registerReferenceGate(references, { runnable: RUNNABLE, caseIds: ROW_IDS });
}

// ---------------------------------------------------------------------------
// ⑤ draft parity (real GPU, upstream parity)
// ---------------------------------------------------------------------------

const f32Of = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  assert(tensor.data instanceof Float32Array, `${where}: not f32`);
  return tensor.data;
};

describe.skipIf(!UPSTREAM_PARITY || !RUNNABLE)(
  `gemma4 e4b drafter: drafts match the torch drafter in ≥ 99% (real GPU)${SERIES_CHECKS_OPT_IN}${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    let config: PipelineConfig;
    let ple: Gemma4Ple | undefined;
    let target: Session | undefined;
    let drafterSession: Session | undefined;
    let drafter: Gemma4Drafter | undefined;
    let targetOutputs: readonly string[] = [];
    let destroy: (() => void) | undefined;
    const counts: { perStep: number[]; total: number; cycles: number }[] = [];

    beforeAll(async () => {
      config = configOf();
      const choice = { model: MODEL_NAME, quant: REFERENCE_SEAT };
      const { index, openBlock } = await openGemma4Ple(STAGED, choice);
      ple = createGemma4Ple({
        index,
        openBlock,
        vocabSize: VOCAB,
        maxResidentBytes: allResidentBytes(index),
      });
      const targetOpened = await openMirrorComponent(STAGED, MODEL, choice);
      const drafterOpened = await openMirrorComponent(STAGED, DRAFTER, choice);
      const admitted = admitGemma4Drafter("test", drafterOpened.graph, {
        graph: targetOpened.graph,
        rope: config.rope,
        hiddenSize: HIDDEN,
        capacitySymbol: CAPACITY_SYMBOL,
      });
      targetOutputs = targetOpened.graph.outputs;
      const gpu = await acquireGpu();
      destroy = () => gpu.destroy();
      // E2B's draft-match options (`e2e_gemma4_drafter_test.ts`): the states-form attention with
      // the parallel reduction, the order the parallel seat decodes with (its `reorder` folds the
      // row statistics into PV, which adds in the same order — ADR 0102).
      const sessionOptions = {
        numerics: { ops: { attention: "reorder" } },
        [REORDER_STRATEGIES]: { stateAttention: "parallel" },
      } as const;
      const lender = await targetOpened.createSession(gpu, sessionOptions);
      target = lender;
      drafterSession = await drafterOpened.createSession(gpu, {
        ...sessionOptions,
        sharedWeights: Object.fromEntries(
          Object.keys(admitted.sharedWeights).map((borrower) => [
            borrower,
            lender.exportWeight(admitted.sharedWeights[borrower]),
          ]),
        ),
      });
      drafter = {
        session: drafterSession,
        graph: drafterOpened.graph,
        outputs: admitted.outputs,
        rope: config.rope,
        hiddenSize: admitted.hiddenSize,
      };
    });

    afterAll(async () => {
      try {
        // MUST: the borrower first (the lender refuses while its table is lent out).
        await drafterSession?.dispose();
        await target?.dispose();
        destroy?.();
      } finally {
        ple?.dispose();
      }
    });

    /** One target run; returns the selected row's hidden. */
    const runTarget = async (
      context: GenerationContext,
      ids: Int32Array<ArrayBuffer>,
      positions: Int32Array<ArrayBuffer>,
      queryLength: number,
    ): Promise<Float32Array<ArrayBuffer>> => {
      assert(target !== undefined && ple !== undefined, "the target is not open");
      const outputs = await target.run(
        {
          [INPUT_IDS]: { dtype: "i32", shape: [1, ids.length], data: ids },
          [PER_LAYER_INPUTS]: await ple.gather([...ids]),
          ...ropeInputs(config.rope, positions),
          [LAST_ROW]: { dtype: "i32", shape: [1], data: Int32Array.of(queryLength - 1) },
        },
        undefined,
        { context, queryLength },
      );
      assertEquals(outputs[targetOutputs[1]].shape, [1, 1, HIDDEN], "hidden shape");
      return f32Of(outputs[targetOutputs[1]], "hidden");
    };

    for (const name of CASES) {
      it(`⑤ ${name}: 200 cycles × 3 steps against the golden`, async () => {
        assert(target !== undefined && drafter !== undefined, "the pair is not open");
        const golden = await readGolden(name);
        const chunkLength = config.chunkLength;
        const capacity = Math.max(chunkLength, golden.prompt.length + golden.cycles);
        const started = performance.now();
        const context = await target.createGenerationContext({
          bindings: { [CAPACITY_SYMBOL]: capacity },
          chunkLength,
        });
        const borrowed = await openDrafterContext(drafter, context);
        const perStep = Array.from({ length: GEMMA4_DRAFT_STEPS }, () => 0);
        const distinct = new Set<number>();
        try {
          let hidden!: Float32Array<ArrayBuffer>;
          for (const chunk of planPrefillChunks(golden.prompt.length, chunkLength)) {
            const ids = new Int32Array(new ArrayBuffer(chunkLength * 4));
            const positions = new Int32Array(new ArrayBuffer(chunkLength * 4));
            for (let row = 0; row < chunk.queryLength; row += 1) {
              ids[row] = golden.prompt[chunk.position + row];
              positions[row] = chunk.position + row;
            }
            hidden = await runTarget(context, ids, positions, chunk.queryLength);
          }
          for (let cycle = 0; cycle < golden.cycles; cycle += 1) {
            const position = golden.prompt.length + cycle;
            if (cycle > 0) {
              hidden = await runTarget(
                context,
                Int32Array.of(golden.tokens[cycle - 1]),
                Int32Array.of(position - 1),
                1,
              );
            }
            assertEquals(context.pastLength, position, `${name}: logical length at ${cycle}`);
            const draft = await draftOnce(drafter, borrowed, {
              token: golden.tokens[cycle],
              hidden,
              position,
            });
            for (let step = 0; step < GEMMA4_DRAFT_STEPS; step += 1) {
              distinct.add(draft[step]);
              if (draft[step] === golden.draft[cycle * GEMMA4_DRAFT_STEPS + step]) {
                perStep[step] += 1;
              }
            }
          }
        } finally {
          // MUST: borrower → lender (the runtime refuses the reverse order).
          await borrowed.dispose();
          await context.dispose();
        }
        assert(
          distinct.size >= 32,
          `${name}: only ${distinct.size} distinct draft tokens (the drafter's output is stuck)`,
        );
        const total = perStep.reduce((sum, value) => sum + value, 0);
        counts.push({ perStep, total, cycles: golden.cycles });
        console.log(
          `[e2e] gemma4 e4b drafter ⑤ ${name}: T=${golden.prompt.length} / match` +
            ` ${total}/${golden.cycles * GEMMA4_DRAFT_STEPS} (per step ${perStep
              .map((hit) => `${((hit / golden.cycles) * 100).toFixed(1)}%`)
              .join(" / ")}) / ${(performance.now() - started).toFixed(0)} ms`,
        );
      });
    }

    it("⑤ the three cases together reach the floor", () => {
      assertEquals(counts.length, CASES.length, "cases matched");
      const matched = counts.reduce((sum, one) => sum + one.total, 0);
      const drafts = counts.reduce((sum, one) => sum + one.cycles * GEMMA4_DRAFT_STEPS, 0);
      const rate = matched / drafts;
      console.log(
        `[e2e] gemma4 e4b drafter ⑤ total: ${matched}/${drafts} = ${(rate * 100).toFixed(2)}%` +
          ` (floor ${(MATCH_FLOOR * 100).toFixed(0)}%)`,
      );
      assert(
        rate >= MATCH_FLOOR,
        `the draft match ${(rate * 100).toFixed(2)}% is under the ${MATCH_FLOOR * 100}% floor` +
          ` (per case ${counts
            .map((one, index) => `${CASES[index]} ${one.perStep.join(",")}`)
            .join(" / ")})`,
      );
    });
  },
);
