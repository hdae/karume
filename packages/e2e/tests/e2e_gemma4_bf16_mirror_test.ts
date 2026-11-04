// Real-weight Gemma 4 E2B / E4B of the regular line's preset `bf16` — the original weights, the
// distribution's one preset and default — through the distribution mirror (`models/karume-gemma4/`
// — `dist.py --pipeline gemma4`).
//
// The preset's `model` container stores Google's BF16 checkpoint as it is, the per-layer embeddings
// included (PLE index schema 5: a bf16 `values` table and no `scales` — ADR 0085 addendum
// 2026-11-04). E2B's drafter is the regular int8 one, borrowing the bf16 target's
// `model.lm_head.weight`. The cases, per model:
//
// ① the manifest entry (no GPU): `defaultQuant` is `bf16`, the one preset; it names the bf16
//    container (and E2B's int8 drafter) with the reference summation order (`session` {}), the
//    `model` role's one storage; its container's PLE index is schema 5 with 2 bytes per value
// ② greedy parity (real GPU): `fromPretrained` with no `quant` (the default, so it is the bf16
//    preset that runs), PLE gathered on the host: the greedy chain of each golden case of the bf16
//    series equals the torch golden token for token — for E2B both without and with speculation.
//    The series rounds nothing, so the goldens are the upstream checkpoint's own continuations (in
//    f32); each golden's top1 − top2 margin exceeds 2 × atol, so the comparison does not hinge on
//    numeric noise (token identity, not a tolerance band)
// ③ E2B speculation (real GPU): a greedy chat turn with `speculative: "always"` gives the same
//    text, stop and token count as the turn without it, with at least one draft accepted (the int8
//    drafter really ran on the borrowed bf16 table)
// ④ the PLE resident on the GPU (real GPU): the bf16 values (E2B 4.70 GB, E4B 5.64 GB) need one
//    binding; on a device whose binding limit is smaller, `pleResidency: "gpu"` fails loudly with
//    the shortfall instead of falling back to the host; on a larger one the greedy chain of the
//    first golden case equals the golden
//
// ## Assets
//
// The weights are the local mirror (ADR 0139 decision 1); without it the cases SKIP explicitly.
// The goldens are the bf16 series' recipe outputs in the test-asset root (decision 2):
// `goldens/gemma4-<model>-product-bf16/` — `greedy.<case>.safetensors` and the `reference.json`
// that binds them — under the fetched rule: absent from a complete fetch of the pinned dataset ⇒
// FAIL; otherwise SKIP with the missing files and `node tools/test-assets/fetch.ts` in the title.
//
// ## Memory
//
// One pipeline at a time. The bf16 weights the GPU holds are about 4.6 GB (E2B) and 9.3 GB (E4B);
// the PLE (4.70 / 5.64 GB) stays on the host, read whole into RAM here (`maxResidentPleBytes` from
// the index), so no row is read twice.

import { isNotFound, readFile, readTextFileSync, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { type Manifest, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { type ChatMessage, type GenerationSequence, Pipeline } from "../../models/gemma.ts";
import { GEMMA4_DRAFT_STEPS } from "../../models/src/gemma/speculative.ts";
import { openGemma4Ple } from "../../models/tests/helpers/gemma-mirror.ts";
import { mirrorDirectory, mirrorHasModel, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { allResidentBytes } from "../../models/tests/helpers/ple-budget.ts";
import {
  goldenDir,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
} from "@karume/test-support/test-assets";

const MIRROR = mirrorRoot("karume-gemma4");
const ASSEMBLE =
  "cd tools/export-recipes && uv run python dist.py --pipeline gemma4 --model e2b --model e4b" +
  " --out ../../models/karume-gemma4";
/** The command that restores the goldens (ADR 0139 decision 2). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

const MODELS = ["e2b", "e4b"] as const;
type Model = (typeof MODELS)[number];

/** The roles and the preset (`gemma4/distribution.py`). */
const MODEL = "model";
const DRAFTER = "drafter";
const PRESET = "bf16";
/** The bf16 preset's choice of the mirror. */
const choiceOf = (model: Model) => ({ model, quant: PRESET }) as const;

/** Which models ship the int8 drafter (`gemma4.card.GEMMA4_DRAFTER_UPSTREAM`). */
const HAS_DRAFTER: Record<Model, boolean> = { e2b: true, e4b: false };

const VOCAB = 262144;
/** Bytes of one bf16 value. */
const BF16_BYTES = 2;
/** The gemma4 logits tolerance the margin gate is stated against (`e2e_gemma4_12b_test.ts`). */
const ATOL = 1e-2;
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";
const REFERENCE_FILE = "reference.json";
/** The refusal of a GPU-resident bf16 PLE too large for one binding (`src/gemma/ple-gpu.ts`). */
const TOO_LARGE = "bytes of bf16 values in a single binding";

/** The chat turn of ③ (an English question whose greedy answer runs past a few tokens). */
const CHAT: readonly ChatMessage[] = [
  { role: "user", content: "Explain in about five sentences why the sky is blue." },
];
const CHAT_TOKENS = 64;

const isFile = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** The goldens of `files` that are absent, as paths under the test-asset root. */
const missingGoldens = (files: readonly URL[]): string[] =>
  files
    .filter((file) => !isFile(file))
    .map((file) => file.href.slice(TEST_ASSETS_ROOT.href.length));

/**
 * ADR 0139's fetched rule for the goldens a case reads: absent with a complete fetch of the pin ⇒
 * the case runs and {@link assertGoldensPresent} FAILs naming them; absent without one ⇒ the case
 * SKIPs, and `title` says why.
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

const PRESENT: Record<Model, boolean> = {
  e2b: mirrorHasModel(MIRROR, "e2b"),
  e4b: mirrorHasModel(MIRROR, "e4b"),
};
if (MODELS.some((model) => !PRESENT[model])) {
  console.warn(
    `[karume] ${MIRROR.pathname} lacks ${MODELS.filter((model) => !PRESENT[model]).join(" / ")}` +
      ` (or is not karume/5), so those gemma4 bf16 mirror cases SKIP. Assemble: ${ASSEMBLE}`,
  );
}

const manifest: Manifest | undefined = MODELS.some((model) => PRESENT[model])
  ? parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR)))
  : undefined;

const entryOf = (model: Model) => {
  assert(manifest !== undefined, "the mirror's manifest is not read");
  return manifest.models[model];
};

/** The bf16 PLE index of the model's preset container (`readGemma4PleIndex` checks it whole). */
const bf16PleOf = async (model: Model) => (await openGemma4Ple(MIRROR, choiceOf(model))).index;

/** A load of the default preset with the whole PLE resident on the host (no row read twice). */
const loadOptions = async (model: Model) => ({
  model,
  maxResidentPleBytes: allResidentBytes(await bf16PleOf(model)),
});

/** The smaller of the adapter's two limits a single binding must fit. */
const bindingLimit = async (): Promise<number> => {
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter !== null, "no GPU adapter");
  return Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
};

// ---------------------------------------------------------------------------
// ① the manifest entry and the container's PLE (no GPU)
// ---------------------------------------------------------------------------

for (const model of MODELS) {
  test.skipIf(!PRESENT[model])(
    `gemma4 ${model}: bf16 is the one preset and the default, the original weights (no GPU)`,
    async () => {
      const entry = entryOf(model);
      assertEquals(entry.defaultQuant, PRESET, "default preset");
      assertEquals(Object.keys(entry.quants), [PRESET], "presets");
      assertEquals(Object.keys(entry.weights[MODEL]), [PRESET], "model's storages");
      const preset = entry.quants[PRESET];
      assertEquals(
        preset.weights,
        HAS_DRAFTER[model] ? { [MODEL]: PRESET, [DRAFTER]: "i8" } : { [MODEL]: PRESET },
        "the preset's weights",
      );
      assertEquals(preset.session ?? {}, {}, "the reference summation order (session {})");
      assertEquals(preset.label, "Original weights (bf16)", "label");

      const index = await bf16PleOf(model);
      assert(index.storage === "bf16", `PLE storage ${index.storage} (bf16 is needed)`);
      assertEquals(index.tokens, VOCAB, "PLE token rows");
      assertEquals(index.values.rowBytes, index.layers * index.dim * BF16_BYTES, "2 B per value");
    },
  );
}

// ---------------------------------------------------------------------------
// ② ③ greedy parity and speculation through the mirror (real GPU)
// ---------------------------------------------------------------------------

type ObjectRecord = Record<string, unknown>;

const objectAt = (value: unknown, where: string): ObjectRecord => {
  assert(
    typeof value === "object" && value !== null && !Array.isArray(value),
    `${REFERENCE_FILE}: ${where} is not an object`,
  );
  return value as ObjectRecord;
};

const bufferOf = (bytes: Uint8Array<ArrayBuffer>): ArrayBuffer =>
  bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

const i32Of = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined && view.dtype === "I32", `golden '${name}' is not I32`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** The record of the series the goldens come from. */
const recordOf = (root: URL): ObjectRecord =>
  objectAt(JSON.parse(readTextFileSync(new URL(REFERENCE_FILE, root))), "root");

/** The greedy goldens the series' record binds (sorted case names). */
const boundCases = (root: URL): string[] =>
  Object.keys(
    objectAt(objectAt(recordOf(root).reference, "reference").goldens, "reference.goldens"),
  )
    .map((file) => file.slice(GREEDY_PREFIX.length, -SUFFIX.length))
    .sort();

/** One greedy golden, checked against the digest its record binds. */
const readGreedy = async (
  root: URL,
  name: string,
): Promise<{ prompt: number[]; expected: number[]; margin: number[] }> => {
  const file = `${GREEDY_PREFIX}${name}${SUFFIX}`;
  const bytes = await readFile(new URL(file, root));
  const digest = objectAt(
    objectAt(objectAt(recordOf(root).reference, "reference").goldens, "reference.goldens")[file],
    file,
  );
  assertEquals(await sha256Hex(bytes), digest.sha256, `${file}: the sha256 its record binds`);
  const golden = parseSafetensors(bufferOf(bytes));
  const margin = golden.tensors.get("margin");
  assert(margin !== undefined && margin.dtype === "F32", `${name}: 'margin' is not F32`);
  return {
    prompt: [...i32Of(golden, "prompt")],
    expected: [...i32Of(golden, "expected")],
    margin: [...new Float32Array(golden.buffer, margin.byteOffset, margin.byteLength / 4)],
  };
};

/**
 * The greedy chain of `steps` tokens from `prompt`, stop tokens included (the chain of
 * `e2e_gemma4_qat_mirror_test.ts`): a stop token ends a turn and stays the pending frontier, so
 * the next `generate` with an empty prompt continues one uninterrupted greedy run.
 */
const greedyChain = async (
  sequence: GenerationSequence,
  prompt: readonly number[],
  steps: number,
): Promise<number[]> => {
  const ids: number[] = [];
  let next: readonly number[] = prompt;
  while (ids.length < steps) {
    const stream = sequence.generate({ prompt: [...next], maxNewTokens: steps - ids.length });
    for await (const event of stream) {
      if (event.kind === "token") ids.push(event.id);
    }
    const stop = await stream.done;
    if (stop.reason === "max-tokens") break;
    assert(stop.reason === "eos", `unexpected stop '${stop.reason}' after ${ids.length} tokens`);
    ids.push(stop.token);
    next = [];
  }
  return ids;
};

/** Checks the golden's margin gate and returns it. */
const gatedGolden = async (root: URL, name: string) => {
  const golden = await readGreedy(root, name);
  const minMargin = Math.min(...golden.margin);
  assert(
    minMargin > 2 * ATOL,
    `${name}: the golden's smallest margin ${minMargin} is comparable to atol (not a valid gate)`,
  );
  return { ...golden, minMargin };
};

for (const model of MODELS) {
  const root = goldenDir(`gemma4-${model}-product-bf16`);
  const record = new URL(REFERENCE_FILE, root);
  const cases = isFile(record) ? boundCases(root) : [];
  const files = [record, ...cases.map((name) => new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, root))];
  const gate = goldensGate(files);
  test.skipIf(!PRESENT[model] || gate.skip)(
    `gemma4 ${model} bf16: the greedy goldens are in the test-asset root${gate.title}`,
    () => {
      assertGoldensPresent(files);
      assert(cases.length > 0, `${record.pathname}: the record binds no greedy golden`);
    },
  );

  /** One greedy chain per golden case of the series, each against its golden. */
  const parityCases = (opened: () => Pipeline | undefined, speculative: "always" | false) => {
    for (const name of cases) {
      it(`② ${name}: the greedy chain equals the golden (speculative ${speculative})`, async () => {
        const pipeline = opened();
        assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
        const { prompt, expected, minMargin } = await gatedGolden(root, name);
        const sequence = await pipeline.sequence({ speculative });
        try {
          const started = performance.now();
          const ids = await greedyChain(sequence, prompt, expected.length);
          assertEquals(ids, expected, `${name} (speculative ${speculative}): greedy tokens`);
          console.log(
            `[e2e] gemma4 ${model} bf16 ${name} (speculative ${speculative}):` +
              ` T=${prompt.length} + K=${expected.length} / min margin` +
              ` ${minMargin.toExponential(3)} / ${(performance.now() - started).toFixed(0)} ms`,
          );
        } finally {
          await sequence.dispose();
        }
      });
    }
  };

  describe.skipIf(!PRESENT[model] || gate.skip || cases.length === 0 || !GPU_AVAILABLE)(
    `gemma4 ${model} bf16: greedy parity through the mirror's default preset, PLE on the host (real GPU)${gate.title}`,
    () => {
      let pipeline: Pipeline | undefined;
      beforeAll(async () => {
        const started = performance.now();
        // No `quant`: the default preset is the one under test.
        pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), await loadOptions(model));
        console.log(
          `[e2e] gemma4 ${model} bf16: load ${(performance.now() - started).toFixed(0)} ms`,
        );
      });
      afterAll(async () => {
        await pipeline?.dispose();
      });
      parityCases(() => pipeline, false);
    },
  );

  // A separate load: the drafter is built only when `speculative` is requested, so the plain
  // parity above does not depend on the borrowed table.
  describe.skipIf(
    !HAS_DRAFTER[model] || !PRESENT[model] || gate.skip || cases.length === 0 || !GPU_AVAILABLE,
  )(
    `gemma4 ${model} bf16: the int8 drafter on the borrowed bf16 table — greedy parity and a chat turn (real GPU)${gate.title}`,
    () => {
      let pipeline: Pipeline | undefined;
      beforeAll(async () => {
        pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), {
          ...(await loadOptions(model)),
          speculative: { k: GEMMA4_DRAFT_STEPS },
        });
      });
      afterAll(async () => {
        await pipeline?.dispose();
      });
      parityCases(() => pipeline, "always");

      it("③ a speculative greedy chat turn equals the plain one, with drafts accepted", async () => {
        assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
        const opened = pipeline;
        const turn = async (speculative: "always" | false) => {
          const started = performance.now();
          const stream = opened.chat(CHAT, {
            maxNewTokens: CHAT_TOKENS,
            sampler: { temperature: 0 },
            speculative,
          });
          const text = await stream.text();
          return { text, stop: await stream.done, ms: performance.now() - started };
        };
        const speculative = await turn("always");
        const plain = await turn(false);
        assertEquals(speculative.text, plain.text, "the text");
        assertEquals(speculative.stop.reason, plain.stop.reason, "the stop reason");
        assertEquals(speculative.stop.tokens, plain.stop.tokens, "the token count");
        assertEquals(plain.stop.speculation, undefined, "a plain turn carries no speculation");
        const speculation = speculative.stop.speculation;
        assert(speculation !== undefined, "the speculative turn did not speculate");
        assert(
          speculation.accepted > 0,
          `no draft accepted (${speculation.cycles} cycles, ${speculation.drafted} drafted)`,
        );
        console.log(
          `[e2e] gemma4 ${model} bf16 chat: ${speculative.stop.tokens} tokens /` +
            ` ${speculation.cycles} cycles / accepted ${speculation.accepted}/${speculation.drafted}` +
            ` / speculative ${speculative.ms.toFixed(0)} ms vs plain ${plain.ms.toFixed(0)} ms`,
        );
      });
    },
  );

  // -------------------------------------------------------------------------
  // ④ the PLE resident on the GPU (real GPU)
  // -------------------------------------------------------------------------

  describe.skipIf(!PRESENT[model] || gate.skip || cases.length === 0 || !GPU_AVAILABLE)(
    `gemma4 ${model} bf16: the PLE resident on the GPU — parity where one binding holds it, a loud refusal elsewhere (real GPU)${gate.title}`,
    () => {
      /** Why one binding cannot hold the bf16 values here (`undefined` = it can). */
      let tooLarge: string | undefined;
      beforeAll(async () => {
        const index = await bf16PleOf(model);
        const valuesBytes = index.tokens * index.values.rowBytes;
        const limit = await bindingLimit();
        if (limit < valuesBytes) {
          tooLarge = `binding limit ${limit} < ${valuesBytes} bytes of bf16 PLE values`;
        }
      });

      it("④ a device too small for one binding of the bf16 values refuses loudly", async (context) => {
        // The device holds it: the parity case below runs instead.
        if (tooLarge === undefined) context.skip("the bf16 values fit one binding here");
        console.log(`[e2e] gemma4 ${model} bf16 (PLE gpu): ${tooLarge}`);
        await assertRejects(
          async () => {
            const refused = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), {
              model,
              pleResidency: "gpu",
            });
            await refused.dispose();
          },
          Error,
          TOO_LARGE,
        );
      });

      it(`④ ${cases[0] ?? "(none)"}: the greedy chain equals the golden with the PLE resident`, async (context) => {
        // The refusal case above covers this device.
        if (tooLarge !== undefined) context.skip(tooLarge);
        const { prompt, expected } = await gatedGolden(root, cases[0]);
        const pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), {
          model,
          pleResidency: "gpu",
        });
        try {
          const sequence = await pipeline.sequence();
          try {
            assertEquals(await greedyChain(sequence, prompt, expected.length), expected);
          } finally {
            await sequence.dispose();
          }
        } finally {
          await pipeline.dispose();
        }
      });
    },
  );
}
