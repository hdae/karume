// Real-weight Gemma 4 E2B / E4B of the regular line's original-weights preset — the bf16 product
// series (`outputs/series/gemma4-{e2b,e4b}-product-bf16`, recipe
// `gemma4.export_product --storage bf16`) run through the regular gemma pipeline's container entry
// (`Pipeline.fromAssets`).
//
// The series stores Google's BF16 checkpoints as they are, the PLE included: its index is schema 5
// (`bf16` values, no scale table — `karume/ple.py`, ADR 0085 addendum 2026-11-04). The cases, per
// model:
//
// ① the container's PLE index is schema 5 with storage bf16 and a values row of 2 bytes per value,
//    and the provenance record binds the greedy goldens this file reads (no GPU)
// ② greedy parity with the PLE gathered on the host: the pipeline's greedy chain of each bound case
//    equals the torch golden token for token. Each golden's torch top1 − top2 margin exceeds
//    2 × atol, so the comparison does not hinge on numeric noise.
// ③ the PLE resident on the GPU: on a device whose binding limit holds the bf16 values (E2B
//    4.70 GB, E4B 5.64 GB) the same parity; on a smaller one `pleResidency: "gpu"` fails loudly with
//    the shortfall instead of falling back to the host
//
// ## Assets
//
// An implementation-time series check (ADR 0139 decision 6): the original-weights preset has no
// distribution mirror yet, so the cases run only with KARUME_SERIES=1 on a machine that holds the
// series and the model's tokenizer series (`outputs/series/gemma4-<model>-tokenizer`), and SKIP
// explicitly otherwise. `fromAssets` takes every part in RAM (about
// 9.3 GiB for E2B, 14.6 GiB for E4B), one pipeline at a time.

import { isNotFound, readFile, readTextFileSync, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import {
  type GenerationSequence,
  Pipeline,
  type PipelineConfig,
  type PleResidency,
} from "../../models/gemma.ts";
import { gemma4PleAssetSource, readGemma4PleIndex } from "../../models/src/gemma/ple-index.ts";
import {
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "../../runtime/tests/helpers/container-files.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";

const SERIES = new URL("../../../outputs/series/", import.meta.url);
/** The representative path of the `model` container in the series layout (parts are numbered). */
const MODEL_FILE = "model.krm";
/** The model's compiled tokenizer (each model has its own — `gemma4.tokenizer`). */
const tokenizerOf = (model: string): URL =>
  new URL(`gemma4-${model}-tokenizer/tokenizer.json`, SERIES);
const REFERENCE_FILE = "reference.json";
/** The schema of the provenance record (spelling source: `gemma4/provenance.py`). */
const REFERENCE_SCHEMA = 1;
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";
/** The tensors of one greedy golden (`export_decode` layout). */
const GOLDEN_KEYS = ["expected", "margin", "prompt"] as const;
/** The gemma4 logits tolerance the margin gate is stated against (`e2e_gemma4_12b_test.ts`). */
const ATOL = 1e-2;
/** The vocabulary (= the PLE index's token rows). */
const VOCAB = 262144;
/** Bytes of one bf16 value. */
const BF16_BYTES = 2;

/**
 * The E2B / E4B declaration of the regular line's distribution (`models/karume-gemma4`), fixed here
 * so a mirror change does not move this gate.
 */
const CONFIG: PipelineConfig = {
  chunkLength: 768,
  maxChunkLength: 768,
  maxPosition: 131072,
  capacity: 4096,
  rope: {
    sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
  },
};

const isFile = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const objectAt = (value: unknown, where: string): Record<string, unknown> => {
  assert(
    typeof value === "object" && value !== null && !Array.isArray(value),
    `${REFERENCE_FILE}: ${where} is not an object`,
  );
  return value as Record<string, unknown>;
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** One greedy case: the prompt ids, the expected continuation and the torch margin of each step. */
const readCase = async (
  root: URL,
  name: string,
): Promise<{ prompt: number[]; expected: number[]; margin: number[] }> => {
  const bytes = await readFile(new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, root));
  const golden = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  assertEquals([...golden.tensors.keys()].sort(), [...GOLDEN_KEYS], `${name}: tensors`);
  const view = (key: string, dtype: "I32" | "F32") => {
    const found = golden.tensors.get(key);
    assert(found !== undefined && found.dtype === dtype, `${name}: '${key}' is not ${dtype}`);
    return found;
  };
  const prompt = view("prompt", "I32");
  const expected = view("expected", "I32");
  const margin = view("margin", "F32");
  const read = {
    prompt: [...new Int32Array(golden.buffer, prompt.byteOffset, prompt.byteLength / 4)],
    expected: [...new Int32Array(golden.buffer, expected.byteOffset, expected.byteLength / 4)],
    margin: [...new Float32Array(golden.buffer, margin.byteOffset, margin.byteLength / 4)],
  };
  assertEquals(read.margin.length, read.expected.length, `${name}: margin steps`);
  return read;
};

/**
 * The greedy chain of `steps` tokens from `prompt`, stop tokens included (the same chain as
 * `e2e_gemma4_q4_0_qat_product_test.ts`: a stop token ends a turn and stays the pending frontier,
 * so the next `generate` with an empty prompt continues one uninterrupted greedy run — the goldens
 * do not stop).
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

/** The smaller of the adapter's two limits a single binding of `bytes` must fit. */
const bindingLimit = async (): Promise<number> => {
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter !== null, "no GPU adapter");
  return Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
};

for (const model of ["e2b", "e4b"] as const) {
  const series = `gemma4-${model}-product-bf16`;
  const root = new URL(`${series}/`, SERIES);
  const representative = new URL(MODEL_FILE, root);
  const tokenizerFile = tokenizerOf(model);
  const present =
    SERIES_CHECKS &&
    modelPresent(representative) &&
    isFile(new URL(REFERENCE_FILE, root)) &&
    isFile(tokenizerFile);
  if (SERIES_CHECKS && !present) {
    console.warn(
      `[karume] ${root.pathname} (or ${tokenizerFile.pathname}) is absent, so the bf16 ${model}` +
        " product cases SKIP (cd tools/export-recipes && uv run python -m gemma4.export_product" +
        ` --storage bf16 --model ${model})`,
    );
  }
  /** The cases the provenance record binds (read only when the series is present). */
  const cases: string[] = present
    ? Object.keys(
        objectAt(
          objectAt(
            objectAt(JSON.parse(readTextFileSync(new URL(REFERENCE_FILE, root))), "root").reference,
            "reference",
          ).goldens,
          "reference.goldens",
        ),
      )
        .map((file) => file.slice(GREEDY_PREFIX.length, -SUFFIX.length))
        .sort()
    : [];
  test.skipIf(!present)(
    `Gemma 4 ${model} bf16 product: a schema 5 PLE (bf16, no scales) and goldens bound by the record (no GPU)${SERIES_CHECKS_OPT_IN}`,
    async () => {
      const opened = await openSeriesContainer(representative);
      const index = await readGemma4PleIndex(`test: ${series}`, gemma4PleAssetSource(opened));
      assert(index.storage === "bf16", `PLE storage ${index.storage} (bf16 is needed)`);
      assertEquals(index.tokens, VOCAB, "PLE token rows");
      assertEquals(index.values.rowBytes, index.layers * index.dim * BF16_BYTES, "2 B per value");

      const record = objectAt(JSON.parse(readTextFileSync(new URL(REFERENCE_FILE, root))), "root");
      assertEquals(record.schema, REFERENCE_SCHEMA, `${REFERENCE_FILE} schema`);
      assertEquals(record.series, series, "the record's series");
      const reference = objectAt(record.reference, "reference");
      assertEquals(reference.series, series, "the bound series (the series' own goldens)");
      assertEquals(reference.suppressTokens, [], "suppressTokens (E2B / E4B suppress none)");
      assert(cases.length > 0, "the record binds no greedy golden");
      const goldens = objectAt(reference.goldens, "reference.goldens");
      for (const name of cases) {
        const file = `${GREEDY_PREFIX}${name}${SUFFIX}`;
        const digest = objectAt(goldens[file], `reference.goldens.${file}`);
        const bytes = await readFile(new URL(file, root));
        assertEquals(bytes.byteLength, digest.bytes, `${file}: bytes`);
        assertEquals(await sha256Hex(bytes), digest.sha256, `${file}: sha256`);
      }
    },
  );

  const assets = async () => ({
    config: CONFIG,
    model: await Promise.all(resolveParts(representative).map((part) => readFile(part))),
    tokenizer: await readFile(tokenizerFile),
  });

  for (const residency of ["host", "gpu"] as const satisfies readonly PleResidency[]) {
    describe.skipIf(!present || !GPU_AVAILABLE)(
      `Gemma 4 ${model} bf16 product: greedy parity through fromAssets, PLE on the ${residency} (real GPU)${SERIES_CHECKS_OPT_IN}`,
      () => {
        let pipeline: Pipeline | undefined;
        /** Why the GPU-resident PLE cannot be built on this device (`undefined` = it fits). */
        let tooLarge: string | undefined;
        beforeAll(async () => {
          if (residency === "gpu") {
            const opened = await openSeriesContainer(representative);
            const index = await readGemma4PleIndex(series, gemma4PleAssetSource(opened));
            const valuesBytes = index.tokens * index.values.rowBytes;
            const limit = await bindingLimit();
            if (limit < valuesBytes) {
              tooLarge = `binding limit ${limit} < ${valuesBytes} bytes of bf16 PLE values`;
              return;
            }
          }
          const started = performance.now();
          pipeline = await Pipeline.fromAssets(await assets(), { pleResidency: residency });
          console.log(
            `[e2e] gemma4 ${model} bf16 (PLE ${residency}): built in` +
              ` ${(performance.now() - started).toFixed(0)} ms`,
          );
        });
        afterAll(async () => {
          await pipeline?.dispose();
        });

        if (residency === "gpu") {
          it("a device too small for one binding of the bf16 values refuses loudly", async (context) => {
            // The device fits it: the parity cases below run instead.
            if (tooLarge === undefined) context.skip("the bf16 values fit one binding here");
            console.log(`[e2e] gemma4 ${model} bf16 (PLE gpu): ${tooLarge}`);
            await assertRejects(
              async () => {
                const refused = await Pipeline.fromAssets(await assets(), { pleResidency: "gpu" });
                await refused.dispose();
              },
              Error,
              "bytes of bf16 values in a single binding",
            );
          });
        }

        for (const name of cases) {
          it(`${name}: the greedy chain equals the golden token for token`, async (context) => {
            // The refusal case above covers this device.
            if (tooLarge !== undefined) context.skip(tooLarge);
            assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
            const { prompt, expected, margin } = await readCase(root, name);
            const minMargin = Math.min(...margin);
            assert(
              minMargin > 2 * ATOL,
              `${name}: the golden's smallest margin ${minMargin} is comparable to atol — the` +
                " token match could flip on numeric differences (not a valid gate)",
            );
            const sequence = await pipeline.sequence();
            try {
              const started = performance.now();
              const ids = await greedyChain(sequence, prompt, expected.length);
              assertEquals(ids, expected, `${name}: greedy tokens (against the torch golden)`);
              console.log(
                `[e2e] gemma4 ${model} bf16 ${name} (PLE ${residency}): T=${prompt.length} +` +
                  ` K=${expected.length} / min margin ${minMargin.toExponential(3)} /` +
                  ` ${(performance.now() - started).toFixed(0)} ms`,
              );
            } finally {
              await sequence.dispose();
            }
          });
        }
      },
    );
  }
}
