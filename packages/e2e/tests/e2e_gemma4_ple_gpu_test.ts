/**
 * Real-asset acceptance of the PLE's **GPU-resident path** (ADR 0085 addendum "GPU-resident
 * seat") — step H-28.
 *
 * Three checks:
 *
 * ① **Bit identity with the golden**: the output of the in-GPU gather is **u32-identical** to
 *    `ple.probe.safetensors`, which the script took through torch's 35-table path (= exactly the
 *    values `embedding` + the `mul` right after it would have produced had the PLE stayed in the
 *    graph). No tolerance.
 * ② **Cross parity with the host path**: it is **u32-identical** to the output of the host
 *    `ple.gather` for the same tokens. This is the line that tells "the GPU path is wrong" from
 *    "the golden is stale" when ① splits; the host path is the source of truth of ADR 0085
 *    decision 4.
 * ③ **Generation parity**: the 16-token greedy id sequence of a pipeline loaded with
 *    `pleResidency: "gpu"` is identical to `"host"`. Both prefill and decode go through the GPU
 *    gather, so this shows the path does not cover only one of them.
 *
 * ## Assets
 *
 * ①② read the PLE of the mobile QAT mirror's e2b `model` container
 * (`models/karume-gemma4-qat-mobile/` — the container is byte-identical to the mobile line's
 * `gemma4-qat-e2b-product` series it was assembled from; the q4_0 QAT line's E2B series is the
 * other spelling, `gemma4-e2b-qat-product`, and is not read here) and the
 * probe from the test-asset root (`goldens/gemma4-qat-e2b-product/ple.probe.safetensors` —
 * ADR 0139 decisions 1 and 2); ③ reads the mirrors `models/karume-gemma4-qat-mobile/` and
 * `models/karume-gemma4-qat/` (the q4_0 line — the regular gemma pipeline's int4 seats since the
 * regular line ships `bf16` alone, 2026-11-04). Where they are missing the cases **SKIP
 * explicitly** (ADR 0005).
 * The probe follows ADR 0139's fetched rule: absent from a complete fetch of the pinned test-asset dataset it FAILs ①② (the
 * dataset at the pin lacks it); without a complete fetch ①② SKIP with the reason in the title.
 *
 * ## The regular gemma pipeline's PLE depends on the device
 *
 * One binding holds the values, so on a device whose `maxStorageBufferBindingSize` is below the
 * q4_0 E2B's grouped int4 values (schema 4, about 1.09 GiB) the seat cannot be used. ③ then checks
 * that it fails loudly naming the shortfall (not falling back to the host path is the seat's
 * contract). The regular line's `bf16` PLE (4.70 GB) has its own case in
 * `e2e_gemma4_bf16_mirror_test.ts` ④.
 */
import { isNotFound, readFile, statSync } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { acquireGpu, parseSafetensors, type Tensor } from "@karume/runtime";
import { nodeDirectory } from "@karume/hub/node";
import { Pipeline } from "../../models/gemma.ts";
import * as gemma4QatMobile from "../../models/gemma4-qat-mobile.ts";
import { createGemma4Ple } from "../../models/src/gemma/ple.ts";
import { createGemma4PleResident, gemma4PleGpuBytes } from "../../models/src/gemma/ple-gpu.ts";
import { mirrorAvailable, openGemma4Ple } from "../../models/tests/helpers/gemma-mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { countFences } from "../../runtime/tests/helpers/fences.ts";
import {
  goldenDir,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
} from "@karume/test-support/test-assets";

/**
 * The mobile QAT product series' goldens (the PLE probe — not the q4_0 line's
 * `gemma4-e2b-qat-product`).
 */
const PRODUCT_ROOT = goldenDir("gemma4-qat-e2b-product");
const QAT_ROOT = new URL("../../../models/karume-gemma4-qat-mobile/", import.meta.url);
/** The q4_0 QAT line's mirror — the regular gemma pipeline's int4 seats (E2B is its default model). */
const GEMMA_ROOT = new URL("../../../models/karume-gemma4-qat/", import.meta.url);
/**
 * The seat the regular-pipeline case reads: the q4_0 line's `i4-fast`, the fused-parallel seat the
 * case ran on the regular line before (named because the line's default is the reference `i4`).
 */
const INT4_SEAT = { quant: "i4-fast" } as const;
const PLE_PROBE_FILE = "ple.probe.safetensors";
const PROBE_TOKENS_KEY = "tokens";
const PROBE_INPUTS_KEY = "per_layer_inputs";
const VOCAB = 262144;
/**
 * The step count of the generation parity (the same K as the cross parity of
 * `e2e_gemma4_product_test.ts`).
 */
const GREEDY_STEPS = 16;
/**
 * An open-ended input that does not hit EOS early (if the parity closes after 1–2 tokens, the
 * decode side checks nothing). The question has a long lead-in so that prefill runs over several
 * rows rather than one bucket.
 */
const PROMPT =
  "List three things that make the ocean important to the planet, with one " +
  "short sentence for each.";

const isFile = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const QAT_PRESENT = mirrorAvailable(QAT_ROOT);
const GEMMA_PRESENT = mirrorAvailable(GEMMA_ROOT);
const PROBE = new URL(PLE_PROBE_FILE, PRODUCT_ROOT);
const PROBE_RELATIVE = PROBE.href.slice(TEST_ASSETS_ROOT.href.length);
/** ADR 0139's fetched rule: an absent probe SKIPs only without a complete fetch of the pin. */
const PROBE_SKIP = !isFile(PROBE) && !testAssetsFetched();
// The PLE is an asset of the container, so the index is present exactly when the container is
// (if it is missing, `openGemma4Ple` fails loudly when it cross-checks the assets).
const PRODUCT_PRESENT = QAT_PRESENT && !PROBE_SKIP;
/** The title suffix of ①② that names the probe SKIP (empty otherwise). */
const PROBE_SKIP_REASON = PROBE_SKIP
  ? ` (SKIP: ${PROBE_RELATIVE} absent and outputs/test-assets/ is not a complete fetch of the ` +
    "pin — node tools/test-assets/fetch.ts)"
  : "";

if (!QAT_PRESENT) {
  console.warn(
    "[karume] models/karume-gemma4-qat-mobile/ is absent (or not karume/5), so the PLE GPU-resident " +
      "golden case ①② and the QAT generation-parity cases SKIP (assemble it with dist.py in " +
      "tools/export-recipes)",
  );
}
if (!GEMMA_PRESENT) {
  console.warn(
    "[karume] models/karume-gemma4-qat/ is absent (or not karume/5), so the gemma pipeline's E2B " +
      "PLE GPU-resident generation-parity case SKIPs (assemble it with dist.py in " +
      "tools/export-recipes)",
  );
}

/** Compares as u32 bit patterns (f32 `===` misses NaN and ±0). */
const assertBitsEqual = (actual: Float32Array, expected: Float32Array, where: string): void => {
  assertEquals(actual.length, expected.length, `${where}: element count`);
  const left = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const right = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  let mismatches = 0;
  let first = -1;
  for (let at = 0; at < left.length; at += 1) {
    if (left[at] === right[at]) continue;
    mismatches += 1;
    if (first < 0) first = at;
  }
  assertEquals(
    mismatches,
    0,
    `${where}: ${mismatches} elements differ（first at [${first}] ` +
      `${first < 0 ? "" : `${expected[first]} ≠ ${actual[first]}`}）`,
  );
};

test.skipIf(!PRODUCT_PRESENT || !GPU_AVAILABLE)(
  `PLE GPU-resident: the gather is u32-identical to both the golden and the host path (real GPU)${PROBE_SKIP_REASON}`,
  async () => {
    assert(
      isFile(PROBE),
      `${PROBE_RELATIVE} absent although outputs/test-assets/ holds a complete fetch of ` +
        `${TEST_ASSETS.repo}@${TEST_ASSETS.revision} — the test-asset dataset at the pin lacks it`,
    );
    // The container is the one the probe was taken from (e2b; every QAT quant shares its i4
    // `model` container, and no Session is built here).
    const { index, openBlock } = await openGemma4Ple(QAT_ROOT, { model: "e2b" });
    const probe = parseSafetensors((await readFile(PROBE)).buffer);
    const tokenView = probe.tensors.get(PROBE_TOKENS_KEY);
    const inputView = probe.tensors.get(PROBE_INPUTS_KEY);
    assert(tokenView !== undefined && inputView !== undefined, "the probe's two tensors");
    const tokens = new Int32Array(probe.buffer, tokenView.byteOffset, tokenView.byteLength / 4);
    const golden = new Float32Array(probe.buffer, inputView.byteOffset, inputView.byteLength / 4);
    assertEquals(
      golden.length,
      tokens.length * index.layers * index.dim,
      "the golden's element count",
    );

    const gpu = await acquireGpu();
    const host = createGemma4Ple({ index, openBlock, vocabSize: VOCAB, maxResidentBytes: 0 });
    try {
      const bytes = gemma4PleGpuBytes(index);
      assert(
        gpu.limits.maxStorageBufferBindingSize >= bytes.values,
        `this device's maxStorageBufferBindingSize ${gpu.limits.maxStorageBufferBindingSize} is` +
          ` too small for the QAT E2B PLE's ${bytes.values} bytes`,
      );
      const resident = await createGemma4PleResident({
        gpu,
        index,
        openBlock,
        vocabSize: VOCAB,
        inputName: "per_layer_inputs",
        idsName: "input_ids",
        rows: [1, tokens.length],
        entry: "gemma4QatMobile.Pipeline",
      });
      try {
        assertEquals(resident.residentBytes, bytes.total, "the resident quantized bytes");
        const batch = await gpu.beginBatch();
        let output;
        try {
          output = await resident.enqueue(batch, tokens);
        } finally {
          await batch.finish();
        }
        const actual = new Float32Array(await output.read());
        // ① Exact match with torch's 35-table path.
        assertBitsEqual(actual, golden, "in-GPU gather vs golden");
        // ② Cross parity with the host path (the source of truth of ADR 0085 decision 4).
        const reference = (await host.gather([...tokens])) as Tensor & { data: Float32Array };
        assertBitsEqual(actual, reference.data, "in-GPU gather vs host gather");
      } finally {
        await resident.dispose();
      }
    } finally {
      host.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The greedy id sequence of one turn (collected through the observation hook, stop token included).
 */
const generateIds = async (
  open: () => Promise<{ chat: Pipeline["chat"]; dispose: () => Promise<void> }>,
): Promise<number[]> => {
  const pipeline = await open();
  try {
    const ids: number[] = [];
    const stream = pipeline.chat([{ role: "user", content: PROMPT }], {
      maxNewTokens: GREEDY_STEPS,
      sampler: { temperature: 0 },
      onToken: (id) => ids.push(id),
    });
    await stream.text();
    return ids;
  } finally {
    await pipeline.dispose();
  }
};

// E2B is i4 (group scale) and E4B is i2 (row scale), and the `embedding` variant changes with the
// storage — without running both, one of them could look up its scale wrongly and still go green.
for (const model of ["e2b", "e4b"] as const) {
  test.skipIf(!QAT_PRESENT || !GPU_AVAILABLE)(
    `PLE GPU-resident: QAT ${model}'s greedy id sequence is identical to host (real GPU)`,
    async () => {
      const source = nodeDirectory(QAT_ROOT);
      const expected = await generateIds(() =>
        gemma4QatMobile.Pipeline.fromPretrained(source, { model }),
      );
      assertEquals(
        expected.length,
        GREEDY_STEPS,
        "the host path did not generate the full step count (the parity would check nothing)",
      );
      const actual = await generateIds(() =>
        gemma4QatMobile.Pipeline.fromPretrained(source, { model, pleResidency: "gpu" }),
      );
      assertEquals(actual, expected, "the GPU-resident path's generated id sequence");
    },
  );
}

test.skipIf(!GEMMA_PRESENT || !GPU_AVAILABLE)(
  "PLE GPU-resident: the gemma pipeline's q4_0 E2B matches the host where it fits and names the shortfall elsewhere (real GPU)",
  async () => {
    const source = nodeDirectory(GEMMA_ROOT);
    // The index is an asset of the distribution mirror's `model` container (ADR 0109 decision 4)
    // — only part 0 and the index are opened.
    const { index } = await openGemma4Ple(GEMMA_ROOT, INT4_SEAT);
    const bytes = gemma4PleGpuBytes(index);
    const gpu = await acquireGpu();
    const fits =
      gpu.limits.maxStorageBufferBindingSize >= bytes.values &&
      gpu.limits.maxBufferSize >= bytes.values;
    gpu.destroy();
    if (!fits) {
      // MUST: no silent fallback to the host path (the caller who asked for the GPU-resident
      // path gets the shortfall back).
      await assertRejects(
        () => Pipeline.fromPretrained(source, { ...INT4_SEAT, pleResidency: "gpu" }),
        Error,
        "maxStorageBufferBindingSize",
      );
      return;
    }
    const expected = await generateIds(() => Pipeline.fromPretrained(source, INT4_SEAT));
    assertEquals(
      expected.length,
      GREEDY_STEPS,
      "the host path did not generate the full step count (the parity would check nothing)",
    );
    const actual = await generateIds(() =>
      Pipeline.fromPretrained(source, { ...INT4_SEAT, pleResidency: "gpu" }),
    );
    assertEquals(actual, expected, "the GPU-resident path's generated id sequence");
  },
);

// Decode at temperature > 0 does not take the greedy path; it runs as a normal run
// (= `enqueueRead`). The gather and the target go into the same batch, so the only extra fences
// the GPU-resident path pays are those of the prefill chunks (ADR 0054 addendum "batched readback
// of graph outputs" — the path's open item ①).
test.skipIf(!QAT_PRESENT || !GPU_AVAILABLE)(
  "PLE GPU-resident: decode on the default sampler path also gives the host's id sequence, with extra fences only for prefill (real GPU)",
  async () => {
    const source = nodeDirectory(QAT_ROOT);
    const generate = async (
      pleResidency: "host" | "gpu",
    ): Promise<{ ids: number[]; fences: number }> => {
      const gpu = await acquireGpu();
      try {
        const pipeline = await gemma4QatMobile.Pipeline.fromPretrained(source, {
          model: "e2b",
          pleResidency,
          gpu,
        });
        const fences = countFences(gpu);
        try {
          const ids: number[] = [];
          const stream = pipeline.chat([{ role: "user", content: PROMPT }], {
            maxNewTokens: GREEDY_STEPS,
            sampler: { temperature: 0.8, topK: 40, seed: 7 },
            onToken: (id) => ids.push(id),
          });
          await stream.text();
          return { ids, fences: fences.count() };
        } finally {
          fences.restore();
          await pipeline.dispose();
        }
      } finally {
        gpu.destroy();
      }
    };
    const host = await generate("host");
    assertEquals(
      host.ids.length,
      GREEDY_STEPS,
      "the host path did not generate the full step count",
    );
    const resident = await generate("gpu");
    assertEquals(
      resident.ids,
      host.ids,
      "the GPU-resident path's generated id sequence (sampler path)",
    );
    // The only extra queue wait the GPU-resident path pays over the host path is the gather batch
    // of one prefill chunk; the 16 decode tokens pay none (they are read back by the map at the
    // end of the same batch). Before the path's open item ① each decode added one (= 16 + 1).
    // The host path also pays one wait: its decode run builds a slot backing, and a run that builds
    // one first waits for released memory (ADR 0095 addendum of 2026-10-24). The resident path
    // builds its backings through enqueue, which never waits, so the two counts now match.
    assertEquals(resident.fences - host.fences, 0, `host ${host.fences} / gpu ${resident.fences}`);
  },
);
