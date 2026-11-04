/**
 * PLE の **GPU 常駐席**（ADR 0085 追記〈GPU 常駐席〉）の実資産検収 — 段 H-28。
 *
 * 門は 3 本:
 *
 * ① **golden とのビット一致**: GPU 内 gather の出力が、台本が torch の 35 表経路で採った
 *    `ple.probe.safetensors`（= PLE をグラフに残していたら `embedding` + 直後の `mul` が
 *    出していた値そのもの）と **u32 完全一致**する。tolerance は持たない。
 * ② **ホスト経路との交差 parity**: 同じ token に対するホスト `ple.gather` の出力と **u32 完全
 *    一致**する。①が割れたときに「GPU 経路が悪いのか golden が古いのか」を切り分ける線で、
 *    ホスト経路は ADR 0085 決定 4 の正本である。
 * ③ **生成 parity**: `pleResidency: "gpu"` で読んだパイプラインの 16 token greedy 生成 id 列が
 *    `"host"` と完全一致する。prefill も decode も GPU gather を通るので、片方だけの席では
 *    ないことがここで出る。
 *
 * ## 資産
 *
 * ①② read the PLE of the mobile QAT mirror's e2b `model` container
 * (`models/karume-gemma4-qat-mobile/` — the container is byte-identical to the mobile line's
 * `gemma4-qat-e2b-product` series it was assembled from; the q4_0 QAT line's E2B series is the
 * other spelling, `gemma4-e2b-qat-product`, and is not read here) and the
 * probe from the test-asset root (`goldens/gemma4-qat-e2b-product/ple.probe.safetensors` —
 * ADR 0139 decisions 1 and 2); ③ reads the mirrors `models/karume-gemma4-qat-mobile/` and
 * `models/karume-gemma4-qat/` (the q4_0 line — the regular gemma pipeline's int4 seats since the
 * regular line ships `bf16` alone, 2026-11-04). 無い環境では**明示 SKIP** する（ADR 0005）。
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
/** 生成 parity の刻み（`e2e_gemma4_product_test.ts` の交差 parity と同じ K）。 */
const GREEDY_STEPS = 16;
/**
 * 早く EOS を引かない開放的な入力（parity が 1〜2 token で閉じると decode 側が空振りする）。
 * 前置きの長い問いにしてあるのは、prefill をバケット 1 本ではなく複数行で踏ませるためである。
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
// PLE は容器の資産なので、索引の有無は容器の有無と同じ（欠けていれば `openGemma4Ple` が
// 資産の突合で fail loudly する）。
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

/** u32 のビット列で突き合わせる（f32 の `===` は NaN と ±0 を取り逃がす）。 */
const assertBitsEqual = (actual: Float32Array, expected: Float32Array, where: string): void => {
  assertEquals(actual.length, expected.length, `${where}: 要素数`);
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
    `${where}: ${mismatches} 要素が不一致（最初は [${first}] ` +
      `${first < 0 ? "" : `${expected[first]} ≠ ${actual[first]}`}）`,
  );
};

test.skipIf(!PRODUCT_PRESENT || !GPU_AVAILABLE)(
  `PLE GPU 常駐: gather が golden ともホスト経路とも u32 完全一致（実GPU）${PROBE_SKIP_REASON}`,
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
    assert(tokenView !== undefined && inputView !== undefined, "probe のテンソル 2 本");
    const tokens = new Int32Array(probe.buffer, tokenView.byteOffset, tokenView.byteLength / 4);
    const golden = new Float32Array(probe.buffer, inputView.byteOffset, inputView.byteLength / 4);
    assertEquals(golden.length, tokens.length * index.layers * index.dim, "golden の要素数");

    const gpu = await acquireGpu();
    const host = createGemma4Ple({ index, openBlock, vocabSize: VOCAB, maxResidentBytes: 0 });
    try {
      const bytes = gemma4PleGpuBytes(index);
      assert(
        gpu.limits.maxStorageBufferBindingSize >= bytes.values,
        `この device の maxStorageBufferBindingSize ${gpu.limits.maxStorageBufferBindingSize} が` +
          ` QAT E2B の PLE ${bytes.values} バイトに足りない`,
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
        assertEquals(resident.residentBytes, bytes.total, "常駐した量子化バイト列");
        const batch = await gpu.beginBatch();
        let output;
        try {
          output = await resident.enqueue(batch, tokens);
        } finally {
          await batch.finish();
        }
        const actual = new Float32Array(await output.read());
        // ① torch の 35 表経路との厳密一致。
        assertBitsEqual(actual, golden, "GPU 内 gather と golden");
        // ② ホスト経路（ADR 0085 決定 4 の正本）との交差 parity。
        const reference = (await host.gather([...tokens])) as Tensor & { data: Float32Array };
        assertBitsEqual(actual, reference.data, "GPU 内 gather とホスト gather");
      } finally {
        await resident.dispose();
      }
    } finally {
      host.dispose();
      gpu.destroy();
    }
  },
);

/** 1 ターンの greedy 生成 id 列（停止 token も含めて観測席で拾う）。 */
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

// E2B は i4（group scale）・E4B は i2（行 scale）で、格納形ごとに `embedding` の変種が変わる —
// 両方を回さないと片方の scale の引き方が黙って間違っていても緑になる。
for (const model of ["e2b", "e4b"] as const) {
  test.skipIf(!QAT_PRESENT || !GPU_AVAILABLE)(
    `PLE GPU 常駐: QAT ${model} の greedy 生成 id 列が host と完全一致（実GPU）`,
    async () => {
      const source = nodeDirectory(QAT_ROOT);
      const expected = await generateIds(() =>
        gemma4QatMobile.Pipeline.fromPretrained(source, { model }),
      );
      assertEquals(
        expected.length,
        GREEDY_STEPS,
        "host 経路が刻みぶん生成していない（parity が空振りする）",
      );
      const actual = await generateIds(() =>
        gemma4QatMobile.Pipeline.fromPretrained(source, { model, pleResidency: "gpu" }),
      );
      assertEquals(actual, expected, "GPU 常駐席の生成 id 列");
    },
  );
}

test.skipIf(!GEMMA_PRESENT || !GPU_AVAILABLE)(
  "PLE GPU-resident: the gemma pipeline's q4_0 E2B matches the host where it fits and names the shortfall elsewhere (real GPU)",
  async () => {
    const source = nodeDirectory(GEMMA_ROOT);
    // 索引は配布形ミラーの `model` 容器の資産（ADR 0109 決定 4）— 開くのは part 0 と索引だけ。
    const { index } = await openGemma4Ple(GEMMA_ROOT, INT4_SEAT);
    const bytes = gemma4PleGpuBytes(index);
    const gpu = await acquireGpu();
    const fits =
      gpu.limits.maxStorageBufferBindingSize >= bytes.values &&
      gpu.limits.maxBufferSize >= bytes.values;
    gpu.destroy();
    if (!fits) {
      // MUST: 黙ってホスト経路へ退避しない（席を要求した利用者に不足を返す）。
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
      "host 経路が刻みぶん生成していない（parity が空振りする）",
    );
    const actual = await generateIds(() =>
      Pipeline.fromPretrained(source, { ...INT4_SEAT, pleResidency: "gpu" }),
    );
    assertEquals(actual, expected, "GPU 常駐席の生成 id 列");
  },
);

// 温度 > 0 の decode は greedy 経路を通らず通常 run（= `enqueueRead`）で走る。gather と target を
// 同じ batch に積むので、GPU 常駐席が余分に払うフェンスは prefill の chunk ぶんだけになる
// （ADR 0054 追記〈グラフ出力の一括読み戻し〉— 席の残件 ①）。
test.skipIf(!QAT_PRESENT || !GPU_AVAILABLE)(
  "PLE GPU 常駐: 既定サンプラー経路の decode も host と同じ id 列で、余分なフェンスは prefill ぶんだけ（実GPU）",
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
    assertEquals(host.ids.length, GREEDY_STEPS, "host 経路が刻みぶん生成していない");
    const resident = await generate("gpu");
    assertEquals(resident.ids, host.ids, "GPU 常駐席の生成 id 列（サンプラー経路）");
    // GPU 常駐席が host 経路より余分に払う queue 待ちは prefill chunk 1 本ぶんの gather batch
    // だけで、decode の 16 token は 1 本も払わない（同じ batch の終端 map で読み戻す）。
    // 席の残件 ① の前は decode ごとに 1 本増えていた（= 16 + 1）。
    // The host path also pays one wait: its decode run builds a slot backing, and a run that builds
    // one first waits for released memory (ADR 0095 addendum of 2026-10-24). The resident path
    // builds its backings through enqueue, which never waits, so the two counts now match.
    assertEquals(resident.fences - host.fences, 0, `host ${host.fences} / gpu ${resident.fences}`);
  },
);
