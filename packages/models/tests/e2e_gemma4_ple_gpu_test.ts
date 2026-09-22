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
 * ①② は `outputs/series/gemma4-qat-e2b-product/`（probe と sidecar が同じ世代で並ぶ唯一の場所 —
 * リポジトリ管理外）、③ は配布形ミラー `models/karume-gemma4-qat/` と `models/karume-gemma4/`。
 * 無い環境では**明示 SKIP** する（ADR 0005）。
 *
 * ## 通常 Gemma 4（i8 sidecar）は device 次第
 *
 * 単一束縛なので、i8 の 2.19GiB が `maxStorageBufferBindingSize` に収まらない device では
 * この席そのものが使えない。③はその場合「不足を名乗って fail loudly する」ことを見る
 * （黙ってホスト経路へ退避しないことが席の契約である）。
 */
import { assert, assertEquals, assertRejects } from "@std/assert";
import { acquireGpu, parseSafetensors, type Tensor } from "@karume/runtime";
import { denoDirectory } from "@karume/hub/deno";
import { Gemma4Pipeline } from "../gemma.ts";
import { Gemma4QatPipeline } from "../gemma4-qat.ts";
import { createGemma4Ple } from "../src/gemma/ple.ts";
import { createGemma4PleResident, gemma4PleGpuBytes } from "../src/gemma/ple-gpu.ts";
// 系列出力の PLE sidecar を容器の資産と同じ面へ畳む adapter（recipe が `krm` を書くのは
// 段 3 — ADR 0109 決定 8）。
import { openSeriesPle } from "./helpers/ple-series.ts";
import { mirrorAvailable, openGemma4Ple } from "./helpers/gemma-mirror.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { countFences } from "../../runtime/tests/helpers/fences.ts";

const PRODUCT_ROOT = new URL("../../../outputs/series/gemma4-qat-e2b-product/", import.meta.url);
const QAT_ROOT = new URL("../../../models/karume-gemma4-qat/", import.meta.url);
const GEMMA_ROOT = new URL("../../../models/karume-gemma4/", import.meta.url);
const PLE_INDEX_FILE = "ple.json";
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
const PROMPT = "List three things that make the ocean important to the planet, with one " +
  "short sentence for each.";

const isFile = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const PRODUCT_PRESENT = isFile(new URL(PLE_PROBE_FILE, PRODUCT_ROOT)) &&
  isFile(new URL(PLE_INDEX_FILE, PRODUCT_ROOT));
const QAT_PRESENT = mirrorAvailable(QAT_ROOT);
const GEMMA_PRESENT = mirrorAvailable(GEMMA_ROOT);

if (!PRODUCT_PRESENT) {
  console.warn(
    "[karume] outputs/series/gemma4-qat-e2b-product/ が無いため PLE GPU 常駐の golden 門を SKIP。" +
      "tools/export-recipes で python -m gemma4.export_product_qat を回して作成する。",
  );
}
if (!QAT_PRESENT || !GEMMA_PRESENT) {
  console.warn(
    "[karume] models/karume-gemma4{,-qat}/ が無いため PLE GPU 常駐の生成 parity を SKIP。" +
      "tools/export-recipes の dist.py で作成する。",
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

Deno.test({
  name: "PLE GPU 常駐: gather が golden ともホスト経路とも u32 完全一致（実GPU）",
  ignore: !PRODUCT_PRESENT || !GPU_AVAILABLE,
  fn: async () => {
    const { index, openBlock } = await openSeriesPle(PRODUCT_ROOT);
    const probe = parseSafetensors(
      (await Deno.readFile(new URL(PLE_PROBE_FILE, PRODUCT_ROOT))).buffer,
    );
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
        entry: "Gemma4QatPipeline",
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
        const reference = await host.gather([...tokens]) as Tensor & { data: Float32Array };
        assertBitsEqual(actual, reference.data, "GPU 内 gather とホスト gather");
      } finally {
        await resident.dispose();
      }
    } finally {
      host.dispose();
      gpu.destroy();
    }
  },
});

/** 1 ターンの greedy 生成 id 列（停止 token も含めて観測席で拾う）。 */
const generateIds = async (
  open: () => Promise<{ chat: Gemma4Pipeline["chat"]; dispose: () => Promise<void> }>,
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
  Deno.test({
    name: `PLE GPU 常駐: QAT ${model} の greedy 生成 id 列が host と完全一致（実GPU）`,
    ignore: !QAT_PRESENT || !GPU_AVAILABLE,
    fn: async () => {
      const source = denoDirectory(QAT_ROOT);
      const expected = await generateIds(() => Gemma4QatPipeline.fromPretrained(source, { model }));
      assertEquals(
        expected.length,
        GREEDY_STEPS,
        "host 経路が刻みぶん生成していない（parity が空振りする）",
      );
      const actual = await generateIds(() =>
        Gemma4QatPipeline.fromPretrained(source, { model, pleResidency: "gpu" })
      );
      assertEquals(actual, expected, "GPU 常駐席の生成 id 列");
    },
  });
}

Deno.test({
  name: "PLE GPU 常駐: 通常 Gemma 4 E2B は載れば parity・載らなければ不足を名乗る（実GPU）",
  ignore: !GEMMA_PRESENT || !GPU_AVAILABLE,
  fn: async () => {
    const source = denoDirectory(GEMMA_ROOT);
    // 索引は配布形ミラーの `model` 容器の資産（ADR 0109 決定 4）— 開くのは part 0 と索引だけ。
    const { index } = await openGemma4Ple(GEMMA_ROOT);
    const bytes = gemma4PleGpuBytes(index);
    const gpu = await acquireGpu();
    const fits = gpu.limits.maxStorageBufferBindingSize >= bytes.values &&
      gpu.limits.maxBufferSize >= bytes.values;
    gpu.destroy();
    if (!fits) {
      // MUST: 黙ってホスト経路へ退避しない（席を要求した利用者に不足を返す）。
      await assertRejects(
        () => Gemma4Pipeline.fromPretrained(source, { pleResidency: "gpu" }),
        Error,
        "maxStorageBufferBindingSize",
      );
      return;
    }
    const expected = await generateIds(() => Gemma4Pipeline.fromPretrained(source, {}));
    assertEquals(
      expected.length,
      GREEDY_STEPS,
      "host 経路が刻みぶん生成していない（parity が空振りする）",
    );
    const actual = await generateIds(() =>
      Gemma4Pipeline.fromPretrained(source, { pleResidency: "gpu" })
    );
    assertEquals(actual, expected, "GPU 常駐席の生成 id 列");
  },
});

// 温度 > 0 の decode は greedy 経路を通らず通常 run（= `enqueueRead`）で走る。gather と target を
// 同じ batch に積むので、GPU 常駐席が余分に払うフェンスは prefill の chunk ぶんだけになる
// （ADR 0054 追記〈グラフ出力の一括読み戻し〉— 席の残件 ①）。
Deno.test({
  name:
    "PLE GPU 常駐: 既定サンプラー経路の decode も host と同じ id 列で、余分なフェンスは prefill ぶんだけ（実GPU）",
  ignore: !QAT_PRESENT || !GPU_AVAILABLE,
  fn: async () => {
    const source = denoDirectory(QAT_ROOT);
    const generate = async (
      pleResidency: "host" | "gpu",
    ): Promise<{ ids: number[]; fences: number }> => {
      const gpu = await acquireGpu();
      try {
        const pipeline = await Gemma4QatPipeline.fromPretrained(source, {
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
    assertEquals(resident.fences - host.fences, 1, `host ${host.fences} / gpu ${resident.fences}`);
  },
});
