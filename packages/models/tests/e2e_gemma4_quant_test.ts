/** 同じ配布重みのquant選択と明示指定が、実際のGEMV加算方式へ届くことを検証する。 */
import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { parseManifest } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import { acquireGpu, DEFAULT_SUBMIT_POLICY, type SessionDiagnostics } from "@karume/runtime";
import { Gemma4Pipeline } from "../gemma.ts";
import { type Gemma4QatFromPretrainedOptions, Gemma4QatPipeline } from "../gemma4-qat.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

for (const family of ["gemma4", "gemma4-qat"] as const) {
  const root = new URL(`../../../models/karume-${family}/`, import.meta.url);
  let available = false;
  try {
    available = Deno.statSync(new URL("karume.json", root)).isFile;
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
  }
  if (!available) console.warn(`[karume] ${root.pathname} が無いためquantの実GPU検査をSKIP`);
  describe({
    name: `${family}: quant定義からGEMVを選ぶ（実GPU）`,
    ignore: !available || !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
    fn: () => {
      it("既定quant・参照quant・明示上書きの優先順位を守る", async () => {
        const original = await Deno.readTextFile(new URL("karume.json", root));
        const parsed = parseManifest(original);
        const raw: unknown = JSON.parse(original);
        assert(isRecord(raw) && isRecord(raw.models));
        const entry = raw.models.e2b;
        assert(isRecord(entry) && isRecord(entry.quants));
        const reference = parsed.models.e2b.quants.i4;
        assertEquals(reference.session, {}, "参照i4の定義が変わっている");
        const manifest = {
          ...raw,
          models: {
            ...raw.models,
            e2b: {
              ...entry,
              defaultQuant: "i4-gemvpar",
              quants: {
                ...entry.quants,
                "i4-gemvpar": { ...reference, session: { linearGemvReduce: "parallel" } },
                unsupported: { ...reference, session: { linearCompute: "f16" } },
              },
            },
          },
        };
        // manifestだけ独立させる。元の配布形と重みは書き換えない。
        const temp = await Deno.makeTempDir({ prefix: "gemma-quant-test-" });
        try {
          for await (const file of Deno.readDir(root)) {
            if (file.name !== "karume.json") {
              await Deno.symlink(new URL(file.name, root), `${temp}/${file.name}`);
            }
          }
          await Deno.writeTextFile(`${temp}/karume.json`, JSON.stringify(manifest), {
            createNew: true,
          });
          const runs = new Map<string, { text: string; stop: unknown }>();
          for (
            const mode of [
              { name: "default", options: {}, parallel: true },
              {
                name: "fused",
                options: {
                  fuseRmsNormAdd: true,
                  submitPolicy: { ...DEFAULT_SUBMIT_POLICY, maxChunkSize: 768 },
                },
                parallel: true,
              },
              {
                name: "explicit-parallel",
                options: { linearGemvReduce: "parallel" },
                parallel: true,
              },
              { name: "reference", options: { quant: "i4" }, parallel: false },
              { name: "override", options: { linearGemvReduce: "sequential" }, parallel: false },
            ] as const
          ) {
            const gpu = await acquireGpu({ gpuTiming: true });
            const keys = new Set<string>();
            try {
              const options = {
                gpu,
                model: "e2b",
                chunkLength: 32,
                maxResidentPleBytes: 0,
                ...mode.options,
                onRunDiagnostics: (d: SessionDiagnostics) => {
                  for (const row of d.lastRunTiming?.entries ?? []) keys.add(row.key);
                },
              } satisfies Gemma4QatFromPretrainedOptions;
              const source = denoDirectory(temp);
              const pipeline = family === "gemma4"
                ? await Gemma4Pipeline.fromPretrained(source, options)
                : await Gemma4QatPipeline.fromPretrained(source, options);
              try {
                const stream = pipeline.chat([
                  {
                    role: "user",
                    content: "Write a numbered list of twenty tips for learning a new language.",
                  },
                ], { capacity: 128, maxNewTokens: 24, sampler: { temperature: 0 } });
                const text = await stream.text();
                const stop = await stream.done;
                assert(text.length > 0);
                assert(stop.tokens > 1);
                runs.set(mode.name, { text, stop });
                assert([...keys].some((key) => key.startsWith("linear_gemv")));
                assertEquals(
                  [...keys].some((key) => key.startsWith("rms_norm_add:")),
                  mode.name === "fused",
                );
                assertEquals(
                  [...keys].some((key) => key.startsWith("linear_gemv_parallel")),
                  mode.parallel,
                  mode.name,
                );
              } finally {
                await pipeline.dispose();
              }
            } finally {
              gpu.destroy();
            }
          }
          assertEquals(runs.get("default"), runs.get("explicit-parallel"));
          assertEquals(runs.get("default"), runs.get("fused"));
          assertEquals(runs.get("reference"), runs.get("override"));
          const loadUnsupported = () =>
            family === "gemma4"
              ? Gemma4Pipeline.fromPretrained(denoDirectory(temp), { quant: "unsupported" })
              : Gemma4QatPipeline.fromPretrained(denoDirectory(temp), { quant: "unsupported" });
          await assertRejects(loadUnsupported, Error, "session.linearComputeは未対応");
        } finally {
          await Deno.remove(temp, { recursive: true });
        }
      });
    },
  });
}
