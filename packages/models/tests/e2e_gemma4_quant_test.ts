/** 同じ配布重みのquant選択と明示指定が、実際のGEMV・融合設定へ届くことを検証する。 */
import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { localDirectory, parseManifest } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import { acquireGpu, type SessionDiagnostics } from "@karume/runtime";
import { gemma4ChatPrompt, Gemma4Pipeline } from "../gemma.ts";
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
        const fastSession = {
          linearGemvReduce: "parallel",
          fuseRmsNormAdd: true,
          ...(family === "gemma4-qat"
            ? { fuseLinearStaticQuantize: true, packedStaticQuantize: true }
            : {}),
        } as const;
        const manifest = {
          ...raw,
          models: {
            ...raw.models,
            e2b: {
              ...entry,
              defaultQuant: "i4-fast",
              quants: {
                ...entry.quants,
                "i4-gemvpar": { ...reference, session: { linearGemvReduce: "parallel" } },
                "i4-fast": { ...reference, session: fastSession },
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
              {
                name: "default",
                options: {},
                parallel: true,
                rms: true,
                srq: family === "gemma4-qat",
                packed: family === "gemma4-qat",
              },
              {
                // i4-gemvpar は parallel だけを宣言する quant なので packed も立たない。
                name: "parallel-reference",
                options: { quant: "i4-gemvpar" },
                parallel: true,
                rms: false,
                srq: false,
                packed: false,
              },
              {
                name: "explicit-fast",
                options: { quant: "i4", ...fastSession },
                parallel: true,
                rms: true,
                srq: family === "gemma4-qat",
                packed: family === "gemma4-qat",
              },
              {
                name: "disable-rms",
                options: { fuseRmsNormAdd: false },
                parallel: true,
                rms: false,
                srq: family === "gemma4-qat",
                packed: family === "gemma4-qat",
              },
              {
                // packed が掴むのは素のまま残った SRQ なので、linear→SRQ 融合とは独立に立つ。
                name: "disable-srq",
                options: { fuseLinearStaticQuantize: false },
                parallel: true,
                rms: true,
                srq: false,
                packed: family === "gemma4-qat",
              },
              {
                name: "disable-both",
                options: { fuseRmsNormAdd: false, fuseLinearStaticQuantize: false },
                parallel: true,
                rms: false,
                srq: false,
                packed: family === "gemma4-qat",
              },
              {
                name: "reference",
                options: { quant: "i4" },
                parallel: false,
                rms: false,
                srq: false,
                packed: false,
              },
              {
                // sequential へ倒すので packed も明示 false で降ろす（quant 由来の true でも
                // parallel 必須の拒否は効く）。
                name: "override",
                options: {
                  linearGemvReduce: "sequential",
                  fuseRmsNormAdd: false,
                  fuseLinearStaticQuantize: false,
                  packedStaticQuantize: false,
                },
                parallel: false,
                rms: false,
                srq: false,
                packed: false,
              },
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
                  [...keys].some((key) => key.endsWith(":static-quantize:v1")),
                  mode.srq,
                );
                assertEquals(
                  [...keys].some((key) => key.startsWith("rms_norm_add:")),
                  mode.rms,
                );
                assertEquals(
                  [...keys].some((key) => key.startsWith("linear_gemv_parallel")),
                  mode.parallel,
                  mode.name,
                );
                // packed 変種のキーは既存キーの末尾に断片が付く（ADR 0105 決定 4）。
                assertEquals(
                  [...keys].some((key) => key.endsWith(":packed-x-i8")),
                  mode.packed,
                  mode.name,
                );
              } finally {
                await pipeline.dispose();
              }
            } finally {
              gpu.destroy();
            }
          }
          for (
            const name of [
              "parallel-reference",
              "explicit-fast",
              "disable-rms",
              "disable-srq",
              "disable-both",
            ]
          ) {
            assertEquals(runs.get("default"), runs.get(name), name);
          }
          assertEquals(runs.get("reference"), runs.get("override"));
          if (family === "gemma4") {
            const gpu = await acquireGpu({ gpuTiming: true });
            const phaseKeys = new Map<string, Set<string>>();
            try {
              const pipeline = await Gemma4Pipeline.fromPretrained(denoDirectory(temp), {
                gpu,
                model: "e2b",
                chunkLength: 32,
                maxResidentPleBytes: 0,
                speculative: { k: 3 },
                // 投機/非投機の厳密一致はsequential席でだけ保証される（docs/limitations.md
                // 「投機デコード」節）。既定のparallelはdecodeとverifyでQKの縮約順が違い、
                // 近い値のtokenでargmaxが割れうるので、一致を門にする席へ倒す。
                stateAttentionReduce: "sequential",
                onRunDiagnostics: (d, phase) => {
                  const keys = phaseKeys.get(phase.kind) ?? new Set<string>();
                  for (const row of d.lastRunTiming?.entries ?? []) keys.add(row.key);
                  phaseKeys.set(phase.kind, keys);
                },
              });
              try {
                const prompt = gemma4ChatPrompt(pipeline.tokenizer, [{
                  role: "user",
                  content: "Write a numbered list of twenty tips for learning a new language.",
                }]);
                const outputs: number[][] = [];
                for (const speculative of [false, "always"] as const) {
                  const sequence = await pipeline.sequence({ capacity: 128, speculative });
                  try {
                    const stream = sequence.generate({
                      prompt,
                      maxNewTokens: 32,
                      sampler: { temperature: 0 },
                    });
                    const ids: number[] = [];
                    for await (const event of stream) {
                      if (event.kind === "token") ids.push(event.id);
                    }
                    const stop = await stream.done;
                    // 先に停止理由を見る。EOSなどで早く止まると一致比較とは無関係に長さが
                    // 割れるので、赤の原因が「早期停止」だと読めるようにしておく。
                    assertEquals(stop.reason, "max-tokens");
                    assertEquals(ids.length, 32);
                    if (speculative === "always") assert((stop.speculation?.cycles ?? 0) > 0);
                    outputs.push(ids);
                  } finally {
                    await sequence.dispose();
                  }
                }
                assertEquals(outputs[1], outputs[0], "高速quantの投機/非投機");
                for (const phase of ["decode", "draft", "verify"]) {
                  const keys = phaseKeys.get(phase);
                  assert(keys !== undefined, phase);
                  assert([...keys].some((k) => k.startsWith("linear_gemv_parallel")), phase);
                  // RMS融合の適用はtarget側のdecode/verifyで見る。
                  if (phase !== "draft") {
                    assert([...keys].some((k) => k.startsWith("rms_norm_add:")), phase);
                  }
                }
              } finally {
                await pipeline.dispose();
              }
            } finally {
              gpu.destroy();
            }
          }
          // slice(1)なのは、先頭shard = グラフshardがadmissionの入力そのもので、門より前に
          // 取る契約だから（packages/models/src/hub/components.ts の streamAssets 相 1）。
          // 門の後にしか触れてはいけないのは2本目以降のshardとassets（tokenizer・PLE sidecar）で、
          // そちらをまとめてこの集合に入れる。
          const heavyPaths = new Set([
            ...Object.values(parsed.models.e2b.weights).flatMap((entry) =>
              Object.values(entry).flatMap((files) => files.shards.slice(1).map((ref) => ref.path))
            ),
            ...Object.values(parsed.models.e2b.assets).map((ref) => ref.path),
          ]);
          assert(heavyPaths.size > 0);
          const requested: string[] = [];
          const noWeightsSource = localDirectory({
            readFile: async (path) => {
              requested.push(path);
              assert(
                !heavyPaths.has(path),
                `不正な実行設定で重みshard・資産を取得している: ${path}`,
              );
              return await Deno.readFile(`${temp}/${path}`);
            },
          }, { label: "invalid-gemma-quant" });
          const loadUnsupported = () =>
            family === "gemma4"
              ? Gemma4Pipeline.fromPretrained(noWeightsSource, { quant: "unsupported" })
              : Gemma4QatPipeline.fromPretrained(noWeightsSource, { quant: "unsupported" });
          await assertRejects(loadUnsupported, Error, "session.linearComputeは未対応");
          if (family === "gemma4-qat") {
            await assertRejects(
              () =>
                Gemma4QatPipeline.fromPretrained(noWeightsSource, {
                  linearGemvReduce: "sequential",
                }),
              Error,
              "parallelが必要",
            );
          }
          assert(requested.includes("karume.json"));
          assert(!requested.some((path) => heavyPaths.has(path)));
        } finally {
          await Deno.remove(temp, { recursive: true });
        }
      });
    },
  });
}
