/** 同じ配布重みのquant選択と明示指定が、実際のGEMV・融合設定へ届くことを検証する。 */
import {
  makeTempDir,
  readDir,
  readFile,
  readTextFile,
  remove,
  symlink,
  writeTextFile,
} from "@karume/test-support/fs";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { localDirectory, parseManifest } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import { acquireGpu, type SessionDiagnostics } from "@karume/runtime";
import { chatPrompt, Pipeline, type RunPhase } from "../../models/gemma.ts";
import * as gemma4QatMobile from "../../models/gemma4-qat-mobile.ts";
import { assertSeatsApplied, mergeCensus } from "../../runtime/tests/helpers/pipeline-census.ts";
import type { SessionOverrides } from "../../models/src/session/options.ts";
import {
  assertRowCensus,
  censusRowOf,
  effectiveSessionOptions,
  gemmaComponentOf,
} from "../../models/tests/helpers/census-table.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** QAT E2B decode 計画で packed になる SRQ の本数（e2e_*_fusion_counts_test.ts と同じ値）。 */
const DECODE_PACKED = 70;

/**
 * The mirror each family (= manifest pipeline name) is read from. Pipeline `gemma4`'s int4 presets
 * are the q4_0 QAT line's (`karume-gemma4-qat` — same pipeline, an `i4` E2B with its QAT drafter):
 * the regular line ships `bf16` alone (2026-11-04), so it has no quant seat left to swap here.
 */
const MIRRORS = { gemma4: "karume-gemma4-qat", "gemma4-qat": "karume-gemma4-qat-mobile" } as const;

for (const family of ["gemma4", "gemma4-qat"] as const) {
  const root = new URL(`../../../models/${MIRRORS[family]}/`, import.meta.url);
  const available = mirrorAvailable(root);
  if (!available) {
    console.warn(
      `[karume] ${root.pathname} が無い（か karume/5 でない）ためquantの実GPU検査をSKIP`,
    );
  }
  describe.skipIf(!available || !GPU_AVAILABLE)(
    `${family}: quant定義からGEMVを選ぶ（実GPU）`,
    () => {
      it("既定quant・参照quant・明示上書きの優先順位を守る", async () => {
        const original = await readTextFile(new URL("karume.json", root));
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
        // 差し替えた manifest の quant 席の宣言（束の census 表を引く鍵の材料にもする）。
        const quantSessions = {
          i4: reference.session,
          "i4-gemvpar": { linearGemvReduce: "parallel" },
          "i4-fast": fastSession,
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
                "i4-gemvpar": { ...reference, session: quantSessions["i4-gemvpar"] },
                "i4-fast": { ...reference, session: quantSessions["i4-fast"] },
                unsupported: { ...reference, session: { linearCompute: "f16" } },
              },
            },
          },
        };
        // manifestだけ独立させる。元の配布形と重みは書き換えない。
        const temp = await makeTempDir({ prefix: "gemma-quant-test-" });
        try {
          for await (const file of readDir(root)) {
            if (file.name !== "karume.json") {
              await symlink(new URL(file.name, root), `${temp}/${file.name}`);
            }
          }
          await writeTextFile(`${temp}/karume.json`, JSON.stringify(manifest), {
            createNew: true,
          });
          const runs = new Map<string, { text: string; stop: unknown }>();
          for (const mode of [
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
          ] as const) {
            // NOTE: census は計測に依らない `lastRunPipelines`。`onRunDiagnostics` を付けると GPU 上の
            // greedy 出力経路は外れる（src/gemma/pipeline.ts の `greedyOutput`）— 下の census は
            // フック有りの経路のもので、この前提は計測を外しても変わらない。
            // 実効の束（差し替えた quant 席の宣言 + 明示指定を家族の合成に通した値）。
            const {
              quant = "i4-fast",
              ...overrides
            }: { readonly quant?: keyof typeof quantSessions } & SessionOverrides = mode.options;
            const row = censusRowOf(
              family,
              "e2b",
              effectiveSessionOptions(family, quantSessions[quant], overrides, mode.name),
            );
            const gpu = await acquireGpu();
            const census: SessionDiagnostics["lastRunPipelines"][] = [];
            const byPhase = new Map<string, SessionDiagnostics["lastRunPipelines"][]>();
            try {
              const options = {
                gpu,
                model: "e2b",
                chunkLength: 32,
                maxResidentPleBytes: 0,
                ...mode.options,
                onRunDiagnostics: (d: SessionDiagnostics, phase: RunPhase) => {
                  census.push(d.lastRunPipelines);
                  byPhase.set(phase.kind, [...(byPhase.get(phase.kind) ?? []), d.lastRunPipelines]);
                },
              } satisfies gemma4QatMobile.FromPretrainedOptions;
              const source = nodeDirectory(temp);
              const pipeline =
                family === "gemma4"
                  ? await Pipeline.fromPretrained(source, options)
                  : await gemma4QatMobile.Pipeline.fromPretrained(source, options);
              try {
                const stream = pipeline.chat(
                  [
                    {
                      role: "user",
                      content: "Write a numbered list of twenty tips for learning a new language.",
                    },
                  ],
                  { capacity: 128, maxNewTokens: 24, sampler: { temperature: 0 } },
                );
                const text = await stream.text();
                const stop = await stream.done;
                assert(text.length > 0);
                assert(stop.tokens > 1);
                runs.set(mode.name, { text, stop });
                const merged = mergeCensus(census, mode.name);
                assert(merged.some((row) => row.key.startsWith("linear_gemv")));
                // 実効の席（quant 席の宣言 + 明示指定の合成結果）を両側で見る — 非参照値は
                // 変種が 1 本以上・参照値は変種が 0 本（packed 変種のキーは既存キーの末尾に
                // 断片が付く — ADR 0105 決定 4。述語は tests/helpers/pipeline-census.ts）。
                assertSeatsApplied(
                  merged,
                  {
                    linearGemvReduce: mode.parallel ? "parallel" : "sequential",
                    fuseRmsNormAdd: mode.rms,
                    fuseLinearStaticQuantize: mode.srq,
                    packedStaticQuantize: mode.packed,
                  },
                  mode.name,
                );
                // 束の census 表に行があれば、相ごと・run ごとに本数ちょうどで見る（並列 GEMV の
                // 適格表から一部の形が外れて逐次へ落ちると、上の 1 本以上の検査は通るがここで赤）。
                if (row !== undefined) {
                  const checked = assertRowCensus(row, byPhase, gemmaComponentOf, mode.name);
                  assert(checked > 0, `${mode.name}: 表の期待と突き合わせた run が 1 本も無い`);
                }
              } finally {
                await pipeline.dispose();
              }
            } finally {
              gpu.destroy();
            }
          }
          for (const name of [
            "parallel-reference",
            "explicit-fast",
            "disable-rms",
            "disable-srq",
            "disable-both",
          ]) {
            assertEquals(runs.get("default"), runs.get(name), name);
          }
          assertEquals(runs.get("reference"), runs.get("override"));
          if (family === "gemma4") {
            const gpu = await acquireGpu();
            const phaseCensus = new Map<string, SessionDiagnostics["lastRunPipelines"][]>();
            try {
              const pipeline = await Pipeline.fromPretrained(nodeDirectory(temp), {
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
                  const phaseRuns = phaseCensus.get(phase.kind) ?? [];
                  phaseRuns.push(d.lastRunPipelines);
                  phaseCensus.set(phase.kind, phaseRuns);
                },
              });
              try {
                const prompt = chatPrompt(pipeline.tokenizer, [
                  {
                    role: "user",
                    content: "Write a numbered list of twenty tips for learning a new language.",
                  },
                ]);
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
                  const phaseRuns = phaseCensus.get(phase);
                  assert(phaseRuns !== undefined, phase);
                  // RMS融合の適用はtarget側のdecode/verifyで見る。
                  assertSeatsApplied(
                    mergeCensus(phaseRuns, phase),
                    {
                      linearGemvReduce: "parallel",
                      ...(phase === "draft" ? {} : { fuseRmsNormAdd: true }),
                    },
                    phase,
                  );
                }
                // 束の census 表（既定 quant の宣言 + 明示の sequential）に行があれば、相ごと・
                // run ごとに本数ちょうどで見る（drafter の逐次 GEMV への縮退本数もここで固定）。
                const row = censusRowOf(
                  family,
                  "e2b",
                  effectiveSessionOptions(
                    family,
                    quantSessions["i4-fast"],
                    {
                      stateAttentionReduce: "sequential",
                    },
                    "speculative",
                  ),
                );
                if (row !== undefined) {
                  const checked = assertRowCensus(
                    row,
                    phaseCensus,
                    gemmaComponentOf,
                    "speculative",
                  );
                  assert(checked > 0, "speculative: 表の期待と突き合わせた run が 1 本も無い");
                }
              } finally {
                await pipeline.dispose();
              }
            } finally {
              gpu.destroy();
            }
          }
          // slice(1)なのは、part 0 = descriptor が admission の入力そのもので、門より前に
          // 取る契約だから（packages/models/src/hub/components.ts の相 1）。門の後にしか
          // 触れてはいけないのは part 1 以降と assets（tokenizer）で、そちらをまとめて
          // この集合に入れる。
          const heavyPaths = new Set([
            ...Object.values(parsed.models.e2b.weights).flatMap((entry) =>
              Object.values(entry).flatMap((weights) =>
                weights.container.parts.slice(1).map((ref) => ref.path),
              ),
            ),
            ...Object.values(parsed.models.e2b.assets).map((ref) => ref.path),
          ]);
          assert(heavyPaths.size > 0);
          const requested: string[] = [];
          const noWeightsSource = localDirectory(
            {
              readFile: async (path) => {
                requested.push(path);
                assert(
                  !heavyPaths.has(path),
                  `不正な実行設定で重みの part・資産を取得している: ${path}`,
                );
                return await readFile(`${temp}/${path}`);
              },
            },
            { label: "invalid-gemma-quant" },
          );
          const loadUnsupported = () =>
            family === "gemma4"
              ? Pipeline.fromPretrained(noWeightsSource, { quant: "unsupported" })
              : gemma4QatMobile.Pipeline.fromPretrained(noWeightsSource, { quant: "unsupported" });
          await assertRejects(loadUnsupported, Error, "session.linearCompute is not supported");
          if (family === "gemma4-qat") {
            await assertRejects(
              () =>
                gemma4QatMobile.Pipeline.fromPretrained(noWeightsSource, {
                  linearGemvReduce: "sequential",
                }),
              Error,
              "supports only linearGemvReduce: parallel / linearCompute: f32",
            );
          }
          assert(requested.includes("karume.json"));
          assert(!requested.some((path) => heavyPaths.has(path)));
        } finally {
          await remove(temp, { recursive: true });
        }
      });
    },
  );
  // Kept as its own case rather than one more mode of the loop above: that loop compares
  // 24-token chat text, while this one fixes 64-token greedy id equality and the exact
  // per-phase packed counts read from lastRunFusions.
  // packed int8 活性（ADR 0105）の e2e 門 — QAT E2B の実配布重みで、席 on / off の
  // **greedy な token id 列が完全一致**することと、対付けの本数（decode 210 / prefill 0）を固定する。
  //
  // 単体の u32 一致門（runtime の `gpu_packed_static_quantize_test.ts`）はカーネル 1 本ぶんしか
  // 見ない。実グラフでは 210 本の SRQ と 275 本の融合エピローグが混ざるので、対付けが 1 本でも
  // 外れた形（生産側だけ packed・消費側だけ packed）はここでしか出ない。
  if (family === "gemma4-qat") {
    describe.skipIf(!available || !GPU_AVAILABLE)(
      "gemma4-qat: packed int8 活性（ADR 0105・実GPU）",
      () => {
        it("席 on / off の 64 token greedy id 列が完全一致し、decode 計画だけが 70 本を掴む", async () => {
          const outputs: number[][] = [];
          const packedByPhase: Record<string, number[]> = {};
          for (const packedStaticQuantize of [false, true]) {
            const gpu = await acquireGpu();
            try {
              const phases: Record<string, number[]> = {};
              const pipeline = await gemma4QatMobile.Pipeline.fromPretrained(nodeDirectory(root), {
                gpu,
                model: "e2b",
                chunkLength: 32,
                maxResidentPleBytes: 0,
                packedStaticQuantize,
                onRunDiagnostics: (d: SessionDiagnostics, phase) => {
                  (phases[phase.kind] ??= []).push(d.lastRunFusions?.packedStaticQuantize ?? -1);
                },
              });
              try {
                const prompt = chatPrompt(pipeline.tokenizer, [
                  {
                    role: "user",
                    content: "Write a numbered list of twenty tips for learning a new language.",
                  },
                ]);
                const sequence = await pipeline.sequence({ capacity: 256 });
                try {
                  const stream = sequence.generate({
                    prompt,
                    maxNewTokens: 64,
                    sampler: { temperature: 0 },
                  });
                  const ids: number[] = [];
                  for await (const event of stream) {
                    if (event.kind === "token") ids.push(event.id);
                  }
                  // 先に停止理由を見る（早期停止だと id 列の一致比較が長さで割れる）。
                  const stop = await stream.done;
                  assertEquals(stop.reason, "max-tokens");
                  assertEquals(ids.length, 64);
                  outputs.push(ids);
                } finally {
                  await sequence.dispose();
                }
              } finally {
                await pipeline.dispose();
              }
              const label = packedStaticQuantize ? "on" : "off";
              for (const [kind, counts] of Object.entries(phases)) {
                packedByPhase[`${label}:${kind}`] = [...new Set(counts)];
              }
            } finally {
              gpu.destroy();
            }
          }
          assertEquals(outputs[0], outputs[1], "席 on / off の greedy id 列");
          assertEquals(packedByPhase["off:decode"], [0], "席 off の decode 計画");
          assertEquals(packedByPhase["off:prefill"], [0], "席 off の prefill 計画");
          assertEquals(packedByPhase["on:decode"], [DECODE_PACKED], "席 on の decode 計画");
          assertEquals(packedByPhase["on:prefill"], [0], "席 on の prefill 計画");
        });
      },
    );
  }
}
