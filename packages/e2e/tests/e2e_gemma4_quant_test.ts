/**
 * Checks that the quant selection and the explicit options over the same distribution weights
 * reach the actual GEMV and fusion settings.
 */
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

/**
 * The number of SRQs that become packed in a QAT E2B decode plan (the same value as
 * e2e_*_fusion_counts_test.ts).
 */
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
      `[karume] ${root.pathname} is missing (or not karume/5), so the quant real-GPU check is SKIPPED`,
    );
  }
  describe.skipIf(!available || !GPU_AVAILABLE)(
    `${family}: picks the GEMV from the quant definition (real GPU)`,
    () => {
      it("keeps the precedence of default quant, reference quant and explicit override", async () => {
        const original = await readTextFile(new URL("karume.json", root));
        const parsed = parseManifest(original);
        const raw: unknown = JSON.parse(original);
        assert(isRecord(raw) && isRecord(raw.models));
        const entry = raw.models.e2b;
        assert(isRecord(entry) && isRecord(entry.quants));
        const reference = parsed.models.e2b.quants.i4;
        assertEquals(reference.session, {}, "the reference i4's definition has changed");
        const fastSession = {
          linearGemvReduce: "parallel",
          fuseRmsNormAdd: true,
          ...(family === "gemma4-qat"
            ? { fuseLinearStaticQuantize: true, packedStaticQuantize: true }
            : {}),
        } as const;
        // The quant preset declarations of the replaced manifest (also the material of the key
        // that looks up the bundle's census table).
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
        // Only the manifest is separated. The original distribution and weights are not rewritten.
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
              // i4-gemvpar is a quant that declares only parallel, so packed is not on either.
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
              // Packed matches the SRQs left standalone, so it is on independently of the
              // linear→SRQ fusion.
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
              // This switches to sequential, so packed is turned off with an explicit false too
              // (even a true from the quant is rejected because it requires parallel).
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
            // NOTE: the census is `lastRunPipelines`, which does not depend on measurement. With
            // `onRunDiagnostics` attached, the on-GPU greedy output path is not taken
            // (`greedyOutput` in src/gemma/pipeline.ts) — the census below is that of the path with
            // the hook, and this premise holds even without measurement.
            // The effective bundle (the replaced quant preset's declaration + the explicit options,
            // passed through the family's merge).
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
                // Checks the effective options (the merge of the quant preset's declaration and the
                // explicit options) from both sides — a non-reference value has at least one
                // variant, a reference value none (a packed variant's key is an existing key with a
                // suffix — ADR 0105 decision 4; the predicate is in
                // tests/helpers/pipeline-census.ts).
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
                // If the bundle's census table has a row, check the exact counts per phase and per
                // run (if some shapes drop out of the parallel GEMV eligibility table and fall to
                // sequential, the at-least-one check above passes but this one goes red).
                if (row !== undefined) {
                  const checked = assertRowCensus(row, byPhase, gemmaComponentOf, mode.name);
                  assert(
                    checked > 0,
                    `${mode.name}: no run was compared against the table's expectation`,
                  );
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
                // Exact agreement between speculative and non-speculative decoding is guaranteed
                // only with sequential (the "speculative decoding" section of docs/limitations.md).
                // The default parallel has a different QK reduction order in decode and verify, and
                // argmax can split on tokens with close values, so this switches to the setting
                // whose agreement the check relies on.
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
                    // Check the stop reason first. An early stop on EOS or the like splits the
                    // length regardless of the comparison, so a red reads as "early stop".
                    assertEquals(stop.reason, "max-tokens");
                    assertEquals(ids.length, 32);
                    if (speculative === "always") assert((stop.speculation?.cycles ?? 0) > 0);
                    outputs.push(ids);
                  } finally {
                    await sequence.dispose();
                  }
                }
                assertEquals(outputs[1], outputs[0], "fast quant, speculative vs non-speculative");
                for (const phase of ["decode", "draft", "verify"]) {
                  const phaseRuns = phaseCensus.get(phase);
                  assert(phaseRuns !== undefined, phase);
                  // The RMS fusion is checked on the target's decode/verify.
                  assertSeatsApplied(
                    mergeCensus(phaseRuns, phase),
                    {
                      linearGemvReduce: "parallel",
                      ...(phase === "draft" ? {} : { fuseRmsNormAdd: true }),
                    },
                    phase,
                  );
                }
                // If the bundle's census table (the default quant's declaration + the explicit
                // sequential) has a row, check the exact counts per phase and per run (this also
                // pins how many of the drafter's linears fall back to the sequential GEMV).
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
                  assert(
                    checked > 0,
                    "speculative: no run was compared against the table's expectation",
                  );
                }
              } finally {
                await pipeline.dispose();
              }
            } finally {
              gpu.destroy();
            }
          }
          // slice(1) because part 0 = the descriptor is the very input of admission, and the
          // contract fetches it before the check (phase 1 of
          // packages/models/src/hub/components.ts). What may be touched only after the check is
          // part 1 onward and the assets (tokenizer), and those all go into this set.
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
                  `fetching a weight part or asset under an invalid run configuration: ${path}`,
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
  // The e2e check of packed int8 activations (ADR 0105) — on the real QAT E2B distribution
  // weights it pins that **the greedy token id sequences are identical** with the option on / off,
  // and the pairing counts (decode 210 / prefill 0).
  //
  // The unit u32 identity check (the runtime's `gpu_packed_static_quantize_test.ts`) covers only
  // one kernel. The real graph mixes 210 SRQs with 275 fused epilogues, so a shape where even one
  // pairing is off (packed only on the producer side, or only on the consumer side) shows up only
  // here.
  if (family === "gemma4-qat") {
    describe.skipIf(!available || !GPU_AVAILABLE)(
      "gemma4-qat: packed int8 activations (ADR 0105, real GPU)",
      () => {
        it("the 64-token greedy id sequences with the option on / off are identical, and only the decode plan matches 70", async () => {
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
                  // Check the stop reason first (an early stop splits the id comparison by length).
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
          assertEquals(outputs[0], outputs[1], "greedy id sequences with the option on / off");
          assertEquals(packedByPhase["off:decode"], [0], "decode plan with the option off");
          assertEquals(packedByPhase["off:prefill"], [0], "prefill plan with the option off");
          assertEquals(
            packedByPhase["on:decode"],
            [DECODE_PACKED],
            "decode plan with the option on",
          );
          assertEquals(packedByPhase["on:prefill"], [0], "prefill plan with the option on");
        });
      },
    );
  }
}
