// Acceptance check that builds the real-weight Gemma 4 E2B **from the distribution** — the pass
// line of step 5 (a real download end to end).
//
// Unlike `fromAssets` (step 4's check is `e2e_gemma4_chat_test.ts`), this file makes the load path
// the same one as production: resolve `karume.json` → fetch only the container's descriptor and
// run admission → stream the weight parts in order → lazily read the blocks of the PLE (an asset
// of the `model` container — ADR 0109 decision 4). Four checks:
//
//  ① **Runs to completion**: `fromPretrained` → `chat` runs on the real weights, and the output at
//     (explicit) temperature 0 matches **the same golden** as `e2e_gemma4_chat_test.ts`. This
//     proves the same bytes are read through another path; if it splits, what moved is the
//     distribution or the fetch path
//  ② **Wiring of the default sampler** (ADR 0083 decision 7): the recommended values the
//     distribution declares land as they are on `pipeline.defaultSampler` and become the default
//     when a request omits `sampler`. Even with the declaration baked into `pipelineConfig`, a
//     missing wire would leave ① (the path that sets temperature 0 explicitly) green, so this
//     check is needed
//  ③ **The PLE is not fully resident** (ADR 0085 decision 3): the blocks the index points at are
//     taken lazily as assets of the `model` container, and only the one touched is read. Going
//     back to streaming everything through `fetchAssets` would keep 4.70 GB (bf16) resident
//  ④ **The distribution's declaration matches what is there**: every block the index names exists
//     as is as an asset of the `model` container (the fetch keys have not gone back to "matching
//     by order")
//
// ## Asset and path
//
// The local distribution mirror `models/karume-gemma4/` (assembled by `dist.py --pipeline gemma4`)
// is read through a throwaway server that serves it over **HF-style HTTP**
// (`examples/shared/local-dist-server.ts` — the same one the demo goes through with
// `--source <local path>`). The mirror is outside the repository, so where it is missing the cases
// **SKIP explicitly**.
//
// The seat is the regular line's `bf16` preset (its one preset since 2026-11-04). The census of the
// summation-order seats reads the q4_0 QAT line's mirror (`models/karume-gemma4-qat/`) instead —
// the same `gemma4` pipeline with the same `i4` / `i4-gemvpar` / `i4-fast` declarations, the line
// that keeps them.
//
// NOTE: the test borrows the tool from `examples/` so that there are not two spellings of the
// fake HF server (neither `packages/models/tests/` nor `examples/` is published). The fetch layer
// copies the assets into the **persistent cache** (DENO_DIR on Deno), so one run of this check
// leaves about 8.6 GiB (bf16 E2B) there (clean it with `clearHubCache` of `@karume/hub`).
//
// ## Directory source
//
// The last test reads the same mirror through `nodeDirectory` (`@karume/hub/node`) instead of
// the HTTP server. Two gates: the temperature-0 chat matches the same golden (the local source
// delivers the same bytes: container descriptor, streamed weight parts and the lazy PLE blocks
// all come from that one source), and neither `fetch` nor `CacheStorage` is touched. A local
// source never copies bytes into a cache, so this test leaves nothing in the persistent cache.

import { makeTempDir, readTextFile, readTextFileSync, remove } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { MANIFEST_FILENAME, type ModelEntry, parseManifest } from "@karume/hub";
import { nodeCacheStorage, nodeDirectory } from "@karume/hub/node";
import { acquireGpu, type SessionDiagnostics } from "@karume/runtime";
// MUST: the entry point is taken from the **public surface** (the `./gemma` subpath) — grabbing
// `src/...` directly would keep the check green even if the surface shrank (the acceptance would
// run through a path consumers cannot write).
import { type ChatMessage, Pipeline } from "../../models/gemma.ts";
import { type LocalDistServer, serveLocalDist } from "@karume/test-support/local-dist-server";
import { mergeCensus } from "../../runtime/tests/helpers/pipeline-census.ts";
import type { SessionOverrides } from "../../models/src/session/options.ts";
import {
  assertRowCensus,
  censusRowOf,
  effectiveSessionOptions,
  gemmaComponentOf,
} from "../../models/tests/helpers/census-table.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { allResidentPleBytesOfMirror } from "../../models/tests/helpers/ple-budget.ts";
import {
  GEMMA4_QAT_MIRROR,
  mirrorAvailable,
  openGemma4Ple,
} from "../../models/tests/helpers/gemma-mirror.ts";

const MIRROR_DIR = new URL("../../../models/karume-gemma4/", import.meta.url);
/**
 * The seat the chat goldens below were taken on: the regular line's `bf16` preset (its default
 * and, since 2026-11-04, its one preset). Named so a later default does not move the goldens.
 */
const BF16_SEAT = { quant: "bf16" } as const;

/** The assembly command, ready to paste when the cases SKIP. */
const ASSEMBLE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline gemma4";

/**
 * The budget (bytes) for keeping the PLE resident. Derived from the index = the whole PLE
 * resident, so a conversation that spans token ranges never re-reads. Written as a constant (it
 * used to be 768 MiB), the same number would mean a different block count once an asset
 * generation changes the block width (the "bytes, not counts" of ADR 0085 addendum 2026-09-02
 * applies to the test's constants too). Not evaluated without the mirror (called after the SKIP
 * decision).
 */
const maxResidentPleBytes = (): Promise<number> =>
  allResidentPleBytesOfMirror(MIRROR_DIR, BF16_SEAT);

/**
 * The recommendation of upstream `generation_config.json` (the values the distribution baked into
 * `pipelineConfig.sampler`).
 */
const RECOMMENDED_SAMPLER = { temperature: 1, topK: 64, topP: 0.95 } as const;

/**
 * Acceptance cases. **The same goldens as CASES in `e2e_gemma4_chat_test.ts`** (this proves the
 * same weights and the same static wiring are read through another path, so the values must not
 * diverge).
 */
const CASES = [
  {
    fixture: "single-user",
    maxNewTokens: 24,
    expected: "The capital of France is **Paris**.",
    stop: { reason: "eos", token: 106, tokens: 9 },
  },
] as const;

/**
 * Cases for the directory source: the same golden as `CASES` plus the Japanese case of
 * `e2e_gemma4_chat_test.ts` (same bytes through another source, so the values must not drift).
 */
const DIRECTORY_CASES = [
  ...CASES,
  {
    fixture: "japanese",
    maxNewTokens: 24,
    expected: "2023年時点の東京の人口は約1,400万人です。",
    stop: { reason: "eos", token: 106, tokens: 20 },
  },
] as const;

type ChatFixture = {
  readonly chat: {
    readonly name: string;
    readonly messages: ChatMessage[];
    readonly ids: number[];
  }[];
};

const fixture = JSON.parse(
  await readTextFile(
    new URL("../../models/tests/fixtures/gemma-text/gemma4-chat.json", import.meta.url),
  ),
) as ChatFixture;

const caseOf = (name: string) => {
  const found = fixture.chat.find((row) => row.name === name);
  assert(found !== undefined, `the fixture has no chat case '${name}'`);
  return found;
};

// A machine without the mirror, or with an old major left in place, SKIPs explicitly (the
// helper's MUST — ADR 0109 decision 9).
const AVAILABLE = mirrorAvailable(MIRROR_DIR);
const MANIFEST_TEXT = AVAILABLE
  ? readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR))
  : undefined;

if (!AVAILABLE) {
  console.warn(
    `[karume] the distribution mirror models/karume-gemma4/ is missing, so step 5's end-to-end acceptance is SKIPPED. ` +
      `Assemble it: ${ASSEMBLE_COMMAND}`,
  );
}

test.skipIf(!AVAILABLE)(
  "gemma4 distribution: the manifest's declaration matches the PLE index (no GPU needed)",
  async () => {
    const manifest = parseManifest(MANIFEST_TEXT ?? "");
    const entry = manifest.models[manifest.defaultModel];
    assertEquals(entry.pipeline, { name: "gemma4", major: 1 }, "pipeline contract");
    // The weights roles are exactly two: the product graph + the speculative drafter (`drafter`
    // was added in ADR 0096 step 2). The tokenizer is in the manifest's assets, the PLE is an asset
    // of the `model` container (ADR 0109 decision 4) — more entries here would be the regression
    // "the sidecar went back into weights".
    assertEquals(
      Object.keys(entry.weights),
      ["model", "drafter"],
      "weights are the product graph + the drafter (the PLE is a container asset, the tokenizer is in assets)",
    );
    assertEquals(
      Object.keys(entry.assets),
      ["tokenizer"],
      "the manifest's assets are the tokenizer only",
    );
    // ④ The blocks the index names are the `model` container's assets as they are — the helper
    // (`readGemma4PleIndex`) matches every one, so a mirror with one side rebuilt fails here. The
    // container is the default `bf16` preset's (schema 5 — the bf16 values, no scale table).
    assertEquals(entry.defaultQuant, "bf16", "the default preset");
    const { index: original } = await openGemma4Ple(MIRROR_DIR, BF16_SEAT);
    assert(original.storage === "bf16", `the default preset's PLE storage is ${original.storage}`);
    assert(!("scales" in original), "a schema 5 index has no scale table");
    assertEquals(
      original.values.rowBytes,
      original.layers * original.dim * 2,
      "schema 5: 2 bytes per bf16 value",
    );
    assert(original.values.blocks.length > 0, "the bf16 index names no values block");
    assertEquals(
      original.values.blocks.at(-1)?.stop,
      original.tokens,
      "bf16 values close at tokens",
    );
  },
);

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  "gemma4 distribution: fromPretrained → chat runs to completion on the real GPU and the default sampler is as declared",
  () => {
    let server: LocalDistServer;
    let cacheDir: string;
    let pipeline: Pipeline;
    beforeAll(async () => {
      server = await serveLocalDist(new URL(".", MIRROR_DIR).pathname);
      // The HTTP source prefetches through a CacheStorage; the real on-disk one in a temporary
      // directory keeps the distribution out of RAM and out of any user cache.
      cacheDir = await makeTempDir({ prefix: "karume-gemma4-cache-" });
      pipeline = await Pipeline.fromPretrained(server.source, {
        ...BF16_SEAT,
        caches: nodeCacheStorage(cacheDir),
        maxResidentPleBytes: await maxResidentPleBytes(),
      });
    });
    afterAll(async () => {
      // All stay unassigned when beforeAll failed before reaching them.
      try {
        await pipeline?.dispose();
      } finally {
        await server?.close();
        if (cacheDir !== undefined) await remove(cacheDir, { recursive: true });
      }
    });

    it("② the recommended sampler the distribution declares is the default when omitted", () => {
      assertEquals(pipeline.defaultSampler, RECOMMENDED_SAMPLER);
    });

    it("① chat at temperature 0 gives the same golden as the fromAssets path", async () => {
      for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
        const { messages } = caseOf(name);
        const started = performance.now();
        const stream = pipeline.chat(messages, {
          // The explicit temperature 0 overrides the default (the recommended sampler) — the
          // golden is the greedy sequence.
          maxNewTokens,
          sampler: { temperature: 0 },
        });
        const parts: string[] = [];
        for await (const chunk of stream) parts.push(chunk);
        const stopped = await stream.done;
        console.log(
          `[e2e] gemma4 fromPretrained ${name}: ${JSON.stringify(parts.join(""))} / ` +
            `${JSON.stringify(stopped)} / ${(performance.now() - started).toFixed(0)}ms`,
        );
        assertEquals(parts.join(""), expected, `${name}: output at temperature 0`);
        assertEquals(stopped, stop, `${name}: stop reason`);
      }
    });

    it("③ the same conversation also runs to completion with the default sampler (temperature 1.0)", async () => {
      const { messages } = caseOf(CASES[0].fixture);
      // Omitting `sampler` = running on the distribution's declaration (the sampling is
      // deterministic through splitmix64, but the values themselves are not pinned — the sequence
      // does not match HF's sampling, so a golden would mean nothing).
      const stream = pipeline.chat(messages, { maxNewTokens: 8 });
      const parts: string[] = [];
      for await (const chunk of stream) parts.push(chunk);
      const stopped = await stream.done;
      console.log(`[e2e] gemma4 default sampler: ${JSON.stringify(parts.join(""))}`);
      assert(parts.join("").length > 0, "the default sampler produced not a single character");
      assert(
        stopped.reason === "eos" || stopped.reason === "max-tokens",
        `stop reason ${stopped.reason}`,
      );
    });
  },
);

/**
 * The q4_0 QAT line's mirror, which the census below reads (the `gemma4` pipeline's int4 seats —
 * the regular line has none left to compare). Checked on its own: the cases above need only the
 * regular mirror.
 */
const QAT_AVAILABLE = mirrorAvailable(GEMMA4_QAT_MIRROR);
if (!QAT_AVAILABLE) {
  console.warn(
    `[karume] ${GEMMA4_QAT_MIRROR.pathname} is absent (or not karume/5), so the gemma4 summation-order` +
      " census SKIPs (cd tools/export-recipes && uv run python dist.py --pipeline gemma4-qat)",
  );
}
/** The seat whose `parallel-fused` declaration the census's last case reads. */
const QAT_FAST_SEAT = { quant: "i4-fast" } as const;

/**
 * **census** (ADR 0058 decision 4 ③) — the ③PV reduction form takes effect **by the merge rule**:
 * explicit > quant declaration > the runtime's reference path (ADR 0104 addendum 2026-09-26 — there
 * is no family code default).
 *
 * - `i4` (empty `session`) runs on the reference path ③ and not a single ③' key appears (a preset
 *   without a declaration is back on the reference path — red if a family code default sneaks in
 *   again).
 * - `i4-gemvpar` (declares `stateAttentionReduce: "parallel"`) runs on ③' and no ③ key appears.
 * - On the same preset an explicit `"sequential"` beats the declaration (the way back still works).
 * - `i4-fast` (declares `"parallel-fused"`) shows no ③ key. Whether the fused kernel
 *   (`attention_state_stats_pv`) is chosen depends on the column limit, so either ③' or the fused
 *   kernel running is enough.
 *
 * NOTE: the preset now affects **only the decode (M = 1) dispatches** (perf-ledger K-13 step 2).
 * A prefill plan (M = `chunkLength` ≥ 16) picks ③ₜ regardless of the preset (shared V row tiles;
 * the key carries the geometry fragment `:f32:reg…`) — it is bit-identical to ③, so it can sit on
 * the default path. One chat therefore **always includes ③ₜ**, and what the preset swaps is the
 * remaining ③ / ③' side.
 * MUST: the source is `lastRunPipelines`, which does not depend on measurement (it runs on a device
 * without measurement too). A run without a census FAILs unconditionally (`mergeCensus` — passing
 * an empty one through would go green with nothing checked).
 * NOTE: with `onRunDiagnostics` attached, the on-GPU greedy output path is not taken
 * (`greedyOutput` in src/gemma/pipeline.ts). The census is that of the path with the hook, and this
 * premise holds even without measurement.
 */
describe.skipIf(!QAT_AVAILABLE || !GPU_AVAILABLE)(
  "gemma4 distribution: the ③PV reduction form is decided in the order explicit > quant declaration > reference path (census)",
  () => {
    const cases = [
      {
        name: "i4 (no declaration) → reference path",
        options: { quant: "i4" },
        expect: "sequential",
      },
      {
        name: "i4-gemvpar (declares parallel) → ③'",
        options: { quant: "i4-gemvpar" },
        expect: "parallel",
      },
      {
        name: "i4-gemvpar + explicit sequential → reference path",
        options: { quant: "i4-gemvpar", stateAttentionReduce: "sequential" },
        expect: "sequential",
      },
      {
        name: "i4-fast (declares parallel-fused) → no ③",
        options: QAT_FAST_SEAT,
        expect: "parallel-fused",
      },
    ] as const;
    let server: LocalDistServer;
    let cacheDir: string;
    let caches: CacheStorage;
    let entry: ModelEntry;
    let qatResidentPleBytes: number;
    beforeAll(async () => {
      server = await serveLocalDist(new URL(".", GEMMA4_QAT_MIRROR).pathname);
      // One on-disk cache for every case: the distribution is warmed once and read back per case.
      cacheDir = await makeTempDir({ prefix: "karume-gemma4-qat-cache-" });
      caches = nodeCacheStorage(cacheDir);
      const manifest = parseManifest(
        readTextFileSync(new URL(MANIFEST_FILENAME, GEMMA4_QAT_MIRROR)),
      );
      entry = manifest.models.e2b;
      // Every seat shares the one i4 `model` container, so one index serves every case.
      qatResidentPleBytes = await allResidentPleBytesOfMirror(GEMMA4_QAT_MIRROR, QAT_FAST_SEAT);
    });
    afterAll(async () => {
      await server?.close();
      if (cacheDir !== undefined) await remove(cacheDir, { recursive: true });
    });
    for (const one of cases) {
      it(one.name, async () => {
        // The effective bundle (the chosen quant preset's declaration + the explicit options,
        // passed through the family's merge) → the row of the bundle's census table.
        // Every case names its seat (the census reads the seat's declaration, not the default).
        const { quant, ...overrides }: { readonly quant: string } & SessionOverrides = one.options;
        const row = censusRowOf(
          "gemma4",
          "e2b",
          effectiveSessionOptions("gemma4", entry.quants[quant].session, overrides, one.name),
        );
        const gpu = await acquireGpu();
        const runs: SessionDiagnostics["lastRunPipelines"][] = [];
        const byPhase = new Map<string, SessionDiagnostics["lastRunPipelines"][]>();
        try {
          const pipeline = await Pipeline.fromPretrained(server.source, {
            gpu,
            caches,
            maxResidentPleBytes: qatResidentPleBytes,
            onRunDiagnostics: (diagnostics, phase) => {
              runs.push(diagnostics.lastRunPipelines);
              byPhase.set(phase.kind, [
                ...(byPhase.get(phase.kind) ?? []),
                diagnostics.lastRunPipelines,
              ]);
            },
            ...one.options,
          });
          try {
            const { messages } = caseOf(CASES[0].fixture);
            const stream = pipeline.chat(messages, {
              maxNewTokens: 4,
              sampler: { temperature: 0 },
            });
            for await (const _ of stream) {
              /* only run it */
            }
            await stream.done;
          } finally {
            await pipeline.dispose();
          }
        } finally {
          gpu.destroy();
        }
        const keys = mergeCensus(runs, one.name).map((row) => row.key);
        const pv = keys.filter((key) => key.startsWith("attention_state_pv"));
        assert(pv.length > 0, `no ③PV key in the breakdown (${keys.join(" / ")})`);
        // Split into three paths (geometry fragment `:f32:reg…` = ③ₜ / `:par` = ③' / the rest =
        // the ③ reference path).
        const tiled = pv.filter((key) => key.includes(":f32:reg"));
        const parallel = pv.filter((key) => key.includes(":par"));
        const sequential = pv.filter((key) => !key.includes(":f32:reg") && !key.includes(":par"));
        // The fusion of row statistics and PV (ADR 0102) is a key family separate from ③PV.
        const fused = keys.filter((key) => key.startsWith("attention_state_stats_pv"));
        // A prefill plan takes ③ₜ regardless of the preset (the preset affects only the decode
        // side).
        assert(tiled.length > 0, `no ③ₜ key ran (${pv.join(" / ")})`);
        if (one.expect === "sequential") {
          assert(sequential.length > 0, `no ③ (sequential) key ran (${pv.join(" / ")})`);
          assertEquals(parallel, [], `③' keys mixed into what should be the reference path`);
          assertEquals(fused, [], `fused keys mixed into what should be the reference path`);
        } else if (one.expect === "parallel") {
          assert(parallel.length > 0, `no ③' key ran (${pv.join(" / ")})`);
          assertEquals(
            sequential,
            [],
            `③ (sequential) keys mixed in despite the parallel declaration`,
          );
          assertEquals(fused, [], `fused keys mixed in despite the parallel declaration`);
        } else {
          assert(
            parallel.length + fused.length > 0,
            `neither ③' nor the fused kernel ran (${keys.join(" / ")})`,
          );
          assertEquals(
            sequential,
            [],
            `③ (sequential) keys mixed in despite the parallel-fused declaration`,
          );
        }
        // If the bundle's census table has a row, check the preset's exact counts per phase and
        // per run (the split of ③ paths is held by the checks above; the table holds the counts
        // of the parallel GEMV, the RMS→add fusion and ①'③').
        if (row !== undefined) {
          const checked = assertRowCensus(row, byPhase, gemmaComponentOf, one.name);
          assert(checked > 0, `${one.name}: no run was compared against the table's expectation`);
        }
      });
    }
  },
);

/**
 * A CacheStorage that fails the moment it is touched (it also counts the calls — so it is noticed
 * even if the failure turns into a diagnostic).
 */
class HostileCacheStorage implements CacheStorage {
  calls = 0;

  open(): Promise<Cache> {
    this.calls += 1;
    throw new Error("test: the local source opened CacheStorage");
  }
  match(): Promise<Response | undefined> {
    this.calls += 1;
    throw new Error("test: the local source looked up CacheStorage");
  }
  has(): Promise<boolean> {
    this.calls += 1;
    throw new Error("test: the local source looked up CacheStorage");
  }
  delete(): Promise<boolean> {
    this.calls += 1;
    throw new Error("test: the local source deleted from CacheStorage");
  }
  keys(): Promise<string[]> {
    this.calls += 1;
    throw new Error("test: the local source looked up CacheStorage");
  }
}

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  "gemma4 direct directory read: fromPretrained → chat gives the same golden (real GPU)",
  () => {
    const caches = new HostileCacheStorage();
    let fetchCalls = 0;
    const fetchStub: typeof globalThis.fetch = (input) => {
      fetchCalls += 1;
      return Promise.reject(
        new Error(
          `test: the local source went to the network (${input instanceof Request ? input.url : String(input)})`,
        ),
      );
    };
    const cacheDiagnostics: string[] = [];
    const retryDiagnostics: string[] = [];

    let pipeline: Pipeline;
    beforeAll(async () => {
      pipeline = await Pipeline.fromPretrained(nodeDirectory(MIRROR_DIR), {
        ...BF16_SEAT,
        maxResidentPleBytes: await maxResidentPleBytes(),
        // Knobs only for the HTTP source. None of them applies to a local source = none is
        // touched.
        fetch: fetchStub,
        caches,
        onCacheError: (diagnostic) => cacheDiagnostics.push(diagnostic.url),
        onRetry: (diagnostic) => retryDiagnostics.push(diagnostic.url),
      });
    });
    afterAll(async () => {
      // Unassigned when beforeAll failed while loading.
      await pipeline?.dispose();
    });

    it("① chat at temperature 0 gives the same golden as the fromAssets / fake HF paths", async () => {
      for (const { fixture: name, maxNewTokens, expected, stop } of DIRECTORY_CASES) {
        const { messages } = caseOf(name);
        const started = performance.now();
        // The explicit temperature 0 overrides the distribution's recommended sampler
        // (temperature 1) — the golden is the greedy sequence.
        const stream = pipeline.chat(messages, { maxNewTokens, sampler: { temperature: 0 } });
        const parts: string[] = [];
        for await (const chunk of stream) parts.push(chunk);
        const stopped = await stream.done;
        console.log(
          `[e2e] gemma4 nodeDirectory ${name}: ${JSON.stringify(parts.join(""))} / ` +
            `${JSON.stringify(stopped)} / ${(performance.now() - started).toFixed(0)}ms`,
        );
        assertEquals(parts.join(""), expected, `${name}: output at temperature 0`);
        assertEquals(stopped, stop, `${name}: stop reason`);
      }
    });

    it("② neither the network nor CacheStorage was touched even once", () => {
      assertEquals(fetchCalls, 0, "the local source called fetch");
      assertEquals(caches.calls, 0, "the local source touched CacheStorage");
      // Nothing was touched, so not a single diagnostic arrives either (it did not fall onto a
      // path that swallows cache failures).
      assertEquals(cacheDiagnostics, [], "cache diagnostics were emitted");
      // Retries belong to the HTTP source only too (a local source does not go through the fetch
      // layer).
      assertEquals(retryDiagnostics, [], "retry notifications were emitted");
    });
  },
);
