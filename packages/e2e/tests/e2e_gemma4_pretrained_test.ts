// 実重み Gemma 4 E2B を**配布形から**組む検収門 —— 段 5 の合格線（実 DL 疎通）。
//
// `fromAssets`（段 4 の門は `e2e_gemma4_chat_test.ts`）と違い、こちらはロード経路を本番と
// 1 本にする: `karume.json` の解決 → 容器の記述文書だけ取って admission → 重みの part の逐次流し
// → PLE（`model` 容器の資産 — ADR 0109 決定 4）の block の遅延読み。門は 4 本:
//
//  ① **完走**: `fromPretrained` → `chat` が実重みで回り、温度 0（明示）の出力が
//     `e2e_gemma4_chat_test.ts` と**同じ golden** に一致する。同じバイト列を別経路で読んでいる
//     ことの証明で、割れたら動いたのは配布形か取得経路である
//  ② **既定サンプラの結線**（ADR 0083 決定 7）: 配布形が宣言した推奨値がそのまま
//     `pipeline.defaultSampler` に載り、要求が `sampler` を省略したときの既定になる。宣言が
//     `pipelineConfig` に焼かれていても結線が抜けていれば、①（温度 0 を明示する経路）は緑の
//     ままなので、この門が要る
//  ③ **PLE が全量常駐しない**（ADR 0085 決定 3）: 索引が指す block は `model` 容器の資産として
//     遅延側で受け、触った 1 本だけを読む。全量を `fetchAssets` へ流す形へ戻ると 2.27GiB が常駐する
//  ④ **配布形の宣言と現物の対応**: 索引が名指しする block が `model` 容器の資産として全件そのまま
//     在ること（取得キーを「並び順で合わせる」形に戻っていないこと）
//
// ## 資産と経路
//
// ローカルの配布形ミラー `models/karume-gemma4/`（`dist.py --pipeline gemma4` が組む）を
// **HF 形の HTTP** で配る使い捨てサーバ越しに読む（`examples/shared/local-dist-server.ts` —
// デモが `--source <ローカルのパス>` で通るのと同じ 1 本）。ミラーはリポジトリ管理外なので、
// 無い環境では**明示 SKIP** する。
//
// NOTE: テストから `examples/` の道具を借りるのは、疑似 HF サーバの綴りを 2 つ持たないため
// （`packages/models/tests/` も `examples/` も publish には入らない）。取得層は資産を
// **永続キャッシュ**（Deno は DENO_DIR）へ写すので、この門を 1 度回すと約 4GiB がそこに残る
// （掃除は `@karume/hub` の `clearHubCache`）。
//
// ## Directory source
//
// The last test reads the same mirror through `nodeDirectory` (`@karume/hub/node`) instead of
// the HTTP server. Two gates: the temperature-0 chat matches the same golden (the local source
// delivers the same bytes: container descriptor, streamed weight parts and the lazy PLE blocks
// all come from that one source), and neither `fetch` nor `CacheStorage` is touched. A local
// source never copies bytes into a cache, so this test leaves nothing in the persistent cache.

import { readTextFile, readTextFileSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { MANIFEST_FILENAME, type ModelEntry, parseManifest } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import { MemoryCacheStorage } from "../../hub/tests/helpers/mock.ts";
import { acquireGpu, type SessionDiagnostics } from "@karume/runtime";
// MUST: 入口は**公開面**（`./gemma` サブパス）から取る — `src/...` を直に掴むと、面が痩せていても
// 門が緑のままになる（消費者が書けない経路で検収したことになる）。
import { type Gemma4ChatMessage, Gemma4Pipeline } from "../../models/gemma.ts";
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
import { mirrorAvailable, openGemma4Ple } from "../../models/tests/helpers/gemma-mirror.ts";

const MIRROR_DIR = new URL("../../../models/karume-gemma4/", import.meta.url);

/** SKIP 時にそのまま貼れる組み立てコマンド。 */
const ASSEMBLE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline gemma4";

/**
 * PLE の常駐に使う予算（バイト）。索引から導く = PLE 全量常駐で、token 範囲をまたぐ会話でも
 * 読み直しが起きない。定数（旧: 768MiB）で書くと、資産世代で block 幅が変われば同じ数が別の
 * 本数を意味してしまう（ADR 0085 追記 2026-09-02 の「本数ではなくバイト」はテスト側の定数にも
 * 同じく効く）。ミラーが無い環境では評価しない（SKIP 判定の後で呼ぶ）。
 */
const maxResidentPleBytes = (): Promise<number> => allResidentPleBytesOfMirror(MIRROR_DIR);

/** 上流 `generation_config.json` の推奨（配布形が `pipelineConfig.sampler` へ焼いた値）。 */
const RECOMMENDED_SAMPLER = { temperature: 1, topK: 64, topP: 0.95 } as const;

/**
 * 検収ケース。**`e2e_gemma4_chat_test.ts` の CASES と同じ golden**（同じ重み・同じ静的配線を
 * 別経路で読んでいることの証明なので、値を割ってはいけない）。
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
    expected: "2023年時点で、およそ1,400万人です。",
    stop: { reason: "eos", token: 106, tokens: 18 },
  },
] as const;

type ChatFixture = {
  readonly chat: {
    readonly name: string;
    readonly messages: Gemma4ChatMessage[];
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
  assert(found !== undefined, `フィクスチャに chat ケース '${name}' が無い`);
  return found;
};

// 無い機も旧 major が残っている機も明示 SKIP（helper の MUST — ADR 0109 決定 9）。
const AVAILABLE = mirrorAvailable(MIRROR_DIR);
const MANIFEST_TEXT = AVAILABLE
  ? readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR))
  : undefined;

if (!AVAILABLE) {
  console.warn(
    `[karume] 配布形ミラー models/karume-gemma4/ が無いため段 5 の疎通検収を SKIP する。` +
      `組み立て: ${ASSEMBLE_COMMAND}`,
  );
}

test.skipIf(!AVAILABLE)("gemma4 配布形: manifest の宣言と PLE 索引の対応（GPU 不要）", async () => {
  const manifest = parseManifest(MANIFEST_TEXT ?? "");
  const entry = manifest.models[manifest.defaultModel];
  assertEquals(entry.pipeline, { name: "gemma4", major: 1 }, "pipeline 契約");
  // weights の役割は製品グラフ + 投機の drafter の 2 本ちょうど（ADR 0096 段 2 で
  // `drafter` が増えた）。tokenizer は manifest の assets の席、PLE は `model` 容器の資産
  // （ADR 0109 決定 4）— ここが増えたら「sidecar を weights へ戻した」退行である。
  assertEquals(
    Object.keys(entry.weights),
    ["model", "drafter"],
    "weights は製品グラフ + drafter の 2 本（PLE は容器の資産・tokenizer は assets の席）",
  );
  assertEquals(Object.keys(entry.assets), ["tokenizer"], "manifest の assets は tokenizer だけ");
  // ④ 索引（schema 3）が名指しする block が、そのまま `model` 容器の資産として在る — 突合は
  // helper（`readGemma4PleIndex`）が全件で行い、片方だけ焼き直したミラーはここで落ちる。
  const { index } = await openGemma4Ple(MIRROR_DIR);
  assert(index.values.blocks.length > 0, "PLE 索引が values の block を 1 本も持たない");
  assert(index.scales.blocks.length > 0, "PLE 索引が scales の block を 1 本も持たない");
  assertEquals(index.values.blocks.at(-1)?.stop, index.tokens, "values の区間が tokens で閉じる");
  assertEquals(index.scales.blocks.at(-1)?.stop, index.tokens, "scales の区間が tokens で閉じる");
});

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  "gemma4 配布形: fromPretrained → chat が実 GPU で完走し既定サンプラが宣言どおり",
  () => {
    let server: LocalDistServer;
    let pipeline: Gemma4Pipeline;
    beforeAll(async () => {
      server = await serveLocalDist(new URL(".", MIRROR_DIR).pathname);
      pipeline = await Gemma4Pipeline.fromPretrained(server.source, {
        // Node has no CacheStorage; the HTTP source needs one (hub prefetches through it).
        caches: new MemoryCacheStorage(),
        maxResidentPleBytes: await maxResidentPleBytes(),
      });
    });
    afterAll(async () => {
      // Both stay unassigned when beforeAll failed before reaching them.
      try {
        await pipeline?.dispose();
      } finally {
        await server?.close();
      }
    });

    it("② 配布形が宣言した推奨サンプラが省略時の既定として載っている", () => {
      assertEquals(pipeline.defaultSampler, RECOMMENDED_SAMPLER);
    });

    it("① 温度 0 の chat が fromAssets 経路と同じ golden を出す", async () => {
      for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
        const { messages } = caseOf(name);
        const started = performance.now();
        const stream = pipeline.chat(messages, {
          // 明示の温度 0 が既定（推奨サンプラ）を上書きする — golden は greedy の列。
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
        assertEquals(parts.join(""), expected, `${name}: 温度 0 の出力`);
        assertEquals(stopped, stop, `${name}: 停止理由`);
      }
    });

    it("③ 既定サンプラ（温度 1.0）でも同じ会話が完走する", async () => {
      const { messages } = caseOf(CASES[0].fixture);
      // `sampler` を省略 = 配布形の宣言で走る（抽選は splitmix64 で決定的だが、値そのものは
      // 固定しない —— HF の sampling とは列が一致しないので golden にする意味が無い）。
      const stream = pipeline.chat(messages, { maxNewTokens: 8 });
      const parts: string[] = [];
      for await (const chunk of stream) parts.push(chunk);
      const stopped = await stream.done;
      console.log(`[e2e] gemma4 既定サンプラ: ${JSON.stringify(parts.join(""))}`);
      assert(parts.join("").length > 0, "既定サンプラで 1 文字も出ていない");
      assert(
        stopped.reason === "eos" || stopped.reason === "max-tokens",
        `停止理由 ${stopped.reason}`,
      );
    });
  },
);

/**
 * **census**（ADR 0058 決定 4 ③）— ③PV の縮約形が**合成の規則どおり**に効くこと: 明示 > quant 宣言 >
 * runtime の参照経路（ADR 0104 追記 2026-09-26 — 家族のコード既定は無い）。
 *
 * - `i4`（`session` が空）は参照経路 ③ で走り、③' のキーが 1 本も出ない（宣言の無い席が参照経路に
 *   戻っていること — 家族のコード既定が再び紛れ込むと赤になる）。
 * - `i4-gemvpar`（`stateAttentionReduce: "parallel"` を宣言）は ③' で走り、③ のキーが出ない。
 * - 同じ席でも明示の `"sequential"` が宣言に勝つ（戻す口が生きていること）。
 * - 既定 quant（E2B = `i4-fast`・`"parallel-fused"` を宣言）は ③ のキーが出ない。融合 kernel
 *   （`attention_state_stats_pv`）が選ばれるかは列上限で決まるので、③' か融合のどちらかが走れば足りる。
 *
 * NOTE: 席が効くのは **decode（M = 1）の dispatch だけ**になった（perf-ledger K-13 段 2）。
 * prefill 計画（M = `chunkLength` ≥ 16）は席に依らず ③ₜ（V 行タイル共有・キーに幾何断片
 * `:f32:reg…` が載る）を選ぶ — ③ とビット同一なので既定経路に置ける。したがって 1 回の chat は
 * **必ず ③ₜ を含み**、席で入れ替わるのは残りの ③ / ③' の側になる。
 * MUST: 源は計測に依らない `lastRunPipelines`（計測の無い device でも走る）。census の無い run は
 * 無条件に FAIL にする（`mergeCensus` — 空なら素通り、にすると無検査のまま緑になる）。
 * NOTE: `onRunDiagnostics` を付けると GPU 上の greedy 出力経路は外れる（src/gemma/pipeline.ts の
 * `greedyOutput`）。census はフック有りの経路のもので、この前提は計測を外しても変わらない。
 */
describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  "gemma4 配布形: ③PV の縮約形は明示 > quant 宣言 > 参照経路の順で決まる（census）",
  () => {
    const cases = [
      { name: "i4（宣言なし）→ 参照経路", options: { quant: "i4" }, expect: "sequential" },
      {
        name: "i4-gemvpar（宣言 parallel）→ ③'",
        options: { quant: "i4-gemvpar" },
        expect: "parallel",
      },
      {
        name: "i4-gemvpar + 明示 sequential → 参照経路",
        options: { quant: "i4-gemvpar", stateAttentionReduce: "sequential" },
        expect: "sequential",
      },
      { name: "既定 quant（宣言 parallel-fused）→ ③ 無し", options: {}, expect: "parallel-fused" },
    ] as const;
    let server: LocalDistServer;
    let entry: ModelEntry;
    beforeAll(async () => {
      server = await serveLocalDist(new URL(".", MIRROR_DIR).pathname);
      const manifest = parseManifest(MANIFEST_TEXT ?? "");
      entry = manifest.models.e2b;
    });
    afterAll(async () => {
      await server?.close();
    });
    for (const one of cases) {
      it(one.name, async () => {
        // 実効の束（選んだ quant 席の宣言 + 明示指定を家族の合成に通した値）→ 束の census 表の行。
        const {
          quant = entry.defaultQuant,
          ...overrides
        }: { readonly quant?: string } & SessionOverrides = one.options;
        const row = censusRowOf(
          "gemma4",
          "e2b",
          effectiveSessionOptions("gemma4", entry.quants[quant].session, overrides, one.name),
        );
        const gpu = await acquireGpu();
        const runs: SessionDiagnostics["lastRunPipelines"][] = [];
        const byPhase = new Map<string, SessionDiagnostics["lastRunPipelines"][]>();
        try {
          const pipeline = await Gemma4Pipeline.fromPretrained(server.source, {
            gpu,
            caches: new MemoryCacheStorage(),
            maxResidentPleBytes: await maxResidentPleBytes(),
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
              /* 走らせるだけ */
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
        assert(pv.length > 0, `内訳に ③PV のキーが無い（${keys.join(" / ")}）`);
        // 3 経路へ分ける（幾何断片 `:f32:reg…` = ③ₜ / `:par` = ③' / 残り = ③ 参照経路）。
        const tiled = pv.filter((key) => key.includes(":f32:reg"));
        const parallel = pv.filter((key) => key.includes(":par"));
        const sequential = pv.filter((key) => !key.includes(":f32:reg") && !key.includes(":par"));
        // 行統計と PV の融合（ADR 0102）は ③PV と別のキー族。
        const fused = keys.filter((key) => key.startsWith("attention_state_stats_pv"));
        // prefill 計画は席に依らず ③ₜ（席の指定は decode 側にしか効かない）。
        assert(tiled.length > 0, `③ₜ のキーが走っていない（${pv.join(" / ")}）`);
        if (one.expect === "sequential") {
          assert(sequential.length > 0, `③（逐次）のキーが走っていない（${pv.join(" / ")}）`);
          assertEquals(parallel, [], `参照経路のはずが ③' のキーが混ざっている`);
          assertEquals(fused, [], `参照経路のはずが融合のキーが混ざっている`);
        } else if (one.expect === "parallel") {
          assert(parallel.length > 0, `③' のキーが走っていない（${pv.join(" / ")}）`);
          assertEquals(sequential, [], `parallel の宣言なのに ③（逐次）のキーが混ざっている`);
          assertEquals(fused, [], `parallel の宣言なのに融合のキーが混ざっている`);
        } else {
          assert(
            parallel.length + fused.length > 0,
            `③' も融合も走っていない（${keys.join(" / ")}）`,
          );
          assertEquals(sequential, [], `parallel-fused の宣言なのに ③（逐次）のキーが混ざっている`);
        }
        // 束の census 表に行があれば、相ごと・run ごとに席の本数ちょうどで見る（③ の経路分けは
        // 上の検査が持ち、表は並列 GEMV・RMS→add 融合・①'③' の本数を持つ）。
        if (row !== undefined) {
          const checked = assertRowCensus(row, byPhase, gemmaComponentOf, one.name);
          assert(checked > 0, `${one.name}: 表の期待と突き合わせた run が 1 本も無い`);
        }
      });
    }
  },
);

/** 触れた瞬間に落ちる CacheStorage（呼ばれた回数も数える — 診断へ化けても気づけるように）。 */
class HostileCacheStorage implements CacheStorage {
  calls = 0;

  open(): Promise<Cache> {
    this.calls += 1;
    throw new Error("test: ローカル取得元が CacheStorage を開いた");
  }
  match(): Promise<Response | undefined> {
    this.calls += 1;
    throw new Error("test: ローカル取得元が CacheStorage を引いた");
  }
  has(): Promise<boolean> {
    this.calls += 1;
    throw new Error("test: ローカル取得元が CacheStorage を引いた");
  }
  delete(): Promise<boolean> {
    this.calls += 1;
    throw new Error("test: ローカル取得元が CacheStorage を消した");
  }
  /** 旧名前空間の回収（hub のセッション入口）だけは通す — 消すものが無いことを答える。 */
  keys(): Promise<string[]> {
    return Promise.resolve([]);
  }
}

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  "gemma4 ディレクトリ直読み: fromPretrained → chat が同じ golden を出す（実 GPU）",
  () => {
    const caches = new HostileCacheStorage();
    let fetchCalls = 0;
    const fetchStub: typeof globalThis.fetch = (input) => {
      fetchCalls += 1;
      return Promise.reject(new Error(`test: ローカル取得元が network へ出た（${String(input)}）`));
    };
    const cacheDiagnostics: string[] = [];
    const retryDiagnostics: string[] = [];

    let pipeline: Gemma4Pipeline;
    beforeAll(async () => {
      pipeline = await Gemma4Pipeline.fromPretrained(nodeDirectory(MIRROR_DIR), {
        maxResidentPleBytes: await maxResidentPleBytes(),
        // HTTP 取得元専用のノブ。ローカル取得元では 1 つも効かない = 触られない。
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

    it("① 温度 0 の chat が fromAssets / 疑似 HF 経路と同じ golden を出す", async () => {
      for (const { fixture: name, maxNewTokens, expected, stop } of DIRECTORY_CASES) {
        const { messages } = caseOf(name);
        const started = performance.now();
        // 明示の温度 0 が配布形の推奨サンプラ（温度 1）を上書きする — golden は greedy の列。
        const stream = pipeline.chat(messages, { maxNewTokens, sampler: { temperature: 0 } });
        const parts: string[] = [];
        for await (const chunk of stream) parts.push(chunk);
        const stopped = await stream.done;
        console.log(
          `[e2e] gemma4 nodeDirectory ${name}: ${JSON.stringify(parts.join(""))} / ` +
            `${JSON.stringify(stopped)} / ${(performance.now() - started).toFixed(0)}ms`,
        );
        assertEquals(parts.join(""), expected, `${name}: 温度 0 の出力`);
        assertEquals(stopped, stop, `${name}: 停止理由`);
      }
    });

    it("② network も CacheStorage も 1 度も通っていない", () => {
      assertEquals(fetchCalls, 0, "ローカル取得元が fetch を呼んだ");
      assertEquals(caches.calls, 0, "ローカル取得元が CacheStorage を触った");
      // 触っていないので診断も 1 本も来ない（キャッシュ失敗を握り潰す経路に落ちていない）。
      assertEquals(cacheDiagnostics, [], "キャッシュ診断が飛んでいる");
      // 再試行も HTTP 取得元だけのもの（ローカル取得元は取得層を通らない）。
      assertEquals(retryDiagnostics, [], "再試行の通知が飛んでいる");
    });
  },
);
