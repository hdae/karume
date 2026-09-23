// 中央値の表の門（GPU 不要）。検収③の読み方を決めている 2 つの規則を固定する:
//
//  ・digest は「全体 (payload)」の 2 数で出す
//  ・`⚠` は **payload 側**が 0 でない warm 行だけに付く（descriptor の 2 回では付かない）
//
// 「descriptor の突合は cold / warm どちらでも必ず掛かる」（container-v1 §7 の①）ので、全体の
// 回数で印を付けると全ての warm 行が要注意になり、印が何も言わなくなる。そこを逆にした実装が
// 通らないように、ここで**印が付かない warm**と**印が付く warm**を両方置いてある。

import { assert, assertEquals } from "@std/assert";
import type { FetchBreakdown } from "./breakdown.ts";
import type { MeasureReport, MeasureState } from "./measure.ts";
import { median, renderSummary, summarize } from "./summary.ts";

const MIB = 1024 * 1024;

/** 宣言からの見積り（表の見出しが読む欄だけ意味のある値を持つ）。 */
const FETCH: FetchBreakdown = {
  model: "anima-turbo-v1.1",
  quant: "f16",
  containerCount: 1,
  descriptorPartBytes: 2048,
  constPartBytes: 16 * MIB,
  weightPartBytes: 1024 * MIB,
  assetPartBytes: 0,
  mixedPartBytes: 0,
  manifestAssetBytes: 4 * MIB,
  totalBytes: 1044 * MIB + 2048,
  maxPartBytes: 256 * MIB,
  blockBytes: { weight: 1000 * MIB, scale: 24 * MIB, zeroPoint: 0, asset: 0 },
  maxBlockBytes: 32 * MIB,
  components: [],
  descriptorLengths: [1024, 1024],
};

/** 最小の `MeasureReport`（表が読む欄だけ埋める）。 */
const report = (
  state: MeasureState,
  values: {
    readonly vmHwmMiB?: number;
    readonly digestCalls: number;
    readonly payloadCalls: number;
    readonly cachePuts?: number;
  },
): MeasureReport => ({
  schema: "karume/ram-peak/2",
  at: "2026-09-22T00:00:00.000Z",
  os: "linux",
  mode: "pipeline",
  state,
  family: "anima",
  component: null,
  source: "models/karume-anima",
  cacheDir: state === "local" ? null : "outputs/ram-peak/cache",
  model: "anima-turbo-v1.1",
  quant: "f16",
  steps: 2,
  size: 512,
  maxNewTokens: null,
  explicitGc: false,
  loadMs: 1000,
  runMs: 200,
  vmHwmMiB: values.vmHwmMiB ?? 4000,
  rssBaselineMiB: 100,
  externalBaselineMiB: 1,
  peaks: {
    load: { rssMaxMiB: 3000, externalMaxMiB: 300, samples: 20 },
    run: { rssMaxMiB: 3200, externalMaxMiB: 120, samples: 4 },
    total: { rssMaxMiB: 3200, externalMaxMiB: 300, samples: 24 },
  },
  digest: {
    total: {
      calls: values.digestCalls,
      bytes: 0,
      descriptorCalls: values.digestCalls - values.payloadCalls,
      descriptorBytes: 0,
      payloadCalls: values.payloadCalls,
      payloadBytes: 0,
      maxBytes: 0,
    },
    load: {
      calls: values.digestCalls,
      bytes: 0,
      descriptorCalls: values.digestCalls - values.payloadCalls,
      descriptorBytes: 0,
      payloadCalls: values.payloadCalls,
      payloadBytes: 0,
      maxBytes: 0,
    },
    run: {
      calls: 0,
      bytes: 0,
      descriptorCalls: 0,
      descriptorBytes: 0,
      payloadCalls: 0,
      payloadBytes: 0,
      maxBytes: 0,
    },
  },
  cache: state === "local" ? null : {
    opens: 1,
    matches: 10,
    hits: 10,
    misses: 0,
    puts: values.cachePuts ?? 0,
    putBytes: (values.cachePuts ?? 0) * 1024 * 1024,
    deletes: 0,
  },
  fetch: FETCH,
  components: {},
  missingFromRuntime: ["carriedScaleBytes …"],
});

Deno.test("summary: 中央値は奇数本で中央・偶数本で中央 2 つの平均", () => {
  assertEquals(median([3, 1, 2]), 2);
  assertEquals(median([4, 1, 2, 3]), 2.5);
  assertEquals(median([]), 0);
});

Deno.test("summary: ⚠ は payload 側が 0 でない warm 行にだけ付く", () => {
  const rows = summarize([
    report("cold", { digestCalls: 6, payloadCalls: 4, cachePuts: 12 }),
    report("warm", { digestCalls: 2, payloadCalls: 0 }),
    report("local", { digestCalls: 40, payloadCalls: 38 }),
  ], ["cold", "warm", "local"]);
  assertEquals(rows.map((row) => row.state), ["cold", "warm", "local"]);
  // descriptor の突合 2 回だけの warm は要注意ではない。
  assertEquals(rows.map((row) => row.flagged), [false, false, false]);
  assertEquals(rows[0].cachePuts, 12);
  assertEquals(rows[0].cachePutMiB, 12);

  const flagged = summarize([report("warm", { digestCalls: 3, payloadCalls: 1 })], ["warm"]);
  assertEquals(flagged[0].flagged, true, "payload が残った warm に印が付いていない");
});

Deno.test("summary: 表には全体と payload の 2 数が出る", () => {
  const markdown = renderSummary([
    report("cold", { digestCalls: 6, payloadCalls: 4, cachePuts: 12 }),
    report("warm", { digestCalls: 3, payloadCalls: 1 }),
  ], {
    label: "anima",
    source: "models/karume-anima",
    repeat: 1,
    states: ["cold", "warm"],
    resultsPath: "outputs/ram-peak/x/results.jsonl",
  });
  assert(markdown.includes("| cold | 1 |"), "cold の行が無い");
  assert(markdown.includes("| ⚠ warm | 1 |"), "要注意の warm 行に印が無い");
  assert(markdown.includes("6 (4)"), "digest の 2 数が出ていない");
  assert(markdown.includes("carriedScaleBytes") === false, "凡例に生の欄名が漏れている");
  assert(markdown.includes("持越し scale"), "持越し scale の欠落が凡例に無い");
});

Deno.test("summary: 集計する run が 1 本も無ければ落ちる", () => {
  let thrown: unknown;
  try {
    renderSummary([], {
      label: "anima",
      source: "x",
      repeat: 0,
      states: ["cold"],
      resultsPath: "x",
    });
  } catch (error) {
    thrown = error;
  }
  assert(thrown instanceof Error, "空の結果で落ちていない（欠けた表を作っている）");
});
