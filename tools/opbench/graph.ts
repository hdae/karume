/**
 * opbench `graph` — 実資産を 1 回走らせ、run ごとの op 別 GPU 時間（パイプラインキー別）と dispatch
 * 数を記録し、静的 census の展開表（op → ノード本数）と突き合わせる。2 段目の 2 本目で、
 * 「単体で測った時間 × census 加重」が実グラフ内の時間とどれだけずれるか（K-11 の 15〜25%）を
 * 資産ごとに出す道具。
 *
 * 観測点は各家族の pipeline が持つ `onRunDiagnostics`（run 1 本ごとの `SessionDiagnostics`）だけ
 * — production は無変更（research 2026-08-10-op-timing-stats §5 と同じ流儀）。op 別 GPU 時間が
 * 要るので `acquireGpu({ gpuTiming: true })` を注入する（1 dispatch = 1 pass に開くため壁時計は
 * 変わる — 壁は timing 無効の別プロセスで採る = `single` と同じ二本立て）。
 *
 * 家族ごとの「1 回」: gemma4 = 短い chat 1 ターン（prefill 1 run + decode N run）、anima = 1 枚
 * （text_encoder / text_conditioner / transformer step / vae_decoder タイル）。他家族は未対応で
 * fail loudly（波 a の対象 = 合格線 K-11 / P-1 の 2 資産）。
 */

import type { GpuContext, SessionDiagnostics } from "../../packages/runtime/mod.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import { AnimaPipeline } from "../../packages/models/mod.ts";
import { Gemma4Pipeline } from "../../packages/models/gemma.ts";
import type { CensusSummary } from "./census.ts";
import type { SingleSummary } from "./single.ts";

/** run 1 本の記録（graph.jsonl の 1 行）。 */
export type RunRecord = {
  readonly index: number;
  readonly component: string;
  /** 家族ごとの意味づけ（gemma4 = prefill / decode-n・anima = <component>-n）。 */
  readonly label: string;
  readonly dispatch_count: number;
  /** 計測無効なら null（wall プロセス）。 */
  readonly total_ns: number | null;
  readonly entries: readonly {
    readonly key: string;
    readonly ns: number;
    readonly dispatch_count: number;
  }[];
  readonly fusions: Readonly<Record<string, number>> | null;
  readonly clamped_negative_samples: number;
};

export const recordRun = (
  index: number,
  component: string,
  label: string,
  diagnostics: SessionDiagnostics,
): RunRecord => {
  const timing = diagnostics.lastRunTiming;
  return {
    index,
    component,
    label,
    dispatch_count: timing?.dispatchCount ?? diagnostics.submit.dispatchCount,
    total_ns: timing?.totalNs ?? null,
    entries: (timing?.entries ?? []).map((entry) => ({
      key: entry.key,
      ns: entry.ns,
      dispatch_count: entry.dispatchCount,
    })),
    fusions: diagnostics.lastRunFusions ?? null,
    clamped_negative_samples: timing?.clampedNegativeSamples ?? 0,
  };
};

/**
 * パイプラインキー → census の op 名。キーの先頭語（`:` の前）は「op 種 + 変種」なので、op と
 * 綴りが違うものだけ表で引く。表に無い先頭語は census の op 名そのものとみなし、census 側にも
 * 無ければ `unmapped` に残す（黙って捨てない）。
 * - `quantize_rows` は a8 経路が linear / attention の脇で出す活性量子化で、census にノードが無い
 *   → `aux` として別勘定（K-11 の対照実測が「linear 277 = 276 + 1」で数えたのと同じ扱い）。
 * - 要素ごとの op は 1 本のカーネル `ew:<版>:<op>:…` に畳まれるので、op は 3 番目の語から取る。
 * - `strided` / `strided_write` は permute / slice / cat / expand の**実体化**で、どの op から出た
 *   dispatch かはキーに残らない → `strided` の 1 バケツ（census 側は op 別なので 1:1 には
 *   ならない — P-5 の kind 別内訳は census 側が持つ）。
 * - 融合ルールのキー（`rope` など）は census では fused_by に畳まれたノードなので `fused`。
 */
/** census の op と 1:1 にならない集計バケツ（unmapped には入れない）。 */
const BUCKETS: ReadonlySet<string> = new Set(["aux", "strided", "fused"]);
const KEY_TO_OP: Readonly<Record<string, string>> = {
  linear_gemv: "linear",
  attention_state_qk: "attention",
  attention_state_pv: "attention",
  attention_state_stats: "attention",
  attention_qk: "attention",
  attention_pv: "attention",
  attention_stats: "attention",
  attention_fused: "attention",
  // adaln_norm は融合ルール adaln が畳んだ modulation 群の 1 カーネル（census では fused_by）。
  adaln_norm: "fused",
  quantize_rows: "aux",
  strided: "strided",
  strided_write: "strided",
  rope: "fused",
  // silu ルールは x·sigmoid(x) の 2 ノードを 1 カーネルに畳む（census 側に silu の op は無い）。
  silu: "fused",
  silu_mul: "fused",
  upsample2x: "fused",
};

export const opOfKey = (key: string): string => {
  const parts = key.split(":");
  if (parts[0] === "ew" && parts.length >= 3) return parts[2];
  return KEY_TO_OP[parts[0]] ?? parts[0];
};

/** op 別の突合行。 */
export type OpComparison = {
  readonly op: string;
  /** census の素のノード本数（融合で消えたノードと 0 dispatch の別名化は含まない）。 */
  readonly census_nodes: number | null;
  readonly measured_dispatches: number;
  readonly measured_ms: number;
  /** `single` の census 加重合計（ms・同じ op の全格納を合算）。single 無し / 未測定なら null。 */
  readonly single_weighted_ms: number | null;
  /** single / graph（1 なら単体の加重合計が実グラフ内の時間と一致）。 */
  readonly single_over_graph: number | null;
};

export type GraphComparison = {
  readonly scenario: string;
  /** 突合に使った run（label が `runs` で始まるもの）の本数 — 表の値はその平均。 */
  readonly runs: number;
  readonly rows: readonly OpComparison[];
  readonly unmapped_keys: readonly string[];
  /** 突合に使った census 側のコンポーネント（run 群が触ったもの）。 */
  readonly components: readonly string[];
  readonly measured_dispatches_total: number;
  readonly census_plain_nodes: number;
};

/**
 * run 群（同じ形の run を平均）を census のシナリオと突き合わせる。census の「素のノード本数」は
 * `by_op[op].nodes` から融合で消えた本数（`by_fusion.absorbed`）を差し引けないので（absorbed は
 * ルール別・op 別ではない）、weights[] の fused_by == null かつ aliases_input == false の行の
 * count を op 別に足して求める。census 側は **run 群が触ったコンポーネントだけ**に絞る（anima の
 * 1024px は 4 コンポーネントの合算だが、transformer の step と突き合わせるのは transformer の行）。
 */
export const compareWithCensus = (
  records: readonly RunRecord[],
  census: CensusSummary,
  scenario: string,
  single?: SingleSummary,
): GraphComparison => {
  const summary = census.scenarios.find((entry) => entry.scenario === scenario);
  if (summary === undefined) {
    throw new Error(
      `census にシナリオ '${scenario}' が無い（既知: ${
        census.scenarios.map((entry) => entry.scenario).join(" / ")
      }）`,
    );
  }
  const components = new Set(records.map((record) => record.component));
  const censusNodes = new Map<string, number>();
  for (const weight of summary.weights) {
    if (weight.fused_by !== null || weight.aliases_input) continue;
    if (components.size > 0 && !components.has(weight.component)) continue;
    censusNodes.set(weight.op, (censusNodes.get(weight.op) ?? 0) + weight.count);
  }
  const measuredDispatches = new Map<string, number>();
  const measuredNs = new Map<string, number>();
  const unmapped = new Set<string>();
  for (const record of records) {
    for (const entry of record.entries) {
      const op = opOfKey(entry.key);
      if (!BUCKETS.has(op) && !censusNodes.has(op)) unmapped.add(entry.key);
      measuredDispatches.set(op, (measuredDispatches.get(op) ?? 0) + entry.dispatch_count);
      measuredNs.set(op, (measuredNs.get(op) ?? 0) + entry.ns);
    }
  }
  const singleByOp = new Map<string, number>();
  if (single !== undefined) {
    for (const [key, ms] of Object.entries(single.weighted_ms_by_op_storage)) {
      const op = key.split("/")[0];
      singleByOp.set(op, (singleByOp.get(op) ?? 0) + ms);
    }
  }
  const runs = Math.max(1, records.length);
  const ops = new Set([...censusNodes.keys(), ...measuredDispatches.keys()]);
  const rows = [...ops].map((op): OpComparison => {
    const ms = (measuredNs.get(op) ?? 0) / runs / 1e6;
    const singleMs = singleByOp.get(op);
    return {
      op,
      census_nodes: censusNodes.get(op) ?? null,
      measured_dispatches: (measuredDispatches.get(op) ?? 0) / runs,
      measured_ms: ms,
      single_weighted_ms: singleMs ?? null,
      single_over_graph: singleMs === undefined || ms === 0 ? null : singleMs / ms,
    };
  }).sort((a, b) => b.measured_ms - a.measured_ms);
  return {
    scenario,
    runs: records.length,
    rows,
    unmapped_keys: [...unmapped].sort(),
    components: [...components].sort(),
    measured_dispatches_total: [...measuredDispatches.values()].reduce((a, b) => a + b, 0) / runs,
    census_plain_nodes: [...censusNodes.values()].reduce((a, b) => a + b, 0),
  };
};

export type DriveOptions = {
  readonly gpu: GpuContext;
  readonly source: string;
  readonly family: "gemma4" | "anima";
  readonly model?: string;
  readonly quant?: string;
  /** gemma4: 生成 token 数（decode run の本数 − 1 に近い — 最後の run の診断は届かない）。 */
  readonly newTokens?: number;
  /** anima: step 数（sigma の linspace のため 2 以上）と正方の辺。 */
  readonly steps?: number;
  readonly size?: number;
  readonly prompt?: string;
};

export type DriveResult = {
  readonly records: readonly RunRecord[];
  readonly wall_ms: number;
  readonly load_ms: number;
};

/** 実資産を 1 回走らせて run ごとの診断を集める（家族ごとの「1 回」はモジュール doc）。 */
export const driveOnce = async (options: DriveOptions): Promise<DriveResult> => {
  const records: RunRecord[] = [];
  const selection = {
    ...(options.model === undefined ? {} : { model: options.model }),
    ...(options.quant === undefined ? {} : { quant: options.quant }),
  };
  const loadStarted = performance.now();
  if (options.family === "gemma4") {
    let runs = 0;
    const pipeline = await Gemma4Pipeline.fromPretrained(denoDirectory(options.source), {
      ...selection,
      gpu: options.gpu,
      onRunDiagnostics: (diagnostics) => {
        const label = runs === 0 ? "prefill" : `decode-${runs}`;
        records.push(recordRun(runs, "model", label, diagnostics));
        runs += 1;
      },
    });
    const loadMs = performance.now() - loadStarted;
    try {
      const started = performance.now();
      await pipeline.chat(
        [{ role: "user", content: options.prompt ?? "Explain WebGPU in one sentence." }],
        { maxNewTokens: options.newTokens ?? 8 },
      ).text();
      return { records, wall_ms: performance.now() - started, load_ms: loadMs };
    } finally {
      await pipeline.dispose();
    }
  }
  const counts = new Map<string, number>();
  const pipeline = await AnimaPipeline.fromPretrained(denoDirectory(options.source), {
    ...selection,
    gpu: options.gpu,
    onRunDiagnostics: (component, diagnostics) => {
      const n = counts.get(component) ?? 0;
      counts.set(component, n + 1);
      records.push(recordRun(records.length, component, `${component}-${n}`, diagnostics));
    },
  });
  const loadMs = performance.now() - loadStarted;
  try {
    const started = performance.now();
    const size = options.size ?? 1024;
    await pipeline.generate({
      prompt: options.prompt ?? "1girl, solo, upper body",
      steps: options.steps ?? 2,
      resolution: { width: size, height: size },
      seed: 1,
    });
    return { records, wall_ms: performance.now() - started, load_ms: loadMs };
  } finally {
    await pipeline.dispose();
  }
};
