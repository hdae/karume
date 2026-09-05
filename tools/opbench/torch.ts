/**
 * opbench `torch` の Deno 側 — CUDA venv の python で torch_bench.py を subprocess として走らせ、
 * 出た torch.jsonl を `single` の記録と突き合わせて列 B との比（karume / torch）を出す。
 *
 * venv の解決と python の起動は tools/_shared/python.ts（fusion-hints `inductor` と共用）。
 */

import { runVenvPython } from "../_shared/python.ts";

export { defaultVenv } from "../_shared/python.ts";

/** torch.jsonl の 1 行（torch_bench.py が書く）。 */
export type TorchRecord = {
  readonly scenario: string;
  readonly component: string;
  readonly op: string;
  readonly in_shapes: readonly (readonly number[])[];
  readonly storage_signature: string;
  readonly count: number;
  readonly karume_ns_per_node_min: number | null;
  readonly karume_keys: readonly string[];
  readonly ms: Readonly<Record<string, number>>;
  readonly reps: Readonly<Record<string, number>>;
  readonly mem_mib: Readonly<Record<string, number>>;
  readonly errors: Readonly<Record<string, string>>;
};

/** op 別の比（karume の 1 ノード時間 / torch の列）。 */
export type TorchOpRatio = {
  readonly op: string;
  readonly cases: number;
  /** 列名 → 比の中央値（karume ms / torch ms・>1 は karume が遅い）。列が無い / 失敗した case は除く。 */
  readonly median_ratio: Readonly<Record<string, number | null>>;
  /**
   * census 加重（count）で足した karume 側の ms と、列ごとの torch 側の ms。
   *
   * MUST: 母数を揃える — **karume 側が測れた case（`karume_ns_per_node_min !== null`）だけ**を
   * 足す。torch 列だけ全行を足すと、karume が測れなかった case が torch 側にだけ乗り、
   * 総和の比が実際と違う向きへ動く（比の中央値 {@link TorchOpRatio.median_ratio} は元から
   * 両側の揃った case でしか採っていないので、2 つの指標の母数が割れる）。
   */
  readonly weighted_ms: Readonly<Record<string, number>>;
};

const median = (values: readonly number[]): number | null => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};

/** op ごとに列 B 比を畳む（比は case 単位で取ってから中央値 — 形の大小で重みが偏らないように）。 */
export const summarizeTorch = (
  records: readonly TorchRecord[],
  columns: readonly string[],
): readonly TorchOpRatio[] => {
  const byOp = new Map<string, TorchRecord[]>();
  for (const record of records) {
    byOp.set(record.op, [...(byOp.get(record.op) ?? []), record]);
  }
  return [...byOp.entries()].map(([op, rows]): TorchOpRatio => {
    const ratios: Record<string, number | null> = {};
    const weighted: Record<string, number> = { karume: 0 };
    for (const row of rows) {
      if (row.karume_ns_per_node_min !== null) {
        weighted.karume += (row.karume_ns_per_node_min / 1e6) * row.count;
      }
    }
    for (const column of columns) {
      const values: number[] = [];
      let total = 0;
      for (const row of rows) {
        const torchMs = row.ms[column];
        // karume 側が測れなかった case は合計にも入れない（karume 列と母数を揃える）。
        if (torchMs === undefined || row.karume_ns_per_node_min === null) continue;
        total += torchMs * row.count;
        values.push(row.karume_ns_per_node_min / 1e6 / torchMs);
      }
      ratios[column] = median(values);
      weighted[column] = total;
    }
    return { op, cases: rows.length, median_ratio: ratios, weighted_ms: weighted };
  }).sort((a, b) => b.weighted_ms.karume - a.weighted_ms.karume);
};

export type TorchRunOptions = {
  readonly venv: string;
  /** `single.jsonl` の**素の path**（URL の `pathname` ではない — `_shared/assets.ts` の externalPath）。 */
  readonly single: string;
  /** 書き出し先ディレクトリの**素の path**（同上）。 */
  readonly out: string;
  readonly rounds?: number;
  readonly compile: boolean;
  readonly limit?: number;
  readonly ops?: readonly string[];
};

/** python の argv を組む（純関数 — テストで見る。台本の path は runVenvPython が先頭に付ける）。 */
export const torchArgs = (options: TorchRunOptions): readonly string[] => [
  "--single",
  options.single,
  "--out",
  options.out,
  ...(options.rounds === undefined ? [] : ["--rounds", String(options.rounds)]),
  ...(options.compile ? ["--compile"] : []),
  ...(options.limit === undefined ? [] : ["--limit", String(options.limit)]),
  ...(options.ops ?? []).flatMap((op) => ["--op", op]),
];

/** venv の python で台本を走らせる（stderr は進捗としてそのまま流す）。 */
export const runTorchBench = (options: TorchRunOptions): Promise<void> =>
  runVenvPython(
    options.venv,
    new URL("./torch_bench.py", import.meta.url),
    torchArgs(options),
  );
