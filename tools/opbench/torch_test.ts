// opbench torch の Deno 側テスト: argv の組み立てと、torch.jsonl の op 別集計（GPU 不要）。

import { assertEquals } from "@std/assert";
import { summarizeTorch, torchArgs, type TorchRecord } from "./torch.ts";

Deno.test("torchArgs: 省略可能なオプションは渡さない・--op は繰り返す", () => {
  assertEquals(
    torchArgs({ venv: "/v", single: "s.jsonl", out: "o", compile: false }),
    ["--single", "s.jsonl", "--out", "o"],
  );
  assertEquals(
    torchArgs({
      venv: "/v",
      single: "s.jsonl",
      out: "o",
      compile: true,
      rounds: 3,
      limit: 2,
      ops: ["linear", "gelu"],
    }),
    [
      "--single",
      "s.jsonl",
      "--out",
      "o",
      "--rounds",
      "3",
      "--compile",
      "--limit",
      "2",
      "--op",
      "linear",
      "--op",
      "gelu",
    ],
  );
});

const record = (
  op: string,
  count: number,
  karumeNs: number | null,
  ms: Record<string, number>,
): TorchRecord => ({
  scenario: "decode",
  component: "model",
  op,
  in_shapes: [[1, 8]],
  storage_signature: "none",
  count,
  karume_ns_per_node_min: karumeNs,
  karume_keys: [],
  ms,
  reps: {},
  mem_mib: {},
  errors: {},
});

Deno.test("summarizeTorch: op ごとに比の中央値と加重 ms を出し、列の無い case は比から除く", () => {
  const rows = [
    record("linear", 10, 2_000_000, { f16: 1.0, bf16: 4.0 }), // karume 2ms → 比 2 / 0.5
    record("linear", 5, 3_000_000, { f16: 1.0 }), // 比 3・bf16 は無い
    record("gelu", 4, null, { f16: 0.5 }), // karume 側未測定 → 比なし・加重 karume 0
  ];
  const [linear, gelu] = summarizeTorch(rows, ["f16", "bf16"]);
  assertEquals(linear.op, "linear");
  assertEquals(linear.cases, 2);
  assertEquals(linear.median_ratio, { f16: 3, bf16: 0.5 });
  assertEquals(linear.weighted_ms, { karume: 35, f16: 15, bf16: 40 });
  assertEquals(gelu.median_ratio, { f16: null, bf16: null });
  assertEquals(gelu.weighted_ms, { karume: 0, f16: 2, bf16: 0 });
});
