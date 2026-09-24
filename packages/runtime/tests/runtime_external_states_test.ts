// 借り物 state スロット（`states[].external`）・readonly attention・共有 initializer の
// **宣言と契約**の門（ADR 0096 段 2 §1.1〜§1.3）。GPU を一切使わない。
//
// 借り手（drafter）のグラフは 3 つの新しい宣言でできている:
//   ① `states[].external: true` — 実体は貸し手 context にあり、自分では確保しない
//   ② `attrs.readonly: true` の attention — ins は q 1 本で、列 `[P−min(P,W), P)` を読む
//   ③ `initializers[].shared` — バイトを配布形に持たず、貸し手の重みをそのまま束ねる
//
// この 3 つは互いに支え合っている（external の読者は readonly だけ / readonly の相手は
// external だけ / 混在は無し）ので、**片方だけ書いた形が全部 fail loudly になること**が
// この門の本体。通してしまうと、どれも例外ではなく「未初期化の過去を読む」「貸し手が確定した
// KV を上書きする」という**別の値**として出る。

import { assertEquals, assertThrows } from "@std/assert";
import { IrError, type IrGraph } from "../src/format/ir.ts";
import { OpContractError } from "../src/ops.ts";
import { ExecutionError, planGraph, validateGraphContracts } from "../src/runtime/plan.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

const HEADS = 4;
const KV_HEADS = 1;
const DEPTH = 8;
const CAPACITY = 16;
const WINDOW = 8;

/**
 * drafter の最小形: 借り物スロット 2 本を readonly attention 1 本が読むだけのグラフ。
 * 入力は q `[1,H,1,D]` の 1 本で、`state_append` は 1 本も無い。
 */
const drafterGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention"] },
  symbols: [],
  inputs: [{ name: "q", dtype: "f32", shape: [1, HEADS, 1, DEPTH] }],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, HEADS, 1, DEPTH] } },
  states: {
    "l13.k": { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH], external: true },
    "l13.v": { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH], external: true },
  },
  nodes: [{
    op: "attention",
    ins: ["q"],
    outs: ["o"],
    attrs: { scale: 0.5, window: WINDOW, readonly: true },
    states: { k: "l13.k", v: "l13.v" },
  }],
});

/** 貸し手側の最小形（自前スロット + 通常の states 形 attention + append 2 本）。 */
const targetGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention", "state_append"] },
  symbols: [],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, HEADS, 1, DEPTH] },
    { name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
    { name: "v", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, HEADS, 1, DEPTH] } },
  states: {
    "l13.k": { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH] },
    "l13.v": { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 0.5, window: WINDOW },
      states: { k: "l13.k", v: "l13.v" },
    },
    {
      op: "state_append",
      ins: ["k"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "l13.k" },
    },
    {
      op: "state_append",
      ins: ["v"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "l13.v" },
    },
  ],
});

/** 宣言 → 合流後のグラフ（helpers/merged-graph.ts — 供給は宣言から自動で作る）。 */
const parse = mergeGraph;

/** 変異させたグラフを parse + 契約検査まで通す（どちらの層で落ちるかは各テストが指定する）。 */
const mutated = (mutate: (graph: DeclarationJson) => void): DeclarationJson => {
  const graph = drafterGraph();
  mutate(graph);
  return graph;
};

const slotShapes = (graph: IrGraph): ReadonlyMap<string, readonly number[]> =>
  new Map(
    Object.entries(graph.states).map(([name, slot]) => [name, slot.shape.map(Number)]),
  );

Deno.test("drafter の最小形（external 2 本 + readonly attention 1 本）は受理される", () => {
  const graph = parse(drafterGraph());
  assertEquals(graph.states["l13.k"].external, true);
  validateGraphContracts(graph);
  const plan = planGraph(graph, {}, slotShapes(graph));
  // 出力は q と同形（[B,H,1,D]）。
  assertEquals(plan.nodes[0].outputs[0].shape, [1, HEADS, 1, DEPTH]);
  // ins は q 1 本だけ（今 step の k / v は取らない）。
  assertEquals(plan.nodes[0].inputShapes.length, 1);
});

Deno.test("貸し手（自前スロット + append）は 1 バイトも影響を受けない（external 無しは無風）", () => {
  const graph = parse(targetGraph());
  assertEquals(graph.states["l13.k"].external, false);
  validateGraphContracts(graph);
});

Deno.test("external スロットへの state_append は拒否する（書けるのは貸し手だけ）", () => {
  const graph = parse(mutated((g) => {
    g.requires.ops = ["attention", "state_append"];
    g.inputs.push({ name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] });
    g.nodes.push({
      op: "state_append",
      ins: ["k"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "l13.k" },
    });
  }));
  assertThrows(() => validateGraphContracts(graph), ExecutionError, "external なのに state_append");
});

Deno.test("external スロットを readonly でない attention が読む形は拒否する", () => {
  // 読者を従来の states 形（ins 3 本）に戻す = 「今 step の k/v を足す」形。書き手が居ない
  // スロットに対してこれを許すと、必ず未初期化行を過去として読む。
  const graph = parse(mutated((g) => {
    g.inputs.push(
      { name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
      { name: "v", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
    );
    g.nodes[0].ins = ["q", "k", "v"];
    g.nodes[0].attrs = { scale: 0.5, window: WINDOW };
  }));
  assertThrows(
    () => validateGraphContracts(graph),
    ExecutionError,
    "readonly でない attention が読む",
  );
});

Deno.test("readonly の読者が非 external スロットを読む形は拒否する", () => {
  const graph = parse(mutated((g) => {
    if (g.states === undefined) throw new Error("states が無い");
    delete g.states["l13.k"].external;
    delete g.states["l13.v"].external;
  }));
  assertThrows(
    () => validateGraphContracts(graph),
    ExecutionError,
    "external でない state スロット",
  );
});

Deno.test("external と非 external の混在は拒否する（借り手は自前スロットを持たない）", () => {
  const graph = parse(mutated((g) => {
    if (g.states === undefined) throw new Error("states が無い");
    delete g.states["l13.v"].external;
  }));
  // 混在の判定より先に「readonly が非 external を読む」で落ちる（どちらも同じ層・同じ型）。
  assertThrows(() => validateGraphContracts(graph), ExecutionError, "ADR 0096 段 2 §1.1");
});

Deno.test("external を宣言しつつ readonly も append も無いスロットは参照検査で落ちる", () => {
  // 読者がゼロのスロットは IR 層（参照完全性）が落とす — external でも例外にならない。
  const graph = mutated((g) => {
    if (g.states === undefined) throw new Error("states が無い");
    g.states["l14.k"] = { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH], external: true };
  });
  assertThrows(() => parse(graph), IrError, "どのノードからも参照されない");
});

Deno.test("states[].external は true 以外を書けない（不存在と同義の綴りを 2 つ持たない）", () => {
  for (const value of [false, 1, "true"] as const) {
    const graph = mutated((g) => {
      if (g.states === undefined) throw new Error("states が無い");
      // deno-lint-ignore no-explicit-any -- 契約外の値を注入する故障注入（型の外から来る形）
      (g.states["l13.k"] as any).external = value;
    });
    assertThrows(() => parse(graph), IrError, "external");
  }
});

Deno.test("readonly の attention は入力 1 本ちょうど（3 本は拒否・states 欄が要る）", () => {
  const threeIns = parse(mutated((g) => {
    g.inputs.push(
      { name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
      { name: "v", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
    );
    g.nodes[0].ins = ["q", "k", "v"];
  }));
  assertThrows(() => validateGraphContracts(threeIns), OpContractError, "入力 1 本ちょうど");

  const noStates = parse(mutated((g) => {
    g.inputs.push(
      { name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
      { name: "v", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
    );
    g.nodes[0].ins = ["q", "k", "v"];
    delete g.nodes[0].states;
    g.nodes[0].attrs = { scale: 0.5, readonly: true };
    delete g.states;
  }));
  assertThrows(
    () => validateGraphContracts(noStates),
    OpContractError,
    "states 欄を持つノードでのみ",
  );
});

Deno.test("attrs.readonly は true のみ・state_append には書けない", () => {
  const falsy = parse(mutated((g) => {
    g.nodes[0].attrs = { scale: 0.5, window: WINDOW, readonly: false };
  }));
  assertThrows(() => validateGraphContracts(falsy), OpContractError, "readonly は true のみ");

  const onAppend = parse(mutated((g) => {
    g.requires.ops = ["attention", "state_append"];
    g.inputs.push({ name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] });
    g.nodes.push({
      op: "state_append",
      ins: ["k"],
      outs: [],
      attrs: { window: WINDOW, readonly: true },
      states: { slot: "l13.k" },
    });
  }));
  assertThrows(() => validateGraphContracts(onAppend), OpContractError, "attrs.readonly は無い");
});

Deno.test("readonly の形検査: M = 1 MUST・B / D 一致・H % Hkv・window ≤ C", () => {
  const cases: readonly (readonly [string, (graph: DeclarationJson) => void, string])[] = [
    ["M が 2", (g) => {
      g.inputs[0].shape = [1, HEADS, 2, DEPTH];
      g.values.o.shape = [1, HEADS, 2, DEPTH];
    }, "M（軸 2）が 1 ちょうど"],
    ["D がスロットと不一致", (g) => {
      g.inputs[0].shape = [1, HEADS, 1, DEPTH * 2];
      g.values.o.shape = [1, HEADS, 1, DEPTH * 2];
    }, "B / D が不一致"],
    ["H が Hkv の倍数でない", (g) => {
      g.inputs[0].shape = [1, HEADS + 1, 1, DEPTH];
      g.values.o.shape = [1, HEADS + 1, 1, DEPTH];
      if (g.states === undefined) throw new Error("states が無い");
      g.states["l13.k"].shape = [1, 2, CAPACITY, DEPTH];
      g.states["l13.v"].shape = [1, 2, CAPACITY, DEPTH];
    }, "正の整数倍でない"],
    ["window が容量を超える", (g) => {
      g.nodes[0].attrs = { scale: 0.5, window: CAPACITY + 1, readonly: true };
    }, "スロット容量"],
    ["k / v スロットが同形でない", (g) => {
      if (g.states === undefined) throw new Error("states が無い");
      g.states["l13.v"].shape = [1, KV_HEADS, CAPACITY * 2, DEPTH];
    }, "同形でない"],
  ];
  for (const [name, mutate, message] of cases) {
    const graph = parse(mutated(mutate));
    assertThrows(
      () => planGraph(graph, {}, slotShapes(graph)),
      OpContractError,
      message,
      `${name}: 拒否されていない`,
    );
  }
});

/** 貸し手の initializer 名（§1.3 — 借り手の宣言名は**貸し手と同じ** MUST）。 */
const SHARED_NAME = "model.lm_head.weight";

/** 共有 initializer（§1.3）を 1 本持つ drafter（embedding の重みを貸し手から借りる形）。 */
const sharedGraph = (): DeclarationJson => {
  const graph = drafterGraph();
  graph.requires.ops = ["attention", "embedding"];
  graph.inputs.push({ name: "token", dtype: "i32", shape: [1] });
  graph.outputs.push("embed");
  graph.initializers[SHARED_NAME] = { shared: true };
  graph.values[SHARED_NAME] = { dtype: "f32", shape: [16, DEPTH] };
  graph.values["embed"] = { dtype: "f32", shape: [1, DEPTH] };
  graph.nodes.push({
    op: "embedding",
    ins: [SHARED_NAME, "token"],
    outs: ["embed"],
    attrs: { padding_idx: -1 },
  });
  return graph;
};

Deno.test("共有 initializer は実体も格納も持たずに受理され、席が shared になる", () => {
  const graph = parse(sharedGraph());
  // 借り手の宣言名がそのまま貸し手の initializer 名（IR v2 では名前が実体の鍵）。
  assertEquals(graph.initializers[SHARED_NAME], { shared: true });
  validateGraphContracts(graph);
  // 席は「借り物」。期待する貸し手の席は借り手側の消費（適格判定 + チャネル軸 0）だけが決まり、
  // 貸し手の codec と突き合わせるのは借り手 Session の構築時。
  assertEquals(planWeightResidency(graph).get(SHARED_NAME), {
    seat: "shared",
    eligible: true,
    i4Eligible: true,
    i2Eligible: true,
    consumerAxis: 0,
  });
});

Deno.test("共有 initializer に実体の鍵や格納を書く形は拒否する", () => {
  // IR v2 の initializer 宣言に書けるのは `shared` だけ（実体との対応は容器の束縛表が持つ）。
  // 旧配布形の `tensor` / `storage` をそのまま書いた形は未知のキーとして落ちる。
  for (const key of ["tensor", "storage"]) {
    const graph = sharedGraph();
    Object.assign(graph.initializers[SHARED_NAME], { [key]: "model.lm_head.weight" });
    assertThrows(() => parse(graph), IrError, `未知のキー '${key}'`);
  }
  // `shared` に書けるのは `true` だけ（`false` は欄の不存在と同じ宣言 — 正準直列化が割れる）。
  const notTrue = sharedGraph();
  Object.assign(notTrue.initializers[SHARED_NAME], { shared: false });
  assertThrows(() => parse(notTrue), IrError, "借り物のときだけ true");
});
