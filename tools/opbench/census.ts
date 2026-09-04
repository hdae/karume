/**
 * 静的 census: IR グラフ 1 本 → **1 ノード 1 行**の表と、その集計。
 *
 * GPU も重みバイトも要らない。`bindSymbols` / `planGraph` / `planFusions` は全て純関数なので、
 * 先頭 shard のヘッダに載った IR にシナリオ束縛を与えるだけで「実行 1 回に出る dispatch の
 * 形」が確定する（ADR 0004 の静的形状 — 全ノードの出力 shape は 1 dispatch も出す前に決まる）。
 *
 * 用途は 2 つ。①候補起票の根拠を外挿から実数へ置き換える（strided 実体化の kind 別内訳など）
 * ②単体マイクロベンチの **census 加重**（同じ (op, shape, dtype, attrs, 格納) が実行 1 回に
 * 何本出るか）を与える。
 */

import type { IrDtype, IrGraph } from "../../packages/runtime/src/format/ir.ts";
import {
  bindSymbols,
  countUses,
  type NodePlan,
  planGraph,
  resolveShape,
} from "../../packages/runtime/src/runtime/plan.ts";
import {
  type ExecStep,
  type FusionCounts,
  planFusions,
} from "../../packages/runtime/src/runtime/fusion.ts";
import {
  type AssetTargets,
  type ComponentTarget,
  CORE_LIMITS,
  type SessionDeclaration,
} from "../_shared/assets.ts";
import { resolveComponentBindings, type Scenario } from "../_shared/scenario.ts";

/** 初期化子の格納（census 行の `storage` 欄 1 要素）。 */
export type StorageRef = {
  readonly tensor: string;
  readonly dtype: string;
  readonly group_size?: number;
  readonly scale?: string;
};

/** census.jsonl の 1 行（= IR ノード 1 本）。 */
export type CensusRow = {
  readonly family: string;
  readonly model: string;
  readonly quant: string;
  readonly component: string;
  readonly component_dtype: string;
  readonly scenario: string;
  /** このコンポーネントに効いた記号束縛（束縛の出どころ = `binding_source`）。 */
  readonly bindings: Readonly<Record<string, number>>;
  readonly binding_source: Scenario["source"];
  readonly node_index: number;
  readonly op: string;
  readonly in_shapes: readonly (readonly number[])[];
  readonly out_shapes: readonly (readonly number[])[];
  readonly in_dtypes: readonly IrDtype[];
  readonly out_dtypes: readonly IrDtype[];
  readonly attrs: Readonly<Record<string, unknown>>;
  /** `ins` と同順・同長（初期化子でない入力は null）。 */
  readonly storage: readonly (StorageRef | null)[];
  /** ノードが触れる state スロットの解決済み shape（触れないノードは空表）。 */
  readonly state_shapes: Readonly<Record<string, readonly number[]>>;
  /** 融合で消えたノードなら掴んだルール名、素のまま並ぶなら null。 */
  readonly fused_by: string | null;
  /** 出力が入力の別名になる（0 dispatch — reshape 常時 / 恒等 expand）。 */
  readonly aliases_input: boolean;
  readonly producers: readonly number[];
  readonly consumers: readonly number[];
};

/** 1 コンポーネント × 1 シナリオぶんの census。 */
export type ComponentCensus = {
  readonly rows: readonly CensusRow[];
  /** 束縛したがこのコンポーネントのどの shape も使わなかった記号（例: 容量が焼き込み済みの `C`）。 */
  readonly unusedBindings: readonly string[];
  /**
   * `planFusions` のルール別**ヒット数**（= 融合ステップの本数）。行の `fused_by` を数えると
   * 「畳まれたノード本数」になり、1 ヒット = 複数ノードなので値が別物になる。実資産の門
   * （packages/runtime/tests/assets_fusion_counts_test.ts）が固定しているのはこちら。
   */
  readonly fusionHits: FusionCounts;
};

/** census を採るときに 1 資産ぶん固定される欄。 */
export type AssetIdentity = {
  readonly family: string;
  readonly model: string;
  readonly quant: string;
};

/** ノード index → 融合ルール名 / 別名化フラグ。 */
type NodeFate = {
  readonly owners: ReadonlyMap<number, string>;
  readonly aliased: ReadonlySet<number>;
};

/**
 * 各ノードの行き先（融合で消えたか・素のまま並んだか・別名化されたか）を復元する。
 *
 * `planFusions` は畳んだノードそのものを返さない（{@link FusedStep} が持つのは本数だけ）ので、
 * 「素のまま並んだノード」の補集合として求める。融合ステップは元の並び順どおりに現れ、窓は
 * 重ならないので、**未出現ノードを昇順に並べて本数ぶんずつ配る**で帰属先が一意に決まる。
 * MUST: 配り終えた数が合わなければ fail loudly（合わないのは復元の前提が崩れたときで、
 * そのまま出すと融合の有無が 1 行ずつずれた census になる）。
 */
const nodeFate = (steps: readonly ExecStep[], planNodes: readonly NodePlan[]): NodeFate => {
  const indexOf = new Map(planNodes.map((plan, index) => [plan, index] as const));
  const emitted = new Set<number>();
  const aliased = new Set<number>();
  for (const step of steps) {
    if (step.kind !== "node") continue;
    const index = indexOf.get(step.plan);
    if (index === undefined) throw new Error("融合計画の素ノードが元のノード列に無い");
    emitted.add(index);
    if (step.aliasesInput) aliased.add(index);
  }
  const absorbed = planNodes.map((_, index) => index).filter((index) => !emitted.has(index));
  const owners = new Map<number, string>();
  let cursor = 0;
  for (const step of steps) {
    if (step.kind !== "fused") continue;
    for (let taken = 0; taken < step.nodeCount; taken += 1) {
      const index = absorbed[cursor];
      if (index === undefined) {
        throw new Error("融合ステップが畳んだノード数が未出現ノード数を超えている");
      }
      owners.set(index, step.rule);
      cursor += 1;
    }
  }
  if (cursor !== absorbed.length) {
    throw new Error(
      `融合で消えたノード ${absorbed.length} 本のうち ${cursor} 本しか帰属先が決まらない`,
    );
  }
  return { owners, aliased };
};

/** 1 コンポーネント × 1 シナリオの census を組む。 */
export const censusComponent = (
  graph: IrGraph,
  target: ComponentTarget,
  identity: AssetIdentity,
  scenario: Scenario,
): ComponentCensus => {
  // 束縛の判定（修飾キーの記号側の誤綴り・未束縛）は tools/_shared/scenario.ts の 1 本
  // （fusion-hints と同じ関数 = 同じ文言で落ちる）。
  const { bindings: bound, unused: unusedBindings } = resolveComponentBindings(
    scenario,
    target.component,
    graph,
  );

  const inputShapes = Object.fromEntries(
    graph.inputs.map((spec) => [spec.name, resolveShape(spec.shape, bound)]),
  );
  // 束縛の往復検査（入力 shape から解き直して宣言と全一致することを runtime と同じ規則で見る）。
  const resolved = bindSymbols(graph, inputShapes);
  const stateShapes = new Map(
    Object.entries(graph.states).map(([name, slot]) => [name, resolveShape(slot.shape, bound)]),
  );
  const plan = planGraph(
    graph,
    { ...bound, ...resolved },
    stateShapes.size === 0 ? undefined : stateShapes,
  );
  const fusion = planFusions(plan.nodes, {
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
    limits: CORE_LIMITS,
  });
  const { owners, aliased } = nodeFate(fusion.steps, plan.nodes);

  const producerOf = new Map<string, number>();
  const consumersOf = new Map<string, number[]>();
  graph.nodes.forEach((node, index) => {
    for (const name of node.outs) producerOf.set(name, index);
    for (const name of node.ins) {
      const list = consumersOf.get(name) ?? [];
      list.push(index);
      consumersOf.set(name, list);
    }
  });

  const rows = plan.nodes.map((node, index): CensusRow => {
    const producers = [
      ...new Set(
        node.node.ins.map((name) => producerOf.get(name)).filter((at) => at !== undefined),
      ),
    ].sort((a, b) => a - b);
    const consumers = [
      ...new Set(
        node.node.outs.flatMap((name) => consumersOf.get(name) ?? []),
      ),
    ].sort((a, b) => a - b);
    return {
      family: identity.family,
      model: identity.model,
      quant: identity.quant,
      component: target.component,
      component_dtype: target.componentDtype,
      scenario: scenario.name,
      bindings: bound,
      binding_source: scenario.source,
      node_index: index,
      op: node.node.op,
      in_shapes: node.inputShapes,
      out_shapes: node.outputs.map((out) => out.shape),
      in_dtypes: node.inputDtypes,
      out_dtypes: node.outputs.map((out) => out.dtype),
      attrs: node.node.attrs,
      storage: node.node.ins.map((name) => storageOf(graph, name)),
      state_shapes: Object.fromEntries(
        Object.entries(node.node.states).map(([role, slot]) => {
          const shape = stateShapes.get(slot);
          if (shape === undefined) throw new Error(`state スロット '${slot}' の宣言が無い`);
          return [role, shape];
        }),
      ),
      fused_by: owners.get(index) ?? null,
      aliases_input: aliased.has(index),
      producers,
      consumers,
    };
  });
  return { rows, unusedBindings, fusionHits: fusion.counts };
};

const storageOf = (graph: IrGraph, name: string): StorageRef | null => {
  if (!Object.hasOwn(graph.initializers, name)) return null;
  const initializer = graph.initializers[name];
  return {
    tensor: initializer.tensor,
    dtype: initializer.storage.dtype,
    ...(initializer.storage.groupSize === undefined
      ? {}
      : { group_size: initializer.storage.groupSize }),
    ...(initializer.storage.scale === undefined ? {} : { scale: initializer.storage.scale }),
  };
};

/** ノード 1 本の出力要素数の合計（strided 実体化の「運ぶ量」の指標）。 */
const outElements = (row: CensusRow): number =>
  row.out_shapes.reduce((total, shape) => total + shape.reduce((size, dim) => size * dim, 1), 0);

/**
 * 加重行 1 入力スロットぶんの格納。census 行の {@link StorageRef} から**テンソル名**
 * （`tensor` / `scale`）を落としてある — 加重行は層をまたいで畳んだ行なので、代表 1 本の
 * safetensors キーを載せると「この行のテンソル」と読めてしまう。カーネルの分かれ目になるのは
 * dtype と group 長だけで、そこは残す。
 */
export type WeightStorage = {
  readonly dtype: string;
  readonly group_size?: number;
};

/** census 行の格納列 → 加重行の格納列（`ins` 同順・初期化子でない入力は null）。 */
const weightStorage = (row: CensusRow): readonly (WeightStorage | null)[] =>
  row.storage.map((ref) =>
    ref === null ? null : {
      dtype: ref.dtype,
      ...(ref.group_size === undefined ? {} : { group_size: ref.group_size }),
    }
  );

/**
 * ノードの格納シグネチャ（初期化子入力の格納 dtype を重複無しで並べたもの）。
 * `i4g32` のように group 長まで含める — 同じ i4 でも group 長でカーネルが変わる。
 */
const storageSignature = (row: CensusRow): string => {
  const tokens = new Set(
    row.storage.filter((ref) => ref !== null).map((ref) =>
      ref.group_size === undefined ? ref.dtype : `${ref.dtype}g${ref.group_size}`
    ),
  );
  return tokens.size === 0 ? "none" : [...tokens].sort().join("+");
};

/**
 * census 加重の 1 行（同一の (op, shape, dtype, attrs, 格納, 融合) が実行 1 回に何本出るか）。
 *
 * MUST: 行が名乗る欄は全て {@link tallyKey} に入っている（= 畳んだ行の中で値が割れない）。
 * 代表 1 本の値を全体の値として載せると、単体ベンチが実在しない形を測ることになる。
 */
export type WeightRow = {
  readonly component: string;
  readonly op: string;
  readonly in_shapes: readonly (readonly number[])[];
  readonly out_shapes: readonly (readonly number[])[];
  readonly in_dtypes: readonly IrDtype[];
  readonly out_dtypes: readonly IrDtype[];
  /**
   * op の属性（IR そのまま）。**加重キーの一部** — 同じ shape / dtype でも `permute` の `dims`
   * や `slice` の範囲が違えばメモリアクセス形が別物で、単体ベンチでは別ケースになる。
   */
  readonly attrs: Readonly<Record<string, unknown>>;
  /** `ins` 同順の格納（初期化子でない入力は null）。 */
  readonly storage: readonly (WeightStorage | null)[];
  /** 格納の集合シグネチャ（`f32+i4g32` など — `by_storage` の集計キーと同じ綴り）。 */
  readonly storage_signature: string;
  readonly fused_by: string | null;
  readonly aliases_input: boolean;
  /** census 加重（この形のノード本数）。 */
  readonly count: number;
  /** 1 本ぶんの出力要素数。 */
  readonly out_elements: number;
};

export type OpTally = { readonly nodes: number; readonly out_elements: number };

/** 1 シナリオぶんの集計。 */
export type ScenarioSummary = {
  readonly scenario: string;
  readonly bindings: Readonly<Record<string, number>>;
  readonly binding_source: Scenario["source"];
  readonly provenance: string;
  /** `<component>:<記号>` — 束縛したが shape が使わなかったもの（容量焼き込み済みの `C` など）。 */
  readonly unused_bindings: readonly string[];
  readonly node_count: number;
  readonly components: Readonly<Record<string, number>>;
  /** op 別（本数と出力要素数の合計）。 */
  readonly by_op: Readonly<Record<string, OpTally>>;
  /** 格納別（初期化子の格納 dtype シグネチャ → 本数）。 */
  readonly by_storage: Readonly<Record<string, number>>;
  /**
   * 融合の有無別。`absorbed` はルール名別に**畳まれたノード本数**、`hits` は**融合ステップの
   * 本数**（1 ヒットが複数ノードを畳むので両者は別物 — 実資産の門が固定しているのは `hits`）、
   * `plain` は素のまま並ぶ本数、`aliased` はそのうち 0 dispatch の別名化。
   */
  readonly by_fusion: {
    readonly absorbed: Readonly<Record<string, number>>;
    readonly hits: Readonly<Record<string, number>>;
    readonly plain: number;
    readonly aliased: number;
  };
  readonly weights: readonly WeightRow[];
};

export type CensusSummary = {
  readonly generated_at: string;
  readonly source: string;
  readonly family: string;
  readonly model: string;
  readonly quant: string;
  /**
   * この quant が宣言した実行変種の**逐語の写し**（manifest 所有の綴りのまま）。系列出力は
   * manifest を持たないので `null` = 実行変種は宣言されていない（呼び手が与える）。
   */
  readonly session: SessionDeclaration | null;
  readonly scenarios: readonly ScenarioSummary[];
};

/**
 * summary.json のヘッダを組む（1 実行 = 1 資産）。
 *
 * `session` の `null` は「宣言が無い」であって「ノブ指定なし」ではない — 配布形は宣言の欄が
 * 空でも `{}` を写す（{@link AssetTargets} の `session` を参照）。
 */
export const buildCensusSummary = (
  source: string,
  asset: AssetTargets,
  scenarios: readonly ScenarioSummary[],
): CensusSummary => ({
  generated_at: new Date().toISOString(),
  source,
  family: asset.family,
  model: asset.model,
  quant: asset.quant,
  session: asset.session ?? null,
  scenarios,
});

/**
 * キー順に依存しない JSON 文字列化（{@link tallyKey} が attrs を混ぜるため）。
 *
 * MUST: オブジェクトはキーを並べ替えてから綴る。attrs は IR の JSON をそのまま持つので、
 * 同じ属性でも書いた側のキー順が違えば `JSON.stringify` の結果は割れる。加重キーが綴り順で
 * 割れると「相異なる形の本数」が資産の書き出し順に依存してしまう。
 */
const stableJson = (value: unknown): string => {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  return `{${
    Object.entries(value)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, nested]) => `${JSON.stringify(key)}:${stableJson(nested)}`)
      .join(",")
  }}`;
};

/**
 * 加重を畳むキー。
 *
 * 格納はシグネチャ（集合）ではなく**スロット同順の列**で持つ — `linear` の `[x, W, bias]` で
 * どのスロットが i4 でどれが f32 かは単体ベンチのカーネル選択そのもので、集合に潰すと
 * 割り当ての違う 2 形が 1 行に畳まれて代表 1 本の割り当てだけが残る。
 */
const tallyKey = (row: CensusRow): string =>
  JSON.stringify([
    row.component,
    row.op,
    row.in_shapes,
    row.out_shapes,
    row.in_dtypes,
    row.out_dtypes,
    stableJson(row.attrs),
    weightStorage(row),
    row.fused_by,
    row.aliases_input,
  ]);

/** 1 シナリオぶんの行から集計を作る。 */
export const summarizeScenario = (
  scenario: Scenario,
  rows: readonly CensusRow[],
  unusedBindings: readonly string[],
  fusionHits: Readonly<Record<string, number>>,
): ScenarioSummary => {
  const byOp = new Map<string, { nodes: number; out_elements: number }>();
  const byStorage = new Map<string, number>();
  const fused = new Map<string, number>();
  const components = new Map<string, number>();
  const weights = new Map<string, { row: CensusRow; count: number }>();
  let plain = 0;
  let aliased = 0;
  for (const row of rows) {
    const elements = outElements(row);
    const op = byOp.get(row.op) ?? { nodes: 0, out_elements: 0 };
    byOp.set(row.op, { nodes: op.nodes + 1, out_elements: op.out_elements + elements });
    const signature = storageSignature(row);
    byStorage.set(signature, (byStorage.get(signature) ?? 0) + 1);
    components.set(row.component, (components.get(row.component) ?? 0) + 1);
    if (row.fused_by === null) {
      plain += 1;
      if (row.aliases_input) aliased += 1;
    } else {
      fused.set(row.fused_by, (fused.get(row.fused_by) ?? 0) + 1);
    }
    const key = tallyKey(row);
    const entry = weights.get(key);
    if (entry === undefined) weights.set(key, { row, count: 1 });
    else entry.count += 1;
  }
  const sortedWeights = [...weights.values()]
    .map(({ row, count }): WeightRow => ({
      component: row.component,
      op: row.op,
      in_shapes: row.in_shapes,
      out_shapes: row.out_shapes,
      in_dtypes: row.in_dtypes,
      out_dtypes: row.out_dtypes,
      attrs: row.attrs,
      storage: weightStorage(row),
      storage_signature: storageSignature(row),
      fused_by: row.fused_by,
      aliases_input: row.aliases_input,
      count,
      out_elements: outElements(row),
    }))
    // 完全等値のときは 0 を返す（比較子の契約 — `1` だと「a > b かつ b > a」を主張する形になる）。
    .sort((a, b) =>
      b.count - a.count || b.out_elements - a.out_elements ||
      (a.op < b.op ? -1 : a.op > b.op ? 1 : 0)
    );
  return {
    scenario: scenario.name,
    bindings: scenario.bindings,
    binding_source: scenario.source,
    provenance: scenario.provenance,
    unused_bindings: unusedBindings,
    node_count: rows.length,
    components: Object.fromEntries([...components].sort()),
    by_op: Object.fromEntries([...byOp].sort((a, b) => b[1].nodes - a[1].nodes)),
    by_storage: Object.fromEntries([...byStorage].sort((a, b) => b[1] - a[1])),
    by_fusion: {
      absorbed: Object.fromEntries([...fused].sort((a, b) => b[1] - a[1])),
      hits: Object.fromEntries(
        Object.entries(fusionHits).filter(([, count]) => count > 0).sort((a, b) => b[1] - a[1]),
      ),
      plain,
      aliased,
    },
    weights: sortedWeights,
  };
};
