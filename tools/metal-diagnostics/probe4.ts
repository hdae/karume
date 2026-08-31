// probe4 — 実グラフの接頭辞二分探索で「最初に NaN を作るノード」を特定する。
//
// probe3a（実プロファイル 835 本の累積確保）と probe3b（実バイト梯子 23 段）が「単体は全て
// 無罪」を出した後に残る容疑は**統合機構**だけになる — 融合鎖（silu / rope / adaln /
// upsample2x / rowBlockAttention）・run 内 temp プールの別名化とディスパッチ間同期・
// uniform params 経路。probe4 は製品グラフ（models/karume-gemma4-e2b の `karume_ir`・
// 1,504 ノード）の**先頭 K ノードだけ**を切り出した接頭辞グラフを実重みで組んで実行し、
// 末端テンソルの NaN 数を K の関数として観測する。
//
// ## MUST: 観測が挙動を変えうることを設計に織り込む
//
// 接頭辞化は次の 3 点で製品実行と条件が変わる:
//   ① 融合窓が K で切れる（鎖の途中で切ると silu / rope / adaln 融合はそもそも成立しない）
//   ② 末端を graph output にする（融合ルールの「内部値が private」判定が外れ、鎖が解ける）
//   ③ 参照される initializer だけを配布形に詰めるので、常駐バイト列と temp プールの
//      寿命表・別名化パターンが製品と一致しない
// したがって K を縮めた瞬間に NaN が「消える」ことがありうる。**それ自体が統合機構犯の強い
// シグナル**なので、`--bisect` は最後に必ず K = 全ノードを撃ち直して再現性を報告する。
// 各行に融合の適用回数（`lastRunFusions`）を出すのは、①②が効いたかどうかを事後に読むため。
//
// ## 終端 state_append の持ち上げ（lift）
//
// KV 共有（ADR 0067 決定 5）のせいで、素朴に切った接頭辞は 1,504 点中 724 点しか
// ランタイムの state 順序契約（決定 5b）を満たさない — `l13.*` は nodes[721] から読まれるのに
// 終端 `state_append` が nodes[1437] まで来ないので、K ∈ [722, 1475] が丸ごと組めなくなる。
// {@link closeStateSlots} が全体グラフの終端 append ノードを**逐語で**末尾へ持ち上げることで
// 全 K を到達可能にしてある（1 run では append は全読者より後に走るので、読みは 1 バイトも
// 変わらない）。持ち上げた本数は報告行の `lift=` に出る。
//
// ## 入力
//
// T=1（`input_ids` = [--token]・M=1・1 run）が既定。`per_layer_inputs` は実 PLE sidecar から
// ホスト gather（packages/models の実装をそのまま使う — 逆量子化の丸め順は ADR 0085 決定 4）。
// state スロットを含む接頭辞では GenerationContext を 1 本作って `queryLength = 1` で撃つ。
//
// 使い方（リポルートで）:
//   deno run -A tools/metal-diagnostics/probe4.ts --at 43
//   deno run -A tools/metal-diagnostics/probe4.ts --coarse 128
//   deno run -A tools/metal-diagnostics/probe4.ts --bisect
//   共通ノブ: --token ID（既定 2）/ --capacity C（既定 640）/ --chunk-length L（既定 32）

import { parseDim } from "../../packages/runtime/src/format/dims.ts";
import { acquireGpu } from "../../packages/runtime/src/gpu/device.ts";
import {
  createSessionFromShards,
  type ModelShard,
} from "../../packages/runtime/src/runtime/executor.ts";
import type { RunInputs } from "../../packages/runtime/src/runtime/session-types.ts";
import {
  buildSafetensors,
  type GraphJson,
  type TensorSpec,
} from "../../packages/runtime/tests/helpers/format.ts";
import { createGemma4Ple, parseGemma4PleIndex } from "../../packages/models/src/gemma/ple.ts";

// ---------------------------------------------------------------------------
// 資産の場所（probe3a / probe3b と同じミラー）
// ---------------------------------------------------------------------------

const MIRROR = "models/karume-gemma4-e2b/e2b";
const MODEL_SHARDS = [
  `${MIRROR}/model/model.i4-00001-of-00003.safetensors`,
  `${MIRROR}/model/model.i4-00002-of-00003.safetensors`,
  `${MIRROR}/model/model.i4-00003-of-00003.safetensors`,
];
const PLE_DIR = `${MIRROR}/ple`;

/**
 * 1 本の重み shard に詰めるバイト数の目安。**下回れないケースが 1 つある** — 実 lm_head は
 * 単体 384MiB なので、それだけで 1 本の shard になる（1 initializer は必ず 1 本に収める）。
 * 小さくしておくのはホスト RAM のピークを抑えるためで、shard 本数が増えると Metal では
 * shard ごとのフェンス（既知の ≈11ms/shard）が積み上がるだけ。
 */
const SHARD_BUDGET_BYTES = 192 * 1024 * 1024;

// ---------------------------------------------------------------------------
// ミラーのテンソル索引（probe3b と同じ「ヘッダだけ読む」形）
// ---------------------------------------------------------------------------

type Located = {
  readonly path: string;
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly start: number;
  readonly end: number;
};

const index = new Map<string, Located>();
let graphJsonText = "";

const readHeader = async (
  path: string,
): Promise<{ header: Record<string, unknown>; dataStart: number }> => {
  const file = await Deno.open(path);
  try {
    const head = new Uint8Array(8);
    await file.read(head);
    const length = Number(new DataView(head.buffer).getBigUint64(0, true));
    const body = new Uint8Array(length);
    let filled = 0;
    while (filled < length) {
      const read = await file.read(body.subarray(filled));
      if (read === null) throw new Error(`${path}: ヘッダが途中で尽きた`);
      filled += read;
    }
    return {
      header: JSON.parse(new TextDecoder().decode(body)) as Record<string, unknown>,
      dataStart: 8 + length,
    };
  } finally {
    file.close();
  }
};

const buildIndex = async (): Promise<void> => {
  for (const path of MODEL_SHARDS) {
    const { header, dataStart } = await readHeader(path);
    const metadata = header["__metadata__"];
    if (metadata !== undefined && typeof metadata === "object" && metadata !== null) {
      const ir = (metadata as Record<string, unknown>)["karume_ir"];
      if (typeof ir === "string") graphJsonText = ir;
    }
    for (const [name, raw] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      const entry = raw as { dtype: string; shape: number[]; data_offsets: [number, number] };
      index.set(name, {
        path,
        dtype: entry.dtype,
        shape: entry.shape,
        start: dataStart + entry.data_offsets[0],
        end: dataStart + entry.data_offsets[1],
      });
    }
  }
  if (graphJsonText === "") throw new Error("ミラーの先頭 shard に karume_ir が無い");
};

/** テンソル 1 本の実バイトを読む（1.5GiB のミラー全体はメモリに載せない）。 */
const readTensor = async (name: string): Promise<TensorSpec> => {
  const located = index.get(name);
  if (located === undefined) throw new Error(`テンソル '${name}' がミラーに無い`);
  const file = await Deno.open(located.path);
  try {
    await file.seek(located.start, Deno.SeekMode.Start);
    const data = new Uint8Array(located.end - located.start);
    let filled = 0;
    while (filled < data.length) {
      const read = await file.read(data.subarray(filled));
      if (read === null) throw new Error(`'${name}' が途中で尽きた`);
      filled += read;
    }
    return { name, dtype: located.dtype, shape: [...located.shape], data };
  } finally {
    file.close();
  }
};

// ---------------------------------------------------------------------------
// 接頭辞グラフの組み立て
// ---------------------------------------------------------------------------

/** 次元式に現れるシンボル名を集める。 */
const symbolsIn = (shape: readonly (number | string)[], into: Set<string>): void => {
  for (const dim of shape) {
    if (typeof dim === "string") into.add(parseDim(dim).sym);
  }
};

type Prefix = {
  readonly k: number;
  readonly graph: GraphJson;
  /** 末端テンソル（graph output に据えた値）と、それを定義したノード。 */
  readonly terminal: string;
  readonly terminalNode: number;
  readonly terminalOp: string;
  /** 実バイトを詰める必要のある initializer 名（宣言順）。 */
  readonly initializerNames: readonly string[];
  /** 接頭辞が state スロットを含むか（GenerationContext の要否）。 */
  readonly stateSymbols: readonly string[];
  readonly stateSlots: number;
  /** 末尾へ持ち上げた終端 `state_append` の本数（{@link closeStateSlots}）。 */
  readonly liftedAppends: number;
};

/** スロット名 → そのスロットの終端 `state_append` のノード添字（全体グラフ）。 */
const appendIndexBySlot = (full: GraphJson): ReadonlyMap<string, number> => {
  const found = new Map<string, number>();
  full.nodes.forEach((node, at) => {
    if (node.op !== "state_append") return;
    const slot = (node.states ?? {})["slot"];
    if (slot !== undefined) found.set(slot, at);
  });
  return found;
};

/**
 * 接頭辞で**読まれたのに終端 `state_append` が落ちたスロット**に対して、全体グラフの
 * 終端 append ノードを**そのまま末尾へ持ち上げる**。
 *
 * これが要る理由は KV 共有（ADR 0067 決定 5）— gemma4 e2b は 35 層が 15 スロットを共有し、
 * `l13.*` は nodes[721] から読まれるのに終端 append が nodes[1437] まで来ない。ランタイムの
 * state 順序契約（ADR 0067 決定 5b: 読者が居るスロットには終端 append がちょうど 1 本・かつ
 * それがそのスロットに触れる最後のノード）は Session 構築時に効くので、持ち上げが無いと
 * **K ∈ [722, 1475] が丸ごと組めない**（実測: 1,504 点中 724 点しか合法にならない）。
 *
 * MUST: 合成せず**実ノードを逐語で**持ち上げる — `ins` も `attrs.window` も製品と同一に
 * なるので、スロットに書かれる値も物理 row の写像も製品と同じ。1 run では append は全ての
 * 読者より後に走るため、この持ち上げは当該 run の読みを 1 バイトも変えない（変えるのは
 * dispatch が数本増えることだけで、それは報告行の `lift=` に出す）。
 * MUST: 持ち上げ先の入力が接頭辞で定義済みでなければ fail loudly（黙って別の値を書かない）。
 */
const closeStateSlots = (
  full: GraphJson,
  nodes: GraphJson["nodes"],
  appends: ReadonlyMap<string, number>,
  produced: ReadonlySet<string>,
  k: number,
): GraphJson["nodes"] => {
  const open = new Set<string>();
  for (const node of nodes) {
    for (const slot of Object.values(node.states ?? {})) open.add(slot);
  }
  for (const node of nodes) {
    if (node.op !== "state_append") continue;
    const slot = (node.states ?? {})["slot"];
    if (slot !== undefined) open.delete(slot);
  }
  return [...open]
    .map((slot) => {
      const at = appends.get(slot);
      if (at === undefined) throw new Error(`K=${k}: スロット '${slot}' の終端 append が無い`);
      return at;
    })
    .sort((left, right) => left - right)
    .map((at) => {
      const node = full.nodes[at];
      if (!produced.has(node.ins[0])) {
        throw new Error(
          `K=${k}: nodes[${at}] (state_append) の入力 '${node.ins[0]}' が接頭辞で未定義`,
        );
      }
      return node;
    });
};

/**
 * 先頭 `k` ノードだけを残した接頭辞グラフを組む。
 *
 * MUST: IR v1 の宣言完全性（`format/ir.ts`）を全部満たす形で組む — 孤立 values / 未参照
 * state スロット / `requires.ops` の過不足はどれも parse で落ちる。落とさずに済ませようと
 * 全宣言を残すと、参照されない initializer が 1.5GiB まるごと常駐して「接頭辞を撃つ」意味が
 * 消える（未使用 initializer にも構築時の GPU 常駐席が与えられる契約 — ADR 0085 Context）。
 */
const buildPrefix = (full: GraphJson, k: number, appends: ReadonlyMap<string, number>): Prefix => {
  if (k < 1 || k > full.nodes.length) {
    throw new Error(`K=${k} が 1..${full.nodes.length} の外`);
  }
  const head = full.nodes.slice(0, k);

  // 末端は「最後に f32 の値を定義したノード」— state_append のような outs 0 本の effect op で
  // 切れた接頭辞でも、直近の観測可能な値まで遡って読む（i32 / bool の値は NaN を持てない）。
  // MUST: 探すのは持ち上げ前の `head` だけ（持ち上げた append は製品の並びで言えば K より
  // 後のノードなので、末端の帰属をそこへ動かすと K の意味が壊れる）。
  let terminal: string | undefined;
  let terminalNode = -1;
  const produced = new Set<string>();
  for (const node of head) {
    for (const out of node.outs) produced.add(out);
  }
  for (let at = head.length - 1; at >= 0 && terminal === undefined; at -= 1) {
    for (const out of head[at].outs) {
      if (full.values[out]?.dtype !== "f32") continue;
      terminal = out;
      terminalNode = at;
      break;
    }
  }
  if (terminal === undefined) throw new Error(`K=${k}: 接頭辞に f32 のノード出力が 1 つも無い`);

  const lifted = closeStateSlots(full, head, appends, produced, k);
  const nodes = [...head, ...lifted];

  const consumed = new Set<string>();
  const slots = new Set<string>();
  const ops = new Set<string>();
  for (const node of nodes) {
    ops.add(node.op);
    for (const name of node.ins) consumed.add(name);
    for (const slot of Object.values(node.states ?? {})) slots.add(slot);
  }

  const initializerNames = Object.keys(full.initializers).filter((name) => consumed.has(name));
  const inputs = full.inputs.filter((spec) => consumed.has(spec.name));

  const values: GraphJson["values"] = {};
  for (const name of initializerNames) values[name] = full.values[name];
  for (const node of nodes) {
    for (const out of node.outs) values[out] = full.values[out];
  }

  const initializers: GraphJson["initializers"] = {};
  for (const name of initializerNames) initializers[name] = full.initializers[name];

  const states: GraphJson["states"] = {};
  const stateSymbols = new Set<string>();
  for (const name of Object.keys(full.states ?? {})) {
    if (!slots.has(name)) continue;
    const slot = (full.states ?? {})[name];
    states[name] = slot;
    symbolsIn(slot.shape, stateSymbols);
  }

  // 宣言できるシンボルは束縛点（入力 shape / states shape）を持つものだけ（ir.ts の
  // checkSymbolBindability）。値 shape だけに現れる記号が残ったら接頭辞の切り方が壊れている
  // ので黙って通さない。
  const declared = new Set<string>();
  for (const spec of inputs) symbolsIn(spec.shape, declared);
  for (const symbol of stateSymbols) declared.add(symbol);
  const used = new Set<string>();
  for (const value of Object.values(values)) symbolsIn(value.shape, used);
  const orphan = [...used].filter((symbol) => !declared.has(symbol));
  if (orphan.length > 0) {
    throw new Error(
      `K=${k}: 値 shape の記号 [${orphan.join(", ")}] を束縛できる入力 / state が接頭辞に無い`,
    );
  }

  const graph: GraphJson = {
    format: full.format,
    version: full.version,
    requires: { ops: [...ops] },
    symbols: full.symbols.filter((symbol) => declared.has(symbol)),
    inputs,
    outputs: [terminal],
    initializers,
    values,
    ...(Object.keys(states).length === 0 ? {} : { states }),
    nodes,
  };

  return {
    k,
    graph,
    terminal,
    terminalNode,
    terminalOp: head[terminalNode].op,
    initializerNames,
    stateSymbols: [...stateSymbols],
    stateSlots: slots.size,
    liftedAppends: lifted.length,
  };
};

// ---------------------------------------------------------------------------
// 配布形（グラフ shard 1 本 + 参照される initializer だけの重み shard 列）
// ---------------------------------------------------------------------------

const shardSequence = async function* (prefix: Prefix): AsyncGenerator<ModelShard> {
  yield {
    id: "probe4-graph",
    bytes: new Uint8Array(
      buildSafetensors([], { karume_ir: JSON.stringify(prefix.graph) }),
    ),
  };
  let group: TensorSpec[] = [];
  let bytes = 0;
  let serial = 0;
  const pack = (): ModelShard => {
    serial += 1;
    const shard: ModelShard = {
      id: `probe4-weights-${serial}`,
      bytes: new Uint8Array(buildSafetensors(group)),
    };
    group = [];
    bytes = 0;
    return shard;
  };
  for (const name of prefix.initializerNames) {
    const initializer = prefix.graph.initializers[name];
    // companion scale は実体と**同じ shard**に置く MUST（ADR 0070 決定 1 の co-shard 契約）。
    const specs = [await readTensor(initializer.tensor)];
    const scale = initializer.storage["scale"];
    if (typeof scale === "string") specs.push(await readTensor(scale));
    const size = specs.reduce((total, spec) => total + spec.data.length, 0);
    if (group.length > 0 && bytes + size > SHARD_BUDGET_BYTES) yield pack();
    group.push(...specs);
    bytes += size;
  }
  if (group.length > 0) yield pack();
};

// ---------------------------------------------------------------------------
// 1 評価
// ---------------------------------------------------------------------------

type Stats = {
  readonly count: number;
  readonly nan: number;
  readonly inf: number;
  readonly min: number;
  readonly max: number;
};

const summarize = (data: ArrayLike<number>): Stats => {
  let nan = 0;
  let inf = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let at = 0; at < data.length; at += 1) {
    const value = data[at];
    if (Number.isNaN(value)) {
      nan += 1;
      continue;
    }
    if (!Number.isFinite(value)) {
      inf += 1;
      continue;
    }
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return { count: data.length, nan, inf, min, max };
};

type Evaluation = Stats & {
  readonly k: number;
  readonly terminal: string;
  readonly terminalNode: number;
  readonly terminalOp: string;
  readonly initializers: number;
  readonly liftedAppends: number;
  readonly fusions: string;
  readonly elapsedMs: number;
};

type Probe = {
  readonly full: GraphJson;
  readonly appends: ReadonlyMap<string, number>;
  readonly gpu: Awaited<ReturnType<typeof acquireGpu>>;
  readonly baseInputs: RunInputs;
  readonly capacity: number;
  readonly chunkLength: number;
};

const evaluate = async (probe: Probe, k: number): Promise<Evaluation> => {
  const started = performance.now();
  const prefix = buildPrefix(probe.full, k, probe.appends);
  const session = await createSessionFromShards(probe.gpu, shardSequence(prefix));
  try {
    // 宣言された入力だけを渡す（実行器は graph.inputs を全件転送するので、余分な名前は拒否）。
    const inputs: Record<string, RunInputs[string]> = {};
    for (const spec of prefix.graph.inputs) {
      const value = probe.baseInputs[spec.name];
      if (value === undefined) throw new Error(`K=${k}: 入力 '${spec.name}' の作り手が無い`);
      inputs[spec.name] = value;
    }
    const bindings: Record<string, number> = {};
    for (const symbol of prefix.stateSymbols) {
      if (symbol !== "C") throw new Error(`K=${k}: 未知の state 記号 '${symbol}'`);
      bindings[symbol] = probe.capacity;
    }
    const context = prefix.stateSlots === 0 ? undefined : await session.createGenerationContext({
      bindings,
      chunkLength: probe.chunkLength,
    });
    try {
      const outputs = await session.run(
        inputs,
        undefined,
        context === undefined ? undefined : { context, queryLength: 1 },
      );
      const stats = summarize(outputs[prefix.terminal].data);
      const counts: Readonly<Record<string, number>> = session.diagnostics().lastRunFusions ?? {};
      const fusions = Object.entries(counts)
        .filter(([, times]) => times > 0)
        .map(([rule, times]) => `${rule}:${times}`)
        .join(",");
      return {
        ...stats,
        k,
        terminal: prefix.terminal,
        terminalNode: prefix.terminalNode,
        terminalOp: prefix.terminalOp,
        initializers: prefix.initializerNames.length,
        liftedAppends: prefix.liftedAppends,
        fusions: fusions === "" ? "-" : fusions,
        elapsedMs: performance.now() - started,
      };
    } finally {
      await context?.dispose();
    }
  } finally {
    await session.dispose();
  }
};

const report = (evaluation: Evaluation): void => {
  const range = evaluation.nan === evaluation.count
    ? "min/max=-"
    : `min=${evaluation.min.toExponential(3)} max=${evaluation.max.toExponential(3)}`;
  console.log(
    `K=${String(evaluation.k).padStart(4)} ` +
      `node[${String(evaluation.terminalNode).padStart(4)}] ${evaluation.terminalOp} ` +
      `'${evaluation.terminal}' n=${evaluation.count} ` +
      `NaN=${evaluation.nan} Inf=${evaluation.inf} ${range} ` +
      `| init=${evaluation.initializers} lift=${evaluation.liftedAppends} ` +
      `fus=${evaluation.fusions} ` +
      `${(evaluation.elapsedMs / 1000).toFixed(2)}s`,
  );
};

// ---------------------------------------------------------------------------
// 入力の材料（T=1 の 4 本）
// ---------------------------------------------------------------------------

const buildBaseInputs = async (token: number): Promise<RunInputs> => {
  const indexJson: unknown = JSON.parse(await Deno.readTextFile(`${PLE_DIR}/ple.json`));
  const pleIndex = parseGemma4PleIndex(indexJson);
  const lmHead = index.get("model.lm_head.weight");
  if (lmHead === undefined) throw new Error("model.lm_head.weight がミラーに無い");
  const ple = createGemma4Ple({
    index: pleIndex,
    vocabSize: lmHead.shape[0],
    readShard: async (file) => (await Deno.readFile(`${PLE_DIR}/${file}`)).buffer as ArrayBuffer,
  });
  try {
    return {
      input_ids: { dtype: "i32", shape: [1, 1], data: Int32Array.of(token) },
      position_ids: { dtype: "i32", shape: [1, 1], data: Int32Array.of(0) },
      last_row: { dtype: "i32", shape: [1], data: Int32Array.of(0) },
      per_layer_inputs: await ple.gather([token]),
    };
  } finally {
    // sidecar shard は 758MB 級。1 token 引いたら常駐は要らない。
    ple.dispose();
  }
};

// ---------------------------------------------------------------------------

const arg = (name: string): string | undefined => {
  const at = Deno.args.indexOf(name);
  return at < 0 ? undefined : Deno.args[at + 1];
};
const numberArg = (name: string, fallback: number): number => {
  const raw = arg(name);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value)) throw new Error(`${name} が整数でない: ${raw}`);
  return value;
};

await buildIndex();
const full = JSON.parse(graphJsonText) as GraphJson;
const total = full.nodes.length;
console.log(
  `[probe4] グラフ ${total} ノード / initializer ${Object.keys(full.initializers).length} 本 / ` +
    `state スロット ${Object.keys(full.states ?? {}).length} 本 / ミラー索引 ${index.size} 本`,
);

const token = numberArg("--token", 2);
const capacity = numberArg("--capacity", 640);
const chunkLength = numberArg("--chunk-length", 32);
const baseInputs = await buildBaseInputs(token);
console.log(`[probe4] 入力 T=1 token=${token} capacity=${capacity} chunkLength=${chunkLength}`);

const gpu = await acquireGpu();
console.log(
  `[probe4] device limits: maxBufferSize=${gpu.limits.maxBufferSize} ` +
    `maxStorageBufferBindingSize=${gpu.limits.maxStorageBufferBindingSize}`,
);
const probe: Probe = {
  full,
  appends: appendIndexBySlot(full),
  gpu,
  baseInputs,
  capacity,
  chunkLength,
};

try {
  const at = arg("--at");
  const coarse = arg("--coarse");
  if (at !== undefined) {
    report(await evaluate(probe, numberArg("--at", 1)));
  } else if (coarse !== undefined) {
    const step = numberArg("--coarse", 128);
    if (step < 1) throw new Error("--coarse は 1 以上");
    const points: number[] = [];
    for (let k = step; k < total; k += step) points.push(k);
    points.push(total);
    for (const k of points) report(await evaluate(probe, k));
  } else if (Deno.args.includes("--bisect")) {
    // 二分の前提は「NaN は K について単調（一度出たら消えない）」。製品グラフでそれが成り
    // 立たない可能性は残るので、確定表示には必ず --coarse の粗マップを添えて読むこと。
    const seed = await evaluate(probe, total);
    report(seed);
    if (seed.nan === 0) {
      console.log("");
      console.log(
        "[probe4] 判定: K=全ノードでも NaN=0 — この機では二分探索が成立しない。読み方は 2 通り: " +
          "①製品グラフがそもそも緑の機（Linux 等）= 期待どおり ②製品経路が NaN を出す機なら、" +
          "K=全ノードは製品と同じ 1,504 ノード・同じ 558 initializer・同じ出力なので、" +
          "残る差は**配布形の詰め方（probe4 が組み直した shard 列）と入力の供給元だけ**に絞れる" +
          "（融合鎖は lift=0・fus= の値が製品と同じなら同一）。",
      );
    } else {
      let low = 1;
      let high = total;
      let culprit = seed;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        const evaluation = await evaluate(probe, middle);
        report(evaluation);
        if (evaluation.nan > 0) {
          high = middle;
          culprit = evaluation;
        } else {
          low = middle + 1;
        }
      }
      const node = full.nodes[low - 1];
      console.log("");
      console.log(
        `[probe4] 最小 K = ${low} — node[${low - 1}] ${node.op} ` +
          `ins=[${node.ins.join(", ")}] outs=[${node.outs.join(", ")}] ` +
          `states=${JSON.stringify(node.states ?? {})} attrs=${JSON.stringify(node.attrs)}`,
      );
      console.log(
        `[probe4] その K の末端 '${culprit.terminal}' で NaN=${culprit.nan}/${culprit.count}`,
      );
      // MUST: 二分の道中は接頭辞グラフばかりを撃つので、最後に全ノードをもう一度撃って
      // 「探索の途中で device が壊れて全部 NaN になっただけ」ではないことを分離する。
      const recheck = await evaluate(probe, total);
      report(recheck);
      console.log(
        recheck.nan > 0
          ? "[probe4] 再確認: K=全ノードで NaN 再現（探索結果はそのまま読める）"
          : "[probe4] 再確認: K=全ノードで NaN が出ない — 探索中の 2 回で挙動が割れた。" +
            "device 状態依存（session churn / 常駐量）の疑いが立つので、最小 K の解釈は保留。",
      );
    }
  } else {
    console.log(
      "使い方: --at K | --coarse N | --bisect（共通: --token / --capacity / --chunk-length）",
    );
  }
} finally {
  gpu.device.destroy();
}
