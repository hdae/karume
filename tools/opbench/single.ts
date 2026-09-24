/**
 * opbench `single` — census 加重表の 1 行（= 1 形状）を 1 ノードのグラフに組み、実 GPU で
 * 計測規約（./bench.ts）どおりに測る。2 段目（GPU）の入口で、K-11 / P-1 のような「単体で測った
 * 利得 × census 加重」を機械的に出すための道具。
 *
 * 入力は `opbench census` の出力ディレクトリ（summary.json の weights[] と session）。1 行が
 * 1 ケースで、実行変種は summary.json の `session`（= 配布 quant の宣言）を runtime の
 * SessionOptions へ写して使う — census 行だけでは「本番と同じカーネル」が立たない（linear a8 /
 * attention a8 の分岐は宣言側にある）。
 *
 * 対象は「融合されず・別名化でもなく・活性入力が f32 の行」。それ以外は理由つきで除外して
 * 件数を summary に残す（黙って落とさない）。組めない / 走らない形（state を触る attention など）
 * は fail loudly の文言をそのまま除外理由に載せて次の行へ進む — 1 行の失敗で掃引全体を止めない。
 */

import type {
  BoundContainer,
  GpuContext,
  MemoryTensor,
  RunInputs,
  Session,
  SessionOptions,
  Tensor,
} from "../../packages/runtime/mod.ts";
import { createSessionFromContainer, parseIrDeclarationValue } from "../../packages/runtime/mod.ts";
import { perChannelGroupSize } from "../../packages/runtime/src/format/container/codecs.ts";
import { WEIGHT_CHANNEL_AXES, WEIGHT_SLOTS } from "../../packages/runtime/src/ops/names.ts";
import { f32ToF16Bits } from "../../packages/runtime/tests/helpers/f16.ts";
import { memoryContainer } from "../_shared/ir-memory.ts";
import { toSessionOptions } from "../../packages/models/src/session/options.ts";
import type { SessionSpec } from "../../packages/hub/src/manifest.ts";
import type { SessionDeclaration } from "../_shared/assets.ts";
import type { CensusSummary, WeightRow } from "./census.ts";
import {
  calibrateReps,
  type Heater,
  MAX_REPS,
  measureTiming,
  measureWall,
  pinClocks,
  ROUNDS,
  sampleTiming,
  TARGET_PASS_MS,
} from "./bench.ts";

/** 1 ケース = census 加重表の 1 行（シナリオ名つき）。 */
export type SingleCase = {
  readonly scenario: string;
  readonly row: WeightRow;
};

export type CaseFilter = {
  readonly ops?: ReadonlySet<string>;
  readonly scenario?: string;
  readonly component?: string;
  /** 加重（count × 出力要素数）の降順で先頭 n 件だけ。 */
  readonly limit?: number;
};

export type Selection = {
  readonly cases: readonly SingleCase[];
  /** 除外理由 → 行数（黙って落とさないための表）。 */
  readonly excluded: Readonly<Record<string, number>>;
};

/** 合成できる格納 codec（台帳の登録名のうち、テスト helper が符号化を持つもの）。 */
const SYNTHESIZABLE_STORAGE: ReadonlySet<string> = new Set([
  "f32",
  "f16",
  "int8-sym",
  "int4-sym-g",
]);

/** 加重行から測れる行を選ぶ。除外は理由ごとに数える。 */
export const selectCases = (summary: CensusSummary, filter: CaseFilter = {}): Selection => {
  const excluded = new Map<string, number>();
  const exclude = (reason: string): void => {
    excluded.set(reason, (excluded.get(reason) ?? 0) + 1);
  };
  const cases: SingleCase[] = [];
  for (const scenario of summary.scenarios) {
    if (filter.scenario !== undefined && scenario.scenario !== filter.scenario) continue;
    for (const row of scenario.weights) {
      if (filter.ops !== undefined && !filter.ops.has(row.op)) continue;
      if (filter.component !== undefined && row.component !== filter.component) continue;
      if (row.fused_by !== null) {
        exclude(`fused (${row.fused_by})`);
        continue;
      }
      if (row.aliases_input) {
        exclude("aliases_input (0 dispatch)");
        continue;
      }
      // state スロットを触る op は GenerationContext（容量・chunkLength）が要り、1 ノードの graph では
      // 組めない（attention の `window` は states 欄を持つノードでだけ宣言できる契約）。
      if (row.op === "state_append" || (row.op === "attention" && "window" in row.attrs)) {
        exclude("state slot op (needs GenerationContext)");
        continue;
      }
      const activationDtype = row.in_dtypes.find((dtype, slot) =>
        row.storage[slot] === null && dtype !== "f32"
      );
      if (activationDtype !== undefined) {
        exclude(`activation dtype ${activationDtype} (f32 only)`);
        continue;
      }
      const storageDtype = row.storage.find((ref) =>
        ref !== null && !SYNTHESIZABLE_STORAGE.has(ref.dtype)
      );
      if (storageDtype !== null && storageDtype !== undefined) {
        exclude(`storage dtype ${storageDtype.dtype} (no encoder)`);
        continue;
      }
      cases.push({ scenario: scenario.scenario, row });
    }
  }
  cases.sort((a, b) =>
    b.row.count * b.row.out_elements - a.row.count * a.row.out_elements ||
    (a.row.op < b.row.op ? -1 : a.row.op > b.row.op ? 1 : 0)
  );
  const limited = filter.limit === undefined ? cases : cases.slice(0, filter.limit);
  return { cases: limited, excluded: Object.fromEntries(excluded) };
};

/**
 * 決定的な値（乱数禁止 — 失敗が再現しないため）。活性は [−0.5, 0.5) の一様風。値の分布は速度に
 * 効かない（分岐の無いカーネルだけを測る前提）ので、周期 TILE の表を繰り返して大きな
 * テンソルも JS 配列を経ずに埋める（lm_head 級の 384MiB を要素ごとに量子化すると配列長で落ちる）。
 */
const TILE = 4096;
const valueAt = (index: number): number => ((index * 7919) % 1000) / 1000 - 0.5;

const tileF32: Float32Array = Float32Array.from({ length: TILE }, (_, i) => valueAt(i));
const tileF16: Uint16Array = Uint16Array.from({ length: TILE }, (_, i) => f32ToF16Bits(valueAt(i)));
/** i8 = [−127, 127] の周期列（−128 は使わない — 符号化器と同じ規律）。 */
const tileI8: Int8Array = Int8Array.from({ length: TILE }, (_, i) => ((i * 7919) % 255) - 127);
/** i4 = 2 要素 / バイト（下位 nibble が要素 2i・格納値 u = q + 8・q ∈ [−7, 7]）。 */
const tileI4: Uint8Array = Uint8Array.from({ length: TILE }, (_, i) => {
  const low = ((i * 2 * 7919) % 15) - 7 + 8;
  const high = (((i * 2 + 1) * 7919) % 15) - 7 + 8;
  return (high << 4) | low;
});

/** タイルを繰り返して bytes を埋める（周期の切れ目は要素境界に揃う）。 */
const repeatTile = (
  tile: ArrayBufferView,
  elementBytes: number,
  count: number,
): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(count * elementBytes);
  const src = new Uint8Array(tile.buffer, tile.byteOffset, tile.byteLength);
  for (let offset = 0; offset < out.length; offset += src.length) {
    out.set(src.subarray(0, Math.min(src.length, out.length - offset)), offset);
  }
  return out;
};

const fillValues = (shape: readonly number[]): Float32Array<ArrayBuffer> => {
  const count = shape.reduce((total, dim) => total * dim, 1);
  const bytes = repeatTile(tileF32, 4, count);
  return new Float32Array(bytes.buffer, 0, count);
};

/** scale は一定（値は速度に効かない・0 と非有限を避ける）。 */
const constantScale = (count: number): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from({ length: count }, () => 0.01).buffer);

const elementCount = (shape: readonly number[]): number =>
  shape.reduce((total, dim) => total * dim, 1);

/**
 * 量子化 initializer のチャネル軸（`Encoding.rowAxis`）。台帳を引くのは、v1 の配布形では
 * ローダが消費側 op から導いていた軸が、v2 では**供給側が宣言する欄**になったため — 合成側が
 * 独自表を持つと、本番の容器と違う軸で scale を並べたケースを黙って測ることになる。
 *
 * 軸が付くのは {@link WEIGHT_SLOTS} が指す**重みスロット**だけ（それ以外は 0）— v1 ローダも
 * 重みスロットの initializer にしか軸を付けていなかったので、bias 等が i8 格納の行を軸 1 で
 * 組んで落とさないため。
 */
const channelAxis = (op: string, slot: number): 0 | 1 =>
  slot === WEIGHT_SLOTS.get(op) && WEIGHT_CHANNEL_AXES.get(op) === 1 ? 1 : 0;

/** 反復ぶんの出力 readback がこの量を超えないよう反復数を抑える（メモリと readback 時間の歯止め）。 */
export const MAX_OUTPUT_BYTES = 256 * 1024 * 1024;

/** 合成グラフの名前（容器 1 本 = グラフ 1 本）。 */
const GRAPH_NAME = "case";

export type CaseModel = {
  /** 合成した重みを載せたメモリ内容器（グラフ名は {@link GRAPH_NAME}）。 */
  readonly container: BoundContainer;
  readonly inputs: RunInputs;
  /** 実際に組んだ反復数（出力バイト上限で減ることがある）。 */
  readonly reps: number;
};

/**
 * 1 行から「同じ op を `reps` 本並べた」グラフを組む。全ノードが同じ入力を読み、出力だけ別名。
 * 1 run に reps 本の dispatch が載るので、timing モードは 1 pass ≈ 目標長まで積める。
 *
 * 初期化子は格納 codec ごとにこの場で符号化し、容器を書かずに**メモリ内容器**へ載せる
 * （本番の pack はエクスポータの担当 — ここは仕様を書き下したもの）。量子化 codec の
 * チャネル軸は消費側 op の重み軸台帳（`WEIGHT_CHANNEL_AXES`）から引く — `conv_transpose1d`
 * だけ `[Cin, Cout, K]` で軸 1 がチャネルで、他は先頭次元。軸が付くのは重みスロット
 * （`WEIGHT_SLOTS`）だけで、bias 等の他スロットは軸 0。
 */
export const buildCaseModel = (row: WeightRow, requestedReps: number): CaseModel => {
  const outBytes = row.out_shapes.reduce((total, shape) => total + elementCount(shape) * 4, 0);
  const reps = Math.max(
    1,
    Math.min(
      requestedReps,
      outBytes === 0 ? requestedReps : Math.floor(MAX_OUTPUT_BYTES / outBytes),
    ),
  );
  // 宣言の節は積みながら組み、最後に 1 度だけ読み手（`parseIrDeclarationValue`）へ通す。
  const declaredInputs: { name: string; dtype: string; shape: (number | string)[] }[] = [];
  const declaredOutputs: string[] = [];
  const declaredInitializers: Record<string, Record<string, never>> = {};
  const declaredValues: Record<string, { dtype: string; shape: (number | string)[] }> = {};
  const declaredNodes: {
    op: string;
    ins: string[];
    outs: string[];
    attrs: Record<string, unknown>;
  }[] = [];
  const tensors: Record<string, MemoryTensor> = {};
  const inputs: Record<string, Tensor> = {};
  const ins: string[] = [];
  row.storage.forEach((ref, slot) => {
    const shape = [...row.in_shapes[slot]];
    const dtype = row.in_dtypes[slot];
    if (ref === null) {
      const name = `x${slot}`;
      declaredInputs.push({ name, dtype, shape });
      inputs[name] = { dtype: "f32", shape, data: fillValues(shape) };
      ins.push(name);
      return;
    }
    // v2 では initializer 名がそのまま実体の鍵（別名のテンソル名は無い）。
    const name = `w${slot}`;
    const count = elementCount(shape);
    declaredValues[name] = { dtype, shape };
    declaredInitializers[name] = {};
    ins.push(name);
    switch (ref.dtype) {
      case "f32":
        tensors[name] = { bytes: repeatTile(tileF32, 4, count), encoding: { codec: "f32" } };
        return;
      case "f16":
        tensors[name] = { bytes: repeatTile(tileF16, 2, count), encoding: { codec: "f16" } };
        return;
      case "int8-sym": {
        // per-channel scale は rank 2 の `[チャネル軸の長さ, 1]`（group 長 = 行長 = 1 行 1 scale）。
        const rowAxis = channelAxis(row.op, slot);
        const rows = shape[rowAxis];
        tensors[name] = {
          bytes: repeatTile(tileI8, 1, count),
          encoding: {
            codec: "int8-sym",
            rowAxis,
            groupSize: perChannelGroupSize(rows === 0 ? 0 : count / rows),
            scale: constantScale(rows),
          },
        };
        return;
      }
      case "int4-sym-g": {
        if (ref.group_size === undefined) {
          throw new Error(`slot ${slot}: int4-sym-g に group_size が無い`);
        }
        // group scale は rank 非依存の rank 2 = [先頭次元, 行長 / group]（ADR 0069 決定 2）。
        const rows = shape[0];
        const width = count / rows;
        if (width % ref.group_size !== 0) {
          throw new Error(
            `slot ${slot}: 行長 ${width} が group_size ${ref.group_size} で割り切れない`,
          );
        }
        const groups = width / ref.group_size;
        tensors[name] = {
          bytes: repeatTile(tileI4, 1, count / 2),
          encoding: {
            codec: "int4-sym-g",
            groupSize: ref.group_size,
            scale: constantScale(rows * groups),
          },
        };
        return;
      }
      default:
        throw new Error(`slot ${slot}: 格納 dtype '${ref.dtype}' の符号化器が無い`);
    }
  });
  for (let rep = 0; rep < reps; rep += 1) {
    const outs = row.out_shapes.map((_, slot) => `y${rep}_${slot}`);
    outs.forEach((name, slot) => {
      declaredValues[name] = { dtype: row.out_dtypes[slot], shape: [...row.out_shapes[slot]] };
      declaredOutputs.push(name);
    });
    declaredNodes.push({ op: row.op, ins: [...ins], outs, attrs: { ...row.attrs } });
  }
  const declaration = parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: [row.op] },
    symbols: [],
    inputs: declaredInputs,
    outputs: declaredOutputs,
    initializers: declaredInitializers,
    values: declaredValues,
    nodes: declaredNodes,
  });
  return { container: memoryContainer(GRAPH_NAME, declaration, tensors), inputs, reps };
};

const LINEAR_COMPUTE = new Set(["f32", "a8", "f16"]);
const ATTENTION_COMPUTE = new Set(["f32", "f16", "a8"]);
const SCORE_STORAGE = new Set(["f32", "f16"]);

/**
 * summary.json の `session`（manifest の逐語・ツール側では未検証）を runtime の SessionOptions へ
 * 写す。写像は packages/models と同じ関数（明示写像 — 綴りが割れれば型検査で落ちる）。census が
 * 検証していないぶん、ここでキーと値を allowlist で見て fail loudly にする。`overrides` は CLI の
 * `--session <knob>=<value>` で、宣言に上書きする（対照実行用）。
 */
export const sessionOptionsOf = (
  declaration: SessionDeclaration | null,
  overrides: Readonly<Record<string, string>> = {},
): SessionOptions => {
  const merged: Record<string, string> = { ...(declaration ?? {}), ...overrides };
  const spec: { -readonly [K in keyof SessionSpec]?: SessionSpec[K] } = {};
  for (const [key, value] of Object.entries(merged)) {
    switch (key) {
      case "linearCompute":
        if (!LINEAR_COMPUTE.has(value)) throw new Error(`session.linearCompute '${value}' は不正`);
        spec.linearCompute = value as SessionSpec["linearCompute"];
        break;
      case "attentionCompute":
        if (!ATTENTION_COMPUTE.has(value)) {
          throw new Error(`session.attentionCompute '${value}' は不正`);
        }
        spec.attentionCompute = value as SessionSpec["attentionCompute"];
        break;
      case "attentionScoreStorage":
        if (!SCORE_STORAGE.has(value)) {
          throw new Error(`session.attentionScoreStorage '${value}' は不正`);
        }
        spec.attentionScoreStorage = value as SessionSpec["attentionScoreStorage"];
        break;
      default:
        throw new Error(
          `session の未知のノブ '${key}'（既知: linearCompute / attentionCompute / attentionScoreStorage）`,
        );
    }
  }
  return toSessionOptions(spec);
};

/** 1 ケースの計測結果（single.jsonl の 1 行）。timing と wall は別欄 — 割った倍率は出さない。 */
export type SingleRecord = {
  readonly scenario: string;
  readonly component: string;
  readonly op: string;
  readonly in_shapes: readonly (readonly number[])[];
  readonly out_shapes: readonly (readonly number[])[];
  readonly attrs: Readonly<Record<string, unknown>>;
  readonly storage: WeightRow["storage"];
  readonly storage_signature: string;
  /** census 加重（この形が実行 1 回に出る本数）。 */
  readonly count: number;
  readonly mode: "timing" | "wall";
  readonly reps: number;
  readonly rounds: number;
  /** timing: min(ノード 1 本ぶんの ns) — 複数 dispatch を出す op はその合計。wall では null。 */
  readonly ns_per_node_min: number | null;
  /** timing: 1 ノードが出した dispatch 数（a8 linear = 2）。wall では null。 */
  readonly dispatches_per_node: number | null;
  /** timing: ns_per_node_min × count（ms）。wall では null。 */
  readonly weighted_ms: number | null;
  /** wall: min(区間の壁時計 / 反復)（ms）。timing では null。 */
  readonly wall_ms_per_rep_min: number | null;
  /** その run で立ったパイプラインキー（timing のみ・ns 降順。wall では空）。 */
  readonly keys: readonly string[];
  readonly clamped_negative_samples: number;
};

export type RunSingleOptions = {
  readonly gpu: GpuContext;
  readonly session: SessionOptions;
  readonly mode: "timing" | "wall";
  readonly rounds?: number;
  /** クロックを張り付かせる filler（{@link createHeater}）。無ければ張り付けをしない（テスト用）。 */
  readonly heater?: Heater;
};

/**
 * filler の形 = f32 linear [2048,4096] × [4096,4096]（≈69 GFLOP・重み 64MiB）を 8 本。P0 で
 * 20ms 級 — 計測パス（≈80ms）の合間に挟んで P8 / P5 へ落ちないようにする。
 */
const HEATER_ROW: WeightRow = {
  component: "heater",
  op: "linear",
  in_shapes: [[2048, 4096], [4096, 4096], [4096]],
  out_shapes: [[2048, 4096]],
  in_dtypes: ["f32", "f32", "f32"],
  out_dtypes: ["f32"],
  attrs: {},
  storage: [null, { dtype: "f32" }, { dtype: "f32" }],
  storage_signature: "f32",
  fused_by: null,
  aliases_input: false,
  count: 1,
  out_elements: 2048 * 4096,
};

/** filler の Session を組む（プロセスで 1 つ・計測の session options とは独立に既定で走る）。 */
export const createHeater = async (
  gpu: GpuContext,
): Promise<Heater & { dispose(): Promise<void> }> => {
  const model = buildCaseModel(HEATER_ROW, 8);
  const session = await createSessionFromContainer(gpu, model.container, GRAPH_NAME, {});
  return {
    run: async () => {
      const started = performance.now();
      await session.run(model.inputs);
      return performance.now() - started;
    },
    dispose: () => session.dispose(),
  };
};

/** 組めない / 走らない行の記録（除外理由として summary に載せる）。 */
export type CaseFailure = {
  readonly scenario: string;
  readonly component: string;
  readonly op: string;
  readonly in_shapes: readonly (readonly number[])[];
  readonly message: string;
};

const withSession = async <T>(
  gpu: GpuContext,
  options: SessionOptions,
  model: CaseModel,
  body: (session: Session) => Promise<T>,
): Promise<T> => {
  const session = await createSessionFromContainer(gpu, model.container, GRAPH_NAME, options);
  try {
    return await body(session);
  } finally {
    await session.dispose();
  }
};

/**
 * 1 ケースを測る。timing は「反復 1 で 1 dispatch の ns を見積り → 目標長ぶん積んだグラフで
 * rounds 回」、wall は「壁時計が目標長に届くまで反復を倍々にして rounds 回」。
 */
export const measureCase = async (
  testCase: SingleCase,
  options: RunSingleOptions,
): Promise<SingleRecord> => {
  const { row } = testCase;
  const rounds = options.rounds ?? ROUNDS;
  const base = {
    scenario: testCase.scenario,
    component: row.component,
    op: row.op,
    in_shapes: row.in_shapes,
    out_shapes: row.out_shapes,
    attrs: row.attrs,
    storage: row.storage,
    storage_signature: row.storage_signature,
    count: row.count,
  };
  if (options.mode === "timing") {
    // 反復数の見積りも張り付いたクロックで採る（P8 で見積ると反復が 20 倍少なくなる）。
    const probe = buildCaseModel(row, 1);
    const estimate = await withSession(options.gpu, options.session, probe, async (session) => {
      await sampleTiming(session, probe.inputs);
      if (options.heater !== undefined) await pinClocks(options.heater);
      const sample = await sampleTiming(session, probe.inputs);
      return sample.totalNs; // 反復 1 = ノード 1 本ぶん
    });
    const model = buildCaseModel(row, calibrateReps(estimate));
    const result = await withSession(
      options.gpu,
      options.session,
      model,
      (session) => measureTiming(session, model.inputs, model.reps, rounds, options.heater),
    );
    return {
      ...base,
      mode: "timing",
      reps: result.reps,
      rounds: result.rounds,
      ns_per_node_min: result.nsPerNodeMin,
      dispatches_per_node: result.dispatchesPerNode,
      weighted_ms: (result.nsPerNodeMin * row.count) / 1e6,
      wall_ms_per_rep_min: null,
      keys: result.keys,
      clamped_negative_samples: result.clampedNegativeSamples,
    };
  }
  let reps = 1;
  for (;;) {
    const model = buildCaseModel(row, reps);
    const result = await withSession(
      options.gpu,
      options.session,
      model,
      (session) =>
        measureWall(options.gpu, session, model.inputs, model.reps, rounds, options.heater),
    );
    const passMs = result.msPerRepMin * model.reps;
    if (passMs >= TARGET_PASS_MS || model.reps >= MAX_REPS || model.reps < reps) {
      return {
        ...base,
        mode: "wall",
        reps: result.reps,
        rounds: result.rounds,
        ns_per_node_min: null,
        dispatches_per_node: null,
        weighted_ms: null,
        wall_ms_per_rep_min: result.msPerRepMin,
        keys: [],
        clamped_negative_samples: 0,
      };
    }
    reps *= 2;
  }
};

/** rig 欄が読む面だけ（`GPUAdapterInfo` はこれを満たす）。 */
export type AdapterInfoFields = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

/** summary.json が「どの機で測ったか」を残す欄。 */
export type Rig = AdapterInfoFields & { readonly deno: string };

/**
 * GPU の名乗りを rig 欄へ写す。
 *
 * MUST: 4 欄は 1 つずつ明示的に読む — `GPUAdapterInfo` の欄は prototype 上の getter なので
 * spread（`{ ...gpu.adapterInfo }`）では 1 つも写らず、rig が空のまま残る。single と graph の
 * 2 つの summary が同じ表を持つので、読みはこの 1 本に寄せる（片方だけ spread へ戻る形を作らない）。
 */
export const rigOf = (gpu: { readonly adapterInfo: AdapterInfoFields }): Rig => ({
  vendor: gpu.adapterInfo.vendor,
  architecture: gpu.adapterInfo.architecture,
  device: gpu.adapterInfo.device,
  description: gpu.adapterInfo.description,
  deno: Deno.version.deno,
});

export type SingleSummary = {
  readonly generated_at: string;
  readonly census: string;
  readonly family: string;
  readonly model: string;
  readonly quant: string;
  readonly session: SessionOptions;
  readonly mode: "timing" | "wall";
  readonly rig: Rig & {
    readonly target_pass_ms: number;
    readonly rounds: number;
  };
  readonly measured: number;
  readonly excluded: Readonly<Record<string, number>>;
  readonly failed: readonly CaseFailure[];
  /** (op, storage_signature) ごとの weighted_ms 合計（timing のみ — K-11 の照合はこの表で）。 */
  readonly weighted_ms_by_op_storage: Readonly<Record<string, number>>;
};

export const buildSingleSummary = (
  census: string,
  summary: CensusSummary,
  gpu: { readonly adapterInfo: AdapterInfoFields },
  options: Pick<RunSingleOptions, "session" | "mode" | "rounds">,
  records: readonly SingleRecord[],
  excluded: Readonly<Record<string, number>>,
  failed: readonly CaseFailure[],
): SingleSummary => {
  const byOpStorage = new Map<string, number>();
  for (const record of records) {
    if (record.weighted_ms === null) continue;
    const key = `${record.op}/${record.storage_signature}`;
    byOpStorage.set(key, (byOpStorage.get(key) ?? 0) + record.weighted_ms);
  }
  return {
    generated_at: new Date().toISOString(),
    census,
    family: summary.family,
    model: summary.model,
    quant: summary.quant,
    session: options.session,
    mode: options.mode,
    rig: {
      ...rigOf(gpu),
      target_pass_ms: TARGET_PASS_MS,
      rounds: options.rounds ?? ROUNDS,
    },
    measured: records.length,
    excluded,
    failed,
    weighted_ms_by_op_storage: Object.fromEntries(byOpStorage),
  };
};
