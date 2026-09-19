/** 最終行の最大値をGPU内で選び、会話の出力転送を8Bにする内部実行面。 */
import {
  type BatchScope,
  createSession,
  estimateSessionMemory,
  type GenerationContext,
  type GpuContext,
  type KarumeModel,
  openModel,
  type ResidentTensor,
  type RunInputs,
  type Session,
} from "@karume/runtime";
import { createOperationChain } from "../concurrency/serial.ts";
import type { GenerationGreedyRun, GenerationSession } from "../generation/sequence.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { gemma4PleGatherIds, type Gemma4PleResident } from "./ple-gpu.ts";

/** 保存済みtargetグラフを変えずに、別の小IRで最大値と最小添字を得る。 */
const selectionModel = (vocabSize: number): KarumeModel => {
  const graph = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["topk"] },
    symbols: [],
    initializers: {},
    inputs: [{ name: "logits", dtype: "f32", shape: [1, 1, vocabSize] }],
    outputs: ["value", "index"],
    values: {
      value: { dtype: "f32", shape: [1, 1, 1] },
      index: { dtype: "i32", shape: [1, 1, 1] },
    },
    nodes: [{ op: "topk", ins: ["logits"], outs: ["value", "index"], attrs: { k: 1 } }],
  };
  const header = new TextEncoder().encode(
    JSON.stringify({ __metadata__: { karume_ir: JSON.stringify(graph) } }),
  );
  const padded = Math.ceil(header.byteLength / 8) * 8;
  const buffer = new ArrayBuffer(8 + padded), bytes = new Uint8Array(buffer);
  new DataView(buffer).setBigUint64(0, BigInt(padded), true);
  bytes.set(header, 8);
  bytes.fill(0x20, 8 + header.byteLength);
  return openModel(buffer);
};

type Resources = {
  readonly selector: Session;
  readonly logits: ResidentTensor;
  readonly value: ResidentTensor;
  readonly index: ResidentTensor;
};

const disposeResources = (resources: Resources): Promise<void> =>
  disposeSteps([
    () => resources.selector.dispose(),
    () => resources.logits.dispose(),
    () => resources.value.dispose(),
    () => resources.index.dispose(),
  ]);

const prepareResources = async (
  gpu: GpuContext,
  model: KarumeModel,
  vocabSize: number,
): Promise<Resources> => {
  let selector: Session | undefined;
  const residents: ResidentTensor[] = [];
  try {
    // 選択グラフは形が1個なのでbackingの複数保持予算を設けない。
    selector = await createSession(gpu, model, { planBackingBudgetBytes: 0 });
    const logits = await gpu.createResident(vocabSize * 4);
    residents.push(logits);
    const value = await gpu.createResident(4);
    residents.push(value);
    const index = await gpu.createResident(4);
    residents.push(index);
    return { selector, logits, value, index };
  } catch (error) {
    try {
      await disposeSteps([() => selector?.dispose(), ...residents.map((r) => () => r.dispose())]);
    } catch (cleanup) {
      throw new AggregateError([error, cleanup], "greedy出力の確保と後始末が失敗した");
    }
    throw error;
  }
};

export type GemmaGreedyOutput = {
  readonly session: GenerationSession;
  readonly greedy: GenerationGreedyRun<GenerationContext>;
  readonly extraBytes: number;
  dispose(): Promise<void>;
};

/**
 * PLEをGPU常駐にした席では、target実行の直前にGPU内gatherを積んで常駐入力を差す（ADR 0085）。
 * batchが開いている間に積めばフェンスは増えず、通常runだけがgather用のbatchを1本余分に払う。
 */
const withPleInput = async (
  ple: Gemma4PleResident | undefined,
  batch: BatchScope,
  inputs: RunInputs,
): Promise<RunInputs> => {
  if (ple === undefined) return inputs;
  const ids = gemma4PleGatherIds("greedy出力", inputs, ple.idsName);
  return { ...inputs, [ple.inputName]: await ple.enqueue(batch, ids) };
};

/**
 * targetの通常runとgreedy batchを同じ鎖へ載せる。別会話の通常runがバッチに先行すると、
 * enqueue→run→scope lock→batchの循環になるため、batchを開く前から順序を保つ。
 * context構築はSession鎖の外にあり、ここでもそのまま通す。GPU資源は最初のgreedy時だけ確保する。
 * DECIDED: docs/decisions/0083-generation-api-surface.md#gemmaの温度0生成の小出力2026-09-12
 *
 * `ple`を渡すとper_layer_inputsをGPU内gatherで作る（ADR 0085の GPU 常駐席）。所有権は呼び手側で、
 * ここはenqueueを積むだけである（disposeはpipelineがSessionの後に行う）。
 */
export const createGemmaGreedyOutput = (
  gpu: GpuContext,
  target: Session,
  logitsName: string,
  vocabSize: number,
  ple?: Gemma4PleResident,
): GemmaGreedyOutput => {
  const model = selectionModel(vocabSize), chain = createOperationChain();
  // selectorの入力は常駐であり、ホスト入力ぶんも含むこの見積りは保守的な上界。
  const extraBytes = vocabSize * 4 + 8 +
    estimateSessionMemory(model, { planBackingBudgetBytes: 0 }).peakAccountedBytes;
  let ready: Promise<Resources> | undefined;
  let disposal: Promise<void> | undefined;
  const assertAlive = (): void => {
    if (disposal !== undefined) throw new Error("greedy出力はdispose済み");
  };
  return {
    extraBytes,
    session: {
      createGenerationContext: (spec) => target.createGenerationContext(spec),
      run: (inputs, bindings, generation) => {
        assertAlive();
        return chain(async () => {
          if (ple === undefined) return await target.run(inputs, bindings, generation);
          // 通常runはコマンド列を自分で閉じるので、gatherは先行する1本のbatchで済ませる。
          // フェンスが1本増えるのはprefillと診断付きdecodeだけで、temperature0のdecodeは
          // greedy側の同一batchへ積む（ADR 0085 の GPU 常駐席）。
          const batch = await gpu.beginBatch();
          let extra: RunInputs;
          try {
            extra = await withPleInput(ple, batch, inputs);
          } finally {
            await batch.finish();
          }
          return await target.run(extra, bindings, generation);
        });
      },
    },
    greedy: (inputs, generation) => {
      assertAlive();
      return chain(async () => {
        ready ??= prepareResources(gpu, model, vocabSize).catch((cause) => {
          ready = undefined;
          throw cause;
        });
        const r = await ready;
        const batch = await gpu.beginBatch();
        try {
          await target.enqueue(await withPleInput(ple, batch, inputs), {
            batch,
            generation,
            copyOutputs: { [logitsName]: r.logits },
          });
          await r.selector.enqueue({ logits: r.logits }, {
            batch,
            copyOutputs: { value: r.value, index: r.index },
          });
          const out = await batch.finishAndRead({ value: r.value, index: r.index });
          return { value: new Float32Array(out.value)[0], index: new Int32Array(out.index)[0] };
        } finally {
          await batch.finish();
        }
      });
    },
    dispose: () => {
      disposal ??= chain(async () => {
        if (ready !== undefined) await disposeResources(await ready);
      });
      return disposal;
    },
  };
};
