/**
 * Internal execution surface that picks the last row's maximum on the GPU and cuts a conversation's
 * output transfer to 8 B.
 */
import {
  type BatchScope,
  type GenerationContext,
  type GpuContext,
  openMemoryContainer,
  parseIrDeclarationValue,
  prepareContainer,
  type PreparedModel,
  type ResidentTensor,
  type RunInputs,
  type Session,
} from "@karume/runtime";
import { createOperationChain } from "../concurrency/serial.ts";
import type { GenerationGreedyRun, GenerationSession } from "../generation/sequence.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { gemma4PleGatherIds, type Gemma4PleResident } from "./ple-gpu.ts";

/**
 * The name of the selection graph (it holds no weight, so only its declaration goes into an in-memory
 * container).
 */
const SELECT_GRAPH = "select";

/**
 * Gets the maximum and its smallest index with a separate small IR, without changing the stored
 * target graph.
 *
 * MUST: the assembled declaration goes through `parseIrDeclarationValue` — nothing here passes
 * through the exporter or the container reader, so without it none of the rules a graph decides on
 * its own (SSA, topological order, `requires.ops` matching the ops actually used, unknown keys) is
 * checked.
 *
 * MUST NOT: add the optional `states` to the literal below. Leaving it out keeps the literal from
 * typing as an `IrDeclaration`, so only the return value of `parseIrDeclarationValue` can become
 * `graph` (an edit that removes the wrapper fails `vp check`, the typecheck).
 */
const selectionModel = (vocabSize: number): PreparedModel => {
  const graph = parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
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
  });
  return prepareContainer(
    openMemoryContainer({ graphs: { [SELECT_GRAPH]: graph }, tensors: {} }),
    SELECT_GRAPH,
  );
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
  model: PreparedModel,
  vocabSize: number,
): Promise<Resources> => {
  let selector: Session | undefined;
  const residents: ResidentTensor[] = [];
  try {
    // The selection graph has a single shape, so no budget for keeping several backings is set.
    selector = await model.createContainerSession(gpu, { planBackingBudgetBytes: 0 });
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
      throw new AggregateError(
        [error, cleanup],
        "Allocating the greedy output and the clean-up both failed",
      );
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
 * With the GPU-resident PLE option, enqueue an on-GPU gather right before the target run and plug in
 * the resident input (ADR 0085). Enqueued while the batch is open it adds no fence;
 * only the ordinary prefill run pays one extra batch for the gather.
 */
const withPleInput = async (
  ple: Gemma4PleResident | undefined,
  batch: BatchScope,
  inputs: RunInputs,
): Promise<RunInputs> => {
  if (ple === undefined) return inputs;
  const ids = gemma4PleGatherIds("greedy output", inputs, ple.idsName);
  return { ...inputs, [ple.inputName]: await ple.enqueue(batch, ids) };
};

/**
 * Puts the target's ordinary runs and the greedy batches on the same chain. If another
 * conversation's ordinary run got ahead of a batch, the cycle enqueue → run → scope lock → batch
 * would form, so the order is kept from before the batch opens.
 * Context construction is outside the Session chain and passes straight through here as well. The
 * GPU resources are allocated only at the first greedy call.
 * DECIDED: docs/decisions/0083-generation-api-surface.md#addendum-2026-09-12--small-output-for-gemmas-temperature-0-decode
 *
 * Passing `ple` builds per_layer_inputs with an on-GPU gather (the GPU-resident option of ADR 0085).
 * Ownership stays with the caller; this only enqueues (the pipeline disposes it after the Session).
 */
export const createGemmaGreedyOutput = (
  gpu: GpuContext,
  target: Session,
  logitsName: string,
  vocabSize: number,
  ple?: Gemma4PleResident,
): GemmaGreedyOutput => {
  const model = selectionModel(vocabSize),
    chain = createOperationChain();
  // The selector's input is resident, so this estimate, which includes the host-input share, is a
  // conservative upper bound.
  const extraBytes =
    vocabSize * 4 + 8 + model.estimate({ planBackingBudgetBytes: 0 }).peakAccountedBytes;
  let ready: Promise<Resources> | undefined;
  let disposal: Promise<void> | undefined;
  const assertAlive = (): void => {
    if (disposal !== undefined) throw new Error("The greedy output is disposed");
  };
  return {
    extraBytes,
    session: {
      createGenerationContext: (spec) => target.createGenerationContext(spec),
      run: (inputs, bindings, generation) => {
        assertAlive();
        return chain(async () => {
          if (ple === undefined) return await target.run(inputs, bindings, generation);
          const rows = gemma4PleGatherIds("greedy output", inputs, ple.idsName).length;
          const batch = await gpu.beginBatch();
          try {
            const extra = await withPleInput(ple, batch, inputs);
            if (rows > 1) {
              // Prefill (M > 1) closes the gather batch first and then hands over to an ordinary run
              // (one more fence per chunk). The enqueue family is contracted to build the slot
              // backing from the first call, so the chunk-shaped backing is not paid from the first
              // chunk on (the same reason the generation surface's prefill stays on run —
              // generation/sequence.ts).
              await batch.finish();
              return await target.run(extra, bindings, generation);
            }
            // Decode (M = 1) enqueues the gather and the target in the same batch and reads the graph
            // outputs back at the batch's closing fence (one fence — the same shape as greedy;
            // ADR 0054 addendum).
            const read = target.enqueueRead(extra, {
              batch,
              generation,
              ...(bindings === undefined ? {} : { bindings }),
            });
            await read.admitted;
            await batch.finish();
            return await read.outputs;
          } finally {
            // finish returns the same outcome however often it is called (on the success path it is
            // already done above).
            await batch.finish();
          }
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
          await r.selector.enqueue(
            { logits: r.logits },
            {
              batch,
              copyOutputs: { value: r.value, index: r.index },
            },
          );
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
