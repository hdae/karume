/**
 * The scaffolding for running one stage — the generation-event emitter, the per-stage Session scope,
 * and the tensor conversions of graph inputs and outputs.
 *
 * The description of the 10 stages (what runs in which order) is canonical in the doc at the top of
 * {@link "./pipeline.ts"}; this module holds only "what every stage needs in the same form". It owns
 * neither the order of the stages nor any resource.
 *
 * It lives here because the 3 modules `./pipeline.ts` / `./conditioning.ts` / `./dit-loop.ts` read
 * the same scaffolding. Left in `./pipeline.ts`, the newer modules would import those values back
 * from it and form an import cycle (type-only references are erased, so `import type` is fine).
 *
 * MUST: every module is free of side effects (no execution at import time, no global mutable state
 * — CLAUDE.md).
 */

import type {
  RunInputs,
  Session,
  SessionDiagnostics,
  SessionOptions,
  Tensor,
} from "@karume/runtime";

import type { IrodoriGenerateEvent, IrodoriRunComponent, IrodoriState } from "./pipeline.ts";
import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import { withSession } from "../session/with-session.ts";

/**
 * Generation-event emitter (folded into a single no-op when nobody subscribes, so the firing points
 * carry no branch).
 */
export type EmitEvent = (event: IrodoriGenerateEvent) => Promise<void>;

/** The emitter when nobody subscribes. */
const NO_EVENTS: EmitEvent = () => Promise.resolve();

/**
 * Folds the request's `onEvent` into an emitter (awaited, exceptions not swallowed —
 * `IrodoriGenerateRequest.onEvent` in {@link "./pipeline.ts"}).
 */
export const emitter = (
  onEvent: ((event: IrodoriGenerateEvent) => void | Promise<void>) | undefined,
): EmitEvent =>
  onEvent === undefined
    ? NO_EVENTS
    : async (event: IrodoriGenerateEvent) => {
        await onEvent(event);
      };

export const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

export const f32 = (data: Float32Array<ArrayBuffer>, shape: readonly number[]): Tensor => ({
  dtype: "f32",
  shape: [...shape],
  data,
});

export const i32 = (data: Int32Array<ArrayBuffer>, shape: readonly number[]): Tensor => ({
  dtype: "i32",
  shape: [...shape],
  data,
});

/** The physical representation of bool is u32 0 / 1 (ADR 0009). */
export const bool = (value: boolean): Tensor => ({
  dtype: "bool",
  shape: [1, 1],
  data: Uint32Array.of(value ? 1 : 0),
});

/** Looks up a graph output **name** by position (names are not hard-coded). */
export const outputNameAt = (model: GraphOwner, index: number): string => {
  const name = model.graph.outputs[index];
  if (name === undefined) {
    throw new Error(`グラフ出力 ${index} が無い（${model.graph.outputs.length} 本しかない）`);
  }
  return name;
};

/** Looks up a graph output by **position**. */
export const outputAt = (
  model: GraphOwner,
  outputs: Readonly<Record<string, Tensor>>,
  index: number,
): Tensor => {
  const name = outputNameAt(model, index);
  const tensor = outputs[name];
  if (tensor === undefined) throw new Error(`グラフ出力 ${index}（'${name}'）が実行結果に無い`);
  return tensor;
};

/**
 * Binds the component name into the diagnostics hook
 * (`IrodoriPipelineOptions.onRunDiagnostics` in {@link "./pipeline.ts"}).
 */
export const observer = (
  state: IrodoriState,
  component: IrodoriRunComponent,
): ((diagnostics: SessionDiagnostics) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  return listener === undefined ? undefined : (diagnostics) => listener(component, diagnostics);
};

/**
 * Runs one stage (emits the `stage` event before the Session is built and after it is released).
 * If it fails midway, no `end` is emitted (the whole generation rejects — the doc of `onEvent`).
 */
export const withStageSession = async <T>(
  state: IrodoriState,
  emit: EmitEvent,
  component: IrodoriRunComponent,
  model: ModelComponent,
  sessionOptions: SessionOptions,
  body: (
    run: (inputs: RunInputs) => Promise<Record<string, Tensor>>,
    session: Session,
  ) => Promise<T>,
): Promise<T> => {
  await emit({ kind: "stage", component, at: "start" });
  const result = await withSession(
    state.gpu,
    model,
    sessionOptions,
    observer(state, component),
    body,
  );
  await emit({ kind: "stage", component, at: "end" });
  return result;
};
