/**
 * 段 1 本を回すための器 — 生成イベントの発火口・段の Session スコープ・グラフ入出力の
 * テンソル変換。
 *
 * 10 段の説明（何をどの順で回すか）は {@link "./pipeline.ts"} 冒頭の doc が正本で、ここは
 * 「どの段でも同じ形で要るもの」だけを持つ。段の順序も資源の所有権も持たない。
 *
 * 置き場がここなのは、`./pipeline.ts` / `./conditioning.ts` / `./dit-loop.ts` の 3 つが同じ器を
 * 読むため。`./pipeline.ts` に置いたままだと、新設側がその値を取りに戻って循環 import になる
 * （型だけの参照は消去されるので `import type` で取ってよい）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { Session, SessionDiagnostics, SessionOptions, Tensor } from "@karume/runtime";

import type { IrodoriGenerateEvent, IrodoriRunComponent, IrodoriState } from "./pipeline.ts";
import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import { withSession } from "../session/with-session.ts";

/** 生成イベントの発火口（未購読なら何もしない 1 本に畳んで、発火点に分岐を置かない）。 */
export type EmitEvent = (event: IrodoriGenerateEvent) => Promise<void>;

/** 未購読のときの発火口。 */
const NO_EVENTS: EmitEvent = () => Promise.resolve();

/**
 * 要求の `onEvent` を発火口に畳む（await して例外は握らない —
 * {@link "./pipeline.ts"} の `IrodoriGenerateRequest.onEvent`）。
 */
export const emitter = (
  onEvent: ((event: IrodoriGenerateEvent) => void | Promise<void>) | undefined,
): EmitEvent =>
  onEvent === undefined ? NO_EVENTS : async (event: IrodoriGenerateEvent) => {
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

/** bool の実表現は u32 の 0 / 1（ADR 0009）。 */
export const bool = (value: boolean): Tensor => ({
  dtype: "bool",
  shape: [1, 1],
  data: Uint32Array.of(value ? 1 : 0),
});

/** グラフ出力**名**を位置で引く（IR v1 の出力名は `output.<i>` — 名前を決め打ちしない）。 */
export const outputNameAt = (model: GraphOwner, index: number): string => {
  const name = model.graph.outputs[index];
  if (name === undefined) {
    throw new Error(`グラフ出力 ${index} が無い（${model.graph.outputs.length} 本しかない）`);
  }
  return name;
};

/** グラフ出力を**位置**で引く。 */
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
 * 観測席（{@link "./pipeline.ts"} の `IrodoriPipelineOptions.onRunDiagnostics`）へ
 * コンポーネント名を焼いて渡す。
 */
export const observer = (
  state: IrodoriState,
  component: IrodoriRunComponent,
): ((diagnostics: SessionDiagnostics) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  return listener === undefined ? undefined : (diagnostics) => listener(component, diagnostics);
};

/**
 * 段 1 本を回す（`stage` イベントを Session 構築の前と解放の後に挟む）。
 * 途中で落ちたら `end` は出ない（生成ごと reject する — `onEvent` の doc）。
 */
export const withStageSession = async <T>(
  state: IrodoriState,
  emit: EmitEvent,
  component: IrodoriRunComponent,
  model: ModelComponent,
  sessionOptions: SessionOptions,
  body: (
    run: (inputs: Record<string, Tensor>) => Promise<Record<string, Tensor>>,
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
