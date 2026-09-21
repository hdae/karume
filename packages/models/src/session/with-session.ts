/**
 * 一時 Session の実行スコープ（**パイプライン非依存の共通処理** — 段ごとに Session を張って
 * は畳む家族が同じ形で使う）。
 *
 * MUST: barrel には出さない（同居する `options.ts` / `dispose-steps.ts` / `gpu-features.ts`
 * と同じ理由 — 利用者が触る面ではない内部機構）。
 */

import type {
  GpuContext,
  Session,
  SessionDiagnostics,
  SessionOptions,
  Tensor,
} from "@karume/runtime";
import type { ModelComponent } from "../hub/components.ts";

/**
 * 1 グラフぶんの Session を張り、使い終わったら必ず解放する。
 * MUST: `finally` で dispose する — 途中で落ちたときに VRAM が残ると、後続の段が確保に
 * 失敗して「最初の失敗とは別の場所」で落ちる。
 *
 * NOTE: `anima/pipeline.ts` にも同名の helper があるが、あちらの `run` は `graph.outputs[0]`
 * を 1 本だけ返す**別物**で、ここには載らない。Anima 側を多出力へ広げると全呼び出し側の分解が
 * 変わり、実 GPU でしか露見しない回帰リスクを負う — 統合は両者が揃ってからのリファクタに回す。
 */
export const withSession = async <T>(
  gpu: GpuContext,
  model: ModelComponent,
  sessionOptions: SessionOptions,
  observe: ((diagnostics: SessionDiagnostics) => void) | undefined,
  body: (
    run: (inputs: Record<string, Tensor>) => Promise<Record<string, Tensor>>,
    session: Session,
  ) => Promise<T>,
): Promise<T> => {
  const session = await model.createSession(gpu, sessionOptions);
  try {
    const run = async (inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> => {
      const outputs = await session.run(inputs);
      if (observe !== undefined) observe(session.diagnostics());
      return outputs;
    };
    return await body(run, session);
  } finally {
    await session.dispose();
  }
};
