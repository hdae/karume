/**
 * Execution scope of a temporary Session (**pipeline-independent shared processing** — used the same
 * way by families that build and tear down a Session per stage).
 *
 * MUST: not exported from the barrel (for the same reason as the co-located `options.ts` /
 * `dispose-steps.ts` / `gpu-features.ts` — an internal mechanism, not a surface the user touches).
 */

import type {
  GpuContext,
  RunInputs,
  Session,
  SessionDiagnostics,
  SessionOptions,
  Tensor,
} from "@karume/runtime";
import type { ModelComponent } from "../hub/components.ts";
import { disposeSteps } from "./dispose-steps.ts";

/**
 * Builds a Session for one graph and always releases it after use.
 * MUST: dispose in `finally` — if VRAM is left behind after a failure midway, the next stage fails
 * to allocate and fails "somewhere other than the first failure".
 *
 * NOTE: `anima/pipeline.ts` has a helper of the same name, but its `run` returns only
 * `graph.outputs[0]` — **a different thing**, not built on this one. Widening the Anima side to
 * multiple outputs would change the destructuring at every call site and carry a regression risk
 * that shows only on a real GPU — merging them is left to a refactor once both line up.
 */
export const withSession = async <T>(
  gpu: GpuContext,
  model: ModelComponent,
  sessionOptions: SessionOptions,
  observe: ((diagnostics: SessionDiagnostics) => void) | undefined,
  body: (
    run: (inputs: RunInputs) => Promise<Record<string, Tensor>>,
    session: Session,
  ) => Promise<T>,
): Promise<T> => {
  const session = await model.createSession(gpu, sessionOptions);
  let failure: { readonly error: unknown } | undefined;
  try {
    const run = async (inputs: RunInputs): Promise<Record<string, Tensor>> => {
      const outputs = await session.run(inputs);
      if (observe !== undefined) observe(session.diagnostics());
      return outputs;
    };
    return await body(run, session);
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST: a dispose failure does not overwrite the failure of body (the doc of `disposeSteps` —
    // throwing in a plain `finally` loses what broke first).
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => session.dispose(),
    ]);
  }
};
