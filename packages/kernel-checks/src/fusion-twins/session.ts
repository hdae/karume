/**
 * What the fusion-twin modules share: opening the two Sessions of a twin under the error scopes of
 * {@link scopedGpuWork} (a pipeline that fails to build or an allocation the device cannot serve
 * throws there, not later as a silent read of initial contents), and collecting a part's cases with
 * the cancellation and progress of {@link CheckOptions}.
 */

import type { BoundContainer } from "../../../runtime/src/format/container/bind.ts";
import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import {
  createSessionFromContainer,
  type Session,
  type SessionOptions,
} from "../../../runtime/src/runtime/executor.ts";
import { GRAPH_NAME } from "../../../runtime/tests/helpers/model-fixture.ts";
import { type CheckCase, type CheckOptions, scopedGpuWork, throwIfAborted } from "../result.ts";

/** Opens the fixture graph of `model` as a Session, under the error scopes. */
export const openTwinSession = (
  gpu: GpuContext,
  where: string,
  model: BoundContainer,
  options: SessionOptions = {},
): Promise<Session> =>
  scopedGpuWork(gpu, `${where}: open`, () =>
    createSessionFromContainer(gpu, model, GRAPH_NAME, options),
  );

/** One side of a twin: its model and its Session options. */
export type TwinSide = {
  readonly model: BoundContainer;
  readonly options?: SessionOptions;
};

/** Opens the fused side, then the plain side, runs `body` on both and disposes both. */
export const withTwinSessions = async <T>(
  gpu: GpuContext,
  where: string,
  fused: TwinSide,
  plain: TwinSide,
  body: (sessions: { readonly fused: Session; readonly plain: Session }) => Promise<T>,
): Promise<T> => {
  const fusedSession = await openTwinSession(gpu, `${where} (fused)`, fused.model, fused.options);
  try {
    const plainSession = await openTwinSession(gpu, `${where} (plain)`, plain.model, plain.options);
    try {
      return await body({ fused: fusedSession, plain: plainSession });
    } finally {
      await plainSession.dispose();
    }
  } finally {
    await fusedSession.dispose();
  }
};

/** The cases of one part as they run. */
export type TwinCases = {
  readonly cases: readonly CheckCase[];
  /** Called before each case: throws the signal's reason when the run was cancelled. */
  readonly next: () => void;
  /** Keeps a case and reports the progress. */
  readonly add: (checkCase: CheckCase) => void;
};

export const twinCases = (part: string, total: number, options: CheckOptions): TwinCases => {
  const cases: CheckCase[] = [];
  return {
    cases,
    next: () => throwIfAborted(options),
    add: (checkCase) => {
      cases.push(checkCase);
      options.onProgress?.({ part, done: cases.length, total });
    },
  };
};
