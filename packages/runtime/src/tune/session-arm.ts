/**
 * `sessionArm` of `@karume/runtime/tune`: an arm of the session A/B (session-ab.ts) built from what
 * a Session needs, through the public API only — a prepared model, the GpuContext, the Session
 * options and the feeds. Opening builds the Session (`PreparedModel.createContainerSession`), a run
 * is `Session.run` with the outputs read back, the report reads the Session's diagnostics, closing
 * disposes it.
 */

import type { GpuContext } from "../gpu/device.ts";
import type { PreparedModel, Session } from "../runtime/executor.ts";
import type { FusionCounts } from "../runtime/fusion.ts";
import type { SymbolBindings } from "../runtime/plan.ts";
import type { RunInputs, SessionOptions } from "../runtime/session-types.ts";
import type { SessionAbArm, SessionAbReport, SessionAbRun } from "./session-ab.ts";

/** The options of {@link sessionArm}. */
export type SessionArmOptions = {
  /** The arm's name in the record. */
  readonly name: string;
  /** The model (`prepareContainer(opened, graphName)`); every open builds a Session from it. */
  readonly prepared: PreparedModel;
  readonly gpu: GpuContext;
  /** The Session's options (the arm's configuration: `fusionRules` and the like). */
  readonly options?: SessionOptions;
  /**
   * The inputs of every run. `Tensor.data` is borrowed by every run (`Session.run`), so it MUST NOT
   * be rewritten while the A/B runs.
   */
  readonly feeds: RunInputs;
  readonly bindings?: SymbolBindings;
};

/** The counters of a run that are present (an external rule's key may be absent). */
const presentCounts = (counts: FusionCounts | undefined): Record<string, number> | undefined => {
  if (counts === undefined) return undefined;
  const present: Record<string, number> = {};
  for (const [name, count] of Object.entries(counts)) {
    if (count !== undefined) present[name] = count;
  }
  return present;
};

/**
 * An arm that runs one Session built from `prepared` with `options` on `feeds`. A run is
 * `Session.run` and nothing else (the timed span holds no diagnostics). The report, which the runner
 * reads after the clock stopped, gives the latest run's fusion counts
 * (`diagnostics().lastRunFusions`) and its dispatches: the increase of
 * `diagnostics().submit.dispatchCount` since the previous report, or since open for the first. A
 * Session with the expert cache refuses `run` and cannot be an arm of this form; wrap its own run
 * instead.
 */
export const sessionArm = (arm: SessionArmOptions): SessionAbArm => {
  let session: Session | undefined;
  /** The Session's dispatch counter at the previous report (or at open). */
  let reportedDispatches = 0;
  const opened = (): Session => {
    if (session === undefined) throw new Error(`session arm "${arm.name}": used before open`);
    return session;
  };
  return {
    name: arm.name,
    open: async () => {
      if (session !== undefined) throw new Error(`session arm "${arm.name}": opened twice`);
      session = await arm.prepared.createContainerSession(arm.gpu, arm.options);
      reportedDispatches = session.diagnostics().submit.dispatchCount;
    },
    run: async (): Promise<SessionAbRun> => {
      const results = await opened().run(arm.feeds, arm.bindings);
      return {
        outputs: Object.fromEntries(
          Object.entries(results).map(([name, tensor]) => [name, tensor.data]),
        ),
      };
    },
    report: async (): Promise<SessionAbReport> => {
      const diagnostics = opened().diagnostics();
      const dispatchCount = diagnostics.submit.dispatchCount;
      const dispatches = dispatchCount - reportedDispatches;
      reportedDispatches = dispatchCount;
      const fusions = presentCounts(diagnostics.lastRunFusions);
      return { dispatches, ...(fusions === undefined ? {} : { fusions }) };
    },
    close: async () => {
      const current = session;
      session = undefined;
      await current?.dispose();
    },
  };
};
