/**
 * The session A/B page's saved record: the runtime's `karume-session-ab/1` record as it is, inside
 * the page's envelope with the request it ran, the distribution it read, the counts of the probe
 * before the A/B (the runner's record carries none), what a reader needs to run it again (the
 * package versions, the rule's dispatch keys, the expert cache as resolved) and the judgements. Pure, so `vp test` covers it; a stored envelope's `record` can be judged again
 * with `judgeSessionAb`.
 */

import type { SessionAbJudgement, SessionAbRecord } from "@karume/runtime/tune";

import type { RecordEnvironment } from "@/lib/environment";
import type { SourceRecord } from "@/lib/record";
import { entryPointOf, type GemmaEntryPoint } from "@/models/gemma4/settings";

import modelsPackage from "../../../../../packages/models/package.json" with { type: "json" };
import runtimePackage from "../../../../../packages/runtime/package.json" with { type: "json" };

import type { SessionAbMossRequest } from "./moss";
import type { SessionAbRequest } from "./request";
import { type ArmCounts, type ArmFacts, type ResolvedExpertCache, ruleKeyPrefix } from "./run";

/** The version of the envelope; bump it when a field changes meaning or goes away. */
export const SESSION_AB_PAGE_FORMAT = "karume-playground-session-ab/2";

/** The versions of the workspace packages a build of the page runs. */
export type SessionAbVersions = { readonly runtime: string; readonly models: string };

/** The versions this build of the page was built with (the packages' own package.json). */
export const PACKAGE_VERSIONS: SessionAbVersions = {
  runtime: runtimePackage.version,
  models: modelsPackage.version,
};

/** The distribution every arm read. */
export type SessionAbDistribution = {
  readonly directory: string;
  /** sha256 of the mount's `karume.json`. */
  readonly manifestSha256: string;
};

/** The rule the `rule` arm passed, and the dispatch keys it ran under. */
export type SessionAbRuleRecord = {
  readonly name: string;
  readonly package: "@karume/candidate-rules";
} & (
  | {
      /** The keys under the rule's prefix in the counted run of the rule arm's probe. */
      readonly dispatchKeys: readonly string[];
    }
  | {
      /** The prefix every dispatch key of the rule carries (no probe, so no keys seen). */
      readonly keyPrefix: string;
    }
);

/** The expert cache as one arm's pipeline resolved it. */
export type SessionAbArmExpertCache = ResolvedExpertCache & { readonly arm: string };

/** What one arm learned, by name (the arm builder's `facts`). */
export type SessionAbArmFacts = { readonly name: string; readonly facts: ArmFacts };

/** What one arm's count probe read, by name (`probeCounts`). */
export type SessionAbArmProbe = { readonly name: string; readonly counts: ArmCounts };

/** One arm's counts as its probe read them. */
export type SessionAbArmCounts = {
  readonly arm: string;
  /** The fusion counts of the probe's counted run. */
  readonly fusions: Readonly<Record<string, number>>;
  /** The planned dispatches of the probe's counted run. */
  readonly dispatches: number;
};

export type SessionAbPageRecord = RecordEnvironment & {
  readonly format: typeof SESSION_AB_PAGE_FORMAT;
  readonly page: "session-ab";
  /** The `@karume/runtime` and `@karume/models` versions the page was built with. */
  readonly versions: SessionAbVersions;
  readonly request: SessionAbRequest;
  readonly pipeline: GemmaEntryPoint;
  readonly distribution: SessionAbDistribution;
  readonly source: SourceRecord;
  readonly rule: SessionAbRuleRecord;
  /**
   * The expert cache as each arm's pipeline resolved it, in the arms' order: slots per routed layer
   * (one count for every layer spread over them), the host tier's bytes and the depth. Absent
   * without the cache.
   */
  readonly expertCache?: readonly SessionAbArmExpertCache[];
  /**
   * The fusion counts and planned dispatches of each arm's counted run (its first decode run, or
   * its first verify run with a self-speculation depth), as the count probe before the A/B read
   * them, in the arms' order. Absent when the counts were not probed. The runtime's `record`
   * carries no counts: the arms it timed were built without the diagnostics callback.
   */
  readonly counts?: readonly SessionAbArmCounts[];
  /** The runtime's record (`karume-session-ab/1`), unchanged. */
  readonly record: SessionAbRecord;
  /** `judgeSessionAb(record, { minGain: request.minGain })`, one per compared arm. */
  readonly judgements: readonly SessionAbJudgement[];
};

const namesOf = (arms: readonly { readonly name: string }[]): string =>
  arms.map(({ name }) => name).join(", ");

/**
 * The envelope of one finished A/B. Refuses a record, arm facts or probes whose arms are not the
 * request's (they would describe different runs), probes the request did not ask for or probes
 * missing when it did, and an arm without its resolved expert cache when the request has one.
 */
export const buildSessionAbPageRecord = (input: {
  readonly environment: RecordEnvironment;
  readonly versions: SessionAbVersions;
  readonly request: SessionAbRequest;
  readonly distribution: SessionAbDistribution;
  readonly source: SourceRecord;
  readonly ruleName: string;
  readonly arms: readonly SessionAbArmFacts[];
  /** The count probes in the arms' order; present exactly when the request probes the counts. */
  readonly probes?: readonly SessionAbArmProbe[];
  readonly record: SessionAbRecord;
  readonly judgements: readonly SessionAbJudgement[];
}): SessionAbPageRecord => {
  const { request, record, ruleName, arms, probes } = input;
  const requested = namesOf(request.arms);
  const recorded = namesOf(record.arms);
  if (requested !== recorded || record.rounds !== request.rounds) {
    throw new Error(
      `The A/B record (${recorded}; ${record.rounds} rounds) is not the request's (${requested}; ${request.rounds} rounds)`,
    );
  }
  if (namesOf(arms) !== requested) {
    throw new Error(`The arm facts (${namesOf(arms)}) are not the request's arms (${requested})`);
  }
  if (request.probeCounts !== (probes !== undefined)) {
    throw new Error(
      request.probeCounts
        ? "The request probes the counts, and no probe was given"
        : "The request does not probe the counts, and probes were given",
    );
  }
  if (probes !== undefined && namesOf(probes) !== requested) {
    throw new Error(`The probes (${namesOf(probes)}) are not the request's arms (${requested})`);
  }
  const ruleArm = request.arms.findIndex(({ fusionRules }) => fusionRules.includes(ruleName));
  if (ruleArm < 0) throw new Error(`No arm of the request passes the rule ${ruleName}`);
  const ruleKeys = probes?.[ruleArm].counts.ruleKeys;
  const expertCache =
    request.expertCache === undefined
      ? undefined
      : arms.map(({ name, facts }): SessionAbArmExpertCache => {
          if (facts.expertCache === undefined) {
            throw new Error(`The arm ${name} reported no resolved expert cache`);
          }
          return { arm: name, ...facts.expertCache };
        });
  return {
    format: SESSION_AB_PAGE_FORMAT,
    page: "session-ab",
    ...input.environment,
    versions: input.versions,
    request,
    pipeline: entryPointOf(request.mount),
    distribution: input.distribution,
    source: input.source,
    rule: {
      name: ruleName,
      package: "@karume/candidate-rules",
      ...(ruleKeys === undefined
        ? { keyPrefix: ruleKeyPrefix(ruleName) }
        : { dispatchKeys: ruleKeys }),
    },
    ...(expertCache === undefined ? {} : { expertCache }),
    ...(probes === undefined
      ? {}
      : {
          counts: probes.map(({ name, counts }): SessionAbArmCounts => ({
            arm: name,
            fusions: counts.fusions,
            dispatches: counts.dispatches,
          })),
        }),
    record,
    judgements: input.judgements,
  };
};

/**
 * The version of the MOSS-Image DiT target's envelope, a record of its own (the Gemma target's
 * `karume-playground-session-ab/2` is unchanged); bump it when a field changes meaning or goes away.
 */
export const SESSION_AB_MOSS_PAGE_FORMAT = "karume-playground-session-ab-moss/1";

/** The container both arms built their Sessions from. */
export type SessionAbMossContainer = {
  readonly component: string;
  /** sha256 of its part 0 (the two documents, which name every block's sha256). */
  readonly part0Sha256: string;
};

export type SessionAbMossPageRecord = RecordEnvironment & {
  readonly format: typeof SESSION_AB_MOSS_PAGE_FORMAT;
  readonly page: "session-ab";
  readonly target: "moss-dit-forward";
  /** The `@karume/runtime` and `@karume/models` versions the page was built with. */
  readonly versions: SessionAbVersions;
  readonly request: SessionAbMossRequest;
  readonly distribution: SessionAbDistribution;
  readonly source: SourceRecord;
  readonly container: SessionAbMossContainer;
  /** The rule the `rule` arm passed; its dispatch keys all start with the prefix. */
  readonly rule: {
    readonly name: string;
    readonly package: "@karume/candidate-rules";
    readonly keyPrefix: string;
  };
  /**
   * The runtime's record (`karume-session-ab/1`), unchanged. Its arms carry the counts: every run of
   * a `sessionArm` reports its fusion counts and dispatches, and the runner refused a run whose
   * counts were not its arm's first run's.
   */
  readonly record: SessionAbRecord;
  /** `judgeSessionAb(record, { minGain: request.minGain })`, one per compared arm. */
  readonly judgements: readonly SessionAbJudgement[];
};

/**
 * The envelope of one finished A/B on the MOSS-Image DiT target. Refuses a record whose arms or
 * round count are not the request's (it would describe different runs), a request without an arm
 * that passes the rule, and a record whose arms carry no counts (every arm of this target reports
 * them).
 */
export const buildSessionAbMossPageRecord = (input: {
  readonly environment: RecordEnvironment;
  readonly versions: SessionAbVersions;
  readonly request: SessionAbMossRequest;
  readonly distribution: SessionAbDistribution;
  readonly source: SourceRecord;
  readonly container: SessionAbMossContainer;
  readonly ruleName: string;
  readonly record: SessionAbRecord;
  readonly judgements: readonly SessionAbJudgement[];
}): SessionAbMossPageRecord => {
  const { request, record, ruleName } = input;
  const requested = namesOf(request.arms);
  const recorded = namesOf(record.arms);
  if (requested !== recorded || record.rounds !== request.rounds) {
    throw new Error(
      `The A/B record (${recorded}; ${record.rounds} rounds) is not the request's (${requested}; ${request.rounds} rounds)`,
    );
  }
  if (!request.arms.some(({ fusionRules }) => fusionRules.includes(ruleName))) {
    throw new Error(`No arm of the request passes the rule ${ruleName}`);
  }
  const uncounted = record.arms.filter(
    (arm) => arm.fusions === undefined || arm.dispatches === undefined,
  );
  if (uncounted.length > 0) {
    throw new Error(`The A/B record's arm(s) ${namesOf(uncounted)} reported no counts`);
  }
  return {
    format: SESSION_AB_MOSS_PAGE_FORMAT,
    page: "session-ab",
    target: "moss-dit-forward",
    ...input.environment,
    versions: input.versions,
    request,
    distribution: input.distribution,
    source: input.source,
    container: input.container,
    rule: {
      name: ruleName,
      package: "@karume/candidate-rules",
      keyPrefix: ruleKeyPrefix(ruleName),
    },
    record,
    judgements: input.judgements,
  };
};
