/**
 * The session A/B entries' `data`: the runtime's `karume-session-ab/1` record as it is, with the
 * request it ran, the distribution it read, the counts of the probe before the A/B (on the Gemma
 * decode, whose runner record carries none), what a reader needs to run it again (the
 * `@karume/models` version, the rule's dispatch keys or key prefix, the expert cache as resolved)
 * and the judgements. The workbench record around it holds the environment, the checkout (the
 * runtime's version) and the backend when it is known. Pure, so `vp test` covers it; a stored
 * `record` can be judged again with `judgeSessionAb`.
 */

import type { SessionAbJudgement, SessionAbRecord } from "@karume/runtime/tune";

import modelsPackage from "../../../../packages/models/package.json" with { type: "json" };
import type { DistributionFacts } from "../entry.ts";
import { entryPointOf, type GemmaEntryPoint } from "../gemma4/settings.ts";

import type { SessionAbMossRequest } from "./moss.ts";
import type { SessionAbRequest } from "./request.ts";
import { type ArmCounts, type ArmFacts, type ResolvedExpertCache, ruleKeyPrefix } from "./run.ts";
import type { SessionAbVaeRequest, SessionAbVaeTarget } from "./vae.ts";

/** The version of `@karume/models` the run used (its own package.json). */
export const MODELS_VERSION: string = modelsPackage.version;

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

/** The `data` of a Gemma 4 decode A/B. */
export type SessionAbGemmaData = {
  readonly request: SessionAbRequest;
  readonly pipeline: GemmaEntryPoint;
  /** The distribution every arm and probe read, as `openDistribution` returned it. */
  readonly distribution: DistributionFacts;
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
  readonly modelsVersion: string;
  /** The runtime's record (`karume-session-ab/1`), unchanged. */
  readonly record: SessionAbRecord;
  /** `judgeSessionAb(record, { minGain: request.minGain })`, one per compared arm. */
  readonly judgements: readonly SessionAbJudgement[];
};

const namesOf = (arms: readonly { readonly name: string }[]): string =>
  arms.map(({ name }) => name).join(", ");

/** Refuses a runner record whose arms or round count are not the request's. */
const checkRecord = (
  request: { readonly arms: readonly { readonly name: string }[]; readonly rounds: number },
  record: SessionAbRecord,
): void => {
  const requested = namesOf(request.arms);
  const recorded = namesOf(record.arms);
  if (requested !== recorded || record.rounds !== request.rounds) {
    throw new Error(
      `The A/B record (${recorded}; ${record.rounds} rounds) is not the request's (${requested}; ${request.rounds} rounds)`,
    );
  }
};

/** Refuses a runner record with an arm that reported no counts. */
const checkCounted = (record: SessionAbRecord): void => {
  const uncounted = record.arms.filter(
    (arm) => arm.fusions === undefined || arm.dispatches === undefined,
  );
  if (uncounted.length > 0) {
    throw new Error(`The A/B record's arm(s) ${namesOf(uncounted)} reported no counts`);
  }
};

/**
 * The data of one finished Gemma 4 decode A/B. Refuses a record, arm facts or probes whose arms are
 * not the request's (they would describe different runs), probes the request did not ask for or
 * probes missing when it did, and an arm without its resolved expert cache when the request has one.
 */
export const sessionAbGemmaData = (input: {
  readonly request: SessionAbRequest;
  readonly distribution: DistributionFacts;
  readonly modelsVersion: string;
  readonly ruleName: string;
  readonly arms: readonly SessionAbArmFacts[];
  /** The count probes in the arms' order; present exactly when the request probes the counts. */
  readonly probes?: readonly SessionAbArmProbe[];
  readonly record: SessionAbRecord;
  readonly judgements: readonly SessionAbJudgement[];
}): SessionAbGemmaData => {
  const { request, record, ruleName, arms, probes } = input;
  checkRecord(request, record);
  const requested = namesOf(request.arms);
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
    request,
    pipeline: entryPointOf(request.mount),
    distribution: input.distribution,
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
    modelsVersion: input.modelsVersion,
    record,
    judgements: input.judgements,
  };
};

/** The container every arm built its Sessions from. */
export type SessionAbContainer = {
  readonly component: string;
  /** sha256 of its part 0 (the two documents, which name every block's sha256). */
  readonly part0Sha256: string;
};

/** The `data` of a MOSS-Image DiT forward A/B. */
export type SessionAbMossData = {
  readonly target: "moss-dit-forward";
  readonly request: SessionAbMossRequest;
  readonly distribution: DistributionFacts;
  readonly container: SessionAbContainer;
  /**
   * The rule the `rule` arm passed (the MOSS-Image pipeline's own, from `@karume/models`); its
   * dispatch keys all start with the prefix.
   */
  readonly rule: {
    readonly name: string;
    readonly package: "@karume/models";
    readonly keyPrefix: string;
  };
  readonly modelsVersion: string;
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
 * The data of one finished A/B on the MOSS-Image DiT target. Refuses a record whose arms or round
 * count are not the request's (it would describe different runs), a request without an arm that
 * passes the rule, and a record whose arms carry no counts (every arm of this target reports them).
 */
export const sessionAbMossData = (input: {
  readonly request: SessionAbMossRequest;
  readonly distribution: DistributionFacts;
  readonly container: SessionAbContainer;
  readonly modelsVersion: string;
  readonly ruleName: string;
  readonly record: SessionAbRecord;
  readonly judgements: readonly SessionAbJudgement[];
}): SessionAbMossData => {
  const { request, record, ruleName } = input;
  checkRecord(request, record);
  if (!request.arms.some(({ fusionRules }) => fusionRules.includes(ruleName))) {
    throw new Error(`No arm of the request passes the rule ${ruleName}`);
  }
  checkCounted(record);
  return {
    target: "moss-dit-forward",
    request,
    distribution: input.distribution,
    container: input.container,
    rule: {
      name: ruleName,
      package: "@karume/models",
      keyPrefix: ruleKeyPrefix(ruleName),
    },
    modelsVersion: input.modelsVersion,
    record,
    judgements: input.judgements,
  };
};

/**
 * A rule an arm of a VAE target passed (the Wan and anima pipelines' own, from `@karume/models`);
 * its dispatch keys all start with the prefix.
 */
export type SessionAbVaeRule = {
  readonly name: string;
  readonly package: "@karume/models";
  readonly keyPrefix: string;
};

/** The `data` of a VAE forward A/B. */
export type SessionAbVaeData = {
  readonly target: SessionAbVaeTarget;
  readonly request: SessionAbVaeRequest;
  readonly distribution: DistributionFacts;
  /** The container every arm built its Sessions from. */
  readonly container: SessionAbContainer;
  /** The rules the arms passed, in the arms' order (each named once). */
  readonly rules: readonly SessionAbVaeRule[];
  readonly modelsVersion: string;
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
 * The data of one finished A/B on a VAE target. Refuses a record whose arms or round count are not
 * the request's (it would describe different runs), a request whose arms pass no rule, and a
 * record whose arms carry no counts (every arm of these targets reports them).
 */
export const sessionAbVaeData = (input: {
  readonly request: SessionAbVaeRequest;
  readonly distribution: DistributionFacts;
  readonly container: SessionAbContainer;
  readonly modelsVersion: string;
  readonly record: SessionAbRecord;
  readonly judgements: readonly SessionAbJudgement[];
}): SessionAbVaeData => {
  const { request, record } = input;
  checkRecord(request, record);
  const names = [...new Set(request.arms.flatMap(({ fusionRules }) => fusionRules))];
  if (names.length === 0) throw new Error("No arm of the request passes a rule");
  checkCounted(record);
  return {
    target: request.target,
    request,
    distribution: input.distribution,
    container: input.container,
    rules: names.map((name) => ({
      name,
      package: "@karume/models",
      keyPrefix: ruleKeyPrefix(name),
    })),
    modelsVersion: input.modelsVersion,
    record,
    judgements: input.judgements,
  };
};
