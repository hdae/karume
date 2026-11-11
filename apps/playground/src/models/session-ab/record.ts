/**
 * The session A/B page's saved record: the runtime's `karume-session-ab/1` record as it is, inside
 * the page's envelope with the request it ran, the distribution it read, what a reader needs to
 * run it again (the package versions, the rule's dispatch keys, the expert cache as resolved) and
 * the judgements. Pure, so `vp test` covers it; a stored envelope's `record` can be judged again
 * with `judgeSessionAb`.
 */

import type { SessionAbJudgement, SessionAbRecord } from "@karume/runtime/tune";

import type { RecordEnvironment } from "@/lib/environment";
import type { SourceRecord } from "@/lib/record";
import { entryPointOf, type GemmaEntryPoint } from "@/models/gemma4/settings";

import modelsPackage from "../../../../../packages/models/package.json" with { type: "json" };
import runtimePackage from "../../../../../packages/runtime/package.json" with { type: "json" };

import type { SessionAbRequest } from "./request";
import { type ArmFacts, type ResolvedExpertCache, ruleKeyPrefix } from "./run";

/** The version of the envelope; bump it when a field changes meaning or goes away. */
export const SESSION_AB_PAGE_FORMAT = "karume-playground-session-ab/1";

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
      /** The keys under the rule's prefix in the rule arm's first counted run (counts read). */
      readonly dispatchKeys: readonly string[];
    }
  | {
      /** The prefix every dispatch key of the rule carries (no counts read, so no keys seen). */
      readonly keyPrefix: string;
    }
);

/** The expert cache as one arm's pipeline resolved it. */
export type SessionAbArmExpertCache = ResolvedExpertCache & { readonly arm: string };

/** What one arm learned, by name (the arm builder's `facts`). */
export type SessionAbArmFacts = { readonly name: string; readonly facts: ArmFacts };

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
  /** The runtime's record (`karume-session-ab/1`), unchanged. */
  readonly record: SessionAbRecord;
  /** `judgeSessionAb(record, { minGain: request.minGain })`, one per compared arm. */
  readonly judgements: readonly SessionAbJudgement[];
};

const namesOf = (arms: readonly { readonly name: string }[]): string =>
  arms.map(({ name }) => name).join(", ");

/**
 * The envelope of one finished A/B. Refuses a record or arm facts whose arms are not the request's
 * (they would describe different runs), and an arm without its resolved expert cache when the
 * request has one.
 */
export const buildSessionAbPageRecord = (input: {
  readonly environment: RecordEnvironment;
  readonly versions: SessionAbVersions;
  readonly request: SessionAbRequest;
  readonly distribution: SessionAbDistribution;
  readonly source: SourceRecord;
  readonly ruleName: string;
  readonly arms: readonly SessionAbArmFacts[];
  readonly record: SessionAbRecord;
  readonly judgements: readonly SessionAbJudgement[];
}): SessionAbPageRecord => {
  const { request, record, ruleName, arms } = input;
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
  const ruleArm = request.arms.findIndex(({ fusionRules }) => fusionRules.includes(ruleName));
  if (ruleArm < 0) throw new Error(`No arm of the request passes the rule ${ruleName}`);
  const ruleKeys = arms[ruleArm].facts.ruleKeys;
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
    record,
    judgements: input.judgements,
  };
};
