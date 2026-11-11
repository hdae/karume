/**
 * The session A/B page's saved record: the runtime's `karume-session-ab/1` record as it is, inside
 * the page's envelope with the request it ran, the distribution it read and the judgements. Pure,
 * so `vp test` covers it; a stored envelope's `record` can be judged again with `judgeSessionAb`.
 */

import type { SessionAbJudgement, SessionAbRecord } from "@karume/runtime/tune";

import type { RecordEnvironment } from "@/lib/environment";
import type { SourceRecord } from "@/lib/record";
import { entryPointOf, type GemmaEntryPoint } from "@/models/gemma4/settings";

import type { SessionAbRequest } from "./request";

/** The version of the envelope; bump it when a field changes meaning or goes away. */
export const SESSION_AB_PAGE_FORMAT = "karume-playground-session-ab/1";

/** The distribution every arm read. */
export type SessionAbDistribution = {
  readonly directory: string;
  /** sha256 of the mount's `karume.json`. */
  readonly manifestSha256: string;
};

export type SessionAbPageRecord = RecordEnvironment & {
  readonly format: typeof SESSION_AB_PAGE_FORMAT;
  readonly page: "session-ab";
  readonly request: SessionAbRequest;
  readonly pipeline: GemmaEntryPoint;
  readonly distribution: SessionAbDistribution;
  readonly source: SourceRecord;
  /** The rule the `rule` arm passed, by name and package. */
  readonly rule: { readonly name: string; readonly package: "@karume/candidate-rules" };
  /** The runtime's record (`karume-session-ab/1`), unchanged. */
  readonly record: SessionAbRecord;
  /** `judgeSessionAb(record, { minGain: request.minGain })`, one per compared arm. */
  readonly judgements: readonly SessionAbJudgement[];
};

/**
 * The envelope of one finished A/B. Refuses a record whose arms are not the request's (the two
 * would describe different runs).
 */
export const buildSessionAbPageRecord = (input: {
  readonly environment: RecordEnvironment;
  readonly request: SessionAbRequest;
  readonly distribution: SessionAbDistribution;
  readonly source: SourceRecord;
  readonly ruleName: string;
  readonly record: SessionAbRecord;
  readonly judgements: readonly SessionAbJudgement[];
}): SessionAbPageRecord => {
  const { request, record } = input;
  const requested = request.arms.map(({ name }) => name).join(", ");
  const recorded = record.arms.map(({ name }) => name).join(", ");
  if (requested !== recorded || record.rounds !== request.rounds) {
    throw new Error(
      `The A/B record (${recorded}; ${record.rounds} rounds) is not the request's (${requested}; ${request.rounds} rounds)`,
    );
  }
  return {
    format: SESSION_AB_PAGE_FORMAT,
    page: "session-ab",
    ...input.environment,
    request,
    pipeline: entryPointOf(request.mount),
    distribution: input.distribution,
    source: input.source,
    rule: { name: input.ruleName, package: "@karume/candidate-rules" },
    record,
    judgements: input.judgements,
  };
};
