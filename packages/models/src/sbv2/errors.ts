import { ModelInputError } from "../errors.ts";

/**
 * SBV2's input error.
 *
 * The classification axis is **where the caller branches** (the same convention as `HubError` in
 * `@karume/hub`): this is thrown only when "the request itself cannot be accepted", which in an
 * HTTP server would be a 400. A broken internal invariant (a `sum(word2ph)` mismatch, a broken
 * tile walk, an inconsistent asset) is thrown as a plain `Error` — that is a 500, and the caller
 * cannot fix it by fixing the input.
 *
 * MUST: splitting the type **does not lower the quality of the message** (state both the
 * expectation and the actual value). The type is for branching, the message is for people, and
 * both are needed.
 *
 * NOTE: it derives from {@link ModelInputError}, and **no further derived types are made**. The
 * family-wide "fix the input" is carried by the parent, so this type remains only to hold by name
 * the line drawn by ADR 0072 decision 6 (which SBV2 checks are input-caused and which are internal
 * invariants). hub splits into 5 types because the user's branch targets really differ (format /
 * reference / path / integrity / fetch); here there is only one, "fix the input".
 */
export class Sbv2InputError extends ModelInputError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "Sbv2InputError";
  }
}
