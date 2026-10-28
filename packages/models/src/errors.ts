/**
 * The **cross-family** input-caused error of `@karume/models`.
 *
 * The classification axis is **where the caller branches** (the same convention as `HubError` in
 * `@karume/hub`): this is thrown only when "the request itself cannot be accepted = fixing the input
 * makes it pass", which in an HTTP server would be a 400. Broken internal invariants, inconsistent
 * assets (manifest / tokenizer / container) and insufficient GPU capacity are thrown as a plain
 * `Error` or an existing type — those are a 500, and the caller cannot fix them by fixing the input.
 *
 * If the type were split per family, a host serving every family would have **no way but reading
 * the message string** to tell 400 from 500. That is why there is one type, and no dedicated type
 * per family.
 *
 * ## Scope
 *
 * Only checks of **the range, type and combination of a generation request**. The following two
 * have different remedies and are not mixed into the same type:
 *
 * - misspelled model / quant / sampler names (the remedy is "look up the accepted set again")
 * - violations of the call order (reuse after `dispose`, double generation — the remedy is "fix the
 *   call order")
 *
 * Both stay a plain `Error`. **Widening** the scope later is compatible for callers, but narrowing
 * it breaks those who already branch on it, so it is a breaking change.
 *
 * MUST: splitting the type **does not lower the quality of the message** (state both the
 * expectation and the actual value). The type is for branching, the message is for people, and both
 * are needed.
 *
 * NOTE: the only derived types are the 3 of `sbv2.InputError` (`./sbv2/errors.ts`),
 * `GenerationCapacityError` (`./generation/sequence.ts`) and `PromptCleanError`
 * (`./wan/text/prompt-clean.ts`). The first holds the line drawn by ADR 0072 decision 6 as its
 * specification, the second carries in fields the actual values the truncation computation needs,
 * and the third carries the reason for the rejection (reason) in a field — each needs a type because
 * it carries **additional information** beyond `instanceof`. Branch targets that add no information
 * are not split by type.
 *
 * DECIDED: [ADR 0107](../../../docs/decisions/0107-model-input-error.md).
 */
export class ModelInputError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "ModelInputError";
  }
}
