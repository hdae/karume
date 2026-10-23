/**
 * The mapping from the manifest's `session` (manifest-owned vocabulary) → runtime `SessionOptions`,
 * and its **composition** with the user's explicit settings (**pipeline-independent shared
 * processing** — every family's admission goes through this one function).
 *
 * The composition rule is one rule for every family: per key, **explicit > quant declaration >
 * runtime default** (ADR 0058 addendum 2026-09-26). Families differ only in "which keys they
 * accept" ({@link FamilySessionPolicy}); precedence, value ranges, combinations and the
 * classification of the thrown type are not written per family.
 *
 * MUST: not exported from the barrel. This is an internal mechanism that translates the
 * distribution's declarations into runtime knobs, not a surface the user touches (`export` exists so
 * the package's tests can exercise the mapping itself — a gap in the mapping sits where it would not
 * show without running the GPU).
 *
 * NOTE: this used to be duplicated byte for byte into the `pipeline.ts` of 7 families. With
 * duplication, "renaming a spelling" is caught by type checking, but "adding a key to
 * `SessionSpec`" **passes type checking in every family** (the mapping just drops keys it does not
 * write), so families that forgot to follow degraded silently. Consolidating into one plus the
 * exhaustive table below removes that room structurally.
 */

import type { SessionSpec } from "@karume/hub";
import { type SessionOptions, sessionOptionsViolation } from "@karume/runtime";

import { ModelInputError } from "../errors.ts";

/**
 * The type of the fields the user can override explicitly (the same key set as the manifest
 * vocabulary; the values take the runtime's value range).
 *
 * NOTE: the value range is the runtime's rather than `SessionSpec`'s because the user may also pick
 * execution forms the distribution does not declare (e.g. `linearGemvReduce: "parallel-subgroup32"`
 * — depends on the device's subgroups).
 */
export type SessionOverrides = Partial<Pick<SessionOptions, keyof SessionSpec>>;

/**
 * Per-family **acceptance table** — true = a key this family accepts both from the manifest's
 * declaration and from explicit settings.
 *
 * MUST: the key set **covers** `Required<SessionSpec>` (the same constraint as {@link WRITERS}).
 * When a knob is added to `SessionSpec`, every family's table fails type checking, so no family can
 * forget to decide "does it accept the new knob". It holds no value defaults (the runtime owns the
 * defaults in one place).
 */
export type FamilySessionPolicy = { readonly [K in keyof Required<SessionSpec>]: boolean };

/** The mapping of one key — builds the `SessionOptions` field only when the `source` field is set. */
type SpecWriter = (source: SessionOverrides) => SessionOptions;

/**
 * The table that maps each manifest-owned key onto a `SessionOptions` field.
 *
 * MUST: the key set **covers** `Required<SessionSpec>` — when a knob is added to `SessionSpec`,
 * this declaration fails type checking, so the mapping cannot fall behind (= a silent degradation
 * where a knob the distribution declared never reaches the runtime).
 *
 * MUST: no writer hands everything over with a spread (ADR 0038 §3 — with a pass-through, the moment
 * a spelling changes the runtime silently ignores the unknown key; with an explicit mapping, type
 * checking fails as soon as the spellings diverge). That knobs without a slot on the manifest side,
 * such as `SessionOptions.submitPolicy` (the TDR budget = host policy), do not get mixed in is also
 * an effect of this explicit mapping.
 * MUST: each writer reads its field **only once** (destructuring). Explicit settings are the user's
 * object, so a getter must not be able to swap the value that was checked for the value that is
 * copied.
 */
const WRITERS: { readonly [K in keyof Required<SessionSpec>]: SpecWriter } = {
  linearCompute: ({ linearCompute }) => (linearCompute === undefined ? {} : { linearCompute }),
  attentionCompute: ({ attentionCompute }) =>
    attentionCompute === undefined ? {} : { attentionCompute },
  attentionScoreStorage: ({ attentionScoreStorage }) =>
    attentionScoreStorage === undefined ? {} : { attentionScoreStorage },
  linearGemvReduce: ({ linearGemvReduce }) =>
    linearGemvReduce === undefined ? {} : { linearGemvReduce },
  stateAttentionReduce: ({ stateAttentionReduce }) =>
    stateAttentionReduce === undefined ? {} : { stateAttentionReduce },
  fuseRmsNormAdd: ({ fuseRmsNormAdd }) => (fuseRmsNormAdd === undefined ? {} : { fuseRmsNormAdd }),
  fuseLinearStaticQuantize: ({ fuseLinearStaticQuantize }) =>
    fuseLinearStaticQuantize === undefined ? {} : { fuseLinearStaticQuantize },
  packedStaticQuantize: ({ packedStaticQuantize }) =>
    packedStaticQuantize === undefined ? {} : { packedStaticQuantize },
};

/** Builds `SessionOptions` holding only the set fields (an unset key gets no field at all). */
const writeOptions = (source: SessionOverrides): SessionOptions => {
  let options: SessionOptions = {};
  for (const write of Object.values(WRITERS)) options = { ...options, ...write(source) };
  return options;
};

/** Builds `SessionOptions` holding only the declared fields (an unset key gets no field at all). */
export const toSessionOptions = (spec: SessionSpec): SessionOptions => writeOptions(spec);

/** Accepted set of strings (`Record<union, true>` — when the union gains a value, the omission turns red in type checking). */
const isOneOf = (accepted: Readonly<Record<string, true>>, value: unknown): boolean =>
  typeof value === "string" && Object.hasOwn(accepted, value);

const LINEAR_COMPUTES: Readonly<Record<NonNullable<SessionOptions["linearCompute"]>, true>> = {
  f32: true,
  a8: true,
  f16: true,
};
const ATTENTION_COMPUTES: Readonly<Record<NonNullable<SessionOptions["attentionCompute"]>, true>> =
  { f32: true, f16: true, a8: true };
const SCORE_STORAGES: Readonly<Record<NonNullable<SessionOptions["attentionScoreStorage"]>, true>> =
  { f32: true, f16: true, none: true };
const LINEAR_GEMV_REDUCES: Readonly<Record<NonNullable<SessionOptions["linearGemvReduce"]>, true>> =
  { sequential: true, parallel: true, "parallel-subgroup32": true };
const STATE_ATTENTION_REDUCES: Readonly<
  Record<NonNullable<SessionOptions["stateAttentionReduce"]>, true>
> = { sequential: true, parallel: true, "parallel-fused": true };

/**
 * The value-range check of one field (passes when the field is absent).
 *
 * MUST: runs **before** the runtime's `sessionOptionsViolation`. The runtime reads omissions as the
 * default via `??`, the same way construction does, so something like `null` ("the field is there
 * but the value is invalid") would pass disguised as the default — rejecting an explicit `null`
 * instead of falling back to the declaration or the default is this layer's job.
 * MUST: diagnostics do not invoke the user's conversions (`toString` / `Symbol.toPrimitive`) — only
 * the field name, never the value.
 */
const VALUE_GATES: {
  readonly [K in keyof Required<SessionSpec>]: (options: SessionOptions) => string | undefined;
} = {
  linearCompute: ({ linearCompute }) =>
    linearCompute === undefined || isOneOf(LINEAR_COMPUTES, linearCompute)
      ? undefined
      : "linearCompute is invalid",
  attentionCompute: ({ attentionCompute }) =>
    attentionCompute === undefined || isOneOf(ATTENTION_COMPUTES, attentionCompute)
      ? undefined
      : "attentionCompute is invalid",
  attentionScoreStorage: ({ attentionScoreStorage }) =>
    attentionScoreStorage === undefined || isOneOf(SCORE_STORAGES, attentionScoreStorage)
      ? undefined
      : "attentionScoreStorage is invalid",
  linearGemvReduce: ({ linearGemvReduce }) =>
    linearGemvReduce === undefined || isOneOf(LINEAR_GEMV_REDUCES, linearGemvReduce)
      ? undefined
      : "linearGemvReduce is invalid",
  stateAttentionReduce: ({ stateAttentionReduce }) =>
    stateAttentionReduce === undefined || isOneOf(STATE_ATTENTION_REDUCES, stateAttentionReduce)
      ? undefined
      : "stateAttentionReduce is invalid",
  fuseRmsNormAdd: ({ fuseRmsNormAdd }) =>
    fuseRmsNormAdd === undefined || typeof fuseRmsNormAdd === "boolean"
      ? undefined
      : "fuseRmsNormAdd must be a boolean",
  fuseLinearStaticQuantize: ({ fuseLinearStaticQuantize }) =>
    fuseLinearStaticQuantize === undefined || typeof fuseLinearStaticQuantize === "boolean"
      ? undefined
      : "fuseLinearStaticQuantize must be a boolean",
  packedStaticQuantize: ({ packedStaticQuantize }) =>
    packedStaticQuantize === undefined || typeof packedStaticQuantize === "boolean"
      ? undefined
      : "packedStaticQuantize must be a boolean",
};

/**
 * Checks value range, type and combination in **one place** (the canonical acceptance condition —
 * conditions are not split by origin).
 *
 * Combinations (`fuseLinearStaticQuantize` needs `linearGemvReduce: parallel`, etc.) borrow the
 * runtime's `sessionOptionsViolation` as is — Session construction rejects with the same function,
 * so a copy here could split the two accepted sets. Conditions that need GPU capabilities
 * (shader-f16 / subgroups) are not decided here (they fail at the step that looks at the device).
 */
const optionsViolation = (options: SessionOptions): string | undefined => {
  for (const gate of Object.values(VALUE_GATES)) {
    const violation = gate(options);
    if (violation !== undefined) return violation;
  }
  return sessionOptionsViolation(options);
};

/** Looks up the acceptance table by name (to accept string keys from `Object.keys`, outside the type). */
const accepts = (policy: FamilySessionPolicy, key: string): boolean => {
  const table: Readonly<Record<string, boolean | undefined>> = policy;
  return Object.hasOwn(table, key) && table[key] === true;
};

/**
 * Composes the quant declaration (`declared`) with the user's explicit settings (`overrides`) and
 * decides the `SessionOptions` passed to the Session (ADR 0058 addendum 2026-09-26 — the rule common
 * to every family).
 *
 * - Per key: **the explicit value if any, else the declaration, and with neither no field at all**
 *   (runtime default). An explicit `false` beats a declared `true` (only `undefined` means "not
 *   given").
 * - An invalid explicit value (`null` / non-boolean / a spelling outside the accepted set) does not
 *   fall back to the declaration; it is a `ModelInputError`.
 * - A key the `policy` does not accept, declared by the manifest, is a plain `Error` (an asset
 *   inconsistency — the caller cannot fix it by fixing the input). A key it does not accept, given
 *   explicitly, is a `ModelInputError` (dropping it silently would mean "I set it but it has no
 *   effect").
 * - When the effective settings violate the acceptance condition, the thrown type is split by origin
 *   (ADR 0107 decision 2): if the same violation arises from the declaration **alone** it is an
 *   `Error`; if it arises only once explicit settings are involved it is a `ModelInputError`.
 *
 * MUST: called from the family's admission step (**before a single byte of the weight parts is
 * fetched**). Called later, both an unsupported declaration and a wrong explicit setting would fail
 * only after a GB-scale fetch.
 *
 * @param where Subject of the diagnostic (`"AnimaPipeline: quant 'f16'"`).
 */
export const resolveSessionOptions = (
  policy: FamilySessionPolicy,
  declared: SessionSpec,
  overrides: SessionOverrides,
  where: string,
): SessionOptions => {
  for (const key of Object.keys(declared)) {
    if (!accepts(policy, key)) throw new Error(`${where}: session.${key} is not supported`);
  }
  // So that a getter cannot swap the value that was checked for the value used in the merge, the
  // explicit settings are read once and copied into a plain object (`writeOptions` reads each field
  // once and creates no field for `undefined`).
  const explicit = writeOptions(overrides);
  for (const key of Object.keys(explicit)) {
    if (!accepts(policy, key)) {
      throw new ModelInputError(`${where}: ${key} cannot be specified for this family`);
    }
  }
  // ?? is not used because it would send null back to the declaration — an explicit field overrides
  // the declaration whatever its value, and an invalid value fails in the check right after.
  const effective: SessionOptions = { ...writeOptions(declared), ...explicit };
  const violation = optionsViolation(effective);
  if (violation !== undefined) {
    // Only the thrown type is split by origin (ADR 0107 decision 2). The decision compares the
    // identity of the violation — if the same violation arises from the declaration alone, the
    // remedy is fixing the distribution (a 500 equivalent); if it arises only once the overrides are
    // involved, the remedy is fixing the settings (a 400 equivalent).
    const message = `${where}: ${violation}`;
    throw violation === optionsViolation(writeOptions(declared))
      ? new Error(message)
      : new ModelInputError(message);
  }
  return effective;
};

/**
 * The check that looks at **only the caller's explicit settings** (input-caused = `ModelInputError`).
 *
 * MUST: entries without a quant declaration (the surfaces of `fromAssets` that take no manifest) run
 * this check **before opening a single byte of the assets**. Without it, the same wrong setting
 * travels down to Session construction, turns into the runtime's `ExecutionError`, and the
 * classification diverges from the entries that have a manifest. It decides the same as
 * {@link resolveSessionOptions} with the declaration `{}`.
 */
export const assertSessionOverrides = (
  policy: FamilySessionPolicy,
  overrides: SessionOverrides,
  where: string,
): void => {
  resolveSessionOptions(policy, {}, overrides, where);
};
