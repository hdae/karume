/**
 * The contract of a workbench entry: one module under `src/entries/` that declares its parameters
 * and runs. An entry holds no UI code; the browser shell and the Node run (`run.ts`) render and drive
 * every entry the same way, and the same module runs in both.
 *
 * Parameter values arrive as strings (a URL query, `--name=value` on the command line, form values)
 * and {@link parseParameters} is the one place that turns them into typed values.
 *
 * SPEC: docs/spec/workbench.md#an-entry
 */

import type { DistributionSource, HubRepoRef } from "@karume/hub";
import type { SourceKind } from "@karume/local-distributions/browser";
import type { MountName } from "@karume/local-distributions/mounts";
import type { AcquireGpuOptions } from "@karume/runtime";

/** What a check-like entry concludes. Absent for an entry that only measures. */
export type Verdict = "pass" | "fail" | "incomplete";

type ParameterBase = {
  readonly name: string;
  readonly label: string;
};

export type BooleanParameter = ParameterBase & {
  readonly kind: "boolean";
  readonly default: boolean;
};

export type IntegerParameter = ParameterBase & {
  readonly kind: "integer";
  readonly min: number;
  readonly max: number;
  readonly default: number;
};

/** One value of `options`. */
export type ChoiceParameter = ParameterBase & {
  readonly kind: "choice";
  readonly options: readonly string[];
  readonly default: string;
};

/** A subset of `options`, kept in the order of `options`. */
export type ChoicesParameter = ParameterBase & {
  readonly kind: "choices";
  readonly options: readonly string[];
  readonly default: readonly string[];
};

/** Any string; the entry gives it its meaning at run and throws {@link ParameterError} naming it. */
export type TextParameter = ParameterBase & {
  readonly kind: "text";
  readonly default: string;
};

/** A finite real number from `min` to `max`. */
export type NumberParameter = ParameterBase & {
  readonly kind: "number";
  readonly min: number;
  readonly max: number;
  readonly default: number;
};

export type Parameter =
  | BooleanParameter
  | IntegerParameter
  | NumberParameter
  | TextParameter
  | ChoiceParameter
  | ChoicesParameter;

export type ParameterValue = boolean | number | string | readonly string[];

export type ParameterValues = Readonly<Record<string, ParameterValue>>;

/** The part of the runtime's `acquireGpu` options the host decides: `{}` in a browser, Dawn's GPU under Node. */
export type GpuOptions = Pick<AcquireGpuOptions, "gpu">;

/**
 * How a run read a distribution: `cached` (the hub's CacheStorage over the dev server's `/hf/`
 * route), `direct` (plain fetch from the dev server's `/models/<mount>/`) or `directory` (the
 * mirror's directory, under Node).
 */
export type DistributionRead = SourceKind | "directory";

/** What a record carries about a distribution a run read. */
export type DistributionFacts = {
  /** The mount's directory relative to the repository root (`models/karume-moss-image`). */
  readonly directory: string;
  /** The sha256 of the mount's `karume.json` as it was read. */
  readonly manifestSha256: string;
  readonly read: DistributionRead;
};

export type OpenedDistribution = {
  /** What `loadManifest` and a family's `fromPretrained` take. */
  readonly ref: HubRepoRef | DistributionSource;
  readonly facts: DistributionFacts;
};

/**
 * Opens a local distribution: `source` is how a browser reads it (`cached` or `direct`); under
 * Node the mirror's directory is read whatever it says. A referenced repository is read from the
 * mount that holds it. Throws `MountUnavailableError` (`@karume/local-distributions/browser`), which
 * names the mount, when its mirror is missing or, for `cached`, when the mount has no HF-compatible
 * coordinates.
 */
export type OpenDistribution = (
  mount: MountName,
  source: SourceKind,
) => Promise<OpenedDistribution>;

export type RunContext = {
  readonly parameters: ParameterValues;
  /** Spread into `acquireGpu`. The entry acquires and releases its own GPU context. */
  readonly gpuOptions: GpuOptions;
  /**
   * Aborted by Cancel (or SIGINT under Node). An entry MAY resolve after the abort with what it has;
   * the record's outcome is then `cancelled` either way.
   */
  readonly signal: AbortSignal;
  /** Reports one status line (the latest replaces the previous one). */
  readonly progress: (text: string) => void;
  /** For an entry that reads a local distribution (a file under `src/entries/local/`). */
  readonly openDistribution: OpenDistribution;
  /**
   * The backend's name as the host knows it — what the record's `environment.backend` holds — for
   * an entry whose own record names the backend. Absent when a browser run declared none.
   */
  readonly backend?: string;
};

export type EntryResult = {
  readonly verdict?: Verdict;
  /** A few plain lines a person reads first. */
  readonly summary: readonly string[];
  /** The entry's own record body. */
  readonly data: unknown;
};

/** An entry id: kebab-case. The registry and the results directory both hold ids to it. */
export const ENTRY_ID = /^[a-z0-9]+(-[a-z0-9]+)*$/;

export type Entry = {
  /** Kebab-case ({@link ENTRY_ID}), equal to the entry's file name without `.ts`. */
  readonly id: string;
  readonly title: string;
  /** One or two sentences. */
  readonly description: string;
  readonly parameters: readonly Parameter[];
  readonly run: (context: RunContext) => Promise<EntryResult>;
};

/** A parameter value that is unknown to the entry or does not fit its declaration. */
export class ParameterError extends Error {
  readonly parameter: string;

  constructor(parameter: string, message: string) {
    super(`Parameter ${parameter}: ${message}`);
    this.name = "ParameterError";
    this.parameter = parameter;
  }
}

const parseValue = (parameter: Parameter, text: string): ParameterValue => {
  switch (parameter.kind) {
    case "boolean": {
      if (text === "true") return true;
      if (text === "false") return false;
      throw new ParameterError(parameter.name, `expected true or false, got "${text}"`);
    }
    case "integer": {
      const value = Number(text);
      if (!/^-?\d+$/.test(text) || value < parameter.min || value > parameter.max) {
        throw new ParameterError(
          parameter.name,
          `expected an integer from ${parameter.min} to ${parameter.max}, got "${text}"`,
        );
      }
      return value;
    }
    case "number": {
      const value = Number(text);
      // `Number` reads "" and surrounding whitespace as a number; the text form refuses them.
      if (
        text === "" ||
        text.trim() !== text ||
        !Number.isFinite(value) ||
        value < parameter.min ||
        value > parameter.max
      ) {
        throw new ParameterError(
          parameter.name,
          `expected a finite number from ${parameter.min} to ${parameter.max}, got "${text}"`,
        );
      }
      return value;
    }
    case "text":
      return text;
    case "choice": {
      if (parameter.options.includes(text)) return text;
      throw new ParameterError(
        parameter.name,
        `expected one of ${parameter.options.join(", ")}, got "${text}"`,
      );
    }
    case "choices": {
      const picked = text === "" ? [] : text.split(",");
      const unknown = picked.filter((option) => !parameter.options.includes(option));
      if (unknown.length > 0) {
        throw new ParameterError(
          parameter.name,
          `${unknown.join(", ")} is not one of ${parameter.options.join(", ")}`,
        );
      }
      return parameter.options.filter((option) => picked.includes(option));
    }
  }
};

/**
 * The typed values of `declared` from a string map: a missing name takes its default, a name the
 * entry does not declare and a value that does not fit throw {@link ParameterError}. `choices` is a
 * comma-separated list (empty = none).
 */
export const parseParameters = (
  declared: readonly Parameter[],
  given: Readonly<Partial<Record<string, string>>>,
): ParameterValues => {
  for (const name of Object.keys(given)) {
    if (!declared.some((parameter) => parameter.name === name)) {
      throw new ParameterError(name, "this entry has no such parameter");
    }
  }
  return Object.fromEntries(
    declared.map((parameter) => {
      const text = given[parameter.name];
      return [parameter.name, text === undefined ? parameter.default : parseValue(parameter, text)];
    }),
  );
};

const optionOf = <T extends string>(name: string, value: unknown, options: readonly T[]): T => {
  const option = options.find((candidate) => candidate === value);
  if (option === undefined) {
    throw new ParameterError(name, `expected one of ${options.join(", ")}, got ${String(value)}`);
  }
  return option;
};

/*
 * The accessors below narrow a value of {@link parseParameters}' result to its declared kind, so an
 * entry reads typed values without casts. A value of another kind means the entry read a name it
 * did not declare, or declared it as another kind: they throw {@link ParameterError}.
 */

export const booleanValue = (values: ParameterValues, name: string): boolean => {
  const value = values[name];
  if (typeof value !== "boolean") throw new ParameterError(name, "is not a boolean");
  return value;
};

export const integerValue = (values: ParameterValues, name: string): number => {
  const value = values[name];
  if (typeof value !== "number") throw new ParameterError(name, "is not an integer");
  return value;
};

export const numberValue = (values: ParameterValues, name: string): number => {
  const value = values[name];
  if (typeof value !== "number") throw new ParameterError(name, "is not a number");
  return value;
};

export const textValue = (values: ParameterValues, name: string): string => {
  const value = values[name];
  if (typeof value !== "string") throw new ParameterError(name, "is not a text");
  return value;
};

/** The value of a `choice` parameter as one of `options` (their literal type). */
export const choiceValue = <T extends string>(
  values: ParameterValues,
  name: string,
  options: readonly T[],
): T => optionOf(name, values[name], options);

/** The value of a `choices` parameter as a list of `options` (their literal type). */
export const choicesValue = <T extends string>(
  values: ParameterValues,
  name: string,
  options: readonly T[],
): readonly T[] => {
  const value = values[name];
  if (typeof value !== "object") throw new ParameterError(name, "is not a list");
  return value.map((item) => optionOf(name, item, options));
};

/** A value as the string {@link parseParameters} reads back (the URL and the form). */
export const parameterText = (value: ParameterValue): string =>
  typeof value === "object" ? value.join(",") : String(value);
