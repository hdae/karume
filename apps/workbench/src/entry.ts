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

export type Parameter = BooleanParameter | IntegerParameter | ChoiceParameter | ChoicesParameter;

export type ParameterValue = boolean | number | string | readonly string[];

export type ParameterValues = Readonly<Record<string, ParameterValue>>;

/** The part of the runtime's `acquireGpu` options the host decides: `{}` in a browser, Dawn's GPU under Node. */
export type GpuOptions = Pick<AcquireGpuOptions, "gpu">;

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

/** A value as the string {@link parseParameters} reads back (the URL and the form). */
export const parameterText = (value: ParameterValue): string =>
  typeof value === "object" ? value.join(",") : String(value);
