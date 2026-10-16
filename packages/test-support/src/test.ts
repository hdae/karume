/**
 * Deno-shaped test registration on top of Vitest (phase-1 port scaffold).
 *
 * The Deno suite uses three shapes that Vitest does not offer as-is:
 * - `Deno.test(name, fn)` / `Deno.test({ name, ignore, fn })` → {@link test};
 * - `describe({ name, ignore, fn })` / `it({ name, ignore, fn })` from `@std/testing/bdd` →
 *   {@link describe} / {@link it};
 * - sequential sub-steps `await t.step(name, fn)` inside a test body → {@link TestContext.step}.
 *
 * Steps run sequentially inside one Vitest test. A failing step records its error and lets the
 * remaining steps run (Deno semantics: `t.step` resolves to `false`); the enclosing test fails
 * afterwards with the first error as `cause`. Deno's sanitizers (`sanitizeResources` / `sanitizeOps`)
 * have no Vitest counterpart; those options are accepted and ignored.
 */

import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe as vitestDescribe,
  test as vitestTest,
} from "vite-plus/test";

export { afterAll, afterEach, beforeAll, beforeEach };

export type TestFunction = (t: TestContext) => void | Promise<void>;

export type StepDefinition = {
  readonly name: string;
  readonly fn: TestFunction;
  readonly ignore?: boolean;
};

export type TestDefinition = StepDefinition & {
  readonly only?: boolean;
  readonly sanitizeResources?: boolean;
  readonly sanitizeOps?: boolean;
  readonly sanitizeExit?: boolean;
};

export type TestContext = {
  readonly name: string;
  step(definition: StepDefinition): Promise<boolean>;
  step(name: string, fn: TestFunction): Promise<boolean>;
};

export type SuiteDefinition = {
  readonly name: string;
  readonly fn: () => void;
  readonly ignore?: boolean;
  readonly only?: boolean;
};

/** Options of the bdd three-argument forms `it(name, options, fn)` / `describe(name, options, fn)`. */
export type RegistrationOptions = {
  readonly ignore?: boolean;
  readonly only?: boolean;
  readonly sanitizeResources?: boolean;
  readonly sanitizeOps?: boolean;
  readonly sanitizeExit?: boolean;
};

type StepFailure = { readonly path: readonly string[]; readonly error: unknown };

/** Error thrown for the enclosing test when one or more steps failed. */
export class StepFailureError extends Error {
  readonly failures: readonly StepFailure[];
  constructor(failures: readonly StepFailure[]) {
    const [first] = failures;
    const where = failures.map((failure) => failure.path.join(" > ")).join(", ");
    super(`${failures.length} step(s) failed: ${where}\n${describeError(first?.error)}`, {
      cause: first?.error,
    });
    this.name = "StepFailureError";
    this.failures = failures;
  }
}

const describeError = (error: unknown): string =>
  error instanceof Error ? (error.stack ?? `${error.name}: ${error.message}`) : String(error);

const createContext = (path: readonly string[], failures: StepFailure[]): TestContext => ({
  name: path[path.length - 1] ?? "",
  async step(first: StepDefinition | string, second?: TestFunction): Promise<boolean> {
    const definition: StepDefinition =
      typeof first === "string" ? { name: first, fn: requireFunction(second, first) } : first;
    if (definition.ignore === true) return true;
    const stepPath = [...path, definition.name];
    const before = failures.length;
    try {
      await definition.fn(createContext(stepPath, failures));
    } catch (error) {
      failures.push({ path: stepPath, error });
      return false;
    }
    return failures.length === before;
  },
});

const requireFunction = (fn: TestFunction | undefined, name: string): TestFunction => {
  if (fn === undefined) throw new TypeError(`step "${name}" has no function`);
  return fn;
};

const runWithSteps = async (name: string, fn: TestFunction): Promise<void> => {
  const failures: StepFailure[] = [];
  await fn(createContext([name], failures));
  if (failures.length > 0) throw new StepFailureError(failures);
};

const toDefinition = (
  first: TestDefinition | string,
  second: TestFunction | RegistrationOptions | undefined,
  third: TestFunction | undefined,
): TestDefinition => {
  if (typeof first !== "string") return first;
  if (typeof second === "function") return { name: first, fn: second };
  return { ...second, name: first, fn: requireFunction(third, first) };
};

/** `Deno.test` / bdd `it` replacement (string, object, and `(name, options, fn)` forms). */
export function test(definition: TestDefinition): void;
export function test(name: string, fn: TestFunction): void;
export function test(name: string, options: RegistrationOptions, fn: TestFunction): void;
export function test(
  first: TestDefinition | string,
  second?: TestFunction | RegistrationOptions,
  third?: TestFunction,
): void {
  const definition = toDefinition(first, second, third);
  const register =
    definition.only === true
      ? vitestTest.only
      : definition.ignore === true
        ? vitestTest.skip
        : vitestTest;
  register(definition.name, () => runWithSteps(definition.name, definition.fn));
}

export const it: typeof test = test;

/** bdd `describe` replacement (string, object, and `(name, options, fn)` forms). */
export function describe(definition: SuiteDefinition): void;
export function describe(name: string, fn: () => void): void;
export function describe(name: string, options: RegistrationOptions, fn: () => void): void;
export function describe(
  first: SuiteDefinition | string,
  second?: (() => void) | RegistrationOptions,
  third?: () => void,
): void {
  const definition: SuiteDefinition =
    typeof first !== "string"
      ? first
      : typeof second === "function"
        ? { name: first, fn: second }
        : { ...second, name: first, fn: requireSuite(third, first) };
  const register =
    definition.only === true
      ? vitestDescribe.only
      : definition.ignore === true
        ? vitestDescribe.skip
        : vitestDescribe;
  register(definition.name, definition.fn);
}

const requireSuite = (fn: (() => void) | undefined, name: string): (() => void) => {
  if (fn === undefined) throw new TypeError(`suite "${name}" has no function`);
  return fn;
};
