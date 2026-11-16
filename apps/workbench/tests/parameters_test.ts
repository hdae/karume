import { describe, expect, it } from "vite-plus/test";

import {
  booleanValue,
  choicesValue,
  choiceValue,
  integerValue,
  type Parameter,
  ParameterError,
  parameterText,
  parseParameters,
} from "../src/entry.ts";

const DECLARED: readonly Parameter[] = [
  { kind: "boolean", name: "warm", label: "Warm up", default: false },
  { kind: "integer", name: "rounds", label: "Rounds", min: 1, max: 9, default: 3 },
  { kind: "choice", name: "storage", label: "Storage", options: ["f16", "f32"], default: "f32" },
  {
    kind: "choices",
    name: "checks",
    label: "Checks",
    options: ["a", "b", "c"],
    default: ["a", "b", "c"],
  },
];

const rejection = (given: Record<string, string>): ParameterError => {
  try {
    parseParameters(DECLARED, given);
  } catch (error) {
    if (error instanceof ParameterError) return error;
    throw error;
  }
  throw new Error("expected a ParameterError");
};

describe("parseParameters", () => {
  it("gives every declared parameter its default when nothing is given", () => {
    expect(parseParameters(DECLARED, {})).toEqual({
      warm: false,
      rounds: 3,
      storage: "f32",
      checks: ["a", "b", "c"],
    });
  });

  it("reads a boolean as true or false only", () => {
    expect(parseParameters(DECLARED, { warm: "true" })["warm"]).toBe(true);
    expect(parseParameters(DECLARED, { warm: "false" })["warm"]).toBe(false);
    expect(rejection({ warm: "1" }).parameter).toBe("warm");
  });

  it("reads an integer within its bounds, and nothing else", () => {
    expect(parseParameters(DECLARED, { rounds: "9" })["rounds"]).toBe(9);
    expect(parseParameters(DECLARED, { rounds: "1" })["rounds"]).toBe(1);
    for (const text of ["0", "10", "2.5", "", "3x", "1e1"]) {
      expect(rejection({ rounds: text }).parameter).toBe("rounds");
    }
  });

  it("reads a choice as one of its options", () => {
    expect(parseParameters(DECLARED, { storage: "f16" })["storage"]).toBe("f16");
    expect(rejection({ storage: "bf16" }).message).toContain("storage");
  });

  it("reads choices as a comma-separated subset, kept in option order, empty for none", () => {
    expect(parseParameters(DECLARED, { checks: "c,a" })["checks"]).toEqual(["a", "c"]);
    expect(parseParameters(DECLARED, { checks: "" })["checks"]).toEqual([]);
    const error = rejection({ checks: "a,z" });
    expect(error.parameter).toBe("checks");
    expect(error.message).toContain("z");
  });

  it("rejects a name the entry does not declare, naming it", () => {
    const error = rejection({ rouds: "3" });
    expect(error.parameter).toBe("rouds");
    expect(error.message).toContain("rouds");
  });
});

describe("parameterText", () => {
  it("writes each value as the text parseParameters reads back", () => {
    const values = parseParameters(DECLARED, {
      warm: "true",
      rounds: "7",
      storage: "f16",
      checks: "b,c",
    });
    const texts = Object.fromEntries(
      Object.entries(values).map(([name, value]) => [name, parameterText(value)]),
    );
    expect(texts).toEqual({ warm: "true", rounds: "7", storage: "f16", checks: "b,c" });
    expect(parseParameters(DECLARED, texts)).toEqual(values);
  });
});

describe("the typed accessors", () => {
  const values = parseParameters(DECLARED, { warm: "true", checks: "c,a" });
  const STORAGES = ["f16", "f32"] as const;

  it("read each parsed value as its declared kind", () => {
    expect(booleanValue(values, "warm")).toBe(true);
    expect(integerValue(values, "rounds")).toBe(3);
    expect(choiceValue(values, "storage", STORAGES)).toBe("f32");
    expect(choicesValue(values, "checks", ["a", "b", "c"])).toEqual(["a", "c"]);
  });

  it("throw a ParameterError naming a value of another kind, an undeclared name or a value off the options", () => {
    expect(() => booleanValue(values, "rounds")).toThrow("Parameter rounds:");
    expect(() => integerValue(values, "warm")).toThrow("Parameter warm:");
    expect(() => choicesValue(values, "storage", STORAGES)).toThrow("Parameter storage:");
    expect(() => integerValue(values, "missing")).toThrow(ParameterError);
    expect(() => choiceValue(values, "storage", ["f16"])).toThrow("Parameter storage:");
  });
});
