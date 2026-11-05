import { type ChatJsonObject, ChatSession, type ChatSessionHost } from "@karume/models/gemma";
import { afterEach, describe, expect, it, vi } from "vite-plus/test";

import {
  evaluateExpression,
  runSampleTool,
  SAMPLE_TOOL_NAMES,
  SAMPLE_TOOLS,
  sampleToolsOf,
  toolResultsOf,
} from "../src/models/gemma4/sample-tools.ts";

/** Runs a call against every sample tool and parses the result text. */
const run = (name: string, args: ChatJsonObject | null): Record<string, unknown> =>
  JSON.parse(runSampleTool(SAMPLE_TOOLS, name, args)) as Record<string, unknown>;

describe("the sample tool declarations", () => {
  it("are accepted by the session's construction check (names, schemas)", () => {
    // The session checks the declarations at construction and reads nothing else of the host there.
    const host = {
      tokenizer: "tokenizer",
      program: { capacity: 4096, chunkLength: 768, maxPosition: 131072 },
      defaultSampler: undefined,
      sequence: () => Promise.reject(new Error("not used")),
    } as unknown as ChatSessionHost;
    const declarations = SAMPLE_TOOLS.map(({ declaration }) => declaration);
    expect(() => new ChatSession(host, { maxNewTokens: 8, tools: declarations })).not.toThrow();
  });

  it("name each tool as its declaration does, in the settings order", () => {
    expect(SAMPLE_TOOLS.map(({ name }) => name)).toEqual([...SAMPLE_TOOL_NAMES]);
    for (const tool of SAMPLE_TOOLS) {
      expect(tool.declaration.function.name).toBe(tool.name);
      expect(tool.name).toMatch(/^\w+$/);
    }
  });

  it("picks the chosen tools in declaration order", () => {
    expect(sampleToolsOf(["get_weather", "calculate"]).map(({ name }) => name)).toEqual([
      "calculate",
      "get_weather",
    ]);
  });
});

describe("evaluateExpression", () => {
  it("applies the usual precedence and parentheses", () => {
    expect(evaluateExpression("1 + 2 * 3")).toBe(7);
    expect(evaluateExpression("(1 + 2) * 3")).toBe(9);
    expect(evaluateExpression("10 - 4 - 3")).toBe(3);
    expect(evaluateExpression("8 / 4 / 2")).toBe(1);
    expect(evaluateExpression("(12.5 + 7) * 3^2 - 4 / 8")).toBe(175);
  });

  it("raises right-associatively and tighter than unary minus", () => {
    expect(evaluateExpression("2^3^2")).toBe(512);
    expect(evaluateExpression("-2^2")).toBe(-4);
    expect(evaluateExpression("(-2)^2")).toBe(4);
    expect(evaluateExpression("2^-1")).toBe(0.5);
  });

  it("reads unary minus, decimals and spaces", () => {
    expect(evaluateExpression("--3")).toBe(3);
    expect(evaluateExpression("-(1.5 + .5)")).toBe(-2);
    expect(evaluateExpression("  3 *-2 ")).toBe(-6);
  });

  it("refuses division by zero and input that does not parse", () => {
    expect(() => evaluateExpression("1 / 0")).toThrow(/division by zero/);
    expect(() => evaluateExpression("1 / (2 - 2)")).toThrow(/division by zero/);
    expect(() => evaluateExpression("2 +")).toThrow(/unexpected end/);
    expect(() => evaluateExpression("(1 + 2")).toThrow(/missing '\)'/);
    expect(() => evaluateExpression("1 + 2)")).toThrow(/unexpected '\)'/);
    expect(() => evaluateExpression("alert(1)")).toThrow(/unexpected 'a'/);
    expect(() => evaluateExpression("")).toThrow(/unexpected end/);
    expect(() => evaluateExpression("10^400")).toThrow(/no finite value/);
  });
});

describe("runSampleTool (the result text sent back to the model)", () => {
  it("returns the calculation with its expression", () => {
    expect(run("calculate", { expression: "6 * 7" })).toEqual({ expression: "6 * 7", result: 42 });
  });

  it("answers a bad or missing argument with an error result instead of throwing", () => {
    expect(run("calculate", { expression: "1 / 0" })).toEqual({
      error: expect.stringMatching(/division by zero/),
    });
    expect(run("calculate", {})).toEqual({
      error: "'expression' is required (a non-empty string)",
    });
    expect(run("calculate", null)).toEqual({
      error: "'expression' is required (a non-empty string)",
    });
    expect(run("calculate", { expression: 42 })).toEqual({
      error: "'expression' must be a string (got 42)",
    });
    expect(run("get_weather", { city: "Kyoto", unit: "kelvin" })).toEqual({
      error: "'unit' must be celsius or fahrenheit (got 'kelvin')",
    });
    expect(run("roll_dice", { count: 11 })).toEqual({
      error: "'count' must be an integer from 1 to 10 (got 11)",
    });
  });

  it("answers a call to a tool the session does not declare with an error result", () => {
    expect(
      JSON.parse(runSampleTool(sampleToolsOf(["calculate"]), "get_weather", { city: "Kyoto" })),
    ).toEqual({ error: "unknown tool 'get_weather' (available: calculate)" });
    expect(JSON.parse(runSampleTool([], "calculate", {}))).toEqual({
      error: "unknown tool 'calculate' (available: none)",
    });
  });

  it("lets a runner's own failure (not a bad argument) through instead of answering it", () => {
    const [calculateTool] = sampleToolsOf(["calculate"]);
    const broken = {
      ...calculateTool,
      run: (): string => {
        throw new TypeError("runner bug");
      },
    };
    expect(() => runSampleTool([broken], "calculate", { expression: "1" })).toThrow("runner bug");
  });

  it("derives the weather from the city name alone, and says it is sample data", () => {
    const kyoto = run("get_weather", { city: "Kyoto" });
    expect(run("get_weather", { city: " kyoto " })).toEqual({ ...kyoto, city: "kyoto" });
    expect(kyoto).toMatchObject({
      city: "Kyoto",
      unit: "celsius",
      note: "Sample data generated from the city name, not a real forecast.",
    });
    expect(run("get_weather", { city: "Paris" })).not.toEqual({ ...kyoto, city: "Paris" });
    const celsius = Number(kyoto["temperature"]);
    expect(run("get_weather", { city: "Kyoto", unit: "fahrenheit" })).toMatchObject({
      temperature: Math.round((celsius * 9) / 5 + 32),
      unit: "fahrenheit",
    });
  });

  describe("get_current_time", () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    it("tells the local time and offset of an IANA zone", () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date("2026-11-05T12:34:56Z"));
      expect(run("get_current_time", { timezone: "Asia/Tokyo" })).toEqual({
        timezone: "Asia/Tokyo",
        local_time: "2026-11-05 21:34:56",
        weekday: "Thursday",
        utc_offset: "+09:00",
      });
      expect(run("get_current_time", { timezone: "UTC" })).toMatchObject({
        local_time: "2026-11-05 12:34:56",
        utc_offset: "+00:00",
      });
    });

    it("falls back to the browser's zone and refuses an unknown one", () => {
      expect(run("get_current_time", {})).toMatchObject({
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      });
      expect(run("get_current_time", { timezone: "Mars/Olympus" })).toEqual({
        error: "unknown time zone 'Mars/Olympus' (use an IANA name such as Asia/Tokyo)",
      });
    });
  });

  it("rolls the dice it is asked for, within their sides", () => {
    const result = run("roll_dice", { count: 5, sides: "4" });
    const rolls = result["rolls"] as number[];
    expect(rolls).toHaveLength(5);
    expect(rolls.every((roll) => Number.isInteger(roll) && roll >= 1 && roll <= 4)).toBe(true);
    expect(result["total"]).toBe(rolls.reduce((sum, roll) => sum + roll, 0));
  });
});

describe("toolResultsOf", () => {
  it("answers each call in order, naming it by id and function", () => {
    const results = toolResultsOf(sampleToolsOf(["calculate"]), [
      { id: "call_1", function: { name: "calculate", arguments: { expression: "1+1" } } },
      { id: "call_2", function: { name: "get_weather", arguments: { city: "Kyoto" } } },
    ]);
    expect(results).toEqual([
      {
        role: "tool",
        tool_call_id: "call_1",
        name: "calculate",
        content: JSON.stringify({ expression: "1+1", result: 2 }),
      },
      {
        role: "tool",
        tool_call_id: "call_2",
        name: "get_weather",
        content: JSON.stringify({ error: "unknown tool 'get_weather' (available: calculate)" }),
      },
    ]);
  });
});
