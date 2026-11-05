/**
 * Sample tools for the chat page: function declarations a session can be built with, and a runner
 * per tool that answers the model's call in the browser. Pure apart from the clock (`get_current_time`)
 * and the dice, so `vp test` covers it.
 *
 * A runner never throws for bad input: missing or ill-typed arguments, an unknown time zone, an
 * expression that does not parse, and a call to a tool the session does not declare all come back
 * as an error result (`{"error": "..."}`) that is sent to the model — the usual function-calling
 * practice, which lets the model correct its call or tell the user.
 */

import type {
  ChatJsonObject,
  ChatJsonValue,
  ChatTool,
  ChatToolMessage,
} from "@karume/models/gemma";

export const SAMPLE_TOOL_NAMES = [
  "get_current_time",
  "calculate",
  "get_weather",
  "roll_dice",
] as const;
export type SampleToolName = (typeof SAMPLE_TOOL_NAMES)[number];

export type SampleTool = {
  readonly name: SampleToolName;
  /** What the settings list says about the tool. */
  readonly summary: string;
  readonly declaration: ChatTool;
  /** The result text sent back to the model (JSON). */
  readonly run: (args: ChatJsonObject) => string;
};

/** Raised inside a runner for a bad argument; {@link runSampleTool} turns it into an error result. */
class ToolArgumentError extends Error {
  override name = "ToolArgumentError";
}

const resultOf = (value: ChatJsonObject): string => JSON.stringify(value);
const errorResult = (message: string): string => resultOf({ error: message });

const optionalString = (args: ChatJsonObject, key: string): string | undefined => {
  const value = args[key];
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string") {
    throw new ToolArgumentError(`'${key}' must be a string (got ${JSON.stringify(value)})`);
  }
  return value;
};

const requiredString = (args: ChatJsonObject, key: string): string => {
  const value = optionalString(args, key);
  if (value === undefined || value.trim() === "") {
    throw new ToolArgumentError(`'${key}' is required (a non-empty string)`);
  }
  return value;
};

/** An optional integer argument in `[min, max]` (a numeric string is accepted, as models write one). */
const optionalInteger = (
  args: ChatJsonObject,
  key: string,
  range: { readonly min: number; readonly max: number; readonly fallback: number },
): number => {
  const raw: ChatJsonValue | undefined = args[key];
  if (raw === undefined || raw === null) return range.fallback;
  const value = typeof raw === "string" && raw.trim() !== "" ? Number(raw) : raw;
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < range.min ||
    value > range.max
  ) {
    throw new ToolArgumentError(
      `'${key}' must be an integer from ${range.min} to ${range.max} (got ${JSON.stringify(raw)})`,
    );
  }
  return value;
};

// get_current_time

const timeIn = (timeZone: string, now: Date): ChatJsonObject => {
  let format: Intl.DateTimeFormat;
  try {
    format = new Intl.DateTimeFormat("en-US", {
      timeZone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      weekday: "long",
      hourCycle: "h23",
      timeZoneName: "longOffset",
    });
  } catch {
    throw new ToolArgumentError(
      `unknown time zone '${timeZone}' (use an IANA name such as Asia/Tokyo)`,
    );
  }
  const part = (type: Intl.DateTimeFormatPartTypes): string =>
    format.formatToParts(now).find((item) => item.type === type)?.value ?? "";
  const offset = part("timeZoneName").replace(/^GMT/, "");
  return {
    timezone: format.resolvedOptions().timeZone,
    local_time: `${part("year")}-${part("month")}-${part("day")} ${part("hour")}:${part("minute")}:${part("second")}`,
    weekday: part("weekday"),
    utc_offset: offset === "" ? "+00:00" : offset,
  };
};

const getCurrentTime: SampleTool = {
  name: "get_current_time",
  summary: "the current date and time in a time zone (the browser's clock)",
  declaration: {
    type: "function",
    function: {
      name: "get_current_time",
      description: "Returns the current local date and time in a time zone.",
      parameters: {
        type: "object",
        properties: {
          timezone: {
            type: "string",
            description:
              "An IANA time zone name such as Asia/Tokyo or Europe/Paris. Omit it for the user's own time zone.",
          },
        },
      },
    },
  },
  run: (args) => {
    const zone = optionalString(args, "timezone")?.trim();
    const timeZone =
      zone === undefined || zone === "" ? Intl.DateTimeFormat().resolvedOptions().timeZone : zone;
    return resultOf(timeIn(timeZone, new Date()));
  },
};

// calculate

/**
 * Evaluates an arithmetic expression: `+ - * / ^`, parentheses, unary minus and plus, decimals.
 * `^` binds tighter than unary minus and is right-associative (`-2^2` = -4, `2^3^2` = 512), as in
 * ordinary mathematical notation. A small recursive-descent parser — never `eval`.
 */
export const evaluateExpression = (source: string): number => {
  let at = 0;
  const skipSpace = (): void => {
    while (at < source.length && /\s/.test(source[at] ?? "")) at += 1;
  };
  const peek = (): string | undefined => {
    skipSpace();
    return source[at];
  };
  const fail = (what: string): never => {
    throw new ToolArgumentError(`${what} at position ${at + 1} of '${source}'`);
  };
  const number = (): number => {
    const match = /^(?:\d+(?:\.\d*)?|\.\d+)/.exec(source.slice(at));
    if (match === null)
      return fail(peek() === undefined ? "unexpected end" : `unexpected '${peek()}'`);
    at += match[0].length;
    return Number(match[0]);
  };
  const primary = (): number => {
    if (peek() === "(") {
      at += 1;
      const value = sum();
      if (peek() !== ")") fail("missing ')'");
      at += 1;
      return value;
    }
    return number();
  };
  const power = (): number => {
    const base = primary();
    if (peek() !== "^") return base;
    at += 1;
    return base ** unary();
  };
  const unary = (): number => {
    const sign = peek();
    if (sign === "-" || sign === "+") {
      at += 1;
      const value = unary();
      return sign === "-" ? -value : value;
    }
    return power();
  };
  const product = (): number => {
    let value = unary();
    for (;;) {
      const operator = peek();
      if (operator !== "*" && operator !== "/") return value;
      at += 1;
      const right = unary();
      if (operator === "/" && right === 0) fail("division by zero");
      value = operator === "*" ? value * right : value / right;
    }
  };
  const sum = (): number => {
    let value = product();
    for (;;) {
      const operator = peek();
      if (operator !== "+" && operator !== "-") return value;
      at += 1;
      const right = product();
      value = operator === "+" ? value + right : value - right;
    }
  };
  const value = sum();
  if (peek() !== undefined) fail(`unexpected '${peek()}'`);
  if (!Number.isFinite(value)) throw new ToolArgumentError(`'${source}' has no finite value`);
  return value;
};

const calculate: SampleTool = {
  name: "calculate",
  summary: "arithmetic with + - * / ^ and parentheses",
  declaration: {
    type: "function",
    function: {
      name: "calculate",
      description:
        "Evaluates an arithmetic expression exactly. Supports + - * / ^ (power), parentheses, unary minus and decimals.",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: "The expression, for example (12.5 + 7) * 3^2.",
          },
        },
        required: ["expression"],
      },
    },
  },
  run: (args) => {
    const expression = requiredString(args, "expression");
    return resultOf({ expression, result: evaluateExpression(expression) });
  },
};

// get_weather

const CONDITIONS = [
  "sunny",
  "partly cloudy",
  "cloudy",
  "light rain",
  "heavy rain",
  "snow",
  "fog",
] as const;

/** FNV-1a over the UTF-16 code units: a stable number per city name. */
const hashOf = (text: string): number => {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
};

const getWeather: SampleTool = {
  name: "get_weather",
  summary: "made-up weather derived from the city name (sample data, not a forecast)",
  declaration: {
    type: "function",
    function: {
      name: "get_weather",
      description:
        "Returns the current weather in a city. This is a demo tool: the values are sample data, not a real forecast.",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string", description: "The city name, for example Kyoto." },
          unit: {
            type: "string",
            description: "The temperature unit (default celsius).",
            enum: ["celsius", "fahrenheit"],
          },
        },
        required: ["city"],
      },
    },
  },
  run: (args) => {
    const city = requiredString(args, "city").trim();
    const unit = optionalString(args, "unit") ?? "celsius";
    if (unit !== "celsius" && unit !== "fahrenheit") {
      throw new ToolArgumentError(`'unit' must be celsius or fahrenheit (got '${unit}')`);
    }
    const hash = hashOf(city.toLowerCase());
    const celsius = (hash % 40) - 5;
    return resultOf({
      city,
      condition: CONDITIONS[(hash >>> 8) % CONDITIONS.length] ?? "sunny",
      temperature: unit === "celsius" ? celsius : Math.round((celsius * 9) / 5 + 32),
      unit,
      humidity_percent: 30 + ((hash >>> 16) % 61),
      wind_kph: (hash >>> 24) % 41,
      note: "Sample data generated from the city name, not a real forecast.",
    });
  },
};

// roll_dice

const rollDice: SampleTool = {
  name: "roll_dice",
  summary: "random dice rolls",
  declaration: {
    type: "function",
    function: {
      name: "roll_dice",
      description: "Rolls dice and returns each roll and the total.",
      parameters: {
        type: "object",
        properties: {
          count: { type: "integer", description: "How many dice, 1 to 10 (default 1)." },
          sides: { type: "integer", description: "Sides per die, 2 to 100 (default 6)." },
        },
      },
    },
  },
  run: (args) => {
    const count = optionalInteger(args, "count", { min: 1, max: 10, fallback: 1 });
    const sides = optionalInteger(args, "sides", { min: 2, max: 100, fallback: 6 });
    const rolls = Array.from({ length: count }, () => 1 + Math.floor(Math.random() * sides));
    return resultOf({ count, sides, rolls, total: rolls.reduce((sum, roll) => sum + roll, 0) });
  },
};

export const SAMPLE_TOOLS: readonly SampleTool[] = [
  getCurrentTime,
  calculate,
  getWeather,
  rollDice,
];

/** The tools of `names`, in {@link SAMPLE_TOOLS} order. */
export const sampleToolsOf = (names: readonly SampleToolName[]): readonly SampleTool[] =>
  SAMPLE_TOOLS.filter((tool) => names.includes(tool.name));

/**
 * Runs one call against the session's tools and returns the result text. A tool the session does
 * not declare and a bad argument are error results; any other failure of a runner throws.
 */
export const runSampleTool = (
  tools: readonly SampleTool[],
  name: string,
  args: ChatJsonObject | null,
): string => {
  const tool = tools.find((candidate) => candidate.name === name);
  if (tool === undefined) {
    const available =
      tools.length === 0 ? "none" : tools.map((candidate) => candidate.name).join(", ");
    return errorResult(`unknown tool '${name}' (available: ${available})`);
  }
  try {
    return tool.run(args ?? {});
  } catch (error) {
    // Only a bad argument is the model's to correct; anything else is a runner bug and fails loudly.
    if (!(error instanceof ToolArgumentError)) throw error;
    return errorResult(error.message);
  }
};

/** One call the model made (the shape of `reply.message.tool_calls`). */
export type ToolCallRequest = {
  readonly id?: string;
  readonly function: { readonly name: string; readonly arguments: ChatJsonObject | null };
};

/**
 * The results of a turn's calls, in order (`sendToolResults`' input): each names its call by id and
 * carries the function's name (the template accepts the two when they agree).
 */
export const toolResultsOf = (
  tools: readonly SampleTool[],
  calls: readonly ToolCallRequest[],
): (ChatToolMessage & { readonly name: string })[] =>
  calls.map((call) => ({
    role: "tool",
    ...(call.id === undefined ? {} : { tool_call_id: call.id }),
    name: call.function.name,
    content: runSampleTool(tools, call.function.name, call.function.arguments),
  }));
