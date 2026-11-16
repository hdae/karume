import { describe, expect, it } from "vite-plus/test";

import { type Entry, parseParameters } from "../src/entry.ts";
import { batchQueryOf, givenOfQuery, queryOf, selectionOfQuery } from "../src/shell/url.ts";

const ENTRY: Entry = {
  id: "probe",
  title: "Probe",
  description: "A test entry.",
  parameters: [
    { kind: "boolean", name: "warm", label: "Warm up", default: false },
    { kind: "integer", name: "rounds", label: "Rounds", min: 1, max: 9, default: 3 },
    { kind: "number", name: "gain", label: "Gain", min: 0, max: 1, default: 0.5 },
    { kind: "text", name: "arms", label: "Arms", default: "" },
    { kind: "choice", name: "set", label: "Set", options: ["quick", "quick+"], default: "quick" },
    {
      kind: "choices",
      name: "checks",
      label: "Checks",
      options: ["a", "b", "c"],
      default: ["a", "b", "c"],
    },
  ],
  run: () => Promise.resolve({ summary: [], data: null }),
};

/** The values a link carries after a trip through the address bar's text. */
const roundTrip = (given: Readonly<Record<string, string>>, backend: string) => {
  const values = parseParameters(ENTRY.parameters, given);
  const text = queryOf(ENTRY, values, backend).toString();
  return {
    values,
    back: parseParameters(ENTRY.parameters, givenOfQuery(new URLSearchParams(text))),
  };
};

describe("the page's URL", () => {
  it("gives back every kind of parameter value it was written with", () => {
    const { values, back } = roundTrip(
      {
        warm: "true",
        rounds: "7",
        gain: "0.25",
        arms: "gemma4/e2b@i4-fast:split, a=b&c",
        set: "quick+",
        checks: "a,c",
      },
      "metal",
    );
    expect(back).toEqual(values);
    expect(back).toEqual({
      warm: true,
      rounds: 7,
      gain: 0.25,
      arms: "gemma4/e2b@i4-fast:split, a=b&c",
      set: "quick+",
      checks: ["a", "c"],
    });
  });

  it("gives back the defaults and an empty choice list", () => {
    expect(roundTrip({}, "").back).toEqual(parseParameters(ENTRY.parameters, {}));
    expect(roundTrip({ checks: "" }, "").back["checks"]).toEqual([]);
  });

  it("names the entry first, then every declared parameter in declaration order", () => {
    const query = queryOf(ENTRY, parseParameters(ENTRY.parameters, {}), "");
    expect([...query.keys()]).toEqual(["entry", "warm", "rounds", "gain", "arms", "set", "checks"]);
    expect(query.get("entry")).toBe("probe");
  });

  it("carries the backend only when one is declared", () => {
    const values = parseParameters(ENTRY.parameters, {});
    expect(queryOf(ENTRY, values, "").has("backend")).toBe(false);
    expect(queryOf(ENTRY, values, "d3d12").get("backend")).toBe("d3d12");
  });

  it("reads the entry and the backend as no parameter", () => {
    expect(givenOfQuery(new URLSearchParams("entry=probe&backend=vulkan&rounds=2"))).toEqual({
      rounds: "2",
    });
    expect(givenOfQuery(new URLSearchParams("entry=probe"))).toEqual({});
  });
});

describe("the page's URL of a batch", () => {
  it("names the batch and, when one is declared, the backend, and nothing else", () => {
    expect(batchQueryOf("synthetic-quick", "").toString()).toBe("batch=synthetic-quick");
    expect(batchQueryOf("synthetic-quick", "d3d12").toString()).toBe(
      "batch=synthetic-quick&backend=d3d12",
    );
  });

  it("selects the batch it names, as an entry query selects its entry", () => {
    expect(selectionOfQuery(batchQueryOf("synthetic-quick", "metal"))).toEqual({
      kind: "batch",
      id: "synthetic-quick",
    });
    const values = parseParameters(ENTRY.parameters, {});
    expect(selectionOfQuery(queryOf(ENTRY, values, "metal"))).toEqual({
      kind: "entry",
      id: "probe",
    });
    expect(selectionOfQuery(new URLSearchParams("backend=metal"))).toEqual({ kind: "none" });
  });

  it("reads an entry and a batch in one query as both, which the shell shows as an error", () => {
    expect(selectionOfQuery(new URLSearchParams("entry=probe&batch=synthetic-quick"))).toEqual({
      kind: "both",
      entry: "probe",
      batch: "synthetic-quick",
    });
  });
});
