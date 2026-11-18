/**
 * The decode bench's profile switches: the profile an arm's devices are acquired with, the GEMV
 * choices the record states for it, and the probe of a bench whose every visit has its own device.
 * No GPU: the adapters are their `vendor` / `architecture` / `description`.
 */

import { BUILTIN_GEOMETRY_PROFILES } from "@karume/runtime";
import { describe, expect, it } from "vite-plus/test";

import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
} from "../../../packages/runtime/src/kernels/geometry-profile.ts";
import {
  armGemvChoices,
  armGeometryProfile,
  type BenchAdapter,
  gemvChoicesLine,
} from "../src/gemma4-bench/profile.ts";
import { createVisitProbe, type VisitDeviceProbe } from "../src/gemma4-bench/visit-probe.ts";

/** An adapter whose table carries GEMV choices of its own (the RTX 3080 Ti's). */
const AMPERE: BenchAdapter = { vendor: "nvidia", architecture: "ampere", description: "" };
/** Another one, whose own parallel rows differ from the default profile's (the RTX 5070 Ti's). */
const BLACKWELL: BenchAdapter = { vendor: "nvidia", architecture: "blackwell", description: "" };
/** An adapter whose table carries none (the Apple M2's). */
const APPLE: BenchAdapter = { vendor: "apple", architecture: "metal-3", description: "Apple M2" };
/** An adapter no embedded table matches. */
const UNKNOWN: BenchAdapter = { vendor: "acme", architecture: "anvil", description: "" };

const tableOf = (id: string) => {
  const table = BUILTIN_GEOMETRY_PROFILES.find((profile) => profile.id === id);
  if (table === undefined) throw new Error(`no embedded table ${id}`);
  return table;
};

const DEFAULT_GEMV = DEFAULT_GEOMETRY_PROFILE.linearGemv;

describe("the profile of a bench arm", () => {
  it("is left to the runtime for an arm without a switch, whose choices are the table's own or, without them, the default profile's", () => {
    expect(armGeometryProfile(AMPERE, [])).toBeUndefined();
    const own = tableOf("nvidia-ampere").linearGemv;
    expect(own).toBeDefined();
    expect(armGemvChoices(AMPERE, [])).toEqual({
      switches: [],
      table: "nvidia-ampere",
      tableCarriesGemv: true,
      linearGemv: own,
    });
    expect(tableOf("apple-metal-3").linearGemv).toBeUndefined();
    expect(armGemvChoices(APPLE, [])).toEqual({
      switches: [],
      table: "apple-metal-3",
      tableCarriesGemv: false,
      linearGemv: DEFAULT_GEMV,
    });
  });

  it("default-gemv is the adapter's table without GEMV choices of its own: the default profile's, every other field the table's", () => {
    const profile = armGeometryProfile(AMPERE, ["default-gemv"]);
    const { linearGemv: _own, ...rest } = tableOf("nvidia-ampere");
    expect(profile).toEqual(rest);
    expect(armGemvChoices(AMPERE, ["default-gemv"]).linearGemv).toBe(DEFAULT_GEMV);
    // The table's own rows are what the switch takes away (the fault: a switch that kept them)
    expect(tableOf("nvidia-ampere").linearGemv?.sequential.shapes.length).toBeGreaterThan(
      DEFAULT_GEMV.sequential.shapes.length,
    );
    // A table without the field is itself under the switch
    expect(armGeometryProfile(APPLE, ["default-gemv"])).toEqual(tableOf("apple-metal-3"));
    // A table whose parallel rows are its own runs the default profile's rows under the switch
    expect(tableOf("nvidia-blackwell").linearGemv?.parallelShapes).not.toEqual(
      DEFAULT_GEMV.parallelShapes,
    );
    expect(armGemvChoices(BLACKWELL, ["default-gemv"]).linearGemv).toBe(DEFAULT_GEMV);
  });

  it("general-sequential removes every per-shape exception of the sequential variant and nothing else", () => {
    for (const [adapter, id] of [
      [AMPERE, "nvidia-ampere"],
      [BLACKWELL, "nvidia-blackwell"],
      [APPLE, "apple-metal-3"],
    ] as const) {
      const table = tableOf(id);
      const before = table.linearGemv ?? DEFAULT_GEMV;
      const profile = armGeometryProfile(adapter, ["general-sequential"]);
      expect(profile).toEqual({
        ...table,
        linearGemv: {
          ...before,
          sequential: { variant: before.sequential.variant, shapes: [] },
        },
      });
      expect(before.sequential.shapes.length).toBeGreaterThan(0);
      expect(armGemvChoices(adapter, ["general-sequential"]).linearGemv).toEqual(
        profile?.linearGemv,
      );
    }
  });

  it("both switches are the default profile's choices without its exceptions", () => {
    expect(armGemvChoices(AMPERE, ["default-gemv", "general-sequential"]).linearGemv).toEqual({
      ...DEFAULT_GEMV,
      sequential: { variant: DEFAULT_GEMV.sequential.variant, shapes: [] },
    });
  });

  it("is the default profile with the switches applied for an adapter no table matches", () => {
    expect(armGemvChoices(UNKNOWN, [])).toMatchObject({
      table: "default",
      linearGemv: DEFAULT_GEMV,
    });
    expect(
      armGeometryProfile(UNKNOWN, ["general-sequential"])?.linearGemv?.sequential.shapes,
    ).toEqual([]);
  });

  it("is a table the runtime's guard accepts, for every adapter and every set of switches", () => {
    for (const adapter of [AMPERE, BLACKWELL, APPLE, UNKNOWN]) {
      for (const switches of [
        ["default-gemv"],
        ["general-sequential"],
        ["default-gemv", "general-sequential"],
      ] as const) {
        const profile = armGeometryProfile(adapter, switches);
        expect(profile).toBeDefined();
        if (profile !== undefined) assertGeometryProfile(profile);
      }
    }
  });

  it("is named in one summary line: the table, whose choices, and how many exceptions and parallel rows", () => {
    const rows = DEFAULT_GEMV.parallelShapes.length;
    const own = tableOf("nvidia-ampere").linearGemv?.sequential.shapes.length;
    expect(gemvChoicesLine("arm 1 X", armGemvChoices(AMPERE, []))).toBe(
      `arm 1 X · table nvidia-ampere · the table's own GEMV choices · sequential c32u4 with ${own} per-shape exceptions · ${rows} parallel rows`,
    );
    expect(gemvChoicesLine("arm 2 X", armGemvChoices(AMPERE, ["default-gemv"]))).toBe(
      `arm 2 X · table nvidia-ampere · the default profile's GEMV choices · sequential c32u4 with ${DEFAULT_GEMV.sequential.shapes.length} per-shape exceptions · ${rows} parallel rows`,
    );
    expect(gemvChoicesLine("arm 2 X", armGemvChoices(APPLE, ["general-sequential"]))).toBe(
      `arm 2 X · table apple-metal-3 · the default profile's GEMV choices, the sequential exceptions removed · sequential c32u4 with 0 per-shape exceptions · ${rows} parallel rows`,
    );
    // A table's own parallel rows are counted as they are
    const blackwell = tableOf("nvidia-blackwell").linearGemv;
    expect(gemvChoicesLine("arm 1 X", armGemvChoices(BLACKWELL, []))).toBe(
      `arm 1 X · table nvidia-blackwell · the table's own GEMV choices · sequential c32u4 with ${blackwell?.sequential.shapes.length} per-shape exceptions · ${blackwell?.parallelShapes.length} parallel rows`,
    );
  });
});

/** A device's probe that records what was asked of it. */
const fakeDeviceProbe = (name: string, log: string[]): VisitDeviceProbe => ({
  enabled: true,
  mode: "passes",
  passes: () => {
    log.push(`${name}.passes`);
    return [];
  },
  submits: () => {
    log.push(`${name}.submits`);
    return 0;
  },
  marks: () => {
    log.push(`${name}.marks`);
    return [];
  },
  readTimes: () => {
    log.push(`${name}.readTimes`);
    return Promise.resolve();
  },
  reset: () => {
    log.push(`${name}.reset`);
  },
});

describe("the probe of a bench whose visits each have a device", () => {
  it("is the probe of the visit's device from attach to detach, off when the device is attached", async () => {
    const log: string[] = [];
    const visit = createVisitProbe();
    const first = fakeDeviceProbe("a", log);
    const second = fakeDeviceProbe("b", log);
    visit.attach(first);
    expect([first.enabled, visit.probe.enabled]).toEqual([false, false]);
    visit.probe.enabled = true;
    visit.probe.mode = "dispatches";
    expect([first.enabled, first.mode, visit.probe.mode]).toEqual([
      true,
      "dispatches",
      "dispatches",
    ]);
    visit.probe.reset();
    visit.probe.passes();
    visit.probe.submits();
    visit.probe.marks?.();
    await visit.probe.readTimes();
    visit.detach();
    visit.attach(second);
    visit.probe.reset();
    expect(log).toEqual(["a.reset", "a.passes", "a.submits", "a.marks", "a.readTimes", "b.reset"]);
    // The first visit's device is not touched after its detach
    expect([first.enabled, second.enabled]).toEqual([true, false]);
  });

  it("can be turned off with no device (the loop does so before a build), and refuses everything else", () => {
    const visit = createVisitProbe();
    visit.probe.enabled = false;
    expect(visit.probe.enabled).toBe(false);
    expect(visit.probe.mode).toBeUndefined();
    const message = "The bench read its probe with no visit's device";
    expect(() => {
      visit.probe.enabled = true;
    }).toThrow(message);
    expect(() => visit.probe.reset()).toThrow(message);
    expect(() => visit.probe.passes()).toThrow(message);
    expect(() => visit.probe.submits()).toThrow(message);
    expect(() => {
      visit.probe.mode = "host";
    }).toThrow(message);
  });
});
