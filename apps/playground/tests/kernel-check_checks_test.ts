import { GEMV_INTEGER_CASES, GEMV_PLANES_CASES } from "@karume/kernel-checks";
import { describe, expect, it } from "vite-plus/test";

import { KERNEL_CHECKS } from "../src/models/kernel-check/checks.ts";
import { KERNEL_CHECK_IDS } from "../src/models/kernel-check/verdict.ts";

describe("the kernel-check page's checks", () => {
  it("has one check per id, in the order the verdicts and the record list them", () => {
    expect(KERNEL_CHECKS.map(({ id }) => id)).toEqual([...KERNEL_CHECK_IDS]);
  });

  it("runs the integer GEMV check on any device, against the case count the check exports", () => {
    const check = KERNEL_CHECKS.find(({ id }) => id === "gemv-integer");
    expect(check?.title).toBe("Integer GEMV");
    expect(check?.needsSubgroups).toBe(false);
    expect(GEMV_INTEGER_CASES).toEqual({ "gemv-integer": 692 });
  });

  it("runs the byte-plane GEMV check on any device, against the case count the check exports", () => {
    const check = KERNEL_CHECKS.find(({ id }) => id === "gemv-planes");
    expect(check?.title).toBe("Byte-plane GEMV");
    expect(check?.needsSubgroups).toBe(false);
    expect(GEMV_PLANES_CASES).toEqual({ "gemv-planes": 240 });
  });
});
