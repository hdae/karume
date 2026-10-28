/**
 * The MOSS-Image presets (`../src/moss/preset.ts` — ADR NNNN decision 13, stage 5): the table,
 * the lookup and the acquisition options. CPU only. The presets' effect on the Sessions and the
 * estimates is checked where they are used (`moss_dit_host_test.ts` for the estimates,
 * `packages/e2e/tests/e2e_moss_ab_test.ts` for the census of the Sessions).
 */

import { describe, expect, it } from "vite-plus/test";
import { assertRejects, assertThrows } from "@std/assert";
import { sessionOptionsViolation } from "@karume/runtime";
import {
  MOSS_PRACTICAL_PRESET,
  MOSS_PRESET_NAMES,
  MOSS_REFERENCE_PRESET,
  mossAcquireGpuOptions,
  mossPreset,
  requireMossPreset,
} from "../src/moss/preset.ts";
import { mossDitStageNeeds } from "../src/moss/dit-stage.ts";
import { generateMoss } from "../src/moss/generate.ts";

describe("MOSS-Image presets", () => {
  it("holds the reference preset (no knob) and the practical preset (a8 linear and attention, online)", () => {
    expect(MOSS_PRESET_NAMES).toEqual(["f32+dit8", "f32+dit8-a8-attn8-online"]);
    expect(mossPreset(MOSS_REFERENCE_PRESET)).toEqual({ name: "f32+dit8", dit: {} });
    expect(mossPreset(MOSS_PRACTICAL_PRESET)).toEqual({
      name: "f32+dit8-a8-attn8-online",
      dit: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" },
    });
  });

  it("gives knobs every Session accepts", () => {
    for (const name of MOSS_PRESET_NAMES) {
      expect(sessionOptionsViolation(mossPreset(name).dit)).toBeUndefined();
    }
  });

  it("refuses an unknown name, naming the presets there are", () => {
    assertThrows(
      () => mossPreset("i8"),
      Error,
      "no preset 'i8' (there are: f32+dit8 / f32+dit8-a8-attn8-online)",
    );
  });

  it("refuses a state without a preset by name at every entry that reads one (there is no default)", async () => {
    const practical = mossPreset(MOSS_PRACTICAL_PRESET);
    expect(requireMossPreset(practical)).toBe(practical);
    const message = "the state names no preset";
    assertThrows(() => requireMossPreset(undefined), Error, message);
    // A caller outside the types (JavaScript): Reflect.apply takes the arguments untyped. The check
    // comes before anything else is read, so the rest of the state and the plan may be absent.
    assertThrows(
      () => Reflect.apply(mossDitStageNeeds, undefined, [{}, undefined]),
      Error,
      message,
    );
    await assertRejects(() => Reflect.apply(generateMoss, undefined, [{}, {}]), Error, message);
  });

  it("asks for subgroups when available for the online preset only, and requires nothing", () => {
    expect(mossAcquireGpuOptions(mossPreset(MOSS_REFERENCE_PRESET))).toEqual({});
    expect(mossAcquireGpuOptions(mossPreset(MOSS_PRACTICAL_PRESET))).toEqual({
      subgroups: "available",
    });
  });
});
