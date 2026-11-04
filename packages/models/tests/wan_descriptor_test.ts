// Internal consistency of the Wan generation descriptors (`src/wan/descriptor.ts`). A disagreement
// between fields surfaces as a request with all defaults failing with `ModelInputError` (an input
// error), mistaking a descriptor bug for a user input error. This test fails per descriptor before
// that. When a generation is added, add a row to GENERATIONS (the same checks apply).

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  WAN21_GENERATION,
  WAN22_TI2V_GENERATION,
  type WanGenerationDescriptor,
} from "../src/wan/descriptor.ts";
import { TEMPORAL_COMPRESSION } from "../src/wan/plan.ts";

/** Whether a frame count fits the VAE's temporal compression (4n+1). */
const isFrameCount = (frames: number): boolean =>
  Number.isInteger(frames) && frames >= 1 && (frames - 1) % TEMPORAL_COMPRESSION === 0;

const isPositiveInteger = (value: number): boolean => Number.isInteger(value) && value >= 1;

/** The disagreements between a descriptor's fields (empty when none). */
const inconsistencies = (generation: WanGenerationDescriptor): string[] => {
  const { acceptedSizes, minFrames, maxFrames, defaults, latents, fps, vaePatchSize } = generation;
  const found: string[] = [];
  if (acceptedSizes.length === 0) {
    found.push("the accepted set is empty");
  }
  // Unless both ends of the range are 4n+1, every choice listed in steps of 4 from the lower end
  // (gpu-lab) is rejected by the check.
  if (!isFrameCount(minFrames) || !isFrameCount(maxFrames)) {
    found.push(`an end of the frame range ${minFrames}..${maxFrames} is not 4n+1`);
  }
  if (minFrames > maxFrames) {
    found.push(`the lower end of the frame range ${minFrames}..${maxFrames} exceeds the upper end`);
  }
  if (!isPositiveInteger(fps)) {
    found.push(`fps ${fps} is not a positive integer`);
  }
  if (!isPositiveInteger(vaePatchSize)) {
    found.push(`vaePatchSize ${vaePatchSize} is not a positive integer`);
  }
  if (
    !acceptedSizes.some((size) => size.width === defaults.width && size.height === defaults.height)
  ) {
    found.push(`the default size ${defaults.width}×${defaults.height} is not in the accepted set`);
  }
  if (!isFrameCount(defaults.frames)) {
    found.push(`the default frame count ${defaults.frames} is not 4n+1`);
  }
  if (defaults.frames < minFrames || defaults.frames > maxFrames) {
    found.push(`the default frame count ${defaults.frames} is outside ${minFrames}..${maxFrames}`);
  }
  if (latents.mean.length !== latents.std.length) {
    found.push(
      `the counts of mean (${latents.mean.length}) and std (${latents.std.length}) differ`,
    );
  }
  return found;
};

const GENERATIONS: readonly (readonly [string, WanGenerationDescriptor])[] = [
  ["Wan2.1", WAN21_GENERATION],
  ["Wan2.2", WAN22_TI2V_GENERATION],
];

describe("internal consistency of the generation descriptors", () => {
  for (const [name, generation] of GENERATIONS) {
    it(`${name}: the accepted set and the range are non-empty with 4n+1 ends, the defaults lie within, fps and patch are positive integers, mean and std have equal counts`, () => {
      assertEquals(inconsistencies(generation), []);
    });
  }

  // An empty accepted set and an inverted range cannot contain the defaults, so the defaults'
  // disagreement shows up as well.
  it("control: a descriptor with one field broken at a time is caught as a disagreement of that field", () => {
    const base = WAN21_GENERATION;
    const broken: readonly (readonly [WanGenerationDescriptor, readonly string[]])[] = [
      [
        { ...base, acceptedSizes: [] },
        ["the accepted set is empty", "the default size 832×480 is not in the accepted set"],
      ],
      [{ ...base, minFrames: 6 }, ["an end of the frame range 6..81 is not 4n+1"]],
      [
        { ...base, minFrames: 37, maxFrames: 29 },
        [
          "the lower end of the frame range 37..29 exceeds the upper end",
          "the default frame count 33 is outside 37..29",
        ],
      ],
      [{ ...base, fps: 0 }, ["fps 0 is not a positive integer"]],
      [{ ...base, vaePatchSize: 1.5 }, ["vaePatchSize 1.5 is not a positive integer"]],
      [
        { ...base, defaults: { ...base.defaults, width: 480 } },
        ["the default size 480×480 is not in the accepted set"],
      ],
      [
        { ...base, defaults: { ...base.defaults, frames: 32 } },
        ["the default frame count 32 is not 4n+1"],
      ],
      [
        { ...base, defaults: { ...base.defaults, frames: 85 } },
        ["the default frame count 85 is outside 5..81"],
      ],
      [
        { ...base, latents: { mean: base.latents.mean, std: base.latents.std.slice(1) } },
        ["the counts of mean (16) and std (15) differ"],
      ],
    ];
    for (const [generation, messages] of broken) {
      assertEquals(inconsistencies(generation), messages);
    }
  });
});
