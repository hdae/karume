import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  checkWanSeamInputBytes,
  estimateWanSeamStep,
  WAN_SEAM_INPUT_SLACK_BYTES,
  WAN_SEAM_INPUTS,
  wanSeamSizes,
} from "./wan-seam-probe.ts";

describe("wan seam sizes", () => {
  describe("at 1280x704 and 33 frames", () => {
    const sizes = wanSeamSizes({ frames: 33, width: 1280, height: 704 });

    it("has 7,920 tokens over 9 latent frames", () => {
      assertStrictEquals(sizes.tokens, 7_920);
      assertEquals(sizes.latentShape, [48, 9, 44, 80]);
    });

    it("sizes each of the seven inputs of one pass", () => {
      assertEquals(sizes.inputs, {
        tokens: 6_082_560,
        rope_cos: 4_055_040,
        rope_sin: 4_055_040,
        encoder_hidden_states: 8_388_608,
        condition_mask: 31_680,
        timesteps_proj: 1_024,
        timesteps_proj_condition: 1_024,
      });
      assertStrictEquals(sizes.inputBytes, 22_614_976);
    });

    it("sizes the output as the f32 tokens", () => {
      assertStrictEquals(sizes.outputBytes, 6_082_560);
    });
  });

  describe("at 1280x704 and 121 frames", () => {
    const sizes = wanSeamSizes({ frames: 121, width: 1280, height: 704 });

    it("has 27,280 tokens", () => {
      assertStrictEquals(sizes.tokens, 27_280);
    });

    it("sums the inputs and the output of one pass", () => {
      assertStrictEquals(sizes.inputBytes, 57_385_536);
      assertStrictEquals(sizes.outputBytes, 20_951_040);
    });
  });

  it("refuses a frame count that is not 4n+1 and a size the tokens do not tile", () => {
    assertThrows(() => wanSeamSizes({ frames: 32, width: 1280, height: 704 }), RangeError, "4n+1");
    assertThrows(() => wanSeamSizes({ frames: 33, width: 1280, height: 700 }), RangeError, "32");
  });
});

describe("wan seam input check", () => {
  const sizes = { inputBytes: 22_614_976 };
  const tolerance = WAN_SEAM_INPUTS.length * WAN_SEAM_INPUT_SLACK_BYTES;

  it("matches the session's owned inputs within the alignment slack of every input", () => {
    const check = checkWanSeamInputBytes(sizes, sizes.inputBytes + tolerance);
    assertStrictEquals(check.toleranceBytes, 1_792);
    assertStrictEquals(check.matches, true);
  });

  it("does not match past the slack, keeping both values", () => {
    assertEquals(checkWanSeamInputBytes(sizes, sizes.inputBytes - tolerance - 4), {
      expectedBytes: 22_614_976,
      planInputBytes: 22_614_976 - tolerance - 4,
      toleranceBytes: tolerance,
      matches: false,
    });
  });
});

describe("wan seam estimate", () => {
  it("adds the host work of one guided step and two passes of upload and readback", () => {
    const estimate = estimateWanSeamStep(
      {
        patchify: 5,
        unpatchify: 3,
        cfg: 4,
        uniPcStep: 13,
        nonFinite: 8,
        firstFrame: 3,
        timestepsProj: 1,
      },
      { uploadMs: 12, readbackMs: 13 },
    );
    assertEquals(estimate, {
      hostT2vMs: 37,
      hostI2vMs: 40,
      transferMs: 50,
      perStepT2vMs: 87,
      perStepI2vMs: 90,
    });
  });
});
