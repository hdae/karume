import { describe, expect, it } from "vite-plus/test";

import gemmaBench from "../src/entries/local/gemma4-bench.ts";
import { ParameterError, parseParameters } from "../src/entry.ts";
import { benchSettingsOf, openBenchArms, parseArmsText } from "../src/gemma4-bench/request.ts";

import { MOBILE, QAT, REGULAR } from "./helpers/gemma4-bench-manifests.ts";
import { hex, memoryDistributions } from "./helpers/session-ab-manifests.ts";

const SIGNAL = new AbortController().signal;

const FORM = "<mount>/<model>[@<preset>][:<switch>]…";
const MOUNTS = "gemma4, gemma4-qat, gemma4-qat-mobile";
const SWITCHES = "tap-off, split, linear-reorder, integer, ple-gpu, ahead2, ahead4, ahead8";

/** The drafts of `text` (the grammar alone; no distribution is read). */
const draftsOf = (text: string) => parseArmsText(text).map(({ draft }) => draft);

describe("the arms text", () => {
  it("reads an arm as <mount>/<model> with the tap on and no preset (the manifest's default)", () => {
    expect(parseArmsText("gemma4-qat/e2b")).toEqual([
      { text: "gemma4-qat/e2b", draft: { key: { mount: "gemma4-qat", model: "e2b" }, tap: true } },
    ]);
  });

  it("reads the preset after @", () => {
    expect(draftsOf("gemma4-qat-mobile/e4b@i4-gemvpar")).toEqual([
      { key: { mount: "gemma4-qat-mobile", model: "e4b" }, quant: "i4-gemvpar", tap: true },
    ]);
  });

  it.each([
    ["tap-off", { tap: false }],
    ["split", { tap: true, split: true }],
    ["linear-reorder", { tap: true, linearReorder: true }],
    ["integer", { tap: true, integerGemv: true }],
    ["ple-gpu", { tap: true, pleOnGpu: true }],
    ["ahead2", { tap: true, decodeAhead: "2" }],
    ["ahead4", { tap: true, decodeAhead: "4" }],
    ["ahead8", { tap: true, decodeAhead: "8" }],
  ] as const)("reads the switch %s", (name, switches) => {
    expect(draftsOf(`gemma4/e2b:${name}`)).toEqual([
      { key: { mount: "gemma4", model: "e2b" }, ...switches },
    ]);
  });

  it("reads the arms in order with several switches each, trimming the space around an arm", () => {
    expect(parseArmsText(" gemma4/e2b , gemma4-qat/e2b@i4-fast:integer:ple-gpu:ahead4 ")).toEqual([
      { text: "gemma4/e2b", draft: { key: { mount: "gemma4", model: "e2b" }, tap: true } },
      {
        text: "gemma4-qat/e2b@i4-fast:integer:ple-gpu:ahead4",
        draft: {
          key: { mount: "gemma4-qat", model: "e2b" },
          quant: "i4-fast",
          tap: true,
          integerGemv: true,
          pleOnGpu: true,
          decodeAhead: "4",
        },
      },
    ]);
  });

  it("is no arm for an empty or blank text (the default arms)", () => {
    expect(parseArmsText("")).toEqual([]);
    expect(parseArmsText("  ")).toEqual([]);
  });

  it.each([
    ["gemma4/e2b,", `arm 2 is empty; the arms are ${FORM}, comma-separated`],
    ["gemma4", `arm 1 "gemma4": expected ${FORM} with a mount of ${MOUNTS}`],
    ["gemma4/", `arm 1 "gemma4/": expected ${FORM} with a mount of ${MOUNTS}`],
    ["/e2b", `arm 1 "/e2b": expected ${FORM} with a mount of ${MOUNTS}`],
    ["gemma4/e2b,anima/e2b", `arm 2 "anima/e2b": 'anima' is not one of the mounts ${MOUNTS}`],
    [
      "gemma4/e2b@",
      `arm 1 "gemma4/e2b@": names an empty preset (without '@' the arm runs the default preset)`,
    ],
    [
      "gemma4/e2b@bf16@i4",
      `arm 1 "gemma4/e2b@bf16@i4": names more than one preset; an arm is ${FORM}`,
    ],
    [
      "gemma4/e2b:fast",
      `arm 1 "gemma4/e2b:fast": 'fast' is not a switch; the switches are ${SWITCHES}`,
    ],
    ["gemma4/e2b:", `arm 1 "gemma4/e2b:": has an empty switch; the switches are ${SWITCHES}`],
    ["gemma4/e2b:split:split", `arm 1 "gemma4/e2b:split:split": names the switch 'split' twice`],
    [
      "gemma4/e2b:ple-gpu:ahead2:ahead4",
      `arm 1 "gemma4/e2b:ple-gpu:ahead2:ahead4": names more than one decode run-ahead depth`,
    ],
    [
      "gemma4/e2b:tap-off:split",
      `arm 1 "gemma4/e2b:tap-off:split": split times every dispatch through the tap, so it cannot go with tap-off`,
    ],
  ])("refuses %s, naming the arm", (text, reason) => {
    expect(() => parseArmsText(text)).toThrow(new ParameterError("arms", reason));
  });
});

describe("the arms resolved against the distributions", () => {
  describe("by default", () => {
    it("are one arm per mount whose mirror is present, at its default model and preset, tap on, regular Gemma 4 first", async () => {
      const opened: string[] = [];
      const arms = await openBenchArms(
        "",
        "cached",
        memoryDistributions({ "gemma4-qat-mobile": MOBILE, gemma4: REGULAR }, opened),
        SIGNAL,
      );
      expect(opened).toEqual(["gemma4:cached", "gemma4-qat:cached", "gemma4-qat-mobile:cached"]);
      expect(arms.map(({ mount, model, quant, tap }) => ({ mount, model, quant, tap }))).toEqual([
        { mount: "gemma4", model: "e2b", quant: "bf16", tap: true },
        { mount: "gemma4-qat-mobile", model: "e2b", quant: "i4-fast", tap: true },
      ]);
      expect(arms.map(({ distribution }) => distribution.facts.directory)).toEqual([
        "models/karume-gemma4",
        "models/karume-gemma4-qat-mobile",
      ]);
    });

    it("are refused when no mirror is present", async () => {
      await expect(openBenchArms("", "direct", memoryDistributions({}), SIGNAL)).rejects.toThrow(
        new ParameterError(
          "arms",
          `the default is one arm per mount of ${MOUNTS} whose mirror is present, and none is`,
        ),
      );
    });
  });

  describe("as named", () => {
    it("open each mount once, in the order the arms first name them", async () => {
      const opened: string[] = [];
      const arms = await openBenchArms(
        "gemma4-qat/e2b,gemma4/e4b,gemma4-qat/e4b:tap-off",
        "direct",
        memoryDistributions(
          { gemma4: REGULAR, "gemma4-qat": QAT, "gemma4-qat-mobile": MOBILE },
          opened,
        ),
        SIGNAL,
      );
      expect(opened).toEqual(["gemma4-qat:direct", "gemma4:direct"]);
      expect(
        arms.map(({ mount, model, quant, tap }) => `${mount}/${model} ${quant} ${tap}`),
      ).toEqual(["gemma4-qat/e2b i4 true", "gemma4/e4b bf16 true", "gemma4-qat/e4b i4 false"]);
    });

    it("carry the preset, the label, the declared sampler and the options of their switches", async () => {
      const arms = await openBenchArms(
        "gemma4-qat-mobile/e2b@i4-fast:integer:ple-gpu:ahead8:split",
        "cached",
        memoryDistributions({ "gemma4-qat-mobile": MOBILE }),
        SIGNAL,
      );
      expect(arms).toEqual([
        {
          mount: "gemma4-qat-mobile",
          model: "e2b",
          quant: "i4-fast",
          quantLabel: "i4-fast label",
          isDefaultQuant: true,
          tap: true,
          split: true,
          // i4-fast declares the packed activations and the parallel GEMV.
          session: { integerGemv: true },
          pipeline: { pleResidency: "gpu", decodeAhead: 8 },
          declaredSampler: { temperature: 1, topK: 64, topP: 0.95 },
          distribution: {
            ref: expect.anything(),
            facts: {
              directory: "models/karume-gemma4-qat-mobile",
              manifestSha256: hex("f"),
              read: "directory",
            },
          },
        },
      ]);
    });

    it("refuse an arm the manifests refuse, naming the arm (resolveArm's refusals are the plan's test)", async () => {
      await expect(
        openBenchArms(
          "gemma4/e2b,gemma4/e9b",
          "cached",
          memoryDistributions({ gemma4: REGULAR, "gemma4-qat": QAT, "gemma4-qat-mobile": MOBILE }),
          SIGNAL,
        ),
      ).rejects.toThrow(
        new ParameterError(
          "arms",
          `arm 2 "gemma4/e9b": The gemma4 manifest has no model 'e9b'; its models are e2b, e4b.`,
        ),
      );
    });

    it("refuse a mount whose mirror is missing, naming the first arm that reads it", async () => {
      const opened: string[] = [];
      await expect(
        openBenchArms(
          "gemma4/e2b,gemma4-qat/e2b,gemma4-qat/e4b",
          "cached",
          memoryDistributions({ gemma4: REGULAR }, opened),
          SIGNAL,
        ),
      ).rejects.toThrow(
        new ParameterError(
          "arms",
          `arm 2 "gemma4-qat/e2b": Distribution mount gemma4-qat: its mirror is missing`,
        ),
      );
      expect(opened).toEqual(["gemma4:cached", "gemma4-qat:cached"]);
    });

    it("refuse an arm that does not fit the form before any distribution is read", async () => {
      const opened: string[] = [];
      await expect(
        openBenchArms(
          "gemma4/e2b,gemma4/e2b:nope",
          "cached",
          memoryDistributions({ gemma4: REGULAR }, opened),
          SIGNAL,
        ),
      ).rejects.toThrow(/^Parameter arms: arm 2 "gemma4\/e2b:nope": 'nope' is not a switch/);
      expect(opened).toEqual([]);
    });
  });
});

describe("the workload", () => {
  /** The settings of `given` (strings, as a URL or a command line gives them). */
  const settingsOf = (given: Readonly<Record<string, string>> = {}) =>
    benchSettingsOf(parseParameters(gemmaBench.parameters, given));

  it("defaults to 3 chats of 128 new tokens over 4 rounds, greedy, both prompts", () => {
    expect(settingsOf()).toEqual({
      chats: 3,
      newTokens: 128,
      rounds: 4,
      sampler: "greedy",
      prompts: ["few-ids", "many-ids"],
    });
  });

  it("takes the values given", () => {
    expect(
      settingsOf({
        chats: "2",
        newTokens: "8",
        rounds: "1",
        sampler: "distribution",
        prompts: "many-ids",
      }),
    ).toEqual({
      chats: 2,
      newTokens: 8,
      rounds: 1,
      sampler: "distribution",
      prompts: ["many-ids"],
    });
  });

  it("keeps the prompts in the set's order whatever order they were given in", () => {
    expect(settingsOf({ prompts: "many-ids,few-ids" }).prompts).toEqual(["few-ids", "many-ids"]);
  });

  it("needs a prompt", () => {
    expect(() => settingsOf({ prompts: "" })).toThrow(
      new ParameterError("prompts", "choose at least one prompt"),
    );
  });
});
