import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { REQUIRED_LIMIT_KEYS } from "../../../packages/runtime/src/gpu/acquire.ts";
import type { WanPrompt } from "../../../packages/models/wan.ts";
import type { WanPipelineConfig } from "../../../packages/models/src/wan/config.ts";
import { WAN22_TI2V_GENERATION } from "../../../packages/models/src/wan/descriptor.ts";
import ti2vReferences from "../../../packages/models/tests/fixtures/references/wan-ti2v.json" with {
  type: "json",
};
import {
  buildWanRequest,
  checkWanReference,
  chromeEnvironmentKey,
  formatSpans,
  formatWanDiagnostics,
  judgeWanLimits,
  resolveWanFormSize,
  summarizeWanDiagnostics,
  summarizeWanTimeline,
  WAN21_LAB,
  WAN22_LAB,
  WAN_SIZE_FROM_IMAGE,
  wanAttentionRowBlocks,
  type WanForm,
  wanFrameChoices,
  wanLargestValue,
  type WanLimits,
  wanMaxFramesWithin,
  wanReferenceCaseId,
  type WanResolvedRequest,
  wanRgbBytes,
  wanTokenCount,
} from "./wan-plan.ts";

const LANDSCAPE = { width: 832, height: 480 } as const;
const PORTRAIT = { width: 480, height: 832 } as const;

/** 束縛上限とバッファ上限だけを変えた limits（他の項目は判定に効かないので WebGPU の既定）。 */
const limitsWith = (binding: number, buffer = binding): WanLimits => ({
  maxBufferSize: buffer,
  maxStorageBufferBindingSize: binding,
  maxUniformBufferBindingSize: 65_536,
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 16_384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65_535,
});
/** B570（Deno）の束縛上限（ADR 0118 決定 6 の表の前提）。 */
const B570_BINDING = 2_147_483_644;

const CONFIG: WanPipelineConfig = {
  scheduler: { shift: 3 },
  defaults: { steps: 50, guidance: 5 },
};

const PROMPTS: readonly WanPrompt[] = [
  { name: "boxing-cats", role: "positive", prompt: "Two cats box.\n", normalized: "Two cats box." },
  { name: "ferret", role: "positive", prompt: "A ferret.", normalized: "A ferret." },
  { name: "negative", role: "negative", prompt: "blurry", normalized: "blurry" },
];

const FORM: WanForm = {
  prompt: "boxing-cats",
  negative: "",
  seed: "42",
  frames: "33",
  size: "832x480",
  steps: "",
  guidance: "",
  shift: "",
};

describe("wan token count", () => {
  it("matches the token lengths ADR 0118 measured for 33 and 81 frames, in both orientations", () => {
    assertEquals(wanTokenCount(WAN21_LAB, 33, LANDSCAPE), 14_040);
    assertEquals(wanTokenCount(WAN21_LAB, 81, LANDSCAPE), 32_760);
    assertEquals(wanTokenCount(WAN21_LAB, 81, PORTRAIT), 32_760);
    assertEquals(wanTokenCount(WAN21_LAB, 5, LANDSCAPE), 3_120);
  });

  it("rejects requests outside the accepted set instead of judging them", () => {
    for (const frames of [4, 34, 85, 1]) {
      assertThrows(() => wanTokenCount(WAN21_LAB, frames, LANDSCAPE), RangeError);
    }
    assertThrows(() => wanTokenCount(WAN21_LAB, 33, { width: 640, height: 480 }), RangeError);
  });

  it("offers every 4n+1 frame count from 5 to 81", () => {
    const choices = wanFrameChoices(WAN21_LAB);
    assertEquals(choices.length, 20);
    assertEquals([choices[0], choices[7], choices.at(-1)], [5, 33, 81]);
  });
});

describe("wan largest value", () => {
  it("is the DiT FFN intermediate for long clips (480 MiB at 33 frames, 1.09 GiB at 81)", () => {
    assertEquals(wanLargestValue(WAN21_LAB, 33, LANDSCAPE).bytes, 503_193_600);
    assertEquals(wanLargestValue(WAN21_LAB, 81, LANDSCAPE).bytes, 1_174_118_400);
    assertEquals(
      wanLargestValue(WAN21_LAB, 13, LANDSCAPE).what.startsWith("DiT の FFN 中間"),
      true,
    );
  });

  it("is the VAE intermediate for short clips, whatever the frame count", () => {
    for (const frames of [5, 9]) {
      const largest = wanLargestValue(WAN21_LAB, frames, LANDSCAPE);
      assertEquals(largest.bytes, 201_326_592, `${frames} frames`);
      assertEquals(largest.what.startsWith("VAE"), true);
    }
  });
});

describe("wan attention row blocks", () => {
  it("reproduces the row blocks of ADR 0118 decision 6 on the B570 binding limit", () => {
    const short = wanAttentionRowBlocks(WAN21_LAB, 33, LANDSCAPE, B570_BINDING);
    assertEquals(short.self, { bytesPerRow: 673_920, count: 5, blockBytes: 1_892_367_360 });
    const long = wanAttentionRowBlocks(WAN21_LAB, 81, LANDSCAPE, B570_BINDING);
    assertEquals(long.self, { bytesPerRow: 1_572_480, count: 24, blockBytes: 2_146_435_200 });
    assertEquals(long.cross.count, 1);
  });

  it("splits into more blocks under a smaller limit rather than failing", () => {
    const blocks = wanAttentionRowBlocks(WAN21_LAB, 33, LANDSCAPE, 1024 ** 3);
    assertEquals(blocks.self.count, 9);
    assertEquals((blocks.self.blockBytes ?? Infinity) <= 1024 ** 3, true);
  });

  it("reports a limit below one score row as unsplittable", () => {
    const blocks = wanAttentionRowBlocks(WAN21_LAB, 81, LANDSCAPE, 1_000_000);
    assertEquals(blocks.self, { bytesPerRow: 1_572_480 });
  });
});

describe("wan limits judgement", () => {
  const verdicts = (rows: ReturnType<typeof judgeWanLimits>) =>
    Object.fromEntries(rows.map((row) => [row.key, row.verdict]));

  it("passes 81 frames on the B570 limits and lists every requested limit", () => {
    const rows = judgeWanLimits(WAN21_LAB, limitsWith(B570_BINDING), 81, LANDSCAPE);
    assertEquals(rows.map((row) => row.key), [...REQUIRED_LIMIT_KEYS]);
    assertEquals(verdicts(rows).maxStorageBufferBindingSize, "ok");
    assertEquals(verdicts(rows).maxBufferSize, "ok");
    assertEquals(
      rows.filter((row) => row.verdict === "info").length,
      REQUIRED_LIMIT_KEYS.length - 2,
    );
    const binding = rows.find((row) => row.key === "maxStorageBufferBindingSize");
    assertEquals(binding?.required, 1_174_118_400);
    assertEquals(binding?.note.includes("行ブロック 24 枚"), true);
  });

  it("fails every frame count at the WebGPU default 128 MiB binding (the VAE alone needs 192 MiB)", () => {
    const limits = limitsWith(128 * 1024 ** 2, 256 * 1024 ** 2);
    for (const frames of wanFrameChoices(WAN21_LAB)) {
      assertEquals(
        verdicts(judgeWanLimits(WAN21_LAB, limits, frames, LANDSCAPE)).maxStorageBufferBindingSize,
        "short",
      );
    }
    assertEquals(wanMaxFramesWithin(WAN21_LAB, limits, LANDSCAPE), undefined);
  });

  it("allows 33 but not 81 frames at a 1 GiB binding, and names the largest frame count that fits", () => {
    const limits = limitsWith(1024 ** 3);
    assertEquals(
      verdicts(judgeWanLimits(WAN21_LAB, limits, 33, LANDSCAPE)).maxStorageBufferBindingSize,
      "ok",
    );
    assertEquals(
      verdicts(judgeWanLimits(WAN21_LAB, limits, 81, LANDSCAPE)).maxStorageBufferBindingSize,
      "short",
    );
    // 73 フレーム: S = 19 × 1560 = 29,640 → FFN 中間 1,062,297,600 B ≤ 1 GiB。77 は 1,118,208,000 B で超える
    assertEquals(wanMaxFramesWithin(WAN21_LAB, limits, LANDSCAPE), 73);
    assertEquals(wanMaxFramesWithin(WAN21_LAB, limitsWith(B570_BINDING), LANDSCAPE), 81);
  });

  it("judges maxBufferSize on its own (a large binding limit does not hide a small buffer limit)", () => {
    const rows = judgeWanLimits(
      WAN21_LAB,
      limitsWith(B570_BINDING, 512 * 1024 ** 2),
      81,
      LANDSCAPE,
    );
    assertEquals(verdicts(rows).maxStorageBufferBindingSize, "ok");
    assertEquals(verdicts(rows).maxBufferSize, "short");
  });
});

describe("wan request building", () => {
  it("sends only the fields that were filled and resolves the rest from the pipeline config", () => {
    const { request, resolved } = buildWanRequest(FORM, PROMPTS, CONFIG);
    assertEquals(request, {
      prompt: "Two cats box.\n",
      seed: 42,
      frames: 33,
      width: 832,
      height: 480,
    });
    assertEquals(resolved, {
      prompt: "boxing-cats",
      negative: "negative",
      seed: 42,
      steps: 50,
      guidance: 5,
      shift: 3,
      frames: 33,
      width: 832,
      height: 480,
    });
  });

  it("passes the typed knobs and the chosen negative through", () => {
    const { request, resolved } = buildWanRequest(
      { ...FORM, negative: "ferret", steps: "2", guidance: "4.5", shift: "5", size: "480x832" },
      PROMPTS,
      CONFIG,
    );
    assertEquals(request.negativePrompt, "A ferret.");
    assertEquals([request.steps, request.guidance, request.shift], [2, 4.5, 5]);
    assertEquals([request.width, request.height], [480, 832]);
    assertEquals([resolved.negative, resolved.steps, resolved.guidance], ["ferret", 2, 4.5]);
  });

  it("resolves no negative when guidance 1 turns CFG off", () => {
    assertEquals(
      buildWanRequest({ ...FORM, guidance: "1" }, PROMPTS, CONFIG).resolved.negative,
      undefined,
    );
  });

  it("fails loudly on unknown prompt names and unreadable numbers", () => {
    assertThrows(
      () => buildWanRequest({ ...FORM, prompt: "dog" }, PROMPTS, CONFIG),
      Error,
      "埋め込み資産に無い",
    );
    assertThrows(
      () => buildWanRequest({ ...FORM, steps: "ten" }, PROMPTS, CONFIG),
      Error,
      "数でない",
    );
    assertThrows(
      () => buildWanRequest({ ...FORM, seed: " " }, PROMPTS, CONFIG),
      Error,
      "seed が空欄",
    );
    assertThrows(
      () => buildWanRequest({ ...FORM, size: "832*480" }, PROMPTS, CONFIG),
      Error,
      "WxH",
    );
  });

  describe("with the size taken from the condition image", () => {
    const IMAGE_SIZE = { width: 704, height: 1280 } as const;

    it("leaves width and height out of the request and resolves them to the size chosen from the image", () => {
      const { request, resolved } = buildWanRequest(
        { ...FORM, size: WAN_SIZE_FROM_IMAGE },
        PROMPTS,
        CONFIG,
        IMAGE_SIZE,
      );
      assertEquals(request, { prompt: "Two cats box.\n", seed: 42, frames: 33 });
      assertEquals([resolved.width, resolved.height], [704, 1280]);
    });

    it("still sends an explicitly chosen size when an image is given", () => {
      const { request, resolved } = buildWanRequest(FORM, PROMPTS, CONFIG, IMAGE_SIZE);
      assertEquals([request.width, request.height], [832, 480]);
      assertEquals([resolved.width, resolved.height], [832, 480]);
    });

    it("refuses the image size when no image has been chosen", () => {
      assertThrows(
        () => buildWanRequest({ ...FORM, size: WAN_SIZE_FROM_IMAGE }, PROMPTS, CONFIG),
        Error,
        "条件画像が要る",
      );
      assertThrows(
        () => resolveWanFormSize(WAN_SIZE_FROM_IMAGE, undefined),
        Error,
        "条件画像が要る",
      );
    });

    it("reads a WxH choice as is, whatever the image size", () => {
      assertEquals(resolveWanFormSize("1280x704", IMAGE_SIZE), { width: 1280, height: 704 });
      assertEquals(resolveWanFormSize(WAN_SIZE_FROM_IMAGE, IMAGE_SIZE), IMAGE_SIZE);
    });
  });
});

describe("wan timeline", () => {
  it("attributes each step and tile to the time since the previous one (the first includes the stage start)", () => {
    const timeline = summarizeWanTimeline([
      { kind: "stage", component: "transformer", at: "start", ms: 100 },
      { kind: "step", step: 1, ms: 1100 },
      { kind: "step", step: 2, ms: 1600 },
      { kind: "stage", component: "transformer", at: "end", ms: 1700 },
      { kind: "stage", component: "vae_decoder", at: "start", ms: 1800 },
      { kind: "tile", tile: 1, ms: 2100 },
      { kind: "tile", tile: 2, ms: 2200 },
      { kind: "stage", component: "vae_decoder", at: "end", ms: 2250 },
    ]);
    assertEquals(timeline, {
      stageMs: { transformer: 1600, vae_decoder: 450 },
      stepMs: [1000, 500],
      tileMs: [300, 100],
    });
  });

  it("times the image-to-video encoder stage before the transformer without counting it as a step", () => {
    const timeline = summarizeWanTimeline([
      { kind: "stage", component: "vae_encoder", at: "start", ms: 0 },
      { kind: "stage", component: "vae_encoder", at: "end", ms: 500 },
      { kind: "stage", component: "transformer", at: "start", ms: 600 },
      { kind: "step", step: 1, ms: 1600 },
      { kind: "stage", component: "transformer", at: "end", ms: 1700 },
      { kind: "stage", component: "vae_decoder", at: "start", ms: 1800 },
      { kind: "tile", tile: 1, ms: 2100 },
      { kind: "stage", component: "vae_decoder", at: "end", ms: 2150 },
    ]);
    assertEquals(timeline, {
      stageMs: { vae_encoder: 500, transformer: 1100, vae_decoder: 350 },
      stepMs: [1000],
      tileMs: [300],
    });
  });

  it("refuses a step or a tile inside the encoder stage", () => {
    assertThrows(
      () =>
        summarizeWanTimeline([
          { kind: "stage", component: "vae_encoder", at: "start", ms: 0 },
          { kind: "step", step: 1, ms: 1 },
        ]),
      Error,
      "段 transformer",
    );
    assertThrows(
      () =>
        summarizeWanTimeline([
          { kind: "stage", component: "vae_encoder", at: "start", ms: 0 },
          { kind: "tile", tile: 1, ms: 1 },
        ]),
      Error,
      "段 vae_decoder",
    );
  });

  it("refuses the transformer stage starting before the encoder stage ends", () => {
    assertThrows(
      () =>
        summarizeWanTimeline([
          { kind: "stage", component: "vae_encoder", at: "start", ms: 0 },
          { kind: "stage", component: "transformer", at: "start", ms: 1 },
        ]),
      Error,
      "閉じる前に",
    );
  });

  it("leaves out the stage that never ended", () => {
    const timeline = summarizeWanTimeline([
      { kind: "stage", component: "transformer", at: "start", ms: 0 },
      { kind: "step", step: 1, ms: 10 },
    ]);
    assertEquals(timeline.stageMs, {});
    assertEquals(timeline.stepMs, [10]);
  });

  it("refuses events out of the pipeline's order", () => {
    assertThrows(() => summarizeWanTimeline([{ kind: "step", step: 1, ms: 1 }]));
    assertThrows(() =>
      summarizeWanTimeline([
        { kind: "stage", component: "transformer", at: "start", ms: 0 },
        { kind: "step", step: 2, ms: 1 },
      ])
    );
    assertThrows(() =>
      summarizeWanTimeline([
        { kind: "stage", component: "transformer", at: "start", ms: 0 },
        { kind: "tile", tile: 1, ms: 1 },
      ])
    );
    assertThrows(() =>
      summarizeWanTimeline([
        { kind: "stage", component: "transformer", at: "start", ms: 0 },
        { kind: "stage", component: "vae_decoder", at: "start", ms: 1 },
      ])
    );
  });

  it("summarizes spans as the first one and the median and maximum of the rest", () => {
    assertEquals(formatSpans([]), "—");
    assertEquals(formatSpans([12_300]), "1 回目 12.3 s");
    assertEquals(
      formatSpans([12_300, 10_000, 10_500, 9_000, 10_200]),
      "1 回目 12.3 s · 2 回目以降 中央値 10.1 s / 最大 10.5 s（4 回）",
    );
  });
});

describe("wan rgb bytes", () => {
  it("concatenates every frame's RGB in frame order and drops alpha", () => {
    const frames = [
      new Uint8ClampedArray([1, 2, 3, 255, 4, 5, 6, 255]),
      new Uint8ClampedArray([7, 8, 9, 255, 10, 11, 12, 255]),
    ];
    assertEquals(
      wanRgbBytes(frames, 2),
      new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    );
  });

  it("refuses a frame whose size does not match the plane", () => {
    assertThrows(() => wanRgbBytes([new Uint8ClampedArray(4)], 2), Error, "RGBA");
  });
});

describe("chrome environment key", () => {
  it("uses vendor and architecture when Chrome leaves the description empty", () => {
    assertEquals(
      chromeEnvironmentKey({ vendor: "nvidia", architecture: "blackwell", description: "" }),
      "chrome-nvidia-blackwell",
    );
    assertEquals(
      chromeEnvironmentKey({ vendor: "apple", architecture: "metal-3", description: "" }),
      "chrome-apple-metal-3",
    );
  });

  it("uses the description when the developer features flag fills it, without trademark marks", () => {
    assertEquals(
      chromeEnvironmentKey({
        vendor: "intel",
        architecture: "",
        description: "Intel(R) Graphics (BMG G21)",
      }),
      "chrome-intel-graphics-bmg-g21",
    );
  });

  it("refuses an adapter that names nothing", () => {
    assertThrows(() => chromeEnvironmentKey({ vendor: "", architecture: "", description: "" }));
  });
});

describe("wan reference cases", () => {
  const BASE: WanResolvedRequest = {
    prompt: "boxing-cats",
    negative: "negative",
    seed: 42,
    steps: 2,
    guidance: 5,
    shift: 3,
    frames: 33,
    width: 832,
    height: 480,
  };

  it("names the e2e cases the sha256 rows are kept for", () => {
    assertEquals(
      wanReferenceCaseId(WAN21_LAB, BASE, CONFIG, "negative", "f16"),
      "2step-seed-boxing-cats-seed42",
    );
    assertEquals(
      wanReferenceCaseId(WAN21_LAB, { ...BASE, steps: 50 }, CONFIG, "negative", "f16"),
      "50step-boxing-cats-seed42",
    );
    assertEquals(
      wanReferenceCaseId(WAN21_LAB, { ...BASE, steps: 50, frames: 81 }, CONFIG, "negative", "f16"),
      "50step-boxing-cats-seed42-81f",
    );
  });

  it("puts the quant that ran in front of the id for the int8 quants the e2e keeps rows for", () => {
    assertEquals(
      wanReferenceCaseId(WAN21_LAB, BASE, CONFIG, "negative", "f16+dit8"),
      "f16+dit8-2step-seed-boxing-cats-seed42",
    );
    assertEquals(
      wanReferenceCaseId(WAN21_LAB, BASE, CONFIG, "negative", "f16+dit8-a8-attn8-s16"),
      "f16+dit8-a8-attn8-s16-2step-seed-boxing-cats-seed42",
    );
    assertEquals(
      wanReferenceCaseId(
        WAN21_LAB,
        { ...BASE, steps: 50 },
        CONFIG,
        "negative",
        "f16+dit8-a8-attn8-s16",
      ),
      "f16+dit8-a8-attn8-s16-50step-boxing-cats-seed42",
    );
    assertEquals(
      wanReferenceCaseId(
        WAN21_LAB,
        { ...BASE, steps: 50, frames: 81 },
        CONFIG,
        "negative",
        "f16+dit8-a8-attn8-s16",
      ),
      "f16+dit8-a8-attn8-s16-50step-boxing-cats-seed42-81f",
    );
  });

  it("names no case for a quant the e2e keeps no row for", () => {
    assertEquals(
      wanReferenceCaseId(WAN21_LAB, { ...BASE, steps: 50 }, CONFIG, "negative", "f16+dit8"),
      undefined,
    );
    assertEquals(wanReferenceCaseId(WAN21_LAB, BASE, CONFIG, "negative", "f16+other"), undefined);
  });

  it("names no case when any condition differs from the e2e case", () => {
    for (
      const changed of [
        { seed: 43 },
        { prompt: "ferret" },
        { negative: "ferret" },
        { guidance: 4 },
        { shift: 5 },
        { width: 480, height: 832 },
        { steps: 3 },
        { frames: 81 },
      ]
    ) {
      assertEquals(
        wanReferenceCaseId(WAN21_LAB, { ...BASE, ...changed }, CONFIG, "negative", "f16"),
        undefined,
        JSON.stringify(changed),
      );
    }
  });

  it("compares only with the row of this environment key", () => {
    const references = { cases: { "case-a": { "deno-intel-graphics-bmg-g21": "a".repeat(64) } } };
    assertEquals(
      checkWanReference(references, undefined, "chrome-nvidia-blackwell", "a".repeat(64)),
      {
        kind: "no-case",
      },
    );
    assertEquals(
      checkWanReference(references, "case-a", "chrome-nvidia-blackwell", "a".repeat(64)),
      {
        kind: "no-row",
        caseId: "case-a",
        key: "chrome-nvidia-blackwell",
      },
    );
    assertEquals(
      checkWanReference(references, "case-b", "deno-intel-graphics-bmg-g21", "a".repeat(64)).kind,
      "no-row",
    );
    assertEquals(
      checkWanReference(references, "case-a", "deno-intel-graphics-bmg-g21", "a".repeat(64)).kind,
      "match",
    );
    assertEquals(
      checkWanReference(references, "case-a", "deno-intel-graphics-bmg-g21", "b".repeat(64)),
      {
        kind: "mismatch",
        caseId: "case-a",
        key: "deno-intel-graphics-bmg-g21",
        expected: "a".repeat(64),
      },
    );
  });
});

describe("wan diagnostics", () => {
  it("formats a session's allocations and submit statistics like the e2e observation line", () => {
    const summary = summarizeWanDiagnostics({
      weights: { allocatedBytes: 2.645 * 1024 ** 3 },
      planBacking: { residentBytes: 3 * 1024 ** 3, inputBytes: 0.517 * 1024 ** 3 },
      geometryProfile: "default",
      submit: {
        submitCount: 184_692,
        dispatchCount: 2_000_000,
        chunkBudget: { maxWindowMeanMs: 268.34, overBudgetChunks: 0 },
      },
    });
    assertEquals(
      formatWanDiagnostics("transformer", summary),
      "transformer: 重み 2.645 GiB・backing 3.517 GiB・幾何 default・submit 184,692 本・窓平均の最大 268.3 ms・予算超過 0 本",
    );
  });
});

describe("wan2.2 lab generation", () => {
  const WIDE = { width: 1280, height: 704 } as const;
  const TALL = { width: 704, height: 1280 } as const;

  it("uses the product descriptor as is, up to the official 121 frames", () => {
    assertStrictEquals(WAN22_LAB.descriptor, WAN22_TI2V_GENERATION);
    assertEquals(WAN22_LAB.descriptor.maxFrames, 121);
  });

  it("counts 7,920 tokens at 1280x704x33 and 27,280 at 121 frames (32 pixels per token)", () => {
    assertEquals(wanTokenCount(WAN22_LAB, 33, WIDE), 7_920);
    assertEquals(wanTokenCount(WAN22_LAB, 49, WIDE), 11_440);
    assertEquals(wanTokenCount(WAN22_LAB, 121, WIDE), 27_280);
    assertEquals(wanTokenCount(WAN22_LAB, 121, TALL), 27_280);
    assertEquals(wanTokenCount(WAN22_LAB, 5, WIDE), 1_760);
  });

  it("rejects frame counts beyond 121 and the sizes Wan2.2 does not accept", () => {
    for (const frames of [125, 4, 34]) {
      assertThrows(() => wanTokenCount(WAN22_LAB, frames, WIDE), RangeError);
    }
    assertThrows(() => wanTokenCount(WAN22_LAB, 33, { width: 832, height: 480 }), RangeError);
  });

  it("offers every 4n+1 frame count from 5 to 121", () => {
    const choices = wanFrameChoices(WAN22_LAB);
    assertEquals(choices.length, 30);
    assertEquals([choices[0], choices[7], choices[11], choices.at(-1)], [5, 33, 49, 121]);
  });

  it("takes the FFN intermediate [1,S,14336] for long clips and the 192 MiB VAE value for short ones", () => {
    assertEquals(wanLargestValue(WAN22_LAB, 33, WIDE).bytes, 454_164_480);
    assertEquals(wanLargestValue(WAN22_LAB, 121, WIDE), {
      bytes: 1_564_344_320,
      what: "DiT の FFN 中間 [1,27280,14336] f32",
    });
    assertEquals(wanLargestValue(WAN22_LAB, 13, WIDE).bytes, 201_850_880);
    for (const frames of [5, 9]) {
      assertEquals(wanLargestValue(WAN22_LAB, frames, WIDE), {
        bytes: 201_326_592,
        what: "VAE（next）の中間 [512,6,128,128] f32",
      });
    }
  });

  it("splits the 24-head scores into the row blocks ADR 0121 stage 2 measured on the B570", () => {
    // S = 7,920 は行ブロック 3 枚（段 2 の結果）。121 フレームは 34 枚・1 枚 1.96 GiB（決定 8 の「最大 1.96 GiB」）
    const short = wanAttentionRowBlocks(WAN22_LAB, 33, WIDE, B570_BINDING);
    assertEquals(short.self, { bytesPerRow: 760_320, count: 3, blockBytes: 2_007_244_800 });
    const long = wanAttentionRowBlocks(WAN22_LAB, 121, WIDE, B570_BINDING);
    assertEquals(long.self, { bytesPerRow: 2_618_880, count: 34, blockBytes: 2_102_960_640 });
    assertEquals(long.cross, { bytesPerRow: 49_152, count: 1, blockBytes: 1_340_866_560 });
  });

  it("passes 121 frames on the B570 limits and stops at 81 frames under a 1 GiB binding", () => {
    const binding = (limits: WanLimits, frames: number) =>
      judgeWanLimits(WAN22_LAB, limits, frames, WIDE).find((row) =>
        row.key === "maxStorageBufferBindingSize"
      )?.verdict;
    assertEquals(binding(limitsWith(B570_BINDING), 121), "ok");
    assertEquals(wanMaxFramesWithin(WAN22_LAB, limitsWith(B570_BINDING), WIDE), 121);
    // 81 フレーム: S = 18,480 → 1,059,717,120 B ≤ 1 GiB。85 は 1,110,179,840 B で超える
    assertEquals(binding(limitsWith(1024 ** 3), 85), "short");
    assertEquals(wanMaxFramesWithin(WAN22_LAB, limitsWith(1024 ** 3), WIDE), 81);
  });
});

describe("wan2.2 reference cases", () => {
  const CONFIG_22: WanPipelineConfig = {
    scheduler: { shift: 5 },
    defaults: { steps: 50, guidance: 5 },
  };
  const BASE_22: WanResolvedRequest = {
    prompt: "boxing-cats",
    negative: "negative",
    seed: 42,
    steps: 2,
    guidance: 5,
    shift: 5,
    frames: 17,
    width: 1280,
    height: 704,
  };
  const caseId = (changed: Partial<WanResolvedRequest>, quant: string) =>
    wanReferenceCaseId(WAN22_LAB, { ...BASE_22, ...changed }, CONFIG_22, "negative", quant);

  it("names the 2-step seed cases of the reference quant in both sizes, as the fixture spells them", () => {
    const ids = [caseId({}, "f16+dit8"), caseId({ width: 704, height: 1280 }, "f16+dit8")];
    assertEquals(ids, [
      "f16+dit8-2step-boxing-cats-seed42-1280x704-17f-shift5",
      "f16+dit8-2step-boxing-cats-seed42-704x1280-17f-shift5",
    ]);
    for (const id of ids) assertEquals(Object.hasOwn(ti2vReferences.cases, id ?? ""), true, id);
  });

  it("names the 50-step cases of both quants at 1280x704 and 33 frames", () => {
    assertEquals(
      caseId({ steps: 50, frames: 33 }, "f16+dit8"),
      "f16+dit8-50step-boxing-cats-seed42-1280x704-33f-shift5",
    );
    assertEquals(
      caseId({ steps: 50, frames: 33 }, "f16+dit8-a8-attn8-s16"),
      "f16+dit8-a8-attn8-s16-50step-boxing-cats-seed42-1280x704-33f-shift5",
    );
  });

  it("names the opt-in 2-step case of the reference quant at 1280x704 and 121 frames", () => {
    assertEquals(
      caseId({ frames: 121 }, "f16+dit8"),
      "f16+dit8-2step-boxing-cats-seed42-1280x704-121f-shift5",
    );
  });

  it("names no case for a quant, size, or condition the e2e keeps no row for", () => {
    for (
      const [changed, quant] of [
        [{}, "f16+dit8-a8-attn8-s16"],
        [{}, "f16"],
        [{ steps: 50, frames: 33, width: 704, height: 1280 }, "f16+dit8"],
        [{ frames: 33 }, "f16+dit8"],
        [{ frames: 121 }, "f16+dit8-a8-attn8-s16"],
        [{ frames: 121, width: 704, height: 1280 }, "f16+dit8"],
        [{ steps: 50, frames: 121 }, "f16+dit8"],
        [{ steps: 50 }, "f16+dit8"],
        [{ seed: 43 }, "f16+dit8"],
        [{ guidance: 4 }, "f16+dit8"],
        [{ shift: 3 }, "f16+dit8"],
        [{ negative: "ferret" }, "f16+dit8"],
      ] satisfies [Partial<WanResolvedRequest>, string][]
    ) {
      assertEquals(caseId(changed, quant), undefined, `${JSON.stringify(changed)} ${quant}`);
    }
  });
});
