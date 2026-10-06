// Wan2.1 のパイプラインの GPU 不要の部分（`src/wan/`）— 入力の門（資産の経路と GPU の経路）・`pipelineConfig` の門・
// 家族 admission のグラフ宣言の門（DiT と umT5）・構築の入口（経路の綴り・umT5 の宣言・中断）・既定の negative の
// 出所・テキスト埋め込み資産の検査・潜在の逆正規化・フレームの RGBA 化・乱数・模擬 Session で回す `generate`
// （text 段の順序と畳み方・後始末と非有限の門・中断）。実 GPU の通しは e2e_wan_pipeline_test.ts。

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertNotEquals,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { parseManifest } from "@karume/hub";
import type { CodecName, RunInputs, Tensor } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import {
  type WanGenerateRequest,
  WanPipeline,
  type WanPipelineOptions,
} from "../src/wan/pipeline.ts";
import { planWanGeneration, planWanGpuGeneration, WAN21_FAMILY } from "../src/wan/family.ts";
import { ditContract, ditInputs, wanDitPatch } from "../src/wan/dit-loop.ts";
import { umt5Contract, WAN_DEFAULT_NEGATIVE_PROMPT } from "../src/wan/text-stage.ts";
import {
  WAN21_GENERATION,
  WAN22_TI2V_GENERATION,
  type WanDitInputForm,
  type WanGenerationDescriptor,
} from "../src/wan/descriptor.ts";
import { planWanRequest, type PromptGate } from "../src/wan/plan.ts";
import {
  assertWanVaeMatchesGeneration,
  assertWanVaeTilesCover,
  planWanGenerationTiles,
  wanSpatialCompression,
} from "../src/wan/tile-decode.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import { PromptCleanError } from "../src/wan/text/prompt-clean.ts";
import { wanParityCase, wanParityCases, wanParityEncoder } from "./helpers/wan-parity-encoder.ts";
import {
  declaredContainer,
  parseIrDeclarationValue,
  partAssets,
  tensorlessContainer,
  writeContainer,
} from "./helpers/container-fixture.ts";
import { readFileIfPresent, readTextIfPresent } from "./helpers/read-if-present.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { WAN_UNIPC_CONFIG, wanUniPcSchedule } from "../src/wan/scheduler.ts";
import { WanVaeChunkError, wanVaeChunkLayout } from "../src/wan/vae-chunks.ts";
import type { WanRopeBase } from "../src/wan/dit-rope.ts";
import { type StubDim, stubModel } from "./helpers/stub-model.ts";
import {
  findWanTextEmbedding,
  padWanTextEmbedding,
  parseWanTextEmbeds,
  WAN_TEXT_EMBEDS_METADATA_KEY,
  type WanTextEmbeds,
  WanTextEmbedsError,
} from "../src/wan/text-embeds.ts";
import { denormalizeWanLatents, WAN_LATENTS_MEAN, WAN_LATENTS_STD } from "../src/wan/latents.ts";
import { wanFrameToRgba } from "../src/wan/frames.ts";
import { WanRandn } from "../src/wan/random.ts";
import { parseWanPipelineConfig, type WanPipelineConfig } from "../src/wan/config.ts";

/** 埋め込み資産の 1 行（テスト用の小さい幅）。 */
type Row = {
  readonly name: string;
  readonly role: string;
  readonly prompt: string;
  readonly normalized: string;
  readonly tokens: number;
};

const WIDTH = 3;

const ROWS: readonly Row[] = [
  { name: "cats", role: "positive", prompt: "Two cats.", normalized: "Two cats.", tokens: 2 },
  {
    name: "ferret",
    role: "positive",
    prompt: "\nA ferret,\nleaps.\n",
    normalized: "A ferret, leaps.",
    tokens: 3,
  },
  {
    name: "negative",
    role: "negative",
    prompt: "色调艳丽，过曝",
    normalized: "色调艳丽,过曝",
    tokens: 1,
  },
];

/**
 * 埋め込み資産のバイト列（recipe `wan/text_embeds.py` と同じ形 — メタはキー 1 つに JSON）。値は行ごとに
 * `行番号 + 列 / 10` で、取り違えが値に出る。
 */
const buildAsset = (
  rows: readonly Row[],
  options: {
    readonly metadata?: Record<string, string>;
    readonly tensors?: readonly { name: string; shape: number[] }[];
  } = {},
): ArrayBuffer => {
  const tensors = options.tensors ??
    rows.map((row) => ({ name: row.name, shape: [row.tokens, WIDTH] }));
  const header: Record<string, unknown> = {
    __metadata__: options.metadata ??
      { [WAN_TEXT_EMBEDS_METADATA_KEY]: JSON.stringify({ prompts: rows, source: {} }) },
  };
  const payloads: Float32Array[] = [];
  let offset = 0;
  tensors.forEach(({ name, shape }, index) => {
    const data = new Float32Array(shape[0] * shape[1]).map((_, at) => index + (at % WIDTH) / 10);
    header[name] = { dtype: "F32", shape, data_offsets: [offset, offset + data.byteLength] };
    offset += data.byteLength;
    payloads.push(data);
  });
  const json = new TextEncoder().encode(JSON.stringify(header));
  const headerLength = json.length + ((8 - (json.length % 8)) % 8);
  const buffer = new ArrayBuffer(8 + headerLength + offset);
  const bytes = new Uint8Array(buffer);
  new DataView(buffer).setBigUint64(0, BigInt(headerLength), true);
  bytes.set(json, 8);
  bytes.fill(0x20, 8 + json.length, 8 + headerLength);
  let cursor = 8 + headerLength;
  for (const data of payloads) {
    bytes.set(new Uint8Array(data.buffer), cursor);
    cursor += data.byteLength;
  }
  return buffer;
};

const EMBEDS: WanTextEmbeds = parseWanTextEmbeds(buildAsset(ROWS));
/** 配布形の VAE の chunk グラフの幾何（潜在タイル 32・縮尺 8 — `8t / t`・出口は RGB の 3）。 */
const LAYOUT = { latentChannels: 16, tile: 32, sampleTile: 256, sampleChannels: 3 };
/** 配布形の `pipelineConfig`（recipe `wan/distribution.py` の `WAN_PIPELINE_CONFIG` — 参照の設定）。 */
const CONFIG: WanPipelineConfig = { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } };
const plan = (request: Partial<WanGenerateRequest>) =>
  planWanGeneration(
    { prompt: "Two cats.", ...request },
    EMBEDS,
    LAYOUT,
    CONFIG,
    WAN21_FAMILY.generation,
    WAN21_FAMILY.owner,
  );

describe("テキスト埋め込み資産", () => {
  it("メタの並びのまま行を返し、行は [tokens, width] の f32", () => {
    assertEquals(EMBEDS.width, WIDTH);
    assertEquals(EMBEDS.entries.map((entry) => [entry.name, entry.tokens]), [
      ["cats", 2],
      ["ferret", 3],
      ["negative", 1],
    ]);
    assertEquals(
      [...EMBEDS.entries[1].data],
      [1, 1.1, 1.2, 1, 1.1, 1.2, 1, 1.1, 1.2].map(Math.fround),
    );
  });

  it("原文と正規化後の文字列のどちらでも同じ行を引き、集合の外は引けない", () => {
    assertEquals(findWanTextEmbedding(EMBEDS, "\nA ferret,\nleaps.\n")?.name, "ferret");
    assertEquals(findWanTextEmbedding(EMBEDS, "A ferret, leaps.")?.name, "ferret");
    assertEquals(findWanTextEmbedding(EMBEDS, "A ferret, leaps"), undefined);
    assertEquals(findWanTextEmbedding(EMBEDS, "ferret"), undefined, "名前は受理集合ではない");
  });

  it("文脈の行数までゼロで埋める（有効長の後ろは厳密にゼロ）", () => {
    const padded = padWanTextEmbedding(EMBEDS.entries[0], 4, WIDTH);
    assertEquals([...padded.subarray(0, 6)], [...EMBEDS.entries[0].data]);
    assertEquals([...padded.subarray(6)], [0, 0, 0, 0, 0, 0]);
    assertThrows(
      () => padWanTextEmbedding(EMBEDS.entries[1], 2, WIDTH),
      WanTextEmbedsError,
      "有効長",
    );
    assertThrows(
      () => padWanTextEmbedding(EMBEDS.entries[1], 4, WIDTH + 1),
      WanTextEmbedsError,
      "幅",
    );
  });

  it("メタとテンソルの対応が崩れた資産は読まない", () => {
    const cases: readonly [string, ArrayBuffer][] = [
      ["メタのキー", buildAsset(ROWS, { metadata: { other: "{}" } })],
      [
        "メタのキー",
        buildAsset(ROWS, {
          metadata: { [WAN_TEXT_EMBEDS_METADATA_KEY]: JSON.stringify({ prompts: ROWS }), x: "1" },
        }),
      ],
      ["JSON", buildAsset(ROWS, { metadata: { [WAN_TEXT_EMBEDS_METADATA_KEY]: "{" } })],
      [
        "'cats' が F32 [2, width] でない",
        buildAsset(ROWS, {
          tensors: [{ name: "cats", shape: [3, WIDTH] }, { name: "ferret", shape: [3, WIDTH] }, {
            name: "negative",
            shape: [1, WIDTH],
          }],
        }),
      ],
      [
        "幅",
        buildAsset(ROWS, {
          tensors: [
            { name: "cats", shape: [2, WIDTH] },
            { name: "ferret", shape: [3, WIDTH + 1] },
            { name: "negative", shape: [1, WIDTH] },
          ],
        }),
      ],
      [
        "テンソル 'negative' が無い",
        buildAsset(ROWS, {
          tensors: [{ name: "cats", shape: [2, WIDTH] }, { name: "ferret", shape: [3, WIDTH] }],
        }),
      ],
      [
        "メタに無いテンソル",
        buildAsset(ROWS, {
          tensors: [...ROWS.map((row) => ({ name: row.name, shape: [row.tokens, WIDTH] })), {
            name: "extra",
            shape: [1, WIDTH],
          }],
        }),
      ],
      ["役割", buildAsset([{ ...ROWS[0], role: "neutral" }])],
      ["重複", buildAsset([ROWS[0], ROWS[0]], { tensors: [{ name: "cats", shape: [2, WIDTH] }] })],
      ["行が決まらない", buildAsset([ROWS[0], { ...ROWS[1], normalized: "Two cats." }])],
    ];
    for (const [message, asset] of cases) {
      assertThrows(() => parseWanTextEmbeds(asset), Error, message);
    }
  });
});

describe("planWanGeneration（generate の入口の門）", () => {
  it("省いた欄は参照の設定（50 step・guide 5.0・shift 3.0・832×480・33 フレーム・seed 0）", () => {
    const resolved = plan({});
    assertEquals(
      [
        resolved.steps,
        resolved.guidance,
        resolved.shift,
        resolved.frames,
        resolved.width,
        resolved.height,
      ],
      [50, 5, 3, 33, 832, 480],
    );
    assertEquals(resolved.latentShape, [16, 9, 60, 104]);
    assertEquals(resolved.initial, { kind: "seed", seed: 0 });
    assertEquals(resolved.positive.name, "cats");
    assertEquals(
      resolved.negative?.name,
      "negative",
      "negativePrompt の既定は資産の negative の行",
    );
  });

  it("縦長は潜在の軸を入れ替えずに組む", () => {
    assertEquals(plan({ width: 480, height: 832, frames: 33 }).latentShape, [16, 9, 104, 60]);
    assertEquals(plan({ frames: 5 }).latentShape, [16, 2, 60, 104]);
  });

  it("フレーム数は 4n+1 の 5〜81 だけを受け、85（上限超え）/ 34（4n+1 でない）は範囲を言って拒む", () => {
    const accepted = Array.from({ length: 20 }, (_, index) => 5 + 4 * index);
    assertEquals(accepted.at(-1), 81);
    assertEquals(accepted.map((frames) => plan({ frames }).frames), accepted);
    assertEquals(plan({ frames: 81 }).latentShape, [16, 21, 60, 104]);
    for (const frames of [85, 34]) {
      const error = assertThrows(() => plan({ frames }), ModelInputError, "4n+1 の 5〜81");
      assert(!error.message.includes("段"), `文言に段の番号を出さない: ${error.message}`);
    }
  });

  it("プロンプトは原文でも正規化後でも受け、集合の外は ModelInputError", () => {
    assertEquals(plan({ prompt: "A ferret, leaps." }).positive.name, "ferret");
    assertEquals(plan({ negativePrompt: "色调艳丽，过曝" }).negative?.name, "negative");
    assertEquals(plan({ negativePrompt: "A ferret, leaps." }).negative?.name, "ferret");
    assertThrows(
      () => plan({ prompt: "A dog." }),
      ModelInputError,
      "prompt がテキスト埋め込み資産の集合に無い",
    );
    assertThrows(() => plan({ negativePrompt: "" }), ModelInputError, "negativePrompt");
  });

  it("guidance 1 は uncond を回さず、negativePrompt を渡すと効かないので拒む", () => {
    assertEquals(plan({ guidance: 1 }).negative, undefined);
    assertThrows(
      () => plan({ guidance: 1, negativePrompt: "色调艳丽,过曝" }),
      ModelInputError,
      "効かない",
    );
    for (const guidance of [0.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      assertThrows(() => plan({ guidance }), ModelInputError, "guidance");
    }
  });

  it("negative の行が 1 本でない資産で negativePrompt を省くと拒む", () => {
    const embeds = parseWanTextEmbeds(buildAsset(ROWS.slice(0, 2)));
    assertThrows(
      () =>
        planWanGeneration(
          { prompt: "Two cats." },
          embeds,
          LAYOUT,
          CONFIG,
          WAN21_FAMILY.generation,
          WAN21_FAMILY.owner,
        ),
      ModelInputError,
      "negative の行が 0 本",
    );
    assertEquals(
      planWanGeneration(
        { prompt: "Two cats.", guidance: 1 },
        embeds,
        LAYOUT,
        CONFIG,
        WAN21_FAMILY.generation,
        WAN21_FAMILY.owner,
      ).negative,
      undefined,
    );
  });

  it("受理集合の外の寸法・フレーム数・step 数・shift は ModelInputError", () => {
    const rejected: readonly [Partial<WanGenerateRequest>, string][] = [
      [{ width: 832, height: 832 }, "受理集合"],
      [{ width: 840, height: 480 }, "受理集合"],
      [{ frames: 32 }, "frames"],
      [{ frames: 1 }, "frames"],
      [{ frames: 85 }, "frames"],
      [{ frames: 33.5 }, "frames"],
      [{ steps: 0 }, "steps"],
      [{ steps: 1.5 }, "steps"],
      [{ shift: 0 }, "shift"],
      [{ shift: Number.NaN }, "shift"],
    ];
    for (const [request, message] of rejected) {
      assertThrows(() => plan(request), ModelInputError, message);
    }
    assertEquals(plan({ steps: 1 }).steps, 1);
  });

  it("初期ノイズは seed か latents のどちらか 1 つで、latents は形と有限性を見る", () => {
    const count = 16 * 2 * 60 * 104;
    const latents = new Float32Array(count).fill(0.5);
    const resolved = plan({ frames: 5, latents });
    assert(resolved.initial.kind === "latents" && resolved.initial.data === latents);
    assertThrows(() => plan({ frames: 5, latents, seed: 1 }), ModelInputError, "排他");
    assertThrows(
      () => plan({ frames: 5, latents: new Float32Array(count - 1) }),
      ModelInputError,
      "latents",
    );
    latents[7] = Number.NaN;
    assertThrows(() => plan({ frames: 5, latents }), ModelInputError, "非有限");
    assertThrows(() => plan({ seed: -1 }), ModelInputError, "seed");
    assertEquals(plan({ seed: 42 }).initial, { kind: "seed", seed: 42 });
  });

  it("省いた steps / guidance / shift は manifest の pipelineConfig の値で埋め、明示した値はそれに勝つ", () => {
    // 参照の設定と重ならない値（既定を焼き込んでいれば 50 / 5 / 3 が出る）。
    const config: WanPipelineConfig = {
      scheduler: { shift: 7.5 },
      defaults: { steps: 23, guidance: 4.25 },
    };
    const resolved = planWanGeneration(
      { prompt: "Two cats." },
      EMBEDS,
      LAYOUT,
      config,
      WAN21_FAMILY.generation,
      WAN21_FAMILY.owner,
    );
    assertEquals([resolved.steps, resolved.guidance, resolved.shift], [23, 4.25, 7.5]);
    const explicit = planWanGeneration(
      { prompt: "Two cats.", steps: 2, guidance: 1, shift: 1.5 },
      EMBEDS,
      LAYOUT,
      config,
      WAN21_FAMILY.generation,
      WAN21_FAMILY.owner,
    );
    assertEquals([explicit.steps, explicit.guidance, explicit.shift], [2, 1, 1.5]);
    assertEquals(explicit.negative, undefined, "guidance 1 の明示は既定の 4.25 に勝つ");
  });

  it("guidance は f32 に丸めて有限な値だけを受ける（CFG は f32 で掛けるので、f64 で有限でも溢れる）", () => {
    for (const guidance of [Number.MAX_VALUE, 1e39]) {
      assertThrows(() => plan({ guidance }), ModelInputError, "f32 に収まる");
    }
    const f32Max = 3.4028234663852886e38;
    assertEquals(plan({ guidance: f32Max }).guidance, f32Max, "f32 の最大値は境界の内");
  });

  it("steps × shift の組で σ 列が壊れる要求は ModelInputError（低水準の RangeError は cause に残す）", () => {
    // 単項の門（正の有限の shift・1 以上の整数の steps）は通る組。
    const rejected: readonly Partial<WanGenerateRequest>[] = [
      { shift: 1e6 }, // σ[0] < σ[1]（shift が大きすぎて 1 へ張り付く）
      { shift: 1e-320 }, // σ[0] が Infinity
      { steps: 400_000 }, // 隣り合う σ が f32 で逆転する
    ];
    for (const request of rejected) {
      const error = assertThrows(() => plan(request), ModelInputError, "σ 列が組めない");
      assertInstanceOf(error.cause, RangeError);
    }
  });

  it("計画は要求の steps × shift のスケジュールを持つ（denoise は組み直さずにこれを使う）", () => {
    const explicit = plan({ steps: 3, shift: 1.5 });
    assertEquals(
      explicit.schedule,
      wanUniPcSchedule(3, 1.5, WAN_UNIPC_CONFIG.numTrainTimesteps),
    );
    assertEquals(plan({}).schedule.timesteps.length, 50, "省いた steps は既定の 50");
  });

  it("VAE のタイル計画を denoise の前に立てる（832×480 は 3×4 枚・480×832 は 4×3 枚）", () => {
    const landscape = plan({}).tiles;
    assertEquals([...landscape.rows.starts], [0, 14, 28]);
    assertEquals([...landscape.cols.starts], [0, 24, 48, 72]);
    const portrait = plan({ width: 480, height: 832 }).tiles;
    assertEquals([...portrait.rows.starts], [0, 24, 48, 72]);
    assertEquals([...portrait.cols.starts], [0, 14, 28]);
  });
});

/**
 * VAE の chunk グラフ 2 本（first / next）の宣言だけを持つ偽物（潜在タイル `tile`・縮尺 8・cache 1 本 —
 * `vae-chunks.ts` の取り決めの形）。チャネル数は既定で配布形（潜在 16・出口 RGB の 3）。
 */
const vaeChunkGraphs = (
  tile: number,
  scale = 8,
  channels: { readonly latent: number; readonly sample: number } = { latent: 16, sample: 3 },
) => {
  const graph = (frames: number) =>
    stubModel({
      inputs: [
        { name: "latent", shape: [channels.latent, 1, tile, tile] },
        { name: "cache_00", shape: [16, 2, tile, tile] },
      ],
      outputs: ["frame", "cache_00_out"],
      values: {
        frame: [channels.sample, frames, scale * tile, scale * tile],
        cache_00_out: [16, 2, tile, tile],
      },
    });
  return { first: graph(1), next: graph(4) };
};

describe("家族 admission: VAE のタイルが受理する寸法を全部覆う", () => {
  const layoutOf = (tile: number, scale?: number) => {
    const { first, next } = vaeChunkGraphs(tile, scale);
    return wanVaeChunkLayout(first, next);
  };

  it("潜在タイル 32（配布形）と 60（短辺ちょうど）は 832×480 / 480×832 の両方を覆う", () => {
    for (const tile of [32, 60]) {
      assertWanVaeTilesCover(layoutOf(tile), WAN21_FAMILY.generation, WAN21_FAMILY.owner);
    }
  });

  it("chunk グラフの検査は通るがタイル decode できない資産を、寸法と理由を言って拒む", () => {
    // 8: 重なりの下限 8 がタイル幅未満にならない。64: 潜在の短辺 60 より大きい。
    for (const [tile, reason] of [[8, "最小の重なり"], [64, "タイル幅 64 より小さい"]] as const) {
      const error = assertThrows(
        () => assertWanVaeTilesCover(layoutOf(tile), WAN21_FAMILY.generation, WAN21_FAMILY.owner),
        Error,
        reason,
      );
      assert(error.message.includes("832×480"), error.message);
      assert(!(error instanceof ModelInputError), "資産の齟齬を入力起因にしない");
    }
  });

  it("縮尺で受理する寸法の潜在が整数にならない資産を拒む", () => {
    assertThrows(
      () => assertWanVaeTilesCover(layoutOf(32, 7), WAN21_FAMILY.generation, WAN21_FAMILY.owner),
      Error,
      "整数にならない",
    );
  });
});

describe("家族 admission: VAE のグラフ宣言 × 世代の記述子（統計の本数・出口のチャネル数）", () => {
  const layoutOf = (channels: { readonly latent: number; readonly sample: number }) => {
    const { first, next } = vaeChunkGraphs(32, 8, channels);
    return wanVaeChunkLayout(first, next);
  };

  it("配布形の宣言（潜在 16・出口 3）は Wan2.1 の記述子（統計 16 本・unpatchify 1）と合う", () => {
    assertWanVaeMatchesGeneration(
      layoutOf({ latent: 16, sample: 3 }),
      WAN21_GENERATION,
      "WanPipeline",
    );
  });

  it("統計の本数が潜在のチャネル数と違う資産を拒む（要素数が割り切れて逆正規化が黙って通る形）", () => {
    const error = assertThrows(
      () =>
        assertWanVaeMatchesGeneration(
          layoutOf({ latent: 8, sample: 3 }),
          WAN21_GENERATION,
          "WanPipeline",
        ),
      Error,
      "WanPipeline: 逆正規化の統計（mean 16 本・std 16 本）が VAE の潜在 8 チャネルと違う",
    );
    assert(!(error instanceof ModelInputError), "資産の齟齬を入力起因にしない");
  });

  it("std の本数だけが潜在のチャネル数と違う記述子も拒む（mean と std を別々に見る）", () => {
    // 合成の記述子（mean は 16 本で潜在と合い、std だけが 8 本）。
    const stdOnly: WanGenerationDescriptor = {
      ...WAN21_GENERATION,
      latents: { mean: WAN21_GENERATION.latents.mean, std: Array<number>(8).fill(1) },
    };
    assertThrows(
      () =>
        assertWanVaeMatchesGeneration(layoutOf({ latent: 16, sample: 3 }), stdOnly, "WanPipeline"),
      Error,
      "WanPipeline: 逆正規化の統計（mean 16 本・std 8 本）が VAE の潜在 16 チャネルと違う",
    );
  });

  it("出口のチャネル数が RGB × unpatchify² と違う資産を拒む", () => {
    assertThrows(
      () =>
        assertWanVaeMatchesGeneration(
          layoutOf({ latent: 16, sample: 12 }),
          WAN21_GENERATION,
          "WanPipeline",
        ),
      Error,
      "WanPipeline: VAE の出口 12 チャネルが RGB 3 × unpatchify 1² = 3 と違う",
    );
  });
});

describe("空間の圧縮（グラフの比 × unpatchify）: 潜在の大きさ・覆えるかの門・重なりが同じ 1 本を使う", () => {
  // 合成の世代と資産（値は門を縛るためのもので、src には置かない）: 潜在タイル 16 → 出力 128（グラフの比
  // 8）・unpatchify 2 → 空間の圧縮 16・出口 12 = RGB 3 × 2²。グラフの比だけで割ると潜在も重なりも倍になる。
  const SYNTHETIC: WanGenerationDescriptor = {
    ...WAN21_GENERATION,
    latents: { mean: [0, 0, 0, 0], std: [1, 1, 1, 1] },
    vaePatchSize: 2,
  };
  const graphs = vaeChunkGraphs(16, 8, { latent: 4, sample: 12 });
  const layout = wanVaeChunkLayout(graphs.first, graphs.next);
  const gate: PromptGate<string> = { resolve: (text) => text, defaultNegative: () => "negative" };
  const planOf = (generation: WanGenerationDescriptor) =>
    planWanRequest({ prompt: "p" }, gate, layout, CONFIG, generation);

  it("合成の宣言は記述子との照合を通り、空間の圧縮は 8 × 2 = 16", () => {
    assertWanVaeMatchesGeneration(layout, SYNTHETIC, "WanPipeline");
    assertEquals(wanSpatialCompression(layout, SYNTHETIC), 16);
  });

  it("832×480 の潜在は 30×52 で、計画の潜在の形・タイル計画・重なり（64 px ÷ 16 = 4）が揃う", () => {
    const plan = planOf(SYNTHETIC);
    assertEquals(plan.latentShape, [4, 9, 30, 52]);
    // 計画のタイルは本番の入口（planWanGenerationTiles）を通った計画で、潜在の大きさと同じ軸を覆う
    // （入口自身の重なりの式は、この行ではなく下の本数で縛る）。
    assertEquals(plan.tiles, planWanGenerationTiles(layout, 30, 52, SYNTHETIC));
    assertEquals([plan.tiles.rows.extent, plan.tiles.cols.extent], [30, 52]);
    // 本数は「重なり 4 以上」を満たす最小（重なりを 8 と取ると列が増える）— 式の性質で縛る。
    const overlap = 64 / 16;
    for (const axis of [plan.tiles.rows, plan.tiles.cols]) {
      const span = axis.extent - axis.tile;
      assertEquals(axis.starts.length, Math.ceil(span / (axis.tile - overlap)) + 1);
    }
    // 覆えるかの門も同じ大きさで計画して通る。
    assertWanVaeTilesCover(layout, SYNTHETIC, "WanPipeline");
  });

  it("圧縮 16 で割れない寸法（840×480 — グラフの比 8 だけなら割れる）は、門も計画も同じく拒む", () => {
    const wide: WanGenerationDescriptor = {
      ...SYNTHETIC,
      acceptedSizes: [{ width: 840, height: 480 }],
      defaults: { ...SYNTHETIC.defaults, width: 840 },
    };
    assertThrows(
      () => assertWanVaeTilesCover(layout, wide, "WanPipeline"),
      Error,
      "空間の圧縮 16（VAE の縮尺 128 / 16 × unpatchify 2）では 840×480 の潜在が整数にならない",
    );
    assertThrows(() => planOf(wide), Error, "潜在の形 [4,9,30,52.5] が整数でない（空間の圧縮 16）");
  });
});

describe("家族 admission: DiT のグラフ宣言 × ホストが組む形（rank・batch・可変 S まで）", () => {
  /** RoPE の素表（`ditContract` が見るのは幅 `2·(t + h + w)` = 128 だけ）。 */
  const ROPE: WanRopeBase = {
    rows: 1,
    widths: [22, 21, 21],
    cos: [new Float32Array(22), new Float32Array(21), new Float32Array(21)],
    sin: [new Float32Array(22), new Float32Array(21), new Float32Array(21)],
  };
  type DitShapes = {
    readonly tokens: readonly StubDim[];
    readonly output: readonly StubDim[];
    readonly ropeCos: readonly StubDim[];
    readonly ropeSin: readonly StubDim[];
    readonly proj: readonly StubDim[];
    readonly context: readonly StubDim[];
  };
  /** 配布形と同じ宣言（`[1, S, 64]` ほか — recipe `wan/export_dit.py` の forward）。 */
  const VALID: DitShapes = {
    tokens: [1, "S", 64],
    output: [1, "S", 64],
    ropeCos: [1, "S", 1, 128],
    ropeSin: [1, "S", 1, 128],
    proj: [1, 256],
    context: [1, 512, 4096],
  };
  /** 配布形の VAE の潜在（16 チャネル）で組んだ DiT の patch。 */
  const DIT_PATCH = wanDitPatch(16);
  const transformerOf = (patch: Partial<DitShapes>) => {
    const shapes = { ...VALID, ...patch };
    return stubModel({
      symbols: ["S", "T"],
      inputs: [
        { name: "tokens", shape: shapes.tokens },
        { name: "timesteps_proj", shape: shapes.proj },
        { name: "encoder_hidden_states", shape: shapes.context },
        { name: "rope_cos", shape: shapes.ropeCos },
        { name: "rope_sin", shape: shapes.ropeSin },
      ],
      outputs: ["out"],
      values: { out: shapes.output },
    });
  };

  it("配布形の宣言は通り、文脈と timestep の幅を宣言から引く", () => {
    assertEquals(ditContract(transformerOf({}), ROPE, DIT_PATCH, "t2v", "WanPipeline"), {
      output: "out",
      projWidth: 256,
      contextRows: 512,
      contextWidth: 4096,
      patch: DIT_PATCH,
      form: "t2v",
    });
  });

  it("patch の刻みは上流の (1, 2, 2)・チャネル数は VAE の宣言から（16 なら tokens の幅 64）", () => {
    assertEquals(wanDitPatch(16), { channels: 16, patchFrames: 1, patchHeight: 2, patchWidth: 2 });
  });

  it("VAE の潜在のチャネル数が DiT の tokens の幅と合わなければ落ちる（patch は VAE の宣言から組む）", () => {
    assertThrows(
      () => ditContract(transformerOf({}), ROPE, wanDitPatch(8), "t2v", "WanPipeline"),
      Error,
      "'tokens' の形",
    );
  });

  it("batch 2・固定の S・rank 違い・S の記号の食い違いは、Session を張る前に名指しで落ちる", () => {
    const rejected: readonly [string, Partial<DitShapes>, string][] = [
      ["tokens の batch 2", { tokens: [2, "S", 64] }, "'tokens' の形"],
      ["tokens の固定 S", { tokens: [1, 192, 64] }, "記号次元でない"],
      ["tokens の rank 2", { tokens: ["S", 64] }, "記号次元でない"],
      ["tokens の rank 4", { tokens: [1, "S", 1, 64] }, "'tokens' の形"],
      ["出力の batch 2", { output: [2, "S", 64] }, "transformer の出力 'out' の形"],
      ["出力の固定 S", { output: [1, 192, 64] }, "transformer の出力 'out' の形"],
      ["出力の別の記号", { output: [1, "T", 64] }, "transformer の出力 'out' の形"],
      ["rope_cos の別の記号", { ropeCos: [1, "T", 1, 128] }, "'rope_cos' の形"],
      ["rope_sin の rank 3", { ropeSin: [1, "S", 128] }, "'rope_sin' の形"],
      ["rope_cos の batch 2", { ropeCos: [2, "S", 1, 128] }, "'rope_cos' の形"],
      ["timesteps_proj の batch 2", { proj: [2, 256] }, "'timesteps_proj' の形"],
      [
        "encoder_hidden_states の batch 2",
        { context: [2, 512, 4096] },
        "'encoder_hidden_states' の形",
      ],
      [
        "encoder_hidden_states の rank 4",
        { context: [1, 512, 4096, 1] },
        "'encoder_hidden_states' の形",
      ],
      ["tokens の最終次元", { tokens: [1, "S", 32] }, "'tokens' の形"],
      ["rope の幅", { ropeSin: [1, "S", 1, 64] }, "'rope_sin' の形"],
    ];
    for (const [label, patch, message] of rejected) {
      assertThrows(
        () => ditContract(transformerOf(patch), ROPE, DIT_PATCH, "t2v", "WanPipeline"),
        Error,
        message,
        label,
      );
    }
  });

  /** t2v の 5 本の名前（`wan/export_dit.py` の `INPUT_NAMES`）— 集合の文言の期待に使う。 */
  const T2V_NAMES = "tokens, timesteps_proj, encoder_hidden_states, rope_cos, rope_sin";
  const TI2V_NAMES = `${T2V_NAMES}, timesteps_proj_condition, condition_mask`;

  it("入力の名前の集合の検査は既存の 5 本の検査の後（入力が欠けた 2.1 の資産の文言は変えない）", () => {
    const missing = stubModel({
      symbols: ["S"],
      inputs: [
        { name: "tokens", shape: VALID.tokens },
        { name: "timesteps_proj", shape: VALID.proj },
        { name: "encoder_hidden_states", shape: VALID.context },
        { name: "rope_cos", shape: VALID.ropeCos },
      ],
      outputs: ["out"],
      values: { out: VALID.output },
    });
    assertThrows(
      () => ditContract(missing, ROPE, DIT_PATCH, "t2v", "WanPipeline"),
      Error,
      "WanPipeline: transformer のグラフ入力 'rope_sin' が無い",
    );
  });

  it("故障注入: 2.1 の宣言に 6 本目の入力があると、t2v の集合と一致しないので落ちる", () => {
    const extra = stubModel({
      symbols: ["S"],
      inputs: [
        { name: "tokens", shape: VALID.tokens },
        { name: "timesteps_proj", shape: VALID.proj },
        { name: "encoder_hidden_states", shape: VALID.context },
        { name: "rope_cos", shape: VALID.ropeCos },
        { name: "rope_sin", shape: VALID.ropeSin },
        { name: "attention_mask", shape: [1, "S"] },
      ],
      outputs: ["out"],
      values: { out: VALID.output },
    });
    assertThrows(
      () => ditContract(extra, ROPE, DIT_PATCH, "t2v", "WanPipeline"),
      Error,
      `WanPipeline: transformer のグラフ入力が [${T2V_NAMES}, attention_mask]` +
        `（期待: [${T2V_NAMES}] — 入力の形 't2v'）`,
    );
  });

  it("未知の入力の形は黙って t2v として扱わず落ちる（型の外の値 — JS の呼び手・壊れた記述子）", () => {
    assertThrows(
      () =>
        Reflect.apply(ditContract, undefined, [
          transformerOf({}),
          ROPE,
          DIT_PATCH,
          "i2v",
          "WanPipeline",
        ]),
      Error,
      `WanPipeline: DiT の入力の形 'i2v' は未知（"t2v" / "ti2v"）`,
    );
    // 壊れた資産（入力の欠け）と重なっても、未知の形の文言が先に出る（資産の中身に左右されない）。
    const broken = stubModel({
      symbols: ["S"],
      inputs: [{ name: "tokens", shape: VALID.tokens }],
      outputs: ["out"],
      values: { out: VALID.output },
    });
    assertThrows(
      () => Reflect.apply(ditContract, undefined, [broken, ROPE, DIT_PATCH, "i2v", "WanPipeline"]),
      Error,
      `WanPipeline: DiT の入力の形 'i2v' は未知（"t2v" / "ti2v"）`,
    );
  });

  describe("入力の形 ti2v（Wan2.2 TI2V — 5 本 + 条件側の時刻・条件マスク）", () => {
    type Ti2vInput = {
      readonly name: string;
      readonly shape: readonly StubDim[];
      readonly dtype?: "f32" | "bool";
    };
    /** 潜在 48 チャネルの patch（`tokens` の幅 192）。 */
    const TI2V_PATCH = wanDitPatch(48);
    /** recipe `wan/ti2v_export_dit.py` と同じ宣言（`wan_ti2v_dit_host_test.ts` の `EXPECTED_INPUTS`）。 */
    const TI2V_VALID: readonly Ti2vInput[] = [
      { name: "tokens", shape: [1, "S", 192] },
      { name: "timesteps_proj", shape: [1, 256] },
      { name: "encoder_hidden_states", shape: [1, 512, 4096] },
      { name: "rope_cos", shape: [1, "S", 1, 128] },
      { name: "rope_sin", shape: [1, "S", 1, 128] },
      { name: "timesteps_proj_condition", shape: [1, 256] },
      { name: "condition_mask", shape: [1, "S", 1], dtype: "bool" },
    ];
    const ti2vOf = (inputs: readonly Ti2vInput[]) =>
      stubModel({ symbols: ["S", "T"], inputs, outputs: ["out"], values: { out: [1, "S", 192] } });
    /** 正常形の入力 1 本を差し替える（`replacement` を省けば外す）。 */
    const replaced = (name: string, replacement?: Ti2vInput): readonly Ti2vInput[] =>
      TI2V_VALID.flatMap((input) =>
        input.name !== name ? [input] : replacement === undefined ? [] : [replacement]
      );

    it("正常形は通り、form を ti2v として返す（数値の欄は t2v と同じく宣言から引く）", () => {
      assertEquals(ditContract(ti2vOf(TI2V_VALID), ROPE, TI2V_PATCH, "ti2v", "WanTi2vPipeline"), {
        output: "out",
        projWidth: 256,
        contextRows: 512,
        contextWidth: 4096,
        patch: TI2V_PATCH,
        form: "ti2v",
      });
    });

    it("条件入力の欠け・dtype・形・記号の食い違いと、形と宣言の取り違えは名指しで落ちる", () => {
      const rejected: readonly [string, readonly Ti2vInput[], WanDitInputForm, string][] = [
        [
          "condition_mask の欠け",
          replaced("condition_mask"),
          "ti2v",
          `WanTi2vPipeline: transformer のグラフ入力が [${T2V_NAMES}, timesteps_proj_condition]` +
          `（期待: [${TI2V_NAMES}] — 入力の形 'ti2v'）`,
        ],
        [
          "condition_mask が f32",
          replaced("condition_mask", { name: "condition_mask", shape: [1, "S", 1] }),
          "ti2v",
          "WanTi2vPipeline: transformer のグラフ入力 'condition_mask' の dtype f32 が bool でない",
        ],
        [
          "condition_mask が [1, S, 2]",
          replaced("condition_mask", { name: "condition_mask", shape: [1, "S", 2], dtype: "bool" }),
          "ti2v",
          "WanTi2vPipeline: 'condition_mask' の形 [1, S, 2] がホストの組む [1, S, 1] と違う",
        ],
        [
          "condition_mask の記号が tokens と別",
          replaced("condition_mask", { name: "condition_mask", shape: [1, "T", 1], dtype: "bool" }),
          "ti2v",
          "WanTi2vPipeline: 'condition_mask' の形 [1, T, 1] がホストの組む [1, S, 1] と違う",
        ],
        [
          "条件側の時刻の幅 128",
          replaced("timesteps_proj_condition", {
            name: "timesteps_proj_condition",
            shape: [1, 128],
          }),
          "ti2v",
          "WanTi2vPipeline: 'timesteps_proj_condition' の形 [1, 128] がホストの組む [1, 256] と違う",
        ],
        [
          "条件側の時刻が bool",
          replaced("timesteps_proj_condition", {
            name: "timesteps_proj_condition",
            shape: [1, 256],
            dtype: "bool",
          }),
          "ti2v",
          "WanTi2vPipeline: transformer のグラフ入力 'timesteps_proj_condition' の dtype bool が f32 でない",
        ],
        [
          "TI2V の宣言を t2v として開く（余分な 2 本）",
          TI2V_VALID,
          "t2v",
          `WanTi2vPipeline: transformer のグラフ入力が [${TI2V_NAMES}]` +
          `（期待: [${T2V_NAMES}] — 入力の形 't2v'）`,
        ],
        [
          "2.1 の 5 本の宣言を ti2v として開く",
          TI2V_VALID.slice(0, 5),
          "ti2v",
          `WanTi2vPipeline: transformer のグラフ入力が [${T2V_NAMES}]` +
          `（期待: [${TI2V_NAMES}] — 入力の形 'ti2v'）`,
        ],
      ];
      for (const [label, inputs, form, message] of rejected) {
        assertThrows(
          () => ditContract(ti2vOf(inputs), ROPE, TI2V_PATCH, form, "WanTi2vPipeline"),
          Error,
          message,
          label,
        );
      }
    });
  });
});

describe("ditInputs（DiT の 1 回の forward の入力を組む 1 か所）", () => {
  /** ホスト配列の入力（常駐入力ではない）を取り出す。 */
  const hostTensor = (inputs: RunInputs, name: string): Tensor => {
    assert(Object.hasOwn(inputs, name), `入力 '${name}' が無い`);
    const input = inputs[name];
    assert("dtype" in input, `入力 '${name}' がホスト配列でない`);
    return input;
  };
  const BASE = {
    tokens: new Float32Array(6),
    tokenShape: [1, 3, 2],
    proj: new Float32Array(4),
    projShape: [1, 4],
    context: new Float32Array(2),
    contextShape: [1, 1, 2],
    rope: { cos: new Float32Array(3), sin: new Float32Array(3) },
    ropeShape: [1, 3, 1, 1],
  };

  it("condition なし（t2v）はキーが 5 本ちょうどで、順と配列の同一性を保つ（写さない）", () => {
    const inputs = ditInputs({ ...BASE, condition: undefined });
    assertEquals(Object.keys(inputs), [
      "tokens",
      "timesteps_proj",
      "encoder_hidden_states",
      "rope_cos",
      "rope_sin",
    ]);
    const expected: readonly [string, Float32Array, readonly number[]][] = [
      ["tokens", BASE.tokens, BASE.tokenShape],
      ["timesteps_proj", BASE.proj, BASE.projShape],
      ["encoder_hidden_states", BASE.context, BASE.contextShape],
      ["rope_cos", BASE.rope.cos, BASE.ropeShape],
      ["rope_sin", BASE.rope.sin, BASE.ropeShape],
    ];
    for (const [name, data, shape] of expected) {
      const tensor = hostTensor(inputs, name);
      assertEquals(tensor.dtype, "f32", name);
      assertStrictEquals(tensor.data, data, name);
      assertStrictEquals(tensor.shape, shape, name);
    }
  });

  it("condition あり（ti2v）は 7 本で、条件側の時刻は f32・生成側と同じ形、マスクは bool（借用のまま）", () => {
    const mask = new Uint32Array(3);
    const maskShape = [1, 3, 1];
    const inputs = ditInputs({ ...BASE, condition: { proj: BASE.proj, mask, maskShape } });
    assertEquals(Object.keys(inputs), [
      "tokens",
      "timesteps_proj",
      "encoder_hidden_states",
      "rope_cos",
      "rope_sin",
      "timesteps_proj_condition",
      "condition_mask",
    ]);
    const proj = hostTensor(inputs, "timesteps_proj_condition");
    assertEquals(proj.dtype, "f32");
    assertStrictEquals(proj.data, BASE.proj);
    assertStrictEquals(proj.shape, BASE.projShape);
    const condition = hostTensor(inputs, "condition_mask");
    assertEquals(condition.dtype, "bool");
    assertStrictEquals(condition.data, mask);
    assertStrictEquals(condition.shape, maskShape);
  });
});

describe("モデルカードの受理集合（fixture を挟んだ突き合わせ）", () => {
  // 反対側は recipe の `wan/tests/test_distribution.py`（card.py の表を同じ fixture と比べる）。
  it("受理する寸法とフレーム数の範囲は fixture wan-card-limits.json と同じ", async () => {
    const fixture: unknown = JSON.parse(
      await Deno.readTextFile(new URL("./fixtures/wan-card-limits.json", import.meta.url)),
    );
    const { acceptedSizes, minFrames, maxFrames } = WAN21_GENERATION;
    assertEquals(
      fixture,
      { acceptedSizes, minFrames, maxFrames },
      "descriptor.ts の受理集合を変えたら fixture と card.py の WAN_ACCEPTED_SIZES / WAN_FRAMES も揃える",
    );
  });

  // Wan2.2 TI2V-5B の配布形（`karume-wan2.2`）のカードも同じ形で縛る（ADR 0121 段 8 — 反対側は同じ recipe のテストが
  // card.py の `WAN22_ACCEPTED_SIZES` / `WAN22_FRAMES` を同じ fixture と比べる）。
  it("Wan2.2 TI2V の受理する寸法とフレーム数の範囲は fixture wan-ti2v-card-limits.json と同じ", async () => {
    const fixture: unknown = JSON.parse(
      await Deno.readTextFile(new URL("./fixtures/wan-ti2v-card-limits.json", import.meta.url)),
    );
    const { acceptedSizes, minFrames, maxFrames } = WAN22_TI2V_GENERATION;
    assertEquals(
      fixture,
      { acceptedSizes, minFrames, maxFrames },
      "descriptor.ts の WAN22_TI2V_GENERATION の受理集合を変えたら fixture と card.py の WAN22_ACCEPTED_SIZES / WAN22_FRAMES も揃える",
    );
  });
});

describe("pipelineConfig（manifest の宣言の門）", () => {
  const RAW = { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } };

  it("配布形の宣言をそのまま読む", () => {
    assertEquals(parseWanPipelineConfig(RAW), CONFIG);
  });

  it("未知キー・欠落・値域の外は素の Error（資産の齟齬 — 入力起因ではない）", () => {
    const rejected: readonly [Record<string, unknown>, string][] = [
      [{ ...RAW, steps: 50 }, "未知キー 'steps'"],
      [{ ...RAW, scheduler: { shift: 3, type: "unipc" } }, "未知キー 'type'"],
      [{ ...RAW, defaults: { steps: 50, guidanceScale: 5 } }, "未知キー 'guidanceScale'"],
      [{ defaults: RAW.defaults }, "pipelineConfig.scheduler: 無い"],
      [{ scheduler: RAW.scheduler }, "pipelineConfig.defaults: 無い"],
      [{ ...RAW, scheduler: {} }, "pipelineConfig.scheduler.shift: 無い"],
      [{ ...RAW, scheduler: { shift: 0 } }, "shift"],
      [{ ...RAW, defaults: { steps: 0, guidance: 5 } }, "steps"],
      [{ ...RAW, defaults: { steps: 2.5, guidance: 5 } }, "steps"],
      [{ ...RAW, defaults: { steps: 50, guidance: 0.5 } }, "guidance"],
      [{ ...RAW, defaults: { steps: 50, guidance: "5" } }, "guidance"],
    ];
    for (const [raw, message] of rejected) {
      const error = assertThrows(() => parseWanPipelineConfig(raw), Error, message);
      assert(!(error instanceof ModelInputError), `宣言の齟齬を入力起因にしない: ${message}`);
    }
  });

  it("要求の門と同じ値域: f32 で溢れる guidance・既定の steps × shift で σ 列が組めない宣言も素の Error", () => {
    // 既定の組を門で通しておくことが、planWanGeneration の σ 列の失敗を入力起因と読める前提。
    const rejected: readonly [Record<string, unknown>, string][] = [
      [{ ...RAW, defaults: { steps: 50, guidance: Number.MAX_VALUE } }, "guidance"],
      [{ ...RAW, scheduler: { shift: 1e6 } }, "σ 列が組めない"],
      [{ ...RAW, defaults: { steps: 400_000, guidance: 5 } }, "σ 列が組めない"],
    ];
    for (const [raw, message] of rejected) {
      const error = assertThrows(() => parseWanPipelineConfig(raw), Error, message);
      assert(!(error instanceof ModelInputError), `宣言の齟齬を入力起因にしない: ${message}`);
    }
  });
});

describe("潜在の逆正規化", () => {
  it("定数は f32 の値そのもの（f64 の小数を置くと逆正規化で使う値と食い違う）", () => {
    assertEquals(WAN_LATENTS_MEAN.length, 16);
    assertEquals(WAN_LATENTS_STD.length, 16);
    for (const value of [...WAN_LATENTS_MEAN, ...WAN_LATENTS_STD]) {
      assertEquals(Math.fround(value), value);
    }
  });

  it("上流の順（std の逆数で割ってから mean を足す）で、掛け算の順とは最終ビットが割れる", () => {
    const perChannel = 4096;
    const latents = new Float32Array(16 * perChannel).map((_, index) =>
      Math.fround(Math.sin(index) * 3)
    );
    const got = denormalizeWanLatents(latents, WAN21_GENERATION.latents);
    let differsFromProduct = 0;
    for (let channel = 0; channel < 16; channel += 1) {
      const inverse = Math.fround(1 / WAN_LATENTS_STD[channel]);
      for (let index = 0; index < perChannel; index += 1) {
        const at = channel * perChannel + index;
        assertEquals(
          got[at],
          Math.fround(Math.fround(latents[at] / inverse) + WAN_LATENTS_MEAN[channel]),
        );
        const product = Math.fround(
          Math.fround(latents[at] * WAN_LATENTS_STD[channel]) + WAN_LATENTS_MEAN[channel],
        );
        if (product !== got[at]) differsFromProduct += 1;
      }
    }
    assert(differsFromProduct > 0, "掛け算の順でも一致した（順の取り違えを縛れていない）");
    assertThrows(
      () => denormalizeWanLatents(new Float32Array(17), WAN21_GENERATION.latents),
      Error,
      "割り切れない",
    );
  });

  it("mean と std の本数が違う統計は fail loudly（チャネル数を片方から黙って決めない）", () => {
    const { mean, std } = WAN21_GENERATION.latents;
    assertThrows(
      () => denormalizeWanLatents(new Float32Array(16), { mean, std: std.slice(1) }),
      Error,
      "mean 16 本と std 15 本の数が違う",
    );
  });
});

describe("フレームの RGBA 化", () => {
  // fps を持たない手組みの動画（wanFrameToRgba は fps を読まないので要求しない — 型検査が縛る）。
  const video = {
    frames: 2,
    width: 2,
    height: 1,
    // [3, 2, 1, 2]: チャネル c・フレーム f・画素 x の値。
    data: Float32Array.from([-1, 1, 0, 0.5, 0.25, -0.25, 2, -2, -0.5, 0.75, 1, -1]),
  };

  it("clamp(x/2 + 0.5) を round(·255) で 8bit にし、フレームごとの平面を引く", () => {
    assertEquals([...wanFrameToRgba(video, 0)], [0, 159, 64, 255, 255, 96, 223, 255]);
    assertEquals([...wanFrameToRgba(video, 1)], [128, 255, 255, 255, 191, 0, 0, 255]);
  });

  it("範囲外のフレームと非有限値は fail loudly", () => {
    assertThrows(() => wanFrameToRgba(video, 2), RangeError);
    const broken = { ...video, data: Float32Array.from(video.data).fill(Number.NaN, 3, 4) };
    assertThrows(() => wanFrameToRgba(broken, 1), Error, "非有限");
  });
});

describe("初期ノイズの乱数", () => {
  it("同じ seed なら同じ列・別の seed なら別の列（奇数長でも対を丸ごと消費する）", () => {
    assertEquals([...new WanRandn(7).normals(5)], [...new WanRandn(7).normals(5)]);
    assertNotEquals([...new WanRandn(7).normals(5)], [...new WanRandn(8).normals(5)]);
    const generator = new WanRandn(7);
    generator.normals(3);
    assertEquals([...generator.normals(2)], [...new WanRandn(7).normals(6)].slice(4));
  });

  it("seed → 列を値で固定する（同じ seed なら同じ clip — 公開の約束）", () => {
    // 上の「同じ seed なら同じ列」は生成器が変わっても通る（cos / sin の入れ替え・丸めの変更など）。
    // 列が変われば seed 付きの全 clip が黙って変わるので、先頭の値そのものを固定する。seed 0 は
    // generate の既定・42 は example と opt-in の sha 行の seed。値は Deno（V8）で焼いた f32
    // （ECMAScript は Math.log / cos / sin を実装依存の近似とするので、別のエンジンでは割れうる）。
    assertEquals([...new WanRandn(0).normals(8)], [
      -0.45275774598121643,
      0.20776604115962982,
      2.6506059169769287,
      -0.4904228150844574,
      -0.9886041283607483,
      1.8721014261245728,
      0.2524627149105072,
      -1.853424310684204,
    ]);
    assertEquals([...new WanRandn(42).normals(8)], [
      0.41471976041793823,
      0.6526812314987183,
      -0.8918862342834473,
      1.3268336057662964,
      1.72959303855896,
      -1.883416771888733,
      0.5456204414367676,
      -1.6568357944488525,
    ]);
  });
});

/**
 * 模擬 Session で回す `WanPipeline`（GPU も資産も要らない — 後始末と非有限の門を `generate` の経路で
 * 縛る）。DiT の Session は値を `ditValue` で埋めた出力を返し、VAE は 1 タイル = 1 batch の手順
 * （`decodeWanVaeTile`）を偽の GpuContext で回して、読み戻すフレームを `vaeValue` で埋める。
 *
 * NOTE: コンストラクタは TS の `private`（manifest 検査と資産の突合を迂回させない — ADR 0008）なので、
 * `Reflect.construct` で内部状態を直接渡す（private の迂回はテストだけ）。公開の構築口（`fromAssets`）は
 * krm コンテナのバイト列と GPU の取得を要り、CPU の単体テストでは回せない。内部状態の形は
 * family.ts の `WanState` だが、`Reflect.construct` の引数は型で縛られないので、ここで組む欄は手で
 * 揃える — 欄が欠ければ generate の中の TypeError で落ち、各テストが見る文言と食い違って赤になる。
 */
const mockPipeline = (options: {
  readonly ditValue?: number;
  readonly vaeValue?: number;
  readonly ditDisposeError?: Error;
  /**
   * `"gpu"` なら text 段を模擬の umT5 で回す（プロンプト層はフィクスチャの表 — {@link wanParityEncoder}）。
   * 出力は有効長 L の行を `umt5Value(L)` で埋める（positive と negative の取り違えが値に出る）。
   */
  readonly textEncoder?: "gpu";
  readonly umt5Value?: (tokens: number) => number;
  /** umT5 の run の中で呼ぶ（中断の注入口）。 */
  readonly onUmt5Run?: (tokens: number) => void;
  /**
   * 内部状態の DiT の入力の形（省けば `"t2v"`）。型の外の値も渡せる（`Reflect.construct` は型を見ない —
   * 未知の形の門を縛る）。
   */
  readonly ditForm?: string;
}) => {
  const log: string[] = [];
  /** DiT が受けた文脈（run の順 — uncond → cond）。 */
  const ditContexts: Float32Array[] = [];
  /** DiT の run が受けた入力の全体（run の順）。 */
  const ditRuns: Record<string, Tensor>[] = [];
  const { first, next } = vaeChunkGraphs(32);
  const resident = (byteLength: number) => ({ byteLength, write: () => {}, dispose: () => {} });
  const vaeSession = (name: string) => ({
    createSession: () => {
      log.push(`create:${name}`);
      return Promise.resolve({
        enqueue: () => Promise.resolve(),
        diagnostics: () => ({}),
        dispose: () => {
          log.push(`dispose:${name}`);
          return Promise.resolve();
        },
      });
    },
  });
  const gpuText = options.textEncoder === "gpu";
  const text = gpuText
    ? {
      kind: "gpu",
      encoder: wanParityEncoder(),
      contract: { output: "umt5_out" },
      component: {
        createSession: (_gpu: unknown, sessionOptions: unknown) => {
          log.push(`create:text_encoder:${JSON.stringify(sessionOptions)}`);
          return Promise.resolve({
            run: (inputs: Record<string, Tensor>) => {
              const tokens = inputs.input_ids.shape[1];
              log.push(`run:text_encoder:${tokens}`);
              options.onUmt5Run?.(tokens);
              return Promise.resolve({
                umt5_out: {
                  dtype: "f32",
                  shape: [1, tokens, WIDTH],
                  data: new Float32Array(tokens * WIDTH).fill(
                    options.umt5Value?.(tokens) ?? tokens / 1000,
                  ),
                },
              });
            },
            diagnostics: () => ({}),
            dispose: () => {
              log.push("dispose:text_encoder");
              return Promise.resolve();
            },
          });
        },
      },
    }
    : { kind: "precomputed" };
  const state = {
    gpu: {
      createResident: (bytes: number) => Promise.resolve(resident(bytes)),
      beginBatch: () =>
        Promise.resolve({
          finish: () => Promise.resolve(),
          finishAndRead: (frames: Record<string, { readonly byteLength: number }>) =>
            Promise.resolve(
              Object.fromEntries(
                Object.entries(frames).map(([name, { byteLength }]) => [
                  name,
                  new Float32Array(byteLength / 4).fill(options.vaeValue ?? 0.5).buffer,
                ]),
              ),
            ),
        }),
    },
    ownsGpu: false,
    config: CONFIG,
    sessionOptions: {},
    transformer: {
      createSession: () => {
        log.push("create:transformer");
        return Promise.resolve({
          run: (inputs: Record<string, Tensor>) => {
            const context = inputs.encoder_hidden_states;
            assert(context.dtype === "f32", "DiT の文脈が f32 でない");
            ditContexts.push(context.data);
            ditRuns.push(inputs);
            const tokens = inputs.tokens;
            return Promise.resolve({
              out: {
                dtype: "f32",
                shape: tokens.shape,
                data: new Float32Array(tokens.data.length).fill(options.ditValue ?? 0.1),
              },
            });
          },
          diagnostics: () => ({}),
          dispose: () => {
            log.push("dispose:transformer");
            return options.ditDisposeError === undefined
              ? Promise.resolve()
              : Promise.reject(options.ditDisposeError);
          },
        });
      },
    },
    vaeFirst: vaeSession("vae_decoder_first"),
    vaeNext: vaeSession("vae_decoder_next"),
    layout: wanVaeChunkLayout(first, next),
    // 行数 64 は潜在 [16, 2, 60, 104] の格子（2 × 30 × 52）を覆う。値は門と無関係。
    ropeBase: {
      rows: 64,
      widths: [1, 1, 1],
      cos: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
      sin: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
    },
    // GPU 経路は umT5 の出力（negative は 126 行）を詰めるので、文脈は配布形と同じ 512 行。
    dit: {
      output: "out",
      projWidth: 256,
      contextRows: gpuText ? 512 : 4,
      contextWidth: WIDTH,
      patch: wanDitPatch(16),
      form: options.ditForm ?? "t2v",
    },
    textEmbeds: EMBEDS,
    text,
  };
  const pipeline: WanPipeline = Reflect.construct(WanPipeline, [state]);
  /**
   * 5 フレーム・2 step（DiT 4 回・VAE 12 タイル）の要求で回し、観測したイベントを `log` に積む。プロンプトは
   * 資産の経路が `cats`、GPU 経路がフィクスチャの固定プロンプト `boxing-cats`（28 トークン）。
   */
  const generate = (
    onEvent?: WanGenerateRequest["onEvent"],
    request: Partial<WanGenerateRequest> = {},
  ) =>
    pipeline.generate({
      prompt: gpuText ? wanParityCase("fixed-boxing-cats").text : "Two cats.",
      frames: 5,
      steps: 2,
      ...request,
      onEvent: async (event) => {
        log.push(
          event.kind === "stage"
            ? `${event.component}:${event.at}`
            : event.kind === "denoise-step"
            ? `step:${event.step}`
            : `tile:${event.tile}`,
        );
        await onEvent?.(event);
      },
    });
  return { log, generate, ditContexts, ditRuns };
};

describe("WanPipeline.generate（模擬 Session）", () => {
  it("対照: 有限の DiT / VAE の出力なら最後まで回り、フレームを [-1, 1] へクランプして返す", async () => {
    const { log, generate } = mockPipeline({ vaeValue: 2 });
    const video = await generate();
    assertEquals([video.frames, video.width, video.height], [5, 832, 480]);
    assert(video.data.every((value) => value === 1), "クランプ前の 2 が 1 になっていない");
    assertEquals(log.filter((entry) => entry.startsWith("step:")), ["step:1", "step:2"]);
    assertEquals(log.filter((entry) => entry.startsWith("tile:")).length, 12);
    assertEquals(log.at(-1), "vae_decoder:end");
  });

  it("生成結果は Wan2.1 の fps 16 を持つ（上流の世代の事実 — 要求のノブではない）", async () => {
    const { generate } = mockPipeline({});
    assertEquals((await generate()).fps, 16);
  });

  describe("DiT の入力の形（ADR 0121 決定 3）", () => {
    const T2V_KEYS = ["tokens", "timesteps_proj", "encoder_hidden_states", "rope_cos", "rope_sin"];

    it("t2v は毎回の run に 5 本ちょうどを渡す（条件入力を足さない — 2.1 の Session に渡る入力は不変）", async () => {
      const { generate, ditRuns } = mockPipeline({});
      await generate();
      assertEquals(ditRuns.length, 4);
      for (const inputs of ditRuns) assertEquals(Object.keys(inputs), T2V_KEYS);
    });

    it("ti2v（T2V）は 7 本: 条件マスクは全て 0 の u32 [1, S, 1] を 1 回だけ作り、条件側の時刻は生成側と同じ配列", async () => {
      const { generate, ditRuns } = mockPipeline({ ditForm: "ti2v" });
      await generate();
      assertEquals(ditRuns.length, 4);
      const masks = new Set<unknown>();
      for (const inputs of ditRuns) {
        assertEquals(Object.keys(inputs), [
          ...T2V_KEYS,
          "timesteps_proj_condition",
          "condition_mask",
        ]);
        assertStrictEquals(inputs.timesteps_proj_condition.data, inputs.timesteps_proj.data);
        assertEquals(inputs.timesteps_proj_condition.shape, inputs.timesteps_proj.shape);
        const mask = inputs.condition_mask;
        assertEquals(mask.dtype, "bool");
        assertInstanceOf(mask.data, Uint32Array);
        const tokens = inputs.tokens.shape[1];
        assertEquals(mask.shape, [1, tokens, 1]);
        assertEquals(mask.data.length, tokens);
        assert(mask.data.every((value) => value === 0), "T2V の条件マスクに 0 でない要素がある");
        masks.add(mask.data);
      }
      assertEquals(masks.size, 1, "条件マスクを run ごとに作り直した");
    });

    it("未知の形は DiT の Session を張る前に落ちる（黙って t2v として回さない）", async () => {
      const { log, generate } = mockPipeline({ ditForm: "i2v" });
      await assertRejects(
        () => generate(),
        Error,
        `WanPipeline: DiT の入力の形 'i2v' は未知（"t2v" / "ti2v"）`,
      );
      assert(!log.includes("create:transformer"), `DiT の Session を張った: ${log}`);
    });
  });

  describe("非有限の門", () => {
    for (const ditValue of [Number.NaN, Number.POSITIVE_INFINITY]) {
      it(`DiT の出力が ${ditValue} なら step 1 の更新の後で落ち、VAE の段を張らない`, async () => {
        const { log, generate } = mockPipeline({ ditValue });
        await assertRejects(() => generate(), Error, "step 1/2 の更新後の潜在");
        assert(!log.includes("step:1"), "非有限の潜在を denoise-step で見せた");
        assert(!log.includes("vae_decoder:start"), `VAE の段へ進んだ: ${log}`);
        assertEquals(log.at(-1), "dispose:transformer", "DiT の Session を畳んでいない");
      });
    }

    for (const vaeValue of [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NaN]) {
      it(`VAE の出力が ${vaeValue} ならクランプの前で落ちる（±Inf を ±1 に化かさない）`, async () => {
        const { log, generate } = mockPipeline({ vaeValue });
        await assertRejects(
          () => generate(),
          Error,
          "VAE の出力（クランプ前）の channel 0・フレーム 0・画素 (x=0, y=0) が非有限",
        );
        assert(!log.includes("vae_decoder:end"), `VAE の段を終えた: ${log}`);
      });
    }
  });

  describe("DiT の段の後始末（本体の失敗を後始末の失敗で上書きしない）", () => {
    const thrown = new Error("A: onEvent の中断");
    const disposeFailure = new Error("B: Session.dispose の失敗");
    const abort = (event: Parameters<NonNullable<WanGenerateRequest["onEvent"]>>[0]) => {
      if (event.kind === "denoise-step") throw thrown;
    };

    it("本体 A と後始末 B の両方が落ちたら、A を先頭にした AggregateError で両方を運ぶ", async () => {
      const { log, generate } = mockPipeline({ ditDisposeError: disposeFailure });
      const error = await assertRejects(() => generate(abort), AggregateError);
      assertEquals(error.errors, [thrown, disposeFailure]);
      assert(log.includes("dispose:transformer"));
      assert(!log.includes("transformer:end"), "途中で落ちた段の end を出した");
    });

    it("本体だけが落ちたら A そのものを投げる", async () => {
      const { generate } = mockPipeline({});
      const error = await assertRejects(() => generate(abort));
      assertStrictEquals(error, thrown);
    });

    it("後始末だけが落ちたら B そのものを投げ、VAE の段へ進まない", async () => {
      const { log, generate } = mockPipeline({ ditDisposeError: disposeFailure });
      const error = await assertRejects(() => generate());
      assertStrictEquals(error, disposeFailure);
      assert(!log.includes("vae_decoder:start"), `VAE の段へ進んだ: ${log}`);
    });
  });
});

describe("planWanGpuGeneration（GPU 経路の入口の門）", () => {
  const encoder = wanParityEncoder();
  const BOXING_CATS = wanParityCase("fixed-boxing-cats");
  const gpuPlan = (request: Partial<WanGenerateRequest>) =>
    planWanGpuGeneration(
      { prompt: BOXING_CATS.text, ...request },
      encoder,
      LAYOUT,
      CONFIG,
      WAN21_FAMILY.generation,
    );

  it("プロンプトを上流と同じ id 列にし、省いた negative は公式の sample_neg_prompt を同じ門で符号化する", () => {
    const resolved = gpuPlan({});
    assertEquals([...resolved.positive], BOXING_CATS.ids);
    const negative = wanParityCase("fixed-negative");
    assertEquals([...(resolved.negative ?? [])], negative.ids);
    assertEquals(resolved.negative?.length, 126, "公式 negative は 126 トークン");
    // ノブの既定は資産の経路と同じ 1 本の門（参照の設定）。
    assertEquals(
      [resolved.steps, resolved.guidance, resolved.shift, resolved.frames],
      [50, 5, 3, 33],
    );
  });

  it("資産に無い文字列も受け、明示の negative はその文字列を符号化する・guidance 1 は uncond を回さない", () => {
    // 固定 4 本以外の受理されたケース（境界・乱択 — 資産の集合の外）。
    const free = wanParityCases.find((entry) =>
      entry.ids !== undefined && !entry.id.startsWith("fixed-")
    );
    assert(free !== undefined, "フィクスチャに受理された非固定のケースが無い");
    assertEquals([...gpuPlan({ prompt: free.text }).positive], free.ids);
    assertEquals([...(gpuPlan({ negativePrompt: free.text }).negative ?? [])], free.ids);
    assertEquals(gpuPlan({ guidance: 1 }).negative, undefined);
    assertThrows(
      () => gpuPlan({ guidance: 1, negativePrompt: free.text }),
      ModelInputError,
      "効かない",
    );
  });

  it("前処理の拒否は理由と直し方を言う（R&D → R & D・未割り当て / C1 はその文字を外す）", () => {
    const rejected: readonly [string, string, string][] = [
      ["R&D lab", "entity", "`R&D` → `R & D`"],
      ["a͸b", "unassigned", "その文字を外して渡す"],
      ["a\x85b", "c1", "その文字を外して渡す"],
    ];
    for (const [prompt, reason, hint] of rejected) {
      const error = assertThrows(() => gpuPlan({ prompt }), PromptCleanError);
      assertEquals(error.reason, reason, prompt);
      assertStringIncludes(error.message, hint);
      assertStringIncludes(error.message, "prompt:");
    }
    const negative = assertThrows(() => gpuPlan({ negativePrompt: "R&D lab" }), PromptCleanError);
    assertStringIncludes(negative.message, "negativePrompt:");
    // `R & D` と空白を挟めば前処理は通る（文言の直し方が実際に効く — 落ちるならトークナイザの語彙の側）。
    assertEquals(encoder.clean("R & D lab"), "R & D lab");
  });

  it("空・空白だけ（1 トークン）は 1 語以上を求めて拒む", () => {
    for (const prompt of ["", "   "]) {
      assertThrows(() => gpuPlan({ prompt }), ModelInputError, "1 語以上入れる");
    }
  });

  it("ノブの門は資産の経路と同じ 1 本（寸法・フレーム数・step 数を同じ文言で拒む）", () => {
    for (const request of [{ frames: 34 }, { steps: 0 }, { width: 840 }] as const) {
      const viaAsset = assertThrows(() => plan(request), ModelInputError);
      const viaGpu = assertThrows(() => gpuPlan(request), ModelInputError);
      assertEquals(viaGpu.message, viaAsset.message);
    }
  });
});

describe("既定の negative（公式の sample_neg_prompt）", () => {
  it("recipe の固定プロンプト（prompts.py の negative — fixture の fixed-negative）の原文とビット同一", () => {
    assertEquals(WAN_DEFAULT_NEGATIVE_PROMPT, wanParityCase("fixed-negative").text);
  });
});

/** recipe `wan/text_embeds.py` が書く埋め込み資産（ローカル資産 — 無い機では下の 1 本だけ SKIP）。 */
const TEXT_EMBEDS_SERIES = new URL(
  "../../../outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors",
  import.meta.url,
);
const textEmbedsSeries = await readFileIfPresent(TEXT_EMBEDS_SERIES);
if (textEmbedsSeries === undefined) {
  console.warn(
    `[karume] ${TEXT_EMBEDS_SERIES.pathname} が無いため、既定の negative と資産の行の突き合わせを SKIP する` +
      "（recipe の原文との突き合わせは fixture で回る）。生成: cd tools/export-recipes && " +
      "uv run --group wan --inexact python -m wan.text_embeds",
  );
}

Deno.test({
  name:
    "既定の negative: テキスト埋め込み資産の negative の行の原文とビット同一（2 つの経路の既定が同じ文字列）",
  ignore: textEmbedsSeries === undefined,
  fn: () => {
    if (textEmbedsSeries === undefined) throw new Error("ignore の条件と食い違う");
    const embeds = parseWanTextEmbeds(textEmbedsSeries.buffer);
    assertEquals(
      embeds.entries.filter((entry) => entry.role === "negative").map((entry) => entry.prompt),
      [WAN_DEFAULT_NEGATIVE_PROMPT],
    );
  },
});

describe("家族 admission: umT5 のグラフ宣言 × ホストが組む形（入力名・i32・記号 L・幅・格納）", () => {
  type Umt5Spec = {
    readonly inputs: readonly {
      readonly name: string;
      readonly dtype: "i32" | "f32";
      readonly shape: readonly StubDim[];
    }[];
    readonly outputs: readonly {
      readonly name: string;
      readonly dtype: "f32" | "i32";
      readonly shape: readonly StubDim[];
    }[];
    /** initializer の格納（`shared` は格納を持たない共有の宣言）。 */
    readonly storages: readonly (CodecName | "shared")[];
  };
  /** 配布形と同じ宣言（`umt5_patch.py` の入力名・記号 L・出力 `[1, L, 4096]`・i8 の重みと f32 の表）。 */
  const VALID: Umt5Spec = {
    inputs: [
      { name: "input_ids", dtype: "i32", shape: [1, "L"] },
      { name: "relative_position_buckets", dtype: "i32", shape: ["L", "L"] },
    ],
    outputs: [{ name: "out", dtype: "f32", shape: [1, "L", 4096] }],
    storages: ["int8-sym", "f32"],
  };
  const DIT = { contextRows: 512, contextWidth: 4096 };
  const umt5Of = (patch: Partial<Umt5Spec>): GraphOwner => {
    const spec = { ...VALID, ...patch };
    return {
      graph: {
        format: "karume-ir",
        version: 2,
        requires: { ops: [] },
        symbols: ["L", "M"],
        inputs: spec.inputs.map((input) => ({ ...input, shape: [...input.shape] })),
        outputs: spec.outputs.map((output) => output.name),
        initializers: Object.fromEntries(
          spec.storages.map((
            codec,
            index,
          ) => [`w${index}`, codec === "shared" ? { shared: true } : { storage: { codec } }]),
        ),
        values: Object.fromEntries(
          spec.outputs.map((output) => [output.name, {
            dtype: output.dtype,
            shape: [...output.shape],
          }]),
        ),
        states: {},
        nodes: [],
      },
    };
  };
  const ids = VALID.inputs[0];
  const buckets = VALID.inputs[1];

  it("配布形の宣言は通り、出力の名前を宣言から引く", () => {
    assertEquals(umt5Contract(umt5Of({}), DIT, "WanPipeline"), { output: "out" });
  });

  it("入力名・幅・記号・dtype・本数・格納の食い違いは、umT5 を取る前に名指しで落ちる（素の Error）", () => {
    const rejected: readonly [string, Partial<Umt5Spec>, string][] = [
      [
        "入力名違い",
        { inputs: [{ ...ids, name: "token_ids" }, buckets] },
        "グラフ入力 'input_ids' が無い",
      ],
      [
        "幅 2048",
        { outputs: [{ name: "out", dtype: "f32", shape: [1, "L", 2048] }] },
        "出力 'out' の形",
      ],
      [
        "出力の記号が別",
        { outputs: [{ name: "out", dtype: "f32", shape: [1, "M", 4096] }] },
        "出力 'out' の形",
      ],
      [
        "バケット表の記号が別",
        { inputs: [ids, { ...buckets, shape: ["L", "M"] }] },
        "'relative_position_buckets' の形",
      ],
      ["id 列の L が静的", { inputs: [{ ...ids, shape: [1, 128] }, buckets] }, "記号次元でない"],
      ["id 列が f32", { inputs: [{ ...ids, dtype: "f32" }, buckets] }, "i32 でない"],
      [
        "入力が 3 本",
        { inputs: [ids, buckets, { name: "attention_mask", dtype: "i32", shape: [1, "L"] }] },
        "3 本",
      ],
      [
        "出力が 2 本",
        { outputs: [...VALID.outputs, { name: "extra", dtype: "f32", shape: [1, "L", 4096] }] },
        "出力が 2 本",
      ],
      [
        "出力が i32",
        { outputs: [{ name: "out", dtype: "i32", shape: [1, "L", 4096] }] },
        "f32 でない",
      ],
      ["格納 f16", { storages: ["f16", "f32"] }, "格納 f16 は受けない"],
      ["格納 i4", { storages: ["int8-sym", "int4-sym-g"] }, "格納 int4-sym-g は受けない"],
      ["i8 が無い", { storages: ["f32"] }, "i8 の重みが 1 本も無い"],
      ["共有の宣言", { storages: ["int8-sym", "shared"] }, "共有の宣言"],
    ];
    for (const [label, patch, message] of rejected) {
      const error = assertThrows(
        () => umt5Contract(umt5Of(patch), DIT, "WanPipeline"),
        Error,
        message,
        label,
      );
      assert(!(error instanceof ModelInputError), `資産の齟齬を入力起因にしない: ${label}`);
    }
  });

  it("有効長の上限 512 が DiT の文脈の行数を超える組は落ちる", () => {
    assertThrows(
      () => umt5Contract(umt5Of({}), { contextRows: 256, contextWidth: 4096 }, "WanPipeline"),
      Error,
      "有効長の上限 512 が DiT の文脈の行数 256 を超える",
    );
  });
});

/** 構築の入口のテストが使う manifest（`models/karume-wan2.1/karume.json` の骨格 — 宣言だけ）。 */
const wanManifest = (weightNames: readonly string[]) =>
  parseManifest(JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "t2v-1.3b",
    models: {
      "t2v-1.3b": {
        pipeline: "wan/1",
        weights: Object.fromEntries(
          weightNames.map((name) => [name, { f16: declaredContainer(`${name}/model.f16`) }]),
        ),
        assets: {
          text_embeds: { path: "text_embeds.safetensors", size: 8, sha256: "e".repeat(64) },
          umt5_tokenizer: { path: "tokenizer.json", size: 8, sha256: "f".repeat(64) },
        },
        quants: {
          f16: {
            weights: Object.fromEntries(weightNames.map((name) => [name, "f16"])),
            session: {},
          },
        },
        defaultQuant: "f16",
        pipelineConfig: { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } },
      },
    },
  }));
const WITH_UMT5 = ["transformer", "vae_decoder_first", "vae_decoder_next", "text_encoder"];
const WITHOUT_UMT5 = ["transformer", "vae_decoder_first", "vae_decoder_next"];

/** TS の型を通らない値を 1 欄だけ差した構築オプション（JS の呼び手の綴り違いの再現）。 */
const optionsWith = (key: string, value: unknown): WanPipelineOptions => {
  const options: WanPipelineOptions = {};
  Object.defineProperty(options, key, { value, enumerable: true });
  return options;
};

/** 開ける容器（宣言は最小 — admission のグラフの門で落ちる器。umT5 は入れない）。 */
const SHELL_COMPONENTS: Record<string, Uint8Array<ArrayBuffer>> = {};
for (const name of WITHOUT_UMT5) {
  Object.assign(
    SHELL_COMPONENTS,
    partAssets(
      name,
      await tensorlessContainer(name, {
        inputs: [{ name: "x", shape: [1, 4] }],
        output: { name: "y", shape: [1, 4] },
      }),
    ),
  );
}

describe("構築の入口（経路の綴り・umT5 の宣言・取る部品・中断 — GPU も実資産も要らない範囲）", () => {
  it("textEncoder の未知の綴りは資産に触る前に素の Error（model / quant 名の綴り違いと同じ扱い）", async () => {
    const error = await assertRejects(
      () =>
        WanPipeline.fromAssets(
          { manifest: wanManifest(WITH_UMT5), assets: {} },
          optionsWith("textEncoder", "cpu"),
        ),
      Error,
      "textEncoder 'cpu' は 'gpu' / 'precomputed' のどちらでもない",
    );
    assert(!(error instanceof ModelInputError));
  });

  it("既定の gpu の経路で umT5 を宣言しない manifest は、容器を開く前に precomputed を案内して落ちる", async () => {
    await assertRejects(
      () => WanPipeline.fromAssets({ manifest: wanManifest(WITHOUT_UMT5), assets: {} }),
      Error,
      "umT5 の部品 'text_encoder' が無い（textEncoder の既定 \"gpu\" が取る",
    );
  });

  it("precomputed は umT5 の部品を開かずに admission まで進み、gpu は umT5 の容器を要る", async () => {
    // 同じ資産（umT5 の容器だけが無い）で、経路が開く部品の集合だけが違う。precomputed は 3 部品を開いて
    // admission のグラフの門（VAE の chunk グラフ）で落ちる = umT5 に触っていない。
    const manifest = wanManifest(WITH_UMT5);
    await assertRejects(
      () =>
        WanPipeline.fromAssets(
          { manifest, assets: SHELL_COMPONENTS },
          { textEncoder: "precomputed" },
        ),
      WanVaeChunkError,
    );
    await assertRejects(
      () => WanPipeline.fromAssets({ manifest, assets: SHELL_COMPONENTS }),
      Error,
      "部品 'text_encoder' の容器が無い",
    );
  });

  it("admission は VAE の宣言を記述子と照合する（潜在・出口のチャネル数の食い違いは RoPE の素表を読む前に落ちる）", async () => {
    // chunk グラフの取り決めを満たす宣言だけの VAE（潜在タイル 32・縮尺 8・cache 1 本）。DiT は器のまま
    // なので、照合を通り抜けた資産は RoPE の素表の読み出しで別の文言で落ちる（照合の配線を縛る対照）。
    const vae = async (name: string, frames: number, latent: number, sample: number) =>
      partAssets(
        name,
        await writeContainer({
          graphs: {
            [name]: parseIrDeclarationValue({
              format: "karume-ir",
              version: 2,
              requires: { ops: ["mul"] },
              symbols: [],
              inputs: [
                { name: "latent", dtype: "f32", shape: [latent, 1, 32, 32] },
                { name: "cache_00", dtype: "f32", shape: [16, 2, 32, 32] },
              ],
              outputs: ["frame", "cache_00_out"],
              initializers: {},
              values: {
                frame: { dtype: "f32", shape: [sample, frames, 256, 256] },
                cache_00_out: { dtype: "f32", shape: [16, 2, 32, 32] },
              },
              states: {},
              nodes: [
                { op: "mul", ins: ["latent", "latent"], outs: ["frame"], attrs: {} },
                { op: "mul", ins: ["cache_00", "cache_00"], outs: ["cache_00_out"], attrs: {} },
              ],
            }),
          },
          consts: [],
          weights: [],
          assets: [],
          provenance: { license: "test" },
        }),
      );
    const build = async (latent: number, sample: number) =>
      WanPipeline.fromAssets(
        {
          manifest: wanManifest(WITHOUT_UMT5),
          assets: {
            ...SHELL_COMPONENTS,
            ...await vae("vae_decoder_first", 1, latent, sample),
            ...await vae("vae_decoder_next", 4, latent, sample),
          },
        },
        { textEncoder: "precomputed" },
      );
    const rejected: readonly [string, number, number, string][] = [
      ["出口 12", 16, 12, "WanPipeline: VAE の出口 12 チャネルが RGB 3 × unpatchify 1² = 3 と違う"],
      [
        "潜在 8",
        8,
        3,
        "WanPipeline: 逆正規化の統計（mean 16 本・std 16 本）が VAE の潜在 8 チャネルと違う",
      ],
    ];
    for (const [label, latent, sample, message] of rejected) {
      await assertRejects(() => build(latent, sample), Error, message, label);
    }
    // 対照: 配布形の宣言（潜在 16・出口 3）は照合とタイルの門を通り、器の DiT の RoPE の素表で落ちる。
    await assertRejects(() => build(16, 3), Error, "'rope_base'");
  });

  it("中断済みの signal は資産へ触る前に reason そのままで reject する", async () => {
    // signal 無しなら「部品 'transformer' の容器が無い」で落ちる形が、中断済みなら reason で落ちる。
    const controller = new AbortController();
    const reason = new Error("中止ボタン");
    controller.abort(reason);
    const error = await assertRejects(() =>
      WanPipeline.fromAssets(
        { manifest: wanManifest(WITH_UMT5), assets: {} },
        { signal: controller.signal },
      )
    );
    assertStrictEquals(error, reason);
  });

  it("実行開始後に届いた中断も最初の段の境目で効く（イベントループへ譲ってから見る）", async () => {
    const controller = new AbortController();
    const reason = new Error("中止ボタン（実行中）");
    setTimeout(() => controller.abort(reason), 0);
    const error = await assertRejects(() =>
      WanPipeline.fromAssets(
        { manifest: wanManifest(WITH_UMT5), assets: {} },
        { signal: controller.signal },
      )
    );
    assertStrictEquals(error, reason);
  });
});

/** 配布形ミラーの manifest（`dist.py --pipeline wan` が書く — 無い機では下の 1 本だけ SKIP）。 */
const DIST_MANIFEST = new URL("../../../models/karume-wan2.1/karume.json", import.meta.url);
const distManifestText = await readTextIfPresent(DIST_MANIFEST);
if (distManifestText === undefined) {
  console.warn(
    `[karume] ${DIST_MANIFEST.pathname} が無いため、quant を省いた構築が実用席へ解決する検査を SKIP する。` +
      "組み立て: cd tools/export-recipes && uv run python dist.py --pipeline wan",
  );
}

/** 既定席（ADR 0120 裁定 2026-10-04 の 4 — 今いちばん実用的な席。正本は recipe の `WAN_DEFAULT_QUANT`）。 */
const PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";

Deno.test({
  name:
    "配布形の既定席: quant を省いた構築は実用席へ解決する（manifest の karume.json だけを読む）",
  ignore: distManifestText === undefined,
  fn: async () => {
    assert(distManifestText !== undefined);
    // 席の解決は家族 admission（重みを読む前）で決まる。limits が全部 0 の共有 GPU を渡すと、どの席も宣言する
    // requiredLimits（umT5 の語彙埋め込み）で落ち、その文言が解決した席を名指す — 重みも GPU も使わずに席を観測する。
    await assertRejects(
      () =>
        WanPipeline.fromAssets(
          { manifest: parseManifest(distManifestText), assets: SHELL_COMPONENTS },
          { textEncoder: "precomputed", gpu: fakeGpuContext(fakeDevice()) },
        ),
      Error,
      `WanPipeline: quant '${PRACTICAL_QUANT}' が要求する device limit`,
    );
  },
});

describe("WanPipeline.generate（模擬 Session・GPU 経路の text 段）", () => {
  it("text 段は positive → negative を回して畳み、畳んだ後に DiT の段を張る（Session の実行オプションは {}）", async () => {
    const generated = mockPipeline({ textEncoder: "gpu" });
    await generated.generate();
    assertEquals(generated.log.slice(0, 7), [
      "text_encoder:start",
      "create:text_encoder:{}",
      "run:text_encoder:28",
      "run:text_encoder:126",
      "dispose:text_encoder",
      "text_encoder:end",
      "transformer:start",
    ]);
    assertEquals(generated.log.at(-1), "vae_decoder:end");
  });

  it("DiT へ渡す文脈は umT5 の出力を 512 行までゼロで詰めたもの（uncond = negative・cond = positive）", async () => {
    const { generate, ditContexts } = mockPipeline({ textEncoder: "gpu" });
    await generate();
    // 2 step × (uncond, cond)。
    assertEquals(ditContexts.length, 4);
    const rowsOf = (context: Float32Array, value: number) => {
      const filled = context.findIndex((entry) => entry !== Math.fround(value));
      return filled === -1 ? context.length / WIDTH : filled / WIDTH;
    };
    const [uncond, cond] = ditContexts;
    assertEquals(uncond.length, 512 * WIDTH);
    assertEquals(rowsOf(uncond, 126 / 1000), 126, "uncond は negative（126 トークン）の出力");
    assertEquals(rowsOf(cond, 28 / 1000), 28, "cond は positive（28 トークン）の出力");
    assert(uncond.subarray(126 * WIDTH).every((entry) => entry === 0), "有効長の後ろがゼロでない");
    assert(cond.subarray(28 * WIDTH).every((entry) => entry === 0), "有効長の後ろがゼロでない");
  });

  it("guidance 1 は negative を回さない（umT5 の run は 1 回）", async () => {
    const { log, generate } = mockPipeline({ textEncoder: "gpu" });
    await generate(undefined, { guidance: 1 });
    assertEquals(log.filter((entry) => entry.startsWith("run:text_encoder")), [
      "run:text_encoder:28",
    ]);
  });

  it("umT5 の出力が非有限なら text 段で名指しで落ち、Session を畳み、DiT の段を張らない", async () => {
    const { log, generate } = mockPipeline({
      textEncoder: "gpu",
      umt5Value: (tokens) => (tokens === 126 ? Number.NaN : 0.5),
    });
    await assertRejects(() => generate(), Error, "umT5 の出力（negativePrompt・126 トークン）");
    assertEquals(log.at(-1), "dispose:text_encoder");
    assert(!log.includes("text_encoder:end"), "途中で落ちた段の end を出した");
    assert(!log.includes("create:transformer"), `DiT の段へ進んだ: ${log}`);
  });

  it("入口の門（前処理の拒否）は Session を 1 本も張らずに落ちる", async () => {
    const { log, generate } = mockPipeline({ textEncoder: "gpu" });
    await assertRejects(() => generate(undefined, { prompt: "R&D lab" }), PromptCleanError);
    assertEquals(log.filter((entry) => entry.startsWith("create:")), []);
  });
});

describe("WanPipeline.generate の中断（模擬 Session — signal.reason を包まず・Session を畳んで・次が回る）", () => {
  it("step 1 の後の中断は step 2 の前で効き、DiT の Session を畳み、VAE の段を張らない・次の generate は回る", async () => {
    const { log, generate } = mockPipeline({});
    const controller = new AbortController();
    const reason = new Error("中止ボタン");
    const error = await assertRejects(() =>
      generate((event) => {
        if (event.kind === "denoise-step" && event.step === 1) controller.abort(reason);
      }, { signal: controller.signal })
    );
    assertStrictEquals(error, reason);
    assert(!log.includes("step:2"), `中断の後に step を回した: ${log}`);
    assertEquals(log.at(-1), "dispose:transformer");
    assert(!log.includes("vae_decoder:start"), `VAE の段へ進んだ: ${log}`);
    log.length = 0;
    await generate();
    assertEquals(log.at(-1), "vae_decoder:end", "中断の後の generate が最後まで回らない");
  });

  it("タスクで届く中断（timer）も次の境目で効く — 譲らない検査では最後まで回ってしまう形", async () => {
    const { log, generate } = mockPipeline({});
    const controller = new AbortController();
    const reason = new Error("中止ボタン（timer）");
    const error = await assertRejects(() =>
      generate((event) => {
        if (event.kind === "denoise-step" && event.step === 1) {
          setTimeout(() => controller.abort(reason), 0);
        }
      }, { signal: controller.signal })
    );
    assertStrictEquals(error, reason);
    assert(!log.includes("step:2"), `タスクで届いた中断を step 2 の前で見ていない: ${log}`);
  });

  it("text 段の中の中断は次の run の前で効き、umT5 の Session を畳んで DiT の段を張らない", async () => {
    const controller = new AbortController();
    const reason = new Error("中止ボタン（text 段）");
    const { log, generate } = mockPipeline({
      textEncoder: "gpu",
      onUmt5Run: (tokens) => {
        if (tokens === 28) controller.abort(reason);
      },
    });
    const error = await assertRejects(() => generate(undefined, { signal: controller.signal }));
    assertStrictEquals(error, reason);
    assertEquals(log.filter((entry) => entry.startsWith("run:text_encoder")), [
      "run:text_encoder:28",
    ]);
    assertEquals(log.at(-1), "dispose:text_encoder");
    assert(!log.includes("create:transformer"), `DiT の段へ進んだ: ${log}`);
  });

  it("中断済みの signal は Session を 1 本も張らずに reason を投げる（どちらの経路も）", async () => {
    for (const textEncoder of [undefined, "gpu"] as const) {
      const { log, generate } = mockPipeline(textEncoder === undefined ? {} : { textEncoder });
      const controller = new AbortController();
      const reason = new Error("中止ボタン（開始前）");
      controller.abort(reason);
      const error = await assertRejects(() => generate(undefined, { signal: controller.signal }));
      assertStrictEquals(error, reason);
      assertEquals(log.filter((entry) => entry.startsWith("create:")), [], `${textEncoder}`);
    }
  });

  it("VAE のタイルの間の中断は次のタイルの前で効き、VAE の Session を畳む", async () => {
    const { log, generate } = mockPipeline({});
    const controller = new AbortController();
    const reason = new Error("中止ボタン（VAE）");
    const error = await assertRejects(() =>
      generate((event) => {
        if (event.kind === "vae-tile" && event.tile === 1) controller.abort(reason);
      }, { signal: controller.signal })
    );
    assertStrictEquals(error, reason);
    assertEquals(log.filter((entry) => entry.startsWith("tile:")), ["tile:1"]);
    assert(log.includes("dispose:vae_decoder_first") && log.includes("dispose:vae_decoder_next"));
    assert(!log.includes("vae_decoder:end"), `中断した段の end を出した: ${log}`);
  });
});
