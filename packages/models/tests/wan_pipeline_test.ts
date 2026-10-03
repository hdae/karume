// Wan2.1 のパイプラインの GPU 不要の部分（`src/wan/`）— 入力の門・`pipelineConfig` の門・家族 admission の
// グラフ宣言の門・テキスト埋め込み資産の検査・潜在の逆正規化・フレームの RGBA 化・乱数・模擬 Session で回す
// `generate`（後始末と非有限の門）。実 GPU の通しは e2e_wan_pipeline_test.ts。

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertNotEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import type { Tensor } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import {
  ACCEPTED_SIZES,
  assertWanVaeTilesCoverAcceptedSizes,
  ditContract,
  type GeneratedVideo,
  MAX_FRAMES,
  MIN_FRAMES,
  planWanGeneration,
  type WanGenerateRequest,
  WanPipeline,
} from "../src/wan/pipeline.ts";
import { WAN_UNIPC_CONFIG, wanUniPcSchedule } from "../src/wan/scheduler.ts";
import { wanVaeChunkLayout } from "../src/wan/vae-chunks.ts";
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
/** 配布形の VAE の chunk グラフの幾何（潜在タイル 32・縮尺 8 — `8t / t`）。 */
const LAYOUT = { latentChannels: 16, tile: 32, sampleTile: 256 };
/** 配布形の `pipelineConfig`（recipe `wan/distribution.py` の `WAN_PIPELINE_CONFIG` — 参照の設定）。 */
const CONFIG: WanPipelineConfig = { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } };
const plan = (request: Partial<WanGenerateRequest>) =>
  planWanGeneration({ prompt: "Two cats.", ...request }, EMBEDS, LAYOUT, CONFIG);

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
      () => planWanGeneration({ prompt: "Two cats." }, embeds, LAYOUT, CONFIG),
      ModelInputError,
      "negative の行が 0 本",
    );
    assertEquals(
      planWanGeneration({ prompt: "Two cats.", guidance: 1 }, embeds, LAYOUT, CONFIG).negative,
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
    const resolved = planWanGeneration({ prompt: "Two cats." }, EMBEDS, LAYOUT, config);
    assertEquals([resolved.steps, resolved.guidance, resolved.shift], [23, 4.25, 7.5]);
    const explicit = planWanGeneration(
      { prompt: "Two cats.", steps: 2, guidance: 1, shift: 1.5 },
      EMBEDS,
      LAYOUT,
      config,
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
 * `vae-chunks.ts` の取り決めの形）。
 */
const vaeChunkGraphs = (tile: number, scale = 8) => {
  const graph = (frames: number) =>
    stubModel({
      inputs: [
        { name: "latent", shape: [16, 1, tile, tile] },
        { name: "cache_00", shape: [16, 2, tile, tile] },
      ],
      outputs: ["frame", "cache_00_out"],
      values: {
        frame: [3, frames, scale * tile, scale * tile],
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
    for (const tile of [32, 60]) assertWanVaeTilesCoverAcceptedSizes(layoutOf(tile));
  });

  it("chunk グラフの検査は通るがタイル decode できない資産を、寸法と理由を言って拒む", () => {
    // 8: 重なりの下限 8 がタイル幅未満にならない。64: 潜在の短辺 60 より大きい。
    for (const [tile, reason] of [[8, "最小の重なり"], [64, "タイル幅 64 より小さい"]] as const) {
      const error = assertThrows(
        () => assertWanVaeTilesCoverAcceptedSizes(layoutOf(tile)),
        Error,
        reason,
      );
      assert(error.message.includes("832×480"), error.message);
      assert(!(error instanceof ModelInputError), "資産の齟齬を入力起因にしない");
    }
  });

  it("縮尺で受理する寸法の潜在が整数にならない資産を拒む", () => {
    assertThrows(
      () => assertWanVaeTilesCoverAcceptedSizes(layoutOf(32, 7)),
      Error,
      "整数にならない",
    );
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
    assertEquals(ditContract(transformerOf({}), ROPE), {
      output: "out",
      projWidth: 256,
      contextRows: 512,
      contextWidth: 4096,
    });
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
      assertThrows(() => ditContract(transformerOf(patch), ROPE), Error, message, label);
    }
  });
});

describe("モデルカードの受理集合（fixture を挟んだ突き合わせ）", () => {
  // 反対側は recipe の `wan/tests/test_distribution.py`（card.py の表を同じ fixture と比べる）。
  it("受理する寸法とフレーム数の範囲は fixture wan-card-limits.json と同じ", async () => {
    const fixture: unknown = JSON.parse(
      await Deno.readTextFile(new URL("./fixtures/wan-card-limits.json", import.meta.url)),
    );
    assertEquals(
      fixture,
      { acceptedSizes: ACCEPTED_SIZES, minFrames: MIN_FRAMES, maxFrames: MAX_FRAMES },
      "pipeline.ts の受理集合を変えたら fixture と card.py の WAN_ACCEPTED_SIZES / WAN_FRAMES も揃える",
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
    const got = denormalizeWanLatents(latents);
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
    assertThrows(() => denormalizeWanLatents(new Float32Array(17)), Error, "割り切れない");
  });
});

describe("フレームの RGBA 化", () => {
  const video: GeneratedVideo = {
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
 * krm コンテナのバイト列と GPU の取得を要り、CPU の単体テストでは回せない。内部状態の形
 * （pipeline.ts の `WanState`）は export していないので、ここで組む欄は手で揃える — 欄が欠ければ
 * generate の中の TypeError で落ち、各テストが見る文言と食い違って赤になる。
 */
const mockPipeline = (options: {
  readonly ditValue?: number;
  readonly vaeValue?: number;
  readonly ditDisposeError?: Error;
}) => {
  const log: string[] = [];
  const { first, next } = vaeChunkGraphs(32);
  const resident = (byteLength: number) => ({ byteLength, write: () => {}, dispose: () => {} });
  const vaeSession = (name: string) => ({
    createSession: () => {
      log.push(`create:${name}`);
      return Promise.resolve({
        enqueue: () => Promise.resolve(),
        diagnostics: () => ({}),
        dispose: () => Promise.resolve(),
      });
    },
  });
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
    dit: { output: "out", projWidth: 256, contextRows: 4, contextWidth: WIDTH },
    textEmbeds: EMBEDS,
  };
  const pipeline: WanPipeline = Reflect.construct(WanPipeline, [state]);
  /** 5 フレーム・2 step（DiT 4 回・VAE 12 タイル）の要求で回し、観測したイベントを `log` に積む。 */
  const generate = (onEvent?: WanGenerateRequest["onEvent"]) =>
    pipeline.generate({
      prompt: "Two cats.",
      frames: 5,
      steps: 2,
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
  return { log, generate };
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
