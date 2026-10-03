// Wan2.1 のパイプラインの GPU 不要の部分（`src/wan/`）— 入力の門・`pipelineConfig` の門・テキスト埋め込み
// 資産の検査・潜在の逆正規化・フレームの RGBA 化・乱数。実 GPU の通しは e2e_wan_pipeline_test.ts。

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { ModelInputError } from "../src/errors.ts";
import {
  ACCEPTED_SIZES,
  type GeneratedVideo,
  MAX_FRAMES,
  MIN_FRAMES,
  planWanGeneration,
  type WanGenerateRequest,
} from "../src/wan/pipeline.ts";
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
/** 832×480 の VAE の縮尺（chunk グラフの `8t / t`）。 */
const GEOMETRY = { spatialScale: 8 };
/** 配布形の `pipelineConfig`（recipe `wan/distribution.py` の `WAN_PIPELINE_CONFIG` — 参照の設定）。 */
const CONFIG: WanPipelineConfig = { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } };
const plan = (request: Partial<WanGenerateRequest>) =>
  planWanGeneration({ prompt: "Two cats.", ...request }, EMBEDS, GEOMETRY, CONFIG);

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
      () => planWanGeneration({ prompt: "Two cats." }, embeds, GEOMETRY, CONFIG),
      ModelInputError,
      "negative の行が 0 本",
    );
    assertEquals(
      planWanGeneration({ prompt: "Two cats.", guidance: 1 }, embeds, GEOMETRY, CONFIG).negative,
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
    const resolved = planWanGeneration({ prompt: "Two cats." }, EMBEDS, GEOMETRY, config);
    assertEquals([resolved.steps, resolved.guidance, resolved.shift], [23, 4.25, 7.5]);
    const explicit = planWanGeneration(
      { prompt: "Two cats.", steps: 2, guidance: 1, shift: 1.5 },
      EMBEDS,
      GEOMETRY,
      config,
    );
    assertEquals([explicit.steps, explicit.guidance, explicit.shift], [2, 1, 1.5]);
    assertEquals(explicit.negative, undefined, "guidance 1 の明示は既定の 4.25 に勝つ");
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
});
