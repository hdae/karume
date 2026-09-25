// 家族 admission（`admitIrodori`）の**グラフ突合 13 点**の失敗経路。GPU も実資産も要らない。
//
// 突合の中身は `assertStaticDim` × 10 / `assertOutputScale` × 1 / `assertOutputDim` × 1 /
// 記号次元が 1 本 × 1 で、どれも doc に「MUST: 落とさない。…**沈黙誤値**が出る」と書かれた門
// （shape は合ったまま別の位置の条件を読む形）。`irodori_pipeline_test.ts` は「合成の容器を
// 組む器が無い」ため意図的にこの層を外しているが、器は `tests/helpers/container-fixture.ts`
// に置いたので、いまは実資産なしで踏める。
//
// 観測の仕掛け: 資産は**グラフ 8 本だけ**を渡し、`tokenizer` を入れない。
//  - 正しい 8 本 → 落ちるのは `資産 'tokenizer' が無い`（= 突合を全部通過して次の段へ進んだ）
//  - 1 軸だけ壊す → その軸名・期待値・実測値を含む文言で reject
// の対偶で、門そのものと門の位置（GPU を取りに行く前）を同時に縛る。GPU へは 1 度も触らない。

import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { IrodoriPipeline } from "../src/irodori/pipeline.ts";
import {
  declaredContainer,
  partAssets,
  tensorlessContainer,
  type TensorlessGraphSpec,
  tensorlessInput,
} from "./helpers/container-fixture.ts";
import {
  HUB_URL,
  MANIFEST_PATH,
  REPO,
  serveContainer,
  serveRepos,
  SHA,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** 部品差し替え先の疑似リポ。 */
const OTHER_REPO = "karume-test/replacement";

/** `models/karume-irodori-v4-small/karume.json` の `pipelineConfig` 実物（23 欄）。 */
const CONFIG = {
  maxTextLen: 256,
  maxCaptionLen: 512,
  speakerRows: 751,
  ditSymMax: 750,
  frameRate: 25,
  sampleRate: 48000,
  hopLength: 1920,
  codecHaloFrames: 8,
  latentDim: 32,
  speakerPatchSize: 4,
  speakerDim: 768,
  textDim: 512,
  captionDim: 512,
  timestepEmbedDim: 512,
  steps: 40,
  initScale: 0.999,
  cfgMinT: 0.5,
  cfgMaxT: 1,
  cfgScales: { text: 3, speaker: 5, caption: 3 },
  minSeconds: 0.5,
  maxSeconds: 30,
  speakerUncondMode: "mask",
  cfgGuidanceMode: "independent",
};

const COMPONENTS = [
  "backbone",
  "text_proj",
  "caption_proj",
  "speaker",
  "duration",
  "dit",
  "codec_decoder",
  "codec_encoder",
] as const;

type Component = typeof COMPONENTS[number];

/** 突合を全て通る最小グラフ 8 本（`CONFIG` の値がそのまま宣言に出る）。 */
const graphSpecs = (): Record<Component, TensorlessGraphSpec> => ({
  // 突合の対象外（宣言は何でもよい）。
  backbone: {
    inputs: [{ name: "input_ids", shape: [1, 4] }],
    output: { name: "hidden", shape: [1, 4, CONFIG.textDim] },
  },
  text_proj: {
    inputs: [{ name: "hidden", shape: [1, 4, CONFIG.textDim] }],
    output: { name: "text_state", shape: [1, 4, CONFIG.textDim] },
  },
  caption_proj: {
    inputs: [{ name: "hidden", shape: [1, 4, CONFIG.captionDim] }],
    output: { name: "caption_state", shape: [1, 4, CONFIG.captionDim] },
  },
  // 参照 latent の patch 幅（latentDim × speakerPatchSize）。
  speaker: {
    inputs: [{ name: "latent", shape: [1, 4, CONFIG.latentDim * CONFIG.speakerPatchSize] }],
    output: { name: "speaker_vec", shape: [1, CONFIG.speakerDim] },
  },
  duration: {
    inputs: [
      { name: "text_state", shape: [1, CONFIG.maxTextLen, CONFIG.textDim] },
      { name: "speaker_vec", shape: [1, CONFIG.speakerDim] },
      { name: "caption_vec", shape: [1, CONFIG.captionDim] },
    ],
    output: { name: "log_frames", shape: [1] },
  },
  dit: {
    symbols: ["S"],
    inputs: [
      { name: "x_t", shape: [1, "S", CONFIG.latentDim] },
      { name: "t_embed", shape: [1, CONFIG.timestepEmbedDim] },
      { name: "text_state", shape: [1, CONFIG.maxTextLen, CONFIG.textDim] },
      { name: "speaker_state", shape: [1, CONFIG.speakerRows, CONFIG.speakerDim] },
      { name: "caption_state", shape: [1, CONFIG.maxCaptionLen, CONFIG.captionDim] },
    ],
    output: { name: "v", shape: [1, "S", CONFIG.latentDim] },
  },
  // 1 latent フレーム → hopLength サンプル（出力の派生次元の**係数**まで見る）。
  codec_decoder: {
    symbols: ["S"],
    inputs: [{ name: "latent", shape: [1, "S", CONFIG.latentDim] }],
    output: { name: "wav", shape: [1, 1, `${CONFIG.hopLength}S`] },
  },
  codec_encoder: {
    symbols: ["T"],
    inputs: [{ name: "wav", shape: [1, "T", CONFIG.hopLength] }],
    output: { name: "latent", shape: [1, "T", CONFIG.latentDim] },
  },
});

/** 1 本だけ差し替えた容器 8 本を組む（`tokenizer` は入れない — 観測の仕掛け）。 */
const assetsWith = async (
  patch: Partial<Record<Component, TensorlessGraphSpec>> = {},
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const specs = { ...graphSpecs(), ...patch };
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const name of COMPONENTS) {
    assets = { ...assets, ...partAssets(name, await tensorlessContainer(name, specs[name])) };
  }
  return assets;
};

const FILE = { size: 16, sha256: "a".repeat(64) };

const manifestText = (): string => {
  let weights: Record<string, unknown> = {};
  let mapping: Record<string, string> = {};
  for (const name of COMPONENTS) {
    weights = { ...weights, [name]: { f32: declaredContainer(`${name}/model.f32`) } };
    mapping = { ...mapping, [name]: "f32" };
  }
  return JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "v4-small",
    models: {
      "v4-small": {
        pipeline: "irodori/1",
        weights,
        assets: { tokenizer: { ...FILE, path: "tokenizer.json" } },
        quants: { f32: { weights: mapping, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: CONFIG,
      },
    },
  });
};

const build = async (
  patch: Partial<Record<Component, TensorlessGraphSpec>> = {},
): Promise<unknown> =>
  await IrodoriPipeline.fromAssets({
    manifest: parseManifest(manifestText()),
    assets: await assetsWith(patch),
  });

/** グラフ 1 本の入力 1 本の 1 軸だけを壊す。 */
const breakInput = (
  component: Component,
  inputName: string,
  axis: number,
  value: number | string,
): Partial<Record<Component, TensorlessGraphSpec>> => {
  const spec = graphSpecs()[component];
  return {
    [component]: {
      ...spec,
      inputs: spec.inputs.map((input) =>
        input.name === inputName
          ? { ...input, shape: input.shape.map((dim, index) => index === axis ? value : dim) }
          : input
      ),
    },
  };
};

/** グラフ 1 本の出力の 1 軸だけを壊す。 */
const breakOutput = (
  component: Component,
  axis: number,
  value: number | string,
): Partial<Record<Component, TensorlessGraphSpec>> => {
  const spec = graphSpecs()[component];
  return {
    [component]: {
      ...spec,
      output: {
        ...spec.output,
        shape: spec.output.shape.map((dim, index) => index === axis ? value : dim),
      },
    },
  };
};

Deno.test("admitIrodori: 突合 13 点を全て満たすグラフは資産の段まで進む（門の位置の対偶）", async () => {
  // `tokenizer` を渡していないので、突合を全部通れば落ちるのはその 1 本。ここが
  // 「グラフ入力 '…' の軸 …」で落ちるなら、正常系のはずの宣言が門に引っかかっている。
  await assertRejects(() => build(), Error, "irodori: 資産 'tokenizer' が無い");
});

Deno.test("admitIrodori: 壊した軸ごとに軸名・期待値・実測値を出して落ちる（12 点）", async () => {
  // 各ケースは 1 軸だけを現行値の近傍へずらす（shape の rank は保つ）— rank を変えると
  // 別の門に当たって「この軸の突合が生きている」ことを示せない。
  const cases: readonly {
    readonly where: string;
    readonly patch: Partial<Record<Component, TensorlessGraphSpec>>;
    readonly actual: string;
    readonly expected: string;
  }[] = [
    {
      where: "latentDim",
      patch: breakInput("dit", "x_t", 2, 33),
      actual: "'x_t' の軸 2 が 33",
      expected: "pipelineConfig は 32",
    },
    {
      where: "timestepEmbedDim",
      patch: breakInput("dit", "t_embed", 1, 511),
      actual: "'t_embed' の軸 1 が 511",
      expected: "pipelineConfig は 512",
    },
    {
      where: "maxTextLen",
      patch: breakInput("dit", "text_state", 1, 255),
      actual: "'text_state' の軸 1 が 255",
      expected: "pipelineConfig は 256",
    },
    {
      where: "textDim",
      patch: breakInput("dit", "text_state", 2, 511),
      actual: "'text_state' の軸 2 が 511",
      expected: "pipelineConfig は 512",
    },
    {
      where: "speakerRows",
      patch: breakInput("dit", "speaker_state", 1, 750),
      actual: "'speaker_state' の軸 1 が 750",
      expected: "pipelineConfig は 751",
    },
    {
      where: "speakerDim",
      patch: breakInput("dit", "speaker_state", 2, 767),
      actual: "'speaker_state' の軸 2 が 767",
      expected: "pipelineConfig は 768",
    },
    {
      where: "maxCaptionLen",
      patch: breakInput("dit", "caption_state", 1, 511),
      actual: "'caption_state' の軸 1 が 511",
      expected: "pipelineConfig は 512",
    },
    {
      where: "captionDim",
      patch: breakInput("dit", "caption_state", 2, 511),
      actual: "'caption_state' の軸 2 が 511",
      expected: "pipelineConfig は 512",
    },
    {
      where: "textDim",
      patch: breakInput("duration", "text_state", 2, 511),
      actual: "'text_state' の軸 2 が 511",
      expected: "pipelineConfig は 512",
    },
    {
      where: "speakerDim",
      patch: breakInput("duration", "speaker_vec", 1, 767),
      actual: "'speaker_vec' の軸 1 が 767",
      expected: "pipelineConfig は 768",
    },
    {
      where: "captionDim",
      patch: breakInput("duration", "caption_vec", 1, 511),
      actual: "'caption_vec' の軸 1 が 511",
      expected: "pipelineConfig は 512",
    },
    {
      where: "latentDim × speakerPatchSize",
      patch: breakInput("speaker", "latent", 2, 64),
      actual: "'latent' の軸 2 が 64",
      expected: "pipelineConfig は 128",
    },
  ];

  for (const testCase of cases) {
    const error = await assertRejects(() => build(testCase.patch), Error);
    assertStringIncludes(error.message, testCase.where);
    assertStringIncludes(error.message, testCase.actual);
    assertStringIncludes(error.message, testCase.expected);
  }
  assertEquals(cases.length, 12);
});

Deno.test("admitIrodori: codec_decoder の latent 幅と出力倍率を別々に見る", async () => {
  // 入力幅（latentDim）。
  const width = await assertRejects(
    () => build(breakInput("codec_decoder", "latent", 2, 31)),
    Error,
    "latentDim",
  );
  assertStringIncludes(width.message, "'latent' の軸 2 が 31");

  // 出力の**派生次元の係数**。shape は「それらしい長さの波形」のままなので、ここだけが
  // 「1 フレーム → hopLength サンプル」の破れを捕まえる（秒指定の切り出しと末尾トリムが
  // 静かに別のサンプル位置を指す形）。
  const scale = await assertRejects(
    () => build(breakOutput("codec_decoder", 2, "960S")),
    Error,
    "hopLength",
  );
  assertStringIncludes(scale.message, "軸 2 が 960S");
  assertStringIncludes(scale.message, "期待は '1920S'");
});

Deno.test("admitIrodori: codec_encoder の入力フレーム幅と出力 latent 幅を別々に見る", async () => {
  const frame = await assertRejects(
    () => build(breakInput("codec_encoder", "wav", 2, 1919)),
    Error,
    "hopLength",
  );
  assertStringIncludes(frame.message, "'wav' の軸 2 が 1919");

  const latent = await assertRejects(
    () => build(breakOutput("codec_encoder", 2, 31)),
    Error,
    "latentDim",
  );
  assertStringIncludes(latent.message, "グラフ出力 'latent' の軸 2 が 31");
});

Deno.test("admitIrodori: dit の記号次元が 1 本でなければ落とす（経路に依らず同じ文言）", async () => {
  // 常駐経路は毎 enqueue この記号名で S を束縛する。実行時に置くと ①重みを落とした後にしか
  // 落ちない ②ホスト経路（`gpuTiming` 有効 device / `onEvent` 購読）では走らない、の 2 つが
  // 起きる（同じ配布形が観測経路ごとに違う文言で落ちる）。
  const specs = graphSpecs();
  const twoSymbols: TensorlessGraphSpec = {
    ...specs.dit,
    symbols: ["S", "B"],
    inputs: [
      { name: "x_t", shape: ["B", "S", CONFIG.latentDim] },
      ...specs.dit.inputs.slice(1),
    ],
    output: { name: "v", shape: ["B", "S", CONFIG.latentDim] },
  };
  // 並びが宣言順（`S, B`）でないのは、容器のグラフ記述が**正準形**（記号は符号位置順 —
  // container-v1 / `canonicalIrDocument`）で焼かれるため。文言はその読み戻しを名乗る。
  await assertRejects(
    () => build({ dit: twoSymbols }),
    Error,
    "dit の記号次元が 1 本でない（[B, S]）",
  );
});

Deno.test("fromPretrained: components で差した dit の次元が違えば重みを取る前に落ちる（差し替え席が配線されている）", async () => {
  // 差し替え先の dit だけ latentDim を 64 に焼く（元リポの 8 本は突合 13 点を全て通る）。
  // 差し替え席を配線し忘れると、元リポのまま突合を通って資産の取得へ進み、別の文言で落ちる。
  const specs = graphSpecs();
  let weights: Record<string, unknown> = {};
  let files: (readonly [string, Uint8Array<ArrayBuffer>])[] = [];
  for (const name of COMPONENTS) {
    const served = await serveContainer(`${name}/model.f32`, tensorlessInput(name, specs[name]));
    weights = { ...weights, [name]: { f32: served.entry } };
    files = [...files, ...served.files];
  }
  const wide = [1, "S", CONFIG.latentDim * 2];
  const widened: TensorlessGraphSpec = {
    ...specs.dit,
    inputs: specs.dit.inputs.map((input) =>
      input.name === "x_t" ? { ...input, shape: wide } : input
    ),
    output: { ...specs.dit.output, shape: wide },
  };
  const replacement = await serveContainer("other/dit.f32", tensorlessInput("dit", widened));
  const modelOf = (entries: Record<string, unknown>) => ({
    test: {
      pipeline: "irodori/1",
      weights: entries,
      assets: { tokenizer: { ...FILE, path: "tokenizer.json" } },
      quants: {
        f32: {
          weights: Object.fromEntries(Object.keys(entries).map((key) => [key, "f32"])),
          session: {},
        },
      },
      defaultQuant: "f32",
      pipelineConfig: CONFIG,
    },
  });
  const mock = serveRepos([
    { repo: REPO, models: modelOf(weights), files },
    {
      repo: OTHER_REPO,
      models: modelOf({ dit: { f32: replacement.entry } }),
      files: replacement.files,
    },
  ]);

  const error = await assertRejects(
    () =>
      IrodoriPipeline.fromPretrained({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, {
        fetch: mock.fetch,
        caches: new MemoryCacheStorage(),
        components: { dit: { source: { repo: OTHER_REPO, revision: SHA, hubUrl: HUB_URL } } },
      }),
    Error,
    "グラフ記述が manifest の宣言と違う",
  );
  assertStringIncludes(error.message, "dit");
  // 差し替え先の manifest を引いた（= 席が loader まで届いた）。検査は 2 つの manifest の宣言
  // だけで済むので、どちらのリポの容器も 1 本も取っていない（資産の tokenizer にも進んでいない）。
  assertEquals(
    mock.requests.filter((request) => request.repo === OTHER_REPO),
    [{ repo: OTHER_REPO, path: MANIFEST_PATH }],
  );
  assertEquals(mock.paths.filter((path) => path !== MANIFEST_PATH), []);
});
