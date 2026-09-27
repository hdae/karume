// 家族 admission（`admitIrodori`）の**グラフ突合**の失敗経路。GPU も実資産も要らない。
//
// 突合の中身は `assertStaticDim` / `assertOutputScale` / `assertOutputDim` / `dit` の記号次元が
// 1 本 / `dit_context` → `dit` の配線（記号次元なし・K / V の対・名前の集合・形と dtype —
// ADR 0114）で、どれも doc に「MUST: 落とさない。…**沈黙誤値**が出る」と書かれた門
// （shape は合ったまま別の位置の条件を読む形）。`irodori_pipeline_test.ts` は「合成の容器を
// 組む器が無い」ため意図的にこの層を外しているが、器は `tests/helpers/container-fixture.ts`
// に置いたので、いまは実資産なしで踏める。
//
// 観測の仕掛け: 資産は**グラフ 9 本だけ**を渡し、`tokenizer` を入れない。
//  - 正しい 9 本 → 落ちるのは `資産 'tokenizer' が無い`（= 突合を全部通過して次の段へ進んだ）
//  - 1 軸だけ壊す → その軸名・期待値・実測値を含む文言で reject
// の対偶で、門そのものと門の位置（GPU を取りに行く前）を同時に縛る。GPU へは 1 度も触らない。

import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { admitIrodori, assetOpener } from "../src/irodori/admission.ts";
import { IrodoriPipeline, type IrodoriPipelineOptions } from "../src/irodori/pipeline.ts";
import { ModelInputError } from "../src/errors.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import {
  declaredContainer,
  parseIrDeclarationValue,
  partAssets,
  tensorlessContainer,
  type TensorlessGraphSpec,
  tensorlessInput,
  writeContainer,
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
  "dit_context",
  "codec_decoder",
  "codec_encoder",
] as const;

type Component = typeof COMPONENTS[number];
/** 出力 1 本の宣言で足りる部品（`dit_context` は K / V の対を出すので別の器 — {@link ContextSpec}）。 */
type SingleOutput = Exclude<Component, "dit_context">;

/**
 * 実物の DiT のブロック数（v4 系は 12）。**テストが実物の宣言を写すための数**で、実装は焼かない
 * （ブロック数は `dit_context` の出力列から導く — ADR 0114）。
 */
const BLOCKS = 12;
/** 条件 3 本を連結した context の行数（text + speaker + caption = 1519）。 */
const CONTEXT_ROWS = CONFIG.maxTextLen + CONFIG.speakerRows + CONFIG.maxCaptionLen;
/** context K / V 1 本の形（20 heads × 64 head_dim — reshape 後・transpose 前）。 */
const CONTEXT_SHAPE = [1, CONTEXT_ROWS, 20, 64] as const;
/** `dit_context` の出力 = `dit` の K / V 入力（ブロック順に K, V）。 */
const CONTEXT_NAMES: readonly string[] = Array.from(
  { length: BLOCKS },
  (_, block) => [`context_k_${block}`, `context_v_${block}`],
).flat();

/** 宣言の 1 値（dtype を省略すると f32 — 突合の器 `TensorlessGraphSpec` と同じ既定）。 */
type ValueSpec = {
  readonly name: string;
  readonly shape: readonly (number | string)[];
  readonly dtype?: "f32" | "i32";
};

/** `dit_context` の宣言（出力が複数本 — `TensorlessGraphSpec` は出力 1 本の器なので別に持つ）。 */
type ContextSpec = {
  readonly symbols?: readonly string[];
  readonly inputs: readonly ValueSpec[];
  readonly outputs: readonly ValueSpec[];
};

/**
 * `dit_context` の容器の入力（ノードは各出力につき「出力と同じ dtype の先頭入力を 2 口で受ける
 * `mul` 1 本」— `container-fixture.ts` の `declarationOf` と同じく、宣言の突合が目的で実行は
 * しない。dtype を合わせるのは IR の op 契約の検査を通すため）。
 */
const contextInput = (spec: ContextSpec): ModelInput => ({
  graphs: {
    dit_context: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: spec.symbols ?? [],
      inputs: spec.inputs.map((input) => ({
        name: input.name,
        dtype: input.dtype ?? "f32",
        shape: input.shape,
      })),
      outputs: spec.outputs.map((output) => output.name),
      initializers: {},
      values: Object.fromEntries(
        spec.outputs.map((output) => [
          output.name,
          { dtype: output.dtype ?? "f32", shape: output.shape },
        ]),
      ),
      states: {},
      nodes: spec.outputs.map((output) => {
        const source = spec.inputs.find((input) =>
          (input.dtype ?? "f32") === (output.dtype ?? "f32")
        ) ?? spec.inputs[0];
        return { op: "mul", ins: [source.name, source.name], outs: [output.name], attrs: {} };
      }),
    }),
  },
  consts: [],
  weights: [],
  assets: [],
  provenance: { license: "test" },
});

/** 実物と同じ `dit_context` の宣言（条件 3 本 → K / V 24 本）。 */
const contextSpec = (): ContextSpec => ({
  inputs: [
    { name: "text_state", shape: [1, CONFIG.maxTextLen, CONFIG.textDim] },
    { name: "speaker_state", shape: [1, CONFIG.speakerRows, CONFIG.speakerDim] },
    { name: "caption_state", shape: [1, CONFIG.maxCaptionLen, CONFIG.captionDim] },
  ],
  outputs: CONTEXT_NAMES.map((name) => ({ name, shape: CONTEXT_SHAPE })),
});

/** 突合を全て通る最小グラフ 8 本（`CONFIG` の値がそのまま宣言に出る — `dit_context` は {@link contextSpec}）。 */
const graphSpecs = (): Record<SingleOutput, TensorlessGraphSpec> => ({
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
  // 条件 state 3 本は持たず、`dit_context` の出力（K / V 24 本）を同名・同形で受ける。
  dit: {
    symbols: ["S"],
    inputs: [
      { name: "x_t", shape: [1, "S", CONFIG.latentDim] },
      { name: "t_embed", shape: [1, CONFIG.timestepEmbedDim] },
      { name: "mask", shape: [1, 1, 1, `S+${CONTEXT_ROWS}`] },
      ...CONTEXT_NAMES.map((name) => ({ name, shape: CONTEXT_SHAPE })),
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

/** 部品の差し替え（出力 1 本の部品は `TensorlessGraphSpec`・`dit_context` は {@link ContextSpec}）。 */
type Patch = Partial<Record<SingleOutput, TensorlessGraphSpec>> & {
  readonly dit_context?: ContextSpec;
};

/** 1 本だけ差し替えた容器 9 本を組む（`tokenizer` は入れない — 観測の仕掛け）。 */
const assetsWith = async (
  patch: Patch = {},
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const specs = { ...graphSpecs(), ...patch };
  const context = patch.dit_context ?? contextSpec();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const name of COMPONENTS) {
    const container = name === "dit_context"
      ? await writeContainer(contextInput(context))
      : await tensorlessContainer(name, specs[name]);
    assets = { ...assets, ...partAssets(name, container) };
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
  patch: Patch = {},
): Promise<unknown> =>
  await IrodoriPipeline.fromAssets({
    manifest: parseManifest(manifestText()),
    assets: await assetsWith(patch),
  });

/** 入力列の 1 本の 1 軸だけを壊す（部品の器の型に依らない）。 */
const breakAxis = <
  Input extends { readonly name: string; readonly shape: readonly (number | string)[] },
>(
  inputs: readonly Input[],
  inputName: string,
  axis: number,
  value: number | string,
): Input[] =>
  inputs.map((input) =>
    input.name === inputName
      ? { ...input, shape: input.shape.map((dim, index) => index === axis ? value : dim) }
      : input
  );

/** グラフ 1 本の入力 1 本の 1 軸だけを壊す。 */
const breakInput = (
  component: SingleOutput | "dit_context",
  inputName: string,
  axis: number,
  value: number | string,
): Patch => {
  if (component === "dit_context") {
    const spec = contextSpec();
    return { dit_context: { ...spec, inputs: breakAxis(spec.inputs, inputName, axis, value) } };
  }
  const spec = graphSpecs()[component];
  return { [component]: { ...spec, inputs: breakAxis(spec.inputs, inputName, axis, value) } };
};

/** グラフ 1 本の出力の 1 軸だけを壊す。 */
const breakOutput = (
  component: SingleOutput,
  axis: number,
  value: number | string,
): Patch => {
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

Deno.test("admitIrodori: 突合を全て満たすグラフは資産の段まで進む（門の位置の対偶）", async () => {
  // `tokenizer` を渡していないので、突合を全部通れば落ちるのはその 1 本。ここが
  // 「グラフ入力 '…' の軸 …」で落ちるなら、正常系のはずの宣言が門に引っかかっている。
  await assertRejects(() => build(), Error, "irodori: 資産 'tokenizer' が無い");
});

Deno.test("admitIrodori: 壊した軸ごとに軸名・期待値・実測値を出して落ちる（12 点）", async () => {
  // 各ケースは 1 軸だけを現行値の近傍へずらす（shape の rank は保つ）— rank を変えると
  // 別の門に当たって「この軸の突合が生きている」ことを示せない。
  const cases: readonly {
    readonly where: string;
    readonly patch: Patch;
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
      patch: breakInput("dit_context", "text_state", 1, 255),
      actual: "'text_state' の軸 1 が 255",
      expected: "pipelineConfig は 256",
    },
    {
      where: "textDim",
      patch: breakInput("dit_context", "text_state", 2, 511),
      actual: "'text_state' の軸 2 が 511",
      expected: "pipelineConfig は 512",
    },
    {
      where: "speakerRows",
      patch: breakInput("dit_context", "speaker_state", 1, 750),
      actual: "'speaker_state' の軸 1 が 750",
      expected: "pipelineConfig は 751",
    },
    {
      where: "speakerDim",
      patch: breakInput("dit_context", "speaker_state", 2, 767),
      actual: "'speaker_state' の軸 2 が 767",
      expected: "pipelineConfig は 768",
    },
    {
      where: "maxCaptionLen",
      patch: breakInput("dit_context", "caption_state", 1, 511),
      actual: "'caption_state' の軸 1 が 511",
      expected: "pipelineConfig は 512",
    },
    {
      where: "captionDim",
      patch: breakInput("dit_context", "caption_state", 2, 511),
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

// ---- dit_context → dit の配線（ADR 0114）---------------------------------------------------
//
// 常駐入力は runtime 側で**大きさしか**検査されない（`RunInput` の doc）。形や dtype の食い違いは
// バイト数が合う限り沈黙で通るので、admission の門がここでの唯一の検出器になる。

Deno.test("admitIrodori: 実物と同じ宣言から K / V 24 本の名前と常駐バイト数を導く（178 MiB）", async () => {
  // 資産の段（tokenizer）より前で止まる admission を直接呼ぶ — 導出値は公開面から見えない。
  const admitted = await admitIrodori(
    parseManifest(manifestText()),
    await assetOpener(await assetsWith()),
    {},
  );
  const bytes = CONTEXT_SHAPE.reduce((product, dim) => product * dim, 4);
  assertEquals(
    admitted.ditContextOutputs,
    CONTEXT_NAMES.map((name) => ({ name, byteLength: bytes })),
  );
  // 設計の見積り（24 本 × [1,1519,20,64] f32 = 178.0 MiB）と同じ数になる。
  assertEquals(
    admitted.ditContextOutputs.reduce((total, output) => total + output.byteLength, 0),
    186_654_720,
  );
});

Deno.test("admitIrodori: dit_context が記号次元を持てば落とす（常駐テンソルは確保時に大きさが要る）", async () => {
  const spec = contextSpec();
  const symbolic: ContextSpec = {
    ...spec,
    symbols: ["T"],
    inputs: breakAxis(spec.inputs, "text_state", 0, "T"),
  };
  await assertRejects(
    () => build({ dit_context: symbolic }),
    Error,
    "dit_context が記号次元を持つ（[T]）",
  );
});

Deno.test("admitIrodori: dit_context の出力が K / V の対にならなければ落とす", async () => {
  const spec = contextSpec();
  // 最後の V を落とした 23 本（dit 側も同じ 23 本にそろえ、名前の門より先にここで落ちることを見る）。
  const odd = spec.outputs.slice(0, -1);
  const dit = graphSpecs().dit;
  await assertRejects(
    () =>
      build({
        dit_context: { ...spec, outputs: odd },
        dit: { ...dit, inputs: dit.inputs.filter((input) => input.name !== "context_v_11") },
      }),
    Error,
    "dit_context の出力が 23 本",
  );
});

Deno.test("admitIrodori: dit の入力名が {x_t, t_embed, mask} ∪ dit_context の出力と違えば両側の差分を出して落ちる", async () => {
  const dit = graphSpecs().dit;
  // dit 側だけ V を 1 本取り違える（K / V の対の取り違え = 条件側の配線ずれ）。
  const renamed = await assertRejects(
    () =>
      build({
        dit: {
          ...dit,
          inputs: dit.inputs.map((input) =>
            input.name === "context_v_11" ? { ...input, name: "context_v_12" } : input
          ),
        },
      }),
    Error,
    "dit の入力が {x_t, t_embed, mask} ∪ dit_context の出力と一致しない",
  );
  assertStringIncludes(renamed.message, "dit に無い: [context_v_11]");
  assertStringIncludes(renamed.message, "dit にだけある: [context_v_12]");

  // 旧形（dit が条件 state 3 本を直接受け、mask 以外に K / V を持たない）は名指しで落ちる。
  const legacy = await assertRejects(
    () =>
      build({
        dit: {
          ...dit,
          inputs: [
            ...dit.inputs.filter((input) => !CONTEXT_NAMES.includes(input.name)),
            { name: "text_state", shape: [1, CONFIG.maxTextLen, CONFIG.textDim] },
          ],
        },
      }),
    Error,
    "dit にだけある: [text_state]",
  );
  assertStringIncludes(legacy.message, "context_k_0");
});

Deno.test("admitIrodori: dit_context の出力と dit の同名入力の形・dtype が違えば落とす（大きさが同じでも）", async () => {
  const spec = contextSpec();
  // head と head_dim を入れ替えた形（要素数は同じ = 常駐入力の大きさ検査は通ってしまう形）。
  const transposed: ContextSpec = {
    ...spec,
    outputs: spec.outputs.map((output) =>
      output.name === "context_k_3" ? { ...output, shape: [1, CONTEXT_ROWS, 64, 20] } : output
    ),
  };
  const shape = await assertRejects(
    () => build({ dit_context: transposed }),
    Error,
    "dit_context の出力 'context_k_3'（f32 [1, 1519, 64, 20]）が dit の同名入力（f32 [1, 1519, 20, 64]）と合わない",
  );
  assertStringIncludes(shape.message, "静的な同じ形・同じ dtype が要る");

  // dtype だけが違う（4 バイト同士なのでこれもバイト数では見えない）。i32 の出力を宣言として
  // 成り立たせるために i32 の入力を 1 本足す（突合の門は条件 3 本の軸しか見ない）。
  const retyped: ContextSpec = {
    ...spec,
    inputs: [...spec.inputs, { name: "ids", shape: [1], dtype: "i32" }],
    outputs: spec.outputs.map((output) =>
      output.name === "context_v_0" ? { ...output, dtype: "i32" } : output
    ),
  };
  await assertRejects(
    () => build({ dit_context: retyped }),
    Error,
    "dit_context の出力 'context_v_0'（i32 [1, 1519, 20, 64]）が dit の同名入力（f32 [1, 1519, 20, 64]）と合わない",
  );
});

Deno.test("fromPretrained: components で差した dit の次元が違えば重みを取る前に落ちる（差し替え席が配線されている）", async () => {
  // 差し替え先の dit だけ latentDim を 64 に焼く（元リポの 9 本は突合を全て通る）。
  // 差し替え席を配線し忘れると、元リポのまま突合を通って資産の取得へ進み、別の文言で落ちる。
  const specs = graphSpecs();
  let weights: Record<string, unknown> = {};
  let files: (readonly [string, Uint8Array<ArrayBuffer>])[] = [];
  for (const name of COMPONENTS) {
    const input = name === "dit_context"
      ? contextInput(contextSpec())
      : tensorlessInput(name, specs[name]);
    const served = await serveContainer(`${name}/model.f32`, input);
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

// ---- 実行ノブの明示指定（明示 > quant 宣言 > runtime 既定 — ADR 0058 追記 2026-09-26）----------

/** 型の外から来る呼び手（JS の消費者）の構築オプションを再現する。 */
const optionsWith = (key: string, value: unknown): IrodoriPipelineOptions => {
  const options: IrodoriPipelineOptions = {};
  Object.defineProperty(options, key, { value, enumerable: true });
  return options;
};

Deno.test("admitIrodori: 不正な明示値と attention 系のノブはグラフ突合より前に ModelInputError", async () => {
  // dit の宣言を 1 軸壊してある — 合成がグラフ突合より後ろなら、突合の文言で落ちる。
  const manifest = parseManifest(manifestText());
  const broken = await assetsWith(breakInput("dit", "x_t", 2, CONFIG.latentDim + 1));
  await assertRejects(
    () =>
      IrodoriPipeline.fromAssets({ manifest, assets: broken }, optionsWith("linearCompute", null)),
    ModelInputError,
    "linearComputeが不正",
  );
  // dit の attention は融合 attention の契約に載らない（分解経路）ので、効く席が無い。
  await assertRejects(
    () =>
      IrodoriPipeline.fromAssets(
        { manifest, assets: broken },
        optionsWith("attentionCompute", "f16"),
      ),
    ModelInputError,
    "attentionComputeはこの系列では指定できない",
  );
});

Deno.test("admitIrodori: 受理される linearCompute の明示は突合を通って資産の段まで進む", async () => {
  await assertRejects(
    async () =>
      await IrodoriPipeline.fromAssets({
        manifest: parseManifest(manifestText()),
        assets: await assetsWith(),
      }, { linearCompute: "a8" }),
    Error,
    "irodori: 資産 'tokenizer' が無い",
  );
});

Deno.test("admitIrodori: 明示の f16 を共有 GPU が持たなければ GPU 取得前に名指しで落ちる", async () => {
  // quant 宣言は shader-f16 を要求しない。要求は実効設定から導く（黙って f32 へ落とさず、
  // 重みを上げた後の Session 構築まで遅らせもしない）。
  await assertRejects(
    async () =>
      await IrodoriPipeline.fromAssets({
        manifest: parseManifest(manifestText()),
        assets: await assetsWith(),
      }, { gpu: fakeGpuContext(fakeDevice()), linearCompute: "f16" }),
    Error,
    "shader-f16",
  );
});
