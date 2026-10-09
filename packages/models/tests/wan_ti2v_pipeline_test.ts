// Wan2.2 TI2V 5B の class（`WanTi2vPipeline` — `src/wan/ti2v-pipeline.ts`）の GPU 不要の部分。family.ts が
// class ごとの値（`WanFamilySpec` の owner・pipeline・generation）を本当に通していることを、公開の入口から縛る:
//
// - manifest の門（pipeline 名 / major・quant・`pipelineConfig`）— 2.1 と 2.2 の class が互いの manifest を拒む
// - 家族 admission の 48 ch の経路（tensorless の容器で `fromAssets` を構築の完了まで進める）— VAE を 2.1 の
//   16 ch に差し替える・DiT を 2.1 の 5 入力に差し替えると世代の照合で落ちる（admission が 2.1 の形へ戻る
//   退行の縛り）
// - 取得面（`fromPretrained`）も 2.2 の family を渡す（手元の取得元で `wan/1` を拒み `wan-ti2v/1` を受ける）
// - 計画の門が 2.2 の受理集合で効く・模擬 Session の生成で patch 2 の記述子が VAE の段の末尾（unpatchify）まで届く
// - 文言の接頭辞が `WanTi2vPipeline:`（2.1 の文言で取り違えない）
// - I2V（ADR 0121 決定 11）: encoder の部品の admission・段の順と解放（encoder → 畳む → 解放待ち → text → DiT）・
//   encoder の受け渡し（常駐と bindings）・条件の潜在の非有限の門・DiT の条件づけ（先頭 P トークンのマスク・条件側の
//   時刻 0・モデル入力だけの置き換え・UniPC へは置き換えない潜在・最後の 1 回の置き換え）・要求の門（fit / 寸法 / RGB8）・
//   公式の置き換えの形との最終の一致（UniPC は要素ごと）
//
// 実 GPU の通しは e2e（系列から組む helper — ADR 0121 段 6 のコミット 6）。

import {
  assert,
  assertEquals,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { type DistributionSource, localDirectory, type Manifest, parseManifest } from "@karume/hub";
import type { EnqueueOptions, RunInput, RunInputs, Tensor } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import type { Rgb8Image } from "../src/image/preprocess.ts";
import {
  type WanAssets,
  type WanGenerateRequest,
  WanPipeline,
  type WanPipelineOptions,
} from "../src/wan/pipeline.ts";
import { type WanTi2vGenerateRequest, WanTi2vPipeline } from "../src/wan/ti2v-pipeline.ts";
import { planWanGeneration, WAN22_TI2V_FAMILY } from "../src/wan/family.ts";
import { ditContract, wanDitPatch, withWanFirstFrameCondition } from "../src/wan/dit-loop.ts";
import { timestepsProj } from "../src/wan/dit-timestep.ts";
import { patchifyLatents } from "../src/wan/dit-tokens.ts";
import { preprocessWanI2vImage } from "../src/wan/i2v-preprocess.ts";
import { denormalizeWanLatents, normalizeWanLatents } from "../src/wan/latents.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../src/wan/scheduler.ts";
import { wanVaeEncoderContract } from "../src/wan/vae-encoder.ts";
import { wanParityCase, wanParityEncoder } from "./helpers/wan-parity-encoder.ts";
import type { WanRopeBase } from "../src/wan/dit-rope.ts";
import { wanVaeChunkLayout } from "../src/wan/vae-chunks.ts";
import type { WanPipelineConfig } from "../src/wan/config.ts";
import {
  parseWanTextEmbeds,
  WAN_TEXT_EMBEDS_METADATA_KEY,
  type WanTextEmbeds,
} from "../src/wan/text-embeds.ts";
import {
  declaredContainer,
  parseIrDeclarationValue,
  partAssets,
  writeContainer,
} from "./helpers/container-fixture.ts";
import {
  fileRef,
  MANIFEST_PATH,
  manifestBytes,
  serveContainer,
} from "./helpers/container-loading-fixture.ts";
import { stubModel } from "./helpers/stub-model.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

/** 2.2 の class の文言の接頭辞（F3 — 2.1 の `WanPipeline:` で取り違えない）。 */
const OWNER = "WanTi2vPipeline:";

/** 系列の `pipelineConfig` の形（既定の shift は上流の scheduler の `flow_shift` 5.0）。 */
const CONFIG: WanPipelineConfig = { scheduler: { shift: 5 }, defaults: { steps: 50, guidance: 5 } };

/** 埋め込み資産の 1 行。 */
type Row = {
  readonly name: string;
  readonly role: "positive" | "negative";
  readonly prompt: string;
  readonly normalized: string;
  readonly tokens: number;
};

const ROWS: readonly Row[] = [
  { name: "cats", role: "positive", prompt: "Two cats.", normalized: "Two cats.", tokens: 2 },
  { name: "negative", role: "negative", prompt: "blurry", normalized: "blurry", tokens: 1 },
];

/**
 * 埋め込み資産のバイト列（recipe `wan/text_embeds.py` と同じ形 — メタはキー 1 つに JSON・行は
 * `[tokens, width]` の f32）。
 */
const embedsAsset = (width: number): Uint8Array<ArrayBuffer> => {
  const header: Record<string, unknown> = {
    __metadata__: {
      [WAN_TEXT_EMBEDS_METADATA_KEY]: JSON.stringify({ prompts: ROWS, source: {} }),
    },
  };
  let offset = 0;
  for (const row of ROWS) {
    const bytes = row.tokens * width * 4;
    header[row.name] = {
      dtype: "F32",
      shape: [row.tokens, width],
      data_offsets: [offset, offset + bytes],
    };
    offset += bytes;
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const headerLength = json.length + ((8 - (json.length % 8)) % 8);
  const out = new Uint8Array(8 + headerLength + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerLength), true);
  out.set(json, 8);
  out.fill(0x20, 8 + json.length, 8 + headerLength);
  return out;
};

// ---------------------------------------------------------------------------------------------
// 構築の入口（tensorless の容器 — 宣言だけを持ち、重みは 1 本も無い）
// ---------------------------------------------------------------------------------------------

/** RoPE の素表（`transformer` の容器の資産 `rope_base` — 幅 22 / 21 / 21 で head_dim 128）。 */
const ROPE_BASE = await Deno.readFile(
  new URL("./fixtures/wan-dit/rope_base.safetensors", import.meta.url),
);

type DitInput = {
  readonly name: string;
  readonly shape: readonly (number | string)[];
  readonly dtype?: "f32" | "bool";
};

/** recipe `wan/ti2v_export_dit.py` と同じ 7 本の宣言（潜在 48 ch → `tokens` の幅 192）。 */
const TI2V_DIT_INPUTS: readonly DitInput[] = [
  { name: "tokens", shape: [1, "S", 192] },
  { name: "timesteps_proj", shape: [1, 256] },
  { name: "encoder_hidden_states", shape: [1, 512, 4096] },
  { name: "rope_cos", shape: [1, "S", 1, 128] },
  { name: "rope_sin", shape: [1, "S", 1, 128] },
  { name: "timesteps_proj_condition", shape: [1, 256] },
  { name: "condition_mask", shape: [1, "S", 1], dtype: "bool" },
];

/** DiT の容器の中身（宣言だけ・RoPE の素表の資産つき）。 */
const ditInput = (inputs: readonly DitInput[]): ModelInput => ({
  graphs: {
    transformer: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: ["S"],
      inputs: inputs.map(({ name, shape, dtype }) => ({ name, dtype: dtype ?? "f32", shape })),
      outputs: ["out"],
      initializers: {},
      values: { out: { dtype: "f32", shape: [1, "S", 192] } },
      states: {},
      nodes: [{ op: "mul", ins: ["tokens", "tokens"], outs: ["out"], attrs: {} }],
    }),
  },
  consts: [],
  weights: [],
  assets: [{ name: "rope_base", role: "rope-base", bytes: ROPE_BASE, dedicatedPart: true }],
  provenance: { license: "test" },
});

/** DiT の容器の書き出し設定（素表〈12,704 バイト〉が 1 block に収まる大きさ）。 */
const DIT_WRITE = { partBytes: 1 << 16, blockBytes: 1 << 14 };

/** DiT の容器（`fromAssets` の part 列）。 */
const ditAssets = async (inputs: readonly DitInput[]) =>
  partAssets("transformer", await writeContainer(ditInput(inputs), DIT_WRITE));

type VaeChannels = { readonly latent: number; readonly sample: number };

/** VAE の chunk グラフ 1 本の容器の中身（潜在タイル 16・縮尺 8・cache 1 本 — 2.2 の系列と同じ幾何）。 */
const vaeInput = (name: string, frames: number, channels: VaeChannels): ModelInput => ({
  graphs: {
    [name]: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: [],
      inputs: [
        { name: "latent", dtype: "f32", shape: [channels.latent, 1, 16, 16] },
        { name: "cache_00", dtype: "f32", shape: [4, 2, 16, 16] },
      ],
      outputs: ["frame", "cache_00_out"],
      initializers: {},
      values: {
        frame: { dtype: "f32", shape: [channels.sample, frames, 128, 128] },
        cache_00_out: { dtype: "f32", shape: [4, 2, 16, 16] },
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
});

/** VAE の chunk グラフ 1 本の容器（`fromAssets` の part 列）。 */
const vaeAssets = async (name: string, frames: number, channels: VaeChannels) =>
  partAssets(name, await writeContainer(vaeInput(name, frames, channels)));

/** I2V の VAE encoder のグラフ 1 本の宣言（入力 1 本 → 出力 1 本・記号長）。 */
type EncoderGraph = {
  readonly symbols: readonly string[];
  readonly input: { readonly name: string; readonly shape: readonly (number | string)[] };
  readonly output: readonly (number | string)[];
};

/**
 * recipe `wan/export_vae_encoder.py` と同じ入出力・記号の 3 グラフ（潜在 48・mid block 640・patchify 済みの画像
 * `[12, 1, 8h, 8w]`）。
 */
const ENCODER_GRAPHS: Readonly<Record<string, EncoderGraph>> = {
  vae_encoder_pre: {
    symbols: ["h", "w"],
    input: { name: "image", shape: [12, 1, "8h", "8w"] },
    output: [640, 1, "h", "w"],
  },
  vae_encoder_attn: {
    symbols: ["S"],
    input: { name: "tokens", shape: [640, "S"] },
    output: [640, "S"],
  },
  vae_encoder_post: {
    symbols: ["h", "w"],
    input: { name: "hidden", shape: [640, 1, "h", "w"] },
    output: [48, 1, "h", "w"],
  },
};

/** encoder のグラフ 1 本の容器の中身（宣言だけ）。 */
const encoderInput = (name: string, graph: EncoderGraph): ModelInput => ({
  graphs: {
    [name]: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: graph.symbols,
      inputs: [{ name: graph.input.name, dtype: "f32", shape: graph.input.shape }],
      outputs: ["out"],
      initializers: {},
      values: { out: { dtype: "f32", shape: graph.output } },
      states: {},
      nodes: [{ op: "mul", ins: [graph.input.name, graph.input.name], outs: ["out"], attrs: {} }],
    }),
  },
  consts: [],
  weights: [],
  assets: [],
  provenance: { license: "test" },
});

/** encoder の 3 本の容器（`fromAssets` の part 列 — `replaced` の部品だけ宣言を差し替える）。 */
const encoderAssets = async (
  replaced: Readonly<Record<string, EncoderGraph>> = {},
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const graphs = { ...ENCODER_GRAPHS, ...replaced };
  const parts = await Promise.all(
    Object.entries(graphs).map(async ([name, graph]) =>
      partAssets(name, await writeContainer(encoderInput(name, graph)))
    ),
  );
  return parts.reduce((all, part) => ({ ...all, ...part }), {});
};

/** 2.2 の配布形の部品（DiT・VAE decoder 2 本・I2V の VAE encoder 3 本）。 */
const COMPONENTS = [
  "transformer",
  "vae_decoder_first",
  "vae_decoder_next",
  ...Object.keys(ENCODER_GRAPHS),
];

/** 構築の入口のテストが使う manifest（系列から組む helper と同じ骨格 — 宣言だけ）。 */
const ti2vManifest = (
  overrides: { readonly pipeline?: string; readonly pipelineConfig?: unknown } = {},
): Manifest =>
  parseManifest(JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "ti2v-5b",
    models: {
      "ti2v-5b": {
        pipeline: overrides.pipeline ?? "wan-ti2v/1",
        weights: Object.fromEntries(
          COMPONENTS.map((name) => [name, { f16: declaredContainer(`${name}/model.f16`) }]),
        ),
        assets: {
          text_embeds: { path: "text_embeds.safetensors", size: 8, sha256: "e".repeat(64) },
        },
        quants: {
          f16: {
            weights: Object.fromEntries(COMPONENTS.map((name) => [name, "f16"])),
            session: {},
          },
        },
        defaultQuant: "f16",
        pipelineConfig: overrides.pipelineConfig ?? CONFIG,
      },
    },
  }));

/** 2.2 の宣言の資産一式（48 ch・出口 12 ch の VAE・7 入力の DiT・encoder 3 本・埋め込み幅 4096）。 */
const VALID_ASSETS: WanAssets["assets"] = {
  ...await ditAssets(TI2V_DIT_INPUTS),
  ...await vaeAssets("vae_decoder_first", 1, { latent: 48, sample: 12 }),
  ...await vaeAssets("vae_decoder_next", 4, { latent: 48, sample: 12 }),
  ...await encoderAssets(),
  text_embeds: embedsAsset(4096),
};

/** 資産の一部を差し替えて `WanTi2vPipeline.fromAssets` へ渡す（共有の模擬 GPU — アダプタに触らない）。 */
const buildTi2v = (
  manifest: Manifest,
  replaced: WanAssets["assets"] = {},
  options: WanPipelineOptions = {},
) =>
  WanTi2vPipeline.fromAssets(
    { manifest, assets: { ...VALID_ASSETS, ...replaced } },
    { textEncoder: "precomputed", gpu: fakeGpuContext(fakeDevice()), ...options },
  );

/** TS の型を通らない値を 1 欄だけ差した構築オプション（JS の呼び手の綴り違いの再現）。 */
const optionsWith = (key: string, value: unknown): WanPipelineOptions => {
  const options: WanPipelineOptions = {};
  Object.defineProperty(options, key, { value, enumerable: true });
  return options;
};

describe("WanTi2vPipeline.fromAssets: manifest の門（pipeline 名 / major・quant・pipelineConfig）", () => {
  it("対照: wan-ti2v/1 の manifest と 2.2 の宣言の資産は構築を終え、prompts を資産の並びで返す", async () => {
    const pipeline = await buildTi2v(ti2vManifest());
    assert(pipeline instanceof WanTi2vPipeline);
    assertEquals(pipeline.prompts.map((prompt) => prompt.name), ["cats", "negative"]);
    await pipeline.dispose();
  });

  it("wan/1（Wan2.1）の manifest を拒み、文言は WanTi2vPipeline: で始まる", async () => {
    const error = await assertRejects(
      () => buildTi2v(ti2vManifest({ pipeline: "wan/1" })),
      Error,
      "manifest の pipeline が 'wan/1'（'wan-ti2v/1' が必要）",
    );
    assert(error.message.startsWith(OWNER), error.message);
    assert(!(error instanceof ModelInputError), "資産の齟齬を入力起因にしない");
  });

  it("逆向き: WanPipeline は wan-ti2v/1 の manifest を WanPipeline: の文言で拒む", async () => {
    const error = await assertRejects(
      () =>
        WanPipeline.fromAssets(
          { manifest: ti2vManifest(), assets: VALID_ASSETS },
          { textEncoder: "precomputed", gpu: fakeGpuContext(fakeDevice()) },
        ),
      Error,
      "WanPipeline: manifest の pipeline が 'wan-ti2v/1'（'wan/1' が必要）",
    );
    assert(!error.message.startsWith(OWNER), error.message);
  });

  it("major 2 は「この実装が読めるのは wan-ti2v/1」と言って拒む", async () => {
    const error = await assertRejects(
      () => buildTi2v(ti2vManifest({ pipeline: "wan-ti2v/2" })),
      Error,
      "pipeline 'wan-ti2v/2' の major に未対応（この実装が読めるのは wan-ti2v/1）",
    );
    assert(error.message.startsWith(OWNER), error.message);
  });

  it("manifest に無い quant は利用可能な一覧つきで拒む", async () => {
    const error = await assertRejects(
      () => buildTi2v(ti2vManifest(), {}, { quant: "f16+dit8" }),
      Error,
      "quant 'f16+dit8' は manifest に無い（利用可能: f16）",
    );
    assert(error.message.startsWith(OWNER), error.message);
  });

  it("pipelineConfig の未知キーは 2.1 と同じスキーマの門で拒む", async () => {
    await assertRejects(
      () =>
        buildTi2v(
          ti2vManifest({ pipelineConfig: { ...CONFIG, scheduler: { shift: 5, flowShift: 5 } } }),
        ),
      Error,
      "pipelineConfig.scheduler: 未知キー 'flowShift'（許可: shift）",
    );
  });

  it("textEncoder の未知の綴りは資産に触る前に WanTi2vPipeline: の素の Error", async () => {
    const error = await assertRejects(
      () =>
        WanTi2vPipeline.fromAssets(
          { manifest: ti2vManifest(), assets: {} },
          optionsWith("textEncoder", "cpu"),
        ),
      Error,
      `${OWNER} textEncoder 'cpu' は 'gpu' / 'precomputed' のどちらでもない`,
    );
    assert(!(error instanceof ModelInputError));
  });
});

describe("WanTi2vPipeline.fromAssets: 家族 admission は 2.2 の世代で照合する（48 ch・7 入力）", () => {
  it("VAE を 2.1 の宣言（潜在 16・出口 3）に差し替えると、統計 48 本との照合で落ちる", async () => {
    const vae = {
      ...await vaeAssets("vae_decoder_first", 1, { latent: 16, sample: 3 }),
      ...await vaeAssets("vae_decoder_next", 4, { latent: 16, sample: 3 }),
    };
    await assertRejects(
      () => buildTi2v(ti2vManifest(), vae),
      Error,
      `${OWNER} 逆正規化の統計（mean 48 本・std 48 本）が VAE の潜在 16 チャネルと違う`,
    );
  });

  it("潜在 48 でも出口が RGB の 3 ch（unpatchify の無い宣言）なら、unpatchify 2 との照合で落ちる", async () => {
    const vae = {
      ...await vaeAssets("vae_decoder_first", 1, { latent: 48, sample: 3 }),
      ...await vaeAssets("vae_decoder_next", 4, { latent: 48, sample: 3 }),
    };
    await assertRejects(
      () => buildTi2v(ti2vManifest(), vae),
      Error,
      `${OWNER} VAE の出口 3 チャネルが RGB 3 × unpatchify 2² = 12 と違う`,
    );
  });

  it("DiT を 2.1 の 5 入力に差し替えると、ti2v の入力の集合の文言で落ちる", async () => {
    const t2v = "tokens, timesteps_proj, encoder_hidden_states, rope_cos, rope_sin";
    const dit = await ditAssets(TI2V_DIT_INPUTS.slice(0, 5));
    await assertRejects(
      () => buildTi2v(ti2vManifest(), dit),
      Error,
      `${OWNER} transformer のグラフ入力が [${t2v}]` +
        `（期待: [${t2v}, timesteps_proj_condition, condition_mask] — 入力の形 'ti2v'）`,
    );
  });
});

/**
 * 2.2 の宣言の容器を置いたメモリ上のディレクトリ（手元の配布形の席 — `localDirectory`）。`pipeline` だけを
 * 変えて、取得面の門が class の pipeline 名で効くかを見る。
 */
const ti2vDirectory = async (pipeline: string): Promise<DistributionSource> => {
  const transformer = await serveContainer(
    "ti2v/transformer/model.f16",
    ditInput(TI2V_DIT_INPUTS),
    DIT_WRITE,
  );
  const first = await serveContainer(
    "ti2v/vae_decoder_first/model.f16",
    vaeInput("vae_decoder_first", 1, { latent: 48, sample: 12 }),
  );
  const next = await serveContainer(
    "ti2v/vae_decoder_next/model.f16",
    vaeInput("vae_decoder_next", 4, { latent: 48, sample: 12 }),
  );
  const encoders = await Promise.all(
    Object.entries(ENCODER_GRAPHS).map(async ([name, graph]) =>
      [name, await serveContainer(`ti2v/${name}/model.f16`, encoderInput(name, graph))] as const
    ),
  );
  const embeds = embedsAsset(4096);
  const embedsPath = "text_embeds.safetensors";
  const models = {
    test: {
      pipeline,
      weights: {
        transformer: { f16: transformer.entry },
        vae_decoder_first: { f16: first.entry },
        vae_decoder_next: { f16: next.entry },
        ...Object.fromEntries(encoders.map(([name, served]) => [name, { f16: served.entry }])),
      },
      assets: { text_embeds: await fileRef(embedsPath, embeds) },
      quants: {
        f16: {
          weights: Object.fromEntries(COMPONENTS.map((name) => [name, "f16"])),
          session: {},
        },
      },
      defaultQuant: "f16",
      pipelineConfig: CONFIG,
    },
  };
  const files = new Map<string, Uint8Array<ArrayBuffer>>([
    [MANIFEST_PATH, manifestBytes(models)],
    ...transformer.files,
    ...first.files,
    ...next.files,
    ...encoders.flatMap(([, served]) => [...served.files]),
    [embedsPath, embeds],
  ]);
  return localDirectory({
    readFile: (path) => {
      const bytes = files.get(path);
      if (bytes === undefined) return Promise.reject(new Error(`ti2v: ${path} が無い`));
      // MUST: buffer 全体を占める view を返す（`DirectoryAdapter.readFile` の契約）。
      return Promise.resolve(bytes.slice());
    },
  }, { label: "ti2v" });
};

describe("WanTi2vPipeline.fromPretrained: 取得面も 2.2 の family を渡す（手元の取得元）", () => {
  const fromPretrained = (source: DistributionSource) =>
    WanTi2vPipeline.fromPretrained(source, {
      textEncoder: "precomputed",
      gpu: fakeGpuContext(fakeDevice()),
    });

  it("対照: wan-ti2v/1 の配布形は構築を終え、prompts を資産の並びで返す", async () => {
    const pipeline = await fromPretrained(await ti2vDirectory("wan-ti2v/1"));
    assert(pipeline instanceof WanTi2vPipeline);
    assertEquals(pipeline.prompts.map((prompt) => prompt.name), ["cats", "negative"]);
    await pipeline.dispose();
  });

  it("wan/1（Wan2.1）の配布形を WanTi2vPipeline: の文言で拒む", async () => {
    const error = await assertRejects(
      async () => fromPretrained(await ti2vDirectory("wan/1")),
      Error,
      "manifest の pipeline が 'wan/1'（'wan-ti2v/1' が必要）",
    );
    assert(error.message.startsWith(OWNER), error.message);
  });
});

/** 2.2 の VAE の chunk グラフ 2 本の宣言（潜在 48・タイル 16・縮尺 8・出口 12 = RGB 3 × 2²）。 */
const vaeChunkGraphs = () => {
  const graph = (frames: number) =>
    stubModel({
      inputs: [
        { name: "latent", shape: [48, 1, 16, 16] },
        { name: "cache_00", shape: [4, 2, 16, 16] },
      ],
      outputs: ["frame", "cache_00_out"],
      values: { frame: [12, frames, 128, 128], cache_00_out: [4, 2, 16, 16] },
    });
  return { first: graph(1), next: graph(4) };
};

// NOTE: ここは `WAN22_TI2V_FAMILY` の定数の中身を計画の門で縛る（テストが generation と owner を直接
// 渡す）。family の生成が spec.generation を渡す経路は、模擬 Session の 832×480 のテストが縛る。
describe("WAN22_TI2V_FAMILY の世代の値（計画の門）", () => {
  const embeds: WanTextEmbeds = parseWanTextEmbeds(embedsAsset(3).buffer);
  const graphs = vaeChunkGraphs();
  const layout = wanVaeChunkLayout(graphs.first, graphs.next);
  const plan = (request: Partial<WanGenerateRequest>) =>
    planWanGeneration(
      { prompt: "Two cats.", ...request },
      embeds,
      layout,
      CONFIG,
      WAN22_TI2V_FAMILY.generation,
      WAN22_TI2V_FAMILY.owner,
    );

  it("1280×704 / 704×1280 × 33 フレームは通り、潜在は [48, 9, H/16, W/16]", () => {
    assertEquals(plan({ frames: 33 }).latentShape, [48, 9, 44, 80]);
    assertEquals(plan({ frames: 33, width: 704, height: 1280 }).latentShape, [48, 9, 80, 44]);
  });

  it("省いた寸法とフレーム数は記述子の既定 1280×704・33 フレーム、省いた shift は pipelineConfig の値", () => {
    const planned = plan({});
    assertEquals([planned.width, planned.height, planned.frames], [1280, 704, 33]);
    assertEquals(planned.shift, 5);
  });

  it("既定の 33 を超えて上限 121 までの 4n+1（37 / 41 / 45 / 49 / 53 / 121）は通り、潜在のフレーム数は 10 / 11 / 12 / 13 / 14 / 31", () => {
    for (
      const [frames, latentFrames] of [[37, 10], [41, 11], [45, 12], [49, 13], [53, 14], [121, 31]]
    ) {
      assertEquals(plan({ frames }).latentShape, [48, latentFrames, 44, 80], `frames ${frames}`);
    }
  });

  it("832×480（Wan2.1 の既定）は 2.2 の受理集合の文言で、125 フレームは上限 121 の文言で ModelInputError", () => {
    assertThrows(
      () => plan({ width: 832, height: 480 }),
      ModelInputError,
      "832×480 が受理集合（1280×704 / 704×1280）に無い",
    );
    assertThrows(
      () => plan({ frames: 125 }),
      ModelInputError,
      "frames 125 が受理集合（4n+1 の 5〜121）に無い",
    );
  });
});

// ---------------------------------------------------------------------------------------------
// 模擬 Session の生成（`wan_pipeline_test.ts` の mockPipeline と同じ形 — 2.2 の幾何・資産の経路だけ）
// ---------------------------------------------------------------------------------------------

/** VAE の出口（patchify 空間）のチャネル `s` が返す値 — チャネルごとに違い、2 進で割り切れる。 */
const patchValue = (channel: number): number => channel / 16;

/** encoder の 3 Session の enqueue 1 回（模擬が記録する引数）。 */
type EncoderEnqueue = {
  readonly part: string;
  readonly inputs: RunInputs;
  readonly options: EnqueueOptions;
};

/** encoder の enqueue に注入する失敗（{@link mockTi2v} の `failEnqueue`）。 */
const ENQUEUE_FAILURE = new Error("模擬: encoder の enqueue の失敗");

/** encoder のグラフ 1 本の宣言だけを持つ {@link GraphOwner}（出口の名前は `out`）。 */
const encoderStub = (graph: EncoderGraph): GraphOwner =>
  stubModel({
    symbols: graph.symbols,
    inputs: [graph.input],
    outputs: ["out"],
    values: { out: graph.output },
  });

/** encoder の 3 グラフ（{@link ENCODER_GRAPHS} — `replaced` の部品だけ宣言を差し替える）。 */
const encoderGraphs = (replaced: Readonly<Record<string, EncoderGraph>> = {}) => {
  const graphs = { ...ENCODER_GRAPHS, ...replaced };
  return {
    pre: encoderStub(graphs.vae_encoder_pre),
    attn: encoderStub(graphs.vae_encoder_attn),
    post: encoderStub(graphs.vae_encoder_post),
  };
};

/**
 * 模擬 Session で回す `WanTi2vPipeline`。DiT は定数を返し、VAE は 1 タイル = 1 batch の手順を偽の
 * GpuContext で回して、読み戻すフレームを**チャネルごとの値**（{@link patchValue}）で埋める — unpatchify の
 * 並べ方が出力の画素の値に出る。I2V の encoder の段は 3 Session の enqueue を記録し、mu（読み戻しのキー `mu`）を
 * `options.mu` で埋める。偽の GpuContext の解放待ち（空の submit → `onSubmittedWorkDone`）も `log` に積む。
 *
 * NOTE: コンストラクタは TS の `private` なので `Reflect.construct` で内部状態（family.ts の `WanState`）を
 * 直接渡す（private の迂回はテストだけ — `wan_pipeline_test.ts` の mockPipeline と同じ理由）。
 */
const mockTi2v = (options: {
  /**
   * `"gpu"` なら text 段を模擬の umT5 で回す（プロンプト層はフィクスチャの表 — {@link wanParityEncoder}・出力は
   * 幅 3 の定数）。DiT の文脈は配布形と同じ 512 行（公式の negative は 126 トークン）。
   */
  readonly textEncoder?: "gpu";
  /** encoder の mu の値（添字 → 値 — 省けば 0.25）。 */
  readonly mu?: (index: number) => number;
  /** この部品（`vae_encoder_*`）の enqueue で {@link ENQUEUE_FAILURE} を投げる。 */
  readonly failEnqueue?: string;
} = {}) => {
  const log: string[] = [];
  const ditRuns: Record<string, Tensor>[] = [];
  /** encoder の 3 Session の enqueue（発行順）。 */
  const encoderEnqueues: EncoderEnqueue[] = [];
  /** VAE の chunk の enqueue の `latent`（発行順 — 1 タイル目は first → next）。 */
  const vaeLatents: Tensor[] = [];
  /** `onRunDiagnostics` が受けたコンポーネント名（発行順）。 */
  const diagnosed: string[] = [];
  /** `denoise-step` の `copyLatents()`（step 順）。 */
  const snapshots: Float32Array[] = [];
  const { first, next } = vaeChunkGraphs();
  const resident = (byteLength: number, label: string) => ({
    byteLength,
    label,
    write: () => {},
    dispose: () => {
      log.push(`release:${label}`);
    },
  });
  const vaeSession = (name: string) => ({
    createSession: () => {
      log.push(`create:${name}`);
      return Promise.resolve({
        enqueue: (inputs: Record<string, Tensor>) => {
          vaeLatents.push(inputs.latent);
          return Promise.resolve();
        },
        diagnostics: () => ({}),
        dispose: () => Promise.resolve(),
      });
    },
  });
  const encoderSession = (part: string) => ({
    createSession: (_gpu: unknown, sessionOptions: unknown) => {
      log.push(`create:${part}:${JSON.stringify(sessionOptions)}`);
      return Promise.resolve({
        enqueue: (inputs: EncoderEnqueue["inputs"], enqueueOptions: EncoderEnqueue["options"]) => {
          log.push(`enqueue:${part}`);
          encoderEnqueues.push({ part, inputs, options: enqueueOptions });
          return options.failEnqueue === part ? Promise.reject(ENQUEUE_FAILURE) : Promise.resolve();
        },
        diagnostics: () => ({}),
        dispose: () => {
          log.push(`dispose:${part}`);
          return Promise.resolve();
        },
      });
    },
  });
  const layout = wanVaeChunkLayout(first, next);
  // 行数 64 は潜在 [48, 2, 44, 80] の格子（2 × 22 × 40）を覆う。値は門と無関係。
  const ropeBase: WanRopeBase = {
    rows: 64,
    widths: [1, 1, 1],
    cos: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
    sin: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
  };
  const gpuText = options.textEncoder === "gpu";
  // DiT の取り決めは家族 admission と同じ関数・同じ値（patch は VAE の宣言のチャネル数・入力の形と
  // owner は spec）で組む — spec の入力の形が 2.1 へ戻ると、7 入力の宣言はここで落ちる。
  const dit = ditContract(
    stubModel({
      symbols: ["S"],
      inputs: [
        { name: "tokens", shape: [1, "S", 192] },
        { name: "timesteps_proj", shape: [1, 256] },
        { name: "encoder_hidden_states", shape: [1, gpuText ? 512 : 4, 3] },
        { name: "rope_cos", shape: [1, "S", 1, 6] },
        { name: "rope_sin", shape: [1, "S", 1, 6] },
        { name: "timesteps_proj_condition", shape: [1, 256] },
        { name: "condition_mask", shape: [1, "S", 1], dtype: "bool" },
      ],
      outputs: ["out"],
      values: { out: [1, "S", 192] },
    }),
    ropeBase,
    wanDitPatch(layout.latentChannels),
    WAN22_TI2V_FAMILY.generation.ditInputForm,
    WAN22_TI2V_FAMILY.owner,
  );
  // encoder の取り決めも家族 admission と同じ関数・同じ値（decoder の幾何・2.2 の記述子）で組む。
  const vaeEncoderContract = wanVaeEncoderContract(
    encoderGraphs(),
    layout,
    WAN22_TI2V_FAMILY.generation,
    WAN22_TI2V_FAMILY.owner,
  );
  const text = gpuText
    ? {
      kind: "gpu",
      encoder: wanParityEncoder(),
      contract: { output: "umt5_out" },
      component: {
        createSession: () => {
          log.push("create:text_encoder");
          return Promise.resolve({
            run: (inputs: Record<string, Tensor>) => {
              const tokens = inputs.input_ids.shape[1];
              return Promise.resolve({
                umt5_out: {
                  dtype: "f32",
                  shape: [1, tokens, 3],
                  data: new Float32Array(tokens * 3).fill(tokens / 1000),
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
      createResident: (bytes: number, label: string) => {
        log.push(`resident:${label}`);
        return Promise.resolve(resident(bytes, label));
      },
      beginBatch: () => {
        log.push("batch:begin");
        return Promise.resolve({
          finish: () => {
            log.push("batch:finish");
            return Promise.resolve();
          },
          // encoder の mu は `options.mu` で、VAE のフレームの常駐 `[12, T, 128, 128]`（チャネル優先）はチャネル s の
          // 平面を patchValue(s) で埋める。
          finishAndRead: (outputs: Record<string, { readonly byteLength: number }>) => {
            log.push("batch:read");
            return Promise.resolve(
              Object.fromEntries(
                Object.entries(outputs).map(([name, { byteLength }]) => {
                  const values = new Float32Array(byteLength / 4);
                  if (name === "mu") {
                    for (let at = 0; at < values.length; at += 1) {
                      values[at] = options.mu?.(at) ?? 0.25;
                    }
                    return [name, values.buffer];
                  }
                  const perChannel = values.length / 12;
                  for (let at = 0; at < values.length; at += 1) {
                    values[at] = patchValue(Math.floor(at / perChannel));
                  }
                  return [name, values.buffer];
                }),
              ),
            );
          },
        });
      },
      // 解放待ち（`settleReleasedMemory`）が触る面。
      device: {
        queue: {
          submit: () => {
            log.push("settle:submit");
          },
          onSubmittedWorkDone: () => {
            log.push("settle:done");
            return Promise.resolve();
          },
        },
      },
      onLost: () => () => {},
    },
    ownsGpu: false,
    config: CONFIG,
    sessionOptions: {},
    transformer: {
      createSession: () => {
        log.push("create:transformer");
        return Promise.resolve({
          run: (inputs: Record<string, Tensor>) => {
            ditRuns.push(inputs);
            const tokens = inputs.tokens;
            return Promise.resolve({
              out: {
                dtype: "f32",
                shape: tokens.shape,
                data: new Float32Array(tokens.data.length).fill(0.1),
              },
            });
          },
          diagnostics: () => ({}),
          dispose: () => Promise.resolve(),
        });
      },
    },
    vaeFirst: vaeSession("vae_decoder_first"),
    vaeNext: vaeSession("vae_decoder_next"),
    vaeEncoder: {
      pre: encoderSession("vae_encoder_pre"),
      attn: encoderSession("vae_encoder_attn"),
      post: encoderSession("vae_encoder_post"),
      contract: vaeEncoderContract,
    },
    layout,
    ropeBase,
    dit,
    textEmbeds: parseWanTextEmbeds(embedsAsset(3).buffer),
    text,
    onRunDiagnostics: (component: string) => {
      diagnosed.push(component);
    },
  };
  const pipeline: WanTi2vPipeline = Reflect.construct(WanTi2vPipeline, [state]);
  /**
   * 5 フレーム・2 step（DiT 4 回）の要求で回し、観測したイベントを `log` に・途中の潜在を `snapshots` に積む。
   * プロンプトは資産の経路が `cats`、GPU 経路がフィクスチャの固定プロンプト `boxing-cats`。
   */
  const generate = (request: Partial<WanTi2vGenerateRequest> = {}) =>
    pipeline.generate({
      prompt: gpuText ? wanParityCase("fixed-boxing-cats").text : "Two cats.",
      frames: 5,
      steps: 2,
      ...request,
      onEvent: (event) => {
        if (event.kind === "denoise-step") snapshots.push(event.copyLatents().data);
        log.push(
          event.kind === "stage"
            ? `${event.component}:${event.at}`
            : event.kind === "denoise-step"
            ? `step:${event.step}`
            : `tile:${event.tile}`,
        );
      },
    });
  return {
    pipeline,
    log,
    ditRuns,
    encoderEnqueues,
    vaeLatents,
    diagnosed,
    snapshots,
    generate,
    contract: vaeEncoderContract,
  };
};

describe("WanTi2vPipeline.generate（模擬 Session — 2.2 の世代の値が VAE の段の末尾まで届く）", () => {
  it("出力は [3, F, H, W] で unpatchify 済み（RGB の画素 (y, x) は patch 空間のチャネル c·4 + (x%2)·2 + (y%2)）", async () => {
    const { log, generate } = mockTi2v();
    const video = await generate();
    assertEquals([video.frames, video.width, video.height, video.fps], [5, 1280, 704, 24]);
    assertEquals(video.data.length, 3 * 5 * 704 * 1280);
    // 1280×704 の潜在 44×80 をタイル 16 で覆うと 28 枚（段 5 の凍結表と同じ本数）。
    assertEquals(log.filter((entry) => entry.startsWith("tile:")).length, 28);
    let worst = 0;
    let worstAt = -1;
    for (let channel = 0; channel < 3; channel += 1) {
      for (let frame = 0; frame < 5; frame += 1) {
        for (let y = 0; y < 704; y += 1) {
          const row = ((channel * 5 + frame) * 704 + y) * 1280;
          for (let x = 0; x < 1280; x += 1) {
            const expected = patchValue(channel * 4 + (x % 2) * 2 + (y % 2));
            const diff = Math.abs(video.data[row + x] - expected);
            if (diff > worst) [worst, worstAt] = [diff, row + x];
          }
        }
      }
    }
    // タイルの重なりのブレンドは同じ値どうしの重み付き和なので、f32 の丸め以内で元の値に戻る。
    assert(
      worst <= 1e-6,
      `unpatchify の並べ方と違う画素がある（最大差 ${worst}・添字 ${worstAt}）`,
    );
  });

  it("DiT の run は ti2v の 7 本で、tokens の幅は 192（潜在 48 ch の patch）・T2V の条件入力は全て偽のマスクと生成側と同じ時刻", async () => {
    const { ditRuns, generate } = mockTi2v();
    await generate();
    assertEquals(ditRuns.length, 4);
    const tokens = 2 * 22 * 40;
    for (const inputs of ditRuns) {
      assertEquals(Object.keys(inputs).sort(), [
        "condition_mask",
        "encoder_hidden_states",
        "rope_cos",
        "rope_sin",
        "timesteps_proj",
        "timesteps_proj_condition",
        "tokens",
      ]);
      assertEquals(inputs.tokens.shape, [1, tokens, 192]);
      const mask = inputs.condition_mask;
      assertEquals([mask.dtype, mask.shape], ["bool", [1, tokens, 1]]);
      assertEquals(mask.data.length, tokens);
      assert(mask.data.every((value) => value === 0), "T2V の条件マスクに真の要素がある");
      // 条件側の時刻は生成側と同じ配列（dit-loop の T2V の取り決め — 写しではない）。
      const condition = inputs.timesteps_proj_condition;
      assertEquals([condition.dtype, condition.shape], ["f32", inputs.timesteps_proj.shape]);
      assert(condition.data === inputs.timesteps_proj.data, "条件側の時刻が生成側と別の配列");
    }
  });

  it("832×480 の要求は Session を 1 本も張らずに 2.2 の受理集合の文言で ModelInputError", async () => {
    const { log, generate } = mockTi2v();
    await assertRejects(
      () => generate({ width: 832, height: 480 }),
      ModelInputError,
      "832×480 が受理集合（1280×704 / 704×1280）に無い",
    );
    assertEquals(log.filter((entry) => entry.startsWith("create:")), []);
  });

  it("集合の外のプロンプトの文言は WanTi2vPipeline.prompts を案内する", async () => {
    const { generate } = mockTi2v();
    const error = await assertRejects(
      () => generate({ prompt: "A ferret." }),
      ModelInputError,
    );
    assertStringIncludes(error.message, "WanTi2vPipeline.prompts で引ける");
  });

  it("dispose 済みの生成は WanTi2vPipeline: の文言で拒む", async () => {
    const { pipeline, generate } = mockTi2v();
    await pipeline.dispose();
    await assertRejects(
      () => generate(),
      Error,
      `${OWNER} dispose 済みでは生成できない`,
    );
  });
});

// ---------------------------------------------------------------------------------------------
// I2V（ADR 0121 決定 11 — 条件画像 → encoder の段 → 先頭の潜在フレームで条件づけた DiT の段）
// ---------------------------------------------------------------------------------------------

/** 画素ごとに値の違う RGB8（I2V の条件画像の模擬）。 */
const rgb8 = (width: number, height: number): Rgb8Image => {
  const data = new Uint8Array(width * height * 3);
  for (let index = 0; index < data.length; index += 1) data[index] = (index * 37) % 256;
  return { data, width, height };
};

/** 横長（16:9 — 1280×704 を選ぶ）と縦長（9:16 — 704×1280 を選ぶ）の条件画像。 */
const LANDSCAPE = rgb8(64, 36);
const PORTRAIT = rgb8(36, 64);

/** 1280×704・5 フレームの潜在の形（`[48, 2, 44, 80]`）と、先頭フレームのトークン数（`(704/32)·(1280/32)`）。 */
const I2V_SHAPE = [48, 2, 44, 80] as const;
const I2V_PLANE = 44 * 80;
const FIRST_FRAME_TOKENS = 880;

/** TS の型を通らない値を 1 欄だけ差した生成の要求（JS の呼び手の綴り違い・型のすり抜けの再現）。 */
const requestWith = (key: string, value: unknown): Partial<WanTi2vGenerateRequest> => {
  const request: Partial<WanTi2vGenerateRequest> = {};
  Object.defineProperty(request, key, { value, enumerable: true });
  return request;
};

/** 2 つの数値列がビット単位で同じか（`-0` と `+0`・NaN の払い出しも区別する）。 */
const sameBits = (actual: ArrayLike<number> & ArrayBufferView, expected: Float32Array): boolean =>
  actual.byteLength === expected.byteLength &&
  new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength).every(
    (byte, at) => byte === new Uint8Array(expected.buffer, expected.byteOffset)[at],
  );

/** 常駐でなくホストのテンソルであること（pre の入口は前処理の出力をそのまま受ける）。 */
const hostTensor = (input: RunInput | undefined): Tensor => {
  if (input === undefined || !("data" in input)) throw new Error("ホストのテンソルでない");
  return input;
};

/** mu が定数 0.25 のときの条件の潜在（encoder の段の出口 — `[48, 1, 44, 80]` の正規化）。 */
const conditionOf = (mu: number): Float32Array<ArrayBuffer> =>
  normalizeWanLatents(
    new Float32Array(48 * I2V_PLANE).fill(mu),
    WAN22_TI2V_FAMILY.generation.latents,
  );

/**
 * 先頭の潜在フレームを条件の潜在で**写して**置き換えたモデル入力（置き換えの式とは独立に組む — 先頭フレーム以外の
 * 潜在と条件が ±0 でなければ式と同じ値）。
 */
const copyFirstFrame = (latents: Float32Array, condition: Float32Array): Float32Array => {
  const out = Float32Array.from(latents);
  const [channels, frames] = I2V_SHAPE;
  for (let channel = 0; channel < channels; channel += 1) {
    out.set(
      condition.subarray(channel * I2V_PLANE, (channel + 1) * I2V_PLANE),
      channel * frames * I2V_PLANE,
    );
  }
  return out;
};

describe("WanTi2vPipeline.generate の I2V: 段の順と解放（模擬 Session）", () => {
  it("encoder の 3 Session（実行オプション {}）→ 常駐 3 本 → 1 本の batch で pre → attn → post → Session → 常駐の順に畳む → 解放を待つ → encoder の end → DiT", async () => {
    const { log, diagnosed, generate } = mockTi2v();
    await generate({ image: LANDSCAPE });
    assertEquals(log.slice(0, log.indexOf("create:transformer") + 1), [
      "vae_encoder:start",
      "create:vae_encoder_pre:{}",
      "create:vae_encoder_attn:{}",
      "create:vae_encoder_post:{}",
      "resident:wan vae_encoder_pre",
      "resident:wan vae_encoder_attn",
      "resident:wan vae_encoder_post",
      "batch:begin",
      "enqueue:vae_encoder_pre",
      "enqueue:vae_encoder_attn",
      "enqueue:vae_encoder_post",
      "batch:read",
      "dispose:vae_encoder_pre",
      "dispose:vae_encoder_attn",
      "dispose:vae_encoder_post",
      "release:wan vae_encoder_pre",
      "release:wan vae_encoder_attn",
      "release:wan vae_encoder_post",
      "settle:submit",
      "settle:done",
      "vae_encoder:end",
      "transformer:start",
      "create:transformer",
    ]);
    // 診断は Session 1 本 = 1 名（encoder は 1 回ずつ・段の順）。
    assertEquals(diagnosed.slice(0, 4), [
      "vae_encoder_pre",
      "vae_encoder_attn",
      "vae_encoder_post",
      "transformer",
    ]);
    assertEquals(log.at(-1), "vae_decoder:end");
  });

  it("gpu の経路: encoder の段を畳んで解放を待ってから umT5 を張る（encoder → text → DiT → VAE）", async () => {
    const { log, generate } = mockTi2v({ textEncoder: "gpu" });
    await generate({ image: LANDSCAPE });
    assertEquals(
      log.filter((entry) => /^(vae_encoder|text_encoder|transformer|vae_decoder):/.test(entry)),
      [
        "vae_encoder:start",
        "vae_encoder:end",
        "text_encoder:start",
        "text_encoder:end",
        "transformer:start",
        "transformer:end",
        "vae_decoder:start",
        "vae_decoder:end",
      ],
    );
    assert(
      log.indexOf("settle:done") < log.indexOf("create:text_encoder"),
      `解放を待つ前に umT5 を張った: ${log.slice(0, 30)}`,
    );
  });

  it("画像の無い要求は encoder の段を持たない（Session・常駐・batch・解放待ちのどれも無い — T2V と同じ経路）", async () => {
    const { log, encoderEnqueues, diagnosed, generate } = mockTi2v();
    await generate();
    assertEquals(log.filter((entry) => entry.includes("vae_encoder")), []);
    assertEquals(log.filter((entry) => entry.startsWith("settle:")), []);
    assertEquals(encoderEnqueues, []);
    assertEquals(diagnosed[0], "transformer");
  });

  it("encoder の batch の enqueue が落ちたら区間を閉じ、3 Session と常駐を畳んで解放を待ち、本体の失敗をそのまま投げる（DiT へ進まない）", async () => {
    const { log, generate } = mockTi2v({ failEnqueue: "vae_encoder_attn" });
    const error = await assertRejects(() => generate({ image: LANDSCAPE }));
    assertStrictEquals(error, ENQUEUE_FAILURE);
    const after = log.slice(log.indexOf("enqueue:vae_encoder_attn"));
    // 失敗した段も解放を待つ（同じ pipeline の次の生成の DiT の段に encoder の確保を残さない）。
    assertEquals(after, [
      "enqueue:vae_encoder_attn",
      "batch:finish",
      "dispose:vae_encoder_pre",
      "dispose:vae_encoder_attn",
      "dispose:vae_encoder_post",
      "release:wan vae_encoder_pre",
      "release:wan vae_encoder_attn",
      "release:wan vae_encoder_post",
      "settle:submit",
      "settle:done",
    ]);
  });
});

describe("WanTi2vPipeline.generate の I2V: encoder の受け渡し（模擬 Session）", () => {
  it("pre は前処理の出力 [12, 1, 352, 640] をそのまま受け、attn は S = 3520・post は h 44 / w 80 を bindings で受け、前段の copyOutputs の常駐が次段の入力（同じ batch）", async () => {
    const { encoderEnqueues, generate } = mockTi2v();
    await generate({ image: LANDSCAPE });
    assertEquals(encoderEnqueues.map(({ part }) => part), [
      "vae_encoder_pre",
      "vae_encoder_attn",
      "vae_encoder_post",
    ]);
    const [pre, attn, post] = encoderEnqueues;
    const image = hostTensor(pre.inputs.image);
    const expected = preprocessWanI2vImage(LANDSCAPE, {}, WAN22_TI2V_FAMILY.generation);
    assertEquals([...image.shape], [12, 1, 352, 640]);
    assert(sameBits(image.data, expected.pixels), "pre の入口が前処理の出力と違う");
    assertEquals(pre.options.bindings, undefined);
    const preOut = pre.options.copyOutputs?.out;
    const attnOut = attn.options.copyOutputs?.out;
    const mu = post.options.copyOutputs?.out;
    assert(preOut !== undefined && attnOut !== undefined && mu !== undefined);
    assertStrictEquals(attn.inputs.tokens, preOut);
    assertEquals(attn.options.bindings, { S: 3520 });
    assertStrictEquals(post.inputs.hidden, attnOut);
    assertEquals(post.options.bindings, { h: 44, w: 80 });
    // 大きさは宣言 shape ぶん（mid 640 × S・mu 48 × S — f32）。
    assertEquals(
      [preOut.byteLength, attnOut.byteLength, mu.byteLength],
      [640 * 3520 * 4, 640 * 3520 * 4, 48 * 3520 * 4],
    );
    assert(pre.options.batch === attn.options.batch && attn.options.batch === post.options.batch);
  });

  it("縦長の画像は 704×1280 で回す（pre の入口 [12, 1, 640, 352]・post の h 80 / w 44）", async () => {
    const { encoderEnqueues, generate } = mockTi2v();
    const video = await generate({ image: PORTRAIT });
    assertEquals([video.width, video.height], [704, 1280]);
    const [pre, , post] = encoderEnqueues;
    assertEquals([...hostTensor(pre.inputs.image).shape], [12, 1, 640, 352]);
    assertEquals(post.options.bindings, { h: 80, w: 44 });
  });

  for (
    const [label, mu, message] of [
      [
        "NaN",
        (index: number) => index === 0 ? Number.NaN : 0.25,
        "channel 0・位置 (x=0, y=0) が非有限（NaN）",
      ],
      [
        "Infinity",
        (index: number) => index === I2V_PLANE + 80 + 1 ? Number.POSITIVE_INFINITY : 0.25,
        "channel 1・位置 (x=1, y=1) が非有限（Infinity）",
      ],
    ] as const
  ) {
    it(`条件の潜在（mu の正規化の後）が ${label} なら encoder の段で落ち、DiT の段を張らない（encoder は畳む）`, async () => {
      const { log, generate } = mockTi2v({ mu });
      await assertRejects(
        () => generate({ image: LANDSCAPE }),
        Error,
        `${OWNER} 条件画像の潜在（VAE encoder の mu の正規化の後）の ${message}`,
      );
      assert(!log.includes("create:transformer"), `DiT の Session を張った: ${log}`);
      assert(!log.includes("vae_encoder:end"), "途中で落ちた段の end を出した");
      for (const part of ["vae_encoder_pre", "vae_encoder_attn", "vae_encoder_post"]) {
        assert(log.includes(`dispose:${part}`), `${part} を畳んでいない`);
        assert(log.includes(`release:wan ${part}`), `${part} の常駐を返していない`);
      }
    });
  }
});

describe("WanTi2vPipeline.generate の I2V: DiT の条件づけ（模擬 Session — diffusers の expand_timesteps の形）", () => {
  it("条件マスクは先頭の潜在フレームの 880 トークン（1280×704 — (H/32)·(W/32)）だけが真で、ループの前に 1 回だけ作る", async () => {
    const { ditRuns, generate } = mockTi2v();
    await generate({ image: LANDSCAPE });
    assertEquals(ditRuns.length, 4);
    const masks = new Set<unknown>();
    for (const inputs of ditRuns) {
      const mask = inputs.condition_mask;
      assertEquals([mask.dtype, [...mask.shape]], ["bool", [1, 2 * FIRST_FRAME_TOKENS, 1]]);
      const head = mask.data.subarray(0, FIRST_FRAME_TOKENS);
      const tail = mask.data.subarray(FIRST_FRAME_TOKENS);
      assert(head.every((value) => value === 1), "先頭フレームのトークンに偽がある");
      assert(tail.every((value) => value === 0), "先頭フレームの外に真がある");
      masks.add(mask.data);
    }
    assertEquals(masks.size, 1, "条件マスクを run ごとに作り直した");
  });

  it("条件側の時刻は t = 0 の proj（生成側とは別の配列・全 run で同じ 1 本）", async () => {
    const { ditRuns, generate } = mockTi2v();
    await generate({ image: LANDSCAPE });
    const zero = timestepsProj(0, 256);
    const projs = new Set<unknown>();
    for (const inputs of ditRuns) {
      const condition = inputs.timesteps_proj_condition;
      assertEquals([condition.dtype, [...condition.shape]], ["f32", [1, 256]]);
      assert(sameBits(condition.data, zero), "条件側の時刻が t = 0 の proj でない");
      assert(condition.data !== inputs.timesteps_proj.data, "条件側の時刻が生成側と同じ配列");
      assert(!sameBits(inputs.timesteps_proj.data, zero), "生成側の時刻まで 0 になっている");
      projs.add(condition.data);
    }
    assertEquals(projs.size, 1, "条件側の時刻を run ごとに作り直した");
  });

  it("モデル入力は先頭の潜在フレームだけ条件の潜在・UniPC は置き換えない潜在で進み、copyLatents はその状態（先頭フレームは置き換わらない）", async () => {
    const { ditRuns, snapshots, generate } = mockTi2v();
    const count = I2V_SHAPE.reduce((product, dim) => product * dim, 1);
    // 先頭フレームも条件と違う値（UniPC が置き換えた潜在で進むと step 1 の先頭フレームが変わる）。
    const initial = new Float32Array(count).map((_, index) => Math.fround(Math.sin(index + 1) / 2));
    await generate({ image: LANDSCAPE, latents: initial, guidance: 1 });
    const condition = conditionOf(0.25);
    // 模擬の DiT は 0.1 を返し、guidance 1 は CFG を回さないので、速度は全要素 0.1。
    const sampler = new WanUniPcSampler(
      wanUniPcSchedule(2, CONFIG.scheduler.shift, WAN_UNIPC_CONFIG.numTrainTimesteps),
      WAN_UNIPC_CONFIG,
    );
    const velocity = new Float32Array(count).fill(0.1);
    const first = sampler.step(velocity, initial);
    const second = sampler.step(velocity, first);
    const patch = wanDitPatch(48);
    assertEquals(ditRuns.length, 2);
    for (const [step, latents] of [[0, initial], [1, first]] as const) {
      assert(
        sameBits(
          ditRuns[step].tokens.data,
          patchifyLatents(copyFirstFrame(latents, condition), I2V_SHAPE, patch),
        ),
        `step ${step + 1} のモデル入力が「先頭フレームだけ条件」でない`,
      );
    }
    assertEquals(snapshots.length, 2);
    assert(
      sameBits(snapshots[0], first),
      "step 1 の copyLatents が置き換えない UniPC の潜在でない",
    );
    assert(
      sameBits(snapshots[1], second),
      "step 2 の copyLatents が置き換えない UniPC の潜在でない",
    );
  });

  it("CFG の 2 パス（guidance 5）は uncond と cond の両方が「先頭フレームだけ条件」のモデル入力を受ける", async () => {
    const { ditRuns, generate } = mockTi2v();
    const count = I2V_SHAPE.reduce((product, dim) => product * dim, 1);
    const initial = new Float32Array(count).map((_, index) => Math.fround(Math.sin(index + 1) / 2));
    await generate({ image: LANDSCAPE, latents: initial, guidance: 5 });
    const condition = conditionOf(0.25);
    // 模擬の DiT は両パスとも 0.1 を返すので、CFG の合成も全要素 0.1（合成は製品の関数で作る）。
    const noise = new Float32Array(count).fill(0.1);
    const velocity = wanClassifierFreeGuidance(noise, noise, 5);
    const sampler = new WanUniPcSampler(
      wanUniPcSchedule(2, CONFIG.scheduler.shift, WAN_UNIPC_CONFIG.numTrainTimesteps),
      WAN_UNIPC_CONFIG,
    );
    const first = sampler.step(velocity, initial);
    const patch = wanDitPatch(48);
    // run の並びは step ごとに uncond → cond（B = 1 の逐次 2 回）。
    assertEquals(ditRuns.length, 4);
    ditRuns.forEach((inputs, run) => {
      const step = Math.floor(run / 2);
      const latents = step === 0 ? initial : first;
      assert(
        sameBits(
          inputs.tokens.data,
          patchifyLatents(copyFirstFrame(latents, condition), I2V_SHAPE, patch),
        ),
        `step ${step + 1} の ${
          run % 2 === 0 ? "uncond" : "cond"
        } のモデル入力が「先頭フレームだけ条件」でない`,
      );
    });
  });

  it("ループの後に 1 回置き換えてから VAE へ渡す（1 タイル目の chunk 0 = 条件の潜在・chunk 1 = 最後の UniPC の潜在 — どちらも逆正規化）", async () => {
    const { snapshots, vaeLatents, generate } = mockTi2v();
    const count = I2V_SHAPE.reduce((product, dim) => product * dim, 1);
    const initial = new Float32Array(count).map((_, index) => Math.fround(Math.sin(index + 1) / 2));
    await generate({ image: LANDSCAPE, latents: initial, guidance: 1 });
    const last = snapshots.at(-1);
    assert(last !== undefined);
    const denormalized = denormalizeWanLatents(
      copyFirstFrame(last, conditionOf(0.25)),
      WAN22_TI2V_FAMILY.generation.latents,
    );
    // 1 タイル目（潜在の左上 16×16）の chunk 0（first）と chunk 1（next）の `latent [48, 1, 16, 16]`。
    const tile = (frame: number): Float32Array => {
      const out = new Float32Array(48 * 16 * 16);
      for (let channel = 0; channel < 48; channel += 1) {
        for (let y = 0; y < 16; y += 1) {
          const from = ((channel * 2 + frame) * 44 + y) * 80;
          out.set(denormalized.subarray(from, from + 16), (channel * 16 + y) * 16);
        }
      }
      return out;
    };
    assert(sameBits(vaeLatents[0].data, tile(0)), "VAE へ渡した先頭フレームが条件の潜在でない");
    assert(
      sameBits(vaeLatents[1].data, tile(1)),
      "VAE へ渡した 2 フレーム目が最後の UniPC の潜在でない",
    );
  });

  it("公式の形（初期と各 step の後に置き換え）と diffusers の形（モデル入力だけ・最後に 1 回）の最終の潜在はビット一致する（UniPC は要素ごとの更新）", () => {
    const shape = [3, 3, 4, 4] as const;
    const plane = shape[2] * shape[3];
    const count = shape.reduce((product, dim) => product * dim, 1);
    const initial = new Float32Array(count).map((_, index) => Math.fround(Math.cos(index * 0.7)));
    const condition = new Float32Array(shape[0] * plane).map((_, index) =>
      Math.fround(Math.sin(index * 1.3) * 0.8 + 0.05)
    );
    // 要素をまたぐ擬似 DiT（全要素の平均と離れた要素を混ぜる — 置き換えた要素が他の要素の速度に効く）。
    const model = (input: Float32Array): Float32Array => {
      const mean = input.reduce((total, value) => total + value, 0) / input.length;
      return input.map((value, index) =>
        Math.fround(0.3 * value - 0.2 * input[(index + 5) % input.length] + 0.1 * mean)
      );
    };
    for (const steps of [2, 10, 50]) {
      const schedule = wanUniPcSchedule(steps, 5, WAN_UNIPC_CONFIG.numTrainTimesteps);
      const replace = (latents: Float32Array) =>
        withWanFirstFrameCondition(latents, condition, shape);
      const diffusers = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
      let current: Float32Array = initial;
      for (let step = 0; step < steps; step += 1) {
        current = diffusers.step(model(replace(current)), current);
      }
      const diffusersFinal = replace(current);
      const official = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
      let replaced: Float32Array = replace(initial);
      for (let step = 0; step < steps; step += 1) {
        replaced = replace(official.step(model(replaced), replaced));
      }
      assert(sameBits(replaced, diffusersFinal), `${steps} step で最終の潜在が割れた`);
    }
  });
});

describe("WanTi2vPipeline.generate の I2V: 要求の門（模擬 Session — Session を 1 本も張らずに落ちる）", () => {
  const rejects = async (
    request: Partial<WanTi2vGenerateRequest>,
    errorClass: new (message: string) => Error,
    message: string,
  ): Promise<Error> => {
    const { log, generate } = mockTi2v();
    const error = await assertRejects(() => generate(request), errorClass, message);
    assertEquals(log.filter((entry) => entry.startsWith("create:")), []);
    return error;
  };

  it("image 無しの fit は ModelInputError", async () => {
    await rejects(
      { fit: "stretch" },
      ModelInputError,
      "fit は image を渡したときだけ効く（条件画像の寸法の合わせ方 — image 無しでは渡さない）",
    );
  });

  it("fit の綴り違いは素の Error（ModelInputError ではない — 名前の綴り違いの扱い）", async () => {
    const error = await rejects(
      { ...requestWith("fit", "cover"), image: LANDSCAPE },
      Error,
      "fit 'cover' は 'crop' / 'stretch' のどちらでもない",
    );
    assert(!(error instanceof ModelInputError));
  });

  it("fit の綴り違いは image の有無より先に見る（image 無しの 'cover' も素の Error）", async () => {
    const error = await rejects(
      requestWith("fit", "cover"),
      Error,
      "fit 'cover' は 'crop' / 'stretch' のどちらでもない",
    );
    assert(!(error instanceof ModelInputError));
  });

  for (
    const [label, value, actual] of [
      ["null", null, "null"],
      ["数", 42, "number"],
      ["data の無い object", { width: 4, height: 4 }, "data が無い"],
      [
        "data が Float32Array",
        { width: 4, height: 4, data: new Float32Array(48) },
        "data が Float32Array",
      ],
    ] as const
  ) {
    it(`RGB8 の形でない image（${label}）は ModelInputError（JS の呼び手の型のすり抜け — 長さの検査を通る別の型の配列も）`, async () => {
      await rejects(
        requestWith("image", value),
        ModelInputError,
        `image が RGB8 の画像（data が Uint8Array・width・height の object）でない（実際: ${actual}`,
      );
    });
  }

  it("明示の寸法が受理集合の外なら ModelInputError（片方だけでも・両方でも）", async () => {
    await rejects(
      { image: LANDSCAPE, width: 832 },
      ModelInputError,
      "width 832 が受理集合（1280×704 / 704×1280）に無い",
    );
    await rejects(
      { image: LANDSCAPE, width: 832, height: 480 },
      ModelInputError,
      "832×480 が受理集合（1280×704 / 704×1280）に無い",
    );
  });

  it("RGB8 の長さが寸法と合わない画像は ModelInputError", async () => {
    await rejects(
      { image: { data: new Uint8Array(10), width: 4, height: 4 } },
      ModelInputError,
      "RGB8 の長さ 10 が 3×4×4 と違う",
    );
  });
});

describe("WAN22_TI2V_FAMILY の I2V の計画（計画の門）", () => {
  const embeds: WanTextEmbeds = parseWanTextEmbeds(embedsAsset(3).buffer);
  const graphs = vaeChunkGraphs();
  const layout = wanVaeChunkLayout(graphs.first, graphs.next);
  const plan = (request: Partial<WanTi2vGenerateRequest>) =>
    planWanGeneration(
      { prompt: "Two cats.", ...request },
      embeds,
      layout,
      CONFIG,
      WAN22_TI2V_FAMILY.generation,
      WAN22_TI2V_FAMILY.owner,
    );

  it("寸法を省けば画像の縦横比で選ぶ（横長 → 1280×704・縦長 → 704×1280）・潜在の形もその寸法", () => {
    const landscape = plan({ image: LANDSCAPE });
    assertEquals([landscape.width, landscape.height], [1280, 704]);
    assertEquals(landscape.latentShape, [48, 9, 44, 80]);
    const portrait = plan({ image: PORTRAIT });
    assertEquals([portrait.width, portrait.height], [704, 1280]);
    assertEquals(portrait.latentShape, [48, 9, 80, 44]);
  });

  it("明示の寸法は画像の縦横比より優先する（横長の画像に 704×1280・縦長の画像に width 1280）", () => {
    const explicit = plan({ image: LANDSCAPE, width: 704, height: 1280 });
    assertEquals([explicit.width, explicit.height], [704, 1280]);
    assertEquals(explicit.conditionImage?.shape, [12, 1, 640, 352]);
    const widthOnly = plan({ image: PORTRAIT, width: 1280 });
    assertEquals([widthOnly.width, widthOnly.height], [1280, 704]);
  });

  it("fit を省けば crop（公式）・stretch は直接の伸縮（前処理の結果がそれぞれと一致）", () => {
    const generation = WAN22_TI2V_FAMILY.generation;
    for (const fit of [undefined, "crop", "stretch"] as const) {
      const planned = plan({ image: LANDSCAPE, ...(fit === undefined ? {} : { fit }) });
      const expected = preprocessWanI2vImage(LANDSCAPE, { fit: fit ?? "crop" }, generation);
      assert(planned.conditionImage !== undefined);
      assert(sameBits(planned.conditionImage.pixels, expected.pixels), `fit ${fit}`);
    }
    // crop と stretch は 64×36（16:9）→ 1280×704（20:11）で別の画素になる（同じ値なら上の比較が空振りする）。
    const crop = preprocessWanI2vImage(LANDSCAPE, { fit: "crop" }, generation);
    const stretch = preprocessWanI2vImage(LANDSCAPE, { fit: "stretch" }, generation);
    assert(!sameBits(crop.pixels, stretch.pixels));
  });

  it("画像の無い要求は条件画像を持たない（T2V の計画は画像の欄が無いときと同じ）", () => {
    assertEquals(plan({}).conditionImage, undefined);
  });
});

describe("wanVaeEncoderContract（encoder の 3 グラフの家族 admission の門）", () => {
  const graphs = vaeChunkGraphs();
  const layout = wanVaeChunkLayout(graphs.first, graphs.next);
  const contract = (replaced: Readonly<Record<string, EncoderGraph>> = {}) =>
    wanVaeEncoderContract(
      encoderGraphs(replaced),
      layout,
      WAN22_TI2V_FAMILY.generation,
      WAN22_TI2V_FAMILY.owner,
    );

  it("対照: 系列の宣言は通り、記号の名前・チャネル数・入口の倍率をグラフから読む", () => {
    const admitted = contract();
    assertEquals(
      [admitted.hiddenChannels, admitted.latentChannels, admitted.patchChannels],
      [640, 48, 12],
    );
    assertEquals([admitted.inputScale, admitted.compression], [8, 16]);
    assertEquals(admitted.attn.sequence, "S");
    assertEquals([admitted.post.height, admitted.post.width], ["h", "w"]);
  });

  it("pre の入口の倍率が decoder の空間の圧縮（16 ÷ patchify 2 = 8）と違えば落ちる", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_pre: {
            ...ENCODER_GRAPHS.vae_encoder_pre,
            input: { name: "image", shape: [12, 1, "4h", "4w"] },
          },
        }),
      Error,
      `${OWNER} vae_encoder_pre の入力 'image' の形 [12, 1, 4h, 4w] がホストの組む [12, 1, 8h, 8w] と違う`,
    );
  });

  it("post の出口のチャネルが decoder の潜在 48 と違えば落ちる", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_post: { ...ENCODER_GRAPHS.vae_encoder_post, output: [96, 1, "h", "w"] },
        }),
      Error,
      `${OWNER} vae_encoder_post の出力 'out' の形 [96, 1, h, w] がホストの組む [48, 1, h, w] と違う`,
    );
  });

  it("attn が系列長の記号のほかに記号を持てば落ちる（束縛の源はホストが渡す S だけ）", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_attn: { ...ENCODER_GRAPHS.vae_encoder_attn, symbols: ["S", "T"] },
        }),
      Error,
      `${OWNER} vae_encoder_attn の記号が [S, T]（期待: [S]）`,
    );
  });

  it("post の入口の空間の軸が記号そのものでなければ落ちる（h, w を bindings で渡せない）", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_post: {
            ...ENCODER_GRAPHS.vae_encoder_post,
            input: { name: "hidden", shape: [640, 1, "2h", "w"] },
          },
        }),
      Error,
      `${OWNER} vae_encoder_post の入力 'hidden' の軸 2 が記号そのものでない（2h）`,
    );
  });
});

describe("WanTi2vPipeline.fromAssets: I2V の VAE encoder の部品（家族 admission）", () => {
  it("encoder の部品が無い 2.2 の資産は、部品 'vae_encoder_pre' の容器が無い旨で落ちる（画像を渡さない使い方でも取る）", async () => {
    const assets = Object.fromEntries(
      Object.entries(VALID_ASSETS).filter(([key]) => !key.startsWith("vae_encoder_")),
    );
    await assertRejects(
      () =>
        WanTi2vPipeline.fromAssets(
          { manifest: ti2vManifest(), assets },
          { textEncoder: "precomputed", gpu: fakeGpuContext(fakeDevice()) },
        ),
      Error,
      `${OWNER} 部品 'vae_encoder_pre' の容器が無い`,
    );
  });

  it("encoder の宣言の食い違い（pre の入口の倍率 4）は構築で落ちる（admission が encoder の門を通す）", async () => {
    const encoder = await encoderAssets({
      vae_encoder_pre: {
        ...ENCODER_GRAPHS.vae_encoder_pre,
        input: { name: "image", shape: [12, 1, "4h", "4w"] },
      },
    });
    await assertRejects(
      () => buildTi2v(ti2vManifest(), encoder),
      Error,
      `${OWNER} vae_encoder_pre の入力 'image' の形 [12, 1, 4h, 4w] がホストの組む [12, 1, 8h, 8w] と違う`,
    );
  });
});
