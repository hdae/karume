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
//
// 実 GPU の通しは e2e（系列から組む helper — ADR 0121 段 6 のコミット 6）。

import {
  assert,
  assertEquals,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { type DistributionSource, localDirectory, type Manifest, parseManifest } from "@karume/hub";
import type { Tensor } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import {
  type WanAssets,
  type WanGenerateRequest,
  WanPipeline,
  type WanPipelineOptions,
} from "../src/wan/pipeline.ts";
import { WanTi2vPipeline } from "../src/wan/ti2v-pipeline.ts";
import { planWanGeneration, WAN22_TI2V_FAMILY } from "../src/wan/family.ts";
import { ditContract, wanDitPatch } from "../src/wan/dit-loop.ts";
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

const COMPONENTS = ["transformer", "vae_decoder_first", "vae_decoder_next"];

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

/** 2.2 の宣言の資産一式（48 ch・出口 12 ch の VAE・7 入力の DiT・埋め込み幅 4096）。 */
const VALID_ASSETS: WanAssets["assets"] = {
  ...await ditAssets(TI2V_DIT_INPUTS),
  ...await vaeAssets("vae_decoder_first", 1, { latent: 48, sample: 12 }),
  ...await vaeAssets("vae_decoder_next", 4, { latent: 48, sample: 12 }),
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
  const embeds = embedsAsset(4096);
  const embedsPath = "text_embeds.safetensors";
  const models = {
    test: {
      pipeline,
      weights: {
        transformer: { f16: transformer.entry },
        vae_decoder_first: { f16: first.entry },
        vae_decoder_next: { f16: next.entry },
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

/**
 * 模擬 Session で回す `WanTi2vPipeline`。DiT は定数を返し、VAE は 1 タイル = 1 batch の手順を偽の
 * GpuContext で回して、読み戻すフレームを**チャネルごとの値**（{@link patchValue}）で埋める — unpatchify の
 * 並べ方が出力の画素の値に出る。
 *
 * NOTE: コンストラクタは TS の `private` なので `Reflect.construct` で内部状態（family.ts の `WanState`）を
 * 直接渡す（private の迂回はテストだけ — `wan_pipeline_test.ts` の mockPipeline と同じ理由）。
 */
const mockTi2v = () => {
  const log: string[] = [];
  const ditRuns: Record<string, Tensor>[] = [];
  const { first, next } = vaeChunkGraphs();
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
  const layout = wanVaeChunkLayout(first, next);
  // 行数 64 は潜在 [48, 2, 44, 80] の格子（2 × 22 × 40）を覆う。値は門と無関係。
  const ropeBase: WanRopeBase = {
    rows: 64,
    widths: [1, 1, 1],
    cos: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
    sin: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
  };
  // DiT の取り決めは家族 admission と同じ関数・同じ値（patch は VAE の宣言のチャネル数・入力の形と
  // owner は spec）で組む — spec の入力の形が 2.1 へ戻ると、7 入力の宣言はここで落ちる。
  const dit = ditContract(
    stubModel({
      symbols: ["S"],
      inputs: [
        { name: "tokens", shape: [1, "S", 192] },
        { name: "timesteps_proj", shape: [1, 256] },
        { name: "encoder_hidden_states", shape: [1, 4, 3] },
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
  const state = {
    gpu: {
      createResident: (bytes: number) => Promise.resolve(resident(bytes)),
      beginBatch: () =>
        Promise.resolve({
          finish: () => Promise.resolve(),
          // フレームの常駐は `[12, T, 128, 128]`（チャネル優先）— チャネル s の平面を patchValue(s) で埋める。
          finishAndRead: (frames: Record<string, { readonly byteLength: number }>) =>
            Promise.resolve(
              Object.fromEntries(
                Object.entries(frames).map(([name, { byteLength }]) => {
                  const values = new Float32Array(byteLength / 4);
                  const perChannel = values.length / 12;
                  for (let at = 0; at < values.length; at += 1) {
                    values[at] = patchValue(Math.floor(at / perChannel));
                  }
                  return [name, values.buffer];
                }),
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
    layout,
    ropeBase,
    dit,
    textEmbeds: parseWanTextEmbeds(embedsAsset(3).buffer),
    text: { kind: "precomputed" },
  };
  const pipeline: WanTi2vPipeline = Reflect.construct(WanTi2vPipeline, [state]);
  /** 5 フレーム・2 step（DiT 4 回）の要求で回し、観測したイベントを `log` に積む。 */
  const generate = (request: Partial<WanGenerateRequest> = {}) =>
    pipeline.generate({
      prompt: "Two cats.",
      frames: 5,
      steps: 2,
      ...request,
      onEvent: (event) => {
        log.push(
          event.kind === "stage"
            ? `${event.component}:${event.at}`
            : event.kind === "denoise-step"
            ? `step:${event.step}`
            : `tile:${event.tile}`,
        );
      },
    });
  return { pipeline, log, ditRuns, generate };
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
