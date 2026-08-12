// `IrodoriPipeline` の**構築ガード**。GPU も実資産も要らない範囲だけを見る（実 GPU の突合は
// `e2e_irodori_*_test.ts` 群が持つ）。
//
// ここで押さえるのは 1 点だけ: `fromAssets` は **manifest の契約違反を、資産を開く前・GPU を
// 取りに行く前**に落とす（`src/irodori/pipeline.ts` の `openIrodoriState` が掲げる MUST）。
// 順序がずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
//
// 観測の仕掛け: **全ケースで `assets` は空**。したがって
//  - manifest 契約の違反ケースが「その違反の文言」で落ちる  = 資産解析より前に落ちている
//  - 正しい manifest + 空 assets が `資産 'backbone' が無い` で落ちる = 契約検査が全部済んだ後に
//    初めて資産へ触る（上の対偶）
// の 2 つが噛み合って、門の順序そのものを縛る。グラフ宣言との 12 点突合はこの `assetBuffer` の
// さらに後段なので、合成 IR コンテナを組む器（`tests/helpers/` に無い）が要る — ここでは扱わない。

import { assertRejects } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { IrodoriPipeline } from "../src/irodori/pipeline.ts";

const FILE = {
  path: "dit/model.f32.safetensors",
  size: 16,
  sha256: "a".repeat(64),
};

/** グラフ資産の名前（`openIrodoriState` が `assetBuffer` で引く順に並べる）。 */
const WEIGHT_NAMES = [
  "backbone",
  "text_proj",
  "caption_proj",
  "speaker",
  "duration",
  "dit",
  "codec_decoder",
  "codec_encoder",
] as const;

/** `models/karume-irodori-v4-small/karume.json` の `pipelineConfig` 実物（23 欄）。 */
const PIPELINE_CONFIG: Record<string, unknown> = {
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

/** 配布形の骨格（検査に要る欄だけ）。`patch` は `models["v4-small"]` の中身を上書きする。 */
const manifestText = (patch: Record<string, unknown> = {}): string => {
  let weights: Record<string, unknown> = {};
  let mapping: Record<string, string> = {};
  for (const name of WEIGHT_NAMES) {
    weights = { ...weights, [name]: { f32: { file: { ...FILE, path: `${name}/model.f32` } } } };
    mapping = { ...mapping, [name]: "f32" };
  }
  return JSON.stringify({
    format: "karume/2",
    generator: "karume/0.1.0",
    defaultModel: "v4-small",
    models: {
      "v4-small": {
        pipeline: "irodori/1",
        weights,
        assets: { tokenizer: { ...FILE, path: "tokenizer.json" } },
        quants: { f32: { weights: mapping, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });
};

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

/** `pipelineConfig` だけを差し替えた manifest（残りは骨格のまま）。 */
const withConfig = (config: Record<string, unknown>): string =>
  manifestText({ pipelineConfig: config });

Deno.test("fromAssets: 存在しない model は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: emptyAssets }, { model: "nope" }),
    Error,
    "model 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipeline の契約名が irodori でない manifest を落とす", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/1" }));
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "manifest の pipeline が 'sbv2/1'",
  );
});

Deno.test("fromAssets: 未知 major は fail loudly（検査責務は models 側 — ADR 0038 §1）", async () => {
  // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門。hub は major を検査しない。
  const manifest = parseManifest(manifestText({ pipeline: "irodori/2" }));
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "major に未対応",
  );
});

Deno.test("fromAssets: 存在しない quant は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: emptyAssets }, { quant: "nope" }),
    Error,
    "quant 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipelineConfig の未知キーは構築時に落ちる", async () => {
  // 綴り違い（`steps` に対する `step`）が黙って既定へ縮退する経路を作らない（config.ts の MUST）。
  const manifest = parseManifest(withConfig({ ...PIPELINE_CONFIG, step: 40 }));
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig: 未知キー 'step'",
  );
});

Deno.test("fromAssets: pipelineConfig の欄が欠けていれば構築時に落ちる", async () => {
  // 欠けた欄が既定で埋まると、ホストだけが別の数を持ったまま **shape は合う**形で沈黙誤値になる
  // （config.ts 冒頭の MUST — モデル固有の数は manifest が正本）。
  const { hopLength: _dropped, ...missing } = PIPELINE_CONFIG;
  const manifest = parseManifest(withConfig(missing));
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig.hopLength: 無い",
  );
});

Deno.test("fromAssets: manifest 契約を全て満たして初めて資産へ触る（門の順序の対偶）", async () => {
  // 上の 6 ケースが「資産が空でも manifest の文言で落ちる」ことの裏返し。正しい manifest なら
  // 検査は資産まで進み、最初に引く `backbone` の不在で落ちる（= 契約検査は全て資産より前）。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "資産 'backbone' が無い",
  );
});
