// `Sbv2Pipeline` の**構築ガード**。GPU も実資産も要らない範囲だけを見る（実 GPU の突合は
// example の dump 経路と golden E2E が持つ）。
//
// ここで押さえるのは 2 つ:
//  ① `fromAssets` は GPU を取りに行く**前**に manifest の契約違反を落とす（pipeline 名 /
//     未知 major / 未知 preset / pipelineConfig のスキーマ）。落とす位置がずれると、GPU の
//     無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
//  ② `styles` / `speakers` は「名前 → **表の行番号**」で、値が `0..件数-1` の順列であること。
//     行番号がずれても `style_vec` / `g` の shape は合ったままなので、**別のスタイル・別の
//     話者の声が出る**だけで沈黙する（`src/sbv2/config.ts` の doc）。既定が受理集合の外を
//     指すのも同じクラスで、どちらも parse 時に落とす。

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { Sbv2Pipeline, toSessionOptions } from "../src/sbv2/pipeline.ts";
import { parseSbv2PipelineConfig } from "../src/sbv2/config.ts";

const FILE = {
  path: "front/model.i8.safetensors",
  size: 16,
  sha256: "a".repeat(64),
};

/** `models/sbv2-FN4/karume.json` の骨格（検査に要る欄だけ）。 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/1",
    generator: "karume/0.1.0",
    pipeline: "sbv2/1",
    components: { front: { file: FILE } },
    presets: { w8: { weights: {}, session: {} } },
    defaultPreset: "w8",
    pipelineConfig: {
      styles: { Neutral: 0, high: 1 },
      speakers: { FN4: 0 },
      defaults: {
        speaker: "FN4",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    },
    ...patch,
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

/** `pipelineConfig` だけを差し替えた manifest（残りは骨格のまま）。 */
const withConfig = (config: Record<string, unknown>): string =>
  manifestText({ pipelineConfig: config });

Deno.test("fromAssets: pipeline の契約名が sbv2 でない manifest を落とす", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "anima/1" }));
  await assertRejects(
    () => Sbv2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "'anima/1'",
  );
});

Deno.test("fromAssets: 未知 major は fail loudly（検査責務は models 側 — ADR 0038 §1）", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/2" }));
  // hub は `pipeline` の major を検査しない（読めるかどうかはパイプライン実装しか知らない）。
  assertEquals(manifest.pipeline, { name: "sbv2", major: 2 });
  await assertRejects(
    () => Sbv2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "major に未対応",
  );
});

Deno.test("fromAssets: 存在しない preset は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Sbv2Pipeline.fromAssets({ manifest, assets: emptyAssets }, { preset: "f16" }),
    Error,
    "利用可能: w8",
  );
});

Deno.test("fromAssets: pipelineConfig の未知キーは構築時に落ちる", async () => {
  const manifest = parseManifest(
    withConfig({
      styles: { Neutral: 0 },
      speakers: { FN4: 0 },
      // 配布形に無い節。綴り違いが黙って既定へ縮退する経路を作らない。
      sampleRate: 44100,
      defaults: {
        speaker: "FN4",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    }),
  );
  await assertRejects(
    () => Sbv2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig: 未知キー 'sampleRate'",
  );
});

Deno.test("fromAssets: defaults.style が styles に無ければ構築時に落ちる", async () => {
  const manifest = parseManifest(
    withConfig({
      styles: { Neutral: 0, high: 1 },
      speakers: { FN4: 0 },
      defaults: {
        speaker: "FN4",
        style: "Angry",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    }),
  );
  // 「生成を 1 回走らせて初めて分かる」を作らない — 既定は受理集合の内側であること。
  await assertRejects(
    () => Sbv2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig.defaults.style: 'Angry' が styles に無い",
  );
});

Deno.test("fromAssets: defaults.speaker が speakers に無ければ構築時に落ちる", async () => {
  const manifest = parseManifest(
    withConfig({
      styles: { Neutral: 0 },
      speakers: { FN4: 0 },
      defaults: {
        speaker: "jvnv-F1-jp",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    }),
  );
  await assertRejects(
    () => Sbv2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig.defaults.speaker: 'jvnv-F1-jp' が speakers に無い",
  );
});

Deno.test("fromAssets: 行番号が順列でない styles は構築時に落ちる（沈黙誤値クラス）", async () => {
  // 件数 2 に対して行番号が {0, 5}。表の行数（2）に対して 5 行目は無いので、通せば
  // 「範囲外の行を引く」か「別の行を引く」のどちらかになる。
  const manifest = parseManifest(
    withConfig({
      styles: { Neutral: 0, high: 5 },
      speakers: { FN4: 0 },
      defaults: {
        speaker: "FN4",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    }),
  );
  await assertRejects(
    () => Sbv2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "行番号が 0..1 の整数でない",
  );
});

Deno.test("parseSbv2PipelineConfig: 行番号の重複を落とす（名前と行が 1 対 1 でない）", () => {
  // 範囲内に収まっていても重複していれば表の 1 行が誰からも指されない = 別のスタイルの声。
  assertThrows(
    () =>
      parseSbv2PipelineConfig({
        styles: { Neutral: 0, high: 0 },
        speakers: { FN4: 0 },
        defaults: {
          speaker: "FN4",
          style: "Neutral",
          styleWeight: 1,
          sdpRatio: 0.2,
          noiseScale: 0.6,
          noiseScaleW: 0.8,
          lengthScale: 1,
        },
      }),
    Error,
    "行番号 0 が重複している",
  );
});

Deno.test("parseSbv2PipelineConfig: 配布形の pipelineConfig を読み切る", () => {
  // `models/sbv2-FN4/karume.json` の実物と同じ形。名前 → 行番号が Map で引けること。
  const config = parseSbv2PipelineConfig({
    styles: { Neutral: 0, high: 1, low: 2, NSFW: 3 },
    speakers: { FN4: 0 },
    defaults: {
      speaker: "FN4",
      style: "Neutral",
      styleWeight: 1,
      sdpRatio: 0.2,
      noiseScale: 0.6,
      noiseScaleW: 0.8,
      lengthScale: 1,
    },
  });
  assertEquals([...config.styles], [["Neutral", 0], ["high", 1], ["low", 2], ["NSFW", 3]]);
  assertEquals([...config.speakers], [["FN4", 0]]);
  assertEquals(config.defaults.style, "Neutral");
  assertEquals(config.defaults.lengthScale, 1);
});

Deno.test("parseSbv2PipelineConfig: lengthScale 0 は落とす（総フレーム数が 0 になる）", () => {
  assertThrows(
    () =>
      parseSbv2PipelineConfig({
        styles: { Neutral: 0 },
        speakers: { FN4: 0 },
        defaults: {
          speaker: "FN4",
          style: "Neutral",
          styleWeight: 1,
          sdpRatio: 0.2,
          noiseScale: 0.6,
          noiseScaleW: 0.8,
          lengthScale: 0,
        },
      }),
    Error,
    "pipelineConfig.defaults.lengthScale: 正の有限数でない",
  );
});

Deno.test("toSessionOptions: 3 キーを 1 つずつ写す（未指定は欄ごと作らない）", () => {
  // ADR 0038 §3 の綴りの契約。抜けは**沈黙劣化**（未知キーは runtime が黙って無視する）。
  assertEquals(toSessionOptions({}), {});
  assertEquals(toSessionOptions({ linearCompute: "i8a8" }), { linearCompute: "i8a8" });
  assertEquals(
    toSessionOptions({
      linearCompute: "i8a8",
      attentionCompute: "i8a8",
      attentionScoreStorage: "f16",
    }),
    { linearCompute: "i8a8", attentionCompute: "i8a8", attentionScoreStorage: "f16" },
  );
  // `submitPolicy`（TDR 予算 = ホスト政策）は manifest 側に無いので写さない。
  const mapped = toSessionOptions({ linearCompute: "i8a8" }) as Record<string, unknown>;
  assertEquals(Object.keys(mapped), ["linearCompute"]);
});
