// `pipelineConfig`（irodori）のスキーマ検証の挙動テスト。GPU も実資産も要らない。
//
// ここで押さえるのは 3 つ:
//  ① 未知キー / 欠落 / 値域外は **parse 時**に落ちる（配布形の綴り違いが黙って既定へ縮退する
//     経路を作らない — `src/irodori/config.ts` の doc）
//  ② `speakerUncondMode` / `cfgGuidanceMode` は **受理集合が 1 値**（ADR 0047 決定 1）。
//     対応外のモードは「値を保持して分岐」ではなく**拒否**で、型としても表せない
//  ③ 区間の宣言（cfgMinT/cfgMaxT・minSeconds/maxSeconds）が逆順なら落とす — 逆順は
//     「CFG が 1 度も掛からない」形で沈黙し、forward 数が減るだけで例外が出ない

import { assertEquals, assertThrows } from "@std/assert";
import {
  IRODORI_PIPELINE_MAJOR,
  IRODORI_PIPELINE_NAME,
  parseIrodoriPipelineConfig,
} from "../src/irodori/config.ts";

/** 実重み v4-small が要求する値（配布形が書く想定の骨格）。 */
const BASE: Record<string, unknown> = {
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

const withPatch = (patch: Record<string, unknown>): Record<string, unknown> => ({
  ...BASE,
  ...patch,
});

const without = (key: string): Record<string, unknown> => {
  const { [key]: _removed, ...rest } = BASE;
  return rest;
};

Deno.test("契約名と major はこの実装が固定する（manifest 側の綴りと突き合わせる先）", () => {
  assertEquals(IRODORI_PIPELINE_NAME, "irodori");
  assertEquals(IRODORI_PIPELINE_MAJOR, 1);
});

Deno.test("parseIrodoriPipelineConfig: 実重みの値を読み切る", () => {
  const config = parseIrodoriPipelineConfig(BASE);
  assertEquals(config.maxTextLen, 256);
  assertEquals(config.speakerRows, 751);
  assertEquals(config.cfgScales, { text: 3, speaker: 5, caption: 3 });
  assertEquals(config.speakerUncondMode, "mask");
  assertEquals(config.cfgGuidanceMode, "independent");
});

Deno.test("parseIrodoriPipelineConfig: 未知キーは落とす（綴り違いの黙認を作らない）", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ step: 40 })),
    Error,
    "未知キー 'step'",
  );
});

Deno.test("parseIrodoriPipelineConfig: cfgScales の未知キーも落とす", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ cfgScales: { text: 3, speaker: 5, style: 3 } })),
    Error,
    "未知キー 'style'",
  );
});

Deno.test("parseIrodoriPipelineConfig: 欠落は欄名を添えて落とす", () => {
  for (const key of Object.keys(BASE)) {
    assertThrows(
      () => parseIrodoriPipelineConfig(without(key)),
      Error,
      key === "cfgScales" ? "pipelineConfig.cfgScales" : `pipelineConfig.${key}`,
    );
  }
});

Deno.test("parseIrodoriPipelineConfig: 対応外の speakerUncondMode は拒否する（ADR 0047 決定 1）", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ speakerUncondMode: "noise" })),
    Error,
    "'mask' だけ",
  );
});

Deno.test("parseIrodoriPipelineConfig: 対応外の cfgGuidanceMode は拒否する（ADR 0047 決定 1）", () => {
  for (const mode of ["joint", "alternating"]) {
    assertThrows(
      () => parseIrodoriPipelineConfig(withPatch({ cfgGuidanceMode: mode })),
      Error,
      "'independent' だけ",
    );
  }
});

Deno.test("parseIrodoriPipelineConfig: 幅と長さの値域を見る", () => {
  // token 列は BOS を必ず 1 本置くので、本文の予算が残らない 1 は組めない。
  assertThrows(() => parseIrodoriPipelineConfig(withPatch({ maxTextLen: 1 })), Error, "2 以上");
  // t_embed は前半 cos / 後半 sin に割るので奇数幅は組めない。
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ timestepEmbedDim: 511 })),
    Error,
    "正の偶数でない",
  );
  assertThrows(() => parseIrodoriPipelineConfig(withPatch({ latentDim: 0 })), Error, "正の整数");
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ initScale: 0 })),
    Error,
    "正の有限数でない",
  );
  assertThrows(
    () =>
      parseIrodoriPipelineConfig(withPatch({ cfgScales: { text: -1, speaker: 5, caption: 3 } })),
    Error,
    "非負の有限数でない",
  );
});

Deno.test("parseIrodoriPipelineConfig: CFG の窓が逆順なら落とす（沈黙で CFG が消える）", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ cfgMinT: 1, cfgMaxT: 0.5 })),
    Error,
    "CFG が 1 度も掛からない",
  );
});

Deno.test("parseIrodoriPipelineConfig: 秒の範囲が逆順なら落とす", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ minSeconds: 30, maxSeconds: 0.5 })),
    Error,
    "minSeconds 30",
  );
});

Deno.test("parseIrodoriPipelineConfig: codec の 3 欄は正の整数（0 も小数も落とす）", () => {
  for (const key of ["sampleRate", "hopLength", "codecHaloFrames"]) {
    for (const value of [0, -1, 1.5, "48000"]) {
      assertThrows(
        () => parseIrodoriPipelineConfig(withPatch({ [key]: value })),
        Error,
        `pipelineConfig.${key}`,
      );
    }
  }
});

Deno.test("parseIrodoriPipelineConfig: sampleRate ÷ hopLength が frameRate と違えば落とす", () => {
  // 秒 → フレームと 秒 → サンプル → フレームの 2 系統が独立に動く形を作らない。
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ hopLength: 1000 })),
    Error,
    "frameRate 25 × hopLength 1000",
  );
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ frameRate: 24 })),
    Error,
    "sampleRate 48000",
  );
});

Deno.test("parseIrodoriPipelineConfig: speakerRows 1 は落とす（参照 latent が 0 行になる）", () => {
  // speaker 条件は「平均トークン 1 本 + patch した参照」なので、1 行では参照が 1 行も載らない。
  // 素通しすると参照音声の上限サンプル数が 0 になり、`integratedLoudness` の「波形が空」という
  // **無関係な文言**で生成の途中に落ちる。maxTextLen / maxCaptionLen と同じ理屈・同じ受理集合。
  assertThrows(() => parseIrodoriPipelineConfig(withPatch({ speakerRows: 1 })), Error, "2 以上");
  assertThrows(() => parseIrodoriPipelineConfig(withPatch({ speakerRows: 0 })), Error, "2 以上");
});

Deno.test("parseIrodoriPipelineConfig: maxSeconds から到達しうる S が ditSymMax を超えたら落とす", () => {
  // 素通しすると、長い発話のときだけ「決まった latent 長が dit の宣言上限を超えている」で
  // duration 段まで走ってから落ちる（短い発話では永久に露見しない）。
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ maxSeconds: 30.5 })),
    Error,
    "到達しうる latent 長 763",
  );
  // 手動秒経路（`ceil(trunc(秒×sampleRate) / hopLength)`）だけが 1 フレーム上に出る宣言。
  // duration 経路の `floor(秒×frameRate)` だけを見る式ではここを取りこぼす。
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ maxSeconds: 30.02 })),
    Error,
    "到達しうる latent 長 751",
  );
  // 実配布の 30 秒 / ditSymMax 750 はちょうど成立する（境界を締めすぎていない）。
  assertEquals(parseIrodoriPipelineConfig(withPatch({})).ditSymMax, 750);
});

Deno.test("parseIrodoriPipelineConfig: cfgScales 0 は正規の値（その条件の CFG を回さない）", () => {
  const config = parseIrodoriPipelineConfig(
    withPatch({ cfgScales: { text: 3, speaker: 0, caption: 0 } }),
  );
  assertEquals(config.cfgScales.speaker, 0);
});

Deno.test("parseIrodoriPipelineConfig: f32 で厳密に表せない cfgScale は宣言段で落とす", () => {
  // 1.3 は f64 と f32 で値が違う。ホスト経路（f64 乗算）と常駐経路（f32 乗算）が 1〜2 ulp
  // 割れ、「2 経路の出力は同じ」MUST が条件付きになるので、配布形の宣言側で落とす。
  assertThrows(
    () =>
      parseIrodoriPipelineConfig(withPatch({ cfgScales: { text: 1.3, speaker: 5, caption: 3 } })),
    Error,
    "f32 で厳密に表せない",
  );
  // 実配布の 3 / 5 / 3 と、f32 厳密な非整数（2 の冪の和）は通る。
  const config = parseIrodoriPipelineConfig(
    withPatch({ cfgScales: { text: 1.25, speaker: 5, caption: 3 } }),
  );
  assertEquals(config.cfgScales.text, 1.25);
});
