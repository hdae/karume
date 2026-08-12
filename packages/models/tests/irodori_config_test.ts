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

Deno.test("parseIrodoriPipelineConfig: cfgScales 0 は正規の値（その条件の CFG を回さない）", () => {
  const config = parseIrodoriPipelineConfig(
    withPatch({ cfgScales: { text: 3, speaker: 0, caption: 0 } }),
  );
  assertEquals(config.cfgScales.speaker, 0);
});
