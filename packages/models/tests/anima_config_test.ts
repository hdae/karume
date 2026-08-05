// `pipelineConfig` の手書きスキーマ検証（ADR 0038 §1 — スキーマは models 側が所有）。
//
// hub は `pipelineConfig` を素通しするので、**ここが唯一の門**。主に縛るのは 2 つ:
// ①必須欄の欠落・型違いを名指しで落とす ②未知キーを fail loudly（黙って既定へ縮退させない）。

import { assertEquals, assertThrows } from "@std/assert";
import { parseAnimaPipelineConfig } from "../src/anima/config.ts";

/** `models/anima/karume.json` と同じ形（実配布物の写し）。 */
const VALID = {
  scheduler: { shift: 3, numTrainTimesteps: 1000 },
  defaults: {
    steps: 10,
    guidanceScale: 1,
    resolution: { width: 1024, height: 1024 },
    negativePrompt: "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
  },
} as const;

const withDefaults = (patch: Record<string, unknown>): Record<string, unknown> => ({
  ...VALID,
  defaults: { ...VALID.defaults, ...patch },
});

Deno.test("parseAnimaPipelineConfig: 実配布物の形を読む", () => {
  const config = parseAnimaPipelineConfig(VALID);
  assertEquals(config.scheduler, { shift: 3, numTrainTimesteps: 1000 });
  assertEquals(config.defaults.steps, 10);
  assertEquals(config.defaults.guidanceScale, 1);
  assertEquals(config.defaults.resolution, { width: 1024, height: 1024 });
  assertEquals(config.defaults.negativePrompt, VALID.defaults.negativePrompt);
});

Deno.test("parseAnimaPipelineConfig: negativePrompt は任意（欄ごと無くてよい）", () => {
  const config = parseAnimaPipelineConfig({
    scheduler: VALID.scheduler,
    defaults: {
      steps: 8,
      guidanceScale: 4,
      resolution: { width: 1344, height: 768 },
    },
  });
  assertEquals(config.defaults.negativePrompt, undefined);
  assertEquals(config.defaults.resolution, { width: 1344, height: 768 });
});

Deno.test("parseAnimaPipelineConfig: 未知キーは fail loudly（綴り違いを既定へ縮退させない）", () => {
  assertThrows(
    () => parseAnimaPipelineConfig({ ...VALID, sampler: {} }),
    Error,
    "pipelineConfig: 未知キー 'sampler'",
  );
  assertThrows(
    () => parseAnimaPipelineConfig({ ...VALID, scheduler: { ...VALID.scheduler, sigmaMax: 1 } }),
    Error,
    "pipelineConfig.scheduler: 未知キー 'sigmaMax'",
  );
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ step: 8 })),
    Error,
    "pipelineConfig.defaults: 未知キー 'step'",
  );
  assertThrows(
    () =>
      parseAnimaPipelineConfig(withDefaults({ resolution: { width: 1024, height: 1024, w: 1 } })),
    Error,
    "未知キー 'w'",
  );
});

Deno.test("parseAnimaPipelineConfig: 必須欄の欠落を名指しで落とす", () => {
  assertThrows(() => parseAnimaPipelineConfig({}), Error, "pipelineConfig.scheduler: 無い");
  assertThrows(
    () => parseAnimaPipelineConfig({ scheduler: VALID.scheduler }),
    Error,
    "pipelineConfig.defaults: 無い",
  );
  assertThrows(
    () => parseAnimaPipelineConfig({ scheduler: { shift: 3 }, defaults: VALID.defaults }),
    Error,
    "pipelineConfig.scheduler.numTrainTimesteps: 無い",
  );
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ resolution: { width: 1024 } })),
    Error,
    "height: 無い",
  );
});

Deno.test("parseAnimaPipelineConfig: 型と値域を検査する", () => {
  assertThrows(
    () => parseAnimaPipelineConfig({ ...VALID, scheduler: { shift: 0, numTrainTimesteps: 1000 } }),
    Error,
    "正の有限数でない",
  );
  assertThrows(
    () =>
      parseAnimaPipelineConfig({
        ...VALID,
        scheduler: { shift: 3, numTrainTimesteps: 1000.5 },
      }),
    Error,
    "正の整数でない",
  );
  // steps < 2 は sigma の linspace が組めない（sampler の受理集合と同じ下限）。
  assertThrows(() => parseAnimaPipelineConfig(withDefaults({ steps: 1 })), Error, "2 以上の整数");
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ guidanceScale: "1" })),
    Error,
    "有限の数でない",
  );
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ negativePrompt: 42 })),
    Error,
    "文字列でない",
  );
});

Deno.test("parseAnimaPipelineConfig: 既定の解像度も受理集合の内側であることを要求する", () => {
  // 配布時に外れていたら構築で落とす — 「生成を 1 回走らせて初めて分かる」を作らない。
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ resolution: { width: 1000, height: 1024 } })),
    Error,
    "倍数でない",
  );
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ resolution: { width: 256, height: 256 } })),
    Error,
    "下限",
  );
});
