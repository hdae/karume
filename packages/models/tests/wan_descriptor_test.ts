// Wan の世代の記述子（`src/wan/descriptor.ts`）の内部整合。欄どうしの食い違いは、既定のままの要求が
// `ModelInputError`（入力起因）で落ちる形で出て、記述子の誤りを利用者の入力の誤りに取り違える。
// その前に記述子ごとにここで落とす。世代を足したら GENERATIONS に 1 行足す（同じ検査を当てる）。

import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  WAN21_GENERATION,
  WAN22_TI2V_GENERATION,
  type WanGenerationDescriptor,
} from "../src/wan/descriptor.ts";
import { TEMPORAL_COMPRESSION } from "../src/wan/plan.ts";

/** VAE の時間圧縮に合うフレーム数（4n+1）か。 */
const isFrameCount = (frames: number): boolean =>
  Number.isInteger(frames) && frames >= 1 && (frames - 1) % TEMPORAL_COMPRESSION === 0;

const isPositiveInteger = (value: number): boolean => Number.isInteger(value) && value >= 1;

/** 記述子の欄どうしの食い違い（無ければ空）。 */
const inconsistencies = (generation: WanGenerationDescriptor): string[] => {
  const { acceptedSizes, minFrames, maxFrames, defaults, latents, fps, vaePatchSize } = generation;
  const found: string[] = [];
  if (acceptedSizes.length === 0) {
    found.push("受理集合が空");
  }
  // 範囲の両端が 4n+1 でないと、下端から 4 刻みで並べた選択肢（gpu-lab）が全て門に拒まれる。
  if (!isFrameCount(minFrames) || !isFrameCount(maxFrames)) {
    found.push(`フレーム数の範囲 ${minFrames}〜${maxFrames} の端が 4n+1 でない`);
  }
  if (minFrames > maxFrames) {
    found.push(`フレーム数の範囲 ${minFrames}〜${maxFrames} の下端が上端より大きい`);
  }
  if (!isPositiveInteger(fps)) {
    found.push(`fps ${fps} が正の整数でない`);
  }
  if (!isPositiveInteger(vaePatchSize)) {
    found.push(`vaePatchSize ${vaePatchSize} が正の整数でない`);
  }
  if (
    !acceptedSizes.some((size) => size.width === defaults.width && size.height === defaults.height)
  ) {
    found.push(`既定の寸法 ${defaults.width}×${defaults.height} が受理集合に無い`);
  }
  if (!isFrameCount(defaults.frames)) {
    found.push(`既定のフレーム数 ${defaults.frames} が 4n+1 でない`);
  }
  if (defaults.frames < minFrames || defaults.frames > maxFrames) {
    found.push(`既定のフレーム数 ${defaults.frames} が ${minFrames}〜${maxFrames} の外`);
  }
  if (latents.mean.length !== latents.std.length) {
    found.push(`mean ${latents.mean.length} 本と std ${latents.std.length} 本の数が違う`);
  }
  return found;
};

const GENERATIONS: readonly (readonly [string, WanGenerationDescriptor])[] = [
  ["Wan2.1", WAN21_GENERATION],
  ["Wan2.2", WAN22_TI2V_GENERATION],
];

describe("世代の記述子の内部整合", () => {
  for (const [name, generation] of GENERATIONS) {
    it(`${name}: 受理集合と範囲は空でなく端は 4n+1・既定はその中・fps と patch は正の整数・mean と std の本数が等しい`, () => {
      assertEquals(inconsistencies(generation), []);
    });
  }

  // 空の受理集合と逆転した範囲は、既定が中に入り得ないので既定の食い違いも併せて出る。
  it("対照: 欄を 1 つずつ壊した記述子は、壊した欄の食い違いとして見つかる", () => {
    const base = WAN21_GENERATION;
    const broken: readonly (readonly [WanGenerationDescriptor, readonly string[]])[] = [
      [{ ...base, acceptedSizes: [] }, ["受理集合が空", "既定の寸法 832×480 が受理集合に無い"]],
      [{ ...base, minFrames: 6 }, ["フレーム数の範囲 6〜81 の端が 4n+1 でない"]],
      [
        { ...base, minFrames: 37, maxFrames: 29 },
        [
          "フレーム数の範囲 37〜29 の下端が上端より大きい",
          "既定のフレーム数 33 が 37〜29 の外",
        ],
      ],
      [{ ...base, fps: 0 }, ["fps 0 が正の整数でない"]],
      [{ ...base, vaePatchSize: 1.5 }, ["vaePatchSize 1.5 が正の整数でない"]],
      [
        { ...base, defaults: { ...base.defaults, width: 480 } },
        ["既定の寸法 480×480 が受理集合に無い"],
      ],
      [
        { ...base, defaults: { ...base.defaults, frames: 32 } },
        ["既定のフレーム数 32 が 4n+1 でない"],
      ],
      [
        { ...base, defaults: { ...base.defaults, frames: 85 } },
        ["既定のフレーム数 85 が 5〜81 の外"],
      ],
      [
        { ...base, latents: { mean: base.latents.mean, std: base.latents.std.slice(1) } },
        ["mean 16 本と std 15 本の数が違う"],
      ],
    ];
    for (const [generation, messages] of broken) {
      assertEquals(inconsistencies(generation), messages);
    }
  });
});
