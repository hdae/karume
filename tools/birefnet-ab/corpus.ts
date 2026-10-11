/**
 * A/B の素材（ADR 0125 — 裁定 31: 両系列とも anima で生成する）。**プロンプト・seed・寸法の正本は
 * ここ**で、生成のコマンド（{@link animaCommand}）も道具の読む名前（{@link animaFile}）もここから作る。
 *
 * - 拡大 4 枚: 2048×2048 で生成（anima の各辺の上限。1920 を超える寸法は anima が学習時に宣言した
 *   位置の外側で、品質が落ちうる — 輪郭の A/B には足りる前提）。1024² のグラフの α を 2048² へ戻す。
 *   2048² モデルの σ を擬似正解にできる（総合列 — `analysis.ts`）。
 * - 縮小 4 枚: 832×512 で生成し、上下 16 画素を切り落として 832×480 にする（anima の各辺の下限は 512。
 *   切り落としは補間を通さない）。1024² のグラフの α を 832×480 へ戻す（横は 1024 → 832・縦は
 *   1024 → 480 の縮小）。
 *
 * 被写体は髪・毛皮・細い物（輪郭の補間の差が出やすい）で、無難な題材に限る。生成物は `outputs/` だけに
 * 置き、リポの記録には画像を載せない。seed は画像ごとに変える（`examples/anima/main.ts` の出力名は
 * プロンプトを含まず、同じ寸法・同じ seed だと別のプロンプトの画像を上書きする）。
 */

import type { Window } from "./analysis.ts";

/** 縮尺の向き（グラフの 1024² → 目標の寸法）。 */
export type Direction = "up" | "down";

/** 素材 1 枚。 */
export type CorpusEntry = {
  /** 道具の中の名前（出力のファイル名と回答用紙に出る — 手順を含まない）。 */
  readonly name: string;
  readonly direction: Direction;
  readonly prompt: string;
  readonly seed: number;
  /** anima で生成する寸法。 */
  readonly width: number;
  readonly height: number;
  /** 生成物から切り出す矩形（補間を通さない）。省くと全体。 */
  readonly crop?: Window;
};

/** 品質タグ（`examples/anima/main.ts` の既定のプロンプトと同じ流儀）。 */
const QUALITY = "masterpiece, best quality";

/** 縮小の素材の切り出し（832×512 の上下 16 画素を落とす）。 */
const DOWN_CROP: Window = { left: 0, top: 16, width: 832, height: 480 };

export const CORPUS: readonly CorpusEntry[] = [
  {
    name: "up-hair",
    direction: "up",
    prompt: `1girl, solo, very long hair, flowing hair, wind, from behind, blue sky, ${QUALITY}`,
    seed: 2101,
    width: 2048,
    height: 2048,
  },
  {
    name: "up-cat-fur",
    direction: "up",
    prompt: `no humans, cat, white cat, long fur, fluffy, sitting, grey background, ${QUALITY}`,
    seed: 2102,
    width: 2048,
    height: 2048,
  },
  {
    name: "up-dandelion",
    direction: "up",
    prompt:
      `no humans, dandelion, seed head, close-up, simple background, dark background, ${QUALITY}`,
    seed: 2103,
    width: 2048,
    height: 2048,
  },
  {
    name: "up-bicycle",
    direction: "up",
    prompt: `no humans, bicycle, spokes, parked, street, outdoors, ${QUALITY}`,
    seed: 2104,
    width: 2048,
    height: 2048,
  },
  {
    name: "down-dog-fur",
    direction: "down",
    prompt: `no humans, dog, golden retriever, fluffy fur, grass, outdoors, ${QUALITY}`,
    seed: 2201,
    width: 832,
    height: 512,
    crop: DOWN_CROP,
  },
  {
    name: "down-hair",
    direction: "down",
    prompt: `1girl, solo, long hair, wind, upper body, outdoors, smile, ${QUALITY}`,
    seed: 2202,
    width: 832,
    height: 512,
    crop: DOWN_CROP,
  },
  {
    name: "down-fern",
    direction: "down",
    prompt: `no humans, potted plant, fern, thin leaves, indoors, window, ${QUALITY}`,
    seed: 2203,
    width: 832,
    height: 512,
    crop: DOWN_CROP,
  },
  {
    name: "down-bird",
    direction: "down",
    prompt: `no humans, bird, feathers, perched, branch, outdoors, ${QUALITY}`,
    seed: 2204,
    width: 832,
    height: 512,
    crop: DOWN_CROP,
  },
];

/** `examples/anima/main.ts` が既定の量子化・既定の段数で書くファイル名。 */
export const animaFile = (entry: CorpusEntry): string =>
  `anima-default-${entry.width}x${entry.height}-defaultstep-seed${entry.seed}.png`;

/** 素材 1 枚を生成するコマンド（リポの根で実行する）。 */
export const animaCommand = (entry: CorpusEntry): string =>
  `deno task demo:anima --prompt '${entry.prompt}' --resolution ${entry.width}x${entry.height}` +
  ` --seed ${entry.seed}`;
