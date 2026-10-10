/**
 * Anima の受理集合 × 融合 attention の経路（runtime の `attentionRoute` — 実行と同じ判定）。GPU も資産も
 * 要らない。
 *
 * 既定席（`f16+dit8-a8-attn8-s16` — tools/export-recipes/anima/distribution.py）は a8 + s16 を宣言するが、
 * self-attention のキー長 N = S = (W/16)·(H/16) が 4 の倍数でない解像度では ③PV と S の格納が f32 へ
 * 落ちる（ADR 0058 決定 3 が許す縮退 — docs/limitations.md）。ここはその部分集合を受理集合の全形で固定し、
 * 系列の A/B 門（e2e_anima_ab_test.ts の完全適格の検査）が掛かる形が適格であることの裏付けにする。
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  type AttentionRoute,
  attentionRoute,
  type AttentionSeats,
} from "../../runtime/src/runtime/attention-route.ts";
import {
  ANIMA_SPATIAL_COMPRESSION,
  ditPatchGeometry,
  tokenCount,
} from "../src/anima/dit-tokens.ts";
import {
  assertAcceptableResolution,
  type ImageSize,
  MAX_RESOLUTION_SIDE,
  MIN_RESOLUTION_SIDE,
  RESOLUTION_GRANULARITY,
} from "../src/anima/resolution.ts";

/** 既定席の attention の宣言（a8 + s16）。 */
const DEFAULT_SEATS: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "f16" };
/** 宣言どおりの経路。 */
const REQUESTED: AttentionRoute = { qk: "i8a8", pv: "i8a8", score: "f16" };
/** N % 4 != 0 の self で実際に走る経路（①QK は D で決まるので i8a8 のまま）。 */
const DEGRADED: AttentionRoute = { qk: "i8a8", pv: "dense", score: "f32" };

/**
 * 配布形の transformer のグラフ宣言の写し: `tokens [1,S,68]` / 出力幅 64（patch 幾何は
 * `ditPatchGeometry` で割り出す — pipeline.ts の `planDynDit` と同じ経路）・`rope_cos [1,1,S,128]`
 * の head_dim 128 = D・`encoder_hidden_states [1,512,1024]` の 512 = cross のキー長。
 */
const GEOMETRY = ditPatchGeometry(68, 64);
const HEAD_DIM = 128;
const CROSS_KEYS = 512;

/** self-attention のキー長 S（pipeline.ts の `planDynDit` と同じく、解像度 → latent → `tokenCount`）。 */
const selfKeys = ({ width, height }: ImageSize): number =>
  tokenCount(
    [1, GEOMETRY.channels, height / ANIMA_SPATIAL_COMPRESSION, width / ANIMA_SPATIAL_COMPRESSION],
    GEOMETRY,
  );

/** 受理集合の全形（各辺は刻みの上の下限〜天井 — 受理の門を通るものだけ）。 */
const ACCEPTED = ((): readonly ImageSize[] => {
  const sides: number[] = [];
  for (
    let side = MIN_RESOLUTION_SIDE;
    side <= MAX_RESOLUTION_SIDE;
    side += RESOLUTION_GRANULARITY
  ) {
    sides.push(side);
  }
  const sizes = sides.flatMap((width) => sides.map((height) => ({ width, height })));
  for (const size of sizes) assertAcceptableResolution(size);
  return sizes;
})();

const label = ({ width, height }: ImageSize): string => `${width}×${height}`;

describe("Anima の受理集合 × 既定席の attention の経路", () => {
  it("受理集合は 1 辺 97 通りの 9,409 形", () => {
    assertEquals(ACCEPTED.length, 97 * 97);
  });

  it("self は S % 4 != 0 のときだけ縮退し（③PV と S が f32）、それ以外は宣言どおり", () => {
    for (const size of ACCEPTED) {
      const keys = selfKeys(size);
      assertEquals(
        attentionRoute(DEFAULT_SEATS, HEAD_DIM, keys),
        keys % 4 === 0 ? REQUESTED : DEGRADED,
        `${label(size)}（S=${keys}）`,
      );
    }
  });

  it("縮退する形は 9,409 形のうち 4,608 形", () => {
    const degraded = ACCEPTED.filter((size) =>
      attentionRoute(DEFAULT_SEATS, HEAD_DIM, selfKeys(size)).pv === "dense"
    );
    assertEquals(degraded.length, 4608);
  });

  it("両辺が 32 px の倍数なら縮退しない", () => {
    const aligned = ACCEPTED.filter(({ width, height }) => width % 32 === 0 && height % 32 === 0);
    assert(aligned.length > 0);
    for (const size of aligned) {
      assertEquals(attentionRoute(DEFAULT_SEATS, HEAD_DIM, selfKeys(size)), REQUESTED, label(size));
    }
  });

  it("具体例: 528×528（S=1,089）は縮退し、1024×1024 / 1344×768 / 896×1152 は宣言どおり", () => {
    assertEquals(selfKeys({ width: 528, height: 528 }), 1089);
    assertEquals(attentionRoute(DEFAULT_SEATS, HEAD_DIM, 1089), DEGRADED);
    for (
      const size of [
        { width: 1024, height: 1024 },
        { width: 1344, height: 768 },
        { width: 896, height: 1152 },
      ]
    ) {
      assertEquals(attentionRoute(DEFAULT_SEATS, HEAD_DIM, selfKeys(size)), REQUESTED, label(size));
    }
  });

  it("cross（キー長 512）は解像度に依らず宣言どおり", () => {
    assertEquals(attentionRoute(DEFAULT_SEATS, HEAD_DIM, CROSS_KEYS), REQUESTED);
  });
});
