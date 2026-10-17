// Asset-free invariants of the host-side DeBERTa relative-position table builder
// (buildRelPosTables). The byte parity against the Python-generated golden tables and the
// "no table baked into the container" check read the series assets, so they live in
// packages/e2e/tests/e2e_sbv2_rel_pos_parity_test.ts.

import { test } from "@karume/test-support/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { buildRelPosTables } from "../src/sbv2/text/rel-pos-tables.ts";

// ---- 資産の要らない門（上のパリティは golden が無いと 0 本走る）----------------
//
// 対の `sbv2_relattn_parity_test.ts` には資産不要のテストが 2 本あるのに、こちら側は
// golden が無いと `PARITY_CASES` が空になり **1 本も走らない**（引数門も数学的性質も
// 未検証のまま緑）。Python 側とのバイト一致は上のパリティにしか置けないが、式が満たす
// べき構造（引数の受理集合・転置・Toeplitz）は資産なしで縛れる。

test("buildRelPosTables: 引数の受理集合（式が成立する下限を割る値は落とす）", () => {
  // 長さ 0 は表が空になるだけでなく、対角配列 `2·length-1` が負長になる。
  assertThrows(() => buildRelPosTables(0, 256, 512), RangeError, "1 以上の整数でない");
  // bucketSize 1 は mid = 0 で対数の分母（log((maxPosition-1)/mid)）が壊れる。
  assertThrows(() => buildRelPosTables(4, 1, 512), RangeError, "bucketSize 1 が 2 以上");
  // maxPosition 1 は log(0/…) で -Infinity を作る。
  assertThrows(() => buildRelPosTables(4, 256, 1), RangeError, "maxPosition 1 が 2 以上");
});

test("buildRelPosTables: 対数バケットの式が成立しない組は NaN 経由の誤った添字を作らずに落ちる", () => {
  // (bucket 2, maxPosition 2): mid = 1 で対数の分母 log(1) = 0、係数 mid − 1 = 0 →
  // `Infinity × 0 = NaN`。NaN は clamp の比較をすり抜け、Int32Array で 0 になって形の合った
  // 誤った gather 添字になる（|d| = 2 の位置が 0 に化ける）。
  assertThrows(() => buildRelPosTables(4, 2, 2), RangeError, "対数バケット");
  // mid ≥ 2 でも maxPosition − 1 ≤ mid なら分母が 0 以下で、遠方の向きが壊れる。
  assertThrows(() => buildRelPosTables(4, 4, 3), RangeError, "対数バケット");
  assertThrows(() => buildRelPosTables(4, 3, 512), RangeError, "対数バケット");
});

test("buildRelPosTables: 配布値（256 / 512）の近傍は線形のまま d + 256", () => {
  // |d| < mid = 128 は対数で圧縮しない（`make_log_bucket_position` の近傍分岐）。入口の
  // 検査を足しても配布値の表が 1 要素も変わらないことの対。
  const length = 7;
  const { c2pPos } = buildRelPosTables(length, 256, 512);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      assertEquals(c2pPos.data[i * length + j], i - j + 256, `c2p[${i}][${j}]`);
    }
  }
});

test("buildRelPosTables: p2c は c2p の転置（バケット化が奇関数であることの帰結）", () => {
  const length = 7;
  const { c2pPos, p2cPos } = buildRelPosTables(length, 256, 512);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      assertEquals(
        p2cPos.data[i * length + j],
        c2pPos.data[j * length + i],
        `p2c[${i}][${j}] が c2p[${j}][${i}] と違う`,
      );
    }
  }
});

test("buildRelPosTables: 表は i−j にしか依存しない（Toeplitz）", () => {
  const length = 7;
  const { c2pPos } = buildRelPosTables(length, 256, 512);
  for (let i = 0; i + 1 < length; i += 1) {
    for (let j = 0; j + 1 < length; j += 1) {
      assertEquals(
        c2pPos.data[i * length + j],
        c2pPos.data[(i + 1) * length + (j + 1)],
        `対角 ${i - j} の値が位置で変わる`,
      );
    }
  }
  // 上の 2 本が恒真でないことの対: `maxPosition` は遠方（対数域）の値を実際に動かす。
  // 近傍だけの表（bucketSize 256 に対し length 7）では対数域へ入らないので、
  // 対数域へ届く形（mid = 4 に対し |i−j| が 4 を超える）で見る。
  const wide = buildRelPosTables(16, 8, 512);
  const narrow = buildRelPosTables(16, 8, 64);
  assert(
    [...wide.c2pPos.data].some((value, index) => value !== narrow.c2pPos.data[index]),
    "maxPosition を変えても表が同じ（バケット規則を読んでいない）",
  );
  // 近傍（線形域）は maxPosition に依らない — 差が出るのは遠方だけであることの固定。
  assertEquals(wide.c2pPos.data[0 * 16 + 1], narrow.c2pPos.data[0 * 16 + 1]);
});
