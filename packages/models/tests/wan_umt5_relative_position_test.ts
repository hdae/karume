// umT5 の相対位置のバケット表について、ホストの TS 生成器（src/wan/umt5/relative-position.ts）と
// Python 側の正本（recipe `wan/umt5_patch.py` — 上流 `_relative_position_bucket` を呼ぶ）の
// **バイト一致**を固定する（ADR 0119 決定 3 — sbv2_rel_pos_parity_test.ts と同じ責務）。
//
// 表はグラフ入力で、golden の表も Python が作る。式が割れると shape は合ったまま別の距離のバイアスを
// 足し、ホストとゴールデンが同じ誤りを共有すると E2E もすり抜けるので、E2E とは別にここで直接
// 突き合わせる。上流は f32 の `log`・ホストは f64 の `Math.log` — 境界で切り捨てが割れないことを
// 実データで縛るのはここだけ。
//
// fixture は `fixtures/wan-umt5/`（生成: tools/export-recipes/wan/umt5_host_fixture.py）:
//  ・`bucket_by_distance` — 距離 −511〜511 の全部（要素で名指しする用）
//  ・代表の有効長の表（バケットの境界を跨ぐ長さと固定プロンプトの長さ）
//  ・有効長 2〜512 の全表の digest（表ごとの SHA-256 を昇順に連ねたものの SHA-256）
// GPU も実資産も要らない。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import {
  buildUmt5RelativePositionBuckets,
  WAN_UMT5_MAX_LENGTH,
  WAN_UMT5_RELATIVE_POSITION,
} from "../src/wan/umt5/relative-position.ts";
import { WAN_PROMPT_MIN_TOKENS } from "../src/wan/text/tokenizer.ts";

const FIXTURE_DIR = new URL("./fixtures/wan-umt5/", import.meta.url);

type RelativePositionMeta = {
  readonly num_buckets: number;
  readonly max_distance: number;
  readonly bidirectional: boolean;
  readonly min_length: number;
  readonly max_length: number;
  readonly distance_offset: number;
  readonly representative_lengths: readonly number[];
  readonly digest: string;
};

const meta = JSON.parse(
  await Deno.readTextFile(new URL("relative-position.json", FIXTURE_DIR)),
) as RelativePositionMeta;
const fixtureBytes = await Deno.readFile(new URL("relative-position.safetensors", FIXTURE_DIR));
const tensors = parseSafetensors(
  fixtureBytes.buffer.slice(
    fixtureBytes.byteOffset,
    fixtureBytes.byteOffset + fixtureBytes.byteLength,
  ) as ArrayBuffer,
);

/** fixture の I32 テンソル（無ければ・I32 でなければ落とす）。 */
const fixtureInt32 = (key: string): { shape: readonly number[]; data: Int32Array } => {
  const view = tensors.tensors.get(key);
  assert(view !== undefined, `fixture に '${key}' が無い`);
  assertEquals(view.dtype, "I32", `'${key}' の dtype`);
  return {
    shape: view.shape,
    data: new Int32Array(tensors.buffer, view.byteOffset, view.byteLength / 4),
  };
};

/** 要素ごとの厳密一致（最初に割れた位置を名指しする）。 */
const assertSameElements = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: (index: number) => string,
): void => {
  assertEquals(actual.length, expected.length, "要素数");
  for (let index = 0; index < expected.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(
        `${where(index)} が食い違う（TS=${actual[index]} / Python=${expected[index]}）`,
      );
    }
  }
};

const hex = (bytes: ArrayBuffer): string =>
  [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");

Deno.test("umT5 のバケットの構成と有効長の範囲が fixture（pin した config）と一致", () => {
  assertEquals(
    {
      numBuckets: WAN_UMT5_RELATIVE_POSITION.numBuckets,
      maxDistance: WAN_UMT5_RELATIVE_POSITION.maxDistance,
      bidirectional: true,
      minLength: WAN_PROMPT_MIN_TOKENS,
      maxLength: WAN_UMT5_MAX_LENGTH,
    },
    {
      numBuckets: meta.num_buckets,
      maxDistance: meta.max_distance,
      bidirectional: meta.bidirectional,
      minLength: meta.min_length,
      maxLength: meta.max_length,
    },
  );
});

Deno.test("umT5 の距離 −511〜511 のバケットが上流の式と一致", () => {
  // 上限の表の 0 行目は距離 0〜511、0 列目は距離 0〜−511（table[i][j] = bucket(j − i)）。
  const table = buildUmt5RelativePositionBuckets(WAN_UMT5_MAX_LENGTH).data;
  const expected = fixtureInt32("bucket_by_distance");
  assertEquals(expected.shape, [2 * WAN_UMT5_MAX_LENGTH - 1]);
  const actual = new Int32Array(expected.data.length);
  for (let distance = meta.distance_offset; distance < WAN_UMT5_MAX_LENGTH; distance += 1) {
    actual[distance - meta.distance_offset] = distance >= 0
      ? table[distance]
      : table[-distance * WAN_UMT5_MAX_LENGTH];
  }
  assertSameElements(
    actual,
    expected.data,
    (index) => `距離 ${index + meta.distance_offset} のバケット`,
  );
});

for (const length of meta.representative_lengths) {
  Deno.test(`umT5 のバケット表 L=${length} が Python の表とバイト一致`, () => {
    const built = buildUmt5RelativePositionBuckets(length);
    const expected = fixtureInt32(`table.l${String(length).padStart(3, "0")}`);
    assertEquals(built.dtype, "i32");
    assertEquals(built.shape, [length, length]);
    assertEquals(expected.shape, [length, length]);
    assertSameElements(
      built.data,
      expected.data,
      (index) => `L=${length} の [${Math.floor(index / length)}][${index % length}]`,
    );
  });
}

Deno.test("umT5 のバケット表: 有効長 2〜512 の全表の digest が Python と一致", async () => {
  // digest は表の i32 リトルエンディアンのバイト列で定義される（fixture の doc）。Int32Array の
  // 中身はエンジンのバイト順なので、ビッグエンディアンの機では比べる前に止める。
  assertEquals(new Uint8Array(new Uint32Array([1]).buffer)[0], 1, "リトルエンディアンの機ではない");
  assertEquals([meta.min_length, meta.max_length], [WAN_PROMPT_MIN_TOKENS, WAN_UMT5_MAX_LENGTH]);
  const chained = new Uint8Array(32 * (WAN_UMT5_MAX_LENGTH - WAN_PROMPT_MIN_TOKENS + 1));
  for (let length = WAN_PROMPT_MIN_TOKENS; length <= WAN_UMT5_MAX_LENGTH; length += 1) {
    const { data } = buildUmt5RelativePositionBuckets(length);
    const digest = await crypto.subtle.digest("SHA-256", data);
    chained.set(new Uint8Array(digest), 32 * (length - WAN_PROMPT_MIN_TOKENS));
  }
  assertEquals(hex(await crypto.subtle.digest("SHA-256", chained)), meta.digest);
});

Deno.test("buildUmt5RelativePositionBuckets: 有効長の受理集合（グラフの記号次元と同じ 2〜512）", () => {
  for (const length of [0, 1, WAN_UMT5_MAX_LENGTH + 1, 2.5, Number.NaN]) {
    assertThrows(
      () => buildUmt5RelativePositionBuckets(length),
      RangeError,
      "の整数でない",
      `有効長 ${length}`,
    );
  }
});
