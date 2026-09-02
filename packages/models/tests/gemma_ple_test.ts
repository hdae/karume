// PLE sidecar のホスト gather（`src/gemma/ple.ts` — ADR 0085）の寿命と中断の門。GPU も実資産も
// 要らない（合成 sidecar を `writeSafetensors` で組む）。
//
// ここで縛るのは 3 つ:
//
// - **解放**: `dispose()` で常駐（`maxResidentBytes` ぶんのホスト RAM）が空になり、以後の
//   gather は fail loudly。口が無いと「パイプラインを dispose してもホスト RAM が返らない」形が
//   復活するが、解放は例外にならないので stats の実数で見るしかない。
// - **中断の透過**: gather の `signal` が shard の読み口まで降りる（best-effort）。降りないと
//   「停止を押しても shard 1 本の読みが終わるまで返らない」。中断された読みは常駐に残らず、
//   同じ id をもう一度引けば読み直す（拒否済み Promise を掴み続けない）。
// - **バイト予算**: 常駐上限は**本数ではなくバイト**（shard 幅は資産世代で変わる）。索引だけから
//   決まる計算・既定（最大 shard 2 本ぶん）・予算内の LRU 追い出し順・0（常駐なし）・
//   1 本すら載らない予算の拒否を凍結する。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import {
  createGemma4Ple,
  defaultGemma4PleResidentBytes,
  type Gemma4PleIndex,
  gemma4PleShardBytes,
} from "../src/gemma/ple.ts";
import { type DumpTensor, writeSafetensors } from "./helpers/safetensors-write.ts";

const LAYERS = 2;
const DIM = 2;
const ROWS_PER_SHARD = 2;
const TOKENS = 6;
/** 2 冪（f32 の乗算が厳密 — ADR 0085 決定 4 と同じ性質を合成側でも保つ）。 */
const EMBED_SCALE = 4;
const SHARD_FILES = [
  "ple-00001-of-00003.safetensors",
  "ple-00002-of-00003.safetensors",
  "ple-00003-of-00003.safetensors",
] as const;

const INDEX: Gemma4PleIndex = {
  tokens: TOKENS,
  layers: LAYERS,
  dim: DIM,
  embedScale: EMBED_SCALE,
  shards: SHARD_FILES.map((file, position) => ({
    file,
    start: position * ROWS_PER_SHARD,
    stop: (position + 1) * ROWS_PER_SHARD,
  })),
};

/** shard 1 本ぶんのバイト数（この索引は全 shard 同幅）。 */
const SHARD_BUDGET = gemma4PleShardBytes(INDEX, INDEX.shards[0]);

/** i8 の値は `id * 10 + 層 * 2 + 列`、per-row scale は `1 / 2^(層+1)`（2 冪で厳密）。 */
const quantized = (id: number, layer: number, column: number): number =>
  id * 10 + layer * 2 + column;
const scaleOf = (layer: number): number => 1 / 2 ** (layer + 1);
const expectedValue = (id: number, layer: number, column: number): number =>
  Math.fround(quantized(id, layer, column) * scaleOf(layer)) * EMBED_SCALE;

const shardBytes = (position: number): Uint8Array<ArrayBuffer> => {
  const shard = INDEX.shards[position];
  const rows = shard.stop - shard.start;
  const values = new Int8Array(rows * LAYERS * DIM);
  const scales = new Float32Array(rows * LAYERS);
  for (let row = 0; row < rows; row += 1) {
    for (let layer = 0; layer < LAYERS; layer += 1) {
      scales[row * LAYERS + layer] = scaleOf(layer);
      for (let column = 0; column < DIM; column += 1) {
        values[(row * LAYERS + layer) * DIM + column] = quantized(
          shard.start + row,
          layer,
          column,
        );
      }
    }
  }
  const tensors = new Map<string, DumpTensor>([
    ["values", { dtype: "I8", shape: [rows, LAYERS, DIM], data: values }],
    ["scales", { dtype: "F32", shape: [rows, LAYERS], data: scales }],
  ]);
  return writeSafetensors(tensors, {
    karume_ple: JSON.stringify({
      schema: 1,
      tokens: TOKENS,
      layers: LAYERS,
      dim: DIM,
      embedScale: EMBED_SCALE,
      start: shard.start,
      stop: shard.stop,
    }),
  });
};

const SHARD_BYTES = INDEX.shards.map((_shard, position) => shardBytes(position));

type ReadCall = { readonly file: string; readonly signal: AbortSignal | undefined };

/** 読み口の fake（呼び出しを記録し、`signal` を honor するかを切り替えられる）。 */
const fakeReader = (options: { readonly honorSignal?: boolean } = {}) => {
  const calls: ReadCall[] = [];
  const readShard = (file: string, readOptions?: { readonly signal?: AbortSignal }) => {
    const signal = readOptions?.signal;
    calls.push({ file, signal });
    const position = SHARD_FILES.indexOf(file as (typeof SHARD_FILES)[number]);
    if (position < 0) return Promise.reject(new Error(`fake: 知らない shard '${file}'`));
    if (options.honorSignal === true && signal !== undefined) {
      return new Promise<ArrayBuffer>((resolve, reject) => {
        // 「読みに時間がかかる」形（中断はこの待ちの間に届く）。
        const timer = setTimeout(() => resolve(SHARD_BYTES[position].buffer), 0);
        signal.addEventListener("abort", () => {
          clearTimeout(timer);
          reject(signal.reason);
        }, { once: true });
      });
    }
    return Promise.resolve(SHARD_BYTES[position].buffer);
  };
  return { calls, readShard };
};

Deno.test("Gemma4Ple: dispose で常駐が空になり、以後の gather は fail loudly", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({
    index: INDEX,
    readShard: reader.readShard,
    vocabSize: TOKENS,
    maxResidentBytes: 2 * SHARD_BUDGET,
  });

  // 2 本にまたがる id を引く（常駐が実際に埋まっている状態を作る）。
  const tensor = await ple.gather([0, 3]);
  assertEquals(tensor.shape, [1, 2, LAYERS, DIM]);
  assert("data" in tensor && tensor.data instanceof Float32Array);
  assertEquals(
    [...tensor.data],
    [
      expectedValue(0, 0, 0),
      expectedValue(0, 0, 1),
      expectedValue(0, 1, 0),
      expectedValue(0, 1, 1),
      expectedValue(3, 0, 0),
      expectedValue(3, 0, 1),
      expectedValue(3, 1, 0),
      expectedValue(3, 1, 1),
    ],
  );
  assertEquals(ple.stats(), { loads: 2, resident: 2, residentBytes: 2 * SHARD_BUDGET });

  ple.dispose();
  // 解放は例外にならないので、実数で見るしかない（ここが 2 のままなら shard は返っていない）。
  assertEquals(ple.stats().resident, 0);
  await assertRejects(() => ple.gather([0]), Error, "dispose 済み");
  // 拒否は「引けない」だけで、読み直しも起きない。
  assertEquals(reader.calls.length, 2);

  // 冪等（2 度目の dispose も、その後の stats も落ちない）。
  ple.dispose();
  assertEquals(ple.stats(), { loads: 2, resident: 0, residentBytes: 0 });
});

Deno.test("Gemma4Ple: gather の signal は shard の読み口へ降りる（best-effort）", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({ index: INDEX, readShard: reader.readShard, vocabSize: TOKENS });
  const controller = new AbortController();

  await ple.gather([0], { signal: controller.signal });
  assertEquals(reader.calls.length, 1);
  assert(
    reader.calls[0].signal === controller.signal,
    "gather の signal が読み口へ降りていない（shard 1 本の読みが中断の届かない区間になる）",
  );

  // 省略した gather は何も渡さない（購読していない呼び出しに signal を捏造しない）。
  await ple.gather([2]);
  assertEquals(reader.calls[1].signal, undefined);
});

Deno.test("Gemma4Ple: 中断された読みは常駐に残らず、引き直しで読み直す", async () => {
  const reader = fakeReader({ honorSignal: true });
  const ple = createGemma4Ple({ index: INDEX, readShard: reader.readShard, vocabSize: TOKENS });
  const controller = new AbortController();
  const reason = new Error("呼び手が止めた");

  const pending = ple.gather([0], { signal: controller.signal });
  controller.abort(reason);
  let caught: unknown;
  try {
    await pending;
  } catch (error) {
    caught = error;
  }
  // 読み口が honor した中断はそのまま上がる（包まない）。
  assert(caught === reason, `中断の例外が包まれている: ${String(caught)}`);
  assertEquals(
    ple.stats(),
    { loads: 1, resident: 0, residentBytes: 0 },
    "拒否された取得を常駐させている",
  );

  // 同じ id をもう一度引けば読み直す（拒否済み Promise を掴み続けない）。
  const tensor = await ple.gather([0]);
  assert("data" in tensor);
  assertEquals(tensor.data[0], expectedValue(0, 0, 0));
  assertEquals(ple.stats(), { loads: 2, resident: 1, residentBytes: SHARD_BUDGET });
});

/**
 * 幅の違う shard を持つ索引（バイト計算だけを見る — 読み口は呼ばれない）。
 *
 * 本数で数える限り「2 本ぶん」は幅に依存して別の RAM を指す。ここが**索引だけで決まる**ことが、
 * 予算をバイトで受ける根拠そのものである。
 */
const UNEVEN_INDEX: Gemma4PleIndex = {
  tokens: 10,
  layers: 3,
  dim: 8,
  embedScale: EMBED_SCALE,
  shards: [
    { file: "wide.safetensors", start: 0, stop: 7 },
    { file: "narrow.safetensors", start: 7, stop: 10 },
  ],
};

/** 読み口が呼ばれたら落とす（構築時の検査だけを見るテスト用）。 */
const unusedReader = (file: string): Promise<ArrayBuffer> =>
  Promise.reject(new Error(`fake: 読んではいけない '${file}'`));

Deno.test("Gemma4Ple: shard の常駐バイトは索引だけから決まる（i8 values + f32 scales）", () => {
  // 7 行 × 3 層 ×（8 列 i8 + 4B scale）= 252 / 3 行ぶん = 108。
  assertEquals(gemma4PleShardBytes(UNEVEN_INDEX, UNEVEN_INDEX.shards[0]), 252);
  assertEquals(gemma4PleShardBytes(UNEVEN_INDEX, UNEVEN_INDEX.shards[1]), 108);
  assertEquals(gemma4PleShardBytes(INDEX, INDEX.shards[0]), ROWS_PER_SHARD * LAYERS * (DIM + 4));
});

Deno.test("Gemma4Ple: 既定の予算は最大 shard 2 本ぶん（どの 2 本でも収まる）", () => {
  // 合計（252 + 108 = 360）でも小さい方の 2 本ぶんでもなく、**最大**の 2 本ぶん。
  assertEquals(defaultGemma4PleResidentBytes(UNEVEN_INDEX), 504);
  assertEquals(defaultGemma4PleResidentBytes(INDEX), 2 * SHARD_BUDGET);
});

Deno.test("Gemma4Ple: 既定は shard 2 本常駐と等価（3 本目で最古が落ちる）", async () => {
  const reader = fakeReader();
  // `maxResidentBytes` を渡さない = 既定（最大 shard 2 本ぶん）。
  const ple = createGemma4Ple({ index: INDEX, readShard: reader.readShard, vocabSize: TOKENS });

  await ple.gather([0]);
  await ple.gather([2]);
  assertEquals(ple.stats(), { loads: 2, resident: 2, residentBytes: 2 * SHARD_BUDGET });

  await ple.gather([4]);
  assertEquals(
    ple.stats(),
    { loads: 3, resident: 2, residentBytes: 2 * SHARD_BUDGET },
    "3 本目を載せても 2 本ぶんに収まっていない（既定が本数の 2 と等価でない）",
  );
  // 落ちたのは最古の shard 0（引き直せば読み直しになる）。
  await ple.gather([0]);
  assertEquals(ple.stats().loads, 4);
});

Deno.test("Gemma4Ple: 追い出しは LRU（参照した shard は予算内に残る）", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({
    index: INDEX,
    readShard: reader.readShard,
    vocabSize: TOKENS,
    maxResidentBytes: 2 * SHARD_BUDGET,
  });

  await ple.gather([0]);
  await ple.gather([2]);
  // shard 0 を触り直す = 最近使ったのは 0 → 次に落ちるのは 1。
  await ple.gather([0]);
  assertEquals(ple.stats().loads, 2, "常駐にある shard を読み直している");

  await ple.gather([4]);
  assertEquals(ple.stats(), { loads: 3, resident: 2, residentBytes: 2 * SHARD_BUDGET });

  // 残っているのは 0 と 4 の shard（0 は読み直しゼロ・1 は読み直しになる）。
  await ple.gather([0]);
  assertEquals(ple.stats().loads, 3, "参照した shard が落ちている（LRU でなく FIFO）");
  await ple.gather([2]);
  assertEquals(ple.stats().loads, 4);
});

Deno.test("Gemma4Ple: 予算 0 は常駐なし（値は揃うが毎回読み直す）", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({
    index: INDEX,
    readShard: reader.readShard,
    vocabSize: TOKENS,
    maxResidentBytes: 0,
  });

  const tensor = await ple.gather([0, 3]);
  assert("data" in tensor && tensor.data instanceof Float32Array);
  // 常駐しなくても値は同じ（予算は RAM と読み直しの交換で、数値契約には触らない）。
  assertEquals(tensor.data[0], expectedValue(0, 0, 0));
  assertEquals(tensor.data[4], expectedValue(3, 0, 0));
  assertEquals(
    ple.stats(),
    { loads: 2, resident: 0, residentBytes: 0 },
    "予算 0 なのに gather 後も常駐が残っている",
  );

  // 同じ id でも読み直す（キャッシュが無いことの裏取り）。
  await ple.gather([0]);
  assertEquals(ple.stats(), { loads: 3, resident: 0, residentBytes: 0 });
});

Deno.test("Gemma4Ple: shard 1 本すら載らない予算は構築時に fail loudly", () => {
  assertThrows(
    () =>
      createGemma4Ple({
        index: UNEVEN_INDEX,
        readShard: unusedReader,
        vocabSize: UNEVEN_INDEX.tokens,
        // 小さい方（108）は載るが最大（252）は載らない = 引く id 次第で黙って超過する。
        maxResidentBytes: 251,
      }),
    Error,
    "PLE shard 1 本ぶん 252 バイトに満たない",
  );
  // 最大 shard ちょうどは通る（0 も「常駐させない」指定として通る）。
  createGemma4Ple({
    index: UNEVEN_INDEX,
    readShard: unusedReader,
    vocabSize: UNEVEN_INDEX.tokens,
    maxResidentBytes: 252,
  });
  createGemma4Ple({
    index: UNEVEN_INDEX,
    readShard: unusedReader,
    vocabSize: UNEVEN_INDEX.tokens,
    maxResidentBytes: 0,
  });

  assertThrows(
    () =>
      createGemma4Ple({
        index: INDEX,
        readShard: unusedReader,
        vocabSize: TOKENS,
        maxResidentBytes: -1,
      }),
    Error,
    "0 以上の整数でない",
  );
  assertThrows(
    () =>
      createGemma4Ple({
        index: INDEX,
        readShard: unusedReader,
        vocabSize: TOKENS,
        maxResidentBytes: 1.5,
      }),
    Error,
    "0 以上の整数でない",
  );
});
