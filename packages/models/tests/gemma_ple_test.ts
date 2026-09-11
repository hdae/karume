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
// - **行読みの方針表**（ADR 0085 追記 2026-09-07）: 区間読みを持つ読み口で、shard ごとの一意行数と
//   費用の型（seek / scan）と予算の空きから「行読み / 全量読み」が決まる。値は経路に依らず
//   ビット同一で、ヘッダは shard ごとに 1 度しか読まない。

import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import {
  createGemma4Ple,
  defaultGemma4PleResidentBytes,
  type Gemma4PleIndex,
  type Gemma4PleReadOptions,
  gemma4PleShardBytes,
  type Gemma4PleShardSource,
  parseGemma4PleIndex,
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

/**
 * 索引 1 本ぶんの合成 shard を組む。
 *
 * NOTE: 行数の多い索引（{@link WIDE_INDEX}）では `quantized` が i8 の範囲を超えて切り詰められるが、
 * 切り詰めは決定的なので**経路どうしの比較**（全量読み ↔ 行読み）には影響しない。
 * {@link expectedValue} と突き合わせるのは既定の索引（id 0..5）だけ。
 */
const shardBytesOf = (index: Gemma4PleIndex, position: number): Uint8Array<ArrayBuffer> => {
  const shard = index.shards[position];
  const rows = shard.stop - shard.start;
  const values = new Int8Array(rows * index.layers * index.dim);
  const scales = new Float32Array(rows * index.layers);
  for (let row = 0; row < rows; row += 1) {
    for (let layer = 0; layer < index.layers; layer += 1) {
      scales[row * index.layers + layer] = scaleOf(layer);
      for (let column = 0; column < index.dim; column += 1) {
        values[(row * index.layers + layer) * index.dim + column] = quantized(
          shard.start + row,
          layer,
          column,
        );
      }
    }
  }
  const tensors = new Map<string, DumpTensor>([
    ["values", { dtype: "I8", shape: [rows, index.layers, index.dim], data: values }],
    ["scales", { dtype: "F32", shape: [rows, index.layers], data: scales }],
  ]);
  return writeSafetensors(tensors, {
    karume_ple: JSON.stringify({
      schema: 1,
      tokens: index.tokens,
      layers: index.layers,
      dim: index.dim,
      embedScale: index.embedScale,
      start: shard.start,
      stop: shard.stop,
    }),
  });
};

const SHARD_BYTES = INDEX.shards.map((_shard, position) => shardBytesOf(INDEX, position));

type ReadAllCall = { readonly file: string; readonly signal: AbortSignal | undefined };
type RangeCall = { readonly file: string; readonly offset: number; readonly length: number };

/**
 * 読み口の fake（呼び出しを全部記録する）。
 *
 * 方針表の分岐は「どの口が何回・どの区間で呼ばれたか」でしか観測できないので、全量読み
 * （{@link ReadAllCall}）と区間読み（{@link RangeCall}）を別々に記録する。`cost` を渡した
 * ときだけ区間読みが生え、渡さなければ**従来どおり全量だけの読み口**になる。
 *
 * 区間読みは **in-flight 本数のピーク**（`maxInFlight`）も持つ。1 read = 1 fd を取る取得元
 * （`denoDirectory`）では同時発行数がそのまま fd の占有なので、上限が効いていることは
 * 「呼ばれた回数」ではなくこのピークでしか観測できない。
 */
const fakeSources = (
  files: readonly string[],
  bytesOf: (position: number) => ArrayBuffer,
  options: { readonly honorSignal?: boolean; readonly cost?: "seek" | "scan" } = {},
) => {
  const opens: string[] = [];
  const readAll: ReadAllCall[] = [];
  const ranges: RangeCall[] = [];
  let inFlight = 0;
  const peak = { inFlight: 0 };
  const openShard = (file: string): Promise<Gemma4PleShardSource> => {
    opens.push(file);
    const position = files.indexOf(file);
    if (position < 0) return Promise.reject(new Error(`fake: 知らない shard '${file}'`));
    const bytes = bytesOf(position);
    return Promise.resolve({
      bytes: bytes.byteLength,
      readAll: (readOptions?: Gemma4PleReadOptions) => {
        const signal = readOptions?.signal;
        readAll.push({ file, signal });
        if (options.honorSignal === true && signal !== undefined) {
          return new Promise<ArrayBuffer>((resolve, reject) => {
            // 既に中断済みなら待たずに拒否する（実物の読み口は読みを始める前に `aborted` を
            // 見る — hub の `openAsset` / `streamAssets` はどちらも `throwIfAborted` が先）。
            if (signal.aborted) {
              reject(signal.reason);
              return;
            }
            // 「読みに時間がかかる」形（中断はこの待ちの間に届く）。
            const timer = setTimeout(() => resolve(bytes), 0);
            signal.addEventListener("abort", () => {
              clearTimeout(timer);
              reject(signal.reason);
            }, { once: true });
          });
        }
        return Promise.resolve(bytes);
      },
      ...(options.cost === undefined ? {} : {
        range: {
          cost: options.cost,
          read: (offset: number, length: number) => {
            ranges.push({ file, offset, length });
            inFlight += 1;
            peak.inFlight = Math.max(peak.inFlight, inFlight);
            // 減らすのは**解決した後**（`finally`）— 返す前に減らすと、この数は常に 1 になり
            // 「撒いたまま返ってきていない本数」を測らなくなる。
            return Promise.resolve(bytes.slice(offset, offset + length)).finally(() => {
              inFlight -= 1;
            });
          },
        },
      }),
    });
  };
  return { opens, readAll, ranges, peak, openShard };
};

/** 既定の索引（全 shard 同幅）に対する全量だけの読み口。 */
const fakeReader = (options: { readonly honorSignal?: boolean } = {}) =>
  fakeSources(SHARD_FILES, (position) => SHARD_BYTES[position].buffer, options);

Deno.test("Gemma4Ple: dispose で常駐が空になり、以後の gather は fail loudly", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({
    index: INDEX,
    openShard: reader.openShard,
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
  assertEquals(ple.stats(), {
    loads: 2,
    rowReads: 0,
    resident: 2,
    residentBytes: 2 * SHARD_BUDGET,
  });

  ple.dispose();
  // 解放は例外にならないので、実数で見るしかない（ここが 2 のままなら shard は返っていない）。
  assertEquals(ple.stats().resident, 0);
  await assertRejects(() => ple.gather([0]), Error, "dispose 済み");
  // 拒否は「引けない」だけで、読み直しも起きない。
  assertEquals(reader.readAll.length, 2);

  // 冪等（2 度目の dispose も、その後の stats も落ちない）。
  ple.dispose();
  assertEquals(ple.stats(), { loads: 2, rowReads: 0, resident: 0, residentBytes: 0 });
});

Deno.test("Gemma4Ple: gather の signal は shard の読み口へ降りる（best-effort）", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({ index: INDEX, openShard: reader.openShard, vocabSize: TOKENS });
  const controller = new AbortController();

  await ple.gather([0], { signal: controller.signal });
  assertEquals(reader.readAll.length, 1);
  assert(
    reader.readAll[0].signal === controller.signal,
    "gather の signal が読み口へ降りていない（shard 1 本の読みが中断の届かない区間になる）",
  );

  // 省略した gather は何も渡さない（購読していない呼び出しに signal を捏造しない）。
  await ple.gather([2]);
  assertEquals(reader.readAll[1].signal, undefined);
});

Deno.test("Gemma4Ple: 中断された読みは常駐に残らず、引き直しで読み直す", async () => {
  const reader = fakeReader({ honorSignal: true });
  const ple = createGemma4Ple({ index: INDEX, openShard: reader.openShard, vocabSize: TOKENS });
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
    { loads: 1, rowReads: 0, resident: 0, residentBytes: 0 },
    "拒否された取得を常駐させている",
  );

  // 同じ id をもう一度引けば読み直す（拒否済み Promise を掴み続けない）。
  const tensor = await ple.gather([0]);
  assert("data" in tensor);
  assertEquals(tensor.data[0], expectedValue(0, 0, 0));
  assertEquals(ple.stats(), { loads: 2, rowReads: 0, resident: 1, residentBytes: SHARD_BUDGET });
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

/** 読み口が開かれたら落とす（構築時の検査だけを見るテスト用）。 */
const unusedOpen = (file: string): Promise<Gemma4PleShardSource> =>
  Promise.reject(new Error(`fake: 開いてはいけない '${file}'`));

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
  const ple = createGemma4Ple({ index: INDEX, openShard: reader.openShard, vocabSize: TOKENS });

  await ple.gather([0]);
  await ple.gather([2]);
  assertEquals(ple.stats(), {
    loads: 2,
    rowReads: 0,
    resident: 2,
    residentBytes: 2 * SHARD_BUDGET,
  });

  await ple.gather([4]);
  assertEquals(
    ple.stats(),
    { loads: 3, rowReads: 0, resident: 2, residentBytes: 2 * SHARD_BUDGET },
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
    openShard: reader.openShard,
    vocabSize: TOKENS,
    maxResidentBytes: 2 * SHARD_BUDGET,
  });

  await ple.gather([0]);
  await ple.gather([2]);
  // shard 0 を触り直す = 最近使ったのは 0 → 次に落ちるのは 1。
  await ple.gather([0]);
  assertEquals(ple.stats().loads, 2, "常駐にある shard を読み直している");

  await ple.gather([4]);
  assertEquals(ple.stats(), {
    loads: 3,
    rowReads: 0,
    resident: 2,
    residentBytes: 2 * SHARD_BUDGET,
  });

  // 残っているのは 0 と 4 の shard（0 は読み直しゼロ・1 は読み直しになる）。
  await ple.gather([0]);
  assertEquals(ple.stats().loads, 3, "参照した shard が落ちている（LRU でなく FIFO）");
  await ple.gather([2]);
  assertEquals(ple.stats().loads, 4);
});

/** 各 shard の先頭 id（shard 0 → 0 / shard 1 → 2 / shard 2 → 4）。 */
const FIRST_ID = INDEX.shards.map((shard) => shard.start);

Deno.test("Gemma4Ple.gather: 常駐している shard を先に処理する（1 回の gather が自分の hit を追い出さない）", async (t) => {
  const warmed = async () => {
    const reader = fakeReader();
    const ple = createGemma4Ple({
      index: INDEX,
      openShard: reader.openShard,
      vocabSize: TOKENS,
      maxResidentBytes: 2 * SHARD_BUDGET,
    });
    // shard 1 と 2 を常駐させる（予算ちょうど = 次の miss が必ず 1 本追い出す状態）。
    await ple.gather([FIRST_ID[1], FIRST_ID[2]]);
    assertEquals(ple.stats(), {
      loads: 2,
      rowReads: 0,
      resident: 2,
      residentBytes: 2 * SHARD_BUDGET,
    });
    return ple;
  };

  await t.step("miss を先頭に置いた順（0,1,2）でも読みは未常駐の 1 本だけ", async () => {
    const ple = await warmed();
    const tensor = await ple.gather([FIRST_ID[0], FIRST_ID[1], FIRST_ID[2]]);
    // 未常駐は shard 0 の 1 本だけ。hit を後回しにすると shard 0 の読みが 1 を、1 の読みが 2 を
    // 追い出して 3 本読むことになる（この gather の中で自分の hit を捨てている形）。
    assertEquals(ple.stats().loads, 3, "1 回の gather の中で常駐 shard を読み直している");
    assertEquals(tensor.shape, [1, 3, LAYERS, DIM]);
    assert("data" in tensor && tensor.data instanceof Float32Array);
    assertEquals(tensor.data[0], expectedValue(FIRST_ID[0], 0, 0));
    assertEquals(tensor.data[LAYERS * DIM], expectedValue(FIRST_ID[1], 0, 0));
    assertEquals(tensor.data[2 * LAYERS * DIM], expectedValue(FIRST_ID[2], 0, 0));
  });

  await t.step(
    "hit を先頭に置いた順（1,2,0）も同じ読み回数（順序で結果が変わらない）",
    async () => {
      const ple = await warmed();
      await ple.gather([FIRST_ID[1], FIRST_ID[2], FIRST_ID[0]]);
      assertEquals(ple.stats().loads, 3);
    },
  );
});

Deno.test("Gemma4Ple.gather: 同じ id が並ぶ列でも各行は単発 gather とビット一致する", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({
    index: INDEX,
    openShard: reader.openShard,
    vocabSize: TOKENS,
    // 3 本とも常駐させる（読み回数ではなく行の値だけを見るため）。
    maxResidentBytes: 3 * SHARD_BUDGET,
  });
  const stride = LAYERS * DIM;

  // 参照 = 各 id をちょうど 1 件ずつ引いた行。
  const reference = await ple.gather([0, 1, 2, 3, 4, 5]);
  assert("data" in reference && reference.data instanceof Float32Array);

  // prefill の pad 行と同じ形（大半が id 0・数十箇所だけ別の id）。
  const input = Array.from({ length: 768 }, (_value, position) => (
    position % 41 === 7 ? position % TOKENS : 0
  ));
  const duplicated = input.filter((id) => id === 0).length;
  assert(duplicated > 700, `重複していない列を測っている（id 0 は ${duplicated} 件）`);

  const bulk = await ple.gather(input);
  assertEquals(bulk.shape, [1, input.length, LAYERS, DIM]);
  assert("data" in bulk && bulk.data instanceof Float32Array);
  // 陰性対照 — 全部 0 の配列同士を比べて緑になっていない。
  assertEquals(bulk.data[1], expectedValue(0, 0, 1));

  const bulkWords = new Uint32Array(bulk.data.buffer);
  const referenceWords = new Uint32Array(reference.data.buffer);
  let mismatch = -1;
  for (let word = 0; word < bulkWords.length; word += 1) {
    const id = input[Math.floor(word / stride)];
    if (bulkWords[word] !== referenceWords[id * stride + (word % stride)]) {
      mismatch = word;
      break;
    }
  }
  assertEquals(
    mismatch,
    -1,
    `複写した行が単発 gather の行とビット一致しない（word ${mismatch} = 位置 ${
      Math.floor(mismatch / stride)
    }）`,
  );
});

Deno.test("Gemma4Ple.gather: 同じ未常駐 shard を同時に要求しても読みは 1 回", async () => {
  const reader = fakeReader({ honorSignal: true });
  const ple = createGemma4Ple({ index: INDEX, openShard: reader.openShard, vocabSize: TOKENS });
  // 中断しない signal を渡して fake を「読みに時間がかかる」分岐へ入れる（pending の窓を実際に
  // 開ける — signal 無しだと即時解決で窓が 1 microtask しか無い）。
  const { signal } = new AbortController();

  // 2 本目の gather は 1 本目が登録した **pending** を掴む（解決を待たずに hit と見なす）。
  const [first, second] = await Promise.all([
    ple.gather([0], { signal }),
    ple.gather([1], { signal }),
  ]);
  assertEquals(
    reader.readAll.length,
    1,
    "同じ shard を 2 度読みに行っている（758MB 級の二重読み）",
  );
  assertEquals(ple.stats(), { loads: 1, rowReads: 0, resident: 1, residentBytes: SHARD_BUDGET });
  assert("data" in first && "data" in second);
  assertEquals(first.data[0], expectedValue(0, 0, 0));
  assertEquals(second.data[0], expectedValue(1, 0, 0));
});

Deno.test("Gemma4Ple: 予算 0 は常駐なし（値は揃うが毎回読み直す）", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({
    index: INDEX,
    openShard: reader.openShard,
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
    { loads: 2, rowReads: 0, resident: 0, residentBytes: 0 },
    "予算 0 なのに gather 後も常駐が残っている",
  );

  // 同じ id でも読み直す（キャッシュが無いことの裏取り）。
  await ple.gather([0]);
  assertEquals(ple.stats(), { loads: 3, rowReads: 0, resident: 0, residentBytes: 0 });
});

Deno.test("Gemma4Ple: shard 1 本すら載らない予算は構築時に fail loudly", () => {
  assertThrows(
    () =>
      createGemma4Ple({
        index: UNEVEN_INDEX,
        openShard: unusedOpen,
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
    openShard: unusedOpen,
    vocabSize: UNEVEN_INDEX.tokens,
    maxResidentBytes: 252,
  });
  createGemma4Ple({
    index: UNEVEN_INDEX,
    openShard: unusedOpen,
    vocabSize: UNEVEN_INDEX.tokens,
    maxResidentBytes: 0,
  });

  assertThrows(
    () =>
      createGemma4Ple({
        index: INDEX,
        openShard: unusedOpen,
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
        openShard: unusedOpen,
        vocabSize: TOKENS,
        maxResidentBytes: 1.5,
      }),
    Error,
    "0 以上の整数でない",
  );
});

// ---- 資産境界の拒否経路（外部入力としての `ple.json` と shard）-----------------
//
// `parseGemma4PleIndex` の門・`createGemma4Ple` の vocab 相互照合（ADR 0085 決定 5 の沈黙誤値
// ガード）・`gather` の id 値域は、実際に呼ぶのが実資産 e2e の**正常系 1 通り**だけだった
// （しかも資産の無い環境では 1 度も走らない）。索引と shard を片方だけ焼き直した組み合わせ・
// 別語彙で焼いた sidecar は、この 3 つのガードが唯一の検出線である。

/** `unknown` 境界へ渡す素の索引（`Gemma4PleIndex` 型は `schema` 欄を持たないので別に組む）。 */
const RAW_INDEX: Record<string, unknown> = {
  schema: 1,
  tokens: TOKENS,
  layers: LAYERS,
  dim: DIM,
  embedScale: EMBED_SCALE,
  shards: INDEX.shards.map((shard) => ({ ...shard })),
};

Deno.test("parseGemma4PleIndex: 壊れた索引を黙って読まない", async (t) => {
  await t.step("正常系（陰性対照 — 常に落ちる門になっていない）", () => {
    assertEquals(parseGemma4PleIndex(RAW_INDEX), INDEX);
  });

  await t.step("知らない版は読まない", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...RAW_INDEX, schema: 3 }),
      Error,
      "ple.json.schema 3 が 1 / 2 でない",
    );
  });

  await t.step("未知キー（綴り違いが黙って既定へ縮退しない）", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...RAW_INDEX, extra: 1 }),
      Error,
      "未知キー 'extra'",
    );
  });

  await t.step("embedScale は正の有限数（0 は行が全部 0 になる）", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...RAW_INDEX, embedScale: 0 }),
      Error,
      "ple.json.embedScale 0 が正の有限数でない",
    );
  });

  await t.step("shard 名の重複（同じ実体を 2 つの範囲が名乗る）", () => {
    assertThrows(
      () =>
        parseGemma4PleIndex({
          ...RAW_INDEX,
          tokens: 4,
          shards: [
            { file: SHARD_FILES[0], start: 0, stop: 2 },
            { file: SHARD_FILES[0], start: 2, stop: 4 },
          ],
        }),
      Error,
      `.file '${SHARD_FILES[0]}' が重複している`,
    );
  });

  await t.step("範囲の非連続（引けない id が黙って生まれる）", () => {
    assertThrows(
      () =>
        parseGemma4PleIndex({
          ...RAW_INDEX,
          tokens: 5,
          shards: [
            { file: SHARD_FILES[0], start: 0, stop: 2 },
            { file: SHARD_FILES[1], start: 3, stop: 5 },
          ],
        }),
      Error,
      ".start 3 が直前の shard の末尾 2 と連続しない",
    );
  });

  await t.step("空範囲", () => {
    assertThrows(
      () =>
        parseGemma4PleIndex({
          ...RAW_INDEX,
          tokens: 2,
          shards: [
            { file: SHARD_FILES[0], start: 0, stop: 2 },
            { file: SHARD_FILES[1], start: 2, stop: 2 },
          ],
        }),
      Error,
      "範囲 [2, 2) が空",
    );
  });

  await t.step("合計行数の不一致（索引だけ焼き直した組み合わせ）", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...RAW_INDEX, tokens: TOKENS + 1 }),
      Error,
      `shard の合計 ${TOKENS} 行が tokens ${TOKENS + 1} と違う`,
    );
  });
});

Deno.test("createGemma4Ple: vocab の相互照合は読み口に触る前に落ちる", () => {
  // ADR 0085 決定 5 — 別語彙で焼いた sidecar は「引ける id が食い違ったまま形は合う」。
  const reader = fakeReader();
  assertThrows(
    () =>
      createGemma4Ple({
        index: INDEX,
        openShard: reader.openShard,
        vocabSize: TOKENS + 1,
      }),
    Error,
    `PLE sidecar の行数 ${TOKENS} が主 embedding の vocab 行数 ${TOKENS + 1} と違う`,
  );
  assertEquals(reader.readAll.length, 0, "照合の前に shard を読みに行っている");
});

Deno.test("Gemma4Ple.gather: id の値域と空列は fail loudly（別 token の有効な行を引かせない）", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({ index: INDEX, openShard: reader.openShard, vocabSize: TOKENS });
  for (const bad of [TOKENS, -1, 1.5]) {
    await assertRejects(
      () => ple.gather([bad]),
      Error,
      `token id[0] ${bad} が PLE sidecar の 0..${TOKENS - 1} の外`,
    );
  }
  await assertRejects(() => ple.gather([]), Error, "PLE gather の token 列が空");
  assertEquals(reader.readAll.length, 0, "値域の門より先に shard を読みに行っている");
});

/** 1 shard だけの索引（shard 側 metadata の門を 1 本の読みで踏むため）。 */
const SOLO_FILE = "solo.safetensors";
const SOLO_INDEX: Gemma4PleIndex = {
  tokens: 2,
  layers: LAYERS,
  dim: DIM,
  embedScale: EMBED_SCALE,
  shards: [{ file: SOLO_FILE, start: 0, stop: 2 }],
};

/** 1 shard ぶんのバイト列を 1 点だけ壊して組む（正常系はそのまま通ることを対で見る）。 */
const soloBytes = (
  patch: {
    readonly metadata?: boolean;
    readonly start?: number;
    readonly valuesAsF32?: boolean;
  } = {},
): ArrayBuffer => {
  const rows = 2;
  const count = rows * LAYERS * DIM;
  const values: DumpTensor = patch.valuesAsF32 === true
    ? { dtype: "F32", shape: [rows, LAYERS, DIM], data: new Float32Array(count) }
    : { dtype: "I8", shape: [rows, LAYERS, DIM], data: new Int8Array(count) };
  const tensors = new Map<string, DumpTensor>([
    ["values", values],
    ["scales", { dtype: "F32", shape: [rows, LAYERS], data: new Float32Array(rows * LAYERS) }],
  ]);
  const metadata: Record<string, string> = patch.metadata === false ? {} : {
    karume_ple: JSON.stringify({
      schema: 1,
      tokens: SOLO_INDEX.tokens,
      layers: LAYERS,
      dim: DIM,
      embedScale: EMBED_SCALE,
      start: patch.start ?? 0,
      stop: 2,
    }),
  };
  return writeSafetensors(tensors, metadata).buffer;
};

Deno.test("Gemma4Ple: shard 側 metadata は索引と突き合わせる（片方だけ焼き直した組み合わせ）", async (t) => {
  const open = (bytes: ArrayBuffer, cost?: "seek") => {
    const reader = fakeSources([SOLO_FILE], () => bytes, cost === undefined ? {} : { cost });
    return {
      reader,
      ple: createGemma4Ple({
        index: SOLO_INDEX,
        openShard: reader.openShard,
        vocabSize: SOLO_INDEX.tokens,
      }),
    };
  };

  // 全量経路と行読み経路の**両方**を同じ表で踏む。検査は 1 実装（`assertShardTables`）だが、
  // 通る道が 2 本ある以上「片方だけ緩い」形は門でしか塞げない（ADR 0085 追記 2026-09-07）。
  for (const cost of [undefined, "seek"] as const) {
    const path = cost === undefined ? "全量" : "行読み";

    await t.step(`${path}: 正常系（陰性対照 — 読みは 1 回で通る）`, async () => {
      const { ple, reader } = open(soloBytes(), cost);
      const tensor = await ple.gather([0]);
      assertEquals(tensor.shape, [1, 1, LAYERS, DIM]);
      assertEquals(reader.opens, [SOLO_FILE]);
      assertEquals(
        reader.readAll.length,
        cost === undefined ? 1 : 0,
        "行読みできる読み口なのに全量を読んでいる（1 id の gather）",
      );
    });

    await t.step(`${path}: metadata 欄そのものが無い（別形式の資産）`, async () => {
      const { ple, reader } = open(soloBytes({ metadata: false }), cost);
      await assertRejects(() => ple.gather([0]), Error, "__metadata__.karume_ple が無い");
      assertEquals(reader.opens.length, 1, "読んだ後にしか分からない門である");
    });

    await t.step(
      `${path}: 範囲が索引とずれている（形も dtype も合ったまま別 token の行を引く）`,
      async () => {
        const { ple } = open(soloBytes({ start: 1 }), cost);
        const error = await assertRejects(
          () => ple.gather([0]),
          Error,
          "karume_ple が索引と食い違う",
        );
        assert(error.message.includes("start 1 ≠ 0"), error.message);
      },
    );

    await t.step(`${path}: values の格納 dtype が違う`, async () => {
      const { ple } = open(soloBytes({ valuesAsF32: true }), cost);
      await assertRejects(
        () => ple.gather([0]),
        Error,
        "'values' の格納 dtype が F32（I8 でない）",
      );
    });
  }
});

// ---- 行読みの方針表（ADR 0085 追記 2026-09-07）------------------------------
//
// 方針は shard ごとの**一意行数**と費用の型（seek / scan）と予算の空きだけで決まる。値は経路に
// 依らずビット同一で、それを確かめる唯一の方法は「同じ id を両経路で引いて u32 で比べる」ことで
// ある（f32 の `===` は NaN / ±0 を取りこぼす）。
//
// 既定の索引（{@link INDEX}）は 1 shard 2 行しかないので、`rows ≥ 32` と `rows > 2` の段を踏むには
// 広い shard の索引が要る。

const WIDE_ROWS = 40;
const WIDE_FILES = ["wide-00001-of-00002.safetensors", "wide-00002-of-00002.safetensors"] as const;
const WIDE_INDEX: Gemma4PleIndex = {
  tokens: 2 * WIDE_ROWS,
  layers: LAYERS,
  dim: DIM,
  embedScale: EMBED_SCALE,
  shards: WIDE_FILES.map((file, position) => ({
    file,
    start: position * WIDE_ROWS,
    stop: (position + 1) * WIDE_ROWS,
  })),
};
const WIDE_BUDGET = gemma4PleShardBytes(WIDE_INDEX, WIDE_INDEX.shards[0]);
const WIDE_BYTES = WIDE_INDEX.shards.map((_shard, position) => shardBytesOf(WIDE_INDEX, position));

const wideSources = (cost: "seek" | "scan") =>
  fakeSources(WIDE_FILES, (position) => WIDE_BYTES[position].buffer, { cost });

/** その shard の全行の id（`rows ≥ 32` の段を踏むための列）。 */
const wideIds = (shard: number): number[] =>
  Array.from({ length: WIDE_ROWS }, (_value, row) => shard * WIDE_ROWS + row);

/** f32 の bit 列（`===` では NaN / ±0 を取りこぼす）。 */
const words = (data: Float32Array<ArrayBuffer>): number[] => [...new Uint32Array(data.buffer)];

Deno.test("Gemma4Ple.gather: seek の小さい gather は全量を読まず行だけを引く（decode 1 token）", async () => {
  const reader = fakeSources(SHARD_FILES, (position) => SHARD_BYTES[position].buffer, {
    cost: "seek",
  });
  // 予算は既定（最大 shard 2 本ぶん）— **空きがあっても** 1 行なら行読みへ倒れる。
  const ple = createGemma4Ple({ index: INDEX, openShard: reader.openShard, vocabSize: TOKENS });

  const tensor = await ple.gather([3]);
  assertEquals(
    ple.stats(),
    { loads: 0, rowReads: 1, resident: 0, residentBytes: 0 },
    "1 id の gather で shard 全量（253MiB 級）を読んでいる",
  );
  assertEquals(reader.readAll.length, 0);
  // 読みはヘッダ 2 段 + 行 2 区間の 4 回だけ（発行順は values → scales）。
  assertEquals(reader.ranges.length, 4);
  assertEquals(reader.ranges[0].offset, 0);
  assertEquals(reader.ranges[0].length, 8, "1 段目はヘッダ長の 8 バイト");
  assertEquals(reader.ranges[1].offset, 8, "2 段目はヘッダ JSON 本体");
  assertEquals(reader.ranges[2].length, LAYERS * DIM, "values 1 行（i8）");
  assertEquals(reader.ranges[3].length, LAYERS * 4, "scales 1 行（f32）");

  // 全量経路と**ビット一致**（同じバイト列を同じ 2 段丸めに掛けているだけ）。
  const full = fakeReader();
  const reference = createGemma4Ple({
    index: INDEX,
    openShard: full.openShard,
    vocabSize: TOKENS,
  });
  const expected = await reference.gather([3]);
  assertEquals(full.readAll.length, 1, "陰性対照が行読みへ倒れている（区間読みを持たない口）");
  assert("data" in tensor && tensor.data instanceof Float32Array);
  assert("data" in expected && expected.data instanceof Float32Array);
  assertEquals(words(tensor.data), words(expected.data), "行読みの値が全量経路とビット一致しない");
  assertEquals([...tensor.data], [
    expectedValue(3, 0, 0),
    expectedValue(3, 0, 1),
    expectedValue(3, 1, 0),
    expectedValue(3, 1, 1),
  ]);
});

Deno.test("Gemma4Ple.gather: seek の全量読みは「予算の空きに追い出し無しで載る」ときだけ", async (t) => {
  await t.step("空きがある → 全量 1 回で常駐（以後の hit のために載せる）", async () => {
    const reader = wideSources("seek");
    const ple = createGemma4Ple({
      index: WIDE_INDEX,
      openShard: reader.openShard,
      vocabSize: WIDE_INDEX.tokens,
      maxResidentBytes: 2 * WIDE_BUDGET,
    });
    await ple.gather(wideIds(0));
    assertEquals(ple.stats(), {
      loads: 1,
      rowReads: 0,
      resident: 1,
      residentBytes: WIDE_BUDGET,
    });
    assertEquals(reader.ranges.length, 0, "全量へ倒したのに区間読みを呼んでいる");
  });

  await t.step("空きが無い → 追い出さずに行読みへ倒す", async () => {
    const reader = wideSources("seek");
    const ple = createGemma4Ple({
      index: WIDE_INDEX,
      openShard: reader.openShard,
      vocabSize: WIDE_INDEX.tokens,
      maxResidentBytes: WIDE_BUDGET,
    });
    await ple.gather(wideIds(0));
    assertEquals(ple.stats().resident, 1, "予算ちょうどが埋まっていない");
    await ple.gather(wideIds(1));
    assertEquals(
      ple.stats(),
      { loads: 1, rowReads: WIDE_ROWS, resident: 1, residentBytes: WIDE_BUDGET },
      "常駐を追い出して 2 本目を載せている（9 本に散る自然文で LRU が回り続ける形）",
    );
  });

  await t.step("1 回の gather の中でも予算を超えて載せない", async () => {
    const reader = wideSources("seek");
    const ple = createGemma4Ple({
      index: WIDE_INDEX,
      openShard: reader.openShard,
      vocabSize: WIDE_INDEX.tokens,
      maxResidentBytes: WIDE_BUDGET,
    });
    // 2 本とも `rows ≥ 32` だが、予算に載るのは先頭の 1 本だけ。
    await ple.gather([...wideIds(0), ...wideIds(1)]);
    assertEquals(ple.stats(), {
      loads: 1,
      rowReads: WIDE_ROWS,
      resident: 1,
      residentBytes: WIDE_BUDGET,
    });
  });
});

Deno.test("Gemma4Ple.gather: scan は 2 行までが行読み（3 行目からは全量 1 回が安い）", async (t) => {
  await t.step("2 行 → 行読み", async () => {
    const reader = wideSources("scan");
    const ple = createGemma4Ple({
      index: WIDE_INDEX,
      openShard: reader.openShard,
      vocabSize: WIDE_INDEX.tokens,
    });
    await ple.gather([0, 1]);
    assertEquals(ple.stats(), { loads: 0, rowReads: 2, resident: 0, residentBytes: 0 });
  });

  await t.step("3 行 → 全量読み + LRU", async () => {
    const reader = wideSources("scan");
    const ple = createGemma4Ple({
      index: WIDE_INDEX,
      openShard: reader.openShard,
      vocabSize: WIDE_INDEX.tokens,
    });
    await ple.gather([0, 1, 2]);
    assertEquals(ple.stats(), {
      loads: 1,
      rowReads: 0,
      resident: 1,
      residentBytes: WIDE_BUDGET,
    });
    assertEquals(reader.ranges.length, 0, "全量へ倒したのに区間読みを呼んでいる");
  });

  await t.step("同じ id の重複は 1 行に束ねてから数える（列の長さでは決まらない）", async () => {
    const reader = wideSources("scan");
    const ple = createGemma4Ple({
      index: WIDE_INDEX,
      openShard: reader.openShard,
      vocabSize: WIDE_INDEX.tokens,
    });
    // 位置は 6 つでも一意行は 2 つ（prefill の pad 行と同じ形）。
    await ple.gather([0, 0, 0, 1, 1, 0]);
    assertEquals(ple.stats(), { loads: 0, rowReads: 2, resident: 0, residentBytes: 0 });
  });
});

Deno.test("Gemma4Ple.gather: seek の境目は 32 行（予算に空きがあっても 31 行は行読み）", async (t) => {
  /** 予算に**空きがある**状態で `rows` 行だけを引く（境目そのものを見る）。 */
  const gatherRows = async (rows: number) => {
    const reader = wideSources("seek");
    const ple = createGemma4Ple({
      index: WIDE_INDEX,
      openShard: reader.openShard,
      vocabSize: WIDE_INDEX.tokens,
      // 2 本ぶん = この gather が載せる 1 本には必ず空きがある（分岐の軸を行数だけにする）。
      maxResidentBytes: 2 * WIDE_BUDGET,
    });
    await ple.gather(wideIds(0).slice(0, rows));
    return ple.stats();
  };

  // 境目は `ple.ts` の `SEEK_FULL_ROWS`（= 32）。
  await t.step("31 行 → 行読み（載せる価値が出るのは 32 行から）", async () => {
    assertEquals(await gatherRows(31), { loads: 0, rowReads: 31, resident: 0, residentBytes: 0 });
  });

  await t.step("32 行 → 全量読み + 常駐（以後の gather の hit のために載せる）", async () => {
    assertEquals(await gatherRows(32), {
      loads: 1,
      rowReads: 0,
      resident: 1,
      residentBytes: WIDE_BUDGET,
    });
  });
});

/** 行読みを 200 行ぶん撒く索引（1 shard・同時発行の上限だけを見る）。 */
const POOL_FILE = "pool.safetensors";
const POOL_ROWS = 200;
const POOL_INDEX: Gemma4PleIndex = {
  tokens: POOL_ROWS,
  layers: LAYERS,
  dim: DIM,
  embedScale: EMBED_SCALE,
  shards: [{ file: POOL_FILE, start: 0, stop: POOL_ROWS }],
};

Deno.test("Gemma4Ple.gather: 行読みの同時発行は上限を超えない（1 read = 1 fd の取得元で fd を枯らさない）", async () => {
  const bytes = shardBytesOf(POOL_INDEX, 0);
  const reader = fakeSources([POOL_FILE], () => bytes.buffer, { cost: "seek" });
  const ple = createGemma4Ple({
    index: POOL_INDEX,
    openShard: reader.openShard,
    vocabSize: POOL_ROWS,
    // 常駐させない = 200 行が全部行読みへ倒れる（prefill の 768 行 chunk と同じ形）。
    maxResidentBytes: 0,
  });

  await ple.gather(Array.from({ length: POOL_ROWS }, (_value, row) => row));
  assertEquals(ple.stats(), { loads: 0, rowReads: POOL_ROWS, resident: 0, residentBytes: 0 });

  // 陰性対照 — 直列に流していれば上限は意味を持たない（この数が 1 なら並べていない）。
  assert(reader.peak.inFlight > 1, `行読みが直列化している（ピーク ${reader.peak.inFlight}）`);
  // 上限は `ple.ts` の `ROW_READ_CONCURRENCY`（= 16・実測のピークもちょうど 16）。上限が
  // 無いと引く行数ぶん（ここでは 200 本）が同時に立ち、1 read = 1 fd の取得元では
  // `ulimit -n 1024` を prefill の 768 行 chunk が丸ごと食う（1,200 行で EMFILE を実測）。
  assert(
    reader.peak.inFlight <= 16,
    `行読みの同時発行のピーク ${reader.peak.inFlight} が上限 16 を超えた`,
  );
});

Deno.test("Gemma4Ple.gather: shard のヘッダは 1 度しか読まない（行読みを繰り返しても）", async () => {
  const reader = fakeSources(SHARD_FILES, (position) => SHARD_BYTES[position].buffer, {
    cost: "seek",
  });
  const ple = createGemma4Ple({ index: INDEX, openShard: reader.openShard, vocabSize: TOKENS });

  await ple.gather([0]);
  const headerReads = () => reader.ranges.filter((call) => call.offset === 0).length;
  assertEquals(headerReads(), 1);
  assertEquals(reader.ranges.length, 4, "ヘッダ 2 段 + 行 2 区間でない");

  // 同じ shard の別の行（ヘッダは掴んだままのはず）。
  await ple.gather([1]);
  assertEquals(headerReads(), 1, "2 度目の gather でヘッダを読み直している");
  assertEquals(reader.ranges.length, 6, "2 度目の gather が行 2 区間だけで済んでいない");
  assertEquals(reader.opens, [SHARD_FILES[0]], "読み口を 2 度開いている");
  assertEquals(ple.stats(), { loads: 0, rowReads: 2, resident: 0, residentBytes: 0 });
});

Deno.test("Gemma4Ple.gather: 宣言 bytes に収まらない区間は読み口を呼ばずに落ちる", async () => {
  // 宣言 4 バイト = ヘッダ長すら無い shard。8 バイト要求が宣言の外なので、読み口は 1 度も
  // 呼ばれない（短い戻りを黙って受けると、0 埋めのヘッダを解こうとして別の失敗に化ける）。
  const reader = fakeSources([SOLO_FILE], () => new ArrayBuffer(4), { cost: "seek" });
  const ple = createGemma4Ple({
    index: SOLO_INDEX,
    openShard: reader.openShard,
    vocabSize: SOLO_INDEX.tokens,
  });

  await assertRejects(
    () => ple.gather([0]),
    Error,
    `${SOLO_FILE}: 区間 [0, 8) が宣言 4 バイトの外`,
  );
  assertEquals(reader.ranges.length, 0, "範囲外の要求を読み口へ渡している");
  assertEquals(reader.readAll.length, 0);
});

Deno.test("Gemma4Ple: 先行読みが dispose 後に完了しても常駐を復活させない", async (t) => {
  for (const stage of ["open", "readAll", "range"] as const) {
    await t.step(stage, async () => {
      const entered = Promise.withResolvers<void>();
      const resume = Promise.withResolvers<void>();
      const pause = async (): Promise<void> => {
        entered.resolve();
        await resume.promise;
      };
      const bytes = SHARD_BYTES[0].buffer;
      const ple = createGemma4Ple({
        index: INDEX,
        vocabSize: TOKENS,
        maxResidentBytes: 2 * SHARD_BUDGET,
        openShard: async () => {
          if (stage === "open") await pause();
          return {
            bytes: bytes.byteLength,
            readAll: async () => {
              if (stage === "readAll") await pause();
              return bytes;
            },
            ...(stage !== "range" ? {} : {
              range: {
                cost: "seek" as const,
                read: async (offset: number, length: number) => {
                  await pause();
                  return bytes.slice(offset, offset + length);
                },
              },
            }),
          };
        },
      });
      const pending = ple.gather([0]);
      await entered.promise;
      ple.dispose();
      ple.dispose();
      resume.resolve();
      const output = await pending;
      assert("data" in output);
      assertEquals([...output.data], [0, 2, 2, 3]);
      assertEquals(ple.stats().resident, 0);
      assertEquals(ple.stats().residentBytes, 0);
      await assertRejects(() => ple.gather([0]), Error, "dispose 済み");
    });
  }
});

for (const rows of [1, 8, 9, 16]) {
  Deno.test(`Gemma4Ple.gather: ${rows} 行の値とscaleの読込を上限内で重ねる`, async () => {
    const bytes = shardBytesOf(POOL_INDEX, 0);
    const reader = fakeSources([POOL_FILE], () => bytes.buffer, { cost: "seek" });
    const ple = createGemma4Ple({
      index: POOL_INDEX,
      openShard: reader.openShard,
      vocabSize: POOL_ROWS,
      maxResidentBytes: 0,
    });
    try {
      await ple.gather(Array.from({ length: rows }, (_, row) => row));
      assertEquals(reader.peak.inFlight, rows <= 8 ? rows * 2 : rows);
      assertEquals(ple.stats().rowReads, rows);
    } finally {
      ple.dispose();
    }
  });
}

for (const rejected of ["values", "scales", "both"] as const) {
  Deno.test(`Gemma4Ple.gather: ${rejected} の拒否でも相手の読込を待って原因を保持する`, async () => {
    const bytes = SHARD_BYTES[0].buffer;
    const gate = Promise.withResolvers<void>();
    const both = Promise.withResolvers<void>();
    const valuesError = new Error("values failure");
    const scalesError = new Error("scales failure");
    let inject = false;
    let calls = 0;
    const ple = createGemma4Ple({
      index: INDEX,
      vocabSize: TOKENS,
      maxResidentBytes: 0,
      openShard: () =>
        Promise.resolve({
          bytes: bytes.byteLength,
          readAll: () => Promise.resolve(bytes),
          range: {
            cost: "seek",
            read: async (offset, length) => {
              if (inject) {
                const call = calls++;
                assert(call < 2);
                if (calls === 2) both.resolve();
                if (call === 0 && rejected !== "scales") throw valuesError;
                if (call === 1 && rejected === "scales") throw scalesError;
                await gate.promise;
                if (rejected === "both") throw scalesError;
              }
              return bytes.slice(offset, offset + length);
            },
          },
        }),
    });
    await ple.gather([0]);
    inject = true;
    const running = ple.gather([1]);
    let settled = false;
    const checked = running.then(() => {
      settled = true;
    }, () => {
      settled = true;
    });
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        both.promise,
        new Promise<never>((_, reject) => {
          timeout = setTimeout(() => reject(Error("row reads were serialized")), 1000);
        }),
      ]);
      await new Promise((resolve) => setTimeout(resolve, 0));
      assertEquals(settled, false, "もう片方の読込が未完了のままgatherを返さない");
      gate.resolve();
      const error = await assertRejects(() => running);
      assertStrictEquals(error, rejected === "scales" ? scalesError : valuesError);
      assertEquals(ple.stats().rowReads, 1, "失敗した行を完了済みとして数えない");
      inject = false;
      await ple.gather([1]);
      assertEquals(ple.stats().rowReads, 2);
    } finally {
      if (timeout !== undefined) clearTimeout(timeout);
      gate.resolve();
      await checked;
      ple.dispose();
    }
  });
}
