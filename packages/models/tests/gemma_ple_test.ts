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
      () => parseGemma4PleIndex({ ...RAW_INDEX, schema: 2 }),
      Error,
      "ple.json.schema 2 が 1 でない",
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
        readShard: reader.readShard,
        vocabSize: TOKENS + 1,
      }),
    Error,
    `PLE sidecar の行数 ${TOKENS} が主 embedding の vocab 行数 ${TOKENS + 1} と違う`,
  );
  assertEquals(reader.calls.length, 0, "照合の前に shard を読みに行っている");
});

Deno.test("Gemma4Ple.gather: id の値域と空列は fail loudly（別 token の有効な行を引かせない）", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({ index: INDEX, readShard: reader.readShard, vocabSize: TOKENS });
  for (const bad of [TOKENS, -1, 1.5]) {
    await assertRejects(
      () => ple.gather([bad]),
      Error,
      `token id[0] ${bad} が PLE sidecar の 0..${TOKENS - 1} の外`,
    );
  }
  await assertRejects(() => ple.gather([]), Error, "PLE gather の token 列が空");
  assertEquals(reader.calls.length, 0, "値域の門より先に shard を読みに行っている");
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
  const open = (bytes: ArrayBuffer) => {
    const calls: string[] = [];
    const ple = createGemma4Ple({
      index: SOLO_INDEX,
      readShard: (file) => {
        calls.push(file);
        return Promise.resolve(bytes);
      },
      vocabSize: SOLO_INDEX.tokens,
    });
    return { ple, calls };
  };

  await t.step("正常系（陰性対照 — 読みは 1 回で通る）", async () => {
    const { ple, calls } = open(soloBytes());
    const tensor = await ple.gather([0]);
    assertEquals(tensor.shape, [1, 1, LAYERS, DIM]);
    assertEquals(calls, [SOLO_FILE]);
  });

  await t.step("metadata 欄そのものが無い（別形式の資産）", async () => {
    const { ple, calls } = open(soloBytes({ metadata: false }));
    await assertRejects(() => ple.gather([0]), Error, "__metadata__.karume_ple が無い");
    assertEquals(calls.length, 1, "読んだ後にしか分からない門である");
  });

  await t.step("範囲が索引とずれている（形も dtype も合ったまま別 token の行を引く）", async () => {
    const { ple } = open(soloBytes({ start: 1 }));
    const error = await assertRejects(
      () => ple.gather([0]),
      Error,
      "karume_ple が索引と食い違う",
    );
    assert(error.message.includes("start 1 ≠ 0"), error.message);
  });

  await t.step("values の格納 dtype が違う", async () => {
    const { ple } = open(soloBytes({ valuesAsF32: true }));
    await assertRejects(
      () => ple.gather([0]),
      Error,
      "'values' の格納 dtype が F32（I8 でない）",
    );
  });
});
