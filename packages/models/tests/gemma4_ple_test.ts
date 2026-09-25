// PLE のホスト gather（`src/gemma/ple.ts` — ADR 0085 / 0109 決定 4）の寿命・予算・読み方の門。
// GPU も実資産も要らない（合成の索引と block は `helpers/ple-fixture.ts` が作る）。
//
// ここで縛るのは 6 つ:
//
// - **索引の受理集合**（`parseGemma4PleIndex`）: schema 3 だけ・未知キー・区間の連続性・
//   `rowBytes` の整合。索引だけを焼き直した組み合わせはここでしか落ちない。
// - **索引と容器の資産の突合**（`assertGemma4PleAssets` / `readGemma4PleIndex`）: block の実在・
//   役割・論理長を**全件列挙**で見る。片方だけ焼き直した配布形は数十件が同時にずれるので、
//   1 件ずつ直す往復にしない。
// - **行の翻訳**: token → (block, 行 offset) が block 境界と block 跨ぎで正しいこと。ずれると
//   OOB ではなく**別 token の有効な行**を引く（ADR 0085 決定 5 の沈黙誤値）。
// - **方針表**: block ごとの一意行数と予算の空きから「全量読み / 行読み」が決まる。値は経路に
//   依らずビット同一で、それを確かめる唯一の方法は「同じ id を両経路で引いて u32 で比べる」
//   ことである（f32 の `===` は NaN / ±0 を取りこぼす）。
// - **バイト予算**: 常駐上限は**本数ではなくバイト**（block 幅は資産世代で変わる）。既定・
//   LRU の追い出し順・0（常駐なし）・1 本すら載らない予算の拒否を凍結する。
// - **解放**: `dispose()` で常駐が空になり、以後の gather は fail loudly。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { createGemma4Ple } from "../src/gemma/ple.ts";
import {
  assertGemma4PleAssets,
  defaultGemma4PleResidentBytes,
  gemma4PleBlockBytes,
  gemma4PleTotalBytes,
  parseGemma4PleIndex,
  readGemma4PleIndex,
} from "../src/gemma/ple-index.ts";
import {
  expectedValue,
  pleComponentOf,
  pleFixture,
  type PleFixtureSpec,
} from "./helpers/ple-fixture.ts";

/**
 * 既定の寸法。`values` と `scales` の 1 行を**同じバイト数**（8 B）にしてあるので、block も
 * 同じ大きさ（2 行 = 16 B）になり、予算の勘定が本数でそのまま読める。
 */
const SPEC: PleFixtureSpec = {
  tokens: 6,
  layers: 2,
  dim: 4,
  // 2 冪（f32 の乗算が厳密 — ADR 0085 決定 4 と同じ性質を合成側でも保つ）。
  embedScale: 4,
  valueRows: 2,
  scaleRows: 2,
};

/** block 1 本ぶん（この索引は 2 表とも同幅）。 */
const BLOCK_BYTES = SPEC.layers * SPEC.dim * (SPEC.valueRows as number);
/** 1 行ぶん（values + scales）— 行キャッシュの単位。 */
const ROW_BYTES = SPEC.layers * SPEC.dim + SPEC.layers * 4;

const fixture = (options: { readonly defer?: boolean } = {}) => pleFixture(SPEC, options);

const pleOf = (
  rig: ReturnType<typeof fixture>,
  maxResidentBytes?: number,
) =>
  createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: SPEC.tokens,
    ...(maxResidentBytes === undefined ? {} : { maxResidentBytes }),
  });

/** `[1, ids, layers, dim]` の f32 を期待値と突き合わせる。 */
const assertRows = (tensor: { readonly data: Float32Array }, ids: readonly number[]): void => {
  const stride = SPEC.layers * SPEC.dim;
  for (const [position, id] of ids.entries()) {
    for (let layer = 0; layer < SPEC.layers; layer += 1) {
      for (let column = 0; column < SPEC.dim; column += 1) {
        assertEquals(
          tensor.data[position * stride + layer * SPEC.dim + column],
          expectedValue(SPEC, id, layer, column),
          `位置 ${position}（token ${id}）/ 層 ${layer} / 列 ${column}`,
        );
      }
    }
  }
};

const tensorOf = async (
  ple: ReturnType<typeof pleOf>,
  ids: readonly number[],
): Promise<{ readonly data: Float32Array }> => {
  const value = await ple.gather(ids);
  assertEquals(value.dtype, "f32");
  assert("data" in value && value.data instanceof Float32Array);
  assertEquals(value.shape, [1, ids.length, SPEC.layers, SPEC.dim]);
  return { data: value.data };
};

// ---- 索引の受理集合 --------------------------------------------------------

Deno.test("parseGemma4PleIndex: 壊れた索引を黙って読まない", async (t) => {
  const raw = fixture().document;

  await t.step("正常系（陰性対照 — 常に落ちる門になっていない）", () => {
    assertEquals(parseGemma4PleIndex(raw).tokens, SPEC.tokens);
  });

  await t.step("旧 sidecar の版は読まない（両読みしない）", () => {
    for (const schema of [1, 2, 4]) {
      assertThrows(
        () => parseGemma4PleIndex({ ...raw, schema }),
        Error,
        `ple_index.schema ${schema} が 3 でない`,
      );
    }
  });

  await t.step("未知キー（綴り違いが黙って既定へ縮退しない）", () => {
    assertThrows(() => parseGemma4PleIndex({ ...raw, shards: [] }), Error, "未知キー 'shards'");
  });

  await t.step("storage は i8 / i4 / i2", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...raw, storage: "i3" }),
      Error,
      "ple_index.storage i3 が i8 / i4 / i2 でない",
    );
  });

  await t.step("storage の欠落を i8 と読まない（schema 3 では必須）", () => {
    const { storage: _, ...withoutStorage } = raw;
    assertThrows(
      () => parseGemma4PleIndex(withoutStorage),
      Error,
      "ple_index.storage undefined が i8 / i4 / i2 でない",
    );
  });

  await t.step("embedScale は正の有限数（0 は行が全部 0 になる）", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...raw, embedScale: 0 }),
      Error,
      "ple_index.embedScale 0 が正の有限数でない",
    );
  });

  await t.step("rowBytes は宣言（層数 / 次元 / 格納）から決まる値と一致する", () => {
    const values = raw.values as { rowBytes: number; blocks: unknown[] };
    assertThrows(
      () => parseGemma4PleIndex({ ...raw, values: { ...values, rowBytes: values.rowBytes + 4 } }),
      Error,
      `ple_index.values.rowBytes ${values.rowBytes + 4} が宣言から決まる ${values.rowBytes} と違う`,
    );
  });

  await t.step("block 名の重複（同じ実体を 2 つの範囲が名乗る）", () => {
    assertThrows(
      () =>
        parseGemma4PleIndex({
          ...raw,
          tokens: 4,
          values: {
            rowBytes: SPEC.layers * SPEC.dim,
            blocks: [
              { asset: "ple.values.0", start: 0, stop: 2 },
              { asset: "ple.values.0", start: 2, stop: 4 },
            ],
          },
          scales: {
            rowBytes: SPEC.layers * 4,
            blocks: [{ asset: "ple.scales.0", start: 0, stop: 4 }],
          },
        }),
      Error,
      ".asset 'ple.values.0' が重複している",
    );
  });

  await t.step("範囲の非連続（引けない id が黙って生まれる）", () => {
    assertThrows(
      () =>
        parseGemma4PleIndex({
          ...raw,
          tokens: 5,
          values: {
            rowBytes: SPEC.layers * SPEC.dim,
            blocks: [
              { asset: "ple.values.0", start: 0, stop: 2 },
              { asset: "ple.values.1", start: 3, stop: 5 },
            ],
          },
          scales: {
            rowBytes: SPEC.layers * 4,
            blocks: [{ asset: "ple.scales.0", start: 0, stop: 5 }],
          },
        }),
      Error,
      ".start 3 が直前の block の末尾 2 と連続しない",
    );
  });

  await t.step("合計行数の不一致（索引だけ焼き直した組み合わせ）", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...raw, tokens: SPEC.tokens + 1 }),
      Error,
      `block の合計 ${SPEC.tokens} 行が tokens ${SPEC.tokens + 1} と違う`,
    );
  });

  await t.step("2 表は独立に分割できる（block 境界が揃っている必要は無い）", () => {
    const index = pleFixture({ ...SPEC, valueRows: 2, scaleRows: 3 }).index;
    assertEquals(index.values.blocks.length, 3);
    assertEquals(index.scales.blocks.length, 2);
  });
});

// ---- 索引と容器の資産の突合（admission の席）-------------------------------

Deno.test("readGemma4PleIndex: 容器の資産から索引を読み、指し先を全件列挙で突合する", async (t) => {
  const rig = fixture();

  await t.step("正常系（陰性対照）", async () => {
    const index = await readGemma4PleIndex("test", pleComponentOf(rig));
    assertEquals(index, rig.index);
  });

  await t.step("索引そのものが無い", async () => {
    await assertRejects(
      () => readGemma4PleIndex("test", pleComponentOf(rig, { ple_index: undefined })),
      Error,
      "容器が資産 'ple_index' を宣言していない",
    );
  });

  await t.step("索引の役割が違う", async () => {
    await assertRejects(
      () => readGemma4PleIndex("test", pleComponentOf(rig, { ple_index: { role: "rope-base" } })),
      Error,
      "資産 'ple_index' の役割が 'rope-base'",
    );
  });

  await t.step("指し先の block が容器に無い / 役割が違う / 長さが違う — 全件列挙", async () => {
    const error = await assertRejects(
      () =>
        readGemma4PleIndex(
          "test",
          pleComponentOf(rig, {
            "ple.values.0": undefined,
            "ple.values.1": { role: "ple-scales" },
            "ple.scales.2": { length: 1 },
          }),
        ),
      Error,
      "PLE の索引と容器の資産が食い違う（3 件）",
    );
    assert(error.message.includes("ple.values.0: 容器が宣言していない"), error.message);
    assert(error.message.includes("ple.values.1: 役割 'ple-scales'"), error.message);
    assert(error.message.includes("ple.scales.2: 論理長 1"), error.message);
  });

  await t.step("容器にあって索引が指していない block（1 本も読まないまま動く形）", () => {
    assertThrows(
      () =>
        assertGemma4PleAssets("test", rig.index, {
          assets: {
            ...pleComponentOf(rig).assets,
            "ple.values.99": "ple-values",
          },
          asset: pleComponentOf(rig).asset,
        }),
      Error,
      "ple.values.99: 役割 'ple-values' の資産だが索引が指していない",
    );
  });
});

// ---- 行の翻訳（block 境界と block 跨ぎ）------------------------------------

Deno.test("Gemma4Ple.gather: token → (block, 行 offset) の翻訳が境界で正しい", async () => {
  const rig = fixture();
  const ple = pleOf(rig, 0);
  // 3 本の block の境界をまたぐ列（先頭・末尾・跨ぎ）。
  const ids = [0, 1, 2, 3, 4, 5];
  assertRows(await tensorOf(ple, ids), ids);

  // 引いた区間が「その block の中の (id - start) 行目」ちょうどであること。恒真でない見方は
  // 区間の実測しかない（値の一致だけなら、別 block を読んでも同じ式で合ってしまう）。
  const rowBytes = { values: rig.index.values.rowBytes, scales: rig.index.scales.rowBytes };
  const expected = ids.flatMap((id) => [
    {
      asset: `ple.values.${Math.floor(id / 2)}`,
      offset: (id % 2) * rowBytes.values,
      length: rowBytes.values,
    },
    {
      asset: `ple.scales.${Math.floor(id / 2)}`,
      offset: (id % 2) * rowBytes.scales,
      length: rowBytes.scales,
    },
  ]);
  assertEquals(
    [...rig.reads].sort((a, b) => a.asset.localeCompare(b.asset) || a.offset - b.offset),
    expected.sort((a, b) => a.asset.localeCompare(b.asset) || a.offset - b.offset),
  );
});

Deno.test("Gemma4Ple.gather: 2 表の block 境界がずれていても同じ行を引く", async () => {
  // `values` は 2 行 / `scales` は 3 行で切る（実資産では 1 行の byte 数が違うので普通に起きる）。
  const spec: PleFixtureSpec = { ...SPEC, valueRows: 2, scaleRows: 3 };
  const rig = pleFixture(spec);
  const ple = createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: spec.tokens,
    maxResidentBytes: 0,
  });
  const ids = [2, 3];
  const tensor = await ple.gather(ids);
  assert("data" in tensor && tensor.data instanceof Float32Array);
  const stride = spec.layers * spec.dim;
  for (const [position, id] of ids.entries()) {
    for (let layer = 0; layer < spec.layers; layer += 1) {
      for (let column = 0; column < spec.dim; column += 1) {
        assertEquals(
          tensor.data[position * stride + layer * spec.dim + column],
          expectedValue(spec, id, layer, column),
        );
      }
    }
  }
  // token 2 は values の block 1（行 0）と scales の block 0（行 2）— 表ごとに別の block。
  assert(
    rig.reads.some((read) => read.asset === "ple.values.1" && read.offset === 0),
    JSON.stringify(rig.reads),
  );
  assert(
    rig.reads.some((read) =>
      read.asset === "ple.scales.0" && read.offset === 2 * rig.index.scales.rowBytes
    ),
    JSON.stringify(rig.reads),
  );
});

Deno.test("Gemma4Ple.gather: 同じ id が並ぶ列でも各行は単発 gather とビット一致する", async () => {
  const rig = fixture();
  const ple = pleOf(rig, 0);
  const single = await tensorOf(ple, [3]);
  const repeated = await tensorOf(ple, [3, 3, 3]);
  const stride = SPEC.layers * SPEC.dim;
  for (let position = 0; position < 3; position += 1) {
    assertEquals(
      [...new Uint32Array(repeated.data.buffer, position * stride * 4, stride)],
      [...new Uint32Array(single.data.buffer)],
      `位置 ${position} が単発 gather とビット一致しない`,
    );
  }
});

// ---- 方針表（全量読み ↔ 行読み）-------------------------------------------

Deno.test("Gemma4Ple.gather: 1 行だけの gather は block 全量を読まない（decode 1 token）", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  assertRows(await tensorOf(ple, [0]), [0]);
  const stats = ple.stats();
  assertEquals(stats.loads, 0, "1 行なのに block 全量を読んでいる");
  assertEquals(stats.rowReads, 2, "行読みは values / scales の 2 区間");
  assertEquals(stats.resident, 0);
  assertEquals(
    rig.reads.map((read) => read.length),
    [rig.index.values.rowBytes, rig.index.scales.rowBytes],
  );
});

Deno.test("Gemma4Ple.gather: block の全行を触る gather は全量読みへ倒れる", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  // block は 2 行なので、2 行触れば下限（= min(32, 行数)）に届く。
  assertRows(await tensorOf(ple, [0, 1]), [0, 1]);
  const stats = ple.stats();
  assertEquals(stats.loads, 2, "values / scales の block を 1 本ずつ全量で読む");
  assertEquals(stats.rowReads, 0);
  assertEquals(stats.resident, 2);
  assertEquals(stats.residentBytes, 2 * BLOCK_BYTES);
  assertEquals(
    rig.reads.map((read) => read.offset),
    [0, 0],
    "全量読みなのに途中から読んでいる",
  );
});

Deno.test("Gemma4Ple.gather: 行読みと全量読みの値はビット同一", async () => {
  const ids = [0, 1];
  const rows = await tensorOf(pleOf(fixture(), 0), ids);
  const blocks = await tensorOf(pleOf(fixture()), ids);
  assertEquals(
    [...new Uint32Array(rows.data.buffer)],
    [...new Uint32Array(blocks.data.buffer)],
    "行読みの値が全量読みとビット一致しない",
  );
  assertRows(rows, ids);
});

Deno.test("Gemma4Ple.gather: 全量読みは予算に追い出し無しで載るときだけ", async () => {
  const rig = fixture();
  // 予算は block 1 本ぶん。values が載った時点で scales は載らない（同じ gather の中で
  // 自分が載せたものを追い出さない）。
  const ple = pleOf(rig, BLOCK_BYTES);
  assertRows(await tensorOf(ple, [0, 1]), [0, 1]);
  const stats = ple.stats();
  assertEquals(stats.loads, 1, "予算 1 本ぶんなのに 2 本読んでいる");
  assertEquals(stats.resident, 1);
  assertEquals(stats.rowReads, 2, "載らなかった表は行読みへ倒れる");
});

Deno.test("Gemma4Ple.gather: 32 行未満でも block が細ければ全量へ倒れる（端数 block）", async () => {
  // 64 行の block 1 本（下限は 32 行）— 31 行なら行読み・32 行なら全量。
  const wide: PleFixtureSpec = { tokens: 64, layers: 1, dim: 4, embedScale: 4 };
  const ids = Array.from({ length: 31 }, (_row, index) => index);
  const narrow = pleFixture(wide);
  const rowPle = createGemma4Ple({
    index: narrow.index,
    openBlock: narrow.openBlock,
    vocabSize: wide.tokens,
    maxResidentBytes: gemma4PleTotalBytes(narrow.index),
  });
  await rowPle.gather(ids);
  assertEquals(rowPle.stats().loads, 0, "31 行で全量読みへ倒れている（下限は 32 行）");
  assertEquals(rowPle.stats().rowReads, 62);

  const full = pleFixture(wide);
  const fullPle = createGemma4Ple({
    index: full.index,
    openBlock: full.openBlock,
    vocabSize: wide.tokens,
    maxResidentBytes: gemma4PleTotalBytes(full.index),
  });
  await fullPle.gather([...ids, 31]);
  assertEquals(fullPle.stats().loads, 2, "32 行なのに行読みのまま");
  assertEquals(fullPle.stats().rowReads, 0);
});

Deno.test("Gemma4Ple.gather: 行読みの同時発行は上限を超えない（fd を枯らさない）", async () => {
  // 1 block 1 行（= 常に行読み）の索引を 128 行ぶん引く。
  const spec: PleFixtureSpec = { tokens: 128, layers: 1, dim: 4, embedScale: 4, valueRows: 1 };
  const rig = pleFixture(spec, { defer: true });
  const ple = createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: spec.tokens,
    maxResidentBytes: 0,
  });
  await ple.gather(Array.from({ length: 128 }, (_row, index) => index));
  assertEquals(ple.stats().rowReads, 256);
  assert(rig.peak.inFlight <= 16, `同時発行のピーク ${rig.peak.inFlight} が上限 16 を超えた`);
  assert(rig.peak.inFlight > 1, "そもそも重ねて発行していない（上限の門が空振り）");
});

// ---- 行キャッシュ ----------------------------------------------------------

Deno.test("Gemma4Ple.gather: 行読みで引いた行は予算の空きぶん再利用される", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  await ple.gather([0]);
  const first = rig.reads.length;
  assertEquals(ple.stats().residentBytes, ROW_BYTES, "行キャッシュが予算に計上されていない");
  await ple.gather([0]);
  assertEquals(rig.reads.length, first, "行キャッシュに当たらず読み直している");
  assertEquals(ple.stats().rowReads, 2);
});

Deno.test("Gemma4Ple.gather: 予算 0 は行も保持しない（毎回読み直す）", async () => {
  const rig = fixture();
  const ple = pleOf(rig, 0);
  await ple.gather([0]);
  await ple.gather([0]);
  assertEquals(ple.stats().residentBytes, 0);
  assertEquals(ple.stats().rowReads, 4, "予算 0 なのに行を保持している");
});

// ---- 予算と LRU ------------------------------------------------------------

Deno.test("Gemma4Ple: 常駐バイトは索引だけから決まる（block 幅に依らない意味）", () => {
  const rig = fixture();
  assertEquals(gemma4PleBlockBytes(rig.index.values, rig.index.values.blocks[0]), BLOCK_BYTES);
  assertEquals(gemma4PleTotalBytes(rig.index), 6 * BLOCK_BYTES);
  // 既定は「最大 block 2 本ぶん」— 幅が変わっても意味（どの 2 本でも収まる）が保たれる。
  assertEquals(defaultGemma4PleResidentBytes(rig.index), 2 * BLOCK_BYTES);
  const wide = pleFixture({ ...SPEC, valueRows: 3, scaleRows: 3 }).index;
  assertEquals(defaultGemma4PleResidentBytes(wide), 2 * 3 * SPEC.layers * SPEC.dim);
});

Deno.test("Gemma4Ple: 追い出しは LRU（前の gather が載せた block から落ちる）", async () => {
  const rig = fixture();
  const ple = pleOf(rig, 2 * BLOCK_BYTES);
  await ple.gather([0, 1]);
  assertEquals(ple.stats().resident, 2);
  // 別の block 対を全量で載せると、前の対が落ちる（予算は 2 本ぶん）。
  await ple.gather([2, 3]);
  assertEquals(ple.stats().resident, 2);
  assertEquals(ple.stats().residentBytes, 2 * BLOCK_BYTES);
  const loads = ple.stats().loads;
  // 落ちたのは古い方 — 引き直すと読みが増える。
  await ple.gather([0, 1]);
  assertEquals(ple.stats().loads, loads + 2, "追い出された block が読み直されていない");
  // 直前に触った block は生きている。
  const after = ple.stats().loads;
  await ple.gather([0, 1]);
  assertEquals(ple.stats().loads, after, "常駐している block を読み直している");
});

Deno.test("Gemma4Ple.gather: 1 回の gather は自分の hit を追い出さない", async () => {
  const rig = fixture();
  const ple = pleOf(rig, 2 * BLOCK_BYTES);
  await ple.gather([0, 1]);
  const loads = ple.stats().loads;
  // hit（block 対 0）と miss（block 対 1）を同じ gather で触る。予算は 2 本ぶんしかないので、
  // miss を全量で載せると hit が落ちる — 方針表は hit のぶんを先に数えるのでそうならない。
  const ids = [0, 1, 2, 3];
  assertRows(await tensorOf(ple, ids), ids);
  assertEquals(ple.stats().loads, loads, "hit のはずの block を読み直している");
  assertEquals(ple.stats().resident, 2);
  assertEquals(ple.stats().rowReads, 2 * 2, "載らなかった miss は行読みへ倒れる");
});

Deno.test("Gemma4Ple: block 1 本すら載らない予算は構築時に fail loudly", () => {
  const rig = fixture();
  const error = assertThrows(
    () => pleOf(rig, BLOCK_BYTES - 1),
    ModelInputError,
    `maxResidentBytes ${BLOCK_BYTES - 1} が PLE の block 1 本ぶん ${BLOCK_BYTES} バイトに満たない`,
  );
  assert(error instanceof ModelInputError, "呼び手の指定なので入力起因（ADR 0107 決定 2）");
  // 0 は「常駐させない」という正当な指定（陰性対照）。
  assertEquals(pleOf(rig, 0).stats().residentBytes, 0);
  assertThrows(() => pleOf(rig, -1), ModelInputError, "が 0 以上の整数でない");
});

// ---- id 空間の相互照合と値域 -----------------------------------------------

Deno.test("createGemma4Ple: vocab の相互照合は読み口に触る前に落ちる", () => {
  const rig = fixture();
  const error = assertThrows(
    () =>
      createGemma4Ple({
        index: rig.index,
        openBlock: rig.openBlock,
        vocabSize: SPEC.tokens + 1,
      }),
    Error,
    `PLE の索引の行数 ${SPEC.tokens} が主 embedding の vocab 行数 ${SPEC.tokens + 1} と違う`,
  );
  assertEquals(rig.opens.length, 0, "照合の前に block を開きに行っている");
  // 焼いた組み合わせの齟齬なので、同じ関数の予算の門と違って入力起因では**ない**
  // （呼び手が `maxResidentPleBytes` をどう直しても直らない — ADR 0107 決定 2）。
  assert(!(error instanceof ModelInputError));
});

Deno.test("Gemma4Ple.gather: id の値域と空列は fail loudly（別 token の有効な行を引かせない）", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  for (const bad of [SPEC.tokens, -1, 1.5]) {
    await assertRejects(
      () => ple.gather([bad]),
      Error,
      `token id[0] ${bad} が PLE の索引の 0..${SPEC.tokens - 1} の外`,
    );
  }
  await assertRejects(() => ple.gather([]), Error, "PLE gather の token 列が空");
  assertEquals(rig.opens.length, 0, "値域の門より先に block を開きに行っている");
});

// ---- 中断と解放 ------------------------------------------------------------

Deno.test("Gemma4Ple.gather: 中断済みの signal では block を 1 本も開かない", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  const controller = new AbortController();
  controller.abort(new Error("test abort"));
  await assertRejects(
    () => ple.gather([0], { signal: controller.signal }),
    Error,
    "test abort",
  );
  assertEquals(rig.opens.length, 0);
});

Deno.test("Gemma4Ple.gather: 走行中の中断は段の境目で効き、読めた行は残さない", async () => {
  // 1 block 1 行の索引（= 常に行読み）。1 本目の読みが決着した後の境目で中断が効く。
  const spec: PleFixtureSpec = { tokens: 64, layers: 1, dim: 4, embedScale: 4, valueRows: 1 };
  const rig = pleFixture(spec, { defer: true });
  const ple = createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: spec.tokens,
    maxResidentBytes: 0,
  });
  const controller = new AbortController();
  const pending = ple.gather(
    Array.from({ length: 64 }, (_row, index) => index),
    { signal: controller.signal },
  );
  controller.abort(new Error("test abort"));
  await assertRejects(() => pending, Error, "test abort");
  assert(rig.reads.length < 128, `中断後も全区間を読み切っている（${rig.reads.length} 本）`);
  assert(rig.reads.length > 0, "そもそも 1 本も読んでいない（中断の門が空振り）");
});

Deno.test("Gemma4Ple: dispose で常駐が空になり、以後の gather は fail loudly", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  await ple.gather([0, 1]);
  assert(ple.stats().residentBytes > 0);
  ple.dispose();
  assertEquals(ple.stats(), { loads: 2, rowReads: 0, resident: 0, residentBytes: 0 });
  await assertRejects(() => ple.gather([0]), Error, "dispose 済みの索引は引けない");
  // 冪等。
  ple.dispose();
  assertEquals(ple.stats().residentBytes, 0);
});

Deno.test("Gemma4Ple: 先行読みが dispose 後に完了しても常駐を復活させない", async () => {
  const rig = fixture({ defer: true });
  const ple = pleOf(rig);
  const pending = ple.gather([0, 1]);
  ple.dispose();
  await pending;
  assertEquals(ple.stats().resident, 0);
  assertEquals(ple.stats().residentBytes, 0);
});
