// PLE sidecar のホスト gather（`src/gemma/ple.ts` — ADR 0085）の寿命と中断の門。GPU も実資産も
// 要らない（合成 sidecar を `writeSafetensors` で組む）。
//
// ここで縛るのは 2 つ:
//
// - **解放**: `dispose()` で常駐（shard 1 本 758MB 級 × 既定 2 本）が空になり、以後の gather は
//   fail loudly。口が無いと「パイプラインを dispose しても 1.5GiB がホスト RAM に残る」形が
//   復活するが、解放は例外にならないので stats の実数で見るしかない。
// - **中断の透過**: gather の `signal` が shard の読み口まで降りる（best-effort）。降りないと
//   「停止を押しても 758MB の読みが終わるまで返らない」。中断された読みは常駐に残らず、
//   同じ id をもう一度引けば読み直す（拒否済み Promise を掴み続けない）。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { createGemma4Ple, type Gemma4PleIndex } from "../src/gemma/ple.ts";
import { type DumpTensor, writeSafetensors } from "./helpers/safetensors-write.ts";

const LAYERS = 2;
const DIM = 2;
const ROWS_PER_SHARD = 2;
const TOKENS = 4;
/** 2 冪（f32 の乗算が厳密 — ADR 0085 決定 4 と同じ性質を合成側でも保つ）。 */
const EMBED_SCALE = 4;
const SHARD_FILES = ["ple-00001-of-00002.safetensors", "ple-00002-of-00002.safetensors"] as const;

const INDEX: Gemma4PleIndex = {
  tokens: TOKENS,
  layers: LAYERS,
  dim: DIM,
  embedScale: EMBED_SCALE,
  shards: [
    { file: SHARD_FILES[0], start: 0, stop: ROWS_PER_SHARD },
    { file: SHARD_FILES[1], start: ROWS_PER_SHARD, stop: TOKENS },
  ],
};

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
    residentShards: 2,
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
  assertEquals(ple.stats(), { loads: 2, resident: 2 });

  ple.dispose();
  // 解放は例外にならないので、実数で見るしかない（ここが 2 のままなら shard は返っていない）。
  assertEquals(ple.stats().resident, 0);
  await assertRejects(() => ple.gather([0]), Error, "dispose 済み");
  // 拒否は「引けない」だけで、読み直しも起きない。
  assertEquals(reader.calls.length, 2);

  // 冪等（2 度目の dispose も、その後の stats も落ちない）。
  ple.dispose();
  assertEquals(ple.stats(), { loads: 2, resident: 0 });
});

Deno.test("Gemma4Ple: gather の signal は shard の読み口へ降りる（best-effort）", async () => {
  const reader = fakeReader();
  const ple = createGemma4Ple({ index: INDEX, readShard: reader.readShard, vocabSize: TOKENS });
  const controller = new AbortController();

  await ple.gather([0], { signal: controller.signal });
  assertEquals(reader.calls.length, 1);
  assert(
    reader.calls[0].signal === controller.signal,
    "gather の signal が読み口へ降りていない（758MB の読みが中断の届かない区間になる）",
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
  assertEquals(ple.stats(), { loads: 1, resident: 0 }, "拒否された取得を常駐させている");

  // 同じ id をもう一度引けば読み直す（拒否済み Promise を掴み続けない）。
  const tensor = await ple.gather([0]);
  assert("data" in tensor);
  assertEquals(tensor.data[0], expectedValue(0, 0, 0));
  assertEquals(ple.stats(), { loads: 2, resident: 1 });
});
