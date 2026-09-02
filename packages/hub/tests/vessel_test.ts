// 逐次面の器の使い回し（`FileReadOptions.into` / `DirectoryAdapter.readFileInto`）。
// 縛るのは 3 点: ①器を持つ取得元では yield される bytes が全 shard で**同じ buffer** の prefix view
// ②器を持たない取得元は従来どおり毎回新しい tight view（器は確保されない）③器経由でも size 違反は
// 同じ IntegrityError で落ちる（アダプターは判定せず実長を返すだけ）。
import { assert, assertEquals, assertNotStrictEquals, assertRejects } from "@std/assert";
import {
  type FileRef,
  IntegrityError,
  loadManifest,
  localDirectory,
  resolveFiles,
  streamAssets,
} from "../mod.ts";
import { buildLocalDist, memoryDirectory, SHARD_PATHS } from "./helpers/local.ts";
import { MemoryCacheStorage, payloadFor } from "./helpers/mock.ts";

const LABEL = "./models/karume-test";

const shardRefs = async (files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>, vessel: boolean) => {
  const directory = memoryDirectory(files, { vessel });
  const loaded = await loadManifest(localDirectory(directory.adapter, { label: LABEL }), {
    caches: new MemoryCacheStorage(),
  });
  const resolved = resolveFiles(loaded.manifest);
  const refs = SHARD_PATHS.map((_path, index): FileRef => resolved[`net[${index}]`]);
  return { directory, loaded, refs };
};

Deno.test("streamAssets: 器を持つ取得元では全 shard が同じ buffer の prefix view で届く", async () => {
  const dist = await buildLocalDist();
  const { loaded, refs } = await shardRefs(dist.files, true);
  const buffers = new Set<ArrayBuffer>();
  for await (const asset of streamAssets(loaded, refs)) {
    // 中身は取得元の実体そのもの（器へ読んだ後の view が正しい長さを指している）。
    assertEquals(asset.bytes, payloadFor(asset.id));
    assertEquals(asset.bytes.byteOffset, 0);
    buffers.add(asset.bytes.buffer);
  }
  assertEquals(buffers.size, 1, "器が使い回されていない（shard ごとに別 buffer）");
  const [vessel] = buffers;
  const largest = refs.reduce((max, ref) => Math.max(max, ref.size), 0);
  assertEquals(vessel.byteLength, largest, "器の長さが最大 shard 長でない");
});

Deno.test("streamAssets: 器を持たない取得元は従来どおり毎回新しい tight view", async () => {
  const dist = await buildLocalDist();
  const { loaded, refs } = await shardRefs(dist.files, false);
  let previous: ArrayBuffer | undefined;
  for await (const asset of streamAssets(loaded, refs)) {
    assertEquals(asset.bytes.byteLength, asset.bytes.buffer.byteLength, "tight view でない");
    if (previous !== undefined) assertNotStrictEquals(asset.bytes.buffer, previous);
    previous = asset.bytes.buffer;
  }
});

Deno.test("streamAssets: 器経由でも size 不一致は IntegrityError（アダプターは実長を返すだけ）", async () => {
  const dist = await buildLocalDist();
  const tampered = new Map(dist.files);
  // 途中で切れたコピー（短い）と、器に収まらない長いコピーの両方を見る。
  const [first, second] = SHARD_PATHS;
  tampered.set(first, payloadFor(`${first}:truncated`).subarray(0, 8) as Uint8Array<ArrayBuffer>);
  const { loaded, refs } = await shardRefs(tampered, true);
  const short = await assertRejects(
    async () => {
      for await (const _asset of streamAssets(loaded, refs)) { /* 1 本目で落ちる */ }
    },
    IntegrityError,
  );
  assertEquals(short.path, first);
  const oversize = new Map(dist.files);
  const original = payloadFor(second);
  const longer = new Uint8Array(new ArrayBuffer(original.byteLength + 1));
  longer.set(original);
  oversize.set(second, longer);
  const grown = await shardRefs(oversize, true);
  const long = await assertRejects(
    async () => {
      for await (const _asset of streamAssets(grown.loaded, grown.refs)) { /* 2 本目で落ちる */ }
    },
    IntegrityError,
  );
  assertEquals(long.path, second);
  assert(long.message.includes(String(original.byteLength + 1)), long.message);
});
