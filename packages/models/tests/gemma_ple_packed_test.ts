/** 固定 packed PLE と独立 Torch CPU fixture の突合（ADR 0097）。 */
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import {
  createGemma4Ple,
  type Gemma4PleReadOptions,
  gemma4PleShardBytes,
  parseGemma4PleIndex,
} from "../src/gemma/ple.ts";
for (const dtype of ["i2", "i4"] as const) {
  Deno.test(`PLE ${dtype}: 全量・行読込・常駐の結果が独立CPU参照とビット一致する`, async () => {
    const dir = new URL(`./fixtures/gemma4-ple-packed/${dtype}/`, import.meta.url);
    const raw = JSON.parse(await Deno.readTextFile(new URL("ple.json", dir)));
    const index = parseGemma4PleIndex(raw);
    const oracle = parseSafetensors(
      (await Deno.readFile(new URL("oracle.safetensors", dir))).buffer,
    );
    const iv = oracle.tensors.get("ids")!, ev = oracle.tensors.get("expected")!;
    const ids = Array.from(
      new Int32Array(oracle.buffer, iv.byteOffset, iv.byteLength / 4),
    );
    const expected = new Uint32Array(
      oracle.buffer,
      ev.byteOffset,
      ev.byteLength / 4,
    );
    const factor = dtype === "i2" ? 4 : 2;
    assertEquals(
      gemma4PleShardBytes(index, index.shards[0]),
      3 * 3 * (32 / factor + 4),
    );
    for (const mode of ["full", "seek", "resident", "cached"] as const) {
      let all = 0, range = 0;
      const ple = createGemma4Ple({
        index,
        vocabSize: index.tokens,
        maxResidentBytes: mode === "resident" || mode === "cached" ? 100000 : 0,
        openShard: async (name) => {
          const bytes = await Deno.readFile(new URL(name, dir));
          return {
            bytes: bytes.byteLength,
            readAll: (options?: Gemma4PleReadOptions) => {
              options?.signal?.throwIfAborted();
              all++;
              return Promise.resolve(bytes.slice().buffer);
            },
            ...(mode === "seek" || mode === "cached"
              ? {
                range: {
                  cost: "seek" as const,
                  read: (
                    offset: number,
                    length: number,
                    options?: Gemma4PleReadOptions,
                  ) => {
                    options?.signal?.throwIfAborted();
                    range++;
                    return Promise.resolve(bytes.slice(offset, offset + length).buffer);
                  },
                },
              }
              : {}),
          };
        },
      });
      const value = await ple.gather(ids);
      assert(value.dtype === "f32");
      assertEquals(new Uint32Array(value.data.buffer), expected);
      const loads = all, reads = range;
      const repeat = await ple.gather(ids);
      assert(repeat.dtype === "f32");
      assertEquals(new Uint32Array(repeat.data.buffer), expected);
      if (mode === "resident") {
        assertEquals(all, loads);
        assertEquals(ple.stats().resident, 3);
        assertEquals(
          ple.stats().residentBytes,
          index.shards.reduce((sum, shard) => sum + gemma4PleShardBytes(index, shard), 0),
        );
      }
      if (mode === "seek") {
        assertEquals(all, 0);
        assert(range > 0);
      }
      if (mode === "cached") {
        assertEquals(all, 0);
        assert(reads > 0);
        assertEquals(range, reads, "反復で行を読み直している");
        assertEquals(ple.stats().rowReads, new Set(ids).size);
        assertEquals(
          ple.stats().residentBytes,
          new Set(ids).size * index.layers * (index.dim / factor + 4),
        );
      }
      ple.dispose();
      assertEquals(ple.stats().residentBytes, 0);
      await assertRejects(() => ple.gather(ids));
    }
    const mismatched = createGemma4Ple({
      index: { ...index, storage: dtype === "i2" ? "i4" : "i2" },
      vocabSize: index.tokens,
      openShard: async (name) => {
        const bytes = await Deno.readFile(new URL(name, dir));
        return { bytes: bytes.byteLength, readAll: () => Promise.resolve(bytes.buffer) };
      },
    });
    try {
      await assertRejects(() => mismatched.gather(ids), Error, "storage");
    } finally {
      mismatched.dispose();
    }
    for (
      const invalid of [
        { storage: undefined },
        { storage: "i8" },
        { dim: 31 },
        { schema: 3 },
        { schema: 1 },
      ]
    ) {
      assertThrows(() => parseGemma4PleIndex({ ...raw, ...invalid }));
    }
  });
}
