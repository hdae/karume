/**
 * ディレクトリの実ファイルから PLE shard の読み口を組む（実資産 e2e が `fromAssets` /
 * `createGemma4Ple` へ渡す 1 本）。
 *
 * 費用の型は `"seek"` — 実装は `@karume/hub/deno` の `readFileRange` と同じ形（`Deno.open` →
 * `seek` → 読みループ → `finally` で close）である。3 本の e2e がこの 1 実装を共有するのは、
 * **行読み経路を踏むかどうか**がテストごとに揺れないようにするため（読み口の綴りを写すと、
 * 片方だけ range を落として「従来経路だけ緑」という形が作れてしまう）。
 */

import type { Gemma4PleShardSource } from "../../src/gemma/ple.ts";

/** view が buffer 全体を占めるなら slice しない（PLE shard は 1 本 250MiB 級）。 */
const wholeBuffer = (bytes: Uint8Array<ArrayBuffer>): ArrayBuffer =>
  bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

/**
 * `root` 直下の `file` を読む口を開く（`stat` は 1 度だけ = 宣言 size の代わり）。
 *
 * MUST: 短い戻りを返さない（要求長ちょうどか throw）— 読み口の契約そのもので、緩めると
 * 消費側が 0 埋めの行を正常な値として読む。
 */
export const openPleShardAt = async (root: URL, file: string): Promise<Gemma4PleShardSource> => {
  const url = new URL(file, root);
  const { size } = await Deno.stat(url);
  return {
    bytes: size,
    readAll: async () => wholeBuffer(await Deno.readFile(url)),
    range: {
      cost: "seek",
      read: async (offset, length) => {
        const handle = await Deno.open(url);
        try {
          await handle.seek(offset, Deno.SeekMode.Start);
          const target = new Uint8Array(new ArrayBuffer(length));
          let filled = 0;
          while (filled < length) {
            const read = await handle.read(target.subarray(filled));
            if (read === null) {
              throw new Error(
                `${url.pathname} が offset ${offset} からの ${length} バイトに足りない`,
              );
            }
            filled += read;
          }
          return target.buffer;
        } finally {
          handle.close();
        }
      },
    },
  };
};
