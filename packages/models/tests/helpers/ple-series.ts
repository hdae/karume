/**
 * recipe の**系列出力**（`outputs/series/**` の `ple.json` + `ple-NNNNN.safetensors`）を、
 * 容器の資産と同じ面（{@link Gemma4PleHandle}）へ畳むテスト用アダプタ。
 *
 * ## なぜ要るのか
 *
 * 配布形は ADR 0109 決定 4 で PLE を `model` 容器の資産（役割 `ple-values` / `ple-scales` の
 * block 列 + 索引 `ple_index`）へ移したが、**recipe が `krm` を直接書くのは段 3**（同 決定 8）
 * なので、段 2 の系列出力は旧形の sidecar のままである。torch との突合（`ple.probe`）と
 * 素の decode golden はその系列出力に対する門なので、資産を差し替えるのではなく**読み口を
 * 合わせる**。移行 CLI は shard を token 順に連結して行の倍数で切り直すだけなので、値は
 * 移行前後でビット同一であり、断定の力は変わらない。
 *
 * ## 畳み方
 *
 * 旧 shard 1 本は `values [rows, layers, dim]` と `scales [rows, layers]` が連続で並んだ
 * safetensors なので、**そのテンソル領域がそのまま block 1 本**になる（asset 名は
 * `<ファイル名>:values` / `<ファイル名>:scales`）。索引は schema 3 の文書を組み立ててから
 * `parseGemma4PleIndex` に通す — 受理集合（区間の連続性・`rowBytes` の整合）の門を
 * テスト側に写さないためである。
 *
 * NOTE: hub / runtime の**テストの都合**は import しない（`helpers/memory-cache.ts` の規律）。
 * ここが触るのは公開面（`@karume/runtime` の safetensors パーサ）だけである。
 */

import { parseSafetensorsHeader, safetensorsHeaderLength } from "@karume/runtime";
import type { AssetReader, SafetensorsHeader, TensorView } from "@karume/runtime";
import { parseGemma4PleIndex, SCALE_BYTES } from "../../src/gemma/ple-index.ts";
import type { Gemma4PleHandle } from "./gemma-mirror.ts";

/** 旧索引（`ple.json`）のうち、この面が読む欄だけ。 */
type LegacyIndex = {
  readonly storage?: "i2" | "i4";
  readonly tokens: number;
  readonly layers: number;
  readonly dim: number;
  readonly embedScale: number;
  readonly shards: readonly {
    readonly file: string;
    readonly start: number;
    readonly stop: number;
  }[];
};

/** block 1 本の実体（旧 shard の中のテンソル領域）。 */
type Region = {
  readonly url: URL;
  readonly offset: number;
  readonly length: number;
};

const HEADER_LENGTH_BYTES = 8;

const readAt = async (
  url: URL,
  offset: number,
  length: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  const handle = await Deno.open(url, { read: true });
  try {
    await handle.seek(offset, Deno.SeekMode.Start);
    const into = new Uint8Array(new ArrayBuffer(length));
    let filled = 0;
    while (filled < length) {
      const read = await handle.read(into.subarray(filled));
      if (read === null) {
        throw new Error(
          `test: ${url.pathname} が offset ${offset} からの ${length} バイトに足りない`,
        );
      }
      filled += read;
    }
    return into;
  } finally {
    handle.close();
  }
};

/** safetensors のヘッダだけを 2 段で読む。 */
const readHeader = async (url: URL): Promise<SafetensorsHeader> => {
  const { size } = await Deno.stat(url);
  const head = await readAt(url, 0, HEADER_LENGTH_BYTES);
  const headerLength = safetensorsHeaderLength(head);
  const prefix = new Uint8Array(new ArrayBuffer(HEADER_LENGTH_BYTES + headerLength));
  prefix.set(head);
  prefix.set(await readAt(url, HEADER_LENGTH_BYTES, headerLength), HEADER_LENGTH_BYTES);
  return parseSafetensorsHeader(prefix, size);
};

const tensorOf = (header: SafetensorsHeader, name: string, url: URL): TensorView => {
  const view = header.tensors.get(name);
  if (view === undefined) throw new Error(`test: ${url.pathname} にテンソル '${name}' が無い`);
  return view;
};

/**
 * 系列出力の PLE sidecar を開く（索引 + block の読み口）。
 *
 * `root` は `ple.json` と shard が並ぶディレクトリ（recipe の系列出力）。
 */
export const openSeriesPle = async (root: URL): Promise<Gemma4PleHandle> => {
  const legacy = JSON.parse(
    await Deno.readTextFile(new URL("ple.json", root)),
  ) as LegacyIndex;
  const factor = legacy.storage === "i2" ? 4 : legacy.storage === "i4" ? 2 : 1;
  const rowBytes = {
    values: legacy.layers * legacy.dim / factor,
    scales: legacy.layers * SCALE_BYTES,
  } as const;
  const regions = new Map<string, Region>();
  const blocks = { values: [] as unknown[], scales: [] as unknown[] };
  for (const shard of legacy.shards) {
    const url = new URL(shard.file, root);
    const header = await readHeader(url);
    for (const table of ["values", "scales"] as const) {
      const view = tensorOf(header, table, url);
      const asset = `${shard.file}:${table}`;
      regions.set(asset, { url, offset: view.byteOffset, length: view.byteLength });
      blocks[table].push({ asset, start: shard.start, stop: shard.stop });
    }
  }
  const index = parseGemma4PleIndex(
    {
      schema: 3,
      storage: legacy.storage ?? "i8",
      tokens: legacy.tokens,
      layers: legacy.layers,
      dim: legacy.dim,
      embedScale: legacy.embedScale,
      values: { rowBytes: rowBytes.values, blocks: blocks.values },
      scales: { rowBytes: rowBytes.scales, blocks: blocks.scales },
    },
    `${root.pathname}ple.json`,
  );
  return {
    index,
    openBlock: (asset: string): AssetReader => {
      const region = regions.get(asset);
      if (region === undefined) throw new Error(`test: PLE の block '${asset}' が系列出力に無い`);
      return {
        role: asset.endsWith(":values") ? "ple-values" : "ple-scales",
        length: region.length,
        read: (offset, length) => {
          if (offset < 0 || length < 0 || offset + length > region.length) {
            throw new Error(
              `test: block '${asset}' の区間 [${offset}, ${offset + length}) が長さ` +
                ` ${region.length} の外`,
            );
          }
          return readAt(region.url, region.offset + offset, length);
        },
      };
    },
  };
};
