/**
 * PLE sidecar の常駐予算を**索引から**導く（e2e が渡す `maxResidentPleBytes` の出所）。
 *
 * MUST: 予算をテスト側の定数で書かない。「1 本 ≈253MiB だから 768MiB で 3 本」のような数は
 * 資産世代（書き手の shard 上限）が変わった瞬間に別の本数を意味し、テストの意図
 * （範囲をまたぐ会話で shard を読み直さない）が例外なしに崩れる — 本数ではなくバイトで
 * 受ける理由（ADR 0085 追記 2026-09-02）がそのままテスト側にも効く。
 *
 * 全量常駐（{@link allResidentBytes}）を渡せば読み直しはどの世代でも起きない。先例は
 * `e2e_gemma4_product_test.ts` で、その勘定をここへ移した（同じ式を e2e ごとに写さない）。
 */

import { MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import {
  type Gemma4PleIndex,
  gemma4PleShardBytes,
  parseGemma4PleIndex,
} from "../../src/gemma/ple.ts";

/** sidecar 全量を常駐させるバイト数（= 読み直しゼロ）。索引だけで決まる。 */
export const allResidentBytes = (index: Gemma4PleIndex): number =>
  index.shards.reduce((sum, shard) => sum + gemma4PleShardBytes(index, shard), 0);

/**
 * 資産の PLE 索引（`ple.json`）を読んで全量常駐の予算を出す。
 *
 * 資産が無い環境では呼ばない（呼び手は SKIP 判定の後で使う）— 読めない索引は握り潰さず
 * 伝播させる。
 */
export const allResidentPleBytesAt = (indexUrl: URL): number =>
  allResidentBytes(parseGemma4PleIndex(JSON.parse(Deno.readTextFileSync(indexUrl))));

/**
 * 配布形ミラーの全量常駐予算（索引の置き場は manifest の `ple_index` 資産から引く — ミラーの
 * ディレクトリ構造をテスト側に写さない）。
 */
export const allResidentPleBytesOfMirror = (mirrorDir: URL): number => {
  const manifest = parseManifest(Deno.readTextFileSync(new URL(MANIFEST_FILENAME, mirrorDir)));
  const entry = manifest.models[manifest.defaultModel];
  const index = entry.assets["ple_index"];
  if (index === undefined) {
    throw new Error(`配布形ミラー ${mirrorDir} の manifest に 'ple_index' 資産が無い`);
  }
  return allResidentPleBytesAt(new URL(index.path, mirrorDir));
};
