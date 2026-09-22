/**
 * PLE の常駐予算を**索引から**導く（e2e が渡す `maxResidentPleBytes` の出所）。
 *
 * MUST: 予算をテスト側の定数で書かない。「1 本 ≈32MiB だから 96MiB で 3 本」のような数は
 * 資産世代（書き手の block 上限）が変わった瞬間に別の本数を意味し、テストの意図
 * （範囲をまたぐ会話で block を読み直さない）が例外なしに崩れる — 本数ではなくバイトで
 * 受ける理由（ADR 0085 追記 2026-09-02）がそのままテスト側にも効く。
 *
 * 全量常駐（{@link allResidentBytes}）を渡せば読み直しはどの世代でも起きない。
 */

import { gemma4PleTotalBytes } from "../../src/gemma/ple-index.ts";
import type { Gemma4PleIndex } from "../../src/gemma/ple-index.ts";
import { type MirrorChoice, openGemma4Ple } from "./gemma-mirror.ts";

/** 索引が指す block 全部ぶん（= 読み直しゼロ）。索引だけで決まる。 */
export const allResidentBytes = (index: Gemma4PleIndex): number => gemma4PleTotalBytes(index);

/**
 * 配布形ミラーの全量常駐予算（索引は `model` 容器の資産なので、容器を開いて読む — ミラーの
 * ディレクトリ構造をテスト側に写さない）。
 *
 * 資産が無い環境では呼ばない（呼び手は SKIP 判定の後で使う）— 読めない索引は握り潰さず
 * 伝播させる。
 */
export const allResidentPleBytesOfMirror = async (
  mirror: URL,
  choice: MirrorChoice = {},
): Promise<number> => allResidentBytes((await openGemma4Ple(mirror, choice)).index);
