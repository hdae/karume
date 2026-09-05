/**
 * 全ファミリの**公開配布リポ対応表**を 1 つに畳んだ表（ADR 0092）。
 *
 * 家族ごとの表（`<FAMILY>_SOURCES`）が正本で、ここはその和集合を barrel から 1 本で引ける形に
 * するだけ。値も**キーも家族側と同じ**なので、`KARUME_SOURCES["anima"]` と
 * `ANIMA_SOURCES["anima"]` は同一のエントリを指す。
 *
 * 使い道は「公開している配布リポを全部なめる」側（疎通スモーク・キャッシュの事前取得・
 * 一覧の表示）で、1 家族しか使わない消費者はサブパスの家族表を引く方が tree-shaking に効く
 * （barrel を引くと 6 家族の config が繋がる）。
 *
 * MUST: 家族側の表を**そのまま**畳む — ここで綴り直すと、家族側を直しても barrel だけが古い
 * revision を配り続ける形（導出できる値の二重持ち）になる。キーが重なると後勝ちで 1 本が
 * 黙って消えるので、重複が無いことは `tests/sources_test.ts` の門が見る（キーは HF リポ名から
 * 導かれるので、重複は「同じリポを 2 家族が持っている」ときにしか起きない）。
 */

import type { HubRepoRef } from "@karume/hub";

import { ANIMA_SOURCES } from "./anima/config.ts";
import { IRODORI_SOURCES } from "./irodori/config.ts";
import { SBV2_SOURCES } from "./sbv2/config.ts";
import { GEMMA4_SOURCES } from "./gemma/config.ts";
import { SIGLIP2_SOURCES } from "./siglip2/config.ts";
import { DEPTH_ANYTHING_SOURCES } from "./depth-anything/config.ts";
import { BIREFNET_SOURCES } from "./birefnet/config.ts";

export const KARUME_SOURCES = {
  ...ANIMA_SOURCES,
  ...IRODORI_SOURCES,
  ...SBV2_SOURCES,
  ...GEMMA4_SOURCES,
  ...SIGLIP2_SOURCES,
  ...DEPTH_ANYTHING_SOURCES,
  ...BIREFNET_SOURCES,
} as const satisfies Record<string, HubRepoRef>;
