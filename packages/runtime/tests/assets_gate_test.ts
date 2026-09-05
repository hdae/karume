// ADR 0005「全ケース SKIP は明示 FAIL」の**実資産**版の門番（GPU 版は tests/gpu_gate_test.ts）。
//
// 実重み e2e（birefnet / dacvae / deberta / depth_anything / embeddinggemma / gemma4 / irodori /
// minicpm5 / sbv2 / siglip2 / vowel_detector）は `ignore: !available || !GPU_AVAILABLE` で、
// `available` は系列 root の列挙が空なら偽になる。GPU はあるが実重み資産が 1 つも無い機
// （新しい作業機・worktree を別ホストへ持ち出した場合）では、これらが丸ごと SKIP したまま
// `deno task verify` が緑になる — 「検証していない」を「検証済み」と誤読させる無音の見かけ
// 成功なので、ここで 1 本だけ落とす。
//
// この門番自身は資産がある環境では**通る（緑の 1 件として見える）**。ignore にすると
// 「門番が効いているのか、門番ごと消えているのか」が区別できなくなるため（GPU 版と同じ理由）。
//
// NOTE: 見るのは実重み e2e 11 本が共有する `outputs/series/` の 1 根だけ。融合ヒット数門が
// 使う `models/`（配布形ミラー）は別系統で、そちらの SKIP は本門番の射程外。

import { assert } from "@std/assert";

/**
 * 「実資産無しでの全 SKIP」を明示的に許可する opt-out（`KARUME_ALLOW_NO_GPU` と同形）。
 *
 * MUST: 既定は fail loudly。models パッケージだけを触る作業機や資産を持たない CI で通すには、
 * 意図表明としてこの環境変数を要求する。
 */
const ALLOW_NO_ASSETS = Deno.env.get("KARUME_ALLOW_NO_ASSETS") === "1";

/** 実重み e2e が資産を探す唯一の根（各 e2e の `SERIES_ROOT` / `SERIES_PARENT` の親）。 */
const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);

/**
 * 系列がひとつでも置かれているか。
 *
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、門番自身が
 * 環境の壊れを資産の不在として報告する。数えるのはディレクトリ（と、それへの symlink）だけで、
 * 直下に紛れ込んだ素のファイルは系列として数えない。
 */
const seriesPresent = (): boolean => {
  try {
    for (const entry of Deno.readDirSync(SERIES_PARENT)) {
      if (entry.isDirectory || entry.isSymlink) return true;
    }
    return false;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

Deno.test({
  name: "資産門番: 実重み系列が 1 つも無い全 SKIP は明示 FAIL（ADR 0005）",
  // opt-out は「実資産無しを承知で通す」意図表明のときだけ。既定では ignore しない。
  ignore: ALLOW_NO_ASSETS,
  fn: () => {
    assert(
      seriesPresent(),
      `${SERIES_PARENT.pathname} に実重み系列が 1 つも無く、実重み e2e が全て SKIP された。` +
        "ADR 0005 によりこれは FAIL として扱う（リリース判定は実資産の golden 突合が必須）。" +
        "資産の無い環境で意図的に通すには KARUME_ALLOW_NO_ASSETS=1 を設定すること。",
    );
  },
});
