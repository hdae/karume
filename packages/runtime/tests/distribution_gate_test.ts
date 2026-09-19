// ADR 0005「全ケース SKIP は明示 FAIL」の**配布形ミラー**版の門番（実重み系列版は
// tests/assets_gate_test.ts・GPU 版は tests/gpu_gate_test.ts）。
//
// 配布形ミラー（`models/karume-<family>/`）を根にする検査 — QAT の公開入口 e2e
// （packages/models/tests/e2e_gemma4_qat_test.ts）・quant 解決の e2e
// （同 e2e_gemma4_quant_test.ts）・融合ヒット数の配布形節（tests/assets_fusion_counts_test.ts）
// — は、いずれも `karume.json` の有無で `ignore` し `console.warn` だけを残して緑になる。
// `outputs/series/` だけ持つ機（新しい作業機・worktree を別ホストへ持ち出した場合）では
// assets_gate_test.ts が緑のまま QAT / quant の実資産検査が丸ごと消えるので、ここで 1 本落とす。
//
// この門番自身は資産がある環境では**通る（緑の 1 件として見える）**。ignore にすると
// 「門番が効いているのか、門番ごと消えているのか」が区別できなくなるため（他の 2 門番と同じ理由）。

import { assert } from "@std/assert";

/**
 * 「配布形ミラー無しでの全 SKIP」を明示的に許可する opt-out（`KARUME_ALLOW_NO_ASSETS` と同形）。
 *
 * MUST: 既定は fail loudly。配布形を持たない作業機や CI で通すには、意図表明としてこの環境変数を
 * 要求する（実重み系列側の opt-out とは別に持つ — 片方だけ持つ機が実在する）。
 */
const ALLOW_NO_DISTRIBUTION = Deno.env.get("KARUME_ALLOW_NO_DISTRIBUTION") === "1";

/** 配布形ミラーの根（上記 3 本の検査が `karume.json` を読む先と同じ URL の組み立て方）。 */
const DISTRIBUTIONS = ["karume-gemma4", "karume-gemma4-qat"] as const;

/**
 * その系列の manifest が置かれているか。
 *
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、門番自身が
 * 環境の壊れを資産の不在として報告する。見るのは `karume.json` がファイルであることだけで、
 * 中身（models 欄・quant 席）は各検査の担当。
 */
const manifestPresent = (series: string): boolean => {
  try {
    return Deno.statSync(new URL(`../../../models/${series}/karume.json`, import.meta.url)).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

Deno.test({
  name: "配布門番: 配布形ミラーが欠けた全 SKIP は明示 FAIL（ADR 0005）",
  // opt-out は「配布形ミラー無しを承知で通す」意図表明のときだけ。既定では ignore しない。
  ignore: ALLOW_NO_DISTRIBUTION,
  fn: () => {
    const missing = DISTRIBUTIONS.filter((series) => !manifestPresent(series));
    assert(
      missing.length === 0,
      `models/ に配布形ミラーの karume.json が無い系列がある: ${missing.join(", ")}。` +
        "配布形ミラーが無い機では QAT / quant の実資産 e2e が無音 SKIP するので、" +
        "ADR 0005 によりこれは FAIL として扱う。" +
        "tools/export-recipes の dist.py --pipeline gemma4 / gemma4-qat で作成すること。" +
        "配布形の無い環境で意図的に通すには KARUME_ALLOW_NO_DISTRIBUTION=1 を設定すること。",
    );
  },
});
