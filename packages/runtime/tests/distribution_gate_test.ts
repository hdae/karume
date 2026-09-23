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
// 見るのは**有無だけではない**（ADR 0108 段 3 検収②）。`karume.json` はあるが中身が旧 major の
// まま、あるいは宣言された part が 1 本足りない・長さが宣言と違う、という形は上の e2e を
// 「開けない資産」で落とすか、悪くすると**前回の書き出しの残骸**を今回の期待値で読ませる。
// どちらも「資産が無い」とは別の壊れ方なので、門番が manifest を parse して part の実在と
// 長さまで突き合わせる。中身（block の sha256）は読まない — 実バイトの突合は各 e2e が
// `openContainer` の経路で必ず通る（§7 のハッシュ 3 分離）。
//
// この門番自身は資産がある環境では**通る（緑の 1 件として見える）**。ignore にすると
// 「門番が効いているのか、門番ごと消えているのか」が区別できなくなるため（他の 2 門番と同じ理由）。

import { assert, assertEquals } from "@std/assert";
import { parseManifest, resolveSelection } from "@karume/hub";

/**
 * 「配布形ミラー無しでの全 SKIP」を明示的に許可する opt-out（`KARUME_ALLOW_NO_ASSETS` と同形）。
 *
 * MUST: 既定は fail loudly。配布形を持たない作業機や CI で通すには、意図表明としてこの環境変数を
 * 要求する（実重み系列側の opt-out とは別に持つ — 片方だけ持つ機が実在する）。
 */
const ALLOW_NO_DISTRIBUTION = Deno.env.get("KARUME_ALLOW_NO_DISTRIBUTION") === "1";

/** 配布形ミラーの根（上記 3 本の検査が `karume.json` を読む先と同じ URL の組み立て方）。 */
const DISTRIBUTIONS = ["karume-gemma4", "karume-gemma4-qat"] as const;

/** この版の読み手が受け付ける配布 manifest の major（旧版は読まない — ADR 0109 決定 1）。 */
const MANIFEST_FORMAT = "karume/5";

const distributionRoot = (series: string): URL =>
  new URL(`../../../models/${series}/`, import.meta.url);

/**
 * その系列の manifest が置かれているか。
 *
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、門番自身が
 * 環境の壊れを資産の不在として報告する。ここが見るのは `karume.json` がファイルであることだけで、
 * 中身は下の 1 本が見る。
 */
const manifestPresent = (series: string): boolean => {
  try {
    return Deno.statSync(new URL("karume.json", distributionRoot(series))).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

/** ファイル長（不在は `undefined`・それ以外の I/O 異常は伝播させる）。 */
const fileBytes = (url: URL): number | undefined => {
  try {
    const stat = Deno.statSync(url);
    return stat.isFile ? stat.size : undefined;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
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

for (const series of DISTRIBUTIONS) {
  Deno.test({
    name: `配布門番: ${series} の manifest が karume/5 で、既定 quant の part が宣言どおり在る`,
    // 不在は上の 1 本が名指しで落とす（同じ壊れで 2 本赤くしても読み手の仕事は増えない）。
    ignore: !manifestPresent(series),
    fn: async () => {
      const root = distributionRoot(series);
      const manifest = parseManifest(await Deno.readTextFile(new URL("karume.json", root)));
      // parse 自身も major を見るが、`karume/5` を**名指しで**断言しておく — hub が受ける major を
      // 増やした日に、この門番が「旧 major のミラーを据えたまま」を素通しするのを防ぐ。
      assertEquals(manifest.format, MANIFEST_FORMAT, `${series}: manifest の format`);
      const selection = resolveSelection(manifest);
      const broken: string[] = [];
      for (const [component, container] of Object.entries(selection.containers)) {
        for (const [index, part] of container.parts.entries()) {
          const where = `${component}[${index}] ${part.path}`;
          if (part.repo !== undefined) {
            // 越境参照はローカルミラーに実体を持たない（ADR 0038 §7）— 取得層の担当で、
            // ここで「無い」と数えると門番が別の系統の話を報告することになる。
            continue;
          }
          const actual = fileBytes(new URL(part.path, root));
          if (actual === undefined) {
            broken.push(`${where}: 宣言された part が無い`);
          } else if (actual !== part.size) {
            broken.push(`${where}: 長さ ${actual} が宣言の ${part.size} と違う`);
          }
        }
      }
      assertEquals(
        broken,
        [],
        `models/${series}/ の配布形が manifest の宣言と食い違っている` +
          `（model '${selection.model}' / quant '${selection.quant}'）。` +
          "焼き直しの途中や前回の残骸を e2e が読むのを避けるため、ADR 0005 によりこれは FAIL。" +
          "tools/export-recipes の dist.py で作り直すこと。",
      );
    },
  });
}
