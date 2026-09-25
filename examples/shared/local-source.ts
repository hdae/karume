/**
 * デモの `--source` を **`fromPretrained` の取得元**へ写す 1 本（ローカルの配布形なら
 * `denoDirectory`、それ以外は HF リポジトリ名の文字列）。
 *
 * ## なぜ HTTP を挟まないのか
 *
 * 手元の配布形は取得元ハンドル（`@karume/hub/deno` の `denoDirectory`）で直に読める。容器の
 * 読みは part 単位（part 0 の 2 文書だけで `prepareContainer` まで進み、重みの block は part
 * ごとに取り出す）なので、RAM に載るのは常に「今の part」だけ — 使い捨ての HF 形サーバ
 * （`local-dist-server.ts`）を挟んでいた頃の利点は全部残り、**永続キャッシュへの複製もポートも
 * 要らなくなった**。あちらには実 DL 経路を通す消費者（疑似 HF の HTTP 疎通そのものを見る門
 * `packages/models/tests/e2e_gemma4_pretrained_test.ts` と、cold / warm を測る
 * `tools/ram-peak/measure.ts`）が残っている。
 *
 * ## 越境参照は明示 mapping だけ
 *
 * 配布形は自リポの外を指せる（`FileRef` の `repo` / `revision` — ADR 0038 §7。例:
 * `karume-anima-extra` の text stack は `hdae/karume-anima` の 1 commit を指す）。
 * `localDirectory` は**隣接する同名ディレクトリを推測しない**ので、越境先は `--source-map
 * owner/name=<パス>` で名指しする。未指定のまま越境を踏めば hub がそのまま案内を出して落ちる
 * （取り違えたバイト列を黙って読ませないための線 — `packages/hub/src/sources/local.ts`）。
 */

import type { DistributionSource } from "../../packages/hub/mod.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import { isLocalDist } from "./local-assets.ts";

/** `owner/name=<パス>` の綴り（デモの `--source-map` はこれを繰り返し受ける）。 */
const parseCrossRepo = (
  spellings: readonly string[],
): Record<string, DistributionSource> => {
  const mapping: Record<string, DistributionSource> = {};
  for (const spelled of spellings) {
    const cut = spelled.indexOf("=");
    // MUST: 綴りの崩れは落とす。黙って無視すると「mapping を渡したのに越境で落ちる」になる。
    if (cut <= 0 || cut === spelled.length - 1) {
      throw new Error(`--source-map ${spelled} が owner/name=<パス> の形でない`);
    }
    const repo = spelled.slice(0, cut);
    if (Object.hasOwn(mapping, repo)) throw new Error(`--source-map の repo ${repo} が重複`);
    mapping[repo] = denoDirectory(spelled.slice(cut + 1));
  }
  return mapping;
};

/**
 * `--source`（と繰り返し可能な `--source-map`）から取得元を作る。
 *
 * `karume.json` を持つディレクトリなら `denoDirectory`、それ以外は HF リポジトリ名としてその
 * まま返す（どちらも `fromPretrained` の 1 本）。
 */
export const distributionSource = async (
  source: string,
  sourceMaps: readonly string[] = [],
): Promise<string | DistributionSource> => {
  if (!await isLocalDist(source)) {
    // MUST: HF リポ名に越境 mapping は渡せない（HF 取得元は宣言された (repo, revision) を
    // 自分で開く）。黙って捨てると「効かないノブ」が静かに残る。
    if (sourceMaps.length > 0) {
      throw new Error(
        `--source-map はローカルの配布形にだけ効く（--source ${source} は HF リポ名）`,
      );
    }
    return source;
  }
  // 空の mapping は「宣言が無い」と同義（越境を踏んだ時点で hub が案内付きで落ちる）。
  return denoDirectory(source, { crossRepo: parseCrossRepo(sourceMaps) });
};

/** 既定のローカルミラーを優先し、不在のときだけ検証済み公開 revision を使う。 */
export const localOrPinnedSource = async (
  directory: string,
  pinned: { readonly repo: string; readonly revision: string },
): Promise<{
  readonly from: DistributionSource | { readonly repo: string; readonly revision: string };
  readonly label: string;
}> => {
  try {
    await Deno.stat(directory);
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
    return { from: pinned, label: `${pinned.repo}@${pinned.revision}` };
  }
  // 存在するが未完成のミラーは、ネットワーク取得で覆い隠さない。
  if (!await isLocalDist(directory)) {
    throw new Error(`${directory} に karume.json が無い（ローカル配布形を確認してください）`);
  }
  return { from: denoDirectory(directory), label: directory };
};
