/**
 * `fromPretrained` の `ref`（文字列 | {@link HubRepoRef}）を hub へ渡す形へ正規化する
 * （**パイプライン非依存の共通処理** — 7 家族の `fromPretrained` が同じ形で使う）。
 *
 * MUST: barrel には出さない。これは引数の綴りを揃える内部機構で、利用者が触る面ではない。
 *
 * ## MUST: 取得元に既定を持たない（ADR 0073 決定 2 の「ref を optional 化」は 0.5.0 で撤回）
 *
 * 既定があると「どのリポから取っているか」を 1 文字も綴らないコードが書け、パッケージ版を
 * 上げた瞬間に取得先が黙って動く（pin 定数の更新はパッチ公開でも起きる）。取得元は呼び出し側の
 * 決定として**必ず綴らせる**。TS では引数必須なので型検査が落ち、型を持たない JS からの
 * 引数なし呼び出しは {@link toRepoRef} が記述例つきで落とす — `undefined` を hub まで滑らせて
 * 「repo が undefined の URL を叩いた」という原因の遠い失敗にしない。
 */

import type { HubRepoRef } from "@karume/hub";

/**
 * `ref` を {@link HubRepoRef} にする。文字列は `{ repo }` と読む（= `main` 追従）。
 *
 * `undefined` を受ける型なのは**呼び出し側の型を緩めるためではない** — `fromPretrained` の
 * `ref` は必須で、`undefined` が来るのは型検査の外（JS）からだけ。ここで落とすべき値を型から
 * 消してしまうと、この門自体を素直に叩けなくなる。
 *
 * @param where 診断の主語（`"AnimaPipeline.fromPretrained"`）。
 * @param current このファミリの `*_CURRENT` 定数の綴り。公開配布リポを持たないファミリでは
 *   省く（存在しない識別子を案内しない）。
 */
export const toRepoRef = (
  ref: string | HubRepoRef | undefined,
  where: string,
  current?: string,
): HubRepoRef => {
  const repo: unknown = typeof ref === "string" ? ref : ref?.repo;
  if (ref === undefined || typeof repo !== "string" || repo.length === 0) {
    throw new Error(
      `${where}: repo が必須（取得元に既定は無い）— ` +
        '{ repo: "owner/name", revision: "<40 桁の commit SHA>" } を渡すか、' +
        (current === undefined
          ? "リポ名の文字列（= main 追従）を渡す"
          : `このパッケージ版が検証した ${current} を渡す`),
    );
  }
  return typeof ref === "string" ? { repo } : ref;
};
