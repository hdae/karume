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
 * HF の repo 名 1 セグメントの許可文字（hub の `manifest.ts` の越境参照検査と**同じ規則** —
 * どちらも「取得層が URL へ綴り込む生文字列」なので、綴りの受理集合は 1 つに揃える）。
 */
const REPO_SEGMENT_RE = /^[A-Za-z0-9_-][A-Za-z0-9._-]*$/;

/**
 * `repo` が HF の `owner/name` の綴りであることを見る（**取得層より前**）。
 *
 * ローカルのディレクトリパスを渡した呼び出しは、この門が無いと**そのまま HF へ飛ぶ**
 * （`./models/foo` が URL の一部として綴り込まれ、返るのは 401 / 404 — 「取得先が存在しない」
 * という原因の遠い診断になる）。パスにしか見えない綴り（先頭 `./` `../` `/`・末尾 `/`・
 * スラッシュ 2 個以上・空セグメント・許可外の文字）はここで落とす。
 *
 * NOTE: `models/karume-gemma4-e2b` のような**合法な `owner/name` に見えるパス**は救えない
 * （HF の repo 名として完全に正しい綴りで、区別できる情報がこの引数の中に無い）。ここが見るのは
 * 綴りだけで、「実在するか」は取得層の仕事である。
 */
const assertRepoSpelling = (repo: string, where: string): void => {
  const segments = repo.split("/");
  if (segments.length !== 2 || !segments.every((segment) => REPO_SEGMENT_RE.test(segment))) {
    throw new Error(
      `${where}: repo '${repo}' が HF の 'owner/name' でない` +
        `（2 セグメントちょうど・許可文字 [A-Za-z0-9._-]・先頭ドット禁止）`,
    );
  }
};

/**
 * `ref` を {@link HubRepoRef} にする。文字列は `{ repo }` と読む（= `main` 追従）。
 *
 * 綴りは HF の `owner/name` であることまで見る（{@link assertRepoSpelling}）— ローカルの
 * ディレクトリパスを渡した呼び出しを、URL へ綴り込む前に落とす。
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
  assertRepoSpelling(repo, where);
  return typeof ref === "string" ? { repo } : ref;
};
