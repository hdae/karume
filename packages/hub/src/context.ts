/**
 * 取得の文脈 — エラーが名乗る「実際に取りに行った先」の組み立て点。
 *
 * MUST: 取得元（`sources/`）はエラーを組み立てない。文脈（repo / 世代識別子 / 利用可能ラベル）を
 * 持つのは共通層だけであり、取得元ごとに組み立てると「存在しない repo を指す診断」や
 * 「越境先ではなくセッションの repo を名乗る診断」が取得元の数だけ再生産される。
 *
 * NOTE: 現状の文言と {@link ../errors.ts HubFetchError} / {@link ../errors.ts IntegrityError} の
 * 必須欄（`repo` / `revisionSha`）は HF の語彙のまま。世代識別子や repo という概念を持たない
 * 取得元が入るときに何を名乗らせるかは**このファイルだけ**を書き換えれば済む形にしてある。
 */

import {
  type AvailableLabels,
  HubFetchError,
  IntegrityError,
  ManifestFormatError,
} from "./errors.ts";
import { type FileRef, MANIFEST_FILENAME, MAX_MANIFEST_BYTES } from "./manifest.ts";
import type { LoadedManifest } from "./session.ts";
import type { SizeViolation } from "./source.ts";

/** 取得失敗の文言に入る動詞（面によって「取得」と「事前取得」で割れる）。 */
export type FetchVerb = "取得" | "事前取得";

/** 資産の取得面が使う診断の組み立て（manifest 取得前の面はこの下の関数群を使う）。 */
export type FetchContext = {
  /**
   * 診断の文言に載るセッションの名乗り（現状 `repo <owner/name> @ <commit SHA>`）。
   * 世代識別子や repo を持たない取得元が入るときに書き換えるのはここ 1 箇所。
   */
  readonly session: string;
  /** 失敗時に提示する「今このリポで選べるもの」の一覧（全 hub エラーの必須欄）。 */
  readonly available: AvailableLabels;
  /**
   * 実際に取りに行った (repo, 世代識別子)。越境参照は宣言された側を名乗る
   * （セッションの repo を名乗ると、そのリポには存在しない path を指す診断になる）。
   */
  readonly originOf: (ref: FileRef) => { readonly repo: string; readonly revisionSha: string };
  /** バイト数が manifest の `size` と食い違ったときのエラー。 */
  readonly sizeViolation: (ref: FileRef) => SizeViolation;
  /** 取得元由来の失敗（404・認証・cache I/O 等）を文脈付きで包む。 */
  readonly fetchFailure: (ref: FileRef, verb: FetchVerb, cause: unknown) => HubFetchError;
};

export const createFetchContext = (loaded: LoadedManifest): FetchContext => {
  const { repo, revisionSha } = loaded;
  const available: AvailableLabels = loaded.manifest.available;
  const originOf = (ref: FileRef): { readonly repo: string; readonly revisionSha: string } => ({
    repo: ref.repo ?? repo,
    revisionSha: ref.revision ?? revisionSha,
  });
  return {
    session: `repo ${repo} @ ${revisionSha}`,
    available,
    originOf,
    sizeViolation: (ref) => (actual, where) =>
      new IntegrityError(
        `${ref.path}: ${where} が manifest の size と食い違う（期待 ${ref.size} / 実際 ${actual}）`,
        {
          ...originOf(ref),
          path: ref.path,
          expected: String(ref.size),
          actual: String(actual),
          source: "network",
          available,
        },
      ),
    fetchFailure: (ref, verb, cause) => {
      const origin = originOf(ref);
      return new HubFetchError(
        `${ref.path} の${verb}に失敗した（repo ${origin.repo} @ ${origin.revisionSha}）`,
        { ...origin, path: ref.path, available, cause },
      );
    },
  };
};

/**
 * 世代識別子の解決に失敗した。manifest をまだ取得できていないので利用可能ラベルは空
 * （`errors.ts` 冒頭が挙げる 2 つの正当な空のうち片方）。
 */
export const revisionResolutionFailure = (
  repo: string,
  revision: string,
  cause: unknown,
): HubFetchError =>
  new HubFetchError(
    `revision '${revision}' の解決に失敗した（repo ${repo}）。可変 ref はオフラインで` +
      `起動できない — revision に commit SHA を渡すと解決要求そのものが発生しない`,
    { repo, cause },
  );

/** `karume.json` が上限バイト数を超えた。 */
export const manifestOversize =
  (repo: string, revisionSha: string): SizeViolation => (actual, where) =>
    new ManifestFormatError(
      `manifest: ${MANIFEST_FILENAME} が上限 ${MAX_MANIFEST_BYTES} バイトを超えた` +
        `（${where} = ${actual} — repo ${repo} @ ${revisionSha}）`,
    );

/** `karume.json` そのものの取得に失敗した。 */
export const manifestFetchFailure = (
  repo: string,
  revisionSha: string,
  cause: unknown,
): HubFetchError =>
  new HubFetchError(`${MANIFEST_FILENAME} の取得に失敗した（repo ${repo} @ ${revisionSha}）`, {
    repo,
    revisionSha,
    path: MANIFEST_FILENAME,
    cause,
  });
