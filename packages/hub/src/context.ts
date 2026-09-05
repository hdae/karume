/**
 * 取得の文脈 — エラーが名乗る「実際に取りに行った先」の組み立て点。
 *
 * MUST: 取得元（`sources/`）はエラーを組み立てない。文脈（取得元の名乗り・利用可能ラベル）を
 * 持つのは共通層だけであり、取得元ごとに組み立てると「存在しない repo を指す診断」や
 * 「越境先ではなくセッションの repo を名乗る診断」が取得元の数だけ再生産される。
 *
 * 取得元ごとに語彙が違う部分は {@link ../source.ts SourceOrigin} 1 つに畳んである — 文言に
 * 出るのは `label`（HF: `repo owner/name @ <SHA>` / ローカル: `ディレクトリ <ラベル>`）で、
 * 構造化欄（`repo` / `revisionSha`）は**持っている取得元だけ**が名乗る。
 */

import {
  type AvailableLabels,
  HubFetchError,
  IntegrityError,
  ManifestFormatError,
} from "./errors.ts";
import { crossRefOf, type FileRef, MANIFEST_FILENAME, MAX_MANIFEST_BYTES } from "./manifest.ts";
import type { LoadedManifest } from "./session.ts";
import type { SizeViolation, SourceOrigin } from "./source.ts";

/** 取得失敗の文言に入る動詞（面によって「取得」と「事前取得」で割れる）。 */
export type FetchVerb = "取得" | "事前取得";

/**
 * 名乗りのうち診断が使う部分。**失敗元（{@link ../source.ts SourceOrigin.integrity}）は
 * 含まない** — 越境参照の名乗りは宣言（repo / revision）から作れるが、実際にバイト列を返した
 * 取得元がローカルか network かは宣言から決まらない（`localDirectory` は越境先にリモートの
 * 取得元を正式に受ける）。失敗元は {@link ../source.ts SizeViolation} を呼ぶ取得元が名乗る。
 */
type OriginNaming = Omit<SourceOrigin, "integrity">;

/** 資産の取得面が使う診断の組み立て（manifest 取得前の面はこの下の関数群を使う）。 */
export type FetchContext = {
  /** 診断の文言に載るセッションの名乗り（{@link ../source.ts SourceOrigin.label}）。 */
  readonly session: string;
  /** 失敗時に提示する「今このリポで選べるもの」の一覧（全 hub エラーの必須欄）。 */
  readonly available: AvailableLabels;
  /**
   * 実際に取りに行った先の名乗り。越境参照は**宣言された側**を名乗る（セッションの取得元を
   * 名乗ると、そこには存在しない path を指す診断になる）。
   */
  readonly originOf: (ref: FileRef) => OriginNaming;
  /** バイト数が manifest の `size` と食い違ったときのエラー。 */
  readonly sizeViolation: (ref: FileRef) => SizeViolation;
  /** 取得元由来の失敗（404・認証・cache I/O・ローカルの欠損等）を文脈付きで包む。 */
  readonly fetchFailure: (ref: FileRef, verb: FetchVerb, cause: unknown) => HubFetchError;
};

/** 名乗りのうち、エラーの構造化欄に載せる部分だけを取り出す（持たない欄は載せない）。 */
const identityOf = (
  origin: OriginNaming,
): { readonly repo?: string; readonly revisionSha?: string } => ({
  ...(origin.repo === undefined ? {} : { repo: origin.repo }),
  ...(origin.revisionSha === undefined ? {} : { revisionSha: origin.revisionSha }),
});

/**
 * 越境参照が宣言している座標の名乗り。**取得元には問い合わせない** — 越境先の取得元を引くと
 * 未 mapping の fail loudly（`sources/local.ts`）がエラーの組み立て中に飛び、真の失敗理由を
 * 覆い隠す。宣言そのもの（ADR 0038 §7 の repo / revision）は manifest が持っている一次情報。
 */
const crossOrigin = (repo: string, revision: string): OriginNaming => ({
  label: `repo ${repo} @ ${revision}`,
  repo,
  revisionSha: revision,
});

export const createFetchContext = (
  loaded: LoadedManifest,
  session: SourceOrigin,
): FetchContext => {
  const available: AvailableLabels = loaded.manifest.available;
  const originOf = (ref: FileRef): OriginNaming => {
    const cross = crossRefOf(ref);
    return cross === undefined ? session : crossOrigin(cross.repo, cross.revision);
  };
  return {
    session: session.label,
    available,
    originOf,
    sizeViolation: (ref) => (actual, where, integrity) => {
      const origin = originOf(ref);
      return new IntegrityError(
        `${ref.path}: ${where} が manifest の size と食い違う` +
          `（期待 ${ref.size} / 実際 ${actual} — ${origin.label}）`,
        {
          ...identityOf(origin),
          path: ref.path,
          expected: String(ref.size),
          actual: String(actual),
          // 失敗元は**実際に読んだ取得元**が名乗る（越境参照ではセッションと違う）。
          source: integrity,
          available,
        },
      );
    },
    fetchFailure: (ref, verb, cause) => {
      const origin = originOf(ref);
      return new HubFetchError(`${ref.path} の${verb}に失敗した（${origin.label}）`, {
        ...identityOf(origin),
        path: ref.path,
        available,
        cause,
      });
    },
  };
};

/**
 * 世代識別子の解決に失敗した。manifest をまだ取得できていないので利用可能ラベルは空
 * （`errors.ts` 冒頭が挙げる 2 つの正当な空のうち片方）。
 *
 * 名乗りは**世代解決前**のもの（{@link ../source.ts SourceDriver.origin} — HF なら要求した
 * ref まで含む）。世代の概念を持たない取得元はここで落ちようがないので、案内は可変 ref 向け。
 */
export const revisionResolutionFailure = (
  origin: SourceOrigin,
  cause: unknown,
): HubFetchError =>
  new HubFetchError(
    `世代の解決に失敗した（${origin.label}）。可変 ref はオフラインで` +
      `起動できない — revision に commit SHA を渡すと解決要求そのものが発生しない`,
    { ...identityOf(origin), cause },
  );

/** `karume.json` が上限バイト数を超えた。 */
export const manifestOversize = (origin: SourceOrigin): SizeViolation => (actual, where) =>
  new ManifestFormatError(
    `manifest: ${MANIFEST_FILENAME} が上限 ${MAX_MANIFEST_BYTES} バイトを超えた` +
      `（${where} = ${actual} — ${origin.label}）`,
  );

/** `karume.json` そのものの取得に失敗した。 */
export const manifestFetchFailure = (origin: SourceOrigin, cause: unknown): HubFetchError =>
  new HubFetchError(`${MANIFEST_FILENAME} の取得に失敗した（${origin.label}）`, {
    ...identityOf(origin),
    path: MANIFEST_FILENAME,
    cause,
  });
