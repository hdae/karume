/**
 * 検証結果の保存（`outputs/verify/<環境キー>/<日付>_<系列>/`）。
 *
 * 参照値（追跡が要る・環境ごとの行）と結果（追跡外・走らせるたびに増える）は性格が違うので
 * 席を分ける。ここは**消して安全**な側 — `rm -rf outputs/verify` で常に作り直せる
 * （docs/assets-layout.md の outputs 根と同じ流儀で、`bench`（ベンチ・評価）とは別の根）。
 *
 * 置くのは 2 種類だけ:
 *
 * - `results.json` — ケースごとの決着（一致 / 不一致 / 参照値を作った）と、それを採った環境・
 *   チェックアウト。席を作った時点で `cases` 空の「走行中」として先に書き、`record` のたびに
 *   丸ごと書き直す。記録に届く前に落ちた回でも、前回の走行の決着が居座ることはない
 *   （実物だけ今回のバイトに入れ替わった席に、前回の PASS が残るのが最悪の形）。
 * - 実物（PNG / WAV） — **成功・失敗を問わず毎回**。不一致のときだけ残す形だと、一致した
 *   ときの実物が手元に無く、次に割れたときの A/B が採れない。
 *
 * 同じ日・同じ系列を 2 度走らせたら**最後の走行が残る**（日付までで席を分け、走行ごとには
 * 分けない — 分けると「最新はどれか」を人が数えることになる）。
 *
 * 席の日付は**UTC**（走行を始めた時刻の `toISOString()` の日付部）。ローカル日付とはずれる —
 * JST の機では 09:00 より前の走行が前日の席に入る。
 */

import type { Tolerance } from "../../src/reference/allclose.ts";
import { ENVIRONMENT, type Environment } from "./environment.ts";

/** 結果の根（`outputs/verify/`）。 */
const DEFAULT_ROOT = new URL("../../../../outputs/verify/", import.meta.url);

/** リポジトリ根（`git` を回す作業ディレクトリ）。 */
const REPO_ROOT = new URL("../../../../", import.meta.url);

/** ケース 1 件の決着。 */
export type ResultStatus = "pass" | "fail" | "written" | "rewritten";

/**
 * 許容差判定の実測 1 本（golden 系列のケースが**出力ごとに**積む）。
 *
 * 許容差の判定は落ちたときにしか数値を見せないので、**合格した回の差**はどこにも残らない。
 * 「この機でどれだけ差が出ているか」を毎回残しておくと、帯を緩めるかどうかの判断材料が
 * 割れる前から手元に揃う（帯そのものの値・判定・assert はこの記録とは無関係）。
 *
 * MUST: 派生値（帯に対する比など）は持たない — 導けるものは読む側（`tools/verify-diff`）が
 * 導く。2 か所に同じ数の別表現が乗ると、片方だけ直った形が作れてしまう。
 * NOTE: 非有限（NaN / ±Inf）は `JSON.stringify` が `null` にする（読む側は null を受ける）。
 */
export type Measurement = {
  /** グラフの出力名。 */
  readonly output: string;
  readonly maxAbs: number;
  readonly maxRel: number;
  /** 受理に使った帯。 */
  readonly tolerance: Tolerance;
  /** どの段で受理したか（fail のときは最後に測った段）。 */
  readonly stage: "karume" | "spec";
};

/** `results.json` の `cases` に積む 1 件。 */
export type ResultEntry = {
  readonly id: string;
  readonly status: ResultStatus;
  /** 突き合わせた参照値（参照値を作った回は持たない）。 */
  readonly expected?: string;
  /** 実測の sha256（sha を採らないケースは持たない）。 */
  readonly actual?: string;
  /** 実物のファイル名（{@link Results.dir} からの相対）。 */
  readonly artifact?: string;
  /**
   * ケース 1 件の所要時間。**計測範囲は系列ごとの実装に依る**（Session を 1 本だけ組んで
   * 複数ケースを回す系列では、構築と解放はどのケースにも入らない）。系列をまたいだ時間の
   * 比較には使わない — 同じ系列の走行どうしを見るための欄である。
   */
  readonly elapsedMs: number;
  readonly note?: string;
  /**
   * 許容差判定の実測。{@link runRecordedCase} 経由のケースは常に持つ（本体が測る前に投げた回は
   * 空配列）。欄ごと持たないのは {@link Results.record} を直に呼ぶケースだけ。
   */
  readonly measurements?: readonly Measurement[];
};

/** 系列 1 本ぶんの結果の置き場。 */
export type Results = {
  readonly dir: URL;
  /** 実物 1 件の置き場（ディレクトリは初回の呼びで作る）。 */
  artifact(name: string): URL;
  /** 1 件を積んで `results.json` を書き直す。 */
  record(entry: ResultEntry): Promise<void>;
};

type Checkout = { readonly sha: string; readonly dirty: boolean };

/**
 * `git` の出力 1 本。**`git` が無い環境では `undefined`**（結果は残す — チェックアウトが
 * 分からないことと、結果を取らないことは別）。NotFound 以外は伝播させる。
 */
const git = async (...args: string[]): Promise<string | undefined> => {
  try {
    const { success, stdout } = await new Deno.Command("git", {
      args,
      cwd: REPO_ROOT,
      stdout: "piped",
      stderr: "null",
    }).output();
    return success ? new TextDecoder().decode(stdout) : undefined;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
    throw cause;
  }
};

const readCheckout = async (): Promise<Checkout | undefined> => {
  const sha = await git("rev-parse", "HEAD");
  if (sha === undefined) return undefined;
  const status = await git("status", "--porcelain");
  if (status === undefined) return undefined;
  return { sha: sha.trim(), dirty: status.trim() !== "" };
};

/** プロセスで 1 回だけ確定する（同じ走行の全ケースが同じ時刻・同じチェックアウトを名乗る）。 */
const STARTED_AT = new Date().toISOString();
/** 席の日付（UTC — モジュール doc）。 */
const TODAY = STARTED_AT.slice(0, 10);
const CHECKOUT: Checkout | undefined = await readCheckout();

/**
 * 系列 1 本の結果の置き場を開く。
 *
 * MUST: 置き場の決定も作成も**使うときまで遅らせる** — このモジュールは GPU 無しの機でも
 * 読み込まれる（テストは ignore で SKIP される）ので、開いた時点で環境キーを要求すると
 * 「SKIP されるはずのファイル」がモジュール評価で落ちる。
 */
export const openResults = (
  family: string,
  options: { readonly root?: URL; readonly environment?: Environment } = {},
): Results => {
  const environment = options.environment ?? ENVIRONMENT;
  const root = options.root ?? DEFAULT_ROOT;
  const cases: ResultEntry[] = [];
  let created = false;
  const directory = (): URL => {
    const { key } = environment;
    if (key === undefined) {
      throw new Error(
        `この環境には環境キーが無い（GPU アダプタが取れていない）ため ${family} の結果を ` +
          "どの席へ書くか決められない",
      );
    }
    return new URL(`${key}/${TODAY}_${family}/`, root);
  };
  const documentText = (): string =>
    `${
      JSON.stringify(
        {
          schema: 1,
          family,
          environment,
          checkout: CHECKOUT ?? null,
          startedAt: STARTED_AT,
          cases,
        },
        undefined,
        2,
      )
    }\n`;
  // 置き場は**使うときに**作る（資産が無くて全 SKIP の機に空ディレクトリを残さない）。
  const ensureDir = (): URL => {
    const dir = directory();
    if (!created) {
      Deno.mkdirSync(dir, { recursive: true });
      // 「走行中」を先に置く（`cases` 空・この走行の startedAt）。記録の前に例外で抜けても、
      // 同じ席に居る前回の決着が今回の実物と組で読まれることが無くなる。
      Deno.writeTextFileSync(new URL("results.json", dir), documentText());
      created = true;
    }
    return dir;
  };
  return {
    get dir(): URL {
      return directory();
    },
    artifact: (name: string): URL => {
      if (name.includes("/")) {
        throw new Error(`実物の名前にディレクトリ区切りを含められない: '${name}'`);
      }
      return new URL(name, ensureDir());
    },
    record: async (entry: ResultEntry): Promise<void> => {
      const dir = ensureDir();
      cases.push(entry);
      await Deno.writeTextFile(new URL("results.json", dir), documentText());
    },
  };
};

/**
 * 失敗した回の決着を残す（**記録の失敗で元の検証例外を置き換えない**）。
 *
 * 呼ぶのは検証例外を掴んだ catch の中だけである。そこで {@link Results.record} を裸で待つと、
 * 席が作れない・書けないといった I/O 障害がそのまま伝播して、呼び手が抱えている元の例外
 * —— 何が壊れたのかを言う唯一の診断 —— が I/O 例外に置き換わる。記録できなかったことは
 * `console.error` で言い、元の例外は呼び手がそのまま投げる。
 *
 * NOTE: 成功側の `record` はこれを通さない（書けないなら fail loudly が筋で、消す診断も無い）。
 */
export const recordFailure = async (results: Results, entry: ResultEntry): Promise<void> => {
  try {
    await results.record(entry);
  } catch (cause) {
    console.error(
      `[karume] ケース '${entry.id}' の決着を results.json へ記録できなかった: ` +
        `${cause instanceof Error ? cause.message : String(cause)}`,
    );
  }
};

/** {@link runRecordedCase} の本体へ渡す、ケース 1 件ぶんの記録の口。 */
export type RecordedCase = {
  /** 出力ごとの実測（合格した回も残す — 判定には使わない）。本体が積み、決着に載る。 */
  readonly measurements: Measurement[];
};

/**
 * 本体が例外なしに戻ったときの決着。戻り値を省いた本体は `pass`。
 *
 * `fail` を返すのは「全出力を測り終えてから落とす」ケース — 決着（note つき）を残したあとで
 * 呼び手が assert する。
 */
export type CaseVerdict = {
  readonly status: "pass" | "fail";
  readonly note?: string;
};

/** {@link runRecordedCase} が記録するケースの名乗りと、失敗した回の扱い。 */
export type RecordedCaseSpec = {
  /** `results.json` の `cases[].id`。 */
  readonly id: string;
  /** 投げた回の note（省けば note を持たない）。 */
  readonly failureNote?: (cause: unknown) => string;
  /**
   * 本体が投げた回にだけ、記録の前に呼ぶ。複数ケースを 1 本のテストで回す系列が「この失敗は
   * ケースの席に残った」を知る口 — 成功側の記録が書けずに投げた回は呼ばない（その回は
   * ケースの決着が残っていない）。
   */
  readonly onFailure?: () => void;
};

/**
 * ケース 1 件を回し、決着を必ず `results.json` へ残す（実重み golden の各ケースの共通の枠）。
 *
 * - 本体が戻った: その決着（既定 `pass`）を {@link Results.record} で積む。書けなければ
 *   fail loudly（{@link recordFailure} の NOTE と同じ理由）。
 * - 本体が投げた: `fail` を {@link recordFailure} で積んでから**同じ例外を投げ直す**。決着の
 *   無いまま抜けると、この席には同じ日の前回の走行の決着が居座る。
 *
 * どちらの経路でも欄の並び（id / status / elapsedMs / note / measurements）は同じで、
 * 所要時間は呼んだ時点から記録の直前までを測る。
 */
export const runRecordedCase = async (
  results: Results,
  recordedCase: RecordedCaseSpec,
  body: (recorded: RecordedCase) => Promise<CaseVerdict | undefined>,
): Promise<void> => {
  const { id, failureNote, onFailure } = recordedCase;
  const startedAt = performance.now();
  const measurements: Measurement[] = [];
  const elapsedMs = (): number => Math.round(performance.now() - startedAt);
  let verdict: CaseVerdict | undefined;
  try {
    verdict = await body({ measurements });
  } catch (cause) {
    onFailure?.();
    await recordFailure(results, {
      id,
      status: "fail",
      elapsedMs: elapsedMs(),
      ...(failureNote === undefined ? {} : { note: failureNote(cause) }),
      measurements,
    });
    throw cause;
  }
  await results.record({
    id,
    status: verdict?.status ?? "pass",
    elapsedMs: elapsedMs(),
    ...(verdict?.note === undefined ? {} : { note: verdict.note }),
    measurements,
  });
};
