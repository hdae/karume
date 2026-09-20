/**
 * sha256 参照値の**環境別の席**（追跡 JSON）と、その引き当て。
 *
 * ## なぜ定数ではなく環境ごとの行なのか
 *
 * 参照門が主張するのは「この機で数値が 1 ビットも動いていない」ことで、クロスデバイスの
 * ビット同一ではない（docs/limitations.md）。参照値をテストのソースに定数で持つと、GPU を
 * 替えた瞬間に全門が赤になり、どの赤が「退行」でどの赤が「機が違うだけ」なのか区別が付かなく
 * なる。行を {@link Environment.key} で分けると、赤は常に「その機での退行」を意味する。
 *
 * ## 3 つのモード（環境変数 `KARUME_REFERENCE`）
 *
 * | モード | 行が無いケース | 行があるケース |
 * | --- | --- | --- |
 * | 未設定 | 明示 SKIP（{@link References.warnMissing} が作り方を出す） | 比較（不一致は赤） |
 * | `write` | 実測を**追加**して緑 | 比較（**上書きしない**） |
 * | `rewrite` | 実測を追加して緑 | 実測で**上書き**して緑（旧→新を印字） |
 *
 * MUST: 不一致を tolerance で吸収しない。`rewrite` は「何が変わったのかを先に言えるとき」
 * だけの操作で、**他環境の行には決して触らない**（別の機の参照値を巻き添えにしない）。
 *
 * MUST: 「参照が無いので全 SKIP」を無音の緑にしない。{@link registerReferenceGate} が
 * ADR 0005 の門番と同じ形（opt-out つき）で 1 本落とす。
 */

import { assert } from "@std/assert";
import { ENVIRONMENT, type Environment } from "./environment.ts";

/** 参照値を作るモード。未設定（= 比較だけ）は `undefined` で表す。 */
export type ReferenceMode = "write" | "rewrite";

/** fixture の形式版（読めない版は throw する — 黙って空として扱わない）。 */
const SCHEMA = 1;
/** fixture が持つ値の種類（今のところ sha256 だけ）。 */
const KIND = "sha256";

/** `KARUME_REFERENCE` の値を読む。未知の綴りは throw（黙って「未設定」に落とさない）。 */
export const parseReferenceMode = (raw: string | undefined): ReferenceMode | undefined => {
  if (raw === undefined) return undefined;
  if (raw === "write" || raw === "rewrite") return raw;
  throw new Error(
    `KARUME_REFERENCE は 'write' か 'rewrite'（受け取った値: '${raw}'）。` +
      "未設定なら既存の参照値との比較だけを行う",
  );
};

/** この実行の参照モード（モジュール評価時に 1 回だけ確定）。 */
export const REFERENCE_MODE: ReferenceMode | undefined = parseReferenceMode(
  Deno.env.get("KARUME_REFERENCE"),
);

/**
 * 「この環境の参照値が 1 つも無いままの SKIP」を明示的に許可する opt-out
 * （`helpers/gpu.ts` の `ALLOW_NO_GPU` と同型）。
 */
export const ALLOW_NO_REFERENCE: boolean = Deno.env.get("KARUME_ALLOW_NO_REFERENCE") === "1";

/** 1 ケースぶんの突合の決着。 */
export type ReferenceCheck =
  /** 現環境の行と一致した。 */
  | { readonly status: "pass"; readonly expected: string }
  /** 現環境の行と食い違った（呼び手が診断を付けて落とす）。 */
  | { readonly status: "fail"; readonly expected: string }
  /** 行が無かったので実測を追加した（`write` / `rewrite`）。 */
  | { readonly status: "written" }
  /** 行を実測で上書きした（`rewrite`）。 */
  | { readonly status: "rewritten"; readonly previous: string };

/**
 * 突合の決着を結果 JSON の `expected` 欄へ写す。参照値を**作った**回は突き合わせる相手が
 * いないので持たない（焼き直した回は旧い値が入る — 何から何へ動いたかが結果に残る）。
 */
export const expectedOf = (check: ReferenceCheck): string | undefined => {
  if (check.status === "written") return undefined;
  return check.status === "rewritten" ? check.previous : check.expected;
};

/** 参照値を作った / 焼き直した回を 1 行で残す（比較だけの回は何も言わない）。 */
export const announceCheck = (caseId: string, check: ReferenceCheck, actual: string): void => {
  if (check.status === "written") {
    console.log(`[reference] ${caseId}: この環境の参照値として ${actual} を書いた`);
  } else if (check.status === "rewritten") {
    console.log(`[reference] ${caseId}: 参照値を焼き直した ${check.previous} → ${actual}`);
  }
};

/** 参照値 fixture 1 本の読み書き口。 */
export type References = {
  /** この実行の参照モード。 */
  readonly mode: ReferenceMode | undefined;
  /** 現環境の行（無ければ `undefined`）。 */
  lookup(caseId: string): string | undefined;
  /** 現環境の行が 1 つでもあるか（参照門が見る）。 */
  hasCurrent(): boolean;
  /** 参照値が無く、作るモードでもない（= そのケースは明示 SKIP する）。 */
  lacksReference(caseId: string): boolean;
  /** 現環境の行を書き戻す（モードに依らず呼んだぶんだけ書く — 判定は {@link check}）。 */
  record(caseId: string, sha256: string): "written" | "rewritten";
  /** モードに従って突き合わせ、必要なら行を書く。 */
  check(caseId: string, sha256: string): ReferenceCheck;
  /** 参照値が無いケースを 1 度だけ警告する（登録時に呼ぶ）。 */
  warnMissing(caseIds: readonly string[]): void;
  /** fixture の置き場（診断の文面に出す）。 */
  readonly fixtureUrl: URL;
};

type ReferenceDocument = {
  readonly schema: number;
  readonly kind: string;
  readonly cases: Record<string, Record<string, string>>;
};

/** fixture を読む。無い・壊れている・版が違うは全て throw（空として扱わない）。 */
const readDocument = (fixtureUrl: URL): ReferenceDocument => {
  let text: string;
  try {
    text = Deno.readTextFileSync(fixtureUrl);
  } catch (cause) {
    throw new Error(`参照値の fixture が読めない: ${fixtureUrl.pathname}`, { cause });
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    throw new Error(`参照値の fixture が JSON として壊れている: ${fixtureUrl.pathname}`, { cause });
  }
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error(`参照値の fixture がオブジェクトでない: ${fixtureUrl.pathname}`);
  }
  const document = parsed as Partial<ReferenceDocument>;
  if (document.schema !== SCHEMA) {
    throw new Error(
      `参照値の fixture の schema が ${String(document.schema)}（期待 ${SCHEMA}）: ` +
        fixtureUrl.pathname,
    );
  }
  if (document.kind !== KIND) {
    throw new Error(
      `参照値の fixture の kind が '${String(document.kind)}'（期待 '${KIND}'）: ` +
        fixtureUrl.pathname,
    );
  }
  const { cases } = document;
  if (typeof cases !== "object" || cases === null) {
    throw new Error(`参照値の fixture に cases が無い: ${fixtureUrl.pathname}`);
  }
  for (const caseId of Object.keys(cases)) {
    const rows = cases[caseId];
    if (typeof rows !== "object" || rows === null) {
      throw new Error(`参照値の fixture のケース '${caseId}' が行の表でない`);
    }
    for (const key of Object.keys(rows)) {
      if (typeof rows[key] !== "string") {
        throw new Error(`参照値の fixture のケース '${caseId}' / 環境 '${key}' が文字列でない`);
      }
    }
  }
  return { schema: SCHEMA, kind: KIND, cases };
};

/**
 * 書き出しを安定させる（ケース名・環境キーとも辞書順・2 スペース・末尾改行）。差分が
 * 「実際に変わった行」だけになるのは、複数環境の行が 1 ファイルに同居する席では必須条件。
 */
const serialize = (cases: Record<string, Record<string, string>>): string => {
  const sorted: Record<string, Record<string, string>> = {};
  for (const caseId of Object.keys(cases).sort()) {
    const rows = cases[caseId];
    const row: Record<string, string> = {};
    for (const key of Object.keys(rows).sort()) row[key] = rows[key];
    sorted[caseId] = row;
  }
  return `${JSON.stringify({ schema: SCHEMA, kind: KIND, cases: sorted }, undefined, 2)}\n`;
};

/**
 * 参照値 fixture を開く。
 *
 * `environment` / `mode` を引数で受けるのは単体テストのため（実 GPU も環境変数も要らずに
 * モードの分岐を検査できる）。実行時の呼びは既定のまま使う。
 */
export const openReferences = (
  fixtureUrl: URL,
  environment: Environment = ENVIRONMENT,
  mode: ReferenceMode | undefined = REFERENCE_MODE,
): References => {
  const { cases } = readDocument(fixtureUrl);
  const { key } = environment;
  const requireKey = (): string => {
    if (key === undefined) {
      throw new Error(
        `この環境には環境キーが無い（GPU アダプタが取れていない）ため ${fixtureUrl.pathname} ` +
          "の行を書けない",
      );
    }
    return key;
  };
  const lookup = (caseId: string): string | undefined =>
    key === undefined ? undefined : cases[caseId]?.[key];
  const record = (caseId: string, sha256: string): "written" | "rewritten" => {
    const current = requireKey();
    const rows = cases[caseId] ?? {};
    const previous = rows[current];
    // 他環境の行はここで写し取られるだけ（触らない）。
    cases[caseId] = { ...rows, [current]: sha256 };
    Deno.writeTextFileSync(fixtureUrl, serialize(cases));
    return previous === undefined ? "written" : "rewritten";
  };
  return {
    mode,
    fixtureUrl,
    lookup,
    hasCurrent: (): boolean =>
      key !== undefined && Object.keys(cases).some((caseId) => cases[caseId][key] !== undefined),
    lacksReference: (caseId: string): boolean => mode === undefined && lookup(caseId) === undefined,
    record,
    check: (caseId: string, sha256: string): ReferenceCheck => {
      const expected = lookup(caseId);
      if (expected === undefined) {
        if (mode === undefined) {
          throw new Error(
            `ケース '${caseId}' にこの環境（${key ?? "GPU なし"}）の参照値が無い。` +
              "作るには KARUME_REFERENCE=write で同じレーンを回すこと",
          );
        }
        record(caseId, sha256);
        return { status: "written" };
      }
      if (mode === "rewrite") {
        record(caseId, sha256);
        return { status: "rewritten", previous: expected };
      }
      return { status: sha256 === expected ? "pass" : "fail", expected };
    },
    warnMissing: (caseIds: readonly string[]): void => {
      if (mode !== undefined) return;
      const missing = caseIds.filter((caseId) => lookup(caseId) === undefined);
      if (missing.length === 0) return;
      console.warn(
        `[karume] この環境（${key ?? "GPU なし"}）の参照値が無いケース: ${missing.join(" / ")}。` +
          "そのケースは明示 SKIP する。作るには KARUME_REFERENCE=write で同じレーンを回すこと " +
          `（行の置き場: ${fixtureUrl.pathname}）`,
      );
    },
  };
};

/**
 * 参照門（各 sha ファイルに 1 本）。「この環境の参照値がまだ無い」状態を**無音の緑にしない**
 * ための門番で、ADR 0005 の GPU 門番と同じく opt-out つき。
 */
export const registerReferenceGate = (
  references: References,
  options: { readonly runnable: boolean },
): void => {
  Deno.test({
    name: `参照門: この環境（${ENVIRONMENT.key ?? "GPU なし"}）の参照値がある`,
    // GPU も資産も無い環境では sha 門自体が走らない（この門番も鳴らさない）。
    ignore: ALLOW_NO_REFERENCE || !options.runnable,
    fn: () => {
      assert(
        references.hasCurrent() || references.mode !== undefined,
        `この環境（${ENVIRONMENT.key ?? "GPU なし"}）の参照値が 1 件も無いため sha 門が全て ` +
          "SKIP された。ADR 0005 と同じ理由でこれは FAIL として扱う（検証していないものを " +
          "検証済みと誤読させる）。この機の参照値を作るには KARUME_REFERENCE=write を付けて " +
          `同じレーンを回すこと（行の置き場: ${references.fixtureUrl.pathname}）。` +
          "参照値を持たないまま意図的に通すには KARUME_ALLOW_NO_REFERENCE=1 を設定すること。",
      );
    },
  });
};
