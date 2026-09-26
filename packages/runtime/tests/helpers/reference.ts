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
 * ADR 0005 の門番と同じ形（opt-out つき）で 1 本落とす。門が数えるのは**呼び手が登録した
 * ケース**だけである（{@link referenceGatePasses}）。
 *
 * ## 呼び手の型（sha 門を持つ e2e）
 *
 * - **既存テストの中の追加検査**として sha を採る: 元の検査（tolerance 突合・判別・`.lab` の
 *   完全一致など）が**通った後で** {@link settleOrObserve} を呼ぶ（元の検査が割れた出力から行を
 *   作らない）→ 返った欄（{@link ReferenceOutcome.fields}）を結果へ積む → **積んだ後に**
 *   突合の `fail` を {@link referenceMismatchMessage} で落とす（先に投げると実測 sha が結果に
 *   残らない）。比較モードで行が無いケースも同じ 1 本を通り、決着の形は 1 つに決まっている —
 *   実物を書き、sha を採り、`status: "pass"` + `actual` + `artifact`（`expected` 無し）を積む。
 *   突合はしないが実測 sha は結果に残るので、`tools/verify-diff` が環境をまたいで `actual` を
 *   比べられる（行の作り方は登録時の {@link References.warnMissing} が出す）。
 * - **テストそのものが sha 門**: `ignore: !RUNNABLE || references.lacksReference(id)` で登録し、
 *   本体は {@link settleReference} → {@link referenceEntryFields} を上と同じ順で積んでから落とす
 *   （行が無いケースは本体まで来ない）。
 *
 * どちらもファイル末尾で {@link References.warnMissing} と {@link registerReferenceGate} を呼ぶ。
 * 実物が f32 テンソルなら {@link f32ArtifactBytes}（safetensors 1 本・metadata なし）で作る。
 */

import { assert } from "@std/assert";
import { ENVIRONMENT, type Environment } from "./environment.ts";
import type { ResultEntry, Results } from "./results.ts";
import { buildSafetensors } from "./safetensors.ts";

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

/** {@link openReferences} の注入席（単体テスト用 — 実行時の呼びは何も渡さない）。 */
export type OpenReferencesOptions = {
  /** 参照値を索く環境（省略時はこの実行環境）。 */
  readonly environment?: Environment;
  /**
   * この走行の参照モード。
   *
   * MUST: **欄ごと省いたときだけ**環境変数 `KARUME_REFERENCE` を読む。既定引数で受けると
   * 「比較専用のつもりで明示的に渡した `undefined`」が環境変数の write / rewrite に化け、
   * `KARUME_REFERENCE=write` を付けた走行で単体テストが書き込みモードで回る。
   */
  readonly mode?: ReferenceMode | undefined;
};

/**
 * 参照値 fixture を開く。
 *
 * `environment` / `mode` を注入できるのは単体テストのため（実 GPU も環境変数も要らずに
 * モードの分岐を検査できる）。実行時の呼びは第 2 引数ごと省く。
 */
export const openReferences = (
  fixtureUrl: URL,
  options: OpenReferencesOptions = {},
): References => {
  const environment = options.environment ?? ENVIRONMENT;
  const mode = Object.hasOwn(options, "mode") ? options.mode : REFERENCE_MODE;
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
  /**
   * 現環境の行を**ディスクから**読む（判定の唯一の材料）。
   *
   * MUST: 突合の判定は開いた時点の写しではなく最新のディスク行に対して行う。同じ環境キーで
   * 2 ハンドルを開くと、写しで判定する形は先に書かれた行を「無い」と見なし、`write` が
   * その行を黙って上書きする（冒頭表の MUST「`write` は行があるケースを上書きしない」が破れる）。
   */
  const onDiskLookup = (caseId: string): string | undefined =>
    key === undefined ? undefined : readDocument(fixtureUrl).cases[caseId]?.[key];
  const record = (caseId: string, sha256: string): "written" | "rewritten" => {
    const current = requireKey();
    // MUST: 書く直前に読み直す。開いた時点の写しをそのまま書き戻すと、その間に別のハンドルが
    // 足した**他環境の行**が消える（「他環境の行には決して触らない」はこの読み直しで成り立つ）。
    const onDisk = readDocument(fixtureUrl).cases;
    // 旧値も同じ読み直しから採る（写しから採ると、別ハンドルが足した行を「無かった」と報告する）。
    const previous = onDisk[caseId]?.[current];
    onDisk[caseId] = { ...onDisk[caseId], [current]: sha256 };
    Deno.writeTextFileSync(fixtureUrl, serialize(onDisk));
    cases[caseId] = { ...cases[caseId], [current]: sha256 };
    return previous === undefined ? "written" : "rewritten";
  };
  return {
    mode,
    fixtureUrl,
    lookup,
    lacksReference: (caseId: string): boolean => mode === undefined && lookup(caseId) === undefined,
    record,
    check: (caseId: string, sha256: string): ReferenceCheck => {
      const expected = onDiskLookup(caseId);
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
 * 参照門の緑条件（純関数 — 回帰テストはここを直に突く）。
 *
 * 数えるのは**この走行が登録したケース**だけである。fixture 全体を横断して 1 行でもあれば
 * 緑にすると、ケースの改名・削除で残った孤児行（もう誰も突き合わせない行）が、現役ケース
 * 全 SKIP を緑で隠す。
 *
 * MUST: 現役ケースの**一部**にだけ行がある状態は緑のまま（ADR 0106 の設計 — この門が言うのは
 * 「この環境の参照値が 1 件も無いのではない」ことだけで、全ケース検証済みとは言わない）。
 */
export const referenceGatePasses = (
  references: Pick<References, "mode" | "lookup">,
  caseIds: readonly string[],
): boolean =>
  references.mode !== undefined ||
  caseIds.some((caseId) => references.lookup(caseId) !== undefined);

/**
 * 参照門（各 sha ファイルに 1 本）。「この環境の参照値がまだ無い」状態を**無音の緑にしない**
 * ための門番で、ADR 0005 の GPU 門番と同じく opt-out つき。
 *
 * `caseIds` はそのファイルが登録したケース ID 全部（`warnMissing` へ渡すものと同じ集合）。
 */
export const registerReferenceGate = (
  references: References,
  options: { readonly runnable: boolean; readonly caseIds: readonly string[] },
): void => {
  Deno.test({
    name: `参照門: この環境（${ENVIRONMENT.key ?? "GPU なし"}）の参照値がある`,
    // GPU も資産も無い環境では sha 門自体が走らない（この門番も鳴らさない）。
    ignore: ALLOW_NO_REFERENCE || !options.runnable,
    fn: () => {
      assert(
        referenceGatePasses(references, options.caseIds),
        `この環境（${ENVIRONMENT.key ?? "GPU なし"}）の参照値が、このファイルが登録した ` +
          `${options.caseIds.length} ケースのどれにも無いため sha 門が全て SKIP された。` +
          "ADR 0005 と同じ理由でこれは FAIL として扱う（検証していないものを " +
          "検証済みと誤読させる）。この機の参照値を作るには KARUME_REFERENCE=write を付けて " +
          `同じレーンを回すこと（行の置き場: ${references.fixtureUrl.pathname}）。` +
          "参照値を持たないまま意図的に通すには KARUME_ALLOW_NO_REFERENCE=1 を設定すること。",
      );
    },
  });
};

/**
 * バイト列の sha256（小文字 16 進 64 桁）。
 *
 * NOTE: `src/format/container` にも同名の関数があるが、こちらは検証の物差しなので
 * 検査対象の実装に寄りかからず Web Crypto を直に呼ぶ。
 */
export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** {@link settleReference} の決着（結果への記録は呼び手が行う）。 */
export type ReferenceSettlement = {
  /** 実物ファイルのバイト列の sha256（結果の `actual` 欄に入る値）。 */
  readonly sha256: string;
  readonly check: ReferenceCheck;
  /** 実物の置き場（結果の席の中）。 */
  readonly artifactUrl: URL;
};

/**
 * 実物を結果の席へ書き、その sha256 を参照値と突き合わせる（sha 門の共通の末端）。
 *
 * - 実物は**突合の前に**、成功・失敗を問わず書く（不一致のときだけ残す形だと、一致した回の
 *   実物が手元に無く、次に割れたときの A/B が採れない）。
 * - sha は書いたのと同じバイト列から採る（結果の `actual` と実物の sha が常に一致する）。
 *
 * 結果への記録はしない — 呼び手が {@link referenceEntryFields} で `record` するか、
 * `runRecordedCase` の決着に載せる。不一致でも投げない（記録より先に投げると実測 sha が結果に
 * 残らない）。比較モードで行が無いケースは {@link References.check} が投げる — 追加検査として
 * sha を採る呼び手はこれではなく {@link settleOrObserve} を使う（行が無いケースの決着の形が
 * そこで 1 つに決まる）。
 */
export const settleReference = async (
  references: References,
  results: Pick<Results, "artifact">,
  settled: {
    /** 参照値のケース ID。 */
    readonly id: string;
    /** 実物のファイル名（結果の席からの相対）。 */
    readonly artifact: string;
    readonly bytes: Uint8Array<ArrayBuffer>;
  },
): Promise<ReferenceSettlement> => {
  const artifactUrl = results.artifact(settled.artifact);
  await Deno.writeFile(artifactUrl, settled.bytes);
  const sha256 = await sha256Hex(settled.bytes);
  const check = references.check(settled.id, sha256);
  announceCheck(settled.id, check, sha256);
  return { sha256, check, artifactUrl };
};

/**
 * 決着を結果 1 件の欄へ写す（`status` / `expected` / `actual` / `artifact` の順）。
 *
 * `expected` は {@link expectedOf} に従う（作った回は欄ごと持たない・焼き直した回は旧い値）。
 */
export const referenceEntryFields = (
  settlement: ReferenceSettlement,
  artifact: string,
): Pick<ResultEntry, "status" | "expected" | "actual" | "artifact"> => {
  const expected = expectedOf(settlement.check);
  return {
    status: settlement.check.status,
    ...(expected === undefined ? {} : { expected }),
    actual: settlement.sha256,
    artifact,
  };
};

/** {@link settleOrObserve} の決着。 */
export type ReferenceOutcome = {
  /** 結果 1 件の欄（`status` / `expected` / `actual` / `artifact` の順 — そのまま広げて積む）。 */
  readonly fields: Pick<ResultEntry, "status" | "expected" | "actual" | "artifact">;
  /** 突き合わせた決着。比較モードで行が無く、突合を飛ばした回は `undefined`。 */
  readonly settlement: ReferenceSettlement | undefined;
};

/**
 * 追加検査としての sha の末端（モジュール doc「呼び手の型」の 1 つめ）。
 *
 * - 行がある、または作るモード: {@link settleReference} の決着（欄は {@link referenceEntryFields}）。
 * - 比較モードで行が無い: 突合だけを飛ばす。実物は同じく書き、sha も同じバイト列から採る。
 *   決着は `status: "pass"`（呼ぶのは元の検査が通った後）+ `actual` + `artifact` で、`expected`
 *   は持たない（突き合わせた相手がいない）。
 *
 * 分岐を呼び手ごとに書かせないのは、行が無いケースの結果の形が系列ごとに割れるため（実測 sha を
 * 積む系列と何も積まない系列が混ざると、環境横断の `actual` 比較が一部の系列でしか効かない）。
 */
export const settleOrObserve = async (
  references: References,
  results: Pick<Results, "artifact">,
  settled: {
    /** 参照値のケース ID。 */
    readonly id: string;
    /** 実物のファイル名（結果の席からの相対）。 */
    readonly artifact: string;
    readonly bytes: Uint8Array<ArrayBuffer>;
  },
): Promise<ReferenceOutcome> => {
  if (references.lacksReference(settled.id)) {
    await Deno.writeFile(results.artifact(settled.artifact), settled.bytes);
    return {
      fields: {
        status: "pass",
        actual: await sha256Hex(settled.bytes),
        artifact: settled.artifact,
      },
      settlement: undefined,
    };
  }
  const settlement = await settleReference(references, results, settled);
  return { fields: referenceEntryFields(settlement, settled.artifact), settlement };
};

/**
 * f32 テンソル 1 本を実物（safetensors 1 本）のバイト列にする。テンソル名は呼び手が渡す
 * グラフの出力名、dtype は `F32`、shape は出力のまま。
 *
 * MUST: metadata を入れない — 走行ごとに変わる値（時刻・チェックアウト）が混ざると数値が同じ
 * でも sha が動き、参照行が「数値の退行」ではなく「走らせた時刻」を掴む。
 * MUST: shape の要素数とデータ長の食い違いは throw（切り出し違いの実物を黙って固定しない）。
 */
export const f32ArtifactBytes = (
  name: string,
  shape: readonly number[],
  data: Float32Array<ArrayBuffer>,
): Uint8Array<ArrayBuffer> => {
  const count = shape.reduce((product, dim) => product * dim, 1);
  if (count !== data.length) {
    throw new Error(
      `実物 '${name}': shape [${
        shape.join(",")
      }] の要素数 ${count} とデータ長 ${data.length} が食い違う`,
    );
  }
  return new Uint8Array(buildSafetensors([{
    name,
    dtype: "F32",
    shape,
    data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  }]));
};

/**
 * 参照値と食い違ったときの診断文。
 *
 * MUST: ここで tolerance に逃げない。参照は sha256 しか無い（参照のバイト列は持っていない）
 * ので先頭差分位置は原理的に出せない — 代わりに実物と fixture の置き場を並べ、人が突き合わせ
 * られる形にする。不一致でない決着を渡すのは呼び手の誤りなので throw する。
 */
export const referenceMismatchMessage = (
  label: string,
  settlement: ReferenceSettlement,
  references: Pick<References, "fixtureUrl">,
): string => {
  const { check } = settlement;
  if (check.status !== "fail") {
    throw new Error(`${label}: 決着が '${check.status}' なので不一致の診断は作れない`);
  }
  return `${label}: 実物の sha256 が参照と一致しない\n` +
    `  期待 ${check.expected}\n  実際 ${settlement.sha256}\n` +
    `  実物 ${settlement.artifactUrl.pathname}（参照はバイト列ではなく sha256 のみなので先頭差分位置は出せない）\n` +
    `  参照値 ${references.fixtureUrl.pathname}\n` +
    "  tolerance に逃げない — 意図した変更なら、何が変わったのかを言えたうえで " +
    "KARUME_REFERENCE=rewrite で焼き直すこと";
};
