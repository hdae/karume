/**
 * manifest の `session`（manifest 所有語彙）→ runtime `SessionOptions` の写像と、利用者の
 * 明示指定との**合成**（**パイプライン非依存の共通処理** — 8 家族の admission が同じ 1 本を通る）。
 *
 * 合成規則は全家族共通の 1 本: キーごとに **明示 > quant 宣言 > runtime 既定**（ADR 0058 追記
 * 2026-09-26）。家族が違うのは「どのキーを受理するか」（{@link FamilySessionPolicy}）だけで、
 * 優先順位・値域・組合せ・送出型の分類は家族ごとに書かない。
 *
 * MUST: barrel には出さない。これは配布形の宣言を runtime のノブへ翻訳する内部機構で、
 * 利用者が触る面ではない（`export` はパッケージ内テストが写像そのものを叩くため — 写像の
 * 抜けは GPU を回さないと露見しない位置にある）。
 *
 * NOTE: 元は 7 家族の `pipeline.ts` へバイト単位で複製されていた。複製は「綴りの改名」は
 * 型検査で捕まえられる一方、「`SessionSpec` へのキー追加」は**どの家族も型検査を通る**
 * （写像は書いていないキーを黙って落とすだけ）ため、追随を忘れた家族がそのまま沈黙劣化した。
 * 1 本化と下の網羅表で、その余地を構造的に消している。
 */

import type { SessionSpec } from "@karume/hub";
import { type SessionOptions, sessionOptionsViolation } from "@karume/runtime";

import { ModelInputError } from "../errors.ts";

/**
 * 利用者が明示で上書きできる欄の型（manifest 語彙と同じキー集合・値は runtime の値域）。
 *
 * NOTE: 値域が `SessionSpec` でなく runtime 側なのは、配布形が宣言しない実行形（例
 * `linearGemvReduce: "parallel-subgroup32"` — device の subgroups 次第）も利用者は選べるため。
 */
export type SessionOverrides = Partial<Pick<SessionOptions, keyof SessionSpec>>;

/**
 * 家族ごとの**受理表** — true = この家族が manifest の宣言と明示指定の両方で受けるキー。
 *
 * MUST: キー集合は `Required<SessionSpec>` の**網羅**（{@link WRITERS} と同じ縛り）。
 * `SessionSpec` にノブが増えると全家族の表が型検査で落ちるので、「新しいノブを受けるか」を
 * 家族ごとに決め忘れる余地が無い。値の既定は持たない（既定は runtime が 1 か所で持つ）。
 */
export type FamilySessionPolicy = { readonly [K in keyof Required<SessionSpec>]: boolean };

/** 写像 1 キーぶん — `source` の欄が埋まっているときだけ `SessionOptions` の欄を作る。 */
type SpecWriter = (source: SessionOverrides) => SessionOptions;

/**
 * manifest 所有の各キーを `SessionOptions` の欄へ写す表。
 *
 * MUST: キー集合は `Required<SessionSpec>` の**網羅** — `SessionSpec` にノブが増えたら
 * この宣言が型検査で落ちるので、写像の追随漏れ（= 配布形が宣言したノブが runtime へ届かない
 * 沈黙劣化）が起きない。
 *
 * MUST: 個々の writer はスプレッドで丸投げしない（ADR 0038 §3 — 素通しにすると綴りが変わった
 * 瞬間に runtime が未知キーを黙って無視する。写像を明示すると綴りが割れた時点で型検査が
 * 落ちる）。`SessionOptions.submitPolicy`（TDR 予算 = ホスト政策）のように manifest 側に
 * 席の無いノブが混ざらないのも、この明示写像の効果。
 * MUST: 各 writer は欄を**1 度だけ**読む（分割代入）。明示指定は利用者のオブジェクトなので、
 * getter が判定した値と写した値をすり替えられないようにする。
 */
const WRITERS: { readonly [K in keyof Required<SessionSpec>]: SpecWriter } = {
  linearCompute: ({ linearCompute }) => linearCompute === undefined ? {} : { linearCompute },
  attentionCompute: ({ attentionCompute }) =>
    attentionCompute === undefined ? {} : { attentionCompute },
  attentionScoreStorage: ({ attentionScoreStorage }) =>
    attentionScoreStorage === undefined ? {} : { attentionScoreStorage },
  linearGemvReduce: ({ linearGemvReduce }) =>
    linearGemvReduce === undefined ? {} : { linearGemvReduce },
  fuseRmsNormAdd: ({ fuseRmsNormAdd }) => fuseRmsNormAdd === undefined ? {} : { fuseRmsNormAdd },
  fuseLinearStaticQuantize: ({ fuseLinearStaticQuantize }) =>
    fuseLinearStaticQuantize === undefined ? {} : { fuseLinearStaticQuantize },
  packedStaticQuantize: ({ packedStaticQuantize }) =>
    packedStaticQuantize === undefined ? {} : { packedStaticQuantize },
};

/** 埋まっている欄だけを持つ `SessionOptions` を組む（未指定のキーは欄ごと作らない）。 */
const writeOptions = (source: SessionOverrides): SessionOptions => {
  let options: SessionOptions = {};
  for (const write of Object.values(WRITERS)) options = { ...options, ...write(source) };
  return options;
};

/** 宣言された欄だけを持つ `SessionOptions` を組む（未指定のキーは欄ごと作らない）。 */
export const toSessionOptions = (spec: SessionSpec): SessionOptions => writeOptions(spec);

/** 文字列の受理集合（`Record<union, true>` — union に値が増えると型検査で欠落が赤くなる）。 */
const isOneOf = (accepted: Readonly<Record<string, true>>, value: unknown): boolean =>
  typeof value === "string" && Object.hasOwn(accepted, value);

const LINEAR_COMPUTES: Readonly<Record<NonNullable<SessionOptions["linearCompute"]>, true>> = {
  f32: true,
  a8: true,
  f16: true,
};
const ATTENTION_COMPUTES: Readonly<Record<NonNullable<SessionOptions["attentionCompute"]>, true>> =
  { f32: true, f16: true, a8: true };
const SCORE_STORAGES: Readonly<
  Record<NonNullable<SessionOptions["attentionScoreStorage"]>, true>
> = { f32: true, f16: true };
const LINEAR_GEMV_REDUCES: Readonly<Record<NonNullable<SessionOptions["linearGemvReduce"]>, true>> =
  { sequential: true, parallel: true, "parallel-subgroup32": true };

/**
 * 欄 1 つぶんの値域の門（欄が無ければ通す）。
 *
 * MUST: runtime の `sessionOptionsViolation` より**前**に見る。runtime は構築と同じ読み方で
 * 省略を `??` で既定へ読むので、`null` のような「欄はあるが値が不正」が既定に化けて通る —
 * 明示の `null` を宣言へも既定へも戻さず拒否するのはこの層の責務。
 * MUST: 診断で利用者の変換（`toString` / `Symbol.toPrimitive`）を呼ばない — 値は載せず欄名だけ。
 */
const VALUE_GATES: {
  readonly [K in keyof Required<SessionSpec>]: (options: SessionOptions) => string | undefined;
} = {
  linearCompute: ({ linearCompute }) =>
    linearCompute === undefined || isOneOf(LINEAR_COMPUTES, linearCompute)
      ? undefined
      : "linearComputeが不正",
  attentionCompute: ({ attentionCompute }) =>
    attentionCompute === undefined || isOneOf(ATTENTION_COMPUTES, attentionCompute)
      ? undefined
      : "attentionComputeが不正",
  attentionScoreStorage: ({ attentionScoreStorage }) =>
    attentionScoreStorage === undefined || isOneOf(SCORE_STORAGES, attentionScoreStorage)
      ? undefined
      : "attentionScoreStorageが不正",
  linearGemvReduce: ({ linearGemvReduce }) =>
    linearGemvReduce === undefined || isOneOf(LINEAR_GEMV_REDUCES, linearGemvReduce)
      ? undefined
      : "linearGemvReduceが不正",
  fuseRmsNormAdd: ({ fuseRmsNormAdd }) =>
    fuseRmsNormAdd === undefined || typeof fuseRmsNormAdd === "boolean"
      ? undefined
      : "fuseRmsNormAddはbooleanでなければならない",
  fuseLinearStaticQuantize: ({ fuseLinearStaticQuantize }) =>
    fuseLinearStaticQuantize === undefined || typeof fuseLinearStaticQuantize === "boolean"
      ? undefined
      : "fuseLinearStaticQuantizeはbooleanでなければならない",
  packedStaticQuantize: ({ packedStaticQuantize }) =>
    packedStaticQuantize === undefined || typeof packedStaticQuantize === "boolean"
      ? undefined
      : "packedStaticQuantizeはbooleanでなければならない",
};

/**
 * 値域・型・組合せを**1 本**で見る（受理条件の正本 — 条件は出所で分けない）。
 *
 * 組合せ（`fuseLinearStaticQuantize` は `linearGemvReduce: parallel` が要る 等）は runtime の
 * `sessionOptionsViolation` をそのまま借りる — Session 構築が同じ関数で落とすので、ここに写しを
 * 置くと 2 つの受理集合が割れうる。GPU の能力（shader-f16 / subgroups）を要る条件はここでは
 * 決まらない（device を見る段で落ちる）。
 */
const optionsViolation = (options: SessionOptions): string | undefined => {
  for (const gate of Object.values(VALUE_GATES)) {
    const violation = gate(options);
    if (violation !== undefined) return violation;
  }
  return sessionOptionsViolation(options);
};

/** 受理表を名前で引く（`Object.keys` 由来の文字列キーを型の外から受けるため）。 */
const accepts = (policy: FamilySessionPolicy, key: string): boolean => {
  const table: Readonly<Record<string, boolean | undefined>> = policy;
  return Object.hasOwn(table, key) && table[key] === true;
};

/**
 * quant 宣言（`declared`）と利用者の明示指定（`overrides`）を合成して、Session へ渡す
 * `SessionOptions` を決める（ADR 0058 追記 2026-09-26 — 全家族共通の規則）。
 *
 * - キーごとに **明示があればそれ、無ければ宣言、どちらも無ければ欄を作らない**（runtime 既定）。
 *   明示の `false` は宣言の `true` に勝つ（`undefined` だけが「指定なし」）。
 * - 不正な明示値（`null` / 非 boolean / 受理集合外の綴り）は宣言へ戻さず `ModelInputError`。
 * - `policy` が受けないキーを manifest が宣言していたら素の `Error`（資産の齟齬 — 呼び手が
 *   入力を直しても直らない）。受けないキーを明示されたら `ModelInputError`（黙って捨てると
 *   「指定したのに効かない」になる）。
 * - 実効設定が受理条件を破ったら、送出型を出所で分ける（ADR 0107 決定 2）: 同じ違反が宣言
 *   **だけ**でも成立するなら `Error`、明示指定が関与して初めて成立するなら `ModelInputError`。
 *
 * MUST: 家族の admission 席（**重みの part を 1 バイトも取る前**）から呼ぶ。後段で呼ぶと、
 * 未対応の宣言も誤った明示指定も GB 級の取得の後にしか落ちない。
 *
 * @param where 診断の主語（`"AnimaPipeline: quant 'f16'"`）。
 */
export const resolveSessionOptions = (
  policy: FamilySessionPolicy,
  declared: SessionSpec,
  overrides: SessionOverrides,
  where: string,
): SessionOptions => {
  for (const key of Object.keys(declared)) {
    if (!accepts(policy, key)) throw new Error(`${where}: session.${key}は未対応`);
  }
  // 判定した値とマージに使う値が getter ですり替わらないよう、明示指定は 1 度だけ読んで
  // 素のオブジェクトへ写す（`writeOptions` は欄を 1 度ずつ読み、`undefined` の欄は作らない）。
  const explicit = writeOptions(overrides);
  for (const key of Object.keys(explicit)) {
    if (!accepts(policy, key)) {
      throw new ModelInputError(`${where}: ${key}はこの系列では指定できない`);
    }
  }
  // ?? は null を宣言へ戻すので使わない — 明示の欄は値に依らず宣言を上書きし、不正な値は
  // 直後の門で落とす。
  const effective: SessionOptions = { ...writeOptions(declared), ...explicit };
  const violation = optionsViolation(effective);
  if (violation !== undefined) {
    // 送出型だけを出所で分ける（ADR 0107 決定 2）。判定は違反の同一性で見る — 同じ違反が
    // 宣言だけでも成立するなら打つ手は配布の修正（500 相当）、上書きが関与して初めて
    // 成立するなら打つ手は指定の修正（400 相当）。
    const message = `${where}: ${violation}`;
    throw violation === optionsViolation(writeOptions(declared))
      ? new Error(message)
      : new ModelInputError(message);
  }
  return effective;
};

/**
 * 呼び手の**明示指定だけ**を相手にする門（入力起因 = `ModelInputError`）。
 *
 * MUST: quant 宣言を持たない入口（`fromAssets` のうち manifest を受けない面）は、この門を
 * **資産を 1 バイトも開く前**に通す。通さないと同じ誤指定が Session 構築まで降り、runtime の
 * `ExecutionError` に化けて manifest を持つ入口と分類が割れる。宣言 `{}` を相手にした
 * {@link resolveSessionOptions} と同じ判定になる。
 */
export const assertSessionOverrides = (
  policy: FamilySessionPolicy,
  overrides: SessionOverrides,
  where: string,
): void => {
  resolveSessionOptions(policy, {}, overrides, where);
};
