/**
 * descriptor / IR の JSON を**安全に読む**ための門と、正準直列化の補助 — docs/container-v1.md §0 /
 * §10、docs/ir-v2.md「正準直列化」。
 *
 * - 読み: UTF-8 として不正・JSON として不正・非有限数（`1e999` は文法上正当で Infinity に丸まる）・
 *   深すぎる入れ子・`__proto__` キーは fail loudly。
 * - 書き: 名前をキーに持つ map は **code point 順**（UTF-16 単位の順ではない）。数値の綴りは
 *   `JSON.stringify` = ECMAScript `Number::toString` がそのまま正準。
 */

import { ContainerFormatError } from "./header.ts";
import { MAX_JSON_DEPTH } from "./limits.ts";

export type JsonObject = Readonly<Record<string, unknown>>;

export const isJsonObject = (value: unknown): value is JsonObject =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * `JSON.parse` を通ったあとの値を検査する（fail loudly の理由は各分岐）。
 * 深さは根を 0 と数える。
 */
export const assertJsonSafe = (
  value: unknown,
  path: string,
  depth: number,
  fail: (message: string) => never,
): void => {
  if (depth > MAX_JSON_DEPTH) fail(`${path}: 入れ子が深さ上限 ${MAX_JSON_DEPTH} を超えた`);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) fail(`${path}: 非有限数（1e999 等の溢れを含む）`);
    return;
  }
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertJsonSafe(item, `${path}[${index}]`, depth + 1, fail));
    return;
  }
  if (isJsonObject(value)) {
    for (const key of Object.keys(value)) {
      // 素の `{}` へ代入すると [[Prototype]] 設定に化けて own property が作られない名前。
      // Deno は既定でこの setter を無効化しているが、ブラウザ（対象実行系の一方）では起きる。
      if (key === "__proto__") fail(`${path}: キー '__proto__' は受理しない`);
      assertJsonSafe(value[key], `${path}.${key}`, depth + 1, fail);
    }
    return;
  }
  fail(`${path}: JSON として表せない値`);
};

/** バイト列 → 検査済みの JSON 値。`maxBytes` は文書ごとの上限（§10）。 */
export const decodeJsonDocument = (
  bytes: Uint8Array<ArrayBuffer>,
  path: string,
  maxBytes: number,
  fail: (message: string) => never,
): unknown => {
  if (bytes.byteLength > maxBytes) {
    fail(`${path} が上限 ${maxBytes} バイトを超える: ${bytes.byteLength}`);
  }
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    fail(`${path} が UTF-8 として不正: ${String(cause)}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    fail(`${path} が JSON として不正: ${String(cause)}`);
  }
  assertJsonSafe(parsed, path, 0, fail);
  return parsed;
};

/**
 * code point 順の比較（ir-v2.md「正準直列化」）。`Array.prototype.sort` の既定は UTF-16 単位の
 * 順なので、非 BMP 文字（サロゲートペア）で順序が変わる — 名前は実測で全て ASCII だが、規則は
 * 書き手に依らず 1 つに固定する。
 */
export const compareCodePoints = (a: string, b: string): number => {
  const ia = a[Symbol.iterator]();
  const ib = b[Symbol.iterator]();
  for (;;) {
    const na = ia.next();
    const nb = ib.next();
    if (na.done && nb.done) return 0;
    if (na.done) return -1;
    if (nb.done) return 1;
    const ca = na.value.codePointAt(0) as number;
    const cb = nb.value.codePointAt(0) as number;
    if (ca !== cb) return ca < cb ? -1 : 1;
  }
};

export const sortedByCodePoints = (keys: Iterable<string>): string[] =>
  [...keys].sort(compareCodePoints);

/**
 * 名前をキーに持つ map の正準形 — code point 順に並べたエントリ列。直列化は
 * {@link canonicalJsonText} だけが行う。
 *
 * MUST: 正準順をオブジェクトの列挙順で表さない。JS のオブジェクトは整数に見えるキー（"2" / "10"）を
 * 挿入順に関係なく先に昇順で列挙するので、並べた順に代入しても `JSON.stringify` は "2","10" と
 * 書く（code point 順は "10","2" — Python の書き手はこちらを書く）。
 */
export class SortedEntries {
  readonly entries: readonly (readonly [string, unknown])[];
  constructor(entries: readonly (readonly [string, unknown])[]) {
    this.entries = entries;
  }
}

/**
 * オブジェクトのキーを code point 順に並べたエントリ列（値は `map` で変換 — 既定は恒等）。
 *
 * MUST: キー `__proto__` は書き手の側でも拒否する（docs/container-v1.md §0）。読み手が拒否する
 * 名前を書けば、自分の出力を自分で読めない。
 */
export const sortedObject = <T>(
  source: Readonly<Record<string, T>>,
  map: (value: T, key: string) => unknown = (value) => value,
): SortedEntries =>
  new SortedEntries(
    sortedByCodePoints(Object.keys(source)).map((key) => {
      if (key === "__proto__") throw new ContainerFormatError("キー '__proto__' は書けない");
      return [key, map(source[key], key)] as const;
    }),
  );

/**
 * 自由形の JSON 値（ノードの `attrs` など）を再帰的に正準化する — オブジェクトはキーを code point
 * 順に、配列は宣言順のまま、スカラーはそのまま。
 */
export const canonicalJsonValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(canonicalJsonValue);
  if (isJsonObject(value)) return sortedObject(value, canonicalJsonValue);
  return value;
};

const fieldText = ([key, value]: readonly [string, unknown]): string =>
  `${JSON.stringify(key)}:${canonicalJsonText(value)}`;

/**
 * 正準 JSON の直列化器（空白なし）。{@link SortedEntries} はエントリ列の順に、固定スキーマの
 * オブジェクトは組み立てた欄の順に書く（固定スキーマの欄名は整数に見えないので列挙順 = 組み立て順）。
 * 並べ替えはしない — 順は組み立て側が決める（Python の `canonical_json` と同じ分担）。
 *
 * `JSON.stringify` が黙って書き換える値（非有限数 → null・配列の穴と undefined → null）は
 * fail loudly。固定スキーマの undefined の欄は書かない（`JSON.stringify` と同じ — 省略可能な欄）。
 */
export const canonicalJsonText = (value: unknown): string => {
  if (value instanceof SortedEntries) return `{${value.entries.map(fieldText).join(",")}}`;
  // Array.from は穴を undefined として渡す（map は穴を飛ばし、join が空文字に綴る）。
  if (Array.isArray(value)) return `[${Array.from(value, canonicalJsonText).join(",")}]`;
  if (isJsonObject(value)) {
    const fields = Object.entries(value).filter(([, field]) => field !== undefined);
    return `{${fields.map(fieldText).join(",")}}`;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new ContainerFormatError(`非有限数 ${value} は書けない`);
    return JSON.stringify(value);
  }
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  throw new ContainerFormatError(`JSON として表せない値: ${typeof value}`);
};

export const encodeJsonBytes = (value: unknown): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(canonicalJsonText(value)) as Uint8Array<ArrayBuffer>;
