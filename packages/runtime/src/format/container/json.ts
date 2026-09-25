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
 * オブジェクトのキーを code point 順に並べ替えた写し（値は `map` で変換 — 既定は恒等）。
 *
 * MUST: キー `__proto__` は書き手の側でも拒否する（docs/container-v1.md §0）。読み手が拒否する
 * 名前を書けば、自分の出力を自分で読めない。写しの器も null プロトタイプにして、キーが
 * [[Prototype]] の設定に化けて黙って消える経路を持たない。
 */
export const sortedObject = <T>(
  source: Readonly<Record<string, T>>,
  map: (value: T, key: string) => unknown = (value) => value,
): Record<string, unknown> => {
  const out: Record<string, unknown> = Object.create(null);
  for (const key of sortedByCodePoints(Object.keys(source))) {
    if (key === "__proto__") throw new ContainerFormatError("キー '__proto__' は書けない");
    out[key] = map(source[key], key);
  }
  return out;
};

/**
 * 自由形の JSON 値（ノードの `attrs` など）を再帰的に正準化する — オブジェクトはキーを code point
 * 順に、配列は宣言順のまま、スカラーはそのまま。
 */
export const canonicalJsonValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(canonicalJsonValue);
  if (isJsonObject(value)) return sortedObject(value, canonicalJsonValue);
  return value;
};

export const encodeJsonBytes = (value: unknown): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(JSON.stringify(value)) as Uint8Array<ArrayBuffer>;
