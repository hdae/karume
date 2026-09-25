/**
 * ローカル資産の「在れば読む」読み口（無い = `undefined` を SKIP 判定の源にする門が使う）。
 *
 * MUST: NotFound 以外は伝播させる — 権限エラー・ディレクトリ・読込途中の失敗などを
 * 「資産が無い」と読み替えると、実行されていない検証が SKIP として静かに緑になる
 * （ADR 0005 の明示 SKIP・fail loudly）。
 *
 * NOTE: ローカル読みに `Deno.readFile` を使うのはテストだけ。パッケージ本体は Web 標準 API
 * のみ（横断不変条件）。
 */

const undefinedIfNotFound = (cause: unknown): undefined => {
  if (cause instanceof Deno.errors.NotFound) return undefined;
  throw cause;
};

/** テキストとして読む。path が無ければ `undefined`、それ以外の失敗は投げる。 */
export const readTextIfPresent = (path: URL): Promise<string | undefined> =>
  Deno.readTextFile(path).catch(undefinedIfNotFound);

/** バイト列として読む。path が無ければ `undefined`、それ以外の失敗は投げる。 */
export const readFileIfPresent = (path: URL): Promise<Uint8Array<ArrayBuffer> | undefined> =>
  Deno.readFile(path).catch(undefinedIfNotFound);
