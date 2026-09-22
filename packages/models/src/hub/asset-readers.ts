/**
 * 取得済み資産バイト列の読み口（family 横断で 1 本に寄せたもの）。
 *
 * `fromAssets` / `fromPretrained` のどちらから来ても、パイプラインが資産のバイト列に触る入口は
 * ここだけになる。family ごとに違うのは**文言のラベル**（family 名と、取得キーが載る manifest の
 * 表）だけなので、それを引数で受けて手続きは 1 本に保つ — 文言は family 側の門として逐語で
 * 縛られている（`packages/models/tests/*_pipeline_test.ts`）ので、ラベルを落とさない。
 */

/**
 * 取得キーが載る manifest の表。weights しか持たない family（画像系）と、資産表も併せて持つ
 * family（tokenizer / 表を配るもの）で「どこを直せばいいか」が違うので、文言に出す。
 */
type ManifestTables = "weights" | "weights / assets";

/**
 * 取得済みバイト列を `openModel` へ渡せる ArrayBuffer にする。
 *
 * MUST: `slice` で写さない — 重み 1 本が GB 級になる family があり、写すとホスト RAM のピークが
 * 倍になる。hub は buffer 全体を占める view を返す契約なので、崩れていたら**取得層の不変条件
 * 破れ**として落とす。
 */
export const readAssetBuffer = (
  family: string,
  tables: ManifestTables,
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => {
  if (!Object.hasOwn(assets, key)) {
    throw new Error(
      `${family}: 資産 '${key}' が無い（manifest の ${tables} に ${key} が要る）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const bytes = assets[key];
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new Error(
      `${family}: 資産 '${key}' の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes.buffer;
};

/**
 * 資産 JSON を読む。
 *
 * MUST: `fatal: true` で decode する。既定の TextDecoder は不正 UTF-8 を U+FFFD へ黙って
 * 置換するので、壊れたバイト列が「内容の違う valid JSON」として通ってしまう（hub の
 * manifest・safetensors ヘッダと同じ流儀で fail loudly）。
 * MUST: decode 段と parse 段を別の文言で落とす。どちらで壊れたかは配布物の直し方が違う。
 */
export const readAssetJson = (
  family: string,
  tables: ManifestTables,
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): unknown => {
  const buffer = readAssetBuffer(family, tables, assets, key);
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch (cause) {
    throw new Error(`${family}: 資産 '${key}' が UTF-8 として読めない`, { cause });
  }
  try {
    return JSON.parse(text);
  } catch (cause) {
    throw new Error(`${family}: 資産 '${key}' が JSON として読めない`, { cause });
  }
};
