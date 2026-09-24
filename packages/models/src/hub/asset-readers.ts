/**
 * 取得済み資産バイト列の読み口（family 横断で 1 本に寄せたもの）。
 *
 * `fromAssets` / `fromPretrained` のどちらから来ても、パイプラインが資産のバイト列に触る入口は
 * ここだけになる。family ごとに違うのは**文言のラベル**（family 名と、取得キーが載る manifest の
 * 表）だけなので、それを引数で受けて手続きは 1 本に保つ — 文言は family 側の門として逐語で
 * 縛られている（`packages/models/tests/*_pipeline_test.ts`）ので、ラベルを落とさない。
 *
 * 容器の内側の資産（`rope_base` / PLE — ADR 0109 決定 4）は manifest の表に載らないので、
 * 読み口は {@link readWholeAsset}（`AssetReader` 1 本を全量で読む）が受ける。
 */

import type { AssetReader } from "@karume/runtime";

/**
 * 取得キーが載る manifest の表。weights しか持たない family（画像系）と、資産表も併せて持つ
 * family（tokenizer / 表を配るもの）で「どこを直せばいいか」が違うので、文言に出す。
 */
type ManifestTables = "weights" | "weights / assets";

/**
 * 取得済みバイト列を `openContainer` へ渡せる ArrayBuffer にする。
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
 * **容器の資産**の `[offset, offset + length)` を読んで、**自分の buffer を丸ごと占める**
 * ArrayBuffer にする。
 *
 * MUST: buffer 全体を占めていなければ**写す**。区間読みの返りは取得元の器の view でありうるので、
 * `bytes.buffer` をそのまま渡すと block の詰め物や隣の資産まで見せることになる（資産の論理長は
 * 宣言値で、block 長はそれを 4 の倍数へ切り上げた値 — container-v1 §2.2）。写した後なら
 * `new Float32Array(buffer)` のような整列要件のある view もそのまま作れる。RAM の面でも写しが
 * 要る: hub の scan 経路では器は part 全体（既定 256 MiB）なので、数 KB の行の view を握り続けると
 * part 全体が生き残る。buffer 全体を占める返り（part ちょうどの資産）は写さない — 器と共有するが、
 * 握る量は写しと同じで、呼び手は書き換えない。
 *
 * NOTE: 範囲外の区間は runtime の `AssetReader.read` が宣言 `length` と突き合わせて落とす —
 * 同じ検査をここへ写さない。
 */
export const readAssetRange = async (
  reader: AssetReader,
  offset: number,
  length: number,
): Promise<ArrayBuffer> => {
  const bytes = await reader.read(offset, length);
  return bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.slice().buffer;
};

/**
 * **容器の資産** 1 本を論理長ぶん全量読んで ArrayBuffer にする（全量パーサ =
 * `parseSafetensors` / `JSON.parse` へ渡す口）。区間読みで足りる消費側（PLE の行読み）は
 * {@link readAssetRange} を使う。
 */
export const readWholeAsset = (reader: AssetReader): Promise<ArrayBuffer> =>
  readAssetRange(reader, 0, reader.length);

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
