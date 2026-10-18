/**
 * Readers for already-fetched asset bytes (consolidated into one across families).
 *
 * Whether the call comes from `fromAssets` or `fromPretrained`, this is the only entry through which
 * a pipeline touches asset bytes. What differs per family is only **the label in the message** (the
 * family name and the manifest table that carries the fetch key), so it is taken as an argument and
 * the procedure stays single — the messages are pinned verbatim as the family's checks
 * (`packages/models/tests/*_pipeline_test.ts`), so the label must not be dropped.
 *
 * Assets inside a container (`rope_base` / PLE — ADR 0109 decision 4) are not in the manifest's
 * tables, so their reader is {@link readWholeAsset} (reads one `AssetReader` in full).
 */

import type { AssetReader } from "@karume/runtime";

/**
 * The manifest table that carries the fetch key. A family with weights only (the image families)
 * and a family that also has an assets table (those that ship a tokenizer / tables) differ in
 * "where to fix it", so it appears in the message.
 */
type ManifestTables = "weights" | "weights / assets";

/**
 * Turns already-fetched bytes into an ArrayBuffer that can be passed to `openContainer`.
 *
 * MUST: do not copy with `slice` — some families have a single weight in the GB range, and copying
 * doubles the host RAM peak. hub's contract is to return a view that covers the whole buffer, so if
 * that breaks it fails as **a broken invariant of the fetch layer**.
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
 * Reads `[offset, offset + length)` of **a container asset** into an ArrayBuffer that **covers its
 * own buffer entirely**.
 *
 * MUST: **copy** when the bytes do not cover the whole buffer. The result of a range read may be a
 * view into the source's holder, so passing `bytes.buffer` as is would expose block padding and even
 * neighbouring assets (an asset's logical length is the declared value; the block length is that
 * rounded up to a multiple of 4 — container-v1 §2.2). After the copy, views with alignment
 * requirements such as `new Float32Array(buffer)` can be built directly. RAM needs the copy too: on
 * hub's scan path the holder is a whole part (256 MiB by default), so holding on to a view of a few
 * KB of rows keeps the whole part alive. A result that covers the whole buffer (an asset exactly the
 * size of its part) is not copied — it shares the holder, but holds as much as a copy would, and the
 * caller does not write to it.
 *
 * NOTE: an out-of-range interval is rejected by the runtime's `AssetReader.read` against the
 * declared `length` — the same check is not duplicated here.
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
 * Reads one **container asset** in full, up to its logical length, into an ArrayBuffer (the entry
 * for whole-input parsers = `parseSafetensors` / `JSON.parse`). Consumers that need only range
 * reads (PLE row reads) use {@link readAssetRange}.
 */
export const readWholeAsset = (reader: AssetReader): Promise<ArrayBuffer> =>
  readAssetRange(reader, 0, reader.length);

/**
 * Reads an asset JSON.
 *
 * MUST: decode with `fatal: true`. The default TextDecoder silently replaces invalid UTF-8 with
 * U+FFFD, so broken bytes would pass as "valid JSON with different content" (fail loudly, in the
 * same manner as hub's manifest and the safetensors header).
 * MUST: fail the decode step and the parse step with different messages. Which one broke changes
 * how the distribution is fixed.
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
