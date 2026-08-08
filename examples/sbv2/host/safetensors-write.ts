/**
 * safetensors の**書き出し**（デモの dump 用の最小実装）。
 *
 * `@karume/runtime` の safetensors は読み取り専用（ランタイムに書き出しの用は無い）。デモは
 * torch 参照へ「離散入力と乱数列そのもの」を渡す必要があり、その運搬形式を新設せずに
 * 既存の資産形式へ揃えたいのでここに置く。ランタイムの公開面（ADR 0008）には出さない。
 *
 * レイアウト: `[u64 LE ヘッダ長][ヘッダ JSON][データ節]`。`data_offsets` はデータ節先頭からの
 * 相対で、データ節の開始が 8 バイト境界に乗るようヘッダを空白で詰める（参照実装と同じ）。
 */

const ALIGNMENT = 8;

/** 書き出せる要素型（意味論 dtype と 1 対 1 — bool は使わないので持たない）。 */
export type DumpTensor =
  | { readonly dtype: "F32"; readonly shape: readonly number[]; readonly data: Float32Array }
  | { readonly dtype: "I32"; readonly shape: readonly number[]; readonly data: Int32Array };

const elementCount = (shape: readonly number[]): number =>
  shape.reduce((product, dim) => product * dim, 1);

export const writeSafetensors = (
  tensors: ReadonlyMap<string, DumpTensor>,
  metadata: Readonly<Record<string, string>>,
): Uint8Array<ArrayBuffer> => {
  const header: Record<string, unknown> = { __metadata__: metadata };
  const chunks: Uint8Array[] = [];
  let offset = 0;
  for (const [name, tensor] of tensors) {
    if (elementCount(tensor.shape) !== tensor.data.length) {
      throw new Error(
        `テンソル '${name}': shape [${tensor.shape}] の要素数が data 長 ${tensor.data.length} と違う`,
      );
    }
    const bytes = new Uint8Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength,
    );
    chunks.push(bytes);
    header[name] = {
      dtype: tensor.dtype,
      shape: [...tensor.shape],
      data_offsets: [offset, offset + bytes.byteLength],
    };
    offset += bytes.byteLength;
  }

  const encoder = new TextEncoder();
  const headerBytes = encoder.encode(JSON.stringify(header));
  // データ節の開始（8 + ヘッダ長）を 8 バイト境界へ。詰めるのは JSON として無害な空白。
  const padding = (ALIGNMENT - ((headerBytes.byteLength + ALIGNMENT) % ALIGNMENT)) % ALIGNMENT;
  const headerLength = headerBytes.byteLength + padding;

  const out = new Uint8Array(ALIGNMENT + headerLength + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerLength), true);
  out.set(headerBytes, ALIGNMENT);
  out.fill(0x20, ALIGNMENT + headerBytes.byteLength, ALIGNMENT + headerLength);
  let cursor = ALIGNMENT + headerLength;
  for (const chunk of chunks) {
    out.set(chunk, cursor);
    cursor += chunk.byteLength;
  }
  return out;
};
