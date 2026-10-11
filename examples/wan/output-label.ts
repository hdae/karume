/**
 * 出力先の名前に入れる印（examples/wan の台本 `main.ts` が使う）。台本はトップレベルで走り、テストから import
 * できないので、名前の組み立てだけをここへ切り出す。
 */

/** バイト列の sha256（小文字 16 進 — 出力先の名前に使う）。 */
export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");

/**
 * 出力先の名前の画像の印 `-img-<sha8>[-<fit>]`: ファイルのバイト列の sha256 の先頭 8 桁（パスを名前に入れない —
 * 同じ中身なら同じ名前）。fit は明示したときだけ足す — 省いたときの既定はパイプラインが持つので、台本に写すと既定が
 * 変わったときに名前と中身が食い違う（`main.ts` の steps と同じ理由。steps と違い fit はイベントが運ばないので、
 * 明示しない回は名前に入れない）。
 */
export const outputImageLabel = async (
  imageBytes: Uint8Array<ArrayBuffer>,
  fit: string | undefined,
): Promise<string> =>
  `-img-${(await sha256Hex(imageBytes)).slice(0, 8)}${fit === undefined ? "" : `-${fit}`}`;
