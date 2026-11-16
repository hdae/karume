/**
 * The sha256 the workbench records of what a run read: the manifest of a distribution it opened
 * (both hosts' `openDistribution`), a container's part 0 and generated token ids (the local entries).
 */

/** Lower-case hex sha256 of `bytes` (WebCrypto, in a browser and under Node alike). */
export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> => {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
};
