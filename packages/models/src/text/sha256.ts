/**
 * Synchronous SHA-256 (FIPS 180-4). It checks fixed values inside asset JSON (such as translated
 * regular expressions) against the expected digest before use.
 *
 * It is implemented here because the Web-standard `crypto.subtle.digest` is asynchronous and cannot
 * be used inside asset parsing (synchronous — callers use it synchronously in both tests and
 * construction). The values checked are KB-sized; speed does not matter.
 *
 * MUST: every module is free of side effects (the constants below are pure constant construction
 * only — ADR 0117 addendum 2026-10-02).
 */

/**
 * Round constants K (the top 32 bits of the fractional parts of the cube roots of the first 64
 * primes).
 */
const ROUND_CONSTANTS: readonly number[] = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/**
 * Initial hash value H(0) (the top 32 bits of the fractional parts of the square roots of the
 * first 8 primes).
 */
const INITIAL_HASH: readonly number[] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const BLOCK_BYTES = 64;

const rotateRight = (value: number, bits: number): number =>
  (value >>> bits) | (value << (32 - bits));

/** Returns the SHA-256 of a byte array as 64 lowercase hex characters. */
export const sha256HexSync = (bytes: Uint8Array<ArrayBuffer>): string => {
  // Padding: 0x80 → zero fill → bit length in the last 8 bytes (big-endian).
  const blocks = Math.ceil((bytes.length + 9) / BLOCK_BYTES);
  const padded = new Uint8Array(blocks * BLOCK_BYTES);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  const bitLength = bytes.length * 8;
  view.setUint32(padded.length - 8, Math.floor(bitLength / 2 ** 32));
  view.setUint32(padded.length - 4, bitLength >>> 0);

  const hash = [...INITIAL_HASH];
  const schedule = new Uint32Array(64);
  for (let offset = 0; offset < padded.length; offset += BLOCK_BYTES) {
    for (let t = 0; t < 16; t += 1) schedule[t] = view.getUint32(offset + 4 * t);
    for (let t = 16; t < 64; t += 1) {
      const w15 = schedule[t - 15];
      const w2 = schedule[t - 2];
      const sigma0 = rotateRight(w15, 7) ^ rotateRight(w15, 18) ^ (w15 >>> 3);
      const sigma1 = rotateRight(w2, 17) ^ rotateRight(w2, 19) ^ (w2 >>> 10);
      schedule[t] = schedule[t - 16] + sigma0 + schedule[t - 7] + sigma1;
    }
    let [a, b, c, d, e, f, g, h] = hash;
    for (let t = 0; t < 64; t += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choice = (e & f) ^ (~e & g);
      const temp1 = (h + sum1 + choice + ROUND_CONSTANTS[t] + schedule[t]) >>> 0;
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (sum0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }
    [a, b, c, d, e, f, g, h].forEach((value, index) => {
      hash[index] = (hash[index] + value) >>> 0;
    });
  }
  return hash.map((word) => word.toString(16).padStart(8, "0")).join("");
};
