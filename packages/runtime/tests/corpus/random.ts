/**
 * The seeded random words of the regression corpus.
 *
 * One xorshift128 stream per (seed, label): the label is an initializer name for a weight and a
 * `run/step/input` path for an input, so a tensor's words depend on its own name only — the same
 * initializer gets the same bytes in the full graph and in a cut of it. Integer arithmetic only, so
 * the words are the same on every machine (the generator's sha256 pins rely on it).
 */

/** The state of one stream (four u32 words, never all zero). */
export type RandomStream = { readonly state: Uint32Array<ArrayBuffer> };

/** FNV-1a over the UTF-16 code units of `text`, started from `seed`. */
const fnv1a = (seed: number, text: string): number => {
  let hash = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
};

/** One splitmix32 output: spreads a hash over all 32 bits before it seeds xorshift. */
const splitmix32 = (value: number): number => {
  let z = (value + 0x9e3779b9) >>> 0;
  z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
  z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
  return (z ^ (z >>> 16)) >>> 0;
};

/** Opens the stream of `label` under `seed` (a u32). */
export const randomStream = (seed: number, label: string): RandomStream => {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error(
      `corpus seed ${seed} is not an integer in [0, 2^32): give the entry a u32 seed`,
    );
  }
  const hash = fnv1a(seed, label);
  const state = new Uint32Array(4);
  for (let i = 0; i < 4; i += 1) state[i] = splitmix32((hash + Math.imul(i, 0x9e3779b9)) >>> 0);
  if (state[0] === 0 && state[1] === 0 && state[2] === 0 && state[3] === 0) state[0] = 1;
  return { state };
};

/** Writes the next `count` words of the stream into `out[0..count)` (xorshift128). */
export const fillWords = (
  stream: RandomStream,
  out: Uint32Array<ArrayBuffer>,
  count: number = out.length,
): void => {
  const { state } = stream;
  let a = state[0];
  let b = state[1];
  let c = state[2];
  let d = state[3];
  for (let i = 0; i < count; i += 1) {
    const t = a ^ (a << 11);
    a = b;
    b = c;
    c = d;
    d = (d ^ (d >>> 19) ^ (t ^ (t >>> 8))) >>> 0;
    out[i] = d;
  }
  state[0] = a;
  state[1] = b;
  state[2] = c;
  state[3] = d;
};

/** A word as a value in [-1, 1). */
export const signedUnit = (word: number): number => word * 2 ** -31 - 1;
