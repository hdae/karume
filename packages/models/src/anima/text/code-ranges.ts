/**
 * UTF-8 length needed by Anima's normalisation (the closed-interval table itself, the binary search
 * and reading it from the asset JSON live in `src/text/code-ranges.ts`).
 */

/** UTF-8 byte length of one code point. */
export const utf8Length = (cp: number): number =>
  cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
