/**
 * Python's `str.strip()` (no argument) — what Jinja's `trim` filter runs, so a chat template that
 * trims a message (`content|trim`) needs exactly this, not JS's `String.prototype.trim()`.
 *
 * It is **family-independent** (gemma4's chat format and the Qwen3.5 chat prompt both trim with
 * it), so it lives in `src/text/`.
 */

/**
 * The code points for which Python's `str.isspace()` is true (29 of them).
 *
 * The set differs from JS's `String.prototype.trim()` in exactly two points: Python treats
 * `\x1c`–`\x1f` and `\x85` as whitespace, and does not treat U+FEFF as whitespace. The difference is
 * exotic, but what comes out is **a different id sequence, not an exception**, so the set is ported
 * explicitly. All 29 are in the BMP, so testing UTF-16 units is the same as testing code points.
 */
const isPythonSpace = (code: number): boolean =>
  (code >= 0x09 && code <= 0x0d) || // \t \n \v \f \r
  (code >= 0x1c && code <= 0x20) || // file / group / record / unit separators + space
  code === 0x85 ||
  code === 0xa0 ||
  code === 0x1680 ||
  (code >= 0x2000 && code <= 0x200a) ||
  code === 0x2028 ||
  code === 0x2029 ||
  code === 0x202f ||
  code === 0x205f ||
  code === 0x3000;

/** Python's `str.strip()` = Jinja's `trim`. */
export const pythonStrip = (text: string): string => {
  let start = 0;
  let end = text.length;
  while (start < end && isPythonSpace(text.charCodeAt(start))) start += 1;
  while (end > start && isPythonSpace(text.charCodeAt(end - 1))) end -= 1;
  return text.slice(start, end);
};
