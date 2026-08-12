/**
 * 追加語彙（特殊トークン）の切り出し。
 *
 * **ファミリ非依存**（`tokenizers` の `AddedVocabulary` はモデル種別に依らず正規化の手前に
 * 挟まる）なので `src/text/` に置く。切り出された断片は正規化も pre-token 化も通さず id を
 * 直接出す、という契約もモデル種別に依らない。
 */

/**
 * 追加語彙を leftmost-longest で切り出す（正本の `AddedVocabulary` = AhoCorasick の
 * LeftmostLongest と同じ規則）。位置が同じなら長い方、位置が違えば左が勝つ。
 *
 * MUST: 長い順に並べて先頭一致で採るだけでは足りない — **各位置で全候補中の最長**を採る。
 */
export const splitAddedTokens = (
  text: string,
  added: readonly string[],
): { text: string; added: boolean }[] => {
  if (added.some((token) => token === "")) {
    // 空文字は全位置で一致して走査が進まない（無限ループ）。黙って読み飛ばすと切り出し規則
    // そのものが別物になるので落とす。
    throw new Error("追加語彙に空文字がある — 切り出しが進まない");
  }
  const out: { text: string; added: boolean }[] = [];
  let buffer = "";
  let i = 0;
  while (i < text.length) {
    let hit: string | undefined;
    for (const token of added) {
      if (text.startsWith(token, i) && (hit === undefined || token.length > hit.length)) {
        hit = token;
      }
    }
    if (hit === undefined) {
      buffer += text[i];
      i += 1;
      continue;
    }
    if (buffer !== "") {
      out.push({ text: buffer, added: false });
      buffer = "";
    }
    out.push({ text: hit, added: true });
    i += hit.length;
  }
  if (buffer !== "") out.push({ text: buffer, added: false });
  return out;
};
