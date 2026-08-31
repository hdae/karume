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
 *
 * 候補は**先頭 1 単位で束ねてから**引く。規則は変わらない（先頭が違えばその位置では
 * 一致しない）が、追加語彙が数千本ある資産（EmbeddingGemma は 6,415 本）では素朴な全走査が
 * 文字数 × 語彙数になり、長文 1 本で桁違いの時間を食う。
 */
export const splitAddedTokens = (
  text: string,
  added: readonly string[],
): { text: string; added: boolean }[] => {
  const byHead = new Map<string, string[]>();
  for (const token of added) {
    if (token === "") {
      // 空文字は全位置で一致して走査が進まない（無限ループ）。黙って読み飛ばすと切り出し規則
      // そのものが別物になるので落とす。
      throw new Error("追加語彙に空文字がある — 切り出しが進まない");
    }
    const bucket = byHead.get(token[0]);
    if (bucket === undefined) byHead.set(token[0], [token]);
    else bucket.push(token);
  }
  const out: { text: string; added: boolean }[] = [];
  let buffer = "";
  let i = 0;
  while (i < text.length) {
    let hit: string | undefined;
    for (const token of byHead.get(text[i]) ?? []) {
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
