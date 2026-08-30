/**
 * 資産 JSON（外部境界）の値域門 — 家族横断で 1 本に寄せたもの。
 *
 * 配布資産は exporter が焼くが、`fromPretrained(ref)` は任意の repo を指せるので、資産 JSON は
 * **信用できない外部入力**として扱う。ここが縛るのは「壊れた表が例外にならず、静かに別の
 * トークン列 / 別の埋め込み行へ化ける」形（沈黙誤値）だけ:
 *
 *   ・行番号 = id の表に重複行があると `Map` の後勝ちで**先の id が引けなくなる**
 *   ・鍵つき写像の重複鍵も同じく後勝ちで、正規化 / 文字分類だけが静かに変わる
 *   ・コードポイントでない値が区間表や合成鍵に入ると、二分探索・鍵の一意性が黙って崩れる
 *   ・長さ / 周波数の欄が 0 や負でも、切り詰めや突合は「別の正常」に見える形で通る
 *
 * 門の粒度は家族ごとにばらついていた（最も厳しいのが sbv2 の `parseRanges` / `symbols.ts`、
 * 最も緩いのが anima のテキスト資産）ので、**厳しい側に揃えて**ここへ寄せてある。
 */

/** Unicode コードポイントの上限（区間表も合成鍵もコードポイントを鍵にする）。 */
export const MAX_CODE_POINT = 0x10FFFF;

/**
 * コードポイントとして受ける数値。
 *
 * MUST: `typeof number` だけで通さない。区間表は二分探索で引かれ、写像は
 * `cp * 0x110000 + cp` のような合成鍵で引かれる — どちらも非整数・域外の値を**例外にせず**
 * 「当たらない / 別の鍵に当たる」へ倒すので、文字分類と正規化だけが静かに変わる。
 */
export const assertCodePoint = (cp: number, label: string): void => {
  if (!Number.isInteger(cp) || cp < 0 || cp > MAX_CODE_POINT) {
    throw new Error(`${label}: コードポイントが 0..${MAX_CODE_POINT} の整数でない（${cp}）`);
  }
};

/**
 * 長さ・件数・周波数として受ける数値（正の安全整数）。
 *
 * MUST: 上限も見る。2^53 を超える値は加算・比較が精度を落として黙って別の値になり、
 * `slice` の上限として使えば「切り詰めが一度も効かない」状態が正常値の顔で通る。
 */
export const asPositiveInteger = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label}: 正の安全整数でない（${String(value)}）`);
  }
  return value;
};

/**
 * スコアなど、実数として受ける数値。
 *
 * MUST: 有限性まで見る。`NaN` / `±Infinity` は最小値・最大値の畳み込みを汚染し、Viterbi の
 * 未知ノード重みや探索幅が「全経路が同値」へ潰れる（例外は出ない）。
 */
export const asFiniteNumber = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label}: 有限の数値でない（${String(value)}）`);
  }
  return value;
};

/**
 * 行番号 = id の表に重複行が無いことを検める。
 *
 * MUST: 行数の突合だけでは足りない — 重複は行数を変えないので、`vocabCount` / `scores` の
 * 本数と突き合わせても通ってしまう。通した先では `Map` の後勝ちで先の id が引けなくなり、
 * embedding gather は**合法な別の行**を引く（例外にならない配布破損）。
 */
export const assertUniqueLines = (lines: readonly string[], label: string): void => {
  const firstAt = new Map<string, number>();
  for (const [index, line] of lines.entries()) {
    const seen = firstAt.get(line);
    if (seen !== undefined) {
      throw new Error(
        `${label}: 行 ${index} が行 ${seen} と同じトークン ${JSON.stringify(line)}` +
          "（行番号 = id なので重複は配布の破損）",
      );
    }
    firstAt.set(line, index);
  }
};

/**
 * 写像へ 1 件入れる。既に同じ鍵があれば落とす（`Map.set` の後勝ちを禁じる）。
 *
 * MUST: 重複鍵を「正規化」して救わない。BPE の merge 順・追加語彙・正規化表はどれも鍵の
 * 一意性が前提で、後勝ちで上書きされると分割規則だけが静かに変わる。
 */
export const setUnique = <K, V>(map: Map<K, V>, key: K, value: V, label: string): void => {
  if (map.has(key)) {
    throw new Error(
      `${label}: 鍵 ${JSON.stringify(key)} が重複している（後勝ちで黙って上書きされる）`,
    );
  }
  map.set(key, value);
};
