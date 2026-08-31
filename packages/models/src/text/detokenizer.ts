/**
 * byte_fallback を持つ tokenizer の**逐次**復号（`push(id) → 確定した文字列`）。
 *
 * **ファミリ非依存**（`ByteFallback` + `Fuse` のデコーダ鎖はモデル種別に依らない）なので
 * `src/text/` に置く。id → 綴りの解決と特殊トークンの扱いはファミリ側の責務で、ここは
 * {@link DetokenizerSource} 経由で受け取る。
 *
 * 正本は `tokenizers` の `ByteFallback::decode_chain`:
 *
 *   `<0xHH>` の連なり（**run**）を溜め、run が切れた時点で UTF-8 として復号する。
 *   run 全体が正しい UTF-8 なら 1 つの文字列に、そうでなければ **1 バイトにつき 1 個**の
 *   U+FFFD になる（先頭の妥当な部分だけを取り出す形にはならない — だから run は最後まで
 *   持ち越すしかない）。
 *
 * MUST: **run 以外は 1 トークンも溜めない**。`finish()` まで全部溜めてから返す「偽
 * streaming」は、ADR 0083 決定 2 の `AsyncIterable` の 1 反復を無意味にする（消費者から見て
 * 途中経過が出ない）。この点はテストが直接縛る。
 *
 * NOTE: 逆に run の**途中**では出せない（「3 バイト揃ったので 1 文字出す」ができない）—
 * 後続の 1 バイトで run 全体が不正になれば、先頭の妥当な部分まで置換文字へ変わるため。
 * 実運用で効く場面は限られる: 語彙に載っている文字（日本語も含む）は byte run を作らず、
 * run になるのは語彙外の記号・絵文字だけで、長さも数バイトで閉じる。
 *
 * MUST: skip 対象の特殊トークンは **run を切らない**。正本は「id 列 → 綴り列」を作る段で
 * skip し、デコーダ鎖はその後で走るので、skip されたトークンを挟んだ byte は隣接する
 * （`[<0xE3>, <eos>, <0x81>, <0x82>]` は skip 有りで「あ」1 文字になる）。
 */

/** id を綴りへ解決する口（ファミリ側が持つ知識をここへ渡す）。 */
export type DetokenizerSource = {
  /** id が byte_fallback 語彙なら byte 値、そうでなければ `undefined`。 */
  readonly byteOf: (id: number) => number | undefined;
  /**
   * id の出力綴り（デコーダ鎖のうちトークン単位の変換まで適用済み）。skip 対象の特殊
   * トークンだけが `undefined` を返す。
   *
   * MUST: **未知の id は `undefined` を返さず落とす**（正本は黙って読み飛ばすが、こちらは
   * 資産と id 列の食い違いを沈黙させない — 横断不変条件の fail loudly）。
   */
  readonly textOf: (id: number) => string | undefined;
};

/** `push(id) → 確定した文字列` の状態機械。状態は byte run 1 本だけ。 */
export class StreamingDetokenizer {
  readonly #source: DetokenizerSource;
  /**
   * MUST: BOM を剥がさせない（`ignoreBOM: true`）。既定の `TextDecoder` は先頭の U+FEFF を
   * 黙って捨てるが、正本の `String::from_utf8` は捨てない — `EF BB BF` の run だけが
   * 静かに消える。
   */
  readonly #decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
  #pending: number[] = [];

  constructor(source: DetokenizerSource) {
    this.#source = source;
  }

  /** 1 トークン進める。返るのは**この push で確定した**ぶんだけ（無ければ空文字）。 */
  push(id: number): string {
    const text = this.#source.textOf(id);
    if (text === undefined) return "";
    const byte = this.#source.byteOf(id);
    if (byte !== undefined) {
      this.#pending.push(byte);
      return "";
    }
    return this.#flush() + text;
  }

  /** 列の終わり。持ち越していた byte run を確定させる。 */
  finish(): string {
    return this.#flush();
  }

  #flush(): string {
    if (this.#pending.length === 0) return "";
    const bytes = Uint8Array.from(this.#pending);
    this.#pending = [];
    try {
      return this.#decoder.decode(bytes);
    } catch {
      // run 全体が不正なら 1 バイト 1 個の置換文字（正本の `String::from_utf8` の失敗経路）。
      return "�".repeat(bytes.length);
    }
  }
}

/** 列を一括で復号する（逐次と同じ状態機械を通す — 2 実装を持たない）。 */
export const detokenize = (source: DetokenizerSource, ids: Iterable<number>): string => {
  const detokenizer = new StreamingDetokenizer(source);
  let out = "";
  for (const id of ids) out += detokenizer.push(id);
  return out + detokenizer.finish();
};
