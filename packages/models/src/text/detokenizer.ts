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
 *
 * ## 停止文字列の判定も同じ層に置く（{@link createStopStringFilter}）
 *
 * 停止**文字列**は「確定した文字列の列」の上でしか判定できず、しかもファミリ非依存なので席は
 * ここである。復号の状態機械とは**別の状態**（保留中の末尾 1 本）を持つ 2 本目の機械として置き、
 * {@link StreamingDetokenizer} の「確定した片だけ返す」契約には手を入れない。
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

/**
 * 停止**文字列**の判定で、1 回の {@link StopStringFilter.push} が確定させたぶん。
 *
 * `matched` が立った時点で、その push が渡した文字列は**停止文字列の手前までしか** `text` に
 * 入っていない（停止文字列そのものと、その後ろは捨てる）。
 */
export type StopStringChunk = {
  /** 出力してよい文字列（保留ぶんは含まない・出すものが無ければ空文字）。 */
  readonly text: string;
  /** 一致した停止文字列（`undefined` = まだ止まっていない）。 */
  readonly matched?: string;
};

/**
 * 確定済み文字列の列 → 停止文字列の手前で切った列（`push` / `finish` の状態機械）。
 *
 * 状態は**保留中の末尾 1 本**だけ。
 */
export type StopStringFilter = {
  /** 確定した片を 1 つ進める（{@link StreamingDetokenizer.push} の返り値をそのまま渡せる）。 */
  push(text: string): StopStringChunk;
  /** 列の終わり。保留していたぶんを確定させる（止まらずに終わった場合の尻尾）。 */
  finish(): string;
};

/**
 * 「停止文字列の**接頭辞**になりうる末尾」の長さ。
 *
 * これが保留の量である。停止文字列は token 境界を**跨いで**現れうるので、確定した文字列を
 * そのまま全部出すと「止めるべき文字列の一部を出し終えてから止まる」形になる。逆に固定長
 * （最長の停止文字列 − 1 文字）で保留すると、停止しないターンでも常に描画が遅れるので、
 * **本当に接頭辞になっている**ぶんだけを残す。
 */
const partialTail = (buffer: string, stopStrings: readonly string[]): number => {
  let longest = 0;
  for (const stop of stopStrings) {
    // 全体一致は呼び出し側が先に見ているので、ここで見るのは真の接頭辞（`length < stop.length`）。
    for (let length = Math.min(stop.length - 1, buffer.length); length > longest; length -= 1) {
      if (buffer.endsWith(stop.slice(0, length))) {
        longest = length;
        break;
      }
    }
  }
  return longest;
};

/**
 * 停止**文字列**の逐次判定（ADR 0083 追記 2026-09-02 の 2 層のうち上の層）。
 *
 * 停止 token は sequence 層が判定するが（token id は文字列を知らない層で完結する）、停止
 * **文字列**は復号の後にしか判定できない — 1 つの停止文字列が複数 token に割れることも、
 * 1 つの token が停止文字列の末尾と次の本文をまたぐこともあるためである。よって席はここ
 * （確定文字列の列の上）で、`byte_fallback` の run 持ち越しとは**別の保留**である
 * （{@link StreamingDetokenizer} は「確定した片だけ返す」契約のまま）。
 *
 * MUST: 空文字列と重複は fail loudly。空文字列は「常に一致する」= 1 文字も出せない指定で、
 * 重複は同じ条件を 2 度書いた以上の意味を持てない（どちらも呼び手の取り違えの徴候）。
 */
export const createStopStringFilter = (stopStrings: readonly string[]): StopStringFilter => {
  // MUST: 発行時に写す（走行中に配列を書き換えられると停止条件が途中で変わる）。
  const stops = [...stopStrings];
  const declared = new Set<string>();
  stops.forEach((stop, index) => {
    if (stop === "") throw new Error(`stopStrings[${index}] が空文字列`);
    if (declared.has(stop)) throw new Error(`stopStrings に ${JSON.stringify(stop)} が 2 度出る`);
    declared.add(stop);
  });

  let held = "";
  return {
    push(text: string): StopStringChunk {
      const buffer = held + text;
      // 一致は**いちばん手前**を採る（複数の停止文字列が同時に見つかっても、実際に流れを
      // 切ったのは先に現れた方）。同じ位置に複数あれば先に宣言された方。
      let at = -1;
      let matched: string | undefined;
      for (const stop of stops) {
        const index = buffer.indexOf(stop);
        if (index >= 0 && (at < 0 || index < at)) {
          at = index;
          matched = stop;
        }
      }
      if (matched !== undefined) {
        held = "";
        return { text: buffer.slice(0, at), matched };
      }
      const keep = partialTail(buffer, stops);
      held = buffer.slice(buffer.length - keep);
      return { text: buffer.slice(0, buffer.length - keep) };
    },
    finish(): string {
      const tail = held;
      held = "";
      return tail;
    },
  };
};

/** 列を一括で復号する（逐次と同じ状態機械を通す — 2 実装を持たない）。 */
export const detokenize = (source: DetokenizerSource, ids: Iterable<number>): string => {
  const detokenizer = new StreamingDetokenizer(source);
  let out = "";
  for (const id of ids) out += detokenizer.push(id);
  return out + detokenizer.finish();
};
