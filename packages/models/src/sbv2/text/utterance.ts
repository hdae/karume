/**
 * SBV2 の入力語彙（発話 = モーラ列 + 語アライメント）と、句構造からの変換。
 *
 * 責務境界（0.6.0 の再裁定）: テキスト解析（辞書・修正辞書）は**呼び手の責務**で、この
 * パッケージは解析済みの構造だけを受ける。したがってここの型は全て karume 所有で、解析器の
 * 型を import しない — {@link Sbv2Phrases} は「上流の解析結果が**構造的に**満たす形」を
 * 書いたものであって、特定の解析器への依存ではない。
 *
 * MUST: 発話は**フラット**（句の入れ子を持たない）。句は「核 → トーン」を決めるためだけの
 * 構造で、SBV2 が読むのはモーラごとの 2 値トーンそのものなので、句を残すと「核とトーンの
 * どちらが正か」という二重表現になる。{@link toSbv2Utterance} が核を展開しきってから渡す。
 */

import { Sbv2InputError } from "../errors.ts";

/**
 * 語アライメントの 1 要素（1 語、または実在記号 1 文字）。
 *
 * `phones` は**その語が生む音素列**（両端 PAD を含まない）。記号要素は正規形 1 個
 * （例 `["!"]`）で、`surface` は正規化後テキスト上の生の 1 文字（例 `"！"`）。
 */
export type Sbv2Word = {
  /** 語の表層（DeBERTa のトークン数と BERT 入力テキストの出所）。 */
  readonly surface: string;
  /** その語が生む音素列（両端 PAD を含まない）。 */
  readonly phones: readonly string[];
};

/**
 * 発話の 1 モーラ。**編集面はここ**（`tone` を直せばアクセントが動く）。
 *
 * MUST: 音素は `consonant` / `vowel` から導く（`consonant` があれば `[consonant, vowel]`、
 * 無ければ `[vowel]`）。派生した音素列を別欄で持たせない — 直して戻す往復で古い派生欄が
 * 同梱され、「無視する（編集を黙って捨てる）」か「落とす（正当な往復が通らない）」の二択に
 * なる。
 */
export type Sbv2Mora = {
  /** カタカナ 1 モーラ（拗音は 1 モーラ）。表示・突合用。 */
  readonly kana: string;
  /** 子音音素（例 `"ky"`）。母音のみ・撥音・促音のモーラには無い。 */
  readonly consonant?: string;
  /**
   * 母音音素（`"a"`…`"o"` / 撥音 `"N"` / 促音 `"q"`）。長音は直前母音に解決済み。
   *
   * NOTE: **音素そのもの**が入る（促音は `"q"`）。上流の解析結果が促音を `"cl"` で表す場合の
   * 畳み込みは {@link toSbv2Utterance} が済ませる。
   */
  readonly vowel: string;
  /** 2 値トーン（0 = 低・1 = 高）。子音・母音とも同一トーンになる。 */
  readonly tone: 0 | 1;
  /**
   * このモーラの**直後**にテキスト上実在した記号の正規形列（出現順）。ポーズからの合成は
   * しない（実在しない記号は音素にしない — 参照実装 g2p と同方針）。
   */
  readonly punctuations?: readonly string[];
};

/**
 * 1 回の合成に渡す発話（{@link Sbv2Pipeline.generate} の第 1 引数）。
 *
 * `moras` が編集面で、`words` は**読み取り専用**（BERT 入力テキストと word2ph の出所）。
 * `words` を書き換えると音素列と語割りが食い違い、合成は入口の門で落ちる — 読みを変えたい
 * ときは解析側（呼び手の辞書・修正辞書）からやり直す。
 */
export type Sbv2Utterance = {
  /** 先頭のモーラより前に実在した記号の正規形列（記号だけの入力では全てここに入る）。 */
  readonly leadingPunctuations: readonly string[];
  /** 発話のモーラ列（句の境界は残らない — モジュール doc の MUST）。 */
  readonly moras: readonly Sbv2Mora[];
  /** 語アライメント（読み取り専用 — BERT 入力テキストと word2ph の出所）。 */
  readonly words: readonly Sbv2Word[];
};

/**
 * 句構造の解析結果（{@link toSbv2Utterance} の入力）。
 *
 * 「アクセント句とその核」で読みを返す解析器の出力が**構造的に**満たす形を書いてある
 * （余分な欄を持っていても通る）。karume 側はこの部分集合しか読まない。
 */
export type Sbv2Phrases = {
  readonly result: {
    /** 先頭句より前に実在した記号の正規形列。 */
    readonly leadingPunctuations: readonly string[];
    readonly accentPhrases: readonly {
      readonly moras: readonly {
        readonly kana: string;
        readonly consonant?: string;
        readonly vowel: string;
      }[];
      /** アクセント核（1-origin・0 = 平板）。上端は {@link toSbv2Utterance} が丸める。 */
      readonly accentNucleus: number;
      /** 句の直後に実在した記号の正規形列（出現順）。 */
      readonly punctuations: readonly string[];
    }[];
  };
  /** 同一解析から採った語アライメント（音素列と語割りが整合している必要がある）。 */
  readonly words: readonly Sbv2Word[];
};

/**
 * アクセント核を句内のモーラ 2 値トーンへ展開する。
 *
 * 規則（上流 SBV2 の g2p と同一）:
 *
 * - 平板（k = 0）: 1 モーラ目だけ 0、以降 1。
 * - 頭高（k = 1）: 1 モーラ目だけ 1、以降 0。
 * - 中高・尾高（k > 1）: 1 モーラ目 0、2..k が 1、k+1 以降 0。
 *
 * MUST: 核の**上端**はモーラ数へ丸める（範囲外核は辞書差・修正辞書由来で普通に出る）。
 * 丸めた値は同じトーン列を生むので音は変わらない。**下端**は丸めない — 負の核は「核の
 * 意味を取り違えた入力」で、黙って平板へ倒すと指定と別のアクセントで合成される。
 */
const phraseTones = (accentNucleus: number, moraCount: number, where: string): (0 | 1)[] => {
  if (!Number.isInteger(accentNucleus) || accentNucleus < 0) {
    throw new Sbv2InputError(
      `${where}.accentNucleus = ${accentNucleus} が 0 以上の整数でない` +
        "（0 = 平板、1-origin の核位置、モーラ数 = 尾高）",
    );
  }
  const k = Math.min(accentNucleus, moraCount);
  if (k === 0) return Array.from({ length: moraCount }, (_, index) => (index === 0 ? 0 : 1));
  if (k === 1) return Array.from({ length: moraCount }, (_, index) => (index === 0 ? 1 : 0));
  return Array.from({ length: moraCount }, (_, index) => (index >= 1 && index + 1 <= k ? 1 : 0));
};

/** 上流が促音を `"cl"` で表す場合の畳み込み（`Sbv2Mora.vowel` は音素そのもの）。 */
const toVowelPhone = (vowel: string): string => (vowel === "cl" ? "q" : vowel);

/**
 * 句構造の解析結果を発話へ落とす（純関数・GPU 不要）。
 *
 * 核はここで {@link phraseTones} によりモーラごとの 2 値トーンへ展開しきる。句の直後の記号は
 * **その句の末尾モーラ**へ付け替える（フラット化しても音素の並びは変わらない）。句より前の
 * 記号は {@link Sbv2Utterance.leadingPunctuations} へそのまま移す。
 *
 * NOTE: モーラを 1 つも持たない句の記号は、直前のモーラ（無ければ先頭の記号列）へ寄せる。
 * 音素列としての位置は同じなので、寄せ先だけが変わる。
 */
export const toSbv2Utterance = (phrases: Sbv2Phrases): Sbv2Utterance => {
  const leadingPunctuations = [...phrases.result.leadingPunctuations];
  // 記号を後から足すので、組み立て中だけ可変にする（返す前に読み取り専用の面へ載せる）。
  const moras: {
    kana: string;
    consonant?: string;
    vowel: string;
    tone: 0 | 1;
    punctuations?: string[];
  }[] = [];

  for (const [index, phrase] of phrases.result.accentPhrases.entries()) {
    const tones = phraseTones(
      phrase.accentNucleus,
      phrase.moras.length,
      `accentPhrases[${index}]`,
    );
    for (const [at, mora] of phrase.moras.entries()) {
      moras.push({
        kana: mora.kana,
        ...(mora.consonant === undefined ? {} : { consonant: mora.consonant }),
        vowel: toVowelPhone(mora.vowel),
        tone: tones[at],
      });
    }
    if (phrase.punctuations.length === 0) continue;
    const last = moras[moras.length - 1];
    if (last === undefined) {
      leadingPunctuations.push(...phrase.punctuations);
      continue;
    }
    last.punctuations = [...(last.punctuations ?? []), ...phrase.punctuations];
  }

  return {
    leadingPunctuations,
    moras,
    words: phrases.words.map((word) => ({ surface: word.surface, phones: [...word.phones] })),
  };
};
