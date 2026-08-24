/**
 * 下書き（句 / モーラ構造）の語彙と、外から戻された下書きを検める門。
 *
 * 責務境界: yomi の `FrontendResult` は**モデル非依存の解析結果**で、SBV2 が読まない欄
 * （`normalizedText` / `pauseAfter` / `Mora.devoiced`）まで持つ。ここで karume 所有の
 * {@link Sbv2Prosody} へ落とすのは、公開面に「渡せるのに効かない欄」を作らないため —
 * 外へ出す構造は **SBV2 が実際に読む欄だけ**にする（`toSbv2PhoneTone` が読むのと同じ 3 欄）。
 *
 * NOTE: `OverlayEntry` の素通し（ADR 0072 決定 3）はそのまま。あちらは検証の正本が yomi 側に
 * あるので別名を作らない方が正しく、こちらは逆に「SBV2 が読む部分集合」であることが情報。
 */

import type { FrontendResult } from "@hdae/yomi";
import { Sbv2InputError } from "../errors.ts";

/** 下書きの 1 モーラ（`kana` は表示・突合用、`consonant` / `vowel` が音素になる）。 */
export type Sbv2Mora = {
  /** カタカナ 1 モーラ（拗音は 1 モーラ）。 */
  readonly kana: string;
  /** 子音音素（例 `"ky"`）。母音のみ・撥音・促音のモーラには無い。 */
  readonly consonant?: string;
  /** 母音音素（`"a"`…`"o"` / 撥音 `"N"` / 促音 `"cl"`）。長音は直前母音に解決済み。 */
  readonly vowel: string;
};

/** 下書きの 1 アクセント句。 */
export type Sbv2AccentPhrase = {
  readonly moras: readonly Sbv2Mora[];
  /**
   * アクセント核（1-origin・0 = 平板）。**`0..moras.length` の正規形**で出し、その範囲外を
   * 受けたら落とす。
   *
   * NOTE: 2 値トーンでは平板（0）と尾高（= モーラ数）が同じ列になる（`moraTones`）ので、
   * 呼び手側で平板を尾高へ潰していても音は変わらない。
   */
  readonly accentNucleus: number;
  /** 句の直後にテキスト上実在した記号の正規形列（出現順）。 */
  readonly punctuations: readonly string[];
};

/**
 * 編集して戻せる下書き（`analyzeProsody` が返し、`generate` が受ける形）。
 *
 * MUST: ここに派生欄（音素列・トーン列）を持たせない。核を直して戻す往復で派生欄が古いまま
 * 同梱され、「無視する（編集を黙って捨てる）」か「落とす（正当な往復が通らない）」の二択に
 * なる。派生は常にこの構造から導出する。
 */
export type Sbv2Prosody = {
  /** 先頭句より前に実在した記号の正規形列（記号だけの入力では全てここに入る）。 */
  readonly leadingPunctuations: readonly string[];
  readonly phrases: readonly Sbv2AccentPhrase[];
};

/**
 * yomi の解析結果を下書きへ落とす。
 *
 * MUST: 核の**上端**を `moras.length` へクランプして出す（`Math.min` — 下端は見ていない。
 * 負の核は実測で出ておらず、出たら受理側の {@link assertProsodyShape} が `0..moras.length` の
 * 対称な範囲検査で落とす）。yomi の `moraTones` は範囲外核（辞書差・オーバーレイ由来）を尾高
 * 相当へ黙ってクランプしてトーンを作るので、生値のまま下書きに載せると「解析どおりの下書きを
 * 戻したのに範囲検査で落ちる」往復不能が起きる。クランプ後の値は同じトーン列を生む
 * （音は変わらない）。
 */
export const toSbv2Prosody = (result: FrontendResult): Sbv2Prosody => ({
  leadingPunctuations: [...result.leadingPunctuations],
  phrases: result.accentPhrases.map((phrase) => ({
    moras: phrase.moras.map((mora) => ({
      kana: mora.kana,
      ...(mora.consonant === undefined ? {} : { consonant: mora.consonant }),
      vowel: mora.vowel,
    })),
    accentNucleus: Math.min(phrase.accentNucleus, phrase.moras.length),
    punctuations: [...phrase.punctuations],
  })),
});

/**
 * 外から戻された下書きの自己整合を検める（音素へ落とす**前**の門）。
 *
 * MUST: 核の範囲をここで見る。`moraTones` は範囲外を黙ってクランプするので、通してしまうと
 * 「指定した核とは別のアクセントで合成される」沈黙誤値になる。
 */
export const assertProsodyShape = (prosody: Sbv2Prosody): void => {
  for (const [index, phrase] of prosody.phrases.entries()) {
    const { accentNucleus, moras } = phrase;
    if (!Number.isInteger(accentNucleus) || accentNucleus < 0 || accentNucleus > moras.length) {
      throw new Sbv2InputError(
        `prosody.phrases[${index}].accentNucleus = ${accentNucleus} が 0..${moras.length}` +
          " の範囲外（0 = 平板、1-origin の核位置、モーラ数 = 尾高）",
      );
    }
  }
};

/**
 * 戻された下書きが「解析と同じ音素列」を生むことを検める門。**この席の本体**。
 *
 * 長さだけを見ると、梱包規則（両端 PAD・モーラ → 音素の展開数・記号 tone 0・句ごとの
 * 立ち上がり）を外部で再実装した結果が上流とズレても素通りする — 長さは合うのに音だけ崩れる。
 * 位置ごとの内容一致まで見れば、モーラの読み替え・記号の増減・text と下書きの取り違えが
 * 全部ここで止まる。
 *
 * MUST: 音素列は**常に解析由来**を採る（この門を通れば両者は同一なので、下書きが動かせるのは
 * トーンだけ）。下書き側の音素列を採ると、word2ph の `sum(word2ph) === P` を破った要求が
 * DeBERTa を回した後の shape 不一致（原因が遠い失敗）として出る。
 */
export const assertProsodyPhones = (
  analyzed: readonly string[],
  edited: readonly string[],
): void => {
  if (analyzed.length !== edited.length) {
    throw new Sbv2InputError(
      `prosody から組んだ音素数 ${edited.length} が解析の音素数 ${analyzed.length} と違う` +
        "（音素数が変わる編集〈モーラの読み替え・記号の増減〉は受け付けない — 読みの変更は" +
        " overlay で解析からやり直す）",
    );
  }
  for (const [index, phone] of edited.entries()) {
    if (phone !== analyzed[index]) {
      throw new Sbv2InputError(
        `prosody から組んだ音素[${index}] = ${JSON.stringify(phone)} が解析の` +
          ` ${JSON.stringify(analyzed[index])} と違う（下書きは同じ text・同じ overlay で` +
          "採ったものを戻す — 動かせるのは accentNucleus だけ）",
      );
    }
  }
};
