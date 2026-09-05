/**
 * JP-Extra の ID 化規則（`symbols.json` 資産の読み取りと、それを使った ID 列の組み立て）。
 *
 * ## MUST: 定数表をここに書かない
 *
 * 音素記号表・tone 基点・言語 ID・add_blank の挿入値は **`style_bert_vits2` の実物から
 * 引いた資産**（`tools/export-recipes/sbv2/demo.py` の `assets` サブコマンドが出す `symbols.json`）
 * だけを正とする。
 * これらは多言語版と JP-Extra で同じに見えて、ずれても **shape は合ったまま音だけが
 * 壊れる**（記号表の並びが 1 つずれれば別の音素、tone 基点がずれれば別の埋め込み行）。
 * TS 側に写した定数表を持つと、その齟齬を検出する手段が消える。
 *
 * ## 参照実装との対応
 *
 * - `phonesToIds` / `applyToneStart` / `languageIds` = `nlp.cleaned_text_to_sequence`
 * - `intersperse` = `models.commons.intersperse`（`infer.get_text` の add_blank 分岐）
 * - `addBlankWord2ph` = 同分岐の `word2ph[i] *= 2; word2ph[0] += 1`
 */

import { asPositiveInteger } from "../../text/asset-gates.ts";
import { Sbv2InputError } from "../errors.ts";

/**
 * 相対位置の添字表を作るためのバケット規則（DeBERTa の config 由来）。
 *
 * `buildRelPosTables` の引数そのもの。近傍は線形・遠方は対数で圧縮した添字を作る式で、
 * `positionBuckets` がバケット総数（= 添字の中心オフセット）、`maxPosition` が対数圧縮の
 * 基準になる最大距離。
 */
type Sbv2BertRelPos = {
  /** `position_buckets`（char-wwm は 256）。 */
  readonly positionBuckets: number;
  /** `max_relative_positions`（1 未満なら `max_position_embeddings` — char-wwm は 512）。 */
  readonly maxPosition: number;
};

/** 実行時ノブ（`style_bert_vits2/constants.py` の既定値が資産経由で届く）。 */
export type Sbv2Knobs = {
  /** sdp と dp の混合比（`logw = sdp·r + dp·(1−r)`）。 */
  readonly sdpRatio: number;
  /** z_p のノイズ倍率。 */
  readonly noiseScale: number;
  /** sdp reverse のノイズ倍率（`z_noise` に乗せる）。 */
  readonly noiseScaleW: number;
  /** 継続長のスケール（大きいほどゆっくり）。 */
  readonly lengthScale: number;
};

/** JP-Extra の ID 化規則とモデル定数（`symbols.json` の内容）。 */
export type JpExtraRules = {
  /** 音素記号表。添字が `enc_p.emb` の行番号。 */
  readonly symbols: readonly string[];
  /** 音素記号 → ID。 */
  readonly symbolToId: ReadonlyMap<string, number>;
  /** PAD 記号（音素列の両端に置く。`SYMBOLS[0]`）。 */
  readonly pad: string;
  /** 音素として受け付ける正規形句読点。 */
  readonly punctuations: ReadonlySet<string>;
  /** JP のトーン基点（`LANGUAGE_TONE_START_MAP["JP"]`）。given_tone に加算する。 */
  readonly toneStart: number;
  /** JP の言語 ID（`LANGUAGE_ID_MAP["JP"]`）。 */
  readonly languageId: number;
  /** tone 埋め込みの行数（範囲検査に使う）。 */
  readonly numTones: number;
  /** language 埋め込みの行数（範囲検査に使う）。 */
  readonly numLanguages: number;
  /** add_blank で挟む ID（3 系列とも同じ値 — 資産生成側がソースから抜いて確認済み）。 */
  readonly blankId: number;
  /** 出力 WAV のサンプリング周波数。 */
  readonly samplingRate: number;
  /** 1 フレームあたりのサンプル数（`audio 長 = hopLength × フレーム数` の検算に使う）。 */
  readonly hopLength: number;
  /** BERT 特徴に使う hidden_states の末尾からの位置（配布グラフは 1 本出しなので 1）。 */
  readonly bertHiddenFromEnd: number;
  /**
   * 相対位置の添字表を作るためのバケット規則（DeBERTa の config 由来）。
   *
   * 表そのものはグラフ入力で、実長ぶんをホストが作る（`rel-pos-tables.ts`）。値を写経せず
   * 資産から引くのは ADR 0039 決定 3 と同じ規律 — モデルが変われば規則も変わるのに、
   * shape は合ったまま**別の位置埋め込みを gather する**（沈黙誤値）。
   */
  readonly bertRelPos: Sbv2BertRelPos;
  /**
   * 実行時ノブの既定値（**任意**）。
   *
   * NOTE: `Sbv2Pipeline` はここを読まない — ノブの既定は manifest の
   * `pipelineConfig.defaults` が正本で、同じ値を 2 箇所から導くのは二重保持だから
   * （`src/sbv2/config.ts`）。配布形にはまだ両方が並ぶので「あれば読める」形にしてある。
   */
  readonly defaults?: Sbv2Knobs;
};

const asRecord = (value: unknown, where: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${where}: オブジェクトでない`);
  }
  return value as Record<string, unknown>;
};

const asNumber = (value: unknown, where: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${where}: 有限の数値でない（${String(value)}）`);
  }
  return value;
};

const asInteger = (value: unknown, where: string): number => {
  const numeric = asNumber(value, where);
  if (!Number.isInteger(numeric)) throw new Error(`${where}: 整数でない（${numeric}）`);
  return numeric;
};

const asStringArray = (value: unknown, where: string): string[] => {
  if (!Array.isArray(value)) throw new Error(`${where}: 配列でない`);
  return value.map((entry, index) => {
    if (typeof entry !== "string") throw new Error(`${where}[${index}]: 文字列でない`);
    return entry;
  });
};

const parseKnobs = (raw: unknown, where: string): Sbv2Knobs => {
  const knobs = asRecord(raw, where);
  return {
    sdpRatio: asNumber(knobs["sdpRatio"], `${where}.sdpRatio`),
    noiseScale: asNumber(knobs["noiseScale"], `${where}.noiseScale`),
    noiseScaleW: asNumber(knobs["noiseScaleW"], `${where}.noiseScaleW`),
    lengthScale: asNumber(knobs["lengthScale"], `${where}.lengthScale`),
  };
};

const parseBertRelPos = (raw: unknown, where: string): Sbv2BertRelPos => {
  const rule = asRecord(raw, where);
  const positionBuckets = asInteger(rule["positionBuckets"], `${where}.positionBuckets`);
  const maxPosition = asInteger(rule["maxPosition"], `${where}.maxPosition`);
  // 2 未満だと mid = 0 で対数の分母が壊れる（式が成立する下限をここで縛る）。
  if (positionBuckets < 2) {
    throw new Error(`${where}.positionBuckets: 2 以上でない（${positionBuckets}）`);
  }
  if (maxPosition < 2) {
    throw new Error(`${where}.maxPosition: 2 以上でない（${maxPosition}）`);
  }
  return { positionBuckets, maxPosition };
};

/**
 * `symbols.json` を検査して読む。**壊れた資産を黙って使わない** — 記号表が空でも
 * ID 化は「未知の音素」で落ちるだけで、tone 基点が欠けて 0 に縮退すると別の埋め込み行を
 * 静かに引く。
 */
export const parseJpExtraRules = (raw: unknown, where: string): JpExtraRules => {
  const root = asRecord(raw, where);
  const symbols = asStringArray(root["symbols"], `${where}.symbols`);
  if (symbols.length === 0) throw new Error(`${where}.symbols: 空`);
  const symbolToId = new Map(symbols.map((symbol, index) => [symbol, index] as const));
  if (symbolToId.size !== symbols.length) {
    throw new Error(`${where}.symbols: 重複がある（ID の一意性が崩れる）`);
  }
  const pad = root["pad"];
  if (typeof pad !== "string" || !symbolToId.has(pad)) {
    throw new Error(`${where}.pad: 記号表に無い（${JSON.stringify(pad)}）`);
  }
  const rules: JpExtraRules = {
    symbols,
    symbolToId,
    pad,
    punctuations: new Set(asStringArray(root["punctuations"], `${where}.punctuations`)),
    toneStart: asInteger(root["toneStart"], `${where}.toneStart`),
    languageId: asInteger(root["languageId"], `${where}.languageId`),
    numTones: asPositiveInteger(root["numTones"], `${where}.numTones`),
    numLanguages: asPositiveInteger(root["numLanguages"], `${where}.numLanguages`),
    blankId: asInteger(root["blankId"], `${where}.blankId`),
    // MUST: 正値まで見る。`samplingRate` は `Sbv2Pipeline.generate` の `sampleRate` として
    // **そのまま公開結果へ出る**（唯一の消費者が呼び手）ので、負値や 0 はどこでも例外に
    // ならず「正常に見える壊れた WAV」になる。`hopLength` / `bertHiddenFromEnd` は遅れて
    // fail loudly するが、門をここへ揃えて資産の不正として構築時に落とす。
    samplingRate: asPositiveInteger(root["samplingRate"], `${where}.samplingRate`),
    hopLength: asPositiveInteger(root["hopLength"], `${where}.hopLength`),
    bertHiddenFromEnd: asPositiveInteger(
      root["bertHiddenFromEnd"],
      `${where}.bertHiddenFromEnd`,
    ),
    bertRelPos: parseBertRelPos(root["bertRelPos"], `${where}.bertRelPos`),
    ...(root["defaults"] === undefined
      ? {}
      : { defaults: parseKnobs(root["defaults"], `${where}.defaults`) }),
  };
  if (root["addBlank"] !== true) {
    // add_blank=False のモデルは intersperse を通さないので、音素長も word2ph も別の式に
    // なる。デモは add_blank 前提の 1 経路しか持たない — 黙って通すと 2 倍長い列を作る。
    throw new Error(`${where}.addBlank: true でない（add_blank 前提のデモでは未対応）`);
  }
  if (rules.blankId < 0 || rules.blankId >= symbols.length) {
    throw new Error(`${where}.blankId: 記号表の範囲外（${rules.blankId}）`);
  }
  if (rules.languageId < 0 || rules.languageId >= rules.numLanguages) {
    throw new Error(
      `${where}.languageId: language 埋め込みの範囲外（${rules.languageId} / ` +
        `0..${rules.numLanguages - 1}）`,
    );
  }
  return rules;
};

/**
 * 音素記号列を ID 列に変換する。未知の音素は握りつぶさず throw する（fail loudly）。
 * 参照実装 `cleaned_text_to_sequence` の KeyError と同じ挙動。
 */
export const phonesToIds = (rules: JpExtraRules, phones: readonly string[]): number[] =>
  phones.map((phone) => {
    const id = rules.symbolToId.get(phone);
    if (id === undefined) {
      // 呼び手の発話だけで到達する（未知の音素を moras / words に書く）ので入力起因 = 400。
      throw new Sbv2InputError(
        `記号表に無い音素 ${JSON.stringify(phone)} が phones に含まれる（ID 化不能）。` +
          " yomi の音素記号と JP-Extra モデルの記号表の齟齬を疑う。",
      );
    }
    return id;
  });

/**
 * add_blank=True の後処理。要素間・両端に `item` を挟んで `2·len+1` にする
 * （`commons.intersperse`）。
 */
export const intersperse = (seq: readonly number[], item: number): number[] => {
  const result = new Array<number>(seq.length * 2 + 1).fill(item);
  for (const [index, value] of seq.entries()) result[index * 2 + 1] = value;
  return result;
};

/** add_blank 適用済みの front 入力 ID 列（長さは全て `2·len+1`）。 */
export type ModelIdSequences = {
  readonly phoneIds: readonly number[];
  /** JP のトーン基点を加算済み。 */
  readonly toneIds: readonly number[];
  /** 実音素位置は `languageId`、blank 位置は `blankId`。 */
  readonly languageIds: readonly number[];
};

/**
 * given_phone / given_tone を front 入力用の ID 列（add_blank 適用済み）へ変換する。
 * 参照実装 `cleaned_text_to_sequence` + `commons.intersperse` の忠実移植。
 *
 * MUST: language は**全 0 ではない**（実音素位置は `languageId`、blank だけが `blankId`）。
 * JP-Extra も `language_emb` を持ち、`infer.get_text` は JP の言語 ID を配る。
 */
export const phonesTonesToModelIds = (
  rules: JpExtraRules,
  phones: readonly string[],
  tones: readonly number[],
): ModelIdSequences => {
  if (phones.length !== tones.length) {
    throw new Error(`phones(${phones.length}) と tones(${tones.length}) の長さが不一致`);
  }
  const phoneIds = phonesToIds(rules, phones);
  const shiftedTones = tones.map((tone, index) => {
    const shifted = tone + rules.toneStart;
    if (shifted < 0 || shifted >= rules.numTones) {
      throw new Error(
        `トーン ID ${shifted}（位置 ${index}・生値 ${tone}）が tone 埋め込みの範囲外` +
          `（0..${rules.numTones - 1}）`,
      );
    }
    return shifted;
  });
  if (rules.languageId < 0 || rules.languageId >= rules.numLanguages) {
    throw new Error(`言語 ID ${rules.languageId} が language 埋め込みの範囲外`);
  }
  return {
    phoneIds: intersperse(phoneIds, rules.blankId),
    toneIds: intersperse(shiftedTones, rules.blankId),
    languageIds: intersperse(phoneIds.map(() => rules.languageId), rules.blankId),
  };
};

/**
 * base word2ph（add_blank 前）から add_blank 後の word2ph を作る。
 * 参照実装 `infer.get_text` の `word2ph[i] *= 2; word2ph[0] += 1` の忠実移植で、これにより
 * `sum(word2ph)` が add_blank 後の音素列長 `2·len+1` に一致する。
 */
export const addBlankWord2ph = (baseWord2ph: readonly number[]): number[] => {
  if (baseWord2ph.length === 0) {
    throw new Error("base word2ph が空（両端の番兵が欠落）");
  }
  const doubled = baseWord2ph.map((count) => count * 2);
  doubled[0] += 1;
  return doubled;
};
