/**
 * JP-Extra's ID rules (reading the `symbols.json` asset, and building the ID sequences with it).
 *
 * ## MUST: do not write constant tables here
 *
 * The phoneme symbol table, the tone base, the language ID and the add_blank insertion value are
 * authoritative only as **the asset taken from the actual `style_bert_vits2`** (the `symbols.json`
 * written by the `assets` subcommand of `tools/export-recipes/sbv2/demo.py`).
 * They look the same in the multilingual version and JP-Extra, and when they drift **the shapes
 * still match and only the sound breaks** (one position off in the symbol table gives a different
 * phoneme; a shifted tone base gives a different embedding row). Holding a copied constant table
 * on the TS side removes the means of detecting that mismatch.
 *
 * ## Correspondence with the reference implementation
 *
 * - `phonesToIds` / `applyToneStart` / `languageIds` = `nlp.cleaned_text_to_sequence`
 * - `intersperse` = `models.commons.intersperse` (the add_blank branch of `infer.get_text`)
 * - `addBlankWord2ph` = the same branch's `word2ph[i] *= 2; word2ph[0] += 1`
 */

import { asPositiveInteger } from "../../text/asset-gates.ts";
import { assertLogBucketDomain } from "./rel-pos-tables.ts";
import { Sbv2InputError } from "../errors.ts";

/**
 * Bucket rules for building the relative-position index tables (from DeBERTa's config).
 *
 * Exactly the arguments of `buildRelPosTables`. The formula builds indices that are linear near and
 * log-compressed far; `positionBuckets` is the total bucket count (= the centre offset of the
 * index), and `maxPosition` is the maximum distance the log compression is based on.
 */
type Sbv2BertRelPos = {
  /** `position_buckets` (256 for char-wwm). */
  readonly positionBuckets: number;
  /** `max_relative_positions` (`max_position_embeddings` when below 1 — 512 for char-wwm). */
  readonly maxPosition: number;
};

/** Run-time knobs (the defaults of `style_bert_vits2/constants.py` arrive through the asset). */
export type Sbv2Knobs = {
  /** Mixing ratio of sdp and dp (`logw = sdp·r + dp·(1−r)`). */
  readonly sdpRatio: number;
  /** Noise scale of z_p. */
  readonly noiseScale: number;
  /** Noise scale of the sdp reverse (multiplied into `z_noise`). */
  readonly noiseScaleW: number;
  /** Duration scale (larger is slower). */
  readonly lengthScale: number;
};

/** JP-Extra's ID rules and model constants (the content of `symbols.json`). */
export type JpExtraRules = {
  /** Phoneme symbol table. The index is the row number of `enc_p.emb`. */
  readonly symbols: readonly string[];
  /** Phoneme symbol → ID. */
  readonly symbolToId: ReadonlyMap<string, number>;
  /** The PAD symbol (placed at both ends of the phoneme sequence. `SYMBOLS[0]`). */
  readonly pad: string;
  /** Normal-form punctuation accepted as phonemes. */
  readonly punctuations: ReadonlySet<string>;
  /** JP's tone base (`LANGUAGE_TONE_START_MAP["JP"]`). Added to given_tone. */
  readonly toneStart: number;
  /** JP's language ID (`LANGUAGE_ID_MAP["JP"]`). */
  readonly languageId: number;
  /** Row count of the tone embedding (used for the range check). */
  readonly numTones: number;
  /** Row count of the language embedding (used for the range check). */
  readonly numLanguages: number;
  /**
   * The ID inserted by add_blank (the same value for all 3 sequences — confirmed by the asset
   * generator, which extracted it from the source).
   */
  readonly blankId: number;
  /** Sampling rate of the output WAV. */
  readonly samplingRate: number;
  /** Samples per frame (used to verify `audio length = hopLength × frames`). */
  readonly hopLength: number;
  /**
   * Position from the end of the hidden_states used for BERT features (1, since the distributed
   * graph has a single output).
   */
  readonly bertHiddenFromEnd: number;
  /**
   * Bucket rules for building the relative-position index tables (from DeBERTa's config).
   *
   * The tables themselves are graph inputs, built by the host for the actual length
   * (`rel-pos-tables.ts`). Taking the values from the asset instead of copying them is the same
   * discipline as ADR 0039 decision 3 — when the model changes the rules change too, yet the shapes
   * still match and it **gathers different position embeddings** (a silent wrong value).
   */
  readonly bertRelPos: Sbv2BertRelPos;
  /**
   * Defaults of the run-time knobs (**optional**).
   *
   * NOTE: `Sbv2Pipeline` does not read this — the canonical knob defaults are the manifest's
   * `pipelineConfig.defaults`, and deriving the same values from 2 places is duplicate state
   * (`src/sbv2/config.ts`). The distribution still carries both side by side, so this has the form
   * "readable if present".
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
  // Below 2, mid = 0 breaks the log's denominator (bound here the lower limit at which the formula
  // holds).
  if (positionBuckets < 2) {
    throw new Error(`${where}.positionBuckets: 2 以上でない（${positionBuckets}）`);
  }
  if (maxPosition < 2) {
    throw new Error(`${where}.maxPosition: 2 以上でない（${maxPosition}）`);
  }
  // Even above the lower limits, some pairs produce NaN ({@link assertLogBucketDomain}) — fail when
  // the asset is read rather than waiting for the table to be built on each generation.
  assertLogBucketDomain(positionBuckets, maxPosition, where);
  return { positionBuckets, maxPosition };
};

/**
 * Checks and reads `symbols.json`. **A broken asset is not used silently** — an empty symbol table
 * only fails at ID conversion as "unknown phoneme", but a missing tone base that degenerates to 0
 * quietly looks up a different embedding row.
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
    // MUST: check down to positivity. `samplingRate` **goes out as is in the public result** as
    // `Sbv2Pipeline.generate`'s `sampleRate` (its only consumer is the caller), so a negative
    // value or 0 raises an exception nowhere and becomes "a broken WAV that looks normal".
    // `hopLength` / `bertHiddenFromEnd` fail loudly later, but the checks are aligned here so they
    // fail at construction as an invalid asset.
    samplingRate: asPositiveInteger(root["samplingRate"], `${where}.samplingRate`),
    hopLength: asPositiveInteger(root["hopLength"], `${where}.hopLength`),
    bertHiddenFromEnd: asPositiveInteger(root["bertHiddenFromEnd"], `${where}.bertHiddenFromEnd`),
    bertRelPos: parseBertRelPos(root["bertRelPos"], `${where}.bertRelPos`),
    ...(root["defaults"] === undefined
      ? {}
      : { defaults: parseKnobs(root["defaults"], `${where}.defaults`) }),
  };
  if (root["addBlank"] !== true) {
    // A model with add_blank=False does not go through intersperse, so the phoneme length and
    // word2ph follow different formulas. The pipeline has only the one path that assumes add_blank
    // — letting it through silently would build a sequence twice as long.
    throw new Error(`${where}.addBlank: true でない（add_blank 前提のパイプラインでは未対応）`);
  }
  // MUST: blank goes into **all 3** ID sequences — phoneme, tone and language — with the same value
  // (`intersperse`), so being outside any one of the 3 tables gives an out-of-range embedding
  // gather (the phoneme table alone is not enough).
  const blankLimit = Math.min(symbols.length, rules.numTones, rules.numLanguages);
  if (rules.blankId < 0 || rules.blankId >= blankLimit) {
    throw new Error(
      `${where}.blankId: 音素・tone・language の埋め込みの範囲外（${rules.blankId} / ` +
        `0..${blankLimit - 1} — 記号表 ${symbols.length} 件・numTones ${rules.numTones}・` +
        `numLanguages ${rules.numLanguages}）`,
    );
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
 * Converts a phoneme symbol sequence to an ID sequence. Unknown phonemes are not swallowed; they
 * throw (fail loudly). The same behaviour as the KeyError of the reference implementation's
 * `cleaned_text_to_sequence`.
 */
export const phonesToIds = (rules: JpExtraRules, phones: readonly string[]): number[] =>
  phones.map((phone) => {
    const id = rules.symbolToId.get(phone);
    if (id === undefined) {
      // Reachable through the caller's utterance alone (writing an unknown phoneme into moras /
      // words), so it is input-caused = 400.
      throw new Sbv2InputError(
        `記号表に無い音素 ${JSON.stringify(phone)} が phones に含まれる（ID 化不能）。` +
          " 発話（moras / words）の音素記号と JP-Extra モデルの記号表の齟齬を疑う。",
      );
    }
    return id;
  });

/**
 * Post-processing for add_blank=True. Inserts `item` between elements and at both ends, giving
 * `2·len+1` (`commons.intersperse`).
 */
export const intersperse = (seq: readonly number[], item: number): number[] => {
  const result = new Array<number>(seq.length * 2 + 1).fill(item);
  for (const [index, value] of seq.entries()) result[index * 2 + 1] = value;
  return result;
};

/** front's input ID sequences with add_blank applied (all of length `2·len+1`). */
export type ModelIdSequences = {
  readonly phoneIds: readonly number[];
  /** With JP's tone base added. */
  readonly toneIds: readonly number[];
  /** `languageId` at real phoneme positions, `blankId` at blank positions. */
  readonly languageIds: readonly number[];
};

/**
 * Converts given_phone / given_tone to front's input ID sequences (add_blank applied).
 * A faithful port of the reference implementation's `cleaned_text_to_sequence` +
 * `commons.intersperse`.
 *
 * MUST: language is **not all 0** (`languageId` at real phoneme positions, `blankId` only at
 * blanks). JP-Extra also has `language_emb`, and `infer.get_text` hands out JP's language ID.
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
    languageIds: intersperse(
      phoneIds.map(() => rules.languageId),
      rules.blankId,
    ),
  };
};

/**
 * Builds the word2ph after add_blank from the base word2ph (before add_blank).
 * A faithful port of the reference implementation `infer.get_text`'s
 * `word2ph[i] *= 2; word2ph[0] += 1`, which makes `sum(word2ph)` equal the phoneme sequence length
 * after add_blank, `2·len+1`.
 */
export const addBlankWord2ph = (baseWord2ph: readonly number[]): number[] => {
  if (baseWord2ph.length === 0) {
    throw new Error("base word2ph が空（両端の番兵が欠落）");
  }
  const doubled = baseWord2ph.map((count) => count * 2);
  doubled[0] += 1;
  return doubled;
};
