/**
 * Text analysis for the Style-Bert-VITS2 page. Since 0.6.0 reading and accent analysis is the
 * caller's job (`Pipeline.generate` takes an analysed `Utterance`), so the page is the caller
 * here: `@hdae/yomi` turns the text into accent phrases with their nuclei and the word alignment,
 * which structurally satisfies `Phrases`, and `toUtterance` lowers that to the utterance.
 * This is the same analyser the SBV2 regression test uses (packages/e2e/tests/e2e_sbv2_wav_test.ts).
 *
 * The dictionary (JTD1, about 6.4 MB gzip) is not in a distribution: `getDictionary` fetches it from
 * the analyser's Hugging Face dataset at the revision its version pins, and keeps it in this origin's
 * Cache Storage (`yomi-dict`), so only the first visit needs the network. It takes no AbortSignal.
 */

import { analyzeWithWords, type JtdDictionary, VERSION } from "@hdae/yomi";
import { getDictionary } from "@hdae/yomi/loader";
import { type Utterance, toUtterance } from "@karume/models/sbv2";

export type ReadingDictionary = JtdDictionary;

/** The analyser and its version (the version pins the dictionary revision; both shape the input). */
export const ANALYZER = { name: "@hdae/yomi", version: VERSION } as const;

/** Fetches (or reads from Cache Storage) and verifies the reading dictionary. */
export const loadReadingDictionary = (
  onProgress: (loaded: number, total: number | undefined) => void,
): Promise<ReadingDictionary> =>
  getDictionary({ onProgress: ({ loaded, total }) => onProgress(loaded, total) });

/** Text → the utterance `generate` takes (pure and synchronous once the dictionary is there). */
export const analyzeText = (dictionary: ReadingDictionary, text: string): Utterance =>
  toUtterance(analyzeWithWords(dictionary, text));
