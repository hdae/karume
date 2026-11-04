// The reader and the replay generator shared by the random and sweep tests of the Wan2.1
// preprocessing (src/wan/text/prompt-clean.ts).
//
// The expectations live in the fixture `wan-text/prompt-clean-sweep.json` (digests that the recipe
// `tools/export-recipes/wan/umt5_tokenizer.py` took from upstream); this file only builds the same
// sequences with **the same procedure** as the recipe and takes their digests (a copy of
// `XorShift32` / `fuzz_strings` / `digest` / `sweep_blocks` of `wan/prompt_clean.py`). If the
// procedure drifts, every digest misses (it never stays green while looking at other sequences).

import { readTextFile } from "@karume/test-support/fs";
import {
  cleanPrompt,
  parsePromptCleanTables,
  PromptCleanError,
  type PromptCleanTables,
} from "../../src/wan/text/prompt-clean.ts";

export type FuzzRun = {
  readonly name: string;
  readonly pool: string;
  readonly seed: number;
  readonly count: number;
  readonly maxLength: number;
  readonly chunk: number;
  readonly chunks: string[];
  readonly outcomes: Record<string, number>;
  readonly maxRounds: number;
};

export type BadnessCheck = {
  readonly name: string;
  readonly digest: string;
  readonly hits: number;
  readonly pool?: string;
  readonly seed?: number;
  readonly count?: number;
  readonly maxLength?: number;
  readonly template?: string;
};

export type Sweep = { readonly name: string; readonly template: string; readonly blocks: string[] };

export type SweepFixture = {
  readonly prng: { readonly name: string; readonly seed: number; readonly first: number[] };
  readonly block: number;
  readonly pools: Record<string, [number, number][]>;
  readonly clean: FuzzRun[];
  readonly badness: BadnessCheck[];
  readonly cleanSweeps: Sweep[];
  readonly nfcSweeps: Sweep[];
};

export const loadSweepFixture = async (): Promise<SweepFixture> =>
  JSON.parse(
    await readTextFile(new URL("../fixtures/wan-text/prompt-clean-sweep.json", import.meta.url)),
  ) as SweepFixture;

/**
 * The preprocessing tables (`promptClean` of the parity fixture — the whole, in the same shape as
 * the asset).
 */
export const loadPromptCleanTables = async (): Promise<PromptCleanTables> =>
  parsePromptCleanTables(
    (
      JSON.parse(
        await readTextFile(new URL("../fixtures/wan-text/parity.json", import.meta.url)),
      ) as { promptClean: unknown }
    ).promptClean,
    "promptClean",
  );

/** The number of hex digits of each block's digest (the recipe's `BLOCK_DIGEST_HEX`). */
const BLOCK_DIGEST_HEX = 16;

/** A 32-bit xorshift (the same sequence as the recipe's `XorShift32`). */
export const xorshift32 = (seed: number): (() => number) => {
  let x = seed >>> 0;
  return () => {
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return x;
  };
};

export const expandRanges = (ranges: readonly (readonly [number, number])[]): number[] => {
  const out: number[] = [];
  for (const [start, end] of ranges) for (let cp = start; cp <= end; cp++) out.push(cp);
  return out;
};

/**
 * Random strings of length 1..maxLength (drawn length first, then each character — the recipe's
 * `fuzz_strings`).
 */
export function* fuzzStrings(
  pool: readonly number[],
  seed: number,
  count: number,
  maxLength: number,
): Generator<string> {
  const next = xorshift32(seed);
  for (let index = 0; index < count; index++) {
    const length = 1 + (next() % maxLength);
    let text = "";
    for (let k = 0; k < length; k++) text += String.fromCodePoint(pool[next() % pool.length]);
    yield text;
  }
}

/**
 * The outcome of one input (`A` + output when accepted, `R` + reason when rejected — the recipe's
 * `record`).
 */
export const recordOf = (tables: PromptCleanTables, text: string): string => {
  try {
    return "A" + cleanPrompt(tables, text);
  } catch (error) {
    if (error instanceof PromptCleanError) return "R" + error.reason;
    throw error;
  }
};

const hex = (buffer: ArrayBuffer): string =>
  Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");

const encoder = new TextEncoder();

/** The SHA-256 of a sequence of records (U+0000 after each record — the recipe's `digest`). */
export const digestRecords = async (records: readonly string[]): Promise<string> => {
  let joined = "";
  for (const item of records) joined += item + "\u0000";
  return hex(await crypto.subtle.digest("SHA-256", encoder.encode(joined)));
};

/** The SHA-256 of a `0` / `1` string (the recipe's `bits_digest`). */
export const digestBits = async (bits: string): Promise<string> =>
  hex(await crypto.subtle.digest("SHA-256", encoder.encode(bits)));

/** A digest per `chunk` records (the recipe's `chunked_digests`). */
export const chunkedDigests = async (records: readonly string[], chunk: number) => {
  const out: string[] = [];
  for (let i = 0; i < records.length; i += chunk) {
    out.push((await digestRecords(records.slice(i, i + chunk))).slice(0, BLOCK_DIGEST_HEX));
  }
  return out;
};

/** Every code point except surrogates (the recipe's `ALL_CODEPOINTS`). */
export const allCodePoints = (): number[] => {
  const out: number[] = [];
  for (let cp = 0; cp <= 0x10ffff; cp++) if (cp < 0xd800 || cp > 0xdfff) out.push(cp);
  return out;
};

/**
 * Digests of `domain` split into blocks (an empty block also gets one — the recipe's
 * `sweep_blocks`).
 */
export const sweepBlocks = async (
  domain: readonly number[],
  recordOfCodePoint: (cp: number) => string,
  block: number,
): Promise<string[]> => {
  const buckets: string[][] = Array.from({ length: 0x110000 / block }, () => []);
  for (const cp of domain) buckets[Math.floor(cp / block)].push(recordOfCodePoint(cp));
  const out: string[] = [];
  for (const bucket of buckets) out.push((await digestRecords(bucket)).slice(0, BLOCK_DIGEST_HEX));
  return out;
};

/**
 * The positions of blocks that differ from the expectation (shows in the message where it broke).
 */
export const mismatchedIndices = (actual: readonly string[], expected: readonly string[]) =>
  expected.flatMap((digest, index) => (actual[index] === digest ? [] : [index]));
