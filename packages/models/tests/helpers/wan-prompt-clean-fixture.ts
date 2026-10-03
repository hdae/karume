// Wan2.1 の前処理（src/wan/text/prompt-clean.ts）の乱択・掃引テストが共有する読み口と再生器。
//
// 期待はフィクスチャ `wan-text/prompt-clean-sweep.json`（recipe
// `tools/export-recipes/wan/umt5_tokenizer.py` が上流から採った digest）が持ち、ここは recipe と
// **同じ手順**で同じ列を作って digest を取るだけ（`wan/prompt_clean.py` の `XorShift32` /
// `fuzz_strings` / `digest` / `sweep_blocks` の写経）。手順がずれれば digest が全部外れる（緑のまま
// 別の列を見ることはない）。

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
    await Deno.readTextFile(
      new URL("../fixtures/wan-text/prompt-clean-sweep.json", import.meta.url),
    ),
  ) as SweepFixture;

/** 前処理の表（parity フィクスチャの `promptClean` — 資産と同じ形の全体）。 */
export const loadPromptCleanTables = async (): Promise<PromptCleanTables> =>
  parsePromptCleanTables(
    (JSON.parse(
      await Deno.readTextFile(new URL("../fixtures/wan-text/parity.json", import.meta.url)),
    ) as { promptClean: unknown }).promptClean,
    "promptClean",
  );

/** 塊ごとの digest の桁数（recipe の `BLOCK_DIGEST_HEX`）。 */
const BLOCK_DIGEST_HEX = 16;

/** 32 ビットの xorshift（recipe の `XorShift32` と同じ列）。 */
export const xorshift32 = (seed: number): () => number => {
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

/** 長さ 1..maxLength の乱択列（長さ → 各文字の順に引く — recipe の `fuzz_strings`）。 */
export function* fuzzStrings(
  pool: readonly number[],
  seed: number,
  count: number,
  maxLength: number,
): Generator<string> {
  const next = xorshift32(seed);
  for (let index = 0; index < count; index++) {
    const length = 1 + next() % maxLength;
    let text = "";
    for (let k = 0; k < length; k++) text += String.fromCodePoint(pool[next() % pool.length]);
    yield text;
  }
}

/** 1 入力の決着（受理なら `A` + 出力・拒否なら `R` + 理由 — recipe の `record`）。 */
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

/** 記録の列の SHA-256（各記録の後に U+0000 — recipe の `digest`）。 */
export const digestRecords = async (records: readonly string[]): Promise<string> => {
  let joined = "";
  for (const item of records) joined += item + "\u0000";
  return hex(await crypto.subtle.digest("SHA-256", encoder.encode(joined)));
};

/** `0` / `1` の列の SHA-256（recipe の `bits_digest`）。 */
export const digestBits = async (bits: string): Promise<string> =>
  hex(await crypto.subtle.digest("SHA-256", encoder.encode(bits)));

/** `chunk` 件ごとの digest（recipe の `chunked_digests`）。 */
export const chunkedDigests = async (records: readonly string[], chunk: number) => {
  const out: string[] = [];
  for (let i = 0; i < records.length; i += chunk) {
    out.push((await digestRecords(records.slice(i, i + chunk))).slice(0, BLOCK_DIGEST_HEX));
  }
  return out;
};

/** サロゲートを除く全コードポイント（recipe の `ALL_CODEPOINTS`）。 */
export const allCodePoints = (): number[] => {
  const out: number[] = [];
  for (let cp = 0; cp <= 0x10ffff; cp++) if (cp < 0xd800 || cp > 0xdfff) out.push(cp);
  return out;
};

/** `domain` を塊に分けた digest（空の塊も 1 本 — recipe の `sweep_blocks`）。 */
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

/** 期待と違う塊の位置（どこが割れたかをメッセージに出す）。 */
export const mismatchedIndices = (actual: readonly string[], expected: readonly string[]) =>
  expected.flatMap((digest, index) => (actual[index] === digest ? [] : [index]));
