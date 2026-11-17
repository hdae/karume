import { type DistributionSource, localDirectory } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { hasRareTokenSplitTable, readSplitTable } from "../src/models/irodori/split-table.ts";

const hex = (character: string): string => character.repeat(64);
const V4 = "hdae/karume-irodori-v4-small";
const REVISION = "4".repeat(40);
const TOKENIZER_PATH = "tokenizer/tokenizer.json";

const bytesOf = (value: unknown): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(JSON.stringify(value));

/** Shaped like the small models' Unigram asset (no `format` key), with or without the table. */
const unigramAsset = (table: boolean) => ({
  vocabText: "<pad>\n<unk>",
  ...(table ? { rareTokenSplit: { disabledIds: [1] } } : {}),
});

/** Shaped like v4-large's Gemma BPE asset. */
const GEMMA_ASSET = { format: "karume-gemma-tokenizer/1", vocab: ["<pad>"] };

/** A `karume/6` manifest whose model has one tokenizer asset, its own or borrowed from v4-small. */
const manifestBytes = (tokenizer: Uint8Array<ArrayBuffer>, borrowed: boolean) =>
  bytesOf({
    format: "karume/6",
    generator: "test",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "irodori/1",
        weights: {
          dit: {
            i8: {
              container: {
                descriptor: {
                  graph: { length: 8, sha256: hex("d") },
                  model: { length: 8, sha256: hex("e") },
                },
                parts: [
                  { path: "dit-00001-of-00002.krm", size: 40, sha256: hex("a") },
                  { path: "dit-00002-of-00002.krm", size: 8, sha256: hex("b") },
                ],
              },
            },
          },
        },
        assets: {
          tokenizer: {
            path: TOKENIZER_PATH,
            size: tokenizer.byteLength,
            sha256: hex("c"),
            ...(borrowed ? { repo: V4, revision: REVISION } : {}),
          },
        },
        quants: { i8: { weights: { dit: "i8" }, session: {} } },
        defaultQuant: "i8",
        pipelineConfig: {},
      },
    },
  });

/** An in-memory directory (`localDirectory` checks only the size of what it reads). */
const directory = (
  files: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  crossRepo: Readonly<Record<string, DistributionSource>> = {},
): DistributionSource =>
  localDirectory(
    {
      readFile: (path) => {
        const bytes = Object.hasOwn(files, path) ? files[path] : undefined;
        return bytes === undefined
          ? Promise.reject(new Error(`no ${path}`))
          : Promise.resolve(bytes);
      },
    },
    { label: "test", crossRepo },
  );

const distribution = (asset: unknown): DistributionSource => {
  const tokenizer = bytesOf(asset);
  return directory({ "karume.json": manifestBytes(tokenizer, false), [TOKENIZER_PATH]: tokenizer });
};

describe("hasRareTokenSplitTable", () => {
  it("is true for a Unigram asset that carries the rareTokenSplit table", () => {
    expect(hasRareTokenSplitTable(unigramAsset(true))).toBe(true);
  });

  it("is false for a Unigram asset without the table and for the Gemma asset", () => {
    expect(hasRareTokenSplitTable(unigramAsset(false))).toBe(false);
    expect(hasRareTokenSplitTable(GEMMA_ASSET)).toBe(false);
  });

  it.each([[null], [[]], ["tokenizer"]])("fails loudly on the non-object %j", (asset) => {
    expect(() => hasRareTokenSplitTable(asset)).toThrow("not a JSON object");
  });
});

describe("readSplitTable", () => {
  it("reads the default model's tokenizer asset and answers from it", async () => {
    await expect(readSplitTable(distribution(unigramAsset(true)))).resolves.toBe(true);
    await expect(readSplitTable(distribution(unigramAsset(false)))).resolves.toBe(false);
    await expect(readSplitTable(distribution(GEMMA_ASSET))).resolves.toBe(false);
  });

  it("follows a borrowed tokenizer asset to the source that holds its repository", async () => {
    const tokenizer = bytesOf(unigramAsset(true));
    const holder = directory({ [TOKENIZER_PATH]: tokenizer });
    const borrowing = directory(
      { "karume.json": manifestBytes(tokenizer, true) },
      { [V4]: holder },
    );
    await expect(readSplitTable(borrowing)).resolves.toBe(true);
  });

  it("stops when its signal is aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      readSplitTable(distribution(unigramAsset(true)), controller.signal),
    ).rejects.toThrow();
  });
});
