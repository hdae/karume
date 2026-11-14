/**
 * The rare-token split table on the **real v4-small distribution** (ADR 0149 decision 5): the
 * tokenizer asset of the mirror `models/karume-irodori-v4-small/` carries `rareTokenSplit`, the
 * pipeline's default for a request without `splitRareTokens` resolves to the split, and
 * `splitRareTokens: false` gives the ids the asset gave before the table existed.
 *
 * No GPU: the text stage is the tokenizer and the packing the pipeline's stage 1 runs
 * (`resolveSplitRareTokens` → `packIds`), and nothing after it.
 *
 * The text is taken from the asset itself — a disabled token that normalization leaves as it is and
 * that the unsplit lattice emits as one id — so the check follows upstream's lists without
 * restating them. "Before the table existed" is the same asset without the key: the recipe writes
 * the key last, and the asset without it is byte-identical to the earlier revision
 * (`tools/export-recipes/irodori/tokenizer_ref.py`).
 *
 * MUST: the mirror is the distribution (ADR 0139 decision 1); without it, or with a mirror whose
 * tokenizer predates the table (assembled before the reassembly), the cases SKIP explicitly.
 */

import { readFile } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals } from "@std/assert";
import { parseIrodoriPipelineConfig } from "../../models/src/irodori/config.ts";
import { packIds } from "../../models/src/irodori/host/pack.ts";
import { resolveSplitRareTokens } from "../../models/src/irodori/pipeline.ts";
import { normalizeText } from "../../models/src/irodori/text/normalize.ts";
import {
  IrodoriTokenizer,
  parseIrodoriTokenizerAsset,
} from "../../models/src/irodori/text/tokenizer.ts";
import { ASSETS_DIR, manifestText, modelEntry, readManifest } from "./helpers/irodori-assets.ts";

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const entry = manifestText === undefined ? undefined : modelEntry(readManifest());
const tokenizerRef = entry?.assets["tokenizer"];
const tokenizerBytes =
  tokenizerRef === undefined ? undefined : await readFile(new URL(tokenizerRef.path, ASSETS_DIR));
const tokenizerJson =
  tokenizerBytes === undefined
    ? undefined
    : (JSON.parse(new TextDecoder().decode(tokenizerBytes)) as Record<string, unknown>);
const CARRIES_TABLE = tokenizerJson !== undefined && Object.hasOwn(tokenizerJson, "rareTokenSplit");
if (!CARRIES_TABLE) {
  console.warn(
    `[karume] skipping the rare-token split check of the real distribution (it needs the mirror` +
      ` ${ASSETS_DIR.pathname} reassembled with the rare-token split table — ADR 0149 decision 5)`,
  );
}

describe.skipIf(!CARRIES_TABLE)(
  "e2e (no GPU): the v4-small distribution's tokenizer asset carries the rare-token split table",
  () => {
    const json = tokenizerJson as Record<string, unknown>;
    const assets = parseIrodoriTokenizerAsset(json);
    const tokenizer = new IrodoriTokenizer(assets);
    const { rareTokenSplit: _table, ...previous } = json;
    const previousTokenizer = new IrodoriTokenizer(parseIrodoriTokenizerAsset(previous));
    const maxTextLen = parseIrodoriPipelineConfig(
      (entry as NonNullable<typeof entry>).pipelineConfig,
    ).maxTextLen;
    const tokens = new Map([...assets.vocab].map(([token, { id }]) => [id, token]));

    /** The first disabled token (ascending id) that normalization keeps and the unsplit path emits whole. */
    const probe = (): { readonly text: string; readonly id: number } => {
      for (const id of assets.rareTokenSplit?.disabledIds ?? []) {
        const text = tokens.get(id);
        if (text === undefined || text.includes("▁") || normalizeText(text).trim() !== text) {
          continue;
        }
        const ids = packIds(tokenizer, text, maxTextLen, "text", { splitRareTokens: false });
        if (ids.length === 2 && ids[1] === id) return { text, id };
      }
      throw new Error("no disabled token of the asset encodes to itself on the unsplit path");
    };

    it("the asset is the one the manifest declares, and it has the split path", async () => {
      const ref = tokenizerRef as NonNullable<typeof tokenizerRef>;
      const bytes = tokenizerBytes as Uint8Array<ArrayBuffer>;
      assertEquals([bytes.byteLength, await sha256Hex(bytes)], [ref.size, ref.sha256]);
      assert(tokenizer.hasRareTokenSplit);
      assert(!previousTokenizer.hasRareTokenSplit);
    });

    it("a request without splitRareTokens encodes a text with a disabled token split", () => {
      const { text, id } = probe();
      const split = resolveSplitRareTokens(undefined, tokenizer);
      const ids = Array.from(
        packIds(tokenizer, text, maxTextLen, "text", { splitRareTokens: split }),
      );

      assertEquals(split, true);
      assertEquals(ids[0], assets.bosId);
      assert(!ids.includes(id), `the disabled id ${id} is still emitted: ${ids.join(" ")}`);
      assertNotEquals(ids, [assets.bosId, id]);
    });

    it("splitRareTokens: false gives the ids of the asset before the table existed", () => {
      const { text, id } = probe();
      const unsplit = resolveSplitRareTokens(false, tokenizer);
      const ids = Array.from(
        packIds(tokenizer, text, maxTextLen, "text", { splitRareTokens: unsplit }),
      );
      const before = Array.from(
        packIds(previousTokenizer, text, maxTextLen, "text", { splitRareTokens: false }),
      );

      assertEquals(unsplit, false);
      assertEquals(ids, before);
      assertEquals(ids, [assets.bosId, id]);
    });
  },
);
