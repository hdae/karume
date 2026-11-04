/**
 * Derives the PLE resident budget **from the index** (the source of the `maxResidentPleBytes` the
 * e2e pass).
 *
 * MUST: do not write the budget as a test-side constant. A number like "one block ≈32MiB, so 96MiB
 * for 3 blocks" means a different block count the moment the asset generation (the writer's block
 * limit) changes, and the test's intent (no block is re-read in a conversation that crosses ranges)
 * breaks without exception — the reason the budget is taken in bytes rather than a block count
 * (ADR 0085 addendum 2026-09-02) applies on the test side as well.
 *
 * Passing the full resident size ({@link allResidentBytes}) means no re-read happens in any
 * generation.
 */

import { gemma4PleTotalBytes } from "../../src/gemma/ple-index.ts";
import type { Gemma4PleIndex } from "../../src/gemma/ple-index.ts";
import { type MirrorChoice, openGemma4Ple } from "./gemma-mirror.ts";

/** The size of every block the index points to (= zero re-reads). Decided by the index alone. */
export const allResidentBytes = (index: Gemma4PleIndex): number => gemma4PleTotalBytes(index);

/**
 * The full resident budget of a distribution mirror (the index is an asset of the `model`
 * container, so the container is opened to read it — the mirror's directory layout is not copied
 * into the test).
 *
 * Not called where the assets are missing (callers use it after their SKIP decision) — an index
 * that cannot be read propagates rather than being swallowed.
 */
export const allResidentPleBytesOfMirror = async (
  mirror: URL,
  choice: MirrorChoice = {},
): Promise<number> => allResidentBytes((await openGemma4Ple(mirror, choice)).index);
