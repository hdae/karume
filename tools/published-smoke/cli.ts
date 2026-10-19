/**
 * The pure logic of published-smoke (the argument check and the type guard of a pin entry).
 *
 * `main.ts` dynamically imports the published packages (`@karume/*`) from the registry as soon as it
 * starts, so tests cannot import it (it needs the network and the published packages). Only the
 * checks live here, and `tests/main_test.ts` tests them.
 */

/** One entry of the source table (a `{repo, revision}` pin — ADR 0092). */
export type PinRef = { readonly repo: string; readonly revision: string };

export const isPinRef = (value: unknown): value is PinRef =>
  typeof value === "object" &&
  value !== null &&
  typeof (value as { repo?: unknown }).repo === "string" &&
  typeof (value as { revision?: unknown }).revision === "string";

/** The two ways to run it (spelled as in the README — on Deno, `deno run -A main.ts` takes the same arguments). */
export const USAGE = "node tools/published-smoke/main.ts [--manifests-only]";

/** The only accepted argument (`--manifests-only` = skip fromPretrained, which needs a GPU). */
const KNOWN: ReadonlySet<string> = new Set(["--manifests-only"]);

/**
 * Reads the arguments.
 *
 * MUST: an unknown argument fails (the same discipline as `tools/verify-diff`). Silently dropping a
 * typo such as `--manifest-only` (singular) falls quietly through to the full path with the GPU that
 * the run meant to skip. On a machine without a GPU, sbv2's `fromPretrained` then fails for another
 * reason, so the connectivity check right after a release reads as a **false red** ("the published
 * packages are broken").
 */
export const parseSmokeArgs = (argv: readonly string[]): { readonly manifestsOnly: boolean } => {
  for (const arg of argv) {
    if (!KNOWN.has(arg)) throw new Error(`unknown argument ${arg} (usage: ${USAGE})`);
  }
  return { manifestsOnly: argv.includes("--manifests-only") };
};
