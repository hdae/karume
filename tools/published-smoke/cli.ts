/**
 * The pure logic of published-smoke (the argument check and the type guards of a pin entry and of
 * the sbv2 module).
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

/**
 * The part of `@karume/models/sbv2` the full run uses, spelled with the bare names of 0.15.0 (ADR
 * 0136). 0.14.x exports `Sbv2Pipeline` / `SBV2_SOURCES` instead.
 */
export type Sbv2Module = {
  readonly Pipeline: {
    fromPretrained: (
      ref: PinRef,
      options?: { readonly caches?: CacheStorage },
    ) => Promise<{ dispose: () => Promise<void> }>;
  };
  readonly SOURCES: Readonly<Record<string, unknown>>;
};

/**
 * Whether a dynamically imported module has the shape of {@link Sbv2Module}. Checked at run time
 * so that a published version with another surface fails with a message naming the mismatch, not
 * with a `TypeError` on `undefined` further down.
 */
export const isSbv2Module = (value: unknown): value is Sbv2Module => {
  if (typeof value !== "object" || value === null) return false;
  if (!("Pipeline" in value) || !("SOURCES" in value)) return false;
  const { Pipeline: pipeline, SOURCES: sources } = value;
  return (
    typeof sources === "object" &&
    sources !== null &&
    typeof pipeline === "function" &&
    "fromPretrained" in pipeline &&
    typeof pipeline.fromPretrained === "function"
  );
};

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
