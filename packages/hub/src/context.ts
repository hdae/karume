/**
 * The fetch context — where the "place actually fetched from" that errors name is assembled.
 *
 * MUST: sources (`sources/`) do not assemble errors. Only the common layer holds the context (the
 * source's naming and the available labels); assembling it per source would reproduce "diagnostics
 * that point at a repo that does not exist" and "diagnostics that name the session's repo instead
 * of the cross-repo target" once for every source.
 *
 * The parts whose vocabulary differs per source are folded into one
 * {@link ../source.ts SourceOrigin} — the message shows `label` (HF: `repo owner/name @ <SHA>` /
 * local: the directory label as `localDirectory` formats it), and the structured fields (`repo` /
 * `revisionSha`) are named **only by the sources that have them**.
 */

import {
  type AvailableLabels,
  HubFetchError,
  IntegrityError,
  ManifestFormatError,
} from "./errors.ts";
import { crossRefOf, type FileRef, MANIFEST_FILENAME, MAX_MANIFEST_BYTES } from "./manifest.ts";
import type { LoadedManifest } from "./session.ts";
import type { SizeViolation, SourceOrigin } from "./source.ts";

/** The verb in the failure message (the surfaces split between the fetch and the prefetch verb). */
export type FetchVerb = "fetch" | "prefetch";

/**
 * The part of the naming that diagnostics use. **It does not include the failure origin
 * ({@link ../source.ts SourceOrigin.integrity})** — the naming of a cross-repo reference can be
 * built from the declaration (repo / revision), but whether the source that actually returned the
 * bytes is local or network is not determined by the declaration (`localDirectory` formally accepts
 * a remote source as a cross-repo target). The failure origin is named by the source that calls
 * {@link ../source.ts SizeViolation}.
 */
type OriginNaming = Omit<SourceOrigin, "integrity">;

/**
 * Diagnostics assembly used by the asset fetch surfaces (surfaces before the manifest fetch use the
 * functions below).
 */
export type FetchContext = {
  /**
   * The session's naming shown in diagnostic messages ({@link ../source.ts SourceOrigin.label}).
   */
  readonly session: string;
  /**
   * The list of "what this repo offers right now" shown on failure (a required field of every hub
   * error).
   */
  readonly available: AvailableLabels;
  /**
   * The naming of the place actually fetched from. A cross-repo reference names **the declared
   * side** (naming the session's source would produce a diagnostic that points at a path that does
   * not exist there).
   */
  readonly originOf: (ref: FileRef) => OriginNaming;
  /** The error for a byte count that disagrees with the manifest's `size`. */
  readonly sizeViolation: (ref: FileRef) => SizeViolation;
  /**
   * Wraps a failure from the source (404, auth, cache I/O, a missing local file, …) with context.
   */
  readonly fetchFailure: (ref: FileRef, verb: FetchVerb, cause: unknown) => HubFetchError;
};

/**
 * Extracts only the part of the naming that goes into the error's structured fields (absent fields
 * are left out).
 */
const identityOf = (
  origin: OriginNaming,
): { readonly repo?: string; readonly revisionSha?: string } => ({
  ...(origin.repo === undefined ? {} : { repo: origin.repo }),
  ...(origin.revisionSha === undefined ? {} : { revisionSha: origin.revisionSha }),
});

/**
 * The naming of the coordinates a cross-repo reference declares. **It does not ask the source** —
 * looking up the target's source would fire the fail loudly for an unmapped target
 * (`sources/local.ts`) in the middle of assembling the error and hide the real failure. The
 * declaration itself (the repo / revision of ADR 0038 §7) is primary information the manifest
 * holds.
 */
const crossOrigin = (repo: string, revision: string): OriginNaming => ({
  label: `repo ${repo} @ ${revision}`,
  repo,
  revisionSha: revision,
});

export const createFetchContext = (loaded: LoadedManifest, session: SourceOrigin): FetchContext => {
  const available: AvailableLabels = loaded.manifest.available;
  const originOf = (ref: FileRef): OriginNaming => {
    const cross = crossRefOf(ref);
    return cross === undefined ? session : crossOrigin(cross.repo, cross.revision);
  };
  return {
    session: session.label,
    available,
    originOf,
    sizeViolation: (ref) => (actual, integrity) => {
      const origin = originOf(ref);
      return new IntegrityError(
        `${ref.path}: byte count does not match the manifest's size` +
          ` (expected ${ref.size} / actual ${actual} — ${origin.label})`,
        {
          ...identityOf(origin),
          path: ref.path,
          expected: String(ref.size),
          actual: String(actual),
          // The failure origin is named by **the source that actually read the bytes** (for a
          // cross-repo reference it differs from the session).
          source: integrity,
          available,
        },
      );
    },
    fetchFailure: (ref, verb, cause) => {
      const origin = originOf(ref);
      return new HubFetchError(`${ref.path}: ${verb} failed (${origin.label})`, {
        ...identityOf(origin),
        path: ref.path,
        available,
        cause,
      });
    },
  };
};

/**
 * Resolving the generation identifier failed. The manifest has not been fetched yet, so the
 * available labels are empty (one of the two legitimate empty cases listed at the top of
 * `errors.ts`).
 *
 * The naming is the one **before generation resolution** ({@link ../source.ts SourceDriver.origin}
 * — for HF it includes the requested ref). A source without the notion of generations cannot fail
 * here, so the guidance targets mutable refs.
 */
export const revisionResolutionFailure = (origin: SourceOrigin, cause: unknown): HubFetchError =>
  new HubFetchError(
    `failed to resolve the revision (${origin.label}). A mutable ref cannot start` +
      ` offline — passing a commit SHA as revision avoids the resolve request altogether`,
    { ...identityOf(origin), cause },
  );

/** `karume.json` exceeded the byte limit. */
export const manifestOversize =
  (origin: SourceOrigin): SizeViolation =>
  (actual) =>
    new ManifestFormatError(
      `manifest: ${MANIFEST_FILENAME} exceeds the limit of ${MAX_MANIFEST_BYTES} bytes` +
        ` (${actual} bytes — ${origin.label})`,
    );

/** Fetching `karume.json` itself failed. */
export const manifestFetchFailure = (origin: SourceOrigin, cause: unknown): HubFetchError =>
  new HubFetchError(`${MANIFEST_FILENAME}: fetch failed (${origin.label})`, {
    ...identityOf(origin),
    path: MANIFEST_FILENAME,
    cause,
  });
