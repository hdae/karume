/**
 * Normalises the first argument of `fromPretrained` (string | {@link HubRepoRef} |
 * {@link DistributionSource}) into the shape passed to hub (**pipeline-independent shared
 * processing** — every family's `fromPretrained` uses it the same way).
 *
 * MUST: not exported from the barrel. It is an internal mechanism that keeps the argument spelling
 * uniform, not a surface the user touches.
 *
 * ## MUST: no default for the source (ADR 0073 decision 2, "make ref optional", was withdrawn in 0.5.0)
 *
 * A default lets code be written that never spells out "which repo it fetches from", and the fetch
 * target moves silently the moment the package version is bumped (pin constants are updated even in
 * patch releases). The source is the caller's decision and **must always be spelled out**. In TS the
 * argument is required, so type checking fails; an argument-less call from untyped JS is rejected by
 * {@link toRepoRef} with an example — `undefined` is not allowed to slide down to hub and become a
 * failure far from its cause ("hit a URL whose repo is undefined").
 */

import { type DistributionSource, type HubRepoRef, isDistributionSource } from "@karume/hub";

/**
 * Characters allowed in one segment of an HF repo name (**the same rule** as the cross-repo
 * reference check in hub's `manifest.ts` — both are "raw strings the fetch layer spells into a
 * URL", so there is one accepted set of spellings).
 */
const REPO_SEGMENT_RE = /^[A-Za-z0-9_-][A-Za-z0-9._-]*$/;

/**
 * Checks that `repo` is spelled as an HF `owner/name` (**before the fetch layer**).
 *
 * Without this check a call that passes a local directory path **goes straight to HF**
 * (`./models/foo` is spelled into the URL and what comes back is 401 / 404 — a diagnosis far from
 * the cause, "the fetch target does not exist"). Spellings that can only be paths (leading `./`
 * `../` `/`, trailing `/`, two or more slashes, empty segments, disallowed characters) fail here.
 *
 * NOTE: a **path that looks like a legal `owner/name`**, such as `models/karume-gemma4`, cannot be
 * caught (it is a perfectly valid HF repo name, and nothing in this argument tells them apart).
 * This check looks only at the spelling; "does it exist" is the fetch layer's job.
 */
const assertRepoSpelling = (repo: string, where: string): void => {
  const segments = repo.split("/");
  if (segments.length !== 2 || !segments.every((segment) => REPO_SEGMENT_RE.test(segment))) {
    throw new Error(
      `${where}: repo '${repo}' が HF の 'owner/name' でない` +
        `（2 セグメントちょうど・許可文字 [A-Za-z0-9._-]・先頭ドット禁止）`,
    );
  }
};

/**
 * Turns `ref` into a {@link HubRepoRef}. A string reads as `{ repo }` (= tracks `main`).
 *
 * The spelling is checked down to being an HF `owner/name` ({@link assertRepoSpelling}) — a call
 * that passes a local directory path fails before it is spelled into a URL.
 *
 * The type accepts `undefined` **not to loosen the caller's type** — `ref` of `fromPretrained` is
 * required, and `undefined` arrives only from outside type checking (JS). Removing from the type
 * the very value this function must reject would make the check itself impossible to exercise
 * plainly.
 *
 * @param where Subject of the diagnostic (`"AnimaPipeline.fromPretrained"`).
 * @param sourcesExample The spelling that looks up this family's source table
 *   (`ANIMA_SOURCES["anima"]` — ADR 0092). Omitted for families without a public distribution repo
 *   (so no non-existent identifier is suggested).
 */
export const toRepoRef = (
  ref: string | HubRepoRef | undefined,
  where: string,
  sourcesExample?: string,
): HubRepoRef => {
  const repo: unknown = typeof ref === "string" ? ref : ref?.repo;
  if (ref === undefined || typeof repo !== "string" || repo.length === 0) {
    throw new Error(
      `${where}: repo が必須（取得元に既定は無い）— ` +
        '{ repo: "owner/name", revision: "<40 桁の commit SHA>" } を渡すか、' +
        (sourcesExample === undefined
          ? "リポ名の文字列（= main 追従）を渡す"
          : `このパッケージ版が検証した ${sourcesExample} を渡す`),
    );
  }
  assertRepoSpelling(repo, where);
  return typeof ref === "string" ? { repo } : ref;
};

/**
 * Maps the first argument of `fromPretrained` onto the first argument of `loadManifest`. A source
 * handle ({@link DistributionSource} — the opaque value returned by `localDirectory` /
 * `denoDirectory` / `nodeDirectory`) **passes through as is**: it is not an HF repo, so running it
 * through the {@link toRepoRef} check that requires an `owner/name` spelling is pointless (a local
 * distribution would fail with "repo is required").
 *
 * MUST: leave the discrimination to hub's {@link isDistributionSource} — writing a structural test
 * (`"repo" in ref`) or a brand-field check here would let a misspelled `HubRepoRef` pass as a
 * source.
 */
export const toManifestSource = (
  ref: string | HubRepoRef | DistributionSource | undefined,
  where: string,
  sourcesExample?: string,
): HubRepoRef | DistributionSource =>
  isDistributionSource(ref) ? ref : toRepoRef(ref, where, sourcesExample);
