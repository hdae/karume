/**
 * The `pipeline` contract name of the umT5 encoder (distribution `karume-umt5-xxl`) and the schema
 * validation of its `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns and validates
 * its schema).
 *
 * `pipelineConfig` accepts only the empty `{}`: the numbers the run needs (input / output shapes,
 * width, valid-length range, bucket configuration) are held by the graph declarations and host
 * constants (`./relative-position.ts`), and there is no field for the distribution to declare
 * (`UMT5_PIPELINE_CONFIG` in the recipe `wan/umt5_distribution.py`). If a field appears it fails
 * rather than passing unread (unknown keys fail loudly — a misspelt declaration must not be ignored
 * silently).
 *
 * The table of public distribution repositories ({@link SOURCES}) also lives here (ADR 0092 —
 * a family's `config.ts` is where the table lives).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import type { HubRepoRef } from "@karume/hub";
import { assertAllowedKeys, readRecord } from "../config/readers.ts";

/** The `pipeline` contract name and the major this implementation accepts (ADR 0038 §1). */
export const UMT5_PIPELINE_NAME = "umt5-encoder";
export const UMT5_PIPELINE_MAJOR = 1;

/**
 * umT5's **table of public distribution repositories** (ADR 0092 — one table per family, **no
 * default entry**). The values are **the sources this package version verified** (pinned commit
 * SHAs — ADR 0073).
 *
 * A key is the basename of the HF repository name without `karume-` (`"karume-" + key` gives back
 * the basename — the check in `tests/sources_test.ts` holds this invariant). One family = one
 * repository:
 *
 * - `"umt5-xxl"` = `hdae/karume-umt5-xxl` (the encoder of model `"xxl"` and the tokenizer asset)
 *
 * NOTE: this is not the recipe's `wan.sources.UMT5_SOURCES` (`tools/export-recipes/wan/sources.py` —
 * the table of upstream pins of the original `google/umt5-xxl` that the writer reads). This table
 * points at the public repository of the built distribution.
 *
 * Pass it **as an opt-in to follow the package version automatically** — to pin reproducibility
 * yourself, write your own `{ repo, revision }` instead of this table (`fromPretrained` has no
 * default).
 *
 * MUST: pin the revision to a commit SHA — branches and tags can be moved on the distribution side,
 * so the bytes this published package reads would change silently at the network's convenience (an
 * unrecoverable accident). A SHA removes the revision-resolution request itself, so an offline start
 * with a full cache works at the same time (ADR 0038). A user who needs to follow main chooses
 * `{ ...SOURCES["umt5-xxl"], revision: "main" }` explicitly.
 */
// NOTE: the revision is updated to the SHA of main after the upload, in §3 of the release runbook
// (docs/release-runbook.md) (inherits the maintenance duty of ADR 0073 decision 3 — by hand + the
// runbook check).
export const SOURCES = {
  "umt5-xxl": {
    repo: "hdae/karume-umt5-xxl",
    revision: "c66a8db12e005ba37027f2afa5b15efee4a681a7",
  },
} as const satisfies Record<string, HubRepoRef>;

/**
 * Checks the manifest's `pipelineConfig` (the raw value the hub passed through); only the empty
 * object is accepted.
 */
export const assertUmt5PipelineConfig = (raw: unknown): void => {
  assertAllowedKeys(readRecord(raw, "pipelineConfig"), [], "pipelineConfig");
};
