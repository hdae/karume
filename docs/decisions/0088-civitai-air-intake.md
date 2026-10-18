# 0088: Civitai intake through an AIR-addressed recipe command, with a license review by the agent beforehand

- Status: accepted (2026-09-01 — the maintainer's ruling "1. OK / 2. normalise to name + version")
- Date: 2026-09-01
- Related: [0077](0077-model-version-naming.md) (upstream version in the model name — this ADR adds a
  normalisation rule) / [0087](0087-anima-official-extra-repos.md) (the extra repository, home of
  third-party fine-tunes) / [0073](0073-models-source-pin.md) (pinning the default source)

## Context

Adding a third-party fine-tune to the extra repository (ADR 0087) was manual throughout: download the
checkpoint from Civitai in a browser, place it in `inputs/anima/`, and keep the provenance only in
the commit message and a hand-written `card.py`. sha256 checks and naming were left to people, and
"which version, when, under which terms" was not machine-readable.

Facts from a first check (2026-09-01, wai `2544636@2983680`):

- The metadata API (`/api/v1/model-versions/{id}`, `/api/v1/models/{id}`) needs no authentication,
  and **the model-versions response returns the AIR
  (`urn:air:anima:checkpoint:civitai:2544636@2983680`) as a server-issued value**, with sha256, file
  types (Model / Text Encoder / VAE), the four permission fields (allowNoCredit /
  allowCommercialUse / allowDerivatives / allowDifferentLicense), usageControl and description (HTML).
- **Only the file download returns 401** — it needs an API token.
- The API fields alone do not settle the license: the description may add terms that disagree.

## Decision

**Intake goes through the `anima.civitai` command (AIR / URL), responsible for "fetch + verify +
record" only.** Conversion to diffusers stays with `anima.single_file`; export and registration in
distribution / card follow the existing procedure (the command does not edit the registration).

1. **Two entry forms, AIR and URL.** A URL without a version only lists the versions (nothing is
   fetched). The server-issued AIR wins; if it differs from the locally built one, it is taken with a
   warning.
2. **The model name is normalised mechanically to `anima-<model name>-<version name>`** (ruling of
   2026-09-01 — revises ADR 0077's "the spelling is not normalised"): lower case, characters outside
   `[a-z0-9._-]` become `-`, parenthesised notes and the token `anima` in the model name are removed.
   This turns the existing names (`anima-wai-v1.0`, `anima-copycat-20260610`) into a rule; **the
   verbatim upstream spelling stays in `civitai.json`**, so matching the source page stays possible.
   `--name` overrides it (the door for a human ruling — as in 0077, the automatic default does not
   replace a ruling).
3. **Provenance is chained:** `inputs/anima/civitai-<versionId>/` holds the weights and
   `civitai.json` (AIR, verbatim version name, sha256, permission fields, description, fetch time —
   **owned by the machine**, never edited by people); `single_file` carries the adjacent
   `civitai.json` into `source_provenance.json`, unbroken up to dist.
4. **sha256 is checked against the API value before finalising** (`.part` → rename on match); a
   mismatch fails loudly. An existing file with a matching sha is not fetched again (idempotent).
5. **License: the agent reviews beforehand** (ruling of 2026-09-01). The command only records the
   permission fields and the text; it **neither judges nor enforces a confirmation flag**. The agent
   doing the intake compares fields and text before and after fetching, records disagreements and
   grey areas in `license-review.md` in the same directory (written by people and agents, never by the
   command) and reports to the maintainer. Anima derivatives are assumed to inherit the base license
   (v1.2); an upstream claim of another license goes to a ruling.
6. **The download token is the env `CIVITAI_API_TOKEN`, passed as the `?token=` query** (an
   Authorization header may conflict with the signed URL on the 307 redirect to S3); output masks it.

## Consequences

- `inputs/anima/civitai-<versionId>/` is the canonical intake location (`docs/assets-layout.md`),
  unique by versionId; the human-facing name is `derived_name` in `civitai.json`.
- The machine record (`civitai.json`) and the human review (`license-review.md`) are separate files,
  so a re-fetch does not erase the review.
- Registration (`distribution.py`, `card.py`) stays hand-written — intake automation does not
  propose registrations (adding a model is a ruling each time).
- Only the primary (Model-type) file is taken; Text Encoder / VAE files are only listed (what a
  checkpoint lacks is shared with base, as `single_file` assumes).
- Other sources (HF directly, manual placement) work as before; without `civitai.json`, `single_file`
  writes the former provenance.

## Addendum (2026-09-21) — how the main file is chosen (the Consequences' "primary (Model-type)" becomes a decision)

- **Only files with `type == "Model"` are candidates.** The primary flag alone would take a VAE or
  Text Encoder that upstream marked primary (upstream does not guarantee primary means Model).
- **Without a primary file, a single Model-type file is taken.** Such versions exist; with several
  Model-type files the command fails loudly — the main file cannot be decided mechanically.
