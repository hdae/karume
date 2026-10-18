# Working instructions for Codex

This file does not duplicate the conventions. The conventions have three sources of truth; reread them before every
piece of work or review.

- This repository's [`CLAUDE.md`](CLAUDE.md) — the project conventions (layout, verification commands, cross-cutting
  invariants, where docs go). When it conflicts with the general conventions, it takes precedence.
- [`.claude/ACTIVE_DESIGN.md`](.claude/ACTIVE_DESIGN.md) — the index of the current design focus and pitfalls.
  Always read it before planning or reviewing.
- `~/.claude/CLAUDE.md` and `~/.claude/rules/*.md` — the maintainer's general conventions (language, how to explain,
  how to reach agreement, root-cause investigation, tests, commits, TypeScript).

Explicit instructions from the maintainer take precedence over every convention.
The current state of the design (the manifest version, the development machine's GPU, how verification is run and so
on) is not written here; read the sources of truth.
With the maintainer's permission, this file may be edited as needed to match the working policy.

## What Codex reads differently

- The Claude-only means of execution in the sources of truth (specifying model names, Workflow, skills, delegation to
  subagents, memory) are not assumed to be available as Codex means.
  Follow the features available in this session and their instructions; do not report having used a feature you did
  not use.
- An explicit work request or instruction to continue is treated as approval for its scope.
  Do not ask again at each step of an approved plan. For a change beyond the scope or a broken premise, stop and
  present the diff and the decision point.
- The working branch follows the maintainer's instructions. Without an instruction, work on the current branch and do
  not switch on your own. Check the branch and the diff at the start and before committing, and do not overwrite
  uncommitted changes of other work.
- Verification follows the "Verification commands" section of `CLAUDE.md` (including lane runs and per-environment
  reference values). Do not run GPU verification in parallel with benchmarks. Do not write "verified" for a check on
  real hardware that was not done.
- The React conventions apply only when React is involved. Do not bring in dependencies or a UI foundation because of
  the general conventions.
- The final report is in Japanese, in a form readable without the conversation history (what was done / list of
  commits / what is unfinished / what to do next).

## Operating instructions kept here until they move to a source of truth

Instructions the maintainer gave for Codex's work that are not yet in a source of truth (`CLAUDE.md` / `docs/`).
Once moved to a source of truth, delete them here.

- For an experiment, create a new dedicated directory whose name includes the date and time and the purpose, and save
  the preparation scripts, candidate code, logs and results in it (location: [docs/assets-layout.md](docs/assets-layout.md)).
  If the target already exists, fail (do not use a `mkdir` that tolerates an existing one; create files exclusively).
  A re-run also uses a new output location, and a dependent step runs only when the previous step succeeded.
- When asking the maintainer to re-run a benchmark, reflect the required models, comparison targets and run settings in
  the initial selection of the comparison screen, so that the start button alone runs the requested measurement.
  Explain what was selected and which comparisons are needed.
- An unexpected problem that requires stopping means an incident such as data loss or a wrong operation.
  Performance regressions, increased numeric error, test failures and rejected candidates are ordinary subjects of
  investigation and are not reasons to stop on their own.
