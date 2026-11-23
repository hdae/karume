# 0156: A shared geometry profile table predicts for the adapters of one vendor and architecture, adopted device by device

- Status: accepted (2026-11-23 — the maintainer approved the design; built on the development machine, unreleased)
- Date: 2026-11-23
- Revisits: the maintainer's ruling of 2026-11-20 in the [backlog](../backlog.md) that an Apple adapter reporting no
  description (the M5 in Chrome) gets a table derived from its own records. Such an adapter now gets a table shared by
  the measured Apple devices; the M5's own records make a measured table matched by its description. The ruling's other
  half stands: `apple-metal-3` keeps its id and matches `Apple M2` alone.
- Related: [0115](0115-geometry-profiles.md) (geometry profiles and the selection order) /
  [0116](0116-geometry-profile-row-buckets.md) (row-count buckets) / [0117](0117-app-geometry-tuning.md) decisions 4 and 5
  (provenance and matching) and its addendum 2026-11-12 (per op family, the base table) /
  [0152](0152-maintained-specs-and-declared-numeric-requirements.md) (maintained specs) — the rules as built are in
  [spec/geometry-profiles.md](../spec/geometry-profiles.md#shared-tables)

## Context

`selectGeometryProfile` picks a table in four steps: a match on vendor, architecture and description; then on vendor and
architecture with no description; then on vendor alone; then the default profile. Chrome without its developer-features
flag reports an empty description, so an ordinary browser user reaches the last three steps only. Every Apple table so far
names a description (`apple-metal-3` names `Apple M2`), so every Apple adapter in an ordinary browser — the M2 and the M5
among them — ran the default profile.

The derivation refuses records whose four adapter fields differ (ADR 0117 decision 4): a table's provenance holds one
adapter, and matching compares those four fields. The refusal also keeps two devices' timings apart, which matters on its
own: geometry ratios point in opposite directions between GPUs (ADR 0115), so a pooled mean can call a geometry that is
×1.6 on one device and ×0.8 on another a win.

The Apple M5 has a full geometry sweep (2026-11-23), and the M2 has records of the geometries the M5 adopts timed on the
M2 (the same day). On their own, the M5 adopts nine fields; the M2 is ×1.05 or faster on four of them in every case and
slower, or screened out by the quick look, in some case of the other five.

## Decision

1. **Three levels, by name.** A table matched by description is a *measured* table: that device's own records. A table
   matched by vendor and architecture is a *predicted* table: the adapter runs geometries measured on other devices of its
   architecture. No table matches: the *default*. The selection order does not change.
2. **A shared derivation, asked for explicitly** (`shared` of `deriveGeometryProfile`, `--shared` of the generator). It
   needs a vendor and an architecture, takes no description, and needs the records of at least two adapters distinct by
   the four fields. Anything else is refused.
3. **Adopted device by device.** The records are grouped by adapter. Inside a group every rule of a measured table holds
   as it does there. A geometry is eligible only when it passes that rule on every device, on the device's own cases
   against the device's own default; among the eligible ones the table takes the one with the largest geometric mean of
   the per-device geometric means (devices weigh equally), ties by name. A field some device has no case of keeps the
   default.
4. **Tile geometries only.** A shared table derives neither the online attention's key axis nor the GEMV choices of
   linear; the default profile's choices run on its adapters.
5. **The provenance names every device**: `provenance.adapters` in place of `provenance.adapter`. Matching compares vendor
   and architecture for that form, every family's keys as before, and the four fields for a measured table as before.
6. **The single-adapter guard stays** as the rule of a measured table, unchanged; the shared derivation is a second path
   with its own checks, not a looser first one.
7. **The Apple tables**: `apple-m5`, measured, matched by `Apple M5`, from the M5's full sweep; `apple-metal-3-shared`,
   predicted, matched by `apple` / `metal-3`, from that sweep and the M2's three records of 2026-11-23.

## Why

- **Why a shared table.** The predicted level is the only one an ordinary browser reaches, so a table there decides what
  most Apple users run. One device's table matched without its description would put that device's choices on every
  device of the architecture; the M2 and M5 records show five of the M5's nine choices slower on the M2 or not timed close
  enough there. Requiring the threshold on every measured device makes the table hold a statement on each of them: no
  adopted field is slower than the default in any case either device measured. For a device not measured it remains a
  prediction, and it says so.
- **Why the per-device rule and not pooled timings.** A ratio is a geometry against its own device's default on that
  device; ratios of two devices measure two different baselines. Pooling them over all cases lets the device with more
  cases, or with the larger gain, outvote the other, and a loss on one device can average into a win — the reason a
  measured table already requires every case of a field applies to every device. Devices weigh equally because the table
  predicts for devices that were not measured: none of the measured ones is more representative, and how many cases a
  device has of a field depends on which sweep was taken.
- **Why the guard stays for a measured table.** A measured table claims to be one adapter's own answer, and matching
  checks it on all four fields. Letting it take a second device's records would make the claim false while every check
  still passed.

## Alternatives not taken

- **Pooling the timings** of all devices' cases into one geometric mean: see above.
- **Nearest-neighbour matching** (an unmeasured `Apple M3` taking the M2's or the M5's measured table by the closeness of
  the name): `selectGeometryProfile`'s MUST forbids it — the sha256 reference rows are kept per environment key, and each
  must say which table that machine ran with.
- **Widening `apple-metal-3`'s match** to vendor and architecture: the M2's choices on every Apple adapter, which the
  M5's records do not support.
- **Deriving the attention key axis and the GEMV choices** of a shared table: there is no rule yet for combining those
  across devices (the GEMV derivation reads a reference time repeated within one adapter's records), and the M5's GEMV
  records are still being taken.

## Consequences

- An Apple `metal-3` adapter with an empty or unmeasured description runs `apple-metal-3-shared`: four fields differ from
  the default (`gemmRows` 33 to 64 rows, both `conv2d` fields, `i8a8.linear` up to 256 rows). The M2 and the M5 with
  Chrome's flag run their measured tables. No NVIDIA adapter's table changes.
- `apple-m5` and `apple-metal-3-shared` carry no `linearGemv`; their adapters run the default profile's GEMV choices, as
  every Apple adapter without a description did before.
- The sweep's `quick+` and evaluation set `k88` measure the new tables' geometries too (more rows per sweep).
- `geometryProfileMismatch` does not tell the devices of a shared table's architecture apart; that is the table's
  purpose. A list of fewer than two adapters is a mismatch.
- Left: the M5's GEMV choices for `apple-m5` (its GEMV records are being taken); an M2 full sweep, which re-derives
  `apple-metal-3`'s tile families and lets the shared table compare every candidate on both devices; the M1, M3 and M4
  are not measured.
