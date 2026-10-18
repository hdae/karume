# 0081: Shard specification v2 — graph-only shard, one 1 GiB limit, balanced filling

- Status: superseded in practice (2026-10-18 triage): decisions 1 and 3 by
  [0108](0108-container-format.md) (accepted 2026-09-22, in effect 2026-09-24); decision 2 by
  [0090](0090-shard-spec-v3-tensor-pieces.md) (2026-09-02) and then 0108; decision 4 by 0108
  (2026-09-24, when step 3d retired the old writer) and the open consideration of decision 5 by 0108
  (2026-09-22). No decision remains in force. Originally accepted 2026-08-30
  (the maintainer's ruling: "the graph lives apart, one limit, remainders removed by balancing").
- Date: 2026-08-30
- Related: ADR [0070](0070-shard-loading-admission.md) (shard loader contract — this ADR replaced
  the writer rule of its decision 1, "sequential filling plus tail slack"; its decision 2 RAM peak
  target and decision 5 admission hold literally only with this ADR) /
  [0071](0071-manifest-v3-shards.md) (this ADR withdrew "1 GiB or less is a single file"; the
  `shards` schema was unchanged) / [0063](0063-safetensors-physical-layout.md) (physical layout of
  one shard, unchanged) / [0041](0041-manifest-v2.md) (repository layout, unchanged).
- Today the distribution is the `krm` container (ADR 0108, `docs/container-v1.md`). Only the
  migration CLI reads old shard lists (`tools/exporter/src/karume/legacy.py`).

## Context

The split rule of the ADR 0070 addendum of 2026-08-29 filled shards sequentially in write order with
tail slack, so **the leading graph shard also carried real weights up to a 1 GiB data section**:

- Reading the graph meant reading those weights. The graph shards of the four anima Base f16
  components totalled 2,460.9 MiB, and the resident size right after `fromPretrained` was
  2,461.6 MiB (ADR 0070 addendum of 2026-08-30).
- Not holding `PreparedModel` in models cut that to 0.7 MiB, but every Session construction then
  re-read and re-parsed the 2.46 GiB of graph shards (512², 2-step `generate`: 6.0 s → 7.8–12.6 s).
- ADR 0070 decision 5 admission assumed "fetch only the graph first, then decide", which did not
  hold while one graph meant a multi-GB download.

The tail slack (an unclosed remainder up to 1.5 GiB is packed in) came from keeping re-dist output
byte-identical with single-file distributions, which created a second limit. One re-upload removes
that constraint.

## Decision

The rules split into a **reader contract** (format invariants, checked by `karume verify` /
`karume dist`) and a **writer policy** (shard count and cut points, at the writer's discretion;
changing it keeps existing distributions readable).

### 1. Reader contract: shard 0 is graph-only (zero tensors in the data section)

(Superseded by ADR 0108 on 2026-09-24: the graph lives in the part 0 descriptor.)

The first shard holds **only** `__metadata__.karume_ir`; its data section is empty (the 8-byte header
length plus the header JSON). The container stays safetensors. Fetching the graph then costs a few
MB, so admission (ADR 0070 decision 5) and the RAM peak target (decision 2, O(largest weight shard))
hold literally.

### 2. Reader contract: one limit, `SHARD_BYTE_LIMIT` (1 GiB)

(Superseded by ADR 0090 on 2026-09-02: 256 MiB, measured on the file length; then by ADR 0108 on
2026-09-22: part lengths chosen by the writer and independent limits, decisions 6 and 9.)

Every shard's data section is at most 1 GiB, with no exception by position; `SHARD_TAIL_LIMIT`
(1.5 GiB tail slack) is **abolished**. "A weight + scale pair over 1 GiB cannot be distributed"
**follows** from this one limit and is not a separate constant: two limits make the rule decide
which limit applies, which is exactly what this correction cleaned up.

### 3. Reader contract: always split (no fat graph shard, no single-file distribution)

(Superseded by ADR 0108 on 2026-09-24: split and single forms share one byte order and differ only
in where the cuts are, 0108 decisions 2 and 3.)

A component with at least one tensor has a graph shard plus one or more weight shards. **A single-file
distribution carried no acceptance guarantee.** The writer and the Python verify enforced it (the
old writer and its read-back check were removed with 0108 step 3d). Kept: declaration order,
co-shard atomicity of a weight and its scale, no duplicates across shards, `karume_ir` only in
shard 0, at most 1024 shards (ADR 0070 decision 1 / ADR 0071 decision 2).

### 4. Writer policy: minimal count, then balanced

(Superseded by ADR 0108 on 2026-09-24: the container writer fills parts of a fixed length in
order, opening a new part when a group does not fit — `docs/container-v1.md` §4.2,
`_plan_data_parts` in `tools/exporter/src/karume/container.py`.)

- Count k = **the minimal number of contiguous splits under 1 GiB** (greedy filling gives it).
- With k fixed, **balance**: each shard targets remaining bytes ÷ remaining shards; units (weight +
  scale pairs) are not split, and a **suffix-feasibility guard** cuts only where the remaining units
  still fit the remaining shards. Deterministic (same input → same split → same bytes); 3.2 GiB
  becomes `[0.8, 0.8, 0.8, 0.8]`, not `[1, 1, 1.2]`.

NOTE: under contiguous order and pair atomicity, k can exceed `ceil(total / limit)` (three 0.6 GiB
pairs need three shards). Reordering would change the reader contract, so it was not done.

### 5. Door: cut-point preferences are a future writer-policy extension

Preferences such as "cut at layer boundaries" or "cut per MoE expert" can be added without changing
the reader contract; not implemented until partial or per-layer loading needs them.

The open consideration of 2026-08-30 (separate the graph from the shard sequence as an independent
asset at the next format change) was taken up by ADR 0108 (superseded by ADR 0108 on 2026-09-22:
the graph container `krg` and the part 0 descriptor, decisions 1 and 4).

## Consequences

- Every distribution needed re-dist, HF re-upload, pin updates and rebuilt cross-references; local
  mirrors written under the old rule failed `karume verify` by design.
- More files: an asset under 1 GiB became two files, the 1.11 GiB text encoder three (the re-upload
  used the fragmentation countermeasures of `docs/release-runbook.md`).
- Runtime, hub, manifest schema and `karume.json` were unchanged; only file names, counts and bytes
  changed.
- The ADR 0070 acceptance condition "re-dist is byte-identical" was abandoned in exchange for
  decision 1; the writer, verify and dist lost the single-file branch.

## Addendum (2026-09-02) — writer target 256 MiB

A writer target `SHARD_TARGET_BYTES` = 256 MiB (effective target = max(target, largest unit)) was
added below the 1 GiB limit to cut the host RAM peak (ruling of 2026-09-02); ADR 0090 replaced it the
same day. Details: [research/2026-09-02-shard-size-ram-peak.md](../research/2026-09-02-shard-size-ram-peak.md).

## Addendum (2026-09-22) — decisions 1 and 3 overridden by ADR 0108

ADR 0108 took effect on 2026-09-24, when step 3d (`9e905d9f`) removed the old readers; the manifest
`shards` field was retired by ADR [0109](0109-manifest-v5-container.md).
