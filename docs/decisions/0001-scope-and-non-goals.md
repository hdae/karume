# 0001: Scope and non-goals

- Status: accepted (2026-08-01, the maintainer's ruling); partly superseded by
  [0037](0037-karume-monorepo.md) §1 (flat layout, 2026-08-05) and
  [0127](0127-node-dawn-toolchain.md) decisions 3 and 6 (scope sentence, bare-Node non-goal; ruled
  2026-10-18, in effect since 0127 was accepted on 2026-10-29).
- Date: 2026-08-01

## Decision

Karume is a **general-purpose WebGPU neural-network inference runtime that runs on both Deno and the
browser** *(superseded by ADR 0127 decisions 3 and 6 on 2026-10-18: browser and Node, importable
from Deno; Deno numerics best effort from 0.15)*. It is published on JSR as `@karume/runtime` (npm
in addition is open — ADR 0127 Open points). Models come from PyTorch (`torch.export`).

- **Independent of the prototype** (the earlier experimental project): its code and types are never
  imported, only read and ported. This project carries out the move to JSR / Deno that the
  prototype's decision 0013 chose.
- Not a plain port: the prototype's measured bottlenecks, bug fixes and design hardening are built
  in from the start (the ten weaknesses of [recon](../research/2026-08-01-prototype-recon.md) §9).
- Custom WebGPU kernels (WGSL generated at run time). The reasons for not using WebNN, inherited
  from the prototype's decision 0001 (no int4 type, no precision-conformance requirement, no custom
  shaders, no in-place operations), SHOULD hold until WebNN or its implementations change
  *(re-graded 2026-10-18 to SHOULD: they describe another specification at one point in time)*.

## Non-goals

- ONNX input (only `torch.export` → Karume's IR; inherited from the prototype). A WASM fallback.
- An immediate WebNN backend. The IR keeps four boundaries open for a later backend (an NPU entry):
  no baked-in layout, logical quantisation kept, fine op granularity, async initialisation allowed.
- Environments without `navigator.gpu` (bare Node, etc.) *(superseded by ADR 0127 decision 6 on
  2026-10-18: Node with an application-supplied WebGPU is supported; only an environment with no
  WebGPU implementation stays out)*.
- MiniMax-M3-class models (428B / 23B MoE, block-sparse MSA) in the browser, by memory size (ruling
  of 2026-08-16; autoregressive work was accepted on Gemma 4 E2B / MiniCPM5-1B — [backlog](../backlog.md)).

## Consequences

- The flat repository layout was replaced on 2026-08-05 by the monorepo of ADR 0037 §1.
- The exporter is new code inside Karume (the maintainer's ruling), porting the prototype's knowledge.
