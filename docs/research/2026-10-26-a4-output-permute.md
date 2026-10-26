> This document is a point-in-time snapshot (2026-10-26) of perf-ledger F-3's last item, A4: the fused attention absorbs
> its output permute on the online route, measured on the RTX 3080 Ti under Node + Dawn (Vulkan). It is not a design
> source of truth (the design is ADR 0040's addendum of 2026-10-26). Raw material is under `outputs/diag/a4-2026-10-26/`
> (not tracked).

# A4: the online attention stores O token-major

## 1. Summary

- On the online route the attention writes O token-major itself, so the strided copy after it is gone: one dispatch fewer
  per attention, bit-identical (every sha256 row of the Wan2.2, Wan2.1 and anima lanes unchanged).
- Per Wan2.2 DiT forward (30 self- and 30 cross-attentions): −14.8 / −14.9 ms of GPU time at 33 frames and −44.0 / −48.7
  ms at 121 frames (two in-process runs, §3). The token-major store itself costs nothing measurable.
- The slot backing does not grow (estimate = slot table): Wan2.2 DiT 717.8 / 2,457.2 MiB at S 7,920 / 27,280 with or
  without O absorbed.

## 2. The forward-level A/B could not resolve it

Base = wip/0.15 `71a9c343`, A4 = `b27a7a37`; Wan2.2 default preset, separate processes in ABBA order, cooled to 60 °C
before each run (`ab.zsh`, `summarize.mjs`).

| Per 33f forward (GPU, mean of forwards 2–4 of two runs) | Base | A4 |
| --- | ---: | ---: |
| Total | 2,816.3 ms | 2,810.0 ms (−6.3) |
| strided copies | 17.5 | 2.4 (−15.15) |
| attention_online | 684.6 | 689.0 (+4.4) |
| linear (kernel unchanged) | 1,910.7 | 1,914.9 (+4.2) |

The forwards drift by about +20 ms within one run (heat), and the unchanged linear moved by +4.2 ms; the cooler first base
run alone carries the attention's +4.4 ms (the later base run and both A4 runs agree within 1 ms per forward). The −6.3 ms
is below the plan's 7.6 ms threshold but inside this noise, so the attention kernel was measured in one process (§3).

## 3. In-process kernel A/B

`kernel-ab.ts`: per attention, (1) the online kernel storing O head-major, (2) the same followed by the strided permute
(the graph before A4), (3) the online kernel storing O token-major (A4); rotating order, GPU timestamps per pass, H 24,
D 128, `dp4a`, `sg32`, the production windows. Medians of the per-repetition differences:

| Shape | Saved by A4 per attention, run 1 / run 2 | Token-major store cost, run 1 / run 2 |
| --- | ---: | ---: |
| self, S 7,920 (33f) | 0.251 / 0.253 ms | −0.014 / −0.013 ms |
| cross, 512 keys (33f) | 0.242 / 0.244 ms | 0.002 / 0.000 ms |
| self, S 27,280 (121f, 5 windows) | 0.651 / 0.797 ms | 0.568 / 0.047 ms |
| cross, 512 keys (121f) | 0.815 / 0.826 ms | 0.003 / 0.002 ms |
| **per 33f forward** | **14.8 / 14.9 ms** | — |
| **per 121f forward** | **44.0 / 48.7 ms** | — |

Runs of 30 and 40 repetitions. The 121f self-attention's spread includes this card's sustained-load slow state (one
repetition at about 400 ms), so its medians are the least certain; the 33f numbers and both cross-attention shapes agree
between runs within 0.01 ms.

## 4. Not measured

- D3D12 / Metal (the RTX 5070 Ti) — the playground's 17-frame quick-check hash is the end-to-end check there.
- Wall time at 121 frames beyond two first forwards per side (15,618 / 15,677 ms, inside the run-to-run spread).
