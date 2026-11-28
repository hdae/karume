/**
 * Geometry profile `apple-m5` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for the adapter match: vendor apple, architecture metal-3, description Apple M5.
 * The runtime only selects the table by the adapter's match; nothing is measured at run time.
 * To change a value, take a sweep with the workbench's `geometry-sweep` entry and regenerate the
 * table with `node tools/geometry-profiles/generate.ts apple-m5 --records=<path>[,<path>…]`.
 * The GEMV choices of linear (`linearGemv`) come from records of the `gemv-sweep` entry
 * (`--gemv-records=<path>,<path>[,…]`).
 *
 * Derived in this generation: the GEMV choices of linear (GEMV sweep records).
 *
 * - outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json (sha256 7b776b0e1e10f7e570b52653f13630d250302753841a4df9e205819a0d8c6d70, 2026-11-23T16:43:28.766Z)
 *   workbench; Chrome 158 on macOS; backend metal (declared); timestamps not quantised; checkout 8c9364d8481f83e45f18cc29d813ffd3350d1841
 * - outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T18-03-03.json (sha256 5cbb8dbf059130b14f565fe7492f6382b74a6ce287f6f119d646d8a993181355, 2026-11-23T18:03:02.056Z)
 *   workbench; Chrome 158 on macOS; backend metal (declared); timestamps not quantised; checkout 8c9364d8481f83e45f18cc29d813ffd3350d1841
 *
 * Carried over from the registered table at commit 0fe1e1d3556834806be157c1d5ae5f704ccb60f4: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d.
 * Their adoption lines are in `git show 0fe1e1d3556834806be157c1d5ae5f704ccb60f4:packages/runtime/src/kernels/geometry-profiles/apple-m5.ts`.
 *
 * GEMV choices, one steady record: a record is steady when 20 or more of its groups have a
 * baseline spread and at most 20 % of those spread beyond ×1.050. A shape a steady record
 * measured is decided by the steady records: at one row a candidate qualifies when its check held
 * (identical class: the baseline's words; reorder class: inside the band) and its time beats the
 * default's choice in the same group by at least ×1.065 × its gap × the reference's gap (a gap: the
 * second-smallest round ÷ the smallest) in every steady record that times it — a group with a pass
 * under 20 ms, or whose reference re-timed after the rounds departs beyond that bar, is no
 * evidence —; a parallel row's kernel at 4 and 8 rows has a time in those records and is slower than
 * the default's choice in none, and a row block is taken only where it beats the row's own kernel
 * by the bar at one row count and is slower at none. Among those, the largest gain; a choice the
 * table already held stays unless a challenger beats it by the bar in the same group.
 * Any other shape, two records: a candidate replaces the default profile's choice only when it is
 * at least ×1.100 faster than that choice in every record, with its check held in every one
 * (the parallel rows also measured at every row count of the parallel range and not slower at
 * any); among those, the one whose smallest gain is largest. The choice's time is its fastest over
 * the records, and only when a second record repeats it within ×1.100. Shared sequential keys whose
 * cases disagree keep their choice, and so do the rows with packed activations.
 *
 * - linearGemv (the GEMV choices of linear), from 2 GEMV sweep records (the rule of one steady record at the margin ×1.065):
 *   - [1] outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json (2026-11-23T16:43:28.766Z, set full): 345 groups — 345 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current; steady (20 of 345 groups beyond ×1.050, 5.8 %): read by the rule of one steady record
 *   - [2] outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T18-03-03.json (2026-11-23T18:03:02.056Z, set full): 345 groups — 345 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current; steady (17 of 345 groups beyond ×1.050, 4.9 %): read by the rule of one steady record
 * - identical class (sequential.shapes, M = 1): 2 adopted, 97 kept
 *   - i4 n8960 k1536 (case i4g32-n8960-k1536): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.928 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.926))); c32u2: ×1.244 [1], ×1.254 [2]; break-even ×1.243 [1], ×1.251 [2]
 *   - i4 n2048 k1536 (cases i4g32-n2048-k1536, i4g512-n2048-k1536): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n2048-k1536 c32u2; i4g512-n2048-k1536 c32u4))
 *   - i4 n256 k1536 (cases i4g32-n256-k1536, i4g512-n256-k1536): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n256-k1536 c16u2; i4g512-n256-k1536 c32u4))
 *   - i4 n1536 k2048 (cases i4g32-n1536-k2048, i4g2048-n1536-k2048): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n1536-k2048 c32u2; i4g2048-n1536-k2048 c32u4))
 *   - i4 n6144 k1536 (cases i4g32-n6144-k1536, i4g512-n6144-k1536): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n6144-k1536 c64u2; i4g512-n6144-k1536 c32u4))
 *   - i4 n1536 k6144 (cases i4g32-n1536-k6144, i4g2048-n1536-k6144): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n1536-k6144 c16u2; i4g2048-n1536-k6144 c32u4))
 *   - i4 n1536 k256 (case i4g32-n1536-k256): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.923 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.107 (break-even ×0.888))); c32u2: ×1.404 [1], ×1.367 [2]; break-even ×1.344 [1], ×1.335 [2]
 *   - i4 n4096 k1536 (cases i4g32-n4096-k1536, i4g512-n4096-k1536): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n4096-k1536 c32u2; i4g512-n4096-k1536 c32u4))
 *   - i4 n512 k1536 (cases i4g32-n512-k1536, i4g512-n512-k1536): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n512-k1536 c16u2; i4g512-n512-k1536 c32u4))
 *   - i4 n1536 k4096 (cases i4g32-n1536-k4096, i4g4096-n1536-k4096): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n1536-k4096 c128u2; i4g4096-n1536-k4096 c32u4))
 *   - i4 n12288 k1536 (case i4g32-n12288-k1536): one steady record: c32u4 -> c32u2 (×1.134 [1], ×1.123 [2]; break-even ×1.103 [1], ×1.090 [2])
 *   - i4 n1536 k12288 (case i4g32-n1536-k12288): one steady record: keeps c64u2 (the default's: c32u4) (no challenger beats the table's choice c64u2 (nearest c16u2: ×0.975 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.078 (break-even ×0.964))); c64u2: ×1.622 [1], ×1.638 [2]; break-even ×1.594 [1], ×1.622 [2]
 *   - i8 n262144 k1536 (case i8-n262144-k1536): one steady record: keeps c32u4 (no candidate qualifies (nearest c16u16: its words differ from the baseline's in [1] at M=1)); c16u16: ×1.203 [1], ×1.223 [2]; break-even ×1.195 [1], ×1.219 [2]
 *   - i4 n262144 k1536 (case i4g32-n262144-k1536): one steady record: keeps c16u8 (the default's: c32u4) (no challenger beats the table's choice c16u8 (nearest c32u8: ×0.931 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.929))); c16u8: ×1.232 [1], ×1.237 [2]; break-even ×1.230 [1], ×1.235 [2]
 *   - i4 n256 k3072 (case i4g32-n256-k3072): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c16u8: ×0.874 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.874))); c16u2: ×1.740 [1], ×1.754 [2]; break-even ×1.738 [1], ×1.751 [2]
 *   - i4 n1024 k256 (case i4g32-n1024-k256): one steady record: keeps c64u2 (the default's: c32u4) (no challenger beats the table's choice c64u2 (nearest c16u2: ×0.936 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.086 (break-even ×0.918))); c64u2: ×1.476 [1], ×1.471 [2]; break-even ×1.437 [1], ×1.380 [2]
 *   - i4 n256 k1024 (case i4g32-n256-k1024): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c16u8: ×0.868 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.087 (break-even ×0.850))); c16u2: ×1.704 [1], ×1.702 [2]; break-even ×1.673 [1], ×1.699 [2]
 *   - i4 n2048 k256 (case i4g32-n2048-k256): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.842 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.083 (break-even ×0.828))); c32u2: ×1.328 [1], ×1.363 [2]; break-even ×1.269 [1], ×1.340 [2]
 *   - i4 n256 k2048 (case i4g32-n256-k2048): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c16u8: ×0.841 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.840))); c16u2: ×1.743 [1], ×1.742 [2]; break-even ×1.742 [1], ×1.737 [2]
 *   - i4 n262144 k256 (case i4g32-n262144-k256): one steady record: keeps c32u4 (no candidate qualifies (nearest c16u8: ×1.052 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.072 (break-even ×1.045))); c16u8: ×1.052 [1], ×1.068 [2]; break-even ×1.045 [1], ×1.067 [2]
 *   - i8 n256 k3072 (case i8-n256-k3072): one steady record: keeps c16u4 (the default's: c32u4) (no challenger beats the table's choice c16u4 (nearest c32u4: ×0.867 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.866))); c16u4: ×1.154 [1], ×1.111 [2]; break-even ×1.153 [1], ×1.108 [2]
 *   - i8 n1024 k256 (case i8-n1024-k256): one steady record: keeps c32u4 (no candidate qualifies (nearest c64u8: its words differ from the baseline's in [1] at M=1)); c64u8: ×1.115 [1], ×1.114 [2]; break-even ×1.060 [1], ×1.095 [2]
 *   - i8 n256 k1024 (case i8-n256-k1024): one steady record: keeps c16u4 (the default's: c32u4) (no challenger beats the table's choice c16u4 (nearest c32u4: ×0.855 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.078 (break-even ×0.844))); c16u4: ×1.170 [1], ×1.137 [2]; break-even ×1.155 [1], ×1.127 [2]
 *   - i8 n2048 k256 (case i8-n2048-k256): one steady record: keeps c32u4 (no candidate qualifies (nearest c32u8: its words differ from the baseline's in [1] at M=1)); c32u8: ×1.059 [1], ×1.031 [2]; break-even ×1.048 [1], ×1.029 [2]
 *   - i8 n256 k2048 (case i8-n256-k2048): one steady record: keeps c16u4 (the default's: c32u4) (no challenger beats the table's choice c16u4 (nearest c32u4: ×0.878 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.875))); c16u4: ×1.139 [1], ×1.130 [2]; break-even ×1.136 [1], ×1.128 [2]
 *   - i8 n262144 k256 (case i8-n262144-k256): one steady record: keeps c32u4 (no candidate qualifies (nearest c16u16: its words differ from the baseline's in [1] at M=1)); c16u16: ×1.240 [1], ×1.230 [2]; break-even ×1.228 [1], ×1.223 [2]
 *   - i8 n256 k1536 (case i8-n256-k1536): one steady record: keeps c16u4 (the default's: c32u4) (no challenger beats the table's choice c16u4 (nearest c32u4: ×0.896 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.072 (break-even ×0.890))); c16u4: ×1.116 [1], ×1.117 [2]; break-even ×1.108 [1], ×1.115 [2]
 *   - i8 n1536 k256 (case i8-n1536-k256): one steady record: keeps c32u4 (no candidate qualifies (nearest c16u8: its words differ from the baseline's in [1] at M=1)); c16u8: ×1.045 [1], ×1.071 [2]; break-even ×1.023 [1], ×1.039 [2]
 *   - i2 n12288 k1536 (case i2-n12288-k1536): one steady record: keeps c32u4 (no candidate qualifies (nearest c32u1: its words differ from the baseline's in [1] at M=1)); c32u1: ×1.032 [1], ×1.034 [2]; break-even ×1.031 [1], ×1.033 [2]
 *   - i2 n1536 k12288 (case i2-n1536-k12288): one steady record: keeps c32u4 (no candidate qualifies (nearest c64u4: ×0.999 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.999))); c64u4: ×0.999 [1], ×0.997 [2]; break-even ×0.999 [1], ×0.997 [2]
 *   - i2 n262144 k1536 (case i2-n262144-k1536): one steady record: keeps c32u4 (no candidate qualifies (nearest c128u1: its words differ from the baseline's in [1] at M=1)); c128u1: ×1.027 [1], ×1.012 [2]; break-even ×1.004 [1], ×0.990 [2]
 *   - i4 n10752 k2560 (case i4g32-n10752-k2560): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c16u8: ×0.955 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.071 (break-even ×0.949))); c16u2: ×1.179 [1], ×1.180 [2]; break-even ×1.166 [1], ×1.161 [2]
 *   - i4 n2048 k2560 (cases i4g32-n2048-k2560, i4g512-n2048-k2560): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n2048-k2560 c16u2; i4g512-n2048-k2560 c32u4))
 *   - i4 n512 k2560 (cases i4g32-n512-k2560, i4g512-n512-k2560): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n512-k2560 c16u2; i4g512-n512-k2560 c32u4))
 *   - i4 n256 k2560 (case i4g32-n256-k2560): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c16u8: ×0.853 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.853))); c16u2: ×1.740 [1], ×1.747 [2]; break-even ×1.740 [1], ×1.746 [2]
 *   - i4 n2560 k2048 (cases i4g32-n2560-k2048, i4g2048-n2560-k2048): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n2560-k2048 c16u2; i4g2048-n2560-k2048 c32u4))
 *   - i4 n10240 k2560 (cases i4g32-n10240-k2560, i4g512-n10240-k2560): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n10240-k2560 c32u8; i4g512-n10240-k2560 c32u4))
 *   - i4 n2560 k10240 (cases i4g32-n2560-k10240, i4g2048-n2560-k10240): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n2560-k10240 c256u2; i4g2048-n2560-k10240 c32u4))
 *   - i4 n2560 k256 (case i4g32-n2560-k256): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c32u8: ×0.927 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.120 (break-even ×0.881))); c32u2: ×1.331 [1], ×1.403 [2]; break-even ×1.245 [1], ×1.334 [2]
 *   - i4 n4096 k2560 (cases i4g32-n4096-k2560, i4g512-n4096-k2560): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n4096-k2560 c64u2; i4g512-n4096-k2560 c32u4))
 *   - i4 n1024 k2560 (cases i4g32-n1024-k2560, i4g512-n1024-k2560): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n1024-k2560 c32u2; i4g512-n1024-k2560 c32u4))
 *   - i4 n2560 k4096 (cases i4g32-n2560-k4096, i4g4096-n2560-k4096): one steady record: keeps c32u4 (the cases that share the key disagree, so the table's choice stays (i4g32-n2560-k4096 c16u2; i4g4096-n2560-k4096 c32u4))
 *   - i8 n262144 k2560 (case i8-n262144-k2560): one steady record: keeps c32u4 (no candidate qualifies (nearest c16u16: its words differ from the baseline's in [1] at M=1)); c16u16: ×1.200 [1], ×1.222 [2]; break-even ×1.182 [1], ×1.209 [2]
 *   - i4 n262144 k2560 (case i4g32-n262144-k2560): one steady record: keeps c128u8 (the default's: c32u4) (no challenger beats the table's choice c128u8 (nearest c16u2: ×0.905 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×0.902))); c32u8: ×1.245 [1], ×1.288 [2]; break-even ×1.230 [1], ×1.235 [2]
 *   - i4 n256 k5120 (case i4g32-n256-k5120): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c16u8: ×0.874 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.874))); c16u2: ×1.740 [1], ×1.743 [2]; break-even ×1.739 [1], ×1.743 [2]
 *   - i8 n256 k2560 (case i8-n256-k2560): one steady record: c32u4 -> c16u4 (×1.099 [1], ×1.087 [2]; break-even ×1.099 [1], ×1.084 [2])
 *   - i8 n2560 k256 (case i8-n2560-k256): one steady record: keeps c32u4 (no candidate qualifies (nearest c64u8: its words differ from the baseline's in [1] at M=1)); c64u8: ×1.036 [1], ×1.029 [2]; break-even ×0.972 [1], ×1.018 [2]
 *   - i2 n262144 k2560 (case i2-n262144-k2560): one steady record: keeps c32u4 (no candidate qualifies (nearest c64u1: its words differ from the baseline's in [1] at M=1)); c64u1: ×1.010 [1], ×1.012 [2]; break-even ×0.986 [1], ×0.993 [2]
 *   - i4 n15360 k3840 (case i4g32-n15360-k3840): one steady record: keeps c64u2 (the default's: c32u4) (no challenger beats the table's choice c64u2 (nearest c16u2: ×0.971 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.070 (break-even ×0.966))); c64u2: ×1.178 [1], ×1.177 [2]; break-even ×1.162 [1], ×1.172 [2]
 *   - i4 n8192 k3840 (case i4g32-n8192-k3840): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.953 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×0.950))); c32u2: ×1.204 [1], ×1.192 [2]; break-even ×1.182 [1], ×1.182 [2]
 *   - i4 n4096 k3840 (case i4g32-n4096-k3840): one steady record: keeps c256u2 (the default's: c32u4) (no challenger beats the table's choice c256u2 (nearest c32u2: ×0.993 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.091 (break-even ×0.970))); c256u2: ×1.352 [1], ×1.378 [2]; break-even ×1.238 [1], ×1.352 [2]
 *   - i4 n2048 k3840 (case i4g32-n2048-k3840): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c32u2: ×0.969 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.968))); c16u2: ×1.656 [1], ×1.652 [2]; break-even ×1.655 [1], ×1.651 [2]
 *   - i4 n512 k3840 (case i4g32-n512-k3840): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c16u8: ×0.866 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T18-03-03.json, under its bar ×1.065 (break-even ×0.866))); c16u2: ×1.736 [1], ×1.733 [2]; break-even ×1.734 [1], ×1.732 [2]
 *   - i4 n3840 k4096 (case i4g32-n3840-k4096): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.873 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.079 (break-even ×0.861))); c32u2: ×1.416 [1], ×1.419 [2]; break-even ×1.406 [1], ×1.410 [2]
 *   - i4 n3840 k8192 (case i4g32-n3840-k8192): one steady record: keeps c256u2 (the default's: c32u4) (no challenger beats the table's choice c256u2 (nearest c16u2: ×0.927 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.071 (break-even ×0.921))); c256u2: ×1.322 [1], ×1.325 [2]; break-even ×1.317 [1], ×1.322 [2]
 *   - i4 n3840 k15360 (case i4g32-n3840-k15360): one steady record: keeps c256u2 (the default's: c32u4) (no challenger beats the table's choice c256u2 (nearest c16u2: ×0.831 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.829))); c256u2: ×1.294 [1], ×1.295 [2]; break-even ×1.290 [1], ×1.289 [2]
 *   - i4 n262144 k3840 (case i4g32-n262144-k3840): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c32u2: ×0.977 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×0.972))); c16u2: ×1.257 [1], ×1.254 [2]; break-even ×1.255 [1], ×1.254 [2]
 *   - i4 n1024 k7680 (case i4g32-n1024-k7680): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.977 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.977))); c32u2: ×1.772 [1], ×1.816 [2]; break-even ×1.751 [1], ×1.815 [2]
 *   - i4 n4096 k1024 (case i4g32-n4096-k1024): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.966 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.965))); c32u2: ×1.257 [1], ×1.260 [2]; break-even ×1.255 [1], ×1.259 [2]
 *   - i4 n1024 k4096 (case i4g32-n1024-k4096): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.982 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.980))); c32u2: ×1.763 [1], ×1.782 [2]; break-even ×1.732 [1], ×1.778 [2]
 *   - i4 n8192 k1024 (case i4g32-n8192-k1024): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.919 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.917))); c32u2: ×1.232 [1], ×1.237 [2]; break-even ×1.228 [1], ×1.232 [2]
 *   - i4 n1024 k8192 (case i4g32-n1024-k8192): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.987 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.987))); c64u2: ×1.745 [1], ×1.775 [2]; break-even ×1.744 [1], ×1.770 [2]
 *   - i4 n3840 k1024 (case i4g32-n3840-k1024): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.962 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×0.959))); c32u2: ×1.307 [1], ×1.302 [2]; break-even ×1.303 [1], ×1.302 [2]
 *   - i4 n262144 k1024 (case i4g32-n262144-k1024): one steady record: keeps c16u8 (the default's: c32u4) (no challenger beats the table's choice c16u8 (nearest c32u8: ×0.917 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.915))); c16u8: ×1.230 [1], ×1.232 [2]; break-even ×1.228 [1], ×1.232 [2]
 *   - i4 n8192 k2816 (case i4g32-n8192-k2816): one steady record: keeps c32u8 (the default's: c32u4) (no challenger beats the table's choice c32u8 (nearest c16u2: ×0.964 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.103 (break-even ×0.931))); c32u2: ×1.169 [1], ×1.176 [2]; break-even ×1.152 [1], ×1.137 [2]
 *   - i4 n4096 k2816 (case i4g32-n4096-k2816): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.864 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.863))); c32u2: ×1.346 [1], ×1.353 [2]; break-even ×1.345 [1], ×1.352 [2]
 *   - i4 n2048 k2816 (case i4g32-n2048-k2816): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c32u2: ×0.973 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.972))); c16u2: ×1.619 [1], ×1.613 [2]; break-even ×1.619 [1], ×1.611 [2]
 *   - i4 n2112 k2816 (case i4g32-n2112-k2816): one steady record: keeps c256u2 (the default's: c32u4) (no challenger beats the table's choice c256u2 (nearest c16u2: ×0.989 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.987))); c256u2: ×1.633 [1], ×1.637 [2]; break-even ×1.632 [1], ×1.635 [2]
 *   - i4 n1024 k2816 (case i4g32-n1024-k2816): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.966 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.966))); c32u2: ×1.688 [1], ×1.674 [2]; break-even ×1.688 [1], ×1.671 [2]
 *   - i4 n128 k2816 (case i4g32-n128-k2816): one steady record: keeps c16u2 (the default's: c32u4) (no challenger beats the table's choice c16u2 (nearest c16u8: ×0.865 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.864))); c16u2: ×1.743 [1], ×1.744 [2]; break-even ×1.742 [1], ×1.744 [2]
 *   - i4 n2816 k4096 (case i4g32-n2816-k4096): one steady record: keeps c64u2 (the default's: c32u4) (no challenger beats the table's choice c64u2 (nearest c16u2: ×0.927 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.925))); c64u2: ×1.442 [1], ×1.441 [2]; break-even ×1.441 [1], ×1.437 [2]
 *   - i4 n2816 k8192 (case i4g32-n2816-k8192): one steady record: keeps c32u2 (the default's: c32u4) (no challenger beats the table's choice c32u2 (nearest c16u2: ×0.867 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.143 (break-even ×0.807))); c32u2: ×1.338 [1], ×1.345 [2]; break-even ×1.265 [1], ×1.334 [2]
 *   - i4 n2816 k2112 (case i4g32-n2816-k2112): one steady record: keeps c32u4 (no candidate qualifies (nearest c16u2: its words differ from the baseline's in [1] at M=1)); c16u2: ×1.635 [1], ×1.631 [2]; break-even ×1.634 [1], ×1.631 [2]
 *   - i4 n262144 k2816 (case i4g32-n262144-k2816): one steady record: keeps c32u8 (the default's: c32u4) (no challenger beats the table's choice c32u8 (nearest c16u2: ×0.950 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.085 (break-even ×0.933))); c32u8: ×1.234 [1], ×1.259 [2]; break-even ×1.140 [1], ×1.257 [2]
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): one steady record: keeps c16u8 (no candidate qualifies (nearest c16u16: ×0.998 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×0.994))); c16u16: ×0.998 [1], ×0.997 [2]; break-even ×0.994 [1], ×0.991 [2]
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): one steady record: keeps c16u16 (no candidate qualifies (nearest c16u4: ×1.008 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.008))); c16u4: ×1.008 [1], ×1.003 [2]; break-even ×1.008 [1], ×1.003 [2]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): one steady record: keeps c16u16 (no candidate qualifies (nearest c16u8: ×0.907 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.073 (break-even ×0.901))); c16u8: ×0.907 [1], ×0.892 [2]; break-even ×0.901 [1], ×0.877 [2]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): one steady record: keeps c16u8 (no candidate qualifies (nearest c16u4: ×0.999 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.998))); c16u4: ×0.999 [1], ×1.001 [2]; break-even ×0.998 [1], ×1.001 [2]
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): one steady record: keeps c16u8 (no candidate qualifies (nearest c16u16: ×0.994 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.993))); c16u16: ×0.994 [1], ×0.997 [2]; break-even ×0.993 [1], ×0.993 [2]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): one steady record: keeps c16u16 (no candidate qualifies (nearest c16u8: ×1.037 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.035))); c16u8: ×1.037 [1], ×1.032 [2]; break-even ×1.035 [1], ×1.029 [2]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): one steady record: keeps c32u16 (no candidate qualifies (nearest c16u16: ×0.927 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.078 (break-even ×0.916))); c16u16: ×0.927 [1], ×0.943 [2]; break-even ×0.916 [1], ×0.898 [2]
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): one steady record: keeps c16u8 (no candidate qualifies (nearest c16u16: ×0.994 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.128 (break-even ×0.938))); c16u16: ×0.994 [1], ×0.996 [2]; break-even ×0.938 [1], ×0.986 [2]
 *   - bf16 n512 k1536 (case bf16-n512-k1536): one steady record: keeps c16u16 (no candidate qualifies (nearest c32u16: ×0.978 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.098 (break-even ×0.949))); c32u16: ×0.978 [1], ×0.979 [2]; break-even ×0.949 [1], ×0.957 [2]
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): one steady record: keeps c16u16 (no candidate qualifies (nearest c64u16: ×1.006 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.126 (break-even ×0.952))); c64u16: ×1.006 [1], ×1.016 [2]; break-even ×0.952 [1], ×0.975 [2]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): one steady record: keeps c16u8 (no candidate qualifies (nearest c16u16: ×1.001 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.070 (break-even ×0.996))); c16u16: ×1.001 [1], ×0.999 [2]; break-even ×0.996 [1], ×0.996 [2]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): one steady record: keeps c16u16 (no candidate qualifies (nearest c16u8: ×1.031 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.030))); c16u8: ×1.031 [1], ×1.034 [2]; break-even ×1.030 [1], ×1.028 [2]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): one steady record: keeps c32u4 (no candidate qualifies (nearest c16u16: ×1.070 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.072 (break-even ×1.063))); c16u16: ×1.070 [1], ×1.070 [2]; break-even ×1.063 [1], ×1.067 [2]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): one steady record: keeps c16u8 (no candidate qualifies (nearest c16u16: ×1.003 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.001))); c16u16: ×1.003 [1], ×0.999 [2]; break-even ×1.001 [1], ×0.996 [2]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): one steady record: keeps c16u8 (no candidate qualifies (nearest c128u8: ×1.007 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.076 (break-even ×0.997))); c128u8: ×1.007 [1], ×1.011 [2]; break-even ×0.997 [1], ×0.991 [2]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): one steady record: keeps c16u16 (no candidate qualifies (nearest c16u8: ×1.017 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.015))); c16u8: ×1.017 [1], ×1.012 [2]; break-even ×1.015 [1], ×1.010 [2]
 *   - bf16 n256 k2560 (case bf16-n256-k2560): one steady record: keeps c16u16 (no candidate qualifies (nearest c16u8: ×0.884 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.071 (break-even ×0.878))); c16u8: ×0.884 [1], ×0.892 [2]; break-even ×0.878 [1], ×0.890 [2]
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): one steady record: keeps c16u8 (no candidate qualifies (nearest c16u16: ×1.016 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.073 (break-even ×1.009))); c16u16: ×1.016 [1], ×1.015 [2]; break-even ×1.009 [1], ×1.005 [2]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): one steady record: keeps c32u4 (no candidate qualifies (nearest c16u16: ×1.062 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T18-03-03.json, under its bar ×1.071 (break-even ×1.056))); c16u16: ×1.074 [1], ×1.062 [2]; break-even ×1.068 [1], ×1.056 [2]
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): one steady record: keeps c16u8 (no candidate qualifies (nearest c256u8: ×1.011 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.085 (break-even ×0.993))); c256u8: ×1.011 [1], ×1.012 [2]; break-even ×0.993 [1], ×1.010 [2]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): one steady record: keeps c16u16 (no candidate qualifies (nearest c32u16: ×1.047 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.118 (break-even ×0.997))); c32u16: ×1.047 [1], ×1.026 [2]; break-even ×0.997 [1], ×0.979 [2]
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): one steady record: keeps c16u8 (no candidate qualifies (nearest c16u16: ×0.996 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.995))); c16u16: ×0.996 [1], ×0.998 [2]; break-even ×0.995 [1], ×0.994 [2]
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): one steady record: keeps c16u16 (no candidate qualifies (nearest c64u16: ×1.002 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.001))); c64u16: ×1.002 [1], ×1.003 [2]; break-even ×1.001 [1], ×1.002 [2]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): one steady record: keeps c16u8 (no candidate qualifies (nearest c64u8: ×1.007 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.072 (break-even ×1.000))); c64u8: ×1.007 [1], ×1.004 [2]; break-even ×1.000 [1], ×0.996 [2]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): one steady record: keeps c32u4 (no candidate qualifies (nearest c16u16: ×1.069 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.075 (break-even ×1.059))); c16u16: ×1.069 [1], ×1.070 [2]; break-even ×1.059 [1], ×1.065 [2]
 * - reorder class (parallelShapes, M = 1 to 8): 0 adopted, 115 kept
 *   - i4 g32 n8960 k1536 (case i4g32-n8960-k1536): one steady record: keeps parallel-l4 with row blocks of 4 (no candidate qualifies (nearest parallel-l16: ×1.016 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.016))); parallel-l16: ×1.016 [1], ×1.015 [2]; break-even ×1.016 [1], ×1.014 [2]
 *   - i4 g32 n2048 k1536 (case i4g32-n2048-k1536): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4 with row blocks of 2) (no candidate qualifies (nearest parallel-l8: ×1.005 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×1.001))); parallel-l8: ×1.005 [1], ×1.025 [2]; break-even ×1.001 [1], ×1.023 [2]
 *   - i4 g32 n256 k1536 (case i4g32-n256-k1536): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l32 with row blocks of 2) (no challenger beats the table's choice parallel-l16 (nearest parallel-l8: ×0.890 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.076 (break-even ×0.881))); parallel-l16: ×1.243 [1], ×1.193 [2]; break-even ×1.220 [1], ×1.161 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.405 at M=1): a Session option, not a row
 *   - i4 g32 n1536 k2048 (case i4g32-n1536-k2048): one steady record: keeps parallel-l8 with row blocks of 4 (the default's: staged-l32) (no challenger beats the table's choice parallel-l8 (nearest parallel-l2: ×0.862 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T18-03-03.json, under its bar ×1.071 (break-even ×0.858))); parallel-l8: ×1.510 [1], ×1.531 [2]; break-even ×1.508 [1], ×1.523 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.601 at M=1): a Session option, not a row
 *   - i4 g32 n6144 k1536 (case i4g32-n6144-k1536): one steady record: keeps parallel-l4 with row blocks of 4 (no candidate qualifies (nearest parallel-l16: ×1.025 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.025))); parallel-l16: ×1.025 [1], ×1.024 [2]; break-even ×1.025 [1], ×1.014 [2]
 *   - i4 g32 n1536 k6144 (case i4g32-n1536-k6144): one steady record: keeps staged-l32 with row blocks of 2 (no candidate qualifies (nearest parallel-l4: ×1.065 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.064))); parallel-l4: ×1.065 [1], ×1.060 [2]; break-even ×1.064 [1], ×1.060 [2]
 *   - i4 g32 n1536 k256 (case i4g32-n1536-k256): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4) (no candidate qualifies (nearest staged-l4: ×1.079 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.088 (break-even ×1.056))); staged-l4: ×1.079 [1], ×1.073 [2]; break-even ×1.056 [1], ×1.034 [2]
 *     - subgroup32-l4 meets the rule too (qualifies by ×1.122 at M=1): a Session option, not a row
 *   - i4 g32 n4096 k1536 (case i4g32-n4096-k1536): one steady record: keeps parallel-l4 with row blocks of 4 (no candidate qualifies (nearest parallel-l8: ×1.017 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.016))); parallel-l8: ×1.017 [1], ×1.014 [2]; break-even ×1.016 [1], ×1.013 [2]
 *   - i4 g32 n512 k1536 (case i4g32-n512-k1536): one steady record: keeps parallel-l8 with row blocks of 4 (the default's: parallel-l32) (no challenger beats the table's choice parallel-l8 (nearest parallel-l4: ×0.913 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.107 (break-even ×0.878))); parallel-l8: ×1.241 [1], ×1.275 [2]; break-even ×1.203 [1], ×1.264 [2]
 *     - subgroup32-l8 meets the rule too (qualifies by ×1.402 at M=1): a Session option, not a row
 *   - i4 g32 n1536 k4096 (case i4g32-n1536-k4096): one steady record: keeps staged-l32 with row blocks of 2 (the default's: staged-l32) (no candidate qualifies (nearest parallel-l4: ×1.066 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×1.062))); parallel-l4: ×1.066 [1], ×1.064 [2]; break-even ×1.062 [1], ×1.064 [2]
 *   - i4 g32 n12288 k1536 (case i4g32-n12288-k1536): one steady record: keeps parallel-l4 with row blocks of 4 (no candidate qualifies (nearest parallel-l16: parallel-l16 has no time at M=4 in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, which timed it at M=1 (a parallel row serves M = 1 to 8))); parallel-l16: ×1.210 [1], ×1.193 [2]; break-even ×1.199 [1], ×1.159 [2]
 *   - i4 g32 n1536 k12288 (case i4g32-n1536-k12288): one steady record: keeps staged-l32 with row blocks of 2 (no candidate qualifies (nearest parallel-l32: ×1.052 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.072 (break-even ×1.045))); parallel-l32: ×1.052 [1], ×1.058 [2]; break-even ×1.045 [1], ×1.057 [2]
 *   - i8 n262144 k1536 (case i8-n262144-k1536): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l16) (no candidate qualifies (nearest staged-l16: ×1.003 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.002))); staged-l16: ×1.003 [1], ×1.003 [2]; break-even ×1.002 [1], ×1.001 [2]
 *   - i4 g32 n262144 k1536 (case i4g32-n262144-k1536): one steady record: keeps no row (c16u8) (the default's: no row (c32u4)) (no challenger beats the table's choice c16u8 (nearest c32u4: ×0.812 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.811))); staged-l16: ×1.308 [1], ×1.315 [2]; break-even ×1.305 [1], ×1.314 [2]
 *   - i4 g32 n256 k3072 (case i4g32-n256-k3072): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l32) (no challenger beats the table's choice parallel-l16 (nearest parallel-l32: ×0.888 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.113 (break-even ×0.850))); parallel-l16: ×1.126 [1], ×1.112 [2]; break-even ×1.077 [1], ×1.103 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.218 at M=1): a Session option, not a row
 *   - i4 g32 n1024 k256 (case i4g32-n1024-k256): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: no row (c32u4)) (no challenger beats the table's choice parallel-l4 (nearest staged-l4: ×0.982 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.142 (break-even ×0.915))); parallel-l4: ×1.768 [1], ×1.823 [2]; break-even ×1.646 [1], ×1.700 [2]
 *     - subgroup32-l4 meets the rule too (qualifies by ×2.008 at M=1): a Session option, not a row
 *   - i4 g32 n256 k1024 (case i4g32-n256-k1024): one steady record: keeps parallel-l32 (no candidate qualifies (nearest parallel-l16: ×1.098 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.110 (break-even ×1.054))); parallel-l16: ×1.098 [1], ×1.155 [2]; break-even ×1.054 [1], ×1.154 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.380 at M=1): a Session option, not a row
 *   - i4 g32 n2048 k256 (case i4g32-n2048-k256): one steady record: keeps staged-l4 with row blocks of 4 (the default's: no row (c32u4)) (no challenger beats the table's choice staged-l4 (nearest parallel-l2: ×0.979 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.092 (break-even ×0.954))); staged-l4: ×1.403 [1], ×1.438 [2]; break-even ×1.349 [1], ×1.417 [2]
 *     - subgroup32-l4 meets the rule too (qualifies by ×1.530 at M=1): a Session option, not a row
 *   - i4 g32 n256 k2048 (case i4g32-n256-k2048): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: no row (c32u4)) (no challenger beats the table's choice parallel-l16 (nearest parallel-l8: ×0.903 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.077 (break-even ×0.893))); parallel-l16: ×7.269 [1], ×7.322 [2]; break-even ×7.210 [1], ×7.177 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×8.071 at M=1): a Session option, not a row
 *   - i4 g32 n262144 k256 (case i4g32-n262144-k256): one steady record: keeps no row (c32u4) (no candidate qualifies (nearest staged-l8: staged-l8 has no time at M=4 in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, which timed it at M=1 (a parallel row serves M = 1 to 8))); staged-l8: ×1.079 [1], ×1.095 [2]; break-even ×1.072 [1], ×1.092 [2]
 *   - i8 n256 k3072 (case i8-n256-k3072): one steady record: keeps parallel-l16 with row blocks of 2 (the default's: no row (c32u4)) (no challenger beats the table's choice parallel-l16 (nearest parallel-l32: ×0.999 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.086 (break-even ×0.980))); parallel-l16: ×4.765 [1], ×4.675 [2]; break-even ×4.737 [1], ×4.661 [2]
 *     - subgroup32-l32 meets the rule too (qualifies by ×5.269 at M=1): a Session option, not a row
 *   - i8 n1024 k256 (case i8-n1024-k256): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: no row (c32u4)) (no challenger beats the table's choice parallel-l4 (nearest parallel-l2: ×0.898 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.091 (break-even ×0.877))); parallel-l4: ×1.258 [1], ×1.298 [2]; break-even ×1.224 [1], ×1.274 [2]
 *     - subgroup32-l4 meets the rule too (qualifies by ×1.439 at M=1): a Session option, not a row
 *   - i8 n256 k1024 (case i8-n256-k1024): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: no row (c32u4)) (no challenger beats the table's choice parallel-l16 (nearest parallel-l8: ×0.880 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.080 (break-even ×0.868))); parallel-l16: ×3.205 [1], ×3.021 [2]; break-even ×3.128 [1], ×2.981 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×3.669 at M=1): a Session option, not a row
 *   - i8 n2048 k256 (case i8-n2048-k256): one steady record: keeps staged-l4 with row blocks of 4 (the default's: no row (c32u4)) (no challenger beats the table's choice staged-l4 (nearest parallel-l4: ×1.018 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.141 (break-even ×0.951))); staged-l4: ×1.119 [1], ×1.135 [2]; break-even ×1.089 [1], ×1.125 [2]
 *   - i8 n256 k2048 (case i8-n256-k2048): one steady record: keeps staged-l32 with row blocks of 4 (the default's: no row (c32u4)) (no challenger beats the table's choice staged-l32 (nearest parallel-l16: ×1.069 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.080 (break-even ×1.055))); staged-l32: ×4.087 [1], ×4.053 [2]; break-even ×4.071 [1], ×4.005 [2]
 *     - subgroup32-l32 meets the rule too (qualifies by ×4.587 at M=1): a Session option, not a row
 *   - i8 n262144 k256 (case i8-n262144-k256): one steady record: keeps no row (c32u4) (no candidate qualifies (nearest parallel-l8: parallel-l8 has no time at M=4 in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, which timed it at M=1 (a parallel row serves M = 1 to 8))); parallel-l8: ×1.307 [1], ×1.303 [2]; break-even ×1.305 [1], ×1.300 [2]
 *   - i4 g512 n2048 k1536 (case i4g512-n2048-k1536): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4) (no candidate qualifies (nearest parallel-l8: ×1.017 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.073 (break-even ×1.010))); parallel-l8: ×1.017 [1], ×1.010 [2]; break-even ×1.010 [1], ×1.006 [2]
 *   - i4 g512 n256 k1536 (case i4g512-n256-k1536): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l32) (no challenger beats the table's choice parallel-l16 (nearest parallel-l32: ×0.860 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.091 (break-even ×0.840))); parallel-l16: ×1.162 [1], ×1.109 [2]; break-even ×1.134 [1], ×1.050 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.364 at M=1): a Session option, not a row
 *   - i4 g2048 n1536 k2048 (case i4g2048-n1536-k2048): one steady record: keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l8 meets the rule too (qualifies by ×1.326 at M=1): a Session option, not a row
 *   - i4 g512 n6144 k1536 (case i4g512-n6144-k1536): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4) (no candidate qualifies (nearest parallel-l8: ×1.021 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.020))); parallel-l8: ×1.021 [1], ×1.025 [2]; break-even ×1.020 [1], ×1.024 [2]
 *   - i4 g2048 n1536 k6144 (case i4g2048-n1536-k6144): one steady record: keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k1536 (case i8-n256-k1536): one steady record: keeps parallel-l32 with row blocks of 4 (the default's: parallel-l32) (no candidate qualifies (nearest parallel-l16: ×1.026 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.105 (break-even ×0.989))); parallel-l16: ×1.026 [1], ×0.984 [2]; break-even ×0.989 [1], ×0.956 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.146 at M=1): a Session option, not a row
 *   - i8 n1536 k256 (case i8-n1536-k256): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4) (no candidate qualifies (nearest staged-l4: ×0.970 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.104 (break-even ×0.936))); staged-l4: ×0.970 [1], ×0.977 [2]; break-even ×0.936 [1], ×0.943 [2]
 *   - i4 g512 n4096 k1536 (case i4g512-n4096-k1536): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4) (no candidate qualifies (nearest parallel-l16: ×1.020 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.019))); parallel-l16: ×1.020 [1], ×1.011 [2]; break-even ×1.019 [1], ×1.010 [2]
 *   - i4 g512 n512 k1536 (case i4g512-n512-k1536): one steady record: keeps staged-l16 with row blocks of 4 (the default's: parallel-l32) (no challenger beats the table's choice staged-l16 (nearest parallel-l4: ×0.891 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.071 (break-even ×0.885))); parallel-l8: ×1.248 [1], ×1.247 [2]; break-even ×1.229 [1], ×1.198 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.393 at M=1): a Session option, not a row
 *   - i4 g4096 n1536 k4096 (case i4g4096-n1536-k4096): one steady record: keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n12288 k1536 (case i2-n12288-k1536): one steady record: keeps parallel-l2 with row blocks of 4 (the default's: parallel-l2) (no candidate qualifies (nearest parallel-l4: ×1.018 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×1.015))); parallel-l4: ×1.018 [1], ×1.018 [2]; break-even ×1.015 [1], ×1.011 [2]
 *   - i2 n1536 k12288 (case i2-n1536-k12288): one steady record: keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k1536 (case i2-n262144-k1536): one steady record: keeps no row (c32u4) (no candidate qualifies (nearest parallel-l2: ×0.997 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.082 (break-even ×0.982))); parallel-l2: ×0.997 [1], ×1.008 [2]; break-even ×0.982 [1], ×0.974 [2]
 *   - i4 g32 n10752 k2560 (case i4g32-n10752-k2560): one steady record: keeps parallel-l4 with row blocks of 4 (no candidate qualifies (nearest parallel-l32: parallel-l32 has no time at M=4 in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, which timed it at M=1 (a parallel row serves M = 1 to 8))); parallel-l32: ×1.380 [1], ×1.412 [2]; break-even ×1.373 [1], ×1.407 [2]
 *   - i4 g32 n2048 k2560 (case i4g32-n2048-k2560): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4 with row blocks of 2) (no candidate qualifies (nearest parallel-l16: ×1.040 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.039))); parallel-l16: ×1.040 [1], ×1.037 [2]; break-even ×1.039 [1], ×1.036 [2]
 *   - i4 g32 n512 k2560 (case i4g32-n512-k2560): one steady record: keeps staged-l16 with row blocks of 4 (the default's: parallel-l32) (no challenger beats the table's choice staged-l16 (nearest parallel-l8: ×1.008 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.081 (break-even ×0.992))); parallel-l16: ×1.152 [1], ×1.170 [2]; break-even ×1.136 [1], ×1.152 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.307 at M=1): a Session option, not a row
 *   - i4 g32 n256 k2560 (case i4g32-n256-k2560): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l32) (no challenger beats the table's choice parallel-l16 (nearest parallel-l32: ×0.865 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.146 (break-even ×0.804))); parallel-l16: ×1.156 [1], ×1.206 [2]; break-even ×1.074 [1], ×1.192 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.314 at M=1): a Session option, not a row
 *   - i4 g32 n2560 k2048 (case i4g32-n2560-k2048): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: staged-l32) (no challenger beats the table's choice parallel-l4 (nearest parallel-l2: ×0.975 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.974))); parallel-l4: ×1.132 [1], ×1.138 [2]; break-even ×1.131 [1], ×1.138 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.176 at M=1): a Session option, not a row
 *   - i4 g32 n10240 k2560 (case i4g32-n10240-k2560): one steady record: keeps parallel-l4 with row blocks of 4 (no candidate qualifies (nearest parallel-l32: parallel-l32 has no time at M=4 in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, which timed it at M=1 (a parallel row serves M = 1 to 8))); parallel-l32: ×1.389 [1], ×1.387 [2]; break-even ×1.379 [1], ×1.386 [2]
 *   - i4 g32 n2560 k10240 (case i4g32-n2560-k10240): one steady record: keeps staged-l32 with row blocks of 2 (no candidate qualifies (nearest parallel-l32: ×1.031 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.092 (break-even ×1.006))); parallel-l32: ×1.031 [1], ×1.012 [2]; break-even ×1.006 [1], ×1.006 [2]
 *   - i4 g32 n2560 k256 (case i4g32-n2560-k256): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4 with row blocks of 2) (no candidate qualifies (nearest staged-l4: ×1.039 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.082 (break-even ×1.023))); staged-l4: ×1.039 [1], ×1.067 [2]; break-even ×1.023 [1], ×1.032 [2]
 *   - i4 g32 n4096 k2560 (case i4g32-n4096-k2560): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4 with row blocks of 2) (no candidate qualifies (nearest parallel-l32: ×1.022 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.022))); parallel-l32: ×1.022 [1], ×1.024 [2]; break-even ×1.022 [1], ×1.024 [2]
 *   - i4 g32 n1024 k2560 (case i4g32-n1024-k2560): one steady record: keeps parallel-l8 with row blocks of 4 (the default's: parallel-l32 with row blocks of 2) (no challenger beats the table's choice parallel-l8 (nearest parallel-l4: ×0.909 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.087 (break-even ×0.891))); parallel-l8: ×1.310 [1], ×1.311 [2]; break-even ×1.297 [1], ×1.299 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.409 at M=1): a Session option, not a row
 *   - i4 g32 n2560 k4096 (case i4g32-n2560-k4096): one steady record: keeps staged-l32 with row blocks of 2 (no candidate qualifies (nearest parallel-l32: ×1.032 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.031))); parallel-l32: ×1.032 [1], ×1.028 [2]; break-even ×1.031 [1], ×1.028 [2]
 *   - i8 n262144 k2560 (case i8-n262144-k2560): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l16) (no candidate qualifies (nearest staged-l16: ×1.002 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×0.999))); staged-l16: ×1.002 [1], ×1.005 [2]; break-even ×0.999 [1], ×1.004 [2]
 *   - i4 g32 n262144 k2560 (case i4g32-n262144-k2560): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: no row (c32u4)) (no challenger beats the table's choice parallel-l16 (nearest parallel-l8: ×0.982 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.072 (break-even ×0.976))); parallel-l16: ×1.419 [1], ×1.484 [2]; break-even ×1.397 [1], ×1.432 [2]
 *   - i4 g32 n256 k5120 (case i4g32-n256-k5120): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: no row (c32u4)) (no challenger beats the table's choice parallel-l16 (nearest parallel-l8: ×0.883 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.881))); parallel-l16: ×10.209 [1], ×10.299 [2]; break-even ×10.205 [1], ×10.287 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×11.009 at M=1): a Session option, not a row
 *   - i4 g512 n2048 k2560 (case i4g512-n2048-k2560): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4) (no candidate qualifies (nearest parallel-l8: ×0.969 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.968))); parallel-l8: ×0.969 [1], ×0.970 [2]; break-even ×0.968 [1], ×0.968 [2]
 *   - i4 g512 n512 k2560 (case i4g512-n512-k2560): one steady record: keeps staged-l16 with row blocks of 4 (the default's: parallel-l32) (no challenger beats the table's choice staged-l16 (nearest parallel-l16: ×1.015 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.075 (break-even ×1.005))); staged-l16: ×1.112 [1], ×1.119 [2]; break-even ×1.098 [1], ×1.092 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.233 at M=1): a Session option, not a row
 *   - i4 g2048 n2560 k2048 (case i4g2048-n2560-k2048): one steady record: keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l4 meets the rule too (qualifies by ×1.128 at M=1): a Session option, not a row
 *   - i4 g512 n10240 k2560 (case i4g512-n10240-k2560): one steady record: keeps staged-l16 with row blocks of 4 (the default's: parallel-l4) (no challenger beats the table's choice staged-l16 (nearest parallel-l8: ×0.970 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.079 (break-even ×0.958))); staged-l16: ×1.332 [1], ×1.323 [2]; break-even ×1.317 [1], ×1.301 [2]
 *     - subgroup32-l8 meets the rule too (qualifies by ×1.306 at M=1): a Session option, not a row
 *   - i4 g2048 n2560 k10240 (case i4g2048-n2560-k10240): one steady record: keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k2560 (case i8-n256-k2560): one steady record: keeps parallel-l32 with row blocks of 4 (the default's: parallel-l32) (no candidate qualifies (nearest parallel-l16: ×1.025 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.073 (break-even ×1.017))); parallel-l16: ×1.025 [1], ×0.993 [2]; break-even ×1.017 [1], ×0.980 [2]
 *     - subgroup32-l32 meets the rule too (qualifies by ×1.100 at M=1): a Session option, not a row
 *   - i8 n2560 k256 (case i8-n2560-k256): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4) (no candidate qualifies (nearest parallel-l2: ×1.027 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.077 (break-even ×1.016))); parallel-l2: ×1.027 [1], ×1.002 [2]; break-even ×1.016 [1], ×0.987 [2]
 *     - subgroup32-l4 meets the rule too (qualifies by ×1.108 at M=1): a Session option, not a row
 *   - i4 g512 n4096 k2560 (case i4g512-n4096-k2560): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4) (no candidate qualifies (nearest parallel-l16: ×1.035 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.035))); parallel-l16: ×1.035 [1], ×1.036 [2]; break-even ×1.035 [1], ×1.035 [2]
 *   - i4 g512 n1024 k2560 (case i4g512-n1024-k2560): one steady record: keeps parallel-l8 with row blocks of 4 (the default's: parallel-l32) (no challenger beats the table's choice parallel-l8 (nearest parallel-l4: ×0.905 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.904))); parallel-l8: ×1.276 [1], ×1.276 [2]; break-even ×1.273 [1], ×1.265 [2]
 *     - subgroup32-l16 meets the rule too (qualifies by ×1.391 at M=1): a Session option, not a row
 *   - i4 g4096 n2560 k4096 (case i4g4096-n2560-k4096): one steady record: keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k2560 (case i2-n262144-k2560): one steady record: keeps no row (c32u4) (no candidate qualifies (nearest parallel-l2: ×1.003 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.099 (break-even ×0.972))); parallel-l2: ×1.003 [1], ×1.005 [2]; break-even ×0.972 [1], ×0.982 [2]
 *   - i4 g32 n15360 k3840 (case i4g32-n15360-k3840): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l4) (no challenger beats the table's choice parallel-l16 (nearest parallel-l8: ×0.980 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.980))); parallel-l16: ×1.273 [1], ×1.283 [2]; break-even ×1.271 [1], ×1.281 [2]
 *   - i4 g32 n8192 k3840 (case i4g32-n8192-k3840): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l4) (no challenger beats the table's choice parallel-l16 (nearest parallel-l8: ×0.984 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.981))); parallel-l16: ×1.282 [1], ×1.294 [2]; break-even ×1.274 [1], ×1.292 [2]
 *   - i4 g32 n4096 k3840 (case i4g32-n4096-k3840): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4 with row blocks of 2) (no candidate qualifies (nearest parallel-l32: ×1.062 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.060))); parallel-l32: ×1.062 [1], ×1.060 [2]; break-even ×1.060 [1], ×1.055 [2]
 *   - i4 g32 n2048 k3840 (case i4g32-n2048-k3840): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4 with row blocks of 2) (no candidate qualifies (nearest parallel-l32: parallel-l32 has no time at M=4 in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, which timed it at M=1 (a parallel row serves M = 1 to 8))); parallel-l32: ×1.075 [1], ×1.081 [2]; break-even ×1.074 [1], ×1.081 [2]
 *   - i4 g32 n512 k3840 (case i4g32-n512-k3840): one steady record: keeps parallel-l32 with row blocks of 2 (no candidate qualifies (nearest parallel-l8: ×1.060 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.077 (break-even ×1.049))); parallel-l8: ×1.060 [1], ×1.066 [2]; break-even ×1.049 [1], ×1.060 [2]
 *     - subgroup32-l32 meets the rule too (qualifies by ×1.122 at M=1): a Session option, not a row
 *   - i4 g32 n3840 k4096 (case i4g32-n3840-k4096): one steady record: keeps staged-l32 with row blocks of 2 (the default's: staged-l32) (no candidate qualifies (nearest parallel-l32: ×1.038 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.037))); parallel-l32: ×1.038 [1], ×1.038 [2]; break-even ×1.037 [1], ×1.034 [2]
 *   - i4 g32 n3840 k8192 (case i4g32-n3840-k8192): one steady record: keeps staged-l32 with row blocks of 2 (no candidate qualifies (nearest parallel-l32: ×1.002 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.000))); parallel-l32: ×1.002 [1], ×1.000 [2]; break-even ×1.000 [1], ×0.999 [2]
 *   - i4 g32 n3840 k15360 (case i4g32-n3840-k15360): one steady record: keeps staged-l32 with row blocks of 2 (the default's: staged-l32) (no candidate qualifies (nearest parallel-l32: ×1.036 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.034))); parallel-l32: ×1.036 [1], ×1.038 [2]; break-even ×1.034 [1], ×1.036 [2]
 *   - i4 g32 n262144 k3840 (case i4g32-n262144-k3840): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l16) (no candidate qualifies (nearest staged-l16: ×0.998 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.996))); staged-l16: ×0.998 [1], ×0.997 [2]; break-even ×0.996 [1], ×0.996 [2]
 *   - i4 g32 n1024 k7680 (case i4g32-n1024-k7680): one steady record: keeps staged-l32 with row blocks of 2 (the default's: staged-l32) (no candidate qualifies (nearest parallel-l32: parallel-l32 has no time at M=4 in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, which timed it at M=1 (a parallel row serves M = 1 to 8))); parallel-l32: ×1.083 [1], ×1.083 [2]; break-even ×1.083 [1], ×1.081 [2]
 *     - subgroup32-l32 meets the rule too (qualifies by ×1.118 at M=1): a Session option, not a row
 *   - i4 g32 n4096 k1024 (case i4g32-n4096-k1024): one steady record: keeps no row (c32u2) (the default's: no row (c32u4)) (no challenger beats the table's choice c32u2 (nearest c32u4: ×0.795 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.794))); parallel-l16: ×1.406 [1], ×1.409 [2]; break-even ×1.404 [1], ×1.407 [2]
 *   - i4 g32 n1024 k4096 (case i4g32-n1024-k4096): one steady record: keeps staged-l32 with row blocks of 2 (the default's: staged-l32) (no candidate qualifies (nearest parallel-l32: parallel-l32 has no time at M=4 in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, which timed it at M=1 (a parallel row serves M = 1 to 8))); parallel-l32: ×1.092 [1], ×1.084 [2]; break-even ×1.091 [1], ×1.084 [2]
 *     - subgroup32-l32 meets the rule too (qualifies by ×1.153 at M=1): a Session option, not a row
 *   - i4 g32 n8192 k1024 (case i4g32-n8192-k1024): one steady record: keeps no row (c32u2) (the default's: no row (c32u4)) (no challenger beats the table's choice c32u2 (nearest c32u4: ×0.812 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.809))); parallel-l16: ×1.383 [1], ×1.386 [2]; break-even ×1.381 [1], ×1.385 [2]
 *   - i4 g32 n1024 k8192 (case i4g32-n1024-k8192): one steady record: keeps staged-l32 with row blocks of 2 (the default's: staged-l32) (no candidate qualifies (nearest parallel-l32: ×1.038 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.037))); parallel-l32: ×1.038 [1], ×1.037 [2]; break-even ×1.037 [1], ×1.037 [2]
 *   - i4 g32 n3840 k1024 (case i4g32-n3840-k1024): one steady record: keeps parallel-l4 with row blocks of 2 (the default's: no row (c32u4)) (no challenger beats the table's choice parallel-l4 (nearest parallel-l2: ×0.994 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.991))); parallel-l4: ×1.445 [1], ×1.439 [2]; break-even ×1.444 [1], ×1.438 [2]
 *     - subgroup32-l4 meets the rule too (qualifies by ×1.494 at M=1): a Session option, not a row
 *   - i4 g32 n262144 k1024 (case i4g32-n262144-k1024): one steady record: keeps no row (c16u8) (the default's: no row (c32u4)) (no challenger beats the table's choice c16u8 (nearest c32u4: ×0.813 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.811))); staged-l16: ×1.342 [1], ×1.343 [2]; break-even ×1.339 [1], ×1.341 [2]
 *   - i4 g32 n8192 k2816 (case i4g32-n8192-k2816): one steady record: keeps parallel-l4 with row blocks of 4 (no candidate qualifies (nearest parallel-l32: parallel-l32 has no time at M=4 in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, which timed it at M=1 (a parallel row serves M = 1 to 8))); parallel-l32: ×1.355 [1], ×1.348 [2]; break-even ×1.344 [1], ×1.338 [2]
 *   - i4 g32 n4096 k2816 (case i4g32-n4096-k2816): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4 with row blocks of 2) (no candidate qualifies (nearest parallel-l32: ×1.042 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.042))); parallel-l32: ×1.042 [1], ×1.045 [2]; break-even ×1.042 [1], ×1.045 [2]
 *   - i4 g32 n2048 k2816 (case i4g32-n2048-k2816): one steady record: keeps parallel-l4 with row blocks of 4 (the default's: parallel-l4 with row blocks of 2) (no candidate qualifies (nearest parallel-l32: ×1.057 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.057))); parallel-l32: ×1.057 [1], ×1.065 [2]; break-even ×1.057 [1], ×1.065 [2]
 *   - i4 g32 n2112 k2816 (case i4g32-n2112-k2816): one steady record: keeps parallel-l4 with row blocks of 8 (the default's: parallel-l4 with row blocks of 2) (no candidate qualifies (nearest parallel-l32: ×1.051 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.050))); parallel-l32: ×1.051 [1], ×1.050 [2]; break-even ×1.050 [1], ×1.049 [2]
 *   - i4 g32 n1024 k2816 (case i4g32-n1024-k2816): one steady record: keeps parallel-l8 with row blocks of 4 (the default's: parallel-l32) (no challenger beats the table's choice parallel-l8 (nearest parallel-l4: ×0.903 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.089 (break-even ×0.883))); parallel-l8: ×1.245 [1], ×1.246 [2]; break-even ×1.221 [1], ×1.241 [2]
 *     - subgroup32-l8 meets the rule too (qualifies by ×1.278 at M=1): a Session option, not a row
 *   - i4 g32 n128 k2816 (case i4g32-n128-k2816): one steady record: keeps parallel-l32 (no candidate qualifies (nearest parallel-l16: ×0.928 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T18-03-03.json, under its bar ×1.110 (break-even ×0.891))); parallel-l16: ×0.928 [2]; break-even ×0.891 [2]
 *     - subgroup32-l32 meets the rule too (qualifies by ×1.184 at M=1): a Session option, not a row
 *   - i4 g32 n2816 k4096 (case i4g32-n2816-k4096): one steady record: keeps staged-l32 with row blocks of 2 (the default's: staged-l32) (no candidate qualifies (nearest parallel-l32: ×1.024 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.023))); parallel-l32: ×1.024 [1], ×1.020 [2]; break-even ×1.023 [1], ×1.020 [2]
 *   - i4 g32 n2816 k8192 (case i4g32-n2816-k8192): one steady record: keeps staged-l32 with row blocks of 2 (the default's: staged-l32) (no candidate qualifies (nearest parallel-l32: ×1.005 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×1.002))); parallel-l32: ×1.005 [1], ×1.008 [2]; break-even ×1.002 [1], ×1.001 [2]
 *   - i4 g32 n2816 k2112 (case i4g32-n2816-k2112): one steady record: keeps parallel-l2 with row blocks of 4 (the default's: staged-l32 with row blocks of 2) (no challenger beats the table's choice parallel-l2 (nearest parallel-l4: ×0.938 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.938))); parallel-l2: ×1.330 [1], ×1.329 [2]; break-even ×1.330 [1], ×1.326 [2]
 *     - subgroup32-l2 meets the rule too (qualifies by ×1.337 at M=1): a Session option, not a row
 *   - i4 g32 n262144 k2816 (case i4g32-n262144-k2816): one steady record: keeps parallel-l16 with row blocks of 4 (the default's: parallel-l4) (no challenger beats the table's choice parallel-l16 (nearest parallel-l8: ×0.972 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×0.970))); parallel-l16: ×1.294 [1], ×1.301 [2]; break-even ×1.290 [1], ×1.296 [2]
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): one steady record: keeps staged-l32 (the default's: no row (c16u8)) (no challenger beats the table's choice staged-l32 (nearest parallel-l8: ×0.967 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.070 (break-even ×0.962))); staged-l32: ×1.248 [1], ×1.243 [2]; break-even ×1.242 [1], ×1.235 [2]
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): one steady record: keeps parallel-l16 (no candidate qualifies (nearest parallel-l4: ×1.041 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.041))); parallel-l4: ×1.041 [1], ×1.038 [2]; break-even ×1.041 [1], ×1.038 [2]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): one steady record: keeps parallel-l32 (no candidate qualifies (nearest parallel-l16: ×1.024 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.100 (break-even ×0.991))); parallel-l16: ×1.024 [1], ×0.988 [2]; break-even ×0.991 [1], ×0.983 [2]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): one steady record: keeps no row (c16u8) (no candidate qualifies (nearest parallel-l4: ×0.996 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×0.996))); parallel-l4: ×0.996 [1], ×0.996 [2]; break-even ×0.996 [1], ×0.996 [2]
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): one steady record: keeps parallel-l8 (no candidate qualifies (nearest staged-l32: ×1.031 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.069 (break-even ×1.027))); staged-l32: ×1.031 [1], ×1.029 [2]; break-even ×1.027 [1], ×1.024 [2]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): one steady record: keeps parallel-l16 (no candidate qualifies (nearest parallel-l32: ×1.013 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×1.010))); parallel-l32: ×1.013 [1], ×1.013 [2]; break-even ×1.010 [1], ×1.011 [2]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): one steady record: keeps parallel-l8 (no candidate qualifies (nearest parallel-l4: ×1.060 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T18-03-03.json, under its bar ×1.083 (break-even ×1.042))); parallel-l4: ×1.089 [1], ×1.060 [2]; break-even ×1.072 [1], ×1.042 [2]
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): one steady record: keeps parallel-l8 (no candidate qualifies (nearest parallel-l16: ×1.002 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.090 (break-even ×0.980))); parallel-l16: ×1.002 [1], ×1.004 [2]; break-even ×0.980 [1], ×1.001 [2]
 *   - bf16 n512 k1536 (case bf16-n512-k1536): one steady record: keeps parallel-l16 (no candidate qualifies (nearest parallel-l8: ×0.934 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.086 (break-even ×0.917))); parallel-l8: ×0.934 [1], ×0.921 [2]; break-even ×0.917 [1], ×0.879 [2]
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): one steady record: keeps staged-l32 (no candidate qualifies (nearest parallel-l8: ×1.052 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.082 (break-even ×1.035))); parallel-l8: ×1.052 [1], ×1.057 [2]; break-even ×1.035 [1], ×1.041 [2]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): one steady record: keeps staged-l32 (the default's: no row (c16u8)) (no challenger beats the table's choice staged-l32 (nearest parallel-l8: ×0.971 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.968))); staged-l32: ×1.251 [1], ×1.250 [2]; break-even ×1.245 [1], ×1.246 [2]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): one steady record: keeps parallel-l16 (no candidate qualifies (nearest parallel-l32: ×1.023 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.021))); parallel-l32: ×1.023 [1], ×1.024 [2]; break-even ×1.021 [1], ×1.020 [2]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): one steady record: keeps staged-l16 (the default's: no row (c32u4)) (no challenger beats the table's choice staged-l16 (nearest parallel-l8: ×0.981 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.978))); staged-l16: ×1.317 [1], ×1.322 [2]; break-even ×1.310 [1], ×1.318 [2]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): one steady record: keeps staged-l32 (the default's: no row (c16u8)) (no challenger beats the table's choice staged-l32 (nearest parallel-l8: ×0.971 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×0.970))); staged-l32: ×1.266 [1], ×1.257 [2]; break-even ×1.265 [1], ×1.253 [2]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): one steady record: keeps staged-l32 (no candidate qualifies (nearest parallel-l16: ×1.022 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.020))); parallel-l16: ×1.022 [1], ×1.022 [2]; break-even ×1.020 [1], ×1.021 [2]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): one steady record: keeps parallel-l32 (no candidate qualifies (nearest parallel-l4: ×1.059 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.057))); parallel-l4: ×1.059 [1], ×1.064 [2]; break-even ×1.057 [1], ×1.063 [2]
 *   - bf16 n256 k2560 (case bf16-n256-k2560): one steady record: keeps parallel-l32 (no candidate qualifies (nearest parallel-l16: ×0.931 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.112 (break-even ×0.892))); parallel-l16: ×0.931 [1], ×0.955 [2]; break-even ×0.892 [1], ×0.931 [2]
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): one steady record: keeps parallel-l8 (no candidate qualifies (nearest parallel-l16: ×0.993 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.990))); parallel-l16: ×0.993 [1], ×0.997 [2]; break-even ×0.990 [1], ×0.996 [2]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): one steady record: keeps staged-l32 (the default's: no row (c32u4)) (no challenger beats the table's choice staged-l32 (nearest parallel-l8: ×0.973 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.971))); staged-l32: ×1.337 [1], ×1.327 [2]; break-even ×1.331 [1], ×1.321 [2]
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): one steady record: keeps parallel-l16 (no candidate qualifies (nearest parallel-l32: ×1.010 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.100 (break-even ×0.978))); parallel-l32: ×1.010 [1], ×1.009 [2]; break-even ×0.978 [1], ×1.008 [2]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): one steady record: keeps no row (c16u16) (no candidate qualifies (nearest parallel-l4: ×1.092 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.153 (break-even ×1.009))); parallel-l4: ×1.092 [1], ×1.066 [2]; break-even ×1.009 [1], ×0.983 [2]
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): one steady record: keeps parallel-l8 (no candidate qualifies (nearest staged-l32: ×1.033 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.067 (break-even ×1.030))); staged-l32: ×1.033 [1], ×1.032 [2]; break-even ×1.030 [1], ×1.030 [2]
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): one steady record: keeps parallel-l16 (no candidate qualifies (nearest parallel-l4: ×1.034 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.065 (break-even ×1.034))); parallel-l4: ×1.034 [1], ×1.035 [2]; break-even ×1.034 [1], ×1.035 [2]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): one steady record: keeps parallel-l16 (no candidate qualifies (nearest staged-l32: ×1.002 against the default's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.066 (break-even ×1.002))); staged-l32: ×1.002 [1], ×1.000 [2]; break-even ×1.002 [1], ×0.999 [2]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): one steady record: keeps parallel-l16 (the default's: no row (c32u4)) (no challenger beats the table's choice parallel-l16 (nearest parallel-l8: ×0.982 against the table's choice in outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json, under its bar ×1.068 (break-even ×0.980))); parallel-l16: ×1.315 [1], ×1.311 [2]; break-even ×1.302 [1], ×1.307 [2]
 * - rowsThreadTarget: keeps 16384 (the full and row-blocks sets measure the row-block heights per case and row count, but the field is one thread target for every shape, and no rule turns the heights into it)
 * - GEMV decisions that change at ×1.030 (GEMV_LOWEST_MARGIN, re-decided from this output): 8
 *   - sequential.shapes i4 n262144 k256 (i4g32-n262144-k256): c32u4 -> c16u8
 *   - sequential.shapes bf16 n262144 k1536 (bf16-n262144-k1536): c32u4 -> c16u16
 *   - sequential.shapes bf16 n10240 k2560 (bf16-n10240-k2560): c32u4 -> c16u16
 *   - sequential.shapes bf16 n262144 k2560 (bf16-n262144-k2560): c32u4 -> c16u16
 *   - parallelShapes i8 n262144 k1536 (i8-n262144-k1536): parallel-l16 with row blocks of 4 -> parallel-l16 with row blocks of 8
 *   - parallelShapes i4 g32 n256 k1024 (i4g32-n256-k1024): parallel-l32 -> parallel-l32 with row blocks of 2
 *   - parallelShapes i8 n1536 k256 (i8-n1536-k256): parallel-l4 with row blocks of 4 -> parallel-l4 with row blocks of 8
 *   - parallelShapes i8 n262144 k2560 (i8-n262144-k2560): parallel-l16 with row blocks of 4 -> parallel-l16 with row blocks of 8
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const APPLE_M5: GeometryProfile = {
  id: "apple-m5",
  match: { vendor: "apple", architecture: "metal-3", description: "Apple M5" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 8, form: "tas-tb-un" } },
    { maxRows: 64, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: 512, geometry: { regM: 8, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 8, wgX: 16, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
    pv: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 4, wgX: 16, wgY: 8, form: "tas-tb-un", decode: "counter" },
    rows32: { regM: 4, regN: 4, wgX: 16, wgY: 8, form: "tas-tb-un", decode: "counter" },
  },
  i8a8: {
    linear: [
      { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 4, tileK: 32 } },
      {
        maxRows: 512,
        geometry: { regM: 8, regN: 4, wgX: 16, wgY: 4, tileK: 16, skeleton: "vec4" },
      },
      {
        maxRows: Number.POSITIVE_INFINITY,
        geometry: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
      },
    ],
    attentionQk: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
    attentionPv: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 16 },
    conv3d: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
  },
  linearGemv: {
    sequential: {
      variant: { cols: 32, unroll: 4 },
      shapes: [
        { storage: "bf16", n: 8960, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 2048, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 256, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 1536, k: 2048, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 6144, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 1536, k: 6144, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 1536, k: 256, variant: { cols: 32, unroll: 16 } },
        { storage: "bf16", n: 4096, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 512, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 1536, k: 4096, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 12288, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 1536, k: 12288, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 10752, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 2048, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 512, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 2560, k: 2048, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 2560, k: 10240, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 2560, k: 256, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 4096, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 1024, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 2560, k: 4096, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 8960, k: 1536, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 1536, k: 256, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 1536, k: 12288, variant: { cols: 64, unroll: 2 } },
        { storage: "i4", n: 262144, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 256, k: 3072, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 1024, k: 256, variant: { cols: 64, unroll: 2 } },
        { storage: "i4", n: 256, k: 1024, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 2048, k: 256, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 256, k: 2048, variant: { cols: 16, unroll: 2 } },
        { storage: "i8", n: 256, k: 3072, variant: { cols: 16, unroll: 4 } },
        { storage: "i8", n: 256, k: 1024, variant: { cols: 16, unroll: 4 } },
        { storage: "i8", n: 256, k: 2048, variant: { cols: 16, unroll: 4 } },
        { storage: "i8", n: 256, k: 1536, variant: { cols: 16, unroll: 4 } },
        { storage: "i4", n: 10752, k: 2560, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 256, k: 2560, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 2560, k: 256, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 262144, k: 2560, variant: { cols: 128, unroll: 8 } },
        { storage: "i4", n: 256, k: 5120, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 15360, k: 3840, variant: { cols: 64, unroll: 2 } },
        { storage: "i4", n: 8192, k: 3840, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 4096, k: 3840, variant: { cols: 256, unroll: 2 } },
        { storage: "i4", n: 2048, k: 3840, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 512, k: 3840, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 3840, k: 4096, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 3840, k: 8192, variant: { cols: 256, unroll: 2 } },
        { storage: "i4", n: 3840, k: 15360, variant: { cols: 256, unroll: 2 } },
        { storage: "i4", n: 262144, k: 3840, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 1024, k: 7680, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 4096, k: 1024, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 1024, k: 4096, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 8192, k: 1024, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 1024, k: 8192, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 3840, k: 1024, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 262144, k: 1024, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 8192, k: 2816, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 4096, k: 2816, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 2048, k: 2816, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 2112, k: 2816, variant: { cols: 256, unroll: 2 } },
        { storage: "i4", n: 1024, k: 2816, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 128, k: 2816, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 2816, k: 4096, variant: { cols: 64, unroll: 2 } },
        { storage: "i4", n: 2816, k: 8192, variant: { cols: 32, unroll: 2 } },
        { storage: "i4", n: 262144, k: 2816, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 12288, k: 1536, variant: { cols: 32, unroll: 2 } },
        { storage: "i8", n: 256, k: 2560, variant: { cols: 16, unroll: 4 } },
      ],
    },
    rowsThreadTarget: 16384,
    parallelShapes: [
      {
        storage: "i4",
        n: 8960,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2048,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 256,
        k: 1536,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1536,
        k: 2048,
        group: 32,
        lanes: 8,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 6144,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1536,
        k: 6144,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1536,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 4096,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 1536,
        group: 32,
        lanes: 8,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1536,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 12288,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1536,
        k: 12288,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      { storage: "i8", n: 262144, k: 1536, lanes: 16, packedActivations: false, rowBlock: 4 },
      {
        storage: "i4",
        n: 2048,
        k: 1536,
        group: 512,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 256,
        k: 1536,
        group: 512,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 6144,
        k: 1536,
        group: 512,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 1536, lanes: 32, packedActivations: false, rowBlock: 4 },
      { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false, rowBlock: 4 },
      {
        storage: "i4",
        n: 4096,
        k: 1536,
        group: 512,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 1536,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      { storage: "i2", n: 12288, k: 1536, lanes: 2, packedActivations: false, rowBlock: 4 },
      { storage: "i2", n: 1536, k: 12288, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 10752,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2048,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 2560,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 256,
        k: 2560,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2560,
        k: 2048,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 10240,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2560,
        k: 10240,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2560,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 4096,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2560,
        group: 32,
        lanes: 8,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2560,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      { storage: "i8", n: 262144, k: 2560, lanes: 16, packedActivations: false, rowBlock: 4 },
      {
        storage: "i4",
        n: 2048,
        k: 2560,
        group: 512,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 2560,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      { storage: "i4", n: 2560, k: 2048, group: 2048, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 10240,
        k: 2560,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 2560, lanes: 32, packedActivations: false, rowBlock: 4 },
      { storage: "i8", n: 2560, k: 256, lanes: 4, packedActivations: false, rowBlock: 4 },
      {
        storage: "i4",
        n: 4096,
        k: 2560,
        group: 512,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2560,
        group: 512,
        lanes: 8,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "i4", n: 2560, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 256,
        k: 3072,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "i4", n: 256, k: 1024, group: 32, lanes: 32, packedActivations: false },
      {
        storage: "i4",
        n: 15360,
        k: 3840,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 8192,
        k: 3840,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 4096,
        k: 3840,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2048,
        k: 3840,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 3840,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 8192,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 15360,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 262144,
        k: 3840,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1024,
        k: 7680,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 8192,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 8192,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 4096,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2048,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2112,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 8,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2816,
        group: 32,
        lanes: 8,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "i4", n: 128, k: 2816, group: 32, lanes: 32, packedActivations: false },
      {
        storage: "i4",
        n: 2816,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2816,
        k: 8192,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2816,
        k: 2112,
        group: 32,
        lanes: 2,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 262144,
        k: 2816,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "bf16", n: 2048, k: 1536, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 256, k: 1536, lanes: 32, packedActivations: false },
      { storage: "bf16", n: 6144, k: 1536, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 1536, k: 6144, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 1536, k: 256, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 4096, k: 1536, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 512, k: 1536, lanes: 16, packedActivations: false },
      {
        storage: "bf16",
        n: 1536,
        k: 4096,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "bf16", n: 1536, k: 12288, lanes: 16, packedActivations: false },
      {
        storage: "bf16",
        n: 2048,
        k: 2560,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "bf16", n: 512, k: 2560, lanes: 32, packedActivations: false },
      { storage: "bf16", n: 256, k: 2560, lanes: 32, packedActivations: false },
      { storage: "bf16", n: 2560, k: 2048, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 2560, k: 10240, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 4096, k: 2560, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 1024, k: 2560, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 2560, k: 4096, lanes: 16, packedActivations: false },
      {
        storage: "i4",
        n: 1024,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2048,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 256,
        k: 2048,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "i8", n: 256, k: 3072, lanes: 16, packedActivations: false, rowBlock: 2 },
      { storage: "i8", n: 1024, k: 256, lanes: 4, packedActivations: false, rowBlock: 4 },
      { storage: "i8", n: 256, k: 1024, lanes: 16, packedActivations: false, rowBlock: 4 },
      {
        storage: "i8",
        n: 2048,
        k: 256,
        lanes: 4,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i8",
        n: 256,
        k: 2048,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 262144,
        k: 2560,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 256,
        k: 5120,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 3840,
        k: 1024,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "bf16",
        n: 8960,
        k: 1536,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "bf16",
        n: 12288,
        k: 1536,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "bf16",
        n: 262144,
        k: 1536,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "bf16",
        n: 10752,
        k: 2560,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "bf16",
        n: 10240,
        k: 2560,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "bf16", n: 262144, k: 2560, lanes: 16, packedActivations: false },
    ],
  },
  provenance: {
    adapter: {
      vendor: "apple",
      architecture: "metal-3",
      device: "0x0000",
      description: "Apple M5",
    },
    families: {
      gemm: {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "743eba3965aa8f5d",
        caseSet: "e6ddd45226ab2155",
      },
      attention: {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "c3f4a946ccbf1df9",
        caseSet: "9ee211cf69674928",
      },
      conv2d: {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "82887096d13feac2",
        caseSet: "02186e45558e4151",
      },
      "i8a8-linear": {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "221915f80d9eaabf",
        caseSet: "3d32dc6f10559af2",
      },
      "i8a8-attention": {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "43c3180afc0bc5cd",
        caseSet: "8ab1a86b13087640",
      },
      "i8a8-conv3d": {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "a1666319e42c7dcd",
        caseSet: "27ce9da45b359b14",
      },
    },
  },
};
