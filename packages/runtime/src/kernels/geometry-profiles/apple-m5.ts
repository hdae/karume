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
 * Carried over from the registered table at commit 80ca34104c3d6d6a5e74add5c8f6e3bfb1319071: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d.
 * Their adoption lines are in `git show 80ca34104c3d6d6a5e74add5c8f6e3bfb1319071:packages/runtime/src/kernels/geometry-profiles/apple-m5.ts`.
 *
 * GEMV choices: a candidate replaces the default profile's choice for a shape only when it is at
 * least ×1.100 faster than that choice in every GEMV sweep record, with its words equal to
 * the baseline's (identical class: the sequential variant's per-shape exceptions) or inside the
 * band (reorder class: the parallel rows, measured at every row count of the parallel range and
 * not slower at any) in every record; among those, the one whose smallest gain is largest. The
 * choice's time is its fastest over the records, and only when a second record repeats it within
 * ×1.100: a shape whose choice has no such time keeps the default, and so do the shared
 * sequential keys whose cases disagree and the rows with packed activations.
 *
 * - linearGemv (the GEMV choices of linear), from 2 GEMV sweep records:
 *   - [1] outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T16-43-29.json (2026-11-23T16:43:28.766Z, set full): 345 groups — 345 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [2] outputs/workbench/browser-apple-m5/gemv-sweep_2026-11-23T18-03-03.json (2026-11-23T18:03:02.056Z, set full): 345 groups — 345 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 * - identical class (sequential.shapes, M = 1): 43 adopted, 56 kept
 *   - i4 n8960 k1536 (case i4g32-n8960-k1536): c32u4 -> c32u2 (×1.244 [1], ×1.245 [2])
 *   - i4 n2048 k1536 (cases i4g32-n2048-k1536, i4g512-n2048-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2048-k1536 adopts c32u2; i4g512-n2048-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.995 < ×1.100 in [1])))
 *   - i4 n256 k1536 (cases i4g32-n256-k1536, i4g512-n256-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n256-k1536 adopts c16u2; i4g512-n256-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.063 < ×1.100 in [1])))
 *   - i4 n1536 k2048 (cases i4g32-n1536-k2048, i4g2048-n1536-k2048): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n1536-k2048 adopts c64u2; i4g2048-n1536-k2048 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.992 < ×1.100 in [1])))
 *   - i4 n6144 k1536 (cases i4g32-n6144-k1536, i4g512-n6144-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n6144-k1536 adopts c64u2; i4g512-n6144-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u4: ×1.002 < ×1.100 in [1])))
 *   - i4 n1536 k6144 (cases i4g32-n1536-k6144, i4g2048-n1536-k6144): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n1536-k6144 adopts c16u2; i4g2048-n1536-k6144 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.004 < ×1.100 in [1])))
 *   - i4 n1536 k256 (case i4g32-n1536-k256): c32u4 -> c32u2 (×1.368 [1], ×1.367 [2])
 *   - i4 n4096 k1536 (cases i4g32-n4096-k1536, i4g512-n4096-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n4096-k1536 adopts c32u2; i4g512-n4096-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×1.000 < ×1.100 in [1])))
 *   - i4 n512 k1536 (cases i4g32-n512-k1536, i4g512-n512-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n512-k1536 adopts c16u2; i4g512-n512-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.066 < ×1.100 in [1])))
 *   - i4 n1536 k4096 (cases i4g32-n1536-k4096, i4g4096-n1536-k4096): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n1536-k4096 adopts c128u2; i4g4096-n1536-k4096 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×0.989 < ×1.100 in [1])))
 *   - i4 n12288 k1536 (case i4g32-n12288-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u2: ×1.098 < ×1.100 in [1])); c32u2: ×1.098 [1], ×1.123 [2]
 *   - i4 n1536 k12288 (case i4g32-n1536-k12288): c32u4 -> c64u2 (×1.622 [1], ×1.624 [2])
 *   - i8 n262144 k1536 (case i8-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.999 < ×1.100 in [1])); c64u4: ×0.999 [1], ×0.974 [2]
 *   - i4 n262144 k1536 (case i4g32-n262144-k1536): c32u4 -> c16u8 (×1.232 [1], ×1.219 [2])
 *   - i4 n256 k3072 (case i4g32-n256-k3072): c32u4 -> c16u2 (×1.740 [1], ×1.750 [2])
 *   - i4 n1024 k256 (case i4g32-n1024-k256): c32u4 -> c64u2 (×1.476 [1], ×1.454 [2])
 *   - i4 n256 k1024 (case i4g32-n256-k1024): c32u4 -> c16u2 (×1.704 [1], ×1.690 [2])
 *   - i4 n2048 k256 (case i4g32-n2048-k256): c32u4 -> c32u2 (×1.328 [1], ×1.331 [2])
 *   - i4 n256 k2048 (case i4g32-n256-k2048): c32u4 -> c16u2 (×1.743 [1], ×1.736 [2])
 *   - i4 n262144 k256 (case i4g32-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.052 < ×1.100 in [1])); c16u8: ×1.052 [1], ×1.053 [2]
 *   - i8 n256 k3072 (case i8-n256-k3072): c32u4 -> c16u4 (×1.132 [1], ×1.111 [2])
 *   - i8 n1024 k256 (case i8-n1024-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.041 < ×1.100 in [1])); c16u4: ×1.041 [1], ×1.011 [2]
 *   - i8 n256 k1024 (case i8-n256-k1024): c32u4 -> c16u4 (×1.164 [1], ×1.137 [2])
 *   - i8 n2048 k256 (case i8-n2048-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.974 < ×1.100 in [1])); c64u4: ×0.974 [1], ×0.966 [2]
 *   - i8 n256 k2048 (case i8-n256-k2048): c32u4 -> c16u4 (×1.131 [1], ×1.130 [2])
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u4: ×1.002 < ×1.100 in [1])); c128u4: ×1.002 [1], ×1.001 [2]
 *   - i8 n256 k1536 (case i8-n256-k1536): c32u4 -> c16u4 (×1.116 [1], ×1.101 [2])
 *   - i8 n1536 k256 (case i8-n1536-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.008 < ×1.100 in [1])); c16u4: ×1.008 [1], ×1.007 [2]
 *   - i2 n12288 k1536 (case i2-n12288-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.985 < ×1.100 in [1])); c64u4: ×0.985 [1], ×0.993 [2]
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.998 < ×1.100 in [1])); c64u4: ×0.998 [1], ×0.997 [2]
 *   - i2 n262144 k1536 (case i2-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×1.000 < ×1.100 in [1])); c64u4: ×1.000 [1], ×1.004 [2]
 *   - i4 n10752 k2560 (case i4g32-n10752-k2560): c32u4 -> c16u2 (×1.179 [1], ×1.142 [2])
 *   - i4 n2048 k2560 (cases i4g32-n2048-k2560, i4g512-n2048-k2560): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2048-k2560 adopts c16u2; i4g512-n2048-k2560 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.998 < ×1.100 in [1])))
 *   - i4 n512 k2560 (cases i4g32-n512-k2560, i4g512-n512-k2560): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n512-k2560 adopts c16u2; i4g512-n512-k2560 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.053 < ×1.100 in [1])))
 *   - i4 n256 k2560 (case i4g32-n256-k2560): c32u4 -> c16u2 (×1.740 [1], ×1.742 [2])
 *   - i4 n2560 k2048 (cases i4g32-n2560-k2048, i4g2048-n2560-k2048): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2560-k2048 adopts c16u2; i4g2048-n2560-k2048 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.996 < ×1.100 in [1])))
 *   - i4 n10240 k2560 (cases i4g32-n10240-k2560, i4g512-n10240-k2560): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n10240-k2560 adopts c32u8; i4g512-n10240-k2560 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u4: ×0.975 < ×1.100 in [1])))
 *   - i4 n2560 k10240 (cases i4g32-n2560-k10240, i4g2048-n2560-k10240): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2560-k10240 adopts c256u2; i4g2048-n2560-k10240 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×0.947 < ×1.100 in [1])))
 *   - i4 n2560 k256 (case i4g32-n2560-k256): c32u4 -> c32u2 (×1.331 [1], ×1.359 [2])
 *   - i4 n4096 k2560 (cases i4g32-n4096-k2560, i4g512-n4096-k2560): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n4096-k2560 adopts c64u2; i4g512-n4096-k2560 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×1.000 < ×1.100 in [1])))
 *   - i4 n1024 k2560 (cases i4g32-n1024-k2560, i4g512-n1024-k2560): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n1024-k2560 adopts c32u2; i4g512-n1024-k2560 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.033 < ×1.100 in [1])))
 *   - i4 n2560 k4096 (cases i4g32-n2560-k4096, i4g4096-n2560-k4096): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2560-k4096 adopts c16u2; i4g4096-n2560-k4096 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.995 < ×1.100 in [1])))
 *   - i8 n262144 k2560 (case i8-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.994 < ×1.100 in [1])); c64u4: ×0.994 [1], ×0.989 [2]
 *   - i4 n262144 k2560 (case i4g32-n262144-k2560): c32u4 -> c128u8 (×1.241 [1], ×1.238 [2])
 *   - i4 n256 k5120 (case i4g32-n256-k5120): c32u4 -> c16u2 (×1.740 [1], ×1.736 [2])
 *   - i8 n256 k2560 (case i8-n256-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.099 < ×1.100 in [1])); c16u4: ×1.099 [1], ×1.086 [2]
 *   - i8 n2560 k256 (case i8-n2560-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×0.992 < ×1.100 in [1])); c64u4: ×0.992 [1], ×1.011 [2]
 *   - i2 n262144 k2560 (case i2-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u4: ×1.005 < ×1.100 in [1])); c64u4: ×1.005 [1], ×1.001 [2]
 *   - i4 n15360 k3840 (case i4g32-n15360-k3840): c32u4 -> c64u2 (×1.175 [1], ×1.177 [2])
 *   - i4 n8192 k3840 (case i4g32-n8192-k3840): c32u4 -> c32u2 (×1.204 [1], ×1.181 [2])
 *   - i4 n4096 k3840 (case i4g32-n4096-k3840): c32u4 -> c256u2 (×1.352 [1], ×1.359 [2])
 *   - i4 n2048 k3840 (case i4g32-n2048-k3840): c32u4 -> c16u2 (×1.656 [1], ×1.652 [2])
 *   - i4 n512 k3840 (case i4g32-n512-k3840): c32u4 -> c16u2 (×1.736 [1], ×1.731 [2])
 *   - i4 n3840 k4096 (case i4g32-n3840-k4096): c32u4 -> c32u2 (×1.406 [1], ×1.419 [2])
 *   - i4 n3840 k8192 (case i4g32-n3840-k8192): c32u4 -> c256u2 (×1.308 [1], ×1.325 [2])
 *   - i4 n3840 k15360 (case i4g32-n3840-k15360): c32u4 -> c256u2 (×1.284 [1], ×1.295 [2])
 *   - i4 n262144 k3840 (case i4g32-n262144-k3840): c32u4 -> c16u2 (×1.257 [1], ×1.248 [2])
 *   - i4 n1024 k7680 (case i4g32-n1024-k7680): c32u4 -> c32u2 (×1.772 [1], ×1.799 [2])
 *   - i4 n4096 k1024 (case i4g32-n4096-k1024): c32u4 -> c32u2 (×1.257 [1], ×1.257 [2])
 *   - i4 n1024 k4096 (case i4g32-n1024-k4096): c32u4 -> c32u2 (×1.763 [1], ×1.765 [2])
 *   - i4 n8192 k1024 (case i4g32-n8192-k1024): c32u4 -> c32u2 (×1.232 [1], ×1.234 [2])
 *   - i4 n1024 k8192 (case i4g32-n1024-k8192): c32u4 -> c32u2 (×1.743 [1], ×1.767 [2])
 *   - i4 n3840 k1024 (case i4g32-n3840-k1024): c32u4 -> c32u2 (×1.303 [1], ×1.302 [2])
 *   - i4 n262144 k1024 (case i4g32-n262144-k1024): c32u4 -> c16u8 (×1.230 [1], ×1.229 [2])
 *   - i4 n8192 k2816 (case i4g32-n8192-k2816): c32u4 -> c32u8 (×1.130 [1], ×1.141 [2])
 *   - i4 n4096 k2816 (case i4g32-n4096-k2816): c32u4 -> c32u2 (×1.346 [1], ×1.353 [2])
 *   - i4 n2048 k2816 (case i4g32-n2048-k2816): c32u4 -> c16u2 (×1.619 [1], ×1.600 [2])
 *   - i4 n2112 k2816 (case i4g32-n2112-k2816): c32u4 -> c256u2 (×1.626 [1], ×1.637 [2])
 *   - i4 n1024 k2816 (case i4g32-n1024-k2816): c32u4 -> c32u2 (×1.688 [1], ×1.674 [2])
 *   - i4 n128 k2816 (case i4g32-n128-k2816): c32u4 -> c16u2 (×1.730 [1], ×1.744 [2])
 *   - i4 n2816 k4096 (case i4g32-n2816-k4096): c32u4 -> c64u2 (×1.439 [1], ×1.441 [2])
 *   - i4 n2816 k8192 (case i4g32-n2816-k8192): c32u4 -> c32u2 (×1.338 [1], ×1.313 [2])
 *   - i4 n2816 k2112 (case i4g32-n2816-k2112): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: its words differ from the baseline's in [1] at M=1))
 *   - i4 n262144 k2816 (case i4g32-n262144-k2816): c32u4 -> c32u8 (×1.234 [1], ×1.183 [2])
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.998 < ×1.100 in [1])); c16u16: ×0.998 [1], ×0.996 [2]
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.008 < ×1.100 in [1])); c16u4: ×1.008 [1], ×1.000 [2]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.897 < ×1.100 in [1])); c16u8: ×0.897 [1], ×0.892 [2]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×0.999 < ×1.100 in [1])); c16u4: ×0.999 [1], ×1.001 [2]
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.994 < ×1.100 in [1])); c16u16: ×0.994 [1], ×0.986 [2]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.037 < ×1.100 in [1])); c16u8: ×1.037 [1], ×1.019 [2]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): keeps c32u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.927 < ×1.100 in [1])); c16u16: ×0.927 [1], ×0.908 [2]
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.994 < ×1.100 in [1])); c16u16: ×0.994 [1], ×0.960 [2]
 *   - bf16 n512 k1536 (case bf16-n512-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×0.976 < ×1.100 in [1])); c32u16: ×0.976 [1], ×0.979 [2]
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u16: ×0.990 < ×1.100 in [1])); c64u16: ×0.990 [1], ×1.016 [2]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.001 < ×1.100 in [1])); c16u16: ×1.001 [1], ×0.993 [2]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.026 < ×1.100 in [1])); c32u8: ×1.026 [1], ×1.022 [2]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.064 < ×1.100 in [1])); c16u8: ×1.064 [1], ×1.057 [2]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.003 < ×1.100 in [1])); c16u16: ×1.003 [1], ×0.997 [2]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u16: ×1.006 < ×1.100 in [1])); c128u16: ×1.006 [1], ×1.007 [2]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.015 < ×1.100 in [1])); c16u8: ×1.015 [1], ×1.012 [2]
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.884 < ×1.100 in [1])); c16u8: ×0.884 [1], ×0.888 [2]
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.984 < ×1.100 in [1])); c16u16: ×0.984 [1], ×1.015 [2]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.067 < ×1.100 in [1])); c16u16: ×1.067 [1], ×1.062 [2]
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c256u8: ×1.010 < ×1.100 in [1])); c256u8: ×1.010 [1], ×1.012 [2]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.028 < ×1.100 in [1])); c32u16: ×1.028 [1], ×1.026 [2]
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.988 < ×1.100 in [1])); c16u16: ×0.988 [1], ×0.998 [2]
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u16: ×0.998 < ×1.100 in [1])); c64u16: ×0.998 [1], ×1.003 [2]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u8: ×0.996 < ×1.100 in [1])); c64u8: ×0.996 [1], ×1.004 [2]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.068 < ×1.100 in [1])); c16u16: ×1.068 [1], ×1.070 [2]
 * - reorder class (parallelShapes, M = 1 to 8): 66 adopted, 49 kept
 *   - i4 g32 n8960 k1536 (case i4g32-n8960-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.016 < ×1.100 in [1])); parallel-l16: ×1.016 [1], ×1.013 [2]
 *   - i4 g32 n2048 k1536 (case i4g32-n2048-k1536): parallel-l4 with row blocks of 2 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.141 [1] M=4, ×1.125 [2] M=4, ×1.221 [1] M=8, ×1.213 [2] M=8
 *   - i4 g32 n256 k1536 (case i4g32-n256-k1536): parallel-l32 with row blocks of 2 -> parallel-l16 with row blocks of 4 (×1.243 [1], ×1.181 [2]); at the other row counts: ×1.416 [1] M=4, ×1.395 [2] M=4, ×1.637 [1] M=8, ×1.632 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.435 [1], ×1.391 [2]): a Session option, not a row
 *   - i4 g32 n1536 k2048 (case i4g32-n1536-k2048): staged-l32 -> parallel-l8 with row blocks of 4 (×1.510 [1], ×1.525 [2]); at the other row counts: ×2.449 [1] M=4, ×2.457 [2] M=4, ×2.621 [1] M=8, ×2.611 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.601 [1], ×1.602 [2]): a Session option, not a row
 *   - i4 g32 n6144 k1536 (case i4g32-n6144-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.025 < ×1.100 in [1])); parallel-l16: ×1.025 [1], ×1.023 [2]
 *   - i4 g32 n1536 k6144 (case i4g32-n1536-k6144): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: ×1.065 < ×1.100 in [1])); parallel-l4: ×1.065 [1], ×1.055 [2]
 *   - i4 g32 n1536 k256 (case i4g32-n1536-k256): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.303 [1] M=4, ×1.302 [2] M=4, ×1.347 [1] M=8, ×1.348 [2] M=8
 *     - subgroup32-l4 meets the rule too (×1.109 [1], ×1.123 [2]): a Session option, not a row
 *   - i4 g32 n4096 k1536 (case i4g32-n4096-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.010 < ×1.100 in [1])); parallel-l8: ×1.010 [1], ×1.014 [2]
 *   - i4 g32 n512 k1536 (case i4g32-n512-k1536): parallel-l32 -> parallel-l8 with row blocks of 4 (×1.241 [1], ×1.236 [2]); at the other row counts: ×1.901 [1] M=4, ×1.901 [2] M=4, ×2.115 [1] M=8, ×2.117 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.401 [1], ×1.367 [2]): a Session option, not a row
 *   - i4 g32 n1536 k4096 (case i4g32-n1536-k4096): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.276 [1] M=4, ×1.274 [2] M=4, ×1.256 [1] M=8, ×1.254 [2] M=8
 *   - i4 g32 n12288 k1536 (case i4g32-n12288-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4))); parallel-l16: ×1.168 [1], ×1.193 [2]
 *   - i4 g32 n1536 k12288 (case i4g32-n1536-k12288): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.052 < ×1.100 in [1])); parallel-l32: ×1.052 [1], ×1.038 [2]
 *   - i8 n262144 k1536 (case i8-n262144-k1536): parallel-l16 -> parallel-l16 with row blocks of 4; at the other row counts: ×1.654 [1] M=4, ×1.646 [2] M=4, ×1.750 [1] M=8, ×1.719 [2] M=8
 *   - i4 g32 n262144 k1536 (case i4g32-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4))); staged-l16: ×1.308 [1], ×1.295 [2]
 *   - i4 g32 n256 k3072 (case i4g32-n256-k3072): parallel-l32 -> parallel-l16 with row blocks of 4 (×1.108 [1], ×1.112 [2]); at the other row counts: ×1.385 [1] M=4, ×1.382 [2] M=4, ×1.499 [1] M=8, ×1.490 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.200 [1], ×1.223 [2]): a Session option, not a row
 *   - i4 g32 n1024 k256 (case i4g32-n1024-k256): no row (c32u4) -> parallel-l4 with row blocks of 4 (×1.768 [1], ×1.802 [2]); at the other row counts: ×1.213 [1] M=4, ×1.222 [2] M=4, ×1.187 [1] M=8, ×1.193 [2] M=8
 *     - subgroup32-l4 meets the rule too (×2.047 [1], ×1.984 [2]): a Session option, not a row
 *   - i4 g32 n256 k1024 (case i4g32-n256-k1024): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.098 < ×1.100 in [1])); parallel-l16: ×1.098 [1], ×1.142 [2]
 *     - subgroup32-l16 meets the rule too (×1.380 [1], ×1.371 [2]): a Session option, not a row
 *   - i4 g32 n2048 k256 (case i4g32-n2048-k256): no row (c32u4) -> staged-l4 with row blocks of 4 (×1.403 [1], ×1.405 [2]); at the other row counts: ×1.245 [1] M=4, ×1.235 [2] M=4, ×1.336 [1] M=8, ×1.336 [2] M=8
 *     - subgroup32-l4 meets the rule too (×1.530 [1], ×1.535 [2]): a Session option, not a row
 *   - i4 g32 n256 k2048 (case i4g32-n256-k2048): no row (c32u4) -> parallel-l16 with row blocks of 4 (×7.269 [1], ×7.298 [2]); at the other row counts: ×2.173 [1] M=4, ×2.175 [2] M=4, ×1.550 [1] M=8, ×1.554 [2] M=8
 *     - subgroup32-l16 meets the rule too (×8.261 [1], ×8.046 [2]): a Session option, not a row
 *   - i4 g32 n262144 k256 (case i4g32-n262144-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: ×1.079 < ×1.100 in [1])); staged-l8: ×1.079 [1], ×1.080 [2]
 *   - i8 n256 k3072 (case i8-n256-k3072): no row (c32u4) -> parallel-l16 with row blocks of 2 (×4.676 [1], ×4.675 [2]); at the other row counts: ×2.691 [1] M=4, ×2.728 [2] M=4, ×1.759 [1] M=8, ×1.768 [2] M=8
 *     - subgroup32-l32 meets the rule too (×5.488 [1], ×5.269 [2]): a Session option, not a row
 *   - i8 n1024 k256 (case i8-n1024-k256): no row (c32u4) -> parallel-l4 with row blocks of 4 (×1.258 [1], ×1.273 [2]); at the other row counts: ×1.225 [1] M=4, ×1.229 [2] M=4, ×1.320 [1] M=8, ×1.320 [2] M=8
 *     - subgroup32-l4 meets the rule too (×1.439 [1], ×1.426 [2]): a Session option, not a row
 *   - i8 n256 k1024 (case i8-n256-k1024): no row (c32u4) -> parallel-l16 with row blocks of 4 (×3.189 [1], ×3.021 [2]); at the other row counts: ×1.891 [1] M=4, ×1.911 [2] M=4, ×1.627 [1] M=8, ×1.626 [2] M=8
 *     - subgroup32-l16 meets the rule too (×3.693 [1], ×3.669 [2]): a Session option, not a row
 *   - i8 n2048 k256 (case i8-n2048-k256): no row (c32u4) -> staged-l4 with row blocks of 4 (×1.119 [1], ×1.112 [2]); at the other row counts: ×1.388 [1] M=4, ×1.388 [2] M=4, ×1.395 [1] M=8, ×1.395 [2] M=8
 *   - i8 n256 k2048 (case i8-n256-k2048): no row (c32u4) -> staged-l32 with row blocks of 4 (×4.058 [1], ×4.053 [2]); at the other row counts: ×2.146 [1] M=4, ×2.150 [2] M=4, ×1.553 [1] M=8, ×1.550 [2] M=8
 *     - subgroup32-l32 meets the rule too (×4.798 [1], ×4.587 [2]): a Session option, not a row
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4))); parallel-l8: ×1.306 [1], ×1.303 [2]
 *   - i4 g512 n2048 k1536 (case i4g512-n2048-k1536): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.512 [1] M=4, ×1.512 [2] M=4, ×1.613 [1] M=8, ×1.614 [2] M=8
 *   - i4 g512 n256 k1536 (case i4g512-n256-k1536): parallel-l32 -> parallel-l16 with row blocks of 4 (×1.122 [1], ×1.109 [2]); at the other row counts: ×1.506 [1] M=4, ×1.499 [2] M=4, ×1.690 [1] M=8, ×1.697 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.358 [1], ×1.364 [2]): a Session option, not a row
 *   - i4 g2048 n1536 k2048 (case i4g2048-n1536-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l8 meets the rule too (×1.334 [1], ×1.326 [2]): a Session option, not a row
 *   - i4 g512 n6144 k1536 (case i4g512-n6144-k1536): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.573 [1] M=4, ×1.573 [2] M=4, ×1.640 [1] M=8, ×1.644 [2] M=8
 *   - i4 g2048 n1536 k6144 (case i4g2048-n1536-k6144): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k1536 (case i8-n256-k1536): parallel-l32 -> parallel-l32 with row blocks of 4; at the other row counts: ×1.208 [1] M=4, ×1.204 [2] M=4, ×1.257 [1] M=8, ×1.256 [2] M=8
 *     - subgroup32-l32 meets the rule too (×1.128 [1], ×1.134 [2]): a Session option, not a row
 *   - i8 n1536 k256 (case i8-n1536-k256): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.418 [1] M=4, ×1.449 [2] M=4, ×1.421 [1] M=8, ×1.425 [2] M=8
 *   - i4 g512 n4096 k1536 (case i4g512-n4096-k1536): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.562 [1] M=4, ×1.569 [2] M=4, ×1.629 [1] M=8, ×1.631 [2] M=8
 *   - i4 g512 n512 k1536 (case i4g512-n512-k1536): parallel-l32 -> staged-l16 with row blocks of 4 (×1.234 [1], ×1.238 [2]); at the other row counts: ×1.722 [1] M=4, ×1.718 [2] M=4, ×1.928 [1] M=8, ×1.928 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.368 [1], ×1.393 [2]): a Session option, not a row
 *   - i4 g4096 n1536 k4096 (case i4g4096-n1536-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n12288 k1536 (case i2-n12288-k1536): parallel-l2 -> parallel-l2 with row blocks of 4; at the other row counts: ×1.694 [1] M=4, ×1.696 [2] M=4, ×1.786 [1] M=8, ×1.788 [2] M=8
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k1536 (case i2-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: ×0.985 < ×1.100 in [1])); parallel-l2: ×0.985 [1], ×1.008 [2]
 *   - i4 g32 n10752 k2560 (case i4g32-n10752-k2560): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4))); parallel-l32: ×1.380 [1], ×1.384 [2]
 *   - i4 g32 n2048 k2560 (case i4g32-n2048-k2560): parallel-l4 with row blocks of 2 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.128 [1] M=4, ×1.128 [2] M=4, ×1.183 [1] M=8, ×1.184 [2] M=8
 *   - i4 g32 n512 k2560 (case i4g32-n512-k2560): parallel-l32 -> staged-l16 with row blocks of 4 (×1.159 [1], ×1.152 [2]); at the other row counts: ×1.638 [1] M=4, ×1.633 [2] M=4, ×1.777 [1] M=8, ×1.778 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.304 [1], ×1.311 [2]): a Session option, not a row
 *   - i4 g32 n256 k2560 (case i4g32-n256-k2560): parallel-l32 -> parallel-l16 with row blocks of 4 (×1.156 [1], ×1.138 [2]); at the other row counts: ×1.459 [1] M=4, ×1.460 [2] M=4, ×1.603 [1] M=8, ×1.603 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.314 [1], ×1.301 [2]): a Session option, not a row
 *   - i4 g32 n2560 k2048 (case i4g32-n2560-k2048): staged-l32 -> parallel-l4 with row blocks of 4 (×1.132 [1], ×1.134 [2]); at the other row counts: ×2.705 [1] M=4, ×2.709 [2] M=4, ×2.924 [1] M=8, ×2.924 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.176 [1], ×1.173 [2]): a Session option, not a row
 *   - i4 g32 n10240 k2560 (case i4g32-n10240-k2560): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4))); parallel-l32: ×1.388 [1], ×1.387 [2]
 *   - i4 g32 n2560 k10240 (case i4g32-n2560-k10240): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.016 < ×1.100 in [1])); parallel-l32: ×1.016 [1], ×1.012 [2]
 *   - i4 g32 n2560 k256 (case i4g32-n2560-k256): parallel-l4 with row blocks of 2 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.131 [1] M=4, ×1.141 [2] M=4, ×1.196 [1] M=8, ×1.199 [2] M=8
 *   - i4 g32 n4096 k2560 (case i4g32-n4096-k2560): parallel-l4 with row blocks of 2 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.072 [1] M=4, ×1.081 [2] M=4, ×1.200 [1] M=8, ×1.203 [2] M=8
 *   - i4 g32 n1024 k2560 (case i4g32-n1024-k2560): parallel-l32 with row blocks of 2 -> parallel-l8 with row blocks of 4 (×1.310 [1], ×1.306 [2]); at the other row counts: ×1.853 [1] M=4, ×1.854 [2] M=4, ×2.049 [1] M=8, ×2.051 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.409 [1], ×1.409 [2]): a Session option, not a row
 *   - i4 g32 n2560 k4096 (case i4g32-n2560-k4096): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.028 < ×1.100 in [1])); parallel-l32: ×1.028 [1], ×1.028 [2]
 *   - i8 n262144 k2560 (case i8-n262144-k2560): parallel-l16 -> parallel-l16 with row blocks of 4; at the other row counts: ×1.614 [1] M=4, ×1.617 [2] M=4, ×1.730 [1] M=8, ×1.758 [2] M=8
 *   - i4 g32 n262144 k2560 (case i4g32-n262144-k2560): no row (c32u4) -> parallel-l16 with row blocks of 4 (×1.419 [1], ×1.419 [2]); at the other row counts: ×1.416 [1] M=4, ×1.419 [2] M=4, ×1.214 [1] M=8, ×1.221 [2] M=8
 *   - i4 g32 n256 k5120 (case i4g32-n256-k5120): no row (c32u4) -> parallel-l16 with row blocks of 4 (×10.209 [1], ×10.261 [2]); at the other row counts: ×2.813 [1] M=4, ×2.804 [2] M=4, ×1.803 [1] M=8, ×1.805 [2] M=8
 *     - subgroup32-l16 meets the rule too (×11.009 [1], ×11.299 [2]): a Session option, not a row
 *   - i4 g512 n2048 k2560 (case i4g512-n2048-k2560): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.512 [1] M=4, ×1.515 [2] M=4, ×1.599 [1] M=8, ×1.599 [2] M=8
 *   - i4 g512 n512 k2560 (case i4g512-n512-k2560): parallel-l32 -> staged-l16 with row blocks of 4 (×1.112 [1], ×1.116 [2]); at the other row counts: ×1.640 [1] M=4, ×1.639 [2] M=4, ×1.776 [1] M=8, ×1.777 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.233 [1], ×1.240 [2]): a Session option, not a row
 *   - i4 g2048 n2560 k2048 (case i4g2048-n2560-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l4 meets the rule too (×1.128 [1], ×1.128 [2]): a Session option, not a row
 *   - i4 g512 n10240 k2560 (case i4g512-n10240-k2560): parallel-l4 -> staged-l16 with row blocks of 4 (×1.315 [1], ×1.323 [2]); at the other row counts: ×1.339 [1] M=4, ×1.363 [2] M=4, ×1.350 [1] M=8, ×1.360 [2] M=8
 *   - i4 g2048 n2560 k10240 (case i4g2048-n2560-k10240): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k2560 (case i8-n256-k2560): parallel-l32 -> parallel-l32 with row blocks of 4; at the other row counts: ×1.231 [1] M=4, ×1.237 [2] M=4, ×1.273 [1] M=8, ×1.273 [2] M=8
 *     - subgroup32-l32 meets the rule too (×1.125 [1], ×1.100 [2]): a Session option, not a row
 *   - i8 n2560 k256 (case i8-n2560-k256): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.519 [1] M=4, ×1.521 [2] M=4, ×1.579 [1] M=8, ×1.579 [2] M=8
 *     - subgroup32-l4 meets the rule too (×1.122 [1], ×1.108 [2]): a Session option, not a row
 *   - i4 g512 n4096 k2560 (case i4g512-n4096-k2560): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.528 [1] M=4, ×1.526 [2] M=4, ×1.625 [1] M=8, ×1.625 [2] M=8
 *   - i4 g512 n1024 k2560 (case i4g512-n1024-k2560): parallel-l32 -> parallel-l8 with row blocks of 4 (×1.275 [1], ×1.276 [2]); at the other row counts: ×1.980 [1] M=4, ×1.975 [2] M=4, ×2.210 [1] M=8, ×2.210 [2] M=8
 *     - subgroup32-l16 meets the rule too (×1.394 [1], ×1.391 [2]): a Session option, not a row
 *   - i4 g4096 n2560 k4096 (case i4g4096-n2560-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k2560 (case i2-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: ×1.003 < ×1.100 in [1])); parallel-l2: ×1.003 [1], ×0.999 [2]
 *   - i4 g32 n15360 k3840 (case i4g32-n15360-k3840): parallel-l4 -> parallel-l16 with row blocks of 4 (×1.273 [1], ×1.274 [2]); at the other row counts: ×1.333 [1] M=4, ×1.323 [2] M=4, ×1.335 [1] M=8, ×1.321 [2] M=8
 *   - i4 g32 n8192 k3840 (case i4g32-n8192-k3840): parallel-l4 -> parallel-l16 with row blocks of 4 (×1.282 [1], ×1.282 [2]); at the other row counts: ×1.306 [1] M=4, ×1.311 [2] M=4, ×1.302 [1] M=8, ×1.312 [2] M=8
 *   - i4 g32 n4096 k3840 (case i4g32-n4096-k3840): parallel-l4 with row blocks of 2 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.037 [1] M=4, ×1.070 [2] M=4, ×1.172 [1] M=8, ×1.174 [2] M=8
 *   - i4 g32 n2048 k3840 (case i4g32-n2048-k3840): parallel-l4 with row blocks of 2 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.152 [1] M=4, ×1.143 [2] M=4, ×1.188 [1] M=8, ×1.185 [2] M=8
 *   - i4 g32 n512 k3840 (case i4g32-n512-k3840): keeps parallel-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.060 < ×1.100 in [1])); parallel-l8: ×1.060 [1], ×1.058 [2]
 *     - subgroup32-l32 meets the rule too (×1.122 [1], ×1.126 [2]): a Session option, not a row
 *   - i4 g32 n3840 k4096 (case i4g32-n3840-k4096): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.278 [1] M=4, ×1.280 [2] M=4, ×1.268 [1] M=8, ×1.271 [2] M=8
 *   - i4 g32 n3840 k8192 (case i4g32-n3840-k8192): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.002 < ×1.100 in [1])); parallel-l32: ×1.002 [1], ×0.995 [2]
 *   - i4 g32 n3840 k15360 (case i4g32-n3840-k15360): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.213 [1] M=4, ×1.213 [2] M=4, ×1.187 [1] M=8, ×1.196 [2] M=8
 *   - i4 g32 n262144 k3840 (case i4g32-n262144-k3840): parallel-l16 -> parallel-l16 with row blocks of 4; at the other row counts: ×1.540 [1] M=4, ×1.539 [2] M=4, ×1.591 [1] M=8, ×1.592 [2] M=8
 *   - i4 g32 n1024 k7680 (case i4g32-n1024-k7680): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.253 [1] M=4, ×1.255 [2] M=4, ×1.245 [1] M=8, ×1.245 [2] M=8
 *     - subgroup32-l32 meets the rule too (×1.118 [1], ×1.117 [2]): a Session option, not a row
 *   - i4 g32 n4096 k1024 (case i4g32-n4096-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4))); parallel-l16: ×1.406 [1], ×1.406 [2]
 *   - i4 g32 n1024 k4096 (case i4g32-n1024-k4096): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.263 [1] M=4, ×1.263 [2] M=4, ×1.258 [1] M=8, ×1.259 [2] M=8
 *     - subgroup32-l32 meets the rule too (×1.153 [1], ×1.149 [2]): a Session option, not a row
 *   - i4 g32 n8192 k1024 (case i4g32-n8192-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4))); parallel-l16: ×1.383 [1], ×1.383 [2]
 *   - i4 g32 n1024 k8192 (case i4g32-n1024-k8192): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.237 [1] M=4, ×1.237 [2] M=4, ×1.232 [1] M=8, ×1.233 [2] M=8
 *   - i4 g32 n3840 k1024 (case i4g32-n3840-k1024): no row (c32u4) -> parallel-l4 with row blocks of 2 (×1.440 [1], ×1.439 [2]); at the other row counts: ×1.308 [1] M=4, ×1.305 [2] M=4, ×1.320 [1] M=8, ×1.319 [2] M=8
 *     - subgroup32-l4 meets the rule too (×1.492 [1], ×1.494 [2]): a Session option, not a row
 *   - i4 g32 n262144 k1024 (case i4g32-n262144-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4))); staged-l16: ×1.342 [1], ×1.341 [2]
 *   - i4 g32 n8192 k2816 (case i4g32-n8192-k2816): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4))); parallel-l32: ×1.333 [1], ×1.348 [2]
 *   - i4 g32 n4096 k2816 (case i4g32-n4096-k2816): parallel-l4 with row blocks of 2 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.092 [1] M=4, ×1.089 [2] M=4, ×1.209 [1] M=8, ×1.211 [2] M=8
 *   - i4 g32 n2048 k2816 (case i4g32-n2048-k2816): parallel-l4 with row blocks of 2 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.148 [1] M=4, ×1.149 [2] M=4, ×1.192 [1] M=8, ×1.189 [2] M=8
 *   - i4 g32 n2112 k2816 (case i4g32-n2112-k2816): parallel-l4 with row blocks of 2 -> parallel-l4 with row blocks of 8; at the other row counts: ×1.203 [1] M=4, ×1.206 [2] M=4, ×1.235 [1] M=8, ×1.234 [2] M=8
 *   - i4 g32 n1024 k2816 (case i4g32-n1024-k2816): parallel-l32 -> parallel-l8 with row blocks of 4 (×1.245 [1], ×1.245 [2]); at the other row counts: ×1.887 [1] M=4, ×1.885 [2] M=4, ×2.089 [1] M=8, ×2.086 [2] M=8
 *     - subgroup32-l8 meets the rule too (×1.285 [1], ×1.276 [2]): a Session option, not a row
 *   - i4 g32 n128 k2816 (case i4g32-n128-k2816): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.911 < ×1.100 in [2])); parallel-l16: ×0.911 [2]
 *     - subgroup32-l32 meets the rule too (×1.184 [1], ×1.194 [2]): a Session option, not a row
 *   - i4 g32 n2816 k4096 (case i4g32-n2816-k4096): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.283 [1] M=4, ×1.281 [2] M=4, ×1.264 [1] M=8, ×1.262 [2] M=8
 *   - i4 g32 n2816 k8192 (case i4g32-n2816-k8192): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.249 [1] M=4, ×1.242 [2] M=4, ×1.236 [1] M=8, ×1.237 [2] M=8
 *   - i4 g32 n2816 k2112 (case i4g32-n2816-k2112): staged-l32 with row blocks of 2 -> parallel-l2 with row blocks of 4 (×1.327 [1], ×1.329 [2]); at the other row counts: ×2.249 [1] M=4, ×2.253 [2] M=4, ×2.554 [1] M=8, ×2.554 [2] M=8
 *     - subgroup32-l2 meets the rule too (×1.337 [1], ×1.337 [2]): a Session option, not a row
 *   - i4 g32 n262144 k2816 (case i4g32-n262144-k2816): parallel-l4 -> parallel-l16 with row blocks of 4 (×1.294 [1], ×1.287 [2]); at the other row counts: ×1.273 [1] M=4, ×1.271 [2] M=4, ×1.291 [1] M=8, ×1.295 [2] M=8
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): no row (c16u8) -> staged-l32 (×1.248 [1], ×1.243 [2]); at the other row counts: ×1.999 [1] M=4, ×1.999 [2] M=4, ×1.122 [1] M=8, ×1.126 [2] M=8
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: ×1.041 < ×1.100 in [1])); parallel-l4: ×1.041 [1], ×1.035 [2]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.024 < ×1.100 in [1])); parallel-l16: ×1.024 [1], ×0.981 [2]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: ×0.996 < ×1.100 in [1])); parallel-l4: ×0.996 [1], ×0.996 [2]
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×1.030 < ×1.100 in [1])); staged-l32: ×1.030 [1], ×1.029 [2]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.007 < ×1.100 in [1])); parallel-l8: ×1.007 [1], ×1.006 [2]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: ×1.089 < ×1.100 in [1])); parallel-l4: ×1.089 [1], ×1.047 [2]
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×0.997 < ×1.100 in [1])); staged-l32: ×0.997 [1], ×0.990 [2]
 *   - bf16 n512 k1536 (case bf16-n512-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×0.927 < ×1.100 in [1])); parallel-l8: ×0.927 [1], ×0.921 [2]
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.052 < ×1.100 in [1])); parallel-l8: ×1.052 [1], ×1.057 [2]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): no row (c16u8) -> staged-l32 (×1.251 [1], ×1.242 [2]); at the other row counts: ×2.004 [1] M=4, ×1.998 [2] M=4, ×1.098 [1] M=8, ×1.087 [2] M=8
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.023 < ×1.100 in [1])); parallel-l32: ×1.023 [1], ×1.016 [2]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): no row (c32u4) -> staged-l16 (×1.317 [1], ×1.303 [2]); at the other row counts: ×2.183 [1] M=4, ×2.173 [2] M=4, ×1.283 [1] M=8, ×1.254 [2] M=8
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): no row (c16u8) -> staged-l32 (×1.266 [1], ×1.255 [2]); at the other row counts: ×2.185 [1] M=4, ×2.174 [2] M=4, ×1.201 [1] M=8, ×1.187 [2] M=8
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.022 < ×1.100 in [1])); parallel-l16: ×1.022 [1], ×1.020 [2]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: ×1.059 < ×1.100 in [1])); parallel-l4: ×1.059 [1], ×1.055 [2]
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.931 < ×1.100 in [1])); parallel-l16: ×0.931 [1], ×0.921 [2]
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.992 < ×1.100 in [1])); parallel-l16: ×0.992 [1], ×0.997 [2]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): no row (c32u4) -> staged-l32 (×1.328 [1], ×1.327 [2]); at the other row counts: ×2.199 [1] M=4, ×2.199 [2] M=4, ×1.214 [1] M=8, ×1.213 [2] M=8
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.008 < ×1.100 in [1])); parallel-l32: ×1.008 [1], ×1.009 [2]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): keeps no row (c16u16) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: ×1.073 < ×1.100 in [1])); parallel-l4: ×1.073 [1], ×1.066 [2]
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×1.032 < ×1.100 in [1])); staged-l32: ×1.032 [1], ×1.032 [2]
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: ×1.032 < ×1.100 in [1])); parallel-l4: ×1.032 [1], ×1.035 [2]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×0.997 < ×1.100 in [1])); staged-l32: ×0.997 [1], ×1.000 [2]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): no row (c32u4) -> parallel-l16 (×1.314 [1], ×1.311 [2]); at the other row counts: ×2.423 [1] M=4, ×2.427 [2] M=4, ×1.508 [1] M=8, ×1.489 [2] M=8
 * - rowsThreadTarget: keeps 16384 (the full and row-blocks sets measure the row-block heights per case and row count, but the field is one thread target for every shape, and no rule turns the heights into it)
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
