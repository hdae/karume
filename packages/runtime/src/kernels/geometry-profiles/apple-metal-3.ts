/**
 * Geometry profile `apple-metal-3` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for the adapter match: vendor apple, architecture metal-3, description Apple M2.
 * The runtime only selects the table by the adapter's match; nothing is measured at run time.
 * To change a value, take a sweep with the workbench's `geometry-sweep` entry and regenerate the
 * table with `node tools/geometry-profiles/generate.ts apple-metal-3 --records=<path>[,<path>…]`.
 * The GEMV choices of linear (`linearGemv`) come from records of the `gemv-sweep` entry
 * (`--gemv-records=<path>,<path>[,…]`).
 *
 * Derived in this generation: the GEMV choices of linear (GEMV sweep records).
 *
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-18T23-15-21.json (sha256 05287881e296ba3073c037043a2ad790267da11fdb3e7f998cac74985f233a7f, 2026-11-18T23:15:21.277Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T02-48-31.json (sha256 b2d853d20991c6ed27790cbe825c5e27c7f3bc767bea83fd6891fb32b94a1929, 2026-11-19T02:48:31.513Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T03-12-40.json (sha256 88fc71575de3034263750e8a3baeb2d5f229cbeb05e1a38b3774a3ee216bd23a, 2026-11-19T03:12:40.044Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T03-25-41.json (sha256 6eda1bf8d52ba0f7f8d67758286c1b8170421319dda2bae31eedd45392bcf140, 2026-11-19T03:25:40.991Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T03-51-51.json (sha256 1bbbe932d7ef351298af20dc71d107883f08e907180db326bac7c5bbf9e61dfb, 2026-11-19T03:51:51.650Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T04-17-20.json (sha256 81c9513c8fdc5036625d6f7c93f61792d6275648baf62e01888a060b9826ccdb, 2026-11-19T04:17:20.696Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T06-12-50.json (sha256 3ea4e42ed215c3263f78fc9abd48727f671c745ad2ec3b9e76e64fff9e058589, 2026-11-19T06:12:50.159Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T06-24-14.json (sha256 a7210d217a8f383a83c674876a99db951ae81d71a137db2adbb8c5400c138eb9, 2026-11-19T06:24:14.620Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T06-35-19.json (sha256 85fd1d3f56bf905710813dbc939f29dbfac16cd7d1846aedf8bc32d232126642, 2026-11-19T06:35:18.894Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T06-52-24.json (sha256 6bb4ccae78e8b1ce50a582c21a25b2f773e31529b7ea828280b516f62e5c153a, 2026-11-19T06:52:24.809Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout 453921bcde14648096d7fd843c1a2d1759a84859
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T12-43-35.json (sha256 507e3eb42ec8ad3fb6f6542065a082f7fe1fd03de993c49f0a003e815df98764, 2026-11-19T12:43:35.046Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout b527f471d990d0f5b216b8aba20f2888749e86cd
 * - outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T12-58-40.json (sha256 3522eac3a10e06cece28590188f1c0bdba1af7f6f767d585e9c7da215fa44777, 2026-11-19T12:58:40.694Z)
 *   workbench; Chrome 157 on macOS; backend metal (declared); timestamps not quantised; checkout b527f471d990d0f5b216b8aba20f2888749e86cd
 *
 * Carried over from the registered table at commit 18a60a8b1236c21d87c772651c58972b854710eb: gemm, attention, conv2d, i8a8-linear, i8a8-attention.
 * Their adoption lines are in `git show 18a60a8b1236c21d87c772651c58972b854710eb:packages/runtime/src/kernels/geometry-profiles/apple-metal-3.ts`.
 *
 * GEMV choices: a candidate replaces the default profile's choice for a shape only when it is at
 * least ×1.100 faster than that choice in every GEMV sweep record, with its words equal to
 * the baseline's (identical class: the sequential variant's per-shape exceptions) or inside the
 * band (reorder class: the parallel rows, measured at every row count of the parallel range and
 * not slower at any) in every record; among those, the one whose smallest gain is largest. A shape
 * that a record measured while the baseline's rounds spread beyond ×1.250 keeps the default,
 * and so do the shared sequential keys whose cases disagree and the rows with packed activations.
 *
 * - linearGemv (the GEMV choices of linear), from 12 GEMV sweep records:
 *   - [1] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-18T23-15-21.json (2026-11-18T23:15:21.277Z, set full): 87 groups — 87 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [2] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T02-48-31.json (2026-11-19T02:48:31.513Z, set full): 87 groups — 87 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [3] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T03-12-40.json (2026-11-19T03:12:40.044Z, set full): 48 groups — 47 current and steady, 0 unsteady, 1 not current; 0 rows of current groups not current
 *     - 1 not current: the default profile's choices moved
 *   - [4] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T03-25-41.json (2026-11-19T03:25:40.991Z, set full): 48 groups — 48 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [5] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T03-51-51.json (2026-11-19T03:51:51.650Z, set full): 75 groups — 24 current and steady, 0 unsteady, 51 not current; 0 rows of current groups not current
 *     - 51 not current: the default profile's choices moved
 *   - [6] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T04-17-20.json (2026-11-19T04:17:20.696Z, set full): 87 groups — 87 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [7] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T06-12-50.json (2026-11-19T06:12:50.159Z, set full): 87 groups — 86 current and steady, 1 unsteady, 0 not current; 0 rows of current groups not current
 *   - [8] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T06-24-14.json (2026-11-19T06:24:14.620Z, set full): 48 groups — 46 current and steady, 1 unsteady, 1 not current; 0 rows of current groups not current
 *     - 1 not current: the default profile's choices moved
 *   - [9] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T06-35-19.json (2026-11-19T06:35:18.894Z, set full): 48 groups — 48 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [10] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T06-52-24.json (2026-11-19T06:52:24.809Z, set full): 75 groups — 24 current and steady, 0 unsteady, 51 not current; 0 rows of current groups not current
 *     - 51 not current: the default profile's choices moved
 *   - [11] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T12-43-35.json (2026-11-19T12:43:35.046Z, set full): 75 groups — 75 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [12] outputs/workbench/browser-apple-m2/gemv-sweep_2026-11-19T12-58-40.json (2026-11-19T12:58:40.694Z, set full): 75 groups — 75 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 * - identical class (sequential.shapes, M = 1): 81 adopted, 18 kept
 *   - i4 n8960 k1536 (case i4g32-n8960-k1536): c32u4 -> c128u4 (×1.247 [1], ×1.207 [6])
 *   - i4 n2048 k1536 (cases i4g32-n2048-k1536, i4g512-n2048-k1536): c32u4 -> c256u4 (×1.347 [1] i4g32-n2048-k1536, ×1.322 [6] i4g32-n2048-k1536, ×1.205 [4] i4g512-n2048-k1536, ×1.190 [9] i4g512-n2048-k1536)
 *   - i4 n256 k1536 (cases i4g32-n256-k1536, i4g512-n256-k1536): c32u4 -> c128u4 (×1.470 [1] i4g32-n256-k1536, ×1.472 [6] i4g32-n256-k1536, ×1.237 [4] i4g512-n256-k1536, ×1.227 [9] i4g512-n256-k1536)
 *   - i4 n1536 k2048 (cases i4g32-n1536-k2048, i4g2048-n1536-k2048): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n1536-k2048 adopts c128u4; i4g2048-n1536-k2048 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c256u4: ×1.082 < ×1.100 in [4])))
 *   - i4 n6144 k1536 (cases i4g32-n6144-k1536, i4g512-n6144-k1536): c32u4 -> c128u4 (×1.302 [1] i4g32-n6144-k1536, ×1.279 [6] i4g32-n6144-k1536, ×1.277 [4] i4g512-n6144-k1536, ×1.271 [9] i4g512-n6144-k1536)
 *   - i4 n1536 k6144 (cases i4g32-n1536-k6144, i4g2048-n1536-k6144): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n1536-k6144 adopts c256u4; i4g2048-n1536-k6144 adopts c128u4))
 *   - i4 n1536 k256 (case i4g32-n1536-k256): c32u4 -> c128u4 (×1.187 [1], ×1.185 [6])
 *   - i4 n4096 k1536 (cases i4g32-n4096-k1536, i4g512-n4096-k1536): c32u4 -> c128u4 (×1.366 [1] i4g32-n4096-k1536, ×1.413 [6] i4g32-n4096-k1536, ×1.189 [4] i4g512-n4096-k1536, ×1.384 [9] i4g512-n4096-k1536)
 *   - i4 n512 k1536 (cases i4g32-n512-k1536, i4g512-n512-k1536): c32u4 -> c128u4 (×1.531 [1] i4g32-n512-k1536, ×1.504 [6] i4g32-n512-k1536, ×1.268 [4] i4g512-n512-k1536, ×1.258 [9] i4g512-n512-k1536)
 *   - i4 n1536 k4096 (cases i4g32-n1536-k4096, i4g4096-n1536-k4096): c32u4 -> c128u4 (×1.312 [1] i4g32-n1536-k4096, ×1.303 [6] i4g32-n1536-k4096, ×1.233 [4] i4g4096-n1536-k4096, ×1.206 [9] i4g4096-n1536-k4096)
 *   - i4 n12288 k1536 (case i4g32-n12288-k1536): c32u4 -> c256u4 (×1.238 [1], ×1.243 [6])
 *   - i4 n1536 k12288 (case i4g32-n1536-k12288): c32u4 -> c128u4 (×1.397 [1], ×1.389 [6])
 *   - i8 n262144 k1536 (case i8-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not current in [3] (the default profile's choice at exact is c16u4 in the record, c32u4 now); M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not current in [8] (the default profile's choice at exact is c16u4 in the record, c32u4 now); M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12])))
 *   - i4 n262144 k1536 (case i4g32-n262144-k1536): c32u4 -> c128u4 (×1.126 [1], ×1.128 [6])
 *   - i4 n256 k3072 (case i4g32-n256-k3072): c32u4 -> c128u4 (×1.479 [1], ×1.498 [6])
 *   - i4 n1024 k256 (case i4g32-n1024-k256): c32u4 -> c128u4 (×1.454 [1], ×1.442 [6])
 *   - i4 n256 k1024 (case i4g32-n256-k1024): c32u4 -> c128u4 (×1.403 [1], ×1.436 [6])
 *   - i4 n2048 k256 (case i4g32-n2048-k256): c32u4 -> c256u4 (×1.310 [1], ×1.299 [6])
 *   - i4 n256 k2048 (case i4g32-n256-k2048): c32u4 -> c128u4 (×1.457 [1], ×1.490 [6])
 *   - i4 n262144 k256 (case i4g32-n262144-k256): c32u4 -> c128u4 (×1.158 [1], ×1.161 [6])
 *   - i8 n256 k3072 (case i8-n256-k3072): c32u4 -> c128u4 (×2.582 [3], ×2.563 [8])
 *   - i8 n1024 k256 (case i8-n1024-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u4: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; unsteady in [8] at M=1 (baseline spread ×1.313 > ×1.250); M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]))); c128u4: ×2.065 [3]
 *   - i8 n256 k1024 (case i8-n256-k1024): c32u4 -> c128u4 (×2.444 [3], ×2.436 [8])
 *   - i8 n2048 k256 (case i8-n2048-k256): c32u4 -> c256u4 (×1.793 [3], ×1.826 [8])
 *   - i8 n256 k2048 (case i8-n256-k2048): c32u4 -> c128u4 (×2.537 [3], ×2.608 [8])
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u4: ×1.092 < ×1.100 in [3])); c128u4: ×1.092 [3], ×1.097 [8]
 *   - i8 n256 k1536 (case i8-n256-k1536): c32u4 -> c128u4 (×2.559 [3], ×2.531 [8])
 *   - i8 n1536 k256 (case i8-n1536-k256): c32u4 -> c256u4 (×1.722 [3], ×1.756 [8])
 *   - i2 n12288 k1536 (case i2-n12288-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u4: ×1.072 < ×1.100 in [3])); c128u4: ×1.072 [3], ×1.099 [8]
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u4: ×1.053 < ×1.100 in [3])); c128u4: ×1.053 [3], ×1.083 [8]
 *   - i2 n262144 k1536 (case i2-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c256u4: ×1.047 < ×1.100 in [3])); c256u4: ×1.047 [3], ×1.047 [8]
 *   - i4 n10752 k2560 (case i4g32-n10752-k2560): c32u4 -> c128u4 (×1.162 [1], ×1.159 [6])
 *   - i4 n2048 k2560 (cases i4g32-n2048-k2560, i4g512-n2048-k2560): c32u4 -> c256u4 (×1.388 [1] i4g32-n2048-k2560, ×1.361 [6] i4g32-n2048-k2560, ×1.175 [4] i4g512-n2048-k2560, ×1.207 [9] i4g512-n2048-k2560)
 *   - i4 n512 k2560 (cases i4g32-n512-k2560, i4g512-n512-k2560): c32u4 -> c128u4 (×1.513 [1] i4g32-n512-k2560, ×1.497 [6] i4g32-n512-k2560, ×1.277 [4] i4g512-n512-k2560, ×1.284 [9] i4g512-n512-k2560)
 *   - i4 n256 k2560 (case i4g32-n256-k2560): c32u4 -> c128u4 (×1.469 [1], ×1.452 [6])
 *   - i4 n2560 k2048 (cases i4g32-n2560-k2048, i4g2048-n2560-k2048): c32u4 -> c128u4 (×1.191 [1] i4g32-n2560-k2048, ×1.224 [6] i4g32-n2560-k2048, ×1.146 [4] i4g2048-n2560-k2048, ×1.177 [9] i4g2048-n2560-k2048)
 *   - i4 n10240 k2560 (cases i4g32-n10240-k2560, i4g512-n10240-k2560): c32u4 -> c256u4 (×1.213 [1] i4g32-n10240-k2560, ×1.216 [6] i4g32-n10240-k2560, ×1.310 [4] i4g512-n10240-k2560, ×1.385 [9] i4g512-n10240-k2560)
 *   - i4 n2560 k10240 (cases i4g32-n2560-k10240, i4g2048-n2560-k10240): c32u4 -> c128u4 (×1.278 [1] i4g32-n2560-k10240, ×1.288 [6] i4g32-n2560-k10240, ×1.321 [4] i4g2048-n2560-k10240, ×1.333 [9] i4g2048-n2560-k10240)
 *   - i4 n2560 k256 (case i4g32-n2560-k256): c32u4 -> c128u4 (×1.179 [1], ×1.173 [6])
 *   - i4 n4096 k2560 (cases i4g32-n4096-k2560, i4g512-n4096-k2560): c32u4 -> c128u4 (×1.316 [1] i4g32-n4096-k2560, ×1.395 [6] i4g32-n4096-k2560, ×1.409 [4] i4g512-n4096-k2560, ×1.415 [9] i4g512-n4096-k2560)
 *   - i4 n1024 k2560 (cases i4g32-n1024-k2560, i4g512-n1024-k2560): c32u4 -> c128u4 (×1.577 [1] i4g32-n1024-k2560, ×1.570 [6] i4g32-n1024-k2560, ×1.344 [4] i4g512-n1024-k2560, ×1.314 [9] i4g512-n1024-k2560)
 *   - i4 n2560 k4096 (cases i4g32-n2560-k4096, i4g4096-n2560-k4096): c32u4 -> c128u4 (×1.211 [2] i4g32-n2560-k4096, ×1.208 [7] i4g32-n2560-k4096, ×1.171 [4] i4g4096-n2560-k4096, ×1.151 [9] i4g4096-n2560-k4096)
 *   - i8 n262144 k2560 (case i8-n262144-k2560): c32u4 -> c128u4 (×1.112 [3], ×1.112 [8])
 *   - i4 n262144 k2560 (case i4g32-n262144-k2560): c32u4 -> c128u8 (×1.122 [2], ×1.127 [7])
 *   - i4 n256 k5120 (case i4g32-n256-k5120): c32u4 -> c128u4 (×1.503 [2], ×1.488 [7])
 *   - i8 n256 k2560 (case i8-n256-k2560): c32u4 -> c128u4 (×2.544 [3], ×2.545 [8])
 *   - i8 n2560 k256 (case i8-n2560-k256): c32u4 -> c128u4 (×1.450 [3], ×1.435 [8])
 *   - i2 n262144 k2560 (case i2-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c256u4: ×1.046 < ×1.100 in [3])); c256u4: ×1.046 [3], ×1.044 [8]
 *   - i4 n15360 k3840 (case i4g32-n15360-k3840): c32u4 -> c128u4 (×1.244 [2], ×1.230 [7])
 *   - i4 n8192 k3840 (case i4g32-n8192-k3840): c32u4 -> c128u4 (×1.372 [2], ×1.392 [7])
 *   - i4 n4096 k3840 (case i4g32-n4096-k3840): c32u4 -> c128u4 (×1.430 [2], ×1.422 [7])
 *   - i4 n2048 k3840 (case i4g32-n2048-k3840): c32u4 -> c256u4 (×1.392 [2], ×1.393 [7])
 *   - i4 n512 k3840 (case i4g32-n512-k3840): c32u4 -> c128u4 (×1.530 [2], ×1.533 [7])
 *   - i4 n3840 k4096 (case i4g32-n3840-k4096): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c256u4: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; unsteady in [7] at M=1 (baseline spread ×1.276 > ×1.250); M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]))); c256u4: ×1.319 [2]
 *   - i4 n3840 k8192 (case i4g32-n3840-k8192): c32u4 -> c256u4 (×1.322 [2], ×1.310 [7])
 *   - i4 n3840 k15360 (case i4g32-n3840-k15360): c32u4 -> c256u4 (×1.222 [2], ×1.242 [7])
 *   - i4 n262144 k3840 (case i4g32-n262144-k3840): c32u4 -> c128u8 (×1.200 [2], ×1.171 [7])
 *   - i4 n1024 k7680 (case i4g32-n1024-k7680): c32u4 -> c128u4 (×1.515 [2], ×1.510 [7])
 *   - i4 n4096 k1024 (case i4g32-n4096-k1024): c32u4 -> c128u4 (×1.468 [2], ×1.412 [7])
 *   - i4 n1024 k4096 (case i4g32-n1024-k4096): c32u4 -> c128u4 (×1.497 [2], ×1.439 [7])
 *   - i4 n8192 k1024 (case i4g32-n8192-k1024): c32u4 -> c128u4 (×1.358 [2], ×1.360 [7])
 *   - i4 n1024 k8192 (case i4g32-n1024-k8192): c32u4 -> c128u4 (×1.463 [2], ×1.485 [7])
 *   - i4 n3840 k1024 (case i4g32-n3840-k1024): c32u4 -> c128u4 (×1.167 [2], ×1.177 [7])
 *   - i4 n262144 k1024 (case i4g32-n262144-k1024): c32u4 -> c128u4 (×1.124 [2], ×1.128 [7])
 *   - i4 n8192 k2816 (case i4g32-n8192-k2816): c32u4 -> c256u4 (×1.392 [2], ×1.369 [7])
 *   - i4 n4096 k2816 (case i4g32-n4096-k2816): c32u4 -> c128u4 (×1.473 [2], ×1.486 [7])
 *   - i4 n2048 k2816 (case i4g32-n2048-k2816): c32u4 -> c256u4 (×1.390 [2], ×1.417 [7])
 *   - i4 n2112 k2816 (case i4g32-n2112-k2816): c32u4 -> c256u4 (×1.271 [2], ×1.265 [7])
 *   - i4 n1024 k2816 (case i4g32-n1024-k2816): c32u4 -> c128u4 (×1.585 [2], ×1.587 [7])
 *   - i4 n128 k2816 (case i4g32-n128-k2816): c32u4 -> c128u4 (×1.503 [2], ×1.489 [7])
 *   - i4 n2816 k4096 (case i4g32-n2816-k4096): c32u4 -> c128u4 (×1.241 [2], ×1.228 [7])
 *   - i4 n2816 k8192 (case i4g32-n2816-k8192): c32u4 -> c128u4 (×1.287 [2], ×1.286 [7])
 *   - i4 n2816 k2112 (case i4g32-n2816-k2112): c32u4 -> c128u4 (×1.219 [2], ×1.223 [7])
 *   - i4 n262144 k2816 (case i4g32-n262144-k2816): c32u4 -> c128u8 (×1.243 [2], ×1.218 [7])
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): c16u8 -> c128u16 (×1.118 [5], ×1.124 [10], ×1.110 [11], ×1.108 [12])
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): c16u16 -> c256u8 (×1.169 [11], ×1.165 [12])
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u8: ×1.098 < ×1.100 in [11])); c128u8: ×1.098 [11], ×1.085 [12]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): c16u8 -> c256u8 (×1.164 [5], ×1.102 [10], ×1.148 [11], ×1.190 [12])
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u16: ×1.070 < ×1.100 in [11])); c128u16: ×1.070 [11], ×1.079 [12]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): c16u16 -> c256u8 (×1.220 [11], ×1.225 [12])
 *   - bf16 n1536 k256 (case bf16-n1536-k256): c32u16 -> c64u2 (×1.132 [11], ×1.131 [12])
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): c16u8 -> c128u16 (×1.133 [11], ×1.112 [12])
 *   - bf16 n512 k1536 (case bf16-n512-k1536): c16u16 -> c128u8 (×1.190 [11], ×1.189 [12])
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): c16u16 -> c256u8 (×1.277 [11], ×1.261 [12])
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c256u16: ×1.069 < ×1.100 in [5])); c256u16: ×1.069 [5], ×1.070 [10], ×1.069 [11], ×1.059 [12]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): c16u16 -> c128u8 (×1.197 [11], ×1.195 [12])
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.028 < ×1.100 in [5])); c16u16: ×1.028 [5], ×1.030 [10], ×1.029 [11], ×1.027 [12]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u16: ×1.056 < ×1.100 in [5])); c128u16: ×1.056 [5], ×1.046 [10], ×1.051 [11], ×1.048 [12]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c256u8: ×1.074 < ×1.100 in [12])); c256u8: ×1.131 [11], ×1.074 [12]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): c16u16 -> c128u8 (×1.377 [11], ×1.386 [12])
 *   - bf16 n256 k2560 (case bf16-n256-k2560): c16u16 -> c128u8 (×1.112 [11], ×1.117 [12])
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): c16u8 -> c128u16 (×1.257 [11], ×1.277 [12])
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u16: ×1.049 < ×1.100 in [5])); c128u16: ×1.049 [5], ×1.069 [10], ×1.037 [11], ×1.040 [12]
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): c16u8 -> c128u16 (×1.509 [11], ×1.503 [12])
 *   - bf16 n2560 k256 (case bf16-n2560-k256): c16u16 -> c32u2 (×1.545 [5], ×1.564 [10], ×1.544 [11], ×1.548 [12])
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): c16u8 -> c128u16 (×1.104 [11], ×1.105 [12])
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): c16u16 -> c256u8 (×1.443 [11], ×1.455 [12])
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): c16u8 -> c128u16 (×1.276 [11], ×1.264 [12])
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c256u16: ×1.047 < ×1.100 in [5])); c256u16: ×1.047 [5], ×1.041 [10], ×1.046 [11], ×1.028 [12]
 * - reorder class (parallelShapes, M = 1 to 8): 65 adopted, 50 kept
 *   - i4 g32 n8960 k1536 (case i4g32-n8960-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: ×0.987 < ×1.100 in [1])); parallel-l2: ×0.987 [1], ×0.991 [6]
 *   - i4 g32 n2048 k1536 (case i4g32-n2048-k1536): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×0.989 < ×1.100 in [1])); parallel-l8: ×0.989 [1], ×0.970 [6]
 *   - i4 g32 n256 k1536 (case i4g32-n256-k1536): parallel-l32 with row blocks of 2 -> parallel-l16 with row blocks of 4 (×1.594 [1], ×1.542 [6]); at the other row counts: ×1.869 [1] M=4, ×1.889 [6] M=4, ×1.844 [1] M=8, ×1.825 [6] M=8
 *     - subgroup32-l16 meets the rule too (×1.311 [1], ×1.307 [6]): a Session option, not a row
 *   - i4 g32 n1536 k2048 (case i4g32-n1536-k2048): staged-l32 -> parallel-l4 with row blocks of 2 (×1.676 [1], ×1.631 [6]); at the other row counts: ×2.298 [1] M=4, ×2.342 [6] M=4, ×2.494 [1] M=8, ×2.511 [6] M=8
 *     - subgroup32-l4 meets the rule too (×1.366 [1], ×1.341 [6]): a Session option, not a row
 *   - i4 g32 n6144 k1536 (case i4g32-n6144-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: ×0.986 < ×1.100 in [1])); parallel-l2: ×0.986 [1], ×0.995 [6]
 *   - i4 g32 n1536 k6144 (case i4g32-n1536-k6144): staged-l32 with row blocks of 2 -> parallel-l4 with row blocks of 2 (×1.393 [1], ×1.403 [6]); at the other row counts: ×1.498 [1] M=4, ×1.536 [6] M=4, ×1.608 [1] M=8, ×1.623 [6] M=8
 *   - i4 g32 n1536 k256 (case i4g32-n1536-k256): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.214 [1] M=4, ×1.191 [6] M=4, ×1.270 [1] M=8, ×1.262 [6] M=8
 *   - i4 g32 n4096 k1536 (case i4g32-n4096-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×0.942 < ×1.100 in [1])); parallel-l8: ×0.942 [1], ×0.938 [6]
 *   - i4 g32 n512 k1536 (case i4g32-n512-k1536): parallel-l32 -> parallel-l8 with row blocks of 4 (×1.699 [1], ×1.724 [6]); at the other row counts: ×2.532 [1] M=4, ×2.516 [6] M=4, ×2.481 [1] M=8, ×2.465 [6] M=8
 *     - subgroup32-l16 meets the rule too (×1.492 [1], ×1.494 [6]): a Session option, not a row
 *   - i4 g32 n1536 k4096 (case i4g32-n1536-k4096): staged-l32 -> parallel-l4 with row blocks of 2 (×1.433 [1], ×1.414 [6]); at the other row counts: ×2.133 [1] M=4, ×2.107 [6] M=4, ×2.200 [1] M=8, ×2.211 [6] M=8
 *     - subgroup32-l4 meets the rule too (×1.176 [1], ×1.187 [6]): a Session option, not a row
 *   - i4 g32 n12288 k1536 (case i4g32-n12288-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.006 < ×1.100 in [1])); parallel-l8: ×1.006 [1], ×1.011 [6]
 *   - i4 g32 n1536 k12288 (case i4g32-n1536-k12288): staged-l32 with row blocks of 2 -> parallel-l16 with row blocks of 2 (×1.236 [1], ×1.262 [6]); at the other row counts: ×1.403 [1] M=4, ×1.405 [6] M=4, ×1.433 [1] M=8, ×1.436 [6] M=8
 *   - i8 n262144 k1536 (case i8-n262144-k1536): parallel-l16 -> parallel-l16 with row blocks of 4; at the other row counts: ×1.198 [3] M=4, ×1.196 [8] M=4, ×1.225 [3] M=8, ×1.226 [8] M=8
 *   - i4 g32 n262144 k1536 (case i4g32-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; screened out by the quick look in [6] at M=4; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]))); parallel-l4: ×1.176 [1], ×1.177 [6]
 *   - i4 g32 n256 k3072 (case i4g32-n256-k3072): parallel-l32 -> parallel-l16 with row blocks of 4 (×1.402 [1], ×1.365 [6]); at the other row counts: ×2.014 [1] M=4, ×2.021 [6] M=4, ×2.039 [1] M=8, ×1.994 [6] M=8
 *     - subgroup32-l16 meets the rule too (×1.131 [1], ×1.134 [6]): a Session option, not a row
 *   - i4 g32 n1024 k256 (case i4g32-n1024-k256): no row (c32u4) -> parallel-l4 with row blocks of 4 (×1.662 [1], ×1.675 [6]); at the other row counts: ×1.322 [1] M=4, ×1.389 [6] M=4, ×1.398 [1] M=8, ×1.379 [6] M=8
 *   - i4 g32 n256 k1024 (case i4g32-n256-k1024): parallel-l32 -> parallel-l16 with row blocks of 4 (×1.388 [1], ×1.450 [6]); at the other row counts: ×1.858 [1] M=4, ×1.912 [6] M=4, ×1.848 [1] M=8, ×1.890 [6] M=8
 *     - subgroup32-l16 meets the rule too (×1.225 [1], ×1.252 [6]): a Session option, not a row
 *   - i4 g32 n2048 k256 (case i4g32-n2048-k256): no row (c32u4) -> parallel-l2 with row blocks of 4 (×1.349 [1], ×1.359 [6]); at the other row counts: ×1.672 [1] M=4, ×1.711 [6] M=4, ×1.550 [1] M=8, ×1.541 [6] M=8
 *     - subgroup32-l2 meets the rule too (×1.140 [1], ×1.141 [6]): a Session option, not a row
 *   - i4 g32 n256 k2048 (case i4g32-n256-k2048): no row (c32u4) -> parallel-l16 with row blocks of 4 (×5.601 [1], ×5.581 [6]); at the other row counts: ×2.826 [1] M=4, ×2.813 [6] M=4, ×1.678 [1] M=8, ×1.718 [6] M=8
 *     - subgroup32-l16 meets the rule too (×4.786 [1], ×4.732 [6]): a Session option, not a row
 *   - i4 g32 n262144 k256 (case i4g32-n262144-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; screened out by the quick look in [6] at M=4; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]))); parallel-l2: ×1.155 [1], ×1.158 [6]
 *   - i8 n256 k3072 (case i8-n256-k3072): no row (c32u4) -> parallel-l16 with row blocks of 8 (×11.178 [3], ×11.873 [8]); at the other row counts: ×5.411 [3] M=4, ×5.472 [8] M=4, ×3.425 [3] M=8, ×3.476 [8] M=8
 *     - subgroup32-l16 meets the rule too (×11.151 [3], ×11.168 [8]): a Session option, not a row
 *   - i8 n1024 k256 (case i8-n1024-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; unsteady in [8] at M=1 (baseline spread ×1.313 > ×1.250); M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]))); parallel-l4: ×2.749 [3]
 *   - i8 n256 k1024 (case i8-n256-k1024): no row (c32u4) -> parallel-l16 with row blocks of 8 (×8.142 [3], ×8.089 [8]); at the other row counts: ×4.085 [3] M=4, ×4.113 [8] M=4, ×2.567 [3] M=8, ×2.602 [8] M=8
 *     - subgroup32-l16 meets the rule too (×8.350 [3], ×8.394 [8]): a Session option, not a row
 *   - i8 n2048 k256 (case i8-n2048-k256): no row (c32u4) -> parallel-l2 with row blocks of 8 (×1.930 [3], ×1.924 [8]); at the other row counts: ×2.534 [3] M=4, ×2.478 [8] M=4, ×2.552 [3] M=8, ×2.580 [8] M=8
 *     - subgroup32-l4 meets the rule too (×1.815 [3], ×1.832 [8]): a Session option, not a row
 *   - i8 n256 k2048 (case i8-n256-k2048): no row (c32u4) -> parallel-l16 with row blocks of 8 (×10.176 [3], ×10.188 [8]); at the other row counts: ×4.898 [3] M=4, ×4.970 [8] M=4, ×3.164 [3] M=8, ×3.126 [8] M=8
 *     - subgroup32-l16 meets the rule too (×10.674 [3], ×10.350 [8]): a Session option, not a row
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; screened out by the quick look in [8] at M=4; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]))); parallel-l8: ×1.124 [3], ×1.127 [8]
 *   - i4 g512 n2048 k1536 (case i4g512-n2048-k1536): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.402 [4] M=4, ×1.428 [9] M=4, ×1.391 [4] M=8, ×1.404 [9] M=8
 *   - i4 g512 n256 k1536 (case i4g512-n256-k1536): parallel-l32 -> parallel-l16 with row blocks of 4 (×1.527 [4], ×1.459 [9]); at the other row counts: ×2.159 [4] M=4, ×2.140 [9] M=4, ×2.109 [4] M=8, ×2.084 [9] M=8
 *     - subgroup32-l16 meets the rule too (×1.274 [4], ×1.220 [9]): a Session option, not a row
 *   - i4 g2048 n1536 k2048 (case i4g2048-n1536-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l4 meets the rule too (×1.474 [4], ×1.459 [9]): a Session option, not a row
 *   - i4 g512 n6144 k1536 (case i4g512-n6144-k1536): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.454 [4] M=4, ×1.462 [9] M=4, ×1.489 [4] M=8, ×1.491 [9] M=8
 *   - i4 g2048 n1536 k6144 (case i4g2048-n1536-k6144): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l4 meets the rule too (×1.372 [4], ×1.326 [9]): a Session option, not a row
 *   - i8 n256 k1536 (case i8-n256-k1536): parallel-l32 -> parallel-l16 with row blocks of 8 (×1.277 [3], ×1.264 [8]); at the other row counts: ×1.576 [3] M=4, ×1.591 [8] M=4, ×1.769 [3] M=8, ×1.776 [8] M=8
 *     - subgroup32-l16 meets the rule too (×1.289 [3], ×1.286 [8]): a Session option, not a row
 *   - i8 n1536 k256 (case i8-n1536-k256): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.146 [3] M=4, ×1.156 [8] M=4, ×1.138 [3] M=8, ×1.176 [8] M=8
 *   - i4 g512 n4096 k1536 (case i4g512-n4096-k1536): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.453 [4] M=4, ×1.438 [9] M=4, ×1.490 [4] M=8, ×1.486 [9] M=8
 *   - i4 g512 n512 k1536 (case i4g512-n512-k1536): parallel-l32 -> parallel-l8 with row blocks of 4 (×1.755 [4], ×1.666 [9]); at the other row counts: ×2.540 [4] M=4, ×2.503 [9] M=4, ×2.522 [4] M=8, ×2.464 [9] M=8
 *   - i4 g4096 n1536 k4096 (case i4g4096-n1536-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l4 meets the rule too (×1.381 [4], ×1.368 [9]): a Session option, not a row
 *   - i2 n12288 k1536 (case i2-n12288-k1536): parallel-l2 -> parallel-l2 with row blocks of 2; at the other row counts: ×1.538 [3] M=4, ×1.532 [8] M=4, ×1.560 [3] M=8, ×1.557 [8] M=8
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l4 meets the rule too (×1.429 [3], ×1.443 [8]): a Session option, not a row
 *   - i2 n262144 k1536 (case i2-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: ×1.014 < ×1.100 in [3])); parallel-l2: ×1.014 [3], ×1.013 [8]
 *   - i4 g32 n10752 k2560 (case i4g32-n10752-k2560): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.059 < ×1.100 in [1])); parallel-l8: ×1.059 [1], ×1.066 [6]
 *   - i4 g32 n2048 k2560 (case i4g32-n2048-k2560): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.032 < ×1.100 in [1])); parallel-l8: ×1.032 [1], ×1.017 [6]
 *   - i4 g32 n512 k2560 (case i4g32-n512-k2560): parallel-l32 -> parallel-l8 with row blocks of 4 (×1.646 [1], ×1.606 [6]); at the other row counts: ×2.311 [1] M=4, ×2.328 [6] M=4, ×2.367 [1] M=8, ×2.309 [6] M=8
 *     - subgroup32-l16 meets the rule too (×1.353 [1], ×1.323 [6]): a Session option, not a row
 *   - i4 g32 n256 k2560 (case i4g32-n256-k2560): parallel-l32 -> parallel-l16 with row blocks of 4 (×1.393 [1], ×1.440 [6]); at the other row counts: ×2.031 [1] M=4, ×2.100 [6] M=4, ×2.009 [1] M=8, ×2.026 [6] M=8
 *     - subgroup32-l16 meets the rule too (×1.159 [1], ×1.195 [6]): a Session option, not a row
 *   - i4 g32 n2560 k2048 (case i4g32-n2560-k2048): staged-l32 -> parallel-l4 with row blocks of 2 (×1.711 [1], ×1.648 [6]); at the other row counts: ×2.517 [1] M=4, ×2.548 [6] M=4, ×2.603 [1] M=8, ×2.599 [6] M=8
 *     - subgroup32-l8 meets the rule too (×1.374 [1], ×1.380 [6]): a Session option, not a row
 *   - i4 g32 n10240 k2560 (case i4g32-n10240-k2560): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.059 < ×1.100 in [1])); parallel-l8: ×1.059 [1], ×1.059 [6]
 *   - i4 g32 n2560 k10240 (case i4g32-n2560-k10240): staged-l32 with row blocks of 2 -> parallel-l8 with row blocks of 2 (×1.252 [1], ×1.250 [6]); at the other row counts: ×1.506 [1] M=4, ×1.506 [6] M=4, ×1.557 [1] M=8, ×1.558 [6] M=8
 *   - i4 g32 n2560 k256 (case i4g32-n2560-k256): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: ×1.083 < ×1.100 in [1])); parallel-l2: ×1.083 [1], ×1.082 [6]
 *   - i4 g32 n4096 k2560 (case i4g32-n4096-k2560): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×0.958 < ×1.100 in [1])); parallel-l8: ×0.958 [1], ×0.970 [6]
 *   - i4 g32 n1024 k2560 (case i4g32-n1024-k2560): parallel-l32 with row blocks of 2 -> parallel-l4 with row blocks of 4 (×1.682 [1], ×1.762 [6]); at the other row counts: ×2.118 [1] M=4, ×2.094 [6] M=4, ×2.105 [1] M=8, ×2.128 [6] M=8
 *     - subgroup32-l8 meets the rule too (×1.511 [1], ×1.539 [6]): a Session option, not a row
 *   - i4 g32 n2560 k4096 (case i4g32-n2560-k4096): staged-l32 with row blocks of 2 -> parallel-l4 with row blocks of 2 (×1.476 [2], ×1.474 [7]); at the other row counts: ×1.758 [2] M=4, ×1.753 [7] M=4, ×1.798 [2] M=8, ×1.800 [7] M=8
 *     - subgroup32-l8 meets the rule too (×1.225 [2], ×1.217 [7]): a Session option, not a row
 *   - i8 n262144 k2560 (case i8-n262144-k2560): parallel-l16 -> parallel-l16 with row blocks of 4; at the other row counts: ×1.242 [3] M=4, ×1.241 [8] M=4, ×1.288 [3] M=8, ×1.288 [8] M=8
 *   - i4 g32 n262144 k2560 (case i4g32-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; screened out by the quick look in [2] at M=4; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; screened out by the quick look in [7] at M=4; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]))); parallel-l8: ×1.250 [2], ×1.253 [7]
 *   - i4 g32 n256 k5120 (case i4g32-n256-k5120): no row (c32u4) -> parallel-l16 with row blocks of 4 (×7.032 [2], ×6.985 [7]); at the other row counts: ×3.430 [2] M=4, ×3.372 [7] M=4, ×2.087 [2] M=8, ×2.084 [7] M=8
 *   - i4 g512 n2048 k2560 (case i4g512-n2048-k2560): parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×1.425 [4] M=4, ×1.404 [9] M=4, ×1.409 [4] M=8, ×1.408 [9] M=8
 *   - i4 g512 n512 k2560 (case i4g512-n512-k2560): parallel-l32 -> parallel-l8 with row blocks of 4 (×1.659 [4], ×1.647 [9]); at the other row counts: ×2.439 [4] M=4, ×2.344 [9] M=4, ×2.435 [4] M=8, ×2.391 [9] M=8
 *   - i4 g2048 n2560 k2048 (case i4g2048-n2560-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l8 meets the rule too (×1.415 [4], ×1.406 [9]): a Session option, not a row
 *   - i4 g512 n10240 k2560 (case i4g512-n10240-k2560): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.471 [4] M=4, ×1.473 [9] M=4, ×1.504 [4] M=8, ×1.504 [9] M=8
 *   - i4 g2048 n2560 k10240 (case i4g2048-n2560-k10240): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l8 meets the rule too (×1.310 [4], ×1.316 [9]): a Session option, not a row
 *   - i8 n256 k2560 (case i8-n256-k2560): parallel-l32 -> parallel-l16 with row blocks of 8 (×1.247 [3], ×1.237 [8]); at the other row counts: ×1.681 [3] M=4, ×1.688 [8] M=4, ×1.886 [3] M=8, ×1.869 [8] M=8
 *     - subgroup32-l16 meets the rule too (×1.259 [3], ×1.260 [8]): a Session option, not a row
 *   - i8 n2560 k256 (case i8-n2560-k256): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.146 [3] M=4, ×1.172 [8] M=4, ×1.199 [3] M=8, ×1.161 [8] M=8
 *   - i4 g512 n4096 k2560 (case i4g512-n4096-k2560): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.466 [4] M=4, ×1.461 [9] M=4, ×1.501 [4] M=8, ×1.495 [9] M=8
 *   - i4 g512 n1024 k2560 (case i4g512-n1024-k2560): parallel-l32 -> parallel-l4 with row blocks of 4 (×1.719 [4], ×1.727 [9]); at the other row counts: ×2.590 [4] M=4, ×2.539 [9] M=4, ×2.638 [4] M=8, ×2.640 [9] M=8
 *   - i4 g4096 n2560 k4096 (case i4g4096-n2560-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *     - subgroup32-l8 meets the rule too (×1.354 [4], ×1.399 [9]): a Session option, not a row
 *   - i2 n262144 k2560 (case i2-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l2: ×1.022 < ×1.100 in [3])); staged-l2: ×1.022 [3], ×1.020 [8]
 *   - i4 g32 n15360 k3840 (case i4g32-n15360-k3840): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.450 [2] M=4, ×1.451 [7] M=4, ×1.492 [2] M=8, ×1.492 [7] M=8
 *   - i4 g32 n8192 k3840 (case i4g32-n8192-k3840): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.433 [2] M=4, ×1.432 [7] M=4, ×1.485 [2] M=8, ×1.485 [7] M=8
 *   - i4 g32 n4096 k3840 (case i4g32-n4096-k3840): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.053 < ×1.100 in [2])); parallel-l8: ×1.053 [2], ×1.037 [7]
 *   - i4 g32 n2048 k3840 (case i4g32-n2048-k3840): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.011 < ×1.100 in [2])); parallel-l8: ×1.011 [2], ×1.020 [7]
 *   - i4 g32 n512 k3840 (case i4g32-n512-k3840): parallel-l32 with row blocks of 2 -> parallel-l8 with row blocks of 4 (×1.545 [2], ×1.531 [7]); at the other row counts: ×1.694 [2] M=4, ×1.720 [7] M=4, ×1.686 [2] M=8, ×1.702 [7] M=8
 *   - i4 g32 n3840 k4096 (case i4g32-n3840-k4096): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.296 [2] M=4, ×1.291 [7] M=4, ×1.298 [2] M=8, ×1.293 [7] M=8
 *   - i4 g32 n3840 k8192 (case i4g32-n3840-k8192): staged-l32 with row blocks of 2 -> parallel-l8 with row blocks of 2 (×1.318 [2], ×1.308 [7]); at the other row counts: ×1.569 [2] M=4, ×1.570 [7] M=4, ×1.609 [2] M=8, ×1.597 [7] M=8
 *   - i4 g32 n3840 k15360 (case i4g32-n3840-k15360): staged-l32 -> parallel-l8 with row blocks of 2 (×1.220 [2], ×1.211 [7]); at the other row counts: ×1.946 [2] M=4, ×1.946 [7] M=4, ×2.003 [2] M=8, ×2.006 [7] M=8
 *     - subgroup32-l8 meets the rule too (×1.131 [2], ×1.127 [7]): a Session option, not a row
 *   - i4 g32 n262144 k3840 (case i4g32-n262144-k3840): parallel-l16 -> parallel-l16 with row blocks of 2; at the other row counts: ×1.413 [2] M=4, ×1.412 [7] M=4, ×1.416 [2] M=8, ×1.416 [7] M=8
 *   - i4 g32 n1024 k7680 (case i4g32-n1024-k7680): staged-l32 -> parallel-l16 with row blocks of 2 (×1.463 [2], ×1.423 [7]); at the other row counts: ×1.952 [2] M=4, ×1.950 [7] M=4, ×1.984 [2] M=8, ×1.980 [7] M=8
 *     - subgroup32-l8 meets the rule too (×1.264 [2], ×1.284 [7]): a Session option, not a row
 *   - i4 g32 n4096 k1024 (case i4g32-n4096-k1024): no row (c32u4) -> parallel-l4 with row blocks of 2 (×1.449 [2], ×1.457 [7]); at the other row counts: ×1.693 [2] M=4, ×1.696 [7] M=4, ×1.307 [2] M=8, ×1.316 [7] M=8
 *   - i4 g32 n1024 k4096 (case i4g32-n1024-k4096): staged-l32 -> parallel-l4 with row blocks of 4 (×1.418 [2], ×1.427 [7]); at the other row counts: ×2.147 [2] M=4, ×2.144 [7] M=4, ×2.185 [2] M=8, ×2.202 [7] M=8
 *     - subgroup32-l8 meets the rule too (×1.258 [2], ×1.255 [7]): a Session option, not a row
 *   - i4 g32 n8192 k1024 (case i4g32-n8192-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: slower than the default's choice in [2] at M=4 (×0.927))); parallel-l4: ×1.385 [2], ×1.385 [7]
 *   - i4 g32 n1024 k8192 (case i4g32-n1024-k8192): staged-l32 -> parallel-l4 with row blocks of 4 (×1.363 [2], ×1.344 [7]); at the other row counts: ×2.010 [2] M=4, ×1.894 [7] M=4, ×2.053 [2] M=8, ×1.998 [7] M=8
 *     - subgroup32-l8 meets the rule too (×1.213 [2], ×1.210 [7]): a Session option, not a row
 *   - i4 g32 n3840 k1024 (case i4g32-n3840-k1024): no row (c32u4) -> parallel-l4 with row blocks of 2 (×1.171 [2], ×1.193 [7]); at the other row counts: ×1.597 [2] M=4, ×1.602 [7] M=4, ×1.600 [2] M=8, ×1.607 [7] M=8
 *   - i4 g32 n262144 k1024 (case i4g32-n262144-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; screened out by the quick look in [2] at M=4; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; screened out by the quick look in [7] at M=4; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]))); parallel-l2: ×1.146 [2], ×1.149 [7]
 *   - i4 g32 n8192 k2816 (case i4g32-n8192-k2816): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.073 < ×1.100 in [2])); parallel-l8: ×1.073 [2], ×1.075 [7]
 *   - i4 g32 n4096 k2816 (case i4g32-n4096-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×0.991 < ×1.100 in [2])); parallel-l8: ×0.991 [2], ×0.988 [7]
 *   - i4 g32 n2048 k2816 (case i4g32-n2048-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.034 < ×1.100 in [2])); parallel-l8: ×1.034 [2], ×1.006 [7]
 *   - i4 g32 n2112 k2816 (case i4g32-n2112-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.029 < ×1.100 in [2])); parallel-l8: ×1.029 [2], ×1.017 [7]
 *   - i4 g32 n1024 k2816 (case i4g32-n1024-k2816): parallel-l32 -> parallel-l4 with row blocks of 4 (×1.688 [2], ×1.656 [7]); at the other row counts: ×2.538 [2] M=4, ×2.551 [7] M=4, ×2.591 [2] M=8, ×2.568 [7] M=8
 *     - subgroup32-l8 meets the rule too (×1.479 [2], ×1.446 [7]): a Session option, not a row
 *   - i4 g32 n128 k2816 (case i4g32-n128-k2816): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.135 [2] M=4, ×1.136 [7] M=4, ×1.227 [2] M=8, ×1.258 [7] M=8
 *   - i4 g32 n2816 k4096 (case i4g32-n2816-k4096): staged-l32 -> parallel-l8 with row blocks of 2 (×1.490 [2], ×1.504 [7]); at the other row counts: ×2.191 [2] M=4, ×2.157 [7] M=4, ×2.243 [2] M=8, ×2.213 [7] M=8
 *     - subgroup32-l8 meets the rule too (×1.242 [2], ×1.243 [7]): a Session option, not a row
 *   - i4 g32 n2816 k8192 (case i4g32-n2816-k8192): staged-l32 -> parallel-l8 with row blocks of 2 (×1.343 [2], ×1.331 [7]); at the other row counts: ×2.016 [2] M=4, ×2.026 [7] M=4, ×2.108 [2] M=8, ×2.094 [7] M=8
 *     - subgroup32-l8 meets the rule too (×1.172 [2], ×1.171 [7]): a Session option, not a row
 *   - i4 g32 n2816 k2112 (case i4g32-n2816-k2112): staged-l32 with row blocks of 2 -> parallel-l4 with row blocks of 2 (×2.048 [2], ×2.021 [7]); at the other row counts: ×2.374 [2] M=4, ×2.373 [7] M=4, ×2.517 [2] M=8, ×2.499 [7] M=8
 *     - subgroup32-l2 meets the rule too (×1.723 [2], ×1.757 [7]): a Session option, not a row
 *   - i4 g32 n262144 k2816 (case i4g32-n262144-k2816): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.463 [2] M=4, ×1.459 [7] M=4, ×1.493 [2] M=8, ×1.490 [7] M=8
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): no row (c16u8) -> parallel-l8 (×1.162 [5], ×1.165 [10], ×1.148 [11], ×1.150 [12]); at the other row counts: ×4.164 [5] M=4, ×4.168 [10] M=4, ×4.173 [11] M=4, ×4.169 [12] M=4, ×2.683 [5] M=8, ×2.683 [10] M=8, ×2.686 [11] M=8, ×2.696 [12] M=8
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: ×0.992 < ×1.100 in [11])); parallel-l2: ×0.992 [11], ×0.996 [12]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): parallel-l32 -> parallel-l16 (×1.270 [11], ×1.192 [12]); at the other row counts: ×1.313 [11] M=4, ×1.255 [12] M=4, ×1.341 [11] M=8, ×1.335 [12] M=8
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): no row (c16u8) -> parallel-l4 (×1.162 [5], ×1.110 [10], ×1.130 [11], ×1.228 [12]); at the other row counts: ×5.682 [5] M=4, ×5.886 [10] M=4, ×5.686 [11] M=4, ×5.568 [12] M=4, ×3.321 [5] M=8, ×3.329 [10] M=8, ×3.304 [11] M=8, ×3.277 [12] M=8
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×0.986 < ×1.100 in [11])); parallel-l32: ×0.986 [11], ×0.984 [12]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: ×1.021 < ×1.100 in [11])); staged-l8: ×1.021 [11], ×1.011 [12]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): parallel-l8 -> parallel-l4 (×1.134 [11], ×1.143 [12]); at the other row counts: ×1.272 [11] M=4, ×1.310 [12] M=4, ×1.305 [11] M=8, ×1.300 [12] M=8
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×0.987 < ×1.100 in [11])); parallel-l32: ×0.987 [11], ×0.979 [12]
 *   - bf16 n512 k1536 (case bf16-n512-k1536): parallel-l16 -> parallel-l8 (×1.109 [11], ×1.109 [12]); at the other row counts: ×1.177 [11] M=4, ×1.199 [12] M=4, ×1.268 [11] M=8, ×1.301 [12] M=8
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: ×1.030 < ×1.100 in [11])); staged-l8: ×1.030 [11], ×1.018 [12]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.097 < ×1.100 in [12])); parallel-l8: ×1.107 [5], ×1.102 [10], ×1.101 [11], ×1.097 [12]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: ×1.009 < ×1.100 in [11])); staged-l8: ×1.009 [11], ×1.021 [12]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: ×1.056 < ×1.100 in [5])); staged-l8: ×1.056 [5], ×1.057 [10], ×1.056 [11], ×1.053 [12]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.099 < ×1.100 in [10])); parallel-l8: ×1.112 [5], ×1.099 [10], ×1.101 [11], ×1.099 [12]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.010 < ×1.100 in [11])); parallel-l32: ×1.010 [11], ×1.003 [12]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.021 < ×1.100 in [11])); parallel-l8: ×1.021 [11], ×1.032 [12]
 *   - bf16 n256 k2560 (case bf16-n256-k2560): parallel-l32 -> parallel-l16 (×1.254 [11], ×1.249 [12]); at the other row counts: ×1.330 [11] M=4, ×1.332 [12] M=4, ×1.345 [11] M=8, ×1.348 [12] M=8
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.999 < ×1.100 in [11])); parallel-l16: ×0.999 [11], ×0.993 [12]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.086 < ×1.100 in [5])); parallel-l8: ×1.086 [5], ×1.111 [10], ×1.071 [11], ×1.081 [12]
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.011 < ×1.100 in [11])); parallel-l32: ×1.011 [11], ×1.009 [12]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): no row (c16u16) -> parallel-l4 (×2.143 [5], ×2.207 [10], ×2.091 [11], ×2.105 [12]); at the other row counts: ×6.318 [5] M=4, ×6.251 [10] M=4, ×6.202 [11] M=4, ×6.200 [12] M=4, ×3.508 [5] M=8, ×3.531 [10] M=8, ×3.497 [11] M=8, ×3.459 [12] M=8
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.002 < ×1.100 in [11])); parallel-l32: ×1.002 [11], ×0.995 [12]
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.015 < ×1.100 in [11])); parallel-l8: ×1.015 [11], ×0.996 [12]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×0.997 < ×1.100 in [11])); parallel-l8: ×0.997 [11], ×1.011 [12]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.085 < ×1.100 in [5])); parallel-l8: ×1.085 [5], ×1.077 [10], ×1.082 [11], ×1.065 [12]
 * - rowsThreadTarget: keeps 16384 (the full and row-blocks sets measure the row-block heights per case and row count, but the field is one thread target for every shape, and no rule turns the heights into it)
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const APPLE_METAL_3: GeometryProfile = {
  id: "apple-metal-3",
  match: { vendor: "apple", architecture: "metal-3", description: "Apple M2" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 2, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 64, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 4, wgX: 8, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 4, wgX: 8, wgY: 16 },
    pv: { regM: 4, regN: 8, wgX: 8, wgY: 16 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 4, wgX: 16, wgY: 16 },
    rows32: { regM: 8, regN: 4, wgX: 16, wgY: 16 },
  },
  i8a8: {
    linear: [
      { maxRows: 256, geometry: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 } },
      { maxRows: 512, geometry: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 } },
      {
        maxRows: Number.POSITIVE_INFINITY,
        geometry: { regM: 4, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
      },
    ],
    attentionQk: { regM: 4, regN: 8, wgX: 8, wgY: 8, tileK: 16 },
    attentionPv: { regM: 4, regN: 8, wgX: 16, wgY: 4, tileK: 16 },
    conv3d: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
  },
  linearGemv: {
    sequential: {
      variant: { cols: 32, unroll: 4 },
      shapes: [
        { storage: "bf16", n: 8960, k: 1536, variant: { cols: 128, unroll: 16 } },
        { storage: "bf16", n: 2048, k: 1536, variant: { cols: 256, unroll: 8 } },
        { storage: "bf16", n: 256, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 1536, k: 2048, variant: { cols: 256, unroll: 8 } },
        { storage: "bf16", n: 6144, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 1536, k: 6144, variant: { cols: 256, unroll: 8 } },
        { storage: "bf16", n: 1536, k: 256, variant: { cols: 64, unroll: 2 } },
        { storage: "bf16", n: 4096, k: 1536, variant: { cols: 128, unroll: 16 } },
        { storage: "bf16", n: 512, k: 1536, variant: { cols: 128, unroll: 8 } },
        { storage: "bf16", n: 1536, k: 4096, variant: { cols: 256, unroll: 8 } },
        { storage: "bf16", n: 12288, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 1536, k: 12288, variant: { cols: 128, unroll: 8 } },
        { storage: "bf16", n: 10752, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 2048, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 512, k: 2560, variant: { cols: 128, unroll: 8 } },
        { storage: "bf16", n: 256, k: 2560, variant: { cols: 128, unroll: 8 } },
        { storage: "bf16", n: 2560, k: 2048, variant: { cols: 128, unroll: 16 } },
        { storage: "bf16", n: 2560, k: 10240, variant: { cols: 128, unroll: 16 } },
        { storage: "bf16", n: 2560, k: 256, variant: { cols: 32, unroll: 2 } },
        { storage: "bf16", n: 4096, k: 2560, variant: { cols: 128, unroll: 16 } },
        { storage: "bf16", n: 1024, k: 2560, variant: { cols: 256, unroll: 8 } },
        { storage: "bf16", n: 2560, k: 4096, variant: { cols: 128, unroll: 16 } },
        { storage: "i4", n: 8960, k: 1536, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2048, k: 1536, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 256, k: 1536, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 6144, k: 1536, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 1536, k: 256, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 4096, k: 1536, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 512, k: 1536, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 1536, k: 4096, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 12288, k: 1536, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 1536, k: 12288, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 262144, k: 1536, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 256, k: 3072, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 1024, k: 256, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 256, k: 1024, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2048, k: 256, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 256, k: 2048, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 262144, k: 256, variant: { cols: 128, unroll: 4 } },
        { storage: "i8", n: 256, k: 3072, variant: { cols: 128, unroll: 4 } },
        { storage: "i8", n: 256, k: 1024, variant: { cols: 128, unroll: 4 } },
        { storage: "i8", n: 2048, k: 256, variant: { cols: 256, unroll: 4 } },
        { storage: "i8", n: 256, k: 2048, variant: { cols: 128, unroll: 4 } },
        { storage: "i8", n: 256, k: 1536, variant: { cols: 128, unroll: 4 } },
        { storage: "i8", n: 1536, k: 256, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 10752, k: 2560, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2048, k: 2560, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 512, k: 2560, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 256, k: 2560, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2560, k: 2048, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 10240, k: 2560, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 2560, k: 10240, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2560, k: 256, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 4096, k: 2560, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 1024, k: 2560, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2560, k: 4096, variant: { cols: 128, unroll: 4 } },
        { storage: "i8", n: 262144, k: 2560, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 262144, k: 2560, variant: { cols: 128, unroll: 8 } },
        { storage: "i4", n: 256, k: 5120, variant: { cols: 128, unroll: 4 } },
        { storage: "i8", n: 256, k: 2560, variant: { cols: 128, unroll: 4 } },
        { storage: "i8", n: 2560, k: 256, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 15360, k: 3840, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 8192, k: 3840, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 4096, k: 3840, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2048, k: 3840, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 512, k: 3840, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 3840, k: 8192, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 3840, k: 15360, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 262144, k: 3840, variant: { cols: 128, unroll: 8 } },
        { storage: "i4", n: 1024, k: 7680, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 4096, k: 1024, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 1024, k: 4096, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 8192, k: 1024, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 1024, k: 8192, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 3840, k: 1024, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 262144, k: 1024, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 8192, k: 2816, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 4096, k: 2816, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2048, k: 2816, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 2112, k: 2816, variant: { cols: 256, unroll: 4 } },
        { storage: "i4", n: 1024, k: 2816, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 128, k: 2816, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2816, k: 4096, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2816, k: 8192, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 2816, k: 2112, variant: { cols: 128, unroll: 4 } },
        { storage: "i4", n: 262144, k: 2816, variant: { cols: 128, unroll: 8 } },
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
        rowBlock: 2,
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
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
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
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1536,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
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
        lanes: 4,
        packedActivations: false,
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
        lanes: 16,
        packedActivations: false,
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
        rowBlock: 2,
      },
      { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 1536, lanes: 16, packedActivations: false, rowBlock: 8 },
      { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false, rowBlock: 2 },
      {
        storage: "i4",
        n: 4096,
        k: 1536,
        group: 512,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 512,
        k: 1536,
        group: 512,
        lanes: 8,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      { storage: "i2", n: 12288, k: 1536, lanes: 2, packedActivations: false, rowBlock: 2 },
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
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 512,
        k: 2560,
        group: 32,
        lanes: 8,
        packedActivations: false,
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
        rowBlock: 2,
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
        lanes: 8,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2560,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 4096,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2560,
        k: 4096,
        group: 32,
        lanes: 4,
        packedActivations: false,
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
        lanes: 8,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "i4", n: 2560, k: 2048, group: 2048, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 10240,
        k: 2560,
        group: 512,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 2560, lanes: 16, packedActivations: false, rowBlock: 8 },
      { storage: "i8", n: 2560, k: 256, lanes: 4, packedActivations: false, rowBlock: 2 },
      {
        storage: "i4",
        n: 4096,
        k: 2560,
        group: 512,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2560,
        group: 512,
        lanes: 4,
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
      {
        storage: "i4",
        n: 256,
        k: 1024,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 15360,
        k: 3840,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 8192,
        k: 3840,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 4096,
        k: 3840,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2048,
        k: 3840,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 512,
        k: 3840,
        group: 32,
        lanes: 8,
        packedActivations: false,
        rowBlock: 4,
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
        lanes: 8,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 15360,
        group: 32,
        lanes: 8,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 262144,
        k: 3840,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 7680,
        group: 32,
        lanes: 16,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 4096,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1024,
        k: 8192,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
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
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2048,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2112,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 128,
        k: 2816,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2816,
        k: 4096,
        group: 32,
        lanes: 8,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2816,
        k: 8192,
        group: 32,
        lanes: 8,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2816,
        k: 2112,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 262144,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "bf16", n: 2048, k: 1536, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 256, k: 1536, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 6144, k: 1536, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 1536, k: 6144, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 1536, k: 256, lanes: 4, packedActivations: false },
      { storage: "bf16", n: 4096, k: 1536, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 512, k: 1536, lanes: 8, packedActivations: false },
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
      { storage: "bf16", n: 256, k: 2560, lanes: 16, packedActivations: false },
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
        lanes: 2,
        packedActivations: false,
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
      { storage: "i8", n: 256, k: 3072, lanes: 16, packedActivations: false, rowBlock: 8 },
      { storage: "i8", n: 256, k: 1024, lanes: 16, packedActivations: false, rowBlock: 8 },
      { storage: "i8", n: 2048, k: 256, lanes: 2, packedActivations: false, rowBlock: 8 },
      { storage: "i8", n: 256, k: 2048, lanes: 16, packedActivations: false, rowBlock: 8 },
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
        n: 4096,
        k: 1024,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
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
      { storage: "bf16", n: 8960, k: 1536, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 1536, k: 2048, lanes: 4, packedActivations: false },
      { storage: "bf16", n: 2560, k: 256, lanes: 4, packedActivations: false },
    ],
  },
  provenance: {
    adapter: {
      vendor: "apple",
      architecture: "metal-3",
      device: "0x0000",
      description: "Apple M2",
    },
    families: {
      gemm: {
        sweep:
          "outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T13-08-11.329Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json, outputs/bench-browser/geometry-sweep-browser-2026-10-01T12-28-47.052Z.json",
        sha256:
          "921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034, f464ca82eb61f4c8fd60efa06365815eb25cec5e1f8ed905cccd242cadacfc2a, 38d97dac5f04a7e80fb6591ab437da7dd2b3cee66b0d03e40cd1475c0f398550, d9e5e7814370c741562819474f5e0837012d05f233869688d28bb91902ac8621, 11fed6a888c91d5e5217a8cd1205dbd46b5ffa2c639dcf2f2af9d17be962eefd",
        date: "2026-09-27T16:28:00.787Z, 2026-09-27T18:31:46.471Z, 2026-09-29T13:08:11.329Z, 2026-09-29T21:04:14.586Z, 2026-10-01T12:28:47.052Z",
        candidateSet: "quick, full, full, full, full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
        ],
        kernels: "fbc7d11de06482c4",
        caseSet: "e6ddd45226ab2155",
      },
      attention: {
        sweep:
          "outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json, outputs/bench-browser/geometry-sweep-browser-2026-10-01T12-28-47.052Z.json",
        sha256:
          "921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034, f464ca82eb61f4c8fd60efa06365815eb25cec5e1f8ed905cccd242cadacfc2a, d9e5e7814370c741562819474f5e0837012d05f233869688d28bb91902ac8621, 11fed6a888c91d5e5217a8cd1205dbd46b5ffa2c639dcf2f2af9d17be962eefd",
        date: "2026-09-27T16:28:00.787Z, 2026-09-27T18:31:46.471Z, 2026-09-29T21:04:14.586Z, 2026-10-01T12:28:47.052Z",
        candidateSet: "quick, full, full, full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
        ],
        kernels: "7c23ff15792d67ec",
        caseSet: "9ee211cf69674928",
      },
      conv2d: {
        sweep:
          "outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json, outputs/bench-browser/geometry-sweep-browser-2026-10-01T12-28-47.052Z.json",
        sha256:
          "921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034, f464ca82eb61f4c8fd60efa06365815eb25cec5e1f8ed905cccd242cadacfc2a, d9e5e7814370c741562819474f5e0837012d05f233869688d28bb91902ac8621, 11fed6a888c91d5e5217a8cd1205dbd46b5ffa2c639dcf2f2af9d17be962eefd",
        date: "2026-09-27T16:28:00.787Z, 2026-09-27T18:31:46.471Z, 2026-09-29T21:04:14.586Z, 2026-10-01T12:28:47.052Z",
        candidateSet: "quick, full, full, full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
        ],
        kernels: "1e1bddcc0638175e",
        caseSet: "02186e45558e4151",
      },
      "i8a8-linear": {
        sweep:
          "outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json, outputs/bench-browser/geometry-sweep-browser-2026-10-01T12-28-47.052Z.json",
        sha256:
          "921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034, d9e5e7814370c741562819474f5e0837012d05f233869688d28bb91902ac8621, 11fed6a888c91d5e5217a8cd1205dbd46b5ffa2c639dcf2f2af9d17be962eefd",
        date: "2026-09-27T16:28:00.787Z, 2026-09-29T21:04:14.586Z, 2026-10-01T12:28:47.052Z",
        candidateSet: "quick, full, full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
        ],
        kernels: "9cc5917c8287af47",
        caseSet: "3d32dc6f10559af2",
      },
      "i8a8-attention": {
        sweep:
          "outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json, outputs/bench-browser/geometry-sweep-browser-2026-10-01T12-28-47.052Z.json",
        sha256:
          "921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034, d9e5e7814370c741562819474f5e0837012d05f233869688d28bb91902ac8621, 11fed6a888c91d5e5217a8cd1205dbd46b5ffa2c639dcf2f2af9d17be962eefd",
        date: "2026-09-27T16:28:00.787Z, 2026-09-29T21:04:14.586Z, 2026-10-01T12:28:47.052Z",
        candidateSet: "quick, full, full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
        ],
        kernels: "5544095375968b39",
        caseSet: "8ab1a86b13087640",
      },
    },
  },
};
