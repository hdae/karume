/**
 * Geometry profile `nvidia-ampere` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for the adapter match: vendor nvidia, architecture ampere.
 * The runtime only selects the table by the adapter's match; nothing is measured at run time.
 * To change a value, take a sweep with the workbench's `geometry-sweep` entry and regenerate the
 * table with `node tools/geometry-profiles/generate.ts nvidia-ampere --records=<path>[,<path>…]`.
 * The GEMV choices of linear (`linearGemv`) come from records of the `gemv-sweep` entry
 * (`--gemv-records=<path>,<path>[,…]`).
 *
 * Derived in this generation: the GEMV choices of linear (GEMV sweep records).
 *
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T03-35-23.json (sha256 53b238e84b5455719fca532af1d9a9e5c7be41a6153b237ca9e9513e7ce86cc4, 2026-11-18T03:35:23.669Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 7834560b50cc76b7d47e2ae91ad4b25642b200f6 (dirty)
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T03-41-44.json (sha256 955ab6421e73819a343f108d3b52082837bc9aa23345c394c375e964a3941577, 2026-11-18T03:41:44.165Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout a9daa29134a122518cb2cb39eb293a2413b6f0ef
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T04-12-00.json (sha256 c09f717f43651218a17cb22e167026d5913ceffdf3cc1322da4f1b13a170b2ce, 2026-11-18T04:12:00.301Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 7834560b50cc76b7d47e2ae91ad4b25642b200f6 (dirty)
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T04-39-54.json (sha256 5af2fa6ddb94820eb1131d99affbd9ad14a3cbfa1b1b24f52bf6bde266be5e85, 2026-11-18T04:39:54.559Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 7834560b50cc76b7d47e2ae91ad4b25642b200f6 (dirty)
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T04-56-22.json (sha256 0c847e75412f43c4f0d4d8a4f33687c870580f833bb543538186a7c0a71e1d69, 2026-11-18T04:56:22.175Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout a9daa29134a122518cb2cb39eb293a2413b6f0ef (dirty)
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-32-42.json (sha256 ff5ff89a044d6ce413b5a03e1e0056947d049fd4bf625a68cfb538a0862b31cd, 2026-11-18T11:32:42.796Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-36-30.json (sha256 c44e7275b49fbdc07824bae5edb2c308ee03352c643fc68c093d3937b2b40d21, 2026-11-18T11:36:30.026Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-41-59.json (sha256 99111aed549909648caf4a7c6c8eb1b7fc05cbe358a52daabae759b0d1cd73b7, 2026-11-18T11:41:59.472Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-43-05.json (sha256 ba4744d59cb1a5ead70c56158fea8ed8e624966118592523bfe46b33939fa1d7, 2026-11-18T11:43:05.129Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-46-29.json (sha256 d0049ceb945dc0144df911c4624a89dcb8113595001a043d5ab801c1f5a168f7, 2026-11-18T11:46:29.709Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-49-44.json (sha256 6156119f534df8ac0d3bd32bf050a12d64ff3c0550e898c4ea714f4e5fea368d, 2026-11-18T11:49:44.127Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-53-13.json (sha256 b7834871edd090bde302ba923817eac7c742199419d59ddc85022d2bc18c47c5, 2026-11-18T11:53:13.036Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-56-28.json (sha256 1d562068bd1df811426b936f73056378f8968e49a494ceb246abf9b2ab17328a, 2026-11-18T11:56:28.481Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-59-54.json (sha256 9a8d0ef60ae8a3d163c46d89eacce59fc2df9f70606795cf67d9e9a306e67d5b, 2026-11-18T11:59:54.197Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T12-03-07.json (sha256 8d03e1ddaf52513236bed7ca9157583370674bf97a23bad011e6a1e73a1fab14, 2026-11-18T12:03:07.560Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T10-36-02.json (sha256 1b231c8bc8fd1d810078713019229cf8e7dc0e93d16952fa422dff0d768affee, 2026-11-19T10:36:02.190Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 83b9a2c0d76e47fffb96e1252de204befec771ec
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T10-49-27.json (sha256 207729d81cdb2c073807c00f43957d47368c21218495f226bd444e08001a1e56, 2026-11-19T10:49:27.638Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 83b9a2c0d76e47fffb96e1252de204befec771ec
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T11-04-24.json (sha256 a66fa62fba392d2ddf4924f4cd5ca2bab3022664cfedc78e57bbc65c733a1089, 2026-11-19T11:04:24.090Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 83b9a2c0d76e47fffb96e1252de204befec771ec
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T11-17-54.json (sha256 68a9a99337caf2e0b77f205823351a3284aee2d93cd67abd0c4201f1862d6629, 2026-11-19T11:17:54.434Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 83b9a2c0d76e47fffb96e1252de204befec771ec
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T12-29-01.json (sha256 16536bc022608ba674cd0a1c11b1a41adedc3a1ada741f9a936a0cc66167ae37, 2026-11-19T12:29:01.826Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout fb5c0e5290353f73e63b836f773fb50ed3f3c4ca
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T12-42-12.json (sha256 f3ae490f6030cb670cae21daebbe0f1ae968e5948300cdfaedbd2fead1e4ce79, 2026-11-19T12:42:12.275Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout fb5c0e5290353f73e63b836f773fb50ed3f3c4ca (dirty)
 *
 * Carried over from the registered table at commit 10b322c025bc24a8475fb0503154ed0a2270d6a8: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d.
 * Their adoption lines are in `git show 10b322c025bc24a8475fb0503154ed0a2270d6a8:packages/runtime/src/kernels/geometry-profiles/nvidia-ampere.ts`.
 *
 * GEMV choices: a candidate replaces the default profile's choice for a shape only when it is at
 * least ×1.100 faster than that choice in every GEMV sweep record, with its words equal to
 * the baseline's (identical class: the sequential variant's per-shape exceptions) or inside the
 * band (reorder class: the parallel rows, measured at every row count of the parallel range and
 * not slower at any) in every record; among those, the one whose smallest gain is largest. A shape
 * that a record measured while the baseline's rounds spread beyond ×1.250 keeps the default,
 * and so do the shared sequential keys whose cases disagree and the rows with packed activations.
 *
 * - linearGemv (the GEMV choices of linear), from 21 GEMV sweep records:
 *   - [1] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T03-35-23.json (2026-11-18T03:35:23.669Z, set quick): 3 groups — 0 current and steady, 0 unsteady, 3 not current; 0 rows of current groups not current
 *     - 3 not current: the default profile's choices moved
 *   - [2] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T03-41-44.json (2026-11-18T03:41:44.165Z, set full): 24 groups — 15 current and steady, 3 unsteady, 6 not current; 0 rows of current groups not current
 *     - 6 not current: the default profile's choices moved
 *   - [3] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T04-12-00.json (2026-11-18T04:12:00.301Z, set full): 39 groups — 8 current and steady, 1 unsteady, 30 not current; 0 rows of current groups not current
 *     - 30 not current: the default profile's choices moved
 *   - [4] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T04-39-54.json (2026-11-18T04:39:54.559Z, set full): 36 groups — 8 current and steady, 2 unsteady, 26 not current; 0 rows of current groups not current
 *     - 26 not current: the default profile's choices moved
 *   - [5] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T04-56-22.json (2026-11-18T04:56:22.175Z, set full): 24 groups — 13 current and steady, 5 unsteady, 6 not current; 0 rows of current groups not current
 *     - 6 not current: the default profile's choices moved
 *   - [6] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-32-42.json (2026-11-18T11:32:42.796Z, set full): 13 groups — 1 current and steady, 0 unsteady, 12 not current; 0 rows of current groups not current
 *     - 12 not current: the default profile's choices moved
 *   - [7] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-36-30.json (2026-11-18T11:36:30.026Z, set full): 2 groups — 0 current and steady, 0 unsteady, 2 not current; 0 rows of current groups not current
 *     - 2 not current: the default profile's choices moved
 *   - [8] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-41-59.json (2026-11-18T11:41:59.472Z, set full): 12 groups — 1 current and steady, 1 unsteady, 10 not current; 0 rows of current groups not current
 *     - 10 not current: the default profile's choices moved
 *   - [9] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-43-05.json (2026-11-18T11:43:05.129Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [10] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-46-29.json (2026-11-18T11:46:29.709Z, set full): 13 groups — 1 current and steady, 0 unsteady, 12 not current; 0 rows of current groups not current
 *     - 12 not current: the default profile's choices moved
 *   - [11] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-49-44.json (2026-11-18T11:49:44.127Z, set full): 12 groups — 2 current and steady, 0 unsteady, 10 not current; 0 rows of current groups not current
 *     - 10 not current: the default profile's choices moved
 *   - [12] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-53-13.json (2026-11-18T11:53:13.036Z, set full): 13 groups — 1 current and steady, 0 unsteady, 12 not current; 0 rows of current groups not current
 *     - 12 not current: the default profile's choices moved
 *   - [13] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-56-28.json (2026-11-18T11:56:28.481Z, set full): 12 groups — 1 current and steady, 1 unsteady, 10 not current; 0 rows of current groups not current
 *     - 10 not current: the default profile's choices moved
 *   - [14] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T11-59-54.json (2026-11-18T11:59:54.197Z, set full): 13 groups — 1 current and steady, 0 unsteady, 12 not current; 0 rows of current groups not current
 *     - 12 not current: the default profile's choices moved
 *   - [15] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-18T12-03-07.json (2026-11-18T12:03:07.560Z, set full): 12 groups — 1 current and steady, 1 unsteady, 10 not current; 0 rows of current groups not current
 *     - 10 not current: the default profile's choices moved
 *   - [16] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T10-36-02.json (2026-11-19T10:36:02.190Z, set full): 75 groups — 19 current and steady, 5 unsteady, 51 not current; 0 rows of current groups not current
 *     - 51 not current: the default profile's choices moved
 *   - [17] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T10-49-27.json (2026-11-19T10:49:27.638Z, set full): 75 groups — 20 current and steady, 4 unsteady, 51 not current; 0 rows of current groups not current
 *     - 51 not current: the default profile's choices moved
 *   - [18] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T11-04-24.json (2026-11-19T11:04:24.090Z, set full): 75 groups — 20 current and steady, 4 unsteady, 51 not current; 0 rows of current groups not current
 *     - 51 not current: the default profile's choices moved
 *   - [19] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T11-17-54.json (2026-11-19T11:17:54.434Z, set full): 75 groups — 19 current and steady, 5 unsteady, 51 not current; 0 rows of current groups not current
 *     - 51 not current: the default profile's choices moved
 *   - [20] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T12-29-01.json (2026-11-19T12:29:01.826Z, set full): 75 groups — 50 current and steady, 25 unsteady, 0 not current; 0 rows of current groups not current
 *   - [21] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-19T12-42-12.json (2026-11-19T12:42:12.275Z, set full): 75 groups — 61 current and steady, 14 unsteady, 0 not current; 0 rows of current groups not current
 * - identical class (sequential.shapes, M = 1): 5 adopted, 28 kept
 *   - i4 n262144 k1536 (case i4g32-n262144-k1536): c32u4 -> c16u2 (×1.839 [2], ×1.870 [5], ×2.523 [9])
 *   - i4 n256 k3072 (case i4g32-n256-k3072): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not current in [2] (the default profile's choice at reorder is c32u4 in the record, parallel-l32 now); M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not current in [5] (the default profile's choice at reorder is c32u4 in the record, parallel-l32 now); M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21])))
 *   - i4 n1024 k256 (case i4g32-n1024-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.019 < ×1.100 in [2])); c32u8: ×1.019 [2]
 *   - i4 n256 k1024 (case i4g32-n256-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not current in [2] (the default profile's choice at reorder is c32u4 in the record, parallel-l32 now); M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not current in [5] (the default profile's choice at reorder is c32u4 in the record, parallel-l32 now); M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21])))
 *   - i4 n2048 k256 (case i4g32-n2048-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.033 < ×1.100 in [2])); c32u8: ×1.033 [2], ×1.034 [5]
 *   - i4 n256 k2048 (case i4g32-n256-k2048): c32u4 -> c16u16 (×1.182 [2], ×1.182 [5])
 *   - i4 n262144 k256 (case i4g32-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.006 < ×1.100 in [2])); c32u8: ×1.006 [2], ×1.017 [5]
 *   - i4 n262144 k2560 (case i4g32-n262144-k2560): c32u4 -> c16u2 (×1.699 [2], ×1.713 [5], ×1.678 [9])
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×0.966 < ×1.100 in [16])); c32u8: ×0.966 [16], ×0.967 [17], ×0.959 [19]
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.018 < ×1.100 in [21])); c32u16: ×1.018 [21]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.915 < ×1.100 in [20])); c16u8: ×0.915 [20]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.092 < ×1.100 in [16])); c16u16: ×1.092 [16], ×1.091 [17], ×1.091 [18], ×1.089 [19]
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×0.911 < ×1.100 in [21])); c32u8: ×0.911 [21]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.944 < ×1.100 in [20])); c16u8: ×0.944 [20], ×0.982 [21]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): keeps c32u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.011 < ×1.100 in [20])); c16u16: ×1.011 [20], ×0.954 [21]
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not current in [3] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not current in [6] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not current in [10] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [11]; M=1 is not current in [12] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [13]; M=1 is not current in [14] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [15]; M=1 is not current in [16] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); screened out by the quick look in [20] at M=1; screened out by the quick look in [21] at M=1)))
 *   - bf16 n512 k1536 (case bf16-n512-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×0.996 < ×1.100 in [20])); c32u16: ×0.996 [20]
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.984 < ×1.100 in [21])); c16u8: ×0.984 [21]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.781 < ×1.100 in [17])); c16u16: ×0.781 [17], ×0.881 [18], ×0.890 [20], ×0.879 [21]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.979 < ×1.100 in [20])); c16u8: ×0.979 [20], ×0.961 [21]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×0.933 < ×1.100 in [3])); c16u4: ×0.933 [3], ×1.556 [6], ×1.547 [17], ×1.553 [18], ×1.550 [19], ×1.549 [21]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×0.967 < ×1.100 in [18])); c32u8: ×0.967 [18], ×0.983 [19]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.002 < ×1.100 in [21])); c32u8: ×1.002 [21]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.899 < ×1.100 in [21])); c16u8: ×0.899 [21]
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.901 < ×1.100 in [21])); c16u8: ×0.901 [21]
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.995 < ×1.100 in [21])); c16u16: ×0.995 [21]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): c32u4 -> c16u8 (×1.689 [4], ×1.702 [8], ×1.688 [11], ×1.688 [13], ×1.699 [15], ×1.690 [16], ×1.688 [17], ×1.653 [18], ×1.642 [19], ×1.697 [20])
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.022 < ×1.100 in [20])); c16u16: ×1.022 [20], ×1.038 [21]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.004 < ×1.100 in [16])); c32u16: ×1.004 [16], ×0.998 [17], ×0.999 [18], ×0.997 [19], ×0.979 [20], ×0.983 [21]
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not current in [4] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not current in [8] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not current in [11] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [12]; M=1 is not current in [13] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [14]; M=1 is not current in [15] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not current in [16] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); screened out by the quick look in [20] at M=1; screened out by the quick look in [21] at M=1)))
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×0.911 < ×1.100 in [21])); c32u16: ×0.911 [21]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.021 < ×1.100 in [20])); c32u16: ×1.021 [20], ×0.939 [21]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): c32u4 -> c16u4 (×2.139 [16], ×1.616 [21])
 * - reorder class (parallelShapes, M = 1 to 8): 1 adopted, 32 kept
 *   - i4 g32 n262144 k1536 (case i4g32-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; screened out by the quick look in [2] at M=4; M=4 is not measured in [3]; M=4 is not measured in [4]; screened out by the quick look in [5] at M=4; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]))); staged-l8: ×3.591 [2], ×3.715 [5], ×4.732 [9]
 *   - i4 g32 n256 k3072 (case i4g32-n256-k3072): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not current in [2] (the default profile's choice at reorder is c32u4 in the record, parallel-l32 now); M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not current in [5] (the default profile's choice at reorder is c32u4 in the record, parallel-l32 now); M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21])))
 *   - i4 g32 n1024 k256 (case i4g32-n1024-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [3]; M=1 is not measured in [4]; unsteady in [5] at M=1 (baseline spread ×1.265 > ×1.250); M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]))); parallel-l4: ×1.235 [2]
 *   - i4 g32 n256 k1024 (case i4g32-n256-k1024): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not current in [2] (the default profile's choice at reorder is c32u4 in the record, parallel-l32 now); M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not current in [5] (the default profile's choice at reorder is c32u4 in the record, parallel-l32 now); M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21])))
 *   - i4 g32 n2048 k256 (case i4g32-n2048-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: slower than the default's choice in [2] at M=4 (×0.846))); parallel-l4: ×1.227 [2], ×1.227 [5]
 *   - i4 g32 n256 k2048 (case i4g32-n256-k2048): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [3]; M=4 is not measured in [4]; unsteady in [5] at M=4 (baseline spread ×1.257 > ×1.250); M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]))); parallel-l32: ×4.854 [2], ×4.877 [5]
 *   - i4 g32 n262144 k256 (case i4g32-n262144-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l4: ×0.940 < ×1.100 in [2])); staged-l4: ×0.940 [2], ×0.948 [5]
 *   - i4 g32 n262144 k2560 (case i4g32-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; screened out by the quick look in [2] at M=4; M=4 is not measured in [3]; M=4 is not measured in [4]; screened out by the quick look in [5] at M=4; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]))); staged-l8: ×4.408 [2], ×4.097 [5], ×4.331 [9]
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.081 < ×1.100 in [16])); parallel-l32: ×1.081 [16], ×1.089 [17], ×1.079 [19]
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not current in [3] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not current in [6] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not current in [7] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not current in [10] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [11]; M=1 is not current in [12] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [13]; M=1 is not current in [14] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [15]; M=1 is not current in [16] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); unsteady in [20] at M=1 (baseline spread ×1.744 > ×1.250)))); parallel-l32: ×1.216 [21]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.864 < ×1.100 in [20])); parallel-l16: ×0.864 [20]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; unsteady in [3] at M=4 (baseline spread ×1.382 > ×1.250); M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; unsteady in [16] at M=4 (baseline spread ×1.295 > ×1.250); unsteady in [17] at M=4 (baseline spread ×1.343 > ×1.250); unsteady in [18] at M=4 (baseline spread ×1.340 > ×1.250); unsteady in [19] at M=4 (baseline spread ×1.328 > ×1.250); unsteady in [20] at M=4 (baseline spread ×1.293 > ×1.250); unsteady in [21] at M=4 (baseline spread ×1.260 > ×1.250)))); parallel-l32: ×3.524 [16], ×3.519 [17], ×3.520 [18], ×3.506 [19]
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.024 < ×1.100 in [21])); parallel-l16: ×1.024 [21]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.089 < ×1.100 in [20])); parallel-l32: ×1.089 [20], ×1.105 [21]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.021 < ×1.100 in [20])); parallel-l16: ×1.021 [20], ×1.022 [21]
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not current in [3] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not current in [6] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not current in [10] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [11]; M=1 is not current in [12] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [13]; M=1 is not current in [14] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [15]; M=1 is not current in [16] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); screened out by the quick look in [20] at M=1; screened out by the quick look in [21] at M=1)))
 *   - bf16 n512 k1536 (case bf16-n512-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not current in [3] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not current in [6] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not current in [10] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [11]; M=1 is not current in [12] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [13]; M=1 is not current in [14] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [15]; M=1 is not current in [16] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); unsteady in [21] at M=1 (baseline spread ×1.992 > ×1.250)))); parallel-l32: ×1.207 [20]
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not current in [3] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not current in [6] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not current in [10] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [11]; M=1 is not current in [12] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [13]; M=1 is not current in [14] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [15]; M=1 is not current in [16] (the default profile's choice at reorder is c16u16 in the record, staged-l32 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u16 in the record, staged-l32 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u16 in the record, staged-l32 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u16 in the record, staged-l32 now); unsteady in [20] at M=1 (baseline spread ×2.163 > ×1.250)))); parallel-l32: ×1.113 [21]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not measured in [1]; M=8 is not measured in [2]; screened out by the quick look in [3] at M=8; M=8 is not measured in [4]; M=8 is not measured in [5]; M=8 is not measured in [6]; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; screened out by the quick look in [16] at M=8; screened out by the quick look in [17] at M=8; screened out by the quick look in [18] at M=8; screened out by the quick look in [19] at M=8; screened out by the quick look in [20] at M=8; screened out by the quick look in [21] at M=8))); parallel-l16: ×1.116 [17], ×1.120 [18], ×1.114 [20], ×1.108 [21]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.095 < ×1.100 in [20])); parallel-l32: ×1.095 [20], ×1.079 [21]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not measured in [1]; M=8 is not measured in [2]; screened out by the quick look in [3] at M=8; M=8 is not measured in [4]; M=8 is not measured in [5]; M=8 is not measured in [6]; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; screened out by the quick look in [16] at M=8; screened out by the quick look in [17] at M=8; screened out by the quick look in [18] at M=8; screened out by the quick look in [19] at M=8; screened out by the quick look in [20] at M=8; screened out by the quick look in [21] at M=8))); parallel-l16: ×1.676 [3], ×2.792 [6], ×2.781 [17], ×2.793 [18], ×2.789 [19], ×2.783 [21]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: slower than the default's choice in [16] at M=8 (×0.813))); parallel-l32: ×1.139 [18], ×1.160 [19]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.061 < ×1.100 in [21])); parallel-l32: ×1.061 [21]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not current in [4] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not current in [8] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not current in [11] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [12]; M=1 is not current in [13] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [14]; M=1 is not current in [15] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not current in [16] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); screened out by the quick look in [20] at M=1; screened out by the quick look in [21] at M=1)))
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not current in [4] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not current in [8] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not current in [11] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [12]; M=1 is not current in [13] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [14]; M=1 is not current in [15] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not current in [16] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); screened out by the quick look in [20] at M=1; screened out by the quick look in [21] at M=1)))
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not current in [4] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not current in [8] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not current in [11] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [12]; M=1 is not current in [13] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [14]; M=1 is not current in [15] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not current in [16] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); unsteady in [20] at M=1 (baseline spread ×2.375 > ×1.250)))); parallel-l32: ×1.222 [21]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not measured in [1]; M=8 is not measured in [2]; M=8 is not measured in [3]; screened out by the quick look in [4] at M=8; M=8 is not measured in [5]; M=8 is not measured in [6]; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; screened out by the quick look in [16] at M=8; screened out by the quick look in [17] at M=8; screened out by the quick look in [18] at M=8; screened out by the quick look in [19] at M=8; screened out by the quick look in [20] at M=8; screened out by the quick look in [21] at M=8))); parallel-l32: ×1.935 [4], ×1.929 [8], ×1.981 [16], ×1.975 [17], ×1.977 [18], ×1.950 [19], ×1.924 [20]
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.016 < ×1.100 in [20])); parallel-l32: ×1.016 [20], ×1.020 [21]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): no row (c16u16) -> parallel-l16 (×1.135 [16], ×1.141 [17], ×1.133 [18], ×1.120 [19], ×1.119 [20], ×1.124 [21]); at the other row counts: ×2.122 [17] M=4, ×2.174 [20] M=4, ×1.337 [4] M=8, ×1.273 [16] M=8, ×1.168 [17] M=8, ×1.327 [18] M=8, ×1.212 [19] M=8, ×1.269 [20] M=8, ×1.273 [21] M=8
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not current in [4] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not current in [8] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not current in [11] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [12]; M=1 is not current in [13] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [14]; M=1 is not current in [15] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not current in [16] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); screened out by the quick look in [20] at M=1; screened out by the quick look in [21] at M=1)))
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not current in [4] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not current in [8] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not current in [11] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [12]; M=1 is not current in [13] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [14]; M=1 is not current in [15] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not current in [16] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not current in [17] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not current in [18] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not current in [19] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); unsteady in [20] at M=1 (baseline spread ×1.895 > ×1.250)))); parallel-l32: ×1.267 [21]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.026 < ×1.100 in [20])); parallel-l32: ×1.026 [20], ×1.043 [21]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not measured in [1]; M=8 is not measured in [2]; M=8 is not measured in [3]; screened out by the quick look in [4] at M=8; M=8 is not measured in [5]; M=8 is not measured in [6]; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; screened out by the quick look in [16] at M=8; screened out by the quick look in [17] at M=8; screened out by the quick look in [18] at M=8; screened out by the quick look in [19] at M=8; screened out by the quick look in [20] at M=8; screened out by the quick look in [21] at M=8))); parallel-l16: ×4.027 [16], ×3.040 [21]
 * - rowsThreadTarget: keeps 16384 (the full and row-blocks sets measure the row-block heights per case and row count, but the field is one thread target for every shape, and no rule turns the heights into it)
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const NVIDIA_AMPERE: GeometryProfile = {
  id: "nvidia-ampere",
  match: { vendor: "nvidia", architecture: "ampere" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 64, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 8, wgX: 16, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
    pv: { regM: 4, regN: 4, wgX: 16, wgY: 16 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 8, wgX: 16, wgY: 8 },
    rows32: { regM: 8, regN: 8, wgX: 16, wgY: 4 },
  },
  i8a8: {
    linear: [
      { maxRows: 256, geometry: { regM: 8, regN: 4, wgX: 8, wgY: 4, tileK: 64, skeleton: "vec4" } },
      {
        maxRows: 512,
        geometry: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      },
      {
        maxRows: Number.POSITIVE_INFINITY,
        geometry: { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      },
    ],
    attentionQk: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
    attentionPv: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 16 },
    conv3d: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
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
        { storage: "i4", n: 262144, k: 1536, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 256, k: 2048, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 262144, k: 2560, variant: { cols: 16, unroll: 2 } },
        { storage: "bf16", n: 10240, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 262144, k: 2560, variant: { cols: 16, unroll: 4 } },
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
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1536,
        k: 2048,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
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
      { storage: "i4", n: 1536, k: 256, group: 32, lanes: 4, packedActivations: false },
      {
        storage: "i4",
        n: 4096,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      { storage: "i4", n: 512, k: 1536, group: 32, lanes: 32, packedActivations: false },
      {
        storage: "i4",
        n: 1536,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
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
      { storage: "i8", n: 262144, k: 1536, lanes: 16, packedActivations: false },
      { storage: "i4", n: 2048, k: 1536, group: 512, lanes: 4, packedActivations: false },
      { storage: "i4", n: 256, k: 1536, group: 512, lanes: 32, packedActivations: false },
      { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i4", n: 6144, k: 1536, group: 512, lanes: 4, packedActivations: false },
      { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 1536, lanes: 32, packedActivations: false },
      { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false },
      { storage: "i4", n: 4096, k: 1536, group: 512, lanes: 4, packedActivations: false },
      { storage: "i4", n: 512, k: 1536, group: 512, lanes: 32, packedActivations: false },
      { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      { storage: "i2", n: 12288, k: 1536, lanes: 2, packedActivations: false },
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
      { storage: "i4", n: 512, k: 2560, group: 32, lanes: 32, packedActivations: false },
      { storage: "i4", n: 256, k: 2560, group: 32, lanes: 32, packedActivations: false },
      {
        storage: "i4",
        n: 2560,
        k: 2048,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
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
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
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
      { storage: "i8", n: 262144, k: 2560, lanes: 16, packedActivations: false },
      { storage: "i4", n: 2048, k: 2560, group: 512, lanes: 4, packedActivations: false },
      { storage: "i4", n: 512, k: 2560, group: 512, lanes: 32, packedActivations: false },
      { storage: "i4", n: 2560, k: 2048, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i4", n: 10240, k: 2560, group: 512, lanes: 4, packedActivations: false },
      { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 2560, lanes: 32, packedActivations: false },
      { storage: "i8", n: 2560, k: 256, lanes: 4, packedActivations: false },
      { storage: "i4", n: 4096, k: 2560, group: 512, lanes: 4, packedActivations: false },
      { storage: "i4", n: 1024, k: 2560, group: 512, lanes: 32, packedActivations: false },
      { storage: "i4", n: 2560, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      { storage: "i4", n: 256, k: 3072, group: 32, lanes: 32, packedActivations: false },
      { storage: "i4", n: 256, k: 1024, group: 32, lanes: 32, packedActivations: false },
      { storage: "i4", n: 15360, k: 3840, group: 32, lanes: 4, packedActivations: false },
      { storage: "i4", n: 8192, k: 3840, group: 32, lanes: 4, packedActivations: false },
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
      },
      { storage: "i4", n: 262144, k: 3840, group: 32, lanes: 16, packedActivations: false },
      {
        storage: "i4",
        n: 1024,
        k: 7680,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 1024,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 1024,
        k: 8192,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
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
      { storage: "i4", n: 1024, k: 2816, group: 32, lanes: 32, packedActivations: false },
      { storage: "i4", n: 128, k: 2816, group: 32, lanes: 32, packedActivations: false },
      {
        storage: "i4",
        n: 2816,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 2816,
        k: 8192,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 2816,
        k: 2112,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      { storage: "i4", n: 262144, k: 2816, group: 32, lanes: 4, packedActivations: false },
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
      { storage: "bf16", n: 2560, k: 256, lanes: 16, packedActivations: false },
    ],
  },
  provenance: {
    adapter: {
      vendor: "nvidia",
      architecture: "ampere",
      device: "nvidia-geforce-rtx-3080-ti",
      description: "NVIDIA: 615.78.08 615.78.8.0",
    },
    families: {
      gemm: {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "ca7e78e7f763ea6a",
        caseSet: "e6ddd45226ab2155",
      },
      attention: {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "427da13d86d8f3f7",
        caseSet: "9ee211cf69674928",
      },
      conv2d: {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "e6004bc50ba66635",
        caseSet: "02186e45558e4151",
      },
      "i8a8-linear": {
        sweep:
          "outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk00.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk01.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk02.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk03.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk04.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk05.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk06.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk07.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk08.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk09.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk10.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk11.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk12.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk13.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk14.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk15.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk16.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk17.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk18.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk19.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk20.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk21.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk22.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk23.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk24.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk25.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk26.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk27.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk28.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk00.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk01.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk02.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk03.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk04.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk05.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk06.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk07.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk08.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk09.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk10.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk11.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk12.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk13.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk14.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk15.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk16.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk17.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk18.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk19.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk20.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk21.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk22.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk23.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk24.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk25.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk26.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk27.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk28.json",
        sha256:
          "0329bb33aab749a4e0ce4dc31410c43a29e6555f3409c1e140d7fb1fc689ef23, d37547bee2af0de170337a5d3055fefc0c099ee84c61c11ff97b5e8589daf25f, 21c4d32d4c28e24dd6fe793c63502ce65e4ae63399f79a630e46e4ee09161ac9, 3e0f6e03cc01746ca21130630e428122713d3bb1a3210eef1bb3b859dbe672ba, 327dbd0d37467e919df7f9c67f29f084df54ac7896127f02e46d4d2100f90c94, c3dda389f5817bc33981345a2d145fb29d01f63d8430a0d1d2f0675645d7f9b9, bedd36845b17bb55b940af92f71c03e9ef25d58cf216c8fb93cc3215c6238f3e, 0957c9a30f8762878280cfd630375fc249ce1f59ef606db3a409e23b1967376d, d862d190d365fc8fe715350b39c1950c4943cbc37f6cb579c10d84462c3d67df, 4d35dbe7c5f9d520e67031fa388b9399d5b25ab78c5ed09280e75d0cee871f25, c8faa4655b378a019d4494877171b6f19cddea9f94681f3a2087e7ddeb516bb3, e01484781866705a1b9c4684f783d2135f83294a64f7b6c92ef0cd7d4b1e3b19, d7ebacbeaf55d4d71cd0e1d9a43769c313b9064699ead206963b7c8ff59a608f, 78c20112462dc9b88fbeba6e148a3e2a3cd8f483c55006966bc85a6dd3771532, d8427fdbcf5e3a2bfcd531ff5d25b20320b3c42ede1ec88daad2173c8596dc67, 2e4ecae8a55e1688bb65a6c6c8bfd4597d75eaad041b55091b1cd337b2070362, cc33762f678409cff2c49708e40f40886bbaacbd30c88d8c76d0d1abb34cd5ca, 7c53855ea4137ebf0ae8ecff3f8294da595da865719d170ab7ee10adfb5b29d8, 78bfc9ee01790bbb1ceec01e1a08883b28f1607ba4f85fa9685b59cfd6661318, 7cee21018aa72bfdd447d8be2a4484b0cc13da3877cdd8e198ba3ad75d3720d0, bd41470bee17e9b9b1baf318e94efddca2d0986a272c963063d57e8f974b7699, 63f5ad6727c53da5a74a46df8dabb35cba2fb084565e38cb8ef0e270f387b60f, 2187c03e311268ab7b0192206845d17ff8d95937ccd75292ff7318fb74283b2a, 8d6400cbeedd4bc3435c15df892a895cffe88c7a05799aaa1101e69854725b4a, 954f11d3882c0a28c7aa9e3e71c4f687b0f707d92cda050583c72b2fd61cc828, 63baef9fe699384280a4fb8445f66475005965aac5932af6b38e7f9f46784eee, 89ae2d654e4a8626404d587802b7580e27376eb456a56ff9493fbb865057fd10, b3bd81a935800ef67cf0b43a6b4247c6d392e48bf68894a5b5cef9499b21ed51, 7b842f9c18d5e0958a1a430305339c7c2e3fd5775fbac986affd0d051e6577b9, 6f336d35408a8cf96e1826c9eac70d9b9d8ce0a674e083d1ac6f61798c764c39, 152363090860f00f4c1caf3f1e2216f8f0380c5731d26d4824ad12de19343953, 76fd26809b5aaa5481185f95cdfebd087f255e940f6e515986324d6189bcb2a2, 53d4d8aa77864d7dbb7ef74013806a51242d694cdd06a70062fbe82f0e262a2a, 59dc2fcc29408f5b8ced409700ab52eca170b0119426e1f9724b37908b8d44af, 50e40590ec838283e6ff44ef73c124d20fd34340bfd44c12a339a12d0ab749b5, 0f183b2b5e6b34e70e78e3dc4f2d44f19f9fa24f80d9c6718eff17c69d5faa8b, 4a54ea752118e0dfcba6f7e7ff0169466e692b1195e2a6299d0e09ece94a13e7, bc03ce41e9a4fb28eedddcb4ed9991049fcd7e3c9ddb5b9f1cf970d1d5b4ef11, d0a72b63f0e95c6319e7ee43ccf6d0fe43084e770ae08b24195211f5a0852996, 0ffec9fecaca5edb40a4d56739fac289308f4b9dc110c17ecc367dc4cf270ea9, 0cae1e5e2b32b5c1f86d63487312c1e0677d0d78ffccb376fd36f728d8c8e7fc, 7699d6c2172b26354f3ec069dcdc9f07fde49c3b879866b2cf037e6578bfd18a, 2a28702a04faecafb79386307f0f585d3c2c474e6f45bc9bb0c090d044edc5de, ea69d254b5f5aa794cc4ebcac012022e5ee9acb7357dc535286c9d50fb741853, 1f9d6b3708cce7017a20c8faff39e4a31fe30cfa0f5ce9a50bc50e4d3081fc08, 89f24c9980cf4eeacb3a301d3651f99fe6a52d4429931984de1b336bb01f185f, 1a535a4880f5f922fe17e939684da5132febc2964140fad9947559e8366a40a5, 9e758cabb23aa1b809c2791dec4e290a67391a57ccfb5198606767b029fc4370, 51677686cf19906a5e02612c3b7d463554b49cf6b53ad0c58288ef22e65f5772, 74c68e0ce7e84b9d8600ba4add66610e47825b4e7260355c972c9ce9d78c3b9c, 378d29fad7a6e6ad87c7694b5d008f9edfc64a3c5b164c267ead42af08f3360d, 08e93a743c7f28a7962582081a532d1dc65d037a95aee963f469fc0a5acc93e3, b186145fe83b72e39ca5e7043f45239fd39b6195f30d881db7b8cb26ff672c41, fc31d7f840c3ff4cc601e69d60ef3bc855c8d5f4a67254ff6beefc6b7be15797, f0c0d37c7094ddb8c569c7bd4e7d41f0449710c756123cb55e20e926f4337eea, cbec8948b28d7922eaff803dc94c0318be72a52cae57597e2c72395464860df3, 180f4a42f6561d0df0ba63a14548af306160af5066599f13dec067580bd78356, a73c8067767a11233e756b095140099f175d19aa37824d5f9ccf65791a1417b1",
        date: "2026-11-12T19:07:45.413Z, 2026-11-12T19:20:16.703Z, 2026-11-12T19:35:33.963Z, 2026-11-12T19:48:16.473Z, 2026-11-12T20:00:36.426Z, 2026-11-12T20:12:57.272Z, 2026-11-12T20:27:46.518Z, 2026-11-12T20:40:22.082Z, 2026-11-12T20:51:50.705Z, 2026-11-12T21:04:50.877Z, 2026-11-12T21:18:30.344Z, 2026-11-12T21:29:43.985Z, 2026-11-12T21:41:35.965Z, 2026-11-12T21:54:25.953Z, 2026-11-12T22:06:13.631Z, 2026-11-12T22:17:12.106Z, 2026-11-12T22:28:09.315Z, 2026-11-12T22:41:30.022Z, 2026-11-12T22:53:05.183Z, 2026-11-12T23:05:59.394Z, 2026-11-12T23:21:02.324Z, 2026-11-12T23:32:39.244Z, 2026-11-12T23:45:54.276Z, 2026-11-12T23:57:26.676Z, 2026-11-13T00:09:59.560Z, 2026-11-13T00:22:12.553Z, 2026-11-13T00:35:35.441Z, 2026-11-13T00:49:26.776Z, 2026-11-13T00:59:38.687Z, 2026-11-13T01:10:44.869Z, 2026-11-13T01:23:12.386Z, 2026-11-13T01:36:36.635Z, 2026-11-13T01:47:36.702Z, 2026-11-13T01:59:50.752Z, 2026-11-13T02:11:17.364Z, 2026-11-13T02:21:54.699Z, 2026-11-13T02:32:39.393Z, 2026-11-13T02:43:08.546Z, 2026-11-13T02:54:15.361Z, 2026-11-13T03:05:31.815Z, 2026-11-13T03:17:55.116Z, 2026-11-13T03:30:26.736Z, 2026-11-13T03:43:21.147Z, 2026-11-13T03:56:13.458Z, 2026-11-13T04:09:14.912Z, 2026-11-13T04:21:24.188Z, 2026-11-13T04:33:27.546Z, 2026-11-13T04:46:57.493Z, 2026-11-13T04:58:25.466Z, 2026-11-13T05:11:54.581Z, 2026-11-13T05:25:18.915Z, 2026-11-13T05:38:55.566Z, 2026-11-13T05:51:50.755Z, 2026-11-13T06:03:39.432Z, 2026-11-13T06:15:29.737Z, 2026-11-13T06:27:12.039Z, 2026-11-13T06:39:08.540Z, 2026-11-13T06:48:24.510Z",
        candidateSet:
          "full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full",
        userAgent: ["Node.js/24"],
        kernels: "67d915bfeae0cc15",
        caseSet: "3d32dc6f10559af2",
      },
      "i8a8-attention": {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "d5b184a653a1b381",
        caseSet: "8ab1a86b13087640",
      },
      "i8a8-conv3d": {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "c91b3368411e71d9",
        caseSet: "27ce9da45b359b14",
      },
    },
  },
};
