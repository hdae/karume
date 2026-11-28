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
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-20T16-44-59.json (sha256 be430dfec92c34d8b6d35dd2268154882cba5f3368c438fd1ea3438273133ea1, 2026-11-20T16:44:59.053Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 10809406fb2b7f0d843fe4f274ff0343f391cd79
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-20T16-48-17.json (sha256 2d1e4495508e320b3692c4818adc8dcb1e049828d0b9301a085f89cd49f0653d, 2026-11-20T16:48:17.924Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 10809406fb2b7f0d843fe4f274ff0343f391cd79
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-20T19-17-07.json (sha256 9e1475075a20c239bd59d060a8ef1835391d8aa813f2ad33325d5645da3eef87, 2026-11-20T19:17:06.991Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 15a44045f9be8eed42ae2c4d820c3637934facfe
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-20T20-37-57.json (sha256 a1011d9d84cbadfb968c733998497afebab480feb067dec2aa5fbfbad7488ca8, 2026-11-20T20:37:57.772Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 15a44045f9be8eed42ae2c4d820c3637934facfe (dirty)
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-29-20.json (sha256 00e66289045d9956e7238bd7d429503b1c3e4728145ed7c76eeab98755c6c9c9, 2026-11-21T03:29:20.159Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-29-42.json (sha256 3fecca5bdb96f2ae6b786f58450f83867ff4f4657550b894b36a7871089949e0, 2026-11-21T03:29:42.746Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-33-02.json (sha256 9afe2e1e796bccf3cdddef33510752bf94c0316a0d107d7fecf86583f9831912, 2026-11-21T03:33:02.141Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-33-15.json (sha256 9a47d2dc5996b6d286e15d1338263b76e3445557e54fe731dc6322a420738fc5, 2026-11-21T03:33:15.682Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-34-06.json (sha256 9eeca34b93064fb10f4ace4717616dd4b380935e803ef9870dbd6f5506b5cfc4, 2026-11-21T03:34:06.498Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-34-24.json (sha256 adbbed70bc61f460232a97c0de945c8170861cf7ad4b4410f321d3f24d760f4f, 2026-11-21T03:34:24.636Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-34-37.json (sha256 759c5ca102e534a0a873851c8decd19b005d992ac71a8678952430eb2e087065, 2026-11-21T03:34:37.328Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-34-46.json (sha256 ad68a3f120b7f0e2dd0c37e2421b698835faab1f8fd1e20234dbd2f9bb952968, 2026-11-21T03:34:46.160Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-35-01.json (sha256 74eb56d3b6a73bc85efe5067c43a3c02b42aee6361be32eb0a633f7464104566, 2026-11-21T03:35:01.709Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-35-37.json (sha256 36aa7554c1d5ed24b9000b8ef14811582b877abd8ccbbd424136ba30ab295c87, 2026-11-21T03:35:37.935Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-36-11.json (sha256 c4821adaa86c677fd15a023b6002569fda69416acdf255e2224ea83872d0e10b, 2026-11-21T03:36:11.885Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-36-32.json (sha256 8caaf43817cbcd0ced7661996cd859042c47922b7d4461b5bc3fb7aeac4ea44d, 2026-11-21T03:36:32.686Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-40-01.json (sha256 049c77e617067b2e0a15f34a1d6b909f44c6e5d5c7cb4c07b30f57ec07caf94e, 2026-11-21T03:40:01.101Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-40-15.json (sha256 aa88fa04ad515681012db7bd631d3d956a7d7d5c2ef5d7c510dd6400d70e3916, 2026-11-21T03:40:15.915Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-41-08.json (sha256 f611d08c8897daea6300af3ea6a128cd27be6791f37e605ff5858e60d1c07dfc, 2026-11-21T03:41:08.915Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-41-29.json (sha256 97960f5a093c181700c8a1104f4a37637dd839613f4e8a1a52397be42cec5c80, 2026-11-21T03:41:29.000Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-41-42.json (sha256 d37e169527eaae801954042953d41c1f883d9b13200a9c56975c99958cd010e0, 2026-11-21T03:41:42.270Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-41-53.json (sha256 2b62a099db2af75a5f0232c7aa60ae2afade2b95a95539c3b879375823cacd6b, 2026-11-21T03:41:53.195Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-42-08.json (sha256 7959dcaf51a694aafd9fd9147eac7778cd4d88de9921f5ef01c5249ecfa69f9f, 2026-11-21T03:42:08.964Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-42-42.json (sha256 b556fe582b096e1d048df650edb630dd8c06ead4a76c80a86154f4cbb305cda3, 2026-11-21T03:42:42.516Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout c39495833ce2dc3477609006f0cf1ff60a8f8091
 *
 * Carried over from the registered table at commit 8713ad4c63fafff9d62586a0ed71207ef1b7fe51: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d.
 * Their adoption lines are in `git show 8713ad4c63fafff9d62586a0ed71207ef1b7fe51:packages/runtime/src/kernels/geometry-profiles/nvidia-ampere.ts`.
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
 * - linearGemv (the GEMV choices of linear), from 24 GEMV sweep records:
 *   - [1] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-20T16-44-59.json (2026-11-20T16:44:59.053Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [2] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-20T16-48-17.json (2026-11-20T16:48:17.924Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [3] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-20T19-17-07.json (2026-11-20T19:17:06.991Z, set full): 345 groups — 345 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [4] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-20T20-37-57.json (2026-11-20T20:37:57.772Z, set full): 345 groups — 345 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [5] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-29-20.json (2026-11-21T03:29:20.159Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [6] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-29-42.json (2026-11-21T03:29:42.746Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [7] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-33-02.json (2026-11-21T03:33:02.141Z, set full): 8 groups — 8 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [8] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-33-15.json (2026-11-21T03:33:15.682Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [9] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-34-06.json (2026-11-21T03:34:06.498Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [10] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-34-24.json (2026-11-21T03:34:24.636Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [11] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-34-37.json (2026-11-21T03:34:37.328Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [12] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-34-46.json (2026-11-21T03:34:46.160Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [13] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-35-01.json (2026-11-21T03:35:01.709Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [14] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-35-37.json (2026-11-21T03:35:37.935Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [15] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-36-11.json (2026-11-21T03:36:11.885Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [16] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-36-32.json (2026-11-21T03:36:32.686Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [17] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-40-01.json (2026-11-21T03:40:01.101Z, set full): 8 groups — 8 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [18] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-40-15.json (2026-11-21T03:40:15.915Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [19] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-41-08.json (2026-11-21T03:41:08.915Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [20] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-41-29.json (2026-11-21T03:41:29.000Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [21] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-41-42.json (2026-11-21T03:41:42.270Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [22] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-41-53.json (2026-11-21T03:41:53.195Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [23] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-42-08.json (2026-11-21T03:42:08.964Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [24] outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_2026-11-21T03-42-42.json (2026-11-21T03:42:42.516Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 * - identical class (sequential.shapes, M = 1): 50 adopted, 49 kept
 *   - i4 n8960 k1536 (case i4g32-n8960-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.084 < ×1.100 in [3])); c32u16: ×1.084 [3], ×1.020 [4]
 *   - i4 n2048 k1536 (cases i4g32-n2048-k1536, i4g512-n2048-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2048-k1536 adopts c32u16; i4g512-n2048-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.092 < ×1.100 in [3])))
 *   - i4 n256 k1536 (cases i4g32-n256-k1536, i4g512-n256-k1536): c32u4 -> c32u16 (×1.146 [3] i4g32-n256-k1536, ×1.140 [4] i4g32-n256-k1536, ×1.222 [3] i4g512-n256-k1536, ×1.215 [4] i4g512-n256-k1536)
 *   - i4 n1536 k2048 (cases i4g32-n1536-k2048, i4g2048-n1536-k2048): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n1536-k2048 adopts c16u16; i4g2048-n1536-k2048 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.086 < ×1.100 in [3])))
 *   - i4 n6144 k1536 (cases i4g32-n6144-k1536, i4g512-n6144-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n6144-k1536 adopts c32u16; i4g512-n6144-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.009 < ×1.100 in [4])))
 *   - i4 n1536 k6144 (cases i4g32-n1536-k6144, i4g2048-n1536-k6144): c32u4 -> c16u16 (×1.237 [3] i4g32-n1536-k6144, ×1.241 [4] i4g32-n1536-k6144, ×1.266 [3] i4g2048-n1536-k6144, ×1.267 [4] i4g2048-n1536-k6144)
 *   - i4 n1536 k256 (case i4g32-n1536-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.020 < ×1.100 in [3])); c32u8: ×1.020 [3], ×1.020 [4]
 *   - i4 n4096 k1536 (cases i4g32-n4096-k1536, i4g512-n4096-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n4096-k1536 adopts c16u16; i4g512-n4096-k1536 adopts c32u16))
 *   - i4 n512 k1536 (cases i4g32-n512-k1536, i4g512-n512-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n512-k1536 adopts c16u16; i4g512-n512-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.083 < ×1.100 in [4])))
 *   - i4 n1536 k4096 (cases i4g32-n1536-k4096, i4g4096-n1536-k4096): c32u4 -> c16u16 (×1.178 [3] i4g32-n1536-k4096, ×1.174 [4] i4g32-n1536-k4096, ×1.106 [3] i4g4096-n1536-k4096, ×1.102 [4] i4g4096-n1536-k4096)
 *   - i4 n12288 k1536 (case i4g32-n12288-k1536): c32u4 -> c32u16 (×1.161 [3], ×1.162 [4])
 *   - i4 n1536 k12288 (case i4g32-n1536-k12288): c32u4 -> c32u16 (×1.196 [3], ×1.196 [4])
 *   - i8 n262144 k1536 (case i8-n262144-k1536): c32u4 -> c16u4 (×1.433 [3], ×1.433 [4])
 *   - i4 n262144 k1536 (case i4g32-n262144-k1536): c32u4 -> c16u1 (×1.876 [3], ×1.882 [4])
 *   - i4 n256 k3072 (case i4g32-n256-k3072): c32u4 -> c16u16 (×1.190 [3], ×1.187 [4])
 *   - i4 n1024 k256 (case i4g32-n1024-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.017 < ×1.100 in [3])); c32u8: ×1.017 [3], ×1.019 [4]
 *   - i4 n256 k1024 (case i4g32-n256-k1024): c32u4 -> c16u16 (×1.170 [3], ×1.164 [4])
 *   - i4 n2048 k256 (case i4g32-n2048-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.033 < ×1.100 in [3])); c32u8: ×1.033 [3], ×1.027 [4]
 *   - i4 n256 k2048 (case i4g32-n256-k2048): c32u4 -> c16u16 (×1.178 [3], ×1.182 [4])
 *   - i4 n262144 k256 (case i4g32-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.035 < ×1.100 in [3])); c32u8: ×1.035 [3], ×1.027 [4]
 *   - i8 n256 k3072 (case i8-n256-k3072): c32u4 -> c16u16 (×1.276 [3], ×1.286 [4])
 *   - i8 n1024 k256 (case i8-n1024-k256): c32u4 -> c32u16 (×1.152 [3], ×1.154 [4])
 *   - i8 n256 k1024 (case i8-n256-k1024): c32u4 -> c16u16 (×1.171 [3], ×1.212 [4])
 *   - i8 n2048 k256 (case i8-n2048-k256): c32u4 -> c32u16 (×1.168 [3], ×1.164 [4])
 *   - i8 n256 k2048 (case i8-n256-k2048): c32u4 -> c16u16 (×1.278 [3], ×1.278 [4])
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.008 < ×1.100 in [3])); c32u16: ×1.008 [3], ×1.008 [4]
 *   - i8 n256 k1536 (case i8-n256-k1536): c32u4 -> c16u16 (×1.271 [3], ×1.271 [4])
 *   - i8 n1536 k256 (case i8-n1536-k256): c32u4 -> c32u16 (×1.166 [3], ×1.163 [4])
 *   - i2 n12288 k1536 (case i2-n12288-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.038 < ×1.100 in [3])); c32u8: ×1.038 [3], ×1.030 [4]
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.044 < ×1.100 in [3])); c32u8: ×1.044 [3], ×1.044 [4]
 *   - i2 n262144 k1536 (case i2-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×0.988 < ×1.100 in [3])); c32u8: ×0.988 [3], ×0.997 [4]
 *   - i4 n10752 k2560 (case i4g32-n10752-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.095 < ×1.100 in [3])); c16u16: ×1.095 [3], ×1.094 [4]
 *   - i4 n2048 k2560 (cases i4g32-n2048-k2560, i4g512-n2048-k2560): c32u4 -> c16u16 (×1.170 [3] i4g32-n2048-k2560, ×1.171 [4] i4g32-n2048-k2560, ×1.187 [3] i4g512-n2048-k2560, ×1.102 [4] i4g512-n2048-k2560)
 *   - i4 n512 k2560 (cases i4g32-n512-k2560, i4g512-n512-k2560): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n512-k2560 adopts c32u16; i4g512-n512-k2560 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.088 < ×1.100 in [4])))
 *   - i4 n256 k2560 (case i4g32-n256-k2560): c32u4 -> c16u16 (×1.293 [3], ×1.269 [4])
 *   - i4 n2560 k2048 (cases i4g32-n2560-k2048, i4g2048-n2560-k2048): c32u4 -> c16u16 (×1.174 [3] i4g32-n2560-k2048, ×1.157 [4] i4g32-n2560-k2048, ×1.106 [3] i4g2048-n2560-k2048, ×1.109 [4] i4g2048-n2560-k2048)
 *   - i4 n10240 k2560 (cases i4g32-n10240-k2560, i4g512-n10240-k2560): keeps c32u4 (i4g32-n10240-k2560: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c64u2: ×1.081 < ×1.100 in [3]); i4g512-n10240-k2560: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.053 < ×1.100 in [1]))
 *   - i4 n2560 k10240 (cases i4g32-n2560-k10240, i4g2048-n2560-k10240): c32u4 -> c32u16 (×1.216 [3] i4g32-n2560-k10240, ×1.226 [4] i4g32-n2560-k10240, ×1.183 [3] i4g2048-n2560-k10240, ×1.189 [4] i4g2048-n2560-k10240)
 *   - i4 n2560 k256 (case i4g32-n2560-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.042 < ×1.100 in [3])); c32u8: ×1.042 [3], ×1.034 [4]
 *   - i4 n4096 k2560 (cases i4g32-n4096-k2560, i4g512-n4096-k2560): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n4096-k2560 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.056 < ×1.100 in [3]); i4g512-n4096-k2560 adopts c16u16))
 *   - i4 n1024 k2560 (cases i4g32-n1024-k2560, i4g512-n1024-k2560): c32u4 -> c16u16 (×1.164 [3] i4g32-n1024-k2560, ×1.156 [4] i4g32-n1024-k2560, ×1.172 [7] i4g32-n1024-k2560, ×1.168 [17] i4g32-n1024-k2560, ×1.100 [3] i4g512-n1024-k2560, ×1.102 [4] i4g512-n1024-k2560)
 *   - i4 n2560 k4096 (cases i4g32-n2560-k4096, i4g4096-n2560-k4096): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2560-k4096 adopts c32u16; i4g4096-n2560-k4096 adopts c16u16))
 *   - i8 n262144 k2560 (case i8-n262144-k2560): c32u4 -> c16u4 (×1.516 [3], ×1.496 [4], ×1.519 [7], ×1.347 [17])
 *   - i4 n262144 k2560 (case i4g32-n262144-k2560): c32u4 -> c16u2 (×1.704 [3], ×1.709 [4])
 *   - i4 n256 k5120 (case i4g32-n256-k5120): c32u4 -> c16u16 (×1.192 [3], ×1.192 [4], ×1.187 [8], ×1.186 [18])
 *   - i8 n256 k2560 (case i8-n256-k2560): c32u4 -> c16u16 (×1.205 [3], ×1.220 [4])
 *   - i8 n2560 k256 (case i8-n2560-k256): c32u4 -> c16u16 (×1.156 [3], ×1.157 [4])
 *   - i2 n262144 k2560 (case i2-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.002 < ×1.100 in [3])); c32u8: ×1.002 [3], ×0.996 [4]
 *   - i4 n15360 k3840 (case i4g32-n15360-k3840): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u1: ×0.990 < ×1.100 in [3])); c32u1: ×0.990 [3], ×0.990 [4]
 *   - i4 n8192 k3840 (case i4g32-n8192-k3840): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.051 < ×1.100 in [3])); c32u16: ×1.051 [3], ×1.041 [4]
 *   - i4 n4096 k3840 (case i4g32-n4096-k3840): c32u4 -> c32u16 (×1.213 [3], ×1.206 [4])
 *   - i4 n2048 k3840 (case i4g32-n2048-k3840): c32u4 -> c16u16 (×1.139 [3], ×1.146 [4])
 *   - i4 n512 k3840 (case i4g32-n512-k3840): c32u4 -> c16u16 (×1.140 [3], ×1.140 [4])
 *   - i4 n3840 k4096 (case i4g32-n3840-k4096): c32u4 -> c32u16 (×1.206 [3], ×1.239 [4])
 *   - i4 n3840 k8192 (case i4g32-n3840-k8192): c32u4 -> c32u16 (×1.279 [3], ×1.283 [4])
 *   - i4 n3840 k15360 (case i4g32-n3840-k15360): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; the reference c32u4's fastest time at M=1 ([4]) is repeated within ×1.100 by no second record; the reference c32u4's fastest time at M=1 ([4]) is repeated within ×1.100 by no second record; M=1 is not measured in [5]; M=1 is not measured in [6]; screened out by the quick look in [7] at M=1; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; screened out by the quick look in [17] at M=1; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24])))
 *   - i4 n262144 k3840 (case i4g32-n262144-k3840): c32u4 -> c16u2 (×1.577 [3], ×1.569 [4], ×1.566 [7], ×1.567 [17])
 *   - i4 n1024 k7680 (case i4g32-n1024-k7680): c32u4 -> c16u16 (×1.191 [3], ×1.189 [4])
 *   - i4 n4096 k1024 (case i4g32-n4096-k1024): c32u4 -> c16u16 (×1.157 [3], ×1.158 [4])
 *   - i4 n1024 k4096 (case i4g32-n1024-k4096): c32u4 -> c16u16 (×1.186 [3], ×1.188 [4])
 *   - i4 n8192 k1024 (case i4g32-n8192-k1024): c32u4 -> c32u16 (×1.132 [3], ×1.136 [4])
 *   - i4 n1024 k8192 (case i4g32-n1024-k8192): c32u4 -> c16u16 (×1.192 [3], ×1.190 [4])
 *   - i4 n3840 k1024 (case i4g32-n3840-k1024): c32u4 -> c32u16 (×1.154 [3], ×1.154 [4])
 *   - i4 n262144 k1024 (case i4g32-n262144-k1024): c32u4 -> c16u8 (×1.318 [3], ×1.314 [4])
 *   - i4 n8192 k2816 (case i4g32-n8192-k2816): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.055 < ×1.100 in [3])); c32u16: ×1.055 [3], ×1.053 [4], ×1.072 [7], ×1.072 [17]
 *   - i4 n4096 k2816 (case i4g32-n4096-k2816): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.040 < ×1.100 in [1])); c32u16: ×1.040 [1], ×1.047 [2], ×1.034 [3], ×1.035 [4]
 *   - i4 n2048 k2816 (case i4g32-n2048-k2816): c32u4 -> c16u16 (×1.108 [3], ×1.107 [4])
 *   - i4 n2112 k2816 (case i4g32-n2112-k2816): c32u4 -> c16u16 (×1.108 [3], ×1.108 [4])
 *   - i4 n1024 k2816 (case i4g32-n1024-k2816): c32u4 -> c16u16 (×1.112 [3], ×1.109 [4])
 *   - i4 n128 k2816 (case i4g32-n128-k2816): c32u4 -> c16u16 (×1.111 [3], ×1.119 [4])
 *   - i4 n2816 k4096 (case i4g32-n2816-k4096): c32u4 -> c32u16 (×1.207 [3], ×1.206 [4])
 *   - i4 n2816 k8192 (case i4g32-n2816-k8192): c32u4 -> c32u16 (×1.254 [3], ×1.255 [4])
 *   - i4 n2816 k2112 (case i4g32-n2816-k2112): c32u4 -> c16u16 (×1.161 [3], ×1.162 [4])
 *   - i4 n262144 k2816 (case i4g32-n262144-k2816): c32u4 -> c16u1 (×1.659 [3], ×1.662 [4])
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×0.942 < ×1.100 in [3])); c32u8: ×0.942 [3], ×0.941 [4]
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.893 < ×1.100 in [3])); c16u8: ×0.893 [3], ×0.889 [4], ×0.791 [12], ×0.937 [22]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.000 < ×1.100 in [3])); c32u16: ×1.000 [3], ×1.000 [4]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.060 < ×1.100 in [3])); c16u16: ×1.060 [3], ×1.062 [4]
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.960 < ×1.100 in [3])); c16u16: ×0.960 [3], ×0.931 [4]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.976 < ×1.100 in [3])); c16u8: ×0.976 [3], ×1.002 [4]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): keeps c32u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.949 < ×1.100 in [3])); c16u16: ×0.949 [3], ×0.952 [4]
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.949 < ×1.100 in [3])); c16u16: ×0.949 [3], ×0.952 [4]
 *   - bf16 n512 k1536 (case bf16-n512-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×0.907 < ×1.100 in [3])); c32u16: ×0.907 [3], ×1.024 [4]
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.002 < ×1.100 in [3])); c32u16: ×1.002 [3], ×1.002 [4]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×0.998 < ×1.100 in [3])); c32u8: ×0.998 [3], ×1.000 [4], ×0.985 [13], ×0.972 [23]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×0.982 < ×1.100 in [3])); c16u8: ×0.982 [3], ×0.982 [4]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): c32u4 -> c16u4 (×1.549 [3], ×1.547 [4])
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×0.977 < ×1.100 in [3])); c32u8: ×0.977 [3], ×0.986 [4]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×0.990 < ×1.100 in [3])); c32u16: ×0.990 [3], ×0.992 [4]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.002 < ×1.100 in [3])); c32u16: ×1.002 [3], ×1.004 [4]
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×0.999 < ×1.100 in [3])); c32u16: ×0.999 [3], ×0.980 [4]
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×0.977 < ×1.100 in [3])); c32u16: ×0.977 [3], ×0.904 [4]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): c32u4 -> c32u8 (×1.735 [3], ×1.730 [4])
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.010 < ×1.100 in [3])); c16u16: ×1.010 [3], ×1.003 [4]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.000 < ×1.100 in [3])); c32u16: ×1.000 [3], ×0.981 [4]
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.954 < ×1.100 in [3])); c16u16: ×0.954 [3], ×0.966 [4]
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×0.896 < ×1.100 in [3])); c32u16: ×0.896 [3], ×0.995 [4]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.984 < ×1.100 in [3])); c16u16: ×0.984 [3], ×0.978 [4]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×0.998 < ×1.100 in [3])); c16u4: ×0.998 [3], ×0.999 [4], ×0.975 [14], ×0.974 [24]
 * - reorder class (parallelShapes, M = 1 to 8): 49 adopted, 66 kept
 *   - i4 g32 n8960 k1536 (case i4g32-n8960-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l8: ×1.232 [3], ×1.192 [4]
 *   - i4 g32 n2048 k1536 (case i4g32-n2048-k1536): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: slower than the default's choice in [3] at M=4 (×0.952))); staged-l16: ×1.479 [3], ×1.454 [4]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]); 4 rows — slower than the reference's row block in [3] at M=4 (×0.934); 8 rows — slower than the reference's row block in [3] at M=4 (×0.934)
 *   - i4 g32 n256 k1536 (case i4g32-n256-k1536): keeps parallel-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×0.917 < ×1.100 in [3])); staged-l32: ×0.917 [3], ×0.915 [4]
 *   - i4 g32 n1536 k2048 (case i4g32-n1536-k2048): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.328 [1] M=4, ×1.342 [2] M=4, ×1.309 [3] M=4, ×1.298 [4] M=4, ×1.320 [1] M=8, ×1.230 [2] M=8, ×1.317 [3] M=8, ×1.302 [4] M=8
 *   - i4 g32 n6144 k1536 (case i4g32-n6144-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l16: ×1.308 [3], ×1.322 [4]
 *   - i4 g32 n1536 k6144 (case i4g32-n1536-k6144): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.908 < ×1.100 in [3])); staged-l16: ×0.908 [3], ×0.908 [4]
 *   - i4 g32 n1536 k256 (case i4g32-n1536-k256): parallel-l4 -> parallel-l4 with row blocks of 2; at the other row counts: ×1.067 [3] M=4, ×1.059 [4] M=4, ×1.300 [3] M=8, ×1.279 [4] M=8
 *   - i4 g32 n4096 k1536 (case i4g32-n4096-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; screened out by the quick look in [3] at M=1; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]))); staged-l16: ×1.374 [4]
 *   - i4 g32 n512 k1536 (case i4g32-n512-k1536): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.397 [3] M=4, ×1.391 [4] M=4, ×1.968 [3] M=8, ×1.953 [4] M=8
 *   - i4 g32 n1536 k4096 (case i4g32-n1536-k4096): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.374 [3] M=4, ×1.355 [4] M=4, ×1.305 [3] M=8, ×1.362 [4] M=8
 *   - i4 g32 n12288 k1536 (case i4g32-n12288-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l8: ×1.137 [3], ×1.170 [4]
 *   - i4 g32 n1536 k12288 (case i4g32-n1536-k12288): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.862 < ×1.100 in [3])); staged-l16: ×0.862 [3], ×0.858 [4]
 *   - i8 n262144 k1536 (case i8-n262144-k1536): parallel-l16 -> parallel-l16 with row blocks of 4; at the other row counts: ×2.601 [3] M=4, ×2.461 [4] M=4, ×2.523 [3] M=8, ×2.526 [4] M=8
 *   - i4 g32 n262144 k1536 (case i4g32-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l8: ×3.798 [3], ×3.802 [4]
 *   - i4 g32 n256 k3072 (case i4g32-n256-k3072): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.254 [3] M=4, ×1.253 [4] M=4, ×1.773 [3] M=8, ×1.762 [4] M=8
 *     - subgroup32-l32 meets the rule too (×1.104 [3], ×1.104 [4]): a Session option, not a row
 *   - i4 g32 n1024 k256 (case i4g32-n1024-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: slower than the default's choice in [3] at M=8 (×0.862))); parallel-l4: ×1.227 [3], ×1.228 [4]
 *   - i4 g32 n256 k1024 (case i4g32-n256-k1024): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.090 [3] M=4, ×1.087 [4] M=4, ×1.411 [3] M=8, ×1.411 [4] M=8
 *     - subgroup32-l32 meets the rule too (×1.135 [3], ×1.129 [4]): a Session option, not a row
 *   - i4 g32 n2048 k256 (case i4g32-n2048-k256): no row (c32u4) -> parallel-l4 with row blocks of 2 (×1.235 [3], ×1.226 [4]); at the other row counts: ×1.025 [3] M=4, ×1.026 [4] M=4, ×1.100 [3] M=8, ×1.105 [4] M=8
 *   - i4 g32 n256 k2048 (case i4g32-n256-k2048): no row (c32u4) -> parallel-l32 with row blocks of 2 (×4.873 [3], ×4.878 [4]); at the other row counts: ×3.589 [3] M=4, ×3.589 [4] M=4, ×2.932 [3] M=8, ×2.920 [4] M=8
 *     - subgroup32-l32 meets the rule too (×5.404 [3], ×5.408 [4]): a Session option, not a row
 *   - i4 g32 n262144 k256 (case i4g32-n262144-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l4: ×0.961 < ×1.100 in [3])); staged-l4: ×0.961 [3], ×0.961 [4]
 *   - i8 n256 k3072 (case i8-n256-k3072): no row (c32u4) -> parallel-l32 (×6.384 [3], ×6.432 [4]); at the other row counts: ×4.144 [3] M=4, ×4.153 [4] M=4, ×3.003 [3] M=8, ×2.998 [4] M=8
 *     - subgroup32-l32 meets the rule too (×6.844 [3], ×6.884 [4]): a Session option, not a row
 *   - i8 n1024 k256 (case i8-n1024-k256): no row (c32u4) -> parallel-l4 (×1.265 [3], ×1.269 [4]); at the other row counts: ×1.215 [3] M=4, ×1.215 [4] M=4, ×1.198 [3] M=8, ×1.200 [4] M=8
 *     - subgroup32-l8 meets the rule too (×1.463 [3], ×1.461 [4]): a Session option, not a row
 *   - i8 n256 k1024 (case i8-n256-k1024): no row (c32u4) -> parallel-l32 (×3.209 [3], ×3.227 [4]); at the other row counts: ×2.538 [3] M=4, ×2.597 [4] M=4, ×2.139 [3] M=8, ×2.139 [4] M=8
 *     - subgroup32-l32 meets the rule too (×3.544 [3], ×3.568 [4]): a Session option, not a row
 *   - i8 n2048 k256 (case i8-n2048-k256): no row (c32u4) -> parallel-l4 (×1.280 [3], ×1.278 [4]); at the other row counts: ×1.165 [3] M=4, ×1.162 [4] M=4, ×1.142 [3] M=8, ×1.133 [4] M=8
 *     - subgroup32-l4 meets the rule too (×1.327 [3], ×1.325 [4]): a Session option, not a row
 *   - i8 n256 k2048 (case i8-n256-k2048): no row (c32u4) -> parallel-l32 (×5.094 [3], ×5.092 [4]); at the other row counts: ×3.540 [3] M=4, ×3.544 [4] M=4, ×3.653 [6] M=4, ×3.527 [16] M=4, ×2.733 [3] M=8, ×2.727 [4] M=8
 *     - subgroup32-l32 meets the rule too (×5.545 [3], ×5.542 [4]): a Session option, not a row
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: ×1.031 < ×1.100 in [3])); parallel-l4: ×1.031 [3], ×1.033 [4]
 *   - i4 g512 n2048 k1536 (case i4g512-n2048-k1536): parallel-l4 -> staged-l16 with row blocks of 4 (×1.579 [3], ×1.580 [4]); at the other row counts: ×1.858 [3] M=4, ×1.859 [4] M=4, ×2.040 [3] M=8, ×2.041 [4] M=8
 *   - i4 g512 n256 k1536 (case i4g512-n256-k1536): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.060 [3] M=4, ×1.070 [4] M=4, ×1.363 [3] M=8, ×1.360 [4] M=8
 *   - i4 g2048 n1536 k2048 (case i4g2048-n1536-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i4 g512 n6144 k1536 (case i4g512-n6144-k1536): parallel-l4 -> staged-l16 with row blocks of 4 (×1.337 [3], ×1.343 [4]); at the other row counts: ×1.835 [3] M=4, ×1.797 [4] M=4, ×2.046 [3] M=8, ×2.023 [4] M=8
 *   - i4 g2048 n1536 k6144 (case i4g2048-n1536-k6144): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k1536 (case i8-n256-k1536): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.896 < ×1.100 in [3])); parallel-l16: ×0.896 [3], ×0.896 [4]
 *   - i8 n1536 k256 (case i8-n1536-k256): keeps parallel-l4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.032 < ×1.100 in [3])); parallel-l8: ×1.032 [3], ×1.029 [4]
 *   - i4 g512 n4096 k1536 (case i4g512-n4096-k1536): parallel-l4 -> staged-l16 with row blocks of 4 (×1.374 [3], ×1.373 [4]); at the other row counts: ×2.110 [3] M=4, ×2.044 [4] M=4, ×2.412 [3] M=8, ×2.369 [4] M=8
 *   - i4 g512 n512 k1536 (case i4g512-n512-k1536): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.422 [3] M=4, ×1.421 [4] M=4, ×1.967 [3] M=8, ×1.982 [4] M=8
 *   - i4 g4096 n1536 k4096 (case i4g4096-n1536-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n12288 k1536 (case i2-n12288-k1536): parallel-l2 -> staged-l4 with row blocks of 8 (×1.111 [3], ×1.125 [4]); at the other row counts: ×1.932 [3] M=4, ×1.853 [4] M=4, ×1.963 [3] M=8, ×1.961 [4] M=8
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k1536 (case i2-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l2: ×0.992 < ×1.100 in [3])); staged-l2: ×0.992 [3], ×1.007 [4]
 *   - i4 g32 n10752 k2560 (case i4g32-n10752-k2560): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l8: ×1.288 [3], ×1.305 [4]
 *   - i4 g32 n2048 k2560 (case i4g32-n2048-k2560): parallel-l4 with row blocks of 2 -> staged-l16 with row blocks of 4 (×1.668 [3], ×1.667 [4]); at the other row counts: ×1.554 [3] M=4, ×1.553 [4] M=4, ×1.402 [3] M=8, ×1.398 [4] M=8
 *   - i4 g32 n512 k2560 (case i4g32-n512-k2560): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.701 [3] M=4, ×1.705 [4] M=4, ×2.335 [3] M=8, ×2.372 [4] M=8
 *   - i4 g32 n256 k2560 (case i4g32-n256-k2560): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.266 [3] M=4, ×1.263 [4] M=4, ×1.548 [3] M=8, ×1.545 [4] M=8
 *   - i4 g32 n2560 k2048 (case i4g32-n2560-k2048): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.343 [3] M=4, ×1.331 [4] M=4, ×1.376 [3] M=8, ×1.378 [4] M=8
 *   - i4 g32 n10240 k2560 (case i4g32-n10240-k2560): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l8: ×1.297 [3], ×1.317 [4]
 *   - i4 g32 n2560 k10240 (case i4g32-n2560-k10240): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.911 < ×1.100 in [3])); staged-l16: ×0.911 [3], ×0.887 [4]
 *   - i4 g32 n2560 k256 (case i4g32-n2560-k256): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: ×0.946 < ×1.100 in [3])); staged-l8: ×0.946 [3], ×0.939 [4]
 *   - i4 g32 n4096 k2560 (case i4g32-n4096-k2560): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l16: ×1.551 [3], ×1.527 [4]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]); 4 rows — slower than the reference's row block in [3] at M=4 (×0.920); 8 rows — slower than the reference's row block in [3] at M=4 (×0.920)
 *   - i4 g32 n1024 k2560 (case i4g32-n1024-k2560): parallel-l32 with row blocks of 2 -> staged-l16 with row blocks of 2 (×1.225 [3], ×1.222 [4]); at the other row counts: ×1.113 [3] M=4, ×1.115 [4] M=4, ×1.268 [3] M=8, ×1.286 [4] M=8
 *   - i4 g32 n2560 k4096 (case i4g32-n2560-k4096): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.989 < ×1.100 in [3])); staged-l16: ×0.989 [3], ×0.988 [4]
 *   - i8 n262144 k2560 (case i8-n262144-k2560): parallel-l16 -> parallel-l16 with row blocks of 4; at the other row counts: ×2.589 [3] M=4, ×2.575 [4] M=4, ×2.607 [3] M=8, ×2.610 [4] M=8
 *   - i4 g32 n262144 k2560 (case i4g32-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l8: ×4.400 [3], ×4.395 [4]
 *   - i4 g32 n256 k5120 (case i4g32-n256-k5120): no row (c32u4) -> parallel-l32 with row blocks of 2 (×7.168 [3], ×7.167 [4], ×7.114 [8], ×7.135 [18]); at the other row counts: ×4.372 [3] M=4, ×4.399 [4] M=4, ×3.506 [3] M=8, ×3.507 [4] M=8, ×3.529 [9] M=8, ×3.526 [19] M=8
 *     - subgroup32-l32 meets the rule too (×7.815 [3], ×7.813 [4], ×7.784 [8], ×7.785 [18]): a Session option, not a row
 *   - i4 g512 n2048 k2560 (case i4g512-n2048-k2560): parallel-l4 -> staged-l16 with row blocks of 4 (×1.701 [3], ×1.706 [4]); at the other row counts: ×2.098 [3] M=4, ×2.122 [4] M=4, ×2.185 [3] M=8, ×2.174 [4] M=8
 *   - i4 g512 n512 k2560 (case i4g512-n512-k2560): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.718 [3] M=4, ×1.715 [4] M=4, ×2.350 [3] M=8, ×2.347 [4] M=8
 *   - i4 g2048 n2560 k2048 (case i4g2048-n2560-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i4 g512 n10240 k2560 (case i4g512-n10240-k2560): parallel-l4 -> staged-l8 with row blocks of 4 (×1.248 [1], ×1.238 [2], ×1.242 [3], ×1.249 [4]); at the other row counts: ×2.550 [1] M=4, ×2.397 [2] M=4, ×2.461 [3] M=4, ×2.484 [4] M=4, ×2.848 [1] M=8, ×2.773 [2] M=8, ×2.551 [3] M=8, ×2.549 [4] M=8
 *   - i4 g2048 n2560 k10240 (case i4g2048-n2560-k10240): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k2560 (case i8-n256-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; screened out by the quick look in [3] at M=1; screened out by the quick look in [4] at M=1; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24])))
 *   - i8 n2560 k256 (case i8-n2560-k256): keeps parallel-l4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: ×1.011 < ×1.100 in [3])); parallel-l8: ×1.011 [3], ×1.013 [4]
 *   - i4 g512 n4096 k2560 (case i4g512-n4096-k2560): parallel-l4 -> staged-l16 with row blocks of 4 (×1.446 [3], ×1.456 [4]); at the other row counts: ×2.174 [3] M=4, ×2.194 [4] M=4, ×2.487 [3] M=8, ×2.460 [4] M=8
 *   - i4 g512 n1024 k2560 (case i4g512-n1024-k2560): parallel-l32 -> staged-l32 with row blocks of 2 (×1.355 [3], ×1.359 [4]); at the other row counts: ×2.683 [3] M=4, ×2.671 [4] M=4, ×2.382 [3] M=8, ×2.486 [4] M=8
 *   - i4 g4096 n2560 k4096 (case i4g4096-n2560-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k2560 (case i2-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l4: ×1.073 < ×1.100 in [3])); staged-l4: ×1.073 [3], ×1.011 [4]
 *   - i4 g32 n15360 k3840 (case i4g32-n15360-k3840): parallel-l4 -> staged-l8 with row blocks of 8 (×1.450 [3], ×1.447 [4]); at the other row counts: ×2.798 [3] M=4, ×2.992 [4] M=4, ×3.288 [3] M=8, ×3.254 [4] M=8
 *   - i4 g32 n8192 k3840 (case i4g32-n8192-k3840): keeps parallel-l4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l8: ×1.458 [3], ×1.459 [4]
 *     - no other row block replaces the default's: 2 rows — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]); 4 rows — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]); 8 rows — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; the reference parallel-l4's fastest time at M=4 ([4]) is repeated within ×1.100 by no second record; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24])
 *   - i4 g32 n4096 k3840 (case i4g32-n4096-k3840): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l16: ×1.662 [3], ×1.675 [4]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]); 4 rows — slower than the reference's row block in [3] at M=4 (×0.919); 8 rows — slower than the reference's row block in [3] at M=4 (×0.919)
 *   - i4 g32 n2048 k3840 (case i4g32-n2048-k3840): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l32: ×1.773 [3], ×1.765 [4]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]); 4 rows — slower than the reference's row block in [3] at M=4 (×0.938); 8 rows — slower than the reference's row block in [3] at M=4 (×0.938)
 *   - i4 g32 n512 k3840 (case i4g32-n512-k3840): keeps parallel-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×1.020 < ×1.100 in [3])); staged-l32: ×1.020 [3], ×1.022 [4]
 *   - i4 g32 n3840 k4096 (case i4g32-n3840-k4096): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.409 [3] M=4, ×1.407 [4] M=4, ×1.439 [3] M=8, ×1.457 [4] M=8
 *   - i4 g32 n3840 k8192 (case i4g32-n3840-k8192): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×1.049 < ×1.100 in [3])); staged-l16: ×1.049 [3], ×1.052 [4]
 *   - i4 g32 n3840 k15360 (case i4g32-n3840-k15360): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.382 [3] M=4, ×1.257 [4] M=4, ×1.482 [3] M=8, ×1.380 [4] M=8
 *   - i4 g32 n262144 k3840 (case i4g32-n262144-k3840): parallel-l16 -> staged-l8 with row blocks of 4 (×2.256 [3], ×2.265 [4]); at the other row counts: ×4.567 [3] M=4, ×4.618 [4] M=4, ×4.847 [3] M=8, ×4.717 [4] M=8
 *   - i4 g32 n1024 k7680 (case i4g32-n1024-k7680): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.310 [3] M=4, ×1.315 [4] M=4, ×1.213 [3] M=8, ×1.206 [4] M=8
 *   - i4 g32 n4096 k1024 (case i4g32-n4096-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not measured in [1]; M=8 is not measured in [2]; screened out by the quick look in [3] at M=8; screened out by the quick look in [4] at M=8; M=8 is not measured in [5]; M=8 is not measured in [6]; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]))); staged-l16: ×1.771 [3], ×1.781 [4]
 *   - i4 g32 n1024 k4096 (case i4g32-n1024-k4096): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.347 [3] M=4, ×1.346 [4] M=4, ×1.271 [3] M=8, ×1.262 [4] M=8
 *   - i4 g32 n8192 k1024 (case i4g32-n8192-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l8: ×1.335 [3], ×1.336 [4]
 *   - i4 g32 n1024 k8192 (case i4g32-n1024-k8192): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.355 [3] M=4, ×1.384 [4] M=4, ×1.232 [3] M=8, ×1.304 [4] M=8
 *   - i4 g32 n3840 k1024 (case i4g32-n3840-k1024): no row (c32u4) -> staged-l16 with row blocks of 4 (×1.800 [3], ×1.800 [4]); at the other row counts: ×1.724 [3] M=4, ×1.729 [4] M=4, ×1.523 [3] M=8, ×1.609 [4] M=8
 *   - i4 g32 n262144 k1024 (case i4g32-n262144-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l4: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; screened out by the quick look in [11] at M=4; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; screened out by the quick look in [21] at M=4; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l4: ×2.030 [3], ×2.018 [4]
 *   - i4 g32 n8192 k2816 (case i4g32-n8192-k2816): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l8: ×1.403 [3], ×1.398 [4]
 *   - i4 g32 n4096 k2816 (case i4g32-n4096-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l16: ×1.576 [1], ×1.559 [2], ×1.564 [3], ×1.565 [4]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: screened out by the quick look in [1] at M=4; screened out by the quick look in [2] at M=4; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]); 4 rows — slower than the reference's row block in [1] at M=4 (×0.936); 8 rows — slower than the reference's row block in [1] at M=4 (×0.936)
 *   - i4 g32 n2048 k2816 (case i4g32-n2048-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: slower than the default's choice in [3] at M=4 (×0.863))); staged-l32: ×1.681 [3], ×1.669 [4]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]); 4 rows — slower than the reference's row block in [3] at M=4 (×0.949); 8 rows — slower than the reference's row block in [3] at M=4 (×0.949)
 *   - i4 g32 n2112 k2816 (case i4g32-n2112-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); staged-l32: ×1.618 [3], ×1.613 [4]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]); 4 rows — slower than the reference's row block in [3] at M=4 (×0.915); 8 rows — slower than the reference's row block in [3] at M=4 (×0.915)
 *   - i4 g32 n1024 k2816 (case i4g32-n1024-k2816): parallel-l32 -> staged-l32 with row blocks of 2 (×1.395 [3], ×1.392 [4]); at the other row counts: ×2.617 [3] M=4, ×2.604 [4] M=4, ×2.566 [3] M=8, ×2.636 [4] M=8
 *   - i4 g32 n128 k2816 (case i4g32-n128-k2816): parallel-l32 -> parallel-l32 with row blocks of 2; at the other row counts: ×1.001 [3] M=4, ×1.002 [4] M=4, ×1.420 [3] M=8, ×1.419 [4] M=8
 *   - i4 g32 n2816 k4096 (case i4g32-n2816-k4096): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.364 [3] M=4, ×1.397 [4] M=4, ×1.400 [3] M=8, ×1.405 [4] M=8
 *   - i4 g32 n2816 k8192 (case i4g32-n2816-k8192): staged-l32 -> staged-l32 with row blocks of 2; at the other row counts: ×1.300 [3] M=4, ×1.311 [4] M=4, ×1.364 [3] M=8, ×1.393 [4] M=8
 *   - i4 g32 n2816 k2112 (case i4g32-n2816-k2112): staged-l32 with row blocks of 2 -> staged-l16 with row blocks of 4 (×1.175 [3], ×1.177 [4]); at the other row counts: ×1.437 [3] M=4, ×1.455 [4] M=4, ×1.482 [3] M=8, ×1.587 [4] M=8
 *   - i4 g32 n262144 k2816 (case i4g32-n262144-k2816): parallel-l4 -> staged-l8 with row blocks of 8 (×1.782 [3], ×1.770 [4]); at the other row counts: ×3.765 [3] M=4, ×3.315 [4] M=4, ×4.057 [3] M=8, ×4.084 [4] M=8
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.046 < ×1.100 in [3])); parallel-l32: ×1.046 [3], ×1.046 [4]
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: slower than the default's choice in [3] at M=4 (×0.960))); parallel-l32: ×1.203 [3], ×1.202 [4]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.859 < ×1.100 in [3])); parallel-l16: ×0.859 [3], ×0.859 [4]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): no row (c16u8) -> parallel-l32 (×3.538 [3], ×3.540 [4]); at the other row counts: ×6.098 [3] M=4, ×6.118 [4] M=4, ×3.423 [3] M=8, ×3.411 [4] M=8
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.039 < ×1.100 in [3])); parallel-l16: ×1.039 [3], ×1.038 [4]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.071 < ×1.100 in [3])); parallel-l32: ×1.071 [3], ×1.076 [4]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.018 < ×1.100 in [3])); parallel-l16: ×1.018 [3], ×1.017 [4]
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.051 < ×1.100 in [3])); parallel-l16: ×1.051 [3], ×1.051 [4]
 *   - bf16 n512 k1536 (case bf16-n512-k1536): parallel-l16 -> parallel-l32 (×1.201 [3], ×1.202 [4]); at the other row counts: ×1.067 [3] M=4, ×1.069 [4] M=4, ×1.002 [3] M=8, ×1.003 [4] M=8
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.081 < ×1.100 in [3])); parallel-l32: ×1.081 [3], ×1.081 [4]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.087 < ×1.100 in [3])); parallel-l16: ×1.087 [3], ×1.087 [4], ×1.087 [13], ×1.087 [23]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.076 < ×1.100 in [3])); parallel-l32: ×1.076 [3], ×1.079 [4]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: slower than the default's choice in [3] at M=4 (×0.984))); parallel-l16: ×2.813 [3], ×2.813 [4]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not measured in [1]; M=8 is not measured in [2]; screened out by the quick look in [3] at M=8; screened out by the quick look in [4] at M=8; M=8 is not measured in [5]; M=8 is not measured in [6]; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]))); parallel-l32: ×1.107 [3], ×1.107 [4]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.079 < ×1.100 in [3])); parallel-l32: ×1.079 [3], ×1.079 [4]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; screened out by the quick look in [3] at M=1; screened out by the quick look in [4] at M=1; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24])))
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; screened out by the quick look in [3] at M=1; screened out by the quick look in [4] at M=1; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24])))
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; screened out by the quick look in [3] at M=4; screened out by the quick look in [4] at M=4; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]))); parallel-l32: ×1.239 [3], ×1.237 [4]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not measured in [1]; M=8 is not measured in [2]; screened out by the quick look in [3] at M=8; screened out by the quick look in [4] at M=8; M=8 is not measured in [5]; M=8 is not measured in [6]; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]))); parallel-l32: ×1.905 [3], ×1.903 [4]
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.017 < ×1.100 in [3])); parallel-l32: ×1.017 [3], ×1.011 [4]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): no row (c16u16) -> parallel-l8 (×1.106 [3], ×1.103 [4]); at the other row counts: ×2.737 [3] M=4, ×2.727 [4] M=4, ×1.641 [3] M=8, ×1.635 [4] M=8
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: slower than the default's choice in [3] at M=4 (×0.839))); parallel-l16: ×1.105 [3], ×1.107 [4]
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: slower than the default's choice in [3] at M=4 (×0.955))); parallel-l32: ×1.283 [3], ×1.284 [4]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: ×1.049 < ×1.100 in [3])); parallel-l32: ×1.049 [3], ×1.046 [4]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: slower than the default's choice in [3] at M=4 (×0.992))); parallel-l16: ×1.844 [3], ×1.844 [4], ×1.844 [14], ×1.843 [24]
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
        { storage: "i4", n: 256, k: 1536, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 1536, k: 6144, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 1536, k: 4096, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 12288, k: 1536, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 1536, k: 12288, variant: { cols: 32, unroll: 16 } },
        { storage: "i8", n: 262144, k: 1536, variant: { cols: 16, unroll: 4 } },
        { storage: "i4", n: 262144, k: 1536, variant: { cols: 16, unroll: 1 } },
        { storage: "i4", n: 256, k: 3072, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 256, k: 1024, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 256, k: 2048, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 256, k: 3072, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 1024, k: 256, variant: { cols: 32, unroll: 16 } },
        { storage: "i8", n: 256, k: 1024, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 2048, k: 256, variant: { cols: 32, unroll: 16 } },
        { storage: "i8", n: 256, k: 2048, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 256, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 1536, k: 256, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 2048, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 2560, k: 2048, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 2560, k: 10240, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 1024, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 262144, k: 2560, variant: { cols: 16, unroll: 4 } },
        { storage: "i4", n: 262144, k: 2560, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 256, k: 5120, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 2560, k: 256, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 4096, k: 3840, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 2048, k: 3840, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 512, k: 3840, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 3840, k: 4096, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 3840, k: 8192, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 262144, k: 3840, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 1024, k: 7680, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 4096, k: 1024, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 1024, k: 4096, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 8192, k: 1024, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 1024, k: 8192, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 3840, k: 1024, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 262144, k: 1024, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 2048, k: 2816, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 2112, k: 2816, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 1024, k: 2816, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 128, k: 2816, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 2816, k: 4096, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 2816, k: 8192, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 2816, k: 2112, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 262144, k: 2816, variant: { cols: 16, unroll: 1 } },
        { storage: "bf16", n: 262144, k: 1536, variant: { cols: 16, unroll: 4 } },
        { storage: "bf16", n: 10240, k: 2560, variant: { cols: 32, unroll: 8 } },
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
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
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
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 256,
        k: 1536,
        group: 512,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 6144,
        k: 1536,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 1536, lanes: 32, packedActivations: false },
      { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false },
      {
        storage: "i4",
        n: 4096,
        k: 1536,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 1536,
        group: 512,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      {
        storage: "i2",
        n: 12288,
        k: 1536,
        lanes: 4,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 8,
      },
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
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 2560,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 256,
        k: 2560,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2560,
        k: 2048,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
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
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
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
      { storage: "i8", n: 262144, k: 2560, lanes: 16, packedActivations: false, rowBlock: 4 },
      {
        storage: "i4",
        n: 2048,
        k: 2560,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 2560,
        group: 512,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "i4", n: 2560, k: 2048, group: 2048, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 10240,
        k: 2560,
        group: 512,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 2560, lanes: 32, packedActivations: false },
      { storage: "i8", n: 2560, k: 256, lanes: 4, packedActivations: false },
      {
        storage: "i4",
        n: 4096,
        k: 2560,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2560,
        group: 512,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      { storage: "i4", n: 2560, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 256,
        k: 3072,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 256,
        k: 1024,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 15360,
        k: 3840,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 8,
      },
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
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
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
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
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
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 262144,
        k: 2816,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 8,
      },
      { storage: "bf16", n: 2048, k: 1536, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 256, k: 1536, lanes: 32, packedActivations: false },
      { storage: "bf16", n: 6144, k: 1536, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 1536, k: 6144, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 1536, k: 256, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 4096, k: 1536, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 512, k: 1536, lanes: 32, packedActivations: false },
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
        n: 2048,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 256,
        k: 2048,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "i8", n: 256, k: 3072, lanes: 32, packedActivations: false },
      { storage: "i8", n: 1024, k: 256, lanes: 4, packedActivations: false },
      { storage: "i8", n: 256, k: 1024, lanes: 32, packedActivations: false },
      { storage: "i8", n: 2048, k: 256, lanes: 4, packedActivations: false },
      { storage: "i8", n: 256, k: 2048, lanes: 32, packedActivations: false },
      {
        storage: "i4",
        n: 256,
        k: 5120,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 1024,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      { storage: "bf16", n: 1536, k: 2048, lanes: 32, packedActivations: false },
      { storage: "bf16", n: 2560, k: 256, lanes: 8, packedActivations: false },
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
