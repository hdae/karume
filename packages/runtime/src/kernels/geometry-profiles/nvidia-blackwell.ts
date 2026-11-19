/**
 * Geometry profile `nvidia-blackwell` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for the adapter match: vendor nvidia, architecture blackwell.
 * The runtime only selects the table by the adapter's match; nothing is measured at run time.
 * To change a value, take a sweep with the workbench's `geometry-sweep` entry and regenerate the
 * table with `node tools/geometry-profiles/generate.ts nvidia-blackwell --records=<path>[,<path>…]`.
 * The GEMV choices of linear (`linearGemv`) come from records of the `gemv-sweep` entry
 * (`--gemv-records=<path>,<path>[,…]`).
 *
 * Derived in this generation: the GEMV choices of linear (GEMV sweep records).
 *
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T03-12-30.json (sha256 d5b1e89bc5bd3736b12b056b84ce0d044a14864c853e4cc43ed7146a0a58b957, 2026-11-18T03:12:41.307Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 3e17aef4915cd86bf70a4be46a57e7d83529e3f2 (dirty)
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T03-48-09.json (sha256 2442d08ec2c2f5a8fdab92bc7eeab07183c6733b0db1afbeb2dfdd5e085368aa, 2026-11-18T03:48:20.090Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 7834560b50cc76b7d47e2ae91ad4b25642b200f6
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-42-50.json (sha256 a1d20b372b9bf1617fdf3d4d9f5e104f9ae1b254fb2dc6a1e3c7154a350040b0, 2026-11-18T11:43:00.512Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-45-01.json (sha256 0fc89d3cd53ac01a4ef071846d63a893a5e0412ff98256222ef4bbd87d0236e0, 2026-11-18T11:45:11.863Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-47-49.json (sha256 0cfd66100be4375e2959e56af550c1ea02a15856623f2775ccb1de25c48bba42, 2026-11-18T11:47:58.934Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-50-33.json (sha256 520dc241e6f44b0aa851ecbe30bc8c345056df999de8da7f751aceb5eb712955, 2026-11-18T11:50:43.452Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-53-26.json (sha256 e23386896fade0498dbddcc25d0a80ca6a8319735445ea25fc67979f470f269c, 2026-11-18T11:53:36.238Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-56-37.json (sha256 da3378db31659a3d34f7e2c5db6ae24120b4a5abae66b75ceb1dde8f19d12f7f, 2026-11-18T11:56:46.999Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-05-42.json (sha256 602db3c39a37380d552c74e63621aece2ec8871607b898bc8139d88cb9b52295, 2026-11-18T12:05:52.524Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-15-56.json (sha256 bf89ba71fbeda791713b3cb0c5c0f34b0a6332afcce4f3e38a0bf03830ee8906, 2026-11-18T12:16:05.831Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-17-52.json (sha256 94f22c57906678d14ca5058bc72b9513e9759b1e073d0fa95a9b18087a00b1ac, 2026-11-18T12:18:02.079Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-18-51.json (sha256 dcc0ae4fad8d81e4e5c41b1f1d59382c440f912c150a02ca74177f5d6977ff03, 2026-11-18T12:19:01.093Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-22-44.json (sha256 d9af91db173ee40edec1a0a1cd60cf550990e20525b317f1ea89b17eab436286, 2026-11-18T12:22:54.271Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-24-34.json (sha256 cc0d14893ac824aa3aa6e45559d524fd27f5aa35e1eed63e5246f97742a516c0, 2026-11-18T12:24:44.107Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-26-18.json (sha256 96ab86a0ce5447de2de3cab5367ea02e7391aa18b444959bbb1be28aba6832c9, 2026-11-18T12:26:28.310Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-26-51.json (sha256 4d610552af744b78f441b4d4844da6235484cdbabd786d6c6ad58cad9fba1ccd, 2026-11-18T12:27:01.149Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-27-31.json (sha256 e08ac27f1aec427a1bb5f86e4e559287b54e61e8b8391c53d1de74cdadb10cc5, 2026-11-18T12:27:41.555Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-28-16.json (sha256 c9ede9ecdb7b793258b971c781be3758b3e6a5157c32b10925da878acef828ea, 2026-11-18T12:28:26.772Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-32-24.json (sha256 e62addccc27ea0a2025f29519bc4d4223f0f1fd3a8d51275df549f8d60904c61, 2026-11-18T12:32:33.886Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-35-02.json (sha256 bdcc497e29acd6987f2f734879ea84ccc8e4f69c9d4cd638e5d20139518e7083, 2026-11-18T12:35:11.843Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-37-46.json (sha256 9c0bb33946fc6c367459ff590513152ced15185484af71b0102d7f7d81e25932, 2026-11-18T12:37:56.596Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-39-39.json (sha256 3311051b968a0bcb2e96b571162946321fa1a483ed0edeebc419d754f21656f7, 2026-11-18T12:39:48.916Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-41-45.json (sha256 afb97fb05fb1158db55b1ff08107c633936bceef5f4f83429621200ba5652a92, 2026-11-18T12:41:55.441Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-41-55.json (sha256 85909421065f582604c0b80bdd67663cd7d774650ac1e155d117b0ca519eed2b, 2026-11-18T12:42:05.104Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-42-47.json (sha256 7c7e38b61816f7c336f538ad84ac878e957f9c26cee95bf094f6934d14a86c7a, 2026-11-18T12:42:57.675Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-43-49.json (sha256 dc73d1b41dbcc91db97473eacfcab1db1ce9eae6a9cd735f923fd521f2419dfd, 2026-11-18T12:43:59.659Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-44-14.json (sha256 d1e13ae48b4955fac327eb187947698136334e31147021e1b33819910e8d2861, 2026-11-18T12:44:24.495Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-44-42.json (sha256 f06eb322647a8db2cc4c76cb44c3370951db4c413f07bb20bad7b46f2e75ca4f, 2026-11-18T12:44:52.107Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-45-28.json (sha256 5b8191e8ca4b5c2fb0aa1c0a7dc006ca9274a7ec77a03f03d71dfc53f321630e, 2026-11-18T12:45:38.127Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-46-22.json (sha256 30a2cae07b49cb5a798406d2b5bcc9c566985b9a49069f5f2c5997cc1cff0847, 2026-11-18T12:46:32.115Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-47-31.json (sha256 a5c3752eefd2522cb785d82719ca2a9a3c11a2ebc9bfee7809ab877897f49283, 2026-11-18T12:47:41.602Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-48-24.json (sha256 ee9f05a86e1b751be0be393ccd61d7805b880e4948b8cdf7f344fd7c05289847, 2026-11-18T12:48:33.862Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-49-11.json (sha256 4c8d488efe4c19e445562d481a8ec2de736c2a08fd02e75b5f5ed81039b8a4e4, 2026-11-18T12:49:21.226Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-50-06.json (sha256 928569242db42b3aed1e3a26679f7d3d25364044ca199b7853f58aa35868f8dc, 2026-11-18T12:50:15.881Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-58-03.json (sha256 f0d908cdf62da18703eb6bc93c684078d27b0d4bae9d0e3867dec30e33d269b2, 2026-11-18T12:58:13.613Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-07-53.json (sha256 973c8fa08b3d007baf631395bc708c9fccb3714b5530f45f727ef5ecb9a26da9, 2026-11-18T13:08:03.426Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-09-44.json (sha256 a001e5a58075a2973e4e65ace1d52cc14eece17fec8f102ee451b84fb2ae0f57, 2026-11-18T13:09:54.671Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-11-18.json (sha256 017f438ce7197ac45822a8d64a13176e97ee81033e9fab3e93ecb95de1f26a7a, 2026-11-18T13:11:27.919Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-13-17.json (sha256 92e5076e29b813959b33ae39f6b84aa0b996551bd073979d76fce860134a8b7a, 2026-11-18T13:13:27.165Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-15-28.json (sha256 11890400fe7f56918ebf9ba7a99d05514763591bad057b8760f9544612f2fdb1, 2026-11-18T13:15:38.368Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-18-15.json (sha256 749d2c8d647056bdc9f266ecb75c08fc983317ac180dcea016eb6b798652492e, 2026-11-18T13:18:24.985Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-20-58.json (sha256 b7a4fd6a0a4a23a09c84e981d9262746eb6738d357fc0f6e85c8ee03e0e3e96a, 2026-11-18T13:21:08.586Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-23-51.json (sha256 45b46ed317f6878f8fc48ab7cb3c48efdaea425a7be3e4eff01bdcedb22b4be9, 2026-11-18T13:24:01.715Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-27-03.json (sha256 d9552a6bae23a5a49d509b45c096c155529bdad73d63b83bd57a297e0f089dbd, 2026-11-18T13:27:13.229Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-36-08.json (sha256 b0682f5eb9d54d8574a57a337e3cc248d008f8158f475fede2e26dca3f16fd37, 2026-11-18T13:36:18.456Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-46-21.json (sha256 309bfcaef21652d3bff22d0e9f653d94f3ef034fbd8ad178aca8a3abc8df99aa, 2026-11-18T13:46:31.538Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-48-21.json (sha256 e74eb7045aad3aed9f9c905b73fcff6d98e51b044108ec10486b781266617458, 2026-11-18T13:48:31.196Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-49-22.json (sha256 d526085434a651c9a30f16424b7f0f6cca520f728ec34cc1ae923f8fcb571fdb, 2026-11-18T13:49:32.200Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-53-17.json (sha256 cef65e5da5abdd3157be092e9c3626f58ab193c0340a7d678b576437d53a586c, 2026-11-18T13:53:26.808Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-55-08.json (sha256 7a37d285d156c856f9b090a57e9b23301b16b59bea8d145dd9831c70eefaf05f, 2026-11-18T13:55:17.939Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-56-53.json (sha256 da9e0f0bf43b62c6120d66a3084a841912b377ee732a484f513a67d5d5abe928, 2026-11-18T13:57:03.328Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-57-26.json (sha256 e9b733d3ffa23911ff8c3f79fc46b3eb78c171e5d9a18670edcc1f96b49218ff, 2026-11-18T13:57:36.088Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-58-06.json (sha256 2734ea3076154bdacc600ea25853bb43d45d79c349e566d1e8856c13a21759ba, 2026-11-18T13:58:16.009Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-58-51.json (sha256 8ac70ca88767b6687834fb9f7d6c7fa7920accf21610c63f8855456bbd223309, 2026-11-18T13:59:01.315Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-02-58.json (sha256 a491dd0358675cc6c13fe333d163447c5b894c0d4bbf8d9de631cb8af581f142, 2026-11-18T14:03:08.610Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-05-36.json (sha256 a7aa3a0a0a0f1aca0ebf043fbcc9779d5396a63ba6d7c7824b233bf104f81bdd, 2026-11-18T14:05:46.384Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-08-21.json (sha256 58f95ed81ec704d4132e5db7685fa34707e9348134ce729f894471347364511d, 2026-11-18T14:08:31.647Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-10-13.json (sha256 11c0ef38f03c8b91aa749e02eb1156856330cf055b09e52c8a7667a55da88011, 2026-11-18T14:10:23.206Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-12-19.json (sha256 f0004588141e562414a4f1ca967f6a9b6fb15752250c92ee6d82c8181c002c35, 2026-11-18T14:12:29.391Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-12-29.json (sha256 54b70497a87c2079cc3794aebce939cb08b45e9fa94f60d6cd152255880da2fa, 2026-11-18T14:12:39.262Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-13-22.json (sha256 0303b9a16dc4b0689f68ad76759b74e8a808293e517ae16ae6648a67ca5f6932, 2026-11-18T14:13:32.113Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-14-24.json (sha256 656eac0f7053c8328203b979635d24e8ee0a05958c88df8557d89866a7a78c88, 2026-11-18T14:14:33.893Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-14-48.json (sha256 e24c99d996f7f0488658d88c87861db61b3df8b57594f690fe1c20a72590df14, 2026-11-18T14:14:58.657Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-15-16.json (sha256 44126de8e978912ebced335878083c0538318a26213cd85131596290950957b1, 2026-11-18T14:15:26.126Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-16-03.json (sha256 ac151958249627a8844c24bd9532d490c500ffdba05a66ecffadf82a48219fd6, 2026-11-18T14:16:13.476Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-16-59.json (sha256 64c44f9c4ad0d039413859ffe2be6c739f51467c8597159be12d2fc34aee9f6c, 2026-11-18T14:17:08.721Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-18-10.json (sha256 d92322d1b32d0ad4929b3187eb46263127248e74d7a2cbc9517f88850e37db2b, 2026-11-18T14:18:20.372Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-19-03.json (sha256 d76560b261793b6a7f33b037b796526b1d442d8676f168f53e46bf2cd7a07e98, 2026-11-18T14:19:13.097Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-19-50.json (sha256 9534cdc85929c4253fd0a5ddbea99b0b8380f6017ee3618ae05022844866ce81, 2026-11-18T14:20:00.590Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-20-45.json (sha256 8b595f8043bbc5a0ff76a7911da85e5c165fc16d6b7df528f133f0f00bb770e9, 2026-11-18T14:20:55.320Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-28-44.json (sha256 b931aff5dc38b3c0b4d02fecd72c1692a9386a8b29bc1d32819050cdf5c739ed, 2026-11-18T14:28:53.871Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-39-19.json (sha256 094fd64b2cd00c8ceffa242dc4ca6b8ca350b38513221a3fbe67329a2698ab0d, 2026-11-18T14:39:29.511Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-41-11.json (sha256 3b7c0f7b90d3bd9dd45e72b585526a49a10a8645be19e19a133bf56a328e4513, 2026-11-18T14:41:21.118Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-42-44.json (sha256 16081777d39506ca2c2ddf28b949642c640dfce7c40f704edab4d3cd6426da25, 2026-11-18T14:42:54.524Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 72f2ada9186d7613faafd8c99d36d04197e8a110
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-31-11.json (sha256 bf562c417200b6cd03c78c3a2bf4c547d71af019a1c8f538e8c4b2caacbee00d, 2026-11-18T16:31:21.502Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 2fcacc4ea7daeb591de977b97f54570d45bb3fc2
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-35-04.json (sha256 272b702bfa42f7496a1884bc2a0c3ec7c89c22bfd6d0bbb0bbf75f0e815bac9d, 2026-11-18T16:35:14.423Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 2fcacc4ea7daeb591de977b97f54570d45bb3fc2
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-38-00.json (sha256 d141b659a8fc91fe683cfedd8f6dfa7239aa69257242d0e41f37923f20436aa8, 2026-11-18T16:38:10.246Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 2fcacc4ea7daeb591de977b97f54570d45bb3fc2
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-38-10.json (sha256 65cc04617b771212586355439b4fcc5d1eed468d2d093b515d6bea5e885a0c13, 2026-11-18T16:38:19.845Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 2fcacc4ea7daeb591de977b97f54570d45bb3fc2
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-38-19.json (sha256 860278dd26b516b51316b053ea34f4592d2f913a6e4fd2be9ec50c782b90a8c8, 2026-11-18T16:38:29.433Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 2fcacc4ea7daeb591de977b97f54570d45bb3fc2
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-42-14.json (sha256 6ee03e1f25e045cbfc464d0bd125084b7e6a06fc6f53108805c8a429a3b25a3c, 2026-11-18T16:42:24.158Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 2fcacc4ea7daeb591de977b97f54570d45bb3fc2
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-45-10.json (sha256 edaf5804a0c667b9b4b94436d818e1323541b096575fa9043174c4880fb66e96, 2026-11-18T16:45:19.589Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 2fcacc4ea7daeb591de977b97f54570d45bb3fc2
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-45-19.json (sha256 08921950d39fd7db30accbe0e1d328c27ebdfdd5aa79b857fc009ab660ce89c8, 2026-11-18T16:45:29.220Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 2fcacc4ea7daeb591de977b97f54570d45bb3fc2
 *
 * Carried over from the registered table at commit f0156ac9b42f627a50c3a4c9b5e98da55f8dd638: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d.
 * Their adoption lines are in `git show f0156ac9b42f627a50c3a4c9b5e98da55f8dd638:packages/runtime/src/kernels/geometry-profiles/nvidia-blackwell.ts`.
 *
 * GEMV choices: a candidate replaces the default profile's choice for a shape only when it is at
 * least ×1.100 faster than that choice in every GEMV sweep record, with its words equal to
 * the baseline's (identical class: the sequential variant's per-shape exceptions) or inside the
 * band (reorder class: the parallel rows, measured at every row count of the parallel range and
 * not slower at any) in every record; among those, the one whose smallest gain is largest. A shape
 * that a record measured while the baseline's rounds spread beyond ×1.250 keeps the default,
 * and so do the shared sequential keys whose cases disagree and the rows with packed activations.
 *
 * - linearGemv (the GEMV choices of linear), from 82 GEMV sweep records:
 *   - [1] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T03-12-30.json (2026-11-18T03:12:41.307Z, set full): 198 groups — 147 current and steady, 0 unsteady, 51 not current; 0 rows of current groups not current
 *     - 51 not current: the default profile's choices moved
 *   - [2] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T03-48-09.json (2026-11-18T03:48:20.090Z, set full): 198 groups — 147 current and steady, 0 unsteady, 51 not current; 0 rows of current groups not current
 *     - 51 not current: the default profile's choices moved
 *   - [3] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-42-50.json (2026-11-18T11:43:00.512Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [4] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-45-01.json (2026-11-18T11:45:11.863Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [5] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-47-49.json (2026-11-18T11:47:58.934Z, set full): 9 groups — 9 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [6] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-50-33.json (2026-11-18T11:50:43.452Z, set full): 9 groups — 8 current and steady, 1 unsteady, 0 not current; 0 rows of current groups not current
 *   - [7] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-53-26.json (2026-11-18T11:53:36.238Z, set full): 9 groups — 9 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [8] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-56-37.json (2026-11-18T11:56:46.999Z, set full): 9 groups — 9 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [9] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-05-42.json (2026-11-18T12:05:52.524Z, set full): 31 groups — 31 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [10] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-15-56.json (2026-11-18T12:16:05.831Z, set full): 31 groups — 31 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [11] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-17-52.json (2026-11-18T12:18:02.079Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [12] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-18-51.json (2026-11-18T12:19:01.093Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [13] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-22-44.json (2026-11-18T12:22:54.271Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [14] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-24-34.json (2026-11-18T12:24:44.107Z, set full): 5 groups — 5 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [15] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-26-18.json (2026-11-18T12:26:28.310Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [16] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-26-51.json (2026-11-18T12:27:01.149Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [17] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-27-31.json (2026-11-18T12:27:41.555Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [18] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-28-16.json (2026-11-18T12:28:26.772Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [19] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-32-24.json (2026-11-18T12:32:33.886Z, set full): 12 groups — 12 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [20] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-35-02.json (2026-11-18T12:35:11.843Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [21] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-37-46.json (2026-11-18T12:37:56.596Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [22] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-39-39.json (2026-11-18T12:39:48.916Z, set full): 5 groups — 5 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [23] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-41-45.json (2026-11-18T12:41:55.441Z, set full): 5 groups — 5 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [24] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-41-55.json (2026-11-18T12:42:05.104Z, set full): 1 groups — 1 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [25] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-42-47.json (2026-11-18T12:42:57.675Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [26] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-43-49.json (2026-11-18T12:43:59.659Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [27] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-44-14.json (2026-11-18T12:44:24.495Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [28] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-44-42.json (2026-11-18T12:44:52.107Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [29] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-45-28.json (2026-11-18T12:45:38.127Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [30] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-46-22.json (2026-11-18T12:46:32.115Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [31] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-47-31.json (2026-11-18T12:47:41.602Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [32] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-48-24.json (2026-11-18T12:48:33.862Z, set full): 3 groups — 3 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [33] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-49-11.json (2026-11-18T12:49:21.226Z, set full): 3 groups — 3 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [34] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-50-06.json (2026-11-18T12:50:15.881Z, set full): 3 groups — 3 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [35] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-58-03.json (2026-11-18T12:58:13.613Z, set full): 23 groups — 1 current and steady, 0 unsteady, 22 not current; 0 rows of current groups not current
 *     - 22 not current: the default profile's choices moved
 *   - [36] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-07-53.json (2026-11-18T13:08:03.426Z, set full): 46 groups — 12 current and steady, 0 unsteady, 34 not current; 0 rows of current groups not current
 *     - 34 not current: the default profile's choices moved
 *   - [37] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-09-44.json (2026-11-18T13:09:54.671Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [38] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-11-18.json (2026-11-18T13:11:27.919Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [39] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-13-17.json (2026-11-18T13:13:27.165Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [40] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-15-28.json (2026-11-18T13:15:38.368Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [41] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-18-15.json (2026-11-18T13:18:24.985Z, set full): 9 groups — 9 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [42] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-20-58.json (2026-11-18T13:21:08.586Z, set full): 9 groups — 9 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [43] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-23-51.json (2026-11-18T13:24:01.715Z, set full): 9 groups — 9 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [44] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-27-03.json (2026-11-18T13:27:13.229Z, set full): 9 groups — 9 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [45] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-36-08.json (2026-11-18T13:36:18.456Z, set full): 31 groups — 31 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [46] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-46-21.json (2026-11-18T13:46:31.538Z, set full): 31 groups — 31 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [47] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-48-21.json (2026-11-18T13:48:31.196Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [48] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-49-22.json (2026-11-18T13:49:32.200Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [49] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-53-17.json (2026-11-18T13:53:26.808Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [50] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-55-08.json (2026-11-18T13:55:17.939Z, set full): 5 groups — 5 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [51] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-56-53.json (2026-11-18T13:57:03.328Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [52] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-57-26.json (2026-11-18T13:57:36.088Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [53] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-58-06.json (2026-11-18T13:58:16.009Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [54] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-58-51.json (2026-11-18T13:59:01.315Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [55] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-02-58.json (2026-11-18T14:03:08.610Z, set full): 12 groups — 12 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [56] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-05-36.json (2026-11-18T14:05:46.384Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [57] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-08-21.json (2026-11-18T14:08:31.647Z, set full): 7 groups — 7 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [58] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-10-13.json (2026-11-18T14:10:23.206Z, set full): 5 groups — 5 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [59] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-12-19.json (2026-11-18T14:12:29.391Z, set full): 5 groups — 5 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [60] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-12-29.json (2026-11-18T14:12:39.262Z, set full): 1 groups — 1 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [61] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-13-22.json (2026-11-18T14:13:32.113Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [62] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-14-24.json (2026-11-18T14:14:33.893Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [63] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-14-48.json (2026-11-18T14:14:58.657Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [64] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-15-16.json (2026-11-18T14:15:26.126Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [65] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-16-03.json (2026-11-18T14:16:13.476Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [66] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-16-59.json (2026-11-18T14:17:08.721Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [67] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-18-10.json (2026-11-18T14:18:20.372Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [68] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-19-03.json (2026-11-18T14:19:13.097Z, set full): 3 groups — 3 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [69] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-19-50.json (2026-11-18T14:20:00.590Z, set full): 3 groups — 3 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [70] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-20-45.json (2026-11-18T14:20:55.320Z, set full): 3 groups — 3 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [71] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-28-44.json (2026-11-18T14:28:53.871Z, set full): 23 groups — 1 current and steady, 0 unsteady, 22 not current; 0 rows of current groups not current
 *     - 22 not current: the default profile's choices moved
 *   - [72] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-39-19.json (2026-11-18T14:39:29.511Z, set full): 46 groups — 12 current and steady, 0 unsteady, 34 not current; 0 rows of current groups not current
 *     - 34 not current: the default profile's choices moved
 *   - [73] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-41-11.json (2026-11-18T14:41:21.118Z, set full): 2 groups — 2 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [74] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-42-44.json (2026-11-18T14:42:54.524Z, set full): 4 groups — 4 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [75] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-31-11.json (2026-11-18T16:31:21.502Z, set full): 1 groups — 1 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [76] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-35-04.json (2026-11-18T16:35:14.423Z, set full): 11 groups — 4 current and steady, 0 unsteady, 7 not current; 0 rows of current groups not current
 *     - 7 not current: the default profile's choices moved
 *   - [77] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-38-00.json (2026-11-18T16:38:10.246Z, set full): 10 groups — 1 current and steady, 0 unsteady, 9 not current; 0 rows of current groups not current
 *     - 9 not current: the default profile's choices moved
 *   - [78] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-38-10.json (2026-11-18T16:38:19.845Z, set full): 1 groups — 0 current and steady, 0 unsteady, 1 not current; 0 rows of current groups not current
 *     - 1 not current: the default profile's choices moved
 *   - [79] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-38-19.json (2026-11-18T16:38:29.433Z, set full): 1 groups — 1 current and steady, 0 unsteady, 0 not current; 0 rows of current groups not current
 *   - [80] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-42-14.json (2026-11-18T16:42:24.158Z, set full): 11 groups — 4 current and steady, 0 unsteady, 7 not current; 0 rows of current groups not current
 *     - 7 not current: the default profile's choices moved
 *   - [81] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-45-10.json (2026-11-18T16:45:19.589Z, set full): 10 groups — 1 current and steady, 0 unsteady, 9 not current; 0 rows of current groups not current
 *     - 9 not current: the default profile's choices moved
 *   - [82] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-45-19.json (2026-11-18T16:45:29.220Z, set full): 1 groups — 0 current and steady, 0 unsteady, 1 not current; 0 rows of current groups not current
 *     - 1 not current: the default profile's choices moved
 * - identical class (sequential.shapes, M = 1): 42 adopted, 57 kept
 *   - i4 n8960 k1536 (case i4g32-n8960-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.085 < ×1.100 in [1])); c32u8: ×1.085 [1], ×1.090 [2]
 *   - i4 n2048 k1536 (cases i4g32-n2048-k1536, i4g512-n2048-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2048-k1536 adopts c32u8; i4g512-n2048-k1536 adopts c16u8))
 *   - i4 n256 k1536 (cases i4g32-n256-k1536, i4g512-n256-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n256-k1536 adopts c32u8; i4g512-n256-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.095 < ×1.100 in [1])))
 *   - i4 n1536 k2048 (cases i4g32-n1536-k2048, i4g2048-n1536-k2048): c32u4 -> c16u8 (×1.101 [1] i4g32-n1536-k2048, ×1.106 [2] i4g32-n1536-k2048, ×1.108 [1] i4g2048-n1536-k2048, ×1.109 [2] i4g2048-n1536-k2048)
 *   - i4 n6144 k1536 (cases i4g32-n6144-k1536, i4g512-n6144-k1536): keeps c32u4 (i4g32-n6144-k1536: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.100 < ×1.100 in [2]); i4g512-n6144-k1536: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.082 < ×1.100 in [1]))
 *   - i4 n1536 k6144 (cases i4g32-n1536-k6144, i4g2048-n1536-k6144): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n1536-k6144 adopts c32u8; i4g2048-n1536-k6144 adopts c16u8))
 *   - i4 n1536 k256 (case i4g32-n1536-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.078 < ×1.100 in [1])); c32u8: ×1.078 [1], ×1.075 [2]
 *   - i4 n4096 k1536 (cases i4g32-n4096-k1536, i4g512-n4096-k1536): c32u4 -> c32u8 (×1.110 [1] i4g32-n4096-k1536, ×1.106 [2] i4g32-n4096-k1536, ×1.106 [1] i4g512-n4096-k1536, ×1.105 [2] i4g512-n4096-k1536)
 *   - i4 n512 k1536 (cases i4g32-n512-k1536, i4g512-n512-k1536): c32u4 -> c16u8 (×1.126 [1] i4g32-n512-k1536, ×1.125 [2] i4g32-n512-k1536, ×1.112 [1] i4g512-n512-k1536, ×1.118 [2] i4g512-n512-k1536)
 *   - i4 n1536 k4096 (cases i4g32-n1536-k4096, i4g4096-n1536-k4096): c32u4 -> c16u8 (×1.108 [1] i4g32-n1536-k4096, ×1.108 [2] i4g32-n1536-k4096, ×1.111 [1] i4g4096-n1536-k4096, ×1.110 [2] i4g4096-n1536-k4096)
 *   - i4 n12288 k1536 (case i4g32-n12288-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.043 < ×1.100 in [1])); c32u8: ×1.043 [1], ×1.043 [2]
 *   - i4 n1536 k12288 (case i4g32-n1536-k12288): c32u4 -> c32u8 (×1.112 [1], ×1.110 [2])
 *   - i8 n262144 k1536 (case i8-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not current in [1] (the default profile's choice at exact is c16u4 in the record, c32u4 now); M=1 is not current in [2] (the default profile's choice at exact is c16u4 in the record, c32u4 now); M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not measured in [35]; M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not measured in [71]; M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - i4 n262144 k1536 (case i4g32-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u16: ×1.045 < ×1.100 in [13])); c128u16: ×1.045 [13], ×1.028 [49]
 *   - i4 n256 k3072 (case i4g32-n256-k3072): c32u4 -> c32u8 (×1.105 [16], ×1.105 [52])
 *   - i4 n1024 k256 (case i4g32-n1024-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.090 < ×1.100 in [19])); c16u8: ×1.090 [19], ×1.079 [55]
 *   - i4 n256 k1024 (case i4g32-n256-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.093 < ×1.100 in [16])); c32u8: ×1.093 [16], ×1.094 [52]
 *   - i4 n2048 k256 (case i4g32-n2048-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.076 < ×1.100 in [19])); c32u8: ×1.076 [19], ×1.060 [55]
 *   - i4 n256 k2048 (case i4g32-n256-k2048): c32u4 -> c32u8 (×1.102 [19], ×1.104 [55])
 *   - i4 n262144 k256 (case i4g32-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.007 < ×1.100 in [13])); c32u8: ×1.007 [13], ×1.005 [49]
 *   - i8 n256 k3072 (case i8-n256-k3072): c32u4 -> c16u16 (×1.282 [19], ×1.274 [55])
 *   - i8 n1024 k256 (case i8-n1024-k256): c32u4 -> c16u16 (×1.200 [19], ×1.200 [55])
 *   - i8 n256 k1024 (case i8-n256-k1024): c32u4 -> c16u16 (×1.237 [19], ×1.252 [55])
 *   - i8 n2048 k256 (case i8-n2048-k256): c32u4 -> c16u16 (×1.160 [19], ×1.160 [55])
 *   - i8 n256 k2048 (case i8-n256-k2048): c32u4 -> c16u16 (×1.256 [19], ×1.251 [55])
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.004 < ×1.100 in [13])); c32u16: ×1.004 [13], ×1.003 [49]
 *   - i8 n256 k1536 (case i8-n256-k1536): c32u4 -> c16u16 (×1.262 [1], ×1.260 [2])
 *   - i8 n1536 k256 (case i8-n1536-k256): c32u4 -> c16u16 (×1.142 [1], ×1.149 [2])
 *   - i2 n12288 k1536 (case i2-n12288-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u2: ×0.960 < ×1.100 in [1])); c32u2: ×0.960 [1], ×0.961 [2]
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.001 < ×1.100 in [1])); c16u4: ×1.001 [1], ×1.000 [2]
 *   - i2 n262144 k1536 (case i2-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u1: ×0.979 < ×1.100 in [13])); c32u1: ×0.979 [13], ×0.980 [49]
 *   - i4 n10752 k2560 (case i4g32-n10752-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.063 < ×1.100 in [1])); c32u8: ×1.063 [1], ×1.064 [2]
 *   - i4 n2048 k2560 (cases i4g32-n2048-k2560, i4g512-n2048-k2560): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2048-k2560 adopts c32u8; i4g512-n2048-k2560 adopts c16u8))
 *   - i4 n512 k2560 (cases i4g32-n512-k2560, i4g512-n512-k2560): c32u4 -> c16u8 (×1.136 [1] i4g32-n512-k2560, ×1.136 [2] i4g32-n512-k2560, ×1.126 [1] i4g512-n512-k2560, ×1.126 [2] i4g512-n512-k2560)
 *   - i4 n256 k2560 (case i4g32-n256-k2560): c32u4 -> c16u8 (×1.110 [1], ×1.111 [2])
 *   - i4 n2560 k2048 (cases i4g32-n2560-k2048, i4g2048-n2560-k2048): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2560-k2048 adopts c16u8; i4g2048-n2560-k2048 adopts c32u8))
 *   - i4 n10240 k2560 (cases i4g32-n10240-k2560, i4g512-n10240-k2560): keeps c32u4 (i4g32-n10240-k2560: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.060 < ×1.100 in [1]); i4g512-n10240-k2560: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.054 < ×1.100 in [1]))
 *   - i4 n2560 k10240 (cases i4g32-n2560-k10240, i4g2048-n2560-k10240): c32u4 -> c16u8 (×1.138 [1] i4g32-n2560-k10240, ×1.148 [2] i4g32-n2560-k10240, ×1.116 [1] i4g2048-n2560-k10240, ×1.113 [2] i4g2048-n2560-k10240)
 *   - i4 n2560 k256 (case i4g32-n2560-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.064 < ×1.100 in [1])); c32u8: ×1.064 [1], ×1.082 [2]
 *   - i4 n4096 k2560 (cases i4g32-n4096-k2560, i4g512-n4096-k2560): c32u4 -> c32u8 (×1.111 [1] i4g32-n4096-k2560, ×1.117 [2] i4g32-n4096-k2560, ×1.112 [1] i4g512-n4096-k2560, ×1.110 [2] i4g512-n4096-k2560)
 *   - i4 n1024 k2560 (cases i4g32-n1024-k2560, i4g512-n1024-k2560): c32u4 -> c16u8 (×1.133 [1] i4g32-n1024-k2560, ×1.133 [2] i4g32-n1024-k2560, ×1.123 [1] i4g512-n1024-k2560, ×1.123 [2] i4g512-n1024-k2560)
 *   - i4 n2560 k4096 (cases i4g32-n2560-k4096, i4g4096-n2560-k4096): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2560-k4096 adopts c16u8; i4g4096-n2560-k4096 adopts c32u8))
 *   - i8 n262144 k2560 (case i8-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u2: ×1.000 < ×1.100 in [1])); c32u2: ×1.000 [1], ×0.999 [2]
 *   - i4 n262144 k2560 (case i4g32-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u16: ×1.084 < ×1.100 in [13])); c128u16: ×1.084 [13], ×1.087 [49]
 *   - i4 n256 k5120 (case i4g32-n256-k5120): c32u4 -> c16u8 (×1.109 [19], ×1.108 [55])
 *   - i8 n256 k2560 (case i8-n256-k2560): c32u4 -> c16u16 (×1.298 [1], ×1.296 [2])
 *   - i8 n2560 k256 (case i8-n2560-k256): c32u4 -> c16u16 (×1.160 [1], ×1.160 [2])
 *   - i2 n262144 k2560 (case i2-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u1: ×0.992 < ×1.100 in [13])); c32u1: ×0.992 [13], ×0.993 [49]
 *   - i4 n15360 k3840 (case i4g32-n15360-k3840): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.039 < ×1.100 in [1])); c32u8: ×1.039 [1], ×1.040 [2]
 *   - i4 n8192 k3840 (case i4g32-n8192-k3840): c32u4 -> c32u8 (×1.113 [1], ×1.117 [2])
 *   - i4 n4096 k3840 (case i4g32-n4096-k3840): c32u4 -> c32u8 (×1.118 [1], ×1.115 [2])
 *   - i4 n2048 k3840 (case i4g32-n2048-k3840): c32u4 -> c32u8 (×1.113 [1], ×1.113 [2])
 *   - i4 n512 k3840 (case i4g32-n512-k3840): c32u4 -> c16u8 (×1.133 [1], ×1.133 [2])
 *   - i4 n3840 k4096 (case i4g32-n3840-k4096): c32u4 -> c32u8 (×1.115 [1], ×1.116 [2])
 *   - i4 n3840 k8192 (case i4g32-n3840-k8192): c32u4 -> c32u8 (×1.115 [1], ×1.115 [2])
 *   - i4 n3840 k15360 (case i4g32-n3840-k15360): c32u4 -> c32u8 (×1.120 [1], ×1.118 [2])
 *   - i4 n262144 k3840 (case i4g32-n262144-k3840): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.051 < ×1.100 in [1])); c32u16: ×1.051 [1], ×1.056 [2]
 *   - i4 n1024 k7680 (case i4g32-n1024-k7680): c32u4 -> c16u8 (×1.136 [32], ×1.136 [68])
 *   - i4 n4096 k1024 (case i4g32-n4096-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.095 < ×1.100 in [19])); c32u8: ×1.095 [19], ×1.095 [55]
 *   - i4 n1024 k4096 (case i4g32-n1024-k4096): c32u4 -> c16u8 (×1.129 [32], ×1.128 [68])
 *   - i4 n8192 k1024 (case i4g32-n8192-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.093 < ×1.100 in [19])); c32u8: ×1.093 [19], ×1.089 [55]
 *   - i4 n1024 k8192 (case i4g32-n1024-k8192): c32u4 -> c16u8 (×1.129 [32], ×1.130 [68])
 *   - i4 n3840 k1024 (case i4g32-n3840-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.093 < ×1.100 in [19])); c32u8: ×1.093 [19], ×1.098 [55]
 *   - i4 n262144 k1024 (case i4g32-n262144-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u16: ×1.023 < ×1.100 in [13])); c128u16: ×1.023 [13], ×1.024 [49]
 *   - i4 n8192 k2816 (case i4g32-n8192-k2816): c32u4 -> c32u8 (×1.106 [1], ×1.104 [2])
 *   - i4 n4096 k2816 (case i4g32-n4096-k2816): c32u4 -> c32u8 (×1.114 [1], ×1.110 [2])
 *   - i4 n2048 k2816 (case i4g32-n2048-k2816): c32u4 -> c32u8 (×1.108 [1], ×1.110 [2])
 *   - i4 n2112 k2816 (case i4g32-n2112-k2816): c32u4 -> c32u8 (×1.109 [1], ×1.107 [2])
 *   - i4 n1024 k2816 (case i4g32-n1024-k2816): c32u4 -> c16u8 (×1.130 [1], ×1.127 [2])
 *   - i4 n128 k2816 (case i4g32-n128-k2816): c32u4 -> c16u8 (×1.135 [1], ×1.132 [2])
 *   - i4 n2816 k4096 (case i4g32-n2816-k4096): c32u4 -> c16u8 (×1.139 [1], ×1.137 [2])
 *   - i4 n2816 k8192 (case i4g32-n2816-k8192): c32u4 -> c16u8 (×1.141 [1], ×1.144 [2])
 *   - i4 n2816 k2112 (case i4g32-n2816-k2112): c32u4 -> c32u8 (×1.148 [1], ×1.147 [2])
 *   - i4 n262144 k2816 (case i4g32-n262144-k2816): c32u4 -> c32u16 (×1.130 [1], ×1.129 [2])
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.044 < ×1.100 in [76])); c32u16: ×1.044 [76], ×1.045 [80]
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [82])))
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.952 < ×1.100 in [76])); c16u16: ×0.952 [76], ×0.950 [80]
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n1536 k256 (case bf16-n1536-k256): keeps c32u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c32u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c32u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not measured in [77]; M=1 is not current in [78] (the default profile's choice at reorder is c32u16 in the record, parallel-l8 now); M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not measured in [81]; M=1 is not current in [82] (the default profile's choice at reorder is c32u16 in the record, parallel-l8 now))))
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n512 k1536 (case bf16-n512-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, staged-l32 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, staged-l32 now); M=1 is not measured in [82])))
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.011 < ×1.100 in [76])); c16u16: ×1.011 [76], ×1.012 [80]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.028 < ×1.100 in [37])); c16u8: ×1.028 [37], ×1.028 [73]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.053 < ×1.100 in [76])); c16u16: ×1.053 [76], ×1.052 [80]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, staged-l32 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, staged-l32 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [82])))
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [82])))
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.012 < ×1.100 in [35])); c32u16: ×1.012 [35], ×1.019 [71]
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l16 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l16 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n2560 k256 (case bf16-n2560-k256): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.009 < ×1.100 in [77])); c16u8: ×1.009 [77], ×1.004 [81]
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l16 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l16 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: ×1.022 < ×1.100 in [37])); c16u1: ×1.022 [37], ×1.021 [73]
 * - reorder class (parallelShapes, M = 1 to 8): 61 adopted, 54 kept
 *   - i4 g32 n8960 k1536 (case i4g32-n8960-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); M=4 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); staged-l8: ×1.676 [1], ×1.664 [2]
 *   - i4 g32 n2048 k1536 (case i4g32-n2048-k1536): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r2 now); M=8 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r2 now); M=8 is not measured in [3]; M=8 is not measured in [4]; M=8 is not measured in [5]; screened out by the quick look in [6] at M=8; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]; M=8 is not measured in [25]; M=8 is not measured in [26]; M=8 is not measured in [27]; M=8 is not measured in [28]; M=8 is not measured in [29]; M=8 is not measured in [30]; M=8 is not measured in [31]; M=8 is not measured in [32]; M=8 is not measured in [33]; M=8 is not measured in [34]; M=8 is not measured in [35]; M=8 is not measured in [36]; M=8 is not measured in [37]; M=8 is not measured in [38]; M=8 is not measured in [39]; M=8 is not measured in [40]; M=8 is not measured in [41]; screened out by the quick look in [42] at M=8; M=8 is not measured in [43]; M=8 is not measured in [44]; M=8 is not measured in [45]; M=8 is not measured in [46]; M=8 is not measured in [47]; M=8 is not measured in [48]; M=8 is not measured in [49]; M=8 is not measured in [50]; M=8 is not measured in [51]; M=8 is not measured in [52]; M=8 is not measured in [53]; M=8 is not measured in [54]; M=8 is not measured in [55]; M=8 is not measured in [56]; M=8 is not measured in [57]; M=8 is not measured in [58]; M=8 is not measured in [59]; M=8 is not measured in [60]; M=8 is not measured in [61]; M=8 is not measured in [62]; M=8 is not measured in [63]; M=8 is not measured in [64]; M=8 is not measured in [65]; M=8 is not measured in [66]; M=8 is not measured in [67]; M=8 is not measured in [68]; M=8 is not measured in [69]; M=8 is not measured in [70]; M=8 is not measured in [71]; M=8 is not measured in [72]; M=8 is not measured in [73]; M=8 is not measured in [74]; M=8 is not measured in [75]; M=8 is not measured in [76]; M=8 is not measured in [77]; M=8 is not measured in [78]; M=8 is not measured in [79]; M=8 is not measured in [80]; M=8 is not measured in [81]; M=8 is not measured in [82]))); staged-l16: ×2.186 [1], ×2.184 [2]
 *   - i4 g32 n256 k1536 (case i4g32-n256-k1536): parallel-l32 with row blocks of 2 -> staged-l16 (×1.455 [1], ×1.446 [2]); at the other row counts: ×1.211 [7] M=4, ×1.216 [43] M=4, ×1.247 [8] M=8, ×1.237 [44] M=8
 *   - i4 g32 n1536 k2048 (case i4g32-n1536-k2048): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.955 < ×1.100 in [1])); staged-l16: ×0.955 [1], ×0.962 [2]
 *   - i4 g32 n6144 k1536 (case i4g32-n6144-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); M=4 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); staged-l16: ×1.765 [1], ×1.782 [2]
 *   - i4 g32 n1536 k6144 (case i4g32-n1536-k6144): staged-l32 with row blocks of 2 -> staged-l32; at the other row counts: ×1.160 [7] M=4, ×1.160 [43] M=4, ×1.042 [8] M=8, ×1.043 [44] M=8
 *   - i4 g32 n1536 k256 (case i4g32-n1536-k256): parallel-l4 -> staged-l8 (×1.187 [1], ×1.164 [2]); at the other row counts: ×1.202 [1] M=4, ×1.212 [2] M=4, ×1.407 [1] M=8, ×1.401 [2] M=8
 *   - i4 g32 n4096 k1536 (case i4g32-n4096-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: slower than the default's choice in [3] at M=4 (×0.891))); staged-l16: ×1.792 [1], ×1.795 [2]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); M=4 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]); 2 rows — slower than the reference's row block in [4] at M=8 (×0.955); 8 rows — slower than the reference's row block in [4] at M=8 (×0.887)
 *   - i4 g32 n512 k1536 (case i4g32-n512-k1536): parallel-l32 -> staged-l32 (×2.264 [1], ×2.257 [2]); at the other row counts: ×4.658 [1] M=4, ×4.647 [2] M=4, ×5.429 [1] M=8, ×5.423 [2] M=8
 *   - i4 g32 n1536 k4096 (case i4g32-n1536-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.935 < ×1.100 in [1])); staged-l16: ×0.935 [1], ×0.936 [2]
 *   - i4 g32 n12288 k1536 (case i4g32-n12288-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); M=4 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); staged-l8: ×1.906 [1], ×1.898 [2]
 *   - i4 g32 n1536 k12288 (case i4g32-n1536-k12288): staged-l32 with row blocks of 2 -> staged-l32; at the other row counts: ×1.164 [7] M=4, ×1.164 [43] M=4, ×1.017 [8] M=8, ×1.017 [44] M=8
 *   - i8 n262144 k1536 (case i8-n262144-k1536): parallel-l16 -> parallel-l16 with row blocks of 4; at the other row counts: ×3.584 [11] M=4, ×3.568 [47] M=4, ×3.578 [12] M=8, ×3.581 [48] M=8
 *   - i4 g32 n262144 k1536 (case i4g32-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; screened out by the quick look in [14] at M=4; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; screened out by the quick look in [50] at M=4; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); staged-l8: ×1.570 [13], ×1.544 [49]
 *   - i4 g32 n256 k3072 (case i4g32-n256-k3072): parallel-l32 -> staged-l32 (×1.926 [16], ×1.924 [52]); at the other row counts: ×4.798 [17] M=4, ×4.826 [53] M=4, ×6.486 [18] M=8, ×6.648 [54] M=8
 *   - i4 g32 n1024 k256 (case i4g32-n1024-k256): no row (c32u4) -> staged-l8 (×1.772 [19], ×1.789 [55]); at the other row counts: ×1.640 [20] M=4, ×1.621 [56] M=4, ×1.521 [21] M=8, ×1.521 [57] M=8
 *     - subgroup32-l4 meets the rule too (×1.555 [19], ×1.574 [55]): a Session option, not a row
 *   - i4 g32 n256 k1024 (case i4g32-n256-k1024): parallel-l32 -> staged-l32 (×1.479 [16], ×1.481 [52]); at the other row counts: ×3.195 [17] M=4, ×3.188 [53] M=4, ×4.260 [18] M=8, ×4.288 [54] M=8
 *   - i4 g32 n2048 k256 (case i4g32-n2048-k256): no row (c32u4) -> staged-l8 with row blocks of 2 (×1.748 [19], ×1.739 [55]); at the other row counts: ×1.676 [20] M=4, ×1.675 [56] M=4, ×1.877 [21] M=8, ×1.893 [57] M=8
 *     - subgroup32-l4 meets the rule too (×1.510 [19], ×1.531 [55]): a Session option, not a row
 *   - i4 g32 n256 k2048 (case i4g32-n256-k2048): no row (c32u4) -> staged-l32 (×6.716 [19], ×6.713 [55]); at the other row counts: ×5.360 [20] M=4, ×5.376 [56] M=4, ×3.977 [21] M=8, ×3.979 [57] M=8
 *   - i4 g32 n262144 k256 (case i4g32-n262144-k256): no row (c32u4) -> staged-l4 with row blocks of 8 (×1.970 [13], ×1.972 [49]); at the other row counts: ×1.399 [14] M=4, ×1.401 [50] M=4, ×1.446 [15] M=8, ×1.440 [51] M=8
 *   - i8 n256 k3072 (case i8-n256-k3072): no row (c32u4) -> staged-l32 (×8.256 [19], ×8.159 [55]); at the other row counts: ×6.273 [22] M=4, ×6.288 [58] M=4, ×4.561 [23] M=8, ×4.556 [59] M=8
 *     - subgroup32-l32 meets the rule too (×7.438 [19], ×7.419 [55]): a Session option, not a row
 *   - i8 n1024 k256 (case i8-n1024-k256): no row (c32u4) -> staged-l16 with row blocks of 2 (×1.992 [19], ×2.000 [55]); at the other row counts: ×1.782 [22] M=4, ×1.786 [58] M=4, ×1.863 [23] M=8, ×1.862 [59] M=8
 *     - subgroup32-l8 meets the rule too (×1.992 [19], ×2.000 [55]): a Session option, not a row
 *   - i8 n256 k1024 (case i8-n256-k1024): no row (c32u4) -> staged-l32 (×5.278 [19], ×5.336 [55]); at the other row counts: ×4.418 [22] M=4, ×4.418 [58] M=4, ×3.229 [23] M=8, ×3.236 [59] M=8
 *     - subgroup32-l32 meets the rule too (×4.912 [19], ×4.934 [55]): a Session option, not a row
 *   - i8 n2048 k256 (case i8-n2048-k256): no row (c32u4) -> staged-l16 with row blocks of 2 (×1.884 [19], ×1.838 [55]); at the other row counts: ×1.867 [22] M=4, ×1.885 [58] M=4, ×1.939 [23] M=8, ×1.937 [59] M=8
 *     - subgroup32-l8 meets the rule too (×1.781 [19], ×1.778 [55]): a Session option, not a row
 *   - i8 n256 k2048 (case i8-n256-k2048): no row (c32u4) -> staged-l32 (×7.139 [19], ×7.164 [55]); at the other row counts: ×5.844 [22] M=4, ×5.808 [58] M=4, ×4.121 [23] M=8, ×4.103 [59] M=8
 *     - subgroup32-l32 meets the rule too (×6.560 [19], ×6.573 [55]): a Session option, not a row
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; screened out by the quick look in [14] at M=4; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; screened out by the quick look in [50] at M=4; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); parallel-l4: ×1.279 [13], ×1.277 [49]
 *   - i4 g512 n2048 k1536 (case i4g512-n2048-k1536): parallel-l4 -> staged-l16 (×2.146 [1], ×2.146 [2]); at the other row counts: ×1.649 [1] M=4, ×1.652 [2] M=4, ×1.805 [1] M=8, ×1.808 [2] M=8
 *   - i4 g512 n256 k1536 (case i4g512-n256-k1536): parallel-l32 -> staged-l32 (×1.578 [1], ×1.582 [2]); at the other row counts: ×3.335 [1] M=4, ×3.351 [2] M=4, ×4.482 [1] M=8, ×4.478 [2] M=8
 *   - i4 g2048 n1536 k2048 (case i4g2048-n1536-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i4 g512 n6144 k1536 (case i4g512-n6144-k1536): parallel-l4 -> staged-l16 (×1.730 [1], ×1.730 [2]); at the other row counts: ×1.654 [1] M=4, ×1.651 [2] M=4, ×1.680 [1] M=8, ×1.675 [2] M=8
 *   - i4 g2048 n1536 k6144 (case i4g2048-n1536-k6144): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k1536 (case i8-n256-k1536): parallel-l32 -> staged-l32 (×1.133 [1], ×1.133 [2]); at the other row counts: ×1.959 [1] M=4, ×1.961 [2] M=4, ×2.744 [1] M=8, ×2.745 [2] M=8
 *   - i8 n1536 k256 (case i8-n1536-k256): keeps parallel-l4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: slower than the default's choice in [1] at M=4 (×0.991))); staged-l16: ×1.188 [1], ×1.184 [2]
 *   - i4 g512 n4096 k1536 (case i4g512-n4096-k1536): parallel-l4 -> staged-l16 (×1.767 [1], ×1.765 [2]); at the other row counts: ×1.772 [1] M=4, ×1.771 [2] M=4, ×1.711 [1] M=8, ×1.713 [2] M=8
 *   - i4 g512 n512 k1536 (case i4g512-n512-k1536): parallel-l32 -> staged-l32 (×2.262 [1], ×2.279 [2]); at the other row counts: ×4.675 [1] M=4, ×4.669 [2] M=4, ×5.403 [1] M=8, ×5.397 [2] M=8
 *   - i4 g4096 n1536 k4096 (case i4g4096-n1536-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n12288 k1536 (case i2-n12288-k1536): parallel-l2 -> staged-l4 (×1.412 [1], ×1.417 [2]); at the other row counts: ×1.312 [1] M=4, ×1.303 [2] M=4, ×1.278 [1] M=8, ×1.278 [2] M=8
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k1536 (case i2-n262144-k1536): no row (c32u4) -> staged-l2 with row blocks of 4 (×1.531 [13], ×1.533 [49]); at the other row counts: ×2.246 [29] M=4, ×2.231 [65] M=4, ×2.261 [30] M=8, ×2.255 [66] M=8
 *   - i4 g32 n10752 k2560 (case i4g32-n10752-k2560): parallel-l4 with row blocks of 4 -> parallel-l4 with row blocks of 8; at the other row counts: ×1.000 [3] M=4, ×1.000 [39] M=4, ×1.144 [4] M=8, ×1.146 [40] M=8
 *   - i4 g32 n2048 k2560 (case i4g32-n2048-k2560): parallel-l4 with row blocks of 2 -> staged-l16 with row blocks of 2 (×2.457 [1], ×2.465 [2]); at the other row counts: ×1.443 [5] M=4, ×1.447 [41] M=4, ×1.027 [42] M=8, ×1.021 [75] M=8, ×1.021 [79] M=8
 *   - i4 g32 n512 k2560 (case i4g32-n512-k2560): parallel-l32 -> staged-l32 (×2.652 [1], ×2.647 [2]); at the other row counts: ×5.762 [1] M=4, ×5.763 [2] M=4, ×6.401 [1] M=8, ×6.415 [2] M=8
 *   - i4 g32 n256 k2560 (case i4g32-n256-k2560): parallel-l32 -> staged-l32 (×1.757 [1], ×1.758 [2]); at the other row counts: ×4.211 [1] M=4, ×4.209 [2] M=4, ×5.582 [1] M=8, ×5.584 [2] M=8
 *   - i4 g32 n2560 k2048 (case i4g32-n2560-k2048): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×1.068 < ×1.100 in [1])); staged-l16: ×1.068 [1], ×1.069 [2]
 *   - i4 g32 n10240 k2560 (case i4g32-n10240-k2560): parallel-l4 with row blocks of 4 -> parallel-l4 with row blocks of 8; at the other row counts: ×1.000 [3] M=4, ×1.000 [39] M=4, ×1.141 [4] M=8, ×1.140 [40] M=8
 *   - i4 g32 n2560 k10240 (case i4g32-n2560-k10240): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×1.021 < ×1.100 in [1])); staged-l16: ×1.021 [1], ×1.025 [2]
 *   - i4 g32 n2560 k256 (case i4g32-n2560-k256): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: slower than the default's choice in [5] at M=4 (×0.965))); staged-l8: ×1.274 [1], ×1.285 [2]
 *   - i4 g32 n4096 k2560 (case i4g32-n4096-k2560): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: slower than the default's choice in [5] at M=4 (×0.830))); staged-l16: ×1.875 [1], ×1.886 [2]
 *   - i4 g32 n1024 k2560 (case i4g32-n1024-k2560): parallel-l32 with row blocks of 2 -> staged-l16 with row blocks of 2 (×4.077 [1], ×4.066 [2]); at the other row counts: ×1.700 [7] M=4, ×1.708 [43] M=4, ×1.575 [8] M=8, ×1.571 [44] M=8
 *   - i4 g32 n2560 k4096 (case i4g32-n2560-k4096): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×1.064 < ×1.100 in [1])); staged-l16: ×1.064 [1], ×1.070 [2]
 *   - i8 n262144 k2560 (case i8-n262144-k2560): parallel-l16 -> staged-l16 (×1.230 [1], ×1.229 [2]); at the other row counts: ×1.234 [1] M=4, ×1.234 [2] M=4, ×1.236 [1] M=8, ×1.236 [2] M=8
 *     - subgroup32-l8 meets the rule too (×1.230 [1], ×1.230 [2]): a Session option, not a row
 *   - i4 g32 n262144 k2560 (case i4g32-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; screened out by the quick look in [14] at M=4; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; screened out by the quick look in [50] at M=4; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); staged-l8: ×1.605 [13], ×1.606 [49]
 *   - i4 g32 n256 k5120 (case i4g32-n256-k5120): no row (c32u4) -> staged-l32 (×9.210 [19], ×9.227 [55]); at the other row counts: ×7.302 [20] M=4, ×7.240 [56] M=4, ×4.909 [21] M=8, ×4.860 [57] M=8
 *   - i4 g512 n2048 k2560 (case i4g512-n2048-k2560): parallel-l4 -> staged-l16 (×2.435 [1], ×2.441 [2]); at the other row counts: ×1.747 [1] M=4, ×1.754 [2] M=4, ×1.916 [1] M=8, ×1.911 [2] M=8
 *   - i4 g512 n512 k2560 (case i4g512-n512-k2560): parallel-l32 -> staged-l32 (×2.679 [1], ×2.679 [2]); at the other row counts: ×5.814 [1] M=4, ×5.815 [2] M=4, ×6.446 [1] M=8, ×6.446 [2] M=8
 *   - i4 g2048 n2560 k2048 (case i4g2048-n2560-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i4 g512 n10240 k2560 (case i4g512-n10240-k2560): parallel-l4 -> staged-l16 (×1.816 [1], ×1.820 [2]); at the other row counts: ×1.809 [1] M=4, ×1.811 [2] M=4, ×1.758 [1] M=8, ×1.757 [2] M=8
 *   - i4 g2048 n2560 k10240 (case i4g2048-n2560-k10240): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k2560 (case i8-n256-k2560): parallel-l32 -> staged-l32 (×1.127 [1], ×1.127 [2]); at the other row counts: ×2.241 [1] M=4, ×2.238 [2] M=4, ×3.093 [1] M=8, ×3.086 [2] M=8
 *   - i8 n2560 k256 (case i8-n2560-k256): parallel-l4 -> staged-l8 (×1.131 [1], ×1.145 [2]); at the other row counts: ×1.072 [1] M=4, ×1.078 [2] M=4, ×1.096 [1] M=8, ×1.093 [2] M=8
 *   - i4 g512 n4096 k2560 (case i4g512-n4096-k2560): parallel-l4 -> staged-l16 (×1.851 [1], ×1.847 [2]); at the other row counts: ×1.883 [1] M=4, ×1.883 [2] M=4, ×1.805 [1] M=8, ×1.809 [2] M=8
 *   - i4 g512 n1024 k2560 (case i4g512-n1024-k2560): parallel-l32 -> staged-l32 (×4.478 [1], ×4.479 [2]); at the other row counts: ×6.497 [1] M=4, ×6.464 [2] M=4, ×7.332 [1] M=8, ×7.319 [2] M=8
 *   - i4 g4096 n2560 k4096 (case i4g4096-n2560-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k2560 (case i2-n262144-k2560): no row (c32u4) -> staged-l2 with row blocks of 4 (×1.604 [13], ×1.603 [49]); at the other row counts: ×2.283 [29] M=4, ×2.265 [65] M=4, ×2.294 [30] M=8, ×2.296 [66] M=8
 *   - i4 g32 n15360 k3840 (case i4g32-n15360-k3840): parallel-l4 -> staged-l16 (×1.720 [1], ×1.725 [2]); at the other row counts: ×1.718 [1] M=4, ×1.736 [2] M=4, ×1.723 [1] M=8, ×1.724 [2] M=8
 *   - i4 g32 n8192 k3840 (case i4g32-n8192-k3840): parallel-l4 -> staged-l8 (×1.735 [1], ×1.738 [2]); at the other row counts: ×1.956 [1] M=4, ×1.959 [2] M=4, ×2.013 [1] M=8, ×2.002 [2] M=8
 *   - i4 g32 n4096 k3840 (case i4g32-n4096-k3840): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: slower than the default's choice in [5] at M=4 (×0.833))); staged-l8: ×1.852 [1], ×1.856 [2]
 *   - i4 g32 n2048 k3840 (case i4g32-n2048-k3840): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r2 now); M=8 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r2 now); M=8 is not measured in [3]; M=8 is not measured in [4]; M=8 is not measured in [5]; screened out by the quick look in [6] at M=8; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]; M=8 is not measured in [25]; M=8 is not measured in [26]; M=8 is not measured in [27]; M=8 is not measured in [28]; M=8 is not measured in [29]; M=8 is not measured in [30]; M=8 is not measured in [31]; M=8 is not measured in [32]; M=8 is not measured in [33]; M=8 is not measured in [34]; M=8 is not measured in [35]; M=8 is not measured in [36]; M=8 is not measured in [37]; M=8 is not measured in [38]; M=8 is not measured in [39]; M=8 is not measured in [40]; M=8 is not measured in [41]; screened out by the quick look in [42] at M=8; M=8 is not measured in [43]; M=8 is not measured in [44]; M=8 is not measured in [45]; M=8 is not measured in [46]; M=8 is not measured in [47]; M=8 is not measured in [48]; M=8 is not measured in [49]; M=8 is not measured in [50]; M=8 is not measured in [51]; M=8 is not measured in [52]; M=8 is not measured in [53]; M=8 is not measured in [54]; M=8 is not measured in [55]; M=8 is not measured in [56]; M=8 is not measured in [57]; M=8 is not measured in [58]; M=8 is not measured in [59]; M=8 is not measured in [60]; M=8 is not measured in [61]; M=8 is not measured in [62]; M=8 is not measured in [63]; M=8 is not measured in [64]; M=8 is not measured in [65]; M=8 is not measured in [66]; M=8 is not measured in [67]; M=8 is not measured in [68]; M=8 is not measured in [69]; M=8 is not measured in [70]; M=8 is not measured in [71]; M=8 is not measured in [72]; M=8 is not measured in [73]; M=8 is not measured in [74]; M=8 is not measured in [75]; M=8 is not measured in [76]; M=8 is not measured in [77]; M=8 is not measured in [78]; M=8 is not measured in [79]; M=8 is not measured in [80]; M=8 is not measured in [81]; M=8 is not measured in [82]))); staged-l16: ×2.559 [1], ×2.559 [2]
 *   - i4 g32 n512 k3840 (case i4g32-n512-k3840): keeps parallel-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: slower than the default's choice in [7] at M=4 (×0.983))); staged-l32: ×2.962 [1], ×2.972 [2]
 *   - i4 g32 n3840 k4096 (case i4g32-n3840-k4096): staged-l32 -> staged-l16 (×1.207 [1], ×1.205 [2]); at the other row counts: ×1.212 [1] M=4, ×1.212 [2] M=4, ×1.223 [1] M=8, ×1.224 [2] M=8
 *   - i4 g32 n3840 k8192 (case i4g32-n3840-k8192): staged-l32 with row blocks of 2 -> staged-l16 with row blocks of 4 (×1.178 [1], ×1.179 [2]); at the other row counts: ×1.616 [7] M=4, ×1.612 [43] M=4, ×1.515 [8] M=8, ×1.516 [44] M=8
 *   - i4 g32 n3840 k15360 (case i4g32-n3840-k15360): staged-l32 -> staged-l16 (×1.137 [1], ×1.146 [2]); at the other row counts: ×1.149 [1] M=4, ×1.149 [2] M=4, ×1.162 [1] M=8, ×1.164 [2] M=8
 *   - i4 g32 n262144 k3840 (case i4g32-n262144-k3840): parallel-l16 -> staged-l8 (×3.547 [1], ×3.547 [2]); at the other row counts: ×3.571 [1] M=4, ×3.548 [2] M=4, ×3.558 [1] M=8, ×3.541 [2] M=8
 *   - i4 g32 n1024 k7680 (case i4g32-n1024-k7680): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; screened out by the quick look in [32] at M=1; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not measured in [35]; M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; screened out by the quick look in [68] at M=1; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not measured in [71]; M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - i4 g32 n4096 k1024 (case i4g32-n4096-k1024): no row (c32u4) -> staged-l16 with row blocks of 2 (×2.723 [19], ×2.696 [55]); at the other row counts: ×2.481 [20] M=4, ×2.483 [56] M=4, ×1.750 [21] M=8, ×1.762 [57] M=8
 *   - i4 g32 n1024 k4096 (case i4g32-n1024-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.821 < ×1.100 in [32])); staged-l16: ×0.821 [32], ×0.819 [68]
 *   - i4 g32 n8192 k1024 (case i4g32-n8192-k1024): no row (c32u4) -> staged-l8 with row blocks of 2 (×2.278 [19], ×2.286 [55]); at the other row counts: ×2.140 [20] M=4, ×2.155 [56] M=4, ×1.724 [21] M=8, ×1.721 [57] M=8
 *   - i4 g32 n1024 k8192 (case i4g32-n1024-k8192): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; screened out by the quick look in [32] at M=1; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not measured in [35]; M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; screened out by the quick look in [68] at M=1; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not measured in [71]; M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - i4 g32 n3840 k1024 (case i4g32-n3840-k1024): no row (c32u4) -> staged-l16 with row blocks of 2 (×2.871 [19], ×2.862 [55]); at the other row counts: ×2.280 [20] M=4, ×2.291 [56] M=4, ×2.558 [21] M=8, ×2.557 [57] M=8
 *   - i4 g32 n262144 k1024 (case i4g32-n262144-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l4: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; screened out by the quick look in [14] at M=4; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; screened out by the quick look in [50] at M=4; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); staged-l4: ×1.527 [13], ×1.535 [49]
 *   - i4 g32 n8192 k2816 (case i4g32-n8192-k2816): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); M=4 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); staged-l8: ×1.720 [1], ×1.717 [2]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); M=4 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r4 now); screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]); 2 rows — slower than the reference's row block in [3] at M=4 (×0.906); 8 rows — at no row count ×1.100 faster than the reference's row block in every group (at most ×1.095)
 *   - i4 g32 n4096 k2816 (case i4g32-n4096-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: slower than the default's choice in [5] at M=4 (×0.831))); staged-l8: ×1.807 [1], ×1.816 [2]
 *   - i4 g32 n2048 k2816 (case i4g32-n2048-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r2 now); M=8 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r2 now); M=8 is not measured in [3]; M=8 is not measured in [4]; M=8 is not measured in [5]; screened out by the quick look in [6] at M=8; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]; M=8 is not measured in [25]; M=8 is not measured in [26]; M=8 is not measured in [27]; M=8 is not measured in [28]; M=8 is not measured in [29]; M=8 is not measured in [30]; M=8 is not measured in [31]; M=8 is not measured in [32]; M=8 is not measured in [33]; M=8 is not measured in [34]; M=8 is not measured in [35]; M=8 is not measured in [36]; M=8 is not measured in [37]; M=8 is not measured in [38]; M=8 is not measured in [39]; M=8 is not measured in [40]; M=8 is not measured in [41]; screened out by the quick look in [42] at M=8; M=8 is not measured in [43]; M=8 is not measured in [44]; M=8 is not measured in [45]; M=8 is not measured in [46]; M=8 is not measured in [47]; M=8 is not measured in [48]; M=8 is not measured in [49]; M=8 is not measured in [50]; M=8 is not measured in [51]; M=8 is not measured in [52]; M=8 is not measured in [53]; M=8 is not measured in [54]; M=8 is not measured in [55]; M=8 is not measured in [56]; M=8 is not measured in [57]; M=8 is not measured in [58]; M=8 is not measured in [59]; M=8 is not measured in [60]; M=8 is not measured in [61]; M=8 is not measured in [62]; M=8 is not measured in [63]; M=8 is not measured in [64]; M=8 is not measured in [65]; M=8 is not measured in [66]; M=8 is not measured in [67]; M=8 is not measured in [68]; M=8 is not measured in [69]; M=8 is not measured in [70]; M=8 is not measured in [71]; M=8 is not measured in [72]; M=8 is not measured in [73]; M=8 is not measured in [74]; M=8 is not measured in [75]; M=8 is not measured in [76]; M=8 is not measured in [77]; M=8 is not measured in [78]; M=8 is not measured in [79]; M=8 is not measured in [80]; M=8 is not measured in [81]; M=8 is not measured in [82]))); staged-l16: ×2.386 [1], ×2.385 [2]
 *   - i4 g32 n2112 k2816 (case i4g32-n2112-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: M=8 is not current in [1] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r2 now); M=8 is not current in [2] (the default profile's choice at reorder is parallel-l4 in the record, staged-l4r2 now); M=8 is not measured in [3]; M=8 is not measured in [4]; M=8 is not measured in [5]; screened out by the quick look in [6] at M=8; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]; M=8 is not measured in [25]; M=8 is not measured in [26]; M=8 is not measured in [27]; M=8 is not measured in [28]; M=8 is not measured in [29]; M=8 is not measured in [30]; M=8 is not measured in [31]; M=8 is not measured in [32]; M=8 is not measured in [33]; M=8 is not measured in [34]; M=8 is not measured in [35]; M=8 is not measured in [36]; M=8 is not measured in [37]; M=8 is not measured in [38]; M=8 is not measured in [39]; M=8 is not measured in [40]; M=8 is not measured in [41]; screened out by the quick look in [42] at M=8; M=8 is not measured in [43]; M=8 is not measured in [44]; M=8 is not measured in [45]; M=8 is not measured in [46]; M=8 is not measured in [47]; M=8 is not measured in [48]; M=8 is not measured in [49]; M=8 is not measured in [50]; M=8 is not measured in [51]; M=8 is not measured in [52]; M=8 is not measured in [53]; M=8 is not measured in [54]; M=8 is not measured in [55]; M=8 is not measured in [56]; M=8 is not measured in [57]; M=8 is not measured in [58]; M=8 is not measured in [59]; M=8 is not measured in [60]; M=8 is not measured in [61]; M=8 is not measured in [62]; M=8 is not measured in [63]; M=8 is not measured in [64]; M=8 is not measured in [65]; M=8 is not measured in [66]; M=8 is not measured in [67]; M=8 is not measured in [68]; M=8 is not measured in [69]; M=8 is not measured in [70]; M=8 is not measured in [71]; M=8 is not measured in [72]; M=8 is not measured in [73]; M=8 is not measured in [74]; M=8 is not measured in [75]; M=8 is not measured in [76]; M=8 is not measured in [77]; M=8 is not measured in [78]; M=8 is not measured in [79]; M=8 is not measured in [80]; M=8 is not measured in [81]; M=8 is not measured in [82]))); staged-l16: ×2.386 [1], ×2.394 [2]
 *   - i4 g32 n1024 k2816 (case i4g32-n1024-k2816): parallel-l32 -> staged-l32 (×4.655 [1], ×4.629 [2]); at the other row counts: ×6.919 [1] M=4, ×6.878 [2] M=4, ×7.770 [1] M=8, ×7.739 [2] M=8
 *   - i4 g32 n128 k2816 (case i4g32-n128-k2816): parallel-l32 -> staged-l32 (×1.833 [1], ×1.840 [2]); at the other row counts: ×2.813 [1] M=4, ×2.863 [2] M=4, ×4.257 [1] M=8, ×4.326 [2] M=8
 *   - i4 g32 n2816 k4096 (case i4g32-n2816-k4096): staged-l32 -> staged-l16 (×1.120 [1], ×1.124 [2]); at the other row counts: ×1.208 [1] M=4, ×1.204 [2] M=4, ×1.225 [1] M=8, ×1.227 [2] M=8
 *   - i4 g32 n2816 k8192 (case i4g32-n2816-k8192): staged-l32 -> staged-l16 (×1.114 [1], ×1.108 [2]); at the other row counts: ×1.169 [1] M=4, ×1.170 [2] M=4, ×1.181 [1] M=8, ×1.186 [2] M=8
 *   - i4 g32 n2816 k2112 (case i4g32-n2816-k2112): staged-l32 with row blocks of 2 -> staged-l16 with row blocks of 2 (×1.197 [1], ×1.206 [2]); at the other row counts: ×1.506 [7] M=4, ×1.503 [43] M=4, ×1.701 [8] M=8, ×1.711 [44] M=8
 *   - i4 g32 n262144 k2816 (case i4g32-n262144-k2816): parallel-l4 -> staged-l8 (×1.315 [1], ×1.310 [2]); at the other row counts: ×1.302 [1] M=4, ×1.298 [2] M=4, ×1.293 [1] M=8, ×1.238 [2] M=8
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): no row (c16u8) -> parallel-l8 (×3.788 [76], ×3.787 [80]); at the other row counts: ×2.131 [36] M=4, ×2.135 [72] M=4, ×1.153 [36] M=8, ×1.155 [72] M=8
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [82])))
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): no row (c16u8) -> staged-l32 (×4.025 [76], ×4.045 [80]); at the other row counts: ×4.102 [36] M=4, ×4.093 [72] M=4, ×2.202 [36] M=8, ×2.202 [72] M=8
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n1536 k256 (case bf16-n1536-k256): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c32u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c32u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not measured in [77]; M=1 is not current in [78] (the default profile's choice at reorder is c32u16 in the record, parallel-l8 now); M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not measured in [81]; M=1 is not current in [82] (the default profile's choice at reorder is c32u16 in the record, parallel-l8 now))))
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n512 k1536 (case bf16-n512-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, staged-l32 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, staged-l32 now); M=1 is not measured in [82])))
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): no row (c16u8) -> parallel-l8 (×4.095 [76], ×4.084 [80]); at the other row counts: ×2.224 [36] M=4, ×2.236 [72] M=4, ×1.190 [36] M=8, ×1.193 [72] M=8
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; screened out by the quick look in [38] at M=4; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; screened out by the quick look in [74] at M=4; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); parallel-l16: ×1.527 [37], ×1.527 [73]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: slower than the default's choice in [36] at M=8 (×0.929))); parallel-l8: ×1.997 [76], ×1.985 [80]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, staged-l32 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, staged-l32 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [82])))
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l32 now); M=1 is not measured in [82])))
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): no row (c32u4) -> staged-l16 (×2.865 [35], ×2.856 [71]); at the other row counts: ×1.924 [36] M=4, ×1.949 [72] M=4, ×1.043 [36] M=8, ×1.047 [72] M=8
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l16 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l16 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n2560 k256 (case bf16-n2560-k256): no row (c16u16) -> parallel-l8 (×2.086 [77], ×2.073 [81]); at the other row counts: ×2.549 [36] M=4, ×2.503 [72] M=4, ×1.413 [36] M=8, ×1.385 [72] M=8
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l8 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u16 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not current in [77] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not current in [81] (the default profile's choice at reorder is c16u16 in the record, parallel-l16 now); M=1 is not measured in [82])))
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not current in [35] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not current in [71] (the default profile's choice at exact is c32u4 in the record, c16u8 now); M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not current in [76] (the default profile's choice at reorder is c16u8 in the record, parallel-l16 now); M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not current in [80] (the default profile's choice at reorder is c16u8 in the record, parallel-l16 now); M=1 is not measured in [81]; M=1 is not measured in [82])))
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; screened out by the quick look in [38] at M=4; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; screened out by the quick look in [74] at M=4; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]))); staged-l16: ×1.579 [37], ×1.579 [73]
 * - rowsThreadTarget: keeps 16384 (the full and row-blocks sets measure the row-block heights per case and row count, but the field is one thread target for every shape, and no rule turns the heights into it)
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const NVIDIA_BLACKWELL: GeometryProfile = {
  id: "nvidia-blackwell",
  match: { vendor: "nvidia", architecture: "blackwell" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 64, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 16, form: "tb-un" } },
    { maxRows: 512, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 16, form: "tas-tb-un" } },
    {
      maxRows: Number.POSITIVE_INFINITY,
      geometry: { regM: 8, regN: 8, wgX: 16, wgY: 16, form: "tb-un" },
    },
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
      { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 8, tileK: 32 } },
      {
        maxRows: 512,
        geometry: { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      },
      {
        maxRows: Number.POSITIVE_INFINITY,
        geometry: { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      },
    ],
    attentionQk: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
    attentionPv: { regM: 4, regN: 8, wgX: 16, wgY: 16, tileK: 32 },
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
        { storage: "i4", n: 1536, k: 2048, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 4096, k: 1536, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 512, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 1536, k: 4096, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 1536, k: 12288, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 256, k: 3072, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 256, k: 2048, variant: { cols: 32, unroll: 8 } },
        { storage: "i8", n: 256, k: 3072, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 1024, k: 256, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 256, k: 1024, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 2048, k: 256, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 256, k: 2048, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 256, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 1536, k: 256, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 512, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 256, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 2560, k: 10240, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 4096, k: 2560, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 1024, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 256, k: 5120, variant: { cols: 16, unroll: 8 } },
        { storage: "i8", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 2560, k: 256, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 8192, k: 3840, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 4096, k: 3840, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 2048, k: 3840, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 512, k: 3840, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 3840, k: 4096, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 3840, k: 8192, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 3840, k: 15360, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 1024, k: 7680, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 1024, k: 4096, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 1024, k: 8192, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 8192, k: 2816, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 4096, k: 2816, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 2048, k: 2816, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 2112, k: 2816, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 1024, k: 2816, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 128, k: 2816, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 2816, k: 4096, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 2816, k: 8192, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 2816, k: 2112, variant: { cols: 32, unroll: 8 } },
        { storage: "i4", n: 262144, k: 2816, variant: { cols: 32, unroll: 16 } },
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
        stagedActivations: true,
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
      },
      {
        storage: "i4",
        n: 1536,
        k: 256,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
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
        stagedActivations: true,
      },
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
      },
      {
        storage: "i4",
        n: 256,
        k: 1536,
        group: 512,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
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
      },
      { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
      {
        storage: "i8",
        n: 256,
        k: 1536,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false },
      {
        storage: "i4",
        n: 4096,
        k: 1536,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 512,
        k: 1536,
        group: 512,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      {
        storage: "i2",
        n: 12288,
        k: 1536,
        lanes: 4,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "i2", n: 1536, k: 12288, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 10752,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 8,
      },
      {
        storage: "i4",
        n: 2048,
        k: 2560,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 512,
        k: 2560,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 256,
        k: 2560,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
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
        rowBlock: 8,
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
      {
        storage: "i8",
        n: 262144,
        k: 2560,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 2048,
        k: 2560,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 512,
        k: 2560,
        group: 512,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
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
      },
      { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
      {
        storage: "i8",
        n: 256,
        k: 2560,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i8",
        n: 2560,
        k: 256,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 4096,
        k: 2560,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2560,
        group: 512,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "i4", n: 2560, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 256,
        k: 3072,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 256,
        k: 1024,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 15360,
        k: 3840,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 8192,
        k: 3840,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
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
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 4096,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 3840,
        k: 8192,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 3840,
        k: 15360,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 262144,
        k: 3840,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
      },
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
      {
        storage: "i4",
        n: 1024,
        k: 2816,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 128,
        k: 2816,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 2816,
        k: 4096,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 2816,
        k: 8192,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 2816,
        k: 2112,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 262144,
        k: 2816,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
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
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 2048,
        k: 256,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 256,
        k: 2048,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i4",
        n: 262144,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 8,
      },
      {
        storage: "i8",
        n: 256,
        k: 3072,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i8",
        n: 1024,
        k: 256,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i8",
        n: 256,
        k: 1024,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i8",
        n: 2048,
        k: 256,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i8",
        n: 256,
        k: 2048,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i2",
        n: 262144,
        k: 1536,
        lanes: 2,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 256,
        k: 5120,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      {
        storage: "i2",
        n: 262144,
        k: 2560,
        lanes: 2,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 4096,
        k: 1024,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 8192,
        k: 1024,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
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
        rowBlock: 2,
      },
      { storage: "bf16", n: 8960, k: 1536, lanes: 8, packedActivations: false },
      {
        storage: "bf16",
        n: 1536,
        k: 2048,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "bf16", n: 12288, k: 1536, lanes: 8, packedActivations: false },
      {
        storage: "bf16",
        n: 10240,
        k: 2560,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "bf16", n: 2560, k: 256, lanes: 8, packedActivations: false },
    ],
  },
  provenance: {
    adapter: {
      vendor: "nvidia",
      architecture: "blackwell",
      device: "0x2c05",
      description: "NVIDIA GeForce RTX 5070 Ti",
    },
    families: {
      gemm: {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "5beadfe7409496c8",
        caseSet: "e6ddd45226ab2155",
      },
      attention: {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "c3f4a946ccbf1df9",
        caseSet: "9ee211cf69674928",
      },
      conv2d: {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "82887096d13feac2",
        caseSet: "02186e45558e4151",
      },
      "i8a8-linear": {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "834e8806e13400a9",
        caseSet: "3d32dc6f10559af2",
      },
      "i8a8-attention": {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "b605a1cbd123789d",
        caseSet: "8ab1a86b13087640",
      },
      "i8a8-conv3d": {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "c91b3368411e71d9",
        caseSet: "27ce9da45b359b14",
      },
    },
  },
};
