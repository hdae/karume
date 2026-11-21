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
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-19T12-43-27.json (sha256 5c60bd5cc3cdf6dd840124d4b67d4b34e4b5144131252924fc858e352c817c96, 2026-11-19T12:43:38.554Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout b527f471d990d0f5b216b8aba20f2888749e86cd
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-19T12-55-39.json (sha256 b00068bca97ea06d133efb75f9b394ff7383d1163a5a696873fadd46369438e7, 2026-11-19T12:55:49.990Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout b527f471d990d0f5b216b8aba20f2888749e86cd
 *
 * Carried over from the registered table at commit 8713ad4c63fafff9d62586a0ed71207ef1b7fe51: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d.
 * Their adoption lines are in `git show 8713ad4c63fafff9d62586a0ed71207ef1b7fe51:packages/runtime/src/kernels/geometry-profiles/nvidia-blackwell.ts`.
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
 * - linearGemv (the GEMV choices of linear), from 84 GEMV sweep records:
 *   - [1] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T03-12-30.json (2026-11-18T03:12:41.307Z, set full): 198 groups — 198 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 50 at reorder; 0 rows of current groups not current
 *   - [2] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T03-48-09.json (2026-11-18T03:48:20.090Z, set full): 198 groups — 198 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 50 at reorder; 0 rows of current groups not current
 *   - [3] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-42-50.json (2026-11-18T11:43:00.512Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [4] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-45-01.json (2026-11-18T11:45:11.863Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [5] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-47-49.json (2026-11-18T11:47:58.934Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [6] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-50-33.json (2026-11-18T11:50:43.452Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [7] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-53-26.json (2026-11-18T11:53:36.238Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [8] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T11-56-37.json (2026-11-18T11:56:46.999Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [9] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-05-42.json (2026-11-18T12:05:52.524Z, set full): 31 groups — 31 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [10] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-15-56.json (2026-11-18T12:16:05.831Z, set full): 31 groups — 31 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [11] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-17-52.json (2026-11-18T12:18:02.079Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [12] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-18-51.json (2026-11-18T12:19:01.093Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [13] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-22-44.json (2026-11-18T12:22:54.271Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [14] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-24-34.json (2026-11-18T12:24:44.107Z, set full): 5 groups — 5 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [15] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-26-18.json (2026-11-18T12:26:28.310Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [16] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-26-51.json (2026-11-18T12:27:01.149Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [17] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-27-31.json (2026-11-18T12:27:41.555Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [18] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-28-16.json (2026-11-18T12:28:26.772Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [19] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-32-24.json (2026-11-18T12:32:33.886Z, set full): 12 groups — 12 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [20] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-35-02.json (2026-11-18T12:35:11.843Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [21] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-37-46.json (2026-11-18T12:37:56.596Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [22] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-39-39.json (2026-11-18T12:39:48.916Z, set full): 5 groups — 5 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [23] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-41-45.json (2026-11-18T12:41:55.441Z, set full): 5 groups — 5 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [24] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-41-55.json (2026-11-18T12:42:05.104Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [25] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-42-47.json (2026-11-18T12:42:57.675Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [26] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-43-49.json (2026-11-18T12:43:59.659Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [27] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-44-14.json (2026-11-18T12:44:24.495Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [28] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-44-42.json (2026-11-18T12:44:52.107Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [29] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-45-28.json (2026-11-18T12:45:38.127Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [30] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-46-22.json (2026-11-18T12:46:32.115Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [31] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-47-31.json (2026-11-18T12:47:41.602Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [32] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-48-24.json (2026-11-18T12:48:33.862Z, set full): 3 groups — 3 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [33] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-49-11.json (2026-11-18T12:49:21.226Z, set full): 3 groups — 3 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [34] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-50-06.json (2026-11-18T12:50:15.881Z, set full): 3 groups — 3 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [35] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T12-58-03.json (2026-11-18T12:58:13.613Z, set full): 23 groups — 23 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [36] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-07-53.json (2026-11-18T13:08:03.426Z, set full): 46 groups — 46 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [37] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-09-44.json (2026-11-18T13:09:54.671Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [38] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-11-18.json (2026-11-18T13:11:27.919Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [39] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-13-17.json (2026-11-18T13:13:27.165Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [40] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-15-28.json (2026-11-18T13:15:38.368Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [41] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-18-15.json (2026-11-18T13:18:24.985Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [42] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-20-58.json (2026-11-18T13:21:08.586Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [43] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-23-51.json (2026-11-18T13:24:01.715Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [44] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-27-03.json (2026-11-18T13:27:13.229Z, set full): 9 groups — 9 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [45] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-36-08.json (2026-11-18T13:36:18.456Z, set full): 31 groups — 31 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [46] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-46-21.json (2026-11-18T13:46:31.538Z, set full): 31 groups — 31 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [47] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-48-21.json (2026-11-18T13:48:31.196Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [48] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-49-22.json (2026-11-18T13:49:32.200Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [49] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-53-17.json (2026-11-18T13:53:26.808Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [50] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-55-08.json (2026-11-18T13:55:17.939Z, set full): 5 groups — 5 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [51] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-56-53.json (2026-11-18T13:57:03.328Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [52] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-57-26.json (2026-11-18T13:57:36.088Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [53] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-58-06.json (2026-11-18T13:58:16.009Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [54] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T13-58-51.json (2026-11-18T13:59:01.315Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [55] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-02-58.json (2026-11-18T14:03:08.610Z, set full): 12 groups — 12 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [56] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-05-36.json (2026-11-18T14:05:46.384Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [57] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-08-21.json (2026-11-18T14:08:31.647Z, set full): 7 groups — 7 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [58] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-10-13.json (2026-11-18T14:10:23.206Z, set full): 5 groups — 5 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [59] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-12-19.json (2026-11-18T14:12:29.391Z, set full): 5 groups — 5 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [60] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-12-29.json (2026-11-18T14:12:39.262Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [61] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-13-22.json (2026-11-18T14:13:32.113Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [62] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-14-24.json (2026-11-18T14:14:33.893Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [63] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-14-48.json (2026-11-18T14:14:58.657Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [64] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-15-16.json (2026-11-18T14:15:26.126Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [65] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-16-03.json (2026-11-18T14:16:13.476Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [66] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-16-59.json (2026-11-18T14:17:08.721Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [67] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-18-10.json (2026-11-18T14:18:20.372Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [68] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-19-03.json (2026-11-18T14:19:13.097Z, set full): 3 groups — 3 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [69] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-19-50.json (2026-11-18T14:20:00.590Z, set full): 3 groups — 3 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [70] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-20-45.json (2026-11-18T14:20:55.320Z, set full): 3 groups — 3 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [71] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-28-44.json (2026-11-18T14:28:53.871Z, set full): 23 groups — 23 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [72] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-39-19.json (2026-11-18T14:39:29.511Z, set full): 46 groups — 46 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [73] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-41-11.json (2026-11-18T14:41:21.118Z, set full): 2 groups — 2 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [74] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T14-42-44.json (2026-11-18T14:42:54.524Z, set full): 4 groups — 4 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [75] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-31-11.json (2026-11-18T16:31:21.502Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [76] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-35-04.json (2026-11-18T16:35:14.423Z, set full): 11 groups — 11 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [77] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-38-00.json (2026-11-18T16:38:10.246Z, set full): 10 groups — 10 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [78] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-38-10.json (2026-11-18T16:38:19.845Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [79] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-38-19.json (2026-11-18T16:38:29.433Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [80] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-42-14.json (2026-11-18T16:42:24.158Z, set full): 11 groups — 11 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [81] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-45-10.json (2026-11-18T16:45:19.589Z, set full): 10 groups — 10 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [82] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-18T16-45-19.json (2026-11-18T16:45:29.220Z, set full): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [83] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-19T12-43-27.json (2026-11-19T12:43:38.554Z, set full): 75 groups — 75 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 *   - [84] outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/gemv-sweep_2026-11-19T12-55-39.json (2026-11-19T12:55:49.990Z, set full): 75 groups — 75 current, 0 not current (their case is not the current one); of the current, 0 without a usable reference at exact and 0 at reorder; 0 rows of current groups not current
 * - identical class (sequential.shapes, M = 1): 43 adopted, 56 kept
 *   - i4 n8960 k1536 (case i4g32-n8960-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.085 < ×1.100 in [1])); c32u8: ×1.085 [1], ×1.083 [2]
 *   - i4 n2048 k1536 (cases i4g32-n2048-k1536, i4g512-n2048-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2048-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.099 < ×1.100 in [2]); i4g512-n2048-k1536 adopts c16u8))
 *   - i4 n256 k1536 (cases i4g32-n256-k1536, i4g512-n256-k1536): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n256-k1536 adopts c32u8; i4g512-n256-k1536 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.092 < ×1.100 in [1])))
 *   - i4 n1536 k2048 (cases i4g32-n1536-k2048, i4g2048-n1536-k2048): c32u4 -> c16u8 (×1.100 [1] i4g32-n1536-k2048, ×1.106 [2] i4g32-n1536-k2048, ×1.108 [1] i4g2048-n1536-k2048, ×1.108 [2] i4g2048-n1536-k2048)
 *   - i4 n6144 k1536 (cases i4g32-n6144-k1536, i4g512-n6144-k1536): keeps c32u4 (i4g32-n6144-k1536: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.095 < ×1.100 in [2]); i4g512-n6144-k1536: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.079 < ×1.100 in [1]))
 *   - i4 n1536 k6144 (cases i4g32-n1536-k6144, i4g2048-n1536-k6144): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n1536-k6144 adopts c32u8; i4g2048-n1536-k6144 adopts c16u8))
 *   - i4 n1536 k256 (case i4g32-n1536-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.078 < ×1.100 in [1])); c32u8: ×1.078 [1], ×1.061 [2]
 *   - i4 n4096 k1536 (cases i4g32-n4096-k1536, i4g512-n4096-k1536): c32u4 -> c32u8 (×1.110 [1] i4g32-n4096-k1536, ×1.104 [2] i4g32-n4096-k1536, ×1.106 [1] i4g512-n4096-k1536, ×1.105 [2] i4g512-n4096-k1536)
 *   - i4 n512 k1536 (cases i4g32-n512-k1536, i4g512-n512-k1536): c32u4 -> c16u8 (×1.121 [1] i4g32-n512-k1536, ×1.125 [2] i4g32-n512-k1536, ×1.112 [1] i4g512-n512-k1536, ×1.105 [2] i4g512-n512-k1536)
 *   - i4 n1536 k4096 (cases i4g32-n1536-k4096, i4g4096-n1536-k4096): c32u4 -> c16u8 (×1.108 [1] i4g32-n1536-k4096, ×1.107 [2] i4g32-n1536-k4096, ×1.110 [1] i4g4096-n1536-k4096, ×1.110 [2] i4g4096-n1536-k4096)
 *   - i4 n12288 k1536 (case i4g32-n12288-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.043 < ×1.100 in [1])); c32u8: ×1.043 [1], ×1.042 [2]
 *   - i4 n1536 k12288 (case i4g32-n1536-k12288): c32u4 -> c32u8 (×1.110 [1], ×1.110 [2])
 *   - i8 n262144 k1536 (case i8-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u2: ×0.999 < ×1.100 in [1])); c32u2: ×0.999 [1], ×0.999 [2]
 *   - i4 n262144 k1536 (case i4g32-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u16: ×1.026 < ×1.100 in [13])); c128u16: ×1.026 [13], ×1.028 [49]
 *   - i4 n256 k3072 (case i4g32-n256-k3072): c32u4 -> c32u8 (×1.105 [16], ×1.103 [52])
 *   - i4 n1024 k256 (case i4g32-n1024-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.090 < ×1.100 in [19])); c16u8: ×1.090 [19], ×1.077 [55]
 *   - i4 n256 k1024 (case i4g32-n256-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.093 < ×1.100 in [16])); c32u8: ×1.093 [16], ×1.094 [52]
 *   - i4 n2048 k256 (case i4g32-n2048-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.076 < ×1.100 in [19])); c32u8: ×1.076 [19], ×1.060 [55]
 *   - i4 n256 k2048 (case i4g32-n256-k2048): c32u4 -> c32u8 (×1.102 [19], ×1.102 [55])
 *   - i4 n262144 k256 (case i4g32-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.007 < ×1.100 in [13])); c32u8: ×1.007 [13], ×1.005 [49]
 *   - i8 n256 k3072 (case i8-n256-k3072): c32u4 -> c16u16 (×1.278 [19], ×1.274 [55])
 *   - i8 n1024 k256 (case i8-n1024-k256): c32u4 -> c16u16 (×1.200 [19], ×1.195 [55])
 *   - i8 n256 k1024 (case i8-n256-k1024): c32u4 -> c16u16 (×1.229 [19], ×1.252 [55])
 *   - i8 n2048 k256 (case i8-n2048-k256): c32u4 -> c16u16 (×1.159 [19], ×1.160 [55])
 *   - i8 n256 k2048 (case i8-n256-k2048): c32u4 -> c16u16 (×1.256 [19], ×1.248 [55])
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.003 < ×1.100 in [13])); c32u16: ×1.003 [13], ×1.003 [49]
 *   - i8 n256 k1536 (case i8-n256-k1536): c32u4 -> c16u16 (×1.262 [1], ×1.260 [2])
 *   - i8 n1536 k256 (case i8-n1536-k256): c32u4 -> c16u16 (×1.142 [1], ×1.144 [2])
 *   - i2 n12288 k1536 (case i2-n12288-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u2: ×0.960 < ×1.100 in [1])); c32u2: ×0.960 [1], ×0.959 [2]
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.000 < ×1.100 in [1])); c16u4: ×1.000 [1], ×1.000 [2]
 *   - i2 n262144 k1536 (case i2-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u1: ×0.979 < ×1.100 in [13])); c32u1: ×0.979 [13], ×0.978 [49]
 *   - i4 n10752 k2560 (case i4g32-n10752-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.063 < ×1.100 in [1])); c32u8: ×1.063 [1], ×1.059 [2]
 *   - i4 n2048 k2560 (cases i4g32-n2048-k2560, i4g512-n2048-k2560): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2048-k2560 no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.096 < ×1.100 in [2]); i4g512-n2048-k2560 adopts c16u8))
 *   - i4 n512 k2560 (cases i4g32-n512-k2560, i4g512-n512-k2560): c32u4 -> c16u8 (×1.136 [1] i4g32-n512-k2560, ×1.136 [2] i4g32-n512-k2560, ×1.126 [1] i4g512-n512-k2560, ×1.126 [2] i4g512-n512-k2560)
 *   - i4 n256 k2560 (case i4g32-n256-k2560): c32u4 -> c16u8 (×1.110 [1], ×1.110 [2])
 *   - i4 n2560 k2048 (cases i4g32-n2560-k2048, i4g2048-n2560-k2048): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2560-k2048 adopts c16u8; i4g2048-n2560-k2048 adopts c32u8))
 *   - i4 n10240 k2560 (cases i4g32-n10240-k2560, i4g512-n10240-k2560): keeps c32u4 (i4g32-n10240-k2560: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.060 < ×1.100 in [1]); i4g512-n10240-k2560: no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.052 < ×1.100 in [1]))
 *   - i4 n2560 k10240 (cases i4g32-n2560-k10240, i4g2048-n2560-k10240): c32u4 -> c16u8 (×1.138 [1] i4g32-n2560-k10240, ×1.137 [2] i4g32-n2560-k10240, ×1.114 [1] i4g2048-n2560-k10240, ×1.113 [2] i4g2048-n2560-k10240)
 *   - i4 n2560 k256 (case i4g32-n2560-k256): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.064 < ×1.100 in [1])); c32u8: ×1.064 [1], ×1.076 [2]
 *   - i4 n4096 k2560 (cases i4g32-n4096-k2560, i4g512-n4096-k2560): c32u4 -> c32u8 (×1.111 [1] i4g32-n4096-k2560, ×1.114 [2] i4g32-n4096-k2560, ×1.110 [1] i4g512-n4096-k2560, ×1.110 [2] i4g512-n4096-k2560)
 *   - i4 n1024 k2560 (cases i4g32-n1024-k2560, i4g512-n1024-k2560): c32u4 -> c16u8 (×1.133 [1] i4g32-n1024-k2560, ×1.133 [2] i4g32-n1024-k2560, ×1.123 [1] i4g512-n1024-k2560, ×1.123 [2] i4g512-n1024-k2560)
 *   - i4 n2560 k4096 (cases i4g32-n2560-k4096, i4g4096-n2560-k4096): keeps c32u4 (the cases that share the key disagree, so the default stays (i4g32-n2560-k4096 adopts c16u8; i4g4096-n2560-k4096 adopts c32u8))
 *   - i8 n262144 k2560 (case i8-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u2: ×0.999 < ×1.100 in [1])); c32u2: ×0.999 [1], ×0.999 [2]
 *   - i4 n262144 k2560 (case i4g32-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c128u16: ×1.084 < ×1.100 in [13])); c128u16: ×1.084 [13], ×1.084 [49]
 *   - i4 n256 k5120 (case i4g32-n256-k5120): c32u4 -> c16u8 (×1.109 [19], ×1.108 [55])
 *   - i8 n256 k2560 (case i8-n256-k2560): c32u4 -> c16u16 (×1.298 [1], ×1.296 [2])
 *   - i8 n2560 k256 (case i8-n2560-k256): c32u4 -> c16u16 (×1.160 [1], ×1.153 [2])
 *   - i2 n262144 k2560 (case i2-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u1: ×0.991 < ×1.100 in [13])); c32u1: ×0.991 [13], ×0.993 [49]
 *   - i4 n15360 k3840 (case i4g32-n15360-k3840): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.039 < ×1.100 in [1])); c32u8: ×1.039 [1], ×1.039 [2]
 *   - i4 n8192 k3840 (case i4g32-n8192-k3840): c32u4 -> c32u8 (×1.113 [1], ×1.112 [2])
 *   - i4 n4096 k3840 (case i4g32-n4096-k3840): c32u4 -> c32u8 (×1.118 [1], ×1.111 [2])
 *   - i4 n2048 k3840 (case i4g32-n2048-k3840): c32u4 -> c32u8 (×1.113 [1], ×1.105 [2])
 *   - i4 n512 k3840 (case i4g32-n512-k3840): c32u4 -> c16u8 (×1.133 [1], ×1.132 [2])
 *   - i4 n3840 k4096 (case i4g32-n3840-k4096): c32u4 -> c32u8 (×1.115 [1], ×1.115 [2])
 *   - i4 n3840 k8192 (case i4g32-n3840-k8192): c32u4 -> c32u8 (×1.113 [1], ×1.115 [2])
 *   - i4 n3840 k15360 (case i4g32-n3840-k15360): c32u4 -> c32u8 (×1.120 [1], ×1.115 [2])
 *   - i4 n262144 k3840 (case i4g32-n262144-k3840): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.051 < ×1.100 in [1])); c32u16: ×1.051 [1], ×1.049 [2]
 *   - i4 n1024 k7680 (case i4g32-n1024-k7680): c32u4 -> c16u8 (×1.136 [32], ×1.136 [68])
 *   - i4 n4096 k1024 (case i4g32-n4096-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.082 < ×1.100 in [19])); c32u8: ×1.082 [19], ×1.095 [55]
 *   - i4 n1024 k4096 (case i4g32-n1024-k4096): c32u4 -> c16u8 (×1.128 [32], ×1.128 [68])
 *   - i4 n8192 k1024 (case i4g32-n8192-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.090 < ×1.100 in [19])); c32u8: ×1.090 [19], ×1.089 [55]
 *   - i4 n1024 k8192 (case i4g32-n1024-k8192): c32u4 -> c16u8 (×1.129 [32], ×1.130 [68])
 *   - i4 n3840 k1024 (case i4g32-n3840-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.084 < ×1.100 in [19])); c32u8: ×1.084 [19], ×1.098 [55]
 *   - i4 n262144 k1024 (case i4g32-n262144-k1024): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.022 < ×1.100 in [13])); c32u16: ×1.022 [13], ×1.024 [49]
 *   - i4 n8192 k2816 (case i4g32-n8192-k2816): c32u4 -> c32u8 (×1.106 [1], ×1.102 [2])
 *   - i4 n4096 k2816 (case i4g32-n4096-k2816): c32u4 -> c32u8 (×1.114 [1], ×1.103 [2])
 *   - i4 n2048 k2816 (case i4g32-n2048-k2816): c32u4 -> c32u8 (×1.108 [1], ×1.110 [2])
 *   - i4 n2112 k2816 (case i4g32-n2112-k2816): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u8: ×1.094 < ×1.100 in [2])); c32u8: ×1.109 [1], ×1.094 [2]
 *   - i4 n1024 k2816 (case i4g32-n1024-k2816): c32u4 -> c16u8 (×1.130 [1], ×1.113 [2])
 *   - i4 n128 k2816 (case i4g32-n128-k2816): c32u4 -> c16u8 (×1.135 [1], ×1.118 [2])
 *   - i4 n2816 k4096 (case i4g32-n2816-k4096): c32u4 -> c16u8 (×1.139 [1], ×1.133 [2])
 *   - i4 n2816 k8192 (case i4g32-n2816-k8192): c32u4 -> c16u8 (×1.141 [1], ×1.142 [2])
 *   - i4 n2816 k2112 (case i4g32-n2816-k2112): c32u4 -> c32u8 (×1.148 [1], ×1.140 [2])
 *   - i4 n262144 k2816 (case i4g32-n262144-k2816): c32u4 -> c32u16 (×1.130 [1], ×1.128 [2])
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.037 < ×1.100 in [35])); c32u16: ×1.037 [35], ×1.042 [71], ×1.041 [76], ×1.042 [80], ×1.041 [83], ×1.033 [84]
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.051 < ×1.100 in [35])); c16u8: ×1.051 [35], ×1.051 [71], ×1.049 [77], ×1.051 [81], ×1.051 [83], ×1.043 [84]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.026 < ×1.100 in [35])); c16u8: ×1.026 [35], ×1.021 [71], ×1.029 [77], ×1.027 [81], ×1.019 [83], ×1.019 [84]
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.956 < ×1.100 in [35])); c16u16: ×0.956 [35], ×0.950 [71], ×0.950 [76], ×0.949 [80], ×0.958 [83], ×0.944 [84]
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.013 < ×1.100 in [35])); c16u16: ×1.013 [35], ×1.014 [71], ×1.016 [76], ×1.016 [80], ×1.011 [83], ×1.013 [84]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.047 < ×1.100 in [35])); c16u8: ×1.047 [35], ×1.050 [71], ×1.048 [77], ×1.049 [81], ×1.042 [83], ×1.042 [84]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): c32u16 -> c16u8 (×1.212 [35], ×1.197 [71], ×1.194 [78], ×1.211 [82], ×1.206 [83], ×1.189 [84])
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.067 < ×1.100 in [35])); c16u16: ×1.067 [35], ×1.070 [71], ×1.070 [76], ×1.074 [80], ×1.065 [83], ×1.054 [84]
 *   - bf16 n512 k1536 (case bf16-n512-k1536): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.022 < ×1.100 in [35])); c16u8: ×1.022 [35], ×1.025 [71], ×1.028 [77], ×1.024 [81], ×1.023 [83], ×1.017 [84]
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.047 < ×1.100 in [35])); c16u8: ×1.047 [35], ×1.049 [71], ×1.049 [77], ×1.049 [81], ×1.039 [83], ×1.042 [84]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.011 < ×1.100 in [35])); c16u16: ×1.011 [35], ×1.012 [71], ×1.011 [76], ×1.011 [80], ×1.009 [83], ×1.007 [84]
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.040 < ×1.100 in [35])); c16u8: ×1.040 [35], ×1.041 [71], ×1.042 [77], ×1.043 [81], ×1.033 [83], ×1.035 [84]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.028 < ×1.100 in [37])); c16u8: ×1.028 [37], ×1.028 [73], ×1.022 [83], ×1.020 [84]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×1.035 < ×1.100 in [35])); c16u16: ×1.035 [35], ×1.037 [71], ×1.038 [76], ×1.038 [80], ×1.036 [83], ×1.036 [84]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.977 < ×1.100 in [35])); c16u16: ×0.977 [35], ×0.972 [71], ×0.968 [76], ×0.969 [80], ×0.964 [83], ×0.960 [84]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.030 < ×1.100 in [35])); c16u8: ×1.030 [35], ×1.024 [71], ×1.027 [77], ×1.025 [81], ×1.025 [83], ×1.020 [84]
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.030 < ×1.100 in [35])); c16u8: ×1.030 [35], ×1.023 [71], ×1.026 [77], ×1.026 [81], ×1.021 [83], ×1.018 [84]
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.943 < ×1.100 in [35])); c16u16: ×0.943 [35], ×0.944 [71], ×0.943 [76], ×0.943 [80], ×0.936 [83], ×0.937 [84]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c32u16: ×1.011 < ×1.100 in [35])); c32u16: ×1.011 [35], ×1.019 [71], ×1.008 [83], ×1.004 [84]
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u16: ×0.997 < ×1.100 in [35])); c16u16: ×0.997 [35], ×0.993 [71], ×0.997 [76], ×0.984 [80], ×1.016 [83], ×1.016 [84]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.007 < ×1.100 in [35])); c16u8: ×1.007 [35], ×1.004 [71], ×1.007 [77], ×1.004 [81], ×0.991 [83], ×0.997 [84]
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): c16u8 -> c16u16 (×1.119 [35], ×1.119 [71], ×1.120 [76], ×1.118 [80], ×1.113 [83], ×1.111 [84])
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps c16u16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u8: ×1.028 < ×1.100 in [35])); c16u8: ×1.028 [35], ×1.023 [71], ×1.025 [77], ×1.026 [81], ×1.018 [83], ×1.017 [84]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): keeps c16u8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u4: ×1.024 < ×1.100 in [35])); c16u4: ×1.024 [35], ×1.023 [71], ×1.030 [76], ×1.017 [80], ×1.029 [83], ×1.004 [84]
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps c32u4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest c16u1: ×1.022 < ×1.100 in [37])); c16u1: ×1.022 [37], ×1.021 [73], ×1.010 [83], ×1.013 [84]
 * - reorder class (parallelShapes, M = 1 to 8): 62 adopted, 53 kept
 *   - i4 g32 n8960 k1536 (case i4g32-n8960-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r4 is not measured in [1] at M=4; the reference staged-l4r4 is not measured in [2] at M=4; screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]))); staged-l8: ×1.676 [1], ×1.655 [2]
 *   - i4 g32 n2048 k1536 (case i4g32-n2048-k1536): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r2 is not measured in [1] at M=8; the reference staged-l4r2 is not measured in [2] at M=8; M=8 is not measured in [3]; M=8 is not measured in [4]; M=8 is not measured in [5]; screened out by the quick look in [6] at M=8; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]; M=8 is not measured in [25]; M=8 is not measured in [26]; M=8 is not measured in [27]; M=8 is not measured in [28]; M=8 is not measured in [29]; M=8 is not measured in [30]; M=8 is not measured in [31]; M=8 is not measured in [32]; M=8 is not measured in [33]; M=8 is not measured in [34]; M=8 is not measured in [35]; M=8 is not measured in [36]; M=8 is not measured in [37]; M=8 is not measured in [38]; M=8 is not measured in [39]; M=8 is not measured in [40]; M=8 is not measured in [41]; screened out by the quick look in [42] at M=8; M=8 is not measured in [43]; M=8 is not measured in [44]; M=8 is not measured in [45]; M=8 is not measured in [46]; M=8 is not measured in [47]; M=8 is not measured in [48]; M=8 is not measured in [49]; M=8 is not measured in [50]; M=8 is not measured in [51]; M=8 is not measured in [52]; M=8 is not measured in [53]; M=8 is not measured in [54]; M=8 is not measured in [55]; M=8 is not measured in [56]; M=8 is not measured in [57]; M=8 is not measured in [58]; M=8 is not measured in [59]; M=8 is not measured in [60]; M=8 is not measured in [61]; M=8 is not measured in [62]; M=8 is not measured in [63]; M=8 is not measured in [64]; M=8 is not measured in [65]; M=8 is not measured in [66]; M=8 is not measured in [67]; M=8 is not measured in [68]; M=8 is not measured in [69]; M=8 is not measured in [70]; M=8 is not measured in [71]; M=8 is not measured in [72]; M=8 is not measured in [73]; M=8 is not measured in [74]; M=8 is not measured in [75]; M=8 is not measured in [76]; M=8 is not measured in [77]; M=8 is not measured in [78]; M=8 is not measured in [79]; M=8 is not measured in [80]; M=8 is not measured in [81]; M=8 is not measured in [82]; M=8 is not measured in [83]; M=8 is not measured in [84]))); staged-l16: ×2.186 [1], ×2.180 [2]
 *   - i4 g32 n256 k1536 (case i4g32-n256-k1536): parallel-l32 with row blocks of 2 -> staged-l16 (×1.454 [1], ×1.446 [2]); at the other row counts: ×1.211 [7] M=4, ×1.214 [43] M=4, ×1.247 [8] M=8, ×1.237 [44] M=8
 *   - i4 g32 n1536 k2048 (case i4g32-n1536-k2048): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.954 < ×1.100 in [1])); staged-l16: ×0.954 [1], ×0.962 [2]
 *   - i4 g32 n6144 k1536 (case i4g32-n6144-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r4 is not measured in [1] at M=4; the reference staged-l4r4 is not measured in [2] at M=4; screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]))); staged-l16: ×1.765 [1], ×1.760 [2]
 *   - i4 g32 n1536 k6144 (case i4g32-n1536-k6144): staged-l32 with row blocks of 2 -> staged-l32; at the other row counts: ×1.160 [7] M=4, ×1.157 [43] M=4, ×1.042 [8] M=8, ×1.043 [44] M=8
 *   - i4 g32 n1536 k256 (case i4g32-n1536-k256): parallel-l4 -> staged-l8 (×1.187 [1], ×1.164 [2]); at the other row counts: ×1.202 [1] M=4, ×1.210 [2] M=4, ×1.402 [1] M=8, ×1.397 [2] M=8
 *   - i4 g32 n4096 k1536 (case i4g32-n4096-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: slower than the default's choice in [3] at M=4 (×0.891))); staged-l16: ×1.792 [1], ×1.783 [2]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r4 is not measured in [1] at M=4; the reference staged-l4r4 is not measured in [2] at M=4; screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]); 2 rows — slower than the reference's row block in [4] at M=8 (×0.955); 8 rows — slower than the reference's row block in [4] at M=8 (×0.887)
 *   - i4 g32 n512 k1536 (case i4g32-n512-k1536): parallel-l32 -> staged-l32 (×2.264 [1], ×2.251 [2]); at the other row counts: ×4.655 [1] M=4, ×4.642 [2] M=4, ×5.417 [1] M=8, ×5.398 [2] M=8
 *   - i4 g32 n1536 k4096 (case i4g32-n1536-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.935 < ×1.100 in [1])); staged-l16: ×0.935 [1], ×0.935 [2]
 *   - i4 g32 n12288 k1536 (case i4g32-n12288-k1536): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r4 is not measured in [1] at M=4; the reference staged-l4r4 is not measured in [2] at M=4; screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]))); staged-l8: ×1.906 [1], ×1.898 [2]
 *   - i4 g32 n1536 k12288 (case i4g32-n1536-k12288): staged-l32 with row blocks of 2 -> staged-l32; at the other row counts: ×1.164 [7] M=4, ×1.163 [43] M=4, ×1.017 [8] M=8, ×1.016 [44] M=8
 *   - i8 n262144 k1536 (case i8-n262144-k1536): parallel-l16 -> staged-l16 (×1.249 [1], ×1.249 [2]); at the other row counts: ×1.253 [1] M=4, ×1.252 [2] M=4, ×1.256 [1] M=8, ×1.256 [2] M=8
 *     - subgroup32-l8 meets the rule too (×1.249 [1], ×1.248 [2]): a Session option, not a row
 *   - i4 g32 n262144 k1536 (case i4g32-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; screened out by the quick look in [14] at M=4; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; screened out by the quick look in [50] at M=4; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]))); staged-l8: ×1.542 [13], ×1.544 [49]
 *   - i4 g32 n256 k3072 (case i4g32-n256-k3072): parallel-l32 -> staged-l32 (×1.926 [16], ×1.922 [52]); at the other row counts: ×4.798 [17] M=4, ×4.826 [53] M=4, ×6.485 [18] M=8, ×6.648 [54] M=8
 *   - i4 g32 n1024 k256 (case i4g32-n1024-k256): no row (c32u4) -> staged-l8 (×1.772 [19], ×1.787 [55]); at the other row counts: ×1.621 [20] M=4, ×1.621 [56] M=4, ×1.515 [21] M=8, ×1.521 [57] M=8
 *     - subgroup32-l4 meets the rule too (×1.555 [19], ×1.572 [55]): a Session option, not a row
 *   - i4 g32 n256 k1024 (case i4g32-n256-k1024): parallel-l32 -> staged-l32 (×1.479 [16], ×1.478 [52]); at the other row counts: ×3.188 [17] M=4, ×3.188 [53] M=4, ×4.260 [18] M=8, ×4.285 [54] M=8
 *   - i4 g32 n2048 k256 (case i4g32-n2048-k256): no row (c32u4) -> staged-l8 with row blocks of 2 (×1.748 [19], ×1.738 [55]); at the other row counts: ×1.676 [20] M=4, ×1.666 [56] M=4, ×1.873 [21] M=8, ×1.893 [57] M=8
 *     - subgroup32-l4 meets the rule too (×1.510 [19], ×1.531 [55]): a Session option, not a row
 *   - i4 g32 n256 k2048 (case i4g32-n256-k2048): no row (c32u4) -> staged-l32 (×6.716 [19], ×6.702 [55]); at the other row counts: ×5.360 [20] M=4, ×5.373 [56] M=4, ×3.977 [21] M=8, ×3.967 [57] M=8
 *   - i4 g32 n262144 k256 (case i4g32-n262144-k256): no row (c32u4) -> staged-l4 with row blocks of 8 (×1.970 [13], ×1.971 [49]); at the other row counts: ×1.399 [14] M=4, ×1.396 [50] M=4, ×1.446 [15] M=8, ×1.439 [51] M=8
 *   - i8 n256 k3072 (case i8-n256-k3072): no row (c32u4) -> staged-l32 (×8.230 [19], ×8.159 [55]); at the other row counts: ×6.273 [22] M=4, ×6.284 [58] M=4, ×4.559 [23] M=8, ×4.556 [59] M=8
 *     - subgroup32-l32 meets the rule too (×7.414 [19], ×7.419 [55]): a Session option, not a row
 *   - i8 n1024 k256 (case i8-n1024-k256): no row (c32u4) -> staged-l16 with row blocks of 2 (×1.992 [19], ×1.992 [55]); at the other row counts: ×1.782 [22] M=4, ×1.782 [58] M=4, ×1.862 [23] M=8, ×1.862 [59] M=8
 *     - subgroup32-l8 meets the rule too (×1.992 [19], ×1.992 [55]): a Session option, not a row
 *   - i8 n256 k1024 (case i8-n256-k1024): no row (c32u4) -> staged-l32 (×5.245 [19], ×5.336 [55]); at the other row counts: ×4.408 [22] M=4, ×4.418 [58] M=4, ×3.229 [23] M=8, ×3.205 [59] M=8
 *     - subgroup32-l32 meets the rule too (×4.880 [19], ×4.934 [55]): a Session option, not a row
 *   - i8 n2048 k256 (case i8-n2048-k256): no row (c32u4) -> staged-l16 with row blocks of 2 (×1.881 [19], ×1.838 [55]); at the other row counts: ×1.867 [22] M=4, ×1.876 [58] M=4, ×1.939 [23] M=8, ×1.934 [59] M=8
 *     - subgroup32-l8 meets the rule too (×1.778 [19], ×1.778 [55]): a Session option, not a row
 *   - i8 n256 k2048 (case i8-n256-k2048): no row (c32u4) -> staged-l32 (×7.139 [19], ×7.146 [55]); at the other row counts: ×5.844 [22] M=4, ×5.808 [58] M=4, ×4.121 [23] M=8, ×4.093 [59] M=8
 *     - subgroup32-l32 meets the rule too (×6.560 [19], ×6.556 [55]): a Session option, not a row
 *   - i8 n262144 k256 (case i8-n262144-k256): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l4: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; screened out by the quick look in [14] at M=4; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; screened out by the quick look in [50] at M=4; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]))); parallel-l4: ×1.277 [13], ×1.277 [49]
 *   - i4 g512 n2048 k1536 (case i4g512-n2048-k1536): parallel-l4 -> staged-l16 (×2.146 [1], ×2.146 [2]); at the other row counts: ×1.648 [1] M=4, ×1.652 [2] M=4, ×1.789 [1] M=8, ×1.789 [2] M=8
 *   - i4 g512 n256 k1536 (case i4g512-n256-k1536): parallel-l32 -> staged-l32 (×1.578 [1], ×1.578 [2]); at the other row counts: ×3.333 [1] M=4, ×3.349 [2] M=4, ×4.470 [1] M=8, ×4.469 [2] M=8
 *   - i4 g2048 n1536 k2048 (case i4g2048-n1536-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i4 g512 n6144 k1536 (case i4g512-n6144-k1536): parallel-l4 -> staged-l16 (×1.730 [1], ×1.726 [2]); at the other row counts: ×1.654 [1] M=4, ×1.648 [2] M=4, ×1.680 [1] M=8, ×1.674 [2] M=8
 *   - i4 g2048 n1536 k6144 (case i4g2048-n1536-k6144): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k1536 (case i8-n256-k1536): parallel-l32 -> staged-l32 (×1.133 [1], ×1.131 [2]); at the other row counts: ×1.959 [1] M=4, ×1.959 [2] M=4, ×2.744 [1] M=8, ×2.745 [2] M=8
 *   - i8 n1536 k256 (case i8-n1536-k256): keeps parallel-l4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: slower than the default's choice in [1] at M=4 (×0.982))); staged-l16: ×1.188 [1], ×1.184 [2]
 *   - i4 g512 n4096 k1536 (case i4g512-n4096-k1536): parallel-l4 -> staged-l16 (×1.764 [1], ×1.765 [2]); at the other row counts: ×1.764 [1] M=4, ×1.764 [2] M=4, ×1.711 [1] M=8, ×1.708 [2] M=8
 *   - i4 g512 n512 k1536 (case i4g512-n512-k1536): parallel-l32 -> staged-l32 (×2.262 [1], ×2.267 [2]); at the other row counts: ×4.671 [1] M=4, ×4.660 [2] M=4, ×5.403 [1] M=8, ×5.394 [2] M=8
 *   - i4 g4096 n1536 k4096 (case i4g4096-n1536-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n12288 k1536 (case i2-n12288-k1536): parallel-l2 -> staged-l4 (×1.412 [1], ×1.410 [2]); at the other row counts: ×1.312 [1] M=4, ×1.302 [2] M=4, ×1.278 [1] M=8, ×1.274 [2] M=8
 *   - i2 n1536 k12288 (case i2-n1536-k12288): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k1536 (case i2-n262144-k1536): no row (c32u4) -> staged-l2 with row blocks of 4 (×1.531 [13], ×1.529 [49]); at the other row counts: ×2.232 [29] M=4, ×2.231 [65] M=4, ×2.257 [30] M=8, ×2.255 [66] M=8
 *   - i4 g32 n10752 k2560 (case i4g32-n10752-k2560): parallel-l4 with row blocks of 4 -> parallel-l4 with row blocks of 8; at the other row counts: ×1.000 [3] M=4, ×0.996 [39] M=4, ×1.144 [4] M=8, ×1.144 [40] M=8
 *   - i4 g32 n2048 k2560 (case i4g32-n2048-k2560): parallel-l4 with row blocks of 2 -> staged-l16 with row blocks of 2 (×2.455 [1], ×2.465 [2]); at the other row counts: ×1.443 [5] M=4, ×1.443 [41] M=4, ×1.035 [6] M=8, ×1.024 [42] M=8, ×1.020 [75] M=8, ×1.021 [79] M=8
 *   - i4 g32 n512 k2560 (case i4g32-n512-k2560): parallel-l32 -> staged-l32 (×2.652 [1], ×2.646 [2]); at the other row counts: ×5.760 [1] M=4, ×5.761 [2] M=4, ×6.372 [1] M=8, ×6.386 [2] M=8
 *   - i4 g32 n256 k2560 (case i4g32-n256-k2560): parallel-l32 -> staged-l32 (×1.756 [1], ×1.758 [2]); at the other row counts: ×4.210 [1] M=4, ×4.209 [2] M=4, ×5.582 [1] M=8, ×5.583 [2] M=8
 *   - i4 g32 n2560 k2048 (case i4g32-n2560-k2048): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×1.064 < ×1.100 in [1])); staged-l16: ×1.064 [1], ×1.069 [2]
 *   - i4 g32 n10240 k2560 (case i4g32-n10240-k2560): parallel-l4 with row blocks of 4 -> parallel-l4 with row blocks of 8; at the other row counts: ×1.000 [3] M=4, ×0.997 [39] M=4, ×1.140 [4] M=8, ×1.140 [40] M=8
 *   - i4 g32 n2560 k10240 (case i4g32-n2560-k10240): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×1.021 < ×1.100 in [1])); staged-l16: ×1.021 [1], ×1.025 [2]
 *   - i4 g32 n2560 k256 (case i4g32-n2560-k256): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: slower than the default's choice in [5] at M=4 (×0.965))); staged-l8: ×1.274 [1], ×1.274 [2]
 *   - i4 g32 n4096 k2560 (case i4g32-n4096-k2560): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: slower than the default's choice in [5] at M=4 (×0.830))); staged-l16: ×1.875 [1], ×1.885 [2]
 *   - i4 g32 n1024 k2560 (case i4g32-n1024-k2560): parallel-l32 with row blocks of 2 -> staged-l16 with row blocks of 2 (×4.072 [1], ×4.066 [2]); at the other row counts: ×1.700 [7] M=4, ×1.706 [43] M=4, ×1.565 [8] M=8, ×1.571 [44] M=8
 *   - i4 g32 n2560 k4096 (case i4g32-n2560-k4096): keeps staged-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×1.064 < ×1.100 in [1])); staged-l16: ×1.064 [1], ×1.068 [2]
 *   - i8 n262144 k2560 (case i8-n262144-k2560): parallel-l16 -> staged-l16 (×1.230 [1], ×1.229 [2]); at the other row counts: ×1.230 [1] M=4, ×1.230 [2] M=4, ×1.234 [1] M=8, ×1.234 [2] M=8
 *     - subgroup32-l8 meets the rule too (×1.230 [1], ×1.230 [2]): a Session option, not a row
 *   - i4 g32 n262144 k2560 (case i4g32-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; screened out by the quick look in [14] at M=4; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; screened out by the quick look in [50] at M=4; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]))); staged-l8: ×1.605 [13], ×1.602 [49]
 *   - i4 g32 n256 k5120 (case i4g32-n256-k5120): no row (c32u4) -> staged-l32 (×9.210 [19], ×9.223 [55]); at the other row counts: ×7.289 [20] M=4, ×7.240 [56] M=4, ×4.896 [21] M=8, ×4.860 [57] M=8
 *   - i4 g512 n2048 k2560 (case i4g512-n2048-k2560): parallel-l4 -> staged-l16 (×2.435 [1], ×2.441 [2]); at the other row counts: ×1.745 [1] M=4, ×1.750 [2] M=4, ×1.899 [1] M=8, ×1.900 [2] M=8
 *   - i4 g512 n512 k2560 (case i4g512-n512-k2560): parallel-l32 -> staged-l32 (×2.676 [1], ×2.679 [2]); at the other row counts: ×5.810 [1] M=4, ×5.813 [2] M=4, ×6.446 [1] M=8, ×6.437 [2] M=8
 *   - i4 g2048 n2560 k2048 (case i4g2048-n2560-k2048): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i4 g512 n10240 k2560 (case i4g512-n10240-k2560): parallel-l4 -> staged-l16 (×1.816 [1], ×1.819 [2]); at the other row counts: ×1.806 [1] M=4, ×1.809 [2] M=4, ×1.758 [1] M=8, ×1.757 [2] M=8
 *   - i4 g2048 n2560 k10240 (case i4g2048-n2560-k10240): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i8 n256 k2560 (case i8-n256-k2560): parallel-l32 -> staged-l32 (×1.127 [1], ×1.125 [2]); at the other row counts: ×2.239 [1] M=4, ×2.238 [2] M=4, ×3.092 [1] M=8, ×3.086 [2] M=8
 *   - i8 n2560 k256 (case i8-n2560-k256): parallel-l4 -> staged-l8 (×1.131 [1], ×1.145 [2]); at the other row counts: ×1.072 [1] M=4, ×1.073 [2] M=4, ×1.095 [1] M=8, ×1.093 [2] M=8
 *   - i4 g512 n4096 k2560 (case i4g512-n4096-k2560): parallel-l4 -> staged-l16 (×1.850 [1], ×1.847 [2]); at the other row counts: ×1.868 [1] M=4, ×1.867 [2] M=4, ×1.805 [1] M=8, ×1.805 [2] M=8
 *   - i4 g512 n1024 k2560 (case i4g512-n1024-k2560): parallel-l32 -> staged-l32 (×4.478 [1], ×4.476 [2]); at the other row counts: ×6.497 [1] M=4, ×6.461 [2] M=4, ×7.331 [1] M=8, ×7.318 [2] M=8
 *   - i4 g4096 n2560 k4096 (case i4g4096-n2560-k4096): keeps packed-l32 (the default takes packed activations, whose kernels the sweep does not measure)
 *   - i2 n262144 k2560 (case i2-n262144-k2560): no row (c32u4) -> staged-l2 with row blocks of 4 (×1.601 [13], ×1.603 [49]); at the other row counts: ×2.259 [29] M=4, ×2.265 [65] M=4, ×2.294 [30] M=8, ×2.294 [66] M=8
 *   - i4 g32 n15360 k3840 (case i4g32-n15360-k3840): parallel-l4 -> staged-l16 (×1.720 [1], ×1.720 [2]); at the other row counts: ×1.716 [1] M=4, ×1.733 [2] M=4, ×1.720 [1] M=8, ×1.721 [2] M=8
 *   - i4 g32 n8192 k3840 (case i4g32-n8192-k3840): parallel-l4 -> staged-l8 (×1.735 [1], ×1.737 [2]); at the other row counts: ×1.956 [1] M=4, ×1.952 [2] M=4, ×2.006 [1] M=8, ×1.994 [2] M=8
 *   - i4 g32 n4096 k3840 (case i4g32-n4096-k3840): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: slower than the default's choice in [5] at M=4 (×0.833))); staged-l8: ×1.852 [1], ×1.846 [2]
 *   - i4 g32 n2048 k3840 (case i4g32-n2048-k3840): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r2 is not measured in [1] at M=8; the reference staged-l4r2 is not measured in [2] at M=8; M=8 is not measured in [3]; M=8 is not measured in [4]; M=8 is not measured in [5]; screened out by the quick look in [6] at M=8; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]; M=8 is not measured in [25]; M=8 is not measured in [26]; M=8 is not measured in [27]; M=8 is not measured in [28]; M=8 is not measured in [29]; M=8 is not measured in [30]; M=8 is not measured in [31]; M=8 is not measured in [32]; M=8 is not measured in [33]; M=8 is not measured in [34]; M=8 is not measured in [35]; M=8 is not measured in [36]; M=8 is not measured in [37]; M=8 is not measured in [38]; M=8 is not measured in [39]; M=8 is not measured in [40]; M=8 is not measured in [41]; screened out by the quick look in [42] at M=8; M=8 is not measured in [43]; M=8 is not measured in [44]; M=8 is not measured in [45]; M=8 is not measured in [46]; M=8 is not measured in [47]; M=8 is not measured in [48]; M=8 is not measured in [49]; M=8 is not measured in [50]; M=8 is not measured in [51]; M=8 is not measured in [52]; M=8 is not measured in [53]; M=8 is not measured in [54]; M=8 is not measured in [55]; M=8 is not measured in [56]; M=8 is not measured in [57]; M=8 is not measured in [58]; M=8 is not measured in [59]; M=8 is not measured in [60]; M=8 is not measured in [61]; M=8 is not measured in [62]; M=8 is not measured in [63]; M=8 is not measured in [64]; M=8 is not measured in [65]; M=8 is not measured in [66]; M=8 is not measured in [67]; M=8 is not measured in [68]; M=8 is not measured in [69]; M=8 is not measured in [70]; M=8 is not measured in [71]; M=8 is not measured in [72]; M=8 is not measured in [73]; M=8 is not measured in [74]; M=8 is not measured in [75]; M=8 is not measured in [76]; M=8 is not measured in [77]; M=8 is not measured in [78]; M=8 is not measured in [79]; M=8 is not measured in [80]; M=8 is not measured in [81]; M=8 is not measured in [82]; M=8 is not measured in [83]; M=8 is not measured in [84]))); staged-l16: ×2.555 [1], ×2.559 [2]
 *   - i4 g32 n512 k3840 (case i4g32-n512-k3840): keeps parallel-l32 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: slower than the default's choice in [7] at M=4 (×0.983))); staged-l32: ×2.962 [1], ×2.967 [2]
 *   - i4 g32 n3840 k4096 (case i4g32-n3840-k4096): staged-l32 -> staged-l16 (×1.207 [1], ×1.200 [2]); at the other row counts: ×1.205 [1] M=4, ×1.203 [2] M=4, ×1.215 [1] M=8, ×1.213 [2] M=8
 *   - i4 g32 n3840 k8192 (case i4g32-n3840-k8192): staged-l32 with row blocks of 2 -> staged-l16 with row blocks of 4 (×1.178 [1], ×1.179 [2]); at the other row counts: ×1.616 [7] M=4, ×1.612 [43] M=4, ×1.515 [8] M=8, ×1.516 [44] M=8
 *   - i4 g32 n3840 k15360 (case i4g32-n3840-k15360): staged-l32 -> staged-l16 (×1.137 [1], ×1.136 [2]); at the other row counts: ×1.145 [1] M=4, ×1.144 [2] M=4, ×1.153 [1] M=8, ×1.153 [2] M=8
 *   - i4 g32 n262144 k3840 (case i4g32-n262144-k3840): parallel-l16 -> staged-l8 (×3.547 [1], ×3.546 [2]); at the other row counts: ×3.548 [1] M=4, ×3.524 [2] M=4, ×3.546 [1] M=8, ×3.528 [2] M=8
 *   - i4 g32 n1024 k7680 (case i4g32-n1024-k7680): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; screened out by the quick look in [32] at M=1; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not measured in [35]; M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; screened out by the quick look in [68] at M=1; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not measured in [71]; M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not measured in [81]; M=1 is not measured in [82]; M=1 is not measured in [83]; M=1 is not measured in [84])))
 *   - i4 g32 n4096 k1024 (case i4g32-n4096-k1024): no row (c32u4) -> staged-l16 with row blocks of 2 (×2.690 [19], ×2.696 [55]); at the other row counts: ×2.476 [20] M=4, ×2.483 [56] M=4, ×1.750 [21] M=8, ×1.756 [57] M=8
 *   - i4 g32 n1024 k4096 (case i4g32-n1024-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.820 < ×1.100 in [32])); staged-l16: ×0.820 [32], ×0.819 [68]
 *   - i4 g32 n8192 k1024 (case i4g32-n8192-k1024): no row (c32u4) -> staged-l8 with row blocks of 2 (×2.270 [19], ×2.286 [55]); at the other row counts: ×2.140 [20] M=4, ×2.149 [56] M=4, ×1.722 [21] M=8, ×1.721 [57] M=8
 *   - i4 g32 n1024 k8192 (case i4g32-n1024-k8192): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; screened out by the quick look in [32] at M=1; M=1 is not measured in [33]; M=1 is not measured in [34]; M=1 is not measured in [35]; M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; screened out by the quick look in [68] at M=1; M=1 is not measured in [69]; M=1 is not measured in [70]; M=1 is not measured in [71]; M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; M=1 is not measured in [77]; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; M=1 is not measured in [81]; M=1 is not measured in [82]; M=1 is not measured in [83]; M=1 is not measured in [84])))
 *   - i4 g32 n3840 k1024 (case i4g32-n3840-k1024): no row (c32u4) -> staged-l16 with row blocks of 2 (×2.847 [19], ×2.862 [55]); at the other row counts: ×2.280 [20] M=4, ×2.291 [56] M=4, ×2.547 [21] M=8, ×2.557 [57] M=8
 *   - i4 g32 n262144 k1024 (case i4g32-n262144-k1024): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l4: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; screened out by the quick look in [14] at M=4; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; screened out by the quick look in [50] at M=4; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]))); staged-l4: ×1.527 [13], ×1.528 [49]
 *   - i4 g32 n8192 k2816 (case i4g32-n8192-k2816): keeps parallel-l4 with row blocks of 4 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r4 is not measured in [1] at M=4; the reference staged-l4r4 is not measured in [2] at M=4; screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]))); staged-l8: ×1.720 [1], ×1.715 [2]
 *     - no other row block replaces the default's: none — evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r4 is not measured in [1] at M=4; the reference staged-l4r4 is not measured in [2] at M=4; screened out by the quick look in [3] at M=4; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; M=4 is not measured in [38]; screened out by the quick look in [39] at M=4; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; M=4 is not measured in [74]; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; M=4 is not measured in [83]; M=4 is not measured in [84]); 2 rows — slower than the reference's row block in [3] at M=4 (×0.906); 8 rows — at no row count ×1.100 faster than the reference's row block in every group (at most ×1.095)
 *   - i4 g32 n4096 k2816 (case i4g32-n4096-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l8: slower than the default's choice in [5] at M=4 (×0.823))); staged-l8: ×1.807 [1], ×1.799 [2]
 *   - i4 g32 n2048 k2816 (case i4g32-n2048-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r2 is not measured in [1] at M=8; the reference staged-l4r2 is not measured in [2] at M=8; M=8 is not measured in [3]; M=8 is not measured in [4]; M=8 is not measured in [5]; screened out by the quick look in [6] at M=8; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]; M=8 is not measured in [25]; M=8 is not measured in [26]; M=8 is not measured in [27]; M=8 is not measured in [28]; M=8 is not measured in [29]; M=8 is not measured in [30]; M=8 is not measured in [31]; M=8 is not measured in [32]; M=8 is not measured in [33]; M=8 is not measured in [34]; M=8 is not measured in [35]; M=8 is not measured in [36]; M=8 is not measured in [37]; M=8 is not measured in [38]; M=8 is not measured in [39]; M=8 is not measured in [40]; M=8 is not measured in [41]; screened out by the quick look in [42] at M=8; M=8 is not measured in [43]; M=8 is not measured in [44]; M=8 is not measured in [45]; M=8 is not measured in [46]; M=8 is not measured in [47]; M=8 is not measured in [48]; M=8 is not measured in [49]; M=8 is not measured in [50]; M=8 is not measured in [51]; M=8 is not measured in [52]; M=8 is not measured in [53]; M=8 is not measured in [54]; M=8 is not measured in [55]; M=8 is not measured in [56]; M=8 is not measured in [57]; M=8 is not measured in [58]; M=8 is not measured in [59]; M=8 is not measured in [60]; M=8 is not measured in [61]; M=8 is not measured in [62]; M=8 is not measured in [63]; M=8 is not measured in [64]; M=8 is not measured in [65]; M=8 is not measured in [66]; M=8 is not measured in [67]; M=8 is not measured in [68]; M=8 is not measured in [69]; M=8 is not measured in [70]; M=8 is not measured in [71]; M=8 is not measured in [72]; M=8 is not measured in [73]; M=8 is not measured in [74]; M=8 is not measured in [75]; M=8 is not measured in [76]; M=8 is not measured in [77]; M=8 is not measured in [78]; M=8 is not measured in [79]; M=8 is not measured in [80]; M=8 is not measured in [81]; M=8 is not measured in [82]; M=8 is not measured in [83]; M=8 is not measured in [84]))); staged-l16: ×2.386 [1], ×2.385 [2]
 *   - i4 g32 n2112 k2816 (case i4g32-n2112-k2816): keeps parallel-l4 with row blocks of 2 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: the reference staged-l4r2 is not measured in [1] at M=8; the reference staged-l4r2 is not measured in [2] at M=8; M=8 is not measured in [3]; M=8 is not measured in [4]; M=8 is not measured in [5]; screened out by the quick look in [6] at M=8; M=8 is not measured in [7]; M=8 is not measured in [8]; M=8 is not measured in [9]; M=8 is not measured in [10]; M=8 is not measured in [11]; M=8 is not measured in [12]; M=8 is not measured in [13]; M=8 is not measured in [14]; M=8 is not measured in [15]; M=8 is not measured in [16]; M=8 is not measured in [17]; M=8 is not measured in [18]; M=8 is not measured in [19]; M=8 is not measured in [20]; M=8 is not measured in [21]; M=8 is not measured in [22]; M=8 is not measured in [23]; M=8 is not measured in [24]; M=8 is not measured in [25]; M=8 is not measured in [26]; M=8 is not measured in [27]; M=8 is not measured in [28]; M=8 is not measured in [29]; M=8 is not measured in [30]; M=8 is not measured in [31]; M=8 is not measured in [32]; M=8 is not measured in [33]; M=8 is not measured in [34]; M=8 is not measured in [35]; M=8 is not measured in [36]; M=8 is not measured in [37]; M=8 is not measured in [38]; M=8 is not measured in [39]; M=8 is not measured in [40]; M=8 is not measured in [41]; screened out by the quick look in [42] at M=8; M=8 is not measured in [43]; M=8 is not measured in [44]; M=8 is not measured in [45]; M=8 is not measured in [46]; M=8 is not measured in [47]; M=8 is not measured in [48]; M=8 is not measured in [49]; M=8 is not measured in [50]; M=8 is not measured in [51]; M=8 is not measured in [52]; M=8 is not measured in [53]; M=8 is not measured in [54]; M=8 is not measured in [55]; M=8 is not measured in [56]; M=8 is not measured in [57]; M=8 is not measured in [58]; M=8 is not measured in [59]; M=8 is not measured in [60]; M=8 is not measured in [61]; M=8 is not measured in [62]; M=8 is not measured in [63]; M=8 is not measured in [64]; M=8 is not measured in [65]; M=8 is not measured in [66]; M=8 is not measured in [67]; M=8 is not measured in [68]; M=8 is not measured in [69]; M=8 is not measured in [70]; M=8 is not measured in [71]; M=8 is not measured in [72]; M=8 is not measured in [73]; M=8 is not measured in [74]; M=8 is not measured in [75]; M=8 is not measured in [76]; M=8 is not measured in [77]; M=8 is not measured in [78]; M=8 is not measured in [79]; M=8 is not measured in [80]; M=8 is not measured in [81]; M=8 is not measured in [82]; M=8 is not measured in [83]; M=8 is not measured in [84]))); staged-l16: ×2.386 [1], ×2.382 [2]
 *   - i4 g32 n1024 k2816 (case i4g32-n1024-k2816): parallel-l32 -> staged-l32 (×4.653 [1], ×4.629 [2]); at the other row counts: ×6.886 [1] M=4, ×6.846 [2] M=4, ×7.770 [1] M=8, ×7.739 [2] M=8
 *   - i4 g32 n128 k2816 (case i4g32-n128-k2816): parallel-l32 -> staged-l32 (×1.833 [1], ×1.839 [2]); at the other row counts: ×2.802 [1] M=4, ×2.853 [2] M=4, ×4.252 [1] M=8, ×4.326 [2] M=8
 *   - i4 g32 n2816 k4096 (case i4g32-n2816-k4096): staged-l32 -> staged-l16 (×1.120 [1], ×1.124 [2]); at the other row counts: ×1.196 [1] M=4, ×1.195 [2] M=4, ×1.217 [1] M=8, ×1.217 [2] M=8
 *   - i4 g32 n2816 k8192 (case i4g32-n2816-k8192): staged-l32 -> staged-l16 (×1.113 [1], ×1.108 [2]); at the other row counts: ×1.158 [1] M=4, ×1.158 [2] M=4, ×1.174 [1] M=8, ×1.175 [2] M=8
 *   - i4 g32 n2816 k2112 (case i4g32-n2816-k2112): staged-l32 with row blocks of 2 -> staged-l16 with row blocks of 2 (×1.197 [1], ×1.206 [2]); at the other row counts: ×1.504 [7] M=4, ×1.503 [43] M=4, ×1.701 [8] M=8, ×1.707 [44] M=8
 *   - i4 g32 n262144 k2816 (case i4g32-n262144-k2816): parallel-l4 -> staged-l8 (×1.314 [1], ×1.310 [2]); at the other row counts: ×1.302 [1] M=4, ×1.298 [2] M=4, ×1.293 [1] M=8, ×1.238 [2] M=8
 *   - bf16 n8960 k1536 (case bf16-n8960-k1536): no row (c16u8) -> parallel-l8 (×3.751 [35], ×3.755 [71], ×3.776 [76], ×3.776 [80], ×3.778 [83], ×3.750 [84]); at the other row counts: ×2.124 [36] M=4, ×2.131 [72] M=4, ×2.138 [83] M=4, ×2.111 [84] M=4, ×1.148 [36] M=8, ×1.151 [72] M=8, ×1.150 [83] M=8, ×1.138 [84] M=8
 *   - bf16 n2048 k1536 (case bf16-n2048-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×1.002 < ×1.100 in [35])); staged-l32: ×1.002 [35], ×1.009 [71], ×1.002 [77], ×1.005 [81], ×1.010 [83], ×1.003 [84]
 *   - bf16 n256 k1536 (case bf16-n256-k1536): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; screened out by the quick look in [35] at M=1; M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; screened out by the quick look in [71] at M=1; M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; screened out by the quick look in [77] at M=1; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; screened out by the quick look in [81] at M=1; M=1 is not measured in [82]; screened out by the quick look in [83] at M=1; screened out by the quick look in [84] at M=1)))
 *   - bf16 n1536 k2048 (case bf16-n1536-k2048): no row (c16u8) -> staged-l32 (×4.036 [35], ×4.023 [71], ×4.019 [76], ×4.039 [80], ×4.033 [83], ×4.007 [84]); at the other row counts: ×4.102 [36] M=4, ×4.091 [72] M=4, ×4.089 [83] M=4, ×4.085 [84] M=4, ×2.191 [36] M=8, ×2.195 [72] M=8, ×2.198 [83] M=8, ×2.193 [84] M=8
 *   - bf16 n6144 k1536 (case bf16-n6144-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.971 < ×1.100 in [35])); staged-l16: ×0.971 [35], ×0.962 [71], ×0.968 [76], ×0.977 [80], ×0.979 [83], ×0.963 [84]
 *   - bf16 n1536 k6144 (case bf16-n1536-k6144): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×1.073 < ×1.100 in [35])); staged-l32: ×1.073 [35], ×1.075 [71], ×1.074 [77], ×1.073 [81], ×1.070 [83], ×1.069 [84]
 *   - bf16 n1536 k256 (case bf16-n1536-k256): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×1.061 < ×1.100 in [35])); parallel-l16: ×1.061 [35], ×1.078 [71], ×1.059 [78], ×1.090 [82], ×1.080 [83], ×1.066 [84]
 *   - bf16 n4096 k1536 (case bf16-n4096-k1536): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.890 < ×1.100 in [35])); staged-l16: ×0.890 [35], ×0.887 [71], ×0.891 [76], ×0.898 [80], ×0.894 [83], ×0.884 [84]
 *   - bf16 n512 k1536 (case bf16-n512-k1536): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l32: slower than the default's choice in [36] at M=4 (×0.658))); parallel-l32: ×1.165 [35], ×1.170 [71], ×1.169 [77], ×1.168 [81], ×1.170 [83], ×1.163 [84]
 *   - bf16 n1536 k4096 (case bf16-n1536-k4096): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.946 < ×1.100 in [35])); parallel-l16: ×0.946 [35], ×0.943 [71], ×0.941 [77], ×0.939 [81], ×0.936 [83], ×0.935 [84]
 *   - bf16 n12288 k1536 (case bf16-n12288-k1536): no row (c16u8) -> parallel-l8 (×4.086 [35], ×4.070 [71], ×4.094 [76], ×4.083 [80], ×4.056 [83], ×4.081 [84]); at the other row counts: ×2.224 [36] M=4, ×2.231 [72] M=4, ×2.207 [83] M=4, ×2.207 [84] M=4, ×1.190 [36] M=8, ×1.193 [72] M=8, ×1.183 [83] M=8, ×1.183 [84] M=8
 *   - bf16 n1536 k12288 (case bf16-n1536-k12288): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×1.088 < ×1.100 in [35])); staged-l32: ×1.088 [35], ×1.089 [71], ×1.090 [77], ×1.091 [81], ×1.083 [83], ×1.078 [84]
 *   - bf16 n262144 k1536 (case bf16-n262144-k1536): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; screened out by the quick look in [38] at M=4; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; screened out by the quick look in [74] at M=4; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; screened out by the quick look in [83] at M=4; screened out by the quick look in [84] at M=4))); parallel-l16: ×1.527 [37], ×1.527 [73], ×1.517 [83], ×1.517 [84]
 *   - bf16 n10752 k2560 (case bf16-n10752-k2560): keeps no row (c16u8) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l8: slower than the default's choice in [36] at M=8 (×0.929))); parallel-l8: ×1.981 [35], ×1.957 [71], ×1.969 [76], ×1.958 [80], ×2.056 [83], ×2.053 [84]
 *   - bf16 n2048 k2560 (case bf16-n2048-k2560): keeps staged-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.947 < ×1.100 in [35])); parallel-l16: ×0.947 [35], ×0.943 [71], ×0.942 [76], ×0.944 [80], ×0.934 [83], ×0.937 [84]
 *   - bf16 n512 k2560 (case bf16-n512-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l16: ×0.801 < ×1.100 in [35])); parallel-l16: ×0.801 [35], ×0.803 [71], ×0.798 [77], ×0.805 [81], ×0.800 [83], ×0.797 [84]
 *   - bf16 n256 k2560 (case bf16-n256-k2560): keeps parallel-l32 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest parallel-l2: evidence at M=1 from fewer than 2 records (M=1 is not measured in [1]; M=1 is not measured in [2]; M=1 is not measured in [3]; M=1 is not measured in [4]; M=1 is not measured in [5]; M=1 is not measured in [6]; M=1 is not measured in [7]; M=1 is not measured in [8]; M=1 is not measured in [9]; M=1 is not measured in [10]; M=1 is not measured in [11]; M=1 is not measured in [12]; M=1 is not measured in [13]; M=1 is not measured in [14]; M=1 is not measured in [15]; M=1 is not measured in [16]; M=1 is not measured in [17]; M=1 is not measured in [18]; M=1 is not measured in [19]; M=1 is not measured in [20]; M=1 is not measured in [21]; M=1 is not measured in [22]; M=1 is not measured in [23]; M=1 is not measured in [24]; M=1 is not measured in [25]; M=1 is not measured in [26]; M=1 is not measured in [27]; M=1 is not measured in [28]; M=1 is not measured in [29]; M=1 is not measured in [30]; M=1 is not measured in [31]; M=1 is not measured in [32]; M=1 is not measured in [33]; M=1 is not measured in [34]; screened out by the quick look in [35] at M=1; M=1 is not measured in [36]; M=1 is not measured in [37]; M=1 is not measured in [38]; M=1 is not measured in [39]; M=1 is not measured in [40]; M=1 is not measured in [41]; M=1 is not measured in [42]; M=1 is not measured in [43]; M=1 is not measured in [44]; M=1 is not measured in [45]; M=1 is not measured in [46]; M=1 is not measured in [47]; M=1 is not measured in [48]; M=1 is not measured in [49]; M=1 is not measured in [50]; M=1 is not measured in [51]; M=1 is not measured in [52]; M=1 is not measured in [53]; M=1 is not measured in [54]; M=1 is not measured in [55]; M=1 is not measured in [56]; M=1 is not measured in [57]; M=1 is not measured in [58]; M=1 is not measured in [59]; M=1 is not measured in [60]; M=1 is not measured in [61]; M=1 is not measured in [62]; M=1 is not measured in [63]; M=1 is not measured in [64]; M=1 is not measured in [65]; M=1 is not measured in [66]; M=1 is not measured in [67]; M=1 is not measured in [68]; M=1 is not measured in [69]; M=1 is not measured in [70]; screened out by the quick look in [71] at M=1; M=1 is not measured in [72]; M=1 is not measured in [73]; M=1 is not measured in [74]; M=1 is not measured in [75]; M=1 is not measured in [76]; screened out by the quick look in [77] at M=1; M=1 is not measured in [78]; M=1 is not measured in [79]; M=1 is not measured in [80]; screened out by the quick look in [81] at M=1; M=1 is not measured in [82]; screened out by the quick look in [83] at M=1; screened out by the quick look in [84] at M=1)))
 *   - bf16 n2560 k2048 (case bf16-n2560-k2048): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: slower than the default's choice in [36] at M=4 (×0.756))); staged-l32: ×1.149 [35], ×1.140 [71], ×1.147 [76], ×1.142 [80], ×1.135 [83], ×1.134 [84]
 *   - bf16 n10240 k2560 (case bf16-n10240-k2560): no row (c32u4) -> staged-l16 (×2.863 [35], ×2.856 [71], ×3.007 [83], ×3.043 [84]); at the other row counts: ×1.924 [36] M=4, ×1.947 [72] M=4, ×1.930 [83] M=4, ×1.939 [84] M=4, ×1.043 [36] M=8, ×1.046 [72] M=8, ×1.074 [83] M=8, ×1.074 [84] M=8
 *   - bf16 n2560 k10240 (case bf16-n2560-k10240): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×1.028 < ×1.100 in [35])); staged-l32: ×1.028 [35], ×0.965 [71], ×1.039 [76], ×1.018 [80], ×1.152 [83], ×1.156 [84]
 *   - bf16 n2560 k256 (case bf16-n2560-k256): no row (c16u16) -> parallel-l8 (×2.082 [35], ×2.082 [71], ×2.082 [77], ×2.073 [81], ×2.061 [83], ×2.072 [84]); at the other row counts: ×2.547 [36] M=4, ×2.503 [72] M=4, ×2.466 [83] M=4, ×2.434 [84] M=4, ×1.413 [36] M=8, ×1.384 [72] M=8, ×1.377 [83] M=8, ×1.372 [84] M=8
 *   - bf16 n4096 k2560 (case bf16-n4096-k2560): keeps parallel-l8 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: ×0.896 < ×1.100 in [35])); staged-l16: ×0.896 [35], ×0.919 [71], ×0.910 [76], ×0.914 [80], ×0.903 [83], ×0.903 [84]
 *   - bf16 n1024 k2560 (case bf16-n1024-k2560): keeps parallel-l16 (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l32: ×0.979 < ×1.100 in [35])); staged-l32: ×0.979 [35], ×0.976 [71], ×0.980 [77], ×0.980 [81], ×0.970 [83], ×0.969 [84]
 *   - bf16 n2560 k4096 (case bf16-n2560-k4096): parallel-l16 -> staged-l32 (×1.232 [35], ×1.232 [71], ×1.228 [76], ×1.226 [80], ×1.219 [83], ×1.222 [84]); at the other row counts: ×1.182 [36] M=4, ×1.183 [72] M=4, ×1.171 [83] M=4, ×1.171 [84] M=4, ×1.166 [36] M=8, ×1.171 [72] M=8, ×1.169 [83] M=8, ×1.172 [84] M=8
 *   - bf16 n262144 k2560 (case bf16-n262144-k2560): keeps no row (c32u4) (no candidate is ×1.100 faster in the groups of 2 or more records that are evidence, with its check held (nearest staged-l16: evidence at M=4 from fewer than 2 records (a parallel row serves M = 1 to 8: M=4 is not measured in [1]; M=4 is not measured in [2]; M=4 is not measured in [3]; M=4 is not measured in [4]; M=4 is not measured in [5]; M=4 is not measured in [6]; M=4 is not measured in [7]; M=4 is not measured in [8]; M=4 is not measured in [9]; M=4 is not measured in [10]; M=4 is not measured in [11]; M=4 is not measured in [12]; M=4 is not measured in [13]; M=4 is not measured in [14]; M=4 is not measured in [15]; M=4 is not measured in [16]; M=4 is not measured in [17]; M=4 is not measured in [18]; M=4 is not measured in [19]; M=4 is not measured in [20]; M=4 is not measured in [21]; M=4 is not measured in [22]; M=4 is not measured in [23]; M=4 is not measured in [24]; M=4 is not measured in [25]; M=4 is not measured in [26]; M=4 is not measured in [27]; M=4 is not measured in [28]; M=4 is not measured in [29]; M=4 is not measured in [30]; M=4 is not measured in [31]; M=4 is not measured in [32]; M=4 is not measured in [33]; M=4 is not measured in [34]; M=4 is not measured in [35]; M=4 is not measured in [36]; M=4 is not measured in [37]; screened out by the quick look in [38] at M=4; M=4 is not measured in [39]; M=4 is not measured in [40]; M=4 is not measured in [41]; M=4 is not measured in [42]; M=4 is not measured in [43]; M=4 is not measured in [44]; M=4 is not measured in [45]; M=4 is not measured in [46]; M=4 is not measured in [47]; M=4 is not measured in [48]; M=4 is not measured in [49]; M=4 is not measured in [50]; M=4 is not measured in [51]; M=4 is not measured in [52]; M=4 is not measured in [53]; M=4 is not measured in [54]; M=4 is not measured in [55]; M=4 is not measured in [56]; M=4 is not measured in [57]; M=4 is not measured in [58]; M=4 is not measured in [59]; M=4 is not measured in [60]; M=4 is not measured in [61]; M=4 is not measured in [62]; M=4 is not measured in [63]; M=4 is not measured in [64]; M=4 is not measured in [65]; M=4 is not measured in [66]; M=4 is not measured in [67]; M=4 is not measured in [68]; M=4 is not measured in [69]; M=4 is not measured in [70]; M=4 is not measured in [71]; M=4 is not measured in [72]; M=4 is not measured in [73]; screened out by the quick look in [74] at M=4; M=4 is not measured in [75]; M=4 is not measured in [76]; M=4 is not measured in [77]; M=4 is not measured in [78]; M=4 is not measured in [79]; M=4 is not measured in [80]; M=4 is not measured in [81]; M=4 is not measured in [82]; screened out by the quick look in [83] at M=4; screened out by the quick look in [84] at M=4))); staged-l16: ×1.579 [37], ×1.579 [73], ×1.570 [83], ×1.570 [84]
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
        { storage: "bf16", n: 1536, k: 256, variant: { cols: 16, unroll: 8 } },
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
        { storage: "bf16", n: 4096, k: 2560, variant: { cols: 16, unroll: 16 } },
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
      {
        storage: "i8",
        n: 262144,
        k: 1536,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
      },
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
      {
        storage: "bf16",
        n: 2560,
        k: 4096,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
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
