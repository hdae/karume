# Node + Dawn port: what the first full run showed (2026-10-16)

> Point-in-time snapshot: one day's runs on one machine. The decisions live in ADR 0127 (toolchain)
> and ADR 0129 (test layout); this note records what the port produced and where Node + Dawn
> behaved differently from Deno + wgpu.

## Setup

|                 | Deno (baseline, main `d06f0815`)              | Node (branch `wip/0.15`)                           |
| --------------- | --------------------------------------------- | -------------------------------------------------- |
| Runtime         | Deno 2.9.6 (V8 15.0, TypeScript 6.0.3)        | Node 24.22.0 (V8 13.6, TypeScript 7.0.2)           |
| WebGPU          | wgpu (Deno's built-in)                        | Dawn, npm `webgpu` 0.6.2, Vulkan backend           |
| GPU             | NVIDIA GeForce RTX 3080 Ti, driver 615.71.09  | same machine                                       |
| Environment key | `deno-nvidia-geforce-rtx-3080-ti`             | `node-nvidia-geforce-rtx-3080-ti`                  |
| Test runner     | `deno test` lanes (`deno task test:models:*`) | Vitest 5.0.4 projects (`vp test --project <lane>`) |

The baseline is the full verify that judged the 0.14.0 release, frozen under
`outputs/misc/0.15-baseline-2026-10-16/` (36 lane directories, 157 artifact files, `SHA256SUMS`).

## Outputs: Dawn and wgpu produce the same numbers

`node tools/verify-diff/artifacts.ts --before <deno env dir> --after <node env dir>` over every
family (`outputs/misc/0.15-port/artifacts-deno-vs-node-pass3.txt`):

- safetensors, wav and raw rgb artifacts: byte-identical.
- PNG artifacts: every pixel identical (u8 max difference 0 in every file); the bytes differ because
  the PNG encoder's zlib is not the same implementation on the two runtimes.
- sha256 reference rows: 80 `node-…` rows written; 69 equal the `deno-…` rows; the 11 that differ
  are all anima PNG rows (compressed bytes, see above).

So on this GPU the model outputs are bit-identical between Dawn and wgpu. The PNG rows hash the
compressed bytes, which makes them runtime-specific even when the image is not; hashing the pixels
instead is a phase-2 candidate (ADR 0127, open points).

## Lane wall clock (GPU lock held, same machine)

Deno: the frozen baseline run. Node: the reference-writing pass (`pass3/`; anima and gemma4 from
the second writing pass, which also wrote the residency rows). Minutes:seconds.

| Lane           | Deno    | Node        |
| -------------- | ------- | ----------- |
| core           | 4:58    | 3:20        |
| anima          | 18:46   | 13:37       |
| sbv2           | 0:46    | 0:48        |
| irodori        | 2:51    | 2:49        |
| gemma4         | 9:02    | 11:15       |
| gemma4-qat     | 1:59    | 2:06        |
| birefnet       | 3:40    | 3:49        |
| depth-anything | 0:06    | 0:06        |
| siglip2        | 0:24    | 0:31        |
| vowel-detector | 0:04    | 0:02        |
| minicpm5       | 0:30    | 0:32        |
| embeddinggemma | 0:10    | 0:12        |
| deberta        | 0:14    | 0:17        |
| dacvae         | 0:06    | 0:05        |
| wan            | 43:19   | 36:04       |
| wan-ti2v       | 45:54   | 43:03       |
| umt5           | 0:14    | 0:15        |
| **all lanes**  | 2:13:03 | 2:01:11 |

The family lanes are dominated by the real-weight e2e tests (now `packages/e2e`); the triage list
(`2026-10-16-test-triage.md`) is the phase-2 input for shortening them.

## Where Node + Dawn differed from Deno + wgpu

Each item: what differed → how the port handles it.

1. **Timestamp queries are quantized** to 65,536 ns by Dawn, so short passes read as 0 →
   `packages/test-support/src/setup.ts` creates Dawn with
   `disable-dawn-features=timestamp_quantization` (override with `KARUME_DAWN_FLAGS`); the unit on
   Node is ns (`TIMESTAMP_UNIT_NS` has a `node-…` row).
2. **Destroyed buffers are reclaimed only after the next submission completes** → the OOM gate
   submits an empty command buffer and waits before the survivor allocation.
3. **`GPUAdapterInfo.description` is the driver version** on Dawn (`NVIDIA: 615.71.09 …`) → the
   environment key is derived from `device` on Node (`node-nvidia-geforce-rtx-3080-ti`).
4. **`@webgpu/types` brands its interfaces** (`__brand`) → hand-built fakes go through
   `platformObject<T>()` (`@karume/test-support/webgpu`).
5. **No `caches` (CacheStorage) on Node** → the hub's HTTP source cannot run as shipped; the tests
   pass a `MemoryCacheStorage`; a file-backed CacheStorage for Node is an open point of ADR 0127.
6. **`@hdae/fetch-cache` picks its read strategy by `globalThis.Deno`** ("blob" elsewhere) → the hub
   passes `"stream"` explicitly on Deno and Node.
7. **`fs.readFile` refuses files over 2 GiB and `FileHandle.read` takes an int32 length** → the
   test-support file helpers read in 1 GiB steps (the Wan reference band is 2.6 GB).
8. **Node's ICU 78 implements Unicode 17.0**; the NFKC table generated with Python's `unicodedata`
   is older → one code point (U+A7F1) is pinned as a known difference in the irodori text test.
9. **Node's external memory baseline is about 9 MiB** (module sources) → the weight-block
   retention probe measures the delta from a baseline instead of an absolute.
10. **No `globalThis.addEventListener("error" | "unhandledrejection")`** → `process` events via
    `@karume/test-support/process-events`.
11. **The WebGPU constants (`GPUBufferUsage` …) do not exist without Dawn** → the setup installs
    them unconditionally, so the CI configuration (`KARUME_NO_DAWN=1`) runs the non-GPU tests.
12. **Dawn exposes shader-f16, subgroups and dp4a on this machine** where Deno's wgpu did not; the
    variants Deno skipped (for example RMSNorm subgroup32) now run in the core lane.
13. **`setTimeout` can fire before `performance.now()` says the delay has elapsed** (libuv's
    millisecond loop time; 19.88 ms observed for a 20 ms timer, once in the full run) → the
    weight-supply timing tests sleep until the deadline is reached on `performance.now()`.

## Counts

| Run                                            | Result                              |
| ---------------------------------------------- | ----------------------------------- |
| core, CI configuration (`KARUME_NO_DAWN=1`)    | 2,292 passed, 637 skipped, 0 failed |
| core, Dawn                                     | 2,925 passed, 0 failed              |
| every lane, compare mode (pass 4, after Oxfmt) | 5,048 passed, 17 skipped (opt-in cases), 0 failed                        |

Pass 4 ran on `26053ef2` (the formatted tree): the 16 family lanes passed as listed; the core
lane had the one timer misfire of item 13 (1 failed / 2,928 passed) and passed 2,929 / 2,929 on the
re-run after the fix (`1c8f6f8b`, 3:06). The family lanes were not re-run; the fix touches one
test file in the core lane only.

`vp check` after the port: 0 errors; 331 Oxlint warnings from the default rule set, left for the
phase-2 rule decision (no code was changed for lint during the port).
