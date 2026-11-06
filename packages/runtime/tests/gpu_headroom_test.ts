/**
 * Probe allocation of free VRAM (`fitsHeadroom` — `src/gpu/headroom.ts`).
 *
 * On a real GPU this pins three things: (1) an amount that fits gives true; (2) an amount that does
 * not fit gives false **with the device still alive** (a small allocation right after succeeds and
 * `device.lost` does not resolve); (3) after true, the same amount can really be allocated (the
 * call does not return before the probe allocation has been released). (2) also confirms that this
 * machine's implementation keeps the WebGPU spec's guarantee that an OOM in createBuffer is not
 * fatal (wgpu loses the device on an OOM on the writeBuffer side —
 * docs/research/2026-09-27-h35-oom-device-lost.md).
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { fitsHeadroom, probePieces } from "../src/gpu/headroom.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const MIB = 1024 ** 2;

describe("probePieces (how the probe allocation is split)", () => {
  it("0 bytes makes no pieces", () => {
    assertEquals(probePieces(0, 1024), []);
  });
  it("up to maxBufferSize is one piece, rounded up to 4 bytes", () => {
    assertEquals(probePieces(7, 1024), [8]);
    assertEquals(probePieces(1024, 1024), [1024]);
  });
  it("an amount above maxBufferSize is split evenly and every piece is at most maxBufferSize", () => {
    const pieces = probePieces(5 * 1024 + 1, 1024);
    assertEquals(pieces.length, 6);
    for (const piece of pieces) {
      assert(piece <= 1024 && piece % 4 === 0, `piece ${piece}`);
    }
    assert(pieces.reduce((sum, piece) => sum + piece, 0) >= 5 * 1024 + 1);
  });
  it("adds one piece when maxBufferSize is not a multiple of 4 and rounding up exceeds the limit", () => {
    // Deno's maxBufferSize is 2^31 - 1, which has this shape: one piece of 1023 → 1024 after 4-byte
    // alignment > 1023.
    const pieces = probePieces(1023, 1023);
    assertEquals(pieces, [512, 512]);
    const wide = probePieces(2 ** 31 - 1, 2 ** 31 - 1);
    assertEquals(wide, [2 ** 30, 2 ** 30]);
  });
  it("negative and non-integer sizes are a RangeError", () => {
    assertThrows(() => probePieces(-4, 1024), RangeError);
    assertThrows(() => probePieces(1.5, 1024), RangeError);
    assertThrows(() => probePieces(4, 2), RangeError);
  });
});

describe("fitsHeadroom (real GPU)", () => {
  it.skipIf(!GPU_AVAILABLE)(
    "the largest amount that fits still fits on the next probes and can really be allocated",
    async () => {
      // Dawn returns a destroyed buffer's memory only when a later submit completes, so a probe that
      // does not settle that stays on the device: the second probe of this amount gives false, and
      // the allocation the probe was checking fails.
      const gpu = await acquireGpu();
      try {
        let bytes = 64 * 1024 * MIB;
        while (bytes > 64 * MIB && !(await fitsHeadroom(gpu, bytes))) bytes /= 2;
        assert(bytes > 64 * MIB, "not even 64 MiB fits");
        for (let probe = 0; probe < 3; probe += 1) {
          assert(await fitsHeadroom(gpu, bytes), `probe ${probe} of ${bytes} bytes`);
        }
        const buffers = probePieces(bytes, gpu.limits.maxBufferSize).map((size) =>
          gpu.device.createBuffer({ size, usage: BUFFER_USAGE.STORAGE }),
        );
        gpu.device.pushErrorScope("out-of-memory");
        const extra = gpu.device.createBuffer({ size: 4, usage: BUFFER_USAGE.STORAGE });
        const failure = await gpu.device.popErrorScope();
        assertEquals(failure, null);
        extra.destroy();
        for (const buffer of buffers) buffer.destroy();
      } finally {
        gpu.destroy();
      }
    },
  );

  it.skipIf(!GPU_AVAILABLE)(
    "a small amount fits, and an amount above the ceiling gives false with the device kept alive",
    async () => {
      const gpu = await acquireGpu();
      try {
        let lost = false;
        gpu.onLost(() => {
          lost = true;
        });
        assertEquals(await fitsHeadroom(gpu, 4), true);
        // 1 TiB — fits on no device (split into maxBufferSize pieces allocated in turn; OOMs part
        // way).
        assertEquals(await fitsHeadroom(gpu, 2 ** 40), false);
        // Not fatal: a small amount fits right after, and a real allocation + write + fence also
        // succeed.
        assertEquals(await fitsHeadroom(gpu, 4), true);
        gpu.device.pushErrorScope("out-of-memory");
        const buffer = gpu.device.createBuffer({
          size: 16 * MIB,
          usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST,
        });
        gpu.device.queue.writeBuffer(buffer, 0, new Uint8Array(16 * MIB));
        assertEquals(await gpu.device.popErrorScope(), null);
        await gpu.device.queue.onSubmittedWorkDone();
        buffer.destroy();
        assertEquals(lost, false);
        assertEquals(gpu.lost, undefined);
      } finally {
        gpu.destroy();
      }
    },
  );

  it.skipIf(!GPU_AVAILABLE)(
    "after true the same amount can really be allocated (it returns only after the release)",
    async () => {
      const gpu = await acquireGpu();
      try {
        const bytes = 512 * MIB;
        if (!(await fitsHeadroom(gpu, bytes))) {
          // A machine where even 512 MiB does not fit breaks the premise (out of scope — fail
          // explicitly).
          throw new Error("cannot check on a device where a 512 MiB probe allocation fails");
        }
        const pieces = probePieces(bytes, gpu.limits.maxBufferSize);
        gpu.device.pushErrorScope("out-of-memory");
        const buffers = pieces.map((size) =>
          gpu.device.createBuffer({ size, usage: BUFFER_USAGE.STORAGE }),
        );
        assertEquals(await gpu.device.popErrorScope(), null);
        for (const buffer of buffers) buffer.destroy();
        await gpu.device.queue.onSubmittedWorkDone();
      } finally {
        gpu.destroy();
      }
    },
  );
});
