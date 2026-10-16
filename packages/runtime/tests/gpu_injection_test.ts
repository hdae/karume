// `AcquireGpuOptions.gpu` — the WebGPU implementation can be passed explicitly (Node applications
// hand over Dawn's `GPU` instead of defining `navigator.gpu`). No real GPU is needed: the fake
// adapter counts the requests, and `navigator.gpu` is made to throw so that any fallback to the
// global is caught.

import { assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "@karume/test-support/test";
import { acquireGpu, readAdapterLimits } from "../src/gpu/acquire.ts";
import { REQUIRED_LIMIT_KEYS } from "../src/gpu/device.ts";
import { fakeDevice } from "./helpers/fake-gpu.ts";

const fakeGpu = (): { readonly gpu: GPU; readonly requests: () => number } => {
  let requests = 0;
  const limits = Object.fromEntries(REQUIRED_LIMIT_KEYS.map((key) => [key, 1 << 20]));
  const adapter = {
    limits,
    features: new Set<string>(),
    requestDevice: (): Promise<GPUDevice> =>
      Promise.resolve(Object.assign(fakeDevice(), { limits })),
  };
  const gpu = {
    requestAdapter: (): Promise<GPUAdapter> => {
      requests += 1;
      return Promise.resolve(adapter as unknown as GPUAdapter);
    },
  } as unknown as GPU;
  return { gpu, requests: () => requests };
};

/** Runs `body` while any read of `navigator.gpu` throws (the global must not be consulted). */
const withoutNavigatorGpu = async (body: () => Promise<void>): Promise<void> => {
  const original = Object.getOwnPropertyDescriptor(navigator, "gpu");
  Object.defineProperty(navigator, "gpu", {
    get: () => {
      throw new Error("navigator.gpu was read although a GPU instance was injected");
    },
    configurable: true,
  });
  try {
    await body();
  } finally {
    if (original === undefined) Reflect.deleteProperty(navigator, "gpu");
    else Object.defineProperty(navigator, "gpu", original);
  }
};

describe("AcquireGpuOptions.gpu（WebGPU 実装の注入）", () => {
  it("acquireGpu は渡された GPU からアダプタを取り、navigator.gpu を読まない", async () => {
    const fake = fakeGpu();
    await withoutNavigatorGpu(async () => {
      const gpu = await acquireGpu({ gpu: fake.gpu });
      gpu.destroy();
    });
    assertEquals(fake.requests(), 1);
  });

  it("readAdapterLimits も同じ口を使う", async () => {
    const fake = fakeGpu();
    await withoutNavigatorGpu(async () => {
      const limits = await readAdapterLimits({ gpu: fake.gpu });
      assertEquals(typeof limits, "object");
    });
    assertEquals(fake.requests(), 1);
  });

  it("渡さなければ navigator.gpu に戻る（無ければ GpuUnavailableError）", async () => {
    const original = Object.getOwnPropertyDescriptor(navigator, "gpu");
    Object.defineProperty(navigator, "gpu", { value: undefined, configurable: true });
    try {
      await assertRejects(() => acquireGpu(), Error, "navigator.gpu が存在しない");
    } finally {
      if (original === undefined) Reflect.deleteProperty(navigator, "gpu");
      else Object.defineProperty(navigator, "gpu", original);
    }
  });
});
