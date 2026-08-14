import { assertRejects } from "@std/assert";
import { createGuardedFetch } from "../src/transport.ts";

Deno.test("createGuardedFetch: 予算の無い URL は素通しする", async () => {
  let calls = 0;
  const base: typeof globalThis.fetch = () => {
    calls++;
    return Promise.resolve(new Response("ok", { status: 200 }));
  };
  const guarded = createGuardedFetch(base, new Map());
  const response = await guarded("https://example.test/no-budget");
  await response.body?.cancel();
  if (calls !== 1) throw new Error(`expected 1 call, got ${calls}`);
});

Deno.test(
  "createGuardedFetch: init.signal 無しで Request（既に abort 済み）を渡すと abort として reject する",
  async () => {
    // Request.signal を合成に含めない実装だと、この abort が握り潰されて base fetch が
    // 実際に発火してしまう — network に出ずに検出できるよう base 側で確認する。
    let baseCalled = false;
    const base: typeof globalThis.fetch = (_input, init) => {
      baseCalled = true;
      // init.signal が既に abort 済みなら実装は reject するはずなので、ここに到達したら
      // signal が正しく合成されていない。
      if (init?.signal?.aborted) {
        return Promise.reject(new DOMException("propagated abort", "AbortError"));
      }
      return Promise.resolve(new Response("ok", { status: 200 }));
    };
    const guarded = createGuardedFetch(
      base,
      new Map([["https://example.test/budgeted", {
        maxBytes: 1024,
        exact: false,
        violation: (actual) => new Error(`too big: ${actual}`),
      }]]),
    );

    const controller = new AbortController();
    controller.abort(new DOMException("already aborted", "AbortError"));
    const request = new Request("https://example.test/budgeted", { signal: controller.signal });

    await assertRejects(
      () => guarded(request),
      DOMException,
    );
    if (!baseCalled) {
      throw new Error("base fetch was not invoked — outer signal was not propagated at all");
    }
  },
);
