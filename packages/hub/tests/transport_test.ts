import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { type ByteBudget, createGuardedFetch } from "../src/transport.ts";

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

/** size ちょうどを要求する門（`fetchAssets` / 相 1 が張るのと同じ形）。 */
const exactBudget = (maxBytes: number): ByteBudget => ({
  maxBytes,
  exact: true,
  violation: (actual, where) => new Error(`budget: ${where} = ${actual}`),
});

Deno.test("createGuardedFetch: 予算表と引き当てで URL 表記が違っても門は外れない", async (t) => {
  // 予算表のキーは `hfResolveUrl` の生出力（アプリ指定の hubUrl がそのまま載る）だが、下層が
  // `fetch` へ渡すのは正規化済みの文字列。両者が食い違うと門が沈黙で無効化される。
  const spellings: readonly (readonly [string, string])[] = [
    ["既定ポートの明記", "https://mirror.example:443/owner/name/resolve/abc/model.safetensors"],
    ["大文字ホスト", "https://MIRROR.example/owner/name/resolve/abc/model.safetensors"],
    ["fragment 付き", "https://mirror.example/owner/name/resolve/abc/model.safetensors#frag"],
  ];

  for (const [label, declared] of spellings) {
    await t.step(label, async () => {
      // 下層が実際に渡す形（`new URL(u).href` + fragment 剥がし）。
      const requested = new URL(declared);
      requested.hash = "";
      let calls = 0;
      const base: typeof globalThis.fetch = () => {
        calls++;
        // 予算 8 バイトに対して 9 バイトを名乗る応答（門が生きていれば受信前に落ちる）。
        return Promise.resolve(new Response("123456789", { headers: { "content-length": "9" } }));
      };
      const guarded = createGuardedFetch(base, new Map([[declared, exactBudget(8)]]));

      const error = await assertRejects(() => guarded(requested.href), Error);
      assertEquals(error.message, "budget: content-length = 9", "門が引き当てに失敗している");
      assertEquals(calls, 1);
    });
  }
});

Deno.test("createGuardedFetch: 解釈できない予算 URL は黙って落とさず fail loudly", () => {
  // 表から黙って落とすと、その 1 本だけ門が外れた `fetch` が出来上がる。
  assertThrows(
    () => createGuardedFetch(globalThis.fetch, new Map([["/relative/path", exactBudget(8)]])),
    Error,
    "予算表の URL を解釈できない",
  );
});
