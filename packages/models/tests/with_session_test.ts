/**
 * `withSession`（`src/session/with-session.ts`）の実行スコープ契約のうち、GPU を取らない枝
 * （Session を張って run / dispose する枝は gpu_with_session_test.ts）:
 *
 * - Session の構築が落ちたら body は呼ばれず、構築の例外がそのまま呼び手へ渡る。
 *
 * NOTE: `withSession` は最初に `createSession(gpu, …)` を呼ぶだけで、それより前に gpu を触らない。
 * 構築が即 reject する部品なら実アダプタは要らないので、`GpuContext` は偽物（`fake-gpu.ts`）を渡す。
 */

import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import type { ModelComponent } from "../src/hub/components.ts";
import { withSession } from "../src/session/with-session.ts";
import { openerOf, single } from "./helpers/asset-container-fixture.ts";

describe("withSession", () => {
  describe("Session の構築が落ちたとき", () => {
    it("body を呼ばず、構築の例外をそのまま伝える", async () => {
      const failure = new Error("Session を張れなかった");
      const open = await openerOf({ dit: single() });
      const component: ModelComponent = {
        ...open("dit"),
        createSession: () => Promise.reject(failure),
      };
      let bodyCalled = false;

      const error = await assertRejects(() =>
        withSession(fakeGpuContext(fakeDevice()), component, {}, undefined, () => {
          bodyCalled = true;
          return Promise.resolve();
        })
      );

      assertStrictEquals(error, failure);
      assertEquals(bodyCalled, false);
    });
  });
});
