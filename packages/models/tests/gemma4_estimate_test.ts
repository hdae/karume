// gemma4 の見積り（`Gemma4Pipeline.estimateSessionMemory`）が `estimateGraphMemory` へ渡す
// generation の指定（`gemma4EstimateGenerations`）の配線。GPU は使わない。
//
// 実資産の gemma4 E2B ではバケット形がピークを支配しないので、`peakAccountedBytes` を見る門
// （`e2e_gemma4_drafter_test.ts` ②）はバケットの配線を外しても割れない。配線はここで直接固定する。

import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { gemma4EstimateGenerations } from "../src/gemma/pipeline.ts";
import { fakeSession, programOf } from "./helpers/generation-fake.ts";

/** pipeline の chunk 長 16・バケット梯子 [4, 8]（容量記号 `C`・行数記号 `R`）。 */
const wiring = () =>
  programOf(fakeSession(), { chunkLength: 16, chunkBuckets: [4, 8], capacity: 64 });

describe("gemma4 の見積りが本体へ渡す generation の指定", () => {
  it("pipeline の chunkBuckets のうち見積る chunkLength 未満の段を渡す", () => {
    const at = (chunkLength: number) =>
      gemma4EstimateGenerations(wiring(), { capacity: 64, chunkLength }).target.chunkBuckets;
    assertEquals(at(16), [4, 8], "pipeline と同じ chunk 長では梯子をそのまま渡す");
    assertEquals(at(8), [4], "chunk 長と同じ段は受理集合（chunk 長未満）に入らない");
    assertEquals(at(4), [], "受理集合が空なら 1 段も渡さない");
  });

  it("仮定の chunkLength を pipeline の値より大きくしても、渡す段は pipeline の梯子のまま", () => {
    const { target } = gemma4EstimateGenerations(wiring(), { capacity: 64, chunkLength: 32 });
    assertEquals(target.chunkBuckets, [4, 8]);
  });

  it("見積る chunk 長と容量をそのまま渡す", () => {
    const { target } = gemma4EstimateGenerations(wiring(), { capacity: 48, chunkLength: 16 });
    assertEquals(target.chunkLength, 16);
    assertEquals(target.bindings, { C: 48 });
  });

  it("drafter の無い pipeline では追加シナリオを足さない", () => {
    const { target } = gemma4EstimateGenerations(wiring(), { capacity: 64, chunkLength: 16 });
    assertEquals(target.scenarios, undefined);
  });

  it("投機ありでは verify を k+1 行・物理行数はバケットへ丸めた行数で足す", () => {
    const { target } = gemma4EstimateGenerations(wiring(), {
      capacity: 64,
      chunkLength: 16,
      verifyQueryLength: 3,
    });
    assertEquals(target.scenarios, [{ name: "verify", chunkLength: 4, bindings: { R: 3 } }]);
  });
});

describe("gemma4 の見積りが drafter（借り手 context）へ渡す generation の指定", () => {
  it("chunk 長 1・貸し手と同じ容量で、バケットは渡さない", () => {
    const { borrower } = gemma4EstimateGenerations(wiring(), {
      capacity: 48,
      chunkLength: 16,
      verifyQueryLength: 4,
    });
    assertEquals(borrower, { chunkLength: 1, bindings: { C: 48 } });
  });
});
