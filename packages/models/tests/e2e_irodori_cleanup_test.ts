// 常駐ループの本体・batch 終了・複数 cleanup が失敗しても、全資源の解放を試みる門。
// 本物の解放の後に故障を注入するので、試験自体は VRAM や参照を残さない。
// prototype の差し替えはこのテスト内だけで、必ず戻す。他ファイルは Deno の別 worker。
import { assert, assertEquals, assertRejects } from "@std/assert";
import { IrodoriPipeline } from "../mod.ts";
import {
  DIST_COMMAND,
  hasQuantSeat,
  loadLocalAssets,
  MODEL,
  readManifest,
} from "./helpers/irodori-assets.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { Session } from "../../runtime/src/runtime/executor.ts";
import { BatchScope, ResidentTensor } from "../../runtime/src/gpu/device.ts";

if (!hasQuantSeat("i8")) {
  console.warn(`[karume] Irodori cleanup テストの i8 資産が無いため SKIP。生成: ${DIST_COMMAND}`);
}
Deno.test({
  name:
    "Irodori 常駐ループ: 本体と多重 cleanup の失敗を保ち、全 Session と resident を解放する（実 GPU）",
  ignore: !GPU_AVAILABLE || !hasQuantSeat("i8"),
  fn: async () => {
    const manifest = readManifest();
    const assets = await loadLocalAssets(manifest, "i8");
    const pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: "i8",
    });
    const enqueue = Session.prototype.enqueue;
    const sessionDispose = Session.prototype.dispose;
    const residentDispose = ResidentTensor.prototype.dispose;
    const finish = BatchScope.prototype.finish;
    const bodyFault = new Error("probe: enqueue failed");
    const finishFault = new Error("probe: finish failed");
    const sessionFaults = [
      new Error("probe: dispose session 1"),
      new Error("probe: dispose session 2"),
    ];
    const residentFault = new Error("probe: dispose resident 1");
    let entered = false;
    let sessions = 0;
    let residents = 0;
    let finishes = 0;
    const order: string[] = [];
    try {
      Session.prototype.enqueue = function (): Promise<never> {
        entered = true;
        order.push("enqueue");
        return Promise.reject(bodyFault);
      };
      BatchScope.prototype.finish = async function () {
        await finish.call(this);
        if (entered) {
          finishes++;
          order.push("finish");
          throw finishFault;
        }
      };
      Session.prototype.dispose = async function () {
        await sessionDispose.call(this);
        if (entered) {
          sessions++;
          order.push(`session${sessions}`);
          if (sessions <= sessionFaults.length) throw sessionFaults[sessions - 1];
        }
      };
      ResidentTensor.prototype.dispose = function () {
        residentDispose.call(this);
        if (entered) {
          residents++;
          order.push(`resident${residents}`);
          if (residents === 1) throw residentFault;
        }
      };
      const error = await assertRejects(() =>
        pipeline.generateLatent({ text: "こんにちは。", durationSeconds: 0.5, seed: 7 })
      );
      const flatten = (error: unknown): unknown[] =>
        error instanceof AggregateError ? error.errors.flatMap(flatten) : [error];
      const failures = flatten(error);
      assertEquals(failures, [bodyFault, finishFault, ...sessionFaults, residentFault]);
      assertEquals(finishes, 1);
      assertEquals(sessions, 3);
      // 4 作業用テンソル + text / speaker / caption の 3 条件。
      assertEquals(residents, 7);
      assertEquals(order.slice(0, 5), ["enqueue", "finish", "session1", "session2", "session3"]);
      assert(order.slice(5).every((value) => value.startsWith("resident")));
      console.log(
        JSON.stringify({
          finishes,
          sessions,
          residents,
          order,
          failures: failures.map((error) => error instanceof Error ? error.message : String(error)),
        }),
      );
    } finally {
      Session.prototype.enqueue = enqueue;
      Session.prototype.dispose = sessionDispose;
      ResidentTensor.prototype.dispose = residentDispose;
      BatchScope.prototype.finish = finish;
      await pipeline.dispose();
    }
  },
});
