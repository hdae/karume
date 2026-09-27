// 常駐ループの本体・batch 終了・複数 cleanup が失敗しても、全資源の解放を試みる門。
// 本物の解放の後に故障を注入するので、試験自体は VRAM や参照を残さない。
// prototype の差し替えはこのテスト内だけで、必ず戻す。他ファイルは Deno の別 worker。
//
// 段 6'（条件側 K/V 射影 `dit_context` — ADR 0114）は前段の batch で回り切ってから Session を
// 畳み、その後で `dit` / combine / euler を開く（VRAM ピークの MUST — `dit-loop.ts` の
// `projectContextOnGpu`）。故障はループ本体（`dit` の最初の enqueue）に注入し、前段は素通しに
// して、その順序と解放の本数を同じ 1 回の生成で見る。
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
    const build = Session.build;
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
    // 段の進み: 前段（`dit_context` の enqueue を見た後）→ ループ（`dit` の enqueue で故障）。
    let phase: "before" | "context" | "loop" = "before";
    let sessions = 0;
    let loopSessions = 0;
    let residents = 0;
    let finishes = 0;
    const order: string[] = [];
    try {
      // Session の構築は全て `Session.build` を通る（容器の部品もホストの小グラフも）。
      Session.build = function (...args: Parameters<typeof build>) {
        order.push("open");
        return build.apply(this, args);
      };
      Session.prototype.enqueue = function (inputs, options): Promise<void> {
        // `dit` だけが `x_t` を、前段の `dit_context` だけが `speaker_state` を受ける（小グラフは別名）。
        if (Object.hasOwn(inputs, "x_t")) {
          phase = "loop";
          order.push("enqueue:dit");
          return Promise.reject(bodyFault);
        }
        if (Object.hasOwn(inputs, "speaker_state")) {
          phase = "context";
          order.push("enqueue:dit_context");
        }
        return enqueue.call(this, inputs, options);
      };
      BatchScope.prototype.finish = async function () {
        await finish.call(this);
        if (phase === "context") order.push("finish:dit_context");
        if (phase === "loop") {
          finishes++;
          order.push("finish");
          throw finishFault;
        }
      };
      Session.prototype.dispose = async function () {
        await sessionDispose.call(this);
        if (phase === "before") return;
        sessions++;
        order.push("dispose");
        if (phase === "loop") {
          loopSessions++;
          if (loopSessions <= sessionFaults.length) throw sessionFaults[loopSessions - 1];
        }
      };
      ResidentTensor.prototype.dispose = function () {
        residentDispose.call(this);
        if (phase === "loop") {
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
      // dit_context（前段で解放済み）+ dit + combine + euler。
      assertEquals(sessions, 4);
      // 4 作業用テンソル + 条件側 K / V 24 本（12 ブロック × K / V）。
      assertEquals(residents, 28);
      // 前段の Session は `dit` を開く**前**に畳まれている（VRAM ピークの MUST）。
      const start = order.lastIndexOf("open", order.indexOf("enqueue:dit_context"));
      assert(start >= 0, `dit_context の Session 構築が観測されない（${order.join(" / ")}）`);
      assertEquals(order.slice(start, start + 12), [
        "open",
        "enqueue:dit_context",
        "finish:dit_context",
        "dispose",
        "open",
        "open",
        "open",
        "enqueue:dit",
        "finish",
        "dispose",
        "dispose",
        "dispose",
      ]);
      assert(order.slice(start + 12).every((value) => value.startsWith("resident")));
      console.log(
        JSON.stringify({
          finishes,
          sessions,
          residents,
          order: order.slice(start),
          failures: failures.map((error) => error instanceof Error ? error.message : String(error)),
        }),
      );
    } finally {
      Session.build = build;
      Session.prototype.enqueue = enqueue;
      Session.prototype.dispose = sessionDispose;
      ResidentTensor.prototype.dispose = residentDispose;
      BatchScope.prototype.finish = finish;
      await pipeline.dispose();
    }
  },
});
