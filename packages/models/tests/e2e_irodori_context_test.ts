/**
 * Irodori の**段 6'（条件側 K/V 射影 `dit_context` — ADR 0114）の門**（実 GPU）。
 *
 * 押さえるのは 2 点:
 *
 * - **(a) 回る回数** — 生成 1 回で `dit_context` は 1 回だけ、`dit` は forward 本数ぶん回る。
 *   常駐経路（`enqueue`）とホスト経路（`run` — `onEvent` 購読）の両方で見る。ホイストが外れて
 *   forward ごとに回る形・前段が回らない形・2 経路のどちらかだけが別の回し方をする形を、数で
 *   捕まえる（WAV 門は数値が変わらない限り緑なので、ホイストの外れは見えない）。
 * - **(b) 検出力の確認（故障注入）** — `dit` へ渡る `context_k_0` と `context_v_0` を入れ替えると、
 *   生成 WAV の sha256 が差し替え無しの生成と**変わる**。変わらなければ、K / V の配線は WAV の
 *   sha256 門（`e2e_irodori_wav_test.ts`）から見えていない — 「グラフを割っても参照 sha が
 *   不変」を分割の正しさの証拠に使えなくなる。
 *
 * prototype の差し替えはこのテスト内だけで、必ず戻す（`e2e_irodori_cleanup_test.ts` と同じ
 * 作法 — 他ファイルは Deno の別 worker）。
 *
 * MUST: 資産は `models/karume-irodori-v4-small/` の `i8` 席（untracked・実 GPU 機のローカル資産）。
 * 欠けた環境と GPU 無し環境は生成コマンド付きで**明示 SKIP** する（ADR 0005）。
 */

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import type { RunInputs } from "@karume/runtime";
import {
  encodeWav,
  type IrodoriGenerateRequest,
  IrodoriPipeline,
  type IrodoriRunComponent,
} from "../mod.ts";
import {
  DIST_COMMAND,
  hasQuantSeat,
  loadLocalAssets,
  MODEL,
  readManifest,
} from "./helpers/irodori-assets.ts";
import { sha256Hex } from "./helpers/container-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { Session } from "../../runtime/src/runtime/executor.ts";

const QUANT = "i8";

if (!hasQuantSeat(QUANT)) {
  console.warn(
    `[karume] Irodori の dit_context 門の ${QUANT} 資産が無いため SKIP。生成: ${DIST_COMMAND}`,
  );
}
const RUNNABLE = GPU_AVAILABLE && hasQuantSeat(QUANT);

/**
 * 短く保つ要求 — `durationSeconds` で S を固定して `duration` を回さず、参照も caption も
 * 渡さない（text の CFG 1 本だけが立つので、forward 本数は step 数より多い）。
 */
const REQUEST = {
  text: "こんにちは、これはテストです。",
  seed: 11,
  durationSeconds: 1,
} as const satisfies IrodoriGenerateRequest;

/** Session への呼び出し 1 本の種別（`dit` だけが `x_t` を、`dit_context` だけが `speaker_state` を受ける）。 */
const kindOf = (inputs: RunInputs): "context" | "dit" | undefined =>
  Object.hasOwn(inputs, "x_t")
    ? "dit"
    : Object.hasOwn(inputs, "speaker_state")
    ? "context"
    : undefined;

type Calls = { context: number; dit: number };

const wavSha = async (pipeline: IrodoriPipeline): Promise<string> => {
  const audio = await pipeline.generate(REQUEST);
  return await sha256Hex(encodeWav(audio.data, audio.sampleRate));
};

Deno.test({
  name: "e2e(実GPU): dit_context は生成 1 回に 1 回だけ回り、K / V の配線は WAV sha から見える",
  ignore: !RUNNABLE,
  fn: async (t) => {
    const manifest = readManifest();
    const assets = await loadLocalAssets(manifest, QUANT);
    const observed: IrodoriRunComponent[] = [];
    await using pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: QUANT,
      onRunDiagnostics: (component) => observed.push(component),
    });
    const run = Session.prototype.run;
    const enqueue = Session.prototype.enqueue;

    await t.step(
      "(a) 回る回数 — 常駐経路とホスト経路の両方で dit_context 1 回・dit forward 本数",
      async () => {
        const calls: { run: Calls; enqueue: Calls } = {
          run: { context: 0, dit: 0 },
          enqueue: { context: 0, dit: 0 },
        };
        const reset = (): void => {
          calls.run = { context: 0, dit: 0 };
          calls.enqueue = { context: 0, dit: 0 };
          observed.length = 0;
        };
        const count = (via: keyof typeof calls, inputs: RunInputs): void => {
          const kind = kindOf(inputs);
          if (kind !== undefined) calls[via][kind] += 1;
        };
        const observedCount = (component: IrodoriRunComponent): number =>
          observed.filter((name) => name === component).length;
        try {
          Session.prototype.run = function (inputs, bindings, generation) {
            count("run", inputs);
            return run.call(this, inputs, bindings, generation);
          };
          Session.prototype.enqueue = function (inputs, options) {
            count("enqueue", inputs);
            return enqueue.call(this, inputs, options);
          };

          // 常駐経路（計測の無い device・onEvent 無し）— 前段も本体も enqueue だけで回る。
          reset();
          const resident = await pipeline.generateLatent(REQUEST);
          assert(resident.forwards > 1, `forwards ${resident.forwards} では「1 回」と区別できない`);
          assertEquals(
            calls.enqueue,
            { context: 1, dit: resident.forwards },
            "常駐経路の enqueue 数",
          );
          assertEquals(calls.run, { context: 0, dit: 0 }, "常駐経路で run が走った");
          // 観測席（`onRunDiagnostics`）も同じ本数を名乗る — census はこの名乗りで部品を分ける。
          assertEquals(observedCount("dit-context"), 1);
          assertEquals(observedCount("dit"), resident.forwards);

          // ホスト経路（onEvent 購読）— 前段も本体も run で回り、段は dit の前で開いて閉じる。
          reset();
          const stages: string[] = [];
          const host = await pipeline.generateLatent({
            ...REQUEST,
            onEvent: (event) => {
              if (event.kind === "stage") stages.push(`${event.component}:${event.at}`);
            },
          });
          assertEquals(host.forwards, resident.forwards, "2 経路の forward 本数が違う");
          assertEquals(calls.run, { context: 1, dit: host.forwards }, "ホスト経路の run 数");
          assertEquals(calls.enqueue, { context: 0, dit: 0 }, "ホスト経路で enqueue が走った");
          assertEquals(observedCount("dit-context"), 1);
          assertEquals(observedCount("dit"), host.forwards);
          const ditStart = stages.indexOf("dit:start");
          assertEquals(
            stages.slice(ditStart - 2, ditStart + 1),
            ["dit-context:start", "dit-context:end", "dit:start"],
            `段の並びが違う（${stages.join(" / ")}）`,
          );
          // 2 経路の出力はビット同一（`runDitLoopResident` の MUST — 段 6' の写し方の違いを含む）。
          const bytes = (data: Float32Array<ArrayBuffer>): Uint8Array<ArrayBuffer> =>
            new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
          assertEquals(
            bytes(host.data),
            bytes(resident.data),
            "常駐経路とホスト経路の latent がビット同一でない",
          );
        } finally {
          Session.prototype.run = run;
          Session.prototype.enqueue = enqueue;
        }
      },
    );

    await t.step(
      "(b) 故障注入 — context_k_0 と context_v_0 を入れ替えると WAV sha が変わる",
      async () => {
        const baseline = await wavSha(pipeline);
        // 決定性が無いと「入れ替えで変わった」を読めないので、差し替え無しの 2 回を先に見る。
        assertEquals(
          await wavSha(pipeline),
          baseline,
          "差し替え無しの生成 2 回の WAV が一致しない",
        );
        let swapped = 0;
        try {
          Session.prototype.enqueue = function (inputs, options) {
            if (kindOf(inputs) !== "dit") return enqueue.call(this, inputs, options);
            if (!Object.hasOwn(inputs, "context_k_0") || !Object.hasOwn(inputs, "context_v_0")) {
              throw new Error(
                `dit の入力に context_k_0 / context_v_0 が無い（${
                  Object.keys(inputs).join(", ")
                }）`,
              );
            }
            swapped += 1;
            return enqueue.call(this, {
              ...inputs,
              context_k_0: inputs["context_v_0"],
              context_v_0: inputs["context_k_0"],
            }, options);
          };
          const faulted = await wavSha(pipeline);
          assert(
            swapped > 0,
            "dit の enqueue が 1 本も差し替わっていない（常駐経路を通っていない）",
          );
          assertNotEquals(
            faulted,
            baseline,
            "K / V を入れ替えても WAV が変わらない — WAV の sha256 門は条件側 K/V の配線を見ていない",
          );
        } finally {
          Session.prototype.enqueue = enqueue;
        }
      },
    );
  },
});
