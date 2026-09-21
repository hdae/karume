/**
 * shard 面のロード経路（`src/hub/components.ts`）の門のうち、実 GPU を取る 4 点
 * （GPU に依らない門は shard_loading_test.ts）:
 *
 * ④ **Session 構築は進捗を動かさない**（prefetch 済みキャッシュの読み直しで `complete` が
 *    もう一度飛ぶと、集約 `loaded` が二重計上になる）。
 * ② **同じ供給口から Session を 2 本続けて張れる**（使い切った列を使い回すと 2 本目が空の列を
 *    受ける）。
 * ⑤ **ロード時の `signal` は Session 構築へ持ち越さない**（`AbortSignal.timeout` や画面の
 *    アンマウントで「ロードは成功したのに以後の生成が全部落ちる」形を作らない）。対の
 *    「abort 済みで始めたロードは落ちる」は GPU を取らないので向こうに置く。
 * ⑨ **`requiredLimits` 超過は自前 GPU 取得の経路でも重み shard を取らない**（突き合わせ相手が
 *    `readAdapterLimits()` のアダプタ実測値に変わる側 — ADR 0089 決定 5）。
 */

import { assertEquals, assertRejects } from "@std/assert";
import type { AssetProgress } from "@karume/hub";
import { acquireGpu } from "@karume/runtime";
import { loadShardComponents } from "../src/hub/components.ts";
import { Siglip2Pipeline } from "../src/siglip2/pipeline.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  HUB_URL,
  NO_FAMILY_GATE,
  prepareSiglip2,
  prepareTwoShard,
  quantsRequiring,
  REPO,
  SHA,
} from "./helpers/shard-loading-fixture.ts";

Deno.test({
  name: "loadShardComponents: Session 構築は進捗を 1 イベントも動かさない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const { loaded, files, mock, hubOptions } = await prepareTwoShard();

    const events: AssetProgress[] = [];
    const { open } = await loadShardComponents(
      "test.fromPretrained",
      loaded,
      files,
      ["dit"],
      NO_FAMILY_GATE,
      { ...hubOptions, onProgress: (progress) => events.push(progress) },
    );
    const afterLoad = events.length;
    const calls = mock.paths.length;
    // complete はファイル数ぶんちょうど（ロードを抜けた時点で全ファイルが終端に達している）。
    assertEquals(events.filter((event) => event.phase === "complete").length, 2);

    const gpu = await acquireGpu();
    try {
      // 重み shard は 2 本目にしかないので、Session が張れた時点で shard 列は流れている。
      const session = await open("dit").createSession(gpu);
      await session.dispose();
    } finally {
      gpu.destroy();
    }

    assertEquals(events.length, afterLoad, "Session 構築が進捗イベントを追加している");
    assertEquals(mock.paths.length, calls, "Session 構築が network へ出ている");
  },
});

Deno.test({
  name: "loadShardComponents: 同じ供給口から Session を 2 本続けて張れる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const { loaded, files, hubOptions } = await prepareTwoShard();
    const { open } = await loadShardComponents(
      "test.fromPretrained",
      loaded,
      files,
      ["dit"],
      NO_FAMILY_GATE,
      hubOptions,
    );

    const gpu = await acquireGpu();
    try {
      // 2 本目が張れるのは、shard 列を**呼ぶたびに**新しく作っているとき（使い切った列を
      // 使い回すと 2 本目が空の列を受けて「重みが足りない」で落ちる）。グラフ shard も列に
      // 含むようになった後も同じ規律が要る。
      for (let index = 0; index < 2; index += 1) {
        const session = await open("dit").createSession(gpu);
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "loadShardComponents: ロード時の signal は Session 構築へ持ち越さない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const { loaded, files, hubOptions } = await prepareTwoShard();
    const controller = new AbortController();
    const { open } = await loadShardComponents(
      "test.fromPretrained",
      loaded,
      files,
      ["dit"],
      NO_FAMILY_GATE,
      { ...hubOptions, signal: controller.signal },
    );

    // 呼び手の中断ノブは「このロード 1 回」の寿命のもの — ロード成功の後に発火するのは
    // `AbortSignal.timeout` でも画面のアンマウントでもごく普通の綴り。
    controller.abort();

    const gpu = await acquireGpu();
    try {
      // 持ち越していると相 2 の `throwIfAborted()` で落ちる（キャッシュ完備でも確実に）。
      const session = await open("dit").createSession(gpu);
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "requiredLimits 超過は自前 GPU 取得の経路でも重み shard を取らない（アダプタ実測）",
  // MUST: アダプタ無し環境は明示 SKIP（ADR 0005）。この経路の突き合わせ相手は
  // `readAdapterLimits()` なので、アダプタが無いと GpuUnavailableError に化けて門が見えない。
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // どの実機も満たせない要求（`Number.MAX_SAFE_INTEGER` — manifest が受ける最大値）。
    const { refs, mock, caches } = await prepareSiglip2({
      quants: quantsRequiring(Number.MAX_SAFE_INTEGER),
    });

    const error = await assertRejects(
      () =>
        Siglip2Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          { fetch: mock.fetch, caches },
        ),
      Error,
    );
    if (!error.message.includes("maxBufferSize")) {
      throw new Error(`limits 門の文言でない: ${error.message}`);
    }
    assertEquals(mock.paths.includes(refs.graph.path), true);
    assertEquals(mock.paths.includes(refs.weights.path), false);
  },
});
