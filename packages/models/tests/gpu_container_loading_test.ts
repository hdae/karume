/**
 * コンテナ経路のロード（`src/hub/components.ts`）の門のうち、実 GPU を取る 4 点
 * （GPU に依らない門は container_loading_test.ts）:
 *
 * ④ **Session 構築は進捗を動かさない**（prefetch 済みキャッシュの読み直しで `complete` が
 *    もう一度飛ぶと、集約 `loaded` が二重計上になる）。
 * ② **同じ供給口から Session を 2 本続けて張れる**（block は呼ぶたびに part 順で読み直す）。
 * ⑤ **ロード時の `signal` は Session 構築へ持ち越さない**（`AbortSignal.timeout` や画面の
 *    アンマウントで「ロードは成功したのに以後の生成が全部落ちる」形を作らない）。対の
 *    「abort 済みで始めたロードは落ちる」は GPU を取らないので向こうに置く。
 * ⑨ **`requiredLimits` 超過は自前 GPU 取得の経路でも重みの part を取らない**（突き合わせ相手が
 *    `readAdapterLimits()` のアダプタ実測値に変わる側 — ADR 0089 決定 5）。
 */

import { assertEquals, assertRejects } from "@std/assert";
import type { AssetProgress } from "@karume/hub";
import { acquireGpu } from "@karume/runtime";
import { loadContainerComponents } from "../src/hub/components.ts";
import { Siglip2Pipeline } from "../src/siglip2/pipeline.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  HUB_URL,
  NO_FAMILY_GATE,
  prepareComponent,
  prepareSiglip2,
  quantsRequiring,
  REPO,
  SHA,
} from "./helpers/container-loading-fixture.ts";

Deno.test({
  name: "loadContainerComponents: Session 構築は進捗を 1 イベントも動かさない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const rig = await prepareComponent();
    const fetched = rig.parts["dit"].filter((part) => part.size > 0).length;

    const events: AssetProgress[] = [];
    const { open } = await loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["dit"],
      NO_FAMILY_GATE,
      { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
    );
    const afterLoad = events.length;
    const calls = rig.mock.paths.length;
    // complete はファイル数ぶんちょうど（ロードを抜けた時点で全ファイルが終端に達している）。
    assertEquals(events.filter((event) => event.phase === "complete").length, fetched);

    const gpu = await acquireGpu();
    try {
      // 重みの block は末尾の part にしかないので、Session が張れた時点で block は読まれている。
      const session = await open("dit").createSession(gpu);
      await session.dispose();
    } finally {
      gpu.destroy();
    }

    assertEquals(events.length, afterLoad, "Session 構築が進捗イベントを追加している");
    assertEquals(rig.mock.paths.length, calls, "Session 構築が network へ出ている");
  },
});

Deno.test({
  name: "loadContainerComponents: 同じ供給口から Session を 2 本続けて張れる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const rig = await prepareComponent();
    const { open } = await loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["dit"],
      NO_FAMILY_GATE,
      rig.hubOptions,
    );

    const gpu = await acquireGpu();
    try {
      // 2 本目が張れるのは、block を**呼ぶたびに**読み直しているとき（1 度読んだら終わりの
      // 列にすると 2 本目が空を受けて「重みが足りない」で落ちる）。
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
  name: "loadContainerComponents: ロード時の signal は Session 構築へ持ち越さない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const rig = await prepareComponent();
    const controller = new AbortController();
    const { open } = await loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["dit"],
      NO_FAMILY_GATE,
      { ...rig.hubOptions, signal: controller.signal },
    );

    // 呼び手の中断ノブは「このロード 1 回」の寿命のもの — ロード成功の後に発火するのは
    // `AbortSignal.timeout` でも画面のアンマウントでもごく普通の綴り。
    controller.abort();

    const gpu = await acquireGpu();
    try {
      // 持ち越していると block の読みで落ちる（キャッシュ完備でも確実に）。
      const session = await open("dit").createSession(gpu);
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "requiredLimits 超過は自前 GPU 取得の経路でも重みの part を取らない（アダプタ実測）",
  // MUST: アダプタ無し環境は明示 SKIP（ADR 0005）。この経路の突き合わせ相手は
  // `readAdapterLimits()` なので、アダプタが無いと GpuUnavailableError に化けて門が見えない。
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // どの実機も満たせない要求（`Number.MAX_SAFE_INTEGER` — manifest が受ける最大値）。
    const rig = await prepareSiglip2({ quants: quantsRequiring(Number.MAX_SAFE_INTEGER) });
    const parts = rig.parts["vision"];

    const error = await assertRejects(
      () =>
        Siglip2Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          { fetch: rig.mock.fetch, caches: rig.hubOptions.caches },
        ),
      Error,
    );
    if (!error.message.includes("maxBufferSize")) {
      throw new Error(`limits 門の文言でない: ${error.message}`);
    }
    assertEquals(rig.mock.paths.includes(parts[0].path), true);
    for (const part of parts.slice(1)) {
      assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} を取っている`);
    }
  },
});
