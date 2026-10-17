/**
 * コンテナ経路のロード（`src/hub/components.ts`）の門のうち、GPU を取らずに回る側 —
 * 取得層だけを差し替えて固定する:
 *
 * ① **進捗はモデル全体で 1 本のまま**（取得が「descriptor の温め + 重みの part + 資産の
 *    全量面」へ割れても `loaded` は単調増加で全ファイルの size 合計に着地し、`complete` は
 *    ファイル数ぶん）。越境参照が自リポと同じ path を持っても、合計は 2 本ぶんのまま。
 * ② **admission は重みの part を取る前に落ちる**（part 0 の descriptor だけで capability 違反が
 *    決まり、重みの part の URL は 1 度も叩かれない — ADR 0070 決定 5 の存在理由そのもの）。
 *    この門は prefetch が admission の**後**に置かれていることの門でもある。
 * ③ **重みの part はロード時に落ち切る**（Session を 1 本も張らないうちに DL 済み — 遅延構築の
 *    家族で「初回実行まで DL が遅れ、ロード進捗にも現れない」形を無くす）。
 * ⑤ **abort 済みの `signal` で始めたロードは落ちる**（ロード中の中断契約が消えていないこと。
 *    対の「ロード時の signal を Session 構築へ持ち越さない」は GPU 側）。
 * ⑥ **重みの block のバイト列を握らない**（部品の供給口を保持したままでも常駐ゼロ）。gc を強制
 *    する必要があるので別プロセスの台本（`helpers/weight-block-retention.ts`）へ出し、ここでは
 *    終了コードだけを見る。
 * ⑦ **家族 admission の違反でも重みの part は取得されない**（②の家族版）。② は runtime の
 *    capability 門だけを踏むので、「実行できないモデルの重みは 1 バイトも落とさない」の
 *    うち家族側（pipeline major / `pipelineConfig`）が前段に居ることは縛れない。実家族
 *    （siglip2 = 部品 1 本の最小形）を `fromPretrained` で通して同じ観測法で見る。
 * ⑧ **`requiredLimits` 超過でも重みの part は取得されない（共有 GPU）**（⑦の limits 版 —
 *    ADR 0089 決定 5）。突き合わせ相手は渡された `GpuContext.limits`。
 * ⑩ **選択と供給口の失敗診断**（manifest が持たない部品名 / 開いていない役割）。
 *
 * NOTE: hub / runtime の**テストの都合**は import しない（`helpers/memory-cache.ts` と同じ
 * 規律）。モックは `helpers/container-loading-fixture.ts` に最小限だけ組んである。例外は 2 つ
 * — 容器の**書き手**（形式の道具。fixture の NOTE）と、⑧の `fake-gpu.ts`（`GpuContext` は
 * runtime が値として公開しない〈`acquireGpu` が唯一の入口 — ADR 0008〉ので、共有 GPU 経路を
 * GPU 無し環境で踏むには向こうの実物を包む helper が要る。ここで自前に偽物を組むと、検査対象
 * そのもの（limits を持つ GpuContext）が偽物になる）。
 *
 * ## Component replacement seat (`LoadContainerOptions.components` — ADR 0108 decision 19 / ADR 0109 decision 10)
 *
 * Also GPU-free: two mock HF repos, observing which (repo, path) pairs are requested.
 *
 * - Only the replaced role comes from the other repo; the same role's weights of the original repo are never
 *   fetched, and roles that are not replaced stay on the original repo. Inherited (non-own) properties of
 *   `components` never take a seat.
 * - A container whose graph description differs is rejected by admission before any `.krm` of either repo is
 *   fetched.
 * - A role name this family does not have fails loudly, listing the replaceable roles.
 * - A replacement part that has the same path as another component's part of the original repo is counted as a
 *   separate file in progress. The replacement manifest references it as its own repo (no `repo` field), so
 *   this pins the per-seat manifest in the progress key, which the cross-repo case above does not reach.
 *
 * ## Real-GPU gates
 *
 * Explicitly skipped without an adapter (ADR 0005).
 *
 * - Building a Session does not move progress (re-reading the prefetched cache must not emit `complete` again,
 *   which would double-count the aggregated `loaded`) and does not go to the network.
 * - One opener can build two Sessions in a row (blocks are re-read in part order on every call); checked by the
 *   run output, not by the build succeeding.
 * - The load-time `signal` is not carried into Session building (an abort after a successful load must not
 *   break later Sessions). The paired "a load started with an aborted signal fails" gate is ⑤ above.
 * - `requiredLimits` beyond the adapter also blocks weight parts on the self-acquired GPU path (compared against
 *   `readAdapterLimits()` — ADR 0089 decision 5); the shared-GPU variant is ⑧ above.
 */

import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, it, test } from "vite-plus/test";
import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import type { AssetProgress, DistributionSource } from "@karume/hub";
import { acquireGpu } from "@karume/runtime";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { loadContainerComponents } from "../src/hub/components.ts";
import { Siglip2Pipeline } from "../src/siglip2/pipeline.ts";
import { LINEAR_PROBE_Y, linearComponent, runLinearProbe } from "./helpers/container-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  HUB_URL,
  MANIFEST_PATH,
  manifestBytes,
  NO_FAMILY_GATE,
  prepareComponent,
  prepareSiglip2,
  quantsRequiring,
  REPO,
  serveContainer,
  type ServedContainer,
  serveRepo,
  serveRepos,
  SHA,
  SIGLIP2_CONFIG,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";
import { loadManifest, localDirectory, resolveSelection } from "@karume/hub";

/** 長さ 0 でない part（= 取得層が実際に取りに行く列）。 */
const fetched = (parts: readonly { path: string; size: number }[]): readonly string[] =>
  parts.filter((part) => part.size > 0).map((part) => part.path);

test("loadContainerComponents: 複数部品でも進捗はモデル全体で 1 本（単調増加・合計へ着地）", async () => {
  const front = await serveContainer("front/model.f32", linearComponent("front"));
  const voice = await serveContainer("voice/model.f32", linearComponent("voice"));
  const tokenizer = new TextEncoder().encode(JSON.stringify({ vocab: ["a", "b"] }));
  const tokenizerRef = {
    path: "tokenizer/tokenizer.json",
    size: tokenizer.byteLength,
    sha256: [...new Uint8Array(await crypto.subtle.digest("SHA-256", tokenizer))]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join(""),
  };
  const rig = await serveRepo(
    {
      test: {
        pipeline: "test/1",
        weights: { front: { f32: front.entry }, voice: { f32: voice.entry } },
        assets: { tokenizer: tokenizerRef },
        quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
    new Map([
      ...front.files,
      ...voice.files,
      [tokenizerRef.path, tokenizer as Uint8Array<ArrayBuffer>],
    ]),
    { front: front.parts, voice: voice.parts },
  );

  const events: AssetProgress[] = [];
  const { open, assets } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
  );

  // admission を通ったこと（グラフ宣言が読める = 合流まで済んでいる）。
  assertEquals(open("front").graph.outputs, ["y"]);
  assertEquals(open("voice").graph.outputs, ["y"]);
  // 残り資産は全量面のまま届く。
  assertEquals(Object.keys(assets), ["tokenizer"]);

  const paths = [...fetched(front.parts), ...fetched(voice.parts), tokenizerRef.path];
  const total =
    [...front.parts, ...voice.parts]
      .filter((part) => part.size > 0)
      .reduce((sum, part) => sum + part.size, 0) + tokenizerRef.size;
  // total は全イベントで**モデル全体の合計**の 1 値（取得の割れ方に依らない）。
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  // loaded は単調増加で、最後は合計に一致する。
  let previous = 0;
  for (const event of events) {
    if (event.loaded < previous) {
      throw new Error(
        `loaded が巻き戻った: ${previous} → ${event.loaded}（${event.phase} ${event.path}）`,
      );
    }
    previous = event.loaded;
  }
  assertEquals(events[events.length - 1].loaded, total);
  // complete はファイル 1 本につき 1 回（per-file 欄も取得層の契約どおり fileTotal で揃う）。
  const completes = events.filter((event) => event.phase === "complete");
  assertEquals(completes.length, paths.length);
  assertEquals(new Set(completes.map((event) => event.path)), new Set(paths));
  for (const event of completes) assertEquals(event.fileLoaded, event.fileTotal);
  // 長さ 0 の part は取得にも進捗にも現れない（ADR 0109 決定 3）。
  const empty = [...front.parts, ...voice.parts].filter((part) => part.size === 0);
  assertEquals(empty.length > 0, true, "長さ 0 の part を持つ容器で観測していない");
  for (const part of empty) assertEquals(rig.mock.paths.includes(part.path), false);
});

test("loadContainerComponents: 越境参照が自リポと同じ path でも、進捗は別の 1 本として数える", async () => {
  // `voice` は越境先のリポから借りる容器（容器単位の越境 — 全 part が越境先の座標を名乗る）。
  // 書き手の規約どおりの part 名は stem だけで決まるので、同じ stem の 2 本は同じ path を持つ。
  const CROSS_REPO = "karume-test/borrowed";
  const own = await serveContainer("shared/model.f32", linearComponent("front"));
  const borrowed = await serveContainer("shared/model.f32", linearComponent("voice"));
  const crossParts = borrowed.parts.map((part) => ({
    ...part,
    repo: CROSS_REPO,
    revision: SHA,
  }));
  const models = {
    test: {
      pipeline: "test/1",
      weights: {
        front: { f32: own.entry },
        voice: {
          f32: { container: { descriptor: borrowed.written.descriptor, parts: crossParts } },
        },
      },
      assets: {},
      quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
      defaultQuant: "f32",
      pipelineConfig: {},
    },
  };
  const mock = serveRepos([
    { repo: REPO, models, files: own.files },
    { repo: CROSS_REPO, models: {}, files: borrowed.files },
  ]);
  const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  const shared = fetched(own.parts).filter((path) => fetched(borrowed.parts).includes(path));
  assertEquals(shared.length > 0, true, "同じ path を持つ 2 本で観測していない");

  const events: AssetProgress[] = [];
  await loadContainerComponents(
    "test.fromPretrained",
    loaded,
    resolveSelection(loaded.manifest),
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...hubOptions, onProgress: (progress) => events.push(progress) },
  );

  // path で畳むと `total` が 1 本ぶん小さくなり、`loaded` も同じだけ手前で止まる。
  const total = [...own.parts, ...borrowed.parts].reduce((sum, part) => sum + part.size, 0);
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  assertEquals(events[events.length - 1].loaded, total);
  // 取得先の欄は素通しされ、同じ path の 2 本を消費側でも見分けられる。
  for (const path of shared) {
    const origins = events
      .filter((event) => event.phase === "complete" && event.path === path)
      .map((event) => event.repo)
      .sort((a, b) => (a! < b! ? -1 : a! > b! ? 1 : 0));
    assertEquals(origins, [CROSS_REPO, REPO].sort(), `${path} の 2 本が見分けられない`);
  }
});

test("loadContainerComponents: 選択内の 2 部品が part を共有しても、その part は 1 回だけ温める", async () => {
  // manifest の parse は選択内で別の容器が同じ part ファイルを共有することを許す。取得計画を
  // 容器ごとの part 列の連結のまま渡すと、温め面の重複検査（同じ参照を 2 回渡すのは呼び手の
  // 誤り）で、宣言は通るのにロードだけが落ちる。
  // 重みが同じ値の 2 部品はグラフ名だけが違うので、part 1 以降（重みの block）は同じバイト列に
  // なる — `voice` の part 1 以降を `front` のファイルへ向けて共有させる。
  const front = await serveContainer("front/model.f32", linearComponent("front"));
  const voice = await serveContainer("voice/model.f32", linearComponent("voice"));
  const sharedParts = front.parts.slice(1);
  assertEquals(
    voice.parts.slice(1).map((part) => part.sha256),
    sharedParts.map((part) => part.sha256),
    "part 1 以降が同じバイト列の 2 部品で観測していない",
  );
  const voiceParts = [voice.parts[0], ...sharedParts];
  const rig = await serveRepo(
    {
      test: {
        pipeline: "test/1",
        weights: {
          front: { f32: front.entry },
          voice: {
            f32: { container: { descriptor: voice.written.descriptor, parts: voiceParts } },
          },
        },
        assets: {},
        quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
    new Map([...front.files, ...voice.files.filter(([path]) => path === voice.parts[0].path)]),
    { front: front.parts, voice: voiceParts },
  );

  const events: AssetProgress[] = [];
  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
  );

  assertEquals(open("voice").graph.outputs, ["y"]);
  const shared = fetched(sharedParts);
  assertEquals(shared.length > 0, true, "取得される共有 part で観測していない");
  for (const path of shared) {
    assertEquals(
      rig.mock.paths.filter((requested) => requested === path).length,
      1,
      `${path} を 2 回以上取りに行っている`,
    );
    assertEquals(
      events.filter((event) => event.phase === "complete" && event.path === path).length,
      1,
      `${path} の complete が 1 回でない`,
    );
  }
  // 進捗の分母も共有 part を 1 本と数え、最後は合計に着地する。
  const total = [front.parts[0], voice.parts[0], ...sharedParts].reduce(
    (sum, part) => sum + part.size,
    0,
  );
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  assertEquals(events[events.length - 1].loaded, total);
});

/** メモリ上の配布形ディレクトリ（`karume.json` + 配るファイル）を取得元にする。 */
const memorySource = (
  models: unknown,
  files: Iterable<readonly [string, Uint8Array<ArrayBuffer>]>,
): DistributionSource => {
  const table = new Map<string, Uint8Array<ArrayBuffer>>([
    [MANIFEST_PATH, manifestBytes(models)],
    ...files,
  ]);
  return localDirectory(
    {
      readFile: (path) => {
        const bytes = table.get(path);
        if (bytes === undefined) return Promise.reject(new Error(`test: ${path} が無い`));
        // MUST: buffer 全体を占める view を返す（`DirectoryAdapter.readFile` の契約）。
        return Promise.resolve(bytes.slice());
      },
    },
    { label: "test-directory" },
  );
};

/** 進捗の列が単調増加で合計へ着地し、ファイル 1 本につき complete が 1 回であることを見る。 */
const assertLandsOnTotal = (
  events: readonly AssetProgress[],
  files: readonly { readonly path: string; readonly size: number }[],
): void => {
  const total = files.reduce((sum, file) => sum + file.size, 0);
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  let previous = 0;
  for (const event of events) {
    if (event.loaded < previous) {
      throw new Error(
        `loaded が巻き戻った: ${previous} → ${event.loaded}（${event.phase} ${event.path}）`,
      );
    }
    previous = event.loaded;
  }
  assertEquals(events.at(-1)?.loaded, total, "最後の通知が合計に届いていない");
  const completes = events.filter((event) => event.phase === "complete");
  assertEquals(
    completes.map((event) => event.path).sort(),
    files.map((file) => file.path).sort(),
    "complete がファイル 1 本につき 1 回でない",
  );
};

describe("loadContainerComponents: ローカル取得元だけのロードでも、進捗は合計へ着地する", () => {
  // 相 1（温め）を持たない取得元の part も、温めの段で complete を 1 回出す（ADR 0086 追記
  // 2026-10-11）— 出さないと分母に入った part の分だけ最後の値が届かず、資産が無ければ通知が
  // 1 度も来ない。
  let front: ServedContainer;
  let voice: ServedContainer;
  let tokenizer: Uint8Array<ArrayBuffer>;
  let tokenizerRef: { path: string; size: number; sha256: string };
  let parts: ServedContainer["parts"];
  beforeAll(async () => {
    front = await serveContainer("front/model.f32", linearComponent("front"));
    voice = await serveContainer("voice/model.f32", linearComponent("voice"));
    tokenizer = new TextEncoder().encode(JSON.stringify({ vocab: ["a"] }));
    tokenizerRef = {
      path: "tokenizer/tokenizer.json",
      size: tokenizer.byteLength,
      sha256: [...new Uint8Array(await crypto.subtle.digest("SHA-256", tokenizer))]
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join(""),
    };
    parts = [...front.parts, ...voice.parts].filter((part) => part.size > 0);
  });
  for (const withAssets of [false, true]) {
    it(withAssets ? "資産あり" : "資産なし", async () => {
      const source = memorySource(
        {
          test: {
            pipeline: "test/1",
            weights: { front: { f32: front.entry }, voice: { f32: voice.entry } },
            assets: withAssets ? { tokenizer: tokenizerRef } : {},
            quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
            defaultQuant: "f32",
            pipelineConfig: {},
          },
        },
        [...front.files, ...voice.files, [tokenizerRef.path, tokenizer]],
      );
      const loaded = await loadManifest(source);
      const events: AssetProgress[] = [];
      await loadContainerComponents(
        "test.fromPretrained",
        loaded,
        resolveSelection(loaded.manifest),
        ["front", "voice"],
        NO_FAMILY_GATE,
        { onProgress: (progress) => events.push(progress) },
      );
      assertLandsOnTotal(events, withAssets ? [...parts, tokenizerRef] : parts);
    });
  }
});

test("loadContainerComponents: HF とローカルが混ざっても、進捗は 1 つの規則で合計へ着地する", async () => {
  // 主の配布形はローカル、`voice` だけを疑似 HF のリポから差し替える（同じグラフ記述の容器）。
  const front = await serveContainer("front/model.f32", linearComponent("front"));
  const voice = await serveContainer("voice/model.f32", linearComponent("voice"));
  const models = {
    test: {
      pipeline: "test/1",
      weights: { front: { f32: front.entry }, voice: { f32: voice.entry } },
      assets: {},
      quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
      defaultQuant: "f32",
      pipelineConfig: {},
    },
  };
  const loaded = await loadManifest(memorySource(models, [...front.files, ...voice.files]));
  const mock = serveRepos([{ repo: REPO, models, files: voice.files }]);
  const events: AssetProgress[] = [];
  await loadContainerComponents(
    "test.fromPretrained",
    loaded,
    resolveSelection(loaded.manifest),
    ["front", "voice"],
    NO_FAMILY_GATE,
    {
      fetch: mock.fetch,
      caches: new MemoryCacheStorage(),
      components: { voice: { source: { repo: REPO, revision: SHA, hubUrl: HUB_URL } } },
      onProgress: (progress) => events.push(progress),
    },
  );
  assertEquals(
    fetched(voice.parts).filter((path) => !mock.paths.includes(path)),
    [],
    "差し替えた部品を HF から取っていない",
  );
  // HF の part は取得先（repo）を名乗り、ローカルの part は名乗らない — 両方が 1 本の列に並ぶ。
  const origins = new Set(
    events.filter((event) => event.phase === "complete").map((event) => event.repo),
  );
  assertEquals(origins, new Set([REPO, undefined]), "HF とローカルの混在で観測していない");
  assertLandsOnTotal(
    events,
    [...front.parts, ...voice.parts].filter((part) => part.size > 0),
  );
});

test("loadContainerComponents: capability 違反は descriptor だけで落ち、重みの part は取得されない", async () => {
  const rig = await prepareComponent({ op: "karume_test_unsupported_op" });
  const parts = rig.parts["dit"];

  const error = await assertRejects(
    () =>
      loadContainerComponents(
        "test.fromPretrained",
        rig.loaded,
        rig.selection,
        ["dit"],
        NO_FAMILY_GATE,
        rig.hubOptions,
      ),
    Error,
  );
  // 落ちた理由が capability 門であること（別の失敗で「重みを取らなかった」が成立しない）。
  if (!error.message.includes("karume_test_unsupported_op")) {
    throw new Error(`capability 門の文言でない: ${error.message}`);
  }
  // part 0（descriptor）は取りに行き、重みの part は 1 度も叩いていない（= 重み DL 前
  // admission。ロード時 prefetch が admission の後に置かれていることの門でもある）。
  assertEquals(rig.mock.paths.includes(parts[0].path), true);
  for (const part of parts.slice(1)) {
    assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} を取っている`);
  }
});

test("loadContainerComponents: 重みの part は Session を張る前（ロード時）に落ち切り、進捗にも現れる", async () => {
  const rig = await prepareComponent();
  const parts = rig.parts["dit"];

  const events: AssetProgress[] = [];
  await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["dit"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
  );

  // Session は 1 本も張っていない。それでも重みの part の URL は叩かれている。
  for (const path of fetched(parts)) {
    assertEquals(rig.mock.paths.includes(path), true, `${path} がロード時に落ちない`);
  }
  // 進捗にも取得した part ぶんが乗る（遅延構築の家族でも「ロード = 全 DL」の表示が成立する）。
  const completes = events.filter((event) => event.phase === "complete");
  assertEquals(new Set(completes.map((event) => event.path)), new Set(fetched(parts)));
  assertEquals(
    events[events.length - 1].loaded,
    parts.reduce((sum, part) => sum + part.size, 0),
  );
});

test("loadContainerComponents: abort 済みの signal で始めたロードは落ちる（中断契約の維持）", async () => {
  const rig = await prepareComponent();
  const reason = new Error("test: ロード開始前に中断済み");

  const error = await assertRejects(() =>
    loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["dit"],
      NO_FAMILY_GATE,
      { ...rig.hubOptions, signal: AbortSignal.abort(reason) },
    ),
  );
  assertEquals(error, reason);
});

test("家族 admission（pipeline major 不一致）は descriptor だけで落ち、重みの part は取得されない", async () => {
  const rig = await prepareSiglip2({ pipeline: "siglip2/99" });
  const parts = rig.parts["vision"];

  const error = await assertRejects(
    () =>
      Siglip2Pipeline.fromPretrained(
        { repo: REPO, revision: SHA, hubUrl: HUB_URL },
        { fetch: rig.mock.fetch, caches: rig.hubOptions.caches },
      ),
    Error,
  );
  // 落ちた理由が家族の major 門であること（別の失敗で「重みを取らなかった」が成立しない）。
  if (!error.message.includes("major に未対応")) {
    throw new Error(`家族 admission の文言でない: ${error.message}`);
  }
  // descriptor は取りに行き、重みの part は 1 度も叩いていない（ADR 0070 決定 5 の文面
  // 「実行できないモデルの重みは 1 バイトも落とさない」が家族の門にも及んでいる）。
  assertEquals(rig.mock.paths.includes(parts[0].path), true);
  for (const part of parts.slice(1)) {
    assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} を取っている`);
  }
});

test("家族 admission（pipelineConfig の schema 違反）でも重みの part は取得されない", async () => {
  const rig = await prepareSiglip2({
    pipelineConfig: { ...SIGLIP2_CONFIG, karumeUnknownKey: 1 },
  });
  const parts = rig.parts["vision"];

  const error = await assertRejects(
    () =>
      Siglip2Pipeline.fromPretrained(
        { repo: REPO, revision: SHA, hubUrl: HUB_URL },
        { fetch: rig.mock.fetch, caches: rig.hubOptions.caches },
      ),
    Error,
  );
  if (!error.message.includes("karumeUnknownKey")) {
    throw new Error(`pipelineConfig の門の文言でない: ${error.message}`);
  }
  assertEquals(rig.mock.paths.includes(parts[0].path), true);
  for (const part of parts.slice(1)) {
    assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} を取っている`);
  }
});

test("家族 admission（requiredLimits 超過）でも重みの part は取得されない（共有 GPU）", async () => {
  // 全 limit が 0 の GpuContext（`fake-gpu.ts` の ZERO_LIMITS）へ、1 バイトでも要求する
  // 配布形を渡す。共有 GPU は取り直せない（feature も limits も device 生成時の話）ので、
  // 落とせる唯一の場所が admission 席になる。
  const rig = await prepareSiglip2({ quants: quantsRequiring(1) });
  const parts = rig.parts["vision"];

  const error = await assertRejects(
    () =>
      Siglip2Pipeline.fromPretrained(
        { repo: REPO, revision: SHA, hubUrl: HUB_URL },
        {
          fetch: rig.mock.fetch,
          caches: rig.hubOptions.caches,
          gpu: fakeGpuContext(fakeDevice()),
        },
      ),
    Error,
  );
  // 落ちた理由が limits 検査であること（別の失敗で「重みを取らなかった」が成立しない）。
  if (!error.message.includes("maxBufferSize")) {
    throw new Error(`limits 門の文言でない: ${error.message}`);
  }
  assertEquals(rig.mock.paths.includes(parts[0].path), true);
  for (const part of parts.slice(1)) {
    assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} を取っている`);
  }
});

test("loadContainerComponents: 重みの block のバイト列を握らない（別プロセスで gc 観測）", async () => {
  // MUST: 別プロセス — 到達不能なだけの状態と握られた状態を区別するには gc の強制が要り、
  // `deno test` に `--v8-flags` を渡す口が無い。
  const script = new URL("./helpers/weight-block-retention.ts", import.meta.url);
  const { code, stdout, stderr } = await new Promise<{
    code: number | null;
    stdout: string;
    stderr: string;
  }>((resolve) =>
    execFile(
      process.execPath,
      ["--expose-gc", fileURLToPath(script)],
      { encoding: "utf8" },
      (error, stdout, stderr) =>
        resolve({
          code: error === null ? 0 : ((error.code as number | null) ?? null),
          stdout,
          stderr,
        }),
    ),
  );
  assertEquals(code, 0, `${stdout}${stderr}`);
});

// ---- 選択と供給口の失敗診断（⑩）--------------------------------------------

test("loadContainerComponents: manifest が持たない部品名は選択の部品一覧つきで落ちる", async () => {
  const rig = await prepareComponent();
  const error = await assertRejects(
    () =>
      loadContainerComponents(
        "test.fromPretrained",
        rig.loaded,
        rig.selection,
        ["dit", "vae"],
        NO_FAMILY_GATE,
        rig.hubOptions,
      ),
    Error,
    "部品 'vae' の容器が manifest に無い",
  );
  // 読み手が現物と突き合わせられる形（既存の資産診断の流儀）。
  assertStringIncludes(error.message, "持つ部品: dit");
});

test("loadContainerComponents: 開いていない役割で open() すると開いた一覧つきで落ちる", async () => {
  const rig = await prepareComponent();
  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["dit"],
    NO_FAMILY_GATE,
    rig.hubOptions,
  );
  const error = assertThrows(() => open("vae"), Error, "部品 'vae' は開いていない");
  assertStringIncludes(error.message, "開いた部品: dit");
});

// ---- Component replacement seat (`LoadContainerOptions.components`) -------------------

/** 差し替え先の疑似リポ。 */
const OTHER_REPO = "karume-test/replacement";

const modelsOf = (entries: Readonly<Record<string, ServedContainer>>): unknown => ({
  test: {
    pipeline: "test/1",
    weights: Object.fromEntries(
      Object.entries(entries).map(([key, served]) => [key, { f32: served.entry }]),
    ),
    assets: {},
    quants: {
      f32: {
        weights: Object.fromEntries(Object.keys(entries).map((key) => [key, "f32"])),
        session: {},
      },
    },
    defaultQuant: "f32",
    pipelineConfig: {},
  },
});

/**
 * 2 リポを立てる。元リポは `front` / `voice` の 2 部品、差し替え先は `voice` 1 本だけ
 * （`sameGraph` が偽なら**別のグラフ**で書く = 記述の sha256 が動く）。
 */
const prepareSeats = async (options: {
  readonly sameGraph: boolean;
  readonly replacementStem?: string;
}) => {
  const front = await serveContainer("front/model.f32", linearComponent("front"));
  const voice = await serveContainer("voice/model.f32", linearComponent("voice", { w: 0.5 }));
  // 差し替え先: 同じグラフ宣言 + 別の重み値（グラフ記述はバイト同一で sha256 も一致する）。
  const replacement = await serveContainer(
    options.replacementStem ?? "other/voice.f32",
    options.sameGraph
      ? linearComponent("voice", { w: 1.5 })
      : // `op` が違えば `requires.ops` とノード列が動く = グラフ記述の sha256 も動く。
        linearComponent("voice", { op: "add" }),
  );
  const mock = serveRepos([
    { repo: REPO, models: modelsOf({ front, voice }), files: [...front.files, ...voice.files] },
    { repo: OTHER_REPO, models: modelsOf({ voice: replacement }), files: replacement.files },
  ]);
  const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  return {
    mock,
    hubOptions,
    loaded,
    selection: resolveSelection(loaded.manifest),
    parts: { front: front.parts, voice: voice.parts, replacement: replacement.parts },
    source: { repo: OTHER_REPO, revision: SHA, hubUrl: HUB_URL },
  };
};

/** 取得した `.krm` の path（リポ問わず）。 */
const fetchedContainers = (paths: readonly string[]): readonly string[] =>
  paths.filter((path) => path !== MANIFEST_PATH);

test("components: 差した役割だけが別リポから来る（元リポの同じ役割は取らない）", async () => {
  const rig = await prepareSeats({ sameGraph: true });

  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, components: { voice: { source: rig.source } } },
  );
  // 2 本とも開けている（合流まで済んでいる）。
  assertEquals(open("front").graph.outputs, ["y"]);
  assertEquals(open("voice").graph.outputs, ["y"]);

  const asked = (repo: string): readonly string[] =>
    rig.mock.requests.filter((request) => request.repo === repo).map((request) => request.path);
  // 差し替え先の part は全部（長さ 0 を除く）取られている。
  for (const part of rig.parts.replacement.filter((part) => part.size > 0)) {
    assertEquals(asked(OTHER_REPO).includes(part.path), true, `${part.path} を取っていない`);
  }
  // 元リポの `voice` は 1 バイトも取られていない（差し替えが**取得ごと**置き換わっている）。
  for (const part of rig.parts.voice) {
    assertEquals(asked(REPO).includes(part.path), false, `${part.path} を取っている`);
  }
  // 差していない `front` は元リポのまま。
  for (const part of rig.parts.front.filter((part) => part.size > 0)) {
    assertEquals(asked(REPO).includes(part.path), true, `${part.path} を取っていない`);
  }
});

test("components: グラフ記述が違う容器は admission で落ちる（重みは 1 本も取らない）", async () => {
  const rig = await prepareSeats({ sameGraph: false });

  const error = await assertRejects(
    () =>
      loadContainerComponents(
        "test.fromPretrained",
        rig.loaded,
        rig.selection,
        ["front", "voice"],
        NO_FAMILY_GATE,
        { ...rig.hubOptions, components: { voice: { source: rig.source } } },
      ),
    Error,
    "グラフ記述が manifest の宣言と違う",
  );
  // 読み手が現物と突き合わせられる形（宣言と差し替えの 2 つの sha256 が並ぶ）。
  assertStringIncludes(error.message, rig.selection.containers["voice"].descriptor.graph.sha256);
  // 落ちるのは**宣言だけを読んだ時点** — `.krm` はどちらのリポからも 1 本も取っていない。
  assertEquals(fetchedContainers(rig.mock.paths), []);
});

test("components: この系列が持たない役割名は差し替えられる役割つきで落ちる", async () => {
  const rig = await prepareSeats({ sameGraph: true });

  const error = await assertRejects(
    () =>
      loadContainerComponents(
        "test.fromPretrained",
        rig.loaded,
        rig.selection,
        ["front", "voice"],
        NO_FAMILY_GATE,
        { ...rig.hubOptions, components: { vae: { source: rig.source } } },
      ),
    Error,
    "components の 'vae' はこの系列の部品ではない",
  );
  assertStringIncludes(error.message, "差し替えられる役割: front / voice");
  // 綴り間違いは取得へ 1 度も出ない。
  assertEquals(fetchedContainers(rig.mock.paths), []);
});

test("components: 継承プロパティは差し替えに使わない（own の役割だけが席に座る）", async () => {
  // 役割の検査は own の列挙可能キーで行うので、継承値（`Object.create` やプロトタイプ汚染）を
  // 直接添字で読むと、検査を通っていない値が差し替えに使われる。
  const rig = await prepareSeats({ sameGraph: true });
  const inherited: Record<string, { readonly source: typeof rig.source }> = Object.create({
    voice: { source: rig.source },
  });

  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, components: inherited },
  );
  assertEquals(open("voice").graph.outputs, ["y"]);
  assertEquals(
    rig.mock.requests.filter((request) => request.repo === OTHER_REPO),
    [],
    "継承値の差し替え先を取りに行っている",
  );
  for (const part of rig.parts.voice.filter((part) => part.size > 0)) {
    assertEquals(
      rig.mock.requests.some((request) => request.repo === REPO && request.path === part.path),
      true,
      `元リポの ${part.path} を取っていない`,
    );
  }
});

test("components: 差し替え先の part が元リポの別部品と同じ path でも進捗は 2 本として数える", async () => {
  // 差し替え先の `voice` を元リポの `front` と同じ stem で書く = 同じ path の part が 2 リポに並ぶ。
  const rig = await prepareSeats({ sameGraph: true, replacementStem: "front/model.f32" });
  const nonEmpty = (parts: readonly { path: string; size: number }[]) =>
    parts.filter((part) => part.size > 0);
  const shared = nonEmpty(rig.parts.front)
    .map((part) => part.path)
    .filter((path) => nonEmpty(rig.parts.replacement).some((part) => part.path === path));
  assertEquals(shared.length > 0, true, "同じ path を持つ 2 本で観測していない");

  const events: AssetProgress[] = [];
  await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    {
      ...rig.hubOptions,
      components: { voice: { source: rig.source } },
      onProgress: (progress) => events.push(progress),
    },
  );

  const total = [...nonEmpty(rig.parts.front), ...nonEmpty(rig.parts.replacement)].reduce(
    (sum, part) => sum + part.size,
    0,
  );
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  assertEquals(events[events.length - 1].loaded, total);
  for (const path of shared) {
    const origins = events
      .filter((event) => event.phase === "complete" && event.path === path)
      .map((event) => event.repo)
      .sort((a, b) => (a! < b! ? -1 : a! > b! ? 1 : 0));
    assertEquals(origins, [OTHER_REPO, REPO].sort(), `${path} の 2 本が見分けられない`);
  }
});

// ---- Real-GPU gates -------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "loadContainerComponents: Session 構築は進捗を 1 イベントも動かさない（実 GPU）",
  async () => {
    const rig = await prepareComponent();
    const fetchedCount = fetched(rig.parts["dit"]).length;

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
    assertEquals(events.filter((event) => event.phase === "complete").length, fetchedCount);

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
);

test.skipIf(!GPU_AVAILABLE)(
  "loadContainerComponents: 同じ供給口から Session を 2 本続けて張れる（実 GPU）",
  async () => {
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
      // 2 本目が正しい出力を返すのは、block を**呼ぶたびに**読み直しているとき。Session 構築は
      // 宣言した initializer の全到着を突き合わせないので、1 度読んだら終わりの列にすると
      // 2 本目は空の block 列を受けたまま張れてしまう — 張れたことではなく run の出力で見る。
      for (let index = 0; index < 2; index += 1) {
        const session = await open("dit").createSession(gpu);
        assertEquals(await runLinearProbe(session), LINEAR_PROBE_Y, `Session ${index + 1} 本目`);
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "loadContainerComponents: ロード時の signal は Session 構築へ持ち越さない（実 GPU）",
  async () => {
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
);

// MUST: アダプタ無し環境は明示 SKIP（ADR 0005）。この経路の突き合わせ相手は
// `readAdapterLimits()` なので、アダプタが無いと GpuUnavailableError に化けて門が見えない。
test.skipIf(!GPU_AVAILABLE)(
  "requiredLimits 超過は自前 GPU 取得の経路でも重みの part を取らない（アダプタ実測）",
  async () => {
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
);
