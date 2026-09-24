/**
 * **部品差し替え席**（`LoadContainerOptions.components` — ADR 0108 決定 19 / ADR 0109 決定 10）の
 * 門。GPU も実資産も要らない（疑似 HF リポを 2 つ立てて、叩かれた (リポ, path) を観測する）。
 *
 * 押さえるのは 3 点:
 *
 * ① **差し替えた役割だけが別リポから来る** — 差した部品の part は差し替え先のリポから取られ、
 *    元リポの同じ役割の重みは 1 バイトも取られない。差していない役割は元リポのまま。
 * ② **グラフ記述が違う容器は admission で落ちる** — 落ちるのは**重みを 1 バイトも取る前**で、
 *    どちらのリポの `.krm` も 1 本も叩かれていない（宣言だけで判る、が実装でも成立している）。
 * ③ **この系列が持たない役割名は fail loudly** — 綴り間違いを黙って「差し替えない」に畳まない。
 * ④ **差し替え先の part が元リポの別部品と同じ path でも、進捗は別の 1 本として数える** — 差し替え
 *    先の part は向こうの manifest では自リポ参照（`repo` を持たない）なので、宣言の path だけでは
 *    元リポのファイルと見分けが付かない。
 */

import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import type { AssetProgress } from "@karume/hub";
import { loadContainerComponents } from "../src/hub/components.ts";
import { linearComponent } from "./helpers/container-fixture.ts";
import {
  HUB_URL,
  MANIFEST_PATH,
  NO_FAMILY_GATE,
  REPO,
  serveContainer,
  type ServedContainer,
  serveRepos,
  SHA,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";
import { loadManifest, resolveSelection } from "@karume/hub";

/** 差し替え先の疑似リポ。 */
const OTHER_REPO = "karume-test/replacement";

const modelsOf = (
  entries: Readonly<Record<string, ServedContainer>>,
): unknown => ({
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
const prepareSeats = async (
  options: { readonly sameGraph: boolean; readonly replacementStem?: string },
) => {
  const front = await serveContainer("front/model.f32", linearComponent("front"));
  const voice = await serveContainer("voice/model.f32", linearComponent("voice", { w: 0.5 }));
  // 差し替え先: 同じグラフ宣言 + 別の重み値（グラフ記述はバイト同一で sha256 も一致する）。
  const replacement = await serveContainer(
    options.replacementStem ?? "other/voice.f32",
    options.sameGraph
      ? linearComponent("voice", { w: 1.5 })
      // `op` が違えば `requires.ops` とノード列が動く = グラフ記述の sha256 も動く。
      : linearComponent("voice", { op: "add" }),
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

Deno.test("components: 差した役割だけが別リポから来る（元リポの同じ役割は取らない）", async () => {
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

Deno.test("components: グラフ記述が違う容器は admission で落ちる（重みは 1 本も取らない）", async () => {
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

Deno.test("components: この系列が持たない役割名は差し替えられる役割つきで落ちる", async () => {
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

Deno.test("components: 差し替え先の part が元リポの別部品と同じ path でも進捗は 2 本として数える", async () => {
  // 差し替え先の `voice` を元リポの `front` と同じ stem で書く = 同じ path の part が 2 リポに並ぶ。
  const rig = await prepareSeats({ sameGraph: true, replacementStem: "front/model.f32" });
  const nonEmpty = (parts: readonly { path: string; size: number }[]) =>
    parts.filter((part) => part.size > 0);
  const shared = nonEmpty(rig.parts.front).map((part) => part.path)
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

  const total = [...nonEmpty(rig.parts.front), ...nonEmpty(rig.parts.replacement)]
    .reduce((sum, part) => sum + part.size, 0);
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  assertEquals(events[events.length - 1].loaded, total);
  for (const path of shared) {
    const origins = events
      .filter((event) => event.phase === "complete" && event.path === path)
      .map((event) => event.repo)
      .sort();
    assertEquals(origins, [OTHER_REPO, REPO].sort(), `${path} の 2 本が見分けられない`);
  }
});
