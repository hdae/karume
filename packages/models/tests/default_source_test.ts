// 公開配布リポの既定ソース pin（ADR 0073）の門。**ネットワークには一切出ない** — 公開面の
// `fetch` 注入席へ記録用のスタブを渡し、取得層が組み立てた URL だけを見て落とす。
//
// ここで押さえるのは 2 つ:
//  ① 定数の形 — repo 名と、revision が 40 桁 hex の commit SHA であること。SHA でなければ hub は
//     revision 解決 API を叩きに出る（= オフライン起動が壊れ、かつ付け替え可能な ref に戻る）。
//  ② `fromPretrained` の ref を省略したとき、その定数の repo / revision がそのまま取得 URL に
//     載ること。既定が別のリポ参照へ滑っても型も shape も変わらないので、ここで見ないと沈黙する。
//
// 綴りの正しさを見るのは①だけ（②は期待 URL を定数から導くので、定数が間違っていても通る）。
// 2 本セットで「正しい repo が / 実際に使われる」を挟む。
//
// pin の無いファミリ（birefnet / depth-anything / siglip2 / vowel-detector）は公開リポが無く
// ref 必須のままなので、この門の対象外（ADR 0073 決定 1）。

import { assertEquals, assertMatch, assertRejects } from "@std/assert";
import { ANIMA_BASE_SOURCE, ANIMA_DEFAULT_SOURCE } from "../src/anima/config.ts";
import { IRODORI_DEFAULT_SOURCE } from "../src/irodori/config.ts";
import { SBV2_DEFAULT_SOURCE } from "../src/sbv2/config.ts";
import { AnimaPipeline } from "../src/anima/pipeline.ts";
import { IrodoriPipeline } from "../src/irodori/pipeline.ts";
import { Sbv2Pipeline } from "../src/sbv2/pipeline.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** hub が解決要求を出さずに済む形（`@hdae/fetch-cache` の `isCommitSha` と同じ綴り）。 */
const COMMIT_SHA = /^[0-9a-f]{40}$/;

const SOURCES = [
  { family: "sbv2", source: SBV2_DEFAULT_SOURCE, repo: "hdae/karume-sbv2-jvnv" },
  { family: "irodori", source: IRODORI_DEFAULT_SOURCE, repo: "hdae/karume-irodori-v4-small" },
  { family: "anima", source: ANIMA_DEFAULT_SOURCE, repo: "hdae/karume-anima-turbo" },
] as const;

for (const { family, source, repo } of SOURCES) {
  Deno.test(`${family}: 既定ソースは pin 済みリポ + 40 桁 hex の commit SHA`, () => {
    assertEquals(source.repo, repo);
    assertMatch(source.revision, COMMIT_SHA);
  });
}

/** 取得 URL を記録して**その場で落とす** fetch（network へは 1 バイトも出ない）。 */
const recordingFetch = (urls: string[]): typeof globalThis.fetch => (input) => {
  urls.push(input instanceof Request ? input.url : String(input));
  return Promise.reject(new Error("default_source_test: ここから先へは出ない"));
};

/**
 * `ref` 省略の `fromPretrained` が最初に取りに行く URL の pathname 列を返す。
 *
 * host は伏せる（`hubUrl` の既定はミラー設定で動く — 見るべきは repo と revision）。
 */
const requestedPaths = async (
  load: (options: { fetch: typeof globalThis.fetch; caches: CacheStorage }) => Promise<unknown>,
): Promise<string[]> => {
  const urls: string[] = [];
  // MUST: `caches` も注入する — 実 Cache Storage に触れると、他テストの残骸でヒット経路に
  // 化けて「network を叩かない」別の話を検査してしまう。
  await assertRejects(() =>
    load({ fetch: recordingFetch(urls), caches: new MemoryCacheStorage() })
  );
  return urls.map((url) => new URL(url).pathname);
};

const manifestPath = (source: { repo: string; revision: string }): string =>
  `/${source.repo}/resolve/${source.revision}/karume.json`;

Deno.test("sbv2: ref 省略の fromPretrained は SBV2_DEFAULT_SOURCE を取りに行く", async () => {
  const paths = await requestedPaths((options) => Sbv2Pipeline.fromPretrained(undefined, options));
  assertEquals(paths, [manifestPath(SBV2_DEFAULT_SOURCE)]);
});

Deno.test("irodori: ref 省略の fromPretrained は IRODORI_DEFAULT_SOURCE を取りに行く", async () => {
  const paths = await requestedPaths((options) =>
    IrodoriPipeline.fromPretrained(undefined, options)
  );
  assertEquals(paths, [manifestPath(IRODORI_DEFAULT_SOURCE)]);
});

Deno.test("anima: ref 省略の fromPretrained は ANIMA_DEFAULT_SOURCE を取りに行く", async () => {
  const paths = await requestedPaths((options) => AnimaPipeline.fromPretrained(undefined, options));
  assertEquals(paths, [manifestPath(ANIMA_DEFAULT_SOURCE)]);
});

// 素版 3 モデルの同居リポ（既定ではないが pin の MUST は同じ — 公開面へ出す以上、付け替え
// 可能な ref に戻ったら公開済みパッケージの読むバイト列が黙って変わる）。
Deno.test("anima: 素版リポの pin は pin 済みリポ + 40 桁 hex の commit SHA", () => {
  assertEquals(ANIMA_BASE_SOURCE.repo, "hdae/karume-anima");
  assertMatch(ANIMA_BASE_SOURCE.revision, COMMIT_SHA);
});

Deno.test("anima: ANIMA_BASE_SOURCE を渡すと素版リポの pin がそのまま取得 URL に載る", async () => {
  // SHA 固定なので revision 解決 API を 1 往復もしない（= オフライン起動可）。ここが
  // `/api/models/.../revision/...` から始まったら pin が ref へ戻っている。
  const paths = await requestedPaths((options) =>
    AnimaPipeline.fromPretrained(ANIMA_BASE_SOURCE, options)
  );
  assertEquals(paths, [manifestPath(ANIMA_BASE_SOURCE)]);
});

// 文字列 ref の意味（`{ repo }` = main 追従）は pin 導入後も不変（ADR 0073 決定 2）。main は
// commit SHA ではないので、manifest を取る**前に** revision 解決 API を 1 往復する。
Deno.test("sbv2: 文字列 ref は main 追従のまま（revision 解決 API を経由する）", async () => {
  const paths = await requestedPaths((options) =>
    Sbv2Pipeline.fromPretrained("someone/karume-sbv2-fork", options)
  );
  assertEquals(paths, ["/api/models/someone/karume-sbv2-fork/revision/main"]);
});
