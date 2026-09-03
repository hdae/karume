// 公開配布リポの pin 定数（`*_CURRENT` — ADR 0073）の門。**ネットワークには一切出ない** —
// 公開面の `fetch` 注入席へ記録用のスタブを渡し、取得層が組み立てた URL だけを見て落とす。
//
// ここで押さえるのは 2 つ:
//  ① 定数の形 — repo 名と、revision が 40 桁 hex の commit SHA であること。SHA でなければ hub は
//     revision 解決 API を叩きに出る（= オフライン起動が壊れ、かつ付け替え可能な ref に戻る）。
//  ② その定数を `fromPretrained` へ渡すと、repo / revision がそのまま取得 URL に載ること。
//     定数が別のリポ参照へ滑っても型も shape も変わらないので、ここで見ないと沈黙する。
//
// 綴りの正しさを見るのは①だけ（②は期待 URL を定数から導くので、定数が間違っていても通る）。
// 2 本セットで「正しい repo が / 実際に使われる」を挟む。
//
// 定数は**公開 HF リポ 1 つにつき 1 本**（0.5.0 の改名 — `*_DEFAULT_SOURCE` /`*_BASE_SOURCE` は
// 「既定席かどうか」を名前に持っていたが、`fromPretrained` の既定が無くなって区別が消えた）。
// 公開リポを持たないファミリ（birefnet / depth-anything / siglip2 / vowel-detector）は定数自体を
// 持たないので、この門の対象外（ADR 0073 決定 1）。

import { assertEquals, assertMatch, assertRejects } from "@std/assert";
import { ANIMA_CURRENT, ANIMA_EXTRA_CURRENT } from "../src/anima/config.ts";
import { IRODORI_V4_1_SMALL_CURRENT, IRODORI_V4_SMALL_CURRENT } from "../src/irodori/config.ts";
import { SBV2_JVNV_CURRENT } from "../src/sbv2/config.ts";
import { GEMMA4_CURRENT } from "../src/gemma/config.ts";
import { AnimaPipeline } from "../src/anima/pipeline.ts";
import { IrodoriPipeline } from "../src/irodori/pipeline.ts";
import { Sbv2Pipeline } from "../src/sbv2/pipeline.ts";
import { Gemma4Pipeline } from "../src/gemma/pipeline.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** hub が解決要求を出さずに済む形（`@hdae/fetch-cache` の `isCommitSha` と同じ綴り）。 */
const COMMIT_SHA = /^[0-9a-f]{40}$/;

/** 取得 URL を記録して**その場で落とす** fetch（network へは 1 バイトも出ない）。 */
const recordingFetch = (urls: string[]): typeof globalThis.fetch => (input) => {
  urls.push(input instanceof Request ? input.url : String(input));
  return Promise.reject(new Error("current_source_test: ここから先へは出ない"));
};

/**
 * `fromPretrained` が最初に取りに行く URL の pathname 列を返す。
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

const SOURCES = [
  { name: "SBV2_JVNV_CURRENT", source: SBV2_JVNV_CURRENT, repo: "hdae/karume-sbv2-jvnv" },
  {
    name: "IRODORI_V4_SMALL_CURRENT",
    source: IRODORI_V4_SMALL_CURRENT,
    repo: "hdae/karume-irodori-v4-small",
  },
  {
    name: "IRODORI_V4_1_SMALL_CURRENT",
    source: IRODORI_V4_1_SMALL_CURRENT,
    repo: "hdae/karume-irodori-v4.1-small",
  },
  { name: "ANIMA_CURRENT", source: ANIMA_CURRENT, repo: "hdae/karume-anima" },
  { name: "ANIMA_EXTRA_CURRENT", source: ANIMA_EXTRA_CURRENT, repo: "hdae/karume-anima-extra" },
  { name: "GEMMA4_CURRENT", source: GEMMA4_CURRENT, repo: "hdae/karume-gemma4-e2b" },
] as const;

for (const { name, source, repo } of SOURCES) {
  Deno.test(`${name}: pin 済みリポ + 40 桁 hex の commit SHA`, () => {
    assertEquals(source.repo, repo);
    assertMatch(source.revision, COMMIT_SHA);
  });
}

// SHA 固定なので revision 解決 API を 1 往復もしない（= オフライン起動可）。取得 URL が
// `/api/models/.../revision/...` から始まったら pin が可変 ref へ戻っている。
Deno.test("sbv2: SBV2_JVNV_CURRENT の pin がそのまま取得 URL に載る", async () => {
  const paths = await requestedPaths((options) =>
    Sbv2Pipeline.fromPretrained(SBV2_JVNV_CURRENT, options)
  );
  assertEquals(paths, [manifestPath(SBV2_JVNV_CURRENT)]);
});

Deno.test("irodori: IRODORI_V4_SMALL_CURRENT の pin がそのまま取得 URL に載る", async () => {
  const paths = await requestedPaths((options) =>
    IrodoriPipeline.fromPretrained(IRODORI_V4_SMALL_CURRENT, options)
  );
  assertEquals(paths, [manifestPath(IRODORI_V4_SMALL_CURRENT)]);
});

// 上流 v4.1（duration predictor だけ再学習）は別リポ = 別定数。旧 pin は温存する。
Deno.test("irodori: IRODORI_V4_1_SMALL_CURRENT の pin がそのまま取得 URL に載る", async () => {
  const paths = await requestedPaths((options) =>
    IrodoriPipeline.fromPretrained(IRODORI_V4_1_SMALL_CURRENT, options)
  );
  assertEquals(paths, [manifestPath(IRODORI_V4_1_SMALL_CURRENT)]);
});

// 公式モデル同居リポ（既定 = Turbo・既定以外は `{ model }` で選ぶ席）。旧
// ANIMA_TURBO_CURRENT は廃止（2026-09-01・breaking — anima/config.ts の NOTE）。
Deno.test("anima: ANIMA_CURRENT の pin がそのまま取得 URL に載る", async () => {
  const paths = await requestedPaths((options) =>
    AnimaPipeline.fromPretrained(ANIMA_CURRENT, options)
  );
  assertEquals(paths, [manifestPath(ANIMA_CURRENT)]);
});

// 第三者 fine-tune 同居リポ（既定 = wai — ADR 0087）。text stack を公式リポから越境参照する側だが、
// **manifest 自体はこのリポから取る**（越境は manifest の中のファイル参照の話）。
Deno.test("anima: ANIMA_EXTRA_CURRENT の pin がそのまま取得 URL に載る", async () => {
  const paths = await requestedPaths((options) =>
    AnimaPipeline.fromPretrained(ANIMA_EXTRA_CURRENT, options)
  );
  assertEquals(paths, [manifestPath(ANIMA_EXTRA_CURRENT)]);
});

Deno.test("gemma4: GEMMA4_CURRENT の pin がそのまま取得 URL に載る", async () => {
  const paths = await requestedPaths((options) =>
    Gemma4Pipeline.fromPretrained(GEMMA4_CURRENT, options)
  );
  assertEquals(paths, [manifestPath(GEMMA4_CURRENT)]);
});

// 文字列 ref の意味（`{ repo }` = main 追従）は pin 導入後も ref 必須化後も不変。main は
// commit SHA ではないので、manifest を取る**前に** revision 解決 API を 1 往復する。
Deno.test("sbv2: 文字列 ref は main 追従のまま（revision 解決 API を経由する）", async () => {
  const paths = await requestedPaths((options) =>
    Sbv2Pipeline.fromPretrained("someone/karume-sbv2-fork", options)
  );
  assertEquals(paths, ["/api/models/someone/karume-sbv2-fork/revision/main"]);
});
