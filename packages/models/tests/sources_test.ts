// 公開配布リポの対応表（`<FAMILY>_SOURCES` — ADR 0092・値の pin は ADR 0073）の門。
// **ネットワークには一切出ない** — 公開面の `fetch` 注入席へ記録用のスタブを渡し、取得層が
// 組み立てた URL だけを見て落とす。
//
// ここで押さえるのは 3 つ:
//  ① エントリの形 — owner が `hdae`・**キーが repo 名から導ける**こと
//     （`"karume-" + key` === repo の basename）・revision が 40 桁 hex の commit SHA。
//     SHA でなければ hub は revision 解決 API を叩きに出る（= オフライン起動が壊れ、かつ
//     付け替え可能な ref に戻る）。キーの規則を門にするのは、表が「キー = リポ」の対応表
//     だからで、綴りが自由になると `KARUME_SOURCES` のキーがリポを指さなくなる。
//  ② `KARUME_SOURCES` が家族表の**和集合**であること。畳み込みはスプレッドなので、キーが
//     重なると後勝ちで 1 本が黙って消える（型でも実行時でも何も起きない）。
//  ③ そのエントリを `fromPretrained` へ渡すと、repo / revision がそのまま取得 URL に載ること。
//     エントリが別のリポ参照へ滑っても型も shape も変わらないので、ここで見ないと沈黙する。
//
// 綴りの正しさを見るのは①だけ（③は期待 URL をエントリから導くので、値が間違っていても通る）。
// 2 本セットで「正しい repo が / 実際に使われる」を挟む。
//
// 公開リポを持たないファミリ（vowel-detector）は表自体を持たないので、この門の対象外
// （ADR 0073 決定 1）。

import { assert, assertEquals, assertMatch, assertRejects } from "@std/assert";
import type { HubRepoRef } from "@karume/hub";
import { ANIMA_SOURCES } from "../src/anima/config.ts";
import { IRODORI_SOURCES } from "../src/irodori/config.ts";
import { SBV2_SOURCES } from "../src/sbv2/config.ts";
import { GEMMA4_SOURCES } from "../src/gemma/config.ts";
import { SIGLIP2_SOURCES } from "../src/siglip2/config.ts";
import { DEPTH_ANYTHING_SOURCES } from "../src/depth-anything/config.ts";
import { BIREFNET_SOURCES } from "../src/birefnet/config.ts";
import { KARUME_SOURCES } from "../src/sources.ts";
import { AnimaPipeline } from "../src/anima/pipeline.ts";
import { IrodoriPipeline } from "../src/irodori/pipeline.ts";
import { Sbv2Pipeline } from "../src/sbv2/pipeline.ts";
import { Gemma4Pipeline } from "../src/gemma/pipeline.ts";
import { Siglip2Pipeline } from "../src/siglip2/pipeline.ts";
import { DepthAnythingPipeline } from "../src/depth-anything/pipeline.ts";
import { BirefnetPipeline } from "../src/birefnet/pipeline.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** hub が解決要求を出さずに済む形（`@hdae/fetch-cache` の `isCommitSha` と同じ綴り）。 */
const COMMIT_SHA = /^[0-9a-f]{40}$/;

/** 取得 URL を記録して**その場で落とす** fetch（network へは 1 バイトも出ない）。 */
const recordingFetch = (urls: string[]): typeof globalThis.fetch => (input) => {
  urls.push(input instanceof Request ? input.url : String(input));
  return Promise.reject(new Error("sources_test: ここから先へは出ない"));
};

type LoadOptions = { fetch: typeof globalThis.fetch; caches: CacheStorage };

/**
 * `fromPretrained` が最初に取りに行く URL の pathname 列を返す。
 *
 * host は伏せる（`hubUrl` の既定はミラー設定で動く — 見るべきは repo と revision）。
 */
const requestedPaths = async (
  load: (options: LoadOptions) => Promise<unknown>,
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

/** 家族表 1 つぶんの素性（表そのもの + そのエントリを実際に読む `fromPretrained`）。 */
type Family = {
  readonly name: string;
  readonly sources: Record<string, HubRepoRef>;
  /** 期待するキー集合（**外部の期待**として名指しで置く — 下の `keys` の門を参照）。 */
  readonly keys: readonly string[];
  readonly load: (ref: HubRepoRef, options: LoadOptions) => Promise<unknown>;
};

const FAMILIES: readonly Family[] = [
  {
    name: "ANIMA_SOURCES",
    sources: ANIMA_SOURCES,
    // 公式モデル同居リポと第三者 fine-tune 同居リポ（ADR 0087）。extra は text stack を公式から
    // 越境参照するが、**manifest 自体は自分のリポから取る**（越境は manifest の中の話）。
    keys: ["anima", "anima-extra"],
    load: (ref, options) => AnimaPipeline.fromPretrained(ref, options),
  },
  {
    name: "IRODORI_SOURCES",
    sources: IRODORI_SOURCES,
    // 世代・版が変わるものは別リポ（ADR 0092 決定 1）— v4 と v4.1 が並ぶ。
    keys: ["irodori-v4-small", "irodori-v4.1-small"],
    load: (ref, options) => IrodoriPipeline.fromPretrained(ref, options),
  },
  {
    name: "SBV2_SOURCES",
    sources: SBV2_SOURCES,
    keys: ["sbv2-jvnv"],
    load: (ref, options) => Sbv2Pipeline.fromPretrained(ref, options),
  },
  {
    name: "GEMMA4_SOURCES",
    sources: GEMMA4_SOURCES,
    // 単一リポの家族もキーは家族名（ADR 0092 決定 3 — 1 本でも表の形を崩さない）。
    keys: ["gemma4"],
    load: (ref, options) => Gemma4Pipeline.fromPretrained(ref, options),
  },
  {
    name: "SIGLIP2_SOURCES",
    sources: SIGLIP2_SOURCES,
    // base と so400m は寸法だけが違う同じ経路なので 1 リポに同居する（既定 = base）。
    keys: ["siglip2"],
    load: (ref, options) => Siglip2Pipeline.fromPretrained(ref, options),
  },
  {
    name: "DEPTH_ANYTHING_SOURCES",
    sources: DEPTH_ANYTHING_SOURCES,
    // 世代はリポ名に入る（ADR 0092 決定 1）— V3 は別アーキなので別キーとして並ぶ。
    keys: ["depth-anything-v2"],
    load: (ref, options) => DepthAnythingPipeline.fromPretrained(ref, options),
  },
  {
    name: "BIREFNET_SOURCES",
    sources: BIREFNET_SOURCES,
    // checkpoint ごとに 1 リポ（派生 Lucida は別リポ — ADR 0092 決定 1）。各リポには解像度ごとの
    // 別グラフが "1024" / "2048" の 2 モデルとして同居する（決定 9）。
    keys: ["birefnet-hr", "lucida"],
    load: (ref, options) => BirefnetPipeline.fromPretrained(ref, options),
  },
];

// キーは repo 名から機械導出される（`"karume-" + key` === basename）ので、①の門は「キーと repo が
// **互いに**整合すること」しか見ていない — 両方を同時に書き換える改名（`karume-gemma4-e2b` →
// `karume-gemma4` の類）は素通りする。集合をここで外部の期待として固定すると、repo の綴りも
// 間接的に固定される（改名はこのテストを落として初めて通る = リリース手順を踏ませる席になる）。
for (const { name, sources, keys } of FAMILIES) {
  Deno.test(`${name}: キー集合が期待どおり`, () => {
    assertEquals(Object.keys(sources).toSorted(), [...keys].toSorted());
  });
}

// 家族と公開リポの総数も名指しで置く。リポの新設（波 b の birefnet-hr / lucida）はここを
// 落とすので、`KARUME_SOURCES` への畳み込み漏れが黙って通らない。
Deno.test("取得元対応表を持つのは 7 家族・公開リポは 10 本", () => {
  assertEquals(FAMILIES.map(({ name }) => name), [
    "ANIMA_SOURCES",
    "IRODORI_SOURCES",
    "SBV2_SOURCES",
    "GEMMA4_SOURCES",
    "SIGLIP2_SOURCES",
    "DEPTH_ANYTHING_SOURCES",
    "BIREFNET_SOURCES",
  ]);
  assertEquals(Object.keys(KARUME_SOURCES).length, 10);
});

for (const { name, sources } of FAMILIES) {
  for (const [key, source] of Object.entries(sources)) {
    Deno.test(`${name}["${key}"]: キーが repo から導ける + 40 桁 hex の commit SHA`, () => {
      const segments = source.repo.split("/");
      assertEquals(segments.length, 2, `repo が 'owner/name' でない: ${source.repo}`);
      assertEquals(segments[0], "hdae");
      assertEquals(`karume-${key}`, segments[1]);
      assertMatch(source.revision ?? "", COMMIT_SHA);
    });
  }
}

// 畳み込みはスプレッドなので、キーが重なると後勝ちで 1 本が黙って消える（同じリポを 2 家族が
// 持っていない限り起きないが、起きたときに気づく手段が他に無い）。
Deno.test("KARUME_SOURCES: 家族表の和集合でキーの重複が無い", () => {
  // リテラル型のままだとエントリ同士の比較が「別のキーの値」で型エラーになるので、表として
  // 引ける形（キー = 任意の綴り）へ広げてから中身を突き合わせる。
  const merged: Record<string, HubRepoRef> = KARUME_SOURCES;
  const entries = FAMILIES.flatMap(({ sources }) => Object.entries(sources));
  assertEquals(
    Object.keys(merged).length,
    entries.length,
    "キーが重なって畳み込みで消えたエントリがある",
  );
  // 陽性対照 — 空の表を回して緑になる形にしない。
  assert(entries.length >= 8, `家族表が痩せている（${entries.length} 件）`);
  for (const [key, source] of entries) assertEquals(merged[key], source);
});

// SHA 固定なので revision 解決 API を 1 往復もしない（= オフライン起動可）。取得 URL が
// `/api/models/.../revision/...` から始まったら pin が可変 ref へ戻っている。
for (const { name, sources, load } of FAMILIES) {
  for (const [key, source] of Object.entries(sources)) {
    Deno.test(`${name}["${key}"]: pin がそのまま取得 URL に載る`, async () => {
      const paths = await requestedPaths((options) => load(source, options));
      assertEquals(paths, [manifestPath({ repo: source.repo, revision: source.revision ?? "" })]);
    });
  }
}

// 文字列 ref の意味（`{ repo }` = main 追従）は pin 導入後も ref 必須化後も不変。main は
// commit SHA ではないので、manifest を取る**前に** revision 解決 API を 1 往復する。
Deno.test("sbv2: 文字列 ref は main 追従のまま（revision 解決 API を経由する）", async () => {
  const paths = await requestedPaths((options) =>
    Sbv2Pipeline.fromPretrained("someone/karume-sbv2-fork", options)
  );
  assertEquals(paths, ["/api/models/someone/karume-sbv2-fork/revision/main"]);
});
