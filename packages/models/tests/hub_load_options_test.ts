// 8 家族の `fromPretrained` が hub へ透過するオプション（`src/hub/load-options.ts`）の門。
// **GPU も実網も要らない**。
//
// 0.11.0 の hub が `onRetry`（429 / 503 の再試行通知）を足したとき、8 家族の
// `XFromPretrainedOptions` は同じ 5 欄を手書きで複製していたため 8 か所とも欄が欠け、
// 「新しいノブが黙って届かない」形になった。複製を 1 本（{@link hubLoadOptions}）へ畳んだので、
// ここで押さえるのは 4 つ:
//  ① 定義済みの 6 欄が**同じ参照のまま**写ること（写しの途中で包み直さない）。
//  ② 未定義の欄は**キーごと**現れないこと — `key: undefined` は「無指定」と別物として
//     取得層の分岐に効きうる（明示 `undefined` の `fetch` が既定 `globalThis.fetch` を潰す等）。
//  ③ `onProgress` は写さないこと（進捗は家族側が `loadContainerComponents` へ別途載せる）。
//  ④ 型の門 — 8 家族**全て**が `onRetry` を受けること（欠けたら `deno task check` が赤くなる）。
// 最後に、公開面から見た対として「anima の `fromPretrained` が `onRetry` を `loadManifest` まで
// 運ぶ」ことを mock fetch の 429 で見る（1 家族で足りる — 運ぶ経路は 8 家族とも同じ 1 本）。

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import type { AssetProgress, CacheDiagnostic, RetryDiagnostic } from "@karume/hub";
import { hubLoadOptions } from "../src/hub/load-options.ts";
import { AnimaPipeline } from "../src/anima/pipeline.ts";
import type { AnimaFromPretrainedOptions } from "../src/anima/pipeline.ts";
import type { BirefnetFromPretrainedOptions } from "../src/birefnet/pipeline.ts";
import type { DepthAnythingFromPretrainedOptions } from "../src/depth-anything/pipeline.ts";
import type { Gemma4FromPretrainedOptions } from "../src/gemma/pipeline.ts";
import type { IrodoriFromPretrainedOptions } from "../src/irodori/pipeline.ts";
import type { Sbv2FromPretrainedOptions } from "../src/sbv2/pipeline.ts";
import type { Siglip2FromPretrainedOptions } from "../src/siglip2/pipeline.ts";
import type { VowelDetectorFromPretrainedOptions } from "../src/vowel-detector/pipeline.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

const noopRetry = (diagnostic: RetryDiagnostic): void => void diagnostic;

Deno.test("hubLoadOptions: 定義済みの 6 欄が同じ参照のまま写る", () => {
  const signal = AbortSignal.abort();
  const headers = new Headers({ authorization: "Bearer test" });
  const onCacheError = (diagnostic: CacheDiagnostic): void => void diagnostic;
  const fetchStub: typeof globalThis.fetch = () => Promise.reject(new Error("test: 呼ばない"));
  const caches = new MemoryCacheStorage();

  const hubOptions = hubLoadOptions({
    signal,
    headers,
    onCacheError,
    onRetry: noopRetry,
    fetch: fetchStub,
    caches,
  });

  // 包み直すと「アプリが渡したのと別の関数が呼ばれる」形になり、購読の解除も比較もできない。
  assertStrictEquals(hubOptions.signal, signal);
  assertStrictEquals(hubOptions.headers, headers);
  assertStrictEquals(hubOptions.onCacheError, onCacheError);
  assertStrictEquals(hubOptions.onRetry, noopRetry);
  assertStrictEquals(hubOptions.fetch, fetchStub);
  assertStrictEquals(hubOptions.caches, caches);
  assertEquals(Object.keys(hubOptions).length, 6, "写した欄が 6 つでない");
});

Deno.test("hubLoadOptions: 未定義の欄はキーごと無い（`key: undefined` を作らない）", () => {
  // 明示的な `undefined` を置くと、取得層の `options.fetch ?? globalThis.fetch` のような
  // 既定の当て方は救うが、`Object.hasOwn` や分割代入の既定値で分岐する側は救えない。
  // 「無指定」と「undefined を指定」は別物のまま届けるのが写しの契約。
  for (const options of [{}, { signal: undefined, headers: undefined, onRetry: undefined }]) {
    const hubOptions: Record<string, unknown> = hubLoadOptions(options);
    assertEquals(Object.keys(hubOptions), [], "未定義の欄がキーとして現れている");
    for (const key of ["signal", "headers", "onCacheError", "onRetry", "fetch", "caches"]) {
      assertEquals(Object.hasOwn(hubOptions, key), false, `${key} のキーが作られている`);
    }
  }
});

Deno.test("hubLoadOptions: onProgress は写さない（進捗は家族側が別途載せる）", () => {
  // manifest 取得に進捗は無く、資産取得の進捗は家族ごとに集約してから
  // `loadContainerComponents` へ渡す。ここで写すと集約前の生の進捗が二重に流れる。
  const onProgress = (progress: AssetProgress): void => void progress;
  const hubOptions: Record<string, unknown> = hubLoadOptions({ onProgress, onRetry: noopRetry });
  assertEquals(Object.hasOwn(hubOptions, "onProgress"), false, "onProgress が写っている");
  assertEquals(Object.keys(hubOptions), ["onRetry"]);
});

Deno.test("型の門: 8 家族の fromPretrained オプションが onRetry を受ける", () => {
  // `satisfies` が本体（`deno task check` が門）。1 家族でも共有型から外れると赤くなる。
  const perFamily = [
    { onRetry: noopRetry } satisfies AnimaFromPretrainedOptions,
    { onRetry: noopRetry } satisfies BirefnetFromPretrainedOptions,
    { onRetry: noopRetry } satisfies DepthAnythingFromPretrainedOptions,
    { onRetry: noopRetry } satisfies Gemma4FromPretrainedOptions,
    { onRetry: noopRetry } satisfies IrodoriFromPretrainedOptions,
    { onRetry: noopRetry } satisfies Sbv2FromPretrainedOptions,
    { onRetry: noopRetry } satisfies Siglip2FromPretrainedOptions,
    { onRetry: noopRetry } satisfies VowelDetectorFromPretrainedOptions,
  ];
  assertEquals(perFamily.length, 8, "家族が 8 つ並んでいない");
  // 写しの入口（`hubLoadOptions`）が 8 家族の型を**そのまま**受けることも同じ 1 本で見る。
  for (const options of perFamily) assertStrictEquals(hubLoadOptions(options).onRetry, noopRetry);
});

// ---- 公開面から見た透過（anima 1 家族）----------------------------------------

const HUB_URL = "https://hub.test";
const REPO = "someone/karume-anima";
const SHA = "0123456789abcdef0123456789abcdef01234567";
const MANIFEST_URL = `${HUB_URL}/${REPO}/resolve/${SHA}/karume.json`;

/**
 * 部品 1 本ぶんの dtype エントリ（実体は 1 度も届かないので sha256 は綴りだけ合わせる）。
 * part 0 の size は「ヘッダ 24 + 2 文書」と一致させる（manifest の門 — container-v1 §8）。
 */
const weightEntry = (stem: string) => ({
  f16: {
    container: {
      descriptor: {
        graph: { length: 10, sha256: "b".repeat(64) },
        model: { length: 8, sha256: "c".repeat(64) },
      },
      parts: [
        { path: `${stem}-00001-of-00002.krm`, size: 42, sha256: "a".repeat(64) },
        { path: `${stem}-00002-of-00002.krm`, size: 64, sha256: "d".repeat(64) },
      ],
    },
  },
});

const MANIFEST = {
  format: "karume/5",
  generator: "karume/0.11.0",
  defaultModel: "anima",
  models: {
    anima: {
      pipeline: "anima/1",
      weights: {
        text_encoder: weightEntry("text_encoder/model.f16"),
        text_conditioner: weightEntry("text_conditioner/model.f16"),
        transformer: weightEntry("transformer/model.f16"),
        vae_decoder: weightEntry("vae_decoder/model.f16"),
      },
      quants: {
        f16: {
          weights: {
            text_encoder: "f16",
            text_conditioner: "f16",
            transformer: "f16",
            vae_decoder: "f16",
          },
          session: {},
        },
      },
      defaultQuant: "f16",
      // parse が席の存在を要求する 2 つ（空でも明示 — 中身はこのテストでは読まれない）。
      assets: { tokenizer: { path: "tokenizer/tokenizer.json", size: 42, sha256: "b".repeat(64) } },
      pipelineConfig: {},
    },
  },
};

/**
 * `karume.json` の**初回だけ** 429（`retry-after: 0`）を返し、2 回目で manifest を返す `fetch`。
 * それ以外の URL（= グラフ shard）は 404 — 見たいのは manifest 取得までの経路だけなので、
 * その先は落ちてよい。
 */
const createRateLimitedManifestFetch = (): { fetch: typeof globalThis.fetch; calls: string[] } => {
  const calls: string[] = [];
  let rateLimited = false;
  const fetch: typeof globalThis.fetch = (input) => {
    const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    calls.push(href);
    if (href !== MANIFEST_URL) {
      return Promise.resolve(new Response(null, { status: 404, statusText: "Not Found" }));
    }
    if (!rateLimited) {
      rateLimited = true;
      return Promise.resolve(
        new Response(null, {
          status: 429,
          statusText: "Too Many Requests",
          headers: { "retry-after": "0" },
        }),
      );
    }
    return Promise.resolve(
      new Response(JSON.stringify(MANIFEST), { headers: { "content-type": "application/json" } }),
    );
  };
  return { fetch, calls };
};

Deno.test("AnimaPipeline.fromPretrained: onRetry が loadManifest まで届く（429 で 1 回）", async () => {
  const { fetch, calls } = createRateLimitedManifestFetch();
  const retries: RetryDiagnostic[] = [];
  // manifest の後（容器の part 0）は 404 なので落ちる。ここで見たいのは「家族の
  // `fromPretrained` が `onRetry` を取得層まで運ぶ」ことだけ。
  await assertRejects(() =>
    AnimaPipeline.fromPretrained({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, {
      fetch,
      caches: new MemoryCacheStorage(),
      onRetry: (diagnostic) => retries.push(diagnostic),
    })
  );
  assertEquals(retries.length, 1, "再試行の通知が 1 回だけ届いていない");
  assertEquals(retries[0].status, 429);
  assertEquals(retries[0].url, MANIFEST_URL, "通知が karume.json 以外の URL を名乗っている");
  assertEquals(retries[0].attempt, 1);
  // 429 の後に取り直していること（= 通知だけ出して諦めた形でない）。
  assertEquals(calls.filter((url) => url === MANIFEST_URL).length, 2);
  assert(
    calls.some((url) => url.endsWith("/text_encoder/model.f16-00001-of-00002.krm")),
    "manifest を読めていない（descriptor の取得まで進んでいない）",
  );
});
