// 実重み Gemma 4 E2B を**手元のディレクトリから直読み**して組む検収門 —— 取得元ハンドル
// （`@karume/hub/deno` の `denoDirectory`）を `fromPretrained` へ渡す経路の合格線。
//
// `e2e_gemma4_pretrained_test.ts`（疑似 HF サーバ = HTTP + 永続キャッシュ + self-heal）と
// **同じ配布形ミラーを、取得元だけ替えて**読む。門は 2 本:
//
//  ① **同じ golden**: 温度 0 の chat が `e2e_gemma4_chat_test.ts` / `e2e_gemma4_pretrained_test.ts`
//     と文字単位で一致する。取得元を替えても届くバイト列が同じであることの証明で、割れたら
//     動いたのはローカル取得元（size 門・越境・逐次面）である。グラフ shard → 重み shard の
//     逐次流し → **遅延側の PLE sidecar 3 本**まで、全部この 1 つの取得元から来る
//  ② **network も CacheStorage も 1 度も通らない**: 触れば落ちる `fetch` / `CacheStorage` を
//     渡し、かつ呼び出し回数 0 を数える（診断コールバックへ化けて握り潰される経路があっても
//     数で気づけるように、throw と数の両方で見る）。ローカル取得元の最大の利点は「バイト列の
//     複製が 1 つも増えない」ことで、キャッシュへ写す形へ退行するとここで落ちる
//
// ## 資産
//
// 配布形ミラー `models/karume-gemma4-e2b/`（`dist.py --pipeline gemma4` が組む）。リポジトリ
// 管理外なので、無い環境では**明示 SKIP** する。取得元がローカルなので、この門は約 4GiB を
// 永続キャッシュへ**残さない**（疑似 HF 経路との違い）。

import { assert, assertEquals } from "@std/assert";
import { MANIFEST_FILENAME } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
// MUST: 入口は**公開面**（`./gemma` サブパス）から取る — `src/...` を直に掴むと、面が痩せていても
// 門が緑のままになる（消費者が書けない経路で検収したことになる）。
import { type Gemma4ChatMessage, Gemma4Pipeline } from "../gemma.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const MIRROR_DIR = new URL("../../../models/karume-gemma4-e2b/", import.meta.url);

/** SKIP 時にそのまま貼れる組み立てコマンド。 */
const ASSEMBLE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline gemma4";

/**
 * PLE の常駐に使う予算（バイト）。現行世代の shard は 1 本 ≈253MiB なので 3 本ぶんが載る
 * （token 範囲をまたぐ会話で読み直しを増やしすぎない）。**本数ではなくバイト**で渡すのは、
 * shard 幅が資産世代で変わると「N 本」が別の RAM を意味するため（ADR 0085 追記 2026-09-02）。
 */
const MAX_RESIDENT_PLE_BYTES = 768 * 1024 * 1024;

/**
 * 検収ケース。**`e2e_gemma4_chat_test.ts` の CASES と同じ golden**（同じバイト列を別の取得元で
 * 読んでいることの証明なので、値を割ってはいけない）。
 */
const CASES = [
  {
    fixture: "single-user",
    maxNewTokens: 24,
    expected: "The capital of France is **Paris**.",
    stop: { reason: "eos", token: 106, tokens: 9 },
  },
  {
    fixture: "japanese",
    maxNewTokens: 24,
    expected: "2023年時点で、およそ1,400万人です。",
    stop: { reason: "eos", token: 106, tokens: 18 },
  },
] as const;

type ChatFixture = {
  readonly chat: {
    readonly name: string;
    readonly messages: Gemma4ChatMessage[];
    readonly ids: number[];
  }[];
};

const fixture = JSON.parse(
  await Deno.readTextFile(new URL("fixtures/gemma-text/gemma4-chat.json", import.meta.url)),
) as ChatFixture;

const caseOf = (name: string) => {
  const found = fixture.chat.find((row) => row.name === name);
  assert(found !== undefined, `フィクスチャに chat ケース '${name}' が無い`);
  return found;
};

const manifestExists = (): boolean => {
  try {
    return Deno.statSync(new URL(MANIFEST_FILENAME, MIRROR_DIR)).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const AVAILABLE = manifestExists();

if (!AVAILABLE) {
  console.warn(
    `[karume] 配布形ミラー models/karume-gemma4-e2b/ が無いためディレクトリ直読み検収を SKIP する。` +
      `組み立て: ${ASSEMBLE_COMMAND}`,
  );
}

/** 触れた瞬間に落ちる CacheStorage（呼ばれた回数も数える — 診断へ化けても気づけるように）。 */
class HostileCacheStorage implements CacheStorage {
  calls = 0;

  open(): Promise<Cache> {
    this.calls += 1;
    throw new Error("test: ローカル取得元が CacheStorage を開いた");
  }
  match(): Promise<Response | undefined> {
    this.calls += 1;
    throw new Error("test: ローカル取得元が CacheStorage を引いた");
  }
  has(): Promise<boolean> {
    this.calls += 1;
    throw new Error("test: ローカル取得元が CacheStorage を引いた");
  }
  delete(): Promise<boolean> {
    this.calls += 1;
    throw new Error("test: ローカル取得元が CacheStorage を消した");
  }
  /** 旧名前空間の回収（hub のセッション入口）だけは通す — 消すものが無いことを答える。 */
  keys(): Promise<string[]> {
    return Promise.resolve([]);
  }
}

Deno.test({
  name: "gemma4 ディレクトリ直読み: fromPretrained → chat が同じ golden を出す（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const caches = new HostileCacheStorage();
    let fetchCalls = 0;
    const fetchStub: typeof globalThis.fetch = (input) => {
      fetchCalls += 1;
      return Promise.reject(
        new Error(`test: ローカル取得元が network へ出た（${String(input)}）`),
      );
    };
    const cacheDiagnostics: string[] = [];

    const pipeline = await Gemma4Pipeline.fromPretrained(denoDirectory(MIRROR_DIR), {
      maxResidentPleBytes: MAX_RESIDENT_PLE_BYTES,
      // HTTP 取得元専用のノブ。ローカル取得元では 1 つも効かない = 触られない。
      fetch: fetchStub,
      caches,
      onCacheError: (diagnostic) => cacheDiagnostics.push(diagnostic.url),
    });
    try {
      await t.step(
        "① 温度 0 の chat が fromAssets / 疑似 HF 経路と同じ golden を出す",
        async () => {
          for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
            const { messages } = caseOf(name);
            const started = performance.now();
            // 明示の温度 0 が配布形の推奨サンプラ（温度 1）を上書きする — golden は greedy の列。
            const stream = pipeline.chat(messages, { maxNewTokens, sampler: { temperature: 0 } });
            const parts: string[] = [];
            for await (const chunk of stream) parts.push(chunk);
            const stopped = await stream.done;
            console.log(
              `[e2e] gemma4 denoDirectory ${name}: ${JSON.stringify(parts.join(""))} / ` +
                `${JSON.stringify(stopped)} / ${(performance.now() - started).toFixed(0)}ms`,
            );
            assertEquals(parts.join(""), expected, `${name}: 温度 0 の出力`);
            assertEquals(stopped, stop, `${name}: 停止理由`);
          }
        },
      );

      await t.step("② network も CacheStorage も 1 度も通っていない", () => {
        assertEquals(fetchCalls, 0, "ローカル取得元が fetch を呼んだ");
        assertEquals(caches.calls, 0, "ローカル取得元が CacheStorage を触った");
        // 触っていないので診断も 1 本も来ない（キャッシュ失敗を握り潰す経路に落ちていない）。
        assertEquals(cacheDiagnostics, [], "キャッシュ診断が飛んでいる");
      });
    } finally {
      await pipeline.dispose();
    }
  },
});
