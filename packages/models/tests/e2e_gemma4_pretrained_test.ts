// 実重み Gemma 4 E2B を**配布形から**組む検収門 —— 段 5 の合格線（実 DL 疎通）。
//
// `fromAssets`（段 4 の門は `e2e_gemma4_chat_test.ts`）と違い、こちらはロード経路を本番と
// 1 本にする: `karume.json` の解決 → グラフ shard だけ取って admission → 重み shard の逐次流し
// → PLE sidecar の遅延読み。門は 4 本:
//
//  ① **完走**: `fromPretrained` → `chat` が実重みで回り、温度 0（明示）の出力が
//     `e2e_gemma4_chat_test.ts` と**同じ golden** に一致する。同じバイト列を別経路で読んでいる
//     ことの証明で、割れたら動いたのは配布形か取得経路である
//  ② **既定サンプラの結線**（ADR 0083 決定 7）: 配布形が宣言した推奨値がそのまま
//     `pipeline.defaultSampler` に載り、要求が `sampler` を省略したときの既定になる。宣言が
//     `pipelineConfig` に焼かれていても結線が抜けていれば、①（温度 0 を明示する経路）は緑の
//     ままなので、この門が要る
//  ③ **PLE sidecar が全量常駐しない**（ADR 0085 決定 3）: 索引の shard は `assets` の遅延側で
//     受け、触った 1 本だけを読む。全量を `fetchAssets` へ流す形へ戻ると 2.27GiB が常駐する
//  ④ **配布形の宣言と現物の対応**: 索引が名指しする shard ファイル名が manifest の asset 名
//     そのものであること（取得キーを「並び順で合わせる」形に戻っていないこと）
//
// ## 資産と経路
//
// ローカルの配布形ミラー `models/karume-gemma4-e2b/`（`dist.py --pipeline gemma4` が組む）を
// **HF 形の HTTP** で配る使い捨てサーバ越しに読む（`examples/shared/local-dist-server.ts` —
// デモが `--source <ローカルのパス>` で通るのと同じ 1 本）。ミラーはリポジトリ管理外なので、
// 無い環境では**明示 SKIP** する。
//
// NOTE: テストから `examples/` の道具を借りるのは、疑似 HF サーバの綴りを 2 つ持たないため
// （`packages/models/tests/` も `examples/` も publish には入らない）。取得層は資産を
// **永続キャッシュ**（Deno は DENO_DIR）へ写すので、この門を 1 度回すと約 4GiB がそこに残る
// （掃除は `@karume/hub` の `clearHubCache`）。

import { assert, assertEquals } from "@std/assert";
import { MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { Gemma4Pipeline } from "../src/gemma/pipeline.ts";
import type { Gemma4ChatMessage } from "../src/gemma/text/chat.ts";
import { serveLocalDist } from "../../../examples/shared/local-dist-server.ts";
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

/** 上流 `generation_config.json` の推奨（配布形が `pipelineConfig.sampler` へ焼いた値）。 */
const RECOMMENDED_SAMPLER = { temperature: 1, topK: 64, topP: 0.95 } as const;

/**
 * 検収ケース。**`e2e_gemma4_chat_test.ts` の CASES と同じ golden**（同じ重み・同じ静的配線を
 * 別経路で読んでいることの証明なので、値を割ってはいけない）。
 */
const CASES = [
  {
    fixture: "single-user",
    maxNewTokens: 24,
    expected: "The capital of France is **Paris**.",
    stop: { reason: "eos", token: 106, tokens: 9 },
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

const manifestText = (): string | undefined => {
  try {
    return Deno.readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR));
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
    throw cause;
  }
};

const MANIFEST_TEXT = manifestText();
const AVAILABLE = MANIFEST_TEXT !== undefined;

if (!AVAILABLE) {
  console.warn(
    `[karume] 配布形ミラー models/karume-gemma4-e2b/ が無いため段 5 の疎通検収を SKIP する。` +
      `組み立て: ${ASSEMBLE_COMMAND}`,
  );
}

Deno.test({
  name: "gemma4 配布形: manifest の宣言と PLE 索引の対応（GPU 不要）",
  ignore: !AVAILABLE,
  fn: () => {
    const manifest = parseManifest(MANIFEST_TEXT ?? "");
    const entry = manifest.models[manifest.defaultModel];
    assertEquals(entry.pipeline, { name: "gemma4", major: 1 }, "pipeline 契約");
    assertEquals(
      Object.keys(entry.weights),
      ["model"],
      "weights は製品グラフ 1 本（PLE も tokenizer も assets の席）",
    );
    // ④ 索引が名指しする shard ファイル名が、そのまま manifest の asset 名（= 取得キー）。
    const index = JSON.parse(
      Deno.readTextFileSync(new URL(entry.assets["ple_index"].path, MIRROR_DIR)),
    ) as { readonly shards: readonly { readonly file: string }[] };
    const declared = index.shards.map((shard) => shard.file);
    assert(declared.length > 0, "PLE 索引が shard を 1 本も持たない");
    for (const file of declared) {
      assert(
        Object.hasOwn(entry.assets, file),
        `PLE 索引の '${file}' が manifest の assets に無い（取得キーは索引の綴りそのもの）`,
      );
    }
    // 逆向き（assets 側に索引の知らない sidecar が残っていない）。
    const sidecars = Object.keys(entry.assets).filter(
      (name) => name !== "tokenizer" && name !== "ple_index",
    );
    assertEquals([...sidecars].sort(), [...declared].sort(), "assets の sidecar と索引の対応");
  },
});

Deno.test({
  name: "gemma4 配布形: fromPretrained → chat が実 GPU で完走し既定サンプラが宣言どおり",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    await using server = serveLocalDist(new URL(".", MIRROR_DIR).pathname);
    const pipeline = await Gemma4Pipeline.fromPretrained(server.source, {
      maxResidentPleBytes: MAX_RESIDENT_PLE_BYTES,
    });
    try {
      await t.step("② 配布形が宣言した推奨サンプラが省略時の既定として載っている", () => {
        assertEquals(pipeline.defaultSampler, RECOMMENDED_SAMPLER);
      });

      await t.step("① 温度 0 の chat が fromAssets 経路と同じ golden を出す", async () => {
        for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
          const { messages } = caseOf(name);
          const started = performance.now();
          const stream = pipeline.chat(messages, {
            // 明示の温度 0 が既定（推奨サンプラ）を上書きする — golden は greedy の列。
            maxNewTokens,
            sampler: { temperature: 0 },
          });
          const parts: string[] = [];
          for await (const chunk of stream) parts.push(chunk);
          const stopped = await stream.done;
          console.log(
            `[e2e] gemma4 fromPretrained ${name}: ${JSON.stringify(parts.join(""))} / ` +
              `${JSON.stringify(stopped)} / ${(performance.now() - started).toFixed(0)}ms`,
          );
          assertEquals(parts.join(""), expected, `${name}: 温度 0 の出力`);
          assertEquals(stopped, stop, `${name}: 停止理由`);
        }
      });

      await t.step("③ 既定サンプラ（温度 1.0）でも同じ会話が完走する", async () => {
        const { messages } = caseOf(CASES[0].fixture);
        // `sampler` を省略 = 配布形の宣言で走る（抽選は splitmix64 で決定的だが、値そのものは
        // 固定しない —— HF の sampling とは列が一致しないので golden にする意味が無い）。
        const stream = pipeline.chat(messages, { maxNewTokens: 8 });
        const parts: string[] = [];
        for await (const chunk of stream) parts.push(chunk);
        const stopped = await stream.done;
        console.log(`[e2e] gemma4 既定サンプラ: ${JSON.stringify(parts.join(""))}`);
        assert(parts.join("").length > 0, "既定サンプラで 1 文字も出ていない");
        assert(
          stopped.reason === "eos" || stopped.reason === "max-tokens",
          `停止理由 ${stopped.reason}`,
        );
      });
    } finally {
      await pipeline.dispose();
    }
  },
});
