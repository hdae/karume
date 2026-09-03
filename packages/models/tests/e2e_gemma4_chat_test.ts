// 実重み Gemma 4 E2B の **文字列 in → 文字列 out** の検収門 — 段 4 の合格線。
//
// 検収するのは `src/gemma/pipeline.ts`（`Gemma4Pipeline`）と `src/gemma/text/chat.ts`
// （`gemma4ChatPrompt`）の結線で、門は 5 本:
//
// ① **完走**: `fromAssets` → `chat([...])` が実重みで回り、温度 0（この層の既定）の出力が
//    固定した greedy golden と文字単位で一致する。golden は**この経路自身で採った**もので、
//    モデルの正しさは証明しない — 証明するのは「結線を変えても出力が動かない」ことである
//    （数値の正は `e2e_gemma4_product_test.ts` の交差 parity が持つ）
// ② **chat の id 列**が HF `apply_chat_template` のフィクスチャと一致する。`gemma_chat_test.ts`
//    は部分集合と実資産で同じことを見るが、こちらは**パイプラインが握っている資産**で見る
//    （別の tokenizer を掴んでいれば描画は合ったまま id 列だけが違う）
// ③ **逐次と一括の一致**: streaming の片を連結したものが、同じ会話を低レベル面（`sequence()`）
//    で回して得た token 列の一括 decode と一致する。逐次復号が byte run を取りこぼしたり
//    「finish() まで全部溜める偽 streaming」になっても①は緑のままなので、この門が要る
// ④ **停止集合**が上流の `generation_config.json` の宣言（`[1, 106, 50]`）と一致する
// ⑤ **射程外は GPU に触る前に落ちる**（tools / 未知 role — `chat` は同期に throw する）
// ⑥ **増分描画の多ターン**（`gemma4ChatTurn` + `sequence()`）が、同じ会話を毎ターン全体描画で
//    回した `chat()` と**逐語一致**する。turn-local 契約（`gemma_chat_test.ts` の門）が id 列の
//    等式で、こちらは同じ等式を**実重みの出力**で見る門である
// ⑦ **`Gemma4ChatSession` の 2 ターン**が⑥と同じ文字列になる。⑥は「差分描画を手で回せば一致
//    する」を、⑦は「中間層に任せても同じ会話になる」を見る（KV の継続条件と履歴の積み方は
//    セッションの中にあるので、実重みで見る門はここにしかない — `gemma_chat_session_test.ts`
//    は偽 sequence で組み立てだけを見る）
//
// MUST: 入口は公開面（`../gemma.ts`）から import する — `src/...` を直に掴むと、面が痩せていても
// 門が緑のままになる（消費者が書けない経路で検収したことになる）。
//
// ## 資産
//
// `outputs/series/gemma4-e2b-product/`（製品グラフ + PLE sidecar）と
// `outputs/series/gemma4-e2b-tokenizer/tokenizer.json`。どちらもリポジトリ管理外で、無い環境
// では**明示 SKIP** する。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
// MUST: 入口は**公開面**（`./gemma` サブパス）から取る — 消費者が書けない import で検収すると、
// 「面が痩せていること」に門が気づけない（`src/...` を直に掴めば何でも見える）。
import {
  type Gemma4ChatMessage,
  gemma4ChatPrompt,
  Gemma4ChatSession,
  type Gemma4ChatStream,
  gemma4ChatTurn,
  Gemma4Pipeline,
} from "../gemma.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const PRODUCT_ROOT = new URL("../../../outputs/series/gemma4-e2b-product/", import.meta.url);
const TOKENIZER_ASSET = new URL(
  "../../../outputs/series/gemma4-e2b-tokenizer/tokenizer.json",
  import.meta.url,
);
const PLE_INDEX_FILE = "ple.json";
const MODEL_SHARD = /^model-\d+-of-\d+\.safetensors$/;

/** SKIP 時にそのまま貼れる生成コマンド。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m gemma4.export_product" +
  "（tokenizer 資産は … python -m gemma4.tokenizer）";

/** 実行条件は既存の gemma4 検収門と同値（同じ資産世代の裁定をそのまま使う）。 */
const CHUNK_LENGTH = 768;
/** 記号 `M` を焼いた trace 上限（配布形の宣言 `maxChunkLength` と同値 — `chunkLength` の門）。 */
const MAX_CHUNK_LENGTH = 768;
const CAPACITY = 4096;
const MAX_POSITION = 131072;
/**
 * RoPE のパラメータ（配布形の宣言と同じ値 — 表は資産に無く、cos / sin はホストが作る）。
 * 正本は上流 config で、綴りと値域の門は `src/gemma/rope.ts`。
 */
const ROPE = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
} as const;
/**
 * PLE の常駐に使う予算（バイト）。現行世代の shard は 1 本 ≈253MiB なので 3 本ぶんが載る
 * （token 範囲をまたぐ会話で読み直しを増やしすぎない）。**本数ではなくバイト**で渡すのは、
 * shard 幅が資産世代で変わると「N 本」が別の RAM を意味するため（ADR 0085 追記 2026-09-02）。
 */
const MAX_RESIDENT_PLE_BYTES = 768 * 1024 * 1024;

/** gemma-4-E2B-it の `generation_config.json` の `eos_token_id`（ADR 0083 決定 8）。 */
const STOP_TOKENS = [1, 106, 50];

/**
 * 検収ケース。`prompt` はフィクスチャ（`gemma4-chat.json`）のケース名で、`expected` は
 * **この経路で採った温度 0 の出力**である（2026-08-31・RTX 3080 Ti）。
 *
 * MUST: 期待値を「実行結果で上書きして緑にする」ことをしない — 割れたら、まず
 * `e2e_gemma4_product_test.ts` の交差 parity（torch との突合）が緑かどうかを見る。あちらが
 * 緑でここだけ割れるなら、動いたのは結線（chat の描画 / 逐次復号 / sampler の既定）である。
 */
const CASES = [
  {
    fixture: "single-user",
    maxNewTokens: 24,
    expected: "The capital of France is **Paris**.",
    // `<turn|>`（106）で自分から turn を閉じる = 停止集合が実出力に効いている証拠。
    // `tokens` は**停止 token 込み**の生成数（`GenerationStop.tokens` の doc — 本文は 8 個）。
    stop: { reason: "eos", token: 106, tokens: 9 },
  },
  {
    fixture: "japanese",
    maxNewTokens: 24,
    // 多ターン（system + user + assistant + user）の 4 通目への応答。
    expected: "2023年時点で、およそ1,400万人です。",
    stop: { reason: "eos", token: 106, tokens: 18 },
  },
] as const;

/** ⑥ の 2 ターン目（1 ターン目は `CASES[0]` の会話をそのまま使う）。 */
const FOLLOW_UP: Gemma4ChatMessage = { role: "user", content: "And of Japan?" };
const FOLLOW_UP_TOKENS = 24;

type ChatFixture = {
  readonly stopTokens: number[];
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

const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const shardFiles = (): string[] => {
  try {
    return [...Deno.readDirSync(PRODUCT_ROOT)]
      .map((entry) => entry.name)
      .filter((name) => MODEL_SHARD.test(name))
      .sort();
  } catch {
    return [];
  }
};

const MODEL_SHARDS = shardFiles();
const AVAILABLE = MODEL_SHARDS.length > 0 &&
  exists(new URL(PLE_INDEX_FILE, PRODUCT_ROOT)) && exists(TOKENIZER_ASSET);

if (!AVAILABLE) {
  console.warn(
    `[karume] 製品系列 / tokenizer 資産が無いため Gemma 4 E2B chat 検収を SKIP する。` +
      `生成: ${GENERATE_COMMAND}`,
  );
}

/**
 * ファイル 1 本を `ArrayBuffer` として読む。
 * MUST: view が buffer 全体を覆っているなら slice しない（PLE sidecar は 1 本 758MB 級）。
 */
const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const openPipeline = async (): Promise<Gemma4Pipeline> => {
  const model: Uint8Array<ArrayBuffer>[] = [];
  for (const file of MODEL_SHARDS) model.push(new Uint8Array(await readBuffer(PRODUCT_ROOT, file)));
  return await Gemma4Pipeline.fromAssets({
    config: {
      chunkLength: CHUNK_LENGTH,
      maxChunkLength: MAX_CHUNK_LENGTH,
      maxPosition: MAX_POSITION,
      capacity: CAPACITY,
      rope: ROPE,
    },
    model,
    tokenizer: await Deno.readFile(TOKENIZER_ASSET),
    pleIndex: await Deno.readFile(new URL(PLE_INDEX_FILE, PRODUCT_ROOT)),
    readPleShard: (file) => readBuffer(PRODUCT_ROOT, file),
  }, { maxResidentPleBytes: MAX_RESIDENT_PLE_BYTES });
};

Deno.test({
  name: "Gemma 4 E2B chat 検収: 文字列 in → 文字列 out・逐次と一括の一致（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const pipeline = await openPipeline();
    /** dispose の**前**に発行し、汲み始めるのは**後**にする stream（下の寿命の門で使う）。 */
    let issuedBeforeDispose: Gemma4ChatStream | undefined;
    try {
      await t.step("① 温度 0 の chat が実重みで完走し、固定した greedy golden と一致", async () => {
        for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
          const { messages } = caseOf(name);
          const started = performance.now();
          const stream = pipeline.chat(messages, { maxNewTokens });
          const parts: string[] = [];
          for await (const chunk of stream) parts.push(chunk);
          const stopped = await stream.done;

          console.log(
            `[e2e] gemma4 chat ${name}: ${JSON.stringify(parts.join(""))} / ` +
              `${JSON.stringify(stopped)} / ${(performance.now() - started).toFixed(0)}ms`,
          );
          assertEquals(parts.join(""), expected, `${name}: 温度 0 の出力`);
          assertEquals(stopped, stop, `${name}: 停止理由`);
          // 片は「確定したぶんだけ」なので空文字は流れない（偽の 1 反復を作らない）。
          assertEquals(parts.filter((part) => part === "").length, 0, `${name}: 空の片`);
        }
      });

      await t.step("② パイプラインが握る資産の chat id 列がフィクスチャと一致", () => {
        for (const { fixture: name } of CASES) {
          const { messages, ids } = caseOf(name);
          assertEquals(gemma4ChatPrompt(pipeline.tokenizer, messages), ids, name);
        }
      });

      await t.step("③ streaming の片の連結 = 同じ会話の token 列の一括 decode", async () => {
        for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
          const { ids } = caseOf(name);
          // 低レベル面は 1 会話 = 1 sequence（context を跨がせない）。
          const sequence = await pipeline.sequence();
          try {
            const stream = sequence.generate({ prompt: ids, maxNewTokens });
            const tokens: number[] = [];
            for await (const event of stream) if (event.kind === "token") tokens.push(event.id);
            assertEquals(await stream.done, stop, `${name}: 低レベル面の停止理由`);
            assertEquals(
              pipeline.tokenizer.decode(tokens),
              expected,
              `${name}: 一括 decode（逐次が byte run を取りこぼしていれば割れる）`,
            );
          } finally {
            await sequence.dispose();
          }
        }
      });

      await t.step("④ 停止集合は上流の generation_config.json の宣言と一致", () => {
        const ascending = (ids: readonly number[]): number[] => [...ids].sort((a, b) => a - b);
        assertEquals(ascending(pipeline.program.stopTokens), ascending(STOP_TOKENS), "program");
        assertEquals(ascending(fixture.stopTokens), ascending(STOP_TOKENS), "フィクスチャ");
      });

      await t.step("⑤ 射程外の会話は GPU に触る前に落ちる（同期の throw）", () => {
        assertThrows(
          () =>
            pipeline.chat(
              [{ role: "user", content: "hi", tools: [] }] as unknown as Gemma4ChatMessage[],
              { maxNewTokens: 4 },
            ),
          Error,
          "射程外の欄",
        );
        assertThrows(
          () =>
            pipeline.chat(
              [{ role: "tool", content: "…" }] as unknown as Gemma4ChatMessage[],
              { maxNewTokens: 4 },
            ),
          Error,
          "role",
        );
      });

      /** ⑥ が採った 2 ターン目の参照（⑦ が同じ会話をセッションで組んで突き合わせる）。 */
      let followUpAnswer: string | undefined;

      await t.step("⑥ 増分描画の多ターンが chat() の全体描画と逐語一致", async () => {
        const { fixture: name, maxNewTokens } = CASES[0];
        const first = caseOf(name).messages;

        /** 高レベル面（毎ターン会話全体を描き直す）で 1 ターン汲む。 */
        const viaChat = async (
          messages: readonly Gemma4ChatMessage[],
          limit: number,
        ): Promise<string> => {
          const parts: string[] = [];
          for await (const chunk of pipeline.chat(messages, { maxNewTokens: limit })) {
            parts.push(chunk);
          }
          return parts.join("");
        };

        const answer1 = await viaChat(first, maxNewTokens);
        // 2 ターン目の参照は「1 ターン目の応答を会話へ入れて全体を描き直した」もの。
        const grown: Gemma4ChatMessage[] = [
          ...first,
          { role: "assistant", content: answer1 },
          FOLLOW_UP,
        ];
        const answer2 = await viaChat(grown, FOLLOW_UP_TOKENS);
        followUpAnswer = answer2;

        // 低レベル面: 1 会話 = 1 sequence。2 ターン目に流すのは**差分だけ**で、前 turn を閉じる
        // `<turn|>` は `pendingToken` が前置する（turn-local 契約 — ADR 0083 決定 4 / 0084 決定 5）。
        const sequence = await pipeline.sequence();
        try {
          const turn = async (prompt: readonly number[], limit: number) => {
            const stream = sequence.generate({ prompt, maxNewTokens: limit });
            const ids: number[] = [];
            for await (const event of stream) if (event.kind === "token") ids.push(event.id);
            return { text: pipeline.tokenizer.decode(ids), stop: await stream.done };
          };

          const prompt1 = gemma4ChatPrompt(pipeline.tokenizer, first);
          const first1 = await turn(prompt1, maxNewTokens);
          assertEquals(first1.text, answer1, "1 ターン目（全体描画は chat と同じ経路）");
          // `used` は「会話が占めている論理位置」— prompt + 生成（停止 token 込み）で説明が付く。
          assertEquals(
            sequence.used,
            prompt1.length + first1.stop.tokens,
            "1 ターン目の後の used",
          );

          const delta = gemma4ChatTurn(pipeline.tokenizer, FOLLOW_UP);
          const second = await turn(delta, FOLLOW_UP_TOKENS);
          console.log(
            `[e2e] gemma4 増分ターン: ${JSON.stringify(second.text)} / ` +
              `差分 ${delta.length} token / used ${sequence.used}`,
          );
          assertEquals(
            second.text,
            answer2,
            "2 ターン目が全体描画の chat() と逐語一致しない（差分描画か pendingToken の連結が" +
              "会話を別物にしている）",
          );
          assertEquals(
            sequence.used,
            prompt1.length + first1.stop.tokens + delta.length + second.stop.tokens,
            "2 ターン目の後の used",
          );
        } finally {
          await sequence.dispose();
        }
      });

      await t.step("⑦ ChatSession の 2 ターンが⑥と同じ会話になる", async () => {
        const { fixture: name, maxNewTokens, expected } = CASES[0];
        const first = caseOf(name).messages;
        // このケースは user 1 発話（セッションは文字列で受けるので、system も過去 turn も無い）。
        assertEquals(first.length, 1, `${name}: 1 発話のケースでないと send に写せない`);
        assert(followUpAnswer !== undefined, "⑥ が 2 ターン目の参照を採っていない");

        await using session = new Gemma4ChatSession(pipeline, { maxNewTokens });
        // 1 ターン目は golden そのもの（`chat()` と同じ全体描画から始まる）。
        assertEquals(await session.send(first[0].content).text(), expected, "1 ターン目");
        // 2 ターン目は差分描画 + KV の継続。セッションが継ぐ条件（前ターンが EOS で閉じた）と
        // 履歴の積み方が壊れれば、ここで⑥の参照から離れる。
        const second = session.send(FOLLOW_UP.content, { maxNewTokens: FOLLOW_UP_TOKENS });
        assertEquals(await second.text(), followUpAnswer, "2 ターン目（⑥ の参照と逐語一致）");
        assertEquals((await second.done).reason, "eos", "2 ターン目の停止理由");
        assertEquals(session.turns, [
          first[0],
          { role: "assistant", content: expected },
          FOLLOW_UP,
          { role: "assistant", content: followUpAnswer },
        ], "履歴は「流した本文」だけで積まれる");
      });

      // async generator の本体は最初の `next()` まで走らないので、発行時の検査だけでは
      // 「発行 → dispose → 汲み始める」が抜ける。汲まないまま dispose を跨がせる。
      issuedBeforeDispose = pipeline.chat([{ role: "user", content: "hi" }], { maxNewTokens: 4 });
    } finally {
      await pipeline.dispose();
    }

    await t.step("dispose 済みのパイプラインは生成を受けない", async () => {
      assertThrows(
        () => pipeline.chat([{ role: "user", content: "hi" }], { maxNewTokens: 4 }),
        Error,
        "dispose 済み",
      );
      await assertRejects(() => pipeline.sequence(), Error, "dispose 済み");

      // 発行済み・未反復の stream も**発行時と同じ pipeline の文言**で閉じる（ランタイムの
      // Session 文言に化けると、呼び手には真因＝自分が dispose したことが読み取れない）。
      const issued = issuedBeforeDispose;
      assert(issued !== undefined, "dispose の前に stream を発行していない");
      await assertRejects(
        async () => {
          for await (const _chunk of issued) { /* 1 個も来ない */ }
        },
        Error,
        "Gemma4Pipeline: dispose 済みでは生成できない",
      );
    });

    await t.step("公開の program 面は凍結された数だけで、dispose 後も読める", () => {
      // NOTE: PLE sidecar のホストキャッシュが dispose で返ることは `gemma_ple_test.ts` の
      // 単体門が持つ（公開面から `derivedInputs.derive` は引けない = 面を絞った意図どおり）。
      // ここで見るのは絞った後の面の性質 — 配布形が宣言した数がそのまま読め、消費者側の
      // 書き込みが生成ループの停止集合へ届かないこと。
      const program = pipeline.program;
      assertEquals(program.chunkLength, CHUNK_LENGTH);
      assertEquals(program.capacity, CAPACITY);
      assertEquals(program.maxPosition, MAX_POSITION);
      assertEquals(Object.isFrozen(program), true, "program が凍結されていない");
      assertEquals(Object.isFrozen(program.stopTokens), true, "stopTokens が凍結されていない");
      // 凍結コピーなので、停止集合を空にしようとしても落ちる（黙って EOS で止まらなくならない）。
      assertThrows(() => {
        (program.stopTokens as number[]).length = 0;
      }, TypeError);
    });
  },
});
