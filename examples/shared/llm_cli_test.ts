import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { createLlmTokenizer, type LlmFamily, type LlmTokenizer, record } from "./llm-tokenizer.ts";
import { llmProfile, localFileUrl, selectLlmSource } from "./llm-source.ts";
import {
  checkLlmRequest,
  LlmCapacityError,
  type LlmGraph,
  LlmSequence,
  streamLlm,
} from "./llm-generate.ts";
import type { GreedySession } from "../../packages/models/src/generation/greedy.ts";
import { LlmChat, prepareLlmChat, readLlmLines } from "./llm-chat.ts";
import fixtures from "./fixtures/llm-tokenizer-parity.json" with { type: "json" };
import unicode from "./llm-unicode.json" with { type: "json" };

/** ID 列は fixture の可読性のため空白区切りで保存する。比較は整数列の厳密一致。 */
const parseIds = (text: string): number[] => {
  assert(/^\d+( \d+)*$/.test(text));
  return text.split(" ").map(Number);
};

const withDirectory = async (fn: (root: string) => Promise<void>): Promise<void> => {
  const root = await Deno.makeTempDir();
  try {
    await fn(root);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
};
Deno.test("LLM source: ローカルの GPTQ を優先し、量子化と明示 path も選べる", async () => {
  await withDirectory(async (root) => {
    await Deno.mkdir(`${root}/qwen3-06b-i8`);
    assertEquals(
      await selectLlmSource("qwen3", undefined, undefined, root),
      `${root}/qwen3-06b-i8`,
    );
    await Deno.mkdir(`${root}/qwen3-06b-gptq-i4`);
    assertEquals(
      await selectLlmSource("qwen3", undefined, undefined, root),
      `${root}/qwen3-06b-gptq-i4`,
    );
    assertEquals(
      await selectLlmSource("qwen3", undefined, "i8", root),
      `${root}/qwen3-06b-i8`,
    );
    assertEquals(await selectLlmSource("qwen3", "/chosen", undefined, root), "/chosen");
    await assertRejects(() => selectLlmSource("qwen3", "/chosen", "i4", root), Error, "排他");
  });
});
Deno.test("LLM source: 資産不在・未対応 quant は黙って選ばない", async () => {
  await withDirectory(async (root) => {
    await assertRejects(
      () => selectLlmSource("qwen3", undefined, undefined, root),
      Error,
      "変換済みの系列が",
    );
    await assertRejects(
      () => selectLlmSource("qwen3", undefined, "unknown", root),
      Error,
      "未対応",
    );
  });
});
/**
 * 旧 shard 形の研究記録（`-probe`）は容器ではないので選ばない。2026-09-23 までは正規表現で
 * 拾っていて、デモは `model.krm` の素の `NotFound` で落ちていた（案内も出なかった）。
 */
Deno.test("LLM source: 旧 shard 形の -probe 系列は選ばず、在ることを名指して落ちる", async () => {
  await withDirectory(async (root) => {
    await Deno.mkdir(`${root}/qwen3-06b-gptq-i4-2026-09-10-probe`);
    await Deno.mkdir(`${root}/qwen3-06b-i8-2026-09-10-probe`);
    const error = await assertRejects(
      () => selectLlmSource("qwen3", undefined, undefined, root),
      Error,
      "qwen3-06b-gptq-i4-2026-09-10-probe",
    );
    assertEquals(error.message.includes("qwen3-06b-i8-2026-09-10-probe"), true, error.message);
    assertEquals(error.message.includes("容器ではないので読めません"), true, error.message);
    // 量子化を名指ししても旧形へは落ちない。
    await assertRejects(
      () => selectLlmSource("qwen3", undefined, "i8", root),
      Error,
      "変換済みの系列が",
    );
    // 変換済みが 1 本でも在れば、旧形が同居していてもそちらを選ぶ。
    await Deno.mkdir(`${root}/qwen3-06b-i8`);
    assertEquals(
      await selectLlmSource("qwen3", undefined, undefined, root),
      `${root}/qwen3-06b-i8`,
    );
  });
});
Deno.test("LLM source: path の空白・日本語・%・#・? を保ったまま読める", async () => {
  await withDirectory(async (root) => {
    const path = `${root}/あ %20#?.txt`;
    await Deno.writeTextFile(path, "same file");
    assertEquals(await Deno.readTextFile(localFileUrl(path)), "same file");
  });
});

for (const family of ["qwen3", "minicpm5"] satisfies LlmFamily[]) {
  const tokenizerPath = llmProfile(family).tokenizer;
  const tokenizerFile = new URL(`../../${tokenizerPath}`, import.meta.url);
  let available = false;
  try {
    available = Deno.statSync(tokenizerFile).isFile;
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
  }
  if (!available) {
    console.warn(
      `[karume] ${tokenizerPath} が無いため ${family} の tokenizer パリティを SKIP する` +
        "（生成: 公式モデルの tokenizer.json をこの位置へ手置きする — inputs/ は git 追跡外・" +
        "docs/assets-layout.md）",
    );
  }
  Deno.test({
    name: `${family}: 公式 tokenizer の入力・復号・チャット参照列と厳密一致（ローカル資産）`,
    ignore: !available,
    fn: async () => {
      const raw: unknown = JSON.parse(await Deno.readTextFile(tokenizerFile));
      const tokenizer = createLlmTokenizer(family, raw, unicode);
      for (const test of fixtures[family].cases) {
        assertEquals(
          tokenizer.encode(test.text, true),
          parseIds(test.completion),
          JSON.stringify(test.text),
        );
        const decoder = tokenizer.decoder();
        let decoded = "";
        for (const token of parseIds(test.completion)) decoded += decoder.push(token);
        assertEquals(decoded + decoder.finish(), test.decoded, JSON.stringify(test.text));
      }
      for (const test of fixtures[family].chats) {
        assertEquals(tokenizer.chat(test.prompt, test.system ?? undefined), parseIds(test.ids));
      }
      for (const test of fixtures[family].multiturn) {
        assertEquals(
          tokenizer.chat(test.prompt, test.system ?? undefined, test.turns),
          parseIds(test.ids),
        );
      }
      assertThrows(
        () => tokenizer.chat("<tool_response>unsupported</tool_response>"),
        Error,
        "未対応",
      );
      // 未対応の正規化を既定扱いすると、日本語など一部入力だけの誤値に化ける。
      assertThrows(
        () =>
          createLlmTokenizer(
            family,
            { ...record(raw, "tokenizer"), normalizer: { type: "NFKC" } },
            unicode,
          ),
        Error,
        "normalizer",
      );
    },
  });
}

const graph: LlmGraph = { token: "token", vocabSize: 100, maxPosition: 256 };
Deno.test("LLM request: 未 commit の最終 token は容量を使わず、超過は生成前に拒否する", () => {
  checkLlmRequest(Array(64).fill(1), 65, graph);
  assertThrows(() => checkLlmRequest(Array(64).fill(1), 66, graph), Error, "容量 128");
  assertThrows(() => checkLlmRequest([], 1, graph), Error, "入力 token");
  assertThrows(() => checkLlmRequest([100], 1, graph), Error, "入力 token");
  assertThrows(() => checkLlmRequest([1], 0, graph), Error, "1 以上");
});

type Context = { pastLength: number; dispose(): Promise<void> };
const fakeSession = (responses: readonly number[], failAt = -1): {
  session: GreedySession<Context>;
  runs: { ids: number[]; positions: number[]; queryLength: number }[];
  disposed: () => number;
} => {
  let disposed = 0;
  const runs: { ids: number[]; positions: number[]; queryLength: number }[] = [];
  return {
    runs,
    disposed: () => disposed,
    session: {
      createGenerationContext: () =>
        Promise.resolve({
          pastLength: 0,
          dispose: () => {
            disposed++;
            return Promise.resolve();
          },
        }),
      run: (inputs, _bindings, options) => {
        if (runs.length === failAt) throw new Error("run failure");
        assert("data" in inputs.input_ids && "data" in inputs.position_ids);
        const ids = Array.from(inputs.input_ids.data, Number);
        const positions = Array.from(inputs.position_ids.data, Number);
        const queryLength = options.queryLength;
        const chosen = responses[runs.length];
        runs.push({ ids, positions, queryLength });
        options.context.pastLength += queryLength;
        const data = new Int32Array(ids.length).fill(99);
        data[queryLength - 1] = chosen;
        return Promise.resolve({ token: { dtype: "i32", shape: [1, ids.length, 1], data } });
      },
    },
  };
};
Deno.test("LLM stream: 複数 chunk の位置・pad・最終有効行を使い EOS 後は run しない", async () => {
  const fake = fakeSession([10, 20, 30, 2]);
  const tokens = await Array.fromAsync(streamLlm(fake.session, graph, Array(65).fill(1), 8, [2]));
  assertEquals(tokens, [20, 30, 2]);
  assertEquals(fake.runs.map((run) => run.queryLength), [64, 1, 1, 1]);
  assertEquals(fake.runs[1].ids, [1, ...Array(63).fill(0)]);
  assertEquals(fake.runs[1].positions, [64, ...Array(63).fill(0)]);
  assertEquals(fake.runs[2].ids, [20]);
  assertEquals(fake.runs[2].positions, [65]);
  assertEquals(fake.disposed(), 1);
});
Deno.test("LLM stream: 生成上限・消費者の中断・run の例外でも context を返す", async () => {
  const limited = fakeSession([20, 30]);
  assertEquals(await Array.fromAsync(streamLlm(limited.session, graph, [1], 1, [2])), [20]);
  assertEquals(limited.runs.length, 1);
  assertEquals(limited.disposed(), 1);
  const early = fakeSession([20]);
  for await (const _token of streamLlm(early.session, graph, [1], 8, [2])) break;
  assertEquals(early.disposed(), 1);
  const failed = fakeSession([20], 0);
  await assertRejects(
    () => Array.fromAsync(streamLlm(failed.session, graph, [1], 8, [2])),
    Error,
    "run failure",
  );
  assertEquals(failed.disposed(), 1);
});
Deno.test("LLM stream: AbortSignal で中断した後は追加の decode を行わない", async () => {
  const fake = fakeSession([20, 30]);
  const abort = new AbortController();
  await assertRejects(
    async () => {
      for await (const _token of streamLlm(fake.session, graph, [1], 8, [2], abort.signal)) {
        abort
          .abort();
      }
    },
    DOMException,
    "aborted",
  );
  assertEquals(fake.runs.length, 1);
  assertEquals(fake.disposed(), 1);
});

const testTokenizer = (): LlmTokenizer => ({
  stopTokens: [2],
  encode: (text) => Array.from(text, (char) => char.charCodeAt(0) % 100),
  chat(prompt, system, turns = []): number[] {
    return this.encode(
      (system ?? "") + turns.map((turn) => turn.user + turn.assistant).join("") + prompt,
    );
  },
  decoder: () => ({ push: (token) => String.fromCharCode(token), finish: () => "" }),
});

Deno.test("LLM 多ターン: 同じ prefix は EOS の未 commit 分を含む差分だけ流す", async () => {
  const fake = fakeSession([20, 2, 30, 2]);
  await using sequence = new LlmSequence(fake.session, graph);
  assertEquals(await Array.fromAsync(sequence.stream([1, 3], 8, [2])), [20, 2]);
  assertEquals(fake.disposed(), 0);
  const progress: number[] = [];
  assertEquals(
    await Array.fromAsync(
      sequence.stream(
        [1, 3, 20, 2, 5],
        8,
        [2],
        undefined,
        ({ reusedTokens }) => progress.push(reusedTokens),
      ),
    ),
    [30, 2],
  );
  assertEquals(progress, [3]);
  assertEquals(fake.runs[2].ids.slice(0, 2), [2, 5]);
  assertEquals(fake.runs[2].positions.slice(0, 2), [3, 4]);
  await sequence.reset();
  assertEquals(fake.disposed(), 1);
});

Deno.test("LLM 多ターン: テンプレートが変わった prefix と reset 後は先頭から再計算する", async () => {
  const fake = fakeSession([20, 2, 2, 2]);
  await using sequence = new LlmSequence(fake.session, graph);
  await Array.fromAsync(sequence.stream([1, 3], 8, [2]));
  await Array.fromAsync(sequence.stream([1, 4, 20, 2, 5], 8, [2]));
  assertEquals(fake.disposed(), 1);
  assertEquals(fake.runs[2].positions[0], 0);
  await sequence.reset();
  await Array.fromAsync(sequence.stream([1, 4, 20, 2, 5], 8, [2]));
  assertEquals(fake.runs[3].queryLength, 5);
  assertEquals(fake.runs[3].positions[0], 0);
  await sequence.dispose();
  assertEquals(fake.disposed(), 3);
  await assertRejects(() => Array.fromAsync(sequence.stream([1], 1, [2])), Error, "dispose 済み");
});

Deno.test("LLM 多ターン: 中断や並行操作で走行中の context を別の生成へ渡さない", async () => {
  const fake = fakeSession([20, 2]);
  await using sequence = new LlmSequence(fake.session, graph);
  const stream = sequence.stream([1], 8, [2]);
  await stream.next();
  await assertRejects(() => Array.fromAsync(sequence.stream([1], 1, [2])), Error, "並行");
  await assertRejects(() => sequence.reset(), Error, "生成中");
  await assertRejects(() => sequence.dispose(), Error, "生成中");
  await stream.return(undefined);
  assertEquals(fake.disposed(), 1);
  await Array.fromAsync(sequence.stream([1, 20, 3], 8, [2]));
  assertEquals(fake.runs[1].positions[0], 0);
});

Deno.test("LLM 多ターン: 元の失敗と解放時の失敗を両方残す", async () => {
  const failure = new Error("run failed"), release = new Error("dispose failed");
  const session: GreedySession<Context> = {
    createGenerationContext: () =>
      Promise.resolve({ pastLength: 0, dispose: () => Promise.reject(release) }),
    run: () => Promise.reject(failure),
  };
  await using sequence = new LlmSequence(session, graph);
  const caught = await assertRejects(
    () => Array.fromAsync(sequence.stream([1], 8, [2])),
    SuppressedError,
  );
  assertEquals(caught.error, release);
  assertEquals(caught.suppressed, failure);
});

Deno.test("LLM 多ターン: 容量超過は古い発話の対だけを落とし、収まらない質問では履歴を変えない", async () => {
  const tokenizer = testTokenizer();
  const turns = [{ user: "a".repeat(30), assistant: "b".repeat(30) }, {
    user: "c",
    assistant: "d",
  }];
  const plan = prepareLlmChat(tokenizer, graph, turns, "next", 64, "system");
  assertEquals(plan.turns, [turns[1]]);
  assertEquals(plan.droppedTurns, 1);
  assertEquals(plan.ids, tokenizer.encode("systemcdnext"));
  assertEquals(turns.length, 2);
  assertThrows(
    () => prepareLlmChat(tokenizer, graph, turns, "x".repeat(129), 1, "system"),
    LlmCapacityError,
  );
  const fake = fakeSession([65, 2]);
  await using sequence = new LlmSequence(fake.session, graph);
  const chat = new LlmChat(sequence, tokenizer, graph, 8, "system");
  await chat.send("a", () => {});
  const before = chat.turns;
  await assertRejects(() => chat.send("x".repeat(129), () => {}), LlmCapacityError);
  assertEquals(chat.turns, before);
  assertEquals(fake.runs.length, 2);
});

Deno.test("LLM 多ターン: 部分回答を中断後の履歴に残し、未出力の質問は残さない", async () => {
  const fake = fakeSession([65, 2, 66]);
  await using sequence = new LlmSequence(fake.session, graph);
  const chat = new LlmChat(sequence, testTokenizer(), graph, 8);
  const abort = new AbortController();
  const result = await chat.send("a", () => abort.abort(), { signal: abort.signal });
  assertEquals(result.stop, "aborted");
  assertEquals(result.text, "A");
  assertEquals(chat.turns, [{ user: "a", assistant: "A" }]);
  assertEquals(fake.disposed(), 1);
  const next = await chat.send("b", () => {});
  assertEquals(next.promptTokens, testTokenizer().encode("aAb"));
  assertEquals(next.reusedTokens, 0);
  const before = chat.turns;
  const prefillAbort = new AbortController();
  const empty = await chat.send("c", () => {}, {
    signal: prefillAbort.signal,
    onPrefill: () => prefillAbort.abort(),
  });
  assertEquals(empty.stop, "aborted");
  assertEquals(empty.tokens, []);
  assertEquals(chat.turns, before);
  await chat.reset();
  assertEquals(chat.turns, []);
});

Deno.test("LLM 多ターン: 生成上限で切った回答を次ターンへ渡し、空の EOS も会話として閉じる", async () => {
  const fake = fakeSession([65, 2]);
  await using sequence = new LlmSequence(fake.session, graph);
  const chat = new LlmChat(sequence, testTokenizer(), graph, 1);
  assertEquals((await chat.send("a", () => {})).stop, "length");
  assertEquals(fake.disposed(), 1);
  const result = await chat.send("b", () => {});
  assertEquals(result.promptTokens, testTokenizer().encode("aAb"));
  assertEquals(result.stop, "eos");
  assertEquals(chat.turns, [{ user: "a", assistant: "A" }, { user: "b", assistant: "" }]);
});

Deno.test("LLM 多ターン: 行読みは CRLF・空行・UTF-8 分割・最後の改行なしを保つ", async () => {
  const bytes = new TextEncoder().encode("東京\r\n\nnext");
  const stream = new ReadableStream<Uint8Array<ArrayBuffer>>({
    start(controller): void {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      controller.close();
    },
  });
  assertEquals(await Array.fromAsync(readLlmLines(stream)), ["東京", "", "next"]);
});
