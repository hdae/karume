import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { createLlmTokenizer, type LlmFamily, record } from "./llm-tokenizer.ts";
import { llmProfile, localFileUrl, selectLlmSource } from "./llm-source.ts";
import { checkLlmRequest, type LlmGraph, streamLlm } from "./llm-generate.ts";
import type { GreedySession } from "../../packages/models/src/generation/greedy.ts";
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
    await Deno.mkdir(`${root}/qwen3-06b-i8-2026-09-10-probe`);
    assertEquals(
      await selectLlmSource("qwen3", undefined, undefined, root),
      `${root}/qwen3-06b-i8-2026-09-10-probe`,
    );
    await Deno.mkdir(`${root}/qwen3-06b-gptq-i4-2026-09-10-probe`);
    assertEquals(
      await selectLlmSource("qwen3", undefined, undefined, root),
      `${root}/qwen3-06b-gptq-i4-2026-09-10-probe`,
    );
    assertEquals(
      await selectLlmSource("qwen3", undefined, "i8", root),
      `${root}/qwen3-06b-i8-2026-09-10-probe`,
    );
    assertEquals(await selectLlmSource("qwen3", "/chosen", undefined, root), "/chosen");
    await assertRejects(() => selectLlmSource("qwen3", "/chosen", "i4", root), Error, "排他");
  });
});
Deno.test("LLM source: 複数の同種実験・資産不在・未対応 quant は黙って選ばない", async () => {
  await withDirectory(async (root) => {
    await assertRejects(
      () => selectLlmSource("qwen3", undefined, undefined, root),
      Error,
      "変換済みモデルが",
    );
    await assertRejects(
      () => selectLlmSource("qwen3", undefined, "unknown", root),
      Error,
      "未対応",
    );
    await Deno.mkdir(`${root}/qwen3-06b-i8-2026-09-10-probe`);
    await Deno.mkdir(`${root}/qwen3-06b-i8-2026-09-11-probe`);
    await assertRejects(() => selectLlmSource("qwen3", undefined, undefined, root), Error, "複数");
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
  const tokenizerFile = new URL(`../../${llmProfile(family).tokenizer}`, import.meta.url);
  let available = false;
  try {
    available = Deno.statSync(tokenizerFile).isFile;
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
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
