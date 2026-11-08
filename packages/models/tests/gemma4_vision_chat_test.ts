// Image parts in the chat layer (ADR 0144): the rendered ids against upstream's processor, the
// refusals, and how `ChatSession` holds an image over a conversation. No GPU.
//
// The expectations are the `imageChat` / `rejectedImageChat` rows of the 12B fixture, harvested by
// `tools/export-recipes/gemma4/chat.py` from `Gemma4UnifiedProcessor.apply_chat_template` — independent
// of this renderer (ADR 0084 decision 7). An image is stated by its size there: the soft tokens depend
// on nothing else.

import { readTextFile } from "@karume/test-support/fs";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import { ChatSession, type ChatSessionHost } from "../src/gemma/chat-session.ts";
import type { PipelineVisionConfig } from "../src/gemma/config.ts";
import type { GemmaTokenizerAssets } from "../src/gemma/text/asset.ts";
import {
  type ChatContentPart,
  chatImagePrompt,
  chatImageTurn,
  type ChatMessage,
  chatPrompt,
  type ChatTool,
  chatTurn,
  gemma4StopTokens,
} from "../src/gemma/text/chat.ts";
import { Tokenizer } from "../src/gemma/text/tokenizer.ts";
import { imageSoftTokens, type ImageEmbedding } from "../src/gemma/vision.ts";
import type { GenerationProgram } from "../src/generation/program.ts";
import type {
  GenerationEvent,
  GenerationRequest,
  GenerationSequence,
  GenerationStop,
  GenerationStream,
} from "../src/generation/sequence.ts";
import type { Rgb8Image } from "../src/image/preprocess.ts";
import { createBpeModel } from "../src/text/bpe.ts";

/** A part as the fixture states it (the size stands for the pixels). */
type FixturePart =
  | { readonly type: "text"; readonly text: string }
  | { readonly type: "image"; readonly width: number; readonly height: number };

type FixtureMessage = {
  readonly role: string;
  readonly content: string | readonly FixturePart[];
};

type ImageRow = {
  readonly name: string;
  readonly why: string;
  readonly messages: readonly FixtureMessage[];
  readonly rendered: string;
  readonly ids: readonly number[];
};

type RejectedImageRow = {
  readonly name: string;
  readonly why: string;
  readonly messages: readonly FixtureMessage[];
  readonly ids?: readonly number[];
  readonly error?: string;
};

type Fixture = {
  readonly asset: {
    readonly vocab: [number, string][];
    readonly merges: [number, number, number][];
    readonly byteIds: number[];
    readonly addedTokens: [string, number][];
    readonly specialIds: number[];
    readonly unkId: number;
    readonly bosId: number;
    readonly eosId: number;
  };
  readonly visionTokens: Omit<PipelineVisionConfig, "hiddenSize">;
  readonly imageChat: readonly ImageRow[];
  readonly rejectedImageChat: readonly RejectedImageRow[];
};

const fixture = JSON.parse(
  await readTextFile(new URL("fixtures/gemma-text/gemma4-12b-chat.json", import.meta.url)),
) as Fixture;

const assets: GemmaTokenizerAssets = {
  spec: { postProcessor: "none" },
  model: createBpeModel({
    vocab: fixture.asset.vocab.map(([id, token]) => [id, token] as const),
    merges: fixture.asset.merges.map(([left, right, rank]) => [left, right, rank] as const),
    byteIds: fixture.asset.byteIds,
  }),
  addedTokens: new Map(fixture.asset.addedTokens),
  specialIds: new Set(fixture.asset.specialIds),
  unkId: fixture.asset.unkId,
  bosId: fixture.asset.bosId,
  eosId: fixture.asset.eosId,
};
const tokenizer = new Tokenizer(assets, "gemma4-unified");
const STOP_TOKENS = gemma4StopTokens(tokenizer);
const END_OF_TURN = tokenizer.addedTokenId("<turn|>") as number;
/** The rows the embedder returns in these tests (any small width: nothing here reads a vector's size). */
const HIDDEN = 4;
const VISION: PipelineVisionConfig = { ...fixture.visionTokens, hiddenSize: HIDDEN };

const imageOf = (width: number, height: number, fill = 0): Rgb8Image => ({
  data: new Uint8Array(width * height * 3).fill(fill),
  width,
  height,
});

/** A function declaration whose description is `description`. */
const plainTool = (description: string): ChatTool => ({
  type: "function",
  function: { name: "look", description },
});

/** Declarations that spell an image marker where the system block writes the caller's text. */
const SPELLED_TOOLS: readonly (readonly [string, ChatTool])[] = [
  ["the soft token in the description", plainTool("Shows <|image|> here.")],
  ["the opening marker in the description", plainTool("Shows <|image> here.")],
  [
    "the closing marker in a parameter's description",
    {
      type: "function",
      function: {
        name: "look",
        description: "Shows a picture.",
        parameters: {
          type: "object",
          properties: { what: { type: "string", description: "after <image|>" } },
        },
      },
    },
  ],
];

const messagesOf = (messages: readonly FixtureMessage[]): ChatMessage[] =>
  messages.map(
    (message) =>
      ({
        role: message.role,
        content:
          typeof message.content === "string"
            ? message.content
            : message.content.map((part): ChatContentPart =>
                part.type === "text"
                  ? part
                  : { type: "image", image: imageOf(part.width, part.height) },
              ),
      }) as ChatMessage,
  );

/** The runs of the soft token in upstream's ids: `[first row, row count]` each. */
const softRunsOf = (ids: readonly number[]): [number, number][] => {
  const runs: [number, number][] = [];
  ids.forEach((id, index) => {
    if (id !== VISION.imageTokenId) return;
    const last = runs.at(-1);
    if (last !== undefined && last[0] + last[1] === index) last[1] += 1;
    else runs.push([index, 1]);
  });
  return runs;
};

describe("chatImagePrompt: the ids equal upstream's processor", () => {
  for (const row of fixture.imageChat) {
    it(`${row.name} — ${row.why}`, () => {
      const { prompt, images } = chatImagePrompt(tokenizer, messagesOf(row.messages), VISION);
      assertEquals(prompt, [...row.ids]);
      // The positions and counts come from upstream's ids, and the markers sit around each block.
      assertEquals(
        images.map(({ start, softTokens }) => [start, softTokens]),
        softRunsOf(row.ids),
      );
      for (const { start, softTokens } of images) {
        assertEquals(prompt[start - 1], VISION.beginImageTokenId);
        assertEquals(prompt[start + softTokens], VISION.endImageTokenId);
      }
    });
  }

  it("the battery holds images (the cases are not vacuous)", () => {
    assert(fixture.imageChat.some((row) => softRunsOf(row.ids).length === 2));
    assertEquals(VISION.imageTokenId, 258880);
    assertEquals(VISION.beginImageTokenId, 255999);
    assertEquals(VISION.endImageTokenId, 258882);
    assertEquals(VISION.softTokenBudget, 280);
  });

  it("a conversation without an image gives chatPrompt's ids", () => {
    const messages: ChatMessage[] = [{ role: "user", content: "Name a color." }];
    assertEquals(chatImagePrompt(tokenizer, messages, VISION), {
      prompt: chatPrompt(tokenizer, messages),
      images: [],
    });
  });
});

describe("chatImageTurn: the delta of a message with images", () => {
  const whole = fixture.imageChat.find((row) => row.name === "images-in-a-conversation");
  assert(whole !== undefined);
  const messages = messagesOf(whole.messages);

  it("is the tail of the whole render, with the block's start relative to the delta", () => {
    const delta = chatImageTurn(tokenizer, messages[3], VISION);
    assertEquals(delta.prompt, [...whole.ids.slice(-delta.prompt.length)]);
    assertEquals(delta.images.length, 1);
    const [start, softTokens] = softRunsOf(whole.ids).at(-1) as [number, number];
    assertEquals(delta.images[0].start, start - (whole.ids.length - delta.prompt.length));
    assertEquals(delta.images[0].softTokens, softTokens);
  });

  it("a string message gives chatTurn's ids", () => {
    const message: ChatMessage = { role: "user", content: "Name a color." };
    assertEquals(chatImageTurn(tokenizer, message, VISION).prompt, chatTurn(tokenizer, message));
  });

  it("refuses an assistant message, as chatTurn does", () => {
    assertThrows(
      () => chatImageTurn(tokenizer, { role: "assistant", content: "x" }, VISION),
      ModelInputError,
      "role 'assistant' cannot be a delta",
    );
  });
});

describe("what is refused", () => {
  for (const row of fixture.rejectedImageChat) {
    it(`${row.name} — ${row.why}`, () => {
      assert(
        (row.ids === undefined) !== (row.error === undefined),
        "upstream's outcome is recorded",
      );
      assertThrows(
        () => chatImagePrompt(tokenizer, messagesOf(row.messages), VISION),
        ModelInputError,
        "encodes to the image token",
      );
    });
  }

  it("upstream's count check is what stands behind the soft token's refusal", () => {
    const row = fixture.rejectedImageChat.find((entry) => entry.name === "text-part-soft-token");
    assert(row?.error?.includes("should be the same as the number of images"), row?.error);
  });

  const user = (content: unknown): ChatMessage => ({ role: "user", content }) as ChatMessage;

  it("an empty array", () => {
    assertThrows(
      () => chatImagePrompt(tokenizer, [user([])], VISION),
      ModelInputError,
      "is an empty array",
    );
  });

  for (const type of ["audio", "input_audio", "video", "image_url"]) {
    it(`a part of type ${type}, naming the reason`, () => {
      assertThrows(
        () => chatImagePrompt(tokenizer, [user([{ type }])], VISION),
        ModelInputError,
        `'${type}' parts are not supported`,
      );
    });
  }

  it("an unknown part, a bad text and bad pixels", () => {
    assertThrows(
      () => chatImagePrompt(tokenizer, [user([{ type: "x" }])], VISION),
      ModelInputError,
      "is not 'text' or 'image'",
    );
    assertThrows(
      () => chatImagePrompt(tokenizer, [user([{ type: "text" }])], VISION),
      ModelInputError,
      ".text is not a string",
    );
    assertThrows(
      () => chatImagePrompt(tokenizer, [user([{ type: "image", image: {} }])], VISION),
      ModelInputError,
      "is not RGB8 pixels",
    );
    assertThrows(
      () =>
        chatImagePrompt(
          tokenizer,
          [user([{ type: "image", image: { data: new Uint8Array(2), width: 1, height: 1 } }])],
          VISION,
        ),
      ModelInputError,
      "does not match",
    );
    assertThrows(
      () =>
        chatImagePrompt(
          tokenizer,
          [user([{ type: "image", image: imageOf(1, 1), detail: "high" }])],
          VISION,
        ),
      ModelInputError,
      "out-of-scope fields detail",
    );
  });

  it("an array on another role", () => {
    for (const role of ["system", "developer", "assistant", "tool"]) {
      assertThrows(
        () =>
          chatImagePrompt(
            tokenizer,
            [{ role, content: [{ type: "text", text: "x" }] } as ChatMessage],
            VISION,
          ),
        ModelInputError,
        "taken on role 'user' only",
      );
    }
  });

  it("an array in the entries without an image path", () => {
    const parts = user([{ type: "text", text: "x" }]);
    assertThrows(() => chatPrompt(tokenizer, [parts]), ModelInputError, "content parts need");
    assertThrows(() => chatTurn(tokenizer, parts), ModelInputError, "content parts need");
  });

  it("the E2B / E4B dialect", () => {
    const plain = new Tokenizer(assets);
    const message = user([{ type: "text", text: "x" }]);
    assertThrows(
      () => chatImagePrompt(plain, [message], VISION),
      ModelInputError,
      "needs the gemma4-unified chat dialect",
    );
    assertThrows(
      () => chatImageTurn(plain, message, VISION),
      ModelInputError,
      "needs the gemma4-unified chat dialect",
    );
  });

  it("leaves what the model wrote alone: a re-render reproduces it", () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "Spell it." },
      { role: "assistant", content: "It is <|image|>." },
      { role: "user", content: "Thanks." },
    ];
    assertEquals(
      chatImagePrompt(tokenizer, messages, VISION).prompt,
      chatPrompt(tokenizer, messages),
    );
  });

  describe("a spelled marker in a function declaration (the system block renders it)", () => {
    const messages: ChatMessage[] = [{ role: "user", content: "Hi." }];
    for (const [label, tool] of SPELLED_TOOLS) {
      it(label, () => {
        assertThrows(
          () => chatImagePrompt(tokenizer, messages, VISION, { tools: [tool] }),
          ModelInputError,
          "options.tools[0]: the declaration encodes to the image token",
        );
      });
    }

    it("a declaration without one renders as chatPrompt does", () => {
      const options = { tools: [plainTool("Shows a picture.")] };
      assertEquals(
        chatImagePrompt(tokenizer, messages, VISION, options).prompt,
        chatPrompt(tokenizer, messages, options),
      );
    });
  });
});

// ---- ChatSession ----------------------------------------------------------------------------

/** One turn of the fake sequences' script. */
type Answer = { readonly text: string; readonly closes: boolean };

const program = (capacity: number, chunkLength = 768): GenerationProgram =>
  Object.freeze({
    chunkLength,
    maxPosition: 262144,
    capacity,
    vocabSize: 262144,
    stopTokens: Object.freeze([...STOP_TOKENS]),
    chunkBuckets: Object.freeze([]),
  });

/** A host whose sequences answer from a script, and whose embedder counts what it was given. */
const hostOf = (
  answers: readonly Answer[],
  options: {
    readonly capacity?: number;
    readonly chunkLength?: number;
    readonly embedError?: unknown;
    readonly noVision?: boolean;
  } = {},
) => {
  const requests: GenerationRequest[] = [];
  /** What the embedder read of each image: its first byte, at the moment it ran. */
  const embedded: { readonly image: Rgb8Image; readonly firstByte: number }[] = [];
  let turn = 0;
  let forks = 0;
  const makeSequence = (capacity: number, startUsed: number): GenerationSequence => {
    let used = startUsed;
    const sequence: GenerationSequence = {
      capacity,
      get used(): number {
        return used;
      },
      generate(request: GenerationRequest): GenerationStream {
        requests.push(request);
        // As the real sequence: an abort that arrived before the first run settles the turn as aborted.
        const signal = request.signal;
        if (signal?.aborted === true) {
          const aborted = async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
            yield* [];
            throw signal.reason;
          };
          return {
            [Symbol.asyncIterator]: aborted,
            done: Promise.resolve({ reason: "aborted", tokens: 0 }),
          };
        }
        const answer = answers[turn];
        assert(answer !== undefined, `the script has no turn ${turn + 1}`);
        turn += 1;
        const ids = tokenizer.encode(answer.text);
        used += request.prompt.length;
        const stop: GenerationStop = answer.closes
          ? { reason: "eos", token: END_OF_TURN, tokens: ids.length + 1 }
          : { reason: "max-tokens", tokens: ids.length };
        const events = async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
          yield { kind: "prefill", chunk: 1, chunks: 1 };
          for (const id of ids) {
            used += 1;
            yield { kind: "token", id, position: used };
          }
          if (answer.closes) used += 1;
        };
        return { [Symbol.asyncIterator]: events, done: Promise.resolve(stop) };
      },
      fork: () => {
        forks += 1;
        return Promise.resolve(makeSequence(capacity, used));
      },
      dispose: () => Promise.resolve(),
    };
    return sequence;
  };
  const host: ChatSessionHost = {
    tokenizer,
    program: program(options.capacity ?? 4096, options.chunkLength),
    defaultSampler: undefined,
    sequence: (sequenceOptions) =>
      Promise.resolve(
        makeSequence(Number(sequenceOptions?.capacity ?? options.capacity ?? 4096), 0),
      ),
    ...(options.noVision === true
      ? {}
      : {
          vision: VISION,
          embedImage: (image: Rgb8Image): Promise<ImageEmbedding> => {
            if (options.embedError !== undefined) return Promise.reject(options.embedError);
            embedded.push({ image, firstByte: image.data[0] });
            const softTokens = imageSoftTokens(image, VISION.softTokenBudget);
            return Promise.resolve({
              softTokens,
              rows: new Float32Array(softTokens * HIDDEN).fill(embedded.length),
            });
          },
        }),
  };
  return { host, requests, embedded, forks: () => forks };
};

const pixels = (fill: number): ChatContentPart => ({
  type: "image",
  image: imageOf(64, 64, fill),
});
const ask = (fill: number, text = "What is this?"): readonly ChatContentPart[] => [
  pixels(fill),
  { type: "text", text },
];

/** The image object of a history entry. */
const imageIn = (message: ChatMessage | undefined): Rgb8Image => {
  assert(message?.role === "user" && typeof message.content !== "string", "not a parts message");
  const part = message.content.find((item) => item.type === "image");
  assert(part?.type === "image", "no image part");
  return part.image;
};

/** The blocks a `generate` received, against the ids of its prompt. */
const assertBlocksFitPrompt = (request: GenerationRequest): void => {
  for (const { start, rows } of request.softBlocks ?? []) {
    const softTokens = rows.length / HIDDEN;
    assertEquals(request.prompt[start - 1], VISION.beginImageTokenId);
    assertEquals(request.prompt[start], VISION.imageTokenId);
    assertEquals(request.prompt[start + softTokens - 1], VISION.imageTokenId);
    assertEquals(request.prompt[start + softTokens], VISION.endImageTokenId);
  }
};

describe("ChatSession with images", () => {
  it("passes the rows as softBlocks at the block's position, and keeps the message in the history", async () => {
    const rig = hostOf([{ text: "A cat.", closes: true }]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    assertEquals(await session.send(ask(7)).text(), "A cat.");
    assertEquals(rig.embedded.length, 1);
    const [request] = rig.requests;
    assertEquals(request.softBlocks?.length, 1);
    assertBlocksFitPrompt(request);
    assertEquals(
      session.turns.map((turn) => turn.role),
      ["user", "assistant"],
    );
    assertEquals(imageIn(session.turns[0]).width, 64);
  });

  it("the second turn is a delta: no block, no embedding", async () => {
    const rig = hostOf([
      { text: "A cat.", closes: true },
      { text: "Grey.", closes: true },
    ]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    await session.send(ask(7)).text();
    await session.send("What color?").text();
    assertEquals(rig.embedded.length, 1);
    assertEquals(rig.requests[1].softBlocks, undefined);
    assertEquals(
      [...rig.requests[1].prompt],
      chatTurn(tokenizer, { role: "user", content: "What color?" }),
    );
  });

  it("a message with an image in the KV-carrying delta gets its block too", async () => {
    const rig = hostOf([
      { text: "A cat.", closes: true },
      { text: "A dog.", closes: true },
    ]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    await session.send("Hello.").text();
    await session.send(ask(9)).text();
    assertEquals(rig.embedded.length, 1);
    assertEquals(rig.requests[0].softBlocks, undefined);
    assertBlocksFitPrompt(rig.requests[1]);
  });

  it("a whole re-render reuses the rows: the embedder does not run on a replay", async () => {
    const rig = hostOf([
      // A cut-off turn: the next one re-renders the history from the start.
      { text: "A", closes: false },
      { text: "B.", closes: true },
    ]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    await session.send(ask(7)).text();
    await session.send("Go on.").text();
    assertEquals(rig.embedded.length, 1);
    const [first, second] = rig.requests;
    assertBlocksFitPrompt(second);
    assertEquals(second.softBlocks?.[0].start, first.softBlocks?.[0].start);
    // The same rows, and the same soft ids at the same places (the identity of the module doc).
    assert(second.softBlocks?.[0].rows === first.softBlocks?.[0].rows, "the rows were rebuilt");
    assertEquals(
      second.prompt.slice(0, first.prompt.length - 5),
      first.prompt.slice(0, first.prompt.length - 5),
    );
  });

  it("the KV equals the whole render, past the empty thought channel of the 12B dialect", async () => {
    const rig = hostOf([
      { text: "A cat.", closes: true },
      { text: "Grey.", closes: true },
    ]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    await session.send(ask(7)).text();
    await session.send(ask(8, "And this?")).text();
    const kept = session.turns.slice(0, -1);
    const whole = chatImagePrompt(tokenizer, kept, VISION);
    // What a re-render drops: the empty thought channel that opened the past model turn.
    const thought = tokenizer.encode("<|channel>thought\n<channel|>");
    const first = rig.requests[0].prompt;
    const head = first.slice(0, first.length - thought.length);
    const carried = [
      ...head,
      ...tokenizer.encode("A cat."),
      END_OF_TURN,
      ...rig.requests[1].prompt,
    ];
    assertEquals(carried, whole.prompt);
    // Both images' blocks are at the places the whole render puts them.
    assertEquals(rig.requests[0].softBlocks?.[0].start, whole.images[0].start);
    const secondBase = whole.prompt.length - rig.requests[1].prompt.length;
    assertEquals((rig.requests[1].softBlocks?.[0].start ?? 0) + secondBase, whole.images[1].start);
  });

  it("a fork reuses the rows, and hands out pixels of its own", async () => {
    const rig = hostOf([
      { text: "A", closes: false },
      { text: "B.", closes: true },
    ]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    await session.send(ask(7)).text();
    await using branch = await session.fork();
    imageIn(session.turns[0]).data.fill(0);
    assertEquals(imageIn(branch.turns[0]).data[0], 7);
    await branch.send("Go on.").text();
    assertEquals(rig.embedded.length, 1);
    assertBlocksFitPrompt(rig.requests[1]);
  });

  it("copies the pixels at issue, once, and hands out a frozen copy per read", async () => {
    const rig = hostOf([{ text: "A.", closes: true }]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    const part = pixels(7);
    assert(part.type === "image");
    const stream = session.send([part]);
    part.image.data.fill(200);
    await stream.text();
    assertEquals(rig.embedded[0].firstByte, 7);
    assert(rig.embedded[0].image !== part.image, "the session embedded the caller's object");
    const handed = imageIn(session.turns[0]);
    assert(handed.data.buffer !== rig.embedded[0].image.data.buffer, "turns shared the pixels");
    assert(imageIn(session.turns[0]).data.buffer !== handed.data.buffer, "two reads, one copy");
    assertEquals(handed.data, rig.embedded[0].image.data);
    assertEquals(Reflect.set(handed, "width", 1), false);
  });

  it("a write into the handed-out history reaches neither the embedder nor the history", async () => {
    const rig = hostOf([{ text: "A.", closes: true }]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    const stream = session.send(ask(7));
    // Before the stream drains: embedding happens then.
    imageIn(session.turns.at(-1)).data.fill(0);
    await stream.text();
    assertEquals(rig.embedded[0].firstByte, 7);
    imageIn(session.turns[0]).data.fill(0);
    assertEquals(imageIn(session.turns[0]).data[0], 7);
  });

  it("two messages get one embedding each", async () => {
    const rig = hostOf([
      { text: "A", closes: false },
      { text: "B", closes: false },
      { text: "C.", closes: true },
    ]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    await session.send(ask(1)).text();
    await session.send(ask(2)).text();
    assertEquals(rig.embedded.length, 2);
    await session.send("Which?").text();
    assertEquals(rig.embedded.length, 2);
    assertEquals(rig.requests[2].softBlocks?.length, 2);
    assertBlocksFitPrompt(rig.requests[2]);
  });

  it("releases the rows with the dropped turn: the same image object embeds again", async () => {
    const rig = hostOf(
      [
        { text: "A.", closes: true },
        { text: "B.", closes: true },
        { text: "C.", closes: true },
      ],
      { capacity: 768 },
    );
    let dropped: ChatMessage | undefined;
    await using session = new ChatSession(rig.host, {
      maxNewTokens: 8,
      capacity: 768,
      onOverflow: ({ turns }) => {
        // First overflow: drop the exchange holding the image. Second: bring that message back.
        if (dropped === undefined) {
          dropped = turns[0];
          return turns.slice(-1);
        }
        return [dropped, turns.at(-1) as ChatMessage];
      },
    });
    await session.send(ask(5)).text();
    await session.send(ask(6, "a".repeat(250))).text();
    assertEquals(rig.embedded.length, 2);
    await session.send(ask(6, "b".repeat(150))).text();
    // The image of the first message was released with it: rendering it again takes the embedder.
    assertEquals(
      rig.embedded.filter(({ firstByte }) => firstByte === 5).length,
      2,
      "the rows outlived their entry",
    );
  });
});

describe("message equality never takes two images for one", () => {
  /** A deep copy of a message with images, as a policy may write it (other objects, same bytes). */
  const deepCopy = (message: ChatMessage): ChatMessage => {
    assert(message.role === "user" && typeof message.content !== "string");
    return {
      role: "user",
      content: message.content.map((part) =>
        part.type === "image"
          ? { type: "image", image: { ...part.image, data: part.image.data.slice() } }
          : { ...part },
      ),
    };
  };
  /** A session whose second message overflows: `policy` decides what remains. */
  const overflowing = async (
    policy: NonNullable<ConstructorParameters<typeof ChatSession>[1]["onOverflow"]>,
    reply = "A.",
    text = "a".repeat(250),
  ) => {
    const rig = hostOf(
      [
        { text: reply, closes: true },
        { text: "B.", closes: true },
      ],
      { capacity: 768 },
    );
    const session = new ChatSession(rig.host, {
      maxNewTokens: 8,
      capacity: 768,
      onOverflow: policy,
    });
    await session.send(ask(3)).text();
    return { rig, session, stream: session.send(ask(4, text)) };
  };

  it("a tail with other pixels of the same size is a dropped utterance", async () => {
    const { session, stream } = await overflowing(({ turns }) => {
      const asked = deepCopy(turns.at(-1) as ChatMessage);
      imageIn(asked).data.fill(99);
      return [asked];
    });
    await assertRejects(() => stream.text(), Error, "dropped the current utterance");
    await session.dispose();
  });

  it("a tail with the same pixels in a deep copy is the same utterance", async () => {
    const { rig, session, stream } = await overflowing(({ turns }) => [
      deepCopy(turns.at(-1) as ChatMessage),
    ]);
    await stream.text();
    assertEquals(
      rig.embedded.map(({ firstByte }) => firstByte),
      [3, 4],
    );
    await session.dispose();
  });

  it("an older message in a deep copy is a new image to the embedder", async () => {
    // A long reply and a shorter question: dropping the reply alone makes the turn fit.
    const { rig, session, stream } = await overflowing(
      ({ turns }) => [deepCopy(turns[0]), turns.at(-1) as ChatMessage],
      "x".repeat(100),
      "a".repeat(130),
    );
    await stream.text();
    // The copy of the first message missed the rows of the original, by identity.
    assertEquals(
      rig.embedded.map(({ firstByte }) => firstByte),
      [3, 3, 4],
    );
    assertBlocksFitPrompt(rig.requests[1]);
    assertEquals(rig.requests[1].softBlocks?.length, 2);
    await session.dispose();
  });

  it("an older message returned as given keeps its rows", async () => {
    const { rig, session, stream } = await overflowing(
      ({ turns }) => [turns[0], turns.at(-1) as ChatMessage],
      "x".repeat(100),
      "a".repeat(130),
    );
    await stream.text();
    assertEquals(
      rig.embedded.map(({ firstByte }) => firstByte),
      [3, 4],
    );
    await session.dispose();
  });

  describe("a policy that writes into the pixels it was given writes into its own copy", () => {
    it("an older image it edits is a new image: its rows come from the edited pixels", async () => {
      const { rig, session, stream } = await overflowing(
        ({ turns }) => {
          imageIn(turns[0]).data.fill(99);
          return [turns[0], turns.at(-1) as ChatMessage];
        },
        "x".repeat(100),
        "a".repeat(130),
      );
      await stream.text();
      assertEquals(
        rig.embedded.map(({ firstByte }) => firstByte),
        [3, 99, 4],
      );
      assertEquals(imageIn(session.turns[0]).data[0], 99);
      await session.dispose();
    });

    it("the utterance it edits is a dropped utterance, not other pixels for the same one", async () => {
      const { rig, session, stream } = await overflowing(({ turns }) => {
        const asked = turns.at(-1) as ChatMessage;
        imageIn(asked).data.fill(99);
        return [asked];
      });
      await assertRejects(() => stream.text(), Error, "dropped the current utterance");
      assertEquals(
        rig.embedded.map(({ firstByte }) => firstByte),
        [3],
      );
      await session.dispose();
    });

    it("an image of its own is copied, never frozen or shared", async () => {
      const own = imageOf(64, 64, 5);
      const { session, stream } = await overflowing(
        ({ turns }) => [
          { role: "user", content: [{ type: "image", image: own }] },
          turns.at(-1) as ChatMessage,
        ],
        "x".repeat(100),
        "a".repeat(130),
      );
      await stream.text();
      assertEquals(imageIn(session.turns[0]).data[0], 5);
      assertEquals(Object.isFrozen(own), false);
      own.data.fill(1);
      assertEquals(imageIn(session.turns[0]).data[0], 5);
      await session.dispose();
    });
  });
});

describe("what send refuses, before the history moves", () => {
  it("content parts on a host without an embedder, naming it", async () => {
    const rig = hostOf([], { noVision: true });
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    assertThrows(() => session.send(ask(1)), ModelInputError, "vision: true");
    assertEquals(session.turns, []);
  });

  it("a host without an embedder takes a spelled marker as it always did", async () => {
    const rig = hostOf([{ text: "A.", closes: true }], { noVision: true });
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    await session.send("<|image|>").text();
    assertEquals(rig.requests[0].softBlocks, undefined);
  });

  for (const [label, message, fragment] of [
    ["an empty array", [], "empty array"],
    ["an audio part", [{ type: "audio" }], "'audio' parts are not supported"],
    ["a hole", new Array(1), "is not an object"],
    ["a spelled soft token", [{ type: "text", text: "<|image|>" }], "encodes to the image token"],
    ["a spelled marker", [{ type: "text", text: "a<image|>" }], "encodes to the image token"],
    ["a spelled marker in a string", "<|image>", "encodes to the image token"],
    [
      "a soft token split over two text parts",
      [
        { type: "text", text: "look <|ima" },
        { type: "text", text: "ge|> here" },
      ],
      "encodes to the image token",
    ],
    ["a number", 3, "not a string or a content-part array"],
  ] as const) {
    it(label, async () => {
      const rig = hostOf([]);
      await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
      assertThrows(() => session.send(message as unknown as string), ModelInputError, fragment);
      assertEquals(session.turns, []);
      assertEquals(rig.requests, []);
    });
  }

  describe("an image longer than one chunk", () => {
    /** 4800×48: a wide strip, far fewer soft tokens than a square at the same budget. */
    const strip = (fill: number): ChatContentPart => ({
      type: "image",
      image: imageOf(4800, 48, fill),
    });
    const CHUNK = 200;

    it("is refused by send, naming both numbers, before anything is embedded", async () => {
      const rig = hostOf([], { chunkLength: CHUNK });
      await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
      const square = imageSoftTokens(imageOf(64, 64), VISION.softTokenBudget);
      assert(square > CHUNK, `${square} soft tokens fit the chunk`);
      assertThrows(
        () => session.send(ask(1)),
        ModelInputError,
        `becomes ${square} soft tokens, more than chunkLength ${CHUNK}`,
      );
      assertEquals(session.turns, []);
      assertEquals(rig.embedded, []);
    });

    it("an image that fits the chunk is taken (the check is not too wide)", async () => {
      const rig = hostOf([{ text: "A.", closes: true }], { chunkLength: CHUNK });
      await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
      const part = strip(1);
      assert(part.type === "image");
      assert(imageSoftTokens(part.image, VISION.softTokenBudget) <= CHUNK);
      await session.send([part]).text();
      assertEquals(rig.embedded.length, 1);
    });

    it("one an overflow policy brings in is refused before it is embedded", async () => {
      const rig = hostOf(
        [
          { text: "A.", closes: true },
          { text: "B.", closes: true },
        ],
        { capacity: 768, chunkLength: CHUNK },
      );
      await using session = new ChatSession(rig.host, {
        maxNewTokens: 8,
        capacity: 768,
        onOverflow: ({ turns }) => [
          { role: "user", content: [pixels(9)] },
          turns.at(-1) as ChatMessage,
        ],
      });
      await session.send([strip(1), { type: "text", text: "a".repeat(400) }]).text();
      const stream = session.send("b".repeat(300));
      await assertRejects(() => stream.text(), ModelInputError, `more than chunkLength ${CHUNK}`);
      assertEquals(
        rig.embedded.map(({ firstByte }) => firstByte),
        [1],
      );
    });
  });

  it("a failing embedder closes the turn as never having happened", async () => {
    const failure = new Error("embedder lost");
    const rig = hostOf([], { embedError: failure });
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    const stream = session.send(ask(1));
    assertEquals(await assertRejects(() => stream.text()), failure);
    assertEquals(session.turns, []);
    assertEquals(rig.requests, []);
  });

  it("an abort before the image is embedded stops there", async () => {
    const rig = hostOf([]);
    await using session = new ChatSession(rig.host, { maxNewTokens: 8 });
    const controller = new AbortController();
    const reason = new Error("stop");
    controller.abort(reason);
    const stream = session.send(ask(1), { signal: controller.signal });
    assertEquals(await assertRejects(() => stream.text()), reason);
    // The same settlement as a turn without images: done resolves, it does not reject.
    assertEquals((await stream.done).reason, "aborted");
    assertEquals(rig.embedded, []);
  });
});

describe("what the constructor refuses on a host with images, before any turn", () => {
  it("a spelled marker in the system text", () => {
    const rig = hostOf([]);
    assertThrows(
      () => new ChatSession(rig.host, { maxNewTokens: 8, system: "Describe <|image|> well." }),
      ModelInputError,
      "gemma.ChatSession: the text of a 'system' message encodes to the image token",
    );
  });

  for (const [label, tool] of SPELLED_TOOLS) {
    it(`${label} of a declared tool`, () => {
      const rig = hostOf([]);
      assertThrows(
        () => new ChatSession(rig.host, { maxNewTokens: 8, tools: [tool] }),
        ModelInputError,
        "gemma.ChatSession: options.tools[0]: the declaration encodes to the image token",
      );
    });
  }

  it("a system text and declarations without one are taken (the check is not too wide)", async () => {
    const rig = hostOf([{ text: "A.", closes: true }]);
    await using session = new ChatSession(rig.host, {
      maxNewTokens: 8,
      system: "Describe images well.",
      tools: [plainTool("Shows a picture.")],
    });
    await session.send(ask(1)).text();
    assertEquals(rig.requests.length, 1);
  });

  it("a host without an embedder takes them as it always did", async () => {
    const rig = hostOf([{ text: "A.", closes: true }], { noVision: true });
    await using session = new ChatSession(rig.host, {
      maxNewTokens: 8,
      system: "Describe <|image|> well.",
      tools: SPELLED_TOOLS.map(([, tool]) => tool),
    });
    await session.send("Hi.").text();
    assertEquals(rig.requests.length, 1);
  });
});
