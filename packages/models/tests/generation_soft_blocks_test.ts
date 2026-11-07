/**
 * The soft inputs of the generation surface (`GenerationProgramSpec.softInputs` /
 * `GenerationRequest.softBlocks`, ADR 0144), without a GPU.
 *
 * The invariant: **every run of a program with soft inputs binds both**. A run without a block binds
 * zeros and false, and a block reaches exactly one prefill chunk, whole, with the row range the
 * attention needs. Each path that issues a run is driven here (prefill, per-step decode, the greedy
 * hook, speculative verify, the run-ahead) and its inputs are read as the Session receives them.
 */
import { assert, assertEquals, assertThrows } from "@std/assert";
import type { RunInputs, RunOutputs } from "@karume/runtime";
import { describe, it } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import {
  planPrefillChunks,
  planPrefillChunksWithBlocks,
  type PrefillBlock,
} from "../src/generation/greedy.ts";
import type { GenerationGraph, GenerationProgramSpec } from "../src/generation/program.ts";
import {
  createGenerationSequence,
  type GenerationGreedyAhead,
  type GenerationGreedyRun,
  type GenerationSequence,
  type GenerationSequenceOptions,
  type GenerationSession,
} from "../src/generation/sequence.ts";
import type { DraftFace } from "../src/generation/speculation.ts";
import {
  drain,
  type FakeContext,
  fakeArgmax,
  type FakeOptions,
  fakeSession,
  graphOf,
  programOf,
  tokenIds,
  VOCAB,
} from "./helpers/generation-fake.ts";

const EMBEDS = "soft_embeds";
const MASK = "soft_mask";
/** The width of a vector in these tests (any small value: only the positions matter). */
const WIDTH = 3;
const SOFT = { embeds: EMBEDS, mask: MASK, hidden: WIDTH } as const;
const CHUNK = 8;

/** Each row's argmax is the next id, so a turn's tokens are known in advance. */
const SUCCESSOR = Array.from({ length: VOCAB }, (_, id) => (id + 1) % VOCAB);

const softGraph = (): GenerationGraph => {
  const graph = graphOf();
  return {
    ...graph,
    inputs: [
      ...graph.inputs,
      { name: EMBEDS, dtype: "f32", shape: [1, "M", WIDTH] },
      { name: MASK, dtype: "bool", shape: [1, "M", 1] },
    ],
  };
};

/** The soft inputs of one run as the Session got them. */
type SoftView = {
  readonly embeds: readonly number[];
  readonly embedsShape: readonly number[];
  readonly mask: readonly number[];
  readonly maskShape: readonly number[];
  /** The arrays themselves: whether a shape's zeros are reused shows in their identity. */
  readonly embedsData: Float32Array;
  readonly maskData: Uint32Array;
};

const softOf = (inputs: RunInputs): SoftView => {
  assert(Object.hasOwn(inputs, EMBEDS), "the run has no soft_embeds");
  assert(Object.hasOwn(inputs, MASK), "the run has no soft_mask");
  const embeds = inputs[EMBEDS];
  const mask = inputs[MASK];
  assert("data" in embeds && embeds.dtype === "f32", "soft_embeds is not a host f32 tensor");
  assert("data" in mask && mask.dtype === "bool", "soft_mask is not a host bool tensor");
  return {
    embeds: [...embeds.data],
    embedsShape: embeds.shape,
    mask: [...mask.data],
    maskShape: mask.shape,
    embedsData: embeds.data,
    maskData: mask.data,
  };
};

/** One run the Session received. */
type Seen = SoftView & {
  readonly queryLength: number;
  readonly commit: "deferred" | undefined;
  readonly bidirectional: { readonly start: number; readonly end: number } | undefined;
};

const assertZeros = (seen: SoftView, rows: number): void => {
  assertEquals(seen.embedsShape, [1, rows, WIDTH]);
  assertEquals(seen.maskShape, [1, rows, 1]);
  assertEquals(
    seen.embeds,
    Array.from({ length: rows * WIDTH }, () => 0),
  );
  assertEquals(
    seen.mask,
    Array.from({ length: rows }, () => 0),
  );
};

type RigOptions = {
  readonly fake?: FakeOptions;
  readonly program?: Partial<GenerationProgramSpec>;
  readonly sequence?: (
    fake: ReturnType<typeof fakeSession>,
  ) => Partial<GenerationSequenceOptions<FakeContext>>;
};

/** A sequence over a fake whose Session records the soft inputs of every run. */
const open = async (
  options: RigOptions = {},
): Promise<{
  readonly fake: ReturnType<typeof fakeSession>;
  readonly seen: Seen[];
  readonly sequence: GenerationSequence;
}> => {
  const fake = fakeSession({ successor: SUCCESSOR, ...options.fake });
  const seen: Seen[] = [];
  const session: GenerationSession<FakeContext> = {
    createGenerationContext: (spec) => fake.session.createGenerationContext(spec),
    run: (inputs, bindings, generation): Promise<RunOutputs> => {
      seen.push({
        ...softOf(inputs),
        queryLength: generation.queryLength,
        commit: generation.commit,
        bidirectional: generation.bidirectionalRows,
      });
      return fake.session.run(inputs, bindings, generation);
    },
  };
  const sequence = await createGenerationSequence({
    session,
    program: programOf(fake, {
      graph: softGraph(),
      softInputs: SOFT,
      chunkLength: CHUNK,
      derivedInputs: { ...fake.derivedInputs, positionsOnly: true },
      ...options.program,
    }),
    ...options.sequence?.(fake),
  });
  return { fake, seen, sequence };
};

/** `rows` vectors, distinct from each other and from zero. */
const vectors = (rows: number, seed = 1): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: rows * WIDTH }, (_, index) => seed + index);

describe("softInputs: the wiring check", () => {
  const spec = (override: Partial<GenerationProgramSpec>) => {
    const fake = fakeSession();
    return () => programOf(fake, { graph: softGraph(), softInputs: SOFT, ...override });
  };

  it("accepts the two inputs and carries them in the wiring", () => {
    assertEquals(spec({})().softInputs, SOFT);
  });

  it("a graph that declares them and a program that leaves them unwired fails by name", () => {
    assertThrows(spec({ softInputs: undefined }), Error, "soft_embeds / soft_mask are not wired");
  });

  it("refuses a name that is not a graph input", () => {
    assertThrows(
      spec({ softInputs: { ...SOFT, embeds: "nope" } }),
      Error,
      "soft embeds input 'nope'",
    );
  });

  it("refuses a width other than the graph's", () => {
    assertThrows(spec({ softInputs: { ...SOFT, hidden: 4 } }), Error, "[1,M,3]");
    assertThrows(
      spec({ softInputs: { ...SOFT, hidden: 0 } }),
      Error,
      "softInputs.hidden 0 is not a positive integer",
    );
  });

  it("refuses the wrong dtypes", () => {
    const swapped = softGraph();
    const graph: GenerationGraph = {
      ...swapped,
      inputs: swapped.inputs.map((input) =>
        input.name === MASK ? { ...input, dtype: "f32" } : input,
      ),
    };
    assertThrows(spec({ graph }), Error, "has dtype f32 (not bool)");
  });

  it("refuses a row symbol other than the token ids'", () => {
    const base = softGraph();
    const graph: GenerationGraph = {
      ...base,
      symbols: [...base.symbols, "N"],
      inputs: base.inputs.map((input) =>
        input.name === EMBEDS ? { ...input, shape: [1, "N", WIDTH] } : input,
      ),
    };
    assertThrows(spec({ graph }), Error, "not [1,M,3]");
  });

  it("refuses a name another producer claims", () => {
    const fake = fakeSession();
    assertThrows(
      () =>
        programOf(fake, {
          graph: softGraph(),
          softInputs: SOFT,
          residentInputs: [MASK],
        }),
      Error,
      "contain duplicates",
    );
  });
});

describe("planPrefillChunksWithBlocks", () => {
  it("without blocks it is planPrefillChunks exactly", () => {
    for (const promptLength of [1, 2, 7, 8, 9, 16, 17, 100]) {
      for (const chunkLength of [2, 8, 768]) {
        assertEquals(
          planPrefillChunksWithBlocks(promptLength, chunkLength, []),
          planPrefillChunks(promptLength, chunkLength),
        );
      }
    }
  });

  const plan = (promptLength: number, blocks: readonly PrefillBlock[], chunkLength = CHUNK) =>
    planPrefillChunksWithBlocks(promptLength, chunkLength, blocks).map((chunk) => [
      chunk.position,
      chunk.queryLength,
      chunk.block,
    ]);

  it("keeps the text before a block in its chunk when both fit", () => {
    assertEquals(plan(6, [{ start: 2, length: 3 }]), [[0, 6, { index: 0, start: 2, end: 5 }]]);
  });

  it("cuts in front of a block that does not fit behind the text, not earlier", () => {
    assertEquals(plan(12, [{ start: 6, length: 4 }]), [
      [0, 6, undefined],
      [6, 6, { index: 0, start: 0, end: 4 }],
    ]);
  });

  it("does not cut a chunk that the block sits past", () => {
    assertEquals(plan(20, [{ start: 10, length: 3 }]), [
      [0, 8, undefined],
      [8, 8, { index: 0, start: 2, end: 5 }],
      [16, 4, undefined],
    ]);
  });

  it("gives a block that fills the chunk the whole chunk", () => {
    assertEquals(plan(8, [{ start: 0, length: 8 }]), [[0, 8, { index: 0, start: 0, end: 8 }]]);
  });

  it("keeps at most one block per chunk even when two would fit", () => {
    assertEquals(
      plan(8, [
        { start: 1, length: 2 },
        { start: 4, length: 2 },
      ]),
      [
        [0, 4, { index: 0, start: 1, end: 3 }],
        [4, 4, { index: 1, start: 0, end: 2 }],
      ],
    );
  });

  it("two adjacent blocks: the first chunk ends where the next block starts", () => {
    assertEquals(
      plan(6, [
        { start: 0, length: 3 },
        { start: 3, length: 3 },
      ]),
      [
        [0, 3, { index: 0, start: 0, end: 3 }],
        [3, 3, { index: 1, start: 0, end: 3 }],
      ],
    );
  });

  it("covers every row exactly once, whatever the blocks", () => {
    const promptLength = 40;
    for (let start = 0; start + 5 <= promptLength; start += 1) {
      const chunks = planPrefillChunksWithBlocks(promptLength, CHUNK, [{ start, length: 5 }]);
      let position = 0;
      for (const chunk of chunks) {
        assertEquals(chunk.position, position);
        assert(chunk.queryLength >= 1 && chunk.queryLength <= CHUNK, "a chunk's length");
        position += chunk.queryLength;
      }
      assertEquals(position, promptLength);
      assertEquals(chunks.filter((chunk) => chunk.block !== undefined).length, 1);
    }
  });
});

describe("a run without a block binds zeros and false", () => {
  it("prefill and the per-step decode, with the decode shape's arrays reused", async () => {
    const { seen, sequence } = await open();
    const { events } = await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 4 }));
    assertEquals(tokenIds(events), [4, 5, 6, 7]);
    const [prefill, ...decodes] = seen;
    assertZeros(prefill, CHUNK);
    assertEquals(prefill.bidirectional, undefined);
    assertEquals(decodes.length, 3);
    for (const decode of decodes) {
      assertZeros(decode, 1);
      assertEquals(decode.bidirectional, undefined);
      // MUST: one pair per row count: a decode step allocates no zeros of its own.
      assert(decode.embedsData === decodes[0].embedsData, "decode allocated new soft_embeds");
      assert(decode.maskData === decodes[0].maskData, "decode allocated a new soft_mask");
    }
    await sequence.dispose();
  });

  it("the greedy hook gets them too", async () => {
    const steps: SoftView[] = [];
    const { sequence } = await open({
      sequence: (fake) => {
        const greedy: GenerationGreedyRun<FakeContext> = async (inputs, generation) => {
          steps.push(softOf(inputs));
          return fakeArgmax(await fake.session.run(inputs, undefined, generation));
        };
        return { greedy };
      },
    });
    await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 3 }));
    assertEquals(steps.length, 2);
    for (const step of steps) assertZeros(step, 1);
    await sequence.dispose();
  });

  it("every step of the run-ahead, without reading a token", async () => {
    const steps: SoftView[] = [];
    const { sequence } = await open({
      sequence: (fake) => {
        const ahead = fake.greedyAhead(4);
        const greedyAhead: GenerationGreedyAhead<FakeContext> = {
          depth: ahead.depth,
          run: (request) => {
            for (const step of request.steps) steps.push(softOf(step));
            return ahead.run(request);
          },
        };
        return { greedyAhead };
      },
    });
    const { events } = await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 7 }));
    assertEquals(tokenIds(events), [3, 4, 5, 6, 7, 8, 9]);
    assertEquals(steps.length, 6);
    for (const step of steps) {
      assertZeros(step, 1);
      assert(step.embedsData === steps[0].embedsData, "a run-ahead step allocated new zeros");
    }
    await sequence.dispose();
  });

  it("a program with no derived inputs at all", async () => {
    const graph = softGraph();
    const { seen, sequence } = await open({
      program: {
        graph: {
          ...graph,
          inputs: graph.inputs.filter((input) => input.dtype !== "f32" || input.name === EMBEDS),
        },
        derivedInputs: undefined,
      },
    });
    await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 2 }));
    assertZeros(seen[0], CHUNK);
    assertZeros(seen[1], 1);
    await sequence.dispose();
  });

  it("every verify of a speculative turn", async () => {
    const face: DraftFace = {
      steps: 3,
      // Every draft hits the fake target, so each cycle keeps the verify shape.
      draft: ({ token }) =>
        Promise.resolve(Int32Array.of(SUCCESSOR[token], SUCCESSOR[SUCCESSOR[token]], 0)),
      dispose: () => Promise.resolve(),
    };
    const { seen, sequence } = await open({
      program: { chunkBuckets: [4] },
      sequence: () => ({
        speculative: { open: () => Promise.resolve(face), k: 3, policy: "always" },
      }),
    });
    await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 6 }));
    const verifies = seen.filter((run) => run.commit === "deferred");
    assert(verifies.length >= 1, "the turn ran no verify");
    for (const verify of verifies) {
      assertZeros(verify, verify.embedsShape[1]);
      assertEquals(verify.bidirectional, undefined);
    }
    await sequence.dispose();
  });
});

describe("softBlocks: a chunk with a block", () => {
  const rowsOf = (seen: SoftView, row: number): readonly number[] =>
    seen.embeds.slice(row * WIDTH, (row + 1) * WIDTH);

  it("fills the block's rows and mask, passes its range, and leaves the rest zeros and false", async () => {
    const { seen, sequence } = await open();
    const block = vectors(3);
    await drain(
      sequence.generate({
        prompt: [1, 2, 3, 4, 5, 6],
        maxNewTokens: 2,
        softBlocks: [{ start: 1, rows: block }],
      }),
    );
    const [prefill, decode] = seen;
    assertEquals(prefill.queryLength, 6);
    assertEquals(prefill.bidirectional, { start: 1, end: 4 });
    assertEquals(prefill.embedsShape, [1, CHUNK, WIDTH]);
    assertEquals(prefill.maskShape, [1, CHUNK, 1]);
    assertEquals(prefill.mask, [0, 1, 1, 1, 0, 0, 0, 0]);
    assertEquals(prefill.embeds.slice(WIDTH, 4 * WIDTH), [...block]);
    for (const row of [0, 4, 5, 6, 7]) assertEquals(rowsOf(prefill, row), [0, 0, 0]);
    // The run after it is a run without a block.
    assertZeros(decode, 1);
    assertEquals(decode.bidirectional, undefined);
    await sequence.dispose();
  });

  it("reaches only its chunk: the text chunks of the same prompt pass no block", async () => {
    const { seen, sequence } = await open();
    const block = vectors(3);
    await drain(
      sequence.generate({
        prompt: Array.from({ length: 20 }, (_, index) => (index % (VOCAB - 1)) + 1),
        maxNewTokens: 1,
        softBlocks: [{ start: 10, rows: block }],
      }),
    );
    assertEquals(
      seen.map((run) => [run.queryLength, run.bidirectional]),
      [
        [8, undefined],
        [8, { start: 2, end: 5 }],
        [4, undefined],
      ],
    );
    assertZeros(seen[0], CHUNK);
    assertZeros(seen[2], CHUNK);
    assertEquals(seen[1].embeds.slice(2 * WIDTH, 5 * WIDTH), [...block]);
    await sequence.dispose();
  });

  it("gives each block its own chunk, with its own rows", async () => {
    const { seen, sequence } = await open();
    const first = vectors(2, 100);
    const second = vectors(3, 200);
    await drain(
      sequence.generate({
        prompt: [1, 2, 3, 4, 5, 6, 7, 8],
        maxNewTokens: 1,
        softBlocks: [
          { start: 1, rows: first },
          { start: 4, rows: second },
        ],
      }),
    );
    assertEquals(
      seen.map((run) => [run.queryLength, run.bidirectional]),
      [
        [4, { start: 1, end: 3 }],
        [4, { start: 0, end: 3 }],
      ],
    );
    assertEquals(seen[0].embeds.slice(WIDTH, 3 * WIDTH), [...first]);
    assertEquals(seen[1].embeds.slice(0, 3 * WIDTH), [...second]);
    await sequence.dispose();
  });

  it("moves with the prepended pendingToken: start stays an index of the request's prompt", async () => {
    const { fake, seen, sequence } = await open();
    await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 1 }));
    seen.length = 0;
    const block = vectors(2);
    await drain(
      sequence.generate({
        prompt: [5, 6, 7],
        maxNewTokens: 1,
        softBlocks: [{ start: 1, rows: block }],
      }),
    );
    // The frontier (3, the first turn's token) sits in row 0, so rows 1..2 of the prompt are 2..3.
    assertEquals(fake.calls.at(-1)?.ids.slice(0, 4), [3, 5, 6, 7]);
    assertEquals(seen[0].queryLength, 4);
    assertEquals(seen[0].bidirectional, { start: 2, end: 4 });
    assertEquals(seen[0].mask.slice(0, 5), [0, 0, 1, 1, 0]);
    assertEquals(seen[0].embeds.slice(2 * WIDTH, 4 * WIDTH), [...block]);
    await sequence.dispose();
  });

  it("copies the rows at issue", async () => {
    const { seen, sequence } = await open();
    const block = vectors(1);
    const expected = [...block];
    const stream = sequence.generate({
      prompt: [1, 2],
      maxNewTokens: 1,
      softBlocks: [{ start: 0, rows: block }],
    });
    block.fill(-1);
    await drain(stream);
    assertEquals(seen[0].embeds.slice(0, WIDTH), expected);
    await sequence.dispose();
  });

  it("runs after a fork, on the branch's own arrays", async () => {
    const { seen, sequence } = await open();
    await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 1 }));
    const branch = await sequence.fork();
    await drain(
      branch.generate({
        prompt: [4],
        maxNewTokens: 1,
        softBlocks: [{ start: 0, rows: vectors(1) }],
      }),
    );
    assertEquals(seen.at(-1)?.bidirectional, { start: 1, end: 2 });
    await branch.dispose();
    await sequence.dispose();
  });
});

describe("softBlocks: what is refused, synchronously", () => {
  const refuses = async (
    blocks: readonly { readonly start: number; readonly rows: Float32Array<ArrayBuffer> }[],
    fragment: string,
    prompt: readonly number[] = [1, 2, 3, 4],
  ): Promise<void> => {
    const { seen, sequence } = await open();
    assertThrows(
      () => sequence.generate({ prompt, maxNewTokens: 1, softBlocks: blocks }),
      ModelInputError,
      fragment,
    );
    assertEquals(seen, [], "a run was issued");
    await sequence.dispose();
  };

  it("a program without soft inputs", async () => {
    const fake = fakeSession({ successor: SUCCESSOR });
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: programOf(fake),
    });
    assertThrows(
      () =>
        sequence.generate({
          prompt: [1],
          maxNewTokens: 1,
          softBlocks: [{ start: 0, rows: vectors(1) }],
        }),
      ModelInputError,
      "no soft inputs",
    );
    await drain(sequence.generate({ prompt: [1], maxNewTokens: 1, softBlocks: [] }));
    await sequence.dispose();
  });

  it("rows that are not a positive multiple of the hidden size", async () => {
    await refuses([{ start: 0, rows: new Float32Array(WIDTH + 1) }], "positive multiple");
    await refuses([{ start: 0, rows: new Float32Array(0) }], "positive multiple");
  });

  it("a block outside the prompt", async () => {
    await refuses([{ start: 3, rows: vectors(2) }], "outside the prompt");
    await refuses([{ start: -1, rows: vectors(1) }], "is not a row range");
    await refuses([{ start: 0.5, rows: vectors(1) }], "is not a row range");
  });

  it("blocks that are not ascending, or overlap", async () => {
    await refuses(
      [
        { start: 2, rows: vectors(1) },
        { start: 0, rows: vectors(1) },
      ],
      "before the previous block's end",
    );
    await refuses(
      [
        { start: 0, rows: vectors(2) },
        { start: 1, rows: vectors(1) },
      ],
      "before the previous block's end",
    );
  });

  it("a block longer than a chunk", async () => {
    await refuses(
      [{ start: 0, rows: vectors(CHUNK + 1) }],
      "exceed chunkLength",
      Array.from({ length: CHUNK + 1 }, () => 1),
    );
  });
});
