import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { createSampler, type SamplerSpec } from "../src/generation/sampler.ts";
import {
  createGenerationSequence,
  type GenerationGreedyRun,
  type GreedyRunResult,
} from "../src/generation/sequence.ts";
import {
  drain,
  type FakeOptions,
  type FakeSession,
  fakeSession,
  programOf,
} from "./helpers/generation-fake.ts";

const small = (fake: FakeSession) => {
  let calls = 0;
  const greedy: GenerationGreedyRun<FakeSession["context"]> = async (inputs, generation) => {
    calls++;
    const out = await fake.session.run(inputs, undefined, generation),
      x = out.logits;
    assert(x.dtype === "f32");
    let index = 0,
      value = -Infinity;
    for (let i = 0; i < x.data.length; i++) {
      if (Number.isNaN(x.data[i])) return { index: i, value: x.data[i] };
      if (x.data[i] > value) {
        index = i;
        value = x.data[i];
      }
    }
    return { index, value };
  };
  return { greedy, calls: () => calls };
};
test("the greedy capability comes only from a fixed snapshot with temperature 0 and no logit processing", () => {
  for (const spec of [
    {},
    { temperature: 0 },
    { topK: 5, topP: 0.1 },
    {
      repetitionPenalty: 1,
      logitBias: [],
    },
  ]) {
    const s = createSampler(spec);
    assert(s.greedy !== undefined);
    assertEquals(s.greedy(4, 7), 7);
    for (const value of [NaN, Infinity, -Infinity]) assertThrows(() => s.greedy?.(value, 2));
  }
  for (const spec of [
    { temperature: 1 },
    { temperature: 1, topK: 1 },
    { repetitionPenalty: 1.1 },
    {
      logitBias: [[2, 0]],
    },
  ] satisfies SamplerSpec[]) {
    assertEquals(createSampler(spec).greedy, undefined);
  }
  const mutable = { temperature: 0, logitBias: [] as [number, number][] },
    s = createSampler(mutable);
  mutable.temperature = 1;
  mutable.logitBias.push([1, 3]);
  assert(s.greedy !== undefined);
});

test("the small output returns the same tokens and history across multiple chunks, multiple turns and a switch to sampling", async () => {
  const a = fakeSession({ successor: Array.from({ length: 16 }, (_, i) => (i + 1) % 16) });
  const b = fakeSession({ successor: Array.from({ length: 16 }, (_, i) => (i + 1) % 16) });
  const fast = small(b),
    slow = await createGenerationSequence({ session: a.session, program: programOf(a) });
  const quick = await createGenerationSequence({
    session: b.session,
    program: programOf(b),
    greedy: fast.greedy,
  });
  try {
    for (const sampler of [{ temperature: 0 }, { temperature: 1, seed: 17 }, { temperature: 0 }]) {
      const request = { prompt: [1, 2, 3, 4, 5, 6], maxNewTokens: 4, sampler };
      const x = await drain(slow.generate(request)),
        y = await drain(quick.generate(request));
      assertEquals(y, x);
      assertEquals(quick.used, slow.used);
      assertEquals(b.calls, a.calls);
    }
    assertEquals(fast.calls(), 6);
  } finally {
    await quick.dispose();
    await slow.dispose();
  }
});

test("penalty, bias, diagnostics and speculation use the regular run", async () => {
  const noGreedy = (): Promise<GreedyRunResult> => {
    throw new Error("must not call greedy");
  };
  for (const kind of ["penalty", "bias", "diagnostics", "speculative"]) {
    const f = fakeSession({ tokens: [5, 6, 7] });
    const sequence = await createGenerationSequence({
      session: f.session,
      program: programOf(f, { chunkBuckets: [2] }),
      greedy: noGreedy,
      ...(kind === "diagnostics" ? { onRun: () => {} } : {}),
      ...(kind === "speculative"
        ? {
            speculative: {
              policy: "always" as const,
              open: () =>
                Promise.resolve({
                  steps: 1,
                  draft: () => Promise.resolve(Int32Array.of(6)),
                  dispose: () => Promise.resolve(),
                }),
            },
          }
        : {}),
    });
    try {
      await drain(
        sequence.generate({
          prompt: [1, 2],
          maxNewTokens: 2,
          sampler:
            kind === "penalty"
              ? { repetitionPenalty: 1.1 }
              : kind === "bias"
                ? { logitBias: [[3, 1]] }
                : { temperature: 0 },
        }),
      );
    } finally {
      await sequence.dispose();
    }
  }
});

for (const nanAt of [0, 1]) {
  test(`after a small-output NaN failure the already-written frontier is not fed again: ${nanAt}`, async () => {
    const options: FakeOptions = { tokens: [5, 6, 7, 8, 9], nanAt };
    const a = fakeSession(options),
      b = fakeSession(options),
      fast = small(b);
    const slow = await createGenerationSequence({ session: a.session, program: programOf(a) });
    const quick = await createGenerationSequence({
      session: b.session,
      program: programOf(b),
      greedy: fast.greedy,
    });
    try {
      const request = { prompt: [1, 2], maxNewTokens: 3 };
      const x = await assertRejects(() => drain(slow.generate(request)), Error);
      const y = await assertRejects(() => drain(quick.generate(request)), Error);
      assertEquals(y.message, x.message);
      assertEquals(quick.used, slow.used);
      assertEquals(await drain(quick.generate(request)), await drain(slow.generate(request)));
      assertEquals(b.calls, a.calls);
    } finally {
      await quick.dispose();
      await slow.dispose();
    }
  });
}

test("a NaN in an intermediate prefill is not mistaken for the last chunk's sampling", async () => {
  const f = fakeSession({ tokens: [5, 6], nanAt: 0 }),
    fast = small(f);
  const sequence = await createGenerationSequence({
    session: f.session,
    program: programOf(f),
    greedy: fast.greedy,
  });
  try {
    const r = await drain(sequence.generate({ prompt: [1, 2, 3, 4, 5], maxNewTokens: 1 }));
    assertEquals(r.stop, { reason: "max-tokens", tokens: 1 });
    assertEquals(fast.calls(), 0);
  } finally {
    await sequence.dispose();
  }
});

test("the small output also resumes the conversation after an early exit or abort from the same position as the regular run", async () => {
  for (const how of ["break", "abort"]) {
    const results = [];
    for (const optimized of [false, true]) {
      const f = fakeSession({ successor: Array.from({ length: 16 }, (_, i) => (i + 1) % 16) }),
        fast = small(f);
      const sequence = await createGenerationSequence({
        session: f.session,
        program: programOf(f),
        ...(optimized ? { greedy: fast.greedy } : {}),
      });
      const control = new AbortController(),
        reason = Error("stop test");
      try {
        const stream = sequence.generate({
          prompt: [1, 2],
          maxNewTokens: 5,
          signal: control.signal,
        });
        let delivered = 0;
        const run = async () => {
          for await (const e of stream) {
            if (e.kind === "token") {
              if (++delivered < 2) continue;
              if (how === "break") break;
              control.abort(reason);
            }
          }
        };
        if (how === "abort") await assertRejects(run);
        else await run();
        assertEquals(delivered, 2);
        assertEquals(fast.calls(), optimized ? 1 : 0);
        const stopped = await stream.done;
        const resumed = await drain(sequence.generate({ prompt: [3], maxNewTokens: 2 }));
        results.push({ stopped, resumed, calls: f.calls, used: sequence.used });
      } finally {
        await sequence.dispose();
      }
    }
    assertEquals(results[1], results[0]);
  }
});

test("a suppressed set turns the GPU selection off: decode reads the logits and masks them", async () => {
  // The GPU selection returns the raw maximum, which is the suppressed id here (fake logits: 10 at
  // the run's token, 6 at the runner-up).
  const f = fakeSession({ tokens: [5, 5, 5], runnerUp: { id: 6, logit: 6 } });
  const fast = small(f);
  const sequence = await createGenerationSequence({
    session: f.session,
    program: programOf(f, { suppressTokens: [5] }),
    greedy: fast.greedy,
  });
  try {
    const { events } = await drain(
      sequence.generate({ prompt: [1, 2], maxNewTokens: 3, sampler: { temperature: 0 } }),
    );
    assertEquals(
      events.filter((event) => event.kind === "token").map((event) => event.id),
      [6, 6, 6],
    );
    assertEquals(fast.calls(), 0, "the GPU selection ran although tokens are suppressed");
  } finally {
    await sequence.dispose();
  }
});
