import type {
  Gemma4ChatSessionHost,
  GenerationRequest,
  GenerationSequence,
  GenerationStream,
} from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import { type GenerateObservation, observeHost } from "../src/models/gemma4/observed-host.ts";

/**
 * A host whose sequences only count: `generate` advances `used` by the prompt plus one generated
 * token and returns an empty stream. Enough to see what the wrapper reports; no GPU.
 */
const fakeHost = () => {
  const disposed: GenerationSequence[] = [];
  const sequence = (capacity: number): GenerationSequence => {
    let used = 0;
    const self: GenerationSequence = {
      capacity,
      get used() {
        return used;
      },
      generate: (request: GenerationRequest): GenerationStream => {
        used += request.prompt.length + 1;
        return {
          [Symbol.asyncIterator]: async function* () {},
          done: Promise.resolve({ reason: "eos", token: 1, tokens: 1 }),
        };
      },
      fork: () => Promise.resolve(sequence(capacity)),
      dispose: () => {
        disposed.push(self);
        return Promise.resolve();
      },
    };
    return self;
  };
  // The tokenizer and the program are opaque to the wrapper (passed through, never read), and the
  // public surface offers no way to build them without a distribution: stand-ins behind a cast.
  const host = {
    tokenizer: "tokenizer",
    program: "program",
    defaultSampler: { temperature: 1, topK: 64, topP: 0.95 },
    sequence: (options?: { capacity?: number }) =>
      Promise.resolve(sequence(options?.capacity ?? 4096)),
  } as unknown as Gemma4ChatSessionHost;
  return { host, disposed };
};

describe("observeHost", () => {
  it("passes the pipeline's surface through unchanged", () => {
    const { host } = fakeHost();
    const observed = observeHost(host, () => {});
    expect(observed.host.tokenizer).toBe(host.tokenizer);
    expect(observed.host.program).toBe(host.program);
    expect(observed.host.defaultSampler).toBe(host.defaultSampler);
  });

  it("reports each generate's prompt length and the positions already held", async () => {
    const { host } = fakeHost();
    const seen: GenerateObservation[] = [];
    const observed = observeHost(host, (observation) => seen.push(observation));
    const sequence = await observed.host.sequence({ capacity: 2048 });
    sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 8 });
    sequence.generate({ prompt: [4, 5], maxNewTokens: 8 });
    expect(seen).toEqual([
      { promptTokens: 3, usedBefore: 0, capacity: 2048 },
      { promptTokens: 2, usedBefore: 4, capacity: 2048 },
    ]);
  });

  it("reads the held sequence's positions live and none once the session disposes it", async () => {
    const { host, disposed } = fakeHost();
    const observed = observeHost(host, () => {});
    expect(observed.kvUsed()).toBeUndefined();
    const sequence = await observed.host.sequence();
    sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 8 });
    expect(observed.kvUsed()).toBe(4);
    await sequence.dispose();
    expect(observed.kvUsed()).toBeUndefined();
    expect(disposed).toHaveLength(1);
  });

  it("keeps tracking the newest sequence when an older one is disposed late", async () => {
    const { host } = fakeHost();
    const observed = observeHost(host, () => {});
    const first = await observed.host.sequence();
    const second = await observed.host.sequence();
    second.generate({ prompt: [1], maxNewTokens: 1 });
    await first.dispose();
    expect(observed.kvUsed()).toBe(2);
  });
});
