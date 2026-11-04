// The accepted set of `Pipeline`'s `speculative` option. It is rejected before any asset is
// read, so this case needs neither the distribution mirror nor a GPU.

import { test } from "vite-plus/test";
import { assertRejects } from "@std/assert";
import { Pipeline } from "../gemma.ts";
import type { Assets } from "../src/gemma/pipeline.ts";
import { GEMMA4_DRAFT_STEPS } from "../src/gemma/speculative.ts";

test("gemma.Pipeline: speculative k is 1..3 and is not accepted by fromAssets", async () => {
  // The accepted-set check runs **before the source is touched** (`ref` is not even resolved). The
  // upper bound is the distribution's step count (the number of exits is the limit as is); the
  // lower bound rules out "speculation that accepts not a single draft" = a setting with no
  // meaning.
  await assertRejects(
    () => Pipeline.fromPretrained("owner/name", { speculative: { k: 0 } }),
    Error,
    `speculative.k 0 is outside 1..${GEMMA4_DRAFT_STEPS}`,
  );
  await assertRejects(
    () =>
      Pipeline.fromPretrained("owner/name", {
        speculative: { k: GEMMA4_DRAFT_STEPS + 1 },
      }),
    Error,
    `speculative.k ${GEMMA4_DRAFT_STEPS + 1} is outside 1..${GEMMA4_DRAFT_STEPS}`,
  );
  // `Assets` has no place for a drafter, so it refuses rather than silently building without
  // speculation.
  const assets: Assets = {
    config: {
      chunkLength: 2,
      maxChunkLength: 2,
      maxPosition: 2,
      capacity: 2,
      rope: {
        sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
        full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
      },
    },
    model: [],
    tokenizer: new Uint8Array(new ArrayBuffer(0)),
  };
  await assertRejects(
    () => Pipeline.fromAssets(assets, { speculative: {} }),
    Error,
    "speculative is not accepted",
  );
});
