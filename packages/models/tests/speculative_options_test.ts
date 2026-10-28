// The accepted set of `Pipeline`'s `speculative` option. It is rejected before any asset is
// read, so this case needs neither the distribution mirror nor a GPU.

import { test } from "vite-plus/test";
import { assertRejects } from "@std/assert";
import { Pipeline } from "../gemma.ts";
import type { Assets } from "../src/gemma/pipeline.ts";
import { GEMMA4_DRAFT_STEPS } from "../src/gemma/speculative.ts";

test("gemma.Pipeline: speculative の k は 1..3・fromAssets では受けない", async () => {
  // 受理集合の門は**取得元へ触る前**（`ref` は解決すらされない）。上は配布形の段数
  // （出口の本数がそのまま上限）、下は「draft を 1 本も採らない投機」= 意味を持たない指定。
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
  // `Assets` に drafter の席が無いので、黙って投機なしで組まずに断る。
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
