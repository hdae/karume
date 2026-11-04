// The prepare phase of the two-stage boundary (ADR 0070 decision 5 / graph-first) — admission
// completes with **only the declarations** of the opened container. Only the layer that needs
// neither a GPU nor weight blocks is checked here (the real-GPU check is
// gpu_prepared_model_test.ts).
//
// Two things to detect:
// ① the estimate comes out "without fetching a single weight block" (= the very reason the
//   two-stage boundary exists: the requirements are known before downloading weights). Shown by
//   counting fetches.
// ② a model that cannot run fails at prepareContainer — shown by the same fetch count that no
//   weight block had been touched when it failed.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { OpContractError } from "../src/ops.ts";
import { RuntimeSupportError } from "../src/ops/support.ts";
import { prepareContainer } from "../src/runtime/executor.ts";
import { buildFixture, countingContainer } from "./helpers/mixed-codec-fixture.ts";
import {
  baseDeclaration,
  baseTensors,
  GRAPH_NAME,
  memoryModel,
  openModelBytes,
} from "./helpers/model-fixture.ts";

test("prepareContainer's estimate comes out without fetching a single weight block", async () => {
  const fixture = buildFixture();
  const counted = countingContainer(await openModelBytes(fixture.declaration, fixture.tensors));
  const prepared = prepareContainer(counted.container, GRAPH_NAME);
  const estimate = prepared.estimate();
  assertEquals(counted.reads(), 0, "the estimate fetches weight blocks");

  // The same numbers from different sources (the krm path / an in-memory container — the estimate
  // depends only on declarations)
  assertEquals(
    estimate,
    prepareContainer(memoryModel(fixture.declaration, fixture.tensors), GRAPH_NAME).estimate(),
  );
  // A guard against tautology: the fixture mixes 4 codecs, so both compressed and uncompressed
  // residency are non-zero (closing off a pass from all fields matching at 0).
  const { weights } = estimate.resident;
  assert(weights.compressedBytes > 0, "compressed residency is 0 (the fixture is broken)");
  assert(weights.uncompressedBytes > 0, "uncompressed residency is 0 (the fixture is broken)");
  assert(
    estimate.peakAccountedBytes > weights.totalBytes,
    "the peak is weight residency alone (the scenario part is not added)",
  );
  // The same no matter how many times it is called on the same PreparedModel (the prepare phase is
  // not consumed)
  assertEquals(prepared.estimate(), estimate);
});

test("prepareContainer rejects missing capabilities and contract violations before fetching weight blocks", async () => {
  // A semantic dtype that cannot run even for a supported op (matmul is f32 only — an i32 weight is
  // listed as a missing capability).
  // NOTE: this cannot be built on the storage codec side — every codec in the ledger can run (bf16
  // was opened by ADR 0131).
  const unsupportedDtype = baseDeclaration();
  unsupportedDtype.values["w"] = { dtype: "i32", shape: [4, 3] };
  const unsupportedCodec = countingContainer(
    await openModelBytes(unsupportedDtype, [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: new Uint8Array(new ArrayBuffer(48)),
        encoding: { codec: "i32" },
      },
      baseTensors()[1],
    ]),
  );
  assertThrows(
    () => prepareContainer(unsupportedCodec.container, GRAPH_NAME),
    RuntimeSupportError,
    "i32",
  );
  assertEquals(unsupportedCodec.reads(), 0, "weight blocks fetched before the capability check");

  const badArity = baseDeclaration();
  // A third input to add (the capability table looks only at slot dtypes, so arity is the contract
  // check's job)
  badArity.nodes[1].ins = ["h", "b", "b"];
  const contractViolation = countingContainer(await openModelBytes(badArity, baseTensors()));
  assertThrows(() => prepareContainer(contractViolation.container, GRAPH_NAME), OpContractError);
  assertEquals(contractViolation.reads(), 0, "weight blocks fetched before the contract check");
});
