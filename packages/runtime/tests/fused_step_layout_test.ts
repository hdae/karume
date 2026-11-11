// The check of the device-free bookkeeping of an execution step (src/runtime/fused-step-layout.ts):
// the output specs, the release list and, for a fused step, the schedule of its temporaries and
// what each binding points at. The derivation phase replays it into the recipe and the estimate maps
// it to its allocation program, so a slip here makes both agree on the wrong slot table — no GPU
// needed to see it.
//
// Steps are built by hand (no rule involved) so each case pins one rule of the layout, and the
// expected values are written out, not recomputed with the module's own loops.

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { StorageRoles } from "../src/gpu/pipeline-cache.ts";
import { RESHAPE_OP, TOPK_OP } from "../src/ops.ts";
import { resolveOpContract } from "../src/ops/contracts.ts";
import {
  assertExternalOutputWritten,
  assertExternalWrites,
  fusedStepLayout,
  stepOutputs,
  stepReleases,
} from "../src/runtime/fused-step-layout.ts";
import type {
  ExecStep,
  FusedDispatch,
  FusedOperand,
  FusedStep,
  FusedTemp,
} from "../src/runtime/fusion.ts";
import { ExecutionError, type NodePlan } from "../src/runtime/plan.ts";

const dispatch = (key: string, operands?: readonly FusedOperand[]): FusedDispatch => ({
  key,
  wgsl: () => "",
  params: new Uint32Array(4),
  workgroups: { kind: "gridStride", items: 1, size: 64 },
  ...(operands === undefined ? {} : { operands }),
});

const fused = (fields: Partial<FusedStep> & Pick<FusedStep, "dispatches">): FusedStep => ({
  kind: "fused",
  rule: "test/layout",
  ins: ["a", "b"],
  binds: ["a", "b"],
  nodeCount: 2,
  outputName: "y",
  outputShape: [2, 3],
  temps: [],
  ...fields,
});

const temp = (byteLength: number, allocBefore: number, releaseAfter: number): FusedTemp => ({
  byteLength,
  allocBefore,
  releaseAfter,
});

const bind = (index: number): FusedOperand => ({ kind: "bind", index });
const tempOperand = (id: number): FusedOperand => ({ kind: "temp", id });
const OUTPUT: FusedOperand = { kind: "output" };

const roles = (reads: readonly number[], writes: readonly number[]): StorageRoles => ({
  reads: new Set(reads),
  writes: new Set(writes),
});

const nodeStep = (
  op: string,
  ins: readonly string[],
  outputs: NodePlan["outputs"],
  aliasesInput: boolean,
): ExecStep => ({
  kind: "node",
  plan: {
    node: { op, ins, outs: outputs.map((out) => out.name), attrs: {}, states: {} },
    contract: resolveOpContract(op),
    inputShapes: ins.map(() => [6]),
    inputDtypes: ins.map(() => "f32"),
    outputs,
  },
  aliasesInput,
});

describe("fusedStepLayout", () => {
  describe("when allocBefore is not monotone in the declared id", () => {
    // Declared t0 before t1, but t1 is allocated first: t0 at dispatch 1, t1 at dispatch 0, and t2
    // with t0 at dispatch 1.
    const step = fused({
      temps: [temp(100, 1, 2), temp(200, 0, 2), temp(300, 1, 1)],
      dispatches: [
        dispatch("d0", [bind(0), tempOperand(1)]),
        dispatch("d1", [tempOperand(0), tempOperand(2), tempOperand(1)]),
        dispatch("d2", [tempOperand(0), tempOperand(1), OUTPUT]),
      ],
    });
    const layout = fusedStepLayout(step);

    it("lists the temporaries in allocation order, by boundary and then by declared id", () => {
      assertEquals(
        layout.temps.map((t) => t.byteLength),
        [200, 100, 300],
      );
      assertEquals(
        layout.dispatches.map((d) => d.allocate),
        [[0], [1, 2], []],
      );
    });

    it("points each temporary operand at its allocation-order index, not its declared id", () => {
      assertEquals(layout.dispatches[0].operands, [
        { kind: "bind", index: 0, name: "a" },
        { kind: "temp", temp: 0 },
      ]);
      assertEquals(layout.dispatches[1].operands, [
        { kind: "temp", temp: 1 },
        { kind: "temp", temp: 2 },
        { kind: "temp", temp: 0 },
      ]);
    });

    it("releases the temporaries of one boundary in descending declared id", () => {
      // After dispatch 2: declared t1 (allocated first, index 0), then declared t0 (index 1).
      assertEquals(
        layout.dispatches.map((d) => d.release),
        [[], [2], [0, 1]],
      );
    });
  });

  describe("when several temporaries end at the same boundary", () => {
    const step = fused({
      temps: [temp(4, 0, 1), temp(8, 0, 1), temp(12, 1, 1)],
      dispatches: [
        dispatch("d0", [tempOperand(0), tempOperand(1)]),
        dispatch("d1", [tempOperand(0), tempOperand(1), tempOperand(2), OUTPUT]),
      ],
    });

    it("releases them all after that dispatch, the last declared first", () => {
      assertEquals(
        fusedStepLayout(step).dispatches.map((d) => d.release),
        [[], [2, 1, 0]],
      );
    });
  });

  describe("when a dispatch declares no operands", () => {
    const step = fused({ binds: ["a", "b", "c"], dispatches: [dispatch("d0")] });

    it("binds every bind in declaration order, then the output", () => {
      assertEquals(fusedStepLayout(step).dispatches[0].operands, [
        { kind: "bind", index: 0, name: "a" },
        { kind: "bind", index: 1, name: "b" },
        { kind: "bind", index: 2, name: "c" },
        { kind: "output" },
      ]);
    });
  });

  describe("when an operand points at nothing", () => {
    const step = fused({
      temps: [temp(4, 1, 1)],
      dispatches: [
        dispatch("d0", [bind(2), tempOperand(0), tempOperand(7), OUTPUT]),
        dispatch("d1", [tempOperand(0), OUTPUT]),
      ],
    });

    it("returns it unresolved with the reason instead of throwing", () => {
      const layout = fusedStepLayout(step);
      assertEquals(layout.dispatches[0].operands, [
        {
          kind: "unresolved",
          operand: "bind",
          reason: "bind index 2 is outside the 2 declared binds",
        },
        { kind: "unresolved", operand: "temp", reason: "unallocated temporary 0" },
        { kind: "unresolved", operand: "temp", reason: "unallocated temporary 7" },
        { kind: "output" },
      ]);
      // The same temporary resolves once its dispatch allocated it.
      assertEquals(layout.dispatches[1].operands[0], { kind: "temp", temp: 0 });
    });
  });

  describe("when a dispatch binds a weight scale", () => {
    const step = fused({
      dispatches: [dispatch("d0", [{ kind: "weightScale", index: 1 }, OUTPUT])],
    });

    it("names the weight at that bind index", () => {
      assertEquals(fusedStepLayout(step).dispatches[0].operands[0], {
        kind: "weightScale",
        index: 1,
        name: "b",
      });
    });
  });

  describe("when a temporary is released before it is allocated", () => {
    const step = fused({
      temps: [temp(4, 1, 0)],
      dispatches: [dispatch("d0", [OUTPUT]), dispatch("d1", [OUTPUT])],
    });

    it("fails loudly naming the rule and the temporary", () => {
      assertThrows(
        () => fusedStepLayout(step),
        ExecutionError,
        "fusion rule 'test/layout': temporary 0 is released after dispatch 0, before it is allocated",
      );
    });
  });
});

describe("stepOutputs", () => {
  const bookkeeping = {
    useCounts: new Map([
      ["y", 3],
      ["values", 1],
    ]),
    outputNames: new Set(["y", "indices"]),
  };

  describe("for a fused step", () => {
    it("allocates its single output at numel × 4 with the plan's uses and pinned", () => {
      assertEquals(stepOutputs(fused({ dispatches: [dispatch("d0")] }), bookkeeping, "a"), [
        { name: "y", uses: 3, pinned: true, kind: "alloc", byteLength: 24 },
      ]);
    });
  });

  describe("for a plain node with several outputs", () => {
    const step = nodeStep(
      TOPK_OP,
      ["x"],
      [
        { name: "values", shape: [2, 5], dtype: "f32" },
        { name: "indices", shape: [2, 5], dtype: "i32" },
      ],
      false,
    );

    it("lists them in ascending output slot, an unused value with 0 uses", () => {
      assertEquals(stepOutputs(step, bookkeeping, "x"), [
        { name: "values", uses: 1, pinned: false, kind: "alloc", byteLength: 40 },
        { name: "indices", uses: 0, pinned: true, kind: "alloc", byteLength: 40 },
      ]);
    });
  });

  describe("for a plain node that aliases its input", () => {
    const step = nodeStep(RESHAPE_OP, ["x"], [{ name: "y", shape: [2, 3], dtype: "f32" }], true);

    it("makes the output an alias of the given source and allocates nothing", () => {
      assertEquals(stepOutputs(step, bookkeeping, "source of x"), [
        { name: "y", uses: 3, pinned: true, kind: "alias", source: "source of x" },
      ]);
    });
  });
});

describe("stepReleases", () => {
  it("releases a fused step's inputs once per use, not once per bind", () => {
    const step = fused({ ins: ["a", "b", "a"], binds: ["a", "b"], dispatches: [dispatch("d0")] });
    assertEquals(stepReleases(step), ["a", "b", "a"]);
  });

  it("releases a plain node's inputs as the node lists them", () => {
    const step = nodeStep(RESHAPE_OP, ["x", "x"], [{ name: "y", shape: [6], dtype: "f32" }], false);
    assertEquals(stepReleases(step), ["x", "x"]);
  });
});

describe("assertExternalWrites", () => {
  const step = fused({
    temps: [temp(4, 0, 0)],
    dispatches: [dispatch("ext:test:d0", [bind(0), tempOperand(0), OUTPUT])],
  });
  const operands = fusedStepLayout(step).dispatches[0].operands;

  it("accepts writes to the output and the temporaries", () => {
    assertExternalWrites(step, 0, operands, roles([1], [2, 3]));
  });

  it("refuses a write to an input, naming the rule, the binding and the input", () => {
    assertThrows(
      () => assertExternalWrites(step, 0, operands, roles([2], [1, 3])),
      ExecutionError,
      "fusion rule 'test/layout': dispatch 0 ('ext:test:d0') declares binding 1 read_write, but it binds the input 'a'",
    );
  });

  it("refuses a write to the params", () => {
    assertThrows(
      () => assertExternalWrites(step, 0, operands, roles([1, 2], [0, 3])),
      ExecutionError,
      "declares binding 0 read_write, but it binds the params",
    );
  });
});

describe("assertExternalOutputWritten", () => {
  const step = fused({
    temps: [temp(4, 0, 0)],
    dispatches: [
      dispatch("ext:test:d0", [bind(0), tempOperand(0)]),
      dispatch("ext:test:d1", [bind(1), OUTPUT]),
    ],
  });
  const layout = fusedStepLayout(step);

  it("accepts a step whose output some dispatch binds read_write", () => {
    assertExternalOutputWritten(step, layout, [roles([1], [2]), roles([1], [2])]);
  });

  it("refuses a step whose every dispatch binds the output read only", () => {
    // Dispatch 0 writes binding 2, but that is its temporary; dispatch 1 reads the output there.
    assertThrows(
      () => assertExternalOutputWritten(step, layout, [roles([1], [2]), roles([1, 2], [])]),
      ExecutionError,
      "fusion rule 'test/layout': no dispatch declares the binding of the step's output 'y' read_write",
    );
  });
});
