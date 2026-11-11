/**
 * The value text of the plain elementwise kernels as the kernel entry hands it out
 * (`elementwiseValueWgsl` of `@karume/runtime/kernel`, ADR 0148): for every op it serves, the value
 * on the operands `v0` / `v1` and the scalars `s0` / `s1` is the very text the plain kernel's WGSL
 * stores to `out[i]`, and the helper functions it returns are the ones that WGSL declares, in its
 * order. The expected text is read from `elementwiseWgsl`'s output, not from a copy, so a change to
 * one kernel's text that the entry does not follow turns this red. The ops served are exactly the
 * unary and binary ops the contract table maps f32 to f32, and a call with another op, or with the
 * wrong number of operands or scalars, throws. No GPU needed.
 */

import { describe, it } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { CodegenError, type ElementwiseValueOp, elementwiseValueWgsl } from "../kernel.ts";
import { elementwiseWgsl } from "../src/codegen/elementwise.ts";
import {
  BINARY_OPS,
  outputDtypeOf,
  resolveOpContract,
  scalarParamCount,
  slotDtypesOf,
  UNARY_OPS,
} from "../src/ops.ts";

/** Every op of the type: the literal fails the type check when it misses one or names another. */
const SERVED = {
  neg: true,
  abs: true,
  exp: true,
  log: true,
  log1p: true,
  sqrt: true,
  sin: true,
  tanh: true,
  sigmoid: true,
  relu: true,
  gelu: true,
  gelu_tanh: true,
  clamp: true,
  clamp_min: true,
  leaky_relu: true,
  add: true,
  sub: true,
  mul: true,
  div: true,
} as const satisfies Record<ElementwiseValueOp, true>;

const isServed = (op: string): op is ElementwiseValueOp => Object.hasOwn(SERVED, op);

const SERVED_OPS: readonly ElementwiseValueOp[] = Object.keys(SERVED).filter(isServed);

/** The operand and scalar names the plain kernel splices (`v<k>` per slot, `s<j>` per scalar). */
const OPERANDS = ["v0", "v1"];
const SCALARS = ["s0", "s1"];

/** The names `op` is called with: one per slot of its contract, one per scalar attr. */
const namesOf = (op: string) => ({
  operands: OPERANDS.slice(0, resolveOpContract(op).arity),
  scalars: SCALARS.slice(0, scalarParamCount(op)),
});

/** The expression the plain kernel stores: the one `out[i] = …;` line of its loop. */
const storedValue = (wgsl: string, where: string): string => {
  const stores = [...wgsl.matchAll(/^ {4}out\[i\] = (.*);$/gm)];
  assertEquals(stores.length, 1, `${where}: one store in the plain kernel`);
  return stores[0][1];
};

/**
 * The module-scope functions the plain kernel declares, as the text it has between its bindings and
 * its entry point (each function followed by an empty line).
 */
const declaredHelpers = (wgsl: string, where: string): string => {
  const outputBinding = wgsl.indexOf("var<storage, read_write> out:");
  const bindingsEnd = wgsl.indexOf("\n\n", outputBinding);
  const entry = wgsl.indexOf("@compute");
  assertEquals(outputBinding >= 0 && bindingsEnd >= 0 && entry > bindingsEnd, true, where);
  return wgsl.slice(bindingsEnd + 2, entry);
};

/** A call with an op name the type does not allow (it is checked at run time as well). */
const callWith = (op: string, operands: readonly string[], scalars: readonly string[]) =>
  Reflect.apply(elementwiseValueWgsl, undefined, [op, operands, scalars]);

describe("elementwiseValueWgsl", () => {
  it("serves exactly the unary and binary ops the contract table maps f32 to f32", () => {
    const f32ToF32 = [...UNARY_OPS, ...BINARY_OPS].filter((op) => {
      const contract = resolveOpContract(op);
      return (
        slotDtypesOf(contract).every((accepted) => accepted.includes("f32")) &&
        outputDtypeOf(contract, 0, "f32", "test") === "f32"
      );
    });
    assertEquals([...SERVED_OPS].sort(), [...f32ToF32].sort());
  });

  it("returns the value the plain kernel stores, on its operand and scalar names, for every op", () => {
    for (const op of SERVED_OPS) {
      const { operands, scalars } = namesOf(op);
      for (const rank of [1, 3]) {
        const where = `${op} (rank ${rank})`;
        const plain = elementwiseWgsl({ op, rank, dtype: "f32" });
        assertEquals(
          elementwiseValueWgsl(op, operands, scalars).value,
          storedValue(plain, where),
          where,
        );
      }
    }
  });

  it("returns the helper functions the plain kernel declares, in its order, for every op", () => {
    const withHelpers: string[] = [];
    for (const op of SERVED_OPS) {
      const { operands, scalars } = namesOf(op);
      const { helpers } = elementwiseValueWgsl(op, operands, scalars);
      const plain = elementwiseWgsl({ op, rank: 2, dtype: "f32" });
      assertEquals(
        helpers.map((helper) => `${helper}\n\n`).join(""),
        declaredHelpers(plain, op),
        op,
      );
      if (helpers.length > 0) withHelpers.push(op);
    }
    // The control: the comparison covered ops that call helpers, not only ones that call none.
    assertEquals(
      ["clamp", "gelu", "gelu_tanh", "log1p", "relu", "sigmoid", "tanh"].every((op) =>
        withHelpers.includes(op),
      ),
      true,
      `ops with helpers: ${withHelpers.join(", ")}`,
    );
  });

  it("splices the operands it is given in slot order, so a node's operand order is kept", () => {
    assertEquals(elementwiseValueWgsl("sub", ["a[i]", "b[0]"]).value, "a[i] - b[0]");
    assertEquals(elementwiseValueWgsl("div", ["b[0]", "a[i]"]).value, "b[0] / a[i]");
  });

  it("throws a CodegenError for the wrong number of operands or scalars", () => {
    for (const op of SERVED_OPS) {
      const { operands, scalars } = namesOf(op);
      const wrong: (readonly [readonly string[], readonly string[]])[] = [
        [[...operands, "v9"], scalars],
        [operands, [...scalars, "s9"]],
        [operands.slice(1), scalars],
        ...(scalars.length > 0 ? [[operands, scalars.slice(1)] as const] : []),
      ];
      for (const [given, givenScalars] of wrong) {
        assertThrows(
          () => elementwiseValueWgsl(op, given, givenScalars),
          CodegenError,
          `op '${op}' takes ${operands.length} operands and ${scalars.length} scalars`,
        );
      }
    }
  });

  it("throws a CodegenError for an op outside the type", () => {
    const outside = [...UNARY_OPS, ...BINARY_OPS].filter((op) => !isServed(op));
    // The control: the elementwise vocabulary has ops the type leaves out (bool outputs or inputs).
    assertEquals(outside.length > 0, true);
    for (const op of [...outside, "where", "cast", "softmax", "unknown"]) {
      assertThrows(
        () => callWith(op, OPERANDS, []),
        CodegenError,
        `op '${op}' has no f32 value expression`,
      );
    }
  });
});
