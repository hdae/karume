/**
 * The axis reduce's summation tree, checked symbolically and without a GPU (perf-ledger V-7 — ADR
 * 0035 addendum 2026-10-26). The GPU parity test (tests/gpu_reduce_axis_parity_test.ts) compares
 * bits, which only shows a reordering when the data happen to round differently; a string tree shows
 * every reordering, including a swapped operand order that IEEE commutativity hides numerically.
 *
 * Both trees are built from the **generated WGSL**: the numbers and operand orders that decide the
 * order of evaluation (the identity each fold starts from, leaves per thread, the bit-reversal shift,
 * the carry-stack depth, the combine levels, which side is the left operand, and which lane's slot
 * the output store reads) are read out of `axisReduceWgsl` / `reduceWgsl`, and a small model replays
 * the kernel's loops with them. A generator change that moves the tree therefore fails here, and a
 * generator change that drops one of the read lines fails the read.
 *
 * What the model fixes by hand: it replays one output (`o = 0`). Lanes with `o ≠ 0` run the same
 * loops and differ only in the output index `t` and the base address, which a string tree cannot
 * express; the size-derived invariants in tests/codegen_wgsl_test.ts pin those lines and the GPU
 * parity test (tests/gpu_reduce_axis_parity_test.ts) runs them. Barriers are not modelled either
 * (the barrier-order test in tests/codegen_wgsl_test.ts pins them).
 */

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { axisReduceWgsl, reduceWgsl } from "../src/codegen/reduce.ts";

const SPEC = { op: "sum", dtype: "f32" } as const;
const SPLITS = [1, 2, 4, 8, 16, 32, 64, 128, 256];

/** The first match's numeric captures, or a failure naming the line the generator no longer has. */
const read = (wgsl: string, pattern: RegExp, what: string): string[] => {
  const match = pattern.exec(wgsl);
  assert(match !== null, `generated WGSL has no ${what} (${pattern})`);
  return match.slice(1);
};

/** `a + b` with the operands in the order the WGSL line wrote them. */
type Order = "kept" | "swapped";
const orderOf = (
  left: string,
  right: string,
  first: string,
  second: string,
  what: string,
): Order => {
  if (left === first && right === second) return "kept";
  if (left === second && right === first) return "swapped";
  throw new Error(`generated WGSL: ${what} has unexpected operands ${left} + ${right}`);
};
const add = (order: Order, a: string, b: string): string =>
  order === "kept" ? `(${a}+${b})` : `(${b}+${a})`;

/** 32-bit `reverseBits` (WGSL built-in). */
const reverseBits = (value: number): number => {
  let out = 0;
  for (let bit = 0; bit < 32; bit += 1) out = (out << 1) | ((value >>> bit) & 1);
  return out >>> 0;
};

/** The row reduce's tree (lane fold over `x[i], x[i+256], …`, then the halving-stride tree). */
const rowTree = (axisLen: number): string => {
  const wgsl = reduceWgsl(SPEC);
  const [lanes] = read(wgsl, /@workgroup_size\((\d+)\)/, "workgroup size").map(Number);
  const [laneStride] = read(wgsl, /i = i \+ (\d+)u;/, "lane stride").map(Number);
  const [firstStride] = read(wgsl, /var stride = (\d+)u;/, "first tree stride").map(Number);
  const [foldLeft, foldRight] = read(
    wgsl,
    /acc = (acc|x\[base \+ i\]) \+ (acc|x\[base \+ i\]);/,
    "lane fold",
  );
  const foldOrder = orderOf(foldLeft, foldRight, "acc", "x[base + i]", "lane fold");
  const [treeLeft, treeRight] = read(
    wgsl,
    /scratch\[lid\] = (scratch\[lid(?: \+ stride)?\]) \+ (scratch\[lid(?: \+ stride)?\]);/,
    "tree combine",
  );
  const treeOrder = orderOf(treeLeft, treeRight, "scratch[lid]", "scratch[lid + stride]", "tree");
  const [identity] = read(wgsl, /var acc = (\S+);/, "lane fold identity");
  const scratch: string[] = [];
  for (let lane = 0; lane < lanes; lane += 1) {
    let acc = identity;
    for (let i = lane; i < axisLen; i += laneStride) acc = add(foldOrder, acc, `x${i}`);
    scratch.push(acc);
  }
  for (let stride = firstStride; stride > 0; stride = Math.floor(stride / 2)) {
    for (let lane = 0; lane < stride; lane += 1) {
      scratch[lane] = add(treeOrder, scratch[lane], scratch[lane + stride]);
    }
  }
  return scratch[0];
};

/** One combine level of the split kernel: lanes with `s % modulus == 0` add the lane `offset` away. */
type Level = { readonly modulus: number; readonly offset: number; readonly order: Order };

/** The axis reduce's tree for one output (`o = 0`), replayed from its generated WGSL. */
const axisTreeOf = (wgsl: string, axisLen: number): string => {
  const [lanes] = read(wgsl, /@workgroup_size\((\d+)\)/, "workgroup size").map(Number);
  const [outputs] = read(wgsl, /let s = lid\.x \/ (\d+)u;/, "subtree index").map(Number);
  const [leaves] = read(wgsl, /for \(var j = 0u; j < (\d+)u;/, "leaf loop").map(Number);
  const [base, shift] = read(
    wgsl,
    /let slot = reverseBits\(s \* (\d+)u \+ j\) >> (\d+)u;/,
    "leaf slot",
  ).map(Number);
  const [leafStride] = read(wgsl, /c = c \+ (\d+)u;/, "leaf stride").map(Number);
  const [foldLeft, foldRight] = read(
    wgsl,
    /v = (v|x\[base \+ c \* inner\]) \+ (v|x\[base \+ c \* inner\]);/,
    "leaf fold",
  );
  const foldOrder = orderOf(foldLeft, foldRight, "v", "x[base + c * inner]", "leaf fold");
  const [carryLeft, carryRight] = read(wgsl, /v = (acc\[k\]|v) \+ (acc\[k\]|v);/, "carry");
  const carryOrder = orderOf(carryLeft, carryRight, "acc[k]", "v", "carry");
  const [depth] = read(wgsl, /var acc: array<f32, (\d+)>;/, "carry stack").map(Number);
  const [top] = read(wgsl, /root = acc\[(\d+)u\];/, "subtree root").map(Number);
  const [identity] = read(wgsl, /var v = (\S+);/, "leaf fold identity");
  const levels: Level[] = [];
  const levelPattern =
    /if \(s % (\d+)u == 0u\) \{\n\s*roots\[lid\.x\] = (roots\[lid\.x(?: \+ \d+u)?\]) \+ (roots\[lid\.x(?: \+ \d+u)?\]);/g;
  for (const match of wgsl.matchAll(levelPattern)) {
    const lanesAway = (operand: string): number => Number(/\+ (\d+)u/.exec(operand)?.[1] ?? 0);
    const [left, right] = [lanesAway(match[2]), lanesAway(match[3])];
    assert(left === 0 || right === 0, `level reads two remote lanes: ${match[0]}`);
    const offset = left + right;
    assertEquals(offset % outputs, 0, `level offset ${offset} is not a whole subtree`);
    levels.push({
      modulus: Number(match[1]),
      offset: offset / outputs,
      order: left === 0 ? "kept" : "swapped",
    });
  }
  // The one lane per output that stores, and the slot it reads: `roots[o]` is subtree 0's slot,
  // `roots[lid.x]` the storing lane's own.
  const [writer, slot] = read(
    wgsl,
    /if \(s == (\d+)u && t < params\.out_count\) \{\n\s*out\[t\] = roots\[(o|lid\.x)\];/,
    "output store",
  );
  const stored = slot === "o" ? 0 : Number(writer);

  const roots: string[] = [];
  for (let s = 0; s < lanes / outputs; s += 1) {
    const acc: (string | undefined)[] = new Array<string | undefined>(depth).fill(undefined);
    for (let j = 0; j < leaves; j += 1) {
      const slot = reverseBits(s * base + j) >>> shift;
      let v = identity;
      for (let c = slot; c < axisLen; c += leafStride) v = add(foldOrder, v, `x${c}`);
      let k = 0;
      while ((j & (1 << k)) !== 0) {
        v = add(carryOrder, acc[k] ?? "<unwritten>", v);
        k += 1;
      }
      assert(k < depth, `carry stack overflows at j = ${j} (depth ${depth})`);
      acc[k] = v;
    }
    roots.push(acc[top] ?? "<unwritten>");
  }
  for (const { modulus, offset, order } of levels) {
    for (let s = 0; s < roots.length; s += modulus) {
      roots[s] = add(order, roots[s], roots[s + offset] ?? "<out of range>");
    }
  }
  return roots[stored] ?? "<out of range>";
};

const axisTree = (axisLen: number, split: number): string =>
  axisTreeOf(axisReduceWgsl(SPEC, split), axisLen);

describe("axis reduce summation tree", () => {
  it("equals the row reduce's tree for every axis length 1..512 and every split 1..256", () => {
    for (let axisLen = 1; axisLen <= 512; axisLen += 1) {
      const row = rowTree(axisLen);
      for (const split of SPLITS) {
        assertEquals(axisTree(axisLen, split), row, `axis_len ${axisLen}, split ${split}`);
      }
    }
  });

  // Fault injection on the generated text: each fault keeps the WGSL readable by the model, so the
  // test above would report it as a tree mismatch rather than a missing line.
  const FAULTS: readonly (readonly [string, (wgsl: string) => string])[] = [
    // a misaligned leaf range: subtree s starts at 8·s instead of 16·s (ranges overlap)
    [
      "shifted range",
      (wgsl) => wgsl.replace("reverseBits(s * 16u + j)", "reverseBits(s * 8u + j)"),
    ],
    // the cross-subtree combine with its operands swapped (equal under commutative addition)
    [
      "swapped level operands",
      (wgsl) =>
        wgsl.replace(
          "roots[lid.x] = roots[lid.x] + roots[lid.x + 16u];",
          "roots[lid.x] = roots[lid.x + 16u] + roots[lid.x];",
        ),
    ],
    // the carry stack with its operands swapped
    ["swapped carry operands", (wgsl) => wgsl.replace("v = acc[k] + v;", "v = v + acc[k];")],
    // the leaf fold started from another identity (-0.0 + -0.0 stays -0.0, so the bits can move)
    ["different leaf identity", (wgsl) => wgsl.replace("var v = 0.0;", "var v = -0.0;")],
    // the output stored from another subtree's lane, which holds only a partial root
    [
      "store of a partial root",
      (wgsl) =>
        wgsl.replace(
          "if (s == 0u && t < params.out_count) {\n      out[t] = roots[o];",
          "if (s == 1u && t < params.out_count) {\n      out[t] = roots[lid.x];",
        ),
    ],
  ];

  for (const [name, fault] of FAULTS) {
    it(`tells a ${name} apart from the row reduce's tree`, () => {
      const wgsl = axisReduceWgsl(SPEC, 16);
      const faulty = fault(wgsl);
      assertNotEquals(faulty, wgsl, "the fault did not apply");
      const differs = [2, 17, 255, 256, 300, 512].some(
        (axisLen) => axisTreeOf(faulty, axisLen) !== rowTree(axisLen),
      );
      assert(differs, `${name}: the tree still equals the row reduce's`);
    });
  }
});
