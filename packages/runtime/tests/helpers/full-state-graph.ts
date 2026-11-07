// The small states-form graph of the `fullStateStorage` tests.
//
// MUST: `append_k` / `append_v` are inputs of their own, apart from the `k` / `v` the attention reads.
// That is what lets an f32 Session stand as the expected value of an f16 one: feed it the host-rounded
// rows to append and the unrounded rows to read, and its slot holds what the f16 slot decodes to, with
// the current step read in f32 on both sides.

import type { DeclarationJson } from "./model-fixture.ts";
import { halfScale } from "./state-dispatch.ts";

export const FULL_STATE_WINDOW = 8;

export type FullStateShape = {
  readonly heads: number;
  readonly kvHeads: number;
  readonly depth: number;
  /** Which attentions the graph has: over full slots, over sliding slots, or one over each. */
  readonly slots: "full" | "sliding" | "both";
};

/** The lender's graph: one attention per slot pair, with its two terminal appends. */
export const fullStateGraph = ({
  heads,
  kvHeads,
  depth,
  slots,
}: FullStateShape): DeclarationJson => {
  const kvRows = (name: string) => ({ name, dtype: "f32", shape: [1, kvHeads, "M", depth] });
  const out = { dtype: "f32", shape: [1, heads, "M", depth] };
  const slot = { dtype: "f32", shape: [1, kvHeads, "C", depth] };
  const pairs = [
    ...(slots === "sliding" ? [] : [{ out: "o", k: "kf", v: "vf", attrs: {} }]),
    ...(slots === "full"
      ? []
      : [{ out: "os", k: "ks", v: "vs", attrs: { window: FULL_STATE_WINDOW } }]),
  ];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention", "state_append"] },
    symbols: ["C", "M"],
    inputs: [
      { name: "q", dtype: "f32", shape: [1, heads, "M", depth] },
      kvRows("k"),
      kvRows("v"),
      kvRows("append_k"),
      kvRows("append_v"),
    ],
    outputs: pairs.map((pair) => pair.out),
    initializers: {},
    values: Object.fromEntries(pairs.map((pair) => [pair.out, out])),
    states: Object.fromEntries(
      pairs.flatMap((pair) => [
        [pair.k, slot],
        [pair.v, slot],
      ]),
    ),
    nodes: [
      ...pairs.map((pair) => ({
        op: "attention",
        ins: ["q", "k", "v"],
        outs: [pair.out],
        attrs: { scale: halfScale(depth), ...pair.attrs },
        states: { k: pair.k, v: pair.v },
      })),
      ...pairs.flatMap((pair) => [
        {
          op: "state_append",
          ins: ["append_k"],
          outs: [],
          attrs: pair.attrs,
          states: { slot: pair.k },
        },
        {
          op: "state_append",
          ins: ["append_v"],
          outs: [],
          attrs: pair.attrs,
          states: { slot: pair.v },
        },
      ]),
    ],
  };
};

/** The drafter's graph: readonly attentions over external slots with the lender's names. */
export const fullStateBorrowerGraph = ({
  heads,
  kvHeads,
  depth,
  slots,
}: FullStateShape): DeclarationJson => {
  const out = { dtype: "f32", shape: [1, heads, 1, depth] };
  const slot = { dtype: "f32", shape: [1, kvHeads, "C", depth], external: true };
  const pairs = [
    ...(slots === "sliding" ? [] : [{ out: "o", k: "kf", v: "vf", attrs: {} }]),
    ...(slots === "full"
      ? []
      : [{ out: "os", k: "ks", v: "vs", attrs: { window: FULL_STATE_WINDOW } }]),
  ];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention"] },
    symbols: ["C"],
    inputs: [{ name: "q", dtype: "f32", shape: [1, heads, 1, depth] }],
    outputs: pairs.map((pair) => pair.out),
    initializers: {},
    values: Object.fromEntries(pairs.map((pair) => [pair.out, out])),
    states: Object.fromEntries(
      pairs.flatMap((pair) => [
        [pair.k, slot],
        [pair.v, slot],
      ]),
    ),
    nodes: pairs.map((pair) => ({
      op: "attention",
      ins: ["q"],
      outs: [pair.out],
      attrs: { scale: halfScale(depth), readonly: true, ...pair.attrs },
      states: { k: pair.k, v: pair.v },
    })),
  };
};
