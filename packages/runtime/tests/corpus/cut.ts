/**
 * Cuts a layered declaration down to chosen layers, from the declaration alone.
 *
 * - Layer membership: an initializer belongs to layer n when its name has the segment `layers.<n>.`
 *   (the exporter's module path — a naming convention, not an IR field). A node belongs to the
 *   layer of its own weights, else to the latest layer among its producers; a node with neither is
 *   the prelude (layer -1), which is always kept.
 * - The residual: where kept layer L follows a removed range, the one value L read from that range
 *   (the residual stream, produced by layer L − 1) is rewired to the value the last kept producer P
 *   (a kept layer, or the prelude) handed to layer P + 1 — the value P produced that only layer
 *   P + 1 reads, with the same dtype and shape.
 * - Then nodes that no output and no kept effect needs, and the values, initializers, state slots
 *   and symbols nothing references any more are removed, and `requires.ops` is recomputed.
 *
 * It fails loudly instead of guessing: when the boundary into a kept layer is not a single residual
 * value, when a kept layer reads a value or a state slot of a removed layer, or when a graph output
 * is computed in a removed layer (each message names the layer and the value). The result is the
 * canonical form of the cut and has passed the IR parser.
 */

import { parseDim } from "../../src/format/dims.ts";
import {
  canonicalIrJson,
  type IrDeclaration,
  type IrNode,
  parseIrDeclarationValue,
} from "../../src/format/ir.ts";

/** A cut the declaration does not allow. */
export class CorpusCutError extends Error {
  override readonly name = "CorpusCutError";
}

/** What a cut did (for the authoring tool's report). */
export type CutReport = {
  readonly kept: readonly number[];
  /** `[residual read from the removed range, the value it was rewired to]`, per boundary. */
  readonly rewired: readonly (readonly [string, string])[];
  readonly nodes: number;
  readonly initializers: number;
  readonly states: readonly string[];
};

/** The prelude's layer number (nodes that belong to no layer). */
const PRELUDE = -1;

const LAYER_SEGMENT = /(?:^|\.)layers\.(\d+)\./;

/** The layer an initializer belongs to, by its name (`undefined` for none). */
export const layerOfInitializer = (name: string): number | undefined => {
  const match = LAYER_SEGMENT.exec(name);
  return match === null ? undefined : Number(match[1]);
};

/** The layer of every node, in node order. */
export const nodeLayers = (declaration: IrDeclaration): readonly number[] => {
  const producerLayer = new Map<string, number>();
  const layers: number[] = [];
  declaration.nodes.forEach((node, index) => {
    const own = new Set<number>();
    for (const input of node.ins) {
      if (!Object.hasOwn(declaration.initializers, input)) continue;
      const layer = layerOfInitializer(input);
      if (layer !== undefined) own.add(layer);
    }
    if (own.size > 1) {
      throw new CorpusCutError(
        `node ${index} (${node.op} → ${node.outs.join(", ")}) reads weights of layers ${[...own].join(" and ")}, so it belongs to no single layer`,
      );
    }
    let layer = PRELUDE;
    if (own.size === 1) layer = [...own][0];
    else {
      for (const input of node.ins) layer = Math.max(layer, producerLayer.get(input) ?? PRELUDE);
    }
    for (const out of node.outs) producerLayer.set(out, layer);
    layers.push(layer);
  });
  return layers;
};

/** Cuts `declaration` to the layers in `keep` (any order, no repeats). */
export const cutLayers = (
  declaration: IrDeclaration,
  keep: readonly number[],
): { readonly declaration: IrDeclaration; readonly report: CutReport } => {
  const layers = nodeLayers(declaration);
  const present = new Set(layers.filter((layer) => layer !== PRELUDE));
  const kept = [...new Set(keep)].sort((a, b) => a - b);
  if (kept.length === 0 || kept.length !== keep.length) {
    throw new CorpusCutError(
      `the layers to keep [${keep.join(", ")}] must be non-empty and distinct`,
    );
  }
  const absent = kept.filter((layer) => !present.has(layer));
  if (absent.length > 0) {
    throw new CorpusCutError(
      `the graph has no layer ${absent.join(", ")} (its layers are 0..${Math.max(...present)})`,
    );
  }
  const keptSet = new Set([PRELUDE, ...kept]);
  const producerLayer = new Map<string, number>();
  declaration.nodes.forEach((node, index) => {
    for (const out of node.outs) producerLayer.set(out, layers[index]);
  });
  /** Layers that read each value, and each state slot's writer layers. */
  const readers = new Map<string, Set<number>>();
  const slotWriters = new Map<string, Set<number>>();
  declaration.nodes.forEach((node, index) => {
    for (const input of node.ins) {
      readers.set(input, (readers.get(input) ?? new Set()).add(layers[index]));
    }
    if (node.op === "state_append") {
      for (const slot of Object.values(node.states)) {
        slotWriters.set(slot, (slotWriters.get(slot) ?? new Set()).add(layers[index]));
      }
    }
  });
  const removed = (layer: number | undefined): boolean =>
    layer !== undefined && !keptSet.has(layer);

  // The residual into each kept layer that follows a removed range, and its replacement.
  const rewire = new Map<number, readonly [string, string]>();
  let previous = PRELUDE;
  for (const layer of kept) {
    if (layer !== previous + 1) {
      const fromRange = new Set<string>();
      declaration.nodes.forEach((node, index) => {
        if (layers[index] !== layer) return;
        for (const input of node.ins) {
          const from = producerLayer.get(input);
          if (from !== undefined && from > previous && from < layer) fromRange.add(input);
        }
      });
      const read = [...fromRange];
      if (read.length !== 1 || producerLayer.get(read[0]) !== layer - 1) {
        throw new CorpusCutError(
          `the boundary into layer ${layer} is not a single residual value: it reads [${read
            .map((value) => `${value} (layer ${producerLayer.get(value)})`)
            .join(", ")}] from the removed layers ${previous + 1}..${layer - 1}`,
        );
      }
      const residual = read[0];
      const { dtype, shape } = declaration.values[residual];
      const handed = Object.keys(declaration.values).filter((value) => {
        const by = readers.get(value);
        const info = declaration.values[value];
        return (
          producerLayer.get(value) === previous &&
          by !== undefined &&
          by.has(previous + 1) &&
          [...by].every((reader) => reader <= previous + 1) &&
          info.dtype === dtype &&
          JSON.stringify(info.shape) === JSON.stringify(shape)
        );
      });
      if (handed.length !== 1) {
        throw new CorpusCutError(
          `the boundary out of ${previous === PRELUDE ? "the prelude" : `layer ${previous}`} is not a single residual value: [${handed.join(
            ", ",
          )}] are handed to layer ${previous + 1} with the residual's dtype and shape (${dtype} [${shape.join(",")}])`,
        );
      }
      rewire.set(layer, [residual, handed[0]]);
    }
    previous = layer;
  }

  // What the kept layers read from removed ones, beyond the rewired residuals.
  declaration.nodes.forEach((node, index) => {
    const layer = layers[index];
    if (!keptSet.has(layer)) return;
    const residual = rewire.get(layer)?.[0];
    for (const input of node.ins) {
      const from = producerLayer.get(input);
      if (input !== residual && removed(from)) {
        throw new CorpusCutError(
          `layer ${layer} reads value '${input}' of removed layer ${from}: keep layer ${from} too, or cut elsewhere`,
        );
      }
    }
    if (node.op === "state_append") return;
    for (const slot of Object.values(node.states)) {
      for (const writer of slotWriters.get(slot) ?? []) {
        if (removed(writer)) {
          throw new CorpusCutError(
            `layer ${layer} reads state '${slot}' of removed layer ${writer}: keep layer ${writer} too, or cut elsewhere`,
          );
        }
      }
    }
  });
  for (const output of declaration.outputs) {
    const from = producerLayer.get(output);
    if (removed(from)) {
      throw new CorpusCutError(
        `graph output '${output}' is computed in removed layer ${from}: the cut must keep that layer`,
      );
    }
  }

  // Keep the prelude and the kept layers, rewired; then drop what no output or kept effect needs.
  const candidates: IrNode[] = [];
  declaration.nodes.forEach((node, index) => {
    const layer = layers[index];
    if (!keptSet.has(layer)) return;
    const pair = rewire.get(layer);
    candidates.push(
      pair === undefined
        ? node
        : { ...node, ins: node.ins.map((input) => (input === pair[0] ? pair[1] : input)) },
    );
  });
  const live = new Set<string>(declaration.outputs);
  const nodes: IrNode[] = [];
  for (let index = candidates.length - 1; index >= 0; index -= 1) {
    const node = candidates[index];
    // A node without outputs is an effect (a state write): it stays with its layer.
    if (node.outs.length === 0 || node.outs.some((out) => live.has(out))) {
      nodes.push(node);
      for (const input of node.ins) live.add(input);
    }
  }
  nodes.reverse();

  const referenced = new Set<string>();
  const slots = new Set<string>();
  for (const node of nodes) {
    for (const name of [...node.ins, ...node.outs]) referenced.add(name);
    for (const slot of Object.values(node.states)) slots.add(slot);
  }
  const initializers = Object.fromEntries(
    Object.entries(declaration.initializers).filter(([name]) => referenced.has(name)),
  );
  const values = Object.fromEntries(
    Object.entries(declaration.values).filter(([name]) => referenced.has(name)),
  );
  const states = Object.fromEntries(
    Object.entries(declaration.states).filter(([name]) => slots.has(name)),
  );
  const bindable = new Set<string>();
  for (const shape of [
    ...declaration.inputs.map((input) => input.shape),
    ...Object.values(states).map((slot) => slot.shape),
  ]) {
    for (const dim of shape) if (typeof dim === "string") bindable.add(parseDim(dim).sym);
  }
  const candidate: IrDeclaration = {
    ...declaration,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: declaration.symbols.filter((symbol) => bindable.has(symbol)),
    initializers,
    values,
    states,
    nodes,
  };
  // The canonical text through the parser: the cut is checked like any graph a container carries.
  const parsed = parseIrDeclarationValue(JSON.parse(canonicalIrJson(candidate)));
  return {
    declaration: parsed,
    report: {
      kept,
      rewired: [...rewire.values()],
      nodes: parsed.nodes.length,
      initializers: Object.keys(parsed.initializers).length,
      states: Object.keys(parsed.states),
    },
  };
};
