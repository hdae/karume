import type { DeclarationJson } from "./model-fixture.ts";

export type RmsNormAddGraphOptions = {
  rows?: number;
  dim?: number;
  normFirst?: boolean;
  normOutput?: boolean;
  extraNormConsumer?: boolean;
  interpose?: boolean;
  broadcast?: boolean;
  sharedResidual?: boolean;
};

/** 内部値の公開・別consumer・broadcastで融合の適格性だけを変える。 */
export const rmsNormAddGraph = (options: RmsNormAddGraphOptions = {}): DeclarationJson => {
  const dim = options.dim ?? 1536, shape = [options.rows ?? 4, dim];
  const n = options.interpose ? "alias" : "n";
  const residual = options.sharedResidual ? "x" : "r";
  const inputs: DeclarationJson["inputs"] = [
    { name: "x", dtype: "f32", shape },
    { name: "w", dtype: "f32", shape: [dim] },
  ];
  if (!options.sharedResidual) {
    inputs.push({ name: "r", dtype: "f32", shape: options.broadcast ? [1, dim] : shape });
  }
  const values: DeclarationJson["values"] = {
    n: { dtype: "f32", shape },
    y: { dtype: "f32", shape },
    copy: { dtype: "f32", shape },
  };
  const nodes: DeclarationJson["nodes"] = [
    { op: "rms_norm", ins: ["x", "w"], outs: ["n"], attrs: { eps: 1e-6 } },
  ];
  if (options.interpose) {
    values.alias = { dtype: "f32", shape };
    nodes.push({ op: "reshape", ins: ["n"], outs: ["alias"], attrs: {} });
  }
  nodes.push({
    op: "add",
    ins: options.normFirst ? [n, residual] : [residual, n],
    outs: ["y"],
    attrs: {},
  });
  // 融合後もxを生かす必要がある。外部入力の解放が早すぎればcopyで検出する。
  nodes.push({ op: "neg", ins: ["x"], outs: ["copy"], attrs: {} });
  const outputs = ["y", "copy", ...(options.normOutput ? ["n"] : [])];
  if (options.extraNormConsumer) {
    values.n_copy = { dtype: "f32", shape };
    nodes.push({ op: "neg", ins: ["n"], outs: ["n_copy"], attrs: {} });
    outputs.push("n_copy");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["rms_norm", "add", "neg", ...(options.interpose ? ["reshape"] : [])] },
    symbols: [],
    inputs,
    outputs,
    initializers: {},
    values,
    nodes,
  };
};
