// IR v1（docs/ir-v1.md）のグラフ JSON 型と構造検証。
// 検証はここに 1 本化する: safetensors との突合は container.ts、次元文法は dims.ts が持ち、
// 本ファイルはグラフ単体で決まる規則（宣言・SSA・トポロジカル順・語彙）だけを見る。

import { isSymbolName, parseDim, tryParseDim } from "./dims.ts";

/** 意味論 dtype。計算は常にこの型で行う（f16/bf16/i8 は格納だけの概念）。 */
export type IrDtype = "f32" | "i32" | "bool";

/**
 * 格納 dtype。f32 / f16 / bf16 / i8 は**意味論 f32 の符号化**で、`i32` だけが生の int32
 * （記号依存定数の焼き込み先 — ADR 0010 の明示的な例外）。実行できるのは f32 / f16 / i8 / i32
 * で（f16 は ADR 0018・i8 は ADR 0019 — どちらも適格な重みスロットは圧縮のまま GPU 常駐・
 * 適格外はロード時に CPU で f32 展開）、bf16 は宣言として受理するだけ。
 */
export type IrStorageDtype = "f32" | "f16" | "bf16" | "i8" | "i32";

/** 非負整数、または `coeff·sym+offset` の正準表記（dims.ts）。 */
export type IrDim = number | string;

export type IrValueInfo = {
  readonly dtype: IrDtype;
  readonly shape: readonly IrDim[];
};

export type IrInput = {
  readonly name: string;
  readonly dtype: IrDtype;
  readonly shape: readonly IrDim[];
};

export type IrStorage = {
  readonly dtype: IrStorageDtype;
  /**
   * 量子化格納の scale テンソルの safetensors キー（`dtype: "i8"` では**必須** — ADR 0019）。
   * 実体は重みと同 rank の keepdim broadcast 形の F32（検証は format/container.ts）。
   */
  readonly scale?: string;
  readonly groupSize?: number;
};

export type IrInitializer = {
  /** safetensors のテンソルキー。 */
  readonly tensor: string;
  readonly storage: IrStorage;
};

export type IrNode = {
  readonly op: string;
  readonly ins: readonly string[];
  readonly outs: readonly string[];
  readonly attrs: Readonly<Record<string, unknown>>;
};

export type IrGraph = {
  readonly format: typeof IR_FORMAT;
  readonly version: typeof IR_VERSION;
  readonly requires: { readonly ops: readonly string[] };
  readonly symbols: readonly string[];
  readonly inputs: readonly IrInput[];
  readonly outputs: readonly string[];
  readonly initializers: Readonly<Record<string, IrInitializer>>;
  readonly values: Readonly<Record<string, IrValueInfo>>;
  readonly nodes: readonly IrNode[];
};

export class IrError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IrError";
  }
}

export const IR_FORMAT = "karume-ir";
export const IR_VERSION = 1;

const TOP_LEVEL_KEYS = [
  "format",
  "version",
  "requires",
  "symbols",
  "inputs",
  "outputs",
  "initializers",
  "values",
  "nodes",
] as const;

/** 意味論 dtype の全語彙（宣言としての受理集合 — 実行可否は契約表 src/ops.ts が持つ）。 */
export const SEMANTIC_DTYPES = ["f32", "i32", "bool"] as const;
const STORAGE_DTYPES = ["f32", "f16", "bf16", "i8", "i32"] as const;

/**
 * initializer の意味論 dtype → 許される格納 dtype（docs/ir-v1.md「値と型」）。
 *
 * MUST: 意味論と格納の組は**この表だけ**が決める。f32 の格納語彙は「f32 値の符号化」で、
 * i32 は生の int32 1 通りのみ（ADR 0010）— 交差を許すと `i32` 宣言の initializer が f16 の
 * ビット列として読まれる沈黙誤値になる。bool の initializer は語彙に無い（実測に無く、
 * safetensors 側の BOOL は 1 バイト格納で 4 バイト前提の転送とも噛み合わない）。
 */
const INITIALIZER_STORAGE: ReadonlyMap<IrDtype, readonly IrStorageDtype[]> = new Map([
  ["f32", ["f32", "f16", "bf16", "i8"]],
  ["i32", ["i32"]],
]);

export const isSemanticDtype = (value: unknown): value is IrDtype =>
  SEMANTIC_DTYPES.some((dtype) => dtype === value);

const isStorageDtype = (value: string): value is IrStorageDtype =>
  STORAGE_DTYPES.some((dtype) => dtype === value);

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const asPlainObject = (value: unknown, where: string): Record<string, unknown> => {
  if (!isPlainObject(value)) throw new IrError(`${where}: オブジェクトでない`);
  return value;
};

const asArray = (value: unknown, where: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new IrError(`${where}: 配列でない`);
  const list: readonly unknown[] = value;
  return list;
};

const asNonEmptyString = (value: unknown, where: string): string => {
  if (typeof value !== "string" || value.length === 0) {
    throw new IrError(`${where}: 空でない文字列でない`);
  }
  return value;
};

/** 重複を許さない文字列配列（集合として扱う欄 — 重複は集合等価の判定を曖昧にする）。 */
const asUniqueStrings = (value: unknown, where: string): string[] => {
  const list = asArray(value, where).map((item, index) =>
    asNonEmptyString(item, `${where}[${index}]`)
  );
  const seen = new Set<string>();
  for (const item of list) {
    if (seen.has(item)) throw new IrError(`${where}: '${item}' が重複している`);
    seen.add(item);
  }
  return list;
};

/** 未リリースにつき前方互換チャネルは持たない — 未知キーは黙って無視せず fail loudly。 */
const checkKeys = (
  obj: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[],
  where: string,
): void => {
  for (const key of required) {
    if (!Object.hasOwn(obj, key)) throw new IrError(`${where}: 必須キー '${key}' が無い`);
  }
  for (const key of Object.keys(obj)) {
    if (!required.includes(key) && !optional.includes(key)) {
      throw new IrError(`${where}: 未知のキー '${key}'`);
    }
  }
};

const asSemanticDtype = (value: unknown, where: string): IrDtype => {
  const dtype = asNonEmptyString(value, where);
  if (!isSemanticDtype(dtype)) {
    throw new IrError(
      `${where}: 意味論 dtype '${dtype}' は語彙外（${SEMANTIC_DTYPES.join(" / ")}）`,
    );
  }
  return dtype;
};

const asStorageDtype = (value: unknown, where: string): IrStorageDtype => {
  const dtype = asNonEmptyString(value, where);
  if (!isStorageDtype(dtype)) {
    // 語彙は STORAGE_DTYPES が正本（i32 は ADR 0010 の生 int32）。診断に並べる名前を
    // 手書きすると語彙を足したときに片方だけ古くなる。
    throw new IrError(`${where}: 格納 dtype '${dtype}' は語彙外（${STORAGE_DTYPES.join(" / ")}）`);
  }
  return dtype;
};

const parseShape = (value: unknown, symbols: ReadonlySet<string>, where: string): IrDim[] =>
  asArray(value, where).map((dim, index) => {
    const at = `${where}[${index}]`;
    if (typeof dim === "number") {
      if (!Number.isSafeInteger(dim) || dim < 0) {
        throw new IrError(`${at}: 次元 ${dim} が非負整数でない`);
      }
      return dim;
    }
    if (typeof dim !== "string") throw new IrError(`${at}: 次元が数値でも文字列でもない`);
    const expr = tryParseDim(dim);
    if (expr === undefined) {
      throw new IrError(`${at}: 次元式 '${dim}' が正準文法 coeff·sym+offset に適合しない`);
    }
    if (!symbols.has(expr.sym)) {
      throw new IrError(`${at}: シンボル '${expr.sym}' が graph.symbols で宣言されていない`);
    }
    return dim;
  });

const parseStorage = (value: unknown, where: string): IrStorage => {
  const obj = asPlainObject(value, where);
  checkKeys(obj, ["dtype"], ["scale", "group_size"], where);
  const dtype = asStorageDtype(obj["dtype"], `${where}.dtype`);
  const hasScale = Object.hasOwn(obj, "scale");
  const hasGroupSize = Object.hasOwn(obj, "group_size");
  // scale / group_size は量子化格納の記述子。非量子化 dtype に付いているのはエクスポータの
  // 取り違えなので受理しない（黙って無視すると格納の意味が二重化する）。
  if (dtype !== "i8" && (hasScale || hasGroupSize)) {
    throw new IrError(`${where}: 格納 dtype '${dtype}' に scale / group_size は付けられない`);
  }
  // MUST: i8 は scale を**明示宣言**する（ADR 0019）。既定 1.0 で補完すると、scale を
  // 書き忘れたモデルが「量子化前の 1/127 倍の重み」で静かに走る。
  if (dtype === "i8" && !hasScale) {
    throw new IrError(`${where}: 格納 dtype 'i8' には scale（scale テンソルのキー）が要る`);
  }
  const storage: { dtype: IrStorageDtype; scale?: string; groupSize?: number } = { dtype };
  if (hasScale) storage.scale = asNonEmptyString(obj["scale"], `${where}.scale`);
  if (hasGroupSize) {
    const groupSize = obj["group_size"];
    if (typeof groupSize !== "number" || !Number.isSafeInteger(groupSize) || groupSize < 1) {
      throw new IrError(`${where}.group_size: 正整数でない`);
    }
    storage.groupSize = groupSize;
  }
  return storage;
};

/**
 * NaN / Infinity は JSON リテラルに無いが、`1e999` のような指数は Infinity へ丸まる。
 * 受理集合をブラウザ JSON.parse に揃える契約（docs/ir-v1.md）を保つため、非有限数を含む
 * グラフは MUST NOT 受理。
 */
const parseJson = (json: string): unknown => {
  try {
    return JSON.parse(json, (key: string, value: unknown) => {
      if (typeof value === "number" && !Number.isFinite(value)) {
        throw new IrError(`グラフ JSON の '${key}' が有限な数値でない`);
      }
      return value;
    });
  } catch (cause) {
    if (cause instanceof IrError) throw cause;
    throw new IrError(`グラフ JSON を解析できない: ${String(cause)}`);
  }
};

export const parseIrGraph = (json: string): IrGraph => {
  const root = asPlainObject(parseJson(json), "graph");
  checkKeys(root, TOP_LEVEL_KEYS, [], "graph");

  const format = root["format"];
  if (format !== IR_FORMAT) {
    throw new IrError(`graph.format が '${IR_FORMAT}' でない: ${JSON.stringify(format)}`);
  }
  const version = root["version"];
  if (version !== IR_VERSION) {
    throw new IrError(`graph.version が ${IR_VERSION} でない: ${JSON.stringify(version)}`);
  }

  const requires = asPlainObject(root["requires"], "graph.requires");
  checkKeys(requires, ["ops"], [], "graph.requires");
  const requiredOps = asUniqueStrings(requires["ops"], "graph.requires.ops");

  const symbols = asUniqueStrings(root["symbols"], "graph.symbols");
  for (const symbol of symbols) {
    if (!isSymbolName(symbol)) throw new IrError(`graph.symbols: シンボル名 '${symbol}' が不正`);
  }
  const symbolSet = new Set(symbols);

  const inputs: IrInput[] = asArray(root["inputs"], "graph.inputs").map((raw, index) => {
    const where = `graph.inputs[${index}]`;
    const obj = asPlainObject(raw, where);
    checkKeys(obj, ["name", "dtype", "shape"], [], where);
    return {
      name: asNonEmptyString(obj["name"], `${where}.name`),
      dtype: asSemanticDtype(obj["dtype"], `${where}.dtype`),
      shape: parseShape(obj["shape"], symbolSet, `${where}.shape`),
    };
  });

  const outputs = asUniqueStrings(root["outputs"], "graph.outputs");

  // MUST: パース入力由来のキーを蓄積する器は null プロトタイプ。素の `{}` では名前が
  // "__proto__" のとき代入が [[Prototype]] 設定に化け、own property が作られないまま宣言が
  // 黙って消えてグラフが受理される。Deno は既定でこの setter を無効化しているため手元では
  // 再現しないが、ブラウザ（対象実行系の一方）では起きる。
  const values: Record<string, IrValueInfo> = Object.create(null);
  for (const [name, raw] of Object.entries(asPlainObject(root["values"], "graph.values"))) {
    const where = `graph.values['${name}']`;
    const obj = asPlainObject(raw, where);
    checkKeys(obj, ["dtype", "shape"], [], where);
    values[name] = {
      dtype: asSemanticDtype(obj["dtype"], `${where}.dtype`),
      shape: parseShape(obj["shape"], symbolSet, `${where}.shape`),
    };
  }

  // MUST: values と同じ理由で null プロトタイプ（上のコメント参照）。
  const initializers: Record<string, IrInitializer> = Object.create(null);
  for (
    const [name, raw] of Object.entries(asPlainObject(root["initializers"], "graph.initializers"))
  ) {
    const where = `graph.initializers['${name}']`;
    const obj = asPlainObject(raw, where);
    checkKeys(obj, ["tensor", "storage"], [], where);
    initializers[name] = {
      tensor: asNonEmptyString(obj["tensor"], `${where}.tensor`),
      storage: parseStorage(obj["storage"], `${where}.storage`),
    };
  }

  const nodes: IrNode[] = asArray(root["nodes"], "graph.nodes").map((raw, index) => {
    const where = `graph.nodes[${index}]`;
    const obj = asPlainObject(raw, where);
    checkKeys(obj, ["op", "ins", "outs", "attrs"], [], where);
    const outs = asArray(obj["outs"], `${where}.outs`).map((out, i) =>
      asNonEmptyString(out, `${where}.outs[${i}]`)
    );
    if (outs.length === 0) {
      throw new IrError(`${where}: outs が空（値を定義しないノードは静的 DAG に置けない）`);
    }
    return {
      op: asNonEmptyString(obj["op"], `${where}.op`),
      ins: asArray(obj["ins"], `${where}.ins`).map((input, i) =>
        asNonEmptyString(input, `${where}.ins[${i}]`)
      ),
      outs,
      attrs: asPlainObject(obj["attrs"], `${where}.attrs`),
    };
  });

  checkSymbolBindability(symbols, inputs);
  const defined = checkDefinitions(inputs, initializers, nodes, outputs);
  checkDeclarations(inputs, initializers, values, nodes, defined);
  checkRequiredOps(requiredOps, nodes);

  return {
    format: IR_FORMAT,
    version: IR_VERSION,
    requires: { ops: requiredOps },
    symbols,
    inputs,
    outputs,
    initializers,
    values,
    nodes,
  };
};

/**
 * 束縛は入力 shape の次元位置から直接取る（要素数からの逆算はしない）ため、宣言された
 * シンボルは少なくとも 1 つの入力 shape に素の形（係数 1・オフセット 0）で現れる MUST。
 */
const checkSymbolBindability = (
  symbols: readonly string[],
  inputs: readonly IrInput[],
): void => {
  const bindable = new Set<string>();
  for (const input of inputs) {
    for (const dim of input.shape) {
      if (typeof dim !== "string") continue;
      const expr = parseDim(dim);
      if (expr.coeff === 1 && expr.offset === 0) bindable.add(expr.sym);
    }
  }
  for (const symbol of symbols) {
    if (!bindable.has(symbol)) {
      throw new IrError(
        `graph.symbols: '${symbol}' が入力 shape に素の形（'${symbol}'）で現れない — 束縛が取れない`,
      );
    }
  }
};

/** SSA 単一代入 + トポロジカル順（前方参照拒否）+ outputs の定義済み検査。 */
const checkDefinitions = (
  inputs: readonly IrInput[],
  initializers: Readonly<Record<string, IrInitializer>>,
  nodes: readonly IrNode[],
  outputs: readonly string[],
): ReadonlySet<string> => {
  const defined = new Set<string>();
  const define = (name: string, where: string): void => {
    if (defined.has(name)) {
      throw new IrError(`${where}: 値 '${name}' が二重に定義されている（SSA 単一代入違反）`);
    }
    defined.add(name);
  };
  for (const input of inputs) define(input.name, "graph.inputs");
  for (const name of Object.keys(initializers)) define(name, "graph.initializers");
  nodes.forEach((node, index) => {
    const where = `graph.nodes[${index}] (${node.op})`;
    for (const ref of node.ins) {
      // nodes はトポロジカル順で格納される MUST — 前方参照は実行順を暗黙の依存解析に
      // 依存させるので、ここで落とす。
      if (!defined.has(ref)) {
        throw new IrError(`${where}: 入力 '${ref}' が未定義（前方参照または未宣言）`);
      }
    }
    for (const out of node.outs) define(out, where);
  });
  for (const output of outputs) {
    if (!defined.has(output)) throw new IrError(`graph.outputs: '${output}' が未定義`);
  }
  return defined;
};

/**
 * 宣言の完全性: 入力は inputs[] が、initializer とノード出力は values{} が、
 * それぞれちょうど 1 回宣言する。孤立宣言（誰も定義しない values）も fail loudly。
 */
const checkDeclarations = (
  inputs: readonly IrInput[],
  initializers: Readonly<Record<string, IrInitializer>>,
  values: Readonly<Record<string, IrValueInfo>>,
  nodes: readonly IrNode[],
  defined: ReadonlySet<string>,
): void => {
  const inputNames = new Set(inputs.map((input) => input.name));
  for (const name of Object.keys(values)) {
    if (inputNames.has(name)) {
      throw new IrError(`graph.values['${name}']: 入力は inputs[] で宣言済み（二重宣言）`);
    }
    if (!defined.has(name)) {
      throw new IrError(`graph.values['${name}']: どのノードでも定義されない宣言`);
    }
  }
  for (const name of Object.keys(initializers)) {
    if (!Object.hasOwn(values, name)) {
      throw new IrError(`graph.initializers['${name}']: values に dtype/shape 宣言が無い`);
    }
    // MUST: グラフ単体で決まる仕様規則はパーサ 1 箇所で見る — ロード経路ごとに書くと規則が
    // 割れる。格納 dtype の実行可否（f16/bf16/i8 は宣言のみ）は宣言の valid 性とは別の層で、
    // assertRuntimeSupport が持つ。
    const allowedStorage = INITIALIZER_STORAGE.get(values[name].dtype);
    if (allowedStorage === undefined) {
      throw new IrError(
        `graph.values['${name}']: initializer の意味論 dtype '${
          values[name].dtype
        }' は語彙外（f32 / i32 のみ）`,
      );
    }
    const storageDtype = initializers[name].storage.dtype;
    if (!allowedStorage.includes(storageDtype)) {
      throw new IrError(
        `graph.initializers['${name}']: 意味論 dtype '${
          values[name].dtype
        }' に格納 dtype '${storageDtype}' は組めない（${allowedStorage.join(" / ")} のみ）`,
      );
    }
    // initializer は束縛前に確定していなければ safetensors 側 shape と突合できない。
    if (values[name].shape.some((dim) => typeof dim !== "number")) {
      throw new IrError(`graph.values['${name}']: initializer の shape に記号次元は使えない`);
    }
  }
  for (const node of nodes) {
    for (const out of node.outs) {
      if (!Object.hasOwn(values, out)) {
        throw new IrError(`graph.values: ノード出力 '${out}' の dtype/shape 宣言が無い`);
      }
    }
  }
};

/** requires.ops ≡ nodes で実際に使われる op 集合（ランタイム突合の前提）。 */
const checkRequiredOps = (requiredOps: readonly string[], nodes: readonly IrNode[]): void => {
  const used = new Set(nodes.map((node) => node.op));
  const declared = new Set(requiredOps);
  const missing = [...used].filter((op) => !declared.has(op)).sort();
  const extra = [...declared].filter((op) => !used.has(op)).sort();
  if (missing.length > 0 || extra.length > 0) {
    throw new IrError(
      `graph.requires.ops が使用 op 集合と一致しない: 宣言漏れ [${missing.join(", ")}] / 余剰 [${
        extra.join(", ")
      }]`,
    );
  }
};
