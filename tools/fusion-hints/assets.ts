/**
 * 実配布資産 / 系列出力から IR グラフを見つけて読む（融合候補列挙の入力側）。
 *
 * MUST: safetensors は**ヘッダだけ**を読む。実体は合計 GB 級で、IR は
 * `__metadata__.karume_ir` に載っている（ADR 0070 決定 3 — グラフを持つのは先頭 shard だけ）。
 * 読み方は packages/runtime/tests/assets_fusion_counts_test.ts と同じ。
 */

import { type IrGraph, parseIrGraph } from "../../packages/runtime/src/format/ir.ts";
import { resolveShape, type SymbolBindings } from "../../packages/runtime/src/runtime/plan.ts";

/** 読み出し対象のグラフ 1 本。 */
export type GraphSource = {
  /** 表の見出し（配布形は `<モデル>/<コンポーネント>`・系列は相対ディレクトリ）。 */
  readonly name: string;
  readonly url: URL;
};

/**
 * 先頭 shard のファイル名。
 *
 * - `model.safetensors` / `model.f16.safetensors`（分割なし）
 * - `model-00001-of-00019.safetensors` / `model.i4-00001-of-00007.safetensors`（分割 v2）
 *
 * MUST: 格納 dtype の綴りは `[A-Za-z0-9]+` に限る。`-` や数字を含む文字クラスにすると
 * `model.i4-00002-of-00007.safetensors` の連番部分まで dtype として飲み込み、2 本目以降の
 * shard（IR を持たない）を先頭として拾う。
 */
const HEAD_SHARD = /^model(?:\.[A-Za-z0-9]+)?(?:-00001-of-\d{5})?\.safetensors$/;

/** safetensors のヘッダ JSON だけを読み、載っている IR を返す。 */
export const readIrGraph = async (url: URL): Promise<IrGraph> => {
  const file = await Deno.open(url, { read: true });
  try {
    const lengthBytes = new Uint8Array(8);
    await file.read(lengthBytes);
    const length = Number(new DataView(lengthBytes.buffer).getBigUint64(0, true));
    const headerBytes = new Uint8Array(length);
    for (let read = 0; read < length;) {
      const chunk = await file.read(headerBytes.subarray(read));
      if (chunk === null) {
        throw new Error(`${url.pathname}: ヘッダ ${length} バイトを読み切れない`);
      }
      read += chunk;
    }
    const header = JSON.parse(new TextDecoder().decode(headerBytes));
    const ir = header.__metadata__?.karume_ir;
    if (typeof ir !== "string") {
      throw new Error(`${url.pathname}: __metadata__.karume_ir が無い（IR を載せない資産）`);
    }
    return parseIrGraph(ir);
  } finally {
    file.close();
  }
};

/** 配布 manifest（karume.json）のうち shard の綴りを引く欄だけ。 */
type Manifest = {
  readonly defaultModel: string;
  readonly models: Readonly<
    Record<string, {
      readonly weights: Readonly<
        Record<
          string,
          Readonly<Record<string, { readonly shards: readonly ShardRef[] }>>
        >
      >;
    }>
  >;
};
type ShardRef = { readonly path: string; readonly repo?: string };

/**
 * 配布形（karume.json のある根）のグラフ列。
 *
 * 融合ヒット数は**ノード列だけ**で決まる（格納 dtype はどの initializer をどう読むかしか
 * 変えない）ので、コンポーネントごとに dtype は 1 つだけ見る。
 */
const manifestSources = async (
  root: URL,
  model: string | undefined,
): Promise<readonly GraphSource[]> => {
  const manifest: Manifest = JSON.parse(await Deno.readTextFile(new URL("karume.json", root)));
  const name = model ?? manifest.defaultModel;
  const entry = manifest.models[name];
  if (entry === undefined) {
    throw new Error(
      `モデル '${name}' が manifest に無い（既知: ${Object.keys(manifest.models).join(" / ")}）`,
    );
  }
  return Object.entries(entry.weights).map(([component, byDtype]) => {
    const dtype = Object.keys(byDtype).sort()[0];
    const [head] = byDtype[dtype].shards;
    if (head.repo !== undefined) {
      // 越境参照（ADR 0038 §7）はローカルミラーを解決しないと読めない。黙って飛ばすと
      // 「候補ゼロのコンポーネント」として表に出てしまう。
      throw new Error(
        `${name}/${component}: 越境参照 '${head.repo}' のローカルミラーを解決していない`,
      );
    }
    return { name: `${name}/${component}`, url: new URL(head.path, root) };
  });
};

/** 系列出力（manifest 無し）のグラフ列 — 先頭 shard をディレクトリ木から探す。 */
const seriesSources = async (root: URL): Promise<GraphSource[]> => {
  const found: GraphSource[] = [];
  const walk = async (dir: URL, relative: string, depth: number): Promise<void> => {
    const heads: string[] = [];
    const children: string[] = [];
    for await (const entry of Deno.readDir(dir)) {
      if (entry.isDirectory) children.push(entry.name);
      else if (entry.isFile && HEAD_SHARD.test(entry.name)) heads.push(entry.name);
    }
    // 同じコンポーネントの dtype 違い（`model.f16-…` / `model.i8-…`）は 1 本だけ見る。
    const [head] = heads.sort();
    if (head !== undefined) {
      found.push({ name: relative === "" ? rootName(root) : relative, url: new URL(head, dir) });
    }
    if (depth === 0) return;
    for (const child of children.sort()) {
      await walk(
        new URL(`${child}/`, dir),
        relative === "" ? child : `${relative}/${child}`,
        depth - 1,
      );
    }
  };
  await walk(root, "", 3);
  return found;
};

/** ディレクトリ URL の末尾要素（`…/gemma4-e2b-decode/` → `gemma4-e2b-decode`）。 */
const rootName = (root: URL): string =>
  decodeURIComponent(root.pathname).replace(/\/$/, "").split("/").pop() ?? "?";

/**
 * 取得元（配布形 or 系列ディレクトリ）に載っているグラフ列。
 * karume.json があれば配布形として manifest から引き、無ければ木を歩いて先頭 shard を拾う。
 */
export const discoverGraphs = async (
  root: URL,
  model?: string,
): Promise<readonly GraphSource[]> => {
  try {
    await Deno.stat(new URL("karume.json", root));
  } catch (cause) {
    // MUST: NotFound 以外は伝播させる — 全 I/O エラーを「manifest 無し」に丸めると、資産根の
    // マウント異常が「系列ディレクトリ」の解釈に化ける。
    if (!(cause instanceof Deno.errors.NotFound)) throw cause;
    return await seriesSources(root);
  }
  return await manifestSources(root, model);
};

/** シンボル束縛から導いた、計画に渡す具体 shape。 */
export type BoundShapes = {
  readonly symbols: SymbolBindings;
  readonly inputShapes: Readonly<Record<string, readonly number[]>>;
  readonly stateShapes: ReadonlyMap<string, readonly number[]>;
};

/**
 * グラフのシンボルを束縛して入力 / state の具体 shape を作る。
 *
 * MUST: 束縛の無いシンボルは fail loudly。既定値で黙って埋めると、`M`（chunk 行数）のような
 * **ヒット数を左右する**記号が知らない値で解かれた表が出る（gemma4 decode の rope は M=1 で
 * 15・M=32 で 0）。
 */
export const bindGraphSymbols = (
  graph: IrGraph,
  explicit: Readonly<Record<string, number>>,
  fallback: number | undefined,
): BoundShapes => {
  const symbols: Record<string, number> = {};
  for (const symbol of graph.symbols) {
    const value = Object.hasOwn(explicit, symbol) ? explicit[symbol] : fallback;
    if (value === undefined) {
      throw new Error(
        `シンボル '${symbol}' の束縛が無い（--bind ${symbol}=<値> か --default-symbol <値>）`,
      );
    }
    symbols[symbol] = value;
  }
  return {
    symbols,
    inputShapes: Object.fromEntries(
      graph.inputs.map((spec) => [spec.name, resolveShape(spec.shape, symbols)]),
    ),
    stateShapes: new Map(
      Object.entries(graph.states).map(([name, slot]) => [name, resolveShape(slot.shape, symbols)]),
    ),
  };
};
