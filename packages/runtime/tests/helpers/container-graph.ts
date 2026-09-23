/**
 * 配布形ミラー（`krm`）の part 0 から 1 グラフの IR を読む（グラフ記述 × 束縛表の合流まで）。
 *
 * 実資産の門（融合ヒット数など）はグラフ宣言だけを見るので、重みの block へは進まない —
 * ヘッダ（24 B）と 2 文書だけを読む。グラフ名 = 配布 manifest の weights キー（container-v1 §12）。
 *
 * MUST: 名指しのグラフが無ければ在るグラフを並べて落とす（黙って 1 本目を採ると、部品名と
 * グラフ名がずれた容器で「別の部品を数えた表」が静かに出る）。
 */

import { bindGraphs, mergedGraph } from "../../src/format/container/bind.ts";
import {
  parseGraphDescriptor,
  parseModelDescriptor,
} from "../../src/format/container/descriptor.ts";
import { readHeader } from "../../src/format/container/header.ts";
import { HEADER_BYTES } from "../../src/format/container/limits.ts";
import type { IrGraph } from "../../src/format/ir.ts";

/** `karume/5` の manifest のうち、ここが引く欄だけ（綴りを持つのは配布形）。 */
export type ContainerManifest = {
  readonly defaultModel: string;
  readonly models: Readonly<
    Record<string, {
      readonly weights: Readonly<
        Record<
          string,
          Readonly<
            Record<string, { readonly container: { readonly parts: readonly PartRef[] } }>
          >
        >
      >;
    }>
  >;
};
export type PartRef = { readonly path: string; readonly repo?: string };

const readPrefix = async (source: URL, length: number): Promise<Uint8Array<ArrayBuffer>> => {
  const file = await Deno.open(source, { read: true });
  try {
    const bytes = new Uint8Array(new ArrayBuffer(length));
    for (let read = 0; read < length;) {
      const chunk = await file.read(bytes.subarray(read));
      if (chunk === null) throw new Error(`${source.pathname}: ${length} バイトを読み切れない`);
      read += chunk;
    }
    return bytes;
  } finally {
    file.close();
  }
};

/** 容器の part 0（`…-00001-of-NNNNN.krm` または単一形）から `graphName` の合流済み IR を読む。 */
export const readContainerGraph = async (part0: URL, graphName: string): Promise<IrGraph> => {
  const header = readHeader(await readPrefix(part0, HEADER_BYTES));
  const documents = await readPrefix(
    part0,
    HEADER_BYTES + header.graphDescriptorLength + header.modelDescriptorLength,
  );
  const graph = parseGraphDescriptor(
    documents.slice(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength),
  );
  const model = header.kind === "model"
    ? parseModelDescriptor(documents.slice(HEADER_BYTES + header.graphDescriptorLength))
    : undefined;
  const bound = bindGraphs(graph, model)[graphName];
  if (bound === undefined) {
    throw new Error(
      `${part0.pathname}: 容器にグラフ '${graphName}' が無い（在るのは ${
        Object.keys(graph.graphs).join(" / ")
      }）`,
    );
  }
  return mergedGraph(bound, graphName);
};

/**
 * manifest から (model, component, dtype) の容器の part 0 を引く。越境参照（`repo` 付き）は
 * 呼び手が別のミラーへ振り分けるので、ここでは path をそのまま返す。
 */
export const containerPart0 = (
  manifest: ContainerManifest,
  model: string,
  component: string,
  dtype: string,
): PartRef => {
  const entry = manifest.models[model]?.weights[component]?.[dtype];
  if (entry === undefined) {
    throw new Error(`manifest に ${model} / ${component} / ${dtype} の容器が無い`);
  }
  return entry.container.parts[0];
};
