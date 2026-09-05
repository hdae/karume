// テンソルを 1 本も持たない IR コンテナをバイト列で組む（テスト用の器）。
//
// 家族 admission のグラフ突合（`src/irodori/pipeline.ts` の `assertStaticDim` 群）は
// **失敗経路しか無い門**で、破れると「shape は合ったまま別の位置の条件を読む」沈黙誤値になる。
// その門を踏むには実 IR コンテナが要るが、配布形の重みは GB 級なのでテストからは使えない。
// `src/irodori/host/sampler-graph.ts` の `packGraph` が**まさにその器**（重みを持たない小さな
// グラフを safetensors バイト列で組む）なので、同型のものをテスト側に置く。
//
// MUST: `sampler-graph.ts` 側をここへ寄せない — あちらは**数値の正本**（ノード列が `sampler.ts`
// の写しであること自体が門）なので、テスト helper に依存させるとその門が緩む。

/** グラフ 1 本の指定（宣言だけ — 値は流さないので演算の中身は問わない）。 */
export type TensorlessGraphSpec = {
  /** 記号次元の名前（省略時は静的グラフ）。入力 shape のどこかに現れる必要がある。 */
  readonly symbols?: readonly string[];
  /** 入力の宣言（`assertStaticDim` が読む面）。 */
  readonly inputs: readonly {
    readonly name: string;
    readonly shape: readonly (number | string)[];
  }[];
  /** 出力 1 本の宣言（`assertOutputScale` / `assertOutputDim` が読む面）。 */
  readonly output: {
    readonly name: string;
    readonly shape: readonly (number | string)[];
  };
};

/** グラフ JSON を載せる safetensors `__metadata__` のキー（runtime の `IR_METADATA_KEY`）。 */
const IR_METADATA_KEY = "karume_ir";

/**
 * 指定から IR v1 のコンテナ（`openModel` がそのまま読めるバイト列）を組む。
 *
 * ノードは「先頭入力を 2 口で受ける `mul` 1 本」だけ — 宣言の突合が目的なので実行はしない
 * （runtime は open 時に shape 推論をしない）。ヘッダは 8 バイト境界へ空白で詰める
 * （safetensors の慣例で、runtime の整列検査もこれを前提にしている）。
 */
export const packTensorlessGraph = (spec: TensorlessGraphSpec): ArrayBuffer => {
  const graph = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["mul"] },
    symbols: spec.symbols ?? [],
    inputs: spec.inputs.map((input) => ({ name: input.name, dtype: "f32", shape: input.shape })),
    outputs: [spec.output.name],
    initializers: {},
    values: { [spec.output.name]: { dtype: "f32", shape: spec.output.shape } },
    nodes: [{
      op: "mul",
      ins: [spec.inputs[0].name, spec.inputs[0].name],
      outs: [spec.output.name],
      attrs: {},
    }],
  };
  const headerBytes = new TextEncoder().encode(
    JSON.stringify({ __metadata__: { [IR_METADATA_KEY]: JSON.stringify(graph) } }),
  );
  const headerLength = headerBytes.length + ((8 - (headerBytes.length % 8)) % 8);
  const buffer = new ArrayBuffer(8 + headerLength);
  const bytes = new Uint8Array(buffer);
  new DataView(buffer).setBigUint64(0, BigInt(headerLength), true);
  bytes.set(headerBytes, 8);
  bytes.fill(0x20, 8 + headerBytes.length, 8 + headerLength);
  return buffer;
};

/** {@link packTensorlessGraph} の結果を `fromAssets` の資産形（buffer 全体を占める view）で返す。 */
export const tensorlessGraphAsset = (spec: TensorlessGraphSpec): Uint8Array<ArrayBuffer> =>
  new Uint8Array(packTensorlessGraph(spec));
