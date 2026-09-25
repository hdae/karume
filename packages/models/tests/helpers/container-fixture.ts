/**
 * テストから **`krm` コンテナ**と `karume/5` の配布形を組む道具（models 側のテストが共有する
 * 1 本）。
 *
 * NOTE: 容器の**書き手**は `packages/runtime/tests/helpers/container-write.ts` を借りる。
 * これは「runtime のテストの都合」ではなく**形式の道具**（Python の `karume.container` と同じ
 * 位置づけ）で、models 側に写しを持つと 2 実装がずれる。`helpers/memory-cache.ts` の規律
 * （向こうの都合をこちらへ持ち込まない）はそのまま。
 *
 * ここが足すのは「容器 → 配布形」の側だけ:
 *
 * - {@link writeContainer} … `ModelInput` を書いて、manifest が要る**期待値**（2 文書の
 *   length + sha256）と part 列を一緒に返す
 * - {@link partAssets} … 書いた容器を全量面のキー（`<部品>[i]`）へ畳む
 * - {@link declaredContainer} … manifest の `weights.<部品>.<dtype>` に入る**宣言だけ**の容器
 * - {@link linearComponent} / {@link tensorlessContainer} … 実行できる最小の部品（`linear`
 *   1 段）と、宣言だけの部品（突合の門を踏むための器）
 * - {@link runLinearProbe} … `linearComponent` の Session を 1 回 run して出力を返す（重みが
 *   GPU まで届いたかの観測点）
 */

import {
  type AssetInput,
  type ModelInput,
  type TensorInput,
  writeModelContainer,
  type WriteOptions,
} from "../../../runtime/tests/helpers/container-write.ts";
import { type IrDeclaration, parseIrDeclarationValue, type Session } from "@karume/runtime";

export { parseIrDeclarationValue };

/** 容器が内包する 1 文書の期待値（manifest の `container.descriptor` と同じ形）。 */
type DocumentExpectation = { readonly length: number; readonly sha256: string };

/** 書き出した容器 1 本（単一形・分割形・manifest が要る期待値）。 */
export type TestContainer = {
  /** 分割形（index 0 が part 0 = ヘッダ + 2 文書）。 */
  readonly parts: readonly Uint8Array<ArrayBuffer>[];
  /** 単一形（全量 1 本）。 */
  readonly single: Uint8Array<ArrayBuffer>;
  readonly descriptor: {
    readonly graph: DocumentExpectation;
    readonly model: DocumentExpectation;
  };
};

export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const documentOf = async (bytes: Uint8Array<ArrayBuffer>): Promise<DocumentExpectation> => ({
  length: bytes.byteLength,
  sha256: await sha256Hex(bytes),
});

/** 容器を 1 本書く（既定は part を小さく切って分割形を作る）。 */
export const writeContainer = async (
  input: ModelInput,
  options: WriteOptions = { partBytes: 4096, blockBytes: 512 },
): Promise<TestContainer> => {
  const written = await writeModelContainer(input, options);
  return {
    parts: written.parts,
    single: written.single,
    descriptor: {
      graph: await documentOf(written.graphDescriptorBytes),
      model: await documentOf(written.modelDescriptorBytes),
    },
  };
};

/** 宣言だけの入力（値は流さないので演算の中身は問わない）。 */
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

/**
 * 宣言から IR v2 のグラフ宣言を組む。
 *
 * ノードは「先頭入力を 2 口で受ける `mul` 1 本」だけ — 宣言の突合が目的なので実行はしない
 * （initializer を 1 本も持たないので束縛表も空になる）。
 */
const declarationOf = (spec: TensorlessGraphSpec): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["mul"] },
    symbols: spec.symbols ?? [],
    inputs: spec.inputs.map((input) => ({ name: input.name, dtype: "f32", shape: input.shape })),
    outputs: [spec.output.name],
    initializers: {},
    values: { [spec.output.name]: { dtype: "f32", shape: spec.output.shape } },
    states: {},
    nodes: [{
      op: "mul",
      ins: [spec.inputs[0].name, spec.inputs[0].name],
      outs: [spec.output.name],
      attrs: {},
    }],
  });

/**
 * 宣言だけの容器 1 本の**書く前の入力**（疑似 HF で配る `serveContainer` へ渡す形）。グラフ名 =
 * 配布 manifest の weights キー — container-v1 §2.1。
 */
export const tensorlessInput = (
  graph: string,
  spec: TensorlessGraphSpec,
  assets: readonly AssetInput[] = [],
): ModelInput => ({
  graphs: { [graph]: declarationOf(spec) },
  consts: [],
  weights: [],
  assets: [...assets],
  provenance: { license: "test" },
});

/** 宣言だけの容器 1 本（グラフ名 = 配布 manifest の weights キー — container-v1 §2.1）。 */
export const tensorlessContainer = async (
  graph: string,
  spec: TensorlessGraphSpec,
  assets: readonly AssetInput[] = [],
): Promise<TestContainer> => await writeContainer(tensorlessInput(graph, spec, assets));

const f32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from(values).buffer);

/**
 * 実行できる最小の部品（`linear(x, w, b)` 1 段・静的 2×2）。`op` を差し替えると「実行できない
 * グラフ」になる（IR パーサは op 名の綴りを見ないので、落ちるのは capability 門 = admission）。
 */
export const linearComponent = (
  graph: string,
  values: { readonly op?: string; readonly w?: number; readonly b?: number } = {},
): ModelInput => {
  const op = values.op ?? "linear";
  const declaration = parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: [op] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [2, 2] },
      b: { dtype: "f32", shape: [2] },
      y: { dtype: "f32", shape: [2, 2] },
    },
    states: {},
    nodes: [{ op, ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  });
  const weights: readonly TensorInput[] = [
    {
      graph,
      initializer: "w",
      bytes: f32Bytes(Array(4).fill(values.w ?? 0.5)),
      encoding: { codec: "f32" },
    },
    {
      graph,
      initializer: "b",
      bytes: f32Bytes(Array(2).fill(values.b ?? 0.25)),
      encoding: { codec: "f32" },
    },
  ];
  return {
    graphs: { [graph]: declaration },
    consts: [],
    weights,
    assets: [],
    provenance: { license: "test" },
  };
};

/**
 * {@link runLinearProbe} の期待値（{@link linearComponent} の既定値 w = 0.5・b = 0.25 のとき）。
 *
 * `linear` は `y[m, n] = Σ_k x[m, k] · w[n, k] + b[n]`（runtime の CPU 参照 `reference/ops.ts`）
 * なので、`x = [[1, 2], [3, 4]]` から `[[1.75, 1.75], [3.75, 3.75]]`。値は全て 2 進で割り切れる
 * ので f32 で厳密に一致する。重みの block を読まずに 0 埋めの buffer を上げた形では 0 が出る。
 */
export const LINEAR_PROBE_Y: readonly number[] = [1.75, 1.75, 3.75, 3.75];

/** {@link linearComponent} の Session を `x = [[1, 2], [3, 4]]` で 1 回 run し、`y` を平らな列で返す。 */
export const runLinearProbe = async (session: Session): Promise<number[]> => {
  const outputs = await session.run({
    x: { dtype: "f32", shape: [2, 2], data: Float32Array.from([1, 2, 3, 4]) },
  });
  return Array.from(outputs["y"].data);
};

/** 全量面（`from*Assets`）へ渡す部品キー（part 0 から添字順）。 */
export const partAssets = (
  key: string,
  container: TestContainer,
): Record<string, Uint8Array<ArrayBuffer>> =>
  Object.fromEntries(container.parts.map((part, index) => [`${key}[${index}]`, part]));

/** コンテナのヘッダ長（container-v1 §1 — part 0 は「ヘッダ + 2 文書ちょうど」）。 */
const HEADER_BYTES = 24;

/**
 * manifest の `weights.<部品>.<dtype>` に入る**宣言だけ**の容器（`fromAssets` の門は part の
 * 実体を読まないので、3 点セットは綴りだけ合わせる）。part 0 の size は宣言した 2 文書の長さと
 * 整合させる — そこは hub の parse が見る（container-v1 §8）。
 */
export const declaredContainer = (stem: string): Record<string, unknown> => {
  const graph = { length: 16, sha256: "b".repeat(64) };
  const model = { length: 8, sha256: "c".repeat(64) };
  return {
    container: {
      descriptor: { graph, model },
      parts: [
        {
          path: `${stem}-00001-of-00002.krm`,
          size: HEADER_BYTES + graph.length + model.length,
          sha256: "a".repeat(64),
        },
        { path: `${stem}-00002-of-00002.krm`, size: 64, sha256: "d".repeat(64) },
      ],
    },
  };
};
