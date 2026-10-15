/**
 * umT5-XXL のグラフ宣言の取り決め — ホストが組む入力（id 列 `[1, L]` とバケット表 `[L, L]` の i32 —
 * `./session-io.ts`）と読む出力（`[1, L, W]` の f32）・格納（ADR 0119 決定 3・4・5）にグラフ宣言が合うかを、
 * 重みを 1 バイトも取る前に見る門。
 *
 * 単体の `Umt5Encoder`（`./encoder.ts`）と Wan の text 段（`src/wan/text-stage.ts` の `umt5Contract` — DiT の
 * 文脈入力との突き合わせを足す）が共有する。`owner` は文言の接頭辞（`"Umt5Encoder"` / `"WanPipeline"` など）。
 *
 * NOTE: 内部だけ（`mod.ts` / サブパス面には出さない — ADR 0008）。
 *
 * MUST: モジュール副作用ゼロ（CLAUDE.md）。
 */

import { codecLayout } from "@karume/runtime";

import type { GraphOwner } from "../hub/components.ts";
import { UMT5_INPUT_IDS, UMT5_RELATIVE_POSITION_BUCKETS } from "./session-io.ts";

/** 構築時に確かめた umT5 のグラフの取り決め。 */
export type Umt5GraphContract = {
  /** グラフ出力の名前（`[1, L, hiddenSize]`）。 */
  readonly output: string;
  /** 出力の幅（umT5-XXL は 4,096 — グラフ宣言の静的次元から読む）。 */
  readonly hiddenSize: number;
};

/** umT5 のグラフ入力の宣言（無ければ fail loudly）。 */
const umt5Input = (
  owner: string,
  textEncoder: GraphOwner,
  name: string,
): GraphOwner["graph"]["inputs"][number] => {
  const spec = textEncoder.graph.inputs.find((input) => input.name === name);
  if (spec === undefined) {
    const declared = textEncoder.graph.inputs.map((input) => input.name).join(" / ");
    throw new Error(
      `${owner}: text_encoder のグラフ入力 '${name}' が無い（宣言: ${declared}）`,
    );
  }
  return spec;
};

/**
 * 宣言の形を rank と全軸で照合する（数は静的次元・文字列は記号次元の名前）。`expected` はホストが組む形
 * そのもの（文言は `src/wan/graph-io.ts` の `assertDims` と同じ形）。
 */
const assertDims = (
  owner: string,
  dims: readonly (number | string)[],
  expected: readonly (number | string)[],
  where: string,
): void => {
  if (dims.length !== expected.length || dims.some((dim, axis) => dim !== expected[axis])) {
    const declared = dims.join(", ");
    const host = expected.join(", ");
    throw new Error(`${owner}: ${where} の形 [${declared}] がホストの組む [${host}] と違う`);
  }
};

/**
 * umT5 のグラフ宣言を、ホストが組む入力と読む出力・受ける格納に突き合わせる（ADR 0119 決定 3・4・5）。
 *
 * MUST: 家族 admission（重みの part を取る前）で呼ぶ。ホストは自分の定数で入力を組み、出力を
 * `[1, L, hiddenSize]` として読むので、宣言が違っていても落ちるのは umT5 の重み（i8 で約 5.3 GiB）を上げた後の
 * Session の shape 検査か、出力の検査になる。
 *
 * - 入力はちょうど 2 本（`input_ids` `[1, L]`・`relative_position_buckets` `[L, L]`・どちらも i32）で、
 *   L は**同じ 1 つの記号次元**（有効長ごとに形を変えて渡す — 決定 4）。
 * - 出力は 1 本の f32 `[1, L, W]` で、L は入力と同じ記号。W は `hiddenSize` を渡せばその値（Wan は DiT の
 *   `encoder_hidden_states` の幅を渡す）、省けば宣言の静的な正の整数をそのまま採る（単体の encoder）。
 * - 格納は i8（`int8-sym` = per-channel）の重みと f32 の表だけ（決定 5 — GPU の移植の門で検証した
 *   組み合わせ。umT5 の Session は quant の実行ノブを受けないので、格納が実行の形を決める）。
 *   i8 を 1 本も持たない容器（全部 f32 など）も受けない。
 */
export const umt5GraphContract = (
  textEncoder: GraphOwner,
  owner: string,
  hiddenSize?: number,
): Umt5GraphContract => {
  const { graph } = textEncoder;
  const ids = umt5Input(owner, textEncoder, UMT5_INPUT_IDS);
  const buckets = umt5Input(owner, textEncoder, UMT5_RELATIVE_POSITION_BUCKETS);
  if (graph.inputs.length !== 2) {
    throw new Error(
      `${owner}: text_encoder のグラフ入力が ${graph.inputs.length} 本（` +
        `'${UMT5_INPUT_IDS}' と '${UMT5_RELATIVE_POSITION_BUCKETS}' の 2 本だけを組む）`,
    );
  }
  const length = ids.shape.at(1);
  if (typeof length !== "string") {
    throw new Error(
      `${owner}: text_encoder の '${UMT5_INPUT_IDS}' の軸 1 が記号次元でない` +
        `（${String(length)}）— ホストは有効長 L ごとに形を変えて渡す`,
    );
  }
  for (const [spec, expected] of [[ids, [1, length]], [buckets, [length, length]]] as const) {
    if (spec.dtype !== "i32") {
      throw new Error(
        `${owner}: text_encoder の '${spec.name}' の dtype ${spec.dtype} が i32 でない`,
      );
    }
    assertDims(owner, spec.shape, expected, `text_encoder の '${spec.name}'`);
  }
  if (graph.outputs.length !== 1) {
    throw new Error(
      `${owner}: text_encoder の出力が ${graph.outputs.length} 本（1 本の [1, L, W]）`,
    );
  }
  const [output] = graph.outputs;
  if (!Object.hasOwn(graph.values, output)) {
    throw new Error(`${owner}: グラフの値 '${output}' の宣言が無い`);
  }
  const value = graph.values[output];
  if (value.dtype !== "f32") {
    throw new Error(
      `${owner}: text_encoder の出力 '${output}' の dtype ${value.dtype} が f32 でない`,
    );
  }
  const declaredWidth = value.shape.at(2);
  const width = hiddenSize ?? declaredWidth;
  if (typeof width !== "number" || !Number.isInteger(width) || width <= 0) {
    throw new Error(
      `${owner}: text_encoder の出力 '${output}' の形 [${value.shape.join(", ")}] の軸 2 が` +
        "静的な正の整数でない（ホストは [1, L, W] の W を宣言から読む）",
    );
  }
  assertDims(owner, value.shape, [1, length, width], `text_encoder の出力 '${output}'`);
  let quantized = 0;
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    if (initializer.storage === undefined) {
      // 共有の宣言（貸し手の Session の重みを借りる — 格納を持たない）。umT5 に貸し手は居ない。
      throw new Error(
        `${owner}: text_encoder の initializer '${name}' が共有の宣言（umT5 は重みを借りない）`,
      );
    }
    const layout = codecLayout(initializer.storage.codec);
    if (layout === "i8") quantized += 1;
    else if (layout !== "f32") {
      throw new Error(
        `${owner}: text_encoder の initializer '${name}' の格納 ${initializer.storage.codec} は受けない` +
          "（受けるのは i8 per-channel の重みと f32 の表だけ — ADR 0119 決定 5）",
      );
    }
  }
  if (quantized === 0) {
    throw new Error(
      `${owner}: text_encoder に i8 の重みが 1 本も無い（受けるのは i8 per-channel の重みと f32 の表 — ` +
        "ADR 0119 決定 5）",
    );
  }
  return { output, hiddenSize: width };
};
