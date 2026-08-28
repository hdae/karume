/**
 * 重みの**常駐分類**（席）— グラフ宣言だけで決まる純関数プランナと、その結果を GPU 実体へ
 * 結び付けた Session 側の判別 union。
 *
 * MUST: 分類の正本はここ 1 本（{@link planWeightResidency}）。Session 構築（executor.ts）と
 * 見積り（estimate.ts）が別々に「適格判定 + 格納 dtype の分岐」を書くと、片方だけ直された
 * ときに **見積りと実ロードが別のモデルを説明する**（例外も警告も出ない）。
 * MUST: 実テンソル（safetensors）を見ない。バイト数は宣言 shape と格納メタデータから導き、
 * 実バイトとの一致は container の突合門が保証する（{@link declaredPayloadBytes} の doc）。
 */

import { declaredPayloadBytes, declaredScaleBytes } from "../format/container.ts";
import { groupScaleShape } from "../format/i4.ts";
import type { IrGraph } from "../format/ir.ts";
import { numel } from "../ops.ts";
import {
  eligibleCompressedInitializers,
  ExecutionError,
  i4EligibleInitializers,
  weightChannelAxes,
} from "./plan.ts";

/**
 * 圧縮のまま GPU 常駐した重み 1 本（Session の重み台帳と対の索引 — カーネル変種の選択と
 * 追加束縛がここだけで決まる）。
 *
 * MUST: payload の `GPUBuffer` は持たない（重み台帳 `weightBuffers` が所有 — 二重保持にすると
 * 「どちらが本物か」が生まれる）。scale だけは重み台帳に載らない実体なのでここが所有する。
 * MUST: 席ごとの付随情報を**型で**要求する（i8 / i4 は scale 必須・i4 は group 長必須）。
 * 3 本の並列 Map で持つと「i4 なのに group 長が無い」形が型の上では作れてしまい、実行時検査
 * だけが最後の砦になる。
 */
export type ResidentWeight =
  | { readonly storage: "f16" }
  | { readonly storage: "i8"; readonly scale: GPUBuffer }
  | { readonly storage: "i4"; readonly scale: GPUBuffer; readonly groupSize: number };

/**
 * initializer 1 本の常駐分類（{@link planWeightResidency} の値）。
 *
 * 席は 5 つ:
 * - `raw` — 圧縮しない格納（f32 / i32 / bf16）を生バイトのまま GPU 常駐（executor の分岐 3 本目）
 * - `f16` / `i8` / `i4` — 圧縮のまま常駐し dequant はカーネル内（ADR 0018 / 0019 / 0069）
 * - `expanded` — 適格外でロード時に CPU で f32 展開（正しさは保たれ VRAM 削減はゼロ）
 *
 * `payloadBytes` は**格納バイト列そのものの長さ**（整列詰め物もバッファ床も含まない）。
 * 整列は転送側（`alignF16Payload` / `alignI8Payload`）とアリーナ（`toSizeClass`）が持つ。
 */
export type WeightResidency =
  | { readonly seat: "raw"; readonly payloadBytes: number }
  | { readonly seat: "f16"; readonly payloadBytes: number }
  | {
    readonly seat: "i8";
    readonly payloadBytes: number;
    readonly scaleBytes: number;
    /** per-channel scale が掛かる軸（消費側 op から決まる — ADR 0019）。 */
    readonly channelAxis: number;
  }
  | {
    readonly seat: "i4";
    readonly payloadBytes: number;
    readonly scaleBytes: number;
    readonly groupSize: number;
  }
  | {
    readonly seat: "expanded";
    readonly payloadBytes: number;
    /** CPU で f32 へ展開した後のバイト数（常駐するのはこちら）。 */
    readonly expandedBytes: number;
  };

/**
 * グラフだけから initializer ごとの常駐分類を決める（GPU も safetensors も要らない純関数）。
 *
 * 分類は 3 点 — 適格判定 {@link eligibleCompressedInitializers}、i4 だけ
 * {@link i4EligibleInitializers} との積（ADR 0069 決定 5）、格納 dtype ごとの分岐。
 *
 * MUST: `graph.initializers` の**全件**を返す（宣言順）。欠けを許すと、消費側が「表に無い =
 * f32 で読む」と解釈する既定と区別できなくなる。
 */
export const planWeightResidency = (graph: IrGraph): ReadonlyMap<string, WeightResidency> => {
  // 圧縮格納のまま上げてよい initializer（消費が重みスロットだけ — ADR 0018）。
  const eligible = eligibleCompressedInitializers(graph);
  // i4 の適格はさらに狭く「重みスロットでの消費が linear / embedding / conv1d(groups==1) だけ」
  // （ADR 0069 決定 5 とその追補 — 展開経路を持つカーネルはこの 3 つ）。
  const i4Eligible = i4EligibleInitializers(graph);
  // i8 の per-channel scale が掛かる軸（消費側 op から決まる — ADR 0019）。
  const channelAxes = weightChannelAxes(graph);
  const plan = new Map<string, WeightResidency>();
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const where = `initializer '${name}'`;
    // initializer の宣言 shape は数値のみ（parseIrGraph が保証 — 記号次元は拒否）。
    const shape = graph.values[name].shape.map(Number);
    const count = numel(shape);
    const storage = initializer.storage.dtype;
    const payloadBytes = declaredPayloadBytes(storage, count, where);
    if (storage === "f32" || storage === "i32" || storage === "bf16") {
      // 圧縮しない格納は生バイトがそのまま GPU 表現。
      plan.set(name, { seat: "raw", payloadBytes });
      continue;
    }
    const resident = storage === "i4"
      ? eligible.has(name) && i4Eligible.has(name)
      : eligible.has(name);
    if (!resident) {
      plan.set(name, { seat: "expanded", payloadBytes, expandedBytes: count * 4 });
      continue;
    }
    if (storage === "f16") {
      plan.set(name, { seat: "f16", payloadBytes });
      continue;
    }
    if (storage === "i8") {
      const channelAxis = channelAxes.get(name);
      if (channelAxis === undefined) {
        throw new ExecutionError(`${where}: per-channel scale のチャネル軸が決まらない`);
      }
      const channels = shape[channelAxis];
      if (channels === undefined) {
        throw new ExecutionError(
          `${where}: 重み [${shape.join(",")}] にチャネル軸 ${channelAxis} が無い`,
        );
      }
      // GPU 常駐経路の scale は「チャネル軸だけが伸びた keepdim 形」でなければならない
      // （executor の `assertChannelScale` が実テンソル側の門）ので、要素数はチャネル数に等しい。
      plan.set(name, {
        seat: "i8",
        payloadBytes,
        scaleBytes: declaredScaleBytes(channels, where),
        channelAxis,
      });
      continue;
    }
    // 値域（2 冪 ≥ 16・整除）と存在は parseIrGraph が保証済み。存在は型の上でだけ optional
    // なので、黙って読み飛ばさず言い直す（「格納 i8 なのに scale が無い」と同じ流儀）。
    const groupSize = initializer.storage.groupSize;
    if (groupSize === undefined) {
      throw new ExecutionError(`${where}: 格納 i4 なのに group_size が無い`);
    }
    plan.set(name, {
      seat: "i4",
      payloadBytes,
      // group 形は container の検査と展開が共有する 1 本から引く（ADR 0069 決定 3）。
      scaleBytes: declaredScaleBytes(numel(groupScaleShape(shape, groupSize)), where),
      groupSize,
    });
  }
  return plan;
};
