/**
 * `ternary` codec（container-v1 §6.3・ADR 0108 決定 13）— `int2-off` の値域の部分集合を別名で
 * 宣言する codec。
 *
 * 固定するのは 3 つ:
 * - 詰め方と復元式は i2 と同一なので、展開経路は i2 を**そのまま共有する**（受理と layout）。
 * - 「三値である」という主張の検査可能な中身 = payload の全 2 bit コードが `{1, 2, 3}`。判定は
 *   1 バイトの 4 レーンのマスクで、どのレーンのコード 0 も落とす（1 レーンの見落としは、
 *   その位置の要素が三値の外の値 `q = −2` のまま黙って通る）。
 * - その検査が Session 構築の入口（`containerBatches` の items）で実際に効くこと。
 *
 * 実 GPU は使わない（検査は block を読んだ直後のホスト側）。
 */

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { assertTernaryCodes } from "../src/format/container/bind.ts";
import { codecLayout } from "../src/format/container/codecs.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { containerBatches } from "../src/runtime/executor.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  memoryModel,
  openModelBytes,
} from "./helpers/model-fixture.ts";

const ROWS = 4;
const ROW_LENGTH = 16;

/** y = linear(x, w, b) — `w` は [4,16]（i2 経路の宣言 shape: 正の rank 2・行長 16 の倍数）。 */
const declaration = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, ROW_LENGTH] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [ROWS, ROW_LENGTH] },
    b: { dtype: "f32", shape: [ROWS] },
    y: { dtype: "f32", shape: [2, ROWS] },
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/** `w` を ternary（per-channel・f32 scale）で、`payload` をそのまま供給する（16 B = 64 要素）。 */
const tensors = (payload: Uint8Array<ArrayBuffer>): readonly TensorInput[] => [
  {
    graph: GRAPH_NAME,
    initializer: "w",
    bytes: payload,
    encoding: {
      codec: "ternary",
      groupSize: ROW_LENGTH,
      scale: { bytes: f32Bytes(new Array(ROWS).fill(0.5)), dtype: "f32" },
    },
  },
  {
    graph: GRAPH_NAME,
    initializer: "b",
    bytes: f32Bytes(new Array(ROWS).fill(0)),
    encoding: { codec: "f32" },
  },
];

/** 全コードが 1 / 2 / 3 の payload（0x55 = コード 1 × 4・0xFF = 3 × 4・0xAA = 2 × 4）。 */
const validPayload = (): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length: (ROWS * ROW_LENGTH) / 4 }, (_, i) => [0x55, 0xff, 0xaa][i % 3]);

describe("ternary: i2 経路の共有", () => {
  it("krm の ternary 宣言は受理され、展開経路は i2", async () => {
    const opened = await openModelBytes(declaration(), tensors(validPayload()));
    assertEquals(opened.graphs[GRAPH_NAME].supplies.get("w")?.encoding.codec, "ternary");
    assertEquals(codecLayout("ternary"), "i2");
  });
});

describe("assertTernaryCodes: payload の全コードが {1, 2, 3}", () => {
  it("コード 0 を含まない payload は通す", () => {
    assertTernaryCodes(Uint8Array.of(0x55, 0xff, 0xaa), "w");
  });

  it("コード 0 を含むバイトを、その位置つきで落とす", () => {
    // 0x54 = 01 01 01 00 — 最下位レーンだけがコード 0。
    assertThrows(
      () => assertTernaryCodes(Uint8Array.of(0x55, 0x54), "w"),
      ContainerFormatError,
      "バイト 1",
    );
  });

  it("4 レーンのどの位置のコード 0 も落とす", () => {
    // 各レーンだけを 0 にした 4 通り（他の 3 レーンはコード 1）。
    for (const byte of [0x54, 0x51, 0x45, 0x15]) {
      assertThrows(
        () => assertTernaryCodes(Uint8Array.of(byte), "w"),
        ContainerFormatError,
        "コード 0",
        `0x${byte.toString(16)}`,
      );
    }
  });
});

describe("ternary: Session 構築の入口（containerBatches）", () => {
  it("コード 0 を含む供給は items の反復で落ちる", async () => {
    const payload = validPayload();
    payload[9] = 0xa8; // 10 10 10 00 — 最下位レーンだけがコード 0。
    const opened = memoryModel(declaration(), tensors(payload));
    await assertRejects(
      async () => {
        for await (const batch of containerBatches(opened, GRAPH_NAME)) {
          for await (const _item of batch.items) { /* 引くたびに block を読んで検査する */ }
        }
      },
      ContainerFormatError,
      "バイト 9",
    );
  });

  it("コード 0 を含まない供給は items を最後まで引ける", async () => {
    const opened = memoryModel(declaration(), tensors(validPayload()));
    const names: string[] = [];
    for await (const batch of containerBatches(opened, GRAPH_NAME)) {
      for await (const item of batch.items) names.push(item.name);
    }
    assertEquals(names.sort(), ["b", "w"]);
  });
});
