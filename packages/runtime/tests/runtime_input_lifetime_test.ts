// run / enqueue の入力の寿命契約（発行時 metadata snapshot + borrowed な data）の門。
//
// run 本体は `#serialize` 越し（1 マイクロタスク以降）に走るので、JSDoc が勧める非 await の
// 並行発行と「発行直後の書き換え」を組むと、利用者の Record / shape 配列 / bindings が本体から
// 見えてしまう。実装はそれを断つために**発行の同期区間で metadata だけ写す**ので、この門が
// 固定するのは ①shape の書き換えが束縛解決に届かない ②Record の member 差し替えが届かない
// ③bindings の書き換えが届かない ④それでも `Tensor.data` は**借りたまま**（= 契約どおり
// clone しない）の 4 点と、⑤enqueue が同型であること。
//
// ①〜③ が無いと、退行は「たまたま fail loudly」か「沈黙誤値」のどちらかに割れる形で戻る
// （どちらも run の戻り値の見た目は正常）。④ は負の門で、契約を clone 側へ倒したら赤くなる。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext, type ResidentTensor } from "../src/gpu/device.ts";
import {
  createSession,
  type RunInput,
  type Session,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;
const BYTES = COUNT * 4;

/**
 * y = x + x。行数を記号次元 `N` にしてあるのは、**bindings と入力 shape の両方が束縛解決の
 * 材料**になる形（= 発行後の書き換えが「束縛が衝突」という無関係な失敗に化ける形）を実際に
 * 通すため。
 */
const GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["add"] },
  symbols: ["N"],
  inputs: [{ name: "x", dtype: "f32", shape: ["N", COLS] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: ["N", COLS] } },
  nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
};

/**
 * `phase` ごとに値が変わる入力。`shape` を外から渡せるのは、**呼び出し側が握ったままの配列**を
 * 発行後に書き換える形を作るため（`Tensor.shape` の `readonly` はコンパイル時の話でしかなく、
 * 実体は利用者が持つ可変配列 — 契約が守るべきはこちら）。
 */
const input = (phase: number, shape: number[] = [ROWS, COLS]): Tensor => ({
  dtype: "f32",
  shape,
  data: Float32Array.from({ length: COUNT }, (_, i) => ((i + phase * 3) % 9 - 4) * 0.5),
});

/** 参照値（f32 で丸めながら手計算する — 実装とは独立）。 */
const doubled = (phase: number): Float32Array<ArrayBuffer> => {
  const x = input(phase).data;
  return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
};

/** ビット列比較（丸めの取り違えを許容しない）。常駐テンソルの読み戻しは素の ArrayBuffer。 */
const bits = (data: Tensor["data"] | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

const openSession = (gpu: GpuContext): Promise<Session> =>
  createSession(gpu, openModel(graphModelBuffer(GRAPH)));

Deno.test({
  name: "run 直後の shape 書き換えは解決 shape に届かない（発行時 snapshot・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      const shape = [ROWS, COLS];
      // await しない発行（run の JSDoc が勧める形）→ 同じ tick で shape を壊す。
      const pending = session.run({ x: input(0, shape) });
      shape[0] = ROWS * 2;

      const outputs = await pending;
      assertEquals(bits(outputs["y"].data), bits(doubled(0)), "発行時の shape で走っていない");
      // 解決 shape そのものの門: 要素数が合う書き換え方だと検査を素通りして、出力の**形だけ**が
      // 静かにずれる（値は正しく見える）。
      assertEquals(outputs["y"].shape, [ROWS, COLS], "解決 shape が発行後の書き換えで動いた");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "run 直後の inputs member 差し替えは実行に届かない（発行時 snapshot・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      // 恒真化の門: 2 つの phase の期待値が同じなら取り違えを検出できない。
      assert(
        JSON.stringify(bits(doubled(0))) !== JSON.stringify(bits(doubled(1))),
        "phase ごとに期待値が変わっていない（検出器として空振る）",
      );
      const inputs: Record<string, RunInput> = { x: input(0) };
      const pending = session.run(inputs);
      inputs["x"] = input(1);

      const outputs = await pending;
      assertEquals(bits(outputs["y"].data), bits(doubled(0)), "差し替え後の Tensor が実行された");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "run 直後の bindings 書き換えは束縛衝突を起こさない（発行時 snapshot・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      // 入力 shape からも解ける記号を明示指定する形（= 書き換えると本体で「N の束縛が衝突」に
      // なる形）。発行時に写していれば、この run は最後まで N=ROWS で走る。
      const bindings: Record<string, number> = { N: ROWS };
      const pending = session.run({ x: input(0) }, bindings);
      bindings["N"] = ROWS + 1;

      const outputs = await pending;
      assertEquals(bits(outputs["y"].data), bits(doubled(0)));
      assertEquals(outputs["y"].shape, [ROWS, COLS], "発行後の束縛で解決 shape が動いた");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "run 直後の data 書き換えは実行に届く（borrowed の負の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      const tensor = input(0);
      const pending = session.run({ x: tensor });
      // MUST NOT（利用者側の契約）: 戻り Promise が settle する前の書き換え。ここでは契約違反を
      // **わざと**踏んで、ランタイムが `data` を clone しないことを固定する。
      tensor.data.fill(0);

      const outputs = await pending;
      // これは仕様（metadata だけ写し、data は借りる — GiB 級の複製を毎 run 払わないため）。
      // 契約を clone へ変えたらこのケースが赤くなる = 契約変更を黙って入れられない。
      assertEquals(
        bits(outputs["y"].data),
        bits(new Float32Array(COUNT)),
        "data が clone されている（借用契約が変わった — 仕様変更なら ADR とこの門を同時に直す）",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "enqueue も発行時 snapshot で走る（inputs / shape / copyOutputs・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    const sink = await gpu.createResident(BYTES, "sink");
    const decoy = await gpu.createResident(BYTES, "decoy");
    try {
      // decoy には既知の値を先に入れておく（写し先の差し替えが通ったら**ここが上書きされる**
      // ので、0 初期化に頼らず検出できる）。
      decoy.write(input(2).data);
      const shape = [ROWS, COLS];
      const inputs: Record<string, RunInput> = { x: input(0, shape) };
      const copyOutputs: Record<string, ResidentTensor> = { y: sink };
      const batch = await gpu.beginBatch();
      // enqueue は非 await 発行が前提の面（戻り Promise を待たずに finish を呼べる）なので、
      // 発行直後の書き換えは run より踏みやすい。
      const pending = session.enqueue(inputs, { batch, copyOutputs });
      inputs["x"] = input(1);
      shape[0] = ROWS * 2;
      copyOutputs["y"] = decoy;

      await pending;
      await batch.finish();
      assertEquals(bits(await sink.read()), bits(doubled(0)), "発行後の差し替えが実行に届いた");
      assertEquals(bits(await decoy.read()), bits(input(2).data), "写し先の差し替えが実行に届いた");
    } finally {
      await session.dispose();
      sink.dispose();
      decoy.dispose();
      gpu.destroy();
    }
  },
});
