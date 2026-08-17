// torch 由来 golden fixtures の実 GPU 突合（ADR 0005 の段 3）。エクスポート → 推論の貫通を
// 実証する常設テストで、tolerance の根拠づけもここが正本。
//
// 対象は tests/fixtures/golden/ 配下の**全ディレクトリを列挙**して決める。モデルを足したら
// 自動で対象になる（列挙をハードコードすると、golden を足したのにテストが増えない
// 「無音の見かけ成功」ができる）。
//
// ランタイム面は公開 API（mod.ts）だけで呼ぶ。io.safetensors の読み出しは配布形ではなく
// テスト側のフィクスチャ読みなので src/format/safetensors.ts を直に使う。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, capabilities, createSession, openModel, type Tensor } from "../mod.ts";
import { parseSafetensors } from "../src/format/safetensors.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * torch CPU 期待値との突合に使う許容誤差。
 *
 * 実測（**全 31 モデル 73 出力**のうち f32 が 64 本。i32 / bool の 9 本は差 0 の厳密一致で、
 * この値の対象外 — ADR 0009。`argmax_pick` の 2 出力は添字なのでこちら側）の最悪値は
 * **maxAbs 4.77e-7**（deform_conv2d_block output.0）/
 * **maxRel 1.77e-5**（batch_matmul output.0）。前者は**その要素の値が 4.82 と他の golden より
 * 大きいだけで、ulp で測れば 1 ulp ちょうど**（ulp(4.82) = 2² · 2⁻²³ = 4.77e-7）。後者は
 * ゼロ近傍の要素で相対誤差が伸びたもので、その要素の絶対誤差は 1.19e-7（= 1 ulp）にすぎない。
 * 出所は近似ではなく演算の並べ替えで、① GPU 側の fma 融合 ② matmul / 行 reduce の
 * 縮約順序が torch と違う ③ 超越関数の実装差（WGSL 仕様は数 ulp を許す）— いずれも 1 出力
 * あたり数 ulp の桁。
 *
 * **判定は atol が主導する**（`|x−y| ≤ atol + rtol·|ref|`）: maxRel が rtol を上回る要素も
 * 絶対誤差が atol の 1/8 に収まっているので通る。atol 1e-6 は実測最悪 4.77e-7 の約 2.1 倍で、
 * 縮約長とチェーン長が伸びる余地を見た余裕はここに集約されている。**値域が O(10) を超える
 * golden を足すなら atol ではなく rtol 側で受ける**（絶対誤差は値に比例するので、atol だけを
 * 上げると小さい出力の実装バグに対する網が同時に緩む）。実装バグ（op 取り違え・添字ずれ）の
 * 誤差は O(1) で、この閾値の 6 桁上に出る。
 *
 * **i8 格納の golden もこの 1 本で見る**（`i8_weights` — ADR 0019）。fake-quant が正なので
 * 期待値も丸め済み重みで計算されており、ここで測っているのは**実装誤差だけ**（実測
 * maxAbs 1.19e-7 / 8.94e-8 / 2.38e-7・maxRel 最大 1.83e-6 — f32 の他モデルと同じ桁で、
 * 最悪値をどちらも更新しない）。「量子化の質」はこの網の外の別軸。
 */
const GOLDEN_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 1e-5 };

const GOLDEN_ROOT = new URL("./fixtures/golden/", import.meta.url);

/** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
const goldenModels = (): readonly string[] =>
  [...Deno.readDirSync(GOLDEN_ROOT)]
    .filter((entry) => entry.isDirectory)
    .map((entry) => entry.name)
    .sort();

const MODELS = goldenModels();

const readBuffer = async (model: string, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(`${model}/${file}`, GOLDEN_ROOT));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

Deno.test("golden fixtures が 1 件以上あり、全件がテストとして登録される", () => {
  // 列挙が空でも「テストが 0 本で緑」になるだけなので、ここで下限を固定する（ADR 0005）。
  assert(MODELS.length > 0, `${GOLDEN_ROOT.pathname} に golden モデルが 1 件も無い`);
});

// 被覆検査は生成側（tools/exporter の goldens.py）にもあるが、あちらは**エクスポータが
// emit しうる op**との突合で、fixture を再生成しない限り走らない。ランタイム側が op を
// 足したのに golden を足していない状態は、fixture を触らないので生成側では検出できない。
// ここは**実行できる op（capabilities()）が golden に 1 本も現れていない**ことを、
// 配布形そのものを読んで固定する（GPU アダプタ非依存 — 突合ではなく宣言の集合演算）。
Deno.test("全 golden の requires.ops が実行可能な op 集合を覆う", async () => {
  const covered = new Set<string>();
  for (const model of MODELS) {
    const graph = openModel(await readBuffer(model, "model.safetensors")).graph;
    for (const op of graph.requires.ops) covered.add(op);
  }
  const uncovered = capabilities().ops.filter((op) => !covered.has(op));
  assertEquals(uncovered, [], "golden が 1 本も踏んでいない op（op を足したら golden も足す）");
});

for (const model of MODELS) {
  Deno.test({
    name: `golden 突合: ${model}（実 GPU / torch CPU 期待値）`,
    ignore: !GPU_AVAILABLE,
    fn: async () => {
      const [modelBytes, ioBytes] = await Promise.all([
        readBuffer(model, "model.safetensors"),
        readBuffer(model, "io.safetensors"),
      ]);
      const parsed = openModel(modelBytes);
      const io = parseSafetensors(ioBytes);

      // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
      // 命名規約は tools/exporter/README.md「golden レイアウト」が正本。
      const expectedKeys = [
        ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
        ...parsed.graph.outputs.map((_, index) => `output.${index}`),
      ].sort();
      assertEquals([...io.tensors.keys()].sort(), expectedKeys, "io.safetensors のテンソルキー");

      // 記号次元 T は golden の入力 shape の実長から束縛される（明示 bindings を渡さない）。
      const inputs: Record<string, Tensor> = {};
      for (const spec of parsed.graph.inputs) {
        const view = io.tensors.get(`input.${spec.name}`);
        assert(view !== undefined, `input.${spec.name} が io.safetensors に無い`);
        inputs[spec.name] = ioTensor(io, view, spec.dtype);
      }

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      try {
        const outputs = await session.run(inputs);
        assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

        parsed.graph.outputs.forEach((name, index) => {
          const view = io.tensors.get(`output.${index}`);
          assert(view !== undefined, `output.${index} が io.safetensors に無い`);
          const where = `${model} output.${index} ('${name}')`;
          const declared = parsed.graph.values[name].dtype;
          assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
          assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
          // f32 は allclose、i32 / bool は厳密一致（整数演算に近似の余地は無い）
          const report = compareTensors(
            outputs[name],
            ioTensor(io, view, declared),
            GOLDEN_TOLERANCE,
          );
          assert(report.pass, `${where}: ${formatAllclose(report)}`);
        });
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  });
}
