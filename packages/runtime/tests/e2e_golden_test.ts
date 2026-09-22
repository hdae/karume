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
import { acquireGpu, capabilities, prepareModel, type Tensor } from "../mod.ts";
import { parseSafetensors } from "../src/format/safetensors.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ENVIRONMENT } from "./helpers/environment.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { type Measurement, openResults } from "./helpers/results.ts";
import { readShard, resolveShards, streamShards } from "./helpers/shard-files.ts";

/**
 * torch CPU 期待値との突合に使う許容誤差 = **判定の 1 段目**（Karume 独自基準・全出力共通）。
 * ここを超えただけでは赤にせず、2 段目の {@link OUTPUT_TOLERANCE}（WGSL 仕様帯）で受け止めて
 * warning として記録する。
 *
 * 実測（**全 32 モデル 76 出力**のうち f32 が 67 本。i32 / bool の 9 本は差 0 の厳密一致で、
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

/**
 * 出力ごとの **WGSL 仕様帯** = 判定の 2 段目。外側のキーは `<model>/<出力名>`、内側は
 * **環境キー**（`<ランタイム>-<アダプタ名 slug>` — ADR 0106 決定 2）。**走らせている機の行がある
 * 出力だけ**、{@link GOLDEN_TOLERANCE} を超えても赤にならずに済む。
 *
 * 行を環境キーごとに持つのは、**緩めを足した機の外へ緩めを広げない**ため。仕様帯が要るのは
 * 「この GPU の実装がそこまで外れる」と実測で分かった出力だけで、同じ op を仕様どおりほぼ
 * 正しく丸める機もある。行を全機共通にすると、そちらの機の退行検出の網まで同じだけ緩む。
 * **行が無い機では 2 段目そのものが無い**: 1 段目の独自基準だけで測り、超えれば赤になる。
 *
 * 2 段の判定（裁定 2026-09-20）:
 *
 * 1. Karume 独自基準（{@link GOLDEN_TOLERANCE}・全出力共通）で通れば **pass**。
 * 2. 落ちた出力にこの機の `spec` があればそれで測り直し、通れば **pass + warning** — 結果 JSON
 *    （`outputs/verify/<環境キー>/<日付>_golden/results.json`）の `note` に
 *    「どの出力を仕様帯で受理したか」が、`measurements` にその実測と受理に使った帯が残る。
 * 3. この機の `spec` が無い / `spec` でも落ちるなら **fail**（メッセージは従来どおり）。
 *
 * 独自基準は「従来この値で通っていた」を見失わないための目安であって仕様上の根拠は無く、
 * **容易に撤廃してよい**（実装バグを掴む網は仕様帯の側にある — op 取り違え・添字ずれの誤差は
 * O(1) で、仕様帯からも 4 桁上に出る）。
 *
 * MUST: ここへ行を足すのは **op 単位・WGSL 仕様の精度保証の範囲内・実害が無い場合**に限り、
 * 根拠（仕様の該当節と実測値）を行ごとに書く。全体を一度に緩めない。
 *
 * - `activations/sin` の `deno-intel-graphics-bmg-g21`: WGSL 仕様の `sin(x)` は |x| ≤ π で
 *   **絶対誤差 2⁻¹¹ まで**を許す（§ Accuracy of Concrete Floating Point Expressions）。golden の
 *   入力は [−1.28, 1.89] でこの区間の内側。Intel Arc B570（Mesa ANV）は x = 1.5908 で 2.68e-5
 *   （2026-09-20 実測 — 仕様の内・実装バグの O(1) からは 4 桁下）。NVIDIA（RTX 3080 Ti）は
 *   ほぼ正しく丸めるので 1e-6 で通り、行を持たない。
 */
const OUTPUT_TOLERANCE: Readonly<
  Record<string, Readonly<Record<string, { readonly spec: Tolerance }>>>
> = {
  "activations/sin": {
    "deno-intel-graphics-bmg-g21": { spec: { atol: 2 ** -11, rtol: 0 } },
  },
};

/**
 * この走行の機に対する仕様帯（無ければ `undefined` = 2 段目が無い）。
 *
 * 環境キーが無いのは GPU アダプタが取れない機だけで、そこでは golden テスト自体が登録時点で
 * SKIP される（それでも行を引けないことに変わりはないので `undefined` を返す）。
 */
const specTolerance = (model: string, output: string): Tolerance | undefined => {
  const environment = ENVIRONMENT.key;
  if (environment === undefined) return undefined;
  const key = `${model}/${output}`;
  if (!Object.hasOwn(OUTPUT_TOLERANCE, key)) return undefined;
  const byEnvironment = OUTPUT_TOLERANCE[key];
  if (!Object.hasOwn(byEnvironment, environment)) return undefined;
  return byEnvironment[environment].spec;
};

/** 決着と warning の置き場（`outputs/verify/<環境キー>/<日付>_golden/` — 消して安全）。 */
const results = openResults("golden");

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

/**
 * golden 1 件の配布形 shard 列（先頭がグラフ shard — ADR 0081）。テンソルを 1 本も持たない
 * spec は分割されないので `model.safetensors` 1 本のまま来る（見つけ方はどちらも同じ）。
 */
const modelShards = (model: string): readonly URL[] =>
  resolveShards(new URL(`${model}/model.safetensors`, GOLDEN_ROOT));

Deno.test("golden fixtures が 1 件以上あり、全件がテストとして登録される", () => {
  // 列挙が空でも「テストが 0 本で緑」になるだけなので、ここで下限を固定する（ADR 0005）。
  assert(MODELS.length > 0, `${GOLDEN_ROOT.pathname} に golden モデルが 1 件も無い`);
});

// 被覆検査は生成側（tools/exporter の goldens.py）にもあるが、あちらは**エクスポータが
// emit しうる op**との突合で、fixture を再生成しない限り走らない。ランタイム側が op を
// 足したのに golden を足していない状態は、fixture を触らないので生成側では検出できない。
// ここは**実行できる op（capabilities()）が golden に 1 本も現れていない**ことを、
// 配布形そのものを読んで固定する（GPU アダプタ非依存 — 突合ではなく宣言の集合演算）。
/**
 * golden を持てない op（Python 側 `NON_EMITTABLE_OPS` の鏡像 — 理由つき列挙 MUST）。
 *
 * `topk` は契約表にあるが **torch から出せない**（ADR 0068 追記）: aten ハンドラが無く、
 * 多出力 aten のタプル meta + `operator.getitem` のスロット結線が新機構として残っている
 * （sampling の実需まで先送り — 実測で止まるのは `aten.topk.default` ではなく getitem）。
 * `state_append` も torch から出せない（ADR 0067 決定 5）: **aten に対応物が無い** effect op で、
 * 発行するのは decode グラフ台本（実装波）だけ。加えて state スロットの実体は
 * GenerationContext が持つので、1-shot の golden 配布形では実行そのものが成立しない。
 * MUST: 下の突合は**両方向**（この列挙と実際の未被覆集合が完全一致）で見る。golden が topk を
 * 踏み始めた日にもここが赤くなり、席を外すことを強制する。
 */
const OPS_WITHOUT_GOLDEN: readonly string[] = ["state_append", "topk"];

Deno.test("全 golden の requires.ops が実行可能な op 集合を覆う", async () => {
  const covered = new Set<string>();
  for (const model of MODELS) {
    const graph = prepareModel(await readShard(modelShards(model)[0])).graph;
    for (const op of graph.requires.ops) covered.add(op);
  }
  const uncovered = capabilities().ops.filter((op) => !covered.has(op));
  assertEquals(
    uncovered,
    OPS_WITHOUT_GOLDEN,
    "golden が 1 本も踏んでいない op（op を足したら golden も足す / 出せない op は理由つきで OPS_WITHOUT_GOLDEN へ）",
  );
});

for (const model of MODELS) {
  Deno.test({
    name: `golden 突合: ${model}（実 GPU / torch CPU 期待値）`,
    ignore: !GPU_AVAILABLE,
    fn: async () => {
      const startedAt = performance.now();
      /** Karume 独自基準を超えたが仕様帯で受理した出力名（結果 JSON の note になる）。 */
      const accepted: string[] = [];
      /** 出力ごとの実測（合格した回も残す — 判定には使わない）。 */
      const measurements: Measurement[] = [];
      /** 仕様帯でも受からなかった出力のメッセージ（1 本目でテストを落とす）。 */
      const failures: string[] = [];
      try {
        const shards = modelShards(model);
        const [graphShard, ioBytes] = await Promise.all([
          readShard(shards[0]),
          readBuffer(model, "io.safetensors"),
        ]);
        const parsed = prepareModel(graphShard);
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
        const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
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
            const expected = ioTensor(io, view, declared);
            // f32 は allclose、i32 / bool は厳密一致（整数演算に近似の余地は無い）
            const karume = compareTensors(outputs[name], expected, GOLDEN_TOLERANCE);
            // 1 段目を落ちた出力だけが 2 段目（WGSL 仕様帯）へ来る。受かれば pass + warning。
            const spec = karume.pass ? undefined : specTolerance(model, name);
            if (spec === undefined) {
              // 2 段目が無い（1 段目で受かった / この機に行が無い）= 1 段目が決着の段。
              measurements.push({
                output: name,
                maxAbs: karume.maxAbsError,
                maxRel: karume.maxRelError,
                tolerance: GOLDEN_TOLERANCE,
                stage: "karume",
              });
              if (!karume.pass) failures.push(`${where}: ${formatAllclose(karume)}`);
              return;
            }
            const report = compareTensors(outputs[name], expected, spec);
            // 実測（maxAbs / maxRel）は帯に依らないので 2 段目の報告をそのまま採る。
            measurements.push({
              output: name,
              maxAbs: report.maxAbsError,
              maxRel: report.maxRelError,
              tolerance: spec,
              stage: "spec",
            });
            if (!report.pass) {
              failures.push(`${where}: ${formatAllclose(report)}`);
              return;
            }
            // 数値は measurements が持つので、note は受理した出力名だけにする（二重に持たない）。
            accepted.push(name);
          });
        } finally {
          await session.dispose();
          gpu.destroy();
        }
      } catch (cause) {
        // 許容差以外の失敗（資産の読み・createSession・出力キー・shape / dtype・run の例外）も
        // 席に残す。決着の無いまま抜けると、この席には前回の走行の results.json が居座る。
        await results.record({
          id: model,
          status: "fail",
          elapsedMs: Math.round(performance.now() - startedAt),
          note: [
            ...(accepted.length === 0 ? [] : [`仕様帯で受理: ${accepted.join(", ")}`]),
            `例外: ${cause instanceof Error ? cause.message : String(cause)}`,
          ].join("; "),
          measurements,
        });
        throw cause;
      }
      // 決着は投げる前に残す（赤で終わった回の note も手元に要る）。
      await results.record({
        id: model,
        status: failures.length === 0 ? "pass" : "fail",
        elapsedMs: Math.round(performance.now() - startedAt),
        ...(accepted.length === 0 ? {} : { note: `仕様帯で受理: ${accepted.join(", ")}` }),
        measurements,
      });
      assert(failures.length === 0, failures[0]);
    },
  });
}
