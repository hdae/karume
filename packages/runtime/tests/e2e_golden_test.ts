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
import {
  acquireGpu,
  capabilities,
  type OpenedContainer,
  prepareContainer,
  type Tensor,
} from "../mod.ts";
import { parseSafetensors } from "../src/format/safetensors.ts";
import {
  compareTensors,
  EXACT_TOLERANCE,
  formatAllclose,
  type Tolerance,
} from "../src/reference/allclose.ts";
import { assertAdapterMatchesEnvironment, ENVIRONMENT } from "./helpers/environment.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { openResults, runRecordedCase } from "./helpers/results.ts";
import { openSeriesContainer } from "./helpers/container-files.ts";

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

/** golden 1 件の容器の代表 path（`goldens.py` の `MODEL_FILE`）。 */
const MODEL_FILE = "model.krm";

/**
 * golden 1 件の容器を開く。golden は全て分割形で置かれる（const が空なら part 1 は 0 バイトの
 * ファイル — container-v1 §8）。見つけ方は `resolveParts`。
 *
 * **グラフ名は置き場のディレクトリ名**（`goldens.py` の `graph_name=spec.name`）。
 */
const openGolden = async (model: string): Promise<OpenedContainer> =>
  await openSeriesContainer(new URL(`${model}/${MODEL_FILE}`, GOLDEN_ROOT));

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
    const graph = prepareContainer(await openGolden(model), model).graph;
    for (const op of graph.requires.ops) covered.add(op);
  }
  const uncovered = capabilities().ops.filter((op) => !covered.has(op));
  assertEquals(
    uncovered,
    OPS_WITHOUT_GOLDEN,
    "golden が 1 本も踏んでいない op（op を足したら golden も足す / 出せない op は理由つきで OPS_WITHOUT_GOLDEN へ）",
  );
});

/**
 * 環境キーの綴り（`environmentKey` が作る形 = `<ランタイム>-<英小文字・数字の語をハイフン 1 本で
 * つないだ slug>`）。
 */
const ENVIRONMENT_KEY_PATTERN = /^(deno|chrome)-[a-z0-9]+(-[a-z0-9]+)*$/;

// 仕様帯の行は**その機で 1 段目を落ちた出力**でしか引かれないので、孤児になった行（golden の削除・
// 再 export での出力名の改名）や環境キーの綴り違いは「行が無い機」と同じ扱いになり、黙って残る。
// 行が実体を指していることを、GPU アダプタに依らずここで固定する（ADR 0106 追記決定 1 の台帳を
// 実体とずらさない）。
Deno.test("OUTPUT_TOLERANCE の各行は実在する golden の出力と正しい綴りの環境キーを指す", async () => {
  for (const [key, byEnvironment] of Object.entries(OUTPUT_TOLERANCE)) {
    const slash = key.indexOf("/");
    assert(slash > 0, `${key}: キーが <model>/<出力名> の形でない`);
    const model = key.slice(0, slash);
    const output = key.slice(slash + 1);
    assert(MODELS.includes(model), `${key}: golden '${model}' が ${GOLDEN_ROOT.pathname} に無い`);
    const outputs = prepareContainer(await openGolden(model), model).graph.outputs;
    const listed = outputs.join(", ");
    assert(
      outputs.includes(output),
      `${key}: 出力 '${output}' が golden '${model}' の graph.outputs（${listed}）に無い`,
    );
    for (const [environment, { spec }] of Object.entries(byEnvironment)) {
      assert(
        ENVIRONMENT_KEY_PATTERN.test(environment),
        `${key}: 環境キー '${environment}' が ${ENVIRONMENT_KEY_PATTERN} に一致しない`,
      );
      for (const [name, value] of Object.entries(spec)) {
        assert(
          Number.isFinite(value) && value >= 0,
          `${key} / ${environment}: ${name} = ${value} は有限の非負数でない`,
        );
      }
    }
  }
});

for (const model of MODELS) {
  Deno.test({
    name: `golden 突合: ${model}（実 GPU / torch CPU 期待値）`,
    ignore: !GPU_AVAILABLE,
    fn: async () => {
      /** Karume 独自基準を超えたが仕様帯で受理した出力名（結果 JSON の note になる）。 */
      const accepted: string[] = [];
      /** 仕様帯でも受からなかった出力のメッセージ（1 本目でテストを落とす）。 */
      const failures: string[] = [];
      /** 仕様帯で受理した出力名の note（無ければ持たない — 数値は measurements が持つ）。 */
      const acceptedNote = (): string | undefined =>
        accepted.length === 0 ? undefined : `仕様帯で受理: ${accepted.join(", ")}`;
      await runRecordedCase(results, {
        id: model,
        // 許容差以外の失敗（資産の読み・createSession・出力キー・shape / dtype・run の例外）も
        // 席に残す。記録が落ちても元の例外（何が壊れたかを言う唯一の診断）は置き換えない。
        failureNote: (cause) =>
          [
            acceptedNote(),
            `例外: ${cause instanceof Error ? cause.message : String(cause)}`,
          ].filter((part) => part !== undefined).join("; "),
      }, async ({ measurements }) => {
        const [opened, ioBytes] = await Promise.all([
          openGolden(model),
          readBuffer(model, "io.safetensors"),
        ]);
        const parsed = prepareContainer(opened, model);
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
        // MUST: device の破棄は adapter 検査と `createSession` の失敗も通す。取り逃がすと、
        // その走行の残りが破棄されない device を抱えたまま進み、後続が OOM で赤くなる
        // （known-issues の遅延解放）。
        try {
          // 参照値・結果をこの機の行として残す経路なので、キーを採ったアダプタと実行アダプタの
          // 同一性をここで見る（複数 GPU の機で取り違えると、別の機の帯で測ることになる）。
          assertAdapterMatchesEnvironment(gpu);
          const session = await parsed.createContainerSession(gpu);
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
              // f32 は allclose、i32 / bool は厳密一致（整数演算に近似の余地は無い）。
              // MUST: `compareTensors` は f32 以外で引数の帯を捨てて {@link EXACT_TOLERANCE} で
              // 測るので、**記録に載せる帯も同じ選び方**でここ 1 か所から採る（`measurements`
              // の `tolerance` は「受理に使った帯」— 存在しない余裕を読み手に見せない）。
              const band = (declaredBand: Tolerance): Tolerance =>
                declared === "f32" ? declaredBand : EXACT_TOLERANCE;
              const karumeBand = band(GOLDEN_TOLERANCE);
              const karume = compareTensors(outputs[name], expected, karumeBand);
              // 1 段目を落ちた出力だけが 2 段目（WGSL 仕様帯）へ来る。受かれば pass + warning。
              const spec = karume.pass ? undefined : specTolerance(model, name);
              if (spec === undefined) {
                // 2 段目が無い（1 段目で受かった / この機に行が無い）= 1 段目が決着の段。
                measurements.push({
                  output: name,
                  maxAbs: karume.maxAbsError,
                  maxRel: karume.maxRelError,
                  tolerance: karumeBand,
                  stage: "karume",
                });
                if (!karume.pass) failures.push(`${where}: ${formatAllclose(karume)}`);
                return;
              }
              const specBand = band(spec);
              const report = compareTensors(outputs[name], expected, specBand);
              // 実測（maxAbs / maxRel）は帯に依らないので 2 段目の報告をそのまま採る。
              measurements.push({
                output: name,
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance: specBand,
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
          }
        } finally {
          gpu.destroy();
        }
        // 決着は投げる前に残す（赤で終わった回の note も手元に要る）。
        return { status: failures.length === 0 ? "pass" : "fail", note: acceptedNote() };
      });
      assert(failures.length === 0, failures[0]);
    },
  });
}
