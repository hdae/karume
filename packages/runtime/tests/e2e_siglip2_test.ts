// 実重み SigLIP2 の **vision tower** の実 GPU golden E2E（ADR 0005 の段 3）。
//
// tiny golden（tests/e2e_golden_test.ts）が「op 契約の被覆」を、実重み SBV2 / Irodori が
// 「音響チェーン側の実重み」を、EmbeddingGemma（tests/e2e_embeddinggemma_test.ts）が
// 「単一ベクトル出力のテキスト系」を受け持つのに対し、こちらは**単一ベクトル出力の画像系**
// を受け持つ。対象は `outputs/series/<系列>/`（重み 350MB〜1.7GB のためリポジトリ管理外 —
// `.gitignore` の `outputs/`）。生成は `tools/exporter/export_siglip2.py`（コマンドは下の
// generateCommand がそのまま正本）。
//
// 系列は **モデルごとに 1 本**（下の SERIES）。base（`patch16-224` — hidden 768 / 12 層 /
// 196 パッチ）と so400m（`patch14-384` — hidden 1152 / 27 層 / 729 パッチ）は同じ経路の
// 大小 2 点で、グラフの形も op 表も同じ。**tolerance だけは系列ごとに実測から導く**
// （片方の実測で他方を通さない — 桁が違えば誤差も違う）。格納 dtype 系列は f32 の 1 本
// のみなので、そちらの軸は持たない。
//
// グラフは vision tower 1 本で、出力も **pooler_output（MAP head 経由の `[1,hidden]`）
// 1 本だけ**（text tower も `last_hidden_state` も載っていない — export_siglip2.py の
// docstring）。入力は**正規化済みの** `pixel_values f32 [1,3,解像度,解像度]` 1 本で、記号
// 次元は無い（解像度もパッチ数も固定）。画像の前処理（resize / rescale / normalize）はまだ
// karume 側に無いので、golden の入力は実画像ではなく合成画像（`build_cases`）。
//
// 資産が無い環境では**明示 SKIP**する（系列ごとに独立 — 片方だけ生成した環境でも、ある方は
// 実走する）。ADR 0005 の「全 SKIP は明示 FAIL」門番（tests/gpu_gate_test.ts）は *GPU
// アダプタの有無* だけを見ており、この SKIP とは独立。逆に資産が**中途半端に**（ケース欠け）
// 存在する場合は SKIP ではなく FAIL にする（下の「資産の完全性」テスト）— そこは無音の
// 見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  createSession,
  type KarumeModel,
  openModel,
  parseSafetensors,
  type SafetensorsFile,
  type Tensor,
} from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * 実重み SigLIP2 **base**（`patch16-224`）の torch CPU 期待値との突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,768]`）:
 *
 * | ケース   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | ------- | ------- | ------------ | ------------------ |
 * | checker  | 1.05e-5 | 7.45e-3 | 6.842        | 2.04e-4            |
 * | noise    | 9.54e-6 | 6.11e-3 | 7.147        | 1.02e-4            |
 * | ramp     | 6.44e-6 | 4.13e-3 | 6.909        | 1.70e-4            |
 * | ramp-dim | 1.05e-5 | 3.03e-3 | 6.950        | 2.36e-4            |
 *
 * **判定は atol が主導する**（`|x−y| ≤ atol + rtol·|ref|`）: pooler_output は L2 正規化されて
 * いない生のベクトルで、値域は 0 を跨いで \|ref\| 最小非ゼロ 1.02e-4 まで薄く広がるため、
 * rtol を主役にすると 0 近傍で発散する（実測 maxRel 7.45e-3 の要素も絶対誤差は
 * `7.45e-3 × 1.4e-3 ≈ 1.0e-5` でしかない）。rtol 1e-6 の寄与は \|ref\| 上端 7.147 でも
 * 7.1e-6 で、atol の 1/7 と判定を主導しない。
 *
 * atol 5e-5 は実測最悪 1.05e-5（checker / ramp-dim）の約 4.8 倍。**EmbeddingGemma の
 * `EMBEDDINGGEMMA_TOLERANCE`（atol 1e-6）より 2 桁緩いのは値域の違いだけ** — あちらの出力は
 * L2 正規化済みの単位ベクトル（\|ref\| 上端 0.25）で、値域で割った相対量はどちらも 1.5e-6 級で
 * 揃っている（1.05e-5 / 7.15 ≈ 1.5e-6 対 2.8e-7 / 0.25 ≈ 1.1e-6）。誤差の出所も同じ（fma 融合・
 * linear / attention の縮約順序が torch と違う・超越関数（tanh / exp / rsqrt）の実装差）で、
 * SDPA は分解経路（mul×2 + bmm + softmax + bmm）を 13 本とも通っている。
 *
 * NOTE: MAP head の q/k/v 明示化パッチ由来の差（`export_siglip2.py --verify` の実測
 * 7.75e-7〜2.38e-6）は**ここには入らない** — golden の期待値は差し替え**後**のモジュールから
 * 採っており、この突合はランタイム側の誤差だけを見ている。
 *
 * 実装バグ（パッチ埋め込みのチャネル取り違え・位置埋め込みの並び違い・attention の head 分割
 * 誤り・MAP head の probe 取り違え）の誤差は出力の値域と同じ O(1)〜O(7) で、この閾値の
 * 5 桁以上上に出る。
 */
const BASE_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 1e-6 };

/**
 * 実重み SigLIP2 **so400m**（`patch14-384`）の torch CPU 期待値との突合に使う許容誤差。
 *
 * MUST: base と**独立に**実測から導く（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本
 * `[1,1152]`）。同じ経路でも hidden 1152 / 27 層 / 729 パッチと縮約の長さが違い、誤差の
 * 積み上がり方も違う:
 *
 * | ケース   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | ------- | ------- | ------------ | ------------------ |
 * | checker  | 8.46e-6 | 4.64e-3 | 10.389       | 1.00e-4            |
 * | noise    | 1.05e-5 | 9.76e-3 | 8.451        | 1.28e-4            |
 * | ramp     | 6.68e-6 | 2.55e-2 | 10.753       | 4.68e-6            |
 * | ramp-dim | 6.08e-6 | 8.35e-3 | 10.874       | 1.89e-4            |
 *
 * atol 5e-5 は実測最悪 1.05e-5（noise）の約 4.8 倍。**base と同じ値になったのは実測の帰結**
 * で、共有はしない（片方を測り直したときにもう片方が黙って緩む）。判定を atol が主導する
 * 理由も base と同じ — \|ref\| 最小非ゼロが 4.68e-6（ramp）まで薄く広がり、maxRel 2.55e-2 の
 * 要素も絶対誤差は 1.2e-7 でしかない。rtol 1e-6 の寄与は \|ref\| 上端 10.874 でも 1.1e-5 で、
 * atol の 1/5 と判定を主導しない。
 *
 * NOTE: MAP head の q/k/v 明示化パッチ由来の差（`--verify` の so400m 実測
 * 1.07e-6〜2.86e-6・形の畳み込みは 4 ケースともビット同一）は base と同じくここには入らない。
 */
const SO400M_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 1e-6 };

/** 系列（= モデル）1 本の宣言。系列名は exporter の `--model-dir` のディレクトリ名。 */
type Series = {
  readonly name: string;
  readonly tolerance: Tolerance;
};

/**
 * 実走する系列。**列挙ではなくここで固定する** — 生成済みのものを拾う形にすると、
 * 系列ごと生成し忘れた環境で「緑だが未検証」になる。
 */
const SERIES: readonly Series[] = [
  { name: "siglip2-base-patch16-224", tolerance: BASE_TOLERANCE },
  { name: "siglip2-so400m-patch14-384", tolerance: SO400M_TOLERANCE },
];

const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

const seriesRoot = (series: Series): URL => new URL(`${series.name}/`, SERIES_PARENT);

/**
 * SKIP 時にそのまま貼れる生成コマンド（tools/exporter/export_siglip2.py の docstring）。
 * 系列名は実重みのディレクトリ名でもある（どちらも HF のリポジトリ名）。
 */
const generateCommand = (series: Series): string =>
  "cd tools/exporter && uv run --group siglip2 python export_siglip2.py" +
  ` --model-dir ../../inputs/siglip2/${series.name}`;

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する** — 列挙だけに頼ると生成を
 * 一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/exporter/export_siglip2.py` の `build_cases`。
 */
const EXPECTED_CASES = ["checker", "noise", "ramp", "ramp-dim"] as const;

/**
 * 判別検査で見る cosine の対（構造の近い対, 構造ごと違う対）。正本は `export_siglip2.py` の
 * `NEAR_PAIR` / `FAR_PAIR` で、あちらは torch 出力に、こちらは**実 GPU 出力**に掛ける。
 */
const NEAR_PAIR = ["ramp", "ramp-dim"] as const;
const FAR_PAIR = ["ramp", "checker"] as const;

/**
 * 資産ディレクトリの列挙。存在しない場合だけ空に縮退する。
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、
 * 実行されていない検証が SKIP として静かに緑になる。
 */
const listDir = (url: URL): readonly Deno.DirEntry[] => {
  try {
    return [...Deno.readDirSync(url)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) {
      return [];
    }
    throw cause;
  }
};

const discoverCases = (root: URL): readonly string[] =>
  listDir(root)
    .filter((entry) =>
      entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
    )
    .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
    .sort();

const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/** golden の入力を宣言 dtype の view で組む（記号次元が無いので明示 bindings も不要）。 */
const goldenInputs = (parsed: KarumeModel, io: SafetensorsFile): Record<string, Tensor> => {
  const inputs: Record<string, Tensor> = {};
  for (const spec of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${spec.name}`);
    assert(view !== undefined, `input.${spec.name} が golden に無い`);
    inputs[spec.name] = ioTensor(io, view, spec.dtype);
  }
  return inputs;
};

const cosine = (first: Float32Array, second: Float32Array): number => {
  let dot = 0;
  let firstNorm = 0;
  let secondNorm = 0;
  for (let i = 0; i < first.length; i += 1) {
    dot += first[i] * second[i];
    firstNorm += first[i] * first[i];
    secondNorm += second[i] * second[i];
  }
  return dot / Math.sqrt(firstNorm * secondNorm);
};

for (const series of SERIES) {
  const root = seriesRoot(series);
  /** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
  const cases = discoverCases(root);
  /** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
  const available = cases.length > 0;

  if (!available) {
    console.warn(
      `[karume] ${root.pathname} に export 済み資産が無いため実重み SigLIP2 vision E2E ` +
        `(${series.name}) を SKIP する（重みがリポジトリ管理外）。` +
        `生成: ${generateCommand(series)}`,
    );
  }

  Deno.test({
    name: `SigLIP2 資産: ${series.name} — 期待するケースとモデル本体が揃っている`,
    // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
    ignore: !available,
    fn: () => {
      assertEquals(cases, [...EXPECTED_CASES], `${root.pathname} の golden ケース`);
      const model = new URL(MODEL_FILE, root);
      assert(Deno.statSync(model).isFile, `${MODEL_FILE} が無い`);
    },
  });

  for (const caseName of cases) {
    Deno.test({
      name: `SigLIP2 golden 突合: ${series.name} / ${caseName}（実 GPU / torch CPU 期待値）`,
      ignore: !available || !GPU_AVAILABLE,
      fn: async () => {
        const [modelBytes, ioBytes] = await Promise.all([
          readBuffer(root, MODEL_FILE),
          readBuffer(root, `${IO_PREFIX}${caseName}${IO_SUFFIX}`),
        ]);
        const parsed = openModel(modelBytes);
        const io = parseSafetensors(ioBytes);

        // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
        const expectedKeys = [
          ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
          ...parsed.graph.outputs.map((_, index) => `output.${index}`),
        ].sort();
        assertEquals([...io.tensors.keys()].sort(), expectedKeys, "io.safetensors のテンソルキー");

        const inputs = goldenInputs(parsed, io);

        const gpu = await acquireGpu();
        const session = await createSession(gpu, parsed);
        try {
          const outputs = await session.run(inputs);
          assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

          parsed.graph.outputs.forEach((name, index) => {
            const view = io.tensors.get(`output.${index}`);
            assert(view !== undefined, `output.${index} が golden に無い`);
            const where = `${series.name} / ${caseName} output.${index} ('${name}')`;
            const declared = parsed.graph.values[name].dtype;
            assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
            assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
            const report = compareTensors(
              outputs[name],
              ioTensor(io, view, declared),
              series.tolerance,
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

  Deno.test({
    name: `SigLIP2 判別: ${series.name} — 実 GPU 出力の cosine が構造の近さの順に並ぶ`,
    ignore: !available || !GPU_AVAILABLE,
    fn: async () => {
      // golden 突合だけだと「期待値と合っている」ことしか言えず、埋め込みとして意味のある
      // 出力かは別問題（1 点へ潰れた出力は期待値も同じく潰れていれば通ってしまう）。ここは
      // **別々の入力どうしの順序**を見るので、出力がケース間で定数なら cosine が全て 1 に
      // なって落ちる。閾値は置かない（順序そのものが検査対象 — export_siglip2.py の `_sanity`
      // と同じ形で、あちらは torch 側に掛かっている）。
      const modelBytes = await readBuffer(root, MODEL_FILE);
      const parsed = openModel(modelBytes);
      const [outputName] = parsed.graph.outputs;

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      const pooled = new Map<string, Float32Array>();
      try {
        // 4 ケースを 1 Session で回す（重みは 350MB〜1.7GB — ケースごとに組み直す理由が無い）。
        for (const caseName of cases) {
          const io = parseSafetensors(
            await readBuffer(root, `${IO_PREFIX}${caseName}${IO_SUFFIX}`),
          );
          const output = (await session.run(goldenInputs(parsed, io)))[outputName];
          // 判別子で絞る（Float32Array へのキャストは dtype がずれたときに黙って通る）。
          assert(output.dtype === "f32", `${caseName}: pooler_output の dtype が ${output.dtype}`);
          pooled.set(caseName, output.data);
        }
      } finally {
        await session.dispose();
        gpu.destroy();
      }

      const cosineOf = (pair: readonly [string, string]): number => {
        const [first, second] = pair.map((name) => pooled.get(name));
        assert(first !== undefined && second !== undefined, `${pair.join("×")} の出力が無い`);
        return cosine(first, second);
      };
      const near = cosineOf(NEAR_PAIR);
      const far = cosineOf(FAR_PAIR);
      assert(
        near > far,
        `cosine の順序が構造と逆: ${NEAR_PAIR.join("×")}=${near} <= ${FAR_PAIR.join("×")}=${far}`,
      );
    },
  });
}
