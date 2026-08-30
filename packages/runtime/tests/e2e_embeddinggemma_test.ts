// 実重み EmbeddingGemma-300m（文埋め込み）の実 GPU golden E2E（ADR 0005 の段 3）。
//
// tiny golden（tests/e2e_golden_test.ts）が「op 契約の被覆」を、実重み SBV2（tests/e2e_sbv2_test.ts）
// が「音響チェーン側の実重み」を受け持つのに対し、こちらは**単一ベクトル出力の実重みテキスト系**
// を受け持つ。対象は `outputs/series/embeddinggemma-300m/`（重み 1.2GB 級のためリポジトリ管理外
// — `.gitignore` の `outputs/`）。生成は `tools/exporter/export_embeddinggemma.py`（コマンドは下の
// GENERATE_COMMAND がそのまま正本）。
//
// SBV2 と違い格納 dtype 系列は f32 の 1 本のみ（f16 / i8 は別系列で決める話 — exporter 冒頭
// docstring）なので、系列パラメタ化はしない。グラフも target 分割が無く 1 本（Transformer →
// masked mean → Dense×2 → L2 正規化）で、ディレクトリ構造も
// `outputs/series/embeddinggemma-300m/{model,io.<case>}.safetensors` とフラット。
//
// 資産が無い環境では**明示 SKIP**する。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。
// 逆に資産が**中途半端に**（ケース欠け）存在する場合は SKIP ではなく FAIL にする
// （下の「資産の完全性」テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, parseSafetensors, prepareModel, type Tensor } from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { modelPresent, readShard, resolveShards, streamShards } from "./helpers/shard-files.ts";

/**
 * 実重み EmbeddingGemma-300m の torch CPU 期待値との突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、5 ケース × 出力 1 本 `[1,768]`）:
 *
 * | ケース        | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------------- | --- | ------- | ------- | ------------ | ------------------ |
 * | query-en      | 16  | 2.31e-7 | 3.91e-4 | 0.172        | 7.63e-5             |
 * | document-en   | 19  | 2.83e-7 | 8.41e-4 | 0.249        | 5.79e-5             |
 * | query-ja      | 16  | 2.83e-7 | 6.82e-4 | 0.181        | 9.37e-5             |
 * | bare          | 12  | 2.76e-7 | 1.65e-3 | 0.170        | 2.98e-5             |
 * | long-document | 318 | 3.87e-7 | 3.87e-4 | 0.156        | 6.64e-5             |
 *
 * **判定は tiny golden（`GOLDEN_TOLERANCE`）と同じく atol が主導する**（`|x−y| ≤ atol +
 * rtol·|ref|`）: 出力は L2 正規化済みの単位ベクトルで値域が 0 近傍まで薄く広がる（\|ref\|
 * 最小非ゼロが 2.98e-5 まで下がる）ため、そこで maxRel が 1.65e-3 まで伸びるが、その要素の
 * 絶対誤差は `1.65e-3 × 2.98e-5 ≈ 4.9e-8` でしかない。rtol の寄与は \|ref\| 上端 0.249 でも
 * `1e-5 × 0.249 ≈ 2.5e-6` と atol の 1/4 程度で、判定を主導しない。
 *
 * atol 1e-6 は実測最悪 3.87e-7（long-document）の約 2.6 倍。**tiny golden の `GOLDEN_TOLERANCE`
 * と同じ値になったのは独立に実測した結果の一致であって、流用ではない**（SBV2 dp の
 * `SBV2_TOLERANCE` が同じ値になったのと同じ理由 — このモデル自身の実測がその内側に収まった）。
 * 誤差の出所も同じ（fma 融合・matmul / masked mean の縮約順序が torch と違う・超越関数
 * （sqrt / rsqrt 系）の実装差）で、Transformer 本体の層数に対して T を 12 → 318 と 26 倍にしても
 * maxAbs は 1.7 倍程度にしか伸びていない。実装バグ（プールのマスク軸取り違え・Dense の重み
 * 転置ミス・正規化の分母取り違え）の誤差は出力の値域と同じ O(0.1) 〜 O(1) で、この閾値の
 * 5 桁以上上に出る。
 */
const EMBEDDINGGEMMA_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 1e-5 };

const SERIES_ROOT = new URL("../../../outputs/series/embeddinggemma-300m/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド（tools/exporter/export_embeddinggemma.py の docstring）。 */
const GENERATE_COMMAND =
  "cd tools/exporter && uv run --with 'transformers==5.14.1' python export_embeddinggemma.py";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する** — 列挙だけに頼ると生成を
 * 一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/exporter/export_embeddinggemma.py` の GOLDEN_CASES。
 */
const EXPECTED_CASES = ["bare", "document-en", "long-document", "query-en", "query-ja"] as const;

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

/** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
const CASES = discoverCases(SERIES_ROOT);
/** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
const AVAILABLE = CASES.length > 0;
/**
 * **何か 1 つでも**残っているか（完全性テストの SKIP 述語 — Codex 波 H 指摘 H-02）。
 * io が全滅してモデルだけ残った欠損は `AVAILABLE` では偽になり、`ignore: !AVAILABLE` だと
 * 完全性テスト自身が SKIP される — 欠損を FAIL にする述語は「完全に空」でだけ寝てよい。
 */
const ANY_PRESENT = AVAILABLE || modelPresent(new URL(MODEL_FILE, SERIES_ROOT));

if (!AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に export 済み資産が無いため実重み EmbeddingGemma E2E を ` +
      `SKIP する（重み 1.2GB 級につきリポジトリ管理外）。生成: ${GENERATE_COMMAND}`,
  );
}

Deno.test({
  name: "EmbeddingGemma 資産: 期待するケースとモデル本体が揃っている",
  // 完全に空の環境だけ「生成していない」として SKIP。**何か 1 つでも**あれば欠けは FAIL
  //（モデルだけ残って golden が全滅した欠損も拾う — `ANY_PRESENT` の JSDoc）。
  ignore: !ANY_PRESENT,
  fn: () => {
    assertEquals(CASES, [...EXPECTED_CASES], `${SERIES_ROOT.pathname} の golden ケース`);
    assert(modelPresent(new URL(MODEL_FILE, SERIES_ROOT)), `${MODEL_FILE} が無い`);
  },
});

for (const caseName of CASES) {
  Deno.test({
    name: `EmbeddingGemma golden 突合: ${caseName}（実 GPU / torch CPU 期待値）`,
    ignore: !AVAILABLE || !GPU_AVAILABLE,
    fn: async () => {
      const shards = resolveShards(new URL(MODEL_FILE, SERIES_ROOT));
      const [graphShard, ioBytes] = await Promise.all([
        readShard(shards[0]),
        readBuffer(SERIES_ROOT, `${IO_PREFIX}${caseName}${IO_SUFFIX}`),
      ]);
      const parsed = prepareModel(graphShard);
      const io = parseSafetensors(ioBytes);

      // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
      const expectedKeys = [
        ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
        ...parsed.graph.outputs.map((_, index) => `output.${index}`),
      ].sort();
      assertEquals([...io.tensors.keys()].sort(), expectedKeys, "io.safetensors のテンソルキー");

      // 記号次元 T は golden の入力 shape の実長から束縛される（明示 bindings を渡さない）。
      // ケースごとに T が違う（12 / 16 / 16 / 19 / 318）ので、宣言上限 Tmax = 512（sym_max —
      // export_embeddinggemma.py の SYM_MAX）に依存した実装（プランを Tmax で組む等）は
      // ここで値か shape が壊れる。
      const inputs: Record<string, Tensor> = {};
      for (const spec of parsed.graph.inputs) {
        const view = io.tensors.get(`input.${spec.name}`);
        assert(view !== undefined, `input.${spec.name} が golden に無い`);
        inputs[spec.name] = ioTensor(io, view, spec.dtype);
      }

      const gpu = await acquireGpu();
      const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
      try {
        const outputs = await session.run(inputs);
        assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

        parsed.graph.outputs.forEach((name, index) => {
          const view = io.tensors.get(`output.${index}`);
          assert(view !== undefined, `output.${index} が golden に無い`);
          const where = `${caseName} output.${index} ('${name}')`;
          const declared = parsed.graph.values[name].dtype;
          assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
          assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
          const report = compareTensors(
            outputs[name],
            ioTensor(io, view, declared),
            EMBEDDINGGEMMA_TOLERANCE,
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
