// 実重み DACVAE（Semantic-DACVAE-Japanese-32dim）の**コーデック 2 本**の実 GPU golden E2E
// （ADR 0005 の段 3）。
//
// Irodori-TTS v4 のテキスト条件エンコーダと DiT は tests/e2e_irodori_test.ts が受け持ち、
// こちらはその**前後**にあたる波形 ↔ latent の変換を受け持つ。対象は
// `outputs/series/dacvae-32dim/<target>/`（decoder の重みだけで 261MB のためリポジトリ管理外
// — `.gitignore` の `outputs/`）。生成は `tools/exporter/export_dacvae.py`（コマンドは下の
// GENERATE_COMMAND がそのまま正本）。
//
// ターゲットは 2 本（recon の G6 / G7）:
//
// - `decoder` — `quantizer.out_proj` + decoder 主経路。`[1,S,32]` latent → `[1,1,1920S]` 波形
// - `encoder` — encoder + `quantizer.in_proj` の前半 32 行。`[1,T,1920]` 波形 → `[1,T,32]` latent
//
// **どちらも位置表もマスクも持たない純粋な畳み込み網**で、他の実重み E2E と違って
// attention が 1 本も無い。踏んでいる op は conv1d / conv_transpose1d / sin / tanh /
// add / mul / reshape / permute だけで、**`sin` を実資産で踏む唯一の門**でもある
// （Snake 活性 — ADR 0043 の第 1 層）。
//
// `encoder` の入力が `[1,1,1920T]` ではなく**フレーム分割済みの `[1,T,1920]`** なのは、IR の
// 束縛規則がシンボルの**素の形**での出現を要求するため（`karume.verify` の
// `_check_symbol_bindability` / `plan.ts` の `bindSymbols` は coeff ≠ 1 の次元から束縛しない）。
// 要素順を変えない読み替えなので、ホストは連続した波形バッファをそのまま渡せる。
//
// golden の入力は**合成乱数ではない** — latent は `irodori_pipeline.py` の full-loop golden
// （実テキストから 40 step 回した最終 z）、波形は公式サンプルの参照音声を −16 LUFS へ
// 正規化したもの。Snake の `sin(αx)` は入力の値域に強く効くので、tolerance の根拠を実運用の
// 値域と対応させるための選択（生成側 docstring の DECODER_CASES / ENCODER_CASES）。
//
// 資産が無い環境では**明示 SKIP**する。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。
// 逆に資産が**中途半端に**（ターゲット欠け / ケース欠け）存在する場合は SKIP ではなく
// FAIL にする（下の「資産の完全性」テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, createSession, openModel, parseSafetensors, type Tensor } from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * DACVAE decoder（`out_proj` + 4 段 upsample + 波形ヘッド）の torch CPU 期待値との突合に使う
 * 許容誤差。出力は **Tanh を通った波形**（値域 ±1）。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,1920S]`）:
 *
 * | ケース    | S   | 出力サンプル | maxAbs  | maxRel | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | --------- | --- | ------------ | ------- | ------ | ------------ | ------------------ |
 * | z-min     | 2   | 3,840        | 8.16e-8 | 0.213  | 4.51e-3      | 1.49e-7            |
 * | z-no-ref  | 116 | 222,720      | 3.78e-6 | 0.495  | 0.879        | 3.17e-8            |
 * | z-full    | 161 | 309,120      | 4.15e-6 | 5.67   | 0.9998       | 6.93e-9            |
 * | z-max     | 750 | 1,440,000    | 4.75e-6 | 8.30   | 0.9998       | 1.02e-9            |
 *
 * **判定の主役は atol**。波形は 0 を跨ぐ値域で、最小の非ゼロ \|ref\| が 1e-9 まで下がる
 * （ゼロ交差の直近）ので、rtol を主役にすると発散する — 実測 maxRel は 8.3 に達するが、
 * その要素の絶対誤差は 1e-8 級でしかない。rtol 1e-6 は値域上端 \|ref\| = 1 でも寄与が
 * 1e-6（atol の 1/30）で、判定を主導しない。
 *
 * atol 3e-5 は実測最悪 4.75e-6（z-max）の約 6.3 倍。フルスケール ±1 に対して −110dBFS 級で、
 * 16bit PCM の 1LSB（3.05e-5）とほぼ同じ — **WAV に書いた時点で消える**大きさに揃えてある。
 *
 * NOTE: **`sin` の引数が π を大きく超える経路の初の観測**。Snake は `x + (α+1e-9)⁻¹ sin(αx)²`
 * で、実 latent での \|αx\| は decoder 側の最大が **15.9（≈ 5.06π）**（最終段の Snake(96)。
 * 中間層は 4.9〜7.1 = 1.6π〜2.3π）。それでも誤差が 5e-6 に留まるということは、GPU の `sin` が
 * この引数域で f32 の精度を保っている（引数簡約が効いている）ということ。誤差の主因は
 * conv の縮約順序差のほうで、S を 2 → 750 と 375 倍にしても maxAbs が 58 倍にしか
 * 伸びない（層方向の縮約長は S に依らない）ことと整合する。
 *
 * 実装バグ（upsample の stride 取り違え・残差の足し忘れ・Snake の α のチャネル取り違え）の
 * 誤差は値域と同じ O(0.1〜1) で、この閾値の 4〜5 桁上に出る。
 */
const DECODER_TOLERANCE: Tolerance = { atol: 3e-5, rtol: 1e-6 };

/**
 * DACVAE encoder（4 段 downsample + 切り詰めた `in_proj`）の許容誤差。出力は latent
 * `[1,T,32]`（DiT が食う値そのもの）。
 *
 * 実測（`atol=rtol=0`、2 ケース × 出力 1 本）:
 *
 * | ケース   | T   | 入力サンプル | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | --- | ------------ | ------- | ------- | ------------ | ------------------ |
 * | wav-min  | 2   | 3,840        | 1.11e-5 | 2.05e-3 | 2.93         | 7.23e-4            |
 * | wav-ref  | 190 | 364,800      | 2.56e-5 | 6.95e-2 | 5.31         | 4.62e-5            |
 *
 * **decoder より 1 桁緩い**のは値域が違うから — latent は ±5.3 で、波形（±1）の 5 倍。
 * 相対量で見ると 2.56e-5 / 5.31 = 4.8e-6 で decoder 側（4.75e-6 / 1.0）とほぼ同じであり、
 * 誤差の出所（conv の縮約順序差）が同じであることを示している。
 *
 * atol 2e-4 は実測最悪 2.56e-5（wav-ref）の約 7.8 倍。rtol 1e-6 の寄与は上端
 * \|ref\| = 5.31 でも 5.3e-6（atol の 1/38）。
 *
 * NOTE: encoder 側の \|αx\| は最大 **71.3（≈ 22.7π）**（先頭の Snake(64) — 入力が生波形で
 * α が大きい層）と decoder 側よりさらに深く π を超えるが、誤差は decoder と同じ相対量に
 * 留まる。`sin` の引数域そのものは精度の支配項ではない、というのがこの 2 本の観測。
 *
 * NOTE: 値は latent の値域（±5.3）の上に立っている。別の値域の出力へ流用してはならない。
 */
const ENCODER_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 1e-6 };

/**
 * ターゲット別 tolerance（**出力位置ごとの配列**）。表の穴は「別ターゲット / 別出力の値で
 * 突合する」沈黙誤りになるので、本数が IR の出力数と合っているかもケースごとに検査する。
 */
const TOLERANCES: Readonly<Record<string, readonly Tolerance[]>> = {
  "decoder": [DECODER_TOLERANCE],
  "encoder": [ENCODER_TOLERANCE],
};

const SERIES_ROOT = new URL("../../../outputs/series/dacvae-32dim/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド（tools/exporter/export_dacvae.py の docstring）。 */
const GENERATE_COMMAND =
  "cd tools/exporter && uv run --with descript-audiotools --with einops python export_dacvae.py";

/**
 * 生成されているはずのターゲットとケース。**列挙結果ではなくここで固定する** — 列挙だけに
 * 頼ると生成を一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/exporter/export_dacvae.py` の TARGETS / DECODER_CASES / ENCODER_CASES。
 */
const EXPECTED_CASES: Readonly<Record<string, readonly string[]>> = {
  // S = 161（実 z）/ 750（宣言上限そのもの）/ 2（記号次元の下限）/ 116（参照なしの実 z）。
  "decoder": ["z-full", "z-max", "z-min", "z-no-ref"],
  // T = 2（記号次元の下限）/ 190（参照音声 7.6 秒の全長）。
  "encoder": ["wav-min", "wav-ref"],
};

const EXPECTED_TARGETS = Object.keys(EXPECTED_CASES).sort();

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

/**
 * 系列ディレクトリ直下にある**グラフ以外**の生成物。ここは IR ターゲットの列挙なので、
 * 別の台本が書くホスト側資産は除く（`dacvae_host.py` の前処理 golden）。
 *
 * MUST: 「知らないディレクトリは無視」にしない — 除外は名前で明示する。未知の名前が
 * 増えたらこのテストが落ちて、門の対象から漏れていることに気づける。
 */
const NON_GRAPH_DIRS: ReadonlySet<string> = new Set(["host"]);

const discoverTargets = (root: URL): readonly string[] =>
  listDir(root)
    .filter((entry) => entry.isDirectory && !NON_GRAPH_DIRS.has(entry.name))
    .map((entry) => entry.name)
    .sort();

const discoverCases = (root: URL, target: string): readonly string[] =>
  listDir(new URL(`${target}/`, root))
    .filter((entry) =>
      entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
    )
    .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
    .sort();

const readBuffer = async (root: URL, target: string, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(`${target}/${file}`, root));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
const TARGETS = discoverTargets(SERIES_ROOT);
/** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
const AVAILABLE = TARGETS.length > 0;

if (!AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に export 済みターゲットが無いため実重み DACVAE ` +
      `コーデック E2E を SKIP する（decoder だけで 261MB につきリポジトリ管理外）。` +
      `生成: ${GENERATE_COMMAND}`,
  );
}

Deno.test({
  name: "DACVAE 資産: 期待するターゲットとケースが揃っている",
  // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
  ignore: !AVAILABLE,
  fn: () => {
    assertEquals(TARGETS, EXPECTED_TARGETS, `${SERIES_ROOT.pathname} のターゲット`);
    assertEquals(Object.keys(TOLERANCES).sort(), EXPECTED_TARGETS, "ターゲット別 tolerance の表");
    for (const target of TARGETS) {
      assert(Object.hasOwn(EXPECTED_CASES, target), `${target} の期待ケース表が無い`);
      assertEquals(
        discoverCases(SERIES_ROOT, target),
        [...EXPECTED_CASES[target]],
        `${target} の golden ケース`,
      );
      const model = new URL(`${target}/${MODEL_FILE}`, SERIES_ROOT);
      assert(Deno.statSync(model).isFile, `${target}/${MODEL_FILE} が無い`);
    }
  },
});

for (const target of TARGETS) {
  for (const caseName of discoverCases(SERIES_ROOT, target)) {
    Deno.test({
      name: `DACVAE golden 突合: ${target} / ${caseName}（実 GPU / torch CPU 期待値）`,
      ignore: !AVAILABLE || !GPU_AVAILABLE,
      fn: async () => {
        const ioFile = `${IO_PREFIX}${caseName}${IO_SUFFIX}`;
        const [modelBytes, ioBytes] = await Promise.all([
          readBuffer(SERIES_ROOT, target, MODEL_FILE),
          readBuffer(SERIES_ROOT, target, ioFile),
        ]);
        const parsed = openModel(modelBytes);
        const io = parseSafetensors(ioBytes);
        // Object.hasOwn で見る（素の `TOLERANCES[target]` はプロトタイプ由来のキーを拾う）。
        assert(Object.hasOwn(TOLERANCES, target), `${target} の tolerance が無い`);
        const tolerances = TOLERANCES[target];
        assertEquals(
          tolerances.length,
          parsed.graph.outputs.length,
          `${target} の tolerance 本数が IR 出力数と違う`,
        );

        // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
        const expectedKeys = [
          ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
          ...parsed.graph.outputs.map((_, index) => `output.${index}`),
        ].sort();
        assertEquals([...io.tensors.keys()].sort(), expectedKeys, `${ioFile} のテンソルキー`);

        // 記号次元は golden の入力 shape の実長から束縛される（明示 bindings を渡さない）。
        // decoder は `latent[1,S,32]` の次元 1 が、encoder は `wav[1,T,1920]` の次元 1 が
        // 束縛源で、**出力側は係数付きの派生次元**（decoder の `1920S`）。bindSymbols の
        // 2 巡目と planGraph が係数を評価し直すので、`coeff·sym` の評価が壊れれば
        // ここで shape が合わなくなる。
        const inputs: Record<string, Tensor> = {};
        for (const spec of parsed.graph.inputs) {
          const view = io.tensors.get(`input.${spec.name}`);
          assert(view !== undefined, `input.${spec.name} が ${ioFile} に無い`);
          inputs[spec.name] = ioTensor(io, view, spec.dtype);
        }

        const gpu = await acquireGpu();
        const session = await createSession(gpu, parsed);
        try {
          const outputs = await session.run(inputs);
          assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

          parsed.graph.outputs.forEach((name, index) => {
            const view = io.tensors.get(`output.${index}`);
            assert(view !== undefined, `output.${index} が ${ioFile} に無い`);
            const where = `${target}/${caseName} output.${index} ('${name}')`;
            const declared = parsed.graph.values[name].dtype;
            assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
            assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
            const report = compareTensors(
              outputs[name],
              ioTensor(io, view, declared),
              tolerances[index],
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
}
