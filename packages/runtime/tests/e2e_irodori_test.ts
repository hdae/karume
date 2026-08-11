// 実重み Irodori-TTS v4-Small の**テキスト条件エンコーダ**の実 GPU golden E2E（ADR 0005 の段 3）。
//
// tiny golden（tests/e2e_golden_test.ts）が「op 契約の被覆」を、実重み SBV2
// （tests/e2e_sbv2_test.ts）が「音響チェーン側の実重み」を、EmbeddingGemma
// （tests/e2e_embeddinggemma_test.ts）が「単一ベクトル出力のテキスト系」を受け持つのに対し、
// こちらは **token 列を返すテキスト条件エンコーダ**（`[1,T,H]` の系列出力）を受け持つ。
// 対象は `outputs/series/irodori-v4-small/<target>/`（backbone の重みだけで 1.26GB のため
// リポジトリ管理外 — `.gitignore` の `outputs/`）。生成は `tools/exporter/export_irodori.py`
// （コマンドは下の GENERATE_COMMAND がそのまま正本）。
//
// ターゲットは 3 本（recon の G1 / G1a / G1b）:
//
// - `backbone`     — 同梱 ModernBERT-ja-310m（25 層）。`[1,T]` ids → `[1,T,768]`
// - `text-proj`    — text 側 projector。`[1,T,768]` → `[1,T,512]`
// - `caption-proj` — caption 側 projector（同形・別重み）
//
// backbone を projector と融合していないのは **backbone が text / caption で共有**だから
// （融合すると 1.26GB の重みが 2 部できる）。ホストは backbone を 2 回回して各 projector へ
// 流すので、この E2E も projector の入力に **backbone の torch 期待値**をそのまま食わせる。
//
// SBV2 と違い格納 dtype 系列は f32 の 1 本のみ（f16 / i8 は別系列で決める話）なので、
// 系列パラメタ化はしない。
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
 * 実重み Irodori backbone（ModernBERT-ja-310m 25 層）の torch CPU 期待値との突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、6 ケース × 出力 1 本 `[1,T,768]`）:
 *
 * | ケース       | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------------ | --- | ------- | ------- | ------------ | ------------------ |
 * | text-short   | 7   | 2.43e-5 | 6.42e-3 | 34.55        | 5.31e-5            |
 * | text-formal  | 13  | 1.53e-5 | 9.80e-2 | 34.37        | 8.24e-7            |
 * | caption-ja   | 22  | 1.74e-5 | 1.62e-1 | 34.29        | 4.32e-6            |
 * | text-emoji   | 29  | 1.96e-5 | 9.98e-3 | 34.30        | 8.29e-6            |
 * | text-long    | 144 | 2.67e-5 | 8.84e-2 | 34.21        | 1.13e-5            |
 * | caption-long | 404 | 2.72e-5 | 8.66e-1 | 34.22        | 2.06e-6            |
 *
 * **判定の主役は atol**。出力は隠れ状態そのもの（0 を跨ぐ値域で、最小の非ゼロ \|ref\| が
 * 8.24e-7 まで下がる）なので、rtol を主役にすると 0 近傍で発散する — 実測 maxRel は 0.866 に
 * 達するが、その要素の絶対誤差は 2e-6 級でしかない。rtol 1e-6 は値域上端 \|ref\| = 34.55 でも
 * 寄与が 3.5e-5（atol の 1/6）で、判定を主導しない。
 *
 * atol 2e-4 は実測最悪 2.72e-5（caption-long）の約 7.4 倍。誤差の出所は他の実重み E2E と
 * 同じ（fma 融合・linear / attention の縮約順序が torch と違う・超越関数の実装差）で、
 * T を 7 → 404 と 58 倍にしても maxAbs は 1.1 倍にしかならない（層方向の縮約長は T に
 * 依らず、T は独立な列方向にしか効かない — SBV2 flow と同じ構造）。実装バグ（RoPE の
 * θ 2 系統の取り違え・sliding window の窓幅取り違え・qkv 分割の取り違え）の誤差は出力の
 * 値域と同じ O(10) で、この閾値の 4〜5 桁上に出る。
 *
 * NOTE: この値は **`|ref|` 上端が 34 という値域**の上に立っている。同じ 2e-4 を値域 O(1) の
 * 出力へ流用してはならない（同じ手順で実測し直すこと）。
 */
const BACKBONE_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 1e-6 };

/**
 * text 側 projector（`[1,T,768]` → `[1,T,512]` の residual_mlp）の許容誤差。
 *
 * 実測（`atol=rtol=0`、6 ケース × 出力 1 本）:
 *
 * | ケース       | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------------ | --- | ------- | ------- | ------------ | ------------------ |
 * | text-short   | 7   | 7.15e-6 | 2.68e-4 | 11.89        | 3.00e-3            |
 * | text-formal  | 13  | 8.11e-6 | 7.88e-4 | 12.76        | 2.00e-4            |
 * | caption-ja   | 22  | 8.11e-6 | 4.55e-3 | 12.97        | 1.96e-4            |
 * | text-emoji   | 29  | 9.06e-6 | 1.53e-3 | 12.74        | 7.85e-4            |
 * | text-long    | 144 | 1.14e-5 | 1.57e-1 | 14.95        | 7.21e-6            |
 * | caption-long | 404 | 1.29e-5 | 2.32e-1 | 13.35        | 1.65e-5            |
 *
 * backbone と同じく **atol 主役**（最小の非ゼロ \|ref\| が 7.21e-6 まで下がる）。atol 1e-4 は
 * 実測最悪 1.29e-5 の約 7.8 倍で、rtol 1e-6 の寄与は上端 \|ref\| = 14.95 でも 1.5e-5。
 *
 * **backbone より 1 桁近く小さい**のは、この graph が 3 本の linear + rms_norm + sigmoid の
 * 7 ノードしかなく、25 層ぶんの縮約誤差の蓄積が無いから（入力は torch が出した backbone の
 * 期待値そのもので、backbone 側の誤差はここには入らない）。
 */
const TEXT_PROJ_TOLERANCE: Tolerance = { atol: 1e-4, rtol: 1e-6 };

/**
 * caption 側 projector の許容誤差。**`TEXT_PROJ_TOLERANCE` は流用しない** — 同形の graph でも
 * 重みが別なので、縮約の丸まり方も出力の値域も同じにはならない（実測が実際に違う）。
 *
 * 実測（`atol=rtol=0`、6 ケース × 出力 1 本）:
 *
 * | ケース       | T   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------------ | --- | ------- | ------- | ------------ | ------------------ |
 * | text-short   | 7   | 1.14e-5 | 1.16e-3 | 15.82        | 1.12e-3            |
 * | text-formal  | 13  | 1.05e-5 | 3.39e-3 | 16.16        | 4.57e-4            |
 * | caption-ja   | 22  | 3.05e-5 | 4.21e-2 | 32.83        | 2.94e-5            |
 * | text-emoji   | 29  | 1.53e-5 | 3.25e-3 | 16.27        | 8.07e-4            |
 * | text-long    | 144 | 1.53e-5 | 1.37e-2 | 18.04        | 8.25e-5            |
 * | caption-long | 404 | 1.14e-5 | 7.85e-2 | 18.09        | 2.29e-5            |
 *
 * atol 2e-4 は実測最悪 3.05e-5（caption-ja）の約 6.6 倍。最悪ケースが caption-ja なのは、
 * **本物の caption を食わせたときだけ出力の値域が 2 倍近く伸びる**（\|ref\| 上端 32.8 対
 * 他の 16〜18）ためで、誤差は値域に比例して伸びている。text 側 projector が同じケースで
 * 8.11e-6 に留まるのと対照的 — 2 本の tolerance を分けている理由がここに出ている。
 */
const CAPTION_PROJ_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 1e-6 };

/** ターゲット別 tolerance。表の穴は「別ターゲットの値で突合する」沈黙誤りになる。 */
const TOLERANCES: Readonly<Record<string, Tolerance>> = {
  "backbone": BACKBONE_TOLERANCE,
  "text-proj": TEXT_PROJ_TOLERANCE,
  "caption-proj": CAPTION_PROJ_TOLERANCE,
};

const SERIES_ROOT = new URL("../../../outputs/series/irodori-v4-small/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド（tools/exporter/export_irodori.py の docstring）。 */
const GENERATE_COMMAND =
  "cd tools/exporter && uv run --with 'transformers==5.14.1' python export_irodori.py";

/**
 * 生成されているはずのターゲットとケース。**列挙結果ではなくここで固定する** — 列挙だけに
 * 頼ると生成を一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/exporter/export_irodori.py` の TARGETS / GOLDEN_CASES。
 *
 * MUST: ターゲット（G2〜G7）を足すときはこの表と TOLERANCES を同時に伸ばす。
 */
const EXPECTED_TARGETS = ["backbone", "caption-proj", "text-proj"] as const;
const EXPECTED_CASES = [
  "caption-ja",
  "caption-long",
  "text-emoji",
  "text-formal",
  "text-long",
  "text-short",
] as const;

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

const discoverTargets = (root: URL): readonly string[] =>
  listDir(root).filter((entry) => entry.isDirectory).map((entry) => entry.name).sort();

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
    `[karume] ${SERIES_ROOT.pathname} に export 済みターゲットが無いため実重み Irodori ` +
      `テキスト系 E2E を SKIP する（backbone だけで 1.26GB につきリポジトリ管理外）。` +
      `生成: ${GENERATE_COMMAND}`,
  );
}

Deno.test({
  name: "Irodori 資産: 期待するターゲットとケースが揃っている",
  // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
  ignore: !AVAILABLE,
  fn: () => {
    assertEquals(TARGETS, [...EXPECTED_TARGETS], `${SERIES_ROOT.pathname} のターゲット`);
    assertEquals(
      Object.keys(TOLERANCES).sort(),
      [...EXPECTED_TARGETS].sort(),
      "ターゲット別 tolerance の表",
    );
    for (const target of TARGETS) {
      assertEquals(
        discoverCases(SERIES_ROOT, target),
        [...EXPECTED_CASES],
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
      name: `Irodori golden 突合: ${target} / ${caseName}（実 GPU / torch CPU 期待値）`,
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
        const tolerance = TOLERANCES[target];

        // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
        const expectedKeys = [
          ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
          ...parsed.graph.outputs.map((_, index) => `output.${index}`),
        ].sort();
        assertEquals([...io.tensors.keys()].sort(), expectedKeys, `${ioFile} のテンソルキー`);

        // 記号次元 T は golden の入力 shape の実長から束縛される（明示 bindings を渡さない）。
        // ケースごとに T が違う（7 / 13 / 22 / 29 / 144 / 404）ので、宣言上限 Tmax = 512
        // （export_irodori.py の SYM_MAX — 帯マスクと RoPE 表の焼き付け点）に依存した実装は
        // ここで値か shape が壊れる。
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
            const report = compareTensors(outputs[name], ioTensor(io, view, declared), tolerance);
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
