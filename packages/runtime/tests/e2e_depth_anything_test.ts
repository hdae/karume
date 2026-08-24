// 実重みの Depth Anything V2（単一画像の相対深度推定）の実 GPU golden E2E（ADR 0005 の段 3）。
//
// BiRefNet（tests/e2e_birefnet_test.ts）が「画素ごとの出力を持つ画像系」の**二値の地図**を
// 受け持つのに対し、こちらは同じ画素ごとの出力でも**連続値の地図**（`[1, S, S]` の相対深度）
// を受け持つ。相対深度には単位も原点も無く、意味を持つのは**大小の順序だけ**なので、門の
// 立て方も「値が合っている」（golden 突合）と「順序が構図と合っている」（判別）に分かれる。
//
// 対象は `outputs/series/depth-anything-v2-small-hf/`（重み + 焼いた定数で 99MB のためリポジ
// トリ管理外 — `.gitignore` の `outputs/`）。生成は
// `tools/export-recipes/depth_anything/export.py`（コマンドは {@link GENERATE} がそのまま正本）。
//
// 系列は **1 本だけ**（Small）。台本は `--model-dir` でサイズ軸を受けるが、Base / Large は
// 上流のライセンスが CC BY-NC 4.0 で重みを取得していない（Apache-2.0 は Small のみ）。
//
// 解像度も**軸ではない** — 518²（patch 14 × 37）の 1 点固定で、外れると DINOv2 の位置埋め込みが
// bicubic 補間へ落ちる（`patch_depth_anything` の ② が fail loudly にしてある）。
//
// ## golden の 2 群（合成画像 + 実画像）— どちらも残す
//
// 入力は**どちらの群も golden に焼かれた `pixel_values` そのもの**（ビット同一）なので、
// 突合に出るのは**ランタイムの数値誤差だけ**で、`GOLDEN_TOLERANCE` 1 本が全 8 ケースを見る。
// 2 群を持つのは踏む分布が違うから:
//
// - **合成画像 4 ケース**（`checker` / `disc` / `noise` / `ramp`）: `ramp` は単調な奥行き
//   手掛かりを持つ唯一の 1 枚で、幾何の判別（対角ランプとの相関が `ramp` で最大）の土台。
// - **実画像 4 ケース**（`photo-*`）: 自然画像の分布点でのランタイム忠実度。
//
// ## TS 前処理を含む鎖は、2 つの門の**合成**で持つ
//
// 「PNG を渡したら `DPTImageProcessor` + torch と同じ深度地図が返る」という鎖は、このテスト
// 単独ではなく `packages/models/tests/e2e_depth_anything_real_test.ts` の**入力側 parity 門**
// （同じ PNG から同じ `pixel_values` が出る — 入力差 ≤ 1 LSB = 1.8e-2・相違率 ≤ 2%）と、
// 本テストの**golden 入力での忠実度**（実画像 4 ケースを含む）の合成で持つ。前処理をここで
// 通さないのは依存方向のため（runtime のテストから models の実装を相対 import するのは逆向き）。
// 分けた副産物として、落ちたときに前処理と推論のどちらが動いたのかがテストの名前で分かる。
//
// MUST: DA-V2 の resize は **bicubic**（`preprocessor_config.json` の `"resample": 3`）で、
// SigLIP2 / BiRefNet の bilinear（2）と**違う**。既定へ落ちると `pixel_values` が最大 0.59
// ずれる（1 LSB = 0.0175 の 34 倍 — 実測）が、その取り違えを見るのは models 側の parity 門
// （こちらは golden の `pixel_values` しか読まない）。
//
// 意味の判別のうち**実画像側**（構図から言える近い領域 > 遠い領域）と、そのついでに書く深度
// PNG も同じ理由で models 側にある（`encodePng` が models の実装で、入力も TS 前処理を通した
// ものだから）。**合成画像側の幾何判別**（`ramp` との相関）は golden の入力だけで完結するので
// ここに残る。
//
// 資産が無い環境では**明示 SKIP** する。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。逆に
// 資産が**中途半端に**（ケース欠け）存在する場合は SKIP ではなく FAIL にする（下の「資産の
// 完全性」テスト）— そこは無音の見かけ成功になる。

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
 * **golden の入力で回した**ときの許容誤差（合成 4 + 実画像 4 の全 8 ケース共通）。
 *
 * 実測（`atol=rtol=0` の素の突合、出力 1 本 `[1,518,518]`）:
 *
 * | ケース          | maxAbs  | maxRel  | \|ref\| 上端 |
 * | --------------- | ------- | ------- | ------------ |
 * | checker         | 4.53e-6 | 1.98e-6 | 4.048        |
 * | disc            | 6.62e-6 | 7.25e-6 | 4.438        |
 * | noise           | 3.58e-6 | 1.23e-6 | 4.212        |
 * | ramp            | 1.86e-5 | 8.78e-4 | 4.896        |
 * | photo-portrait  | 2.69e-5 | —       | 5.333        |
 * | photo-landscape | 3.96e-5 | —       | 5.143        |
 * | photo-corridor  | 8.35e-6 | —       | 5.411        |
 * | photo-street    | 3.82e-5 | —       | 4.925        |
 *
 * atol 2e-4 は実測最悪 3.96e-5（photo-landscape）の約 5.1 倍。
 *
 * **rtol は 0**。head 末尾が ReLU なので、遠景は**厳密に 0** の広い平地になる（実測の最小は
 * 8 ケースとも 0.000）。相対量を主役にすると、その平地で分母 0 の判定が発散する — 実測
 * maxRel は実画像側で 45.7 まで出るが、その要素の絶対誤差は 3.8e-5 でしかない。
 *
 * 誤差の出所は SigLIP2 / BiRefNet と同じ（fma 融合・linear / conv の縮約順序が torch と違う・
 * 超越関数の実装差）。相対量で見ると 3.96e-5 / 5.14 ≈ 7.7e-6 で、BiRefNet（≈1.7e-5）より
 * 1 桁小さい — DINOv2 + DPT は BiRefNet の decoder ほど深い飽和を持たない。
 *
 * 実装バグ（reassemble の並べ替え取り違え・DPT fusion の段順違い・upsample の軸違い）の誤差は
 * 出力の値域と同じ O(1)〜O(5) で、この閾値の 4 桁上に出る。
 *
 * **合成と実画像で共有する**のは、入力の作り方が同じ（どちらも golden の `pixel_values`）で
 * 実測も同じ桁だから。前処理の 1 LSB 差を含んだ鎖の主張は別物で、そちらは models 側の
 * parity 門との合成で持つ（モジュール docstring）。
 */
const GOLDEN_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 0 };

/** `outputs/series/` 直下のディレクトリ名（`depth_anything.export.default_out_dir` の綴り）。 */
const SERIES_NAME = "depth-anything-v2-small-hf";

/** SKIP 時にそのまま貼れる生成コマンド（実画像 golden まで含む形）。 */
const GENERATE = "cd tools/export-recipes && uv run --group depth-anything-preprocess" +
  " python -m depth_anything.export --real-images";

const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES_NAME}/`, import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/**
 * 生成されているはずの**合成画像**ケース。正本は `depth_anything/export.py` の
 * `build_cases`（モデル軸に依らず同じ 4 枚）。
 */
const SYNTHETIC_CASES = ["checker", "disc", "noise", "ramp"] as const;

/**
 * **実画像**ケース（`--real-images` を付けた emit だけが持つ）。正本は
 * `depth_anything/export.py` の `REAL_CASES`。ここが要るのは golden のケース名だけで、元に
 * なった PNG との対応は `packages/models/tests/e2e_depth_anything_real_test.ts` が持つ
 * （このテストは PNG を読まない — モジュール docstring の「2 つの門の合成」）。
 */
const REAL_CASES = ["photo-portrait", "photo-landscape", "photo-corridor", "photo-street"] as const;

/**
 * 幾何の判別に使うケース（対角ランプとの相関がここで最大になる）。正本は
 * `depth_anything/export.py` の `RAMP_CASE`。
 */
const RAMP_CASE = "ramp";

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

const readBuffer = async (file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, SERIES_ROOT));
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

/** グラフ入力の静的次元（記号次元は無い — `depth_anything/export.py` の `symbol_names=()`）。 */
const staticDim = (parsed: KarumeModel, axis: number): number => {
  const dim = parsed.graph.inputs[0].shape[axis];
  assert(typeof dim === "number", `pixel_values の軸 ${axis} が記号次元 '${String(dim)}'`);
  return dim;
};

/** 深度地図と対角ランプ座標のピアソン相関（`depth_anything/export.py` の `_ramp_correlation`）。 */
const rampCorrelation = (depth: Float32Array, size: number): number => {
  const plane = new Float64Array(depth.length);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      plane[y * size + x] = (y / (size - 1) + x / (size - 1)) / 2;
    }
  }
  let depthMean = 0;
  let planeMean = 0;
  for (let index = 0; index < depth.length; index += 1) {
    depthMean += depth[index];
    planeMean += plane[index];
  }
  depthMean /= depth.length;
  planeMean /= plane.length;
  let covariance = 0;
  let depthNorm = 0;
  let planeNorm = 0;
  for (let index = 0; index < depth.length; index += 1) {
    const left = depth[index] - depthMean;
    const right = plane[index] - planeMean;
    covariance += left * right;
    depthNorm += left * left;
    planeNorm += right * right;
  }
  return covariance / Math.sqrt(depthNorm * planeNorm);
};

/** ファイルの有無。MUST: NotFound 以外は伝播させる（`listDir` と同じ理由）。 */
const fileExists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const DISCOVERED = discoverCases(SERIES_ROOT);
const realNames = new Set<string>(REAL_CASES);
const FOUND_SYNTHETIC = DISCOVERED.filter((name) => !realNames.has(name));
const FOUND_REAL = DISCOVERED.filter((name) => realNames.has(name));
/** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
const AVAILABLE = DISCOVERED.length > 0;
/**
 * **何か 1 つでも**残っているか（完全性テストの SKIP 述語 — Codex 波 H 指摘 H-02）。
 * golden が全滅してモデルだけ残った欠損は `AVAILABLE` では偽になり、`ignore: !AVAILABLE` だと
 * 完全性テスト自身が SKIP される — 欠損を FAIL にする述語は「完全に空」でだけ寝てよい。
 */
const ANY_PRESENT = AVAILABLE || fileExists(new URL(MODEL_FILE, SERIES_ROOT));

if (!AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に export 済み資産が無いため実重み Depth Anything V2 ` +
      `E2E を SKIP する（重みがリポジトリ管理外）。生成: ${GENERATE}`,
  );
}

Deno.test({
  name: "Depth Anything 資産: 期待するケースとモデル本体が揃っている",
  // 完全に空の環境だけ「生成していない」として SKIP。**何か 1 つでも**あれば欠けは FAIL
  //（モデルだけ残って golden が全滅した欠損も拾う — `ANY_PRESENT` の JSDoc）。
  ignore: !ANY_PRESENT,
  fn: () => {
    assertEquals(
      FOUND_SYNTHETIC,
      [...SYNTHETIC_CASES],
      `${SERIES_ROOT.pathname} の合成画像 golden ケース`,
    );
    // 実画像は**任意だが全部か 0 か**（`--real-images` を付けた emit は 4 本まとめて書く）。
    // 部分的な欠けを SKIP に丸めると、採り直しの途中で落ちた資産が黙って通る。
    assert(
      FOUND_REAL.length === 0 || FOUND_REAL.length === REAL_CASES.length,
      `${SERIES_ROOT.pathname} の実画像 golden が ${FOUND_REAL.length}/${REAL_CASES.length} 本` +
        `（採り直す: ${GENERATE}）`,
    );
    assert(fileExists(new URL(MODEL_FILE, SERIES_ROOT)), `${MODEL_FILE} が無い`);
  },
});

for (const caseName of DISCOVERED) {
  Deno.test({
    name: `Depth Anything golden 突合: ${caseName}（golden 入力 / 実 GPU 対 torch CPU 期待値）`,
    ignore: !AVAILABLE || !GPU_AVAILABLE,
    fn: async () => {
      const [modelBytes, ioBytes] = await Promise.all([
        readBuffer(MODEL_FILE),
        readBuffer(`${IO_PREFIX}${caseName}${IO_SUFFIX}`),
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

        const [name] = parsed.graph.outputs;
        const view = io.tensors.get("output.0");
        assert(view !== undefined, "output.0 が golden に無い");
        const where = `${caseName} output.0 ('${name}')`;
        const declared = parsed.graph.values[name].dtype;
        assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
        assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
        const expected = ioTensor(io, view, declared);
        const report = compareTensors(outputs[name], expected, GOLDEN_TOLERANCE);
        assert(report.pass, `${where}: ${formatAllclose(report)}`);
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  });
}

Deno.test({
  name: "Depth Anything 幾何判別: 対角ランプとの相関が ramp ケースで最大になる",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async () => {
    // golden 突合だけだと「期待値と合っている」ことしか言えず、深度として意味のある出力かは
    // 別問題（一様に潰れた出力は期待値も同じく潰れていれば通ってしまう）。ここは**単調な
    // 奥行き手掛かりを持つ 1 枚だけが立つ**という順序を見るので、出力が入力の幾何を追えて
    // いなければ落ちる。閾値は置かない（順序そのものが検査対象 — `depth_anything/export.py`
    // の `_sanity` と同じ形で、あちらは torch 側に掛かっている）。実測は
    // ramp 0.7705 / noise 0.4128 / checker 0.0414 / disc −0.1837（torch 側と 4 桁一致）。
    const modelBytes = await readBuffer(MODEL_FILE);
    const parsed = openModel(modelBytes);
    const size = staticDim(parsed, 3);
    assertEquals(size, staticDim(parsed, 2), "相関は正方形の入力を前提にする");
    const [name] = parsed.graph.outputs;

    const gpu = await acquireGpu();
    const session = await createSession(gpu, parsed);
    const correlations = new Map<string, number>();
    try {
      for (const caseName of SYNTHETIC_CASES) {
        const io = parseSafetensors(await readBuffer(`${IO_PREFIX}${caseName}${IO_SUFFIX}`));
        const output = (await session.run(goldenInputs(parsed, io)))[name];
        assert(output.dtype === "f32", `${caseName}: 深度の dtype が ${output.dtype}`);
        assertEquals(output.shape, [1, size, size], `${caseName}: 深度地図の形`);
        correlations.set(caseName, rampCorrelation(output.data, size));
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }

    const rampValue = correlations.get(RAMP_CASE);
    assert(rampValue !== undefined, `${RAMP_CASE} の相関が無い`);
    for (const [caseName, value] of correlations) {
      if (caseName === RAMP_CASE) continue;
      assert(
        rampValue > value,
        `対角ランプとの相関が ${caseName}=${value} で ${RAMP_CASE}=${rampValue} 以上 —` +
          " 出力が入力の幾何を追えていない",
      );
    }
  },
});
