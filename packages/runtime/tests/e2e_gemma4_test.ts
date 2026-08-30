// 実重み Gemma 4 E2B（1-shot causal LM）の実 GPU golden E2E — **混成格納（embedding i8 ×
// linear i4）と層種別 2 本の attention が実モデルで効いていること**の検収。
//
// `e2e_minicpm5_test.ts` が「真の GQA が 24 層の言語モデルとして正しい数値を出す」を受け持つ
// 鏡像元で、こちらが増やして見るのは Gemma 4 に固有の 3 点:
//
// - **MQA 8:1**（k / v の head が 1 本 — minicpm5 の 16:2 より縮退が強い形）
// - **層種別 D**（sliding 層は head_dim 256 / full 層は 512。`hidden_size / heads` = 192 は
//   どちらとも違うので、片方の値で全層を見ると残りが丸ごと無検査になる）
// - **混成量子化**（ADR 0069 決定 5 — 語彙 262144 × PLE 35 表という「embedding が重みの過半を
//   占める」形なので、i4 一本でも i8 一本でも成り立たない）
//
// 門は 4 本:
//
// ① 形 + 格納内訳（attention 35 本の head 軸 / 層種別 D、initializer の格納 dtype 本数）
// ② logits の tolerance 突合（torch CPU 期待値・3 ケース） — {@link GEMMA4_TOLERANCE}
// ③ 最終位置の greedy 一致（数値がずれても 1 位が動かないことの、②とは独立な意味論の線）
// ④ census（`attention_qk` / `attention_pv` が全て `:gqa` + linear / embedding が
//    **圧縮格納のキーで走った**証明 = 適格落ちして黙って f32 に展開されていないこと）
//
// 対象は `outputs/series/gemma4-e2b/`（混成格納でも 3.97GiB あるためリポジトリ管理外 —
// `.gitignore` の `outputs/`。資産 3 根の綴りの正本は tools/export-recipes/_shared/paths.py と
// docs/assets-layout.md）。生成は `tools/export-recipes/gemma4/export.py`（コマンドは下の
// GENERATE_COMMAND がそのまま正本）で、格納内訳・層構成の正本もそこの `assert_ir_form`。
//
// KV cache は載っていない（1-shot = prefill 相当の 1 本道。states 形 decode と KV 共有の検収は
// packages/models/tests/e2e_gemma4_greedy_test.ts）。
//
// 資産が無い環境では**明示 SKIP** する。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。逆に資産が
// **中途半端に**（ケース欠け）存在する場合は SKIP ではなく FAIL にする（下の「資産の完全性」
// テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  parseSafetensors,
  type PreparedModel,
  prepareModel,
  type Tensor,
} from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";
import { modelPresent, readShard, resolveShards, streamShards } from "./helpers/shard-files.ts";

/**
 * 生 logits（`[1,T,262144]`）の torch CPU 期待値との突合に使う許容誤差。
 *
 * 実測（2026-08-19 初回実走・`atol=rtol=0` の素の突合、3 ケース × 出力 1 本 — 煙試験と
 * 公式実走で maxAbs は完全一致 = 決定的）:
 *
 * | ケース     | T   | maxAbs   | maxRel  |
 * | ---------- | --- | -------- | ------- |
 * | capital-en | 6   | 1.802e-4 | 7.46e-1 |
 * | capital-ja | 10  | 2.680e-4 | 6.84e-1 |
 * | context-en | 598 | 2.232e-3 | 4.37e+2 |
 *
 * atol 1e-2 は実測最悪 2.232e-3（context-en）の約 4.5 倍。maxRel が跳ねるのは 0 近傍の参照
 * 要素の見かけ（allclose は分母を下限クリップする — 他の実重み門と同じ形）で、判定は
 * atol 単独に置く。
 *
 * 最悪の context-en 2.23e-3 は minicpm5（7.44e-5）より 1 桁半上だが、logits の値域が softcap で
 * |x| ≤ 30 に張り付く（相対 ~7e-5）・T と D（512）が大きく縮約が深い・標本が 1.57 億要素、の
 * 3 点で説明が付く水準。greedy の門の成立条件（golden margin > 2×atol）は最小 margin 3.56 で
 * atol 1e-2 の 178 倍 — 判定は揺らがない。
 *
 * **rtol は 0**（判定は atol 単独）— 他の実重み門（DeBERTa / EmbeddingGemma / MiniCPM5）と同じ
 * 形。参照値がちょうど 0.0 の要素があると allclose は分母を 1e-12 で下限クリップするので
 * maxRel が跳ねるが、その要素の絶対誤差は atol の遥か下にある。実測表が埋まった時点で、この
 * 段落も「その形だったか」を実測で言い直す。
 *
 * 誤差の出所は他の実重み門と同じ（fma 融合・linear / rms_norm / attention の縮約順序が torch と
 * 違う・超越関数の実装差）に、この系列だけの **dequant 経路**が乗る — ただし丸めは export 側の
 * fake-quant が参照より前に済ませており（ADR 0006）、GPU の `f32(q) * scale` は torch の
 * `round(w/s)*s` と**同じ f32 積**なので、格納 i4 / i8 それ自体は誤差を足さない設計。ここが
 * 崩れていれば実測が他系列より桁で悪くなる形で出る。
 *
 * NOTE: 語彙 262144 × 最長 T=598 = 1.57 億要素を 1 ケースで見るので、最悪値の統計は minicpm5
 * （130560 × 87）より 1 桁多い標本から取られる。同じ実装品質でも maxAbs はやや上に出る。
 */
const GEMMA4_TOLERANCE: Tolerance = { atol: 1e-2, rtol: 0 };

const SERIES_ROOT = new URL("../../../outputs/series/gemma4-e2b/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド（tools/export-recipes/gemma4/README.md と同じ綴り）。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m gemma4.export";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する** — 列挙だけに頼ると生成を
 * 一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/export-recipes/gemma4/export.py` の `GOLDEN_CASES`（T = 6 / 10 / 598）。
 */
const EXPECTED_CASES = ["capital-en", "capital-ja", "context-en"] as const;

/** census / 形の検査に使う `attention` ノードの本数 = 層数（config の `num_hidden_layers`）。 */
const LAYERS = 35;
/** q 側の head 数（config の `num_attention_heads`）。 */
const HEADS = 8;
/** k / v 側の head 数（config の `num_key_value_heads`）— 8:1 の MQA。 */
const KV_HEADS = 1;

/**
 * full attention の層 index（残りは sliding）。正本は `config.layer_types` で、export.py が
 * 層順に mask 定数と head_dim を突き合わせている。**ここでも固定で持つ**のは、層種別が
 * 数値の合ったまま入れ替わる（sliding 層に causal が届く / D を取り違える）事故が、
 * T ≤ 512 のケースだけ見ていると素通りするから。
 */
const FULL_LAYERS: ReadonlySet<number> = new Set([4, 9, 14, 19, 24, 29, 34]);

/** 層種別ごとの head_dim（sliding = `head_dim` / full = `global_head_dim`）。 */
const SLIDING_HEAD_DIM = 256;
const FULL_HEAD_DIM = 512;

const headDimOf = (layer: number): number =>
  FULL_LAYERS.has(layer) ? FULL_HEAD_DIM : SLIDING_HEAD_DIM;

/**
 * 圧縮格納の initializer 本数（`export.py` の summary `form.storage` と同じ数）。
 *
 * i8 = 主 embedding（tied `lm_head` に畳まれた 1 本）+ PLE 35 表 = 36 本。
 * i4 = `lm_head` を除く全 linear = 276 本。
 *
 * MUST: **厳密一致**で見る。圧縮の適格判定を外した重みは例外を出さず f32 のまま残る
 * （`emit._plan_weight_dtype` の既定側）ので、本数を数えないと「1 本だけ f32 に落ちた」が
 * 誰にも見えない。f32 の本数（norm weight / mask 定数 / layer_scalar など）は
 * 定数畳み込みの都合で動きうるので数えない。
 */
const STORAGE_EXPECTATION: Readonly<Record<string, number>> = { i8: 36, i4: 276 };

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

/**
 * ファイル 1 本を `ArrayBuffer` として読む。
 *
 * MUST: view が buffer 全体を覆っているなら slice しない — model.safetensors は 3.97GiB・
 * io.context-en は 627MB で、無条件の `slice` は峰値を倍増させる。
 */
const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
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
    `[karume] ${SERIES_ROOT.pathname} に export 済み資産が無いため実重み Gemma 4 E2B E2E を ` +
      `SKIP する（混成格納でも 3.97GiB につきリポジトリ管理外）。生成: ${GENERATE_COMMAND}`,
  );
}

/** `io.<case>.safetensors` を読んでグラフ入力 1 本ぶんの `Tensor` 表を作る。 */
const loadCase = async (
  caseName: string,
  inputSpecs: readonly { readonly name: string; readonly dtype: "f32" | "i32" | "bool" }[],
): Promise<{ readonly inputs: Record<string, Tensor>; readonly expected: Tensor }> => {
  const io = parseSafetensors(await readBuffer(SERIES_ROOT, `${IO_PREFIX}${caseName}${IO_SUFFIX}`));
  // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
  assertEquals(
    [...io.tensors.keys()].sort(),
    [...inputSpecs.map((spec) => `input.${spec.name}`), "output.0"].sort(),
    `${IO_PREFIX}${caseName}${IO_SUFFIX} のテンソルキー`,
  );
  const inputs: Record<string, Tensor> = {};
  for (const spec of inputSpecs) {
    const view = io.tensors.get(`input.${spec.name}`);
    assert(view !== undefined, `input.${spec.name} が ${caseName} の golden に無い`);
    inputs[spec.name] = ioTensor(io, view, spec.dtype);
  }
  const output = io.tensors.get("output.0");
  assert(output !== undefined, `output.0 が ${caseName} の golden に無い`);
  return { inputs, expected: ioTensor(io, output, "f32") };
};

/**
 * 最終位置（`[1,T,V]` の T−1 行）の 1 位トークンと**2 位との差**。
 *
 * 走査は `>` の狭義比較なので、同値のときは**小さい添字**が残る（GPU 側と golden 側で同じ規則
 * を使う）。`margin` は「この差より小さい数値差では 1 位が動かない」という判定の余裕そのもの。
 */
const greedyTop = (
  logits: Tensor,
  where: string,
): { readonly top: number; readonly margin: number } => {
  assert(logits.dtype === "f32", `${where}: logits が f32 でない`);
  assertEquals(logits.shape.length, 3, `${where}: logits の rank`);
  const vocab = logits.shape[2];
  const row = logits.data.subarray((logits.shape[1] - 1) * vocab, logits.shape[1] * vocab);
  let top = 0;
  let best = row[0];
  let second = Number.NEGATIVE_INFINITY;
  for (let index = 1; index < row.length; index += 1) {
    const value = row[index];
    if (value > best) {
      second = best;
      best = value;
      top = index;
    } else if (value > second) {
      second = value;
    }
  }
  return { top, margin: best - second };
};

/**
 * ①形と格納内訳（**数値が合ったまま静かに壊れる**性質だけを見る）。
 *
 * MUST: この検査を落とさない。`repeat_kv` 実体化形（Hkv=8）の資産は**数値が完全に同じ**なので
 * ②の tolerance 突合も③の greedy も素通りし、MQA の検収という意味だけが静かに消える
 * （export.py の `assert_ir_form` が書き手側で見ているのと同じ性質を、読み手側でも見る）。
 * 層種別 D と格納内訳も同じ性質 — 前者は T ≤ 512 のケースだけなら数値が合い、後者は
 * f32 に落ちても正しい値が出る。
 */
const assertGemma4Form = (model: PreparedModel): number => {
  const graph = model.graph;
  const attentions = graph.nodes.filter((node) => node.op === "attention");
  assertEquals(attentions.length, LAYERS, "attention ノードの本数（= 層数）");
  attentions.forEach((node, layer) => {
    const type = FULL_LAYERS.has(layer) ? "full" : "sliding";
    // 1-shot 形の attention は q / k / v / mask の 4 本（states 形は 3 本 — decode 門の担当）。
    const shapes = node.ins.slice(0, 3).map((name) => graph.values[name].shape);
    assertEquals(
      shapes.map((shape) => shape[1]),
      [HEADS, KV_HEADS, KV_HEADS],
      `attention[${layer}] (${type}): head 軸`,
    );
    assertEquals(
      shapes.map((shape) => shape[3]),
      [headDimOf(layer), headDimOf(layer), headDimOf(layer)],
      `attention[${layer}] (${type}): D 軸（層種別の head_dim）`,
    );
  });

  const census: Record<string, number> = {};
  for (const initializer of Object.values(graph.initializers)) {
    const dtype = initializer.storage.dtype;
    census[dtype] = (census[dtype] ?? 0) + 1;
  }
  for (const [dtype, expected] of Object.entries(STORAGE_EXPECTATION)) {
    assertEquals(
      Object.hasOwn(census, dtype) ? census[dtype] : 0,
      expected,
      `格納 ${dtype} の initializer 本数（全内訳 ${JSON.stringify(census)}）— ` +
        `適格判定を外した重みは黙って f32 のまま残る`,
    );
  }
  return attentions.length;
};

Deno.test({
  name: "Gemma 4 E2B 資産: 期待するケースとモデル本体が揃っている",
  // 完全に空の環境だけ「生成していない」として SKIP。**何か 1 つでも**あれば欠けは FAIL
  //（モデルだけ残って golden が全滅した欠損も拾う — `ANY_PRESENT` の JSDoc）。
  ignore: !ANY_PRESENT,
  fn: () => {
    assertEquals(CASES, [...EXPECTED_CASES], `${SERIES_ROOT.pathname} の golden ケース`);
    assert(modelPresent(new URL(MODEL_FILE, SERIES_ROOT)), `${MODEL_FILE} が無い`);
  },
});

/**
 * ②logits の tolerance 突合 と ③最終位置 greedy の一致（実 GPU / torch CPU 期待値）。
 *
 * MUST: Session は **1 本だけ組んで 3 ケースを順に run する**。この資産は 3.97GiB で、ケース
 * ごとに組み直すと同じアップロードを 3 回払う。記号次元 T は golden の入力 shape の実長から
 * 束縛されるので（明示 bindings は渡さない）、**同じ Session で T = 6 / 10 / 598 を跨ぐ**ことが
 * そのまま「宣言上限 Tmax = 768 に依存した実装（プランを Tmax で組む・mask 定数を Tmax の
 * まま食う）」の検出線にもなる。T=598 は `sliding_window` 512 を超える唯一のケースで、
 * sliding 層の帯が causal と一致しない形が実際に評価されるのはここだけ。
 */
Deno.test({
  name: "Gemma 4 E2B golden 突合: 3 ケースの logits と最終位置 greedy（実 GPU / torch CPU 期待値）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async () => {
    const shards = resolveShards(new URL(MODEL_FILE, SERIES_ROOT));
    const parsed = prepareModel(await readShard(shards[0]));
    assertEquals(parsed.graph.inputs.map((spec) => spec.name), ["input_ids"], "グラフ入力");
    assertEquals(parsed.graph.outputs.length, 1, "graph.outputs の本数（1-shot の logits 1 本）");
    const outputName = parsed.graph.outputs[0];
    const declared = parsed.graph.values[outputName].dtype;
    // MUST: 形と格納の検査は数値門でも独立に持つ — 下の census は timestamp-query が無い device
    // で SKIP するので、そこに結線しておくと `repeat_kv` 実体化形へ再エクスポートした資産や
    // f32 に落ちた資産が数値だけ通り、MQA / 混成格納の検収でなくなる。
    assertGemma4Form(parsed);

    const gpu = await acquireGpu();
    const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
    try {
      /** ケースごとの最終位置 1 位（全ケース同一 = 定数出力の検出に使う）。 */
      const tops: number[] = [];
      for (const caseName of CASES) {
        const { inputs, expected } = await loadCase(caseName, parsed.graph.inputs);
        const outputs = await session.run(inputs);
        assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());
        const actual = outputs[outputName];
        const where = `${caseName} output.0 ('${outputName}')`;
        assertEquals(actual.shape, expected.shape, `${where}: shape`);
        assertEquals(actual.dtype, declared, `${where}: dtype`);

        // ② 数値
        const report = compareTensors(actual, expected, GEMMA4_TOLERANCE);
        assert(report.pass, `${where}: ${formatAllclose(report)}`);

        // ③ 意味論（最終位置の 1 位）。②が緩んでも独立に残る線で、MQA の head 写像違い・
        // 層種別 mask の取り違え・RoPE の位置ずれ・PLE の層割り付け違いはどれもここで 1 位を
        // 動かす。
        const golden = greedyTop(expected, `${caseName} golden`);
        const observed = greedyTop(actual, `${caseName} GPU`);
        // 判定が成立する形であることを先に固定する（この門が運任せでないことの根拠）:
        // golden の 1 位と 2 位の差が atol の 2 倍を超えていれば、②の許容内の数値差で 1 位は
        // 動けない。
        assert(
          golden.margin > 2 * GEMMA4_TOLERANCE.atol,
          `${caseName}: golden の 1 位 / 2 位の差 ${golden.margin} が atol と同程度 — ` +
            `この形では greedy 一致が数値差で反転しうる（門として成立しない）`,
        );
        assertEquals(
          observed.top,
          golden.top,
          `${caseName}: 最終位置の 1 位が golden と違う（GPU 余裕 ${observed.margin} / ` +
            `golden 余裕 ${golden.margin}）`,
        );
        console.log(
          `[e2e] gemma4 1-shot ${caseName}: T=${expected.shape[1]} / top=${observed.top} / ` +
            `golden 余裕 ${golden.margin.toExponential(3)} / ${formatAllclose(report)}`,
        );
        tops.push(observed.top);
      }
      // 恒真化の門: 全ケースの 1 位が同一なら定数出力（export.py の `_sanity` と同じ独立線を
      // ランタイム側にも置く）。期待は ` Paris` / `東京` の 2 種（export.py の
      // `GREEDY_EXPECTATIONS`）。
      assert(new Set(tops).size > 1, `全ケースの最終位置の 1 位が同一 ${tops[0]} — 定数出力`);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/** 内訳 1 本ぶん（キーと dispatch 数だけを見る — 実時間はこの門の関心事ではない）。 */
type TimingEntry = { readonly key: string; readonly dispatchCount: number };

const dispatches = (of: readonly TimingEntry[]): number =>
  of.reduce((total, entry) => total + entry.dispatchCount, 0);

/**
 * **混成格納が実際に走った証明**（ADR 0069 決定 5）。数値（②）はキーを区別しない — 適格から
 * 落ちて CPU で f32 展開された重みも、正しく実装されていれば同じ値を出す。**キーだけがこの
 * 区別をする**。
 *
 * - `linear:` は全て `:wi4g32`（i4 群量子化）か `:wi8`（tied `lm_head`）を含む。裸の
 *   `linear:v2:f32:…` が 1 本でも出たら、その重みは適格判定を外れて f32 に展開されている。
 * - `embedding:` は全て `:wi8`（主 embedding + PLE 35 表）。i4 の実行経路は embedding にも
 *   あるが、**この系列は embedding を i8 に割り付けている**（`gemma4/export.py` の混成指定）—
 *   ここに `:wi4g32` が出たら割り付けが黙って変わったということ。
 * - 両方の linear 変種が出ること = 混成そのもの（片側だけなら「一様格納が通っただけ」）。
 */
const assertStorageKeys = (entries: readonly TimingEntry[], shown: string): void => {
  const linear = entries.filter((entry) => entry.key.startsWith("linear:"));
  assert(linear.length > 0, `linear の内訳が無い（走った内訳: ${shown}）`);
  assertEquals(
    linear
      .filter((entry) => !entry.key.includes(":wi4g32") && !entry.key.includes(":wi8"))
      .map((entry) => entry.key),
    [],
    `圧縮格納でない linear が走った（適格落ちで f32 展開された重み — 走った内訳: ${shown}）`,
  );
  assert(
    linear.some((entry) => entry.key.includes(":wi4g32")) &&
      linear.some((entry) => entry.key.includes(":wi8")),
    `linear の格納が 1 種類しか走っていない（混成でない — 走った内訳: ${shown}）`,
  );

  const embedding = entries.filter((entry) => entry.key.startsWith("embedding:"));
  assert(embedding.length > 0, `embedding の内訳が無い（走った内訳: ${shown}）`);
  assertEquals(
    embedding.filter((entry) => !entry.key.includes(":wi8")).map((entry) => entry.key),
    [],
    `i8 格納でない embedding が走った（走った内訳: ${shown}）`,
  );
};

/**
 * ④census（ADR 0058 決定 4 / 0067 受入条件 ③ / 0069 決定 5）。実モデルの `attention` 35 本が
 * **全て GQA 変種のパイプラインで走った**ことと、linear / embedding が**圧縮格納のまま走った**
 * ことを、パイプラインキーの側から見る。
 *
 * MUST: 計測を要求しない device（`TIMESTAMP_QUERY_AVAILABLE` が偽）では**明示 SKIP** する。
 * `TIMING_ACQUIRE_OPTIONS` は feature 不在で `gpuTiming: false` に落ちるので、そのまま走らせると
 * `lastRunTiming` が undefined になり、この検査が黙って空振りして緑になる。
 * NOTE: 数値の門（上のテスト）は素の `acquireGpu()` で走らせる — 常用と同じ device 構成のまま
 * 突合したいので、timestamp 書き込みが混ざる構成はこの 1 ケースに閉じる。
 */
Deno.test({
  name: "Gemma 4 E2B census: MQA 35 本と混成格納のキー（実 GPU / timestamp-query）",
  ignore: !AVAILABLE || !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const shards = resolveShards(new URL(MODEL_FILE, SERIES_ROOT));
    const parsed = prepareModel(await readShard(shards[0]));
    const layers = assertGemma4Form(parsed);

    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
    try {
      // 混成格納の常駐そのもの（ADR 0069 の検収条件 — キー検査と独立の実測線）。適格落ちは
      // 例外を出さず CPU で f32 展開されるだけなので、hostExpandedBytes が唯一の直接観測。
      const storage = session.diagnostics().storage;
      assert(storage !== undefined, "diagnostics.storage が無い");
      assertEquals(storage.hostExpandedBytes, 0, "hostExpandedBytes（適格落ちの CPU 展開）");
      assert(
        storage.residentCompressedBytes > 3_000_000_000,
        `residentCompressedBytes ${storage.residentCompressedBytes} が混成常駐の規模でない`,
      );

      // 最短のケース（T=6）1 本で足りる — 見るのは走ったパイプラインの種類と本数。
      const { inputs } = await loadCase("capital-en", parsed.graph.inputs);
      await session.run(inputs);
      const entries = session.diagnostics().lastRunTiming?.entries;
      assert(entries !== undefined, "lastRunTiming が無い（計測が有効な device のはず）");
      assert(entries.length > 0, "内訳が空（キー検査が空振りしている）");
      const shown = entries.map((entry) => `${entry.key}×${entry.dispatchCount}`).join(" / ");

      // ①QK と ③PV（②stats は行統計で GQA の軸を持たないので対象外 — キーにも `:gqa` は付かない）
      const qk = entries.filter((entry) => entry.key.startsWith("attention_qk"));
      const pv = entries.filter((entry) => entry.key.startsWith("attention_pv"));
      assert(qk.length > 0 && pv.length > 0, `attention の内訳が無い: ${entries.length} 本`);
      assertEquals(
        [...qk, ...pv].filter((entry) => !entry.key.endsWith(":gqa")).map((entry) => entry.key),
        [],
        `8:1 の MQA なのに非 GQA キーが走った（走った内訳: ${shown}）`,
      );

      // 全層ぶん dispatch されている（1 種のパイプラインを 35 層が共有するので実測は 35 本ずつ。
      // 行ブロック分割〈ADR 0060 / 0067 決定 7〉が保存経路へ来れば増える側なので下限で見る）。
      assert(
        dispatches(qk) >= layers && dispatches(pv) >= layers,
        `attention の dispatch が層数 ${layers} に足りない: ` +
          `①QK ${dispatches(qk)} / ③PV ${dispatches(pv)}`,
      );

      assertStorageKeys(entries, shown);
      console.log(
        `[e2e] gemma4 1-shot census: QK ${dispatches(qk)} / PV ${dispatches(pv)} / ` +
          `linear ${dispatches(entries.filter((entry) => entry.key.startsWith("linear:")))} / ` +
          `embedding ${dispatches(entries.filter((entry) => entry.key.startsWith("embedding:")))}`,
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
