// 実重み MiniCPM5-1B（1-shot causal LM）の実 GPU golden E2E — **GQA が実モデルで効いている
// こと**の検収（ADR 0067 決定 1 / 受入条件 ③ の e2e 版）。
//
// tests/gpu_attention_gqa_test.ts が合成 1 ノードで「repeat_kv 実体化版とビット同一」＋故障注入を
// 受け持つのに対し、こちらは**実重み 24 層・真の GQA 形 16:2 が言語モデルとして正しい数値を
// 出す**ことを受け持つ。門は 3 本:
//
// ① logits の tolerance 突合（torch CPU 期待値・4 ケース） — {@link MINICPM5_TOLERANCE}
// ② 最終位置の greedy 一致（数値がずれても 1 位が動かないことの、①とは独立な意味論の線）
// ③ census（`attention_qk` / `attention_pv` が**全て `:gqa`** = GQA 経路が実際に走った証明）
//
// 対象は `outputs/series/minicpm5-1b/`（f32 重み 4.03GiB のためリポジトリ管理外 — `.gitignore` の
// `outputs/`。資産 3 根の綴りの正本は tools/export-recipes/_shared/paths.py と
// docs/assets-layout.md）。生成は `tools/export-recipes/minicpm5/export.py`（コマンドは下の
// GENERATE_COMMAND がそのまま正本）。
//
// 格納 dtype 系列は f32 の 1 本だけ（f16 / i8 / w4 は別系列で決める話 — export.py の
// `DEFAULT_OUT_DIR`）なので DeBERTa のような系列パラメタ化はしない。KV cache も載っていない
// （1-shot = prefill 相当の 1 本道で、state スロットと decode 経路は ADR 0066 の実装波）。
//
// 資産が無い環境では**明示 SKIP**する。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。逆に資産が
// **中途半端に**（ケース欠け）存在する場合は SKIP ではなく FAIL にする（下の「資産の完全性」
// テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, createSession, openModel, parseSafetensors, type Tensor } from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

/**
 * 生 logits（`[1,T,130560]`）の torch CPU 期待値との突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本）:
 *
 * | ケース     | T  | maxAbs  | maxRel   | maxRel(ref≠0) | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ---------- | -- | ------- | -------- | ------------- | ------------ | ------------------ |
 * | capital-en | 6  | 8.44e-5 | 1.81     | 1.81          | 19.606       | 1.10e-5            |
 * | capital-ja | 12 | 7.00e-5 | 4.22     | 4.22          | 20.855       | 3.58e-6            |
 * | context-ja | 61 | 1.39e-4 | 1.55e6   | 53.37         | 24.529       | 2.38e-7            |
 * | context-en | 87 | 7.44e-5 | 3.78e6   | 13.73         | 23.678       | 1.79e-6            |
 *
 * atol 1e-3 は実測最悪 1.39e-4（context-ja）の約 7.2 倍。
 *
 * **rtol は 0**（判定は atol 単独）。maxRel が context 2 ケースで 1e6 級に跳ねるのは、その 2
 * ケースに**参照値がちょうど 0.0 の要素が 1 本ずつある**からで（allclose は分母を 1e-12 で
 * 下限クリップする）、その要素の絶対誤差は 1.55e-6 / 3.78e-6 = atol の 1/265 以下しかない。
 * ref≠0 に限っても maxRel は 53.4 まで伸びるが、それは \|ref\| 2.38e-7 の 0 近傍要素の見かけで、
 * その要素の絶対誤差は 1.27e-5。下限項として rtol 1e-6 を足しても \|ref\| 上端 24.5 での寄与は
 * 2.45e-5 = atol の 1/41 と判定を動かさないので置かない（DeBERTa / EmbeddingGemma と同じ形）。
 *
 * 誤差の出所は他の実重み門と同じ（fma 融合・linear / rms_norm / attention の縮約順序が torch と
 * 違う・超越関数（exp / rsqrt / sigmoid）の実装差）。**T では単調に伸びない** — 最悪は最長の
 * context-en（T=87）ではなく context-ja（T=61）で、logits の値域（\|ref\| 上端 19.6〜24.5）が
 * ケースごとに違うことの反映でしかない。DeBERTa 24 層の実測最悪 8.32e-5（`atol=5e-4`）と**同桁**
 * で、1.67 倍に収まった。実装バグ（GQA の head 写像違い・mask の向き・RoPE の位置ずれ・重みの
 * 取り違え）の誤差は出力の値域と同じ O(1)〜O(25) で、この閾値の 4 桁以上上に出る。
 */
const MINICPM5_TOLERANCE: Tolerance = { atol: 1e-3, rtol: 0 };

const SERIES_ROOT = new URL("../../../outputs/series/minicpm5-1b/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド（tools/export-recipes/minicpm5/README.md と同じ綴り）。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m minicpm5.export";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する** — 列挙だけに頼ると生成を
 * 一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/export-recipes/minicpm5/export.py` の `GOLDEN_CASES`（T = 6 / 12 / 87 / 61）。
 */
const EXPECTED_CASES = ["capital-en", "capital-ja", "context-en", "context-ja"] as const;

/** census / 形の検査に使う `attention` ノードの本数 = 層数（config の `num_hidden_layers`）。 */
const LAYERS = 24;
/** q 側の head 数（config の `num_attention_heads`）。 */
const HEADS = 16;
/** k / v 側の head 数（config の `num_key_value_heads`）— 16:2 で r = 8。 */
const KV_HEADS = 2;

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
 * MUST: view が buffer 全体を覆っているなら slice しない — model.safetensors は 4.03GiB で、
 * 他の実重み門が使っている無条件の `slice` はここでは峰値を 8GiB に倍増させる。
 */
const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
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

/** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
const CASES = discoverCases(SERIES_ROOT);
/** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
const AVAILABLE = CASES.length > 0;
/**
 * **何か 1 つでも**残っているか（完全性テストの SKIP 述語 — Codex 波 H 指摘 H-02）。
 * io が全滅してモデルだけ残った欠損は `AVAILABLE` では偽になり、`ignore: !AVAILABLE` だと
 * 完全性テスト自身が SKIP される — 欠損を FAIL にする述語は「完全に空」でだけ寝てよい。
 */
const ANY_PRESENT = AVAILABLE || fileExists(new URL(MODEL_FILE, SERIES_ROOT));

if (!AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に export 済み資産が無いため実重み MiniCPM5-1B E2E を ` +
      `SKIP する（f32 重み 4.03GiB につきリポジトリ管理外）。生成: ${GENERATE_COMMAND}`,
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

/** `attention` ノードの宣言 shape が真の GQA 形（q は H・k / v は Hkv）であることを見る。 */
const assertGqaForm = (model: ReturnType<typeof openModel>): number => {
  const attentions = model.graph.nodes.filter((node) => node.op === "attention");
  assertEquals(attentions.length, LAYERS, "attention ノードの本数（= 層数）");
  attentions.forEach((node, index) => {
    // MUST: この検査を落とさない。`repeat_kv` 実体化形（Hkv=16）の資産は**数値が完全に同じ**
    // なので tolerance 突合も greedy も素通りし、GQA の検収という意味だけが静かに消える
    // （export.py の `assert_ir_form` が書き手側で見ているのと同じ性質を、読み手側でも見る）。
    const heads = node.ins.slice(0, 3).map((name) => model.graph.values[name].shape[1]);
    assertEquals(heads, [HEADS, KV_HEADS, KV_HEADS], `attention[${index}]: head 軸`);
  });
  return attentions.length;
};

Deno.test({
  name: "MiniCPM5 資産: 期待するケースとモデル本体が揃っている",
  // 完全に空の環境だけ「生成していない」として SKIP。**何か 1 つでも**あれば欠けは FAIL
  //（モデルだけ残って golden が全滅した欠損も拾う — `ANY_PRESENT` の JSDoc）。
  ignore: !ANY_PRESENT,
  fn: () => {
    assertEquals(CASES, [...EXPECTED_CASES], `${SERIES_ROOT.pathname} の golden ケース`);
    assert(fileExists(new URL(MODEL_FILE, SERIES_ROOT)), `${MODEL_FILE} が無い`);
  },
});

/**
 * ①logits の tolerance 突合 と ②最終位置 greedy の一致（実 GPU / torch CPU 期待値）。
 *
 * MUST: Session は **1 本だけ組んで 4 ケースを順に run する**（1 ケース 1 テストの他の実重み門と
 * 違う点）。この資産は 4.03GiB の f32 重みで、ケースごとに組み直すと同じアップロードを 4 回
 * 払う。記号次元 T は golden の入力 shape の実長から束縛されるので（明示 bindings は渡さない）、
 * **同じ Session で T = 6 / 12 / 87 / 61 を跨ぐ**ことがそのまま「宣言上限 Tmax = 512 に依存した
 * 実装（プランを Tmax で組む・causal 定数を Tmax のまま食う）」の検出線にもなる。
 */
Deno.test({
  name: "MiniCPM5 golden 突合: 4 ケースの logits と最終位置 greedy（実 GPU / torch CPU 期待値）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async () => {
    const parsed = openModel(await readBuffer(SERIES_ROOT, MODEL_FILE));
    assertEquals(parsed.graph.inputs.map((spec) => spec.name), ["input_ids"], "グラフ入力");
    assertEquals(parsed.graph.outputs.length, 1, "graph.outputs の本数（1-shot の logits 1 本）");
    const outputName = parsed.graph.outputs[0];
    const declared = parsed.graph.values[outputName].dtype;
    // MUST: 形の検査は数値門でも独立に持つ — 下の census は timestamp-query が無い device で
    // SKIP するので、そこに結線しておくと `repeat_kv` 実体化形（Hkv=16）へ再エクスポートした
    // 資産が数値だけ通って GQA の検収でなくなる（数値は実体化形と完全に同じ）。
    assertGqaForm(parsed);

    const gpu = await acquireGpu();
    const session = await createSession(gpu, parsed);
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

        // ① 数値
        const report = compareTensors(actual, expected, MINICPM5_TOLERANCE);
        assert(report.pass, `${where}: ${formatAllclose(report)}`);

        // ② 意味論（最終位置の 1 位）。①が緩んでも独立に残る線で、GQA の head 写像違い・
        // mask の向き・RoPE の位置ずれはどれもここで 1 位を動かす。
        const golden = greedyTop(expected, `${caseName} golden`);
        const observed = greedyTop(actual, `${caseName} GPU`);
        // 判定が成立する形であることを先に固定する（この門が運任せでないことの根拠）:
        // golden の 1 位と 2 位の差が atol の 2 倍を超えていれば、①の許容内の数値差で 1 位は
        // 動けない。実測の最小余裕は context-ja の 0.188 = atol 1e-3 の 188 倍。
        assert(
          golden.margin > 2 * MINICPM5_TOLERANCE.atol,
          `${caseName}: golden の 1 位 / 2 位の差 ${golden.margin} が atol と同程度 — ` +
            `この形では greedy 一致が数値差で反転しうる（門として成立しない）`,
        );
        assertEquals(
          observed.top,
          golden.top,
          `${caseName}: 最終位置の 1 位が golden と違う（GPU 余裕 ${observed.margin} / ` +
            `golden 余裕 ${golden.margin}）`,
        );
        tops.push(observed.top);
      }
      // 恒真化の門: 全ケースの 1 位が同一なら定数出力（export.py の `_sanity` と同じ独立線を
      // ランタイム側にも置く）。実測は ` Paris` / `東京` の 2 種が交互に出る。
      assert(new Set(tops).size > 1, `全ケースの最終位置の 1 位が同一 ${tops[0]} — 定数出力`);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * ③census（ADR 0067 受入条件 ③ / ADR 0058 決定 4）。実モデルの `attention` 24 本が**全て GQA
 * 変種のパイプラインで走った**ことを、パイプラインキーの側から見る。
 *
 * 16:2 なので非 GQA キー（`:gqa` 無し）が 1 本でも出たら、その層は k / v を H まで広げた形で
 * 走ったか、GQA 判定が落ちている。数値は tolerance 内に収まりうるので、**キーだけがこの
 * 区別をする**。
 *
 * MUST: 計測を要求しない device（`TIMESTAMP_QUERY_AVAILABLE` が偽）では**明示 SKIP** する。
 * `TIMING_ACQUIRE_OPTIONS` は feature 不在で `gpuTiming: false` に落ちるので、そのまま走らせると
 * `lastRunTiming` が undefined になり、この検査が黙って空振りして緑になる。
 * NOTE: 数値の門（上のテスト）は素の `acquireGpu()` で走らせる — 常用と同じ device 構成のまま
 * 突合したいので、timestamp 書き込みが混ざる構成はこちらの 1 ケースに閉じる。
 */
Deno.test({
  name: "MiniCPM5 census: attention 24 本が全て GQA 変種のキーで走る（実 GPU / timestamp-query）",
  ignore: !AVAILABLE || !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const parsed = openModel(await readBuffer(SERIES_ROOT, MODEL_FILE));
    const layers = assertGqaForm(parsed);

    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    const session = await createSession(gpu, parsed);
    try {
      // 最短のケース（T=6）1 本で足りる — 見るのは走ったパイプラインの種類と本数。
      const { inputs } = await loadCase("capital-en", parsed.graph.inputs);
      await session.run(inputs);
      const entries = session.diagnostics().lastRunTiming?.entries;
      assert(entries !== undefined, "lastRunTiming が無い（計測が有効な device のはず）");
      assert(entries.length > 0, "内訳が空（キー検査が空振りしている）");

      // ①QK と ③PV（②stats は行統計で GQA の軸を持たないので対象外 — キーにも `:gqa` は付かない）
      const qk = entries.filter((entry) => entry.key.startsWith("attention_qk"));
      const pv = entries.filter((entry) => entry.key.startsWith("attention_pv"));
      assert(qk.length > 0 && pv.length > 0, `attention の内訳が無い: ${entries.length} 本`);
      const plain = [...qk, ...pv].filter((entry) => !entry.key.endsWith(":gqa"));
      assertEquals(
        plain.map((entry) => entry.key),
        [],
        `16:2 の GQA なのに非 GQA キーが走った（走った内訳: ${
          [...qk, ...pv].map((entry) => `${entry.key}×${entry.dispatchCount}`).join(" / ")
        }）`,
      );

      // 全層ぶん dispatch されている（1 種のパイプラインを 24 層が共有するので実測は 24 本ずつ。
      // 行ブロック分割〈ADR 0060 / 0067 決定 7〉が保存経路へ来れば増える側なので下限で見る）。
      const dispatches = (of: readonly { readonly dispatchCount: number }[]): number =>
        of.reduce((total, entry) => total + entry.dispatchCount, 0);
      assert(
        dispatches(qk) >= layers && dispatches(pv) >= layers,
        `attention の dispatch が層数 ${layers} に足りない: ` +
          `①QK ${dispatches(qk)} / ③PV ${dispatches(pv)}`,
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
