// 実重み DeBERTa-v2（SBV2 text front が使う BERT）の実 GPU golden E2E（ADR 0005 の段 3）。
//
// tiny golden（tests/e2e_golden_test.ts）が「op 契約の被覆」を、実重み SBV2 / Irodori が
// 「音響チェーン側の実重み」を受け持つのに対し、こちらは**テキスト側 24 層の数値一致**を
// 受け持つ。この門が無い間、text_encoder の回帰は SBV2 の WAV sha256 門の「1 bit 差」と
// してしか現れず原因の局在ができなかった（ADR 0026 の「検出器の現況」/ ADR 0045 の
// Consequences / docs/research/2026-08-11-deberta-size-recon.md「既知の欠落」）。
//
// 対象は `outputs/series/deberta{,-i8}/<variant>/`（重み 307MB〜1.3GB のためリポジトリ管理外
// — `.gitignore` の `outputs/`）。生成は `tools/export-recipes/deberta/export.py`
// （コマンドは下の各 variant の `generate` がそのまま正本 — 綴りは
// tools/export-recipes/deberta/README.md と同じ）。
//
// ## 実走する variant（格納 dtype 系列 × 層数）
//
// 系列は格納 dtype で分かれる（ADR 0019 — f32 は `deberta/`・i8 は `deberta-i8/`。同居させると
// f32 の tolerance が黙って i8 資産に掛かる）。層数 variant は同じグラフの長短で、**24 層は
// 全層の hidden_states 25 本出し**（層別に誤差の伸びが読めるので tolerance の導出がここで
// 実測になる — ADR 0026 決定 2）、**22 層は SBV2 の配布形で最終層 1 本出し**（ADR 0045）。
// **tolerance は variant ごとに独立に実測から導く**（系列間・variant 間で流用しない）。
//
// `deberta/{dev-2layer,sbv2-22layer}` と `deberta-i8/dev-2layer` は**宣言しない** — 参照環境に
// 資産が無く tolerance を実測から導けないため（値を発明しない）。足すときは下の 3 本と同じ
// 手順（`atol=rtol=0` の素の突合 → 実測最悪の 5〜10 倍）で導いてから宣言する。
//
// ## w8a8 の鏡像 io は拾わない
//
// `--act-quant` が書く `io-i8a8.<case>` は w8a8（`SessionOptions.linearCompute: "i8a8"`）の
// torch 鏡像で、**この門の期待値ではない**（w8 = f32 計算の期待値として読むと活性量子化ごと
// 汚染される）。通常ケースの列挙が `io.` の startsWith なのはそのため（正本は
// `deberta/export.py` の `IO_PREFIX` / `ACT_IO_PREFIX` の docstring）。鏡像を消費する
// `e2e_deberta_w8a8_test.ts`（ADR 0026 決定 3）はリポジトリに無い。
//
// 資産が無い環境では**その variant を SKIP** する。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。
// 逆に、資産が**中途半端に**存在する場合（ケース欠け）は SKIP ではなく FAIL にする（下の
// 「資産の完全性」テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, createSession, openModel, parseSafetensors, type Tensor } from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * **f32 系列 / full-24layer**（全層 hidden_states 25 本出し）の torch CPU 期待値との突合に
 * 使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 25 本・maxAbs は 25 本の最悪と、それが
 * 出た出力番号）:
 *
 * | ケース   | T  | maxAbs           | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | -- | ---------------- | ------- | ------------ | ------------------ |
 * | case0    | 11 | 3.08e-5 (out22)  | 7.88    | 28.659       | 1.22e-7            |
 * | case1    | 26 | 2.48e-5 (out23)  | 5.31e-1 | 28.856       | 1.01e-6            |
 * | case2    | 35 | 8.32e-5 (out19)  | 5.31e-1 | 28.903       | 6.77e-6            |
 * | padded   | 16 | 2.58e-5 (out18)  | 3.30    | 28.892       | 1.96e-7            |
 *
 * atol 5e-4 は実測最悪 8.32e-5（case2 の output.19）の約 6.0 倍。
 *
 * **rtol は 0**（判定は atol 単独）。rtol を主役に据えられないのは \|ref\| 最小非ゼロが
 * 1.22e-7 まで薄く広がるからで、実測 maxRel 7.88 はその 0 近傍要素の見かけでしかない
 * （その要素の絶対誤差は 1e-6 級）。下限項として rtol 1e-6 を足しても \|ref\| 上端 28.9 で
 * 寄与は 2.9e-5 = atol の 1/17 と判定を動かさないので置かない。
 *
 * 誤差の出所は tiny golden と同じ（fma 融合・linear / layer_norm / attention の縮約順序が
 * torch と違う・超越関数（gelu / exp / rsqrt）の実装差）。**層数で単調に伸びる**のがこの
 * モデルの性質で、埋め込み出力（output.0）は 9.54e-7 なのに対し 18〜23 層目で 2〜8e-5 に
 * なる（SBV2 の dp のように P を 256 倍にしても伸びない形とは構造が違う）。全層出しの
 * variant を残しているのはこの伸びを実測で読むため。実装バグ（相対位置表の添字ずれ・
 * head 分割誤り・マスク経路の取りこぼし）の誤差は出力の値域と同じ O(1)〜O(29) で、この
 * 閾値の 4 桁以上上に出る。
 */
const DEBERTA_TOLERANCE: Tolerance = { atol: 5e-4, rtol: 0 };

/**
 * **i8 系列 / full-24layer**（per-channel i8 格納・計算は f32 = w8 — ADR 0019）の許容誤差。
 *
 * MUST: f32 系列の値を流用しない。golden は fake-quant 後の重みで採ってあるので**量子化誤差は
 * 差に入らない**（ADR 0006 の方法論）が、丸め後の重みは値そのものが変わるため縮約の丸め方も
 * 変わる。
 *
 * 実測（`atol=rtol=0`、4 ケース × 出力 25 本）:
 *
 * | ケース   | T  | maxAbs           | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | -- | ---------------- | ------- | ------------ | ------------------ |
 * | case0    | 11 | 4.44e-5 (out18)  | 2.82e-1 | 28.661       | 4.67e-6            |
 * | case1    | 26 | 2.67e-5 (out18)  | 1.25    | 28.855       | 3.72e-7            |
 * | case2    | 35 | 1.23e-4 (out19)  | 4.54e-1 | 28.898       | 1.71e-7            |
 * | padded   | 16 | 4.10e-5 (out18)  | 4.88e-1 | 28.913       | 4.23e-6            |
 *
 * atol 7e-4 は実測最悪 1.23e-4（case2 の output.19）の約 5.7 倍。rtol が 0 な理由は f32 系列と
 * 同じ（\|ref\| 最小非ゼロ 1.71e-7 で rtol は主役になれず、1e-6 の寄与 2.9e-5 は atol の 1/24）。
 *
 * 実測最悪 1.23e-4 が f32 系列の 8.32e-5 と**同桁**なのは「golden が量子化後の重みで採れて
 * いる」ことの裏取りでもある（掛け忘れなら差は i8 の量子化誤差そのもの = 1e-1 級で 3 桁上に
 * 出る）。この 2 値は ADR 0026 の歴史値（f32 8.32e-5 / i8 1.23e-4・atol 7e-4）と一致した。
 */
const DEBERTA_I8_TOLERANCE: Tolerance = { atol: 7e-4, rtol: 0 };

/**
 * **i8 系列 / sbv2-22layer**（22 層カットの配布形・最終層 1 本出し — ADR 0045）の許容誤差。
 *
 * MUST: full-24layer の値を流用しない（層数が違えば誤差の積み上がりも違う）。実測
 * （`atol=rtol=0`、4 ケース × 出力 1 本）:
 *
 * | ケース   | T  | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | -- | ------- | ------- | ------------ | ------------------ |
 * | case0    | 11 | 2.26e-5 | 4.29e-2 | 16.143       | 3.19e-5            |
 * | case1    | 26 | 1.48e-5 | 4.83e-2 | 20.244       | 1.77e-5            |
 * | case2    | 35 | 1.62e-5 | 4.35e-1 | 21.244       | 4.69e-6            |
 * | padded   | 16 | 2.17e-5 | 1.63e-2 | 28.913       | 3.21e-5            |
 *
 * atol 1.5e-4 は実測最悪 2.26e-5（case0）の約 6.6 倍。rtol が 0 な理由は他 2 variant と同じ
 * （\|ref\| 最小非ゼロ 4.69e-6 で maxRel 4.35e-1 はその要素の見かけ・絶対誤差は 2e-6 級）。
 *
 * full-24layer（i8）より 1 桁近く小さいのは、この variant の出力が**そちらの output.22 に
 * 相当する層**で止まっており、誤差が最大になる 18〜19 層目を出力に持たないから。実測でも
 * 4 ケースとも full-24layer(i8) の output.22 の maxAbs と一致した（case0 2.26e-5 / case1
 * 1.48e-5 / case2 1.62e-5 / padded 2.17e-5）— ADR 0045 が torch で採った「22 層モデルの最終
 * 出力は 24 層モデルの hidden_states[-3] とビット一致」がランタイム側でも保たれている傍証。
 */
const DEBERTA_I8_22LAYER_TOLERANCE: Tolerance = { atol: 1.5e-4, rtol: 0 };

const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";
/** w8a8 鏡像 io の prefix（`deberta/export.py` の `ACT_IO_PREFIX`）— この門は読まない。 */
const ACT_IO_PREFIX = "io-i8a8.";

/** SKIP 時にそのまま貼れる生成コマンドの共通部（tools/export-recipes/deberta/README.md）。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m deberta.export";

/** 格納 dtype 系列 × 層数 variant 1 本の宣言。 */
type DebertaVariant = {
  /** テスト名に出る綴り（`<系列>/<variant ディレクトリ>`）。 */
  readonly name: string;
  readonly root: URL;
  /** **variant ごとに実測導出**（上の MUST）。 */
  readonly tolerance: Tolerance;
  /**
   * `graph.outputs` の本数（全層出し 25 / 配布形 1）。
   *
   * MUST: この検査を落とさない。配布形（1 本出し）と検証用（全層出し）は**同じ shape の
   * テンソルを出す**ので、variant を取り違えた資産はロードも実行も突合も通ってしまう
   * （`deberta/export.py` の `Variant` docstring が名指しする沈黙誤り）。
   */
  readonly outputs: number;
  /**
   * この variant の資産が宣言しているべき圧縮格納 dtype（無ければ `undefined` = 全て f32）。
   *
   * MUST: この検査を落とさない。**系列 root の取り違えは数値では検出できない** — 実測最悪は
   * f32 8.32e-5 / i8 1.23e-4 と同桁で、互いの tolerance を素通りする。同じ理由で
   * 「`--dtype i8` のつもりが f32 で書けていた」も数値では見えない（ADR 0027 が SBV2 で
   * 故障注入まで含めて確定した検出限界と同じ形）。格納宣言だけが区別する。
   */
  readonly compressedStorage?: "i8";
  /** SKIP 時にそのまま貼れる生成コマンド。 */
  readonly generate: string;
};

/**
 * 実走する variant。**列挙ではなくここで固定する** — 生成済みのものを拾う形にすると、
 * variant ごと生成し忘れた環境で「緑だが未検証」になる。
 */
const VARIANTS: readonly DebertaVariant[] = [
  {
    name: "f32/full-24layer",
    root: new URL("../../../outputs/series/deberta/full-24layer/", import.meta.url),
    tolerance: DEBERTA_TOLERANCE,
    outputs: 25,
    generate: `${GENERATE_COMMAND} --layers 24`,
  },
  {
    name: "i8/full-24layer",
    root: new URL("../../../outputs/series/deberta-i8/full-24layer/", import.meta.url),
    tolerance: DEBERTA_I8_TOLERANCE,
    outputs: 25,
    compressedStorage: "i8",
    generate: `${GENERATE_COMMAND} --dtype i8 --layers 24`,
  },
  {
    name: "i8/sbv2-22layer",
    root: new URL("../../../outputs/series/deberta-i8/sbv2-22layer/", import.meta.url),
    tolerance: DEBERTA_I8_22LAYER_TOLERANCE,
    outputs: 1,
    compressedStorage: "i8",
    generate: `${GENERATE_COMMAND} --dtype i8 --layers 22`,
  },
];

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する** — 列挙だけに頼ると生成を
 * 一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/export-recipes/deberta/export.py` の `GOLDEN_SENTENCES` + `padded`（T = 11 / 26 /
 * 35 / 16）。`padded` だけが `attention_mask=0` を混ぜ、マスク経路（mul → cast →
 * bitwise_not → masked_fill）を踏む。
 */
const EXPECTED_CASES = ["case0", "case1", "case2", "padded"] as const;

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

/** 通常ケースの golden ファイル名（`io.<case>.safetensors`）。鏡像は prefix で外れる。 */
const goldenFiles = (root: URL): readonly string[] =>
  listDir(root)
    .filter((entry) =>
      entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
    )
    .map((entry) => entry.name)
    .sort();

const caseNameOf = (file: string): string =>
  file.slice(IO_PREFIX.length, file.length - IO_SUFFIX.length);

const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
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

Deno.test({
  name: "DeBERTa golden: w8a8 鏡像 io が通常ケースの列挙に紛れない",
  // 資産の有無に依らず走る（prefix の規約そのものを見るテスト）。
  fn: () => {
    // 正本は `deberta/export.py` の `ACT_IO_PREFIX` の MUST。`io.` の startsWith で通常ケースを
    // 拾う以上、鏡像の prefix が `io.` で始まった時点で w8 の期待値が汚染される。
    assert(
      !ACT_IO_PREFIX.startsWith(IO_PREFIX),
      `鏡像 prefix '${ACT_IO_PREFIX}' が通常ケースの prefix '${IO_PREFIX}' で始まっている`,
    );
    for (const variant of VARIANTS) {
      for (const file of goldenFiles(variant.root)) {
        assert(
          !file.startsWith(ACT_IO_PREFIX),
          `${variant.name}: 鏡像 golden '${file}' が通常ケースとして拾われている`,
        );
      }
    }
  },
});

for (const variant of VARIANTS) {
  /** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
  const files = goldenFiles(variant.root);
  /** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
  const available = files.length > 0;
  /**
   * **何か 1 つでも**残っているか（完全性テストの SKIP 述語 — Codex 波 H 指摘 H-02）。
   * golden が全滅してモデルだけ残った欠損は `available` では偽になり、`ignore: !available`
   * だと完全性テスト自身が SKIP される — 欠損を FAIL にする述語は「完全に空」でだけ寝てよい。
   */
  const anyPresent = available || fileExists(new URL(MODEL_FILE, variant.root));

  if (!available) {
    console.warn(
      `[karume] ${variant.root.pathname} に export 済み資産が無いため実重み DeBERTa E2E ` +
        `（${variant.name}）を SKIP する（重み 307MB〜1.3GB につきリポジトリ管理外）。` +
        `生成: ${variant.generate}`,
    );
  }

  Deno.test({
    name: `DeBERTa 資産（${variant.name}）: 期待するケースとモデル本体が揃っている`,
    // 完全に空の環境だけ「生成していない」として SKIP。**何か 1 つでも**あれば欠けは FAIL
    //（モデルだけ残って golden が全滅した欠損も拾う — `anyPresent` の JSDoc）。
    ignore: !anyPresent,
    fn: () => {
      assertEquals(
        files.map(caseNameOf),
        [...EXPECTED_CASES],
        `${variant.root.pathname} の golden ケース`,
      );
      assert(fileExists(new URL(MODEL_FILE, variant.root)), `${MODEL_FILE} が無い`);
    },
  });

  for (const file of files) {
    const caseName = caseNameOf(file);
    Deno.test({
      name: `DeBERTa golden 突合: ${variant.name} / ${caseName}（実 GPU / torch CPU 期待値）`,
      ignore: !available || !GPU_AVAILABLE,
      fn: async () => {
        const [modelBytes, ioBytes] = await Promise.all([
          readBuffer(variant.root, MODEL_FILE),
          readBuffer(variant.root, file),
        ]);
        const parsed = openModel(modelBytes);
        const io = parseSafetensors(ioBytes);

        // 出力の本数で variant を見分ける（配布形 1 本出しと検証用 25 本出しの取り違えは
        // shape が同じで数値にも出ない — 上の `outputs` の MUST）。
        assertEquals(
          parsed.graph.outputs.length,
          variant.outputs,
          `${variant.name}: graph.outputs の本数`,
        );

        // 系列と資産の格納 dtype が一致する（root 取り違え / 圧縮の掛け忘れの唯一の検出器
        // — 上の `compressedStorage` の MUST）。「現れた圧縮 dtype の集合」が宣言とちょうど
        // 一致することを見る（本数ではなく集合で見るのは、i8 系列に f16 資産が混ざる形を
        // 「圧縮が 1 本以上ある」で通さないため）。
        // NOTE: `i32` 格納（記号依存定数 — ADR 0010）は圧縮ではないのでここでは数えない。
        const compressed = [
          ...new Set(
            Object.values(parsed.graph.initializers)
              .map((initializer) => initializer.storage.dtype)
              .filter((dtype) => dtype === "f16" || dtype === "i8"),
          ),
        ].sort();
        assertEquals(
          compressed,
          variant.compressedStorage === undefined ? [] : [variant.compressedStorage],
          `${variant.name}: 圧縮格納 dtype の集合が宣言と食い違う`,
        );
        // i8 は companion scale が無いと値が復元できない（ADR 0019）。宣言と実体の両方を見る
        // — 宣言だけならキーが実在しない形が、実体だけなら別の重みの scale を読む形が通る。
        if (variant.compressedStorage === "i8") {
          for (const [name, initializer] of Object.entries(parsed.graph.initializers)) {
            if (initializer.storage.dtype !== "i8") continue;
            const scale = initializer.storage.scale;
            assert(scale !== undefined, `${variant.name}: '${name}' に scale 宣言が無い`);
            assert(
              parsed.file.tensors.has(scale),
              `${variant.name}: '${name}' の scale '${scale}' が資産に無い`,
            );
          }
        }

        // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
        const expectedKeys = [
          ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
          ...parsed.graph.outputs.map((_, index) => `output.${index}`),
        ].sort();
        assertEquals([...io.tensors.keys()].sort(), expectedKeys, `${file} のテンソルキー`);

        // 記号次元 T は golden の入力 shape の実長から束縛される（明示 bindings を渡さない）。
        // ケースごとに T が違う（11 / 26 / 35 / 16）ので、宣言上限 Tmax = 512 に依存した実装
        // （プランを Tmax で組む・相対位置表を Tmax で焼く等）はここで値か shape が壊れる。
        const inputs: Record<string, Tensor> = {};
        for (const spec of parsed.graph.inputs) {
          const view = io.tensors.get(`input.${spec.name}`);
          assert(view !== undefined, `input.${spec.name} が ${file} に無い`);
          inputs[spec.name] = ioTensor(io, view, spec.dtype);
        }

        const gpu = await acquireGpu();
        const session = await createSession(gpu, parsed);
        try {
          const outputs = await session.run(inputs);
          assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

          parsed.graph.outputs.forEach((name, index) => {
            const view = io.tensors.get(`output.${index}`);
            assert(view !== undefined, `output.${index} が ${file} に無い`);
            const where = `${variant.name}/${caseName} output.${index} ('${name}')`;
            const declared = parsed.graph.values[name].dtype;
            assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
            assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
            const report = compareTensors(
              outputs[name],
              ioTensor(io, view, declared),
              variant.tolerance,
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
