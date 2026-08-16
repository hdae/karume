// BiRefNet 系の**実画像**の門 —— TS 前処理層（`src/image/preprocess.ts`）と PNG 符号化
// （`src/image/png.ts`）を通す 2 つの主張を持つ。
//
// ## この門が受け持つ主張と、runtime 側との分担
//
// もともとは `packages/runtime/tests/e2e_birefnet_test.ts` に同居していた（TS 前処理を通した
// `pixel_values` で推論し、入力側と出力側を 1 本のテストで見ていた）。それを 2 つに割った
// —— SigLIP2 で先に踏んだのと同じ形（`image_preprocess_real_test.ts` 冒頭）:
//
//  ・**入力側（ここ）**: 「同じ PNG から Python と同じ `pixel_values` が出る」—— GPU も
//    モデル本体（1 系列 964MB）も要らない、前処理層だけの純粋なパリティ。
//  ・**出力側（runtime）**: golden に焼かれた `pixel_values` を入力にした実 GPU 忠実度
//    （`e2e_birefnet_test.ts` の実画像 golden 突合）。
//
// 「PNG を渡したら Python と同じマットが返る」という鎖の主張は、この 2 つの**合成**で持つ
// （ここが入力差 ≤ {@link PIXEL_ATOL} を保証し、runtime 側がその入力での忠実度を保証する）。
// 割った理由は依存方向 —— runtime のテストが models の実装を相対 import していた（逆向き）。
//
// ## 意味の判別と成果物 PNG はここに残る（golden を経由しない門だから）
//
// 顕著物体のある 2 枚の前景比が無い 2 枚を上回る、という**判別**は golden のテンソルを 1 本も
// 読まない（読むのはモデル本体と PNG だけ）ので、割りようが無い。TS 前処理を通した入力のまま
// **逐語で**こちらへ移した —— 「PNG を渡したら意味のあるマットが返る」という鎖の主張は、
// 合成に頼らずここが直接持つ。マット PNG / 白地合成 PNG を書き出すのも同じ理由でここ
// （`encodePng` は models の実装）。
//
// ## 資産が無い環境
//
// golden（`outputs/series/`）も PNG（`outputs/demo/`）もリポジトリ管理外で、`rm -rf` で消せる
// 席。**1 件も無ければ明示 SKIP**、**golden が中途半端に欠けていれば FAIL**（欠けの FAIL は
// runtime 側の「資産の完全性」テストが名指しで出す —— 系列ディレクトリの列挙はあちらが持つ）。
// ADR 0005 の「全 SKIP は明示 FAIL」門番は *GPU アダプタの有無* だけを見ており、この SKIP とは
// 独立。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  createSession,
  type KarumeModel,
  openModel,
  parseSafetensors,
} from "@karume/runtime";
import { normalizeToNchw, resizeRgb8 } from "../src/image/preprocess.ts";
import { encodePng } from "../src/image/png.ts";
// NOTE: PNG デコーダは karume 本体が持たない設計（preprocess.ts のモジュール doc —— 入口は
// RGB8 の画素列）。テストが実画像を読むためだけの道具を runtime のテスト配下から借りる。
// models → runtime は**正方向**の依存で、かつ両者とも `tests/` は publish から除外される
// （packages/*/deno.json の `publish.exclude`）ので、配布物には影響しない。
import { decodePng } from "../../runtime/tests/helpers/png-decode.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * 実画像ケースの**入力側**（TS 前処理 対 Python 前処理）の許容誤差。
 *
 * 実画像も系列も 1024² なので **resize は恒等**（`resizeRgb8` の台は scale 1 で重み 1 点に
 * 縮む）。残るのは正規化の畳み方の差だけ: TS は `(u8 − mean·255) / (std·255)` を 1 回で、
 * torch 側は同梱 `handler.py` の `ToTensor`（`u8 / 255`）→ `Normalize`（`(x − mean) / std`）で
 * **丸めが 1 回多い**。実測はこの差だけで、**2 系列 × 4 ケースとも maxAbs 4.768e-7**
 * （\|ref\| 上端 2.64 に対し 2 ulp）。
 *
 * atol 1e-6 は実測 4.768e-7 の約 2.1 倍。
 *
 * **これだけは系列で共有する**（runtime 側の出力 tolerance は共有しない）。ここが比べているのは
 * `pixel_values` を作る鎖だけで、両辺ともモデルの重みに 1 度も触っていない —— 同じ画像・同じ
 * 前処理定数・同じ寸法なので、系列ごとに測っても**同じ計算の同じ結果**にしかならない。
 *
 * この突合を出力側と分けて持つのは、落ちたときに前処理と推論のどちらが動いたのかを分ける
 * ため（出力側だけだと、統計の取り違えと実装バグが同じ「大きい maxAbs」に見える）。
 * 統計の取り違え（SigLIP2 の mean = std = 0.5 を当てる / RGB の順序違い）はここで
 * O(0.1)〜O(1) の差になり、4 桁以上上に出る。
 *
 * NOTE: 判定は `|got − expected| ≤ atol`（rtol は 0 だった）に**非有限の不合格**を足したもの
 * —— runtime の `compareTensors` が rtol 0 で行う判定と同じ式で、NaN / ±Inf が「差 0」に化けて
 * 通る穴も同じように塞いである。
 */
const PIXEL_ATOL = 1e-6;

/** 実走する 1 系列（モデル × 解像度）。正本は runtime 側 e2e の `SERIES`。 */
type Series = {
  /** `outputs/series/` 直下のディレクトリ名（`birefnet.export.default_out_dir` の綴り）。 */
  readonly name: string;
  /** SKIP 時にそのまま貼れる生成コマンド。 */
  readonly generate: string;
};

const EXPORT_PREFIX =
  "cd tools/export-recipes && uv run --group birefnet python -m birefnet.export";

const SERIES: readonly Series[] = [
  { name: "birefnet-hr-1024", generate: `${EXPORT_PREFIX} --real-images` },
  {
    name: "lucida-1024",
    generate: `${EXPORT_PREFIX} --model-dir <リポ>/inputs/birefnet/lucida --real-images`,
  },
];

const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);
const DEMO_DIR = new URL("../../../outputs/demo/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

const seriesRoot = (series: Series): URL => new URL(`${series.name}/`, SERIES_PARENT);

/**
 * マット PNG（目視確認用の成果物）の置き場。`outputs/` 配下なので git 追跡外。**系列ごとに
 * 分ける** — 同じ席へ書くと、後に走った系列のマットが先の系列のものを黙って置き換える。
 */
const artifactDir = (series: Series): URL => new URL(`birefnet/${series.name}/`, DEMO_DIR);

/** 実画像そのものを焼き直すコマンド（プロンプト / seed の正本は台本側）。 */
const IMAGE_COMMAND = "deno task demo:eval-images --source <Anima 配布形のパス>";

/**
 * **実画像**ケース（`--real-images` を付けた emit だけが持つ）。ケース名とファイル名の正本は
 * `birefnet/export.py` の `REAL_CASES`、画像そのもの（プロンプト / seed / 解像度）の正本は
 * `examples/anima/eval-images.ts`。**列挙結果ではなくここで固定する** — 列挙に頼ると、一部
 * だけ生成した環境でケースが黙って消えて「緑だが未検証」になる。
 */
const REAL_CASES = [
  { name: "photo-portrait", file: "anima-default-1024x1024-defaultstep-seed42.png" },
  { name: "photo-landscape", file: "anima-default-1024x1024-defaultstep-seed43.png" },
  { name: "photo-corridor", file: "anima-default-1024x1024-defaultstep-seed44.png" },
  { name: "photo-street", file: "anima-default-1024x1024-defaultstep-seed45.png" },
] as const;

/**
 * 実画像の判別で見る 2 群（顕著物体 = 人物が写っている 2 枚 / 写っていない 2 枚）。正本は
 * `birefnet/export.py` の `REAL_PERSON_CASES` / `REAL_SCENE_CASES` で、あちらは torch 出力に、
 * こちらは**実 GPU 出力**に掛ける。
 */
const REAL_PERSON_CASES = ["photo-portrait", "photo-street"] as const;
const REAL_SCENE_CASES = ["photo-landscape", "photo-corridor"] as const;

/** golden の入力テンソルのキー（正本は `birefnet/export.py` の `INPUT_NAME`）。 */
const PIXEL_INPUT_KEY = "input.pixel_values";

/** 実画像 golden の `__metadata__` の欄（正本は `birefnet/export.py` の同名定数）。 */
const SOURCE_IMAGE_KEY = "source_image";
const SOURCE_SHA256_KEY = "source_sha256";

/**
 * 前処理の正規化定数（ImageNet 統計）。正本は同梱 `handler.py` の `ImagePreprocessor` で、
 * 写しは `birefnet/export.py` の `IMAGENET_MEAN` / `IMAGENET_STD`。**SigLIP2（mean = std =
 * 0.5）とは別の統計**なので共有しない — 取り違えは {@link PIXEL_ATOL} が落とす。
 * 2 系列で同じ統計なのは、Lucida が BiRefNet_HR の fine-tune で前処理を変えていないため
 * （上流モデルカードの利用例が同じ 3 段を綴っている）。
 */
const IMAGE_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
const IMAGE_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

/**
 * ファイルの存在。
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、
 * 実行されていない検証が SKIP として静かに緑になる。
 */
const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const goldenUrl = (series: Series, caseName: string): URL =>
  new URL(`${IO_PREFIX}${caseName}${IO_SUFFIX}`, seriesRoot(series));

/** 登録時点で必要なので同期で数える（`Deno.test` の ignore 判定に使う）。 */
const goldenCount = (series: Series): number =>
  REAL_CASES.filter((entry) => exists(goldenUrl(series, entry.name))).length;

/**
 * 実画像が 4 枚とも揃っているか（`outputs/demo/` は `rm -rf` で消せる席）。
 * MUST: NotFound 以外は伝播させる（`exists` と同じ理由）。
 */
const IMAGES_PRESENT: boolean = REAL_CASES.every((entry) => exists(new URL(entry.file, DEMO_DIR)));

const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** 実画像 1 枚のバイト列（decode 前 — sha256 の突合にも使うので生のまま持つ）。 */
const readImage = (file: string): Promise<Uint8Array<ArrayBuffer>> =>
  Deno.readFile(new URL(file, DEMO_DIR));

/** グラフ入力の静的次元（記号次元は無い — `birefnet/export.py` の `symbol_names=()`）。 */
const staticDim = (parsed: KarumeModel, axis: number): number => {
  const dim = parsed.graph.inputs[0].shape[axis];
  assert(typeof dim === "number", `pixel_values の軸 ${axis} が記号次元 '${String(dim)}'`);
  return dim;
};

/** 前景（`logit > 0`）の面積比。判別（顕著物体の有無）はこの量の順序だけを見る。 */
const foregroundRatio = (logits: Float32Array): number => {
  let foreground = 0;
  for (let index = 0; index < logits.length; index += 1) {
    if (logits[index] > 0) foreground += 1;
  }
  return foreground / logits.length;
};

/** マットの α（`sigmoid` はホスト側の責務 — グラフは logit を出す）。 */
const alphaFromLogits = (logits: Float32Array): Float32Array => {
  const alpha = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    alpha[index] = 1 / (1 + Math.exp(-logits[index]));
  }
  return alpha;
};

/** グレースケール（α をそのまま明度に）の PNG バイト列。 */
const matteToPng = (
  alpha: Float32Array,
  width: number,
  height: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  const rgba = new Uint8Array(width * height * 4);
  for (let index = 0; index < alpha.length; index += 1) {
    const value = Math.round(alpha[index] * 255);
    rgba[index * 4] = value;
    rgba[index * 4 + 1] = value;
    rgba[index * 4 + 2] = value;
    rgba[index * 4 + 3] = 255;
  }
  return encodePng(rgba, width, height);
};

/** 前景を白地へ合成した PNG バイト列（背景抜きの見た目そのもの）。 */
const cutoutToPng = (
  image: { readonly data: Uint8Array; readonly width: number; readonly height: number },
  alpha: Float32Array,
): Promise<Uint8Array<ArrayBuffer>> => {
  const rgba = new Uint8Array(image.width * image.height * 4);
  for (let index = 0; index < alpha.length; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const source = image.data[index * 3 + channel];
      rgba[index * 4 + channel] = Math.round(source * alpha[index] + 255 * (1 - alpha[index]));
    }
    rgba[index * 4 + 3] = 255;
  }
  return encodePng(rgba, image.width, image.height);
};

for (const series of SERIES) {
  const goldens = goldenCount(series);
  /** 実画像の群。golden と画像の**両方**が揃ってはじめて実走する。 */
  const realAvailable = goldens > 0 && IMAGES_PRESENT;

  if (!realAvailable) {
    console.warn(
      `[karume] ${series.name} の実画像ケースを SKIP する（golden ${goldens}/` +
        `${REAL_CASES.length} 本・画像 ${IMAGES_PRESENT ? "有" : "無"}）。` +
        `画像の生成: ${IMAGE_COMMAND} / golden の生成: ${series.generate}`,
    );
  }

  for (const entry of REAL_CASES) {
    Deno.test({
      name: `BiRefNet 実画像前処理パリティ: ${series.name} / ${entry.name}` +
        "（TS の畳んだ正規化 対 handler.py の 2 段）",
      ignore: !realAvailable,
      fn: async () => {
        const [goldenBytes, png] = await Promise.all([
          readBuffer(goldenUrl(series, entry.name)),
          readImage(entry.file),
        ]);
        const io = parseSafetensors(goldenBytes);

        // ① golden を採った画像と、いま読んでいる画像が同一であること。**tolerance では
        // 吸収されない差**（生成台本を回し直して golden を採り直していない）を、突合の前に
        // 名指しで落とす。ここが緩むと「別の画像どうしを比べて緑」が成立しうる。
        assertEquals(io.metadata.get(SOURCE_IMAGE_KEY), entry.file, "golden が指す元画像");
        assertEquals(
          io.metadata.get(SOURCE_SHA256_KEY),
          await sha256Hex(png),
          `${entry.file} が golden を採った画像と違う（採り直す: ${series.generate}）`,
        );

        const view = io.tensors.get(PIXEL_INPUT_KEY);
        assert(view !== undefined, `${PIXEL_INPUT_KEY} が golden に無い`);
        assertEquals(view.dtype, "F32", `${PIXEL_INPUT_KEY} の格納 dtype`);
        assertEquals(view.shape.length, 4, `${PIXEL_INPUT_KEY} の階数`);
        const [batch, channels, height, width] = view.shape;
        assertEquals([batch, channels], [1, 3], `${PIXEL_INPUT_KEY} の先頭 2 軸`);
        const expected = new Float32Array(io.buffer, view.byteOffset, view.byteLength / 4);

        // ② TS 側の経路（PNG decode → resize → 融合正規化）。
        const decoded = await decodePng(png, entry.file);
        const got = normalizeToNchw(resizeRgb8(decoded, width, height), IMAGE_MEAN, IMAGE_STD);

        assertEquals(got.length, expected.length, `${entry.name}: pixel_values の長さ`);
        let maxAbs = 0;
        let worst = 0;
        let nonFinite = 0;
        for (let index = 0; index < got.length; index += 1) {
          if (!Number.isFinite(got[index]) || !Number.isFinite(expected[index])) {
            nonFinite += 1;
            continue;
          }
          const delta = Math.abs(got[index] - expected[index]);
          if (delta > maxAbs) {
            maxAbs = delta;
            worst = index;
          }
        }
        // MUST: NaN / ±Inf は不合格。素朴な差分判定だと NaN が「差 0」に化けて通る。
        assertEquals(nonFinite, 0, `${entry.name}: pixel_values に非有限の標本`);
        assert(
          maxAbs <= PIXEL_ATOL,
          `${series.name} / ${entry.name}: pixel_values の maxAbs=${maxAbs}` +
            `（標本 ${worst}: ${got[worst]} 対 ${expected[worst]}）が ${PIXEL_ATOL} を超えた`,
        );
      },
    });
  }

  Deno.test({
    name: `BiRefNet 実画像 判別: ${series.name} — 顕著物体のある 2 枚の前景比が無い 2 枚を` +
      "上回る（マット PNG も書く）",
    ignore: !realAvailable || !GPU_AVAILABLE,
    fn: async () => {
      // 判別は**前景比の順序**だけを見る（閾値は置かない — `birefnet/export.py` の
      // `_real_sanity` と同じ形で、あちらは torch 側に掛かっている）。一様に潰れた出力（全前景
      // / 全背景）も入力非依存の出力も、両群の前景比が並ぶので落ちる。実測は人物側が
      // BiRefNet_HR で 0.5512 / 0.1139・Lucida で 0.5545 / 0.1134、風景側はどちらも 0.005 未満。
      //
      // 入力は**実画像を TS 前処理で通したもの**（golden の入力ではない）— 「PNG を渡したら
      // 意味のあるマットが返る」ところまでを検査にする。ついでに α マットと白地合成を PNG で
      // 書き出す（数値の門だけでは形が見えないため — 目視確認用の成果物であって、門ではない）。
      const modelBytes = await readBuffer(new URL(MODEL_FILE, seriesRoot(series)));
      const parsed = openModel(modelBytes);
      const [outputName] = parsed.graph.outputs;
      const width = staticDim(parsed, 3);
      const height = staticDim(parsed, 2);
      const inputName = parsed.graph.inputs[0].name;
      const artifacts = artifactDir(series);
      await Deno.mkdir(artifacts, { recursive: true });

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      const ratios = new Map<string, number>();
      try {
        // 4 枚を 1 Session で回す（重みは 964MB — 画像ごとに組み直す理由が無い）。
        for (const real of REAL_CASES) {
          const image = resizeRgb8(
            await decodePng(await readImage(real.file), real.file),
            width,
            height,
          );
          const pixels = normalizeToNchw(image, IMAGE_MEAN, IMAGE_STD);
          const output = (await session.run({
            [inputName]: { dtype: "f32", shape: [1, 3, height, width], data: pixels },
          }))[outputName];
          // 判別子で絞る（Float32Array へのキャストは dtype がずれたときに黙って通る）。
          assert(output.dtype === "f32", `${real.name}: マットの dtype が ${output.dtype}`);
          ratios.set(real.name, foregroundRatio(output.data));

          const alpha = alphaFromLogits(output.data);
          await Deno.writeFile(
            new URL(`${real.name}-matte.png`, artifacts),
            await matteToPng(alpha, width, height),
          );
          await Deno.writeFile(
            new URL(`${real.name}-cutout.png`, artifacts),
            await cutoutToPng(image, alpha),
          );
        }
      } finally {
        await session.dispose();
        gpu.destroy();
      }

      const ratioOf = (name: string): number => {
        const ratio = ratios.get(name);
        assert(ratio !== undefined, `${name} の前景比が無い`);
        return ratio;
      };
      for (const personCase of REAL_PERSON_CASES) {
        for (const sceneCase of REAL_SCENE_CASES) {
          assert(
            ratioOf(personCase) > ratioOf(sceneCase),
            `前景比の順序が逆: ${personCase}=${ratioOf(personCase)} <=` +
              ` ${sceneCase}=${ratioOf(sceneCase)}`,
          );
        }
      }
      console.log(
        `[karume] BiRefNet のマット PNG を ${artifacts.pathname} へ書いた` +
          `（<ケース>-matte.png / <ケース>-cutout.png）`,
      );
    },
  });
}
