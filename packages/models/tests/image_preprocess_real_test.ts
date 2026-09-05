// 画像前処理層（src/image/preprocess.ts）の Python 正本とのパリティ検証 —— **実画像**編。
//
// 合成フィクスチャ側（image_preprocess_test.ts）が「資産の要らない 22×33 級の合成画像で
// カーネルの署名を縛る」のに対し、こちらは **1024×1024 の実画像を 224 / 384 へ縮小した
// 実寸の一点**を縛る。縮尺（4.6 倍 / 2.7 倍）も画の性質（生成画像の自然な階調）もフィクスチャ
// とは違うので、あちらの相違率・許容差はそのままでは通用しない（実測は下の導出表）。
//
// 突合の相手は **SigLIP2 の golden に焼かれた Python 側 `pixel_values`**
// （`outputs/series/<系列>/io.photo-*.safetensors` の `input.pixel_values`）。生成は
// `tools/export-recipes/siglip2/export.py --real-images` で、正本は transformers の
// `SiglipImageProcessor`（`TorchvisionBackend`）。画像そのものの正本は
// `examples/anima/eval-images.ts`（`deno task demo:eval-images` で同じ 4 枚が焼き直せる）。
//
// ## この門が受け持つ主張と、runtime 側との分担
//
// この検証はもともと `packages/runtime/tests/e2e_siglip2_test.ts` の実画像ケースに同居して
// いた（TS 前処理を通した `pixel_values` で推論し、入力側と出力側を 1 本のテストで見ていた）。
// それを 2 つに割った:
//
//  ・**入力側（ここ）**: 「同じ PNG から Python と同じ `pixel_values` が出る」— GPU も
//    モデル本体（350MB〜1.7GB）も要らない、前処理層だけの純粋なパリティ。
//  ・**出力側（runtime）**: golden に焼かれた `pixel_values` を入力にした実 GPU 忠実度。
//
// 「PNG を渡したら Python と同じ埋め込みが返る」という鎖の主張は、この 2 つの**合成**で持つ
// （ここが入力差 ≤ {@link PIXEL_ATOL} を保証し、runtime 側がその入力での忠実度を保証する）。
// 割った理由は依存方向 —— runtime のテストが models の実装を相対 import していた（逆向き）。
//
// ## tolerance の導出（実測 2026-08-14・素の突合 `atol=rtol=0`）
//
// uint8 の 1 LSB は正規化後 `1 / 127.5 = 7.8431e-3` で、f32 へ写す丸めが乗る。8 ケース
// （2 系列 × 実画像 4 枚）の実測:
//
// | 系列   | ケース          | maxAbs     | 相違標本 / 全標本  | 相違率 |
// | ------ | --------------- | ---------- | ------------------ | ------ |
// | base   | photo-portrait  | 7.84317e-3 | 204 / 150,528      | 0.136% |
// | base   | photo-landscape | 7.84314e-3 | 220 / 150,528      | 0.146% |
// | base   | photo-corridor  | 7.84315e-3 | 198 / 150,528      | 0.132% |
// | base   | photo-street    | 7.84315e-3 | 139 / 150,528      | 0.092% |
// | so400m | photo-portrait  | 7.84317e-3 | 3,817 / 442,368    | 0.863% |
// | so400m | photo-landscape | 7.84317e-3 | 3,681 / 442,368    | 0.832% |
// | so400m | photo-corridor  | 7.84317e-3 | 4,000 / 442,368    | 0.904% |
// | so400m | photo-street    | 7.84317e-3 | 3,502 / 442,368    | 0.792% |
//
// maxAbs は 8 ケースとも 1 LSB ちょうど（TS 側は f64 で積算し、torchvision は f32 で積算する
// ので、丸め境界に載った標本だけが 1 ずれる —— 導出の詳細は image_preprocess_test.ts 冒頭）。
// **相違率が 384 側で 6 倍多い**のは縮尺が緩い（2.7 倍縮小）ぶん丸め境界に載る標本が増える
// ためで、フィクスチャ側の実測 0.23% とも桁が違う。値を共有せずここで実測している理由。
//
// ## 資産が無い環境
//
// golden（`outputs/series/`）も PNG（`outputs/misc/corpus/`）もリポジトリ管理外。golden は
// export の再実行で作り直せるが、PNG はホスト資産（消すと台本での焼き直しと凍結コピーが要る）。
// **1 件も無ければ明示 SKIP**、**中途半端に欠けていれば FAIL**（欠けを SKIP に丸めると、
// 採り直しの途中で落ちた資産が黙って通る）。runtime 側 e2e と同じ流儀。

import { assert, assertEquals } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { normalizeToNchw, resizeRgb8 } from "../src/image/preprocess.ts";
// NOTE: PNG デコーダは karume 本体が持たない設計（preprocess.ts のモジュール doc — 入口は
// RGB8 の画素列）。テストが実画像を読むためだけの道具を runtime のテスト配下から借りる。
// models → runtime は**正方向**の依存で、かつ両者とも `tests/` は publish から除外される
// （packages/*/deno.json の `publish.exclude`）ので、配布物には影響しない。
import { decodePng } from "../../runtime/tests/helpers/png-decode.ts";

/**
 * 全経路（PNG → resize → normalize）の許容差。uint8 の 1 LSB が正規化後 `1 / 127.5` に
 * 写った幅で、実測は 8 ケースとも 7.84314e-3〜7.84317e-3（= 常にちょうど 1 LSB）。
 * その上に僅かな余裕を取った値 —— 導出表はファイル冒頭。
 *
 * `image_preprocess_test.ts` の `PIXEL_ATOL` と同じ値になったが**共有はしない**（あちらは
 * 22×33 級の合成画像・こちらは 1024×1024 の実画像で、縮尺も画の性質も違う。片方を測り直した
 * ときにもう片方が黙って動く形にはしない）。
 *
 * 2 LSB 以上ずれる実装差（丸めの取り違え・補間カーネルの取り違え・チャネル順の反転・
 * 解像度の取り違え）はこの閾値を越える。**全画素が 1 ずつずれる**形の誤りはこの門では
 * 捕まらないが、そこは `image_preprocess_test.ts` の相違率上限（`MISMATCH_RATE_CAP`）が
 * 受け持つ。
 */
const PIXEL_ATOL = 7.85e-3;

/** 前処理の正規化定数。2 モデルとも mean = std = 0.5（正本は `preprocessor_config.json`）。 */
const IMAGE_MEAN: readonly [number, number, number] = [0.5, 0.5, 0.5];
const IMAGE_STD: readonly [number, number, number] = [0.5, 0.5, 0.5];

/**
 * golden を持つ系列（= モデル）。解像度は**書かない** —— golden の `pixel_values` の shape
 * から取る（ここに 224 / 384 を書くと、もう片方で黙って別の画を作って突合が形骸化する）。
 */
const SERIES = ["siglip2-base-patch16-224", "siglip2-so400m-patch14-384"] as const;

/**
 * 実画像ケース。ケース名とファイル名の対応の正本は
 * `tools/export-recipes/siglip2/export.py` の `REAL_CASES`、
 * 画像そのもの（プロンプト / seed / 解像度）の正本は `examples/anima/eval-images.ts`。
 * **列挙結果ではなくここで固定する** —— 列挙に頼ると、一部だけ生成した環境でケースが黙って
 * 消えて「緑だが未検証」になる。
 */
const REAL_CASES = [
  { name: "photo-portrait", file: "anima-default-1024x1024-defaultstep-seed42.png" },
  { name: "photo-landscape", file: "anima-default-1024x1024-defaultstep-seed43.png" },
  { name: "photo-corridor", file: "anima-default-1024x1024-defaultstep-seed44.png" },
  { name: "photo-street", file: "anima-default-1024x1024-defaultstep-seed45.png" },
] as const;

/** golden の入力テンソルのキー（正本は `tools/export-recipes/siglip2/export.py` の `INPUT_NAME`）。 */
const PIXEL_INPUT_KEY = "input.pixel_values";

/** 実画像 golden の `__metadata__` の欄（正本は `tools/export-recipes/siglip2/export.py` の同名定数）。 */
const SOURCE_IMAGE_KEY = "source_image";
const SOURCE_SHA256_KEY = "source_sha256";

const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);
/** 入力の実画像コーパス（凍結コピー — ホスト資産なので消すと焼き直しが要る）。 */
const CORPUS_DIR = new URL("../../../outputs/misc/corpus/", import.meta.url);

const goldenUrl = (series: string, caseName: string): URL =>
  new URL(`${series}/io.${caseName}.safetensors`, SERIES_PARENT);

/** golden を採り直すコマンド（`--real-images` はグループが違う）。 */
const generateCommand = (series: string): string =>
  "cd tools/export-recipes && uv run --group siglip2-preprocess python -m siglip2.export" +
  ` --real-images --model-dir ../../inputs/siglip2/${series}`;

/**
 * 実画像そのものを焼き直すコマンド（プロンプト / seed の正本は台本側）。台本は
 * `outputs/bench/<model>/<日付>_eval-images/` へ焼くので、採用分は {@link CORPUS_DIR} へ
 * **人手で凍結コピー**する。
 */
const IMAGE_COMMAND = "deno task demo:eval-images --source <Anima 配布形のパス>";

/**
 * ファイルの存在。
 * MUST: NotFound 以外は伝播させる —— 権限エラー等を「資産が無い」と読み替えると、
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

/** 登録時点で必要なので同期で数える（Deno.test の ignore 判定に使う）。 */
const goldenCount = (series: string): number =>
  REAL_CASES.filter((entry) => exists(goldenUrl(series, entry.name))).length;

const imageCount = REAL_CASES.filter((entry) => exists(new URL(entry.file, CORPUS_DIR))).length;

const readBytes = (url: URL): Promise<Uint8Array<ArrayBuffer>> => Deno.readFile(url);

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

for (const series of SERIES) {
  const goldens = goldenCount(series);
  /** golden も画像も 1 件も無い = 生成していない環境（部分的な欠けは完全性テストが FAIL）。 */
  const anyAsset = goldens > 0 || imageCount > 0;
  const runnable = goldens > 0 && imageCount > 0;

  if (!runnable) {
    console.warn(
      `[karume] 実画像の前処理パリティ (${series}) を SKIP する（golden ${goldens}/` +
        `${REAL_CASES.length} 本・画像 ${imageCount}/${REAL_CASES.length} 枚）。` +
        `golden の生成: ${generateCommand(series)} / 画像の生成: ${IMAGE_COMMAND}`,
    );
  }

  Deno.test({
    name: `実画像前処理 資産: ${series} — golden と PNG が揃っている`,
    // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
    ignore: !anyAsset,
    fn: () => {
      assert(
        goldens === 0 || goldens === REAL_CASES.length,
        `${series} の実画像 golden が ${goldens}/${REAL_CASES.length} 本` +
          `（採り直す: ${generateCommand(series)}）`,
      );
      assert(
        imageCount === 0 || imageCount === REAL_CASES.length,
        `${CORPUS_DIR.pathname} の実画像が ${imageCount}/${REAL_CASES.length} 枚` +
          `（焼き直す: ${IMAGE_COMMAND}）`,
      );
    },
  });

  for (const entry of REAL_CASES) {
    Deno.test({
      name: `実画像前処理パリティ: ${series} / ${entry.name}（TS の f64 経路 対 torchvision）`,
      ignore: !runnable,
      fn: async () => {
        const [goldenBytes, png] = await Promise.all([
          readBytes(goldenUrl(series, entry.name)),
          readBytes(new URL(entry.file, CORPUS_DIR)),
        ]);
        const io = parseSafetensors(
          goldenBytes.buffer.slice(
            goldenBytes.byteOffset,
            goldenBytes.byteOffset + goldenBytes.byteLength,
          ),
        );

        // ① golden を採った画像と、いま読んでいる画像が同一であること。**tolerance では
        // 吸収されない差**（生成台本を回し直して golden を採り直していない）を、突合の前に
        // 名指しで落とす。ここが緩むと「別の画像どうしを比べて緑」が成立しうる。
        assertEquals(io.metadata.get(SOURCE_IMAGE_KEY), entry.file, "golden が指す元画像");
        assertEquals(
          io.metadata.get(SOURCE_SHA256_KEY),
          await sha256Hex(png),
          `${entry.file} が golden を採った画像と違う（採り直す: ${generateCommand(series)}）`,
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
          `${series} / ${entry.name}: pixel_values の maxAbs=${maxAbs}` +
            `（標本 ${worst}: ${got[worst]} 対 ${expected[worst]}）が ${PIXEL_ATOL} を超えた`,
        );
      },
    });
  }
}
