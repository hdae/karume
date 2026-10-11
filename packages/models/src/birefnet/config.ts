/**
 * `pipelineConfig` のスキーマ検証（ADR 0038 §1 — スキーマは各パイプライン実装が所有・検証）。
 *
 * hub は `pipelineConfig` を素通しする（禁止キーの一掃と規模上限だけを見る）。したがって
 * **形の正本はこのモジュール**で、手書きの検査を全て parse 時に走らせる。
 *
 * MUST: 未知キーは fail loudly（`{ imageMean: …, image_mean: … }` のような綴り違いが黙って
 * 既定へ縮退すると、配布者の意図した前処理と実行が食い違ったまま気づけない）。
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 *
 * ## MUST: モデル固有の数はここ（= manifest）が正本で、TS 側に定数を置かない
 *
 * BiRefNet 系は上流に `preprocessor_config.json` が無く、正規化定数は同梱 `handler.py` の
 * `ImagePreprocessor`（ImageNet 統計）にしか無い。だからといって TS に 0.485 を直書きすると、
 * 別の統計で学習された派生モデル（この形の配布形は fine-tune 違いが並ぶ）で**沈黙誤値**に
 * なる — 正規化がずれても shape は合い、マットは「それらしく」出るからである。
 * `imageMean` / `imageStd` を宣言として持ち、配布形ごとに読む。
 *
 * ## MUST: 補間は**宣言**として持ち、対応外は値を保持せずパース時に拒否する
 *
 * 上流（`handler.py` / モデルカードの利用例）は `torchvision.transforms.Resize((S, S))` を
 * 既定の補間で通す = bilinear なので、この家族の前処理は bilinear が正本。bicubic を要求する
 * 派生を黙って bilinear で通すと **resize の値が最大 47/255 ずれたまま**ロードも実行も通る
 * （実測 — 前処理層のモジュール doc）。分岐を持つのではなく**受理しない**（型としても
 * `"bilinear"` しか表せない）。前処理層（`src/image/preprocess.ts`）は bicubic も実装して
 * いるので、受理集合を広げるかは「その派生の上流がどの補間で学習されたか」の判断であって
 * 実装の有無ではない。
 *
 * ## MUST: 後処理の手順も**宣言**として持ち、閉じた集合の外は fail loudly
 *
 * logit → 8bit の α の後処理（`./matte.ts` の {@link MatteProcedure}）は系列ごとに上流が違う
 * （HR = 8bit へ丸めてから Pillow の BICUBIC・Lucida = 浮動小数のまま antialias bilinear → 8bit —
 * ADR 0125）。正規化定数と同じ理由で「系列 → 手順」の表を TS に置かない（表に無い第三者の
 * fine-tune で沈黙の取り違えになる）。組み立て段（`distribution.py`）が checkpoint から決めて
 * `postprocess` に書き、ここは受理するだけ。欄は必須で既定を持たない — 書き忘れた配布形が、黙って
 * 片方の系列の手順で走らないようにするため。必須の欄の追加は旧い配布形を読めなくするので、
 * pipeline の major を 2 に上げた（ADR 0038 の繰り上げ規則）。
 *
 * NOTE: rescale の除数（255）はここに無い。`normalizeToNchw` の入口が 8bit の画素列
 * （`Rgb8Image`）で閉じており、実行時に選べない数を宣言だけ持たせても正本が 2 つ増える
 * （`src/siglip2/config.ts` と同じ判断）。
 *
 * NOTE: `imageWidth` / `imageHeight` は焼かれたグラフの入力宣言と**同じ数**で、組み立て段
 * （`tools/export-recipes/birefnet/distribution.py`）がグラフから導いて書く。それでも宣言を
 * 置くのは、①前処理の resize 先はグラフを開く前に読めるべき欄で ②モデルカードが解像度を
 * 説明できるようにするため。二重化した分は {@link BirefnetPipelineConfig} を使う側
 * （`pipeline.ts` の `assertStaticDim`）が**毎回グラフと突き合わせる**ので、食い違ったまま
 * 走ることはない。
 */

import type { HubRepoRef } from "@karume/hub";
import {
  assertAllowedKeys,
  isPositiveInteger,
  readChannels,
  readNumber,
  readOnly,
} from "../config/readers.ts";
import { normalizationOverflow } from "../image/preprocess.ts";
import { MATTE_PROCEDURES, type MatteProcedure } from "./matte.ts";

/**
 * `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。
 *
 * major 2 = `postprocess` の欄が必須になった形（モジュール doc — ADR 0125）。major 1 の配布形は
 * 手順を宣言していないので読まない。
 */
export const BIREFNET_PIPELINE_NAME = "birefnet";
export const BIREFNET_PIPELINE_MAJOR = 2;

/**
 * BiRefNet 系の**公開配布リポ対応表**（ADR 0092 — 家族 1 つにつき 1 表・**既定の席は無い**）。
 * 値は**このパッケージ版が検証した取得元**（pin 済み commit SHA — ADR 0073）。
 *
 * キーは HF リポ名の basename から `karume-` を落とした綴り（`"karume-" + key` がリポ名の
 * basename に戻る — この不変条件は `tests/sources_test.ts` の門が見る）。checkpoint ごとに
 * 1 リポで、各リポに**解像度ごとの別グラフ**が 2 モデル同居する（ADR 0092 決定 9）:
 *
 * - `"birefnet-hr"` = `hdae/karume-birefnet-hr`（上流 BiRefNet_HR — モデル `"1024"` / `"2048"`・
 *   既定 `"1024"`）
 * - `"lucida"` = `hdae/karume-lucida`（BiRefNet_HR の派生 Lucida — 同じ 2 モデル・既定 `"1024"`）
 *
 * 1 リポ = 2 モデルなので、2048² を使うときは
 * `fromPretrained(BIREFNET_SOURCES["birefnet-hr"], { model: "2048" })` と綴る
 * （`BirefnetPipelineOptions.model` — `./pipeline.ts`）。モデル名は入力解像度そのもので、
 * 前処理の resize 先は `pipelineConfig.imageWidth` / `imageHeight` がモデルごとに宣言する。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン**として渡す — 再現性を自分で
 * 固定したい場合は、この表ではなく自分の `{ repo, revision }` を書く（`fromPretrained` に
 * 既定は無い）。
 *
 * MUST: revision は commit SHA で固定する — ブランチ・タグは配布側で付け替えられるので、
 * 公開済みのこのパッケージが読むバイト列がネットワーク側の都合で黙って変わる（回復不能側の
 * 事故）。SHA 指定は revision 解決要求そのものを消すため、完全キャッシュ時のオフライン起動も
 * 同時に成立する（ADR 0038）。main 追従が要る利用者は
 * `{ ...BIREFNET_SOURCES["birefnet-hr"], revision: "main" }` を明示的に選ぶ。
 */
// NOTE: revision はリリース手順書（docs/release-runbook.md）§3 で、アップロード後の main の
// SHA に更新する（ADR 0073 決定 3 の維持義務を継承 — 手書き + 手順書ゲート）。
export const BIREFNET_SOURCES = {
  "birefnet-hr": {
    repo: "hdae/karume-birefnet-hr",
    revision: "b470ac9ab676d18356f666da10fae15d2f5351ad",
  },
  "lucida": {
    repo: "hdae/karume-lucida",
    revision: "779ee5afcb946d7c147bf63414e81037873d2306",
  },
} as const satisfies Record<string, HubRepoRef>;

const ROOT_KEYS: readonly string[] = [
  "imageWidth",
  "imageHeight",
  "imageMean",
  "imageStd",
  "interpolation",
  "postprocess",
];

/** この実装が受理する唯一の補間（モジュール doc の MUST）。 */
const INTERPOLATION = "bilinear";

/** 正規化の定数（`[0, 1]` 尺度 — 上流 `handler.py` の綴りのまま）。 */
type BirefnetChannels = readonly [number, number, number];

export type BirefnetPipelineConfig = {
  /** 前処理の resize 先（= 焼かれたグラフの入力幅）。 */
  readonly imageWidth: number;
  /** 前処理の resize 先（= 焼かれたグラフの入力高さ）。 */
  readonly imageHeight: number;
  readonly imageMean: BirefnetChannels;
  readonly imageStd: BirefnetChannels;
  /** ADR 0038 §1 の流儀で**宣言**として持つ（分岐用ではない — モジュール doc の MUST）。 */
  readonly interpolation: typeof INTERPOLATION;
  /** logit → 8bit の α の手順（モジュール doc の MUST — 系列ごとの上流の手順を配布形が宣言する）。 */
  readonly postprocess: MatteProcedure;
};

/**
 * `postprocess` を読む（閉じた集合 {@link MATTE_PROCEDURES} の外は落とす）。
 *
 * 受理集合の配列と `===` で突き合わせる（表引きをしないので、`"toString"` のような
 * Object.prototype の綴りを手順と取り違える余地が無い）。
 */
const readPostprocess = (raw: Readonly<Record<string, unknown>>, where: string): MatteProcedure => {
  const key = "postprocess";
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  const procedure = MATTE_PROCEDURES.find((name) => name === value);
  if (procedure === undefined) {
    throw new Error(
      `${where}.${key}: 未知の後処理の手順 ${JSON.stringify(value)}（期待 ${
        MATTE_PROCEDURES.map((name) => `'${name}'`).join(" / ")
      }）`,
    );
  }
  return procedure;
};

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査して読む。 */
export const parseBirefnetPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): BirefnetPipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positive = "正の整数でない";
  const config: BirefnetPipelineConfig = {
    imageWidth: readNumber(raw, "imageWidth", where, isPositiveInteger, positive),
    imageHeight: readNumber(raw, "imageHeight", where, isPositiveInteger, positive),
    // mean は負でも構わない（`(x − mean·255) / (std·255)` の平行移動）が、有限でなければ
    // 全画素が NaN になる。
    imageMean: readChannels(raw, "imageMean", where, Number.isFinite, "有限の数でない要素がある"),
    imageStd: readChannels(
      raw,
      "imageStd",
      where,
      (value) => Number.isFinite(value) && value > 0,
      "正の有限数でない要素がある",
    ),
    interpolation: readOnly(
      raw,
      "interpolation",
      where,
      INTERPOLATION,
      "上流 handler.py の torchvision Resize は既定の補間（= bilinear）で、" +
        "bicubic で通すと resize の値が最大 47/255 ずれる",
    ),
    postprocess: readPostprocess(raw, where),
  };
  // 個々の欄が有限でも、組で正規化後の値が f32 で溢れる配布形は資産の不正として構築時に落とす
  // （`normalizeToNchw` まで待つと、生成のたびに入力起因の型で落ちる）。
  const overflow = normalizationOverflow(config.imageMean, config.imageStd);
  if (overflow !== undefined) throw new Error(`${where}.imageMean / imageStd: ${overflow}`);
  return config;
};
