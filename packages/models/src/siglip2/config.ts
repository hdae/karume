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
 * SigLIP2 は base（224 / hidden 768）と so400m（384 / hidden 1152）で寸法だけが違う同じ経路
 * なので、TS に 224 や 768 を直書きすると**もう片方の配布形で沈黙誤値になる**（前処理が別の
 * 寸法へ resize したまま shape だけ合う、ということは起きない — グラフが弾く — が、寸法を
 * 取り違えた側は resize を 2 度通ったような画になる）。実グラフとの突合は
 * `Siglip2Pipeline.fromAssets` が資産に対して行う。
 *
 * ## MUST: 補間は**宣言**として持ち、対応外は値を保持せずパース時に拒否する
 *
 * 上流 `preprocessor_config.json` の `"resample"` が 2（PIL の BILINEAR）なので、この家族の
 * 前処理は bilinear が正本。bicubic を要求するチェックポイントを黙って bilinear で通すと、
 * **resize の値が最大 47/255 ずれたまま**ロードも実行も通る（実測 — 前処理層のモジュール
 * doc）。分岐を持つのではなく**受理しない**（型としても `"bilinear"` しか表せない）。
 * 前処理層（`src/image/preprocess.ts`）は bicubic も実装しているので、受理集合を広げるかは
 * 「この配布形の上流がどの `resample` で学習されたか」の判断であって実装の有無ではない。
 *
 * NOTE: rescale の除数（255）はここに無い。`normalizeToNchw` の入口が 8bit の画素列
 * （`Rgb8Image`）で閉じており、`rescale_factor` が 1/255 でないチェックポイントは
 * **配布形を組む段**で落ちる（`tools/export-recipes/siglip2/distribution.py`）。実行時に
 * 選べない数を宣言だけ持たせても、二重の正本が増えるだけになる。
 *
 * NOTE: 公開配布リポの対応表（{@link SIGLIP2_SOURCES}）もここに置く。上の MUST が禁じる
 * 「モデル固有の数」ではなく「どの manifest を取りに行くか」の側で、そもそも配布形が持てない
 * 値だから（ADR 0073）。
 */

import type { HubRepoRef } from "@karume/hub";
import {
  assertAllowedKeys,
  isPositiveInteger,
  readChannels,
  readNumber,
  readOnly,
} from "../config/readers.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const SIGLIP2_PIPELINE_NAME = "siglip2";
export const SIGLIP2_PIPELINE_MAJOR = 1;

/**
 * SigLIP2 ファミリの**公開配布リポ対応表**（ADR 0092 — 家族 1 つにつき 1 表・**既定の席は
 * 無い**）。値は**このパッケージ版が検証した取得元**（pin 済み commit SHA — ADR 0073）。
 *
 * キーは HF リポ名の basename から `karume-` を落とした綴り（`"karume-" + key` がリポ名の
 * basename に戻る — この不変条件は `tests/sources_test.ts` の門が見る）。同一家族 = 1 リポで、
 * 寸法だけが違う 2 モデルが同居する:
 *
 * - `"siglip2"` = `hdae/karume-siglip2`（base 224 / hidden 768 と so400m 384 / hidden 1152 が
 *   同居・既定 = base）
 *
 * 1 リポ = 複数モデルなので、リポ参照だけでは 1 本に決まらない — so400m を使うときは
 * `fromPretrained(SIGLIP2_SOURCES["siglip2"], { model: "so400m" })` と綴る
 * （`Siglip2PipelineOptions.model` — `./pipeline.ts`）。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン**として渡す — 再現性を自分で
 * 固定したい場合は、この表ではなく自分の `{ repo, revision }` を書く（`fromPretrained` に
 * 既定は無い）。
 *
 * MUST: revision は commit SHA で固定する — ブランチ・タグは配布側で付け替えられるので、
 * 公開済みのこのパッケージが読むバイト列がネットワーク側の都合で黙って変わる（回復不能側の
 * 事故）。SHA 指定は revision 解決要求そのものを消すため、完全キャッシュ時のオフライン起動も
 * 同時に成立する（ADR 0038）。main 追従が要る利用者は
 * `{ ...SIGLIP2_SOURCES["siglip2"], revision: "main" }` を明示的に選ぶ。
 */
// NOTE: revision はリリース手順書（docs/release-runbook.md）§3 で、アップロード後の main の
// SHA に更新する（ADR 0073 決定 3 の維持義務を継承 — 手書き + 手順書ゲート）。
export const SIGLIP2_SOURCES = {
  "siglip2": {
    repo: "hdae/karume-siglip2",
    revision: "71077b1810e42e9a1a25d80ee0f43c75db5d3b23",
  },
} as const satisfies Record<string, HubRepoRef>;

const ROOT_KEYS: readonly string[] = [
  "imageWidth",
  "imageHeight",
  "imageMean",
  "imageStd",
  "hiddenDim",
  "interpolation",
];

/** この実装が受理する唯一の補間（モジュール doc の MUST）。 */
const INTERPOLATION = "bilinear";

/** 正規化の定数（`[0, 1]` 尺度 — `preprocessor_config.json` の綴りのまま）。 */
type Siglip2Channels = readonly [number, number, number];

export type Siglip2PipelineConfig = {
  /** 前処理の resize 先（`preprocessor_config.json` の `size.width`）。 */
  readonly imageWidth: number;
  /** 前処理の resize 先（`preprocessor_config.json` の `size.height`）。 */
  readonly imageHeight: number;
  readonly imageMean: Siglip2Channels;
  readonly imageStd: Siglip2Channels;
  /** `pooler_output` の幅（vision tower の hidden — base 768 / so400m 1152）。 */
  readonly hiddenDim: number;
  /** ADR 0038 §1 の流儀で**宣言**として持つ（分岐用ではない — モジュール doc の MUST）。 */
  readonly interpolation: typeof INTERPOLATION;
};

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査して読む。 */
export const parseSiglip2PipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): Siglip2PipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positive = "正の整数でない";
  return {
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
    hiddenDim: readNumber(raw, "hiddenDim", where, isPositiveInteger, positive),
    interpolation: readOnly(
      raw,
      "interpolation",
      where,
      INTERPOLATION,
      "上流 preprocessor_config.json の resample は 2（PIL の BILINEAR）で、" +
        "bicubic で通すと resize の値が最大 47/255 ずれる",
    ),
  };
};
