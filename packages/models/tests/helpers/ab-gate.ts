/**
 * 自機 A/B 門（ADR 0110 決定 5）の**純関数部**。GPU も FS も触らない — 実 GPU の門
 * （`e2e_anima_ab_test.ts` / `e2e_irodori_ab_test.ts` / `e2e_sbv2_ab_test.ts`）は観測点の
 * 配列と census を採るところまでを持ち、「何を赤にするか」はここの 1 本に寄せる（家族ごとに
 * 判定の順と文面が割れない）。振る舞いは `ab_gate_test.ts`（core レーン）が固定する。
 *
 * 比較相手は**同じ機の参照層**（同じ重み・`session` が空の席）で、torch golden は使わない。
 * 同機の参照層は決定的なので対照側の雑音は 0 — 束のノブが丸ごと Session に届かなかったとき、
 * 実用席の観測点は参照席と**1 bit も違わない**。床の最小要件はこの一点である（決定 5-2）。
 */

import type { GpuFeaturesSpec, SessionSpec } from "@karume/hub";
import type { SessionOptions } from "@karume/runtime";
import {
  type CensusRow,
  countDispatches,
  type KeyPredicate,
  type NumericSeat,
  SEAT_REFERENCE,
  SEAT_SIGNATURES,
} from "../../../runtime/tests/helpers/pipeline-census.ts";
import type { Comparison } from "../../../runtime/tests/helpers/results.ts";

/**
 * 帯（ADR 0110 決定 4 / 5 — 宣言であって環境キー別の行ではない・`KARUME_REFERENCE` で書き換え
 * ない）。指標は観測点の相対 RMS（{@link relRms}）。
 *
 * - `floor` — 1 bit 以上違う（決定 5-2 の MUST — {@link judgeAb} が帯と無関係に常に見る）に
 *   **加えて**置く下限。束の一部だけが落ちた形を数値で掴めると実測で示せたときだけ 0 より上に
 *   置く（掴めないならノブ単位の縮退は census が持つ — 決定 5-1）。
 * - `ceiling` — 崩壊上限（決定 5-3 — 観測点の実測 × 2 程度）。
 */
export type AbBand = { readonly floor: number; readonly ceiling: number };

/** 観測点 1 つぶんの A/B の実測。 */
export type AbMeasurement = {
  /** ‖p − r‖₂ / ‖r‖₂（{@link relRms}）。 */
  readonly relRms: number;
  readonly maxAbs: number;
  /** 実用席と参照席の観測点がビット単位で同一か（床の判定材料）。 */
  readonly identical: boolean;
};

/** {@link referencePartnerOf} が読む manifest の部分（`Manifest` はこれに代入できる）。 */
export type QuantTable = {
  readonly models: Readonly<
    Record<string, {
      readonly quants: Readonly<
        Record<string, {
          readonly weights: Readonly<Record<string, string>>;
          readonly session: SessionSpec;
          readonly gpuFeatures?: GpuFeaturesSpec;
        }>
      >;
    }>
  >;
};

type QuantEntry = QuantTable["models"][string]["quants"][string];

/** model / quant の宣言を引く（無ければ fail loudly — あるものを並べる）。 */
export const quantOf = (manifest: QuantTable, model: string, quant: string): QuantEntry => {
  if (!Object.hasOwn(manifest.models, model)) {
    throw new Error(
      `manifest に model '${model}' が無い（あるもの: ${
        Object.keys(manifest.models).join(" / ")
      }）`,
    );
  }
  const { quants } = manifest.models[model];
  if (!Object.hasOwn(quants, quant)) {
    throw new Error(
      `model '${model}' に quant '${quant}' が無い（あるもの: ${Object.keys(quants).join(" / ")}）`,
    );
  }
  return quants[quant];
};

const sameWeights = (
  left: Readonly<Record<string, string>>,
  right: Readonly<Record<string, string>>,
): boolean => {
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length &&
    keys.every((key) => Object.hasOwn(right, key) && right[key] === left[key]);
};

/** 要求が立っている feature 名（`shaderF16: false` は要求していないのと同じ）。 */
const requiredFeatures = (features: GpuFeaturesSpec | undefined): ReadonlySet<string> =>
  new Set(
    Object.entries(features ?? {}).filter(([, required]) => required === true).map(([name]) =>
      name
    ),
  );

/**
 * 実用席の**参照席**（同じ `weights` 写像・`session` が空・参照席の `gpuFeatures` の要求が
 * 実用席の要求の部分集合、がちょうど 1 つ）を manifest から導く。
 *
 * 部分集合にするのは、実用席だけが要る feature（`f16-c16` の shader-f16）を参照席が持たない形を
 * 対にするため（反証 verify/A2 §4-3 — 「同じ」では `f16-c16 ↔ f16` が候補 0 になる）。逆向き
 * （参照席だけが feature を要る）は、実用席が取る device で参照席が組めないので対にしない。
 *
 * MUST: 候補 0 / 2 以上は throw する。0 は「実用席に参照層の対照が配られていない」、2 以上は
 * 「どちらと比べたかで床の意味が変わる」— どちらも黙って 1 つを選ぶと門の主張が曖昧になる。
 */
export const referencePartnerOf = (manifest: QuantTable, model: string, quant: string): string => {
  const practical = quantOf(manifest, model, quant);
  if (Object.keys(practical.session).length === 0) {
    throw new Error(`quant '${quant}' は session が空（参照席）— A/B の実用席にならない`);
  }
  const practicalFeatures = requiredFeatures(practical.gpuFeatures);
  const { quants } = manifest.models[model];
  const candidates = Object.keys(quants).filter((name) => {
    if (name === quant) return false;
    const candidate = quants[name];
    return sameWeights(candidate.weights, practical.weights) &&
      Object.keys(candidate.session).length === 0 &&
      [...requiredFeatures(candidate.gpuFeatures)].every((feature) =>
        practicalFeatures.has(feature)
      );
  });
  if (candidates.length !== 1) {
    throw new Error(
      `quant '${quant}' の参照席（同じ weights・session が空・gpuFeatures ⊆ 実用席）が ` +
        `${candidates.length} 個（${candidates.join(" / ") || "なし"}）— ちょうど 1 つでないと` +
        "床の対照が決まらない",
    );
  }
  return candidates[0];
};

const assertSameLength = (practical: Float32Array, reference: Float32Array): void => {
  if (practical.length !== reference.length) {
    throw new Error(
      `観測点の要素数が違う（実用席 ${practical.length} / 参照席 ${reference.length}）— ` +
        "同じ入力で同じ観測点を採っていない",
    );
  }
};

/**
 * 相対 RMS ‖p − r‖₂ / ‖r‖₂（f64 で積む）。
 *
 * 非有限は伝播させる（NaN / ±Inf の要素・‖r‖₂ = 0 は非有限の値になる）— 判定は
 * {@link judgeAb} が上限側の失敗として扱う。
 */
export const relRms = (practical: Float32Array, reference: Float32Array): number => {
  assertSameLength(practical, reference);
  let difference = 0;
  let norm = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const delta = practical[index] - reference[index];
    difference += delta * delta;
    norm += reference[index] * reference[index];
  }
  return Math.sqrt(difference) / Math.sqrt(norm);
};

/**
 * 全要素の最大絶対差。NaN の要素があれば NaN を返す（`>` の比較は NaN で常に偽なので、
 * 素朴に書くと NaN が 0 として素通りする — `helpers/irodori-assets.ts` の `worstDifference`）。
 */
export const maxAbs = (practical: Float32Array, reference: Float32Array): number => {
  assertSameLength(practical, reference);
  let worst = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const difference = Math.abs(practical[index] - reference[index]);
    if (Number.isNaN(difference)) return Number.NaN;
    if (difference > worst) worst = difference;
  }
  return worst;
};

const bitsOf = (values: Float32Array): Uint32Array =>
  new Uint32Array(values.buffer, values.byteOffset, values.length);

/** ビット列が違う最初の添字と、違う要素の数（同一なら undefined）。 */
const bitDifference = (
  left: Float32Array,
  right: Float32Array,
): { readonly first: number; readonly count: number } | undefined => {
  assertSameLength(left, right);
  const leftBits = bitsOf(left);
  const rightBits = bitsOf(right);
  let first = -1;
  let count = 0;
  for (let index = 0; index < leftBits.length; index += 1) {
    if (leftBits[index] === rightBits[index]) continue;
    if (first < 0) first = index;
    count += 1;
  }
  return first < 0 ? undefined : { first, count };
};

/** 観測点 1 つぶんの実測（{@link judgeAb} の入力）。 */
export const measureAb = (practical: Float32Array, reference: Float32Array): AbMeasurement => ({
  relRms: relRms(practical, reference),
  maxAbs: maxAbs(practical, reference),
  identical: bitDifference(practical, reference) === undefined,
});

/**
 * デバイス内決定性（ADR 0110 決定 6 の MUST）の機械検査 — 同じ席・同じ入力・独立 2 Session の
 * 観測点がビット単位で同一であること。
 */
export const assertBitIdentical = (
  first: Float32Array,
  second: Float32Array,
  where: string,
): void => {
  const difference = bitDifference(first, second);
  if (difference === undefined) return;
  const { first: at, count } = difference;
  throw new Error(
    `${where}: 同じ席・同じ入力の 2 回がビット一致しない（決定性 MUST 違反 — ADR 0110 決定 6）\n` +
      `  ${count} / ${first.length} 要素が違う。最初は ${at}: ${first[at]} / ${second[at]}\n` +
      "  浮動小数の atomics・到着順の合流・キーに載らない実行時パラメータを疑う",
  );
};

const format = (value: number): string =>
  Number.isFinite(value) ? value.toExponential(4) : String(value);

/**
 * 1 観測点の A/B を判定する。赤の順は ① 非有限（上限側）② 1 bit も違わない（床 MUST）
 * ③ 帯が未導出 ④ 帯の床 ⑤ 帯の上限。
 *
 * MUST: ② は帯の有無に関係なく見る — 束全体の沈黙縮退（ノブが 1 つも Session に届かない）は
 * 帯を導く前から検出できるし、そこで緑にすると未導出の回が「届いていない」を隠す。
 * MUST: ③ は実測を文面に出して赤で止める（`e2e_irodori_w8a8_test.ts` の `MEASURED` と同じ流儀 —
 * 通る値を仮置きして検出力の無い門を増やさない）。
 */
export const judgeAb = (measured: AbMeasurement, band: AbBand | undefined, where: string): void => {
  const observed = describeAb(measured);
  if (!Number.isFinite(measured.relRms) || !Number.isFinite(measured.maxAbs)) {
    throw new Error(
      `${where}: 上限側の失敗 — 非有限（${observed}）。観測点に NaN / Inf が出たか、参照席の` +
        "観測点がゼロ（‖r‖₂ = 0）",
    );
  }
  if (measured.identical) {
    throw new Error(
      `${where}: 床の失敗 — 実用席が参照席と 1 bit も違わない（束のノブが Session に届いて` +
        "いない疑い — quant の session → SessionOptions → Session の配線か適格判定を見る）",
    );
  }
  if (band === undefined) {
    throw new Error(
      `${where}: 帯が未導出（実測 ${observed}）— この実測から ADR 0110 決定 5 の規則で帯を導き、` +
        "導出表と一緒にテストの定数へ書く",
    );
  }
  if (!(band.floor >= 0 && band.ceiling > band.floor && Number.isFinite(band.ceiling))) {
    throw new Error(
      `${where}: 帯の宣言が壊れている（floor ${band.floor} / ceiling ${band.ceiling}）`,
    );
  }
  if (measured.relRms < band.floor) {
    throw new Error(
      `${where}: 床の失敗 — ${observed} が床 ${band.floor} を下回った（束の一部のノブが落ちて` +
        "差が縮んだ疑い — 差が小さいのは沈黙フォールバックの兆候で、良化ではない）",
    );
  }
  if (measured.relRms > band.ceiling) {
    throw new Error(
      `${where}: 上限の失敗 — ${observed} が上限 ${band.ceiling} を超えた（崩壊の検出）。帯を` +
        "広げる前に、どのカーネルの数値が動いたかを op の門で確かめる",
    );
  }
};

/**
 * 観測点を 1 度採ったところで生成を打ち切る（購読側の throw が生成を落とす流儀 — anima /
 * irodori の `onEvent` の doc）。`run` は受け取った `capture` を観測点で呼ぶ。
 *
 * MUST: 打ち切りの印以外の例外はそのまま上げる（生成の失敗を「打ち切った」で隠さない）。
 * 観測点が 1 度も来ないまま生成が終わった回も落とす（step 1 に届かない設定で緑にしない）。
 */
export const captureThenAbort = async <T>(
  where: string,
  run: (capture: (value: T) => never) => Promise<unknown>,
): Promise<T> => {
  const stop = new Error(`${where}: 観測点を採ったので生成を打ち切る`);
  let captured: { readonly value: T } | undefined;
  try {
    await run((value) => {
      captured = { value };
      throw stop;
    });
  } catch (error) {
    if (error !== stop) throw error;
  }
  if (captured === undefined) {
    throw new Error(`${where}: 観測点が 1 度も来ないまま生成が終わった`);
  }
  return captured.value;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const recordAt = (
  parent: Record<string, unknown>,
  key: string,
  where: string,
): Record<string, unknown> => {
  const child = parent[key];
  if (!Object.hasOwn(parent, key) || !isRecord(child)) {
    throw new Error(`${where}.${key}: オブジェクトでない / 無い`);
  }
  return child;
};

/**
 * 配布 JSON（`karume.json` の生の中身）の 1 席の `session` だけを差し替えた写しを返す
 * （故障注入用の一時 manifest — 重みも他の席も触らない）。元の値は書き換えない。
 */
export const overrideQuantSession = (
  raw: unknown,
  model: string,
  quant: string,
  session: Readonly<Record<string, unknown>>,
): Record<string, unknown> => {
  if (!isRecord(raw)) throw new Error("manifest: オブジェクトでない");
  const models = recordAt(raw, "models", "manifest");
  const entry = recordAt(models, model, "manifest.models");
  const quants = recordAt(entry, "quants", `manifest.models.${model}`);
  const seat = recordAt(quants, quant, `manifest.models.${model}.quants`);
  return {
    ...raw,
    models: {
      ...models,
      [model]: { ...entry, quants: { ...quants, [quant]: { ...seat, session: { ...session } } } },
    },
  };
};

/** census の網羅表の席名（`Object.keys` は string[] を返すので網羅表の型から取り直す）。 */
const NUMERIC_SEATS = Object.keys(SEAT_REFERENCE) as readonly NumericSeat[];

/**
 * 束の各ノブ（非参照値の席）の変種が走った dispatch 本数の 1 行表示（`linearCompute=a8 ×317 / …`）。
 * 決着の note とログに残す — 「効いた」の判定は `assertSeatsApplied` が持ち、ここは本数の記録だけ。
 */
export const describeBundleCensus = (
  rows: readonly CensusRow[],
  declared: SessionOptions,
): string =>
  NUMERIC_SEATS.flatMap((seat) => {
    const value = declared[seat];
    if (value === undefined || value === SEAT_REFERENCE[seat]) return [];
    const signatures: Readonly<Record<string, KeyPredicate>> = SEAT_SIGNATURES[seat];
    const predicate = Object.hasOwn(signatures, String(value))
      ? signatures[String(value)]
      : undefined;
    const count = predicate === undefined ? "述語なし" : String(countDispatches(rows, predicate));
    return [`${seat}=${String(value)} ×${count}`];
  }).join(" / ");

/**
 * 結果の席（`results.json` の `comparisons`）へ積む 1 本。判定の**前**に積む — 赤の回も実測が
 * 決着に残る（帯を導く材料は割れた回にも要る）。
 */
export const comparisonOf = (
  observation: { readonly output: string; readonly reference: string; readonly practical: string },
  measured: AbMeasurement,
  band: AbBand | undefined,
): Comparison => ({
  ...observation,
  relRms: measured.relRms,
  maxAbs: measured.maxAbs,
  band: band === undefined ? undefined : { metric: "relRms", ...band },
});

/** 実測のログ表示（`relRMS 1.2345e-2 / maxAbs 3.1000e-1`）。 */
export const describeAb = (measured: AbMeasurement): string =>
  `relRMS ${format(measured.relRms)} / maxAbs ${format(measured.maxAbs)}`;
