/**
 * `pipelineConfig` のスキーマ検証（ADR 0038 §1 — スキーマは各パイプライン実装が所有・検証）。
 *
 * hub は `pipelineConfig` を素通しする（禁止キーの一掃と規模上限だけを見る）。したがって
 * **形の正本はこのモジュール**で、手書きの検査を全て parse 時に走らせる。
 *
 * MUST: 未知キーは fail loudly（`{ noiseScale: 0.6, noise_scale: 0.6 }` のような綴り違いが
 * 黙って既定へ縮退すると、配布者の意図した既定と実行が食い違ったまま気づけない）。
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 *
 * ## MUST: `styles` / `speakers` は「名前 → **表の行番号**」であって ID の別名ではない
 *
 * 解決先は配布形の `style_vectors` / `speaker_embeddings`（`style.ts`）で、行番号がずれても
 * shape は合ったままロードも実行も通り、**別のスタイル・別の話者の声が出る**だけで沈黙する。
 * 検出手段が他に無いので、値が `0..rows-1` の**順列**であることを parse 時に見る（表の行数
 * との突合は `Sbv2Pipeline.fromAssets` が実資産に対して行う）。
 *
 * ## MUST: `maxTokens` / `maxFrames` は配布形が宣言する運用上限
 *
 * 窓付き相対位置注意の `(T, T)` 表は ADR 0045 でホストへ外出しされ、確保（8·T² bytes 級）が
 * ホストの責務になった。焼いたグラフの記号次元の上限を配布形から受け、`generate` が**表を
 * 確保する前**にこの値で落とす（正本は exporter の `dist.py`）。TS 側に定数を持たない —
 * 別の上限で焼いた配布形が来たときに、ホストだけが古い数を持つ形を作らないため。
 *
 * NOTE: 実行時ノブの既定は `symbols.json` にも同じ値が並ぶ（exporter 側が両者の食い違いを
 * 組み立て時に落とす）。パイプラインが読むのは**こちら**だけで、`symbols.json` の `defaults`
 * は見ない — 導出元を二重に持たないため（`text/symbols.ts` の同節は optional）。
 */

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const SBV2_PIPELINE_NAME = "sbv2";
export const SBV2_PIPELINE_MAJOR = 1;

const ROOT_KEYS: readonly string[] = ["styles", "speakers", "maxTokens", "maxFrames", "defaults"];
const DEFAULTS_KEYS: readonly string[] = [
  "speaker",
  "style",
  "styleWeight",
  "sdpRatio",
  "noiseScale",
  "noiseScaleW",
  "lengthScale",
];

/** 配布者の推奨既定（`generate` の未指定欄を埋める）。 */
export type Sbv2Defaults = {
  readonly speaker: string;
  readonly style: string;
  readonly styleWeight: number;
  readonly sdpRatio: number;
  readonly noiseScale: number;
  readonly noiseScaleW: number;
  readonly lengthScale: number;
};

export type Sbv2PipelineConfig = {
  /** スタイル名 → `style_vectors` の行番号（`0..rows-1` の順列）。 */
  readonly styles: ReadonlyMap<string, number>;
  /** 話者名 → `speaker_embeddings` の行番号（`0..rows-1` の順列）。 */
  readonly speakers: ReadonlyMap<string, number>;
  /** DeBERTa へ渡せるトークン列の上限（焼いたグラフの記号次元の上限）。 */
  readonly maxTokens: number;
  /** flow / voice へ渡せる総フレーム数の上限（同上）。 */
  readonly maxFrames: number;
  readonly defaults: Sbv2Defaults;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const assertAllowedKeys = (
  value: Record<string, unknown>,
  allowed: readonly string[],
  where: string,
): void => {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) {
      throw new Error(`${where}: 未知キー '${key}'（許可: ${allowed.join(" / ")}）`);
    }
  }
};

const readRecord = (raw: unknown, where: string): Record<string, unknown> => {
  if (!isRecord(raw)) throw new Error(`${where}: 無い / オブジェクトでない`);
  return raw;
};

const readNumber = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
  check: (value: number) => boolean,
  requirement: string,
): number => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  if (typeof value !== "number" || !check(value)) {
    throw new Error(`${where}.${key}: ${requirement}（${String(value)}）`);
  }
  return value;
};

/** 確保サイズの比較に使う数なので、整数であるだけでなく**安全整数**であることまで見る。 */
const isPositiveSafeInteger = (value: number): boolean => Number.isSafeInteger(value) && value > 0;

const readString = (raw: Record<string, unknown>, key: string, where: string): string => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${where}.${key}: 非空の文字列でない（${String(value)}）`);
  }
  return value;
};

/**
 * 「名前 → 行番号」の表を読む。値が `0..件数-1` の順列であることまで見る。
 *
 * MUST: 順列検査を落とさない。行番号は表の物理行そのもので、重複や飛びがあっても
 * `style_vec` / `g` の shape は変わらない — 検査を抜けた瞬間から沈黙誤値になる。
 */
const parseRowIndex = (raw: unknown, where: string): ReadonlyMap<string, number> => {
  const record = readRecord(raw, where);
  const names = Object.keys(record);
  if (names.length === 0) throw new Error(`${where}: 空（1 件以上必要）`);
  const table = new Map<string, number>();
  const seen = new Set<number>();
  for (const name of names) {
    const row = record[name];
    if (typeof row !== "number" || !Number.isInteger(row) || row < 0 || row >= names.length) {
      throw new Error(
        `${where}['${name}']: 行番号が 0..${names.length - 1} の整数でない（${String(row)}）`,
      );
    }
    if (seen.has(row)) {
      throw new Error(
        `${where}: 行番号 ${row} が重複している（'${name}' — 表の行と名前が 1 対 1 でない）`,
      );
    }
    seen.add(row);
    table.set(name, row);
  }
  return table;
};

const parseDefaults = (
  raw: unknown,
  styles: ReadonlyMap<string, number>,
  speakers: ReadonlyMap<string, number>,
): Sbv2Defaults => {
  const where = "pipelineConfig.defaults";
  const record = readRecord(raw, where);
  assertAllowedKeys(record, DEFAULTS_KEYS, where);
  const speaker = readString(record, "speaker", where);
  const style = readString(record, "style", where);
  // 既定値も受理集合の内側でなければならない（配布時に外れていたら fromAssets で落ちる —
  // 「生成を 1 回走らせて初めて分かる」を作らない）。
  if (!speakers.has(speaker)) {
    throw new Error(
      `${where}.speaker: '${speaker}' が speakers に無い（利用可能: ${
        [...speakers.keys()].join(" / ")
      }）`,
    );
  }
  if (!styles.has(style)) {
    throw new Error(
      `${where}.style: '${style}' が styles に無い（利用可能: ${[...styles.keys()].join(" / ")}）`,
    );
  }
  return {
    speaker,
    style,
    styleWeight: readNumber(record, "styleWeight", where, Number.isFinite, "有限の数でない"),
    sdpRatio: readNumber(record, "sdpRatio", where, Number.isFinite, "有限の数でない"),
    noiseScale: readNumber(record, "noiseScale", where, Number.isFinite, "有限の数でない"),
    noiseScaleW: readNumber(record, "noiseScaleW", where, Number.isFinite, "有限の数でない"),
    // lengthScale は継続長に直接掛かる。0 以下だと総フレーム数が 0 になり「発話にならない」
    // 形で下流（durationsToFrames）が落ちる — 原因の遠い失敗にしない。
    lengthScale: readNumber(
      record,
      "lengthScale",
      where,
      (value) => Number.isFinite(value) && value > 0,
      "正の有限数でない",
    ),
  };
};

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査して読む。 */
export const parseSbv2PipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): Sbv2PipelineConfig => {
  assertAllowedKeys(raw, ROOT_KEYS, "pipelineConfig");
  const styles = parseRowIndex(
    Object.hasOwn(raw, "styles") ? raw["styles"] : undefined,
    "pipelineConfig.styles",
  );
  const speakers = parseRowIndex(
    Object.hasOwn(raw, "speakers") ? raw["speakers"] : undefined,
    "pipelineConfig.speakers",
  );
  const limit = "正の安全整数でない";
  return {
    styles,
    speakers,
    maxTokens: readNumber(raw, "maxTokens", "pipelineConfig", isPositiveSafeInteger, limit),
    maxFrames: readNumber(raw, "maxFrames", "pipelineConfig", isPositiveSafeInteger, limit),
    defaults: parseDefaults(
      Object.hasOwn(raw, "defaults") ? raw["defaults"] : undefined,
      styles,
      speakers,
    ),
  };
};
