/**
 * `rope_base.safetensors`（軸別 rope 素表）の読み取り。
 *
 * ## なぜ models 側に最小の safetensors 読みを持つのか
 *
 * 素表は IR コンテナではない**素の safetensors**（`transformer` の `extras.rope_base` —
 * ADR 0038 §2）なので `openModel` では開けず、`@karume/runtime` の公開面（ADR 0008 の
 * 薄い面）は `parseSafetensors` を出していない。したがってここは「F32・rank2 の表を数本
 * 引く」だけに限定した最小の読み手を持つ。汎用のリーダにはしない — 用途が広がったら
 * runtime 側の公開面に載せるかを設計判断する（現状はそこまでの需要が無い）。
 *
 * ## rope 表は「計算」ではなく「素表からの並べ替え」
 *
 * MUST: cos / sin を TS で計算し**ない**。`torch` の f32 三角関数は正しく丸めた値と 1 ulp
 * ずれることがあり（実測: 位置 × 周波数 8,192 通りのうち cos 472 件 / sin 231 件）、JS の
 * `Math.cos` は f64 で正しく丸まるので**必ず食い違う**。静的グラフには torch の値が焼かれて
 * いる以上、S 形とのビット同一は「軸ごとの素表を焼いて並べ替えるだけ」でしか成立しない。
 *
 * 素表は**解像度に依らない**（軸 × 位置の表なので、任意の H' / W' を組める）。行数は上流の
 * `seq = arange(max(max_size))` の長さ = モデル側の位置表の天井そのもの。
 */

/** 素表のテンソルキー（順序は t → h → w のブロック順）。 */
const ROPE_AXES = ["t", "h", "w"] as const;

const HEADER_LENGTH_BYTES = 8;
/** F32 のみ受ける（素表は cos / sin の実数表で、他の格納形は上流に存在しない）。 */
const F32_DTYPE = "F32";
const F32_BYTES = 4;

/** 軸ごとの cos / sin 素表（行 = 位置・列 = その軸のブロック幅）。 */
export type RopeBase = {
  /** 全軸で共通の行数（= モデル側の位置表の天井）。 */
  readonly rows: number;
  /** 軸ごとのブロック幅（`t + h + w` の 2 倍が head_dim）。 */
  readonly widths: readonly [number, number, number];
  /** `[t, h, w]` の順に並べた cos 素表。 */
  readonly cos: readonly Float32Array[];
  /** 同 sin。 */
  readonly sin: readonly Float32Array[];
};

type RawTable = {
  readonly data: Float32Array;
  readonly shape: readonly [number, number];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const asIndex = (value: unknown, where: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`rope 素表 ${where}: ${String(value)} が非負整数でない`);
  }
  return value;
};

/**
 * `[u64 LE ヘッダ長][ヘッダ JSON][データ節]` から F32・rank2 の表だけを引く。
 * data_offsets はデータ節先頭からの相対。
 */
const readF32Tables = (buffer: ArrayBuffer): ReadonlyMap<string, RawTable> => {
  if (buffer.byteLength < HEADER_LENGTH_BYTES) {
    throw new Error(`rope 素表: ファイルが短すぎる（${buffer.byteLength} バイト）`);
  }
  const rawHeaderLength = new DataView(buffer).getBigUint64(0, true);
  if (rawHeaderLength > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`rope 素表: ヘッダ長 ${rawHeaderLength} が安全整数を超える`);
  }
  const headerLength = Number(rawHeaderLength);
  const dataStart = HEADER_LENGTH_BYTES + headerLength;
  if (dataStart > buffer.byteLength) {
    throw new Error(`rope 素表: ヘッダ長 ${headerLength} がファイル長を超える`);
  }
  let header: unknown;
  try {
    header = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(
        new Uint8Array(buffer, HEADER_LENGTH_BYTES, headerLength),
      ),
    );
  } catch (cause) {
    throw new Error("rope 素表: ヘッダを JSON として読めない", { cause });
  }
  if (!isRecord(header)) throw new Error("rope 素表: ヘッダがオブジェクトでない");

  const tables = new Map<string, RawTable>();
  for (const name of Object.keys(header)) {
    if (name === "__metadata__") continue;
    const raw = header[name];
    if (!isRecord(raw)) throw new Error(`rope 素表 '${name}': 項目がオブジェクトでない`);
    if (raw["dtype"] !== F32_DTYPE) {
      throw new Error(`rope 素表 '${name}': 格納 dtype が ${String(raw["dtype"])}（F32 が必要）`);
    }
    const shapeRaw = raw["shape"];
    if (!Array.isArray(shapeRaw) || shapeRaw.length !== 2) {
      throw new Error(`rope 素表 '${name}': rank が 2 でない`);
    }
    const shape: readonly [number, number] = [
      asIndex(shapeRaw[0], `'${name}'.shape[0]`),
      asIndex(shapeRaw[1], `'${name}'.shape[1]`),
    ];
    const offsets = raw["data_offsets"];
    if (!Array.isArray(offsets) || offsets.length !== 2) {
      throw new Error(`rope 素表 '${name}': data_offsets が 2 要素の配列でない`);
    }
    const begin = asIndex(offsets[0], `'${name}'.data_offsets[0]`);
    const end = asIndex(offsets[1], `'${name}'.data_offsets[1]`);
    const expected = shape[0] * shape[1] * F32_BYTES;
    if (end - begin !== expected) {
      throw new Error(
        `rope 素表 '${name}': サイズ不一致 offsets=${end - begin} 期待=${expected}`,
      );
    }
    if (dataStart + end > buffer.byteLength) {
      throw new Error(`rope 素表 '${name}': データ節の範囲外`);
    }
    if ((dataStart + begin) % F32_BYTES !== 0) {
      // コピーを作らず typed array view を張る前提が崩れるため受理しない。
      throw new Error(`rope 素表 '${name}': 絶対 offset が 4 バイト境界に整列していない`);
    }
    tables.set(name, {
      data: new Float32Array(buffer, dataStart + begin, shape[0] * shape[1]),
      shape,
    });
  }
  return tables;
};

const tableOf = (tables: ReadonlyMap<string, RawTable>, name: string): RawTable => {
  const table = tables.get(name);
  if (table === undefined) throw new Error(`rope 素表に '${name}' が無い`);
  return table;
};

/**
 * `rope_base.safetensors` を読む。
 *
 * MUST: 行数が全軸で揃っていることを見る。揃っていないと「h の行を w の表から読む」形の
 * 取り違えが範囲内に収まって黙って通る。
 */
export const parseRopeBase = (buffer: ArrayBuffer): RopeBase => {
  const tables = readF32Tables(buffer);
  const cos: Float32Array[] = [];
  const sin: Float32Array[] = [];
  const widths: number[] = [];
  let rows: number | undefined;
  for (const axis of ROPE_AXES) {
    const cosTable = tableOf(tables, `cos_${axis}`);
    const sinTable = tableOf(tables, `sin_${axis}`);
    if (cosTable.shape[0] !== sinTable.shape[0] || cosTable.shape[1] !== sinTable.shape[1]) {
      throw new Error(`rope 素表 ${axis} の cos / sin で shape が違う`);
    }
    if (rows !== undefined && cosTable.shape[0] !== rows) {
      throw new Error(`rope 素表 ${axis} の行数 ${cosTable.shape[0]} が他軸の ${rows} と違う`);
    }
    rows = cosTable.shape[0];
    widths.push(cosTable.shape[1]);
    cos.push(cosTable.data);
    sin.push(sinTable.data);
  }
  return { rows: rows ?? 0, widths: [widths[0], widths[1], widths[2]], cos, sin };
};

/** rope 表 1 行の幅（= attention の head_dim）。 */
export const ropeWidth = (base: RopeBase): number =>
  2 * (base.widths[0] + base.widths[1] + base.widths[2]);

/** 軸の並び（`ropeTables` が読み出し位置を組むのに使う）。 */
export const ROPE_AXIS_COUNT = ROPE_AXES.length;
