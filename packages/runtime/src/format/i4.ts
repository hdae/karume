/**
 * i4（K 方向 group symmetric packed 4bit）格納の CPU 側展開（ADR 0069 の適格外スロット用 /
 * GPU 側 `unpack4xU8` + マスク／シフト + scale 乗算のホスト鏡像）。
 *
 * MUST: 展開はこの 1 箇所だけに置く（format/i8.ts と同文の理由 — ロード経路とテストが別々に
 * i4 を解釈すると、pack 順〈要素 2i = 下位 nibble / 2i+1 = 上位・格納値 u = q + 8〉の
 * 取り違えが沈黙する）。pack 順の正本はエクスポータ `karume/emit.py: pack_int4`。
 *
 * ビット一致の根拠: 復元値は `f32(q) · s` の **f32 丸め 1 回**（`q ∈ [−7,7]` は f32 で厳密・
 * i8.ts と同じ論証）。エクスポータの逆変換門（`dequant(unpack(pack(w)))` のビット一致 —
 * ADR 0069 決定 4 ③）も同じ式で張られているため、TS / Python / WGSL の 3 実装が同じ数を出す。
 */

/** 要素数（format 層は src/ops.ts に依存しない — i8.ts と同じ理由でこの 1 行だけ持つ）。 */
const numel = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/** 展開で契約が破れた（バイト長・group 形 scale・group 長の不一致）。 */
export class I4Error extends Error {
  override readonly name = "I4Error";
}

/**
 * i4 の量子化行の長さ = **先頭次元を行・残りを平坦化**した長さ（`numel / shape[0]`）。
 *
 * 適格 op の `channel_rows`（エクスポータ `karume/quantize.py`）はどれもこの形
 * （linear `[O,I]` → `I` / embedding `[V,D]` → `D` / conv1d `[O,Cin,K]` → `Cin·K`）で、
 * rank 2 の重みでは「最終次元」と一致する。
 */
const rowLength = (shape: readonly number[]): number => numel(shape) / shape[0];

/**
 * group scale の論理形 = **rank 非依存の rank 2 規則**
 * `[shape[0], (numel / shape[0]) / groupSize]`（ADR 0069 決定 3 / 波 J-5b の一般化）。
 *
 * rank 2 の重み（linear `[O,I]` / embedding `[V,D]`）では従来の「同 rank・最終次元だけ
 * group 数」と**同値**なので、既存資産の検査結果は 1 件も変わらない。conv1d `[O,Cin,K]` →
 * `[O, (Cin·K)/g]` が唯一の新しい形。
 *
 * MUST: 検査側（format/container.ts）と展開側（{@link decodeI4}）が**この 1 本を共有する**
 * — 2 箇所に規則を書くと、受理した形と展開が読む形が静かに食い違う。
 * NOTE: 割り切れない / 記号次元は非整数・NaN のまま返す（呼び出し側が「形が違う」として
 * fail loudly にする — ここで投げると検査側のエラー型が混ざる）。
 */
export const groupScaleShape = (
  shape: readonly (number | string)[],
  groupSize: number,
): readonly number[] => {
  const dims = shape.map(Number);
  return [dims[0], rowLength(dims) / groupSize];
};

/**
 * packed 4bit のバイト列を group 形 scale で f32 へ展開する。
 *
 * `scaleShape` は {@link groupScaleShape} の rank 2 形（ADR 0069 決定 3 — i8 の keepdim
 * broadcast 形とは受理集合が交わらない別物。取り違えはここで fail loudly にする — 黙って
 * broadcast 解釈すると group scale が 1 チャネル 1 値として配られる沈黙誤値になる）。
 *
 * MUST: nibble の並びは**平坦メモリ順**（`[O,Cin,K]` row-major = `[O, Cin·K]` 平坦と同一
 * バイト列）。行 = 先頭次元・group = 行内の連続 `groupSize` 要素で、GPU 側の平坦添字
 * （`arow·(k>>shift) + (ak0>>shift)`）と同じ式になる。
 */
export const decodeI4 = (
  bytes: Uint8Array<ArrayBuffer>,
  shape: readonly number[],
  scale: Float32Array<ArrayBuffer>,
  scaleShape: readonly number[],
  groupSize: number,
): Float32Array<ArrayBuffer> => {
  if (shape.length === 0) {
    throw new I4Error("i4 の重みに量子化軸が無い（rank 0）");
  }
  const count = numel(shape);
  if (count % 2 !== 0 || bytes.byteLength * 2 !== count) {
    throw new I4Error(
      `i4 ペイロード ${bytes.byteLength} バイトが shape [${shape.join(",")}] と違う` +
        `（numel / 2 = ${count / 2} バイトが要る）`,
    );
  }
  const width = rowLength(shape);
  if (!Number.isInteger(groupSize) || groupSize < 1 || width % groupSize !== 0) {
    throw new I4Error(`group_size ${groupSize} が量子化軸（行長）${width} を割り切らない`);
  }
  const groups = width / groupSize;
  const expected = groupScaleShape(shape, groupSize);
  if (
    scaleShape.length !== expected.length ||
    scaleShape.some((dim, axis) => dim !== expected[axis])
  ) {
    throw new I4Error(
      `scale [${scaleShape.join(",")}] が重み [${shape.join(",")}] の group 形 [${
        expected.join(",")
      }]（group_size=${groupSize}）でない`,
    );
  }
  if (scale.length !== numel(scaleShape)) {
    throw new I4Error(
      `scale の要素数 ${scale.length} が shape [${scaleShape.join(",")}] と合わない`,
    );
  }
  const out = new Float32Array(count);
  const rows = width === 0 ? 0 : count / width;
  let i = 0;
  for (let row = 0; row < rows; row += 1) {
    const scaleBase = row * groups;
    for (let group = 0; group < groups; group += 1) {
      const s = scale[scaleBase + group];
      for (let element = 0; element < groupSize; element += 1, i += 1) {
        const byte = bytes[i >> 1];
        const u = (i & 1) === 1 ? byte >> 4 : byte & 0x0f;
        // MUST: 積の丸めは 1 回だけ（i8.ts と同文 — GPU 側は f32 の乗算 1 回）。offset 8 は
        // 「zero_point 省略時の既定」（ADR 0069 決定 3 の予約 2）で、pack 側 emit.py と対。
        out[i] = Math.fround((u - 8) * s);
      }
    }
  }
  return out;
};
