/**
 * VAE decode の**固定タイル化**（ホストグルーのみ — ランタイムは 1 行も関わらない）。ADR 0033。
 *
 * ## なぜホストだけで足りるのか
 *
 * VAE decoder のグラフは**解像度に対して構造不変**である（実測: 512px 用と 1024px 用の
 * 重みペイロードは sha256 同一で、違うのは中間値の宣言 shape だけ — ADR 0038 §4 の実測 2）。
 * したがってタイル decoder（入力 `[1,16,64,64]`）を敷き詰めれば全解像度をカバーできる。
 * 要るのはタイルの切り出し・ブレンド・貼り付けだけで、それがこのモジュール。
 *
 * ## タイル幾何 = 等間隔スナップ配置（diffusers からの意図的な逸脱）
 *
 * 上流（`autoencoder_kl_qwenimage.py` の `tiled_decode`）は `range(0, H, stride)` で走査する
 * ので**最後のタイルが短くなる**。こちらのタイル decoder は**形が固定**なので短いタイルを
 * 食えない。そこで走査を「最後のタイルの開始位置を `extent − tile` にスナップする」形へ
 * 変える。開始位置を等間隔に取れば `(extent − tile) % stride == 0` が要求されるが、これは
 * **タイル本数から stride を決める**ことで常に満たせる（{@link planTileAxis}）。
 *
 * 帰結として、上流の「各タイルを stride 幅へ切り詰めて連結 → 全体 crop」はそのままでは
 * **画像の末端に届かない**（`n·stride = extent − 重なり < extent`）。等間隔スナップでの
 * 正しい対応物は「タイル i の担当領域 = `[starts[i], starts[i+1])`、最後だけ `[starts[n−1],
 * extent)`」という**領域割り当て**で、これは上流の crop + 全体 crop と同じ分割を与えつつ
 * 末端まで覆う（{@link assembleTiles}）。
 *
 * ## ブレンドは上流の線形ランプと同型
 *
 * `blend_v` / `blend_h` と同じ式・同じ順序（**上のタイルとの縦ブレンドが先・左との横ブレンドが
 * 後**）・同じ **sample 空間**で行う。上流はタイル配列を in-place に書き換えるので、隣に効くのは
 * **ブレンド済みのタイル**である — ここでも同じ。in-place で進める本体は 1 本だけ持ち、
 * {@link assembleTiles}（公開面）は写しを取ってからそれを呼び、{@link decodeTiled}（配列を
 * 自分で所有する内部経路）は写さずに直接呼ぶ。
 *
 * ## 縮退（タイル 1 枚）は非タイル経路と**ビット同一** MUST
 *
 * `extent === tile`（512px = latent 64）ではタイルが 1 枚になり、ブレンドは 1 度も走らず
 * 担当領域が画像全体になる。この経路が非タイル decode と 1 ビットでも違ったら、それは幾何か
 * 貼り付けの誤り — 数値ではなく**構造で**恒等性を固定するための縮退点で、実 GPU の E2E が
 * Uint32 完全一致で門にする。
 */

const f32 = Math.fround;

/**
 * 隣り合うタイルが latent で重なる最小幅（= sample 64px）。
 *
 * 上流の既定ブレンド幅（`tile_sample_min − tile_sample_stride` = 256−192 = 64px）と同じ
 * オーダーに合わせてある。decoder は 3×3 conv 12 本 + upsample 3 段で sample 空間の実効
 * 受容野が数十 px なので、これを下回ると継ぎ目がランプで隠しきれなくなる。
 */
export const MIN_TILE_OVERLAP_LATENT = 8;

/** latent の 1 軸ぶんのタイル配置。 */
export type TileAxis = {
  /** この軸の latent 全長。 */
  readonly extent: number;
  /** タイル 1 枚の latent 幅（= タイル decoder の入力形）。 */
  readonly tile: number;
  /** 隣り合うタイルの開始位置の差（latent）。縮退時は `tile` と等しい。 */
  readonly stride: number;
  /** 各タイルの開始位置（latent・昇順・末尾はちょうど `extent − tile`）。 */
  readonly starts: readonly number[];
};

/** 2 軸ぶんのタイル配置と、latent ↔ sample の縮尺。 */
export type TileGeometry = {
  /** latent 1 あたりの sample 画素数。 */
  readonly scale: number;
  /** latent のチャネル数（切り出しの平面幅を決める）。 */
  readonly channels: number;
  /** 高さ軸。 */
  readonly rows: TileAxis;
  /** 幅軸。 */
  readonly cols: TileAxis;
};

/**
 * 1 軸ぶんの等間隔スナップ配置を決める。
 *
 * 「重なり `tile − stride` が `minOverlap` 以上」を満たす**最小のタイル本数**を選び、
 * その本数から stride を割り出す（`stride = (extent − tile) / (本数 − 1)`）。本数から
 * 決めるので `(extent − tile) % stride == 0` は構成上つねに成立する。
 *
 * MUST: stride を先に決めて本数を割り出す形に**しない** — 割り切れない stride を選ぶと
 * 最後のタイルだけ重なりが変わり、ブレンド幅がタイル位置ごとに変わる（上流の
 * `blend_extent = min(a, b, blend_extent)` は短いタイルの安全弁で、幾何そのものを
 * 揃えるものではない）。
 */
export const planTileAxis = (extent: number, tile: number, minOverlap: number): TileAxis => {
  if (!Number.isInteger(extent) || !Number.isInteger(tile) || !Number.isInteger(minOverlap)) {
    throw new Error(`タイル配置は整数で組む（extent=${extent} tile=${tile} 重なり=${minOverlap}）`);
  }
  if (tile < 1) throw new Error(`タイル幅 ${tile} が 1 未満`);
  if (extent < tile) {
    throw new Error(
      `latent 全長 ${extent} がタイル幅 ${tile} より小さい（固定形の decoder に食わせられない）`,
    );
  }
  if (minOverlap < 0 || minOverlap >= tile) {
    throw new Error(`最小の重なり ${minOverlap} が [0, ${tile}) の外（重なりはタイル幅未満）`);
  }
  const span = extent - tile;
  if (span === 0) {
    // 縮退: 1 枚で覆える。stride を tile と等しくしておくと重なり 0 → ブレンド幅 0 になり、
    // 貼り付けが素の写しになる（非タイル経路とビット同一 — モジュール doc の MUST）。
    return { extent, tile, stride: tile, starts: [0] };
  }
  for (let count = 2; count <= span + 1; count += 1) {
    if (span % (count - 1) !== 0) continue;
    const stride = span / (count - 1);
    if (tile - stride < minOverlap) continue;
    const starts: number[] = [];
    for (let index = 0; index < count; index += 1) starts.push(index * stride);
    return { extent, tile, stride, starts };
  }
  // `count − 1 = span` なら stride 1・重なり `tile − 1 ≥ minOverlap`（入口で minOverlap <
  // tile を検査済み）なので到達しない。到達したら上の不変条件が壊れている。
  throw new Error(`latent ${extent} をタイル ${tile}（最小の重なり ${minOverlap}）で覆えない`);
};

/** ブレンドの幅（**sample 空間** — 上流の `blend_height` / `blend_width` と同じ単位）。 */
export const blendExtent = (axis: TileAxis, scale: number): number =>
  (axis.tile - axis.stride) * scale;

/**
 * latent の形とタイル decoder の入出力の形から幾何を組む。
 *
 * `latentShape` は `[1,C,H,W]`（DiT が吐く latent）、`tileShape` はタイル decoder の
 * **グラフ入力** `[1,C,th,tw]`、`sampleShape` はその**グラフ出力** `[1,3,th·s,tw·s]`。
 * 縮尺 `s` を出力形から割り出すのは、呼び出し側に `8` を literal で置かないため。
 */
export const planVaeTiling = (
  latentShape: readonly number[],
  tileShape: readonly number[],
  sampleShape: readonly number[],
  minOverlap: number = MIN_TILE_OVERLAP_LATENT,
): TileGeometry => {
  if (latentShape.length !== 4 || tileShape.length !== 4 || sampleShape.length !== 4) {
    throw new Error(
      `タイル化は rank4 前提（latent [${latentShape}] / タイル入力 [${tileShape}] / タイル出力 [${sampleShape}]）`,
    );
  }
  if (latentShape[0] !== 1 || tileShape[0] !== 1) {
    throw new Error(
      `タイル化は batch=1 前提（latent B=${latentShape[0]} / タイル B=${tileShape[0]}）`,
    );
  }
  if (latentShape[1] !== tileShape[1]) {
    throw new Error(
      `latent のチャネル数 ${latentShape[1]} がタイル decoder の入力 ${tileShape[1]} と違う`,
    );
  }
  const scale = sampleShape[2] / tileShape[2];
  if (!Number.isInteger(scale) || scale < 1 || sampleShape[3] !== tileShape[3] * scale) {
    throw new Error(
      `タイル decoder の縮尺が軸で揃わない（入力 ${tileShape[2]}×${tileShape[3]} → 出力 ${
        sampleShape[2]
      }×${sampleShape[3]}）`,
    );
  }
  return {
    scale,
    channels: latentShape[1],
    rows: planTileAxis(latentShape[2], tileShape[2], minOverlap),
    cols: planTileAxis(latentShape[3], tileShape[3], minOverlap),
  };
};

/** タイルの総枚数。 */
export const tileCount = (geometry: TileGeometry): number =>
  geometry.rows.starts.length * geometry.cols.starts.length;

/**
 * latent `[1,C,H,W]` から `(row, col)` のタイル `[1,C,th,tw]` を切り出す（平面ごとのコピー）。
 */
export const latentTile = (
  latents: Float32Array,
  geometry: TileGeometry,
  row: number,
  col: number,
): Float32Array<ArrayBuffer> => {
  const { rows, cols, channels } = geometry;
  if (row < 0 || row >= rows.starts.length || col < 0 || col >= cols.starts.length) {
    throw new RangeError(
      `タイル (${row}, ${col}) が範囲外（${rows.starts.length}×${cols.starts.length}）`,
    );
  }
  const sourcePlane = rows.extent * cols.extent;
  if (latents.length !== channels * sourcePlane) {
    throw new Error(
      `latent の要素数 ${latents.length} が [1,${channels},${rows.extent},${cols.extent}] と違う`,
    );
  }
  const top = rows.starts[row];
  const left = cols.starts[col];
  const tilePlane = rows.tile * cols.tile;
  const out = new Float32Array(channels * tilePlane);
  for (let channel = 0; channel < channels; channel += 1) {
    for (let y = 0; y < rows.tile; y += 1) {
      const from = channel * sourcePlane + (top + y) * cols.extent + left;
      out.set(latents.subarray(from, from + cols.tile), channel * tilePlane + y * cols.tile);
    }
  }
  return out;
};

/**
 * 上のタイルとの線形ランプ合成（上流 `blend_v` と同型・**in-place**）。
 *
 * `b[y] = a[H−blend+y]·(1 − y/blend) + b[y]·(y/blend)`。MUST: 重み 2 本を先に f32 へ丸めて
 * から掛ける（torch は Python float のスカラをテンソル dtype へ落として演算する）— 丸めの
 * 位置を動かすと参照と最終桁で割れる。
 */
const blendVertical = (
  above: Float32Array,
  current: Float32Array,
  planes: number,
  height: number,
  width: number,
  blend: number,
): void => {
  for (let y = 0; y < blend; y += 1) {
    const weightAbove = f32(1 - y / blend);
    const weightCurrent = f32(y / blend);
    for (let plane = 0; plane < planes; plane += 1) {
      const base = plane * height * width;
      const from = base + (height - blend + y) * width;
      const to = base + y * width;
      for (let x = 0; x < width; x += 1) {
        current[to + x] = f32(
          f32(above[from + x] * weightAbove) + f32(current[to + x] * weightCurrent),
        );
      }
    }
  }
};

/** 左のタイルとの線形ランプ合成（上流 `blend_h` と同型・**in-place**）。 */
const blendHorizontal = (
  left: Float32Array,
  current: Float32Array,
  planes: number,
  height: number,
  width: number,
  blend: number,
): void => {
  for (let x = 0; x < blend; x += 1) {
    const weightLeft = f32(1 - x / blend);
    const weightCurrent = f32(x / blend);
    for (let plane = 0; plane < planes; plane += 1) {
      const base = plane * height * width;
      for (let y = 0; y < height; y += 1) {
        const from = base + y * width + (width - blend + x);
        const to = base + y * width + x;
        current[to] = f32(f32(left[from] * weightLeft) + f32(current[to] * weightCurrent));
      }
    }
  }
};

/**
 * 貼り合わせの本体（**渡された配列の上で in-place にブレンドする** — 呼び出し側がその配列を
 * 所有していることが前提）。形と枚数の検査もここに置くので、写しを取る公開面
 * {@link assembleTiles} と、配列を所有する内部経路 {@link decodeTiled} が同じ検査を通る。
 */
const assembleOwnedTiles = (
  working: Float32Array[],
  geometry: TileGeometry,
): Float32Array<ArrayBuffer> => {
  const { rows, cols, scale } = geometry;
  const rowCount = rows.starts.length;
  const colCount = cols.starts.length;
  if (working.length !== rowCount * colCount) {
    throw new Error(`タイル ${working.length} 枚が幾何の ${rowCount}×${colCount} と違う`);
  }
  const tileHeight = rows.tile * scale;
  const tileWidth = cols.tile * scale;
  const tilePlane = tileHeight * tileWidth;
  const channels = working[0].length / tilePlane;
  if (!Number.isInteger(channels) || channels < 1) {
    throw new Error(
      `decode 出力の要素数 ${working[0].length} が ${tileHeight}×${tileWidth} の平面で割り切れない`,
    );
  }
  for (const [index, tile] of working.entries()) {
    if (tile.length !== channels * tilePlane) {
      throw new Error(
        `タイル ${index} の要素数 ${tile.length} が 1 枚目の ${channels * tilePlane} と違う`,
      );
    }
  }

  const blendRows = blendExtent(rows, scale);
  const blendCols = blendExtent(cols, scale);
  // MUST: 縦（上）→ 横（左）の順（上流 `tiled_decode` と同じ）。順序を入れ替えると角の
  // 4 枚が重なる領域で係数の積の順が変わり、継ぎ目に十字の筋が出る。
  for (let row = 0; row < rowCount; row += 1) {
    for (let col = 0; col < colCount; col += 1) {
      const current = working[row * colCount + col];
      if (row > 0) {
        blendVertical(
          working[(row - 1) * colCount + col],
          current,
          channels,
          tileHeight,
          tileWidth,
          blendRows,
        );
      }
      if (col > 0) {
        blendHorizontal(
          working[row * colCount + col - 1],
          current,
          channels,
          tileHeight,
          tileWidth,
          blendCols,
        );
      }
    }
  }

  const height = rows.extent * scale;
  const width = cols.extent * scale;
  const image = new Float32Array(channels * height * width);
  for (let row = 0; row < rowCount; row += 1) {
    // 担当領域 = 次のタイルの開始まで（最後だけ末端まで）。上流の「stride 幅へ切り詰め +
    // 全体 crop」の等間隔スナップ版で、末端まで覆う（モジュール doc）。
    //
    // MUST: 幅は担当領域ちょうどにする。**過大**（例: 常にタイル全幅）にしても後続タイルが
    // 上書きするので結果は 1 ビットも変わらず、数値網では検出できない（実測: 故障注入が
    // 緑のまま通る）。この「無害さ」は行優先の走査順に依存しているので、貼り付けを並列化
    // または逆順化した瞬間に沈黙誤値へ変わる。**過小**な側（常に stride 幅）は末端が欠ける
    // ので検出できる。
    const top = rows.starts[row] * scale;
    const spanRows =
      ((row + 1 < rowCount ? rows.starts[row + 1] : rows.extent) - rows.starts[row]) * scale;
    for (let col = 0; col < colCount; col += 1) {
      const left = cols.starts[col] * scale;
      const spanCols =
        ((col + 1 < colCount ? cols.starts[col + 1] : cols.extent) - cols.starts[col]) * scale;
      const tile = working[row * colCount + col];
      for (let channel = 0; channel < channels; channel += 1) {
        for (let y = 0; y < spanRows; y += 1) {
          const from = channel * tilePlane + y * tileWidth;
          image.set(
            tile.subarray(from, from + spanCols),
            channel * height * width + (top + y) * width + left,
          );
        }
      }
    }
  }
  return image;
};

/**
 * decode 済みタイル（行優先）をブレンドして 1 枚の画像 `[1,C,H·s,W·s]` に貼り合わせる。
 *
 * MUST: 渡された配列を破壊しない。ブレンドは in-place で進むので、この公開面は写しを取って
 * から本体へ渡す — 呼び出し側が持っている Tensor の中身が黙って変わる形の罠を踏まないため。
 * 写しは**この公開契約のためだけ**に在り、配列を自分で所有する内部経路（{@link decodeTiled}）
 * は写さずに本体を直接呼ぶ。
 */
export const assembleTiles = (
  decoded: readonly Float32Array[],
  geometry: TileGeometry,
): Float32Array<ArrayBuffer> =>
  assembleOwnedTiles(decoded.map((tile) => Float32Array.from(tile)), geometry);

/**
 * タイル decode の駆動（唯一の非純粋な継ぎ目 = `decode` コールバック）。
 *
 * MUST: `decode` は **1 本の session を使い回す**呼び出しにする（タイルごとに開き直すと
 * 重みのロードが枚数ぶん走り、タイル化の狙い〈メモリのピークを 512px 相当に抑える〉に対して
 * 時間だけが跳ね上がる）。
 *
 * MUST: `decode` が返した配列は**所有権ごと渡す** — 呼び出し側は保持も再利用もしてはならない
 * （ここでそのまま in-place にブレンドする）。写しを 1 枚省くぶんがそのままピークの節約で、
 * 全タイルぶん（1024px で 27MiB / 2048px で 75MiB — docs/perf-ledger.md H-6）に効く。
 * `Session.run` の出力は readback ごとに新しく確保される（`executor.ts` が
 * `getMappedRange().slice(0)` を `hostTensor` へ渡す）ので、anima パイプラインのコールバックは
 * そのままこの契約を満たしている。
 */
export const decodeTiled = async (
  latents: Float32Array,
  geometry: TileGeometry,
  decode: (tile: Float32Array<ArrayBuffer>, row: number, col: number) => Promise<Float32Array>,
): Promise<Float32Array<ArrayBuffer>> => {
  const decoded: Float32Array[] = [];
  for (let row = 0; row < geometry.rows.starts.length; row += 1) {
    for (let col = 0; col < geometry.cols.starts.length; col += 1) {
      decoded.push(await decode(latentTile(latents, geometry, row, col), row, col));
    }
  }
  return assembleOwnedTiles(decoded, geometry);
};
