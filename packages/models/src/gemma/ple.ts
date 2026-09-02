/**
 * Gemma 4 の PLE（per-layer embeddings）を**ホスト側で gather** する実装（ADR 0085）。
 *
 * ## なぜホストが引くのか
 *
 * PLE は `input_ids` **だけ**を引数に取る純粋な行 lookup で、E2B では i8 35 表 = 2,240MiB。
 * ランタイムは全 initializer に Session 構築時の GPU 常駐席を与える契約なので、グラフに
 * 残す限り lazy にならない（ADR 0085 Context）。グラフから外して `per_layer_inputs[1,M,35,256]`
 * を**通常のグラフ入力**として供給すると、常駐が 3.70 → 1.51 GiB になる。ランタイムの契約は
 * 1 文字も変わらない（ADR 0085 決定 6 — 「pageable initializer」の席は作らない）。
 *
 * 前例は SBV2 の相対位置表（`sbv2/relattn-tables.ts`）と Anima の rope 素表 — どちらも
 * 「モデル固有の入力の作り方」は models 側の知識で、runtime の語彙ではない（ADR 0008）。
 *
 * ## 配布形（sidecar）
 *
 * 書き手の正本は `tools/export-recipes/gemma4/export_product.py`:
 *
 * - `ple.json` — 索引（token 総数 / 層数 / 層当たり次元 / embed scale / shard の token 範囲）
 * - `ple-NNNNN-of-NNNNN.safetensors` — **token-major**。`values` `[rows, layers, dim]` i8 と
 *   `scales` `[rows, layers]` f32 で、1 token の PLE が**連続 1 読み**になる（ADR 0085 決定 1）。
 *   vocab の範囲で shard し、上限は書き手の容量（ADR 0090 — 256MiB − ヘッダ余裕 = 1 回の読みの上限）。
 *
 * ## MUST: 逆量子化は GPU 側 `embedding` とビット一致する
 *
 * グラフに残していれば `embedding` が `f32(i8) × per-row scale` を計算し、その直後の `mul` が
 * `embed_scale`（= `hidden_size_per_layer_input ** 0.5` = 16.0）を掛けていた。ここも**同じ順序**
 * （`(q × scale) × embedScale`）で組む。`embedScale` は 2 冪なので f32 の乗算が厳密で、順序さえ
 * 揃えば一致する（ADR 0085 決定 4）。割れると token 列 parity が割れ、「機能不変であること」の
 * 証明が使えなくなる。門は `packages/models/tests/e2e_gemma4_product_test.ts` の
 * `ple.probe.safetensors` 突合（torch が 35 表経路で計算した値との**厳密一致**）。
 *
 * ## MUST: id 空間を相互照合する（ADR 0085 決定 5）
 *
 * sidecar の行数 / 主 embedding の vocab 行数 / 実際に引く id を突き合わせる。ここがずれると
 * **OOB ではなく「別 token の有効な行」**を引く（例外なしで沈黙して壊れる）ので、fail loudly の
 * 門を置く場所はここしかない。
 */

import { parseSafetensors, type SafetensorsFile, type Tensor } from "@karume/runtime";

/** sidecar shard 1 本の受け持つ token 範囲（`[start, stop)`）。 */
export type Gemma4PleShard = {
  /** 配布形の相対ファイル名（読み手が {@link Gemma4PleOptions.readShard} へ渡す綴り）。 */
  readonly file: string;
  readonly start: number;
  readonly stop: number;
};

/** `ple.json` の受理形（書き手の正本は `gemma4/export_product.py`）。 */
export type Gemma4PleIndex = {
  /** sidecar が持つ token 行数（= `vocab_size_per_layer_input`）。 */
  readonly tokens: number;
  /** 層数（E2B は 35）。 */
  readonly layers: number;
  /** 層当たりの次元（E2B は 256）。 */
  readonly dim: number;
  /** lookup 後に掛かる embed scale（`hidden_size_per_layer_input ** 0.5`）。 */
  readonly embedScale: number;
  /** token 範囲の昇順・隙間なしの分割（先頭は 0・末尾は `tokens`）。 */
  readonly shards: readonly Gemma4PleShard[];
};

/**
 * shard 読みへ透過するノブ（{@link Gemma4PleOptions.readShard} と {@link Gemma4Ple.gather}）。
 *
 * MUST: **best-effort** の契約である — 読み口が無視しても壊れない（無視した実装では中断が
 * 「この shard を読み終わってから」効くだけで、値も寿命も変わらない）。生成側は run の発行前に
 * 自分で `signal` を見る（`generation/sequence.ts`）ので、中断の正しさをここへ委ねていない。
 */
export type Gemma4PleReadOptions = {
  /** この読みの中断（生成 1 回ぶんの `signal` がそのまま降りてくる）。 */
  readonly signal?: AbortSignal;
};

export type Gemma4PleOptions = {
  readonly index: Gemma4PleIndex;
  /** shard 1 本ぶんのバイト列を取る（ファイル読み / hub の `streamAssets` — 呼び手の責務）。 */
  readonly readShard: (file: string, options?: Gemma4PleReadOptions) => Promise<ArrayBuffer>;
  /**
   * 主 embedding の vocab 行数（id 空間の相互照合 — ADR 0085 決定 5）。
   *
   * MUST: 省略可能にしない。sidecar と主 embedding が別の語彙で焼かれた組み合わせは、
   * shape も dtype も合ったまま**別 token の行**を引く。
   */
  readonly vocabSize: number;
  /** 常駐させる shard 数の上限（LRU — ADR 0085 決定 3。既定 2）。 */
  readonly residentShards?: number;
};

/** 遅延ロードの実測（門が「触った shard だけ読んだ」を恒真でなく見るための欄）。 */
export type Gemma4PleStats = {
  /** shard を実際に取りに行った回数（キャッシュミスの数）。 */
  readonly loads: number;
  /** 現在常駐している shard 数。 */
  readonly resident: number;
};

export type Gemma4Ple = {
  /**
   * token id 列 → `per_layer_inputs` の `[1, ids.length, layers, dim]` f32。
   *
   * prefill の pad 行も**そのまま id を渡す**（ホストは `input_ids` の pad を 0 で埋めるので、
   * ここにも 0 行の PLE が入る）— グラフ内で引いていたときと同じ値になり、pad 行の値契約
   * （ADR 0066 追記 6）が保たれる。
   */
  gather(ids: readonly number[], options?: Gemma4PleReadOptions): Promise<Tensor>;
  stats(): Gemma4PleStats;
  /**
   * 常駐 shard を解放する（ホスト RAM で shard 1 本 758MB 級 × 既定 2 本）。
   *
   * MUST: 解放口を持つ。GPU 側の常駐は `Session.dispose` が返すが、ここは**ホスト RAM の
   * キャッシュ**なので、口が無いと「パイプラインを dispose しても 1.5GiB が返らない」
   * （実体を掴む参照を 1 つ残せばプロセス寿命まで残る）。
   *
   * 以後の {@link Gemma4Ple.gather} は fail loudly — 解放済みの実体が黙って読み直しを始めると、
   * 「dispose したのに RAM が戻らない」形が復活する。冪等。
   */
  dispose(): void;
};

/** sidecar のテンソルキーと索引のメタデータキー（綴りの正本は `gemma4/export_product.py`）。 */
const VALUES_KEY = "values";
const SCALES_KEY = "scales";
const METADATA_KEY = "karume_ple";

/** 索引と shard メタデータの版（知らない版を黙って読まない）。 */
const SCHEMA = 1;

const INDEX_KEYS: readonly string[] = ["schema", "tokens", "layers", "dim", "embedScale", "shards"];
const SHARD_KEYS: readonly string[] = ["file", "start", "stop"];

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

const readCount = (raw: Record<string, unknown>, key: string, where: string): number => {
  const value = Object.hasOwn(raw, key) ? raw[key] : undefined;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${where}.${key} ${String(value)} が 1 以上の整数でない`);
  }
  return value;
};

const readOffset = (raw: Record<string, unknown>, key: string, where: string): number => {
  const value = Object.hasOwn(raw, key) ? raw[key] : undefined;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${where}.${key} ${String(value)} が 0 以上の整数でない`);
  }
  return value;
};

/**
 * `ple.json` を受理形へ落とす（未知キー・欠け・不連続な範囲は fail loudly）。
 *
 * MUST: shard の範囲は `[0, tokens)` の**隙間も重なりも無い昇順分割**であること。緩めると
 * 「引けない id がある索引」や「2 本が同じ id を持つ索引」が通り、後者は**どちらの行を
 * 引いたか**で結果が変わる（沈黙誤値）。
 */
export const parseGemma4PleIndex = (raw: unknown, where = "ple.json"): Gemma4PleIndex => {
  const root = readRecord(raw, where);
  assertAllowedKeys(root, INDEX_KEYS, where);
  if (root.schema !== SCHEMA) {
    throw new Error(`${where}.schema ${String(root.schema)} が ${SCHEMA} でない`);
  }
  const tokens = readCount(root, "tokens", where);
  const layers = readCount(root, "layers", where);
  const dim = readCount(root, "dim", where);
  const embedScale = root.embedScale;
  if (typeof embedScale !== "number" || !Number.isFinite(embedScale) || embedScale <= 0) {
    throw new Error(`${where}.embedScale ${String(embedScale)} が正の有限数でない`);
  }
  if (!Array.isArray(root.shards) || root.shards.length === 0) {
    throw new Error(`${where}.shards が非空の配列でない`);
  }
  const shards: Gemma4PleShard[] = [];
  const files = new Set<string>();
  let expected = 0;
  root.shards.forEach((entry, position) => {
    const at = `${where}.shards[${position}]`;
    const shard = readRecord(entry, at);
    assertAllowedKeys(shard, SHARD_KEYS, at);
    const file = shard.file;
    if (typeof file !== "string" || file === "") throw new Error(`${at}.file が非空の文字列でない`);
    if (files.has(file)) throw new Error(`${at}.file '${file}' が重複している`);
    files.add(file);
    const start = readOffset(shard, "start", at);
    const stop = readOffset(shard, "stop", at);
    if (start !== expected) {
      throw new Error(`${at}.start ${start} が直前の shard の末尾 ${expected} と連続しない`);
    }
    if (stop <= start) throw new Error(`${at}: 範囲 [${start}, ${stop}) が空`);
    expected = stop;
    shards.push({ file, start, stop });
  });
  if (expected !== tokens) {
    throw new Error(`${where}: shard の合計 ${expected} 行が tokens ${tokens} と違う`);
  }
  return { tokens, layers, dim, embedScale, shards };
};

/** 読み込み済みの shard 1 本（i8 値と per-row scale の**生の並び**）。 */
type ResidentShard = {
  readonly start: number;
  readonly values: Int8Array<ArrayBuffer>;
  readonly scales: Float32Array<ArrayBuffer>;
};

const tensorView = (file: SafetensorsFile, name: string, where: string) => {
  const view = file.tensors.get(name);
  if (view === undefined) throw new Error(`${where}: テンソル '${name}' が無い`);
  return view;
};

const assertShape = (
  actual: readonly number[],
  expected: readonly number[],
  where: string,
): void => {
  if (actual.length !== expected.length || actual.some((dim, axis) => dim !== expected[axis])) {
    throw new Error(`${where}: shape [${actual.join(",")}] が [${expected.join(",")}] でない`);
  }
};

/**
 * shard のメタデータが索引と同じ資産世代を名乗っていることを見る。
 *
 * MUST: 範囲まで突き合わせる — 索引だけ差し替えた組み合わせは**形も dtype も合う**まま
 * 別 token の行を引く（ADR 0085 決定 5 の沈黙誤値そのもの）。
 */
const assertShardMetadata = (
  file: SafetensorsFile,
  index: Gemma4PleIndex,
  shard: Gemma4PleShard,
): void => {
  const raw = file.metadata.get(METADATA_KEY);
  if (raw === undefined) {
    throw new Error(`${shard.file}: __metadata__.${METADATA_KEY} が無い（別形式の資産）`);
  }
  const declared = readRecord(JSON.parse(raw), `${shard.file} の ${METADATA_KEY}`);
  const mismatches = (
    [
      ["schema", SCHEMA],
      ["tokens", index.tokens],
      ["layers", index.layers],
      ["dim", index.dim],
      ["embedScale", index.embedScale],
      ["start", shard.start],
      ["stop", shard.stop],
    ] as const
  ).filter(([key, want]) => (Object.hasOwn(declared, key) ? declared[key] : undefined) !== want);
  if (mismatches.length > 0) {
    throw new Error(
      `${shard.file}: ${METADATA_KEY} が索引と食い違う（` +
        mismatches
          .map(([key, want]) =>
            `${key} ${String(Object.hasOwn(declared, key) ? declared[key] : undefined)} ≠ ${want}`
          )
          .join(" / ") +
        `）— 片方だけ作り直した組み合わせ`,
    );
  }
};

const readResidentShard = (
  bytes: ArrayBuffer,
  index: Gemma4PleIndex,
  shard: Gemma4PleShard,
): ResidentShard => {
  const file = parseSafetensors(bytes);
  assertShardMetadata(file, index, shard);
  const rows = shard.stop - shard.start;
  const values = tensorView(file, VALUES_KEY, shard.file);
  if (values.dtype !== "I8") {
    throw new Error(`${shard.file}: '${VALUES_KEY}' の格納 dtype が ${values.dtype}（I8 でない）`);
  }
  assertShape(values.shape, [rows, index.layers, index.dim], `${shard.file} の '${VALUES_KEY}'`);
  const scales = tensorView(file, SCALES_KEY, shard.file);
  if (scales.dtype !== "F32") {
    throw new Error(`${shard.file}: '${SCALES_KEY}' の格納 dtype が ${scales.dtype}（F32 でない）`);
  }
  assertShape(scales.shape, [rows, index.layers], `${shard.file} の '${SCALES_KEY}'`);
  return {
    start: shard.start,
    values: new Int8Array(file.buffer, values.byteOffset, values.byteLength),
    scales: new Float32Array(file.buffer, scales.byteOffset, scales.byteLength / 4),
  };
};

/**
 * sidecar を**触った shard だけ**遅延ロードし、LRU で落とす gather を組む（ADR 0085 決定 3）。
 *
 * hub には部分読み（Range）の席を新設しない — 最小単位はファイル 1 本のまま。配布形は
 * 決定 1 で固定されているので、実需が出たときにこの実装だけ差し替えれば「行だけ読む」へ移れる
 * （ADR 0085 の代替案 b）。
 *
 * MUST: モジュール副作用ゼロ（この関数を呼ぶまで何も起きない）。
 */
export const createGemma4Ple = (options: Gemma4PleOptions): Gemma4Ple => {
  const { index, readShard, vocabSize } = options;
  const capacity = options.residentShards ?? 2;
  if (!Number.isSafeInteger(capacity) || capacity < 1) {
    throw new Error(`residentShards ${capacity} が 1 以上の整数でない`);
  }
  // ① sidecar の行数 と ② 主 embedding の vocab 行数（ADR 0085 決定 5 の相互照合）。
  if (index.tokens !== vocabSize) {
    throw new Error(
      `PLE sidecar の行数 ${index.tokens} が主 embedding の vocab 行数 ${vocabSize} と違う` +
        `（別の語彙で焼かれた組み合わせ — 引ける id が食い違ったまま形は合う）`,
    );
  }
  const stride = index.layers * index.dim;

  /** 挿入順 = LRU（触った shard を末尾へ付け替え、超過分は先頭から落とす）。 */
  const resident = new Map<number, Promise<ResidentShard>>();
  let loads = 0;
  let disposed = false;

  const shardOf = (id: number): number => {
    // 索引は昇順の隙間なし分割（`parseGemma4PleIndex` の MUST）なので二分探索でよい。
    let low = 0;
    let high = index.shards.length - 1;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (id < index.shards[middle].stop) high = middle;
      else low = middle + 1;
    }
    return low;
  };

  /**
   * shard 1 本を常駐キャッシュから取る（未常駐なら読む）。
   *
   * NOTE: `signal` が効くのは**その shard の読みを始めた gather** に対してだけである。同じ
   * shard を待つ後続の gather は先行の読みに相乗りするので、先行が中断されれば同じ拒否を
   * 受ける（中断は自分のものでないので失敗として上がる = 沈黙はしない）。読みを要求ごとに
   * 分けると 758MB の二重読みになるため、best-effort の側を取っている。
   */
  const acquire = (
    position: number,
    options: Gemma4PleReadOptions,
  ): Promise<ResidentShard> => {
    const cached = resident.get(position);
    if (cached !== undefined) {
      // 参照したので末尾へ付け替える（Map の反復順 = 挿入順）。
      resident.delete(position);
      resident.set(position, cached);
      return cached;
    }
    const shard = index.shards[position];
    loads += 1;
    const pending = readShard(shard.file, options).then((bytes) =>
      readResidentShard(bytes, index, shard)
    );
    // MUST: 失敗した取得を常駐させない（次の gather が同じ拒否済み Promise を掴み続ける）。
    pending.catch(() => {
      if (resident.get(position) === pending) resident.delete(position);
    });
    resident.set(position, pending);
    for (const oldest of resident.keys()) {
      if (resident.size <= capacity) break;
      resident.delete(oldest);
    }
    return pending;
  };

  return {
    async gather(ids: readonly number[], options: Gemma4PleReadOptions = {}): Promise<Tensor> {
      if (disposed) {
        throw new Error("PLE gather: dispose 済みの sidecar は引けない（常駐は解放済み）");
      }
      if (ids.length < 1) throw new Error("PLE gather の token 列が空");
      ids.forEach((id, position) => {
        // ③ 実際に引く id（tokenizer が生成しうる special id を含む）— 範囲外は OOB ではなく
        // 「別 token の有効な行」になるので、ここが唯一の fail loudly の位置。
        if (!Number.isSafeInteger(id) || id < 0 || id >= index.tokens) {
          throw new Error(
            `token id[${position}] ${id} が PLE sidecar の 0..${index.tokens - 1} の外`,
          );
        }
      });
      const data = new Float32Array(ids.length * stride);
      // shard ごとに束ねて引く（同じ shard の行が散っていても取得は 1 回）。
      const grouped = new Map<number, number[]>();
      ids.forEach((id, position) => {
        const shard = shardOf(id);
        const rows = grouped.get(shard);
        if (rows === undefined) grouped.set(shard, [position]);
        else rows.push(position);
      });
      for (const [shard, positions] of grouped) {
        const loaded = await acquire(shard, options);
        for (const position of positions) {
          const row = ids[position] - loaded.start;
          const source = row * stride;
          const scaleRow = row * index.layers;
          let target = position * stride;
          for (let layer = 0; layer < index.layers; layer += 1) {
            const scale = loaded.scales[scaleRow + layer];
            const base = source + layer * index.dim;
            for (let column = 0; column < index.dim; column += 1) {
              // MUST: 逆量子化 → embed scale の **2 段**（GPU 側 `embedding` と直後の `mul` と
              // 同じ順序・同じ丸め点）。JS の算術は f64 なので、`Math.fround` で 1 段目を f32 へ
              // 落としてから掛ける — f64 のまま 1 度に丸めると subnormal 域で 2 段丸めと
              // 結果が割れうる（ADR 0085 決定 4 のビット一致 MUST）。
              data[target + column] = Math.fround(loaded.values[base + column] * scale) *
                index.embedScale;
            }
            target += index.dim;
          }
        }
      }
      return { dtype: "f32", shape: [1, ids.length, index.layers, index.dim], data };
    },
    stats(): Gemma4PleStats {
      return { loads, resident: resident.size };
    },
    dispose(): void {
      disposed = true;
      // 走行中の読みまでは止めない（返ってきた buffer は誰も掴まないので回収される）。
      resident.clear();
    },
  };
};
