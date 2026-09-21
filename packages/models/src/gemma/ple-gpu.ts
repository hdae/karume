/**
 * Gemma 4 の PLE（per-layer embeddings）を **GPU に常駐させ、gather も GPU 内で行う席**
 * （既定は従来どおりホスト — ADR
 * [0085](../../../../docs/decisions/0085-ple-host-gather.md) 追記〈GPU 常駐席〉）。
 *
 * ## 何を消すのか
 *
 * ホスト経路（`./ple.ts`）は run ごとに `per_layer_inputs[1,M,layers,dim]` を**ホストで逆量子化
 * して writeBuffer** する。decode の 1 token でも 35,840 B の転送と行 1 本の逆量子化が要り、
 * prefill の 768 行では 27.5MB の転送になる。この席はその 2 つを消す — sidecar の量子化バイト列
 * を**ロード時に 1 度だけ** GPU へ上げ、run では token id だけを渡して GPU 内で引く。
 *
 * ## 機構（`./greedy-output.ts` と同じ手筋）
 *
 * gather 用の小さな IR（`embedding` → `mul embed_scale`）を持つ Session を 1 本張り、target の
 * run と**同じ batch へ先に enqueue** して、その出力を常駐テンソル（`copyOutputs`）へ書く。
 * target はその常駐テンソルを `per_layer_inputs` の**常駐入力**として受ける（writeBuffer を
 * 1 度も出さず bind group へ焼く）ので、1 token あたりのフェンスは従来どおり 1 本のままである。
 *
 * ## MUST: ホスト経路とビット一致する（ADR 0085 決定 4）
 *
 * 重みは `[tokens×layers, dim]`・添字は `id×layers + layer` の形で持つ。この形なら
 * **1 行 = 1 層ぶんの dim 要素**になり、格納 dtype に依らず sidecar の `scales[rows, layers]` が
 * そのまま行ごとの scale の並びになる（i8 / i2 は行 scale・i4 は `group_size = dim` の group
 * scale で、どちらも平坦添字は `id×layers + layer`）。逆量子化は `embedding` が
 * `f32(q) × scale` を 1 回の f32 乗算で行い、`embed_scale` はその**後**に別ノードで掛かる —
 * ホストの `Math.fround(q × scale) × embedScale` と同じ 2 段・同じ丸め点である。
 *
 * ## MUST: 単一束縛（利用者裁定 2026-09-19 案 A）
 *
 * `values` は分割せず 1 本の GPU バッファへ載せる。**アダプタの束縛上限（`maxStorageBufferBindingSize`
 * / `maxBufferSize`）に収まらない配布形はこの席を使えない** — 黙ってホスト経路へ退避せず、
 * 何バイト足りないかを添えて fail loudly にする。
 *
 * ## ロード（shard ごとに writeBuffer — ADR 0085 決定 2）
 *
 * ホストで 1 本の巨大 ArrayBuffer に連結しない。sidecar の shard を 1 本ずつ読み、**ランタイムの
 * shard 逐次面**（ADR 0070 決定 3 / 0090 の piece）へ流す — ランタイムは piece を親 1 本ぶんの
 * GPU バッファへ行オフセット位置に `queue.writeBuffer` し、shard ごとにフェンスを 1 本立てて
 * 参照を手放す。器（合成 shard の buffer）は 1 本を使い回すので、ホスト RAM のピークは
 * 「最大 shard 1 本 + 器 1 本」に収まる。
 *
 * companion scale は piece 1 と同じ shard に置く契約（ADR 0090 決定 1）なので、**scale だけは
 * 先に全量を集める**（区間読みできる読み口なら shard あたり数 MB の 2 読みで済み、持たない
 * 読み口では shard を 1 度余分に読む）。
 *
 * MUST: 全モジュール副作用ゼロ（この関数群を呼ぶまで何も起きない）。
 */

import {
  type BatchScope,
  type GpuContext,
  type ModelShard,
  parseSafetensors,
  parseSafetensorsHeader,
  type PreparedModel,
  prepareModel,
  type ResidentTensor,
  type RunInput,
  type RunInputs,
  type Session,
  type TensorView,
} from "@karume/runtime";
import { disposeSteps } from "../session/dispose-steps.ts";
import {
  type Gemma4PleIndex,
  type Gemma4PleShard,
  HEADER_LENGTH_BYTES,
  packFactor,
  SCALE_BYTES,
} from "./ple-index.ts";
import {
  type Gemma4PleShardSource,
  gemma4PleShardViews,
  readGemma4PleHeaderPrefix,
} from "./ple-shard.ts";

/** PLE をどこに置くか（{@link Gemma4PipelineOptions.pleResidency} の値域）。 */
export type Gemma4PleResidency = "host" | "gpu";

const PLE_RESIDENCIES: readonly Gemma4PleResidency[] = ["host", "gpu"];

/**
 * 席の指定を検査して返す（**資産を 1 バイトも読む前**に同期で落とす）。
 *
 * MUST: 型の外から来た綴り違いを既定へ倒さない — `"GPU"` のような指定が黙ってホスト経路で
 * 走ると、「GPU 常駐にしたのに速くならない」が例外も警告も無いまま残る。
 */
export const assertGemma4PleResidency = (
  where: string,
  value: unknown,
): Gemma4PleResidency => {
  if (typeof value !== "string" || !PLE_RESIDENCIES.includes(value as Gemma4PleResidency)) {
    throw new Error(
      `${where}: pleResidency ${JSON.stringify(value)} が ${PLE_RESIDENCIES.join(" / ")} でない`,
    );
  }
  return value as Gemma4PleResidency;
};

/** gather IR の綴り（この 1 箇所が正本 — 合成コンテナと Session の両方が引く）。 */
const INDEX_INPUT = "ple_index";
const GATHER_OUTPUT = "per_layer";
const RAW_VALUE = "ple_raw";
const WEIGHT_TENSOR = "ple";
const SCALE_TENSOR = "ple_scale";
const EMBED_SCALE_TENSOR = "embed_scale";
/** 物理行数（token 行）の記号 — 束縛源は {@link INDEX_INPUT} の shape だけ。 */
const ROW_SYMBOL = "M";

/**
 * 合成 shard のヘッダ領域（空白で詰めて固定長にする）。
 *
 * 固定長にするのは**器を使い回す**ため — 長さが shard ごとに動くと、器の先頭からの書き出し位置が
 * 毎回変わって「前回の残り」を踏む形を自分で作ることになる。piece 1 本ぶんのヘッダは
 * 100 バイト前後で、512 は桁 1 つぶんの余裕である（超えたら fail loudly）。
 */
const HEADER_BYTES = 512;

/** 格納 dtype → safetensors の綴り（合成コンテナが書く側）。 */
const storageDtype = (index: Gemma4PleIndex): "I8" | "I4" | "I2" =>
  index.storage === "i2" ? "I2" : index.storage === "i4" ? "I4" : "I8";

/** 格納 dtype → IR の綴り。 */
const irStorage = (index: Gemma4PleIndex): "i8" | "i4" | "i2" => index.storage ?? "i8";

/**
 * GPU 常駐に要るバイト数（**索引だけで決まる** — バイト列を読む前に分かる）。
 *
 * `values` が単一束縛の実体で、`scales` は companion scale である。束縛上限の検査も見積りも
 * この 1 本から引く。
 */
export const gemma4PleGpuBytes = (
  index: Gemma4PleIndex,
): { readonly values: number; readonly scales: number; readonly total: number } => {
  const values = index.tokens * index.layers * index.dim / packFactor(index);
  const scales = index.tokens * index.layers * SCALE_BYTES;
  return { values, scales, total: values + scales };
};

/**
 * gather 用の IR（`embedding` → `mul embed_scale`）。
 *
 * 添字入力を `[1, M, layers]` にしてあるので、出力はそのまま `[1, M, layers, dim]` =
 * `per_layer_inputs` の宣言形になる（reshape は要らない — `embedding` の出力形は
 * 「添字の形 + 重みの最終次元」）。
 *
 * MUST: `embed_scale` は**別ノード**で掛ける（`embedding` の中へ畳まない）。ホスト経路の
 * `Math.fround(q × scale) × embedScale` と丸め点を合わせるのがこの 2 段の唯一の目的である
 * （ADR 0085 決定 4）。
 */
export const gemma4PleGatherGraph = (index: Gemma4PleIndex): Record<string, unknown> => {
  const rows = index.tokens * index.layers;
  const storage = irStorage(index);
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["embedding", "mul"] },
    symbols: [ROW_SYMBOL],
    initializers: {
      [WEIGHT_TENSOR]: {
        tensor: WEIGHT_TENSOR,
        storage: {
          dtype: storage,
          scale: SCALE_TENSOR,
          // i4 だけ group 長を宣言する（ADR 0069 決定 2）。group = dim にすると group scale の
          // 平坦添字が行番号そのものになり、i8 / i2 の行 scale と同じ並びに揃う。
          ...(storage === "i4" ? { group_size: index.dim } : {}),
        },
      },
      [EMBED_SCALE_TENSOR]: { tensor: EMBED_SCALE_TENSOR, storage: { dtype: "f32" } },
    },
    inputs: [{ name: INDEX_INPUT, dtype: "i32", shape: [1, ROW_SYMBOL, index.layers] }],
    outputs: [GATHER_OUTPUT],
    values: {
      [WEIGHT_TENSOR]: { dtype: "f32", shape: [rows, index.dim] },
      [EMBED_SCALE_TENSOR]: { dtype: "f32", shape: [1] },
      [RAW_VALUE]: { dtype: "f32", shape: [1, ROW_SYMBOL, index.layers, index.dim] },
      [GATHER_OUTPUT]: { dtype: "f32", shape: [1, ROW_SYMBOL, index.layers, index.dim] },
    },
    nodes: [
      {
        op: "embedding",
        ins: [WEIGHT_TENSOR, INDEX_INPUT],
        outs: [RAW_VALUE],
        // `padding_idx` は契約表の必須 attr だが forward には効かない（ADR 0012）。番兵 −1 =
        // 未指定で、ホスト経路の行 lookup と同じ「素の gather」になる。
        attrs: { padding_idx: -1 },
      },
      { op: "mul", ins: [RAW_VALUE, EMBED_SCALE_TENSOR], outs: [GATHER_OUTPUT], attrs: {} },
    ],
  };
};

/** 合成する safetensors のテンソル 1 本（payload は呼び手が持つ view）。 */
type SynthTensor = {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly bytes: Uint8Array<ArrayBuffer>;
};

/**
 * テンソル列から safetensors 1 本のバイト列を組む（`into` を渡すと器を使い回す）。
 *
 * MUST: データ節は宣言で隙間なく覆う（パーサの MUST）。ヘッダは 4 の倍数へ詰めるので、
 * データ節の先頭は常に 4 整列 = I4 / I2 / F32 の整列要件を満たす。
 */
const writeSafetensors = (
  metadata: Readonly<Record<string, string>> | undefined,
  tensors: readonly SynthTensor[],
  into?: Uint8Array<ArrayBuffer>,
): Uint8Array<ArrayBuffer> => {
  const header: Record<string, unknown> = {};
  if (metadata !== undefined) header.__metadata__ = metadata;
  let payload = 0;
  for (const tensor of tensors) {
    header[tensor.name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [payload, payload + tensor.bytes.byteLength],
    };
    payload += tensor.bytes.byteLength;
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  // 器を使い回す側は固定長、単発で組む側は 4 の倍数へ詰める（どちらもデータ節が 4 整列）。
  const headerLength = into === undefined ? Math.ceil(json.byteLength / 4) * 4 : HEADER_BYTES;
  if (json.byteLength > headerLength) {
    throw new Error(
      `PLE GPU 常駐: 合成 shard のヘッダ ${json.byteLength} バイトが領域 ${headerLength} を超えた`,
    );
  }
  const total = HEADER_LENGTH_BYTES + headerLength + payload;
  const target = into ?? new Uint8Array(new ArrayBuffer(total));
  if (target.byteLength < total) {
    throw new Error(
      `PLE GPU 常駐: 合成 shard ${total} バイトが器 ${target.byteLength} に収まらない`,
    );
  }
  new DataView(target.buffer, target.byteOffset, target.byteLength)
    .setBigUint64(0, BigInt(headerLength), true);
  target.set(json, HEADER_LENGTH_BYTES);
  target.fill(0x20, HEADER_LENGTH_BYTES + json.byteLength, HEADER_LENGTH_BYTES + headerLength);
  let cursor = HEADER_LENGTH_BYTES + headerLength;
  for (const tensor of tensors) {
    target.set(tensor.bytes, cursor);
    cursor += tensor.bytes.byteLength;
  }
  return target.subarray(0, total);
};

/** piece キー（ADR 0090 決定 1 — 5 桁ゼロ詰め・1 始まり）。 */
const pieceKey = (position: number, count: number): string =>
  `${WEIGHT_TENSOR}#${String(position).padStart(5, "0")}-of-${String(count).padStart(5, "0")}`;

/** sidecar shard 1 本の読み口と、その中の `values` / `scales` の位置。 */
type ShardHead = {
  readonly source: Gemma4PleShardSource;
  readonly values: TensorView;
  readonly scales: TensorView;
};

/** 合成 shard 1 本が運ぶ piece の出どころ。 */
type PiecePlan = {
  readonly shard: number;
  /** その shard の `values` 先頭から飛ばす行数（piece 1 を graph shard へ出した先頭だけ 1）。 */
  readonly skipRows: number;
  readonly rows: number;
};

/**
 * sidecar shard 1 本のヘッダを解き、`scales`（と先頭 shard だけ 1 行ぶんの `values`）を読む。
 *
 * 区間読みを持つ読み口では数 MB の 2〜3 読みで済む。持たない読み口では全量を 1 度読む
 * （この席のロードだけが払う費用 — 値も token 列も変わらない）。
 */
const readShardHead = async (
  index: Gemma4PleIndex,
  shard: Gemma4PleShard,
  source: Gemma4PleShardSource,
  rowBytes: number,
  wantFirstRow: boolean,
): Promise<{
  readonly head: ShardHead;
  readonly scales: Uint8Array<ArrayBuffer>;
  readonly firstRow?: Uint8Array<ArrayBuffer>;
}> => {
  const { range } = source;
  if (range === undefined) {
    const buffer = await source.readAll();
    const file = parseSafetensors(buffer);
    const { values, scales } = gemma4PleShardViews(file, index, shard);
    return {
      head: { source, values, scales },
      // 全量 buffer はこの後捨てるので写す（sidecar 1 本 250MiB 級を掴み続けない）。
      scales: new Uint8Array(
        buffer.slice(scales.byteOffset, scales.byteOffset + scales.byteLength),
      ),
      ...(wantFirstRow
        ? {
          firstRow: new Uint8Array(
            buffer.slice(values.byteOffset, values.byteOffset + rowBytes),
          ),
        }
        : {}),
    };
  }
  const prefix = await readGemma4PleHeaderPrefix(range, source.bytes, shard.file);
  const header = parseSafetensorsHeader(prefix, source.bytes);
  const { values, scales } = gemma4PleShardViews(header, index, shard);
  const scaleBytes = new Uint8Array(await range.read(scales.byteOffset, scales.byteLength));
  return {
    head: { source, values, scales },
    scales: scaleBytes,
    ...(wantFirstRow
      ? { firstRow: new Uint8Array(await range.read(values.byteOffset, rowBytes)) }
      : {}),
  };
};

/** piece 1 本ぶんの量子化バイト列を読む（区間読みがあればその区間だけ）。 */
const readPiece = async (
  head: ShardHead,
  offset: number,
  length: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  const { range } = head.source;
  if (range !== undefined) return new Uint8Array(await range.read(offset, length));
  const buffer = await head.source.readAll();
  return new Uint8Array(buffer, offset, length);
};

/**
 * ホストの GPU 常駐 PLE（**target の run へ差す常駐入力**を作る面）。
 *
 * MUST: `enqueue` は target の enqueue と**同じ batch**へ積む（フェンスを増やさない）。
 */
export type Gemma4PleResident = {
  /** target が受け取るグラフ入力の名前（`per_layer_inputs`）。 */
  readonly inputName: string;
  /** その run の token id 列を運ぶグラフ入力の名前（`input_ids`）。 */
  readonly idsName: string;
  /**
   * `ids` の PLE を GPU 内で引き、常駐テンソルへ書く enqueue を `batch` へ積む。
   *
   * 返るのは target の `per_layer_inputs` に束ねる常駐テンソルで、行数ごとに 1 本を使い回す
   * （同じ形の run は同じバッファを踏むので、bind group の焼き直しも起きない）。
   */
  enqueue(batch: BatchScope, ids: Int32Array<ArrayBuffer>): Promise<ResidentTensor>;
  /** 常駐 PLE（values + scales）と gather Session のぶんの見積り（auxiliaryBytes に載る）。 */
  readonly extraBytes: number;
  /** 常駐した量子化バイト列そのもの（values + scales）。 */
  readonly residentBytes: number;
  dispose(): Promise<void>;
};

export type Gemma4PleResidentOptions = {
  readonly gpu: GpuContext;
  readonly index: Gemma4PleIndex;
  readonly openShard: (file: string) => Promise<Gemma4PleShardSource>;
  /** 主 embedding の vocab 行数（id 空間の相互照合 — ADR 0085 決定 5）。 */
  readonly vocabSize: number;
  /** target が受け取るグラフ入力の名前。 */
  readonly inputName: string;
  /** token id 列を運ぶグラフ入力の名前。 */
  readonly idsName: string;
  /** この pipeline が流しうる物理行数（見積りの上界に使う — decode 1 + バケット + chunk 長）。 */
  readonly rows: readonly number[];
  /** 診断の主語（`Gemma4Pipeline` / `Gemma4QatPipeline`）。 */
  readonly entry: string;
};

/**
 * 合成コンテナ（gather IR + PLE の piece 列）— ランタイムの shard 逐次面へそのまま流す形。
 *
 * MUST: `weightShards` は**グラフ shard を含まない**（`PreparedModel.createSession` の契約）。
 */
export type Gemma4PleGatherShards = {
  readonly graphShard: ModelShard;
  readonly weightShards: AsyncIterable<ModelShard>;
};

/**
 * sidecar から合成コンテナを組む（**GPU を 1 度も触らない**）。
 *
 * 段は 2 つ: ①全 shard の `scales`（と先頭 1 行）を集めて graph shard を作る ②sidecar の shard を
 * 1 本ずつ読み、piece 1 本ぶんの合成 shard として器へ書いて流す。piece 1 を先頭 1 行に切って
 * graph shard へ同居させてあるのは、companion scale の co-shard 契約（ADR 0090 決定 1）を
 * 満たしつつ graph shard を小さく保つためである（`PreparedModel` が Session 構築まで掴む）。
 *
 * NOTE: `export` は合成の突合を GPU 無しで縛るため（`mod.ts` / サブパス面には出さない）。
 */
export const buildGemma4PleGatherShards = async (
  index: Gemma4PleIndex,
  openShard: (file: string) => Promise<Gemma4PleShardSource>,
  entry: string,
): Promise<Gemma4PleGatherShards> => {
  const factor = packFactor(index);
  const rowBytes = index.dim / factor;
  if (!Number.isSafeInteger(rowBytes) || rowBytes % 4 !== 0) {
    throw new Error(
      `${entry}: PLE の 1 層ぶん ${index.dim} 要素（格納 ${irStorage(index)}）が` +
        ` ${rowBytes} バイトで 4 整列しない（合成コンテナの piece が組めない）`,
    );
  }
  const bytes = gemma4PleGpuBytes(index);
  // ── scale は piece 1 と同じ shard に置く契約（ADR 0090 決定 1）なので先に全量を集める。
  const scaleBytes = new Uint8Array(new ArrayBuffer(bytes.scales));
  const heads: ShardHead[] = [];
  let firstRow: Uint8Array<ArrayBuffer> | undefined;
  for (const [position, shard] of index.shards.entries()) {
    const source = await openShard(shard.file);
    const read = await readShardHead(index, shard, source, rowBytes, position === 0);
    scaleBytes.set(read.scales, shard.start * index.layers * SCALE_BYTES);
    if (read.firstRow !== undefined) firstRow = read.firstRow;
    heads.push(read.head);
  }
  if (firstRow === undefined) throw new Error(`${entry}: PLE sidecar の先頭 shard が無い`);

  // ── piece の割り付け（piece 1 = 先頭 1 行で graph shard に同居・残りは sidecar shard 順）。
  const plans: PiecePlan[] = [];
  for (const [position, shard] of index.shards.entries()) {
    const skipRows = position === 0 ? 1 : 0;
    const rows = (shard.stop - shard.start) * index.layers - skipRows;
    if (rows > 0) plans.push({ shard: position, skipRows, rows });
  }
  const pieceCount = plans.length + 1;
  if (pieceCount < 2) {
    throw new Error(`${entry}: PLE sidecar の行数 ${index.tokens * index.layers} が 2 行未満`);
  }

  const graphShard: ModelShard = {
    id: "ple.gather.graph",
    bytes: writeSafetensors(
      { karume_ir: JSON.stringify(gemma4PleGatherGraph(index)) },
      [
        {
          name: EMBED_SCALE_TENSOR,
          dtype: "F32",
          shape: [1],
          bytes: new Uint8Array(Float32Array.of(index.embedScale).buffer),
        },
        {
          name: SCALE_TENSOR,
          dtype: "F32",
          shape: [index.tokens * index.layers, 1],
          bytes: scaleBytes,
        },
        {
          name: pieceKey(1, pieceCount),
          dtype: storageDtype(index),
          shape: [1, index.dim],
          bytes: firstRow,
        },
      ],
    ),
  };

  // 器は 1 本だけ確保して使い回す（ADR 0070 追記の RAM ピーク係数 1 化と同じ流儀）。
  const container = new Uint8Array(
    new ArrayBuffer(
      HEADER_LENGTH_BYTES + HEADER_BYTES +
        plans.reduce((largest, plan) => Math.max(largest, plan.rows * rowBytes), 0),
    ),
  );
  const weightShards = async function* (): AsyncGenerator<ModelShard, void, unknown> {
    for (const [position, plan] of plans.entries()) {
      const head = heads[plan.shard];
      const payload = await readPiece(
        head,
        head.values.byteOffset + plan.skipRows * rowBytes,
        plan.rows * rowBytes,
      );
      yield {
        id: index.shards[plan.shard].file,
        bytes: writeSafetensors(undefined, [{
          name: pieceKey(position + 2, pieceCount),
          dtype: storageDtype(index),
          shape: [plan.rows, index.dim],
          bytes: payload,
        }], container),
      };
    }
  };
  return {
    graphShard,
    weightShards: { [Symbol.asyncIterator]: () => weightShards()[Symbol.asyncIterator]() },
  };
};

/**
 * 束縛上限に収まるかを見る（**重みを 1 バイトも取る前**）。
 *
 * MUST: 黙ってホスト経路へ退避しない。席を明示した利用者に「何が」「どれだけ」足りないかを
 * 返す（`acquireGpu` はアダプタ実測値をそのまま要求するので、ここで足りない値はその機の
 * 上限そのものである）。
 */
const assertBindingLimits = (options: Gemma4PleResidentOptions, valuesBytes: number): void => {
  const { limits } = options.gpu;
  const shortfalls = (
    [
      ["maxStorageBufferBindingSize", limits.maxStorageBufferBindingSize],
      ["maxBufferSize", limits.maxBufferSize],
    ] as const
  ).filter(([, granted]) => granted < valuesBytes);
  if (shortfalls.length === 0) return;
  throw new Error(
    `${options.entry}: pleResidency: "gpu" は PLE の量子化バイト列 ${valuesBytes} バイトを` +
      `単一束縛で載せるが、この device は ` +
      shortfalls.map(([key, granted]) => `${key} ${granted}`).join(" / ") +
      `（不足 ${
        shortfalls.map(([key, granted]) => `${key} ${valuesBytes - granted}`).join(" / ")
      } バイト）— 既定の pleResidency: "host" で読むか、束縛上限の大きい device で開く`,
  );
};

/**
 * PLE を GPU へ常駐させ、GPU 内 gather の面を返す。
 *
 * 順序は「索引の門 → 束縛上限の門 → scale の収集 → 合成 shard の逐次消費」で、GPU を触るのは
 * 最後の 1 段だけである（ADR 0070 決定 5 の graph-first と同じ並び）。
 */
export const createGemma4PleResident = async (
  options: Gemma4PleResidentOptions,
): Promise<Gemma4PleResident> => {
  const { gpu, index, entry } = options;
  // ① sidecar の行数 と ② 主 embedding の vocab 行数（ADR 0085 決定 5 の相互照合 — ホスト経路の
  // `createGemma4Ple` が持つ門と同じ関係を、この席でも通す）。
  if (index.tokens !== options.vocabSize) {
    throw new Error(
      `${entry}: PLE sidecar の行数 ${index.tokens} が主 embedding の vocab 行数` +
        ` ${options.vocabSize} と違う（別の語彙で焼かれた組み合わせ）`,
    );
  }
  const bytes = gemma4PleGpuBytes(index);
  assertBindingLimits(options, bytes.values);
  const { graphShard, weightShards } = await buildGemma4PleGatherShards(
    index,
    options.openShard,
    entry,
  );

  const prepared: PreparedModel = prepareModel(graphShard);
  // 出力の常駐は run の形ごとに 1 本（行数の集合は decode 1 + バケット + chunk 長で有界）。
  const outputBytes = [...new Set(options.rows)].reduce(
    (total, rows) => total + rows * index.layers * index.dim * 4,
    0,
  );
  // MUST: Session と見積りに**同じ予算**を渡す（片方だけ既定に落ちると、報告のピークが実際の
  // 保持集合と別の予算を名乗る）。形ごとの backing は「gather の中間 + 出力」の 2 本ぶんなので、
  // 全形を保持できる値を明示する（既定の 256MiB を使うと、この小さなグラフの報告が予算その
  // ものの数字になる）。
  const backingBudget = outputBytes * 2;
  const maxRows = options.rows.reduce((largest, rows) => Math.max(largest, rows), 1);
  const estimate = prepared.estimate({
    bindings: { [ROW_SYMBOL]: maxRows },
    planBackingBudgetBytes: backingBudget,
  });
  const session: Session = await prepared.createSession(gpu, weightShards, {
    planBackingBudgetBytes: backingBudget,
  });

  /** 行数ごとの出力常駐と添字バッファ（同じ形の run は同じ実体を踏む）。 */
  const outputs = new Map<number, ResidentTensor>();
  const indices = new Map<number, Int32Array<ArrayBuffer>>();
  let disposal: Promise<void> | undefined;

  const outputFor = async (rows: number): Promise<ResidentTensor> => {
    const cached = outputs.get(rows);
    if (cached !== undefined) return cached;
    const resident = await gpu.createResident(
      rows * index.layers * index.dim * 4,
      `ple-gather-${rows}`,
    );
    outputs.set(rows, resident);
    return resident;
  };

  return {
    inputName: options.inputName,
    idsName: options.idsName,
    extraBytes: estimate.peakAccountedBytes + outputBytes,
    residentBytes: bytes.total,
    async enqueue(batch: BatchScope, ids: Int32Array<ArrayBuffer>): Promise<ResidentTensor> {
      if (disposal !== undefined) throw new Error(`${entry}: dispose 済みの PLE は引けない`);
      const rows = ids.length;
      if (rows < 1) throw new Error(`${entry}: PLE gather の token 列が空`);
      let expanded = indices.get(rows);
      if (expanded === undefined) {
        expanded = new Int32Array(new ArrayBuffer(rows * index.layers * 4));
        indices.set(rows, expanded);
      }
      for (let row = 0; row < rows; row += 1) {
        const id = ids[row];
        // 範囲外は OOB ではなく「別 token の有効な行」/ NaN 汚染になるので、ここが fail loudly の
        // 位置である（ホスト経路 `Gemma4Ple.gather` の ③ と同じ門）。
        if (id < 0 || id >= index.tokens) {
          throw new Error(
            `${entry}: token id[${row}] ${id} が PLE sidecar の 0..${index.tokens - 1} の外`,
          );
        }
        const base = id * index.layers;
        for (let layer = 0; layer < index.layers; layer += 1) {
          expanded[row * index.layers + layer] = base + layer;
        }
      }
      const output = await outputFor(rows);
      await session.enqueue(
        {
          [INDEX_INPUT]: {
            dtype: "i32",
            shape: [1, rows, index.layers],
            data: expanded,
          },
        },
        { batch, copyOutputs: { [GATHER_OUTPUT]: output } },
      );
      return output;
    },
    dispose(): Promise<void> {
      // 順序は Session → 常駐テンソル（焼き込み参照を返してから解放する — ADR 0004）。
      disposal ??= disposeSteps([
        () => session.dispose(),
        ...[...outputs.values()].map((resident) => () => resident.dispose()),
      ]).then(() => {
        outputs.clear();
        indices.clear();
      });
      return disposal;
    },
  };
};

/**
 * run の入力から token id 列を取り出す（GPU 内 gather の添字の唯一の出どころ）。
 *
 * MUST: ホストの `Tensor`（i32）であることまで見る。常駐入力（`ResidentTensor`）にはホスト側の
 * 値が無いので、黙って読むと「空の添字で引いた PLE」が例外なしで流れる。
 */
export const gemma4PleGatherIds = (
  where: string,
  inputs: RunInputs,
  name: string,
): Int32Array<ArrayBuffer> => {
  const input: RunInput | undefined = Object.hasOwn(inputs, name) ? inputs[name] : undefined;
  if (input === undefined || !("dtype" in input) || input.dtype !== "i32") {
    throw new Error(
      `${where}: 入力 '${name}' が i32 のホストテンソルでない（GPU 内 gather の添字）`,
    );
  }
  return input.data;
};
