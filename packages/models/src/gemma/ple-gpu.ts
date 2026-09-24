/**
 * Gemma 4 の PLE（per-layer embeddings）を **GPU に常駐させ、gather も GPU 内で行う席**
 * （既定は従来どおりホスト — ADR
 * [0085](../../../../docs/decisions/0085-ple-host-gather.md) 追記〈GPU 常駐席〉）。
 *
 * ## 何を消すのか
 *
 * ホスト経路（`./ple.ts`）は run ごとに `per_layer_inputs[1,M,layers,dim]` を**ホストで逆量子化
 * して writeBuffer** する。decode の 1 token でも 35,840 B の転送と行 1 本の逆量子化が要り、
 * prefill の 768 行では 27.5MB の転送になる。この席はその 2 つを消す — PLE の量子化バイト列
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
 * **1 行 = 1 層ぶんの dim 要素**になり、格納 dtype に依らず `scales` の `[tokens, layers]` が
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
 * ## ロード（piece ごとに writeBuffer — ADR 0085 決定 2 / ADR 0109 決定 4）
 *
 * ホストで 1 本の巨大 ArrayBuffer に連結しない。索引が指す `values` の block を**メモリ内容器の
 * piece**（`openMemoryContainer` の `pieces`）としてそのまま渡す — Session 構築は part
 * （= piece 1 本）ごとに `read()` を呼び、親 1 本ぶんの GPU バッファへ行オフセット位置に
 * `queue.writeBuffer` し、part ごとにフェンスを 1 本立てて参照を手放す。piece のバイト列は
 * その 1 回の読みぶんしか確保しない（器は使い回さず piece ごとに取る）ので、`values` 由来の
 * ホスト RAM は「最大 block 1 本」に収まる。
 *
 * companion scale は piece 1 と同じ part に置かれる契約（container-v1 §13.3 の規則③）なので、
 * **scale だけは先に全量を集める**。ここが構築時ピークの支配項である — `scales` は
 * `tokens × layers × 4` バイトで、E2B（262,144 token × 35 層）なら 1 本の器に 35 MiB を
 * 集める。容器はその器をそのまま抱える（複製しない）ので、ホスト RAM のピークは
 * 「scale 表の全量 + 最大 block 1 本」である。「最大 block 1 本」だけで見積もると支配項を
 * 丸ごと落とす。
 *
 * MUST: **読み口を 1 回の読みより長く持たない**（`./ple.ts` と同じ MUST）。検証済みでない
 * 取得元の `AssetReader` は block を読み口の寿命ぶん保持するので、掴み続けると「1 本ずつ流す」
 * 形がホスト RAM の上では成立しなくなる。
 *
 * MUST: 全モジュール副作用ゼロ（この関数群を呼ぶまで何も起きない）。
 */

import {
  type AssetReader,
  type BatchScope,
  type BoundContainer,
  type CodecName,
  type GpuContext,
  type IrDeclaration,
  type MemoryEncoding,
  type MemoryTensor,
  openMemoryContainer,
  parseIrDeclarationValue,
  prepareContainer,
  type PreparedModel,
  type ResidentTensor,
  type RunInput,
  type RunInputs,
  type Session,
} from "@karume/runtime";
import { disposeSteps } from "../session/dispose-steps.ts";
import { readAssetRange } from "../hub/asset-readers.ts";
import {
  type Gemma4PleBlock,
  gemma4PleBlockBytes,
  type Gemma4PleIndex,
  type Gemma4PleTable,
  packFactor,
} from "./ple-index.ts";

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

/** gather IR の綴り（この 1 箇所が正本 — メモリ内容器と Session の両方が引く）。 */
const INDEX_INPUT = "ple_index";
const GATHER_OUTPUT = "per_layer";
const RAW_VALUE = "ple_raw";
const WEIGHT_TENSOR = "ple";
const EMBED_SCALE_TENSOR = "embed_scale";
/** メモリ内容器に載せるグラフの名前（初期化子名 {@link WEIGHT_TENSOR} とは別の名前空間）。 */
const GATHER_GRAPH = "ple";
/** 物理行数（token 行）の記号 — 束縛源は {@link INDEX_INPUT} の shape だけ。 */
const ROW_SYMBOL = "M";

/** 格納 dtype → codec 台帳の登録名（`values` の詰め方 — container-v1 §6.3）。 */
const pleCodec = (index: Gemma4PleIndex): CodecName =>
  index.storage === "i2" ? "int2-off" : index.storage === "i4" ? "int4-sym-g" : "int8-sym";

/**
 * GPU 常駐に要るバイト数（**索引だけで決まる** — バイト列を読む前に分かる）。
 *
 * `values` が単一束縛の実体で、`scales` は companion scale である。束縛上限の検査も見積りも
 * この 1 本から引く。
 */
export const gemma4PleGpuBytes = (
  index: Gemma4PleIndex,
): { readonly values: number; readonly scales: number; readonly total: number } => {
  const values = index.tokens * index.values.rowBytes;
  const scales = index.tokens * index.scales.rowBytes;
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
 *
 * MUST: 組んだ宣言は `parseIrDeclarationValue` に通す — ここは exporter も容器の読み手も
 * 経由しないので、通さないとグラフ単体で決まる規則（SSA・トポロジカル順・`requires.ops` と
 * 実使用 op の一致・未知キー・記号名の正準表記）が 1 つも検査されない。
 *
 * MUST NOT: 下のリテラルに省略可能な `states` を書き足さない。書かないことで、リテラルは
 * {@link IrDeclaration} として型が付かず、`parseIrDeclarationValue` の戻り値だけがこの関数の
 * 返り値になれる（包みを外した編集は `deno check` で落ちる）。
 */
export const gemma4PleGatherGraph = (index: Gemma4PleIndex): IrDeclaration => {
  const rows = index.tokens * index.layers;
  return parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["embedding", "mul"] },
    symbols: [ROW_SYMBOL],
    // IR v2 の宣言は名前だけ（格納は束縛表 = メモリ内容器の `encoding` が持つ）。`shared` は
    // 書かない = 実体を持つ側（`false` は綴れない — 欄の不存在と同じ宣言）。
    initializers: {
      [WEIGHT_TENSOR]: {},
      [EMBED_SCALE_TENSOR]: {},
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
  });
};

/**
 * block 1 本の区間を読む（読み口は**この読み 1 回ぶん**だけ生かす — モジュール doc の MUST）。
 */
const readBlockRange = async (
  openBlock: (asset: string) => AssetReader,
  block: Gemma4PleBlock,
  offset: number,
  length: number,
): Promise<Uint8Array<ArrayBuffer>> =>
  new Uint8Array(await readAssetRange(openBlock(block.asset), offset, length));

/** `scales` の block を索引の順に読んで、1 本の行列へ並べ直す（companion scale の全量）。 */
const readScaleTable = async (
  table: Gemma4PleTable,
  openBlock: (asset: string) => AssetReader,
  total: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  const bytes = new Uint8Array(new ArrayBuffer(total));
  for (const block of table.blocks) {
    const payload = await readBlockRange(openBlock, block, 0, gemma4PleBlockBytes(table, block));
    bytes.set(payload, block.start * table.rowBytes);
  }
  return bytes;
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
  /** 索引が指す block 1 本の読み口を開く（`open(MODEL).asset` そのもの）。 */
  readonly openBlock: (asset: string) => AssetReader;
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
 * 索引の指す block からメモリ内容器を組む（**GPU を 1 度も触らない**）。
 *
 * `values` の block 1 本 = piece 1 本で、piece の `read()` はそのときだけ区間読みを出す
 * （この層はバイト列を 1 本も抱えない）。行範囲は `[block.start × layers, block.stop × layers)`
 * — IR の 1 行 = 1 層ぶんなので、token 区間はそのまま行区間になる。
 *
 * companion scale は規則③で piece 1 と同じ part に置かれるので、scale だけ先に全量を集める。
 * block が 1 本しか無い索引は `pieces`（2 本以上 MUST）に割れないので全量 1 本で渡す。
 *
 * NOTE: `export` は合成の突合を GPU 無しで縛るため（`mod.ts` / サブパス面には出さない）。
 */
export const buildGemma4PleGatherContainer = async (
  index: Gemma4PleIndex,
  openBlock: (asset: string) => AssetReader,
  entry: string,
): Promise<BoundContainer> => {
  const factor = packFactor(index);
  const rowBytes = index.dim / factor;
  if (!Number.isSafeInteger(rowBytes) || rowBytes % 4 !== 0) {
    throw new Error(
      `${entry}: PLE の 1 層ぶん ${index.dim} 要素（格納 ${index.storage}）が` +
        ` ${rowBytes} バイトで 4 整列しない（容器の block 末尾整列を満たせない）`,
    );
  }
  const bytes = gemma4PleGpuBytes(index);
  const scale = await readScaleTable(index.scales, openBlock, bytes.scales);
  // i4 は group = 行長（ADR 0069 決定 2 の「2 冪 ≥ 16」を満たす）・i8 / i2 は per-channel で
  // groupSize = 行長。どちらも group 数 1 なので、scale は `[tokens × layers, 1]` の f32 列
  // そのもの（= `readScaleTable` が並べた順）になる。
  const encoding: MemoryEncoding = {
    codec: pleCodec(index),
    rowAxis: 0,
    groupSize: index.dim,
    scale,
  };
  const table = index.values;
  const readBlock = (block: Gemma4PleBlock) => (): Promise<Uint8Array<ArrayBuffer>> =>
    readBlockRange(openBlock, block, 0, gemma4PleBlockBytes(table, block));
  const values: MemoryTensor = table.blocks.length < 2
    ? { encoding, bytes: await readBlock(table.blocks[0])() }
    : {
      encoding,
      pieces: table.blocks.map((block) => ({
        rows: [block.start * index.layers, block.stop * index.layers] as const,
        read: readBlock(block),
      })),
    };
  return openMemoryContainer({
    graphs: { [GATHER_GRAPH]: gemma4PleGatherGraph(index) },
    tensors: {
      [GATHER_GRAPH]: {
        [WEIGHT_TENSOR]: values,
        [EMBED_SCALE_TENSOR]: {
          encoding: { codec: "f32" },
          bytes: new Uint8Array(Float32Array.of(index.embedScale).buffer),
        },
      },
    },
  });
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
 * 順序は「索引の門 → 束縛上限の門 → scale の収集 → piece の逐次消費」で、GPU を触るのは
 * 最後の 1 段だけである（ADR 0070 決定 5 の graph-first と同じ並び）。
 *
 * NOTE: `values` の block が 1 本しか無い索引（全量が block 上限に収まる小さな配布形）だけは
 * piece に割れないので、scale の収集と同じ段で values も全量を読む — capability の門
 * （`prepareContainer`）より前に読み切る唯一の形である。
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
  const bound = await buildGemma4PleGatherContainer(index, options.openBlock, entry);

  const prepared: PreparedModel = prepareContainer(bound, GATHER_GRAPH);
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
  const session: Session = await prepared.createContainerSession(gpu, {
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
