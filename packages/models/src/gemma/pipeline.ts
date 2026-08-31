/**
 * `Gemma4Pipeline` — **文字列 in → 文字列 out** の 1 本の面（生成 API 波の段 4）。
 *
 * 束ねるのは 4 つで、どれも既に別の場所で正本を持っている:
 *
 * 1. 製品グラフ（PLE 外出し + 最終行 logits 出口）の Session — `tools/export-recipes/gemma4/
 *    export_product.py` が書く shard 列
 * 2. ホスト PLE gather（`src/gemma/ple.ts` — ADR
 *    [0085](../../../../docs/decisions/0085-ple-host-gather.md)）を
 *    {@link GenerationProgram.derivedInputs} の席へ差す
 * 3. compile 済み tokenizer（`src/gemma/text/`）と chat フォーマット（`src/gemma/text/chat.ts`）
 * 4. 生成ループ（`src/generation/` — ADR
 *    [0083](../../../../docs/decisions/0083-generation-api-surface.md) の program / sequence /
 *    sampler）
 *
 * ここが足すのは**結線と id 空間の相互照合だけ**で、数値も語彙も 1 つも持たない。
 *
 * ## MUST: 配布形（manifest）はまだ持たない
 *
 * 他ファミリの `fromAssets` は `{ manifest, assets }` を受け、静的配線を `pipelineConfig` から
 * 引く（ADR 0038 §1）。gemma4 の配布形は**段 5**（ADR 0065 stage 6 のライセンス門待ち）なので、
 * ここは manifest を受けない — 代わりに {@link Gemma4Assets.config} が同じ欄を持ち、段 5 では
 * `pipelineConfig` がその値を宣言するだけになる（型はそのまま流用できる）。`fromPretrained` /
 * pin 定数もまだ無い。**既定値は置かない** — chunk 長も容量も位置上限も資産世代ごとに動く
 * ので、黙って古い数を使う形を作らない。
 *
 * ## MUST: id 空間を相互照合する（ADR 0085 決定 5）
 *
 * tokenizer が生成しうる id / 主 embedding の vocab 行数 / PLE sidecar の行数を
 * {@link admitGemma4} が突き合わせる。ここがずれると **OOB ではなく「別 token の有効な行」**を
 * 引く（例外なしで沈黙して壊れる）ので、fail loudly の門を置く場所はここしかない。
 *
 * ## MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）
 */

import {
  acquireGpu,
  createSessionFromShards,
  type GpuContext,
  type ModelShard,
  prepareModel,
  type Session,
} from "@karume/runtime";

import { createOperationChain } from "../concurrency/serial.ts";
import {
  createGenerationProgram,
  type GenerationGraph,
  type GenerationProgram,
} from "../generation/program.ts";
import {
  createGenerationSequence,
  type GenerationSequence,
  type GenerationStop,
  type GenerationStream,
} from "../generation/sequence.ts";
import type { SamplerSpec } from "../generation/sampler.ts";
import { createGemma4Ple, parseGemma4PleIndex } from "./ple.ts";
import { parseGemmaTokenizerAsset } from "./text/asset.ts";
import { GemmaTokenizer } from "./text/tokenizer.ts";
import { type Gemma4ChatMessage, gemma4ChatPrompt, gemma4StopTokens } from "./text/chat.ts";

/** グラフ入力の名前（正本は `export_product.py` の定数）。 */
const INPUT_IDS = "input_ids";
const POSITION_IDS = "position_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";

/**
 * 静的配線のうち**資産世代ごとに動く数**（段 5 では manifest の `pipelineConfig` が宣言する）。
 *
 * NOTE: 記号（full スロットの容量記号）はここに置かない — グラフから導出できるものを宣言に
 * 二重持ちすると、片方だけ古びる（{@link capacitySymbolOf}）。
 */
export type Gemma4PipelineConfig = {
  /** 固定長 prefill chunk の行数（ADR 0066 決定 4 — context の計画時定数）。 */
  readonly chunkLength: number;
  /** 資産が引ける絶対位置の排他的上限（= 焼き込んだ RoPE 表の行数）。 */
  readonly maxPosition: number;
  /** full スロットの容量（会話が使える最大の論理長 — 実行時に選ぶ）。 */
  readonly capacity: number;
  /**
   * 配布形が宣言する sampler の既定（ADR 0083 決定 7）。
   *
   * NOTE: 段 4 の時点では**誰も宣言しない** = 省略時は低層の既定（温度 0 = greedy）のまま
   * である。推奨値（gemma-4-E2B-it は temperature 1.0 / top_k 64 / top_p 0.95）を焼くのは
   * 段 5 の `pipelineConfig` の仕事で、席だけ先に開けてある。
   */
  readonly sampler?: SamplerSpec;
};

/**
 * 取得済み資産から組むときの入力（**製品系列 1 世代ぶん**）。
 *
 * MUST: PLE sidecar だけ「バイト列」ではなく**読み口**を受ける。全量は i8 で 2,240MiB あり、
 * 常駐させると単一 ArrayBuffer 天井の議論（ADR 0085 決定 2）をホスト側で再現することになる。
 * 触った shard だけを遅延ロードする形（同 決定 3）が成立する唯一の受け方である。
 */
export type Gemma4Assets = {
  readonly config: Gemma4PipelineConfig;
  /** 製品グラフのコンテナ shard 列（**宣言順** — 先頭がグラフ shard。ADR 0081）。 */
  readonly model: readonly Uint8Array<ArrayBuffer>[];
  /** compile 済み tokenizer 資産のバイト列（ADR 0084 決定 1）。 */
  readonly tokenizer: Uint8Array<ArrayBuffer>;
  /** PLE sidecar の索引（`ple.json` のバイト列）。 */
  readonly pleIndex: Uint8Array<ArrayBuffer>;
  /** PLE sidecar shard 1 本を取る（ファイル読み / hub の `streamAssets` — 呼び手の責務）。 */
  readonly readPleShard: (file: string) => Promise<ArrayBuffer>;
};

export type Gemma4PipelineOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので {@link Gemma4Pipeline.dispose} は
   * 破棄しない。省略時はパイプラインが内部で `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /**
   * PLE sidecar shard の常駐本数（LRU — ADR 0085 決定 3。省略時は `ple.ts` の既定 2）。
   *
   * shard 1 本は 758MB 級なので、常駐を絞るほど RAM は減り、範囲をまたぐ会話では読み直しが
   * 増える。全部載せる（= shard 本数）と読み直しはゼロになる。
   */
  readonly residentPleShards?: number;
};

/** 1 ターンぶんの chat リクエスト。 */
export type Gemma4ChatOptions = {
  /** 生成する token 数の上限（1 以上）。停止 token はこの数に**含めない**。 */
  readonly maxNewTokens: number;
  /** sampling の指定（省略時は {@link Gemma4PipelineConfig.sampler}、それも無ければ greedy）。 */
  readonly sampler?: SamplerSpec;
  /** 中断（段の境目で検査し `signal.reason` をそのまま throw する — ADR 0083 決定 5）。 */
  readonly signal?: AbortSignal;
};

/**
 * 文字列片の列（`for await` で汲む）+ 停止理由。
 *
 * 片は逐次復号器が**確定させたぶん**だけで（ADR 0084 決定 4）、byte_fallback の途中は次の
 * token まで持ち越される。連結すると `decode(全 token id)` と一致する。
 *
 * MUST: `done` は**二次的な**通知路である（`GenerationStream.done` と同じ規律）— 失敗は
 * iterable 側が throw するのが一次で、`done` は同じ例外で reject するだけ。
 */
export type Gemma4ChatStream = AsyncIterable<string> & {
  readonly done: Promise<GenerationStop>;
};

/** {@link Gemma4Pipeline} の内部状態（公開面には出さない）。 */
type Gemma4State = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly session: Session;
  readonly program: GenerationProgram;
  readonly tokenizer: GemmaTokenizer;
  readonly config: Gemma4PipelineConfig;
};

/**
 * 家族 admission（GPU を取りに行く前に通す門）が確定させる材料。
 *
 * NOTE: PLE loader はここに載せない — `program.derivedInputs.derive` の閉包が持つのが唯一の
 * 参照で、席を 2 つ作ると「片方だけ差し替えた」形が書ける。
 */
type Gemma4Admission = {
  readonly shards: readonly ModelShard[];
  readonly program: GenerationProgram;
  readonly tokenizer: GemmaTokenizer;
  readonly config: Gemma4PipelineConfig;
};

const assertPositiveInteger = (value: number, where: string): void => {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`Gemma4Pipeline: ${where} ${value} が 1 以上の整数でない`);
  }
};

/**
 * 最終行 logits 出口の語彙数をグラフから引く（`[1, 1, V]` — ADR 0083 決定 6）。
 *
 * MUST: 呼び手に宣言させない。V は主 embedding の行数そのもので、宣言と食い違えば PLE
 * sidecar との相互照合（ADR 0085 決定 5）が**間違った基準**で通ってしまう。形の検査は
 * `createGenerationProgram` が同じ値でもう一度行う。
 */
const vocabSizeOf = (graph: GenerationGraph): number => {
  if (graph.outputs.length !== 1) {
    throw new Error(
      `Gemma4Pipeline: グラフ出力が ${graph.outputs.length} 本` +
        `（製品グラフの出口は最終行 logits の 1 本 — ADR 0083 決定 6）`,
    );
  }
  const name = graph.outputs[0];
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`Gemma4Pipeline: グラフ出力 '${name}' の値情報が無い`);
  }
  const shape = graph.values[name].shape;
  const vocab = shape[2];
  if (shape.length !== 3 || typeof vocab !== "number") {
    throw new Error(
      `Gemma4Pipeline: グラフ出力 '${name}' の shape [${shape.join(",")}] が [1,1,V] でない`,
    );
  }
  return vocab;
};

/**
 * full スロットの容量記号をグラフから引く。
 *
 * 記号は「入力 shape から決まらないもの」がちょうど 1 本のはずで（chunk 長の記号は
 * `input_ids` の 2 次元目から決まる・容量記号は states にしか現れない）、それを
 * `createGenerationContext` の束縛点へ渡す（ADR 0066 追記 7）。綴りを定数で持たないのは、
 * 資産側の綴りが変わったときに**黙って束縛されない記号**が残るのを避けるため。
 */
const capacitySymbolOf = (graph: GenerationGraph): string => {
  const fromInputs = new Set<string>();
  for (const input of graph.inputs) {
    for (const dim of input.shape) if (typeof dim === "string") fromInputs.add(dim);
  }
  const free = graph.symbols.filter((symbol) => !fromInputs.has(symbol));
  if (free.length !== 1) {
    throw new Error(
      `Gemma4Pipeline: 入力 shape から決まらない記号が ${free.length} 本` +
        `（[${free.join(", ")}] — full スロットの容量記号 1 本であること）`,
    );
  }
  return free[0];
};

/**
 * この資産の組み合わせを gemma4 として実行できるかを見る（**GPU を取りに行く前**に全部通す）。
 *
 * MUST: 家族の門はこの 1 本に集める（他ファミリの `admit*` と同じ規律）— 後段へ散らすと、
 * 3.7GiB の資産を読んだ**後**にしか落ちない。
 */
const admitGemma4 = (input: Gemma4Assets, options: Gemma4PipelineOptions): Gemma4Admission => {
  const { config } = input;
  assertPositiveInteger(config.chunkLength, "config.chunkLength");
  assertPositiveInteger(config.maxPosition, "config.maxPosition");
  assertPositiveInteger(config.capacity, "config.capacity");
  if (input.model.length === 0) {
    throw new Error("Gemma4Pipeline: 製品グラフの shard 列が空（先頭がグラフ shard）");
  }
  const shards: ModelShard[] = input.model.map((bytes, index) => ({
    id: `model[${index}]`,
    bytes,
  }));
  // MUST: `PreparedModel` は握らず `IrGraph` だけ残す（`hub/components.ts` と同じ規律 —
  // Session はグラフ shard も含めた列を毎回流し直す）。
  const graph = prepareModel(shards[0]).graph;
  const vocabSize = vocabSizeOf(graph);

  const tokenizer = new GemmaTokenizer(parseGemmaTokenizerAsset(input.tokenizer));
  // ① tokenizer が生成しうる id と ② 主 embedding の vocab 行数（ADR 0085 決定 5）。
  if (tokenizer.maxTokenId >= vocabSize) {
    throw new Error(
      `Gemma4Pipeline: tokenizer の最大 token id ${tokenizer.maxTokenId} が` +
        ` 主 embedding の vocab 行数 ${vocabSize} の外（別の語彙で焼かれた組み合わせ）`,
    );
  }
  // ③ PLE sidecar の行数（この突合は `createGemma4Ple` が持つ — 同じ検査を 2 実装持たない）。
  const ple = createGemma4Ple({
    index: parseGemma4PleIndex(
      JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(input.pleIndex)),
    ),
    readShard: input.readPleShard,
    vocabSize,
    ...(options.residentPleShards === undefined
      ? {}
      : { residentShards: options.residentPleShards }),
  });

  const program = createGenerationProgram({
    graph,
    inputIds: INPUT_IDS,
    positionIds: POSITION_IDS,
    lastRow: LAST_ROW,
    logits: graph.outputs[0],
    chunkLength: config.chunkLength,
    maxPosition: config.maxPosition,
    capacity: config.capacity,
    vocabSize,
    // 停止集合は tokenizer 資産の追加語彙から導出する（ADR 0083 決定 8 / 0084 決定 5 —
    // chat 形式と同じ digest set から来る）。
    stopTokens: gemma4StopTokens(tokenizer),
    bindings: { [capacitySymbolOf(graph)]: config.capacity },
    // ホスト由来の per-chunk 入力の席に PLE gather を差す（ADR 0085）。
    derivedInputs: {
      names: [PER_LAYER_INPUTS],
      derive: async (ids) => ({ [PER_LAYER_INPUTS]: await ple.gather(ids) }),
    },
  });
  return { shards, program, tokenizer, config };
};

/**
 * gemma4 の chat パイプライン（製品グラフ 1 本 + ホスト PLE + tokenizer）。
 *
 * 構築の入口は {@link Gemma4Pipeline.fromAssets} だけ — コンストラクタを private にしてある
 * のは、資産の突合を迂回した半端な状態を作れないようにするため（ADR 0008）。
 * `fromPretrained`（HF 配布形）は段 5。
 */
export class Gemma4Pipeline {
  readonly #state: Gemma4State;
  /** chat と dispose の直列化鎖（1 つの Session を 2 本の会話で同時に押さない）。 */
  readonly #chain = createOperationChain();
  /** {@link Gemma4Pipeline.sequence} が渡した実体（dispose の取りこぼしを塞ぐ）。 */
  readonly #handed = new Set<GenerationSequence>();
  /** dispose の 1 本。**undefined でないことが「dispose 済み」**（派生状態を別に持たない）。 */
  #disposal: Promise<void> | undefined;

  private constructor(state: Gemma4State) {
    this.#state = state;
  }

  /**
   * 取得済み資産から組む。資産の解釈・グラフとの突合・id 空間の相互照合を全てここで済ませ、
   * **製品グラフの Session を 1 本張って**返す。
   *
   * Session を 1 本持ち続けるのは siglip2 と同じ理由で、畳む相手（同時に載せられない別の
   * 巨大グラフ）が居ないため — 会話ごとに張り直すと 1.5GiB の重みを毎回アップロードし直す。
   */
  static async fromAssets(
    input: Gemma4Assets,
    options: Gemma4PipelineOptions = {},
  ): Promise<Gemma4Pipeline> {
    const admitted = admitGemma4(input, options);
    const gpu = options.gpu ?? await acquireGpu();
    const ownsGpu = options.gpu === undefined;
    try {
      return new Gemma4Pipeline({
        gpu,
        ownsGpu,
        // shard 列は使い切りで、Session はこの 1 本きり（`fromAssets` は 1 回しか呼ばれない）。
        session: await createSessionFromShards(gpu, toShardStream(admitted.shards)),
        program: admitted.program,
        tokenizer: admitted.tokenizer,
        config: admitted.config,
      });
    } catch (error) {
      // 内部で取った GPU は、構築に失敗したら誰も解放できなくなるのでここで返す。
      if (ownsGpu) gpu.destroy();
      throw error;
    }
  }

  /**
   * 会話 1 ターンを回し、**確定した文字列片**を流す（ADR 0084 決定 4 の逐次復号）。
   *
   * 会話の描画と符号化は `gemma4ChatPrompt`（射程外の入力はここで**同期に** fail loudly）、
   * 生成は `GenerationSequence` 1 本で、汲み切る / `break` する / 中断するのいずれでも
   * sequence は返る。**1 ターン = 1 sequence** なので過去 turn は残らない — 多ターンの会話を
   * 自分で回すなら {@link Gemma4Pipeline.sequence} を使う。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（1 つの Session を 2 本の会話で同時に押さない）。
   */
  chat(
    messages: readonly Gemma4ChatMessage[],
    options: Gemma4ChatOptions,
  ): Gemma4ChatStream {
    if (this.#disposal !== undefined) {
      throw new Error("Gemma4Pipeline: dispose 済みでは生成できない");
    }
    // 受理集合は同期に落とす（GPU にも順番待ちにも入る前）。
    const prompt = gemma4ChatPrompt(this.#state.tokenizer, messages);
    const sampler = options.sampler ?? this.#state.config.sampler;

    let settle!: (stop: GenerationStop) => void;
    let fail!: (error: unknown) => void;
    const done = new Promise<GenerationStop>((resolve, reject) => {
      settle = resolve;
      fail = reject;
    });
    // 二次的な通知路なので、読まれなくても unhandled rejection にしない。
    done.catch(() => {});

    const state = this.#state;
    const acquire = this.#acquire.bind(this);
    const chunks = async function* (): AsyncGenerator<string, void, undefined> {
      let release: (() => void) | undefined;
      let sequence: GenerationSequence | undefined;
      let stream: GenerationStream | undefined;
      let failure: { readonly error: unknown } | undefined;
      try {
        release = await acquire();
        sequence = await createGenerationSequence({
          session: state.session,
          program: state.program,
        });
        const detokenizer = state.tokenizer.createDetokenizer();
        stream = sequence.generate({
          prompt,
          maxNewTokens: options.maxNewTokens,
          ...(sampler === undefined ? {} : { sampler }),
          ...(options.signal === undefined ? {} : { signal: options.signal }),
        });
        for await (const event of stream) {
          if (event.kind !== "token") continue;
          const text = detokenizer.push(event.id);
          if (text !== "") yield text;
        }
        const tail = detokenizer.finish();
        if (tail !== "") yield tail;
      } catch (error) {
        failure = { error };
        // MUST: 包まずそのまま投げる（ADR 0083 決定 5 — 消費側が `error === signal.reason` で
        // 自分の中断を識別できる）。
        throw error;
      } finally {
        // 停止理由は**内側の `done` をそのまま**運ぶ（中断は resolve `aborted`・失敗は reject
        // という sequence 側の分け方を、ここで作り直さない）。
        if (stream === undefined) {
          if (failure !== undefined) fail(failure.error);
          else settle({ reason: "closed" });
        } else {
          try {
            settle(await stream.done);
          } catch (error) {
            fail(error);
          }
        }
        // 1 ターン = 1 sequence（context を抱えたままにしない）。
        if (sequence !== undefined) await sequence.dispose();
        release?.();
      }
    };

    const iterable = chunks();
    return { [Symbol.asyncIterator]: () => iterable, done };
  }

  /**
   * 低レベル面 — 多ターンの会話を自分で回す（ADR 0083 決定 1〜4 の `GenerationSequence`）。
   *
   * `prompt` は token id 列なので、会話の描画は {@link gemma4ChatPrompt}（`./gemma` サブパス）
   * を呼び手が通す。**返った実体は呼び手が `dispose()` する** — 会話が終わった時点で返すのが
   * 正で、取りこぼしても {@link Gemma4Pipeline.dispose} が巻き取る（Session を live な context
   * ごと畳まないため）。
   *
   * MUST: {@link Gemma4Pipeline.chat} との直列化はしない（別の会話は別の context なので
   * ランタイム側は受ける）。同時に走らせれば KV も 2 本ぶん常駐する。
   */
  async sequence(): Promise<GenerationSequence> {
    if (this.#disposal !== undefined) {
      throw new Error("Gemma4Pipeline: dispose 済みでは sequence を作れない");
    }
    const sequence = await createGenerationSequence({
      session: this.#state.session,
      program: this.#state.program,
    });
    this.#handed.add(sequence);
    return sequence;
  }

  /** 静的配線（`sequence()` で回すときに chunk 長・容量・停止集合を読む口）。 */
  get program(): GenerationProgram {
    return this.#state.program;
  }

  /** 資産から組んだ tokenizer（chat の描画・復号に要る — 同じ digest set の 1 員）。 */
  get tokenizer(): GemmaTokenizer {
    return this.#state.tokenizer;
  }

  /**
   * 解放する。渡した sequence を先に畳み、Session を畳み、**内部で取得した GPU だけ**破棄する。
   *
   * MUST: in-flight の生成の完了を待ってから破棄する（flush-before-destroy）— 破棄も鎖に
   * 載せることで、待ちと破棄の順序を 1 箇所で決める。2 度目以降も同じ完了を返す。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(async () => {
      for (const sequence of this.#handed) await sequence.dispose();
      this.#handed.clear();
      await this.#state.session.dispose();
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
    });
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }

  /**
   * 直列化鎖の席を取り、返した関数で手放す。
   *
   * 席を取るのは**本体が回り出した時**（async generator の本体は最初の `next()` まで走らない）。
   * 発行時に取ると、汲まれないまま捨てられた stream が鎖を永久に握り、`dispose` まで
   * 巻き添えになる。
   */
  #acquire(): Promise<() => void> {
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    return new Promise<() => void>((admitted) => {
      void this.#chain(() => {
        admitted(release);
        return held;
      });
    });
  }
}

/**
 * 手元の shard 列を 1 度だけ流す（Session はこのパイプラインで 1 本きり）。
 *
 * NOTE: 使い切りで良いのは `fromAssets` が Session を 1 本しか張らないため — 張り直す面が
 * 増えたら `hub/components.ts` の `assetShardStream` と同じ「呼ぶたびに新しい iterator」へ
 * 変える必要がある。
 */
const toShardStream = (shards: readonly ModelShard[]): AsyncIterable<ModelShard> => ({
  [Symbol.asyncIterator]: async function* () {
    for (const shard of shards) yield shard;
  },
});
