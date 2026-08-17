/**
 * GenerationContext — 生成 1 本ぶんの可変 state を所有する器（ADR 0066 決定 1 / 3 / 5 / 6 / 7）。
 *
 * Session（不変重み + 計画キャッシュ）から `states{}` 宣言（ADR 0066 決定 2 — format/ir.ts）を
 * 読み、スロットごとの物理バッファ 1 本と、論理長を運ぶ可変 uniform 1 本を確保する。これは
 * **第 5 の寿命クラス**で、既存 4 クラス（重みアリーナ / slot backing / run アリーナ /
 * {@link ResidentTensor}）のどれにも載せない — 前 3 者は寿命が `Session.dispose` に一本化されて
 * おり、ResidentTensor は GpuContext 所有の別 dispose 契約を持つため、どちらも「context 単位で
 * 返す」（決定 6）と粒度が合わない。素の `createBuffer` + 自前の簿記が最小の形になる。
 *
 * MUST: 本波（波 C）の時点で **state を読み書きする実行経路は無い** — ノードの `states` 欄・
 * `state_append`・attention 統合は波 D（ADR 0067 決定 4 / 5）。ここにあるのは所有権・寿命・
 * 搬送路（論理長 uniform）だけで、実行統合の結線点は「波 D」と書いた注記がそのまま指す。
 * MUST: executor.ts を import しない（Session → context の一方向 import を型でも崩さない）。
 * Session から借りる面は {@link GenerationContextHost} の構造的な数欄だけ
 * （`RecipeBuilderContext` と同じ流儀）。
 */

import { evalDim, parseDim } from "../format/dims.ts";
import type { IrDim, IrGraph } from "../format/ir.ts";
import { STORAGE_USAGE } from "../gpu/arena.ts";
import {
  discardFailureScopes,
  type GpuContext,
  GpuDeviceLostError,
  popFailureScopes,
  pushFailureScopes,
  RUNTIME_INTERNAL,
} from "../gpu/device.ts";
import { numel } from "../ops.ts";
import { ExecutionError, type SymbolBindings } from "./plan.ts";
import type { GenerationContextSpec } from "./session-types.ts";

/**
 * 論理長 uniform のバイト数（`pastLength` / `queryLength` の 2 語 — ADR 0066 追記 4）。
 *
 * MUST: 2 語ちょうど。カーネル側の `struct { past: u32, query: u32 }` と同じ大きさで、
 * 既存の params uniform（`struct Params { rows: u32, dim: u32 }` — codegen/reduce.ts）と
 * 同じ形なので、束縛の最小サイズはこの 8 バイトで足りる。
 */
const LENGTHS_BYTES = 8;

/**
 * 論理長 uniform の usage。
 *
 * `COPY_SRC` を付けるのは**内容を読み戻せる唯一の観測点**を残すため — `queue.writeBuffer` は
 * 無効なバッファや整列違反に対して警告すら出さない no-op になるので、「書いた値が実際に
 * 載っているか」は読み戻す以外に確かめる手段が無い（実行経路はこのコピーを出さない）。
 */
const LENGTHS_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/** state スロットの dtype は f32 のみ（`STATE_DTYPES` — f16 席は ADR 0066 追記 5 の予約）。 */
const STATE_ELEMENT_BYTES = 4;

/** state スロット 1 本の物理実体（束縛解決済み — 実行中に再確保しない = 静的物理格納）。 */
export type StateSlotBacking = {
  readonly buffer: GPUBuffer;
  /** 束縛解決済みの容量込み具体形（宣言 shape と同 rank）。 */
  readonly shape: readonly number[];
  readonly byteLength: number;
};

/**
 * context が Session から借りる面（**必要な欄だけ**の構造的な面）。
 *
 * MUST: executor.ts の `SessionState` を import しない（モジュール doc の一方向 import）。
 * Session は自分の状態から組んだこの束を渡す。
 */
export type GenerationContextHost = {
  readonly gpu: GpuContext;
  readonly graph: IrGraph;
  /**
   * 未 submit のエンコードを出し切る（flush-before-destroy — ADR 0004。実体は
   * `SubmitScheduler.flush`）。
   */
  flush(): Promise<void>;
  /** Session の run / enqueue / dispose と同じチェーンへ積む（実体は `Session.#serialize`）。 */
  serialize<T>(body: () => Promise<T>): Promise<T>;
  /** 破棄の決着を Session の診断へ返す（`stateBacking.residentBytes` の生存集合から外す）。 */
  forget(context: GenerationContext): void;
};

/**
 * {@link GenerationContext} のランタイム内部面（利用者ストーリーに対応しない実体と進行）。
 *
 * MUST: 素の名前で公開しない（ADR 0008 の薄い面）。論理長の進行は run の成功で起きる契約
 * （決定 6 — ホスト側の手動加算は API にしない）なので、`advance` / `poison` / `writeLengths` は
 * **波 D の実行統合だけが呼ぶ**。
 */
type GenerationContextInternals = {
  /** state スロットの実体（波 D の bind group はここから束ねる — 決定 5 の焼き込み分離）。 */
  readonly slots: ReadonlyMap<string, StateSlotBacking>;
  /** 論理長 uniform（波 D のレシピは固定束縛でこれを参照する — 追記 4）。 */
  readonly lengths: GPUBuffer;
  /** この context が常駐させている GPU バイト数（診断 `stateBacking.residentBytes` の元）。 */
  readonly bytes: number;
  /** 論理長を書き出す（毎 run の encode 前 — {@link GenerationContext} の doc）。 */
  writeLengths(queryLength: number): void;
  /** 論理長を進める（**run の成功でのみ** — 決定 6）。 */
  advance(queryLength: number): void;
  /** 汚染する（state 変更 dispatch を含む run の失敗 — 追記 3）。 */
  poison(reason: string): void;
};

/**
 * `spec.bindings` を検査して null プロトタイプの表へ写す。
 *
 * MUST: 束縛の器は null プロトタイプ（plan.ts の `bindSymbols` と同じ理由 — シンボルの文法
 * `[A-Za-z_][A-Za-z0-9_]*` は "__proto__" にマッチし、素の `{}` では代入が [[Prototype]] 設定に
 * 化けて own property が作られない）。
 * MUST: 記号容量は**ここで与えられた値だけ**で決まる。states は束縛源にならず（ADR 0066
 * 決定 2）、context は入力を 1 本も持たないので、入力 shape からの推定は原理的に不可能。
 */
const resolveBindings = (
  graph: IrGraph,
  bindings: SymbolBindings | undefined,
): SymbolBindings => {
  const symbols = new Set(graph.symbols);
  const resolved: Record<string, number> = Object.create(null);
  for (const [sym, value] of Object.entries(bindings ?? {})) {
    if (!symbols.has(sym)) {
      throw new ExecutionError(
        `束縛 '${sym}' はグラフの symbols [${graph.symbols.join(", ")}] に無い`,
      );
    }
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new ExecutionError(`束縛 '${sym}': ${value} が非負整数でない`);
    }
    resolved[sym] = value;
  }
  return resolved;
};

/**
 * スロット 1 本の宣言 shape を束縛で具体化する（容量込みの具体形 — ADR 0066 決定 2）。
 *
 * MUST: 未束縛シンボルは fail loudly。ここを 0 や 1 で埋めると、容量の足りないスロットのまま
 * 生成が走り出して沈黙 OOB になる。
 */
const resolveSlotShape = (
  name: string,
  shape: readonly IrDim[],
  bindings: SymbolBindings,
): number[] =>
  shape.map((dim, index) => {
    if (typeof dim === "number") return dim;
    const expr = parseDim(dim);
    if (!Object.hasOwn(bindings, expr.sym)) {
      throw new ExecutionError(
        `state '${name}': 記号次元 '${dim}' のシンボル '${expr.sym}' が束縛されていない` +
          "（createGenerationContext の bindings で与えること — states は束縛源にならない）",
      );
    }
    const extent = evalDim(expr, bindings);
    if (extent < 1) {
      throw new ExecutionError(
        `state '${name}': 次元 ${index} の容量 ${extent} が正でない` +
          `（束縛 '${expr.sym}' = ${bindings[expr.sym]}）`,
      );
    }
    return extent;
  });

export class GenerationContext {
  /**
   * 固定長 prefill chunk の行数（ADR 0066 決定 4 — context 生成時に確定する計画時定数）。
   * decode は `queryLength = 1` 固定形で、prefill の末尾 chunk は pad で埋める（切るのは
   * `queryLength`）。
   */
  readonly chunkLength: number;
  /** ランタイム内部面（利用者が触る面ではない）。 */
  readonly [RUNTIME_INTERNAL]: GenerationContextInternals;
  readonly #host: GenerationContextHost;
  readonly #slots: ReadonlyMap<string, StateSlotBacking>;
  readonly #lengths: GPUBuffer;
  /**
   * 論理長の書き出し値。**全域を毎回書く**（部分書きにすると、片方だけ更新された組が残って
   * 「pastLength は新しく queryLength は前 step」という混ざった uniform で dispatch が回る）。
   */
  readonly #lengthValues = new Uint32Array(2);
  #pastLength = 0;
  /** 汚染の理由（追記 3）。立つと `pastLength` 読みと `dispose` 以外の全操作を拒否する。 */
  #poisoned: string | undefined;
  #disposal: Promise<void> | undefined;

  /** MUST: 構築の入口は {@link GenerationContext.create} だけ（errorScope の門を迂回させない）。 */
  private constructor(
    host: GenerationContextHost,
    slots: ReadonlyMap<string, StateSlotBacking>,
    lengths: GPUBuffer,
    chunkLength: number,
  ) {
    this.#host = host;
    this.#slots = slots;
    this.#lengths = lengths;
    this.chunkLength = chunkLength;
    this[RUNTIME_INTERNAL] = {
      slots,
      lengths,
      // 容量は確定済み（静的物理格納 — ADR 0066 決定 3）なので、ここで 1 度畳んで持つ。
      bytes: [...slots.values()].reduce((total, slot) => total + slot.byteLength, 0) +
        LENGTHS_BYTES,
      writeLengths: (queryLength: number): void => this.#writeLengths(queryLength),
      advance: (queryLength: number): void => this.#advance(queryLength),
      poison: (reason: string): void => this.#poison(reason),
    };
  }

  /**
   * スロット容量と `chunkLength` を確定して物理確保する（ADR 0066 決定 6）。
   *
   * MUST: async なのは errorScope で囲むため（決定 6 の「確保失敗は out-of-memory errorScope で
   * fail loudly」）。上限超過 / 余力切れの `createBuffer` は同期例外を投げず**無効なバッファを
   * 返す**ので、囲まないと空の KV を束ねたまま生成ループが回る。
   * MUST: 上限ゲート（追記 5）は確保の**前**に通す。後にすると、超過は「無効バッファ由来の
   * validation」という一段派生した診断になり、どのスロットが大きすぎたのかが消える。
   */
  static async create(
    host: GenerationContextHost,
    spec: GenerationContextSpec,
  ): Promise<GenerationContext> {
    const { gpu, graph } = host;
    const names = Object.keys(graph.states);
    // MUST: state の無いグラフでは作らせない。context は「1 生成ぶんの可変 state の所有者」
    // なので、states 宣言が 0 本のモデルに対しては器そのものが無意味 — 取り違え（別モデルの
    // Session から作った）の検出線をここに置く。
    if (names.length === 0) {
      throw new ExecutionError(
        "このグラフは states 宣言を持たない（GenerationContext は state スロットの所有者なので、" +
          "state の無いモデルでは作れない — 1-shot 実行は Session.run / enqueue をそのまま使う）",
      );
    }
    if (!Number.isSafeInteger(spec.chunkLength) || spec.chunkLength < 1) {
      throw new ExecutionError(
        `chunkLength ${spec.chunkLength} が 1 以上の整数でない` +
          "（固定長 prefill chunk の行数 — ADR 0066 決定 4）",
      );
    }
    const bindings = resolveBindings(graph, spec.bindings);
    const limit = gpu.limits.maxStorageBufferBindingSize;
    const planned = names.map((name) => {
      const shape = resolveSlotShape(name, graph.states[name].shape, bindings);
      const byteLength = numel(shape) * STATE_ELEMENT_BYTES;
      // MUST: スロット単体の束縛バイト数が上限を超える容量指定は fail loudly（追記 5）。
      // 分割して束ねる形は持たない（KV は連続容量 — 決定 8 の明示選択）ので、超過は容量設計の
      // 誤りとして呼び出し点で落とす以外に手が無い。
      if (byteLength > limit) {
        throw new ExecutionError(
          `state '${name}': 容量 [${shape.join(",")}] の ${byteLength} バイトが ` +
            `maxStorageBufferBindingSize ${limit} バイトを超える（ADR 0066 追記 5）。` +
            "容量を下げるか、スロットを分けてグラフを組み直すこと",
        );
      }
      return { name, shape, byteLength };
    });

    // MUST: push から pop の**発行**までに await を挟まない（device 単位 LIFO の交錯を防ぐ根拠 —
    // GpuContext 冒頭「errorScope 区間の不変条件」の 3 つ目。同期区間で完結するのでロック不要）。
    pushFailureScopes(gpu.device);
    const created: GPUBuffer[] = [];
    const slots = new Map<string, StateSlotBacking>();
    let lengths: GPUBuffer;
    try {
      for (const { name, shape, byteLength } of planned) {
        const buffer = gpu.device.createBuffer({
          label: `state '${name}'`,
          size: byteLength,
          usage: STORAGE_USAGE,
        });
        created.push(buffer);
        slots.set(name, { buffer, shape, byteLength });
      }
      lengths = gpu.device.createBuffer({
        label: "generation lengths",
        size: LENGTHS_BYTES,
        usage: LENGTHS_USAGE,
      });
      created.push(lengths);
    } catch (cause) {
      // MUST: 後始末の失敗で本体の例外を上書きしない（Session.create と同じ規律）。push した
      // 2 本は必ず pop し、確保済みは 1 本残らず返す（この窓で漏れた実体は dispose からも
      // 到達できない）。
      await discardFailureScopes(gpu.device);
      for (const buffer of created) buffer.destroy();
      throw cause;
    }
    const failure = await popFailureScopes(gpu.device, "GenerationContext の state 確保");
    if (failure !== undefined) {
      for (const buffer of created) buffer.destroy();
      throw failure;
    }
    return new GenerationContext(host, slots, lengths, spec.chunkLength);
  }

  /**
   * 確定済み KV の論理長（ADR 0066 決定 6）。
   *
   * 進行は **run の成功でのみ**起きる（ホスト側の手動加算は API にしない — 二重簿記の禁止）。
   * 汚染後も読める（追記 3 — 失われるのは物理 state で、「どこまで進んだか」という事実は残る）。
   * device 消失後は読めない（決定 7 — 背後の物理 state は回復不能なので、この数値を
   * 「ここから再開できる」と読ませない）。
   */
  get pastLength(): number {
    this.#assertReadable("pastLength");
    return this.#pastLength;
  }

  /**
   * 論理位置を切り詰める（ADR 0066 決定 6）。`0 ≤ position ≤ pastLength` の整数のみ。
   *
   * NOTE（波 D で結線）: **sliding スロットを含む context の位置指定 rewind は fail loudly**
   * （追記 2 — エビクト発生後は resident な位置でも物理配置と論理範囲が一致しないため、ORT
   * GenAI と同じく全拒否）。sliding 性はノード attrs `window`（ADR 0067 決定 4）由来で、ノードの
   * `states` 欄が IR にまだ無い本波では判定材料が存在しない。波 D の `states` 欄実装と同時に、
   * この位置へ拒否を入れる。
   */
  rewind(position: number): void {
    this.#assertUsable("rewind");
    if (!Number.isSafeInteger(position) || position < 0) {
      throw new ExecutionError(`rewind: 位置 ${position} が非負整数でない`);
    }
    if (position > this.#pastLength) {
      throw new ExecutionError(
        `rewind: 位置 ${position} が現在の pastLength ${this.#pastLength} を超えている` +
          "（前進は run の成功でのみ起きる）",
      );
    }
    this.#pastLength = position;
  }

  /**
   * state スロットと論理長 uniform を返す（ADR 0066 決定 6 — flush-before-destroy）。
   *
   * MUST: 2 度目以降も同じ完了を返す（`Session.dispose` と同じ理由 — 先に返すと呼び手が
   * 「破棄済み」と見なして `device.destroy()` まで進み、flush-before-destroy が崩れる）。
   * MUST: flush が失敗（主因は device 消失）してもバッファ破棄と簿記の返却は必ず行い、失敗
   * 自体は握り潰さず後始末の後に伝播させる（`RunArena.#destroyOnce` と同じ規律）。
   * NOTE: Session より長生きしてよい。この経路が触るのは注入された flush と自分のバッファだけで、
   * Session の重み・計画キャッシュには手を出さない（順序の依存を作らない）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#host.serialize(async () => {
      try {
        await this.#host.flush();
      } finally {
        for (const slot of this.#slots.values()) slot.buffer.destroy();
        this.#lengths.destroy();
        this.#host.forget(this);
      }
    });
    return this.#disposal;
  }

  /**
   * 論理長 uniform を書く（ADR 0066 追記 4 の搬送路）。
   *
   * MUST: params の内容アドレスキャッシュ（ADR 0042）には**載せない**。毎 step 値が変わるものを
   * 内容アドレスに載せると「キャッシュ無界成長」と「PreparedPlan ヒット時に導出相が走らず更新
   * 不能」の両方を踏む。実体は context 所有のこのバッファ 1 本きりで、レシピからは固定束縛で
   * 参照する（波 D — 束縛の結線は `states` 欄つき attention / `state_append` の実装と同時）。
   * MUST: 呼ぶのは**毎 run の encode 前**（`queue.writeBuffer` は issue 順で queue timeline に
   * 載るので、submit 済みの dispatch を追い越さない — ADR 0004 不変条件④）。
   */
  #writeLengths(queryLength: number): void {
    this.#assertUsable("writeLengths");
    this.#assertQueryLength(queryLength, "writeLengths");
    this.#lengthValues[0] = this.#pastLength;
    this.#lengthValues[1] = queryLength;
    this.#host.gpu.device.queue.writeBuffer(this.#lengths, 0, this.#lengthValues);
  }

  /**
   * 論理長を進める（ADR 0066 決定 6 — **run の成功でのみ**呼ぶ）。
   *
   * NOTE（波 D で結線）: full スロットの実行時検査 `pastLength + queryLength ≤ 容量`
   * （ADR 0067 決定 4 の④ — context 側検査）はここに入る。容量軸がどの次元かは
   * ノードの `states` 欄と attrs `window` が決めるため、本波では判定材料が存在しない。
   */
  #advance(queryLength: number): void {
    this.#assertUsable("advance");
    this.#assertQueryLength(queryLength, "advance");
    this.#pastLength += queryLength;
  }

  /**
   * 汚染する（ADR 0066 追記 3 — state 変更 dispatch を含む run の失敗）。
   *
   * 論理長は進まないが物理 ring は上書きされ得るので、rollback / staging を持たない設計では
   * 「以後の全操作を拒否」以外に整合を主張する手段が無い（復旧 = 新しい context + ホスト側
   * 再構築）。トリガの結線は波 D の実行統合（run の失敗経路）。
   *
   * MUST: 2 度目以降は最初の理由を保つ（真因を後続の失敗で上書きしない）。
   */
  #poison(reason: string): void {
    this.#poisoned ??= reason;
  }

  /**
   * 読み取り（`pastLength`）だけを許す条件。破棄済みと device 消失をここで落とす。
   *
   * MUST: device 消失は {@link GpuDeviceLostError}（決定 7 — lost device 由来の GPU 資源は
   * WebGPU 仕様上回復不能で、生成は失われる）。
   */
  #assertReadable(where: string): void {
    if (this.#disposal !== undefined) {
      throw new ExecutionError(`${where}: dispose 済みの GenerationContext は使えない`);
    }
    if (this.#host.gpu.lost !== undefined) {
      throw new GpuDeviceLostError(
        `${where}: device が失われた GenerationContext は使えない` +
          "（生成は失われる — device を取り直して作り直すこと）",
      );
    }
  }

  /** 状態を変える操作の条件（読み取りの条件 + 汚染の拒否 — 追記 3）。 */
  #assertUsable(where: string): void {
    this.#assertReadable(where);
    if (this.#poisoned !== undefined) {
      throw new ExecutionError(
        `${where}: 汚染された GenerationContext は使えない（${this.#poisoned}）。` +
          "復旧は新しい context + ホスト側の state 再構築",
      );
    }
  }

  /**
   * 今 step の実 token 数の検査（ADR 0066 決定 4 の 2 つの実行形に共通）。
   *
   * prefill-chunk は `queryLength ≤ chunkLength`（超えた行は物理 shape に載らない）、decode は
   * `queryLength = 1`。0 を許さないのは「何も進めない run」が state の書き込み範囲を空にして、
   * 進行と物理内容の対応が観測できなくなるため。
   */
  #assertQueryLength(queryLength: number, where: string): void {
    if (!Number.isSafeInteger(queryLength) || queryLength < 1) {
      throw new ExecutionError(`${where}: queryLength ${queryLength} が 1 以上の整数でない`);
    }
    if (queryLength > this.chunkLength) {
      throw new ExecutionError(
        `${where}: queryLength ${queryLength} が chunkLength ${this.chunkLength} を超えている` +
          "（prefill は固定長 chunk + pad・decode は 1 — ADR 0066 決定 4）",
      );
    }
  }
}
