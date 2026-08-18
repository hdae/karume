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
 * MUST: ここが持つのは所有権・寿命・搬送路（論理長 uniform）と、**state を束ねる bind group の
 * 置き場**（ADR 0066 決定 5 の分離焼き込み）だけ。束ね方を決めるのは実行側で、context はその
 * 成果物を backing の世代識別子と対で預かるだけの器（{@link GenerationContextInternals}）。
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
import { numel, stateWindow } from "../ops.ts";
import { ExecutionError, type SymbolBindings } from "./plan.ts";
import type { BakedGroups } from "./recipe.ts";
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

/**
 * 論理長の上限（搬送先が `Uint32Array` / WGSL `u32` — ADR 0066 追記 4）。
 *
 * MUST: safe integer だけでは足りない。`Uint32Array` への代入は範囲外を**黙って切り詰める**ので、
 * `queryLength = 2**32` は uniform 上で 0 になり、ホストが持つ論理長と GPU が見る値が例外も警告も
 * 無いまま分裂する（`writeBuffer` は値を検査しない）。
 */
const MAX_LOGICAL_LENGTH = 0xffffffff;

/**
 * device が使える状態かの同期判定（ADR 0066 決定 7 の遮断面 — 判定はここ 1 箇所）。
 *
 * MUST: `lost` だけでなく `destroyRequested` も見る。`destroy()` はフラグを同期に立てるのに
 * `device.lost` の reaction が走るのは以後のタスクなので、`lost` だけだとその窓で操作が通り、
 * 特に `writeLengths` が破棄済みバッファへの**沈黙 no-op**（警告すら出ない）になる。
 * MUST: 型は {@link GpuDeviceLostError}（lost device 由来の GPU 資源は WebGPU 仕様上回復不能で、
 * 生成は失われる）。意図的な破棄と予期しない消失で復旧手段は変わらないので型は分けない。
 */
const assertDeviceUsable = (gpu: GpuContext, where: string): void => {
  if (gpu.destroyRequested || gpu.lost !== undefined) {
    throw new GpuDeviceLostError(
      `${where}: device が失われた（生成は失われる — device を取り直して作り直すこと）`,
    );
  }
};

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
 * **実行統合（`Session.run` の generation 面）だけが呼ぶ**。
 */
type GenerationContextInternals = {
  /** state スロットの実体（generation run の bind group はここから束ねる — 決定 5）。 */
  readonly slots: ReadonlyMap<string, StateSlotBacking>;
  /** 論理長 uniform（レシピは固定束縛でこれを参照する — 追記 4）。 */
  readonly lengths: GPUBuffer;
  /** この context が常駐させている GPU バイト数（診断 `stateBacking.residentBytes` の元）。 */
  readonly bytes: number;
  /**
   * 容量記号の解決済み束縛（`spec.bindings` の検査済みの写し — ADR 0066 追記 7）。
   *
   * states と入力の**両方**に現れる記号は、context 側（容量）と run 側（入力 shape）の 2 箇所で
   * 独立に決まる。実行統合はこの表と run の解決済み束縛を照合して分裂を fail loudly にする
   * （割れたまま走ると、確保容量と計画が別の値で組まれた state を沈黙で読む）。
   */
  readonly bindings: SymbolBindings;
  /**
   * 進行中の generation run を 1 本受け付ける（**`Session.run` の同期区間で**取り、run の決着で
   * 必ず返す）。取れなければ fail loudly（dispose 要求後・汚染後・device 消失後）。
   */
  acquireRun(): void;
  /** 進行中の generation run を返す（成功・失敗の両経路で必ず 1 度）。 */
  releaseRun(): void;
  /**
   * 論理長の内部読み（run の頭で 1 度だけ — 遮断面は**破棄本体の実行後**）。
   *
   * MUST: 利用者面の `pastLength` を run から読まない。dispose は 2 段（受付終了 → chained な
   * 破棄本体）で、受理済み run は 1 段目と 2 段目の間で走るため、利用者面の判定では
   * 「`run(); dispose()` の非 await 並び」が受理済み run を殺す。
   */
  pastLength(): number;
  /**
   * `token` の backing に対して焼いてある context 側 bind group（無ければ undefined = 焼き直し）。
   *
   * MUST: 引くときに必ず `token` を照合する（**引ける形を token 無しで作らない**）。束は
   * backing 所有のバッファ（slot / 入力）も掴んでいるので、退役した backing の token で焼いた束を
   * そのまま dispatch すると破棄済みバッファを読む — 照合と焼き直しと dispatch の順を
   * 1 箇所に閉じるための引数（executor の `#generationGroups`）。
   */
  bakedGroups(token: number): BakedGroups | undefined;
  /** 焼き直した束を預ける（前の束は捨てる — 退役した backing への参照を残さない）。 */
  setBakedGroups(token: number, groups: BakedGroups): void;
  /**
   * 論理長を書き出す（毎 run の encode 前 — {@link GenerationContext} の doc）。
   * `pastLength` は**呼び出し側が run の頭で捕捉した値**（内部の現在値との一致を照合する）。
   */
  writeLengths(pastLength: number, queryLength: number): void;
  /** 論理長を進める（**run の成功でのみ** — 決定 6）。捕捉 P の照合は `writeLengths` と同じ。 */
  advance(pastLength: number, queryLength: number): void;
  /** 汚染する（state 変更 dispatch を含む run の失敗 — 追記 3）。 */
  poison(reason: string): void;
};

/**
 * sliding なスロットの名前（ノード attrs `window` 由来 — ADR 0067 決定 4）。
 *
 * MUST: 判定材料はノード側にしかない（`graph.states` の宣言は容量だけを持ち、窓は
 * **参照するノード**が宣言する）。同一スロットに触れる全ノードで `window` が一致することは
 * `validateGraphContracts` の `assertStateOrder` が Session 構築時に済ませているので、
 * ここは 1 本でも sliding 宣言があれば sliding として拾えばよい。
 */
const slidingSlotNames = (graph: IrGraph): ReadonlySet<string> => {
  const sliding = new Set<string>();
  graph.nodes.forEach((node, index) => {
    const slots = Object.values(node.states);
    if (slots.length === 0) return;
    const window = stateWindow(node.attrs, `nodes[${index}] (${node.op})`);
    if (window === undefined) return;
    for (const slot of slots) sliding.add(slot);
  });
  return sliding;
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
  /** sliding なスロット名（{@link GenerationContext.rewind} の全拒否条件 — ADR 0066 追記 2）。 */
  readonly #slidingSlots: ReadonlySet<string>;
  readonly #lengths: GPUBuffer;
  /**
   * 論理長の書き出し値。**全域を毎回書く**（部分書きにすると、片方だけ更新された組が残って
   * 「pastLength は新しく queryLength は前 step」という混ざった uniform で dispatch が回る）。
   */
  readonly #lengthValues = new Uint32Array(2);
  #pastLength = 0;
  /**
   * context 側で焼いた bind group 束と、それを焼いた相手の backing の世代識別子
   * （ADR 0066 決定 5）。**GPUBindGroup は destroy 不要**（GC 任せ）だが、掴んでいる backing 所有の
   * バッファの寿命を延ばすので、焼き直しでは必ず前の束ごと置き換える。
   */
  #baked: { readonly token: number; readonly groups: BakedGroups } | undefined;
  /** 汚染の理由（追記 3）。立つと `dispose` 以外の全操作を拒否する（読みも含む）。 */
  #poisoned: string | undefined;
  /**
   * 進行中の generation run の本数（**リース**）。`Session.run` の同期区間で取り、run の決着で
   * 返す。0 でない間は {@link GenerationContext.rewind} を拒否する。
   */
  #runs = 0;
  /**
   * dispose の 1 段目 — **新規受付の終了**（同期に立つ）。以後の run admission と利用者面の
   * 読み書きを拒否するが、受理済み run の内部面（論理長の搬送・進行）は**まだ通す**。
   */
  #disposeRequested = false;
  /** dispose の 2 段目 — 破棄本体が走り出したこと（内部面もここで閉じる）。 */
  #disposed = false;
  #disposal: Promise<void> | undefined;

  /** MUST: 構築の入口は {@link GenerationContext.create} だけ（errorScope の門を迂回させない）。 */
  private constructor(
    host: GenerationContextHost,
    slots: ReadonlyMap<string, StateSlotBacking>,
    slidingSlots: ReadonlySet<string>,
    lengths: GPUBuffer,
    chunkLength: number,
    bindings: SymbolBindings,
  ) {
    this.#host = host;
    this.#slots = slots;
    this.#slidingSlots = slidingSlots;
    this.#lengths = lengths;
    this.chunkLength = chunkLength;
    this[RUNTIME_INTERNAL] = {
      slots,
      lengths,
      // 容量は確定済み（静的物理格納 — ADR 0066 決定 3）なので、ここで 1 度畳んで持つ。
      bytes: [...slots.values()].reduce((total, slot) => total + slot.byteLength, 0) +
        LENGTHS_BYTES,
      bindings,
      acquireRun: (): void => {
        this.#assertUsable("run");
        this.#runs += 1;
      },
      releaseRun: (): void => {
        if (this.#runs < 1) {
          throw new ExecutionError(
            "releaseRun: 進行中の generation run が居ないのにリースを返した（簿記の破れ）",
          );
        }
        this.#runs -= 1;
      },
      pastLength: (): number => {
        this.#assertInternalUsable("pastLength");
        return this.#pastLength;
      },
      bakedGroups: (token: number): BakedGroups | undefined =>
        this.#baked?.token === token ? this.#baked.groups : undefined,
      setBakedGroups: (token: number, groups: BakedGroups): void => {
        this.#baked = { token, groups };
      },
      writeLengths: (pastLength: number, queryLength: number): void =>
        this.#writeLengths(pastLength, queryLength),
      advance: (pastLength: number, queryLength: number): void =>
        this.#advance(pastLength, queryLength),
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
    // MUST: 確保を始める前に device の使用可否を見る（決定 7）。失われた device 上の
    // createBuffer は無効なバッファを返すだけなので、通すと「空の KV を持つ context」が
    // 出来上がる（errorScope も device 消失後は失敗を報告しない）。
    assertDeviceUsable(gpu, "createGenerationContext");
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
    // MUST: 上限は u32（`queryLength ≤ chunkLength` の門を通じて論理長の上限もここで決まる —
    // {@link MAX_LOGICAL_LENGTH}）。
    if (
      !Number.isSafeInteger(spec.chunkLength) || spec.chunkLength < 1 ||
      spec.chunkLength > MAX_LOGICAL_LENGTH
    ) {
      throw new ExecutionError(
        `chunkLength ${spec.chunkLength} が 1..${MAX_LOGICAL_LENGTH} の整数でない` +
          "（固定長 prefill chunk の行数・搬送先は u32 — ADR 0066 決定 4 / 追記 4）",
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
    const where = "GenerationContext の state 確保";
    const created: GPUBuffer[] = [];
    const slots = new Map<string, StateSlotBacking>();
    let popped = false;
    let context: GenerationContext | undefined;
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
      const lengths = gpu.device.createBuffer({
        label: "generation lengths",
        size: LENGTHS_BYTES,
        usage: LENGTHS_USAGE,
      });
      created.push(lengths);
      const pending = popFailureScopes(gpu.device, where);
      popped = true;
      // MUST: 消失後の popErrorScope が解決しない実装がありうる（ResidentTensor.read の
      // mapAsync と同じ理由）ため競わせる — ハングを失敗に変換し、消失を
      // GpuDeviceLostError へ正規化する。
      const failure = await gpu[RUNTIME_INTERNAL].raceDeviceLost(pending, where);
      if (failure !== undefined) throw failure;
      context = new GenerationContext(
        host,
        slots,
        slidingSlotNames(graph),
        lengths,
        spec.chunkLength,
        bindings,
      );
      return context;
    } finally {
      // MUST: 後始末の失敗で本体の例外を上書きしない（Session.create と同じ規律）。push した
      // 2 本は必ず pop する（pop 発行前に抜けた場合だけ — 二重 pop は他所のスコープを取る）。
      if (!popped) await discardFailureScopes(gpu.device);
      // MUST: context を返す場合**以外**は確保済みを 1 本残らず返す（同期 throw・pop の
      // reject・device 消失のいずれでも漏らさない。この窓で漏れた実体は dispose からも
      // 到達できない）。destroy はこの finally 1 回きりなので二重呼び出しにならない。
      if (context === undefined) {
        for (const buffer of created) buffer.destroy();
      }
    }
  }

  /**
   * 確定済み KV の論理長（ADR 0066 決定 6）。
   *
   * 進行は **run の成功でのみ**起きる（ホスト側の手動加算は API にしない — 二重簿記の禁止）。
   * 汚染後・device 消失後は読めない（追記 3 の「以後の全操作 fail loudly」と決定 7 — どちらも
   * 背後の物理 state は回復不能なので、この数値を「ここから再開できる」と読ませない。読めるのは
   * 「どこまで進んだか」であって「そこから続けられるか」ではなく、区別できない形で返すと
   * ホストは必ず後者として使う）。
   */
  get pastLength(): number {
    this.#assertUsable("pastLength");
    return this.#pastLength;
  }

  /**
   * 論理位置を切り詰める（ADR 0066 決定 6）。`0 ≤ position ≤ pastLength` の整数のみ。
   *
   * MUST: **進行中の generation run が居る間は fail loudly**。run は頭で捕捉した `P` で
   * uniform を書き・dispatch 数を算出し・成功時に `advance` するので、その途中で論理長を横から
   * 動かすと「GPU が読んだ P」と「進行の基準にした P」が分裂する（例外は出ず、KV の論理位置
   * だけが静かにずれる）。リースは `Session.run` の**同期区間**で立つので、`run()` を await せず
   * 直後に呼んだ形も捕まる。
   * MUST: **sliding スロットを 1 本でも含む context は全拒否**（ADR 0066 追記 2）。ring は
   * エビクトが起きた後、resident な位置への巻き戻しでも物理配置と論理範囲が一致しない
   * （左詰め compaction を持たないため）。ORT GenAI が同じ理由で current 未満への rewind を
   * 全拒否しているのと同じ契約で、緩めるなら compaction の実装と対にする。
   */
  rewind(position: number): void {
    this.#assertUsable("rewind");
    if (this.#runs > 0) {
      throw new ExecutionError(
        `rewind: 進行中の generation run が ${this.#runs} 本ある間は巻き戻せない` +
          "（run は頭で捕捉した pastLength で uniform と dispatch 数を決めるので、横から動かすと" +
          "GPU が見た論理長と進行の基準が分裂する）。run の決着を await してから呼ぶこと",
      );
    }
    if (this.#slidingSlots.size > 0) {
      throw new ExecutionError(
        `rewind: sliding スロット [${[...this.#slidingSlots].join(", ")}] を含む context は` +
          "巻き戻せない（ring はエビクト後に物理配置と論理範囲が一致しないため — ADR 0066 " +
          "追記 2）。有効なのは全スロットが非 sliding の context だけで、復旧は新しい context",
      );
    }
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
   * MUST: **2 段**にする。同期に立つのは 1 段目（新規受付の終了 — 以後の run admission と
   * 利用者面の読み書きを拒否）だけで、内部面（論理長の搬送・進行）の遮断は Session チェーンに
   * 積んだ破棄本体が走り出した 2 段目。dispose 本体は先行 run の**後**に走るので、既に受理された
   * run はここで殺されず完走する（`run(); context.dispose();` の非 await 並びが Session の
   * `run(); session.dispose();` と同じ意味論になる）。1 段で閉じると、受理済み run の
   * `writeLengths` / `advance` が「dispose 済み」で落ちる。
   * MUST: 2 度目以降も同じ完了を返す（`Session.dispose` と同じ理由 — 先に返すと呼び手が
   * 「破棄済み」と見なして `device.destroy()` まで進み、flush-before-destroy が崩れる）。
   * MUST: flush が失敗（主因は device 消失）してもバッファ破棄と簿記の返却は必ず行い、失敗
   * 自体は握り潰さず後始末の後に伝播させる（`RunArena.#destroyOnce` と同じ規律）。
   * NOTE: Session より長生きしてよい。この経路が触るのは注入された flush と自分のバッファだけで、
   * Session の重み・計画キャッシュには手を出さない（順序の依存を作らない）。
   */
  dispose(): Promise<void> {
    this.#disposeRequested = true;
    this.#disposal ??= this.#host.serialize(async () => {
      this.#disposed = true;
      try {
        await this.#host.flush();
      } finally {
        for (const slot of this.#slots.values()) slot.buffer.destroy();
        this.#lengths.destroy();
        // MUST: 焼いた束もここで手放す。以後 run は来ない（`#assertUsable` が落とす）ので
        // 正しさには効かないが、掴んだままだと破棄済みバッファを参照する bind group が
        // context の参照ぶんだけ生き残る。
        this.#baked = undefined;
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
   * 不能」の両方を踏む。実体は context 所有のこのバッファ 1 本きりで、レシピからは固定束縛
   * （`BindingSource` の `lengths`）で参照する。
   * MUST: 呼ぶのは**毎 run の encode 前**（`queue.writeBuffer` は issue 順で queue timeline に
   * 載るので、submit 済みの dispatch を追い越さない — ADR 0004 不変条件④）。
   */
  #writeLengths(pastLength: number, queryLength: number): void {
    this.#assertInternalUsable("writeLengths");
    this.#assertCapturedPast(pastLength, "writeLengths");
    this.#assertQueryLength(queryLength, "writeLengths");
    this.#lengthValues[0] = this.#pastLength;
    this.#lengthValues[1] = queryLength;
    this.#host.gpu.device.queue.writeBuffer(this.#lengths, 0, this.#lengthValues);
  }

  /**
   * 論理長を進める（ADR 0066 決定 6 — **run の成功でのみ**呼ぶ）。
   *
   * NOTE: full スロットの実行時検査 `pastLength + queryLength ≤ 容量`（ADR 0067 決定 4 の④）は
   * **run のエンコード前**に居る（`assertGenerationRun` — src/runtime/recipe.ts）。容量軸がどの
   * 次元かを決めるのは op 契約なので、導出相が集めた {@link GenerationLimits} が正本で、
   * ここでは重ねて見ない（進行の時点で検査しても、既に書かれた後で手遅れになる）。
   */
  #advance(pastLength: number, queryLength: number): void {
    this.#assertInternalUsable("advance");
    this.#assertCapturedPast(pastLength, "advance");
    this.#assertQueryLength(queryLength, "advance");
    const next = this.#pastLength + queryLength;
    // MUST: 加算後の論理長も u32 に収まること。両項が u32 以下でも和は溢れるので、ここが
    // 唯一の検査点になる（超えたまま代入すると、次の writeLengths が切り詰めた past を
    // 沈黙のまま GPU に載せる — {@link MAX_LOGICAL_LENGTH}）。
    if (next > MAX_LOGICAL_LENGTH) {
      throw new ExecutionError(
        `advance: pastLength ${this.#pastLength} + queryLength ${queryLength} = ${next} が ` +
          `u32 の上限 ${MAX_LOGICAL_LENGTH} を超える（論理長の搬送先は u32 — ADR 0066 追記 4）`,
      );
    }
    this.#pastLength = next;
  }

  /**
   * 汚染する（ADR 0066 追記 3 — state 変更 dispatch を含む run の失敗）。
   *
   * 論理長は進まないが物理 ring は上書きされ得るので、rollback / staging を持たない設計では
   * 「以後の全操作を拒否」以外に整合を主張する手段が無い（復旧 = 新しい context + ホスト側
   * 再構築）。トリガは run の失敗経路（executor の `#poisonOnStateWrite`）。
   *
   * MUST: 2 度目以降は最初の理由を保つ（真因を後続の失敗で上書きしない）。
   */
  #poison(reason: string): void {
    this.#poisoned ??= reason;
  }

  /**
   * **利用者面**の使用可否（読み取りを含む全操作 — 例外は後始末の
   * {@link GenerationContext.dispose} だけ）。dispose 要求・device 消失・汚染をここで落とす。
   *
   * MUST: 読みと書きで条件を分けない。汚染も device 消失も「背後の物理 state が失われた」状態で、
   * そこで論理長だけを返せるようにすると復旧不能な context が正常値を持つ器に見える
   * （追記 3 の「以後の全操作 fail loudly」）。
   * MUST: 見るのは dispose の**1 段目**（受付終了）。run admission もこの面なので、
   * `dispose()` の後に発行された run はここで拒否される。
   */
  #assertUsable(where: string): void {
    if (this.#disposeRequested) {
      throw new ExecutionError(`${where}: dispose 済みの GenerationContext は使えない`);
    }
    this.#assertLive(where);
  }

  /**
   * **内部面**（論理長の搬送・進行・読み）の使用可否。見るのは dispose の**2 段目**（破棄本体が
   * 走り出したか）で、1 段目と 2 段目の間に居る受理済み run は通す。
   */
  #assertInternalUsable(where: string): void {
    if (this.#disposed) {
      throw new ExecutionError(`${where}: dispose 済みの GenerationContext は使えない`);
    }
    this.#assertLive(where);
  }

  /** 2 つの遮断面が共有する「背後の物理 state が生きているか」。 */
  #assertLive(where: string): void {
    assertDeviceUsable(this.#host.gpu, where);
    if (this.#poisoned !== undefined) {
      throw new ExecutionError(
        `${where}: 汚染された GenerationContext は使えない（${this.#poisoned}）。` +
          "復旧は新しい context + ホスト側の state 再構築",
      );
    }
  }

  /**
   * run が頭で捕捉した `pastLength` が今の論理長と一致すること。
   *
   * MUST: uniform へ書く値・dispatch 数の算出・容量の検査・進行が**同じ 1 つの P** から出るのが
   * states 形の前提（ADR 0066 決定 3 / 追記 4）。リースがあれば横から動く経路は塞がっているので、
   * ここが割れたら実装の不変条件破れ — 沈黙で続けさせず即死させる。
   */
  #assertCapturedPast(pastLength: number, where: string): void {
    if (pastLength !== this.#pastLength) {
      throw new ExecutionError(
        `${where}: run が捕捉した pastLength ${pastLength} が context の現在値 ` +
          `${this.#pastLength} と食い違う（進行中 run の論理長が横から動いた — 内部の不変条件破れ）`,
      );
    }
  }

  /**
   * 今 step の実 token 数の検査（ADR 0066 決定 4 の 2 つの実行形に共通）。
   *
   * prefill-chunk は `queryLength ≤ chunkLength`（超えた行は物理 shape に載らない）、decode は
   * `queryLength = 1`。0 を許さないのは「何も進めない run」が state の書き込み範囲を空にして、
   * 進行と物理内容の対応が観測できなくなるため。
   *
   * NOTE: u32 の上限（{@link MAX_LOGICAL_LENGTH}）はここで重ねて見ない — `chunkLength` が
   * create で u32 以下に絞られているので `queryLength ≤ chunkLength` がそのまま上限になる
   * （`rewind` の `position ≤ pastLength` も同じ形で上限が伝わる）。
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
