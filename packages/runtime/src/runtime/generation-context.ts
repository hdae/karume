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
  describeDeviceLoss,
  type GpuContext,
  GpuDeviceLostError,
  RUNTIME_INTERNAL,
} from "../gpu/device.ts";
import { discardFailureScopes, popFailureScopes, pushFailureScopes } from "../gpu/error-scope.ts";
import { BUFFER_USAGE } from "../gpu/webgpu-constants.ts";
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
export const LENGTHS_BYTES: number = 8;

/**
 * 論理長 uniform の usage。
 *
 * `COPY_SRC` を付けるのは**内容を読み戻せる唯一の観測点**を残すため — `queue.writeBuffer` は
 * 無効なバッファや整列違反に対して警告すら出さない no-op になるので、「書いた値が実際に
 * 載っているか」は読み戻す以外に確かめる手段が無い（実行経路はこのコピーを出さない）。
 */
const LENGTHS_USAGE = BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC;

/** state スロットの dtype は f32 のみ（`STATE_DTYPES` — f16 席は ADR 0066 追記 5 の予約）。 */
export const STATE_ELEMENT_BYTES: number = 4;

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
      `${where}: device が失われた（生成は失われる — device を取り直して作り直すこと）` +
        describeDeviceLoss(gpu.lost),
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
  /**
   * この context が**借り手**か（ADR 0096 段 2 §2.1）。真のとき論理長は自分では進まず、
   * スロットの実体も lengths 以外は貸し手のもの。
   *
   * MUST: 実行統合（`Session.run`）はこの 1 欄で「進行させない」を判断する。借り手側の
   * `advance` / `defer` を黙って no-op にすると、貸し手の P だけが動く形と区別が付かない。
   */
  readonly borrowing: boolean;
  /** この context の device（借用時の同一 device 照合 — 別 device のスロットは束縛できない）。 */
  readonly gpu: GpuContext;
  /**
   * sliding なスロット名 → 窓幅（借用時に**貸し手と借り手で一致**を見るための面）。
   *
   * MUST: 論理 col → 物理 row の写像は読み書き同式（ADR 0067 決定 4）。グラフ内の一致は
   * `validateGraphContracts` が見るが、貸し借りは**2 つのグラフに跨る**ので、ここで突き合わせる
   * 以外に検出点が無い（借り手の窓だけ広いと、窓の外に落ちた行を過去として読む）。
   */
  readonly slidingSlots: ReadonlyMap<string, number>;
  /** 貸し手として使えるか（借り手 context の生成時に 1 度だけ）。 */
  assertLendable(where: string): void;
  /** 借り手を登録する（生きている間は貸し手の `dispose` を拒否する）。 */
  registerBorrower(borrower: GenerationContext): void;
  /** 借り手の登録を外す（借り手の `dispose`）。 */
  forgetBorrower(borrower: GenerationContext): void;
  /**
   * この context が許す物理 chunk 行数の集合 = `{1} ∪ chunkBuckets ∪ {chunkLength}`
   * （run 前検査 `assertGenerationRun` — src/runtime/recipe.ts が読む唯一の形）。
   *
   * MUST: 構築は context 生成時の 1 度きり。run ごとに組み直すと、decode のホットパスに
   * バケット本数ぶんの Set 構築が毎 step 乗る（実行形の本数はここでは固定なのに）。
   */
  readonly allowedRows: ReadonlySet<number>;
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
   * 必ず返す）。取れなければ fail loudly（dispose 要求後・汚染後・device 消失後、および
   * **同一 context に未決着 run が既に 1 本ある**とき — {@link GenerationContext.rewind} の doc と
   * 同じ「論理長が横から動く」形が、2 本目の発行そのもので起きる）。
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
  /** 焼いた束を預ける（`token` の backing が保持されている間だけ引ける）。 */
  setBakedGroups(token: number, groups: BakedGroups): void;
  /**
   * `token` の backing の束を捨てる（Session が backing を退役させるときに呼ぶ）。
   * MUST: 退役と同時に捨てる — 束は退役した実体を掴んでいるので、残すと参照ぶんの寿命が延びる。
   */
  dropBakedGroups(token: number): void;
  /**
   * 論理長を書き出す（毎 run の encode 前 — {@link GenerationContext} の doc）。
   * `pastLength` は**呼び出し側が run の頭で捕捉した値**（内部の現在値との一致を照合する）。
   */
  writeLengths(pastLength: number, queryLength: number): void;
  /** 論理長を進める（**run の成功でのみ** — 決定 6）。捕捉 P の照合は `writeLengths` と同じ。 */
  advance(pastLength: number, queryLength: number): void;
  /**
   * 論理長を**進めずに保留する**（`commit: "deferred"` の run が例外なく返ったとき）。
   *
   * 進行の権利をホストへ 1 度だけ渡す形で、渡した先は {@link GenerationContext.commit}。
   * 保留がある間は新しい run のリースと `rewind` を拒否するので、「GPU が見た論理長」と
   * 「進行の基準」が割れる窓は開かない（ADR 0066 決定 6 の二重簿記の禁止はこの形でも保たれる —
   * 論理長を動かせるのは依然 1 経路だけ）。捕捉 P の照合は `advance` と同じ。
   */
  defer(pastLength: number, queryLength: number): void;
  /** 汚染する（state 変更 dispatch を含む run の失敗 — 追記 3）。 */
  poison(reason: string): void;
};

/**
 * sliding なスロットの名前 → 窓幅 `W`（ノード attrs `window` 由来 — ADR 0067 決定 4）。
 *
 * MUST: 判定材料はノード側にしかない（`graph.states` の宣言は容量だけを持ち、窓は
 * **参照するノード**が宣言する）。同一スロットに触れる全ノードで `window` が一致することは
 * `validateGraphContracts` の `assertStateOrder` が Session 構築時に済ませているので、
 * ここは 1 本でも sliding 宣言があれば sliding として拾えばよい。
 */
const slidingSlotWindows = (graph: IrGraph): ReadonlyMap<string, number> => {
  const sliding = new Map<string, number>();
  graph.nodes.forEach((node, index) => {
    const slots = Object.values(node.states);
    if (slots.length === 0) return;
    const window = stateWindow(node.attrs, `nodes[${index}] (${node.op})`);
    if (window === undefined) return;
    for (const slot of slots) sliding.set(slot, window);
  });
  return sliding;
};

/**
 * sliding スロットの**余裕**（`capacity − window` の最小 — sliding が 1 本も無ければ undefined）。
 *
 * 余裕は「論理長より先に書かれた行」の置き場で、ring の法が窓ではなく容量であること
 * （`src/kernels/state-attention.ts` の `stateSlotRowWgsl`）と対で意味を持つ。ホストは
 * この数を超える行数を投機的に書いてはいけない — 超えると棄却行が live な過去 KV を潰す。
 *
 * MUST: 容量軸は**スロット shape の軸 2**（`[B,Hkv,C,D]` — states 形 op の契約。実行計画側
 * `recipe-builder.ts` の `#buildStateAttention` / `#buildStateAppend` も同じ軸を読む）。
 * ここで別の軸を読むと、公開する余裕が実際の物理行数と無関係な数になる。
 */
const slidingSlackRows = (
  slots: ReadonlyMap<string, StateSlotBacking>,
  windows: ReadonlyMap<string, number>,
): number | undefined => {
  let slack: number | undefined;
  for (const [name, window] of windows) {
    // 参照完全性（states 宣言とノードの states 欄の対応）は IR 層が済ませているので、名前は
    // 必ず引ける。rank が足りない形は shape 層が run で落とすが、ここで黙って飛ばすと
    // 「余裕なし」と区別の付かない undefined か、別軸由来の過大な余裕を公開してしまう。
    const shape = slots.get(name)?.shape;
    if (shape === undefined || shape.length < 3) {
      throw new ExecutionError(
        `state '${name}': sliding 宣言（window ${window}）に対して容量形 ` +
          `[${shape?.join(",") ?? "?"}] から行容量 C（軸 2）が読めない`,
      );
    }
    const rows = shape[2] - window;
    if (rows < 0) {
      throw new ExecutionError(
        `state '${name}': window ${window} が行容量 ${shape[2]} を超える（ADR 0067 決定 4 ③）`,
      );
    }
    slack = slack === undefined ? rows : Math.min(slack, rows);
  }
  return slack;
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
/**
 * `chunkLength` の値域検査（GPU 非依存の純関数）。
 *
 * MUST: 上限は u32（`queryLength ≤ chunkLength` の門を通じて論理長の上限もここで決まる —
 * {@link MAX_LOGICAL_LENGTH}）。estimator（estimate.ts）も同じ門を通す — 実構築が拒否する
 * 指定に見積りだけが正常値を返すと、作れない構成へ admission の数字が与えられる。
 */
export const assertChunkLength = (chunkLength: number): void => {
  if (
    !Number.isSafeInteger(chunkLength) || chunkLength < 1 ||
    chunkLength > MAX_LOGICAL_LENGTH
  ) {
    throw new ExecutionError(
      `chunkLength ${chunkLength} が 1..${MAX_LOGICAL_LENGTH} の整数でない` +
        "（固定長 prefill chunk の行数・搬送先は u32 — ADR 0066 決定 4 / 追記 4）",
    );
  }
};

/**
 * `chunkBuckets` の値域・順序検査（GPU 非依存の純関数 — {@link assertChunkLength} と同じ流儀）。
 *
 * 受理するのは「2 以上 `chunkLength` 未満の整数の**狭義昇順**列」だけ。3 つの拒否理由:
 * `1` は decode 形そのもので追加の実行形にならない・`chunkLength` 以上は
 * {@link GenerationContext} の `queryLength ≤ chunkLength` 契約の外（宣言 shape に載らない行が
 * 出る）・重複と降順は「`queryLength` 以上の最小バケット」という選び方が線形走査で決まらなく
 * なる（呼び出し側が並べ替えを持つと、context が許す集合と選ぶ集合が別々に育つ）。
 *
 * MUST: 見積り（estimate.ts）も同じ門を通す — 実構築が拒否する指定に見積りだけが正常値を
 * 返すと、作れない構成へ admission の数字が与えられる。
 */
export const assertChunkBuckets = (
  chunkBuckets: readonly number[] | undefined,
  chunkLength: number,
): void => {
  if (chunkBuckets === undefined) return;
  let previous = 1;
  for (const [index, rows] of chunkBuckets.entries()) {
    if (!Number.isSafeInteger(rows) || rows < 2 || rows >= chunkLength) {
      throw new ExecutionError(
        `chunkBuckets[${index}] ${rows} が 2..${chunkLength - 1} の整数でない` +
          "（1 は decode 形・chunkLength は prefill 形そのもの — ADR 0066 決定 4 / 追記〈バケット〉）",
      );
    }
    if (rows <= previous) {
      throw new ExecutionError(
        `chunkBuckets[${index}] ${rows} が直前の ${previous} 以下（狭義昇順でない）` +
          "。queryLength 以上の最小バケットを選ぶ側が並べ替えを持たない前提",
      );
    }
    previous = rows;
  }
};

export const resolveBindings = (
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
export const resolveSlotShape = (
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

/**
 * 借り手 context が抱える借用の状態（ADR 0096 段 2 §2.1）。**貸し手の実体そのもの**を持つ
 * （写しではない — 写すと「どちらが本物か」が生まれる）。
 */
type BorrowState = {
  readonly lender: GenerationContext;
  /** 貸し手の `slidingSlack` の写し（借り手は自前のスロットを持たないので導出できない）。 */
  readonly slidingSlack: number | undefined;
};

export class GenerationContext {
  /**
   * 固定長 prefill chunk の行数（ADR 0066 決定 4 — context 生成時に確定する計画時定数）。
   * decode は `queryLength = 1` 固定形で、prefill の末尾 chunk は pad で埋める（切るのは
   * `queryLength`）。
   */
  readonly chunkLength: number;
  /**
   * prefill 形として `chunkLength` に加えて許す物理 chunk 行数（狭義昇順・凍結コピー —
   * ADR 0066 決定 4 / 追記〈バケット〉）。宣言していなければ空。
   *
   * 呼び出し側は chunk ごとに `queryLength` 以上の最小の値を物理行数に選ぶ（無ければ
   * `chunkLength`）。`chunkLength` はバケットを足しても**最大値のまま**で、
   * `queryLength ≤ chunkLength` の上限も動かない。
   */
  readonly chunkBuckets: readonly number[];
  /**
   * sliding スロットの**余裕行数** = `capacity − window` の最小（sliding が 1 本も無ければ
   * undefined）。ring の法は窓ではなく容量（`src/kernels/state-attention.ts` の
   * `stateSlotRowWgsl`）なので、この行数までは**論理長より先に**物理 ring へ書いても、棄却して
   * 良い（= `commit` で受理しない）行が live な過去 KV を潰さない。
   *
   * 投機デコード（draft を検証してから受理行数を確定する形）の `queryLength` の上限がこれで、
   * 超えた run は例外を出さずに過去 KV を壊す — 上限の執行はホスト側の責務（ランタイムは
   * `queryLength ≤ chunkLength` までしか見ない）。
   */
  readonly slidingSlack: number | undefined;
  /** ランタイム内部面（利用者が触る面ではない）。 */
  readonly [RUNTIME_INTERNAL]: GenerationContextInternals;
  readonly #host: GenerationContextHost;
  readonly #slots: ReadonlyMap<string, StateSlotBacking>;
  /** 借用の状態（undefined = 自前のスロットを持つ通常の context）。 */
  readonly #borrow: BorrowState | undefined;
  /**
   * この context を借りている context の集合（ADR 0096 段 2 §2.1）。
   *
   * MUST: 生きている間の `dispose()` は fail loudly。借り手の bind group は貸し手のスロット
   * バッファを掴んでいるので、破棄すると次の draft が破棄済みバッファを読む。
   */
  readonly #borrowers = new Set<GenerationContext>();
  /**
   * sliding なスロット名 → 窓幅（{@link GenerationContext.rewind} の全拒否条件 —
   * ADR 0066 追記 2。窓幅は {@link GenerationContext.slidingSlack} の算出にも使う）。
   */
  readonly #slidingSlots: ReadonlyMap<string, number>;
  readonly #lengths: GPUBuffer;
  /**
   * 論理長の書き出し値。**全域を毎回書く**（部分書きにすると、片方だけ更新された組が残って
   * 「pastLength は新しく queryLength は前 step」という混ざった uniform で dispatch が回る）。
   */
  readonly #lengthValues = new Uint32Array(2);
  #pastLength = 0;
  /**
   * commit 待ちの deferred run（{@link GenerationContext.commit} が畳むまで残る）。
   *
   * MUST: 立っている間は run のリースと `rewind` を拒否する。物理 ring には `queryLength` 行が
   * 既に書かれていて論理長だけが止まっている状態で、次の run や巻き戻しを通すと「どこまでが
   * 確定した KV か」を 2 箇所が別々に決めることになる。
   */
  #pending: { readonly pastLength: number; readonly queryLength: number } | undefined;
  /**
   * context 側で焼いた bind group 束（backing の世代識別子 → 束 — ADR 0066 決定 5）。Session が
   * backing を複数保持する（perf-ledger H-15）ので、保持中の backing ごとに 1 束を持ち、退役した
   * backing の束は Session が `dropBakedGroups` で捨てる。**GPUBindGroup は destroy 不要**
   * （GC 任せ）だが、掴んでいる backing 所有のバッファの寿命を延ばすので、退役と同時に手放す。
   */
  readonly #baked = new Map<number, BakedGroups>();
  /** 汚染の理由（追記 3）。立つと `dispose` 以外の全操作を拒否する（読みも含む）。 */
  #poisoned: string | undefined;
  /**
   * 進行中の generation run の本数（**リース**）。`Session.run` の同期区間で取り、run の決着で
   * 返す。0 でない間は {@link GenerationContext.rewind} を拒否する。
   *
   * MUST: 取れるのは高々 1 本（2 本目の発行は `acquireRun` が拒否する）。本数のまま持つのは
   * 「返し損ね」を releaseRun が簿記の破れとして落とせるようにするため。
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
    slidingSlots: ReadonlyMap<string, number>,
    lengths: GPUBuffer,
    chunkLength: number,
    chunkBuckets: readonly number[],
    bindings: SymbolBindings,
    borrow?: BorrowState,
  ) {
    this.#host = host;
    this.#slots = slots;
    this.#slidingSlots = slidingSlots;
    this.#borrow = borrow;
    // 借り手の余裕は**貸し手の写し**（ADR 0096 段 2 §2.1）。自前で導出すると、貸し手が
    // sliding 宣言を持つのに借り手の readonly ノードが window を宣言していない形で
    // 「余裕なし」を名乗り、投機の上限がホスト側で緩む。
    this.slidingSlack = borrow === undefined
      ? slidingSlackRows(slots, slidingSlots)
      : borrow.slidingSlack;
    this.#lengths = lengths;
    this.chunkLength = chunkLength;
    // 凍結コピー: 呼び出し側の配列を後から書き換えられると、許可集合（下）と公開面が割れる。
    this.chunkBuckets = Object.freeze([...chunkBuckets]);
    this[RUNTIME_INTERNAL] = {
      slots,
      borrowing: borrow !== undefined,
      gpu: host.gpu,
      slidingSlots,
      // 昇順のまま入れる（`assertGenerationRun` の診断がこの反復順をそのまま列挙する）。
      allowedRows: new Set([1, ...this.chunkBuckets, chunkLength]),
      lengths,
      // 容量は確定済み（静的物理格納 — ADR 0066 決定 3）なので、ここで 1 度畳んで持つ。
      // MUST: 借り手はスロットのバイト数を数えない（実体は貸し手 context の所有物で、
      // 両方が数えると診断 `stateBacking.residentBytes` が同じ VRAM を二重計上する）。
      bytes: borrow === undefined
        ? [...slots.values()].reduce((total, slot) => total + slot.byteLength, 0) + LENGTHS_BYTES
        : LENGTHS_BYTES,
      bindings,
      assertLendable: (where: string): void => {
        this.#assertUsable(where);
        if (this.#borrow !== undefined) {
          throw new ExecutionError(
            `${where}: 借り手 context は貸し手になれない（借用の連鎖は持たない — ` +
              "ADR 0096 段 2 §2.1）",
          );
        }
      },
      registerBorrower: (borrower: GenerationContext): void => {
        this.#assertUsable("createGenerationContext(borrow)");
        this.#borrowers.add(borrower);
      },
      forgetBorrower: (borrower: GenerationContext): void => {
        this.#borrowers.delete(borrower);
      },
      acquireRun: (): void => {
        this.#assertUsable("run");
        // MUST: 未 commit の deferred run がある間は次を発行させない。2 本目は「1 本目が
        // 書いた物理行のうちどこまでが確定か」が決まらないまま P を捕捉するので、commit(rows)
        // が後から論理長を動かした時点で GPU が見た P と食い違う（例外の出ない位置ずれ）。
        if (this.#pending !== undefined) {
          throw new ExecutionError(
            `run: commit 待ちの generation run がある（pastLength ${this.#pending.pastLength} + ` +
              `queryLength ${this.#pending.queryLength} まで書き込み済み）。` +
              "context.commit(rows) で受理した行数を確定させてから次を発行すること",
          );
        }
        // MUST: 同一 context への未決着 run は 1 本まで。2 本目は 1 本目が進めた論理長 P' で
        // uniform と dispatch を組むが、位置入力（RoPE の position_ids 等）は**呼び出し側が
        // 発行時に組んだ通常のグラフ入力**で、ランタイムは中身を見ない。つまり KV の論理長は
        // 正しいまま位置だけが静かにずれる（例外も警告も出ない沈黙誤値）。1 本ずつ await して
        // 発行すること — 並行させたい生成は context を分ける。
        if (this.#runs > 0) {
          throw new ExecutionError(
            "run: 進行中の generation run がある GenerationContext へ並行発行された" +
              "（2 本目は 1 本目の進行後の論理長で走る一方、位置入力は発行時の論理長のままなので、" +
              "例外の出ない位置ずれになる）。前の run の決着を await してから発行するか、" +
              "context を分けること",
          );
        }
        // MUST: 借り手の run は**貸し手のリースも**取る（ADR 0096 段 2 §2.1 — 既存 acquireRun と
        // 同じ席）。これで貸し手の run / commit / rewind と直列化され、貸し手の poison・未 commit
        // の deferred run・進行中 run がそのまま借り手の拒否理由になる。
        // MUST: 自分の検査を全て通してから取る（取ってから落ちると、返し手の居ないリースが
        // 貸し手に 1 本残って以後の rewind / commit が永久に拒否される）。
        this.#borrow?.lender[RUNTIME_INTERNAL].acquireRun();
        this.#runs += 1;
      },
      releaseRun: (): void => {
        if (this.#runs < 1) {
          throw new ExecutionError(
            "releaseRun: 進行中の generation run が居ないのにリースを返した（簿記の破れ）",
          );
        }
        this.#runs -= 1;
        this.#borrow?.lender[RUNTIME_INTERNAL].releaseRun();
      },
      pastLength: (): number => {
        this.#assertInternalUsable("pastLength");
        // 借り手の論理長は**貸し手の P の写し**（ADR 0096 段 2 §2.1）。写すのは run の頭の
        // この 1 点だけで、以後の `writeLengths` はこの値との一致を照合する（リースを握って
        // いる間は貸し手の P が動かないので、写しと現物は run の決着まで一致し続ける）。
        if (this.#borrow !== undefined) {
          this.#pastLength = this.#borrow.lender[RUNTIME_INTERNAL].pastLength();
        }
        return this.#pastLength;
      },
      bakedGroups: (token: number): BakedGroups | undefined => this.#baked.get(token),
      setBakedGroups: (token: number, groups: BakedGroups): void => {
        this.#baked.set(token, groups);
      },
      dropBakedGroups: (token: number): void => {
        this.#baked.delete(token);
      },
      writeLengths: (pastLength: number, queryLength: number): void =>
        this.#writeLengths(pastLength, queryLength),
      advance: (pastLength: number, queryLength: number): void =>
        this.#advance(pastLength, queryLength),
      defer: (pastLength: number, queryLength: number): void =>
        this.#defer(pastLength, queryLength),
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
    assertChunkLength(spec.chunkLength);
    // MUST: 検査した実体をそのまま持ち回る（`spec` から読み直さない）。検査点と下の
    // constructor 渡しの間には確保の await（`raceDeviceLost`）があり、その窓で呼び手が渡した
    // 配列を書き換えると、未検査の M が許可集合に載る（TOCTOU）。
    const chunkBuckets = Object.freeze([...(spec.chunkBuckets ?? [])]);
    assertChunkBuckets(chunkBuckets, spec.chunkLength);
    // 借り物スロット（external — ADR 0096 段 2 §1.1）と `borrow` は**対**。片方だけの形は
    // どちらの向きも fail loudly（external があるのに自前確保すると空の過去を読み、borrow だけ
    // なら誰も読まないスロットを貸し手から掴む）。
    const external = names.filter((name) => graph.states[name].external);
    if (external.length > 0 && spec.borrow === undefined) {
      throw new ExecutionError(
        `このグラフは external な state スロット [${external.join(", ")}] を持つ` +
          "（借り物の実体は貸し手 context にあるので createGenerationContext({ borrow }) が要る" +
          " — ADR 0096 段 2 §2.1）",
      );
    }
    if (spec.borrow !== undefined) {
      if (external.length === 0) {
        throw new ExecutionError(
          "borrow を指定できるのは全スロットが external なグラフだけ" +
            `（自前スロット [${names.join(", ")}] を持つ — ADR 0096 段 2 §2.1）`,
        );
      }
      return await GenerationContext.#createBorrowed(host, spec, spec.borrow, names);
    }
    const bindings = resolveBindings(graph, spec.bindings);
    // MUST: 上限は 2 本とも見る。`maxStorageBufferBindingSize ≤ maxBufferSize` は device を計画
    // する側（gpu/device.ts の `planRequiredLimits`）が保っている関係であって、外から渡された
    // GpuContext にまで効く保証ではない — 束縛上限だけを見る形にすると、関係が崩れた device で
    // 「確保そのものが通らない大きさ」を素通しして無効バッファを掴む。
    const limits = [
      ["maxStorageBufferBindingSize", gpu.limits.maxStorageBufferBindingSize],
      ["maxBufferSize", gpu.limits.maxBufferSize],
    ] as const;
    const planned = names.map((name) => {
      const shape = resolveSlotShape(name, graph.states[name].shape, bindings);
      const byteLength = numel(shape) * STATE_ELEMENT_BYTES;
      // MUST: スロット単体のバイト数が上限を超える容量指定は fail loudly（追記 5）。
      // 分割して束ねる形は持たない（KV は連続容量 — 決定 8 の明示選択）ので、超過は容量設計の
      // 誤りとして呼び出し点で落とす以外に手が無い。
      for (const [limitName, limit] of limits) {
        if (byteLength > limit) {
          throw new ExecutionError(
            `state '${name}': 容量 [${shape.join(",")}] の ${byteLength} バイトが ` +
              `${limitName} ${limit} バイトを超える（ADR 0066 追記 5）。` +
              "容量を下げるか、スロットを分けてグラフを組み直すこと",
          );
        }
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
        slidingSlotWindows(graph),
        lengths,
        spec.chunkLength,
        chunkBuckets,
        bindings,
      );
      return context;
    } finally {
      // MUST: 後始末の失敗で本体の例外を上書きしない（Session 構築と同じ規律）。push した
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
   * **借り手 context** を作る（ADR 0096 段 2 §2.1 — drafter が target の KV を読む形）。
   *
   * 確保するのは論理長 uniform 1 枚だけで、スロットは貸し手の実体をそのまま束ねる。束ね方は
   * **名前**（借り手の宣言名 = 貸し手の宣言名 MUST）で、形は「借り手の宣言 shape を**貸し手の
   * bindings**（同名記号 MUST — 容量 `C`）で解いた値」が貸し手スロットの実形と一致すること。
   *
   * MUST: `bindings` を受けない（貸し手のものを継承する — 2 つの束縛点を持つと、容量記号が
   * 割れたまま「貸し手の 131072 行のスロットを 512 行として読む」形が例外なしに成立する）。
   * MUST: `chunkLength` は 1 ちょうど（借り手の実行形は decode 1 本だけ = readonly attention は
   * M 1 固定 — §1.2 の形検査と対）。`chunkBuckets` の禁止はここに書かない — `chunkLength = 1`
   * では既存の {@link assertChunkBuckets}（要素は `2..chunkLength−1`）が空以外を全て落とすので、
   * 重ねて書くと到達しない分岐になる。
   */
  static async #createBorrowed(
    host: GenerationContextHost,
    spec: GenerationContextSpec,
    lender: GenerationContext,
    names: readonly string[],
  ): Promise<GenerationContext> {
    const { gpu, graph } = host;
    const internals = lender[RUNTIME_INTERNAL];
    internals.assertLendable("createGenerationContext(borrow)");
    if (internals.gpu !== gpu) {
      throw new ExecutionError(
        "createGenerationContext(borrow): 貸し手 context と GpuContext（device）が別" +
          "（別 device のスロットは束縛できない）",
      );
    }
    if (spec.chunkLength !== 1) {
      throw new ExecutionError(
        `createGenerationContext(borrow): chunkLength ${spec.chunkLength} は 1 ちょうど` +
          "（借り手の実行形は decode 1 本だけ — ADR 0096 段 2 §2.1）",
      );
    }
    if (spec.bindings !== undefined) {
      throw new ExecutionError(
        "createGenerationContext(borrow): bindings は貸し手のものを継承する（渡せない）" +
          "— 2 つの束縛点を持つと容量記号が割れたまま別容量のスロットを読む",
      );
    }
    const bindings = internals.bindings;
    const windows = slidingSlotWindows(graph);
    const slots = new Map<string, StateSlotBacking>();
    for (const name of names) {
      const backing = internals.slots.get(name);
      if (backing === undefined) {
        throw new ExecutionError(
          `state '${name}': 貸し手 context に同名のスロットが無い` +
            "（external スロットは貸し手と同じ名前で宣言する MUST — ADR 0096 段 2 §2.1）",
        );
      }
      // MUST: 窓は貸し手と**存在有無も値も**一致（読み書き同式 — ADR 0067 決定 4 を貸し借りへ
      // 延長した面）。借り手の窓だけ広いと、貸し手が既に上書きした行を過去として読む。
      const window = windows.get(name);
      const lent = internals.slidingSlots.get(name);
      if (window !== lent) {
        const show = (value: number | undefined): string => value?.toString() ?? "宣言なし";
        throw new ExecutionError(
          `state '${name}': attrs.window が貸し手と食い違う（貸し手 ${show(lent)} / 借り手 ${
            show(window)
          }）— 論理 col → 物理 row の写像は読み書き同式 MUST（ADR 0067 決定 4）`,
        );
      }
      const shape = resolveSlotShape(name, graph.states[name].shape, bindings);
      if (shape.length !== backing.shape.length || shape.some((d, i) => d !== backing.shape[i])) {
        throw new ExecutionError(
          `state '${name}': 借り手の宣言 [${shape.join(",")}]（貸し手の bindings で解決）が` +
            `貸し手スロットの実形 [${backing.shape.join(",")}] と違う`,
        );
      }
      slots.set(name, backing);
    }

    // 確保するのは lengths 1 枚だけ（スロットは借り物）。errorScope の規律は自前確保の経路と
    // 同じ — createBuffer は上限超過でも同期例外を投げないため、囲まないと無効なバッファへ
    // 論理長を書き続ける沈黙 no-op になる。
    pushFailureScopes(gpu.device);
    const where = "GenerationContext（借り手）の論理長確保";
    let popped = false;
    let created: GPUBuffer | undefined;
    let context: GenerationContext | undefined;
    try {
      created = gpu.device.createBuffer({
        label: "generation lengths (borrowed)",
        size: LENGTHS_BYTES,
        usage: LENGTHS_USAGE,
      });
      const pending = popFailureScopes(gpu.device, where);
      popped = true;
      const failure = await gpu[RUNTIME_INTERNAL].raceDeviceLost(pending, where);
      if (failure !== undefined) throw failure;
      const borrowed = new GenerationContext(
        host,
        slots,
        slidingSlotWindows(graph),
        created,
        spec.chunkLength,
        [],
        bindings,
        { lender, slidingSlack: lender.slidingSlack },
      );
      // MUST: 登録は構築の**後**（借り手の実体が出来てから貸し手の dispose を塞ぐ）。ここが
      // 落ちる（await の窓で貸し手が dispose された）なら lengths を返して漏らさない。
      internals.registerBorrower(borrowed);
      context = borrowed;
      return context;
    } finally {
      if (!popped) await discardFailureScopes(gpu.device);
      if (context === undefined) created?.destroy();
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
   * commit 待ちの deferred run（`GenerationRun.commit: "deferred"` — 無ければ undefined）。
   *
   * `pastLength` は run が捕捉した論理長・`queryLength` は物理 ring へ書いた行数で、
   * {@link GenerationContext.commit} が受け取れる `rows` の上限がそのまま `queryLength`。
   */
  get pendingCommit(): { readonly pastLength: number; readonly queryLength: number } | undefined {
    this.#assertUsable("pendingCommit");
    return this.#pending;
  }

  /**
   * 保留中の deferred run のうち**受理した行数だけ**論理長を進める（ADR 0066 決定 6 の
   * 「論理長は run の成功で進む」を投機デコードの検証形へ広げた面）。
   *
   * `rows` は `0 ≤ rows ≤ pendingCommit.queryLength` の整数で、`0` は「1 行も受理しない」
   * （論理長は動かず、保留だけが畳まれる）。物理 ring には `queryLength` 行が書かれたままだが、
   * 受理しなかった行が潰した論理列は sliding の余裕（{@link GenerationContext.slidingSlack}）の
   * 外に落ちる — 余裕を超える `queryLength` を投げないのはホスト側の契約。
   *
   * MUST: **進行中の generation run が居る間は fail loudly**（`rewind` と同じ根拠 — run は頭で
   * 捕捉した P で uniform と dispatch 数を決めるので、横から動かすと GPU が見た論理長と進行の
   * 基準が分裂する）。
   * MUST: 保留が無い呼びは fail loudly（immediate な run の後に呼ばれた commit を黙って
   * no-op にすると、ホストは「受理行数を伝えた」と信じたまま二重に進んだ論理長で走り続ける）。
   */
  commit(rows: number): void {
    this.#assertUsable("commit");
    // 借り手は論理長を持たない（進めるのも確定させるのも貸し手 — ADR 0096 段 2 §2.1）。
    this.#assertNotBorrowing("commit");
    if (this.#runs > 0) {
      throw new ExecutionError(
        `commit: 進行中の generation run が ${this.#runs} 本ある間は確定できない` +
          "（run の決着を await してから呼ぶこと）",
      );
    }
    const pending = this.#pending;
    if (pending === undefined) {
      throw new ExecutionError(
        "commit: 確定待ちの generation run が無い" +
          "（commit を要するのは GenerationRun.commit を 'deferred' で発行した run だけ）",
      );
    }
    if (!Number.isSafeInteger(rows) || rows < 0) {
      throw new ExecutionError(`commit: 受理行数 ${rows} が非負整数でない`);
    }
    if (rows > pending.queryLength) {
      throw new ExecutionError(
        `commit: 受理行数 ${rows} が保留中の run の queryLength ${pending.queryLength} を超える` +
          "（書いていない行は確定できない）",
      );
    }
    // 保留を立てた run 以降に論理長が動いていないこと（リースと rewind を塞いである以上、
    // 割れたら実装の不変条件破れ — 沈黙で続けさせない）。
    this.#assertCapturedPast(pending.pastLength, "commit");
    const next = pending.pastLength + rows;
    // MUST: 和の u32 上限は `#advance` と同じ理由でここでも見る（両項が u32 以下でも溢れる）。
    if (next > MAX_LOGICAL_LENGTH) {
      throw new ExecutionError(
        `commit: pastLength ${pending.pastLength} + 受理行数 ${rows} = ${next} が ` +
          `u32 の上限 ${MAX_LOGICAL_LENGTH} を超える（論理長の搬送先は u32 — ADR 0066 追記 4）`,
      );
    }
    // MUST: 論理長の更新と保留の解除は不可分（先に解除すると、上の検査で落ちた commit の後に
    // 「保留も無く論理長も進んでいない」状態が残り、書かれた行が誰からも辿れなくなる）。
    this.#pastLength = next;
    this.#pending = undefined;
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
    this.#assertNotBorrowing("rewind");
    if (this.#runs > 0) {
      throw new ExecutionError(
        `rewind: 進行中の generation run が ${this.#runs} 本ある間は巻き戻せない` +
          "（run は頭で捕捉した pastLength で uniform と dispatch 数を決めるので、横から動かすと" +
          "GPU が見た論理長と進行の基準が分裂する）。run の決着を await してから呼ぶこと",
      );
    }
    if (this.#pending !== undefined) {
      throw new ExecutionError(
        `rewind: commit 待ちの generation run がある間は巻き戻せない` +
          `（pastLength ${this.#pending.pastLength} + queryLength ${this.#pending.queryLength} ` +
          "まで物理 ring へ書き込み済み）。context.commit(rows) で確定させてから呼ぶこと",
      );
    }
    if (this.#slidingSlots.size > 0) {
      throw new ExecutionError(
        `rewind: sliding スロット [${
          [...this.#slidingSlots.keys()].join(", ")
        }] を含む context は` +
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
    // MUST: 借り手が生きている間は破棄しない（ADR 0096 段 2 §2.1）。借り手の bind group は
    // このスロットバッファを掴んでおり、破棄すると次の draft が破棄済みバッファを読む。
    // `dispose` の「冪等・非 throw」契約からの**意図的な逸脱**で、受付終了フラグを立てる前に
    // 返す（立ててから落とすと、以後どの操作も通らない context が残る）。
    if (this.#borrowers.size > 0 && this.#disposal === undefined) {
      return Promise.reject(
        new ExecutionError(
          `dispose: この GenerationContext を借りている context が ${this.#borrowers.size} 本ある` +
            "（借り手の bind group が貸し手のスロットを掴んでいる）。借り手を先に dispose すること",
        ),
      );
    }
    this.#disposeRequested = true;
    this.#disposal ??= this.#host.serialize(async () => {
      this.#disposed = true;
      try {
        await this.#host.flush();
      } finally {
        // MUST: 借り物のスロットは破棄しない（所有者は貸し手 context）。借り手が破棄すると、
        // 貸し手の次の run が破棄済みバッファへ書く。
        if (this.#borrow === undefined) {
          for (const slot of this.#slots.values()) slot.buffer.destroy();
        } else {
          this.#borrow.lender[RUNTIME_INTERNAL].forgetBorrower(this);
        }
        this.#lengths.destroy();
        // MUST: 焼いた束もここで手放す。以後 run は来ない（`#assertUsable` が落とす）ので
        // 正しさには効かないが、掴んだままだと破棄済みバッファを参照する bind group が
        // context の参照ぶんだけ生き残る。
        this.#baked.clear();
        // 未 commit の保留はここで捨てる（物理バッファごと畳むので、確定させる相手が居ない）。
        this.#pending = undefined;
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
   * WHY 決定 6 の「論理長は run の成功で進む」は 2 形になった: `commit: "immediate"`（既定・
   * 従来）は run の成功でここが進め、`commit: "deferred"` は成功で {@link GenerationContext.commit}
   * へ権利を渡す（{@link #defer}）。どちらも論理長を動かす経路は**同時に 1 本**で、未 commit の
   * 間は次の run も rewind も拒否されるので、二重簿記（ホスト側にもう 1 つの論理長が生まれる形）は
   * 生まない。
   *
   * NOTE: full スロットの実行時検査 `pastLength + queryLength ≤ 容量`（ADR 0067 決定 4 の④）は
   * **run のエンコード前**に居る（`assertGenerationRun` — src/runtime/recipe.ts）。容量軸がどの
   * 次元かを決めるのは op 契約なので、導出相が集めた {@link GenerationLimits} が正本で、
   * ここでは重ねて見ない（進行の時点で検査しても、既に書かれた後で手遅れになる）。
   */
  #advance(pastLength: number, queryLength: number): void {
    this.#assertInternalUsable("advance");
    this.#assertNotBorrowing("advance");
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
   * 論理長を進めずに保留する（`commit: "deferred"` の run の成功でのみ — {@link #advance} の対）。
   *
   * MUST: 保留は高々 1 本（未 commit の間は `acquireRun` が次の run を拒否するので、2 本目の
   * `defer` に到達する経路そのものが無い）。到達したらランタイム内部の不変条件破れなので、
   * 上書きせず即死させる — 上書きすると 1 本目が書いた行がどこからも辿れなくなる。
   */
  #defer(pastLength: number, queryLength: number): void {
    this.#assertInternalUsable("defer");
    this.#assertNotBorrowing("defer");
    this.#assertCapturedPast(pastLength, "defer");
    this.#assertQueryLength(queryLength, "defer");
    if (this.#pending !== undefined) {
      throw new ExecutionError(
        `defer: commit 待ちの generation run が既にある（pastLength ${this.#pending.pastLength} / ` +
          `queryLength ${this.#pending.queryLength}）— 内部の不変条件破れ`,
      );
    }
    this.#pending = { pastLength, queryLength };
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

  /**
   * 借り手 context が触れない面（論理長を動かす 4 つ）— `commit` / `rewind` は利用者面の
   * 拒否、`advance` / `defer` はランタイム内部の不変条件破れ（実行統合が
   * {@link GenerationContextInternals.borrowing} を見て呼ばない契約 — ADR 0096 段 2 §2.1）。
   */
  #assertNotBorrowing(where: string): void {
    if (this.#borrow === undefined) return;
    throw new ExecutionError(
      `${where}: 借り手 context は論理長を持たない（進行も巻き戻しも貸し手 context の側で` +
        "起きる — ADR 0096 段 2 §2.1）",
    );
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
