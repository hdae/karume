/**
 * メモリ必要量 estimator（ADR 0070 決定 5）— GPU に一切触れない純関数で「必要側」のバイト数を
 * カテゴリ別に出す。
 *
 * MUST: **空き側との比較はしない**。WebGPU は総 / 空き VRAM を露出しないので、比較の形にした
 * 瞬間に「予算が取れない環境での当て推量」になる（ADR 0070 決定 5 — llama.cpp の 0/0 デバイス
 * 除外と同型の前例）。ここが返すのは数字だけで、可否の判定は一切しない。
 * MUST: 見積りは**絶対保証ではない**。勘定に入っていないものは
 * {@link MemoryEstimate.unaccounted} が形式として認め、実際の**最終門は out-of-memory
 * errorScope**（src/gpu/device.ts の 2 本組）のまま — estimator は Session 構築 /
 * `createGenerationContext` の事前診断であって、超えていても実行は止めない。
 *
 * 実測（{@link SessionDiagnostics}）との対応:
 * - {@link MemoryEstimate.compressedWeightBytes} ↔ `storage.residentCompressedBytes`
 * - {@link MemoryEstimate.expandedWeightBytes} ↔ `storage.hostExpandedBytes`
 * - {@link MemoryEstimate.uncompressedWeightBytes} — 対応する診断欄は無い（ADR 0006 の
 *   `storage` 診断は低精度格納の実績だけを数える。実測の総量は `weights.allocatedBytes` —
 *   params 込み — でしか観測できない）
 * - {@link MemoryEstimate.stateBytes} ↔ `stateBacking.residentBytes`（context 1 本のとき）
 * - {@link MemoryEstimate.transientBytes} ↔ `planBacking.residentBytes` /
 *   `lastRun.peakTransientBytes`
 *
 * MUST: 分類も算式も**実装と同じ導出元**から引く（適格判定は plan.ts の純関数 2 本、整列と
 * サイズクラスは `toSizeClass`、state の 1 要素バイト数と論理長 uniform は
 * generation-context.ts の定数）。式をここで書き直すと、片方だけ直された実装に対して
 * estimator が例外も警告も無く別の数を主張し続ける。
 */

import { assertRuntimeSupport, type KarumeModel } from "../format/container.ts";
import type { IrGraph } from "../format/ir.ts";
import type { SafetensorsFile } from "../format/safetensors.ts";
import { toSizeClass } from "../gpu/arena.ts";
import { numel, RUNTIME_SUPPORT } from "../ops.ts";
import { assertGenerationBindings } from "./executor.ts";
import { aliasesInput } from "./fusion.ts";
import {
  assertChunkLength,
  LENGTHS_BYTES,
  resolveBindings,
  resolveSlotShape,
  STATE_ELEMENT_BYTES,
} from "./generation-context.ts";
import {
  countUses,
  eligibleCompressedInitializers,
  ExecutionError,
  i4EligibleInitializers,
  type NodePlan,
  planGraph,
  statesOnlySymbols,
  type SymbolBindings,
} from "./plan.ts";
import type { GenerationContextSpec } from "./session-types.ts";

/** カテゴリ別の必要バイト数（{@link estimateSessionMemory} の戻り）。 */
export type MemoryEstimate = {
  /**
   * 圧縮のまま GPU 常駐する重み + companion scale のバイト数（整列詰め物込み — 厳密）。
   * 診断の `storage.residentCompressedBytes` と厳密に対応する（0 要素テンソルの 4 バイト床
   * だけが唯一の差 — 診断側は床なしの実バイトを数える）。
   */
  readonly compressedWeightBytes: number;
  /**
   * 圧縮しない格納（f32 / i32）のまま常駐する重みのバイト数（厳密）。
   *
   * MUST: 圧縮 / 展開と別の欄で持つ — 診断の `storage` は低精度格納の実績だけを数える
   * （ADR 0006）ので、ここを {@link MemoryEstimate.compressedWeightBytes} に混ぜると
   * 「診断との厳密一致」が f32 重みを持つモデルで沈黙して崩れる。
   */
  readonly uncompressedWeightBytes: number;
  /** 適格外でロード時に f32 展開して常駐する重みの展開後バイト数（厳密）。 */
  readonly expandedWeightBytes: number;
  /**
   * `GenerationContext` 1 本ぶんの state スロット + 論理長 uniform（厳密）。
   *
   * NOTE: `options.generation` を省略すると 0。`graph.states` が非空なグラフでは
   * {@link MemoryEstimate.transientBytes} の計画が state スロットの解決済み shape を要求する
   * ので、そもそも generation 無しでは見積れず fail loudly になる（黙って 0 を返す窓は無い）。
   */
  readonly stateBytes: number;
  /** グラフ入力バッファ + 出力 readback staging（厳密）。 */
  readonly ioBytes: number;
  /**
   * 中間（transient）slot 表の必要バイト = prepared backing の必要側と同義（**近似**）。
   *
   * 融合前のノード列を宣言順に歩き、実行相と同じ確保規則（exact-size LIFO 再利用・
   * 消費回数は `countUses`・グラフ出力は pinned で解放しない・reshape / 恒等 expand の出力は
   * 確保せず入力の実体を別名で使う）で slot 表を再生した総バイト。融合が消す中間・行ブロック
   * 分割やノード内の一時は勘定に入らない（{@link MemoryEstimate.unaccounted}）。
   */
  readonly transientBytes: number;
  /** 上記の合計。 */
  readonly totalBytes: number;
  /** 勘定に入っていないもの（見積りが絶対保証でないことを形式が認める欄 — ADR 0070 決定 5）。 */
  readonly unaccounted: readonly string[];
};

export type EstimateOptions = {
  /**
   * 記号次元の値（グラフ入力側 — run に渡すのと同じ束縛）。states 専用記号は
   * {@link EstimateOptions.generation} の側で与える。
   */
  readonly bindings?: SymbolBindings;
  /** 見積る `GenerationContext` の仕様（省略すると state を数えない）。 */
  readonly generation?: GenerationContextSpec;
};

/**
 * 勘定に入っていないもの（定数）。
 *
 * MUST: 「入っていない」と分かっている項目を全部書く。空欄にすると、見積りが上限であるかの
 * ように読まれる（ADR 0070 決定 5 が unaccounted 欄を要求した理由そのもの）。
 */
const UNACCOUNTED: readonly string[] = Object.freeze([
  "融合が畳んで消す中間と、行ブロック分割の一時（どちらも device limit と融合の成立に依存する。reshape / 恒等 expand の別名は勘定に入っている）",
  "params バッファ（カーネル定数 — Session 常駐・内容アドレスキャッシュ）",
  "queue.writeBuffer の実装 staging（submit の完了まで解放されない）",
]);

/** 解決済み shape を引く（planGraph が全値を載せているので、欠けは簿記の破れ）。 */
const shapeOf = (
  shapes: ReadonlyMap<string, readonly number[]>,
  name: string,
): readonly number[] => {
  const shape = shapes.get(name);
  if (shape === undefined) throw new ExecutionError(`値 '${name}' の shape が未解決`);
  return shape;
};

/** safetensors 実テンソルのバイト数（container が「実バイト = 宣言由来バイト」を保証済み）。 */
const tensorByteLength = (file: SafetensorsFile, where: string, key: string): number => {
  const view = file.tensors.get(key);
  if (view === undefined) throw new ExecutionError(`${where}: テンソル '${key}' が無い`);
  return view.byteLength;
};

/**
 * 値 shape の解決に使う束縛を検査して写す。
 *
 * MUST: states 専用記号をここで受けない（`bindSymbols` と同じ分担 — 容量の正本は
 * `createGenerationContext` の側で、ここで受けると「state 容量を渡したのに 0 と出る」
 * 沈黙誤答になる）。未束縛は fail loudly（黙って 0 で埋めない）。
 */
const planBindings = (graph: IrGraph, bindings: SymbolBindings | undefined): SymbolBindings => {
  // 記号の実在と非負整数の検査、および null プロトタイプの器は resolveBindings が持つ。
  const resolved = resolveBindings(graph, bindings);
  const statesOnly = statesOnlySymbols(graph);
  for (const sym of Object.keys(resolved)) {
    if (!statesOnly.has(sym)) continue;
    throw new ExecutionError(
      `束縛 '${sym}' は states 専用記号 — options.generation.bindings で与えること` +
        "（ADR 0066 追記 7 の束縛点 2）",
    );
  }
  for (const sym of graph.symbols) {
    if (statesOnly.has(sym) || Object.hasOwn(resolved, sym)) continue;
    throw new ExecutionError(
      `シンボル '${sym}' が束縛されていない（options.bindings で与えること）`,
    );
  }
  return resolved;
};

/** state スロットの見積り（{@link GenerationContext} の確保と同じ式 — 値の複製はしない）。 */
const stateEstimate = (
  graph: IrGraph,
  spec: GenerationContextSpec | undefined,
): {
  readonly bytes: number;
  readonly shapes: ReadonlyMap<string, readonly number[]> | undefined;
  readonly bindings: SymbolBindings | undefined;
} => {
  if (spec === undefined) return { bytes: 0, shapes: undefined, bindings: undefined };
  const names = Object.keys(graph.states);
  // MUST: states 宣言の無いグラフで generation を受けない（`GenerationContext.create` が
  // 拒否する形。通すと「作れない context のバイト数」を数えた見積りが返る）。
  if (names.length === 0) {
    throw new ExecutionError(
      "このグラフは states 宣言を持たない（GenerationContext は作れないので options.generation を渡さないこと）",
    );
  }
  // MUST: 実構築（GenerationContext.create）が拒否する spec に見積りを返さない — 値域は
  // 同じ門（assertChunkLength）を通す。
  assertChunkLength(spec.chunkLength);
  const bindings = resolveBindings(graph, spec.bindings);
  const shapes = new Map<string, readonly number[]>();
  let bytes = 0;
  for (const name of names) {
    const shape = resolveSlotShape(name, graph.states[name].shape, bindings);
    shapes.set(name, shape);
    bytes += numel(shape) * STATE_ELEMENT_BYTES;
  }
  // 論理長 uniform は context 1 本につき 1 枚（スロット数に依らない）。
  return { bytes: bytes + LENGTHS_BYTES, shapes, bindings };
};

/**
 * 重みの見積り。分類は Session 構築（executor.ts の `Session.create`）と同じ 3 点 —
 * 適格判定 {@link eligibleCompressedInitializers}、i4 だけ {@link i4EligibleInitializers}
 * との積（ADR 0069 決定 5）、格納 dtype ごとの分岐。
 *
 * バイト数は宣言 shape から求める（container の突合が「実バイト = 宣言由来バイト」を保証済み）。
 * 整列詰め物とバッファ床は {@link toSizeClass} が持つ — `alignF16Payload` / `alignI8Payload` の
 * 4 バイト切り上げと `allocHostWritten` の `Math.max(4, …)` を合わせた値と同じ。
 */
const weightEstimate = (
  model: KarumeModel,
): { readonly compressed: number; readonly uncompressed: number; readonly expanded: number } => {
  const { graph, file } = model;
  const eligible = eligibleCompressedInitializers(graph);
  const i4Eligible = i4EligibleInitializers(graph);
  let compressed = 0;
  let uncompressed = 0;
  let expanded = 0;
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const where = `initializer '${name}'`;
    // initializer の宣言 shape は数値のみ（parseIrGraph が保証 — 記号次元は拒否）。
    const count = numel(graph.values[name].shape.map(Number));
    const storage = initializer.storage.dtype;
    if (storage === "f32" || storage === "i32" || storage === "bf16") {
      // 圧縮しない格納は生バイトがそのまま GPU 表現（executor の分岐 3 本目）。
      uncompressed += toSizeClass(tensorByteLength(file, where, initializer.tensor));
      continue;
    }
    // i4 の適格は f16 / i8 より狭い「重みスロットでの消費が linear / embedding /
    // conv1d(groups==1) だけ」（ADR 0069 決定 5 とその追補）。
    const residentEligible = storage === "i4"
      ? eligible.has(name) && i4Eligible.has(name)
      : eligible.has(name);
    if (!residentEligible) {
      // 適格外はロード時に CPU で f32 展開する（VRAM 削減はゼロ）。
      expanded += count * 4;
      continue;
    }
    // f16 は numel×2 / i8 は numel / i4 は numel÷2（packed nibble — ADR 0069 決定 2）。
    const payload = storage === "f16" ? count * 2 : storage === "i8" ? count : count / 2;
    compressed += toSizeClass(payload);
    if (storage === "f16") continue;
    // MUST: companion scale のバッファも数える（executor の residentCompressedBytes と同じ
    // 規律 — 実際に GPU が抱えるバイト数を表す欄なので、scale を除くと実績と食い違う）。
    const scaleKey = initializer.storage.scale;
    if (scaleKey === undefined) {
      throw new ExecutionError(`${where}: 格納 ${storage} なのに storage.scale が無い`);
    }
    compressed += toSizeClass(tensorByteLength(file, where, scaleKey));
  }
  return { compressed, uncompressed, expanded };
};

/**
 * 中間（transient）slot 表の必要バイト（近似）。融合前のノード列を宣言順に歩き、実行相と
 * **同じ確保規則**を再生する: 解放済み slot の再利用は**サイズクラスの厳密一致だけ**
 * （RunArena / `derivePlanSlots` の LIFO プール — 近いサイズへの縮めはしない）で、一致が
 * 無ければ新しい slot が増える。返すのは生成された slot の総バイト = prepared backing が
 * 常駐させる必要側。
 *
 * MUST: 「同時生存バイトの最大」で代用しない — 実行側は exact-size 再利用なので、サイズが
 * 揃わない列では解放済みぶんが再利用されずに slot が累積し、生存ピークは系統的に過小になる
 * （8→12→4 バイトの 3 段で 20 vs 24 — 断片化は unaccounted ではなく規則そのもの）。
 * MUST: 数えるのは**ノード出力だけ**。initializer / グラフ入力は重み・io の側で、state
 * スロットは context の側で数えており、ここで重ねると同じバイトが 2 回総計に乗る。
 * MUST: 簿記の単位は**値名ではなく slot**（`derivePlanSlots` の写し）。reshape / 恒等 expand の
 * 出力は確保を出さず入力の実体を指すだけなので（別名 — 判定は {@link aliasesInput} の 1 本）、
 * 名前ごとに独立した生存区間を持たせると ①別名鎖の根が二重に数えられ ②別名越しの消費が
 * 根へ届かず ③グラフ出力が別名名義のときに根の実体だけプールへ戻る。retain / release は
 * **根の slot** へ合算する（実行相の「別名でも retain は実バッファに積む」— recipe.ts の
 * `executeStepRecipe`）。
 */
const transientSlotBytes = (graph: IrGraph, nodes: readonly NodePlan[]): number => {
  const uses = countUses(graph);
  const outputNames = new Set(graph.outputs);
  /** slot 添字 → サイズクラス（`derivePlanSlots` の `bytes` と同じ表）。 */
  const bytes: number[] = [];
  /** サイズクラス → 解放済み slot（LIFO — RunArena の `#pool` と同じ形）。 */
  const pool = new Map<number, number[]>();
  const refs = new Map<number, number>();
  const pinned = new Set<number>();
  /** 値名 → slot。別名は**根の slot**を指し、slot を持たない値（グラフ入力・重み）は載せない。 */
  const env = new Map<string, number>();
  let total = 0;

  const alloc = (byteLength: number): number => {
    const sizeClass = toSizeClass(byteLength);
    const reused = pool.get(sizeClass)?.pop();
    if (reused !== undefined) return reused;
    bytes.push(sizeClass);
    total += sizeClass;
    return bytes.length - 1;
  };
  const retain = (slot: number | undefined, count: number, isPinned: boolean): void => {
    if (slot === undefined) return;
    if (isPinned) pinned.add(slot);
    refs.set(slot, (refs.get(slot) ?? 0) + count + 1);
  };
  const release = (slot: number | undefined): void => {
    if (slot === undefined) return;
    const left = (refs.get(slot) ?? 0) - 1;
    if (left < 0) throw new ExecutionError("中間の見積り: 参照カウントが負（消費計数の誤り）");
    refs.set(slot, left);
    if (left > 0 || pinned.has(slot)) return;
    const bucket = pool.get(bytes[slot]);
    if (bucket === undefined) pool.set(bytes[slot], [slot]);
    else bucket.push(slot);
  };

  for (const node of nodes) {
    // MUST: 別名元は入力の先頭（recipe-builder の `#buildStep` が `binds[0]` を指す）。元が
    // slot を持たない値（グラフ入力・重み）なら、この出力も slot を持たない。
    const isAlias = aliasesInput(node);
    const aliasSlot = isAlias ? env.get(node.node.ins[0]) : undefined;
    const slots = node.outputs.map((out) => {
      // ステップ出力の確保サイズは常に numel×4（recipe-builder の `#buildStep`）。
      const slot = isAlias ? aliasSlot : alloc(numel(out.shape) * 4);
      retain(slot, uses.get(out.name) ?? 0, outputNames.has(out.name));
      if (slot === undefined) env.delete(out.name);
      else env.set(out.name, slot);
      return slot;
    });
    for (const name of node.node.ins) release(env.get(name));
    // 消費者ゼロの中間出力が解放されるのはこの 1 本だけ（実行相の「定義ぶんの解放」と同位置）。
    for (const slot of slots) release(slot);
  }
  return total;
};

/**
 * Session 1 本ぶんの必要メモリを見積る（ADR 0070 決定 5 — GPU も device も要らない純関数）。
 *
 * 判定はしない: 返るのはカテゴリ別のバイト数と、勘定に入っていないものの列だけ。可否の
 * 最終門は out-of-memory errorScope のままで、この見積りを超えていても実行は止まらない。
 *
 * @param options.bindings グラフ入力側の記号次元（run に渡すのと同じ束縛）。未束縛の記号が
 *   要る形は fail loudly。
 * @param options.generation 見積る `GenerationContext` の仕様。省略すると
 *   {@link MemoryEstimate.stateBytes} は 0（states 形グラフでは省略できない — 中間ピークの
 *   計画がスロットの解決済み shape を要求する）。
 */
export const estimateSessionMemory = (
  model: KarumeModel,
  options: EstimateOptions = {},
): MemoryEstimate => {
  const graph = model.graph;
  // MUST: 実構築（`createSession` / `createSessionFromShards`）と**同じ門**を先に通す。
  // 作れない構成へ見積りを返すと、格納 dtype や op が非対応のモデルに対して estimator だけが
  // もっともらしい総量を主張する（例: 格納 `bf16` は IR の語彙にはあるが RUNTIME_SUPPORT に
  // 無く、Session 構築は必ず落ちる）。門を共有すれば語彙が増えたときの抜けも同時に塞がる。
  // NOTE: これは capability 検査であって空き側との比較ではないので、ADR 0070 決定 5 の
  // 「比較をしない」規律には抵触しない（GPU にも触らず純関数のまま）。
  assertRuntimeSupport(graph, RUNTIME_SUPPORT);
  const bindings = planBindings(graph, options.bindings);
  const state = stateEstimate(graph, options.generation);
  // MUST: states と入力の両方に現れる記号は 2 つの束縛点で同じ値（run が拒否する分裂 —
  // ADR 0066 追記 7 — に見積りだけが正常値を返さない）。
  if (state.bindings !== undefined) assertGenerationBindings(state.bindings, bindings);
  const weights = weightEstimate(model);
  const plan = planGraph(graph, bindings, state.shapes);

  // 入力バッファと出力 readback staging は同じ床つきの式（executor の `#bindInput` /
  // `#activateBacking` / `#stageOutputs` — `Math.max(4, numel×4)` は toSizeClass と同値）。
  let ioBytes = 0;
  for (const spec of graph.inputs) {
    ioBytes += toSizeClass(numel(shapeOf(plan.shapes, spec.name)) * 4);
  }
  for (const name of graph.outputs) ioBytes += toSizeClass(numel(shapeOf(plan.shapes, name)) * 4);
  const transientBytes = transientSlotBytes(graph, plan.nodes);

  return {
    compressedWeightBytes: weights.compressed,
    uncompressedWeightBytes: weights.uncompressed,
    expandedWeightBytes: weights.expanded,
    stateBytes: state.bytes,
    ioBytes,
    transientBytes,
    totalBytes: weights.compressed + weights.uncompressed + weights.expanded + state.bytes +
      ioBytes + transientBytes,
    unaccounted: UNACCOUNTED,
  };
};
