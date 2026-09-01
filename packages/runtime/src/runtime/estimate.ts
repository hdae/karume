/**
 * メモリ必要量 estimator（ADR 0070 決定 5）— GPU に一切触れない純関数で「必要側」のバイト数を
 * カテゴリ別に出す。
 *
 * MUST: **空き側との比較はしない**。WebGPU は総 / 空き VRAM を露出しないので、比較の形にした
 * 瞬間に「予算が取れない環境での当て推量」になる（ADR 0070 決定 5 — llama.cpp の 0/0 デバイス
 * 除外と同型の前例）。ここが返すのは数字だけで、可否の判定は一切しない。
 * MUST: 見積りは**絶対保証ではない**。勘定に入っていないものは
 * {@link AdmissionReport.unaccounted} が形式として認め、実際の**最終門は out-of-memory
 * errorScope**（src/gpu/device.ts の 2 本組）のまま — estimator は Session 構築 /
 * `createGenerationContext` の事前診断であって、超えていても実行は止めない。
 *
 * 報告の形は**常駐 + シナリオ**の 2 段（ADR 0070 決定 5 の再具体化）。Session が寿命を通じて
 * 抱えるもの（重み・state スロット）は {@link AdmissionReport.resident} の 1 組で、run の形
 * （prefill / decode）ごとに変わるものは {@link AdmissionReport.scenarios} に**別々の計算**
 * として並ぶ。同じ数字を 1 本に潰すと、prefill 形でしか出ない中間の山を decode しか回さない
 * 呼び手に押しつけるか、その逆に decode の数字で prefill を通すかのどちらかになる。
 *
 * 実測（{@link SessionDiagnostics}）との対応:
 * - {@link AdmissionReport.resident}`.weights.compressedBytes` ↔ `storage.residentCompressedBytes`
 * - {@link AdmissionReport.resident}`.weights.expandedBytes` ↔ `storage.hostExpandedBytes`
 * - {@link AdmissionReport.resident}`.weights.uncompressedBytes` — 対応する診断欄は無い
 *   （ADR 0006 の `storage` 診断は低精度格納の実績だけを数える。実測の総量は
 *   `weights.allocatedBytes` — params 込み — でしか観測できない）
 * - {@link AdmissionReport.resident}`.stateBytes` ↔ `stateBacking.residentBytes`（context 1 本のとき）
 * - {@link AdmissionScenario.workspaceBytes} ↔ `planBacking.residentBytes` /
 *   `lastRun.peakTransientBytes`（そのシナリオの形で回した run のもの）
 *
 * MUST: 分類も算式も**実装と同じ導出元**から引く（重みの席と宣言由来バイト数は
 * weight-residency.ts の純関数プランナ、整列とサイズクラスは `toSizeClass`、state の 1 要素
 * バイト数と論理長 uniform は generation-context.ts の定数、物理 chunk 行 `M` の載る軸は
 * ops/shapes.ts の op 契約）。式をここで書き直すと、片方だけ直された実装に対して estimator が
 * 例外も警告も無く別の数を主張し続ける。
 */

import { assertRuntimeSupport, type KarumeModel } from "../format/container.ts";
import { parseDim, solveDim } from "../format/dims.ts";
import type { IrDim, IrGraph } from "../format/ir.ts";
import { toSizeClass } from "../gpu/arena.ts";
import { numel, RUNTIME_SUPPORT } from "../ops.ts";
import { aliasesInput } from "./fusion.ts";
import {
  assertChunkLength,
  LENGTHS_BYTES,
  resolveBindings,
  resolveSlotShape,
  STATE_ELEMENT_BYTES,
} from "./generation-context.ts";
import {
  assertGenerationBindings,
  countUses,
  ExecutionError,
  type NodePlan,
  planGraph,
  statesOnlySymbols,
  type SymbolBindings,
  validateGraphContracts,
} from "./plan.ts";
import type { GenerationContextSpec } from "./session-types.ts";
import {
  planWeightBuffers,
  planWeightResidency,
  type WeightResidency,
} from "./weight-residency.ts";

/**
 * シナリオの名前。`generation` を渡さない見積りは `"run"` の 1 本、渡した見積りは
 * ADR 0066 決定 4 の実行 2 形（`"prefill"` / `"decode"`）。
 */
export type AdmissionScenarioName = "run" | "prefill" | "decode";

/**
 * run の形 1 つぶんの必要バイト数（{@link AdmissionReport.scenarios} の 1 要素）。
 *
 * ここに入るのは**run の形で変わるものだけ** — 重みと state スロットは形に依らないので
 * {@link AdmissionReport.resident} の側にある。
 */
export type AdmissionScenario = {
  readonly name: AdmissionScenarioName;
  /** グラフ入力バッファ + 出力 readback staging（厳密）。 */
  readonly ioBytes: number;
  /**
   * 中間（transient）slot 表の必要バイト = prepared backing の必要側と同義（**近似**）。
   *
   * 融合前のノード列を宣言順に歩き、実行相と同じ確保規則（exact-size LIFO 再利用・
   * 消費回数は `countUses`・グラフ出力は pinned で解放しない・reshape / 恒等 expand の出力は
   * 確保せず入力の実体を別名で使う）で slot 表を再生した総バイト。融合が消す中間・行ブロック
   * 分割やノード内の一時は勘定に入らない（{@link AdmissionReport.unaccounted}）。
   */
  readonly workspaceBytes: number;
};

/** 必要バイト数の報告（{@link estimateSessionMemory} / `PreparedModel.estimate` の戻り）。 */
export type AdmissionReport = {
  /** run の形に依らず Session の寿命ぶん抱えるもの。 */
  readonly resident: {
    readonly weights: {
      /**
       * 圧縮のまま GPU 常駐する重み + companion scale のバイト数（整列詰め物込み — 厳密）。
       * 診断の `storage.residentCompressedBytes` と厳密に対応する（0 要素テンソルの 4 バイト床
       * だけが唯一の差 — 診断側は床なしの実バイトを数える）。
       */
      readonly compressedBytes: number;
      /**
       * 圧縮しない格納（f32 / i32）のまま常駐する重みのバイト数（厳密）。
       *
       * MUST: 圧縮 / 展開と別の欄で持つ — 診断の `storage` は低精度格納の実績だけを数える
       * （ADR 0006）ので、ここを `compressedBytes` に混ぜると「診断との厳密一致」が f32 重みを
       * 持つモデルで沈黙して崩れる。
       */
      readonly uncompressedBytes: number;
      /** 適格外でロード時に f32 展開して常駐する重みの展開後バイト数（厳密）。 */
      readonly expandedBytes: number;
      /** 上記 3 欄の和。 */
      readonly totalBytes: number;
    };
    /**
     * `GenerationContext` 1 本ぶんの state スロット + 論理長 uniform（厳密）。
     *
     * NOTE: `options.generation` を省略すると 0。`graph.states` が非空なグラフでは
     * {@link AdmissionScenario.workspaceBytes} の計画が state スロットの解決済み shape を
     * 要求するので、そもそも generation 無しでは見積れず fail loudly になる（黙って 0 を返す
     * 窓は無い）。
     */
    readonly stateBytes: number;
  };
  /** run の形ごとの必要バイト数（1 要素以上 — 名前の決まり方は {@link AdmissionScenarioName}）。 */
  readonly scenarios: readonly AdmissionScenario[];
  /**
   * 常駐の総和 + シナリオ側の最大（= `resident.weights.totalBytes + resident.stateBytes +
   * max(ioBytes + workspaceBytes)`）。
   *
   * **上限保証ではなく「勘定に入れた分のピーク」**を名乗る欄（名前の由来）。シナリオ側を和では
   * なく max で足すのは、Session が抱える slot backing が**同時に 1 本**だから — 計画
   * （`PreparedPlan`）は `PREPARED_PLAN_CAPACITY = 4` 本まで LRU で残るが、実体を持つ
   * `ActiveBacking` は容量 1 で、別 signature の run はまず現行 backing を退役させてから
   * 確保し直す（executor の `Session.#activateBacking` / `#retireBacking`）。退役から実際の
   * `destroy()` までの窓で 2 本ぶんが同時に載る点は {@link AdmissionReport.unaccounted} 側。
   */
  readonly peakAccountedBytes: number;
  /** 勘定に入っていないもの（見積りが絶対保証でないことを形式が認める欄 — ADR 0070 決定 5）。 */
  readonly unaccounted: readonly string[];
};

export type EstimateOptions = {
  /**
   * 記号次元の値（グラフ入力側 — run に渡すのと同じ束縛）。states 専用記号（容量）と
   * 物理 chunk 行 `M` の記号は、どちらも {@link EstimateOptions.generation} の側から決まるので
   * ここでは受けない（前者は `generation.bindings`・後者は `generation.chunkLength`）。
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
  "states 形 attention のノード内一時（スコア S と行統計）— 融合の成立に依存せず必ず出る。行ブロック 1 枚ぶんが同時生存し、大きさは B·H × 行ブロック行数 × 列容量 × 4 バイト（行統計は列容量の代わりに固定 stride）",
  "params バッファ（カーネル定数 — Session 常駐・内容アドレスキャッシュ）",
  "queue.writeBuffer の実装 staging（submit の完了まで解放されない）",
  "シナリオ切替の窓（退役した slot backing は次の計画の確保より前に destroy されず flush 後の後始末まで生きるので、prefill ⇄ decode の切替 run では 2 シナリオぶんの io + workspace が同時に載る）",
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

/**
 * 物理 chunk 行 `M` が載る軸。
 *
 * MUST: 位置の正本は **op 契約**（ops/shapes.ts）。states 形の attention は
 * `q[B,H,M,D]` / `k`,`v``[B,Hkv,M,D]`（M は q / k / v で一致することも契約が見ている）、
 * `state_append` は `x[B,Hkv,M,D]` で、どちらも軸 2。ここで別の軸を書くと、estimator だけが
 * 実行と違う次元を chunk 行と見なして prefill / decode を取り違える。
 */
const CHUNK_ROW_AXIS = 2;

/**
 * 物理 chunk 行 `M` を決めている**次元宣言**（記号を含むものだけ・重複排除済み）。
 *
 * 判別は「states 欄が非空なノードの入力」= states 形の判別そのもの（ADR 0067 決定 4 —
 * recipe-builder の `#buildStateAttention` / `#buildStateAppend` が `states.chunkRows` へ
 * 積むのと同じノード集合）。数値次元は記号で動かない形なので拾わない — その形の decode step は
 * 物理 chunk 行を prefill と同じまま `queryLength = 1` で回る（`assertGenerationRun` は
 * `rows === chunkLength` を decode でも許す）ので、2 シナリオが同じ数字になるのが正しい。
 */
const chunkRowDims = (graph: IrGraph): readonly string[] => {
  const declared = new Map<string, readonly IrDim[]>();
  for (const spec of graph.inputs) declared.set(spec.name, spec.shape);
  for (const [name, value] of Object.entries(graph.values)) declared.set(name, value.shape);
  const dims = new Set<string>();
  for (const node of graph.nodes) {
    if (Object.keys(node.states).length === 0) continue;
    // 宣言が引けない名前（initializer だけの値など）は planGraph が落とす — ここでは飛ばす。
    for (const name of node.ins) {
      const dim = declared.get(name)?.[CHUNK_ROW_AXIS];
      if (typeof dim === "string") dims.add(dim);
    }
  }
  return [...dims];
};

/**
 * 値 shape の解決に使う束縛を検査して写す。
 *
 * MUST: states 専用記号をここで受けない（`bindSymbols` と同じ分担 — 容量の正本は
 * `createGenerationContext` の側で、ここで受けると「state 容量を渡したのに 0 と出る」
 * 沈黙誤答になる）。未束縛は fail loudly（黙って 0 で埋めない）。
 * MUST: chunk 行の記号もここで受けない（束縛点は `options.generation.chunkLength` と decode 形の
 * 1 で、シナリオごとに値が変わる第 3 の束縛点）。受けて上書きすると「prefill と決め打った値を
 * 渡したのに decode の数字が返る」沈黙上書きになり、受けて優先すると 2 本のシナリオが同じ形に
 * 潰れる。
 */
const planBindings = (
  graph: IrGraph,
  bindings: SymbolBindings | undefined,
  chunkSymbols: ReadonlySet<string>,
): SymbolBindings => {
  // 記号の実在と非負整数の検査、および null プロトタイプの器は resolveBindings が持つ。
  const resolved = resolveBindings(graph, bindings);
  const statesOnly = statesOnlySymbols(graph);
  for (const sym of Object.keys(resolved)) {
    if (statesOnly.has(sym)) {
      throw new ExecutionError(
        `束縛 '${sym}' は states 専用記号 — options.generation.bindings で与えること` +
          "（ADR 0066 追記 7 の束縛点 2）",
      );
    }
    if (chunkSymbols.has(sym)) {
      throw new ExecutionError(
        `束縛 '${sym}' は物理 chunk 行 M の記号 — 値は options.generation.chunkLength` +
          "（prefill 形）と 1（decode 形）から導く（ADR 0066 決定 4 の実行 2 形）",
      );
    }
  }
  for (const sym of graph.symbols) {
    if (statesOnly.has(sym) || chunkSymbols.has(sym) || Object.hasOwn(resolved, sym)) continue;
    throw new ExecutionError(
      `シンボル '${sym}' が束縛されていない（options.bindings で与えること）`,
    );
  }
  return resolved;
};

/**
 * chunk 行を `rows` 行にしたシナリオの束縛（`base` の写しに chunk 記号を足す）。
 *
 * MUST: 逆解きは実行と同じ 1 本（{@link solveDim} — `bindSymbols` が実入力の実寸から束縛を
 * 解くのに使うのと同じ関数）。派生形（`2M`）を丸めて受けると、宣言と 1 行ずれた chunk を
 * 見積った数字が「その形で走る」顔をして返る。
 */
const bindChunkRows = (
  base: SymbolBindings,
  dims: readonly string[],
  rows: number,
  scenario: AdmissionScenarioName,
): SymbolBindings => {
  const bindings: Record<string, number> = Object.assign(Object.create(null), base);
  for (const dim of dims) {
    const expr = parseDim(dim);
    const solved = solveDim(expr, rows);
    if (solved === undefined) {
      throw new ExecutionError(
        `シナリオ '${scenario}': 物理 chunk 行 ${rows} が宣言 '${dim}' の形をしていない` +
          `（${expr.coeff} で割り切れる ${expr.offset} 以上の行数が要る — ADR 0066 決定 4）`,
      );
    }
    if (Object.hasOwn(bindings, expr.sym) && bindings[expr.sym] !== solved) {
      throw new ExecutionError(
        `シナリオ '${scenario}': 記号 '${expr.sym}' の束縛が衝突（${bindings[expr.sym]} と ` +
          `${solved}） — chunk 行の宣言が 1 グラフ内で割れている`,
      );
    }
    bindings[expr.sym] = solved;
  }
  return bindings;
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
 * 重みの見積り。分類も宣言由来のバイト数も **Session 構築と同じプランナ**
 * （{@link planWeightResidency}）の結果を受け取り、席から実バッファへの展開も構築 / 上限検査と
 * 同じ 1 本（{@link planWeightBuffers}）を通す — ここで分類や展開を書き直すと、片方だけ直された
 * 実装に対して estimator が例外も警告も無く別の数を主張し続ける（モジュール doc の MUST）。
 *
 * ここが足すのは席ごとのカテゴリ分けだけ。整列詰め物とバッファ床（{@link toSizeClass}）は
 * `WeightBuffer.byteLength` 側が持つ — `alignF16Payload` / `alignI8Payload` の 4 バイト切り上げと
 * `allocHostWritten` の `Math.max(4, …)` を合わせた値と同じ。
 */
const weightEstimate = (
  residency: ReadonlyMap<string, WeightResidency>,
): { readonly compressed: number; readonly uncompressed: number; readonly expanded: number } => {
  let compressed = 0;
  let uncompressed = 0;
  let expanded = 0;
  for (const buffer of planWeightBuffers(residency)) {
    switch (buffer.seat) {
      case "raw":
        // 圧縮しない格納は生バイトがそのまま GPU 表現（executor の分岐 3 本目）。
        uncompressed += buffer.byteLength;
        break;
      case "expanded":
        // 適格外はロード時に CPU で f32 展開する（VRAM 削減はゼロ）。
        // MUST: この欄だけ整列前のバイト数で数える — 診断 `storage.hostExpandedBytes` は
        // writeBuffer した実バイトを積むので 4 バイト床を含まず、ここを確保寸法に替えると
        // 0 要素テンソルを持つモデルで「厳密対応」が沈黙して崩れる。
        expanded += buffer.declaredBytes;
        break;
      // 圧縮常駐は payload と companion scale の**両方**を数える（executor の
      // residentCompressedBytes と同じ規律 — 実際に GPU が抱えるバイト数を表す欄なので、
      // scale を除くと実績と食い違う）。どちらも `WeightBuffer` として並んでいる。
      case "f16":
      case "i8":
      case "i4":
        compressed += buffer.byteLength;
        break;
    }
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
 *   要る形は fail loudly。物理 chunk 行 `M` の記号だけは**ここでは受けない**（束縛点は
 *   `options.generation.chunkLength` と decode 形の 1 — {@link planBindings}）。
 * @param options.generation 見積る `GenerationContext` の仕様。省略すると
 *   `resident.stateBytes` は 0・シナリオは `"run"` の 1 本（states 形グラフでは省略できない —
 *   中間ピークの計画がスロットの解決済み shape を要求する）。渡すとシナリオは
 *   `"prefill"` / `"decode"` の 2 本になる。
 */
export const estimateSessionMemory = (
  model: KarumeModel,
  options: EstimateOptions = {},
): AdmissionReport => estimateGraphMemory(model.graph, planWeightResidency(model.graph), options);

/**
 * 見積りの本体（グラフと常駐計画だけで完結する — 全量面 {@link estimateSessionMemory} と
 * `PreparedModel.estimate` が共有する 1 本）。
 *
 * MUST: 常駐計画は**呼び手が持っているものを渡す**（`PreparedModel` は prepare 時に 1 回だけ
 * 計算して構築とも共有する）。ここで引き直すと「見積りに使った席」と「実際に上げた席」が
 * 別の計算結果になりうる形が復活する。
 */
export const estimateGraphMemory = (
  graph: IrGraph,
  residency: ReadonlyMap<string, WeightResidency>,
  options: EstimateOptions,
): AdmissionReport => {
  // MUST: 実構築（`createSession` / `createSessionFromShards`）と**同じ門**を先に通す。
  // 作れない構成へ見積りを返すと、格納 dtype や op が非対応のモデルに対して estimator だけが
  // もっともらしい総量を主張する（例: 格納 `bf16` は IR の語彙にはあるが RUNTIME_SUPPORT に
  // 無く、Session 構築は必ず落ちる）。門を共有すれば語彙が増えたときの抜けも同時に塞がる。
  // NOTE: これは capability 検査であって空き側との比較ではないので、ADR 0070 決定 5 の
  // 「比較をしない」規律には抵触しない（GPU にも触らず純関数のまま）。
  assertRuntimeSupport(graph, RUNTIME_SUPPORT);
  // グラフ全体を見ないと決まらない契約（state_append の本数と終端・Tmax 形・slice / flip の
  // 静的軸・cat の連結軸・入力の意味論 dtype）も同じ位置で通す。`PreparedModel.estimate` は
  // 構築時と二重に通ることになるが、どちらも純関数・冪等でグラフ 1 走査ぶんの費用しかない。
  validateGraphContracts(graph);
  const chunkDims = chunkRowDims(graph);
  const chunkSymbols = new Set(chunkDims.map((dim) => parseDim(dim).sym));
  const bindings = planBindings(graph, options.bindings, chunkSymbols);
  const state = stateEstimate(graph, options.generation);
  const weights = weightEstimate(residency);

  const chunkLength = options.generation?.chunkLength;
  if (chunkLength === undefined && chunkDims.length > 0) {
    // states 形ノードを持つグラフは GenerationContext 無しでは走らない（ADR 0066 決定 1）。
    // ここで落とさないと、chunk 記号が未束縛のまま planGraph の次元評価で落ちて、束縛点を
    // 名乗らない DimError になる。
    throw new ExecutionError(
      `物理 chunk 行 M の記号 [${chunkDims.join(", ")}] は options.generation.chunkLength で` +
        "しか束縛できない（states 形ノードを持つグラフは GenerationContext が要る）",
    );
  }
  // ADR 0066 決定 4 の実行 2 形をそれぞれ独立に計算する（chunk 行だけが違う同じグラフ）。
  const plans: readonly (readonly [AdmissionScenarioName, SymbolBindings])[] =
    chunkLength === undefined ? [["run", bindings]] : [
      ["prefill", bindChunkRows(bindings, chunkDims, chunkLength, "prefill")],
      ["decode", bindChunkRows(bindings, chunkDims, 1, "decode")],
    ];

  const scenarios = plans.map(([name, scenarioBindings]): AdmissionScenario => {
    // MUST: states と入力の両方に現れる記号は 2 つの束縛点で同じ値（run が拒否する分裂 —
    // ADR 0066 追記 7 — に見積りだけが正常値を返さない）。シナリオごとに解決値が変わるので
    // 照合もシナリオごと。
    if (state.bindings !== undefined) assertGenerationBindings(state.bindings, scenarioBindings);
    const plan = planGraph(graph, scenarioBindings, state.shapes);
    // 入力バッファと出力 readback staging は同じ床つきの式（executor の `#bindInput` /
    // `#activateBacking` / `#stageOutputs` — `Math.max(4, numel×4)` は toSizeClass と同値）。
    let ioBytes = 0;
    for (const spec of graph.inputs) {
      ioBytes += toSizeClass(numel(shapeOf(plan.shapes, spec.name)) * 4);
    }
    for (const name of graph.outputs) ioBytes += toSizeClass(numel(shapeOf(plan.shapes, name)) * 4);
    return { name, ioBytes, workspaceBytes: transientSlotBytes(graph, plan.nodes) };
  });

  const weightBytes = weights.compressed + weights.uncompressed + weights.expanded;
  const peakScenarioBytes = Math.max(
    ...scenarios.map((scenario) => scenario.ioBytes + scenario.workspaceBytes),
  );
  return {
    resident: {
      weights: {
        compressedBytes: weights.compressed,
        uncompressedBytes: weights.uncompressed,
        expandedBytes: weights.expanded,
        totalBytes: weightBytes,
      },
      stateBytes: state.bytes,
    },
    scenarios,
    peakAccountedBytes: weightBytes + state.bytes + peakScenarioBytes,
    unaccounted: UNACCOUNTED,
  };
};
