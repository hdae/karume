/**
 * 中間バッファ（ノード出力・ノード内一時）の**静的 liveness パッキング**（ADR 0093）。
 *
 * 確保プログラム（{@link TransientProgram} — レシピ列または融合前ノード列から組む）を、実行相と
 * 同じイベント順で再生して各論理テンソルの生存区間と実寸を取り、時間区間が重ならないテンソル同士が
 * 同じバイトを共有できるように領域バッファの offset へ配置する。返るのは領域の本数と大きさ・
 * slot ごとの `(region, offset, byteLength)`・ステップごとの slot 割当。
 *
 * MUST: 純関数（GPU 資源を作らず、入力も変更しない）。実行（recipe.ts）・焼き込み（backing）・
 * 見積り（estimate.ts）は**この 1 本**を共有する — 規則が 2 箇所に分かれると、確保と見積り
 * （または確保と束縛）が例外なしに別の数を主張する。
 * MUST: 再生の順序は `executeStepRecipe` / 旧 `derivePlanSlots` と同一 — ステップ出力の確保
 * （slot 昇順）→ dispatch ごとに一時の確保（`allocBefore`）→ dispatch → 一時の解放（`releaseAfter`・
 * 確保の逆順）→ 入力の解放（延べ列）→ 定義ぶんの解放（出力 slot 昇順）。順序が割れると
 * 「まだ読まれる入力が出力として配り直される」形が例外なしに生まれる。
 * MUST: **同じ dispatch で片方が読まれ他方が書かれる 2 テンソルは別の領域**に置く。WebGPU の
 * usage scope 検証は GPUBuffer 単位（束縛範囲は見ない）で、同じバッファを `read`
 * （read-only-storage）と `read_write`（storage）の両方で 1 dispatch に束ねると validation で
 * 落ちる。読み同士・書き同士の同居は合法。
 * MUST: 実寸は {@link toSizeClass}（4 バイト整列・4 バイト床）— 束縛 `size` も offset もこの単位。
 * offset は `offsetAlignment`（device の `minStorageBufferOffsetAlignment`・仕様既定 256）に整列する。
 * MUST: 上限超過（slot 実寸 > `maxStorageBufferBindingSize`・領域 > `maxBufferSize` に収まらない
 * slot）は**配置の前に全件列挙して**落とす（ADR 0093 決定 5 — errorScope より前の決定論的検査）。
 *
 * 計算量: slot 数 n に対し配置は O(n · 領域内の重なり候補) で、実資産（n ≈ 2,000）でもミリ秒級。
 * 決定性: 並び順は (実寸 降順, 確保イベント 昇順) で固定し、同じ入力からは常に同じ計画が出る。
 */

import { toSizeClass } from "../gpu/arena.ts";
import { ExecutionError } from "./plan.ts";

/** 再生で束縛先を指す参照（値名 = ステップ出力・グラフ入力・重み / 一時 = 同一ステップ内の添字）。 */
export type TransientRef =
  | { readonly kind: "value"; readonly name: string }
  | { readonly kind: "temp"; readonly id: number };

/** ステップ出力 1 本の確保仕様（`StepOutput` の計画向きの写し）。 */
export type TransientOutputSpec = {
  readonly name: string;
  /** `alias` は入力実体をそのまま出力にする形（reshape / 恒等 expand）— 確保が出ない。 */
  readonly kind: "alloc" | "alias";
  /** `alloc` の実寸（整列前）。`alias` では読まない。 */
  readonly byteLength: number;
  /** `alias` の元の値名（グラフ入力 / 重みなど slot を持たない値は再生側で undefined になる）。 */
  readonly source?: string;
  /** 出力値の将来の消費回数（`node.ins` の厳密な延べ計数）。 */
  readonly uses: number;
  /** グラフ出力（末尾まで生存・他の値と共有しない）。 */
  readonly pinned: boolean;
};

/** ノード内一時の確保仕様（`TempRecipe` と同形）。 */
export type TransientTempSpec = {
  readonly byteLength: number;
  readonly allocBefore: number;
  readonly releaseAfter: number;
};

/**
 * dispatch 1 本の読み / 書きの役割（usage scope の制約に使う）。役割はカーネルの WGSL 宣言
 * （`var<storage, read>` / `read_write`）から採る — 手書きの表を持たない。
 */
export type TransientDispatchSpec = {
  readonly reads: readonly TransientRef[];
  readonly writes: readonly TransientRef[];
};

/** 実行ステップ 1 つぶんの確保プログラム。 */
export type TransientStepSpec = {
  readonly outputs: readonly TransientOutputSpec[];
  readonly temps: readonly TransientTempSpec[];
  readonly dispatches: readonly TransientDispatchSpec[];
  /** ステップ末尾に解放する入力の**延べ列**。 */
  readonly releases: readonly string[];
};

export type TransientProgram = readonly TransientStepSpec[];

/** 計画が従う device の上限（granted 値。見積りは WebGPU core 既定を渡す）。 */
export type TransientLimits = {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly offsetAlignment: number;
};

/** 配置済み slot（領域添字・領域内 offset・束縛 size = 整列済み実寸）。 */
export type TransientSlot = {
  readonly region: number;
  readonly offset: number;
  readonly byteLength: number;
};

/** ステップごとの slot 割当（`alias` の出力は undefined）。 */
export type TransientStepSlots = {
  readonly outputs: readonly (number | undefined)[];
  readonly temps: readonly number[];
};

export type TransientPlan = {
  /** 領域ごとのバイト数（`createBuffer` する大きさ・4 バイト整列）。 */
  readonly regions: readonly number[];
  readonly slots: readonly TransientSlot[];
  readonly steps: readonly TransientStepSlots[];
  /** グラフ出力の slot（readback を許す集合）。 */
  readonly pinned: ReadonlySet<number>;
  /** 領域の総和（= 中間に要する GPU バイト数）。 */
  readonly totalBytes: number;
  /** 同時生存バイトの最大（下界 — first-fit がこれに一致すれば断片化ゼロ）。 */
  readonly peakLiveBytes: number;
};

/** WebGPU core の保証既定（見積りが device 無しで使う）。 */
export const CORE_TRANSIENT_LIMITS: TransientLimits = {
  maxBufferSize: 256 * 1024 * 1024,
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  offsetAlignment: 256,
};

type Interval = {
  readonly byteLength: number;
  readonly start: number;
  end: number;
  readonly label: string;
  pinned: boolean;
};

const overlaps = (a: Interval, b: Interval): boolean => a.start < b.end && b.start < a.end;

const pairKey = (a: number, b: number): number => (a < b ? a * 0x100000 + b : b * 0x100000 + a);

const assertLimits = (limits: TransientLimits): void => {
  for (const [name, value] of Object.entries(limits)) {
    if (!Number.isSafeInteger(value) || value < 4) {
      throw new ExecutionError(`transient 計画の上限 ${name} が不正: ${value}`);
    }
  }
  if (limits.offsetAlignment % 4 !== 0) {
    throw new ExecutionError(
      `transient 計画の offsetAlignment は 4 の倍数: ${limits.offsetAlignment}`,
    );
  }
};

/**
 * 確保プログラムを再生して配置する（モジュール doc の MUST 群）。
 *
 * 再生の簿記（参照計数）は旧 `derivePlanSlots` と同じ: 定義ぶんの 1 + `uses` を retain し、
 * 入力の解放と定義ぶんの解放で減らす。負になれば消費計数の誤り、末尾に残れば解放漏れ — どちらも
 * fail loudly（pinned は残ってよい）。
 */
export const planTransients = (
  program: TransientProgram,
  limits: TransientLimits,
): TransientPlan => {
  assertLimits(limits);
  const intervals: Interval[] = [];
  const refs = new Map<number, number>();
  const env = new Map<string, number>();
  const conflicts = new Set<number>();
  const steps: TransientStepSlots[] = [];
  let clock = 0;

  const alloc = (byteLength: number, label: string): number => {
    const id = intervals.length;
    intervals.push({
      byteLength: toSizeClass(byteLength),
      start: clock,
      end: Infinity,
      label,
      pinned: false,
    });
    clock += 1;
    return id;
  };
  const retain = (slot: number | undefined, count: number, pinned: boolean): void => {
    if (slot === undefined) return;
    if (pinned) intervals[slot].pinned = true;
    refs.set(slot, (refs.get(slot) ?? 0) + count + 1);
  };
  const release = (slot: number | undefined): void => {
    if (slot === undefined) return;
    const left = (refs.get(slot) ?? 0) - 1;
    if (left < 0) {
      throw new ExecutionError(
        `transient 計画: '${
          intervals[slot].label
        }' の参照カウントが負（解放過多 — 消費計数の誤り）`,
      );
    }
    refs.set(slot, left);
    if (left > 0 || intervals[slot].pinned) return;
    intervals[slot].end = clock;
    clock += 1;
  };
  const resolve = (ref: TransientRef, temps: readonly number[]): number | undefined =>
    ref.kind === "temp" ? temps[ref.id] : env.get(ref.name);

  program.forEach((step, stepIndex) => {
    const outputs = step.outputs.map((output) => {
      const slot = output.kind === "alias"
        ? (output.source === undefined ? undefined : env.get(output.source))
        : alloc(output.byteLength, output.name);
      retain(slot, output.uses, output.pinned);
      if (slot === undefined) env.delete(output.name);
      else env.set(output.name, slot);
      return slot;
    });
    const temps: number[] = [];
    step.dispatches.forEach((dispatch, index) => {
      step.temps.forEach((temp, id) => {
        if (temp.allocBefore !== index) return;
        const slot = alloc(temp.byteLength, `ステップ ${stepIndex} の一時 ${id}`);
        retain(slot, 0, false);
        temps[id] = slot;
      });
      // usage scope: この dispatch で読まれる slot と書かれる slot は同じ領域に置けない。
      const reads = dispatch.reads.map((ref) => resolve(ref, temps)).filter((s): s is number =>
        s !== undefined
      );
      const writes = dispatch.writes.map((ref) => resolve(ref, temps)).filter((s): s is number =>
        s !== undefined
      );
      for (const r of reads) for (const w of writes) if (r !== w) conflicts.add(pairKey(r, w));
      for (let id = step.temps.length - 1; id >= 0; id -= 1) {
        if (step.temps[id].releaseAfter === index) release(temps[id]);
      }
    });
    for (const name of step.releases) release(env.get(name));
    for (const slot of outputs) release(slot);
    // 別名の出力は slot を持たない（束縛は ValueSource の元をそのまま解決する — 旧 derivePlanSlots と同じ契約）。
    steps.push({
      outputs: step.outputs.map((output, index) =>
        output.kind === "alias" ? undefined : outputs[index]
      ),
      temps,
    });
  });

  const leaked = [...refs].filter(([slot, count]) => count > 0 && !intervals[slot].pinned);
  if (leaked.length > 0) {
    throw new ExecutionError(
      `transient 計画: 解放されない中間が残る（消費計数の誤り）: ${
        leaked.map(([slot, count]) => `'${intervals[slot].label}'（refs=${count}）`).join(" / ")
      }`,
    );
  }

  // 上限は配置の前に全件列挙して落とす（ADR 0093 決定 5）。
  const bindingCap = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  const oversized = intervals.filter((iv) => iv.byteLength > bindingCap);
  if (oversized.length > 0) {
    throw new ExecutionError(
      `中間バッファが device の上限を超える（maxStorageBufferBindingSize ${limits.maxStorageBufferBindingSize} / ` +
        `maxBufferSize ${limits.maxBufferSize}）: ${
          oversized.map((iv) => `'${iv.label}' ${iv.byteLength}B`).join(" / ")
        }`,
    );
  }

  // 配置: 実寸の大きい順（同寸は確保順）に first-fit。
  const order = intervals.map((_, id) => id).sort((a, b) =>
    intervals[b].byteLength - intervals[a].byteLength || intervals[a].start - intervals[b].start ||
    a - b
  );
  const regions: { placed: number[]; size: number }[] = [];
  const slots: TransientSlot[] = new Array(intervals.length);
  const align = (value: number): number =>
    Math.ceil(value / limits.offsetAlignment) * limits.offsetAlignment;
  for (const id of order) {
    const iv = intervals[id];
    let placed = false;
    for (let region = 0; region < regions.length && !placed; region += 1) {
      const entry = regions[region];
      if (entry.placed.some((other) => conflicts.has(pairKey(id, other)))) continue;
      const busy = entry.placed
        .filter((other) => overlaps(iv, intervals[other]))
        .map((other) => slots[other])
        .sort((a, b) => a.offset - b.offset);
      let offset = 0;
      for (const b of busy) {
        if (offset + iv.byteLength <= b.offset) break;
        offset = Math.max(offset, align(b.offset + b.byteLength));
      }
      if (offset + iv.byteLength > limits.maxBufferSize) continue;
      slots[id] = { region, offset, byteLength: iv.byteLength };
      entry.placed.push(id);
      entry.size = Math.max(entry.size, offset + iv.byteLength);
      placed = true;
    }
    if (!placed) {
      slots[id] = { region: regions.length, offset: 0, byteLength: iv.byteLength };
      regions.push({ placed: [id], size: iv.byteLength });
    }
  }

  // 生存ピーク（下界）。
  let peakLiveBytes = 0;
  for (const iv of intervals) {
    let live = 0;
    for (const other of intervals) {
      if (other.start <= iv.start && iv.start < other.end) live += other.byteLength;
    }
    peakLiveBytes = Math.max(peakLiveBytes, live);
  }

  const regionBytes = regions.map((entry) => toSizeClass(entry.size));
  return {
    regions: regionBytes,
    slots,
    steps,
    pinned: new Set(intervals.flatMap((iv, id) => (iv.pinned ? [id] : []))),
    totalBytes: regionBytes.reduce((total, size) => total + size, 0),
    peakLiveBytes,
  };
};
