// 中間バッファの静的 liveness パッキング（src/runtime/transient-plan.ts — ADR 0093）の門。
// GPU 不要の純関数。縛るのは「配置の正しさ」（時間区間が重なる slot は同じ領域で空間が重ならない・
// 同じ dispatch で読み書きが分かれる slot は別領域）と「規則そのもの」（別名・pinned・一時の LIFO・
// 上限の fail loudly・決定性）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  planTransients,
  type TransientLimits,
  type TransientPlan,
  type TransientProgram,
  type TransientRef,
  type TransientStepSpec,
} from "../src/runtime/transient-plan.ts";
import { ExecutionError } from "../src/runtime/plan.ts";

/** 小さな形を見るための緩い上限（整列 4 = 実寸そのまま）。 */
const LOOSE: TransientLimits = {
  maxBufferSize: 1 << 30,
  maxStorageBufferBindingSize: 1 << 30,
  offsetAlignment: 4,
};

const value = (name: string): TransientRef => ({ kind: "value", name });
const temp = (id: number): TransientRef => ({ kind: "temp", id });

/** 1 出力・1 dispatch の素のノード（入力 `ins` を読み、出力 `name` を書く）。 */
const node = (
  name: string,
  byteLength: number,
  ins: readonly string[],
  uses: number,
  options: { pinned?: boolean; temps?: TransientStepSpec["temps"]; dispatches?: number } = {},
): TransientStepSpec => {
  const dispatches = options.dispatches ?? 1;
  return {
    outputs: [{ name, kind: "alloc", byteLength, uses, pinned: options.pinned ?? false }],
    temps: options.temps ?? [],
    dispatches: Array.from({ length: dispatches }, (_, index) => ({
      reads: [
        ...ins.map(value),
        ...(options.temps ?? []).flatMap((t, id) =>
          t.allocBefore < index && t.releaseAfter >= index ? [temp(id)] : []
        ),
      ],
      writes: [
        value(name),
        ...(options.temps ?? []).flatMap((t, id) => (t.allocBefore === index ? [temp(id)] : [])),
      ],
    })),
    releases: [...ins],
  };
};

/**
 * 配置の不変条件を全ペアで検査する（テストの側の検出器 — 実装がこれを破ると例外なしに
 * 値が壊れるので、ここで必ず落とす）。
 */
const assertWellFormed = (
  plan: TransientPlan,
  program: TransientProgram,
  limits: TransientLimits,
): void => {
  // 区間を再生し直す（実装と独立に、ステップ境界の粗い区間で見る: 出力はステップ開始〜最終消費ステップ末尾）。
  const lastUse = new Map<string, number>();
  program.forEach((step, index) => {
    for (const name of step.releases) lastUse.set(name, index);
    for (const d of step.dispatches) {
      for (const r of [...d.reads, ...d.writes]) {
        if (r.kind === "value") lastUse.set(r.name, Math.max(lastUse.get(r.name) ?? -1, index));
      }
    }
  });
  const live: { slot: number; from: number; to: number }[] = [];
  program.forEach((step, index) => {
    step.outputs.forEach((output, position) => {
      const slot = plan.steps[index].outputs[position];
      if (slot === undefined || output.kind === "alias") return;
      const to = output.pinned ? Infinity : Math.max(index, lastUse.get(output.name) ?? index);
      live.push({ slot, from: index, to });
    });
    // 一時は dispatch 粒度の区間（同じステップ内で解放済みの一時のバイトは次の一時が使ってよい）。
    const n = step.dispatches.length + 1;
    plan.steps[index].temps.forEach((slot, id) => {
      const t = step.temps[id];
      live.push({ slot, from: index + t.allocBefore / n, to: index + (t.releaseAfter + 0.5) / n });
    });
  });
  for (let a = 0; a < live.length; a += 1) {
    for (let b = a + 1; b < live.length; b += 1) {
      const x = live[a], y = live[b];
      if (x.slot === y.slot) continue;
      const sx = plan.slots[x.slot], sy = plan.slots[y.slot];
      const timeOverlap = x.from <= y.to && y.from <= x.to;
      if (!timeOverlap || sx.region !== sy.region) continue;
      const spaceOverlap = sx.offset < sy.offset + sy.byteLength &&
        sy.offset < sx.offset + sx.byteLength;
      assert(
        !spaceOverlap,
        `slot ${x.slot} と ${y.slot} が同じ領域 ${sx.region} で時間も空間も重なる`,
      );
    }
  }
  for (const slot of plan.slots) {
    assertEquals(slot.offset % limits.offsetAlignment, 0, "offset は整列済み");
    assert(slot.offset + slot.byteLength <= plan.regions[slot.region], "領域からはみ出さない");
    assert(slot.byteLength <= limits.maxStorageBufferBindingSize, "束縛上限の内側");
  }
  assertEquals(plan.totalBytes, plan.regions.reduce((t, r) => t + r, 0));
  assert(plan.totalBytes >= plan.peakLiveBytes, "領域総和は生存ピーク以上");
};

Deno.test("鎖 8 → 12 → 4 バイト: 厳密一致プールでは 24 だが、パッキングは生存ピーク 20 に一致する", () => {
  const program: TransientProgram = [
    node("a", 8, [], 1),
    node("b", 12, ["a"], 1),
    node("c", 4, ["b"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  assertEquals(plan.peakLiveBytes, 20);
  assertEquals(plan.totalBytes, 20);
  // a は b の dispatch で読まれながら b が書かれる → 別領域。c は a の領域（a は既に解放済み）へ入る。
  assertEquals(plan.regions, [12, 8]);
  const [c] = plan.steps[2].outputs;
  assert(c !== undefined);
  assertEquals(plan.pinned, new Set([c]));
});

Deno.test("同じ dispatch で読まれる入力と書かれる出力は別の領域へ置かれる（usage scope の制約）", () => {
  // a（8B）を読みながら b（8B）を書く。時間は重なる（a は b の後で解放）ので同じ領域なら別 offset で
  // 置けるが、read + read_write の同居は validation で落ちるため別領域が要る。
  const program: TransientProgram = [node("a", 8, [], 1), node("b", 8, ["a"], 0, { pinned: true })];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  const [a] = plan.steps[0].outputs, [b] = plan.steps[1].outputs;
  assert(a !== undefined && b !== undefined);
  assert(plan.slots[a].region !== plan.slots[b].region, "読み側と書き側が同じ領域に居る");
});

Deno.test("読み同士は同じ領域を共有できる（3 入力の add でも領域は 2 本で足りる）", () => {
  const program: TransientProgram = [
    node("a", 8, [], 1),
    node("b", 8, [], 1),
    node("c", 8, [], 1),
    node("d", 8, ["a", "b", "c"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  assertEquals(plan.regions.length, 2);
});

Deno.test("別名（reshape）は確保を出さず、根の slot の生存を延ばす", () => {
  const program: TransientProgram = [
    node("a", 8, [], 1),
    {
      outputs: [{ name: "v", kind: "alias", byteLength: 0, source: "a", uses: 1, pinned: false }],
      temps: [],
      dispatches: [],
      releases: ["a"],
    },
    node("b", 8, ["v"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  assertEquals(plan.slots.length, 2, "別名は slot を作らない");
  assertEquals(plan.steps[1].outputs[0], undefined);
  // a は v 越しに b が読むまで生きる → b と別領域（読み書きの分離）。
  const [a] = plan.steps[0].outputs, [b] = plan.steps[2].outputs;
  assert(a !== undefined && b !== undefined && plan.slots[a].region !== plan.slots[b].region);
});

Deno.test("グラフ出力（pinned）は末尾まで生存し、後続の確保に上書きされない", () => {
  const program: TransientProgram = [
    node("out1", 8, [], 0, { pinned: true }),
    node("x", 8, [], 1),
    node("out2", 8, ["x"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  assertEquals(plan.peakLiveBytes, 24);
  assertEquals(plan.totalBytes, 24);
});

Deno.test("一時は dispatch 境界で確保・解放され、解放済みの一時のバイトは次の一時が再利用する", () => {
  // 2 dispatch: 一時 0 は dispatch 0 の間だけ、一時 1 は dispatch 1 の間だけ生きる → 同じバイト。
  const program: TransientProgram = [
    node("y", 4, [], 0, {
      pinned: true,
      dispatches: 2,
      temps: [
        { byteLength: 16, allocBefore: 0, releaseAfter: 0 },
        { byteLength: 16, allocBefore: 1, releaseAfter: 1 },
      ],
    }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  const [t0, t1] = plan.steps[0].temps;
  assertEquals(plan.slots[t0].region, plan.slots[t1].region);
  assertEquals(plan.slots[t0].offset, plan.slots[t1].offset);
  assertEquals(plan.peakLiveBytes, 20);
});

Deno.test("領域は maxBufferSize を超えず、収まらない slot は次の領域へ回る", () => {
  const limits: TransientLimits = {
    maxBufferSize: 128,
    maxStorageBufferBindingSize: 128,
    offsetAlignment: 4,
  };
  const program: TransientProgram = [
    node("a", 100, [], 1),
    node("b", 100, [], 1),
    node("c", 4, ["a", "b"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, limits);
  assertWellFormed(plan, program, limits);
  assert(plan.regions.every((size) => size <= 128));
  assert(plan.regions.length >= 3, `a / b / c が 3 領域に分かれる（${plan.regions.length}）`);
});

Deno.test("offset は offsetAlignment に整列する（実寸は整列しない）", () => {
  const limits: TransientLimits = { ...LOOSE, offsetAlignment: 256 };
  const program: TransientProgram = [
    node("a", 4, [], 1),
    node("b", 4, [], 1),
    node("c", 4, ["a", "b"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, limits);
  assertWellFormed(plan, program, limits);
  const offsets = plan.slots.map((s) => s.offset);
  assert(offsets.every((o) => o % 256 === 0));
  assert(plan.slots.every((s) => s.byteLength === 4), "束縛 size は実寸のまま");
});

Deno.test("束縛上限を超える中間は配置の前に全件列挙して落ちる", () => {
  const limits: TransientLimits = {
    maxBufferSize: 1 << 20,
    maxStorageBufferBindingSize: 64,
    offsetAlignment: 4,
  };
  const program: TransientProgram = [
    node("big1", 100, [], 1),
    node("big2", 200, ["big1"], 0, { pinned: true }),
  ];
  const error = assertThrows(() => planTransients(program, limits), ExecutionError);
  assert(
    error.message.includes("'big1' 100B") && error.message.includes("'big2' 200B"),
    error.message,
  );
});

Deno.test("消費計数の誤り（解放過多 / 解放漏れ）は fail loudly", () => {
  const over: TransientProgram = [node("a", 8, [], 0), {
    outputs: [],
    temps: [],
    dispatches: [],
    releases: ["a"],
  }];
  assertThrows(() => planTransients(over, LOOSE), ExecutionError, "参照カウントが負");
  const leak: TransientProgram = [node("a", 8, [], 2), node("b", 8, ["a"], 0, { pinned: true })];
  assertThrows(() => planTransients(leak, LOOSE), ExecutionError, "解放されない中間");
});

Deno.test("同じプログラムからは常に同じ計画が出る（決定性）", () => {
  const program: TransientProgram = [
    node("a", 8, [], 2),
    node("b", 12, ["a"], 1),
    node("c", 8, ["a", "b"], 0, { pinned: true }),
  ];
  assertEquals(planTransients(program, LOOSE), planTransients(program, LOOSE));
});

Deno.test("疑似乱数で組んだ 200 ステップの鎖でも配置の不変条件を破らない（fuzz）", () => {
  let seed = 20260905;
  const rand = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let round = 0; round < 8; round += 1) {
    const program: TransientStepSpec[] = [];
    const names: string[] = [];
    const remaining = new Map<string, number>();
    for (let i = 0; i < 200; i += 1) {
      const candidates = names.filter((n) => (remaining.get(n) ?? 0) > 0);
      const ins = candidates.filter(() => rand() < 0.4).slice(0, 3);
      for (const n of ins) remaining.set(n, (remaining.get(n) ?? 0) - 1);
      const name = `v${i}`;
      const uses = i > 190 ? 0 : 1 + Math.floor(rand() * 3);
      const bytes = 4 * (1 + Math.floor(rand() * 64));
      const temps = rand() < 0.3
        ? [{ byteLength: 4 * (1 + Math.floor(rand() * 16)), allocBefore: 0, releaseAfter: 0 }]
        : [];
      program.push(node(name, bytes, ins, uses, { pinned: uses === 0, temps }));
      names.push(name);
      remaining.set(name, uses);
    }
    // 未消費の残りは末尾で読み切る（解放漏れを作らない）。
    const tail = names.filter((n) => (remaining.get(n) ?? 0) > 0);
    const tailReleases = tail.flatMap((n) =>
      Array.from({ length: remaining.get(n) ?? 0 }, () => n)
    );
    program.push({
      outputs: [{ name: "sink", kind: "alloc", byteLength: 4, uses: 0, pinned: true }],
      temps: [],
      dispatches: [{ reads: tail.map(value), writes: [value("sink")] }],
      releases: tailReleases,
    });
    const limits: TransientLimits = {
      maxBufferSize: 4096,
      maxStorageBufferBindingSize: 4096,
      offsetAlignment: 16,
    };
    const plan = planTransients(program, limits);
    assertWellFormed(plan, program, limits);
    assert(
      plan.totalBytes < 6 * plan.peakLiveBytes,
      `断片化が異常（${plan.totalBytes} vs 生存ピーク ${plan.peakLiveBytes}）`,
    );
  }
});
