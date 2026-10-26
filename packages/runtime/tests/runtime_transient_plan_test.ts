// 中間バッファの静的 liveness パッキング（src/runtime/transient-plan.ts — ADR 0093）の門。
// GPU 不要の純関数。縛るのは「配置の正しさ」（時間区間が重なる slot は同じ領域で空間が重ならない・
// 同じ dispatch で読み書きが分かれる slot は別領域）と「規則そのもの」（別名・pinned・一時の LIFO・
// 上限の fail loudly・決定性）。

import { test } from "vite-plus/test";
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
          t.allocBefore < index && t.releaseAfter >= index ? [temp(id)] : [],
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
      const x = live[a],
        y = live[b];
      if (x.slot === y.slot) continue;
      const sx = plan.slots[x.slot],
        sy = plan.slots[y.slot];
      const timeOverlap = x.from <= y.to && y.from <= x.to;
      if (!timeOverlap || sx.region !== sy.region) continue;
      const spaceOverlap =
        sx.offset < sy.offset + sy.byteLength && sy.offset < sx.offset + sx.byteLength;
      assert(
        !spaceOverlap,
        `slot ${x.slot} と ${y.slot} が同じ領域 ${sx.region} で時間も空間も重なる`,
      );
    }
  }
  // Usage scope, resolved from the program through the plan's per-step slots (not from the
  // planner's conflict lists): a slot read and a slot written by the same dispatch never share a
  // region. An alias output resolves to its source's slot, a name without a slot (graph input,
  // weight) to nothing.
  const env = new Map<string, number>();
  program.forEach((step, index) => {
    step.outputs.forEach((output, position) => {
      const slot =
        output.kind === "alias"
          ? output.source === undefined
            ? undefined
            : env.get(output.source)
          : plan.steps[index].outputs[position];
      if (slot === undefined) env.delete(output.name);
      else env.set(output.name, slot);
    });
    const slotOf = (ref: TransientRef): number | undefined =>
      ref.kind === "temp" ? plan.steps[index].temps[ref.id] : env.get(ref.name);
    step.dispatches.forEach((d, dispatchIndex) => {
      for (const read of d.reads) {
        for (const write of d.writes) {
          const r = slotOf(read),
            w = slotOf(write);
          if (r === undefined || w === undefined || r === w) continue;
          assert(
            plan.slots[r].region !== plan.slots[w].region,
            `step ${index} dispatch ${dispatchIndex} reads slot ${r} and writes slot ${w} in region ${plan.slots[r].region}`,
          );
        }
      }
    });
  });
  for (const slot of plan.slots) {
    assertEquals(slot.offset % limits.offsetAlignment, 0, "offset は整列済み");
    assert(slot.offset + slot.byteLength <= plan.regions[slot.region], "領域からはみ出さない");
    assert(slot.byteLength <= limits.maxStorageBufferBindingSize, "束縛上限の内側");
  }
  for (const [region, size] of plan.regions.entries()) {
    assert(
      size <= limits.maxBufferSize,
      `region ${region} (${size} B) exceeds maxBufferSize ${limits.maxBufferSize}`,
    );
  }
  assertEquals(
    plan.totalBytes,
    plan.regions.reduce((t, r) => t + r, 0),
  );
  assert(plan.totalBytes >= plan.peakLiveBytes, "領域総和は生存ピーク以上");
};

test("鎖 8 → 12 → 4 バイト: 厳密一致プールでは 24 だが、パッキングは生存ピーク 20 に一致する", () => {
  const program: TransientProgram = [
    node("a", 8, [], 1),
    node("b", 12, ["a"], 1),
    node("c", 4, ["b"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  assertEquals(plan.peakLiveBytes, 20);
  assertEquals(plan.totalBytes, 20);
  // Intervals a [0,2), b [1,4), c [3,end). b (largest) opens region 0. a overlaps b, and behind b it
  // would grow region 0 by its full 8 B, so it opens region 1 (growth decides; the read / write rule
  // would also keep it out). c reads b, so it avoids region 0 and takes a's dead bytes in region 1
  // (growth 0 there, 4 behind b, so it would land there anyway). The total meets the live peak, so
  // only order 1 runs.
  assertEquals(plan.regions, [12, 8]);
  const [c] = plan.steps[2].outputs;
  assert(c !== undefined);
  assertEquals(plan.pinned, new Set([c]));
});

test("同じ dispatch で読まれる入力と書かれる出力は別の領域へ置かれる（usage scope の制約）", () => {
  // b (8 B) is written while a (8 B) is read. x (24 B, no consumer) is dead before either is born,
  // so both fit into x's bytes (a at 0, b at 8) without growing its region: only the usage-scope
  // rule keeps them apart (read + read_write on one GPUBuffer fails validation). Intervals x [0,1),
  // a [2,4), b [3,end); both orders place x, a, b. a takes offset 0 of x's region (growth 0); b may
  // not share it and opens region 1 = [24, 8]. Without the rule b sits at offset 8 and the total is
  // 24.
  const program: TransientProgram = [
    node("x", 24, [], 0),
    node("a", 8, [], 1),
    node("b", 8, ["a"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  const [x] = plan.steps[0].outputs,
    [a] = plan.steps[1].outputs,
    [b] = plan.steps[2].outputs;
  assert(x !== undefined && a !== undefined && b !== undefined);
  assertEquals(plan.slots[a].region, plan.slots[x].region);
  assert(plan.slots[a].region !== plan.slots[b].region, "読み側と書き側が同じ領域に居る");
  assertEquals(plan.regions, [24, 8]);
});

test("読み同士は同じ領域を共有できる（3 入力の add でも領域は 2 本で足りる）", () => {
  // a, b and c (8 B each, all read by d) are born after x (24 B) is dead, so they fit into x's bytes
  // at offsets 0 / 8 / 16 without growing its region. d conflicts with a, b and c and opens a second
  // region; y (4 B, written while x is read and read while a / b / c are written) joins d's region
  // (their lifetimes do not overlap). Both orders place x, a, b, c, d, y and give [24, 8] = the live
  // peak at d's allocation. Two regions are possible only when a, b and c share one.
  const program: TransientProgram = [
    node("x", 24, [], 1),
    node("y", 4, ["x"], 3),
    node("a", 8, ["y"], 1),
    node("b", 8, ["y"], 1),
    node("c", 8, ["y"], 1),
    node("d", 8, ["a", "b", "c"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  assertEquals(plan.regions.length, 2);
});

test("a slot goes to the region it grows least, using a later region's free bytes instead of growing an earlier one", () => {
  // a (100 B) lives to the end. b (60 B) is written while a is read, so it gets its own region. c (40 B)
  // is born after b is released and lives with a: a's region would grow by 40 (offset 100), b's region
  // holds it at offset 0 without growing. First-fit grew a's region (total 204 = 140 + 60 + 4).
  const program: TransientProgram = [
    node("a", 100, [], 2),
    node("b", 60, ["a"], 1),
    node("d", 4, ["b"], 1),
    node("c", 40, ["d"], 1),
    node("e", 4, ["a", "c"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  const [a] = plan.steps[0].outputs,
    [b] = plan.steps[1].outputs,
    [c] = plan.steps[3].outputs;
  assert(a !== undefined && b !== undefined && c !== undefined);
  assertEquals(plan.slots[c], { region: plan.slots[b].region, offset: 0, byteLength: 40 });
  // The live peak is at d's allocation (a, b and d live together: 100 + 60 + 4); the total meets it.
  assertEquals(plan.peakLiveBytes, 164);
  assertEquals(plan.regions, [100, 60, 4]);
});

test("among regions that would each grow by less than the slot's size, the slot takes the one that grows least", () => {
  // Intervals: a0 100 B [0,3), a1 100 B [1,4), k 4 B [2,6), b0 80 B [5,10), b1 72 B [7,11),
  // s 40 B [8,12), e 4 B [9,end). Read / write pairs: k–a0, k–a1, b0–k, b1–b0, e–b0, e–b1, e–s.
  // Sizes are distinct but for the 4 B pair, so both orders place a0, a1, b0, b1, s, k, e:
  // a0 opens region 0; a1 overlaps a0 (growth 100 = its size) and opens region 1; b0 lands on a0's
  // dead bytes (region 0, offset 0); b1 may not share b0's region and lands on a1's (region 1,
  // offset 0). s overlaps b0 and b1: region 0 grows by 80 + 40 - 100 = 20, region 1 by
  // 72 + 40 - 100 = 12, both below 40, so s takes region 1 at offset 72. k and e share region 2.
  // Taking the first region that grows by less than 40 (or first-fit) puts s at offset 80 of region
  // 0 instead: [120, 100, 4] = 224.
  const program: TransientProgram = [
    node("a0", 100, [], 1),
    node("a1", 100, [], 1),
    node("k", 4, ["a0", "a1"], 1),
    node("b0", 80, ["k"], 2),
    node("b1", 72, ["b0"], 1),
    node("s", 40, [], 1),
    node("e", 4, ["b0", "b1", "s"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  const [b1] = plan.steps[4].outputs,
    [s] = plan.steps[5].outputs;
  assert(b1 !== undefined && s !== undefined);
  assertEquals(plan.slots[s], { region: plan.slots[b1].region, offset: 72, byteLength: 40 });
  assertEquals(plan.regions, [100, 112, 4]);
});

test("a slot that would grow every region by its full size opens a new region instead of growing one", () => {
  // a and b (8 B each) live together and c reads both. Stacking b at offset 8 of a's region and opening
  // a new region both cost 8 bytes, so the planner takes the new region, which has no read / write
  // conflicts.
  const program: TransientProgram = [
    node("a", 8, [], 1),
    node("b", 8, [], 1),
    node("c", 4, ["a", "b"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  const [a] = plan.steps[0].outputs,
    [b] = plan.steps[1].outputs;
  assert(a !== undefined && b !== undefined);
  assert(plan.slots[a].region !== plan.slots[b].region, "b was stacked in a's region");
  assertEquals(plan.regions, [8, 8, 4]);
  assertEquals(plan.totalBytes, plan.peakLiveBytes);
});

test("when the order of equal sizes changes the total, the smaller placement (longer-lived first) is kept", () => {
  // a → b → c (8 B each; neighbours are read and written together, so they never share a region) →
  // d (12 B, a graph output). Intervals: a [0,2), b [1,4), c [3,6), d [5,end). In allocation order
  // (d, a, b, c) a joins d's region and b and c open one region each: 28. Longer-lived first
  // (d, b, c, a) puts b in d's region and a in c's region: 20 = the live peak (c + d).
  const program: TransientProgram = [
    node("a", 8, [], 1),
    node("b", 8, ["a"], 1),
    node("c", 8, ["b"], 1),
    node("d", 12, ["c"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  const [a, b, c, d] = program.map((_, index) => plan.steps[index].outputs[0]);
  assert(a !== undefined && b !== undefined && c !== undefined && d !== undefined);
  assertEquals(plan.slots[b].region, plan.slots[d].region);
  assertEquals(plan.slots[a].region, plan.slots[c].region);
  assertEquals(plan.regions, [12, 8]);
  assertEquals(plan.totalBytes, plan.peakLiveBytes);
});

test("when allocation order gives the smaller total, it is kept over the longer-lived-first order", () => {
  // Intervals: a 16 B [0,4), b 4 B [1,2), c 8 B [3,6), d 8 B [5,9), e 12 B [7,11), f 8 B [8,end)
  // (graph output), z 4 B [10,end) (graph output); the final clock is 12. Read / write pairs: b–a,
  // c–a, d–c, e–d, f–d, z–e. Live peak 28 (d + e + f when f is allocated).
  // Order 1 (a, e, c, d, f, b, z): a opens region 0 (16); e lands on a's dead bytes (region 0,
  // offset 0); c reads a and opens region 1 (8); d may share neither c's nor e's region and opens
  // region 2 (8); f grows region 0 by 12 + 8 - 16 = 4 but region 1 by 0 (c is dead), so it takes
  // region 1; b and z land on dead bytes in regions 1 and 2 = [16, 8, 8] = 32.
  // Order 2 (lifetimes d 4 / f 4 / c 3, z 2 / b 1: a, e, d, f, c, z, b): d opens region 1; f may not
  // share d's region and grows region 0 by 4 (offset 12, less than its 8 B); c may share neither
  // a's nor d's region and opens region 2 = [20, 8, 8] = 36. 32 < 36 keeps order 1.
  const program: TransientProgram = [
    node("a", 16, [], 2),
    node("b", 4, ["a"], 0),
    node("c", 8, ["a"], 1),
    node("d", 8, ["c"], 2),
    node("e", 12, ["d"], 1),
    node("f", 8, ["d"], 0, { pinned: true }),
    node("z", 4, ["e"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  const [c] = plan.steps[2].outputs,
    [f] = plan.steps[5].outputs;
  assert(c !== undefined && f !== undefined);
  assertEquals(plan.slots[f], { region: plan.slots[c].region, offset: 0, byteLength: 8 });
  assertEquals(plan.regions, [16, 8, 8]);
  assertEquals(plan.peakLiveBytes, 28);
});

test("order 2 measures a graph output's lifetime to the final clock, not as infinite", () => {
  // Intervals: a 16 B [0,2), b 16 B [1,6), c 4 B [3,7), d 4 B [4,end) and z 4 B [5,end) (graph
  // outputs); the final clock is 8, so the lifetimes are b 5 / a 2 and c 4 / d 4 / z 3. Read / write
  // pairs: b–a, d–c, z–b, z–c. Live peak 32 (a + b).
  // Order 1 (a, b, c, d, z): b reads a and opens region 1; c lands on a's dead bytes (region 0);
  // d may not share c's region and behind b would grow region 1 by its full 4 B, so it opens region
  // 2; z may share neither b's nor c's region and behind d grows region 2 by 4, so it opens region 3
  // = [16, 16, 4, 4] = 40. Order 2 (b, a, c, d, z — c and d tie at 4 and go by allocation) places
  // the 4 B slots the same way = 40, and the tie keeps order 1.
  // Taken as infinite, d and z would come before c: d lands on a's dead bytes, z fits behind it at
  // offset 4 without growing region 1 (16 B), and c, which may not share d's region, opens region 2
  // = [16, 16, 4] = 36, which order 2 would win.
  const program: TransientProgram = [
    node("a", 16, [], 1),
    node("b", 16, ["a"], 1),
    node("c", 4, [], 2),
    node("d", 4, ["c"], 0, { pinned: true }),
    node("z", 4, ["b", "c"], 0, { pinned: true }),
  ];
  const plan = planTransients(program, LOOSE);
  assertWellFormed(plan, program, LOOSE);
  assertEquals(plan.peakLiveBytes, 32);
  assertEquals(plan.regions, [16, 16, 4, 4]);
});

test("別名（reshape）は確保を出さず、根の slot の生存を延ばす", () => {
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
  // a lives until b reads it through v, so the two overlap: b is written while a is read, and
  // stacking b on a would grow the region by b's full size anyway — different regions.
  const [a] = plan.steps[0].outputs,
    [b] = plan.steps[2].outputs;
  assert(a !== undefined && b !== undefined && plan.slots[a].region !== plan.slots[b].region);
});

test("グラフ出力（pinned）は末尾まで生存し、後続の確保に上書きされない", () => {
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

test("一時は dispatch 境界で確保・解放され、解放済みの一時のバイトは次の一時が再利用する", () => {
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

test("領域は maxBufferSize を超えず、収まらない slot は次の領域へ回る", () => {
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

test("offset は offsetAlignment に整列する（実寸は整列しない）", () => {
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
  assert(
    plan.slots.every((s) => s.byteLength === 4),
    "束縛 size は実寸のまま",
  );
});

test("束縛上限を超える中間は配置の前に全件列挙して落ちる", () => {
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

test("消費計数の誤り（解放過多 / 解放漏れ）は fail loudly", () => {
  const over: TransientProgram = [
    node("a", 8, [], 0),
    {
      outputs: [],
      temps: [],
      dispatches: [],
      releases: ["a"],
    },
  ];
  assertThrows(() => planTransients(over, LOOSE), ExecutionError, "has a negative reference count");
  const leak: TransientProgram = [node("a", 8, [], 2), node("b", 8, ["a"], 0, { pinned: true })];
  assertThrows(
    () => planTransients(leak, LOOSE),
    ExecutionError,
    "intermediates are left unreleased",
  );
});

test("同じプログラムからは常に同じ計画が出る（決定性）", () => {
  const program: TransientProgram = [
    node("a", 8, [], 2),
    node("b", 12, ["a"], 1),
    node("c", 8, ["a", "b"], 0, { pinned: true }),
  ];
  assertEquals(planTransients(program, LOOSE), planTransients(program, LOOSE));
});

test("疑似乱数で組んだ 200 ステップの鎖でも配置の不変条件を破らない（fuzz）", () => {
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
      const temps =
        rand() < 0.3
          ? [{ byteLength: 4 * (1 + Math.floor(rand() * 16)), allocBefore: 0, releaseAfter: 0 }]
          : [];
      program.push(node(name, bytes, ins, uses, { pinned: uses === 0, temps }));
      names.push(name);
      remaining.set(name, uses);
    }
    // 未消費の残りは末尾で読み切る（解放漏れを作らない）。
    const tail = names.filter((n) => (remaining.get(n) ?? 0) > 0);
    const tailReleases = tail.flatMap((n) =>
      Array.from({ length: remaining.get(n) ?? 0 }, () => n),
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
