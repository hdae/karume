/**
 * レシピの**確保プログラム**（src/runtime/recipe.ts の `buildTransientProgram` → `planRecipes`）と
 * **宣言の静的検査**（同 `validateStepRecipe`）の構造テスト。GPU を 1 つも触らないので、
 * アダプタ無し環境でも走る。
 *
 * 見ているのは「{@link StepRecipe.outputs} の列と dispatch の束縛から、確保・retain・解放と
 * 読み書きの役割が計画（src/runtime/transient-plan.ts — ADR 0093）へ正しく写るか」— 実 GPU の
 * footprint 門（tests/gpu_plan_backing_test.ts が「backed の領域総和 = ミス run の領域確保」を
 * 突き合わせる）とは別の角度で、**期待値を手計算で固定する**。配置そのものの性質（first-fit・
 * usage scope の別領域・上限）は tests/runtime_transient_plan_test.ts が持つ。
 *
 * ADR 0068 決定 1 の受入条件「単一出力ノードのレシピ表現は列化前と同値」がここの 1 本目。
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { StorageRoles } from "../src/gpu/pipeline-cache.ts";
import type { SubmitScheduler } from "../src/gpu/submit.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  bakeAllBindGroups,
  type BakedGeneration,
  type BindingRecipe,
  type BindingSource,
  buildTransientProgram,
  type DispatchWorkgroups,
  executeBakedPlan,
  planRecipes,
  type StepOutput,
  type StepRecipe,
  type TempRecipe,
  validateStepRecipe,
} from "../src/runtime/recipe.ts";
import { CORE_TRANSIENT_LIMITS } from "../src/runtime/transient-plan.ts";

/** 出力列だけのステップ（dispatch と一時は確保簿記の別軸なので空で足りる）。 */
const step = (
  outputs: readonly StepOutput[],
  releases: readonly string[] = [],
): StepRecipe => ({ outputs, temps: [], dispatches: [], releases, writesState: false });

/** 領域から確保する出力（`byteLength` は 4 の倍数 = サイズクラスと同値にしておく）。 */
const alloc = (
  name: string,
  byteLength: number,
  uses: number,
  pinned = false,
): StepOutput => ({ kind: "alloc", byteLength, name, uses, pinned });

/** 別名の出力（入力実体をそのまま出力にする reshape / 恒等 expand — ADR 0011）。 */
const alias = (name: string, source: string, uses: number): StepOutput => ({
  kind: "alias",
  source: { kind: "value", name: source },
  name,
  uses,
  pinned: false,
});

/** `DispatchRecipe` は非公開型なので、レシピ列の要素型として引く。 */
type Dispatch = StepRecipe["dispatches"][number];

/** WGSL の宣言から採る役割のスタブ（binding 番号 → 読み / 読み書き）。 */
const roles = (reads: readonly number[], writes: readonly number[]): StorageRoles => ({
  reads: new Set(reads),
  writes: new Set(writes),
});

/**
 * dispatch のスタブ。宣言の静的検査と確保プログラムが読むのは `key` / `bindings` / `roles`
 * だけで、パイプラインと bind group layout は GPU 資源なので持たせない（型を満たすためだけの
 * 実体化を避ける）。
 */
const dispatch = (
  key: string,
  bindings: readonly BindingRecipe[] = [],
  access: StorageRoles = roles([], []),
): Dispatch => ({ key, bindings, roles: access }) as unknown as Dispatch;

Deno.test("単一出力ステップの確保簿記（alloc / alias / pin / 消費）が確保プログラムへ写る", () => {
  const program = buildTransientProgram([
    step([alloc("a", 64, 1)]),
    step([alloc("b", 64, 0)], []),
    step([alloc("c", 64, 0, true)], ["a"]),
    // 別名は根の名前を持ち、確保を出さない。
    step([alias("d", "c", 0)]),
    // 別名元が常駐（重み）なら根なし（slot を持たない値）。
    step([{
      kind: "alias",
      source: { kind: "resident", buffer: ({}) as unknown as GPUBuffer },
      name: "e",
      uses: 0,
      pinned: false,
    }]),
  ]);

  assertEquals(program.map((s) => s.outputs), [
    [{ name: "a", kind: "alloc", byteLength: 64, source: undefined, uses: 1, pinned: false }],
    [{ name: "b", kind: "alloc", byteLength: 64, source: undefined, uses: 0, pinned: false }],
    [{ name: "c", kind: "alloc", byteLength: 64, source: undefined, uses: 0, pinned: true }],
    [{ name: "d", kind: "alias", byteLength: 0, source: "c", uses: 0, pinned: false }],
    [{ name: "e", kind: "alias", byteLength: 0, source: undefined, uses: 0, pinned: false }],
  ]);
  assertEquals(program.map((s) => s.releases), [[], [], ["a"], [], []]);

  // 計画（256 整列）: a は c の確保より後（c を書く dispatch が a を読む）に生存を終えるので
  // a と c は重なり、c が掴めるのは消費者ゼロで先に生存を終えた b の区間（offset 256）。
  const plan = planRecipes([
    step([alloc("a", 64, 1)]),
    step([alloc("b", 64, 0)], []),
    step([alloc("c", 64, 0, true)], ["a"]),
    step([alias("d", "c", 0)]),
  ], CORE_TRANSIENT_LIMITS);
  assertEquals(plan.regions, [256 + 64]);
  assertEquals(plan.steps.map((assigned) => assigned.outputs), [[0], [1], [2], [undefined]]);
  assertEquals(plan.slots.map((slot) => slot.offset), [0, 256, 256]);
  assertEquals(plan.pinned, new Set([2]));
  assertEquals(plan.sharedSlots, 1, "c が b のバイトを配り直されている");
});

/**
 * `topk` の実形（ADR 0068 決定 3 の最初の入居者）。実 GPU 側の寿命門
 * （tests/runtime_executor_test.ts の「多出力ノードの slot ごとの uses / pin」）と**同じグラフ**
 * の計画を、GPU に触らない純関数の側から手計算で固定する:
 *
 * ```
 * topk(x[3,8], k=2) → v[3,2] f32（neg が 1 度消費）+ i[3,2] i32（グラフ出力）
 * neg(v) → nv[3,2]（exp が 1 度消費）
 * exp(nv) → e[3,2]（グラフ出力）
 * ```
 *
 * MUST: 2 本の出力は**同じバイト数**（どちらも 3·2 語の 4 バイト要素 = 24）— 総バイト数では
 * 取り違えに気づけない。だから **どの offset を掴むか**まで期待値に置く（`e` が掴むのは消費済みの
 * `v` の区間で、pin された添字 `i` の区間ではない）。
 */
Deno.test("topk の実形（値 + 添字・同一サイズ）の計画が手計算どおり", () => {
  const bytes = 3 * 2 * 4;
  const plan = planRecipes([
    step([alloc("v", bytes, 1), alloc("i", bytes, 0, true)]),
    step([alloc("nv", bytes, 1)], ["v"]),
    step([alloc("e", bytes, 0, true)], ["nv"]),
  ], CORE_TRANSIENT_LIMITS);

  // 実寸 24 は 256 整列の offset で並ぶ: v 0 / i 256 / nv 512（v が生きている間）/ e 0（v の跡）。
  assertEquals(plan.slots.map((slot) => slot.offset), [0, 256, 512, 0]);
  assertEquals(plan.steps.map((assigned) => assigned.outputs), [[0, 1], [2], [3]]);
  assertEquals(plan.pinned, new Set([1, 3]));
  // 生存ピーク = v / i / nv の 3 本ぶん（実 GPU 側の `peakTransientBytes` 72 と一致する）。
  assertEquals(plan.peakLiveBytes, 72);
});

Deno.test("多出力ステップは出力 slot 昇順に確保し、slot ごとに独立して retain / pin する", () => {
  const plan = planRecipes([
    step([alloc("v", 64, 1), alloc("i", 32, 0, true)]),
    step([alloc("y", 64, 0, true)], ["v"]),
  ], CORE_TRANSIENT_LIMITS);

  assertEquals(plan.slots.map((slot) => slot.byteLength), [64, 32, 64]);
  assertEquals(plan.steps.map((assigned) => assigned.outputs), [[0, 1], [2]]);
  assertEquals(plan.pinned, new Set([1, 2]));
});

// ---------------------------------------------------------------------------
// 読み書きの役割（WGSL 宣言 → 計画）
// ---------------------------------------------------------------------------

Deno.test("dispatch の読み / 書きは WGSL の役割から採り、常駐・state・論理長は計画に現れない", () => {
  const program = buildTransientProgram([{
    outputs: [alloc("y", 64, 0, true)],
    temps: [{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }],
    dispatches: [
      dispatch("k", [
        { binding: 1, source: { kind: "value", name: "x" } },
        { binding: 2, source: { kind: "resident", buffer: ({}) as unknown as GPUBuffer } },
        { binding: 3, source: { kind: "temp", id: 0 } },
        { binding: 4, source: { kind: "state", name: "kv.k" } },
        { binding: 5, source: { kind: "lengths" } },
        { binding: 6, source: { kind: "value", name: "y" } },
      ], roles([1, 2, 4, 5], [3, 6])),
    ],
    releases: [],
    writesState: false,
  }]);
  assertEquals(program[0].dispatches, [{
    reads: [{ kind: "value", name: "x" }],
    writes: [{ kind: "temp", id: 0 }, { kind: "value", name: "y" }],
  }]);
});

Deno.test("役割の無い束縛（WGSL に storage 宣言が無い binding）は fail loudly", () => {
  const recipes = [{
    outputs: [alloc("y", 64, 0, true)],
    temps: [],
    dispatches: [
      dispatch("gemm", [{ binding: 7, source: { kind: "value", name: "y" } }], roles([1], [2])),
    ],
    releases: [],
    writesState: false,
  }];
  const error = assertThrows(() => buildTransientProgram(recipes), ExecutionError);
  assert(
    error.message.includes("dispatch 'gemm' の束縛 7 に WGSL の storage 宣言が無い"),
    error.message,
  );
});

Deno.test("同じ dispatch で読む入力と書く出力は別領域に置かれる（計画まで通した形）", () => {
  const plan = planRecipes([
    step([alloc("h", 64, 1)]),
    {
      outputs: [alloc("y", 64, 0, true)],
      temps: [],
      dispatches: [
        dispatch("neg", [
          { binding: 1, source: { kind: "value", name: "h" } },
          { binding: 2, source: { kind: "value", name: "y" } },
        ], roles([1], [2])),
      ],
      releases: ["h"],
      writesState: false,
    },
  ], CORE_TRANSIENT_LIMITS);
  assertEquals(plan.regions, [64, 64]);
  assertEquals(plan.slots.map((slot) => slot.region), [0, 1]);
});

// ---------------------------------------------------------------------------
// 宣言の静的検査（validateStepRecipe）と計画の閉包検査
// ---------------------------------------------------------------------------

/** 一時と dispatch だけを差し替えるステップ（出力簿記は宣言検査の別軸なので固定）。 */
const tempStep = (temps: readonly TempRecipe[], dispatches: readonly Dispatch[]): StepRecipe => ({
  outputs: [alloc("y", 64, 0)],
  temps,
  dispatches,
  releases: [],
  writesState: false,
});

Deno.test("dispatch 列の内側で閉じていない一時の寿命宣言は宣言の受け口で落ちる", () => {
  const two = [dispatch("k0"), dispatch("k1")];
  const broken: readonly (readonly [TempRecipe, string])[] = [
    // 解放が来ないまま組み上がった一時（`releaseTemp` の呼び忘れ = 実行相で漏れる）。
    [{ byteLength: 64, allocBefore: 0, releaseAfter: -1 }, "一時 0 の寿命宣言 [0, -1]"],
    // 確保より前の dispatch で解放する形（束ねる dispatch が確保前に並ぶ）。
    [{ byteLength: 64, allocBefore: 1, releaseAfter: 0 }, "一時 0 の寿命宣言 [1, 0]"],
    // allocBefore が列の外（負）。
    [{ byteLength: 64, allocBefore: -1, releaseAfter: 1 }, "一時 0 の寿命宣言 [-1, 1]"],
    // releaseAfter が列の外（dispatch 2 本に対する添字 2）。
    [{ byteLength: 64, allocBefore: 0, releaseAfter: 2 }, "一時 0 の寿命宣言 [0, 2]"],
    // 確保仕様として意味を成さないバイト数。
    [{ byteLength: 0, allocBefore: 0, releaseAfter: 1 }, "（0B）"],
  ];
  for (const [temp, message] of broken) {
    assertThrows(() => validateStepRecipe(tempStep([temp], two)), ExecutionError, message);
  }
  // 対照: 列の内側で閉じた宣言は通る（上の 5 本は寿命宣言だけが違う）。
  validateStepRecipe(tempStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 1 }], two));
});

Deno.test("宣言外の一時を束ねる dispatch は宣言の受け口で落ちる", () => {
  const temps: readonly TempRecipe[] = [{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }];
  const bind = (id: number): readonly Dispatch[] => [
    dispatch("gemm", [{ binding: 1, source: { kind: "temp", id } }]),
  ];
  assertThrows(
    () => validateStepRecipe(tempStep(temps, bind(1))),
    ExecutionError,
    "dispatch 'gemm' の束縛 1 が一時 1 を指すが、宣言は 1 本",
  );
  // 対照: 宣言済みの添字なら通る（束縛の並びは同じで添字だけが違う）。
  validateStepRecipe(tempStep(temps, bind(0)));
});

Deno.test("消費計数が実際の解放より多いレシピ列は計画の閉包検査で落ちる", () => {
  assertThrows(
    // 消費者 1 本を宣言しながら、その値を解放するステップがどこにも無い形。
    () => planRecipes([step([alloc("a", 64, 1)])], CORE_TRANSIENT_LIMITS),
    ExecutionError,
    "解放されない中間が残る",
  );
  // 対照: 消費者ぶんの解放が来れば閉じる（同じ列に消費ステップを 1 本足すだけ）。
  assertEquals(
    planRecipes([step([alloc("a", 64, 1)]), step([], ["a"])], CORE_TRANSIENT_LIMITS).regions,
    [64],
  );
});

Deno.test("同じ値を二重に解放するレシピ列は参照カウントが負で落ちる", () => {
  assertThrows(
    () => planRecipes([step([alloc("a", 64, 0)]), step([], ["a", "a"])], CORE_TRANSIENT_LIMITS),
    ExecutionError,
    "参照カウントが負",
  );
});

// ---------------------------------------------------------------------------
// 一時（temps）の区間と配り直し
// ---------------------------------------------------------------------------

/**
 * 一時を持つステップ（`dispatches` は境界の本数だけが効くのでスタブで足りる）。
 *
 * `steps[].temps` は `bakeGroups` が bind group の実体解決にそのまま使うので、temp → slot の
 * 対応が 1 つずれると**同一サイズの別区間を束ねる**（layout は満たすので validation を通り、
 * 値だけが静かに変わる）。上の topk のコメントが出力側について書いている論法を、そのまま
 * 一時側へ当てる。
 */
const tempSlotsStep = (
  temps: readonly TempRecipe[],
  dispatchCount: number,
  outputs: readonly StepOutput[] = [],
  releases: readonly string[] = [],
): StepRecipe => ({
  outputs,
  temps,
  dispatches: Array.from({ length: dispatchCount }, (_, index) => dispatch(`k${index}`)),
  releases,
  writesState: false,
});

Deno.test("同時生存する入れ子寿命の一時は 3 本とも別区間を掴む", () => {
  // i8a8 attention と同型の「ループ外一時（0 と 2）がループ内一時（1）を挟む」形。
  const plan = planRecipes([
    tempSlotsStep([
      { byteLength: 64, allocBefore: 0, releaseAfter: 2 },
      { byteLength: 64, allocBefore: 1, releaseAfter: 1 },
      { byteLength: 64, allocBefore: 0, releaseAfter: 2 },
    ], 3),
  ], CORE_TRANSIENT_LIMITS);

  // 3 本とも同時生存するので配り直しが起きない（生存ピーク = 3 本ぶん・共有 0）。
  assertEquals(plan.peakLiveBytes, 192);
  assertEquals(plan.sharedSlots, 0);
  // 確保順は「境界ごとに宣言順」— 境界 0 で id 0 / id 2 が区間 0 / 1 を取り、境界 1 で
  // 生える id 1 が区間 2 になる（宣言順 = 区間順ではない）。
  assertEquals(plan.steps[0].temps, [0, 2, 1]);
  assertEquals(new Set(plan.slots.map((slot) => slot.offset)).size, 3, "offset が全て異なる");
});

Deno.test("同一境界で解放された一時のバイトは、次のステップの一時が配り直される", () => {
  const plan = planRecipes([
    tempSlotsStep([
      { byteLength: 64, allocBefore: 0, releaseAfter: 0 },
      { byteLength: 64, allocBefore: 0, releaseAfter: 0 },
    ], 2),
    tempSlotsStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }], 1),
  ], CORE_TRANSIENT_LIMITS);

  // 2 本目のステップは新しいバイトを生やさない（領域は 2 本ぶんのまま）。
  assertEquals(plan.regions, [64 + 256]);
  assertEquals(plan.steps[0].temps, [0, 1]);
  assertEquals(plan.steps[1].temps, [2]);
  assertEquals(plan.slots[2].offset, 0, "first-fit は空いた先頭の区間を配る");
  assertEquals(plan.sharedSlots, 1);
});

Deno.test("出力と一時は同じ領域を共有し、pin された区間だけが配り直されない", () => {
  // 消費者ゼロの出力はステップ末尾で生存を終えるので、次のステップの一時がそのバイトを掴む。
  const shared = planRecipes([
    tempSlotsStep([], 0, [alloc("a", 64, 0)]),
    tempSlotsStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }], 1),
  ], CORE_TRANSIENT_LIMITS);
  assertEquals(shared.regions, [64]);
  assertEquals(shared.steps[1].temps, [1]);
  assertEquals(shared.slots[1].offset, 0);

  // pin（グラフ出力）は refs が 0 になっても生存を終えない。
  const pinnedPlan = planRecipes([
    tempSlotsStep([], 0, [alloc("y", 64, 0, true)]),
    tempSlotsStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }], 1),
  ], CORE_TRANSIENT_LIMITS);
  assertEquals(pinnedPlan.regions, [64 + 256]);
  assertEquals(pinnedPlan.slots[1].offset, 256);
  assertEquals(pinnedPlan.pinned, new Set([0]));
});

// ---------------------------------------------------------------------------
// generation 面を欠いた実行の fail loudly（GPU 資源を 1 つも作らない）
// ---------------------------------------------------------------------------

/** 実行相まで届くスタブ dispatch（`workgroups` は焼き込み実行が読む欄）。 */
const runnableDispatch = (
  key: string,
  workgroups: DispatchWorkgroups,
  bindings: readonly BindingRecipe[] = [],
): Dispatch => ({ key, bindings, workgroups, roles: roles([1], []) }) as unknown as Dispatch;

/** dispatch だけのステップ（出力も一時も持たないので簿記は動かない）。 */
const dispatchStep = (dispatches: readonly Dispatch[]): StepRecipe => ({
  outputs: [],
  temps: [],
  dispatches,
  releases: [],
  writesState: false,
});

/** 積まれた dispatch を記録するだけのスケジューラ（GPU 資源を持たない）。 */
const recordingScheduler = (
  log: string[],
): SubmitScheduler =>
  ({
    dispatch: (
      _pipeline: unknown,
      _group: unknown,
      workgroups: readonly [number, number, number],
      key: string,
    ) => {
      log.push(`${key} [${workgroups.join(",")}]`);
    },
  }) as unknown as SubmitScheduler;

const stubGroup = (): GPUBindGroup => ({}) as unknown as GPUBindGroup;

Deno.test("焼き込み経路は context 側の焼き込みを経ていない dispatch で落ちる", () => {
  const recipes = [dispatchStep([runnableDispatch("attn.qk", [1, 1, 1])])];
  const log: string[] = [];

  const error = assertThrows(
    () => executeBakedPlan(recipes, [[undefined]], recordingScheduler(log)),
    ExecutionError,
  );
  assert(error.message.includes("dispatch 'attn.qk'"), error.message);
  assert(error.message.includes("ADR 0066 決定 5"), error.message);
  assertEquals(log, []);

  // 対照: 穴が埋まっていれば throw せず 1 本積まれる（上が空振りでない証明）。
  executeBakedPlan(recipes, [[stubGroup()]], recordingScheduler(log));
  assertEquals(log, ["attn.qk [1,1,1]"]);
});

Deno.test("workgroup 数を論理長から算出する dispatch は GenerationContext 無しで落ちる", () => {
  const log: string[] = [];
  const derived = [dispatchStep([runnableDispatch("state.pv", (past, query) => [past, query, 1])])];

  const error = assertThrows(
    () => executeBakedPlan(derived, [[stubGroup()]], recordingScheduler(log)),
    ExecutionError,
    "workgroup 数を論理長から算出する dispatch を GenerationContext 無しで",
  );
  assert(error.message.includes("dispatch 'state.pv'"), error.message);

  // 対照: 静的な 3 つ組は generation 無しでも積まれる。**0 でも積む**のが既存契約
  // （0 要素テンソルの経路は要素数 0 の dispatch を黙って飛ばさない）。
  executeBakedPlan(
    [dispatchStep([runnableDispatch("elementwise", [0, 1, 1])])],
    [[stubGroup()]],
    recordingScheduler(log),
  );
  assertEquals(log, ["elementwise [0,1,1]"]);
});

Deno.test("論理長から算出した workgroup 数が 0 の軸を持つ dispatch だけは積まれない", () => {
  const log: string[] = [];
  const recipes = [dispatchStep([runnableDispatch("state.pv", (past, query) => [past, query, 1])])];
  const generation = (past: number, query: number): BakedGeneration => ({
    groups: [[undefined]],
    encoding: {
      slots: new Map(),
      lengths: ({}) as unknown as GPUBuffer,
      past,
      query,
    },
    onStep: () => {},
  });

  // 有効行ゼロ（この行ブロックが丸ごと pad 行）は積まない。
  executeBakedPlan(recipes, [[stubGroup()]], recordingScheduler(log), generation(0, 4));
  assertEquals(log, []);
  // 対照: 有効行があれば同じレシピが積まれる。
  executeBakedPlan(recipes, [[stubGroup()]], recordingScheduler(log), generation(2, 4));
  assertEquals(log, ["state.pv [2,4,1]"]);
});

/** bind group の実体解決だけを走らせる（GPU 資源を持たない device スタブ）。 */
const bakeOne = (
  source: BindingSource,
  generation?: { readonly slots: ReadonlyMap<string, GPUBuffer> },
): void => {
  const recipes = [
    dispatchStep([runnableDispatch("attn.pv", [1, 1, 1], [{ binding: 1, source }])]),
  ];
  bakeAllBindGroups(
    recipes,
    {
      device: ({ createBindGroup: stubGroup }) as unknown as GPUDevice,
      plan: planRecipes(recipes, CORE_TRANSIENT_LIMITS),
      regions: [],
      inputs: new Map(),
    },
    generation === undefined ? undefined : {
      slots: generation.slots,
      lengths: ({}) as unknown as GPUBuffer,
      past: 0,
      query: 1,
    },
  );
};

Deno.test("generation 面を欠いた実行は state / 論理長の束縛で fail loudly", () => {
  const lengths = assertThrows(() => bakeOne({ kind: "lengths" }), ExecutionError);
  assert(
    lengths.message.includes("論理長 uniformを束ねる dispatch を GenerationContext 無しで"),
    lengths.message,
  );

  const state = assertThrows(() => bakeOne({ kind: "state", name: "kv.k" }), ExecutionError);
  assert(
    state.message.includes("state スロット 'kv.k'を束ねる dispatch を GenerationContext 無しで"),
    state.message,
  );
});

Deno.test("GenerationContext が持たないスロット名は実体解決で fail loudly", () => {
  const error = assertThrows(
    () =>
      bakeOne({ kind: "state", name: "kv.v" }, {
        slots: new Map([["kv.k", ({}) as unknown as GPUBuffer]]),
      }),
    ExecutionError,
  );
  assert(
    error.message.includes("state スロット 'kv.v' の実体が GenerationContext に無い"),
    error.message,
  );
});
