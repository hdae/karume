/**
 * レシピの **slot 導出**（src/runtime/recipe.ts の `derivePlanSlots`）と**宣言の静的検査**
 * （同 `validateStepRecipe`）の構造テスト。GPU を 1 つも触らないので、アダプタ無し環境でも走る。
 *
 * 見ているのは「{@link StepRecipe.outputs} の列から、確保・retain・解放の簿記が
 * {@link RunArena} のサイズクラス LIFO どおりに再生されるか」— 実 GPU の footprint 門
 * （tests/gpu_plan_backing_test.ts が「slot 表の総バイト数 = 非 backed run のプール確保」を
 * 突き合わせる）とは別の角度で、**期待値を手計算で固定する**。
 *
 * ADR 0068 決定 1 の受入条件「単一出力ノードのレシピ表現は列化前と同値」がここの 1 本目。
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { RunArena } from "../src/gpu/arena.ts";
import type { SubmitScheduler } from "../src/gpu/submit.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  type BakedGeneration,
  type BindingRecipe,
  type BindingSource,
  derivePlanSlots,
  type DispatchWorkgroups,
  executeBakedPlan,
  executeStepRecipe,
  type StepOutput,
  type StepRecipe,
  type TempRecipe,
  validateStepRecipe,
} from "../src/runtime/recipe.ts";

/** 出力列だけのステップ（dispatch と一時は slot 導出の別軸なので空で足りる）。 */
const step = (
  outputs: readonly StepOutput[],
  releases: readonly string[] = [],
): StepRecipe => ({ outputs, temps: [], dispatches: [], releases, writesState: false });

/** プールから確保する出力（`byteLength` は 4 の倍数 = サイズクラスと同値にしておく）。 */
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

Deno.test("単一出力ステップの slot 割当は alloc / alias / pin / LIFO 再利用で決まる", () => {
  const slots = derivePlanSlots([
    // 消費者 1 本ある中間値 → ステップ末尾では refs が残り、プールへ返らない。
    step([alloc("a", 64, 1)]),
    // 消費者ゼロの中間値（到達不能な値）。定義ぶんの解放でそのままプールへ戻る。
    step([alloc("b", 64, 0)], []),
    // 直前の 2 本がここで返っているので、LIFO の頂点（"b" の slot 1）を掴む。
    step([alloc("c", 64, 0, true)], ["a"]),
    // 別名は slot を増やさない（実体は元の値のもの）。
    step([alias("d", "c", 0)]),
    // 別名元が slot を持たない値（グラフ入力）なら、この出力も slot を持たない。
    step([alias("e", "x", 0)]),
  ]);

  // 新規確保は 2 本だけ（3 本目は "b" の slot を再利用し、別名 2 本は確保を出さない）。
  assertEquals(slots.bytes, [64, 64]);
  assertEquals(slots.steps.map((assigned) => assigned.outputs), [
    [0],
    [1],
    [1],
    [undefined],
    [undefined],
  ]);
  // pin した slot 1 はグラフ出力 "c" の実体。以後どの値にも配り直されない。
  assertEquals(slots.pinned, new Set([1]));
});

/**
 * `topk` の実形（ADR 0068 決定 3 の最初の入居者）。実 GPU 側の寿命門
 * （tests/runtime_executor_test.ts の「多出力ノードの slot ごとの uses / pin」）と**同じグラフ**
 * の slot 割当を、GPU に触らない純関数の側から手計算で固定する:
 *
 * ```
 * topk(x[3,8], k=2) → v[3,2] f32（neg が 1 度消費）+ i[3,2] i32（グラフ出力）
 * neg(v) → nv[3,2]（exp が 1 度消費）
 * exp(nv) → e[3,2]（グラフ出力）
 * ```
 *
 * MUST: 2 本の出力は**同じバイト数**（どちらも 3·2 語の 4 バイト要素 = 24）— サイズクラスが
 * 同じなので、slot を取り違えても総バイト数では気づけない。だから **どの slot が再利用されるか**
 * まで期待値に置く（`e` が掴むのは消費済みの `v` の slot 0 で、pin された添字の slot 1 では
 * ない）。
 */
Deno.test("topk の実形（値 + 添字・同一サイズクラス）の slot 割当が手計算どおり", () => {
  // rows·k·4 = 3·2·4（実 GPU 側の門と同じ形から導く）
  const bytes = 3 * 2 * 4;
  const slots = derivePlanSlots([
    // topk: 値（消費者 1 本）+ 添字（グラフ出力 = pin）を出力 slot 昇順で
    step([alloc("v", bytes, 1), alloc("i", bytes, 0, true)]),
    // neg: `v` を消費し、`nv`（消費者 1 本）を定義する。確保の時点でプールは空
    // （`v` は refs 1・`i` は pin）なので新しい slot が生える
    step([alloc("nv", bytes, 1)], ["v"]),
    // exp: `nv` を消費し、`e`（グラフ出力）を定義する。ここで掴むのは**返ってきた `v` の
    // slot 0**（LIFO の頂点）— pin された添字の slot 1 は候補にならない
    step([alloc("e", bytes, 0, true)], ["nv"]),
  ]);

  assertEquals(slots.bytes, [bytes, bytes, bytes]);
  assertEquals(slots.steps.map((assigned) => assigned.outputs), [[0, 1], [2], [0]]);
  // pin は添字（slot 1）と `e`（slot 0）。実 GPU 側の生存ピーク 72 バイト = 3 slot ぶんと一致する。
  assertEquals(slots.pinned, new Set([1, 0]));
});

/**
 * 多出力ステップ（ADR 0068 決定 1）の合成形。上の `topk` が実装済みの実形を押さえる一方、
 * こちらは**サイズクラスの違う 2 本**でレシピ表現の側だけを純関数で固定する。**出力 slot 昇順**
 * の確保と、slot ごとに独立した `uses` / `pinned` の簿記が要点。
 */
Deno.test("多出力ステップは出力 slot 昇順に確保し、slot ごとに独立して retain / pin する", () => {
  const slots = derivePlanSlots([
    // 値 [64B・消費者 1 本] と添字 [32B・グラフ出力] の 2 本を 1 ステップで定義する形。
    step([alloc("v", 64, 1), alloc("i", 32, 0, true)]),
    // "v" を消費するステップ。確保の時点でプールは空（"v" は refs 1・"i" は pin）なので
    // 新しい slot が生える。
    step([alloc("y", 64, 0, true)], ["v"]),
  ]);

  assertEquals(slots.bytes, [64, 32, 64]);
  assertEquals(slots.steps.map((assigned) => assigned.outputs), [[0, 1], [2]]);
  assertEquals(slots.pinned, new Set([1, 2]));
});

// ---------------------------------------------------------------------------
// 宣言の静的検査（validateStepRecipe）と slot 導出の閉包検査
// ---------------------------------------------------------------------------

/** `DispatchRecipe` は非公開型なので、レシピ列の要素型として引く。 */
type Dispatch = StepRecipe["dispatches"][number];

/**
 * dispatch のスタブ。宣言の静的検査が読むのは `key` と `bindings` だけで、パイプラインと
 * bind group layout は GPU 資源なので持たせない（型を満たすためだけの実体化を避ける）。
 */
const dispatch = (key: string, bindings: readonly BindingRecipe[] = []): Dispatch =>
  ({ key, bindings }) as unknown as Dispatch;

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

Deno.test("消費計数が実際の解放より多いレシピ列は slot 導出の閉包検査で落ちる", () => {
  assertThrows(
    // 消費者 1 本を宣言しながら、その値を解放するステップがどこにも無い形
    // （run 経路のアリーナなら assertDrained が見る破れ）。
    () => derivePlanSlots([step([alloc("a", 64, 1)])]),
    ExecutionError,
    "未解放の参照が残存（slot 0, refs=1, 64B",
  );
  // 対照: 消費者ぶんの解放が来れば閉じる（同じ列に消費ステップを 1 本足すだけ）。
  assertEquals(derivePlanSlots([step([alloc("a", 64, 1)]), step([], ["a"])]).bytes, [64]);
});

Deno.test("同じ値を二重に解放するレシピ列は参照カウントが負で落ちる", () => {
  assertThrows(
    () => derivePlanSlots([step([alloc("a", 64, 0)]), step([], ["a", "a"])]),
    ExecutionError,
    "参照カウントが負",
  );
});

// ---------------------------------------------------------------------------
// 一時（temps）の slot 割当と LIFO 解放
// ---------------------------------------------------------------------------

/**
 * 一時を持つステップ（`dispatches` は境界の本数だけが効くのでスタブで足りる）。
 *
 * `StepSlots.temps` は `bakeGroups` が bind group の実体解決にそのまま使うので、
 * temp → slot の対応が 1 つずれると**同一サイズクラスの別バッファを束ねる**（layout は
 * 満たすので validation を通り、値だけが静かに変わる）。上の topk のコメントが出力側について
 * 書いている論法を、そのまま一時側へ当てる。
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

Deno.test("同時生存する入れ子寿命の一時は 3 本とも別 slot を掴む", () => {
  // i8a8 attention と同型の「ループ外一時（0 と 2）がループ内一時（1）を挟む」形。
  const slots = derivePlanSlots([
    tempSlotsStep([
      { byteLength: 64, allocBefore: 0, releaseAfter: 2 },
      { byteLength: 64, allocBefore: 1, releaseAfter: 1 },
      { byteLength: 64, allocBefore: 0, releaseAfter: 2 },
    ], 3),
  ]);

  // 3 本とも同時生存するので再利用が起きない。
  assertEquals(slots.bytes, [64, 64, 64]);
  // 確保順は「境界ごとに宣言順」— 境界 0 で id 0 / id 2 が slot 0 / 1 を取り、境界 1 で
  // 生える id 1 が slot 2 になる（宣言順 = slot 順ではない）。
  assertEquals(slots.steps[0].temps, [0, 2, 1]);
});

Deno.test("同一境界の一時は確保の逆順で返り、次のステップは最後に返った slot を掴む", () => {
  const slots = derivePlanSlots([
    // 境界 0 で 2 本確保し、同じ境界 0 で両方解放する（逆順 = id 1 → id 0 の順にプールへ）。
    tempSlotsStep([
      { byteLength: 64, allocBefore: 0, releaseAfter: 0 },
      { byteLength: 64, allocBefore: 0, releaseAfter: 0 },
    ], 2),
    tempSlotsStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }], 1),
  ]);

  // 2 本目のステップは新しい slot を生やさない。
  assertEquals(slots.bytes, [64, 64]);
  assertEquals(slots.steps[0].temps, [0, 1]);
  // プールへ戻った順は [1, 0] なので LIFO の頂点は**最後に返った slot 0**。ここが 1 になったら
  // executeStepRecipe（同じ逆順ループ）と食い違っている＝ 2 実装が別の bind group を組む。
  assertEquals(slots.steps[1].temps, [0]);
});

Deno.test("出力と一時は同じプールを共有し、pin された slot だけが配り直されない", () => {
  // 消費者ゼロの出力はステップ末尾でプールへ戻るので、次のステップの一時が掴む。
  const shared = derivePlanSlots([
    tempSlotsStep([], 0, [alloc("a", 64, 0)]),
    tempSlotsStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }], 1),
  ]);
  assertEquals(shared.bytes, [64]);
  assertEquals(shared.steps[1].temps, [0]);

  // pin（グラフ出力）は refs が 0 になってもプールへ返らない。
  const pinnedSlots = derivePlanSlots([
    tempSlotsStep([], 0, [alloc("y", 64, 0, true)]),
    tempSlotsStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }], 1),
  ]);
  assertEquals(pinnedSlots.bytes, [64, 64]);
  assertEquals(pinnedSlots.steps[1].temps, [1]);
  assertEquals(pinnedSlots.pinned, new Set([0]));
});

// ---------------------------------------------------------------------------
// generation 面を欠いた実行の fail loudly（GPU 資源を 1 つも作らない）
// ---------------------------------------------------------------------------

/** 実行相まで届くスタブ dispatch（`workgroups` は焼き込み実行が読む欄）。 */
const runnableDispatch = (
  key: string,
  workgroups: DispatchWorkgroups,
  bindings: readonly BindingRecipe[] = [],
): Dispatch => ({ key, bindings, workgroups }) as unknown as Dispatch;

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

Deno.test("generation 面を欠いた実行は state / 論理長の束縛で fail loudly", () => {
  const execute = (source: BindingSource): void =>
    executeStepRecipe(
      dispatchStep([runnableDispatch("attn.pv", [1, 1, 1], [{ binding: 1, source }])]),
      {
        arena: ({}) as unknown as RunArena,
        device: ({}) as unknown as GPUDevice,
        scheduler: recordingScheduler([]),
        env: new Map(),
      },
    );

  const lengths = assertThrows(() => execute({ kind: "lengths" }), ExecutionError);
  assert(
    lengths.message.includes("論理長 uniformを束ねる dispatch を GenerationContext 無しで"),
    lengths.message,
  );

  const state = assertThrows(() => execute({ kind: "state", name: "kv.k" }), ExecutionError);
  assert(
    state.message.includes("state スロット 'kv.k'を束ねる dispatch を GenerationContext 無しで"),
    state.message,
  );
});

Deno.test("GenerationContext が持たないスロット名は実体解決で fail loudly", () => {
  const error = assertThrows(
    () =>
      executeStepRecipe(
        dispatchStep([
          runnableDispatch("attn.pv", [1, 1, 1], [{
            binding: 1,
            source: { kind: "state", name: "kv.v" },
          }]),
        ]),
        {
          arena: ({}) as unknown as RunArena,
          device: ({}) as unknown as GPUDevice,
          scheduler: recordingScheduler([]),
          env: new Map(),
          generation: {
            slots: new Map([["kv.k", ({}) as unknown as GPUBuffer]]),
            lengths: ({}) as unknown as GPUBuffer,
            past: 0,
            query: 1,
          },
        },
      ),
    ExecutionError,
  );
  assert(
    error.message.includes("state スロット 'kv.v' の実体が GenerationContext に無い"),
    error.message,
  );
});
