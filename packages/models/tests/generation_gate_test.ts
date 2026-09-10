// 投機の自己採算ゲート（ADR 0096 段 4-B ④）の挙動テスト。純関数なので GPU も時計も要らない
// （壁は呼び手が渡す = このテストが渡す定数そのもの）。
//
// ここで縛るのは 3 点である:
//
// 1. 「負ける文脈では decode 形へ落ち、勝てる文脈では投機のまま」— その判断が**実測の数字**
//    （`docs/research/2026-09-09-mtp-stage4.md` §2 表 1 / §5.1）で正しい側へ倒れること（T8）。
//    閾値は host で 1.7〜2.6 と動くので固定値に焼けず、ゲートが自分で測るしかない。
// 2. **1 サンプルでは判定しない**こと（T2 / T3）。受理数は cycle あたり sd 0.88（平均 2.03 の
//    43%）で散るので、移動平均の種付け（`ewma(undefined, sample) = sample`）で判定すると
//    最初の 1 cycle が受理 0 だった会話が即座に plain へ落ちる。実走で観測した欠陥がこれで、
//    `200 token / 7 cycle / plain step 190 / 切替 1`・always の 7,066 ms に対し 9,262 ms だった。
//    推定器は `window` 本の非重複ブロックの和で、`confirm` 本連続の負けだけが抜ける根拠になる。
// 3. **「強い負け」のノブが speculate 側と plain 側で別**であること（T2′ / T4″）。speculate 側の
//    早抜け（`earlyLeave`）は**未指定 = off** で、既定では比 1.30 でも `confirm` 本連続を待つ
//    （実走では勝つ課題でも 1 ブロックが 1.16 を超える区間が混ざり、warm な会話で 7 ターン中
//    3 ターンが誤退出した）。plain 側の打ち切り（`burstAbort`・既定 0.15）は残す — 誤発火は
//    「誤って抜けた後のバースト」でしか起こりえず、代償は探索 1 回ぶんだからである。

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import {
  createSpeculationGate,
  type SpeculationDecision,
  type SpeculationGate,
} from "../src/generation/speculation-gate.ts";

/** 1 cycle ぶんの壁（ゲートに渡す観測 — 投機 cycle と plain step の対）。 */
type Walls = {
  /** 投機 cycle の壁 ms。 */
  readonly cycle: number;
  /** その cycle が確定させた token 数。 */
  readonly delivered: number;
  /** plain step（M=1）の壁 ms。 */
  readonly plain: number;
};

/**
 * ゲートを 1 cycle 回す（決定に応じた観測を返し、その決定を返す）。
 *
 * 呼び手の契約（decide 1 回 → 観測 1 回）をそのまま写した口である。
 */
const step = (gate: SpeculationGate, walls: Walls): SpeculationDecision => {
  const decision = gate.decide();
  if (decision === "speculate") gate.observeCycle(walls.cycle, walls.delivered);
  else gate.observePlain(walls.plain);
  return decision;
};

/** 1 step ぶんの記録（決定と、その step の**前後**の定常モード）。 */
type Observed = {
  readonly decision: SpeculationDecision;
  /** 決定を出した時点のモード（この step が探索側かどうかはこちらで読む）。 */
  readonly before: SpeculationDecision;
  /** 観測を返した後のモード（切替は観測で起きる）。 */
  readonly mode: SpeculationDecision;
};

const trace = (gate: SpeculationGate, walls: Walls, count: number): Observed[] =>
  Array.from({ length: count }, () => {
    const before = gate.mode;
    const decision = step(gate, walls);
    return { decision, before, mode: gate.mode };
  });

const run = (gate: SpeculationGate, walls: Walls, count: number): SpeculationDecision[] =>
  trace(gate, walls, count).map((observed) => observed.decision);

/** 決定列の中で `plain` が出た添字。 */
const plainAt = (decisions: readonly SpeculationDecision[]): number[] =>
  decisions.flatMap((decision, index) => decision === "plain" ? [index] : []);

/** 決定列のうち speculate の本数（= ブロック / バーストに積まれた cycle の数）。 */
const speculateCount = (decisions: readonly SpeculationDecision[]): number =>
  decisions.filter((decision) => decision === "speculate").length;

/**
 * 連続した同種の step の長さの列（`kind` が `"plain"` なら探索間隔・`"speculate"` なら
 * 探索バーストの長さ）。
 *
 * **決定を出した時点のモードが plain の step だけ**を見る。speculate モードで挟まる 1 手の
 * `W1` の測り直しを混ぜないためと、倒れた瞬間の step（決定は speculate・観測の後は plain）を
 * バーストに数えないためである。
 */
const plainModeRuns = (
  steps: readonly Observed[],
  kind: SpeculationDecision,
): number[] => {
  const lengths: number[] = [];
  let current = 0;
  for (const observed of steps) {
    if (observed.before === "plain" && observed.decision === kind) current += 1;
    else if (current > 0) {
      lengths.push(current);
      current = 0;
    }
  }
  return lengths;
};

/** 投機が明らかに勝つ壁（比 (100/4)/50 = 0.5）。 */
const WINNING: Walls = { cycle: 100, delivered: 4, plain: 50 };
/** 投機が明らかに負ける壁（比 (200/2)/50 = 2.0 — plain の壁は WINNING と同じ 50）。 */
const LOSING: Walls = { cycle: 200, delivered: 2, plain: 50 };
/**
 * 僅かに負ける壁（比 (110/2)/50 = 1.10）— `leave`（1.01）は超えるが「強い負け」の閾値
 * `leave + 0.15`（1.16 — `burstAbort` の既定・`earlyLeave` を渡す席でも同じ値を使う）には届かない。
 */
const MILD: Walls = { cycle: 110, delivered: 2, plain: 50 };
/** 強く負ける壁（比 (130/2)/50 = 1.30 — 上の 1.16 を超える）。 */
const STRONG_LOSS: Walls = { cycle: 130, delivered: 2, plain: 50 };

Deno.test("ゲート T1 起点: 初期は speculate で、W1 は 2 回目の決定で 1 回だけ採る", () => {
  // `W1`（decode の 1 token あたりの壁）を測るまでは「抜けるべきか」を判断できない。未観測の
  // まま 16 cycle 回すと、M2 自由文（0.76×）では 2% を先に失う。
  const gate = createSpeculationGate();
  assertEquals(gate.mode, "speculate", "初期モード");
  assertEquals(gate.switches, 0);
  assertEquals(run(gate, WINNING, 6), [
    "speculate",
    "plain",
    "speculate",
    "speculate",
    "speculate",
    "speculate",
  ]);
  // 1 step の plain は `W1` の測り直しであってモードの切替ではない。
  assertEquals(gate.mode, "speculate");
  assertEquals(gate.switches, 0, "`W1` の測り直しを切替に数えている");
});

Deno.test("ゲート T2 ブロック: 満ちるまで判定せず、負けブロック 2 本連続で初めて抜ける", () => {
  const gate = createSpeculationGate();
  // 既定の window は 16。`W1` の測り直しは決定 1 / 15 に入るので、決定 17 本で投機 cycle は
  // ちょうど 15 本 = ブロックはまだ満ちない。負けの壁でもここでは倒れない。
  const first = run(gate, MILD, 17);
  assertEquals(speculateCount(first), 15, "ブロックが満ちない前提が崩れている");
  assertEquals(gate.mode, "speculate", "ブロックが満ちる前に抜けている");
  assertEquals(gate.switches, 0);

  // 16 本目でブロック 1 が満ちる（比 1.10 > leave 1.01）が、`confirm` は 2 なのでまだ抜けない
  // （既定では早抜けが off なので、比の大きさで待つ長さは変わらない — T2′）。
  run(gate, MILD, 1);
  assertEquals(gate.mode, "speculate", "負けブロック 1 本だけで抜けている");
  assertEquals(gate.switches, 0);

  // 続く 17 本（決定 18..34・測り直しは決定 31 の 1 本）で投機 cycle が 16 本 = ブロック 2 が
  // 満ちる。最後の 1 本の観測まで speculate のままで、その観測で倒れる。
  const second = trace(gate, MILD, 17);
  assertEquals(speculateCount(second.map((observed) => observed.decision)), 16);
  assertEquals(
    second.slice(0, -1).map((observed) => observed.mode),
    Array.from({ length: 16 }, () => "speculate"),
    "ブロック 2 が満ちる前に倒れている",
  );
  assertEquals(gate.mode, "plain");
  assertEquals(gate.switches, 1);
});

Deno.test("ゲート T2′ 早抜け: 既定は off（比 1.30 でも 2 ブロック）・earlyLeave の席だけ 1 本で倒れる", () => {
  // 既定 off の根拠は机上の確率ではなく実走である。受理はターン内で定常でなく、勝つ課題
  // （対話・比 0.90）でも 16 cycle 級で受理が落ちる区間が混ざるので、sequence を使い回した会話で
  // 7 ターン中 3 ターンが誤って抜けた（research §6.5 表 11″）。利得は「抜けるまでの 1 ブロック」で
  // sequence の寿命に 1 回、害は勝つ課題の毎ターンなので、既定では取らない。
  const patient = createSpeculationGate();
  const steps = trace(patient, STRONG_LOSS, 18);
  assertEquals(speculateCount(steps.map((observed) => observed.decision)), 16);
  assertEquals(patient.mode, "speculate", "既定で 1 ブロックで倒れている（早抜けは off）");
  assertEquals(patient.switches, 0);
  // 2 ブロック目（決定 17 本で投機 cycle 16 本 — T2 と同じ数え）で初めて倒れる。
  run(patient, STRONG_LOSS, 17);
  assertEquals(patient.mode, "plain", "2 ブロック連続の負けで倒れていない");
  assertEquals(patient.switches, 1);

  // ノブを渡した席は 1 ブロックで倒れる（= 早抜けは `earlyLeave` の閾値だけで効いている）。
  const eager = createSpeculationGate({ earlyLeave: 0.15 });
  const eagerSteps = trace(eager, STRONG_LOSS, 18);
  assertEquals(speculateCount(eagerSteps.map((observed) => observed.decision)), 16);
  assertEquals(
    eagerSteps.slice(0, -1).map((observed) => observed.mode),
    Array.from({ length: 17 }, () => "speculate"),
    "ブロックが満ちる前に倒れている（早抜けも満ちた 1 本の判定である）",
  );
  assertEquals(eager.mode, "plain", "earlyLeave 0.15 の席で 1 ブロックで倒れていない");
  assertEquals(eager.switches, 1);

  // 対（同じ席でも比 1.10 は `leave + earlyLeave` = 1.16 に届かないので `confirm` を待つ）。
  const mild = createSpeculationGate({ earlyLeave: 0.15 });
  run(mild, MILD, 18);
  assertEquals(mild.mode, "speculate", "1.16 に届かない負けで早抜けしている");
  assertEquals(mild.switches, 0);

  // 対 2（配線の取り違え）: speculate 側の閾値は `earlyLeave` であって `burstAbort` ではない。
  // 比 1.30 は `leave + burstAbort` = 1.16 は超えるが `leave + earlyLeave` = 1.31 には届かない。
  const wired = createSpeculationGate({ earlyLeave: 0.3, burstAbort: 0.15 });
  run(wired, STRONG_LOSS, 18);
  assertEquals(wired.mode, "speculate", "speculate 側が burstAbort の閾値を読んでいる");
  assertEquals(wired.switches, 0);
});

/** 最初の cycle だけ配送 1（受理 0）— 旧 EWMA 設計が 1 サンプルで種付けされた形。 */
const SEED_FIRST: Walls = { cycle: 100, delivered: 1, plain: 40 };
/** 以後の定常（比 (100/3)/40 = 0.83 = 投機が勝っている）。 */
const SEED_REST: Walls = { cycle: 100, delivered: 3, plain: 40 };

Deno.test("ゲート T3 種付け: 最初の cycle が受理 0 でも、勝っている文脈では抜けない", () => {
  // 移動平均で `A` を持つ設計では `ewma(undefined, 1) = 1` になり、次の決定で比が
  // `(100/1)/40 = 2.5 > leave` → **3 手目で plain へ落ちる**（実走で観測した欠陥そのもの）。
  // ブロック集計では 16 本の和で見るので `1600 / (1 + 15×3) = 34.8 ms/token` < `40 × 1.01`
  // となり、判定は正しく「勝っている」側に出る。
  const gate = createSpeculationGate();
  step(gate, SEED_FIRST);
  run(gate, SEED_REST, 59);
  assertEquals(gate.mode, "speculate", "1 サンプルの種付けで抜けている");
  assertEquals(gate.switches, 0);

  // 対（フォールト注入）: 同じ種付けでも、定常が本当に負けていれば抜ける（比 (100/2)/40 = 1.25 が
  // `confirm` = 2 ブロック続く）。これが無いと上の緑は「そもそも抜けないゲート」でも通る。
  const losing = createSpeculationGate();
  step(losing, SEED_FIRST);
  run(losing, { cycle: 100, delivered: 2, plain: 40 }, 59);
  assertEquals(losing.mode, "plain", "本当に負けている壁でも抜けない");
  assertEquals(losing.switches, 1);
});

Deno.test("ゲート T4 探索: plain 側はバーストで測り、外れるたび間隔が倍になる", () => {
  // 1 cycle 単発の探索では受理数の sd 0.88 に埋もれて判定できない（旧設計が plain から戻れ
  // なかった理由）。バーストで測り、外れたら遠ざける（固定間隔のままだと探索そのものの
  // 損失が M2 自由文で 0.950× になり、ゲートが救った負けを探索で払い直す）。
  const gate = createSpeculationGate({ window: 2, confirm: 1 });
  const steps = trace(gate, LOSING, 2000);
  assertEquals(
    plainModeRuns(steps, "plain").slice(0, 7),
    [16, 32, 64, 128, 256, 512, 512],
    `探索の間隔が幾何バックオフになっていない: ${plainModeRuns(steps, "plain").join(",")}`,
  );
  // 比 2.0 は `leave + burstAbort` = 1.16 を超えるので、外れが確定したバーストは `burstMin` = 4 本で
  // 畳まれる（T4″）。
  assertEquals(
    plainModeRuns(steps, "speculate").slice(0, 6),
    [4, 4, 4, 4, 4, 4],
    `探索バーストが 4 cycle で畳まれていない: ${plainModeRuns(steps, "speculate").join(",")}`,
  );
  assertEquals(gate.mode, "plain", "外れ続ける探索で戻ってしまっている");
  assertEquals(gate.switches, 1, "探索バーストを切替に数えている");
});

Deno.test("ゲート T4′ 戻る: バーストの集計が enter を下回ったら戻り、間隔は 16 に戻る", () => {
  const gate = createSpeculationGate({ window: 2, confirm: 1 });
  run(gate, LOSING, 3);
  assertEquals(gate.mode, "plain", "負けの壁で plain へ落ちる前提が崩れている");
  assertEquals(gate.switches, 1);

  // 課題が変わって投機が勝つようになる → 間隔 16 ぶん plain を回した後、8 cycle のバーストで
  // 測る（当たりのバーストは満ちるまで回る = 打ち切りは負け側にしか効かない）。
  const steps = trace(gate, WINNING, 24);
  assertEquals(steps.map((observed) => observed.decision), [
    ...Array.from({ length: 16 }, () => "plain"),
    ...Array.from({ length: 8 }, () => "speculate"),
  ]);
  assertEquals(
    steps.slice(16, 23).map((observed) => observed.mode),
    Array.from({ length: 7 }, () => "plain"),
    "バーストが満ちる前に戻っている（1 cycle の当たりで戻してはならない）",
  );
  assertEquals(gate.mode, "speculate", "当たりのバーストで戻っていない");
  assertEquals(gate.switches, 2);

  // もう一度負けさせると plain へ落ち、探索は**16 から**やり直す（伸びた間隔を持ち越さない）。
  const again = trace(gate, LOSING, 40);
  assertEquals(gate.mode, "plain", "負けに戻ったのに plain へ落ちていない");
  assertEquals(gate.switches, 3);
  assertEquals(
    plainModeRuns(again, "plain")[0],
    16,
    `戻った後の探索間隔が 16 でない: ${plainModeRuns(again, "plain").join(",")}`,
  );
});

Deno.test("ゲート T4″ 打ち切り: 強い負け（burstAbort）が出たバーストは burstMin 本目で畳む", () => {
  // 負ける文脈が払う探索費の主はここである（バースト 8 cycle × 間隔 16 → 32 → …）。当たりの
  // 判定は満ちたバーストのままなので、`burst` そのものを縮めるのとは違って戻る精度は落ちない。
  const gate = createSpeculationGate({ window: 2, confirm: 1 });
  run(gate, LOSING, 3);
  assertEquals(gate.mode, "plain", "負けの壁で plain へ落ちる前提が崩れている");
  assertEquals(gate.switches, 1);

  // 間隔 16 ぶん plain を回してバースト → 4 本目の観測で比 1.30 > 1.16 なので畳み、以後は
  // 倍の間隔（32 step）まで plain である。
  const steps = trace(gate, STRONG_LOSS, 53);
  assertEquals(
    plainModeRuns(steps, "plain"),
    [16, 32],
    `打ち切ったバーストの後の間隔が倍になっていない: ${plainModeRuns(steps, "plain").join(",")}`,
  );
  assertEquals(
    plainModeRuns(steps, "speculate"),
    [4],
    `バーストが 4 本目で畳まれていない: ${plainModeRuns(steps, "speculate").join(",")}`,
  );
  assertEquals(gate.mode, "plain");
  assertEquals(gate.switches, 1, "打ち切ったバーストを切替に数えている");

  // 対 1（打ち切りの位置は `burstMin` が決める — 同じ壁でも 5 本目までは畳まない）。
  const later = createSpeculationGate({ window: 2, confirm: 1, burstMin: 5 });
  run(later, LOSING, 3);
  assertEquals(
    plainModeRuns(trace(later, STRONG_LOSS, 22), "speculate"),
    [5],
    "burstMin を上げてもバーストが 4 本で畳まれている",
  );

  // 対 2（`leave` は超えるが強い負けでない比 1.10 のバーストは 8 cycle 回り切る）。
  const full = createSpeculationGate({ window: 2, confirm: 1 });
  run(full, LOSING, 3);
  const mild = trace(full, MILD, 25);
  assertEquals(
    plainModeRuns(mild, "speculate"),
    [8],
    `弱い負けのバーストを打ち切っている: ${plainModeRuns(mild, "speculate").join(",")}`,
  );
  assertEquals(full.mode, "plain", "比 1.10 のバーストで speculate へ戻っている");

  // 対 3（閾値は `burstAbort` が持つ — 上げれば同じ比 1.30 のバーストが回り切る。speculate 側の
  // `earlyLeave` とは別のノブであることの対でもある）。
  const lenient = createSpeculationGate({ window: 2, confirm: 1, burstAbort: 0.5 });
  run(lenient, LOSING, 3);
  const kept = trace(lenient, STRONG_LOSS, 25);
  assertEquals(
    plainModeRuns(kept, "speculate"),
    [8],
    `burstAbort を上げてもバーストを打ち切っている: ${plainModeRuns(kept, "speculate").join(",")}`,
  );
  assertEquals(lenient.mode, "plain", "比 1.30 のバーストで speculate へ戻っている");
});

Deno.test("ゲート T5 定常: 16 観測に 1 回 plain を測り、その step はブロックに数えない", () => {
  // `W1` が古いと「抜けるべきか」の判断材料が古いままになる。間隔は判定ブロックと同じ 16 で、
  // 費用は 16 cycle につき plain 価格の 1 token（always 比 1% 前後）— これは払う。
  const gate = createSpeculationGate();
  const decisions = run(gate, WINNING, 40);
  assertEquals(plainAt(decisions), [1, 15, 31]);
  assertEquals(gate.mode, "speculate");
  assertEquals(gate.switches, 0);

  // ブロックが数えるのは**投機 cycle だけ**である。window 4 の負け壁では、間に `W1` の plain が
  // 1 本挟まるので 4 本目の投機 cycle = 決定 5 本目でようやく満ちる。plain step もブロックに
  // 数える実装だと 1 手早く（決定 4 本目で）倒れる。
  const counted = createSpeculationGate({ window: 4, confirm: 1 });
  const steps = trace(counted, LOSING, 5);
  assertEquals(steps.map((observed) => observed.mode), [
    "speculate",
    "speculate",
    "speculate",
    "speculate",
    "plain",
  ]);
  assertEquals(counted.switches, 1);
});

Deno.test("ゲート T6 skip: 周期だけ進み、W1 もブロックも動かさない", async (t) => {
  await t.step("周期は進む（`W1` の初回サンプルがそのぶん早く来る）", () => {
    // 呼び手が「壁の観測に混ぜない」と決めた cycle（各ターンの最初の cycle）— 決定は読むが、
    // 返すのは観測ではなく `skip()` である。混ぜない cycle でも GPU の仕事は 1 本走っている。
    const gate = createSpeculationGate();
    assertEquals(gate.decide(), "speculate");
    gate.skip();
    assertEquals(gate.decide(), "plain", "skip した cycle のぶん周期が進んでいない");
    gate.observePlain(WINNING.plain);
    assertEquals(gate.mode, "speculate");
    assertEquals(gate.switches, 0);
  });

  await t.step("ブロックには数えない", () => {
    const gate = createSpeculationGate({ window: 2, confirm: 1 });
    assertEquals(gate.decide(), "speculate");
    gate.skip();
    assertEquals(gate.decide(), "plain");
    gate.observePlain(LOSING.plain);
    assertEquals(step(gate, LOSING), "speculate", "ブロック 1 本目");
    assertEquals(gate.mode, "speculate", "skip がブロックに数えられている");
    assertEquals(step(gate, LOSING), "speculate", "ブロック 2 本目 = 満ちる");
    assertEquals(gate.mode, "plain");
    assertEquals(gate.switches, 1);
  });
});

Deno.test("ゲート T7 冪等: decide は状態を動かさない（切替は観測で起きる）", async (t) => {
  // `decide()` は純粋な読み取りである。ここで縛るのは 2 方向: 同じ状態での重複読みが状態を
  // 進めないことと、状態が動いた（観測 / `skip`）直後の読みが**必ず作り直される**ことである。
  // 後者を見ないと、決定をメモ化して観測でだけ捨てる実装（= `skip` した cycle の決定が 1 手
  // 古いまま出る）が緑のまま通る。
  await t.step("起点（まだ何も測っていない）", () => {
    const gate = createSpeculationGate();
    assertEquals([gate.decide(), gate.decide(), gate.decide()], [
      "speculate",
      "speculate",
      "speculate",
    ]);
    assertEquals(gate.switches, 0);
  });

  await t.step("倒れた直後に何度読んでも、切替は 1 回・探索の周期もずれない", () => {
    const gate = createSpeculationGate({ window: 2, confirm: 1 });
    // 3 手目の**観測**で plain へ倒れる（決定ではなく観測が状態を動かす）。
    run(gate, LOSING, 3);
    assertEquals(gate.mode, "plain");
    assertEquals(gate.switches, 1);
    assertEquals([gate.decide(), gate.decide(), gate.decide()], ["plain", "plain", "plain"]);
    assertEquals(gate.switches, 1, "決定を読んだだけで切替が進んでいる");

    // 読みの回数は周期に効かないので、探索バーストはきっかり 17 手目に始まる。
    assertEquals(run(gate, LOSING, 17), [
      ...Array.from({ length: 16 }, () => "plain"),
      "speculate",
    ], "重複読みが探索の周期を進めている");
    assertEquals(gate.switches, 1);
    assertEquals(gate.mode, "plain");
  });

  await t.step("観測を返せば次の決定は作り直される", () => {
    const gate = createSpeculationGate();
    // 16 手目 = 定常の `W1` の測り直し。
    run(gate, WINNING, 15);
    assertEquals(gate.decide(), "plain");
    assertEquals(gate.decide(), "plain", "観測を返す前に決定が動いた");
    gate.observePlain(WINNING.plain);
    assertEquals(gate.decide(), "speculate", "観測を返しても決定が更新されていない");
  });

  await t.step("skip した cycle の決定も持ち越されない（周期は 1 進む）", () => {
    const gate = createSpeculationGate();
    assertEquals(gate.decide(), "speculate");
    gate.skip();
    // 周期が 1 進んだので、次は `W1` の初回サンプル（2 回目の決定 = plain）。決定を持ち越す
    // 実装だと、ここに 1 手古い speculate が出る。
    assertEquals(gate.decide(), "plain", "skip した cycle の決定が持ち越されている");
    gate.observePlain(WINNING.plain);
    // `skip` はブロックを動かさない = 投機側の観測はまだ 1 本も無く、判定は出ない。
    assertEquals(gate.mode, "speculate");
    assertEquals(gate.switches, 0);
  });
});

/** 実測 1 条件（`docs/research/2026-09-09-mtp-stage4.md` §2 表 1 = RTX / §5.1 = M2・greedy）。 */
type Condition = Walls & {
  readonly name: string;
  /** 表から導いた比 `(Wc/A)/W1`（1 未満なら投機が速い）。 */
  readonly ratio: number;
  /** ブロック 2 本ぶん回した後に居るべきモード（既定の席 = 早抜け off）。 */
  readonly mode: SpeculationDecision;
  /**
   * `earlyLeave: 0.15` を渡した席で plain へ倒れるまでに要るブロック数（`mode` が plain の条件だけ
   * 持つ）。
   *
   * 1 = 比が `leave + earlyLeave` = 1.16 を超えるので早抜けする条件・2 = `leave` は超えるが 1.16 に
   * 届かないので `confirm` 本連続を待つ条件。**既定の席ではどの条件も 2 である**（早抜けが off）。
   */
  readonly eagerBlocks?: 1 | 2;
};

const CONDITIONS: readonly Condition[] = [
  { name: "RTX 抽出", cycle: 53.7, delivered: 3.39, plain: 29.22, ratio: 0.542, mode: "speculate" },
  { name: "RTX 要約", cycle: 55.8, delivered: 2.65, plain: 29.67, ratio: 0.710, mode: "speculate" },
  { name: "RTX 対話", cycle: 46.8, delivered: 2.03, plain: 27.13, ratio: 0.850, mode: "speculate" },
  {
    name: "RTX 自由文",
    cycle: 46.5,
    delivered: 1.66,
    plain: 27.02,
    ratio: 1.037,
    mode: "plain",
    eagerBlocks: 2,
  },
  { name: "M2 抽出", cycle: 172.4, delivered: 3.39, plain: 66.03, ratio: 0.770, mode: "speculate" },
  { name: "M2 要約", cycle: 170.7, delivered: 2.65, plain: 66.39, ratio: 0.970, mode: "speculate" },
  {
    name: "M2 対話",
    cycle: 123.0,
    delivered: 2.03,
    plain: 56.14,
    ratio: 1.079,
    mode: "plain",
    eagerBlocks: 2,
  },
  {
    name: "M2 自由文",
    cycle: 118.2,
    delivered: 1.66,
    plain: 54.45,
    ratio: 1.308,
    mode: "plain",
    eagerBlocks: 1,
  },
];

Deno.test("ゲート T8 実測: 8 条件のうち負ける 3 本だけが plain へ落ちる", async (t) => {
  for (const condition of CONDITIONS) {
    await t.step(`${condition.name}（比 ${condition.ratio}）`, () => {
      // 定数が research の表と同じものであること（表を写し間違えた瞬間に赤くなる）。
      assertAlmostEquals(
        (condition.cycle / condition.delivered) / condition.plain,
        condition.ratio,
        0.001,
        `${condition.name}: 壁の定数から出る比が表の値と違う`,
      );
      const gate = createSpeculationGate();
      // 決定 18 本 = 投機 cycle 16 本 = ブロックちょうど 1 本（plain は `W1` の測り直しだけで、
      // 決定 1 / 15 の 2 本 ≒ 1/16）。既定は早抜け off なので、**どの条件もここでは倒れない**。
      const decisions = run(gate, condition, 18);
      assertEquals(plainAt(decisions), [1, 15], `${condition.name}: 測り直しの本数`);
      assertEquals(
        gate.mode,
        "speculate",
        `${condition.name}: ブロック 1 本ぶんの後のモード（既定は早抜け off）`,
      );

      // 続く決定 17 本（測り直しは決定 31 の 1 本）でブロック 2 本目が満ちる。
      run(gate, condition, 17);
      assertEquals(gate.mode, condition.mode, `${condition.name}: ブロック 2 本ぶんの後のモード`);
      assertEquals(
        gate.switches,
        condition.mode === "plain" ? 1 : 0,
        `${condition.name}: 切替回数`,
      );

      // 早抜けを渡した席では、比 1.16 を超える条件だけが 1 ブロックで倒れる（勝つ 5 条件は
      // どちらの席でも倒れない — 早抜けは「抜けるまでの費用」しか動かさない）。
      const eager = createSpeculationGate({ earlyLeave: 0.15 });
      run(eager, condition, 18);
      assertEquals(
        eager.mode,
        condition.eagerBlocks === 1 ? "plain" : "speculate",
        `${condition.name}: earlyLeave 0.15 の席でブロック 1 本ぶんの後のモード`,
      );
      run(eager, condition, 17);
      assertEquals(
        eager.mode,
        condition.mode,
        `${condition.name}: earlyLeave 0.15 の席でブロック 2 本ぶんの後のモード`,
      );
    });
  }

  await t.step("不感帯（M2 要約の 0.970）は enter も leave も跨がない", () => {
    // 0.970 は「戻る」側の閾値 0.97 のすぐ上で、`< enter` を満たさない = 現状維持である。
    // 3〜4% の差は受理数のノイズ（cycle あたり sd は平均の 43%）に埋もれるので、動かさない
    // のが正しい。
    const gate = createSpeculationGate();
    const summarize = CONDITIONS[5];
    run(gate, summarize, 120);
    assertEquals(gate.mode, "speculate");
    assertEquals(gate.switches, 0);

    // 同じ比でも、plain 側から見ると戻らない（不感帯は両側から効く）。`W1` の EWMA は
    // 54.45 から 66.39 へ**下から**寄るので、戻る閾値 `W1 × 0.97` は 64.40 を超えない
    // （バーストの実測は 170.7 / 2.65 = 64.42 ms/token）。
    const stuck = createSpeculationGate();
    run(stuck, CONDITIONS[7], 60);
    assertEquals(stuck.mode, "plain", "M2 自由文で落ちる前提が崩れている");
    run(stuck, summarize, 600);
    assertEquals(stuck.mode, "plain", "不感帯の比で speculate へ戻っている");
    assertEquals(stuck.switches, 1);
  });
});

Deno.test("ゲート T9 門: ノブの値域は生成時に fail loudly", async (t) => {
  await t.step("alpha は 0 < α ≤ 1", () => {
    assertThrows(
      () => createSpeculationGate({ alpha: 0 }),
      Error,
      "alpha 0 が正の有限数でない",
    );
    assertThrows(
      () => createSpeculationGate({ alpha: Number.NaN }),
      Error,
      "alpha NaN が正の有限数でない",
    );
    assertThrows(
      () => createSpeculationGate({ alpha: 1.5 }),
      Error,
      "alpha 1.5 が 0 < α ≤ 1 の外",
    );
  });

  await t.step("ブロックは 2 cycle 以上（1 なら 1 サンプル判定に戻る）", () => {
    assertThrows(
      () => createSpeculationGate({ window: 1 }),
      Error,
      "window 1 が 2 以上の整数でない",
    );
    assertThrows(
      () => createSpeculationGate({ window: 16.5 }),
      Error,
      "window 16.5 が 2 以上の整数でない",
    );
  });

  await t.step("連続本数とバーストは 1 以上の整数", () => {
    assertThrows(
      () => createSpeculationGate({ confirm: 0 }),
      Error,
      "confirm 0 が 1 以上の整数でない",
    );
    assertThrows(
      () => createSpeculationGate({ burst: 0 }),
      Error,
      "burst 0 が 1 以上の整数でない",
    );
  });

  await t.step("打ち切りの下限は 1 以上 burst 以下の整数", () => {
    assertThrows(
      () => createSpeculationGate({ burstMin: 0 }),
      Error,
      "burstMin 0 が 1 以上の整数でない",
    );
    assertThrows(
      () => createSpeculationGate({ burstMin: 4.5 }),
      Error,
      "burstMin 4.5 が 1 以上の整数でない",
    );
    // `burstMin > burst` は一度も発火しない死んだノブ（既定の `burst` は 8）。
    assertThrows(
      () => createSpeculationGate({ burstMin: 9 }),
      Error,
      "burstMin 9 が burst 8 を超えている",
    );
    // 既定の `burstMin` は `burst` で抑えてある（バースト 1 cycle の席が門で落ちない）。
    assertEquals(createSpeculationGate({ burst: 1 }).mode, "speculate");
  });

  await t.step("強い負けの上乗せは 2 本とも正の有限数（早抜けは未指定 = off）", () => {
    for (
      const [name, value] of [
        ["earlyLeave", 0],
        ["earlyLeave", -1],
        ["earlyLeave", Number.POSITIVE_INFINITY],
        ["burstAbort", 0],
        ["burstAbort", -1],
        ["burstAbort", Number.POSITIVE_INFINITY],
      ] as const
    ) {
      assertThrows(
        () => createSpeculationGate({ [name]: value }),
        Error,
        `${name} ${value} が正の有限数でない`,
      );
    }
    // 未指定の `earlyLeave` は「off」であって門の対象ではない（`0` を渡す形と混ぜない — 0 は
    // 「閾値 = `leave` の早抜け」を意味してしまうので落とす）。
    assertEquals(createSpeculationGate({}).mode, "speculate");
    assertEquals(createSpeculationGate({ earlyLeave: undefined }).mode, "speculate");
  });

  await t.step("探索の間隔は 1 以上の整数・上限は基本間隔以上", () => {
    assertThrows(
      () => createSpeculationGate({ exploreBase: 0 }),
      Error,
      "exploreBase 0 が 1 以上の整数でない",
    );
    assertThrows(
      () => createSpeculationGate({ exploreBase: 1.5 }),
      Error,
      "exploreBase 1.5 が 1 以上の整数でない",
    );
    assertThrows(
      () => createSpeculationGate({ exploreBase: 8, exploreMax: 4 }),
      Error,
      "exploreMax 4 が 8 以上の整数でない",
    );
  });

  await t.step("不感帯は 1 を挟む（enter < 1 < leave）", () => {
    assertThrows(
      () => createSpeculationGate({ enter: 1 }),
      Error,
      "enter 1 < 1 < leave 1.01 でない",
    );
    assertThrows(
      () => createSpeculationGate({ leave: 0.99 }),
      Error,
      "enter 0.97 < 1 < leave 0.99 でない",
    );
    assertThrows(
      () => createSpeculationGate({ enter: -0.5 }),
      Error,
      "enter -0.5 が正の有限数でない",
    );
  });

  await t.step("端の値は通る（門が広すぎないことの対）", () => {
    const gate = createSpeculationGate({
      alpha: 1,
      window: 2,
      confirm: 1,
      burst: 1,
      burstMin: 1,
      exploreBase: 1,
      exploreMax: 1,
      enter: 0.5,
      leave: 2,
      earlyLeave: 0.5,
      burstAbort: 0.5,
    });
    assert(gate.mode === "speculate");
  });
});
