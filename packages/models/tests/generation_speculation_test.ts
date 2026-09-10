// 投機的デコード（ADR 0096 段 3）の挙動テスト。GPU も実資産も要らない。
//
// ここで縛るのは「投機は速さだけを変え、**出力は変えない**」という契約である:
//
// - token 列・位置・停止理由・`tokens` の勘定が、同じ要求を**非投機で走らせた参照走行**と
//   厳密に一致する（受理 0 でも全受理でも部分受理でも）。ここが割れる退行は例外を出さず、
//   「たまに文章が変わる」形でしか現れない。**温度に依らない** — 受理は行ごとに `sampler.next`
//   を非投機の decode と同じ logits・同じ history・同じ順で 1 回ずつ（確定 token 1 個につき
//   1 回）呼ぶので、RNG の消費列も列も一致する（T3′）。
// - 会話に入るのは「消費者が受け取った token + frontier 1 個」だけである（`commit(rows)` の
//   行数）。棄却行・停止 token より後ろの確定候補・`break` 後の未配送 token は KV に入らない。
// - 保留（deferred run）を残したまま抜ける経路が無い（停止・`break`・例外・中断のどれでも）。
//   保留が残った context は dispose しか受け付けなくなるので、次のターンが必ず落ちる。
//
// 参照走行との突合を「同じ fake・同じ後続関数」で採れるのは、fake の logits が
// {@link FakeOptions.successor}（その行の入力 token → その行の argmax）で決まるからである。
// 実装が verify の行を取り違えれば、参照走行と違う列が出る。
//
// T1〜T15 は**ゲート抜き**（`policy: "always"`）の契約である — 自己採算ゲート（段 4-B ④）が
// 割り込むと 1 cycle = draft + verify の勘定が崩れ、壁時計にも依存する。ゲートの席は T16 が
// 偽時計で見る（ゲートそのものの判断は `generation_gate_test.ts`）。
//
// T17 は観測席が運ぶ**壁と局面**（`wallMs` / `delivered` / `gate`）の席である。run の種別と番号
// だけを見る門（T7 / T9）は `runShape` で壁の欄を落として比べる — 非投機の sequence には偽時計を
// 差せないので、壁の実値はそれらの門では見られない。

import {
  assert,
  assertEquals,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import type { GenerationProgramSpec } from "../src/generation/program.ts";
import { createSampler, type Sampler } from "../src/generation/sampler.ts";
import {
  createGenerationSequence,
  type GenerationEvent,
  type GenerationGateTrace,
  type GenerationRequest,
  type GenerationRunPhase,
  type GenerationSequence,
  type GenerationStop,
  type GenerationStream,
} from "../src/generation/sequence.ts";
import {
  acceptDrafts,
  type DraftCycle,
  type DraftFace,
  planDraftLength,
  takeDrafts,
  verifyRowIndices,
} from "../src/generation/speculation.ts";
import type { SpeculationGateOptions } from "../src/generation/speculation-gate.ts";
import {
  drain,
  type FakeOptions,
  type FakeSession,
  fakeSession,
  hiddenMark,
  programOf,
  type RunCall,
  runShape,
  tokenIds,
  VOCAB,
} from "./helpers/generation-fake.ts";

// ---- fake target（後続関数）と配線 ------------------------------------------

/**
 * fake target の後続関数（行の入力 token → その行の argmax）。
 *
 * 像は 2..14 なので **0 / 1 / 15 は絶対に出ない** — 0 は pad 行の値、15 は
 * {@link REJECTED}（必ず棄却される draft）として使える。
 */
const SUCCESSOR: readonly number[] = Array.from(
  { length: VOCAB },
  (_unused, token) => ((token * 7 + 3) % 13) + 2,
);

/** target が選びえない token（`SUCCESSOR` の像の外）= 必ず棄却される draft。 */
const REJECTED = 15;

/**
 * 温度 > 0 のターンで実際に 2 択になる logits（第 1 候補 10 / 対抗馬 9.5 ≒ 62% : 38%）。
 *
 * 既定の fake は「狙った id だけ 10・他は全部 0」の単峰なので、温度を上げても抽選が実質 1 点に
 * 潰れる — RNG は消費されるが**列が動かない**ので、「投機と非投機で RNG の消費列が並ぶ」ことを
 * 見たことにならない。対抗馬の id は `SUCCESSOR` の像（2..14）の外に取る（行の第 1 候補と
 * 衝突すると、その行だけ単峰へ戻る）。
 */
const RUNNER_UP = 1;
const TWO_WAY: FakeOptions = { successor: SUCCESSOR, runnerUp: { id: RUNNER_UP, logit: 9.5 } };
/** 抽選が第 1 候補と対抗馬の両方を引く seed（下の step が「違う列になる」ことで縛る）。 */
const TWO_WAY_SEED = 7;

/** そのターンの prompt（末尾 2 から `SUCCESSOR` を辿ると 6, 8, 9, 3, 13, 5, … と続く）。 */
const PROMPT: readonly number[] = [1, 2];

/** draft の段数（配布形の drafter に相当）。 */
const K = 3;

/** 投機の配線: verify の `k+1 = 4` 行を載せるバケットが要る（chunkLength は別形として残す）。 */
const SPEC_CHUNK_LENGTH = 8;
const SPEC_BUCKETS: readonly number[] = [4];

const specProgram = (fake: FakeSession, override: Partial<GenerationProgramSpec> = {}) =>
  programOf(fake, {
    chunkLength: SPEC_CHUNK_LENGTH,
    chunkBuckets: [...SPEC_BUCKETS],
    ...override,
  });

/** `token` の次から `count` 個ぶん、target が本当に出す列（= 全部当たる draft）。 */
const chainFrom = (token: number, count: number): number[] => {
  const chain: number[] = [];
  let current = token;
  for (let index = 0; index < count; index += 1) {
    current = SUCCESSOR[current];
    chain.push(current);
  }
  return chain;
};

// ---- fake drafter（DI で差す借り手の面）--------------------------------------

/** drafter が受け取った 1 cycle ぶんの入力（写しで持つ — 呼び手が使い回しても記録は動かない）。 */
type DraftRecord = {
  readonly token: number;
  readonly position: number;
  readonly hidden: readonly number[];
};

type FakeDrafter = ReturnType<typeof fakeDrafter>;

const fakeDrafter = (options: {
  /** 1 run で出す本数（既定 {@link K}）。呼び手はこのうち先頭 `k'` 本しか使わない。 */
  readonly steps?: number;
  /** cycle ごとの draft（`index` は 0 始まりの cycle 番号）。 */
  readonly draft: (cycle: DraftCycle, index: number) => readonly number[];
  /** dispose の順序を見るための記録先（`"drafter"` を積む）。 */
  readonly disposeLog?: string[];
}) => {
  const steps = options.steps ?? K;
  const calls: DraftRecord[] = [];
  let disposals = 0;
  /** 貸し手（{@link watch} で結ぶ）— 借り手 run が実 runtime で拒まれる条件をここで写す。 */
  let lender: FakeSession | undefined;
  const face: DraftFace = {
    steps,
    draft: (cycle) => {
      // MUST: 実 runtime では借り手 run が**貸し手のリースも**取るので、貸し手に未 commit の
      // deferred run が残っている間の draft は拒否される（`generation-context.ts` の
      // `acquireRun`）。draft を verify の直前から受理直後へ前倒しする最適化はこの規則を破るが、
      // 貸し手を見ない fake だと GPU 無しでは 1 本も赤くならない。
      assertEquals(
        lender?.pendingCommit(),
        undefined,
        "貸し手に保留（未 commit の verify）が残ったまま draft を採った",
      );
      const index = calls.length;
      calls.push({ token: cycle.token, position: cycle.position, hidden: [...cycle.hidden] });
      return Promise.resolve(Int32Array.from(options.draft(cycle, index)));
    },
    dispose: (): Promise<void> => {
      disposals += 1;
      options.disposeLog?.push("drafter");
      return Promise.resolve();
    },
  };
  return {
    face,
    calls,
    disposals: (): number => disposals,
    /** 貸し手を結ぶ（sequence を組む側が呼ぶ — 借り手は貸し手の後に作られる）。 */
    watch: (fake: FakeSession): void => {
      lender = fake;
    },
  };
};

/** 常に外れる drafter（受理 0 を強制する）。 */
const wrongDrafter = (disposeLog?: string[]): FakeDrafter =>
  fakeDrafter({
    draft: () => Array.from({ length: K }, () => REJECTED),
    disposeLog,
  });

/** target の未来をそのまま写す drafter（全受理を強制する）。 */
const oracleDrafter = (steps = K): FakeDrafter =>
  fakeDrafter({ steps, draft: (cycle) => chainFrom(cycle.token, steps) });

/** 先頭 `accept` 本だけ当てる drafter（部分受理 — `accept` 本目の次で必ず外す）。 */
const partialDrafter = (accept: number): FakeDrafter =>
  fakeDrafter({
    draft: (cycle) =>
      chainFrom(cycle.token, K).map((id, index) => (index < accept ? id : REJECTED)),
  });

// ---- 走行のひな型 -------------------------------------------------------------

const verifyCalls = (fake: FakeSession): RunCall[] =>
  fake.calls.filter((call) => call.commit === "deferred");

/** `speculation` 欄を落とした停止（非投機の参照走行と比べるため）。 */
const withoutSpeculation = (stop: GenerationStop): Record<string, unknown> => {
  const fields: Record<string, unknown> = { ...stop };
  delete fields.speculation;
  return fields;
};

const openSpeculative = async (options: {
  readonly drafter: FakeDrafter;
  readonly k?: number;
  readonly program?: Partial<GenerationProgramSpec>;
  readonly session?: FakeOptions;
  /**
   * 投機の張り方（既定 `"always"`）。
   *
   * このファイルの門は**ゲート抜きの契約**（1 cycle = draft + verify）なので、既定を
   * `"always"` に倒してある — 自己採算ゲートの席は T16 が偽時計で別に見る。
   */
  readonly policy?: "auto" | "always";
  /**
   * ゲートのノブ（`policy: "auto"` のときだけ効く）。
   *
   * 既定のブロックは 16 cycle × 2 本連続なので、20 token のターンでは**一度も判定が出ない**。
   * ゲートの配線（落ちた step の勘定・hidden の繋がり・観測に混ぜない cycle）を見る席では
   * ブロックを縮めて、判定そのものは `generation_gate_test.ts` で見る。
   */
  readonly gate?: SpeculationGateOptions;
  /** 偽時計（`policy: "auto"` のときだけ読まれる）。 */
  readonly now?: () => number;
  readonly clock?: { readonly now: () => number; readonly afterRun: () => void };
  readonly onRun?: (phase: GenerationRunPhase) => void;
}) => {
  const fake = fakeSession({ successor: SUCCESSOR, ...options.session });
  options.drafter.watch(fake);
  const phases: GenerationRunPhase[] = [];
  const sequence = await createGenerationSequence({
    session: {
      ...fake.session,
      run: async (...args) => {
        const result = await fake.session.run(...args);
        if (args[2]?.commit === "deferred") options.clock?.afterRun();
        return result;
      },
    },
    program: specProgram(fake, options.program),
    speculative: {
      open: () => Promise.resolve(options.drafter.face),
      k: options.k,
      policy: options.policy ?? "always",
      ...(options.gate === undefined ? {} : { gate: options.gate }),
      ...((options.clock?.now ?? options.now) === undefined
        ? {}
        : { now: options.clock?.now ?? options.now }),
    },
    onRun: (phase) => {
      phases.push(phase);
      options.onRun?.(phase);
    },
  });
  return { fake, drafter: options.drafter, phases, sequence };
};

const runSpeculative = async (options: {
  readonly drafter: FakeDrafter;
  readonly request: GenerationRequest;
  readonly k?: number;
  readonly program?: Partial<GenerationProgramSpec>;
  readonly session?: FakeOptions;
  readonly policy?: "auto" | "always";
  readonly gate?: SpeculationGateOptions;
  readonly now?: () => number;
}) => {
  const opened = await openSpeculative(options);
  const drained = await drain(opened.sequence.generate(options.request));
  return { ...opened, ...drained };
};

/** **非投機**（drafter を差さない）sequence を組む — 投機と同じ不変条件を見る対の側。 */
const openPlain = async (
  program: Partial<GenerationProgramSpec> = {},
  session: FakeOptions = {},
) => {
  const fake = fakeSession({ successor: SUCCESSOR, ...session });
  const phases: GenerationRunPhase[] = [];
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: specProgram(fake, program),
    onRun: (phase) => {
      phases.push(phase);
    },
  });
  return { fake, phases, sequence };
};

/** 同じ要求を**非投機**で走らせた参照走行。 */
const runPlain = async (
  request: GenerationRequest,
  program: Partial<GenerationProgramSpec> = {},
  session: FakeOptions = {},
) => {
  const opened = await openPlain(program, session);
  const drained = await drain(opened.sequence.generate(request));
  return { ...opened, ...drained };
};

/**
 * 投機の走行が非投機の走行と**同じ出力**であることを見る（この契約の芯）。
 *
 * 比べるのはイベント列（token id と位置・prefill の進捗）と停止（`speculation` 欄を除く全欄）。
 */
const assertMatchesPlain = async (
  request: GenerationRequest,
  speculated: { readonly events: readonly GenerationEvent[]; readonly stop: GenerationStop },
  program: Partial<GenerationProgramSpec> = {},
  session: FakeOptions = {},
): Promise<void> => {
  const plain = await runPlain(request, program, session);
  assertEquals(speculated.events, plain.events, "投機の token 列 / 位置が非投機と一致しない");
  assertEquals(
    withoutSpeculation(speculated.stop),
    withoutSpeculation(plain.stop),
    "投機の停止が非投機と一致しない",
  );
  assertEquals(
    Object.hasOwn(plain.stop, "speculation"),
    false,
    "非投機の sequence の停止に speculation 欄がある",
  );
};

// ---- T1〜T3: 受理数と出力の同一性 ---------------------------------------------

Deno.test("T1 強制棄却: 全部外れる draft でも非投機と同じ列（cycle ごとに frontier 1 個だけ確定）", async () => {
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 5 };
  const run = await runSpeculative({ drafter: wrongDrafter(), request });
  await assertMatchesPlain(request, run);

  // 受理 0 の cycle が確定させるのは b′ 1 個だけ = 非投機の decode 1 step への退化。
  assertEquals(run.fake.commits, [1, 1, 1, 1], "受理 0 の cycle が 1 行より多く commit した");
  assertEquals(run.stop.speculation, {
    cycles: 4,
    draftRuns: 3,
    drafted: 6,
    accepted: 0,
    delivered: 4,
    acceptedHistogram: [4, 0, 0, 0],
  });
  // 予算の残り（`k' = min(k, 残り − 1)`）で本数が縮み、最後の cycle は draft を採らない。
  assertEquals(run.drafter.calls.length, 3);
  assertEquals(verifyCalls(run.fake).map((call) => call.queryLength), [4, 3, 2, 1]);
});

Deno.test("T2 全受理: 未来をそのまま写す draft は 1 verify で k+1 個を確定させる", async () => {
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
  const run = await runSpeculative({ drafter: oracleDrafter(), request });
  await assertMatchesPlain(request, run);

  assertEquals(run.fake.commits, [4, 4], "全受理の cycle は k+1 行を確定させる");
  assertEquals(run.stop.speculation, {
    cycles: 2,
    draftRuns: 2,
    drafted: 6,
    accepted: 6,
    delivered: 8,
    acceptedHistogram: [0, 0, 0, 2],
  });
  // 貸し手の run は prefill 1 + verify 2 の 3 本（非投機は prefill 1 + decode 8 の 9 本）。
  assertEquals(run.fake.calls.length, 3);
  assertEquals((await runPlain(request)).fake.calls.length, 9);
});

Deno.test("T3 部分受理: 先頭一致で止まった位置が受理数になる（a = 1 / 2）", async (t) => {
  await t.step("a = 1（1 本目だけ当たる）", async () => {
    const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 7 };
    const run = await runSpeculative({ drafter: partialDrafter(1), request });
    await assertMatchesPlain(request, run);

    // 確定は `[d₁, b′]` の 2 個ずつ（棄却行 d₂ / d₃ は会話に入らない）。
    assertEquals(run.fake.commits, [2, 2, 2]);
    assertEquals(run.stop.speculation, {
      cycles: 3,
      draftRuns: 3,
      drafted: 7,
      accepted: 3,
      delivered: 6,
      acceptedHistogram: [0, 3, 0, 0],
    });
  });

  await t.step("a = 2（2 本目まで当たる）", async () => {
    const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 7 };
    const run = await runSpeculative({ drafter: partialDrafter(2), request });
    await assertMatchesPlain(request, run);

    assertEquals(run.fake.commits, [3, 3]);
    assertEquals(run.stop.speculation, {
      cycles: 2,
      draftRuns: 2,
      drafted: 5,
      accepted: 4,
      delivered: 6,
      acceptedHistogram: [0, 0, 2, 0],
    });
  });
});

Deno.test("T3′ 温度 > 0: 抽選が走るターンでも投機を張り、列は非投機と厳密一致する", async (t) => {
  // 受理は行ごとに `sampler.next` を「非投機の decode が同じ位置で行う抽選」と同じ logits・
  // 同じ history・同じ順で **確定 token 1 個につき 1 回**呼ぶ。RNG の消費列が非投機と 1 対 1 に
  // 並ぶので、温度 > 0 でも token 列は変わらない（one-hot draft の speculative sampling）。
  const request = (seed: number): GenerationRequest => ({
    prompt: PROMPT,
    maxNewTokens: 9,
    sampler: { temperature: 1, topK: 2, seed },
  });

  await t.step("投機 / 非投機の列・位置・停止が一致し、勘定が載る", async () => {
    const run = await runSpeculative({
      drafter: oracleDrafter(),
      request: request(TWO_WAY_SEED),
      session: TWO_WAY,
    });
    await assertMatchesPlain(request(TWO_WAY_SEED), run, {}, TWO_WAY);

    const speculation = run.stop.speculation;
    assert(speculation !== undefined, "温度 > 0 のターンに speculation 欄が載っていない");
    assert(speculation.cycles > 0, "verify run が 1 本も回っていない");
    assertEquals(withoutSpeculation(run.stop), { reason: "max-tokens", tokens: 9 });
    // draft（= 温度 0 の連鎖）が当たる cycle と外れる cycle の両方を通る = 受理判定が抽選の
    // 結果で分岐している（全受理／全棄却に張り付いていない）。
    assert(
      speculation.accepted > 0 && speculation.accepted < speculation.drafted,
      `受理が全部か 0 に張り付いている: ${JSON.stringify(speculation)}`,
    );
  });

  await t.step("抽選は実際に走っている（温度 0 の列とは違う列になる）", async () => {
    const sampled = await runSpeculative({
      drafter: oracleDrafter(),
      request: request(TWO_WAY_SEED),
      session: TWO_WAY,
    });
    const greedy = await runSpeculative({
      drafter: oracleDrafter(),
      request: { prompt: PROMPT, maxNewTokens: 9 },
      session: TWO_WAY,
    });
    // 温度 0 の列は後続関数そのもの（対抗馬 9.5 は第 1 候補 10 を越えない）。
    assertEquals(tokenIds(greedy.events), chainFrom(PROMPT[PROMPT.length - 1], 9));
    assert(
      tokenIds(sampled.events).join() !== tokenIds(greedy.events).join(),
      `温度 1 でも argmax の列しか出ていない（RNG が消費されていない）: ${
        tokenIds(sampled.events).join(",")
      }`,
    );
  });

  await t.step("自己採算ゲートが落とす plain step を挟んでも列は同じ", async () => {
    // 温度 > 0 とゲートの積（負ける壁を偽時計で流す）。ゲートが落とす decode 形の step も、
    // 抽選は「非投機の decode が同じ位置で行う抽選」そのもの（同じ logits・同じ history・
    // 同じ順で 1 回）なので、RNG の消費列は投機 cycle と plain step が混ざっても並んだままである。
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      session: TWO_WAY,
      clock: fakeClock(drafter, (_cycle, drafted) => drafted ? 100 : 10),
    });
    // ブロック（2 cycle）が満ちるまで回す必要があるので、この step だけ予算を伸ばす。
    const budgeted: GenerationRequest = { ...request(TWO_WAY_SEED), maxNewTokens: 24 };
    const gated = { ...opened, ...await drain(opened.sequence.generate(budgeted)) };
    // ゲートが本当に**倒れている**（`W1` の測り直し 1 手だけを見て「落ちた」と読まない）。
    assertEquals(
      gated.stop.speculation?.switches,
      1,
      `ゲートが plain へ倒れていない: ${JSON.stringify(gated.stop.speculation)}`,
    );
    await assertMatchesPlain(budgeted, gated, {}, TWO_WAY);

    // `always`（ゲート無し）とも一致する = ゲートは速さだけを変えている。
    const always = await runSpeculative({
      drafter: oracleDrafter(),
      request: budgeted,
      session: TWO_WAY,
    });
    assertEquals(tokenIds(gated.events), tokenIds(always.events), "ゲート付きと always の列が違う");
  });

  await t.step("seed が違えば列も違う（RNG 状態が生成 1 本に張り付いている）", async () => {
    const first = await runSpeculative({
      drafter: oracleDrafter(),
      request: request(TWO_WAY_SEED),
      session: TWO_WAY,
    });
    const second = await runSpeculative({
      drafter: oracleDrafter(),
      request: request(TWO_WAY_SEED + 1),
      session: TWO_WAY,
    });
    await assertMatchesPlain(request(TWO_WAY_SEED + 1), second, {}, TWO_WAY);
    assert(
      tokenIds(first.events).join() !== tokenIds(second.events).join(),
      "seed を変えても同じ列が出ている",
    );
  });
});

// ---- T4 / T5: 停止 token -------------------------------------------------------

Deno.test("T4 停止 token が draft の途中: そこで受理を打ち切り、後ろの行は確定させない", async () => {
  // 連鎖は 6 → 8 → 9 → 3 → 13。停止 token 9 は cycle 1 の d₂ に当たる。
  const stopTokens = [9];
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request,
    program: { stopTokens },
  });
  await assertMatchesPlain(request, run, { stopTokens });

  // 停止 token 自体は本文でないので `token` イベントに出さず、勘定には 1 個として入る。
  assertEquals(tokenIds(run.events), [6, 8]);
  assertEquals(withoutSpeculation(run.stop), { reason: "eos", token: 9, tokens: 3 });
  // 受理は停止 token の 9 まで（d₃ の行は抽選もしない）= 確定列 `[8, 9]` そのものが配送列。
  const speculation = run.stop.speculation;
  assertEquals(speculation, {
    cycles: 1,
    draftRuns: 1,
    drafted: 3,
    accepted: 2,
    delivered: 2,
    acceptedHistogram: [0, 0, 1, 0],
  });
  // token/cycle の正本が `delivered / cycles` である理由（この cycle は `1 + a` 個を確定させない）。
  assert(speculation !== undefined);
  assertEquals(speculation.accepted + speculation.cycles, 3, "旧式の分子は 1 だけ過大になる");
  assertEquals(run.fake.commits, [2], "停止 token より後ろの確定候補まで commit した");
  assertEquals(verifyCalls(run.fake)[0].queryLength, 4, "物理 ring へは k+1 行が書かれている");

  // 停止 token は会話に残る = 次ターンの prefill 先頭へ連結される（非投機と同じ後始末）。
  await drain(run.sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  const next = run.fake.calls[run.fake.calls.length - 1];
  assertEquals(next.ids, [9, 4, 0, 0]);
  assertEquals(next.positions, [4, 5, 0, 0]);
});

Deno.test("T4′ 停止 token の先の行は抽選しない（非投機が触れない logits で落ちない）", async () => {
  // 故障注入: verify（call 1）の**行 2 だけ**に NaN を混ぜる。行 2 は「停止 token 9 を確定した
  // 後の行」= 非投機なら run そのものが出ない位置である。受理を停止 token で打ち切らない実装は
  // ここでも `sampler.next` を呼ぶので `assertNoNaN` が投げる（範囲外 gather の行ごと NaN 汚染で
  // 実際に起きる形 — 「投機のときだけ落ちる」= 出力が変わらないという契約の破れ）。
  const stopTokens = [9];
  const session: FakeOptions = { successor: SUCCESSOR, nanAt: 1, nanRow: 2 };
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request,
    program: { stopTokens },
    session,
  });
  // 非投機の参照走行（decode は 1 行なので行 2 が無く、汚染に当たらない）と同じ列で閉じる。
  await assertMatchesPlain(request, run, { stopTokens }, session);

  assertEquals(tokenIds(run.events), [6, 8]);
  assertEquals(withoutSpeculation(run.stop), { reason: "eos", token: 9, tokens: 3 });
  assertEquals(run.fake.commits, [2]);
});

Deno.test("T5 停止 token が b′: 受理した draft は届き、新 frontier で閉じる", async (t) => {
  await t.step("受理 1 本の後に b′ が停止 token", async () => {
    // 6 → 8（受理）→ b′ = 9（停止）。打ち切りは起きず、確定列がそのまま停止で閉じる形。
    const stopTokens = [9];
    const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
    const run = await runSpeculative({
      drafter: partialDrafter(1),
      request,
      program: { stopTokens },
    });
    await assertMatchesPlain(request, run, { stopTokens });

    assertEquals(tokenIds(run.events), [6, 8]);
    assertEquals(withoutSpeculation(run.stop), { reason: "eos", token: 9, tokens: 3 });
    assertEquals(run.stop.speculation, {
      cycles: 1,
      draftRuns: 1,
      drafted: 3,
      accepted: 1,
      delivered: 2,
      acceptedHistogram: [0, 1, 0, 0],
    });
    assertEquals(run.fake.commits, [2]);
  });

  await t.step("受理 0 で b′ が停止 token（確定列は 1 個）", async () => {
    const stopTokens = [8];
    const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
    const run = await runSpeculative({ drafter: wrongDrafter(), request, program: { stopTokens } });
    await assertMatchesPlain(request, run, { stopTokens });

    assertEquals(tokenIds(run.events), [6]);
    assertEquals(withoutSpeculation(run.stop), { reason: "eos", token: 8, tokens: 2 });
    assertEquals(run.fake.commits, [1]);
  });
});

// ---- T6: 予算末尾 --------------------------------------------------------------

Deno.test("T6 予算末尾: 残り 1 個の cycle は draft を採らず decode 形の verify 1 行で閉じる", async () => {
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 6 };
  const run = await runSpeculative({ drafter: oracleDrafter(), request });
  await assertMatchesPlain(request, run);

  // 1 cycle 目で 4 個確定（残り 1）→ 2 cycle 目は `k' = 0`。
  assertEquals(run.fake.commits, [4, 1]);
  assertEquals(run.drafter.calls.length, 1, "残り 1 個の cycle で draft run を出した");
  const verifies = verifyCalls(run.fake);
  assertEquals(verifies.map((call) => call.queryLength), [4, 1]);
  assertEquals(verifies.map((call) => call.lastRows), [[0, 1, 2, 3], [0]]);
  assertEquals(verifies.map((call) => call.idsShape), [[1, 4], [1, 1]], "k'=0 は decode 形");
  assertEquals(verifies.map((call) => call.commit), ["deferred", "deferred"]);
  // `max-tokens` の tokens は要求した数そのもの（超えても足りなくてもいけない）。
  assertEquals(withoutSpeculation(run.stop), { reason: "max-tokens", tokens: 6 });
  // `k' = 0` の cycle は draft を採らないので受理数 0 = histogram の添字 0 に入る。
  assertEquals(run.stop.speculation, {
    cycles: 2,
    draftRuns: 1,
    drafted: 3,
    accepted: 3,
    delivered: 5,
    acceptedHistogram: [1, 0, 0, 1],
  });
});

// ---- T7: 中断・例外・break ------------------------------------------------------

Deno.test("T7 中断: draft の前で止めれば run も draft も 1 本増えない", async () => {
  const controller = new AbortController();
  const reason = new Error("draft の前で止めた");
  const opened = await openSpeculative({ drafter: oracleDrafter() });
  const stream = opened.sequence.generate({
    prompt: PROMPT,
    maxNewTokens: 9,
    signal: controller.signal,
  });

  const seen: number[] = [];
  let caught: unknown;
  try {
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      seen.push(event.id);
      controller.abort(reason);
    }
  } catch (error) {
    caught = error;
  }
  assert(caught === reason, `中断が包まれている: ${String(caught)}`);
  assertEquals(seen, [6]);
  assertEquals(await stream.done, {
    reason: "aborted",
    tokens: 1,
    speculation: {
      cycles: 0,
      draftRuns: 0,
      drafted: 0,
      accepted: 0,
      delivered: 0,
      acceptedHistogram: [0, 0, 0, 0],
    },
  });
  assertEquals(opened.drafter.calls.length, 0);
  assertEquals(opened.fake.calls.length, 1, "prefill より後の run が出ている");
  assertEquals(opened.fake.commits, []);
});

Deno.test("T7 中断: draft の後・verify の前で止めれば保留は 1 つも残らない", async () => {
  const controller = new AbortController();
  const reason = new Error("draft と verify の間で止めた");
  const drafter = fakeDrafter({
    draft: (cycle) => {
      controller.abort(reason);
      return chainFrom(cycle.token, K);
    },
  });
  const opened = await openSpeculative({ drafter });
  const stream = opened.sequence.generate({
    prompt: PROMPT,
    maxNewTokens: 9,
    signal: controller.signal,
  });

  let caught: unknown;
  try {
    for await (const _event of stream) { /* 中断まで汲む */ }
  } catch (error) {
    caught = error;
  }
  assert(caught === reason, `中断が包まれている: ${String(caught)}`);
  assertEquals(await stream.done, {
    reason: "aborted",
    tokens: 1,
    speculation: {
      cycles: 0,
      draftRuns: 1,
      drafted: 3,
      accepted: 0,
      delivered: 0,
      acceptedHistogram: [0, 0, 0, 0],
    },
  });
  assertEquals(opened.drafter.calls.length, 1, "draft は 1 本走っている");
  assertEquals(opened.fake.calls.length, 1, "verify が出ている（中断は run の発行直前で効く）");
  assertEquals(opened.fake.pendingCommit(), undefined, "保留が残っている");
  assertEquals(opened.fake.commits, []);
});

Deno.test("T7 中断: verify の後で止めても、その cycle の確定ぶんは配送も commit も済む", async () => {
  const controller = new AbortController();
  const reason = new Error("配送の途中で止めた");
  const opened = await openSpeculative({ drafter: oracleDrafter() });
  const stream = opened.sequence.generate({
    prompt: PROMPT,
    maxNewTokens: 9,
    signal: controller.signal,
  });

  const seen: number[] = [];
  let caught: unknown;
  try {
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      seen.push(event.id);
      // cycle 1 の 1 個目を受け取った時点で中断する（残りは同期区間なので配送され切る）。
      if (seen.length === 2) controller.abort(reason);
    }
  } catch (error) {
    caught = error;
  }
  assert(caught === reason, `中断が包まれている: ${String(caught)}`);
  // 中断は**段の境目**（次の cycle の頭）で効く — 走行中の cycle を途中で捨てない。
  assertEquals(seen, [6, 8, 9, 3, 13]);
  assertEquals(await stream.done, {
    reason: "aborted",
    tokens: 5,
    speculation: {
      cycles: 1,
      draftRuns: 1,
      drafted: 3,
      accepted: 3,
      delivered: 4,
      acceptedHistogram: [0, 0, 0, 1],
    },
  });
  assertEquals(opened.fake.commits, [4], "配送した token ぶんが commit されていない");
  assertEquals(opened.fake.pendingCommit(), undefined);
  assertEquals(opened.fake.calls.length, 2);
  // run 1 本につき 1 通（中断で 2 度目の verify が出ない）。
  assertEquals(runShape(opened.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 3 },
  ]);
});

Deno.test("T7′ 例外復旧: verify の戻りで落ちても保留は残らず、同じ sequence が次を通す", async () => {
  // 故障注入: 1 本目の verify（call 1）だけが頼んだ行数と違う行数を返す。読み口の形検査が
  // 落ちるのは「deferred run の戻り〜配送」の同期区間で、ここが保留を残す唯一の窓である。
  const opened = await openSpeculative({
    drafter: oracleDrafter(),
    session: { successor: SUCCESSOR, logitsRows: 2, logitsRowsAt: 1 },
  });
  await assertRejects(
    () => drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 })),
    Error,
    `verify@1: 'logits' の形 [1,2,${VOCAB}] が [1,4,${VOCAB}] でない`,
  );
  // `commit(0)` で畳む: b は未投入のまま・棄却行は次の run が上書きする。
  assertEquals(opened.fake.commits, [0], "例外の経路が commit(0) を通っていない");
  assertEquals(opened.fake.pendingCommit(), undefined, "保留が残っている（次の run が落ちる）");
  assertEquals(opened.fake.pastLength(), PROMPT.length, "論理長が verify のぶん進んでいる");
  // 出力を読み終える前に落ちた run は観測にも出さない（`onRun` は「読み終えた run」の口）。
  assertEquals(runShape(opened.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
  ]);

  // 同じ sequence で次のターンが通る（frontier 6 は連結され、1 token も落ちない）。
  const next = await drain(opened.sequence.generate({ prompt: [4], maxNewTokens: 2 }));
  assertEquals(opened.fake.calls[2].ids, [6, 4, 0, 0]);
  assertEquals(opened.fake.calls[2].positions, [2, 3, 0, 0]);
  assertEquals(tokenIds(next.events), [SUCCESSOR[4], SUCCESSOR[SUCCESSOR[4]]]);
  assertEquals(withoutSpeculation(next.stop), { reason: "max-tokens", tokens: 2 });
});

Deno.test("T7″ 途中 break: 配送した token までが commit され、その最後が次ターンの先頭になる", async () => {
  const opened = await openSpeculative({ drafter: oracleDrafter() });
  const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 });

  const seen: number[] = [];
  for await (const event of stream) {
    if (event.kind !== "token") continue;
    seen.push(event.id);
    // cycle 1 の 2 個目（通算 3 個目）で打ち切る = `return()` 経由で finally へ入る。
    if (seen.length === 3) break;
  }
  assertEquals(seen, [6, 8, 9]);
  // `delivered` は受理が決まった時点で cycle ぶんを積む（`accepted` と同じ位置）ので、`break` で
  // 配送が途中で閉じたこのターンでは消費者が受け取った数（2）より多い 4 になる。
  assertEquals(await stream.done, {
    reason: "closed",
    tokens: 3,
    speculation: {
      cycles: 1,
      draftRuns: 1,
      drafted: 3,
      accepted: 3,
      delivered: 4,
      acceptedHistogram: [0, 0, 0, 1],
    },
  });
  // 未配送の確定候補（3 / 13）は会話に入らない — 消費者の transcript と会話が一致する。
  assertEquals(opened.fake.commits, [2]);
  assertEquals(opened.fake.pendingCommit(), undefined);
  // 配送が途中で閉じても、走った verify run は 1 通だけ報告される。
  assertEquals(runShape(opened.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 3 },
  ]);

  await drain(opened.sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  const next = opened.fake.calls[opened.fake.calls.length - 1];
  assertEquals(next.ids, [9, 4, 0, 0], "break した位置の frontier が次ターンの先頭に来ていない");
  assertEquals(next.positions, [4, 5, 0, 0]);
});

// ---- T8: 門（sequence 生成時・sampler）-----------------------------------------

Deno.test("T8 門: 投機の指定は sequence 生成時に落ち、開いた面は全部畳まれる", async (t) => {
  /** 生成時に拒まれること + 借り手 → 貸し手の順で畳まれることを 1 本で見る。 */
  const assertRejected = async (
    options: {
      readonly k?: number;
      readonly program?: Partial<GenerationProgramSpec>;
      readonly session?: FakeOptions;
      readonly steps?: number;
    },
    message: string,
  ): Promise<void> => {
    const disposeLog: string[] = [];
    const fake = fakeSession({ successor: SUCCESSOR, disposeLog, ...options.session });
    const drafter = fakeDrafter({
      steps: options.steps ?? K,
      draft: () => chainFrom(1, K),
      disposeLog,
    });
    await assertRejects(
      () =>
        createGenerationSequence({
          session: fake.session,
          program: specProgram(fake, options.program),
          speculative: {
            open: () => Promise.resolve(drafter.face),
            k: options.k,
            policy: "always",
          },
        }),
      Error,
      message,
    );
    assertEquals(drafter.disposals(), 1, "drafter の面が畳まれていない");
    assertEquals(fake.disposals(), 1, "貸し手 context が畳まれていない");
    assertEquals(disposeLog, ["drafter", "context"], "畳む順が 借り手 → 貸し手 でない");
    assertEquals(fake.calls.length, 0);
  };

  await t.step("k + 1 が sliding ring の余裕を超える", async () => {
    await assertRejected(
      { k: K, session: { slidingSlack: K } },
      `speculative.k ${K} は sliding ring の余裕 ${K} に入らない`,
    );
  });

  await t.step("k が drafter の段数を超える", async () => {
    await assertRejected({ k: K + 1 }, `speculative.k ${K + 1} が 1..${K}（drafter の段数）の外`);
  });

  await t.step("k が 1 未満", async () => {
    await assertRejected({ k: 0 }, `speculative.k 0 が 1..${K}（drafter の段数）の外`);
  });

  await t.step("drafter が名乗る段数が 1 以上の整数でない", async () => {
    // `k` を省くと `k = steps` になるので、段数を見ない実装では `k` の検査が素通りする
    // （NaN はどの比較も false・1.5 は `1.5 > 1.5` が false）。段数の門が `k` の門より
    // **前**に立っていることを、文言（段数の側）で縛る。
    await assertRejected({ steps: Number.NaN }, "drafter の段数 NaN が 1 以上の整数でない");
    await assertRejected({ steps: 0 }, "drafter の段数 0 が 1 以上の整数でない");
    await assertRejected({ steps: 1.5 }, "drafter の段数 1.5 が 1 以上の整数でない");
  });

  await t.step("k + 1 行が chunkLength を超える", async () => {
    await assertRejected(
      { k: K, program: { chunkLength: K, chunkBuckets: [] } },
      `speculative.k ${K} の verify ${K + 1} 行が chunkLength ${K} を超える`,
    );
  });

  await t.step("k + 1 行を載せるバケットが無い", async () => {
    await assertRejected(
      { k: K, program: { chunkBuckets: [2] } },
      `verify ${K + 1} 行を載せるバケットが無い（chunkBuckets [2]）`,
    );
  });

  await t.step("sliding スロットが無い context は余裕の門を通さない", async () => {
    // `slidingSlack` が `undefined`（sliding 無し）なら棄却行が潰す過去 KV も無い。
    const run = await runSpeculative({
      drafter: oracleDrafter(),
      request: { prompt: PROMPT, maxNewTokens: 5 },
      session: { successor: SUCCESSOR, slidingSlack: undefined },
    });
    assertEquals(run.fake.commits, [4]);
  });

  await t.step(
    'policy "always" にゲートのノブを渡すと open の前に落ち、貸し手は畳まれる',
    async () => {
      const fake = fakeSession({ successor: SUCCESSOR });
      let caught: unknown;
      try {
        await createGenerationSequence({
          session: fake.session,
          program: specProgram(fake),
          speculative: {
            open: () => Promise.reject(new Error("open まで届いてはいけない")),
            policy: "always",
            gate: { burstAbort: 10 },
          },
        });
      } catch (error) {
        caught = error;
      }
      assert(caught instanceof Error, `落ちていない: ${String(caught)}`);
      assertStringIncludes(caught.message, '"always" にはゲートが居ない');
      assertEquals(fake.disposals(), 1, "門で落ちた借り手の巻き添えで貸し手が漏れている");
    },
  );

  await t.step("open が投げたら貸し手 context を畳んでから素通しする", async () => {
    const fake = fakeSession({ successor: SUCCESSOR });
    const reason = new Error("drafter が開けない");
    let caught: unknown;
    try {
      await createGenerationSequence({
        session: fake.session,
        program: specProgram(fake),
        speculative: { open: () => Promise.reject(reason), policy: "always" },
      });
    } catch (error) {
      caught = error;
    }
    assert(caught === reason, `open の失敗が包まれている: ${String(caught)}`);
    assertEquals(fake.disposals(), 1, "開けなかった借り手の巻き添えで貸し手が漏れている");
  });
});

Deno.test("T8 門: open は context 確保の直後に、貸し手 context そのものを受けて呼ばれる", async () => {
  const fake = fakeSession({ successor: SUCCESSOR });
  const drafter = oracleDrafter();
  const seen: unknown[] = [];
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: specProgram(fake),
    speculative: {
      open: (context) => {
        seen.push(context);
        // 借り手を開くのは context 確保の直後（run は 1 本も出ていない）。
        assertEquals(fake.specs.length, 1);
        assertEquals(fake.calls.length, 0);
        return Promise.resolve(drafter.face);
      },
      policy: "always",
    },
  });
  assertEquals(seen.length, 1);
  assert(seen[0] === fake.context, "open が貸し手 context 以外を受け取っている");

  // 借り手 → 貸し手の順で畳む（`dispose` の鎖）。
  const disposeLog: string[] = [];
  const logged = fakeSession({ successor: SUCCESSOR, disposeLog });
  const loggedDrafter = fakeDrafter({ draft: () => chainFrom(1, K), disposeLog });
  const disposable = await createGenerationSequence({
    session: logged.session,
    program: specProgram(logged),
    speculative: { open: () => Promise.resolve(loggedDrafter.face), policy: "always" },
  });
  await disposable.dispose();
  assertEquals(disposeLog, ["drafter", "context"]);
  await sequence.dispose();
});

Deno.test("T8 門: speculative を渡さない sequence の停止には speculation 欄が無い", async () => {
  const plain = await runPlain({ prompt: PROMPT, maxNewTokens: 3 });
  assertEquals(Object.hasOwn(plain.stop, "speculation"), false);
  assertEquals(withoutSpeculation(plain.stop), { reason: "max-tokens", tokens: 3 });
});

// ---- T9: onRun hook ------------------------------------------------------------

Deno.test("T9 onRun: 投機は prefill → (draft, verify) × cycle（cycle は 1 始まり）", async () => {
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request: { prompt: PROMPT, maxNewTokens: 9 },
  });
  assertEquals(runShape(run.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 3 },
    { kind: "draft", cycle: 2 },
    { kind: "verify", cycle: 2, rows: 4, accepted: 3 },
  ]);
});

Deno.test("T9 onRun: k' = 0 の cycle は draft を名乗らず verify 1 行だけを報告する", async () => {
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request: { prompt: PROMPT, maxNewTokens: 6 },
  });
  assertEquals(runShape(run.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 3 },
    { kind: "verify", cycle: 2, rows: 1, accepted: 0 },
  ]);
});

Deno.test("T9 onRun: 部分受理の accepted と、chunk が割れた prefill の進捗", async () => {
  const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  const run = await runSpeculative({
    drafter: partialDrafter(1),
    request: { prompt, maxNewTokens: 3 },
  });
  assertEquals(runShape(run.phases), [
    { kind: "prefill", chunk: 1, chunks: 2 },
    { kind: "prefill", chunk: 2, chunks: 2 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 2, accepted: 1 },
  ]);
});

Deno.test("T9 onRun: 非投機は prefill → decode × (tokens − 1)（step は 1 始まり）", async () => {
  const plain = await runPlain({ prompt: PROMPT, maxNewTokens: 3 });
  assertEquals(runShape(plain.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "decode", step: 1 },
    { kind: "decode", step: 2 },
  ]);
});

Deno.test("T9 onRun: run 1 本につき 1 通 — 停止で閉じた cycle の verify も報告される", async () => {
  // `onRun` は run の数を数える口である（診断・tok/s の分母）。停止 token で閉じた cycle の
  // verify を落とすと、その 1 本が観測から消える（非投機の decode は停止した step も報告する）。
  const stopTokens = [9];
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request: { prompt: PROMPT, maxNewTokens: 9 },
    program: { stopTokens },
  });
  assertEquals(run.fake.calls.length, 2, "走った run は prefill 1 + verify 1");
  // 受理は停止 token（d₂ = 9）で打ち切られるので `accepted` は 2（配送した draft の数）。
  assertEquals(runShape(run.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 2 },
  ]);

  // 対（非投機は停止した decode run も 1 通報告する）。
  const plain = await runPlain({ prompt: PROMPT, maxNewTokens: 9 }, { stopTokens });
  assertEquals(runShape(plain.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "decode", step: 1 },
    { kind: "decode", step: 2 },
  ]);
});

Deno.test("T9 onRun: hook を渡さない sequence も同じ列を出す（観測席は挙動を変えない）", async () => {
  const fake = fakeSession({ successor: SUCCESSOR });
  const drafter = oracleDrafter();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: specProgram(fake),
    speculative: { open: () => Promise.resolve(drafter.face), policy: "always" },
  });
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
  const run = await drain(sequence.generate(request));
  await assertMatchesPlain(request, run);
  assertEquals(fake.commits, [4, 4]);
});

// ---- T10: speculation の勘定 ---------------------------------------------------

Deno.test("T10 勘定: histogram は長さ k+1・添字が受理数・合計が cycles / accepted と整合する", async (t) => {
  const tally = async (k: number, drafter: FakeDrafter, maxNewTokens: number) => {
    const run = await runSpeculative({
      drafter,
      k,
      request: { prompt: PROMPT, maxNewTokens },
    });
    const speculation = run.stop.speculation;
    assert(speculation !== undefined, "投機のターンに speculation 欄が載っていない");
    return speculation;
  };

  await t.step("k = 3（drafter の段数と同じ）", async () => {
    const speculation = await tally(K, partialDrafter(2), 7);
    assertEquals(speculation.acceptedHistogram.length, K + 1);
    assertEquals(speculation.acceptedHistogram, [0, 0, 2, 0]);
    assertEquals(
      speculation.acceptedHistogram.reduce((sum, count) => sum + count, 0),
      speculation.cycles,
      "histogram の総数が cycles と合わない",
    );
    assertEquals(
      speculation.acceptedHistogram.reduce((sum, count, accepted) => sum + count * accepted, 0),
      speculation.accepted,
      "histogram の重み付き和が accepted と合わない",
    );
    // 停止 token で打ち切った cycle が 1 本も無いターンでは `delivered = accepted + cycles`
    // （どの cycle も `1 + a` 個を確定させる）。停止で打ち切った cycle があるとここが割れる。
    assertEquals(speculation.delivered, speculation.accepted + speculation.cycles);
  });

  await t.step("k = 2（段数より短く取る）", async () => {
    const speculation = await tally(2, oracleDrafter(), 7);
    // 1 cycle が 3 個ずつ確定 → 2 cycle（1 + 3 + 3 = 7）。
    assertEquals(speculation.acceptedHistogram.length, 3);
    assertEquals(speculation, {
      cycles: 2,
      draftRuns: 2,
      drafted: 4,
      accepted: 4,
      delivered: 6,
      acceptedHistogram: [0, 0, 2],
    });
  });

  await t.step("k = 1（draft 1 本 = verify 2 行）", async () => {
    const speculation = await tally(1, wrongDrafter(), 4);
    assertEquals(speculation, {
      cycles: 3,
      draftRuns: 2,
      drafted: 2,
      accepted: 0,
      delivered: 3,
      acceptedHistogram: [3, 0],
    });
  });
});

// ---- T11: hidden の写し --------------------------------------------------------

Deno.test("T11 hidden: 次 cycle の drafter へ渡るのは b′ を出した行（行 a）の hidden", async () => {
  // 受理 1 本の cycle が続くので、行 0 でも行 k でもない**行 1** が正解になる配置。
  const run = await runSpeculative({
    drafter: partialDrafter(1),
    request: { prompt: PROMPT, maxNewTokens: 7 },
  });
  assertEquals(run.drafter.calls.length, 3);

  // cycle 1: prefill（run 0）の最終有効行（添字 1）の hidden。
  assertEquals(run.drafter.calls[0], {
    token: 6,
    position: 2,
    hidden: [hiddenMark(0, 1), 2, 0, 0],
  });
  // cycle 2: verify（run 1）の行 1 = b′ 9 を出した行（行 0 なら 101・行 3 なら 104 になる）。
  assertEquals(run.drafter.calls[1], {
    token: 9,
    position: 4,
    hidden: [hiddenMark(1, 1), 8, 0, 0],
  });
  // cycle 3: verify（run 2）の行 1。
  assertEquals(run.drafter.calls[2], {
    token: 13,
    position: 6,
    hidden: [hiddenMark(2, 1), 3, 0, 0],
  });
});

Deno.test("T11 hidden: 最初の cycle の hidden は prefill の**最終 chunk** の行 0", async () => {
  // chunk が 2 本に割れる prompt（9 token / chunkLength 8）— 先頭 chunk の hidden を掴む実装は
  // ここで割れる（末尾 chunk は有効行 1 本 = decode 形なので行 0）。
  const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request: { prompt, maxNewTokens: 3 },
  });
  assertEquals(run.fake.calls.length, 3, "prefill 2 本 + verify 1 本");
  assertEquals(run.drafter.calls, [{
    token: SUCCESSOR[9],
    position: prompt.length,
    hidden: [hiddenMark(1, 0), 9, 0, 0],
  }]);
});

// ---- T12 / T13: draft の値域門と last_row の pad ---------------------------------

Deno.test("T12 値域門: 語彙外・本数不足の draft は verify を出す前に fail loudly", async (t) => {
  const assertDraftRejected = async (
    draft: () => readonly number[],
    message: string,
  ): Promise<void> => {
    const opened = await openSpeculative({ drafter: fakeDrafter({ draft }) });
    await assertRejects(
      () => drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 })),
      Error,
      message,
    );
    // draft は verify の `input_ids` と history に入る値なので、prompt と同じ位置で落とす。
    assertEquals(opened.fake.calls.length, 1, "値域外の draft で verify が発行された");
    assertEquals(opened.fake.commits, []);
    assertEquals(opened.fake.pendingCommit(), undefined);
    // draft run 自体は**完了している**（借り手は draft を返し切った）ので 1 通名乗る。落ちるのは
    // 戻り値の値域検査で、これは verify の形検査（T7′ — 読み終える前に落ちるので名乗らない）とは
    // 別の位置である。
    assertEquals(runShape(opened.phases), [
      { kind: "prefill", chunk: 1, chunks: 1 },
      { kind: "draft", cycle: 1 },
    ]);
  };

  await t.step("語彙の上を超える", async () => {
    await assertDraftRejected(() => [VOCAB, 8, 9], `draft[0] ${VOCAB} が語彙 0..${VOCAB - 1} の外`);
  });

  await t.step("負の id", async () => {
    await assertDraftRejected(() => [8, -1, 9], `draft[1] -1 が語彙 0..${VOCAB - 1} の外`);
  });

  await t.step("本数が足りない", async () => {
    await assertDraftRejected(() => [8, 9], "draft が 2 本しか無い（3 本要る）");
  });
});

Deno.test("T13 last_row: k' が縮んでも R は k+1 のまま（末尾添字で pad）", async (t) => {
  const verifyOnce = async (maxNewTokens: number) => {
    const run = await runSpeculative({
      drafter: oracleDrafter(),
      request: { prompt: PROMPT, maxNewTokens },
    });
    await assertMatchesPlain({ prompt: PROMPT, maxNewTokens }, run);
    assertEquals(verifyCalls(run.fake).length, 1);
    return { call: verifyCalls(run.fake)[0], run };
  };

  await t.step("k' = 1（R は 4 本のまま・末尾を pad）", async () => {
    const { call, run } = await verifyOnce(3);
    assertEquals(call.queryLength, 2);
    assertEquals(call.lastRows, [0, 1, 1, 1]);
    assertEquals(call.lastRowShape, [K + 1]);
    assertEquals(call.idsShape, [1, 4]);
    assertEquals(call.ids, [6, 8, 0, 0]);
    assertEquals(call.positions, [2, 3, 0, 0]);
    assertEquals(run.fake.commits, [2]);
  });

  await t.step("k' = 2（pad は 1 本だけ）", async () => {
    const { call, run } = await verifyOnce(4);
    assertEquals(call.queryLength, 3);
    assertEquals(call.lastRows, [0, 1, 2, 2]);
    assertEquals(call.ids, [6, 8, 9, 0]);
    assertEquals(call.positions, [2, 3, 4, 0]);
    assertEquals(run.fake.commits, [3]);
  });

  await t.step("k' = 0（pad せず decode 形の [0]）", async () => {
    const { call, run } = await verifyOnce(2);
    assertEquals(call.queryLength, 1);
    assertEquals(call.lastRows, [0]);
    assertEquals(call.lastRowShape, [1]);
    assertEquals(call.idsShape, [1, 1]);
    assertEquals(call.ids, [6]);
    assertEquals(call.positions, [2]);
    assertEquals(run.fake.commits, [1]);
  });
});

// ---- T14: 走行中の `used` -------------------------------------------------------

/**
 * 配送の**直後**に `used` を読み、`position + 1` であることを見る（返すのは配送した位置の列）。
 *
 * yield から戻った同期区間で読むので、投機では**保留中の verify が残っている窓**を観測できる。
 * ここが崩れると `used` を見て次ターンの予算を決める呼び手（`GenerationSequence.used` の doc の
 * 式）が、cycle の途中では最大 k だけ甘い上限を通してしまう。
 */
const drainAssertingUsed = async (
  sequence: GenerationSequence,
  stream: GenerationStream,
): Promise<number[]> => {
  const positions: number[] = [];
  for await (const event of stream) {
    if (event.kind !== "token") continue;
    positions.push(event.position);
    assertEquals(
      sequence.used,
      event.position + 1,
      `位置 ${event.position} の token を配送した直後の used`,
    );
  }
  return positions;
};

Deno.test("T14 used: 配送の直後は常に「その token の position + 1」", async (t) => {
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 7 };
  /** 配送される位置の列（prompt 2 token の直後から 7 個）。 */
  const expected = [2, 3, 4, 5, 6, 7, 8];

  await t.step("投機（部分受理 — cycle の途中で読む）", async () => {
    const opened = await openSpeculative({ drafter: partialDrafter(1) });
    assertEquals(
      await drainAssertingUsed(opened.sequence, opened.sequence.generate(request)),
      expected,
    );
    // 決着後も同じ式（frontier 1 個は未 commit のまま = 次ターンの先頭へ連結される）。
    assertEquals(opened.sequence.used, 9);
    assertEquals(opened.fake.pendingCommit(), undefined);
  });

  await t.step("非投機（回帰の対）", async () => {
    const opened = await openPlain();
    assertEquals(
      await drainAssertingUsed(opened.sequence, opened.sequence.generate(request)),
      expected,
    );
    assertEquals(opened.sequence.used, 9);
  });

  await t.step("cycle の途中で break した後も、配送した最後の position + 1", async () => {
    const opened = await openSpeculative({ drafter: oracleDrafter() });
    const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 });
    let last = -1;
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      last = event.position;
      assertEquals(opened.sequence.used, last + 1, `配送直後（位置 ${last}）`);
      // cycle 1 の 2 個目（= 保留がまだ残っている窓）で打ち切る。
      if (last === 4) break;
    }
    assertEquals((await stream.done).reason, "closed");
    // commit は「未 commit 行 → pastLength」の付け替えなので、`used` は commit を跨いで動かない。
    assertEquals(opened.fake.commits, [2]);
    assertEquals(opened.sequence.used, 5, "break 後の used が配送した最後の position + 1 でない");
  });

  await t.step(
    "配送中に消費者が例外を投げても（finally の commit）、最後の position + 1",
    async () => {
      const opened = await openSpeculative({ drafter: oracleDrafter() });
      const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 });
      let last = -1;
      await assertRejects(
        async () => {
          for await (const event of stream) {
            if (event.kind !== "token") continue;
            last = event.position;
            // break と同じ窓（cycle 1 の 2 個目 = 保留が残っている）で消費者側が落ちる。
            if (last === 4) throw new Error("consumer");
          }
        },
        Error,
        "consumer",
      );
      // 消費者の例外は generator には入らない（`return()` 経由 = closed）。frontier までの commit と
      // `frontierRows` の 0 戻しは同じ finally を通る。
      assertEquals((await stream.done).reason, "closed");
      assertEquals(opened.fake.commits, [2]);
      assertEquals(opened.sequence.used, 5, "例外後の used が配送した最後の position + 1 でない");
    },
  );
});

// ---- T15: 抽選が落ちた decode の後始末 -------------------------------------------

/**
 * 全 run の有効行を（位置, token）へ畳む = **KV に入った並び**（非投機の走行に限る — deferred な
 * verify は棄却行も物理 ring へ書くので、投機では commit 行数と併せて読む必要がある）。
 *
 * 同じ token が 2 つの位置に居る形は token 列の比較では見えない（次ターンの入力を作るのは
 * `pendingToken` で、消費者が受け取った列には現れない）。位置つきで畳むのが唯一の検出線である。
 */
const placedRows = (fake: FakeSession): number[][] =>
  fake.calls.flatMap((call) =>
    call.ids.slice(0, call.queryLength).map((id, row) => [call.positions[row], id])
  );

Deno.test("T15 抽選の失敗: run が通った後に抽選が落ちても、旧 frontier は次ターンへ再投入されない", async () => {
  // 故障注入: decode run（call 1）の logits に NaN を混ぜる。run 自体は**成功**しているので
  // frontier b（= 6）は位置 2 の KV に入っており、`pendingToken` に残したまま次ターンへ行くと
  // 同じ token が位置 3 にも入る（例外を出さない沈黙劣化 — 会話が 1 token 太る）。
  const opened = await openPlain({}, { nanAt: 1 });
  await assertRejects(
    () => drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 5 })),
    Error,
    // 落ちるのは抽選（`assertNoNaN`）— run の失敗ではない（汚染した id は fake の実装事情なので見ない）。
    "が NaN（非有限）",
  );
  assertEquals(
    opened.fake.pastLength(),
    3,
    "抽選が落ちた decode run は成功している（論理長は進む）",
  );

  // 同じ sequence で次のターン（frontier は既に KV に居るので prompt はそのまま流れる）。
  const next = await drain(opened.sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  assertEquals(tokenIds(next.events), [SUCCESSOR[4]]);
  assertEquals(placedRows(opened.fake), [[0, 1], [1, 2], [2, 6], [3, 4]]);
});

// ---- T16: 自己採算ゲート（`policy: "auto"`・偽時計）-------------------------------

/** 時計を読む回数に依存せず、fake target run の完了時に cycle の費用を加える。 */
const fakeClock = (
  drafter: FakeDrafter,
  wallOf: (cycle: number, drafted: boolean) => number,
): { readonly now: () => number; readonly afterRun: () => void } => {
  let elapsed = 0;
  let cycle = 0;
  let draftsAtLastRun = 0;
  return {
    now: (): number => elapsed,
    afterRun: (): void => {
      elapsed += wallOf(cycle, drafter.calls.length > draftsAtLastRun);
      draftsAtLastRun = drafter.calls.length;
      cycle += 1;
    },
  };
};

/**
 * ゲートを 2 ブロックぶん待たずに倒すノブ（既定は 16 cycle × 2 本連続）。
 *
 * ここで見るのは**配線**（落ちた step の勘定・hidden の繋がり・観測に混ぜない cycle）で、
 * 「何 cycle 測ってから倒すか」の判断そのものは `generation_gate_test.ts` が持つ。
 */
const FAST_GATE: SpeculationGateOptions = { window: 2, confirm: 1 };

Deno.test("T16 ゲート: 投機が負ける壁では decode 形へ落ち、それでも列は非投機と同一", async () => {
  // 投機 cycle 100ms で 4 個確定（25ms/token）・plain step 10ms/token = 投機が 2.5 倍遅い壁。
  const drafter = oracleDrafter();
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 20 };
  const opened = await openSpeculative({
    drafter,
    policy: "auto",
    gate: FAST_GATE,
    clock: fakeClock(drafter, (_cycle, drafted) => drafted ? 100 : 10),
  });
  const run = { ...opened, ...await drain(opened.sequence.generate(request)) };

  // 契約の芯は変わらない（ゲートは速さだけを変え、出力は変えない）。
  await assertMatchesPlain(request, run);
  const always = await runSpeculative({ drafter: oracleDrafter(), request });
  assertEquals(tokenIds(run.events), tokenIds(always.events), "ゲート付きと always の列が違う");

  // 落ちた step は投機の勘定に入らず、`plainSteps` が数える（cycle 番号も消費しない）。
  // 予算末尾の強制 plain（`k' = 0`）は従来どおり verify として数える = `cycles` は 4 で、
  // その 1 本が `acceptedHistogram[0]` に入る。
  assertEquals(run.stop.speculation, {
    cycles: 4,
    draftRuns: 3,
    drafted: 9,
    accepted: 9,
    delivered: 13,
    acceptedHistogram: [1, 0, 0, 3],
    plainSteps: 6,
    switches: 1,
  });
  // 配送は 4 →（`W1` の初回サンプル）1 → 4 → 4（ここでブロック 2 本目が満ちて倒れる）→
  // plain 5 本 → 予算末尾の 1（= 20 token）。2 本目が plain なのは「`W1` の初回サンプルは
  // 2 回目の決定」だからで、ゲートが倒れるのは 2 本目の投機 cycle を**観測した**時である。
  assertEquals(run.fake.commits, [4, 1, 4, 4, 1, 1, 1, 1, 1, 1]);
  // ゲートの plain step は `decode` を名乗る（公開型に枝を足さない）。番号は 1 始まりの通し。
  assertEquals(
    runShape(run.phases).filter((phase) => phase.kind === "decode"),
    Array.from({ length: 6 }, (_unused, index) => ({ kind: "decode", step: index + 1 })),
  );
  // verify の cycle 番号は 1..4（plain step が番号を飛ばさない）。
  assertEquals(
    run.phases.filter((phase) => phase.kind === "verify").map((phase) => phase.cycle),
    [1, 2, 3, 4],
  );
});

Deno.test("T16 ゲート: plain step を挟んでも drafter 入力（hidden と frontier）が繋がる", async () => {
  // ゲートの plain step は verify 形の 1 行（deferred）なので `hidden` が読める。非投機の decode
  // 経路（logits しか読まない）へ分岐すると、戻った cycle の draft が古い hidden を食う。
  // 1 ターン（20 token）の中で「倒れる → plain 2 本 → 探索」まで回すため、バーストは 1 cycle・
  // plain 側の間隔は 2 step に縮める（バーストを何 cycle 測るかは generation_gate_test.ts）。
  const drafter = oracleDrafter();
  const opened = await openSpeculative({
    drafter,
    policy: "auto",
    gate: { ...FAST_GATE, burst: 1, exploreBase: 2 },
    clock: fakeClock(drafter, (_cycle, drafted) => drafted ? 100 : 10),
  });
  await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 20 }));

  // 最後の draft は plain 側の探索バーストのもので、その直前は plain step が 2 本続いている
  // （run 添字 6 と 7 — 添字 0 は prefill）。受け取るのは run 7 の行 0 の hidden と、run 7 が
  // 配送した frontier（run 7 の入力 token 3 → `SUCCESSOR[3]` = 13）である。
  const last = opened.drafter.calls[opened.drafter.calls.length - 1];
  assertEquals(opened.drafter.calls.length, 4);
  assertEquals(last, { token: SUCCESSOR[3], position: 18, hidden: [hiddenMark(7, 0), 3, 0, 0] });
});

Deno.test("T16 ゲート: ターン最初の cycle と予算末尾の強制 plain は観測に混ぜない", async (t) => {
  // 最初の cycle は PLE gather の cold miss を含み、強制 plain は呼び手の予算で形が決まる —
  // どちらも「投機が遅い」証拠にならない。故障注入で確かめる: その 2 種類にだけ 10,000ms を
  // 置き、ゲートが 1 度も倒れないこと（混ぜる実装は即座に plain へ落ちる）。
  const turn1: GenerationRequest = { prompt: PROMPT, maxNewTokens: 6 };
  const turn2: GenerationRequest = { prompt: [4], maxNewTokens: 20 };
  /** ターン 1 は cycle 2 本（最初の cycle + 予算末尾の強制 plain）・ターン 2 の頭が 3 本目。 */
  const excluded = 3;
  /** ターン 2 で最初に**観測に入る投機 cycle**（頭の 2 本は `W1` を採る plain step）。 */
  const firstMeasuredCycle = 4;

  const play = async (slow: (cycle: number) => boolean) => {
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      clock: fakeClock(
        drafter,
        (cycle, drafted) => slow(cycle) ? 10_000 : (drafted ? 20 : 10),
      ),
    });
    await drain(opened.sequence.generate(turn1));
    const second = await drain(opened.sequence.generate(turn2));
    return second.stop.speculation;
  };

  await t.step("混ぜない: 2 ターン目も投機のまま（倒れない）", async () => {
    // ターン 2 の頭 2 本が plain なのは `W1` がまだ未観測だからである（1 本目は混ぜない cycle
    // なので観測を返さず、2 本目でようやく `W1` を採る）。定常の測り直しは 16 観測ごとなので、
    // 20 token のターンには入らない = plain step は頭の 2 本だけである。最後の 1 本は予算末尾の
    // 強制 plain（`k' = 0`）で、verify として数えつつ `acceptedHistogram[0]` に入る。
    assertEquals(await play((cycle) => cycle < excluded), {
      cycles: 5,
      draftRuns: 4,
      drafted: 12,
      accepted: 12,
      delivered: 17,
      acceptedHistogram: [1, 0, 0, 4],
      plainSteps: 2,
      switches: 0,
    });
  });

  await t.step("対（同じ壁を観測に入る投機 cycle へ置くと落ちる）", async () => {
    // 10,000ms を 1 本だけ、ターン 2 で最初に観測に入る**投機**の cycle へ置くと、ゲートは
    // その観測の直後に plain へ倒れる。上の緑が「そもそも倒れない設定」ではないことの対。
    // plain step（`W1` を採る cycle）へ置いてはいけない — `W1` が 10,000ms になると比が小さく
    // なり、投機が有利に見えて倒れない = 対にならない。
    const speculation = await play((cycle) => cycle === firstMeasuredCycle);
    assertEquals(speculation?.switches, 1, "観測に入る遅い cycle でも倒れない");
    assert(
      (speculation?.plainSteps ?? 0) > 3,
      `倒れた後も plain へ落ちていない: ${JSON.stringify(speculation)}`,
    );
  });

  await t.step("skip した cycle はブロックに入らないが、探索の周期は進む", async () => {
    // 混ぜない cycle でも GPU の仕事は 1 本走っているので、探索の周期はそのぶん進める
    // （ゲートの `skip()`）。1 ターンで見ると: 頭の cycle は観測に入らないまま周期を 1 進め、
    // `W1` の初回サンプル（plain 1 手）は**2 本目**に来て、以後の探索は `exploreBase` ごとである。
    // 周期が止まる実装だと、この 2 つがどちらも 1 本ずつ後ろへずれる。
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      // ブロックが 2 cycle なので、頭の 10,000ms が 1 本でも混ざれば最初のブロックで倒れる。
      // 既定の測り直し間隔（16 観測）は 1 ターンの予算に入らないので、周期そのものは 4 に縮めて
      // 見る（既定値の側は `generation_gate_test.ts` の T5 が持つ）。
      gate: { ...FAST_GATE, exploreBase: 4 },
      clock: fakeClock(drafter, (cycle, drafted) => cycle === 0 ? 10_000 : (drafted ? 20 : 10)),
    });
    const run = {
      ...opened,
      ...await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 30 })),
    };
    assertEquals(
      run.phases.flatMap((phase) =>
        phase.kind === "verify" || phase.kind === "decode" ? [phase.kind] : []
      ),
      [
        "verify",
        "decode",
        "verify",
        "decode",
        "verify",
        "verify",
        "verify",
        "decode",
        "verify",
        "verify",
      ],
      "探索の周期が混ぜない cycle のぶん進んでいない",
    );
    assertEquals(run.stop.speculation?.switches, 0, "混ぜない cycle の壁がブロックに入っている");
  });
});

// ---- T17: 観測席の壁と局面（`wallMs` / `delivered` / `gate`）------------------------

/** 生成相の観測（壁と局面を名乗る 2 種別 — prefill と draft は名乗らない）。 */
type StepPhase = Extract<GenerationRunPhase, { kind: "decode" | "verify" }>;

const stepsOf = (phases: readonly GenerationRunPhase[]): readonly StepPhase[] =>
  phases.flatMap((phase) => phase.kind === "decode" || phase.kind === "verify" ? [phase] : []);

/** 期待値 1 通ぶんのゲートの状態。 */
const gateOf = (
  mode: "speculate" | "plain",
  switches: number,
  measured: boolean,
): GenerationGateTrace => ({ mode, switches, measured });

Deno.test("T17 観測席の壁: 投機の cycle と plain step の壁が載り、消費者の遅れを含まない", async (t) => {
  await t.step("T16 と同じ壁を流すと、run ごとの wallMs が偽時計の刻みと一致する", async () => {
    // T16 と**同じ設定**（投機 cycle 100ms / plain step 10ms・ブロックは 2 cycle）。あちらが
    // 「勘定と決定列」を見る席で、ここは同じ走行の**壁と局面**を見る席である。
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      clock: fakeClock(drafter, (_cycle, drafted) => drafted ? 100 : 10),
    });
    const run = {
      ...opened,
      ...await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 20 })),
    };
    const steps = stepsOf(run.phases);

    // 配送は 4 →（`W1` の初回サンプル）1 → 4 → 4 → plain 5 本 → 予算末尾の 1（T16 の commits）。
    // 壁はその形のとおり「draft を採った cycle だけ 100ms」である。
    assertEquals(steps.map((phase) => phase.kind), [
      "verify",
      "decode",
      "verify",
      "verify",
      "decode",
      "decode",
      "decode",
      "decode",
      "decode",
      "verify",
    ]);
    assertEquals(
      steps.map((phase) => phase.wallMs),
      [100, 10, 100, 100, 10, 10, 10, 10, 10, 10],
      "run の壁が偽時計の刻みと一致しない",
    );
  });

  await t.step("ゲートの状態は観測の**後**の値（先頭 cycle は measured が立たない）", async () => {
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      clock: fakeClock(drafter, (_cycle, drafted) => drafted ? 100 : 10),
    });
    const run = {
      ...opened,
      ...await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 20 })),
    };

    // 3 本目の投機 cycle（run 添字 3）でブロック 2 本目が満ちて倒れる = その観測の**結果**が
    // `mode: "plain"` / `switches: 1` である。観測の前に読む実装だと 1 本ぶん後ろへずれる。
    assertEquals(stepsOf(run.phases).map((phase) => phase.gate), [
      // ターン最初の cycle は壁の観測に混ぜない（`skip()` — 混ぜない 2 種類の 1 つ）。
      gateOf("speculate", 0, false),
      // `W1` の初回サンプル（2 回目の決定で 1 step だけ plain を挟む）。
      gateOf("speculate", 0, true),
      gateOf("speculate", 0, true),
      gateOf("plain", 1, true),
      gateOf("plain", 1, true),
      gateOf("plain", 1, true),
      gateOf("plain", 1, true),
      gateOf("plain", 1, true),
      gateOf("plain", 1, true),
      // 予算末尾の強制 plain も混ぜない（呼び手の予算で形が決まった cycle）。
      gateOf("plain", 1, false),
    ]);
    assertEquals(run.stop.speculation?.switches, 1, "T16 と同じ位置で倒れていない");
  });

  await t.step("消費者が配送の合間に時計を進めても壁は動かない（フォールト注入）", async () => {
    // 壁を「配送の後」で採る実装だと、遅い消費者ほど投機が遅く見える（= ゲートが投機を切る）。
    // 偽時計を cycle の**中**（draft の中）と配送の**合間**（消費者）の 2 箇所で進め、run の壁に
    // 前者だけが乗ることを見る。ゲートは要らない（`policy: "always"` でも壁は載る）。
    const cycleMs = 100;
    const deliveryMs = 1000;
    let elapsed = 0;
    const drafter = fakeDrafter({
      draft: (cycle) => {
        elapsed += cycleMs;
        return chainFrom(cycle.token, K);
      },
    });
    const opened = await openSpeculative({ drafter, now: (): number => elapsed });
    const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 });
    let delivered = 0;
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      delivered += 1;
      elapsed += deliveryMs;
    }
    assertEquals((await stream.done).reason, "max-tokens");

    // 故障注入が効いていること（消費者のぶんが時計の大半である）。
    assertEquals(delivered, 9);
    assertEquals(elapsed, 2 * cycleMs + 9 * deliveryMs);
    assertEquals(
      stepsOf(opened.phases).map((phase) => phase.wallMs),
      [cycleMs, cycleMs],
      "run の壁に配送（消費者）の時間が入っている",
    );
    // `policy: "always"` はゲートを作らないので、局面の欄も生えない。
    assertEquals(
      stepsOf(opened.phases).map((phase) => Object.hasOwn(phase, "gate")),
      [false, false],
      "ゲート無しのターンに gate の欄がある",
    );
  });

  await t.step("非投機の decode も壁を載せ、ゲートの欄は持たない", async () => {
    const plain = await runPlain({ prompt: PROMPT, maxNewTokens: 3 });
    const steps = stepsOf(plain.phases);
    assertEquals(steps.map((phase) => phase.kind), ["decode", "decode"]);
    for (const phase of steps) {
      const wallMs = phase.wallMs;
      // 非投機の sequence は偽時計を差せない（`now` は投機の指定の欄）ので、実値ではなく
      // 「載っていること」だけを見る。刻みの一致は上の投機の席が見る。
      assert(
        wallMs !== undefined && Number.isFinite(wallMs) && wallMs >= 0,
        `非投機の decode に壁が載っていない: ${JSON.stringify(phase)}`,
      );
      assertEquals(Object.hasOwn(phase, "gate"), false, "非投機の観測にゲートの欄がある");
    }
  });

  await t.step(
    "delivered はその cycle が確定させた token 数（accepted + 1 から作らない）",
    async () => {
      // 停止 token で列挙を打ち切った cycle は `accepted + 1` 個を確定させない（T4 と同じ配置 —
      // 連鎖 6 → 8 → 9 で 9 が停止 token）。導出している実装はここで 1 だけ多く名乗る。
      const stopTokens = [9];
      const run = await runSpeculative({
        drafter: oracleDrafter(),
        request: { prompt: PROMPT, maxNewTokens: 9 },
        program: { stopTokens },
      });
      const verifies = stepsOf(run.phases).flatMap((phase) =>
        phase.kind === "verify" ? [phase] : []
      );
      assertEquals(verifies.map((phase) => [phase.accepted, phase.delivered]), [[2, 2]]);
      assertEquals(
        verifies.reduce((sum, phase) => sum + (phase.delivered ?? 0), 0),
        run.stop.speculation?.delivered,
        "観測の delivered の合計が勘定の delivered と合わない",
      );
    },
  );

  await t.step("delivered の合計は勘定と一致する（受理が散るターン）", async () => {
    // 部分受理（a = 1）の cycle が続くターン: 1 cycle 2 個ずつ = 観測の合計も勘定と並ぶ。
    const run = await runSpeculative({
      drafter: partialDrafter(1),
      request: { prompt: PROMPT, maxNewTokens: 7 },
    });
    const delivered = stepsOf(run.phases).flatMap((phase) =>
      phase.kind === "verify" ? [phase.delivered] : []
    );
    assertEquals(delivered, [2, 2, 2]);
    assertEquals(run.stop.speculation?.delivered, 6);
  });
});

// ---- 純関数（`src/generation/speculation.ts`）------------------------------------

Deno.test("planDraftLength: 確定は最大 k'+1 個なので残り予算を超えない", () => {
  assertEquals(planDraftLength(3, 9), 3, "予算が余っていれば k そのもの");
  assertEquals(planDraftLength(3, 4), 3, "ちょうど k+1 個ぶん残っている");
  assertEquals(planDraftLength(3, 3), 2);
  assertEquals(planDraftLength(3, 2), 1);
  assertEquals(planDraftLength(3, 1), 0, "残り 1 個は draft を採らず verify 1 行で確定する");
  assertEquals(planDraftLength(1, 5), 1);
});

Deno.test("verifyRowIndices: R は常に k+1（k' が縮む cycle も末尾添字で埋める）", () => {
  assertEquals(verifyRowIndices(3, 3), [0, 1, 2, 3]);
  assertEquals(verifyRowIndices(2, 3), [0, 1, 2, 2]);
  assertEquals(verifyRowIndices(1, 3), [0, 1, 1, 1]);
  // `k' = 0` だけは decode 形そのもの（PreparedPlan の鍵まで decode と同一になる）。
  assertEquals(verifyRowIndices(0, 3), [0]);
  assertEquals(verifyRowIndices(1, 1), [0, 1]);
});

Deno.test("takeDrafts: 先頭 k' 本だけを取り、語彙の外と本数不足は fail loudly", async (t) => {
  await t.step("先頭 k' 本（余りは無視する）", () => {
    assertEquals(takeDrafts(Int32Array.from([8, 9, 3]), 2, VOCAB), [8, 9]);
    assertEquals(takeDrafts(Int32Array.from([8, 9, 3]), 0, VOCAB), []);
  });

  await t.step("本数が足りない", () => {
    assertThrows(
      () => takeDrafts(Int32Array.from([8, 9]), 3, VOCAB),
      Error,
      "draft が 2 本しか無い（3 本要る）",
    );
  });

  await t.step("語彙の外（範囲外 gather は行ごと NaN 汚染になる）", () => {
    assertThrows(
      () => takeDrafts(Int32Array.from([8, VOCAB]), 2, VOCAB),
      Error,
      `draft[1] ${VOCAB} が語彙 0..${VOCAB - 1} の外`,
    );
    assertThrows(
      () => takeDrafts(Int32Array.from([-1]), 1, VOCAB),
      Error,
      `draft[0] -1 が語彙 0..${VOCAB - 1} の外`,
    );
  });
});

/** 行 `j` が `tokens[j]` を argmax にする logits の読み口（読んだ行の添字も記録する）。 */
const rowsOf = (tokens: readonly number[]) => {
  const read: number[] = [];
  const row = (index: number): Float32Array<ArrayBuffer> => {
    read.push(index);
    if (index >= tokens.length) throw new Error(`用意していない行 ${index} を読んだ`);
    const data = new Float32Array(new ArrayBuffer(VOCAB * 4));
    data[tokens[index]] = 10;
    return data;
  };
  return { row, read };
};

/** 抽選の履歴を記録する sampler（実体は温度 0 の本物 — 判定は argmax）。 */
const recordingSampler = (): { readonly sampler: Sampler; readonly histories: number[][] } => {
  const inner = createSampler();
  const histories: number[][] = [];
  return {
    sampler: {
      next: (logits, history): number => {
        histories.push([...history]);
        return inner.next(logits, history);
      },
    },
    histories,
  };
};

/** 停止 token が 1 つも無いターン（受理は先頭一致だけで決まる）。 */
const noStop = (): boolean => false;

Deno.test("acceptDrafts: 先頭一致で止まった位置が受理数・その行の抽選が b′", async (t) => {
  await t.step("全受理（行 k' が b′ を出す）", () => {
    const { row, read } = rowsOf([8, 9, 3, 13]);
    const { sampler } = recordingSampler();
    assertEquals(acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], noStop), {
      accepted: 3,
      confirmed: [8, 9, 3, 13],
    });
    assertEquals(read, [0, 1, 2, 3], "全受理では draft の本数 + 1 行を読む");
  });

  await t.step("途中で外れたら以降の行は読まない（早期打ち切り）", () => {
    // 行 1 の抽選は 7 で、draft の 9 と食い違う → そこで止まり b′ = 7。
    const { row, read } = rowsOf([8, 7, 3, 13]);
    const { sampler } = recordingSampler();
    assertEquals(acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], noStop), {
      accepted: 1,
      confirmed: [8, 7],
    });
    assertEquals(read, [0, 1], "棄却した行より後ろの logits を読んでいる");
  });

  await t.step("受理 0 でも b′ は必ず 1 個確定する（decode 1 step への退化）", () => {
    const { row, read } = rowsOf([8, 9, 3, 13]);
    const { sampler } = recordingSampler();
    assertEquals(acceptDrafts(row, [REJECTED, REJECTED, REJECTED], sampler, [1, 2, 6], noStop), {
      accepted: 0,
      confirmed: [8],
    });
    assertEquals(read, [0]);
  });

  await t.step("行 j の抽選には「その行までの確定列」を渡す", () => {
    const { row } = rowsOf([8, 9, 3, 13]);
    const { sampler, histories } = recordingSampler();
    const history: readonly number[] = [1, 2, 6];
    acceptDrafts(row, [8, 9, 3], sampler, history, noStop);
    assertEquals(histories, [
      [1, 2, 6],
      [1, 2, 6, 8],
      [1, 2, 6, 8, 9],
      [1, 2, 6, 8, 9, 3],
    ]);
    // 呼び手の `history` は触らない（確定列は呼び手が配送しながら伸ばす）。
    assertEquals(history, [1, 2, 6]);
  });

  await t.step("棄却した行の draft は履歴に積まない", () => {
    const { row } = rowsOf([8, 7, 3, 13]);
    const { sampler, histories } = recordingSampler();
    acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], noStop);
    assertEquals(histories, [[1, 2, 6], [1, 2, 6, 8]]);
  });

  await t.step("受理した draft が停止 token なら列挙を止め、後続行の抽選をしない", () => {
    // 行 2 以降を**用意しない**読み口（読めば投げる）= 停止で止めない実装だけが赤くなる。
    // 非投機はこの位置で生成を終えるので、その先の logits には触れないのが正しい。
    const { row, read } = rowsOf([8, 9]);
    const { sampler, histories } = recordingSampler();
    assertEquals(acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], (token) => token === 9), {
      accepted: 2,
      confirmed: [8, 9],
    });
    assertEquals(read, [0, 1], "停止 token の先の行を読んでいる");
    // 抽選も 2 回だけ（確定 token 1 個につき 1 回 = RNG の消費列が非投機と並ぶ）。
    assertEquals(histories, [[1, 2, 6], [1, 2, 6, 8]]);
  });

  await t.step("棄却時の b′ が停止 token なら確定列は accepted + 1 個のまま", () => {
    // 行 1 で外れて b′ = 7 を引き、その 7 が停止 token。列は `[d₁, b′]` = 長さ accepted + 1。
    const { row, read } = rowsOf([8, 7, 3, 13]);
    const { sampler } = recordingSampler();
    assertEquals(acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], (token) => token === 7), {
      accepted: 1,
      confirmed: [8, 7],
    });
    assertEquals(read, [0, 1]);
  });
});

Deno.test("投機の診断負荷は cycle の壁とゲート判断に混ざらない", async () => {
  const results = [];
  for (const observerMs of [0, 5, 50]) {
    let elapsed = 0;
    const drafter = fakeDrafter({
      draft: (cycle) => {
        elapsed += 1;
        return chainFrom(cycle.token, K);
      },
    });
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      clock: {
        now: () => elapsed,
        afterRun: () => {
          elapsed += 10;
        },
      },
      onRun: () => {
        elapsed += observerMs;
      },
    });
    const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 60 });
    const events = [];
    for await (const event of stream) events.push(event);
    results.push({ events, stop: await stream.done, phases: opened.phases });
    await opened.sequence.dispose();
  }
  assertEquals(results[1], results[0]);
  assertEquals(results[2], results[0]);
});

Deno.test("空 draft の抽選は不要な履歴の読み取りをしない", () => {
  const history = new Proxy([1, 2, 6], {
    get: () => {
      throw new Error("抽選に不要な履歴をコピーした");
    },
  });
  let draws = 0;
  const { row, read } = rowsOf([8]);
  const result = acceptDrafts(
    row,
    [],
    {
      next: (_logits, seen) => {
        assert(seen === history);
        draws += 1;
        return 8;
      },
    },
    history,
    noStop,
  );
  assertEquals(result, { accepted: 0, confirmed: [8] });
  assertEquals(read, [0]);
  assertEquals(draws, 1);
});
