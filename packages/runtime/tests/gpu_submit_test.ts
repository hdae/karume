import { assertEquals, assertLess, assertRejects, assertThrows } from "@std/assert";
import { type GpuContext, GpuDeviceLostError } from "../src/gpu/device.ts";
import {
  DEFAULT_SUBMIT_POLICY,
  MEASURED_HISTORY,
  type SubmitPolicy,
  SubmitPolicyError,
  SubmitScheduler,
} from "../src/gpu/submit.ts";
import { fakeGpuContext } from "./helpers/fake-gpu.ts";

/** SubmitScheduler が触る面だけを持つフェイク。DOM 型全体は再現しないため cast で渡す。 */
type FakeGpu = {
  /** device だけをフェイクにし、GpuContext（消失購読・ロック）は本番実装のまま使う。 */
  readonly context: GpuContext;
  /** submit された command buffer ごとの dispatch 件数。 */
  readonly submitted: number[];
  /**
   * `queue.onSubmittedWorkDone` の呼び出し回数。
   *
   * MUST: ここを見るテストを残す。submit ごとにこれを呼ぶと、実装によっては同期部分が
   * GPU 完了までホストをブロックし、CPU エンコードと GPU 実行が完全に直列化する
   * （submit.ts のモジュール doc）。**呼び出し回数以外にこの退行を捉える面は無い** —
   * 出力も統計も 1 ビットも変わらず、遅くなるだけで緑のまま通る。
   */
  readonly calls: { workDone: number };
};

type FakeOptions = {
  readonly lost?: Promise<GPUDeviceLostInfo>;
  readonly workDone?: () => Promise<void>;
};

const createFakeGpu = (options: FakeOptions = {}): FakeGpu => {
  const submitted: number[] = [];
  const calls = { workDone: 0 };
  const workDone = options.workDone ?? ((): Promise<void> => Promise.resolve());
  const device = {
    lost: options.lost ?? new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set<string>(),
    queue: {
      submit: (buffers: readonly { readonly count: number }[]): void => {
        for (const buffer of buffers) {
          submitted.push(buffer.count);
        }
      },
      onSubmittedWorkDone: (): Promise<void> => {
        calls.workDone += 1;
        return workDone();
      },
    },
    createCommandEncoder: () => {
      let count = 0;
      return {
        beginComputePass: () => ({
          setPipeline: (): void => {},
          setBindGroup: (): void => {},
          dispatchWorkgroups: (): void => {
            count += 1;
          },
          end: (): void => {},
        }),
        finish: () => ({ count }),
      };
    },
  };
  return { context: fakeGpuContext(device as unknown as GPUDevice), submitted, calls };
};

/**
 * 「窓の間に GPU が `elapsedMs` だけ走った」を作るフェイク時計。
 *
 * 計測窓は「窓最初の submit → flush の `onSubmittedWorkDone` 完了」なので、完了の瞬間に
 * 時計を進めると窓の実測がちょうど `elapsedMs` になる（注入した時刻源で規則を固定できる形）。
 */
const gpuTimeClock = (elapsedMs: () => number): {
  readonly now: () => number;
  /** ホスト側で時間が経った（= エンコードに掛かった）ことを作る。 */
  readonly advance: (ms: number) => void;
  readonly workDone: () => Promise<void>;
} => {
  let clock = 0;
  return {
    now: (): number => clock,
    advance: (ms: number): void => {
      clock += ms;
    },
    workDone: (): Promise<void> => {
      clock += elapsedMs();
      return Promise.resolve();
    },
  };
};

const fakePipeline = {} as unknown as GPUComputePipeline;
const fakeBindGroup = {} as unknown as GPUBindGroup;

/** 適応を止めた固定チャンク政策（分割境界を決定的に観測するため）。 */
const fixedPolicy = (size: number): SubmitPolicy => ({
  timeBudgetMs: 100,
  initialChunkSize: size,
  minChunkSize: size,
  maxChunkSize: size,
});

/** 1 dispatch = workgroup 1 個ぶんの仕事（時間予算の単位が dispatch 数と一致する形）。 */
const dispatchMany = (scheduler: SubmitScheduler, count: number): void => {
  for (let i = 0; i < count; i += 1) {
    scheduler.dispatch(fakePipeline, fakeBindGroup, [1, 1, 1], "test:fake");
  }
};

/** 適応制御の政策（実測が付くまでは 4 dispatch / 予算 100ms）。 */
const adaptivePolicy: SubmitPolicy = {
  timeBudgetMs: 100,
  initialChunkSize: 4,
  minChunkSize: 1,
  maxChunkSize: 1024,
};

Deno.test("SubmitScheduler は dispatch 列をチャンクサイズごとに分割して submit する", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 10);
  assertEquals(gpu.submitted, [4, 4], "チャンクが埋まった時点で submit されている");

  await scheduler.flush();
  assertEquals(gpu.submitted, [4, 4, 2], "flush が端数を出し切る");
  assertEquals(scheduler.stats.submitCount, 3);
  assertEquals(scheduler.stats.dispatchCount, 10);
});

Deno.test("SubmitScheduler の flush は未 submit が無ければ何も submit しない", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  await scheduler.flush();
  assertEquals(gpu.submitted, []);
  assertEquals(scheduler.stats.submitCount, 0);
  // 空でも完了は待つ（flush-before-destroy は「出すものが無い」でも成立させる不変条件）。
  assertEquals(gpu.calls.workDone, 1, "空 flush でも完了は待つ");
  // 窓は 1 度も開いていない。空の flush を「実測 0 の窓」として数えると、診断の履歴が
  // 中身の無い件で埋まり、適応の傾向が読めなくなる。
  assertEquals(scheduler.stats.measuredMs, [], "submit していない flush は窓を作らない");
  assertEquals(scheduler.stats.measuredCount, 0);
});

// 案 1 の唯一の構造検出器（{@link FakeGpu.calls} の MUST）。submit ごとに
// onSubmittedWorkDone を呼ぶ実装に戻しても、出力も統計も 1 ビットも変わらない —
// 呼び出し回数を数えるここだけが赤くなる。
Deno.test("onSubmittedWorkDone を呼ぶのは flush の 1 回だけ（submit ごとには呼ばない）", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4), () => 0);

  dispatchMany(scheduler, 20);
  assertEquals(gpu.submitted, Array(5).fill(4), "5 チャンクが submit 済み");
  assertEquals(gpu.calls.workDone, 0, "submit の時点では 1 度も呼ばない");

  await scheduler.flush();
  assertEquals(scheduler.stats.submitCount, 5);
  assertEquals(gpu.calls.workDone, 1, "5 回 submit しても待ちは flush の 1 回だけ");

  // 2 回目の run（次の flush まで）でも増えるのは flush のぶんだけ。
  dispatchMany(scheduler, 12);
  assertEquals(gpu.calls.workDone, 1);
  await scheduler.flush();
  assertEquals(scheduler.stats.submitCount, 8);
  assertEquals(gpu.calls.workDone, 2, "flush 1 回につき 1 回");
});

Deno.test("SubmitScheduler はチャンクサイズ 0 相当の政策を構築時に拒否する", () => {
  const gpu = createFakeGpu();

  for (const key of ["initialChunkSize", "minChunkSize", "maxChunkSize"] as const) {
    assertThrows(
      () => new SubmitScheduler(gpu.context, { ...DEFAULT_SUBMIT_POLICY, [key]: 0 }),
      SubmitPolicyError,
      key,
    );
  }
  assertThrows(
    () => new SubmitScheduler(gpu.context, { ...DEFAULT_SUBMIT_POLICY, initialChunkSize: 1.5 }),
    SubmitPolicyError,
  );
  assertThrows(
    () => new SubmitScheduler(gpu.context, { ...DEFAULT_SUBMIT_POLICY, timeBudgetMs: 0 }),
    SubmitPolicyError,
  );
  assertThrows(
    () =>
      new SubmitScheduler(gpu.context, {
        ...DEFAULT_SUBMIT_POLICY,
        minChunkSize: 8,
        maxChunkSize: 4,
      }),
    SubmitPolicyError,
  );
});

Deno.test("SubmitScheduler の flush は device 消失を待ち続けずに例外にする", async () => {
  const lostInfo = { reason: "unknown", message: "test" } as unknown as GPUDeviceLostInfo;
  const gpu = createFakeGpu({
    lost: Promise.resolve(lostInfo),
    // 消失した device では完了通知が来ない状況を再現する。
    workDone: () => new Promise<void>(() => {}),
  });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 2);
  await assertRejects(() => scheduler.flush(), GpuDeviceLostError);
});

// 窓の定義そのものの門: 起点は「窓で**最初に** submit した時刻」（直近の submit ではない）、
// 仕事量は「窓に積んだ**全チャンク**の workgroup 合計」（最後のチャンクぶんではない）。
Deno.test("計測窓は最初の submit から flush 完了までで、推定は窓の合計 workgroup で割る", async () => {
  const clock = gpuTimeClock(() => 11);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2), clock.now);

  // chunk1 を t=0 で submit し、ホスト側で 1ms 掛けてから chunk2 を t=1 で submit する。
  dispatchMany(scheduler, 2);
  clock.advance(1);
  dispatchMany(scheduler, 2);
  assertEquals(gpu.submitted, [2, 2], "2 本とも flush 前に submit 済み");
  assertEquals(scheduler.stats.measuredMs, [], "窓は flush まで閉じない");
  assertEquals(scheduler.stats.msPerWorkgroup, undefined);

  // flush の完了で t=1+11=12。起点を「直近の submit（t=1）」にすると 11ms になる。
  await scheduler.flush();
  assertEquals(scheduler.stats.measuredMs, [12], "起点は最初の submit（t=0）");
  assertEquals(scheduler.stats.measuredCount, 1);
  // 12ms / 4 workgroup。最後のチャンクぶん（2 workgroup）で割ると 6 になる。
  assertEquals(scheduler.stats.msPerWorkgroup, 3, "推定は窓の実測 ÷ 窓の合計 workgroup 数");
  assertEquals(scheduler.stats.discardedDispatches, 0, "成功経路では 1 件も捨てていない");
});

Deno.test("measuredMs は直近 MEASURED_HISTORY 件で頭打ちになり、累積は measuredCount に出る", async () => {
  let elapsed = 0;
  const clock = gpuTimeClock(() => elapsed);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(1), clock.now);
  const total = MEASURED_HISTORY + 5;

  // i 本目の窓の実測が i+1 ms になるようにする（保持されたのが「直近」だと分かる形）。
  for (let i = 0; i < total; i += 1) {
    elapsed = i + 1;
    dispatchMany(scheduler, 1);
    await scheduler.flush();
  }

  const stats = scheduler.stats;
  assertEquals(stats.measuredCount, total, "累積件数は履歴から溢れても数え続ける");
  assertEquals(stats.measuredMs.length, MEASURED_HISTORY, "履歴は有界（flush 回数に比例しない）");
  assertLess(stats.measuredMs.length, total, "無制限成長していない");
  assertEquals(stats.measuredMs[0], total - MEASURED_HISTORY + 1, "古い側から捨てる");
  assertEquals(stats.measuredMs[MEASURED_HISTORY - 1], total, "最新の実測が末尾に残る");
});

Deno.test("discard は未 submit のエンコードを submit せずに捨て、件数を診断に出す", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 6);
  assertEquals(gpu.submitted, [4], "埋まった 1 チャンクだけが submit 済み");

  scheduler.discard();
  assertEquals(scheduler.stats.discardedDispatches, 2);

  // 捨てた残骸は後続の flush でも submit されない（失敗経路の後始末が流し直さない根拠）。
  await scheduler.flush();
  assertEquals(gpu.submitted, [4], "flush が残骸を出し直さない");
  assertEquals(scheduler.stats.submitCount, 1);
  assertEquals(scheduler.stats.dispatchCount, 6, "積んだ件数そのものは書き換えない");
});

// 不変条件 1（submit.ts のモジュール doc）: 実測 0 は「速い」ではなく「情報が無い」。
// これを成長の根拠にすると、タイマ分解能の粗い環境でチャンクが無条件に膨らみ、
// dispatch 数がグラフ全体に届いた時点で 1 submit 全積み（= TDR 域）になる。
Deno.test("実測 0 が続く限りチャンクは初期上限から成長しない", async () => {
  const gpu = createFakeGpu();
  // 時刻が 1 度も進まない = 窓の実測が 0（分解能未満の再現）。
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, () => 0);

  for (let window = 0; window < 10; window += 1) {
    dispatchMany(scheduler, 4);
    await scheduler.flush();
  }
  assertEquals(gpu.submitted, Array(10).fill(4), "裏付けが無い間は初期上限のまま");
  assertEquals(scheduler.stats.measuredCount, 10);
  assertEquals(scheduler.stats.measuredMs, Array(10).fill(0), "全窓が実測 0");
  assertEquals(scheduler.stats.msPerWorkgroup, undefined, "実測 0 の窓では推定を更新しない");

  // 10 窓ぶんの「情報なし」を踏んでも上限は動かない（倍々に戻すとここが赤くなる）。
  dispatchMany(scheduler, 12);
  assertEquals(gpu.submitted, Array(13).fill(4), "実測 0 は成長の根拠にならない");
  assertEquals(scheduler.stats.currentChunkSize, 4);
});

// 不変条件 1 の裏面（0 除算側）。仕事量 0 の窓で更新すると msPerWorkgroup が Infinity に
// なり、以後どんな dispatch も予算超過と判定されて minChunkSize へ張り付く（成長の暴走とは
// 逆向きだが、同じ「情報の無い実測を根拠にした」故障）。
Deno.test("仕事量 0 の窓では推定を更新しない", async () => {
  const clock = gpuTimeClock(() => 5);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2), clock.now);

  // workgroup 0 の dispatch だけを積む（GPU 上では no-op だが submit は起きる）。
  scheduler.dispatch(fakePipeline, fakeBindGroup, [0, 1, 1], "test:fake");
  scheduler.dispatch(fakePipeline, fakeBindGroup, [0, 1, 1], "test:fake");
  await scheduler.flush();

  assertEquals(gpu.submitted, [2], "submit 自体は起きている");
  assertEquals(scheduler.stats.measuredMs, [5], "窓は閉じて実測も記録される");
  assertEquals(scheduler.stats.msPerWorkgroup, undefined, "仕事量 0 の窓には情報が無い");
  assertEquals(scheduler.stats.currentChunkSize, 2, "裏付け無しの上限のまま");
});

// 不変条件 2: 1 チャンクの推定時間が予算を超えない。切れ目は
// `timeBudgetMs * CHUNK_TIME_SAFETY / msPerWorkgroup` = 100 * 0.5 / 2 = 25 workgroup。
Deno.test("実測の裏付けが付いたら workgroup 予算でチャンクを切る", async () => {
  const clock = gpuTimeClock(() => 8);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, clock.now);

  dispatchMany(scheduler, 4);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 2, "8ms / 4 workgroup");
  assertEquals(scheduler.stats.currentChunkSize, 1024, "裏付けが付けば上限は硬い上限まで開く");

  dispatchMany(scheduler, 60);
  assertEquals(gpu.submitted, [4, 25, 25], "予算 25 workgroup ちょうどで切る（端数 10 は保留）");
});

// 不変条件 2 の続き: 予算判定は**積む前**。積んでから縮めると、超過したチャンクは既に
// submit 済みで「毎回 1 度は予算を踏む」が定常になる（不変条件 3）。
Deno.test("単独で予算を超える dispatch は、積む前にチャンクを閉じて 1 本で出す", async () => {
  const clock = gpuTimeClock(() => 8);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, clock.now);

  dispatchMany(scheduler, 4);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 2, "予算は 25 workgroup");

  dispatchMany(scheduler, 3);
  assertEquals(gpu.submitted, [4], "3 dispatch（3 workgroup）はまだ予算内");
  // 30 workgroup の 1 本 = 単独で予算超過。積む前に手前の 3 本を締める。
  scheduler.dispatch(fakePipeline, fakeBindGroup, [30, 1, 1], "test:fake");
  assertEquals(gpu.submitted, [4, 3], "重い dispatch を混ぜずに手前を締める");
  dispatchMany(scheduler, 1);
  assertEquals(gpu.submitted, [4, 3, 1], "分割できない 1 本はそれだけで 1 チャンク");
});

// 不変条件 3 の裏面: 遅い実測は次のチャンクへフィードフォワードされる（縮む）。
Deno.test("遅いチャンクを踏んだら次のチャンクが縮む", async () => {
  // 4 workgroup で 200ms = 予算 100ms の 2 倍。
  const clock = gpuTimeClock(() => 200);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, clock.now);

  dispatchMany(scheduler, 4);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 50, "200ms / 4 workgroup");

  // 予算 50ms ÷ 50ms = 1 workgroup → 1 dispatch ずつに縮む。
  dispatchMany(scheduler, 3);
  assertEquals(gpu.submitted, [4, 1, 1], "予算超過の見込みが立った時点で切る");
});

// chunkBudget（推定側）: cost proxy 起票の着手条件「予算超過チャンクが観測されたか」を
// 数える席。切りようが無かったチャンク（単独で予算を超える dispatch）だけが上がる。
Deno.test("チャンク診断は推定時間の最大と予算超過チャンク数を数える", async () => {
  const clock = gpuTimeClock(() => 8);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, clock.now);

  dispatchMany(scheduler, 4);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 2, "8ms / 4 workgroup");
  assertEquals(scheduler.stats.chunkBudget.budgetMs, 50, "100ms × 安全率 0.5");
  // 1 本目のチャンクは裏付けが付く前に出ている = 推定時間を持たない（0ms として数えない）。
  assertEquals(
    scheduler.stats.chunkBudget.maxEstimatedMs,
    undefined,
    "裏付け前のチャンクは推定側に載せない",
  );
  assertEquals(scheduler.stats.chunkBudget.overBudgetChunks, 0);

  // 3 workgroup（推定 6ms）で締めてから、単独 30 workgroup（推定 60ms > 予算 50ms）を出す。
  dispatchMany(scheduler, 3);
  scheduler.dispatch(fakePipeline, fakeBindGroup, [30, 1, 1], "test:fake");
  assertEquals(scheduler.stats.chunkBudget.maxEstimatedMs, 6, "締めた 3 workgroup ぶん");
  assertEquals(scheduler.stats.chunkBudget.overBudgetChunks, 0, "予算内のチャンクは数えない");

  dispatchMany(scheduler, 1);
  assertEquals(gpu.submitted, [4, 3, 1], "分割できない 1 本はそれだけで 1 チャンク");
  assertEquals(scheduler.stats.chunkBudget.maxEstimatedMs, 60, "30 workgroup × 2ms");
  assertEquals(scheduler.stats.chunkBudget.overBudgetChunks, 1, "切りようが無かった 1 本を数える");

  // 制御は 1 ビットも変わっていない（診断は分割境界に影響しない）。
  await scheduler.flush();
  assertEquals(gpu.submitted, [4, 3, 1, 1], "flush が端数を出し切るだけ");
  assertEquals(scheduler.stats.chunkBudget.overBudgetChunks, 1, "端数チャンクは予算内");
});

// chunkBudget（実測側）: 「workgroup 数 = 仕事量」のプロキシがずれると、推定側の席は
// 予算内だと言い続ける（同じプロキシで見積もった値を比べているだけなので原理的に見えない）。
// 窓平均は帰属に依存せず求まり、最大チャンク時間の**下界**になる。
Deno.test("窓平均チャンク時間は推定が見落とす予算超過を下界として捉える", async () => {
  let elapsed = 1;
  const clock = gpuTimeClock(() => elapsed);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  // 1 dispatch = 1 チャンク（窓のチャンク数を決定的にする）。
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(1), clock.now);

  // 1 workgroup が 1ms、という軽いカーネルで裏付けを付ける。
  dispatchMany(scheduler, 1);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 1, "1ms / 1 workgroup");
  assertEquals(scheduler.stats.chunkBudget.maxWindowMeanMs, 1, "1 チャンクの窓は平均 = 実測");

  // 同じ workgroup 数なのに 1 窓 300ms 掛かる（= プロキシが桁で外れている状況）。
  elapsed = 300;
  dispatchMany(scheduler, 2);
  await scheduler.flush();

  const budget = scheduler.stats.chunkBudget;
  assertEquals(budget.maxWindowMeanMs, 150, "300ms / 2 チャンク");
  // 150ms > 予算 50ms なので「予算を超えたチャンクが少なくとも 1 本あった」と言い切れる。
  assertLess(budget.budgetMs, budget.maxWindowMeanMs ?? 0, "窓平均だけで超過を断定できる");
  assertEquals(budget.maxEstimatedMs, 1, "推定側は 1ms のまま（プロキシのずれは見えない）");
  assertEquals(budget.overBudgetChunks, 0, "推定側の席はここでは 1 件も上がらない");
});
