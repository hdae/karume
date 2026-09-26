// Anima の DiT 常駐（ADR 0112）の決定と状態機械の門（GPU も実資産も要らない）。
//
// 実 GPU の e2e（`e2e_anima_test.ts` の「DiT 常駐」節）が縛るのは「常駐しても出る画素が 1 bit も
// 動かない」ことと寿命の観測で、OOM の退避・格下げ・失敗時の破棄は**実機では狙って起こせない**。
// そこでここは `residency.ts` の純関数と状態機械を、pipeline の `#generate` と同じ呼び順
// （text 段 → DiT 段 → VAE 段）で回す偽の generate に載せ、故障（OOM / 失敗 / 後始末の失敗）を
// 注入して遷移を固定する。
//
// NOTE: `AnimaPipeline.generate` そのものは偽の GPU では回せない — 段の Session は実容器の
// グラフを device へ上げて組むので、`fake-gpu.ts` の device では構築が通らない。pipeline 側の
// 結線（段の順序・イベントへの写し）は e2e が見る。pipeline.ts の GPU 側の小物（解放待ちの
// `settleReleasedMemory` と dispose の順序 `disposeResidencyThenGpu`）は偽の GpuContext で直接縛る。

import {
  assert,
  assertEquals,
  assertFalse,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { GpuOutOfMemoryError, GpuValidationError } from "@karume/runtime";
import {
  type AnimaResidency,
  assertAnimaResidency,
  decideAfterFailedGenerate,
  decideAfterTransformer,
  evictsForMemory,
  type ResidencyNotice,
  TransformerResidency,
  transformerSource,
} from "../src/anima/residency.ts";
import { disposeResidencyThenGpu, settleReleasedMemory } from "../src/anima/pipeline.ts";
import { ModelInputError } from "../src/errors.ts";
import { losableGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

/** 偽の DiT Session（破棄されたかと、どの構築で作られたかだけを持つ）。 */
type FakeSession = { readonly id: number; disposed: boolean };

/** 注入する故障（段ごとに 1 回ずつ消費する）。 */
type Fault = "oom" | "failure";

/** 偽の generate 1 回ぶんの要求。 */
type FakeRequest = {
  readonly residency: AnimaResidency;
  /** 段ごとの故障の列（先頭から試行ごとに 1 つ消費・尽きたら成功）。 */
  readonly faults?: Partial<Record<"text" | "transformer" | "vae", Fault[]>>;
};

const oom = (where: string): GpuOutOfMemoryError =>
  new GpuOutOfMemoryError(`${where}: not enough memory left`);

/** 台に注入する後始末側の故障。 */
type HarnessOptions = {
  readonly disposeFails?: boolean;
  readonly settleFails?: boolean;
  /** 通知の購読側が投げる例外（返した値を投げる・undefined なら投げない）。 */
  readonly notifyThrows?: (notice: ResidencyNotice) => unknown;
};

/**
 * pipeline の `#generate` と同じ呼び順で状態機械を回す台（text 段 → DiT 段 → VAE 段 — 失敗したら
 * `releaseIfRequested` を通して投げる）。`log` に GPU 側の出来事（構築 / 破棄 / 解放待ち / 段の試行）と
 * 通知を時系列で積む。`disposals` は破棄の瞬間の状態機械の状態（MUST の順序の観測点）。
 */
const createHarness = (options: HarnessOptions = {}) => {
  const log: string[] = [];
  const sessions: FakeSession[] = [];
  const notices: ResidencyNotice[] = [];
  const disposals: {
    readonly id: number;
    readonly holding: boolean;
    readonly downgraded: boolean;
  }[] = [];
  const residency: TransformerResidency<FakeSession> = new TransformerResidency<FakeSession>({
    dispose: (session) => {
      session.disposed = true;
      log.push(`dispose:${session.id}`);
      disposals.push({
        id: session.id,
        holding: residency.holding,
        downgraded: residency.downgraded,
      });
      if (options.disposeFails === true) throw new Error(`dispose ${session.id} に失敗`);
    },
    settleRelease: () => {
      log.push("settle");
      return options.settleFails === true
        ? Promise.reject(new Error("解放待ちに失敗"))
        : Promise.resolve();
    },
  });
  const notify = (notice: ResidencyNotice): Promise<void> => {
    notices.push(notice);
    log.push(`notice:${notice.action}/${notice.reason}`);
    const thrown = options.notifyThrows?.(notice);
    return thrown === undefined ? Promise.resolve() : Promise.reject(thrown);
  };
  const takeFault = (request: FakeRequest, stage: "text" | "transformer" | "vae") =>
    request.faults?.[stage]?.shift();

  /** 段ごと運転の段（text / VAE）1 回の試行。故障があれば投げる。 */
  const attempt = (request: FakeRequest, stage: "text" | "vae") => (): Promise<string> => {
    log.push(`attempt:${stage}`);
    const fault = takeFault(request, stage);
    if (fault === "oom") return Promise.reject(oom(stage));
    if (fault === "failure") return Promise.reject(new Error(`${stage} が失敗`));
    return Promise.resolve(stage);
  };

  const generate = async (request: FakeRequest): Promise<number> => {
    try {
      await residency.runStage(attempt(request, "text"), notify);
      const used = await residency.runTransformer({
        effective: request.residency,
        open: () => {
          const session: FakeSession = { id: sessions.length + 1, disposed: false };
          sessions.push(session);
          log.push(`open:${session.id}`);
          return Promise.resolve(session);
        },
        body: (session) => {
          log.push(`body:${session.id}`);
          assertFalse(session.disposed, "破棄済みの Session で段を回している");
          const fault = takeFault(request, "transformer");
          if (fault === "oom") return Promise.reject(oom("transformer"));
          if (fault === "failure") return Promise.reject(new Error("transformer が失敗"));
          return Promise.resolve(session.id);
        },
        notify,
      });
      await residency.runStage(attempt(request, "vae"), notify);
      return used;
    } catch (error) {
      throw await residency.releaseIfRequested(error, request.residency, notify);
    }
  };

  return { log, sessions, notices, disposals, residency, notify, generate };
};

describe("assertAnimaResidency", () => {
  it("2 つの綴りだけを受ける", () => {
    assertEquals(assertAnimaResidency("per-stage", "residency"), "per-stage");
    assertEquals(assertAnimaResidency("transformer", "residency"), "transformer");
  });

  it("未知の綴りは既定へ縮退せず、出所と実際の値を名指しして ModelInputError で落とす", () => {
    // 綴り違いが黙って段ごと運転になると、opt-in が効いていないことに速度でしか気付けない。
    for (const [value, actual] of [["transformers", "'transformers'"], [null, "null"], [1, "1"]]) {
      assertThrows(
        () => assertAnimaResidency(value, "residency"),
        ModelInputError,
        `residency: 期待 'per-stage' / 'transformer'（実際 ${actual}）`,
      );
    }
  });
});

describe("transformerSource（DiT 段の Session の出所）", () => {
  it("常駐 DiT があれば、実効値にも格下げにも関わらずそれを使う（読み直さない）", () => {
    for (const effective of ["per-stage", "transformer"] as const) {
      assertEquals(transformerSource({ effective, downgraded: false, held: true }), "resident");
    }
  });

  it("常駐が無ければ、transformer かつ格下げでないときだけ常駐の席に載せて作る", () => {
    assertEquals(
      transformerSource({ effective: "transformer", downgraded: false, held: false }),
      "build-resident",
    );
    assertEquals(
      transformerSource({ effective: "transformer", downgraded: true, held: false }),
      "per-stage",
    );
    assertEquals(
      transformerSource({ effective: "per-stage", downgraded: false, held: false }),
      "per-stage",
    );
  });
});

describe("decideAfterTransformer（DiT 段を抜けるときの決定）", () => {
  it("既定の段ごと運転で常駐と無関係なら、手放すだけで何も名乗らない（既定のイベント列を変えない）", () => {
    assertEquals(
      decideAfterTransformer({ effective: "per-stage", downgraded: false, held: false }, "success"),
      { action: "release" },
    );
  });

  it("transformer で席が空（この generate で作る DiT）なら、成功した段の後に席へ載せて retained / request を名乗る", () => {
    assertEquals(
      decideAfterTransformer(
        { effective: "transformer", downgraded: false, held: false },
        "success",
      ),
      { action: "retain", notice: { action: "retained", reason: "request" } },
    );
  });

  it("席に DiT があるまま格下げ済み（到達しない組）は既定値へ落とさず投げる", () => {
    // 格下げは退避で席を空けるときにだけ立ち、以後は席に載せない。黙って何かを返すと、
    // 「常駐しない判断は必ず名乗る」と食い違う値が残る。
    for (const effective of ["per-stage", "transformer"] as const) {
      for (const outcome of ["success", "failure"] as const) {
        assertThrows(
          () => decideAfterTransformer({ effective, downgraded: true, held: true }, outcome),
          Error,
          "到達しない状態",
        );
      }
    }
  });

  it("transformer で席に載っていれば持ち続け、retained / request を名乗る", () => {
    assertEquals(
      decideAfterTransformer(
        { effective: "transformer", downgraded: false, held: true },
        "success",
      ),
      { action: "retain", notice: { action: "retained", reason: "request" } },
    );
  });

  it("持ち越した常駐 DiT を per-stage の要求で使ったら手放し、released / request を名乗る", () => {
    assertEquals(
      decideAfterTransformer({ effective: "per-stage", downgraded: false, held: true }, "success"),
      { action: "release", notice: { action: "released", reason: "request" } },
    );
  });

  it("格下げ済みで transformer を求められたら持たず、released / downgraded を名乗る", () => {
    assertEquals(
      decideAfterTransformer(
        { effective: "transformer", downgraded: true, held: false },
        "success",
      ),
      { action: "release", notice: { action: "released", reason: "downgraded" } },
    );
  });

  it("失敗した段の DiT は持ち越しなら捨てて evicted / failure を名乗り、この generate で作ったなら畳むだけ", () => {
    for (const effective of ["per-stage", "transformer"] as const) {
      assertEquals(
        decideAfterTransformer({ effective, downgraded: false, held: true }, "failure"),
        { action: "evict", notice: { action: "evicted", reason: "failure" } },
      );
      assertEquals(
        decideAfterTransformer({ effective, downgraded: false, held: false }, "failure"),
        { action: "release" },
      );
    }
  });
});

describe("decideAfterFailedGenerate（DiT 段の外で generate が失敗したとき）", () => {
  it("実効値 per-stage なら席の DiT を手放して released / request を名乗る（失敗しても指示は効く）", () => {
    assertEquals(decideAfterFailedGenerate("per-stage"), {
      action: "release",
      notice: { action: "released", reason: "request" },
    });
  });

  it("実効値 transformer なら持ったまま何も名乗らない", () => {
    assertEquals(decideAfterFailedGenerate("transformer"), { action: "retain" });
  });
});

describe("evictsForMemory（退避に当たる失敗か）", () => {
  it("常駐 DiT があり、失敗が GpuOutOfMemoryError のときだけ退避する", () => {
    assert(evictsForMemory(oom("x"), true));
    assertFalse(evictsForMemory(oom("x"), false), "退避する相手が居ない");
    // validation や素の Error を退避でやり直すと、本当の原因が 2 度目の失敗に埋もれる。
    assertFalse(evictsForMemory(new GpuValidationError("x"), true));
    assertFalse(evictsForMemory(new Error("out-of-memory"), true), "文言では判定しない");
  });
});

describe("TransformerResidency（偽の generate で回す状態機械）", () => {
  describe("① 既定の per-stage", () => {
    it("DiT を generate ごとに作って段の終わりで畳み、何も名乗らない", async () => {
      const h = createHarness();
      assertEquals(await h.generate({ residency: "per-stage" }), 1);
      assertEquals(await h.generate({ residency: "per-stage" }), 2);
      assertEquals(h.log, [
        "attempt:text",
        "open:1",
        "body:1",
        "dispose:1",
        "attempt:vae",
        "attempt:text",
        "open:2",
        "body:2",
        "dispose:2",
        "attempt:vae",
      ]);
      assertEquals(h.notices, []);
      assertFalse(h.residency.holding);
    });
  });

  describe("② transformer", () => {
    it("最初の generate の DiT 段で作り、以後は同じ Session を読み直さずに使う", async () => {
      const h = createHarness();
      assertEquals(await h.generate({ residency: "transformer" }), 1);
      assertEquals(await h.generate({ residency: "transformer" }), 1);
      assertEquals(h.sessions.length, 1, "2 回目の generate で DiT を作り直している");
      assertFalse(h.sessions[0].disposed);
      assert(h.residency.holding);
      assertEquals(h.notices, [
        { action: "retained", reason: "request" },
        { action: "retained", reason: "request" },
      ]);
    });

    it("dispose で常駐 DiT を畳む（2 度目は何もしない）", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      await h.residency.dispose();
      assert(h.sessions[0].disposed);
      assertFalse(h.residency.holding);
      await h.residency.dispose();
      assertEquals(h.log.filter((entry) => entry.startsWith("dispose")), ["dispose:1"]);
    });
  });

  describe("③ request による上書き", () => {
    it("transformer で持った DiT を次の per-stage が使い、その後に手放す（連続生成の最後の 1 枚）", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      assertEquals(
        await h.generate({ residency: "per-stage" }),
        1,
        "持ち越した DiT を使っていない",
      );
      assert(h.sessions[0].disposed);
      assertFalse(h.residency.holding);
      assertEquals(h.notices, [
        { action: "retained", reason: "request" },
        { action: "released", reason: "request" },
      ]);
      // 手放した後の per-stage は常駐と無関係 = 何も名乗らない。
      assertEquals(await h.generate({ residency: "per-stage" }), 2);
      assertEquals(h.notices.length, 2);
    });
  });

  describe("④ OOM の退避と格下げ", () => {
    it("常駐 DiT の上で VAE 段が OOM したら、破棄 → 解放待ち → 名乗る → VAE 段を 1 回だけやり直す", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      h.log.length = 0;
      await h.generate({ residency: "transformer", faults: { vae: ["oom"] } });
      assertEquals(h.log, [
        "attempt:text",
        "body:1",
        "notice:retained/request",
        "attempt:vae",
        "dispose:1",
        "settle",
        "notice:evicted/out-of-memory",
        "attempt:vae",
      ]);
      assert(h.residency.downgraded);
      assertFalse(h.residency.holding);
    });

    it("格下げの後は transformer を求められても持たず、毎回 released / downgraded を名乗る", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      await h.generate({ residency: "transformer", faults: { text: ["oom"] } });
      const before = h.notices.length;
      await h.generate({ residency: "transformer" });
      await h.generate({ residency: "transformer" });
      assertEquals(h.notices.slice(before), [
        { action: "released", reason: "downgraded" },
        { action: "released", reason: "downgraded" },
      ]);
      assertFalse(h.residency.holding);
      // 格下げ後の DiT は毎回作って畳む（OOM を踏み直さない = 常駐の席に載せない）。
      assert(h.sessions.every((session) => session.disposed));
    });

    it("常駐 DiT の run が OOM したら退避して、DiT 段を段ごと運転で最初からやり直す", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      h.log.length = 0;
      const used = await h.generate({ residency: "transformer", faults: { transformer: ["oom"] } });
      assertEquals(used, 2, "やり直しが新しい Session で走っていない");
      assertEquals(h.log, [
        "attempt:text",
        "body:1",
        "dispose:1",
        "settle",
        "notice:evicted/out-of-memory",
        "open:2",
        "body:2",
        "dispose:2",
        "notice:released/downgraded",
        "attempt:vae",
      ]);
      assert(h.residency.downgraded);
    });

    it("やり直しも OOM なら、その例外をそのまま投げる（退避は 1 回だけ）", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { vae: ["oom", "oom"] } }),
        GpuOutOfMemoryError,
      );
      assertEquals(error.message, "vae: not enough memory left");
      assertEquals(h.log.filter((entry) => entry === "attempt:vae").length, 3);
      assertEquals(h.log.filter((entry) => entry === "settle").length, 1);
    });

    it("この generate で作った DiT の run の OOM は退避も格下げもせず、畳んで元の例外を投げる", async () => {
      // 持っている GPU 資源はこの DiT 1 本だけ（段ごと運転と同じ VRAM 構成）— 常駐が原因ではない。
      const h = createHarness();
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { transformer: ["oom"] } }),
        GpuOutOfMemoryError,
      );
      assertEquals(error.message, "transformer: not enough memory left");
      assertEquals(h.log, ["attempt:text", "open:1", "body:1", "dispose:1"]);
      assertEquals(h.notices, [], "席に載る前の失敗を名乗っている");
      assertFalse(h.residency.downgraded, "常駐と無関係な OOM で格下げした");
      assertFalse(h.residency.holding);
      // 格下げしていないので、次の generate はまた常駐を試みる。
      assertEquals(await h.generate({ residency: "transformer" }), 2);
      assert(h.residency.holding);
      assertEquals(h.notices, [{ action: "retained", reason: "request" }]);
    });

    it("退避の破棄が失敗しても、格下げと席の明け渡しが先に立ち、evicted / out-of-memory を名乗ってから投げる", async () => {
      const h = createHarness({ disposeFails: true });
      await h.generate({ residency: "transformer" });
      h.log.length = 0;
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { vae: ["oom"] } }),
        AggregateError,
      );
      assertEquals(
        error.errors.map((cause: Error) => cause.message),
        ["vae: not enough memory left", "dispose 1 に失敗"],
      );
      assertEquals(error.message, "anima: DiT の後始末が失敗した");
      // 破棄の瞬間に格下げは立ち、席は空いている（MUST の順序）。
      assertEquals(h.disposals, [{ id: 1, holding: false, downgraded: true }]);
      assertEquals(h.notices.at(-1), { action: "evicted", reason: "out-of-memory" });
      // 後始末が壊れた状態ではやり直さない（1 回目の試行だけ）。
      assertEquals(h.log.filter((entry) => entry === "attempt:vae").length, 1);
      assert(h.residency.downgraded);
      assertFalse(h.residency.holding);
    });

    it("解放待ちが reject しても evicted / out-of-memory を名乗り、元の OOM を先頭にした AggregateError で投げる", async () => {
      const h = createHarness({ settleFails: true });
      await h.generate({ residency: "transformer" });
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { text: ["oom"] } }),
        AggregateError,
      );
      assertEquals(
        error.errors.map((cause: Error) => cause.message),
        ["text: not enough memory left", "解放待ちに失敗"],
      );
      assertEquals(h.notices.at(-1), { action: "evicted", reason: "out-of-memory" });
      assertEquals(h.log.filter((entry) => entry === "attempt:text").length, 2, "やり直している");
      assert(h.residency.downgraded);
      assertFalse(h.residency.holding);
    });

    it("evicted / out-of-memory の通知で購読側が投げたら、やり直さずにその例外を投げる（格下げは残る）", async () => {
      // 通知の throw は中断の手段（onEvent の契約）。退避そのものは済んでいる。
      const reason = new Error("中止ボタン");
      const h = createHarness({
        notifyThrows: (notice) => notice.action === "evicted" ? reason : undefined,
      });
      await h.generate({ residency: "transformer" });
      h.log.length = 0;
      const error = await assertRejects(() =>
        h.generate({ residency: "transformer", faults: { vae: ["oom"] } })
      );
      assertStrictEquals(error, reason);
      assertEquals(h.log, [
        "attempt:text",
        "body:1",
        "notice:retained/request",
        "attempt:vae",
        "dispose:1",
        "settle",
        "notice:evicted/out-of-memory",
      ]);
      assert(h.residency.downgraded);
      assertFalse(h.residency.holding);
    });

    it("常駐 DiT が無ければ OOM は退避せずに投げ、格下げもしない", async () => {
      const h = createHarness();
      await assertRejects(
        () => h.generate({ residency: "per-stage", faults: { vae: ["oom"] } }),
        GpuOutOfMemoryError,
      );
      assertFalse(h.residency.downgraded);
      assertEquals(h.notices, []);
    });

    it("OOM 以外の失敗は他の段で起きても常駐 DiT を捨てない（退避は余力切れだけ）", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      await assertRejects(
        () => h.generate({ residency: "transformer", faults: { vae: ["failure"] } }),
        Error,
        "vae が失敗",
      );
      assert(h.residency.holding);
      assertFalse(h.residency.downgraded);
    });
  });

  describe("⑤ DiT 段の失敗", () => {
    it("常駐の席に載った DiT は捨てて evicted / failure を名乗り、元の例外をそのまま投げる", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { transformer: ["failure"] } }),
        Error,
      );
      assertEquals(error.message, "transformer が失敗");
      assert(h.sessions[0].disposed);
      assertFalse(h.residency.holding);
      assertFalse(h.residency.downgraded, "失敗の破棄は格下げしない");
      assertEquals(h.notices.at(-1), { action: "evicted", reason: "failure" });
      // 次の generate は作り直して持つ。
      assertEquals(await h.generate({ residency: "transformer" }), 2);
      assert(h.residency.holding);
    });

    it("通知の購読側が同じ例外を投げ直しても、生成が投げるのは元の例外 1 本（中断の識別を壊さない）", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      const reason = new Error("中止ボタン");
      const notified: ResidencyNotice[] = [];
      const error = await assertRejects(() =>
        h.residency.runTransformer({
          effective: "transformer",
          open: () => Promise.reject(new Error("持ち越した DiT があるのに作っている")),
          body: () => Promise.reject(reason),
          notify: (notice) => {
            notified.push(notice);
            return Promise.reject(reason);
          },
        })
      );
      assertStrictEquals(error, reason);
      assertEquals(notified, [{ action: "evicted", reason: "failure" }]);
      assertFalse(h.residency.holding);
    });

    it("後始末が失敗したら、元の例外を先頭にした AggregateError で運ぶ（元の失敗を上書きしない）", async () => {
      const h = createHarness({ disposeFails: true });
      await h.generate({ residency: "transformer" });
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { transformer: ["failure"] } }),
        AggregateError,
      );
      assertEquals(
        error.errors.map((cause: Error) => cause.message),
        ["transformer が失敗", "dispose 1 に失敗"],
      );
      assertFalse(h.residency.holding, "破棄に失敗した Session を席に残している");
    });

    it("席に載っていない DiT（段ごと運転 / この generate で作る DiT）の失敗は畳むだけで何も名乗らない", async () => {
      for (const residency of ["per-stage", "transformer"] as const) {
        const h = createHarness();
        await assertRejects(
          () => h.generate({ residency, faults: { transformer: ["failure"] } }),
          Error,
          "transformer が失敗",
        );
        assert(h.sessions[0].disposed, `${residency}: 畳んでいない`);
        assertEquals(h.notices, [], `${residency}: 名乗っている`);
        assertFalse(h.residency.holding);
      }
    });
  });

  describe("⑥ DiT 段より前で失敗した generate", () => {
    it("実効値 per-stage なら持ち越した DiT を手放して released / request を名乗り、元の例外を投げる", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      h.log.length = 0;
      await assertRejects(
        () => h.generate({ residency: "per-stage", faults: { text: ["failure"] } }),
        Error,
        "text が失敗",
      );
      assertEquals(h.log, ["attempt:text", "dispose:1", "notice:released/request"]);
      assertFalse(h.residency.holding);
      assertFalse(h.residency.downgraded);
      // 次の per-stage は常駐と無関係 = 何も名乗らない。
      await h.generate({ residency: "per-stage" });
      assertEquals(h.notices.at(-1), { action: "released", reason: "request" });
      assertEquals(h.notices.length, 2);
    });

    it("実効値 transformer なら持ったまま何も名乗らない", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      await assertRejects(
        () => h.generate({ residency: "transformer", faults: { text: ["failure"] } }),
        Error,
        "text が失敗",
      );
      assert(h.residency.holding);
      assertEquals(h.notices, [{ action: "retained", reason: "request" }]);
    });

    it("手放しの通知で購読側が同じ例外を投げ直しても元の例外 1 本、破棄の失敗は AggregateError に並べる", async () => {
      const reason = new Error("中止ボタン");
      const h = createHarness({
        notifyThrows: (notice) => notice.action === "released" ? reason : undefined,
      });
      await h.generate({ residency: "transformer" });
      // 返り値が投げるべき例外（pipeline の失敗経路は `throw await releaseIfRequested(...)`）。
      assertStrictEquals(
        await h.residency.releaseIfRequested(reason, "per-stage", h.notify),
        reason,
      );
      assertFalse(h.residency.holding);

      const broken = createHarness({ disposeFails: true });
      await broken.generate({ residency: "transformer" });
      const original = new Error("text が失敗");
      const error = await broken.residency.releaseIfRequested(original, "per-stage", broken.notify);
      assert(error instanceof AggregateError);
      assertStrictEquals(error.errors[0], original, "元の失敗を先頭に置いていない");
      assertEquals(error.errors.map((cause: Error) => cause.message), [
        "text が失敗",
        "dispose 1 に失敗",
      ]);
      assertFalse(broken.residency.holding, "破棄に失敗した Session を席に残している");
      assertEquals(broken.notices.at(-1), { action: "released", reason: "request" });
    });

    it("席が空なら何もせず元の例外を返す", async () => {
      const h = createHarness();
      const original = new Error("text が失敗");
      assertStrictEquals(
        await h.residency.releaseIfRequested(original, "per-stage", h.notify),
        original,
      );
      assertEquals(h.log, []);
    });
  });
});

describe("settleReleasedMemory（退避の後の解放待ち）", () => {
  it("onSubmittedWorkDone が解決すれば戻り、消失の購読を残さない", async () => {
    const { gpu } = losableGpuContext(undefined, {
      onSubmittedWorkDone: () => Promise.resolve(),
    });
    const baseline = gpu.pendingLostListeners;
    await settleReleasedMemory(gpu);
    assertEquals(gpu.pendingLostListeners, baseline, "消失の購読が積み残っている");
  });

  it("onSubmittedWorkDone が解決しなくても、device が失われたら待たずに戻る", async () => {
    // 消失後の onSubmittedWorkDone が解決しない実装への備え（MUST）。競わせないとここで固まる。
    const { gpu, lose } = losableGpuContext(undefined, {
      onSubmittedWorkDone: () => new Promise<void>(() => {}),
    });
    const baseline = gpu.pendingLostListeners;
    const settled = settleReleasedMemory(gpu);
    lose();
    await settled;
    assertEquals(gpu.pendingLostListeners, baseline);
  });

  it("既に失われた device では待たずに戻る", async () => {
    const { gpu, lose } = losableGpuContext(undefined, {
      onSubmittedWorkDone: () => new Promise<void>(() => {}),
    });
    lose();
    // 消失の通知は device.lost の解決（マイクロタスク）越しに届く。
    await Promise.resolve();
    assert(gpu.lost !== undefined, "前提: 消失を観測済み");
    await settleReleasedMemory(gpu);
  });
});

describe("disposeResidencyThenGpu（pipeline の dispose の本体）", () => {
  it("常駐 DiT を畳んでから GPU を破棄する（flush-before-destroy）", async () => {
    const log: string[] = [];
    await disposeResidencyThenGpu(
      { dispose: () => Promise.resolve(void log.push("residency")) },
      () => void log.push("gpu"),
    );
    assertEquals(log, ["residency", "gpu"]);
  });

  it("常駐 DiT の破棄が失敗しても内部で取った GPU は破棄し、失敗を投げる", async () => {
    const log: string[] = [];
    await assertRejects(
      () =>
        disposeResidencyThenGpu(
          { dispose: () => Promise.reject(new Error("常駐 DiT の破棄に失敗")) },
          () => void log.push("gpu"),
        ),
      Error,
      "常駐 DiT の破棄に失敗",
    );
    assertEquals(log, ["gpu"]);
  });

  it("共有 GPU（破棄の口なし）でも常駐 DiT は畳む", async () => {
    const h = createHarness();
    await h.generate({ residency: "transformer" });
    await disposeResidencyThenGpu(h.residency, undefined);
    assert(h.sessions[0].disposed);
    assertFalse(h.residency.holding);
  });
});
