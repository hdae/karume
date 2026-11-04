// The check of the decisions and the state machine of Anima's DiT residency (ADR 0112) (needs
// neither a GPU nor real assets).
//
// What the e2e on a real GPU (the "DiT residency" section of `e2e_anima_test.ts`) pins is that
// "residency moves not a single bit of the output pixels" and the observation of lifetimes; OOM
// eviction, downgrade and disposal on failure **cannot be provoked on purpose on a real machine**.
// So this file puts the pure functions and the state machine of `residency.ts` on a fake generate
// that calls them in the same order as the pipeline's `#generate` (text stage → DiT stage → VAE
// stage), injects faults (OOM / failure / cleanup failure) and pins the transitions.
//
// NOTE: `Pipeline.generate` itself cannot run on a fake GPU — a stage's Session uploads the real
// container's graph to the device, so construction does not pass on the device of `fake-gpu.ts`.
// The pipeline-side wiring (stage order, mapping to events) is checked by the e2e. The disposal
// order of the small GPU-side objects, `disposeResidencyThenGpu` (pipeline.ts), is pinned directly
// with a fake GpuContext. The check of the release wait `settleReleasedMemory` (shared layer) is in
// `settle_released_memory_test.ts` (the core lane).

import {
  assert,
  assertEquals,
  assertFalse,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import { GpuOutOfMemoryError, GpuValidationError } from "@karume/runtime";
import {
  type Residency,
  assertAnimaResidency,
  decideAfterFailedGenerate,
  decideAfterTransformer,
  evictsForMemory,
  generateMemoryNeed,
  type ResidencyNotice,
  stageMemoryNeed,
  TransformerResidency,
  transformerSource,
} from "../src/anima/residency.ts";
import { HEADROOM_MARGIN_BYTES } from "../src/session/headroom-margin.ts";
import { disposeResidencyThenGpu } from "../src/anima/pipeline.ts";
import { ModelInputError } from "../src/errors.ts";

/** A fake DiT Session (holds only whether it was disposed and which construction created it). */
type FakeSession = { readonly id: number; disposed: boolean };

/** A fault to inject (each consumed once per stage). */
type Fault = "oom" | "failure";

/** The request of one fake generate. */
type FakeRequest = {
  readonly residency: Residency;
  /** The faults per stage (one consumed from the front per attempt; success once they run out). */
  readonly faults?: Partial<Record<"text" | "transformer" | "vae", Fault[]>>;
  /**
   * The measuring points at which the trial allocation answers "does not fit" (the trigger of the
   * pipeline's pre-emptive eviction — by default everything fits).
   */
  readonly headroomShort?: readonly HeadroomPoint[];
};

/**
 * The two points the pipeline measures for pre-emptive eviction (before the text stage / after the
 * DiT stage, before the VAE stage).
 */
type HeadroomPoint = "text" | "vae";

/**
 * The need the harness passes per measuring point (the value itself means nothing — a mark that
 * tells which point measured).
 */
const TEXT_NEED = 2 * 1024 * 1024 * 1024;
const VAE_NEED = 1024 * 1024 * 1024;

const oom = (where: string): GpuOutOfMemoryError =>
  new GpuOutOfMemoryError(`${where}: not enough memory left`);

/** Cleanup-side faults to inject into the harness. */
type HarnessOptions = {
  readonly disposeFails?: boolean;
  readonly settleFails?: boolean;
  /**
   * The exception a notice subscriber throws (the returned value is thrown; undefined throws
   * nothing).
   */
  readonly notifyThrows?: (notice: ResidencyNotice) => unknown;
};

/**
 * A harness that runs the state machine in the same call order as the pipeline's `#generate`
 * (pre-emptive measurement → text stage → DiT stage → pre-emptive measurement → VAE stage — on
 * failure it throws through `releaseIfRequested`). `log` accumulates the GPU-side events
 * (construction / disposal / release wait / stage attempts) and the notices in time order.
 * `disposals` is the state of the state machine at the moment of each disposal (the observation
 * point of the MUST ordering).
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
  /**
   * The points and needs at which the pre-emptive eviction probe was called (kept out of `log` —
   * the existing sequences with residency stay unchanged).
   */
  const probes: { readonly point: HeadroomPoint; readonly need: number }[] = [];
  const residency: TransformerResidency<FakeSession> = new TransformerResidency<FakeSession>({
    dispose: (session) => {
      session.disposed = true;
      log.push(`dispose:${session.id}`);
      disposals.push({
        id: session.id,
        holding: residency.holding,
        downgraded: residency.downgraded,
      });
      if (options.disposeFails === true) throw new Error(`dispose ${session.id} failed`);
    },
    settleRelease: () => {
      log.push("settle");
      return options.settleFails === true
        ? Promise.reject(new Error("the release wait failed"))
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

  /** One attempt of a per-stage stage (text / VAE). Throws if there is a fault. */
  const attempt = (request: FakeRequest, stage: "text" | "vae") => (): Promise<string> => {
    log.push(`attempt:${stage}`);
    const fault = takeFault(request, stage);
    if (fault === "oom") return Promise.reject(oom(stage));
    if (fault === "failure") return Promise.reject(new Error(`${stage} failed`));
    return Promise.resolve(stage);
  };

  /**
   * Measures one point the same way as the pipeline (calls `ensureHeadroom` only when a resident
   * DiT exists).
   */
  const ensureHeadroom = async (
    request: FakeRequest,
    point: HeadroomPoint,
    need: number,
  ): Promise<void> => {
    if (!residency.holding) return;
    await residency.ensureHeadroom(
      need,
      (bytes) => {
        probes.push({ point, need: bytes });
        return Promise.resolve(!(request.headroomShort?.includes(point) ?? false));
      },
      notify,
    );
  };

  const generate = async (request: FakeRequest): Promise<number> => {
    try {
      await ensureHeadroom(request, "text", TEXT_NEED);
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
          assertFalse(session.disposed, "a stage is running on a disposed Session");
          const fault = takeFault(request, "transformer");
          if (fault === "oom") return Promise.reject(oom("transformer"));
          if (fault === "failure") return Promise.reject(new Error("transformer failed"));
          return Promise.resolve(session.id);
        },
        notify,
      });
      await ensureHeadroom(request, "vae", VAE_NEED);
      await residency.runStage(attempt(request, "vae"), notify);
      return used;
    } catch (error) {
      throw await residency.releaseIfRequested(error, request.residency, notify);
    }
  };

  return { log, sessions, notices, disposals, probes, residency, notify, generate };
};

describe("assertAnimaResidency", () => {
  it("accepts exactly two spellings", () => {
    assertEquals(assertAnimaResidency("per-stage", "residency"), "per-stage");
    assertEquals(assertAnimaResidency("transformer", "residency"), "transformer");
  });

  it("an unknown spelling is not degraded to the default; it fails with a ModelInputError naming the origin and the actual value", () => {
    // If a misspelling silently became per-stage operation, one would notice that the opt-in has no
    // effect only through speed.
    for (const [value, actual] of [
      ["transformers", "'transformers'"],
      [null, "null"],
      [1, "1"],
    ]) {
      assertThrows(
        () => assertAnimaResidency(value, "residency"),
        ModelInputError,
        `residency: expected 'per-stage' / 'transformer' (actual ${actual})`,
      );
    }
  });
});

describe("transformerSource (the origin of the DiT stage's Session)", () => {
  it("if a resident DiT exists it is used regardless of the effective value or a downgrade (no re-read)", () => {
    for (const effective of ["per-stage", "transformer"] as const) {
      assertEquals(transformerSource({ effective, downgraded: false, held: true }), "resident");
    }
  });

  it("without a resident DiT, it is built into the resident slot only for transformer without a downgrade", () => {
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

describe("decideAfterTransformer (the decision when leaving the DiT stage)", () => {
  it("with the default per-stage operation and nothing to do with residency, it only releases and announces nothing (the default event sequence is unchanged)", () => {
    assertEquals(
      decideAfterTransformer({ effective: "per-stage", downgraded: false, held: false }, "success"),
      { action: "release" },
    );
  });

  it("with transformer and an empty slot (a DiT built in this generate), it puts it in the slot after the successful stage and announces retained / request", () => {
    assertEquals(
      decideAfterTransformer(
        { effective: "transformer", downgraded: false, held: false },
        "success",
      ),
      { action: "retain", notice: { action: "retained", reason: "request" } },
    );
  });

  it("a DiT in the slot while already downgraded (an unreachable pair) throws instead of falling to a default", () => {
    // A downgrade is set only when an eviction empties the slot, and nothing is put in the slot
    // after that. Silently returning something would leave a value that contradicts "a decision not
    // to keep a resident DiT is always announced".
    for (const effective of ["per-stage", "transformer"] as const) {
      for (const outcome of ["success", "failure"] as const) {
        assertThrows(
          () => decideAfterTransformer({ effective, downgraded: true, held: true }, outcome),
          Error,
          "unreachable state",
        );
      }
    }
  });

  it("with transformer and a DiT in the slot, it keeps it and announces retained / request", () => {
    assertEquals(
      decideAfterTransformer(
        { effective: "transformer", downgraded: false, held: true },
        "success",
      ),
      { action: "retain", notice: { action: "retained", reason: "request" } },
    );
  });

  it("when a carried-over resident DiT is used by a per-stage request it is released and released / request is announced", () => {
    assertEquals(
      decideAfterTransformer({ effective: "per-stage", downgraded: false, held: true }, "success"),
      { action: "release", notice: { action: "released", reason: "request" } },
    );
  });

  it("when transformer is requested after a downgrade it keeps nothing and announces released / downgraded", () => {
    assertEquals(
      decideAfterTransformer(
        { effective: "transformer", downgraded: true, held: false },
        "success",
      ),
      { action: "release", notice: { action: "released", reason: "downgraded" } },
    );
  });

  it("the DiT of a failed stage is discarded with evicted / failure announced when carried over, and only disposed when built in this generate", () => {
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

describe("decideAfterFailedGenerate (when a generate fails outside the DiT stage)", () => {
  it("with the effective value per-stage it releases the slot's DiT and announces released / request (the instruction takes effect even on failure)", () => {
    assertEquals(decideAfterFailedGenerate("per-stage"), {
      action: "release",
      notice: { action: "released", reason: "request" },
    });
  });

  it("with the effective value transformer it keeps it and announces nothing", () => {
    assertEquals(decideAfterFailedGenerate("transformer"), { action: "retain" });
  });
});

describe("evictsForMemory (whether a failure calls for an eviction)", () => {
  it("evicts only when a resident DiT exists and the failure is a GpuOutOfMemoryError", () => {
    assert(evictsForMemory(oom("x"), true));
    assertFalse(evictsForMemory(oom("x"), false), "nothing to evict");
    // Retrying a validation error or a plain Error through an eviction would bury the real cause
    // under the second failure.
    assertFalse(evictsForMemory(new GpuValidationError("x"), true));
    assertFalse(
      evictsForMemory(new Error("out-of-memory"), true),
      "it does not judge by the wording",
    );
  });
});

describe("stageMemoryNeed / generateMemoryNeed (the need for pre-emptive eviction)", () => {
  const MIB = 1024 * 1024;

  it("the need of one stage is the estimated peak + the staging of the largest part + a 512 MiB margin", () => {
    assertEquals(HEADROOM_MARGIN_BYTES, 512 * MIB);
    // The shape of the text_encoder measurement (weights 1,138.5 MiB + retention budget 256 MiB,
    // part 256 MiB).
    assertEquals(
      stageMemoryNeed({ peakAccountedBytes: 1394 * MIB }, 256 * MIB),
      (1394 + 256 + 512) * MIB,
    );
    // In a container with only part 0 (no weight parts) the staging term is 0.
    assertEquals(stageMemoryNeed({ peakAccountedBytes: 10 * MIB }, 0), (10 + 512) * MIB);
  });

  it("the need of a generate is the maximum over the stages (stages are opened and disposed in turn, so not the sum)", () => {
    assertEquals(generateMemoryNeed([2162 * MIB, 1281 * MIB, 1212 * MIB]), 2162 * MIB);
    assertEquals(generateMemoryNeed([1 * MIB, 3 * MIB, 2 * MIB]), 3 * MIB);
  });
});

describe("TransformerResidency (the state machine run on a fake generate)", () => {
  describe("① the default per-stage", () => {
    it("builds the DiT per generate, disposes it at the end of the stage and announces nothing", async () => {
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
    it("builds it in the DiT stage of the first generate and afterwards uses the same Session without re-reading", async () => {
      const h = createHarness();
      assertEquals(await h.generate({ residency: "transformer" }), 1);
      assertEquals(await h.generate({ residency: "transformer" }), 1);
      assertEquals(h.sessions.length, 1, "the second generate rebuilt the DiT");
      assertFalse(h.sessions[0].disposed);
      assert(h.residency.holding);
      assertEquals(h.notices, [
        { action: "retained", reason: "request" },
        { action: "retained", reason: "request" },
      ]);
    });

    it("dispose disposes the resident DiT (the second call does nothing)", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      await h.residency.dispose();
      assert(h.sessions[0].disposed);
      assertFalse(h.residency.holding);
      await h.residency.dispose();
      assertEquals(
        h.log.filter((entry) => entry.startsWith("dispose")),
        ["dispose:1"],
      );
    });
  });

  describe("③ override by the request", () => {
    it("a DiT kept by transformer is used by the next per-stage and released after it (the last image of a series)", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      assertEquals(
        await h.generate({ residency: "per-stage" }),
        1,
        "the carried-over DiT was not used",
      );
      assert(h.sessions[0].disposed);
      assertFalse(h.residency.holding);
      assertEquals(h.notices, [
        { action: "retained", reason: "request" },
        { action: "released", reason: "request" },
      ]);
      // A per-stage after the release has nothing to do with residency = announces nothing.
      assertEquals(await h.generate({ residency: "per-stage" }), 2);
      assertEquals(h.notices.length, 2);
    });
  });

  describe("④ OOM eviction and downgrade", () => {
    it("when the VAE stage hits an OOM on top of a resident DiT: dispose → release wait → announce → retry the VAE stage exactly once", async () => {
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

    it("after a downgrade it keeps nothing even when transformer is requested, and announces released / downgraded every time", async () => {
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
      // After the downgrade the DiT is built and disposed every time (no hitting the OOM again =
      // not put in the resident slot).
      assert(h.sessions.every((session) => session.disposed));
    });

    it("when the resident DiT's run hits an OOM it evicts and redoes the DiT stage from the start with per-stage operation", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      h.log.length = 0;
      const used = await h.generate({ residency: "transformer", faults: { transformer: ["oom"] } });
      assertEquals(used, 2, "the retry did not run on a new Session");
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

    it("if the retry also hits an OOM, that exception is thrown as is (only one eviction)", async () => {
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

    it("an OOM in the run of a DiT built in this generate neither evicts nor downgrades; it disposes and throws the original exception", async () => {
      // The only GPU resource held is this one DiT (the same VRAM layout as per-stage operation) —
      // residency is not the cause.
      const h = createHarness();
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { transformer: ["oom"] } }),
        GpuOutOfMemoryError,
      );
      assertEquals(error.message, "transformer: not enough memory left");
      assertEquals(h.log, ["attempt:text", "open:1", "body:1", "dispose:1"]);
      assertEquals(h.notices, [], "it announces a failure that happened before entering the slot");
      assertFalse(h.residency.downgraded, "it downgraded on an OOM unrelated to residency");
      assertFalse(h.residency.holding);
      // It did not downgrade, so the next generate tries residency again.
      assertEquals(await h.generate({ residency: "transformer" }), 2);
      assert(h.residency.holding);
      assertEquals(h.notices, [{ action: "retained", reason: "request" }]);
    });

    it("even when the eviction's disposal fails, the downgrade and giving up the slot happen first, and it throws after announcing evicted / out-of-memory", async () => {
      const h = createHarness({ disposeFails: true });
      await h.generate({ residency: "transformer" });
      h.log.length = 0;
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { vae: ["oom"] } }),
        AggregateError,
      );
      assertEquals(
        error.errors.map((cause: Error) => cause.message),
        ["vae: not enough memory left", "dispose 1 failed"],
      );
      assertEquals(error.message, "anima: DiT clean-up failed");
      // At the moment of disposal the downgrade is set and the slot is empty (the MUST ordering).
      assertEquals(h.disposals, [{ id: 1, holding: false, downgraded: true }]);
      assertEquals(h.notices.at(-1), { action: "evicted", reason: "out-of-memory" });
      // With a broken cleanup it does not retry (only the first attempt).
      assertEquals(h.log.filter((entry) => entry === "attempt:vae").length, 1);
      assert(h.residency.downgraded);
      assertFalse(h.residency.holding);
    });

    it("even when the release wait rejects, it announces evicted / out-of-memory and throws an AggregateError headed by the original OOM", async () => {
      const h = createHarness({ settleFails: true });
      await h.generate({ residency: "transformer" });
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { text: ["oom"] } }),
        AggregateError,
      );
      assertEquals(
        error.errors.map((cause: Error) => cause.message),
        ["text: not enough memory left", "the release wait failed"],
      );
      assertEquals(h.notices.at(-1), { action: "evicted", reason: "out-of-memory" });
      assertEquals(h.log.filter((entry) => entry === "attempt:text").length, 2, "it retries");
      assert(h.residency.downgraded);
      assertFalse(h.residency.holding);
    });

    it("when a subscriber throws on the evicted / out-of-memory notice, it throws that exception without retrying (the downgrade remains)", async () => {
      // Throwing from a notice is the means of aborting (the onEvent contract). The eviction itself
      // is done.
      const reason = new Error("cancel button");
      const h = createHarness({
        notifyThrows: (notice) => (notice.action === "evicted" ? reason : undefined),
      });
      await h.generate({ residency: "transformer" });
      h.log.length = 0;
      const error = await assertRejects(() =>
        h.generate({ residency: "transformer", faults: { vae: ["oom"] } }),
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

    it("without a resident DiT an OOM throws without eviction and does not downgrade", async () => {
      const h = createHarness();
      await assertRejects(
        () => h.generate({ residency: "per-stage", faults: { vae: ["oom"] } }),
        GpuOutOfMemoryError,
      );
      assertFalse(h.residency.downgraded);
      assertEquals(h.notices, []);
    });

    it("a failure other than OOM does not discard the resident DiT even in another stage (eviction is only for running out of memory)", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      await assertRejects(
        () => h.generate({ residency: "transformer", faults: { vae: ["failure"] } }),
        Error,
        "vae failed",
      );
      assert(h.residency.holding);
      assertFalse(h.residency.downgraded);
    });
  });

  describe("④' pre-emptive eviction (ensureHeadroom)", () => {
    const NEED = 3 * 1024 * 1024 * 1024;

    /**
     * A probe that records the needs it is called with and returns a fixed answer (rejects when
     * `answer` is an Error).
     */
    const recordingProbe = (answer: boolean | Error) => {
      const calls: number[] = [];
      const probe = (bytes: number): Promise<boolean> => {
        calls.push(bytes);
        return answer instanceof Error ? Promise.reject(answer) : Promise.resolve(answer);
      };
      return { calls, probe };
    };

    /** A harness whose first generate put the DiT in the slot (`log` / `notices` emptied). */
    const holdingHarness = async (options: HarnessOptions = {}) => {
      const h = createHarness(options);
      await h.generate({ residency: "transformer" });
      assert(h.residency.holding);
      h.log.length = 0;
      h.notices.length = 0;
      return h;
    };

    describe("ensureHeadroom on its own", () => {
      it("without a resident DiT it does not call the probe, announces nothing and does not downgrade", async () => {
        const h = createHarness();
        const { calls, probe } = recordingProbe(false);
        await h.residency.ensureHeadroom(NEED, probe, h.notify);
        assertEquals(calls, []);
        assertEquals(h.notices, []);
        assertEquals(h.log, []);
        assertFalse(h.residency.downgraded);
      });

      it("if it fits, it calls the probe exactly once with the need, announces nothing and keeps the resident DiT", async () => {
        const h = await holdingHarness();
        const { calls, probe } = recordingProbe(true);
        await h.residency.ensureHeadroom(NEED, probe, h.notify);
        assertEquals(calls, [NEED]);
        assertEquals(h.notices, []);
        assertEquals(h.log, []);
        assert(h.residency.holding);
        assertFalse(h.residency.downgraded);
      });

      it("if it does not fit, it evicts in the order dispose → release wait → evicted / headroom, and downgrades to per-stage operation afterwards", async () => {
        const h = await holdingHarness();
        const { calls, probe } = recordingProbe(false);
        await h.residency.ensureHeadroom(NEED, probe, h.notify);
        assertEquals(calls, [NEED]);
        assertEquals(h.log, ["dispose:1", "settle", "notice:evicted/headroom"]);
        // At the moment of disposal the downgrade is set and the slot is empty (the MUST ordering).
        assertEquals(h.disposals.at(-1), { id: 1, holding: false, downgraded: true });
        assertFalse(h.residency.holding);
        assert(h.residency.downgraded);
        assertEquals(
          transformerSource({
            effective: "transformer",
            downgraded: h.residency.downgraded,
            held: h.residency.holding,
          }),
          "per-stage",
        );
      });

      it("if the probe throws it throws as is, neither downgrading nor disposing, and keeps the resident DiT (does not fit ≠ failure)", async () => {
        const h = await holdingHarness();
        const failure = new GpuValidationError("probe: validation");
        const { calls, probe } = recordingProbe(failure);
        const error = await assertRejects(() => h.residency.ensureHeadroom(NEED, probe, h.notify));
        assertStrictEquals(error, failure);
        assertEquals(calls, [NEED]);
        assertEquals(h.log, []);
        assertEquals(h.notices, []);
        assert(h.residency.holding);
        assertFalse(h.residency.downgraded);
      });

      it("even when the eviction's disposal fails it announces evicted / headroom and throws an AggregateError listing only the cleanup failures", async () => {
        const h = await holdingHarness({ disposeFails: true });
        const { probe } = recordingProbe(false);
        const error = await assertRejects(
          () => h.residency.ensureHeadroom(NEED, probe, h.notify),
          AggregateError,
        );
        // There is no original failure — only the cleanup failures are listed (in an OOM eviction
        // the original OOM comes first).
        assertEquals(
          error.errors.map((cause: Error) => cause.message),
          ["dispose 1 failed"],
        );
        assertEquals(error.message, "anima: DiT clean-up failed");
        assertEquals(h.log, ["dispose:1", "settle", "notice:evicted/headroom"]);
        assert(h.residency.downgraded);
        assertFalse(h.residency.holding);
      });

      it("even when the release wait rejects it announces evicted / headroom and throws that failure in an AggregateError", async () => {
        const h = await holdingHarness({ settleFails: true });
        const { probe } = recordingProbe(false);
        const error = await assertRejects(
          () => h.residency.ensureHeadroom(NEED, probe, h.notify),
          AggregateError,
        );
        assertEquals(
          error.errors.map((cause: Error) => cause.message),
          ["the release wait failed"],
        );
        assertEquals(h.notices, [{ action: "evicted", reason: "headroom" }]);
        assert(h.residency.downgraded);
        assertFalse(h.residency.holding);
      });

      it("when a notice failure adds to a cleanup failure, both are listed in the AggregateError", async () => {
        const h = await holdingHarness({
          settleFails: true,
          notifyThrows: (notice) =>
            notice.action === "evicted" ? new Error("subscriber failure") : undefined,
        });
        const { probe } = recordingProbe(false);
        const error = await assertRejects(
          () => h.residency.ensureHeadroom(NEED, probe, h.notify),
          AggregateError,
        );
        assertEquals(
          error.errors.map((cause: Error) => cause.message),
          ["the release wait failed", "subscriber failure"],
        );
      });

      it("when the cleanup succeeds and a subscriber throws on the evicted / headroom notice, that exception is thrown as is (the downgrade remains)", async () => {
        const reason = new Error("cancel button");
        const h = await holdingHarness({
          notifyThrows: (notice) => (notice.action === "evicted" ? reason : undefined),
        });
        const { probe } = recordingProbe(false);
        const error = await assertRejects(() => h.residency.ensureHeadroom(NEED, probe, h.notify));
        assertStrictEquals(error, reason);
        assertEquals(h.log, ["dispose:1", "settle", "notice:evicted/headroom"]);
        assert(h.residency.downgraded);
        assertFalse(h.residency.holding);
      });
    });

    describe("the two measuring points inside generate", () => {
      it("if it does not fit before the text stage, it evicts before opening the stage, rebuilds the DiT stage with per-stage operation and announces downgraded", async () => {
        const h = createHarness();
        await h.generate({ residency: "transformer" });
        h.log.length = 0;
        h.probes.length = 0;
        assertEquals(await h.generate({ residency: "transformer", headroomShort: ["text"] }), 2);
        assertEquals(h.log, [
          "dispose:1",
          "settle",
          "notice:evicted/headroom",
          "attempt:text",
          "open:2",
          "body:2",
          "dispose:2",
          "notice:released/downgraded",
          "attempt:vae",
        ]);
        // The measurement before the text stage uses the text-side need, and after the eviction
        // nothing is measured before the VAE stage (the slot is empty).
        assertEquals(h.probes, [{ point: "text", need: TEXT_NEED }]);
      });

      it("when a generate with the effective value per-stage evicts before the text stage, the DiT stage announces nothing", async () => {
        const h = createHarness();
        await h.generate({ residency: "transformer" });
        h.notices.length = 0;
        await h.generate({ residency: "per-stage", headroomShort: ["text"] });
        assertEquals(h.notices, [{ action: "evicted", reason: "headroom" }]);
        assert(h.residency.downgraded);
      });

      it("even in the first generate, it evicts if it does not fit before the VAE stage after putting the DiT in the slot", async () => {
        const h = createHarness();
        assertEquals(await h.generate({ residency: "transformer", headroomShort: ["vae"] }), 1);
        assertEquals(h.log, [
          "attempt:text",
          "open:1",
          "body:1",
          "notice:retained/request",
          "dispose:1",
          "settle",
          "notice:evicted/headroom",
          "attempt:vae",
        ]);
        // Nothing is measured before the text stage while the slot is empty; only before the VAE
        // stage, with the VAE need.
        assertEquals(h.probes, [{ point: "vae", need: VAE_NEED }]);
        assert(h.residency.downgraded);
        assertFalse(h.residency.holding);
      });

      it("a generate with a carried-over DiT measures at both points and evicts there if it does not fit before the VAE stage", async () => {
        const h = createHarness();
        await h.generate({ residency: "transformer" });
        h.log.length = 0;
        h.probes.length = 0;
        assertEquals(await h.generate({ residency: "transformer", headroomShort: ["vae"] }), 1);
        assertEquals(h.probes, [
          { point: "text", need: TEXT_NEED },
          { point: "vae", need: VAE_NEED },
        ]);
        assertEquals(h.log, [
          "attempt:text",
          "body:1",
          "notice:retained/request",
          "dispose:1",
          "settle",
          "notice:evicted/headroom",
          "attempt:vae",
        ]);
      });

      it("after an eviction the downgrade remains: even when transformer is requested it keeps no resident DiT and measures nothing", async () => {
        const h = createHarness();
        await h.generate({ residency: "transformer" });
        await h.generate({ residency: "transformer", headroomShort: ["text"] });
        h.notices.length = 0;
        h.probes.length = 0;
        await h.generate({ residency: "transformer" });
        assertEquals(h.notices, [{ action: "released", reason: "downgraded" }]);
        assertEquals(h.probes, []);
        assertFalse(h.residency.holding);
      });

      it("after an eviction with a broken cleanup it does not proceed to the stage (before the text stage)", async () => {
        const h = createHarness({ settleFails: true });
        await h.generate({ residency: "transformer" });
        h.log.length = 0;
        await assertRejects(
          () => h.generate({ residency: "transformer", headroomShort: ["text"] }),
          AggregateError,
        );
        assertEquals(h.log, ["dispose:1", "settle", "notice:evicted/headroom"]);
      });
    });
  });

  describe("⑤ failure of the DiT stage", () => {
    it("a DiT in the resident slot is discarded with evicted / failure announced, and the original exception is thrown as is", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { transformer: ["failure"] } }),
        Error,
      );
      assertEquals(error.message, "transformer failed");
      assert(h.sessions[0].disposed);
      assertFalse(h.residency.holding);
      assertFalse(h.residency.downgraded, "a disposal on failure does not downgrade");
      assertEquals(h.notices.at(-1), { action: "evicted", reason: "failure" });
      // The next generate rebuilds it and keeps it.
      assertEquals(await h.generate({ residency: "transformer" }), 2);
      assert(h.residency.holding);
    });

    it("even when the notice subscriber rethrows the same exception, the generation throws only the original one (abort identification stays intact)", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      const reason = new Error("cancel button");
      const notified: ResidencyNotice[] = [];
      const error = await assertRejects(() =>
        h.residency.runTransformer({
          effective: "transformer",
          open: () => Promise.reject(new Error("building although a carried-over DiT exists")),
          body: () => Promise.reject(reason),
          notify: (notice) => {
            notified.push(notice);
            return Promise.reject(reason);
          },
        }),
      );
      assertStrictEquals(error, reason);
      assertEquals(notified, [{ action: "evicted", reason: "failure" }]);
      assertFalse(h.residency.holding);
    });

    it("if the cleanup fails, it carries an AggregateError headed by the original exception (the original failure is not overwritten)", async () => {
      const h = createHarness({ disposeFails: true });
      await h.generate({ residency: "transformer" });
      const error = await assertRejects(
        () => h.generate({ residency: "transformer", faults: { transformer: ["failure"] } }),
        AggregateError,
      );
      assertEquals(
        error.errors.map((cause: Error) => cause.message),
        ["transformer failed", "dispose 1 failed"],
      );
      assertFalse(h.residency.holding, "a Session whose disposal failed is left in the slot");
    });

    it("the failure of a DiT not in the slot (per-stage operation / a DiT built in this generate) only disposes it and announces nothing", async () => {
      for (const residency of ["per-stage", "transformer"] as const) {
        const h = createHarness();
        await assertRejects(
          () => h.generate({ residency, faults: { transformer: ["failure"] } }),
          Error,
          "transformer failed",
        );
        assert(h.sessions[0].disposed, `${residency}: not disposed`);
        assertEquals(h.notices, [], `${residency}: announces something`);
        assertFalse(h.residency.holding);
      }
    });
  });

  describe("⑥ a generate that failed before the DiT stage", () => {
    it("with the effective value per-stage it releases the carried-over DiT, announces released / request and throws the original exception", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      h.log.length = 0;
      await assertRejects(
        () => h.generate({ residency: "per-stage", faults: { text: ["failure"] } }),
        Error,
        "text failed",
      );
      assertEquals(h.log, ["attempt:text", "dispose:1", "notice:released/request"]);
      assertFalse(h.residency.holding);
      assertFalse(h.residency.downgraded);
      // The next per-stage has nothing to do with residency = announces nothing.
      await h.generate({ residency: "per-stage" });
      assertEquals(h.notices.at(-1), { action: "released", reason: "request" });
      assertEquals(h.notices.length, 2);
    });

    it("with the effective value transformer it keeps it and announces nothing", async () => {
      const h = createHarness();
      await h.generate({ residency: "transformer" });
      await assertRejects(
        () => h.generate({ residency: "transformer", faults: { text: ["failure"] } }),
        Error,
        "text failed",
      );
      assert(h.residency.holding);
      assertEquals(h.notices, [{ action: "retained", reason: "request" }]);
    });

    it("even when a subscriber rethrows the same exception on the release notice there is only the original exception, and a disposal failure is listed in an AggregateError", async () => {
      const reason = new Error("cancel button");
      const h = createHarness({
        notifyThrows: (notice) => (notice.action === "released" ? reason : undefined),
      });
      await h.generate({ residency: "transformer" });
      // The return value is the exception to throw (the pipeline's failure path is `throw await
      // releaseIfRequested(...)`).
      assertStrictEquals(
        await h.residency.releaseIfRequested(reason, "per-stage", h.notify),
        reason,
      );
      assertFalse(h.residency.holding);

      const broken = createHarness({ disposeFails: true });
      await broken.generate({ residency: "transformer" });
      const original = new Error("text failed");
      const error = await broken.residency.releaseIfRequested(original, "per-stage", broken.notify);
      assert(error instanceof AggregateError);
      assertStrictEquals(error.errors[0], original, "the original failure is not placed first");
      assertEquals(
        error.errors.map((cause: Error) => cause.message),
        ["text failed", "dispose 1 failed"],
      );
      assertFalse(broken.residency.holding, "a Session whose disposal failed is left in the slot");
      assertEquals(broken.notices.at(-1), { action: "released", reason: "request" });
    });

    it("with an empty slot it does nothing and returns the original exception", async () => {
      const h = createHarness();
      const original = new Error("text failed");
      assertStrictEquals(
        await h.residency.releaseIfRequested(original, "per-stage", h.notify),
        original,
      );
      assertEquals(h.log, []);
    });
  });
});

describe("disposeResidencyThenGpu (the body of the pipeline's dispose)", () => {
  it("disposes the resident DiT before destroying the GPU (flush-before-destroy)", async () => {
    const log: string[] = [];
    await disposeResidencyThenGpu(
      { dispose: () => Promise.resolve(void log.push("residency")) },
      () => void log.push("gpu"),
    );
    assertEquals(log, ["residency", "gpu"]);
  });

  it("even when disposing the resident DiT fails, it destroys the GPU it acquired internally and throws the failure", async () => {
    const log: string[] = [];
    await assertRejects(
      () =>
        disposeResidencyThenGpu(
          { dispose: () => Promise.reject(new Error("disposing the resident DiT failed")) },
          () => void log.push("gpu"),
        ),
      Error,
      "disposing the resident DiT failed",
    );
    assertEquals(log, ["gpu"]);
  });

  it("disposes the resident DiT even with a shared GPU (no destroy hook)", async () => {
    const h = createHarness();
    await h.generate({ residency: "transformer" });
    await disposeResidencyThenGpu(h.residency, undefined);
    assert(h.sessions[0].disposed);
    assertFalse(h.residency.holding);
  });
});
