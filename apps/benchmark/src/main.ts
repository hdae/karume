/**
 * The page: the GPU it found, the note before a run, the three checkboxes, Start and Cancel; while a
 * run goes, the two bars, the time left and the patterns found; at the end, the download. The DOM
 * is only here: what the page says and does is decided by the pure modules beside it (`notes.ts`,
 * `plan.ts`, `progress.ts`, `controller.ts`), which are tested without a page. The run's events are
 * stored as they come and folded and painted on the next animation frame.
 *
 * SPEC: docs/spec/benchmark.md
 */

import "./style.css";

import { gemvSweepSession } from "@karume/runtime/tune";
import { readEnvironment } from "@karume/sweep-app-support/environment";
import { type Checkout, errorText, messageText } from "@karume/sweep-app-support/record";

import { type RunEnding, runBenchmark, type StampedEvent } from "./controller.ts";
import { downloadText, element, setText } from "./dom.ts";
import { remainingEstimate } from "./estimate.ts";
import { afterRunNotes, beforeRunNotes, deviceView, type FoundAdapter } from "./notes.ts";
import {
  type BenchmarkPlan,
  planBenchmark,
  type SweepChoice,
  type Trial,
  trialOfQuery,
} from "./plan.ts";
import {
  foldProgress,
  INITIAL_PROGRESS,
  patternLines,
  type ProgressState,
  type ProgressView,
  progressView,
} from "./progress.ts";
import { hiddenLog } from "./visibility.ts";
import { holdWakeLock } from "./wake-lock.ts";

/** The checkout the page was built from (the config reads it when it loads). */
declare const __BENCHMARK_CHECKOUT__: Checkout;

const HONESTY =
  "Where this run held steady, a table can take these from it alone; where it did not, only after a second run on this machine agrees.";

/** The adapter, or why the page cannot run on this browser. */
const probeGpu = async (): Promise<FoundAdapter | string> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined)
    return "This browser has no WebGPU, so the page cannot measure anything here.";
  try {
    const adapter = await gpu.requestAdapter();
    if (adapter === null) return "WebGPU gives no GPU adapter in this browser.";
    const { vendor, architecture, description } = adapter.info;
    return {
      vendor,
      architecture,
      description,
      timestamps: adapter.features.has("timestamp-query"),
    };
  } catch (error) {
    return `WebGPU failed to give a GPU adapter: ${errorText(error)}`;
  }
};

const bar = (): {
  readonly root: HTMLElement;
  readonly meter: HTMLProgressElement;
  readonly text: HTMLElement;
} => {
  const meter = element("progress", { max: 1, value: 0 });
  const text = element("p", { className: "bar-text" });
  return { root: element("div", { className: "bar" }, [meter, text]), meter, text };
};

const paintBar = (
  target: ReturnType<typeof bar>,
  value: { readonly done: number; readonly total: number; readonly text: string },
): void => {
  target.meter.max = Math.max(1, value.total);
  target.meter.value = Math.min(value.done, target.meter.max);
  setText(target.text, value.text);
};

const main = async (): Promise<void> => {
  // The page load's session: every GEMV record of its runs names it
  const session = gemvSweepSession();
  const app = document.querySelector<HTMLElement>("#app");
  if (app === null) throw new Error("The page has no #app element");
  const checkout = __BENCHMARK_CHECKOUT__;

  const device = element("dl", { className: "device" });
  const deviceNotes = element("div");
  const notes = element("ul", { className: "notes" });
  const choice: Record<keyof SweepChoice, boolean> = { gemv: true, geometry: true, grid: true };
  const gemvBox = element("input", { type: "checkbox", checked: true });
  const geometryBox = element("input", { type: "checkbox", checked: true });
  const gridBox = element("input", { type: "checkbox", checked: true });
  const start = element("button", { type: "button", textContent: "Start", disabled: true });
  const cancel = element("button", { type: "button", textContent: "Cancel", disabled: true });
  const problem = element("p", { className: "problem", hidden: true });

  const overall = bar();
  const caseTitle = element("p", { className: "case-title" });
  const caseBar = bar();
  const caseDetail = element("p", { className: "detail" });
  const remaining = element("p", { className: "remaining" });
  const warning = element("p", { className: "problem", hidden: true });
  const hiddenNote = element("p", { className: "warning", hidden: true });
  const patterns = element("ul", { className: "patterns" });
  const run = element("section", { hidden: true }, [
    element("h2", { textContent: "Progress" }),
    element("p", { className: "label", textContent: "The whole run" }),
    overall.root,
    element("p", { className: "label", textContent: "The case being measured" }),
    caseTitle,
    caseBar.root,
    caseDetail,
    remaining,
    warning,
    hiddenNote,
    element("h3", { textContent: "Optimisable patterns found" }),
    patterns,
    element("p", { className: "detail", textContent: HONESTY }),
  ]);
  const endText = element("p");
  const endNotes = element("div");
  const download = element("button", { type: "button" });
  const sendNote = element("p", {
    textContent: "Send the downloaded file to the maintainer. Thank you!",
  });
  const ending = element("section", { hidden: true }, [
    element("h2", { textContent: "Result" }),
    endText,
    endNotes,
    download,
    sendNote,
  ]);

  app.append(
    element("h1", { textContent: "Karume GPU benchmark" }),
    element("p", {
      className: "lead",
      textContent:
        "This page measures how fast a few of Karume's GPU kernels run on this computer, so that Karume can pick the fastest way to run them on GPUs like yours. It runs in this tab and sends nothing by itself.",
    }),
    element("section", {}, [element("h2", { textContent: "Your GPU" }), device, deviceNotes]),
    element("section", {}, [
      element("h2", { textContent: "Before you start" }),
      notes,
      element("fieldset", {}, [
        element("legend", { textContent: "Measurements" }),
        element("label", {}, [gemvBox, " GEMV sweep"]),
        element("label", {}, [geometryBox, " Tile geometry sweep"]),
        element("label", {}, [gridBox, " GEMM tile ladder"]),
      ]),
      element("div", { className: "actions" }, [start, cancel]),
      problem,
    ]),
    run,
    ending,
    element("footer", {
      textContent: `Built from ${checkout.commit.slice(0, 12)}${checkout.dirty ? " with uncommitted changes" : ""} · runtime ${checkout.runtimeVersion}`,
    }),
  );

  const showProblem = (text: string): void => {
    problem.hidden = false;
    setText(problem, text);
  };

  let trial: Trial;
  try {
    trial = trialOfQuery(location.search);
  } catch (error) {
    showProblem(`The address asks for something this page does not take: ${errorText(error)}`);
    return;
  }

  const found = await probeGpu();
  if (typeof found === "string") {
    showProblem(found);
    return;
  }
  const view = deviceView(found);
  device.replaceChildren(
    ...view.rows.flatMap(([label, value]) => [
      element("dt", { textContent: label }),
      element("dd", { textContent: value }),
    ]),
  );
  deviceNotes.replaceChildren(
    ...(view.blocker === undefined
      ? []
      : [element("p", { className: "problem", textContent: view.blocker })]),
    ...view.notes.map((text) => element("p", { className: "warning", textContent: text })),
  );

  let running: AbortController | undefined;
  let plan: BenchmarkPlan = planBenchmark(choice, trial);

  const showPlan = (): void => {
    plan = planBenchmark(choice, trial);
    const estimate = remainingEstimate(
      plan.cases.map(({ kind }) => kind),
      { finished: new Map() },
      plan.idleSeconds,
    );
    notes.replaceChildren(
      ...beforeRunNotes(plan, estimate.seconds).map((text) => element("li", { textContent: text })),
    );
    start.disabled = running !== undefined || view.blocker !== undefined || plan.cases.length === 0;
    gemvBox.disabled = running !== undefined;
    geometryBox.disabled = running !== undefined;
    gridBox.disabled = running !== undefined;
    cancel.disabled = running === undefined;
  };
  gemvBox.addEventListener("change", () => {
    choice.gemv = gemvBox.checked;
    showPlan();
  });
  geometryBox.addEventListener("change", () => {
    choice.geometry = geometryBox.checked;
    showPlan();
  });
  gridBox.addEventListener("change", () => {
    choice.grid = gridBox.checked;
    showPlan();
  });
  showPlan();

  const paintRun = (
    progress: ProgressView,
    hidden: { readonly hiddenCount: number; readonly hiddenSeconds: number },
  ): void => {
    paintBar(overall, progress.overall);
    if (progress.current === undefined) {
      setText(caseTitle, "Starting");
      paintBar(caseBar, { done: 0, total: 1, text: "" });
      setText(caseDetail, "");
    } else {
      setText(caseTitle, progress.current.title);
      paintBar(caseBar, progress.current);
      setText(caseDetail, progress.current.detail ?? "");
    }
    setText(remaining, progress.remaining);
    warning.hidden = progress.warning === undefined;
    setText(warning, progress.warning ?? "");
    hiddenNote.hidden = hidden.hiddenCount === 0;
    setText(
      hiddenNote,
      `The page was hidden ${hidden.hiddenCount === 1 ? "once" : `${hidden.hiddenCount} times`} (${Math.round(hidden.hiddenSeconds)} s in all) during the run. A browser may slow a hidden page, so the file records it; keep the tab in front.`,
    );
    const lines = progress.patterns;
    if (patterns.children.length !== lines.length) {
      patterns.replaceChildren(...lines.map(() => element("li")));
    }
    lines.forEach((line, index) => {
      const item = patterns.children[index];
      if (item instanceof HTMLElement) setText(item, line);
    });
  };

  const showEnding = (result: RunEnding, state: ProgressState): void => {
    ending.hidden = false;
    const { bundle } = result;
    const counts = patternLines(plan, state).join(". ");
    endNotes.replaceChildren(
      ...afterRunNotes(bundle.records).map((text) =>
        element("p", { className: "warning", textContent: text }),
      ),
    );
    if (bundle.complete && result.downloadError !== undefined) {
      setText(
        endText,
        `Done, but the file could not be downloaded by itself (${result.downloadError}). Download it with the button below. Found in this run: ${counts}. ${HONESTY}`,
      );
      download.hidden = false;
      setText(download, "Download the result");
      sendNote.hidden = false;
      return;
    }
    if (bundle.complete) {
      setText(
        endText,
        `Done. The result was downloaded as ${result.fileName}. Found in this run: ${counts}. ${HONESTY}`,
      );
      download.hidden = false;
      setText(download, "Download again");
      sendNote.hidden = false;
      return;
    }
    const stop = bundle.stop;
    const why =
      stop.kind === "cancelled"
        ? "The run was cancelled."
        : stop.kind === "deviceLost"
          ? `The run stopped: the GPU device was lost (${stop.reason}): ${stop.message}.`
          : stop.kind === "error"
            ? `The run stopped on an error: ${stop.message}`
            : "The run stopped.";
    download.hidden = !result.measured;
    sendNote.hidden = !result.measured;
    setText(download, "Download what was measured");
    setText(
      endText,
      result.measured
        ? `${why} What was measured is kept: the file is marked as incomplete, and it is still worth sending. Found so far: ${counts}.`
        : `${why} Nothing was measured, so there is nothing to send.`,
    );
  };

  /** The page could not end the run as usual: what is known of it, and the file when there is one. */
  const showFailure = (message: string): void => {
    ending.hidden = false;
    endNotes.replaceChildren();
    const offered = last?.measured === true;
    download.hidden = !offered;
    sendNote.hidden = !offered;
    setText(download, "Download what was measured");
    setText(
      endText,
      `The run stopped on an error of this page: ${message}${offered ? " What was measured can still be downloaded." : ""}`,
    );
  };

  let last: RunEnding | undefined;
  download.addEventListener("click", () => {
    if (last !== undefined) downloadText(last.fileName, last.text);
  });
  cancel.addEventListener("click", () => running?.abort());

  start.addEventListener("click", () => {
    const controller = new AbortController();
    running = controller;
    showPlan();
    run.hidden = false;
    ending.hidden = true;
    const runPlan = plan;
    const queue: StampedEvent[] = [];
    let state = INITIAL_PROGRESS;
    const log = hiddenLog(document.hidden, performance.now());
    const lock = holdWakeLock("wakeLock" in navigator ? navigator.wakeLock : undefined);
    const onVisibility = (): void => {
      log.note(document.hidden, performance.now());
      if (!document.hidden) lock.renew();
    };
    document.addEventListener("visibilitychange", onVisibility);
    const frame = (): void => {
      for (const event of queue.splice(0)) state = foldProgress(state, event);
      const now = performance.now();
      paintRun(progressView(runPlan, state, now), log.summary(now));
    };
    let handle = 0;
    let ended = false;
    const loop = (): void => {
      if (ended) return;
      frame();
      handle = requestAnimationFrame(loop);
    };
    handle = requestAnimationFrame(loop);
    /** Stops the painting, the lock and the listener and lets a new run start; once. */
    const endRun = (): void => {
      if (ended) return;
      ended = true;
      cancelAnimationFrame(handle);
      document.removeEventListener("visibilitychange", onVisibility);
      lock.release();
      running = undefined;
      showPlan();
    };
    void runBenchmark(runPlan, {
      session,
      signal: controller.signal,
      onEvent: (event) => queue.push(event),
      readEnvironment: () =>
        readEnvironment(navigator.gpu, {
          host: "browser",
          userAgent: navigator.userAgent,
          crossOriginIsolated: globalThis.crossOriginIsolated,
        }),
      checkout,
      download: downloadText,
      pageFacts: () => ({ ...log.summary(performance.now()), wakeLock: lock.state() }),
    })
      .then((result) => {
        endRun();
        last = result;
        frame();
        showEnding(result, state);
      })
      .catch((error: unknown) => {
        endRun();
        showFailure(messageText(error));
      });
  });
};

void main();
