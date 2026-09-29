/// <reference lib="dom" />
/**
 * 掃引のタブ — タイル幾何の掃引（perf-ledger K-70）を Chrome で回す（旧 `tools/geometry-sweep` の
 * ブラウザのページの移植・測り方・表の列・JSON の欄・保存名は同じ）。
 *
 * Deno の CLI（`tools/geometry-sweep/main.ts`）と同じ計測核（`harness.ts`）と同じ記録の形
 * （`report.ts`）で、「開始」ごとに GPU を取り直して選んだ op のケースを掃引し、行を表に積む。
 * 「JSON を保存」は直近の掃引を `karume-geometry-sweep/2` で書き出す。失敗は表の行に出す（alert しない）。
 *
 * 掃引は各幾何を**明示して**測るので、GPU 設定の幾何プロファイル（注入）は結果に効かない —
 * 掃引の GPU には注入しない。GPU 設定の timestamp の要求は効く。
 */
import { acquireGpu, type GpuContext } from "../../../packages/runtime/mod.ts";
import { SWEEP_CASES, SWEEP_OPS, type SweepOp } from "../../geometry-sweep/cases.ts";
import { DEFAULT_CANDIDATE_SET, isCandidateSet } from "../../geometry-sweep/geometries.ts";
import {
  createSweepContext,
  destroySweepContext,
  runSweep,
  SWEEP_MAX_REPS,
  type SweepContext,
  TARGET_PASS_MS,
  WARMUP_MIN_RUNS,
  WARMUP_NS,
} from "../../geometry-sweep/harness.ts";
import {
  type CaseSummary,
  DEFAULT_DRIFT_RANGE,
  driftOutOfRange,
  type Report,
  REPORT_FORMAT,
  roundsLookQuantized,
  type SweepRow,
  WALL_TIMING_NOTE,
} from "../../geometry-sweep/report.ts";
import {
  adapterSummary,
  checkoutLabel,
  copyFromTextarea,
  downloadText,
  element,
  errorText,
  type Lab,
  setStatus,
} from "./common.ts";

/** 「JSON を保存」の保存名（プロファイルのタブが記録の path の既定にも使う）。 */
export const sweepFileName = (report: Report): string =>
  `geometry-sweep-browser-${report.date.replaceAll(":", "-")}.json`;

/** 「JSON を保存」と同じバイト列（プロファイルのタブが sha256 をこのバイト列で取る）。 */
export const sweepJson = (report: Report): string => JSON.stringify(report, null, 2);

export type SweepTab = {
  /** 直近の掃引（まだ無ければ undefined — 中断した掃引も測れた行までを持つ）。 */
  readonly latestReport: () => Report | undefined;
};

/**
 * `onFinished` は掃引が終わるたび（完了・中断・失敗のどれでも — 直近の結果が変わりうる）に呼ぶ
 * （プロファイルのタブを開いたまま掃引が終わっても「直近の結果」を今に合わせるため）。
 */
export const mountSweepTab = (
  root: HTMLElement,
  lab: Lab,
  onFinished: () => void,
): SweepTab => {
  const ui = {
    ops: element(root, "ops", HTMLFieldSetElement),
    set: element(root, "set", HTMLSelectElement),
    rounds: element(root, "rounds", HTMLInputElement),
    start: element(root, "start", HTMLButtonElement),
    stop: element(root, "stop", HTMLButtonElement),
    exportJson: element(root, "export", HTMLButtonElement),
    showJson: element(root, "show-json", HTMLButtonElement),
    copyJson: element(root, "copy-json", HTMLButtonElement),
    jsonOut: element(root, "json-out", HTMLTextAreaElement),
    status: element(root, "status", HTMLElement),
    info: element(root, "info", HTMLElement),
    rows: element(root, "rows", HTMLTableSectionElement),
  };

  const state: { report?: Report; abort?: AbortController } = {};

  const status = (text: string): void => setStatus(ui.status, text);

  /** チェックの付いた op（並びは {@link SWEEP_OPS} の順）。 */
  const selectedOps = (): SweepOp[] => {
    const checked = new Set(
      [...ui.ops.querySelectorAll<HTMLInputElement>("input[type=checkbox]")]
        .filter((input) => input.checked)
        .map((input) => input.value),
    );
    return SWEEP_OPS.filter((op) => checked.has(op));
  };

  const cell = (text: string, className?: string): HTMLTableCellElement => {
    const td = document.createElement("td");
    td.textContent = text;
    if (className !== undefined) td.className = className;
    return td;
  };

  /** ケースの見出し行（id・形状・census 上の本数）。 */
  const appendCaseHeader = (row: SweepRow): void => {
    const tr = document.createElement("tr");
    tr.className = "case";
    const th = document.createElement("th");
    th.colSpan = 7;
    th.textContent = `${row.caseId}（${row.shape} · census ${row.censusCount} 本）`;
    tr.append(th);
    ui.rows.append(tr);
  };

  const appendRow = (row: SweepRow): void => {
    const tr = document.createElement("tr");
    if (row.isDefault) tr.className = "default";
    const identical = row.identicalToDefault === undefined
      ? cell("—")
      : row.identicalToDefault
      ? cell("一致")
      : cell("不一致（採用不可）", "bad");
    tr.append(
      cell(`${row.isDefault ? "* " : ""}${row.geometry}`),
      cell(row.perDispatch === undefined ? "—" : (row.perDispatch / 1e3).toFixed(1), "num"),
      cell(
        row.speedupVsDefault === undefined ? "—" : `×${row.speedupVsDefault.toFixed(2)}`,
        "num",
      ),
      cell(row.tflops === undefined ? "—" : row.tflops.toFixed(2), "num"),
      cell(row.reps === undefined ? "—" : String(row.reps), "num"),
      identical,
      cell(row.error ?? "", row.error === undefined ? undefined : "error"),
    );
    ui.rows.append(tr);
  };

  /** ケースの末尾の行（既定幾何の再測定と、範囲外なら測り直しの警告）。 */
  const appendCaseSummary = (summary: CaseSummary): void => {
    const tr = document.createElement("tr");
    tr.className = "repeat";
    const td = document.createElement("td");
    td.colSpan = 7;
    if (summary.defaultRepeatError !== undefined) {
      td.textContent = `既定の再測定 失敗: ${summary.defaultRepeatError}`;
      td.className = "error";
    } else if (summary.defaultRepeat === undefined) {
      td.textContent = "既定の再測定 —（既定の初回が失敗・中断・device lost）";
    } else {
      const { driftRatio } = summary.defaultRepeat;
      const drifted = driftOutOfRange(driftRatio);
      td.textContent = `既定の再測定 ×${driftRatio.toFixed(2)}${
        drifted
          ? `  警告: ${DEFAULT_DRIFT_RANGE.min}〜${DEFAULT_DRIFT_RANGE.max} の外 — このケースは測り直しの目安`
          : ""
      }`;
      if (drifted) td.className = "bad";
    }
    tr.append(td);
    ui.rows.append(tr);
  };

  const setRunning = (running: boolean): void => {
    ui.start.disabled = running;
    ui.stop.disabled = !running;
    ui.exportJson.disabled = running || state.report === undefined;
    ui.showJson.disabled = ui.exportJson.disabled;
    ui.copyJson.disabled = ui.exportJson.disabled;
    for (const input of ui.ops.querySelectorAll("input")) input.disabled = running;
    ui.set.disabled = running;
    ui.rounds.disabled = running;
  };

  const buildReport = (
    gpu: GpuContext,
    unit: Report["gpuTiming"]["unit"],
    dp4a: boolean,
    settings: Report["settings"],
    deviceLost: Report["deviceLost"],
    cases: readonly CaseSummary[],
    rows: readonly SweepRow[],
  ): Report => {
    const info = gpu.adapterInfo;
    return {
      format: REPORT_FORMAT,
      date: new Date().toISOString(),
      userAgent: navigator.userAgent,
      adapter: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      },
      checkout: lab.config.revision,
      checkoutDirty: lab.config.dirty,
      bundleSha256: lab.config.bundleSha256,
      gpuTiming: {
        feature: gpu.gpuTimingEnabled,
        unit,
        quantized: roundsLookQuantized(rows, unit),
      },
      dp4a,
      settings,
      deviceLost,
      cases,
      rows,
    };
  };

  const runOnce = async (): Promise<void> => {
    const ops = selectedOps();
    if (ops.length === 0) throw Error("op を 1 つ以上選んでください");
    const rounds = Number(ui.rounds.value);
    if (!Number.isInteger(rounds) || rounds < 1) {
      throw Error(`rounds ${ui.rounds.value} が正の整数でない`);
    }
    const candidateSet = ui.set.value;
    if (!isCandidateSet(candidateSet)) throw Error(`候補集合 ${candidateSet} を知らない`);
    const selected = SWEEP_CASES.filter((sweepCase) => ops.includes(sweepCase.op));
    let deviceLost: Report["deviceLost"] = null;
    const gpu = await acquireGpu({
      ...(lab.timestampFeature && lab.settings().timestamps ? { gpuTiming: true } : {}),
      onDeviceLost: (info) => {
        deviceLost = { reason: info.reason, message: info.message };
        status(`GPU device lost（${info.reason}）: ${info.message}`);
      },
    });
    const abort = new AbortController();
    const rows: SweepRow[] = [];
    const cases: CaseSummary[] = [];
    let current: string | undefined;
    const settings: Report["settings"] = {
      candidateSet,
      quick: candidateSet === "quick",
      ops,
      rounds,
      targetPassMs: TARGET_PASS_MS,
      maxReps: SWEEP_MAX_REPS,
      warmupNs: WARMUP_NS,
      warmupMinRuns: WARMUP_MIN_RUNS,
      wallTimingNote: WALL_TIMING_NOTE,
    };
    // MUST: 文脈の生成も try の内側（生成が失敗しても device を返す — 開始のたびに取り直すので、
    // 返し漏れは押すたびに device を 1 つずつ溜める）
    let context: SweepContext | undefined;
    try {
      context = await createSweepContext(gpu, "ns");
      state.abort = abort;
      ui.rows.replaceChildren();
      ui.info.textContent = `${
        adapterSummary(gpu.adapterInfo)
      } · 単位 ${context.unit} · dp4a ${context.dp4a} · 候補 ${candidateSet} · ${
        checkoutLabel(lab.config)
      }`;
      await runSweep(context, selected, { rounds, candidateSet, timestampUnit: "ns" }, {
        signal: abort.signal,
        onProgress: status,
        onRow: (row) => {
          if (row.caseId !== current) {
            current = row.caseId;
            appendCaseHeader(row);
          }
          rows.push(row);
          appendRow(row);
        },
        onCase: (summary) => {
          cases.push(summary);
          appendCaseSummary(summary);
        },
      });
    } finally {
      state.abort = undefined;
      if (context !== undefined) {
        destroySweepContext(context);
        state.report = buildReport(
          gpu,
          context.unit,
          context.dp4a,
          settings,
          deviceLost,
          cases,
          rows,
        );
      }
      gpu.destroy();
    }
    const failed = rows.filter((row) => row.error !== undefined).length;
    const mismatched = rows.filter((row) => row.identicalToDefault === false).length;
    const drifted = cases.filter((summary) =>
      summary.defaultRepeatError !== undefined ||
      (summary.defaultRepeat !== undefined && driftOutOfRange(summary.defaultRepeat.driftRatio))
    ).length;
    status(
      `${
        abort.signal.aborted ? "中断しました" : "完了"
      } — ${rows.length} 行（失敗 ${failed} · 不一致 ${mismatched} · 既定の再測定が範囲外 / 失敗 ${drifted} ケース）${
        state.report?.gpuTiming.quantized === true
          ? " · timestamp が 100 µs に丸められている疑い"
          : ""
      }。「JSON を保存」で書き出せます。プロファイルのタブで表を作れます。`,
    );
  };

  const exportJson = (): void => {
    const report = state.report;
    if (report === undefined) return;
    downloadText(sweepJson(report), sweepFileName(report), "application/json");
  };

  /**
   * ダウンロードが塞がれた置き場（Artifact のような sandbox）向けの逃げ道: JSON をページ内の textarea
   * に出す。「コピー」は clipboard API が拒まれたら textarea を選択状態にして手でコピーできるようにする。
   */
  const showJson = (): void => {
    const report = state.report;
    if (report === undefined) return;
    ui.jsonOut.value = sweepJson(report);
    ui.jsonOut.hidden = false;
  };

  const copyJson = async (): Promise<void> => {
    showJson();
    status(
      await copyFromTextarea(ui.jsonOut)
        ? "JSON をクリップボードへコピーしました。"
        : "クリップボードが使えないので、選択した JSON を手でコピーしてください。",
    );
  };

  ui.set.value = DEFAULT_CANDIDATE_SET;
  ui.info.textContent = `${adapterSummary(lab.adapterInfo)} · ${checkoutLabel(lab.config)}`;
  ui.start.addEventListener("click", async () => {
    let release: (() => void) | undefined;
    try {
      release = lab.lock("掃引");
    } catch (error) {
      status(errorText(error));
      return;
    }
    setRunning(true);
    try {
      await runOnce();
    } catch (error) {
      status(errorText(error));
    } finally {
      setRunning(false);
      release();
      onFinished();
    }
  });
  ui.stop.addEventListener("click", () => {
    state.abort?.abort();
    status("今の幾何を測り終えたところで止めます…");
  });
  ui.exportJson.addEventListener("click", exportJson);
  ui.showJson.addEventListener("click", showJson);
  ui.copyJson.addEventListener("click", () => void copyJson());
  status("準備完了。op と候補を選んで「開始」。");
  setRunning(false);
  return { latestReport: () => state.report };
};
