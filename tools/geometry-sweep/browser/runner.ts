/// <reference lib="dom" />
/**
 * タイル幾何の掃引（perf-ledger K-70）を Chrome で回すページの本体（`server.ts` が起動時に bundle する）。
 *
 * Deno の CLI（`../main.ts`）と同じ計測核（`../harness.ts`）と同じ記録の形（`../report.ts`）で、
 * 「開始」ごとに GPU を取り直して選んだ op のケースを掃引し、行を表に積む。「JSON を保存」は直近の
 * 掃引を `karume-geometry-sweep/2` で書き出す。失敗は表の行に出す（alert しない）。
 */
import { acquireGpu, type GpuContext } from "../../../packages/runtime/mod.ts";
import { SWEEP_CASES, SWEEP_OPS, type SweepOp } from "../cases.ts";
import {
  createSweepContext,
  destroySweepContext,
  MAX_REPS,
  runSweep,
  type SweepContext,
  TARGET_PASS_MS,
  WARMUP_MIN_RUNS,
  WARMUP_NS,
} from "../harness.ts";
import {
  type CaseSummary,
  DEFAULT_DRIFT_RANGE,
  driftOutOfRange,
  type Report,
  REPORT_FORMAT,
  roundsLookQuantized,
  type SweepRow,
  WALL_TIMING_NOTE,
} from "../report.ts";

const TIMESTAMP_QUERY = "timestamp-query";

type ServerConfig = {
  readonly revision: string;
  readonly dirty: boolean;
  readonly bundleSha256: string;
};

const element = <T extends HTMLElement>(id: string, type: new () => T): T => {
  const found = document.getElementById(id);
  if (!(found instanceof type)) throw Error(`Missing page element #${id}`);
  return found;
};

const ui = {
  ops: element("ops", HTMLFieldSetElement),
  set: element("set", HTMLSelectElement),
  rounds: element("rounds", HTMLInputElement),
  timestamps: element("timestamps", HTMLInputElement),
  timestampsNote: element("timestamps-note", HTMLElement),
  start: element("start", HTMLButtonElement),
  stop: element("stop", HTMLButtonElement),
  exportJson: element("export", HTMLButtonElement),
  status: element("status", HTMLElement),
  environment: element("environment", HTMLElement),
  rows: element("rows", HTMLTableSectionElement),
};

const state: {
  config?: ServerConfig;
  adapterInfo?: GPUAdapterInfo;
  timestampFeature: boolean;
  report?: Report;
  abort?: AbortController;
} = { timestampFeature: false };

const setStatus = (text: string): void => {
  ui.status.textContent = text;
};

const describeError = (error: unknown): string =>
  error instanceof Error ? `${error.name}: ${error.message}` : String(error);

const adapterSummary = (info: GPUAdapterInfo): string =>
  [info.vendor, info.architecture, info.device, info.description].filter((v) => v !== "").join(
    " / ",
  );

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
    cell(row.speedupVsDefault === undefined ? "—" : `×${row.speedupVsDefault.toFixed(2)}`, "num"),
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
  for (const input of ui.ops.querySelectorAll("input")) input.disabled = running;
  ui.set.disabled = running;
  ui.rounds.disabled = running;
  ui.timestamps.disabled = running || !state.timestampFeature;
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
    checkout: state.config?.revision,
    checkoutDirty: state.config?.dirty,
    bundleSha256: state.config?.bundleSha256,
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
  const quick = ui.set.value === "quick";
  const selected = SWEEP_CASES.filter((sweepCase) => ops.includes(sweepCase.op));
  let deviceLost: Report["deviceLost"] = null;
  const gpu = await acquireGpu({
    ...(state.timestampFeature && ui.timestamps.checked ? { gpuTiming: true } : {}),
    onDeviceLost: (info) => {
      deviceLost = { reason: info.reason, message: info.message };
      setStatus(`GPU device lost（${info.reason}）: ${info.message}`);
    },
  });
  const abort = new AbortController();
  const rows: SweepRow[] = [];
  const cases: CaseSummary[] = [];
  let current: string | undefined;
  const settings: Report["settings"] = {
    quick,
    ops,
    rounds,
    targetPassMs: TARGET_PASS_MS,
    maxReps: MAX_REPS,
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
    ui.environment.textContent = `${
      adapterSummary(gpu.adapterInfo)
    } · 単位 ${context.unit} · dp4a ${context.dp4a} · ${state.config?.revision.slice(0, 8) ?? "?"}${
      state.config?.dirty === true ? " (dirty)" : ""
    }`;
    await runSweep(context, selected, { rounds, quick, timestampUnit: "ns" }, {
      signal: abort.signal,
      onProgress: setStatus,
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
  setStatus(
    `${
      abort.signal.aborted ? "中断しました" : "完了"
    } — ${rows.length} 行（失敗 ${failed} · 不一致 ${mismatched} · 既定の再測定が範囲外 / 失敗 ${drifted} ケース）${
      state.report?.gpuTiming.quantized === true
        ? " · timestamp が 100 µs に丸められている疑い"
        : ""
    }。「JSON を保存」で書き出せます。`,
  );
};

const exportJson = (): void => {
  const report = state.report;
  if (report === undefined) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `geometry-sweep-browser-${report.date.replaceAll(":", "-")}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const initialize = async (): Promise<void> => {
  setRunning(true);
  const configResponse = await fetch("/config.json");
  if (!configResponse.ok) throw Error(`config.json HTTP ${configResponse.status}`);
  state.config = await configResponse.json();
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) {
    throw Error("ハードウェア WebGPU を使える Chrome が必要です");
  }
  state.adapterInfo = adapter.info;
  state.timestampFeature = adapter.features.has(TIMESTAMP_QUERY);
  ui.timestamps.checked = state.timestampFeature;
  ui.timestampsNote.textContent = state.timestampFeature
    ? ""
    : `（このアダプタは ${TIMESTAMP_QUERY} を持たないので壁時計で測ります）`;
  ui.environment.textContent = `${adapterSummary(adapter.info)} · ${
    state.config?.revision.slice(0, 8) ?? "?"
  }${state.config?.dirty === true ? " (dirty)" : ""}`;
  setStatus("準備完了。op と候補を選んで「開始」。");
  setRunning(false);
};

ui.start.addEventListener("click", async () => {
  setRunning(true);
  try {
    await runOnce();
  } catch (error) {
    setStatus(describeError(error));
  } finally {
    setRunning(false);
  }
});
ui.stop.addEventListener("click", () => {
  state.abort?.abort();
  setStatus("今の幾何を測り終えたところで止めます…");
});
ui.exportJson.addEventListener("click", exportJson);
initialize().catch((error: unknown) => setStatus(describeError(error)));
