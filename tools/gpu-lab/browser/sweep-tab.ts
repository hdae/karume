/// <reference lib="dom" />
/**
 * 掃引のタブ — タイル幾何の掃引（perf-ledger K-70）を Chrome で回す（旧 `tools/geometry-sweep` の
 * ブラウザのページの移植・測り方・表の列・JSON の欄・保存名は同じ）。
 *
 * 掃引そのもの（「開始」ごとの GPU の取り直し・計測・記録の組み立て）は runtime の `runGeometrySweep`
 * （`@karume/runtime/tune` — ADR 0117 決定 2・Deno の CLI `tools/geometry-sweep/main.ts` と同じ 1 本）で、
 * ここは選んだ op のケースを渡し、進捗の通知で行を表に積む殻。「JSON を保存」は直近の掃引を
 * `karume-geometry-sweep/2` で書き出す（ページの checkout と bundle の sha256 を足す）。失敗は表の行に
 * 出す（alert しない）。
 *
 * 掃引は各幾何を**明示して**測るので、GPU 設定の幾何プロファイル（注入）は結果に効かない —
 * 掃引の GPU には注入しない。GPU 設定の timestamp の要求は効く。
 */
import {
  type GeometrySweepCaseSummary,
  type GeometrySweepReport,
  type GeometrySweepRow,
  runGeometrySweep,
} from "../../../packages/runtime/tune.ts";
import { SWEEP_OPS, type SweepOp } from "../../../packages/runtime/src/tune/cases.ts";
import {
  DEFAULT_CANDIDATE_SET,
  isCandidateSet,
} from "../../../packages/runtime/src/tune/geometries.ts";
import { DEFAULT_DRIFT_RANGE, driftOutOfRange } from "../../../packages/runtime/src/tune/report.ts";
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
export const sweepFileName = (report: GeometrySweepReport): string =>
  `geometry-sweep-browser-${report.date.replaceAll(":", "-")}.json`;

/** 「JSON を保存」と同じバイト列（プロファイルのタブが sha256 をこのバイト列で取る）。 */
export const sweepJson = (report: GeometrySweepReport): string => JSON.stringify(report, null, 2);

export type SweepTab = {
  /** 直近の掃引（まだ無ければ undefined — 中断した掃引も測れた行までを持つ）。 */
  readonly latestReport: () => GeometrySweepReport | undefined;
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

  const state: { report?: GeometrySweepReport; abort?: AbortController } = {};

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
  const appendCaseHeader = (row: GeometrySweepRow): void => {
    const tr = document.createElement("tr");
    tr.className = "case";
    const th = document.createElement("th");
    th.colSpan = 7;
    th.textContent = `${row.caseId}（${row.shape} · census ${row.censusCount} 本）`;
    tr.append(th);
    ui.rows.append(tr);
  };

  const appendRow = (row: GeometrySweepRow): void => {
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
  const appendCaseSummary = (summary: GeometrySweepCaseSummary): void => {
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

  const runOnce = async (): Promise<void> => {
    const ops = selectedOps();
    if (ops.length === 0) throw Error("op を 1 つ以上選んでください");
    const rounds = Number(ui.rounds.value);
    if (!Number.isInteger(rounds) || rounds < 1) {
      throw Error(`rounds ${ui.rounds.value} が正の整数でない`);
    }
    const candidateSet = ui.set.value;
    if (!isCandidateSet(candidateSet)) throw Error(`候補集合 ${candidateSet} を知らない`);
    const abort = new AbortController();
    let current: string | undefined;
    // device の取り直しと後始末（中断・失敗でも捨てる）は runGeometrySweep が持つ
    state.abort = abort;
    let swept: GeometrySweepReport;
    try {
      swept = await runGeometrySweep({
        candidateSet,
        ops,
        rounds,
        timestamps: lab.settings().timestamps,
        signal: abort.signal,
        onProgress: (progress) => {
          switch (progress.kind) {
            case "started":
              ui.rows.replaceChildren();
              ui.info.textContent = `${
                adapterSummary(progress.adapter)
              } · 単位 ${progress.unit} · dp4a ${progress.dp4a} · 候補 ${candidateSet} · ${
                checkoutLabel(lab.config)
              }`;
              break;
            case "status":
              status(progress.message);
              break;
            case "row":
              if (progress.row.caseId !== current) {
                current = progress.row.caseId;
                appendCaseHeader(progress.row);
              }
              appendRow(progress.row);
              break;
            case "case":
              appendCaseSummary(progress.summary);
              break;
            case "deviceLost":
              status(`GPU device lost（${progress.reason}）: ${progress.message}`);
              break;
          }
        },
      });
    } finally {
      state.abort = undefined;
    }
    const report: GeometrySweepReport = {
      ...swept,
      checkout: lab.config.revision,
      checkoutDirty: lab.config.dirty,
      bundleSha256: lab.config.bundleSha256,
    };
    state.report = report;
    const { rows, cases } = report;
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
        report.gpuTiming.quantized ? " · timestamp が 100 µs に丸められている疑い" : ""
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
