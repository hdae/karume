/// <reference lib="dom" />
/**
 * anima の DiT 常駐（ADR 0112）を Chrome で確かめるページの本体（`server.ts` が起動時に bundle する）。
 *
 * 1 本の共有 GPU（`acquireGpu()`）に pipeline（`residency: "transformer"` で 1 度だけ組んで使い回す）と
 * ダミーの STORAGE バッファを同居させ、generate ごとに所要・段の時間・`residency` イベント・PNG の
 * sha256 を表に積む。確かめたいのは 3 点:
 *
 * 1. 常駐 on の 2 枚目以降が 1 枚目より速い（DiT の読み直しが消える）。
 * 2. 常駐 on の各 generate が `retained` / `request` を名乗る。
 * 3. ダミーで VRAM を埋めた後の generate が `evicted` / `headroom`（先回り）か `evicted` /
 *    `out-of-memory`（反応）で退避し、それでも同じ PNG sha を出す。
 *
 * 失敗は表の行に出す（alert しない）— `GpuDeviceLostError` / `GpuOutOfMemoryError` の名前と文言を
 * そのまま残すのが、この確認の一番の観測点だから。
 *
 * op 別 GPU 時間（perf-ledger K-70）: quant と「GPU 時間を採る」は **GPU を取るときに確定**する
 * （feature は device 作成時にしか要求できない — `acquireGpu` の `gpuTiming` / `shaderF16`）。計測が
 * 有効なら `onRunDiagnostics` の `lastRunTiming` を段ごとに足し、`lastRunPipelines`（計測に依らない
 * dispatch 本数）は常に足す（`../timing.ts`）。
 */
import { acquireGpu, type GpuContext } from "../../../packages/runtime/mod.ts";
import { localDirectory, type ModelEntry, parseManifest } from "../../../packages/hub/mod.ts";
import {
  type AnimaGenerateEvent,
  type AnimaGenerateRequest,
  AnimaPipeline,
  type AnimaResidency,
  parseResolution,
} from "../../../packages/models/anima.ts";
import { encodePng } from "../../../packages/models/mod.ts";
import {
  createGenerateRecorder,
  DEFAULT_PROMPT,
  type DummyHold,
  type GenerateRecorder,
  geometryProfilesOf,
  type PipelineLoad,
  type Report,
  REPORT_FORMAT,
  type ResidencyRecord,
  type Row,
  type StageRecord,
} from "../record.ts";
import { looksQuantized, type StageGpuTiming, topEntries } from "../timing.ts";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const TIMESTAMP_QUERY = "timestamp-query";

type ServerConfig = {
  readonly revision: string;
  readonly dirty: boolean;
  readonly bundleSha256: string;
  readonly source: string;
};

/** GPU を取った時点で確定する構成（変えるには「pipeline を破棄」— GPU ごと畳む）。 */
type BuildChoice = { readonly quant: string; readonly gpuTiming: boolean };

const element = <T extends HTMLElement>(id: string, type: new () => T): T => {
  const found = document.getElementById(id);
  if (!(found instanceof type)) throw Error(`Missing page element #${id}`);
  return found;
};

const ui = {
  prompt: element("prompt", HTMLTextAreaElement),
  negative: element("negative", HTMLInputElement),
  resolution: element("resolution", HTMLSelectElement),
  steps: element("steps", HTMLInputElement),
  guidance: element("guidance", HTMLInputElement),
  seed: element("seed", HTMLInputElement),
  residency: element("residency", HTMLSelectElement),
  quant: element("quant", HTMLSelectElement),
  gpuTiming: element("gpu-timing", HTMLInputElement),
  gpuTimingNote: element("gpu-timing-note", HTMLElement),
  count: element("count", HTMLInputElement),
  run: element("run", HTMLButtonElement),
  holdGib: element("hold-gib", HTMLInputElement),
  hold: element("hold", HTMLButtonElement),
  release: element("release", HTMLButtonElement),
  dispose: element("dispose", HTMLButtonElement),
  exportJson: element("export", HTMLButtonElement),
  status: element("status", HTMLElement),
  environment: element("environment", HTMLElement),
  dummies: element("dummies", HTMLElement),
  rows: element("rows", HTMLTableSectionElement),
  image: element("image", HTMLImageElement),
};

const state: {
  config?: ServerConfig;
  manifestSha256?: string;
  defaultModel?: string;
  /** 既定モデルの manifest の欄（quant の選択肢と `gpuFeatures`）。 */
  model?: ModelEntry;
  /** GPU を取ったときに書いた環境行（直近の generate の幾何プロファイルを足す土台）。 */
  environment?: string;
  /** 初期化時に読んだアダプタ（GPU を畳んだ後の書き出しでも機体を残すため）。 */
  adapterInfo?: GPUAdapterInfo;
  /** アダプタが `timestamp-query` を列挙したか。 */
  timestampFeature: boolean;
  gpu?: GpuContext;
  /** {@link BuildChoice}（`gpu` と同じ寿命）。 */
  build?: BuildChoice;
  pipeline?: AnimaPipeline;
  /** 進行中の generate の記録器（pipeline の `onRunDiagnostics` の行き先）。 */
  recorder?: GenerateRecorder;
  deviceLost?: { readonly reason: string; readonly message: string };
  dummies: GPUBuffer[];
  dummyBytes: number;
  holds: DummyHold[];
  pipelineLoads: PipelineLoad[];
  rows: Row[];
  busy: boolean;
  imageUrl?: string;
} = {
  timestampFeature: false,
  dummies: [],
  dummyBytes: 0,
  holds: [],
  pipelineLoads: [],
  rows: [],
  busy: false,
};

const setStatus = (text: string): void => {
  ui.status.textContent = text;
};

const describeError = (error: unknown): { name: string; message: string } => {
  if (error instanceof AggregateError) {
    return {
      name: error.name,
      message: `${error.message} [${
        error.errors.map((cause) => {
          const inner = describeError(cause);
          return `${inner.name}: ${inner.message}`;
        }).join(" | ")
      }]`,
    };
  }
  return error instanceof Error
    ? { name: error.name, message: error.message }
    : { name: "Error", message: String(error) };
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const gib = (bytes: number): string => `${(bytes / GIB).toFixed(2)} GiB`;

const adapterSummary = (info: GPUAdapterInfo): string =>
  [info.vendor, info.architecture, info.device, info.description].filter((v) => v !== "").join(
    " / ",
  );

const renderDummies = (): void => {
  ui.dummies.textContent = `ダミー確保中: ${gib(state.dummyBytes)}（${state.dummies.length} 本）`;
};

/** quant と計測の選択は GPU を持っている間は変えられない（{@link BuildChoice}）。 */
const renderBuildControls = (): void => {
  const locked = state.busy || state.gpu !== undefined;
  ui.quant.disabled = locked || state.model === undefined;
  ui.gpuTiming.disabled = locked || !state.timestampFeature;
};

const setBusy = (busy: boolean): void => {
  state.busy = busy;
  for (const button of [ui.run, ui.hold, ui.release, ui.dispose]) button.disabled = busy;
  ui.exportJson.disabled = busy || (state.rows.length === 0 && state.holds.length === 0);
  renderBuildControls();
};

/** いま選ばれている構成（GPU を取るときに {@link BuildChoice} として確定させる）。 */
const selectedChoice = (): BuildChoice => ({
  quant: ui.quant.value,
  gpuTiming: state.timestampFeature && ui.gpuTiming.checked,
});

/** ボタン操作の排他（同じ pipeline の generate / dispose を重ねない）。失敗は状態行へ。 */
const exclusive = (action: () => Promise<void>) => async (): Promise<void> => {
  if (state.busy) return;
  setBusy(true);
  try {
    await action();
  } catch (error) {
    const { name, message } = describeError(error);
    setStatus(`${name}: ${message}`);
  } finally {
    setBusy(false);
  }
};

const modelSource = localDirectory({
  readFile: async (path, options) => {
    const response = await fetch(`/models/anima/${path}`, options);
    if (!response.ok) throw Error(`Model HTTP ${response.status}: ${path}`);
    return new Uint8Array(await response.arrayBuffer());
  },
  readFileRange: async (path, offset, length, options) => {
    const response = await fetch(`/models/anima/${path}`, {
      ...options,
      headers: { Range: `bytes=${offset}-${offset + length - 1}` },
    });
    if (response.status !== 206) throw Error(`Model range HTTP ${response.status}: ${path}`);
    return new Uint8Array(await response.arrayBuffer());
  },
}, { label: "browser-anima-residency" });

const ensureGpu = async (): Promise<{ gpu: GpuContext; build: BuildChoice }> => {
  if (state.gpu !== undefined && state.build !== undefined) {
    return { gpu: state.gpu, build: state.build };
  }
  const build = selectedChoice();
  const quant = state.model?.quants[build.quant];
  if (quant === undefined) throw Error(`quant ${build.quant} が manifest に無い`);
  // 共有 GPU には pipeline が feature を足せないので、quant の宣言（shader-f16）はここで要求する。
  const gpu = await acquireGpu({
    ...(build.gpuTiming ? { gpuTiming: true } : {}),
    ...(quant.gpuFeatures?.shaderF16 === true ? { shaderF16: true } : {}),
    onDeviceLost: (info) => {
      state.deviceLost = { reason: info.reason, message: info.message };
      setStatus(
        `GPU device lost（${info.reason}）: ${info.message} — 「pipeline を破棄」でやり直せます`,
      );
    },
  });
  state.gpu = gpu;
  state.build = build;
  renderBuildControls();
  state.environment = `${adapterSummary(gpu.adapterInfo)} · 配布形 ${
    state.config?.source ?? "?"
  }（${state.defaultModel ?? "?"}）· quant ${build.quant} · GPU 時間 ${
    gpu.gpuTimingEnabled ? "採る" : "採らない"
  }`;
  ui.environment.textContent = state.environment;
  return { gpu, build };
};

/** pipeline の `onRunDiagnostics` → 進行中の generate の記録器。 */
const forwardRunDiagnostics = (
  ...[component, diagnostics]: Parameters<GenerateRecorder["onRun"]>
): void => {
  if (state.recorder === undefined) throw Error(`${component} の run が generate の外で終わった`);
  state.recorder.onRun(component, diagnostics);
};

const ensurePipeline = async (): Promise<{ pipeline: AnimaPipeline; build: BuildChoice }> => {
  const { gpu, build } = await ensureGpu();
  if (state.pipeline !== undefined) return { pipeline: state.pipeline, build };
  setStatus(`pipeline を構築中（residency: transformer · quant ${build.quant}）`);
  const started = performance.now();
  state.pipeline = await AnimaPipeline.fromPretrained(modelSource, {
    gpu,
    residency: "transformer",
    quant: build.quant,
    onRunDiagnostics: forwardRunDiagnostics,
  });
  const ms = performance.now() - started;
  state.pipelineLoads.push({
    at: new Date().toISOString(),
    ms,
    quant: build.quant,
    gpuTiming: build.gpuTiming,
  });
  setStatus(`pipeline 構築済み（${(ms / 1000).toFixed(2)} s）`);
  return { pipeline: state.pipeline, build };
};

const readRequest = (): Row["request"] => {
  const optionalNumber = (input: HTMLInputElement): number | undefined =>
    input.value.trim() === "" ? undefined : Number(input.value);
  const negativePrompt = ui.negative.value.trim();
  const steps = optionalNumber(ui.steps);
  const guidanceScale = optionalNumber(ui.guidance);
  return {
    prompt: ui.prompt.value,
    ...(negativePrompt === "" ? {} : { negativePrompt }),
    resolution: parseResolution(ui.resolution.value),
    ...(steps === undefined ? {} : { steps }),
    ...(guidanceScale === undefined ? {} : { guidanceScale }),
    seed: Number(ui.seed.value),
  };
};

const readResidency = (): AnimaResidency => {
  const value = ui.residency.value;
  if (value !== "transformer" && value !== "per-stage") throw Error(`Unknown residency ${value}`);
  return value;
};

const formatStages = (stages: readonly StageRecord[]): string =>
  stages.map(({ component, startMs, endMs }) =>
    `${component} ${endMs === undefined ? "(未完了)" : `${Math.round(endMs - startMs)} ms`}`
  ).join(" · ");

const formatResidency = (records: readonly ResidencyRecord[]): string =>
  records.length === 0
    ? "(無し)"
    : records.map(({ action, reason, atMs, position }) =>
      `${action}/${reason} @${(atMs / 1000).toFixed(2)} s（${position}）`
    ).join("\n");

const ms = (ns: number): string => (ns / 1e6).toFixed(1);

/** 段 1 回ぶんの上位 10 キー（キー・ms・dispatch 本数・段に占める %）。 */
const stageDetails = (component: string, gpu: StageGpuTiming): HTMLDetailsElement => {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `${component} ${ms(gpu.totalNs)} ms（run ${gpu.runs}${
    gpu.clampedNegativeSamples === 0 ? "" : ` · 負の標本 ${gpu.clampedNegativeSamples}`
  }${looksQuantized(gpu) ? " · 100 µs 量子化の疑い" : ""}）`;
  const table = document.createElement("table");
  table.className = "keys";
  const head = document.createElement("tr");
  for (const title of ["key", "ms", "dispatch", "%"]) {
    const th = document.createElement("th");
    th.textContent = title;
    head.append(th);
  }
  table.append(head);
  for (const entry of topEntries(gpu, 10)) {
    const tr = document.createElement("tr");
    for (
      const value of [
        entry.key,
        ms(entry.ns),
        entry.dispatchCount.toLocaleString(),
        (entry.share * 100).toFixed(1),
      ]
    ) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    table.append(tr);
  }
  details.append(summary, table);
  return details;
};

const gpuCell = (stages: readonly StageRecord[]): HTMLTableCellElement => {
  const td = document.createElement("td");
  td.className = "wrap";
  const timed = stages.filter((stage) => stage.gpu !== undefined);
  if (timed.length === 0) td.textContent = "—";
  for (const { component, gpu } of timed) {
    if (gpu !== undefined) td.append(stageDetails(component, gpu));
  }
  return td;
};

/** 全段で同じ id なら 1 つ、割れていれば段ごとに全部（ADR 0115 の幾何プロファイル）。 */
const geometryCell = (stages: readonly StageRecord[]): HTMLTableCellElement => {
  const td = document.createElement("td");
  const ids = geometryProfilesOf(stages);
  if (ids.length > 1) {
    td.className = "wrap";
    td.textContent = stages.flatMap(({ component, geometryProfile }) =>
      geometryProfile === undefined ? [] : [`${component} ${geometryProfile}`]
    ).join("\n");
  } else {
    td.textContent = ids[0] ?? "—";
  }
  return td;
};

/** 環境行に直近の generate で選ばれた幾何プロファイルを足す（run が 1 回も終わらなければ据え置き）。 */
const showGeometryProfiles = (row: Row): void => {
  const ids = geometryProfilesOf(row.stages);
  if (ids.length === 0 || state.environment === undefined) return;
  ui.environment.textContent = `${state.environment} · 幾何プロファイル ${ids.join(" / ")}`;
};

const appendRow = (row: Row): void => {
  const tr = document.createElement("tr");
  const cells = [
    String(row.index),
    row.quant,
    row.residencyRequested,
    gib(row.dummyBytesHeld),
    Math.round(row.wallMs).toLocaleString(),
    formatStages(row.stages),
    formatResidency(row.residency),
    row.pngSha256?.slice(0, 12) ?? "—",
    row.error === undefined ? "" : `${row.error.name}: ${row.error.message}`,
  ];
  for (const [at, value] of cells.entries()) {
    const td = document.createElement("td");
    td.textContent = value;
    if (at >= 5) td.className = "wrap";
    if (at === 8 && value !== "") td.className = "wrap error";
    tr.append(td);
    // 「GPU 時間」列は段の時間の隣に置く（壁と GPU を同じ段で見比べるため）。その隣が幾何プロファイル。
    if (at === 5) tr.append(gpuCell(row.stages), geometryCell(row.stages));
  }
  ui.rows.append(tr);
};

const showImage = (png: Uint8Array<ArrayBuffer>): void => {
  if (state.imageUrl !== undefined) URL.revokeObjectURL(state.imageUrl);
  state.imageUrl = URL.createObjectURL(new Blob([png], { type: "image/png" }));
  ui.image.src = state.imageUrl;
  ui.image.hidden = false;
};

const generateOnce = async (label: string): Promise<Row> => {
  const index = state.rows.length + 1;
  const residencyRequested = readResidency();
  const request = readRequest();
  const dummyBytesHeld = state.dummyBytes;
  const recorder = createGenerateRecorder(() => performance.now());
  const onEvent = (event: AnimaGenerateEvent): void => {
    recorder.onEvent(event);
    if (event.kind === "stage") setStatus(`${label}: ${event.component} ${event.at}`);
    else if (event.kind === "denoise-step") {
      setStatus(`${label}: transformer step ${event.step}/${event.steps}`);
    } else if (event.kind === "vae-tile") {
      setStatus(`${label}: vae tile ${event.tile}/${event.tiles}`);
    }
  };
  let quant = state.build?.quant ?? selectedChoice().quant;
  state.recorder = recorder;
  try {
    const built = await ensurePipeline();
    quant = built.build.quant;
    const generateRequest: AnimaGenerateRequest = {
      ...request,
      residency: residencyRequested,
      onEvent,
    };
    recorder.restart();
    const image = await built.pipeline.generate(generateRequest);
    const wallMs = recorder.elapsedMs();
    const png = await encodePng(image.data, image.width, image.height);
    showImage(png);
    return {
      index,
      quant,
      residencyRequested,
      request,
      dummyBytesHeld,
      wallMs,
      ...recorder.finish(),
      pngSha256: await sha256Hex(png),
    };
  } catch (error) {
    return {
      index,
      quant,
      residencyRequested,
      request,
      dummyBytesHeld,
      wallMs: recorder.elapsedMs(),
      ...recorder.finish(),
      error: describeError(error),
    };
  } finally {
    state.recorder = undefined;
  }
};

const runGenerates = async (): Promise<void> => {
  const count = Number(ui.count.value);
  if (!Number.isInteger(count) || count < 1) {
    throw Error(`生成回数 ${ui.count.value} が正の整数でない`);
  }
  for (let i = 0; i < count; i++) {
    const row = await generateOnce(`generate ${i + 1}/${count}`);
    state.rows.push(row);
    appendRow(row);
    showGeometryProfiles(row);
    if (row.error !== undefined) {
      setStatus(`generate ${i + 1}/${count} が失敗したので止めました（表の行を参照）`);
      return;
    }
  }
  const last = state.rows.at(-1);
  setStatus(
    `${count} 回完了 — 最後: ${Math.round(last?.wallMs ?? 0)} ms / sha ${
      last?.pngSha256?.slice(0, 12)
    }`,
  );
};

const holdVram = async (): Promise<void> => {
  const requestedGib = Number(ui.holdGib.value);
  if (!Number.isFinite(requestedGib) || requestedGib <= 0) {
    throw Error(`ダミー量 ${ui.holdGib.value} GiB が正の数でない`);
  }
  const { gpu } = await ensureGpu();
  const target = Math.round(requestedGib * 1024) * MIB;
  // 1 GiB ずつ（maxBufferSize がそれより小さければその大きさで — 4 バイト整列）。
  const pieceMax = Math.floor(Math.min(gpu.limits.maxBufferSize, GIB) / 4) * 4;
  let allocated = 0;
  let buffers = 0;
  let stop: string | undefined;
  while (allocated < target) {
    const size = Math.min(pieceMax, target - allocated);
    gpu.device.pushErrorScope("out-of-memory");
    const buffer = gpu.device.createBuffer({
      label: `anima-residency-dummy-${state.dummies.length}`,
      size,
      usage: GPUBufferUsage.STORAGE,
    });
    const failure = await gpu.device.popErrorScope();
    if (failure !== null) {
      buffer.destroy();
      stop = `${gib(allocated)} で確保失敗: ${failure.message}`;
      break;
    }
    state.dummies.push(buffer);
    state.dummyBytes += size;
    allocated += size;
    buffers++;
    renderDummies();
  }
  state.holds.push({
    at: new Date().toISOString(),
    requestedGib,
    allocatedBytes: allocated,
    buffers,
    ...(stop === undefined ? {} : { stop }),
  });
  renderDummies();
  setStatus(
    `ダミーを ${gib(allocated)} 確保（要求 ${requestedGib} GiB）${
      stop === undefined ? "" : ` — ${stop}`
    }`,
  );
};

const releaseDummies = async (): Promise<void> => {
  const bytes = state.dummyBytes;
  for (const buffer of state.dummies) buffer.destroy();
  state.dummies = [];
  state.dummyBytes = 0;
  await state.gpu?.device.queue.onSubmittedWorkDone();
  renderDummies();
  setStatus(`ダミー ${gib(bytes)} を解放しました`);
};

/** pipeline・ダミー・共有 GPU をまとめて畳む（device lost の後のやり直しもここから）。 */
const disposeAll = async (): Promise<void> => {
  const pipeline = state.pipeline;
  state.pipeline = undefined;
  let failure: string | undefined;
  try {
    await pipeline?.dispose();
  } catch (error) {
    const { name, message } = describeError(error);
    failure = `${name}: ${message}`;
  }
  for (const buffer of state.dummies) buffer.destroy();
  state.dummies = [];
  state.dummyBytes = 0;
  state.gpu?.destroy();
  state.gpu = undefined;
  state.build = undefined;
  state.deviceLost = undefined;
  renderDummies();
  renderBuildControls();
  setStatus(
    `pipeline・ダミー・GPU device を破棄しました${
      failure === undefined ? "" : `（dispose の失敗: ${failure}）`
    }。次の生成で組み直します（quant と GPU 時間の選択はここで変えられます）。`,
  );
};

const exportJson = (): void => {
  // pipeline を破棄した後（quant を替える途中）でも機体を残す — 初期化時に読んだアダプタで補う。
  const info = state.gpu?.adapterInfo ?? state.adapterInfo;
  const current = state.build ?? selectedChoice();
  const report: Report = {
    format: REPORT_FORMAT,
    date: new Date().toISOString(),
    userAgent: navigator.userAgent,
    adapter: info === undefined ? null : {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    },
    checkout: state.config?.revision,
    checkoutDirty: state.config?.dirty,
    bundleSha256: state.config?.bundleSha256,
    source: state.config?.source,
    manifestSha256: state.manifestSha256,
    defaultModel: state.defaultModel,
    quant: current.quant,
    gpuTiming: { enabled: current.gpuTiming, feature: state.timestampFeature, unit: "ns" },
    pipelineResidency: "transformer",
    pipelineLoads: state.pipelineLoads,
    dummies: { heldBytes: state.dummyBytes, buffers: state.dummies.length, holds: state.holds },
    deviceLost: state.deviceLost ?? null,
    rows: state.rows,
  };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `anima-residency-browser-${current.quant}-${
    new Date().toISOString().replaceAll(":", "-")
  }.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

/** quant の選択肢を manifest の既定モデルの欄から埋める（既定 = `defaultQuant`）。 */
const fillQuants = (model: ModelEntry): void => {
  for (const [name, quant] of Object.entries(model.quants)) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = quant.label === undefined ? name : `${name}（${quant.label}）`;
    option.selected = name === model.defaultQuant;
    ui.quant.append(option);
  }
};

const initialize = async (): Promise<void> => {
  setBusy(true);
  ui.prompt.value = DEFAULT_PROMPT;
  const configResponse = await fetch("/config.json");
  if (!configResponse.ok) throw Error(`config.json HTTP ${configResponse.status}`);
  const config: ServerConfig = await configResponse.json();
  state.config = config;
  const manifestResponse = await fetch("/models/anima/karume.json");
  if (!manifestResponse.ok) {
    throw Error(`配布形の karume.json が HTTP ${manifestResponse.status}（--source を確認）`);
  }
  const manifestBytes = await manifestResponse.arrayBuffer();
  state.manifestSha256 = await sha256Hex(new Uint8Array(manifestBytes));
  const manifest = parseManifest(new TextDecoder().decode(manifestBytes));
  state.defaultModel = manifest.defaultModel;
  const model = manifest.models[manifest.defaultModel];
  if (model === undefined) throw Error(`defaultModel ${manifest.defaultModel} が models に無い`);
  state.model = model;
  fillQuants(model);
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) {
    throw Error("ハードウェア WebGPU を使える Chrome が必要です");
  }
  state.adapterInfo = adapter.info;
  state.timestampFeature = adapter.features.has(TIMESTAMP_QUERY);
  ui.gpuTiming.checked = state.timestampFeature;
  ui.gpuTimingNote.textContent = state.timestampFeature
    ? ""
    : `（このアダプタは ${TIMESTAMP_QUERY} を持たないので採れません — 段の壁時計と dispatch 本数だけ記録します）`;
  ui.environment.textContent = `${adapterSummary(adapter.info)} · 配布形 ${config.source}（${
    state.defaultModel ?? "?"
  }）· ${config.revision.slice(0, 8)}${config.dirty ? " (dirty)" : ""}`;
  renderDummies();
  setStatus("準備完了。「N 回生成」で最初の generate が pipeline を組みます。");
  setBusy(false);
};

ui.run.addEventListener("click", exclusive(runGenerates));
ui.hold.addEventListener("click", exclusive(holdVram));
ui.release.addEventListener("click", exclusive(releaseDummies));
ui.dispose.addEventListener("click", exclusive(disposeAll));
ui.exportJson.addEventListener("click", exportJson);
initialize().catch((error: unknown) => {
  const { name, message } = describeError(error);
  setStatus(`${name}: ${message}`);
});
