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
 */
import { acquireGpu, type GpuContext } from "../../../packages/runtime/mod.ts";
import { localDirectory, parseManifest } from "../../../packages/hub/mod.ts";
import {
  type AnimaGenerateEvent,
  type AnimaGenerateRequest,
  AnimaPipeline,
  type AnimaResidency,
  type AnimaResidencyAction,
  type AnimaResidencyReason,
  type AnimaRunComponent,
  parseResolution,
} from "../../../packages/models/anima.ts";
import { encodePng } from "../../../packages/models/mod.ts";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

type ServerConfig = {
  readonly revision: string;
  readonly dirty: boolean;
  readonly bundleSha256: string;
  readonly source: string;
};

/** 段 1 回ぶん（OOM 退避のやり直しでは同じ段が 2 回出る）。時刻は generate 開始からの ms。 */
type StageTiming = { component: AnimaRunComponent; startMs: number; endMs?: number };

type ResidencyRecord = {
  readonly atMs: number;
  readonly action: AnimaResidencyAction;
  readonly reason: AnimaResidencyReason;
  /** イベントが出た位置（開いている段 / 直前に閉じた段の後 / 最初の段の前）。 */
  readonly position: string;
};

type Row = {
  readonly index: number;
  readonly residencyRequested: AnimaResidency;
  readonly request: {
    readonly prompt: string;
    readonly negativePrompt?: string;
    readonly resolution: { readonly width: number; readonly height: number };
    readonly steps?: number;
    readonly guidanceScale?: number;
    readonly seed: number;
  };
  readonly dummyBytesHeld: number;
  readonly wallMs: number;
  readonly stages: readonly StageTiming[];
  readonly residency: readonly ResidencyRecord[];
  readonly pngSha256?: string;
  readonly error?: { readonly name: string; readonly message: string };
};

type DummyHold = {
  readonly at: string;
  readonly requestedGib: number;
  readonly allocatedBytes: number;
  readonly buffers: number;
  readonly stop?: string;
};

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
  gpu?: GpuContext;
  pipeline?: AnimaPipeline;
  deviceLost?: { readonly reason: string; readonly message: string };
  dummies: GPUBuffer[];
  dummyBytes: number;
  holds: DummyHold[];
  pipelineLoads: { readonly at: string; readonly ms: number }[];
  rows: Row[];
  busy: boolean;
  imageUrl?: string;
} = { dummies: [], dummyBytes: 0, holds: [], pipelineLoads: [], rows: [], busy: false };

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

const setBusy = (busy: boolean): void => {
  state.busy = busy;
  for (const button of [ui.run, ui.hold, ui.release, ui.dispose]) button.disabled = busy;
  ui.exportJson.disabled = busy || (state.rows.length === 0 && state.holds.length === 0);
};

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

const ensureGpu = async (): Promise<GpuContext> => {
  if (state.gpu !== undefined) return state.gpu;
  const gpu = await acquireGpu({
    onDeviceLost: (info) => {
      state.deviceLost = { reason: info.reason, message: info.message };
      setStatus(
        `GPU device lost（${info.reason}）: ${info.message} — 「pipeline を破棄」でやり直せます`,
      );
    },
  });
  state.gpu = gpu;
  ui.environment.textContent = `${adapterSummary(gpu.adapterInfo)} · 配布形 ${
    state.config?.source ?? "?"
  }（${state.defaultModel ?? "?"}）`;
  return gpu;
};

const ensurePipeline = async (): Promise<AnimaPipeline> => {
  if (state.pipeline !== undefined) return state.pipeline;
  const gpu = await ensureGpu();
  setStatus("pipeline を構築中（residency: transformer）");
  const started = performance.now();
  state.pipeline = await AnimaPipeline.fromPretrained(modelSource, {
    gpu,
    residency: "transformer",
  });
  const ms = performance.now() - started;
  state.pipelineLoads.push({ at: new Date().toISOString(), ms });
  setStatus(`pipeline 構築済み（${(ms / 1000).toFixed(2)} s）`);
  return state.pipeline;
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

const formatStages = (stages: readonly StageTiming[]): string =>
  stages.map(({ component, startMs, endMs }) =>
    `${component} ${endMs === undefined ? "(未完了)" : `${Math.round(endMs - startMs)} ms`}`
  ).join(" · ");

const formatResidency = (records: readonly ResidencyRecord[]): string =>
  records.length === 0
    ? "(無し)"
    : records.map(({ action, reason, atMs, position }) =>
      `${action}/${reason} @${(atMs / 1000).toFixed(2)} s（${position}）`
    ).join("\n");

const appendRow = (row: Row): void => {
  const tr = document.createElement("tr");
  const cells = [
    String(row.index),
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
    if (at >= 4) td.className = "wrap";
    if (at === 7 && value !== "") td.className = "wrap error";
    tr.append(td);
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
  const stages: StageTiming[] = [];
  const residency: ResidencyRecord[] = [];
  let started = performance.now();
  const position = (): string => {
    const open = stages.findLast((stage) => stage.endMs === undefined);
    if (open !== undefined) return `${open.component} の途中`;
    const last = stages.at(-1);
    return last === undefined ? "最初の段の前" : `${last.component} の後`;
  };
  const onEvent = (event: AnimaGenerateEvent): void => {
    const atMs = performance.now() - started;
    if (event.kind === "stage") {
      if (event.at === "start") stages.push({ component: event.component, startMs: atMs });
      else {
        const open = stages.findLast((stage) =>
          stage.component === event.component && stage.endMs === undefined
        );
        if (open !== undefined) open.endMs = atMs;
      }
      setStatus(`${label}: ${event.component} ${event.at}`);
    } else if (event.kind === "residency") {
      residency.push({ atMs, action: event.action, reason: event.reason, position: position() });
    } else if (event.kind === "denoise-step") {
      setStatus(`${label}: transformer step ${event.step}/${event.steps}`);
    } else setStatus(`${label}: vae tile ${event.tile}/${event.tiles}`);
  };
  const base = { index, residencyRequested, request, dummyBytesHeld };
  try {
    const pipeline = await ensurePipeline();
    const generateRequest: AnimaGenerateRequest = {
      ...request,
      residency: residencyRequested,
      onEvent,
    };
    started = performance.now();
    const image = await pipeline.generate(generateRequest);
    const wallMs = performance.now() - started;
    const png = await encodePng(image.data, image.width, image.height);
    showImage(png);
    return { ...base, wallMs, stages, residency, pngSha256: await sha256Hex(png) };
  } catch (error) {
    return {
      ...base,
      wallMs: performance.now() - started,
      stages,
      residency,
      error: describeError(error),
    };
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
  const gpu = await ensureGpu();
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
  state.deviceLost = undefined;
  renderDummies();
  setStatus(
    `pipeline・ダミー・GPU device を破棄しました${
      failure === undefined ? "" : `（dispose の失敗: ${failure}）`
    }。次の生成で組み直します。`,
  );
};

const exportJson = (): void => {
  const info = state.gpu?.adapterInfo;
  const report = {
    format: "karume-anima-residency-browser/1",
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
  anchor.download = `anima-residency-browser-${new Date().toISOString().replaceAll(":", "-")}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const initialize = async (): Promise<void> => {
  setBusy(true);
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
  state.defaultModel = parseManifest(new TextDecoder().decode(manifestBytes)).defaultModel;
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) {
    throw Error("ハードウェア WebGPU を使える Chrome が必要です");
  }
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
