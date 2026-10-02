/// <reference lib="dom" />
/**
 * Anima のタブ — anima の DiT 常駐（ADR 0112）と段ごとの op 別 GPU 時間（perf-ledger K-70）を Chrome で
 * 確かめる（旧 `tools/anima-residency` のブラウザのページの移植・表の列・JSON の欄・保存名は同じ）。
 *
 * 1 本の GPU（`acquireGpu()`）に pipeline（`residency: "transformer"` で 1 度だけ組んで使い回す）と
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
 * GPU は**ページの GPU 設定**（幾何プロファイルの注入と timestamp の要求 — `main.ts`）と quant で取る。
 * 3 つとも GPU を取るときに確定する（feature は device 作成時にしか要求できない・注入は GPU 単位 —
 * `acquireGpu` の `gpuTiming` / `shaderF16` / `geometryProfile`）。計測が有効なら `onRunDiagnostics` の
 * `lastRunTiming` を段ごとに足し、`lastRunPipelines`（計測に依らない dispatch 本数）は常に足す
 * （`tools/anima-residency/timing.ts`）。
 *
 * 配布形が無い（`/config.json` の `source` が null）ときは操作を全て無効にして、状態行でそう告げる
 * （掃引とプロファイルのタブはモデル無しで使える）。
 */
import {
  acquireGpu,
  type GeometryProfile,
  type GpuContext,
} from "../../../packages/runtime/mod.ts";
import { localDirectory, type ModelEntry, parseManifest } from "../../../packages/hub/mod.ts";
import {
  type AnimaGenerateEvent,
  type AnimaGenerateRequest,
  AnimaPipeline,
  type AnimaResidency,
  parseResolution,
} from "../../../packages/models/anima.ts";
import { encodePng } from "../../../packages/models/mod.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../../../packages/runtime/src/kernels/geometry-profile.ts";
import { infinityJson } from "../../../packages/runtime/src/tune/derive.ts";
import {
  createGenerateRecorder,
  DEFAULT_PROMPT,
  type DummyHold,
  type GenerateRecorder,
  type GeometryProfileRequested,
  geometryProfilesOf,
  type PipelineLoad,
  type Report,
  REPORT_FORMAT,
  type ResidencyRecord,
  type Row,
  type StageRecord,
} from "../../anima-residency/record.ts";
import { looksQuantized, type StageGpuTiming, topEntries } from "../../anima-residency/timing.ts";
import {
  abQuantPlan,
  abQuantsStatusLine,
  type AbQuantSummary,
  type AbSummary,
  abTableRows,
  summarizeAb,
} from "./ab-summary.ts";
import {
  adapterSummary,
  checkoutLabel,
  describeError,
  downloadText,
  element,
  injectedProfile,
  type Lab,
  type ProfileChoice,
  requestedLabel,
  setStatus,
  sha256Hex,
  TIMESTAMP_QUERY,
} from "./common.ts";
import {
  discardSavedIfUnchanged,
  resolveSavedProfile,
  type SavedResolution,
} from "./injectable-tables.ts";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

/** A/B の区間 A の選択（既定の表を注入 — 「表を入れなかったら」の基準）。 */
const AB_BASELINE: ProfileChoice = { kind: "default", profile: DEFAULT_GEOMETRY_PROFILE };

/**
 * GPU を取った時点で確定する構成（変えるには「pipeline を破棄」— GPU ごと畳む。幾何プロファイルと
 * GPU 時間はページの GPU 設定の「適用」でも取り直す）。
 */
type BuildChoice = {
  readonly quant: string;
  readonly gpuTiming: boolean;
  /** 記録に残す綴り（`auto` / `default` / `builtin:<id>` / `generated:<id>` / `saved:<id>`）。 */
  readonly geometryProfileRequested: GeometryProfileRequested;
  /** 注入した表（`saved-matched` は GPU を取った後に照合が一致したときだけ入る）。 */
  readonly geometryProfile?: GeometryProfile;
  /**
   * `saved-matched` の保存物の文字列（GPU 設定の適用時に取ったもの）。GPU を取るときのコールバックで照合する
   * （{@link resolveSavedProfile}）。
   */
  readonly savedStored?: string;
  /** `saved-matched` の照合の結果の 1 行（GPU を取った後に入る — 情報行と状態行に出す）。 */
  readonly savedNote?: string;
};

export type AnimaTab = {
  /**
   * pipeline・ダミー・GPU を畳む（GPU 設定の適用の前段）。実行中なら投げる。返り値 = GPU を
   * 持っていたか（持っていたなら適用の後に取り直す）。
   */
  readonly reset: () => Promise<boolean>;
  /** 適用中の GPU 設定と選択中の quant で GPU を取る（配布形が読めていなければ投げる）。 */
  readonly acquire: () => Promise<void>;
};

export const mountAnimaTab = (root: HTMLElement, lab: Lab): AnimaTab => {
  const ui = {
    prompt: element(root, "prompt", HTMLTextAreaElement),
    negative: element(root, "negative", HTMLInputElement),
    resolution: element(root, "resolution", HTMLSelectElement),
    steps: element(root, "steps", HTMLInputElement),
    guidance: element(root, "guidance", HTMLInputElement),
    seed: element(root, "seed", HTMLInputElement),
    residency: element(root, "residency", HTMLSelectElement),
    quant: element(root, "quant", HTMLSelectElement),
    count: element(root, "count", HTMLInputElement),
    run: element(root, "run", HTMLButtonElement),
    ab: element(root, "ab", HTMLButtonElement),
    holdGib: element(root, "hold-gib", HTMLInputElement),
    hold: element(root, "hold", HTMLButtonElement),
    release: element(root, "release", HTMLButtonElement),
    dispose: element(root, "dispose", HTMLButtonElement),
    exportJson: element(root, "export", HTMLButtonElement),
    status: element(root, "status", HTMLElement),
    info: element(root, "info", HTMLElement),
    dummies: element(root, "dummies", HTMLElement),
    rows: element(root, "rows", HTMLTableSectionElement),
    abSummary: element(root, "ab-summary", HTMLElement),
    image: element(root, "image", HTMLImageElement),
  };

  const state: {
    manifestSha256?: string;
    defaultModel?: string;
    /** 既定モデルの manifest の欄（quant の選択肢と `gpuFeatures`）。 */
    model?: ModelEntry;
    /** GPU を取ったときに書いた情報行（直近の generate の幾何プロファイルを足す土台）。 */
    info?: string;
    gpu?: GpuContext;
    /**
     * A/B の区間の間だけ、ページの GPU 設定の代わりに GPU を取る幾何プロファイルの選び方（区間 A の
     * `default`）。ヘッダの適用状態は動かさない — 区間 A は A/B の内部の一時的な取り直しだから。
     */
    choiceOverride?: ProfileChoice;
    /**
     * A/B の quant ごとの区間の間だけ、select の代わりに GPU を取る quant。`choiceOverride` と同じく select の
     * 表示は動かさない — quant の巡回は A/B の内部の一時的な取り直しだから。
     */
    quantOverride?: string;
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
    dummies: [],
    dummyBytes: 0,
    holds: [],
    pipelineLoads: [],
    rows: [],
    busy: false,
  };

  const status = (text: string): void => setStatus(ui.status, text);

  const source = lab.config.source;
  if (source === null) {
    for (
      const control of [
        ui.run,
        ui.ab,
        ui.hold,
        ui.release,
        ui.dispose,
        ui.exportJson,
        ui.quant,
        ui.prompt,
        ui.negative,
        ui.resolution,
        ui.steps,
        ui.guidance,
        ui.seed,
        ui.residency,
        ui.count,
        ui.holdGib,
      ]
    ) control.disabled = true;
    ui.info.textContent = `${adapterSummary(lab.adapterInfo)} · 配布形 無し · ${
      checkoutLabel(lab.config)
    }`;
    status("配布形が無い（--source で指定）");
    return {
      // GPU を取らないので畳むものも無い
      reset: () => Promise.resolve(false),
      acquire: () => Promise.reject(Error("配布形が無い（--source で指定）ので GPU を取れない")),
    };
  }

  const gib = (bytes: number): string => `${(bytes / GIB).toFixed(2)} GiB`;

  const renderDummies = (): void => {
    ui.dummies.textContent = `ダミー確保中: ${gib(state.dummyBytes)}（${state.dummies.length} 本）`;
  };

  /** quant の選択は GPU を持っている間は変えられない（{@link BuildChoice}）。 */
  const renderBuildControls = (): void => {
    const locked = state.busy || state.gpu !== undefined;
    ui.quant.disabled = locked || state.model === undefined;
  };

  const setBusy = (busy: boolean): void => {
    state.busy = busy;
    for (const button of [ui.run, ui.ab, ui.hold, ui.release, ui.dispose]) button.disabled = busy;
    // 生成の条件は押下時に 1 度だけ読む（{@link readCondition}）— 実行中に入力を変えても効かないので、変えられる
    // ように見せない
    for (
      const input of [
        ui.prompt,
        ui.negative,
        ui.resolution,
        ui.steps,
        ui.guidance,
        ui.seed,
        ui.residency,
        ui.count,
        ui.holdGib,
      ]
    ) input.disabled = busy;
    ui.exportJson.disabled = busy || (state.rows.length === 0 && state.holds.length === 0);
    renderBuildControls();
  };

  /** いま選ばれている構成（GPU を取るときに {@link BuildChoice} として確定させる）。 */
  const selectedChoice = (): BuildChoice => {
    const settings = lab.settings();
    const choice = state.choiceOverride ?? settings.choice;
    const timestamps = settings.timestamps;
    const geometryProfile = injectedProfile(choice);
    return {
      quant: state.quantOverride ?? ui.quant.value,
      gpuTiming: lab.timestampFeature && timestamps,
      geometryProfileRequested: requestedLabel(choice),
      ...(geometryProfile === undefined ? {} : { geometryProfile }),
      ...(choice.kind === "saved-matched" ? { savedStored: choice.stored } : {}),
    };
  };

  /** 「保存した表（照合して注入）」の結果の 1 行（情報行と状態行に出す）。 */
  const savedResolutionText = (resolution: SavedResolution): string =>
    resolution.kind === "matched"
      ? `保存した表 ${resolution.id} は adapter と一致 → 注入`
      : `保存した表を注入しない（自動で選ぶ）— ${resolution.reason}`;

  /**
   * 読めなかった保存物を捨てた結果の句（次の起動でも読み続けない — ADR 0117 決定 10）。照合した文字列と今の値が
   * 違えば消さない（適用から GPU を取るまでの間に保存し直された表を消さない — {@link discardSavedIfUnchanged}）。
   */
  const discardBrokenSaved = (inspected: string): string => {
    try {
      return discardSavedIfUnchanged(localStorage, inspected)
        ? "読めない保存物を消した"
        : "照合の後に表が保存し直されていたので、保存物は消さない";
    } catch {
      // 捨てられなくても実行は既定で進む（理由はこの句で情報行と状態行に出る）
      return "読めない保存物を消せなかった";
    }
  };

  /** 記録に残す幾何プロファイルの欄（要求の綴りと、注入していればその表の値そのもの）。 */
  const requested = (
    build: BuildChoice,
  ): Pick<Row, "geometryProfileRequested" | "geometryProfileInjected"> => ({
    geometryProfileRequested: build.geometryProfileRequested,
    ...(build.geometryProfile === undefined
      ? {}
      : { geometryProfileInjected: build.geometryProfile }),
  });

  /**
   * ボタン操作の排他（同じ pipeline の generate / dispose を重ねない・ページの他の GPU 操作とも
   * 重ねない）。失敗は状態行へ。
   */
  const exclusive = (action: () => Promise<void>) => async (): Promise<void> => {
    if (state.busy) return;
    let release: () => void;
    try {
      release = lab.lock("Anima の操作");
    } catch (error) {
      const { name, message } = describeError(error);
      status(`${name}: ${message}`);
      return;
    }
    setBusy(true);
    try {
      await action();
    } catch (error) {
      const { name, message } = describeError(error);
      status(`${name}: ${message}`);
    } finally {
      setBusy(false);
      release();
    }
  };

  // 相対 path にするのは、同じページと bundle を静的な置き場にそのまま載せるため。
  const modelSource = localDirectory({
    readFile: async (path, options) => {
      const response = await fetch(`models/anima/${path}`, options);
      if (!response.ok) throw Error(`Model HTTP ${response.status}: ${path}`);
      return new Uint8Array(await response.arrayBuffer());
    },
    readFileRange: async (path, offset, length, options) => {
      const response = await fetch(`models/anima/${path}`, {
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
    const selected = selectedChoice();
    const quant = state.model?.quants[selected.quant];
    if (quant === undefined) throw Error(`quant ${selected.quant} が manifest に無い`);
    // 「保存した表（照合して注入）」はアプリの流れのまま: 保存物は適用時に読んである（コールバックで I/O を
    // 待たない）。runtime が実際に取った adapter の情報で照合し、一致した表だけを返す（不一致は undefined =
    // 自動選択 — ADR 0117 決定 6 / 10）
    const { savedStored } = selected;
    let resolution: SavedResolution | undefined;
    const geometryProfile = savedStored === undefined
      ? selected.geometryProfile
      : (adapterInfo: GPUAdapterInfo): GeometryProfile | undefined => {
        resolution = resolveSavedProfile(savedStored, adapterInfo);
        return resolution.kind === "matched" ? resolution.profile : undefined;
      };
    // 共有 GPU には pipeline が feature を足せないので、quant の宣言（shader-f16）はここで要求する。
    const gpu = await acquireGpu({
      ...(selected.gpuTiming ? { gpuTiming: true } : {}),
      ...(quant.gpuFeatures?.shaderF16 === true ? { shaderF16: true } : {}),
      ...(geometryProfile === undefined ? {} : { geometryProfile }),
      onDeviceLost: (info) => {
        state.deviceLost = { reason: info.reason, message: info.message };
        status(
          `GPU device lost（${info.reason}）: ${info.message} — 「pipeline を破棄」でやり直せます`,
        );
      },
    });
    // 記録の注入欄は照合の結果で決まる（一致した表だけ — 不一致は注入なしの行になる）
    const discardNote = resolution?.kind === "broken" && savedStored !== undefined
      ? discardBrokenSaved(savedStored)
      : undefined;
    const savedNote = resolution === undefined
      ? undefined
      : `${savedResolutionText(resolution)}${
        discardNote === undefined ? "" : `（${discardNote}）`
      }`;
    const build: BuildChoice = {
      ...selected,
      ...(resolution?.kind === "matched" ? { geometryProfile: resolution.profile } : {}),
      ...(savedNote === undefined ? {} : { savedNote }),
    };
    state.gpu = gpu;
    state.build = build;
    renderBuildControls();
    state.info = `${adapterSummary(gpu.adapterInfo)} · 配布形 ${source}（${
      state.defaultModel ?? "?"
    }）· quant ${build.quant} · GPU 時間 ${
      gpu.gpuTimingEnabled ? "採る" : "採らない"
    } · 幾何プロファイルの要求 ${build.geometryProfileRequested}${
      savedNote === undefined ? "" : `（${savedNote}）`
    }`;
    ui.info.textContent = state.info;
    if (savedNote !== undefined) status(savedNote);
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
    status(`pipeline を構築中（residency: transformer · quant ${build.quant}）`);
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
    status(`pipeline 構築済み（${(ms / 1000).toFixed(2)} s）`);
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

  /** 1 回の操作（「N 回生成」・A/B）の全 generate に渡す条件（生成要求と residency）。 */
  type GenerateCondition = {
    readonly request: Row["request"];
    readonly residency: AnimaResidency;
  };

  /**
   * 生成の条件を入力から 1 度だけ読む。MUST: 操作の頭（押下時）で読み、全 generate（A/B なら全 quant・全区間）へ
   * 同じ値を渡す — generate ごとに読み直すと、長い操作の途中で変えた入力が区間の間・区間の中で条件を割り、
   * B ÷ A が別条件どうしの比になる（residency は PNG を変えないので sha でも気づけない）。
   */
  const readCondition = (): GenerateCondition => ({
    request: readRequest(),
    residency: readResidency(),
  });

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

  /** 情報行に直近の generate で選ばれた幾何プロファイルを足す（run が 1 回も終わらなければ据え置き）。 */
  const showGeometryProfiles = (row: Row): void => {
    const ids = geometryProfilesOf(row.stages);
    if (ids.length === 0 || state.info === undefined) return;
    ui.info.textContent = `${state.info} · 幾何プロファイル ${ids.join(" / ")}`;
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

  const generateOnce = async (label: string, condition: GenerateCondition): Promise<Row> => {
    const index = state.rows.length + 1;
    const { request, residency: residencyRequested } = condition;
    const dummyBytesHeld = state.dummyBytes;
    const recorder = createGenerateRecorder(() => performance.now());
    const onEvent = (event: AnimaGenerateEvent): void => {
      recorder.onEvent(event);
      if (event.kind === "stage") status(`${label}: ${event.component} ${event.at}`);
      else if (event.kind === "denoise-step") {
        status(`${label}: transformer step ${event.step}/${event.steps}`);
      } else if (event.kind === "vae-tile") {
        status(`${label}: vae tile ${event.tile}/${event.tiles}`);
      }
    };
    let build = state.build ?? selectedChoice();
    state.recorder = recorder;
    try {
      const built = await ensurePipeline();
      build = built.build;
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
        quant: build.quant,
        ...requested(build),
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
        quant: build.quant,
        ...requested(build),
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

  const readCount = (): number => {
    const count = Number(ui.count.value);
    if (!Number.isInteger(count) || count < 1) {
      throw Error(`生成回数 ${ui.count.value} が正の整数でない`);
    }
    return count;
  };

  /**
   * N 回の generate を表に積み、この回で積んだ行を返す（失敗した行で止める）。`label` は進捗の前置
   * （A/B の区間名）。
   */
  const generateBatch = async (
    count: number,
    label: string,
    condition: GenerateCondition,
  ): Promise<Row[]> => {
    const rows: Row[] = [];
    for (let i = 0; i < count; i++) {
      const row = await generateOnce(`${label}generate ${i + 1}/${count}`, condition);
      state.rows.push(row);
      appendRow(row);
      showGeometryProfiles(row);
      rows.push(row);
      if (row.error !== undefined) break;
    }
    return rows;
  };

  const runGenerates = async (): Promise<void> => {
    const count = readCount();
    const rows = await generateBatch(count, "", readCondition());
    if (rows.at(-1)?.error !== undefined) {
      status(`generate ${rows.length}/${count} が失敗したので止めました（表の行を参照）`);
      return;
    }
    const last = state.rows.at(-1);
    status(
      `${count} 回完了 — 最後: ${Math.round(last?.wallMs ?? 0)} ms / sha ${
        last?.pngSha256?.slice(0, 12)
      }`,
    );
  };

  /** A/B の要約の表 1 つ（quant 1 つ分）。 */
  const abSummaryTable = (summary: AbSummary, labelB: string): HTMLTableElement => {
    const table = document.createElement("table");
    const head = document.createElement("tr");
    for (
      const title of [
        "",
        `区間 A（${requestedLabel(AB_BASELINE)}）`,
        `区間 B（${labelB}）`,
        "B ÷ A",
      ]
    ) {
      const th = document.createElement("th");
      th.textContent = title;
      head.append(th);
    }
    const thead = document.createElement("thead");
    thead.append(head);
    const tbody = document.createElement("tbody");
    for (const cells of abTableRows(summary)) {
      const tr = document.createElement("tr");
      for (const value of cells) {
        const td = document.createElement("td");
        td.textContent = value;
        if (value.startsWith("失敗")) td.className = "error";
        else if (value.includes("不一致")) td.className = "bad";
        tr.append(td);
      }
      tbody.append(tr);
    }
    table.append(thead, tbody);
    return table;
  };

  /**
   * A/B の要約の表を quant ごとに縦に並べる（状態行が次の操作で上書きされても読めるように、行の表の下に
   * 残す）。
   */
  const renderAbSummary = (results: readonly AbQuantSummary[], labelB: string): void => {
    const time = new Date().toLocaleTimeString();
    ui.abSummary.replaceChildren(...results.flatMap(({ quant, summary }) => {
      const caption = document.createElement("p");
      caption.className = "muted";
      caption.textContent =
        `A/B の要約 — quant ${quant}（${time}）— 時間は各区間の 2 回目以降の中央値（N = 1 なら 1 回目）。B ÷ A が 1 未満なら区間 B が速い。`;
      return [caption, abSummaryTable(summary, labelB)];
    }));
    ui.abSummary.hidden = false;
  };

  /**
   * 同じ設定のまま、区間 A（`default` を注入）→ 区間 B（適用中の選択）を N 回ずつ回して要約する。表は
   * device 単位で固定（ADR 0115 追記決定 6）なので、各区間の前に pipeline・GPU を畳んでその区間の選択で
   * 取り直す（常駐 DiT も区間をまたがない）。取り直しは「適用」と同じ手順（`disposeAll` → 次の generate の
   * `ensurePipeline`）で、区間 A の選択はタブの中だけで差し替える — ヘッダの適用状態と select は動かさない。
   * quant は既定の席と `f16` の 2 つ（{@link abQuantPlan}）を外側のループにして quant ごとに独立した A/B を回す
   * （quant も GPU を取った時点で確定する — {@link BuildChoice} — ので、区間の前の取り直しでそのまま切り替わる）。
   */
  const runAb = async (): Promise<void> => {
    const applied = lab.settings().choice;
    if (applied.kind === "default") {
      status(
        "適用中の幾何プロファイルが default なので A/B にならない（区間 B が区間 A と同じ）— ヘッダで別の選択を適用してから",
      );
      return;
    }
    if (state.dummies.length > 0) {
      // 取り直すとダミーも畳まれる — 区間 A だけダミー有りの GPU で回る形を作らない
      status(
        "ダミーを確保中なので A/B を回さない（区間ごとに GPU を取り直すとダミーも畳まれ、押す前と条件が変わる）— 「ダミーを解放」してから",
      );
      return;
    }
    const count = readCount();
    // 全 quant・全区間で同じ条件（readCondition の MUST）。区間の頭の disposeAll より前に読む — 壊れた入力は
    // GPU を畳む前に止まる
    const condition = readCondition();
    const selectedQuant = ui.quant.value;
    if (state.model === undefined) throw Error("Anima の配布形が読めていないので A/B を回せない");
    const quants = abQuantPlan(
      Array.from(ui.quant.options, (option) => option.value),
      state.model.defaultQuant,
    );
    const appliedLabel = requestedLabel(applied);
    const intervals: readonly { readonly name: string; readonly choice: ProfileChoice }[] = [
      { name: "A", choice: AB_BASELINE },
      { name: "B", choice: applied },
    ];
    const results: AbQuantSummary[] = [];
    ui.abSummary.hidden = true;
    try {
      for (const quant of quants) {
        state.quantOverride = quant;
        const rowsByInterval: Row[][] = [];
        for (const { name, choice } of intervals) {
          state.choiceOverride = choice;
          await disposeAll();
          const rows = await generateBatch(
            count,
            `A/B ${quant}・区間 ${name}（${requestedLabel(choice)}）`,
            condition,
          );
          rowsByInterval.push(rows);
          // 区間 A で失敗したらその quant の区間 B は回さない（「N 回生成」と同じく失敗で止める）。次の quant へは
          // 進む — quant ごとの A/B は GPU を取り直すので互いに独立
          if (rows.some((row) => row.error !== undefined)) break;
        }
        results.push({
          quant,
          summary: summarizeAb(rowsByInterval[0] ?? [], rowsByInterval[1] ?? []),
        });
      }
    } finally {
      state.choiceOverride = undefined;
      state.quantOverride = undefined;
      // A/B の途中の GPU（default を注入・select と違う quant）を持ったまま終えない — 以後の操作は適用中の
      // 設定と選ばれている quant の GPU で回る
      if (
        state.build !== undefined &&
        (state.build.geometryProfileRequested !== appliedLabel ||
          state.build.quant !== selectedQuant)
      ) {
        await disposeAll();
      }
    }
    renderAbSummary(results, appliedLabel);
    status(abQuantsStatusLine(results));
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
    status(
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
    status(`ダミー ${gib(bytes)} を解放しました`);
  };

  /** pipeline・ダミー・GPU をまとめて畳む（device lost の後のやり直しもここから）。 */
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
    status(
      `pipeline・ダミー・GPU device を破棄しました${
        failure === undefined ? "" : `（dispose の失敗: ${failure}）`
      }。次の生成で組み直します（quant の選択はここで変えられます）。`,
    );
  };

  const exportJson = (): void => {
    // pipeline を破棄した後（quant を替える途中）でも機体を残す — ページが初期化時に読んだアダプタで補う。
    const info = state.gpu?.adapterInfo ?? lab.adapterInfo;
    const current = state.build ?? selectedChoice();
    const report: Report = {
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
      source,
      manifestSha256: state.manifestSha256,
      defaultModel: state.defaultModel,
      quant: current.quant,
      ...requested(current),
      gpuTiming: { enabled: current.gpuTiming, feature: lab.timestampFeature, unit: "ns" },
      pipelineResidency: "transformer",
      pipelineLoads: state.pipelineLoads,
      dummies: { heldBytes: state.dummyBytes, buffers: state.dummies.length, holds: state.holds },
      deviceLost: state.deviceLost ?? null,
      rows: state.rows,
    };
    downloadText(
      // 注入した表の末尾の maxRows（Infinity）を null に落とさない（注入の JSON と同じ 1e999）
      infinityJson(report),
      `anima-residency-browser-${current.quant}-${
        new Date().toISOString().replaceAll(":", "-")
      }.json`,
      "application/json",
    );
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
    const manifestResponse = await fetch("models/anima/karume.json");
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
    ui.info.textContent = `${adapterSummary(lab.adapterInfo)} · 配布形 ${source}（${
      state.defaultModel ?? "?"
    }）· ${checkoutLabel(lab.config)}${
      lab.timestampFeature
        ? ""
        : ` · このアダプタは ${TIMESTAMP_QUERY} を持たないので GPU 時間は採れません（段の壁時計と dispatch 本数だけ記録します）`
    }`;
    renderDummies();
    status("準備完了。「N 回生成」で最初の generate が pipeline を組みます。");
    setBusy(false);
  };

  ui.run.addEventListener("click", exclusive(runGenerates));
  ui.ab.addEventListener("click", exclusive(runAb));
  ui.hold.addEventListener("click", exclusive(holdVram));
  ui.release.addEventListener("click", exclusive(releaseDummies));
  ui.dispose.addEventListener("click", exclusive(disposeAll));
  ui.exportJson.addEventListener("click", exportJson);
  initialize().catch((error: unknown) => {
    const { name, message } = describeError(error);
    status(`${name}: ${message}`);
  });

  return {
    reset: async () => {
      // 配布形が読めていない間は GPU を取れない（ensureGpu が quant を要る）ので、畳むものも無い
      if (state.model === undefined) return false;
      if (state.busy) throw Error("Anima のタブが実行中 — 終わってから適用する");
      const held = state.gpu !== undefined;
      setBusy(true);
      try {
        await disposeAll();
      } finally {
        setBusy(false);
      }
      return held;
    },
    acquire: async () => {
      if (state.model === undefined) throw Error("Anima の配布形が読めていないので GPU を取れない");
      setBusy(true);
      try {
        const { build } = await ensureGpu();
        status(
          `GPU を取り直しました（quant ${build.quant}・幾何プロファイル ${build.geometryProfileRequested}）${
            build.savedNote === undefined ? "" : ` — ${build.savedNote}`
          }`,
        );
      } finally {
        setBusy(false);
      }
    },
  };
};
