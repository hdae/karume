/// <reference lib="dom" />
/**
 * Wan のタブ — Wan2.1 T2V 1.3B を Chrome で回す確認ページ（ADR 0118 段 9）。
 *
 * 確かめたいのは「Chrome の device で完走するか」と、完走しないならどこで止まるか（束縛上限・TDR〈GPU の
 * タイムアウト検出〉・VRAM）。そのために:
 *
 * 1. **事前判定**: アダプタ（と取得した device）の limits を、選んだフレーム数・寸法が要る値と並べる
 *    （`wan-plan.ts` の `judgeWanLimits` — 足りない項目は赤）。
 * 2. **読み込み**: GPU を取り（ページの GPU 設定の幾何プロファイル・`onDeviceLost`）、配布形を
 *    `WanPipeline.fromPretrained` で読む。取得元は既定でこのサーバが配る `models/karume-wan2.1`
 *    （`/models/wan/…` — HF の公開リポはまだ無い）、HF の `owner/name` も入れられる。テキストエンコーダの
 *    経路（ADR 0119 決定 7）はタブの既定が `precomputed`（埋め込み資産 — 段 9 の確認の既定のまま。パイプライン
 *    の既定は `gpu` なので、ここでは必ず明示して渡す）。`gpu` を選ぶと umT5 の越境先（Wan の manifest の
 *    `text_encoder` が宣言する repo）を、このサーバが配る `models/karume-umt5-xxl`（`/models/umt5/…`）へ取得元の
 *    `crossRepo` で結ぶ（HF の取得元なら宣言どおり HF から取る）。
 * 3. **生成**: プロンプト（埋め込み資産の名前 — `gpu` なら自由プロンプトの欄の文字列が優先）・seed・フレーム数・
 *    寸法・steps・guidance・shift で `generate` → 全フレームを canvas に描いて再生 → 所要（段・step・VAE タイル・
 *    `gpu` なら text 段）・Session の診断・RGB の sha256（事前計算の経路の参照ケースなら環境行との照合）を表に積む。
 *
 * 経路の選択と自由プロンプトの欄はページの HTML ではなくここで足す（`index.html` の Wan の節は事前計算の経路の
 * 欄だけを持つ）。
 *
 * GPU は自前で取って pipeline に渡す（共有 GPU）: device lost を `onDeviceLost` で表に残し、幾何プロファイルの
 * 注入をページの GPU 設定に揃えるため。`acquireGpu` は requiredLimits にアダプタ値を要求する（Chrome の既定
 * 128 MiB の束縛上限のままにしない）。計測（`gpuTiming`）は要求しない — `WanPipeline` は計測の device を
 * 構築時に拒む（VAE の段は 1 タイル = 1 batch で、runtime は計測の device で batch を開かない）。
 *
 * 失敗は表の行と状態行に出す（alert しない）— device lost の reason と文言をそのまま残すのが、この確認の
 * 一番の観測点だから。
 */
import {
  acquireGpu,
  type GeometryProfile,
  type GpuContext,
  type SessionDiagnostics,
} from "../../../packages/runtime/mod.ts";
import { REQUIRED_LIMIT_KEYS } from "../../../packages/runtime/src/gpu/acquire.ts";
import {
  type DistributionSource,
  type HubRepoRef,
  loadManifest,
  localDirectory,
  resolveSelection,
} from "../../../packages/hub/mod.ts";
import {
  type GeneratedVideo,
  wanFrameToRgba,
  type WanGenerateEvent,
  WanPipeline,
  type WanPipelineOptions,
  type WanPrompt,
  type WanRunComponent,
} from "../../../packages/models/wan.ts";
import {
  parseWanPipelineConfig,
  type WanPipelineConfig,
} from "../../../packages/models/src/wan/config.ts";
import { ACCEPTED_SIZES } from "../../../packages/models/src/wan/pipeline.ts";
import references from "../../../packages/models/tests/fixtures/references/wan.json" with {
  type: "json",
};
import {
  adapterSummary,
  checkoutLabel,
  describeError,
  downloadText,
  element,
  errorText,
  injectedProfile,
  type Lab,
  requestedLabel,
  setStatus,
  sha256Hex,
} from "./common.ts";
import { resolveSavedProfile } from "./injectable-tables.ts";
import {
  buildWanRequest,
  checkWanReference,
  chromeEnvironmentKey,
  formatBytes,
  formatSpans,
  formatWanDiagnostics,
  judgeWanLimits,
  parseWanSize,
  referenceVerdictText,
  summarizeWanDiagnostics,
  summarizeWanTimeline,
  type WanComponentDiagnostics,
  wanFrameChoices,
  type WanLimits,
  wanMaxFramesWithin,
  wanReferenceCaseId,
  type WanReferenceVerdict,
  type WanResolvedRequest,
  wanRgbBytes,
  wanSizeLabel,
  type WanTimeline,
  type WanTimelineMark,
} from "./wan-plan.ts";

/** 書き出す JSON の版（/2 = 読み込みと行がテキストエンコーダの経路を持つ）。 */
const WAN_REPORT_FORMAT = "karume-wan-browser/2";

/** テキストエンコーダの経路（`WanPipelineOptions.textEncoder`）。 */
type WanTextEncoder = NonNullable<WanPipelineOptions["textEncoder"]>;

/** 経路の選択肢（先頭がタブの既定 — 段 9 の確認の既定の挙動〈事前計算の埋め込み〉を変えない）。 */
const TEXT_ENCODER_CHOICES: readonly { readonly value: WanTextEncoder; readonly label: string }[] =
  [
    { value: "precomputed", label: "precomputed（埋め込み資産の 4 本 — umT5 を取らない）" },
    { value: "gpu", label: "gpu（umT5 i8 を GPU で回す — 任意のプロンプト）" },
  ];

/** Wan の manifest の部品名（umT5 — 越境参照の宣言を引く）。 */
const TEXT_ENCODER = "text_encoder";

/** 再生のフレームレート（上流の例の `export_to_video(fps=16)`）。 */
const PLAYBACK_FPS = 16;

/** 生成 1 回の記録（表の 1 行・JSON の `rows[]`）。 */
type WanRow = {
  readonly index: number;
  readonly at: string;
  readonly request: WanResolvedRequest;
  readonly textEncoder: WanTextEncoder;
  /** `gpu` の経路で渡した自由プロンプト（無ければ `request.prompt` の資産の原文を渡した）。 */
  readonly freePrompt?: string;
  /** `gpu` の経路の text 段の所要（`stage` の start → end — 完走した段だけ）。 */
  readonly textEncoderMs?: number;
  /** 参照ケースの id（sha256 の環境行のキー — 条件が e2e のケースと同じときだけ）。 */
  readonly caseId?: string;
  readonly wallMs: number;
  readonly timeline?: WanTimeline;
  readonly diagnostics: Partial<Record<WanRunComponent, WanComponentDiagnostics>>;
  readonly rgbSha256?: string;
  readonly reference?: WanReferenceVerdict;
  readonly error?: { readonly name: string; readonly message: string };
};

/** 読み込んだ配布形と、その GPU（「pipeline を破棄」まで同じ寿命）。 */
type Loaded = {
  readonly gpu: GpuContext;
  readonly pipeline: WanPipeline;
  readonly textEncoder: WanTextEncoder;
  /** 取得元の表示（`karume-wan2.1（このサーバ）` / HF の `owner/name`）。 */
  readonly source: string;
  readonly repo?: string;
  readonly revisionSha?: string;
  /** ローカルの配布形だけ（HF は解決済みの commit SHA で版を名指す）。 */
  readonly manifestSha256?: string;
  readonly config: WanPipelineConfig;
  readonly prompts: readonly WanPrompt[];
  readonly defaultNegative?: string;
  readonly environmentKey: string;
  readonly geometryProfileRequested: string;
  readonly geometryProfileInjected?: GeometryProfile;
  /** 「保存した表（照合して注入）」の結果の 1 行。 */
  readonly savedNote?: string;
};

export type WanTab = {
  /** pipeline と GPU を畳む（GPU 設定の適用の前段）。実行中なら投げる。返り値 = GPU を持っていたか。 */
  readonly reset: () => Promise<boolean>;
};

export const mountWanTab = (root: HTMLElement, lab: Lab): WanTab => {
  const ui = {
    source: element(root, "source", HTMLInputElement),
    load: element(root, "load", HTMLButtonElement),
    dispose: element(root, "dispose", HTMLButtonElement),
    exportJson: element(root, "export", HTMLButtonElement),
    prompt: element(root, "prompt", HTMLSelectElement),
    negative: element(root, "negative", HTMLSelectElement),
    seed: element(root, "seed", HTMLInputElement),
    frames: element(root, "frames", HTMLSelectElement),
    size: element(root, "size", HTMLSelectElement),
    steps: element(root, "steps", HTMLInputElement),
    guidance: element(root, "guidance", HTMLInputElement),
    shift: element(root, "shift", HTMLInputElement),
    run: element(root, "run", HTMLButtonElement),
    info: element(root, "info", HTMLElement),
    status: element(root, "status", HTMLElement),
    limitsSummary: element(root, "limits-summary", HTMLElement),
    limits: element(root, "limits", HTMLTableSectionElement),
    rows: element(root, "rows", HTMLTableSectionElement),
    canvas: element(root, "canvas", HTMLCanvasElement),
    previous: element(root, "previous", HTMLButtonElement),
    play: element(root, "play", HTMLButtonElement),
    next: element(root, "next", HTMLButtonElement),
    seek: element(root, "seek", HTMLInputElement),
    frameLabel: element(root, "frame-label", HTMLElement),
    textEncoder: document.createElement("select"),
    freePrompt: document.createElement("input"),
  };

  // 経路の選択（取得元の欄の隣）と自由プロンプトの欄（プロンプトの選択の隣）— 冒頭の doc のとおりここで足す。
  const field = (label: string, control: HTMLElement): HTMLLabelElement => {
    const wrapper = document.createElement("label");
    wrapper.className = "field";
    wrapper.append(label, control);
    return wrapper;
  };
  const after = (anchor: HTMLElement, added: HTMLElement): void => {
    const label = anchor.closest("label");
    if (label === null) throw Error("Wan のタブの欄が label の中に無い");
    label.after(added);
  };
  ui.textEncoder.replaceChildren(...TEXT_ENCODER_CHOICES.map(({ value, label }) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }));
  ui.freePrompt.type = "text";
  ui.freePrompt.size = 48;
  ui.freePrompt.placeholder = "gpu の経路だけ — 空欄 = 選んだプロンプトの原文";
  after(ui.source, field("テキストエンコーダ", ui.textEncoder));
  after(ui.prompt, field("自由プロンプト", ui.freePrompt));

  const state: {
    busy: boolean;
    loaded?: Loaded;
    /** 読み込みの記録（JSON の `loads[]`）。 */
    loads: { readonly at: string; readonly ms: number; readonly source: string }[];
    deviceLost?: { readonly reason: string; readonly message: string };
    /** 進行中の generate の診断（component → 直近の run）。 */
    diagnostics?: Map<WanRunComponent, SessionDiagnostics>;
    rows: WanRow[];
    video?: { readonly frames: readonly ImageData[] };
    frame: number;
    player?: ReturnType<typeof setInterval>;
  } = { busy: false, loads: [], rows: [], frame: 0 };

  const status = (text: string): void => setStatus(ui.status, text);

  const environmentKeyOf = (info: GPUAdapterInfo): string => {
    try {
      return chromeEnvironmentKey(info);
    } catch (error) {
      return `（作れない — ${errorText(error)}）`;
    }
  };

  const renderInfo = (): void => {
    const loaded = state.loaded;
    const info = loaded?.gpu.adapterInfo ?? lab.adapterInfo;
    ui.info.textContent = [
      adapterSummary(info),
      `環境キー ${loaded?.environmentKey ?? environmentKeyOf(info)}`,
      loaded === undefined
        ? `配布形 ${
          lab.config.wanSource ?? "無し（HF のリポ名を入れるか --wan-source で指定）"
        }（未読み込み）`
        : `配布形 ${loaded.source}`,
      `テキストエンコーダ ${loaded?.textEncoder ?? `${ui.textEncoder.value}（未読み込み）`}`,
      loaded === undefined
        ? `幾何プロファイル ${requestedLabel(lab.settings().choice)}（読み込み時に確定）`
        : `幾何プロファイルの要求 ${loaded.geometryProfileRequested}${
          loaded.savedNote === undefined ? "" : `（${loaded.savedNote}）`
        }`,
      "GPU 時間は採らない（WanPipeline は計測の device を拒む）",
      checkoutLabel(lab.config),
    ].join(" · ");
  };

  /** 判定表（device を取っていれば device の値で判定し、アダプタの値と並べる）。 */
  const renderLimits = (): void => {
    let frames: number;
    let size: { width: number; height: number };
    try {
      frames = Number(ui.frames.value);
      size = parseWanSize(ui.size.value);
    } catch (error) {
      ui.limitsSummary.textContent = errorText(error);
      return;
    }
    const adapterLimits: WanLimits = lab.adapterLimits;
    const deviceLimits: WanLimits | undefined = state.loaded?.gpu.limits;
    const judged = judgeWanLimits(deviceLimits ?? adapterLimits, frames, size);
    ui.limits.replaceChildren(...judged.map((row) => {
      const tr = document.createElement("tr");
      const cells = [
        row.key,
        adapterLimits[row.key].toLocaleString("en-US"),
        deviceLimits === undefined ? "—" : deviceLimits[row.key].toLocaleString("en-US"),
        row.required === undefined ? "—" : formatBytes(row.required),
        row.verdict === "ok" ? "足りる" : row.verdict === "short" ? "足りない" : "—",
        row.note,
      ];
      for (const [at, value] of cells.entries()) {
        const td = document.createElement("td");
        td.textContent = value;
        if (at >= 1 && at <= 2) td.className = "num";
        if (at === 4 && row.verdict === "short") td.className = "bad";
        if (at === 5) td.className = "wrap";
        tr.append(td);
      }
      return tr;
    }));
    const short = judged.filter((row) => row.verdict === "short").map((row) => row.key);
    const max = wanMaxFramesWithin(deviceLimits ?? adapterLimits, size);
    ui.limitsSummary.textContent = `${frames} フレーム · ${wanSizeLabel(size)} を ${
      deviceLimits === undefined ? "アダプタ" : "取得した device"
    } の limits で判定: ${
      short.length === 0 ? "束縛上限とバッファ上限は足りる" : `足りない — ${short.join(" / ")}`
    } · この limits で回せる最大は ${
      max === undefined ? "無し" : `${max} フレーム`
    }（必要条件だけ — VRAM・submit の時間は別）`;
    ui.limitsSummary.className = short.length === 0 ? "" : "bad";
  };

  const setBusy = (busy: boolean): void => {
    state.busy = busy;
    const loaded = state.loaded !== undefined;
    ui.load.disabled = busy || loaded;
    ui.source.disabled = busy || loaded;
    ui.textEncoder.disabled = busy || loaded;
    ui.dispose.disabled = busy || !loaded;
    ui.run.disabled = busy || !loaded;
    for (
      const input of [
        ui.prompt,
        ui.negative,
        ui.seed,
        ui.frames,
        ui.size,
        ui.steps,
        ui.guidance,
        ui.shift,
      ]
    ) input.disabled = busy;
    ui.prompt.disabled ||= !loaded;
    ui.negative.disabled ||= !loaded;
    ui.freePrompt.disabled = busy || state.loaded?.textEncoder !== "gpu";
    ui.exportJson.disabled = busy || (state.rows.length === 0 && state.loads.length === 0);
  };

  /** ボタン操作の排他（ページの他の GPU 操作とも重ねない）。失敗は状態行へ。 */
  const exclusive = (label: string, action: () => Promise<void>) => async (): Promise<void> => {
    if (state.busy) return;
    let release: () => void;
    try {
      release = lab.lock(label);
    } catch (error) {
      status(errorText(error));
      return;
    }
    setBusy(true);
    try {
      await action();
    } catch (error) {
      status(errorText(error));
    } finally {
      setBusy(false);
      release();
    }
  };

  /**
   * このサーバの配布形（相対 path — 同じページと bundle を静的な置き場に載せても読める）。`route` は配る経路の
   * 名前（`wan` / `umt5` — `server.ts` の `/models/<名前>/`）、`crossRepo` は越境先の mapping。
   */
  const serverSource = (
    route: "wan" | "umt5",
    crossRepo: Readonly<Record<string, DistributionSource>> = {},
  ): DistributionSource =>
    localDirectory({
      readFile: async (path, options) => {
        const response = await fetch(`models/${route}/${path}`, options);
        if (!response.ok) throw Error(`Model HTTP ${response.status}: ${route}/${path}`);
        return new Uint8Array(await response.arrayBuffer());
      },
      readFileRange: async (path, offset, length, options) => {
        const response = await fetch(`models/${route}/${path}`, {
          ...options,
          headers: { Range: `bytes=${offset}-${offset + length - 1}` },
        });
        if (response.status !== 206) {
          throw Error(`Model range HTTP ${response.status}: ${route}/${path}`);
        }
        return new Uint8Array(await response.arrayBuffer());
      },
    }, { label: `browser-${route}`, crossRepo });

  const fillPrompts = (prompts: readonly WanPrompt[]): void => {
    const option = (value: string, text: string): HTMLOptionElement => {
      const created = document.createElement("option");
      created.value = value;
      created.textContent = text;
      return created;
    };
    ui.prompt.replaceChildren(
      ...prompts.filter((p) => p.role === "positive").map((p) => option(p.name, p.name)),
    );
    ui.negative.replaceChildren(
      option("", "既定（資産の negative の行）"),
      ...prompts.map((p) => option(p.name, `${p.name}（${p.role}）`)),
    );
    const first = ui.prompt.options.item(0);
    if (first !== null) ui.prompt.title = prompts.find((p) => p.name === first.value)?.prompt ?? "";
  };

  /** 取得元 → `fromPretrained` の第 1 引数と表示。 */
  const readSource = (): {
    ref: DistributionSource | HubRepoRef;
    label: string;
    local: boolean;
  } => {
    const text = ui.source.value.trim();
    if (text !== "") return { ref: { repo: text }, label: text, local: false };
    if (lab.config.wanSource === null) {
      throw Error(
        "このサーバは Wan の配布形を配っていない（--wan-source で指定するか、HF のリポ名を入れる）",
      );
    }
    return {
      ref: serverSource("wan"),
      label: `${lab.config.wanSource}（このサーバ）`,
      local: true,
    };
  };

  /** 経路の選択（未知の値は選択肢の取り違え — 素の Error）。 */
  const readTextEncoder = (): WanTextEncoder => {
    const found = TEXT_ENCODER_CHOICES.find(({ value }) => value === ui.textEncoder.value);
    if (found === undefined) {
      throw Error(`テキストエンコーダ '${ui.textEncoder.value}' は選択肢に無い`);
    }
    return found.value;
  };

  /**
   * `gpu` の経路でこのサーバの配布形を読む取得元。umT5 が越境参照なら、その repo（manifest の宣言から引く —
   * 名前を写経しない）を `/models/umt5/` の取得元へ結ぶ。umT5 を配っていなければ取得の前に名指しで落とす。
   */
  const gpuServerSource = async (
    manifest: Parameters<typeof resolveSelection>[0],
  ): Promise<DistributionSource> => {
    const repo = resolveSelection(manifest).containers[TEXT_ENCODER]?.parts[0]?.repo;
    if (repo === undefined) return serverSource("wan");
    const probe = await fetch("models/umt5/karume.json", { method: "HEAD" });
    if (!probe.ok) {
      throw Error(
        `このサーバは umT5 の配布形（${repo} の越境先）を配っていない（HTTP ${probe.status}）— ` +
          "--umt5-source で指定して起動し直すか、テキストエンコーダを precomputed にする",
      );
    }
    return serverSource("wan", { [repo]: serverSource("umt5") });
  };

  const load = async (): Promise<void> => {
    const textEncoder = readTextEncoder();
    const { ref, label, local } = readSource();
    const started = performance.now();
    status(`manifest を読み込み中（${label}）`);
    const manifest = await loadManifest(ref);
    const model = manifest.manifest.models[manifest.manifest.defaultModel];
    if (model === undefined) {
      throw Error(`defaultModel ${manifest.manifest.defaultModel} が models に無い`);
    }
    const config = parseWanPipelineConfig(model.pipelineConfig);
    let manifestSha256: string | undefined;
    if (local) {
      const response = await fetch("models/wan/karume.json");
      if (!response.ok) throw Error(`karume.json HTTP ${response.status}`);
      manifestSha256 = await sha256Hex(new Uint8Array(await response.arrayBuffer()));
    }

    // GPU 設定（ヘッダの「適用」）の幾何プロファイル。照合して注入は GPU を取るときのコールバック（ADR 0117 決定 6）
    const choice = lab.settings().choice;
    let savedNote: string | undefined;
    let injected = injectedProfile(choice);
    const geometryProfile = choice.kind === "saved-matched"
      ? (adapterInfo: GPUAdapterInfo): GeometryProfile | undefined => {
        const resolution = resolveSavedProfile(choice.stored, adapterInfo);
        if (resolution.kind === "matched") {
          savedNote = `保存した表 ${resolution.id} は adapter と一致 → 注入`;
          injected = resolution.profile;
          return resolution.profile;
        }
        savedNote = `保存した表を注入しない（自動で選ぶ）— ${resolution.reason}`;
        return undefined;
      }
      : injected;
    // `gpu` でこのサーバの配布形を読むなら越境先を結んだ取得元へ差し替える（HF の取得元は宣言どおり取る）。
    const source = textEncoder === "gpu" && local ? await gpuServerSource(manifest.manifest) : ref;
    status("GPU を取得中");
    const gpu = await acquireGpu({
      ...(geometryProfile === undefined ? {} : { geometryProfile }),
      onDeviceLost: (info) => {
        state.deviceLost = { reason: info.reason, message: info.message };
        status(
          `GPU device lost（${info.reason}）: ${info.message} — 「pipeline を破棄」でやり直せます`,
        );
      },
    });
    let pipeline: WanPipeline;
    try {
      pipeline = await WanPipeline.fromPretrained(source, {
        gpu,
        // MUST: 経路は必ず明示する（パイプラインの既定は "gpu" — タブの既定の precomputed と食い違う）。
        textEncoder,
        onRunDiagnostics: (component, diagnostics) => {
          if (state.diagnostics === undefined) {
            throw Error(`${component} の run が generate の外で終わった`);
          }
          state.diagnostics.set(component, diagnostics);
        },
        onProgress: (progress) => {
          status(
            `取得中 ${(progress.loaded / 2 ** 20).toFixed(0)} / ${
              (progress.total / 2 ** 20).toFixed(0)
            } MiB（${progress.path}）`,
          );
        },
      });
    } catch (error) {
      gpu.destroy();
      throw error;
    }
    const prompts = pipeline.prompts;
    const negatives = prompts.filter((p) => p.role === "negative");
    state.loaded = {
      gpu,
      pipeline,
      textEncoder,
      source: label,
      ...(manifest.repo === undefined ? {} : { repo: manifest.repo }),
      ...(manifest.revisionSha === undefined ? {} : { revisionSha: manifest.revisionSha }),
      ...(manifestSha256 === undefined ? {} : { manifestSha256 }),
      config,
      prompts,
      ...(negatives.length === 1 ? { defaultNegative: negatives[0].name } : {}),
      environmentKey: environmentKeyOf(gpu.adapterInfo),
      geometryProfileRequested: requestedLabel(choice),
      ...(injected === undefined ? {} : { geometryProfileInjected: injected }),
      ...(savedNote === undefined ? {} : { savedNote }),
    };
    const ms = performance.now() - started;
    state.loads.push({ at: new Date().toISOString(), ms, source: label });
    fillPrompts(prompts);
    ui.steps.placeholder = `配布既定 ${config.defaults.steps}`;
    ui.guidance.placeholder = `配布既定 ${config.defaults.guidance}`;
    ui.shift.placeholder = `配布既定 ${config.scheduler.shift}`;
    renderInfo();
    renderLimits();
    status(
      `読み込み済み（${(ms / 1000).toFixed(1)} s）${
        savedNote === undefined ? "" : ` — ${savedNote}`
      }`,
    );
  };

  const stopPlayback = (): void => {
    if (state.player !== undefined) clearInterval(state.player);
    state.player = undefined;
    ui.play.textContent = "再生";
  };

  const showFrame = (frame: number): void => {
    const video = state.video;
    if (video === undefined || video.frames.length === 0) return;
    state.frame = ((frame % video.frames.length) + video.frames.length) % video.frames.length;
    const image = video.frames[state.frame];
    const context = ui.canvas.getContext("2d");
    if (context === null) throw Error("canvas の 2d context を取れない");
    context.putImageData(image, 0, 0);
    ui.seek.value = String(state.frame);
    ui.frameLabel.textContent = `${state.frame + 1} / ${video.frames.length}`;
  };

  const showVideo = (
    video: GeneratedVideo,
    frames: readonly Uint8ClampedArray<ArrayBuffer>[],
  ): void => {
    stopPlayback();
    state.video = { frames: frames.map((rgba) => new ImageData(rgba, video.width, video.height)) };
    ui.canvas.width = video.width;
    ui.canvas.height = video.height;
    ui.canvas.hidden = false;
    ui.seek.max = String(video.frames - 1);
    for (const control of [ui.previous, ui.play, ui.next, ui.seek]) control.disabled = false;
    showFrame(0);
  };

  const formatRequest = (row: WanRow): string =>
    `${row.textEncoder} · ${
      row.freePrompt === undefined ? row.request.prompt : JSON.stringify(row.freePrompt)
    } · ${formatKnobs(row.request)}`;

  const formatKnobs = (request: WanResolvedRequest): string =>
    `seed ${request.seed} · ${request.frames} フレーム · ${request.width}x${request.height} · ${request.steps} step · guidance ${request.guidance} · shift ${request.shift}${
      request.negative === undefined ? "" : ` · negative ${request.negative}`
    }`;

  const appendRow = (row: WanRow): void => {
    const tr = document.createElement("tr");
    const timeline = row.timeline;
    const cells = [
      String(row.index),
      formatRequest(row),
      `${(row.wallMs / 1000).toFixed(1)} s`,
      timeline === undefined ? "—" : [
        ...(row.textEncoderMs === undefined
          ? []
          : [`text_encoder ${(row.textEncoderMs / 1000).toFixed(1)} s`]),
        ...Object.entries(timeline.stageMs).map(([stage, ms]) =>
          `${stage} ${(ms / 1000).toFixed(1)} s`
        ),
      ].join("\n"),
      timeline === undefined ? "—" : formatSpans(timeline.stepMs),
      timeline === undefined ? "—" : formatSpans(timeline.tileMs),
      Object.entries(row.diagnostics).map(([component, summary]) =>
        formatWanDiagnostics(component, summary)
      ).join("\n"),
      row.rgbSha256 ?? "—",
      row.reference === undefined ? "—" : referenceVerdictText(row.reference),
      row.error === undefined ? "" : `${row.error.name}: ${row.error.message}`,
    ];
    for (const [at, value] of cells.entries()) {
      const td = document.createElement("td");
      td.textContent = value;
      if (at >= 1) td.className = "wrap";
      if (at === 8 && row.reference?.kind === "mismatch") td.className = "wrap bad";
      if (at === 9 && value !== "") td.className = "wrap error";
      tr.append(td);
    }
    ui.rows.append(tr);
  };

  const generate = async (): Promise<void> => {
    const loaded = state.loaded;
    if (loaded === undefined) throw Error("先に「読み込む」");
    // 条件は押下時に 1 度だけ読む（実行中に変えた入力は効かない — setBusy が入力を止める）
    const { request, resolved } = buildWanRequest(
      {
        prompt: ui.prompt.value,
        negative: ui.negative.value,
        seed: ui.seed.value,
        frames: ui.frames.value,
        size: ui.size.value,
        steps: ui.steps.value,
        guidance: ui.guidance.value,
        shift: ui.shift.value,
      },
      loaded.prompts,
      loaded.config,
    );
    const freeText = ui.freePrompt.value;
    const freePrompt = loaded.textEncoder === "gpu" && freeText.trim() !== ""
      ? freeText
      : undefined;
    // 参照ケースの照合は事前計算の経路だけ（`wanReferenceCaseId` の id は事前計算の経路の sha 行 — GPU 経路の
    // 動画は同じ条件でも値が違う）。
    const caseId = loaded.textEncoder === "precomputed"
      ? wanReferenceCaseId(resolved, loaded.config, loaded.defaultNegative)
      : undefined;
    const index = state.rows.length + 1;
    const marks: WanTimelineMark[] = [];
    /** text 段の start / end（`WanTimelineMark` は DiT と VAE の段だけを持つので別に採る）。 */
    const textStage: { start?: number; end?: number } = {};
    const diagnostics = new Map<WanRunComponent, SessionDiagnostics>();
    state.diagnostics = diagnostics;
    const started = performance.now();
    const onEvent = (event: WanGenerateEvent): void => {
      const ms = performance.now();
      const elapsed = `${((ms - started) / 1000).toFixed(0)} s`;
      if (event.kind === "stage") {
        const { component } = event;
        if (component === "text_encoder") textStage[event.at] = ms;
        else marks.push({ kind: "stage", component, at: event.at, ms });
        status(`${component} ${event.at}（${elapsed}）`);
      } else if (event.kind === "denoise-step") {
        marks.push({ kind: "step", step: event.step, ms });
        status(`transformer step ${event.step}/${event.steps}（${elapsed}）`);
      } else {
        marks.push({ kind: "tile", tile: event.tile, ms });
        status(`vae tile ${event.tile}/${event.tiles}（${elapsed}）`);
      }
    };
    const summaries = (): WanRow["diagnostics"] =>
      Object.fromEntries(
        [...diagnostics].map(([component, value]) => [component, summarizeWanDiagnostics(value)]),
      );
    const textEncoderMs = (): { textEncoderMs?: number } =>
      textStage.start === undefined || textStage.end === undefined
        ? {}
        : { textEncoderMs: textStage.end - textStage.start };
    const base = {
      index,
      at: new Date().toISOString(),
      request: resolved,
      textEncoder: loaded.textEncoder,
      ...(freePrompt === undefined ? {} : { freePrompt }),
      ...(caseId === undefined ? {} : { caseId }),
    };
    let row: WanRow;
    try {
      const video = await loaded.pipeline.generate({
        ...request,
        ...(freePrompt === undefined ? {} : { prompt: freePrompt }),
        onEvent,
      });
      const wallMs = performance.now() - started;
      status("フレームを画素にして sha256 を計算中");
      const frames = Array.from(
        { length: video.frames },
        (_, frame) => wanFrameToRgba(video, frame),
      );
      const rgbSha256 = await sha256Hex(wanRgbBytes(frames, video.width * video.height));
      const reference = checkWanReference(references, caseId, loaded.environmentKey, rgbSha256);
      // 完走した生成のイベントの並びが崩れていたらパイプラインの取り決めの破れ — 失敗の行にする
      const timeline = summarizeWanTimeline(marks);
      showVideo(video, frames);
      row = {
        ...base,
        ...textEncoderMs(),
        wallMs,
        timeline,
        diagnostics: summaries(),
        rgbSha256,
        reference,
      };
      status(
        `完了 — ${(wallMs / 1000).toFixed(1)} s · sha ${rgbSha256.slice(0, 12)} · ${
          referenceVerdictText(reference)
        }`,
      );
    } catch (error) {
      let timeline: WanTimeline | undefined;
      try {
        timeline = summarizeWanTimeline(marks);
      } catch {
        // 失敗した生成の行では、並びの崩れで生成そのものの失敗（この行の error）を上書きしない —
        // 所要の内訳なしで残す
      }
      row = {
        ...base,
        ...textEncoderMs(),
        wallMs: performance.now() - started,
        ...(timeline === undefined ? {} : { timeline }),
        diagnostics: summaries(),
        error: describeError(error),
      };
      status(`生成が失敗しました（表の行を参照）— ${errorText(error)}`);
    } finally {
      state.diagnostics = undefined;
    }
    state.rows.push(row);
    appendRow(row);
  };

  const dispose = async (): Promise<void> => {
    const loaded = state.loaded;
    state.loaded = undefined;
    state.deviceLost = undefined;
    let failure: string | undefined;
    try {
      await loaded?.pipeline.dispose();
    } catch (error) {
      failure = errorText(error);
    }
    loaded?.gpu.destroy();
    renderInfo();
    renderLimits();
    status(
      `pipeline と GPU device を破棄しました${
        failure === undefined ? "" : `（dispose の失敗: ${failure}）`
      }。「読み込む」で組み直します。`,
    );
  };

  const exportJson = (): void => {
    const loaded = state.loaded;
    const info = loaded?.gpu.adapterInfo ?? lab.adapterInfo;
    const pick = (limits: WanLimits): Record<string, number> =>
      Object.fromEntries(REQUIRED_LIMIT_KEYS.map((key) => [key, limits[key]]));
    const report = {
      format: WAN_REPORT_FORMAT,
      date: new Date().toISOString(),
      userAgent: navigator.userAgent,
      adapter: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      },
      environmentKey: loaded?.environmentKey ?? environmentKeyOf(info),
      adapterLimits: pick(lab.adapterLimits),
      ...(loaded === undefined ? {} : { deviceLimits: pick(loaded.gpu.limits) }),
      checkout: lab.config.revision,
      checkoutDirty: lab.config.dirty,
      bundleSha256: lab.config.bundleSha256,
      ...(loaded === undefined ? {} : {
        source: loaded.source,
        textEncoder: loaded.textEncoder,
        ...(loaded.repo === undefined ? {} : { repo: loaded.repo }),
        ...(loaded.revisionSha === undefined ? {} : { revisionSha: loaded.revisionSha }),
        ...(loaded.manifestSha256 === undefined ? {} : { manifestSha256: loaded.manifestSha256 }),
        geometryProfileRequested: loaded.geometryProfileRequested,
        ...(loaded.geometryProfileInjected === undefined
          ? {}
          : { geometryProfileInjected: loaded.geometryProfileInjected }),
      }),
      loads: state.loads,
      deviceLost: state.deviceLost ?? null,
      rows: state.rows,
    };
    downloadText(
      JSON.stringify(report, null, 2),
      `wan-browser-${new Date().toISOString().replaceAll(":", "-")}.json`,
      "application/json",
    );
  };

  // 選択肢（受理集合は pipeline.ts が正本 — フレーム数は 4n+1 の 5〜81・寸法は 2 通り）
  ui.frames.replaceChildren(
    ...wanFrameChoices().map((frames) => {
      const option = document.createElement("option");
      option.value = String(frames);
      option.textContent = String(frames);
      option.selected = frames === 33;
      return option;
    }),
  );
  ui.size.replaceChildren(...ACCEPTED_SIZES.map((size) => {
    const option = document.createElement("option");
    option.value = wanSizeLabel(size);
    option.textContent = wanSizeLabel(size);
    return option;
  }));
  ui.source.placeholder = lab.config.wanSource === null
    ? "owner/name（このサーバは Wan の配布形を配っていない）"
    : `空欄 = このサーバの ${lab.config.wanSource} · HF なら owner/name`;
  for (const control of [ui.previous, ui.play, ui.next, ui.seek]) control.disabled = true;

  ui.load.addEventListener("click", exclusive("Wan の読み込み", load));
  ui.run.addEventListener("click", exclusive("Wan の生成", generate));
  ui.dispose.addEventListener("click", exclusive("Wan の破棄", dispose));
  ui.exportJson.addEventListener("click", exportJson);
  ui.textEncoder.addEventListener("change", renderInfo);
  ui.frames.addEventListener("change", renderLimits);
  ui.size.addEventListener("change", renderLimits);
  ui.prompt.addEventListener("change", () => {
    ui.prompt.title = state.loaded?.prompts.find((p) => p.name === ui.prompt.value)?.prompt ?? "";
  });
  ui.previous.addEventListener("click", () => {
    stopPlayback();
    showFrame(state.frame - 1);
  });
  ui.next.addEventListener("click", () => {
    stopPlayback();
    showFrame(state.frame + 1);
  });
  ui.seek.addEventListener("input", () => {
    stopPlayback();
    showFrame(Number(ui.seek.value));
  });
  ui.play.addEventListener("click", () => {
    if (state.player !== undefined) {
      stopPlayback();
      return;
    }
    state.player = setInterval(() => showFrame(state.frame + 1), 1000 / PLAYBACK_FPS);
    ui.play.textContent = "停止";
  });

  renderInfo();
  renderLimits();
  setBusy(false);
  status(
    "「読み込む」で GPU を取り、配布形を読みます（テキストエンコーダの経路は読み込み時に決まる）。" +
      "判定表は選んだフレーム数・寸法で更新されます。",
  );

  return {
    reset: async () => {
      if (state.busy) throw Error("Wan のタブが実行中 — 終わってから適用する");
      const held = state.loaded !== undefined;
      if (held) await dispose();
      return held;
    },
  };
};
