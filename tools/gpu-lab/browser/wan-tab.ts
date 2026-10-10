/// <reference lib="dom" />
/**
 * Wan のタブ — Wan2.1 T2V 1.3B（ADR 0118 段 9）と Wan2.2 TI2V 5B の T2V と I2V（ADR 0121 段 8・9）を Chrome で回す確認ページ。
 * 取得元の欄の隣の**世代の選択**で切り替える（読み込み前だけ — 選択でフレーム数・寸法の選択肢・判定表・quant の
 * 選択肢・取得元の placeholder を作り直す。世代で違う値は `wan-plan.ts` の世代の仕様）。
 *
 * 確かめたいのは「Chrome の device で完走するか」と、完走しないならどこで止まるか（束縛上限・TDR〈GPU の
 * タイムアウト検出〉・VRAM）。そのために:
 *
 * 1. **事前判定**: アダプタ（と取得した device）の limits を、選んだフレーム数・寸法が要る値と並べる
 *    （`wan-plan.ts` の `judgeWanLimits` — 足りない項目は赤）。
 * 2. **読み込み**: GPU を取り（ページの GPU 設定の幾何プロファイル・`onDeviceLost`）、配布形を読む。Wan2.1 は
 *    `WanPipeline.fromPretrained`、Wan2.2 は `WanTi2vPipeline.fromPretrained`。2 つの口は同じ形（{@link WanRunner}）に
 *    揃え、生成・記録・照合の本体は 1 本。取得元は既定でこのサーバが配る `models/karume-wan2.1`（`/models/wan/…`）/
 *    `models/karume-wan2.2`（`/models/wan22/…`）で（HF の公開リポはまだ無い）、HF の `owner/name` も入れられる。テキストエンコーダの
 *    経路（ADR 0119 決定 7）はタブの既定が `precomputed`（埋め込み資産 — 段 9 の確認の既定のまま。パイプライン
 *    の既定は `gpu` なので、ここでは必ず明示して渡す）。`gpu` を選ぶと umT5 の越境先（Wan の manifest の
 *    `text_encoder` が宣言する repo）を、このサーバが配る `models/karume-umt5-xxl`（`/models/umt5/…`）へ取得元の
 *    `crossRepo` で結ぶ（HF の取得元なら宣言どおり HF から取る）。quant の席はタブの選択肢（このサーバの配布形の
 *    manifest から埋める）か、既定なら読み込み時に manifest の `defaultQuant` へ解決し、解決した名前を必ず明示して
 *    渡す（参照ケースの id は回した席で決まる — 綴りは `wan-plan.ts` の世代の参照ケースの表）。
 * 3. **生成**: プロンプト（埋め込み資産の名前 — `gpu` なら自由プロンプトの欄の文字列が優先）・seed・フレーム数・
 *    寸法・steps・guidance・shift で `generate` → 全フレームを canvas に描いて生成した動画の fps で再生 → 所要（段・step・VAE タイル・
 *    `gpu` なら text 段）・Session の診断・RGB の sha256（事前計算の経路の参照ケースなら環境行との照合）を表に積む。
 * 4. **I2V の条件画像**（Wan2.2 だけ — 世代の記述子の DiT の入力の形が `"ti2v"` のとき）: 寸法の選択の隣で PNG / JPEG を
 *    選ぶと、ブラウザの標準 API（`createImageBitmap` → OffscreenCanvas の `getImageData`）で RGB8 にして（アルファは捨てる）
 *    要求の `image` と `fit`（crop / stretch）に渡す。画像を選ぶと寸法の選択に「画像から自動」（`width` / `height` を
 *    渡さない — 製品の `selectWanI2vSize` と同じ寸法を判定表と記録に使う）が足されて既定になる。I2V の行は参照ケースと
 *    照合しない（e2e の I2V の sha 行は特定の入力画像の画素に結びつく — このタブの復号の画素とは突き合わせない）。
 *
 * 世代と経路の選択・自由プロンプトの欄・条件画像の操作はページの HTML ではなくここで足す（`index.html` の Wan の節は
 * 事前計算の経路の欄だけを持つ）。
 *
 * GPU は自前で取って pipeline に渡す（共有 GPU）: device lost を `onDeviceLost` で表に残し、幾何プロファイルの
 * 注入をページの GPU 設定に揃えるため。`acquireGpu` は requiredLimits にアダプタ値を要求する（Chrome の既定
 * 128 MiB の束縛上限のままにしない）。計測（`gpuTiming`）は要求しない — Wan の家族 admission は 2 世代とも計測の
 * device を構築時に拒む（VAE の段は 1 タイル = 1 batch で、runtime は計測の device で batch を開かない）。
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
  parseManifest,
  resolveSelection,
} from "../../../packages/hub/mod.ts";
import {
  type GeneratedVideo,
  type Rgb8Image,
  wanFrameToRgba,
  type WanFromPretrainedOptions,
  type WanGenerateEvent,
  type WanI2vFit,
  WanPipeline,
  type WanPipelineOptions,
  type WanPrompt,
  type WanRunComponent,
  type WanTi2vGenerateRequest,
  WanTi2vPipeline,
} from "../../../packages/models/wan.ts";
import {
  parseWanPipelineConfig,
  type WanPipelineConfig,
} from "../../../packages/models/src/wan/config.ts";
import { selectWanI2vSize } from "../../../packages/models/src/wan/i2v-preprocess.ts";
import wan21References from "../../../packages/models/tests/fixtures/references/wan.json" with {
  type: "json",
};
import wan22References from "../../../packages/models/tests/fixtures/references/wan-ti2v.json" with {
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
  referenceVerdictText,
  resolveWanFormSize,
  summarizeWanDiagnostics,
  summarizeWanTimeline,
  WAN_LAB_GENERATIONS,
  WAN_SIZE_FROM_IMAGE,
  type WanComponentDiagnostics,
  wanFrameChoices,
  type WanLabGeneration,
  type WanLabGenerationId,
  type WanLimits,
  wanMaxFramesWithin,
  wanReferenceCaseId,
  type WanReferences,
  type WanReferenceVerdict,
  type WanResolvedRequest,
  wanRgbBytes,
  type WanSize,
  wanSizeLabel,
  type WanTimeline,
  type WanTimelineMark,
} from "./wan-plan.ts";

/**
 * 書き出す JSON の版（/2 = 読み込みと行がテキストエンコーダの経路を持つ・/3 = 読み込みと行が quant の席を持つ・
 * /4 = 読み込み〈`loads[]` と読み込み中の配布形の欄〉と行が世代 `generation`〈`wan2.1` / `wan2.2`〉を持つ・
 * /5 = I2V の行が条件画像 `image` を持ち、段の所要 `timeline.stageMs` に `vae_encoder` が入りうる）。
 */
const WAN_REPORT_FORMAT = "karume-wan-browser/5";

/** テキストエンコーダの経路（`WanPipelineOptions.textEncoder`）。 */
type WanTextEncoder = NonNullable<WanPipelineOptions["textEncoder"]>;

/** 経路の選択肢（先頭がタブの既定 — 段 9 の確認の既定の挙動〈事前計算の埋め込み〉を変えない）。 */
const TEXT_ENCODER_CHOICES: readonly { readonly value: WanTextEncoder; readonly label: string }[] =
  [
    { value: "precomputed", label: "precomputed（埋め込み資産の 4 本 — umT5 を取らない）" },
    { value: "gpu", label: "gpu（umT5 i8 を GPU で回す — 任意のプロンプト）" },
  ];

/** I2V の条件画像の寸法の合わせ方の選択肢（先頭がタブの既定 — パイプラインの既定と同じ crop）。 */
const FIT_CHOICES: readonly { readonly value: WanI2vFit; readonly label: string }[] = [
  { value: "crop", label: "crop（公式 Wan2.2 — 縦横比を保って覆い、中央を切り出す）" },
  { value: "stretch", label: "stretch（diffusers — 出力寸法へ直接伸縮・縦横比は保たない）" },
];

/** Wan の manifest の部品名（umT5 — 越境参照の宣言を引く）。 */
const TEXT_ENCODER = "text_encoder";

/** 世代ごとの sha256 の環境行の表（`fixtures/references/` — ADR 0106）。 */
const REFERENCES: Readonly<Record<WanLabGenerationId, WanReferences>> = {
  "wan2.1": wan21References,
  "wan2.2": wan22References,
};

/** 読み込んだ世代の口（2 世代で同じ形 — 生成・記録・照合の本体は 1 本）。 */
type WanRunner = {
  /** 埋め込み資産のプロンプトの一覧（class の `prompts` と同じ写し）。 */
  readonly prompts: readonly WanPrompt[];
  /** 要求は Wan2.2 の形（`image` / `fit` は Wan2.2 のときだけ渡す — Wan2.1 のパイプラインは `ModelInputError` で拒む）。 */
  readonly generate: (request: WanTi2vGenerateRequest) => Promise<GeneratedVideo>;
  /** 口が持つ資源の解放（GPU はタブの所有物 — {@link Loaded.gpu} を呼び手が破棄する）。 */
  readonly dispose: () => Promise<void>;
};

/** 公開の class 1 本を口の形にする（GPU は `options.gpu` で渡すので、`dispose` は GPU を破棄しない）。 */
const runnerOf = (pipeline: WanPipeline | WanTi2vPipeline): WanRunner => ({
  prompts: pipeline.prompts,
  generate: (request) => pipeline.generate(request),
  dispose: () => pipeline.dispose(),
});

/** 世代の口を組む（Wan2.1 は `WanPipeline`・Wan2.2 は `WanTi2vPipeline` の `fromPretrained`）。 */
const openRunner = async (
  generation: WanLabGeneration,
  source: DistributionSource | HubRepoRef,
  options: WanFromPretrainedOptions,
): Promise<WanRunner> => {
  switch (generation.id) {
    case "wan2.1":
      return runnerOf(await WanPipeline.fromPretrained(source, options));
    case "wan2.2":
      return runnerOf(await WanTi2vPipeline.fromPretrained(source, options));
  }
};

/** `<option>` を 1 つ作る。 */
const optionOf = (value: string, text: string, selected = false): HTMLOptionElement => {
  const created = document.createElement("option");
  created.value = value;
  created.textContent = text;
  created.selected = selected;
  return created;
};

/** 行に載せる I2V の条件画像の記録。 */
type WanRowImage = {
  /** 選んだファイルの名前（`File.name`）。 */
  readonly file: string;
  /** 復号した画像の寸法（EXIF の向きを適用した後 — 出力寸法へ合わせる前）。 */
  readonly width: number;
  readonly height: number;
  readonly fit: WanI2vFit;
  /** 寸法が「画像から自動」だったか（`request.width` / `height` はパイプラインが画像から選んだ寸法）。 */
  readonly sizeFromImage: boolean;
};

/** 選んだ条件画像（復号済み）。 */
type ChosenImage = {
  readonly file: string;
  readonly rgb: Rgb8Image;
  /** 「画像から自動」で使う寸法（製品の `selectWanI2vSize` — パイプラインが選ぶのと同じ値）。 */
  readonly size: WanSize;
};

/**
 * 画像ファイル → RGB8（ブラウザの標準 API — 依存パッケージを持ち込まない）。色空間の変換と事前乗算はしない
 * （画素の値をそのまま使う）。アルファは捨てる。EXIF の向きは適用する（`imageOrientation: "from-image"` を明示 —
 * 既定値はかつての仕様と版で揺れたので頼らない）。
 */
const decodeImage = async (file: File): Promise<Rgb8Image> => {
  const bitmap = await createImageBitmap(file, {
    imageOrientation: "from-image",
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d");
    if (context === null) throw Error("OffscreenCanvas の 2d context を取れない");
    context.drawImage(bitmap, 0, 0);
    const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
    return {
      data: wanRgbBytes([data], bitmap.width * bitmap.height),
      width: bitmap.width,
      height: bitmap.height,
    };
  } finally {
    bitmap.close();
  }
};

/** 生成 1 回の記録（表の 1 行・JSON の `rows[]`）。 */
type WanRow = {
  readonly index: number;
  readonly at: string;
  /** 回した世代（読み込んだ世代 — 照合した sha 行の表もこれで決まる）。 */
  readonly generation: WanLabGenerationId;
  readonly request: WanResolvedRequest;
  /** 回した quant の席（manifest の既定へ解決した後の名前）。 */
  readonly quant: string;
  readonly textEncoder: WanTextEncoder;
  /** `gpu` の経路で渡した自由プロンプト（無ければ `request.prompt` の資産の原文を渡した）。 */
  readonly freePrompt?: string;
  /** I2V の条件画像（渡した要求だけ — 画素は記録しない）。 */
  readonly image?: WanRowImage;
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
  readonly generation: WanLabGeneration;
  readonly runner: WanRunner;
  /** 組んだ quant の席（「既定」を選んだときは manifest の `defaultQuant` へ解決した名前）。 */
  readonly quant: string;
  readonly textEncoder: WanTextEncoder;
  /** 取得元の表示（`karume-wan2.1（このサーバ）` / `karume-wan2.2（このサーバ）` / HF の `owner/name`）。 */
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
    generation: document.createElement("select"),
    textEncoder: document.createElement("select"),
    quant: document.createElement("select"),
    freePrompt: document.createElement("input"),
    image: document.createElement("input"),
    fit: document.createElement("select"),
    clearImage: document.createElement("button"),
  };

  // 世代と経路の選択（取得元の欄の隣）と自由プロンプトの欄（プロンプトの選択の隣）— 冒頭の doc のとおりここで足す。
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
  ui.generation.replaceChildren(
    ...WAN_LAB_GENERATIONS.map(({ id, label }) => optionOf(id, label)),
  );
  ui.textEncoder.replaceChildren(...TEXT_ENCODER_CHOICES.map(({ value, label }) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }));
  ui.freePrompt.type = "text";
  ui.freePrompt.size = 48;
  ui.freePrompt.placeholder = "gpu の経路だけ — 空欄 = 選んだプロンプトの原文";
  // 先頭の「既定」は読み込み時に manifest の `defaultQuant` へ解決する（HF の取得元でも選べる）。席の名前は
  // このサーバの配布形の manifest から足す（{@link fillQuants}）。
  const defaultQuantOption = document.createElement("option");
  defaultQuantOption.value = "";
  defaultQuantOption.textContent = "既定（manifest の defaultQuant）";
  ui.quant.replaceChildren(defaultQuantOption);
  after(ui.source, field("世代", ui.generation));
  after(ui.generation, field("テキストエンコーダ", ui.textEncoder));
  after(ui.textEncoder, field("quant", ui.quant));
  after(ui.prompt, field("自由プロンプト", ui.freePrompt));
  // 条件画像の操作（寸法の選択の隣 — I2V は Wan2.2 だけ）。
  ui.image.type = "file";
  ui.image.accept = "image/png,image/jpeg";
  ui.fit.replaceChildren(...FIT_CHOICES.map(({ value, label }) => optionOf(value, label)));
  ui.clearImage.type = "button";
  ui.clearImage.textContent = "画像を外す";
  after(ui.size, field("条件画像（Wan2.2 の I2V）", ui.image));
  after(ui.image, field("fit", ui.fit));
  after(ui.fit, ui.clearImage);

  const state: {
    busy: boolean;
    /** 選んでいる世代（読み込み中は読み込んだ世代 — 選択は読み込み前だけ変えられる）。 */
    generation: WanLabGeneration;
    /** 選んだ条件画像（復号済み — 世代が画像を受けるときだけ持つ）。 */
    image?: ChosenImage;
    /** 条件画像の復号中（生成を止める — 選んだ画像を渡さずに T2V で回さない）。 */
    imageDecoding: boolean;
    loaded?: Loaded;
    /** 読み込みの記録（JSON の `loads[]`）。 */
    loads: {
      readonly at: string;
      readonly ms: number;
      readonly source: string;
      readonly generation: WanLabGenerationId;
    }[];
    deviceLost?: { readonly reason: string; readonly message: string };
    /** 進行中の generate の診断（component → 直近の run）。 */
    diagnostics?: Map<WanRunComponent, SessionDiagnostics>;
    rows: WanRow[];
    /** 最新の動画（`fps` は生成した動画の値 — 世代の記述子の出力のフレームレート）。 */
    video?: { readonly frames: readonly ImageData[]; readonly fps: number };
    frame: number;
    player?: ReturnType<typeof setInterval>;
  } = {
    busy: false,
    generation: WAN_LAB_GENERATIONS[0],
    imageDecoding: false,
    loads: [],
    rows: [],
    frame: 0,
  };

  /** 世代が I2V の条件画像を受けるか（記述子の DiT の入力の形 — Wan2.2 の `"ti2v"` だけ）。 */
  const acceptsImage = (generation: WanLabGeneration): boolean =>
    generation.descriptor.ditInputForm === "ti2v";

  /** このサーバが配るその世代の配布形の名前（配っていなければ null — `/config.json`）。 */
  const servedName = (generation: WanLabGeneration): string | null =>
    generation.route === "wan" ? lab.config.wanSource : lab.config.wan22Source;

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
    const generation = state.generation;
    const info = loaded?.gpu.adapterInfo ?? lab.adapterInfo;
    ui.info.textContent = [
      adapterSummary(info),
      `環境キー ${loaded?.environmentKey ?? environmentKeyOf(info)}`,
      `世代 ${generation.label}`,
      loaded === undefined
        ? `配布形 ${
          servedName(generation) ??
            `無し（HF のリポ名を入れるか ${generation.serverOption} で指定）`
        }（未読み込み）`
        : `配布形 ${loaded.source}`,
      `テキストエンコーダ ${loaded?.textEncoder ?? `${ui.textEncoder.value}（未読み込み）`}`,
      `quant ${
        loaded?.quant ??
          `${ui.quant.value === "" ? "manifest の既定" : ui.quant.value}（未読み込み）`
      }`,
      loaded === undefined
        ? `幾何プロファイル ${requestedLabel(lab.settings().choice)}（読み込み時に確定）`
        : `幾何プロファイルの要求 ${loaded.geometryProfileRequested}${
          loaded.savedNote === undefined ? "" : `（${loaded.savedNote}）`
        }`,
      "GPU 時間は採らない（Wan のパイプラインは計測の device を拒む）",
      checkoutLabel(lab.config),
    ].join(" · ");
  };

  /** 判定表（device を取っていれば device の値で判定し、アダプタの値と並べる）。 */
  const renderLimits = (): void => {
    let frames: number;
    let size: { width: number; height: number };
    try {
      frames = Number(ui.frames.value);
      size = resolveWanFormSize(ui.size.value, state.image?.size);
    } catch (error) {
      ui.limitsSummary.textContent = errorText(error);
      return;
    }
    const adapterLimits: WanLimits = lab.adapterLimits;
    const deviceLimits: WanLimits | undefined = state.loaded?.gpu.limits;
    const generation = state.generation;
    const judged = judgeWanLimits(generation, deviceLimits ?? adapterLimits, frames, size);
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
    const max = wanMaxFramesWithin(generation, deviceLimits ?? adapterLimits, size);
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
    ui.generation.disabled = busy || loaded;
    ui.textEncoder.disabled = busy || loaded;
    ui.quant.disabled = busy || loaded;
    ui.dispose.disabled = busy || !loaded;
    ui.run.disabled = busy || !loaded || state.imageDecoding;
    // Wan2.1 では画像の操作を無効にする（パイプラインが image / fit を拒む）。
    const imageAccepted = acceptsImage(state.generation);
    ui.image.disabled = busy || !imageAccepted;
    ui.fit.disabled = busy || !imageAccepted;
    ui.clearImage.disabled = busy || (state.image === undefined && !state.imageDecoding);
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
   * 名前（`wan` / `wan22` / `umt5` — `server.ts` の `/models/<名前>/`）、`crossRepo` は越境先の mapping。
   */
  const serverSource = (
    route: WanLabGeneration["route"] | "umt5",
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
    ui.prompt.replaceChildren(
      ...prompts.filter((p) => p.role === "positive").map((p) => optionOf(p.name, p.name)),
    );
    ui.negative.replaceChildren(
      optionOf("", "既定（資産の negative の行）"),
      ...prompts.map((p) => optionOf(p.name, `${p.name}（${p.role}）`)),
    );
    const first = ui.prompt.options.item(0);
    if (first !== null) ui.prompt.title = prompts.find((p) => p.name === first.value)?.prompt ?? "";
  };

  /** 取得元 → `fromPretrained` の第 1 引数と表示（空欄はこのサーバのその世代の配布形）。 */
  const readSource = (generation: WanLabGeneration): {
    ref: DistributionSource | HubRepoRef;
    label: string;
    local: boolean;
  } => {
    const text = ui.source.value.trim();
    if (text !== "") return { ref: { repo: text }, label: text, local: false };
    const served = servedName(generation);
    if (served === null) {
      throw Error(
        `このサーバは ${generation.label} の配布形を配っていない（${generation.serverOption} で指定するか、HF のリポ名を入れる）`,
      );
    }
    return {
      ref: serverSource(generation.route),
      label: `${served}（このサーバ）`,
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

  /** fit の選択（未知の値は選択肢の取り違え — 素の Error）。 */
  const readFit = (): WanI2vFit => {
    const found = FIT_CHOICES.find(({ value }) => value === ui.fit.value);
    if (found === undefined) throw Error(`fit '${ui.fit.value}' は選択肢に無い`);
    return found.value;
  };

  /**
   * `gpu` の経路でこのサーバの配布形を読む取得元。umT5 が越境参照なら、その repo（manifest の宣言から引く —
   * 名前を写経しない）を `/models/umt5/` の取得元へ結ぶ。umT5 を配っていなければ取得の前に名指しで落とす。
   */
  const gpuServerSource = async (
    generation: WanLabGeneration,
    manifest: Parameters<typeof resolveSelection>[0],
  ): Promise<DistributionSource> => {
    const repo = resolveSelection(manifest).containers[TEXT_ENCODER]?.parts[0]?.repo;
    if (repo === undefined) return serverSource(generation.route);
    const probe = await fetch("models/umt5/karume.json", { method: "HEAD" });
    if (!probe.ok) {
      throw Error(
        `このサーバは umT5 の配布形（${repo} の越境先）を配っていない（HTTP ${probe.status}）— ` +
          "--umt5-source で指定して起動し直すか、テキストエンコーダを precomputed にする",
      );
    }
    return serverSource(generation.route, { [repo]: serverSource("umt5") });
  };

  const load = async (): Promise<void> => {
    const generation = state.generation;
    const textEncoder = readTextEncoder();
    const { ref, label, local } = readSource(generation);
    const started = performance.now();
    status(`manifest を読み込み中（${label}）`);
    const manifest = await loadManifest(ref);
    const model = manifest.manifest.models[manifest.manifest.defaultModel];
    if (model === undefined) {
      throw Error(`defaultModel ${manifest.manifest.defaultModel} が models に無い`);
    }
    const config = parseWanPipelineConfig(model.pipelineConfig);
    // 回す席を名前で確定する（参照ケースの id はこの名前で決まる — 既定席が変わった配布形でも取り違えない）。
    const quant = ui.quant.value === "" ? model.defaultQuant : ui.quant.value;
    let manifestSha256: string | undefined;
    if (local) {
      const response = await fetch(`models/${generation.route}/karume.json`);
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
    const source = textEncoder === "gpu" && local
      ? await gpuServerSource(generation, manifest.manifest)
      : ref;
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
    let runner: WanRunner;
    try {
      runner = await openRunner(generation, source, {
        gpu,
        // MUST: 経路は必ず明示する（パイプラインの既定は "gpu" — タブの既定の precomputed と食い違う）。
        textEncoder,
        // MUST: 席も解決した名前で明示する（参照ケースの id と回した席を 1 つの値から決める）。
        quant,
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
    const prompts = runner.prompts;
    const negatives = prompts.filter((p) => p.role === "negative");
    state.loaded = {
      gpu,
      generation,
      runner,
      quant,
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
    state.loads.push({
      at: new Date().toISOString(),
      ms,
      source: label,
      generation: generation.id,
    });
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
    state.video = {
      frames: frames.map((rgba) => new ImageData(rgba, video.width, video.height)),
      fps: video.fps,
    };
    ui.canvas.width = video.width;
    ui.canvas.height = video.height;
    ui.canvas.hidden = false;
    ui.seek.max = String(video.frames - 1);
    for (const control of [ui.previous, ui.play, ui.next, ui.seek]) control.disabled = false;
    showFrame(0);
  };

  const formatImage = (image: WanRowImage): string =>
    `I2V ${image.file}（${image.width}x${image.height}・fit ${image.fit}${
      image.sizeFromImage ? "・寸法は画像から自動" : ""
    }） · `;

  const formatRequest = (row: WanRow): string =>
    `${row.generation} · ${row.quant} · ${row.textEncoder} · ${
      row.image === undefined ? "" : formatImage(row.image)
    }${row.freePrompt === undefined ? row.request.prompt : JSON.stringify(row.freePrompt)} · ${
      formatKnobs(row.request)
    }`;

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
      // 段の順（vae_encoder → text_encoder → transformer → vae_decoder — text は別に採るので並びを明示する）
      timeline === undefined ? "—" : ([
        ["vae_encoder", timeline.stageMs.vae_encoder],
        ["text_encoder", row.textEncoderMs],
        ["transformer", timeline.stageMs.transformer],
        ["vae_decoder", timeline.stageMs.vae_decoder],
      ] as const).flatMap(([stage, ms]) =>
        ms === undefined ? [] : [`${stage} ${(ms / 1000).toFixed(1)} s`]
      ).join("\n"),
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
    const chosen = state.image;
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
      chosen?.size,
    );
    const image: { readonly row: WanRowImage; readonly rgb: Rgb8Image } | undefined =
      chosen === undefined ? undefined : {
        row: {
          file: chosen.file,
          width: chosen.rgb.width,
          height: chosen.rgb.height,
          fit: readFit(),
          sizeFromImage: ui.size.value === WAN_SIZE_FROM_IMAGE,
        },
        rgb: chosen.rgb,
      };
    const freeText = ui.freePrompt.value;
    const freePrompt = loaded.textEncoder === "gpu" && freeText.trim() !== ""
      ? freeText
      : undefined;
    // 参照ケースの照合は事前計算の経路の T2V だけ（`wanReferenceCaseId` の id は事前計算の経路の T2V の sha 行 —
    // GPU 経路や I2V の動画は同じ条件でも値が違う）。
    const caseId = loaded.textEncoder === "precomputed" && image === undefined
      ? wanReferenceCaseId(
        loaded.generation,
        resolved,
        loaded.config,
        loaded.defaultNegative,
        loaded.quant,
      )
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
      generation: loaded.generation.id,
      request: resolved,
      quant: loaded.quant,
      textEncoder: loaded.textEncoder,
      ...(freePrompt === undefined ? {} : { freePrompt }),
      ...(image === undefined ? {} : { image: image.row }),
      ...(caseId === undefined ? {} : { caseId }),
    };
    let row: WanRow;
    try {
      const video = await loaded.runner.generate({
        ...request,
        ...(freePrompt === undefined ? {} : { prompt: freePrompt }),
        ...(image === undefined ? {} : { image: image.rgb, fit: image.row.fit }),
        onEvent,
      });
      const wallMs = performance.now() - started;
      // 「画像から自動」の寸法はタブが製品の規則で求めた値を記録する — パイプラインの選んだ寸法と食い違ったら、
      // 記録の寸法（と判定表）が実物と違うので失敗の行にする。
      if (video.width !== resolved.width || video.height !== resolved.height) {
        throw Error(
          `動画の寸法 ${wanSizeLabel(video)} が記録する寸法 ${wanSizeLabel(resolved)} と違う`,
        );
      }
      status("フレームを画素にして sha256 を計算中");
      const frames = Array.from(
        { length: video.frames },
        (_, frame) => wanFrameToRgba(video, frame),
      );
      const rgbSha256 = await sha256Hex(wanRgbBytes(frames, video.width * video.height));
      const reference = checkWanReference(
        REFERENCES[loaded.generation.id],
        caseId,
        loaded.environmentKey,
        rgbSha256,
      );
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
      await loaded?.runner.dispose();
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
        generation: loaded.generation.id,
        source: loaded.source,
        quant: loaded.quant,
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

  /**
   * 選んだ世代の選択肢（受理集合と既定は世代の仕様の記述子が正本 — Wan2.1 は 4n+1 の 5〜81・Wan2.2 は 5〜121・寸法は
   * 2 通り）と取得元の placeholder。
   */
  const fillChoices = (): void => {
    const generation = state.generation;
    const { defaults } = generation.descriptor;
    ui.frames.replaceChildren(
      ...wanFrameChoices(generation).map((frames) =>
        optionOf(String(frames), String(frames), frames === defaults.frames)
      ),
    );
    fillSizes();
    const served = servedName(generation);
    ui.source.placeholder = served === null
      ? `owner/name（このサーバは ${generation.label} の配布形を配っていない）`
      : `空欄 = このサーバの ${served} · HF なら owner/name`;
  };
  /**
   * 寸法の選択肢（受理集合）。条件画像があれば先頭に「画像から自動」（画像から選ばれる寸法を添える）を足して既定にし、
   * 無ければ記述子の既定の寸法を選ぶ。
   */
  const fillSizes = (): void => {
    const { acceptedSizes, defaults } = state.generation.descriptor;
    const image = state.image;
    ui.size.replaceChildren(
      ...(image === undefined
        ? []
        : [optionOf(WAN_SIZE_FROM_IMAGE, `画像から自動（${wanSizeLabel(image.size)}）`, true)]),
      ...acceptedSizes.map((size) =>
        optionOf(
          wanSizeLabel(size),
          wanSizeLabel(size),
          image === undefined && size.width === defaults.width &&
            size.height === defaults.height,
        )
      ),
    );
  };

  /** 条件画像の選択と復号の世代番号（復号の途中で選び直す・外すと、前の復号の結果は捨てる）。 */
  let imageRequest = 0;

  /**
   * 条件画像の状態だけを画像なしへ戻す（途中の復号の結果も捨てる）。描画（寸法の選択肢・判定表・操作の有効化）は
   * 呼び手が行う — 世代の切り替えでは、判定表を新しい世代の選択肢で作る前に描画すると、旧世代のフレーム数が新しい
   * 世代の受理集合の外で投げる。
   */
  const resetImage = (): void => {
    imageRequest += 1;
    state.image = undefined;
    state.imageDecoding = false;
    ui.image.value = "";
  };
  /** 条件画像を外す（寸法の選択肢も画像なしへ戻す）。 */
  const clearImage = (): void => {
    resetImage();
    fillSizes();
    renderLimits();
    setBusy(state.busy);
  };

  /** 選んだ画像ファイルを復号して持つ（失敗は状態行へ — 画像なしに戻す）。 */
  const chooseImage = async (): Promise<void> => {
    const file = ui.image.files?.item(0) ?? null;
    if (file === null) {
      clearImage();
      return;
    }
    const request = ++imageRequest;
    state.image = undefined;
    state.imageDecoding = true;
    setBusy(state.busy);
    status(`条件画像 ${file.name} を復号中`);
    let chosen: ChosenImage | undefined;
    let failure: unknown;
    try {
      const rgb = await decodeImage(file);
      chosen = {
        file: file.name,
        rgb,
        size: selectWanI2vSize(rgb, {}, state.generation.descriptor),
      };
    } catch (error) {
      failure = error;
    }
    if (request !== imageRequest) return;
    if (chosen === undefined) {
      clearImage();
      status(`条件画像 ${file.name} を読めない — ${errorText(failure)}`);
      return;
    }
    state.image = chosen;
    state.imageDecoding = false;
    fillSizes();
    renderLimits();
    setBusy(state.busy);
    status(
      `条件画像 ${chosen.file}（${chosen.rgb.width}x${chosen.rgb.height}）を読みました — ` +
        `寸法「画像から自動」は ${wanSizeLabel(chosen.size)}`,
    );
  };

  for (const control of [ui.previous, ui.play, ui.next, ui.seek]) control.disabled = true;

  ui.load.addEventListener("click", exclusive("Wan の読み込み", load));
  ui.image.addEventListener("change", () => void chooseImage());
  ui.clearImage.addEventListener("click", clearImage);
  ui.run.addEventListener("click", exclusive("Wan の生成", generate));
  ui.dispose.addEventListener("click", exclusive("Wan の破棄", dispose));
  ui.exportJson.addEventListener("click", exportJson);
  ui.generation.addEventListener("change", () => {
    const found = WAN_LAB_GENERATIONS.find(({ id }) => id === ui.generation.value);
    if (found === undefined) {
      status(`世代 '${ui.generation.value}' は選択肢に無い`);
      return;
    }
    state.generation = found;
    // 画像を受けない世代へ替えたら条件画像を外す（Wan2.1 のパイプラインは image を拒む）。描画は下の fillChoices 以降が
    // 新しい世代の選択肢で 1 回だけ行う。
    if (!acceptsImage(found)) resetImage();
    fillChoices();
    renderInfo();
    renderLimits();
    setBusy(state.busy);
    loadQuants();
  });
  ui.textEncoder.addEventListener("change", renderInfo);
  ui.quant.addEventListener("change", renderInfo);
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
    const video = state.video;
    if (video === undefined) return;
    state.player = setInterval(() => showFrame(state.frame + 1), 1000 / video.fps);
    ui.play.textContent = "停止";
  });

  /**
   * quant の選択肢を、このサーバが配る選んだ世代の配布形の manifest の既定モデルの欄から作り直す（`defaultQuant` に
   * 「（既定）」）。配っていなければ先頭の「既定」だけ（読み込み時に取得元の manifest で解決する）。応答が届く前に
   * 次の作り直しが始まったら（世代の選択を変えた — `request` が最新の番号でない）、この応答の席は足さない（失敗も
   * {@link loadQuants} が捨てる）。
   */
  let quantsRequest = 0;
  const fillQuants = async (generation: WanLabGeneration, request: number): Promise<void> => {
    ui.quant.replaceChildren(defaultQuantOption);
    if (servedName(generation) === null) return;
    const response = await fetch(`models/${generation.route}/karume.json`);
    if (!response.ok) throw Error(`karume.json HTTP ${response.status}`);
    const manifest = parseManifest(await response.text());
    const model = manifest.models[manifest.defaultModel];
    if (model === undefined) {
      throw Error(`defaultModel ${manifest.defaultModel} が models に無い`);
    }
    if (request !== quantsRequest) return;
    ui.quant.append(
      ...Object.entries(model.quants).map(([name, quant]) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = `${name}${name === model.defaultQuant ? "（既定）" : ""}${
          quant.label === undefined ? "" : ` — ${quant.label}`
        }`;
        return option;
      }),
    );
  };

  const loadQuants = (): void => {
    const request = ++quantsRequest;
    fillQuants(state.generation, request).catch((error: unknown) => {
      // 切り替える前の世代の失敗は、今の世代の選択肢の状態ではないので出さない。
      if (request !== quantsRequest) return;
      status(`quant の選択肢を読めない（「既定」だけ選べる）— ${errorText(error)}`);
    });
  };

  fillChoices();
  renderInfo();
  renderLimits();
  setBusy(false);
  status(
    "「読み込む」で GPU を取り、配布形を読みます（世代・テキストエンコーダの経路・quant の席は読み込み時に決まる）。" +
      "判定表は選んだフレーム数・寸法で更新されます。",
  );
  loadQuants();

  return {
    reset: async () => {
      if (state.busy) throw Error("Wan のタブが実行中 — 終わってから適用する");
      const held = state.loaded !== undefined;
      if (held) await dispose();
      return held;
    },
  };
};
