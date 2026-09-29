/**
 * anima の generate 1 回ぶんの記録と、書き出す JSON（`karume-anima-residency-browser/2`）の形。
 *
 * 確認ページ（`tools/gpu-lab` の Anima タブ）と Deno の双子 CLI（`profile.ts`）が同じ記録器と同じ型を使う —
 * 2 本が別々に組むと、JSON を並べて比べる段で形が黙ってずれる（perf-ledger K-70）。
 */
import type {
  AnimaGenerateEvent,
  AnimaResidency,
  AnimaResidencyAction,
  AnimaResidencyReason,
  AnimaRunComponent,
} from "../../packages/models/anima.ts";
import type { GeometryProfile, SessionDiagnostics } from "../../packages/runtime/mod.ts";
import {
  aggregateRuns,
  type PipelineCount,
  type RunDiagnostics,
  type RunSample,
  snapshotRun,
  type StageGpuTiming,
} from "./timing.ts";

export const REPORT_FORMAT = "karume-anima-residency-browser/2";

/** ページと CLI の既定のプロンプト（両者の走行を同じ入力で並べるため 1 か所に置く）。 */
export const DEFAULT_PROMPT =
  "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, outdoors, smile, " +
  "upper body, masterpiece, best quality";

/**
 * 段 1 回ぶん（OOM 退避のやり直しを段の外で組み直した場合は同じ段が 2 回出る）。時刻は generate
 * 開始からの ms。`gpu` / `pipelines` はその段の中で終わった run の合計（{@link aggregateRuns}）。
 */
export type StageRecord = {
  readonly component: AnimaRunComponent;
  readonly startMs: number;
  readonly endMs?: number;
  /**
   * その段の run を回した Session が選んだ GEMM 幾何プロファイルの id（ADR 0115 —
   * `SessionDiagnostics.geometryProfile`）。run が 1 回も終わらなかった段では無い。
   */
  readonly geometryProfile?: string;
  readonly gpu?: StageGpuTiming;
  readonly pipelines: readonly PipelineCount[];
};

export type ResidencyRecord = {
  readonly atMs: number;
  readonly action: AnimaResidencyAction;
  readonly reason: AnimaResidencyReason;
  /** イベントが出た位置（開いている段 / 直前に閉じた段の後 / 最初の段の前）。 */
  readonly position: string;
};

/**
 * どの幾何プロファイルを `acquireGpu` に頼んだか（ADR 0115 追記決定 6 の注入口）: `auto` = 注入なし
 * （adapter の (vendor, architecture, description) で埋め込みの表が選ばれる）・`default` = 既定の表を注入・
 * `builtin:<id>` = 埋め込みの表 `<id>` を注入・`generated:<id>` = ページで掃引から作った表を注入。
 * 段ごとの `geometryProfile`（診断）は使われた表の id だけで注入か埋め込みかを区別しないので、この欄で
 * 補う。CLI は注入しないので常に `auto`。
 */
export type GeometryProfileRequested =
  | "auto"
  | "default"
  | `builtin:${string}`
  | `generated:${string}`;

export type Row = {
  readonly index: number;
  /** この generate を回した pipeline の quant（pipeline を組み直すと変わりうるので行ごと）。 */
  readonly quant: string;
  /** この generate を回した GPU に頼んだ幾何プロファイル（GPU を取り直すと変わりうるので行ごと）。 */
  readonly geometryProfileRequested: GeometryProfileRequested;
  /**
   * この generate を回した GPU に注入した表の値そのもの（`auto` のときは無い）。生成した表は作り直すと
   * 同じ id で中身が変わるので、id の要求だけでは後から何を使ったか辿れない。
   */
  readonly geometryProfileInjected?: GeometryProfile;
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
  readonly stages: readonly StageRecord[];
  readonly residency: readonly ResidencyRecord[];
  readonly pngSha256?: string;
  readonly error?: { readonly name: string; readonly message: string };
};

export type DummyHold = {
  readonly at: string;
  readonly requestedGib: number;
  readonly allocatedBytes: number;
  readonly buffers: number;
  readonly stop?: string;
};

export type PipelineLoad = {
  readonly at: string;
  readonly ms: number;
  readonly quant: string;
  readonly gpuTiming: boolean;
};

export type Report = {
  readonly format: typeof REPORT_FORMAT;
  readonly date: string;
  /** ブラウザは `navigator.userAgent`、Deno の CLI は `{ deno: Deno.version.deno }`。 */
  readonly userAgent: string | { readonly deno: string };
  readonly adapter: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
  } | null;
  readonly checkout?: string;
  readonly checkoutDirty?: boolean;
  /** ページの bundle の sha256（CLI は bundle しないので無い）。 */
  readonly bundleSha256?: string;
  readonly source?: string;
  readonly manifestSha256?: string;
  readonly defaultModel?: string;
  /** いま組んでいる（無ければ次に組む）pipeline の quant。行ごとの値は `rows[].quant`。 */
  readonly quant: string;
  /**
   * いま持っている（無ければ次に取る）GPU に頼んだ幾何プロファイル（{@link GeometryProfileRequested}）。
   * 行ごとの値は `rows[].geometryProfileRequested`。
   */
  readonly geometryProfileRequested: GeometryProfileRequested;
  /**
   * いま持っている（無ければ次に取る）GPU に注入する表の値そのもの（`auto` のときは無い）。行ごとの
   * 値は `rows[].geometryProfileInjected`。
   */
  readonly geometryProfileInjected?: GeometryProfile;
  readonly gpuTiming: {
    /** いま持っている（無ければ次に取る）GPU で計測を要求したか。行ごとの有無は `stages[].gpu`。 */
    readonly enabled: boolean;
    /** アダプタが `timestamp-query` を列挙したか。 */
    readonly feature: boolean;
    /**
     * `stages[].gpu` の `ns` の単位。Chrome（Dawn）は仕様どおり ns、Deno は wgpu の raw tick を
     * 換算しない（docs/known-issues.md「Intel Arc B570」節 — B570 では 1 tick = 52.08 ns）。
     */
    readonly unit: "ns" | "deno-raw-tick";
  };
  readonly pipelineResidency: "transformer";
  readonly pipelineLoads: readonly PipelineLoad[];
  readonly dummies: {
    readonly heldBytes: number;
    readonly buffers: number;
    readonly holds: readonly DummyHold[];
  };
  readonly deviceLost: { readonly reason: string; readonly message: string } | null;
  readonly rows: readonly Row[];
};

type OpenStage = {
  readonly component: AnimaRunComponent;
  readonly startMs: number;
  endMs?: number;
  geometryProfile?: string;
  readonly samples: RunSample[];
};

/** 記録器が run ごとに読む診断（集計の 2 欄 + 幾何プロファイルの id）。 */
export type StageRunDiagnostics = RunDiagnostics & Pick<SessionDiagnostics, "geometryProfile">;

/**
 * 段の記録に出た幾何プロファイルの id（重複を除き、最初に出た順）。
 *
 * 同じ device の Session は全段で同じ id を名乗るはずなので、普通は 1 つ。2 つ以上なら段ごとの
 * `geometryProfile` を読む（ここで 1 つに畳まない）。
 */
export const geometryProfilesOf = (stages: readonly StageRecord[]): string[] => [
  ...new Set(stages.flatMap(({ geometryProfile }) => geometryProfile ?? [])),
];

/** generate 1 回ぶんの記録器（{@link createGenerateRecorder}）。 */
export type GenerateRecorder = {
  /** 時刻の原点を今へ置き直す（pipeline の構築を所要から外すため、generate の直前に呼ぶ）。 */
  readonly restart: () => void;
  readonly elapsedMs: () => number;
  /** `stage` と `residency` を記録する（進捗表示は呼び手が持つ）。 */
  readonly onEvent: (event: AnimaGenerateEvent) => void;
  /** pipeline の `onRunDiagnostics` から回す。run はその時点で開いている同名の段に帰属させる。 */
  readonly onRun: (component: AnimaRunComponent, diagnostics: StageRunDiagnostics) => void;
  readonly finish: () => {
    readonly stages: readonly StageRecord[];
    readonly residency: readonly ResidencyRecord[];
  };
};

export const createGenerateRecorder = (now: () => number): GenerateRecorder => {
  let origin = now();
  const stages: OpenStage[] = [];
  const residency: ResidencyRecord[] = [];
  const openStage = (component?: AnimaRunComponent): OpenStage | undefined =>
    stages.findLast((stage) =>
      stage.endMs === undefined && (component === undefined || stage.component === component)
    );
  const position = (): string => {
    const open = openStage();
    if (open !== undefined) return `${open.component} の途中`;
    const last = stages.at(-1);
    return last === undefined ? "最初の段の前" : `${last.component} の後`;
  };
  return {
    restart: () => {
      origin = now();
    },
    elapsedMs: () => now() - origin,
    onEvent: (event) => {
      const atMs = now() - origin;
      if (event.kind === "stage") {
        if (event.at === "start") {
          stages.push({ component: event.component, startMs: atMs, samples: [] });
        } else {
          const open = openStage(event.component);
          if (open !== undefined) open.endMs = atMs;
        }
      } else if (event.kind === "residency") {
        residency.push({ atMs, action: event.action, reason: event.reason, position: position() });
      }
    },
    onRun: (component, diagnostics) => {
      const open = openStage(component);
      // MUST: 段の外の run は帰属先が無い — pipeline の段の境目の前提が崩れているので落とす
      // （黙って捨てると GPU 時間の合計が段の実際より小さく出る）。
      if (open === undefined) throw new Error(`${component} の run が段の外で終わった`);
      // MUST: 1 段の run は 1 本の device の上の Session なので id は 1 つ — 割れたら前提が
      // 崩れているので落とす（どちらか 1 つを黙って残すと、どの表で走ったかを取り違える）。
      const { geometryProfile } = diagnostics;
      if (open.geometryProfile !== undefined && open.geometryProfile !== geometryProfile) {
        throw new Error(
          `${component} の段で幾何プロファイルが割れた（${open.geometryProfile} と ${geometryProfile}）`,
        );
      }
      open.geometryProfile = geometryProfile;
      open.samples.push(snapshotRun(diagnostics));
    },
    finish: () => ({
      stages: stages.map(({ component, startMs, endMs, geometryProfile, samples }) => ({
        component,
        startMs,
        ...(endMs === undefined ? {} : { endMs }),
        ...(geometryProfile === undefined ? {} : { geometryProfile }),
        ...aggregateRuns(samples),
      })),
      residency,
    }),
  };
};
