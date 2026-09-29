/**
 * タイル幾何の掃引を Deno で回す CLI（ブラウザのページ `browser/` の双子 — perf-ledger K-70）。
 *
 * 同じ shape・同じ入力のまま幾何だけを変えて 1 dispatch の時間と出力 digest を採り、既定幾何に対する
 * 速さの比の表を標準出力へ、全行を JSON（`karume-geometry-sweep/2` — `report.ts`）へ書く。
 * 失敗（`error`）の行が 1 つでもあれば終了コード 1（出力の不一致は測定の失敗ではないので終了コードに
 * 含めず、表と要約で赤く出す）。
 *
 * 使い方（リポ直下から・GPU が学習で使われていないことを先に `outputs/diag/gpu-busy.zsh` で確認）:
 *
 *   deno run -A tools/geometry-sweep/main.ts --quick --op linear --op i8a8-linear
 *
 * フラグ: `--op <族>`（複数可・既定は全族 — linear / matmul / bmm / i8a8-linear / attention /
 * i8a8-attention / conv2d）`--case <id>`（複数可・cases.ts の id で絞る）`--quick`（小集合）`--rounds N`（既定 5）
 * `--out <json>`（既定 outputs/bench/karume/<日付>_geometry-sweep/geometry-sweep-<adapter>-<時刻>.json）。
 *
 * アダプタが `timestamp-query` を列挙すれば `acquireGpu({ gpuTiming: true })` で取り、単位は
 * `deno-raw-tick`（Deno は timestamp を ns に換算しない）。列挙しなければ壁時計（`wall`）で回す。
 *
 * サブコマンド `profile`（GPU を使わない — `profile.ts`）は掃引の JSON から adapter 1 種の幾何
 * プロファイルの生成物を書く（perf-ledger K-71）:
 *
 *   deno run -A tools/geometry-sweep/main.ts profile --from <sweep.json> --id <id> --vendor <v> \
 *     [--architecture <a>] --out packages/runtime/src/kernels/geometry-profiles/<id>.ts \
 *     [--min-speedup 1.05] [--check]
 */
import { acquireGpu } from "../../packages/runtime/mod.ts";
import { readCheckout } from "../anima-residency/browser/server.ts";
import { SWEEP_CASES, SWEEP_OPS, type SweepCase, type SweepOp } from "./cases.ts";
import { runProfileCommand } from "./profile.ts";
import {
  createSweepContext,
  destroySweepContext,
  ROUNDS,
  runSweep,
  SWEEP_MAX_REPS,
  TARGET_PASS_MS,
  WARMUP_MIN_RUNS,
  WARMUP_NS,
} from "./harness.ts";
import {
  type CaseSummary,
  DEFAULT_DRIFT_RANGE,
  driftOutOfRange,
  type Report,
  REPORT_FORMAT,
  roundsLookQuantized,
  type SweepRow,
  type TimingUnit,
  WALL_TIMING_NOTE,
} from "./report.ts";

const TIMESTAMP_QUERY = "timestamp-query";

type Flags = {
  readonly ops: readonly SweepOp[];
  readonly cases: readonly string[];
  readonly quick: boolean;
  readonly rounds: number;
  readonly out?: string;
};

const isSweepOp = (value: string): value is SweepOp =>
  (SWEEP_OPS as readonly string[]).includes(value);

/** `--quick` 以外は `--key value` の対（未知のキーは落とす — 綴り違いが既定で走らない）。 */
export const parseFlags = (argv: readonly string[]): Flags => {
  const ops: SweepOp[] = [];
  const cases: string[] = [];
  let quick = false;
  let rounds = ROUNDS;
  let out: string | undefined;
  for (let at = 0; at < argv.length; at += 1) {
    const key = argv[at];
    if (key === "--quick") {
      quick = true;
      continue;
    }
    const value = argv[at + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`引数を読めない: ${key}`);
    }
    at += 1;
    switch (key) {
      case "--op":
        if (!isSweepOp(value)) {
          throw new Error(`--op ${value} は ${SWEEP_OPS.join(" / ")} のどれでもない`);
        }
        ops.push(value);
        break;
      case "--case":
        if (!SWEEP_CASES.some((sweepCase) => sweepCase.id === value)) {
          throw new Error(`--case ${value} が cases.ts に無い`);
        }
        cases.push(value);
        break;
      case "--rounds":
        rounds = Number(value);
        if (!Number.isInteger(rounds) || rounds < 1) {
          throw new Error(`--rounds は 1 以上の整数（${value}）`);
        }
        break;
      case "--out":
        out = value;
        break;
      default:
        throw new Error(`引数を読めない: ${key} ${value}`);
    }
  }
  return {
    ops: ops.length === 0 ? SWEEP_OPS : ops,
    cases,
    quick,
    rounds,
    ...(out === undefined ? {} : { out }),
  };
};

/** `--op` と `--case` の両方に合うケース（`--case` 無しなら op の全ケース）。 */
export const selectCases = (flags: Pick<Flags, "ops" | "cases">): SweepCase[] => {
  const selected = SWEEP_CASES.filter((sweepCase) =>
    flags.ops.includes(sweepCase.op) &&
    (flags.cases.length === 0 || flags.cases.includes(sweepCase.id))
  );
  if (selected.length === 0) {
    throw new Error(
      `--op [${flags.ops.join(", ")}] と --case [${flags.cases.join(", ")}] に合うケースが無い`,
    );
  }
  return selected;
};

/**
 * ファイル名に載せるアダプタの名前（tools/anima-residency/profile.ts と同じ規則 — architecture が
 * 空なら description を使う。Deno は vendor を PCI ID の 10 進で返し architecture を空にする）。
 */
const adapterSlug = (info: GPUAdapterInfo): string => {
  const parts = info.architecture === ""
    ? [info.description === "" ? info.vendor : info.description]
    : [info.vendor, info.architecture];
  const slug = parts.join("-").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return slug === "" ? "unknown-adapter" : slug;
};

const localDate = (): string => {
  const now = new Date();
  const pad = (value: number): string => String(value).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
};

const unitLabel = (unit: TimingUnit): string =>
  unit === "ns" ? "µs" : unit === "deno-raw-tick" ? "kTick" : "µs(wall)";

/** 赤で 1 行（Deno の console は `%c` の CSS color を端末の色へ写す）。 */
const printLine = (text: string, red = false): void => {
  if (red) console.log("%c%s", "color: red", text);
  else console.log(text);
};

/** 表の 1 行（幾何・1 dispatch・既定比・TFLOPS・digest の一致）。 */
const formatRow = (row: SweepRow): string => {
  const mark = row.isDefault ? "*" : " ";
  if (row.error !== undefined) return `  ${mark} ${row.geometry.padEnd(24)} 失敗: ${row.error}`;
  const time = ((row.perDispatch ?? 0) / 1e3).toFixed(1).padStart(10);
  const speedup = row.speedupVsDefault === undefined
    ? "—".padStart(7)
    : `×${row.speedupVsDefault.toFixed(2)}`.padStart(7);
  const tflops = row.tflops === undefined ? "—".padStart(7) : row.tflops.toFixed(2).padStart(7);
  const identical = row.identicalToDefault === undefined
    ? "—"
    : row.identicalToDefault
    ? "一致"
    : "不一致（採用不可）";
  const clamped = (row.clampedNegativeSamples ?? 0) > 0
    ? `  負の timestamp ${row.clampedNegativeSamples} round を除外`
    : "";
  return `  ${mark} ${row.geometry.padEnd(24)}${time}${speedup}${tflops}  reps ${
    String(row.reps).padStart(4)
  }  ${identical}${clamped}`;
};

/** ケースの末尾の 1 行（既定幾何の再測定と、範囲外なら測り直しの警告）。 */
const formatCase = (summary: CaseSummary): { readonly text: string; readonly red: boolean } => {
  if (summary.defaultRepeatError !== undefined) {
    return { text: `    既定の再測定 失敗: ${summary.defaultRepeatError}`, red: true };
  }
  if (summary.defaultRepeat === undefined) {
    return { text: "    既定の再測定 —（既定の初回が失敗・中断・device lost）", red: false };
  }
  const { driftRatio } = summary.defaultRepeat;
  const drifted = driftOutOfRange(driftRatio);
  return {
    text: `    既定の再測定 ×${driftRatio.toFixed(2)}${
      drifted
        ? `  警告: ${DEFAULT_DRIFT_RANGE.min}〜${DEFAULT_DRIFT_RANGE.max} の外 — このケースは測り直しの目安`
        : ""
    }`,
    red: drifted,
  };
};

const main = async (): Promise<void> => {
  const flags = parseFlags(Deno.args);
  const selected = selectCases(flags);
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) throw new Error("WebGPU アダプタが無い");
  const timestampFeature = adapter.features.has(TIMESTAMP_QUERY);
  let deviceLost: Report["deviceLost"] = null;
  const gpu = await acquireGpu({
    ...(timestampFeature ? { gpuTiming: true } : {}),
    onDeviceLost: (info) => {
      deviceLost = { reason: info.reason, message: info.message };
    },
  });
  const rows: SweepRow[] = [];
  const cases: CaseSummary[] = [];
  const context = await createSweepContext(gpu, "deno-raw-tick");
  const label = unitLabel(context.unit);
  console.log(
    `[geometry-sweep] ${
      gpu.adapterInfo.description || gpu.adapterInfo.vendor
    } · 単位 ${context.unit} · ${
      flags.quick ? "quick" : "full"
    } · rounds ${flags.rounds} · dp4a ${context.dp4a} · ${selected.length} ケース`,
  );
  let current: string | undefined;
  try {
    await runSweep(context, selected, {
      rounds: flags.rounds,
      quick: flags.quick,
      timestampUnit: "deno-raw-tick",
    }, {
      onProgress: (message) => console.error(`  … ${message}`),
      onRow: (row) => {
        if (row.caseId !== current) {
          current = row.caseId;
          console.log(
            `\n${row.caseId}（${row.shape} · census ${row.censusCount} 本）\n    ${
              "geometry".padEnd(24)
            }${`${label}/disp`.padStart(10)}${"vs既定".padStart(7)}${"TFLOPS".padStart(7)}`,
          );
        }
        rows.push(row);
        printLine(formatRow(row), row.error !== undefined || row.identicalToDefault === false);
      },
      onCase: (summary) => {
        cases.push(summary);
        const { text, red } = formatCase(summary);
        printLine(text, red);
      },
    });
  } finally {
    destroySweepContext(context);
    const info = gpu.adapterInfo;
    const report: Report = {
      format: REPORT_FORMAT,
      date: new Date().toISOString(),
      userAgent: { deno: Deno.version.deno },
      adapter: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      },
      ...await readCheckout().then(
        ({ revision, dirty }) => ({ checkout: revision, checkoutDirty: dirty }),
      ),
      gpuTiming: {
        feature: gpu.gpuTimingEnabled,
        unit: context.unit,
        quantized: roundsLookQuantized(rows, context.unit),
      },
      dp4a: context.dp4a,
      settings: {
        quick: flags.quick,
        ops: flags.ops,
        ...(flags.cases.length === 0 ? {} : { cases: flags.cases }),
        rounds: flags.rounds,
        targetPassMs: TARGET_PASS_MS,
        maxReps: SWEEP_MAX_REPS,
        warmupNs: WARMUP_NS,
        warmupMinRuns: WARMUP_MIN_RUNS,
        wallTimingNote: WALL_TIMING_NOTE,
      },
      deviceLost,
      cases,
      rows,
    };
    gpu.destroy();
    const stamp = report.date.replaceAll(":", "-");
    const path = flags.out ??
      `outputs/bench/karume/${localDate()}_geometry-sweep/geometry-sweep-${
        adapterSlug(info)
      }-${stamp}.json`;
    await Deno.mkdir(path.slice(0, Math.max(0, path.lastIndexOf("/"))) || ".", { recursive: true });
    await Deno.writeTextFile(path, `${JSON.stringify(report, null, 2)}\n`);
    console.log(`\n[geometry-sweep] ${path}`);
  }
  const failed = rows.filter((row) => row.error !== undefined).length;
  const mismatched = rows.filter((row) => row.identicalToDefault === false).length;
  const drifted = cases.filter((summary) =>
    summary.defaultRepeatError !== undefined ||
    (summary.defaultRepeat !== undefined && driftOutOfRange(summary.defaultRepeat.driftRatio))
  ).length;
  printLine(
    `[geometry-sweep] ${rows.length} 行 · 失敗 ${failed} · 不一致 ${mismatched} · 既定の再測定が範囲外 / 失敗 ${drifted} ケース`,
    failed + mismatched + drifted > 0,
  );
  // 失敗の行は測れていない組（表が欠けている）ので終了コードで知らせる。不一致は測れた結果
  if (failed > 0) Deno.exitCode = 1;
};

if (import.meta.main) {
  if (Deno.args[0] === "profile") Deno.exitCode = await runProfileCommand(Deno.args.slice(1));
  else await main();
}
