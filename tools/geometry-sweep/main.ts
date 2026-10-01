/**
 * タイル幾何の掃引を Deno で回す CLI（ブラウザのページ `tools/gpu-lab` の掃引タブの双子 — perf-ledger K-70）。
 *
 * 同じ shape・同じ入力のまま幾何だけを変えて 1 dispatch の時間と出力 digest を採り、既定幾何に対する
 * 速さの比の表を標準出力へ、全行を JSON（`karume-geometry-sweep/2`）へ書く。掃引そのもの（専用の
 * device の取り直し・計測・記録の組み立て）は runtime の `runGeometrySweep`（`@karume/runtime/tune` —
 * ADR 0117 決定 2）で、ここは引数・表示・ファイルの書き出し・終了コードだけを持つ殻。
 * 失敗（`error`）の行が 1 つでもあれば終了コード 1（出力の不一致は測定の失敗ではないので終了コードに
 * 含めず、表と要約で赤く出す）。
 *
 * 使い方（リポ直下から・GPU が学習で使われていないことを先に `outputs/diag/gpu-busy.zsh` で確認）:
 *
 *   deno run -A tools/geometry-sweep/main.ts --set quick --op linear --op i8a8-linear
 *
 * フラグ: `--op <族>`（複数可・既定は全族 — linear / matmul / bmm / i8a8-linear / attention /
 * i8a8-attention / conv2d）`--case <id>`（複数可・runtime の `src/tune/cases.ts` の id で絞る）
 * `--set <quick|quick+|full>`（候補集合・既定 quick+ — `src/tune/geometries.ts`）`--quick`
 * （`--set quick` の別名）`--rounds N`（既定 5）
 * `--out <json>`（既定 outputs/bench/karume/<日付>_geometry-sweep/geometry-sweep-<adapter>-<時刻>.json）。
 *
 * アダプタが `timestamp-query` を列挙すれば timestamp で測り、単位は `deno-raw-tick`（Deno は
 * timestamp を ns に換算しない）。列挙しなければ壁時計（`wall`）で回す（どちらも `runGeometrySweep` が決める）。
 * 記録の `userAgent` は `{ deno: <版> }` に書き換え、`checkout` / `checkoutDirty` を足す。
 *
 * サブコマンド `profile`（GPU を使わない — `profile.ts`）は掃引の JSON から adapter 1 種の幾何
 * プロファイルの生成物を書く（perf-ledger K-71）:
 *
 *   deno run -A tools/geometry-sweep/main.ts profile --from <sweep.json> --id <id> \
 *     (--vendor <v> [--architecture <a> [--description <d>]] | --opt-in) \
 *     --out packages/runtime/src/kernels/geometry-profiles/<id>.ts [--min-speedup 1.05] [--check]
 */
import {
  type GeometrySweepAdapter,
  type GeometrySweepCaseSummary,
  type GeometrySweepReport,
  type GeometrySweepRow,
  type GeometrySweepTimingUnit,
  runGeometrySweep,
} from "../../packages/runtime/tune.ts";
import { SWEEP_CASES, SWEEP_OPS, type SweepOp } from "../../packages/runtime/src/tune/cases.ts";
import {
  CANDIDATE_SETS,
  type CandidateSet,
  DEFAULT_CANDIDATE_SET,
  isCandidateSet,
} from "../../packages/runtime/src/tune/geometries.ts";
import { ROUNDS } from "../../packages/runtime/src/tune/measurement.ts";
import { DEFAULT_DRIFT_RANGE, driftOutOfRange } from "../../packages/runtime/src/tune/report.ts";
import { readCheckout } from "../shared/checkout.ts";
import { runProfileCommand } from "./profile.ts";

type Flags = {
  readonly ops: readonly SweepOp[];
  readonly cases: readonly string[];
  readonly candidateSet: CandidateSet;
  readonly rounds: number;
  readonly out?: string;
};

const isSweepOp = (value: string): value is SweepOp =>
  (SWEEP_OPS as readonly string[]).includes(value);

/**
 * `--quick` 以外は `--key value` の対（未知のキーは落とす — 綴り違いが既定で走らない）。
 * 候補集合の指定（`--set` / `--quick`）は 1 回だけ（2 回目は値が同じでも落とす — どちらが効いたかを
 * 読み手に推測させない）。
 */
export const parseFlags = (argv: readonly string[]): Flags => {
  const ops: SweepOp[] = [];
  const cases: string[] = [];
  let candidateSet: CandidateSet | undefined;
  let rounds = ROUNDS;
  let out: string | undefined;
  const chooseSet = (set: CandidateSet, spelled: string): void => {
    if (candidateSet !== undefined) {
      throw new Error(`候補集合を 2 回指定している（${spelled}・先に ${candidateSet}）`);
    }
    candidateSet = set;
  };
  for (let at = 0; at < argv.length; at += 1) {
    const key = argv[at];
    if (key === "--quick") {
      chooseSet("quick", "--quick");
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
          throw new Error(`--case ${value} が形状表（src/tune/cases.ts）に無い`);
        }
        cases.push(value);
        break;
      case "--set":
        if (!isCandidateSet(value)) {
          throw new Error(`--set ${value} は ${CANDIDATE_SETS.join(" / ")} のどれでもない`);
        }
        chooseSet(value, `--set ${value}`);
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
    candidateSet: candidateSet ?? DEFAULT_CANDIDATE_SET,
    rounds,
    ...(out === undefined ? {} : { out }),
  };
};

/**
 * ファイル名に載せるアダプタの名前（tools/anima-residency/profile.ts と同じ規則 — architecture が
 * 空なら description を使う。Deno は vendor を PCI ID の 10 進で返し architecture を空にする）。
 */
const adapterSlug = (info: GeometrySweepAdapter): string => {
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

const unitLabel = (unit: GeometrySweepTimingUnit): string =>
  unit === "ns" ? "µs" : unit === "deno-raw-tick" ? "kTick" : "µs(wall)";

/** 赤で 1 行（Deno の console は `%c` の CSS color を端末の色へ写す）。 */
const printLine = (text: string, red = false): void => {
  if (red) console.log("%c%s", "color: red", text);
  else console.log(text);
};

/** 表の 1 行（幾何・1 dispatch・既定比・TFLOPS・digest の一致）。 */
const formatRow = (row: GeometrySweepRow): string => {
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
const formatCase = (
  summary: GeometrySweepCaseSummary,
): { readonly text: string; readonly red: boolean } => {
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
  let label = "";
  let current: string | undefined;
  const swept = await runGeometrySweep({
    candidateSet: flags.candidateSet,
    ops: flags.ops,
    ...(flags.cases.length === 0 ? {} : { cases: flags.cases }),
    rounds: flags.rounds,
    onProgress: (progress) => {
      switch (progress.kind) {
        case "started":
          label = unitLabel(progress.unit);
          console.log(
            `[geometry-sweep] ${
              progress.adapter.description || progress.adapter.vendor
            } · 単位 ${progress.unit} · ${flags.candidateSet} · rounds ${flags.rounds} · dp4a ${progress.dp4a} · ${progress.caseCount} ケース`,
          );
          break;
        case "status":
          console.error(`  … ${progress.message}`);
          break;
        case "row": {
          const { row } = progress;
          if (row.caseId !== current) {
            current = row.caseId;
            console.log(
              `\n${row.caseId}（${row.shape} · census ${row.censusCount} 本）\n    ${
                "geometry".padEnd(24)
              }${`${label}/disp`.padStart(10)}${"vs既定".padStart(7)}${"TFLOPS".padStart(7)}`,
            );
          }
          printLine(formatRow(row), row.error !== undefined || row.identicalToDefault === false);
          break;
        }
        case "case": {
          const { text, red } = formatCase(progress.summary);
          printLine(text, red);
          break;
        }
        case "deviceLost":
          break;
      }
    },
  });
  // CLI の記録の `userAgent` は Deno の版（`navigator.userAgent` の綴りではなく）— 既存の記録と同じ形
  const { revision, dirty } = await readCheckout();
  const report: GeometrySweepReport = {
    ...swept,
    userAgent: { deno: Deno.version.deno },
    checkout: revision,
    checkoutDirty: dirty,
  };
  const { rows, cases } = report;
  const stamp = report.date.replaceAll(":", "-");
  const path = flags.out ??
    `outputs/bench/karume/${localDate()}_geometry-sweep/geometry-sweep-${
      adapterSlug(report.adapter)
    }-${stamp}.json`;
  await Deno.mkdir(path.slice(0, Math.max(0, path.lastIndexOf("/"))) || ".", { recursive: true });
  await Deno.writeTextFile(path, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`\n[geometry-sweep] ${path}`);
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
