/**
 * 投機デコード（MTP・ADR 0096）の**実用上の取り分**を測る（段 4-A の計測 tool）。
 *
 *     deno run -A tools/mtp-bench/main.ts --workload extract --sampler greedy
 *     deno run -A tools/mtp-bench/main.ts --workload dialogue --sampler recommended --seed 42 \
 *         --new-tokens 200 --rounds 3 --out outputs/bench/karume-gemma4/<日付>_mtp/turns.jsonl
 *     deno run -A tools/mtp-bench/main.ts --workload freeform --sampler greedy --gpu-timing
 *     deno run -A tools/mtp-bench/main.ts --workload freeform --sampler greedy --warm
 *
 * 1 構成 = 1 プロセス（`tools/ram-peak/measure.ts` の流儀）。stdout は最後の **JSON 1 行**だけで、
 * 進捗はすべて stderr に出る。
 *
 * ## 何をどう測るか
 *
 * **同じプロセス・同じ pipeline・同じ prompt** で 3 つの構成を交互に回す — `plain`（非投機）/
 * `always`（常に投機）/ `auto`（自己採算ゲート付き = 既定の席）。別プロセスで測ると、ドライバの
 * clock 状態・PLE の常駐・WGSL の解析結果まで違う走行を比べることになり、差が投機のぶんなのか
 * 環境のぶんなのか分けられない。
 *
 * - **暖機 3 本**（各モード 1 本）— 立ち上げ（WGSL の解析・params の生成）を要約から外す。記録には
 *   残す（立ち上げの費用も後から読めるように）
 * - **ローテーション × `rounds`** — P S S P P A A P（S = always・A = auto）。順序効果（後のほうが
 *   速い / 遅い）を打ち消し、2 つの投機モードを同じ本数の plain で挟む
 * - **中央値** — 1 ターンの跳ね（PLE shard の読み直し・clock 変化）に引きずられない
 *
 * カーネルも生成ループも 1 行も変更しない。run 1 本の壁は `Session.prototype.run` を**この台本が**
 * 包んで採り、その run が何だったか（prefill / decode / draft / verify）は pipeline の観測席
 * `onRunDiagnostics` が同じ同期区間で名乗る（run の戻り直後に呼ばれる）ので、2 つを対にして積む。
 *
 * ## 読み方の注意
 *
 * - `--gpu-timing` を付けた走行の**壁は速度の数値として読めない**（計測が有効な device は
 *   1 dispatch = 1 pass に開く）。op 別の内訳だけを読み、倍率は付けない走行から採る。
 * - 投機は速度だけのノブなので、token 列は plain と一致するのが正しい（`summary.identity`）。
 *   食い違ったら倍率より先にそこを見る。ただし `auto` の `identicalAuto` だけは**不変条件では
 *   ない** — ゲートは壁時計で切るので、既定席の縮約順の違いで近い値の token の argmax が割れうる
 *   （`docs/limitations.md`）。`identical`（plain と always）が落ちたら本物の破れである。
 */

import { gemma4ChatPrompt, gemma4ChatTurn, Gemma4Pipeline } from "../../packages/models/gemma.ts";
import type {
  Gemma4ChatMessage,
  Gemma4RunPhase,
  GenerationSequence,
  GenerationSpeculation,
  GenerationStop,
  SamplerSpec,
  SpeculationGateOptions,
} from "../../packages/models/gemma.ts";
import {
  gemma4PleAssetSource,
  gemma4PleTotalBytes,
  readGemma4PleIndex,
} from "../../packages/models/src/gemma/ple-index.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import {
  loadManifest,
  MANIFEST_FILENAME,
  openContainerSource,
  parseManifest,
  resolveSelection,
} from "../../packages/hub/mod.ts";
import { acquireGpu, openContainer } from "../../packages/runtime/mod.ts";
import type { GpuTimingStats, SessionDiagnostics } from "../../packages/runtime/mod.ts";
// `mod.ts` の `Session` は型としてしか出ていない（構築の入口を絞る面 — ADR 0008）ので、
// prototype を包むための**値**は src から取る。計測の道具だけがここへ降りる。
import { Session } from "../../packages/runtime/src/runtime/executor.ts";
import { runMain } from "../../examples/shared/run-main.ts";
import {
  buildWorkload,
  isDocumentWorkload,
  warmFollowUps,
  WORKLOAD_NAMES,
  type WorkloadName,
} from "./workloads.ts";
import {
  assertWarmCapacity,
  assertWarmFollowUps,
  BENCH_MODES,
  type BenchMode,
  RUN_KINDS,
  type RunKind,
  summarizeTurns,
  tokensAfterFirst,
  tokensPerCycle,
  type TurnPlan,
  turnPlan,
  type TurnRecord,
  warmTurnPrefix,
} from "./summary.ts";
import { emptyTimingTallies, recordRunTiming, summarizeTiming } from "./timing.ts";
import { type TraceBucket, traceOf } from "./trace.ts";

const USAGE = "--source <配布形のパス> --workload <" + WORKLOAD_NAMES.join("|") + ">" +
  " --sampler <greedy|recommended> --seed <整数> --k <整数> --new-tokens <整数>" +
  " --capacity <整数> --document-chars <整数> --rounds <整数>" +
  " --max-resident-ple-bytes <整数> --gemv-rows-target <整数>" +
  " --gate-early-leave <数> --gate-burst-abort <数>" +
  " --gate-burst-min <整数> --gate-explore-base <整数>" +
  " --out <file.jsonl> --gpu-timing --warm";
const KNOWN = new Set([
  "source",
  "workload",
  "sampler",
  "seed",
  "k",
  "new-tokens",
  "capacity",
  "document-chars",
  "rounds",
  "max-resident-ple-bytes",
  "gemv-rows-target",
  "gate-early-leave",
  "gate-burst-abort",
  "gate-burst-min",
  "gate-explore-base",
  "out",
]);
/** 値を取らないスイッチ（`--key value` の対ではなく 1 語で立つ）。 */
const FLAGS = new Set(["gpu-timing", "warm"]);

/** 取得元の既定（`dist.py --pipeline gemma4` が組むローカルミラー — `docs/assets-layout.md`）。 */
const DEFAULT_SOURCE = "models/karume-gemma4";
/** 1 ターンで生成する token 数の上限。 */
const DEFAULT_NEW_TOKENS = 200;
/**
 * この会話が確保する KV の容量。
 *
 * 配布形の既定（4096）を上げてあるのは、長文脈のワークロード（≈4.8K token）がそこに入らないため。
 * 4 種を同じ容量で回すのは、容量が state スロットの物理確保量そのもので、ワークロードごとに
 * 変えると「容量が違う 2 つの走行」を比べることになるからである。
 */
const DEFAULT_CAPACITY = 8192;
/** 文書系ワークロードが文書を切る文字数の上限（段落境界で切る — `workloads.ts`）。 */
const DEFAULT_DOCUMENT_CHARS = 20000;
/** 暖機の後の ABBA の反復回数。 */
const DEFAULT_ROUNDS = 3;
/** `recommended` sampler の seed。 */
const DEFAULT_SEED = 42;

/**
 * `--key value` の対と、値を取らない {@link FLAGS} だけを受ける。
 *
 * MUST: 次のフラグを値として食わない。MUST: 未知のキーは落とす — 打ち間違えたノブが黙って既定で
 * 走ると、JSON 1 行に残る条件と実際に測った条件が食い違う（研究記録の 1 行が測定内容を偽る）。
 */
const args = new Map<string, string>();
const flags = new Set<string>();
for (let at = 0; at < Deno.args.length;) {
  const key = Deno.args[at];
  if (!key.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  const name = key.slice(2);
  if (FLAGS.has(name)) {
    flags.add(name);
    at += 1;
    continue;
  }
  if (!KNOWN.has(name)) throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
  const value = Deno.args[at + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  args.set(name, value);
  at += 2;
}

const integer = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw !== undefined && !/^\d+$/.test(raw)) throw new Error(`--${key} ${raw} が非負整数でない`);
  return raw === undefined ? undefined : Number(raw);
};

/**
 * 有限の実数を取るノブ（ゲートの比の閾値 — 整数に丸めると `earlyLeave 0.15` 級の指定が書けない）。
 */
const number = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (raw.trim() === "" || !Number.isFinite(value)) {
    throw new Error(`--${key} ${raw} が有限の数でない`);
  }
  return value;
};

const source = args.get("source") ?? DEFAULT_SOURCE;

const workloadArg = args.get("workload");
if (workloadArg === undefined) {
  throw new Error(`--workload <${WORKLOAD_NAMES.join("|")}> は必須（使い方: ${USAGE}）`);
}
const isWorkloadName = (name: string): name is WorkloadName =>
  (WORKLOAD_NAMES as readonly string[]).includes(name);
if (!isWorkloadName(workloadArg)) {
  throw new Error(`未知の --workload ${workloadArg}（既知: ${WORKLOAD_NAMES.join(" / ")}）`);
}
const workload: WorkloadName = workloadArg;

/**
 * 抽選の指定 — `greedy` は温度 0（低層の既定と同じ）、`recommended` は配布形の宣言 + seed。
 *
 * 必須なのは、投機の取り分が sampler で変わらない（温度に依らず列は一致する）ことと、**測った
 * 条件が出力に残ること**の両方が要るからである。省略を許すと「どちらで測ったか」が JSON からは
 * 分かるが打ち間違いは分からない、という形になる。
 */
const samplerArg = args.get("sampler");
if (samplerArg !== "greedy" && samplerArg !== "recommended") {
  throw new Error(`--sampler <greedy|recommended> は必須（受けた値: ${samplerArg ?? "無し"}）`);
}
const samplerName: "greedy" | "recommended" = samplerArg;
// 効かないノブは受けない（`--document-chars` と同じ原則）— 黙って無視すると、JSON 1 行に残る
// 条件を読んだ人が「seed を効かせて測った」と誤読する。
if (samplerName === "greedy" && args.has("seed")) {
  throw new Error("--sampler greedy に --seed は効かない（温度 0 は抽選しないので受けない）");
}

const seed = integer("seed") ?? DEFAULT_SEED;
/**
 * 1 cycle で引く draft の本数 — **既定を持たない**。
 *
 * 省略時は `speculative: {}` を渡してライブラリの既定（drafter グラフに焼かれた段数）で回る。
 * ここで 3 を捏造すると、配布形を焼き直して段数が変わった日に「台本が指定した k」と「焼かれた
 * 段数」が黙って食い違う。値域の門もライブラリ側（`assertSpeculative`）— 同じ門を 2 実装持たない。
 */
const kArg = integer("k");
const newTokens = integer("new-tokens") ?? DEFAULT_NEW_TOKENS;
const capacity = integer("capacity") ?? DEFAULT_CAPACITY;
const rounds = integer("rounds") ?? DEFAULT_ROUNDS;
const outPath = args.get("out");

/**
 * 文書を切る文字数の上限。文書系（{@link isDocumentWorkload}）以外に渡されたときの拒否は
 * `buildWorkload` が持つ（同じ門を 2 実装持たない）ので、ここでは既定を**文書系にだけ**入れる。
 */
const documentCharsArg = integer("document-chars");
const documentChars = isDocumentWorkload(workload)
  ? documentCharsArg ?? DEFAULT_DOCUMENT_CHARS
  : documentCharsArg;

/**
 * op 別 GPU 時間の内訳を採る（ADR 0021 — 既定は計測しない）。
 *
 * MUST NOT: 付けた走行の壁時計や倍率を速度の数値として読む（有効な device は 1 dispatch =
 * 1 pass に開くので壁が伸びる）。倍率は付けない走行から採る。
 */
const gpuTiming = flags.has("gpu-timing");

/**
 * モードごとに sequence を 1 本持ち、そのモードの各ターンで**違う** user 発話（`warmFollowUps`）を
 * 追記する（多ターン chat）。同じ発話を繰り返すと model が前の答えを写して受理率が跳ねる。
 *
 * 既定（cold）は 1 ターン = 1 sequence なので、`auto` の自己採算ゲートは**毎ターン初期状態から**
 * 始まり、負ける課題では「抜けるまでの探索」を毎ターン払う（悲観側）。実アプリ
 * （`Gemma4ChatSession`）は sequence を会話のあいだ使い回すので、ゲートが 1 度抜けた後の姿を
 * 見るにはこちらが要る。
 */
const warm = flags.has("warm");

const encoder = new TextEncoder();
const note = (text: string): void => {
  Deno.stderr.writeSync(encoder.encode(text));
};

/** cwd 基準のディレクトリ URL（末尾 `/` を必ず付ける — `new URL` の相対解決の前提）。 */
const directoryUrl = (path: string): URL =>
  new URL(path.endsWith("/") ? path : `${path}/`, `file://${Deno.cwd()}/`);

/** 与えられた PLE 常駐上限（省略時は manifest の索引から全量常駐を導く — {@link resolveAsset}）。 */
const maxResidentPleBytesArg = integer("max-resident-ple-bytes");

/**
 * 行ブロック gemv の並列度目標（runtime の `SessionOptions.linearGemvRowsThreadTarget` へ素通し）。
 *
 * 既定（16384 = 参照 device の飽和点）のままだと、飽和点が小さい GPU では verify（M=k+1）の
 * 本体 linear が `rows=1`（= M=1 カーネルを y に並べる形 = 重みを M 回読む）に落ちる。
 * 下げると `rows` が立って読み直しが減る。**静的**なノブなので、値は `config` に残す
 * （省略は `null` = 「与えていない」— 値域の門は runtime 側 1 箇所）。
 */
const gemvRowsTarget = integer("gemv-rows-target");

/**
 * 自己採算ゲートのノブ（`Gemma4PipelineOptions.speculative.gate` へ素通し）— **`auto` のモードに
 * だけ効く**（`always` はゲートを作らない席・`plain` は投機を張らない）。
 *
 * 部分指定を許すのは、A/B が動かすのが 1〜数本のノブだけであり、残りはライブラリの既定に
 * 従わせたいからである（既定値をここに写すと、ライブラリ側で既定が動いた日にこの台本だけ
 * 古い値で測る）。値域の門もライブラリ側 1 箇所（`createSpeculationGate`）— 同じ門を 2 実装
 * 持たない。指定した綴りは `config.gate` に残す（省略は `null` = 「与えていない」）。
 */
const gateEarlyLeave = number("gate-early-leave");
const gateBurstAbort = number("gate-burst-abort");
const gateBurstMin = integer("gate-burst-min");
const gateExploreBase = integer("gate-explore-base");
const gateKnobs: SpeculationGateOptions | undefined =
  gateEarlyLeave === undefined && gateBurstAbort === undefined && gateBurstMin === undefined &&
    gateExploreBase === undefined
    ? undefined
    : {
      ...(gateEarlyLeave === undefined ? {} : { earlyLeave: gateEarlyLeave }),
      ...(gateBurstAbort === undefined ? {} : { burstAbort: gateBurstAbort }),
      ...(gateBurstMin === undefined ? {} : { burstMin: gateBurstMin }),
      ...(gateExploreBase === undefined ? {} : { exploreBase: gateExploreBase }),
    };

/**
 * 何を測ったかの同定（`config.asset` — JSON 1 行だけで資産まで辿れるように）。
 *
 * 版を名乗るのが manifest **本文の SHA-256** なのは、配布形が版番号を持たないからである
 * （`generator` は焼いたツールの版で、資産の同一性ではない）。manifest は quant の選択・資産の
 * パス・整合値を全て抱えているので、この 1 値が一致すれば同じ資産で測ったと言える。
 */
type AssetIdentity = {
  readonly defaultModel: string;
  readonly defaultQuant: string;
  /** `karume.json` 本文（UTF-8）の SHA-256（hex 全桁）。 */
  readonly manifestSha256: string;
};

const sha256Hex = async (text: string): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(text));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
};

/**
 * PLE を**全量常駐**させるバイト数（ミラーの `model` 容器が持つ索引から導く）。
 *
 * 勘定は `packages/models/tests/helpers/ple-budget.ts` と同じ（容器の資産 `ple_index` →
 * block バイトの合計）。予算を定数で書かないのは、block 幅が資産世代で変わるため「N 本ぶん」が
 * 世代ごとに違う RAM を意味するからで（ADR 0085 追記）、計測では**読み直しゼロ**に固定したい —
 * 常駐が薄いと 1 ターンの間に block の読み直しが入り、その壁が生成相に混ざる。
 *
 * 読むのは part 0（descriptor）と索引の block だけで、重みの part には触らない。
 */
const allResidentPleBytes = async (mirror: URL): Promise<number> => {
  const loaded = await loadManifest(denoDirectory(mirror.pathname));
  const container = resolveSelection(loaded.manifest, { weights: ["model"] }).containers["model"];
  if (container === undefined) {
    throw new Error(`${mirror.href}: manifest に部品 'model' の容器が無い`);
  }
  const opened = await openContainer(
    { kind: "source", source: openContainerSource(loaded, container) },
    container.descriptor,
  );
  return gemma4PleTotalBytes(
    await readGemma4PleIndex(`mtp-bench ${mirror.href}`, gemma4PleAssetSource(opened)),
  );
};

/**
 * 配布形の manifest を **1 回だけ**読み、同定と PLE 予算を同じ本文から出す。
 *
 * 2 度読むと、同定に使った本文と予算を出した本文が別物になり得る（走行中に焼き直せば実際にそう
 * なる）。`--max-resident-ple-bytes` を与えた走行では索引を読まない — 予算が要らないのに索引の
 * 欠けで落ちるのは、測れる走行を測れなくするだけである。
 *
 * MUST: 呼ぶのは `main()` の中（トップレベルで読むと、壊れた `--source` の例外が `runMain` の
 * `printError` を通らず外皮しか画面に残らない）。
 */
const resolveAsset = async (
  mirror: URL,
): Promise<{ asset: AssetIdentity; maxResidentPleBytes: number }> => {
  const text = Deno.readTextFileSync(new URL(MANIFEST_FILENAME, mirror));
  const manifest = parseManifest(text);
  const entry = manifest.models[manifest.defaultModel];
  if (entry === undefined) {
    throw new Error(`${mirror.href}: manifest の defaultModel '${manifest.defaultModel}' が無い`);
  }
  return {
    asset: {
      defaultModel: manifest.defaultModel,
      defaultQuant: entry.defaultQuant,
      manifestSha256: await sha256Hex(text),
    },
    maxResidentPleBytes: maxResidentPleBytesArg ?? await allResidentPleBytes(mirror),
  };
};

/**
 * `--out` の親ディレクトリが**在ること**を確かめる（作らない）。
 *
 * 測り終えてから書けないと分かるのが最悪の順序なので、モデルを読む前に落とす。掘らないのは、
 * 置き場が規約で決まっている（`outputs/bench/<model>/<日付>_<用途>/` — `docs/assets-layout.md`）
 * ためで、打ち間違えた綴りを台本が作ると記録が散る。
 */
const assertOutParent = (path: string): void => {
  const cut = path.lastIndexOf("/");
  if (cut < 0) return;
  const parent = cut === 0 ? "/" : path.slice(0, cut);
  let stat: Deno.FileInfo;
  try {
    stat = Deno.statSync(parent);
  } catch (error) {
    throw new Error(`--out ${path} の親ディレクトリ ${parent} が無い（先に作ること）`, {
      cause: error,
    });
  }
  if (!stat.isDirectory) {
    throw new Error(`--out ${path} の親 ${parent} がディレクトリでない`);
  }
};

/** run 壁の器（1 ターンぶん — hook が積み、ターンの終わりに {@link TurnRecord} へ写す）。 */
type RunTally = { count: number; wallMs: number };
type RunTallies = { readonly [K in RunKind]: RunTally };
const emptyTallies = (): RunTallies => ({
  prefill: { count: 0, wallMs: 0 },
  decode: { count: 0, wallMs: 0 },
  draft: { count: 0, wallMs: 0 },
  verify: { count: 0, wallMs: 0 },
});

const secondsOf = (ms: number): string => (ms / 1000).toFixed(1);

/** 進捗行のモード 1 文字（P = plain・S = always〈常時投機〉・A = auto〈ゲート付き〉）。 */
const MODE_LABEL: { readonly [M in BenchMode]: string } = { plain: "P", always: "S", auto: "A" };

/**
 * 前の model turn を閉じる綴り（正本は `packages/models/src/gemma/text/chat.ts` の `END_OF_TURN`）。
 *
 * `--warm` のときだけ要る。多ターンの差分（`gemma4ChatTurn`）は「前 turn を閉じる `<turn|>` は
 * sequence の frontier が前置する」前提で描かれるが、`--new-tokens` で打ち切ったターンの
 * frontier は本文の token である（`Gemma4ChatSession` はその場合 KV を捨てて全体を描き直す）。
 * 台本は KV を継ぎたいので、閉じ札を**自分で 1 個前置して** model turn を閉じる — 生成が出した
 * ときと同じ id 列になる。綴りを写しているのは公開面に id の口が無いためで（`gemma4StopTokens`
 * は集合を返すだけ）、欠けていれば fail loudly する。
 */
const END_OF_TURN = "<turn|>";

/** そのモードで `Gemma4SequenceOptions.speculative` に渡す値（3 値の対応はここ 1 箇所）。 */
const speculativeOf = (mode: BenchMode): boolean | "always" =>
  mode === "plain" ? false : mode === "always" ? "always" : true;

/**
 * 台本の本体。
 *
 * MUST: `using` / `await using` は全てこの中に置く（トップレベルの `using` が畳んだ
 * `SuppressedError` は外皮しか印字されない — `examples/shared/run-main.ts` の doc）。
 */
const main = async (): Promise<void> => {
  // 書き出し先が無い走行は**測る前に**落とす（数分回してから書けないと分かるのが最悪の順序）。
  if (outPath !== undefined) assertOutParent(outPath);

  // 計測は Metal（Apple GPU）では device ごと落とす。拒否はしない — OS / ドライバ / wgpu 側が
  // 直れば黙って使えるようになる種類の制約なので、karume 側に撤去の宿題が残る門は置かない。
  if (gpuTiming && Deno.build.os === "darwin") {
    note(
      "[mtp-bench] 警告: macOS（Metal）では --gpu-timing の GPU 時間計測が device 消失を招く\n" +
        "            （timestamp 用の counter sample buffer を確保できず device lost になる）。\n" +
        "            内訳は Metal 以外のバックエンドで採ること — 詳細は docs/limitations.md の\n" +
        "            「Metal（Apple GPU）では GPU 側 timestamp 計測が実用にならない」節。\n",
    );
  }

  /**
   * device は**台本が持つ**（`--gpu-timing` の feature は device 作成時にしか要求できない）。
   * 貸した device の破棄は借り手ではなく貸し手の責務で、しかも **pipeline を畳んだ後**でなければ
   * ならない（flush-before-destroy）。`using` の解放は宣言の逆順なので、pipeline より前に
   * 宣言したこの口が最後に片付く。
   */
  const gpu = await acquireGpu(gpuTiming ? { gpuTiming: true } : {});
  using _gpuOwned = { [Symbol.dispose]: (): void => gpu.destroy() };

  /**
   * run 1 本の壁（発行 → 戻り）を採る包み。**production は 1 行も変えない**ための唯一の口で、
   * 先例は `outputs/bench/karume-gemma4/2026-09-07_k21-gemv-rows/turn-wall.ts`。
   *
   * 直近 1 本ぶんしか持たないのは、`onRunDiagnostics` が run の**戻った直後の同期区間**で呼ばれる
   * ため（その席で `phase.kind` と対にすれば取り違えが起きない）。包みは走行の間だけ効かせる
   * （`using` で必ず元へ戻す — グローバルの書き換えを台本の寿命より長く残さない）。
   */
  type SessionRun = Session["run"];
  const originalRun: SessionRun = Session.prototype.run;
  let lastRunWallMs = Number.NaN;
  Session.prototype.run = function (
    this: Session,
    ...runArgs: Parameters<SessionRun>
  ): ReturnType<SessionRun> {
    const started = performance.now();
    return originalRun.apply(this, runArgs).finally(() => {
      lastRunWallMs = performance.now() - started;
    });
  };
  using _runWall = {
    [Symbol.dispose]: (): void => {
      Session.prototype.run = originalRun;
    },
  };

  /** 今走っているターンの器（走行中だけ入る — 席が呼ばれたときに無ければ簿記の破れ）。 */
  let turnTallies: RunTallies | undefined;
  /**
   * 今走っているターンの観測 1 通ずつ（局面別の内訳 `trace.ts` の材料）。
   *
   * 器を分けてあるのは、run の**形**別の壁（`turnTallies`）は台本が採った壁で、局面別の内訳は
   * 生成面が名乗る壁（`phase.wallMs` — 配送の yield を挟まない値）だからである。
   */
  let turnPhases: Gemma4RunPhase[] | undefined;
  /**
   * 今走っているターンのモード（GPU 内訳を mode × kind に割る軸 — `timing.ts`）。
   *
   * run の形（`phase.kind`）だけでは足りない: `auto` のゲートが落とした plain step と `W1`
   * プローブは decode 形なので、`plain` モードの decode と同じ欄に落ちる。
   */
  let turnMode: BenchMode | undefined;
  /** 今走っているターンを GPU 内訳に数えるか（暖機は数えない）。 */
  let measured = false;
  const timing = emptyTimingTallies();

  const observeRun = (diagnostics: SessionDiagnostics, phase: Gemma4RunPhase): void => {
    const tallies = turnTallies;
    const phases = turnPhases;
    const mode = turnMode;
    if (tallies === undefined || phases === undefined || mode === undefined) {
      throw new Error(`[mtp-bench] ターンの外で ${phase.kind} run の観測が届いた`);
    }
    if (!Number.isFinite(lastRunWallMs)) {
      throw new Error(`[mtp-bench] ${phase.kind} run の壁が採れていない（包みが外れている）`);
    }
    const tally = tallies[phase.kind];
    tally.count += 1;
    tally.wallMs += lastRunWallMs;
    phases.push(phase);
    if (!gpuTiming) return;
    const stats: GpuTimingStats | undefined = diagnostics.lastRunTiming;
    if (stats === undefined) {
      // 計測を要求したのに内訳が無い = device が timing 無しで開かれている（黙って落とさない）。
      // 検査は**暖機の run でも**する — 積算に入らないだけで、非対応はその場で分かる。
      throw new Error("[mtp-bench] --gpu-timing を付けたが lastRunTiming が空（device が非対応）");
    }
    // 暖機を落とすのは `recordRunTiming` の中（積む器を選ぶ判断と同じ 1 箇所）。
    recordRunTiming(timing, { mode, kind: phase.kind, measured }, stats);
  };

  const { asset, maxResidentPleBytes } = await resolveAsset(directoryUrl(source));

  const started = performance.now();
  note(`[mtp-bench] ${source} を読み込む（drafter k=${kArg ?? "配布形の段数"}）\n`);
  await using pipeline = await Gemma4Pipeline.fromPretrained(denoDirectory(source), {
    gpu,
    // `k` を渡さない = 配布形の段数（空の `{}` が「drafter を組む」の綴りそのもの）。ゲートの
    // ノブも渡さなければライブラリの既定で回る（`auto` のターンにだけ降りる）。
    speculative: {
      ...(kArg === undefined ? {} : { k: kArg }),
      ...(gateKnobs === undefined ? {} : { gate: gateKnobs }),
    },
    ...(gemvRowsTarget === undefined ? {} : { linearGemvRowsThreadTarget: gemvRowsTarget }),
    maxResidentPleBytes,
    onRunDiagnostics: observeRun,
  });

  /**
   * 抽選の指定。低レベル面（`sequence`）は配布形を知らないので、渡さないと低層の既定（温度 0）で
   * 走る — `greedy` でも**明示して**渡すのは、JSON に残る条件と実際に効いた指定を一致させるため。
   */
  const defaultSampler = pipeline.defaultSampler;
  if (samplerName === "recommended" && defaultSampler === undefined) {
    throw new Error("--sampler recommended だが配布形が sampler を宣言していない");
  }
  const sampler: SamplerSpec = samplerName === "greedy"
    ? { temperature: 0 }
    : { ...defaultSampler, seed };

  const messages: readonly Gemma4ChatMessage[] = buildWorkload(workload, {
    ...(documentChars === undefined ? {} : { documentChars }),
  });
  const prompt = gemma4ChatPrompt(pipeline.tokenizer, messages);

  /**
   * warm の 2 本目以降が追記する差分（**ターンごとに違う** user 発話 — `warmFollowUps`）と
   * 閉じ札の id。自ターン `n` 本目（2 始まり）が流すのは `deltas[n - 2]` である。
   *
   * ターンごとに違う発話を流すのは、同じ発話を追記すると model が前のターンの答えを写し、
   * drafter の受理率が跳ねるためである（実測 1.63 → 3.9 tok/cycle・docs/research 2026-09-09
   * §6.5）— warm は「投機が負ける課題のまま、ゲートだけを暖める」ための口なので、写しが起きた
   * 走行は測りたいものを測っていない。理由の正本は `workloads.ts` の `warmFollowUps` の doc。
   *
   * 描くのは `gemma4ChatTurn` — 多ターンを自分で回すときの正本で、`Gemma4ChatSession` も同じ
   * 関数を同じ使い方で呼ぶ（`packages/models/src/gemma/chat-session.ts:518`）。テンプレート
   * 文字列を手で書かないのは、綴り（`<|turn>` 系）の所有者が chat 関数だからである。全部を
   * 起動時に描くのは、ターンの中で tokenizer を呼ぶと壁にその費用が乗るためである。
   * cold では `undefined`（追記も閉じ札も要らない）。
   */
  const warmTurn = ((): {
    readonly deltas: readonly (readonly number[])[];
    readonly longestDelta: number;
    readonly endOfTurnId: number;
  } | undefined => {
    if (!warm) return undefined;
    const endOfTurnId = pipeline.tokenizer.addedTokenId(END_OF_TURN);
    if (endOfTurnId === undefined) {
      throw new Error(`--warm: トークナイザの追加語彙に閉じ札 ${END_OF_TURN} が無い`);
    }
    const deltas = warmFollowUps(workload).map((content) =>
      gemma4ChatTurn(pipeline.tokenizer, { role: "user", content })
    );
    return {
      deltas,
      longestDelta: Math.max(...deltas.map((delta) => delta.length)),
      endOfTurnId,
    };
  })();

  const plans = turnPlan(rounds);
  if (warmTurn !== undefined) {
    // 発話列が尽きる走行も、容量に入らない走行も、**測る前に**落とす（走ってから尽きる／溢れると
    // 片側だけ短い走行の数字が残る）。追記は最長の 1 本で見て、閉じ札の前置ぶん +1 する
    // （どちらも悲観側）。
    assertWarmFollowUps({ plans, followUps: warmTurn.deltas.length });
    assertWarmCapacity({
      plans,
      capacity,
      promptTokens: prompt.length,
      turnTokens: warmTurn.longestDelta + 1,
      newTokens,
    });
  }
  note(
    `[mtp-bench] ready（${secondsOf(performance.now() - started)} s）` +
      ` / prompt ${prompt.length} token / capacity ${capacity}` +
      (warmTurn === undefined
        ? ""
        : ` / warm 追記 最長 ${warmTurn.longestDelta} token · ${warmTurn.deltas.length} 本`) +
      ` / sampler ${samplerName} ${JSON.stringify(sampler)}` +
      // ゲートのノブは auto のモードにだけ効く（既定のままなら「既定」と名乗る）。
      ` / gate ${gateKnobs === undefined ? "既定" : JSON.stringify(gateKnobs)}\n`,
  );

  /**
   * warm でモードごとに持つ sequence（cold では毎ターン作って畳むので空のまま）。
   *
   * 3 本が同時に生きるので KV の常駐は `capacity` の 3 倍になる。畳むのは全ターンの後で、
   * pipeline より先（`await using` の解放は宣言の逆順）。取りこぼしても
   * `Gemma4Pipeline.dispose` が巻き取る。
   */
  const heldSequences = new Map<BenchMode, GenerationSequence>();
  await using _heldOwned = {
    [Symbol.asyncDispose]: async (): Promise<void> => {
      for (const sequence of heldSequences.values()) await sequence.dispose();
    },
  };
  /** そのモードで何本目のターンか（1 始まり・暖機を 1 本目として数える）。 */
  const ownCounts = new Map<BenchMode, number>();
  /**
   * 直前のターンの停止（モードごと・warm だけが読む）。
   *
   * 次のターンに {@link END_OF_TURN} を前置するかはこの停止だけで決まる。判断そのものは
   * `summary.ts` の `warmTurnPrefix`（純関数）— 閉じ札以外の停止 token で閉じていたら落ちる。
   */
  const lastStops = new Map<BenchMode, GenerationStop>();

  /**
   * 1 ターンを回す。cold は 1 ターン = sequence 1 本（KV は使い回さない — 3 モードが同じ prompt を
   * 同じ位置から流す）。warm はモードごとに 1 本を使い回し、2 本目以降は差分だけを流す。
   *
   * `auto` のゲートは sequence と同じ寿命なので、cold では**毎ターン初期状態から**始まる（移動
   * 平均も探索の周期も持ち越さない）。実アプリの `Gemma4ChatSession` は sequence を使い回すので、
   * cold で出る `auto` の数字は悲観側である。
   */
  const runTurn = async (plan: TurnPlan, at: number): Promise<TurnRecord> => {
    const ownIndex = (ownCounts.get(plan.mode) ?? 0) + 1;
    ownCounts.set(plan.mode, ownIndex);
    const held = heldSequences.get(plan.mode);
    const sequence = held ?? await pipeline.sequence({
      speculative: speculativeOf(plan.mode),
      capacity,
    });
    if (warmTurn !== undefined && held === undefined) heldSequences.set(plan.mode, sequence);
    try {
      // 1 本目は会話全体（`gemma4ChatPrompt`）・2 本目以降は差分だけ。
      const turnPrompt = ((): readonly number[] => {
        if (held === undefined) return prompt;
        if (warmTurn === undefined) {
          throw new Error("[mtp-bench] 簿記の破れ: warm でないのに sequence を継いだ");
        }
        // 前ターンが閉じ札で終わっていれば frontier がそれを前置する（`gemma4ChatTurn` の前提）。
        // 打ち切ったターンの後は閉じ札が要り、閉じ札以外の停止 token で閉じていたら
        // `warmTurnPrefix` が落とす（README の `--warm` 節）。
        const prior = lastStops.get(plan.mode);
        if (prior === undefined) {
          throw new Error("[mtp-bench] 簿記の破れ: 継いだ sequence に前ターンの停止が無い");
        }
        // 自ターン 2 本目が発話列の 1 本目（起動時の `assertWarmFollowUps` が尽きないことを見た）。
        // 添字アクセス（`at` は負の添字で末尾へ回り込み、簿記の破れが黙る）。
        const delta = warmTurn.deltas[ownIndex - 2];
        if (delta === undefined) {
          throw new Error(
            `[mtp-bench] 簿記の破れ: 自ターン ${ownIndex} 本目に対応する追記が無い` +
              `（追記 ${warmTurn.deltas.length} 本）`,
          );
        }
        return [
          ...warmTurnPrefix({ prior, mode: plan.mode, endOfTurnId: warmTurn.endOfTurnId }),
          ...delta,
        ];
      })();
      // 生成の**前**の占有（warm ではここが前ターンまでの積み上がり）。
      const contextTokens = sequence.used;
      const tallies = emptyTallies();
      const phases: Gemma4RunPhase[] = [];
      turnTallies = tallies;
      turnPhases = phases;
      turnMode = plan.mode;
      measured = !plan.warmup;
      const ids: number[] = [];
      let firstTokenMs = Number.NaN;
      const turnStarted = performance.now();
      const stream = sequence.generate({ prompt: turnPrompt, maxNewTokens: newTokens, sampler });
      for await (const event of stream) {
        if (event.kind !== "token") continue;
        if (ids.length === 0) firstTokenMs = performance.now() - turnStarted;
        ids.push(event.id);
      }
      const stop = await stream.done;
      const turnMs = performance.now() - turnStarted;
      if (!Number.isFinite(firstTokenMs)) {
        throw new Error(`[mtp-bench] turn ${at}: token イベントが 1 通も無い（${stop.reason}）`);
      }
      // 次のターンが差分をどう流せるかは、この停止が model turn を閉じたかで決まる。
      if (warmTurn !== undefined) lastStops.set(plan.mode, stop);
      const speculation: GenerationSpeculation | undefined = stop.speculation;
      const record: TurnRecord = {
        ...plan,
        ownIndex,
        contextTokens,
        tokens: stop.tokens,
        stopReason: stop.reason,
        ids,
        text: pipeline.tokenizer.decode(ids),
        turnMs,
        firstTokenMs,
        generationMs: turnMs - firstTokenMs,
        runs: {
          prefill: { ...tallies.prefill },
          decode: { ...tallies.decode },
          draft: { ...tallies.draft },
          verify: { ...tallies.verify },
        },
        trace: traceOf(phases),
        ...(speculation === undefined ? {} : { speculation }),
      };
      // 分母は要約と同じ 1 本（`summary.ts`）— 画面の数字と JSON の数字を別実装にしない。
      const perToken = record.generationMs / tokensAfterFirst(record);
      const perCycle = speculation === undefined
        ? ""
        : ` · ${tokensPerCycle(speculation).toFixed(2)} tok/cycle`;
      // ゲートの働きは `auto` のターンにしか無い（`always` の勘定には欄ごと無い）。欠けた欄を 0 と
      // 書かないのは、簿記の破れを落とす口が要約側（`summary.ts`）の 1 箇所だからである。
      const gate = speculation?.plainSteps === undefined || speculation.switches === undefined
        ? ""
        : ` · plain steps ${speculation.plainSteps} · switches ${speculation.switches}`;
      // warm では同じモードの前のターンからの積み上がりが読めないと数字が解釈できない
      // （後ろのターンほど context が長い）ので、ターン行に生成前の占有を出す。
      const context = warm ? ` · #${ownIndex} ctx ${record.contextTokens}` : "";
      note(
        `[mtp-bench] ${workload}/${samplerName} turn ${at} ` +
          `${MODE_LABEL[plan.mode]}${plan.warmup ? " warmup" : ""}: ` +
          `${record.tokens} tok · gen ${secondsOf(record.generationMs)} s · ` +
          `${perToken.toFixed(1)} ms/tok${perCycle}${gate}${context}\n`,
      );
      return record;
    } finally {
      turnTallies = undefined;
      turnPhases = undefined;
      turnMode = undefined;
      measured = false;
      // warm の sequence は次のターンが継ぐので畳まない（全ターンの後に `_heldOwned` が畳む）。
      if (!warm) await sequence.dispose();
    }
  };

  const turns: TurnRecord[] = [];
  for (const [at, plan] of plans.entries()) {
    turns.push(await runTurn(plan, at + 1));
  }
  const summary = summarizeTurns(turns, { warm });

  /** mode × kind の GPU 内訳（`--gpu-timing` のときだけ・暖機を除く — `timing.ts`）。 */
  const gpuBreakdown = gpuTiming ? summarizeTiming(timing) : undefined;

  const { vendor, architecture, device, description } = gpu.adapterInfo;
  const line = JSON.stringify({
    tool: "mtp-bench",
    host: {
      os: Deno.build.os,
      arch: Deno.build.arch,
      deno: Deno.version.deno,
      adapter: { vendor, architecture, device, description },
    },
    // 測定条件（JSON 1 行から構成が復元できることがこの出力の名乗り）。効いていないノブは
    // null を書いて「与えていない」と読めるようにする（`ram-peak` と同じ体裁）。
    config: {
      source,
      asset,
      workload,
      sampler: samplerName,
      seed: samplerName === "recommended" ? seed : null,
      k: kArg ?? null,
      newTokens,
      capacity,
      documentChars: documentChars ?? null,
      rounds,
      warm,
      maxResidentPleBytes,
      gemvRowsTarget: gemvRowsTarget ?? null,
      // 与えたゲートのノブだけ（`auto` のモードにだけ効く — 既定のままなら null）。
      gate: gateKnobs ?? null,
      gpuTiming,
      out: outPath ?? null,
    },
    prompt: { tokens: prompt.length, messages },
    turns,
    summary,
    ...(gpuBreakdown === undefined ? {} : { gpu: gpuBreakdown }),
  });
  console.log(line);
  if (outPath !== undefined) await Deno.writeTextFile(outPath, `${line}\n`, { append: true });
  note(
    `[mtp-bench] plain ${summary.plain.msPerToken.toFixed(2)} ms/tok` +
      ` / always ${summary.always.msPerToken.toFixed(2)} ms/tok` +
      ` = ${summary.speedup.toFixed(3)}×` +
      ` / auto ${summary.auto.msPerToken.toFixed(2)} ms/tok` +
      `（${summary.speedupAuto.toFixed(3)}×）` +
      ` · 列一致 always ${summary.identity.identical ? "yes" : "NO"}` +
      ` / auto ${summary.identity.identicalAuto ? "yes" : "NO"}` +
      // 実効 k は勘定から出た値（`--k` 省略時は配布形の段数がそのまま出る）。
      ` · 実効 k ${summary.always.k ?? "不明"}` +
      // warm は 3 モードに揃う自ターン番号までしか要約に入れない（`summary.ts`）— 何本で
      // 出した数字かが分からないと、上の倍率がどの範囲の話か読めない。
      (summary.ownTurnLimit === undefined
        ? ""
        : ` · warm（要約は自ターン ≤ ${summary.ownTurnLimit}・` +
          `${summary.plain.turns} / ${summary.always.turns} / ${summary.auto.turns} 本）` +
          // 同じ自ターン番号で 3 モードの context 長が揃っていたか。揃わない番号が出た後は
          // 「同じ位置から始まったターン」の比較でなくなるので、倍率の読み方が変わる。
          ` · ctx aligned ${
            summary.identity.firstContextMismatchTurn === undefined
              ? "yes"
              : `NO@${summary.identity.firstContextMismatchTurn}`
          }`) +
      "\n",
  );
  // ゲート付きのターンの時間がどの局面に落ちたか（中央値 1 ターンぶん — 正本は JSON の
  // `summary.auto.trace`）。倍率だけでは「負けを止めた費用」がどこに乗ったか読めない。
  const trace = summary.auto.trace;
  const bucket = (label: string, one: TraceBucket): string =>
    `${label} ${one.runs} run ${secondsOf(one.ms)} s`;
  note(
    "[mtp-bench] auto 内訳: " +
      [
        bucket("投機", trace.speculate),
        bucket("復帰投機", trace.speculateAfterReturn),
        bucket("burst", trace.burst),
        bucket("W1", trace.w1Probe),
        bucket("plain", trace.plain),
        bucket("未観測", trace.unmeasured),
      ].join(" / ") +
      (trace.firstExitRun === undefined ? "" : ` · 初回離脱 run ${trace.firstExitRun}`) +
      "\n",
  );
  // GPU 内訳の見出しだけ画面に出す（op 別の表は行数が多いので正本は JSON の `gpu`）。mode を
  // ラベルに含めるのは、`auto/decode`（ゲートの plain step）と `plain/decode` が別の欄だと
  // 画面で分かる形にするためである。
  if (gpuBreakdown !== undefined) {
    note("[mtp-bench] GPU 内訳（暖機を除く・op 別は JSON の gpu）:\n");
    for (const mode of BENCH_MODES) {
      const byKind = gpuBreakdown[mode];
      if (byKind === undefined) continue;
      for (const kind of RUN_KINDS) {
        const one = byKind[kind];
        if (one === undefined) continue;
        note(
          `[mtp-bench]   ${mode}/${kind} ${one.runs} run · ` +
            `${one.msPerRun.toFixed(2)} ms/run · ` +
            `${one.dispatchesPerRun.toFixed(1)} dispatch/run\n`,
        );
      }
    }
  }
};

await runMain(main);
