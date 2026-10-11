/**
 * GPU 全体の VRAM 使用量・温度・クロックを下げた理由（`clocks_event_reasons.active`）の標本化（診断 — NVIDIA の
 * `nvidia-smi` がある機でだけ）。
 *
 * fdinfo（{@link "./drm-usage.ts"}）がこの process の確保を見るのに対し、ここが見るのは **GPU 全体**（画面や他の
 * process の確保を含む）で、ADR 0121 の「DiT の段の VRAM の山 11,361 MiB（nvidia-smi・GPU 全体）」と同じ測り方。
 * 温度とクロックを下げた理由は、壁時計の比較が熱制限で崩れていないかを読む材料（93 ℃・`sw_thermal` の前例 —
 * ADR 0121）。`nvidia-smi` が無い機は `undefined`（観測なし）を返し、呼び手はそれを記録に書く。
 * `-lms` の標本なので短い山は取りこぼしうる（門ではなく観測に使う）。
 */

/** 区間 1 つぶんの観測。 */
export type NvidiaPhase = {
  /** GPU 全体の VRAM 使用量の最大（MiB）。標本 0 の区間では 0（`samples` で見分ける）。 */
  readonly peakMemoryMiB: number;
  /** 温度の最大（℃）。標本 0 の区間では 0。 */
  readonly maxTemperatureC: number;
  /** クロックを下げた理由のビット列（`0x…` の綴りのまま・観測した全ての値の論理和）。 */
  readonly eventReasons: string;
  readonly samples: number;
};

/** {@link monitorNvidiaSmi} の決着（`nvidia-smi` が無い機では `undefined`）。 */
export type NvidiaTimeline =
  | {
    /** 区間ごとの観測（`enter` した区間は標本 0 でも載る）。 */
    readonly phases: ReadonlyMap<string, NvidiaPhase>;
    /** 標本として読めなかった行（古いドライバのエラー行など — {@link parseNvidiaSample}）。記録だけで、門にはしない。 */
    readonly errorLines: readonly string[];
  }
  | undefined;

const QUERY = "memory.used,temperature.gpu,clocks_event_reasons.active";

/** `nvidia-smi --query-gpu=<QUERY> --format=csv,noheader,nounits` の 1 行を読んだ値。 */
export type NvidiaSample = {
  readonly memoryMiB: number;
  readonly temperatureC: number;
  /** `0x…` の綴りでない値（`[Not Supported]` など）は `undefined`。 */
  readonly reasons: bigint | undefined;
};

/**
 * 1 行を標本として読む。カンマで 3 欄に割れない行と、使用量・温度が有限の数でない行は `undefined`（エラー行 —
 * 呼び手が記録に残す）。黙って捨てると、古いドライバの機で「標本 0」と「読めない行が来ていた」を区別できない。
 * 数でない欄をそのまま読むと、最大値が NaN に化ける。
 */
export const parseNvidiaSample = (line: string): NvidiaSample | undefined => {
  const fields = line.split(",").map((field) => field.trim());
  if (fields.length !== 3) return undefined;
  const [memory, temperature, reasons] = fields;
  // `Number("")` は 0 になるので、空の欄は数として読まない。
  if (memory === "" || temperature === "") return undefined;
  const memoryMiB = Number(memory);
  const temperatureC = Number(temperature);
  if (!Number.isFinite(memoryMiB) || !Number.isFinite(temperatureC)) return undefined;
  return {
    memoryMiB,
    temperatureC,
    reasons: /^0x[0-9a-fA-F]+$/.test(reasons) ? BigInt(reasons) : undefined,
  };
};

/** 標本を待つ間隔（ms）。 */
const WAIT_STEP_MS = 50;

/**
 * 標本化を始める。`enter(区間名)` で区間を切り替え、`stop()` で止めて区間ごとの最大を返す。区間は
 * 呼び手の時刻で切るので、切り替えの直後の標本が前の区間の山を持ち越すことがある（標本の間隔ぶん）。
 * `waitForSample(上限 ms)` は今の区間に標本が 1 本来るまで上限つきで待ち、来た本数を返す（`nvidia-smi` の
 * 起動は標本の間隔より遅いことがあり、短い区間は待たないと標本 0 になる）。
 */
export const monitorNvidiaSmi = (intervalMs = 100): {
  readonly enter: (phase: string) => void;
  readonly waitForSample: (timeoutMs: number) => Promise<number>;
  readonly stop: () => Promise<NvidiaTimeline>;
} => {
  let child: Deno.ChildProcess;
  try {
    child = new Deno.Command("nvidia-smi", {
      args: [`--query-gpu=${QUERY}`, "--format=csv,noheader,nounits", "-lms", String(intervalMs)],
      stdout: "piped",
      stderr: "null",
    }).spawn();
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) {
      return {
        enter: () => {},
        waitForSample: () => Promise.resolve(0),
        stop: () => Promise.resolve(undefined),
      };
    }
    throw cause;
  }
  type Accumulated = { peak: number; temperature: number; reasons: bigint; samples: number };
  const empty = (): Accumulated => ({ peak: 0, temperature: 0, reasons: 0n, samples: 0 });
  let phase = "start";
  const phases = new Map<string, Accumulated>([[phase, empty()]]);
  const errorLines: string[] = [];
  let finished = false;
  const record = (line: string): void => {
    const sample = parseNvidiaSample(line);
    if (sample === undefined) {
      errorLines.push(line.trim());
      return;
    }
    // 複数 GPU の機では行が GPU ごとに並び、ここは全 GPU の最大を取る（開発機は 1 枚 — 記録に書く前提）。
    const current = phases.get(phase) ?? empty();
    current.peak = Math.max(current.peak, sample.memoryMiB);
    current.temperature = Math.max(current.temperature, sample.temperatureC);
    if (sample.reasons !== undefined) current.reasons |= sample.reasons;
    current.samples += 1;
    phases.set(phase, current);
  };
  const reading = (async () => {
    try {
      let pending = "";
      for await (const chunk of child.stdout.pipeThrough(new TextDecoderStream())) {
        pending += chunk;
        const lines = pending.split("\n");
        pending = lines.pop() ?? "";
        for (const line of lines) if (line.trim() !== "") record(line);
      }
      if (pending.trim() !== "") record(pending);
    } finally {
      finished = true;
    }
  })();
  return {
    enter: (next) => {
      phase = next;
      if (!phases.has(next)) phases.set(next, empty());
    },
    waitForSample: async (timeoutMs) => {
      const deadline = performance.now() + timeoutMs;
      const samplesNow = (): number => phases.get(phase)?.samples ?? 0;
      // 子が先に終わった（nvidia-smi が自分で落ちた）なら、待っても標本は来ない。
      while (samplesNow() === 0 && !finished && performance.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, WAIT_STEP_MS));
      }
      return samplesNow();
    },
    stop: async () => {
      try {
        child.kill("SIGTERM");
      } catch (cause) {
        // 既に終わった子（nvidia-smi が自分で落ちた）は止める必要が無い。それ以外は伝える。
        if (!(cause instanceof TypeError)) throw cause;
      }
      await reading;
      await child.status;
      return {
        phases: new Map(
          [...phases].map(([name, value]) => [name, {
            peakMemoryMiB: value.peak,
            maxTemperatureC: value.temperature,
            eventReasons: `0x${value.reasons.toString(16).padStart(16, "0")}`,
            samples: value.samples,
          }]),
        ),
        errorLines: [...errorLines],
      };
    },
  };
};

/**
 * 記録の 1 行（区間ごとの山・温度・理由と、読めなかった行）。観測なしはそう書き、標本 0 の区間は「標本 0」と
 * 明示する（山 0 MiB と読ませない）。
 */
export const formatNvidiaTimeline = (timeline: NvidiaTimeline): string => {
  if (timeline === undefined) return "nvidia-smi なし（GPU 全体の VRAM は観測していない）";
  const phases = [...timeline.phases]
    // 区間を切る前の既定の区間（start）は、標本が無ければ載せない（呼び手が切った区間だけを記録する）。
    .filter(([name, value]) => name !== "start" || value.samples > 0)
    .map(([name, value]) =>
      value.samples === 0
        ? `[${name}] 標本 0（観測なし）`
        : `[${name}] VRAM 山 ${value.peakMemoryMiB} MiB・最高 ${value.maxTemperatureC} ℃・理由 ${value.eventReasons}`
    );
  // 同じエラー行は標本の間隔ごとに繰り返すので、本数と異なる行の綴りを残す。
  const errors = timeline.errorLines.length === 0 ? [] : [
    `読めない行 ${timeline.errorLines.length} 本（${
      [...new Set(timeline.errorLines)].join(" | ")
    }）`,
  ];
  return [...phases, ...errors].join(" / ");
};
