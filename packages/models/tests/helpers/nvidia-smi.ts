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
  /** GPU 全体の VRAM 使用量の最大（MiB）。 */
  readonly peakMemoryMiB: number;
  /** 温度の最大（℃）。 */
  readonly maxTemperatureC: number;
  /** クロックを下げた理由のビット列（`0x…` の綴りのまま・観測した全ての値の論理和）。 */
  readonly eventReasons: string;
  readonly samples: number;
};

/** {@link monitorNvidiaSmi} の決着（`nvidia-smi` が無い機では `undefined`）。 */
export type NvidiaTimeline = ReadonlyMap<string, NvidiaPhase> | undefined;

const QUERY = "memory.used,temperature.gpu,clocks_event_reasons.active";

/**
 * 標本化を始める。`enter(区間名)` で区間を切り替え、`stop()` で止めて区間ごとの最大を返す。区間は
 * 呼び手の時刻で切るので、切り替えの直後の標本が前の区間の山を持ち越すことがある（標本の間隔ぶん）。
 */
export const monitorNvidiaSmi = (intervalMs = 100): {
  readonly enter: (phase: string) => void;
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
      return { enter: () => {}, stop: () => Promise.resolve(undefined) };
    }
    throw cause;
  }
  let phase = "start";
  const phases = new Map<
    string,
    { peak: number; temperature: number; reasons: bigint; samples: number }
  >();
  const record = (line: string): void => {
    const [memory, temperature, reasons] = line.split(",").map((field) => field.trim());
    // 複数 GPU の機では行が GPU ごとに並び、ここは全 GPU の最大を取る（開発機は 1 枚 — 記録に書く前提）。
    if (memory === undefined || temperature === undefined || reasons === undefined) return;
    const current = phases.get(phase) ?? { peak: 0, temperature: 0, reasons: 0n, samples: 0 };
    current.peak = Math.max(current.peak, Number(memory));
    current.temperature = Math.max(current.temperature, Number(temperature));
    if (/^0x[0-9a-fA-F]+$/.test(reasons)) current.reasons |= BigInt(reasons);
    current.samples += 1;
    phases.set(phase, current);
  };
  const reading = (async () => {
    let pending = "";
    for await (const chunk of child.stdout.pipeThrough(new TextDecoderStream())) {
      pending += chunk;
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      for (const line of lines) if (line.trim() !== "") record(line);
    }
  })();
  return {
    enter: (next) => {
      phase = next;
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
      return new Map(
        [...phases].map(([name, value]) => [name, {
          peakMemoryMiB: value.peak,
          maxTemperatureC: value.temperature,
          eventReasons: `0x${value.reasons.toString(16).padStart(16, "0")}`,
          samples: value.samples,
        }]),
      );
    },
  };
};

/** 記録の 1 行（区間ごとの山・温度・理由）。観測なしはそう書く。 */
export const formatNvidiaTimeline = (timeline: NvidiaTimeline): string =>
  timeline === undefined
    ? "nvidia-smi なし（GPU 全体の VRAM は観測していない）"
    : [...timeline].map(([name, value]) =>
      `[${name}] VRAM 山 ${value.peakMemoryMiB} MiB・最高 ${value.maxTemperatureC} ℃・理由 ${value.eventReasons}`
    ).join(" / ");
