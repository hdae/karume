import { spawn } from "node:child_process";
import { isNotFound } from "@karume/test-support/fs";
/**
 * Samples the whole GPU's VRAM usage, temperature and clock-throttle reasons (`clocks_event_reasons.active`)
 * (diagnostics — only on machines with NVIDIA's `nvidia-smi`).
 *
 * Where fdinfo ({@link "./drm-usage.ts"}) sees this process's allocations, this sees **the whole GPU** (including the
 * display's and other processes' allocations) — the same measurement as ADR 0121's "VRAM peak of the DiT stage,
 * 11,361 MiB (nvidia-smi, whole GPU)". The temperature and throttle reasons are material for reading whether a
 * wall-clock comparison was skewed by thermal limits (the precedent of 93 °C and `sw_thermal` — ADR 0121). A machine
 * without `nvidia-smi` returns `undefined` (not observed), and the caller writes that into the record. The samples come
 * from `-lms`, so short peaks can be missed (use it for observation, not as a check).
 */

/** The observation of one phase. */
export type NvidiaPhase = {
  /** The maximum VRAM usage of the whole GPU (MiB). 0 for a phase with zero samples (tell them apart by `samples`). */
  readonly peakMemoryMiB: number;
  /** The maximum temperature (°C). 0 for a phase with zero samples. */
  readonly maxTemperatureC: number;
  /** The bit set of clock-throttle reasons (in the `0x…` spelling; the OR of every observed value). */
  readonly eventReasons: string;
  readonly samples: number;
};

/** The outcome of {@link monitorNvidiaSmi} (`undefined` on a machine without `nvidia-smi`). */
export type NvidiaTimeline =
  | {
      /** The observation per phase (a phase that was `enter`ed is listed even with zero samples). */
      readonly phases: ReadonlyMap<string, NvidiaPhase>;
      /**
       * Lines that could not be read as samples (error lines of old drivers and the like — {@link parseNvidiaSample}).
       * Recorded only, not checked.
       */
      readonly errorLines: readonly string[];
    }
  | undefined;

const QUERY = "memory.used,temperature.gpu,clocks_event_reasons.active";

/** The values read from one line of `nvidia-smi --query-gpu=<QUERY> --format=csv,noheader,nounits`. */
export type NvidiaSample = {
  readonly memoryMiB: number;
  readonly temperatureC: number;
  /** `undefined` for a value not spelled `0x…` (`[Not Supported]` and the like). */
  readonly reasons: bigint | undefined;
};

/**
 * Reads one line as a sample. A line that does not split into 3 fields at the commas, or whose usage or temperature is
 * not a finite number, is `undefined` (an error line — the caller keeps it in the record). Dropping it silently would
 * make "zero samples" indistinguishable from "unreadable lines were arriving" on machines with old drivers. Reading a
 * non-numeric field as is would turn the maximum into NaN.
 */
export const parseNvidiaSample = (line: string): NvidiaSample | undefined => {
  const fields = line.split(",").map((field) => field.trim());
  if (fields.length !== 3) return undefined;
  const [memory, temperature, reasons] = fields;
  // `Number("")` is 0, so an empty field is not read as a number.
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

/** The interval between samples (ms). */
const WAIT_STEP_MS = 50;

/**
 * Starts sampling. `enter(phase name)` switches the phase, and `stop()` stops and returns the maximum per phase. Phases
 * are cut at the caller's time, so a sample right after a switch can carry over the previous phase's peak (by up to one
 * sampling interval). `waitForSample(limit ms)` waits, up to the limit, until one sample of the current phase arrives
 * and returns how many arrived (`nvidia-smi` can start slower than the sampling interval, and a short phase has zero
 * samples unless it waits).
 */
export const monitorNvidiaSmi = (
  intervalMs = 100,
): {
  readonly enter: (phase: string) => void;
  readonly waitForSample: (timeoutMs: number) => Promise<number>;
  readonly stop: () => Promise<NvidiaTimeline>;
} => {
  type Accumulated = { peak: number; temperature: number; reasons: bigint; samples: number };
  const empty = (): Accumulated => ({ peak: 0, temperature: 0, reasons: 0n, samples: 0 });
  let phase = "start";
  const phases = new Map<string, Accumulated>([[phase, empty()]]);
  const errorLines: string[] = [];
  let finished = false;
  const child = spawn(
    "nvidia-smi",
    [`--query-gpu=${QUERY}`, "--format=csv,noheader,nounits", "-lms", String(intervalMs)],
    { stdio: ["ignore", "pipe", "ignore"] },
  );
  // Node reports a missing binary asynchronously (`error` with ENOENT). Without nvidia-smi the
  // monitor observes nothing — the same outcome as Deno's synchronous NotFound path.
  let missing = false;
  const exited = new Promise<void>((resolve) => {
    child.once("exit", () => resolve());
    child.once("error", (cause) => {
      if (isNotFound(cause)) missing = true;
      else errorLines.push(String(cause));
      resolve();
    });
  });
  const record = (line: string): void => {
    const sample = parseNvidiaSample(line);
    if (sample === undefined) {
      errorLines.push(line.trim());
      return;
    }
    // On a multi-GPU machine the lines come one per GPU, and this takes the maximum over all GPUs (the development
    // machine has one card — an assumption written into the record).
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
      for await (const chunk of child.stdout.setEncoding("utf8")) {
        pending += String(chunk);
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
      // If the child ended first (nvidia-smi exited on its own), no sample will come however long we wait.
      while (samplesNow() === 0 && !finished && performance.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, WAIT_STEP_MS));
      }
      return samplesNow();
    },
    stop: async () => {
      child.kill("SIGTERM"); // false when the child already exited — nothing to stop
      await reading;
      await exited;
      if (missing) return undefined;
      return {
        phases: new Map(
          [...phases].map(([name, value]) => [
            name,
            {
              peakMemoryMiB: value.peak,
              maxTemperatureC: value.temperature,
              eventReasons: `0x${value.reasons.toString(16).padStart(16, "0")}`,
              samples: value.samples,
            },
          ]),
        ),
        errorLines: [...errorLines],
      };
    },
  };
};

/**
 * One line of the record (the peak, temperature and reasons per phase, plus the unreadable lines). No observation is
 * written as such, and a phase with zero samples says "zero samples" explicitly (so it is not read as a 0 MiB peak).
 */
export const formatNvidiaTimeline = (timeline: NvidiaTimeline): string => {
  if (timeline === undefined) return "no nvidia-smi (GPU-wide VRAM not observed)";
  const phases = [...timeline.phases]
    // The default phase before any cut (start) is listed only when it has samples (only the phases the caller cut are
    // recorded).
    .filter(([name, value]) => name !== "start" || value.samples > 0)
    .map(([name, value]) =>
      value.samples === 0
        ? `[${name}] 0 samples (not observed)`
        : `[${name}] VRAM peak ${value.peakMemoryMiB} MiB, max ${value.maxTemperatureC} ℃, reasons ${value.eventReasons}`,
    );
  // The same error line repeats every sampling interval, so the count and the distinct spellings are kept.
  const errors =
    timeline.errorLines.length === 0
      ? []
      : [
          `${timeline.errorLines.length} unreadable lines (${[...new Set(timeline.errorLines)].join(
            " | ",
          )})`,
        ];
  return [...phases, ...errors].join(" / ");
};
