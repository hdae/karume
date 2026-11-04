import {
  type DirEntry,
  isNotFound,
  readDirSync,
  readLinkSync,
  readTextFileSync,
} from "@karume/test-support/fs";
/**
 * Sampling of this process's GPU memory usage (diagnostics — Linux DRM fdinfo).
 *
 * Sums `drm-total-<region>` (vram0 / gtt / system …) of `/proc/self/fdinfo/<renderD fd>` per
 * region. xe also reports engine cycle counts under the same prefix (`drm-total-cycles-<engine>`);
 * those are not counted. An environment without Linux DRM (no fdinfo) returns `undefined`, which
 * callers treat as "no observation". Samples are taken every 10 ms, so short peaks can be missed
 * (use it for observation, not as a check).
 */

const RENDER_NODE = /^\/dev\/dri\/renderD\d+$/;
const FDINFO_UNITS: Readonly<Record<string, number>> = { "": 1, KiB: 1024, MiB: 1024 ** 2 };

/** Region name (`vram0` / `gtt` / `system` …) → bytes. */
export type DrmUsage = ReadonlyMap<string, number>;

/**
 * The per-region sum of `drm-total-<region>` over this process's DRM clients. File descriptors that
 * point to the same client are counted once, by `drm-client-id`.
 */
export const sampleDrmUsage = (): DrmUsage | undefined => {
  let fds: DirEntry[];
  try {
    fds = [...readDirSync("/proc/self/fd")];
  } catch (cause) {
    if (isNotFound(cause)) return undefined;
    throw cause;
  }
  const clients = new Map<string, DrmUsage>();
  for (const { name } of fds) {
    let info: string;
    try {
      if (!RENDER_NODE.test(readLinkSync(`/proc/self/fd/${name}`))) continue;
      info = readTextFileSync(`/proc/self/fdinfo/${name}`);
    } catch (cause) {
      // An fd closed between listing and reading (such as the sampler's own readDir fd) is not
      // counted.
      if (isNotFound(cause)) continue;
      throw cause;
    }
    const client = /^drm-client-id:\s*(\d+)/m.exec(info)?.[1];
    if (client === undefined) continue;
    const regions = new Map<string, number>();
    for (const [, region, value, unit] of info.matchAll(
      /^drm-total-(\S+):\s*(\d+)\s*(KiB|MiB)?/gm,
    )) {
      if (region.startsWith("cycles-")) continue;
      regions.set(region, Number(value) * FDINFO_UNITS[unit ?? ""]);
    }
    clients.set(client, regions);
  }
  if (clients.size === 0) return undefined;
  const total = new Map<string, number>();
  for (const regions of clients.values()) {
    for (const [region, bytes] of regions) total.set(region, (total.get(region) ?? 0) + bytes);
  }
  return total;
};

/** The result of {@link monitorDrmUsage}. */
export type DrmTimeline = {
  /**
   * Phase name → the maximum per region (including the samples taken on entering and leaving it).
   */
  readonly peaks: ReadonlyMap<string, DrmUsage>;
  /** The samples at the points the caller named (in `mark` order). */
  readonly marks: readonly { readonly label: string; readonly usage: DrmUsage | undefined }[];
};

/**
 * Starts sampling that records the peak of each phase. `enter(phase)` switches the phase (taking
 * one sample just before and one just after the switch), `mark(label)` keeps the value at that
 * point, and `stop()` stops sampling and returns the result.
 */
export const monitorDrmUsage = (
  intervalMs = 10,
): {
  readonly enter: (phase: string) => void;
  readonly mark: (label: string) => void;
  readonly stop: () => DrmTimeline;
} => {
  let phase = "start";
  const peaks = new Map<string, Map<string, number>>();
  const marks: { readonly label: string; readonly usage: DrmUsage | undefined }[] = [];
  const sample = (): void => {
    const usage = sampleDrmUsage();
    if (usage === undefined) return;
    const peak = peaks.get(phase) ?? new Map<string, number>();
    for (const [region, bytes] of usage) peak.set(region, Math.max(peak.get(region) ?? 0, bytes));
    peaks.set(phase, peak);
  };
  sample();
  const timer = setInterval(sample, intervalMs);
  return {
    enter: (next) => {
      sample();
      phase = next;
      sample();
    },
    mark: (label) => marks.push({ label, usage: sampleDrmUsage() }),
    stop: () => {
      sample();
      clearInterval(timer);
      return { peaks, marks };
    },
  };
};

const GIB = 1024 ** 3;

/** `vram0=1.234 gtt=0.010 …` (GiB, 3 decimals — `n/a` when there is no observation). */
export const formatDrmUsage = (usage: DrmUsage | undefined): string =>
  usage === undefined
    ? "n/a"
    : [...usage].map(([region, bytes]) => `${region}=${(bytes / GIB).toFixed(3)}`).join(" ");
