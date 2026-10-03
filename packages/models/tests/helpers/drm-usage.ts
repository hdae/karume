/**
 * この process の GPU メモリの使用量の標本化（診断 — Linux の DRM fdinfo）。
 *
 * `/proc/self/fdinfo/<renderD の fd>` の `drm-total-<領域>`（vram0 / gtt / system …）を領域ごとに足す。
 * xe は同じ接頭辞でエンジンのサイクル数（`drm-total-cycles-<engine>`）も出すので、それは数えない。
 * Linux の DRM でない環境（fdinfo が無い）は `undefined` を返し、呼び手は「観測なし」として扱う。
 * 10 ms ごとの標本なので短い山は取りこぼしうる（門ではなく観測に使う）。
 */

const RENDER_NODE = /^\/dev\/dri\/renderD\d+$/;
const FDINFO_UNITS: Readonly<Record<string, number>> = { "": 1, KiB: 1024, MiB: 1024 ** 2 };

/** 領域名（`vram0` / `gtt` / `system` …）→ バイト。 */
export type DrmUsage = ReadonlyMap<string, number>;

/**
 * この process の DRM クライアントの `drm-total-<領域>` を領域ごとに足した値。同じクライアントを指す
 * fd は `drm-client-id` で 1 度だけ数える。
 */
export const sampleDrmUsage = (): DrmUsage | undefined => {
  let fds: Deno.DirEntry[];
  try {
    fds = [...Deno.readDirSync("/proc/self/fd")];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
    throw cause;
  }
  const clients = new Map<string, DrmUsage>();
  for (const { name } of fds) {
    let info: string;
    try {
      if (!RENDER_NODE.test(Deno.readLinkSync(`/proc/self/fd/${name}`))) continue;
      info = Deno.readTextFileSync(`/proc/self/fdinfo/${name}`);
    } catch (cause) {
      // 列挙と読みの間に閉じた fd（標本化の自分の readDir の fd など）は数えない。
      if (cause instanceof Deno.errors.NotFound) continue;
      throw cause;
    }
    const client = /^drm-client-id:\s*(\d+)/m.exec(info)?.[1];
    if (client === undefined) continue;
    const regions = new Map<string, number>();
    for (
      const [, region, value, unit] of info.matchAll(/^drm-total-(\S+):\s*(\d+)\s*(KiB|MiB)?/gm)
    ) {
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

/** {@link monitorDrmUsage} の決着。 */
export type DrmTimeline = {
  /** 区間名 → 領域ごとの最大（区間に入った時点と抜けた時点の標本を含む）。 */
  readonly peaks: ReadonlyMap<string, DrmUsage>;
  /** 呼び手が名指しした時点の標本（`mark` の順）。 */
  readonly marks: readonly { readonly label: string; readonly usage: DrmUsage | undefined }[];
};

/**
 * 区間ごとの山を採る標本化を始める。`enter(区間名)` で区間を切り替え（切り替えの直前と直後に 1 回ずつ
 * 標本を採る）、`mark(名前)` でその時点の値を残し、`stop()` で止めて決着を返す。
 */
export const monitorDrmUsage = (intervalMs = 10): {
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

/** `vram0=1.234 gtt=0.010 …`（GiB・小数 3 桁 — 観測が無ければ `n/a`）。 */
export const formatDrmUsage = (usage: DrmUsage | undefined): string =>
  usage === undefined
    ? "n/a"
    : [...usage].map(([region, bytes]) => `${region}=${(bytes / GIB).toFixed(3)}`).join(" ");
