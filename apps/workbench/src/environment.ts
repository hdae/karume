/**
 * The environment of a record, read from an adapter the host requests itself on the same `GPU`
 * object the entry will use (an entry's own device facts are its business, not the envelope's).
 */

import type { Backend, RecordEnvironment } from "./record.ts";

/** What the host knows and WebGPU does not tell. */
export type HostFacts = {
  readonly host: "browser" | "node";
  readonly userAgent: string;
  readonly crossOriginIsolated?: boolean;
  readonly backend?: Backend;
};

/** A `GPU` whose implementation may predate `wgslLanguageFeatures`. */
type WgslLanguageFeatureHost = GPU & { readonly wgslLanguageFeatures?: Iterable<string> };

/** No adapter to describe (the shell checks for one before it offers Run). */
export class NoAdapterError extends Error {
  constructor() {
    super("WebGPU returned no adapter");
    this.name = "NoAdapterError";
  }
}

export const readEnvironment = async (gpu: GPU, facts: HostFacts): Promise<RecordEnvironment> => {
  const adapter = await gpu.requestAdapter();
  if (adapter === null) throw new NoAdapterError();
  const limits: Record<string, number> = {};
  // GPUSupportedLimits keeps its values on the prototype, so they are read by name.
  for (const name in adapter.limits) {
    const value: unknown = Reflect.get(adapter.limits, name);
    if (typeof value === "number") limits[name] = value;
  }
  const { vendor, architecture, device, description } = adapter.info;
  const host: WgslLanguageFeatureHost = gpu;
  return {
    host: facts.host,
    userAgent: facts.userAgent,
    ...(facts.crossOriginIsolated === undefined
      ? {}
      : { crossOriginIsolated: facts.crossOriginIsolated }),
    ...(facts.backend === undefined ? {} : { backend: facts.backend }),
    adapter: { vendor, architecture, device, description },
    features: [...adapter.features].sort(),
    limits,
    wgslLanguageFeatures: [...(host.wgslLanguageFeatures ?? [])].sort(),
  };
};

/** Dawn's backend when its flags name none. */
const DAWN_DEFAULT_BACKENDS: Readonly<Partial<Record<string, string>>> = {
  linux: "vulkan",
  darwin: "metal",
  win32: "d3d12",
};

/** The backend under Node: a `backend=<name>` Dawn flag, else Dawn's default for the platform. */
export const dawnBackend = (flags: readonly string[], platform: string): Backend => {
  const flag = flags.find((token) => token.startsWith("backend="));
  if (flag !== undefined) return { name: flag.slice("backend=".length), source: "dawn-flag" };
  const name = Object.hasOwn(DAWN_DEFAULT_BACKENDS, platform)
    ? DAWN_DEFAULT_BACKENDS[platform]
    : undefined;
  if (name === undefined) {
    throw new Error(`No Dawn default backend is known for the platform ${platform}`);
  }
  return { name, source: "dawn-default" };
};
