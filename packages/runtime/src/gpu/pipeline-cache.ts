/**
 * A cache of (key → compute pipeline).
 *
 * MUST: the same key always corresponds to byte-identical WGSL (codegen determinism).
 * The browser's implicit pipeline cache keys on the WGSL string itself, and there is no explicit
 * control API. If determinism breaks, we not only pay the first-compile cost every time but also
 * run a different kernel under the same key (a cache hit uses the old WGSL), so the runtime guard
 * below fails a mismatch at once.
 *
 * **One per device** (owned by GpuContext — `GpuContextInternals.pipelines`). Sessions on the same
 * device share this one instance and do not get their own — whether a pipeline can be reused is
 * decided per device, so a per-Session cache would pay the compilation of the same WGSL and the
 * resolution of `getBindGroupLayout` once per Session. Pipelines carried over past a device loss
 * are unusable, so the only way to rebuild is reconstructing the GpuContext from `acquireGpu()`.
 */

import { GpuInternalError, GpuValidationError, withPipelineScope } from "./error-scope.ts";

/** Different WGSL was passed for the same key (determinism broken). */
export class PipelineKeyConflictError extends Error {
  override readonly name = "PipelineKeyConflictError";
}

/**
 * The WGSL storage binding declarations contradict each other (the same binding number appears as
 * both read and read_write).
 */
export class StorageRoleError extends Error {
  override readonly name = "StorageRoleError";
}

/**
 * The **roles** of the group 0 storage bindings (binding number → read / read-write). Taken
 * mechanically from the WGSL `var<storage, read>` / `var<storage, read_write>` declarations (no
 * hand-written table).
 *
 * Used by the region plan for intermediate buffers (src/runtime/transient-plan.ts — ADR 0093
 * decision 1): two tensors of which one is read and the other written by the same dispatch cannot
 * share a GPUBuffer (WebGPU's usage-scope validation works per buffer and does not look at bound
 * ranges). If the plan guessed the roles instead of taking them from the dispatch, only the
 * dispatches where the guess is wrong would fail validation (not as an exception — the output would
 * be all 0).
 */
export type StorageRoles = {
  readonly reads: ReadonlySet<number>;
  readonly writes: ReadonlySet<number>;
};

const STORAGE_DECLARATION =
  /@group\(0\)\s*@binding\((\d+)\)\s*var<storage\s*,\s*(read|read_write)\s*>/g;

/**
 * Takes {@link StorageRoles} from WGSL (a pure function).
 *
 * MUST: a declaration has the one-line form `@group(0) @binding(N) var<storage, read|read_write>`
 * (every declaration in codegen / kernels has this form — confirmed on 2026-09-05 by counting 157
 * declarations). If codegen starts emitting another spelling (`var<storage>` with its default read,
 * or a form with a line break), this function misses it, and that dispatch fails loudly in the plan
 * as "role unknown" (the consumer of {@link SessionPipelines} checks that every binding has a
 * role).
 */
export const parseStorageRoles = (wgsl: string): StorageRoles => {
  const reads = new Set<number>();
  const writes = new Set<number>();
  for (const match of wgsl.matchAll(STORAGE_DECLARATION)) {
    const binding = Number(match[1]);
    const target = match[2] === "read_write" ? writes : reads;
    const other = match[2] === "read_write" ? reads : writes;
    if (other.has(binding)) {
      throw new StorageRoleError(
        `WGSL の binding ${binding} が read と read_write の両方で宣言されている`,
      );
    }
    target.add(binding);
  }
  return { reads, writes };
};

/**
 * The resolved pair returned by {@link PipelineCache.get}. A pipeline's layout never changes, so
 * `getBindGroupLayout(0)` is called once when pipeline creation resolves and kept (not called again
 * for the bind group of every dispatch).
 */
type CachedPipeline = {
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
  /** The roles of the storage bindings (taken once from the WGSL — {@link parseStorageRoles}). */
  readonly roles: StorageRoles;
};

type PipelineCompilation =
  | { readonly ok: true; readonly pipeline: GPUComputePipeline }
  | { readonly ok: false; readonly cause: unknown };

type CacheEntry = {
  readonly wgsl: string;
  /**
   * Shared **including a creation that has not settled yet** (holds the Promise, not the resolved
   * value). If concurrent get() calls for the same key each created a shader module and a pipeline,
   * the measured compile cost would double, and the validation errorScopes would also nest.
   */
  readonly resolved: Promise<CachedPipeline>;
};

export class PipelineCache {
  readonly #device: GPUDevice;
  readonly #entries = new Map<string, CacheEntry>();

  constructor(device: GPUDevice) {
    this.#device = device;
  }

  /** Number of registered entries (including those being created, excluding failed ones). */
  get size(): number {
    return this.#entries.size;
  }

  /**
   * Returns the compute pipeline + bind group layout for the key (creating them if needed).
   * **A creation in progress is awaited through the same entry** — registration happens before the
   * await (registering after the await would make every get() of the same key that arrives in the
   * meantime miss it and create a duplicate).
   *
   * The WGSL mismatch check does not wait for creation to finish (it applies from the moment the
   * entry is in the cache).
   *
   * Creation always happens inside an errorScope (the two scopes internal + validation =
   * {@link withPipelineScope}). An invalid shader / pipeline does not become a synchronous
   * exception; creation looks successful while the dispatch becomes a no-op and the output is
   * all 0. The layout is obtained by calling `getBindGroupLayout(0)` once when pipeline creation
   * resolves and is kept alongside (so callers do not call it again for every dispatch).
   */
  async get(key: string, wgsl: string): Promise<CachedPipeline> {
    const cached = this.#entries.get(key);
    if (cached !== undefined) {
      if (cached.wgsl !== wgsl) {
        throw new PipelineKeyConflictError(
          `パイプラインキー "${key}" に異なる WGSL が渡された（codegen 決定性の破れ）`,
        );
      }
      return await cached.resolved;
    }
    // MUST: do not keep a failed entry (keeping it would hand the same rejected Promise to everyone
    // and remove any chance of a retry). This handler is registered at creation, so it runs before
    // any waiter observes the rejection = only a get() that arrives after the deletion recreates.
    // MUST: take the roles before creation (a contradiction in the declarations fails without
    // waiting for compilation).
    const roles = parseStorageRoles(wgsl);
    const resolved = (async (): Promise<CachedPipeline> => {
      let compiling: Promise<PipelineCompilation> | undefined;
      const created = await withPipelineScope(this.#device, `createShaderModule(${key})`, () => {
        const module = this.#device.createShaderModule({
          label: key,
          code: wgsl,
        });
        // Issue inside the synchronous section that pushed the scope; wait for completion after
        // the pop.
        // DECIDED: docs/decisions/0042-prepared-execution-plan.md#addendum-2026-09-11--asynchronous-compilation
        // Turn a rejection into a value and receive it at once. Even if module validation fails
        // first, no unhandled rejection is left behind.
        const pending = this.#device
          .createComputePipelineAsync({
            label: key,
            layout: "auto",
            compute: { module },
          })
          .then(
            (pipeline) => ({ ok: true as const, pipeline }),
            (cause: unknown) => ({ ok: false as const, cause }),
          );
        compiling = pending;
        return { module, pending };
      }).catch(async (cause: unknown) => {
        // Even when module validation fails, do not move on to a retry or disposal before the
        // already-issued compilation has settled.
        await compiling;
        throw cause;
      });
      const result = await created.pending;
      if (!result.ok) {
        const cause = result.cause;
        if (
          cause instanceof Error &&
          "reason" in cause &&
          (cause.reason === "internal" || cause.reason === "validation")
        ) {
          // Some implementations leave the diagnostics empty with GPUPipelineError.message alone.
          // Even if fetching the supplement fails, do not overwrite the root cause and the
          // rejection reason.
          const details = await created.module.getCompilationInfo().then(
            (info) =>
              info.messages
                .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`)
                .join("; "),
            (diagnosticCause: unknown) => `getCompilationInfo failed: ${String(diagnosticCause)}`,
          );
          const message = `createComputePipelineAsync(${key}): ${String(cause)} ${details}`;
          if (cause.reason === "internal") throw new GpuInternalError(message, { cause });
          throw new GpuValidationError(message, { cause });
        }
        // A rejection reason other than those WebGPU specifies is not classified as validation by
        // guesswork.
        throw cause;
      }
      // Wrap the layout lookup in its own synchronous scope too, so it does not absorb the scopes
      // of other creations while waiting.
      return await withPipelineScope(this.#device, `getBindGroupLayout(${key})`, () => ({
        pipeline: result.pipeline,
        layout: result.pipeline.getBindGroupLayout(0),
        roles,
      }));
    })().catch((cause: unknown) => {
      this.#entries.delete(key);
      throw cause;
    });
    this.#entries.set(key, { wgsl, resolved });
    return await resolved;
  }
}

/**
 * A thin delegate to {@link PipelineCache} that records usage for one Session.
 *
 * Since the cache lives as long as the device, "the number of pipelines on this device" and "the
 * number of pipelines this Session used" became different quantities. The latter is what the
 * diagnostics (`SessionDiagnostics.pipelineCount`) answer — how many kernels a graph and its
 * opt-ins brought up is a per-Session fact, and it is not allowed to change with whether another
 * Session is on the same device. The former is answered by
 * `SessionDiagnostics.devicePipelineCount`.
 *
 * MUST: derive the count from the set of used keys (no independently updated counter). Several
 * steps fetching the same key is the normal case, so an incrementing counter would count "fetches"
 * instead of "keys used", and a drift would raise neither an exception nor a warning.
 */
export class SessionPipelines {
  readonly #cache: PipelineCache;
  readonly #usedKeys = new Set<string>();

  constructor(cache: PipelineCache) {
    this.#cache = cache;
  }

  /** Number of pipeline keys this Session used (derived from the set). */
  get usedCount(): number {
    return this.#usedKeys.size;
  }

  /** The count held by the delegate target = the total over every Session on this device. */
  get deviceCount(): number {
    return this.#cache.size;
  }

  /**
   * Delegates to {@link PipelineCache.get} and records **only resolved keys** in the used set.
   *
   * MUST: record after resolution (failed keys are not counted). The delegate target does not keep
   * failed entries, so recording at issue time would leave the state "not on the device, yet the
   * Session counts it as used".
   */
  async get(key: string, wgsl: string): Promise<CachedPipeline> {
    const resolved = await this.#cache.get(key, wgsl);
    this.#usedKeys.add(key);
    return resolved;
  }
}
