/**
 * The **residency class** of weights — a pure-function planner decided by the graph declaration
 * alone, and the Session-side discriminated union that binds its result to GPU objects.
 *
 * MUST: the classification has one source of truth, here ({@link planWeightResidency}). If Session
 * construction (session-build.ts) and the estimate (estimate.ts) each wrote "eligibility + branch on
 * storage dtype", fixing only one would make **the estimate and the real load describe different
 * models** (with no exception and no warning).
 * MUST: do not look at the real tensors. Byte counts are derived from the declared shape and the
 * storage metadata; agreement with the real bytes is guaranteed by the container's matching checks
 * (the payload-length check of the merge layer and the "N bytes from the declaration" check of the
 * construction phase).
 *
 * The mapping from residency class to "the GPU buffers actually allocated" ({@link planWeightBuffers})
 * and the guard that compares their sizes with the device's absolute limits
 * ({@link assertWeightsWithinLimits}) live here for the same reason — if the allocating side
 * (session-build.ts) and the counting side (estimate.ts) each expanded the residency classes, fixing
 * only one would make the check, the estimate and the real load claim different sizes.
 */

import {
  codecLayout,
  type CodecName,
  groupScaleShape,
  payloadBytes,
  scaleBytes,
} from "../format/container/codecs.ts";
import type { IrDtype, IrGraph } from "../format/ir.ts";
import { toSizeClass } from "../gpu/arena.ts";
import type { GpuContext } from "../gpu/device.ts";
import { RUNTIME_INTERNAL } from "../gpu/device.ts";
import { numel } from "../ops.ts";
import {
  eligibleCompressedInitializers,
  ExecutionError,
  i2EligibleInitializers,
  i4EligibleInitializers,
  weightChannelAxes,
} from "./plan.ts";

/**
 * One weight resident on the GPU in compressed form (the index paired with the Session's weight
 * ledger — kernel-variant selection and extra bindings are decided from here alone).
 *
 * MUST: it does not hold the payload's `GPUBuffer` (the weight ledger `weightBuffers` owns it —
 * holding it twice raises "which one is real"). Only the scale is owned here, because it is an
 * object the weight ledger does not carry.
 * MUST: the extra information of each residency class is required **by the type** (i8 / i2 / i4
 * require a scale, i4 requires a group length). Three parallel Maps would let "i4 without a group
 * length" be built at the type level, leaving the run-time check as the last line of defence.
 */
export type ResidentWeight =
  // f16 and bf16 are separate members (with one member of `"f16" | "bf16"`, narrowing by `!== "f16"`
  // leaves that member in place and cannot narrow to the side that holds a scale).
  | { readonly storage: "f16" }
  | { readonly storage: "bf16" }
  | { readonly storage: "i8" | "i2"; readonly scale: GPUBuffer }
  | { readonly storage: "i4"; readonly scale: GPUBuffer; readonly groupSize: number };

/**
 * The residency class of one initializer (the value of {@link planWeightResidency}).
 *
 * There are eight classes:
 * - `raw` — uncompressed storage (f32 / i32) resident on the GPU as raw bytes (construction phase of session-build.ts)
 * - `f16` / `bf16` / `i8` / `i2` / `i4` — resident in compressed form, dequantized inside the kernel (ADR 0018 / 0131 / 0019 / 0097 / 0069)
 * - `expanded` — ineligible, decoded to f32 on the CPU at load (correctness is kept, VRAM saving is zero)
 * - `shared` — a borrowed weight (ADR 0096 decision 2 and its stage 2 addendum — binds the lending Session's GPU buffer)
 *
 * `payloadBytes` is **the length of the stored bytes themselves** (neither alignment padding nor the
 * buffer floor is included). Alignment is the business of the upload side (`alignF16Payload` /
 * `alignI8Payload`) and the arena (`toSizeClass`).
 */
export type WeightResidency =
  | { readonly seat: "raw"; readonly payloadBytes: number }
  | { readonly seat: "f16" | "bf16"; readonly payloadBytes: number }
  | {
      readonly seat: "i8" | "i2";
      readonly payloadBytes: number;
      readonly scaleBytes: number;
      /** The row axis (where the per-channel scale applies) — the declared `storage.rowAxis` (already checked to match the consuming op's axis). */
      readonly rowAxis: 0 | 1;
    }
  | {
      readonly seat: "i4";
      readonly payloadBytes: number;
      readonly scaleBytes: number;
      readonly groupSize: number;
    }
  | {
      readonly seat: "expanded";
      readonly payloadBytes: number;
      /** Bytes after decoding to f32 on the CPU (this is what is resident). */
      readonly expandedBytes: number;
    }
  | {
      /**
       * **A borrowed weight** (the shared initializer of ADR 0096 decision 2 and its stage 2
       * addendum). This Session allocates and uploads not a single byte; it binds the lending
       * Session's GPU buffer as is (`SessionOptions.sharedWeights`).
       */
      readonly seat: "shared";
      /**
       * The borrower-side consumption (eligibility). The borrower declares no storage (the lender's
       * resident weight is the source of truth), so the expected residency class is derived by
       * {@link resolveSharedWeights} at construction, when the lender's codec is known.
       */
      readonly eligible: boolean;
      readonly i4Eligible: boolean;
      readonly i2Eligible: boolean;
      /** The per-channel scale axis decided by the consuming op (absent when not consumed by a weight slot). */
      readonly consumerAxis?: 0 | 1;
    };

/**
 * Decides the residency class of every initializer from the graph alone (a pure function that needs
 * neither the GPU nor safetensors).
 *
 * The classification has three parts — eligibility {@link eligibleCompressedInitializers}, for i4
 * only the intersection with {@link i4EligibleInitializers} (ADR 0069 decision 5), and the branch per
 * storage dtype.
 *
 * MUST: return **every** entry of `graph.initializers` (in declaration order). Allowing gaps makes
 * them indistinguishable from the consumers' default reading "not in the table = read as f32".
 */
export const planWeightResidency = (graph: IrGraph): ReadonlyMap<string, WeightResidency> => {
  // Initializers that may be uploaded in compressed storage (consumed only by weight slots — ADR 0018).
  const eligible = eligibleCompressedInitializers(graph);
  // i4 eligibility is narrower: "consumed at the weight slot only by linear / embedding /
  // conv1d(groups==1)" (ADR 0069 decision 5 and its addendum — these 3 are the kernels with a
  // decode path).
  const i4Eligible = i4EligibleInitializers(graph);
  const i2Eligible = i2EligibleInitializers(graph);
  // The per-channel scale axis has the **declaration** (`storage.rowAxis` — container-v1 §6.1) as
  // its source of truth. The axis decided by the consuming op (ADR 0019) stays as a cross-check — a
  // mismatch with the declaration would make the GPU-resident path apply the scale to another axis,
  // a silently wrong value.
  const consumerAxes = weightChannelAxes(graph);
  const plan = new Map<string, WeightResidency>();
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const where = `initializer '${name}'`;
    // An initializer's declared shape is numeric only (guaranteed by the parser — symbolic dimensions are rejected).
    const shape = graph.values[name].shape.map(Number);
    const count = numel(shape);
    // A shared initializer decides only its residency class (no bytes are counted — the lender
    // allocates). The borrower declares no storage, so the expected residency class is derived at
    // borrower construction, when the lender's codec is known ({@link resolveSharedWeights}) — here
    // only the borrower-side consumption (eligibility) is recorded.
    if (initializer.shared !== undefined) {
      const consumerAxis = consumerAxes.get(name);
      plan.set(name, {
        seat: "shared",
        eligible: eligible.has(name),
        i4Eligible: i4Eligible.has(name),
        i2Eligible: i2Eligible.has(name),
        ...(consumerAxis === 0 || consumerAxis === 1 ? { consumerAxis } : {}),
      });
      continue;
    }
    const { codec, groupSize, rowAxis } = initializer.storage;
    const layout = codecLayout(codec);
    let bytes: number;
    try {
      bytes = payloadBytes(codec, count, where);
    } catch (cause) {
      throw new ExecutionError(cause instanceof Error ? cause.message : String(cause));
    }
    if (layout === "f32" || layout === "i32") {
      // Uncompressed storage: the raw bytes are the GPU representation as is.
      plan.set(name, { seat: "raw", payloadBytes: bytes });
      continue;
    }
    const resident =
      layout === "i4"
        ? eligible.has(name) && i4Eligible.has(name)
        : layout === "i2"
          ? eligible.has(name) && i2Eligible.has(name)
          : eligible.has(name);
    if (!resident) {
      plan.set(name, { seat: "expanded", payloadBytes: bytes, expandedBytes: count * 4 });
      continue;
    }
    if (layout === "f16" || layout === "bf16") {
      plan.set(name, { seat: layout, payloadBytes: bytes });
      continue;
    }
    // Quantized codecs: the presence and range of rowAxis / groupSize are guaranteed by the merge
    // layer. They are optional only in the type, so restate the check rather than skip silently.
    const axis = rowAxis ?? 0;
    if (groupSize === undefined) {
      throw new ExecutionError(`${where}: quantized codec '${codec}' but no groupSize`);
    }
    const consumerAxis = consumerAxes.get(name);
    if (consumerAxis !== undefined && consumerAxis !== axis) {
      throw new ExecutionError(
        `${where}: the declared rowAxis ${axis} differs from the consuming op's channel axis ${consumerAxis}`,
      );
    }
    const rows = shape[axis];
    if (rows === undefined) {
      throw new ExecutionError(`${where}: weight [${shape.join(",")}] has no row axis ${axis}`);
    }
    const scale = scaleBytes(numel(groupScaleShape(shape, axis, groupSize)), where);
    if (layout === "i8" || layout === "i2") {
      plan.set(name, { seat: layout, payloadBytes: bytes, scaleBytes: scale, rowAxis: axis });
      continue;
    }
    // The i4 group form takes the leading dimension as rows (the premise of the decode kernels and
    // of `decodeI4` — ADR 0069 decision 3).
    if (axis !== 0) {
      throw new ExecutionError(
        `${where}: group quantization (${codec}) supports rowAxis 0 only (declared ${axis})`,
      );
    }
    plan.set(name, { seat: "i4", payloadBytes: bytes, scaleBytes: scale, groupSize });
  }
  return plan;
};

/**
 * Derives the **expected residency class** of the borrowed object from the borrower-side consumption
 * (eligibility) and the lender's codec.
 *
 * MUST: derive it again **independently** from the borrower-side consumption. Even if the lender is
 * i4-resident, the class is `expanded` when the borrower's consumption has no decode path, and the
 * same buffer would be read two ways, "packed i4 bytes" and "f32 values" — with not a single
 * exception.
 */
const expectedSharedSeat = (
  codec: CodecName,
  seat: Extract<WeightResidency, { readonly seat: "shared" }>,
): Exclude<WeightResidency["seat"], "shared"> => {
  const layout = codecLayout(codec);
  if (layout === "f32" || layout === "i32" || layout === "bf16") return "raw";
  const resident =
    layout === "i4"
      ? seat.eligible && seat.i4Eligible
      : layout === "i2"
        ? seat.eligible && seat.i2Eligible
        : seat.eligible;
  return resident ? layout : "expanded";
};

/**
 * The runtime-internal surface of {@link SharedWeight} (not a surface for users).
 *
 * MUST: the lender's objects (buffer, `ResidentWeight`) are held **by reference only**. Taking a copy
 * raises "which one is real", and when the lender changes from f16-resident to the expanded class,
 * only the borrower would run on the old class.
 */
export type SharedWeightInternals = {
  /** The lender's initializer name (for diagnostics). */
  readonly initializer: string;
  /** The lender's device (MUST be the borrower's — a buffer of another device cannot be bound). */
  readonly gpu: GpuContext;
  /** The buffer of the weight body the lender allocated (the borrower does not own it). */
  readonly buffer: GPUBuffer;
  /** The residency class and its objects when resident in compressed form (undefined for f32 / the raw class). */
  readonly resident: ResidentWeight | undefined;
  /** The lender's residency class (MUST match the borrower's expected class). */
  readonly seat: Exclude<WeightResidency["seat"], "shared">;
  /** The row (per-channel scale) axis of the i8 / i2 classes (undefined otherwise). */
  readonly rowAxis: 0 | 1 | undefined;
  /** The lender's storage codec (the borrower's expected class is derived from it and the borrower-side consumption). */
  readonly codec: CodecName;
  /** The lender's declared shape. */
  readonly shape: readonly number[];
  /**
   * The lender's **semantic** dtype (the graph's declaration — not the storage codec). A weight
   * resident in compressed form is f32 once decoded, so this is what is compared with the borrower's
   * declaration (the borrower declares no storage).
   */
  readonly dtype: IrDtype;
  /** Adds one borrow (when the borrowing Session's construction succeeds). */
  retain(): void;
  /** Returns one borrow (on dispose of the borrowing Session / failure of its construction). */
  release(): void;
};

/**
 * An **opaque reference** to one weight the lending Session put on the GPU (ADR 0096 decision 2 and
 * its stage 2 addendum).
 *
 * Only `Session.exportWeight(initializerName)` creates it, and it is passed to the borrowing Session
 * through `SessionOptions.sharedWeights`. The borrower holds no bytes and binds the lender's buffer
 * as is.
 *
 * MUST: the only entry for construction is `Session.exportWeight` (the same convention as
 * `ResidentTensor` — constructing it directly could bypass the residency-class match and the borrow
 * counting).
 */
export class SharedWeight {
  /** Runtime-internal surface (not a surface for users). */
  readonly [RUNTIME_INTERNAL]: SharedWeightInternals;

  constructor(internals: SharedWeightInternals) {
    this[RUNTIME_INTERNAL] = internals;
  }
}

/**
 * Matches the borrower graph's shared-initializer declarations against the given
 * {@link SharedWeight}s (the check of ADR 0096 decision 2 and its stage 2 addendum). Returns the
 * pairs to inject (in declaration order).
 *
 * Six points are checked (the borrower declares no storage, so there is no "storage dtype matches"
 * check; instead point 5 derives the expected residency class):
 * 1. **No missing, no surplus** — exactly one per declaration (a missing one is "running with a
 *    weight that has no bytes", a surplus one is "a weight thought to be passed that nobody uses")
 * 2. **Same device** — a bind group binding another device's buffer fails validation, but the
 *    diagnostic is far from the real cause
 * 3. **Declared shape matches** — byte counts alone let `[2,3]` and `[3,2]` be mixed up
 * 4. **Semantic dtype matches** — f32 and i32 both have the resident form `raw`, so the residency
 *    class match (5, 6) cannot tell them apart, and letting it through reads the same 4 bytes as
 *    another type (f32 1.0 becomes i32 1065353216). Ordinary initializers have their semantic dtype
 *    and storage codec matched by the merge layer (`format/container/bind.ts`), but shared ones have
 *    no storage and do not pass there — they are compared with the lender's declaration instead
 * 5. **Deriving the expected class** — the borrower declares no storage, so the expected class is
 *    derived from the lender's codec and the borrower-side consumption (eligibility)
 *    (`expectedSharedSeat`). If it differs from the lender's actual class, the same buffer is read
 *    another way
 * 6. **Residency class matches** (for i8, down to the per-channel scale axis) — even if the lender is
 *    i4-resident, the class is `expanded` when the borrower's consumption has no decode path, and the
 *    same buffer is read two ways, as packed bytes and as f32. A mismatched i8 axis (embedding vs
 *    linear) becomes a silently wrong value by the same mechanism
 *
 * MUST: all fail loudly. For all six points a violation shows up as **a different value**, not as an
 * exception.
 */
export const resolveSharedWeights = (
  graph: IrGraph,
  residency: ReadonlyMap<string, WeightResidency>,
  gpu: GpuContext,
  provided: Readonly<Record<string, SharedWeight>> | undefined,
): readonly { readonly name: string; readonly shared: SharedWeight }[] => {
  const declared = Object.entries(graph.initializers)
    .filter(([, initializer]) => initializer.shared !== undefined)
    .map(([name]) => name);
  const given = Object.keys(provided ?? {});
  const missing = declared.filter((name) => !Object.hasOwn(provided ?? {}, name));
  const surplus = given.filter((name) => !declared.includes(name));
  if (missing.length > 0 || surplus.length > 0) {
    throw new ExecutionError(
      `options.sharedWeights does not match the graph's shared declarations: missing [${missing.join(
        ", ",
      )}] / extra [${surplus.join(", ")}]` +
        " (exactly one per shared declaration, none missing or extra, MUST — ADR 0096 step 2 §2.2)",
    );
  }
  return declared.map((name) => {
    const weight = (provided ?? {})[name];
    // MUST: a value from outside the type (a JS caller) fails before the internal surface is read —
    // reading it would let a `TypeError` (the `shared.initializer` access) or an exception from the
    // user's getter escape instead of an `ExecutionError`.
    if (!(weight instanceof SharedWeight)) {
      throw new ExecutionError(
        `sharedWeights['${name}']: not a SharedWeight (${typeof weight}) — pass what the lending Session's ` +
          "exportWeight returned",
      );
    }
    const shared = weight[RUNTIME_INTERNAL];
    const where = `sharedWeights['${name}' (lender initializer '${shared.initializer}')`;
    if (shared.gpu !== gpu) {
      throw new ExecutionError(`${where}: lender and borrower GpuContexts (devices) differ`);
    }
    const shape = graph.values[name].shape.map(Number);
    if (shape.length !== shared.shape.length || shape.some((d, i) => d !== shared.shape[i])) {
      throw new ExecutionError(
        `${where}: the declared shape [${shape.join(",")}] differs from the lender's [${shared.shape.join(",")}]`,
      );
    }
    const dtype = graph.values[name].dtype;
    if (dtype !== shared.dtype) {
      throw new ExecutionError(
        `${where}: the semantic dtype '${dtype}' differs from the lender's '${shared.dtype}'` +
          " (the resident form is the same, so the residency class check cannot tell them apart and the same bytes would be read as another type)",
      );
    }
    const seat = residency.get(name);
    if (seat === undefined || seat.seat !== "shared") {
      throw new ExecutionError(`${where}: the residency class is not shared (broken bookkeeping)`);
    }
    // The borrower declares no storage — derive the expected class from the lender's codec and the
    // borrower-side consumption, and match it against the lender's actual class (and the row axis for
    // i8 / i2).
    const expected = expectedSharedSeat(shared.codec, seat);
    const expectedAxis =
      expected === "i8" || expected === "i2" ? (seat.consumerAxis ?? 0) : undefined;
    if (expected !== shared.seat || expectedAxis !== shared.rowAxis) {
      throw new ExecutionError(
        `${where}: the consuming residency class is not compatible with the lender's (borrower class '${expected}'${
          expectedAxis === undefined ? "" : `, row axis ${expectedAxis}`
        }, lender class '${shared.seat}'${
          shared.rowAxis === undefined ? "" : `, row axis ${shared.rowAxis}`
        }) — the same buffer would be read two ways`,
      );
    }
    return { name, shared: weight };
  });
};

/**
 * One buffer that one residency class makes the GPU allocate (an element of {@link planWeightBuffers}).
 *
 * MUST: the branch "which bytes of a residency class become a GPU buffer" lives here alone — if the
 * limit check ({@link assertWeightsWithinLimits}) and the estimate (`weightEstimate` in estimate.ts)
 * each wrote the branch per class, a change in eligibility would make one of them claim a different
 * size.
 */
export type WeightBuffer = {
  readonly name: string;
  /**
   * The residency class that issued this allocation. **A borrowed one (`shared`) never appears** —
   * saying so in the type keeps an exhaustive switch over classes (the estimate's split into 3
   * fields) from counting a class that is not counted.
   */
  readonly seat: Exclude<WeightResidency["seat"], "shared">;
  /** `payload` = the weight body (after f32 decoding for the `expanded` class) / `scale` = companion scale. */
  readonly kind: "payload" | "scale";
  /** Bytes passed to `createBuffer` (`toSizeClass` = 4-byte alignment + 4-byte floor). */
  readonly byteLength: number;
  /** Bytes from the declaration before alignment (what the estimate's "exact" field counts). */
  readonly declaredBytes: number;
};

/**
 * Lists, in declaration order, the buffers the residency plan makes the GPU allocate (a pure function
 * that needs neither the GPU nor a device).
 *
 * The eligible classes (f16 / i8 / i4) and the raw class upload the payload as is; the ineligible
 * class (`expanded`) uploads the bytes **after** decoding to f32 on the CPU (the compressed byte
 * count of the distribution does not reach the GPU). i8 / i4 allocate the companion scale as **one
 * more** buffer separate from the payload (`timedAlloc(Math.max(4, scale.bytes.byteLength))` in
 * session-build.ts).
 */
export const planWeightBuffers = (
  residency: ReadonlyMap<string, WeightResidency>,
): readonly WeightBuffer[] => {
  const buffers: WeightBuffer[] = [];
  const add = (
    name: string,
    seat: WeightBuffer["seat"],
    kind: WeightBuffer["kind"],
    declaredBytes: number,
  ): void => {
    buffers.push({ name, seat, kind, byteLength: toSizeClass(declaredBytes), declaredBytes });
  };
  for (const [name, seat] of residency) {
    // A borrowed class allocates nothing (the lending Session holds the object). This line is the
    // only branch point that takes it out of both the limit check and the estimate.
    if (seat.seat === "shared") continue;
    add(
      name,
      seat.seat,
      "payload",
      seat.seat === "expanded" ? seat.expandedBytes : seat.payloadBytes,
    );
    if (seat.seat === "i8" || seat.seat === "i4" || seat.seat === "i2") {
      add(name, seat.seat, "scale", seat.scaleBytes);
    }
  }
  return buffers;
};

/**
 * The device limits the limit check looks at (a subset of `GpuContext.limits`).
 *
 * MUST: look at both. `maxStorageBufferBindingSize ≤ maxBufferSize` is a relation kept by the side
 * that plans the device (`planRequiredLimits` in gpu/device.ts), not a guarantee that extends to a
 * `GpuContext` passed in from outside — looking at only one stays silent on a device where the
 * relation is broken.
 */
export type WeightLimits = {
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
};

/** The subject of the error text (only the `expanded` class states "what is allocated is the decoded form"). */
const bufferLabel = (buffer: WeightBuffer): string =>
  buffer.kind === "scale"
    ? "scale"
    : buffer.seat === "expanded"
      ? "payload (decoded to f32)"
      : "payload";

/**
 * Compares the allocation size of each weight with the device's absolute limits, and fails **before
 * allocation** when any is exceeded.
 *
 * The motive is the weakness of "detecting allocation failure relies entirely on the errorScope per
 * block (item)" (ADR 0108 decision 9) — exactly the fix candidate (an explicit size check on the
 * weight path) named by docs/known-issues.md "On Metal the out-of-memory errorScope stays silent".
 * The errorScope net falls short in 3 ways:
 * ① **implementation-dependent** — exceeding a limit is normally caught by validation, but devices
 * whose out-of-memory scope on the same path stays silent exist (measured on M2). As long as the net
 * bets on the implementation's reporting quality, "allocation failure = no-op writeBuffer into an
 * invalid buffer = reading garbage" can slip through.
 * ② **late** — detection comes after weight upload has begun, after several GiB were transferred.
 * ③ **coarse** — it can name only the failed block, not which weight exceeded by how many bytes.
 * The sizes are fixed by the declaration alone before allocation (the residency plan is a pure
 * function of the prepare phase), so what can fail deterministically fails here.
 *
 * MUST: look **only at the comparison with absolute limits**. Do not compare with free VRAM (ADR
 * 0070 decision 5 — WebGPU exposes neither total nor free VRAM, so turning it into a comparison
 * makes it a guess). Do not look at the total size either (this checks a per-buffer device
 * constraint; whether the total fits is the final check's job = the errorScope).
 * MUST: **list every violation and fail once**. Failing one at a time makes the next one appear each
 * time the export is redone, and how many must be fixed for it to fit stays unknown to the end.
 */
export const assertWeightsWithinLimits = (
  residency: ReadonlyMap<string, WeightResidency>,
  limits: WeightLimits,
): void => {
  // Pairs that make the text always name "which limit" (the same form as the state-side check —
  // `assertSlotWithinLimits` in generation-context.ts).
  const entries = [
    ["maxStorageBufferBindingSize", limits.maxStorageBufferBindingSize],
    ["maxBufferSize", limits.maxBufferSize],
  ] as const;
  const violations: string[] = [];
  for (const buffer of planWeightBuffers(residency)) {
    const exceeded = entries
      .filter(([, limit]) => buffer.byteLength > limit)
      .map(([key, limit]) => `exceeds ${key} ${limit} bytes by ${buffer.byteLength - limit} bytes`);
    if (exceeded.length === 0) continue;
    violations.push(
      `  - initializer '${buffer.name}' ${bufferLabel(buffer)} (residency class ${buffer.seat}, allocated ` +
        `${buffer.byteLength} bytes): ${exceeded.join(", ")}`,
    );
  }
  if (violations.length === 0) return;
  throw new ExecutionError(
    `${violations.length} weight buffers exceed the device limits (detected before allocation):\n` +
      `${violations.join("\n")}\n` +
      "The limits are per buffer, so splitting into blocks (pieces) does not resolve it — " +
      "re-export with a smaller storage dtype" +
      ", or split the weight and rebuild the graph",
  );
};
