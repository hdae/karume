/**
 * Parsing and full structural checking of `karume.json` (distribution manifest v5 = `karume/5`) —
 * the canonical implementation of ADR 0109 (the quant preset vocabulary is inherited from ADR
 * 0041).
 *
 * MUST: **older versions are not read**. A `format` other than `karume/5` fails as an unsupported
 * format (no two-format parsing — ADR 0109 decision 1). v5 is the version where the distribution
 * moved from shard lists in a safetensors dialect to containers (`krm` — ADR 0108 / container-v1),
 * and the contents of `weights.<component>.<dtype>` changed from `{ shards, extras }` to
 * `{ container }` (the descriptor expectations + the part FileRef list). The `quants` / `session` /
 * `gpuFeatures` / `requiredLimits` / `label` / `description` / `assets` / cross-repo reference
 * fields stay as in v4 (ADR 0109 decisions 2 / 4).
 * MUST: hand-written parsing, Web standard APIs only, unsupported and unexpected input fails loudly
 * (no silent normalisation).
 * MUST: maps from the manifest are looked up only through `Object.hasOwn`, and composed only with
 * spread (no `Object.assign` — the cross-cutting invariant in CLAUDE.md / ADR 0038 §1).
 *
 * **Check boundary** (ADR 0109 decision 6): only **what closes on the declarations alone** is
 * checked here — counts, ceilings, the `size: 0` rule, the length of part 0, uniform cross-repo
 * references, descriptor lengths. Consistency of descriptor and parts, the block index and the
 * codec ledger are matched by **the side that opens the container** (`openContainer` of
 * `@karume/runtime`). Two check points would let the same rule drift apart in two places.
 */

import {
  type AvailableLabels,
  ManifestFormatError,
  ManifestPathError,
  ManifestReferenceError,
  NO_LABELS,
} from "./errors.ts";

/** The manifest's file name (a fixed name at the repository root — ADR 0041 decision 1). */
export const MANIFEST_FILENAME = "karume.json";

/**
 * The size limit of the manifest itself (DoS guard — ADR 0041 decision 7). Checked after receiving
 * everything ({@link parseManifest} and the local source's reader) — the fetch layer has no "cap
 * without exact match", so even the HF source does not stop midway through receiving (ADR 0094
 * decision 4).
 */
export const MAX_MANIFEST_BYTES = 1024 * 1024;

/** Size limits (ADR 0041 decision 7). */
const MAX_MODELS = 32;
const MAX_WEIGHTS = 32;
const MAX_ASSETS = 32;
const MAX_QUANTS = 32;
const MAX_PIPELINE_CONFIG_BYTES = 256 * 1024;
/**
 * The depth limit of the walk over the whole manifest. Real manifests need about 7 levels (envelope
 * → models → weights → dtype → container → parts → element; a dozen or so even with the nesting of
 * `pipelineConfig`). Without a depth check, deep nesting that fits in 1MiB would exhaust the stack
 * in the recursion of `assertNoForbiddenKeys` and escape the `HubError` contract as a bare
 * `RangeError` (measured on Deno: the bare walk hits `RangeError` at 2,410 nested arrays). It sits
 * well above the practical need and far below the measured breaking point, and a manifest that is
 * too deep fails loudly as a `ManifestFormatError`.
 */
const MAX_MANIFEST_DEPTH = 64;
/** The byte limit of one file (kept from ADR 0038 §2). Parts are within it. */
const MAX_FILE_BYTES = 16 * 2 ** 30;
/**
 * The limit on the part count of one container (container-v1 §10). Real distributions need a few
 * dozen, but without a limit an input that packs thousands of part refs into a 1MiB manifest would
 * become a fetch plan as is. The value equals `MAX_PARTS` in the canonical Python
 * `tools/exporter/src/karume/container.py`.
 */
const MAX_PARTS = 1024;
/**
 * The byte limit of one part (1024MiB — the "part length ceiling" of container-v1 §10; the same
 * value applies to every part, parts 0 / 1 included).
 *
 * What is measured is the manifest's `size` = **the file length**, the same quantity the exporter's
 * read-back (`karume verify`) checks against the real file length. Readers are built on the premise
 * of "estimating buffer sizes from the declarations" (ADR 0089), so silently accepting an oversized
 * part would break that premise, and it would hit the fetch layer's byte budget or Chromium's
 * single-`ArrayBuffer` limit **only once in the browser**. Parse time is the right place — the
 * reader contract is a format contract, and not allowing "found out only after the download
 * started" is the very purpose of checking the manifest.
 *
 * MUST: apply it to `container.parts` **only**. The limit is a contract of part splitting; `assets`
 * (auxiliary assets distributed as single files) are outside this rule — confusing them would make
 * real assets above the limit unreadable. The ceiling common to all FileRefs is
 * {@link MAX_FILE_BYTES}.
 *
 * MUST: keep the value equal to `PART_MAX_BYTES` in the canonical Python
 * `tools/exporter/src/karume/container.py` (hub does not depend on the exporter, so this is a
 * copy). The check is the same **closed interval** as there (exactly the limit is legal; only above
 * it fails).
 */
const MAX_PART_BYTES = 1024 * 2 ** 20;
/**
 * The byte limit of one descriptor document (graph descriptor / model descriptor) (32MiB —
 * container-v1 §10). A ceiling that keeps the cost of fetching part 0 in the same order as the
 * block limit; the measured IR is 9.0MiB over 47 deduplicated graphs (ADR 0109 Context). The value
 * equals `MAX_DESCRIPTOR_BYTES` in the canonical Python `tools/exporter/src/karume/container.py`.
 */
const MAX_DESCRIPTOR_BYTES = 32 * 2 ** 20;
/**
 * The container header length (container-v1 §1 — a fixed 24 bytes). Part 0 of the split form is
 * "header + exactly the two documents", so the consistency between declarations
 * (`parts[0].size === 24 + graph.length + model.length`) can be checked **without fetching a byte**
 * (container-v1 §8).
 */
const CONTAINER_HEADER_BYTES = 24;
/**
 * The sha256 of the empty byte string. A zero-length part (part 1 with an empty const region — ADR
 * 0109 decision 3) must declare this value. Any other value is an unreadable declaration, "a 0-byte
 * file with contents", so it fails loudly.
 */
const EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
/** The `format` major hub understands. An unknown major fails loudly (ADR 0109 decision 1). */
const FORMAT_MAJOR = 5;
/** Length limits of the display fields (ADR 0075 decision 1). */
const MAX_LABEL_CHARS = 64;
const MAX_DESCRIPTION_CHARS = 200;

const SHA256_RE = /^[0-9a-f]{64}$/;
const SEGMENT_RE = /^[A-Za-z0-9._-]+$/;
const FORMAT_RE = /^karume\/([1-9][0-9]*)$/;
const PIPELINE_RE = /^([A-Za-z0-9_-]+)\/([1-9][0-9]*)$/;
/**
 * The `revision` of a cross-repo reference (a commit SHA of 40 lowercase hex digits).
 *
 * MUST: reject branches, tags and short SHAs — a cross-repo reference is a field that "points at an
 * **immutable** component of another repo" (ADR 0038 §7), and allowing a mutable ref would undo, at
 * the target, the point of pinning our own revision to a SHA (the manifest and the weights coming
 * from the same commit). Short forms are ambiguous too, so they are not allowed.
 */
const COMMIT_SHA_RE = /^[0-9a-f]{40}$/;
/**
 * One segment (`owner` / `name`) of a cross-repo reference's `repo`. The same allowlisted
 * characters as paths, plus **a leading dot is forbidden** (`.` / `..` themselves also fail here).
 *
 * MUST: use an allowlist, not a denylist — the fetch layer treats the `/` of a repo as structural
 * and does not percent-encode it (`hfResolveUrl`), so a gap in a denylist would become traversal
 * outside the SHA pin as is (the same reason as the path check of ADR 0038 §2).
 */
const REPO_SEGMENT_RE = /^[A-Za-z0-9_-][A-Za-z0-9._-]*$/;

/**
 * Keys used for prototype pollution. JSON.parse puts `__proto__` as **an own property**, so
 * pollution itself does not happen, but they are rejected across the whole manifest to rule out
 * structurally the accident of downstream code re-reading them with a naive assignment (ADR 0038
 * §1).
 */
const FORBIDDEN_KEYS: readonly string[] = ["__proto__", "constructor", "prototype"];

const ENVELOPE_KEYS: readonly string[] = ["format", "generator", "defaultModel", "models"];
const MODEL_KEYS: readonly string[] = [
  "pipeline",
  "weights",
  "assets",
  "quants",
  "defaultQuant",
  "pipelineConfig",
];
const WEIGHT_DTYPE_KEYS: readonly string[] = ["container"];
const CONTAINER_KEYS: readonly string[] = ["descriptor", "parts"];
const DESCRIPTOR_KEYS: readonly string[] = ["graph", "model"];
const DOCUMENT_KEYS: readonly string[] = ["length", "sha256"];
const FILE_REF_KEYS: readonly string[] = ["path", "size", "sha256", "repo", "revision"];
const QUANT_KEYS: readonly string[] = [
  "weights",
  "session",
  "gpuFeatures",
  "label",
  "description",
  "requiredLimits",
];
const GPU_FEATURE_KEYS: readonly string[] = ["shaderF16"];

/**
 * The `session` allowlist (both keys and values — kept from ADR 0038 §3). Not a pass-through of the
 * runtime types.
 *
 * The value `i8a8` was renamed to `a8` in 0.5.0 (ADR 0074 decision 3). This knob decides **only how
 * activations are handled**; the storage layout of the weights is decided by the asset header —
 * spelling the storage layout into the value would keep it twice, in the manifest and in the
 * header, and with mixed assets (i4 + i8) it would be a lie that only one of them can be
 * accelerated.
 */
const LINEAR_COMPUTE: readonly LinearCompute[] = ["f32", "a8", "f16"];
const CONV_COMPUTE: readonly ConvCompute[] = ["f32", "a8"];
const ATTENTION_COMPUTE: readonly AttentionCompute[] = ["f32", "f16", "a8"];
const SCORE_STORAGE: readonly ScoreStorage[] = ["f32", "f16", "none"];
const LINEAR_GEMV_REDUCE = ["sequential", "parallel"] as const;
const STATE_ATTENTION_REDUCE = ["sequential", "parallel", "parallel-fused"] as const;
/**
 * The key allowlist of `session` — held as **an exhaustive table** of `Required<SessionSpec>`.
 *
 * MUST: it is bound exhaustively because if the allowlist, the type and `parseSession` drifted
 * apart, knobs that a distribution declares would be silently dropped (the same check as `WRITERS`
 * in `@karume/models`). Adding a key without a field in `SessionSpec` (as an extra field) or adding
 * a knob only to `SessionSpec` (as a missing field) both make this declaration fail type checking.
 */
const SESSION_KEYS: Readonly<Record<keyof Required<SessionSpec>, true>> = {
  linearCompute: true,
  convCompute: true,
  attentionCompute: true,
  attentionScoreStorage: true,
  linearGemvReduce: true,
  stateAttentionReduce: true,
  fuseRmsNormAdd: true,
  fuseLinearStaticQuantize: true,
  fuseGeluMulStaticQuantize: true,
  packedStaticQuantize: true,
  integerGemv: true,
  linearGemvPlanes: true,
};

export type LinearCompute = "f32" | "a8" | "f16";
/** The execution form of conv3d (ADR 0133 — `"a8"` = int8 activations per input pixel). */
export type ConvCompute = "f32" | "a8";
export type AttentionCompute = "f32" | "f16" | "a8";
/** `"none"` = the online kernel that never stores S (ADR 0132; implemented only with `attentionCompute: "a8"`). */
export type ScoreStorage = "f32" | "f16" | "none";

/**
 * The names of the device limits a quant requires (`requiredLimits`, which ADR 0038 §7 listed as a
 * reserved field).
 *
 * MUST: the vocabulary is **owned by the manifest**, not a pass-through of runtime types (the same
 * discipline as `session` — ADR 0038 §3). Keep the spelling one-to-one with `REQUIRED_LIMIT_KEYS`
 * of `@karume/runtime` (the requiredLimits keys `acquireGpu` passes to `requestDevice`). hub does
 * not depend on the runtime, so this is a copy, but **unknown names are rejected with a fail
 * loudly**, so a drift in spelling is never silently ignored (it is also the boundary that keeps
 * published manifests from being nailed to the runtime's internal spelling).
 */
const REQUIRED_LIMIT_NAMES = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxUniformBufferBindingSize",
  "maxStorageBuffersPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
] as const;

export type RequiredLimitName = (typeof REQUIRED_LIMIT_NAMES)[number];

/**
 * The table of **minimum values** of the limits a quant requires (`limit name → value`). It has the
 * same "name → minimum to satisfy" shape as WebGPU's `GPUDeviceDescriptor.requiredLimits`, so the
 * consumer's check is the plain comparison `adapter.limits[name] >= value`.
 *
 * Only the names written are requirements; a limit not written is "not required" (a partial
 * mapping).
 */
export type RequiredLimitsSpec = Readonly<Partial<Record<RequiredLimitName, number>>>;

/**
 * The triple of a file reference (ADR 0038 §2). The presence and format of all three are mandatory
 * checks at parse time.
 *
 * The checks apply **only on the path that parses manifest data**; this is a public type, so
 * callers can build one by hand (as the doc of {@link crossRefOf} also says). `openAsset`
 * (`fetch.ts`) does not check that a given ref belongs to the loaded manifest, so a hand-built ref
 * passes unchecked, triple consistency included.
 */
export type FileRef = {
  readonly path: string;
  /** The byte count of the raw stored form on the Hub. */
  readonly size: number;
  /** 64 lowercase hex digits. */
  readonly sha256: string;
  /**
   * The source repo of a cross-repo reference (`owner/name` — ADR 0038 §7). When omitted, the
   * manifest's own repo.
   *
   * MUST: appears only **together with** {@link revision} (one alone fails loudly).
   */
  readonly repo?: string;
  /**
   * The commit SHA of a cross-repo reference (always 40 lowercase hex digits). Paired with
   * {@link repo}.
   */
  readonly revision?: string;
};

/**
 * Extracts the declaration of a cross-repo reference (the **pair** of `repo` and `revision`). A
 * declaration with only one of them is not cross-repo (`undefined`) — parsing enforces the pair,
 * but {@link FileRef} is a public type that callers can build by hand.
 *
 * MUST: every "is it cross-repo" decision goes through this one predicate (deduplication keys, the
 * source branch, the naming in diagnostics) — spelling it per surface, a spelling that looks at
 * only one of the two would silently create the mismatch "the key claims cross-repo but the fetch
 * comes from the session's source" (the same file counted as two separate entries).
 */
export const crossRefOf = (
  ref: FileRef,
): { readonly repo: string; readonly revision: string } | undefined =>
  ref.repo === undefined || ref.revision === undefined
    ? undefined
    : { repo: ref.repo, revision: ref.revision };

/**
 * The identity key of a file reference. With cross-repo references (`repo` / `revision` of another
 * repo), **`path` alone no longer identifies one file** — a same-named path in another repo is
 * different bytes, and folding by path would make the triple-match check wrongly reject a valid
 * manifest and the fetch layer return another file's bytes. For a reference to our own repo the key
 * is `path` as is, so the behaviour up to v3 does not change.
 */
export const fileRefKey = (ref: FileRef): string => {
  const cross = crossRefOf(ref);
  return cross === undefined ? ref.path : `${cross.repo}@${cross.revision}/${ref.path}`;
};

/**
 * The **expectation** for one document a container embeds (graph descriptor / model descriptor)
 * (item (1) of container-v1 §7).
 *
 * A descriptor cannot prove its own correctness, so the outside (this manifest) holds the byte
 * length and sha256 of each of the two documents. They are **a separate fact** from the sha256 of
 * the part 0 file (`parts[0].sha256`), and the sha256 of the `graph` document is directly the
 * identity (content hash) of the `krg` (ADR 0109 decision 3).
 */
export type DocumentExpectation = {
  /** Byte length (0 < length <= {@link MAX_DESCRIPTOR_BYTES}). */
  readonly length: number;
  /** 64 lowercase hex digits. */
  readonly sha256: string;
};

/**
 * The entry point of one container (ADR 0109 decision 3). The reader (`openContainer`) needs only
 * two things, "the expectations for the two documents" and "the bytes of each part", so the
 * manifest holds these two fields.
 *
 * MUST: {@link parts} is **every part in index order** (first is part 0 = the descriptor file). No
 * reordering and no folding of duplicates — the index is the part's id, so touching the list breaks
 * the identifiers.
 * MUST: cross-repo references (`repo` / `revision`) are uniform **per container** (all own repo, or
 * all the same (repo, revision)). Parsing rejects a mix (ADR 0109 decision 3).
 *
 * NOTE: the file name convention (`<stem>-NNNNN-of-NNNNN.krm` — container-v1 §8) is the writer's;
 * hub does not check it.
 */
export type ContainerRef = {
  readonly descriptor: {
    readonly graph: DocumentExpectation;
    readonly model: DocumentExpectation;
  };
  /** Every part in index order, starting with part 0 (the descriptor file). */
  readonly parts: readonly FileRef[];
};

/**
 * One dtype of a weights entry = one container (`{ container }`). The successor of v4's
 * `{ shards, extras }`; `shards` / `extras` are retired (ADR 0109 decisions 3 / 4 — the actual
 * extras moved into assets inside the container).
 */
export type WeightContainer = {
  readonly container: ContainerRef;
};

/**
 * One weights entry = **dtype label → container**. The dtype label is vocabulary for selection and
 * display; the source of truth for storage is the descriptor's `encoding.codec` (ADR 0109 decision
 * 2) — hub checks only the completeness of the mapping.
 */
export type WeightEntry = Readonly<Record<string, WeightContainer>>;

/** The manifest-owned vocabulary of execution knobs (ADR 0038 §3 / 0098 / 0104). */
export type SessionSpec = {
  readonly linearCompute?: LinearCompute;
  readonly convCompute?: ConvCompute;
  readonly attentionCompute?: AttentionCompute;
  readonly attentionScoreStorage?: ScoreStorage;
  readonly linearGemvReduce?: "sequential" | "parallel";
  /**
   * The reduction form of (1) QK / (3) PV of states-shaped attention (ADR 0104 addendum 2026-09-26,
   * promotion to the vocabulary). Without a declaration the reader runs on the runtime's reference
   * path (`"sequential"`).
   */
  readonly stateAttentionReduce?: "sequential" | "parallel" | "parallel-fused";
  readonly fuseRmsNormAdd?: boolean;
  readonly fuseLinearStaticQuantize?: boolean;
  /** Folds gelu_tanh → mul → static_quantize into one dispatch (perf-ledger K-48). */
  readonly fuseGeluMulStaticQuantize?: boolean;
  /**
   * Packs the int8 activations of fixed SRQ into u32 and hands them to the parallel GEMV (ADR 0105
   * addendum 2, promotion to the vocabulary). The consumer rejects a declaration without
   * `linearGemvReduce: "parallel"`.
   */
  readonly packedStaticQuantize?: boolean;
  /**
   * Runs the eligible decode linears on the integer GEMV, which reads the packed int8
   * activations of fixed SRQ (the runtime's `SessionOptions.integerGemv`). The consumer rejects a
   * declaration without `packedStaticQuantize: true`.
   */
  readonly integerGemv?: boolean;
  /**
   * Runs the eligible i2 linears (the measured shapes — today the QAT E2B lm_head) on the byte-plane
   * GEMV, which reads f32 activations (the runtime's `SessionOptions.linearGemvPlanes`). Needs no
   * other key.
   */
  readonly linearGemvPlanes?: boolean;
};

/** GPU features needed before device creation (`shaderF16` only — ADR 0038 §3). */
export type GpuFeaturesSpec = {
  readonly shaderF16?: boolean;
};

/** A named quantization / precision mode (v1's Preset — renamed by ADR 0041 decision 3). */
export type Quant = {
  /** A **complete mapping** from weights name → its dtype label. */
  readonly weights: Readonly<Record<string, string>>;
  readonly session: SessionSpec;
  readonly gpuFeatures?: GpuFeaturesSpec;
  /**
   * A short display name for selection UIs (English, at most {@link MAX_LABEL_CHARS} characters —
   * ADR 0075 decision 1). The preset id (`i8-a8-attn8-s16`) serves the machine; this is the side
   * people read. When unset, callers show the id as is.
   */
  readonly label?: string;
  /**
   * A one-line description (English, at most {@link MAX_DESCRIPTION_CHARS} characters — ADR 0075
   * decision 1).
   */
  readonly description?: string;
  /**
   * The minimum device limits this preset requires (ADR 0038 §7).
   *
   * NOTE: hub owns acceptance, checking and exposure in the types; the reader is the family
   * admission of `@karume/models` (compared with the GPU's limits before a single weight byte is
   * fetched — ADR 0089 decision 5). The comparison target is `GpuContext.limits` on the path given
   * a shared GPU, and the adapter's measured values from `readAdapterLimits()` on the path that
   * acquires its own device.
   */
  readonly requiredLimits?: RequiredLimitsSpec;
};

/**
 * The contract name + major of a pipeline implementation (`anima/1`). Judging unknown majors is the
 * models side's job.
 */
export type PipelineId = {
  readonly name: string;
  readonly major: number;
};

/**
 * The declaration of one model. `pipeline` is **per model**, so a family repo does not break when
 * it mixes architectures (ADR 0041 decision 2).
 */
export type ModelEntry = {
  readonly pipeline: PipelineId;
  /** Tensor containers per dtype. */
  readonly weights: Readonly<Record<string, WeightEntry>>;
  /**
   * Unconditional files independent of the quant choice (tokenizer / symbols / style_vectors, …).
   */
  readonly assets: Readonly<Record<string, FileRef>>;
  readonly quants: Readonly<Record<string, Quant>>;
  readonly defaultQuant: string;
  /**
   * Owned by the pipeline — hub passes it through (schema validation is in each `@karume/models`
   * implementation).
   */
  readonly pipelineConfig: Readonly<Record<string, unknown>>;
  /** What can be chosen for this model (also used in errors). */
  readonly available: AvailableLabels;
};

export type Manifest = {
  /** `karume/<major>`. hub has checked the major. */
  readonly format: string;
  /** The version of the tool that wrote it (for matching bug reports; no execution semantics). */
  readonly generator: string;
  /** The model name used when no model is given (required — ADR 0041 decision 2). */
  readonly defaultModel: string;
  readonly models: Readonly<Record<string, ModelEntry>>;
  /**
   * What can be chosen across the repo (`quants` / `dtypes` are empty, as there is no model
   * context).
   */
  readonly available: AvailableLabels;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * The factory of check failures. During parsing, always go through it so that every error carries
 * `available`.
 */
type Fail = {
  format: (message: string, cause?: unknown) => ManifestFormatError;
  reference: (message: string) => ManifestReferenceError;
  path: (message: string, path: string) => ManifestPathError;
};

/**
 * The result of the diagnostic walk. `truncated` says whether a list of a table above its size
 * limit was cut off ({@link surveyLabels}).
 */
type LabelSurvey = {
  readonly available: AvailableLabels;
  readonly truncated: boolean;
};

/**
 * When cut off, the error message says so — on paths that fail an earlier check (unknown keys, …)
 * before the count check, the message says nothing about counts, so this keeps readers from
 * mistaking a partial `available` for the whole.
 */
const createFail = ({ available, truncated }: LabelSurvey): Fail => {
  const note = truncated
    ? " (available labels <available>: tables above their size limit were cut off at the limit count)"
    : "";
  return {
    format: (message, cause) => new ManifestFormatError(message + note, { available, cause }),
    reference: (message) => new ManifestReferenceError(message + note, { available }),
    path: (message, path) => new ManifestPathError(message + note, { available, path }),
  };
};

/**
 * Clears forbidden keys from the whole manifest (including inside `pipelineConfig`). Guarantees
 * once, at hub's entry, that implementations it passes data to cannot have an accident when
 * re-reading with a naive assignment.
 *
 * MUST: count the depth itself and fail at the limit ({@link MAX_MANIFEST_DEPTH}). Descending until
 * the stack runs out would escape as `RangeError`, a type outside the `HubError` contract — the
 * unexpected fails loudly with a type. This is the first full walk of parsing, so this check also
 * protects the walks downstream (`JSON.stringify`, …).
 */
const assertNoForbiddenKeys = (value: unknown, where: string, depth: number): void => {
  if (depth > MAX_MANIFEST_DEPTH) {
    throw new ManifestFormatError(
      `${where}: nesting exceeds the limit of ${MAX_MANIFEST_DEPTH} levels`,
    );
  }
  if (Array.isArray(value)) {
    value.forEach((entry, index) => assertNoForbiddenKeys(entry, `${where}[${index}]`, depth + 1));
    return;
  }
  if (!isRecord(value)) return;
  for (const key of Object.keys(value)) {
    if (FORBIDDEN_KEYS.includes(key)) {
      throw new ManifestFormatError(`${where}: forbidden key '${key}' found`);
    }
    assertNoForbiddenKeys(value[key], `${where}.${key}`, depth + 1);
  }
};

/**
 * Collects "what is available" on a best-effort basis before the checks run. A walk whose only
 * purpose is to let errors carry "then what does work" even for a broken manifest; it never throws.
 *
 * Passing `model` collects that model's quants / dtypes too (no mixing across models — recommending
 * a quant name of another model would mislead).
 *
 * MUST: this walk runs **before** the table count checks ({@link parseMap}), so put no work in it
 * heavier than linear in the input's counts — otherwise one manifest within 1MiB could stall the
 * main thread, and the DoS guard of the size limits (ADR 0041 decision 7) would be bypassed. Each
 * table's list is cut off at that table's size limit. The limit is also the largest legitimate
 * count, so a manifest that passes the checks always gets a complete `available`.
 */
const surveyLabels = (root: Record<string, unknown>, model?: string): LabelSurvey => {
  const modelsRaw = root["models"];
  if (!isRecord(modelsRaw)) return { available: NO_LABELS, truncated: false };
  const models = headKeys(modelsRaw, MAX_MODELS);
  const entry = model === undefined ? undefined : modelsRaw[model];
  if (!isRecord(entry)) {
    return {
      available: { models: models.keys, quants: [], dtypes: {} },
      truncated: models.truncated,
    };
  }
  const quantsRaw = entry["quants"];
  const quants = headKeys(isRecord(quantsRaw) ? quantsRaw : {}, MAX_QUANTS);
  const weightsRaw = entry["weights"];
  const weightsRecord = isRecord(weightsRaw) ? weightsRaw : {};
  const weights = headKeys(weightsRecord, MAX_WEIGHTS);
  // Build it in one go — copying an accumulating object with spread per entry would be quadratic in
  // the count.
  const dtypes = Object.fromEntries(
    weights.keys.flatMap((name) => {
      const labels = weightsRecord[name];
      return isRecord(labels) ? [[name, Object.keys(labels)]] : [];
    }),
  );
  return {
    available: { models: models.keys, quants: quants.keys, dtypes },
    truncated: models.truncated || quants.truncated || weights.truncated,
  };
};

/** Cuts a table's keys off at the first `max` (a part of the diagnostic walk — never throws). */
const headKeys = (
  record: Record<string, unknown>,
  max: number,
): { readonly keys: readonly string[]; readonly truncated: boolean } => {
  const keys = Object.keys(record);
  return { keys: keys.slice(0, max), truncated: keys.length > max };
};

const assertAllowedKeys = (
  fail: Fail,
  value: Record<string, unknown>,
  allowed: readonly string[],
  where: string,
): void => {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) {
      throw fail.reference(`${where}: unknown key '${key}' (allowed: ${allowed.join(" / ")})`);
    }
  }
};

/**
 * The **allowlist** check of paths (ADR 0038 §2, kept in v2 = ADR 0041 decision 6). The fetch layer
 * percent-encodes segments but lets dots through, so with a denylist any gap would become traversal
 * outside the SHA pin as is.
 */
const assertPath = (fail: Fail, path: string, where: string): void => {
  for (const segment of path.split("/")) {
    if (segment === "") {
      throw fail.path(
        `${where}: path '${path}' has an empty segment (leading / trailing / consecutive slash)`,
        path,
      );
    }
    if (segment.startsWith(".")) {
      throw fail.path(
        `${where}: path '${path}' has a segment with a leading dot '${segment}'`,
        path,
      );
    }
    if (!SEGMENT_RE.test(segment)) {
      throw fail.path(
        `${where}: path '${path}' segment '${segment}' does not match the allowed characters [A-Za-z0-9._-]`,
        path,
      );
    }
  }
};

/**
 * Whether a string is spelt as an HF repo name (`owner/name`) — applies the allowlist
 * {@link REPO_SEGMENT_RE} to exactly two segments. It is the raw string the fetch layer writes into
 * the URL, so the cross-repo reference check and the entry that takes HF repo references
 * (`loadManifest` in `fetch.ts`) share **this one predicate** (the accepted set is not split per
 * entry).
 */
export const isRepoName = (repo: string): boolean => {
  const segments = repo.split("/");
  return segments.length === 2 && segments.every((segment) => REPO_SEGMENT_RE.test(segment));
};

/** Checks the `repo` (`owner/name`) of a cross-repo reference ({@link isRepoName}). */
const assertRepo = (fail: Fail, repo: string, where: string): void => {
  if (!isRepoName(repo)) {
    throw fail.format(
      `${where}.repo: not the 2 segments of 'owner/name' / does not match the allowed characters [A-Za-z0-9._-] (no leading dot)` +
        `: '${repo}'`,
    );
  }
};

/** The context the file reference checks share across the whole manifest, for one parse. */
type RefScope = {
  /** Identity ({@link fileRefKey}) → the first ref seen (the table of the triple-match check). */
  readonly seen: Map<string, FileRef>;
  /**
   * The repo the manifest was loaded from (HF source only; `undefined` otherwise).
   *
   * MUST: reject a cross-repo reference that points at this repo itself — ref identity
   * ({@link fileRefKey}) includes the revision while HF's cached files do not, so pointing at
   * another revision of our own repo through a cross-repo reference would make "refs counted as
   * separate files" share one file (inventory protection breaks when one of them is deleted, and
   * the deduplication key of the fetch plan drifts from the consumer's). Files of our own repo can
   * be written without the cross-repo fields.
   */
  readonly ownRepo: string | undefined;
};

/**
 * Reads a cross-repo reference (`repo` + `revision`). **Legal only with both together**; a
 * declaration of one alone fails loudly — `repo` alone would leave "which commit" to a mutable ref
 * and break immutability, and `revision` alone would become a path that silently overrides our own
 * repo's resolved SHA (both are "fetching something without being able to read the declaration's
 * intent", so nothing is filled in by guessing).
 *
 * A cross-repo target that is the repo the manifest was loaded from (`ownRepo`) fails loudly (for
 * the reason see {@link RefScope.ownRepo}).
 */
const parseCrossRepo = (
  fail: Fail,
  raw: Record<string, unknown>,
  where: string,
  ownRepo: string | undefined,
): { readonly repo: string; readonly revision: string } | undefined => {
  const hasRepo = Object.hasOwn(raw, "repo");
  const hasRevision = Object.hasOwn(raw, "revision");
  if (!hasRepo && !hasRevision) return undefined;
  if (hasRepo !== hasRevision) {
    throw fail.format(
      `${where}: a cross-repo reference writes both 'repo' and 'revision' together` +
        ` (expected both / actual ${hasRepo ? "only 'repo'" : "only 'revision'"})`,
    );
  }
  const repo = raw["repo"];
  const revision = raw["revision"];
  if (typeof repo !== "string") throw fail.format(`${where}.repo: not a string`);
  if (typeof revision !== "string") throw fail.format(`${where}.revision: not a string`);
  assertRepo(fail, repo, where);
  if (!COMMIT_SHA_RE.test(revision)) {
    throw fail.format(
      `${where}.revision: must be a 40-digit lowercase hex commit SHA` +
        ` (no branches, tags or short forms): '${revision}'`,
    );
  }
  if (repo === ownRepo) {
    throw fail.format(
      `${where}.repo: the cross-repo reference points at '${repo}', the repo the manifest was loaded from` +
        ` (files of the own repo are referenced without 'repo' / 'revision' — ADR 0038 addendum 2026-10-11)`,
    );
  }
  return { repo, revision };
};

/**
 * Checks and reads the triple. Duplicate refs to the same file are legal (sharing between models
 * rests on this alone — ADR 0041 decision 5) but must match `{size, sha256}` exactly (a
 * contradictory manifest makes self-heal oscillate, evicting a correct cache over and over).
 * Identity is judged by {@link fileRefKey} (a cross-repo reference is one file including its repo
 * and revision).
 */
const parseFileRef = (
  fail: Fail,
  raw: unknown,
  where: string,
  scope: RefScope,
  { allowEmpty = false }: { readonly allowEmpty?: boolean } = {},
): FileRef => {
  if (!isRecord(raw)) throw fail.format(`${where}: file reference is not an object`);
  assertAllowedKeys(fail, raw, FILE_REF_KEYS, where);
  const path = raw["path"];
  const size = raw["size"];
  const sha256 = raw["sha256"];
  if (typeof path !== "string") throw fail.format(`${where}: 'path' missing / not a string`);
  if (typeof size !== "number") throw fail.format(`${where}: 'size' missing / not a number`);
  if (typeof sha256 !== "string") throw fail.format(`${where}: 'sha256' missing / not a string`);
  if (!SHA256_RE.test(sha256)) {
    throw fail.format(`${where}: 'sha256' must be 64 lowercase hex digits: '${sha256}'`);
  }
  // Empty is allowed only for index 1 and above of `container.parts` (a zero-length part — ADR 0109
  // decision 3). In any other field `size: 0` is "a ref that brings no bytes when fetched", so it
  // is rejected as before.
  const floor = allowEmpty ? 0 : 1;
  if (!Number.isSafeInteger(size) || size < floor || size > MAX_FILE_BYTES) {
    throw fail.format(
      `${where}: 'size' must be a safe integer with ${floor} <= size <= ${MAX_FILE_BYTES}` +
        `: ${size}`,
    );
  }
  // There is only one zero-length file, so the sha256 it declares is fixed too (ADR 0109 decision
  // 3).
  if (size === 0 && sha256 !== EMPTY_SHA256) {
    throw fail.format(
      `${where}: the 'sha256' of a reference with 'size' 0 must be the empty-string value ${EMPTY_SHA256}` +
        `: '${sha256}'`,
    );
  }
  assertPath(fail, path, where);
  const crossRepo = parseCrossRepo(fail, raw, where, scope.ownRepo);
  const ref: FileRef = { path, size, sha256, ...crossRepo };
  const key = fileRefKey(ref);
  const previous = scope.seen.get(key);
  if (previous !== undefined) {
    if (previous.size !== size || previous.sha256 !== sha256) {
      throw fail.reference(
        `${where}: the {size, sha256} of duplicate reference '${key}' disagree ` +
          `({${previous.size}, ${previous.sha256}} and {${size}, ${sha256}})`,
      );
    }
    return previous;
  }
  scope.seen.set(key, ref);
  return ref;
};

/**
 * The expectation for one descriptor document (length + sha256). The length ceiling applies to each
 * document independently.
 */
const parseDocument = (fail: Fail, raw: unknown, where: string): DocumentExpectation => {
  if (!isRecord(raw)) throw fail.format(`${where}: missing / not an object`);
  assertAllowedKeys(fail, raw, DOCUMENT_KEYS, where);
  const length = raw["length"];
  const sha256 = raw["sha256"];
  if (typeof length !== "number") throw fail.format(`${where}: 'length' missing / not a number`);
  if (!Number.isSafeInteger(length) || length <= 0 || length > MAX_DESCRIPTOR_BYTES) {
    throw fail.format(
      `${where}: 'length' must be a safe integer with 0 < length <= ${MAX_DESCRIPTOR_BYTES}` +
        `: ${length}`,
    );
  }
  if (typeof sha256 !== "string") throw fail.format(`${where}: 'sha256' missing / not a string`);
  if (!SHA256_RE.test(sha256)) {
    throw fail.format(`${where}: 'sha256' must be 64 lowercase hex digits: '${sha256}'`);
  }
  return { length, sha256 };
};

/**
 * Reads the part list in declaration order. No reordering and no folding of duplicates — the index
 * is the part's id, so touching the list breaks the identifiers (only the triple match of identical
 * paths is checked across the whole manifest by the table of {@link parseFileRef}).
 *
 * The byte limit ({@link MAX_PART_BYTES}) applies here — a check of each part on its own, with no
 * exception by position (first / last). It is not pushed down into {@link parseFileRef} because
 * that function is shared with `assets` (the limit belongs only to the part-splitting contract).
 */
const parseParts = (
  fail: Fail,
  raw: unknown,
  where: string,
  scope: RefScope,
): readonly FileRef[] => {
  if (!Array.isArray(raw)) throw fail.format(`${where}: missing / not an array`);
  if (raw.length < 2) {
    throw fail.format(
      `${where}: ${raw.length} entries (at least 2 are required: part 0 <descriptor> + part 1 <const>` +
        ` — container-v1 §8)`,
    );
  }
  if (raw.length > MAX_PARTS) {
    throw fail.format(`${where}: ${raw.length} entries exceed the limit of ${MAX_PARTS}`);
  }
  return raw.map((entry, index) => {
    // Zero length only in the shape "part 1 with an empty const region" (part 0 always holds the
    // header + the two documents).
    const ref = parseFileRef(fail, entry, `${where}[${index}]`, scope, { allowEmpty: index > 0 });
    if (ref.size > MAX_PART_BYTES) {
      throw fail.format(
        `${where}[${index}]: part '${ref.path}' is ${ref.size} bytes and` +
          ` exceeds the limit of ${MAX_PART_BYTES} (the part length ceiling — container-v1 §10)`,
      );
    }
    return ref;
  });
};

/**
 * Reads the entry of one container. Only **what closes on the declarations alone** is checked here
 * (ADR 0109 decision 6) — descriptor lengths / part count and ceiling / length of part 0 / uniform
 * cross-repo references.
 */
const parseContainer = (fail: Fail, raw: unknown, where: string, scope: RefScope): ContainerRef => {
  if (!isRecord(raw)) throw fail.format(`${where}: missing / not an object`);
  assertAllowedKeys(fail, raw, CONTAINER_KEYS, where);
  const descriptorRaw = raw["descriptor"];
  if (!isRecord(descriptorRaw)) throw fail.format(`${where}.descriptor: missing / not an object`);
  assertAllowedKeys(fail, descriptorRaw, DESCRIPTOR_KEYS, `${where}.descriptor`);
  const graph = parseDocument(fail, descriptorRaw["graph"], `${where}.descriptor.graph`);
  // A krm always has a model descriptor (karume/5 has no field that distributes a graph-only `krg`
  // — ADR 0109 decision 5).
  const model = parseDocument(fail, descriptorRaw["model"], `${where}.descriptor.model`);
  const parts = parseParts(fail, raw["parts"], `${where}.parts`, scope);

  // Part 0 is "header + exactly the two documents" (container-v1 §8). It is consistency between
  // declarations, so it can be checked here without fetching a byte — never noticing a length
  // mismatch only after fetching.
  const part0 = CONTAINER_HEADER_BYTES + graph.length + model.length;
  if (parts[0].size !== part0) {
    throw fail.format(
      `${where}.parts[0]: the size ${parts[0].size} of part 0 differs from` +
        ` header ${CONTAINER_HEADER_BYTES} + graph descriptor ${graph.length} +` +
        ` model descriptor ${model.length} = ${part0} (container-v1 §8)`,
    );
  }

  // Cross-repo references are **per container** (ADR 0109 decision 3). One-sided or mixed
  // references would fetch the remaining parts from the session's repo, where they do not exist,
  // failing only in the middle of the fetch.
  const head = crossRefOf(parts[0]);
  for (const [index, ref] of parts.entries()) {
    const cross = crossRefOf(ref);
    const agrees =
      head === undefined
        ? cross === undefined
        : cross !== undefined && cross.repo === head.repo && cross.revision === head.revision;
    if (!agrees) {
      throw fail.format(
        `${where}.parts[${index}]: cross-repo references are per container — ADR 0109 decision 3` +
          ` (part 0 is ${head === undefined ? "own repo" : `${head.repo}@${head.revision}`} /` +
          ` this part is ${cross === undefined ? "own repo" : `${cross.repo}@${cross.revision}`})`,
      );
    }
  }
  return { descriptor: { graph, model }, parts };
};

const parseWeightContainer = (
  fail: Fail,
  raw: unknown,
  where: string,
  scope: RefScope,
): WeightContainer => {
  if (!isRecord(raw)) throw fail.format(`${where}: dtype entry is not an object`);
  assertAllowedKeys(fail, raw, WEIGHT_DTYPE_KEYS, where);
  return { container: parseContainer(fail, raw["container"], `${where}.container`, scope) };
};

/** One weights entry (dtype label → container). **At least one dtype key is required**. */
const parseWeightEntry = (
  fail: Fail,
  raw: unknown,
  where: string,
  scope: RefScope,
): WeightEntry => {
  if (!isRecord(raw)) throw fail.format(`${where}: weights entry is not an object`);
  const labels = Object.keys(raw);
  if (labels.length === 0) {
    throw fail.format(
      `${where}: empty (at least one dtype label is required — dtype keys are mandatory)`,
    );
  }
  // Build it in one go — the label count of one entry has no count limit of its own (only the
  // manifest's byte limit bounds it), so copying an accumulating object with spread per label would
  // be quadratic in it. `map` keeps the label order of the checks (the first bad label throws).
  return Object.fromEntries(
    labels.map((label) => [
      label,
      parseWeightContainer(fail, raw[label], `${where}.${label}`, scope),
    ]),
  );
};

const readEnum = <T extends string>(
  fail: Fail,
  raw: Record<string, unknown>,
  key: string,
  values: readonly T[],
  where: string,
): T | undefined => {
  if (!Object.hasOwn(raw, key)) return undefined;
  const value = raw[key];
  if (typeof value !== "string") throw fail.format(`${where}.${key}: not a string`);
  const found = values.find((candidate) => candidate === value);
  if (found === undefined) {
    throw fail.reference(
      `${where}.${key}: unknown value '${value}' (allowed: ${values.join(" / ")})`,
    );
  }
  return found;
};

/** Keeps an explicit false, and does not treat null or numbers as unspecified. */
const readBoolean = (
  fail: Fail,
  raw: Record<string, unknown>,
  key: string,
  where: string,
): boolean | undefined => {
  if (!Object.hasOwn(raw, key)) return undefined;
  const value = raw[key];
  if (typeof value !== "boolean") throw fail.format(`${where}.${key}: not a boolean`);
  return value;
};

const parseSession = (fail: Fail, raw: unknown, where: string): SessionSpec => {
  if (raw === undefined) return {};
  if (!isRecord(raw)) throw fail.format(`${where}.session: not an object`);
  assertAllowedKeys(fail, raw, Object.keys(SESSION_KEYS), `${where}.session`);
  const at = `${where}.session`;
  const linearCompute = readEnum(fail, raw, "linearCompute", LINEAR_COMPUTE, at);
  const convCompute = readEnum(fail, raw, "convCompute", CONV_COMPUTE, at);
  const attentionCompute = readEnum(fail, raw, "attentionCompute", ATTENTION_COMPUTE, at);
  const attentionScoreStorage = readEnum(fail, raw, "attentionScoreStorage", SCORE_STORAGE, at);
  const linearGemvReduce = readEnum(fail, raw, "linearGemvReduce", LINEAR_GEMV_REDUCE, at);
  const stateAttentionReduce = readEnum(
    fail,
    raw,
    "stateAttentionReduce",
    STATE_ATTENTION_REDUCE,
    at,
  );
  const fuseRmsNormAdd = readBoolean(fail, raw, "fuseRmsNormAdd", at);
  const fuseLinearStaticQuantize = readBoolean(fail, raw, "fuseLinearStaticQuantize", at);
  const fuseGeluMulStaticQuantize = readBoolean(fail, raw, "fuseGeluMulStaticQuantize", at);
  const packedStaticQuantize = readBoolean(fail, raw, "packedStaticQuantize", at);
  const integerGemv = readBoolean(fail, raw, "integerGemv", at);
  const linearGemvPlanes = readBoolean(fail, raw, "linearGemvPlanes", at);
  return {
    ...(linearCompute === undefined ? {} : { linearCompute }),
    ...(convCompute === undefined ? {} : { convCompute }),
    ...(attentionCompute === undefined ? {} : { attentionCompute }),
    ...(attentionScoreStorage === undefined ? {} : { attentionScoreStorage }),
    ...(linearGemvReduce === undefined ? {} : { linearGemvReduce }),
    ...(stateAttentionReduce === undefined ? {} : { stateAttentionReduce }),
    ...(fuseRmsNormAdd === undefined ? {} : { fuseRmsNormAdd }),
    ...(fuseLinearStaticQuantize === undefined ? {} : { fuseLinearStaticQuantize }),
    ...(fuseGeluMulStaticQuantize === undefined ? {} : { fuseGeluMulStaticQuantize }),
    ...(packedStaticQuantize === undefined ? {} : { packedStaticQuantize }),
    ...(integerGemv === undefined ? {} : { integerGemv }),
    ...(linearGemvPlanes === undefined ? {} : { linearGemvPlanes }),
  };
};

const parseGpuFeatures = (fail: Fail, raw: unknown, where: string): GpuFeaturesSpec | undefined => {
  if (raw === undefined) return undefined;
  if (!isRecord(raw)) throw fail.format(`${where}.gpuFeatures: not an object`);
  assertAllowedKeys(fail, raw, GPU_FEATURE_KEYS, `${where}.gpuFeatures`);
  if (!Object.hasOwn(raw, "shaderF16")) return {};
  const shaderF16 = raw["shaderF16"];
  if (typeof shaderF16 !== "boolean") {
    throw fail.format(`${where}.gpuFeatures.shaderF16: not a boolean`);
  }
  return { shaderF16 };
};

/**
 * Reads a quant's weights mapping. Checks at run time too that it is a complete mapping **from keys
 * that exist in weights → dtype labels that exist in that entry** (types alone cannot constrain the
 * distributed JSON — ADR 0041 decision 3).
 */
const parseQuantWeights = (
  fail: Fail,
  raw: unknown,
  where: string,
  weights: Readonly<Record<string, WeightEntry>>,
): Readonly<Record<string, string>> => {
  if (!isRecord(raw)) throw fail.format(`${where}.weights: missing / not an object`);
  let mapping: Readonly<Record<string, string>> = {};
  for (const name of Object.keys(raw)) {
    if (!Object.hasOwn(weights, name)) {
      throw fail.reference(
        `${where}.weights: unknown weights '${name}'` +
          ` (available: ${Object.keys(weights).join(" / ")})`,
      );
    }
    const label = raw[name];
    if (typeof label !== "string") {
      throw fail.format(`${where}.weights.${name}: dtype label is not a string`);
    }
    const entry = weights[name];
    if (!Object.hasOwn(entry, label)) {
      throw fail.reference(
        `${where}.weights: '${name}' has no dtype '${label}'` +
          ` (available: ${Object.keys(entry).join(" / ")})`,
      );
    }
    mapping = { ...mapping, [name]: label };
  }
  for (const name of Object.keys(weights)) {
    if (!Object.hasOwn(mapping, name)) {
      throw fail.reference(
        `${where}.weights: '${name}' has no dtype assigned (a complete mapping is required)`,
      );
    }
  }
  return mapping;
};

/**
 * Reads a display field (ADR 0075 decision 1). **It only checks the length limit and does not
 * interpret the meaning** (decision 2 — whether a description matches reality cannot be checked,
 * and is not; display is the caller's responsibility). The limit is a boundary check that comes
 * from the premise "the manifest is external input".
 *
 * NOTE: length counts **code points** (not UTF-16 code units) — so that strings that look the same
 * do not pass or fail depending on how they are spelt (surrogate pairs, combining characters).
 */
const parseText = (
  fail: Fail,
  raw: unknown,
  key: string,
  max: number,
  where: string,
): string | undefined => {
  if (raw === undefined) return undefined;
  if (typeof raw !== "string") {
    throw fail.format(`${where}.${key}: expected string / actual ${typeof raw}`);
  }
  const length = Array.from(raw).length;
  if (length > max) {
    throw fail.format(
      `${where}.${key}: expected at most ${max} characters / actual ${length} characters`,
    );
  }
  return raw;
};

/**
 * Reads `requiredLimits` (limit name → minimum to satisfy). Names must be in the allowlist
 * ({@link REQUIRED_LIMIT_NAMES}); values must be positive safe integers.
 *
 * MUST: do not silently ignore unknown names — a misspelt device limit name shows up as **a silent
 * ceiling** (a limit not requested falls back to the spec default), so ignoring it would make
 * "declared but not in effect" the silent norm.
 */
const parseRequiredLimits = (
  fail: Fail,
  raw: unknown,
  where: string,
): RequiredLimitsSpec | undefined => {
  if (raw === undefined) return undefined;
  const at = `${where}.requiredLimits`;
  if (!isRecord(raw)) throw fail.format(`${at}: not an object`);
  assertAllowedKeys(fail, raw, REQUIRED_LIMIT_NAMES, at);
  let limits: RequiredLimitsSpec = {};
  for (const name of Object.keys(raw)) {
    const value = raw[name];
    if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
      throw fail.format(
        `${at}.${name}: expected positive safe integer / actual ${JSON.stringify(value) ?? typeof value}`,
      );
    }
    limits = { ...limits, [name]: value };
  }
  return limits;
};

const parseQuant = (
  fail: Fail,
  raw: unknown,
  where: string,
  weights: Readonly<Record<string, WeightEntry>>,
): Quant => {
  if (!isRecord(raw)) throw fail.format(`${where}: quant is not an object`);
  assertAllowedKeys(fail, raw, QUANT_KEYS, where);
  const mapping = parseQuantWeights(fail, raw["weights"], where, weights);
  const session = parseSession(fail, raw["session"], where);
  const gpuFeatures = parseGpuFeatures(fail, raw["gpuFeatures"], where);
  const label = parseText(fail, raw["label"], "label", MAX_LABEL_CHARS, where);
  const description = parseText(
    fail,
    raw["description"],
    "description",
    MAX_DESCRIPTION_CHARS,
    where,
  );
  const requiredLimits = parseRequiredLimits(fail, raw["requiredLimits"], where);
  return {
    weights: mapping,
    session,
    ...(gpuFeatures === undefined ? {} : { gpuFeatures }),
    ...(label === undefined ? {} : { label }),
    ...(description === undefined ? {} : { description }),
    ...(requiredLimits === undefined ? {} : { requiredLimits }),
  };
};

const parsePipeline = (fail: Fail, raw: unknown, where: string): PipelineId => {
  if (typeof raw !== "string") throw fail.format(`${where}.pipeline: missing / not a string`);
  const matched = PIPELINE_RE.exec(raw);
  if (matched === null) {
    throw fail.format(`${where}.pipeline: not of the form '<name>/<major>': '${raw}'`);
  }
  // The regular expression does not bound the digit count, so check for a safe integer after
  // conversion — otherwise 400 digits become `Infinity`, and above 2^53 a major rounded to a
  // neighbouring integer reaches the models side's decision as "well-formed", making the error
  // message name a value the manifest does not contain.
  const major = Number(matched[2]);
  if (!Number.isSafeInteger(major)) {
    throw fail.format(
      `${where}.pipeline: major is outside the safe integer range (at most ${Number.MAX_SAFE_INTEGER}): '${raw}'`,
    );
  }
  return { name: matched[1], major };
};

const parseFormat = (fail: Fail, raw: unknown): string => {
  if (typeof raw !== "string") throw fail.format("format: missing / not a string");
  const matched = FORMAT_RE.exec(raw);
  if (matched === null) {
    throw fail.format(`format: not 'karume/<major>' (mixed-up file): '${raw}'`);
  }
  const major = Number(matched[1]);
  if (major !== FORMAT_MAJOR) {
    throw fail.format(
      `format: unsupported major '${raw}' — this version reads only karume/${FORMAT_MAJOR}` +
        ` (there is no parser for older versions; re-upload the distribution as karume/${FORMAT_MAJOR},` +
        ` or use a version of @karume/hub that can read that manifest)`,
    );
  }
  return raw;
};

const parsePipelineConfig = (
  fail: Fail,
  raw: unknown,
  where: string,
): Readonly<Record<string, unknown>> => {
  if (!isRecord(raw)) {
    throw fail.format(
      `${where}.pipelineConfig: missing / not an object (write {} explicitly even when empty)`,
    );
  }
  const bytes = new TextEncoder().encode(JSON.stringify(raw)).length;
  if (bytes > MAX_PIPELINE_CONFIG_BYTES) {
    throw fail.format(
      `${where}.pipelineConfig: ${bytes} bytes exceed the limit of ${MAX_PIPELINE_CONFIG_BYTES}`,
    );
  }
  return raw;
};

/**
 * Reads a named map with a limit. `allowEmpty` is used only for a field that "must be declared but
 * may be empty" (`assets` — a weights-only model is not forced to invent non-empty assets).
 */
const parseMap = <T>(
  fail: Fail,
  raw: unknown,
  where: string,
  max: number,
  parseEntry: (name: string, value: unknown) => T,
  allowEmpty = false,
): Readonly<Record<string, T>> => {
  if (!isRecord(raw)) throw fail.format(`${where}: missing / not an object`);
  const names = Object.keys(raw);
  if (names.length === 0 && !allowEmpty)
    throw fail.format(`${where}: empty (at least one is required)`);
  if (names.length > max) {
    throw fail.format(`${where}: ${names.length} entries exceed the limit of ${max}`);
  }
  let parsed: Readonly<Record<string, T>> = {};
  for (const name of names) {
    parsed = { ...parsed, [name]: parseEntry(name, raw[name]) };
  }
  return parsed;
};

const parseModel = (
  root: Record<string, unknown>,
  name: string,
  raw: unknown,
  scope: RefScope,
): ModelEntry => {
  const survey = surveyLabels(root, name);
  const { available } = survey;
  const fail = createFail(survey);
  const where = `models.${name}`;
  if (!isRecord(raw)) throw fail.format(`${where}: model entry is not an object`);
  assertAllowedKeys(fail, raw, MODEL_KEYS, where);

  const pipeline = parsePipeline(fail, raw["pipeline"], where);
  const weights = parseMap(
    fail,
    raw["weights"],
    `${where}.weights`,
    MAX_WEIGHTS,
    (weightName, value) => parseWeightEntry(fail, value, `${where}.weights.${weightName}`, scope),
  );
  const assets = parseMap(
    fail,
    raw["assets"],
    `${where}.assets`,
    MAX_ASSETS,
    (assetName, value) => parseFileRef(fail, value, `${where}.assets.${assetName}`, scope),
    true,
  );
  const quants = parseMap(fail, raw["quants"], `${where}.quants`, MAX_QUANTS, (quantName, value) =>
    parseQuant(fail, value, `${where}.quants.${quantName}`, weights),
  );

  const defaultQuant = raw["defaultQuant"];
  if (typeof defaultQuant !== "string") {
    throw fail.format(`${where}.defaultQuant: missing / not a string`);
  }
  if (!Object.hasOwn(quants, defaultQuant)) {
    throw fail.reference(
      `${where}.defaultQuant: '${defaultQuant}' is not in quants` +
        ` (available: ${Object.keys(quants).join(" / ")})`,
    );
  }
  return {
    pipeline,
    weights,
    assets,
    quants,
    defaultQuant,
    pipelineConfig: parsePipelineConfig(fail, raw["pipelineConfig"], where),
    available,
  };
};

/**
 * Checks and reads the text of `karume.json`. Runs **every** check of ADR 0041 at parse time and
 * throws without starting any fetch if even one fails (not allowing "missing pieces found only
 * after the download started" is the very purpose of introducing the manifest).
 *
 * NOTE: the text alone does not say "whose repo's manifest this is", so the check that no
 * cross-repo reference points at the loading repo itself ({@link parseRepoManifest}) does not apply
 * here. The fetch entry (`loadManifest`) reads through that one, with the source's repo attached.
 */
export const parseManifest = (text: string): Manifest => parseRepoManifest(text, undefined);

/**
 * {@link parseManifest} with the repo the manifest was loaded from (`ownRepo`) attached. A
 * cross-repo reference pointing at `ownRepo` itself fails loudly ({@link RefScope.ownRepo}). A
 * source without the notion of a repo (a local directory) passes `undefined`.
 *
 * MUST NOT: export it from the barrel (`mod.ts`) — it is an entry inside the fetch layer
 * (`loadManifest` in `fetch.ts`).
 */
export const parseRepoManifest = (text: string, ownRepo: string | undefined): Manifest => {
  const bytes = new TextEncoder().encode(text).length;
  if (bytes > MAX_MANIFEST_BYTES) {
    throw new ManifestFormatError(
      `manifest: ${bytes} bytes exceed the limit of ${MAX_MANIFEST_BYTES}`,
      { available: NO_LABELS },
    );
  }
  let root: unknown;
  try {
    root = JSON.parse(text);
  } catch (error) {
    throw new ManifestFormatError(`manifest: cannot be read as JSON`, { cause: error });
  }
  if (!isRecord(root)) throw new ManifestFormatError("manifest: the top level is not an object");
  assertNoForbiddenKeys(root, "manifest", 0);

  const survey = surveyLabels(root);
  const { available } = survey;
  const fail = createFail(survey);
  // MUST: check `format` **before** the unknown-key check. A v1 manifest spells its top level
  // entirely differently, so in the reverse order an off-target diagnostic `unknown key 'components'`
  // would appear and hide the real reason (this version reads only karume/5) (ADR 0109
  // decision 1). The difference from v4 is only inside the dtype entries, so there it becomes the
  // minor diagnostic `unknown key 'shards'`.
  const format = parseFormat(fail, root["format"]);
  assertAllowedKeys(fail, root, ENVELOPE_KEYS, "manifest");
  const generator = root["generator"];
  if (typeof generator !== "string" || generator === "") {
    throw fail.format("generator: missing / not a non-empty string");
  }

  // MUST: the duplicate-path check shares one table across the whole manifest — sharing between
  // models works just by "writing the same path" (ADR 0041 decision 5), so a table per model would
  // miss contradictions.
  const scope: RefScope = { seen: new Map<string, FileRef>(), ownRepo };
  const models = parseMap(fail, root["models"], "models", MAX_MODELS, (name, value) =>
    parseModel(root, name, value, scope),
  );

  const defaultModel = root["defaultModel"];
  if (typeof defaultModel !== "string") throw fail.format("defaultModel: missing / not a string");
  if (!Object.hasOwn(models, defaultModel)) {
    throw fail.reference(
      `defaultModel: '${defaultModel}' is not in models` +
        ` (available: ${Object.keys(models).join(" / ")})`,
    );
  }
  return { format, generator, defaultModel, models, available };
};
