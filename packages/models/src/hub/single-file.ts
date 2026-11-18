/**
 * Internal mechanism that turns a user's single weights file into the converted components of the
 * shared loader (`components.ts`, {@link ConvertedComponent}) — the family-neutral part of every
 * family's `fromSingleFile`, the counterpart of diffusers' `from_single_file`. A family adds the
 * method and a role table ({@link SingleFileRole}); everything else is here.
 *
 * MUST: not exported from the barrel (the same standing as `components.ts`).
 *
 * The order, before a single weight part is fetched (the loader then fetches the descriptors and
 * runs the plan checks, also before any weight part):
 *
 * 1. The distribution's name map (`nameMap` of manifest `karume/6`), fetched through the hub like
 *    any file reference and read with `parseNameMap`. A distribution without one is refused.
 * 2. The file's format, told from its first bytes (only safetensors is read), and its header
 *    (`openSafetensors` — two reads; no tensor byte).
 * 3. The placement (`placeTensors`): a key no component takes is refused, and so is a component of
 *    the map this family does not take from a file.
 * 4. Which role comes from where ({@link SingleFileRole}), the tensor lists and the target
 *    containers ({@link SingleFileFamily}: a float storage is the plan of the family's float
 *    container with its 16-bit float weights stored as the file's dtype); the plan checks are handed
 *    to the loader ({@link ConvertedComponent.checkPlan}).
 *
 * SPEC: docs/spec/single-file.md
 */

import type { CodecName } from "@karume/runtime";
import {
  type ContainerRef,
  fetchAssets,
  type FetchAssetsOptions,
  type LoadedManifest,
  type ModelEntry,
  openContainerSource,
  prefetchAssets,
  type ResolvedSelection,
} from "@karume/hub";
import {
  ENCODED_CODECS,
  type Float16Codec,
  type NameMap,
  parseNameMap,
  placeTensors,
  type PlannedCodec,
  readPlanCodecs,
  type SourceTensor,
  type TensorDtype,
  type TensorPlacement,
} from "@karume/convert";
import { openSafetensors, type SafetensorsFile } from "@karume/safetensors";
import type { ConvertedComponent } from "./components.ts";

/**
 * A weights file the caller holds: a `Blob` (a browser `File` is one) or a ranged reader. The
 * pipeline reads it for as long as it lives — a Session built later reads its weights from it.
 */
export type SingleFileSource =
  | Blob
  | {
      /** The file's length in bytes. */
      readonly size: number;
      /** Bytes `[offset, offset + length)` of the file. */
      read(offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
    };

/**
 * How a family takes one component from a single file. A role the table does not list always comes
 * from the distribution.
 */
export type SingleFileRole = {
  readonly role: string;
  /**
   * `required`: the file must hold it (unless the caller replaces the role); `optional`: from the
   * file when the file holds it, otherwise from the distribution.
   */
  readonly presence: "required" | "optional";
  /**
   * The container of the selected model the file's tensors are converted against. A preset decides
   * what is quantized; what it does not quantize keeps the precision of its source:
   *
   * - `"quant"`: the selected quant decides. A quant that stores the role quantized (its label is
   *   not {@link SingleFileFamily.floatLabel}) gives its own container, which converts from any
   *   float dtype; a quant that stores it as a float gives its own container too — the float one —
   *   with its 16-bit float weights stored as the file's dtype, run with the quant's Session
   *   options (declared for that very container and its graph).
   * - `"file-dtype"`: the float container whatever the quant, its 16-bit float weights stored as
   *   the file's dtype — for a component no preset quantizes.
   */
  readonly storage: "quant" | "file-dtype";
};

/** A family's single-file declaration: its roles and its float storage. */
export type SingleFileFamily = {
  readonly roles: readonly SingleFileRole[];
  /**
   * The weights label of the distribution's float container of every role of the table: the plan a
   * role stored as a float is converted against, whatever the file's float dtype. No container of
   * another label is read for a float storage, whether the distribution has one or not.
   */
  readonly floatLabel: string;
  /**
   * The 16-bit float codec each float dtype is stored as, its own bits (the float container's f16
   * and bf16 weights are stored as it; both take two bytes an element, so no block moves). A dtype
   * with no codec cannot be stored as it is: a role whose storage is chosen by its dtype is refused
   * for it.
   */
  readonly floatCodecs: Readonly<Partial<Record<TensorDtype, Float16Codec>>>;
};

/** The first bytes the format is told from (the 8-byte header length and the header's first byte). */
const SNIFF_BYTES = 16;

/** Whether `bytes` starts with the ASCII characters of `text`. */
const startsWith = (bytes: Uint8Array<ArrayBuffer>, text: string): boolean => {
  for (let index = 0; index < text.length; index += 1) {
    if (bytes[index] !== text.charCodeAt(index)) return false;
  }
  return true;
};

/**
 * What the first bytes of a file are, when they are not a safetensors file's (`undefined`: they
 * are): the formats a model's weights are commonly saved in, else the bytes themselves.
 */
const notSafetensors = (bytes: Uint8Array<ArrayBuffer>, size: number): string | undefined => {
  if (startsWith(bytes, "GGUF")) return "the file is a GGUF file (it starts with 'GGUF')";
  if (startsWith(bytes, "PK\u0003\u0004")) {
    return "the file is a zip archive (a PyTorch checkpoint written by torch.save is one)";
  }
  if (bytes[0] === 0x80 && bytes[1] >= 2 && bytes[1] <= 5) {
    return "the file is a Python pickle (a PyTorch checkpoint of the legacy format is one)";
  }
  // safetensors: an unsigned 64-bit little-endian header length, then the header's JSON object.
  if (size > 8 && bytes[8] === "{".charCodeAt(0)) return undefined;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(" ");
  return size <= 8
    ? `the file is ${size} bytes long, too short for a safetensors header`
    : `the file's first bytes are ${hex} (a safetensors file has the '{' of its header at byte 8)`;
};

/**
 * Opens the file: its format from its first bytes, then its header. A file that is not safetensors
 * is refused naming what was seen; a malformed safetensors file is refused by the reader.
 */
const openSingleFile = async (where: string, file: SingleFileSource): Promise<SafetensorsFile> => {
  const length = Math.min(SNIFF_BYTES, file.size);
  const first =
    "read" in file
      ? await file.read(0, length)
      : new Uint8Array(await file.slice(0, length).arrayBuffer());
  const seen = notSafetensors(first, file.size);
  if (seen !== undefined) {
    throw new Error(`${where}: only safetensors files are read, and ${seen}`);
  }
  return await openSafetensors(file);
};

/** The distribution's name map, fetched through the hub and checked. */
const readNameMap = async (
  where: string,
  loaded: LoadedManifest,
  options: FetchAssetsOptions,
): Promise<NameMap> => {
  const ref = loaded.manifest.nameMap;
  if (ref === undefined) {
    throw new Error(
      `${where}: the distribution carries no name map (its manifest has no 'nameMap'), so the` +
        " file's tensors cannot be placed in its components",
    );
  }
  const { nameMap } = await fetchAssets(loaded, { nameMap: ref }, options);
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(nameMap));
  } catch (cause) {
    throw new Error(`${where}: the distribution's name map ${ref.path} is not UTF-8 JSON`, {
      cause,
    });
  }
  return parseNameMap(parsed);
};

const quoted = (values: readonly string[]): string =>
  values.map((value) => `'${value}'`).join(", ");

/** A tensor of the file as an entry of a tensor list (the float dtypes only). */
const sourceTensor = (file: SafetensorsFile, key: string): SourceTensor | string => {
  const tensor = file.tensors.get(key);
  if (tensor === undefined) return `'${key}' is not a tensor of the file`;
  const { dtype } = tensor;
  if (dtype !== "F32" && dtype !== "F16" && dtype !== "BF16") {
    return `'${key}' is ${dtype}, not a float tensor (F32 / F16 / BF16)`;
  }
  return { dtype, shape: tensor.shape, read: tensor.read };
};

/** Everything the storage of one from-file role is settled from. */
type RoleSetting = {
  readonly entry: SingleFileRole;
  readonly family: SingleFileFamily;
  readonly modelName: string;
  readonly model: ModelEntry;
  /** Initializer name → the file key it was read from. */
  readonly keys: ReadonlyMap<string, string>;
  readonly tensors: ReadonlyMap<string, SourceTensor>;
  /** The role's file keys by dtype, in the order the dtypes first appear. */
  readonly dtypes: ReadonlyMap<TensorDtype, readonly string[]>;
};

/**
 * The container a role is converted against — with the codec its 16-bit float weights are stored
 * as, for a float storage — or why there is none.
 */
type RoleStorage =
  | {
      readonly label: string;
      readonly container: ContainerRef;
      readonly float16Codec?: Float16Codec;
    }
  | { readonly refusal: string };

/** The role's file keys by dtype, in the order the dtypes first appear. */
const keysByDtype = (
  keys: ReadonlyMap<string, string>,
  tensors: ReadonlyMap<string, SourceTensor>,
): ReadonlyMap<TensorDtype, readonly string[]> => {
  const byDtype = new Map<TensorDtype, string[]>();
  for (const [initializer, tensor] of tensors) {
    const list = byDtype.get(tensor.dtype) ?? [];
    list.push(keys.get(initializer) ?? initializer);
    byDtype.set(tensor.dtype, list);
  }
  return byDtype;
};

const isFloatLabel = (family: SingleFileFamily, label: string): boolean =>
  label === family.floatLabel;

/** The family's float storages, as a message names them (`BF16 → 'bf16', F16 → 'f16'`). */
const floatStorages = (family: SingleFileFamily): string =>
  Object.entries(family.floatCodecs)
    .map(([dtype, codec]) => `${dtype} → '${codec}'`)
    .join(", ");

/**
 * What to do when the file's role cannot be stored as it is: take the distribution's (a role that
 * may come from it), or a quant that stores it quantized (one whose quant decides its storage).
 * `refusedLabel` is the label of the container that refused the file's tensors: the quants of that
 * label would refuse them the same way, so they are not named.
 */
const advice = (setting: RoleSetting, refusedLabel?: string): string => {
  const { entry, family, model, modelName } = setting;
  const { role } = entry;
  if (entry.presence === "optional") {
    return `pass components: { ${role}: … } to take the distribution's ${role}`;
  }
  if (entry.storage === "quant") {
    const quantized = model.available.quants.filter((quant) => {
      const label = model.quants[quant].weights[role];
      return !isFloatLabel(family, label) && label !== refusedLabel;
    });
    const other =
      refusedLabel === undefined ? "" : ` in another container than its '${refusedLabel}' one`;
    return quantized.length === 0
      ? `model '${modelName}' has no quant that stores the ${role} quantized${other} (which converts from any float dtype)`
      : `choose a quant that stores the ${role} quantized, which converts from any float dtype: ${quantized.join(" / ")}`;
  }
  return `the file's ${role} must hold ${Object.keys(family.floatCodecs).join(" or ")} tensors`;
};

/**
 * The container the role is converted against under `quantName` ({@link SingleFileRole.storage}),
 * or the refusal: a role stored as a float whose tensors mix dtypes, or whose dtype has no 16-bit
 * float codec; a model without the float container. Manifest data only — nothing is fetched.
 *
 * SPEC: docs/spec/single-file.md#which-component-comes-from-where
 */
const resolveStorage = (setting: RoleSetting, quantName: string): RoleStorage => {
  const { entry, family, model, modelName, dtypes } = setting;
  const { role } = entry;
  const labels = Object.hasOwn(model.weights, role) ? model.weights[role] : {};
  const quant = model.quants[quantName];
  const quantLabel = Object.hasOwn(quant.weights, role) ? quant.weights[role] : undefined;
  if (entry.storage === "quant") {
    if (quantLabel === undefined || !Object.hasOwn(labels, quantLabel)) {
      return { refusal: `the selection ${modelName} / ${quantName} has no ${role}` };
    }
    if (!isFloatLabel(family, quantLabel)) {
      return { label: quantLabel, container: labels[quantLabel].container };
    }
  }
  const asFloat =
    entry.storage === "quant" ? ` (quant '${quantName}' stores the ${role} as a float)` : "";
  const [first, ...others] = dtypes.keys();
  if (others.length > 0) {
    const mix = [...dtypes]
      .map(([dtype, keys]) => `${dtype} (${keys.length} tensor(s), e.g. '${keys[0]}')`)
      .join(", ");
    return {
      refusal:
        `the file's ${role} mixes dtypes: ${mix}; it is stored as the float type of its tensors` +
        `${asFloat}, so they must all have one; ${advice(setting)}`,
    };
  }
  const dtype = first;
  const codec = family.floatCodecs[dtype];
  if (codec === undefined) {
    return {
      refusal:
        `the file's ${role} is ${dtype}${asFloat}, and no container stores ${dtype} as it is` +
        ` (the float storages: ${floatStorages(family)}); ${advice(setting)}`,
    };
  }
  // For a `"quant"` role this is the quant's own label (it is the float label).
  const label = family.floatLabel;
  if (!Object.hasOwn(labels, label)) {
    return {
      refusal:
        `model '${modelName}' has no '${label}' ${role}, whose plan the file's ${role} is converted` +
        ` against (its tensors are ${dtype}); ${advice(setting)}`,
    };
  }
  const { container } = labels[label];
  return { label, container, float16Codec: codec };
};

/** Everything one from-file role needs to check its plan. */
type RoleContext = {
  readonly where: string;
  readonly setting: RoleSetting;
  /** The weights label of the container it is converted against. */
  readonly label: string;
  readonly loaded: LoadedManifest;
  readonly selection: ResolvedSelection;
  readonly options: FetchAssetsOptions;
};

/**
 * The quants of the selected model under which the role's container stores only codecs
 * `@karume/convert` encodes, in the manifest's order: each quant's container as
 * {@link resolveStorage} settles it (a quant refused there does not work), read from its part 0
 * (fetched for this refusal only; no weight part).
 */
const quantsThatEncode = async (context: RoleContext): Promise<readonly string[]> => {
  const { loaded, setting, label: failing, options } = context;
  const working = new Map<string, boolean>();
  const labelOf = new Map<string, string>();
  for (const quant of setting.model.available.quants) {
    const storage = resolveStorage(setting, quant);
    if ("refusal" in storage) continue;
    labelOf.set(quant, storage.label);
    if (storage.label === failing || working.has(storage.label)) continue;
    await prefetchAssets(loaded, [storage.container.parts[0]], options);
    const codecs = await readPlanCodecs(
      openContainerSource(loaded, storage.container, options),
      storage.float16Codec === undefined ? {} : { float16Codec: storage.float16Codec },
    );
    working.set(
      storage.label,
      codecs.every(({ codec }) => ENCODED_CODECS.includes(codec)),
    );
  }
  return setting.model.available.quants.filter((quant) => {
    const label = labelOf.get(quant);
    return label !== undefined && working.get(label) === true;
  });
};

/** The source dtype a float codec stores as it is (`undefined`: a codec that converts any). */
const exactDtype = (codec: CodecName): TensorDtype | undefined =>
  codec === "f16" ? "F16" : codec === "bf16" ? "BF16" : undefined;

/**
 * The refusals of one from-file role that need its plan: a tensor the plan does not use, a codec
 * `@karume/convert` does not encode (one message per role, with the quants that work), and a
 * tensor whose dtype is not the one an initializer's float codec stores as it is (f16 and bf16
 * storage never round, so such a tensor would fail at its first read otherwise). `plan` is the plan
 * the role is converted against — for a float storage, with its 16-bit float weights already
 * stored as the file's dtype, so the last refusal is left to a quantized container's f16 or bf16
 * initializers.
 *
 * SPEC: docs/spec/single-file.md#the-refusals
 */
const checkRolePlan = async (
  context: RoleContext,
  plan: readonly PlannedCodec[],
): Promise<void> => {
  const { where, setting, label, selection } = context;
  const { entry, keys, tensors } = setting;
  const { role } = entry;
  const planned = new Set(plan.map(({ initializer }) => initializer));
  const unused = [...keys].filter(([initializer]) => !planned.has(initializer));
  if (unused.length > 0) {
    throw new Error(
      `${where}: the file holds ${unused.length} tensor(s) placed in the ${role} that its plan does not use` +
        ` (nothing of the file is dropped): ${unused
          .map(([initializer, key]) => `'${key}' (as '${initializer}')`)
          .join(", ")}`,
    );
  }
  const unencoded = [...new Set(plan.map(({ codec }) => codec))].filter(
    (codec) => !ENCODED_CODECS.includes(codec),
  );
  if (unencoded.length > 0) {
    const stored =
      `${where}: the '${label}' ${role} of model '${selection.model}', which the file's ${role} is` +
      ` converted against under quant '${selection.quant}', stores codec ${quoted(unencoded)},` +
      ` which the conversion of a file does not encode (it encodes ${ENCODED_CODECS.join(" / ")})`;
    if (entry.storage === "file-dtype") throw new Error(stored);
    const working = await quantsThatEncode(context);
    throw new Error(
      `${stored}; the quants that work: ${working.length === 0 ? "none" : working.join(" / ")}`,
    );
  }
  for (const { initializer, codec } of plan) {
    const wanted = exactDtype(codec);
    const tensor = tensors.get(initializer);
    if (wanted === undefined || tensor === undefined || tensor.dtype === wanted) continue;
    throw new Error(
      `${where}: the ${role} stores '${initializer}' as ${codec}, which takes ${wanted} values as` +
        ` they are and never rounds, but the file's '${keys.get(initializer)}' is ${tensor.dtype};` +
        ` ${advice(setting, label)}`,
    );
  }
};

/** What {@link prepareSingleFile} settles. */
export type SingleFileComponents = {
  /** The roles that come from the file, in the role table's order. */
  readonly fromFile: readonly string[];
  /** Role → the component converted from the file (the loader's `converted`). */
  readonly converted: Readonly<Record<string, ConvertedComponent>>;
  /** The layout of the name map that recognised the file. */
  readonly layout: string;
};

/**
 * Reads the name map and the file's header, places the file's tensors and decides which role
 * comes from the file and against which container (module doc). `replaced` are the roles the
 * caller replaces with `components`: they never come from the file, whatever it holds.
 * `retainEncoded` goes to every component from the file ({@link ConvertedComponent.retainEncoded}).
 * `options` are the hub's options of the load, without progress (the name map is not part of the
 * model's progress stream).
 *
 * SPEC: docs/spec/single-file.md#which-component-comes-from-where
 */
export const prepareSingleFile = async (
  where: string,
  file: SingleFileSource,
  loaded: LoadedManifest,
  selection: ResolvedSelection,
  family: SingleFileFamily,
  replaced: ReadonlySet<string>,
  retainEncoded: boolean,
  options: FetchAssetsOptions,
): Promise<SingleFileComponents> => {
  const map = await readNameMap(where, loaded, options);
  options.signal?.throwIfAborted();
  const opened = await openSingleFile(where, file);
  options.signal?.throwIfAborted();
  const placement: TensorPlacement = placeTensors(map, opened.tensors.keys());
  if (placement.unplaced.length > 0) {
    throw new Error(
      `${where}: ${placement.unplaced.length} key(s) of the file go to no component of the name map's` +
        ` layout '${placement.layout}' (nothing of the file is dropped): ${quoted(placement.unplaced)}`,
    );
  }
  const { roles } = family;
  const taken = new Set(roles.map(({ role }) => role));
  const foreign = [...placement.components].filter(
    ([role, names]) => names.size > 0 && !taken.has(role) && !replaced.has(role),
  );
  if (foreign.length > 0) {
    throw new Error(
      `${where}: the file holds tensors of ${quoted(foreign.map(([role]) => role))}, which this` +
        ` pipeline does not take from a file (it takes ${quoted([...taken])})`,
    );
  }

  const model = loaded.manifest.models[selection.model];
  const converted: Record<string, ConvertedComponent> = {};
  const fromFile: string[] = [];
  for (const entry of roles) {
    const { role } = entry;
    if (replaced.has(role)) continue;
    const keys = placement.components.get(role) ?? new Map<string, string>();
    if (keys.size === 0) {
      if (entry.presence === "required") {
        throw new Error(
          `${where}: the file holds no ${role} (the name map's layout '${placement.layout}' places` +
            ` none of its ${opened.tensors.size} keys there)`,
        );
      }
      continue;
    }
    const tensors = new Map<string, SourceTensor>();
    const problems: string[] = [];
    for (const [initializer, key] of keys) {
      const tensor = sourceTensor(opened, key);
      if (typeof tensor === "string") problems.push(tensor);
      else tensors.set(initializer, tensor);
    }
    if (problems.length > 0) {
      throw new Error(
        `${where}: the ${role} of the file cannot be converted:\n- ${problems.join("\n- ")}`,
      );
    }
    const setting: RoleSetting = {
      entry,
      family,
      modelName: selection.model,
      model,
      keys,
      tensors,
      dtypes: keysByDtype(keys, tensors),
    };
    const storage = resolveStorage(setting, selection.quant);
    if ("refusal" in storage) throw new Error(`${where}: ${storage.refusal}`);
    const context: RoleContext = {
      where,
      setting,
      label: storage.label,
      loaded,
      selection,
      options,
    };
    converted[role] = {
      container: storage.container,
      tensors,
      ...(storage.float16Codec === undefined ? {} : { float16Codec: storage.float16Codec }),
      checkPlan: (plan) => checkRolePlan(context, plan),
      ...(retainEncoded ? { retainEncoded: true } : {}),
    };
    fromFile.push(role);
  }
  if (fromFile.length === 0) {
    throw new Error(
      `${where}: no component is left to come from the file (${
        replaced.size > 0 ? `components replaces ${quoted([...replaced])}` : "the file holds none"
      }); without one, load the distribution with fromPretrained`,
    );
  }
  return { fromFile, converted, layout: placement.layout };
};
