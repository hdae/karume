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
 *    containers; the plan checks are handed to the loader ({@link ConvertedComponent.checkPlan}).
 *
 * SPEC: docs/spec/single-file.md
 */

import {
  type ContainerRef,
  fetchAssets,
  type FetchAssetsOptions,
  type LoadedManifest,
  openContainerSource,
  prefetchAssets,
  type ResolvedSelection,
} from "@karume/hub";
import {
  ENCODED_CODECS,
  type NameMap,
  parseNameMap,
  placeTensors,
  type PlannedCodec,
  readPlanCodecs,
  type SourceTensor,
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
   * The container of the selected model the file's tensors are converted against: the selected
   * quant's (`"quant"`), or the container of this weights label whatever the quant — for a component
   * no preset quantizes, so that its weights keep the precision of the file.
   */
  readonly storage: "quant" | { readonly label: string };
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

/** What to do about a tensor bf16 storage cannot take, for one role. */
const bf16Advice = (entry: SingleFileRole): string =>
  entry.presence === "optional"
    ? `pass components: { ${entry.role}: … } to take the distribution's ${entry.role}`
    : entry.storage === "quant"
      ? `choose a quant that stores the ${entry.role} as int8, which takes any float tensor`
      : `the file's ${entry.role} must hold BF16 tensors`;

/** Everything one from-file role needs to check its plan. */
type RoleContext = {
  readonly where: string;
  readonly entry: SingleFileRole;
  readonly loaded: LoadedManifest;
  readonly selection: ResolvedSelection;
  /** Initializer name → the file key it was read from. */
  readonly keys: ReadonlyMap<string, string>;
  readonly tensors: ReadonlyMap<string, SourceTensor>;
  readonly options: FetchAssetsOptions;
};

/**
 * The quants of the selected model whose container for the role stores only codecs
 * `@karume/convert` encodes, in the manifest's order — read from the part 0 of each other
 * container of the role (fetched for this refusal only; no weight part).
 */
const quantsThatEncode = async (
  context: RoleContext,
  failingLabel: string,
): Promise<readonly string[]> => {
  const { loaded, selection, entry, options } = context;
  const model = loaded.manifest.models[selection.model];
  const working = new Map<string, boolean>();
  const labels = Object.hasOwn(model.weights, entry.role) ? model.weights[entry.role] : {};
  for (const [label, { container }] of Object.entries(labels)) {
    if (label === failingLabel) continue;
    await prefetchAssets(loaded, [container.parts[0]], options);
    const codecs = await readPlanCodecs(openContainerSource(loaded, container, options));
    working.set(
      label,
      codecs.every(({ codec }) => ENCODED_CODECS.includes(codec)),
    );
  }
  return model.available.quants.filter(
    (quant) => working.get(model.quants[quant].weights[entry.role]) === true,
  );
};

/**
 * The refusals of one from-file role that need its plan: a tensor the plan does not use, a codec
 * `@karume/convert` does not encode (one message per role, with the quants that work), and a
 * tensor that is not BF16 for an initializer the plan stores as bf16 (bf16 storage never rounds,
 * so such a tensor would fail at its first read otherwise).
 *
 * SPEC: docs/spec/single-file.md#the-refusals
 */
const checkRolePlan = async (
  context: RoleContext,
  plan: readonly PlannedCodec[],
): Promise<void> => {
  const { where, entry, selection, keys, tensors } = context;
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
    const codecs = quoted(unencoded);
    const encoded = ENCODED_CODECS.join(" / ");
    if (entry.storage === "quant") {
      const label =
        context.loaded.manifest.models[selection.model].quants[selection.quant].weights[role];
      const working = await quantsThatEncode(context, label);
      throw new Error(
        `${where}: quant '${selection.quant}' stores the ${role} with codec ${codecs}, which the` +
          ` conversion of a file does not encode (it encodes ${encoded}); the quants that work:` +
          ` ${working.length === 0 ? "none" : working.join(" / ")}`,
      );
    }
    throw new Error(
      `${where}: the '${entry.storage.label}' ${role} of model '${selection.model}', which the` +
        ` file's ${role} is converted against, stores codec ${codecs}, which the conversion of a` +
        ` file does not encode (it encodes ${encoded})`,
    );
  }
  for (const { initializer, codec } of plan) {
    const tensor = tensors.get(initializer);
    if (codec !== "bf16" || tensor === undefined || tensor.dtype === "BF16") continue;
    throw new Error(
      `${where}: the ${role} stores '${initializer}' as bf16, which takes BF16 values as they are` +
        ` and never rounds, but the file's '${keys.get(initializer)}' is ${tensor.dtype};` +
        ` ${bf16Advice(entry)}`,
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
 * comes from the file (module doc). `replaced` are the roles the caller replaces with
 * `components`: they never come from the file, whatever it holds. `retainEncoded` goes to every
 * component from the file ({@link ConvertedComponent.retainEncoded}). `options` are the hub's
 * options of the load, without progress (the name map is not part of the model's progress stream).
 *
 * SPEC: docs/spec/single-file.md#which-component-comes-from-where
 */
export const prepareSingleFile = async (
  where: string,
  file: SingleFileSource,
  loaded: LoadedManifest,
  selection: ResolvedSelection,
  roles: readonly SingleFileRole[],
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
    let container: ContainerRef;
    if (entry.storage === "quant") {
      if (!Object.hasOwn(selection.containers, role)) {
        throw new Error(
          `${where}: the selection ${selection.model} / ${selection.quant} has no ${role}`,
        );
      }
      container = selection.containers[role];
    } else {
      const { label } = entry.storage;
      const labels = Object.hasOwn(model.weights, role) ? model.weights[role] : {};
      if (!Object.hasOwn(labels, label)) {
        throw new Error(
          `${where}: model '${selection.model}' has no '${label}' ${role}, which the file's ${role} is` +
            ` stored as; pass components: { ${role}: … } to take the distribution's ${role}`,
        );
      }
      container = labels[label].container;
    }
    const context: RoleContext = { where, entry, loaded, selection, keys, tensors, options };
    converted[role] = {
      container,
      tensors,
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
