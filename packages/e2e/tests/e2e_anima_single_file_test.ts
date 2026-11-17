/**
 * The upstream Anima single files, read one tensor at a time with `@karume/safetensors`, placed
 * with the recipe's name map (`tools/export-recipes/anima/name-map.json`) and converted with
 * `@karume/convert` against the plans of the distributed presets; every written part is compared
 * with the hash the manifest records for that preset. No GPU.
 *
 * 1. `anima-base-v1.0.safetensors`: all 685 keys are placed (567 transformer + 118 text
 *    conditioner, nothing excluded or unplaced), and the transformer written against the `i8` plan
 *    of `anima-v1.0` gives its files.
 * 2. The same file: the transformer against the `bf16` plan, the text conditioner against the
 *    `bf16` plan.
 * 3. `anima-turbo-v1.1.safetensors`: its transformer (`i8`) and its text conditioner (`bf16`)
 *    against the plans of `anima-turbo-v1.1`.
 * 4. The map with `self_attn` rewritten before `adaln_modulation_self_attn.1`: the converting source
 *    refuses the transformer before a tensor is read (the rule order is part of the data).
 *
 * These compare with recorded hashes; they are not sha256 reference rows. The inputs are the
 * upstream files under `inputs/anima/upstream-single-file/` (circlestone-labs/Anima, revision
 * 6e0b2da2112573380abfb6a1435a60a994b254f2; about 4.2 GB each) and the local mirror
 * `models/karume-anima/` (all untracked); where one is missing the cases SKIP explicitly with the
 * reason (ADR 0005).
 */

import { isNotFound, readRange, readTextFile, stat } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assertEquals, assertRejects } from "@std/assert";
import type { BlockSource, DescriptorExpectation } from "@karume/runtime";
import { fileBlockSource } from "../../runtime/tests/helpers/container-files.ts";
import {
  type ContainerSink,
  ConvertError,
  type Digest,
  type NameMap,
  type NameMapRule,
  openConvertingSource,
  placeTensors,
  type SourceTensor,
  type TensorList,
  type TensorPlacement,
  writeContainer,
} from "../../convert/mod.ts";
import {
  openSafetensors,
  type RangedReader,
  type SafetensorsFile,
  type SafetensorsTensor,
} from "../../safetensors/mod.ts";
import animaNameMap from "../../../tools/export-recipes/anima/name-map.json" with { type: "json" };

const NAME_MAP: NameMap = animaNameMap;

const INPUTS = new URL(
  "../../../inputs/anima/upstream-single-file/split_files/diffusion_models/",
  import.meta.url,
);
const BASE_FILE = new URL("anima-base-v1.0.safetensors", INPUTS);
const TURBO_FILE = new URL("anima-turbo-v1.1.safetensors", INPUTS);
const ANIMA_DIR = new URL("../../../models/karume-anima/", import.meta.url);

const exists = async (url: URL): Promise<boolean> => {
  try {
    await stat(url);
    return true;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const MIRROR = await exists(new URL("karume.json", ANIMA_DIR));
const BASE_READY = MIRROR && (await exists(BASE_FILE));
const TURBO_READY = MIRROR && (await exists(TURBO_FILE));
for (const [ready, file] of [
  [BASE_READY, BASE_FILE],
  [TURBO_READY, TURBO_FILE],
] as const) {
  if (!ready) {
    console.warn(
      `[karume] the conversion of ${file.pathname} SKIPs: it needs that upstream file and the local` +
        ` mirror ${ANIMA_DIR.pathname} (karume.json), whose recorded hashes it is compared with`,
    );
  }
}

/** The fields of a `karume/5` manifest these cases read. */
type ContainerRef = {
  readonly descriptor: DescriptorExpectation;
  readonly parts: readonly {
    readonly path: string;
    readonly size: number;
    readonly sha256: string;
  }[];
};
type Manifest = {
  readonly models: Readonly<
    Record<
      string,
      {
        readonly weights: Readonly<
          Record<string, Readonly<Record<string, { readonly container: ContainerRef }>>>
        >;
      }
    >
  >;
};

const MANIFEST: Manifest | undefined = MIRROR
  ? JSON.parse(await readTextFile(new URL("karume.json", ANIMA_DIR)))
  : undefined;

const containerRef = (model: string, component: string, preset: string): ContainerRef => {
  const ref = MANIFEST?.models[model]?.weights[component]?.[preset]?.container;
  if (ref === undefined) throw new Error(`the manifest has no ${model} / ${component} / ${preset}`);
  return ref;
};

/** A local file as a ranged reader (a 4 GB file is never read whole). */
const rangedFile = async (url: URL): Promise<RangedReader> => ({
  size: (await stat(url)).size,
  read: (offset, length) => readRange(url, offset, length),
});

/** A tensor of the file as an entry of the tensor list, which takes the float dtypes only. */
const sourceTensor = (key: string, tensor: SafetensorsTensor): SourceTensor => {
  const { dtype } = tensor;
  if (dtype !== "F32" && dtype !== "F16" && dtype !== "BF16") {
    throw new Error(`${key}: ${dtype} is not a dtype of the tensor list`);
  }
  return { ...tensor, dtype };
};

/** A component's tensor list: the file's tensors under the initializer names the map gives them. */
const componentTensors = (
  file: SafetensorsFile,
  placement: TensorPlacement,
  role: string,
): TensorList => {
  const names = placement.components.get(role);
  if (names === undefined) throw new Error(`layout ${placement.layout} has no component ${role}`);
  return new Map(
    [...names].map(([initializer, key]) => {
      const tensor = file.tensors.get(key);
      if (tensor === undefined) throw new Error(`the file has no tensor ${key}`);
      return [initializer, sourceTensor(key, tensor)];
    }),
  );
};

/** The part files of a container (index = part), read by range and unverified. */
const partsSource = (ref: ContainerRef): BlockSource =>
  fileBlockSource(ref.parts.map((part) => new URL(part.path, ANIMA_DIR)));

/** A source that records which parts it was asked for. */
const recording = (
  source: BlockSource,
): { readonly source: BlockSource; readonly read: Set<number> } => {
  const read = new Set<number>();
  return {
    read,
    source: {
      ...source,
      read: (part, offset, length) => {
        read.add(part);
        return source.read(part, offset, length);
      },
    },
  };
};

const sha256 = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");

/** A sink that keeps no byte: it hashes each part it is given. */
const hashingSink = (): { readonly sink: ContainerSink; readonly parts: Digest[] } => {
  const parts: Digest[] = [];
  return {
    parts,
    sink: {
      write: async (part, offset, bytes) => {
        assertEquals(offset, 0);
        parts[part] = { length: bytes.byteLength, sha256: await sha256(bytes) };
      },
    },
  };
};

/** Opens a file and places its keys with the Anima map; every key is placed (none excluded). */
const openAndPlace = async (
  url: URL,
): Promise<{ readonly file: SafetensorsFile; readonly placement: TensorPlacement }> => {
  const file = await openSafetensors(await rangedFile(url));
  const placement = placeTensors(NAME_MAP, file.tensors.keys());
  const placed = Object.fromEntries(
    [...placement.components].map(([role, names]) => [role, names.size]),
  );
  console.log(
    `[karume] ${url.pathname.split("/").pop()}: ${file.tensors.size} keys, layout ${
      placement.layout
    }, placed ${JSON.stringify(placed)}, excluded ${placement.excluded.length}, unplaced ${
      placement.unplaced.length
    }`,
  );
  assertEquals(file.tensors.size, 685);
  assertEquals(placement.layout, "single-file");
  assertEquals(placed, { text_conditioner: 118, transformer: 567 });
  assertEquals(placement.excluded, []);
  assertEquals(placement.unplaced, []);
  return { file, placement };
};

/**
 * Converts one component of the file against the plan of `preset` of `model`, writes it, and
 * checks every part and both descriptors against the manifest's record of that preset.
 */
const convertAndCompare = async (
  file: SafetensorsFile,
  placement: TensorPlacement,
  model: string,
  component: string,
  preset: string,
): Promise<void> => {
  const started = performance.now();
  const targetRef = containerRef(model, component, preset);
  const target = recording(partsSource(targetRef));
  const converting = await openConvertingSource({
    target: target.source,
    tensors: componentTensors(file, placement, component),
  });
  assertEquals(converting.unusedTensors, []);
  const { sink, parts } = hashingSink();
  const written = await writeContainer(converting.source, sink);
  const recorded = targetRef.parts.map((part) => ({ length: part.size, sha256: part.sha256 }));
  assertEquals(parts, recorded);
  assertEquals(written.parts, recorded);
  assertEquals(written.descriptor, targetRef.descriptor);
  console.log(
    `[karume] ${model} / ${component} / ${preset}: ${parts.length} parts identical in ${(
      (performance.now() - started) /
      1000
    ).toFixed(
      1,
    )} s (target parts read: ${[...target.read].sort((a, b) => a - b).join(", ")}), peak RSS ${Math.round(
      process.resourceUsage().maxRSS / 1024,
    )} MiB`,
  );
};

/** The rules with `self_attn` moved before `adaln_modulation_self_attn.1` (lists without both stay). */
const selfAttnFirst = (rules: readonly NameMapRule[]): readonly NameMapRule[] => {
  const at = rules.findIndex(
    (rule) => "replace" in rule && rule.replace === "adaln_modulation_self_attn.1",
  );
  const selfAttn = rules.find((rule) => "replace" in rule && rule.replace === "self_attn");
  if (at < 0 || selfAttn === undefined) return rules;
  const rest = rules.filter((rule) => rule !== selfAttn);
  return [...rest.slice(0, at), selfAttn, ...rest.slice(at)];
};

describe("the Anima upstream single files converted while read", () => {
  it.skipIf(!BASE_READY)(
    "anima-base-v1.0: every key is placed, and the transformer against the i8 plan of anima-v1.0 gives its files",
    async () => {
      const { file, placement } = await openAndPlace(BASE_FILE);
      await convertAndCompare(file, placement, "anima-v1.0", "transformer", "i8");
    },
  );

  it.skipIf(!BASE_READY)(
    "anima-base-v1.0: the transformer and the text conditioner against the bf16 plans of anima-v1.0 give their files",
    async () => {
      const { file, placement } = await openAndPlace(BASE_FILE);
      await convertAndCompare(file, placement, "anima-v1.0", "transformer", "bf16");
      await convertAndCompare(file, placement, "anima-v1.0", "text_conditioner", "bf16");
    },
  );

  it.skipIf(!TURBO_READY)(
    "anima-turbo-v1.1: the transformer against the i8 plan and the text conditioner against the bf16 plan of anima-turbo-v1.1 give their files",
    async () => {
      const { file, placement } = await openAndPlace(TURBO_FILE);
      await convertAndCompare(file, placement, "anima-turbo-v1.1", "transformer", "i8");
      await convertAndCompare(file, placement, "anima-turbo-v1.1", "text_conditioner", "bf16");
    },
  );

  it.skipIf(!BASE_READY)(
    "the map with self_attn rewritten before adaln_modulation_self_attn.1: the transformer is refused before a tensor is read",
    async () => {
      const reordered: NameMap = {
        ...NAME_MAP,
        layouts: NAME_MAP.layouts.map((layout) => ({
          ...layout,
          components: layout.components.map((component) => ({
            ...component,
            rules: selfAttnFirst(component.rules),
          })),
        })),
      };
      const file = await openSafetensors(await rangedFile(BASE_FILE));
      const placement = placeTensors(reordered, file.tensors.keys());
      const tensors = componentTensors(file, placement, "transformer");
      let reads = 0;
      const counted: TensorList = new Map(
        [...tensors].map(([name, tensor]) => [
          name,
          {
            ...tensor,
            read: () => {
              reads += 1;
              return tensor.read();
            },
          },
        ]),
      );
      await assertRejects(
        () =>
          openConvertingSource({
            target: partsSource(containerRef("anima-v1.0", "transformer", "i8")),
            tensors: counted,
          }),
        ConvertError,
        "initializer 'model.transformer_blocks.0.norm1.linear_1.weight': the tensor list has no tensor",
      );
      assertEquals(reads, 0);
    },
  );
});
