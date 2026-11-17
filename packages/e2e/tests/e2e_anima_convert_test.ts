/**
 * `@karume/convert` on the Anima distribution (no GPU): a distributed container read as a tensor
 * list, converted against the plan of another preset, written out and compared with the hashes the
 * manifest records for that preset.
 *
 * 1. The transformer of `anima-v1.0`: the tensors of the `bf16` preset, the plan of the `i8` preset
 *    (int8-sym DiT weights, f32 biases and norms). Every written part has the size and sha256 the
 *    manifest records for the `i8` files, and the two descriptors the manifest's expectation.
 * 2. The text conditioner of `anima-turbo-v1.1`: its `bf16` tensors against its own `bf16` plan (the
 *    copy path, a pieced tensor included) — the same assertion.
 * 3. The converting source of case 1 opened with `openContainer` and prepared (`prepareContainer`):
 *    it binds every initializer, with nothing missing and nothing left over.
 *
 * These compare with recorded hashes; they are not sha256 reference rows. They read about 6 GB from
 * the local mirror `models/karume-anima/` (untracked); where it is missing the cases SKIP
 * explicitly with the reason (ADR 0005).
 */

import { isNotFound, readTextFile, stat } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  type BlockSource,
  type DescriptorExpectation,
  type OpenedContainer,
  openContainer,
  prepareContainer,
} from "@karume/runtime";
import { fileBlockSource } from "../../runtime/tests/helpers/container-files.ts";
import {
  type ContainerSink,
  type Digest,
  openConvertingSource,
  type SourceTensor,
  type TensorDtype,
  type TensorList,
  writeContainer,
} from "../../convert/mod.ts";

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

const ASSETS_AVAILABLE = await exists(new URL("karume.json", ANIMA_DIR));
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${ANIMA_DIR.pathname} has no karume.json, so the conversion of the Anima containers SKIPs` +
      " (the comparison with the hashes the manifest records needs the local mirror)",
  );
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

const MANIFEST: Manifest | undefined = ASSETS_AVAILABLE
  ? JSON.parse(await readTextFile(new URL("karume.json", ANIMA_DIR)))
  : undefined;

const containerRef = (model: string, component: string, preset: string): ContainerRef => {
  const ref = MANIFEST?.models[model]?.weights[component]?.[preset]?.container;
  if (ref === undefined) throw new Error(`the manifest has no ${model} / ${component} / ${preset}`);
  return ref;
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

const DTYPE_OF_CODEC: Readonly<Record<string, TensorDtype>> = {
  f32: "F32",
  f16: "F16",
  bf16: "BF16",
};

/**
 * A float container read as a tensor list: every model-bound initializer under its own name, its
 * bytes the payloads of its blocks in row order (each block's sha256 checked by `readBlock`, since
 * the local files are unverified).
 */
const containerTensors = (opened: OpenedContainer): TensorList => {
  const tensors = new Map<string, SourceTensor>();
  for (const bound of Object.values(opened.graphs)) {
    for (const [name, supply] of bound.supplies) {
      if (supply.origin !== "model") continue;
      const dtype = DTYPE_OF_CODEC[supply.encoding.codec];
      if (dtype === undefined)
        throw new Error(`${name}: codec ${supply.encoding.codec} is not a float`);
      tensors.set(name, {
        dtype,
        shape: bound.declaration.values[name].shape.map(Number),
        read: async () => {
          const total = supply.blocks.reduce((sum, block) => sum + block.payloadBytes, 0);
          const out = new Uint8Array(new ArrayBuffer(total));
          let cursor = 0;
          for (const block of supply.blocks) {
            out.set((await opened.readBlock(block.id)).subarray(0, block.payloadBytes), cursor);
            cursor += block.payloadBytes;
          }
          return out;
        },
      });
    }
  }
  return tensors;
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

/**
 * Converts `component` of `model` from the tensors of preset `from` against the plan of preset
 * `to`, writes it, and checks every part and both descriptors against the manifest's record of `to`.
 */
const convertAndCompare = async (
  model: string,
  component: string,
  from: string,
  to: string,
): Promise<void> => {
  const started = performance.now();
  const sourceRef = containerRef(model, component, from);
  const targetRef = containerRef(model, component, to);
  const tensors = containerTensors(
    await openContainer({ kind: "source", source: partsSource(sourceRef) }, sourceRef.descriptor),
  );
  const target = recording(partsSource(targetRef));
  const converting = await openConvertingSource({ target: target.source, tensors });
  assertEquals(converting.unusedTensors, []);
  const { sink, parts } = hashingSink();
  const written = await writeContainer(converting.source, sink);
  const recorded = targetRef.parts.map((part) => ({ length: part.size, sha256: part.sha256 }));
  assertEquals(parts, recorded);
  assertEquals(written.parts, recorded);
  assertEquals(written.descriptor, targetRef.descriptor);
  // Only part 0, the const part and the asset part of the target were read (the weights were not).
  const weightParts = new Set(
    (await openContainer({ kind: "source", source: converting.source })).model?.blocks
      .filter((block) => block.role !== "asset")
      .map((block) => block.part),
  );
  assertEquals(
    [...target.read].filter((part) => weightParts.has(part)),
    [],
  );
  console.log(
    `[karume] ${model} / ${component}: ${from} → ${to}, ${parts.length} parts identical in ${(
      (performance.now() - started) /
      1000
    ).toFixed(1)} s, peak RSS ${Math.round(process.resourceUsage().maxRSS / 1024)} MiB`,
  );
};

describe("the Anima containers converted while read", () => {
  it.skipIf(!ASSETS_AVAILABLE)(
    "anima-v1.0 transformer: the bf16 tensors written against the i8 plan give the i8 files",
    async () => {
      await convertAndCompare("anima-v1.0", "transformer", "bf16", "i8");
    },
  );

  it.skipIf(!ASSETS_AVAILABLE)(
    "anima-turbo-v1.1 text conditioner: the bf16 tensors against the bf16 plan give the bf16 files",
    async () => {
      await convertAndCompare("anima-turbo-v1.1", "text_conditioner", "bf16", "bf16");
    },
  );

  it.skipIf(!ASSETS_AVAILABLE)(
    "the converting source of the transformer opens and binds every initializer, without a GPU",
    async () => {
      const sourceRef = containerRef("anima-v1.0", "transformer", "bf16");
      const tensors = containerTensors(
        await openContainer(
          { kind: "source", source: partsSource(sourceRef) },
          sourceRef.descriptor,
        ),
      );
      const converting = await openConvertingSource({
        target: partsSource(containerRef("anima-v1.0", "transformer", "i8")),
        tensors,
      });
      const opened = await openContainer({ kind: "source", source: converting.source });
      const prepared = prepareContainer(opened, "transformer");
      const declared = Object.keys(opened.graphs["transformer"].declaration.initializers).sort();
      assertEquals(Object.keys(prepared.graph.initializers).sort(), declared);
      assertEquals([...opened.graphs["transformer"].supplies.keys()].sort(), declared);
    },
  );
});
