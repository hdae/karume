/**
 * Weights converted while a Session reads them (`@karume/convert`; private for now).
 *
 * The input is a weights file read as a tensor list — format-neutral: per tensor name, its element
 * type (`F32` / `F16` / `BF16`), its shape and a function that reads its bytes — and a distributed
 * preset whose part 0 is the plan: which initializer is stored how, in which blocks, where.
 *
 * - {@link openConvertingSource} builds a `BlockSource` of the runtime over the target preset's
 *   container that encodes each weight block from the tensor list as it is read (`int8-sym`,
 *   `f16`, `bf16` and `f32`, bit for bit as the exporter writes them) and passes part 0, the const part and
 *   the asset parts through from the target. `openContainer` of `@karume/runtime` opens it like any
 *   other source. Every check that needs no tensor byte runs when it is built and is reported
 *   together in one {@link ConvertError}.
 * - {@link writeContainer} optionally writes what a source serves as a container, part by part, to
 *   a sink, with the real sha256 of every block and part, and returns what a manifest entry needs.
 * - {@link placeTensors} reads a name map — JSON that says, per layout of a model's upstream files,
 *   which key of a file is which initializer of which component — against a file's keys, and gives
 *   per component the initializer name → key mapping the tensor list is built from;
 *   {@link parseNameMap} checks a map that arrives as unchecked JSON (a distribution's file).
 * - {@link readPlanCodecs} reads, from a target's part 0 alone, the codec of every initializer it
 *   stores — what a caller checks before it gives a tensor list ({@link ENCODED_CODECS} are the
 *   codecs encoded here). A converting source says which parts it builds (`convertedParts`), so a
 *   caller fetches only the others.
 * - {@link openConvertingSource} and {@link readPlanCodecs} take `float16Codec` ({@link PlanOptions}): the target's plan with every `f16` and `bf16`
 *   initializer stored as that codec instead — the plan of the same container exported with that
 *   16-bit float type, since both take two bytes an element and no block moves.
 *
 * Web standard APIs only; no side effects at import.
 *
 * @module
 */

export { ConvertError } from "./src/errors.ts";
export { ENCODED_CODECS } from "./src/encode.ts";
export { NAME_MAP_FORMAT, parseNameMap, placeTensors } from "./src/name-map.ts";
export { readPlanCodecs } from "./src/plan.ts";
export type { Float16Codec, PlanOptions, PlannedCodec } from "./src/plan.ts";
export type {
  NameMap,
  NameMapComponent,
  NameMapLayout,
  NameMapRule,
  TensorPlacement,
} from "./src/name-map.ts";
export type { SourceTensor, TensorDtype, TensorList } from "./src/tensors.ts";
export { INVALID_SHA256, openConvertingSource } from "./src/source.ts";
export type { ConvertingSource, ConvertingSourceInput } from "./src/source.ts";
export { writeContainer } from "./src/write.ts";
export type { ContainerSink, Digest, WrittenContainer } from "./src/write.ts";
