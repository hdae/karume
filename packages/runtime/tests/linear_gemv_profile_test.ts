/**
 * The GEMV choices of linear as data of the geometry profile (`GeometryProfile.linearGemv`): the
 * values moved from module constants of src/kernels/linear-gemv.ts into the default profile without
 * changing a selection; a table without them runs with the default profile's, resolved where a
 * Session selects its profile; a broken one is refused; and the profile is really the source — a
 * table with other choices changes the selection, the pipeline key and the fusion pass. No GPU.
 */

import { assert, assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
  type GeometryProfile,
  resolveGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import {
  defaultLinearGemvRowsVariant,
  defaultLinearGemvVariant,
  type LinearGemvChoices,
  type LinearGemvParallelLanes,
  type LinearGemvParallelShape,
  type LinearGemvShapeVariant,
  linearGemvPackedEligible,
  linearGemvParallelEligible,
  linearGemvParallelLanes,
  linearGemvRowsForShape,
  linearGemvStagedEligible,
  linearGemvUnit,
  type LinearGemvVariant,
} from "../src/kernels/linear-gemv.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import { FUSION_RULES, type FusionWeightLayout, planFusions } from "../src/runtime/fusion.ts";
import { sessionGeometryProfile } from "../src/runtime/geometry-selection.ts";
import { linearProgramFor } from "../src/runtime/linear-program.ts";
import {
  type LinearKnobs,
  resolveLinearStrategy,
  selectLinearStrategy,
} from "../src/runtime/linear-strategies.ts";
import { countUses, type NodePlan, planGraph } from "../src/runtime/plan.ts";
import { fakeDevice } from "./helpers/fake-gpu.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

/** The default profile's GEMV choices. */
const GEMV = DEFAULT_GEOMETRY_PROFILE.linearGemv;

// ---------------------------------------------------------------------------
// The pinned copy of the default profile's GEMV choices, and the lookups the kernel module had
// before the choices moved into the profile (copied from src/kernels/linear-gemv.ts as it stood
// then).
//
// Why this copy exists: the move had to change nothing, and the old code is gone. The literals
// began as the kernel module's `PARALLEL_SHAPES` (rows, order and flags), its sequential variant and
// its thread target, and the functions below are its lookups, reading the literals. A change of a
// value edits the default profile (re-measuring as its comments require) and this copy with it, so
// the table is the kernel module's 66 rows plus the rows added since: the five MTP drafter rows of
// 2026-11-18 (two after the E4B rows, three after the 12B rows) and the 17 bf16 rows of regular
// E2B / E4B of 2026-11-19 at the end (perf-ledger K-110). The kernel module held no row of a float
// storage, so its lookups' gate on the storage answered nothing for one; the copy below answers a
// float row like any other, as the selection does since. The general sequential variant
// and the thread target are still the kernel module's. The `rowBlock` of a row is another change
// since (2026-11-18, perf-ledger K-108): the kernel module never held it, and the lookups below do
// not read it. The kernel module's one per-shape exception of the sequential variant (the i8 output
// head) is no row of the default profile, so the copy holds the module's general variant alone
// (`OLD_DEFAULT_VARIANT`). The per-shape exceptions are the 22 rows added since
// (`ADDED_SEQUENTIAL_SHAPES`, 2026-11-18, perf-ledger K-112: bf16 linears of regular E2B / E4B that
// no measured device is slower with). The exceptions one device gains from and another loses with
// are rows of that device's table (`AMPERE_SEQUENTIAL_SHAPES`, `AMPERE_CHANGED_PARALLEL_SHAPES` and
// `AMPERE_PARALLEL_SHAPES`: `nvidia-ampere`).
// ---------------------------------------------------------------------------

const PINNED_PARALLEL_SHAPES: readonly LinearGemvParallelShape[] = [
  { storage: "i4", n: 8960, k: 1536, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
  { storage: "i4", n: 2048, k: 1536, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 256, k: 1536, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 1536,
    k: 2048,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 6144, k: 1536, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
  {
    storage: "i4",
    n: 1536,
    k: 6144,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i4", n: 1536, k: 256, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 1536, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
  { storage: "i4", n: 512, k: 1536, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 1536,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 12288, k: 1536, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
  {
    storage: "i4",
    n: 1536,
    k: 12288,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i8", n: 262144, k: 1536, lanes: 16, packedActivations: false },
  { storage: "i4", n: 2048, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 256, k: 1536, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i4", n: 6144, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i8", n: 256, k: 1536, lanes: 32, packedActivations: false },
  { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 1536, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
  { storage: "i2", n: 12288, k: 1536, lanes: 2, packedActivations: false },
  { storage: "i2", n: 1536, k: 12288, lanes: 32, packedActivations: true },
  { storage: "i4", n: 10752, k: 2560, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
  { storage: "i4", n: 2048, k: 2560, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 512, k: 2560, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 256, k: 2560, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 2560,
    k: 2048,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 10240, k: 2560, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
  {
    storage: "i4",
    n: 2560,
    k: 10240,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i4", n: 2560, k: 256, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 4096, k: 2560, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 1024, k: 2560, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 2560,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i8", n: 262144, k: 2560, lanes: 16, packedActivations: false },
  { storage: "i4", n: 2048, k: 2560, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 2560, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 2560, k: 2048, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i4", n: 10240, k: 2560, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i8", n: 256, k: 2560, lanes: 32, packedActivations: false },
  { storage: "i8", n: 2560, k: 256, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 2560, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1024, k: 2560, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 2560, k: 4096, group: 4096, lanes: 32, packedActivations: true },
  { storage: "i4", n: 256, k: 3072, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 256, k: 1024, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 15360, k: 3840, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 8192, k: 3840, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 3840, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 2048, k: 3840, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 512, k: 3840, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 3840,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 3840,
    k: 8192,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  {
    storage: "i4",
    n: 3840,
    k: 15360,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 262144, k: 3840, group: 32, lanes: 16, packedActivations: false },
  {
    storage: "i4",
    n: 1024,
    k: 7680,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 1024,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 1024,
    k: 8192,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 8192, k: 2816, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
  { storage: "i4", n: 4096, k: 2816, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 2048, k: 2816, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 2112, k: 2816, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 1024, k: 2816, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 128, k: 2816, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 2816,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 2816,
    k: 8192,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 2816,
    k: 2112,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i4", n: 262144, k: 2816, group: 32, lanes: 4, packedActivations: false },
  { storage: "bf16", n: 2048, k: 1536, lanes: 16, packedActivations: false },
  { storage: "bf16", n: 256, k: 1536, lanes: 32, packedActivations: false },
  { storage: "bf16", n: 6144, k: 1536, lanes: 8, packedActivations: false },
  { storage: "bf16", n: 1536, k: 6144, lanes: 16, packedActivations: false },
  { storage: "bf16", n: 1536, k: 256, lanes: 8, packedActivations: false },
  { storage: "bf16", n: 4096, k: 1536, lanes: 8, packedActivations: false },
  { storage: "bf16", n: 512, k: 1536, lanes: 16, packedActivations: false },
  {
    storage: "bf16",
    n: 1536,
    k: 4096,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "bf16", n: 1536, k: 12288, lanes: 16, packedActivations: false },
  {
    storage: "bf16",
    n: 2048,
    k: 2560,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "bf16", n: 512, k: 2560, lanes: 32, packedActivations: false },
  { storage: "bf16", n: 256, k: 2560, lanes: 32, packedActivations: false },
  { storage: "bf16", n: 2560, k: 2048, lanes: 8, packedActivations: false },
  { storage: "bf16", n: 2560, k: 10240, lanes: 16, packedActivations: false },
  { storage: "bf16", n: 4096, k: 2560, lanes: 8, packedActivations: false },
  { storage: "bf16", n: 1024, k: 2560, lanes: 16, packedActivations: false },
  { storage: "bf16", n: 2560, k: 4096, lanes: 16, packedActivations: false },
];

const OLD_ROWS_THREAD_TARGET = 16384;

type Shape = { readonly storage: WeightStorage; readonly n: number; readonly k: number };

const OLD_DEFAULT_VARIANT: LinearGemvVariant = { cols: 32, unroll: 4 };

/** The sequential exceptions added since the kernel module's one (the default profile's order). */
const ADDED_SEQUENTIAL_SHAPES: readonly LinearGemvShapeVariant[] = [
  { storage: "bf16", n: 8960, k: 1536, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 2048, k: 1536, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 256, k: 1536, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 1536, k: 2048, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 6144, k: 1536, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 1536, k: 6144, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 1536, k: 256, variant: { cols: 32, unroll: 16 } },
  { storage: "bf16", n: 4096, k: 1536, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 512, k: 1536, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 1536, k: 4096, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 12288, k: 1536, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 1536, k: 12288, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 10752, k: 2560, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 2048, k: 2560, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 512, k: 2560, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 2560, k: 2048, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 2560, k: 10240, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 2560, k: 256, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 4096, k: 2560, variant: { cols: 16, unroll: 8 } },
  { storage: "bf16", n: 1024, k: 2560, variant: { cols: 16, unroll: 16 } },
  { storage: "bf16", n: 2560, k: 4096, variant: { cols: 16, unroll: 8 } },
];

/**
 * The sequential exceptions `nvidia-ampere` carries after the default profile's (the generator's
 * order: the order of the sweep's case table): what the RTX 3080 Ti's GEMV sweep records of
 * 2026-11-20/21 adopt and the default profile does not hold — 37 int4 and 11 int8 shapes of the
 * Gemma 4 QAT models and their MTP drafters, and two bf16 ones, the output head of regular E2B,
 * 262144 × 1536, and 10240 × 2560 of regular E4B. The default's 22 are there with the default's
 * variant: no variant reaches the ×1.10 that replaces one.
 */
const AMPERE_SEQUENTIAL_SHAPES: readonly LinearGemvShapeVariant[] = [
  { storage: "i4", n: 256, k: 1536, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 1536, k: 6144, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 1536, k: 4096, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 12288, k: 1536, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 1536, k: 12288, variant: { cols: 32, unroll: 16 } },
  { storage: "i8", n: 262144, k: 1536, variant: { cols: 16, unroll: 4 } },
  { storage: "i4", n: 262144, k: 1536, variant: { cols: 16, unroll: 1 } },
  { storage: "i4", n: 256, k: 3072, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 256, k: 1024, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 256, k: 2048, variant: { cols: 16, unroll: 16 } },
  { storage: "i8", n: 256, k: 3072, variant: { cols: 16, unroll: 16 } },
  { storage: "i8", n: 1024, k: 256, variant: { cols: 32, unroll: 16 } },
  { storage: "i8", n: 256, k: 1024, variant: { cols: 16, unroll: 16 } },
  { storage: "i8", n: 2048, k: 256, variant: { cols: 32, unroll: 16 } },
  { storage: "i8", n: 256, k: 2048, variant: { cols: 16, unroll: 16 } },
  { storage: "i8", n: 256, k: 1536, variant: { cols: 16, unroll: 16 } },
  { storage: "i8", n: 1536, k: 256, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 2048, k: 2560, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 2560, k: 2048, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 2560, k: 10240, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 1024, k: 2560, variant: { cols: 16, unroll: 16 } },
  { storage: "i8", n: 262144, k: 2560, variant: { cols: 16, unroll: 4 } },
  { storage: "i4", n: 262144, k: 2560, variant: { cols: 16, unroll: 2 } },
  { storage: "i4", n: 256, k: 5120, variant: { cols: 16, unroll: 16 } },
  { storage: "i8", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
  { storage: "i8", n: 2560, k: 256, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 4096, k: 3840, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 2048, k: 3840, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 512, k: 3840, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 3840, k: 4096, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 3840, k: 8192, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 262144, k: 3840, variant: { cols: 16, unroll: 2 } },
  { storage: "i4", n: 1024, k: 7680, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 4096, k: 1024, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 1024, k: 4096, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 8192, k: 1024, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 1024, k: 8192, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 3840, k: 1024, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 262144, k: 1024, variant: { cols: 16, unroll: 8 } },
  { storage: "i4", n: 2048, k: 2816, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 2112, k: 2816, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 1024, k: 2816, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 128, k: 2816, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 2816, k: 4096, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 2816, k: 8192, variant: { cols: 32, unroll: 16 } },
  { storage: "i4", n: 2816, k: 2112, variant: { cols: 16, unroll: 16 } },
  { storage: "i4", n: 262144, k: 2816, variant: { cols: 16, unroll: 1 } },
  { storage: "bf16", n: 262144, k: 1536, variant: { cols: 16, unroll: 4 } },
  { storage: "bf16", n: 10240, k: 2560, variant: { cols: 32, unroll: 8 } },
];

/**
 * The default profile's parallel rows that `nvidia-ampere` holds with what those records adopt in
 * place (the default's order): other lanes, the staged flag or a row block on 37 integer-stored
 * rows, and bf16 512 × 1536 at 32 lanes in place of 16.
 */
const AMPERE_CHANGED_PARALLEL_SHAPES: readonly LinearGemvParallelShape[] = [
  {
    storage: "i4",
    n: 1536,
    k: 2048,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i4", n: 1536, k: 256, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 512, k: 1536, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 1536,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i8", n: 262144, k: 1536, lanes: 16, packedActivations: false, rowBlock: 4 },
  {
    storage: "i4",
    n: 2048,
    k: 1536,
    group: 512,
    lanes: 16,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  { storage: "i4", n: 256, k: 1536, group: 512, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 6144,
    k: 1536,
    group: 512,
    lanes: 16,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  {
    storage: "i4",
    n: 4096,
    k: 1536,
    group: 512,
    lanes: 16,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  { storage: "i4", n: 512, k: 1536, group: 512, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i2",
    n: 12288,
    k: 1536,
    lanes: 4,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 8,
  },
  {
    storage: "i4",
    n: 2048,
    k: 2560,
    group: 32,
    lanes: 16,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  { storage: "i4", n: 512, k: 2560, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 256, k: 2560, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 2560,
    k: 2048,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  {
    storage: "i4",
    n: 1024,
    k: 2560,
    group: 32,
    lanes: 16,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i8", n: 262144, k: 2560, lanes: 16, packedActivations: false, rowBlock: 4 },
  {
    storage: "i4",
    n: 2048,
    k: 2560,
    group: 512,
    lanes: 16,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  { storage: "i4", n: 512, k: 2560, group: 512, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 10240,
    k: 2560,
    group: 512,
    lanes: 8,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  {
    storage: "i4",
    n: 4096,
    k: 2560,
    group: 512,
    lanes: 16,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  {
    storage: "i4",
    n: 1024,
    k: 2560,
    group: 512,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i4", n: 256, k: 3072, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 256, k: 1024, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 15360,
    k: 3840,
    group: 32,
    lanes: 8,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 8,
  },
  {
    storage: "i4",
    n: 3840,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  {
    storage: "i4",
    n: 3840,
    k: 15360,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  {
    storage: "i4",
    n: 262144,
    k: 3840,
    group: 32,
    lanes: 8,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  {
    storage: "i4",
    n: 1024,
    k: 7680,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  {
    storage: "i4",
    n: 1024,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  {
    storage: "i4",
    n: 1024,
    k: 8192,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  {
    storage: "i4",
    n: 1024,
    k: 2816,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  { storage: "i4", n: 128, k: 2816, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 2816,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  {
    storage: "i4",
    n: 2816,
    k: 8192,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 2,
  },
  {
    storage: "i4",
    n: 2816,
    k: 2112,
    group: 32,
    lanes: 16,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  {
    storage: "i4",
    n: 262144,
    k: 2816,
    group: 32,
    lanes: 8,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 8,
  },
  { storage: "bf16", n: 512, k: 1536, lanes: 32, packedActivations: false },
];

/**
 * The parallel rows `nvidia-ampere` adds after the default profile's, on shapes the default has
 * none for: int4 4 and int8 5 (shapes of the MTP drafters), and bf16 1536 × 2048 of regular E2B
 * and 2560 × 256 of regular E4B.
 */
const AMPERE_PARALLEL_SHAPES: readonly LinearGemvParallelShape[] = [
  { storage: "i4", n: 2048, k: 256, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
  { storage: "i4", n: 256, k: 2048, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  { storage: "i8", n: 256, k: 3072, lanes: 32, packedActivations: false },
  { storage: "i8", n: 1024, k: 256, lanes: 4, packedActivations: false },
  { storage: "i8", n: 256, k: 1024, lanes: 32, packedActivations: false },
  { storage: "i8", n: 2048, k: 256, lanes: 4, packedActivations: false },
  { storage: "i8", n: 256, k: 2048, lanes: 32, packedActivations: false },
  { storage: "i4", n: 256, k: 5120, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
  {
    storage: "i4",
    n: 3840,
    k: 1024,
    group: 32,
    lanes: 16,
    packedActivations: false,
    stagedActivations: true,
    rowBlock: 4,
  },
  { storage: "bf16", n: 1536, k: 2048, lanes: 32, packedActivations: false },
  { storage: "bf16", n: 2560, k: 256, lanes: 8, packedActivations: false },
];

/** The pinned sequential variant of a shape: a row added since, otherwise the kernel module's lookup. */
const pinnedDefaultVariant = (shape: Shape): LinearGemvVariant =>
  ADDED_SEQUENTIAL_SHAPES.find(
    (row) => row.storage === shape.storage && row.n === shape.n && row.k === shape.k,
  )?.variant ?? OLD_DEFAULT_VARIANT;

/** The old row-block height (the ceiling of 256 elements per word stayed in the kernel). */
const oldRowsForShape = (
  storage: WeightStorage,
  m: number,
  n: number,
  threadTarget: number = OLD_ROWS_THREAD_TARGET,
): number => {
  const cap = 256 / linearGemvUnit(storage);
  let rows = 1;
  while (rows * 2 <= Math.min(m, cap)) rows *= 2;
  while (rows > 1 && n * Math.ceil(m / rows) < threadTarget) rows /= 2;
  return rows;
};

const oldParallelShapeFor = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelShape | undefined => {
  if (m < 1 || m > 8) return undefined;
  return PINNED_PARALLEL_SHAPES.find(
    (shape) => shape.storage === storage && shape.n === n && shape.k === k && shape.group === group,
  );
};

const oldParallelEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  if (!gemmUsesVec4(k, n)) return undefined;
  if (k <= 0 || k % linearGemvUnit(storage) !== 0) return undefined;
  if (storage === "i4") {
    if (group === undefined || group % linearGemvUnit("i4") !== 0) return undefined;
  } else if (group !== undefined) return undefined;
  return oldParallelShapeFor(storage, m, n, k, group)?.lanes;
};

const oldPackedEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  const lanes = oldParallelEligible(storage, m, n, k, group);
  if (lanes === undefined) return undefined;
  return oldParallelShapeFor(storage, m, n, k, group)?.packedActivations === true
    ? lanes
    : undefined;
};

const oldStagedEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  const lanes = oldParallelEligible(storage, m, n, k, group);
  if (lanes === undefined) return undefined;
  const shape = oldParallelShapeFor(storage, m, n, k, group);
  return shape?.stagedActivations === true && !shape.packedActivations ? lanes : undefined;
};

/** The four parallel-family answers for one shape, new (with `choices`) or old. */
const parallelAnswers = (
  choices: LinearGemvChoices | undefined,
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): readonly (LinearGemvParallelLanes | undefined)[] =>
  choices === undefined
    ? [
        oldParallelShapeFor(storage, m, n, k, group)?.lanes,
        oldParallelEligible(storage, m, n, k, group),
        oldPackedEligible(storage, m, n, k, group),
        oldStagedEligible(storage, m, n, k, group),
      ]
    : [
        linearGemvParallelLanes(choices, storage, m, n, k, group),
        linearGemvParallelEligible(choices, storage, m, n, k, group),
        linearGemvPackedEligible(choices, storage, m, n, k, group),
        linearGemvStagedEligible(choices, storage, m, n, k, group),
      ];

const STORAGES: readonly WeightStorage[] = ["i2", "i4", "i8", "f16", "bf16", "f32"];

describe("the default profile holds the pinned GEMV choices (the kernel module's, and the rows added since)", () => {
  it("holds the pinned table of parallel shapes (rows, order and flags), the kernel module's sequential variant and its thread target", () => {
    assertEquals(GEMV.parallelShapes, PINNED_PARALLEL_SHAPES);
    assertEquals(GEMV.rowsThreadTarget, OLD_ROWS_THREAD_TARGET);
    assertEquals(GEMV.sequential, {
      variant: { cols: 32, unroll: 4 },
      shapes: ADDED_SEQUENTIAL_SHAPES,
    });
  });

  it("answers every row of the pinned table as the kernel module's lookups did, inside and past the parallel range of M", () => {
    for (const { storage, n, k, group } of PINNED_PARALLEL_SHAPES) {
      for (const m of [0, 1, 2, 4, 8, 9, 64]) {
        assertEquals(
          parallelAnswers(GEMV, storage, m, n, k, group),
          parallelAnswers(undefined, storage, m, n, k, group),
          `${storage} ${n}×${k} g${group} M${m}`,
        );
      }
    }
  });

  it("answers a grid of shapes off the table as the old lookups did", () => {
    let compared = 0;
    for (const storage of STORAGES) {
      for (const n of [64, 256, 260, 1536, 2048, 8960, 262144]) {
        for (const k of [256, 1536, 1552, 2048, 2560]) {
          for (const group of [undefined, 32, 64, 512]) {
            for (const m of [1, 8, 9]) {
              assertEquals(
                parallelAnswers(GEMV, storage, m, n, k, group),
                parallelAnswers(undefined, storage, m, n, k, group),
                `${storage} ${n}×${k} g${group} M${m}`,
              );
              compared += 1;
            }
          }
        }
      }
    }
    assertEquals(compared, 2520);
  });

  it("chooses the old sequential variant, the variant of a row added since on that row's shape alone, and the old row-block height", () => {
    assertEquals(defaultLinearGemvVariant(GEMV), OLD_DEFAULT_VARIANT);
    let added = 0;
    for (const storage of STORAGES) {
      for (const [n, k] of [
        [262144, 1536],
        [262144, 2560],
        [262140, 1536],
        [6144, 1536],
        [1536, 256],
        [64, 64],
      ]) {
        const pinned = pinnedDefaultVariant({ storage, n, k });
        assertEquals(
          defaultLinearGemvVariant(GEMV, { storage, n, k }),
          pinned,
          `${storage} ${n}×${k}`,
        );
        if (
          pinned.cols !== OLD_DEFAULT_VARIANT.cols ||
          pinned.unroll !== OLD_DEFAULT_VARIANT.unroll
        ) {
          added += 1;
        }
      }
    }
    // bf16 6144 × 1536 and 1536 × 256: the grid meets rows added since. The int4 and int8 output
    // heads of the grid are no row of the default profile.
    assertEquals(added, 2);
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const n of [1, 256, 2048, 4096, 12288, 262144]) {
        for (let m = 1; m <= 64; m += 1) {
          assertEquals(
            linearGemvRowsForShape(GEMV, storage, m, n),
            oldRowsForShape(storage, m, n),
            `${storage} n=${n} m=${m}`,
          );
          assertEquals(
            defaultLinearGemvRowsVariant(GEMV, storage, m, n),
            { ...OLD_DEFAULT_VARIANT, rows: oldRowsForShape(storage, m, n) },
            `${storage} n=${n} m=${m}: the row-block variant`,
          );
        }
      }
    }
  });
});

// ---------------------------------------------------------------------------
// The profile is the source: a table with other choices changes what a linear selects and the key
// of the kernel it builds (through the selection and the program the recipe builder uses).
// ---------------------------------------------------------------------------

/** The Session's omitted values. */
const KNOBS: LinearKnobs = {
  linearCompute: "f32",
  linearGemvReduce: "sequential",
  integerGemv: false,
  linearGemvPlanes: false,
  linearGemvRowsThreadTarget: undefined,
  linearSplitK: false,
};

const REORDER: Partial<LinearKnobs> = { linearGemvReduce: "parallel" };

/** The plan of one `linear` node `[m, k] × [n, k]ᵀ + [n]`. */
const linearPlan = (m: number, n: number, k: number): NodePlan => {
  const ir = mergeGraph({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  });
  const [plan] = planGraph(ir, {}).nodes;
  return plan;
};

type Linear = {
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly storage: WeightStorage;
  readonly group?: number;
  readonly knobs?: Partial<LinearKnobs>;
  /** Whether the linear is declared to read packed activations (the fusion pass's pairing). */
  readonly packed?: boolean;
};

/** The pipeline key of the one dispatch `linear` builds under `choices`. */
const keyOf = (choices: LinearGemvChoices, linear: Linear): string => {
  const { m, n, k, storage, group, packed = false } = linear;
  const plan = linearPlan(m, n, k);
  const choice = selectLinearStrategy({
    plan,
    weightStorage: storage,
    groupSize: group,
    knobs: { ...KNOBS, ...linear.knobs },
    linearGemv: choices,
    packed: packed ? { role: "read" } : undefined,
    prologue: undefined,
  });
  const [x, weight] = plan.inputShapes;
  const program = linearProgramFor(choice, {
    x,
    weight,
    m,
    n,
    k,
    weightStorage: storage,
    groupSize: group,
    packedScale: packed ? 0.5 : undefined,
    prologue: undefined,
    geometryProfile: DEFAULT_GEOMETRY_PROFILE,
    linearI8a8Dot: "dp4a",
  });
  assertEquals(program.dispatches.length, 1);
  return program.dispatches[0].key;
};

/** A measured row of the default table: i4 g32, lanes 4, neither packed nor staged. */
const WIDE: Linear = { m: 1, n: 8960, k: 1536, storage: "i4", group: 32, knobs: REORDER };
/** A measured row of the default table: i4 g32, lanes 32, staged. */
const STAGED: Linear = { m: 1, n: 3840, k: 4096, storage: "i4", group: 32, knobs: REORDER };

/** Whether a row of a table names the shape of `linear`. */
const rowOf =
  (linear: Linear) =>
  (row: LinearGemvParallelShape): boolean =>
    row.storage === linear.storage &&
    row.n === linear.n &&
    row.k === linear.k &&
    row.group === linear.group;

/** The default table's row for the shape of `linear` (a missing row is a bug of the test). */
const measured = (linear: Linear): LinearGemvParallelShape => {
  const row = GEMV.parallelShapes.find(rowOf(linear));
  if (row === undefined) throw new Error(`the default table names no row ${linear.n}×${linear.k}`);
  return row;
};

/** The default choices with the row of `linear` replaced by `row` (dropped when `undefined`). */
const withRow = (linear: Linear, row: LinearGemvParallelShape | undefined): LinearGemvChoices => {
  const kept = GEMV.parallelShapes.filter((candidate) => !rowOf(linear)(candidate));
  return { ...GEMV, parallelShapes: row === undefined ? kept : [row, ...kept] };
};

describe("a profile with other GEMV choices changes the selection and the pipeline key", () => {
  it("the sequential variant: the M=1 kernel and every row block take the table's", () => {
    const other: LinearGemvChoices = {
      ...GEMV,
      sequential: { ...GEMV.sequential, variant: { cols: 64, unroll: 2 } },
    };
    const decode: Linear = { m: 1, n: 64, k: 64, storage: "i4", group: 32 };
    const verify: Linear = { ...decode, m: 8 };
    assertEquals(keyOf(GEMV, decode), "linear_gemv:v1:f32:c32u4:wi4g32");
    assertEquals(keyOf(other, decode), "linear_gemv:v1:f32:c64u2:wi4g32");
    assertEquals(keyOf(GEMV, verify), "linear_gemv:v2:f32:c32u4r1:wi4g32");
    assertEquals(keyOf(other, verify), "linear_gemv:v2:f32:c64u2r1:wi4g32");
  });

  it("a per-shape exception of the sequential variant: only the named shape takes it, and a table without the default's exception drops it", () => {
    const other: LinearGemvChoices = {
      ...GEMV,
      sequential: {
        variant: GEMV.sequential.variant,
        shapes: [{ storage: "i4", n: 64, k: 64, variant: { cols: 16, unroll: 4 } }],
      },
    };
    const named: Linear = { m: 1, n: 64, k: 64, storage: "i4", group: 32 };
    assertEquals(keyOf(other, named), "linear_gemv:v1:f32:c16u4:wi4g32");
    assertEquals(keyOf(other, { ...named, n: 128 }), "linear_gemv:v1:f32:c32u4:wi4g32");
    const exception: Linear = { m: 1, n: 8960, k: 1536, storage: "bf16" };
    assertEquals(keyOf(GEMV, exception), "linear_gemv:v1:f32:c16u8:wbf16");
    assertEquals(keyOf(other, exception), "linear_gemv:v1:f32:c32u4:wbf16");
  });

  it("the row-block thread target: the table's moves the row count, and SessionOptions.linearGemvRowsThreadTarget replaces it", () => {
    const other: LinearGemvChoices = { ...GEMV, rowsThreadTarget: 2048 };
    const block: Linear = { m: 4, n: 2048, k: 64, storage: "i4", group: 32 };
    assertEquals(keyOf(GEMV, block), "linear_gemv:v2:f32:c32u4r1:wi4g32");
    assertEquals(keyOf(other, block), "linear_gemv:v2:f32:c32u4r4:wi4g32");
    // The option replaces the profile's target in either direction
    assertEquals(
      keyOf(other, { ...block, knobs: { linearGemvRowsThreadTarget: 16384 } }),
      "linear_gemv:v2:f32:c32u4r1:wi4g32",
    );
    assertEquals(
      keyOf(GEMV, { ...block, knobs: { linearGemvRowsThreadTarget: 2048 } }),
      "linear_gemv:v2:f32:c32u4r4:wi4g32",
    );
  });

  it("a parallel row added: a shape the default table does not name takes the parallel GEMV with the row's lanes", () => {
    const shape: Linear = { m: 1, n: 64, k: 1536, storage: "i4", group: 32, knobs: REORDER };
    const other = withRow(shape, {
      storage: "i4",
      n: 64,
      k: 1536,
      group: 32,
      lanes: 8,
      packedActivations: false,
    });
    assertEquals(keyOf(GEMV, shape), "linear_gemv:v1:f32:c32u4:wi4g32");
    assertEquals(keyOf(other, shape), "linear_gemv_parallel:wi4g32:l8");
  });

  it("a parallel row removed: a measured shape falls back to the sequential GEMV even at reorder", () => {
    const other = withRow(WIDE, undefined);
    assertEquals(keyOf(GEMV, WIDE), "linear_gemv_parallel:wi4g32:l4");
    assertEquals(keyOf(other, WIDE), "linear_gemv:v1:f32:c32u4:wi4g32");
  });

  it("the packed flag: a row marked packed reads packed activations, a row without it refuses the pairing", () => {
    const other = withRow(WIDE, { ...measured(WIDE), packedActivations: true });
    const { m, n, k, storage, group } = WIDE;
    assertEquals(linearGemvPackedEligible(GEMV, storage, m, n, k, group), undefined);
    assertEquals(linearGemvPackedEligible(other, storage, m, n, k, group), 4);
    assertEquals(
      keyOf(other, { ...WIDE, packed: true }),
      "linear_gemv_parallel:wi4g32:l4:packed-x-i8",
    );
    const refused = resolveLinearStrategy({
      plan: linearPlan(m, n, k),
      weightStorage: storage,
      groupSize: group,
      knobs: { ...KNOBS, ...REORDER },
      linearGemv: GEMV,
      packed: { role: "read" },
      prologue: undefined,
    });
    assert("refusal" in refused, "the default table pairs the row");
  });

  it("the staged flag: a row marked staged takes the staged kernel, a row without it the plain one", () => {
    assertEquals(keyOf(GEMV, WIDE), "linear_gemv_parallel:wi4g32:l4");
    assertEquals(
      keyOf(withRow(WIDE, { ...measured(WIDE), stagedActivations: true }), WIDE),
      "linear_gemv_parallel:wi4g32:l4:staged-x",
    );
    assertEquals(keyOf(GEMV, STAGED), "linear_gemv_parallel:wi4g32:l32:staged-x");
    const { stagedActivations: _staged, ...unstaged } = measured(STAGED);
    assertEquals(keyOf(withRow(STAGED, unstaged), STAGED), "linear_gemv_parallel:wi4g32:l32");
  });

  it("the row block: a row that carries one takes the row-blocked kernel at M = 2..8 in y tiles of its rows, and a row without it one workgroup row per row (fault: the field removed)", () => {
    const { rowBlock: _rowBlock, ...bare } = measured(WIDE);
    const without = withRow(WIDE, bare);
    const blocked = withRow(WIDE, { ...bare, rowBlock: 4 });
    const at = (m: number, knobs: Partial<LinearKnobs> = REORDER): Linear => ({
      ...WIDE,
      m,
      knobs,
    });
    for (const [m, key] of [
      [2, "linear_gemv_parallel:wi4g32:l4:staged-x:r2"],
      [3, "linear_gemv_parallel:wi4g32:l4:staged-x:r2"],
      [4, "linear_gemv_parallel:wi4g32:l4:staged-x:r4"],
      [8, "linear_gemv_parallel:wi4g32:l4:staged-x:r4"],
    ] as const) {
      assertEquals(keyOf(blocked, at(m)), key, `M=${m}`);
      assertEquals(keyOf(without, at(m)), "linear_gemv_parallel:wi4g32:l4", `M=${m}`);
    }
    // A cap of 8 gives M = 8 one block
    assertEquals(
      keyOf(withRow(WIDE, { ...bare, rowBlock: 8 }), at(8)),
      "linear_gemv_parallel:wi4g32:l4:staged-x:r8",
    );
    // M = 1, the level exact, a row count past the parallel range and the subgroup form keep theirs
    assertEquals(keyOf(blocked, at(1)), "linear_gemv_parallel:wi4g32:l4");
    assertEquals(keyOf(blocked, at(4, {})), keyOf(without, at(4, {})));
    assertEquals(keyOf(blocked, at(9)), keyOf(without, at(9)));
    assertEquals(
      keyOf(blocked, at(4, { linearGemvReduce: "parallel-subgroup32" })),
      "linear_gemv_parallel:wi4g32:l4:subgroup32",
    );
    // The grid: 32 columns a workgroup at 4 lanes, and the block's rows a y tile
    const plan = linearPlan(8, WIDE.n, WIDE.k);
    const choice = selectLinearStrategy({
      plan,
      weightStorage: "i4",
      groupSize: 32,
      knobs: { ...KNOBS, ...REORDER },
      linearGemv: blocked,
      packed: undefined,
      prologue: undefined,
    });
    assert("parallelRows" in choice && choice.parallelRows === 4, JSON.stringify(choice));
    const [x, weight] = plan.inputShapes;
    const [dispatch] = linearProgramFor(choice, {
      x,
      weight,
      m: 8,
      n: WIDE.n,
      k: WIDE.k,
      weightStorage: "i4",
      groupSize: 32,
      packedScale: undefined,
      prologue: undefined,
      geometryProfile: DEFAULT_GEOMETRY_PROFILE,
      linearI8a8Dot: "dp4a",
    }).dispatches;
    assertEquals(dispatch.grid, {
      kind: "tiled",
      axes: [
        { items: WIDE.n, tile: 32 },
        { items: 8, tile: 4 },
        { items: 1, tile: 1 },
      ],
    });
  });

  it("the fusion pass reads the same table: linear → static_quantize fuses only on a shape the table names", () => {
    const { n, k } = WIDE;
    const ir = mergeGraph({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear", "static_quantize"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [1, k] }],
      outputs: ["q"],
      initializers: { w: {}, b: {} },
      values: {
        w: { dtype: "f32", shape: [n, k] },
        b: { dtype: "f32", shape: [n] },
        y: { dtype: "f32", shape: [1, n] },
        q: { dtype: "f32", shape: [1, n] },
      },
      nodes: [
        { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
        { op: "static_quantize", ins: ["y"], outs: ["q"], attrs: { scale: Math.fround(0.021) } },
      ],
    });
    const fusedWith = (choices: LinearGemvChoices): number =>
      planFusions(planGraph(ir, {}).nodes, {
        rules: FUSION_RULES,
        linearGemv: choices,
        useCounts: countUses(ir),
        outputNames: new Set(ir.outputs),
        limits: {
          maxStorageBufferBindingSize: 128 * 1024 * 1024,
          maxComputeWorkgroupsPerDimension: 65535,
        },
        linearGemvReduce: "parallel",
        linearCompute: "f32",
        weightLayouts: new Map<string, FusionWeightLayout>([
          ["w", { storage: "i4", groupSize: 32 }],
        ]),
      }).counts.linearStaticQuantize;
    assertEquals(fusedWith(GEMV), 1);
    assertEquals(fusedWith(withRow(WIDE, undefined)), 0);
  });
});

// ---------------------------------------------------------------------------
// Resolution: a table without the field runs with the default profile's choices.
// ---------------------------------------------------------------------------

const ZERO_LIMITS: RequiredLimits = {
  maxBufferSize: 0,
  maxStorageBufferBindingSize: 0,
  maxUniformBufferBindingSize: 0,
  maxStorageBuffersPerShaderStage: 0,
  maxUniformBuffersPerShaderStage: 0,
  maxComputeWorkgroupStorageSize: 0,
  maxComputeInvocationsPerWorkgroup: 0,
  maxComputeWorkgroupSizeX: 0,
  maxComputeWorkgroupSizeY: 0,
  maxComputeWorkgroupSizeZ: 0,
  maxComputeWorkgroupsPerDimension: 0,
};

/** A real GpuContext over a fake device, with `injected` as the table `acquireGpu` was given. */
const gpuWith = (injected: GeometryProfile | undefined): GpuContext =>
  new GpuContext(fakeDevice(), readAdapterInfo({}), ZERO_LIMITS, new Set(), undefined, injected);

/** A table without the GEMV choices (the shape of every embedded table). */
const { linearGemv: _default, ...WITHOUT } = { ...DEFAULT_GEOMETRY_PROFILE, id: "without-gemv" };

/** A table with GEMV choices of its own. */
const OWN: GeometryProfile = {
  ...WITHOUT,
  id: "own-gemv",
  linearGemv: { ...GEMV, rowsThreadTarget: 2048 },
};

describe("a table without the GEMV choices runs with the default profile's, resolved where a Session selects its profile", () => {
  /** `[storage, n, k]` of a sequential exception, which has no group length. */
  const shapeKey = ({ storage, n, k }: { storage: string; n: number; k: number }): string =>
    [storage, n, k].join(" ");

  /** A parallel row's shape: `[storage, n, k, group]`. */
  const rowKey = (row: LinearGemvParallelShape): string =>
    [row.storage, row.n, row.k, row.group ?? "-"].join(" ");

  it("an embedded table without the field runs with the default profile's choices, and every embedded table carries its own; nvidia-ampere's are, row for row, the default's with 38 of its parallel rows changed in place and 50 sequential exceptions and 11 parallel rows after them", () => {
    const carrying: string[] = [];
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      if ("linearGemv" in profile) {
        carrying.push(profile.id);
        continue;
      }
      assertStrictEquals(resolveGeometryProfile(profile).linearGemv, GEMV, profile.id);
    }
    assertEquals(carrying, ["apple-metal-3", "nvidia-ampere", "nvidia-blackwell"]);
    const ampere = BUILTIN_GEOMETRY_PROFILES.find(({ id }) => id === "nvidia-ampere");
    assert(ampere !== undefined);
    const changed = new Map(AMPERE_CHANGED_PARALLEL_SHAPES.map((row) => [rowKey(row), row]));
    // Every changed row is a row of the default, and none is pinned twice
    assertEquals(changed.size, 38);
    assertEquals(GEMV.parallelShapes.filter((row) => changed.has(rowKey(row))).length, 38);
    // A table holds the field as a whole, so this also pins that its other parallel rows, thread
    // target and general variant are the default's of today: a change of those regenerates the table.
    assertEquals(resolveGeometryProfile(ampere).linearGemv, {
      ...GEMV,
      sequential: {
        variant: GEMV.sequential.variant,
        shapes: [...GEMV.sequential.shapes, ...AMPERE_SEQUENTIAL_SHAPES],
      },
      parallelShapes: [
        ...GEMV.parallelShapes.map((row) => changed.get(rowKey(row)) ?? row),
        ...AMPERE_PARALLEL_SHAPES,
      ],
    });
  });

  /**
   * Pins a table's field against the default's: the general variant and the thread target are the
   * default's of today (the derivation never writes them); every exception of the default is there,
   * in the default's order, with the default's variant except on the shapes `ownVariant` names
   * (each of which the default holds with another variant); `ownExceptions` of the table's own
   * follow; every parallel row of the default is there, in the default's order, `changed` of them
   * other than the default's, and `added` rows follow.
   */
  const assertDerivedFrom = (
    id: string,
    ownVariant: ReadonlyMap<string, LinearGemvVariant>,
    counts: { ownExceptions: number; changed: number; added: number },
  ): void => {
    const table = BUILTIN_GEOMETRY_PROFILES.find((profile) => profile.id === id);
    assert(table?.linearGemv !== undefined, id);
    const own = table.linearGemv;
    assertEquals(own.sequential.variant, GEMV.sequential.variant);
    assertEquals(own.rowsThreadTarget, GEMV.rowsThreadTarget);
    const replaced = GEMV.sequential.shapes.filter((row) => ownVariant.has(shapeKey(row)));
    assertEquals(replaced.length, ownVariant.size);
    for (const row of replaced) {
      assert(JSON.stringify(row.variant) !== JSON.stringify(ownVariant.get(shapeKey(row))));
    }
    const inherited = GEMV.sequential.shapes.length;
    assertEquals(
      own.sequential.shapes.slice(0, inherited),
      GEMV.sequential.shapes.map((row) => {
        const variant = ownVariant.get(shapeKey(row));
        return variant === undefined ? row : { ...row, variant };
      }),
    );
    assertEquals(own.sequential.shapes.length - inherited, counts.ownExceptions);
    assertEquals(
      own.parallelShapes.slice(0, GEMV.parallelShapes.length).map(rowKey),
      GEMV.parallelShapes.map(rowKey),
    );
    const changed = GEMV.parallelShapes.filter(
      (row, index) => JSON.stringify(row) !== JSON.stringify(own.parallelShapes[index]),
    ).length;
    const added = own.parallelShapes.length - GEMV.parallelShapes.length;
    assertEquals([changed, added], [counts.changed, counts.added]);
  };

  it("nvidia-blackwell's are the default's with what its own records adopt in place: two of the default's exceptions with another variant, 41 more sequential exceptions, and 62 parallel rows changed or added", () => {
    assertDerivedFrom(
      "nvidia-blackwell",
      new Map([
        ["bf16 1536 256", { cols: 16, unroll: 8 }],
        ["bf16 4096 2560", { cols: 16, unroll: 16 }],
      ]),
      { ownExceptions: 41, changed: 42, added: 20 },
    );
  });

  it("apple-metal-3's are the default's with what the Apple M2's records adopt in place: 15 of the default's exceptions with another variant, 65 more sequential exceptions, and 65 parallel rows changed or added", () => {
    assertDerivedFrom(
      "apple-metal-3",
      new Map([
        ["bf16 8960 1536", { cols: 128, unroll: 16 }],
        ["bf16 2048 1536", { cols: 256, unroll: 8 }],
        ["bf16 1536 6144", { cols: 256, unroll: 8 }],
        ["bf16 1536 256", { cols: 64, unroll: 2 }],
        ["bf16 4096 1536", { cols: 128, unroll: 16 }],
        ["bf16 512 1536", { cols: 128, unroll: 8 }],
        ["bf16 1536 4096", { cols: 256, unroll: 8 }],
        ["bf16 1536 12288", { cols: 128, unroll: 8 }],
        ["bf16 512 2560", { cols: 128, unroll: 8 }],
        ["bf16 256 2560", { cols: 128, unroll: 8 }],
        ["bf16 2560 2048", { cols: 128, unroll: 8 }],
        ["bf16 2560 10240", { cols: 128, unroll: 16 }],
        ["bf16 2560 256", { cols: 32, unroll: 2 }],
        ["bf16 1024 2560", { cols: 256, unroll: 8 }],
        ["bf16 2560 4096", { cols: 128, unroll: 16 }],
      ]),
      { ownExceptions: 65, changed: 52, added: 13 },
    );
  });

  it("no embedded table removes a parallel row of the default or moves a packed-activation flag: the shapes that read packed activations are the default's on every adapter", () => {
    // The memory estimate has no adapter and plans its fusions with the default profile's choices.
    // The fusion pass reads two things of them: which shapes have a parallel row (the
    // linear → static_quantize fusion) and a row's packed-activation flag (the packed pairing). With
    // this held, a table's plan differs from the estimate's only through the shapes it adds a row
    // for (docs/limitations.md, the `workspaceBytes` section).
    let tables = 0;
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      if (profile.linearGemv === undefined) continue;
      tables += 1;
      const own = profile.linearGemv.parallelShapes;
      for (const row of GEMV.parallelShapes) {
        const match = own.find(
          (one) =>
            one.storage === row.storage &&
            one.n === row.n &&
            one.k === row.k &&
            one.group === row.group,
        );
        assert(match !== undefined, `${profile.id}: no row ${row.storage} ${row.n}×${row.k}`);
        assertEquals(
          match.packedActivations,
          row.packedActivations,
          `${profile.id}: ${row.storage} ${row.n}×${row.k} packedActivations`,
        );
      }
      const packedAdded = own.filter(
        (one) =>
          one.packedActivations &&
          !GEMV.parallelShapes.some(
            (row) =>
              row.storage === one.storage &&
              row.n === one.n &&
              row.k === one.k &&
              row.group === one.group,
          ),
      );
      assertEquals(packedAdded, [], `${profile.id}: an added row with packed activations`);
    }
    assertEquals(tables, 3);
  });

  it("resolveGeometryProfile fills in the default's choices and keeps every other field of the table, and keeps a table's own", () => {
    const resolved = resolveGeometryProfile(WITHOUT);
    assertStrictEquals(resolved.linearGemv, GEMV);
    const { linearGemv: _filled, ...rest } = resolved;
    assertEquals(rest, WITHOUT);
    assertEquals(resolveGeometryProfile(OWN), OWN);
  });

  it("sessionGeometryProfile resolves the injected table and the selected one", () => {
    assertStrictEquals(sessionGeometryProfile(gpuWith(WITHOUT)).linearGemv, GEMV);
    assertEquals(sessionGeometryProfile(gpuWith(WITHOUT)).id, "without-gemv");
    assertEquals(sessionGeometryProfile(gpuWith(OWN)).linearGemv, OWN.linearGemv);
    // An adapter no embedded table matches selects the default profile
    const selected = sessionGeometryProfile(gpuWith(undefined));
    assertEquals(selected.id, "default");
    assertStrictEquals(selected.linearGemv, GEMV);
  });
});

// ---------------------------------------------------------------------------
// The guard: a table carrying the field passes assertGeometryProfile only with well-formed choices.
// ---------------------------------------------------------------------------

/** `OWN` with the value at `path` (inside `linearGemv`) replaced, as a JS caller could pass it. */
const brokenAt = (path: readonly (string | number)[], value: unknown): unknown => {
  const copy: unknown = structuredClone({ ...OWN, id: "broken-gemv" });
  let target: unknown = copy;
  for (const key of path.slice(0, -1)) target = Reflect.get(Object(target), key);
  Reflect.set(Object(target), path[path.length - 1], value);
  return copy;
};

describe("assertGeometryProfile checks the GEMV choices", () => {
  it("accepts the default profile and a table with choices of its own", () => {
    assertGeometryProfile(DEFAULT_GEOMETRY_PROFILE);
    assertGeometryProfile(OWN);
    assertGeometryProfile(WITHOUT);
  });

  it("rejects malformed choices naming the field", () => {
    const where = "Geometry profile 'broken-gemv'";
    for (const [path, value, message] of [
      [["linearGemv"], 5, "linearGemv is not an object"],
      [
        ["linearGemv", "rowsThreadTarget"],
        0,
        "linearGemv.rowsThreadTarget must be a positive integer (0)",
      ],
      [
        ["linearGemv", "rowsThreadTarget"],
        Number.NaN,
        "linearGemv.rowsThreadTarget must be a positive integer (NaN)",
      ],
      [
        ["linearGemv", "sequential", "variant", "cols"],
        300,
        "linearGemv.sequential.variant: cols must be an integer in 1..256 (300)",
      ],
      [
        ["linearGemv", "sequential", "variant", "unroll"],
        "4",
        "linearGemv.sequential.variant: cols and unroll must be numbers",
      ],
      [
        ["linearGemv", "sequential", "shapes", 0, "storage"],
        "i3",
        "linearGemv.sequential.shapes[0].storage must be one of",
      ],
      [["linearGemv", "parallelShapes"], {}, "linearGemv.parallelShapes is not an array"],
      [["linearGemv", "parallelShapes"], new Array(1), "linearGemv.parallelShapes[0] is missing"],
      [
        ["linearGemv", "parallelShapes", 0, "lanes"],
        3,
        "linearGemv.parallelShapes[0].lanes must be one of 2 / 4 / 8 / 16 / 32 (3)",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "storage"],
        "i3",
        "linearGemv.parallelShapes[0].storage must be one of i2 / i4 / i8 / bf16 / f16 / f32",
      ],
      // Row 0 is an i4 row with its group: a float storage takes none
      [
        ["linearGemv", "parallelShapes", 0, "storage"],
        "f32",
        "linearGemv.parallelShapes[0]: a row of f32 storage takes no group and no packedActivations",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "n"],
        0,
        "linearGemv.parallelShapes[0].n must be a positive integer",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "group"],
        1.5,
        "linearGemv.parallelShapes[0].group must be a positive integer",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "packedActivations"],
        undefined,
        "linearGemv.parallelShapes[0].packedActivations is not a boolean",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "stagedActivations"],
        false,
        "linearGemv.parallelShapes[0].stagedActivations must be true or absent",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "rowBlock"],
        3,
        "linearGemv.parallelShapes[0].rowBlock must be one of 2 / 4 / 8 (3)",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "rowBlock"],
        true,
        "linearGemv.parallelShapes[0].rowBlock must be one of 2 / 4 / 8 (true)",
      ],
    ] as const) {
      assertThrows(
        () => assertGeometryProfile(brokenAt(path, value)),
        CodegenError,
        `${where}: ${message}`,
      );
    }
  });

  it("accepts a row block the kernel exists at, and rejects one it does not exist at or one on a packed row", () => {
    const where = "Geometry profile 'broken-gemv'";
    const indexOf = (wanted: (row: LinearGemvParallelShape) => boolean): number => {
      const index = OWN.linearGemv?.parallelShapes.findIndex(wanted) ?? -1;
      assert(index >= 0, "the default table has no such row");
      return index;
    };
    // i4 at 4 lanes: every height. i4 at 32 lanes: 2 rows are the most its workgroup storage holds
    const wide = indexOf(
      (row) => row.storage === "i4" && row.lanes === 4 && !row.packedActivations,
    );
    const narrow = indexOf(
      (row) => row.storage === "i4" && row.lanes === 32 && !row.packedActivations,
    );
    const packed = indexOf((row) => row.packedActivations);
    for (const rows of [2, 4, 8]) {
      assertGeometryProfile(brokenAt(["linearGemv", "parallelShapes", wide, "rowBlock"], rows));
    }
    assertGeometryProfile(brokenAt(["linearGemv", "parallelShapes", narrow, "rowBlock"], 2));
    assertThrows(
      () =>
        assertGeometryProfile(brokenAt(["linearGemv", "parallelShapes", narrow, "rowBlock"], 4)),
      CodegenError,
      `${where}: linearGemv.parallelShapes[${narrow}].rowBlock 4 has no kernel for "i4" at 32 lanes`,
    );
    assertThrows(
      () =>
        assertGeometryProfile(brokenAt(["linearGemv", "parallelShapes", packed, "rowBlock"], 2)),
      CodegenError,
      `${where}: linearGemv.parallelShapes[${packed}].rowBlock is set on a row that takes packed activations`,
    );
  });
});
