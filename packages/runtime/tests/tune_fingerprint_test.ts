// The matching keys of a saved table (src/tune/fingerprint.ts — ADR 0117 decision 4): the kernel
// fingerprint (geometryProfileKernelsId) and the case-set version (sweepCaseSetId). The public
// function (sweepCaseSetId) comes from the public surface (../tune.ts); the fingerprint, which is not
// public (geometryProfileMismatch derives it internally — ADR 0117 decision 2), and the injectable
// bodies and tool functions come from src. No GPU.
//
// Pinned: (1) the hash is FNV-1a 64 bit itself; (2) the fingerprint moves with each of the table's
// 15 fields, the default geometry, both dp4a variants and a dispatch's params and workgroups, and not
// with id / match / provenance or the WGSL's comments and whitespace (ADR 0117 addendum 2026-10-19);
// (3) a dispatch that cannot be built throws naming the case, the field and the geometry; (4) the
// embedded tables' provenance equals the current runtime's values (a change to a GEMM kernel or to
// the cases re-bakes the tables); (5) the case-set version moves with added or removed cases, the
// dispatch-deciding fields and the bounds, and not with informational fields or key order; (6) a
// record's `candidateKernels` (the fingerprint of the measured candidates) moves on a runtime that
// changed only a candidate's WGSL (`defaultKernels` does not), skips failed rows and builds with the
// record's dp4a variant.

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertMatch,
  assertNotEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import { sweepCaseSetId } from "../tune.ts";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import { APPLE_METAL_3 } from "../src/kernels/geometry-profiles/apple-metal-3.ts";
import { PROFILE_GEMM_ROWS_BOUNDS, SWEEP_CASES, type SweepCase } from "../src/tune/cases.ts";
import {
  candidateKernelsId,
  canonicalJson,
  type CasePlanner,
  caseSetId,
  fnv1a64Hex,
  geometryProfileKernelsId,
  KernelsIdError,
  type MeasuredRow,
  profileKernelsId,
  sweepCandidateKernelsId,
  wgslTokenText,
} from "../src/tune/fingerprint.ts";
import {
  type GeometryCandidate,
  quickConv2dCandidates,
  quickGemmCandidates,
  quickI8a8Candidates,
} from "../src/tune/geometries.ts";
import { candidatesFor, type CasePlan, casePlan } from "../src/tune/harness.ts";

type Launch = ReturnType<CasePlan["launch"]>;

/** FNV-1a 64 bit の素朴な実装（BigInt — 指紋の実装と独立のオラクル）。 */
const referenceFnv1a64 = (text: string): string => {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(text)) {
    hash = ((hash ^ BigInt(byte)) * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return hash.toString(16).padStart(16, "0");
};

describe("fnv1a64Hex", () => {
  it("FNV-1a 64 bit の公表値と一致する", () => {
    assertEquals(fnv1a64Hex(""), "cbf29ce484222325");
    assertEquals(fnv1a64Hex("a"), "af63dc4c8601ec8c");
    assertEquals(fnv1a64Hex("foobar"), "85944171f73967e8");
  });

  it("多バイト文字・長い文字列でも BigInt の素朴な実装と一致する（UTF-8 のバイト列のハッシュ）", () => {
    for (const text of [
      "幾何プロファイル",
      "\u0000￿\u{1f600}",
      "x".repeat(10_000),
      JSON.stringify(SWEEP_CASES),
    ]) {
      assertEquals(fnv1a64Hex(text), referenceFnv1a64(text), text.slice(0, 16));
    }
  });
});

/** 欄 1 つの幾何を、同じ族の quick 候補のうち今の値と違う最初のものへ差し替える。 */
const another = <T extends GeometryCandidate["geometry"]>(
  current: T,
  pool: readonly { readonly geometry: T }[],
): T => {
  const found = pool.find(({ geometry }) => JSON.stringify(geometry) !== JSON.stringify(current));
  if (found === undefined) throw new Error("差し替え先の候補が無い");
  return found.geometry;
};

/** One table per field of the 15, with only that field changed (field name → table). */
const singleFieldVariants = (base: GeometryProfile): ReadonlyMap<string, GeometryProfile> => {
  const gemm = quickGemmCandidates();
  const conv2d = quickConv2dCandidates();
  const i8a8 = quickI8a8Candidates();
  const variants = new Map<string, GeometryProfile>();
  base.gemmRows.forEach((rule, index) => {
    variants.set(`gemmRows[${index}]`, {
      ...base,
      gemmRows: base.gemmRows.map((entry, at) =>
        at === index ? { ...entry, geometry: another(rule.geometry, gemm) } : entry,
      ),
    });
  });
  const { attention, conv2d: conv, i8a8: integer } = base;
  variants.set("attention.qk", {
    ...base,
    attention: { ...attention, qk: another(attention.qk, gemm) },
  });
  variants.set("attention.pv", {
    ...base,
    attention: { ...attention, pv: another(attention.pv, gemm) },
  });
  variants.set("conv2d.rows64", {
    ...base,
    conv2d: { ...conv, rows64: another(conv.rows64, conv2d) },
  });
  variants.set("conv2d.rows32", {
    ...base,
    conv2d: { ...conv, rows32: another(conv.rows32, conv2d) },
  });
  variants.set("i8a8.linear", {
    ...base,
    i8a8: { ...integer, linear: another(integer.linear, i8a8) },
  });
  variants.set("i8a8.attentionQk", {
    ...base,
    i8a8: { ...integer, attentionQk: another(integer.attentionQk, i8a8) },
  });
  variants.set("i8a8.attentionPv", {
    ...base,
    i8a8: { ...integer, attentionPv: another(integer.attentionPv, i8a8) },
  });
  variants.set("i8a8.conv3d", {
    ...base,
    i8a8: { ...integer, conv3d: another(integer.conv3d, i8a8) },
  });
  return variants;
};

describe("geometryProfileKernelsId: カーネルの指紋", () => {
  it("同じ表からは同じ値（16 進 16 桁）になり、複製や欄の並びに依らない", () => {
    const value = geometryProfileKernelsId(APPLE_METAL_3);
    assertMatch(value, /^[0-9a-f]{16}$/);
    assertEquals(geometryProfileKernelsId(APPLE_METAL_3), value);
    assertEquals(geometryProfileKernelsId(structuredClone(APPLE_METAL_3)), value);
    const { i8a8, conv2d, attention, gemmRows, id } = APPLE_METAL_3;
    assertEquals(geometryProfileKernelsId({ i8a8, conv2d, attention, gemmRows, id }), value);
  });

  it("id・match・provenance は指紋に効かない（表の幾何だけが効く）", () => {
    const value = geometryProfileKernelsId(APPLE_METAL_3);
    const { match: _match, provenance: _provenance, ...bare } = APPLE_METAL_3;
    assertEquals(geometryProfileKernelsId({ ...bare, id: "other" }), value);
    assertEquals(
      geometryProfileKernelsId({
        ...APPLE_METAL_3,
        match: { vendor: "intel" },
        provenance: {
          sweep: "x.json",
          sha256: "0",
          date: "2000-01-01",
          candidateSet: "quick",
          adapter: { vendor: "x", architecture: "y", device: "z", description: "w" },
          kernels: "0000000000000000",
          caseSet: "0000000000000000",
        },
      }),
      value,
    );
  });

  it("changing any one of the table's 15 fields changes the value, and the 15 values differ from each other", () => {
    const base = geometryProfileKernelsId(APPLE_METAL_3);
    const variants = singleFieldVariants(APPLE_METAL_3);
    assertEquals(variants.size, 15);
    const values = new Set([base]);
    for (const [field, profile] of variants) {
      const value = geometryProfileKernelsId(profile);
      assertNotEquals(value, base, field);
      values.add(value);
    }
    assertEquals(values.size, 16);
  });

  it("gemmRows の段の境界が違う表（既定の 3 段）も、runtime の選択関数でケースに幾何を当てて導ける", () => {
    const value = geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE);
    assertMatch(value, /^[0-9a-f]{16}$/);
    // 既定の表は空の入力のハッシュではない（表と既定の両方のカーネルを数える）
    assertNotEquals(value, fnv1a64Hex(""));
    // 3 段の既定の表と、同じ幾何を 7 段に細分した表は、どのケースにも同じ幾何を当てるので同じ指紋
    const refined: GeometryProfile = {
      ...DEFAULT_GEOMETRY_PROFILE,
      gemmRows: PROFILE_GEMM_ROWS_BOUNDS.map((maxRows) => ({
        maxRows,
        geometry: (
          DEFAULT_GEOMETRY_PROFILE.gemmRows.find((rule) => maxRows <= rule.maxRows) ??
          DEFAULT_GEOMETRY_PROFILE.gemmRows[DEFAULT_GEOMETRY_PROFILE.gemmRows.length - 1]
        ).geometry,
      })),
    };
    assertEquals(geometryProfileKernelsId(refined), value);
  });

  it("既定の幾何（case plan の defaultCandidate）が変わると値が変わる — 比の土台のカーネルも指紋に入る", () => {
    const base = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, casePlan);
    assertEquals(base, geometryProfileKernelsId(APPLE_METAL_3));
    // 1 ケースだけ、既定の幾何を別の候補にした case plan（既定の幾何を変えた runtime を模す）
    const target = SWEEP_CASES.find((sweepCase) => sweepCase.op === "linear");
    assert(target !== undefined);
    const shifted: CasePlanner = (sweepCase, limit, dp4a) => {
      const plan = casePlan(sweepCase, limit, dp4a);
      if (sweepCase !== target) return plan;
      const candidate = quickGemmCandidates().find(
        ({ name }) => name !== plan.defaultCandidate.name,
      );
      assert(candidate !== undefined);
      return { ...plan, defaultCandidate: candidate };
    };
    assertNotEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, shifted), base);
  });

  it("i8a8 のケースは dp4a の両方の変種を数える（どちらの WGSL が変わっても値が変わる）", () => {
    const base = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, casePlan);
    for (const variant of [false, true]) {
      const marked: CasePlanner = (sweepCase, limit, dp4a) => {
        const plan = casePlan(sweepCase, limit, dp4a);
        if (dp4a !== variant) return plan;
        return {
          ...plan,
          launch: (candidate) => {
            const launch = plan.launch(candidate);
            return { ...launch, wgsl: `${launch.wgsl}\nfn changed() {}` };
          },
        };
      };
      const isI8a8 = (sweepCase: SweepCase): boolean =>
        sweepCase.op === "i8a8-linear" ||
        sweepCase.op === "i8a8-attention" ||
        sweepCase.op === "i8a8-conv3d";
      assertNotEquals(
        profileKernelsId(APPLE_METAL_3, SWEEP_CASES.filter(isI8a8), marked),
        profileKernelsId(APPLE_METAL_3, SWEEP_CASES.filter(isI8a8), casePlan),
        `dp4a ${variant}`,
      );
      assertNotEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, marked), base);
    }
  });

  it("the conv3d w8a8 cases count both dp4a variants and read the table's i8a8.conv3d (not another i8a8 field)", () => {
    const conv3dCases = SWEEP_CASES.filter((sweepCase) => sweepCase.op === "i8a8-conv3d");
    assert(conv3dCases.length > 0);
    const plain = profileKernelsId(APPLE_METAL_3, conv3dCases, casePlan);
    for (const variant of [false, true]) {
      const marked: CasePlanner = (sweepCase, limit, dp4a) => {
        const plan = casePlan(sweepCase, limit, dp4a);
        if (dp4a !== variant) return plan;
        return {
          ...plan,
          launch: (candidate) => {
            const launch = plan.launch(candidate);
            return { ...launch, wgsl: `${launch.wgsl}\nfn changed() {}` };
          },
        };
      };
      assertNotEquals(
        profileKernelsId(APPLE_METAL_3, conv3dCases, marked),
        plain,
        `dp4a ${variant}`,
      );
    }
    const { i8a8 } = APPLE_METAL_3;
    const conv3d = quickI8a8Candidates().find(
      ({ geometry }) =>
        ![i8a8.linear, i8a8.attentionQk, i8a8.attentionPv, i8a8.conv3d].some(
          (field) => JSON.stringify(field) === JSON.stringify(geometry),
        ),
    );
    assert(conv3d !== undefined);
    // Only conv3d moves: the conv3d cases see it, the other cases do not
    const moved = { ...APPLE_METAL_3, i8a8: { ...i8a8, conv3d: conv3d.geometry } };
    assertNotEquals(profileKernelsId(moved, conv3dCases, casePlan), plain);
    const others = SWEEP_CASES.filter((sweepCase) => sweepCase.op !== "i8a8-conv3d");
    assertEquals(
      profileKernelsId(moved, others, casePlan),
      profileKernelsId(APPLE_METAL_3, others, casePlan),
    );
  });

  it("dispatch の params だけ・workgroups だけが変わっても値が変わる（キーと WGSL が同じでも別のカーネル）", () => {
    const base = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, casePlan);
    // 全ケースの launch のうち 1 欄だけを差し替えた case plan（その欄だけが違う runtime を模す）
    const altered =
      (change: (launch: Launch) => Launch): CasePlanner =>
      (sweepCase, limit, dp4a) => {
        const plan = casePlan(sweepCase, limit, dp4a);
        return { ...plan, launch: (candidate) => change(plan.launch(candidate)) };
      };
    const params = altered((launch) => ({
      ...launch,
      params: Uint32Array.from(launch.params, (value, index) => (index === 0 ? value + 1 : value)),
    }));
    const workgroups = altered((launch) => ({
      ...launch,
      workgroups: [launch.workgroups[0] + 1, launch.workgroups[1], launch.workgroups[2]],
    }));
    const paramsValue = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, params);
    const workgroupsValue = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, workgroups);
    assertNotEquals(paramsValue, base, "params");
    assertNotEquals(workgroupsValue, base, "workgroups");
    assertNotEquals(paramsValue, workgroupsValue);
  });

  it("WGSL のコメントと空白だけが違う runtime では値が変わらない（トークンが同じなら同じカーネル — ADR 0117 追記 2026-10-19）", () => {
    const base = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, casePlan);
    const rewritten =
      (change: (wgsl: string) => string): CasePlanner =>
      (sweepCase, limit, dp4a) => {
        const plan = casePlan(sweepCase, limit, dp4a);
        return {
          ...plan,
          launch: (candidate) => {
            const launch = plan.launch(candidate);
            return { ...launch, wgsl: change(launch.wgsl) };
          },
        };
      };
    const commented = rewritten(
      (wgsl) => `// 翻訳したコメント /* 入れ子 /* の */ 注記 */\n${wgsl}\n// changed`,
    );
    const stripped = rewritten((wgsl) => wgsl.replace(/\/\/[^\n]*/g, ""));
    const reindented = rewritten((wgsl) => wgsl.replace(/\n[ \t]+/g, "\n\t\t"));
    assertEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, commented), base, "comments added");
    assertEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, stripped), base, "comments removed");
    assertEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, reindented), base, "whitespace");
    // 1 トークンでも変われば別のカーネル
    const token = rewritten((wgsl) => `${wgsl}\nfn changed() {}`);
    assertNotEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, token), base);
  });

  it("組めない dispatch（65535 を超える）は、ケース・表の欄・幾何と 65535 固定の理由を名指して投げる", () => {
    // 門（assertGeometryProfile）は通るが、tileN 4 で conv2d 512² の N = 262144 が 65536 workgroup になる
    const narrow: GeometryProfile = {
      ...APPLE_METAL_3,
      id: "narrow",
      conv2d: { ...APPLE_METAL_3.conv2d, rows32: { regM: 1, regN: 4, wgX: 1, wgY: 16 } },
    };
    const error = assertThrows(() => geometryProfileKernelsId(narrow), KernelsIdError);
    assertInstanceOf(error.cause, DispatchLimitError);
    for (const part of [
      "conv2d-c96-512x512",
      "table 'narrow', conv2d.rows32",
      '{"regM":1,"regN":4,"wgX":1,"wgY":16}',
      "65535",
      "WebGPU guarantees on every device",
    ]) {
      assertStringIncludes(error.message, part);
    }
  });

  it("GPU も時刻も読まない（navigator.gpu・performance.now・Date.now が投げる環境でも同じ値）", () => {
    const expected = geometryProfileKernelsId(APPLE_METAL_3);
    const forbidden = (what: string) => (): never => {
      throw new Error(`${what} を読んだ`);
    };
    // oxlint-disable-next-line typescript/unbound-method -- kept only to reassign in finally; never called
    const originalNow = performance.now;
    const originalDateNow = Date.now;
    Object.defineProperty(navigator, "gpu", {
      get: forbidden("navigator.gpu"),
      configurable: true,
    });
    performance.now = forbidden("performance.now");
    Date.now = forbidden("Date.now");
    try {
      assertEquals(geometryProfileKernelsId(APPLE_METAL_3), expected);
    } finally {
      Reflect.deleteProperty(navigator, "gpu");
      performance.now = originalNow;
      Date.now = originalDateNow;
    }
  });
});

describe("candidateKernelsId: 測った候補のカーネルの指紋（記録の candidateKernels — ADR 0117 決定 8）", () => {
  const caseOf = (op: SweepCase["op"]): SweepCase => {
    const found = SWEEP_CASES.find((sweepCase) => sweepCase.op === op);
    assert(found !== undefined, op);
    return found;
  };
  /** ケースを候補の全幾何で測った行（書き手の行の形のうち指紋が読む欄）。 */
  const rowsOf = (sweepCase: SweepCase, candidates: readonly GeometryCandidate[]): MeasuredRow[] =>
    candidates.map((candidate) => ({ caseId: sweepCase.id, geometryParams: candidate.geometry }));
  const LINEAR = caseOf("linear");
  const I8A8 = caseOf("i8a8-linear");
  const CONV = caseOf("conv2d");
  const F32_ROWS = [
    ...rowsOf(LINEAR, quickGemmCandidates()),
    ...rowsOf(CONV, quickConv2dCandidates()),
  ];
  const ROWS = [...F32_ROWS, ...rowsOf(I8A8, quickI8a8Candidates())];

  it("決定的で（16 進 16 桁）、書き手と読み手の入口は今の runtime の case plan で導いた値", () => {
    const value = sweepCandidateKernelsId(ROWS, false);
    assertMatch(value, /^[0-9a-f]{16}$/);
    assertEquals(sweepCandidateKernelsId(structuredClone(ROWS), false), value);
    assertEquals(candidateKernelsId(ROWS, false, SWEEP_CASES, casePlan), value);
  });

  it("既定を変えず候補の WGSL だけ変えた runtime では値が変わる（既定の指紋 defaultKernels は変わらない）", () => {
    // 既定の幾何と違う幾何の dispatch の WGSL だけに 1 行足した case plan（候補のカーネルだけを直した runtime を模す）
    const candidatesOnly: CasePlanner = (sweepCase, limit, dp4a) => {
      const plan = casePlan(sweepCase, limit, dp4a);
      const defaultGeometry = JSON.stringify(plan.defaultCandidate.geometry);
      return {
        ...plan,
        launch: (candidate) => {
          const launch = plan.launch(candidate);
          return JSON.stringify(candidate.geometry) === defaultGeometry
            ? launch
            : { ...launch, wgsl: `${launch.wgsl}\nfn changed() {}` };
        },
      };
    };
    // 比の土台の指紋はこの変更を見ない（記録の照合が defaultKernels だけでは古い候補の実測が通る）
    assertEquals(
      profileKernelsId(DEFAULT_GEOMETRY_PROFILE, SWEEP_CASES, candidatesOnly),
      geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
    );
    assertNotEquals(
      candidateKernelsId(ROWS, false, SWEEP_CASES, candidatesOnly),
      sweepCandidateKernelsId(ROWS, false),
    );
  });

  it("失敗した行は数えず（採用の材料にならない）、測った行が 1 本違えば値が変わる", () => {
    const value = sweepCandidateKernelsId(ROWS, false);
    const failed: MeasuredRow = {
      caseId: LINEAR.id,
      geometryParams: quickGemmCandidates()[1].geometry,
      error: "device lost",
    };
    assertEquals(sweepCandidateKernelsId([...ROWS, failed], false), value);
    assertNotEquals(sweepCandidateKernelsId(ROWS.slice(1), false), value);
  });

  it("i8a8 の行は記録の dp4a の変種で組み、f32 骨格の行は dp4a に依らない", () => {
    const i8a8Rows = rowsOf(I8A8, quickI8a8Candidates());
    assertNotEquals(
      sweepCandidateKernelsId(i8a8Rows, true),
      sweepCandidateKernelsId(i8a8Rows, false),
    );
    assertEquals(sweepCandidateKernelsId(F32_ROWS, true), sweepCandidateKernelsId(F32_ROWS, false));
  });

  it("今の runtime の形状表に無いケースの行は、ケースを名指して投げる", () => {
    assertThrows(
      () =>
        sweepCandidateKernelsId(
          [{ caseId: "linear-m3", geometryParams: quickGemmCandidates()[0].geometry }],
          false,
        ),
      KernelsIdError,
      "Case linear-m3 of a measured row is not in the current runtime's shape table",
    );
  });

  it("全ケースを full の全候補で測った行からも導ける（65535 で組める — 書き手が掃引の後に投げない）", () => {
    const rows = SWEEP_CASES.flatMap((sweepCase) =>
      rowsOf(sweepCase, [
        casePlan(sweepCase, 65535, false).defaultCandidate,
        ...candidatesFor(sweepCase, "full"),
      ]),
    );
    for (const dp4a of [false, true]) {
      assertMatch(sweepCandidateKernelsId(rows, dp4a), /^[0-9a-f]{16}$/);
    }
  });
});

describe("埋め込みの表の provenance は今の runtime の値と一致する", () => {
  // On failure: a GEMM kernel (key, params, WGSL tokens, dispatch count — comments and whitespace do
  // not count), the default geometry, the sweep cases or the gemmRows bounds changed. Re-derive every
  // embedded table with the current runtime (ADR 0117 decision 4: the regeneration makes the machine
  // say that the kernels changed, and the per-profile GPU test gpu_geometry_profile_test.ts re-checks
  // output identity on the new kernels). A record without caseSet / defaultKernels (the 2026-09/10
  // Chrome records of nvidia-blackwell and apple-metal-3) is still accepted, so those tables re-derive
  // with unchanged values. A record that carries them (nvidia-ampere's) is rejected once they move:
  // that table needs a new sweep, which can re-pick other fields (the 2026-10-26 conv3d re-sweep also
  // moved i8a8.attentionQk — docs/research/2026-10-26-conv3d-geometry-sweep.md §4).
  for (const profile of BUILTIN_GEOMETRY_PROFILES) {
    it(`${profile.id}: provenance.kernels = 今の runtime の指紋・provenance.caseSet = 今のケース集合の版`, () => {
      const provenance = profile.provenance;
      assert(
        provenance !== undefined,
        `${profile.id}: 埋め込みの表は掃引から生成する（provenance が要る）`,
      );
      assertEquals(
        provenance.kernels,
        geometryProfileKernelsId(profile),
        `${profile.id}: カーネルの指紋が今の runtime と違う — 冒頭コメントのコマンドで再生成する`,
      );
      assertEquals(
        provenance.caseSet,
        sweepCaseSetId(),
        `${profile.id}: ケース集合の版が今の runtime と違う — 冒頭コメントのコマンドで再生成する`,
      );
    });
  }
});

describe("sweepCaseSetId: ケース集合の版", () => {
  it("決定的で（16 進 16 桁）、今のケースと境界から導いた値", () => {
    const value = sweepCaseSetId();
    assertMatch(value, /^[0-9a-f]{16}$/);
    assertEquals(sweepCaseSetId(), value);
    assertEquals(caseSetId(SWEEP_CASES, PROFILE_GEMM_ROWS_BOUNDS), value);
  });

  it("ケースを 1 本足す・dispatch を決める欄（shape・op・attr）を変える・境界を変えると値が変わる", () => {
    const base = caseSetId(SWEEP_CASES, PROFILE_GEMM_ROWS_BOUNDS);
    const [first] = SWEEP_CASES;
    assertNotEquals(
      caseSetId([...SWEEP_CASES, { ...first, id: `${first.id}-copy` }], PROFILE_GEMM_ROWS_BOUNDS),
      base,
    );
    /** `at` 番目のケースだけを差し替えた版。 */
    const replaced = (at: number, sweepCase: SweepCase): string =>
      caseSetId(SWEEP_CASES.with(at, sweepCase), PROFILE_GEMM_ROWS_BOUNDS);
    const indexOf = (op: SweepCase["op"]): number => {
      const index = SWEEP_CASES.findIndex((sweepCase) => sweepCase.op === op);
      assert(index >= 0, op);
      return index;
    };
    const linear = indexOf("linear");
    const attention = indexOf("attention");
    const conv2d = indexOf("conv2d");
    const linearCase = SWEEP_CASES[linear];
    const attentionCase = SWEEP_CASES[attention];
    const conv2dCase = SWEEP_CASES[conv2d];
    assert(linearCase.op === "linear");
    assert(attentionCase.op === "attention");
    assert(conv2dCase.op === "conv2d");
    assertNotEquals(replaced(linear, { ...linearCase, m: linearCase.m + 1 }), base, "m");
    assertNotEquals(replaced(linear, { ...linearCase, op: "i8a8-linear" }), base, "op");
    assertNotEquals(replaced(attention, { ...attentionCase, d: 64 }), base, "d");
    assertNotEquals(replaced(attention, { ...attentionCase, scale: 0.5 }), base, "scale");
    assertNotEquals(replaced(attention, { ...attentionCase, score: "f16" }), base, "score");
    assertNotEquals(replaced(conv2d, { ...conv2dCase, width: 64 }), base, "width");
    const conv3d = indexOf("i8a8-conv3d");
    const conv3dCase = SWEEP_CASES[conv3d];
    assert(conv3dCase.op === "i8a8-conv3d");
    for (const field of ["timeIn", "paddingH", "kernelT", "channelsOut"] as const) {
      const dims = { ...conv3dCase.dims, [field]: conv3dCase.dims[field] + 1 };
      assertNotEquals(replaced(conv3d, { ...conv3dCase, dims }), base, `dims.${field}`);
    }
    assertNotEquals(caseSetId(SWEEP_CASES, [16, 32, 64, 512, Number.POSITIVE_INFINITY]), base);
    // 末尾の Infinity も値として数える（素の JSON.stringify は null に落とし、別の値と区別しない）
    assertNotEquals(caseSetId(SWEEP_CASES, [...PROFILE_GEMM_ROWS_BOUNDS.slice(0, -1), 1024]), base);
  });

  it("説明などの情報の欄（source・censusCount・mirrorOf）だけを変えても値は変わらない", () => {
    const base = caseSetId(SWEEP_CASES, PROFILE_GEMM_ROWS_BOUNDS);
    const described = SWEEP_CASES.map((sweepCase): SweepCase =>
      sweepCase.op === "matmul"
        ? { ...sweepCase, source: `${sweepCase.source}（改）`, mirrorOf: "other" }
        : {
            ...sweepCase,
            source: `${sweepCase.source}（改）`,
            censusCount: sweepCase.censusCount + 1,
          },
    );
    assertEquals(caseSetId(described, PROFILE_GEMM_ROWS_BOUNDS), base);
  });

  it("分類の無い欄を持つケースは投げる（版に入れるかを決めていない欄を黙って落とさない）", () => {
    const [first] = SWEEP_CASES;
    const unknown = { ...first, note: "x" };
    assertThrows(
      () => caseSetId([unknown, ...SWEEP_CASES.slice(1)], PROFILE_GEMM_ROWS_BOUNDS),
      Error,
      "field note has no classification",
    );
  });

  it("ケースのキーの並びには依らない（正規の JSON）", () => {
    const reordered = SWEEP_CASES.map((sweepCase) =>
      Object.fromEntries(Object.entries(sweepCase).reverse()),
    );
    assertEquals(canonicalJson(reordered), canonicalJson(SWEEP_CASES));
  });
});

describe("wgslTokenText: 指紋が数える WGSL", () => {
  it("行コメント・入れ子のブロックコメントを落とし、空白の連なりを 1 つに畳む", () => {
    assertEquals(
      wgslTokenText(
        "// head\nfn f() -> u32 { /* a /* nested */ b */ return 1u; } // tail\n\n  let x = 2u;",
      ),
      "fn f() -> u32 { return 1u; } let x = 2u;",
    );
  });

  it("閉じないブロックコメントは末尾まで、コメントの外のトークンはそのまま", () => {
    assertEquals(wgslTokenText("let a = 1u; /* open"), "let a = 1u;");
    assertEquals(wgslTokenText("let url = 1u; // https://example"), "let url = 1u;");
    assertEquals(wgslTokenText("a/b // c\nd*e"), "a/b d*e");
  });
});

describe("canonicalJson", () => {
  it("キーを並べ・空白を除き・undefined の欄を落とし・Infinity を 1e999 と書く", () => {
    assertEquals(
      canonicalJson({ b: [1, Number.POSITIVE_INFINITY, "x"], a: { d: undefined, c: null } }),
      '{"a":{"c":null},"b":[1,1e999,"x"]}',
    );
  });
});
