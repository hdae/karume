// 保存した表の照合 geometryProfileMismatch（src/tune/mismatch.ts — ADR 0117 決定 5）の門。公開の関数は
// 公開面（../tune.ts）から、公開しない指紋（geometryProfileKernelsId）は src から引く。GPU を使わない
// （指紋は codegen だけで導く）。
//
// 固定するのは ① 全て一致なら undefined ② provenance 無し・adapter の 4 欄（空文字も値）・指紋・ケース集合の
// 版のそれぞれの不一致を、その欄を名指す文言で返す ③ 照合の順（最初の不一致だけを返す）④ 投げない
// （表の幾何から掃引の shape のカーネルを組めない表も文言で返す）。

import { assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { geometryProfileMismatch, sweepCaseSetId } from "../tune.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { geometryProfileKernelsId } from "../src/tune/fingerprint.ts";

type Provenance = NonNullable<GeometryProfile["provenance"]>;
type Adapter = Provenance["adapter"];

const ADAPTER: Adapter = {
  vendor: "apple",
  architecture: "metal-3",
  device: "0x0000",
  description: "Apple M2",
};

/** 今の runtime の値で照合の材料を揃えた表（`overrides` で provenance の欄を差し替える）。 */
const stored = (
  table: GeometryProfile = DEFAULT_GEOMETRY_PROFILE,
  overrides: Partial<Provenance> = {},
): GeometryProfile & { readonly provenance: Provenance } => ({
  ...table,
  id: "stored",
  provenance: {
    sweep: "sweep.json",
    sha256: "sha",
    date: "2026-10-02T00:00:00.000Z",
    candidateSet: "quick+",
    adapter: ADAPTER,
    kernels: geometryProfileKernelsId(table),
    caseSet: sweepCaseSetId(),
    ...overrides,
  },
});

/** 照合が不一致を返すこと（undefined なら落とす）。 */
const mismatchOf = (profile: GeometryProfile, adapter: Adapter): string => {
  const message = geometryProfileMismatch(profile, adapter);
  if (message === undefined) throw new Error("一致と判定された");
  return message;
};

describe("geometryProfileMismatch", () => {
  it("adapter の 4 欄・カーネルの指紋・ケース集合の版が全て一致すれば undefined", () => {
    assertEquals(geometryProfileMismatch(stored(), ADAPTER), undefined);
    // GPUAdapterInfo の他の欄（subgroup の幅など）は見ない — GpuContext.adapterInfo をそのまま渡せる
    const info: GPUAdapterInfo = {
      ...ADAPTER,
      subgroupMinSize: 4,
      subgroupMaxSize: 64,
      isFallbackAdapter: false,
    };
    assertEquals(geometryProfileMismatch(stored(), info), undefined);
  });

  it("provenance の無い表（手書きの表）は照合の材料が無いと返す", () => {
    const message = mismatchOf(DEFAULT_GEOMETRY_PROFILE, ADAPTER);
    assertStringIncludes(message, "'default' に provenance が無い");
    assertStringIncludes(message, "照合の材料が無い");
  });

  for (const field of ["vendor", "architecture", "device", "description"] as const) {
    it(`adapter の ${field} が違えば、その欄と両方の値を名指す（空文字も値として比べる）`, () => {
      const blank = mismatchOf(stored(), { ...ADAPTER, [field]: "" });
      assertStringIncludes(blank, `adapter の ${field} が違う`);
      assertStringIncludes(blank, `表 ${JSON.stringify(ADAPTER[field])} / この adapter ""`);
      // 表の側が空（フラグ無しの Chrome で作った表）で、adapter が値を返す（フラグを入れた）場合も不一致
      const filled = mismatchOf(
        stored(DEFAULT_GEOMETRY_PROFILE, { adapter: { ...ADAPTER, [field]: "" } }),
        ADAPTER,
      );
      assertStringIncludes(filled, `adapter の ${field} が違う`);
      // 大小文字も揃えない（文字列の完全一致）
      assertStringIncludes(
        mismatchOf(stored(), { ...ADAPTER, [field]: ADAPTER[field].toUpperCase() + "!" }),
        `adapter の ${field} が違う`,
      );
    });
  }

  it("空文字どうしの欄は一致する（フラグ無しの Chrome で作り、同じ Chrome で照合する）", () => {
    const blank = { ...ADAPTER, device: "", description: "" };
    assertEquals(
      geometryProfileMismatch(stored(DEFAULT_GEOMETRY_PROFILE, { adapter: blank }), blank),
      undefined,
    );
  });

  it("カーネルの指紋が今の runtime の値と違えば返す（表の幾何が保存後に書き換わった場合も同じ）", () => {
    const message = mismatchOf(
      stored(DEFAULT_GEOMETRY_PROFILE, { kernels: "0123456789abcdef" }),
      ADAPTER,
    );
    assertStringIncludes(message, "カーネルの指紋が違う");
    assertStringIncludes(message, "表 0123456789abcdef");
    assertStringIncludes(
      message,
      `今の runtime ${geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE)}`,
    );
    const edited: GeometryProfile = {
      ...stored(),
      attention: {
        ...DEFAULT_GEOMETRY_PROFILE.attention,
        qk: { regM: 4, regN: 4, wgX: 8, wgY: 16 },
      },
    };
    assertStringIncludes(mismatchOf(edited, ADAPTER), "カーネルの指紋が違う");
  });

  it("ケース集合の版が今の runtime の値と違えば返す", () => {
    const message = mismatchOf(
      stored(DEFAULT_GEOMETRY_PROFILE, { caseSet: "fedcba9876543210" }),
      ADAPTER,
    );
    assertStringIncludes(message, "ケース集合の版が違う");
    assertStringIncludes(message, `表 fedcba9876543210 / 今の runtime ${sweepCaseSetId()}`);
  });

  it("最初の不一致だけを返す（adapter の 4 欄 → 指紋 → ケース集合の版の順）", () => {
    const all = stored(DEFAULT_GEOMETRY_PROFILE, {
      kernels: "0123456789abcdef",
      caseSet: "fedcba9876543210",
    });
    assertStringIncludes(mismatchOf(all, { ...ADAPTER, device: "0x1234" }), "adapter の device");
    assertStringIncludes(mismatchOf(all, ADAPTER), "カーネルの指紋が違う");
    assertStringIncludes(
      mismatchOf(
        { ...all, provenance: { ...all.provenance, kernels: geometryProfileKernelsId(all) } },
        ADAPTER,
      ),
      "ケース集合の版が違う",
    );
  });

  it("候補集合と match は照合しない", () => {
    assertEquals(
      geometryProfileMismatch(
        {
          ...stored(DEFAULT_GEOMETRY_PROFILE, { candidateSet: "quick" }),
          match: { vendor: "intel" },
        },
        ADAPTER,
      ),
      undefined,
    );
  });

  it("投げない: 表の幾何から掃引の shape のカーネルを組めない表（dispatch 数が 65535 を超える）は理由を返す", () => {
    // 門（assertGeometryProfile）は通るが、tileN 4 で conv2d 512² の N = 262144 が 65536 workgroup になる
    const narrow: GeometryProfile = {
      ...DEFAULT_GEOMETRY_PROFILE,
      conv2d: { ...DEFAULT_GEOMETRY_PROFILE.conv2d, rows32: { regM: 1, regN: 4, wgX: 1, wgY: 16 } },
    };
    const message = mismatchOf({ ...narrow, provenance: stored().provenance }, ADAPTER);
    assertStringIncludes(message, "カーネルを組めない");
    assertStringIncludes(message, "conv2d-c96-512x512");
  });
});
