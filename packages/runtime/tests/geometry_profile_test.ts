// GEMM 幾何のプロファイル（src/kernels/geometry-profile.ts — ADR 0115）の選択と既定の同一性（GPU 不要）。
//
// 検証の眼目は 3 点:
//
// 1. **選択が決定的**（完全一致 > vendor だけ > 既定・同順位 2 本は adapter に依らず落ちる）。
//    一覧の並び順や他機の都合で選択が揺れると、環境キーごとの参照 sha の行と「どの幾何で
//    走ったか」の対応が崩れる。
// 2. **既定プロファイルは既存の選択と同じ値**で、導出相がそれを明示で渡してもキーと WGSL が
//    省略時とバイト同一（= 既定の機では 1 バイトも動かない）。
// 3. **壊れた表は Session 構築の門で落ちる**（昇順でない規則・最後が Infinity でない規則・
//    整除の破れた幾何）。

import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  assertGeometryProfile,
  conv2dProfileGeometry,
  DEFAULT_GEOMETRY_PROFILE,
  gemmRowsGeometry,
  type GeometryProfile,
  selectGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import {
  defaultGemmGeometry,
  type GemmGeometry,
  gemmGeometryForRows,
} from "../src/kernels/gemm-geometry.ts";
import { defaultI8a8Geometry } from "../src/kernels/i8a8-geometry.ts";
import { gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { matmulKey, matmulWgsl } from "../src/kernels/matmul.ts";
import { bmmKey, bmmWgsl } from "../src/kernels/bmm.ts";
import { linearKey, linearWgsl } from "../src/kernels/linear.ts";
import {
  attentionPvKey,
  attentionPvWgsl,
  attentionQkKey,
  attentionQkWgsl,
} from "../src/kernels/attention.ts";
import { conv2dIgemmKey, conv2dIgemmWgsl } from "../src/kernels/conv2d.ts";
import { linearI8a8Key, linearI8a8Wgsl } from "../src/kernels/linear-i8a8.ts";
import {
  attentionPvI8a8Key,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Wgsl,
} from "../src/kernels/attention-i8a8.ts";

/** 既定と見分けのつく幾何（M2 の掃引で勝った `reg64x32r4x4w8` / `reg64x64r8x4w16`）。 */
const M64N32: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };
const M64N64: GemmGeometry = { regM: 8, regN: 4, wgX: 16, wgY: 8 };

/** 既定プロファイルから id と match だけ差し替えた表（幾何の中身は選択の検査に効かない）。 */
const profileOf = (
  id: string,
  match: GeometryProfile["match"],
  overrides: Partial<GeometryProfile> = {},
): GeometryProfile => ({ ...DEFAULT_GEOMETRY_PROFILE, id, match, ...overrides });

const APPLE = profileOf("apple", { vendor: "apple" });
const APPLE_METAL3 = profileOf("apple-metal-3", { vendor: "apple", architecture: "metal-3" });

/** 表の代表点（バケットごとに「境界ちょうど」と「境界を 1 越えた側」を持つ）。 */
const BUCKET_ROWS = [1, 64, 65, 512, 513, 4096] as const;

describe("selectGeometryProfile の選択順", () => {
  it("(vendor, architecture) の完全一致が vendor だけの規則より先に当たり、一覧の並び順に依らない", () => {
    const adapter = { vendor: "apple", architecture: "metal-3" };
    assertStrictEquals(selectGeometryProfile(adapter, [APPLE, APPLE_METAL3]), APPLE_METAL3);
    assertStrictEquals(selectGeometryProfile(adapter, [APPLE_METAL3, APPLE]), APPLE_METAL3);
  });

  it("vendor だけの規則は architecture の違う adapter（空文字 = Deno を含む）に当たる", () => {
    const profiles = [APPLE, APPLE_METAL3];
    assertStrictEquals(
      selectGeometryProfile({ vendor: "apple", architecture: "metal-2" }, profiles),
      APPLE,
    );
    assertStrictEquals(
      selectGeometryProfile({ vendor: "apple", architecture: "" }, profiles),
      APPLE,
    );
  });

  it("architecture を持つ規則は vendor が一致しても architecture が違えば当たらない", () => {
    assertStrictEquals(
      selectGeometryProfile({ vendor: "apple", architecture: "metal-2" }, [APPLE_METAL3]),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("どれにも当たらない adapter は既定プロファイルへ落ちる（照合は完全一致で、大小文字も揃えない）", () => {
    const profiles = [APPLE, APPLE_METAL3];
    for (
      const adapter of [
        { vendor: "32902", architecture: "" },
        { vendor: "Apple", architecture: "metal-3" },
        { vendor: "", architecture: "" },
      ]
    ) {
      assertStrictEquals(
        selectGeometryProfile(adapter, profiles),
        DEFAULT_GEOMETRY_PROFILE,
        JSON.stringify(adapter),
      );
    }
    assertStrictEquals(
      selectGeometryProfile({ vendor: "apple", architecture: "metal-3" }, []),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("同じ順位に 2 本当たる一覧は、その match の機でなくても fail loudly", () => {
    // 他の機（ここでは intel）で選択しても落ちる — 衝突する機の CI でしか見えない形にしない
    const unrelated = { vendor: "intel", architecture: "xe2" };
    assertThrows(
      () =>
        selectGeometryProfile(unrelated, [
          APPLE_METAL3,
          profileOf("apple-metal-3-copy", { vendor: "apple", architecture: "metal-3" }),
        ]),
      CodegenError,
      "同じ順位に 2 本当たる",
    );
    assertThrows(
      () => selectGeometryProfile(unrelated, [APPLE, profileOf("apple-2", { vendor: "apple" })]),
      CodegenError,
      "同じ順位に 2 本当たる",
    );
  });

  it("全 adapter に当たる表・vendor の無い architecture・空文字の match・重複 id は fail loudly", () => {
    const adapter = { vendor: "apple", architecture: "metal-3" };
    const cases: readonly (readonly [string, GeometryProfile, string])[] = [
      ["match 空", profileOf("everything", {}), "既定プロファイルだけ"],
      ["architecture だけ", profileOf("arch-only", { architecture: "metal-3" }), "vendor と対"],
      ["vendor が空文字", profileOf("empty-vendor", { vendor: "" }), "空文字にしない"],
      [
        "architecture が空文字",
        profileOf("empty-arch", { vendor: "apple", architecture: "" }),
        "空文字にしない",
      ],
      ["既定と同じ id", profileOf("default", { vendor: "apple" }), "id が重複"],
    ];
    for (const [name, profile, message] of cases) {
      assertThrows(() => selectGeometryProfile(adapter, [profile]), CodegenError, message, name);
    }
    assertThrows(
      () => selectGeometryProfile(adapter, [APPLE, profileOf("apple", { vendor: "intel" })]),
      CodegenError,
      "id が重複",
    );
  });

  it("埋め込みの一覧は全て門を通り、既定を含まず、各表は自分の match の adapter で選ばれる", () => {
    // 空の adapter（古い Chromium の adapter.info 欠落）では既定に落ちる
    assertStrictEquals(
      selectGeometryProfile({ vendor: "", architecture: "" }),
      DEFAULT_GEOMETRY_PROFILE,
    );
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      const vendor = profile.match.vendor;
      assertEquals(typeof vendor, "string", `${profile.id}: vendor 未指定の埋め込み表`);
      // architecture 未指定の表は、architecture を返さない adapter（Deno）で当たる
      const adapter = { vendor: vendor ?? "", architecture: profile.match.architecture ?? "" };
      assertStrictEquals(selectGeometryProfile(adapter), profile, profile.id);
    }
  });
});

describe("DEFAULT_GEOMETRY_PROFILE は既存の選択と同じ値", () => {
  it("match は空・id は default で、自身の門を通る", () => {
    assertEquals(DEFAULT_GEOMETRY_PROFILE.id, "default");
    assertEquals(DEFAULT_GEOMETRY_PROFILE.match, {});
    assertEquals(DEFAULT_GEOMETRY_PROFILE.provenance, undefined);
    assertGeometryProfile(DEFAULT_GEOMETRY_PROFILE);
  });

  it("行数バケットは rows 1 / 64 / 65 / 512 / 513 / 4096 で gemmGeometryForRows と一致する", () => {
    for (const rows of [0, ...BUCKET_ROWS, 0xffff_ffff]) {
      assertEquals(
        gemmRowsGeometry(DEFAULT_GEOMETRY_PROFILE, rows),
        gemmGeometryForRows(rows),
        `M=${rows}`,
      );
    }
  });

  it("attention / conv2d / i8a8 の欄は既存の既定（defaultGemmGeometry / gemmMTileGeometry / defaultI8a8Geometry）と一致する", () => {
    assertEquals(DEFAULT_GEOMETRY_PROFILE.attention, {
      qk: defaultGemmGeometry(),
      pv: defaultGemmGeometry(),
    });
    assertEquals(DEFAULT_GEOMETRY_PROFILE.conv2d, {
      rows64: gemmMTileGeometry(64),
      rows32: gemmMTileGeometry(32),
    });
    assertEquals(DEFAULT_GEOMETRY_PROFILE.i8a8, {
      linear: defaultI8a8Geometry("linear"),
      attentionQk: defaultI8a8Geometry("attention_qk"),
      attentionPv: defaultI8a8Geometry("attention_pv"),
    });
  });
});

// 導出相（src/runtime/recipe-builders/）は幾何をプロファイルから引いて**明示で**渡すようになった。
// 既定の機で 1 バイトも動かないことの単体版は「既定プロファイルの値を明示で渡したキー・WGSL が、
// 導入前の呼び方（省略）とバイト同一」。codegen スナップショットは省略形を固定しているので、
// ここが両者を繋ぐ。
describe("既定プロファイルの幾何を明示で渡しても、キーと WGSL は省略時とバイト同一", () => {
  const profile = DEFAULT_GEOMETRY_PROFILE;

  it("matmul / bmm / linear（行数バケット）", () => {
    for (const rows of BUCKET_ROWS) {
      const geometry = gemmRowsGeometry(profile, rows);
      for (const v4 of [true, false]) {
        assertEquals(matmulKey(v4, rows, geometry), matmulKey(v4, rows), `matmul M=${rows}`);
        assertEquals(matmulWgsl(v4, rows, geometry), matmulWgsl(v4, rows), `matmul M=${rows}`);
        assertEquals(bmmKey(v4, rows, undefined, geometry), bmmKey(v4, rows), `bmm M=${rows}`);
        assertEquals(bmmWgsl(v4, rows, undefined, geometry), bmmWgsl(v4, rows), `bmm M=${rows}`);
      }
      for (
        const [weight, v4, compute] of [
          ["f32", true, "f32"],
          ["f16", false, "f32"],
          ["f16", true, "f16"],
          ["i8", true, "f32"],
        ] as const
      ) {
        const where = `linear ${weight}/${v4}/${compute} M=${rows}`;
        assertEquals(
          linearKey(weight, v4, compute, rows, undefined, geometry),
          linearKey(weight, v4, compute, rows),
          where,
        );
        assertEquals(
          linearWgsl(weight, v4, compute, rows, undefined, geometry),
          linearWgsl(weight, v4, compute, rows),
          where,
        );
      }
      assertEquals(
        linearKey("i4", true, "f32", rows, 32, geometry),
        linearKey("i4", true, "f32", rows, 32),
      );
      assertEquals(
        linearWgsl("i4", true, "f32", rows, 32, geometry),
        linearWgsl("i4", true, "f32", rows, 32),
      );
    }
  });

  it("融合 attention f32 の ①QK / ③PV（変種の軸を跨いで）", () => {
    const { qk, pv } = profile.attention;
    for (
      const [v4, compute, score, mask, gqa, rowWindow] of [
        [true, "f32", "f32", false, false, false],
        [false, "f32", "f32", false, false, false],
        [true, "f16", "f32", false, false, false],
        [true, "f32", "f16", true, false, false],
        [true, "f32", "f32", true, true, true],
      ] as const
    ) {
      const where = `v4=${v4} ${compute} s=${score} mask=${mask} gqa=${gqa} rw=${rowWindow}`;
      assertEquals(
        attentionQkKey(v4, compute, score, mask, gqa, rowWindow, qk),
        attentionQkKey(v4, compute, score, mask, gqa, rowWindow),
        where,
      );
      assertEquals(
        attentionQkWgsl(v4, compute, score, mask, gqa, rowWindow, qk),
        attentionQkWgsl(v4, compute, score, mask, gqa, rowWindow),
        where,
      );
      assertEquals(
        attentionPvKey(v4, compute, score, gqa, rowWindow, pv),
        attentionPvKey(v4, compute, score, gqa, rowWindow),
        where,
      );
      assertEquals(
        attentionPvWgsl(v4, compute, score, gqa, rowWindow, pv),
        attentionPvWgsl(v4, compute, score, gqa, rowWindow),
        where,
      );
    }
  });

  it("conv2d の implicit GEMM（m タイルの 64 行 / 32 行クラス）", () => {
    for (const mTile of [64, 32]) {
      const geometry = conv2dProfileGeometry(profile, mTile);
      for (const [weight, v4] of [["f32", true], ["f16", false], ["i8", true]] as const) {
        assertEquals(
          conv2dIgemmKey(weight, v4, mTile, geometry),
          conv2dIgemmKey(weight, v4, mTile),
          `${weight} mTile=${mTile}`,
        );
        assertEquals(
          conv2dIgemmWgsl(weight, v4, mTile, geometry),
          conv2dIgemmWgsl(weight, v4, mTile),
          `${weight} mTile=${mTile}`,
        );
      }
    }
  });

  it("i8a8 の linear / 融合 attention ①QK / ③PV", () => {
    const { linear, attentionQk, attentionPv } = profile.i8a8;
    for (const [v4, dp4a] of [[true, true], [false, false]] as const) {
      assertEquals(linearI8a8Key(v4, dp4a, linear), linearI8a8Key(v4, dp4a));
      assertEquals(linearI8a8Wgsl(v4, dp4a, linear), linearI8a8Wgsl(v4, dp4a));
      assertEquals(
        linearI8a8Key(v4, dp4a, linear, "i4", 32),
        linearI8a8Key(v4, dp4a, undefined, "i4", 32),
      );
      // S の f16 格納（s16）は v4 経路専用（生成の門が落とす）
      for (const score of v4 ? ["f32", "f16"] as const : ["f32"] as const) {
        assertEquals(
          attentionQkI8a8Key(v4, dp4a, score, attentionQk),
          attentionQkI8a8Key(v4, dp4a, score),
        );
        assertEquals(
          attentionQkI8a8Wgsl(v4, dp4a, score, attentionQk),
          attentionQkI8a8Wgsl(v4, dp4a, score),
        );
        assertEquals(
          attentionPvI8a8Key(v4, dp4a, score, attentionPv),
          attentionPvI8a8Key(v4, dp4a, score),
        );
        assertEquals(
          attentionPvI8a8Wgsl(v4, dp4a, score, attentionPv),
          attentionPvI8a8Wgsl(v4, dp4a, score),
        );
      }
    }
  });
});

describe("gemmRowsGeometry", () => {
  const rules = (...entries: (readonly [number, GemmGeometry])[]): GeometryProfile =>
    profileOf("rules", { vendor: "test" }, {
      gemmRows: entries.map(([maxRows, geometry]) => ({ maxRows, geometry })),
    });
  const small = gemmGeometryForRows(1);

  it("maxRows ちょうどはその規則・1 越えると次の規則（最後の Infinity は u32 の上端まで当たる）", () => {
    const profile = rules([10, small], [100, M64N32], [Number.POSITIVE_INFINITY, M64N64]);
    const expected: readonly (readonly [number, GemmGeometry])[] = [
      [0, small],
      [10, small],
      [11, M64N32],
      [100, M64N32],
      [101, M64N64],
      [0xffff_ffff, M64N64],
    ];
    for (const [rows, geometry] of expected) {
      assertEquals(gemmRowsGeometry(profile, rows), geometry, `M=${rows}`);
    }
  });

  it("規則が狭義昇順でない・最後が Infinity でない・空の表は、選択でも引く側でも fail loudly", () => {
    const adapter = { vendor: "test", architecture: "" };
    const broken: readonly (readonly [string, GeometryProfile, string])[] = [
      [
        "降順",
        rules([100, small], [10, M64N32], [Number.POSITIVE_INFINITY, M64N64]),
        "狭義昇順",
      ],
      ["同値", rules([10, small], [10, M64N32], [Number.POSITIVE_INFINITY, M64N64]), "狭義昇順"],
      ["最後が有限", rules([10, small], [100, M64N32]), "Infinity"],
      ["空", rules(), "空"],
    ];
    for (const [name, profile, message] of broken) {
      assertThrows(() => gemmRowsGeometry(profile, 1), CodegenError, message, name);
      assertThrows(() => selectGeometryProfile(adapter, [profile]), CodegenError, message, name);
    }
  });

  it("行数は u32 の非負整数以外を fail loudly にする（gemmGeometryForRows と同じ門）", () => {
    for (const rows of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      assertThrows(() => gemmRowsGeometry(DEFAULT_GEOMETRY_PROFILE, rows), CodegenError);
    }
  });
});

describe("conv2dProfileGeometry と幾何の門", () => {
  it("m タイルのクラス 64 / 32 は rows64 / rows32 を引き、それ以外は fail loudly", () => {
    const profile = profileOf("conv", { vendor: "test" }, {
      conv2d: { rows64: M64N64, rows32: M64N32 },
    });
    assertStrictEquals(conv2dProfileGeometry(profile, 64), M64N64);
    assertStrictEquals(conv2dProfileGeometry(profile, 32), M64N32);
    assertThrows(() => conv2dProfileGeometry(profile, 16), CodegenError, "クラスが無い");
  });

  it("整除の破れた幾何を持つ表は、その op に当たる前（選択の時点）で落ちる", () => {
    const adapter = { vendor: "test", architecture: "" };
    const hole: GemmGeometry = { regM: 3, regN: 4, wgX: 8, wgY: 8 };
    const broken: readonly (readonly [string, GeometryProfile])[] = [
      [
        "attention.qk",
        profileOf("qk", { vendor: "test" }, { attention: { qk: hole, pv: M64N64 } }),
      ],
      [
        "conv2d.rows32",
        profileOf("conv", { vendor: "test" }, { conv2d: { rows64: M64N64, rows32: hole } }),
      ],
      [
        "gemmRows",
        profileOf("rows", { vendor: "test" }, {
          gemmRows: [{ maxRows: Number.POSITIVE_INFINITY, geometry: hole }],
        }),
      ],
      [
        "i8a8.attentionPv",
        profileOf("i8a8", { vendor: "test" }, {
          i8a8: {
            ...DEFAULT_GEOMETRY_PROFILE.i8a8,
            attentionPv: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 6 },
          },
        }),
      ],
    ];
    for (const [name, profile] of broken) {
      assertThrows(() => selectGeometryProfile(adapter, [profile]), CodegenError, name, name);
    }
  });
});
