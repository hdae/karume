// GEMM 幾何のプロファイル（src/kernels/geometry-profile.ts — ADR 0115）の選択と既定の同一性（GPU 不要）。
//
// 検証の眼目は 3 点:
//
// 1. **選択が決定的**（description まで完全一致 > (vendor, architecture) > vendor だけ > 既定・
//    同順位 2 本は adapter に依らず落ちる・`match` を省いた表は選ばれず注入でだけ使われる）。
//    一覧の並び順や他機の都合で選択が揺れると、環境キーごとの参照 sha の行と「どの幾何で
//    走ったか」の対応が崩れる。
// 2. **既定プロファイルは既存の選択と同じ値**で、導出相がそれを明示で渡してもキーと WGSL が
//    省略時とバイト同一（= 既定の機では 1 バイトも動かない）。
// 3. **壊れた表は Session 構築の門で落ちる**（昇順でない規則・最後が Infinity でない規則・
//    整除の破れた幾何）。`acquireGpu({ geometryProfile })` で注入した表は **device を作る前に**
//    落ちる（navigator.gpu を差し替えて requestDevice に届かないことを見る）。コールバック形
//    （ADR 0117 決定 6）は adapter の情報で 1 度だけ呼ばれ、戻りが同じ門を通り、投げた例外と Promise の
//    戻りでも device を作らない。

import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  acquireGpu,
  GpuFeatureError,
  REQUIRED_LIMIT_KEYS,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import { fakeDevice } from "./helpers/fake-gpu.ts";
import {
  assertGeometryProfile,
  conv2dProfileGeometry,
  DEFAULT_GEOMETRY_PROFILE,
  gemmRowsGeometry,
  type GeometryProfile,
  selectGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import { APPLE_METAL_3 } from "../src/kernels/geometry-profiles/apple-metal-3.ts";
import { NVIDIA_BLACKWELL } from "../src/kernels/geometry-profiles/nvidia-blackwell.ts";
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

/**
 * 既定プロファイルから id と match だけ差し替えた表（幾何の中身は選択の検査に効かない）。
 * `match` が undefined なら**キーごと省く**（`match: undefined` の欄を持つ表ではなく、注入専用の表の形）。
 */
const profileOf = (
  id: string,
  match: GeometryProfile["match"],
  overrides: Partial<GeometryProfile> = {},
): GeometryProfile => {
  const { match: _defaultMatch, ...geometry } = DEFAULT_GEOMETRY_PROFILE;
  return match === undefined
    ? { ...geometry, id, ...overrides }
    : { ...geometry, id, match, ...overrides };
};

const APPLE = profileOf("apple", { vendor: "apple" });
const APPLE_METAL3 = profileOf("apple-metal-3", { vendor: "apple", architecture: "metal-3" });
const APPLE_M2 = profileOf("apple-m2", {
  vendor: "apple",
  architecture: "metal-3",
  description: "Apple M2",
});
/** `match` を省いた注入専用の表（自動選択の対象外）。 */
const OPT_IN = profileOf("opt-in", undefined);

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

  it("埋め込みの一覧は全て門を通り、既定を含まず、match のある各表は自分の match の adapter で選ばれる", () => {
    // 空の adapter（古い Chromium の adapter.info 欠落）では既定に落ちる
    assertStrictEquals(
      selectGeometryProfile({ vendor: "", architecture: "", description: "" }),
      DEFAULT_GEOMETRY_PROFILE,
    );
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      assertGeometryProfile(profile);
      assert(profile.id !== DEFAULT_GEOMETRY_PROFILE.id, `${profile.id}: 既定が一覧に入っている`);
      // match を省いた表は注入専用（自動選択の対象外 — 選ばれないことは選択順の検査が見る）
      const { match } = profile;
      if (match === undefined) continue;
      const vendor = match.vendor;
      assertEquals(typeof vendor, "string", `${profile.id}: vendor 未指定の埋め込み表`);
      // architecture 未指定の表は、architecture を返さない adapter（Deno）で当たる
      const adapter = {
        vendor: vendor ?? "",
        architecture: match.architecture ?? "",
        description: match.description ?? "",
      };
      assertStrictEquals(selectGeometryProfile(adapter), profile, profile.id);
      // description で照合する表は、description の取れない adapter（フラグ無しの Chrome）に当たらない
      if (match.description !== undefined) {
        assert(
          selectGeometryProfile({ ...adapter, description: "" }) !== profile,
          `${profile.id}: description の無い adapter に当たった`,
        );
      }
    }
  });
});

/** 値と、そこから辿れる全てのオブジェクト / 配列が凍結されているか（凍結されていない最初の path）。 */
const unfrozenPath = (value: unknown, path: string): string | undefined => {
  if (typeof value !== "object" || value === null) return undefined;
  if (!Object.isFrozen(value)) return path;
  for (const [key, child] of Object.entries(value)) {
    const found = unfrozenPath(child, `${path}.${key}`);
    if (found !== undefined) return found;
  }
  return undefined;
};

describe("公開の一覧 BUILTIN_GEOMETRY_PROFILES は深く凍結した複製", () => {
  it("一覧と各表の全ての入れ子が凍結され、書き換えは TypeError で落ちる", () => {
    assertEquals(unfrozenPath(BUILTIN_GEOMETRY_PROFILES, "BUILTIN_GEOMETRY_PROFILES"), undefined);
    const [first] = BUILTIN_GEOMETRY_PROFILES;
    assertThrows(() => {
      (first.gemmRows[0].geometry as { regM: number }).regM = 1;
    }, TypeError);
    assertThrows(() => {
      (BUILTIN_GEOMETRY_PROFILES as GeometryProfile[]).push(DEFAULT_GEOMETRY_PROFILE);
    }, TypeError);
  });

  it("値は生成物の定数と同じで、定数そのものは凍結しない（読み込み時に他の module の値を書き換えない）", () => {
    // 欄の落ちた複製（凍結の複製が型の新しい欄を写し忘れた形）もここで落ちる
    assertEquals(BUILTIN_GEOMETRY_PROFILES, [APPLE_METAL_3, NVIDIA_BLACKWELL]);
    for (const profile of [APPLE_METAL_3, NVIDIA_BLACKWELL]) {
      assert(!Object.isFrozen(profile), `${profile.id}: 生成物の定数が凍結された`);
      assert(!Object.isFrozen(profile.gemmRows[0].geometry), `${profile.id}: 幾何が凍結された`);
    }
  });
});

describe("description の照合と match を省いた注入専用の表", () => {
  it("description まで一致する規則が (vendor, architecture) だけの規則より先に当たり、一覧の並び順に依らない", () => {
    const adapter = { vendor: "apple", architecture: "metal-3", description: "Apple M2" };
    for (
      const profiles of [
        [APPLE, APPLE_METAL3, APPLE_M2],
        [APPLE_M2, APPLE_METAL3, APPLE],
        [APPLE_METAL3, APPLE_M2],
      ]
    ) {
      assertStrictEquals(selectGeometryProfile(adapter, profiles), APPLE_M2);
    }
  });

  it("description の規則は description が空・無い・違う adapter に当たらず、次の順位へ落ちる", () => {
    const profiles = [APPLE, APPLE_METAL3, APPLE_M2];
    for (
      const adapter of [
        { vendor: "apple", architecture: "metal-3", description: "" },
        { vendor: "apple", architecture: "metal-3" },
        { vendor: "apple", architecture: "metal-3", description: "Apple M5" },
        // 前方一致・部分一致・大小文字の揃えはしない
        { vendor: "apple", architecture: "metal-3", description: "Apple M2 Pro" },
        { vendor: "apple", architecture: "metal-3", description: "apple m2" },
      ]
    ) {
      assertStrictEquals(
        selectGeometryProfile(adapter, profiles),
        APPLE_METAL3,
        JSON.stringify(adapter),
      );
    }
    // (vendor, architecture) の規則が無ければ vendor だけの規則 → 既定の順に落ちる
    const m5 = { vendor: "apple", architecture: "metal-3", description: "Apple M5" };
    assertStrictEquals(selectGeometryProfile(m5, [APPLE, APPLE_M2]), APPLE);
    assertStrictEquals(selectGeometryProfile(m5, [APPLE_M2]), DEFAULT_GEOMETRY_PROFILE);
    // description が同じでも architecture が違えば当たらない
    assertStrictEquals(
      selectGeometryProfile(
        { vendor: "apple", architecture: "metal-2", description: "Apple M2" },
        [APPLE_M2],
      ),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("match を省いた表はどの adapter にも当たらない（照合する欄が全て一致しても既定へ落ちる）", () => {
    assert(!Object.hasOwn(OPT_IN, "match"), "検査対象の表が match のキーを持っている");
    // 省いた表の中身が他の規則の表と同じでも、選択の対象に入らないことを見る
    const adapters = [
      { vendor: "apple", architecture: "metal-3", description: "Apple M2" },
      { vendor: "", architecture: "", description: "" },
      { vendor: "nvidia", architecture: "blackwell" },
    ];
    for (const adapter of adapters) {
      assertStrictEquals(
        selectGeometryProfile(adapter, [OPT_IN]),
        DEFAULT_GEOMETRY_PROFILE,
        JSON.stringify(adapter),
      );
    }
    // 他の規則と並べても選択は他の規則だけで決まる
    assertStrictEquals(selectGeometryProfile(adapters[0], [OPT_IN, APPLE]), APPLE);
  });

  it("match を省いた表も門と id の重複検査は受ける（一覧に置いたまま壊れた表に気づけない形にしない）", () => {
    const adapter = { vendor: "apple", architecture: "metal-3" };
    assertThrows(
      () => selectGeometryProfile(adapter, [OPT_IN, profileOf("opt-in", { vendor: "apple" })]),
      CodegenError,
      "id が重複",
    );
    assertThrows(
      () =>
        selectGeometryProfile(adapter, [
          profileOf("opt-in-broken", undefined, { gemmRows: [] }),
        ]),
      CodegenError,
      "gemmRows が空",
    );
    // 省いた表は同順位の衝突に数えない（match を持たない表同士は照合で競合しない）
    assertStrictEquals(
      selectGeometryProfile(adapter, [OPT_IN, profileOf("opt-in-2", undefined)]),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("description だけ・vendor と description だけ・空文字の description の match は門で落ちる", () => {
    const adapter = { vendor: "apple", architecture: "metal-3", description: "Apple M2" };
    const cases: readonly (readonly [string, GeometryProfile, string])[] = [
      [
        "description だけ",
        profileOf("desc-only", { description: "Apple M2" }),
        "vendor と architecture の両方と組",
      ],
      [
        "vendor と description だけ",
        profileOf("vendor-desc", { vendor: "apple", description: "Apple M2" }),
        "vendor と architecture の両方と組",
      ],
      [
        "architecture と description だけ",
        profileOf("arch-desc", { architecture: "metal-3", description: "Apple M2" }),
        "vendor と対",
      ],
      [
        "description が空文字",
        profileOf("empty-desc", { vendor: "apple", architecture: "metal-3", description: "" }),
        "空文字にしない",
      ],
    ];
    for (const [name, profile, message] of cases) {
      assertThrows(() => assertGeometryProfile(profile), CodegenError, message, name);
      assertThrows(() => selectGeometryProfile(adapter, [profile]), CodegenError, message, name);
    }
  });

  it("同じ (vendor, architecture, description) の 2 本は、その機でなくても fail loudly・description が違えば並べられる", () => {
    const unrelated = { vendor: "intel", architecture: "xe2", description: "Intel Arc B570" };
    assertThrows(
      () =>
        selectGeometryProfile(unrelated, [
          APPLE_M2,
          profileOf("apple-m2-copy", {
            vendor: "apple",
            architecture: "metal-3",
            description: "Apple M2",
          }),
        ]),
      CodegenError,
      "同じ順位に 2 本当たる",
    );
    const m5 = profileOf("apple-m5", {
      vendor: "apple",
      architecture: "metal-3",
      description: "Apple M5",
    });
    const profiles = [APPLE_METAL3, APPLE_M2, m5];
    assertStrictEquals(
      selectGeometryProfile(
        { vendor: "apple", architecture: "metal-3", description: "Apple M5" },
        profiles,
      ),
      m5,
    );
    assertStrictEquals(selectGeometryProfile(unrelated, profiles), DEFAULT_GEOMETRY_PROFILE);
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

/** requestDevice に届いたことを示す番兵（実 device は作らない）。 */
class DeviceRequested extends Error {
  override readonly name = "DeviceRequested";
}

/**
 * `navigator.gpu` を、requestDevice の呼び出し回数を数えて番兵で落ちる偽物に差し替えて `body` を
 * 走らせる（GPU 不要・実機があっても device を作らない）。差し替えは finally で必ず外す。
 */
const withCountingGpu = async (body: () => Promise<void>): Promise<number> => {
  let deviceRequests = 0;
  const adapter = {
    limits: Object.fromEntries(REQUIRED_LIMIT_KEYS.map((key) => [key, 1 << 20])),
    features: new Set<string>(),
    requestDevice: (): Promise<never> => {
      deviceRequests += 1;
      return Promise.reject(new DeviceRequested("requestDevice に届いた"));
    },
  };
  Object.defineProperty(navigator, "gpu", {
    value: { requestAdapter: () => Promise.resolve(adapter) },
    configurable: true,
  });
  try {
    await body();
  } finally {
    Reflect.deleteProperty(navigator, "gpu");
  }
  return deviceRequests;
};

describe("acquireGpu の幾何プロファイル注入口", () => {
  it("正しい表（match を省いた注入専用の表・description 付きの表を含む）は門を通って requestDevice まで進む（差し替えが経路に乗っていることの対照）", async () => {
    for (const profile of [APPLE_METAL3, OPT_IN, APPLE_M2]) {
      const requests = await withCountingGpu(async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: profile }),
          DeviceRequested,
          undefined,
          profile.id,
        );
      });
      assertEquals(requests, 1, profile.id);
    }
  });

  it("公開の一覧の凍結した表もそのまま注入でき、保持する複製（structuredClone）は凍結を持ち越さない", async () => {
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      const requests = await withCountingGpu(async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: profile }),
          DeviceRequested,
          undefined,
          profile.id,
        );
      });
      assertEquals(requests, 1, profile.id);
      // 注入経路の複製は structuredClone（acquire.ts）— 凍結は複製に写らない
      assertEquals(unfrozenPath(structuredClone(profile), profile.id), profile.id);
    }
  });

  it("壊れた表（id が空・整除の破れた幾何）は device を作る前に fail loudly", async () => {
    const hole: GemmGeometry = { regM: 3, regN: 4, wgX: 8, wgY: 8 };
    const broken: readonly (readonly [string, GeometryProfile, string])[] = [
      ["id が空", profileOf("", { vendor: "test" }), "id が空"],
      [
        "description だけの match",
        profileOf("desc-only", { description: "Apple M2" }),
        "vendor と architecture の両方と組",
      ],
      [
        "gemmRows の幾何",
        profileOf("rows", { vendor: "test" }, {
          gemmRows: [{ maxRows: Number.POSITIVE_INFINITY, geometry: hole }],
        }),
        "gemmRows",
      ],
      [
        "attention.pv の幾何",
        profileOf("pv", { vendor: "test" }, { attention: { qk: M64N64, pv: hole } }),
        "attention.pv",
      ],
      [
        "i8a8.linear の幾何",
        profileOf("i8a8", { vendor: "test" }, {
          i8a8: {
            ...DEFAULT_GEOMETRY_PROFILE.i8a8,
            linear: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 6 },
          },
        }),
        "i8a8.linear",
      ],
    ];
    for (const [name, profile, message] of broken) {
      const requests = await withCountingGpu(async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: profile }),
          GpuFeatureError,
          message,
          name,
        );
      });
      assertEquals(requests, 0, `${name}: 壊れた表で requestDevice に届いた`);
    }
  });
});

/** 偽の adapter が返す情報（apple-metal-3 が当たる 4 欄 — 自動選択の結果が既定と見分けられる）。 */
const FAKE_INFO: GPUAdapterInfo = {
  vendor: "apple",
  architecture: "metal-3",
  device: "",
  description: "Apple M2",
  subgroupMinSize: 0,
  subgroupMaxSize: 0,
  isFallbackAdapter: false,
};

/**
 * `navigator.gpu` を、requestDevice の呼び出し回数を数え、limits を満たす偽の device を返す偽物に
 * 差し替えて `body` を走らせる（GPU 不要 — GpuContext は本物で組まれる）。`info` を省くと `adapter.info`
 * の無い adapter（古い Chromium）になる。差し替えは finally で必ず外す。
 */
const withFakeAdapter = async (
  info: GPUAdapterInfo | undefined,
  body: () => Promise<void>,
): Promise<number> => {
  let deviceRequests = 0;
  const limits = Object.fromEntries(REQUIRED_LIMIT_KEYS.map((key) => [key, 1 << 20]));
  const adapter = {
    ...(info === undefined ? {} : { info }),
    limits,
    features: new Set<string>(),
    requestDevice: (): Promise<GPUDevice> => {
      deviceRequests += 1;
      return Promise.resolve(Object.assign(fakeDevice(), { limits }));
    },
  };
  Object.defineProperty(navigator, "gpu", {
    value: { requestAdapter: () => Promise.resolve(adapter) },
    configurable: true,
  });
  try {
    await body();
  } finally {
    Reflect.deleteProperty(navigator, "gpu");
  }
  return deviceRequests;
};

/** コールバックの戻りの型を外れて来る値（JS の呼び手・async 関数）を、型検査を経ずに渡すための口。 */
const untypedCallback = (
  returned: unknown,
): (adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined =>
  // テスト専用の境界: 型が拒む戻り（Promise など）を、型の外から来る値として再現する
  (() => returned) as unknown as (adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined;

describe("acquireGpu の幾何プロファイル注入口（コールバック形 — ADR 0117 決定 6）", () => {
  it("adapter の情報（GpuContext.adapterInfo と同じ値）で 1 度だけ呼ばれ、表を返すとその複製が注入される", async () => {
    const seen: GPUAdapterInfo[] = [];
    const requests = await withFakeAdapter(FAKE_INFO, async () => {
      const gpu = await acquireGpu({
        geometryProfile: (adapterInfo) => {
          seen.push(adapterInfo);
          return OPT_IN;
        },
      });
      try {
        assertEquals(seen.length, 1);
        assertStrictEquals(seen[0], gpu.adapterInfo);
        assertStrictEquals(seen[0], FAKE_INFO);
        const injected = gpu[RUNTIME_INTERNAL].geometryProfile;
        assertEquals(injected, OPT_IN);
        assert(injected !== OPT_IN, "注入した表が複製されていない");
      } finally {
        gpu.destroy();
      }
    });
    assertEquals(requests, 1);
  });

  it("undefined を返すと注入しない（Session の構築が adapter から埋め込みの表を選ぶ — 指定無しと同じ）", async () => {
    await withFakeAdapter(FAKE_INFO, async () => {
      let calls = 0;
      const gpu = await acquireGpu({
        geometryProfile: () => {
          calls += 1;
          return undefined;
        },
      });
      try {
        assertEquals(calls, 1);
        assertEquals(gpu[RUNTIME_INTERNAL].geometryProfile, undefined);
        // 自動選択の対照: この adapter には埋め込みの表が当たる（既定ではない）
        assertEquals(selectGeometryProfile(gpu.adapterInfo).id, APPLE_METAL_3.id);
      } finally {
        gpu.destroy();
      }
    });
  });

  it("adapter.info の無い adapter では空値に正規化した情報で呼ばれる", async () => {
    const seen: GPUAdapterInfo[] = [];
    await withFakeAdapter(undefined, async () => {
      const gpu = await acquireGpu({
        geometryProfile: (adapterInfo) => {
          seen.push(adapterInfo);
          return undefined;
        },
      });
      try {
        assertStrictEquals(seen[0], gpu.adapterInfo);
        assertEquals(
          [seen[0].vendor, seen[0].architecture, seen[0].device, seen[0].description],
          ["", "", "", ""],
        );
      } finally {
        gpu.destroy();
      }
    });
  });

  it("壊れた表を返すと device を作る前に GpuFeatureError（直接渡した表と同じ門）", async () => {
    const broken = profileOf("", { vendor: "test" });
    const requests = await withFakeAdapter(FAKE_INFO, async () => {
      await assertRejects(
        () => acquireGpu({ geometryProfile: () => broken }),
        GpuFeatureError,
        "geometryProfile（コールバックの戻り）: ",
      );
    });
    assertEquals(requests, 0, "壊れた表で requestDevice に届いた");
  });

  it("コールバックが投げた例外はそのまま伝わり、device を作らない", async () => {
    class AppError extends Error {}
    const thrown = new AppError("保存した表が読めない");
    const requests = await withFakeAdapter(FAKE_INFO, async () => {
      const caught = await assertRejects(() =>
        acquireGpu({
          geometryProfile: () => {
            throw thrown;
          },
        })
      );
      assertStrictEquals(caught, thrown);
    });
    assertEquals(requests, 0, "コールバックが投げたのに requestDevice に届いた");
  });

  it("Promise（thenable）を返すと await せずに GpuFeatureError で拒み、device を作らない", async () => {
    let awaited = false;
    const thenable = {
      then: (resolve: (value: GeometryProfile) => void): void => {
        awaited = true;
        resolve(OPT_IN);
      },
    };
    for (const returned of [Promise.resolve(OPT_IN), thenable]) {
      const requests = await withFakeAdapter(FAKE_INFO, async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: untypedCallback(returned) }),
          GpuFeatureError,
          "コールバックが Promise を返した",
        );
      });
      assertEquals(requests, 0, "Promise を返したのに requestDevice に届いた");
    }
    assertEquals(awaited, false, "thenable を await した");
  });

  it("async 関数が reject しても GpuFeatureError で拒み、その reject は unhandled rejection にならない", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (event: PromiseRejectionEvent): void => {
      unhandled.push(event.reason);
      // 捕まえた reject でテストランナーを落とさない（数えて下で assert する）
      event.preventDefault();
    };
    // async 関数そのものをコールバックにする（呼ばれた時点で reject 済みの Promise が生まれる — JS の呼び手の形）。
    // テスト専用の境界: 型が拒む async 関数を、型の外から来る値として渡す
    // deno-lint-ignore require-await -- await の無い async 関数の reject が検査の対象そのもの
    const asyncCallback = (async () => {
      throw new Error("x");
    }) as unknown as (adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined;
    globalThis.addEventListener("unhandledrejection", onUnhandled);
    try {
      const requests = await withFakeAdapter(FAKE_INFO, async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: asyncCallback }),
          GpuFeatureError,
          "コールバックが Promise を返した",
        );
      });
      assertEquals(requests, 0, "Promise を返したのに requestDevice に届いた");
      // unhandled rejection の判定はマイクロタスクを捌き切った後のタスク境界で走る — 1 タスク待ってから見る
      await new Promise((resolve) => setTimeout(resolve, 0));
      assertEquals(unhandled, [], "async コールバックの reject が unhandled rejection になった");
    } finally {
      globalThis.removeEventListener("unhandledrejection", onUnhandled);
    }
  });

  it("adapter.info の無い adapter で渡る空値の情報は凍結されている（共有値をコールバックが書き換えられない）", async () => {
    const seen: GPUAdapterInfo[] = [];
    await withFakeAdapter(undefined, async () => {
      const gpu = await acquireGpu({
        geometryProfile: (adapterInfo) => {
          seen.push(adapterInfo);
          return undefined;
        },
      });
      gpu.destroy();
    });
    assert(Object.isFrozen(seen[0]), "空値の adapter 情報が凍結されていない");
  });

  it("直接渡した表は従来どおり adapter を取る前に門を通る（コールバック形の追加で順序が動いていない）", async () => {
    let adapterRequests = 0;
    Object.defineProperty(navigator, "gpu", {
      value: {
        requestAdapter: () => {
          adapterRequests += 1;
          return Promise.resolve(null);
        },
      },
      configurable: true,
    });
    try {
      await assertRejects(
        () => acquireGpu({ geometryProfile: profileOf("", { vendor: "test" }) }),
        GpuFeatureError,
        "geometryProfile: ",
      );
    } finally {
      Reflect.deleteProperty(navigator, "gpu");
    }
    assertEquals(adapterRequests, 0);
  });
});
