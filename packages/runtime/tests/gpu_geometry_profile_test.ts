// GEMM 幾何のプロファイル（ADR 0115 — src/kernels/geometry-profile.ts）が Session の導出相まで
// 結線されていることの実 GPU 検証。純関数の振る舞いは geometry_profile_test.ts。
//
// 見るのは 2 つ:
//
// 1. 選ばれたプロファイルの名前が診断（`SessionDiagnostics.geometryProfile`）に出る。
// 2. 埋め込みの各プロファイル（src/kernels/geometry-profiles/）で組んだ Session の出力が、既定
//    プロファイルの Session と**同じ入力で Uint32 一致**する。幾何が変えてよいのは担当割りだけ
//    （ADR 0022 の数値契約）で、導出相がキー・WGSL・dispatch のどれか 1 つにだけ別の幾何を
//    渡すと、出力タイルが欠けてここで値の差として出る。
//    MUST: 値の一致だけでは結線の証拠にならない（プロファイルを無視しても一致する）ので、
//    そのプロファイルの幾何判別子が**実際に走ったパイプラインキー**に載ったことも見る。
//
// プロファイルは adapter の (vendor, architecture, description) で選ばれるので、2 は **device は
// 実物のまま、adapterInfo だけをそのプロファイルの match に合わせた GpuContext** で Session を組む
// （ここで走る GPU はプロファイルを作った機ではないが、幾何でビットが動かないことは機に依らない
// 命題 — ADR 0022 追記）。`match` を省いた注入専用の表は adapter では選ばれないので、同じ偽装無しの
// GpuContext に**注入の席**から渡して組む（全表を回す — 自動選択されない表も既定とビット同一で
// なければ注入した利用者の出力が動く）。既定側も adapterInfo を空にした GpuContext で組み、実機の
// adapter に将来プロファイルが当たっても比較の基準が既定のまま動かないようにする。
// 埋め込みが 1 本も無い間は 2 を明示 SKIP する（空の一覧で緑にしない）。
//
// 3. `acquireGpu({ geometryProfile })` で注入した表は、adapter の (vendor, architecture, description) を見ずに
//    使われる（`match` も見ない）。2 と同じ 3 点（診断の名前・実走キーの幾何判別子・既定との Uint32
//    一致）に加え、i8a8 attention の dp4a カナリアがその表の幾何で撃つことを、カナリアが
//    コンパイルした WGSL で見る。
//    コールバック形（ADR 0117 決定 6）で返した表も同じ 3 点で使われ、`undefined` は自動選択になる。

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { acquireGpu, GpuContext } from "../src/gpu/device.ts";
import {
  DEFAULT_GEOMETRY_PROFILE,
  gemmRowsGeometry,
  type GemmRowsRule,
  type GeometryProfile,
  selectGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import { APPLE_METAL_3 } from "../src/kernels/geometry-profiles/apple-metal-3.ts";
import { NVIDIA_BLACKWELL } from "../src/kernels/geometry-profiles/nvidia-blackwell.ts";
import { attentionPvI8a8Wgsl, attentionQkI8a8Wgsl } from "../src/kernels/attention-i8a8.ts";
import { dp4aAvailable } from "../src/kernels/linear-i8a8.ts";
import {
  type GemmGeometry,
  gemmGeometryTileKeyPart,
  gemmTileM,
  gemmTileN,
} from "../src/kernels/gemm-geometry.ts";
import { i8a8GeometryKeyPart } from "../src/kernels/i8a8-geometry.ts";
import { perChannelGroupSize } from "../src/format/container/codecs.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { I8A8_DOT, type SessionOptions } from "../src/runtime/session-types.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import {
  f32Bytes,
  fill,
  GRAPH_NAME,
  openGraphModel,
  openModelBytes,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";

/** 符号と大きさを散らした決定的な列（乱数は使わない — 失敗が再現しないため）。 */
const SIGNED = (index: number): number => ((index % 11) - 5) * 0.375 + 0.125;
const WEIGHT = (index: number): number => ((index % 7) - 3) * 0.5 - 0.0625;
const BIAS = (index: number): number => ((index % 5) - 2) * 0.25;

const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** 実際に走ったキーのうち、`prefix` で始まり `parts` を全て含むものがあること。 */
type ExpectedKey = { readonly prefix: string; readonly parts: readonly string[] };

type ProfileCase = {
  readonly name: string;
  readonly model: () => Promise<OpenedContainer>;
  readonly inputs: Readonly<Record<string, Tensor>>;
  readonly options?: SessionOptions;
  /** そのプロファイルで走るはずの幾何の判別子。 */
  readonly expected: (profile: GeometryProfile) => readonly ExpectedKey[];
};

const rowsKey = (prefix: string, rows: number) => (profile: GeometryProfile) => [
  { prefix, parts: [gemmGeometryTileKeyPart(gemmRowsGeometry(profile, rows))] },
];

const linearCase = (m: number): ProfileCase => {
  const [k, n] = [64, 96];
  return {
    name: `linear M=${m}`,
    model: async () =>
      await openGraphModel(singleOpDeclaration("linear", [[m, k], [n, k], [n]], [[m, n]])),
    inputs: { x0: fill([m, k], SIGNED), x1: fill([n, k], WEIGHT), x2: fill([n], BIAS) },
    expected: rowsKey("linear:", m),
  };
};

/**
 * `gemmRows` の各規則の代表 M（DECIDED: ADR 0116 決定 7）。有限の規則はその上端（= `maxRows` — 段の
 * 上端は掃引で測った点）、末尾（Infinity）の規則は直前の境界 + 512。表から導くので、段数が変わっても
 * ケースが規則の数だけ追随する。
 */
const representativeRows = (rules: readonly GemmRowsRule[]): readonly number[] =>
  rules.map((rule, index) =>
    rule.maxRows === Number.POSITIVE_INFINITY
      ? (rules[index - 1]?.maxRows ?? 0) + 512
      : rule.maxRows
  );

/**
 * 表の `gemmRows` の規則ごとに 1 ケース（代表 M の linear）。期待する判別子は `gemmRowsGeometry` で
 * 引き直さず**その規則の幾何**から取る — 代表 M が隣の規則に落ちる導出の誤りもキーの不一致で赤になる。
 * 既定と同じ幾何の規則では、判別子が既定の幾何のものになることを見ている（違う幾何なら違うキーの裏返し）。
 */
const gemmRowsCases = (rules: readonly GemmRowsRule[]): readonly ProfileCase[] =>
  representativeRows(rules).map((m, index) => {
    const { maxRows, geometry } = rules[index];
    const range = maxRows === Number.POSITIVE_INFINITY
      ? `> ${rules[index - 1]?.maxRows ?? 0}`
      : `≤ ${maxRows}`;
    return {
      ...linearCase(m),
      name: `linear M=${m}（gemmRows ${range}）`,
      expected: () => [{ prefix: "linear:", parts: [gemmGeometryTileKeyPart(geometry)] }],
    };
  });

/** conv2d の implicit GEMM キーは幾何を `igemm{tileM}x{tileN}` と `:wg{x}x{y}` で名乗る。 */
const conv2dParts = (geometry: GemmGeometry): readonly string[] => [
  `igemm${gemmTileM(geometry)}x${gemmTileN(geometry)}`,
  `:wg${geometry.wgX}x${geometry.wgY}`,
];

const conv2dCase = (channelsOut: number, rows: "rows64" | "rows32"): ProfileCase => {
  const channelsIn = 8;
  return {
    name: `conv2d Cout=${channelsOut}（${rows} クラス）`,
    model: async () =>
      await openGraphModel(
        singleOpDeclaration(
          "conv2d",
          [[1, channelsIn, 20, 20], [channelsOut, channelsIn, 3, 3], [channelsOut]],
          [[1, channelsOut, 20, 20]],
          { attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 } },
        ),
      ),
    inputs: {
      x0: fill([1, channelsIn, 20, 20], SIGNED),
      x1: fill([channelsOut, channelsIn, 3, 3], WEIGHT),
      x2: fill([channelsOut], BIAS),
    },
    expected: (profile) => [{
      prefix: "conv2d:v3:f32:igemm",
      parts: conv2dParts(profile.conv2d[rows]),
    }],
  };
};

const attentionCase = (compute: "f32" | "a8"): ProfileCase => {
  const [b, h, m, n, d] = [1, 2, 300, 200, 64];
  return {
    name: `attention ${compute}`,
    model: async () =>
      await openGraphModel(
        singleOpDeclaration("attention", [[b, h, m, d], [b, h, n, d], [b, h, n, d]], [[
          b,
          h,
          m,
          d,
        ]], { attrs: { scale: halfScale(d) } }),
      ),
    inputs: {
      x0: fill([b, h, m, d], SIGNED),
      x1: fill([b, h, n, d], WEIGHT),
      x2: fill([b, h, n, d], BIAS),
    },
    // 内積変種は両 Session で揃える（device 単位カナリアの判定を比較に混ぜない）。
    options: compute === "a8" ? { attentionCompute: "a8", [I8A8_DOT]: "emu" } : {},
    expected: (profile) =>
      compute === "a8"
        ? [
          {
            prefix: "attention_qk:v3:i8a8:",
            parts: [i8a8GeometryKeyPart(profile.i8a8.attentionQk, false)],
          },
          {
            prefix: "attention_pv:v3:i8a8:",
            parts: [i8a8GeometryKeyPart(profile.i8a8.attentionPv, false)],
          },
        ]
        : [
          {
            prefix: "attention_qk:v1:f32:",
            parts: [gemmGeometryTileKeyPart(profile.attention.qk)],
          },
          {
            prefix: "attention_pv:v1:f32:",
            parts: [gemmGeometryTileKeyPart(profile.attention.pv)],
          },
        ],
  };
};

/** i8 常駐の linear（`linearCompute: "a8"` で i8a8 経路へ落ちる形）。 */
const linearI8a8Case = (): ProfileCase => {
  const [m, k, n] = [300, 64, 96];
  const weight = fill([n, k], (i) => WEIGHT(i) * (1 + (Math.floor(i / k) % 7) * 0.25));
  const quantized = quantizeI8(weight.data, [n, k], 0);
  const bias = fill([n], BIAS);
  return {
    name: "linear i8a8",
    model: () =>
      openModelBytes(
        {
          format: "karume-ir",
          version: 2,
          requires: { ops: ["linear"] },
          symbols: [],
          inputs: [{ name: "x0", dtype: "f32", shape: [m, k] }],
          outputs: ["y"],
          initializers: { w: {}, b: {} },
          values: {
            w: { dtype: "f32", shape: [n, k] },
            b: { dtype: "f32", shape: [n] },
            y: { dtype: "f32", shape: [m, n] },
          },
          nodes: [{ op: "linear", ins: ["x0", "w", "b"], outs: ["y"], attrs: {} }],
        },
        [
          {
            graph: GRAPH_NAME,
            initializer: "b",
            bytes: f32Bytes(bias.data),
            encoding: { codec: "f32" },
          },
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: quantized.bytes,
            encoding: {
              codec: "int8-sym",
              groupSize: perChannelGroupSize(k),
              scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
            },
          },
        ],
      ),
    inputs: { x0: fill([m, k], SIGNED) },
    options: { linearCompute: "a8", [I8A8_DOT]: "emu" },
    expected: (profile) => [{
      prefix: "linear:v4:i8a8:",
      parts: [i8a8GeometryKeyPart(profile.i8a8.linear, false)],
    }],
  };
};

/** linear の行数バケット以外のケース（per-profile 検査は linear を表の規則から作る — `gemmRowsCases`）。 */
const OTHER_CASES: readonly ProfileCase[] = [
  {
    name: "matmul M=700",
    model: async () =>
      await openGraphModel(singleOpDeclaration("matmul", [[700, 64], [64, 80]], [[700, 80]])),
    inputs: { x0: fill([700, 64], SIGNED), x1: fill([64, 80], WEIGHT) },
    expected: rowsKey("matmul:", 700),
  },
  {
    name: "bmm M=600",
    model: async () =>
      await openGraphModel(
        singleOpDeclaration("bmm", [[2, 600, 64], [2, 64, 80]], [[2, 600, 80]]),
      ),
    inputs: { x0: fill([2, 600, 64], SIGNED), x1: fill([2, 64, 80], WEIGHT) },
    expected: rowsKey("bmm:", 600),
  },
  attentionCase("f32"),
  attentionCase("a8"),
  // Cout = 96 は M%64 = 32 なので 32 行クラス・128 は 64 行クラス（`conv2dIgemmMTile`）
  conv2dCase(96, "rows32"),
  conv2dCase(128, "rows64"),
  linearI8a8Case(),
];

const CASES: readonly ProfileCase[] = [
  // 行数バケットの 3 段（≤ 64 / 65〜512 / 513〜）を 1 本ずつ
  linearCase(40),
  linearCase(300),
  linearCase(1024),
  ...OTHER_CASES,
];

/**
 * device は実物のまま、adapterInfo の vendor / architecture / description だけを差し替えた
 * GpuContext（プロファイルの選択は Session 構築で adapterInfo から 1 度だけ行われる）。description も
 * 明示で渡す — 実機の値を写すと、description で照合する表が走らせた機によって当たったり外れたりする。
 * `injected` は `acquireGpu({ geometryProfile })` が GpuContext に渡すのと同じ注入の席。
 *
 * MUST: ここで作った GpuContext は destroy しない — device は元の GpuContext と共有で、破棄は
 * 元の側 1 箇所に置く（二重の destroy は消失通知を予期しない側へ流す）。
 */
const contextAs = (
  gpu: GpuContext,
  vendor: string,
  architecture: string,
  description: string,
  injected?: GeometryProfile,
): GpuContext =>
  new GpuContext(
    gpu.device,
    {
      vendor,
      architecture,
      device: gpu.adapterInfo.device,
      description,
      subgroupMinSize: gpu.adapterInfo.subgroupMinSize,
      subgroupMaxSize: gpu.adapterInfo.subgroupMaxSize,
      isFallbackAdapter: gpu.adapterInfo.isFallbackAdapter,
    },
    gpu.limits,
    gpu.wgslLanguageFeatures,
    undefined,
    injected,
  );

const words = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength / 4);

type CaseRun = {
  readonly y: Uint32Array<ArrayBuffer>;
  readonly profile: string;
  readonly keys: readonly string[];
};

const runCase = async (gpu: GpuContext, testCase: ProfileCase): Promise<CaseRun> => {
  const session = await createSessionFromContainer(
    gpu,
    await testCase.model(),
    GRAPH_NAME,
    testCase.options ?? {},
  );
  try {
    const y = words((await session.run(testCase.inputs))["y"]);
    const diagnostics = session.diagnostics();
    const census = diagnostics.lastRunPipelines;
    assert(census !== undefined, `${testCase.name}: 直近 run のパイプライン内訳が無い`);
    return { y, profile: diagnostics.geometryProfile, keys: census.map((row) => row.key) };
  } finally {
    await session.dispose();
  }
};

/** 1 ケースを既定の Session と比べ、`profile` の名前・幾何判別子のキー・Uint32 一致を見る。 */
const assertRunsWithProfile = async (
  baseline: GpuContext,
  subject: GpuContext,
  profile: GeometryProfile,
  testCase: ProfileCase,
): Promise<void> => {
  const expected = await runCase(baseline, testCase);
  assertEquals(expected.profile, DEFAULT_GEOMETRY_PROFILE.id, testCase.name);
  const actual = await runCase(subject, testCase);
  assertEquals(actual.profile, profile.id, testCase.name);
  for (const { prefix, parts } of testCase.expected(profile)) {
    assert(
      actual.keys.some((key) =>
        key.startsWith(prefix) && parts.every((part) => key.includes(part))
      ),
      `${testCase.name}: ${prefix}…${parts.join("…")} のキーで走っていない` +
        `（実際: ${actual.keys.join(" / ")}）`,
    );
  }
  assertEquals(actual.y, expected.y, `${testCase.name}: 既定プロファイルとビット不一致`);
};

describe("gemmRows の代表 M", () => {
  const rulesOf = (bounds: readonly number[]): GemmRowsRule[] =>
    bounds.map((maxRows) => ({ maxRows, geometry: DEFAULT_GEOMETRY_PROFILE.gemmRows[0].geometry }));

  it("3 段の表（既定の 64 / 512 / ∞）では各段の上端と、末尾は直前の境界 + 512 になる", () => {
    assertEquals(representativeRows(DEFAULT_GEOMETRY_PROFILE.gemmRows), [64, 512, 1024]);
  });

  it("7 段の表（16 / 32 / 64 / 128 / 256 / 512 / ∞）では規則の数だけ代表 M が出る", () => {
    const rules = rulesOf([16, 32, 64, 128, 256, 512, Number.POSITIVE_INFINITY]);
    assertEquals(representativeRows(rules), [16, 32, 64, 128, 256, 512, 1024]);
  });

  it("規則ごとのケースは、その規則の幾何の判別子を期待する", () => {
    const cases = gemmRowsCases(DEFAULT_GEOMETRY_PROFILE.gemmRows);
    assertEquals(
      cases.map((testCase) => testCase.expected(DEFAULT_GEOMETRY_PROFILE)[0].parts),
      DEFAULT_GEOMETRY_PROFILE.gemmRows.map((rule) => [gemmGeometryTileKeyPart(rule.geometry)]),
    );
  });
});

describe({
  name: "幾何プロファイルの診断（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => {
    it("Session は実機の adapter から選ばれたプロファイルの名前を診断に出す", async () => {
      const gpu = await acquireGpu();
      try {
        const run = await runCase(gpu, linearCase(40));
        assertEquals(run.profile, selectGeometryProfile(gpu.adapterInfo).id);
        // adapterInfo が空の機（古い Chromium の adapter.info 欠落）は既定に落ちる
        const blank = await runCase(contextAs(gpu, "", "", ""), linearCase(40));
        assertEquals(blank.profile, DEFAULT_GEOMETRY_PROFILE.id);
      } finally {
        gpu.destroy();
      }
    });
  },
});

describe({
  name: "埋め込みプロファイルの幾何は既定プロファイルとビット同一（実 GPU）",
  ignore: !GPU_AVAILABLE || BUILTIN_GEOMETRY_PROFILES.length === 0,
  fn: () => {
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      const route = profile.match === undefined ? "注入" : "adapter の選択";
      it(`'${profile.id}'（${route}）の Session は全ケースで既定と Uint32 一致し、その幾何のキーで走る`, async () => {
        const gpu = await acquireGpu();
        try {
          const baseline = contextAs(gpu, "", "", "");
          // match のある表は adapter を偽装して選択の経路（description を含む）で、match を省いた
          // 注入専用の表は注入の席で組む（adapter は空 — 選択では既定にしか落ちない）。
          const subject = profile.match === undefined
            ? contextAs(gpu, "", "", "", profile)
            : contextAs(
              gpu,
              profile.match.vendor ?? "",
              profile.match.architecture ?? "",
              profile.match.description ?? "",
            );
          const rowCases = gemmRowsCases(profile.gemmRows);
          assertEquals(rowCases.length, profile.gemmRows.length, "gemmRows の規則を取りこぼした");
          for (const testCase of [...rowCases, ...OTHER_CASES]) {
            await assertRunsWithProfile(baseline, subject, profile, testCase);
          }
        } finally {
          gpu.destroy();
        }
      });
    }
  },
});

describe({
  name: "acquireGpu で注入した幾何プロファイル（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => {
    it("apple-metal-3 を注入した device の Session は全ケースでその表の名前・幾何のキーで走り、既定と Uint32 一致する", async () => {
      const plain = await acquireGpu();
      // 渡した表は acquire 時に複製される — 後から書き換えても device の表（id）は変わらない
      const handedOver = structuredClone(APPLE_METAL_3);
      const injected = await acquireGpu({ geometryProfile: handedOver });
      (handedOver as { id: string }).id = "changed-after-acquire";
      try {
        const baseline = contextAs(plain, "", "", "");
        for (const testCase of CASES) {
          await assertRunsWithProfile(baseline, injected, APPLE_METAL_3, testCase);
        }
      } finally {
        injected.destroy();
        plain.destroy();
      }
    });

    it("コールバック形（ADR 0117 決定 6）: device の adapter 情報で 1 度呼ばれ、返した埋め込みの表で Session が走り既定と Uint32 一致する・undefined なら自動選択", async () => {
      const plain = await acquireGpu();
      const seen: GPUAdapterInfo[] = [];
      const injected = await acquireGpu({
        geometryProfile: (adapterInfo) => {
          seen.push(adapterInfo);
          return APPLE_METAL_3;
        },
      });
      const automatic = await acquireGpu({ geometryProfile: () => undefined });
      try {
        assertEquals(seen.length, 1);
        assert(
          seen[0] === injected.adapterInfo,
          "コールバックの adapterInfo が GpuContext のものと別物",
        );
        const baseline = contextAs(plain, "", "", "");
        for (const testCase of [linearCase(40), linearCase(300), linearI8a8Case()]) {
          await assertRunsWithProfile(baseline, injected, APPLE_METAL_3, testCase);
        }
        const run = await runCase(automatic, linearCase(40));
        assertEquals(run.profile, selectGeometryProfile(automatic.adapterInfo).id);
      } finally {
        automatic.destroy();
        injected.destroy();
        plain.destroy();
      }
    });

    it("注入した表は adapter と match を見ない（apple-metal-3 に当たる adapter でも、別 vendor の表が使われる）", async () => {
      const gpu = await acquireGpu();
      try {
        const baseline = contextAs(gpu, "", "", "");
        // 埋め込みの選択なら apple-metal-3 が当たる adapter に、match が nvidia の表を注入する
        const subject = contextAs(
          gpu,
          "apple",
          "metal-3",
          APPLE_METAL_3.match?.description ?? "",
          NVIDIA_BLACKWELL,
        );
        assertEquals(selectGeometryProfile(subject.adapterInfo).id, APPLE_METAL_3.id);
        for (const testCase of [linearCase(40), linearCase(300), linearI8a8Case()]) {
          await assertRunsWithProfile(baseline, subject, NVIDIA_BLACKWELL, testCase);
        }
      } finally {
        gpu.destroy();
      }
    });
  },
});

describe({
  name: "dp4a カナリアは注入した表の i8a8 attention 幾何で撃つ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => {
    it("attentionCompute a8 の Session 構築で、カナリアが注入した表の ①QK / ③PV 幾何の WGSL をコンパイルする", async (context) => {
      const gpu = await acquireGpu({ geometryProfile: APPLE_METAL_3 });
      try {
        if (!dp4aAvailable(gpu.wgslLanguageFeatures)) {
          // 非広告の device ではカナリアが走らない（emu 直行）— 見る対象が無いので明示 SKIP
          console.warn(`SKIP ${context.name}: dot4I8Packed が広告されていない`);
          return;
        }
        const compiled: string[] = [];
        const device = gpu.device;
        const createShaderModule = device.createShaderModule.bind(device);
        device.createShaderModule = (descriptor) => {
          compiled.push(descriptor.code);
          return createShaderModule(descriptor);
        };
        const testCase = attentionCase("a8");
        const session = await createSessionFromContainer(
          gpu,
          await testCase.model(),
          GRAPH_NAME,
          { attentionCompute: "a8" },
        );
        try {
          assertEquals(session.diagnostics().geometryProfile, APPLE_METAL_3.id);
        } finally {
          await session.dispose();
        }
        // カナリアは dp4a の腕を必ず先に撃つ（v4 / S = f32 の組を含む）
        const injectedQk = attentionQkI8a8Wgsl(true, true, "f32", APPLE_METAL_3.i8a8.attentionQk);
        const injectedPv = attentionPvI8a8Wgsl(true, true, "f32", APPLE_METAL_3.i8a8.attentionPv);
        const defaultQk = attentionQkI8a8Wgsl(
          true,
          true,
          "f32",
          DEFAULT_GEOMETRY_PROFILE.i8a8.attentionQk,
        );
        assert(injectedQk !== defaultQk, "注入した表の ①QK 幾何が既定と同じ（検査が空振りする）");
        assert(compiled.includes(injectedQk), "カナリアが注入した表の ①QK 幾何を撃っていない");
        assert(compiled.includes(injectedPv), "カナリアが注入した表の ③PV 幾何を撃っていない");
        assert(!compiled.includes(defaultQk), "カナリアが既定の ①QK 幾何を撃った");
      } finally {
        gpu.destroy();
      }
    });
  },
});
