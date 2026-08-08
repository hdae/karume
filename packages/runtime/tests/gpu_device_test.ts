import {
  assert,
  assertEquals,
  assertGreaterOrEqual,
  assertInstanceOf,
  assertRejects,
  assertThrows,
} from "@std/assert";
import {
  acquireGpu,
  assertLimitsGranted,
  GpuDeviceLostError,
  GpuFeatureError,
  GpuLimitError,
  GpuOutOfMemoryError,
  GpuValidationError,
  planRequiredLimits,
  planShaderF16Feature,
  popFailureScopes,
  pushFailureScopes,
  readAdapterInfo,
  REQUIRED_LIMIT_KEYS,
  type RequiredLimits,
  SHADER_F16_FEATURE,
} from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { fakeDevice, fakeGpuContext, losableGpuContext } from "./helpers/fake-gpu.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";

/** planRequiredLimits に渡す最小のダック型（GPUSupportedLimits の必要な面のみ）。 */
const fakeAdapterLimits = (overrides: Partial<RequiredLimits> = {}): GPUSupportedLimits => {
  const base: RequiredLimits = {
    maxBufferSize: 1024,
    maxStorageBufferBindingSize: 1024,
    maxUniformBufferBindingSize: 256,
    maxStorageBuffersPerShaderStage: 8,
    maxUniformBuffersPerShaderStage: 8,
    maxComputeWorkgroupStorageSize: 32768,
    maxComputeInvocationsPerWorkgroup: 1024,
    maxComputeWorkgroupSizeX: 1024,
    maxComputeWorkgroupSizeY: 1024,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeWorkgroupsPerDimension: 65535,
  };
  return { ...base, ...overrides } as unknown as GPUSupportedLimits;
};

Deno.test("planRequiredLimits は maxStorageBufferBindingSize を maxBufferSize 以下に収める", () => {
  const planned = planRequiredLimits(
    fakeAdapterLimits({ maxBufferSize: 512, maxStorageBufferBindingSize: 4096 }),
  );

  assertEquals(planned.maxStorageBufferBindingSize, 512);
  assertEquals(planned.maxBufferSize, 512);
});

Deno.test("planRequiredLimits は compute 系をアダプタ値まで引き上げる", () => {
  const planned = planRequiredLimits(fakeAdapterLimits());

  assertEquals(planned.maxComputeInvocationsPerWorkgroup, 1024, "仕様既定値 256 に落とさない");
  assertEquals(planned.maxComputeWorkgroupSizeX, 1024, "invocations とは別 limit として要求する");
  assertEquals(planned.maxComputeWorkgroupStorageSize, 32768);
  assertEquals(planned.maxComputeWorkgroupsPerDimension, 65535);
});

Deno.test("assertLimitsGranted は要求を下回る limit を fail loudly にする", () => {
  const planned = planRequiredLimits(fakeAdapterLimits());
  const granted = fakeAdapterLimits({ maxComputeInvocationsPerWorkgroup: 256 });

  assertThrows(
    () => assertLimitsGranted(planned, granted),
    GpuLimitError,
    "maxComputeInvocationsPerWorkgroup",
  );
  assertLimitsGranted(planned, fakeAdapterLimits());
});

Deno.test({
  name: "acquireGpu は引き上げ済み limits の device を返す（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const key of REQUIRED_LIMIT_KEYS) {
        assertGreaterOrEqual(gpu.device.limits[key], gpu.limits[key], `${key} が要求を下回る`);
      }
      // 仕様既定値のままなら compute 系の引き上げが効いていない
      assertGreaterOrEqual(gpu.limits.maxComputeInvocationsPerWorkgroup, 256);
      assertGreaterOrEqual(gpu.limits.maxComputeWorkgroupSizeX, 256);
      assert(gpu.limits.maxStorageBufferBindingSize <= gpu.limits.maxBufferSize);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "acquireGpu は features / wgslLanguageFeatures を集合に正規化する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      assert(gpu.features instanceof Set);
      assert(gpu.wgslLanguageFeatures instanceof Set, "未実装環境では空集合に縮退する");
      for (const feature of gpu.features) {
        assertEquals(typeof feature, "string");
      }
    } finally {
      gpu.destroy();
    }
  },
});

// 「true × feature 不在」だけは実機で作れない（アダプタの feature は消せない）ため、
// 判定を切り出した純関数で全 6 通りを固定する。ADR 0028。
Deno.test("planShaderF16Feature は undefined を「要求しない」に倒し、必須指定の不足は fail loudly", () => {
  const present = new Set(["shader-f16", "timestamp-query"]);
  const absent = new Set(["timestamp-query"]);

  // MUST: gpuTiming の自動判定と**逆**。f16 計算は数値を変えるので、アダプタの能力で
  // 有効・無効が決まると「機械を替えたら黙って出力が変わる」ことになる。
  assertEquals(planShaderF16Feature(present, undefined), false, "既定は要求しない（auto 禁止）");
  assertEquals(planShaderF16Feature(absent, undefined), false);
  assertEquals(planShaderF16Feature(present, true), true);
  assertEquals(planShaderF16Feature(present, false), false);
  assertEquals(planShaderF16Feature(absent, false), false);

  const error = assertThrows(
    () => planShaderF16Feature(absent, true),
    GpuFeatureError,
    SHADER_F16_FEATURE,
  );
  // 不足内容と回避策が読めること（環境で何が使えるか + 既定経路へ戻す手順）。
  assert(error.message.includes("timestamp-query"), error.message);
  assert(error.message.includes("f32"), error.message);
});

Deno.test({
  name:
    "acquireGpu({shaderF16}) は feature を要求し、実走カナリアを通したときだけ有効になる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // ① 既定では要求しない（既存経路の device 能力面を 1 ビットも広げない）
    const plain = await acquireGpu();
    try {
      assertEquals(plain.shaderF16Enabled, false);
      assertEquals(plain.features.has(SHADER_F16_FEATURE), false);
    } finally {
      plain.destroy();
    }
    // ② false 指定は「持っていても要求しない」（意図の表明）
    const declined = await acquireGpu({ shaderF16: false });
    try {
      assertEquals(declined.shaderF16Enabled, false);
    } finally {
      declined.destroy();
    }
    if (!SHADER_F16_AVAILABLE) {
      // ③' 列挙しないアダプタでは true が fail loudly になる（黙って f32 へ落ちない）
      await assertRejects(() => acquireGpu({ shaderF16: true }), GpuFeatureError);
      return;
    }
    // ③ true 指定は feature を載せ、**カナリアの実走**（実行時の値を f16 共有タイルへ通した
    // RTNE の既知解）を通ったときだけ返る。列挙だけで返す実装なら denoland/deno#23125 の
    // 沈黙全 0 を見逃す。
    const enabled = await acquireGpu({ shaderF16: true });
    try {
      assertEquals(enabled.shaderF16Enabled, true);
      assertEquals(enabled.features.has(SHADER_F16_FEATURE), true);
    } finally {
      enabled.destroy();
    }
  },
});

Deno.test("readAdapterInfo は info を持たないアダプタでも安全な空値に正規化する", () => {
  // 古い Chromium（requestAdapterInfo() 時代）は adapter.info を持たない。
  const empty = readAdapterInfo({});
  assertEquals(empty.vendor, "");
  assertEquals(empty.architecture, "");
  assertEquals(empty.device, "");
  assertEquals(empty.description, "");
  assertEquals(empty.isFallbackAdapter, false);

  const present = { vendor: "acme", architecture: "a", device: "d", description: "x" };
  assertEquals(readAdapterInfo({ info: present as GPUAdapterInfo }).vendor, "acme");
});

Deno.test("withScopeLock は errorScope 区間を device 単位で直列化する", async () => {
  const gpu = fakeGpuContext(fakeDevice());
  const trace: string[] = [];
  let active = 0;
  let maxActive = 0;

  /** `ticks` の差で「後発が先に決着しうる」形を作る（ロックが無ければ区間が重なる）。 */
  const body = (label: string, ticks: number) => async (): Promise<string> => {
    active += 1;
    maxActive = Math.max(maxActive, active);
    trace.push(`${label}:enter`);
    for (let i = 0; i < ticks; i += 1) await Promise.resolve();
    trace.push(`${label}:exit`);
    active -= 1;
    return label;
  };

  const results = await Promise.all([
    gpu.withScopeLock(body("a", 6)),
    gpu.withScopeLock(body("b", 1)),
    gpu.withScopeLock(body("c", 3)),
  ]);

  assertEquals(maxActive, 1, "2 つの区間が同時に走っていない");
  assertEquals(trace, ["a:enter", "a:exit", "b:enter", "b:exit", "c:enter", "c:exit"]);
  assertEquals(results, ["a", "b", "c"], "戻り値は各呼び出し自身の結果");
});

Deno.test("withScopeLock は 1 本の区間の失敗を後続に伝播しない", async () => {
  const gpu = fakeGpuContext(fakeDevice());
  const failure = new Error("区間内の失敗");

  const failed = gpu.withScopeLock(() => Promise.reject(failure));
  const following = gpu.withScopeLock(() => Promise.resolve("ok"));

  assertEquals(await assertRejects(() => failed), failure);
  assertEquals(await following, "ok", "チェーンは決着だけを繋ぐので後続は巻き添えにならない");
});

Deno.test("onLost は解除でき、購読時点で消失済みなら即時通知する", async () => {
  const { gpu, lose } = losableGpuContext();
  const seen: string[] = [];

  gpu.onLost(() => seen.push("kept"));
  const cancel = gpu.onLost(() => seen.push("cancelled"));
  cancel();

  lose();
  await gpu.device.lost;
  assertEquals(seen, ["kept"], "解除した購読は呼ばれない");

  // 消失後の購読は取りこぼさないよう即時（同期）に発火する
  gpu.onLost(() => seen.push("late"));
  assertEquals(seen, ["kept", "late"]);
});

Deno.test("raceDeviceLost は消失を例外にし、決着時に購読を解除する", async () => {
  const { gpu, lose } = losableGpuContext();
  assertEquals(gpu.pendingLostListeners, 0);

  // work が先に決着すれば購読は残らない（flush / readback ごとに reaction が積み残ると、
  // 長寿命 Session で単調増加するリークになる — 挙動には現れないので件数で固定する）
  for (let i = 0; i < 3; i += 1) {
    assertEquals(await gpu.raceDeviceLost(Promise.resolve(7), "work"), 7);
  }
  assertEquals(gpu.pendingLostListeners, 0, "決着した待機の購読は 1 本も残らない");

  // 解決しない待機は消失で例外になる（ハングにしない）
  const hanging = gpu.raceDeviceLost(new Promise<void>(() => {}), "flush");
  assertEquals(gpu.pendingLostListeners, 1, "待機中は購読が立っている（数えられている根拠）");
  lose();
  const error = await assertRejects(() => hanging, GpuDeviceLostError);
  assert(error.message.includes("flush"), error.message);
  assertEquals(gpu.pendingLostListeners, 0);
});

/**
 * popFailureScopes が触る面（popErrorScope）だけのフェイク device。`scopes` は pop の発行順
 * ＝ [validation, out-of-memory] に対応する（スタック先頭が validation）。
 */
const poppingDevice = (scopes: readonly (string | undefined)[]): GPUDevice => {
  const queue = [...scopes];
  return {
    popErrorScope: (): Promise<GPUError | null> => {
      const message = queue.shift();
      return Promise.resolve(message === undefined ? null : { message } as GPUError);
    },
  } as unknown as GPUDevice;
};

Deno.test("popFailureScopes は両スコープ捕捉時に OOM を返す（validation は派生）", async () => {
  // 余力切れの createBuffer は無効バッファを返し、それを使う後続の操作が
  // `Buffer with '' label is invalid` を validation に立てる。validation を先に返すと根因の
  // OOM が捨てられ、破棄後使用と区別のつかない報告になる（実測事故の再発防止）。
  const both = await popFailureScopes(
    poppingDevice(["Buffer with '' label is invalid", "not enough memory left"]),
    "重みのアップロード",
  );
  assertInstanceOf(both, GpuOutOfMemoryError, "派生の validation でなく根因の OOM を返す");
  assert(both.message.startsWith("重みのアップロード: "), both.message);
  assert(both.message.includes("not enough memory left"), both.message);

  // 純 validation（OOM スコープが空）の挙動は不変
  const validationOnly = await popFailureScopes(
    poppingDevice(["Buffer with '' label has been destroyed", undefined]),
    "run のエンコード",
  );
  assertInstanceOf(validationOnly, GpuValidationError);
  assert(validationOnly.message.includes("has been destroyed"), validationOnly.message);

  assertEquals(await popFailureScopes(poppingDevice([]), "clean"), undefined);
});

Deno.test({
  name: "並行する errorScope 区間で validation 失敗が正しい側に帰属する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const pipeline = await new PipelineCache(gpu.device).get(
        "scope-lock-probe",
        `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&data)) { data[gid.x] = data[gid.x] + 1.0; }
}
`,
      );
      const layout = pipeline.getBindGroupLayout(0);

      /**
       * #runOnce と同型の区間（push → await を跨ぐエンコード → pop）。`ticks` の差で
       * 「エラーを起こした側が先に pop する」非対称な重なりを作る。
       *
       * ロックが無いとこの形は実測で誤帰属する（faulty=clean / innocent=ERROR）。pop は
       * スタック先頭を無条件に取るため、faulty は後から積まれた innocent の空スコープを取り、
       * innocent が faulty のエラーを拾う。
       */
      const scoped = (faulty: boolean, ticks: number): Promise<Error | undefined> =>
        gpu.withScopeLock(async () => {
          pushFailureScopes(gpu.device);
          // binding 0 が欠落した bindGroup は同期例外にならず errorScope にだけ現れる
          if (faulty) gpu.device.createBindGroup({ layout, entries: [] });
          for (let i = 0; i < ticks; i += 1) await Promise.resolve();
          return await popFailureScopes(gpu.device, faulty ? "faulty" : "innocent");
        });

      const [faultyResult, innocentResult] = await Promise.all([
        scoped(true, 1),
        scoped(false, 8),
      ]);

      assertInstanceOf(faultyResult, GpuValidationError, "エラーを起こした側が捕捉する");
      assert(faultyResult.message.startsWith("faulty:"), faultyResult.message);
      assertEquals(innocentResult, undefined, "無関係な区間が巻き添えで落ちない");
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "GpuContext は device.lost を購読し、意図的な破棄と区別して記録する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    let notified = 0;
    const gpu = await acquireGpu({ onDeviceLost: () => notified += 1 });

    assertEquals(gpu.lost, undefined);
    assertEquals(gpu.destroyRequested, false);

    gpu.destroy();
    // 内部購読はコンストラクタで登録済みなので、この await より先に走る
    const info = await gpu.device.lost;

    assertEquals(info.reason, "destroyed");
    assertEquals(gpu.destroyRequested, true);
    assertEquals(gpu.lost?.reason, "destroyed");
    assertEquals(notified, 0, "意図的な破棄は消失通知の対象外");
  },
});
