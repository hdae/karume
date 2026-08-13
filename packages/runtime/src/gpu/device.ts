/**
 * WebGPU デバイスの取得と能力の正規化。
 *
 * この層の責務は「実装差のある面（features / limits / WGSL 言語機能）を 1 箇所で正規化し、
 * 沈黙故障を全て loud な例外に変換すること」。
 *
 * MUST: device をモジュールスコープに捕獲しない。VRAM を返すのは `device.destroy()` のみで
 * （`buffer.destroy()` は 1 バイトも返さない）、解放後の続行は `acquireGpu()` からの再構築に
 * なる。device を握る層（PipelineCache / SubmitScheduler / RunArena）は全て GpuContext と
 * 同じ寿命で作り直せる構造でなければならない。
 */

/** navigator.gpu が無い / アダプタを取得できない。 */
export class GpuUnavailableError extends Error {
  override readonly name = "GpuUnavailableError";
}

/** 要求した limit を device が満たさない（要求漏れ・仕様既定値への降格の検出）。 */
export class GpuLimitError extends Error {
  override readonly name = "GpuLimitError";
}

/** errorScope が捕捉した validation エラー（無効パイプライン等）。 */
export class GpuValidationError extends Error {
  override readonly name = "GpuValidationError";
}

/**
 * errorScope が捕捉した out-of-memory エラー（確保要求が device の余力を超えた）。
 * validation と違い利用者のモデルサイズと実行環境で決まるため、別型で分岐可能にする。
 */
export class GpuOutOfMemoryError extends Error {
  override readonly name = "GpuOutOfMemoryError";
}

/** device が失われた状態での操作。待ち続ける代わりに必ずこれを投げる。 */
export class GpuDeviceLostError extends Error {
  override readonly name = "GpuDeviceLostError";
}

/**
 * 必須と指定した feature をアダプタが持たない（{@link AcquireGpuOptions.gpuTiming} = `true` 等）。
 * limit の不足（{@link GpuLimitError}）と分けるのは、利用者の分岐先が違うため — feature は
 * 「その環境では諦めて機能を落とす」で続行できるが、limit 不足は続行してはいけない。
 */
export class GpuFeatureError extends Error {
  override readonly name = "GpuFeatureError";
}

/**
 * requiredLimits に明示要求する limit。
 *
 * MUST: 要求しなかった limit はアダプタ値ではなく**仕様既定値**に落ちる。既定は workgroup
 * ストレージ 16384 バイト / workgroup あたり 256 invocation / workgroupSize (256,256,64) で、
 * 要求しないままだとアダプタが 1024 invocation を出せてもカーネルはそこで頭打ちになる。
 *
 * MUST: `maxComputeWorkgroupSizeX/Y/Z` は `maxComputeInvocationsPerWorkgroup` とは別の limit。
 * invocations だけ引き上げても `@workgroup_size(512)` は通らず、しかも**失敗の仕方が静か**
 * （`createComputePipeline` は throw せず、無効パイプラインへの dispatch が no-op になって
 * 出力が全て 0 になる）。この沈黙故障は {@link withValidationScope} でのみ可視化できる。
 */
export const REQUIRED_LIMIT_KEYS = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxUniformBufferBindingSize",
  "maxStorageBuffersPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
] as const;

export type RequiredLimitKey = (typeof REQUIRED_LIMIT_KEYS)[number];

export type RequiredLimits = Readonly<Record<RequiredLimitKey, number>>;

/**
 * アダプタ実測値から requestDevice に渡す requiredLimits を組み立てる。
 * 要求値がアダプタ値そのものなら requestDevice は必ず成功する。
 *
 * MUST: 仕様上 `maxStorageBufferBindingSize ≤ maxBufferSize`。片方だけ引き上げると
 * requestDevice が丸ごと失敗するため、両者は必ず同時に計画する。
 */
export const planRequiredLimits = (adapterLimits: GPUSupportedLimits): RequiredLimits => {
  const maxBufferSize = adapterLimits.maxBufferSize;
  // Record<RequiredLimitKey, number> の網羅性検査で、キー一覧との同期は型で保証される。
  const planned: Record<RequiredLimitKey, number> = {
    maxBufferSize,
    maxStorageBufferBindingSize: Math.min(adapterLimits.maxStorageBufferBindingSize, maxBufferSize),
    maxUniformBufferBindingSize: adapterLimits.maxUniformBufferBindingSize,
    maxStorageBuffersPerShaderStage: adapterLimits.maxStorageBuffersPerShaderStage,
    maxUniformBuffersPerShaderStage: adapterLimits.maxUniformBuffersPerShaderStage,
    maxComputeWorkgroupStorageSize: adapterLimits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: adapterLimits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapterLimits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapterLimits.maxComputeWorkgroupSizeZ,
    maxComputeWorkgroupsPerDimension: adapterLimits.maxComputeWorkgroupsPerDimension,
  };
  return planned;
};

/**
 * 取得した device の limits が要求以上であることを検査する。
 * 要求漏れ・実装都合の降格は「カーネルが静かに頭打ちになる」形で現れるため、ここで落とす。
 */
export const assertLimitsGranted = (
  planned: RequiredLimits,
  granted: GPUSupportedLimits,
): void => {
  const shortfalls = REQUIRED_LIMIT_KEYS
    .filter((key) => granted[key] < planned[key])
    .map((key) => `${key}: granted ${granted[key]} < required ${planned[key]}`);
  if (shortfalls.length > 0) {
    throw new GpuLimitError(`requiredLimits が満たされていない: ${shortfalls.join(", ")}`);
  }
};

/**
 * MUST NOT: 使っていない feature を要求しない。要求した feature は device の能力面を広げる
 * だけでなく、アダプタ側の実装によっては取得の失敗要因にもなる。重み格納の経路は f32 /
 * f16 / i8 とも `enable f16` を出さない（unpack2x16float — ADR 0018）ので、**無条件に要求
 * する feature は無い**。
 * NOTE: 全廃方針は ADR 0021 で「**既定では**何も要求しない」に読み替えた。条件付きで載るのは
 * GPU 側時間計測（{@link TIMESTAMP_QUERY_FEATURE} — {@link AcquireGpuOptions.gpuTiming} の
 * 三値）と f16 **計算**変種（{@link SHADER_F16_FEATURE} —
 * {@link AcquireGpuOptions.shaderF16}）の 2 本だけ。実際に有効化された feature の照会は
 * {@link GpuContext.features} で行える。
 */
const REQUIRED_FEATURES: readonly GPUFeatureName[] = [];

/** GPU 側時間計測（pass 境界の timestamp）に要る feature — ADR 0021。 */
export const TIMESTAMP_QUERY_FEATURE: GPUFeatureName = "timestamp-query";

/**
 * f16 **計算**変種（共有タイルを f16 にする GEMM — `enable f16`）に要る feature。
 *
 * MUST: 重み**格納** f16（ADR 0018）と混同しない。格納側は core WGSL の `unpack2x16float`
 * だけで動き、この feature を一切要求しない。こちらは WGSL の `f16` 型そのものを使うので
 * feature が無ければシェーダのコンパイルが通らない。
 */
export const SHADER_F16_FEATURE: GPUFeatureName = "shader-f16";

/**
 * feature 集合として読む面だけを取り出した形。`adapter.features`
 * （`ReadonlySet<GPUFeatureName>`）と素の `Set<string>` の**両方**を受けるために要る
 * （前者は要素型が狭く、後者は `ReadonlySet<GPUFeatureName>` を満たさない）。
 */
export type GpuFeatureSet = Iterable<string> & { has(feature: string): boolean };

/**
 * {@link AcquireGpuOptions.gpuTiming} の三値から「timestamp-query を要求するか」を決める。
 *
 * MUST: `undefined` は「要求しない」（{@link planShaderF16Feature} と同じ規律）。自動判定に
 * はしない — 計測が有効な device では 1 dispatch = 1 pass に開くため、**アダプタの能力で
 * 壁時計が変わる**（実測: 1 step あたり 370〜375ms・解像度非依存で、しかも作った GPUBuffer
 * の累計に比例して単価が伸びる。`docs/research/2026-08-04-host-overhead-recon.md` §3.2/§4.2）。
 * 診断を無償で配ると「計測すると遅くなる」状態が既定になり、perf の基準そのものが歪む。
 *
 * device 取得から切り出した純関数なのは、`true` × feature 不在という分岐が**実機では作れない**
 * ため（アダプタが持つ feature は消せない）。判定だけを単体テストできる形にしてある。
 */
export const planTimestampFeature = (
  adapterFeatures: GpuFeatureSet,
  requested: boolean | undefined,
): boolean => {
  if (requested !== true) return false;
  if (!adapterFeatures.has(TIMESTAMP_QUERY_FEATURE)) {
    throw new GpuFeatureError(
      `gpuTiming: true を指定したが、アダプタが '${TIMESTAMP_QUERY_FEATURE}' を持たない` +
        `（利用可能: ${[...adapterFeatures].sort().join(", ") || "なし"}）。` +
        "計測なしで続行するなら gpuTiming を省略する（= 要求しない）か false を指定する",
    );
  }
  return true;
};

/**
 * {@link AcquireGpuOptions.shaderF16} から「shader-f16 を要求するか」を決める。
 *
 * MUST: `undefined` は「要求しない」（{@link planTimestampFeature} と同じ規律）。f16 計算変種は
 * **数値を変える**ので、アダプタの能力で有効・無効が決まる形にすると「機械を替えたら黙って
 * 出力が変わる」になる。opt-in 以外はあり得ない（ADR 0028）。
 *
 * {@link planTimestampFeature} と同じく純関数に切り出してあるのは、`true` × feature 不在と
 * いう分岐が**実機では作れない**ため（アダプタが持つ feature は消せない）。
 */
export const planShaderF16Feature = (
  adapterFeatures: GpuFeatureSet,
  requested: boolean | undefined,
): boolean => {
  if (requested !== true) return false;
  if (!adapterFeatures.has(SHADER_F16_FEATURE)) {
    throw new GpuFeatureError(
      `shaderF16: true を指定したが、アダプタが '${SHADER_F16_FEATURE}' を持たない` +
        `（利用可能: ${[...adapterFeatures].sort().join(", ") || "なし"}）。` +
        "f16 計算変種（attentionCompute / linearCompute の 'f16'）はこの環境では使えない — " +
        "shaderF16 を省略して既定の f32 経路で実行する",
    );
  }
  return true;
};

/**
 * shader-f16 を要求して有効化された device で、**既知解を返す極小 f16 カーネルを実走**して
 * 突き合わせる（1 dispatch + 読み戻し）。
 *
 * MUST: feature の列挙を「動く」の証拠にしない。`denoland/deno#23125`（open）は
 * **shader-f16 を要求した `enable f16` のコンピュートシェーダが、エラーも警告も出さずに
 * 出力全 0 になる**という報告で、同じコードが別実装では動く。列挙だけを信じると
 * 「f16 変種を選んだのに全ての attention / linear が 0 を返す」形の沈黙故障になる —
 * Karume が「無効パイプラインの沈黙 no-op」に errorScope を常設しているのと同じ思想で、
 * ここは実走の門を置く以外に検出手段が無い。
 *
 * 検査は 2 点:
 *
 * 1. `out[0] == 4.0029296875` — **実行時の値**を f16 の共有タイルへ通した往復。`lid` 由来の
 *    非定数を通すので定数畳み込みでは消えず、丸めは共有メモリの**要素型**が強制する。
 *    4 レーンの値は `1 + lid·2^-11` で、binary16 の RTNE により
 *    `1.0 / 1.0（同点→偶数側） / 1+2^-10 / 1+2^-9（同点→偶数側）` = 合計 4.0029296875。
 *    全 0（denoland/deno#23125）も、丸めが起きない実装（合計 4.00146484375）も、
 *    共有メモリを素通りさせる実装も、この 1 本で赤くなる。
 * 2. `out[1] == 1.0` — 定数式での f16 セマンティクス（`1 + 2^-11` の同点が偶数側 1.0 へ）。
 *    こちらは**コンパイル時の畳み込み**を見ている可能性が高いので、① の補助と位置づける。
 *
 * NOTE（本波の実測・2026-08-04）: レジスタ上の `vec4<f32>(vec4<f16>(x))` という往復は、
 * この環境のコンパイラに**恒等として消される**（故障注入で確認）。丸めを保証できるのは
 * 「f16 の**格納**を経由する」形だけで、本体のカーネルが共有タイルの要素型で丸めているのは
 * この性質に乗っている。カナリアも同じ理由で共有タイル経由にしてある。
 *
 * コストは `shaderF16: true` を渡した呼び出しにだけ乗る。
 */
const SHADER_F16_CANARY_WGSL =
  `// karume shader-f16 カナリア（既知解の突合 — feature 列挙は「動く」の証拠にならない）
enable f16;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

var<workgroup> tile: array<f16, 4>;

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  // 実行時の値（lid 由来）を f16 の共有タイルへ通す — 定数畳み込みでは消えない経路
  tile[lid.x] = f16(1.0 + f32(lid.x) * 0.00048828125);
  workgroupBarrier();
  if (lid.x == 0u) {
    // ① 共有 f16 タイルの往復 + RTNE（1.0 + 1.0 + 1.0009765625 + 1.001953125）
    var acc = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
      acc = acc + f32(tile[i]);
    }
    out[0] = acc;
    // ② 定数式での f16 セマンティクス（1 + 2^-11 は binary16 の同点 → 偶数側の 1.0）
    out[1] = f32(f16(1.0 + 0.00048828125));
  }
}
`;

/** カナリアの期待値（① 共有 f16 タイル経由の RTNE / ② 定数式の f16 セマンティクス）。 */
const SHADER_F16_CANARY_EXPECTED: readonly number[] = [4.0029296875, 1];

/**
 * カナリア区間**ローカル**の消失レース。`work` の決着と `device.lost` を競わせ、消失が先なら
 * {@link GpuDeviceLostError} にする。
 *
 * MUST: カナリアの待ちは 1 つ残らずここを通す。device 消失後 `popErrorScope` / `mapAsync` は
 * 解決しない（{@link GpuContext.raceDeviceLost} の doc と同じ事実）ので、競わせないと
 * `acquireGpu()` が例外ではなく**ハングで返ってこない**。
 *
 * MUST NOT: この形を待ちの多い層へ持ち出さない。`device.lost.then(...)` は解除手段の無い
 * reaction なので、flush / readback のような繰り返す待ちに張ると単調増加する（購読の一本化は
 * {@link GpuContext.onLost} の責務）。カナリアは `GpuContext` 生成**前**に 1 回だけ走り、
 * 積むのは取得 1 回あたり数本なので、ここに限って直に競わせてよい。
 */
const raceCanaryDeviceLost = <T>(
  device: GPUDevice,
  work: Promise<T>,
  where: string,
): Promise<T> =>
  Promise.race([
    work,
    device.lost.then((): never => {
      throw new GpuDeviceLostError(
        `shader-f16 カナリアの${where}中に device が失われた（再構築が必要）`,
      );
    }),
  ]);

/**
 * カナリア本体（{@link SHADER_F16_CANARY_WGSL} の実走と突合）。
 *
 * NOTE: `export` はテストのため（不一致経路と消失経路は実 GPU では作れない）。公開面は
 * mod.ts の明示列挙なので、ここでの export は API 面を広げない（ADR 0008）。
 */
export const assertShaderF16Executes = async (device: GPUDevice): Promise<void> => {
  const byteLength = SHADER_F16_CANARY_EXPECTED.length * 4;
  const out = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const pipeline = await raceCanaryDeviceLost(
      device,
      withValidationScope(
        device,
        "shader-f16 カナリア",
        () =>
          device.createComputePipeline({
            layout: "auto",
            compute: {
              module: device.createShaderModule({ code: SHADER_F16_CANARY_WGSL }),
              entryPoint: "main",
            },
          }),
      ),
      "パイプライン生成",
    );
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: out } }],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(out, 0, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await raceCanaryDeviceLost(device, staging.mapAsync(GPUMapMode.READ), "読み戻し");
    const observed = [...new Float32Array(staging.getMappedRange().slice(0))];
    staging.unmap();
    const matches = SHADER_F16_CANARY_EXPECTED.every((value, index) => observed[index] === value);
    if (!matches) {
      throw new GpuFeatureError(
        `'${SHADER_F16_FEATURE}' は列挙されているが、f16 カーネルの実走が既知解と一致しない` +
          `（期待 [${SHADER_F16_CANARY_EXPECTED.join(", ")}] / 実測 [${observed.join(", ")}]）。` +
          "この環境の f16 実装は信用できない（全 0 なら denoland/deno#23125 と同じ形の沈黙故障、" +
          "1 要素目だけ違うなら共有 f16 タイル経由の丸めが起きていない）。shaderF16 を省略して" +
          "既定の f32 経路で実行すること",
      );
    }
  } finally {
    staging.destroy();
    out.destroy();
  }
};

/**
 * MUST: `adapter.info` は実装差のある面。仕様上は必須プロパティだが、`requestAdapterInfo()`
 * 時代の古い Chromium では存在せず undefined になる。型が必須と宣言している以上、実装差は
 * 読む側で吸収するしかないため、参照はこの関数 1 箇所に閉じ込めて空値に正規化する
 * （{@link GpuContext.adapterInfo} は常に安全に読めることを保証する）。
 */
type AdapterInfoHost = { readonly info?: GPUAdapterInfo };

const EMPTY_ADAPTER_INFO: GPUAdapterInfo = {
  vendor: "",
  architecture: "",
  device: "",
  description: "",
  subgroupMinSize: 0,
  subgroupMaxSize: 0,
  isFallbackAdapter: false,
};

export const readAdapterInfo = (adapter: AdapterInfoHost): GPUAdapterInfo =>
  adapter.info ?? EMPTY_ADAPTER_INFO;

/**
 * MUST: `navigator.gpu.wgslLanguageFeatures` は実装差のある面（型定義に無い実装・未提供の
 * 実装がある）。直接参照はこの関数 1 箇所に閉じ込め、欠落時は空集合に縮退する。
 * 参考情報であり機能検出には使わない（ハードウェア対応と無関係に列挙されるため）。
 * 唯一の実用は**数値が同一な変種の選択**（w8a8 の dot4I8Packed / エミュ — ADR 0025。
 * 誤った選択でも結果が 1 ビットも変わらない場合に限り、速度の分岐に使ってよい）。
 */
type WgslLanguageFeatureHost = GPU & { readonly wgslLanguageFeatures?: Iterable<string> };

const readWgslLanguageFeatures = (gpu: GPU): ReadonlySet<string> => {
  const host: WgslLanguageFeatureHost = gpu;
  const features = host.wgslLanguageFeatures;
  return new Set(features ?? []);
};

export type DeviceLostHandler = (info: GPUDeviceLostInfo) => void;

export type AcquireGpuOptions = {
  /** requestAdapter にそのまま渡す（powerPreference 等）。 */
  readonly adapter?: GPURequestAdapterOptions;
  /** 予期しない device 消失の通知先。`destroy()` 由来の消失では呼ばれない。 */
  readonly onDeviceLost?: DeviceLostHandler;
  /**
   * GPU 側時間計測（op 別内訳 — ADR 0021）の要求。**三値**で、feature は device 作成時にしか
   * 要求できないためここが唯一の opt-in 点になる:
   *
   * - `undefined`（既定）= **要求しない**。自動判定にはしない — 計測が有効な device では
   *   1 dispatch = 1 pass に開くので、アダプタの能力で**壁時計が変わる**（実測は
   *   {@link planTimestampFeature} の doc）。診断は明示的に払うもの、が既定。
   * - `true` = 必須。持たないアダプタでは {@link GpuFeatureError}。有効になったことは
   *   {@link GpuContext.gpuTimingEnabled} と `Session.diagnostics().lastRunTiming` で観測する。
   * - `false` = 明示的に要求しない（`undefined` と同じ挙動。計測の対照実行で「切ってある」を
   *   表明するためにある）。
   */
  readonly gpuTiming?: boolean;
  /**
   * f16 **計算**変種（ADR 0028）の要求。三値の形も既定の意味も
   * {@link AcquireGpuOptions.gpuTiming} と揃えてある（どちらも `undefined` = 要求しない）:
   *
   * - `undefined`（既定）= **要求しない**。自動判定にはしない — f16 計算は数値を変えるので、
   *   アダプタの能力で有効・無効が決まると「機械を替えたら黙って出力が変わる」ことになる。
   * - `true` = 必須。持たないアダプタでは {@link GpuFeatureError}。有効化できた場合は
   *   **既知解を返す極小 f16 カーネルを 1 dispatch 実走して突合する**（feature の列挙は
   *   「動く」の証拠にならない — denoland/deno#23125）。
   * - `false` = 明示的に要求しない（`undefined` と同じ挙動。意図の表明のためだけにある）。
   *
   * 有効化できたかどうかは {@link GpuContext.shaderF16Enabled} で観測する。
   * `SessionOptions.attentionCompute` / `linearCompute` に `"f16"` を渡すには、ここが `true`
   * である必要がある（Session 構築時に fail loudly）。
   */
  readonly shaderF16?: boolean;
};

/**
 * 取得済み device と正規化済み能力の束。
 *
 * `device.lost` の購読はコンストラクタで**無条件に、かつ 1 回だけ**行う。消失を未処理の
 * まま放置すると `mapAsync` / `onSubmittedWorkDone` が永久に解決せず、失敗ではなくハングと
 * して現れる。購読を任意にしないことで、消失は必ず {@link GpuContext.lost} に記録され、
 * この層の待機は例外に変換される。
 *
 * ## errorScope 区間の不変条件（device 単位・この層が守らせる）
 *
 * errorScope は **device 単位の LIFO スタック**で、`popErrorScope()` は呼んだ時点のスタック
 * 先頭を無条件に取る。つまり「誰のスコープか」という概念が無い。ここから 3 つの規則が出る:
 *
 * - MUST: `await` を跨いで errorScope を張る区間は {@link GpuContext.withScopeLock} の中で
 *   実行する。重なると①自分のエラーが他人のスコープに入り②自分の pop が他人のスコープを
 *   取るため、失敗が無関係な呼び出しに帰属し、本来落ちるべき呼び出しは沈黙のまま全 0 を返す。
 * - MUST NOT: `withScopeLock` の中で `withScopeLock` を再取得する（自己デッドロック）。
 *   ロックは再入可能ではない。検出器は置かない — 「保持中」フラグでは**正当な待ち行列**
 *   （別 Session が先行区間の完了を待って並ぶ形）と再入が区別できず、区別するには async
 *   呼び出しを跨ぐ実行コンテキスト追跡が要る。Web 標準 API のみという制約（ADR 0002）の下で
 *   純粋な手段が無いため、取得点を executor の 1 箇所に限定し、その内側の層（PipelineCache /
 *   SubmitScheduler / RunArena）は同期区間で完結するスコープしか使わない、という層規約で守る。
 * - 同期区間だけで完結するスコープ（{@link withValidationScope} /
 *   {@link pushFailureScopes}〜{@link popFailureScopes}）はロック不要。他のタスクが割り込む
 *   余地が無く、LIFO の入れ子が必ず均衡するため。ロック内から呼ばれる層（PipelineCache 等）
 *   はこの形でなければならない。
 */
export class GpuContext {
  readonly device: GPUDevice;
  readonly adapterInfo: GPUAdapterInfo;
  /** device が実際に有効化した feature（アダプタが持つだけの feature は含まない）。 */
  readonly features: ReadonlySet<string>;
  /** 参考情報。未実装環境では空集合。機能検出には使わない。 */
  readonly wgslLanguageFeatures: ReadonlySet<string>;
  readonly limits: RequiredLimits;
  #lost: GPUDeviceLostInfo | undefined;
  #destroyRequested = false;
  readonly #lostListeners = new Set<DeviceLostHandler>();
  /**
   * errorScope 区間の直列化チェーン。決着（成功・失敗）だけを次に渡すため自身は決して
   * reject しない（1 本の失敗で以後の全区間を巻き添えにしない）。
   */
  #scopeChain: Promise<void> = Promise.resolve();

  constructor(
    device: GPUDevice,
    adapterInfo: GPUAdapterInfo,
    limits: RequiredLimits,
    wgslLanguageFeatures: ReadonlySet<string>,
    onDeviceLost?: DeviceLostHandler,
  ) {
    this.device = device;
    this.adapterInfo = adapterInfo;
    this.limits = limits;
    this.features = new Set(device.features);
    this.wgslLanguageFeatures = wgslLanguageFeatures;
    if (onDeviceLost !== undefined) {
      this.onLost((info) => {
        if (!this.#destroyRequested) {
          onDeviceLost(info);
        }
      });
    }
    void device.lost.then((info) => {
      this.#lost = info;
      // 通知中の解除で反復が壊れないよう複製してから呼ぶ。消失は 1 度きりなので、通知後は
      // 購読を空にして以後の onLost を即時通知経路に落とす。
      const listeners = [...this.#lostListeners];
      this.#lostListeners.clear();
      for (const listener of listeners) {
        // MUST: listener の例外は通知の fan-out を止めない。購読は上で clear 済みで再通知が
        // 無いため、公開 onDeviceLost（挿入順で先）の throw で後続の内部購読
        // （{@link GpuContext.raceDeviceLost}）へ通知が届かないと、消失が例外ではなくハングに
        // なる。捕えた例外は握り潰さず、消失の制御経路から切り離して非同期に再 throw する。
        try {
          listener(info);
        } catch (cause) {
          queueMicrotask(() => {
            throw cause;
          });
        }
      }
    });
  }

  /** 消失済みならその情報。未消失は undefined。`destroy()` 後も記録される。 */
  get lost(): GPUDeviceLostInfo | undefined {
    return this.#lost;
  }

  /** 意図的な破棄を要求済みか（予期しない消失と区別するため）。 */
  get destroyRequested(): boolean {
    return this.#destroyRequested;
  }

  /**
   * GPU 側時間計測が有効か（ADR 0021）。
   *
   * MUST: 要求値を別フィールドに複製せず、**実際に有効化された feature** から導く。
   * device の feature 集合は要求したものそのものなので、複製すると「要求したのに載らなかった」
   * ときに診断だけが有効を主張する形になる（内訳が空のまま「計測中」に見える）。
   */
  get gpuTimingEnabled(): boolean {
    return this.features.has(TIMESTAMP_QUERY_FEATURE);
  }

  /**
   * f16 計算変種が使えるか（ADR 0028）。
   *
   * MUST: {@link GpuContext.gpuTimingEnabled} と同じ規律で、要求値の複製ではなく**実際に
   * 有効化された feature** から導く。ここが true を返すのは `acquireGpu({shaderF16: true})`
   * が feature の要求とカナリアの実走の両方を通ったときだけ。
   */
  get shaderF16Enabled(): boolean {
    return this.features.has(SHADER_F16_FEATURE);
  }

  /**
   * 未解除の消失購読の本数（診断）。
   *
   * {@link GpuContext.raceDeviceLost} は決着時に必ず解除するので、待機が全て決着していれば
   * 0 に戻る。0 に戻らないことが「flush / readback ごとに reaction が積み残る」リークの姿で、
   * 挙動からは観測できない（ハングも誤値も起こさず、長寿命 Session で単調増加するだけ）。
   * 見えない残留を正直に数値で出すためだけの面。
   */
  get pendingLostListeners(): number {
    return this.#lostListeners.size;
  }

  /**
   * device 消失を購読する。戻り値の関数で解除する。購読時点で既に消失していれば**即時
   * （同期）**に通知して取りこぼしを作らない。
   *
   * MUST: 待機のたびに `device.lost.then(...)` を新しく張らない。`lost` は device の寿命の間
   * 未解決のままなので、`.then` は解除手段の無い reaction として promise に積まれ続け、
   * flush / readback ごとに単調増加する（長寿命 Session でのリーク）。購読はここに一本化し、
   * 待機側は {@link GpuContext.raceDeviceLost} を使って決着時に必ず解除する。
   */
  onLost(listener: DeviceLostHandler): () => void {
    const info = this.#lost;
    if (info !== undefined) {
      listener(info);
      return () => {};
    }
    this.#lostListeners.add(listener);
    return () => {
      this.#lostListeners.delete(listener);
    };
  }

  /**
   * `work` の決着と device 消失を競わせ、消失が先なら {@link GpuDeviceLostError} にする。
   *
   * MUST: device 消失後 `onSubmittedWorkDone` / `mapAsync` は解決しない。競わせないと失敗
   * ではなくハングになる。決着時は購読を必ず解除する（reaction を積み残さない）。
   */
  async raceDeviceLost<T>(work: Promise<T>, where: string): Promise<T> {
    let unsubscribe: () => void = () => {};
    const lost = new Promise<never>((_resolve, reject) => {
      unsubscribe = this.onLost(() => {
        reject(new GpuDeviceLostError(`${where} 中に device が失われた（再構築が必要）`));
      });
    });
    try {
      return await Promise.race([work, lost]);
    } finally {
      unsubscribe();
    }
  }

  /**
   * `await` を跨いで errorScope を張る区間を、device 単位で直列化して実行する。
   * 規則の全体はクラス冒頭「errorScope 区間の不変条件」を参照。
   *
   * トレードオフ: 保持区間は run 1 本の GPU 操作全体（エンコード〜`flush()` の完了待ち〜
   * readback〜アリーナ破棄）に及ぶため、同一 device 上の複数 Session の run は丸ごと
   * 直列化される。区間を狭めない理由は「run が GPU 操作を発行するのは自分のロック区間内
   * のみ」という不変条件を単純に保つため（executor の #runOnce を参照）。それでもこの設計を
   * 採るのは、誤帰属の帰結が「無関係な run が落ち、本来落ちるべき run が全 0 を静かに返す」
   * ことだから — 沈黙した誤値は検出手段が無く、失うスループットとは釣り合わない。
   */
  withScopeLock<T>(body: () => Promise<T>): Promise<T> {
    const result = this.#scopeChain.then(body);
    this.#scopeChain = result.then(() => undefined, () => undefined);
    return result;
  }

  /**
   * device を破棄する。VRAM を返すのはこの経路のみ。
   *
   * MUST: 未 submit のエンコードと生存中のバッファを持つ層（RunArena）を先に flush /
   * destroy してから呼ぶこと。破棄後の続行は `acquireGpu()` での再構築になる。
   */
  destroy(): void {
    this.#destroyRequested = true;
    this.device.destroy();
  }
}

/**
 * アダプタ取得 → limits 引き上げ → device 取得 → 取得結果の検証、までを行う。
 * 途中の失敗は全て例外（黙って能力を落とした device を返さない）。
 */
export const acquireGpu = async (options: AcquireGpuOptions = {}): Promise<GpuContext> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) {
    throw new GpuUnavailableError("navigator.gpu が存在しない（WebGPU 非対応環境）");
  }
  const adapter = await gpu.requestAdapter(options.adapter);
  if (adapter === null) {
    throw new GpuUnavailableError("GPUAdapter を取得できない（対応 GPU / ドライバが無い）");
  }
  const limits = planRequiredLimits(adapter.limits);
  // 条件付き feature の判定はここだけ（不足は例外 — 黙って能力を落とさない）。ADR 0021 / 0028。
  const timestampQuery = planTimestampFeature(adapter.features, options.gpuTiming);
  const shaderF16 = planShaderF16Feature(adapter.features, options.shaderF16);
  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...REQUIRED_FEATURES,
      ...(timestampQuery ? [TIMESTAMP_QUERY_FEATURE] : []),
      ...(shaderF16 ? [SHADER_F16_FEATURE] : []),
    ],
    requiredLimits: limits,
  });
  try {
    assertLimitsGranted(limits, device.limits);
    // MUST: 列挙ではなく実走で確かめる（denoland/deno#23125 の沈黙全 0）。
    if (shaderF16) await assertShaderF16Executes(device);
  } catch (cause) {
    device.destroy();
    throw cause;
  }
  return new GpuContext(
    device,
    readAdapterInfo(adapter),
    limits,
    readWgslLanguageFeatures(gpu),
    options.onDeviceLost,
  );
};

/**
 * validation errorScope を張って `body` を実行し、捕捉したエラーを例外に変換する。
 *
 * MUST: パイプライン生成経路には常設する。無効なシェーダ / パイプラインは同期例外にならず、
 * 生成は成功したように見えて dispatch が no-op になり、出力が全て 0 のまま処理が続く。
 * MUST NOT: `body` の中で非同期処理を待たない。errorScope はスタックで、pop は body の同期
 * 実行直後に起きるため、await 後にエンコードした操作は別のスコープに吸われる。
 */
export const withValidationScope = async <T>(
  device: GPUDevice,
  label: string,
  body: () => T,
): Promise<T> => {
  device.pushErrorScope("validation");
  let value: T;
  try {
    value = body();
  } catch (cause) {
    // MUST: body が throw してもスコープは必ず pop する。積み残すと後続の検証結果が
    // 誤ったスコープに吸われ、以後のエラーが恒久的に見えなくなる。pop 自体の失敗は握り潰す
    // （後始末で本体の例外を上書きしない — discardFailureScopes と同じ規律）。
    await device.popErrorScope().catch(() => null);
    throw cause;
  }
  const error = await device.popErrorScope();
  if (error !== null) {
    throw new GpuValidationError(`${label}: ${error.message}`);
  }
  return value;
};

/**
 * out-of-memory + validation の 2 本を張る。確保を伴う区間はこの両建てで囲む。
 *
 * 両方が要るのは失敗の出方が違うため — 上限超過の `createBuffer` は validation、device の
 * 余力切れは out-of-memory で、どちらも同期例外にはならず**無効なバッファが返るだけ**。
 * 囲まないと、そこへの `writeBuffer` が警告すら出さない no-op になり、空の重みや空の中間
 * バッファのまま処理が続く。対応する pop は必ず {@link popFailureScopes} /
 * {@link discardFailureScopes} を使う（pop の順序と同期性がここに閉じている）。
 */
export const pushFailureScopes = (device: GPUDevice): void => {
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
};

/**
 * {@link pushFailureScopes} の 2 本を pop し、捕捉した失敗を型付き例外にして返す
 * （何も捕捉しなければ undefined）。
 *
 * MUST: 2 本の pop は**同一同期区間で発行**する（await するのは発行済みの promise だけ）。
 * pop はスタック先頭を無条件に取るため、発行の間に await を挟むと、その隙に他所が push した
 * スコープを 2 本目が取り、失敗が誤帰属する。
 *
 * MUST: 両方が捕捉されたときは out-of-memory を返す。確保が余力切れで失敗すると `createBuffer`
 * は無効なバッファを返し、それを使う後続の `createBindGroup` / `writeBuffer` が
 * `Buffer with '' label is invalid` という**派生の** validation エラーを立てる。validation を
 * 先に返すと根因の OOM が捨てられ、区別のつかない「無効バッファ」として報告される
 * （docs/research/2026-08-08-vram-oom-misreport.md）。
 */
export const popFailureScopes = async (
  device: GPUDevice,
  label: string,
): Promise<GpuValidationError | GpuOutOfMemoryError | undefined> => {
  const validation = device.popErrorScope();
  const outOfMemory = device.popErrorScope();
  const [validationError, outOfMemoryError] = await Promise.all([validation, outOfMemory]);
  if (outOfMemoryError !== null) {
    return new GpuOutOfMemoryError(`${label}: ${outOfMemoryError.message}`);
  }
  if (validationError !== null) {
    return new GpuValidationError(`${label}: ${validationError.message}`);
  }
  return undefined;
};

/**
 * 失敗経路の後始末。{@link pushFailureScopes} の 2 本を pop して結果を捨てる。
 *
 * MUST: 本体が例外で抜けてもスコープは必ず 2 本とも pop する。積み残すと以後の検証結果が
 * 誤ったスコープに吸われ、エラーが恒久的に見えなくなる。pop 自体の失敗も握り潰す
 * （後始末で本体の例外を上書きしない）。
 */
export const discardFailureScopes = async (device: GPUDevice): Promise<void> => {
  const validation = device.popErrorScope().catch(() => null);
  const outOfMemory = device.popErrorScope().catch(() => null);
  await Promise.all([validation, outOfMemory]);
};
