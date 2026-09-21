/**
 * device を取って {@link GpuContext} を組み立てるまで（アダプタ取得 → limits の計画と検査 →
 * feature の計画とカナリアの実走 → adapterInfo / WGSL 言語機能の正規化）。
 *
 * MUST: 依存は acquire → context の一方向。{@link GpuContext} を構築するのはこのファイルの
 * {@link acquireGpu} だけで、器（`context.ts`）側からここへ実体を import しない。層の入口は
 * 2 ファイルの公開名を再 export する `device.ts`。
 */

import {
  describeDeviceLoss,
  GpuContext,
  GpuDeviceLostError,
  SHADER_F16_FEATURE,
  TIMESTAMP_QUERY_FEATURE,
} from "./context.ts";
import { withPipelineScope } from "./error-scope.ts";
import { BUFFER_USAGE, MAP_MODE } from "./webgpu-constants.ts";

/** navigator.gpu が無い / アダプタを取得できない。 */
export class GpuUnavailableError extends Error {
  override readonly name = "GpuUnavailableError";
}

/** 要求した limit を device が満たさない（要求漏れ・仕様既定値への降格の検出）。 */
export class GpuLimitError extends Error {
  override readonly name = "GpuLimitError";
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
 * 出力が全て 0 になる）。この沈黙故障は {@link withPipelineScope} でのみ可視化できる。
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

type RequiredLimitKey = (typeof REQUIRED_LIMIT_KEYS)[number];

export type RequiredLimits = Readonly<Record<RequiredLimitKey, number>>;

/**
 * requiredLimits の**上限を絞る**指定（キーごとに `min` を取る）。
 *
 * MUST: 絞る向きにしか使えない（アダプタ値を超える要求はアダプタ値のまま）。引き上げに使えて
 * しまうと requestDevice が丸ごと失敗する経路が黙って生まれる。
 */
type LimitCaps = Partial<Record<RequiredLimitKey, number>>;

/**
 * アダプタ実測値から requestDevice に渡す requiredLimits を組み立てる。
 * 要求値がアダプタ値そのものなら requestDevice は必ず成功する。
 *
 * MUST: 仕様上 `maxStorageBufferBindingSize ≤ maxBufferSize`。片方だけ引き上げると
 * requestDevice が丸ごと失敗するため、両者は必ず同時に計画する。
 *
 * `caps` は**テスト専用の絞り**（{@link LIMIT_CAPS}）。要求値を下げるだけだが、
 * `maxBufferSize` を絞る cap は `maxStorageBufferBindingSize` の clamp 元そのものを動かす。
 * MUST: `maxBufferSize` には cap を**先に**適用し、その値で束縛上限を clamp する。後から
 * 掛けると `maxBufferSize` だけを絞った caps で計画値が
 * `maxStorageBufferBindingSize > maxBufferSize` になり、「core 既定機の再現」という
 * この面の目的が黙って外れる（束縛上限が絞られないままバッファ上限だけ下がる）。
 * 残りのキーは互いに独立なので、cap の適用順に依らない。
 */
export const planRequiredLimits = (
  adapterLimits: GPUSupportedLimits,
  caps: LimitCaps = {},
): RequiredLimits => {
  const maxBufferSize = Math.min(adapterLimits.maxBufferSize, caps.maxBufferSize ?? Infinity);
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
  for (const key of REQUIRED_LIMIT_KEYS) {
    const cap = caps[key];
    if (cap !== undefined) planned[key] = Math.min(planned[key], cap);
  }
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
 * {@link AcquireGpuOptions.shaderF16}）と32レーンsubgroup（ADR 0100）。有効なfeatureの照会は
 * {@link GpuContext.features} で行える。
 */
const REQUIRED_FEATURES: readonly GPUFeatureName[] = [];

/**
 * feature 集合として読む面だけを取り出した形。`adapter.features`
 * （`ReadonlySet<GPUFeatureName>`）と素の `Set<string>` の**両方**を受けるために要る
 * （前者は要素型が狭く、後者は `ReadonlySet<GPUFeatureName>` を満たさない）。
 */
type GpuFeatureSet = Iterable<string> & { has(feature: string): boolean };

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
 * MUST: カナリアの待ちは 1 つ残らずここを通す。ただし根拠は「消失後の待ちが必ずハングする」
 * ことではない —— 実測（Deno + wgpu / NVIDIA・2026-08-16）では `device.destroy()` 由来の消失後、
 * `popErrorScope` は **null で resolve**（空スタックでも同じ）し、`mapAsync` は `OperationError`
 * で reject し、消失前に発行して in-flight だった待ちも全て決着した
 * （`docs/research/2026-08-01-m0-review.md` の実測記録と同じ側）。競わせ続けるのは
 * **destroy 以外の消失（実 TDR / ドライバリセット / ブラウザ実装）が未検証**だからで、
 * ここは「解決しない実装に当たったとき `acquireGpu()` がハングで返ってこない」ための保険。
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
  feature = "shader-f16",
): Promise<T> =>
  Promise.race([
    work,
    device.lost.then((info): never => {
      throw new GpuDeviceLostError(
        `${feature} カナリアの${where}中に device が失われた（再構築が必要）${
          describeDeviceLoss(info)
        }`,
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
    usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
  });
  const staging = device.createBuffer({
    size: byteLength,
    usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
  });
  try {
    const pipeline = await raceCanaryDeviceLost(
      device,
      withPipelineScope(
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
    await raceCanaryDeviceLost(device, staging.mapAsync(MAP_MODE.READ), "読み戻し");
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
 * 単独ではGPU対応の証明にしない（ハードウェア対応と無関係に列挙されるため）。
 * 自動選択に使えるのは**数値が同一な変種の選択**（w8a8 の dot4I8Packed / エミュ — ADR 0025。
 * 誤った選択でも結果が 1 ビットも変わらない場合に限り、速度の分岐に使ってよい）。
 * subgroup_idは明示要求時の必要条件としてのみ検査し、device featureと実走も要求する。
 */
type WgslLanguageFeatureHost = GPU & { readonly wgslLanguageFeatures?: Iterable<string> };

const readWgslLanguageFeatures = (gpu: GPU): ReadonlySet<string> => {
  const host: WgslLanguageFeatureHost = gpu;
  const features = host.wgslLanguageFeatures;
  return new Set(features ?? []);
};

/** 実行に必要な3機能を明示要求時だけ検査する。自動有効化・黙った縮退はしない。 */
export const planSubgroup32Features = (
  adapterFeatures: GpuFeatureSet,
  languageFeatures: ReadonlySet<string>,
  requested: boolean | undefined,
): readonly string[] => {
  if (requested !== undefined && typeof requested !== "boolean") {
    throw new GpuFeatureError("subgroups はbooleanでなければならない");
  }
  if (requested !== true) return [];
  for (const feature of ["subgroups", "subgroup-size-control"]) {
    if (!adapterFeatures.has(feature)) {
      throw new GpuFeatureError(`subgroups: true を指定したが、アダプタが '${feature}' を持たない`);
    }
  }
  if (!languageFeatures.has("subgroup_id")) {
    throw new GpuFeatureError("subgroups: true はWGSL言語機能 'subgroup_id' が必要");
  }
  return ["subgroups", "subgroup-size-control"];
};

/** 256レーン全てで32レーン縮約・8部分和・broadcastの既知解を実走する（ADR 0100）。 */
export const assertSubgroup32Executes = async (device: GPUDevice): Promise<void> => {
  const code = `enable subgroups, subgroup_size_control;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
var<workgroup> partial: array<f32, 8>;
@compute @workgroup_size(256) @subgroup_size(32)
fn main(@builtin(local_invocation_index) lid: u32,
  @builtin(subgroup_invocation_id) sub: u32, @builtin(subgroup_id) sg: u32,
  @builtin(num_subgroups) count: u32, @builtin(subgroup_size) size: u32) {
  let value = subgroupAdd(f32(lid + 1u));
  if (subgroupElect()) { partial[sg] = value; }
  workgroupBarrier();
  var across = 0.0;
  if (sub < count) { across = partial[sub]; }
  let sum = subgroupAdd(across);
  out[lid] = select(-1.0, sum, size == 32u && count == 8u);
}`;
  const out = device.createBuffer({
    size: 1024,
    usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
  });
  const staging = device.createBuffer({
    size: 1024,
    usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
  });
  try {
    await raceCanaryDeviceLost(
      device,
      withPipelineScope(device, "subgroups カナリア", () => {
        const pipeline = device.createComputePipeline({
          layout: "auto",
          compute: {
            module: device.createShaderModule({ code }),
            entryPoint: "main",
          },
        });
        const bindings = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: out } },
          ],
        });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(1);
        pass.end();
        encoder.copyBufferToBuffer(out, 0, staging, 0, 1024);
        device.queue.submit([encoder.finish()]);
      }),
      "実行",
      "subgroups",
    );
    await raceCanaryDeviceLost(device, staging.mapAsync(MAP_MODE.READ), "読み戻し", "subgroups");
    const observed = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    if (observed.length !== 256 || observed.some((value) => value !== 32896)) {
      throw new GpuFeatureError("subgroups カナリア: 32レーン縮約の既知解と一致しない");
    }
  } finally {
    // コマンドは読み戻しより先にsubmit済み。失敗時も未投入の参照を残さない。
    staging.destroy();
    out.destroy();
  }
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
  /**
   * 32レーンsubgroupの明示要求（既定false、ADR 0100）。subgroupsとsubgroup-size-control、
   * WGSLのsubgroup_idが必要。取得時に既知解の実走を検査し、不足・不一致は拒否する。
   * 能力を取得するだけでは計算経路を変えない。SessionのrmsNormReduceは別途指定する。
   */
  readonly subgroups?: boolean;
  /** テスト専用（{@link LIMIT_CAPS}）。requiredLimits を**絞る**方向にだけ効く。 */
  readonly [LIMIT_CAPS]?: LimitCaps;
};

/**
 * **テスト専用の非公開面**（mod.ts からは輸出しない — executor の `I8A8_DOT` と同じ流儀）。
 *
 * requiredLimits を絞った device を作る。ポータビリティの門はこれでしか張れない — 手元の
 * アダプタは WebGPU core 既定（`maxStorageBufferBindingSize` = 128MiB）より遥かに大きい値を
 * 出すので、「core 既定の機で確保・束縛が通るか」は**絞った device 上で実走**する以外に
 * 確かめる手段が無い（列挙は「動く」の証拠にならない、と同じ規律）。
 */
export const LIMIT_CAPS: unique symbol = Symbol("karume.limitCaps");

/**
 * `navigator.gpu` とアダプタを取る（{@link acquireGpu} と {@link readAdapterLimits} の共通部）。
 * 無い環境は {@link GpuUnavailableError} — 2 つの入口で診断文言を揃えるためにここ 1 本にする。
 */
const requestAdapterOrThrow = async (
  options: GPURequestAdapterOptions | undefined,
): Promise<{ readonly gpu: GPU; readonly adapter: GPUAdapter }> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) {
    throw new GpuUnavailableError("navigator.gpu が存在しない（WebGPU 非対応環境）");
  }
  const adapter = await gpu.requestAdapter(options);
  if (adapter === null) {
    throw new GpuUnavailableError("GPUAdapter を取得できない（対応 GPU / ドライバが無い）");
  }
  return { gpu, adapter };
};

/**
 * アダプタ取得 → limits 引き上げ → device 取得 → 取得結果の検証、までを行う。
 * 途中の失敗は全て例外（黙って能力を落とした device を返さない）。
 */
export const acquireGpu = async (options: AcquireGpuOptions = {}): Promise<GpuContext> => {
  const { gpu, adapter } = await requestAdapterOrThrow(options.adapter);
  const limits = planRequiredLimits(adapter.limits, options[LIMIT_CAPS]);
  // 条件付き feature の判定はここだけ（不足は例外 — 黙って能力を落とさない）。ADR 0021 / 0028。
  const timestampQuery = planTimestampFeature(adapter.features, options.gpuTiming);
  const shaderF16 = planShaderF16Feature(adapter.features, options.shaderF16);
  const languageFeatures = readWgslLanguageFeatures(gpu);
  const subgroupFeatures = planSubgroup32Features(
    adapter.features,
    languageFeatures,
    options.subgroups,
  );
  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...REQUIRED_FEATURES,
      ...(timestampQuery ? [TIMESTAMP_QUERY_FEATURE] : []),
      ...(shaderF16 ? [SHADER_F16_FEATURE] : []),
      // Denoの型定義に未収録のWebGPU機能。上で実機の広告を検査した境界に限定する。
      ...subgroupFeatures as readonly GPUFeatureName[],
    ],
    requiredLimits: limits,
  });
  try {
    assertLimitsGranted(limits, device.limits);
    // MUST: 列挙ではなく実走で確かめる（denoland/deno#23125 の沈黙全 0）。
    if (shaderF16) await assertShaderF16Executes(device);
    if (subgroupFeatures.length > 0) await assertSubgroup32Executes(device);
  } catch (cause) {
    device.destroy();
    throw cause;
  }
  return new GpuContext(
    device,
    readAdapterInfo(adapter),
    limits,
    languageFeatures,
    options.onDeviceLost,
  );
};

/**
 * アダプタの limits だけを読む（device は作らない）。
 *
 * 配布形が宣言する `requiredLimits`（manifest の quant 欄）を**重みを落とす前**に突き合わせる
 * ための入口（ADR 0089 決定 5）。戻りは {@link acquireGpu} が同じ引数で計画する limits と同じ式
 * （{@link planRequiredLimits}・絞り無し）なので、呼び手は取得後の `GpuContext.limits` と同じ
 * 物差しで比較できる。
 *
 * MUST: アダプタは読んだら捨てる（持ち回らない）。WebGPU のアダプタは「いつでも失効しうる」
 * （仕様 §4.2 — システム状態の変化が無くても数秒〜数分で失効してよい）うえ、失効後の
 * `requestDevice` は例外ではなく生まれた時点で lost な device を返す静かな失敗になる。device は
 * 従来どおり {@link acquireGpu} が直前に取り直したアダプタから作る。
 *
 * NOTE: `requestAdapter` を 2 回呼んで同じ物理アダプタが選ばれる保証は仕様に無い。ここで読む
 * limits は事前検査の材料で、最終の検査は Session 構築時の実バッファ検査
 * （`assertWeightsWithinLimits`）が担う。
 */
export const readAdapterLimits = async (
  options: Pick<AcquireGpuOptions, "adapter"> = {},
): Promise<RequiredLimits> => {
  const { adapter } = await requestAdapterOrThrow(options.adapter);
  return planRequiredLimits(adapter.limits);
};
