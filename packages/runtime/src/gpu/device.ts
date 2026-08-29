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
 *
 * この層はさらに 2 つ、**device と同じ寿命で GpuContext が所有する**器を持つ:
 * {@link ResidentTensor}（Session を跨いで共有できる第 4 の寿命クラス）と
 * {@link BatchScope}（フェンス 1 本で閉じる enqueue 区間）。どちらも errorScope 区間と
 * 消失レースの規律がそのまま効くため、GPU バッファの器でありながらここに置いてある。
 */

// MUST: 型だけを取る（実体を import すると device.ts → カナリア → kernels / reference の
// 依存が生まれ、「この層から kernels / reference への import を作らない」規律が崩れる —
// {@link GpuContextInternals.attentionI8a8Dot}）。`import type` は消去されるので、
// 実行時の import グラフは今までどおり一方向のまま。
import type { AttentionI8a8Decision } from "./attention-dp4a-canary.ts";
import { STORAGE_USAGE } from "./arena.ts";
import { BUFFER_USAGE, MAP_MODE } from "./webgpu-constants.ts";

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

/**
 * errorScope が捕捉した internal エラー（実装側の都合で操作が失敗した — シェーダが複雑すぎて
 * コンパイルできない等）。WGSL 自体は妥当なので validation とは分岐先が違う: 利用者にとって
 * 「記述が不正」ではなく「この環境ではこの形のカーネルが通らない」であり、別型で区別できないと
 * codegen のバグと環境の限界が同じ報告に潰れる。
 */
export class GpuInternalError extends Error {
  override readonly name = "GpuInternalError";
}

/** device が失われた状態での操作。待ち続ける代わりに必ずこれを投げる。 */
export class GpuDeviceLostError extends Error {
  override readonly name = "GpuDeviceLostError";
}

/** 常駐テンソル（{@link ResidentTensor}）の寿命規律の破れ（破棄後利用・参照中の破棄）。 */
export class ResidentTensorError extends Error {
  override readonly name = "ResidentTensorError";
}

/** バッチ区間（{@link BatchScope}）の使い方の破れ（決着後の enqueue・計測との併用など）。 */
export class BatchScopeError extends Error {
  override readonly name = "BatchScopeError";
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
 * `caps` は**テスト専用の絞り**（{@link LIMIT_CAPS}）。要求値を下げるだけなので
 * `maxStorageBufferBindingSize ≤ maxBufferSize` の関係は崩れない。
 */
export const planRequiredLimits = (
  adapterLimits: GPUSupportedLimits,
  caps: LimitCaps = {},
): RequiredLimits => {
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
 * {@link AcquireGpuOptions.shaderF16}）の 2 本だけ。実際に有効化された feature の照会は
 * {@link GpuContext.features} で行える。
 */
const REQUIRED_FEATURES: readonly GPUFeatureName[] = [];

/** GPU 側時間計測（pass 境界の timestamp）に要る feature — ADR 0021。 */
const TIMESTAMP_QUERY_FEATURE: GPUFeatureName = "timestamp-query";

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
 * **ランタイム内部だけが触る面**の鍵（mod.ts からは輸出しない — ADR 0008 の「薄い面」を
 * 汚さない）。{@link GpuContext} / {@link ResidentTensor} / {@link BatchScope} は利用者向けの
 * 数メソッドだけを素の名前で持ち、executor が要る実体（直列化プリミティブ・GPUBuffer・
 * 焼き込み参照計数・メンバ登録）はこの鍵の下に畳む。テスト専用ノブ（`I8A8_DOT`）と同じ流儀。
 *
 * MUST: 宣言は {@link GpuContext} より**前**に置く。クラス本体の計算プロパティ名は
 * クラス定義時に評価されるので、後ろに置くと TDZ で落ちる。
 */
export const RUNTIME_INTERNAL: unique symbol = Symbol("karume.runtimeInternal");

/** {@link GpuContext} のランタイム内部面（利用者ストーリーに対応しない直列化プリミティブ）。 */
type GpuContextInternals = {
  /**
   * `work` の決着と device 消失を競わせ、消失が先なら {@link GpuDeviceLostError} にする。
   *
   * MUST: 待ちは失敗ではなくハングになりうる（`onSubmittedWorkDone` / `mapAsync` が消失後に
   * 解決するかは実装差のある面 — {@link raceCanaryDeviceLost} の doc に実測を記録）。
   * 決着時は購読を必ず解除する（reaction を積み残さない）。
   */
  raceDeviceLost<T>(work: Promise<T>, where: string): Promise<T>;
  /**
   * `await` を跨いで errorScope を張る区間を、device 単位で直列化して実行する。
   * 規則の全体は {@link GpuContext} 冒頭「errorScope 区間の不変条件」を参照。
   *
   * トレードオフ: 保持区間は run 1 本の GPU 操作全体（エンコード〜`flush()` の完了待ち〜
   * readback〜アリーナ破棄）に及ぶため、同一 device 上の複数 Session の run は丸ごと
   * 直列化される。区間を狭めない理由は「run が GPU 操作を発行するのは自分のロック区間内
   * のみ」という不変条件を単純に保つため（executor の #runOnce を参照）。それでもこの設計を
   * 採るのは、誤帰属の帰結が「無関係な run が落ち、本来落ちるべき run が全 0 を静かに返す」
   * ことだから — 沈黙した誤値は検出手段が無く、失うスループットとは釣り合わない。
   *
   * MUST: 利用者の面に出さない。素の名前で公開すると
   * `gpu.withScopeLock(() => session.run(...))` が書けてしまい、run が同じロックを取りに行って
   * **診断も例外も出ないまま自己デッドロック**する（再入検出器は置けない — 下記 doc）。
   */
  withScopeLock<T>(body: () => Promise<T>): Promise<T>;
  /**
   * 融合 attention の整数内積変種を **device 単位で 1 度だけ**決める（遅延・メモ化）。
   *
   * `run` は判定の実体（{@link "./attention-dp4a-canary.ts"} の `decideAttentionI8a8Dot`）を
   * 呼び手が渡す形にしてある — この層から kernels / reference 層への import を作らないため
   * （逆向きの import は canary 側が張る）。MUST: 呼び出し点は Session 構築の 1 箇所だけ
   * （複数の実体を渡せる形にすると、メモが「最初に渡された判定」を意味するだけの席になる）。
   *
   * MUST: メモするのは **Promise そのもの**（値ではない）。attentionCompute "a8" の Session を
   * 並行構築すると、値でメモする形ではカナリアが 2 本走って device 単位 1 回の契約が崩れる。
   * 失敗（両腕とも sanity 帯を外した / device 消失）も同じ Promise のまま配る — device の性質は
   * 走らせ直しても変わらないので、再試行は同じ結論を得るためだけに 1 submit を払う。
   *
   * 席が持つのは変種 1 値ではなく**判定まるごと**（{@link AttentionI8a8Decision}）。「既知解と
   * 厳密一致ではなかったが帯内なので通した」という事実は判定と同じ寿命で、値に潰すと呼び手が
   * 警告を出せなくなる。
   */
  attentionI8a8Dot(run: () => Promise<AttentionI8a8Decision>): Promise<AttentionI8a8Decision>;
};

/**
 * device が使える状態かの同期判定（常駐テンソル経路の受付門 — ADR 0054）。
 *
 * MUST: `lost` だけでなく `destroyRequested` も見る。`destroy()` はフラグを同期に立てるのに
 * `device.lost` の reaction が走るのは以後のタスクなので、`lost` だけだとその窓で操作が通る。
 * 通したときの現れ方が沈黙なのがここを置く理由 — {@link ResidentTensor.write} は破棄済み
 * バッファへの**沈黙 no-op**（警告すら出ない）になり、{@link GpuContext.createResident} は
 * 消失後の `popErrorScope` が null で resolve する（`docs/research/2026-08-16-device-lost-wait-settlement.md`）
 * ため**無効なバッファを掴んだまま成功として返る**。どちらも loud になるのは次のフェンスで、
 * その間の診断は誤導的になる。
 * MUST: 型は {@link GpuDeviceLostError}（`runtime/generation-context.ts` の `assertDeviceUsable`
 * と同じ規律 — lost device 由来の GPU 資源は WebGPU 仕様上回復不能で、意図的な破棄と予期しない
 * 消失で復旧手段は変わらないので型は分けない）。
 */
const assertDeviceUsable = (gpu: GpuContext, where: string): void => {
  if (gpu.destroyRequested || gpu.lost !== undefined) {
    throw new GpuDeviceLostError(
      `${where}: device が失われた（device を取り直して作り直すこと）`,
    );
  }
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
 * - MUST: `await` を跨いで errorScope を張る区間は `withScopeLock`
 *   （{@link GpuContextInternals}）の中で実行する。重なると①自分のエラーが他人のスコープに
 *   入り②自分の pop が他人のスコープを取るため、失敗が無関係な呼び出しに帰属し、本来
 *   落ちるべき呼び出しは沈黙のまま全 0 を返す。
 * - MUST NOT: `withScopeLock` の中で `withScopeLock` を再取得する（自己デッドロック）。
 *   ロックは再入可能ではない。検出器は置かない — 「保持中」フラグでは**正当な待ち行列**
 *   （別 Session が先行区間の完了を待って並ぶ形）と再入が区別できず、区別するには async
 *   呼び出しを跨ぐ実行コンテキスト追跡が要る。Web 標準 API のみという制約（ADR 0002）の下で
 *   純粋な手段が無いため、取得点を executor の 1 箇所に限定し、その内側の層（PipelineCache /
 *   SubmitScheduler / RunArena）は同期区間で完結するスコープしか使わない、という層規約で守る。
 * - 同期区間だけで完結するスコープ（{@link withPipelineScope} /
 *   {@link pushFailureScopes}〜{@link popFailureScopes}）はロック不要。他のタスクが割り込む
 *   余地が無く、LIFO の入れ子が必ず均衡するため。ロック内から呼ばれる層（PipelineCache 等）
 *   はこの形でなければならない。
 */
export class GpuContext {
  /** ランタイム内部面（利用者が触る面ではない）。 */
  readonly [RUNTIME_INTERNAL]: GpuContextInternals;
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
  /**
   * 融合 attention の内積変種カナリアの結果（{@link GpuContextInternals.attentionI8a8Dot}）。
   * 未要求なら undefined のまま = 1 dispatch も出ない（a8 を使わない利用者はコストを払わない）。
   */
  #attentionI8a8Dot: Promise<AttentionI8a8Decision> | undefined;
  /**
   * {@link ResidentTensor} の識別子の発番。**モジュールスコープに置かない**（副作用ゼロの
   * 不変条件）— GpuContext ごとに別空間で足りる（resident は device を跨がない）。
   */
  #residentSeq = 0;

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
    this[RUNTIME_INTERNAL] = {
      raceDeviceLost: <T>(work: Promise<T>, where: string): Promise<T> =>
        this.#raceDeviceLost(work, where),
      withScopeLock: <T>(body: () => Promise<T>): Promise<T> => this.#withScopeLock(body),
      attentionI8a8Dot: (
        run: () => Promise<AttentionI8a8Decision>,
      ): Promise<AttentionI8a8Decision> => {
        this.#attentionI8a8Dot ??= run();
        return this.#attentionI8a8Dot;
      },
    };
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
        // （`raceDeviceLost`）へ通知が届かないと、消失が例外ではなくハングになる。捕えた例外は
        // 握り潰さず、消失の制御経路から切り離して非同期に再 throw する。
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
   * `raceDeviceLost`（{@link GpuContextInternals}）は決着時に必ず解除するので、待機が全て
   * 決着していれば 0 に戻る。0 に戻らないことが「flush / readback ごとに reaction が積み残る」
   * リークの姿で、挙動からは観測できない（ハングも誤値も起こさず、長寿命 Session で単調増加
   * するだけ）。
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
   * 待機側は `raceDeviceLost`（{@link GpuContextInternals}）を使って決着時に必ず解除する。
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

  /** {@link GpuContextInternals.raceDeviceLost} の実体（面は `RUNTIME_INTERNAL` の下だけ）。 */
  async #raceDeviceLost<T>(work: Promise<T>, where: string): Promise<T> {
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

  /** {@link GpuContextInternals.withScopeLock} の実体（面は `RUNTIME_INTERNAL` の下だけ）。 */
  #withScopeLock<T>(body: () => Promise<T>): Promise<T> {
    const result = this.#scopeChain.then(body);
    this.#scopeChain = result.then(() => undefined, () => undefined);
    return result;
  }

  /**
   * Session を跨いで共有できる常駐バッファを作る（**第 4 の寿命クラス** — 重みアリーナ /
   * slot backing / run アリーナのどれにも属さない）。
   *
   * 用途は「生成ループの間ずっと GPU に置いたままにしたい値」— 条件テンソル（ループ前に
   * {@link ResidentTensor.write} で 1 度だけ投入）と、ステップ間で受け渡す潜在
   * （{@link Session.enqueue} の `copyOutputs` で書き、次の enqueue の入力にする）。どちらも
   * ホストを 1 度も経由しないので、run 境界のフェンスを消せる。
   *
   * MUST: async なのは errorScope で囲むため。上限超過 / 余力切れの `createBuffer` は同期
   * 例外を投げず**無効なバッファを返す**ので、囲まないと以後の `writeBuffer` が警告すら
   * 出さない no-op になり、空の常駐テンソルのまま生成ループが回る。
   * MUST: `byteLength` は 4 の倍数（要素は全型 4 バイト — ADR 0009 の意味論 dtype）。
   * MUST: 消失済み device では受け付けない（{@link assertDeviceUsable} — errorScope は
   * 消失を捕らえないので、囲んでいても無効なバッファが成功として返る）。
   */
  async createResident(byteLength: number, label = "resident"): Promise<ResidentTensor> {
    assertDeviceUsable(this, `resident '${label}' の確保`);
    if (!Number.isInteger(byteLength) || byteLength <= 0 || byteLength % 4 !== 0) {
      throw new ResidentTensorError(
        `resident '${label}': byteLength は 4 の倍数の正の整数である必要がある: ${byteLength}`,
      );
    }
    // MUST: push から pop の発行までに await を挟まない（device 単位 LIFO の交錯を防ぐ根拠 —
    // クラス冒頭「errorScope 区間の不変条件」の 3 つ目）。
    pushFailureScopes(this.device);
    let buffer: GPUBuffer;
    try {
      buffer = this.device.createBuffer({ label, size: byteLength, usage: STORAGE_USAGE });
    } catch (cause) {
      await discardFailureScopes(this.device);
      throw cause;
    }
    const failure = await popFailureScopes(this.device, `resident '${label}' の確保`);
    if (failure !== undefined) {
      buffer.destroy();
      throw failure;
    }
    this.#residentSeq += 1;
    return new ResidentTensor(this, this.#residentSeq, buffer, byteLength, label);
  }

  /**
   * フェンス無しの enqueue を束ねる区間を開く（{@link Session.enqueue} — H-5）。
   *
   * 区間の間 **device 単位の errorScope 区間ロックを保持し続ける**
   * （`withScopeLock` — {@link GpuContextInternals}）。これが「batch の内側で出した GPU 操作は
   * 全て batch の errorScope に帰属する」ことと、「pop がスタック先頭を取り違えない」ことの根拠。
   *
   * MUST: 区間中に同一 device の {@link Session.run} / `dispose` を**発行しない**（await の
   * 有無に依らない）。run は区間ロックを取りに行くので {@link BatchScope.finish} まで決着せず、
   * その run が同一 Session の `enqueue` より前に居ると「finish → in-flight リース → enqueue 本体
   * → 先行 run → 区間ロック」の 4 辺が閉じて確定的な自己デッドロックになる（区間を開く**前**に
   * 発行した未 await の run も同じ — `beginBatch` はコンストラクタで同期にロックを先取りするので、
   * 同一 tick なら常に batch が先）。この形は `Session.enqueue` が {@link BatchScopeError} へ
   * 変換する。
   * NOTE: 閉路にならない形でも、**区間中に await する Session 操作全般**（`session.dispose()` /
   * `context.dispose()`）は先行に未決着 run があると `finish()` まで返らず、利用者からは
   * ハングに見える。batch 中は `enqueue` と `finish` だけを使うこと。**Session の構築も同じ**
   * — `attentionCompute: "a8"` の初回構築はカナリア（{@link GpuContextInternals.attentionI8a8Dot}）
   * で区間ロックを取りに行くため、区間中に構築すると `finish()` まで返らない。
   * MUST: 計測が有効な device では開けない。1 dispatch = 1 pass に開いた timestamp は
   * flush でしか回収されないため、batch の間 N run 分が未回収で溜まる（内訳を取るなら
   * 通常の run で計測すること — ADR 0021）。
   * NOTE: 同一 device で 2 本目を開こうとすると、1 本目が {@link BatchScope.finish} で
   * ロックを返すまでここで待つ（区間は device 単位で排他 — 入れ子にはならない）。
   */
  async beginBatch(): Promise<BatchScope> {
    if (this.gpuTimingEnabled) {
      throw new BatchScopeError(
        "gpuTiming が有効な device では batch を開けない（1 dispatch = 1 pass に開いた " +
          "timestamp が flush まで回収されず、batch の間 run 数ぶん溜まる）。" +
          "GPU 時間内訳は通常の run で計測すること",
      );
    }
    const batch = new BatchScope(this);
    await batch[RUNTIME_INTERNAL].entered;
    return batch;
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

/** {@link ResidentTensor} のランタイム内部面。 */
type ResidentInternals = {
  /** 焼き込み / 別名の対象になる実体。 */
  readonly buffer: GPUBuffer;
  /**
   * GpuContext 内で一意な識別子。**導出済み計画のキーと backing signature に載る**ので、
   * resident を差し替えれば別 signature（= 焼き直し）になり、戻せば元の backing に当たる。
   */
  readonly id: number;
  /**
   * この実体を確保した GpuContext。
   *
   * MUST: 束縛する側（executor）が「自分の device の resident か」を照合するために載せる。
   * {@link id} は GpuContext ごとの独立採番なので、別 context の同 id・同サイズな resident は
   * 導出済み計画のキーが衝突する — ヒット run（焼き込み済み backing）では渡された実体が
   * 一切参照されないため、**例外も警告も無く前の context の古い値を読む**。ミス経路だけは
   * device 不一致の validation で偶然落ちるが、キャッシュが当たった瞬間に検出が消える。
   * {@link BatchScope} が owner を検査しているのと同じ門をここにも置く。
   */
  readonly owner: GpuContext;
  /** 焼き込み bind group からの参照を 1 本積む（backing の構築時）。 */
  retainBaked(): void;
  /** 焼き込み参照を 1 本返す（backing の退役時）。 */
  releaseBaked(): void;
  /**
   * **進行中の run が入力として束ねた**ことを 1 本積む（焼き込み参照とは別枠）。
   *
   * 焼き込み参照は backing が生きている間の静的な参照だが、こちらはミス run の
   * 「env へ生バッファを束縛 → パイプライン生成を await → エンコード」という窓を塞ぐための
   * 予約。この窓では焼き込みがまだ 1 本も無いため、参照計数だけでは dispose が素通りする。
   */
  retainBound(): void;
  /** 進行中 run の束縛予約を 1 本返す（run のエンコードと submit が済んだ時点）。 */
  releaseBound(): void;
};

/** ホストから常駐テンソルへ書ける配列（要素は全型 4 バイト — ADR 0009 と同じ規約）。 */
export type ResidentData =
  | Float32Array<ArrayBuffer>
  | Int32Array<ArrayBuffer>
  | Uint32Array<ArrayBuffer>;

/**
 * GpuContext が所有する常駐バッファ（**第 4 の寿命クラス**）。
 *
 * 既存 3 クラス（重みアリーナ / slot backing / run アリーナ）はどれも Session の内側に閉じて
 * いて、Session を跨いだ受け渡しは必ずホスト経由（readback → writeBuffer）になる。生成ループ
 * のようにグラフ間で値を回す形では、その 1 往復ごとにフェンスが 2 本立つ。ここはその往復を
 * 消すためだけの器で、**dtype も shape も持たない**（バイト列と大きさだけ）。
 *
 * MUST: 破棄は「参照されていないこと」を確かめてから（{@link ResidentTensor.dispose}）。
 * flush-before-destroy（ADR 0004）は次の 2 つで満たしている — ①焼き込み bind group からの参照が
 * 1 本でもある間は破棄を拒む ②`enqueue` は末尾で必ず eager submit するので、戻った時点で
 * この実体を参照する**未 submit の**エンコードは存在しない（submit 済みのコマンドが参照する
 * バッファの破棄は WebGPU 的に安全 — 実解放は完了まで実装が遅延する）。
 */
export class ResidentTensor {
  /** 確保したバイト数（要求値そのもの — 4 の倍数）。 */
  readonly byteLength: number;
  /** 診断用の名前（GPUBuffer のラベルと同じ）。 */
  readonly label: string;
  /** ランタイム内部面（利用者が触る面ではない）。 */
  readonly [RUNTIME_INTERNAL]: ResidentInternals;
  readonly #gpu: GpuContext;
  readonly #buffer: GPUBuffer;
  #disposed = false;
  #bakedRefs = 0;
  #boundRefs = 0;

  /** MUST: 構築の入口は {@link GpuContext.createResident} だけ（errorScope の門を迂回させない）。 */
  constructor(gpu: GpuContext, id: number, buffer: GPUBuffer, byteLength: number, label: string) {
    this.#gpu = gpu;
    this.#buffer = buffer;
    this.byteLength = byteLength;
    this.label = label;
    this[RUNTIME_INTERNAL] = {
      buffer,
      id,
      owner: gpu,
      retainBaked: () => {
        this.#assertUsable("焼き込み");
        this.#bakedRefs += 1;
      },
      releaseBaked: () => {
        if (this.#bakedRefs === 0) {
          throw new ResidentTensorError(
            `resident '${this.label}': 焼き込み参照の解放が過多（ランタイム内部の簿記の破れ）`,
          );
        }
        this.#bakedRefs -= 1;
      },
      retainBound: () => {
        this.#assertUsable("束縛");
        this.#boundRefs += 1;
      },
      releaseBound: () => {
        if (this.#boundRefs === 0) {
          throw new ResidentTensorError(
            `resident '${this.label}': 束縛予約の解放が過多（ランタイム内部の簿記の破れ）`,
          );
        }
        this.#boundRefs -= 1;
      },
    };
  }

  /** 破棄済みか。 */
  get disposed(): boolean {
    return this.#disposed;
  }

  /** 焼き込み bind group から参照されている本数（0 でなければ破棄できない）。 */
  get bakedReferences(): number {
    return this.#bakedRefs;
  }

  /** 進行中の run が入力として束ねている本数（0 でなければ破棄できない）。 */
  get boundReferences(): number {
    return this.#boundRefs;
  }

  /**
   * ホストから全域を書く（`queue.writeBuffer`）。生成ループに入る**前**の条件テンソル投入用。
   *
   * MUST: 大きさは厳密一致。部分書きを許すと残りのバイトが前の内容のまま残り、例外も警告も
   * 出ないまま古い条件で回る（full-write — ADR 0014 と同じ思想）。
   * MUST: この `writeBuffer` は issue 順で queue timeline に載るので、**先に submit 済みの
   * dispatch を追い越さない**。追い越すのは未 submit のエンコードだけ（ADR 0004 不変条件④）
   * で、`enqueue` はその末尾で必ず submit するため両者は競合しない。
   * MUST: 消失済み device では書かない（{@link assertDeviceUsable}）。`queue.writeBuffer` は
   * 破棄済みバッファに対して例外も警告も出さない no-op なので、ここで止めないと空の条件の
   * まま生成ループが回る。
   */
  write(data: ResidentData): void {
    this.#assertUsable("write");
    assertDeviceUsable(this.#gpu, `resident '${this.label}' の write`);
    if (data.byteLength !== this.byteLength) {
      throw new ResidentTensorError(
        `resident '${this.label}': write のバイト数 ${data.byteLength} が確保 ${this.byteLength} と合わない`,
      );
    }
    this.#gpu.device.queue.writeBuffer(this.#buffer, 0, data);
  }

  /**
   * 全域をホストへ読み戻す（staging へ copy → `mapAsync`）。**フェンスはこの 1 本だけ**で、
   * 生成ループの終端で潜在を 1 度取り出すために置いてある。
   *
   * MUST: 呼ぶのは {@link BatchScope.finish} の**後**。queue の順序は保たれるので値としては
   * 正しいが、batch の内側で呼ぶとフェンスが 1 本増え、しかもこの submit の失敗が batch の
   * errorScope に帰属して原因の切り分けができなくなる。
   */
  async read(): Promise<ArrayBuffer> {
    this.#assertUsable("read");
    const device = this.#gpu.device;
    const where = `resident '${this.label}' の読み戻し`;
    // MUST: copy → submit も errorScope の両建てで囲む。COPY_SRC 欠落等の validation 失敗も
    // staging の確保失敗も例外にならず、読み戻しが全 0 のまま静かに返る（#readOutputs と同じ）。
    // MUST NOT: push から pop の発行までに await を挟まない（同期区間で完結するのでロック不要）。
    pushFailureScopes(device);
    let popped = false;
    let staging: GPUBuffer | undefined;
    try {
      staging = device.createBuffer({
        size: this.byteLength,
        usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.#buffer, 0, staging, 0, this.byteLength);
      device.queue.submit([encoder.finish()]);
      const pending = popFailureScopes(device, where);
      popped = true;
      const failure = await pending;
      if (failure !== undefined) throw failure;
      // MUST: 消失後の mapAsync が解決しない実装がありうる（実測は raceCanaryDeviceLost の
      // doc）ため競わせる — ハングを失敗に変換する保険。
      await this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(staging.mapAsync(MAP_MODE.READ), where);
      const copy = staging.getMappedRange().slice(0);
      staging.unmap();
      return copy;
    } finally {
      if (!popped) await discardFailureScopes(device);
      // MUST: staging は成否によらず必ず返す（destroy は暗黙 unmap を含む）。
      staging?.destroy();
    }
  }

  /**
   * 破棄する（2 度目以降は no-op）。
   *
   * MUST: 焼き込み bind group から参照されている間は **fail loudly**。黙って破棄すると、
   * その backing の dispatch が破棄済みバッファを束ねたまま submit され、コマンドバッファ
   * ごと失敗して**無関係な dispatch まで実行されないまま誤った値が静かに残る**（ADR 0004）。
   * 参照を外すには、その Session を dispose するか、別 signature の run / enqueue で backing を
   * 切り替える。
   * MUST: **進行中の run が入力として束ねている間**も同じく fail loudly。焼き込みが立つのは
   * backing 構築（= ヒット run 以降）なので、ミス run の「束縛 → パイプライン生成の await →
   * エンコード」の窓は焼き込み参照だけでは守れない（その窓で破棄すると、再開したエンコードが
   * 破棄済みバッファを掴んで run が validation で落ちる）。この予約があると、誤りは
   * dispose の呼び出し点で真因のまま落ちる。
   */
  dispose(): void {
    if (this.#disposed) return;
    if (this.#bakedRefs > 0) {
      throw new ResidentTensorError(
        `resident '${this.label}': 焼き込み bind group から参照中（${this.#bakedRefs} 本）のため破棄できない。` +
          "参照している Session を dispose するか、別 signature の run / enqueue で backing を切り替えること",
      );
    }
    if (this.#boundRefs > 0) {
      throw new ResidentTensorError(
        `resident '${this.label}': 進行中の run が入力として束縛中（${this.#boundRefs} 本）のため破棄できない。` +
          "その run の完了を await してから破棄すること",
      );
    }
    this.#disposed = true;
    this.#buffer.destroy();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }

  #assertUsable(where: string): void {
    if (this.#disposed) {
      throw new ResidentTensorError(
        `resident '${this.label}': 破棄済みの常駐テンソルは使えない（${where}）`,
      );
    }
  }
}

/**
 * batch が決着時に取りまとめる相手（実体は {@link SubmitScheduler}）。
 * 構造型にしてあるのは、device 層が submit 層へ依存しないため。
 */
export type BatchMember = {
  /** 未 submit のエンコードを submit する（**フェンスは張らない**）。 */
  submitPending(): void;
  /** batch のフェンス完了後に計測窓を閉じる。 */
  closeMeasurementWindowAfterFence(): void;
};

/** {@link BatchScope} のランタイム内部面。 */
type BatchInternals = {
  /** errorScope 区間が実際に開くまでの待ち（{@link GpuContext.beginBatch} が await する）。 */
  readonly entered: Promise<void>;
  /** 決着時に取りまとめる相手を登録する（同じ相手の重複登録は無害）。 */
  join(member: BatchMember): void;
  /**
   * enqueue の受け口として使える状態か検査し、**in-flight リースを 1 本取る**
   * （決着済み / 別 device は fail loudly で、リースは取らない）。
   *
   * MUST: 呼ぶのは `Session.enqueue` の**同期区間**（`#chain` に積む前）。enqueue の本体は
   * マイクロタスクを 1 段挟むので、本体でリースを取ると「未 await の enqueue を積んだ直後に
   * `finish()`」で finish が先に決着し、積んだ enqueue が 1 本も dispatch されないまま区間が
   * 正常終了に見える（沈黙の空振り）。
   */
  enter(owner: GpuContext): void;
  /** in-flight リースを 1 本返す（enqueue 本体の成功・失敗どちらの経路でも必ず）。 */
  leave(): void;
};

/**
 * フェンス無しの enqueue を束ねる区間（{@link GpuContext.beginBatch}）。
 *
 * 区間の間 device の errorScope 区間ロックを保持し、`out-of-memory` + `validation` の 2 本を
 * 張り続ける。{@link BatchScope.finish} が ①全メンバの未 submit を出し切り ②
 * `onSubmittedWorkDone` を**1 回だけ**待ち ③スコープを pop して失敗を型付き例外にする。
 *
 * トレードオフ（設計上の受容）:
 *
 * - **失敗の帰属は batch 単位**。1 区間に N 本の enqueue が相乗りするので、validation /
 *   out-of-memory が出ても「どの enqueue か」までは絞れない。切り分けが要るときは通常の
 *   `run` で 1 本ずつ回す（run は従来どおり run 単位で帰属する）。
 * - **device 消失の検出は finish まで遅延する**。区間中の待ちが 1 本も無いのだから当然で、
 *   消失は finish の `raceDeviceLost` が例外へ変換する（enqueue 側はハングしない — 待たない
 *   から）。
 */
export class BatchScope {
  /** ランタイム内部面（利用者が触る面ではない）。 */
  readonly [RUNTIME_INTERNAL]: BatchInternals;
  readonly #gpu: GpuContext;
  readonly #members = new Set<BatchMember>();
  readonly #completion: Promise<void>;
  readonly #release: () => void;
  /** 未返却の in-flight リースが全て返ったことの通知（{@link BatchInternals.enter}）。 */
  readonly #drained = Promise.withResolvers<void>();
  #leases = 0;
  #finished = false;

  /** MUST: 構築の入口は {@link GpuContext.beginBatch} だけ（計測との併用の門をここに置く）。 */
  constructor(gpu: GpuContext) {
    this.#gpu = gpu;
    const hold = Promise.withResolvers<void>();
    const entered = Promise.withResolvers<void>();
    this.#release = hold.resolve;
    this.#completion = gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
      try {
        pushFailureScopes(gpu.device);
      } catch (cause) {
        // MUST: 開けなかったことを beginBatch の待ちへ必ず伝える（伝えないとハングになる）。
        entered.reject(cause);
        throw cause;
      }
      entered.resolve();
      await hold.promise;
      // MUST: フェンスを張る前に in-flight の enqueue を全て決着させる。`finish()` は同期で
      // `#finished` を立てるだけなので、これが無いと「未 await の enqueue を積んだ直後の
      // finish()」で ①まだ本体が走っていない enqueue が全て reject し、区間は 0 dispatch の
      // まま**成功で決着する** ②走り出していた enqueue はフェンスと pop の後に submit し、
      // 未完了の GPU 実行を残したまま finish が返る（直後の read が古い値を返す）。
      await this.#drained.promise;
      const device = gpu.device;
      try {
        // enqueue は末尾で必ず submit するので通常ここは空振りする。それでも出し切るのは、
        // 「batch が閉じた時点で未 submit のエンコードは 1 つも無い」を区間側の責務として
        // 閉じるため（破棄経路の担い手を増やさない — ADR 0004）。
        for (const member of this.#members) member.submitPending();
        // MUST: 消失後の onSubmittedWorkDone が解決しない実装がありうる（実測は
        // raceCanaryDeviceLost の doc）ため競わせる。
        await gpu[RUNTIME_INTERNAL].raceDeviceLost(
          device.queue.onSubmittedWorkDone(),
          "batch の完了",
        );
      } catch (cause) {
        await discardFailureScopes(device);
        throw cause;
      } finally {
        // 計測窓は batch のフェンス 1 回で閉じる。窓に N 本の enqueue が入るぶん推定は粗く
        // （過大に）出るが、過大 = チャンクが小さくなる向き = TDR に対して安全側
        // （src/gpu/submit.ts の「計測の帰属」）。
        for (const member of this.#members) member.closeMeasurementWindowAfterFence();
      }
      const failure = await popFailureScopes(device, "batch のエンコード");
      if (failure !== undefined) throw failure;
    });
    // 決着を finish が受け取るまで未処理拒否にしない（拒否の中身は finish がそのまま返す）。
    void this.#completion.catch(() => undefined);
    this[RUNTIME_INTERNAL] = {
      entered: entered.promise,
      join: (member) => {
        this.#members.add(member);
      },
      enter: (owner) => {
        if (owner !== this.#gpu) {
          throw new BatchScopeError("別の GpuContext で開いた batch には enqueue できない");
        }
        if (this.#finished) {
          throw new BatchScopeError("finish() 済みの batch には enqueue できない");
        }
        this.#leases += 1;
      },
      leave: () => {
        if (this.#leases === 0) {
          throw new BatchScopeError("batch の in-flight リースの返却が過多（内部の簿記の破れ）");
        }
        this.#leases -= 1;
        if (this.#leases === 0 && this.#finished) this.#drained.resolve();
      },
    };
  }

  /** 決着済みか（{@link BatchScope.finish} を 1 度でも呼んだか）。 */
  get finished(): boolean {
    return this.#finished;
  }

  /**
   * 区間を閉じる。**新規 enqueue を拒否 → in-flight の enqueue が全て決着するのを待つ →
   * 未 submit を出し切る → フェンス 1 本で全 enqueue の完了を待つ → errorScope を pop して
   * 失敗を型付き例外にする**、の順で進む。
   *
   * MUST: 2 度目以降も同じ完了を返す（先に返すと呼び出し側が破棄へ進み、ロックと errorScope が
   * 開いたまま残る）。
   * NOTE: in-flight を待つので、`enqueue()` の戻り Promise を await せずに `finish()` を
   * 呼んでも積んだぶんは必ず区間に入って完了する（リースの機構は {@link BatchInternals.enter}）。
   */
  finish(): Promise<void> {
    if (!this.#finished) {
      this.#finished = true;
      // in-flight が 1 本も無ければここで決着させる（enqueue を 1 本も出していない区間・
      // 全て await 済みの区間は従来どおり待ちが増えない）。
      if (this.#leases === 0) this.#drained.resolve();
      this.#release();
    }
    return this.#completion;
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
  const limits = planRequiredLimits(adapter.limits, options[LIMIT_CAPS]);
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
 * internal + validation の 2 本を張って `body`（パイプライン生成）を実行し、捕捉したエラーを
 * 例外に変換する。**パイプライン生成経路はこちらを使う**（validation 1 本では internal
 * エラーが素通りする）。
 *
 * 両方が要るのは失敗の出方が違うため — 上限超過や型不整合は validation、実装側の都合
 * （シェーダが複雑すぎてコンパイルできない等）は internal で、どちらも同期例外にならず
 * **無効なパイプラインが返るだけ**。囲まないと dispatch が no-op 化して出力が全て 0 になる。
 *
 * MUST NOT: `body` の中で非同期処理を待たない。errorScope はスタックで、pop は body の同期
 * 実行直後に起きるため、await 後にエンコードした操作は別のスコープに吸われる。
 */
export const withPipelineScope = async <T>(
  device: GPUDevice,
  label: string,
  body: () => T,
): Promise<T> => {
  device.pushErrorScope("internal");
  device.pushErrorScope("validation");
  let value: T;
  try {
    value = body();
  } catch (cause) {
    // MUST: body が throw しても 2 本とも pop する。片方でも積み残すと後続の検証結果が誤った
    // スコープに吸われ、以後のエラーが恒久的に見えなくなる。pop 自体の失敗は握り潰す
    // （後始末で本体の例外を上書きしない — discardFailureScopes と同じ規律）。
    const validation = device.popErrorScope().catch(() => null);
    const internal = device.popErrorScope().catch(() => null);
    await Promise.all([validation, internal]);
    throw cause;
  }
  // MUST: 2 本の pop は**同一同期区間で発行**する（await するのは発行済みの promise だけ）。
  // pop はスタック先頭を無条件に取るため、発行の間に await を挟むと、その隙に他所が push した
  // スコープを 2 本目が取り、失敗が誤帰属する（popFailureScopes と同じ規律）。
  const validation = device.popErrorScope();
  const internal = device.popErrorScope();
  const [validationError, internalError] = await Promise.all([validation, internal]);
  // MUST: 両方が捕捉されたときは internal を返す。internal で無効化されたパイプラインを触る
  // 後続の操作（`getBindGroupLayout` 等）は**派生の** validation エラーを立てるため、
  // validation を先に返すと根因の internal が捨てられ、原因の分からない「無効パイプライン」
  // として報告される。OOM を validation より先に返すのと同型の判断
  // （docs/research/2026-08-08-vram-oom-misreport.md）。
  if (internalError !== null) {
    throw new GpuInternalError(`${label}: ${internalError.message}`);
  }
  if (validationError !== null) {
    throw new GpuValidationError(`${label}: ${validationError.message}`);
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
