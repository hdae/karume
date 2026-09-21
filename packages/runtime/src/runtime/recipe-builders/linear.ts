/**
 * 族別導出 — 行列積の族（matmul・bmm・linear とその GEMV / i8a8 変種・embedding）。
 *
 * 入口は {@link "../recipe-builder.ts"} の `RecipeBuilder` で、共有サービス（Session の状態・
 * params の書き込み・重みの格納形と scale 束縛）は {@link RecipeBuildFace} 経由でだけ触る。
 */

import type { BindingSource, StepRecipeBuilder } from "../recipe.ts";
import {
  EMBEDDING_SCALE_BINDING,
  EMBEDDING_WORKGROUP_SIZE,
  embeddingKey,
  embeddingParams,
  embeddingWgsl,
} from "../../kernels/embedding.ts";
import { ExecutionError, type NodePlan } from "../plan.ts";
import { type GemmCompute, gemmUsesVec4 } from "../../kernels/gemm.ts";
import {
  LINEAR_ACT_SCALE_BINDING,
  LINEAR_I8A8_MAX_K,
  linearI8a8Key,
  linearI8a8Params,
  linearI8a8UsesVec4,
  linearI8a8Wgsl,
} from "../../kernels/linear-i8a8.ts";
import {
  defaultLinearGemvRowsVariant,
  defaultLinearGemvVariant,
  LINEAR_GEMV_MAX_ROWS,
  linearGemvKey,
  linearGemvPackedEligible,
  linearGemvParallelKey,
  linearGemvParallelLanes,
  linearGemvParallelPackedKey,
  linearGemvParallelPackedParams,
  linearGemvParallelPackedWgsl,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvRowsKey,
  linearGemvRowsWgsl,
  linearGemvSubgroupKey,
  linearGemvSubgroupWgsl,
  linearGemvUnit,
  linearGemvWgsl,
} from "../../kernels/linear-gemv.ts";
import { LINEAR_SCALE_BINDING, linearKey, linearParams, linearWgsl } from "../../kernels/linear.ts";
import type { PackedActivations } from "../fusion.ts";
import type { WeightStorage } from "../../kernels/weight-storage.ts";
import { bmmKey, bmmParams, bmmWgsl } from "../../kernels/bmm.ts";
import { defaultI8a8Geometry, i8a8TileM, i8a8TileN } from "../../kernels/i8a8-geometry.ts";
import { gemmGeometryForRows, gemmTileM, gemmTileN } from "../../kernels/gemm-geometry.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../../codegen/dispatch.ts";
import { matmulKey, matmulParams, matmulWgsl } from "../../kernels/matmul.ts";
import { numel } from "../../ops.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsWgsl,
} from "../../kernels/quantize-rows.ts";
import type { RecipeBuildFace } from "../recipe-builder.ts";
import { PARAMS_UNIFORM_USAGE } from "./params-usage.ts";

export const buildMatmul = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [a, b] = step.inputShapes;
  const [m, k] = a;
  const n = b[1];
  // v4（vec4 の読み書き）は形状から導く 1 ビット。導出時に評価してキーと WGSL の
  // 両方へ渡す（同一キー ⇔ 同一バイト列は保たれる）。
  const v4 = gemmUsesVec4(k, n);
  // MUST: タイル幾何は行数 M のバケット（src/kernels/gemm-geometry.ts）。キー・WGSL・
  // dispatch の 3 つに**同じ m** を通す — 1 つでも渡し忘れると出力タイルが静かに欠ける。
  const key = matmulKey(v4, m);
  const { pipeline, layout, roles } = await face.state.cache.get(key, matmulWgsl(v4, m));
  const params = face.writeParams(matmulParams(m, n, k), PARAMS_UNIFORM_USAGE);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `matmul [${a.join(",")}] × [${b.join(",")}]`;
  const geometry = gemmGeometryForRows(m);
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: binds[1] },
      { binding: 3, source: outs[0] },
    ],
    workgroups: [
      tiledWorkgroups(n, gemmTileN(geometry), limit, where),
      tiledWorkgroups(m, gemmTileM(geometry), limit, where),
      1,
    ],
  });
};

/**
 * バッチ matmul（rank-3）。タイル 2 軸に加えて**バッチを z 軸**へ載せる。
 * MUST: matmul と同じ「1 workgroup = 1 タイル」なので、3 軸とも上限超過は fail loudly。
 */
export const buildBmm = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [a, b] = step.inputShapes;
  const [batch, m, k] = a;
  const n = b[2];
  const v4 = gemmUsesVec4(k, n);
  // 幾何のバケットは**行列 1 枚の m**（バッチは z 軸で、タイル幾何とは独立）。
  const key = bmmKey(v4, m);
  const { pipeline, layout, roles } = await face.state.cache.get(key, bmmWgsl(v4, m));
  const params = face.writeParams(bmmParams(m, n, k), PARAMS_UNIFORM_USAGE);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `bmm [${a.join(",")}] × [${b.join(",")}]`;
  const geometry = gemmGeometryForRows(m);
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: binds[1] },
      { binding: 3, source: outs[0] },
    ],
    workgroups: [
      tiledWorkgroups(n, gemmTileN(geometry), limit, where),
      tiledWorkgroups(m, gemmTileM(geometry), limit, where),
      // バッチは 1 workgroup = 1 バッチ。ここも縮退させるとバッチが丸ごと未書き込みになる。
      tiledWorkgroups(batch, 1, limit, where),
    ],
  });
};

/**
 * linear（融合 op — ADR 0012）。先行次元を平坦化して `[m,k] × [k,n] + bias` の 2 次元
 * GEMM に落とす。重みは `[n,k]` の転置レイアウトのままカーネルが読む（転置コピー無し）。
 * MUST: matmul と同じ「1 workgroup = 1 タイル」なので、上限超過は fail loudly。
 */
export const buildLinear = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  packed: PackedActivations | undefined,
): Promise<void> => {
  const [x, weight] = step.inputShapes;
  const n = weight[0];
  const k = weight[1];
  const m = numel(x.slice(0, -1));
  const weightStorage = face.weightStorage(step);
  if (weightStorage === "i2" && face.state.linearCompute !== "f32") {
    throw new ExecutionError("linear: i2 常駐は linearCompute 'f32' のみ対応（ADR 0097）");
  }
  // packed 活性（ADR 0105）を受け取ると宣言された linear が並列 GEMV へ落ちなければ、
  // `vec4<u32>` の束縛に f32 の語が流れる = 例外なしの沈黙誤値になる。対付けの受理は
  // fusion.ts の 1 箇所だが、**同じ述語で**ここでも確かめる（fail loudly）。
  if (packed !== undefined) {
    const eligible = packed.role === "read" &&
      face.state.linearGemvReduce === "parallel" && face.state.linearCompute === "f32" &&
      linearGemvPackedEligible(
          weightStorage,
          m,
          n,
          k,
          weightStorage === "i4" ? face.weightGroupSize(step) : undefined,
        ) !== undefined;
    if (!eligible) {
      throw new ExecutionError(
        `linear [${x.join(",")}] × [${weight.join(",")}]: ` +
          "packed 活性（ADR 0105）と宣言されたが並列 GEMV へ落ちない",
      );
    }
  }
  // 整数内積の経路は **opt-in × 整数常駐（i8 / i4）× k > 0 × k % 4 == 0** の 4 条件が
  // 揃ったときだけ（ADR 0025 / w4a8 は perf-ledger Q-8）。既定の "f32" では 1 バイトも
  // 挙動が変わらない。
  // MUST: ノブ `linearCompute: "a8"` の意味は「**活性を i8 にして整数内積で計算する**」で、
  // 重みの格納形は別軸（i8 常駐 → w8a8 / i4 常駐 → w4a8）。i4 を述語から外すと、選んでも
  // 挙動が変わらない嘘の席になる（i4 常駐が黙って f32 計算経路へ流れて dp4a の利得だけを
  // 失う — docs/research/2026-08-21-anima-i4-seat-speed.md）。
  // MUST: `k > 0` を含める。`k == 0` は契約上有効な退化 shape（src/ops.ts の linear は
  // in=0 を許す）だが、i8a8 経路の ① `quantize_rows` は `dim >= 1` を要求するので、拾うと
  // i8a8 固有の CodegenError になる — 縮約が空 = 量子化する活性がそもそも無く、i8a8 の門と
  // しての意味を持たない例外なので、経路の選択で失敗の理由が変わらないよう通常経路へ落とす。
  // NOTE: K=0 自体は通常経路でも 0 バイト束縛が最小束縛サイズを割って落ちる（この述語とは
  // 無関係の別要因）。
  if (
    face.state.linearCompute === "a8" &&
    (weightStorage === "i8" || weightStorage === "i4") && k > 0 && k % 4 === 0
  ) {
    await buildLinearI8a8(face, step, binds, outs, builder, m, n, k, weightStorage);
    return;
  }
  // f16 計算変種（ADR 0028）。**i8 常駐の重みとは組めない**（w8a16 は未実装）ので、
  // 黙って f32 へ落とさずここで落とす — 落とすと「i8 の層だけ f16 が効かない」形になり、
  // 診断のキーを見ない限り気づけない。
  const compute: GemmCompute = face.state.linearCompute === "f16" ? "f16" : "f32";
  if (compute === "f16" && weightStorage === "i8") {
    throw new ExecutionError(
      `linear [${x.join(",")}] × [${weight.join(",")}]: ` +
        "linearCompute 'f16' は i8 常駐の重みとは組めない（w8a16 は未実装 — ADR 0028）。" +
        "linearCompute を 'a8' にするか、この重みを f32 / f16 格納で持つこと",
    );
  }
  // i4 も同型（w4a16 は未実装 — ADR 0069。黙って f32 計算へ落とさない理由は i8 と同文）。
  if (compute === "f16" && weightStorage === "i4") {
    throw new ExecutionError(
      `linear [${x.join(",")}] × [${weight.join(",")}]: ` +
        "linearCompute 'f16' は i4 常駐の重みとは組めない（w4a16 は未実装 — ADR 0069）。" +
        "linearCompute を 'a8' にするか、この重みを f32 / f16 格納で持つこと",
    );
  }
  const v4 = gemmUsesVec4(k, n);
  // i4 は group 長がキーと WGSL（shift の焼き込み）の両方に効く（ADR 0069 — 同一キー →
  // バイト同一 WGSL の codegen 決定性）。
  const groupSize = weightStorage === "i4" ? face.weightGroupSize(step) : undefined;
  // **小 M（1 ≤ M ≤ LINEAR_GEMV_MAX_ROWS）は GEMV 族へ分岐する**（ADR 0082 — perf-ledger
  // K-11 / i8 は K-16 / M ≥ 2 の行ブロックは K-21）。既定の GEMM 骨格は
  // M ≤ 64 のバケット（M16N16）で 64 スレッド中 4 本しか出力を書かず、K タイル 16 ごとの二重
  // barrier が重み読みのレイテンシを逐次に露出させる（k 比例・n 非依存・L2 常駐形でも同じ —
  // 帯域飢餓ではない）。この費用は M に無関係（M=2〜16 で GEMM 経路 65〜67 ms が不変 —
  // research 2026-09-07 §7.2）。gemma4 E2B decode で対既定 ×8.45・M=8 で ×5.0・M=32 で ×2.8・
  // M=64 で ×2.7（census 加重 — research 2026-09-07-gemv-rows-k21）。
  // MUST: 縮約順・積和の字面・bias の足し順は既定経路と同一 = **ビット同一**
  // （src/kernels/linear-gemv.ts の数値契約・門は tests/gpu_linear_gemv_test.ts）。
  // 門の内訳:
  // - `1 ≤ m ≤ LINEAR_GEMV_MAX_ROWS` — 1 スレッドが 1 出力列（× 行ブロック）の縮約を丸ごと持つ
  //   形で実測した範囲。上限の外は既定の GEMM 骨格のまま。
  // - `i4` / `i8` 格納 × `f32` 計算。f16 / f32 格納は下の M=1 専用の門で受ける。
  //   i8 × f16 計算（w8a16）は上で落ちている。
  // - `groupSize % <刻み> === 0`（i4 のみ）— 重み 1 語 32 要素が group を跨がない条件
  //   （跨ぐと語あたり 1 個の scale では足りず沈黙誤値になる）。i8 の scale は出力チャネル
  //   ごと 1 本なので、跨ぐ相手がそもそも無い。
  // - `k % <刻み> === 0` — 重み束縛 `vec4<u32>` の 16 B 整列（刻みは格納ごと = i4 32 要素 /
  //   i8 16 要素）。i4 では宣言層の「行長は group_size の倍数」（ADR 0069 決定 2）から従うが、
  //   束縛の要件として言い直す。
  // - `v4` は GEMV の要件では**ない**（出力はスカラ書きなので n の整除は要らない）が、
  //   掃引した実形が全て v4 なので門を実測の範囲に留める。n % 4 != 0 の M=1 は既定の
  //   スカラ変種のまま（値は同じ・速度だけ従来どおり）。
  // f16 / f32 格納も M=1 のみ同じ骨格へ（ADR 0082 追記 6・7）。計算は f32 のまま、
  // 重み 1 語 = f16 は 8 要素、f32 は 4 要素。行ブロックと f16 計算はこの変種の検収範囲に含めない。
  if (
    m === 1 && (weightStorage === "f16" || weightStorage === "f32") &&
    compute === "f32" && v4 && k > 0 && k % linearGemvUnit(weightStorage) === 0
  ) {
    await buildLinearGemv(
      face,
      step,
      binds,
      outs,
      builder,
      weightStorage,
      m,
      n,
      k,
      undefined,
      undefined,
    );
    return;
  }
  const i4Unit = linearGemvUnit("i4");
  const gemvRows = m >= 1 && m <= LINEAR_GEMV_MAX_ROWS;
  if (
    gemvRows && weightStorage === "i2" && compute === "f32" && v4 &&
    k % linearGemvUnit("i2") === 0
  ) {
    await buildLinearGemv(face, step, binds, outs, builder, "i2", m, n, k, undefined, packed);
    return;
  }

  if (
    gemvRows && weightStorage === "i4" && compute === "f32" && v4 &&
    groupSize !== undefined && groupSize % i4Unit === 0 &&
    k % i4Unit === 0
  ) {
    await buildLinearGemv(face, step, binds, outs, builder, "i4", m, n, k, groupSize, packed);
    return;
  }
  // i8 格納（lm_head — perf-ledger K-16）。i4 と同じ族・同じ変種で、違うのは 1 語が運ぶ
  // 要素数（16）と scale の引き方（出力チャネルごと 1 本 = 縮約の外で 1 度だけ束ねる）だけ。
  if (
    gemvRows && weightStorage === "i8" && compute === "f32" && v4 &&
    k % linearGemvUnit("i8") === 0
  ) {
    await buildLinearGemv(face, step, binds, outs, builder, "i8", m, n, k, undefined, packed);
    return;
  }
  // MUST: タイル幾何は平坦化後の行数 m のバケット（src/kernels/gemm-geometry.ts）。
  // キー・WGSL・dispatch に**同じ m** を通す。
  const key = linearKey(weightStorage, v4, compute, m, groupSize);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    linearWgsl(weightStorage, v4, compute, m, groupSize),
  );
  const params = face.writeParams(linearParams(m, n, k), PARAMS_UNIFORM_USAGE);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `linear [${x.join(",")}] × [${weight.join(",")}]`;
  const geometry = gemmGeometryForRows(m);
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, LINEAR_SCALE_BINDING),
    ],
    workgroups: [
      tiledWorkgroups(n, gemmTileN(geometry), limit, where),
      tiledWorkgroups(m, gemmTileM(geometry), limit, where),
      1,
    ],
  });
};

/**
 * linear の **GEMV 族**（1 ≤ M ≤ LINEAR_GEMV_MAX_ROWS × 重み i4 / i8 — ADR 0082）。
 *
 * 束縛・uniform・出力実体は既定経路と同一で、変わるのは「どのスレッドがどの出力を担当するか」
 * だけ（1 スレッド = 1 出力列〈M ≥ 2 は × 行ブロック〉・共有タイルと barrier を持たない）。
 * 例外は明示指定の linearGemvReduce: parallel（ADR 0098）で、対象形状の K を分担する。
 * 1 出力要素あたりの K 縮約順は k 昇順の逐次のままなので**ビット同一**（src/kernels/linear-gemv.ts
 * の数値契約）。
 * MUST: M=1 は M=1 変種（decode の生成物を動かさない）、M ≥ 2 は行ブロック変種で、行数 `rows` は
 * (格納, m, n, 並列度の目標) の純関数 `defaultLinearGemvRowsVariant`（キーに載る）。目標は
 * Session 生成時に固定された静的なノブ（`SessionOptions.linearGemvRowsThreadTarget`）。
 * MUST: `groupSize` は i4 のときだけ渡す（i8 は group を持たない — カーネル側が対を検査する）。
 * MUST: 1 スレッド 1 出力（列 × 行ブロック）なので dispatch は `[ceil(n / cols), ceil(m / rows), 1]`。
 * grid-stride ではないので上限超過は fail loudly（既定経路と同じ規律）。
 */
const buildLinearGemv = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  groupSize: number | undefined,
  /** packed int8 活性の対（ADR 0105）。並列変種のときだけ立ちうる。 */
  packed: PackedActivations | undefined,
): Promise<void> => {
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const [x, weight] = step.inputShapes;
  const where = `linear gemv [${x.join(",")}] × [${weight.join(",")}]`;
  // 数値を変える選択は明示指定時のみ。参照経路の選択・WGSLは従来どおり。
  const subgroup = face.state.linearGemvReduce === "parallel-subgroup32";
  const lanes = face.state.linearGemvReduce !== "sequential"
    ? linearGemvParallelLanes(storage, m, n, k, groupSize)
    : undefined;
  const rowsVariant = m === 1 || lanes !== undefined
    ? undefined
    : defaultLinearGemvRowsVariant(storage, m, n, face.state.linearGemvRowsThreadTarget);
  const variant = rowsVariant ?? defaultLinearGemvVariant({ storage, n, k });
  const rows = rowsVariant?.rows ?? 1;
  // packed 活性の変種は並列（非 subgroup）だけが持つ。`buildLinear` の門を通っていれば
  // ここは必ず `lanes !== undefined && !subgroup` になる。
  const packedScale = packed === undefined ? undefined : packed.scale;
  const key = lanes !== undefined
    ? subgroup
      ? linearGemvSubgroupKey(storage, groupSize, lanes)
      : packedScale === undefined
      ? linearGemvParallelKey(storage, groupSize, lanes)
      : linearGemvParallelPackedKey(storage, groupSize, lanes)
    : rowsVariant === undefined
    ? linearGemvKey(storage, groupSize, variant)
    : linearGemvRowsKey(storage, groupSize, rowsVariant);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    lanes !== undefined
      ? subgroup
        ? linearGemvSubgroupWgsl(storage, groupSize, lanes)
        : packedScale === undefined
        ? linearGemvParallelWgsl(storage, groupSize, lanes)
        : linearGemvParallelPackedWgsl(storage, groupSize, lanes)
      : rowsVariant === undefined
      ? linearGemvWgsl(storage, groupSize, variant)
      : linearGemvRowsWgsl(storage, groupSize, rowsVariant),
  );
  // uniform は既定経路と同じ 3 語（束縛レイアウトを分けない）。整除の検査は族側の 1 箇所。
  // packed 変種だけが Dims の末尾へ活性 scale を 1 語足す。
  const params = face.writeParams(
    packedScale === undefined
      ? linearGemvParams(storage, m, n, k, groupSize)
      : linearGemvParallelPackedParams(storage, m, n, k, packedScale, groupSize),
    PARAMS_UNIFORM_USAGE,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, LINEAR_SCALE_BINDING),
    ],
    workgroups: [
      tiledWorkgroups(n, lanes === undefined ? variant.cols : 128 / lanes, limit, where),
      tiledWorkgroups(m, rows, limit, where),
      1,
    ],
  });
};

/**
 * linear の **w8a8 / w4a8 変種**（opt-in — {@link SessionOptions.linearCompute}）。
 * **1 ノード = 2 dispatch**（融合 attention と同じ「複数 dispatch で 1 ノード」の扱い）:
 *
 * ① `quantize_rows`（活性を per-token i8 へ・行方向 grid-stride）→ ② 整数内積 GEMM
 * （1 workgroup = 1 出力タイルなので上限超過は fail loudly）。①は重みの格納形に依らず同一。
 *
 * MUST: 一時バッファ（`xq` / `xs`）は宣言 → ノード末尾で解放する。これで計画の参照計数が
 * 閉じ、失敗経路でも `arena.destroy()` が領域ごと拾う。
 * MUST: i32 のオーバフロー門は fail loudly。黙って通すと i32 の巻き戻りで符号ごと化ける。
 * 門の軸は格納形で違う — i8 は **k**（縮約全体が 1 つの i32）、i4 は **group 長**
 * （flush が group ごとなので i32 に載るのは 1 group ぶんだけ）。
 */
const buildLinearI8a8 = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  m: number,
  n: number,
  k: number,
  weightStorage: WeightStorage,
): Promise<void> => {
  const [x, weight] = step.inputShapes;
  const where = `linear i8a8 [${x.join(",")}] × [${weight.join(",")}]`;
  // i4 は group 長がキーと WGSL（shift の焼き込み）の両方に効く（ADR 0069）。
  const groupSize = weightStorage === "i4" ? face.weightGroupSize(step) : undefined;
  if (groupSize === undefined && k > LINEAR_I8A8_MAX_K) {
    throw new ExecutionError(
      `${where}: k=${k} が i8a8 経路の i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える` +
        "（linearCompute を 'f32' にするか、この linear を i8 常駐から外す）",
    );
  }
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;

  // 量子化した活性 `xq`（i8 を 4 詰め）と per-token scale `xs`。ノード内で閉じた一時領域。
  const xq = builder.allocTemp(Math.max(4, m * (k / 4) * 4));
  const xs = builder.allocTemp(Math.max(4, m * 4));

  // ① 活性の per-token 量子化（1 行 = 1 workgroup・行方向 grid-stride）
  const quantizeGeometry = quantizeRowsGeometry(k);
  const quantizeKey = quantizeRowsKey(quantizeGeometry);
  const { pipeline: quantizePipeline, layout: quantizeLayout, roles: quantizeRoles } = await face
    .state.cache.get(
      quantizeKey,
      quantizeRowsWgsl(quantizeGeometry),
    );
  builder.dispatch({
    key: quantizeKey,
    pipeline: quantizePipeline,
    layout: quantizeLayout,
    roles: quantizeRoles,
    params: face.writeParams(quantizeRowsParams(m, k), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: xq },
      { binding: 3, source: xs },
    ],
    workgroups: [gridStrideWorkgroups(m, quantizeGeometry.rowsPerGroup, limit), 1, 1],
  });

  // ② 整数内積の GEMM。タイル幾何は op → 幾何の純関数が決める（src/kernels/i8a8-geometry.ts）
  // — キーに載るので「同一キー → バイト同一 WGSL」は保たれる。
  const v4 = linearI8a8UsesVec4(n);
  const geometry = defaultI8a8Geometry("linear");
  const dp4a = face.state.linearI8a8Dot === "dp4a";
  // MUST: key / wgsl とも **実際の常駐形**（`weightStorage`）で引く。i8 固定にすると i4 の
  // group scale が per-channel として配られる沈黙誤値になる（数値契約が別 — ADR 0076）。
  const key = linearI8a8Key(v4, dp4a, geometry, weightStorage, groupSize);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    linearI8a8Wgsl(v4, dp4a, geometry, weightStorage, groupSize),
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params: face.writeParams(linearI8a8Params(m, n, k, groupSize), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: xq },
      { binding: 2, source: binds[1] },
      { binding: 3, source: binds[2] },
      { binding: 4, source: outs[0] },
      // 束縛の**有無**だけが常駐形（i8 / i4）で決まる — バッファ自体は初期化子名で引くので
      // i8 / i4 で同一。scale の**解釈**（per-channel か group か）を決めるのは上の
      // `linearI8a8Wgsl` / `linearI8a8Key` の判別子で、そちらが本当の沈黙誤値の門。
      ...face.weightScaleBindings(step, LINEAR_SCALE_BINDING),
      { binding: LINEAR_ACT_SCALE_BINDING, source: xs },
    ],
    workgroups: [
      tiledWorkgroups(n, i8a8TileN(geometry), limit, where),
      tiledWorkgroups(m, i8a8TileM(geometry), limit, where),
      1,
    ],
  });

  // MUST: ノード境界で一時バッファを返す（アリーナの不変条件）。
  builder.releaseTemp(xs);
  builder.releaseTemp(xq);
};

/**
 * embedding（行 gather）。範囲外添字の扱いは src/kernels/embedding.ts の裁定
 * （GPU は NaN 汚染 / CPU 参照は throw）。attrs の padding_idx は forward に効かないので
 * カーネルへ渡さない。
 */
export const buildEmbedding = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const weight = step.inputShapes[0];
  const count = numel(step.outputs[0].shape);
  const weightStorage = face.weightStorage(step);
  // i4 は group 長を WGSL に焼く（キーの g 部と対 — linear と同じ規律・ADR 0069）。
  const groupSize = weightStorage === "i4" ? face.weightGroupSize(step) : undefined;
  const key = embeddingKey(weightStorage, groupSize);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    embeddingWgsl(weightStorage, groupSize),
  );
  const params = face.writeParams(
    embeddingParams(count, weight[1], weight[0], groupSize),
    PARAMS_UNIFORM_USAGE,
  );
  const groups = gridStrideWorkgroups(
    count,
    EMBEDDING_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: binds[1] },
      { binding: 3, source: outs[0] },
      ...face.weightScaleBindings(step, EMBEDDING_SCALE_BINDING),
    ],
    workgroups: [groups, 1, 1],
  });
};
