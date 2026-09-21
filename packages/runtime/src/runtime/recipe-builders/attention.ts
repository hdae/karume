/**
 * 族別導出 — attention の全形（融合 attention とその i8a8 補助・states 形 / readonly states 形・
 * state_append）。
 *
 * 入口は {@link "../recipe-builder.ts"} の `RecipeBuilder` で、共有サービス（Session の状態と
 * params の書き込み）は {@link RecipeBuildFace} 経由でだけ触る。
 */

import {
  ATTENTION_PV_V_SCALE_BINDING,
  ATTENTION_QK_K_SCALE_BINDING,
  ATTENTION_QK_Q_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../../kernels/attention-i8a8.ts";
import {
  ATTENTION_QK_MASK_BINDING,
  type GemmCompute,
  gemmUsesVec4,
  statePvTiledWgsl,
  stateQkTiledWgsl,
} from "../../kernels/gemm.ts";
import {
  ATTENTION_STATS_STRIDE,
  attentionPvKey,
  attentionPvParams,
  attentionPvWgsl,
  attentionQkKey,
  attentionQkParams,
  attentionQkWgsl,
  attentionStatsKey,
  attentionStatsParams,
  attentionStatsRegCache,
  attentionStatsWgsl,
} from "../../kernels/attention.ts";
import type { BindingSource, StepRecipeBuilder, TempSource } from "../recipe.ts";
import { ExecutionError, type NodePlan } from "../plan.ts";
import { LINEAR_I8A8_MAX_K } from "../../kernels/linear-i8a8.ts";
import {
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
} from "../../codegen/strided.ts";
import {
  attentionScoreUsesF16,
  type ScoreStorage,
  scoreStorageBytes,
} from "../../kernels/score-storage.ts";
import { attentionScale, stateWindow } from "../../ops.ts";
import { defaultGemmGeometry, gemmTileM, gemmTileN } from "../../kernels/gemm-geometry.ts";
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8TileM,
  i8a8TileN,
} from "../../kernels/i8a8-geometry.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../../codegen/dispatch.ts";
import { planRowBlocks } from "../fusion.ts";
import { planStateAttention } from "../state-attention-plan.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsWgsl,
} from "../../kernels/quantize-rows.ts";
import {
  stateAppendKey,
  stateAppendParams,
  stateAppendWgsl,
  stateAppendWorkgroups,
} from "../../kernels/state-append.ts";
import {
  stateAttentionParams,
  statePvKey,
  statePvParallelKey,
  statePvParallelReadonlyKey,
  statePvParallelReadonlyWgsl,
  statePvParallelReadonlyWorkgroups,
  statePvParallelWgsl,
  statePvParallelWorkgroups,
  statePvTiledEligible,
  statePvTiledKey,
  statePvTiledParams,
  statePvTiledWorkgroups,
  statePvWgsl,
  statePvWorkgroups,
  stateQkKey,
  stateQkParallelEligible,
  stateQkParallelKey,
  stateQkParallelReadonlyKey,
  stateQkParallelReadonlyWgsl,
  stateQkParallelReadonlyWorkgroups,
  stateQkParallelWgsl,
  stateQkParallelWorkgroups,
  stateQkTiledEligible,
  stateQkTiledKey,
  stateQkTiledParams,
  stateQkTiledWorkgroups,
  stateQkWgsl,
  stateQkWorkgroups,
  stateSliding,
  stateStatsKey,
  stateStatsParams,
  stateStatsReadonlyKey,
  stateStatsReadonlyWgsl,
  stateStatsReadonlyWorkgroups,
  stateStatsWgsl,
  stateStatsWorkgroups,
} from "../../kernels/state-attention.ts";
import { stateStatsPvKey, stateStatsPvWgsl } from "../../kernels/state-attention-stats-pv.ts";
import type { StorageRoles } from "../../gpu/pipeline-cache.ts";
import type { RecipeBuildFace, StateBuildContext } from "../recipe-builder.ts";
import { PARAMS_STORAGE_USAGE, PARAMS_UNIFORM_USAGE } from "./params-usage.ts";

/** 融合 attention の解決済み形（B·H を畳んだバッチ軸 / クエリ行 M / キー列 N / head 幅 D）。 */
type AttentionShape = {
  readonly batch: number;
  readonly rows: number;
  readonly cols: number;
  readonly depth: number;
};

/** 段（①QK / ③PV）の解決済みパイプライン。 */
type AttentionStagePipeline = {
  readonly key: string;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
  readonly roles: StorageRoles;
};

/**
 * ①QK の実行段。**行ブロックのループより外で 1 度だけ**解決する（i8a8 の量子化は列側が
 * ブロックに依らず、行側も総量が変わらないので、ループへ入れると仕事が枚数倍になるだけ）。
 * 一時（`qq` / `qs` / `kq` / `ks`）はループを跨いで生きるので、解放はノード末尾。
 */
type AttentionQkStage =
  | (AttentionStagePipeline & { readonly kind: "dense" })
  | (AttentionStagePipeline & {
    readonly kind: "i8a8";
    readonly geometry: I8a8Geometry;
    readonly qq: TempSource;
    readonly qs: TempSource;
    readonly kq: TempSource;
    readonly ks: TempSource;
  });

/** ③PV の実行段（①QK と同じ規律 — `vt` / `vq` / `vs` はループを跨いで生きる）。 */
type AttentionPvStage =
  | (AttentionStagePipeline & { readonly kind: "dense" })
  | (AttentionStagePipeline & {
    readonly kind: "i8a8";
    readonly geometry: I8a8Geometry;
    readonly vt: TempSource;
    readonly vq: TempSource;
    readonly vs: TempSource;
  });

/**
 * 融合 attention（ADR 0023）。**1 ノード = 行ブロック 1 枚あたり 3 dispatch**（`cat` と同じ
 * 「複数 dispatch で 1 ノード」の扱い。full-write は**ノードの出力 O について**成立する）:
 *
 * ① QK gemm（S を実体化・scale はタイル充填時に q/k 両方へ）→ ② 行統計（m と 1/Σexp）→
 * ③ PV gemm（A タイル充填時に `exp(S−m)·inv` を評価 = P 非実体化）。
 *
 * ## クエリ行のブロック実行（S の実体化幅の上限対策）
 *
 * S は `B·H · M · N · 格納幅` バイトで、**シンボリック次元 S に対して 2 乗**で伸びる
 * （1824×1248 の DiT で S = 8892 → s16 でも 2.53GB = D3D12 の 2GiB 固定上限超え）。そこで
 * クエリ行を {@link planRowBlocks}（ADR 0060 / states 形 attention と同じ純関数）で
 * 「1 枚が `maxStorageBufferBindingSize` に収まる最小枚数」へ等分し、①②③ をブロックごとに
 * 撃つ。実行時オートチューンは持たない（ADR 0022）— 枚数は device の granted limit と
 * 解決済み shape だけから決まる。
 *
 * ビット同一の根拠は分解経路の行ブロック（fusion.ts の `rowBlockAttention`）と同じ:
 * ①③ は**行の担当割り**だけを変え（{@link "../../kernels/gemm.ts"} `GemmRowWindow`）、②と
 * ③の `exp(S−m)·inv` は**行内で閉じる**ので 1 行あたりの演算列も丸めの並びも動かない。
 * MUST: **n = 1 では行窓を立てない** — キー・uniform・生成 WGSL・dispatch 列が分割前と
 * 完全に同一になる（既存のスナップショットとビット同一門がその検出器）。
 *
 * 省略可能な第 4 入力 `mask[1,1,M,N]`（加算型）は **① の束縛が 1 本増えるだけ**で、
 * dispatch 数も ②③ の経路も変わらない（S が mask 済みで出てくる）。
 *
 * GQA（`H % Hkv == 0` — ADR 0067）は **①③ のキー 1 語と uniform 1 語だけ**の軸で、dispatch 数も
 * 確保も変わらない（K / V の base だけが `wid.z / r` で kv-head へ写る）。i8a8 との組は
 * 未対応で fail loudly（決定 3）。
 *
 * i8a8 変種（opt-in）では **量子化の 4 dispatch がブロックループの外**へ出て、ループ内は
 * 3 dispatch のまま（① と ③ の GEMM が i8a8 カーネルに替わる）。**適格判定は
 * 段ごとに独立**（① は `D % 4 == 0`・③ は `N % 4 == 0`）なので、片方だけ i8a8 の**混成**が
 * 起こりうる — 満たさない段だけが f32 経路へ**沈黙で**縮退する（linear の `k % 4` と同じ
 * 流儀で、落ちたことは診断のパイプラインキーにだけ出る）。
 *
 * MUST: ① と ③ は 1 workgroup = 1 タイルなので、3 軸とも上限超過は fail loudly
 * （{@link tiledWorkgroups}）。縮退させるとタイルが欠落し、例外なしに O の一部が
 * 未書き込み（配り直しなら前の値）で残る。② だけが行方向 grid-stride。
 * MUST: 一時バッファ（S / 行統計）は**ブロックごとに**確保して返す（全ブロックぶんまとめて
 * 取ると、上限を越えない形にした意味が消える）。量子化の一時だけがループを跨ぎ、ノード末尾で
 * 確保の逆順に返る。これで計画の参照計数が閉じ、失敗経路でも `arena.destroy()` が領域ごと
 * 拾う（確保と破棄を 1 箇所へ — ADR 0004）。
 */
export const buildAttention = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [q, k, v] = step.inputShapes;
  // B と H は 1 本のバッチ軸へ畳む（契約は rank-4 head-first で、B は 3 者一致・H は q 側）。
  const batch = q[0] * q[1];
  const rows = q[2];
  const cols = k[2];
  const depth = q[3];
  const scale = attentionScale(step.node.attrs, `nodes (${step.node.op})`);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `attention [${q.join(",")}] × [${k.join(",")}] × [${v.join(",")}]`;

  // GQA（整除 broadcast — ADR 0067 決定 1 / 2）。`r = H / Hkv` は**導出値**（attrs 欄は無い）で、
  // 整除は shape 検査（src/ops/shapes.ts）が保証する。確保・dispatch・S / O / 行統計は
  // すべて q 側の `B·H` のままで、**K / V の読みだけ**が kv-head へ写る。
  // MUST: `r = 1` では GQA ビットを立てない（キーも生成 WGSL もバイト同一 — ADR 0067 決定 2）。
  const kvHeads = k[1];
  const kvRepeat = q[1] / kvHeads;
  const gqa = kvRepeat > 1;

  // f16 計算変種（ADR 0028）。3 カーネルが**同時に**切り替わる — S の格納形が ① の書き手と
  // ②③ の読み手で一致していなければならないので、段ごとの混成はあり得ない。
  const compute: GemmCompute = face.state.attentionCompute === "f16" ? "f16" : "f32";
  // i8a8 変種（設計 §7 の波 1 + 波 2）。**② 行統計は f32 のまま**（S の格納形も f32 の
  // ままなので、②③ が読む S は ① がどちらの経路で書いても同型）。
  // MUST: 適格判定は**段ごとに独立**。① は q / k のパック方向が D・③ は P̃ / Vᵀ の
  // パック方向が N なので、条件も別々になる（両方満たさない形・片方だけの形が実在する）。
  const i8a8 = face.state.attentionCompute === "a8";
  const qkI8a8 = i8a8 && depth % 4 === 0;
  const pvI8a8 = i8a8 && cols % 4 === 0;
  // S の**格納形**（案 γ 波 1 — 計算形と直交する第 2 の軸）。`pack2x16float` の 2 要素／語
  // なので、書き手 ①QK が v4 経路を取る形（`D % 4 == 0 && N % 4 == 0`）だけが適格で、
  // 非適格は f32 格納へ**沈黙で**縮退する（検出器はパイプラインキー）。
  // MUST: 3 カーネルが**同時に**切り替わる — S の格納形が書き手と読み手で一致していなければ
  // ならないので、段ごとの混成はあり得ない（i8a8 の適格判定が段ごとに独立なのとは別の話で、
  // ①が i8a8 でも f32 でも S の格納形は 1 つに決まる）。
  const scoreStorage: ScoreStorage =
    face.state.attentionScoreStorage === "f16" && attentionScoreUsesF16(depth, cols)
      ? "f16"
      : "f32";

  // 加算 mask（省略可能な第 4 入力 — 契約は src/ops.ts）。① の epilogue だけの軸で、
  // ②③ からは「S が既に mask 済み」に見える。
  const mask = binds[3];
  // MUST: i8a8 の ①QK は別カーネル（src/kernels/attention-i8a8.ts）で epilogue を持たない。
  // 黙って f32 経路へ落とすと「i8a8 を頼んだのに効かない」沈黙になり、mask を落とすと
  // 値が壊れるので、組み合わせそのものを**一時バッファを取る前に** fail loudly にする。
  // MUST: 判定は **要求されたモード**（`i8a8`）で見る — `qkI8a8`（D % 4 の適格判定込み）で
  // 見ると D % 4 != 0 のときだけ拒否をすり抜け、f32 の ①QK と i8a8 の ③PV の混成で走って
  // しまう。「mask × i8a8 は無条件に fail loudly」が契約（ADR 0023 / docs/limitations.md）。
  if (mask !== undefined && i8a8) {
    throw new ExecutionError(
      `${where}: 加算 mask 付きの attention は attentionCompute 'a8' と組めない` +
        "（①QK の i8a8 変種は mask の epilogue を持たない — attentionCompute を 'f32' か " +
        "'f16' にすること）",
    );
  }
  // DECIDED: GQA × i8a8 は fail loudly で開始する（docs/decisions/0067 決定 3）。i8a8 は
  // head 基底が 5 本（src/kernels/attention-i8a8.ts）で、K / V の量子化・確保も `B·H` 前提
  // なので、GQA 形をそのまま流すと量子化バッファの取り違えになる。
  // MUST: 判定は **要求されたモード**（`i8a8`）で見る — 段ごとの適格判定（`qkI8a8` /
  // `pvI8a8`）で見ると D%4 / N%4 を満たさない形だけ拒否をすり抜ける（mask と同じ規律）。
  // MUST: 黙って f32 経路へ落とさない（ADR 0058 決定 3 — 性能が静かに変わる）。
  if (gqa && i8a8) {
    throw new ExecutionError(
      `${where}: GQA（H=${q[1]} / Hkv=${kvHeads} → r=${kvRepeat}）× i8a8 は未対応` +
        "（ADR 0067 決定 3・後日サポート予定 — attentionCompute を 'f32' か 'f16' にすること）",
    );
  }

  // クエリ行のブロック分割（ADR 0060 と同じ純関数 — states 形 attention と同じ流儀）。
  // S 1 枚 = `B·H · block · N · 格納幅` バイトがストレージ束縛の上限に収まる最小枚数の等分で、
  // 実行時オートチューンは持たない（ADR 0022）。1 行でも収まらない形は fail loudly。
  // MUST: **n = 1 の機は分割前と完全に同一**（行窓を立てないのでキーも uniform も WGSL も
  // dispatch 列も 1 バイト動かない）。既存のスナップショットとビット同一門がその検出器。
  const scoreBytes = compute === "f16" ? 2 : scoreStorageBytes(scoreStorage);
  const blocks = planRowBlocks(
    rows,
    batch * cols * scoreBytes,
    face.state.gpu.limits.maxStorageBufferBindingSize,
    face.state.rowBlockSplit,
  );
  const windowed = blocks.length > 1;

  // 幾何はブロック行数に依らず既定（`defaultGemmGeometry` / `defaultI8a8Geometry`）なので、
  // ①③ のパイプラインは全ブロックで共有する（ブロック間の差は uniform の
  // `row_offset` / `m` だけ）。
  const geometry = defaultGemmGeometry();
  const hasMask = mask !== undefined;
  const qkV4 = gemmUsesVec4(depth, cols);
  const qkKey = attentionQkKey(qkV4, compute, scoreStorage, hasMask, gqa, windowed);
  // i8a8 の量子化（①の k / ③の Vᵀ = **列側**）はブロックに依存しないので**ループ外で 1 回**。
  // MUST: ループ内へ入れない（値は同じまま仕事が枚数倍になる純粋な性能退行）。
  // q 側（qq / qs）も行に比例する仕事で総量は変わらないため同じくループ外に置き、①QK が
  // 行窓で全 M ストライドから読む（ブロックごとに量子化すると `quantize_rows` に全 M
  // ストライドの行 gather が要り、linear と共有している 1 本を触ることになる）。
  // NOTE: 量子化の一時がループを跨いで生きるので、n = 1 でも同時生存が増える。増分は
  // `O(B·H·(M+N)·D)` で S の `O(B·H·M·N)` より 1 次低い（実測 = anima DiT の自己注意
  // S=8892 で 122.8MiB 対 S 1 枚 2,413.0MiB = +5.1%）。**上限に効くのはバッファ 1 本ごとの
  // サイズ**なので、束縛上限に対する峰は S のブロック 1 枚のまま動かない。
  const qk: AttentionQkStage = qkI8a8
    ? await prepareAttentionQkI8a8(
      face,
      builder,
      binds,
      scoreStorage,
      { batch, rows, cols, depth },
      where,
      windowed,
    )
    : {
      kind: "dense",
      key: qkKey,
      ...await face.state.cache.get(
        qkKey,
        attentionQkWgsl(qkV4, compute, scoreStorage, hasMask, gqa, windowed),
      ),
    };
  const pvV4 = gemmUsesVec4(cols, depth);
  const pvKey = attentionPvKey(pvV4, compute, scoreStorage, gqa, windowed);
  const pv: AttentionPvStage = pvI8a8
    ? await prepareAttentionPvI8a8(
      face,
      builder,
      binds,
      scoreStorage,
      { batch, rows, cols, depth },
      where,
      windowed,
    )
    : {
      kind: "dense",
      key: pvKey,
      ...await face.state.cache.get(
        pvKey,
        attentionPvWgsl(pvV4, compute, scoreStorage, gqa, windowed),
      ),
    };
  // ② 行統計 — 1 行 = 1 workgroup で、行方向は grid-stride（softmax と同じ形）。
  // S を 1 回だけ読む regcache 変種は dim 依存の生成なので、`epc` がキー軸に増える
  // （値はどちらもビット同一 — src/kernels/attention.ts）。ブロック化しても**行内で閉じる**
  // ので、S も行統計もブロック相対のまま素のカーネルがそのまま撃てる（行オフセットは要らない）。
  const statsRegCache = attentionStatsRegCache(cols);
  const statsKey = attentionStatsKey(compute, scoreStorage, statsRegCache);
  const stats = await face.state.cache.get(
    statsKey,
    attentionStatsWgsl(compute, scoreStorage, statsRegCache),
  );

  for (const block of blocks) {
    // S[batch, block, N] と行統計 [batch·block, 2]。O は実行相が確保済みなので、峰は
    // O + S + 統計（分解経路の S + P + 恒等 expand の 3 枚から 1 枚ぶん減る）。
    // f16 変種（`:c16` の array<f16> / s16 の pack2x16float）では S が半分のバイト数になる
    // （1024px の DiT で 1,073.7MB → 536.9MB）。
    // MUST: 一時は**ブロックごとに**確保して返す（全ブロックぶんまとめて取ると、上限を
    // 越えない形にした意味が消える — states 形 attention と同じ規律）。
    const scores = builder.allocTemp(batch * block.rows * cols * scoreBytes);
    const rowStats = builder.allocTemp(batch * block.rows * ATTENTION_STATS_STRIDE * 4);
    // 行窓 2 語（n = 1 では渡さない = uniform のバイト列も従来どおり）。
    const window = windowed ? { offset: block.offset, rowsFull: rows } : undefined;

    // ① QK gemm — 縮約次元は D、出力の列は N。行窓 `"a"` は q を全 M ストライドの
    // `row_offset` 行目から読むだけで、S はブロックとして 0 行目から書く。
    if (qk.kind === "i8a8") {
      builder.dispatch({
        key: qk.key,
        pipeline: qk.pipeline,
        layout: qk.layout,
        roles: qk.roles,
        params: face.writeParams(
          attentionQkI8a8Params(block.rows, cols, depth, scale, window),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: qk.qq },
          { binding: 2, source: qk.kq },
          { binding: 3, source: scores },
          { binding: ATTENTION_QK_Q_SCALE_BINDING, source: qk.qs },
          { binding: ATTENTION_QK_K_SCALE_BINDING, source: qk.ks },
        ],
        workgroups: [
          tiledWorkgroups(cols, i8a8TileN(qk.geometry), limit, `${where} ①QK i8a8`),
          tiledWorkgroups(block.rows, i8a8TileM(qk.geometry), limit, `${where} ①QK i8a8`),
          tiledWorkgroups(batch, 1, limit, `${where} ①QK i8a8`),
        ],
      });
    } else {
      builder.dispatch({
        key: qk.key,
        pipeline: qk.pipeline,
        layout: qk.layout,
        roles: qk.roles,
        params: face.writeParams(
          attentionQkParams(
            block.rows,
            cols,
            depth,
            scale,
            gqa ? kvRepeat : undefined,
            window,
          ),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: binds[0] },
          { binding: 2, source: binds[1] },
          { binding: 3, source: scores },
          ...(mask === undefined ? [] : [{ binding: ATTENTION_QK_MASK_BINDING, source: mask }]),
        ],
        workgroups: [
          tiledWorkgroups(cols, gemmTileN(geometry), limit, `${where} ①QK`),
          tiledWorkgroups(block.rows, gemmTileM(geometry), limit, `${where} ①QK`),
          tiledWorkgroups(batch, 1, limit, `${where} ①QK`),
        ],
      });
    }

    // ② 行統計（ブロック相対 — 行内で閉じるので行を切っても 1 行の演算列は変わらない）。
    builder.dispatch({
      key: statsKey,
      pipeline: stats.pipeline,
      layout: stats.layout,
      roles: stats.roles,
      params: face.writeParams(
        attentionStatsParams(batch * block.rows, cols, statsRegCache),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [{ binding: 1, source: scores }, { binding: 2, source: rowStats }],
      workgroups: [gridStrideWorkgroups(batch * block.rows, 1, limit), 1, 1],
    });

    // ③ PV gemm — 縮約次元は N、出力の列は D。行窓 `"c"` は O を全 M ストライドの
    // `row_offset` 行目へ書くだけで、S も行統計もブロック相対のまま読む。
    if (pv.kind === "i8a8") {
      builder.dispatch({
        key: pv.key,
        pipeline: pv.pipeline,
        layout: pv.layout,
        roles: pv.roles,
        params: face.writeParams(
          attentionPvI8a8Params(block.rows, depth, cols, window),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: scores },
          { binding: 2, source: pv.vq },
          { binding: 3, source: rowStats },
          { binding: 4, source: outs[0] },
          { binding: ATTENTION_PV_V_SCALE_BINDING, source: pv.vs },
        ],
        workgroups: [
          tiledWorkgroups(depth, i8a8TileN(pv.geometry), limit, `${where} ③PV i8a8`),
          tiledWorkgroups(block.rows, i8a8TileM(pv.geometry), limit, `${where} ③PV i8a8`),
          tiledWorkgroups(batch, 1, limit, `${where} ③PV i8a8`),
        ],
      });
    } else {
      builder.dispatch({
        key: pv.key,
        pipeline: pv.pipeline,
        layout: pv.layout,
        roles: pv.roles,
        params: face.writeParams(
          attentionPvParams(block.rows, depth, cols, gqa ? kvRepeat : undefined, window),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: scores },
          { binding: 2, source: binds[2] },
          { binding: 3, source: rowStats },
          { binding: 4, source: outs[0] },
        ],
        workgroups: [
          tiledWorkgroups(depth, gemmTileN(geometry), limit, `${where} ③PV`),
          tiledWorkgroups(block.rows, gemmTileM(geometry), limit, `${where} ③PV`),
          tiledWorkgroups(batch, 1, limit, `${where} ③PV`),
        ],
      });
    }

    // MUST: ノード境界で一時バッファを返す（計画の不変条件 — 抜けると閉包検査で落ちるか、
    // 配り直しから外れて peak が過大に出る）。確保の逆順。
    builder.releaseTemp(rowStats);
    builder.releaseTemp(scores);
  }

  // MUST: 量子化の一時も確保の逆順で返す（`#prepare*` が取ったぶんをここで閉じる — 確保と
  // 破棄はノードの中で対になっている。ADR 0004）。
  if (pv.kind === "i8a8") {
    builder.releaseTemp(pv.vs);
    builder.releaseTemp(pv.vq);
    builder.releaseTemp(pv.vt);
  }
  if (qk.kind === "i8a8") {
    builder.releaseTemp(qk.ks);
    builder.releaseTemp(qk.kq);
    builder.releaseTemp(qk.qs);
    builder.releaseTemp(qk.qq);
  }
};

/**
 * `states` 欄のキーが指すスロットの名前と束縛解決済み容量形。
 *
 * 形の妥当性（`[B,Hkv,C,D]`・ins との B / Hkv / D 一致・`window ≤ C`）は shape 層
 * （src/ops/shapes.ts）が済ませているので、ここで引けないのはランタイム内部の不変条件破れ
 * （導出相が `stateShapes` 無しで state ノードに当たった経路を含む）。
 */
const stateSlot = (
  step: NodePlan,
  key: string,
  states: StateBuildContext,
  where: string,
): { readonly name: string; readonly shape: readonly number[] } => {
  const name = step.node.states[key];
  if (name === undefined) throw new ExecutionError(`${where}: states 欄に '${key}' が無い`);
  const shape = states.shapes.get(name);
  if (shape === undefined) {
    throw new ExecutionError(
      `${where}: state スロット '${name}' の解決済み容量が無い` +
        "（GenerationContext を伴わない実行では state 参照ノードを組めない）",
    );
  }
  return { name, shape };
};

/**
 * states 形 attention（ADR 0067 決定 4 / 6 / 7）。**1 ノード = 行ブロック 1 枚あたり 3 dispatch**
 * で、カーネル族は融合 attention と完全に別（src/kernels/state-attention.ts）。
 *
 * ①QK（S を live 列だけ実体化）→ ②行統計（identity −inf・空行ガード）→ ③PV（P 非実体化）。
 * K / V の出どころは 2 つ（論理 col < `pastLength` は**スロット**・以降は今 step の `ins`）で、
 * 走査範囲は実行時値なので **①QK の dispatch 数だけが論理長から算出**される（②③ は出力側の
 * 形だけで決まり、live の走査は invocation の内側）。
 *
 * MUST: 列容量 `colCap` と行ブロックの割り方・一時のバイト式は {@link planStateAttention}
 * （見積り `estimate.ts` と共有する 1 本）だけから引く。S 1 枚 = `B·H · block · colCap · 4`
 * バイトが `maxStorageBufferBindingSize` に収まる最小枚数の等分で、実行時オートチューンは
 * 持たない（ADR 0022 / ADR 0060）。1 行でも収まらない形は fail loudly。
 * MUST: 一時（S / 行統計）は**ブロックごとに**確保して返す（全ブロックぶんまとめて取ると、
 * 上限を越えない形にした意味が消える）。
 */
export const buildStateAttention = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  states: StateBuildContext,
): Promise<void> => {
  const [q, insK] = step.inputShapes;
  const where = `attention (states) [${q.join(",")}] × [${insK.join(",")}]`;
  const [batch, heads, chunkRows, depth] = q;
  const kvRepeat = heads / insK[1];
  const gqa = kvRepeat > 1;
  const kSlot = stateSlot(step, "k", states, where);
  const vSlot = stateSlot(step, "v", states, where);
  // k / v スロットが同形であることは shape 層が済ませている（容量は片方から引けばよい）。
  const capacity = kSlot.shape[2];
  const window = stateWindow(step.node.attrs, where) ?? 0;
  const sliding = stateSliding(window);
  const scale = attentionScale(step.node.attrs, where);
  // DECIDED: 数値変種 × states 形は fail loudly で開始する（ADR 0058 決定 3 —「未実装の組は
  // 縮退でなく fail loudly」。黙って f32 で走ると opt-in を指定した意味が診断からも数値からも
  // 消える）。判定は**一時バッファを取る前**（GQA × i8a8 の門と同じ位置）。
  if (face.state.attentionCompute !== "f32") {
    throw new ExecutionError(
      `${where}: states 形の attention は attentionCompute '${face.state.attentionCompute}' と` +
        "組めない（f32 の別族カーネルのみ — ADR 0067 決定 3 / 4）",
    );
  }
  if (face.state.attentionScoreStorage !== "f32") {
    throw new ExecutionError(
      `${where}: states 形の attention は attentionScoreStorage ` +
        `'${face.state.attentionScoreStorage}' と組めない（S の格納は f32 のみ）`,
    );
  }
  // run 前検査（ADR 0066 決定 4 / ADR 0067 決定 4 ④）の材料。sliding は ring なので容量の
  // 検査対象にしない。
  states.chunkRows.add(chunkRows);
  if (!sliding) {
    states.fullCapacities.set(kSlot.name, capacity);
    states.fullCapacities.set(vSlot.name, capacity);
  }

  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const batchHeads = batch * heads;
  // 行ブロックの枚数を明示する `rowBlockSplit`（テスト専用 `ROW_BLOCK_SPLIT`）を渡すのは
  // 実行計画側だけ — 見積りには受け口が無く、既定の等分だけを名乗る。
  const { colCap, fusedStatsPv, blocks } = planStateAttention(
    {
      batchHeads,
      chunkRows,
      capacity,
      window,
      fuseStatsPv: face.state.stateAttentionReduce === "parallel-fused",
    },
    face.state.gpu.limits.maxStorageBufferBindingSize,
    face.state.rowBlockSplit,
  );

  // ①QK / ③PV の縮約形は**同じ** opt-in 席（ADR 0058・2026-09-06 裁定でノブは 1 つのまま）。
  // ①' / ③' はどちらも束縛・params が参照経路と同一で、キー・WGSL・workgroup 算出の 3 点
  // だけが替わる（キーの `:par` が census 門の目印）。
  const parallel = face.state.stateAttentionReduce !== "sequential";
  // ①' だけは席に**適用条件**が掛かる: この計画の `M`（= chunkRows）が 8 以下のときだけ選ぶ
  // （decode の M=1 と投機の verify M ≤ 8）。
  // WHY: ①' が縮めるのは 1 invocation の D 逐次の遅延で、それが律速なのは有効 invocation が
  // 「live 列 × 1 行」しか無い decode（M=1）だけ。prefill（M=768）では ① が既に行 × 列で
  // 埋まっており、①' は行タイル幅を 4 → 1 に落として 4 倍の workgroup と barrier を積むだけに
  // なる。実測（2026-09-06）でも decode は ×1.55〜1.72 で効いた一方、prefill の ①QK は GPU
  // 時間が 1.5〜1.9 倍に伸び、prefill 全体で +30〜60% 逆行した。verify（M=4）は中立（2026-09-09）。
  // 条件は**実測した範囲に留める**（ADR 0082 決定 4 と同じ規律）。③' の側は `M < 16` の計画だけが
  // 取る（`M ≥ 16` は席に依らず ③ₜ — 下記）ので、席 1 つで 2 段の適用範囲が違う形は続く。
  // MUST: 判定材料は計画時に決まる静的値だけ（`chunkRows` は宣言 shape 由来）— 実行時の論理長で
  // 分岐すると、同じ計画鍵が run ごとに違うパイプラインを指すことになる。
  const qkParallel = parallel && stateQkParallelEligible(chunkRows);
  // ①ₜ（GEMM 骨格の K 行タイル共有 — perf-ledger K-13）は **① とビット同一**なので席に依らず、
  // `M ≥ 16` の計画で既定として選ぶ（適用条件と WHY は `stateQkTiledEligible`）。2 つの適用条件は
  // 重ならない（1 < 16）ので、この優先順は読みやすさのためだけにある。
  // MUST: ①QK の 3 経路の優先順を持つのは**この 1 箇所**（キー・WGSL・params・workgroup の
  // 4 点がここで揃う）。分散させると「キーは ①ₜ・dispatch は ① の辺」のような組がありえて、
  // タイルが欠落したまま例外が出ない。
  const qkTiled = !qkParallel && stateQkTiledEligible(chunkRows);
  // ③ₜ（GEMM 骨格の V 行タイル共有 — perf-ledger K-13 段 2）も **③ とビット同一**なので
  // 席に依らず、`M ≥ 16` の計画で既定として選ぶ。①ₜ と違って**席より優先する**のがここの
  // 要点で、席が `"parallel"` の prefill 計画は ③' ではなく ③ₜ（= 参照経路の値）になる。
  // WHY: ③' の利得は decode に閉じており prefill 側は K-12 の実測で誤差内だった一方、
  // ③ₜ は traffic を削る。ビット同一の経路を既定に置ける方が数値契約が単純になる。
  // MUST: ③PV の 3 経路の優先順を持つのも**この 1 箇所**（①QK と同文 — キー・WGSL・
  // params・workgroup の 4 点がここで揃う）。
  const pvTiled = statePvTiledEligible(chunkRows);
  const pvParallel = parallel && !pvTiled;
  const qkKey = qkParallel
    ? stateQkParallelKey(sliding, gqa)
    : qkTiled
    ? stateQkTiledKey(sliding, gqa, chunkRows)
    : stateQkKey(sliding, gqa);
  const statsKey = stateStatsKey(sliding);
  const pvKey = fusedStatsPv
    ? stateStatsPvKey(sliding, gqa)
    : pvTiled
    ? statePvTiledKey(sliding, gqa, chunkRows)
    : pvParallel
    ? statePvParallelKey(sliding, gqa)
    : statePvKey(sliding, gqa);
  const qk = await face.state.cache.get(
    qkKey,
    qkParallel
      ? stateQkParallelWgsl(sliding, gqa)
      : qkTiled
      ? stateQkTiledWgsl(sliding, gqa, chunkRows)
      : stateQkWgsl(sliding, gqa),
  );
  const stats = fusedStatsPv
    ? undefined
    : await face.state.cache.get(statsKey, stateStatsWgsl(sliding));
  const pv = await face.state.cache.get(
    pvKey,
    fusedStatsPv
      ? stateStatsPvWgsl(sliding, gqa)
      : pvTiled
      ? statePvTiledWgsl(sliding, gqa, chunkRows)
      : pvParallel
      ? statePvParallelWgsl(sliding, gqa)
      : statePvWgsl(sliding, gqa),
  );

  for (const block of blocks) {
    // 静的幾何の**唯一の出どころ**（内容アドレスキャッシュ適格 — ブロック間の差は rowOffset /
    // rowsBlock だけ）。ここから段ごとに params を組む。
    const geometry = {
      rowsBlock: block.rows,
      rowOffset: block.offset,
      chunkRows,
      depth,
      kvRepeat,
      window,
      capacity,
      colCap,
      scale,
    };
    // ①③ が共有する語順の params。**タイル経路の段は 1 語も読まない**ので、2 段ともタイルの
    // 計画では 1 本も組まない（組んでも誰も束縛しない死んだ uniform になる）。
    let shared: GPUBuffer | undefined;
    const sharedParams = (): GPUBuffer =>
      shared ??= face.writeParams(stateAttentionParams(geometry), PARAMS_UNIFORM_USAGE);
    // ①ₜ / ③ₜ は骨格の `Dims`（先頭 3 語が m / n / k）を binding 0 に置くので、**同じ静的幾何を
    // 別の語順で**組み直す（③ₜ は `neg_inf` / `scale` を読まないので語数も違う）。①③ 共有の
    // 語順は動かさない — ③ と見積りが同じ表を読む。
    const qkParams = qkTiled
      ? face.writeParams(stateQkTiledParams(geometry), PARAMS_UNIFORM_USAGE)
      : sharedParams();
    const pvParams = pvTiled
      ? face.writeParams(statePvTiledParams(geometry), PARAMS_UNIFORM_USAGE)
      : sharedParams();
    const dispatchGeometry = {
      batchHeads,
      rowsBlock: block.rows,
      rowOffset: block.offset,
      depth,
      window,
    };
    const scores = builder.allocTemp(block.scoreBytes);
    const rowStats = fusedStatsPv ? undefined : builder.allocTemp(block.statsBytes);

    // ①QK — **workgroup 数だけが論理長から算出**される（仕事量 ∝ Q × (有効 past + Q) の機構
    // そのもの — ADR 0066 決定 3 の合格条件）。
    builder.dispatch({
      key: qkKey,
      pipeline: qk.pipeline,
      layout: qk.layout,
      roles: qk.roles,
      params: qkParams,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: { kind: "state", name: kSlot.name } },
        { binding: 4, source: scores },
        { binding: 5, source: { kind: "lengths" } },
      ],
      workgroups: (past, query) =>
        qkParallel
          ? stateQkParallelWorkgroups(dispatchGeometry, past, query, limit, `${where} ①QK`)
          : qkTiled
          ? stateQkTiledWorkgroups(
            dispatchGeometry,
            chunkRows,
            past,
            query,
            limit,
            `${where} ①QK`,
          )
          : stateQkWorkgroups(dispatchGeometry, past, query, limit, `${where} ①QK`),
    });

    if (stats !== undefined && rowStats !== undefined) {
      // ② 行統計 — 1 行 = 1 workgroup の行方向 grid-stride（live の走査は行ループの内側）。
      // 覆うのは有効行だけなので、workgroup 数も ① と同じく論理長から算出する。
      builder.dispatch({
        key: statsKey,
        pipeline: stats.pipeline,
        layout: stats.layout,
        roles: stats.roles,
        params: face.writeParams(
          stateStatsParams(batchHeads, block.rows, block.offset, colCap, window),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: scores },
          { binding: 2, source: rowStats },
          { binding: 3, source: { kind: "lengths" } },
        ],
        workgroups: (_past, query) =>
          stateStatsWorkgroups(dispatchGeometry, query, limit, `${where} ②stats`),
      });
    }
    // ③PV — 出力は `rowOffset` からの `rowsBlock` 行**全て**（pad 行も full-write）。
    builder.dispatch({
      key: pvKey,
      pipeline: pv.pipeline,
      layout: pv.layout,
      roles: pv.roles,
      params: pvParams,
      bindings: [
        { binding: 1, source: scores },
        ...(rowStats === undefined ? [] : [{ binding: 2, source: rowStats }]),
        { binding: 3, source: binds[2] },
        { binding: 4, source: { kind: "state", name: vSlot.name } },
        { binding: 5, source: outs[0] },
        { binding: 6, source: { kind: "lengths" } },
      ],
      workgroups: pvTiled
        ? statePvTiledWorkgroups(dispatchGeometry, chunkRows, limit, `${where} ③PV`)
        : pvParallel
        ? statePvParallelWorkgroups(dispatchGeometry, limit, `${where} ③PV`)
        : statePvWorkgroups(dispatchGeometry, limit, `${where} ③PV`),
    });

    // MUST: 確保の逆順で返す（計画の再生と同じ順）。
    if (rowStats !== undefined) builder.releaseTemp(rowStats);
    builder.releaseTemp(scores);
  }
};

/**
 * **readonly states 形 attention**（ADR 0096 段 2 §1.2 / §2.3 — drafter が貸し手の KV だけを
 * 読む形）。`buildStateAttention` との違いは 4 点だけで、骨格（①QK → ②行統計 → ③PV の
 * 3 dispatch × 行ブロック）は同じ:
 *
 * 1. **ins が q 1 本**（今 step の k/v が無い）— 束縛が 1 本ずつ詰まった別カーネル 3 本
 * 2. `M = 1` 固定（shape 層の MUST）なので行ブロックは常に 1 枚・縮約変種の適用条件を見ない
 *    （①' / ③' が席に依らず**常に**選ばれる — `M=1` は両者の適用条件そのもの）
 * 3. **live の式が違う**（`[P−min(P,W), P)` — 今 step のぶんが足されない）ので ①②③ とも
 *    readonly 専用キー（`:ro`）
 * 4. `states.chunkRows` / `fullCapacities` を**登録しない** — 借り手には `state_append` が
 *    1 本も無く（§1.1）、書かない run に「容量に収まるか」の検査は要らない。full の
 *    `P+Q ≤ C` を借り手でも見ると、貸し手が ring で回している sliding 以外のスロットで
 *    「貸し手では正規な P」が借り手側の run だけ拒否される
 *
 * MUST: `colCap` と行ブロックの割り方は {@link planStateAttention}（見積りと共有する 1 本）
 * だけから引く（states 形と同じ規律）。`M = 1` を渡すので sliding の列容量は `W` ちょうどに
 * なり、readonly の live 上限 `min(P, W)` と一致する。
 */
export const buildReadonlyStateAttention = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  states: StateBuildContext,
): Promise<void> => {
  const q = step.inputShapes[0];
  const where = `attention (readonly) [${q.join(",")}]`;
  // q は `[B,H,1,D]`（M = 1 は shape 層が保証済み — ADR 0096 段 2 §1.2）。
  const [batch, heads, , depth] = q;
  const kSlot = stateSlot(step, "k", states, where);
  const vSlot = stateSlot(step, "v", states, where);
  // k / v スロットが同形であることは shape 層が済ませている（容量は片方から引けばよい）。
  const capacity = kSlot.shape[2];
  const kvRepeat = heads / kSlot.shape[1];
  const gqa = kvRepeat > 1;
  const window = stateWindow(step.node.attrs, where) ?? 0;
  const sliding = stateSliding(window);
  const scale = attentionScale(step.node.attrs, where);
  // 数値変種 × states 形は fail loudly（`buildStateAttention` と同じ門 — ADR 0058 決定 3）。
  if (face.state.attentionCompute !== "f32") {
    throw new ExecutionError(
      `${where}: readonly の attention は attentionCompute ` +
        `'${face.state.attentionCompute}' と組めない（f32 の別族カーネルのみ）`,
    );
  }
  if (face.state.attentionScoreStorage !== "f32") {
    throw new ExecutionError(
      `${where}: readonly の attention は attentionScoreStorage ` +
        `'${face.state.attentionScoreStorage}' と組めない（S の格納は f32 のみ）`,
    );
  }

  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const batchHeads = batch * heads;
  const { colCap, blocks } = planStateAttention(
    { batchHeads, chunkRows: 1, capacity, window },
    face.state.gpu.limits.maxStorageBufferBindingSize,
    face.state.rowBlockSplit,
  );
  const qkKey = stateQkParallelReadonlyKey(sliding, gqa);
  const statsKey = stateStatsReadonlyKey(sliding);
  const pvKey = statePvParallelReadonlyKey(sliding, gqa);
  const qk = await face.state.cache.get(qkKey, stateQkParallelReadonlyWgsl(sliding, gqa));
  const stats = await face.state.cache.get(statsKey, stateStatsReadonlyWgsl(sliding));
  const pv = await face.state.cache.get(pvKey, statePvParallelReadonlyWgsl(sliding, gqa));

  for (const block of blocks) {
    const geometry = {
      rowsBlock: block.rows,
      rowOffset: block.offset,
      chunkRows: 1,
      depth,
      kvRepeat,
      window,
      capacity,
      colCap,
      scale,
    };
    // ①③ は同じ params 語順（readonly カーネルは states 形と同じ `Params` struct）。
    const params = face.writeParams(stateAttentionParams(geometry), PARAMS_UNIFORM_USAGE);
    const dispatchGeometry = {
      batchHeads,
      rowsBlock: block.rows,
      rowOffset: block.offset,
      depth,
      window,
    };
    const scores = builder.allocTemp(block.scoreBytes);
    const rowStats = builder.allocTemp(block.statsBytes);

    // ①'QK(ro) — 束縛は [params, q, slot_k, s, lengths]。**workgroup 数だけが論理長から
    // 算出**される（`P = 0` は live 0 で列軸が 0 = dispatch そのものが積まれない）。
    builder.dispatch({
      key: qkKey,
      pipeline: qk.pipeline,
      layout: qk.layout,
      roles: qk.roles,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: { kind: "state", name: kSlot.name } },
        { binding: 3, source: scores },
        { binding: 4, source: { kind: "lengths" } },
      ],
      workgroups: (past) =>
        stateQkParallelReadonlyWorkgroups(dispatchGeometry, past, limit, `${where} ①'QK(ro)`),
    });

    // ② 行統計(ro) — 束縛は [params, s, stats, lengths]。行数は `Q = 1` 固定なので静的。
    builder.dispatch({
      key: statsKey,
      pipeline: stats.pipeline,
      layout: stats.layout,
      roles: stats.roles,
      params: face.writeParams(
        stateStatsParams(batchHeads, block.rows, block.offset, colCap, window),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [
        { binding: 1, source: scores },
        { binding: 2, source: rowStats },
        { binding: 3, source: { kind: "lengths" } },
      ],
      workgroups: stateStatsReadonlyWorkgroups(dispatchGeometry, limit, `${where} ②stats(ro)`),
    });

    // ③'PV(ro) — 束縛は [params, s, stats, slot_v, out, lengths]。出力は行ブロック全体を
    // full-write（空行 = P 0 は厳密 0）。
    builder.dispatch({
      key: pvKey,
      pipeline: pv.pipeline,
      layout: pv.layout,
      roles: pv.roles,
      params,
      bindings: [
        { binding: 1, source: scores },
        { binding: 2, source: rowStats },
        { binding: 3, source: { kind: "state", name: vSlot.name } },
        { binding: 4, source: outs[0] },
        { binding: 5, source: { kind: "lengths" } },
      ],
      workgroups: statePvParallelReadonlyWorkgroups(
        dispatchGeometry,
        limit,
        `${where} ③'PV(ro)`,
      ),
    });

    // MUST: 確保の逆順で返す（計画の再生と同じ順）。
    builder.releaseTemp(rowStats);
    builder.releaseTemp(scores);
  }
};

/**
 * `state_append`（今 step の k / v をスロットへ書く単機能 effect op — ADR 0067 決定 5）。
 * **1 ノード = 1 dispatch・出力 0 本**（`StepRecipe.outputs` は空列）。
 *
 * MUST: workgroup 数は `queryLength` から算出する（容量 `C` に比例させない — ADR 0066 決定 3）。
 * 書くのは先頭 `queryLength` 行だけで、pad 行はカーネルの添字空間に入らない。
 */
export const buildStateAppend = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  builder: StepRecipeBuilder,
  states: StateBuildContext,
): Promise<void> => {
  const x = step.inputShapes[0];
  const where = `state_append [${x.join(",")}]`;
  const slot = stateSlot(step, "slot", states, where);
  const capacity = slot.shape[2];
  const window = stateWindow(step.node.attrs, where) ?? 0;
  const sliding = stateSliding(window);
  const geometry = {
    kvPlanes: x[0] * x[1],
    chunkRows: x[2],
    depth: x[3],
    capacity,
    window,
  };
  states.chunkRows.add(x[2]);
  if (!sliding) states.fullCapacities.set(slot.name, capacity);

  const key = stateAppendKey(sliding);
  const { pipeline, layout, roles } = await face.state.cache.get(key, stateAppendWgsl(sliding));
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params: face.writeParams(stateAppendParams(geometry), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: { kind: "state", name: slot.name } },
      { binding: 3, source: { kind: "lengths" } },
    ],
    workgroups: (_past, query) => stateAppendWorkgroups(geometry, query, limit, where),
  });
};

/**
 * 融合 attention ①QK の **i8a8 変種**（opt-in — {@link SessionOptions.attentionCompute}）の
 * **前段**。① が 1 dispatch から **3 dispatch** に増える（行ブロック n 枚ならノード全体で
 * 2 + 3n）:
 *
 * (a) `quantize_rows`（q を per-token i8 へ）→ (b) `quantize_rows`（k を per-token i8 へ）→
 * (c) i8a8 GEMM（整数内積 + dequant — **ブロックごと**に撃つので呼び手が持つ）。
 *
 * 量子化カーネルは linear の w8a8 と**同じ 1 本**（`QUANTIZE_ROWS_KEY` を共有する — 縮約軸
 * D が q / k とも最内連続なので、行 = token の per-token 量子化がそのまま要求どおりの形に
 * なる）。診断では linear の活性量子化と合算されるので、内訳は E2E のキー本数検査で担保する。
 *
 * MUST: (a)(b) とも**行ブロックのループ外**（= 全 M / 全 N を 1 度）。k 側はブロックに
 * 依存しないので枚数倍は純損、q 側は総量こそ変わらないが、ブロックごとに撃つには
 * `quantize_rows` が全 M ストライドの行 gather を要求する（linear と共有の 1 本を触る）。
 * 代わりに (c) が **A 側の行窓**で `qq` / `qs` を全 M ストライドから読む。
 * MUST: 一時バッファ（`qq` / `qs` / `kq` / `ks`）はここで宣言し、**ノード末尾**（呼び手）で
 * 確保の逆順に解放する。
 * MUST: `D` のオーバフロー門は fail loudly（黙って通すと i32 の巻き戻りで符号ごと化ける。
 * 実測形の D ≤ 384 に対して門は桁で余裕があるが、置かないと退行の受け皿が消える）。
 */
const prepareAttentionQkI8a8 = async (
  face: RecipeBuildFace,
  builder: StepRecipeBuilder,
  binds: readonly BindingSource[],
  scoreStorage: ScoreStorage,
  shape: AttentionShape,
  where: string,
  rowWindow: boolean,
): Promise<AttentionQkStage> => {
  const { batch, rows, cols, depth } = shape;
  if (depth > LINEAR_I8A8_MAX_K) {
    throw new ExecutionError(
      `${where}: D=${depth} が i8a8 経路の i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える` +
        "（attentionCompute を 'f32' にすること）",
    );
  }
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;

  // 量子化した q / k（i8 を 4 詰め）と per-token scale。ノード内で閉じた一時領域で、
  // q 側は**行** scale・k 側は**出力列**の scale になる（同じ per-token 量子化の別の読み方）。
  const qq = builder.allocTemp(Math.max(4, batch * rows * depth));
  const qs = builder.allocTemp(Math.max(4, batch * rows * 4));
  const kq = builder.allocTemp(Math.max(4, batch * cols * depth));
  const ks = builder.allocTemp(Math.max(4, batch * cols * 4));

  // (a)(b) 活性の per-token 量子化（行方向 grid-stride。D が短いので 1 workgroup に複数行を
  // 並べて畳む小 D 変種になる — `quantizeRowsGeometry`・perf-ledger P-1）
  const quantizeGeometry = quantizeRowsGeometry(depth);
  const quantizeKey = quantizeRowsKey(quantizeGeometry);
  const { pipeline: quantizePipeline, layout: quantizeLayout, roles: quantizeRoles } = await face
    .state.cache.get(
      quantizeKey,
      quantizeRowsWgsl(quantizeGeometry),
    );
  const quantize = (
    source: BindingSource,
    payload: TempSource,
    scales: TempSource,
    count: number,
  ): void => {
    builder.dispatch({
      key: quantizeKey,
      pipeline: quantizePipeline,
      layout: quantizeLayout,
      roles: quantizeRoles,
      params: face.writeParams(quantizeRowsParams(count, depth), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source },
        { binding: 2, source: payload },
        { binding: 3, source: scales },
      ],
      workgroups: [gridStrideWorkgroups(count, quantizeGeometry.rowsPerGroup, limit), 1, 1],
    });
  };
  quantize(binds[0], qq, qs, batch * rows);
  quantize(binds[1], kq, ks, batch * cols);

  // (c) 整数内積の GEMM（半スケールは dequant 側で q / k の両方へ — 設計 §2.1）。
  // 幾何は ③PV と**別に**選ぶ（③ だけ N = D の 1 タイル化が勝つ — 実測）。dispatch は
  // ブロックごとなので、ここでは解決だけして呼び手へ返す。
  const v4 = attentionQkI8a8UsesVec4(cols);
  const dp4a = face.state.attentionI8a8Dot === "dp4a";
  const geometry = defaultI8a8Geometry("attention_qk");
  const key = attentionQkI8a8Key(v4, dp4a, scoreStorage, geometry, rowWindow);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    attentionQkI8a8Wgsl(v4, dp4a, scoreStorage, geometry, rowWindow),
  );
  return { kind: "i8a8", key, pipeline, layout, roles, geometry, qq, qs, kq, ks };
};

/**
 * 融合 attention ③PV の **i8a8 変種**（opt-in — {@link SessionOptions.attentionCompute}）の
 * **前段**。③ が 1 dispatch から **3 dispatch** に増える（①も i8a8・行ブロック n 枚なら
 * ノード全体で 4 + 3n）:
 *
 * (a) `strided`（v`[B·H,N,D]` → Vᵀ`[B·H,D,N]` の permute）→ (b) `quantize_rows`
 * （Vᵀ を行 = `(b,h,d)` で量子化）→ (c) i8a8 GEMM（整数内積 + dequant — **ブロックごと**に
 * 撃つので呼び手が持つ）。
 *
 * **新カーネルを 1 本も作らずに per-column 量子化が得られる**のがこの並びの要点（設計 §2.3）:
 * Vᵀ の「行」は `(b,h,d)` なので `quantize_rows` の per-token 量子化がそのまま
 * **V の per-column（N 全体の amax）scale** になり、同時に dp4a が要求する **N 連続パック**も
 * 手に入る。MUST: V を転置せずに量子化してはならない — scale が縮約軸 n の上で変わり、
 * `f32(acc)·s` 形の前提（s が n に依存しない）が壊れる（例外の出ない誤値）。
 *
 * P̃ 側は量子化カーネルを通らない（A タイル充填が `round(127·exp(S−m))` を作る）ので、
 * dispatch も一時バッファも増えない — ②行統計は f32 のまま 1 バイトも変えない。
 *
 * MUST: (a)(b) は**行ブロックのループ外**（Vᵀ は列側 = ブロックに依存しないので、ループへ
 * 入れると permute と量子化が枚数倍になる純粋な性能退行）。ブロック相対なのは S と行統計
 * だけで、(c) は **C 側の行窓**で O の行だけを全 M ストライドへ書く。
 * MUST: 一時バッファ（`vt` / `vq` / `vs`）はここで宣言し、**ノード末尾**（呼び手）で
 * 確保の逆順に解放する。
 * MUST: `N` のオーバフロー門は fail loudly（|acc| ≤ N·127²。実測形の最大 N = 16,384 に対し
 * 門は桁で余裕があるが、置かないと退行の受け皿が消える）。
 */
const prepareAttentionPvI8a8 = async (
  face: RecipeBuildFace,
  builder: StepRecipeBuilder,
  binds: readonly BindingSource[],
  scoreStorage: ScoreStorage,
  shape: AttentionShape,
  where: string,
  rowWindow: boolean,
): Promise<AttentionPvStage> => {
  const { batch, cols, depth } = shape;
  if (cols > LINEAR_I8A8_MAX_K) {
    throw new ExecutionError(
      `${where}: N=${cols} が i8a8 経路の i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える` +
        "（attentionCompute を 'f32' にすること）",
    );
  }
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;

  // Vᵀ（f32・permute の実体化）と、その量子化結果。どれもノード内で閉じた一時領域。
  const vt = builder.allocTemp(Math.max(4, batch * depth * cols * 4));
  const vq = builder.allocTemp(Math.max(4, batch * depth * cols));
  const vs = builder.allocTemp(Math.max(4, batch * depth * 4));

  // (a) v[B·H,N,D] → Vᵀ[B·H,D,N]（既存の strided 読みコピー族 — permute そのもの）。
  // MUST: stride は**入力** `[B·H,N,D]` の連続 stride から組む（出力 shape から組むと
  // D == N のときだけ一致する。実測形は D != N なので露見するが、単体テストが本来の検出器）。
  const stridedSpec = { dtype: "f32" } as const;
  const permuteKey = stridedKey(stridedSpec);
  const { pipeline: permutePipeline, layout: permuteLayout, roles: permuteRoles } = await face.state
    .cache.get(
      permuteKey,
      stridedWgsl(stridedSpec),
    );
  builder.dispatch({
    key: permuteKey,
    pipeline: permutePipeline,
    layout: permuteLayout,
    roles: permuteRoles,
    params: face.writeParams(
      stridedParams(
        [batch, depth, cols],
        permuteSrcStrides([batch, cols, depth], [0, 2, 1]),
        0,
      ),
      PARAMS_STORAGE_USAGE,
    ),
    bindings: [{ binding: 1, source: binds[2] }, { binding: 2, source: vt }],
    workgroups: [
      gridStrideWorkgroups(batch * depth * cols, STRIDED_WORKGROUP_SIZE, limit),
      1,
      1,
    ],
  });

  // (b) Vᵀ の量子化（行 = (b,h,d)・行長 N — per-column scale と N 連続パックが同時に出る）
  const quantizeGeometry = quantizeRowsGeometry(cols);
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
    params: face.writeParams(quantizeRowsParams(batch * depth, cols), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: vt },
      { binding: 2, source: vq },
      { binding: 3, source: vs },
    ],
    workgroups: [gridStrideWorkgroups(batch * depth, quantizeGeometry.rowsPerGroup, limit), 1, 1],
  });

  // (c) 整数内積の GEMM（P̃ は A タイル充填で作る = 非実体化のまま）。dispatch はブロック
  // ごとなので、ここでは解決だけして呼び手へ返す。
  const v4 = attentionPvI8a8UsesVec4(depth);
  const dp4a = face.state.attentionI8a8Dot === "dp4a";
  const geometry = defaultI8a8Geometry("attention_pv");
  const key = attentionPvI8a8Key(v4, dp4a, scoreStorage, geometry, rowWindow);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    attentionPvI8a8Wgsl(v4, dp4a, scoreStorage, geometry, rowWindow),
  );
  return { kind: "i8a8", key, pipeline, layout, roles, geometry, vt, vq, vs };
};
