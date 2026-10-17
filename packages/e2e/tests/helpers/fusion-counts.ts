/**
 * **実配布グラフに対する融合ヒット数の門**（GPU 不要）。
 *
 * 融合はエクスポータのノード発行順が 1 つ変わるだけで黙って外れる。値は正しいまま
 * （掴めなければ素の列に落ちるので）性能だけが戻り、例外も警告も出ない — ADR 0040 §3 が
 * `lastRunFusions` を常設した理由がここで、**その数字に突合相手を与える**のが本ファイル。
 *
 * `planFusions` は純関数なので、実資産から IR を読んで計画するだけで判定できる（1 dispatch も
 * 出さない）。読むのは容器の **part 0**（ヘッダ + 2 文書 — container-v1 §3）だけで、重みの
 * block へは進まない（実体は合計 7GB 級）。
 *
 * Shared by the per-family gates `packages/e2e/tests/e2e_<family>_fusion_counts_test.ts`: each
 * family lane owns its own counts, and this module holds only the planning and reading side they
 * have in common. It has no side effects; asset presence is decided by each test file.
 */

import { isNotFound, stat } from "@karume/test-support/fs";
import type { IrGraph } from "../../../runtime/src/format/ir.ts";
import {
  type FusionCounts,
  type FusionWeightLayout,
  planFusions,
} from "../../../runtime/src/runtime/fusion.ts";
import { bindSymbols, countUses, planGraph } from "../../../runtime/src/runtime/plan.ts";
import { planWeightResidency } from "../../../runtime/src/runtime/weight-residency.ts";
import { resolveParts } from "../../../runtime/tests/helpers/container-files.ts";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "../../../runtime/tests/helpers/container-graph.ts";
import { seriesGraph } from "../../../models/tests/helpers/series-graphs.ts";

/** 系列出力の容器の読み口（part 0 の path と、その容器が名乗るグラフ名）。 */
export type SeriesSource = { readonly part0: URL; readonly graph: string };

/**
 * 系列出力の容器の **part 0**（グラフ記述と束縛表はここに載る — container-v1 §3）と
 * グラフ名。分割されていない容器では代表 path そのものが part 0 になるので、どちらの形でも
 * 同じ 1 行で足りる。
 *
 * MUST: グラフ名は綴らず `packages/models/tests/helpers/series-graphs.ts` の表から引く（門番 `e2e_assets_gate_test.ts`
 * と同じ 1 本）。読み手ごとに綴ると、片方だけ書き換えても誰も落ちない。
 */
export const seriesSource = (series: string): SeriesSource => ({
  part0: resolveParts(
    new URL(`../../../../outputs/series/${series}/model.krm`, import.meta.url),
  )[0],
  graph: seriesGraph(series),
});

/**
 * 資産の有無。
 *
 * MUST: NotFound 以外は伝播させる — 全 I/O エラーを「未生成」に丸めると、資産ルートの
 * マウント異常（EIO 等）が SKIP に化けて、実行されていない検証が静かに緑になる。
 */
export const exists = async (url: URL): Promise<boolean> => {
  try {
    await stat(url);
    return true;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/**
 * 配布形 1 つ（`root` 直下の manifest）の部品の容器の **part 0**（ヘッダ + 2 文書 —
 * ADR 0109 決定 3）からグラフを読む。重みの block は読まない（融合の計画に要るのはグラフ宣言
 * だけ）。越境参照の part 0 は `mirrors`（越境参照の repo → ローカルミラー — ADR 0038 §7）から
 * 読む。
 */
export const readDistributionGraph = (
  root: URL,
  manifest: ContainerManifest,
  model: string,
  component: string,
  dtype: string,
  mirrors: ReadonlyMap<string, URL>,
): Promise<IrGraph> => {
  const head = containerPart0(manifest, model, component, dtype);
  if (head.repo === undefined) {
    return readContainerGraph(new URL(head.path, root), component);
  }
  const mirror = mirrors.get(head.repo);
  if (mirror === undefined) {
    throw new Error(
      `${component}/${dtype} の越境先 '${head.repo}' に対応するローカルミラーが無い` +
        `（既知: ${[...mirrors.keys()].join(" / ")}）`,
    );
  }
  return readContainerGraph(new URL(head.path, mirror), component);
};

export const fusionCounts = (
  graph: IrGraph,
  inputShapes: Readonly<Record<string, readonly number[]>>,
  stateShapes?: ReadonlyMap<string, readonly number[]>,
  fuseRmsNormAdd = false,
  fuseLinearStaticQuantize = false,
  packedStaticQuantize = false,
): FusionCounts =>
  planFusions(planGraph(graph, bindSymbols(graph, inputShapes), stateShapes).nodes, {
    useCounts: countUses(graph),
    fuseRmsNormAdd,
    fuseLinearStaticQuantize,
    packedStaticQuantize,
    linearGemvReduce: "parallel",
    linearCompute: "f32",
    weightLayouts: new Map(
      [...planWeightResidency(graph)].flatMap(([name, weight]): [string, FusionWeightLayout][] => {
        // group 長は席が持つ（合流後の宣言からは常駐プランナが導く — 同じ値を 2 度導かない）。
        if (weight.seat === "i4") return [[name, { storage: "i4", groupSize: weight.groupSize }]];
        return weight.seat === "i2" || weight.seat === "i8" || weight.seat === "f16"
          ? [[name, { storage: weight.seat }]]
          : [];
      }),
    ),
    outputNames: new Set(graph.outputs),
    // WebGPU core 既定（128MiB）を判定に使う。行ブロック枚数はヒット数に効かないが、
    // **上限の値を機の実測から取らない**ことでこの門が機に依らない固定であり続ける。
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  }).counts;

/**
 * states 形 decode グラフの融合ヒット数（物理 chunk 行数 M を振って計画する）。
 *
 * スロットの容量記号は 640 で解く — 融合はノード列と**値**の shape で決まり、スロット容量の
 * 値には依存しない（e2e の検収値と同じ 640 を使うのは読み合わせやすさだけ）。入力は全て
 * `[1, M]`。
 */
export const decodeFusionCounts = (
  graph: IrGraph,
  rows: number,
  fuseRmsNormAdd = false,
  fuseLinearStaticQuantize = false,
  packedStaticQuantize = false,
): FusionCounts => {
  const inputShapes = Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : rows)),
    ]),
  );
  const stateShapes = new Map(
    Object.entries(graph.states).map(([name, slot]) => [
      name,
      slot.shape.map((dim) => (typeof dim === "number" ? dim : 640)),
    ]),
  );
  return fusionCounts(
    graph,
    inputShapes,
    stateShapes,
    fuseRmsNormAdd,
    fuseLinearStaticQuantize,
    packedStaticQuantize,
  );
};

export const NONE: FusionCounts = {
  silu: 0,
  upsample2x: 0,
  rope: 0,
  adaln: 0,
  rmsNormAdd: 0,
  linearStaticQuantize: 0,
  rowBlockAttention: 0,
  identityExpand: 0,
  packedStaticQuantize: 0,
};
