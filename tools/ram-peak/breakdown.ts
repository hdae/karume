/**
 * **取得の内訳を宣言だけで閉じて出す**（container-v1 §11 の「見積りは宣言だけで閉じる」）。
 *
 * 1 バイトも重みを取らずに、選択 (model, quant) の全容器について
 *
 * - part 0（descriptor のファイル）の合計
 * - part 1（const 領域）の合計
 * - 重み part の合計 / 資産 part の合計（part 2 以降を**その part に載る block の role**で分ける）
 * - block 長の role 別合計と最大 block 長
 * - 容器の外の資産（manifest の `assets`）の合計
 *
 * を出す。ホスト RAM のピークの上限は scan 型の区間読みで「最大 part 長 + 重ね合わせ」なので、
 * {@link FetchBreakdown.maxPartBytes} が実測 external の物差しになる。
 *
 * MUST: この面は**計測区間の外**で呼ぶ。`openContainer` は 2 文書を期待値と突合する（= digest を
 * 2 回掛ける）ので、計測中に呼ぶと検収③の計数へ harness 自身の digest が混ざる。
 */

import {
  type ContainerRef,
  type LoadedManifest,
  type LoadManifestOptions,
  openContainerSource,
  type ResolvedSelection,
} from "../../packages/hub/mod.ts";
import { openContainer } from "../../packages/runtime/mod.ts";

/** part の役割（part 2 以降は載る block の role から決まる）。 */
export type PartRole = "descriptor" | "const" | "weight" | "asset" | "mixed";

/** role 別の block バイト数（descriptor の `blocks[].length` の合計）。 */
export type BlockBytes = {
  readonly weight: number;
  readonly scale: number;
  readonly zeroPoint: number;
  readonly asset: number;
};

/** 容器 1 本の内訳。 */
export type ComponentBreakdown = {
  /** 部品名（manifest の `weights` のキー = グラフ名 — container-v1 §12）。 */
  readonly component: string;
  readonly partCount: number;
  /** part 0（ヘッダ + 2 文書）のバイト数。 */
  readonly descriptorPartBytes: number;
  /** part 1（const 領域）のバイト数（const を持たない容器では 0）。 */
  readonly constPartBytes: number;
  /** 重み part（part 2 以降で role が weight / scale / zero-point の block を持つ）の合計。 */
  readonly weightPartBytes: number;
  /** 資産 part（part 2 以降で role が asset の block しか持たない）の合計。 */
  readonly assetPartBytes: number;
  /** 重みと資産が同居している part の合計（分けられない形 — 0 が期待値）。 */
  readonly mixedPartBytes: number;
  /** この容器の最大 part 長（scan 型区間読みのホスト RAM の物差し）。 */
  readonly maxPartBytes: number;
  /** part の合計（= この容器を丸ごと取ったときのバイト数）。 */
  readonly totalPartBytes: number;
  /** グラフ記述が宣言する const 領域長（part 1 の論理長）。 */
  readonly constRegionBytes: number;
  readonly blockBytes: BlockBytes;
  /** 最大 block 長（seek 型区間読みのホスト RAM の物差し）。 */
  readonly maxBlockBytes: number;
  /** part ごとの役割（添字 = part の id）。 */
  readonly partRoles: readonly PartRole[];
  /** 2 文書の宣言バイト長（digest の仕分けに使う — `probes.ts` の `splitDigest`）。 */
  readonly descriptorLengths: readonly [number, number];
};

/** 選択 1 つぶんの内訳。 */
export type FetchBreakdown = {
  readonly model: string;
  readonly quant: string;
  readonly containerCount: number;
  readonly descriptorPartBytes: number;
  readonly constPartBytes: number;
  readonly weightPartBytes: number;
  readonly assetPartBytes: number;
  readonly mixedPartBytes: number;
  /** 容器の外の資産（manifest の `assets`）の合計。 */
  readonly manifestAssetBytes: number;
  /** 上の 6 つの合計（= この選択を丸ごと取ったときのバイト数）。 */
  readonly totalBytes: number;
  /** 全容器を通した最大 part 長。 */
  readonly maxPartBytes: number;
  readonly blockBytes: BlockBytes;
  readonly maxBlockBytes: number;
  readonly components: readonly ComponentBreakdown[];
  /** 全容器の 2 文書の宣言バイト長（重複を畳んだ集合）。 */
  readonly descriptorLengths: readonly number[];
};

const EMPTY_BLOCK_BYTES: BlockBytes = { weight: 0, scale: 0, zeroPoint: 0, asset: 0 };

const addBlockBytes = (left: BlockBytes, right: BlockBytes): BlockBytes => ({
  weight: left.weight + right.weight,
  scale: left.scale + right.scale,
  zeroPoint: left.zeroPoint + right.zeroPoint,
  asset: left.asset + right.asset,
});

/**
 * part 2 以降の役割を block の role から決める。
 *
 * PLE のような資産は**専用 part**へ分けて焼かれる（ADR 0109 決定 4）ので、役割が混じった part は
 * 出ない見込みである。出た場合は `mixed` として別に数える（黙って重み側へ寄せると「資産ぶんが
 * 重みに化けた内訳」になる）。
 */
const roleOfPart = (roles: ReadonlySet<string>): PartRole => {
  if (roles.size === 0) return "weight";
  if (roles.size === 1 && roles.has("asset")) return "asset";
  return roles.has("asset") ? "mixed" : "weight";
};

const breakdownOfContainer = async (
  component: string,
  container: ContainerRef,
  loaded: LoadedManifest,
  options: LoadManifestOptions,
): Promise<ComponentBreakdown> => {
  const source = openContainerSource(loaded, container, options);
  const opened = await openContainer({ kind: "source", source }, container.descriptor);
  const model = opened.model;
  if (model === undefined) {
    // manifest は `krm`（モデル記述つき）しか指さない（ADR 0109 決定 3）。
    throw new Error(
      `ram-peak: 部品 ${component} の容器にモデル記述が無い（krg は manifest に載らない）`,
    );
  }

  // part id → その part に載る block の role 集合。
  const rolesByPart = new Map<number, Set<string>>();
  let blockBytes: BlockBytes = EMPTY_BLOCK_BYTES;
  let maxBlockBytes = 0;
  for (const block of model.blocks) {
    const roles = rolesByPart.get(block.part) ?? new Set<string>();
    roles.add(block.role);
    rolesByPart.set(block.part, roles);
    maxBlockBytes = Math.max(maxBlockBytes, block.length);
    blockBytes = addBlockBytes(blockBytes, {
      weight: block.role === "weight" ? block.length : 0,
      scale: block.role === "scale" ? block.length : 0,
      zeroPoint: block.role === "zero-point" ? block.length : 0,
      asset: block.role === "asset" ? block.length : 0,
    });
  }
  for (const block of opened.graph.const.blocks) {
    maxBlockBytes = Math.max(maxBlockBytes, block.length);
  }

  const partRoles: PartRole[] = [];
  let descriptorPartBytes = 0;
  let constPartBytes = 0;
  let weightPartBytes = 0;
  let assetPartBytes = 0;
  let mixedPartBytes = 0;
  let maxPartBytes = 0;
  let totalPartBytes = 0;
  for (const [index, part] of container.parts.entries()) {
    maxPartBytes = Math.max(maxPartBytes, part.size);
    totalPartBytes += part.size;
    if (index === 0) {
      partRoles.push("descriptor");
      descriptorPartBytes += part.size;
      continue;
    }
    if (index === 1) {
      partRoles.push("const");
      constPartBytes += part.size;
      continue;
    }
    const role = roleOfPart(rolesByPart.get(index) ?? new Set<string>());
    partRoles.push(role);
    if (role === "asset") assetPartBytes += part.size;
    else if (role === "mixed") mixedPartBytes += part.size;
    else weightPartBytes += part.size;
  }

  return {
    component,
    partCount: container.parts.length,
    descriptorPartBytes,
    constPartBytes,
    weightPartBytes,
    assetPartBytes,
    mixedPartBytes,
    maxPartBytes,
    totalPartBytes,
    constRegionBytes: opened.graph.const.length,
    blockBytes,
    maxBlockBytes,
    partRoles,
    descriptorLengths: [container.descriptor.graph.length, container.descriptor.model.length],
  };
};

/**
 * 選択 1 つぶんの取得の内訳を出す。**容器ごとに part 0 の descriptor だけを読む**
 * （`openContainer` は開くだけでは重み block を 1 バイトも取らない）。
 */
export const fetchBreakdown = async (
  loaded: LoadedManifest,
  selection: ResolvedSelection,
  options: LoadManifestOptions = {},
): Promise<FetchBreakdown> => {
  const components: ComponentBreakdown[] = [];
  for (const [component, container] of Object.entries(selection.containers)) {
    components.push(await breakdownOfContainer(component, container, loaded, options));
  }
  const manifestAssetBytes = Object.values(selection.assets)
    .reduce((sum, ref) => sum + ref.size, 0);
  const sum = (pick: (entry: ComponentBreakdown) => number): number =>
    components.reduce((total, entry) => total + pick(entry), 0);
  const descriptorPartBytes = sum((entry) => entry.descriptorPartBytes);
  const constPartBytes = sum((entry) => entry.constPartBytes);
  const weightPartBytes = sum((entry) => entry.weightPartBytes);
  const assetPartBytes = sum((entry) => entry.assetPartBytes);
  const mixedPartBytes = sum((entry) => entry.mixedPartBytes);
  return {
    model: selection.model,
    quant: selection.quant,
    containerCount: components.length,
    descriptorPartBytes,
    constPartBytes,
    weightPartBytes,
    assetPartBytes,
    mixedPartBytes,
    manifestAssetBytes,
    totalBytes: descriptorPartBytes + constPartBytes + weightPartBytes + assetPartBytes +
      mixedPartBytes + manifestAssetBytes,
    maxPartBytes: components.reduce((max, entry) => Math.max(max, entry.maxPartBytes), 0),
    blockBytes: components.reduce(
      (total, entry) => addBlockBytes(total, entry.blockBytes),
      EMPTY_BLOCK_BYTES,
    ),
    maxBlockBytes: components.reduce((max, entry) => Math.max(max, entry.maxBlockBytes), 0),
    components,
    descriptorLengths: [
      ...new Set(components.flatMap((entry) => [...entry.descriptorLengths])),
    ],
  };
};
